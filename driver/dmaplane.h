/* SPDX-License-Identifier: GPL-2.0 */
/*
 * dmaplane — Kernel-internal header
 * Copyright (c) 2026 Graziano Labs Corp.
 *
 * Internal data structures for the dmaplane kernel module.
 * This file includes the UAPI header for shared types (ioctl numbers,
 * parameter structs, constants) and adds kernel-only types: ring
 * buffers, channels, buffers, device context, and file context.
 *
 * Architecture:
 *   main.c and dmabuf_rdma.c include this header.  Userspace test
 *   programs include only dmaplane_uapi.h (via -I../include).
 *   Kernel-only types are behind #ifdef __KERNEL__.
 */
#ifndef _DMAPLANE_H
#define _DMAPLANE_H

#include "../include/dmaplane_uapi.h"

#ifdef __KERNEL__

#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/dma-buf.h>
#include <linux/dma-mapping.h>
#include <linux/kref.h>
#include <linux/kthread.h>
#include <linux/mutex.h>
#include <linux/platform_device.h>
#include <linux/spinlock.h>
#include <linux/wait.h>
#include <linux/atomic.h>
#include <linux/cache.h>
#include <linux/rwsem.h>
#include <linux/scatterlist.h>
#include <linux/numa.h>
#include <linux/nodemask.h>
#include <rdma/ib_verbs.h>
#include "dmaplane_histogram.h"

/* ── Phase 8: GPU buffer constants ──────────────────────── */

#ifdef CONFIG_DMAPLANE_GPU
#define DMAPLANE_MAX_GPU_BUFFERS	16
struct dmaplane_gpu_buffer;	/* Forward declaration — defined in gpu_p2p.h */
#endif

/* ── Phase 2: Buffer constants ──────────────────────────── */

#define DMAPLANE_MAX_BUFFERS	64
#define DMAPLANE_MAX_MRS	64
#define DMAPLANE_MAX_BUF_SIZE	(1ULL << 30)	/* 1 GB hard limit */

/*
 * Ring buffer: fixed-size array with monotonically increasing head and tail.
 *
 * Full  when (head - tail) == DMAPLANE_RING_SIZE
 * Empty when head == tail
 *
 * head and tail only increment; modulo is applied when indexing:
 *   ring->entries[head % DMAPLANE_RING_SIZE]
 *
 * Unsigned arithmetic correctness:
 *   head and tail are unsigned int (32-bit).  They increment without
 *   bound and are never reset.  The difference (head - tail) is
 *   well-defined by C unsigned wraparound: even after 2^32 increments,
 *   (head - tail) still yields the correct occupancy as long as the
 *   ring size (1024) is a power of 2 and the occupancy never exceeds
 *   2^32.  The modulo index (head % RING_SIZE) maps naturally because
 *   RING_SIZE is a power of 2, making % equivalent to a bitmask.
 *
 * This avoids the classic "is the ring full or empty?" ambiguity of
 * wrapped-index designs, where head == tail could mean either state.
 *
 * head and tail are placed on separate cache lines to avoid false
 * sharing between the producer and consumer (see annotations below).
 */
struct dmaplane_ring {
	struct dmaplane_ring_entry entries[DMAPLANE_RING_SIZE];

	/*
	 * lock: Protects ring entries and head/tail indices.
	 * See per-ring comments in the channel struct for which
	 * producer/consumer pairs acquire this lock.
	 * Why spinlock: ring operations are very short (single entry
	 * copy + index increment); sleeping is not needed or desired.
	 */
	spinlock_t lock;

	/*
	 * ____cacheline_aligned_in_smp: kernel macro that pads to a
	 * separate cache line boundary on SMP builds (no-op on UP).
	 * Without this, producer (writing head) and consumer (writing
	 * tail) on different CPUs would bounce a single cache line
	 * between cores on every operation — false sharing.
	 */
	unsigned int head ____cacheline_aligned_in_smp;	/* Producer writes */
	unsigned int tail ____cacheline_aligned_in_smp;	/* Consumer reads  */
};

/*
 * Kernel-internal stats with atomic counters.
 *
 * Safe for concurrent updates (worker thread) and reads (ioctl).
 * The UAPI dmaplane_stats struct uses plain __u64 — IOCTL_GET_STATS
 * converts by reading each atomic into the UAPI struct before
 * copy_to_user.
 */
struct dmaplane_stats_kern {
	atomic64_t total_submissions;	/* Incremented in submit ioctl path */
	atomic64_t total_completions;	/* Incremented in worker thread */
	atomic_t   ring_high_watermark;	/* Max submission ring occupancy;
					 * atomic_t (32-bit) is sufficient */
	atomic_t   dropped_count;	/* Entries dropped (Phase 1: always 0;
					 * worker yields instead of dropping) */

	/* Phase 2: buffer lifecycle counters */
	atomic64_t buffers_created;	/* Lifetime buffers created */
	atomic64_t buffers_destroyed;	/* Lifetime buffers destroyed */

	/* Phase 3: dma-buf export counters — device-level so they
	 * survive individual export/release cycles */
	atomic64_t dmabufs_exported;	/* Lifetime dma-bufs created */
	atomic64_t dmabufs_released;	/* Lifetime dma-bufs released */
	atomic64_t dmabuf_attachments;	/* Lifetime attach calls */
	atomic64_t dmabuf_detachments;	/* Lifetime detach calls */
	atomic64_t dmabuf_maps;		/* Lifetime map_dma_buf calls */
	atomic64_t dmabuf_unmaps;	/* Lifetime unmap_dma_buf calls */

	/* Phase 4: RDMA counters */
	atomic64_t mrs_registered;	/* Lifetime MRs registered */
	atomic64_t mrs_deregistered;	/* Lifetime MRs deregistered */
	atomic64_t sends_posted;	/* Lifetime send WRs posted */
	atomic64_t recvs_posted;	/* Lifetime recv WRs posted */
	atomic64_t completions_polled;	/* Lifetime CQ completions polled */
	atomic64_t completion_errors;	/* Lifetime CQ completion errors */
	atomic64_t bytes_sent;		/* Lifetime bytes sent */
	atomic64_t bytes_received;	/* Lifetime bytes received */

	/* Phase 5: NUMA allocation tracking.
	 * Per-buffer counters — each CREATE_BUFFER increments exactly one.
	 * numa_local: all pages landed on the requested node.
	 * numa_remote: at least one page landed on a different node.
	 * numa_anon: allocated with DMAPLANE_NUMA_ANY. */
	atomic64_t numa_local_allocs;
	atomic64_t numa_remote_allocs;
	atomic64_t numa_anon_allocs;

	/* Phase 6: Flow control tracking.
	 * Cumulative across all sustained streaming runs.
	 * credit_stalls: sender couldn't send due to zero credits.
	 * high/low_watermark_events: hysteresis transitions.
	 * cq_overflows: should always be 0 if credits are configured.
	 * sustained_bytes/ops: total data moved by sustained benchmarks. */
	atomic64_t credit_stalls;
	atomic64_t high_watermark_events;
	atomic64_t low_watermark_events;
	atomic64_t cq_overflows;
	atomic64_t sustained_bytes;
	atomic64_t sustained_ops;
};

/*
 * Channel: one submission ring, one completion ring, one worker kthread.
 * Each open fd gets at most one channel (assigned via IOCTL_CREATE_CHANNEL).
 *
 * Lifetime managed by kref: the channel is freed (slot marked inactive)
 * only when the last reference drops.  Reference holders:
 *   - dmaplane_file_ctx: takes ref at channel creation.
 *   - Worker kthread: takes ref at channel creation.
 * Both refs are dropped by dmaplane_channel_destroy after kthread_stop
 * guarantees the worker has exited.  (The worker cannot drop its own
 * ref because the kthread wrapper may skip threadfn if KTHREAD_SHOULD_STOP
 * is set before the kthread gets a timeslice.)
 * kref_put calls dmaplane_channel_release.
 */
struct dmaplane_channel {
	/*
	 * refcount: channel lifetime — freed when last ref drops.
	 * Two refs taken at creation: one for file_ctx, one for
	 * the worker kthread.  Both dropped by dmaplane_channel_destroy
	 * after kthread_stop guarantees the worker has exited.
	 * kref_put calls dmaplane_channel_release.
	 */
	struct kref refcount;

	/*
	 * lock: per-channel mutex — protects channel state (shutdown flag,
	 * active flag, worker pointer).  Separate from ring spinlocks
	 * which protect ring data.  Separate from dev_mutex which
	 * protects slot allocation.
	 *
	 * Why mutex: only acquired from process context (ioctl, release);
	 * sleeping is permitted.
	 *
	 * Ordering: dev_mutex → channel->lock → ring spinlocks
	 * (if ever nested, though current code avoids holding multiple
	 * simultaneously).
	 */
	struct mutex lock;

	/*
	 * sub_ring: Submission ring.
	 * sub_ring.lock protects ring entries between the ioctl submit
	 * path (producer, advances head) and the worker thread (consumer,
	 * advances tail).
	 * Why spinlock: see dmaplane_ring.lock comment above.
	 * Ordering: never held simultaneously with comp_ring.lock.
	 */
	struct dmaplane_ring sub_ring;

	/*
	 * comp_ring: Completion ring.
	 * comp_ring.lock protects ring entries between the worker thread
	 * (producer, advances head) and the ioctl complete path (consumer,
	 * advances tail).
	 * Why spinlock: see dmaplane_ring.lock comment above.
	 * Ordering: never held simultaneously with sub_ring.lock.
	 */
	struct dmaplane_ring comp_ring;

	struct task_struct *worker;	/* kthread: "dmaplane/<id>" */
	wait_queue_head_t wait_queue;	/* Worker sleeps here; woken by
					 * ioctl submit and channel destroy */

	unsigned int id;		/* Channel index [0, MAX_CHANNELS) */

	/*
	 * in_flight: number of entries that have been submitted but not
	 * yet completed.  Atomic because it is incremented by the submit
	 * ioctl path (process context) and decremented by the worker
	 * kthread without holding a shared lock.
	 */
	atomic_t in_flight;

	/*
	 * shutdown: set to true by dmaplane_release under channel->lock
	 * before calling kthread_stop.  The worker checks
	 * kthread_should_stop() as its primary stop signal; this bool
	 * is a supplementary flag used to wake the worker out of
	 * wait_event_interruptible so it can see the stop flag.
	 */
	bool shutdown;

	/*
	 * active: true when this channel slot is in use.  Protected by
	 * dev_mutex — set to true in create_channel and to false in
	 * dmaplane_channel_release (kref callback).  Used by the
	 * linear scan in create_channel to find a free slot.
	 */
	bool active;

	/*
	 * stats: per-channel counters with atomic types for safe
	 * concurrent access.  total_submissions is incremented in the
	 * submit ioctl path; total_completions in the worker thread;
	 * ring_high_watermark under sub_ring.lock.  IOCTL_GET_STATS
	 * reads atomics into the UAPI struct.
	 */
	struct dmaplane_stats_kern stats;
};

/* ── Phase 2: Buffer tracking ───────────────────────────── */

/*
 * Per-buffer tracking.
 *
 * Two allocation paths, chosen by alloc_type:
 *
 *   BUF_TYPE_COHERENT — dma_alloc_coherent gives physically contiguous,
 *   cache-coherent memory with a single DMA address.  Used for small, hot
 *   control structures (CQ entries, doorbells) where CPU and device need
 *   coherent access without explicit sync barriers.  NUMA placement is not
 *   controllable — the DMA allocator uses the device's local node.
 *
 *   BUF_TYPE_PAGES — alloc_pages gives scattered physical pages that are
 *   vmapped into a contiguous kernel VA.  Used for large streaming buffers
 *   (gradients, weights, KV-cache).  NUMA-steerable in Phase 3 via
 *   alloc_pages_node().  These pages are later handed to ib_dma_map_sg()
 *   during MR registration (Phase 4) to build the scatterlist the NIC
 *   needs for DMA access.
 *
 * Both paths support mmap to userspace, completing the zero-copy chain:
 * userspace writes → same physical pages → NIC reads for TX.
 */
struct dmaplane_buffer {
	unsigned int id;	/* Handle returned to userspace.  Monotonically
				 * increasing, never reused within a session.
				 * 0 is never assigned (sentinel for "no buffer"). */
	int alloc_type;		/* BUF_TYPE_COHERENT or BUF_TYPE_PAGES */
	size_t size;		/* Requested size in bytes */
	bool in_use;		/* Slot is occupied.  Protected by buf_lock. */

	/* Kernel mapping — valid for both allocation paths.
	 * Coherent: vaddr from dma_alloc_coherent (physically contiguous).
	 * Pages: vaddr from vmap() (virtually contiguous over scattered pages). */
	void *vaddr;

	/* Coherent path only — DMA handle for the contiguous allocation.
	 * Used by dma_free_coherent() and dma_mmap_coherent().  Zero for pages. */
	dma_addr_t dma_handle;

	/* Page-backed path only — array of struct page pointers.
	 * NULL for coherent.  These pages are owned by the module and freed
	 * individually via __free_page() on destroy. */
	struct page **pages;
	unsigned int nr_pages;	/* Number of pages allocated */

	/* mmap refcount — prevents destroy while userspace holds a mapping.
	 * Incremented in dmaplane_mmap(), decremented in vma_close callback.
	 * Atomic because mmap and munmap can race from different threads. */
	atomic_t mmap_count;

	/* Phase 3: dma-buf export state.
	 * dmabuf_exported: true while a dma-buf wraps this buffer.  Prevents
	 *   double export (one dma-buf per buffer) and blocks DESTROY_BUFFER.
	 *   Set under buf_lock in the export path, cleared under buf_lock in
	 *   the release callback.
	 * dmabuf: back-pointer to the dma-buf object.  NULL when not exported.
	 *   Used for cleanup and diagnostics. */
	bool dmabuf_exported;
	struct dma_buf *dmabuf;

	/* Phase 5: NUMA placement tracking.
	 * numa_node: the node requested by userspace (-1 = DMAPLANE_NUMA_ANY).
	 * actual_numa_node: the node where the majority of pages actually
	 *   landed, determined post-hoc via page_to_nid(). May differ from
	 *   numa_node due to silent buddy allocator fallback. */
	int numa_node;
	int actual_numa_node;
};

/* ── Phase 4: RDMA types ───────────────────────────────── */

/*
 * Managed CQ completion wait structure.
 * Used with ib_alloc_cq(IB_POLL_DIRECT).  The managed CQ layer
 * requires every wr_cqe->done to point to a valid function.  With
 * IB_POLL_DIRECT we poll explicitly via ib_poll_cq(), so this
 * callback is rarely invoked — but it must exist to satisfy the
 * ib_cqe contract and as a safety net.
 */
struct poll_cq_wait {
	struct ib_cqe cqe;		/* Callback entry: .done = poll_cq_done */
	struct ib_wc wc;		/* Stashed work completion */
	struct completion done;		/* Signaled by poll_cq_done callback */
};

/*
 * Per-MR tracking.
 *
 * Two registration paths, selected by access_flags:
 *
 *   Local-only (IB_ACCESS_LOCAL_WRITE): mr is NULL, uses pd->local_dma_lkey
 *   (rkey=0, no remote access).  sge_addr is the kernel VA for rxe.
 *
 *   Fast-reg (remote access flags): mr is a real ib_mr allocated via
 *   ib_alloc_mr + IB_WR_REG_MR.  lkey/rkey/sge_addr come from the MR.
 *   Required for RDMA WRITE/READ operations.
 */
struct dmaplane_mr_entry {
	__u32 id;		/* MR unique ID (monotonically increasing, never 0) */
	__u32 buf_id;		/* Associated buffer ID */
	struct ib_mr *mr;	/* Fast-reg MR — NULL for local-only access
				 * (pd->local_dma_lkey suffices).  Non-NULL when
				 * access_flags include REMOTE_WRITE or REMOTE_READ. */
	__u32 lkey;		/* Local key (from mr->lkey or pd->local_dma_lkey) */
	__u32 rkey;		/* Remote key (from mr->rkey or 0 for local-only) */
	__u64 sge_addr;		/* Address for SGE — kernel VA for rxe (interprets
				 * via local_dma_lkey), IOMMU addr for real HW */
	struct sg_table *sgt;	/* DMA-mapped scatterlist for all pages */
	int sgt_nents;		/* Mapped nents from ib_dma_map_sg */
	bool in_use;		/* Slot is occupied.  Protected by mr_lock. */
	ktime_t reg_time;	/* Time to register (nanoseconds) */
};

/*
 * RDMA connection context.
 *
 * State machine:
 *   initialized = loopback pair is live (PD, CQ-A/B, QP-A/B created
 *                 and connected).  Set by rdma_engine_setup().
 *
 * The loopback pair (QP-A sends, QP-B receives) lets the module
 * benchmark the full DMA data path without a remote peer.
 */
struct dmaplane_rdma_ctx {
	struct ib_device *ib_dev;	/* IB device reference (from ib_device_get_by_name) */
	struct ib_pd *pd;		/* Protection Domain — root of resource hierarchy */

	/* Loopback pair: QP-A sends, QP-B receives */
	struct ib_cq *cq_a;		/* Completion queue for QP-A */
	struct ib_cq *cq_b;		/* Completion queue for QP-B */
	struct ib_qp *qp_a;		/* QP-A (sender in loopback) */
	struct ib_qp *qp_b;		/* QP-B (receiver in loopback) */

	__u8 port;			/* IB port number */
	__u16 lid;			/* Local Identifier from ib_query_port */
	int gid_index;			/* GID table index (for RoCEv2 selection) */
	union ib_gid gid;		/* Global Identifier (IPv6 format) */

	bool initialized;		/* Loopback pair is live */

	/* Phase 8: Cross-machine peer QP — shares PD with loopback pair.
	 * Created by IOCTL_RDMA_INIT_PEER, connected via TCP metadata
	 * exchange + IOCTL_RDMA_CONNECT_PEER.  Independent of the
	 * loopback pair — both can be active simultaneously. */
	struct ib_cq *cq_peer;		/* Peer QP's completion queue */
	struct ib_qp *qp_peer;		/* Third QP for remote machine */
	bool peer_connected;		/* Peer QP is connected (RTS state) */
};

/*
 * Device context: one per module instance (singleton).
 * Owns all channels, buffers, and the character device registration.
 */
struct dmaplane_dev {
	/* Platform device — provides the struct device * for DMA API calls.
	 * All dma_alloc_coherent, dma_map_sg, dma_mmap_coherent use
	 * &pdev->dev, not the char device's device pointer.  Created in
	 * module init before the char device registration. */
	struct platform_device *pdev;

	struct dmaplane_channel channels[DMAPLANE_MAX_CHANNELS];

	/*
	 * dev_mutex: Protects channel slot allocation and deallocation.
	 * Acquired by: dmaplane_ioctl_create_channel, dmaplane_release,
	 *              dmaplane_exit.
	 * Why mutex: sleeping context OK — only acquired from process
	 *            context (ioctl and file release).  Never acquired
	 *            from interrupt or softirq context.
	 * Ordering: outermost lock.  Always acquired before channel->lock
	 *           (see dmaplane_channel_destroy).  Never held while
	 *           acquiring ring spinlocks directly.  Independent of
	 *           buf_lock (never nested).
	 */
	struct mutex dev_mutex;

	struct cdev cdev;		/* Character device structure */
	struct class *class;		/* Device class for udev */
	dev_t devno;			/* Major/minor number pair */
	struct device *device;		/* /dev/dmaplane device node */

	/* ── Phase 2: Buffer management ── */

	/* Buffer array — global, not per-channel.  Fixed-size array with
	 * linear scan.  64 slots is deliberate simplicity — avoids IDR/hash
	 * complexity for a learning project. */
	struct dmaplane_buffer buffers[DMAPLANE_MAX_BUFFERS];

	/*
	 * buf_lock: Protects the buffers[] array — slot allocation, lookup,
	 * and deallocation.  Also held during mmap to prevent the buffer from
	 * being destroyed while vm_insert_page is in progress.
	 *
	 * Acquired by: IOCTL_CREATE_BUFFER, IOCTL_DESTROY_BUFFER,
	 *              IOCTL_GET_MMAP_INFO, dmaplane_mmap.
	 * Why mutex: sleeping context OK — all acquirers are in process context.
	 * Ordering: independent of dev_mutex (never nested).  Independent of
	 *           channel locks (different resource).  In Phase 4+, ordering
	 *           will be: rdma_sem (read) → buf_lock if both needed.
	 */
	struct mutex buf_lock;

	unsigned int next_buf_id;	/* Monotonically increasing.  Wraps to 1
					 * (skips 0) to avoid handle 0 ambiguity. */

	/*
	 * Device-level statistics — atomic for lock-free updates from
	 * open/close/create_channel/buffer paths.  Readable via pr_info at
	 * module unload; Phase 7 will export through debugfs.
	 */
	struct dmaplane_stats_kern stats;	/* Shared stats (channel + buffer) */

	atomic_t   active_channels;		/* Currently active channels */
	atomic64_t total_opens;			/* Lifetime fd opens */
	atomic64_t total_closes;		/* Lifetime fd closes */
	atomic64_t total_channels_created;	/* Lifetime channels created */
	atomic64_t total_channels_destroyed;	/* Lifetime channels destroyed */

	/* ── Phase 4: RDMA subsystem ── */

	/*
	 * RDMA context — loopback QP pair for self-contained benchmarking.
	 * Protected by rdma_sem (see below).
	 */
	struct dmaplane_rdma_ctx rdma;

	/*
	 * rdma_sem: read-write semaphore protecting the RDMA context.
	 *
	 * Read lock (down_read): MR registration/deregistration, all benchmarks.
	 *   Guarantees RDMA context stays alive.  Multiple readers run concurrently.
	 * Write lock (down_write): RDMA setup and teardown.  Exclusive access.
	 *
	 * Ordering: rdma_sem (read) → buf_lock → mr_lock.  Never hold rdma_sem
	 *   write while holding buf_lock or mr_lock.
	 */
	struct rw_semaphore rdma_sem;

	/* Memory Regions — global, not per-channel.  Fixed-size array with
	 * linear scan.  Same simplicity rationale as buffers[]. */
	struct dmaplane_mr_entry mrs[DMAPLANE_MAX_MRS];

	/*
	 * mr_lock: Protects the mrs[] array — slot allocation, lookup,
	 * and deallocation.
	 *
	 * Acquired by: IOCTL_REGISTER_MR, IOCTL_DEREGISTER_MR, benchmarks
	 *   (for MR snapshot by value).
	 * Why mutex: sleeping context OK — all acquirers are in process context.
	 * Ordering: always acquired after rdma_sem (read) and after buf_lock
	 *   if both needed.
	 */
	struct mutex mr_lock;

	__u32 next_mr_id;	/* Monotonically increasing.  Wraps to 1
				 * (skips 0) to avoid handle 0 ambiguity. */

	/* ── Phase 6: Flow control ── */

	/*
	 * Credit-based backpressure state.
	 *
	 * credits tracks available send slots (atomic for send/completion
	 * race in the general case, though IB_POLL_DIRECT makes polling
	 * synchronous).  paused implements watermark hysteresis — only
	 * accessed from the benchmark's single-threaded send loop, never
	 * concurrently.  max_credits/high_watermark/low_watermark are
	 * set by CONFIGURE_FLOW and read by the benchmark loop.
	 *
	 * No dedicated lock: paused is single-threaded (one benchmark
	 * send loop at a time), credits is atomic, and the configuration
	 * fields are only written by CONFIGURE_FLOW (expected to be called
	 * before running a benchmark).
	 */
	struct {
		atomic_t credits;	/* available send slots */
		unsigned int max_credits;	/* configured ceiling */
		unsigned int high_watermark;	/* pause threshold */
		unsigned int low_watermark;	/* resume threshold */
		bool configured;	/* true after CONFIGURE_FLOW */
		bool paused;		/* sender in backpressure state */
	} flow;

	/* ── Phase 7: Latency histogram ── */

	/*
	 * RDMA latency histogram — records per-operation latencies
	 * during benchmarks (pingpong, sustained streaming, QD sweep).
	 * All fields are atomic — no dedicated lock needed.  Multiple
	 * benchmark code paths can record simultaneously.
	 * Read via IOCTL_GET_HISTOGRAM or debugfs /dmaplane/histogram.
	 */
	struct dmaplane_histogram rdma_hist;

	/* ── Phase 8: GPU memory integration ── */

#ifdef CONFIG_DMAPLANE_GPU
	/*
	 * GPU P2P buffers — pinned GPU VRAM via nvidia_p2p_get_pages.
	 * Fixed-size array of 16 slots (same pattern as host buffers[64]).
	 * Allocated via kcalloc in module init, freed in module exit.
	 *
	 * gpu_buf_lock: protects gpu_buffers[] — slot allocation, lookup,
	 *   and deallocation.  Also held during pin/unpin operations.
	 *   Ordering: independent of rdma_sem and buf_lock.  Never nested
	 *   with buf_lock (DMA functions acquire them sequentially, not nested).
	 */
	struct dmaplane_gpu_buffer *gpu_buffers;
	struct mutex gpu_buf_lock;
	__u32 next_gpu_buf_id;		/* Monotonic, wraps to 1 (skips 0) */
#endif

	/*
	 * GPU statistics — always present, even without CONFIG_DMAPLANE_GPU.
	 * Reported via debugfs /dmaplane/gpu and IOCTL_GET_GPU_STATS.
	 * All zeros if GPU support is not compiled in or NVIDIA driver
	 * is not loaded.
	 */
	struct {
		atomic64_t pins_total;
		atomic64_t unpins_total;
		atomic64_t callbacks_fired;
		atomic64_t dma_h2g_bytes;
		atomic64_t dma_g2h_bytes;
		atomic64_t gpu_mrs_registered;
	} gpu_stats;
};

/*
 * File context: per-open-fd state.
 * Each open() allocates one; release() frees it.
 */
struct dmaplane_file_ctx {
	struct dmaplane_dev *dev;	/* Back-pointer to device singleton */
	struct dmaplane_channel *chan;	/* Assigned channel, or NULL if
					 * IOCTL_CREATE_CHANNEL has not been
					 * called on this fd yet */
};

/* kref release callback — marks slot inactive, cleans up channel */
void dmaplane_channel_release(struct kref *ref);

/*
 * Ring buffer helpers.
 *
 * These rely on unsigned arithmetic (see correctness argument in the
 * dmaplane_ring struct comment above).  All three must be called either
 * under ring->lock or in a context where the caller can tolerate a
 * stale value (e.g., the worker's outer empty-check before acquiring
 * the lock — the authoritative check is repeated under the lock).
 */
static inline bool dmaplane_ring_full(const struct dmaplane_ring *ring)
{
	return (ring->head - ring->tail) == DMAPLANE_RING_SIZE;
}

static inline bool dmaplane_ring_empty(const struct dmaplane_ring *ring)
{
	return ring->head == ring->tail;
}

static inline unsigned int dmaplane_ring_count(const struct dmaplane_ring *ring)
{
	return ring->head - ring->tail;
}

#endif /* __KERNEL__ */
#endif /* _DMAPLANE_H */
