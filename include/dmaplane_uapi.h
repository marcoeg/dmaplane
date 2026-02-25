/* SPDX-License-Identifier: GPL-2.0 */
/*
 * dmaplane — Userspace API header
 * Copyright (c) 2026 Graziano Labs Corp.
 *
 * Ioctl numbers, parameter structs, and constants shared between
 * the kernel module and userspace programs. Includable from both
 * kernel (__KERNEL__) and userspace.
 */
#ifndef _DMAPLANE_UAPI_H
#define _DMAPLANE_UAPI_H

#ifdef __KERNEL__
#include <linux/types.h>
#include <linux/ioctl.h>
#else
#include <stdint.h>
#include <sys/ioctl.h>

/* Map kernel fixed-width types to userspace equivalents.
 * Skipped when <linux/types.h> was already included (e.g., by libibverbs
 * headers) — that header defines __u64 as 'unsigned long long' via
 * asm-generic/int-ll64.h, which conflicts with our uint64_t typedef
 * ('unsigned long' on x86_64). */
#ifndef _LINUX_TYPES_H
typedef uint8_t  __u8;
typedef uint16_t __u16;
typedef uint32_t __u32;
typedef uint64_t __u64;
typedef int32_t  __s32;
#endif
#endif

/* Module name */
#define DMAPLANE_NAME		"dmaplane"

/* Ring buffer constants */
#define DMAPLANE_MAX_CHANNELS	4
/*
 * Must be a power of 2 so that (index % DMAPLANE_RING_SIZE) reduces to
 * a bitmask, and unsigned wrap-around arithmetic on head/tail yields
 * correct occupancy counts.
 */
#define DMAPLANE_RING_SIZE	1024

/* Ioctl magic number */
#define DMAPLANE_IOC_MAGIC	0xE4

/*
 * Ring entry: the fundamental unit passed through submission and
 * completion rings. payload carries the work item data; flags is
 * reserved for future per-entry metadata.
 */
struct dmaplane_ring_entry {
	__u64 payload;	/* in/out — work item data; worker adds 1 in Phase 1 */
	__u32 flags;	/* in/out — reserved for future per-entry metadata */
	__u32 _pad;	/* padding — explicit pad to 16-byte struct alignment */
};

/*
 * Submit parameters: userspace passes this to IOCTL_SUBMIT.
 * The entry is copied into the submission ring.
 */
struct dmaplane_submit_params {
	struct dmaplane_ring_entry entry;	/* in — entry to enqueue */
};

/*
 * Complete parameters: userspace passes this to IOCTL_COMPLETE.
 * On success, the entry is filled with the next completion.
 */
struct dmaplane_complete_params {
	struct dmaplane_ring_entry entry;	/* out — dequeued completion */
};

/*
 * Channel creation parameters: returned to userspace after
 * IOCTL_CREATE_CHANNEL. channel_id is the assigned channel index.
 */
struct dmaplane_channel_params {
	__u32 channel_id;	/* out — assigned channel index [0, MAX_CHANNELS) */
	__u32 _pad;			/* padding — explicit pad for alignment */
};

/*
 * Per-channel statistics returned by IOCTL_GET_STATS.
 * This is a racy snapshot — fields are updated without locks on the
 * hot path.  Values are monotonically increasing and individually
 * consistent, but the set may be momentarily inconsistent.
 */
struct dmaplane_stats {
	__u64 total_submissions;	/* Total entries submitted */
	__u64 total_completions;	/* Total entries completed by the worker */
	__u32 ring_high_watermark;	/* Max submission ring occupancy observed */
	__u32 dropped_count;		/* Entries dropped due to completion ring
								* full.  Phase 1 never drops — the worker
								* yields until space is available — so
								* this reads 0.  Retained as a placeholder
								* for future backpressure policies. */
};

/* Phase 1 ioctl commands: 0x01–0x04 */

/* Allocate a channel and start its worker kthread; returns channel_id. */
#define DMAPLANE_IOCTL_CREATE_CHANNEL \
	_IOR(DMAPLANE_IOC_MAGIC, 0x01, struct dmaplane_channel_params)

/* Enqueue one entry into the channel's submission ring. */
#define DMAPLANE_IOCTL_SUBMIT \
	_IOW(DMAPLANE_IOC_MAGIC, 0x02, struct dmaplane_submit_params)

/* Dequeue one entry from the channel's completion ring (non-blocking). */
#define DMAPLANE_IOCTL_COMPLETE \
	_IOWR(DMAPLANE_IOC_MAGIC, 0x03, struct dmaplane_complete_params)

/* Copy per-channel statistics to userspace (racy snapshot). */
#define DMAPLANE_IOCTL_GET_STATS \
	_IOR(DMAPLANE_IOC_MAGIC, 0x04, struct dmaplane_stats)

/* ── Phase 2: Buffer management ──────────────────────────── */

/*
 * Buffer allocation types.
 *
 * BUF_TYPE_COHERENT: dma_alloc_coherent — physically contiguous, cache-
 *   coherent memory with a single DMA address.  Used for small, hot control
 *   structures (CQ entries, doorbells) where CPU and device need coherent
 *   access without explicit sync barriers.
 *
 * BUF_TYPE_PAGES: alloc_pages + vmap — scattered physical pages vmapped
 *   into a contiguous kernel VA.  Used for large streaming buffers
 *   (gradients, weights, KV-cache).  NUMA-steerable in Phase 3 via
 *   alloc_pages_node().  These pages are later handed to ib_dma_map_sg()
 *   during MR registration (Phase 4).
 */
#define DMAPLANE_BUF_TYPE_COHERENT	0
#define DMAPLANE_BUF_TYPE_PAGES		1

/* NUMA placement: -1 = allocate on current CPU's local node (default) */
#define DMAPLANE_NUMA_ANY		(-1)

/*
 * Buffer creation parameters: passed to IOCTL_CREATE_BUFFER.
 * On success, buf_id and actual_numa_node are filled.
 *
 * numa_node controls NUMA placement:
 *   DMAPLANE_NUMA_ANY (-1): allocate on the current CPU's local node
 *     (same as plain alloc_page behavior).
 *   0..N: request allocation on the specified NUMA node.  This is
 *     best-effort — the kernel may fall back silently to other nodes
 *     if the target node is under memory pressure.  actual_numa_node
 *     reports where pages actually landed.
 */
struct dmaplane_buf_params {
	__u32 buf_id;		/* out — assigned buffer handle (never 0) */
	__u32 alloc_type;	/* in  — DMAPLANE_BUF_TYPE_* */
	__u64 size;		/* in  — buffer size in bytes */
	__s32 numa_node;	/* in  — NUMA node (-1 = any/local) */
	__s32 actual_numa_node;	/* out — node where pages were placed */
};

/*
 * mmap information: returned by IOCTL_GET_MMAP_INFO.
 * Userspace uses mmap_offset and mmap_size to call mmap(2):
 *   mmap(NULL, mmap_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, mmap_offset)
 */
struct dmaplane_mmap_info {
	__u32 buf_id;		/* in  — buffer handle */
	__u32 _pad;		/* padding — explicit pad for natural alignment */
	__u64 mmap_offset;	/* out — offset to pass to mmap(2) */
	__u64 mmap_size;	/* out — mappable size in bytes */
};

/*
 * Buffer statistics: returned by IOCTL_GET_BUF_STATS.
 * Racy snapshot — individually consistent, collectively approximate.
 */
struct dmaplane_buf_stats {
	__u64 buffers_created;		/* out — Lifetime buffers created */
	__u64 buffers_destroyed;	/* out — Lifetime buffers destroyed */
	__u64 numa_local_allocs;	/* out — alloc landed on requested node */
	__u64 numa_remote_allocs;	/* out — alloc fell back to another node */
	__u64 numa_anon_allocs;		/* out — alloc with DMAPLANE_NUMA_ANY */
};

/* Phase 2 ioctl commands: 0x05–0x09 */

/* Allocate a DMA buffer; returns buffer handle in buf_id. */
#define DMAPLANE_IOCTL_CREATE_BUFFER \
	_IOWR(DMAPLANE_IOC_MAGIC, 0x05, struct dmaplane_buf_params)

/* Destroy a buffer by handle (bare __u32, not a struct). */
#define DMAPLANE_IOCTL_DESTROY_BUFFER \
	_IOW(DMAPLANE_IOC_MAGIC, 0x06, __u32)

/* Get mmap offset and size for a buffer. */
#define DMAPLANE_IOCTL_GET_MMAP_INFO \
	_IOWR(DMAPLANE_IOC_MAGIC, 0x08, struct dmaplane_mmap_info)

/* Get buffer allocation statistics. */
#define DMAPLANE_IOCTL_GET_BUF_STATS \
	_IOR(DMAPLANE_IOC_MAGIC, 0x09, struct dmaplane_buf_stats)

/* ── Phase 3: dma-buf export ─────────────────────────────── */

/*
 * dma-buf export parameters: passed to IOCTL_EXPORT_DMABUF.
 * Wraps an existing page-backed buffer as a dma-buf and returns
 * a file descriptor.  Coherent buffers cannot be exported (no
 * page array for SG table construction).
 */
struct dmaplane_export_dmabuf_arg {
	__u32 buf_id;		/* in  — buffer handle to export */
	__s32 fd;		/* out — dma-buf file descriptor */
};

/*
 * dma-buf statistics: returned by IOCTL_GET_DMABUF_STATS.
 * Lifetime counters — survive individual export/release cycles.
 */
struct dmaplane_dmabuf_stats {
	__u64 dmabufs_exported;		/* out — lifetime dma-bufs created */
	__u64 dmabufs_released;		/* out — lifetime dma-bufs released */
	__u64 attachments_total;	/* out — lifetime attach calls */
	__u64 detachments_total;	/* out — lifetime detach calls */
	__u64 maps_total;		/* out — lifetime map_dma_buf calls */
	__u64 unmaps_total;		/* out — lifetime unmap_dma_buf calls */
};

/* Phase 3 ioctl commands: 0x0A–0x0B */

/* Export a page-backed buffer as a dma-buf; returns fd. */
#define DMAPLANE_IOCTL_EXPORT_DMABUF \
	_IOWR(DMAPLANE_IOC_MAGIC, 0x0A, struct dmaplane_export_dmabuf_arg)

/* Get dma-buf export statistics. */
#define DMAPLANE_IOCTL_GET_DMABUF_STATS \
	_IOR(DMAPLANE_IOC_MAGIC, 0x0B, struct dmaplane_dmabuf_stats)

/* ── Phase 4: RDMA integration ───────────────────────────── */

/*
 * RDMA setup parameters: passed to IOCTL_SETUP_RDMA.
 * Creates the full RDMA resource hierarchy: IB device → PD → CQs → QPs.
 * Zero values for cq_depth, max_send_wr, max_recv_wr use defaults.
 */
struct dmaplane_rdma_setup {
	char ib_dev_name[32];	/* in  — IB device name (e.g., "rxe_eth0") */
	__u32 port;		/* in  — IB port number (usually 1; 0 = default) */
	__u32 cq_depth;		/* in  — CQ depth (0 = default 128) */
	__u32 max_send_wr;	/* in  — max send WRs (0 = default 64) */
	__u32 max_recv_wr;	/* in  — max recv WRs (0 = default 64) */
	__u32 status;		/* out — 0 on success */
};

/*
 * MR registration parameters: passed to IOCTL_REGISTER_MR.
 * Registers buffer pages as an RDMA memory region.  For local-only
 * access (IB_ACCESS_LOCAL_WRITE), uses pd->local_dma_lkey (rkey=0).
 * For remote access (REMOTE_WRITE, REMOTE_READ), allocates a fast-reg
 * MR with a real rkey.
 */
struct dmaplane_mr_params {
	__u32 mr_id;		/* out — assigned MR ID */
	__u32 buf_id;		/* in  — buffer to register */
	__u32 access_flags;	/* in  — IB_ACCESS_* flags */
	__u32 lkey;		/* out — local key */
	__u32 rkey;		/* out — remote key (0 for local-only) */
	__u32 _pad;		/* padding — explicit pad for alignment */
	__u64 addr;		/* out — MR base address */
};

/*
 * Loopback test parameters: passed to IOCTL_LOOPBACK_TEST.
 * Single send from QP-A, recv on QP-B.  Validates the RDMA data path
 * works end-to-end; not a throughput measurement.
 */
struct dmaplane_loopback_params {
	__u32 mr_id;		/* in  — MR to use */
	__u32 size;		/* in  — message size in bytes */
	__u32 status;		/* out — 0 on success */
	__u32 pad;		/* padding — explicit pad for alignment */
	__u64 latency_ns;	/* out — round-trip latency in nanoseconds */
};

/*
 * Benchmark parameters: passed to IOCTL_PINGPONG_BENCH and
 * IOCTL_STREAMING_BENCH.  queue_depth is only used by streaming
 * (ignored by ping-pong).
 */
struct dmaplane_bench_params {
	__u32 mr_id;		/* in  — MR to use */
	__u32 msg_size;		/* in  — message size per operation */
	__u32 iterations;	/* in  — number of iterations */
	__u32 queue_depth;	/* in  — outstanding WRs (streaming only; 0 = 16) */
	__u64 total_ns;		/* out — total elapsed time */
	__u64 avg_latency_ns;	/* out — average latency per operation */
	__u64 p99_latency_ns;	/* out — 99th percentile latency */
	__u64 throughput_mbps;	/* out — throughput in MB/s */
	__u64 mr_reg_ns;	/* out — MR registration time */
};

/*
 * RDMA statistics: returned by IOCTL_GET_RDMA_STATS.
 * Lifetime counters — individually consistent, collectively approximate.
 */
struct dmaplane_rdma_stats {
	__u64 mrs_registered;		/* out — lifetime MRs registered */
	__u64 mrs_deregistered;		/* out — lifetime MRs deregistered */
	__u64 sends_posted;		/* out — lifetime send WRs posted */
	__u64 recvs_posted;		/* out — lifetime recv WRs posted */
	__u64 completions_polled;	/* out — lifetime CQ completions polled */
	__u64 completion_errors;	/* out — lifetime CQ completion errors */
	__u64 bytes_sent;		/* out — lifetime bytes sent */
	__u64 bytes_received;		/* out — lifetime bytes received */
};

/* Phase 4 ioctl commands: RDMA 0x10–0x11, MR 0x20–0x21, benchmarks 0x30–0x33 */

/* Initialize RDMA subsystem: IB device → PD → CQs → QPs → loopback. */
#define DMAPLANE_IOCTL_SETUP_RDMA \
	_IOWR(DMAPLANE_IOC_MAGIC, 0x10, struct dmaplane_rdma_setup)

/* Tear down RDMA subsystem; deregisters all MRs first. */
#define DMAPLANE_IOCTL_TEARDOWN_RDMA \
	_IO(DMAPLANE_IOC_MAGIC, 0x11)

/* Register a page-backed buffer as an RDMA MR; returns lkey/rkey. */
#define DMAPLANE_IOCTL_REGISTER_MR \
	_IOWR(DMAPLANE_IOC_MAGIC, 0x20, struct dmaplane_mr_params)

/* Deregister an MR by ID (bare __u32, not a struct). */
#define DMAPLANE_IOCTL_DEREGISTER_MR \
	_IOW(DMAPLANE_IOC_MAGIC, 0x21, __u32)

/* Run single-message loopback test (QP-A → QP-B). */
#define DMAPLANE_IOCTL_LOOPBACK_TEST \
	_IOWR(DMAPLANE_IOC_MAGIC, 0x30, struct dmaplane_loopback_params)

/* Run ping-pong latency benchmark (N iterations). */
#define DMAPLANE_IOCTL_PINGPONG_BENCH \
	_IOWR(DMAPLANE_IOC_MAGIC, 0x31, struct dmaplane_bench_params)

/* Run streaming throughput benchmark (pipelined sends). */
#define DMAPLANE_IOCTL_STREAMING_BENCH \
	_IOWR(DMAPLANE_IOC_MAGIC, 0x32, struct dmaplane_bench_params)

/* Get RDMA statistics (racy snapshot). */
#define DMAPLANE_IOCTL_GET_RDMA_STATS \
	_IOR(DMAPLANE_IOC_MAGIC, 0x33, struct dmaplane_rdma_stats)

/* ── Phase 5: NUMA topology & optimization ───────────────── */

/* Maximum NUMA nodes supported in topology/benchmark structs.
 * 8 covers all practical server topologies (dual=2, quad=4, 8-socket=8). */
#define DMAPLANE_MAX_NUMA_NODES	8

/*
 * NUMA topology: returned by IOCTL_QUERY_NUMA_TOPO.
 *
 * Reports online NUMA nodes, per-node CPU counts, memory sizes,
 * and the ACPI SLIT distance matrix.  Distance values:
 *   10 = same node (local)
 *   20–21 = one QPI/UPI hop (adjacent socket)
 *   30+ = further hops
 *
 * Node IDs may be sparse (e.g., nodes 0 and 2 online, 1 offline).
 * This struct compacts into contiguous indices [0..nr_nodes-1].
 */
struct dmaplane_numa_topo {
	__u32 nr_nodes;		/* out — number of online NUMA nodes */
	__u32 nr_cpus;		/* out — number of online CPUs */
	__u32 node_cpu_count[DMAPLANE_MAX_NUMA_NODES];	/* out — CPUs per node */
	__u32 node_online[DMAPLANE_MAX_NUMA_NODES];	/* out — 1 if online */
	__u64 node_mem_total_kb[DMAPLANE_MAX_NUMA_NODES]; /* out — total mem */
	__u64 node_mem_free_kb[DMAPLANE_MAX_NUMA_NODES];  /* out — free mem */
	/* out — Distance matrix: distances[i][j] = ACPI SLIT distance i to j */
	__u32 distances[DMAPLANE_MAX_NUMA_NODES][DMAPLANE_MAX_NUMA_NODES];
};

/*
 * NUMA bandwidth benchmark: passed to IOCTL_NUMA_BENCH.
 *
 * For each (cpu_node, buf_node) pair, spawns a kthread pinned to
 * cpu_node, measures memcpy throughput to a buffer on buf_node.
 * Result is an NxN bandwidth matrix quantifying cross-node penalty.
 */
struct dmaplane_numa_bench_params {
	__u64 buffer_size;	/* in  — test buffer size per node (bytes) */
	__u32 iterations;	/* in  — memcpy iterations per cell */
	__u32 pad;		/* padding — explicit pad for alignment */
	/* Output: NxN bandwidth matrix in MB/s.
	 * bw_mbps[cpu_node][buf_node] = throughput when a thread on
	 * cpu_node writes to a buffer allocated on buf_node. */
	__u64 bw_mbps[DMAPLANE_MAX_NUMA_NODES][DMAPLANE_MAX_NUMA_NODES];
	/* Output: NxN latency matrix in nanoseconds per iteration. */
	__u64 lat_ns[DMAPLANE_MAX_NUMA_NODES][DMAPLANE_MAX_NUMA_NODES];
	__u32 nr_nodes;		/* out — nodes measured */
	__u32 status;		/* out — 0 on success */
};

/* Phase 5 ioctl commands: NUMA 0x50–0x51 */

/* Query NUMA topology: nodes, CPUs, memory, distance matrix. */
#define DMAPLANE_IOCTL_QUERY_NUMA_TOPO \
	_IOR(DMAPLANE_IOC_MAGIC, 0x50, struct dmaplane_numa_topo)

/* Run NxN cross-node bandwidth benchmark. */
#define DMAPLANE_IOCTL_NUMA_BENCH \
	_IOWR(DMAPLANE_IOC_MAGIC, 0x51, struct dmaplane_numa_bench_params)

/* ── Phase 6: Backpressure & throughput modeling ─────────── */

/*
 * Flow control configuration: passed to IOCTL_CONFIGURE_FLOW.
 *
 * Establishes credit-based backpressure for the RDMA send path.
 * max_credits limits outstanding in-flight operations — setting it
 * at or below CQ depth makes CQ overflow impossible by construction.
 *
 * Watermark hysteresis prevents oscillation:
 *   - Sender pauses when in-flight count reaches high_watermark.
 *   - Sender resumes only when in-flight drops below low_watermark.
 * Without hysteresis, the sender would toggle between send/stall on
 * every single credit change, destroying throughput.
 */
struct dmaplane_flow_params {
	__u32 max_credits;	/* in  — total credits (max in-flight ops) */
	__u32 high_watermark;	/* in  — in-flight count to pause sending */
	__u32 low_watermark;	/* in  — in-flight count to resume sending */
	__u32 status;		/* out — 0 on success */
};

/*
 * Sustained streaming parameters: passed to IOCTL_SUSTAINED_STREAM.
 *
 * Runs a pipelined send/recv benchmark for a configurable wall-clock
 * duration (not iteration count).  Per-second windowing tracks throughput
 * stability — min/max windows reveal variance that averages hide.
 *
 * If CONFIGURE_FLOW was not called, auto-configures with conservative
 * defaults: max_credits = min(2*queue_depth, 128), high = 3/4, low = 1/4.
 */
struct dmaplane_sustained_params {
	__u32 mr_id;			/* in  — memory region to stream */
	__u32 msg_size;			/* in  — bytes per send */
	__u32 queue_depth;		/* in  — pipeline depth */
	__u32 duration_secs;		/* in  — how long to stream (1–600) */
	__u64 total_bytes;		/* out — total bytes transferred */
	__u64 total_ops;		/* out — total send/recv completions */
	__u64 avg_throughput_mbps;	/* out — bytes * 1000 / elapsed_ns */
	__u64 min_window_mbps;		/* out — worst 1-second window */
	__u64 max_window_mbps;		/* out — best 1-second window */
	__u64 stall_count;		/* out — times sender blocked on credits */
	__u64 cq_overflow_count;	/* out — should be 0 */
	__u32 status;			/* out — 0 on success */
	__u32 pad;			/* padding — explicit pad for alignment */
};

/*
 * Queue depth sweep parameters: passed to IOCTL_QDEPTH_SWEEP.
 *
 * Iterates across queue depths from min_qdepth to max_qdepth,
 * running a fixed-iteration streaming benchmark at each point.
 * Produces throughput/latency curves and detects the saturation
 * point — the smallest queue depth where throughput reaches 95%
 * of the maximum observed.
 */
#define DMAPLANE_MAX_SWEEP_POINTS	32

struct dmaplane_sweep_params {
	__u32 mr_id;		/* in  — MR to use */
	__u32 msg_size;		/* in  — message size per operation */
	__u32 iterations;	/* in  — iterations per queue depth point */
	__u32 min_qdepth;	/* in  — starting queue depth */
	__u32 max_qdepth;	/* in  — ending queue depth */
	__u32 step;		/* in  — queue depth increment */
	__u64 throughput_mbps[DMAPLANE_MAX_SWEEP_POINTS];	/* out — MB/s per point */
	__u64 avg_latency_ns[DMAPLANE_MAX_SWEEP_POINTS];	/* out — avg latency per point */
	__u64 p99_latency_ns[DMAPLANE_MAX_SWEEP_POINTS];	/* out — P99 latency per point */
	__u32 nr_points;		/* out — actual points measured */
	__u32 saturation_qdepth;	/* out — QD where throughput plateaus */
	__u32 status;			/* out — 0 on success */
	__u32 pad;			/* padding — explicit pad for alignment */
};

/*
 * Flow control statistics: returned by IOCTL_GET_FLOW_STATS.
 * Lifetime counters — cumulative across all sustained streaming runs.
 * Separately tracked from RDMA stats to avoid ABI changes.
 */
struct dmaplane_flow_stats {
	__u64 credit_stalls;		/* out — sender blocked on zero credits */
	__u64 high_watermark_events;	/* out — times high watermark reached */
	__u64 low_watermark_events;	/* out — times low watermark crossed */
	__u64 cq_overflows;		/* out — CQ full events (should be 0) */
	__u64 total_sustained_bytes;	/* out — cumulative sustained bytes */
	__u64 total_sustained_ops;	/* out — cumulative sustained ops */
};

/* Phase 6 ioctl commands: flow control 0x40–0x43 */

/* Configure credit-based flow control parameters. */
#define DMAPLANE_IOCTL_CONFIGURE_FLOW \
	_IOWR(DMAPLANE_IOC_MAGIC, 0x40, struct dmaplane_flow_params)

/* Run sustained streaming benchmark for a wall-clock duration. */
#define DMAPLANE_IOCTL_SUSTAINED_STREAM \
	_IOWR(DMAPLANE_IOC_MAGIC, 0x41, struct dmaplane_sustained_params)

/* Run queue depth sweep: throughput/latency across queue depths. */
#define DMAPLANE_IOCTL_QDEPTH_SWEEP \
	_IOWR(DMAPLANE_IOC_MAGIC, 0x42, struct dmaplane_sweep_params)

/* Get flow control statistics (racy snapshot). */
#define DMAPLANE_IOCTL_GET_FLOW_STATS \
	_IOR(DMAPLANE_IOC_MAGIC, 0x43, struct dmaplane_flow_stats)

/* ── Phase 7: Instrumentation ─────────────────────────────── */

/*
 * Latency histogram: log₂ buckets in microseconds.
 * 16 buckets cover [0, 1), [1, 2), [2, 4), ... [32768, ∞) µs.
 * For rxe (~200 µs per op), most samples land in buckets 7-8.
 * For ConnectX hardware (~1-5 µs), buckets 0-2.
 *
 * Coarse but constant-space: 128 bytes for bucket counters plus
 * metadata.  Trade precision for zero-allocation O(1) recording.
 * Exact percentiles require sorting N samples (what QD sweep does);
 * the histogram gives approximate percentiles with O(1) memory.
 */
#define DMAPLANE_HIST_BUCKETS	16

/*
 * Histogram output: returned by IOCTL_GET_HISTOGRAM.
 * Bucket counts + computed percentiles + summary statistics.
 * Set reset=1 to atomically read-and-clear the histogram.
 */
struct dmaplane_hist_params {
	__u64 buckets[DMAPLANE_HIST_BUCKETS];	/* out — sample count per bucket */
	__u64 count;		/* out — total samples recorded */
	__u64 p50_ns;		/* out — 50th percentile (bucket upper bound) */
	__u64 p99_ns;		/* out — 99th percentile (bucket upper bound) */
	__u64 p999_ns;		/* out — 99.9th percentile (bucket upper bound) */
	__u64 avg_ns;		/* out — arithmetic mean of all samples */
	__u64 min_ns;		/* out — minimum latency observed */
	__u64 max_ns;		/* out — maximum latency observed */
	__u32 reset;		/* in  — 1 = reset histogram after reading */
	__u32 pad;		/* padding — explicit pad for alignment */
};

/* Phase 7 ioctl commands: instrumentation 0x70 */

/* Read latency histogram; optionally reset after reading. */
#define DMAPLANE_IOCTL_GET_HISTOGRAM \
	_IOWR(DMAPLANE_IOC_MAGIC, 0x70, struct dmaplane_hist_params)

/* ── Phase 8: GPU memory integration ─────────────────────── */

/*
 * GPU buffer allocation type constant.
 * Extends the BUF_TYPE_COHERENT (0) and BUF_TYPE_PAGES (1) namespace.
 */
#define DMAPLANE_BUF_TYPE_GPU		2

/*
 * GPU pin parameters: passed to IOCTL_GPU_PIN.
 * Pins a region of GPU VRAM via the NVIDIA P2P API and maps the
 * resulting BAR1 pages with ioremap_wc for CPU access.
 *
 * Both gpu_va and size must be 64KB-aligned (NVIDIA P2P hardware
 * requirement — BAR1 pages are always 64KB regardless of host PAGE_SIZE).
 */
struct dmaplane_gpu_pin_params {
	__u64 gpu_va;		/* in  — CUDA device pointer (64KB-aligned) */
	__u64 size;		/* in  — bytes to pin (64KB-multiple) */
	__u32 handle;		/* out — GPU buffer handle for subsequent ops */
	__s32 gpu_numa_node;	/* out — GPU's NUMA node (-1 if unknown) */
	__u32 num_pages;	/* out — number of 64KB pages pinned */
	__u32 _pad;		/* padding — explicit pad for alignment */
	__u64 bar1_consumed;	/* out — total BAR1 bytes consumed */
};

/*
 * GPU unpin parameters: passed to IOCTL_GPU_UNPIN.
 * Releases a pinned GPU buffer. Handles both normal and revoked states:
 *   - Normal: nvidia_p2p_put_pages releases the pin.
 *   - Revoked (cudaFree while pinned): nvidia_p2p_free_page_table
 *     frees the struct without taking NVIDIA locks.
 */
struct dmaplane_gpu_unpin_params {
	__u32 handle;		/* in  — GPU buffer handle from GPU_PIN */
};

/*
 * GPU DMA parameters: passed to IOCTL_GPU_DMA_TO_HOST and
 * IOCTL_GPU_DMA_FROM_HOST.
 *
 * GPU_DMA_TO_HOST: reads GPU BAR pages via memcpy_fromio (PCIe
 *   non-posted reads, ~107 MB/s on RTX 5000).
 * GPU_DMA_FROM_HOST: writes to GPU BAR pages via memcpy_toio (PCIe
 *   posted writes, ~10 GB/s with write-combining).
 */
struct dmaplane_gpu_dma_params {
	__u32 gpu_handle;	/* in  — GPU buffer handle */
	__u32 host_handle;	/* in  — host buffer handle (from CREATE_BUFFER) */
	__u64 offset;		/* in  — byte offset within both buffers */
	__u64 size;		/* in  — transfer size in bytes */
	__u64 elapsed_ns;	/* out — transfer time in nanoseconds */
	__u64 throughput_mbps;	/* out — throughput in MB/s */
};

/*
 * GPU benchmark parameters: passed to IOCTL_GPU_BENCH.
 * Runs bidirectional BAR throughput benchmark on pre-pinned GPU
 * and pre-allocated host buffers.  Transfer size is min(gpu, host).
 */
struct dmaplane_gpu_bench_params {
	__u32 gpu_handle;	/* in  — pre-pinned GPU buffer handle */
	__u32 host_handle;	/* in  — pre-allocated host buffer handle */
	__u64 size;		/* in  — reserved (benchmark uses min of buffer sizes) */
	__u32 iterations;	/* in  — transfers per direction */
	__u32 _pad;		/* padding — explicit pad for alignment */
	__u64 h2g_bandwidth_mbps;	/* out — host→GPU throughput in MB/s */
	__u64 g2h_bandwidth_mbps;	/* out — GPU→host throughput in MB/s */
};

/*
 * GPU MR registration parameters: passed to IOCTL_GPU_REGISTER_MR.
 * Registers GPU BAR pages as an RDMA memory region.  Uses the
 * contiguous WC mapping (rdma_vaddr) as sge.addr for rxe's memcpy.
 * Goes into the same mrs[] array as host MRs — deregister via
 * the existing IOCTL_DEREGISTER_MR (0x21).
 */
struct dmaplane_gpu_mr_params {
	__u32 gpu_handle;	/* in  — GPU buffer handle from GPU_PIN */
	__u32 mr_id;		/* out — MR ID (shared namespace with host MRs) */
	__u32 lkey;		/* out — local key (pd->local_dma_lkey) */
	__u32 rkey;		/* out — remote key (0 for local_dma_lkey) */
};

/*
 * GPU RDMA loopback parameters: passed to IOCTL_GPU_LOOPBACK.
 * Sends data from a GPU-backed MR (on QP-A) to a host-backed MR
 * (on QP-B) via the rxe loopback pair.  Proves GPU VRAM can
 * traverse the RDMA data path.
 */
struct dmaplane_gpu_loopback_params {
	__u32 gpu_mr_id;	/* in  — GPU-backed MR (sender on QP-A) */
	__u32 host_mr_id;	/* in  — host-backed MR (receiver on QP-B) */
	__u32 size;		/* in  — bytes to send */
	__u32 _pad;		/* padding — explicit pad for alignment */
	__u64 latency_ns;	/* out — round-trip time in nanoseconds */
	__u32 recv_bytes;	/* out — bytes received by QP-B */
	__u32 status;		/* out — 0 on success */
};

/*
 * GPU statistics: returned by IOCTL_GET_GPU_STATS.
 * Lifetime counters — always present even without CONFIG_DMAPLANE_GPU
 * (all zeros if GPU support not compiled in).
 */
struct dmaplane_gpu_stats {
	__u64 pins_total;		/* out — lifetime GPU buffers pinned */
	__u64 unpins_total;		/* out — lifetime GPU buffers unpinned */
	__u64 callbacks_fired;		/* out — unpin callbacks from NVIDIA */
	__u64 dma_h2g_bytes;		/* out — host→GPU bytes transferred */
	__u64 dma_g2h_bytes;		/* out — GPU→host bytes transferred */
	__u64 gpu_mrs_registered;	/* out — lifetime GPU MRs registered */
};

/* Phase 8 ioctl commands: GPU 0x60–0x67 */

/* Pin GPU VRAM; returns handle, page count, BAR1 consumed. */
#define DMAPLANE_IOCTL_GPU_PIN \
	_IOWR(DMAPLANE_IOC_MAGIC, 0x60, struct dmaplane_gpu_pin_params)

/* Unpin a GPU buffer by handle. */
#define DMAPLANE_IOCTL_GPU_UNPIN \
	_IOW(DMAPLANE_IOC_MAGIC, 0x61, struct dmaplane_gpu_unpin_params)

/* GPU VRAM → host DRAM transfer (memcpy_fromio). */
#define DMAPLANE_IOCTL_GPU_DMA_TO_HOST \
	_IOWR(DMAPLANE_IOC_MAGIC, 0x62, struct dmaplane_gpu_dma_params)

/* Host DRAM → GPU VRAM transfer (memcpy_toio). */
#define DMAPLANE_IOCTL_GPU_DMA_FROM_HOST \
	_IOWR(DMAPLANE_IOC_MAGIC, 0x63, struct dmaplane_gpu_dma_params)

/* Run bidirectional GPU BAR throughput benchmark. */
#define DMAPLANE_IOCTL_GPU_BENCH \
	_IOWR(DMAPLANE_IOC_MAGIC, 0x64, struct dmaplane_gpu_bench_params)

/* Register GPU BAR pages as an RDMA memory region. */
#define DMAPLANE_IOCTL_GPU_REGISTER_MR \
	_IOWR(DMAPLANE_IOC_MAGIC, 0x65, struct dmaplane_gpu_mr_params)

/* RDMA loopback: GPU MR → host MR via QP-A/QP-B. */
#define DMAPLANE_IOCTL_GPU_LOOPBACK \
	_IOWR(DMAPLANE_IOC_MAGIC, 0x66, struct dmaplane_gpu_loopback_params)

/* Get GPU statistics (racy snapshot). */
#define DMAPLANE_IOCTL_GET_GPU_STATS \
	_IOR(DMAPLANE_IOC_MAGIC, 0x67, struct dmaplane_gpu_stats)

/* ── Phase 8: Peer RDMA (cross-machine) ──────────────────── */

/*
 * Peer RDMA connection info: exchanged between machines via TCP.
 * Each side calls IOCTL_RDMA_INIT_PEER to get local metadata,
 * sends it to the remote side, then calls IOCTL_RDMA_CONNECT_PEER
 * with the remote side's metadata to connect the peer QP.
 *
 * The MAC address is required for RoCEv2 — without the correct
 * destination MAC in the Address Handle, Ethernet frames go nowhere.
 */
struct dmaplane_rdma_peer_info {
	__u32 qp_num;		/* out/in — QP number */
	__u16 lid;		/* out/in — Local Identifier (unused for RoCE) */
	__u16 _pad1;		/* padding */
	__u8  gid[16];		/* out/in — GID (IPv6 format, 16 bytes) */
	__u8  mac[6];		/* out/in — Ethernet MAC for RoCEv2 AH */
	__u8  _pad2[2];		/* padding */
	__u32 status;		/* out — 0 on success */
	__u32 _pad3;		/* padding */
};

/*
 * Remote RDMA transfer parameters: passed to IOCTL_RDMA_REMOTE_SEND
 * and IOCTL_RDMA_REMOTE_RECV.
 *
 * REMOTE_SEND: post SEND on qp_peer, poll cq_peer for completion.
 * REMOTE_RECV: post RECV on qp_peer, poll cq_peer for completion
 *   (must be posted BEFORE the remote side sends — RC QP semantics).
 */
struct dmaplane_rdma_remote_xfer_params {
	__u32 mr_id;		/* in  — MR to send from / receive into */
	__u32 size;		/* in  — bytes to transfer */
	__u64 elapsed_ns;	/* out — operation latency in nanoseconds */
	__u64 throughput_mbps;	/* out — throughput in MB/s */
	__u32 status;		/* out — 0 on success */
	__u32 _pad;		/* padding */
};

/* Phase 8 ioctl commands: peer RDMA 0x90–0x94 */

/* Create peer QP + CQ; returns local QP/GID/MAC for TCP exchange. */
#define DMAPLANE_IOCTL_RDMA_INIT_PEER \
	_IOR(DMAPLANE_IOC_MAGIC, 0x90, struct dmaplane_rdma_peer_info)

/* Connect peer QP using remote metadata from TCP exchange. */
#define DMAPLANE_IOCTL_RDMA_CONNECT_PEER \
	_IOW(DMAPLANE_IOC_MAGIC, 0x91, struct dmaplane_rdma_peer_info)

/* Post SEND on peer QP from a local MR. */
#define DMAPLANE_IOCTL_RDMA_REMOTE_SEND \
	_IOWR(DMAPLANE_IOC_MAGIC, 0x92, struct dmaplane_rdma_remote_xfer_params)

/* Post RECV on peer QP into a local MR. */
#define DMAPLANE_IOCTL_RDMA_REMOTE_RECV \
	_IOWR(DMAPLANE_IOC_MAGIC, 0x93, struct dmaplane_rdma_remote_xfer_params)

/* Destroy peer QP and CQ. */
#define DMAPLANE_IOCTL_RDMA_DESTROY_PEER \
	_IO(DMAPLANE_IOC_MAGIC, 0x94)

/* ── Phase 9: RDMA WRITE with Immediate ──────────────────── */

/*
 * IB access flag constants — mirrors kernel IB_ACCESS_* values.
 * Defined here so userspace can set access_flags without depending
 * on rdma/ib_verbs.h.
 */
#define DMAPLANE_IB_ACCESS_LOCAL_WRITE	1
#define DMAPLANE_IB_ACCESS_REMOTE_WRITE	(1 << 1)
#define DMAPLANE_IB_ACCESS_REMOTE_READ	(1 << 2)

/*
 * RDMA WRITE with immediate data parameters.
 *
 * Posts IB_WR_RDMA_WRITE_WITH_IMM on the specified QP: writes 'length'
 * bytes from local MR at local_offset to remote_addr (rkey), and delivers
 * imm_data through the receiver's CQ completion.
 *
 * The receiver MUST have a recv WR posted — WRITE_WITH_IMM consumes one
 * recv WR to deliver the immediate value.  Without a matching recv, the
 * remote QP hits RNR (Receiver Not Ready) and retries until timeout.
 *
 * use_peer_qp: 0 = loopback QP-A→QP-B, 1 = cross-machine peer QP.
 */
struct dmaplane_write_imm_params {
	__u32 local_mr_id;	/* in  — source MR */
	__u32 length;		/* in  — bytes to write */
	__u64 local_offset;	/* in  — offset within source MR */
	__u64 remote_addr;	/* in  — destination address in remote MR */
	__u32 remote_rkey;	/* in  — remote MR's rkey */
	__u32 imm_data;		/* in  — 32-bit immediate value */
	__u32 use_peer_qp;	/* in  — 0=loopback, 1=peer */
	__u32 status;		/* out — WC status (0=success) */
	__u64 elapsed_ns;	/* out — completion time */
};
/* Size: 48 bytes.  Python format: =IIQQIIIIQ */

/*
 * Post a receive WR.  Each posted recv can absorb one WRITE_WITH_IMM
 * completion (the immediate data is delivered via the recv CQ entry).
 * Must be called BEFORE the corresponding write_imm is sent.
 *
 * use_peer_qp: 0 = post on QP-B (loopback recv), 1 = post on qp_peer.
 */
struct dmaplane_post_recv_params {
	__u32 mr_id;		/* in  — MR to receive into */
	__u32 size;		/* in  — max recv size */
	__u32 use_peer_qp;	/* in  — 0=loopback QP-B, 1=peer */
	__u32 status;		/* out — 0=success */
};
/* Size: 16 bytes.  Python format: =IIII */

/*
 * Poll recv CQ for a WRITEIMM completion.  Blocks up to timeout_ms
 * milliseconds.  On success, returns the 32-bit immediate data and
 * byte length from the work completion.
 *
 * The ioctl returns 0 even on timeout; the caller distinguishes
 * success from timeout via the status field (0=success, nonzero=timeout).
 * Only catastrophic WC errors return negative from the ioctl.
 *
 * use_peer_qp: 0 = poll loopback CQ-B, 1 = poll peer CQ.
 */
struct dmaplane_poll_recv_params {
	__u32 use_peer_qp;	/* in  — 0=loopback cq_b, 1=peer cq_peer */
	__u32 timeout_ms;	/* in  — max wait time */
	__u32 status;		/* out — WC status (0=success) */
	__u32 wc_flags;		/* out — IB_WC_WITH_IMM etc */
	__u32 imm_data;		/* out — immediate data from sender */
	__u32 byte_len;		/* out — RDMA payload size */
	__u64 elapsed_ns;	/* out — poll elapsed time */
};
/* Size: 32 bytes.  Python format: =IIIIIIQ */

/* Phase 9 ioctl commands: WRITEIMM 0x80–0x82 */

/* Post RDMA WRITE WITH IMMEDIATE (data + 32-bit immediate). */
#define DMAPLANE_IOCTL_RDMA_WRITE_IMM \
	_IOWR(DMAPLANE_IOC_MAGIC, 0x80, struct dmaplane_write_imm_params)

/* Post a recv WR to absorb one WRITEIMM completion. */
#define DMAPLANE_IOCTL_RDMA_POST_RECV \
	_IOWR(DMAPLANE_IOC_MAGIC, 0x81, struct dmaplane_post_recv_params)

/* Poll recv CQ for WRITEIMM completion; returns imm_data + byte_len. */
#define DMAPLANE_IOCTL_RDMA_POLL_RECV \
	_IOWR(DMAPLANE_IOC_MAGIC, 0x82, struct dmaplane_poll_recv_params)

/* ── Shorthand aliases for userspace convenience ─────────── */

#ifndef __KERNEL__
/* Buffer type constants */
#define BUF_TYPE_COHERENT	DMAPLANE_BUF_TYPE_COHERENT
#define BUF_TYPE_PAGES		DMAPLANE_BUF_TYPE_PAGES

/* Ioctl short names — used by kvcache_common.h and dmaplane_py.py */
#define IOCTL_CREATE_BUFFER	DMAPLANE_IOCTL_CREATE_BUFFER
#define IOCTL_DESTROY_BUFFER	DMAPLANE_IOCTL_DESTROY_BUFFER
#define IOCTL_GET_MMAP_INFO	DMAPLANE_IOCTL_GET_MMAP_INFO
#define IOCTL_SETUP_RDMA	DMAPLANE_IOCTL_SETUP_RDMA
#define IOCTL_TEARDOWN_RDMA	DMAPLANE_IOCTL_TEARDOWN_RDMA
#define IOCTL_REGISTER_MR	DMAPLANE_IOCTL_REGISTER_MR
#define IOCTL_DEREGISTER_MR	DMAPLANE_IOCTL_DEREGISTER_MR
#define IOCTL_GPU_PIN		DMAPLANE_IOCTL_GPU_PIN
#define IOCTL_GPU_UNPIN		DMAPLANE_IOCTL_GPU_UNPIN
#define IOCTL_GPU_REGISTER_MR	DMAPLANE_IOCTL_GPU_REGISTER_MR
#define IOCTL_RDMA_WRITE_IMM	DMAPLANE_IOCTL_RDMA_WRITE_IMM
#define IOCTL_RDMA_POST_RECV	DMAPLANE_IOCTL_RDMA_POST_RECV
#define IOCTL_RDMA_POLL_RECV	DMAPLANE_IOCTL_RDMA_POLL_RECV
#define IOCTL_RDMA_INIT_PEER	DMAPLANE_IOCTL_RDMA_INIT_PEER
#define IOCTL_RDMA_CONNECT_PEER	DMAPLANE_IOCTL_RDMA_CONNECT_PEER
#define IOCTL_RDMA_DESTROY_PEER	DMAPLANE_IOCTL_RDMA_DESTROY_PEER
#define IOCTL_GET_STATS		DMAPLANE_IOCTL_GET_STATS
#define IOCTL_LOOPBACK_TEST	DMAPLANE_IOCTL_LOOPBACK_TEST
#define IOCTL_PINGPONG_BENCH	DMAPLANE_IOCTL_PINGPONG_BENCH
#define IOCTL_STREAMING_BENCH	DMAPLANE_IOCTL_STREAMING_BENCH
#define IOCTL_RDMA_REMOTE_SEND	DMAPLANE_IOCTL_RDMA_REMOTE_SEND
#define IOCTL_RDMA_REMOTE_RECV	DMAPLANE_IOCTL_RDMA_REMOTE_RECV
#endif /* !__KERNEL__ */

#endif /* _DMAPLANE_UAPI_H */
