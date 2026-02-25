// SPDX-License-Identifier: GPL-2.0
/*
 * dmaplane main.c — Character device driver, ioctl dispatch, and mmap
 * Copyright (c) 2026 Graziano Labs Corp.
 *
 * This is the entry point for the dmaplane kernel module.  It registers
 * a platform device for DMA operations, a character device at
 * /dev/dmaplane, dispatches ioctls to per-command handlers, provides
 * mmap support for DMA buffers, and manages per-file context.
 *
 * Architecture:
 *   userspace ioctl → dmaplane_ioctl (this file) → handler functions
 *   Each open fd gets a dmaplane_file_ctx.  Channel creation spawns a
 *   kthread that drains the submission ring and fills the completion ring.
 *   Buffer management is a separate subsystem (dmabuf_rdma.c) — buffers
 *   are global resources, not per-channel.  mmap maps DMA buffer pages
 *   directly into userspace for zero-copy I/O.
 *
 * Data flow (channels):
 *   userspace --[IOCTL_SUBMIT]--> sub_ring --[worker]--> comp_ring
 *             <--[IOCTL_COMPLETE]--
 *
 * Data flow (buffers):
 *   userspace --[IOCTL_CREATE_BUFFER]--> dmabuf_rdma_create_buffer
 *   userspace --[mmap]--> dmaplane_mmap --> vm_insert_page / dma_mmap_coherent
 *
 * Locking summary:
 *   dev_mutex (struct mutex)
 *     Protects: channel slot allocation and deallocation in
 *               dma_dev->channels[].
 *     Acquired by: dmaplane_ioctl_create_channel, dmaplane_release,
 *                  dmaplane_exit.
 *     Why mutex: only taken in process/ioctl context where sleeping is
 *                permitted; never on the submit/complete hot path.
 *     Ordering: outermost lock.  Independent of buf_lock.
 *
 *   buf_lock (struct mutex)
 *     Protects: buffer array, slot allocation, mmap lookups.
 *     Acquired by: buffer ioctl handlers, dmaplane_mmap.
 *     Why mutex: sleeping context OK — only from process context.
 *     Ordering: independent of dev_mutex (never nested).
 *
 *   channel->lock (struct mutex)
 *     Protects: channel state (shutdown flag, worker pointer).
 *     Acquired by: dmaplane_release (shutdown sequence).
 *     Why mutex: sleeping context OK — only from process context.
 *     Ordering: dev_mutex → channel->lock → ring spinlocks.
 *
 *   sub_ring.lock / comp_ring.lock (spinlock_t, irqsave)
 *     Protects: ring entries[] array and head/tail index updates for
 *               the respective ring.
 *     Acquired by: ioctl submit/complete paths (producer or consumer)
 *                  and the worker thread (the other side).
 *     Why spinlock: ring operations are very short and may be called
 *                   from contexts where sleeping is not desired.
 *     Ordering: never nested; each ring lock is independent.
 *
 * Memory ordering:
 *   smp_store_release() is used on every head/tail advancement to
 *   ensure that the entry write (or read) is visible to the other side
 *   before the index update becomes visible.  The matching acquire
 *   semantics are provided by spin_lock_irqsave() on the consumer/
 *   producer side.  This is a no-op on x86 (TSO) but required for
 *   correctness on ARM and RISC-V.
 */

#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/dma-mapping.h>
#include <linux/mm.h>
#include <linux/platform_device.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/kthread.h>
#include <linux/sched.h>
#include <linux/delay.h>
#include <linux/capability.h>

#include "dmaplane.h"
#include "dmabuf_rdma.h"
#include "dmabuf_export.h"
#include "rdma_engine.h"
#include "benchmark.h"
#include "numa_topology.h"
#include "flow_control.h"
#include "dmaplane_trace.h"
#include "dmaplane_debugfs.h"
#include "dmaplane_histogram.h"
#ifdef CONFIG_DMAPLANE_GPU
#include "gpu_p2p.h"
#endif

MODULE_LICENSE("GPL v2");
MODULE_AUTHOR("Graziano Labs Corp.");
MODULE_DESCRIPTION("dmaplane — host-side data path for AI-scale systems");
MODULE_VERSION("0.1.0");
/*
 * Import the DMA_BUF symbol namespace.  Since kernel 5.x, the dma-buf
 * framework exports its symbols (dma_buf_export, dma_buf_fd, dma_buf_put,
 * etc.) under this namespace.  Without this declaration, the module fails
 * to load with "Unknown symbol" errors for all dma-buf API calls.
 */
MODULE_IMPORT_NS(DMA_BUF);

/* Singleton device context — allocated in dmaplane_init, freed in dmaplane_exit */
static struct dmaplane_dev *dma_dev;

/* Module parameter: run dma-buf kernel self-test at load time */
static int test_dmabuf;
module_param(test_dmabuf, int, 0444);
MODULE_PARM_DESC(test_dmabuf, "Run dma-buf export self-test at module load (0=off, 1=on)");

/* ------------------------------------------------------------------ */
/* Ring buffer operations                                              */
/* ------------------------------------------------------------------ */

/*
 * dmaplane_ring_init - Zero-initialise a ring buffer and its lock.
 * @ring: ring to initialise.
 *
 * Called once per ring during channel creation.  No locking required
 * because the channel is not yet visible to other threads.
 */
static void dmaplane_ring_init(struct dmaplane_ring *ring)
{
	memset(ring->entries, 0, sizeof(ring->entries));
	spin_lock_init(&ring->lock);
	ring->head = 0;
	ring->tail = 0;
}

/* ------------------------------------------------------------------ */
/* Worker thread                                                       */
/* ------------------------------------------------------------------ */

/*
 * dmaplane_worker_fn — Per-channel worker thread.
 *
 * Drains submissions from the submission ring, "processes" each entry
 * (Phase 1: increments payload by 1 to prove the worker touched it),
 * and pushes results onto the completion ring.
 *
 * Locking:
 *   Acquires sub_ring.lock (consumer side) to read submissions and
 *   comp_ring.lock (producer side) to write completions.  No lock is
 *   held during entry processing or sleep, minimising contention.
 *
 * Backpressure:
 *   If the completion ring is full, the worker yields via cond_resched()
 *   in a spin-wait loop rather than dropping entries.  This applies
 *   backpressure upstream: the submission ring fills, and userspace
 *   gets -ENOSPC on the next submit, forcing it to drain completions.
 *
 * Shutdown:
 *   The outer loop checks kthread_should_stop().  The wait_event also
 *   checks kthread_should_stop() so the worker wakes up on stop.
 *   The completion ring yield loop additionally checks kthread_should_stop()
 *   to ensure the worker can exit even if the ring stays full.
 */
static int dmaplane_worker_fn(void *data)
{
	struct dmaplane_channel *chan = data;
	struct dmaplane_ring *sub = &chan->sub_ring;
	struct dmaplane_ring *comp = &chan->comp_ring;
	unsigned int batch;

	pr_debug("worker %u started\n", chan->id);

	while (!kthread_should_stop()) {
		/*
		 * Sleep until the submission ring is non-empty or the
		 * kthread stop flag is set.  wait_event_interruptible
		 * puts the worker to sleep on chan->wait_queue and
		 * re-checks the condition atomically on each wakeup,
		 * preventing the lost-wakeup race where a wake_up
		 * arrives between the condition check and the sleep.
		 */
		wait_event_interruptible(chan->wait_queue,
			!dmaplane_ring_empty(sub) || kthread_should_stop());

		/* If stopped and ring is drained, exit cleanly */
		if (kthread_should_stop() && dmaplane_ring_empty(sub))
			break;

		/* Drain submissions */
		batch = 0;
		while (!dmaplane_ring_empty(sub)) {
			struct dmaplane_ring_entry entry;
			unsigned long flags;

			/* Read one submission under the lock */
			spin_lock_irqsave(&sub->lock, flags);
			/* Re-check under lock — the unlocked check above
			 * is an optimisation; this is the authoritative one */
			if (dmaplane_ring_empty(sub)) {
				spin_unlock_irqrestore(&sub->lock, flags);
				break;
			}
			entry = sub->entries[sub->tail % DMAPLANE_RING_SIZE];
			/* smp_store_release: ensure the entry read above
			 * is complete before the tail advance becomes
			 * visible to the submit-side producer. */
			smp_store_release(&sub->tail, sub->tail + 1);
			spin_unlock_irqrestore(&sub->lock, flags);

			/*
			 * Phase 1 processing: increment payload by 1 to
			 * prove the worker touched the entry.  Future phases
			 * replace this with DMA ops, RDMA posts, or GPU
			 * transfers.
			 */
			entry.payload += 1;

			/*
			 * Push to completion ring with backpressure.
			 *
			 * If the completion ring is full (userspace is not
			 * draining completions fast enough), we spin-yield
			 * rather than dropping the entry:
			 *  1. Acquire comp_ring lock.
			 *  2. If room, write entry, advance head with
			 *     smp_store_release, and break.
			 *  3. If full, release lock, cond_resched() to
			 *     yield CPU, then retry.
			 *  4. If kthread_should_stop() during wait, bail
			 *     to done — entry is lost, acceptable during
			 *     channel teardown.
			 */
			for (;;) {
				spin_lock_irqsave(&comp->lock, flags);
				if (!dmaplane_ring_full(comp)) {
					comp->entries[comp->head % DMAPLANE_RING_SIZE] = entry;
					/* smp_store_release: entry store must
					 * be visible before head advance
					 * reaches the complete-ioctl consumer */
					smp_store_release(&comp->head, comp->head + 1);
					spin_unlock_irqrestore(&comp->lock, flags);
					break;
				}
				spin_unlock_irqrestore(&comp->lock, flags);

				if (kthread_should_stop())
					goto done;
				cond_resched();
			}

			atomic_dec(&chan->in_flight);
			atomic64_inc(&chan->stats.total_completions);
			batch++;

			/* Yield every 64 entries to avoid hogging CPU */
			if (batch % 64 == 0)
				cond_resched();
		}
	}

done:
	pr_debug("worker %u stopped\n", chan->id);
	return 0;
}

/* ------------------------------------------------------------------ */
/* Channel management                                                  */
/* ------------------------------------------------------------------ */

/*
 * dmaplane_channel_init - Reset all channel state to initial values.
 * @chan: channel to initialise.
 * @id:  channel index in dma_dev->channels[].
 *
 * Called under dev_mutex during IOCTL_CREATE_CHANNEL.  Initialises both
 * rings, the wait queue, kref (starts at 1 for the file_ctx holder),
 * per-channel mutex, and atomic stats.  The channel is not yet marked
 * active, so no other thread can observe it.
 */
static void dmaplane_channel_init(struct dmaplane_channel *chan, unsigned int id)
{
	kref_init(&chan->refcount);	/* Initial ref for file_ctx */
	mutex_init(&chan->lock);
	dmaplane_ring_init(&chan->sub_ring);
	dmaplane_ring_init(&chan->comp_ring);
	init_waitqueue_head(&chan->wait_queue);
	chan->id = id;
	atomic_set(&chan->in_flight, 0);
	chan->shutdown = false;
	chan->active = false;
	chan->worker = NULL;
	atomic64_set(&chan->stats.total_submissions, 0);
	atomic64_set(&chan->stats.total_completions, 0);
	atomic_set(&chan->stats.ring_high_watermark, 0);
	atomic_set(&chan->stats.dropped_count, 0);
}

/*
 * dmaplane_channel_release - kref release callback.
 * @ref: pointer to the kref embedded in dmaplane_channel.
 *
 * Called when the last reference is dropped (kref hits zero).
 * Marks the channel slot inactive so it can be reused by
 * create_channel.  Also decrements the device-level active_channels
 * counter and increments total_channels_destroyed.
 *
 * At this point the worker has exited and the file_ctx has released
 * its reference — no other code path can access this channel.
 */
void dmaplane_channel_release(struct kref *ref)
{
	struct dmaplane_channel *chan =
		container_of(ref, struct dmaplane_channel, refcount);

	chan->active = false;
	atomic_dec(&dma_dev->active_channels);
	atomic64_inc(&dma_dev->total_channels_destroyed);
	pr_debug("channel %u released (kref=0)\n", chan->id);
}

/*
 * dmaplane_channel_destroy - Stop a channel's worker and drop file_ctx ref.
 * @chan: channel to destroy.
 *
 * Must be called under dev_mutex (from release or module exit).
 *
 * Teardown sequence:
 *  1. Acquire channel->lock.
 *  2. Set chan->shutdown = true to signal the worker.
 *  3. Release channel->lock (so the worker can check shutdown under
 *     lock if needed in future phases).
 *  4. Wake the worker so it can observe the flag and kthread_should_stop().
 *  5. kthread_stop() waits for the worker to exit.  Note: the kthread
 *     wrapper may skip threadfn entirely if KTHREAD_SHOULD_STOP is set
 *     before the kthread gets a timeslice (common when channels are
 *     created and destroyed quickly).
 *  6. kref_put drops the worker's ref (refcount 2 → 1).
 *  7. kref_put drops the file_ctx's ref (refcount 1 → 0), triggering
 *     dmaplane_channel_release which marks the slot inactive.
 *
 * After return, the worker is guaranteed stopped and ring memory is safe
 * to reuse (but not zeroed — stats remain readable until re-init).
 */
static void dmaplane_channel_destroy(struct dmaplane_channel *chan)
{
	if (!chan->active)
		return;

	mutex_lock(&chan->lock);
	chan->shutdown = true;
	mutex_unlock(&chan->lock);

	wake_up_interruptible(&chan->wait_queue);

	if (chan->worker) {
		kthread_stop(chan->worker);
		chan->worker = NULL;
	}

	/*
	 * Drop both refs.  kthread_stop guarantees the worker has exited,
	 * but threadfn may not have run — the kthread wrapper skips
	 * threadfn if KTHREAD_SHOULD_STOP is already set (happens when
	 * a channel is created and destroyed before the scheduler gives
	 * the kthread a timeslice).  We drop the worker's ref here
	 * unconditionally, then the file_ctx's ref triggers release.
	 */
	kref_put(&chan->refcount, dmaplane_channel_release);  /* worker ref */
	kref_put(&chan->refcount, dmaplane_channel_release);  /* file_ctx ref */
	pr_debug("channel %u destroyed\n", chan->id);
}

/* ------------------------------------------------------------------ */
/* Ioctl handlers                                                      */
/* ------------------------------------------------------------------ */

/*
 * dmaplane_ioctl_create_channel - Allocate a channel and start its worker.
 * @ctx: file context (carries the owning dev pointer and channel slot).
 * @arg: userspace pointer to struct dmaplane_channel_params (output).
 *
 * Finds the first free slot in dev->channels[], initialises it, creates
 * a kthread with kthread_create() (which does NOT start it), stores the
 * task pointer, then wake_up_process() makes it runnable.  The two-step
 * create/wake pattern ensures chan->worker is set before the thread can
 * run, preventing a race where the worker dereferences itself before
 * the pointer is stored.
 *
 * Concurrency: dev_mutex serialises slot search and active-flag set so
 * two concurrent CREATE_CHANNEL calls cannot claim the same slot.
 *
 * Returns:
 *   0       on success (channel_id written to userspace via @arg).
 *  -EBUSY   if this fd already owns a channel (one channel per fd).
 *  -ENOSPC  if all DMAPLANE_MAX_CHANNELS slots are in use.
 *  -EFAULT  if copy_to_user fails.
 *  -errno   if kthread_create fails (propagated via PTR_ERR).
 */
static long dmaplane_ioctl_create_channel(struct dmaplane_file_ctx *ctx,
					   unsigned long arg)
{
	struct dmaplane_dev *dev = ctx->dev;
	struct dmaplane_channel_params params;
	struct dmaplane_channel *chan;
	struct task_struct *worker;
	int i;

	/* One channel per fd — simplifies lifecycle; future phases may relax */
	if (ctx->chan)
		return -EBUSY;

	mutex_lock(&dev->dev_mutex);

	/*
	 * Linear scan of DMAPLANE_MAX_CHANNELS slots.  Deliberate
	 * simplicity — IDR/bitmap would be overkill for 4 slots.
	 */
	for (i = 0; i < DMAPLANE_MAX_CHANNELS; i++) {
		if (!dev->channels[i].active)
			break;
	}
	if (i == DMAPLANE_MAX_CHANNELS) {
		mutex_unlock(&dev->dev_mutex);
		return -ENOSPC;
	}

	chan = &dev->channels[i];
	dmaplane_channel_init(chan, i);
	chan->active = true;

	/* Take a second ref for the worker thread (init gave us 1 for file_ctx) */
	kref_get(&chan->refcount);

	/*
	 * kthread_create allocates the task but does NOT start it.
	 * We store chan->worker first, then wake_up_process makes it
	 * runnable, ensuring the pointer is visible before the thread runs.
	 */
	worker = kthread_create(dmaplane_worker_fn, chan, "dmaplane/%d", i);
	if (IS_ERR(worker)) {
		/*
		 * No other code path holds a reference — the channel was
		 * just initialised and never stored in file_ctx.  Reset
		 * directly; kref_init on next reuse will reinitialise.
		 */
		chan->active = false;
		mutex_unlock(&dev->dev_mutex);
		return PTR_ERR(worker);
	}
	chan->worker = worker;
	wake_up_process(worker);

	atomic_inc(&dev->active_channels);
	atomic64_inc(&dev->total_channels_created);

	ctx->chan = chan;

	mutex_unlock(&dev->dev_mutex);

	/* Return channel ID to userspace */
	params.channel_id = i;
	params._pad = 0;
	if (copy_to_user((void __user *)arg, &params, sizeof(params)))
		return -EFAULT;

	pr_debug("channel %d created\n", i);
	return 0;
}

/*
 * dmaplane_ioctl_submit - Enqueue one entry into the submission ring.
 * @ctx: file context (must have an assigned channel).
 * @arg: userspace pointer to struct dmaplane_submit_params (input).
 *
 * Copies the entry from userspace, acquires sub_ring.lock, writes the
 * entry at head % RING_SIZE, advances head with smp_store_release, and
 * wakes the worker.  When the ring is full, returns -ENOSPC — userspace
 * must drain completions to make progress (backpressure mechanism).
 *
 * High watermark tracking: after the entry is written but before the
 * lock is released, we snapshot the occupancy and update the atomic
 * stat if the new value is higher.  Safe because the lock serialises
 * writers; the atomic_set is relaxed (no ordering needed for stats).
 *
 * Returns:
 *   0       on success.
 *  -ENODEV  if no channel has been created on this fd.
 *  -EFAULT  if copy_from_user fails.
 *  -ENOSPC  if the submission ring is full.
 */
static long dmaplane_ioctl_submit(struct dmaplane_file_ctx *ctx,
				   unsigned long arg)
{
	struct dmaplane_channel *chan = ctx->chan;
	struct dmaplane_submit_params params;
	struct dmaplane_ring *ring;
	unsigned long flags;
	unsigned int occupancy;

	if (!chan)
		return -ENODEV;

	if (copy_from_user(&params, (void __user *)arg, sizeof(params)))
		return -EFAULT;

	ring = &chan->sub_ring;

	spin_lock_irqsave(&ring->lock, flags);

	if (dmaplane_ring_full(ring)) {
		spin_unlock_irqrestore(&ring->lock, flags);
		return -ENOSPC;
	}

	ring->entries[ring->head % DMAPLANE_RING_SIZE] = params.entry;
	/* smp_store_release: entry store must be visible before head advance */
	smp_store_release(&ring->head, ring->head + 1);

	/* Track high watermark under lock — compare-and-swap if new max */
	occupancy = dmaplane_ring_count(ring);
	{
		unsigned int old_wm = atomic_read(&chan->stats.ring_high_watermark);
		if (occupancy > old_wm)
			atomic_set(&chan->stats.ring_high_watermark, occupancy);
	}

	spin_unlock_irqrestore(&ring->lock, flags);

	atomic_inc(&chan->in_flight);
	atomic64_inc(&chan->stats.total_submissions);

	trace_dmaplane_ring_submit(chan->id, params.entry.payload,
				   dmaplane_ring_count(&chan->sub_ring));

	/* Wake the worker so it sees the new entry */
	wake_up_interruptible(&chan->wait_queue);

	return 0;
}

/*
 * dmaplane_ioctl_complete - Dequeue one entry from the completion ring.
 * @ctx: file context (must have an assigned channel).
 * @arg: userspace pointer to struct dmaplane_complete_params (output).
 *
 * Non-blocking: returns -EAGAIN immediately if the completion ring is
 * empty.  Userspace should poll/retry or interleave with submits.
 *
 * Returns:
 *   0       on success (entry copied to userspace).
 *  -ENODEV  if no channel has been created on this fd.
 *  -EAGAIN  if the completion ring is empty (try again later).
 *  -EFAULT  if copy_to_user fails.
 */
static long dmaplane_ioctl_complete(struct dmaplane_file_ctx *ctx,
				     unsigned long arg)
{
	struct dmaplane_channel *chan = ctx->chan;
	struct dmaplane_complete_params params;
	struct dmaplane_ring *ring;
	unsigned long flags;

	if (!chan)
		return -ENODEV;

	ring = &chan->comp_ring;

	spin_lock_irqsave(&ring->lock, flags);

	if (dmaplane_ring_empty(ring)) {
		spin_unlock_irqrestore(&ring->lock, flags);
		return -EAGAIN;
	}

	params.entry = ring->entries[ring->tail % DMAPLANE_RING_SIZE];
	/* smp_store_release: entry read must be complete before tail advance */
	smp_store_release(&ring->tail, ring->tail + 1);

	spin_unlock_irqrestore(&ring->lock, flags);

	trace_dmaplane_ring_complete(chan->id, params.entry.payload);

	if (copy_to_user((void __user *)arg, &params, sizeof(params)))
		return -EFAULT;

	return 0;
}

/*
 * dmaplane_ioctl_get_stats - Copy per-channel statistics to userspace.
 * @ctx: file context (must have an assigned channel).
 * @arg: userspace pointer to struct dmaplane_stats (output).
 *
 * Reads atomic counters from dmaplane_stats_kern into a plain
 * dmaplane_stats UAPI struct, then copies to userspace.  Each atomic
 * read is individually consistent, but the set may be momentarily
 * inconsistent (e.g., total_completions could briefly exceed
 * total_submissions if read between the two increments).
 * Acceptable for observability.
 *
 * Returns:
 *   0       on success.
 *  -ENODEV  if no channel has been created on this fd.
 *  -EFAULT  if copy_to_user fails.
 */
static long dmaplane_ioctl_get_stats(struct dmaplane_file_ctx *ctx,
				      unsigned long arg)
{
	struct dmaplane_channel *chan = ctx->chan;
	struct dmaplane_stats ustats;

	if (!chan)
		return -ENODEV;

	ustats.total_submissions  = atomic64_read(&chan->stats.total_submissions);
	ustats.total_completions  = atomic64_read(&chan->stats.total_completions);
	ustats.ring_high_watermark = atomic_read(&chan->stats.ring_high_watermark);
	ustats.dropped_count      = atomic_read(&chan->stats.dropped_count);

	if (copy_to_user((void __user *)arg, &ustats, sizeof(ustats)))
		return -EFAULT;

	return 0;
}

/* ------------------------------------------------------------------ */
/* mmap support                                                        */
/* ------------------------------------------------------------------ */

/*
 * VMA close — decrement mmap refcount when munmap'd or process exits.
 * This is the counterpart to the atomic_inc in dmaplane_mmap().
 */
static void dmaplane_vma_close(struct vm_area_struct *vma)
{
	struct dmaplane_buffer *buf = vma->vm_private_data;

	if (buf)
		atomic_dec(&buf->mmap_count);
}

/*
 * VMA open — increment on fork/mremap.
 * VM_DONTCOPY prevents fork inheritance, so this should only fire
 * on mremap (which VM_DONTEXPAND also blocks).  Defensive increment.
 */
static void dmaplane_vma_open(struct vm_area_struct *vma)
{
	struct dmaplane_buffer *buf = vma->vm_private_data;

	if (buf)
		atomic_inc(&buf->mmap_count);
}

static const struct vm_operations_struct dmaplane_vm_ops = {
	.open  = dmaplane_vma_open,
	.close = dmaplane_vma_close,
};

/*
 * dmaplane_mmap — Map DMA buffer pages into userspace.
 *
 * This completes the zero-copy data path: the same physical pages
 * allocated by dmabuf_rdma and (in Phase 4) registered as RDMA MRs
 * are mapped directly into userspace.  Writes by the application
 * land in the exact memory the NIC reads from for TX — no copies.
 *
 * The vm_pgoff (page offset) encodes the buffer ID.  Userspace
 * obtains the correct offset via IOCTL_GET_MMAP_INFO, then calls
 * mmap(NULL, size, prot, MAP_SHARED, fd, offset).
 *
 * Returns:
 *   0       on success.
 *  -EINVAL  if buffer not found or size exceeds buffer.
 *  -EAGAIN  if vm_insert_page fails for page-backed buffers.
 *  -errno   if dma_mmap_coherent fails for coherent buffers.
 */
static int dmaplane_mmap(struct file *filp, struct vm_area_struct *vma)
{
	struct dmaplane_file_ctx *ctx = filp->private_data;
	struct dmaplane_dev *dev = ctx->dev;
	struct dmaplane_buffer *buf;
	unsigned int buf_id = vma->vm_pgoff;
	unsigned long size = vma->vm_end - vma->vm_start;
	unsigned int i;
	int ret;

	mutex_lock(&dev->buf_lock);

	buf = dmabuf_rdma_find_buffer(dev, buf_id);
	if (!buf) {
		mutex_unlock(&dev->buf_lock);
		return -EINVAL;
	}

	if (size > buf->size) {
		mutex_unlock(&dev->buf_lock);
		return -EINVAL;
	}

	/*
	 * VM_DONTCOPY: no COW copies to forked children.  NIC may be
	 *   DMA-writing into these pages — a child with a stale mapping
	 *   would cause data corruption or use-after-free.
	 * VM_DONTEXPAND: no mremap expansion beyond buffer bounds.
	 * VM_DONTDUMP: exclude DMA memory from core dumps (may contain
	 *   sensitive data, and dump would race with DMA).
	 */
	vm_flags_set(vma, VM_DONTCOPY | VM_DONTEXPAND | VM_DONTDUMP);

	vma->vm_ops = &dmaplane_vm_ops;
	vma->vm_private_data = buf;

	switch (buf->alloc_type) {
	case DMAPLANE_BUF_TYPE_COHERENT:
		/*
		 * dma_mmap_coherent maps physically contiguous DMA memory
		 * into userspace.  It handles the PFN calculation and
		 * cache attribute setup internally.
		 *
		 * Reset vm_pgoff to 0: we used it to encode the buffer ID,
		 * but dma_mmap_coherent interprets it as a page offset into
		 * the coherent allocation.  A non-zero pgoff would cause
		 * ENXIO because the offset exceeds the allocation size.
		 */
		vma->vm_pgoff = 0;
		ret = dma_mmap_coherent(&dev->pdev->dev, vma, buf->vaddr,
					buf->dma_handle, size);
		if (ret) {
			mutex_unlock(&dev->buf_lock);
			return ret;
		}
		break;

	case DMAPLANE_BUF_TYPE_PAGES:
		/*
		 * vm_insert_page maps individual pages into the VMA.
		 * Must be called once per page — the pages are physically
		 * scattered and cannot be mapped with a single remap_pfn_range.
		 */
		for (i = 0; i < buf->nr_pages && i * PAGE_SIZE < size; i++) {
			ret = vm_insert_page(vma,
					     vma->vm_start + i * PAGE_SIZE,
					     buf->pages[i]);
			if (ret) {
				mutex_unlock(&dev->buf_lock);
				return ret;
			}
		}
		break;

	default:
		mutex_unlock(&dev->buf_lock);
		return -EINVAL;
	}

	/*
	 * Increment mmap_count on success.  This is NOT done via vma_open —
	 * Linux does NOT call vma_open for the initial mmap, only for
	 * fork/mremap.  This is a real gotcha: if you only increment in
	 * vma_open, the initial mmap is untracked and destroy won't be
	 * blocked by it.
	 */
	atomic_inc(&buf->mmap_count);

	pr_debug("buffer %u mmapped: size=%lu type=%d\n",
		 buf->id, size, buf->alloc_type);

	mutex_unlock(&dev->buf_lock);
	return 0;
}

/* ------------------------------------------------------------------ */
/* Phase 2 ioctl handlers                                              */
/* ------------------------------------------------------------------ */

/*
 * dmaplane_ioctl_create_buffer - Allocate a DMA buffer.
 * @ctx: file context (carries the owning dev pointer).
 * @arg: userspace pointer to struct dmaplane_buf_params (in/out).
 *
 * Copies params from userspace, delegates to dmabuf_rdma_create_buffer,
 * and copies the result (including the assigned buf_id) back.  If the
 * copy_to_user fails, the buffer is destroyed to prevent a leaked
 * buffer that userspace can never reference.
 *
 * Returns:
 *   0       on success.
 *  -EFAULT  if copy_from_user or copy_to_user fails.
 *  -EINVAL  if size or alloc_type is invalid.
 *  -ENOMEM  if no free slots or allocation fails.
 */
static long dmaplane_ioctl_create_buffer(struct dmaplane_file_ctx *ctx,
					  unsigned long arg)
{
	struct dmaplane_dev *dev = ctx->dev;
	struct dmaplane_buf_params params;
	int ret;

	if (copy_from_user(&params, (void __user *)arg, sizeof(params)))
		return -EFAULT;

	ret = dmabuf_rdma_create_buffer(dev, &params);
	if (ret)
		return ret;

	if (copy_to_user((void __user *)arg, &params, sizeof(params))) {
		/*
		 * Buffer was allocated but handle can't be returned —
		 * destroy it to prevent a leaked buffer that userspace
		 * can never reference.
		 */
		dmabuf_rdma_destroy_buffer(dev, params.buf_id);
		return -EFAULT;
	}

	return 0;
}

/*
 * dmaplane_ioctl_destroy_buffer - Free a DMA buffer by handle.
 * @ctx: file context.
 * @arg: userspace pointer to bare __u32 buffer handle.
 *
 * Returns:
 *   0       on success.
 *  -EFAULT  if copy_from_user fails.
 *  -ENOENT  if buffer not found.
 *  -EBUSY   if active mmap references exist.
 */
static long dmaplane_ioctl_destroy_buffer(struct dmaplane_file_ctx *ctx,
					   unsigned long arg)
{
	struct dmaplane_dev *dev = ctx->dev;
	__u32 buf_id;

	if (copy_from_user(&buf_id, (void __user *)arg, sizeof(buf_id)))
		return -EFAULT;

	return dmabuf_rdma_destroy_buffer(dev, buf_id);
}

/*
 * dmaplane_ioctl_get_mmap_info - Get mmap offset and size for a buffer.
 * @ctx: file context.
 * @arg: userspace pointer to struct dmaplane_mmap_info (in/out).
 *
 * Looks up the buffer by ID, returns the mmap offset (buf_id << PAGE_SHIFT)
 * and the buffer size.  Userspace uses these to call mmap(2).
 *
 * Returns:
 *   0       on success.
 *  -EFAULT  if copy_from_user or copy_to_user fails.
 *  -ENOENT  if buffer not found.
 */
static long dmaplane_ioctl_get_mmap_info(struct dmaplane_file_ctx *ctx,
					  unsigned long arg)
{
	struct dmaplane_dev *dev = ctx->dev;
	struct dmaplane_mmap_info info;
	struct dmaplane_buffer *buf;

	if (copy_from_user(&info, (void __user *)arg, sizeof(info)))
		return -EFAULT;

	mutex_lock(&dev->buf_lock);
	buf = dmabuf_rdma_find_buffer(dev, info.buf_id);
	if (!buf) {
		mutex_unlock(&dev->buf_lock);
		return -ENOENT;
	}

	/*
	 * mmap offset encoding: buffer ID in the pgoff field.
	 * Userspace calls mmap(NULL, size, prot, flags, fd, offset)
	 * where offset = buf_id << PAGE_SHIFT.  The driver extracts
	 * buf_id = vma->vm_pgoff in dmaplane_mmap.
	 */
	info.mmap_offset = (__u64)buf->id << PAGE_SHIFT;
	info.mmap_size = buf->size;

	mutex_unlock(&dev->buf_lock);

	if (copy_to_user((void __user *)arg, &info, sizeof(info)))
		return -EFAULT;

	return 0;
}

/*
 * dmaplane_ioctl_get_buf_stats - Copy buffer statistics to userspace.
 * @ctx: file context.
 * @arg: userspace pointer to struct dmaplane_buf_stats (output).
 *
 * Reads atomic counters into the UAPI struct.  Racy snapshot —
 * individually consistent, collectively approximate.
 *
 * Returns:
 *   0       on success.
 *  -EFAULT  if copy_to_user fails.
 */
static long dmaplane_ioctl_get_buf_stats(struct dmaplane_file_ctx *ctx,
					   unsigned long arg)
{
	struct dmaplane_dev *dev = ctx->dev;
	struct dmaplane_buf_stats bstats;

	bstats.buffers_created   = atomic64_read(&dev->stats.buffers_created);
	bstats.buffers_destroyed = atomic64_read(&dev->stats.buffers_destroyed);
	bstats.numa_local_allocs  = atomic64_read(&dev->stats.numa_local_allocs);
	bstats.numa_remote_allocs = atomic64_read(&dev->stats.numa_remote_allocs);
	bstats.numa_anon_allocs   = atomic64_read(&dev->stats.numa_anon_allocs);

	if (copy_to_user((void __user *)arg, &bstats, sizeof(bstats)))
		return -EFAULT;

	return 0;
}

/* ------------------------------------------------------------------ */
/* Phase 3 ioctl handlers                                              */
/* ------------------------------------------------------------------ */

/*
 * dmaplane_ioctl_export_dmabuf - Export a buffer as a dma-buf.
 * @ctx: file context.
 * @arg: userspace pointer to struct dmaplane_export_dmabuf_arg (in/out).
 *
 * Delegates to dmaplane_dmabuf_export which holds buf_lock for the
 * entire operation.  The fd is installed in the process's fd table by
 * dma_buf_fd before copy_to_user.  If copy_to_user fails, the fd
 * exists but userspace doesn't know the number — it will be cleaned
 * up on process exit.  Do NOT try to uninstall the fd.
 *
 * Returns:
 *   0       on success.
 *  -EFAULT  if copy_from_user or copy_to_user fails.
 *  -EINVAL  if buffer not found or not page-backed.
 *  -EBUSY   if buffer already exported.
 *  -ENOMEM  if allocation fails.
 */
static long dmaplane_ioctl_export_dmabuf(struct dmaplane_file_ctx *ctx,
					  unsigned long arg)
{
	struct dmaplane_dev *dev = ctx->dev;
	struct dmaplane_export_dmabuf_arg ea;
	int ret;

	if (copy_from_user(&ea, (void __user *)arg, sizeof(ea)))
		return -EFAULT;

	ret = dmaplane_dmabuf_export(dev, &ea);
	if (ret)
		return ret;

	/*
	 * No undo on copy_to_user failure — the fd is already installed
	 * in the process's fd table by dma_buf_fd.  The fd will be
	 * cleaned up when the process closes it or exits.
	 */
	if (copy_to_user((void __user *)arg, &ea, sizeof(ea)))
		return -EFAULT;

	return 0;
}

/*
 * dmaplane_ioctl_get_dmabuf_stats - Copy dma-buf statistics to userspace.
 * @ctx: file context.
 * @arg: userspace pointer to struct dmaplane_dmabuf_stats (output).
 *
 * Reads device-level atomic counters.  These survive individual
 * export/release cycles — they are lifetime totals.
 *
 * Returns:
 *   0       on success.
 *  -EFAULT  if copy_to_user fails.
 */
static long dmaplane_ioctl_get_dmabuf_stats(struct dmaplane_file_ctx *ctx,
					     unsigned long arg)
{
	struct dmaplane_dev *dev = ctx->dev;
	struct dmaplane_dmabuf_stats ds;

	ds.dmabufs_exported  = atomic64_read(&dev->stats.dmabufs_exported);
	ds.dmabufs_released  = atomic64_read(&dev->stats.dmabufs_released);
	ds.attachments_total = atomic64_read(&dev->stats.dmabuf_attachments);
	ds.detachments_total = atomic64_read(&dev->stats.dmabuf_detachments);
	ds.maps_total        = atomic64_read(&dev->stats.dmabuf_maps);
	ds.unmaps_total      = atomic64_read(&dev->stats.dmabuf_unmaps);

	if (copy_to_user((void __user *)arg, &ds, sizeof(ds)))
		return -EFAULT;

	return 0;
}

/* ------------------------------------------------------------------ */
/* Phase 4: RDMA ioctl handlers                                        */
/*                                                                     */
/* rdma_sem locking strategy:                                          */
/*   Read lock (down_read): MR registration/deregistration and all     */
/*     benchmarks.  Guarantees RDMA context stays alive.  Multiple     */
/*     readers run concurrently.                                       */
/*   Write lock (down_write): RDMA setup and teardown.  Exclusive      */
/*     access — no other RDMA operations can proceed.                  */
/* ------------------------------------------------------------------ */

/*
 * dmaplane_ioctl_setup_rdma — Initialize RDMA subsystem.
 *
 * Creates IB device → PD → CQs → QPs → loopback connection.
 * If copy_to_user fails after successful setup, tears down RDMA to
 * prevent orphaned resources (userspace never got the status handle).
 *
 * Concurrency: acquires rdma_sem write lock (exclusive).
 *
 * Returns:
 *   0       on success.
 *  -EFAULT  if copy_from_user or copy_to_user fails.
 *  -EBUSY   if RDMA is already initialized.
 *  -ENODEV  if the IB device is not found.
 *   Other negative errno from rdma_engine_setup.
 */
static long dmaplane_ioctl_setup_rdma(struct dmaplane_file_ctx *ctx,
				      unsigned long arg)
{
	struct dmaplane_dev *dev = ctx->dev;
	struct dmaplane_rdma_setup p;
	int ret;

	if (copy_from_user(&p, (void __user *)arg, sizeof(p)))
		return -EFAULT;

	/* Null-terminate to prevent buffer overread in find_ib_device */
	p.ib_dev_name[sizeof(p.ib_dev_name) - 1] = '\0';

	down_write(&dev->rdma_sem);
	ret = rdma_engine_setup(dev, &p);
	up_write(&dev->rdma_sem);
	if (ret)
		return ret;

	/* If copy fails, undo: userspace never got the status, so the
	 * RDMA context would be orphaned without teardown. */
	if (copy_to_user((void __user *)arg, &p, sizeof(p))) {
		down_write(&dev->rdma_sem);
		rdma_engine_teardown(dev);
		up_write(&dev->rdma_sem);
		return -EFAULT;
	}
	return 0;
}

/*
 * dmaplane_ioctl_teardown_rdma — Tear down RDMA subsystem.
 *
 * Deregisters ALL in-use MRs before teardown — their DMA mappings
 * reference ib_dev which teardown will release.
 *
 * Concurrency: acquires rdma_sem write lock (exclusive).
 *
 * Returns: 0 always.
 */
static long dmaplane_ioctl_teardown_rdma(struct dmaplane_file_ctx *ctx)
{
	struct dmaplane_dev *dev = ctx->dev;
	unsigned int j;

	down_write(&dev->rdma_sem);

	/* Deregister all MRs before teardown — their SG table DMA mappings
	 * reference the ib_dev which teardown will release. */
	for (j = 0; j < DMAPLANE_MAX_MRS; j++) {
		if (dev->mrs[j].in_use)
			rdma_engine_deregister_mr(dev, dev->mrs[j].id);
	}
	rdma_engine_teardown(dev);

	up_write(&dev->rdma_sem);
	return 0;
}

/*
 * dmaplane_ioctl_register_mr — Register buffer pages as RDMA MR.
 *
 * If copy_to_user fails after successful registration, deregisters the
 * MR to prevent orphaned RDMA resources.
 *
 * Concurrency: acquires rdma_sem read lock (concurrent with benchmarks).
 *
 * Returns:
 *   0       on success.
 *  -EFAULT  if copy_from_user or copy_to_user fails.
 *  -EINVAL  if buffer is coherent (no page array for SG construction).
 *  -ENOENT  if buffer not found.
 *   Other negative errno from rdma_engine_register_mr.
 */
static long dmaplane_ioctl_register_mr(struct dmaplane_file_ctx *ctx,
				       unsigned long arg)
{
	struct dmaplane_dev *dev = ctx->dev;
	struct dmaplane_mr_params p;
	int ret;

	if (copy_from_user(&p, (void __user *)arg, sizeof(p)))
		return -EFAULT;

	down_read(&dev->rdma_sem);
	ret = rdma_engine_register_mr(dev, &p);
	up_read(&dev->rdma_sem);
	if (ret)
		return ret;

	/* If copy fails, undo: userspace never got the mr_id/lkey/rkey,
	 * so the MR would be orphaned without deregistration. */
	if (copy_to_user((void __user *)arg, &p, sizeof(p))) {
		down_read(&dev->rdma_sem);
		rdma_engine_deregister_mr(dev, p.mr_id);
		up_read(&dev->rdma_sem);
		return -EFAULT;
	}
	return 0;
}

/*
 * dmaplane_ioctl_deregister_mr — Deregister an MR by ID.
 *
 * Concurrency: acquires rdma_sem read lock.
 *
 * Returns:
 *   0       on success.
 *  -EFAULT  if copy_from_user fails.
 *  -ENOENT  if MR not found.
 */
static long dmaplane_ioctl_deregister_mr(struct dmaplane_file_ctx *ctx,
					 unsigned long arg)
{
	struct dmaplane_dev *dev = ctx->dev;
	__u32 mr_id;
	int ret;

	if (copy_from_user(&mr_id, (void __user *)arg, sizeof(mr_id)))
		return -EFAULT;

	down_read(&dev->rdma_sem);
	ret = rdma_engine_deregister_mr(dev, mr_id);
	up_read(&dev->rdma_sem);
	return ret;
}

/*
 * dmaplane_ioctl_loopback_test — Run single-message loopback test.
 *
 * Concurrency: acquires rdma_sem read lock (concurrent with other benchmarks).
 *
 * Returns:
 *   0       on success.
 *  -EFAULT  if copy_from_user or copy_to_user fails.
 *   Other negative errno from benchmark_loopback.
 */
static long dmaplane_ioctl_loopback_test(struct dmaplane_file_ctx *ctx,
					 unsigned long arg)
{
	struct dmaplane_dev *dev = ctx->dev;
	struct dmaplane_loopback_params p;
	int ret;

	if (copy_from_user(&p, (void __user *)arg, sizeof(p)))
		return -EFAULT;

	down_read(&dev->rdma_sem);
	ret = benchmark_loopback(dev, &p);
	up_read(&dev->rdma_sem);
	if (ret)
		return ret;

	if (copy_to_user((void __user *)arg, &p, sizeof(p)))
		return -EFAULT;
	return 0;
}

/*
 * dmaplane_ioctl_pingpong_bench — Run ping-pong latency benchmark.
 *
 * Concurrency: acquires rdma_sem read lock.
 *
 * Returns:
 *   0       on success.
 *  -EFAULT  if copy_from_user or copy_to_user fails.
 *   Other negative errno from benchmark_pingpong.
 */
static long dmaplane_ioctl_pingpong_bench(struct dmaplane_file_ctx *ctx,
					  unsigned long arg)
{
	struct dmaplane_dev *dev = ctx->dev;
	struct dmaplane_bench_params p;
	int ret;

	if (copy_from_user(&p, (void __user *)arg, sizeof(p)))
		return -EFAULT;

	down_read(&dev->rdma_sem);
	ret = benchmark_pingpong(dev, &p);
	up_read(&dev->rdma_sem);
	if (ret)
		return ret;

	if (copy_to_user((void __user *)arg, &p, sizeof(p)))
		return -EFAULT;
	return 0;
}

/*
 * dmaplane_ioctl_streaming_bench — Run streaming throughput benchmark.
 *
 * Concurrency: acquires rdma_sem read lock.
 *
 * Returns:
 *   0       on success.
 *  -EFAULT  if copy_from_user or copy_to_user fails.
 *   Other negative errno from benchmark_streaming.
 */
static long dmaplane_ioctl_streaming_bench(struct dmaplane_file_ctx *ctx,
					   unsigned long arg)
{
	struct dmaplane_dev *dev = ctx->dev;
	struct dmaplane_bench_params p;
	int ret;

	if (copy_from_user(&p, (void __user *)arg, sizeof(p)))
		return -EFAULT;

	down_read(&dev->rdma_sem);
	ret = benchmark_streaming(dev, &p);
	up_read(&dev->rdma_sem);
	if (ret)
		return ret;

	if (copy_to_user((void __user *)arg, &p, sizeof(p)))
		return -EFAULT;
	return 0;
}

/*
 * dmaplane_ioctl_get_rdma_stats — Copy RDMA statistics to userspace.
 *
 * No lock needed — all stats are atomic64_t, individually consistent
 * without locking.  The snapshot may be momentarily inconsistent across
 * counters (same caveat as existing stats ioctls).
 *
 * Returns:
 *   0       on success.
 *  -EFAULT  if copy_to_user fails.
 */
static long dmaplane_ioctl_get_rdma_stats(struct dmaplane_file_ctx *ctx,
					  unsigned long arg)
{
	struct dmaplane_dev *dev = ctx->dev;
	struct dmaplane_rdma_stats rs;

	rs.mrs_registered    = atomic64_read(&dev->stats.mrs_registered);
	rs.mrs_deregistered  = atomic64_read(&dev->stats.mrs_deregistered);
	rs.sends_posted      = atomic64_read(&dev->stats.sends_posted);
	rs.recvs_posted      = atomic64_read(&dev->stats.recvs_posted);
	rs.completions_polled = atomic64_read(&dev->stats.completions_polled);
	rs.completion_errors = atomic64_read(&dev->stats.completion_errors);
	rs.bytes_sent        = atomic64_read(&dev->stats.bytes_sent);
	rs.bytes_received    = atomic64_read(&dev->stats.bytes_received);

	if (copy_to_user((void __user *)arg, &rs, sizeof(rs)))
		return -EFAULT;
	return 0;
}

/* ------------------------------------------------------------------ */
/* Phase 5 ioctl handlers: NUMA                                        */
/* ------------------------------------------------------------------ */

/*
 * dmaplane_ioctl_query_numa_topo — Query NUMA topology.
 *
 * Returns node count, per-node CPU counts, memory sizes, and the
 * ACPI SLIT distance matrix.  No lock needed — topology data is
 * maintained by the kernel and doesn't change at runtime.
 *
 * The struct is heap-allocated (~600 bytes) because it exceeds
 * safe kernel stack usage (8 KB on x86).
 *
 * Returns:
 *   0       on success.
 *  -ENOMEM  if kmalloc fails.
 *  -EFAULT  if copy_to_user fails.
 */
static long dmaplane_ioctl_query_numa_topo(struct dmaplane_file_ctx *ctx,
					   unsigned long arg)
{
	struct dmaplane_numa_topo *topo;
	int ret;

	topo = kmalloc(sizeof(*topo), GFP_KERNEL);
	if (!topo)
		return -ENOMEM;

	ret = dmaplane_query_numa_topo(topo);
	if (ret) {
		kfree(topo);
		return ret;
	}

	if (copy_to_user((void __user *)arg, topo, sizeof(*topo))) {
		kfree(topo);
		return -EFAULT;
	}

	kfree(topo);
	return 0;
}

/*
 * dmaplane_ioctl_numa_bench — Run NxN cross-node bandwidth benchmark.
 *
 * Spawns kthreads to measure memcpy throughput between all NUMA node
 * pairs.  No lock needed — the benchmark allocates its own temporary
 * buffers and doesn't touch any dmaplane device state.
 *
 * The struct is heap-allocated (~1100 bytes) because it exceeds
 * safe kernel stack usage.
 *
 * Returns:
 *   0       on success.
 *  -ENOMEM  if kmalloc fails.
 *  -EFAULT  if copy_from_user/copy_to_user fails.
 *  -EINVAL  if buffer_size or iterations are out of range.
 */
static long dmaplane_ioctl_numa_bench(struct dmaplane_file_ctx *ctx,
				      unsigned long arg)
{
	struct dmaplane_numa_bench_params *p;
	int ret;

	p = kmalloc(sizeof(*p), GFP_KERNEL);
	if (!p)
		return -ENOMEM;

	if (copy_from_user(p, (void __user *)arg, sizeof(*p))) {
		kfree(p);
		return -EFAULT;
	}

	ret = dmaplane_numa_bench(p);
	if (ret) {
		kfree(p);
		return ret;
	}

	if (copy_to_user((void __user *)arg, p, sizeof(*p))) {
		kfree(p);
		return -EFAULT;
	}

	kfree(p);
	return 0;
}

/* ------------------------------------------------------------------ */
/* Phase 6 ioctl handlers: Flow control                                */
/* ------------------------------------------------------------------ */

/*
 * dmaplane_ioctl_configure_flow — Configure credit-based flow control.
 *
 * Sets max_credits, high/low watermarks.  No lock needed — configuration
 * fields are only read by the benchmark send loop (single-threaded).
 * Expected to be called before running a sustained benchmark.
 *
 * Returns:
 *   0       on success.
 *  -EFAULT  if copy_from_user/copy_to_user fails.
 *  -EINVAL  if parameters are out of range.
 */
static long dmaplane_ioctl_configure_flow(struct dmaplane_file_ctx *ctx,
					  unsigned long arg)
{
	struct dmaplane_dev *dev = ctx->dev;
	struct dmaplane_flow_params p;
	int ret;

	if (copy_from_user(&p, (void __user *)arg, sizeof(p)))
		return -EFAULT;

	ret = dmaplane_flow_configure(dev, &p);
	if (ret)
		return ret;

	p.status = 0;
	if (copy_to_user((void __user *)arg, &p, sizeof(p)))
		return -EFAULT;
	return 0;
}

/*
 * dmaplane_ioctl_sustained_stream — Run sustained streaming benchmark.
 *
 * Runs for duration_secs wall-clock seconds with credit-based flow
 * control.  Acquires rdma_sem read lock to protect the RDMA context.
 *
 * The struct is heap-allocated for consistency with other large
 * benchmark handlers (sweep at ~800 bytes requires heap allocation;
 * sustained at ~72 bytes doesn't strictly need it but follows the
 * same pattern).
 *
 * Returns:
 *   0       on success.
 *  -ENOMEM  if kmalloc fails.
 *  -EFAULT  if copy_from_user/copy_to_user fails.
 *  -ENODEV  if RDMA is not initialized.
 */
static long dmaplane_ioctl_sustained_stream(struct dmaplane_file_ctx *ctx,
					    unsigned long arg)
{
	struct dmaplane_dev *dev = ctx->dev;
	struct dmaplane_sustained_params *sp;
	int ret;

	sp = kmalloc(sizeof(*sp), GFP_KERNEL);
	if (!sp)
		return -ENOMEM;

	if (copy_from_user(sp, (void __user *)arg, sizeof(*sp))) {
		kfree(sp);
		return -EFAULT;
	}

	down_read(&dev->rdma_sem);
	if (!dev->rdma.initialized) {
		up_read(&dev->rdma_sem);
		kfree(sp);
		return -ENODEV;
	}
	ret = dmaplane_sustained_stream(dev, sp);
	up_read(&dev->rdma_sem);

	if (ret) {
		kfree(sp);
		return ret;
	}

	if (copy_to_user((void __user *)arg, sp, sizeof(*sp))) {
		kfree(sp);
		return -EFAULT;
	}
	kfree(sp);
	return 0;
}

/*
 * dmaplane_ioctl_qdepth_sweep — Run queue depth sweep benchmark.
 *
 * Iterates across queue depths, producing throughput/latency curves.
 * Acquires rdma_sem read lock.
 *
 * The struct is heap-allocated (~800+ bytes — three arrays of 32 x u64
 * exceed safe kernel stack usage).
 *
 * Returns:
 *   0       on success.
 *  -ENOMEM  if kmalloc fails.
 *  -EFAULT  if copy_from_user/copy_to_user fails.
 *  -ENODEV  if RDMA is not initialized.
 */
static long dmaplane_ioctl_qdepth_sweep(struct dmaplane_file_ctx *ctx,
					unsigned long arg)
{
	struct dmaplane_dev *dev = ctx->dev;
	struct dmaplane_sweep_params *sp;
	int ret;

	sp = kmalloc(sizeof(*sp), GFP_KERNEL);
	if (!sp)
		return -ENOMEM;

	if (copy_from_user(sp, (void __user *)arg, sizeof(*sp))) {
		kfree(sp);
		return -EFAULT;
	}

	down_read(&dev->rdma_sem);
	if (!dev->rdma.initialized) {
		up_read(&dev->rdma_sem);
		kfree(sp);
		return -ENODEV;
	}
	ret = dmaplane_qdepth_sweep(dev, sp);
	up_read(&dev->rdma_sem);

	if (ret) {
		kfree(sp);
		return ret;
	}

	if (copy_to_user((void __user *)arg, sp, sizeof(*sp))) {
		kfree(sp);
		return -EFAULT;
	}
	kfree(sp);
	return 0;
}

/*
 * dmaplane_ioctl_get_flow_stats — Copy flow control statistics to userspace.
 *
 * No lock needed — all stats are atomic64_t, individually consistent.
 *
 * Returns:
 *   0       on success.
 *  -EFAULT  if copy_to_user fails.
 */
static long dmaplane_ioctl_get_flow_stats(struct dmaplane_file_ctx *ctx,
					  unsigned long arg)
{
	struct dmaplane_dev *dev = ctx->dev;
	struct dmaplane_flow_stats fs = {
		.credit_stalls         = atomic64_read(&dev->stats.credit_stalls),
		.high_watermark_events = atomic64_read(&dev->stats.high_watermark_events),
		.low_watermark_events  = atomic64_read(&dev->stats.low_watermark_events),
		.cq_overflows          = atomic64_read(&dev->stats.cq_overflows),
		.total_sustained_bytes = atomic64_read(&dev->stats.sustained_bytes),
		.total_sustained_ops   = atomic64_read(&dev->stats.sustained_ops),
	};

	if (copy_to_user((void __user *)arg, &fs, sizeof(fs)))
		return -EFAULT;
	return 0;
}

/*
 * dmaplane_ioctl_get_histogram — Read latency histogram with optional reset.
 *
 * Computes P50/P99/P999 from bucket counts, copies bucket array and
 * summary to userspace.  If reset=1, atomically clears the histogram
 * after reading.
 *
 * No lock needed — all histogram fields are atomic64_t.
 *
 * Returns:
 *   0       on success.
 *  -EFAULT  if copy_from_user or copy_to_user fails.
 */
static long dmaplane_ioctl_get_histogram(struct dmaplane_file_ctx *ctx,
					  unsigned long arg)
{
	struct dmaplane_dev *dev = ctx->dev;
	struct dmaplane_hist_params hp;
	struct dmaplane_hist_summary summary = {0};
	int i;

	if (copy_from_user(&hp, (void __user *)arg, sizeof(hp)))
		return -EFAULT;

	dmaplane_histogram_summarize(&dev->rdma_hist, &summary);

	hp.count = summary.count;
	hp.p50_ns = summary.p50_ns;
	hp.p99_ns = summary.p99_ns;
	hp.p999_ns = summary.p999_ns;
	hp.avg_ns = summary.avg_ns;
	hp.min_ns = summary.min_ns;
	hp.max_ns = summary.max_ns;

	for (i = 0; i < DMAPLANE_HIST_BUCKETS; i++)
		hp.buckets[i] = atomic64_read(&dev->rdma_hist.buckets[i]);

	if (hp.reset)
		dmaplane_histogram_reset(&dev->rdma_hist);

	if (copy_to_user((void __user *)arg, &hp, sizeof(hp)))
		return -EFAULT;
	return 0;
}

/* ------------------------------------------------------------------ */
/* File operations                                                     */
/* ------------------------------------------------------------------ */

/*
 * dmaplane_open - Allocate per-fd context for a new open().
 * @inode: unused (single device).
 * @filp:  file pointer; private_data will be set.
 *
 * Requires CAP_SYS_ADMIN because DMA buffer allocation and mmap of
 * kernel pages are privileged operations — they expose physical memory
 * layout and bypass normal memory protection.
 *
 * Allocates a dmaplane_file_ctx with kzalloc (GFP_KERNEL — process
 * context, may sleep).  The ctx starts with chan == NULL; the user
 * must issue IOCTL_CREATE_CHANNEL before submit/complete are usable.
 *
 * Returns 0 on success, -EPERM without CAP_SYS_ADMIN, -ENOMEM on
 * allocation failure.
 */
static int dmaplane_open(struct inode *inode, struct file *filp)
{
	struct dmaplane_file_ctx *ctx;

	if (!capable(CAP_SYS_ADMIN))
		return -EPERM;

	ctx = kzalloc(sizeof(*ctx), GFP_KERNEL);
	if (!ctx)
		return -ENOMEM;

	ctx->dev = dma_dev;
	ctx->chan = NULL;
	filp->private_data = ctx;

	atomic64_inc(&dma_dev->total_opens);
	pr_debug("device opened\n");
	return 0;
}

/*
 * dmaplane_release - Clean up per-fd state when the file is closed.
 * @inode: unused.
 * @filp:  file pointer carrying the dmaplane_file_ctx.
 *
 * If a channel was created on this fd, acquires dev_mutex and calls
 * dmaplane_channel_destroy to stop the worker and free the slot.
 * Then frees the file context.  This ensures that closing the fd
 * (including process exit without explicit close) releases all kernel
 * resources — no explicit "destroy channel" ioctl is needed.
 *
 * Returns 0 always.
 */
static int dmaplane_release(struct inode *inode, struct file *filp)
{
	struct dmaplane_file_ctx *ctx = filp->private_data;

	if (ctx->chan) {
		mutex_lock(&ctx->dev->dev_mutex);
		dmaplane_channel_destroy(ctx->chan);
		mutex_unlock(&ctx->dev->dev_mutex);
	}

	kfree(ctx);
	atomic64_inc(&dma_dev->total_closes);
	pr_debug("device closed\n");
	return 0;
}

/*
 * dmaplane_ioctl - Top-level ioctl dispatcher.
 * @filp: file pointer carrying the dmaplane_file_ctx.
 * @cmd:  ioctl command number (DMAPLANE_IOCTL_*).
 * @arg:  userspace pointer argument.
 *
 * Routes to the specific handler based on cmd.  Returns -ENOTTY for
 * unrecognised commands (standard convention for bad ioctl numbers).
 */
static long dmaplane_ioctl(struct file *filp, unsigned int cmd,
			    unsigned long arg)
{
	struct dmaplane_file_ctx *ctx = filp->private_data;
	long ret;

	switch (cmd) {
	/* Phase 1: Channel operations */
	case DMAPLANE_IOCTL_CREATE_CHANNEL:
		return dmaplane_ioctl_create_channel(ctx, arg);
	case DMAPLANE_IOCTL_SUBMIT:
		return dmaplane_ioctl_submit(ctx, arg);
	case DMAPLANE_IOCTL_COMPLETE:
		return dmaplane_ioctl_complete(ctx, arg);
	case DMAPLANE_IOCTL_GET_STATS:
		return dmaplane_ioctl_get_stats(ctx, arg);

	/* Phase 2: Buffer management */
	case DMAPLANE_IOCTL_CREATE_BUFFER:
		return dmaplane_ioctl_create_buffer(ctx, arg);
	case DMAPLANE_IOCTL_DESTROY_BUFFER:
		return dmaplane_ioctl_destroy_buffer(ctx, arg);
	case DMAPLANE_IOCTL_GET_MMAP_INFO:
		return dmaplane_ioctl_get_mmap_info(ctx, arg);
	case DMAPLANE_IOCTL_GET_BUF_STATS:
		return dmaplane_ioctl_get_buf_stats(ctx, arg);

	/* Phase 3: dma-buf export */
	case DMAPLANE_IOCTL_EXPORT_DMABUF:
		return dmaplane_ioctl_export_dmabuf(ctx, arg);
	case DMAPLANE_IOCTL_GET_DMABUF_STATS:
		return dmaplane_ioctl_get_dmabuf_stats(ctx, arg);

	/* Phase 4: RDMA */
	case DMAPLANE_IOCTL_SETUP_RDMA:
		return dmaplane_ioctl_setup_rdma(ctx, arg);
	case DMAPLANE_IOCTL_TEARDOWN_RDMA:
		return dmaplane_ioctl_teardown_rdma(ctx);
	case DMAPLANE_IOCTL_REGISTER_MR:
		return dmaplane_ioctl_register_mr(ctx, arg);
	case DMAPLANE_IOCTL_DEREGISTER_MR:
		return dmaplane_ioctl_deregister_mr(ctx, arg);
	case DMAPLANE_IOCTL_LOOPBACK_TEST:
		return dmaplane_ioctl_loopback_test(ctx, arg);
	case DMAPLANE_IOCTL_PINGPONG_BENCH:
		return dmaplane_ioctl_pingpong_bench(ctx, arg);
	case DMAPLANE_IOCTL_STREAMING_BENCH:
		return dmaplane_ioctl_streaming_bench(ctx, arg);
	case DMAPLANE_IOCTL_GET_RDMA_STATS:
		return dmaplane_ioctl_get_rdma_stats(ctx, arg);

	/* Phase 5: NUMA */
	case DMAPLANE_IOCTL_QUERY_NUMA_TOPO:
		return dmaplane_ioctl_query_numa_topo(ctx, arg);
	case DMAPLANE_IOCTL_NUMA_BENCH:
		return dmaplane_ioctl_numa_bench(ctx, arg);

	/* Phase 6: Flow control */
	case DMAPLANE_IOCTL_CONFIGURE_FLOW:
		return dmaplane_ioctl_configure_flow(ctx, arg);
	case DMAPLANE_IOCTL_SUSTAINED_STREAM:
		return dmaplane_ioctl_sustained_stream(ctx, arg);
	case DMAPLANE_IOCTL_QDEPTH_SWEEP:
		return dmaplane_ioctl_qdepth_sweep(ctx, arg);
	case DMAPLANE_IOCTL_GET_FLOW_STATS:
		return dmaplane_ioctl_get_flow_stats(ctx, arg);

	/* Phase 7: Instrumentation */
	case DMAPLANE_IOCTL_GET_HISTOGRAM:
		return dmaplane_ioctl_get_histogram(ctx, arg);

	/* Phase 8: GPU P2P */
#ifdef CONFIG_DMAPLANE_GPU
	case DMAPLANE_IOCTL_GPU_PIN: {
		struct dmaplane_gpu_pin_params p;

		if (copy_from_user(&p, (void __user *)arg, sizeof(p)))
			return -EFAULT;
		ret = dmaplane_gpu_pin(dma_dev, p.gpu_va, p.size,
				       &p.handle, &p.gpu_numa_node,
				       &p.num_pages, &p.bar1_consumed);
		if (ret)
			return ret;
		if (copy_to_user((void __user *)arg, &p, sizeof(p))) {
			dmaplane_gpu_unpin(dma_dev, p.handle);
			return -EFAULT;
		}
		return 0;
	}
	case DMAPLANE_IOCTL_GPU_UNPIN: {
		struct dmaplane_gpu_unpin_params p;

		if (copy_from_user(&p, (void __user *)arg, sizeof(p)))
			return -EFAULT;
		return dmaplane_gpu_unpin(dma_dev, p.handle);
	}
	case DMAPLANE_IOCTL_GPU_DMA_TO_HOST: {
		struct dmaplane_gpu_dma_params p;
		long ret;

		if (copy_from_user(&p, (void __user *)arg, sizeof(p)))
			return -EFAULT;
		ret = dmaplane_gpu_dma_to_host(dma_dev, p.gpu_handle,
					       p.host_handle, p.offset,
					       p.size, &p.elapsed_ns);
		if (ret)
			return ret;
		if (p.elapsed_ns > 0)
			p.throughput_mbps = div64_u64(p.size * 1000ULL,
						      p.elapsed_ns);
		else
			p.throughput_mbps = 0;
		if (copy_to_user((void __user *)arg, &p, sizeof(p)))
			return -EFAULT;
		return 0;
	}
	case DMAPLANE_IOCTL_GPU_DMA_FROM_HOST: {
		struct dmaplane_gpu_dma_params p;
		long ret;

		if (copy_from_user(&p, (void __user *)arg, sizeof(p)))
			return -EFAULT;
		ret = dmaplane_gpu_dma_from_host(dma_dev, p.host_handle,
						 p.gpu_handle, p.offset,
						 p.size, &p.elapsed_ns);
		if (ret)
			return ret;
		if (p.elapsed_ns > 0)
			p.throughput_mbps = div64_u64(p.size * 1000ULL,
						      p.elapsed_ns);
		else
			p.throughput_mbps = 0;
		if (copy_to_user((void __user *)arg, &p, sizeof(p)))
			return -EFAULT;
		return 0;
	}
	case DMAPLANE_IOCTL_GPU_BENCH: {
		struct dmaplane_gpu_bench_params p;

		if (copy_from_user(&p, (void __user *)arg, sizeof(p)))
			return -EFAULT;
		ret = dmaplane_gpu_benchmark(dma_dev, p.gpu_handle,
					     p.host_handle, p.iterations,
					     &p.h2g_bandwidth_mbps,
					     &p.g2h_bandwidth_mbps);
		if (ret)
			return ret;
		if (copy_to_user((void __user *)arg, &p, sizeof(p)))
			return -EFAULT;
		return 0;
	}
	case DMAPLANE_IOCTL_GPU_REGISTER_MR: {
		struct dmaplane_gpu_mr_params p;

		if (copy_from_user(&p, (void __user *)arg, sizeof(p)))
			return -EFAULT;
		down_read(&dma_dev->rdma_sem);
		ret = dmaplane_gpu_register_mr(dma_dev, &p);
		up_read(&dma_dev->rdma_sem);
		if (ret)
			return ret;
		if (copy_to_user((void __user *)arg, &p, sizeof(p))) {
			rdma_engine_deregister_mr(dma_dev, p.mr_id);
			return -EFAULT;
		}
		return 0;
	}
	case DMAPLANE_IOCTL_GPU_LOOPBACK: {
		struct dmaplane_gpu_loopback_params p;

		if (copy_from_user(&p, (void __user *)arg, sizeof(p)))
			return -EFAULT;
		down_read(&dma_dev->rdma_sem);
		ret = benchmark_gpu_loopback(dma_dev, &p);
		up_read(&dma_dev->rdma_sem);
		if (ret)
			return ret;
		if (copy_to_user((void __user *)arg, &p, sizeof(p)))
			return -EFAULT;
		return 0;
	}
#endif /* CONFIG_DMAPLANE_GPU */

#ifndef CONFIG_DMAPLANE_GPU
	/* GPU support not compiled — return -ENODEV (not -ENOTTY)
	 * so userspace can distinguish "no GPU" from "bad ioctl". */
	case DMAPLANE_IOCTL_GPU_PIN:
	case DMAPLANE_IOCTL_GPU_UNPIN:
	case DMAPLANE_IOCTL_GPU_DMA_TO_HOST:
	case DMAPLANE_IOCTL_GPU_DMA_FROM_HOST:
	case DMAPLANE_IOCTL_GPU_BENCH:
	case DMAPLANE_IOCTL_GPU_REGISTER_MR:
	case DMAPLANE_IOCTL_GPU_LOOPBACK:
		return -ENODEV;
#endif

	case DMAPLANE_IOCTL_GET_GPU_STATS: {
		struct dmaplane_gpu_stats gs;

		gs.pins_total = atomic64_read(&dma_dev->gpu_stats.pins_total);
		gs.unpins_total = atomic64_read(&dma_dev->gpu_stats.unpins_total);
		gs.callbacks_fired = atomic64_read(&dma_dev->gpu_stats.callbacks_fired);
		gs.dma_h2g_bytes = atomic64_read(&dma_dev->gpu_stats.dma_h2g_bytes);
		gs.dma_g2h_bytes = atomic64_read(&dma_dev->gpu_stats.dma_g2h_bytes);
		gs.gpu_mrs_registered = atomic64_read(&dma_dev->gpu_stats.gpu_mrs_registered);
		if (copy_to_user((void __user *)arg, &gs, sizeof(gs)))
			return -EFAULT;
		return 0;
	}

	/* Phase 8: Peer RDMA (cross-machine) */
	case DMAPLANE_IOCTL_RDMA_INIT_PEER: {
		struct dmaplane_rdma_peer_info pi;

		down_read(&dma_dev->rdma_sem);
		ret = rdma_engine_init_peer(dma_dev, &pi);
		up_read(&dma_dev->rdma_sem);
		if (ret)
			return ret;
		if (copy_to_user((void __user *)arg, &pi, sizeof(pi)))
			return -EFAULT;
		return 0;
	}
	case DMAPLANE_IOCTL_RDMA_CONNECT_PEER: {
		struct dmaplane_rdma_peer_info pi;

		if (copy_from_user(&pi, (void __user *)arg, sizeof(pi)))
			return -EFAULT;
		down_read(&dma_dev->rdma_sem);
		ret = rdma_engine_connect_peer(dma_dev, &pi);
		up_read(&dma_dev->rdma_sem);
		if (ret)
			return ret;
		if (copy_to_user((void __user *)arg, &pi, sizeof(pi)))
			return -EFAULT;
		return 0;
	}
	case DMAPLANE_IOCTL_RDMA_REMOTE_SEND: {
		struct dmaplane_rdma_remote_xfer_params xp;

		if (copy_from_user(&xp, (void __user *)arg, sizeof(xp)))
			return -EFAULT;
		down_read(&dma_dev->rdma_sem);
		ret = rdma_engine_remote_send(dma_dev, &xp);
		up_read(&dma_dev->rdma_sem);
		if (ret)
			return ret;
		if (copy_to_user((void __user *)arg, &xp, sizeof(xp)))
			return -EFAULT;
		return 0;
	}
	case DMAPLANE_IOCTL_RDMA_REMOTE_RECV: {
		struct dmaplane_rdma_remote_xfer_params xp;

		if (copy_from_user(&xp, (void __user *)arg, sizeof(xp)))
			return -EFAULT;
		down_read(&dma_dev->rdma_sem);
		ret = rdma_engine_remote_recv(dma_dev, &xp);
		up_read(&dma_dev->rdma_sem);
		if (ret)
			return ret;
		if (copy_to_user((void __user *)arg, &xp, sizeof(xp)))
			return -EFAULT;
		return 0;
	}
	case DMAPLANE_IOCTL_RDMA_DESTROY_PEER:
		down_write(&dma_dev->rdma_sem);
		rdma_engine_teardown_peer(dma_dev);
		up_write(&dma_dev->rdma_sem);
		return 0;

	/* ── Phase 9: RDMA WRITE with Immediate ── */

	/*
	 * RDMA_WRITE_IMM — Post IB_WR_RDMA_WRITE_WITH_IMM.
	 *
	 * Writes data from a local MR to a remote MR address and delivers a
	 * 32-bit immediate through the receiver's CQ.  The primary verb for
	 * KVCache chunked transfer — imm_data encodes (layer, chunk).
	 */
	case DMAPLANE_IOCTL_RDMA_WRITE_IMM: {
		struct dmaplane_write_imm_params wp;

		if (copy_from_user(&wp, (void __user *)arg, sizeof(wp)))
			return -EFAULT;
		down_read(&dma_dev->rdma_sem);
		ret = rdma_engine_write_imm(dma_dev,
					     wp.local_mr_id,
					     wp.local_offset,
					     wp.remote_addr,
					     wp.remote_rkey,
					     wp.length,
					     wp.imm_data,
					     wp.use_peer_qp,
					     &wp.elapsed_ns);
		up_read(&dma_dev->rdma_sem);
		wp.status = ret ? 1 : 0;
		if (copy_to_user((void __user *)arg, &wp, sizeof(wp)))
			return -EFAULT;
		return ret < 0 ? ret : 0;
	}

	/*
	 * RDMA_POST_RECV — Post a recv WR to absorb one WRITEIMM.
	 *
	 * Must be called before the sender's write_imm — otherwise the
	 * receiving QP hits RNR (Receiver Not Ready).
	 */
	case DMAPLANE_IOCTL_RDMA_POST_RECV: {
		struct dmaplane_post_recv_params rp;

		if (copy_from_user(&rp, (void __user *)arg, sizeof(rp)))
			return -EFAULT;
		down_read(&dma_dev->rdma_sem);
		ret = rdma_engine_writeimm_post_recv(dma_dev,
						      rp.mr_id,
						      rp.size,
						      rp.use_peer_qp);
		up_read(&dma_dev->rdma_sem);
		rp.status = ret ? 1 : 0;
		if (copy_to_user((void __user *)arg, &rp, sizeof(rp)))
			return -EFAULT;
		return ret < 0 ? ret : 0;
	}

	/*
	 * RDMA_POLL_RECV — Poll recv CQ for a WRITEIMM completion.
	 *
	 * Returns the 32-bit immediate and payload size.  The ioctl returns 0
	 * even on timeout; the caller checks status (0=success, nonzero=timeout).
	 */
	case DMAPLANE_IOCTL_RDMA_POLL_RECV: {
		struct dmaplane_poll_recv_params pp;

		if (copy_from_user(&pp, (void __user *)arg, sizeof(pp)))
			return -EFAULT;
		down_read(&dma_dev->rdma_sem);
		ret = rdma_engine_writeimm_poll_recv(dma_dev,
						      pp.use_peer_qp,
						      pp.timeout_ms,
						      &pp.status,
						      &pp.wc_flags,
						      &pp.imm_data,
						      &pp.byte_len,
						      &pp.elapsed_ns);
		up_read(&dma_dev->rdma_sem);
		if (copy_to_user((void __user *)arg, &pp, sizeof(pp)))
			return -EFAULT;
		return ret < 0 ? ret : 0;
	}

	default:
		return -ENOTTY;
	}
}

static const struct file_operations dmaplane_fops = {
	.owner		= THIS_MODULE,
	.open		= dmaplane_open,
	.release	= dmaplane_release,
	.unlocked_ioctl	= dmaplane_ioctl,
	.mmap		= dmaplane_mmap,
};

/* ------------------------------------------------------------------ */
/* Module init/exit                                                    */
/* ------------------------------------------------------------------ */

/*
 * dmaplane_init - Module entry point.
 *
 * Init sequence:
 *  1. kzalloc the singleton dmaplane_dev.
 *  2. platform_device_alloc + platform_device_add + dma_set_mask —
 *     creates a DMA-capable device for buffer allocation.
 *  3. alloc_chrdev_region for a dynamic major number.
 *  4. cdev_init + cdev_add to register the file_operations.
 *  5. class_create so udev can see the device class.
 *  6. device_create to create the /dev/dmaplane node.
 *
 * The platform device is created BEFORE the char device because the
 * DMA API (dma_alloc_coherent, dma_map_sg) requires a device that is
 * registered with a bus and has a valid DMA mask.  device_create alone
 * is NOT DMA-capable — it only produces a device node for udev.
 *
 * On any failure, goto-based cleanup tears down in reverse order.
 */
static int __init dmaplane_init(void)
{
	int ret;

	dma_dev = kzalloc(sizeof(*dma_dev), GFP_KERNEL);
	if (!dma_dev)
		return -ENOMEM;

	mutex_init(&dma_dev->dev_mutex);
	mutex_init(&dma_dev->buf_lock);
	dma_dev->next_buf_id = 1;	/* Skip 0 — reserved as "no buffer" */

	/* Device-level counters — kzalloc zeroes them, but be explicit */
	atomic_set(&dma_dev->active_channels, 0);
	atomic64_set(&dma_dev->total_opens, 0);
	atomic64_set(&dma_dev->total_closes, 0);
	atomic64_set(&dma_dev->total_channels_created, 0);
	atomic64_set(&dma_dev->total_channels_destroyed, 0);
	atomic64_set(&dma_dev->stats.buffers_created, 0);
	atomic64_set(&dma_dev->stats.buffers_destroyed, 0);
	atomic64_set(&dma_dev->stats.dmabufs_exported, 0);
	atomic64_set(&dma_dev->stats.dmabufs_released, 0);
	atomic64_set(&dma_dev->stats.dmabuf_attachments, 0);
	atomic64_set(&dma_dev->stats.dmabuf_detachments, 0);
	atomic64_set(&dma_dev->stats.dmabuf_maps, 0);
	atomic64_set(&dma_dev->stats.dmabuf_unmaps, 0);

	/* Phase 4: RDMA subsystem init */
	init_rwsem(&dma_dev->rdma_sem);
	mutex_init(&dma_dev->mr_lock);
	dma_dev->next_mr_id = 1;	/* Skip 0 — reserved as "no MR" */
	atomic64_set(&dma_dev->stats.mrs_registered, 0);
	atomic64_set(&dma_dev->stats.mrs_deregistered, 0);
	atomic64_set(&dma_dev->stats.sends_posted, 0);
	atomic64_set(&dma_dev->stats.recvs_posted, 0);
	atomic64_set(&dma_dev->stats.completions_polled, 0);
	atomic64_set(&dma_dev->stats.completion_errors, 0);
	atomic64_set(&dma_dev->stats.bytes_sent, 0);
	atomic64_set(&dma_dev->stats.bytes_received, 0);

	/* Phase 5: NUMA allocation tracking */
	atomic64_set(&dma_dev->stats.numa_local_allocs, 0);
	atomic64_set(&dma_dev->stats.numa_remote_allocs, 0);
	atomic64_set(&dma_dev->stats.numa_anon_allocs, 0);

	/* Phase 6: Flow control */
	atomic_set(&dma_dev->flow.credits, 0);
	dma_dev->flow.max_credits = 0;
	dma_dev->flow.high_watermark = 0;
	dma_dev->flow.low_watermark = 0;
	dma_dev->flow.configured = false;
	dma_dev->flow.paused = false;
	atomic64_set(&dma_dev->stats.credit_stalls, 0);
	atomic64_set(&dma_dev->stats.high_watermark_events, 0);
	atomic64_set(&dma_dev->stats.low_watermark_events, 0);
	atomic64_set(&dma_dev->stats.cq_overflows, 0);
	atomic64_set(&dma_dev->stats.sustained_bytes, 0);
	atomic64_set(&dma_dev->stats.sustained_ops, 0);

	/* Phase 7: Histogram and debugfs */
	dmaplane_histogram_init(&dma_dev->rdma_hist);

	/* Phase 7: Cacheline alignment verification — compile-time assertion.
	 * Verifies that the ____cacheline_aligned_in_smp annotations on
	 * ring head/tail actually produce separate cache lines. */
	BUILD_BUG_ON(offsetof(struct dmaplane_ring, head) % SMP_CACHE_BYTES != 0);
	BUILD_BUG_ON(offsetof(struct dmaplane_ring, tail) % SMP_CACHE_BYTES != 0);

	/* Phase 8: GPU stats init (always, even without GPU support) */
	atomic64_set(&dma_dev->gpu_stats.pins_total, 0);
	atomic64_set(&dma_dev->gpu_stats.unpins_total, 0);
	atomic64_set(&dma_dev->gpu_stats.callbacks_fired, 0);
	atomic64_set(&dma_dev->gpu_stats.dma_h2g_bytes, 0);
	atomic64_set(&dma_dev->gpu_stats.dma_g2h_bytes, 0);
	atomic64_set(&dma_dev->gpu_stats.gpu_mrs_registered, 0);

	/*
	 * Create platform device for DMA operations.
	 *
	 * The character device (/dev/dmaplane) handles the userspace interface
	 * (open, ioctl, mmap).  But the DMA API requires a device that is
	 * registered with a bus and has a valid DMA mask.  platform_device
	 * provides this — it represents a virtual hardware device that the
	 * DMA subsystem can program IOMMU entries for.
	 *
	 * All DMA operations (dma_alloc_coherent, dma_map_sg, dma_mmap_coherent)
	 * use &dma_dev->pdev->dev as the device parameter.
	 */
	dma_dev->pdev = platform_device_alloc("dmaplane_dma", -1);
	if (!dma_dev->pdev) {
		pr_err("platform_device_alloc failed\n");
		ret = -ENOMEM;
		goto err_free_dev;
	}

	ret = platform_device_add(dma_dev->pdev);
	if (ret) {
		pr_err("platform_device_add failed: %d\n", ret);
		goto err_put_pdev;
	}

	/*
	 * Set DMA mask.  Try 64-bit first (allows DMA to all physical memory),
	 * fall back to 32-bit (4 GB limit) if the platform doesn't support 64-bit.
	 * Without a DMA mask, dma_alloc_coherent will fail.
	 */
	ret = dma_set_mask_and_coherent(&dma_dev->pdev->dev, DMA_BIT_MASK(64));
	if (ret) {
		ret = dma_set_mask_and_coherent(&dma_dev->pdev->dev,
						DMA_BIT_MASK(32));
		if (ret) {
			pr_err("failed to set DMA mask\n");
			goto err_unreg_pdev;
		}
	}

	/* Dynamic major number — kernel assigns unused number */
	ret = alloc_chrdev_region(&dma_dev->devno, 0, 1, DMAPLANE_NAME);
	if (ret < 0) {
		pr_err("alloc_chrdev_region failed: %d\n", ret);
		goto err_unreg_pdev;
	}

	/* Character device — register our file_operations */
	cdev_init(&dma_dev->cdev, &dmaplane_fops);
	dma_dev->cdev.owner = THIS_MODULE;
	ret = cdev_add(&dma_dev->cdev, dma_dev->devno, 1);
	if (ret < 0) {
		pr_err("cdev_add failed: %d\n", ret);
		goto err_unreg_region;
	}

	/* Device class — lets udev see us */
	dma_dev->class = class_create(DMAPLANE_NAME);
	if (IS_ERR(dma_dev->class)) {
		ret = PTR_ERR(dma_dev->class);
		pr_err("class_create failed: %d\n", ret);
		goto err_cdev_del;
	}

	/* Device node: /dev/dmaplane */
	dma_dev->device = device_create(dma_dev->class, NULL,
					dma_dev->devno, NULL, DMAPLANE_NAME);
	if (IS_ERR(dma_dev->device)) {
		ret = PTR_ERR(dma_dev->device);
		pr_err("device_create failed: %d\n", ret);
		goto err_class_destroy;
	}

	/* Phase 7: debugfs — non-fatal on failure */
	dmaplane_debugfs_init(dma_dev);

#ifdef CONFIG_DMAPLANE_GPU
	/* Phase 8: GPU subsystem init — allocate tracking array and
	 * resolve NVIDIA P2P symbols.  Non-fatal on failure — module
	 * operates without GPU if nvidia.ko is not loaded. */
	dma_dev->gpu_buffers = kcalloc(DMAPLANE_MAX_GPU_BUFFERS,
				       sizeof(struct dmaplane_gpu_buffer),
				       GFP_KERNEL);
	if (!dma_dev->gpu_buffers) {
		pr_warn("GPU buffer array allocation failed — GPU support disabled\n");
	} else {
		mutex_init(&dma_dev->gpu_buf_lock);
		dma_dev->next_gpu_buf_id = 1;
		ret = dmaplane_gpu_init();
		if (ret) {
			pr_info("GPU P2P symbols not available — GPU support disabled\n");
			kfree(dma_dev->gpu_buffers);
			dma_dev->gpu_buffers = NULL;
		}
	}
#endif

	pr_info("module loaded (major %d)\n", MAJOR(dma_dev->devno));

	/* Run dma-buf self-test if requested via module parameter */
	if (test_dmabuf) {
		ret = dmaplane_dmabuf_selftest(dma_dev);
		if (ret)
			pr_warn("dmabuf selftest failed: %d\n", ret);
	}

	return 0;

err_class_destroy:
	class_destroy(dma_dev->class);
err_cdev_del:
	cdev_del(&dma_dev->cdev);
err_unreg_region:
	unregister_chrdev_region(dma_dev->devno, 1);
err_unreg_pdev:
	platform_device_unregister(dma_dev->pdev);
	goto err_free_dev;	/* skip put — unregister already put */
err_put_pdev:
	/* platform_device_put for devices allocated but never added */
	platform_device_put(dma_dev->pdev);
err_free_dev:
	kfree(dma_dev);
	return ret;
}

/*
 * dmaplane_exit - Module cleanup.
 *
 * Teardown ordering is critical:
 *  1. Stop any remaining active channels (kthreads must exit first).
 *  2. Tear down char device (prevents new opens/ioctls from racing
 *     with buffer cleanup).
 *  3. Destroy remaining buffers (dma_free_coherent needs the platform
 *     device to be alive).
 *  4. Unregister platform device.
 *  5. Free device context.
 */
static void __exit dmaplane_exit(void)
{
	int i;

	/* 1. Stop any remaining active channels */
	mutex_lock(&dma_dev->dev_mutex);
	for (i = 0; i < DMAPLANE_MAX_CHANNELS; i++)
		dmaplane_channel_destroy(&dma_dev->channels[i]);
	mutex_unlock(&dma_dev->dev_mutex);

	/* Lifetime summary — visible in dmesg after rmmod */
	pr_info("lifetime: %lld opens, %lld closes, %lld channels created, %lld destroyed, %lld buffers created, %lld destroyed\n",
		atomic64_read(&dma_dev->total_opens),
		atomic64_read(&dma_dev->total_closes),
		atomic64_read(&dma_dev->total_channels_created),
		atomic64_read(&dma_dev->total_channels_destroyed),
		atomic64_read(&dma_dev->stats.buffers_created),
		atomic64_read(&dma_dev->stats.buffers_destroyed));

	/* Phase 7: Remove debugfs before device teardown — debugfs show
	 * functions reference dmaplane_dev fields. */
	dmaplane_debugfs_exit();

	/* 2. Tear down char device — prevents new ioctls/opens */
	device_destroy(dma_dev->class, dma_dev->devno);
	class_destroy(dma_dev->class);
	cdev_del(&dma_dev->cdev);
	unregister_chrdev_region(dma_dev->devno, 1);

	/*
	 * 3. RDMA teardown — after char device is gone (no new ioctls),
	 * take write lock to synchronize with any in-flight operations
	 * that started before cdev_del returned.
	 *
	 * Deregister MRs ONLY if RDMA is still up.  If userspace already
	 * called IOCTL_TEARDOWN_RDMA, the PD is gone and attempting to
	 * deregister MRs would use-after-free on the ib_mr.
	 */
	down_write(&dma_dev->rdma_sem);
	if (dma_dev->rdma.initialized) {
		for (i = 0; i < DMAPLANE_MAX_MRS; i++) {
			if (dma_dev->mrs[i].in_use)
				rdma_engine_deregister_mr(dma_dev,
							  dma_dev->mrs[i].id);
		}
		rdma_engine_teardown(dma_dev);
	}
	/* Mark all MR slots as unused regardless (they're invalid now) */
	for (i = 0; i < DMAPLANE_MAX_MRS; i++)
		dma_dev->mrs[i].in_use = false;
	up_write(&dma_dev->rdma_sem);

#ifdef CONFIG_DMAPLANE_GPU
	/* Phase 8: GPU cleanup — unpin all remaining buffers, then
	 * release NVIDIA symbol references.  Must unpin BEFORE
	 * symbol_put — otherwise put_pages pointer is invalid. */
	if (dma_dev->gpu_buffers) {
		for (i = 0; i < DMAPLANE_MAX_GPU_BUFFERS; i++) {
			if (dma_dev->gpu_buffers[i].in_use)
				dmaplane_gpu_unpin(dma_dev,
						   dma_dev->gpu_buffers[i].id);
		}
		kfree(dma_dev->gpu_buffers);
	}
	dmaplane_gpu_exit();
#endif

	/* 4. Free remaining buffers — warn on leaked exports/mmaps */
	for (i = 0; i < DMAPLANE_MAX_BUFFERS; i++) {
		if (dma_dev->buffers[i].in_use) {
			if (dma_dev->buffers[i].dmabuf_exported)
				pr_warn("buffer %u still has active dma-buf export at exit\n",
					dma_dev->buffers[i].id);
			if (atomic_read(&dma_dev->buffers[i].mmap_count) > 0)
				pr_warn("buffer %u still has %d active mmaps at exit\n",
					dma_dev->buffers[i].id,
					atomic_read(&dma_dev->buffers[i].mmap_count));
			dmabuf_rdma_destroy_buffer(dma_dev, dma_dev->buffers[i].id);
		}
	}

	/* 5. Unregister platform device */
	platform_device_unregister(dma_dev->pdev);

	/* 6. Free device context */
	kfree(dma_dev);

	pr_info("module unloaded\n");
}

module_init(dmaplane_init);
module_exit(dmaplane_exit);
