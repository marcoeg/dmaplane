// SPDX-License-Identifier: GPL-2.0
/*
 * dmaplane_debugfs.c — debugfs interface for live driver inspection
 * Copyright (c) 2026 Graziano Labs Corp.
 *
 * Creates /sys/kernel/debug/dmaplane/ with 6 read-only files:
 *   stats     — all device-level atomic counters
 *   buffers   — list of active DMA buffers
 *   rdma      — RDMA context state and MR list
 *   flow      — flow control configuration and state
 *   histogram — latency histogram with percentiles
 *   gpu       — GPU P2P state and statistics (Phase 8)
 *
 * All files use seq_file for output.  debugfs is a debugging aid
 * with no ABI stability guarantees — format may change between
 * phases without notice.
 *
 * debugfs failure (CONFIG_DEBUG_FS=n, not mounted) is never fatal.
 * The module loads and operates normally without it.
 *
 * Lifecycle:
 *   dmaplane_debugfs_init() — called from module init, after all
 *     subsystems are initialized.  Creates directory and files.
 *   dmaplane_debugfs_exit() — called from module exit, BEFORE
 *     device teardown.  Removes all files atomically via
 *     debugfs_remove_recursive().
 */

#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt

#include <linux/debugfs.h>
#include <linux/seq_file.h>
#include "dmaplane.h"
#include "dmaplane_histogram.h"
#ifdef CONFIG_DMAPLANE_GPU
#include "gpu_p2p.h"
#endif

static struct dentry *dmaplane_debugfs_root;

/* ── stats: all device-level atomic counters ──────────────── */

static int dmaplane_stats_show(struct seq_file *s, void *unused)
{
	struct dmaplane_dev *dev = s->private;

	seq_puts(s, "=== dmaplane device stats ===\n");

	seq_printf(s, "channels:       active=%d created=%lld destroyed=%lld\n",
		   atomic_read(&dev->active_channels),
		   atomic64_read(&dev->total_channels_created),
		   atomic64_read(&dev->total_channels_destroyed));

	seq_printf(s, "opens:          total=%lld closes=%lld\n",
		   atomic64_read(&dev->total_opens),
		   atomic64_read(&dev->total_closes));

	seq_printf(s, "buffers:        created=%lld destroyed=%lld\n",
		   atomic64_read(&dev->stats.buffers_created),
		   atomic64_read(&dev->stats.buffers_destroyed));

	seq_printf(s, "  numa:         local=%lld remote=%lld anon=%lld\n",
		   atomic64_read(&dev->stats.numa_local_allocs),
		   atomic64_read(&dev->stats.numa_remote_allocs),
		   atomic64_read(&dev->stats.numa_anon_allocs));

	seq_printf(s, "dmabuf:         exported=%lld released=%lld attach=%lld detach=%lld map=%lld unmap=%lld\n",
		   atomic64_read(&dev->stats.dmabufs_exported),
		   atomic64_read(&dev->stats.dmabufs_released),
		   atomic64_read(&dev->stats.dmabuf_attachments),
		   atomic64_read(&dev->stats.dmabuf_detachments),
		   atomic64_read(&dev->stats.dmabuf_maps),
		   atomic64_read(&dev->stats.dmabuf_unmaps));

	seq_printf(s, "rdma:           mrs_reg=%lld mrs_dereg=%lld sends=%lld recvs=%lld completions=%lld errors=%lld\n",
		   atomic64_read(&dev->stats.mrs_registered),
		   atomic64_read(&dev->stats.mrs_deregistered),
		   atomic64_read(&dev->stats.sends_posted),
		   atomic64_read(&dev->stats.recvs_posted),
		   atomic64_read(&dev->stats.completions_polled),
		   atomic64_read(&dev->stats.completion_errors));

	seq_printf(s, "  bytes:        sent=%lld received=%lld\n",
		   atomic64_read(&dev->stats.bytes_sent),
		   atomic64_read(&dev->stats.bytes_received));

	seq_printf(s, "flow:           stalls=%lld high_wm=%lld low_wm=%lld overflows=%lld\n",
		   atomic64_read(&dev->stats.credit_stalls),
		   atomic64_read(&dev->stats.high_watermark_events),
		   atomic64_read(&dev->stats.low_watermark_events),
		   atomic64_read(&dev->stats.cq_overflows));

	seq_printf(s, "  sustained:    bytes=%lld ops=%lld\n",
		   atomic64_read(&dev->stats.sustained_bytes),
		   atomic64_read(&dev->stats.sustained_ops));

	return 0;
}
DEFINE_SHOW_ATTRIBUTE(dmaplane_stats);

/* ── buffers: list of active DMA buffers ──────────────────── */

static int dmaplane_buffers_show(struct seq_file *s, void *unused)
{
	struct dmaplane_dev *dev = s->private;
	int i, count = 0;

	seq_puts(s, "=== active buffers ===\n");

	mutex_lock(&dev->buf_lock);
	for (i = 0; i < DMAPLANE_MAX_BUFFERS; i++) {
		struct dmaplane_buffer *buf = &dev->buffers[i];

		if (!buf->in_use)
			continue;

		seq_printf(s, "buf_id=%-3u type=%-8s size=%-10zu numa_req=%-3d numa_actual=%-3d mmap=%d dmabuf=%s\n",
			   buf->id,
			   buf->alloc_type == DMAPLANE_BUF_TYPE_COHERENT
				? "coherent" : "pages",
			   buf->size,
			   buf->numa_node,
			   buf->actual_numa_node,
			   atomic_read(&buf->mmap_count),
			   buf->dmabuf_exported ? "yes" : "no");
		count++;
	}
	mutex_unlock(&dev->buf_lock);

	if (count == 0)
		seq_puts(s, "(none)\n");

	return 0;
}
DEFINE_SHOW_ATTRIBUTE(dmaplane_buffers);

/* ── rdma: RDMA context state and MR list ─────────────────── */

static int dmaplane_rdma_show(struct seq_file *s, void *unused)
{
	struct dmaplane_dev *dev = s->private;
	struct dmaplane_rdma_ctx *ctx = &dev->rdma;
	int i, mr_count = 0;

	seq_puts(s, "=== rdma context ===\n");

	down_read(&dev->rdma_sem);

	if (!ctx->initialized) {
		seq_puts(s, "initialized: no\n");
		up_read(&dev->rdma_sem);
		return 0;
	}

	seq_puts(s, "initialized: yes\n");
	seq_printf(s, "ib_device: %s\n", dev_name(&ctx->ib_dev->dev));
	seq_printf(s, "port: %u\n", ctx->port);
	seq_printf(s, "gid_index: %d\n", ctx->gid_index);
	seq_printf(s, "qp_a: qpn=%u\n", ctx->qp_a->qp_num);
	seq_printf(s, "qp_b: qpn=%u\n", ctx->qp_b->qp_num);

	/* MR list under mr_lock */
	mutex_lock(&dev->mr_lock);
	for (i = 0; i < DMAPLANE_MAX_MRS; i++) {
		struct dmaplane_mr_entry *mr = &dev->mrs[i];

		if (!mr->in_use)
			continue;

		seq_printf(s, "  mr_id=%-3u buf_id=%-3u lkey=0x%08x rkey=0x%08x addr=0x%016llx\n",
			   mr->id, mr->buf_id, mr->lkey, mr->rkey, mr->sge_addr);
		mr_count++;
	}
	mutex_unlock(&dev->mr_lock);

	seq_printf(s, "active_mrs: %d\n", mr_count);

	up_read(&dev->rdma_sem);

	return 0;
}
DEFINE_SHOW_ATTRIBUTE(dmaplane_rdma);

/* ── flow: flow control configuration and state ───────────── */

static int dmaplane_flow_show(struct seq_file *s, void *unused)
{
	struct dmaplane_dev *dev = s->private;

	seq_puts(s, "=== flow control ===\n");

	if (!dev->flow.configured) {
		seq_puts(s, "configured: no\n");
		return 0;
	}

	seq_puts(s, "configured: yes\n");
	seq_printf(s, "credits: %d/%u (max)\n",
		   atomic_read(&dev->flow.credits),
		   dev->flow.max_credits);
	seq_printf(s, "watermarks: high=%u low=%u\n",
		   dev->flow.high_watermark,
		   dev->flow.low_watermark);
	seq_printf(s, "paused: %s\n", dev->flow.paused ? "yes" : "no");

	return 0;
}
DEFINE_SHOW_ATTRIBUTE(dmaplane_flow);

/* ── histogram: latency distribution with percentiles ─────── */

static int dmaplane_histogram_show(struct seq_file *s, void *unused)
{
	struct dmaplane_dev *dev = s->private;
	struct dmaplane_hist_summary summary;
	u64 total, cumulative = 0;
	int i;

	dmaplane_histogram_summarize(&dev->rdma_hist, &summary);
	total = summary.count;

	seq_puts(s, "=== rdma latency histogram ===\n");
	seq_printf(s, "samples: %llu  avg: %llu ns  min: %llu ns  max: %llu ns\n",
		   summary.count, summary.avg_ns, summary.min_ns,
		   summary.max_ns);
	seq_printf(s, "p50: %llu ns  p99: %llu ns  p999: %llu ns\n",
		   summary.p50_ns, summary.p99_ns, summary.p999_ns);

	if (total == 0) {
		seq_puts(s, "\n(no samples recorded)\n");
		return 0;
	}

	seq_puts(s, "\nbucket        range_us     count    pct    cumulative\n");

	for (i = 0; i < DMAPLANE_HIST_BUCKETS; i++) {
		u64 count = atomic64_read(&dev->rdma_hist.buckets[i]);
		u64 pct_x10, cum_pct_x10;
		u32 lower, upper;

		cumulative += count;

		/* Bucket range: [2^i, 2^(i+1)) µs */
		lower = (i == 0) ? 0 : (1u << i);
		upper = (i < DMAPLANE_HIST_BUCKETS - 1) ? (1u << (i + 1)) : 0;

		/* Percentage × 10 for one decimal place without floating point */
		pct_x10 = count * 1000;
		if (total > 0)
			pct_x10 = div64_u64(pct_x10, total);
		cum_pct_x10 = cumulative * 1000;
		if (total > 0)
			cum_pct_x10 = div64_u64(cum_pct_x10, total);

		/* Only print buckets with samples to keep output readable */
		if (count == 0)
			continue;

		if (upper > 0)
			seq_printf(s, "  %-2d      [%5u, %5u)  %7llu  %3llu.%llu%%   %3llu.%llu%%\n",
				   i, lower, upper, count,
				   pct_x10 / 10, pct_x10 % 10,
				   cum_pct_x10 / 10, cum_pct_x10 % 10);
		else
			seq_printf(s, "  %-2d      [%5u,    +∞)  %7llu  %3llu.%llu%%   %3llu.%llu%%\n",
				   i, lower, count,
				   pct_x10 / 10, pct_x10 % 10,
				   cum_pct_x10 / 10, cum_pct_x10 % 10);
	}

	return 0;
}
DEFINE_SHOW_ATTRIBUTE(dmaplane_histogram);

/* ── gpu: GPU P2P state and statistics ─────────────────────── */

static int dmaplane_gpu_show(struct seq_file *s, void *unused)
{
	struct dmaplane_dev *dev = s->private;

	seq_puts(s, "=== gpu state ===\n");

#ifdef CONFIG_DMAPLANE_GPU
	{
		int i, pinned = 0;

		seq_puts(s, "symbols: resolved\n");

		mutex_lock(&dev->gpu_buf_lock);
		for (i = 0; i < DMAPLANE_MAX_GPU_BUFFERS; i++) {
			if (dev->gpu_buffers[i].in_use)
				pinned++;
		}
		seq_printf(s, "pinned_buffers: %d\n", pinned);

		for (i = 0; i < DMAPLANE_MAX_GPU_BUFFERS; i++) {
			struct dmaplane_gpu_buffer *gb = &dev->gpu_buffers[i];

			if (!gb->in_use)
				continue;

			seq_printf(s, "  handle=%-3u va=0x%016llx size=%-10llu pages=%-4d "
				   "rdma=%s revoked=%s\n",
				   gb->id, gb->gpu_va, gb->size,
				   gb->num_pages,
				   gb->rdma_vaddr ? "yes" : "no",
				   atomic_read(&gb->gpu_revoked) ? "yes" : "no");
		}
		mutex_unlock(&dev->gpu_buf_lock);
	}
#else
	seq_puts(s, "symbols: not compiled (CONFIG_DMAPLANE_GPU=n)\n");
#endif

	seq_puts(s, "stats:\n");
	seq_printf(s, "  pins=%lld unpins=%lld callbacks=%lld\n",
		   atomic64_read(&dev->gpu_stats.pins_total),
		   atomic64_read(&dev->gpu_stats.unpins_total),
		   atomic64_read(&dev->gpu_stats.callbacks_fired));
	seq_printf(s, "  h2g_bytes=%lld g2h_bytes=%lld gpu_mrs_reg=%lld\n",
		   atomic64_read(&dev->gpu_stats.dma_h2g_bytes),
		   atomic64_read(&dev->gpu_stats.dma_g2h_bytes),
		   atomic64_read(&dev->gpu_stats.gpu_mrs_registered));

	return 0;
}
DEFINE_SHOW_ATTRIBUTE(dmaplane_gpu);

/* ── Lifecycle ────────────────────────────────────────────── */

/*
 * dmaplane_debugfs_init — Create debugfs directory and files.
 *
 * Called from module init after all subsystems are initialized.
 * debugfs_create_dir can return ERR_PTR(-ENODEV) if debugfs is
 * not mounted or CONFIG_DEBUG_FS=n.  Handle gracefully — debugfs
 * is optional, never fatal.
 */
int dmaplane_debugfs_init(struct dmaplane_dev *dev)
{
	dmaplane_debugfs_root = debugfs_create_dir("dmaplane", NULL);
	if (IS_ERR(dmaplane_debugfs_root)) {
		dmaplane_debugfs_root = NULL;
		pr_debug("debugfs not available, skipping\n");
		return 0;
	}

	debugfs_create_file("stats", 0444, dmaplane_debugfs_root, dev,
			    &dmaplane_stats_fops);
	debugfs_create_file("buffers", 0444, dmaplane_debugfs_root, dev,
			    &dmaplane_buffers_fops);
	debugfs_create_file("rdma", 0444, dmaplane_debugfs_root, dev,
			    &dmaplane_rdma_fops);
	debugfs_create_file("flow", 0444, dmaplane_debugfs_root, dev,
			    &dmaplane_flow_fops);
	debugfs_create_file("histogram", 0444, dmaplane_debugfs_root, dev,
			    &dmaplane_histogram_fops);
	debugfs_create_file("gpu", 0444, dmaplane_debugfs_root, dev,
			    &dmaplane_gpu_fops);

	pr_debug("debugfs interface created at /sys/kernel/debug/dmaplane/\n");

	return 0;
}
EXPORT_SYMBOL_GPL(dmaplane_debugfs_init);

/*
 * dmaplane_debugfs_exit — Remove all debugfs entries.
 *
 * Must be called BEFORE device teardown — debugfs show functions
 * reference dmaplane_dev fields.  debugfs_remove_recursive handles
 * NULL root gracefully (no-op).
 */
void dmaplane_debugfs_exit(void)
{
	debugfs_remove_recursive(dmaplane_debugfs_root);
	dmaplane_debugfs_root = NULL;
}
EXPORT_SYMBOL_GPL(dmaplane_debugfs_exit);
