// SPDX-License-Identifier: GPL-2.0
/*
 * benchmark.c — RDMA benchmarks
 * Copyright (c) 2026 Graziano Labs Corp.
 *
 * Implements three benchmark patterns over the loopback QP pair:
 *
 *   loopback  — single send/recv validation test
 *   pingpong  — N-iteration round-trip latency with P99
 *   streaming — pipelined throughput measurement
 *
 * All benchmarks snapshot MR and buffer fields by value under their
 * respective locks, then release the locks before doing I/O.  This
 * decouples benchmark execution from concurrent MR deregistration or
 * buffer destruction.
 *
 * Callers must hold rdma_sem read lock.
 */

#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/delay.h>
#include <linux/sort.h>
#include <linux/slab.h>
#include <linux/completion.h>
#include <linux/jiffies.h>
#include <rdma/ib_verbs.h>

#include "dmaplane.h"
#include "rdma_engine.h"
#include "dmabuf_rdma.h"
#include "benchmark.h"
#include "dmaplane_histogram.h"

#define POLL_TIMEOUT_MS  5000

/*
 * benchmark_loopback — Single send from QP-A, recv on QP-B.
 *
 * Verifies the entire data path works end-to-end: post_recv on B,
 * post_send on A, poll both CQs, measure latency.
 */
int benchmark_loopback(struct dmaplane_dev *edev,
		       struct dmaplane_loopback_params *params)
{
	struct dmaplane_rdma_ctx *ctx = &edev->rdma;
	struct dmaplane_mr_entry local_mr;
	struct poll_cq_wait send_wait, recv_wait;
	ktime_t start, end;
	unsigned int recv_bytes = 0;
	int ret;
	__u32 pattern = 0xDEADBEEF;
	void *buf_vaddr;
	size_t buf_size;

	if (!ctx->initialized)
		return -EINVAL;

	/* Snapshot MR by value under lock — the copy decouples us from
	 * mr_lock so a concurrent deregister can't free the MR while
	 * the benchmark uses the cached lkey/sge_addr */
	mutex_lock(&edev->mr_lock);
	{
		struct dmaplane_mr_entry *mr;

		mr = dmabuf_rdma_find_mr(edev, params->mr_id);
		if (!mr) {
			mutex_unlock(&edev->mr_lock);
			return -ENOENT;
		}
		local_mr = *mr;
	}
	mutex_unlock(&edev->mr_lock);

	/* Snapshot buffer fields under lock */
	mutex_lock(&edev->buf_lock);
	{
		struct dmaplane_buffer *buf;

		buf = dmabuf_rdma_find_buffer(edev, local_mr.buf_id);
		if (!buf || !buf->vaddr) {
			mutex_unlock(&edev->buf_lock);
			return -ENOENT;
		}
		buf_vaddr = buf->vaddr;
		buf_size = buf->size;
	}
	mutex_unlock(&edev->buf_lock);

	if (params->size > buf_size)
		return -EINVAL;

	/* Write test pattern for transfer — loopback validates the data
	 * path works, not data integrity (no memcmp on receive) */
	memset(buf_vaddr, 0, buf_size);
	memcpy(buf_vaddr, &pattern, sizeof(pattern));

	start = ktime_get();

	/* Post receive on QP-B first (must be ready before send) */
	ret = rdma_engine_post_recv(edev, ctx->qp_b, &local_mr, params->size,
				    &recv_wait);
	if (ret) {
		pr_err("loopback post_recv failed: %d\n", ret);
		return ret;
	}

	/* Post send on QP-A */
	ret = rdma_engine_post_send(edev, ctx->qp_a, &local_mr, params->size,
				    &send_wait);
	if (ret) {
		pr_err("loopback post_send failed: %d\n", ret);
		return ret;
	}

	/* Wait for send completion */
	{
		struct ib_wc wc;
		int rc = rdma_engine_poll_cq(ctx->cq_a, &wc, POLL_TIMEOUT_MS);

		if (rc <= 0) {
			pr_err("loopback send completion timeout\n");
			return -ETIMEDOUT;
		}
		if (wc.status != IB_WC_SUCCESS) {
			pr_err("loopback send completion error: %s (%d)\n",
			       ib_wc_status_msg(wc.status), wc.status);
			atomic64_inc(&edev->stats.completion_errors);
			return -EIO;
		}
	}
	atomic64_inc(&edev->stats.completions_polled);
	atomic64_add(params->size, &edev->stats.bytes_sent);

	/* Wait for recv completion */
	{
		struct ib_wc wc;
		int rc = rdma_engine_poll_cq(ctx->cq_b, &wc, POLL_TIMEOUT_MS);

		if (rc <= 0) {
			pr_err("loopback recv completion timeout\n");
			return -ETIMEDOUT;
		}
		if (wc.status != IB_WC_SUCCESS) {
			pr_err("loopback recv completion error: %s (%d)\n",
			       ib_wc_status_msg(wc.status), wc.status);
			atomic64_inc(&edev->stats.completion_errors);
			return -EIO;
		}
		recv_bytes = wc.byte_len;
	}
	atomic64_inc(&edev->stats.completions_polled);
	atomic64_add(recv_bytes, &edev->stats.bytes_received);

	end = ktime_get();

	params->latency_ns = ktime_to_ns(ktime_sub(end, start));
	params->status = 0;

	pr_debug("loopback OK — %u bytes, latency %llu ns\n",
		 params->size, params->latency_ns);
	return 0;
}

/* Use shared comparator from rdma_engine for P99 sorting */

/*
 * benchmark_pingpong — Measure round-trip latency over multiple iterations.
 *
 * For each iteration:
 * 1. Post recv on QP-B
 * 2. Post send on QP-A
 * 3. Wait for send completion on CQ-A
 * 4. Wait for recv completion on CQ-B
 * 5. Record latency
 *
 * P99 via sorted-array indexing.  When iterations < 100, this returns
 * a lower percentile (e.g. iterations=10 gives P90).  Acceptable —
 * benchmarks typically run 100+ iterations.
 */
int benchmark_pingpong(struct dmaplane_dev *edev,
		       struct dmaplane_bench_params *params)
{
	struct dmaplane_rdma_ctx *ctx = &edev->rdma;
	struct dmaplane_mr_entry local_mr;
	struct poll_cq_wait send_wait, recv_wait;
	__u64 *latencies;
	__u64 total_ns = 0;
	ktime_t iter_start, iter_end;
	unsigned int i;
	int ret;
	size_t buf_size;

	if (!ctx->initialized || params->iterations == 0 ||
	    params->msg_size == 0)
		return -EINVAL;

	/* Snapshot MR fields under lock */
	mutex_lock(&edev->mr_lock);
	{
		struct dmaplane_mr_entry *mr;

		mr = dmabuf_rdma_find_mr(edev, params->mr_id);
		if (!mr) {
			mutex_unlock(&edev->mr_lock);
			return -ENOENT;
		}
		local_mr = *mr;
	}
	mutex_unlock(&edev->mr_lock);

	/* Validate msg_size against backing buffer */
	mutex_lock(&edev->buf_lock);
	{
		struct dmaplane_buffer *buf;

		buf = dmabuf_rdma_find_buffer(edev, local_mr.buf_id);
		if (!buf) {
			mutex_unlock(&edev->buf_lock);
			return -ENOENT;
		}
		buf_size = buf->size;
	}
	mutex_unlock(&edev->buf_lock);

	if (params->msg_size > buf_size)
		return -EINVAL;

	latencies = kvcalloc(params->iterations, sizeof(__u64), GFP_KERNEL);
	if (!latencies)
		return -ENOMEM;

	for (i = 0; i < params->iterations; i++) {
		iter_start = ktime_get();

		/* Post recv on B */
		ret = rdma_engine_post_recv(edev, ctx->qp_b, &local_mr,
					    params->msg_size, &recv_wait);
		if (ret)
			goto out;

		/* Post send on A */
		ret = rdma_engine_post_send(edev, ctx->qp_a, &local_mr,
					    params->msg_size, &send_wait);
		if (ret)
			goto out;

		/* Wait for send completion */
		{
			struct ib_wc wc;
			int rc = rdma_engine_poll_cq(ctx->cq_a, &wc,
						     POLL_TIMEOUT_MS);
			if (rc <= 0 || wc.status != IB_WC_SUCCESS) {
				pr_err("pingpong send fail iter %u\n", i);
				atomic64_inc(&edev->stats.completion_errors);
				ret = -EIO;
				goto out;
			}
		}
		atomic64_inc(&edev->stats.completions_polled);

		/* Wait for recv completion */
		{
			struct ib_wc wc;
			int rc = rdma_engine_poll_cq(ctx->cq_b, &wc,
						     POLL_TIMEOUT_MS);
			if (rc <= 0 || wc.status != IB_WC_SUCCESS) {
				pr_err("pingpong recv fail iter %u\n", i);
				atomic64_inc(&edev->stats.completion_errors);
				ret = -EIO;
				goto out;
			}
		}
		atomic64_inc(&edev->stats.completions_polled);

		iter_end = ktime_get();
		latencies[i] = ktime_to_ns(ktime_sub(iter_end, iter_start));
		total_ns += latencies[i];
		dmaplane_histogram_record(&edev->rdma_hist, latencies[i]);

		atomic64_add(params->msg_size, &edev->stats.bytes_sent);
		atomic64_add(params->msg_size, &edev->stats.bytes_received);
	}

	sort(latencies, params->iterations, sizeof(__u64), rdma_engine_cmp_u64, NULL);

	params->total_ns = total_ns;
	params->avg_latency_ns = total_ns / params->iterations;
	params->p99_latency_ns = latencies[(params->iterations * 99) / 100];

	if (total_ns > 0) {
		__u64 bytes_total = (__u64)params->msg_size * params->iterations;
		/* MB/s = bytes / (ns / 1e9) / 1e6 = bytes * 1000 / ns */
		params->throughput_mbps = bytes_total * 1000 / total_ns;
	}
	params->mr_reg_ns = ktime_to_ns(local_mr.reg_time);

	pr_debug("pingpong %u iters x %u bytes: avg=%llu ns, p99=%llu ns, %llu MB/s\n",
		 params->iterations, params->msg_size,
		 params->avg_latency_ns, params->p99_latency_ns,
		 params->throughput_mbps);
	ret = 0;

out:
	kvfree(latencies);
	return ret;
}

/*
 * benchmark_streaming — Measure throughput with pipelined sends.
 *
 * Posts multiple sends back-to-back (up to queue_depth outstanding),
 * polling completions as they arrive.  Pre-posts receives up to
 * 2*qdepth and replenishes during the send/poll loop.
 *
 * Flushes both CQs before starting and after finishing so back-to-back
 * runs don't poison each other with stale completions.
 */
int benchmark_streaming(struct dmaplane_dev *edev,
			struct dmaplane_bench_params *params)
{
	struct dmaplane_rdma_ctx *ctx = &edev->rdma;
	struct dmaplane_mr_entry local_mr;
	struct poll_cq_wait send_wait, recv_wait;
	struct ib_wc wc;
	ktime_t start, end, prev_time;
	unsigned int sent = 0, completed = 0;
	unsigned int recv_posted = 0, recv_drained = 0;
	unsigned int outstanding = 0;
	__u32 qdepth;
	__u64 *latencies = NULL;
	int ret;
	size_t buf_size;

	if (!ctx->initialized || params->iterations == 0 ||
	    params->msg_size == 0)
		return -EINVAL;

	/* Snapshot MR fields under lock */
	mutex_lock(&edev->mr_lock);
	{
		struct dmaplane_mr_entry *mr;

		mr = dmabuf_rdma_find_mr(edev, params->mr_id);
		if (!mr) {
			mutex_unlock(&edev->mr_lock);
			return -ENOENT;
		}
		local_mr = *mr;
	}
	mutex_unlock(&edev->mr_lock);

	/* Validate msg_size against backing buffer */
	mutex_lock(&edev->buf_lock);
	{
		struct dmaplane_buffer *buf;

		buf = dmabuf_rdma_find_buffer(edev, local_mr.buf_id);
		if (!buf) {
			mutex_unlock(&edev->buf_lock);
			return -ENOENT;
		}
		buf_size = buf->size;
	}
	mutex_unlock(&edev->buf_lock);

	if (params->msg_size > buf_size)
		return -EINVAL;

	qdepth = params->queue_depth ? params->queue_depth : 16;

	/* Allocate per-completion latency array for p99 computation.
	 * Non-fatal if allocation fails — we just skip p99. */
	latencies = kvcalloc(params->iterations, sizeof(__u64), GFP_KERNEL);

	/* Flush any stale completions from previous runs */
	rdma_engine_flush_cq(ctx->cq_a);
	rdma_engine_flush_cq(ctx->cq_b);

	/*
	 * Pre-post receives up to 2x queue depth (not all iterations).
	 * We replenish during the send/poll loop so we never overflow
	 * the RQ regardless of how many iterations are requested.
	 */
	{
		unsigned int pre_post = qdepth * 2;

		if (pre_post > params->iterations)
			pre_post = params->iterations;

		for (recv_posted = 0; recv_posted < pre_post; recv_posted++) {
			ret = rdma_engine_post_recv(edev, ctx->qp_b, &local_mr,
						    params->msg_size,
						    &recv_wait);
			if (ret) {
				pr_err("streaming pre-post recv %u failed: %d\n",
				       recv_posted, ret);
				goto done;
			}
		}
	}

	start = ktime_get();
	prev_time = start;

	while (completed < params->iterations) {
		/* Post sends up to queue depth */
		while (sent < params->iterations && outstanding < qdepth) {
			ret = rdma_engine_post_send(edev, ctx->qp_a, &local_mr,
						    params->msg_size,
						    &send_wait);
			if (ret) {
				pr_err("streaming send %u failed: %d\n",
				       sent, ret);
				goto done;
			}
			sent++;
			outstanding++;
		}

		/* Poll send CQ */
		ret = rdma_engine_poll_cq(ctx->cq_a, &wc, POLL_TIMEOUT_MS);
		if (ret > 0 && wc.status == IB_WC_SUCCESS) {
			ktime_t now = ktime_get();

			if (latencies)
				latencies[completed] =
					ktime_to_ns(ktime_sub(now, prev_time));
			prev_time = now;

			atomic64_inc(&edev->stats.completions_polled);
			atomic64_add(params->msg_size, &edev->stats.bytes_sent);
			outstanding--;
			completed++;

			/* Drain recv CQ-B eagerly to prevent overflow.
			 * Non-blocking: each ib_poll_cq call returns
			 * immediately; loop consumes all available. */
			while (ib_poll_cq(ctx->cq_b, 1, &wc) > 0) {
				if (wc.status == IB_WC_SUCCESS) {
					recv_drained++;
					atomic64_inc(&edev->stats.completions_polled);
					atomic64_add(params->msg_size,
						     &edev->stats.bytes_received);
				}
			}

			/* Replenish a recv for the next iteration */
			if (recv_posted < params->iterations) {
				ret = rdma_engine_post_recv(edev, ctx->qp_b,
							    &local_mr,
							    params->msg_size,
							    &recv_wait);
				if (ret) {
					pr_err("streaming replenish recv %u failed: %d\n",
					       recv_posted, ret);
					goto done;
				}
				recv_posted++;
			}
		} else if (ret > 0) {
			pr_err("streaming send completion error: %s\n",
			       ib_wc_status_msg(wc.status));
			atomic64_inc(&edev->stats.completion_errors);
			ret = -EIO;
			goto done;
		} else if (ret == 0) {
			pr_err("streaming send CQ poll timeout at %u\n",
			       completed);
			ret = -ETIMEDOUT;
			goto done;
		}
	}

	/* Drain any remaining recv completions on QP-B */
	while (recv_drained < completed) {
		ret = rdma_engine_poll_cq(ctx->cq_b, &wc, POLL_TIMEOUT_MS);
		if (ret <= 0 || wc.status != IB_WC_SUCCESS) {
			pr_warn("streaming recv drain incomplete at %u/%u\n",
				recv_drained, completed);
			break;
		}
		recv_drained++;
		atomic64_inc(&edev->stats.completions_polled);
		atomic64_add(params->msg_size, &edev->stats.bytes_received);
	}

	ret = 0;

done:
	end = ktime_get();
	params->total_ns = ktime_to_ns(ktime_sub(end, start));

	if (completed > 0)
		params->avg_latency_ns = params->total_ns / completed;

	if (params->total_ns > 0) {
		__u64 bytes_total = (__u64)params->msg_size * completed;
		params->throughput_mbps = bytes_total * 1000 / params->total_ns;
	}

	/* Compute p99 from per-completion inter-arrival times */
	if (latencies && completed > 0) {
		unsigned int j;

		/* Record every latency sample in the histogram before sorting
		 * (sort reorders, but histogram doesn't care about order) */
		for (j = 0; j < completed; j++)
			dmaplane_histogram_record(&edev->rdma_hist,
						  latencies[j]);

		sort(latencies, completed, sizeof(__u64), rdma_engine_cmp_u64, NULL);
		params->p99_latency_ns = latencies[(completed * 99) / 100];
	}
	params->mr_reg_ns = ktime_to_ns(local_mr.reg_time);

	pr_debug("streaming %u/%u completed, %llu ns total, %llu MB/s, p99=%llu ns (qdepth=%u)\n",
		 completed, params->iterations, params->total_ns,
		 params->throughput_mbps, params->p99_latency_ns, qdepth);

	/* Always flush both CQs — leave clean state for next run */
	{
		int fa = rdma_engine_flush_cq(ctx->cq_a);
		int fb = rdma_engine_flush_cq(ctx->cq_b);

		if (fa || fb)
			pr_debug("streaming cleanup flushed %d from CQ-A, %d from CQ-B\n",
				 fa, fb);
	}

	kvfree(latencies);
	return ret;
}

EXPORT_SYMBOL_GPL(benchmark_loopback);
EXPORT_SYMBOL_GPL(benchmark_pingpong);
EXPORT_SYMBOL_GPL(benchmark_streaming);
