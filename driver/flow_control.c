// SPDX-License-Identifier: GPL-2.0
/*
 * flow_control.c — Credit-based flow control and throughput benchmarks
 * Copyright (c) 2026 Graziano Labs Corp.
 *
 * Phase 6 adds three capabilities:
 *
 *   1. Credit-based flow control with high/low watermark hysteresis.
 *      Setting max_credits <= CQ depth makes CQ overflow impossible
 *      by construction.
 *
 *   2. Sustained streaming benchmark.  Runs for a configurable
 *      wall-clock duration (not iteration count), with per-second
 *      throughput windowing to reveal variance that averages hide.
 *
 *   3. Queue depth sweep.  Iterates across queue depths, producing
 *      throughput/latency curves and detecting the saturation point.
 *
 * The credit system is a policy layer on top of rdma_engine.  The
 * engine provides the verbs; this module provides the pacing.
 *
 * All benchmark functions snapshot MR and buffer fields by value
 * under their respective locks (same pattern as benchmark.c).
 *
 * Callers must hold rdma_sem read lock for sustained_stream and
 * qdepth_sweep.
 */

#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/delay.h>
#include <linux/sort.h>
#include <linux/slab.h>
#include <linux/ktime.h>
#include <rdma/ib_verbs.h>

#include "dmaplane.h"
#include "rdma_engine.h"
#include "dmabuf_rdma.h"
#include "flow_control.h"
#include "dmaplane_trace.h"
#include "dmaplane_histogram.h"

#define POLL_TIMEOUT_MS		5000
#define DRAIN_TIMEOUT_MS	5000
#define WINDOW_NS		(1000000000ULL)	/* 1 second in ns */

/* ── Section 1: Flow control configuration ─────────────────── */

/*
 * dmaplane_flow_configure — Set credit-based flow control parameters.
 *
 * Validates constraints:
 *   max_credits > 0 and <= 128 (CQ depth)
 *   high_watermark <= max_credits
 *   low_watermark < high_watermark
 *   low_watermark > 0
 *
 * After configuration, credits are set to max_credits (full) and
 * the sender is unpaused.
 */
int dmaplane_flow_configure(struct dmaplane_dev *dev,
			    struct dmaplane_flow_params *params)
{
	if (params->max_credits == 0 || params->max_credits > 128) {
		pr_err("flow: max_credits %u out of range [1, 128]\n",
		       params->max_credits);
		return -EINVAL;
	}

	if (params->high_watermark > params->max_credits) {
		pr_err("flow: high_watermark %u > max_credits %u\n",
		       params->high_watermark, params->max_credits);
		return -EINVAL;
	}

	if (params->low_watermark >= params->high_watermark) {
		pr_err("flow: low_watermark %u >= high_watermark %u\n",
		       params->low_watermark, params->high_watermark);
		return -EINVAL;
	}

	if (params->low_watermark == 0) {
		pr_err("flow: low_watermark must be > 0\n");
		return -EINVAL;
	}

	dev->flow.max_credits = params->max_credits;
	dev->flow.high_watermark = params->high_watermark;
	dev->flow.low_watermark = params->low_watermark;
	atomic_set(&dev->flow.credits, params->max_credits);
	dev->flow.paused = false;
	dev->flow.configured = true;

	pr_info("flow control configured: credits=%u high=%u low=%u\n",
		params->max_credits, params->high_watermark,
		params->low_watermark);

	return 0;
}
EXPORT_SYMBOL_GPL(dmaplane_flow_configure);

/* ── Section 2: Credit operations ──────────────────────────── */

/*
 * dmaplane_flow_can_send — Check whether the sender may post a send.
 *
 * Implements high/low watermark hysteresis:
 *   - When NOT paused: pause if in-flight reaches high_watermark.
 *   - When paused: only resume if in-flight drops below low_watermark.
 *
 * Without hysteresis, the sender would toggle between send/stall on
 * every credit change, destroying throughput with context-switch overhead.
 *
 * If flow control is not configured, always returns true (no backpressure).
 */
bool dmaplane_flow_can_send(struct dmaplane_dev *dev)
{
	int credits;
	int in_flight;

	if (!dev->flow.configured)
		return true;

	credits = atomic_read(&dev->flow.credits);
	in_flight = dev->flow.max_credits - credits;

	if (dev->flow.paused) {
		/* Resume only when in-flight drops below low watermark */
		if (in_flight <= (int)dev->flow.low_watermark) {
			dev->flow.paused = false;
			atomic64_inc(&dev->stats.low_watermark_events);
			return true;
		}
		return false;
	}

	/* Pause when in-flight reaches high watermark */
	if (in_flight >= (int)dev->flow.high_watermark) {
		dev->flow.paused = true;
		atomic64_inc(&dev->stats.high_watermark_events);
		return false;
	}

	return credits > 0;
}
EXPORT_SYMBOL_GPL(dmaplane_flow_can_send);

/*
 * dmaplane_flow_on_send — Consume one credit after posting a send.
 *
 * Called immediately after a successful rdma_engine_post_send().
 * Decrements the available credit count atomically.
 */
void dmaplane_flow_on_send(struct dmaplane_dev *dev)
{
	atomic_dec(&dev->flow.credits);
}
EXPORT_SYMBOL_GPL(dmaplane_flow_on_send);

/*
 * dmaplane_flow_on_completion — Return one credit after a send completes.
 *
 * Called when a send completion is polled from CQ-A.
 * Increments the available credit count atomically.
 */
void dmaplane_flow_on_completion(struct dmaplane_dev *dev)
{
	atomic_inc(&dev->flow.credits);
}
EXPORT_SYMBOL_GPL(dmaplane_flow_on_completion);

/* ── Section 3: Sustained streaming benchmark ──────────────── */

/*
 * sustained_auto_configure — Set conservative flow control defaults.
 *
 * Called when sustained_stream is invoked without prior CONFIGURE_FLOW.
 * Uses queue_depth to derive safe credit window:
 *   max_credits = min(2 * queue_depth, 128)
 *   high_watermark = max_credits * 3 / 4
 *   low_watermark = max_credits / 4
 */
static void sustained_auto_configure(struct dmaplane_dev *dev,
				     unsigned int queue_depth)
{
	unsigned int max_credits = queue_depth * 2;

	if (max_credits > 128)
		max_credits = 128;
	if (max_credits < 2)
		max_credits = 2;

	dev->flow.max_credits = max_credits;
	dev->flow.high_watermark = max_credits * 3 / 4;
	if (dev->flow.high_watermark < 1)
		dev->flow.high_watermark = 1;
	dev->flow.low_watermark = max_credits / 4;
	if (dev->flow.low_watermark < 1)
		dev->flow.low_watermark = 1;
	/* Ensure low < high */
	if (dev->flow.low_watermark >= dev->flow.high_watermark)
		dev->flow.low_watermark = dev->flow.high_watermark - 1;

	atomic_set(&dev->flow.credits, max_credits);
	dev->flow.paused = false;
	dev->flow.configured = true;

	pr_info("flow auto-configured: credits=%u high=%u low=%u\n",
		max_credits, dev->flow.high_watermark,
		dev->flow.low_watermark);
}

/*
 * dmaplane_sustained_stream — Run sustained streaming benchmark.
 *
 * Runs for duration_secs wall-clock seconds, not a fixed iteration
 * count.  Per-second windowing tracks throughput stability — min/max
 * windows reveal variance that averages hide.
 *
 * Flow:
 *   1. Validate inputs, snapshot MR/buffer under locks
 *   2. Auto-configure flow control if needed
 *   3. Reset credits, flush CQs, pre-post receives
 *   4. Main loop: send with credit check, poll CQ-A, drain CQ-B
 *   5. Drain remaining in-flight, compute stats
 *
 * Single-threaded send loop — matches real data-loader pattern.
 * cond_resched() on stall prevents busy-wait burning a core.
 */
int dmaplane_sustained_stream(struct dmaplane_dev *dev,
			      struct dmaplane_sustained_params *params)
{
	struct dmaplane_rdma_ctx *ctx = &dev->rdma;
	struct dmaplane_mr_entry local_mr;
	struct poll_cq_wait send_wait, recv_wait;
	struct ib_wc wc;
	ktime_t start, deadline, now, window_start;
	__u64 total_bytes = 0, total_ops = 0;
	__u64 window_bytes = 0;
	__u64 min_window_mbps = U64_MAX;
	__u64 max_window_mbps = 0;
	__u64 stall_count = 0;
	__u64 cq_overflow_count = 0;
	unsigned int sent = 0, completed = 0;
	unsigned int recv_posted = 0, recv_completed = 0;
	unsigned int outstanding = 0;
	unsigned int qdepth;
	unsigned int pre_post;
	size_t buf_size;
	int ret;

	/* Phase 7: Timestamp ring for per-operation histogram recording.
	 * 128 entries (power of 2, >= max queue_depth).  Pairs sends with
	 * completions by FIFO order — RC QP guarantees in-order completion.
	 * Heap-allocated to avoid -Wframe-larger-than= warning. */
#define TS_RING_SIZE	128
	ktime_t *send_ts = NULL;
	unsigned int ts_head = 0, ts_tail = 0;

	if (!ctx->initialized)
		return -ENODEV;

	if (params->duration_secs == 0 || params->duration_secs > 600)
		return -EINVAL;
	if (params->msg_size == 0)
		return -EINVAL;
	if (params->queue_depth == 0 || params->queue_depth > 64)
		return -EINVAL;

	qdepth = params->queue_depth;

	/* Snapshot MR under lock */
	mutex_lock(&dev->mr_lock);
	{
		struct dmaplane_mr_entry *mr;

		mr = dmabuf_rdma_find_mr(dev, params->mr_id);
		if (!mr) {
			mutex_unlock(&dev->mr_lock);
			return -ENOENT;
		}
		local_mr = *mr;
	}
	mutex_unlock(&dev->mr_lock);

	/* Validate msg_size against backing buffer */
	mutex_lock(&dev->buf_lock);
	{
		struct dmaplane_buffer *buf;

		buf = dmabuf_rdma_find_buffer(dev, local_mr.buf_id);
		if (!buf) {
			mutex_unlock(&dev->buf_lock);
			return -ENOENT;
		}
		buf_size = buf->size;
	}
	mutex_unlock(&dev->buf_lock);

	if (params->msg_size > buf_size)
		return -EINVAL;

	/* Auto-configure flow control if not explicitly configured */
	if (!dev->flow.configured)
		sustained_auto_configure(dev, qdepth);

	/* Phase 7: Allocate timestamp ring for histogram latency pairing */
	send_ts = kvcalloc(TS_RING_SIZE, sizeof(ktime_t), GFP_KERNEL);
	if (!send_ts)
		return -ENOMEM;

	/* Reset credits for this run */
	atomic_set(&dev->flow.credits, dev->flow.max_credits);
	dev->flow.paused = false;

	/* Flush stale completions from previous runs */
	rdma_engine_flush_cq(ctx->cq_a);
	rdma_engine_flush_cq(ctx->cq_b);

	/* Pre-post receives: bounded by both 2*qdepth and max_credits.
	 * Never pre-post more than credits allow — the entire point of
	 * the credit system. */
	pre_post = qdepth * 2;
	if (pre_post > (unsigned int)dev->flow.max_credits)
		pre_post = dev->flow.max_credits;

	for (recv_posted = 0; recv_posted < pre_post; recv_posted++) {
		ret = rdma_engine_post_recv(dev, ctx->qp_b, &local_mr,
					    params->msg_size, &recv_wait);
		if (ret) {
			pr_err("sustained pre-post recv %u failed: %d\n",
			       recv_posted, ret);
			kvfree(send_ts);
			return ret;
		}
	}

	/* Initialize timing */
	start = ktime_get();
	deadline = ktime_add_ms(start, (u64)params->duration_secs * 1000UL);
	window_start = start;

	/* Main loop — runs until wall-clock deadline */
	while (!ktime_after(ktime_get(), deadline)) {
		now = ktime_get();

		/* Per-second window check */
		if (ktime_to_ns(ktime_sub(now, window_start)) >= WINDOW_NS) {
			__u64 window_ns = ktime_to_ns(ktime_sub(now,
								 window_start));
			__u64 window_mbps = 0;

			if (window_ns > 0 && window_bytes > 0)
				window_mbps = window_bytes * 1000 / window_ns;

			if (window_mbps < min_window_mbps)
				min_window_mbps = window_mbps;
			if (window_mbps > max_window_mbps)
				max_window_mbps = window_mbps;

			window_bytes = 0;
			window_start = now;
			cond_resched();
		}

		/* Send phase: post if credits available AND under qdepth.
		 * Cache can_send result — the function has side effects
		 * (modifies paused flag, increments watermark stats). */
		{
			bool can_send = dmaplane_flow_can_send(dev);

			if (can_send && outstanding < qdepth) {
				ret = rdma_engine_post_send(dev, ctx->qp_a,
							    &local_mr,
							    params->msg_size,
							    &send_wait);
				if (ret) {
					pr_err("sustained send failed at %u: %d\n",
					       sent, ret);
					goto done;
				}
				dmaplane_flow_on_send(dev);
				send_ts[ts_head & (TS_RING_SIZE - 1)] = ktime_get();
				ts_head++;
				sent++;
				outstanding++;
			} else if (!can_send) {
				ktime_t stall_start, stall_end;

				stall_count++;
				atomic64_inc(&dev->stats.credit_stalls);
				/* Yield — let rxe softirq process completions */
				stall_start = ktime_get();
				cond_resched();
				stall_end = ktime_get();
				trace_dmaplane_flow_stall(
					atomic_read(&dev->flow.credits),
					outstanding,
					ktime_to_ns(ktime_sub(stall_end,
							      stall_start)));
			}
		}

		/* Poll send CQ-A */
		ret = ib_poll_cq(ctx->cq_a, 1, &wc);
		if (ret > 0) {
			if (wc.status == IB_WC_SUCCESS) {
				ktime_t lat = ktime_sub(ktime_get(),
					send_ts[ts_tail & (TS_RING_SIZE - 1)]);
				ts_tail++;
				dmaplane_histogram_record(&dev->rdma_hist,
							  ktime_to_ns(lat));

				completed++;
				outstanding--;
				dmaplane_flow_on_completion(dev);
				total_bytes += params->msg_size;
				total_ops++;
				window_bytes += params->msg_size;

				atomic64_inc(&dev->stats.completions_polled);
				atomic64_add(params->msg_size,
					     &dev->stats.bytes_sent);

				/* Replenish a recv — never exceed
				 * sent + qdepth to avoid over-posting */
				if (recv_posted < sent + qdepth) {
					ret = rdma_engine_post_recv(
						dev, ctx->qp_b, &local_mr,
						params->msg_size, &recv_wait);
					if (ret) {
						pr_err("sustained replenish recv %u failed: %d\n",
						       recv_posted, ret);
						goto done;
					}
					recv_posted++;
				}
			} else {
				pr_err("sustained send completion error: %s\n",
				       ib_wc_status_msg(wc.status));
				atomic64_inc(&dev->stats.completion_errors);
				cq_overflow_count++;
				atomic64_inc(&dev->stats.cq_overflows);
				ret = -EIO;
				goto done;
			}
		} else if (ret < 0) {
			pr_err("sustained CQ-A poll error: %d\n", ret);
			ret = -EIO;
			goto done;
		}

		/* Eagerly drain recv CQ-B to prevent overflow */
		while (ib_poll_cq(ctx->cq_b, 1, &wc) > 0) {
			if (wc.status == IB_WC_SUCCESS) {
				recv_completed++;
				atomic64_inc(&dev->stats.completions_polled);
				atomic64_add(params->msg_size,
					     &dev->stats.bytes_received);
			} else {
				cq_overflow_count++;
				atomic64_inc(&dev->stats.cq_overflows);
			}
		}
	}

	ret = 0;

done:
	/* Drain remaining in-flight completions with timeout.
	 * Without this, stale completions poison the next run. */
	{
		ktime_t drain_deadline = ktime_add_ms(ktime_get(),
						      DRAIN_TIMEOUT_MS);

		while (outstanding > 0 &&
		       !ktime_after(ktime_get(), drain_deadline)) {
			if (ib_poll_cq(ctx->cq_a, 1, &wc) > 0) {
				if (wc.status == IB_WC_SUCCESS) {
					outstanding--;
					dmaplane_flow_on_completion(dev);
					total_bytes += params->msg_size;
					total_ops++;
				} else {
					outstanding--;
				}
			}
			/* Also drain recv CQ */
			while (ib_poll_cq(ctx->cq_b, 1, &wc) > 0)
				recv_completed++;
			cond_resched();
		}

		if (outstanding > 0)
			pr_warn("sustained drain incomplete: %u still outstanding\n",
				outstanding);
	}

	/* Final CQ flush for clean state */
	rdma_engine_flush_cq(ctx->cq_a);
	rdma_engine_flush_cq(ctx->cq_b);

	/* Compute results */
	{
		ktime_t end = ktime_get();
		__u64 elapsed_ns = ktime_to_ns(ktime_sub(end, start));

		params->total_bytes = total_bytes;
		params->total_ops = total_ops;
		params->stall_count = stall_count;
		params->cq_overflow_count = cq_overflow_count;

		if (elapsed_ns > 0)
			params->avg_throughput_mbps =
				total_bytes * 1000 / elapsed_ns;
		else
			params->avg_throughput_mbps = 0;

		/* Handle min_window_mbps if no full window was recorded */
		if (min_window_mbps == U64_MAX)
			min_window_mbps = params->avg_throughput_mbps;
		params->min_window_mbps = min_window_mbps;
		params->max_window_mbps = max_window_mbps;
		params->status = ret ? 1 : 0;
	}

	/* Update device-level cumulative stats */
	atomic64_add(total_bytes, &dev->stats.sustained_bytes);
	atomic64_add(total_ops, &dev->stats.sustained_ops);

	pr_debug("sustained %u sent, %u completed, %llu bytes, %llu MB/s, "
		 "stalls=%llu overflows=%llu (duration=%u s)\n",
		 sent, completed, total_bytes, params->avg_throughput_mbps,
		 stall_count, cq_overflow_count, params->duration_secs);

	kvfree(send_ts);
	return ret;
}
EXPORT_SYMBOL_GPL(dmaplane_sustained_stream);

/* ── Section 4: Queue depth sweep ──────────────────────────── */

/*
 * dmaplane_qdepth_sweep — Characterize throughput vs queue depth.
 *
 * Iterates from min_qdepth to max_qdepth by step, running a fixed-
 * iteration streaming benchmark at each point.  Records throughput,
 * average latency, and P99 latency per point.
 *
 * Detects the saturation point: the smallest queue depth where
 * throughput reaches 95% of the maximum observed.
 *
 * Per-iteration ktime_get() adds ~20-50 ns overhead — acceptable
 * for rxe (~200 us per operation) but noted for real hardware.
 */
int dmaplane_qdepth_sweep(struct dmaplane_dev *dev,
			  struct dmaplane_sweep_params *params)
{
	struct dmaplane_rdma_ctx *ctx = &dev->rdma;
	struct dmaplane_mr_entry local_mr;
	struct poll_cq_wait send_wait, recv_wait;
	struct ib_wc wc;
	unsigned int point, nr_points;
	__u64 max_throughput = 0;
	size_t buf_size;
	int ret;

	if (!ctx->initialized)
		return -ENODEV;

	if (params->min_qdepth < 1 || params->max_qdepth > 64 ||
	    params->min_qdepth > params->max_qdepth ||
	    params->step < 1 || params->iterations == 0 ||
	    params->msg_size == 0)
		return -EINVAL;

	nr_points = (params->max_qdepth - params->min_qdepth) / params->step + 1;
	if (nr_points > DMAPLANE_MAX_SWEEP_POINTS)
		nr_points = DMAPLANE_MAX_SWEEP_POINTS;

	/* Snapshot MR under lock */
	mutex_lock(&dev->mr_lock);
	{
		struct dmaplane_mr_entry *mr;

		mr = dmabuf_rdma_find_mr(dev, params->mr_id);
		if (!mr) {
			mutex_unlock(&dev->mr_lock);
			return -ENOENT;
		}
		local_mr = *mr;
	}
	mutex_unlock(&dev->mr_lock);

	/* Validate msg_size against backing buffer */
	mutex_lock(&dev->buf_lock);
	{
		struct dmaplane_buffer *buf;

		buf = dmabuf_rdma_find_buffer(dev, local_mr.buf_id);
		if (!buf) {
			mutex_unlock(&dev->buf_lock);
			return -ENOENT;
		}
		buf_size = buf->size;
	}
	mutex_unlock(&dev->buf_lock);

	if (params->msg_size > buf_size)
		return -EINVAL;

	pr_info("QD sweep: msg_size=%u iterations=%u range=[%u..%u] step=%u\n",
		params->msg_size, params->iterations,
		params->min_qdepth, params->max_qdepth, params->step);

	/* Sweep across queue depths */
	for (point = 0; point < nr_points; point++) {
		unsigned int qdepth = params->min_qdepth +
				      point * params->step;
		__u64 *latencies;
		ktime_t start, prev_time;
		unsigned int sent = 0, completed = 0;
		unsigned int recv_posted = 0, recv_drained = 0;
		unsigned int outstanding = 0;
		unsigned int pre_post;

		/* Flush CQs between points */
		rdma_engine_flush_cq(ctx->cq_a);
		rdma_engine_flush_cq(ctx->cq_b);

		/* Allocate per-iteration latency array */
		latencies = kvcalloc(params->iterations, sizeof(__u64),
				     GFP_KERNEL);
		/* Non-fatal if allocation fails — skip P99 */

		/* Pre-post receives */
		pre_post = qdepth * 2;
		if (pre_post > params->iterations)
			pre_post = params->iterations;

		for (recv_posted = 0; recv_posted < pre_post; recv_posted++) {
			ret = rdma_engine_post_recv(dev, ctx->qp_b, &local_mr,
						    params->msg_size,
						    &recv_wait);
			if (ret) {
				pr_err("sweep QD=%u pre-post recv failed: %d\n",
				       qdepth, ret);
				kvfree(latencies);
				goto out;
			}
		}

		start = ktime_get();
		prev_time = start;

		/* Send/poll loop — same pattern as benchmark_streaming */
		while (completed < params->iterations) {
			while (sent < params->iterations &&
			       outstanding < qdepth) {
				ret = rdma_engine_post_send(dev, ctx->qp_a,
							    &local_mr,
							    params->msg_size,
							    &send_wait);
				if (ret) {
					pr_err("sweep QD=%u send failed: %d\n",
					       qdepth, ret);
					kvfree(latencies);
					goto out;
				}
				sent++;
				outstanding++;
			}

			ret = rdma_engine_poll_cq(ctx->cq_a, &wc,
						  POLL_TIMEOUT_MS);
			if (ret > 0 && wc.status == IB_WC_SUCCESS) {
				ktime_t now = ktime_get();

				if (latencies)
					latencies[completed] =
						ktime_to_ns(ktime_sub(now,
								      prev_time));
				prev_time = now;
				outstanding--;
				completed++;

				/* Eagerly drain recv CQ-B */
				while (ib_poll_cq(ctx->cq_b, 1, &wc) > 0) {
					if (wc.status == IB_WC_SUCCESS)
						recv_drained++;
				}

				/* Replenish recv */
				if (recv_posted < params->iterations) {
					ret = rdma_engine_post_recv(
						dev, ctx->qp_b, &local_mr,
						params->msg_size, &recv_wait);
					if (ret == 0)
						recv_posted++;
				}
			} else if (ret > 0) {
				pr_err("sweep QD=%u completion error: %s\n",
				       qdepth, ib_wc_status_msg(wc.status));
				kvfree(latencies);
				ret = -EIO;
				goto out;
			} else if (ret == 0) {
				pr_err("sweep QD=%u poll timeout at %u/%u\n",
				       qdepth, completed, params->iterations);
				kvfree(latencies);
				ret = -ETIMEDOUT;
				goto out;
			}
		}

		/* Drain remaining recv completions */
		while (recv_drained < completed) {
			ret = rdma_engine_poll_cq(ctx->cq_b, &wc,
						  POLL_TIMEOUT_MS);
			if (ret <= 0 || wc.status != IB_WC_SUCCESS)
				break;
			recv_drained++;
		}

		/* Compute stats for this point */
		{
			ktime_t end = ktime_get();
			__u64 total_ns = ktime_to_ns(ktime_sub(end, start));
			__u64 bytes_total = (__u64)params->msg_size * completed;

			if (total_ns > 0)
				params->throughput_mbps[point] =
					bytes_total * 1000 / total_ns;

			if (completed > 0)
				params->avg_latency_ns[point] =
					total_ns / completed;

			if (latencies && completed > 0) {
				unsigned int j;

				/* Record all latency samples in histogram */
				for (j = 0; j < completed; j++)
					dmaplane_histogram_record(
						&dev->rdma_hist, latencies[j]);

				sort(latencies, completed, sizeof(__u64),
				     rdma_engine_cmp_u64, NULL);
				params->p99_latency_ns[point] =
					latencies[(completed * 99) / 100];
			}

			if (params->throughput_mbps[point] > max_throughput)
				max_throughput = params->throughput_mbps[point];
		}

		kvfree(latencies);

		pr_info("  QD=%2u  %6llu MB/s  avg=%6llu ns  p99=%6llu ns\n",
			qdepth, params->throughput_mbps[point],
			params->avg_latency_ns[point],
			params->p99_latency_ns[point]);

		cond_resched();
	}

	/* Detect saturation point: smallest QD where throughput >= 95% max.
	 *
	 * If the maximum throughput is at the last point, throughput is still
	 * climbing and no true plateau was reached.  Report saturation_qdepth=0
	 * to distinguish "no saturation within range" from a real plateau.
	 * The 95% threshold only detects a plateau when an earlier point
	 * already reaches 95% of the peak — meaning adding more QD yields
	 * less than 5% improvement. */
	params->nr_points = nr_points;
	params->saturation_qdepth = 0;  /* 0 = no saturation detected */

	if (max_throughput > 0 && nr_points >= 2) {
		__u64 threshold = max_throughput * 95 / 100;
		unsigned int max_point = nr_points - 1;

		/* Only report saturation if a point before the last one
		 * already reaches 95% of max — that's a real plateau.
		 * If only the last point hits the threshold, the curve
		 * is still climbing. */
		for (point = 0; point < max_point; point++) {
			if (params->throughput_mbps[point] >= threshold) {
				params->saturation_qdepth =
					params->min_qdepth +
					point * params->step;
				break;
			}
		}
	}

	params->status = 0;
	ret = 0;

	if (params->saturation_qdepth > 0) {
		unsigned int sat_idx = (params->saturation_qdepth -
					params->min_qdepth) / params->step;
		pr_info("Saturation at QD=%u (%llu MB/s)\n",
			params->saturation_qdepth,
			params->throughput_mbps[sat_idx]);
	} else {
		pr_info("No saturation within range [%u..%u] — "
			"throughput still climbing at QD=%u (%llu MB/s)\n",
			params->min_qdepth,
			params->min_qdepth + (nr_points - 1) * params->step,
			params->min_qdepth + (nr_points - 1) * params->step,
			nr_points > 0 ?
			params->throughput_mbps[nr_points - 1] : 0ULL);
	}

out:
	/* Final CQ flush */
	rdma_engine_flush_cq(ctx->cq_a);
	rdma_engine_flush_cq(ctx->cq_b);

	return ret;
}
EXPORT_SYMBOL_GPL(dmaplane_qdepth_sweep);
