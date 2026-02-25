// SPDX-License-Identifier: GPL-2.0
/*
 * dmaplane_histogram.c — Log₂ latency histogram implementation
 * Copyright (c) 2026 Graziano Labs Corp.
 *
 * Provides constant-space latency distribution tracking for RDMA
 * benchmarks.  All operations are lock-free using atomic counters
 * and cmpxchg loops for min/max tracking.
 *
 * Bucket assignment: latency in nanoseconds is converted to
 * microseconds, then bucketed via ilog2.  Bucket i covers
 * [2^i, 2^(i+1)) µs.  ilog2(0) is undefined — zero-latency
 * samples go to bucket 0.
 *
 * Percentile computation: walks buckets from 0 to 15, accumulating
 * counts.  P50/P99/P999 are the upper bound of the bucket where
 * the cumulative count crosses the target percentage.  This is
 * coarse but O(1) in space and computation.
 */

#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt

#include <linux/module.h>
#include <linux/log2.h>
#include <linux/math64.h>
#include "dmaplane_histogram.h"

/*
 * dmaplane_histogram_init — Initialize histogram to empty state.
 *
 * Sets min_ns to U64_MAX so any real sample will be smaller.
 * Must be called before first record.
 */
void dmaplane_histogram_init(struct dmaplane_histogram *hist)
{
	int i;

	for (i = 0; i < DMAPLANE_HIST_BUCKETS; i++)
		atomic64_set(&hist->buckets[i], 0);
	atomic64_set(&hist->count, 0);
	atomic64_set(&hist->sum_ns, 0);
	atomic64_set(&hist->min_ns, U64_MAX);
	atomic64_set(&hist->max_ns, 0);
}
EXPORT_SYMBOL_GPL(dmaplane_histogram_init);

/*
 * dmaplane_histogram_record — Add a latency sample.
 *
 * Lock-free: uses atomic increments for buckets/count/sum and
 * cmpxchg loops for min/max.  Safe to call from multiple contexts
 * concurrently (benchmarks, sustained streaming, QD sweep).
 *
 * The cmpxchg loops for min/max may momentarily lose a race under
 * extreme contention (two threads both trying to update min with
 * nearly identical values), but they converge to the correct value.
 */
void dmaplane_histogram_record(struct dmaplane_histogram *hist,
			       u64 latency_ns)
{
	u64 latency_us = latency_ns / 1000;
	unsigned int bucket;
	u64 old_min, old_max;

	/* ilog2(0) is undefined — route sub-microsecond to bucket 0 */
	if (latency_us == 0)
		bucket = 0;
	else
		bucket = min_t(unsigned int, ilog2(latency_us),
			       DMAPLANE_HIST_BUCKETS - 1);

	atomic64_inc(&hist->buckets[bucket]);
	atomic64_inc(&hist->count);
	atomic64_add(latency_ns, &hist->sum_ns);

	/* Update min — relaxed CAS loop */
	old_min = atomic64_read(&hist->min_ns);
	while (latency_ns < old_min) {
		if (atomic64_try_cmpxchg(&hist->min_ns, &old_min, latency_ns))
			break;
	}

	/* Update max — relaxed CAS loop */
	old_max = atomic64_read(&hist->max_ns);
	while (latency_ns > old_max) {
		if (atomic64_try_cmpxchg(&hist->max_ns, &old_max, latency_ns))
			break;
	}
}
EXPORT_SYMBOL_GPL(dmaplane_histogram_record);

/*
 * dmaplane_histogram_reset — Clear histogram to empty state.
 *
 * Not atomic across all fields — a concurrent record during reset
 * may produce a momentarily inconsistent snapshot.  Acceptable for
 * diagnostic use.  Callers should avoid concurrent reset + record.
 */
void dmaplane_histogram_reset(struct dmaplane_histogram *hist)
{
	int i;

	for (i = 0; i < DMAPLANE_HIST_BUCKETS; i++)
		atomic64_set(&hist->buckets[i], 0);
	atomic64_set(&hist->count, 0);
	atomic64_set(&hist->sum_ns, 0);
	atomic64_set(&hist->min_ns, U64_MAX);
	atomic64_set(&hist->max_ns, 0);
}
EXPORT_SYMBOL_GPL(dmaplane_histogram_reset);

/*
 * dmaplane_histogram_summarize — Compute percentiles and summary stats.
 *
 * Walks buckets from 0 to 15, accumulating sample counts.  Percentiles
 * are reported as the upper bound of the bucket where the cumulative
 * count crosses the target fraction:
 *   P50:  cumulative * 1000 >= total * 500
 *   P99:  cumulative * 1000 >= total * 990
 *   P999: cumulative * 10000 >= total * 9990
 *
 * Overflow note: cumulative * 10000 can overflow u64 if count exceeds
 * ~1.8e15.  For practical use (billions of samples max), this is safe.
 *
 * Bucket upper bound for bucket i: ((u64)1 << (i+1)) * 1000 nanoseconds
 * (i.e., 2^(i+1) microseconds converted to nanoseconds).
 * Bucket 15 (the overflow bucket) reports U64_MAX.
 */
void dmaplane_histogram_summarize(struct dmaplane_histogram *hist,
				  struct dmaplane_hist_summary *out)
{
	u64 total = atomic64_read(&hist->count);
	u64 sum = atomic64_read(&hist->sum_ns);
	u64 cumulative = 0;
	bool found_p50 = false, found_p99 = false, found_p999 = false;
	int i;

	out->count = total;
	out->avg_ns = total > 0 ? div64_u64(sum, total) : 0;
	out->min_ns = total > 0 ? atomic64_read(&hist->min_ns) : 0;
	out->max_ns = atomic64_read(&hist->max_ns);
	out->p50_ns = 0;
	out->p99_ns = 0;
	out->p999_ns = 0;

	if (total == 0)
		return;

	for (i = 0; i < DMAPLANE_HIST_BUCKETS; i++) {
		u64 bucket_upper_ns;

		cumulative += atomic64_read(&hist->buckets[i]);

		/* Upper bound: 2^(i+1) µs in nanoseconds, except last bucket */
		if (i < DMAPLANE_HIST_BUCKETS - 1)
			bucket_upper_ns = ((u64)1 << (i + 1)) * 1000;
		else
			bucket_upper_ns = U64_MAX;

		if (!found_p50 && cumulative * 1000 >= total * 500) {
			out->p50_ns = bucket_upper_ns;
			found_p50 = true;
		}
		if (!found_p99 && cumulative * 1000 >= total * 990) {
			out->p99_ns = bucket_upper_ns;
			found_p99 = true;
		}
		if (!found_p999 && cumulative * 10000 >= total * 9990) {
			out->p999_ns = bucket_upper_ns;
			found_p999 = true;
		}
	}
}
EXPORT_SYMBOL_GPL(dmaplane_histogram_summarize);
