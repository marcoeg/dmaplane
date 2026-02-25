/* SPDX-License-Identifier: GPL-2.0 */
/*
 * dmaplane_histogram.h — Log₂ latency histogram for RDMA operations
 * Copyright (c) 2026 Graziano Labs Corp.
 *
 * Provides constant-space, lock-free latency distribution tracking.
 * 16 buckets cover [2^i, 2^(i+1)) microseconds:
 *   Bucket 0:  [0,     1) µs      Bucket 8:  [256,   512) µs
 *   Bucket 1:  [1,     2) µs      Bucket 9:  [512,  1024) µs
 *   Bucket 2:  [2,     4) µs      Bucket 10: [1024, 2048) µs
 *   Bucket 3:  [4,     8) µs      Bucket 11: [2048, 4096) µs
 *   Bucket 4:  [8,    16) µs      Bucket 12: [4096, 8192) µs
 *   Bucket 5:  [16,   32) µs      Bucket 13: [8192,16384) µs
 *   Bucket 6:  [32,   64) µs      Bucket 14: [16384,32768) µs
 *   Bucket 7:  [128, 256) µs      Bucket 15: [32768,  ∞) µs
 *
 * All fields are atomic — no lock needed.  Multiple concurrent
 * benchmark paths (pingpong, streaming, sustained) can record
 * simultaneously.  The min/max use atomic cmpxchg loops — correct
 * but may momentarily lose races under extreme contention.
 * Acceptable for diagnostic use.
 *
 * Percentile accuracy: coarse (power-of-2 bucket boundaries).
 * For rxe at ~200 µs, P50 reports the bucket upper bound (256 µs)
 * rather than the exact median.  This is by design — kernel-side
 * histograms trade precision for zero-allocation constant-space
 * collection.  Exact percentiles require sorting N samples (what
 * the QD sweep does).
 */
#ifndef _DMAPLANE_HISTOGRAM_H
#define _DMAPLANE_HISTOGRAM_H

#include <linux/types.h>
#include <linux/atomic.h>

/*
 * Use the UAPI definition if already included (via dmaplane.h →
 * dmaplane_uapi.h), otherwise define locally.  Both must agree.
 */
#ifndef DMAPLANE_HIST_BUCKETS
#define DMAPLANE_HIST_BUCKETS	16
#endif

struct dmaplane_histogram {
	atomic64_t buckets[DMAPLANE_HIST_BUCKETS];
	atomic64_t count;	/* total samples recorded */
	atomic64_t sum_ns;	/* sum of all latencies (for average) */
	atomic64_t min_ns;	/* minimum latency observed (init U64_MAX) */
	atomic64_t max_ns;	/* maximum latency observed */
};

/*
 * Percentile computation output — filled by dmaplane_histogram_summarize.
 * Percentiles report the upper bound of the bucket containing the
 * target rank.  This is a conservative estimate: P50=256000 ns means
 * "50% of samples had latency < 256 µs."
 */
struct dmaplane_hist_summary {
	u64 p50_ns;
	u64 p99_ns;
	u64 p999_ns;
	u64 avg_ns;
	u64 min_ns;
	u64 max_ns;
	u64 count;
};

void dmaplane_histogram_init(struct dmaplane_histogram *hist);
void dmaplane_histogram_record(struct dmaplane_histogram *hist,
			       u64 latency_ns);
void dmaplane_histogram_reset(struct dmaplane_histogram *hist);
void dmaplane_histogram_summarize(struct dmaplane_histogram *hist,
				  struct dmaplane_hist_summary *out);

#endif /* _DMAPLANE_HISTOGRAM_H */
