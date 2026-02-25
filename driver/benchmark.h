/* SPDX-License-Identifier: GPL-2.0 */
/*
 * benchmark.h — RDMA benchmark API declarations
 * Copyright (c) 2026 Graziano Labs Corp.
 *
 * Three benchmark functions that exercise the RDMA loopback pair
 * (QP-A sends, QP-B receives) established by rdma_engine_setup().
 *
 * All functions snapshot MR and buffer fields by value under their
 * respective locks, then release the locks before doing I/O.  This
 * decouples benchmark execution from concurrent deregistration.
 *
 * Callers must hold rdma_sem read lock.
 */
#ifndef _BENCHMARK_H
#define _BENCHMARK_H

#include "dmaplane.h"

/*
 * benchmark_loopback() — Single send/recv via QP-A → QP-B.
 *
 * Validates the RDMA data path works end-to-end.  Not a throughput
 * measurement — use streaming for that.
 *
 * Return: 0 on success, negative errno on failure.
 */
int benchmark_loopback(struct dmaplane_dev *edev,
		       struct dmaplane_loopback_params *params);

/*
 * benchmark_pingpong() — Round-trip latency measurement.
 *
 * Runs N iterations of send+recv, collecting per-iteration latency.
 * Computes avg, P99, and throughput.  Best for measuring single-message
 * round-trip time at various message sizes.
 *
 * Return: 0 on success, negative errno on failure.
 */
int benchmark_pingpong(struct dmaplane_dev *edev,
		       struct dmaplane_bench_params *params);

/*
 * benchmark_streaming() — Sustained throughput measurement.
 *
 * Pipelines sends up to queue depth, polling completions concurrently.
 * Measures sustained throughput at various message sizes and queue
 * depths.  queue_depth must be <= max_recv_wr/2 for the pre-posting
 * scheme to work within QP receive queue capacity.
 *
 * Return: 0 on success, negative errno on failure.
 */
int benchmark_streaming(struct dmaplane_dev *edev,
			struct dmaplane_bench_params *params);

#endif /* _BENCHMARK_H */
