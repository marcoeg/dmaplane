/* SPDX-License-Identifier: GPL-2.0 */
/*
 * numa_topology.h — NUMA topology query and benchmark declarations
 * Copyright (c) 2026 Graziano Labs Corp.
 *
 * Phase 5: NUMA awareness. Provides two capabilities:
 *   1. Topology query: enumerates nodes, CPUs, memory, distance matrix.
 *   2. Bandwidth benchmark: NxN cross-node memcpy throughput matrix.
 */
#ifndef _NUMA_TOPOLOGY_H
#define _NUMA_TOPOLOGY_H

#include "dmaplane.h"

/*
 * dmaplane_query_numa_topo — Enumerate NUMA topology.
 *
 * Populates node count, per-node CPU counts, memory sizes, and the
 * ACPI SLIT distance matrix (relative cost of cross-node access:
 * 10 = same node, 20+ = remote).  Works on non-NUMA kernels (reports
 * a single node with system-wide totals).
 *
 * Return: 0 on success, negative errno on failure.
 */
int dmaplane_query_numa_topo(struct dmaplane_numa_topo *topo);

/*
 * dmaplane_numa_bench — Run NxN cross-node memcpy benchmark.
 *
 * For each (source_node, dest_node) pair, spawns a kthread pinned to
 * source_node and measures memcpy throughput to a buffer on dest_node.
 * Populates bandwidth (MB/s) and latency (ns) matrices in params.
 * Uses kthreads (not workqueues) for precise CPU affinity control.
 *
 * Return: 0 on success, negative errno on failure.
 */
int dmaplane_numa_bench(struct dmaplane_numa_bench_params *params);

#endif /* _NUMA_TOPOLOGY_H */
