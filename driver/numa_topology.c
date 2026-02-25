// SPDX-License-Identifier: GPL-2.0
/*
 * numa_topology.c — NUMA topology awareness and cross-node benchmarking
 * Copyright (c) 2026 Graziano Labs Corp.
 *
 * Phase 5 of the dmaplane project: measuring the cost of NUMA
 * disaggregation.  This file implements two capabilities:
 *
 * 1. Topology query: enumerates online NUMA nodes, CPU-to-node
 *    mappings, per-node memory, and the ACPI SLIT distance matrix.
 *    This is the kernel equivalent of what `lstopo` and
 *    `numactl --hardware` expose to userspace.
 *
 * 2. Cross-node bandwidth benchmark: allocates test buffers on
 *    each NUMA node, then for every (cpu_node, buf_node) pair,
 *    pins a kthread to cpu_node and measures memcpy throughput to
 *    the buffer on buf_node.  The result is an NxN bandwidth matrix
 *    that quantifies the cross-node penalty.
 *
 * Why this matters for AI systems:
 *
 * On a multi-socket training node with 8 GPUs across 2 sockets,
 * the data loader must allocate gradient staging buffers on the
 * same NUMA node as the target GPU.  NCCL's topology detection
 * (ncclCommInitRank -> nvidia-smi topo -m) exists for exactly
 * this reason.  A mismatched GPU-NIC pair — GPU on socket 0, NIC
 * on socket 1 — produces no error, just 30-50% lower all-reduce
 * bandwidth.  The NUMA benchmark here measures exactly this cost.
 */

#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/numa.h>
#include <linux/nodemask.h>
#include <linux/cpumask.h>
#include <linux/topology.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>
#include <linux/mm.h>
#include <linux/mmzone.h>
#include <linux/vmstat.h>
#include <linux/kthread.h>
#include <linux/completion.h>
#include <linux/sched.h>
#include <linux/delay.h>
#include "dmaplane.h"
#include "numa_topology.h"

/* ============================================================================
 * Topology Query
 * ============================================================================
 */

/*
 * dmaplane_query_numa_topo — Populate topology information from the kernel.
 *
 * Enumerates online NUMA nodes (up to DMAPLANE_MAX_NUMA_NODES),
 * counts CPUs per node, reads the ACPI SLIT distance matrix via
 * node_distance(), and reports per-node memory totals.
 *
 * Distance values come from the ACPI SLIT (System Locality Information
 * Table) — the hardware-reported relative cost of cross-node access.
 * 10 = same node, 20-21 = one hop, higher = further.  Values are
 * relative, not absolute.  On single-socket systems there's only one
 * node and all distances are 10.
 *
 * Node IDs can be sparse (e.g., node 0 and node 2 online, node 1
 * offline).  We compact into contiguous array indices via node_to_idx[].
 */
int dmaplane_query_numa_topo(struct dmaplane_numa_topo *topo)
{
	int node, i, j;
	unsigned int cpu;
	int node_idx = 0;
	int *node_to_idx;

	memset(topo, 0, sizeof(*topo));

	/* Sparse-to-compact mapping: node_to_idx[kernel_node_id] = array index.
	 * MAX_NUMNODES can be large (1024 on some configs), so heap-allocate. */
	node_to_idx = kvcalloc(MAX_NUMNODES, sizeof(int), GFP_KERNEL);
	if (!node_to_idx)
		return -ENOMEM;
	memset(node_to_idx, -1, MAX_NUMNODES * sizeof(int));

	topo->nr_cpus = num_online_cpus();

	/*
	 * Enumerate online nodes and populate per-node info.
	 */
	for_each_online_node(node) {
		if (node_idx >= DMAPLANE_MAX_NUMA_NODES)
			break;
		node_to_idx[node] = node_idx;
		topo->node_online[node_idx] = 1;

		/* Count CPUs on this node */
		for_each_online_cpu(cpu) {
			if (cpu_to_node(cpu) == node)
				topo->node_cpu_count[node_idx]++;
		}

		/*
		 * Per-node memory info.
		 *
		 * node_present_pages() and NODE_DATA() work on both
		 * NUMA and non-NUMA kernels.  On non-NUMA, NODE_DATA()
		 * returns the single global pgdat, so these report
		 * system-wide totals for node 0 — which is correct
		 * since there's only one node.
		 */
		{
			unsigned long present = node_present_pages(node);
			unsigned long free_pages = 0;
			struct zone *z;

			for (z = NODE_DATA(node)->node_zones;
			     z < NODE_DATA(node)->node_zones + MAX_NR_ZONES;
			     z++) {
				if (managed_zone(z))
					free_pages += zone_page_state(z,
								NR_FREE_PAGES);
			}

			topo->node_mem_total_kb[node_idx] =
				((__u64)present * PAGE_SIZE) >> 10;
			topo->node_mem_free_kb[node_idx] =
				((__u64)free_pages * PAGE_SIZE) >> 10;
		}
		node_idx++;
	}
	topo->nr_nodes = node_idx;

	/*
	 * Populate the ACPI SLIT distance matrix.
	 *
	 * node_distance(a, b) returns the relative distance between
	 * NUMA nodes, as reported by firmware (ACPI SLIT table on x86,
	 * device tree on ARM).  The values are dimensionless ratios
	 * normalized so that local access = 10.
	 *
	 * Typical values:
	 *   Same socket: 10
	 *   Adjacent socket (1 hop QPI/UPI): 20-21
	 *   Remote socket (2 hops): 30-31
	 */
	{
		int ni = 0;
		int nj;

		for_each_online_node(node) {
			if (ni >= DMAPLANE_MAX_NUMA_NODES)
				break;
			nj = 0;
			for_each_online_node(j) {
				if (nj >= DMAPLANE_MAX_NUMA_NODES)
					break;
				topo->distances[ni][nj] = node_distance(node, j);
				nj++;
			}
			ni++;
		}
	}

	pr_debug("NUMA topology: %u nodes, %u CPUs\n",
		 topo->nr_nodes, topo->nr_cpus);
	for (i = 0; (unsigned int)i < topo->nr_nodes; i++) {
		pr_debug("  node %d: %u CPUs, %llu MB total, %llu MB free\n",
			 i, topo->node_cpu_count[i],
			 topo->node_mem_total_kb[i] >> 10,
			 topo->node_mem_free_kb[i] >> 10);
		for (j = 0; (unsigned int)j < topo->nr_nodes; j++)
			pr_debug("    distance[%d][%d] = %u\n",
				 i, j, topo->distances[i][j]);
	}

	kvfree(node_to_idx);
	return 0;
}
EXPORT_SYMBOL_GPL(dmaplane_query_numa_topo);

/* ============================================================================
 * NUMA Bandwidth Benchmark
 * ============================================================================
 */

/*
 * Per-cell benchmark context.
 *
 * For each (cpu_node, buf_node) pair, we spawn a kthread that:
 * 1. Is pinned to a CPU on cpu_node via set_cpus_allowed_ptr()
 * 2. Runs memcpy from a local scratch buffer to the target buffer
 *    allocated on buf_node
 * 3. Measures elapsed time and reports bandwidth
 *
 * This directly measures the cost that a data loader thread on
 * socket 0 pays when accessing gradient buffers on socket 1.
 */
struct numa_bench_cell {
	struct completion done;

	/* Input */
	void *dst;                /* buffer on buf_node */
	void *src;                /* scratch buffer on cpu_node */
	size_t size;
	unsigned int iterations;
	int cpu_node;             /* node to pin the thread to */

	/* Output */
	__u64 elapsed_ns;
	__u64 bw_mbps;
	int status;
};

/*
 * numa_bench_thread — Worker that measures memcpy bandwidth.
 *
 * Pinned to a specific NUMA node.  Performs `iterations` memcpy
 * operations between a node-local source and the target buffer
 * (which may be on a different node).  The memcpy direction is
 * write: src (local) -> dst (possibly remote).  This models the
 * common pattern of a CPU thread writing gradient data into a
 * staging buffer that a NIC will DMA-read.
 *
 * We use plain memcpy rather than DMA because:
 * - It measures the CPU->memory path that page allocation NUMA
 *   placement directly affects
 * - DMA throughput depends on IOMMU config and device, not just
 *   NUMA placement
 * - CPU memcpy is the bottleneck in data loading pipelines
 */
static int numa_bench_thread(void *arg)
{
	struct numa_bench_cell *cell = arg;
	ktime_t start, end;
	unsigned int i;

	/*
	 * Touch both buffers once to ensure pages are faulted in.
	 * Without this, the first iteration includes page fault
	 * overhead that distorts the measurement.
	 */
	memset(cell->src, 0xAA, cell->size);
	memset(cell->dst, 0x55, cell->size);

	start = ktime_get();

	for (i = 0; i < cell->iterations; i++) {
		memcpy(cell->dst, cell->src, cell->size);
		/*
		 * Memory barrier to prevent the compiler from optimizing
		 * away the memcpy (it might notice dst is never read).
		 * Also ensures each iteration's writes are visible before
		 * the next iteration starts — models real-world behavior
		 * where the NIC would read the buffer between writes.
		 */
		barrier();
	}

	end = ktime_get();

	cell->elapsed_ns = ktime_to_ns(ktime_sub(end, start));

	/* Bandwidth = total_bytes / time.
	 * MB/s = (size * iterations) / (elapsed_ns / 1e9) / 1e6
	 *       = (size * iterations * 1000) / elapsed_ns */
	if (cell->elapsed_ns > 0) {
		__u64 total_bytes = (__u64)cell->size * cell->iterations;

		cell->bw_mbps = total_bytes * 1000 / cell->elapsed_ns;
	}

	cell->status = 0;
	complete(&cell->done);
	return 0;
}

/*
 * node_buffer — NUMA-allocated buffer for the benchmark.
 *
 * Uses alloc_pages_node() + vmap() — same pattern as the main
 * buffer allocation in dmabuf_rdma.c, but temporary (freed after
 * each measurement cell).
 */
struct node_buffer {
	void *vaddr;
	struct page **pages;
	unsigned int nr_pages;
	bool is_vmapped;        /* true = page array + vmap */
};

/*
 * alloc_node_buffer — Allocate a buffer on a specific NUMA node.
 *
 * Allocates individual pages via alloc_pages_node() and vmaps them
 * into a contiguous kernel VA.  Unwinds on any failure.
 */
static int alloc_node_buffer(struct node_buffer *nb, int node, size_t size)
{
	unsigned int i;

	memset(nb, 0, sizeof(*nb));

	nb->nr_pages = (size + PAGE_SIZE - 1) / PAGE_SIZE;
	nb->pages = kvcalloc(nb->nr_pages, sizeof(struct page *), GFP_KERNEL);
	if (!nb->pages)
		return -ENOMEM;

	for (i = 0; i < nb->nr_pages; i++) {
		nb->pages[i] = alloc_pages_node(node,
						GFP_KERNEL | __GFP_ZERO, 0);
		if (!nb->pages[i]) {
			while (i > 0)
				__free_page(nb->pages[--i]);
			kvfree(nb->pages);
			nb->pages = NULL;
			return -ENOMEM;
		}
	}

	nb->vaddr = vmap(nb->pages, nb->nr_pages, VM_MAP, PAGE_KERNEL);
	if (!nb->vaddr) {
		for (i = 0; i < nb->nr_pages; i++)
			__free_page(nb->pages[i]);
		kvfree(nb->pages);
		nb->pages = NULL;
		return -ENOMEM;
	}
	nb->is_vmapped = true;
	return 0;
}

/*
 * free_node_buffer — Free a buffer allocated by alloc_node_buffer().
 * Reverse-order cleanup: vunmap, then free each page, then the array.
 */
static void free_node_buffer(struct node_buffer *nb)
{
	unsigned int i;

	if (nb->is_vmapped && nb->vaddr)
		vunmap(nb->vaddr);
	if (nb->pages) {
		for (i = 0; i < nb->nr_pages; i++) {
			if (nb->pages[i])
				__free_page(nb->pages[i]);
		}
		kvfree(nb->pages);
	}
	memset(nb, 0, sizeof(*nb));
}

/*
 * find_cpu_on_node — Return the first online CPU on a NUMA node.
 *
 * Used to verify a node has CPUs and to construct the cpumask for
 * kthread pinning.  Returns -1 if no CPU is found (shouldn't happen
 * for nodes in the online_nodes list).
 */
static int find_cpu_on_node(int node)
{
	unsigned int cpu;

	for_each_online_cpu(cpu) {
		if (cpu_to_node(cpu) == node)
			return cpu;
	}
	return -1;
}

/*
 * dmaplane_numa_bench — Run the NxN cross-node bandwidth benchmark.
 *
 * For each (cpu_node, buf_node) pair among online NUMA nodes:
 *   1. Allocate a target buffer on buf_node
 *   2. Allocate a scratch buffer on cpu_node (source for memcpy)
 *   3. Spawn a kthread pinned to cpu_node
 *   4. Measure memcpy throughput: src (local) -> dst (remote or local)
 *
 * Results populate the bw_mbps and lat_ns matrices.  On a single-node
 * system the matrix is 1x1 with only local bandwidth.  On a dual-socket
 * system you get the classic 2x2 matrix showing local vs. remote penalty.
 *
 * Design notes:
 * - Buffers are allocated per-measurement rather than shared, to avoid
 *   cache warming effects between cells.
 * - kthreads are used rather than work_struct because we need precise
 *   CPU affinity control.  workqueues don't guarantee which CPU runs.
 * - Each cell runs sequentially to avoid cross-cell interference on
 *   the memory bus.  Parallel execution would under-report bandwidth.
 */
int dmaplane_numa_bench(struct dmaplane_numa_bench_params *params)
{
	int online_nodes[DMAPLANE_MAX_NUMA_NODES];
	int nr_nodes = 0;
	int node;
	int ci, bi;  /* cpu_node index, buf_node index */
	int ret = 0;

	if (params->buffer_size == 0 || params->buffer_size > (64ULL << 20))
		return -EINVAL;  /* cap at 64 MB per buffer */
	if (params->iterations == 0 || params->iterations > 100000)
		return -EINVAL;

	memset(params->bw_mbps, 0, sizeof(params->bw_mbps));
	memset(params->lat_ns, 0, sizeof(params->lat_ns));

	/* Enumerate online nodes */
	for_each_online_node(node) {
		if (nr_nodes >= DMAPLANE_MAX_NUMA_NODES)
			break;
		online_nodes[nr_nodes++] = node;
	}
	params->nr_nodes = nr_nodes;

	pr_info("NUMA bench starting: %d nodes, %llu bytes/buf, %u iterations\n",
		nr_nodes, params->buffer_size, params->iterations);

	for (ci = 0; ci < nr_nodes; ci++) {
		int cpu_node = online_nodes[ci];
		int cpu_id;
		cpumask_var_t cpu_mask;

		cpu_id = find_cpu_on_node(cpu_node);
		if (cpu_id < 0) {
			pr_warn("no CPU on node %d, skipping\n", cpu_node);
			continue;
		}

		if (!alloc_cpumask_var(&cpu_mask, GFP_KERNEL)) {
			ret = -ENOMEM;
			goto out;
		}

		/* Build cpumask: all CPUs on cpu_node.
		 * We allow the thread to run on any CPU on the node
		 * rather than pinning to a single core — this is more
		 * realistic (real workloads use all cores on a socket)
		 * and avoids single-core thermal effects. */
		cpumask_clear(cpu_mask);
		{
			unsigned int c;

			for_each_online_cpu(c) {
				if (cpu_to_node(c) == cpu_node)
					cpumask_set_cpu(c, cpu_mask);
			}
		}

		for (bi = 0; bi < nr_nodes; bi++) {
			int buf_node = online_nodes[bi];
			struct node_buffer dst_buf, src_buf;
			struct numa_bench_cell cell;
			struct task_struct *thread;
			unsigned long timeout;

			/* Allocate target buffer on buf_node */
			ret = alloc_node_buffer(&dst_buf, buf_node,
						params->buffer_size);
			if (ret) {
				pr_err("alloc dst buf on node %d failed: %d\n",
				       buf_node, ret);
				free_cpumask_var(cpu_mask);
				goto out;
			}

			/* Allocate source buffer on cpu_node (local scratch) */
			ret = alloc_node_buffer(&src_buf, cpu_node,
						params->buffer_size);
			if (ret) {
				pr_err("alloc src buf on node %d failed: %d\n",
				       cpu_node, ret);
				free_node_buffer(&dst_buf);
				free_cpumask_var(cpu_mask);
				goto out;
			}

			/* Set up the benchmark cell */
			init_completion(&cell.done);
			cell.dst = dst_buf.vaddr;
			cell.src = src_buf.vaddr;
			cell.size = params->buffer_size;
			cell.iterations = params->iterations;
			cell.cpu_node = cpu_node;
			cell.elapsed_ns = 0;
			cell.bw_mbps = 0;
			cell.status = -EINVAL;

			/* Spawn kthread and pin to cpu_node */
			thread = kthread_create(numa_bench_thread, &cell,
						"dmaplane_numa_%d_%d",
						cpu_node, buf_node);
			if (IS_ERR(thread)) {
				ret = PTR_ERR(thread);
				pr_err("kthread create failed: %d\n", ret);
				free_node_buffer(&src_buf);
				free_node_buffer(&dst_buf);
				free_cpumask_var(cpu_mask);
				goto out;
			}

			set_cpus_allowed_ptr(thread, cpu_mask);
			wake_up_process(thread);

			/* Wait with timeout (generous: 60s for large buffers) */
			timeout = wait_for_completion_timeout(&cell.done,
							      msecs_to_jiffies(60000));
			if (!timeout) {
				pr_err("NUMA bench [cpu=%d, buf=%d] timed out\n",
				       cpu_node, buf_node);
				/* Thread may still be running — it will
				 * complete eventually and touch cell.done.
				 * We can't safely free cell, so leak and abort. */
				ret = -ETIMEDOUT;
				free_node_buffer(&src_buf);
				free_node_buffer(&dst_buf);
				free_cpumask_var(cpu_mask);
				goto out;
			}

			params->bw_mbps[ci][bi] = cell.bw_mbps;
			if (cell.iterations > 0 && cell.elapsed_ns > 0)
				params->lat_ns[ci][bi] =
					cell.elapsed_ns / cell.iterations;

			pr_debug("NUMA bench [cpu=%d -> buf=%d]: %llu MB/s, %llu ns/iter\n",
				 cpu_node, buf_node,
				 cell.bw_mbps, params->lat_ns[ci][bi]);

			free_node_buffer(&src_buf);
			free_node_buffer(&dst_buf);

			/* Yield between cells to reduce thermal throttling */
			cond_resched();
		}
		free_cpumask_var(cpu_mask);
	}

	params->status = 0;

	/* Print summary matrix */
	pr_info("NUMA Bandwidth Matrix (MB/s):\n");
	{
		int r, c;

		for (r = 0; r < nr_nodes; r++) {
			for (c = 0; c < nr_nodes; c++) {
				pr_info("  [cpu=%d->buf=%d] %6llu MB/s  %6llu ns/iter\n",
					online_nodes[r], online_nodes[c],
					params->bw_mbps[r][c],
					params->lat_ns[r][c]);
			}
		}
	}

out:
	if (ret)
		params->status = ret;
	return ret;
}
EXPORT_SYMBOL_GPL(dmaplane_numa_bench);
