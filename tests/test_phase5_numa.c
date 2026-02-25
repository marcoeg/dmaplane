// SPDX-License-Identifier: GPL-2.0
/*
 * test_phase5_numa.c — Phase 5 NUMA topology and optimization tests
 * Copyright (c) 2026 Graziano Labs Corp.
 *
 * 8 test cases exercising NUMA-aware buffer allocation, topology query,
 * cross-node bandwidth benchmark, and regression with NUMA_ANY.
 *
 * Requires: sudo (CAP_SYS_ADMIN for /dev/dmaplane)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <stdint.h>

#include "dmaplane_uapi.h"

/* ── Test framework ────────────────────────────────────────── */

static int tests_passed;
static int tests_failed;

#define TEST_START(name) \
	printf("  [%d] %-50s ", tests_passed + tests_failed + 1, name)

#define TEST_PASS() do { printf("PASS\n"); tests_passed++; } while (0)

#define TEST_FAIL(reason) do { \
	printf("FAIL (%s)\n", reason); tests_failed++; \
} while (0)

/* ── Helpers ───────────────────────────────────────────────── */

static unsigned int create_buffer(int fd, unsigned int type,
				  unsigned long size, int numa_node,
				  int *actual_node)
{
	struct dmaplane_buf_params p = { .numa_node = numa_node };

	p.alloc_type = type;
	p.size = size;
	if (ioctl(fd, DMAPLANE_IOCTL_CREATE_BUFFER, &p) < 0)
		return 0;
	if (actual_node)
		*actual_node = p.actual_numa_node;
	return p.buf_id;
}

static int destroy_buffer(int fd, unsigned int buf_id)
{
	__u32 id = buf_id;

	return ioctl(fd, DMAPLANE_IOCTL_DESTROY_BUFFER, &id);
}

/* ── Test 1: Topology query ────────────────────────────────── */

static void test_topology_query(int fd)
{
	struct dmaplane_numa_topo topo;
	unsigned int i, j;

	TEST_START("NUMA topology query");

	memset(&topo, 0, sizeof(topo));
	if (ioctl(fd, DMAPLANE_IOCTL_QUERY_NUMA_TOPO, &topo) < 0) {
		TEST_FAIL(strerror(errno));
		return;
	}

	if (topo.nr_nodes < 1 || topo.nr_cpus < 1) {
		TEST_FAIL("nr_nodes or nr_cpus is 0");
		return;
	}

	if (!topo.node_online[0]) {
		TEST_FAIL("node 0 not online");
		return;
	}

	if (topo.distances[0][0] != 10) {
		char msg[64];

		snprintf(msg, sizeof(msg), "distance[0][0] = %u, expected 10",
			 topo.distances[0][0]);
		TEST_FAIL(msg);
		return;
	}

	TEST_PASS();

	/* Print topology details */
	printf("\n    NUMA Topology:\n");
	printf("      Nodes: %u   CPUs: %u\n", topo.nr_nodes, topo.nr_cpus);
	for (i = 0; i < topo.nr_nodes; i++) {
		printf("      Node %u: %u CPUs, %llu MB total, %llu MB free\n",
		       i, topo.node_cpu_count[i],
		       (unsigned long long)topo.node_mem_total_kb[i] >> 10,
		       (unsigned long long)topo.node_mem_free_kb[i] >> 10);
	}

	printf("\n    NUMA Distance Matrix:\n");
	printf("           ");
	for (j = 0; j < topo.nr_nodes; j++)
		printf("node%-3u", j);
	printf("\n");
	for (i = 0; i < topo.nr_nodes; i++) {
		printf("      node%u ", i);
		for (j = 0; j < topo.nr_nodes; j++)
			printf("%-7u", topo.distances[i][j]);
		printf("\n");
	}
	printf("\n");
}

/* ── Test 2: Allocate on specific node (node 0) ───────────── */

static void test_alloc_node0(int fd)
{
	int actual_node = -99;
	unsigned int buf_id;

	TEST_START("allocate page-backed buffer on node 0");

	buf_id = create_buffer(fd, DMAPLANE_BUF_TYPE_PAGES, 64 * 1024,
			       0, &actual_node);
	if (!buf_id) {
		TEST_FAIL(strerror(errno));
		return;
	}

	if (actual_node != 0) {
		char msg[64];

		snprintf(msg, sizeof(msg), "actual_numa_node = %d, expected 0",
			 actual_node);
		destroy_buffer(fd, buf_id);
		TEST_FAIL(msg);
		return;
	}

	destroy_buffer(fd, buf_id);
	TEST_PASS();
	printf("      actual_numa_node = %d\n", actual_node);
}

/* ── Test 3: Allocate with NUMA_ANY ────────────────────────── */

static void test_alloc_numa_any(int fd)
{
	int actual_node = -99;
	unsigned int buf_id;

	TEST_START("allocate page-backed buffer with NUMA_ANY");

	buf_id = create_buffer(fd, DMAPLANE_BUF_TYPE_PAGES, 64 * 1024,
			       DMAPLANE_NUMA_ANY, &actual_node);
	if (!buf_id) {
		TEST_FAIL(strerror(errno));
		return;
	}

	if (actual_node < 0) {
		char msg[64];

		snprintf(msg, sizeof(msg), "actual_numa_node = %d, expected >= 0",
			 actual_node);
		destroy_buffer(fd, buf_id);
		TEST_FAIL(msg);
		return;
	}

	destroy_buffer(fd, buf_id);
	TEST_PASS();
	printf("      actual_numa_node = %d\n", actual_node);
}

/* ── Test 4: Invalid node (must fail) ──────────────────────── */

static void test_invalid_node(int fd)
{
	unsigned int buf_id;

	TEST_START("invalid NUMA node (99) returns EINVAL");

	buf_id = create_buffer(fd, DMAPLANE_BUF_TYPE_PAGES, 4096, 99, NULL);
	if (buf_id != 0) {
		destroy_buffer(fd, buf_id);
		TEST_FAIL("expected failure, got success");
		return;
	}

	if (errno != EINVAL) {
		char msg[64];

		snprintf(msg, sizeof(msg), "expected EINVAL, got %s",
			 strerror(errno));
		TEST_FAIL(msg);
		return;
	}

	TEST_PASS();
}

/* ── Test 5: Coherent buffer NUMA reporting ────────────────── */

static void test_coherent_numa(int fd)
{
	int actual_node = -99;
	unsigned int buf_id;

	TEST_START("coherent buffer NUMA reporting (informational)");

	buf_id = create_buffer(fd, DMAPLANE_BUF_TYPE_COHERENT, 4096,
			       0, &actual_node);
	if (!buf_id) {
		TEST_FAIL(strerror(errno));
		return;
	}

	/* Coherent cannot be NUMA-steered — actual may differ from requested.
	 * Just verify we get a valid node number back. */
	if (actual_node < 0) {
		char msg[64];

		snprintf(msg, sizeof(msg), "actual_numa_node = %d, expected >= 0",
			 actual_node);
		destroy_buffer(fd, buf_id);
		TEST_FAIL(msg);
		return;
	}

	destroy_buffer(fd, buf_id);
	TEST_PASS();
	printf("      requested=0, actual=%d (coherent cannot be steered)\n",
	       actual_node);
}

/* ── Test 6: NUMA stats via GET_BUF_STATS ──────────────────── */

static void test_numa_stats(int fd)
{
	struct dmaplane_buf_stats bstats;
	unsigned long long total;

	TEST_START("NUMA stats (via GET_BUF_STATS)");

	memset(&bstats, 0, sizeof(bstats));
	if (ioctl(fd, DMAPLANE_IOCTL_GET_BUF_STATS, &bstats) < 0) {
		TEST_FAIL(strerror(errno));
		return;
	}

	total = bstats.numa_local_allocs + bstats.numa_remote_allocs +
		bstats.numa_anon_allocs;
	if (total == 0) {
		TEST_FAIL("all NUMA counters are 0 after allocations");
		return;
	}

	TEST_PASS();
	printf("\n    Phase 5 NUMA Stats:\n");
	printf("      numa_local_allocs:  %llu\n",
	       (unsigned long long)bstats.numa_local_allocs);
	printf("      numa_remote_allocs: %llu\n",
	       (unsigned long long)bstats.numa_remote_allocs);
	printf("      numa_anon_allocs:   %llu\n",
	       (unsigned long long)bstats.numa_anon_allocs);
	printf("\n");
}

/* ── Test 7: Cross-node bandwidth benchmark ────────────────── */

static void test_numa_bench(int fd)
{
	struct dmaplane_numa_bench_params *p;
	unsigned int i, j;

	TEST_START("cross-node bandwidth benchmark");

	/* Heap-allocate — struct is ~1100 bytes */
	p = calloc(1, sizeof(*p));
	if (!p) {
		TEST_FAIL("calloc failed");
		return;
	}

	p->buffer_size = 1024 * 1024;  /* 1 MB */
	p->iterations = 100;

	if (ioctl(fd, DMAPLANE_IOCTL_NUMA_BENCH, p) < 0) {
		TEST_FAIL(strerror(errno));
		free(p);
		return;
	}

	if (p->status != 0) {
		char msg[64];

		snprintf(msg, sizeof(msg), "bench status = %d", p->status);
		TEST_FAIL(msg);
		free(p);
		return;
	}

	if (p->nr_nodes < 1) {
		TEST_FAIL("nr_nodes is 0");
		free(p);
		return;
	}

	/* Verify at least the local cell has non-zero bandwidth */
	if (p->bw_mbps[0][0] == 0) {
		TEST_FAIL("local bandwidth is 0");
		free(p);
		return;
	}

	TEST_PASS();

	if (p->nr_nodes == 1) {
		printf("      Single-node system — no cross-node penalty to measure.\n");
		printf("      Local bandwidth: %llu MB/s\n",
		       (unsigned long long)p->bw_mbps[0][0]);
	} else {
		printf("\n    Cross-Node Bandwidth Matrix (MB/s):\n");
		printf("      cpu\\buf |");
		for (j = 0; j < p->nr_nodes; j++)
			printf(" node%-2u |", j);
		printf("\n      --------");
		for (j = 0; j < p->nr_nodes; j++)
			printf("--------");
		printf("\n");
		for (i = 0; i < p->nr_nodes; i++) {
			printf("      node%-2u |", i);
			for (j = 0; j < p->nr_nodes; j++)
				printf(" %5llu  |",
				       (unsigned long long)p->bw_mbps[i][j]);
			printf("\n");
		}

		/* Penalty analysis for multi-node */
		printf("\n    Cross-Node Penalty Analysis:\n");
		for (i = 0; i < p->nr_nodes; i++) {
			for (j = 0; j < p->nr_nodes; j++) {
				if (i == j)
					continue;
				if (p->bw_mbps[i][i] > 0) {
					unsigned long long local_bw = p->bw_mbps[i][i];
					unsigned long long remote_bw = p->bw_mbps[i][j];
					int penalty = 0;

					if (local_bw > remote_bw)
						penalty = (int)(100 -
							(remote_bw * 100 / local_bw));
					printf("      node%u -> node%u: %d%% penalty "
					       "(%llu vs %llu MB/s)\n",
					       i, j, penalty,
					       (unsigned long long)remote_bw,
					       (unsigned long long)local_bw);
				}
			}
		}
	}
	printf("\n");
	free(p);
}

/* ── Test 8: Regression — buffer + mmap + pattern with NUMA_ANY ── */

static void test_regression_numa_any(int fd)
{
	struct dmaplane_mmap_info info = {0};
	unsigned int buf_id;
	int actual_node;
	void *ptr;
	uint32_t *data;
	unsigned int i;
	size_t buf_size = 64 * 1024;  /* 64 KB */
	unsigned int count = buf_size / sizeof(uint32_t);

	TEST_START("regression: buffer + mmap + pattern with NUMA_ANY");

	/* Create buffer with NUMA_ANY */
	buf_id = create_buffer(fd, DMAPLANE_BUF_TYPE_PAGES, buf_size,
			       DMAPLANE_NUMA_ANY, &actual_node);
	if (!buf_id) {
		TEST_FAIL(strerror(errno));
		return;
	}

	/* Get mmap info */
	info.buf_id = buf_id;
	if (ioctl(fd, DMAPLANE_IOCTL_GET_MMAP_INFO, &info) < 0) {
		TEST_FAIL("GET_MMAP_INFO failed");
		destroy_buffer(fd, buf_id);
		return;
	}

	/* mmap the buffer */
	ptr = mmap(NULL, info.mmap_size, PROT_READ | PROT_WRITE,
		   MAP_SHARED, fd, info.mmap_offset);
	if (ptr == MAP_FAILED) {
		TEST_FAIL("mmap failed");
		destroy_buffer(fd, buf_id);
		return;
	}

	/* Write pattern */
	data = (uint32_t *)ptr;
	for (i = 0; i < count; i++)
		data[i] = 0xDEAD0000 | i;

	/* Read back and verify */
	for (i = 0; i < count; i++) {
		if (data[i] != (0xDEAD0000 | i)) {
			char msg[64];

			snprintf(msg, sizeof(msg),
				 "mismatch at [%u]: got 0x%08x", i, data[i]);
			munmap(ptr, info.mmap_size);
			destroy_buffer(fd, buf_id);
			TEST_FAIL(msg);
			return;
		}
	}

	munmap(ptr, info.mmap_size);
	destroy_buffer(fd, buf_id);
	TEST_PASS();
}

/* ── Test 9: Allocate on node 1 (multi-socket only) ────────── */

static void test_alloc_node1(int fd)
{
	struct dmaplane_numa_topo topo;
	int actual_node = -99;
	unsigned int buf_id;

	TEST_START("allocate on node 1 (multi-socket only)");

	memset(&topo, 0, sizeof(topo));
	if (ioctl(fd, DMAPLANE_IOCTL_QUERY_NUMA_TOPO, &topo) < 0) {
		TEST_FAIL(strerror(errno));
		return;
	}

	if (topo.nr_nodes < 2) {
		printf("SKIP (single-node system)\n");
		tests_passed++;
		return;
	}

	buf_id = create_buffer(fd, DMAPLANE_BUF_TYPE_PAGES, 1 << 20,
			       1, &actual_node);
	if (!buf_id) {
		TEST_FAIL(strerror(errno));
		return;
	}

	if (actual_node != 1) {
		char msg[64];

		snprintf(msg, sizeof(msg), "actual_numa_node = %d, expected 1",
			 actual_node);
		destroy_buffer(fd, buf_id);
		TEST_FAIL(msg);
		return;
	}

	destroy_buffer(fd, buf_id);
	TEST_PASS();
	printf("      node 1 allocation: actual_numa_node = %d\n", actual_node);
}

/* ── Main ──────────────────────────────────────────────────── */

int main(void)
{
	int fd;

	printf("=== Phase 5: NUMA Topology & Optimization Tests ===\n\n");

	fd = open("/dev/dmaplane", O_RDWR);
	if (fd < 0) {
		perror("open /dev/dmaplane");
		return 1;
	}

	test_topology_query(fd);
	test_alloc_node0(fd);
	test_alloc_numa_any(fd);
	test_invalid_node(fd);
	test_coherent_numa(fd);
	test_numa_stats(fd);
	test_numa_bench(fd);
	test_regression_numa_any(fd);
	test_alloc_node1(fd);

	close(fd);

	printf("\n=== Results: %d passed, %d failed ===\n",
	       tests_passed, tests_failed);

	return tests_failed ? 1 : 0;
}
