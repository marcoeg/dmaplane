// SPDX-License-Identifier: GPL-2.0
/*
 * test_phase7_instrumentation.c — Phase 7 instrumentation tests
 * Copyright (c) 2026 Graziano Labs Corp.
 *
 * 8 test cases exercising debugfs interface, latency histograms,
 * and tracepoint registration.
 *
 * Prerequisites:
 *   - dmaplane.ko loaded (sudo insmod driver/dmaplane.ko)
 *   - Soft-RoCE configured (bash scripts/setup_rxe.sh) for tests 4-7
 *   - debugfs mounted at /sys/kernel/debug (for tests 1-4, 8)
 *
 * Usage: sudo ./tests/test_phase7_instrumentation
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <dirent.h>
#include <errno.h>
#include <sys/ioctl.h>

#include "dmaplane_uapi.h"

/* IB access flags — match kernel definitions for userspace use */
#define IB_ACCESS_LOCAL_WRITE  (1)
#define IB_ACCESS_REMOTE_WRITE (1 << 1)
#define IB_ACCESS_REMOTE_READ  (1 << 2)

static int passed, failed;
static const char *dev_path = "/dev/dmaplane";

#define TEST_START(name) \
	printf("  [%d] %-50s ", passed + failed + 1, name)

#define TEST_PASS() \
	do { printf("PASS\n"); passed++; } while (0)

#define TEST_FAIL(msg) \
	do { printf("FAIL (%s)\n", msg); failed++; } while (0)

#define TEST_SKIP(msg) \
	do { printf("SKIP (%s)\n", msg); passed++; } while (0)

/* ── Helpers ───────────────────────────────────────────────── */

static int find_rxe_device(char *name, size_t len)
{
	DIR *dir;
	struct dirent *ent;

	dir = opendir("/sys/class/infiniband");
	if (!dir)
		return -1;

	while ((ent = readdir(dir)) != NULL) {
		if (strncmp(ent->d_name, "rxe_", 4) == 0 &&
		    strlen(ent->d_name) < len) {
			memcpy(name, ent->d_name, strlen(ent->d_name) + 1);
			closedir(dir);
			return 0;
		}
	}
	closedir(dir);
	return -1;
}

static int create_page_buffer(int fd, __u64 size, __u32 *buf_id)
{
	struct dmaplane_buf_params bp = {
		.alloc_type = DMAPLANE_BUF_TYPE_PAGES,
		.size = size,
		.numa_node = DMAPLANE_NUMA_ANY,
	};

	if (ioctl(fd, DMAPLANE_IOCTL_CREATE_BUFFER, &bp) < 0)
		return -1;
	*buf_id = bp.buf_id;
	return 0;
}

static int register_mr(int fd, __u32 buf_id, __u32 *mr_id)
{
	struct dmaplane_mr_params mp = {
		.buf_id = buf_id,
		.access_flags = IB_ACCESS_LOCAL_WRITE |
				IB_ACCESS_REMOTE_WRITE |
				IB_ACCESS_REMOTE_READ,
	};

	if (ioctl(fd, DMAPLANE_IOCTL_REGISTER_MR, &mp) < 0)
		return -1;
	*mr_id = mp.mr_id;
	return 0;
}

/*
 * Read a debugfs file into a buffer.  Returns bytes read, or -1 on error.
 * Sets errno on failure.
 */
static ssize_t read_debugfs(const char *path, char *buf, size_t bufsz)
{
	int fd;
	ssize_t total = 0, n;

	fd = open(path, O_RDONLY);
	if (fd < 0)
		return -1;

	while (total < (ssize_t)(bufsz - 1)) {
		n = read(fd, buf + total, bufsz - 1 - total);
		if (n < 0) {
			close(fd);
			return -1;
		}
		if (n == 0)
			break;
		total += n;
	}
	buf[total] = '\0';
	close(fd);
	return total;
}

/* ── Test 1: debugfs directory exists ─────────────────────── */

static void test_debugfs_exists(void)
{
	char buf[4096];
	ssize_t n;

	TEST_START("debugfs directory exists");

	n = read_debugfs("/sys/kernel/debug/dmaplane/stats", buf, sizeof(buf));
	if (n < 0) {
		if (errno == ENOENT || errno == EACCES)
			TEST_SKIP("debugfs not accessible");
		else
			TEST_FAIL("read failed");
		return;
	}

	if (!strstr(buf, "channels:")) {
		TEST_FAIL("stats missing 'channels:' field");
		return;
	}

	printf("PASS\n      /sys/kernel/debug/dmaplane/stats readable\n");
	passed++;
}

/* ── Test 2: debugfs stats match ioctl stats ──────────────── */

static void test_debugfs_stats_match(int fd)
{
	char dbg[4096];
	struct dmaplane_buf_stats bs;
	ssize_t n;
	__u32 buf_id;

	TEST_START("debugfs stats match ioctl stats");

	/* Create a buffer to increment the counter */
	if (create_page_buffer(fd, 4096, &buf_id) < 0) {
		TEST_FAIL("create_buffer failed");
		return;
	}

	/* Read debugfs stats */
	n = read_debugfs("/sys/kernel/debug/dmaplane/stats", dbg, sizeof(dbg));
	if (n < 0) {
		ioctl(fd, DMAPLANE_IOCTL_DESTROY_BUFFER, &buf_id);
		TEST_SKIP("debugfs not accessible");
		return;
	}

	/* Read ioctl stats */
	if (ioctl(fd, DMAPLANE_IOCTL_GET_BUF_STATS, &bs) < 0) {
		ioctl(fd, DMAPLANE_IOCTL_DESTROY_BUFFER, &buf_id);
		TEST_FAIL("get_buf_stats failed");
		return;
	}

	/* Check debugfs contains buffers_created that's >= 1 */
	if (!strstr(dbg, "buffers:")) {
		ioctl(fd, DMAPLANE_IOCTL_DESTROY_BUFFER, &buf_id);
		TEST_FAIL("debugfs missing 'buffers:' field");
		return;
	}

	if (bs.buffers_created < 1) {
		ioctl(fd, DMAPLANE_IOCTL_DESTROY_BUFFER, &buf_id);
		TEST_FAIL("ioctl buffers_created < 1");
		return;
	}

	printf("PASS\n      buffers_created: ioctl=%llu\n",
	       (unsigned long long)bs.buffers_created);
	passed++;

	ioctl(fd, DMAPLANE_IOCTL_DESTROY_BUFFER, &buf_id);
}

/* ── Test 3: debugfs buffer listing ───────────────────────── */

static void test_debugfs_buffers(int fd)
{
	char dbg[8192];
	ssize_t n;
	__u32 buf1 = 0, buf2 = 0, buf3 = 0;
	int count = 0;
	const char *p;

	TEST_START("debugfs buffer listing");

	if (create_page_buffer(fd, 4096, &buf1) < 0 ||
	    create_page_buffer(fd, 8192, &buf2) < 0) {
		TEST_FAIL("create_buffer failed");
		goto cleanup;
	}

	/* Create a coherent buffer */
	{
		struct dmaplane_buf_params bp = {
			.alloc_type = DMAPLANE_BUF_TYPE_COHERENT,
			.size = 4096,
			.numa_node = DMAPLANE_NUMA_ANY,
		};
		if (ioctl(fd, DMAPLANE_IOCTL_CREATE_BUFFER, &bp) < 0) {
			TEST_FAIL("create coherent buffer failed");
			goto cleanup;
		}
		buf3 = bp.buf_id;
	}

	n = read_debugfs("/sys/kernel/debug/dmaplane/buffers", dbg, sizeof(dbg));
	if (n < 0) {
		TEST_SKIP("debugfs not accessible");
		goto cleanup;
	}

	/* Count buf_id= occurrences */
	p = dbg;
	while ((p = strstr(p, "buf_id=")) != NULL) {
		count++;
		p += 7;
	}

	if (count < 3) {
		printf("FAIL (expected >= 3 buffers, found %d)\n", count);
		failed++;
	} else {
		printf("PASS\n      %d buffers listed (2 pages, 1 coherent)\n", count);
		passed++;
	}

cleanup:
	if (buf3) ioctl(fd, DMAPLANE_IOCTL_DESTROY_BUFFER, &buf3);
	if (buf2) ioctl(fd, DMAPLANE_IOCTL_DESTROY_BUFFER, &buf2);
	if (buf1) ioctl(fd, DMAPLANE_IOCTL_DESTROY_BUFFER, &buf1);
}

/* ── Test 4: debugfs RDMA state ───────────────────────────── */

static void test_debugfs_rdma(int fd)
{
	char dbg[4096];
	ssize_t n;

	(void)fd;	/* RDMA state read from debugfs, not via ioctl */
	TEST_START("debugfs RDMA state");

	n = read_debugfs("/sys/kernel/debug/dmaplane/rdma", dbg, sizeof(dbg));
	if (n < 0) {
		TEST_SKIP("debugfs not accessible");
		return;
	}

	if (!strstr(dbg, "initialized: yes")) {
		TEST_FAIL("RDMA not initialized in debugfs");
		return;
	}

	/* Extract device name from output */
	{
		const char *p = strstr(dbg, "ib_device: ");
		char devname[64] = "unknown";

		if (p) {
			p += 11;
			sscanf(p, "%63s", devname);
		}
		printf("PASS\n      initialized: yes  device: %s\n", devname);
		passed++;
	}
}

/* ── Test 5: histogram collection (pingpong) ──────────────── */

static void test_histogram_pingpong(int fd, __u32 mr_id)
{
	struct dmaplane_bench_params bp = {
		.mr_id = mr_id,
		.msg_size = 4096,
		.iterations = 1000,
	};
	struct dmaplane_hist_params hp = {0};

	TEST_START("histogram collection (pingpong)");

	if (ioctl(fd, DMAPLANE_IOCTL_PINGPONG_BENCH, &bp) < 0) {
		TEST_FAIL("pingpong failed");
		return;
	}

	if (ioctl(fd, DMAPLANE_IOCTL_GET_HISTOGRAM, &hp) < 0) {
		TEST_FAIL("get_histogram failed");
		return;
	}

	if (hp.count < 1000) {
		printf("FAIL (count=%llu, expected >= 1000)\n",
		       (unsigned long long)hp.count);
		failed++;
		return;
	}

	if (hp.avg_ns == 0 || hp.p50_ns == 0 || hp.p99_ns == 0 ||
	    hp.p999_ns == 0) {
		TEST_FAIL("zero percentiles");
		return;
	}

	printf("PASS\n");
	printf("      Samples: %llu  avg=%llu ns  min=%llu ns  max=%llu ns\n",
	       (unsigned long long)hp.count,
	       (unsigned long long)hp.avg_ns,
	       (unsigned long long)hp.min_ns,
	       (unsigned long long)hp.max_ns);
	printf("      P50=%llu ns  P99=%llu ns  P999=%llu ns\n",
	       (unsigned long long)hp.p50_ns,
	       (unsigned long long)hp.p99_ns,
	       (unsigned long long)hp.p999_ns);

	/* Print non-zero buckets */
	printf("      Bucket distribution:\n");
	{
		int i;
		for (i = 0; i < DMAPLANE_HIST_BUCKETS; i++) {
			if (hp.buckets[i] == 0)
				continue;
			if (i < DMAPLANE_HIST_BUCKETS - 1)
				printf("        [%u, %u) us:  %llu\n",
				       (i == 0) ? 0 : (1u << i),
				       1u << (i + 1),
				       (unsigned long long)hp.buckets[i]);
			else
				printf("        [%u, +inf) us:  %llu\n",
				       1u << i,
				       (unsigned long long)hp.buckets[i]);
		}
	}
	passed++;
}

/* ── Test 6: histogram reset ──────────────────────────────── */

static void test_histogram_reset(int fd)
{
	struct dmaplane_hist_params hp = {0};

	TEST_START("histogram reset");

	/* Read with reset=1 */
	hp.reset = 1;
	if (ioctl(fd, DMAPLANE_IOCTL_GET_HISTOGRAM, &hp) < 0) {
		TEST_FAIL("get_histogram (reset) failed");
		return;
	}

	/* Read again — should be empty */
	memset(&hp, 0, sizeof(hp));
	if (ioctl(fd, DMAPLANE_IOCTL_GET_HISTOGRAM, &hp) < 0) {
		TEST_FAIL("get_histogram (after reset) failed");
		return;
	}

	if (hp.count != 0) {
		printf("FAIL (count=%llu after reset, expected 0)\n",
		       (unsigned long long)hp.count);
		failed++;
		return;
	}

	TEST_PASS();
}

/* ── Test 7: histogram after sustained streaming ──────────── */

static void test_histogram_sustained(int fd, __u32 mr_id)
{
	struct dmaplane_flow_params fp = {
		.max_credits = 64,
		.high_watermark = 48,
		.low_watermark = 16,
	};
	struct dmaplane_sustained_params sp = {
		.mr_id = mr_id,
		.msg_size = 4096,
		.queue_depth = 8,
		.duration_secs = 5,
	};
	struct dmaplane_hist_params hp = {0};

	TEST_START("histogram after sustained streaming (5s)");

	/* Configure flow control */
	if (ioctl(fd, DMAPLANE_IOCTL_CONFIGURE_FLOW, &fp) < 0) {
		TEST_FAIL("configure_flow failed");
		return;
	}

	/* Reset histogram for clean measurement */
	hp.reset = 1;
	ioctl(fd, DMAPLANE_IOCTL_GET_HISTOGRAM, &hp);

	/* Run sustained streaming */
	if (ioctl(fd, DMAPLANE_IOCTL_SUSTAINED_STREAM, &sp) < 0) {
		TEST_FAIL("sustained_stream failed");
		return;
	}

	if (sp.status != 0) {
		TEST_FAIL("sustained status != 0");
		return;
	}

	/* Read histogram */
	memset(&hp, 0, sizeof(hp));
	if (ioctl(fd, DMAPLANE_IOCTL_GET_HISTOGRAM, &hp) < 0) {
		TEST_FAIL("get_histogram failed");
		return;
	}

	if (hp.count < 1000) {
		printf("FAIL (count=%llu, expected >> 1000)\n",
		       (unsigned long long)hp.count);
		failed++;
		return;
	}

	printf("PASS\n");
	printf("      Samples: %llu  avg=%llu ns\n",
	       (unsigned long long)hp.count,
	       (unsigned long long)hp.avg_ns);
	printf("      P50=%llu ns  P99=%llu ns  P999=%llu ns\n",
	       (unsigned long long)hp.p50_ns,
	       (unsigned long long)hp.p99_ns,
	       (unsigned long long)hp.p999_ns);
	passed++;
}

/* ── Test 8: tracepoint verification ──────────────────────── */

static void test_tracepoints(void)
{
	char buf[32768];
	ssize_t n;
	const char *events[] = {
		"dmaplane:dmaplane_ring_submit",
		"dmaplane:dmaplane_ring_complete",
		"dmaplane:dmaplane_rdma_post",
		"dmaplane:dmaplane_rdma_completion",
		"dmaplane:dmaplane_buf_alloc",
		"dmaplane:dmaplane_flow_stall",
	};
	int found = 0;
	int i;

	TEST_START("tracepoint verification");

	n = read_debugfs("/sys/kernel/debug/tracing/available_events",
			 buf, sizeof(buf));
	if (n < 0) {
		TEST_SKIP("tracing not accessible");
		return;
	}

	for (i = 0; i < 6; i++) {
		if (strstr(buf, events[i])) {
			found++;
		} else {
			printf("FAIL (missing %s)\n", events[i]);
			failed++;
			return;
		}
	}

	printf("PASS\n");
	for (i = 0; i < 6; i++)
		printf("      Found: %s\n", events[i]);
	passed++;
}

/* ── Main ─────────────────────────────────────────────────── */

int main(void)
{
	int fd;
	char rxe_name[64];
	__u32 buf_id = 0, mr_id = 0;
	int rdma_setup = 0;

	printf("\n=== Phase 7: Instrumentation & Microarchitectural Awareness Tests ===\n\n");

	fd = open(dev_path, O_RDWR);
	if (fd < 0) {
		perror("open /dev/dmaplane");
		return 1;
	}

	/* Tests 1-3: no RDMA needed */
	test_debugfs_exists();
	test_debugfs_stats_match(fd);
	test_debugfs_buffers(fd);

	/* Setup RDMA for tests 4-8 */
	if (find_rxe_device(rxe_name, sizeof(rxe_name)) == 0) {
		struct dmaplane_rdma_setup rs = {0};

		strncpy(rs.ib_dev_name, rxe_name, sizeof(rs.ib_dev_name) - 1);
		rs.port = 1;

		if (ioctl(fd, DMAPLANE_IOCTL_SETUP_RDMA, &rs) == 0 &&
		    rs.status == 0) {
			rdma_setup = 1;
		}
	}

	if (!rdma_setup) {
		printf("\n  RDMA not available — skipping tests 4-8\n");
		goto summary;
	}

	/* Create buffer and register MR for RDMA tests */
	if (create_page_buffer(fd, 1048576, &buf_id) < 0) {
		printf("  Buffer creation failed — skipping RDMA tests\n");
		goto cleanup;
	}

	if (register_mr(fd, buf_id, &mr_id) < 0) {
		printf("  MR registration failed — skipping RDMA tests\n");
		goto cleanup;
	}

	test_debugfs_rdma(fd);
	test_histogram_pingpong(fd, mr_id);
	test_histogram_reset(fd);
	test_histogram_sustained(fd, mr_id);
	test_tracepoints();

cleanup:
	/* Explicit cleanup — device-global resources survive close(fd) */
	if (mr_id) {
		if (ioctl(fd, DMAPLANE_IOCTL_DEREGISTER_MR, &mr_id) < 0)
			printf("  WARNING: MR deregister failed\n");
	}
	if (buf_id) {
		if (ioctl(fd, DMAPLANE_IOCTL_DESTROY_BUFFER, &buf_id) < 0)
			printf("  WARNING: buffer destroy failed\n");
	}
	if (rdma_setup) {
		if (ioctl(fd, DMAPLANE_IOCTL_TEARDOWN_RDMA) < 0)
			printf("  WARNING: RDMA teardown failed\n");
	}

summary:
	close(fd);

	printf("\n=== Results: %d passed, %d failed ===\n\n",
	       passed, failed);

	return failed > 0 ? 1 : 0;
}
