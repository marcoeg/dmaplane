// SPDX-License-Identifier: GPL-2.0
/*
 * test_phase6_backpressure.c — Phase 6 backpressure & throughput tests
 * Copyright (c) 2026 Graziano Labs Corp.
 *
 * 8 test cases exercising credit-based flow control, sustained streaming,
 * queue depth sweep, and regression with Phase 4 benchmarks.
 *
 * Prerequisites:
 *   - dmaplane.ko loaded (sudo insmod driver/dmaplane.ko)
 *   - Soft-RoCE configured (bash scripts/setup_rxe.sh)
 *
 * Usage: sudo ./tests/test_phase6_backpressure
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

/* ── Helpers ───────────────────────────────────────────────── */

/*
 * Auto-detect rxe device by scanning /sys/class/infiniband/.
 * Returns 0 on success (name filled), -1 if not found.
 */
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
	struct dmaplane_mr_params mp;

	memset(&mp, 0, sizeof(mp));
	mp.buf_id = buf_id;
	mp.access_flags = IB_ACCESS_LOCAL_WRITE | IB_ACCESS_REMOTE_WRITE |
			  IB_ACCESS_REMOTE_READ;

	if (ioctl(fd, DMAPLANE_IOCTL_REGISTER_MR, &mp) < 0)
		return -1;
	*mr_id = mp.mr_id;
	return 0;
}

static int setup_rdma(int fd, const char *rxe_name)
{
	struct dmaplane_rdma_setup setup;

	memset(&setup, 0, sizeof(setup));
	snprintf(setup.ib_dev_name, sizeof(setup.ib_dev_name), "%s", rxe_name);
	setup.port = 1;

	return ioctl(fd, DMAPLANE_IOCTL_SETUP_RDMA, &setup);
}

/* ── Test 1: Configure flow control ───────────────────────── */

static void test_configure_flow(int fd)
{
	struct dmaplane_flow_params p;
	struct dmaplane_flow_stats fs;

	TEST_START("configure flow control");

	memset(&p, 0, sizeof(p));
	p.max_credits = 64;
	p.high_watermark = 48;
	p.low_watermark = 16;

	if (ioctl(fd, DMAPLANE_IOCTL_CONFIGURE_FLOW, &p) < 0) {
		TEST_FAIL(strerror(errno));
		return;
	}

	if (p.status != 0) {
		TEST_FAIL("status != 0");
		return;
	}

	/* Verify stats are all zero after fresh configuration */
	memset(&fs, 0, sizeof(fs));
	if (ioctl(fd, DMAPLANE_IOCTL_GET_FLOW_STATS, &fs) < 0) {
		TEST_FAIL("GET_FLOW_STATS failed");
		return;
	}

	if (fs.cq_overflows != 0) {
		TEST_FAIL("cq_overflows not 0 after configure");
		return;
	}

	TEST_PASS();
	printf("      credits=64 high=48 low=16\n");
}

/* ── Test 2: Invalid flow control parameters ──────────────── */

static void test_invalid_flow_params(int fd)
{
	struct dmaplane_flow_params p;
	int ret;

	TEST_START("invalid flow control parameters");

	/* high > max */
	memset(&p, 0, sizeof(p));
	p.max_credits = 64;
	p.high_watermark = 65;
	p.low_watermark = 16;
	ret = ioctl(fd, DMAPLANE_IOCTL_CONFIGURE_FLOW, &p);
	if (ret == 0) {
		TEST_FAIL("high > max should fail");
		return;
	}

	/* low >= high */
	memset(&p, 0, sizeof(p));
	p.max_credits = 64;
	p.high_watermark = 48;
	p.low_watermark = 48;
	ret = ioctl(fd, DMAPLANE_IOCTL_CONFIGURE_FLOW, &p);
	if (ret == 0) {
		TEST_FAIL("low >= high should fail");
		return;
	}

	/* max > 128 */
	memset(&p, 0, sizeof(p));
	p.max_credits = 129;
	p.high_watermark = 96;
	p.low_watermark = 32;
	ret = ioctl(fd, DMAPLANE_IOCTL_CONFIGURE_FLOW, &p);
	if (ret == 0) {
		TEST_FAIL("max > 128 should fail");
		return;
	}

	/* max = 0 */
	memset(&p, 0, sizeof(p));
	p.max_credits = 0;
	p.high_watermark = 0;
	p.low_watermark = 0;
	ret = ioctl(fd, DMAPLANE_IOCTL_CONFIGURE_FLOW, &p);
	if (ret == 0) {
		TEST_FAIL("max = 0 should fail");
		return;
	}

	TEST_PASS();
}

/* ── Test 3: Sustained streaming (10 seconds) ─────────────── */

static void test_sustained_streaming(int fd, __u32 mr_id)
{
	struct dmaplane_flow_params fp;
	struct dmaplane_sustained_params sp;

	TEST_START("sustained streaming (10 seconds)");

	/* Configure flow control first */
	memset(&fp, 0, sizeof(fp));
	fp.max_credits = 64;
	fp.high_watermark = 48;
	fp.low_watermark = 16;
	if (ioctl(fd, DMAPLANE_IOCTL_CONFIGURE_FLOW, &fp) < 0) {
		TEST_FAIL("CONFIGURE_FLOW failed");
		return;
	}

	memset(&sp, 0, sizeof(sp));
	sp.mr_id = mr_id;
	sp.msg_size = 4096;
	sp.queue_depth = 8;
	sp.duration_secs = 10;

	if (ioctl(fd, DMAPLANE_IOCTL_SUSTAINED_STREAM, &sp) < 0) {
		char msg[64];

		snprintf(msg, sizeof(msg), "ioctl failed: %s",
			 strerror(errno));
		TEST_FAIL(msg);
		return;
	}

	if (sp.status != 0) {
		TEST_FAIL("status != 0");
		return;
	}

	if (sp.total_bytes == 0) {
		TEST_FAIL("total_bytes is 0");
		return;
	}

	if (sp.total_ops == 0) {
		TEST_FAIL("total_ops is 0");
		return;
	}

	if (sp.cq_overflow_count != 0) {
		char msg[64];

		snprintf(msg, sizeof(msg), "cq_overflow_count = %llu",
			 (unsigned long long)sp.cq_overflow_count);
		TEST_FAIL(msg);
		return;
	}

	if (sp.avg_throughput_mbps == 0) {
		TEST_FAIL("avg_throughput is 0");
		return;
	}

	if (sp.min_window_mbps == 0) {
		TEST_FAIL("min_window is 0 (dead seconds)");
		return;
	}

	TEST_PASS();
	printf("      Duration: 10 seconds\n");
	printf("      Total: %llu MB transferred, %llu ops\n",
	       (unsigned long long)(sp.total_bytes >> 20),
	       (unsigned long long)sp.total_ops);
	printf("      Throughput: avg=%llu MB/s, min_window=%llu MB/s, "
	       "max_window=%llu MB/s\n",
	       (unsigned long long)sp.avg_throughput_mbps,
	       (unsigned long long)sp.min_window_mbps,
	       (unsigned long long)sp.max_window_mbps);
	printf("      Stalls: %llu   CQ overflows: 0\n",
	       (unsigned long long)sp.stall_count);
}

/* ── Test 4: Sustained without RDMA (must fail) ──────────── */

static void test_sustained_no_rdma(int fd_no_rdma)
{
	struct dmaplane_sustained_params sp;

	TEST_START("sustained without RDMA (must fail)");

	memset(&sp, 0, sizeof(sp));
	sp.mr_id = 1;
	sp.msg_size = 4096;
	sp.queue_depth = 8;
	sp.duration_secs = 5;

	if (ioctl(fd_no_rdma, DMAPLANE_IOCTL_SUSTAINED_STREAM, &sp) == 0) {
		TEST_FAIL("expected failure, got success");
		return;
	}

	if (errno != ENODEV) {
		char msg[64];

		snprintf(msg, sizeof(msg), "expected ENODEV, got %s",
			 strerror(errno));
		TEST_FAIL(msg);
		return;
	}

	TEST_PASS();
}

/* ── Test 5: Credit stall detection ──────────────────────── */

static void test_credit_stall(int fd, __u32 mr_id)
{
	struct dmaplane_flow_params fp;
	struct dmaplane_sustained_params sp;

	TEST_START("credit stall detection (tight credits)");

	/* Configure very tight credits */
	memset(&fp, 0, sizeof(fp));
	fp.max_credits = 4;
	fp.high_watermark = 3;
	fp.low_watermark = 1;
	if (ioctl(fd, DMAPLANE_IOCTL_CONFIGURE_FLOW, &fp) < 0) {
		TEST_FAIL("CONFIGURE_FLOW failed");
		return;
	}

	memset(&sp, 0, sizeof(sp));
	sp.mr_id = mr_id;
	sp.msg_size = 4096;
	sp.queue_depth = 4;
	sp.duration_secs = 5;

	if (ioctl(fd, DMAPLANE_IOCTL_SUSTAINED_STREAM, &sp) < 0) {
		char msg[64];

		snprintf(msg, sizeof(msg), "ioctl failed: %s",
			 strerror(errno));
		TEST_FAIL(msg);
		return;
	}

	if (sp.stall_count == 0) {
		TEST_FAIL("stall_count is 0 with tight credits");
		return;
	}

	if (sp.cq_overflow_count != 0) {
		char msg[64];

		snprintf(msg, sizeof(msg), "cq_overflow_count = %llu",
			 (unsigned long long)sp.cq_overflow_count);
		TEST_FAIL(msg);
		return;
	}

	TEST_PASS();
	printf("      Stalls: %llu (expected >0 with max_credits=4)\n",
	       (unsigned long long)sp.stall_count);
	printf("      CQ overflows: 0 (flow control prevented overflow)\n");
}

/* ── Test 6: Queue depth sweep ───────────────────────────── */

static void test_qdepth_sweep(int fd, __u32 mr_id)
{
	struct dmaplane_sweep_params sp;
	unsigned int i;

	TEST_START("queue depth sweep");

	memset(&sp, 0, sizeof(sp));
	sp.mr_id = mr_id;
	sp.msg_size = 4096;
	sp.iterations = 500;
	sp.min_qdepth = 1;
	sp.max_qdepth = 32;
	sp.step = 4;

	if (ioctl(fd, DMAPLANE_IOCTL_QDEPTH_SWEEP, &sp) < 0) {
		char msg[64];

		snprintf(msg, sizeof(msg), "ioctl failed: %s",
			 strerror(errno));
		TEST_FAIL(msg);
		return;
	}

	if (sp.status != 0) {
		char msg[64];

		snprintf(msg, sizeof(msg), "status = %u", sp.status);
		TEST_FAIL(msg);
		return;
	}

	/* nr_points = (32 - 1) / 4 + 1 = 8 */
	if (sp.nr_points != 8) {
		char msg[64];

		snprintf(msg, sizeof(msg), "nr_points = %u, expected 8",
			 sp.nr_points);
		TEST_FAIL(msg);
		return;
	}

	/* All throughput values should be > 0 */
	for (i = 0; i < sp.nr_points; i++) {
		if (sp.throughput_mbps[i] == 0) {
			char msg[64];

			snprintf(msg, sizeof(msg), "throughput[%u] is 0", i);
			TEST_FAIL(msg);
			return;
		}
	}

	TEST_PASS();
	printf("\n      QD Sweep (msg_size=4096, iterations=500):\n");
	for (i = 0; i < sp.nr_points; i++) {
		unsigned int qd = sp.min_qdepth + i * sp.step;

		printf("        QD=%2u:  %5llu MB/s  avg_lat=%6llu ns  "
		       "p99=%6llu ns\n",
		       qd,
		       (unsigned long long)sp.throughput_mbps[i],
		       (unsigned long long)sp.avg_latency_ns[i],
		       (unsigned long long)sp.p99_latency_ns[i]);
	}
	if (sp.saturation_qdepth > 0)
		printf("      Saturation: QD=%u\n", sp.saturation_qdepth);
	else
		printf("      No saturation within range (throughput still climbing)\n");
}

/* ── Test 7: Flow stats accumulation ─────────────────────── */

static void test_flow_stats(int fd)
{
	struct dmaplane_flow_stats fs;

	TEST_START("flow stats accumulation");

	memset(&fs, 0, sizeof(fs));
	if (ioctl(fd, DMAPLANE_IOCTL_GET_FLOW_STATS, &fs) < 0) {
		TEST_FAIL(strerror(errno));
		return;
	}

	if (fs.total_sustained_bytes == 0) {
		TEST_FAIL("total_sustained_bytes is 0");
		return;
	}

	if (fs.total_sustained_ops == 0) {
		TEST_FAIL("total_sustained_ops is 0");
		return;
	}

	if (fs.cq_overflows != 0) {
		char msg[64];

		snprintf(msg, sizeof(msg), "cq_overflows = %llu",
			 (unsigned long long)fs.cq_overflows);
		TEST_FAIL(msg);
		return;
	}

	TEST_PASS();
	printf("      credit_stalls: %llu\n",
	       (unsigned long long)fs.credit_stalls);
	printf("      high_watermark_events: %llu\n",
	       (unsigned long long)fs.high_watermark_events);
	printf("      low_watermark_events: %llu\n",
	       (unsigned long long)fs.low_watermark_events);
	printf("      cq_overflows: 0\n");
	printf("      total_sustained_bytes: %llu\n",
	       (unsigned long long)fs.total_sustained_bytes);
	printf("      total_sustained_ops: %llu\n",
	       (unsigned long long)fs.total_sustained_ops);
}

/* ── Test 8: Regression — Phase 4 benchmarks still work ──── */

static void test_phase4_regression(int fd, __u32 mr_id)
{
	struct dmaplane_loopback_params lp;
	struct dmaplane_bench_params bp;

	TEST_START("regression: Phase 4 benchmarks still work");

	/* Loopback test */
	memset(&lp, 0, sizeof(lp));
	lp.mr_id = mr_id;
	lp.size = 4096;
	if (ioctl(fd, DMAPLANE_IOCTL_LOOPBACK_TEST, &lp) < 0) {
		char msg[64];

		snprintf(msg, sizeof(msg), "loopback failed: %s",
			 strerror(errno));
		TEST_FAIL(msg);
		return;
	}

	/* Ping-pong */
	memset(&bp, 0, sizeof(bp));
	bp.mr_id = mr_id;
	bp.msg_size = 4096;
	bp.iterations = 100;
	if (ioctl(fd, DMAPLANE_IOCTL_PINGPONG_BENCH, &bp) < 0) {
		char msg[64];

		snprintf(msg, sizeof(msg), "pingpong failed: %s",
			 strerror(errno));
		TEST_FAIL(msg);
		return;
	}

	/* Streaming */
	memset(&bp, 0, sizeof(bp));
	bp.mr_id = mr_id;
	bp.msg_size = 4096;
	bp.iterations = 100;
	bp.queue_depth = 8;
	if (ioctl(fd, DMAPLANE_IOCTL_STREAMING_BENCH, &bp) < 0) {
		char msg[64];

		snprintf(msg, sizeof(msg), "streaming failed: %s",
			 strerror(errno));
		TEST_FAIL(msg);
		return;
	}

	TEST_PASS();
}

/* ── Main ──────────────────────────────────────────────────── */

int main(void)
{
	char rxe_name[64];
	int fd, fd_no_rdma;
	__u32 buf_id, mr_id;

	printf("=== Phase 6: Backpressure & Throughput Modeling Tests ===\n\n");

	fd = open(dev_path, O_RDWR);
	if (fd < 0) {
		perror("open /dev/dmaplane");
		return 1;
	}

	/* Tests 1-2: Flow control configuration (no RDMA needed) */
	test_configure_flow(fd);
	test_invalid_flow_params(fd);

	/* Test 4: Sustained without RDMA — use a separate fd that
	 * doesn't have RDMA set up.  Must run BEFORE RDMA setup. */
	fd_no_rdma = open(dev_path, O_RDWR);
	if (fd_no_rdma < 0) {
		perror("open /dev/dmaplane (second fd)");
		close(fd);
		return 1;
	}
	test_sustained_no_rdma(fd_no_rdma);
	close(fd_no_rdma);

	/* Set up RDMA for tests 3, 5, 6, 7, 8 */
	if (find_rxe_device(rxe_name, sizeof(rxe_name)) < 0) {
		printf("ERROR: No rxe device found. Run scripts/setup_rxe.sh first.\n");
		close(fd);
		return 1;
	}
	printf("    Using IB device: %s\n\n", rxe_name);

	if (setup_rdma(fd, rxe_name) < 0) {
		printf("ERROR: RDMA setup failed: %s\n", strerror(errno));
		close(fd);
		return 1;
	}

	/* Create buffer and register MR */
	if (create_page_buffer(fd, 1 << 20, &buf_id) < 0) {
		printf("ERROR: Buffer creation failed: %s\n", strerror(errno));
		close(fd);
		return 1;
	}

	if (register_mr(fd, buf_id, &mr_id) < 0) {
		printf("ERROR: MR registration failed: %s\n", strerror(errno));
		close(fd);
		return 1;
	}

	/* Tests that require RDMA */
	test_sustained_streaming(fd, mr_id);
	test_credit_stall(fd, mr_id);
	test_qdepth_sweep(fd, mr_id);
	test_flow_stats(fd);
	test_phase4_regression(fd, mr_id);

	/* Clean up: deregister MR, destroy buffer, teardown RDMA.
	 * These are device-global (not per-fd), so they survive close().
	 * Without cleanup, subsequent test phases (2, 4) fail because
	 * buffer slots and RDMA context are still occupied. */
	{
		__u32 id = mr_id;

		ioctl(fd, DMAPLANE_IOCTL_DEREGISTER_MR, &id);
	}
	{
		__u32 id = buf_id;

		ioctl(fd, DMAPLANE_IOCTL_DESTROY_BUFFER, &id);
	}
	ioctl(fd, DMAPLANE_IOCTL_TEARDOWN_RDMA, 0);

	close(fd);

	printf("\n=== Results: %d passed, %d failed ===\n", passed, failed);

	return failed ? 1 : 0;
}
