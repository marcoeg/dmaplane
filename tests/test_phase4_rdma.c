// SPDX-License-Identifier: GPL-2.0
/*
 * test_phase4_rdma.c — Phase 4 RDMA integration tests
 * Copyright (c) 2026 Graziano Labs Corp.
 *
 * Tests: RDMA setup/teardown, MR registration, loopback, ping-pong,
 * streaming benchmark, RDMA stats, re-initialization.
 *
 * Prerequisites:
 *   - dmaplane.ko loaded (sudo insmod driver/dmaplane.ko)
 *   - Soft-RoCE configured (bash scripts/setup_rxe.sh)
 *
 * Usage: sudo ./tests/test_phase4_rdma
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <dirent.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include "dmaplane_uapi.h"

/* IB access flags — match kernel definitions for userspace use */
#define IB_ACCESS_LOCAL_WRITE  (1)
#define IB_ACCESS_REMOTE_WRITE (1 << 1)
#define IB_ACCESS_REMOTE_READ  (1 << 2)

static int passed, failed;
static const char *dev_path = "/dev/dmaplane";

/* Results saved for final summary */
static __u64 loopback_latency_ns;
static __u64 pingpong_avg_ns;
static __u64 pingpong_p99_ns;
static __u64 streaming_throughput_mbps;

#define TEST_START(name) \
	printf("  TEST %d: %-50s ", passed + failed + 1, name)

#define TEST_PASS() \
	do { printf("PASS\n"); passed++; } while (0)

#define TEST_FAIL(msg) \
	do { printf("FAIL (%s)\n", msg); failed++; } while (0)

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

/*
 * Create a page-backed buffer and return its buf_id.
 * Returns -1 on failure.
 */
static int create_page_buffer(int fd, __u64 size, __u32 *buf_id)
{
	struct dmaplane_buf_params bp = {
		.alloc_type = DMAPLANE_BUF_TYPE_PAGES,
		.size = size,
		.numa_node = DMAPLANE_NUMA_ANY,
	};
	int ret;

	ret = ioctl(fd, DMAPLANE_IOCTL_CREATE_BUFFER, &bp);
	if (ret < 0)
		return -1;
	*buf_id = bp.buf_id;
	return 0;
}

/*
 * Test 1: RDMA setup with auto-detected rxe device
 */
static void test_rdma_setup(int fd, const char *rxe_name)
{
	struct dmaplane_rdma_setup setup;
	int ret;

	TEST_START("RDMA setup");

	memset(&setup, 0, sizeof(setup));
	snprintf(setup.ib_dev_name, sizeof(setup.ib_dev_name), "%s", rxe_name);
	setup.port = 1;

	ret = ioctl(fd, DMAPLANE_IOCTL_SETUP_RDMA, &setup);
	if (ret < 0) {
		char msg[64];
		snprintf(msg, sizeof(msg), "ioctl failed: %s", strerror(errno));
		TEST_FAIL(msg);
		return;
	}
	if (setup.status != 0) {
		TEST_FAIL("status != 0");
		return;
	}
	TEST_PASS();
}

/*
 * Test 2: Double setup must fail with EBUSY
 */
static void test_double_setup(int fd, const char *rxe_name)
{
	struct dmaplane_rdma_setup setup;
	int ret;

	TEST_START("Double RDMA setup (must fail EBUSY)");

	memset(&setup, 0, sizeof(setup));
	snprintf(setup.ib_dev_name, sizeof(setup.ib_dev_name), "%s", rxe_name);
	setup.port = 1;

	ret = ioctl(fd, DMAPLANE_IOCTL_SETUP_RDMA, &setup);
	if (ret == 0) {
		TEST_FAIL("second setup succeeded (should fail EBUSY)");
		return;
	}
	if (errno != EBUSY) {
		char msg[64];
		snprintf(msg, sizeof(msg), "expected EBUSY, got %s",
			 strerror(errno));
		TEST_FAIL(msg);
		return;
	}
	TEST_PASS();
}

/*
 * Test 3: MR registration of page-backed buffer
 */
static __u32 test_mr_register(int fd)
{
	struct dmaplane_mr_params mrp;
	__u32 buf_id;
	int ret;

	TEST_START("MR registration (page-backed)");

	if (create_page_buffer(fd, 4096, &buf_id) < 0) {
		TEST_FAIL("buffer creation failed");
		return 0;
	}

	memset(&mrp, 0, sizeof(mrp));
	mrp.buf_id = buf_id;
	mrp.access_flags = IB_ACCESS_LOCAL_WRITE;

	ret = ioctl(fd, DMAPLANE_IOCTL_REGISTER_MR, &mrp);
	if (ret < 0) {
		char msg[64];
		snprintf(msg, sizeof(msg), "ioctl failed: %s", strerror(errno));
		TEST_FAIL(msg);
		return 0;
	}
	if (mrp.mr_id == 0 || mrp.lkey == 0) {
		TEST_FAIL("mr_id or lkey is 0");
		return 0;
	}
	printf("(mr_id=%u lkey=0x%x) ", mrp.mr_id, mrp.lkey);
	TEST_PASS();
	return mrp.mr_id;
}

/*
 * Test 4: MR registration of coherent buffer (must fail)
 */
static void test_mr_register_coherent(int fd)
{
	struct dmaplane_buf_params bp;
	struct dmaplane_mr_params mrp;
	int ret;

	TEST_START("MR registration of coherent (must fail)");

	memset(&bp, 0, sizeof(bp));
	bp.alloc_type = DMAPLANE_BUF_TYPE_COHERENT;
	bp.size = 4096;
	bp.numa_node = DMAPLANE_NUMA_ANY;

	ret = ioctl(fd, DMAPLANE_IOCTL_CREATE_BUFFER, &bp);
	if (ret < 0) {
		TEST_FAIL("coherent buffer creation failed");
		return;
	}

	memset(&mrp, 0, sizeof(mrp));
	mrp.buf_id = bp.buf_id;
	mrp.access_flags = IB_ACCESS_LOCAL_WRITE;

	ret = ioctl(fd, DMAPLANE_IOCTL_REGISTER_MR, &mrp);
	if (ret == 0) {
		TEST_FAIL("coherent MR registration should fail");
		/* Clean up the MR */
		ioctl(fd, DMAPLANE_IOCTL_DEREGISTER_MR, &mrp.mr_id);
		return;
	}
	if (errno != EINVAL) {
		char msg[64];
		snprintf(msg, sizeof(msg), "expected EINVAL, got %s",
			 strerror(errno));
		TEST_FAIL(msg);
		return;
	}

	/* Clean up the coherent buffer */
	ioctl(fd, DMAPLANE_IOCTL_DESTROY_BUFFER, &bp.buf_id);
	TEST_PASS();
}

/*
 * Test 5: Loopback test
 */
static void test_loopback(int fd, __u32 mr_id)
{
	struct dmaplane_loopback_params lp;
	int ret;

	TEST_START("Loopback test");

	memset(&lp, 0, sizeof(lp));
	lp.mr_id = mr_id;
	lp.size = 64;

	ret = ioctl(fd, DMAPLANE_IOCTL_LOOPBACK_TEST, &lp);
	if (ret < 0) {
		char msg[64];
		snprintf(msg, sizeof(msg), "ioctl failed: %s", strerror(errno));
		TEST_FAIL(msg);
		return;
	}
	if (lp.status != 0) {
		TEST_FAIL("loopback status != 0");
		return;
	}
	if (lp.latency_ns == 0) {
		TEST_FAIL("latency_ns is 0");
		return;
	}
	loopback_latency_ns = lp.latency_ns;
	printf("(latency=%llu ns) ", (unsigned long long)lp.latency_ns);
	TEST_PASS();
}

/*
 * Test 6: Ping-pong benchmark
 */
static void test_pingpong(int fd, __u32 mr_id)
{
	struct dmaplane_bench_params bp;
	int ret;

	TEST_START("Ping-pong benchmark (1000 iters, 4KB)");

	memset(&bp, 0, sizeof(bp));
	bp.mr_id = mr_id;
	bp.msg_size = 4096;
	bp.iterations = 1000;

	ret = ioctl(fd, DMAPLANE_IOCTL_PINGPONG_BENCH, &bp);
	if (ret < 0) {
		char msg[64];
		snprintf(msg, sizeof(msg), "ioctl failed: %s", strerror(errno));
		TEST_FAIL(msg);
		return;
	}
	if (bp.avg_latency_ns == 0) {
		TEST_FAIL("avg_latency_ns is 0");
		return;
	}
	pingpong_avg_ns = bp.avg_latency_ns;
	pingpong_p99_ns = bp.p99_latency_ns;
	printf("(avg=%llu ns, p99=%llu ns, %llu MB/s) ",
	       (unsigned long long)bp.avg_latency_ns,
	       (unsigned long long)bp.p99_latency_ns,
	       (unsigned long long)bp.throughput_mbps);
	TEST_PASS();
}

/*
 * Test 7: Streaming benchmark
 */
static void test_streaming(int fd, __u32 mr_id)
{
	struct dmaplane_bench_params bp;
	int ret;

	TEST_START("Streaming benchmark (1000 iters, 4KB, qdepth=8)");

	memset(&bp, 0, sizeof(bp));
	bp.mr_id = mr_id;
	bp.msg_size = 4096;
	bp.iterations = 1000;
	bp.queue_depth = 8;

	ret = ioctl(fd, DMAPLANE_IOCTL_STREAMING_BENCH, &bp);
	if (ret < 0) {
		char msg[64];
		snprintf(msg, sizeof(msg), "ioctl failed: %s", strerror(errno));
		TEST_FAIL(msg);
		return;
	}
	if (bp.throughput_mbps == 0) {
		TEST_FAIL("throughput_mbps is 0");
		return;
	}
	streaming_throughput_mbps = bp.throughput_mbps;
	printf("(%llu MB/s, avg=%llu ns, p99=%llu ns) ",
	       (unsigned long long)bp.throughput_mbps,
	       (unsigned long long)bp.avg_latency_ns,
	       (unsigned long long)bp.p99_latency_ns);
	TEST_PASS();
}

/*
 * Test 8: MR deregistration
 */
static void test_mr_deregister(int fd, __u32 mr_id)
{
	__u32 fake_id = 99999;
	int ret;

	TEST_START("MR deregistration + non-existent check");

	ret = ioctl(fd, DMAPLANE_IOCTL_DEREGISTER_MR, &mr_id);
	if (ret < 0) {
		char msg[64];
		snprintf(msg, sizeof(msg), "deregister failed: %s",
			 strerror(errno));
		TEST_FAIL(msg);
		return;
	}

	/* Deregister non-existent should return ENOENT */
	ret = ioctl(fd, DMAPLANE_IOCTL_DEREGISTER_MR, &fake_id);
	if (ret == 0) {
		TEST_FAIL("non-existent deregister succeeded");
		return;
	}
	if (errno != ENOENT) {
		char msg[64];
		snprintf(msg, sizeof(msg), "expected ENOENT, got %s",
			 strerror(errno));
		TEST_FAIL(msg);
		return;
	}
	TEST_PASS();
}

/*
 * Test 9: RDMA stats verification
 */
static void test_rdma_stats(int fd)
{
	struct dmaplane_rdma_stats rs;
	int ret;

	TEST_START("RDMA stats verification");

	ret = ioctl(fd, DMAPLANE_IOCTL_GET_RDMA_STATS, &rs);
	if (ret < 0) {
		char msg[64];
		snprintf(msg, sizeof(msg), "ioctl failed: %s", strerror(errno));
		TEST_FAIL(msg);
		return;
	}

	if (rs.sends_posted == 0) {
		TEST_FAIL("sends_posted is 0 after benchmarks");
		return;
	}
	if (rs.completions_polled == 0) {
		TEST_FAIL("completions_polled is 0 after benchmarks");
		return;
	}
	if (rs.completion_errors != 0) {
		char msg[64];
		snprintf(msg, sizeof(msg), "completion_errors=%llu (expected 0)",
			 (unsigned long long)rs.completion_errors);
		TEST_FAIL(msg);
		return;
	}

	printf("(sends=%llu, recvs=%llu, completions=%llu) ",
	       (unsigned long long)rs.sends_posted,
	       (unsigned long long)rs.recvs_posted,
	       (unsigned long long)rs.completions_polled);
	TEST_PASS();
}

/*
 * Test 10: RDMA teardown, loopback-after-teardown fails, re-setup succeeds
 */
static void test_teardown_and_reinit(int fd, const char *rxe_name)
{
	struct dmaplane_loopback_params lp;
	struct dmaplane_rdma_setup setup;
	int ret;

	TEST_START("Teardown + re-initialization");

	/* Teardown */
	ret = ioctl(fd, DMAPLANE_IOCTL_TEARDOWN_RDMA);
	if (ret < 0) {
		char msg[64];
		snprintf(msg, sizeof(msg), "teardown failed: %s",
			 strerror(errno));
		TEST_FAIL(msg);
		return;
	}

	/* Loopback after teardown should fail */
	memset(&lp, 0, sizeof(lp));
	lp.mr_id = 1;
	lp.size = 64;
	ret = ioctl(fd, DMAPLANE_IOCTL_LOOPBACK_TEST, &lp);
	if (ret == 0) {
		TEST_FAIL("loopback after teardown should fail");
		return;
	}

	/* Re-setup should succeed */
	memset(&setup, 0, sizeof(setup));
	snprintf(setup.ib_dev_name, sizeof(setup.ib_dev_name), "%s", rxe_name);
	setup.port = 1;
	ret = ioctl(fd, DMAPLANE_IOCTL_SETUP_RDMA, &setup);
	if (ret < 0) {
		char msg[64];
		snprintf(msg, sizeof(msg), "re-setup failed: %s",
			 strerror(errno));
		TEST_FAIL(msg);
		return;
	}
	if (setup.status != 0) {
		TEST_FAIL("re-setup status != 0");
		return;
	}

	/* Final teardown */
	ioctl(fd, DMAPLANE_IOCTL_TEARDOWN_RDMA);
	TEST_PASS();
}

int main(void)
{
	char rxe_name[32];
	__u32 mr_id;
	int fd;

	printf("=== Phase 4: RDMA Integration Tests ===\n\n");

	/* Check for rxe device */
	if (find_rxe_device(rxe_name, sizeof(rxe_name)) < 0) {
		fprintf(stderr,
			"ERROR: No Soft-RoCE (rxe) device found.\n"
			"Setup with:\n"
			"  sudo modprobe rdma_rxe\n"
			"  sudo rdma link add rxe_<iface> type rxe netdev <iface>\n"
			"  # e.g.: sudo rdma link add rxe_enp44s0 type rxe netdev enp44s0\n"
			"Or run: bash scripts/setup_rxe.sh\n");
		return 1;
	}
	printf("  Using rxe device: %s\n\n", rxe_name);

	fd = open(dev_path, O_RDWR);
	if (fd < 0) {
		fprintf(stderr, "ERROR: Cannot open %s: %s\n",
			dev_path, strerror(errno));
		return 1;
	}

	/* Test 1: RDMA setup */
	test_rdma_setup(fd, rxe_name);

	/* Test 2: Double setup (must fail EBUSY) */
	test_double_setup(fd, rxe_name);

	/* Test 3: MR registration */
	mr_id = test_mr_register(fd);

	/* Test 4: MR registration of coherent (must fail) */
	test_mr_register_coherent(fd);

	/* Tests 5-7 require a valid MR */
	if (mr_id > 0) {
		/* Test 5: Loopback */
		test_loopback(fd, mr_id);

		/* Test 6: Ping-pong */
		test_pingpong(fd, mr_id);

		/* Test 7: Streaming */
		test_streaming(fd, mr_id);

		/* Test 8: MR deregistration */
		test_mr_deregister(fd, mr_id);
	} else {
		printf("  SKIP: Tests 5-8 (no valid MR)\n");
		failed += 4;
	}

	/* Test 9: RDMA stats */
	test_rdma_stats(fd);

	/* Test 10: Teardown and re-init */
	test_teardown_and_reinit(fd, rxe_name);

	close(fd);

	/* Results table */
	printf("\n=== Phase 4 RDMA Results ===\n");
	printf("  Loopback latency:     %llu ns\n", (unsigned long long)loopback_latency_ns);
	printf("  Ping-pong avg:        %llu ns\n", (unsigned long long)pingpong_avg_ns);
	printf("  Ping-pong P99:        %llu ns\n", (unsigned long long)pingpong_p99_ns);
	printf("  Streaming throughput: %llu MB/s\n", (unsigned long long)streaming_throughput_mbps);

	printf("\n=== Summary: %d passed, %d failed (of %d) ===\n",
	       passed, failed, passed + failed);

	return failed ? 1 : 0;
}
