// SPDX-License-Identifier: GPL-2.0
/*
 * dmaplane Phase 3 test — dma-buf export, fd lifecycle, mmap via dma-buf
 * Copyright (c) 2026 Graziano Labs Corp.
 *
 * Userspace test suite that exercises the Phase 3 dma-buf export
 * interface: export page-backed buffers as dma-buf fds, reject coherent
 * and double exports, verify the destroy guard, mmap through the
 * dma-buf fd, and check lifetime statistics.
 *
 * Must be run as root after loading the dmaplane.ko module.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <stdint.h>

#include "dmaplane_uapi.h"

#define DEV_PATH	"/dev/dmaplane"

static int tests_passed;
static int tests_failed;

#define TEST_PASS(name) do { \
	printf("  PASS: %s\n", name); \
	tests_passed++; \
} while (0)

#define TEST_FAIL(name, ...) do { \
	printf("  FAIL: %s — ", name); \
	printf(__VA_ARGS__); \
	printf("\n"); \
	tests_failed++; \
} while (0)

/* ── Helpers ─────────────────────────────────────────────── */

static int dev_open(void)
{
	int fd = open(DEV_PATH, O_RDWR);

	if (fd < 0)
		perror("open " DEV_PATH);
	return fd;
}

static unsigned int create_buffer(int fd, unsigned int type, unsigned long size)
{
	struct dmaplane_buf_params p = { .numa_node = DMAPLANE_NUMA_ANY };

	p.alloc_type = type;
	p.size = size;
	if (ioctl(fd, DMAPLANE_IOCTL_CREATE_BUFFER, &p) < 0)
		return 0;
	return p.buf_id;
}

static int destroy_buffer(int fd, unsigned int buf_id)
{
	__u32 id = buf_id;

	if (ioctl(fd, DMAPLANE_IOCTL_DESTROY_BUFFER, &id) < 0)
		return -1;
	return 0;
}

static int export_dmabuf(int fd, unsigned int buf_id)
{
	struct dmaplane_export_dmabuf_arg ea = {0};

	ea.buf_id = buf_id;
	if (ioctl(fd, DMAPLANE_IOCTL_EXPORT_DMABUF, &ea) < 0)
		return -1;
	return ea.fd;
}

static int get_dmabuf_stats(int fd, struct dmaplane_dmabuf_stats *stats)
{
	if (ioctl(fd, DMAPLANE_IOCTL_GET_DMABUF_STATS, stats) < 0)
		return -1;
	return 0;
}

/* ── Test cases ──────────────────────────────────────────── */

/*
 * Test 1: Export page-backed buffer.
 * Create, export, verify fd >= 0, close dma-buf fd, destroy buffer.
 */
static void test_export_pages(int fd)
{
	const char *name = "export page-backed buffer";
	unsigned int buf_id;
	int dmabuf_fd;

	buf_id = create_buffer(fd, DMAPLANE_BUF_TYPE_PAGES, 4096);
	if (!buf_id) {
		TEST_FAIL(name, "create_buffer failed: %s", strerror(errno));
		return;
	}

	dmabuf_fd = export_dmabuf(fd, buf_id);
	if (dmabuf_fd < 0) {
		TEST_FAIL(name, "export failed: %s", strerror(errno));
		destroy_buffer(fd, buf_id);
		return;
	}

	if (dmabuf_fd < 3) {
		/* 0, 1, 2 are stdin/stdout/stderr — fd should be higher */
		TEST_FAIL(name, "suspicious fd %d", dmabuf_fd);
		close(dmabuf_fd);
		destroy_buffer(fd, buf_id);
		return;
	}

	close(dmabuf_fd);
	if (destroy_buffer(fd, buf_id) < 0) {
		TEST_FAIL(name, "destroy after close failed: %s", strerror(errno));
		return;
	}

	TEST_PASS(name);
}

/*
 * Test 2: Export coherent buffer (must fail).
 * Coherent buffers have no page array — cannot build SG tables.
 */
static void test_export_coherent_fails(int fd)
{
	const char *name = "export coherent (must fail EINVAL)";
	unsigned int buf_id;
	int dmabuf_fd;

	buf_id = create_buffer(fd, DMAPLANE_BUF_TYPE_COHERENT, 4096);
	if (!buf_id) {
		TEST_FAIL(name, "create_buffer failed: %s", strerror(errno));
		return;
	}

	dmabuf_fd = export_dmabuf(fd, buf_id);
	if (dmabuf_fd >= 0) {
		TEST_FAIL(name, "export succeeded (expected EINVAL)");
		close(dmabuf_fd);
		destroy_buffer(fd, buf_id);
		return;
	}
	if (errno != EINVAL) {
		TEST_FAIL(name, "expected EINVAL, got %s", strerror(errno));
		destroy_buffer(fd, buf_id);
		return;
	}

	destroy_buffer(fd, buf_id);
	TEST_PASS(name);
}

/*
 * Test 3: Double export (must fail).
 * One dma-buf per buffer — second export returns EBUSY.
 */
static void test_double_export(int fd)
{
	const char *name = "double export (must fail EBUSY)";
	unsigned int buf_id;
	int dmabuf_fd, dmabuf_fd2;

	buf_id = create_buffer(fd, DMAPLANE_BUF_TYPE_PAGES, 4096);
	if (!buf_id) {
		TEST_FAIL(name, "create_buffer failed: %s", strerror(errno));
		return;
	}

	dmabuf_fd = export_dmabuf(fd, buf_id);
	if (dmabuf_fd < 0) {
		TEST_FAIL(name, "first export failed: %s", strerror(errno));
		destroy_buffer(fd, buf_id);
		return;
	}

	dmabuf_fd2 = export_dmabuf(fd, buf_id);
	if (dmabuf_fd2 >= 0) {
		TEST_FAIL(name, "second export succeeded (expected EBUSY)");
		close(dmabuf_fd2);
		close(dmabuf_fd);
		destroy_buffer(fd, buf_id);
		return;
	}
	if (errno != EBUSY) {
		TEST_FAIL(name, "expected EBUSY, got %s", strerror(errno));
		close(dmabuf_fd);
		destroy_buffer(fd, buf_id);
		return;
	}

	close(dmabuf_fd);
	destroy_buffer(fd, buf_id);
	TEST_PASS(name);
}

/*
 * Test 4: Destroy while exported (must fail EBUSY).
 * Close the dma-buf fd, then destroy succeeds.
 */
static void test_destroy_while_exported(int fd)
{
	const char *name = "destroy while exported (must fail EBUSY)";
	unsigned int buf_id;
	int dmabuf_fd;

	buf_id = create_buffer(fd, DMAPLANE_BUF_TYPE_PAGES, 4096);
	if (!buf_id) {
		TEST_FAIL(name, "create_buffer failed: %s", strerror(errno));
		return;
	}

	dmabuf_fd = export_dmabuf(fd, buf_id);
	if (dmabuf_fd < 0) {
		TEST_FAIL(name, "export failed: %s", strerror(errno));
		destroy_buffer(fd, buf_id);
		return;
	}

	/* Destroy must fail while exported */
	if (destroy_buffer(fd, buf_id) == 0) {
		TEST_FAIL(name, "destroy succeeded while exported");
		close(dmabuf_fd);
		return;
	}
	if (errno != EBUSY) {
		TEST_FAIL(name, "expected EBUSY, got %s", strerror(errno));
		close(dmabuf_fd);
		return;
	}

	/* Close dma-buf fd → release callback clears exported flag */
	close(dmabuf_fd);

	/* Destroy should succeed now */
	if (destroy_buffer(fd, buf_id) < 0) {
		TEST_FAIL(name, "destroy after close failed: %s", strerror(errno));
		return;
	}

	TEST_PASS(name);
}

/*
 * Test 5: mmap via dma-buf fd.
 * Export, mmap the dma-buf fd (not /dev/dmaplane), write pattern, verify.
 */
static void test_mmap_via_dmabuf(int fd)
{
	const char *name = "mmap via dma-buf fd";
	unsigned int buf_id;
	int dmabuf_fd;
	void *ptr;
	uint32_t *data;
	unsigned int i;
	size_t size = 64 * 1024;	/* 64 KB */

	buf_id = create_buffer(fd, DMAPLANE_BUF_TYPE_PAGES, size);
	if (!buf_id) {
		TEST_FAIL(name, "create_buffer failed: %s", strerror(errno));
		return;
	}

	dmabuf_fd = export_dmabuf(fd, buf_id);
	if (dmabuf_fd < 0) {
		TEST_FAIL(name, "export failed: %s", strerror(errno));
		destroy_buffer(fd, buf_id);
		return;
	}

	/* mmap the dma-buf fd — offset 0, not the buffer ID encoding */
	ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED,
		   dmabuf_fd, 0);
	if (ptr == MAP_FAILED) {
		TEST_FAIL(name, "mmap dma-buf fd failed: %s", strerror(errno));
		close(dmabuf_fd);
		destroy_buffer(fd, buf_id);
		return;
	}

	/* Write pattern */
	data = (uint32_t *)ptr;
	for (i = 0; i < size / sizeof(uint32_t); i++)
		data[i] = 0xBEEFCAFE;

	/* Verify */
	for (i = 0; i < size / sizeof(uint32_t); i++) {
		if (data[i] != 0xBEEFCAFE) {
			TEST_FAIL(name, "data mismatch at index %u: got 0x%08X",
				  i, data[i]);
			munmap(ptr, size);
			close(dmabuf_fd);
			destroy_buffer(fd, buf_id);
			return;
		}
	}

	munmap(ptr, size);
	close(dmabuf_fd);
	destroy_buffer(fd, buf_id);
	TEST_PASS(name);
}

/*
 * Test 6: Export + close fd + destroy.
 * Verify that closing the dma-buf fd releases the export, allowing
 * buffer destruction.
 */
static void test_export_close_destroy(int fd)
{
	const char *name = "export + close fd + destroy";
	unsigned int buf_id;
	int dmabuf_fd;

	buf_id = create_buffer(fd, DMAPLANE_BUF_TYPE_PAGES, 8192);
	if (!buf_id) {
		TEST_FAIL(name, "create_buffer failed: %s", strerror(errno));
		return;
	}

	dmabuf_fd = export_dmabuf(fd, buf_id);
	if (dmabuf_fd < 0) {
		TEST_FAIL(name, "export failed: %s", strerror(errno));
		destroy_buffer(fd, buf_id);
		return;
	}

	/* Close dma-buf fd — triggers release callback */
	close(dmabuf_fd);

	/* Buffer should be destroyable now */
	if (destroy_buffer(fd, buf_id) < 0) {
		TEST_FAIL(name, "destroy failed: %s", strerror(errno));
		return;
	}

	TEST_PASS(name);
}

/*
 * Test 7: Stats verification.
 * Export/close cycle, verify counters incremented.
 */
static void test_stats(int fd)
{
	const char *name = "stats verification";
	struct dmaplane_dmabuf_stats before, after;
	unsigned int buf_id;
	int dmabuf_fd;

	get_dmabuf_stats(fd, &before);

	buf_id = create_buffer(fd, DMAPLANE_BUF_TYPE_PAGES, 4096);
	if (!buf_id) {
		TEST_FAIL(name, "create_buffer failed: %s", strerror(errno));
		return;
	}

	dmabuf_fd = export_dmabuf(fd, buf_id);
	if (dmabuf_fd < 0) {
		TEST_FAIL(name, "export failed: %s", strerror(errno));
		destroy_buffer(fd, buf_id);
		return;
	}

	close(dmabuf_fd);

	get_dmabuf_stats(fd, &after);

	if (after.dmabufs_exported - before.dmabufs_exported != 1) {
		TEST_FAIL(name, "expected 1 export, got %llu",
			  (unsigned long long)(after.dmabufs_exported - before.dmabufs_exported));
		destroy_buffer(fd, buf_id);
		return;
	}
	if (after.dmabufs_released - before.dmabufs_released != 1) {
		TEST_FAIL(name, "expected 1 release, got %llu",
			  (unsigned long long)(after.dmabufs_released - before.dmabufs_released));
		destroy_buffer(fd, buf_id);
		return;
	}

	destroy_buffer(fd, buf_id);
	TEST_PASS(name);
}

/*
 * Test 8: Multiple buffers exported simultaneously.
 * Export 4 buffers, close in reverse order, verify all clean up.
 */
static void test_multi_export(int fd)
{
	const char *name = "multiple buffers exported";
	unsigned int buf_ids[4];
	int dmabuf_fds[4];
	int i;

	for (i = 0; i < 4; i++) {
		buf_ids[i] = create_buffer(fd, DMAPLANE_BUF_TYPE_PAGES, 4096);
		if (!buf_ids[i]) {
			TEST_FAIL(name, "create_buffer %d failed: %s",
				  i, strerror(errno));
			while (i > 0) {
				i--;
				destroy_buffer(fd, buf_ids[i]);
			}
			return;
		}
	}

	for (i = 0; i < 4; i++) {
		dmabuf_fds[i] = export_dmabuf(fd, buf_ids[i]);
		if (dmabuf_fds[i] < 0) {
			TEST_FAIL(name, "export %d failed: %s",
				  i, strerror(errno));
			while (i > 0)
				close(dmabuf_fds[--i]);
			for (i = 0; i < 4; i++)
				destroy_buffer(fd, buf_ids[i]);
			return;
		}
	}

	/* Close in reverse order */
	for (i = 3; i >= 0; i--)
		close(dmabuf_fds[i]);

	/* All should be destroyable */
	for (i = 0; i < 4; i++) {
		if (destroy_buffer(fd, buf_ids[i]) < 0) {
			TEST_FAIL(name, "destroy buffer %d failed: %s",
				  i, strerror(errno));
			return;
		}
	}

	TEST_PASS(name);
}

/*
 * Test 9: Large buffer export.
 * 16 MB buffer, export, mmap via dma-buf fd, full write/read cycle.
 */
static void test_large_export(int fd)
{
	const char *name = "large buffer export (16 MB)";
	size_t size = 16 * 1024 * 1024;
	unsigned int buf_id;
	int dmabuf_fd;
	void *ptr;
	uint32_t *data;
	unsigned int i;

	buf_id = create_buffer(fd, DMAPLANE_BUF_TYPE_PAGES, size);
	if (!buf_id) {
		TEST_FAIL(name, "create_buffer failed: %s", strerror(errno));
		return;
	}

	dmabuf_fd = export_dmabuf(fd, buf_id);
	if (dmabuf_fd < 0) {
		TEST_FAIL(name, "export failed: %s", strerror(errno));
		destroy_buffer(fd, buf_id);
		return;
	}

	ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED,
		   dmabuf_fd, 0);
	if (ptr == MAP_FAILED) {
		TEST_FAIL(name, "mmap failed: %s", strerror(errno));
		close(dmabuf_fd);
		destroy_buffer(fd, buf_id);
		return;
	}

	/* Write incrementing pattern */
	data = (uint32_t *)ptr;
	for (i = 0; i < size / sizeof(uint32_t); i++)
		data[i] = i;

	/* Verify */
	for (i = 0; i < size / sizeof(uint32_t); i++) {
		if (data[i] != i) {
			TEST_FAIL(name, "mismatch at index %u: expected %u, got %u",
				  i, i, data[i]);
			munmap(ptr, size);
			close(dmabuf_fd);
			destroy_buffer(fd, buf_id);
			return;
		}
	}

	munmap(ptr, size);
	close(dmabuf_fd);
	destroy_buffer(fd, buf_id);
	TEST_PASS(name);
}

/* ── Main ────────────────────────────────────────────────── */

int main(void)
{
	int fd;

	printf("\n=== dmaplane Phase 3 Test Suite ===\n\n");

	fd = dev_open();
	if (fd < 0) {
		printf("Cannot open %s — is the module loaded?\n", DEV_PATH);
		return 1;
	}

	test_export_pages(fd);
	test_export_coherent_fails(fd);
	test_double_export(fd);
	test_destroy_while_exported(fd);
	test_mmap_via_dmabuf(fd);
	test_export_close_destroy(fd);
	test_stats(fd);
	test_multi_export(fd);
	test_large_export(fd);

	close(fd);

	printf("\n--- Results: %d passed, %d failed ---\n\n",
	       tests_passed, tests_failed);

	return tests_failed ? 1 : 0;
}
