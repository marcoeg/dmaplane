// SPDX-License-Identifier: GPL-2.0
/*
 * dmaplane Phase 2 test — DMA buffer allocation, mmap, lifecycle
 * Copyright (c) 2026 Graziano Labs Corp.
 *
 * Userspace test suite that exercises the Phase 2 ioctl interface:
 * coherent and page-backed buffer allocation, mmap zero-copy access,
 * destroy-with-active-mmap guard, allocation stress, max-buffer limit,
 * handle uniqueness, large buffers, and invalid parameter rejection.
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

/*
 * create_buffer — Allocate a buffer via ioctl.
 * Returns buf_id on success, 0 on failure (0 is never a valid handle).
 */
static unsigned int create_buffer(int fd, unsigned int type, unsigned long size)
{
	struct dmaplane_buf_params p = { .numa_node = DMAPLANE_NUMA_ANY };

	p.alloc_type = type;
	p.size = size;
	if (ioctl(fd, DMAPLANE_IOCTL_CREATE_BUFFER, &p) < 0)
		return 0;
	return p.buf_id;
}

/*
 * destroy_buffer — Free a buffer via ioctl.
 * Returns 0 on success, -1 on failure (errno set).
 */
static int destroy_buffer(int fd, unsigned int buf_id)
{
	__u32 id = buf_id;

	if (ioctl(fd, DMAPLANE_IOCTL_DESTROY_BUFFER, &id) < 0)
		return -1;
	return 0;
}

/*
 * get_mmap_info — Get mmap offset/size for a buffer.
 * Returns 0 on success (fills info), -1 on failure.
 */
static int get_mmap_info(int fd, unsigned int buf_id,
			 struct dmaplane_mmap_info *info)
{
	memset(info, 0, sizeof(*info));
	info->buf_id = buf_id;
	if (ioctl(fd, DMAPLANE_IOCTL_GET_MMAP_INFO, info) < 0)
		return -1;
	return 0;
}

/*
 * map_buffer — mmap a buffer into userspace.
 * Returns mapped address or MAP_FAILED.
 */
static void *map_buffer(int fd, unsigned int buf_id, size_t *out_size)
{
	struct dmaplane_mmap_info info;
	void *ptr;

	if (get_mmap_info(fd, buf_id, &info) < 0)
		return MAP_FAILED;

	ptr = mmap(NULL, info.mmap_size, PROT_READ | PROT_WRITE,
		   MAP_SHARED, fd, info.mmap_offset);
	if (out_size)
		*out_size = info.mmap_size;
	return ptr;
}

/*
 * get_buf_stats — Read buffer statistics.
 */
static int get_buf_stats(int fd, struct dmaplane_buf_stats *stats)
{
	if (ioctl(fd, DMAPLANE_IOCTL_GET_BUF_STATS, stats) < 0)
		return -1;
	return 0;
}

/* ── Test cases ──────────────────────────────────────────── */

/*
 * Test 1: Coherent allocation and mmap.
 * Create a 4 KB coherent buffer, mmap it, write a pattern, read back,
 * verify, munmap, destroy.
 */
static void test_coherent_alloc_mmap(int fd)
{
	const char *name = "coherent alloc + mmap";
	unsigned int buf_id;
	void *ptr;
	size_t size;
	uint32_t *data;
	unsigned int i;

	buf_id = create_buffer(fd, DMAPLANE_BUF_TYPE_COHERENT, 4096);
	if (!buf_id) {
		TEST_FAIL(name, "create_buffer failed: %s", strerror(errno));
		return;
	}

	ptr = map_buffer(fd, buf_id, &size);
	if (ptr == MAP_FAILED) {
		TEST_FAIL(name, "mmap failed: %s", strerror(errno));
		destroy_buffer(fd, buf_id);
		return;
	}

	/* Write 0xDEADBEEF pattern */
	data = (uint32_t *)ptr;
	for (i = 0; i < size / sizeof(uint32_t); i++)
		data[i] = 0xDEADBEEF;

	/* Read back and verify */
	for (i = 0; i < size / sizeof(uint32_t); i++) {
		if (data[i] != 0xDEADBEEF) {
			TEST_FAIL(name, "data mismatch at offset %u: got 0x%08X",
				  i, data[i]);
			munmap(ptr, size);
			destroy_buffer(fd, buf_id);
			return;
		}
	}

	munmap(ptr, size);
	if (destroy_buffer(fd, buf_id) < 0) {
		TEST_FAIL(name, "destroy failed: %s", strerror(errno));
		return;
	}

	TEST_PASS(name);
}

/*
 * Test 2: Page-backed allocation and mmap.
 * Create a 1 MB page-backed buffer, same write/verify cycle.
 */
static void test_pages_alloc_mmap(int fd)
{
	const char *name = "page-backed alloc + mmap";
	unsigned int buf_id;
	void *ptr;
	size_t size;
	uint32_t *data;
	unsigned int i;

	buf_id = create_buffer(fd, DMAPLANE_BUF_TYPE_PAGES, 1024 * 1024);
	if (!buf_id) {
		TEST_FAIL(name, "create_buffer failed: %s", strerror(errno));
		return;
	}

	ptr = map_buffer(fd, buf_id, &size);
	if (ptr == MAP_FAILED) {
		TEST_FAIL(name, "mmap failed: %s", strerror(errno));
		destroy_buffer(fd, buf_id);
		return;
	}

	data = (uint32_t *)ptr;
	for (i = 0; i < size / sizeof(uint32_t); i++)
		data[i] = 0xCAFEBABE;

	for (i = 0; i < size / sizeof(uint32_t); i++) {
		if (data[i] != 0xCAFEBABE) {
			TEST_FAIL(name, "data mismatch at offset %u: got 0x%08X",
				  i, data[i]);
			munmap(ptr, size);
			destroy_buffer(fd, buf_id);
			return;
		}
	}

	munmap(ptr, size);
	if (destroy_buffer(fd, buf_id) < 0) {
		TEST_FAIL(name, "destroy failed: %s", strerror(errno));
		return;
	}

	TEST_PASS(name);
}

/*
 * Test 3: Destroy with active mmap.
 * Create a buffer, mmap it, try to destroy (must fail EBUSY),
 * munmap, then destroy (must succeed).
 */
static void test_destroy_with_mmap(int fd)
{
	const char *name = "destroy with active mmap";
	unsigned int buf_id;
	void *ptr;
	size_t size;

	buf_id = create_buffer(fd, DMAPLANE_BUF_TYPE_PAGES, 4096);
	if (!buf_id) {
		TEST_FAIL(name, "create_buffer failed: %s", strerror(errno));
		return;
	}

	ptr = map_buffer(fd, buf_id, &size);
	if (ptr == MAP_FAILED) {
		TEST_FAIL(name, "mmap failed: %s", strerror(errno));
		destroy_buffer(fd, buf_id);
		return;
	}

	/* Destroy must fail with EBUSY while mmap is active */
	if (destroy_buffer(fd, buf_id) == 0) {
		TEST_FAIL(name, "destroy succeeded while mmapped (expected EBUSY)");
		munmap(ptr, size);
		return;
	}
	if (errno != EBUSY) {
		TEST_FAIL(name, "expected EBUSY, got %s", strerror(errno));
		munmap(ptr, size);
		return;
	}

	/* munmap and retry — should succeed now */
	munmap(ptr, size);
	if (destroy_buffer(fd, buf_id) < 0) {
		TEST_FAIL(name, "destroy after munmap failed: %s", strerror(errno));
		return;
	}

	TEST_PASS(name);
}

/*
 * Test 4: Allocation stress.
 * 1000 create/destroy cycles with 4K page-backed buffers.
 * Verify stats match at the end.
 */
static void test_alloc_stress(int fd)
{
	const char *name = "alloc stress (1000 cycles)";
	struct dmaplane_buf_stats stats_before, stats_after;
	unsigned int buf_id;
	int i;

	get_buf_stats(fd, &stats_before);

	for (i = 0; i < 1000; i++) {
		buf_id = create_buffer(fd, DMAPLANE_BUF_TYPE_PAGES, 4096);
		if (!buf_id) {
			TEST_FAIL(name, "create failed at cycle %d: %s",
				  i, strerror(errno));
			return;
		}
		if (destroy_buffer(fd, buf_id) < 0) {
			TEST_FAIL(name, "destroy failed at cycle %d: %s",
				  i, strerror(errno));
			return;
		}
	}

	get_buf_stats(fd, &stats_after);

	if (stats_after.buffers_created - stats_before.buffers_created != 1000) {
		TEST_FAIL(name, "expected 1000 creates, got %llu",
			  (unsigned long long)(stats_after.buffers_created -
					       stats_before.buffers_created));
		return;
	}
	if (stats_after.buffers_destroyed - stats_before.buffers_destroyed != 1000) {
		TEST_FAIL(name, "expected 1000 destroys, got %llu",
			  (unsigned long long)(stats_after.buffers_destroyed -
					       stats_before.buffers_destroyed));
		return;
	}

	TEST_PASS(name);
}

/*
 * Test 5: Max buffers.
 * Allocate DMAPLANE_MAX_BUFFERS (64) buffers, verify the 65th fails,
 * then free all.
 */
static void test_max_buffers(int fd)
{
	const char *name = "max buffers (64)";
	unsigned int ids[64];
	unsigned int extra;
	int i;

	for (i = 0; i < 64; i++) {
		ids[i] = create_buffer(fd, DMAPLANE_BUF_TYPE_PAGES, 4096);
		if (!ids[i]) {
			TEST_FAIL(name, "create failed at slot %d: %s",
				  i, strerror(errno));
			/* Clean up what we allocated */
			while (i > 0)
				destroy_buffer(fd, ids[--i]);
			return;
		}
	}

	/* 65th must fail */
	extra = create_buffer(fd, DMAPLANE_BUF_TYPE_PAGES, 4096);
	if (extra != 0) {
		TEST_FAIL(name, "65th buffer succeeded (expected ENOMEM)");
		destroy_buffer(fd, extra);
		for (i = 0; i < 64; i++)
			destroy_buffer(fd, ids[i]);
		return;
	}
	if (errno != ENOMEM) {
		TEST_FAIL(name, "expected ENOMEM for 65th, got %s",
			  strerror(errno));
		for (i = 0; i < 64; i++)
			destroy_buffer(fd, ids[i]);
		return;
	}

	/* Free all */
	for (i = 0; i < 64; i++) {
		if (destroy_buffer(fd, ids[i]) < 0) {
			TEST_FAIL(name, "destroy failed for id %u: %s",
				  ids[i], strerror(errno));
			return;
		}
	}

	TEST_PASS(name);
}

/*
 * Test 6: Handle uniqueness.
 * Create and destroy 10 buffers, verify handles are monotonically
 * increasing and never 0.
 */
static void test_handle_uniqueness(int fd)
{
	const char *name = "handle uniqueness";
	unsigned int prev_id = 0;
	unsigned int buf_id;
	int i;

	for (i = 0; i < 10; i++) {
		buf_id = create_buffer(fd, DMAPLANE_BUF_TYPE_PAGES, 4096);
		if (!buf_id) {
			TEST_FAIL(name, "create failed at iteration %d: %s",
				  i, strerror(errno));
			return;
		}
		if (buf_id == 0) {
			TEST_FAIL(name, "got handle 0 (reserved)");
			return;
		}
		if (buf_id <= prev_id) {
			TEST_FAIL(name, "handle %u not > previous %u",
				  buf_id, prev_id);
			destroy_buffer(fd, buf_id);
			return;
		}
		prev_id = buf_id;
		destroy_buffer(fd, buf_id);
	}

	TEST_PASS(name);
}

/*
 * Test 7: Large buffer.
 * 16 MB page-backed buffer, mmap, write entire buffer, read back, verify.
 */
static void test_large_buffer(int fd)
{
	const char *name = "large buffer (16 MB)";
	const size_t buf_size = 16 * 1024 * 1024;
	unsigned int buf_id;
	void *ptr;
	size_t size;
	uint32_t *data;
	unsigned int i;

	buf_id = create_buffer(fd, DMAPLANE_BUF_TYPE_PAGES, buf_size);
	if (!buf_id) {
		TEST_FAIL(name, "create_buffer failed: %s", strerror(errno));
		return;
	}

	ptr = map_buffer(fd, buf_id, &size);
	if (ptr == MAP_FAILED) {
		TEST_FAIL(name, "mmap failed: %s", strerror(errno));
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
			TEST_FAIL(name, "data mismatch at index %u: expected %u, got %u",
				  i, i, data[i]);
			munmap(ptr, size);
			destroy_buffer(fd, buf_id);
			return;
		}
	}

	munmap(ptr, size);
	if (destroy_buffer(fd, buf_id) < 0) {
		TEST_FAIL(name, "destroy failed: %s", strerror(errno));
		return;
	}

	TEST_PASS(name);
}

/*
 * Test 8: Invalid parameters.
 * Verify proper error returns for bad inputs.
 */
static void test_invalid_params(int fd)
{
	const char *name = "invalid parameters";
	struct dmaplane_buf_params p = { .numa_node = DMAPLANE_NUMA_ANY };
	struct dmaplane_mmap_info info = {0};
	__u32 bad_id = 99999;

	/* Size 0 → EINVAL */
	p.alloc_type = DMAPLANE_BUF_TYPE_PAGES;
	p.size = 0;
	if (ioctl(fd, DMAPLANE_IOCTL_CREATE_BUFFER, &p) == 0) {
		TEST_FAIL(name, "size=0 succeeded (expected EINVAL)");
		return;
	}
	if (errno != EINVAL) {
		TEST_FAIL(name, "size=0: expected EINVAL, got %s", strerror(errno));
		return;
	}

	/* Bad alloc_type → EINVAL */
	p.alloc_type = 99;
	p.size = 4096;
	if (ioctl(fd, DMAPLANE_IOCTL_CREATE_BUFFER, &p) == 0) {
		TEST_FAIL(name, "bad alloc_type succeeded (expected EINVAL)");
		return;
	}
	if (errno != EINVAL) {
		TEST_FAIL(name, "bad alloc_type: expected EINVAL, got %s",
			  strerror(errno));
		return;
	}

	/* Destroy non-existent → ENOENT */
	if (ioctl(fd, DMAPLANE_IOCTL_DESTROY_BUFFER, &bad_id) == 0) {
		TEST_FAIL(name, "destroy non-existent succeeded (expected ENOENT)");
		return;
	}
	if (errno != ENOENT) {
		TEST_FAIL(name, "destroy non-existent: expected ENOENT, got %s",
			  strerror(errno));
		return;
	}

	/* mmap_info non-existent → ENOENT */
	info.buf_id = 99999;
	if (ioctl(fd, DMAPLANE_IOCTL_GET_MMAP_INFO, &info) == 0) {
		TEST_FAIL(name, "mmap_info non-existent succeeded (expected ENOENT)");
		return;
	}
	if (errno != ENOENT) {
		TEST_FAIL(name, "mmap_info non-existent: expected ENOENT, got %s",
			  strerror(errno));
		return;
	}

	TEST_PASS(name);
}

/* ── Main ────────────────────────────────────────────────── */

int main(void)
{
	int fd;

	printf("\n=== dmaplane Phase 2 Test Suite ===\n\n");

	fd = dev_open();
	if (fd < 0) {
		printf("Cannot open %s — is the module loaded?\n", DEV_PATH);
		return 1;
	}

	test_coherent_alloc_mmap(fd);
	test_pages_alloc_mmap(fd);
	test_destroy_with_mmap(fd);
	test_alloc_stress(fd);
	test_max_buffers(fd);
	test_handle_uniqueness(fd);
	test_large_buffer(fd);
	test_invalid_params(fd);

	close(fd);

	printf("\n--- Results: %d passed, %d failed ---\n\n",
	       tests_passed, tests_failed);

	return tests_failed ? 1 : 0;
}
