// SPDX-License-Identifier: GPL-2.0
/*
 * dma_explorer.c — Interactive DMA buffer exploration demo
 * Copyright (c) 2026 Graziano Labs Corp.
 *
 * Creates one buffer of each type (coherent, page-backed), mmaps them,
 * writes a pattern, reads it back, prints buffer details, and cleans up.
 * Demonstrates the full Phase 2 buffer lifecycle.
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
#define BUF_SIZE	(64 * 1024)	/* 64 KB */

static void explore_buffer(int fd, const char *name, unsigned int type)
{
	struct dmaplane_buf_params p = { .numa_node = DMAPLANE_NUMA_ANY };
	struct dmaplane_mmap_info info = {0};
	void *ptr;
	uint32_t *data;
	unsigned int i;
	int ok = 1;

	printf("\n--- %s buffer (%s, %u bytes) ---\n",
	       name, type == DMAPLANE_BUF_TYPE_COHERENT ? "coherent" : "pages",
	       BUF_SIZE);

	p.alloc_type = type;
	p.size = BUF_SIZE;
	if (ioctl(fd, DMAPLANE_IOCTL_CREATE_BUFFER, &p) < 0) {
		printf("  create failed: %s\n", strerror(errno));
		return;
	}
	printf("  buf_id    = %u\n", p.buf_id);

	info.buf_id = p.buf_id;
	if (ioctl(fd, DMAPLANE_IOCTL_GET_MMAP_INFO, &info) < 0) {
		printf("  get_mmap_info failed: %s\n", strerror(errno));
		goto destroy;
	}
	printf("  mmap_off  = 0x%llx\n", (unsigned long long)info.mmap_offset);
	printf("  mmap_size = %llu\n", (unsigned long long)info.mmap_size);

	ptr = mmap(NULL, info.mmap_size, PROT_READ | PROT_WRITE,
		   MAP_SHARED, fd, info.mmap_offset);
	if (ptr == MAP_FAILED) {
		printf("  mmap failed: %s\n", strerror(errno));
		goto destroy;
	}

	/* Write pattern */
	data = (uint32_t *)ptr;
	for (i = 0; i < info.mmap_size / sizeof(uint32_t); i++)
		data[i] = 0xA5A5A5A5 + i;

	/* Verify */
	for (i = 0; i < info.mmap_size / sizeof(uint32_t); i++) {
		if (data[i] != 0xA5A5A5A5 + i) {
			printf("  MISMATCH at index %u: expected 0x%08X, got 0x%08X\n",
			       i, 0xA5A5A5A5 + i, data[i]);
			ok = 0;
			break;
		}
	}
	if (ok)
		printf("  write/read verified OK (%zu bytes)\n",
		       (size_t)info.mmap_size);

	munmap(ptr, info.mmap_size);

destroy:
	if (ioctl(fd, DMAPLANE_IOCTL_DESTROY_BUFFER, &p.buf_id) < 0)
		printf("  destroy failed: %s\n", strerror(errno));
	else
		printf("  destroyed\n");
}

int main(void)
{
	struct dmaplane_buf_stats stats;
	int fd;

	printf("dmaplane DMA Explorer\n");
	printf("=====================\n");

	fd = open(DEV_PATH, O_RDWR);
	if (fd < 0) {
		perror("open " DEV_PATH);
		return 1;
	}

	explore_buffer(fd, "Coherent", DMAPLANE_BUF_TYPE_COHERENT);
	explore_buffer(fd, "Page-backed", DMAPLANE_BUF_TYPE_PAGES);

	if (ioctl(fd, DMAPLANE_IOCTL_GET_BUF_STATS, &stats) == 0) {
		printf("\n--- Buffer Stats ---\n");
		printf("  created   = %llu\n", (unsigned long long)stats.buffers_created);
		printf("  destroyed = %llu\n", (unsigned long long)stats.buffers_destroyed);
	}

	close(fd);
	printf("\nDone.\n");
	return 0;
}
