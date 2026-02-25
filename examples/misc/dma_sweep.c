// SPDX-License-Identifier: GPL-2.0
/*
 * dma_sweep.c — DMA buffer size sweep benchmark
 * Copyright (c) 2026 Graziano Labs Corp.
 *
 * Allocates page-backed buffers from 4 KB to 64 MB (powers of 2),
 * times the full create/mmap/write/verify/munmap/destroy cycle at
 * each size.  Useful for seeing allocation scaling.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <stdint.h>

#include "dmaplane_uapi.h"

#define DEV_PATH "/dev/dmaplane"

static uint64_t now_ns(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static int sweep_one(int fd, size_t size, uint64_t *create_ns,
		     uint64_t *map_ns, uint64_t *write_ns, uint64_t *total_ns)
{
	struct dmaplane_buf_params p = { .numa_node = DMAPLANE_NUMA_ANY };
	struct dmaplane_mmap_info info = {0};
	void *ptr;
	uint32_t *data;
	unsigned int i;
	uint64_t t0, t1, t2, t3, t4;

	t0 = now_ns();

	p.alloc_type = DMAPLANE_BUF_TYPE_PAGES;
	p.size = size;
	if (ioctl(fd, DMAPLANE_IOCTL_CREATE_BUFFER, &p) < 0)
		return -1;

	t1 = now_ns();

	info.buf_id = p.buf_id;
	if (ioctl(fd, DMAPLANE_IOCTL_GET_MMAP_INFO, &info) < 0)
		goto fail;

	ptr = mmap(NULL, info.mmap_size, PROT_READ | PROT_WRITE,
		   MAP_SHARED, fd, info.mmap_offset);
	if (ptr == MAP_FAILED)
		goto fail;

	t2 = now_ns();

	/* Write */
	data = (uint32_t *)ptr;
	for (i = 0; i < info.mmap_size / sizeof(uint32_t); i++)
		data[i] = i;

	t3 = now_ns();

	munmap(ptr, info.mmap_size);
	ioctl(fd, DMAPLANE_IOCTL_DESTROY_BUFFER, &p.buf_id);

	t4 = now_ns();

	*create_ns = t1 - t0;
	*map_ns = t2 - t1;
	*write_ns = t3 - t2;
	*total_ns = t4 - t0;
	return 0;

fail:
	ioctl(fd, DMAPLANE_IOCTL_DESTROY_BUFFER, &p.buf_id);
	return -1;
}

int main(void)
{
	int fd;
	size_t size;
	uint64_t create_ns, map_ns, write_ns, total_ns;

	printf("dmaplane DMA Size Sweep\n");
	printf("=======================\n\n");

	fd = open(DEV_PATH, O_RDWR);
	if (fd < 0) {
		perror("open " DEV_PATH);
		return 1;
	}

	printf("%-12s  %10s  %10s  %10s  %10s  %10s\n",
	       "Size", "Create", "Map", "Write", "Total", "Pages");
	printf("%-12s  %10s  %10s  %10s  %10s  %10s\n",
	       "----", "------", "---", "-----", "-----", "-----");

	for (size = 4096; size <= 64UL * 1024 * 1024; size *= 2) {
		if (sweep_one(fd, size, &create_ns, &map_ns, &write_ns,
			      &total_ns) < 0) {
			printf("%-12zu  FAILED: %s\n", size, strerror(errno));
			continue;
		}

		char size_str[16];
		if (size >= 1024 * 1024)
			snprintf(size_str, sizeof(size_str), "%zu MB",
				 size / (1024 * 1024));
		else
			snprintf(size_str, sizeof(size_str), "%zu KB",
				 size / 1024);

		printf("%-12s  %8llu us  %8llu us  %8llu us  %8llu us  %10zu\n",
		       size_str,
		       (unsigned long long)(create_ns / 1000),
		       (unsigned long long)(map_ns / 1000),
		       (unsigned long long)(write_ns / 1000),
		       (unsigned long long)(total_ns / 1000),
		       size / 4096);
	}

	close(fd);
	printf("\nDone.\n");
	return 0;
}
