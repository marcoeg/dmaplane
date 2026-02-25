// SPDX-License-Identifier: GPL-2.0
/*
 * test_phase8_gpu.c — Phase 8 GPU memory integration tests
 * Copyright (c) 2026 Graziano Labs Corp.
 *
 * Exercises all 7 GPU ioctls (GPU_PIN, GPU_UNPIN, GPU_DMA_TO_HOST,
 * GPU_DMA_FROM_HOST, GPU_BENCH, GPU_REGISTER_MR, GPU_LOOPBACK)
 * with real CUDA allocations.
 *
 * Build:  make gpu_test   (from tests/)
 *   or:   gcc -O2 -Wall -I../../include -o test_phase8_gpu test_phase8_gpu.c -lcuda -lcudart
 * Run:    sudo ./test_phase8_gpu
 *
 * Requires: dmaplane.ko loaded, nvidia.ko loaded, working CUDA.
 *
 * BAR Throughput Reference (RTX 5000 Ada, PCIe Gen4 x16 laptop):
 *
 *   ioremap (uncacheable):
 *     H->G: ~44 MB/s   G->H: ~6 MB/s
 *
 *   ioremap_wc (write-combining):
 *     H->G: ~10 GB/s   G->H: ~107 MB/s
 *
 *   cudaMemcpy (GPU DMA engine):
 *     H->D: ~12 GB/s   D->H: ~13 GB/s
 *
 *   GPUDirect RDMA (ConnectX NIC DMA engine):
 *     Both: 12-24 GB/s
 *
 *   Why the difference:
 *     UC: each CPU store = one PCIe write TLP. No batching.
 *     WC: CPU stores coalesced into 64-byte cache-line-sized TLPs.
 *         Closes the write gap (10 vs 12 GB/s) but reads remain
 *         CPU-bound — PCIe non-posted reads stall on each load.
 *     cudaMemcpy: GPU DMA engine generates optimal PCIe TLP bursts.
 *     GPUDirect RDMA: NIC DMA engine reads BAR at full PCIe BW,
 *         bypassing the CPU entirely — symmetric bandwidth.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <dirent.h>
#include <cuda_runtime.h>

#include "dmaplane_uapi.h"

static int fd;
static int tests_passed, tests_failed;

#define GPU_PAGE_SIZE  (64 * 1024)   /* 64KB — NVIDIA P2P page granularity */
#define MB             (1024 * 1024)

/* IB access flags — mirror kernel values for userspace */
#define IB_ACCESS_LOCAL_WRITE  (1)

#define DEV_PATH "/dev/dmaplane"

#define CUDA_CHECK(call) do {                                        \
	cudaError_t err = (call);                                    \
	if (err != cudaSuccess) {                                    \
		fprintf(stderr, "  CUDA error at %s:%d: %s\n",      \
			__FILE__, __LINE__,                          \
			cudaGetErrorString(err));                     \
		return -1;                                           \
	}                                                            \
} while (0)

/* ── Helpers ──────────────────────────────────────────────────── */

/*
 * Allocate a host DMA buffer via dmaplane, then mmap it into this
 * process.  Returns the kernel buffer handle and a userspace pointer
 * to the same physical pages the kernel DMA engine will read/write.
 */
static int alloc_host_mapped(uint64_t size, uint32_t *handle, void **ptr)
{
	struct dmaplane_buf_params bp;
	struct dmaplane_mmap_info mi;

	memset(&bp, 0, sizeof(bp));
	bp.alloc_type = DMAPLANE_BUF_TYPE_PAGES;
	bp.size = size;
	bp.numa_node = DMAPLANE_NUMA_ANY;

	if (ioctl(fd, DMAPLANE_IOCTL_CREATE_BUFFER, &bp) < 0) {
		fprintf(stderr, "  CREATE_BUFFER failed: %s\n",
			strerror(errno));
		return -1;
	}
	*handle = bp.buf_id;

	memset(&mi, 0, sizeof(mi));
	mi.buf_id = bp.buf_id;
	if (ioctl(fd, DMAPLANE_IOCTL_GET_MMAP_INFO, &mi) < 0) {
		fprintf(stderr, "  GET_MMAP_INFO failed: %s\n",
			strerror(errno));
		ioctl(fd, DMAPLANE_IOCTL_DESTROY_BUFFER, handle);
		return -1;
	}

	*ptr = mmap(NULL, mi.mmap_size, PROT_READ | PROT_WRITE,
		    MAP_SHARED, fd, mi.mmap_offset);
	if (*ptr == MAP_FAILED) {
		fprintf(stderr, "  mmap failed: %s\n", strerror(errno));
		ioctl(fd, DMAPLANE_IOCTL_DESTROY_BUFFER, handle);
		return -1;
	}

	return 0;
}

static void free_host_mapped(uint32_t handle, void *ptr, uint64_t size)
{
	if (ptr && ptr != MAP_FAILED)
		munmap(ptr, size);
	ioctl(fd, DMAPLANE_IOCTL_DESTROY_BUFFER, &handle);
}

/*
 * find_rxe_device() — locate a soft-RoCE (rxe) InfiniBand device.
 *
 * Scans /sys/class/infiniband/ for any directory matching "rxe_*".
 * Each rxe device is named rxe_<netdev> (e.g. rxe_enp44s0), created
 * by: rdma link add rxe_<iface> type rxe netdev <iface>
 *
 * Returns the first rxe device found, which is sufficient for
 * loopback testing. For multi-NIC setups, a more specific selection
 * could be added.
 *
 * @name: Output buffer for the device name (e.g. "rxe_enp44s0").
 * @len:  Size of the output buffer.
 *
 * Return: 0 on success (name populated), -1 if no rxe device found.
 */
static int find_rxe_device(char *name, size_t len)
{
	DIR *dir = opendir("/sys/class/infiniband");
	if (!dir)
		return -1;

	struct dirent *ent;
	while ((ent = readdir(dir)) != NULL) {
		if (strncmp(ent->d_name, "rxe_", 4) == 0) {
			size_t slen = strlen(ent->d_name);
			if (slen >= len)
				slen = len - 1;
			memcpy(name, ent->d_name, slen);
			name[slen] = '\0';
			closedir(dir);
			return 0;
		}
	}
	closedir(dir);
	return -1;
}

/* ── Test 1: GPU Pin/Unpin Lifecycle ─────────────────────────── */

static void test1_pin_unpin(void)
{
	uint64_t sizes[] = { GPU_PAGE_SIZE, 1 * MB, 16 * MB };
	int nsizes = sizeof(sizes) / sizeof(sizes[0]);
	int pass = 1;

	printf("\nTest 1: GPU Pin/Unpin Lifecycle\n");

	for (int i = 0; i < nsizes; i++) {
		uint64_t sz = sizes[i];
		void *gpu_ptr = NULL;
		struct dmaplane_gpu_pin_params pp;
		struct dmaplane_gpu_unpin_params up;

		if (cudaMalloc(&gpu_ptr, sz) != cudaSuccess) {
			printf("  %4lluKB: cudaMalloc failed               "
			       "FAIL\n",
			       (unsigned long long)(sz / 1024));
			pass = 0;
			continue;
		}

		/* Check 64KB alignment */
		if ((uint64_t)gpu_ptr % GPU_PAGE_SIZE != 0) {
			printf("  %4lluKB: WARNING: cudaMalloc returned "
			       "non-64KB-aligned ptr %p — skipping\n",
			       (unsigned long long)(sz / 1024), gpu_ptr);
			cudaFree(gpu_ptr);
			continue;
		}

		memset(&pp, 0, sizeof(pp));
		pp.gpu_va = (uint64_t)gpu_ptr;
		pp.size = sz;

		if (ioctl(fd, DMAPLANE_IOCTL_GPU_PIN, &pp) < 0) {
			printf("  %4lluKB: GPU_PIN failed: %s            "
			       "FAIL\n",
			       (unsigned long long)(sz / 1024),
			       strerror(errno));
			cudaFree(gpu_ptr);
			pass = 0;
			continue;
		}

		int ok = 1;
		uint32_t expected_pages = sz / GPU_PAGE_SIZE;

		if (pp.num_pages != expected_pages) {
			printf("  %4lluKB: num_pages=%u expected=%u\n",
			       (unsigned long long)(sz / 1024),
			       pp.num_pages, expected_pages);
			ok = 0;
		}
		if (pp.bar1_consumed != sz) {
			printf("  %4lluKB: bar1=%llu expected=%llu\n",
			       (unsigned long long)(sz / 1024),
			       (unsigned long long)pp.bar1_consumed,
			       (unsigned long long)sz);
			ok = 0;
		}

		printf("  %4lluKB: handle=%-3u pages=%-5u numa=%-3d "
		       "bar1=%-12llu %s\n",
		       (unsigned long long)(sz / 1024),
		       pp.handle, pp.num_pages, pp.gpu_numa_node,
		       (unsigned long long)pp.bar1_consumed,
		       ok ? "PASS" : "FAIL");
		if (!ok)
			pass = 0;

		/* Unpin */
		memset(&up, 0, sizeof(up));
		up.handle = pp.handle;
		if (ioctl(fd, DMAPLANE_IOCTL_GPU_UNPIN, &up) < 0) {
			printf("  %4lluKB: GPU_UNPIN failed: %s\n",
			       (unsigned long long)(sz / 1024),
			       strerror(errno));
			pass = 0;
		}

		cudaFree(gpu_ptr);
	}

	if (pass) tests_passed++; else tests_failed++;
}

/* ── Test 2: Host->GPU Data Integrity ────────────────────────── */

static void test2_h2g_integrity(void)
{
	uint64_t sz = 1 * MB;
	void *gpu_ptr = NULL;
	void *host_ptr = NULL;
	uint32_t host_handle = 0;
	uint32_t *verify_buf = NULL;
	struct dmaplane_gpu_pin_params pp;
	struct dmaplane_gpu_unpin_params up;
	struct dmaplane_gpu_dma_params dp;
	int pass = 1;

	printf("\nTest 2: Host->GPU Data Integrity (1 MB)\n");

	/* Allocate host buffer + mmap */
	if (alloc_host_mapped(sz, &host_handle, &host_ptr) < 0) {
		printf("  Host buffer alloc failed                        "
		       "FAIL\n");
		tests_failed++;
		return;
	}

	/* Allocate GPU memory */
	if (cudaMalloc(&gpu_ptr, sz) != cudaSuccess) {
		printf("  cudaMalloc failed                               "
		       "FAIL\n");
		free_host_mapped(host_handle, host_ptr, sz);
		tests_failed++;
		return;
	}

	if ((uint64_t)gpu_ptr % GPU_PAGE_SIZE != 0) {
		printf("  WARNING: non-64KB-aligned GPU ptr — skipping    "
		       "SKIP\n");
		cudaFree(gpu_ptr);
		free_host_mapped(host_handle, host_ptr, sz);
		tests_passed++;
		return;
	}

	/* Pin GPU */
	memset(&pp, 0, sizeof(pp));
	pp.gpu_va = (uint64_t)gpu_ptr;
	pp.size = sz;
	if (ioctl(fd, DMAPLANE_IOCTL_GPU_PIN, &pp) < 0) {
		printf("  GPU_PIN failed: %s                              "
		       "FAIL\n", strerror(errno));
		cudaFree(gpu_ptr);
		free_host_mapped(host_handle, host_ptr, sz);
		tests_failed++;
		return;
	}

	/* Fill host buffer with pattern */
	uint32_t *hp = (uint32_t *)host_ptr;
	uint32_t nwords = sz / sizeof(uint32_t);
	for (uint32_t i = 0; i < nwords; i++)
		hp[i] = 0xDEADBEEF;

	/* DMA host -> GPU */
	memset(&dp, 0, sizeof(dp));
	dp.gpu_handle = pp.handle;
	dp.host_handle = host_handle;
	dp.offset = 0;
	dp.size = sz;
	if (ioctl(fd, DMAPLANE_IOCTL_GPU_DMA_FROM_HOST, &dp) < 0) {
		printf("  GPU_DMA_FROM_HOST failed: %s                    "
		       "FAIL\n", strerror(errno));
		pass = 0;
		goto cleanup2;
	}

	printf("  DMA from host to GPU: %.1f ms, %llu MB/s\n",
	       dp.elapsed_ns / 1e6,
	       (unsigned long long)dp.throughput_mbps);

	/* Read back via cudaMemcpy and compare */
	verify_buf = malloc(sz);
	if (!verify_buf) {
		printf("  malloc for verify failed                        "
		       "FAIL\n");
		pass = 0;
		goto cleanup2;
	}

	if (cudaMemcpy(verify_buf, gpu_ptr, sz, cudaMemcpyDeviceToHost)
	    != cudaSuccess) {
		printf("  cudaMemcpy D2H failed                           "
		       "FAIL\n");
		pass = 0;
		goto cleanup2;
	}
	cudaDeviceSynchronize();

	uint32_t mismatches = 0;
	for (uint32_t i = 0; i < nwords; i++) {
		if (verify_buf[i] != 0xDEADBEEF) {
			mismatches++;
			if (mismatches <= 3)
				printf("  mismatch at word %u: got 0x%08X\n",
				       i, verify_buf[i]);
		}
	}

	printf("  Verification: %u/%u uint32s match              %s\n",
	       nwords - mismatches, nwords,
	       mismatches == 0 ? "PASS" : "FAIL");
	if (mismatches)
		pass = 0;

cleanup2:
	free(verify_buf);
	memset(&up, 0, sizeof(up));
	up.handle = pp.handle;
	ioctl(fd, DMAPLANE_IOCTL_GPU_UNPIN, &up);
	cudaFree(gpu_ptr);
	free_host_mapped(host_handle, host_ptr, sz);

	if (pass) tests_passed++; else tests_failed++;
}

/* ── Test 3: GPU->Host Data Integrity ────────────────────────── */

static void test3_g2h_integrity(void)
{
	uint64_t sz = 1 * MB;
	void *gpu_ptr = NULL;
	void *host_ptr = NULL;
	uint32_t host_handle = 0;
	struct dmaplane_gpu_pin_params pp;
	struct dmaplane_gpu_unpin_params up;
	struct dmaplane_gpu_dma_params dp;
	int pass = 1;

	printf("\nTest 3: GPU->Host Data Integrity (1 MB)\n");

	if (alloc_host_mapped(sz, &host_handle, &host_ptr) < 0) {
		printf("  Host buffer alloc failed                        "
		       "FAIL\n");
		tests_failed++;
		return;
	}

	if (cudaMalloc(&gpu_ptr, sz) != cudaSuccess) {
		printf("  cudaMalloc failed                               "
		       "FAIL\n");
		free_host_mapped(host_handle, host_ptr, sz);
		tests_failed++;
		return;
	}

	if ((uint64_t)gpu_ptr % GPU_PAGE_SIZE != 0) {
		printf("  WARNING: non-64KB-aligned GPU ptr — skipping    "
		       "SKIP\n");
		cudaFree(gpu_ptr);
		free_host_mapped(host_handle, host_ptr, sz);
		tests_passed++;
		return;
	}

	/* Pin GPU */
	memset(&pp, 0, sizeof(pp));
	pp.gpu_va = (uint64_t)gpu_ptr;
	pp.size = sz;
	if (ioctl(fd, DMAPLANE_IOCTL_GPU_PIN, &pp) < 0) {
		printf("  GPU_PIN failed: %s                              "
		       "FAIL\n", strerror(errno));
		cudaFree(gpu_ptr);
		free_host_mapped(host_handle, host_ptr, sz);
		tests_failed++;
		return;
	}

	/* Fill GPU buffer via CUDA */
	if (cudaMemset(gpu_ptr, 0xAB, sz) != cudaSuccess) {
		printf("  cudaMemset failed                               "
		       "FAIL\n");
		pass = 0;
		goto cleanup3;
	}
	cudaDeviceSynchronize();

	/* Clear host buffer */
	memset(host_ptr, 0, sz);

	/* DMA GPU -> host */
	memset(&dp, 0, sizeof(dp));
	dp.gpu_handle = pp.handle;
	dp.host_handle = host_handle;
	dp.offset = 0;
	dp.size = sz;
	if (ioctl(fd, DMAPLANE_IOCTL_GPU_DMA_TO_HOST, &dp) < 0) {
		printf("  GPU_DMA_TO_HOST failed: %s                      "
		       "FAIL\n", strerror(errno));
		pass = 0;
		goto cleanup3;
	}

	printf("  DMA from GPU to host: %.1f ms, %llu MB/s\n",
	       dp.elapsed_ns / 1e6,
	       (unsigned long long)dp.throughput_mbps);

	/* Verify every byte is 0xAB */
	uint8_t *bp = (uint8_t *)host_ptr;
	uint64_t mismatches = 0;
	for (uint64_t i = 0; i < sz; i++) {
		if (bp[i] != 0xAB) {
			mismatches++;
			if (mismatches <= 3)
				printf("  mismatch at byte %llu: got 0x%02X\n",
				       (unsigned long long)i, bp[i]);
		}
	}

	printf("  Verification: %llu/%llu bytes match                "
	       "%s\n",
	       (unsigned long long)(sz - mismatches),
	       (unsigned long long)sz,
	       mismatches == 0 ? "PASS" : "FAIL");
	if (mismatches)
		pass = 0;

cleanup3:
	memset(&up, 0, sizeof(up));
	up.handle = pp.handle;
	ioctl(fd, DMAPLANE_IOCTL_GPU_UNPIN, &up);
	cudaFree(gpu_ptr);
	free_host_mapped(host_handle, host_ptr, sz);

	if (pass) tests_passed++; else tests_failed++;
}

/* ── Test 4: BAR Throughput Benchmark ────────────────────────── */

static double time_ms(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static void test4_benchmark(void)
{
	uint64_t sizes[] = { GPU_PAGE_SIZE, 256 * 1024, 1 * MB,
			     4 * MB, 16 * MB };
	int nsizes = sizeof(sizes) / sizeof(sizes[0]);
	int pass = 1;

	printf("\nTest 4: BAR Throughput Benchmark (write-combining)\n");
	printf("  %-10s %-12s %-12s %-14s %-14s\n",
	       "Size", "H->G MB/s", "G->H MB/s",
	       "cudaH2D MB/s", "cudaD2H MB/s");

	for (int i = 0; i < nsizes; i++) {
		uint64_t sz = sizes[i];
		int iters = (sz >= 16 * MB) ? 10 : 100;
		void *gpu_ptr = NULL;
		void *host_ptr = NULL;
		uint32_t host_handle = 0;
		struct dmaplane_gpu_pin_params pp;
		struct dmaplane_gpu_unpin_params up;
		struct dmaplane_gpu_bench_params bp;
		void *cuda_host = NULL;

		if (cudaMalloc(&gpu_ptr, sz) != cudaSuccess) {
			char lbl[32];
			if (sz >= MB)
				snprintf(lbl, sizeof(lbl), "%lluMB",
					 (unsigned long long)(sz / MB));
			else
				snprintf(lbl, sizeof(lbl), "%lluKB",
					 (unsigned long long)(sz / 1024));
			printf("  %-10s cudaMalloc failed\n", lbl);
			pass = 0;
			continue;
		}

		if ((uint64_t)gpu_ptr % GPU_PAGE_SIZE != 0) {
			printf("  %-10llu non-aligned — skipping\n",
			       (unsigned long long)sz);
			cudaFree(gpu_ptr);
			continue;
		}

		memset(&pp, 0, sizeof(pp));
		pp.gpu_va = (uint64_t)gpu_ptr;
		pp.size = sz;
		if (ioctl(fd, DMAPLANE_IOCTL_GPU_PIN, &pp) < 0) {
			printf("  %-10llu GPU_PIN failed: %s\n",
			       (unsigned long long)sz, strerror(errno));
			cudaFree(gpu_ptr);
			pass = 0;
			continue;
		}

		if (alloc_host_mapped(sz, &host_handle, &host_ptr) < 0) {
			printf("  %-10llu host alloc failed\n",
			       (unsigned long long)sz);
			memset(&up, 0, sizeof(up));
			up.handle = pp.handle;
			ioctl(fd, DMAPLANE_IOCTL_GPU_UNPIN, &up);
			cudaFree(gpu_ptr);
			pass = 0;
			continue;
		}

		/* dmaplane benchmark */
		memset(&bp, 0, sizeof(bp));
		bp.gpu_handle = pp.handle;
		bp.host_handle = host_handle;
		bp.size = sz;     /* reserved/unused by kernel */
		bp.iterations = iters;

		uint64_t h2g = 0, g2h = 0;
		if (ioctl(fd, DMAPLANE_IOCTL_GPU_BENCH, &bp) < 0) {
			printf("  %-10llu GPU_BENCH failed: %s\n",
			       (unsigned long long)sz, strerror(errno));
			pass = 0;
		} else {
			h2g = bp.h2g_bandwidth_mbps;
			g2h = bp.g2h_bandwidth_mbps;
		}

		/* cudaMemcpy comparison */
		double cuda_h2d = 0, cuda_d2h = 0;
		if (cudaMallocHost(&cuda_host, sz) == cudaSuccess) {
			double t0, t1;

			/* H2D */
			cudaDeviceSynchronize();
			t0 = time_ms();
			for (int j = 0; j < iters; j++)
				cudaMemcpy(gpu_ptr, cuda_host, sz,
					   cudaMemcpyHostToDevice);
			cudaDeviceSynchronize();
			t1 = time_ms();
			if (t1 > t0)
				cuda_h2d = (double)sz * iters /
					   ((t1 - t0) / 1000.0) / 1e6;

			/* D2H */
			cudaDeviceSynchronize();
			t0 = time_ms();
			for (int j = 0; j < iters; j++)
				cudaMemcpy(cuda_host, gpu_ptr, sz,
					   cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
			t1 = time_ms();
			if (t1 > t0)
				cuda_d2h = (double)sz * iters /
					   ((t1 - t0) / 1000.0) / 1e6;

			cudaFreeHost(cuda_host);
		}

		char sz_label[32];
		if (sz >= MB)
			snprintf(sz_label, sizeof(sz_label), "%lluMB",
				 (unsigned long long)(sz / MB));
		else
			snprintf(sz_label, sizeof(sz_label), "%lluKB",
				 (unsigned long long)(sz / 1024));

		printf("  %-10s %-12llu %-12llu %-14.0f %-14.0f\n",
		       sz_label, (unsigned long long)h2g,
		       (unsigned long long)g2h, cuda_h2d, cuda_d2h);

		/* Cleanup */
		memset(&up, 0, sizeof(up));
		up.handle = pp.handle;
		ioctl(fd, DMAPLANE_IOCTL_GPU_UNPIN, &up);
		cudaFree(gpu_ptr);
		free_host_mapped(host_handle, host_ptr, sz);
	}

	if (pass) tests_passed++; else tests_failed++;
}

/* ── Test 5: Unpin Callback Safety ───────────────────────────── */

static void test5_callback_safety(void)
{
	uint64_t sz = 1 * MB;
	void *gpu_ptr = NULL;
	void *host_ptr = NULL;
	uint32_t host_handle = 0;
	struct dmaplane_gpu_pin_params pp;
	struct dmaplane_gpu_unpin_params up;
	struct dmaplane_gpu_dma_params dp;
	int pass = 1;

	printf("\nTest 5: Unpin Callback Safety\n");

	if (cudaMalloc(&gpu_ptr, sz) != cudaSuccess) {
		printf("  cudaMalloc failed                               "
		       "FAIL\n");
		tests_failed++;
		return;
	}

	if ((uint64_t)gpu_ptr % GPU_PAGE_SIZE != 0) {
		printf("  WARNING: non-64KB-aligned GPU ptr — skipping    "
		       "SKIP\n");
		cudaFree(gpu_ptr);
		tests_passed++;
		return;
	}

	/* Pin GPU */
	memset(&pp, 0, sizeof(pp));
	pp.gpu_va = (uint64_t)gpu_ptr;
	pp.size = sz;
	if (ioctl(fd, DMAPLANE_IOCTL_GPU_PIN, &pp) < 0) {
		printf("  GPU_PIN failed: %s                              "
		       "FAIL\n", strerror(errno));
		cudaFree(gpu_ptr);
		tests_failed++;
		return;
	}

	/* Free GPU memory while still pinned — triggers revocation callback */
	cudaFree(gpu_ptr);
	gpu_ptr = NULL;
	usleep(10000);  /* 10ms — let callback fire */
	printf("  cudaFree while pinned: callback triggered\n");

	/* Allocate host buffer for the DMA attempt */
	if (alloc_host_mapped(sz, &host_handle, &host_ptr) < 0) {
		printf("  Host buffer alloc failed                        "
		       "FAIL\n");
		/* Still try to unpin */
		memset(&up, 0, sizeof(up));
		up.handle = pp.handle;
		ioctl(fd, DMAPLANE_IOCTL_GPU_UNPIN, &up);
		tests_failed++;
		return;
	}

	/* Try DMA after revocation — should fail */
	memset(&dp, 0, sizeof(dp));
	dp.gpu_handle = pp.handle;
	dp.host_handle = host_handle;
	dp.offset = 0;
	dp.size = sz;
	if (ioctl(fd, DMAPLANE_IOCTL_GPU_DMA_TO_HOST, &dp) < 0) {
		printf("  DMA after revocation: rejected (errno=%d)       "
		       "OK\n", errno);
	} else {
		printf("  DMA after revocation: UNEXPECTED SUCCESS        "
		       "FAIL\n");
		pass = 0;
	}

	/* Unpin the revoked buffer — should succeed cleanly */
	memset(&up, 0, sizeof(up));
	up.handle = pp.handle;
	if (ioctl(fd, DMAPLANE_IOCTL_GPU_UNPIN, &up) < 0) {
		printf("  Unpin after revocation: failed (errno=%d)       "
		       "FAIL\n", errno);
		pass = 0;
	} else {
		printf("  Unpin after revocation: clean                   "
		       "PASS\n");
	}

	free_host_mapped(host_handle, host_ptr, sz);

	if (pass) tests_passed++; else tests_failed++;
}

/* ── Test 6: Alignment Validation ────────────────────────────── */

static void test6_alignment(void)
{
	uint64_t sz = 1 * MB;
	void *gpu_ptr = NULL;
	struct dmaplane_gpu_pin_params pp;
	int pass = 1;

	printf("\nTest 6: Alignment Validation\n");

	if (cudaMalloc(&gpu_ptr, sz) != cudaSuccess) {
		printf("  cudaMalloc failed                               "
		       "FAIL\n");
		tests_failed++;
		return;
	}

	/* Sub-test a: misaligned VA (+1 byte) */
	memset(&pp, 0, sizeof(pp));
	pp.gpu_va = (uint64_t)gpu_ptr + 1;
	pp.size = sz;
	if (ioctl(fd, DMAPLANE_IOCTL_GPU_PIN, &pp) < 0) {
		printf("  Misaligned VA (+1): rejected (errno=%d)         "
		       "PASS\n", errno);
	} else {
		printf("  Misaligned VA (+1): UNEXPECTED SUCCESS          "
		       "FAIL\n");
		struct dmaplane_gpu_unpin_params up = { .handle = pp.handle };
		ioctl(fd, DMAPLANE_IOCTL_GPU_UNPIN, &up);
		pass = 0;
	}

	/* Sub-test b: bad size (not 64KB multiple) */
	memset(&pp, 0, sizeof(pp));
	pp.gpu_va = (uint64_t)gpu_ptr;
	pp.size = 1000;
	if (ioctl(fd, DMAPLANE_IOCTL_GPU_PIN, &pp) < 0) {
		printf("  Bad size (1000): rejected (errno=%d)            "
		       "PASS\n", errno);
	} else {
		printf("  Bad size (1000): UNEXPECTED SUCCESS             "
		       "FAIL\n");
		struct dmaplane_gpu_unpin_params up = { .handle = pp.handle };
		ioctl(fd, DMAPLANE_IOCTL_GPU_UNPIN, &up);
		pass = 0;
	}

	/* Sub-test c: zero size */
	memset(&pp, 0, sizeof(pp));
	pp.gpu_va = (uint64_t)gpu_ptr;
	pp.size = 0;
	if (ioctl(fd, DMAPLANE_IOCTL_GPU_PIN, &pp) < 0) {
		printf("  Zero size: rejected (errno=%d)                  "
		       "PASS\n", errno);
	} else {
		printf("  Zero size: UNEXPECTED SUCCESS                   "
		       "FAIL\n");
		struct dmaplane_gpu_unpin_params up = { .handle = pp.handle };
		ioctl(fd, DMAPLANE_IOCTL_GPU_UNPIN, &up);
		pass = 0;
	}

	cudaFree(gpu_ptr);

	if (pass) tests_passed++; else tests_failed++;
}

/* ── Test 7: GPU RDMA Loopback ───────────────────────────────── */
/*
 * End-to-end test proving GPU VRAM can traverse the RDMA data path.
 *
 * This is the critical RDMA bridge validation: data originates in GPU
 * memory (filled with 0xCD via cudaMemset), travels through the
 * dmaplane RDMA subsystem (rxe soft-RoCE loopback), and arrives in
 * a host DMA buffer — where we verify every byte matches.
 *
 * Data flow:
 *   cudaMemset(gpu_ptr, 0xCD)       <- CUDA fills GPU VRAM
 *   GPU_PIN                         <- pin VRAM, get BAR1 mappings
 *   GPU_REGISTER_MR                 <- register BAR pages as RDMA MR
 *                                      (sge.addr = rdma_vaddr, the
 *                                       contiguous WC mapping)
 *   SETUP_RDMA                      <- create PD/CQ/QP pair on rxe
 *   GPU_LOOPBACK                    <- post send (GPU MR on QP-A)
 *                                      post recv (host MR on QP-B)
 *                                      rxe does memcpy from rdma_vaddr
 *                                      into host buffer
 *   verify host_ptr[0..sz] == 0xCD  <- confirm data integrity
 *
 * The test is skipped (counts as PASS) if no rxe device exists.
 *
 * Cleanup order matters:
 *   1. Deregister MRs (need RDMA context alive)
 *   2. Teardown RDMA (destroys PD/CQ/QP)
 *   3. Unpin GPU buffer (unmaps BAR pages)
 *   4. cudaFree (release GPU VRAM)
 *   5. Free host buffer
 */

static void test7_gpu_rdma_loopback(void)
{
	uint64_t sz = GPU_PAGE_SIZE;   /* 64KB — single GPU page */
	void *gpu_ptr = NULL;
	void *host_ptr = NULL;
	uint32_t host_handle = 0;
	struct dmaplane_gpu_pin_params pp;
	struct dmaplane_gpu_unpin_params up;
	struct dmaplane_gpu_mr_params gmr;
	struct dmaplane_mr_params hmr;
	struct dmaplane_gpu_loopback_params lp;
	struct dmaplane_rdma_setup rs;
	char rxe_name[32];
	int pass = 1;
	int rdma_up = 0, gpu_pinned = 0, gpu_mr_ok = 0, host_mr_ok = 0;

	printf("\nTest 7: GPU RDMA Loopback (rxe, %llu KB)\n",
	       (unsigned long long)(sz / 1024));

	/* Find rxe device */
	if (find_rxe_device(rxe_name, sizeof(rxe_name)) < 0) {
		printf("  No rxe_* device found — skipping                "
		       "SKIP\n");
		tests_passed++;
		return;
	}
	printf("  Using IB device: %s\n", rxe_name);

	/* Defensive: teardown any stale RDMA state (ignore error) */
	ioctl(fd, DMAPLANE_IOCTL_TEARDOWN_RDMA);

	/* Setup RDMA */
	memset(&rs, 0, sizeof(rs));
	memset(rs.ib_dev_name, 0, sizeof(rs.ib_dev_name));
	snprintf(rs.ib_dev_name, sizeof(rs.ib_dev_name), "%s", rxe_name);
	rs.port = 1;
	rs.cq_depth = 16;
	rs.max_send_wr = 16;
	rs.max_recv_wr = 16;
	if (ioctl(fd, DMAPLANE_IOCTL_SETUP_RDMA, &rs) < 0) {
		printf("  SETUP_RDMA failed: %s                           "
		       "FAIL\n", strerror(errno));
		tests_failed++;
		return;
	}
	rdma_up = 1;

	/* Allocate GPU memory + fill with 0xCD */
	if (cudaMalloc(&gpu_ptr, sz) != cudaSuccess) {
		printf("  cudaMalloc failed                               "
		       "FAIL\n");
		pass = 0;
		goto cleanup7;
	}
	if (cudaMemset(gpu_ptr, 0xCD, sz) != cudaSuccess) {
		printf("  cudaMemset failed                               "
		       "FAIL\n");
		pass = 0;
		goto cleanup7;
	}
	cudaDeviceSynchronize();

	if ((uint64_t)gpu_ptr % GPU_PAGE_SIZE != 0) {
		printf("  WARNING: non-64KB-aligned GPU ptr — skipping    "
		       "SKIP\n");
		cudaFree(gpu_ptr);
		gpu_ptr = NULL;
		goto cleanup7;
	}

	/* Pin GPU memory */
	memset(&pp, 0, sizeof(pp));
	pp.gpu_va = (uint64_t)gpu_ptr;
	pp.size = sz;
	if (ioctl(fd, DMAPLANE_IOCTL_GPU_PIN, &pp) < 0) {
		printf("  GPU_PIN failed: %s                              "
		       "FAIL\n", strerror(errno));
		pass = 0;
		goto cleanup7;
	}
	gpu_pinned = 1;

	/* Register GPU MR */
	memset(&gmr, 0, sizeof(gmr));
	gmr.gpu_handle = pp.handle;
	if (ioctl(fd, DMAPLANE_IOCTL_GPU_REGISTER_MR, &gmr) < 0) {
		printf("  GPU_REGISTER_MR failed: %s                      "
		       "FAIL\n", strerror(errno));
		pass = 0;
		goto cleanup7;
	}
	gpu_mr_ok = 1;
	printf("  GPU MR: id=%u lkey=0x%x rkey=0x%x\n",
	       gmr.mr_id, gmr.lkey, gmr.rkey);

	/* Allocate host receive buffer + mmap */
	if (alloc_host_mapped(sz, &host_handle, &host_ptr) < 0) {
		printf("  Host buffer alloc failed                        "
		       "FAIL\n");
		pass = 0;
		goto cleanup7;
	}
	memset(host_ptr, 0, sz);   /* clear receive buffer */

	/* Register host MR (needs LOCAL_WRITE for recv) */
	memset(&hmr, 0, sizeof(hmr));
	hmr.buf_id = host_handle;
	hmr.access_flags = IB_ACCESS_LOCAL_WRITE;
	if (ioctl(fd, DMAPLANE_IOCTL_REGISTER_MR, &hmr) < 0) {
		printf("  REGISTER_MR (host) failed: %s                   "
		       "FAIL\n", strerror(errno));
		pass = 0;
		goto cleanup7;
	}
	host_mr_ok = 1;
	printf("  Host MR: id=%u lkey=0x%x rkey=0x%x\n",
	       hmr.mr_id, hmr.lkey, hmr.rkey);

	/* GPU RDMA loopback: send from GPU MR, recv into host MR */
	memset(&lp, 0, sizeof(lp));
	lp.gpu_mr_id = gmr.mr_id;
	lp.host_mr_id = hmr.mr_id;
	lp.size = sz;
	if (ioctl(fd, DMAPLANE_IOCTL_GPU_LOOPBACK, &lp) < 0) {
		printf("  GPU_LOOPBACK failed: %s                         "
		       "FAIL\n", strerror(errno));
		pass = 0;
		goto cleanup7;
	}

	printf("  Loopback: latency=%llu ns, recv=%u bytes, status=%u\n",
	       (unsigned long long)lp.latency_ns, lp.recv_bytes, lp.status);

	if (lp.status != 0) {
		printf("  Loopback status != 0                            "
		       "FAIL\n");
		pass = 0;
		goto cleanup7;
	}

	if (lp.recv_bytes != sz) {
		printf("  recv_bytes=%u expected=%llu                     "
		       "FAIL\n", lp.recv_bytes, (unsigned long long)sz);
		pass = 0;
		goto cleanup7;
	}

	/* Verify every byte of host buffer == 0xCD */
	uint8_t *bp = (uint8_t *)host_ptr;
	uint64_t mismatches = 0;
	for (uint64_t i = 0; i < sz; i++) {
		if (bp[i] != 0xCD) {
			mismatches++;
			if (mismatches <= 3)
				printf("  mismatch at byte %llu: "
				       "got 0x%02X expected 0xCD\n",
				       (unsigned long long)i, bp[i]);
		}
	}

	printf("  Verification: %llu/%llu bytes match                "
	       "%s\n",
	       (unsigned long long)(sz - mismatches),
	       (unsigned long long)sz,
	       mismatches == 0 ? "PASS" : "FAIL");
	if (mismatches)
		pass = 0;

cleanup7:
	/* Deregister MRs */
	if (host_mr_ok) {
		uint32_t id = hmr.mr_id;
		ioctl(fd, DMAPLANE_IOCTL_DEREGISTER_MR, &id);
	}
	if (gpu_mr_ok) {
		uint32_t id = gmr.mr_id;
		ioctl(fd, DMAPLANE_IOCTL_DEREGISTER_MR, &id);
	}

	/* Teardown RDMA before unpin */
	if (rdma_up)
		ioctl(fd, DMAPLANE_IOCTL_TEARDOWN_RDMA);

	/* Unpin + free GPU */
	if (gpu_pinned) {
		memset(&up, 0, sizeof(up));
		up.handle = pp.handle;
		ioctl(fd, DMAPLANE_IOCTL_GPU_UNPIN, &up);
	}
	if (gpu_ptr)
		cudaFree(gpu_ptr);

	/* Free host buffer */
	if (host_ptr && host_ptr != MAP_FAILED)
		free_host_mapped(host_handle, host_ptr, sz);

	if (pass) tests_passed++; else tests_failed++;
}

/* ── Main ─────────────────────────────────────────────────────── */

int main(void)
{
	printf("========================================\n");
	printf(" dmaplane Phase 8 — GPU P2P Tests\n");
	printf("========================================\n");

	/* Open device */
	fd = open(DEV_PATH, O_RDWR);
	if (fd < 0) {
		fprintf(stderr, "Cannot open %s: %s\n",
			DEV_PATH, strerror(errno));
		return 1;
	}

	/* Verify CUDA is functional */
	int dev_count = 0;
	cudaError_t cerr = cudaGetDeviceCount(&dev_count);
	if (cerr != cudaSuccess || dev_count == 0) {
		fprintf(stderr, "No CUDA devices found: %s\n",
			cudaGetErrorString(cerr));
		close(fd);
		return 1;
	}

	struct cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	printf("\nGPU: %s (CUDA %d.%d)\n", prop.name,
	       prop.major, prop.minor);

	/* Run tests */
	test1_pin_unpin();
	test2_h2g_integrity();
	test3_g2h_integrity();
	test4_benchmark();
	test5_callback_safety();
	test6_alignment();
	test7_gpu_rdma_loopback();

	/* Summary */
	printf("\n========================================\n");
	printf("Results: %d/%d passed, %d failed\n",
	       tests_passed, tests_passed + tests_failed, tests_failed);
	printf("========================================\n");

	close(fd);
	return tests_failed ? 1 : 0;
}
