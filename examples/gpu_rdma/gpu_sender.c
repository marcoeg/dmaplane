// SPDX-License-Identifier: GPL-2.0
/*
 * Copyright (c) 2026 Graziano Labs Corp.
 *
 * gpu_sender.c — Machine A: GPU VRAM -> kernel QP -> rxe -> Ethernet
 *
 * Sends GPU VRAM contents directly over the network using kernel-side
 * RDMA. The data path is: GPU VRAM -> WC BAR -> kernel qp_peer -> rxe ->
 * UDP packets -> Ethernet. No host staging copy.
 *
 * Uses dmaplane ioctls for all RDMA operations and TCP for metadata
 * exchange. Does NOT link against libibverbs or librdmacm.
 *
 * Build:
 *   gcc -O2 -Wall -Wextra -I../../include -o gpu_sender gpu_sender.c \
 *       -lcuda -lcudart -lm
 *
 * Run (Machine A, start AFTER gpu_receiver):
 *   sudo ./gpu_sender <receiver-ip> <port> [ib_device]
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <math.h>
#include <dirent.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <cuda_runtime.h>

#include "dmaplane_uapi.h"

#define DEV_PATH	"/dev/dmaplane"
#define BUF_SIZE	(1 * 1024 * 1024)
#define NUM_FLOATS	(BUF_SIZE / sizeof(float))
#define LAYER		42

/* ──────────────────────────────────────────────────────────────────────────
 *  TCP metadata exchange
 *
 *  We serialize dmaplane_rdma_peer_info fields into a fixed wire format
 *  to avoid ABI coupling between machines running different compiler
 *  versions or padding rules.
 * ────────────────────────────────────────────────────────────────────────── */

struct rdma_metadata {
	uint32_t qp_num;
	uint16_t lid;
	uint16_t pad;
	uint8_t  gid[16];
	uint8_t  mac[6];
	uint8_t  pad2[2];
};

static int send_metadata(int sockfd, struct dmaplane_rdma_peer_info *info)
{
	struct rdma_metadata m = {0};

	m.qp_num = info->qp_num;
	m.lid    = info->lid;
	memcpy(m.gid, info->gid, 16);
	memcpy(m.mac, info->mac, 6);
	return (send(sockfd, &m, sizeof(m), 0) == sizeof(m)) ? 0 : -1;
}

static int recv_metadata(int sockfd, struct dmaplane_rdma_peer_info *info)
{
	struct rdma_metadata m;

	if (recv(sockfd, &m, sizeof(m), MSG_WAITALL) != sizeof(m))
		return -1;
	memset(info, 0, sizeof(*info));
	info->qp_num = m.qp_num;
	info->lid    = m.lid;
	memcpy(info->gid, m.gid, 16);
	memcpy(info->mac, m.mac, 6);
	return 0;
}

static int tcp_connect(const char *ip, int port)
{
	int fd = socket(AF_INET, SOCK_STREAM, 0);
	struct sockaddr_in addr = {
		.sin_family = AF_INET,
		.sin_port = htons(port),
	};

	if (fd < 0)
		return -1;
	inet_pton(AF_INET, ip, &addr.sin_addr);
	if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
		close(fd);
		return -1;
	}
	return fd;
}

/* ──────────────────────────────────────────────────────────────────────────
 *  Auto-detect rxe device from sysfs
 * ────────────────────────────────────────────────────────────────────────── */

static int find_rxe_device(char *name, size_t len)
{
	DIR *dir = opendir("/sys/class/infiniband");
	struct dirent *ent;
	if (!dir) return -1;
	while ((ent = readdir(dir)) != NULL) {
		if (strncmp(ent->d_name, "rxe_", 4) == 0 &&
		    strlen(ent->d_name) < len) {
			snprintf(name, len, "%s", ent->d_name);
			closedir(dir);
			return 0;
		}
	}
	closedir(dir);
	return -1;
}

/* ──────────────────────────────────────────────────────────────────────────
 *  Gradient fill (deterministic for verification)
 * ────────────────────────────────────────────────────────────────────────── */

static void fill_gradient(float *buf, size_t n, int layer)
{
	float s = 1.0f / (1.0f + (float)layer);

	for (size_t i = 0; i < n; i++)
		buf[i] = s * sinf((float)i * 0.001f + (float)layer);
}

/* ──────────────────────────────────────────────────────────────────────────
 *  Main
 * ────────────────────────────────────────────────────────────────────────── */

int main(int argc, char *argv[])
{
	if (argc < 3) {
		fprintf(stderr, "Usage: %s <receiver-ip> <port> [ib_device]\n",
			argv[0]);
		return 1;
	}

	const char *ip     = argv[1];
	int         port   = atoi(argv[2]);
	const char *ib_dev = (argc > 3) ? argv[3] : NULL;

	int dev_fd = -1, tcp_fd = -1;
	void *gpu_ptr = NULL;
	float *host_buf = NULL;
	int gpu_pinned = 0, gpu_mr_registered = 0, rdma_up = 0, peer_init = 0;
	int ret = 0;

	struct dmaplane_gpu_pin_params pp = {0};
	struct dmaplane_gpu_mr_params  gm = {0};

	printf("=== dmaplane: GPU VRAM -> Network (Direct Send) ===\n");
	printf("    Target: %s:%d | Buffer: %zu KB\n\n", ip, port, (size_t)BUF_SIZE >> 10);

	/* [1] Open /dev/dmaplane */
	printf("[1] Opening %s...\n", DEV_PATH);
	dev_fd = open(DEV_PATH, O_RDWR);
	if (dev_fd < 0) {
		perror("    open");
		return 1;
	}

	/* [2] CUDA: allocate GPU memory and fill with gradient data */
	printf("[2] CUDA: allocate %zu KB, fill gradient (layer %d)...\n",
	       (size_t)BUF_SIZE >> 10, LAYER);

	cudaError_t cerr = cudaMalloc(&gpu_ptr, BUF_SIZE);
	if (cerr != cudaSuccess) {
		fprintf(stderr, "    cudaMalloc: %s\n", cudaGetErrorString(cerr));
		ret = 1; goto out;
	}

	host_buf = malloc(BUF_SIZE);
	if (!host_buf) {
		perror("    malloc");
		ret = 1; goto out;
	}

	fill_gradient(host_buf, NUM_FLOATS, LAYER);
	printf("    First 4: [%.6f, %.6f, %.6f, %.6f]\n",
	       host_buf[0], host_buf[1], host_buf[2], host_buf[3]);

	cerr = cudaMemcpy(gpu_ptr, host_buf, BUF_SIZE, cudaMemcpyHostToDevice);
	if (cerr != cudaSuccess) {
		fprintf(stderr, "    cudaMemcpy: %s\n", cudaGetErrorString(cerr));
		ret = 1; goto out;
	}
	cudaDeviceSynchronize();

	/* [3] Pin GPU VRAM via dmaplane */
	printf("[3] DMAPLANE_IOCTL_GPU_PIN...\n");
	pp.gpu_va = (uint64_t)(uintptr_t)gpu_ptr;
	pp.size   = BUF_SIZE;
	if (ioctl(dev_fd, DMAPLANE_IOCTL_GPU_PIN, &pp) < 0) {
		perror("    DMAPLANE_IOCTL_GPU_PIN");
		ret = 1; goto out;
	}
	gpu_pinned = 1;
	printf("    handle=%u, pages=%u, numa=%d\n",
	       pp.handle, pp.num_pages, pp.gpu_numa_node);

	/* [4] Setup kernel RDMA (creates PD, GID) */
	printf("[4] DMAPLANE_IOCTL_SETUP_RDMA");
	{
		struct dmaplane_rdma_setup rs = {
			.port        = 1,
			.cq_depth    = 256,
			.max_send_wr = 64,
			.max_recv_wr = 64,
		};

		if (ib_dev) {
			snprintf(rs.ib_dev_name, sizeof(rs.ib_dev_name),
				 "%s", ib_dev);
			printf(" on %s", ib_dev);
		} else {
			char auto_dev[32];
			if (find_rxe_device(auto_dev, sizeof(auto_dev)) == 0) {
				snprintf(rs.ib_dev_name, sizeof(rs.ib_dev_name),
					 "%s", auto_dev);
				printf(" on %s (auto)", auto_dev);
			} else {
				fprintf(stderr,
					"\n    No rxe device found in /sys/class/infiniband\n");
				ret = 1; goto out;
			}
		}
		printf("...\n");

		if (ioctl(dev_fd, DMAPLANE_IOCTL_SETUP_RDMA, &rs) < 0) {
			perror("    DMAPLANE_IOCTL_SETUP_RDMA");
			ret = 1; goto out;
		}
		rdma_up = 1;
		printf("    status=%u\n", rs.status);
	}

	/* [5] Register GPU buffer as RDMA MR */
	printf("[5] DMAPLANE_IOCTL_GPU_REGISTER_MR...\n");
	gm.gpu_handle = pp.handle;
	if (ioctl(dev_fd, DMAPLANE_IOCTL_GPU_REGISTER_MR, &gm) < 0) {
		perror("    DMAPLANE_IOCTL_GPU_REGISTER_MR");
		ret = 1; goto out;
	}
	gpu_mr_registered = 1;
	printf("    mr_id=%u, lkey=0x%x\n", gm.mr_id, gm.lkey);

	/* [6] Initialize peer QP */
	printf("[6] DMAPLANE_IOCTL_RDMA_INIT_PEER...\n");
	struct dmaplane_rdma_peer_info local_info = {0};
	if (ioctl(dev_fd, DMAPLANE_IOCTL_RDMA_INIT_PEER, &local_info) < 0) {
		perror("    DMAPLANE_IOCTL_RDMA_INIT_PEER");
		ret = 1; goto out;
	}
	peer_init = 1;
	printf("    local qp_num=%u\n", local_info.qp_num);

	/* [7] TCP: connect to receiver, exchange metadata */
	printf("[7] TCP connect to %s:%d...\n", ip, port);
	tcp_fd = tcp_connect(ip, port);
	if (tcp_fd < 0) {
		perror("    tcp_connect");
		ret = 1; goto out;
	}

	/* Sender sends first, then receives */
	struct dmaplane_rdma_peer_info remote_info = {0};
	if (send_metadata(tcp_fd, &local_info) < 0) {
		fprintf(stderr, "    send_metadata failed\n");
		ret = 1; goto out;
	}
	if (recv_metadata(tcp_fd, &remote_info) < 0) {
		fprintf(stderr, "    recv_metadata failed\n");
		ret = 1; goto out;
	}
	printf("    remote qp_num=%u\n", remote_info.qp_num);

	/* [8] Connect peer QP to remote */
	printf("[8] DMAPLANE_IOCTL_RDMA_CONNECT_PEER...\n");
	if (ioctl(dev_fd, DMAPLANE_IOCTL_RDMA_CONNECT_PEER, &remote_info) < 0) {
		perror("    DMAPLANE_IOCTL_RDMA_CONNECT_PEER");
		ret = 1; goto out;
	}
	printf("    Connected.\n");

	/* [9] Wait for receiver to post recv */
	printf("[9] Waiting 200ms for receiver to post recv...\n");
	usleep(200000);

	/* [10] Send GPU VRAM via RDMA SEND */
	printf("[10] DMAPLANE_IOCTL_RDMA_REMOTE_SEND (%zu KB from GPU MR %u)...\n",
	       (size_t)BUF_SIZE >> 10, gm.mr_id);
	{
		struct dmaplane_rdma_remote_xfer_params op = {
			.mr_id = gm.mr_id,
			.size  = BUF_SIZE,
		};
		if (ioctl(dev_fd, DMAPLANE_IOCTL_RDMA_REMOTE_SEND, &op) < 0) {
			perror("    DMAPLANE_IOCTL_RDMA_REMOTE_SEND");
			fprintf(stderr, "    status=%u\n", op.status);
			ret = 1; goto out;
		}
		printf("    Sent %u bytes in %llu ns\n",
		       op.size, (unsigned long long)op.elapsed_ns);
		if (op.elapsed_ns > 0) {
			double mbps = (double)op.size * 1000.0 /
				      (double)op.elapsed_ns;
			printf("    Throughput: %.1f MB/s\n", mbps);
		}
	}

	printf("\n=== Summary ===\n");
	printf("  Path: GPU VRAM -> WC BAR -> kernel QP -> rxe -> Ethernet\n");
	printf("  Size: %zu KB | Layer: %d\n", (size_t)BUF_SIZE >> 10, LAYER);
	printf("  No host staging copy.\n");

out:
	/*
	 * Cleanup order matters:
	 *
	 * 1. RDMA_DESTROY_PEER first -- moves the peer QP to ERR state,
	 *    drains any outstanding WRs. If we deregistered the MR while
	 *    the QP still had a posted WR referencing it, rxe could access
	 *    freed memory on error paths.
	 *
	 * 2. TEARDOWN_RDMA -- destroys loopback QPs, CQs, PD.
	 *
	 * 3. GPU_UNPIN must come before cudaFree -- if CUDA frees the
	 *    allocation while pages are still pinned, NVIDIA fires a
	 *    revocation callback and we race with our own teardown.
	 */
	if (peer_init)
		ioctl(dev_fd, DMAPLANE_IOCTL_RDMA_DESTROY_PEER);
	if (rdma_up)
		ioctl(dev_fd, DMAPLANE_IOCTL_TEARDOWN_RDMA);
	if (gpu_mr_registered) {
		uint32_t mr_id = gm.mr_id;
		ioctl(dev_fd, DMAPLANE_IOCTL_DEREGISTER_MR, &mr_id);
	}
	if (gpu_pinned) {
		struct dmaplane_gpu_unpin_params up = { .handle = pp.handle };
		ioctl(dev_fd, DMAPLANE_IOCTL_GPU_UNPIN, &up);
	}
	if (dev_fd >= 0)
		close(dev_fd);
	if (gpu_ptr)
		cudaFree(gpu_ptr);
	if (host_buf)
		free(host_buf);
	if (tcp_fd >= 0)
		close(tcp_fd);

	return ret;
}
