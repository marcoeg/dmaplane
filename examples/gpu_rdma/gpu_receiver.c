// SPDX-License-Identifier: GPL-2.0
/*
 * Copyright (c) 2026 Graziano Labs Corp.
 *
 * gpu_receiver.c — Machine B: Ethernet -> rxe -> kernel QP -> host DRAM -> verify
 *
 * Receives GPU VRAM contents sent by gpu_sender over kernel-side RDMA.
 * The data path is: Ethernet -> rxe -> kernel qp_peer -> host DMA pages ->
 * mmap -> userspace verification.
 *
 * No GPU required on the receiver -- uses host buffers only.
 * Uses dmaplane ioctls for all RDMA operations and TCP for metadata
 * exchange. Does NOT link against libibverbs, librdmacm, or CUDA.
 *
 * Build:
 *   gcc -O2 -Wall -Wextra -I../../include -o gpu_receiver gpu_receiver.c -lm
 *
 * Run (Machine B, start BEFORE gpu_sender):
 *   sudo ./gpu_receiver <port> [ib_device]
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
#include <sys/mman.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "dmaplane_uapi.h"

/*
 * IB access flag constants -- mirrors the kernel's IB_ACCESS_* values.
 * Defined here because dmaplane_uapi.h does not export them (it would
 * create a dependency on rdma/ib_verbs.h from userspace).
 */
#define IB_ACCESS_LOCAL_WRITE	(1)
#define IB_ACCESS_REMOTE_WRITE	(1 << 1)

#define DEV_PATH	"/dev/dmaplane"
#define BUF_SIZE	(1 * 1024 * 1024)
#define NUM_FLOATS	(BUF_SIZE / sizeof(float))
#define LAYER		42

/* ──────────────────────────────────────────────────────────────────────────
 *  TCP metadata exchange
 *
 *  Wire format matches gpu_sender.c -- both sides must agree on the
 *  serialization format for cross-machine compatibility.
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

static int tcp_accept(int port)
{
	int sfd, cfd, opt = 1;
	struct sockaddr_in addr = {
		.sin_family      = AF_INET,
		.sin_port        = htons(port),
		.sin_addr.s_addr = INADDR_ANY,
	};

	sfd = socket(AF_INET, SOCK_STREAM, 0);
	if (sfd < 0)
		return -1;
	setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
	if (bind(sfd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
		close(sfd);
		return -1;
	}
	if (listen(sfd, 1) < 0) {
		close(sfd);
		return -1;
	}
	cfd = accept(sfd, NULL, NULL);
	close(sfd);
	return cfd;
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
 *  Gradient verification
 * ────────────────────────────────────────────────────────────────────────── */

static int verify_gradient(const float *buf, size_t n, int layer)
{
	float s = 1.0f / (1.0f + (float)layer);
	int errs = 0;

	/* Sample 1000 evenly spaced values */
	for (size_t i = 0; i < n; i += n / 1000) {
		float expected = s * sinf((float)i * 0.001f + (float)layer);

		if (fabsf(buf[i] - expected) > 1e-5f) {
			if (errs < 5)
				fprintf(stderr, "    [%zu] expected=%.6f got=%.6f\n",
					i, expected, buf[i]);
			errs++;
		}
	}
	return errs;
}

/* ──────────────────────────────────────────────────────────────────────────
 *  Main
 * ────────────────────────────────────────────────────────────────────────── */

int main(int argc, char *argv[])
{
	if (argc < 2) {
		fprintf(stderr, "Usage: %s <port> [ib_device]\n", argv[0]);
		return 1;
	}

	int         port   = atoi(argv[1]);
	const char *ib_dev = (argc > 2) ? argv[2] : NULL;

	int dev_fd = -1, tcp_fd = -1;
	float *data = NULL;
	int rdma_up = 0, buf_created = 0, mr_registered = 0, peer_init = 0;
	int ret = 0;

	struct dmaplane_buf_params bp = {0};
	struct dmaplane_mr_params  mr = {0};

	printf("=== dmaplane: GPU VRAM -> Network (Direct Receive) ===\n");
	printf("    Port: %d | Buffer: %zu KB\n\n", port, (size_t)BUF_SIZE >> 10);

	/* [1] Open /dev/dmaplane */
	printf("[1] Opening %s...\n", DEV_PATH);
	dev_fd = open(DEV_PATH, O_RDWR);
	if (dev_fd < 0) {
		perror("    open");
		return 1;
	}

	/* [2] Allocate host receive buffer */
	printf("[2] DMAPLANE_IOCTL_CREATE_BUFFER (%zu KB, page-backed)...\n",
	       (size_t)BUF_SIZE >> 10);
	bp.alloc_type = DMAPLANE_BUF_TYPE_PAGES;
	bp.size       = BUF_SIZE;
	bp.numa_node  = DMAPLANE_NUMA_ANY;
	if (ioctl(dev_fd, DMAPLANE_IOCTL_CREATE_BUFFER, &bp) < 0) {
		perror("    DMAPLANE_IOCTL_CREATE_BUFFER");
		ret = 1; goto out;
	}
	buf_created = 1;
	printf("    buf_id=%u\n", bp.buf_id);

	/* [3] Setup kernel RDMA */
	printf("[3] DMAPLANE_IOCTL_SETUP_RDMA");
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

	/* [4] Register host buffer as MR (with REMOTE_WRITE for future use) */
	printf("[4] DMAPLANE_IOCTL_REGISTER_MR (buf_id=%u, LOCAL_WRITE|REMOTE_WRITE)...\n",
	       bp.buf_id);
	mr.buf_id       = bp.buf_id;
	mr.access_flags = IB_ACCESS_LOCAL_WRITE |
			  IB_ACCESS_REMOTE_WRITE;
	if (ioctl(dev_fd, DMAPLANE_IOCTL_REGISTER_MR, &mr) < 0) {
		perror("    DMAPLANE_IOCTL_REGISTER_MR");
		ret = 1; goto out;
	}
	mr_registered = 1;
	printf("    mr_id=%u, lkey=0x%x, rkey=0x%x, addr=0x%llx\n",
	       mr.mr_id, mr.lkey, mr.rkey,
	       (unsigned long long)mr.addr);

	/* [5] Initialize peer QP */
	printf("[5] DMAPLANE_IOCTL_RDMA_INIT_PEER...\n");
	struct dmaplane_rdma_peer_info local_info = {0};
	if (ioctl(dev_fd, DMAPLANE_IOCTL_RDMA_INIT_PEER, &local_info) < 0) {
		perror("    DMAPLANE_IOCTL_RDMA_INIT_PEER");
		ret = 1; goto out;
	}
	peer_init = 1;
	printf("    local qp_num=%u\n", local_info.qp_num);

	/* [6] TCP: listen for sender, exchange metadata */
	printf("[6] Waiting for connection on port %d...\n", port);
	tcp_fd = tcp_accept(port);
	if (tcp_fd < 0) {
		perror("    tcp_accept");
		ret = 1; goto out;
	}
	printf("    Sender connected.\n");

	/* Receiver: recv first, then send */
	struct dmaplane_rdma_peer_info remote_info = {0};
	if (recv_metadata(tcp_fd, &remote_info) < 0) {
		fprintf(stderr, "    recv_metadata failed\n");
		ret = 1; goto out;
	}
	if (send_metadata(tcp_fd, &local_info) < 0) {
		fprintf(stderr, "    send_metadata failed\n");
		ret = 1; goto out;
	}
	printf("    remote qp_num=%u\n", remote_info.qp_num);

	/* [7] Connect peer QP to remote */
	printf("[7] DMAPLANE_IOCTL_RDMA_CONNECT_PEER...\n");
	if (ioctl(dev_fd, DMAPLANE_IOCTL_RDMA_CONNECT_PEER, &remote_info) < 0) {
		perror("    DMAPLANE_IOCTL_RDMA_CONNECT_PEER");
		ret = 1; goto out;
	}
	printf("    Connected.\n");

	/* [8] Receive data from sender */
	printf("[8] DMAPLANE_IOCTL_RDMA_REMOTE_RECV (%zu KB into MR %u)...\n",
	       (size_t)BUF_SIZE >> 10, mr.mr_id);
	printf("    Waiting for data from sender...\n");
	{
		struct dmaplane_rdma_remote_xfer_params op = {
			.mr_id = mr.mr_id,
			.size  = BUF_SIZE,
		};
		if (ioctl(dev_fd, DMAPLANE_IOCTL_RDMA_REMOTE_RECV, &op) < 0) {
			perror("    DMAPLANE_IOCTL_RDMA_REMOTE_RECV");
			fprintf(stderr, "    status=%u\n", op.status);
			ret = 1; goto out;
		}
		printf("    Received %u bytes in %llu ns\n",
		       op.size, (unsigned long long)op.elapsed_ns);
		if (op.elapsed_ns > 0) {
			double mbps = (double)op.size * 1000.0 /
				      (double)op.elapsed_ns;
			printf("    Throughput: %.1f MB/s\n", mbps);
		}
	}

	/* [9] mmap to read received data */
	printf("[9] mmap kernel buffer to verify data...\n");
	{
		struct dmaplane_mmap_info minfo = { .buf_id = bp.buf_id };

		if (ioctl(dev_fd, DMAPLANE_IOCTL_GET_MMAP_INFO, &minfo) < 0) {
			perror("    DMAPLANE_IOCTL_GET_MMAP_INFO");
			ret = 1; goto out;
		}
		data = mmap(NULL, minfo.mmap_size, PROT_READ,
			    MAP_SHARED, dev_fd, minfo.mmap_offset);
		if (data == MAP_FAILED) {
			perror("    mmap");
			data = NULL;
			ret = 1; goto out;
		}
	}

	/* [10] Verify gradient data */
	printf("[10] Verifying gradient data (layer %d)...\n", LAYER);
	printf("    First 4: [%.6f, %.6f, %.6f, %.6f]\n",
	       data[0], data[1], data[2], data[3]);

	{
		int errs = verify_gradient(data, NUM_FLOATS, LAYER);

		if (errs == 0)
			printf("    PASSED -- all sampled values match.\n");
		else
			printf("    FAILED -- %d mismatches.\n", errs);
	}

	printf("\n=== Summary ===\n");
	printf("  Path: Ethernet -> rxe -> kernel QP -> host DRAM -> mmap -> verify\n");
	printf("  Size: %zu KB | Layer: %d\n", (size_t)BUF_SIZE >> 10, LAYER);
	printf("  No GPU required on receiver.\n");

out:
	if (data)
		munmap(data, BUF_SIZE);
	/*
	 * Cleanup order:
	 *
	 * 1. RDMA_DESTROY_PEER -- moves the peer QP to ERR, drains WRs.
	 *    Must happen before deregistering the MR the QP references.
	 *
	 * 2. TEARDOWN_RDMA -- destroys loopback QPs, CQs, PD.
	 *
	 * 3. DEREGISTER_MR + DESTROY_BUFFER -- release host resources.
	 */
	if (peer_init)
		ioctl(dev_fd, DMAPLANE_IOCTL_RDMA_DESTROY_PEER);
	if (rdma_up)
		ioctl(dev_fd, DMAPLANE_IOCTL_TEARDOWN_RDMA);
	if (mr_registered) {
		uint32_t mr_id = mr.mr_id;
		ioctl(dev_fd, DMAPLANE_IOCTL_DEREGISTER_MR, &mr_id);
	}
	if (buf_created)
		ioctl(dev_fd, DMAPLANE_IOCTL_DESTROY_BUFFER, &bp.buf_id);
	if (dev_fd >= 0)
		close(dev_fd);
	if (tcp_fd >= 0)
		close(tcp_fd);

	return ret;
}
