/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (c) 2026 Graziano Labs Corp.
 *
 * kvcache_common.h — Shared helpers for kvcache sender/receiver
 *
 * Thin ioctl wrappers, timing, credit tracking, recv loop helpers,
 * latency stats, and layer bitmap for KVCache chunk tracking.
 */

#ifndef _KVCACHE_COMMON_H
#define _KVCACHE_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>
#include <math.h>
#include <dirent.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <poll.h>

#include "dmaplane_uapi.h"
#include "kvcache_proto.h"

#define DMAPLANE_DEVICE "/dev/dmaplane"
#define MAX_LAYERS      1024
#define MAX_LATENCIES   (MAX_LAYERS * 64)

/*
 * All ioctl wrappers follow the same pattern: zero-init the struct (kernel
 * expects reserved/padding fields to be 0), fill input fields, call ioctl,
 * check for ioctl error (-1/errno) and kernel-reported status separately.
 */

/* ──────────────────────────────────────────────────────────────────────────
 *  Timing
 * ────────────────────────────────────────────────────────────────────────── */

static inline uint64_t now_ns(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static inline void print_trace(const char *role, uint64_t start_ns,
			       const char *fmt, ...)
{
	uint64_t elapsed = now_ns() - start_ns;
	double secs = (double)elapsed / 1e9;
	va_list ap;

	fprintf(stderr, "[%-7s T+%.3fs] ", role, secs);
	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
	fprintf(stderr, "\n");
}

/* ──────────────────────────────────────────────────────────────────────────
 *  RxE device detection
 * ────────────────────────────────────────────────────────────────────────── */

/*
 * Scan sysfs for a soft-RoCE device (rxe_<netdev>).
 * The kernel creates /sys/class/infiniband/rxe_<ifname> when
 * `rdma link add rxe_<ifname> type rxe netdev <ifname>` is run.
 */
static inline int find_rxe_device(char *name, size_t len)
{
	DIR *dir = opendir("/sys/class/infiniband");
	struct dirent *ent;

	if (!dir)
		return -1;
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

/* ──────────────────────────────────────────────────────────────────────────
 *  Ioctl wrappers
 * ────────────────────────────────────────────────────────────────────────── */

/*
 * Allocate a page-backed DMA buffer. BUF_TYPE_PAGES is used (not COHERENT)
 * because page-backed buffers support mmap and can be arbitrarily large,
 * while coherent buffers are limited by CMA pool size.
 */
static inline int dmaplane_create_buffer(int fd, uint64_t size, int numa_node,
					 uint32_t *buf_id)
{
	struct dmaplane_buf_params bp;

	memset(&bp, 0, sizeof(bp));
	bp.alloc_type = BUF_TYPE_PAGES;
	bp.size = size;
	bp.numa_node = numa_node;
	if (ioctl(fd, IOCTL_CREATE_BUFFER, &bp) < 0)
		return -1;
	*buf_id = bp.buf_id;
	return 0;
}

/*
 * Setup RDMA resources: PD, CQ pair, and loopback QP pair (QP-A + QP-B).
 * If ib_dev is NULL, auto-detects the first rxe_* device in sysfs.
 * The kernel creates two connected RC QPs for loopback testing.
 */
static inline int dmaplane_setup_rdma(int fd, const char *ib_dev,
				      uint32_t cq_depth, uint32_t send_wr,
				      uint32_t recv_wr)
{
	struct dmaplane_rdma_setup rs;
	char rxe_name[32];

	memset(&rs, 0, sizeof(rs));
	if (ib_dev) {
		snprintf(rs.ib_dev_name, sizeof(rs.ib_dev_name), "%s", ib_dev);
	} else {
		if (find_rxe_device(rxe_name, sizeof(rxe_name)) < 0) {
			fprintf(stderr, "  No rxe device found\n");
			return -1;
		}
		snprintf(rs.ib_dev_name, sizeof(rs.ib_dev_name),
			 "%s", rxe_name);
	}
	rs.port = 1;
	rs.cq_depth = cq_depth;
	rs.max_send_wr = send_wr;
	rs.max_recv_wr = recv_wr;
	if (ioctl(fd, IOCTL_SETUP_RDMA, &rs) < 0)
		return -1;
	return 0;
}

static inline void dmaplane_teardown_rdma(int fd)
{
	ioctl(fd, IOCTL_TEARDOWN_RDMA, 0);
}

/*
 * Register a memory region. When access_flags includes REMOTE_WRITE,
 * the kernel uses the fast-reg path (ib_alloc_mr + IB_WR_REG_MR) to
 * produce a valid rkey + addr for RDMA WRITE targets. Without REMOTE_WRITE,
 * it uses local_dma_lkey (rkey=0, addr=0).
 */
static inline int dmaplane_register_mr(int fd, uint32_t buf_id,
				       uint32_t access_flags,
				       struct dmaplane_mr_params *out)
{
	memset(out, 0, sizeof(*out));
	out->buf_id = buf_id;
	out->access_flags = access_flags;
	if (ioctl(fd, IOCTL_REGISTER_MR, out) < 0)
		return -1;
	return 0;
}

static inline void dmaplane_deregister_mr(int fd, uint32_t mr_id)
{
	ioctl(fd, IOCTL_DEREGISTER_MR, &mr_id);
}

static inline void dmaplane_destroy_buffer(int fd, uint32_t buf_id)
{
	ioctl(fd, IOCTL_DESTROY_BUFFER, &buf_id);
}

static inline int dmaplane_get_mmap_info(int fd, uint32_t buf_id,
					 uint64_t *offset, uint64_t *size)
{
	struct dmaplane_mmap_info mi;

	memset(&mi, 0, sizeof(mi));
	mi.buf_id = buf_id;
	if (ioctl(fd, IOCTL_GET_MMAP_INFO, &mi) < 0)
		return -1;
	*offset = mi.mmap_offset;
	*size = mi.mmap_size;
	return 0;
}

/*
 * Convenience: GET_MMAP_INFO + mmap in one call.
 * Returns mapped pointer; stores actual mmap size in *mmap_sz (for munmap).
 */
static inline void *dmaplane_mmap_buffer(int fd, uint32_t buf_id, int prot,
					 uint64_t *mmap_sz)
{
	uint64_t offset, sz;
	void *ptr;

	if (dmaplane_get_mmap_info(fd, buf_id, &offset, &sz) < 0)
		return MAP_FAILED;
	ptr = mmap(NULL, sz, prot, MAP_SHARED, fd, offset);
	if (ptr == MAP_FAILED)
		return MAP_FAILED;
	if (mmap_sz)
		*mmap_sz = sz;
	return ptr;
}

/*
 * RDMA WRITE with immediate data.
 *
 * Posts IB_WR_RDMA_WRITE_WITH_IMM on the send QP: writes 'length' bytes
 * from local MR at local_offset to remote_addr (rkey), and delivers
 * the 32-bit imm_data through the receiver's CQ completion.
 *
 * The receiver MUST have a recv WR posted — WRITE_WITH_IMM consumes
 * one recv WR to deliver the immediate value. Without a matching recv,
 * the remote QP hits RNR (Receiver Not Ready) and retries until timeout.
 *
 * use_peer_qp: 0 = loopback QP-A→QP-B, 1 = cross-machine peer QP.
 *
 * Returns 0 on success. Checks both ioctl errno (post failure) and
 * WC status (completion failure) separately.
 */
static inline int dmaplane_write_imm(int fd, uint32_t local_mr_id,
				     uint64_t local_offset,
				     uint64_t remote_addr,
				     uint32_t remote_rkey,
				     uint32_t length, uint32_t imm_data,
				     int use_peer_qp, uint64_t *elapsed_ns)
{
	struct dmaplane_write_imm_params p;

	memset(&p, 0, sizeof(p));
	p.local_mr_id = local_mr_id;
	p.length = length;
	p.local_offset = local_offset;
	p.remote_addr = remote_addr;
	p.remote_rkey = remote_rkey;
	p.imm_data = imm_data;
	p.use_peer_qp = use_peer_qp;
	if (ioctl(fd, IOCTL_RDMA_WRITE_IMM, &p) < 0)
		return -1;
	if (p.status != 0) {
		errno = EIO;
		return -1;
	}
	if (elapsed_ns)
		*elapsed_ns = p.elapsed_ns;
	return 0;
}

/*
 * Post a receive WR. Each posted recv can absorb one WRITE_WITH_IMM
 * completion (the immediate data is delivered via the recv CQ entry).
 * Must be called BEFORE the corresponding write_imm is sent.
 */
static inline int dmaplane_post_recv(int fd, uint32_t mr_id, uint32_t size,
				     int use_peer_qp)
{
	struct dmaplane_post_recv_params p;

	memset(&p, 0, sizeof(p));
	p.mr_id = mr_id;
	p.size = size;
	p.use_peer_qp = use_peer_qp;
	if (ioctl(fd, IOCTL_RDMA_POST_RECV, &p) < 0)
		return -1;
	if (p.status != 0) {
		errno = EIO;
		return -1;
	}
	return 0;
}

static inline int dmaplane_poll_recv(int fd, int use_peer_qp,
				     uint32_t timeout_ms,
				     struct dmaplane_poll_recv_params *out)
{
	memset(out, 0, sizeof(*out));
	out->use_peer_qp = use_peer_qp;
	out->timeout_ms = timeout_ms;
	if (ioctl(fd, IOCTL_RDMA_POLL_RECV, out) < 0)
		return -1;
	if (out->status != 0) {
		errno = EIO;
		return -1;
	}
	return 0;
}

/*
 * Drain all outstanding recv completions. Call before teardown on error
 * paths — leftover completions in the CQ can prevent ib_destroy_qp()
 * from succeeding. Polls with 100ms timeout until the CQ is empty.
 */
static inline void dmaplane_drain_recv_cq(int fd, int use_peer_qp)
{
	struct dmaplane_poll_recv_params rp;

	for (;;) {
		memset(&rp, 0, sizeof(rp));
		rp.use_peer_qp = use_peer_qp;
		rp.timeout_ms = 100;
		if (ioctl(fd, IOCTL_RDMA_POLL_RECV, &rp) < 0)
			break;
		if (rp.status != 0)
			break;
	}
}

/* ──────────────────────────────────────────────────────────────────────────
 *  Peer RDMA wrappers — INIT_PEER, CONNECT_PEER, DESTROY_PEER
 *
 *  These create/connect/destroy the cross-machine peer QP (qp_peer +
 *  cq_peer). Used by kvcache_sender and kvcache_receiver in --peer mode.
 *  The underlying kernel ioctls were added in Phase 8.
 * ────────────────────────────────────────────────────────────────────────── */

/*
 * dmaplane_init_peer — create qp_peer + cq_peer, return local QP metadata.
 * Must be called after dmaplane_setup_rdma() (needs PD + GID).
 * Fills info with: qp_num, lid, gid[16], mac[6] for TCP exchange.
 */
static inline int dmaplane_init_peer(int fd,
				     struct dmaplane_rdma_peer_info *info)
{
	memset(info, 0, sizeof(*info));
	if (ioctl(fd, IOCTL_RDMA_INIT_PEER, info) < 0)
		return -1;
	if (info->status != 0) {
		errno = EIO;
		return -1;
	}
	return 0;
}

/*
 * dmaplane_connect_peer — connect qp_peer using remote machine's metadata.
 * Transitions qp_peer: INIT → RTR → RTS using remote's QPN, GID, DMAC.
 */
static inline int dmaplane_connect_peer(int fd,
					struct dmaplane_rdma_peer_info *remote)
{
	if (ioctl(fd, IOCTL_RDMA_CONNECT_PEER, remote) < 0)
		return -1;
	return 0;
}

/*
 * dmaplane_destroy_peer — destroy qp_peer + cq_peer.
 * Moves QP to ERR state (drains WRs), then destroys QP and CQ.
 * Must be called before dmaplane_teardown_rdma().
 */
static inline void dmaplane_destroy_peer(int fd)
{
	ioctl(fd, IOCTL_RDMA_DESTROY_PEER, 0);
}

/* ──────────────────────────────────────────────────────────────────────────
 *  TCP metadata exchange + credit flow (peer mode)
 *
 *  The sender and receiver exchange a tcp_metadata struct over TCP to
 *  share QP connection info (GID, QPN, MAC) and MR targeting info
 *  (addr, rkey). The sender also sends its transfer config (layers,
 *  chunks, chunk_size) so the receiver can size its buffer.
 *
 *  After metadata exchange and CONNECT_PEER, a credit flow protocol
 *  uses TCP as a side channel: each 1-byte message = 1 recv credit.
 *  The receiver sends a credit after each recv completion + replenish.
 *  The sender blocks on TCP read when credits hit 0.
 * ────────────────────────────────────────────────────────────────────────── */

struct tcp_metadata {
	uint8_t  gid[16];          /* from INIT_PEER — 16-byte GID */
	uint8_t  mac[6];           /* from INIT_PEER — Ethernet MAC for RoCEv2 */
	uint16_t lid;              /* from INIT_PEER — LID (unused for RoCE) */
	uint32_t qpn;              /* from INIT_PEER — QP number */
	uint64_t mr_addr;          /* receiver's MR addr for RDMA targeting */
	uint32_t mr_rkey;          /* receiver's MR rkey */
	uint32_t num_layers;       /* sender's config */
	uint32_t chunks_per_layer; /* sender's config */
	uint32_t chunk_size;       /* sender's config */
	uint32_t credit_window;    /* sender's config */
	uint32_t _pad;             /* alignment */
	uint64_t buf_size;         /* sender's total_bytes */
};

/* TCP connect — sender side. Returns connected socket fd or -1. */
static inline int tcp_connect(const char *ip, int port)
{
	int fd;
	struct sockaddr_in addr;

	fd = socket(AF_INET, SOCK_STREAM, 0);
	if (fd < 0)
		return -1;
	memset(&addr, 0, sizeof(addr));
	addr.sin_family = AF_INET;
	addr.sin_port = htons(port);
	if (inet_pton(AF_INET, ip, &addr.sin_addr) != 1) {
		close(fd);
		return -1;
	}
	if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
		close(fd);
		return -1;
	}
	return fd;
}

/* TCP listen + accept — receiver side. Returns client socket fd or -1. */
static inline int tcp_listen_accept(int port)
{
	int sfd, cfd, opt = 1;
	struct sockaddr_in addr;

	memset(&addr, 0, sizeof(addr));
	addr.sin_family = AF_INET;
	addr.sin_port = htons(port);
	addr.sin_addr.s_addr = INADDR_ANY;

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
	close(sfd); /* close listener after accept */
	return cfd;
}

/* Reliable send — handles partial writes. */
static inline int tcp_send_all(int sockfd, const void *buf, size_t len)
{
	const uint8_t *p = (const uint8_t *)buf;

	while (len > 0) {
		ssize_t n = send(sockfd, p, len, 0);

		if (n <= 0)
			return -1;
		p += n;
		len -= (size_t)n;
	}
	return 0;
}

/* Reliable recv — handles partial reads. */
static inline int tcp_recv_all(int sockfd, void *buf, size_t len)
{
	uint8_t *p = (uint8_t *)buf;

	while (len > 0) {
		ssize_t n = recv(sockfd, p, len, MSG_WAITALL);

		if (n <= 0)
			return -1;
		p += n;
		len -= (size_t)n;
	}
	return 0;
}

static inline int tcp_send_metadata(int sockfd, struct tcp_metadata *m)
{
	return tcp_send_all(sockfd, m, sizeof(*m));
}

static inline int tcp_recv_metadata(int sockfd, struct tcp_metadata *m)
{
	return tcp_recv_all(sockfd, m, sizeof(*m));
}

/*
 * Fill tcp_metadata with local INIT_PEER info. Caller fills MR and
 * config fields separately.
 */
static inline void tcp_metadata_from_peer_info(struct tcp_metadata *m,
					       struct dmaplane_rdma_peer_info *pi)
{
	memcpy(m->gid, pi->gid, 16);
	memcpy(m->mac, pi->mac, 6);
	m->lid = pi->lid;
	m->qpn = pi->qp_num;
}

/*
 * Build dmaplane_rdma_peer_info from received tcp_metadata for
 * CONNECT_PEER ioctl.
 */
static inline void tcp_metadata_to_peer_info(struct tcp_metadata *m,
					     struct dmaplane_rdma_peer_info *pi)
{
	memset(pi, 0, sizeof(*pi));
	pi->qp_num = m->qpn;
	pi->lid = m->lid;
	memcpy(pi->gid, m->gid, 16);
	memcpy(pi->mac, m->mac, 6);
}

/* Send one credit byte (receiver → sender). */
static inline int tcp_send_credit(int sockfd)
{
	uint8_t one = 1;

	return tcp_send_all(sockfd, &one, 1);
}

/*
 * Non-blocking credit read — returns number of credits received (0 if none).
 * Uses MSG_DONTWAIT so sender can interleave credit checks with the send loop.
 */
static inline int tcp_recv_credits_nonblock(int sockfd)
{
	uint8_t buf[64];
	ssize_t n = recv(sockfd, buf, sizeof(buf), MSG_DONTWAIT);

	if (n <= 0)
		return 0;
	return (int)n;
}

/*
 * Blocking credit wait — blocks until at least 1 credit byte arrives.
 * Used when the sender has 0 credits and must stall. Returns total
 * credits received (1 + any extras drained non-blocking), or -1 on error.
 */
static inline int tcp_recv_credits_blocking(int sockfd)
{
	uint8_t byte;
	ssize_t n;
	int extra;

	/* Block for exactly 1 byte */
	n = recv(sockfd, &byte, 1, 0);
	if (n <= 0)
		return -1;

	/* Drain any additional credits that arrived */
	extra = tcp_recv_credits_nonblock(sockfd);
	return 1 + extra;
}

#ifdef HAVE_CUDA
#include <cuda_runtime.h>

/*
 * GPU P2P pin: maps GPU BAR pages into the kernel's IOMMU via
 * nvidia_p2p_get_pages(). Both gpu_va and size must be 64KB-aligned
 * (NVIDIA P2P API requirement). Returns a handle for gpu_register_mr.
 */
static inline int dmaplane_gpu_pin(int fd, uint64_t gpu_va, uint64_t size,
				   struct dmaplane_gpu_pin_params *out)
{
	memset(out, 0, sizeof(*out));
	out->gpu_va = gpu_va;
	out->size = size;
	if (ioctl(fd, IOCTL_GPU_PIN, out) < 0)
		return -1;
	return 0;
}

static inline void dmaplane_gpu_unpin(int fd, uint32_t handle)
{
	struct dmaplane_gpu_unpin_params up;

	memset(&up, 0, sizeof(up));
	up.handle = handle;
	ioctl(fd, IOCTL_GPU_UNPIN, &up);
}

/*
 * Register a GPU-backed MR from pinned GPU pages. The MR uses
 * local_dma_lkey (suitable as a WRITE_IMM source). For REMOTE_WRITE
 * targets, use a host buffer with the fast-reg path instead.
 */
static inline int dmaplane_gpu_register_mr(int fd, uint32_t gpu_handle,
					   struct dmaplane_gpu_mr_params *out)
{
	memset(out, 0, sizeof(*out));
	out->gpu_handle = gpu_handle;
	if (ioctl(fd, IOCTL_GPU_REGISTER_MR, out) < 0)
		return -1;
	return 0;
}
#endif /* HAVE_CUDA */

/* ──────────────────────────────────────────────────────────────────────────
 *  Credit window tracker
 * ────────────────────────────────────────────────────────────────────────── */

/*
 * Credits track the number of recv WRs currently posted on the receiver.
 * Each WRITE_IMM consumes one recv WR on completion, so credits is
 * decremented after each send. When credits reaches 0, the sender must
 * stall: poll a recv completion, then post a new recv (replenish) to
 * restore one credit before sending again.
 *
 * Only recv_loop_replenish() increments credits — it posts a new recv
 * WR, which is the actual event that creates capacity for another send.
 */
struct credit_tracker {
	int      credits;          /* recv WRs available (can send this many) */
	int      max_credits;      /* initial credit window size */
	int      total_replenished;/* total replenish operations */
	int      stall_count;      /* times sender blocked on credits=0 */
	uint64_t total_stall_ns;   /* total time spent stalled */
};

static inline void credit_init(struct credit_tracker *ct, int max_credits)
{
	memset(ct, 0, sizeof(*ct));
	ct->credits = max_credits;
	ct->max_credits = max_credits;
}

/* ──────────────────────────────────────────────────────────────────────────
 *  Latency stats
 * ────────────────────────────────────────────────────────────────────────── */

struct latency_stats {
	uint64_t *samples;
	int       count;
	int       capacity;
};

static inline int latency_stats_init(struct latency_stats *ls, int capacity)
{
	ls->samples = calloc(capacity, sizeof(uint64_t));
	if (!ls->samples)
		return -1;
	ls->count = 0;
	ls->capacity = capacity;
	return 0;
}

static inline void latency_stats_add(struct latency_stats *ls, uint64_t ns)
{
	if (ls->count < ls->capacity)
		ls->samples[ls->count++] = ns;
}

static inline void latency_stats_free(struct latency_stats *ls)
{
	free(ls->samples);
	ls->samples = NULL;
}

/* Comparison for qsort — avoids subtraction overflow on uint64_t */
static int latency_cmp(const void *a, const void *b)
{
	uint64_t va = *(const uint64_t *)a;
	uint64_t vb = *(const uint64_t *)b;

	return (va > vb) - (va < vb);
}

static inline void latency_stats_compute(struct latency_stats *ls,
					 double *avg_ms, double *p50_ms,
					 double *p99_ms)
{
	if (ls->count == 0) {
		*avg_ms = *p50_ms = *p99_ms = 0.0;
		return;
	}
	qsort(ls->samples, ls->count, sizeof(uint64_t), latency_cmp);
	uint64_t sum = 0;

	for (int i = 0; i < ls->count; i++)
		sum += ls->samples[i];
	*avg_ms = (double)sum / (double)ls->count / 1e6;
	*p50_ms = (double)ls->samples[ls->count / 2] / 1e6;
	*p99_ms = (double)ls->samples[(int)(ls->count * 0.99)] / 1e6;
}

static inline void latency_stats_print(struct latency_stats *ls,
				       const char *role, uint64_t start_ns)
{
	double avg, p50, p99;

	latency_stats_compute(ls, &avg, &p50, &p99);
	print_trace(role, start_ns,
		    "Per-chunk: avg=%.1fms P50=%.1fms P99=%.1fms",
		    avg, p50, p99);
}

/* ──────────────────────────────────────────────────────────────────────────
 *  Layer bitmap — tracks which chunks per layer have been received
 * ────────────────────────────────────────────────────────────────────────── */

/*
 * Tracks chunk arrival per layer using one bit per chunk.
 * Each uint64_t supports up to 64 chunks per layer — sufficient
 * for KVCache workloads (typically 4-16 chunks per layer).
 */
struct layer_bitmap {
	uint64_t bits[MAX_LAYERS];
	int      chunks_per_layer;
	int      num_layers;
};

static inline void bitmap_init(struct layer_bitmap *bm, int num_layers,
			       int chunks_per_layer)
{
	memset(bm, 0, sizeof(*bm));
	bm->num_layers = num_layers;
	bm->chunks_per_layer = chunks_per_layer;
}

static inline void bitmap_set(struct layer_bitmap *bm, int layer, int chunk)
{
	if (layer < MAX_LAYERS && chunk < 64)
		bm->bits[layer] |= (1ULL << chunk);
}

static inline int bitmap_layer_complete(struct layer_bitmap *bm, int layer)
{
	uint64_t expected = (1ULL << bm->chunks_per_layer) - 1;

	return (bm->bits[layer] & expected) == expected;
}

static inline int bitmap_all_complete(struct layer_bitmap *bm)
{
	for (int i = 0; i < bm->num_layers; i++)
		if (!bitmap_layer_complete(bm, i))
			return 0;
	return 1;
}

static inline int bitmap_total_received(struct layer_bitmap *bm)
{
	int total = 0;

	for (int i = 0; i < bm->num_layers; i++)
		total += __builtin_popcountll(bm->bits[i]);
	return total;
}

/* ──────────────────────────────────────────────────────────────────────────
 *  Recv loop helpers — shared between sender (loopback) and receiver (peer)
 * ────────────────────────────────────────────────────────────────────────── */

/*
 * Poll one recv completion, decode imm_data, update bitmap + latency.
 * Returns: 0=chunk received, 1=SENTINEL, -1=error
 */
static inline int recv_loop_poll_and_track(int fd, int use_peer_qp,
					   uint32_t timeout_ms,
					   struct layer_bitmap *bm,
					   struct latency_stats *ls,
					   uint64_t *chunk_start_ns)
{
	struct dmaplane_poll_recv_params rp;
	uint64_t elapsed;

	if (dmaplane_poll_recv(fd, use_peer_qp, timeout_ms, &rp) < 0)
		return -1;

	elapsed = now_ns() - *chunk_start_ns;
	latency_stats_add(ls, elapsed);

	if (rp.imm_data == KVCACHE_SENTINEL)
		return 1;

	uint32_t layer = KVCACHE_IMM_LAYER(rp.imm_data);
	uint32_t chunk = KVCACHE_IMM_CHUNK(rp.imm_data);

	bitmap_set(bm, layer, chunk);
	return 0;
}

/*
 * Post one recv WR and increment credits. This is the ONLY place
 * credits are incremented — posting a recv is the physical event
 * that creates capacity for one more WRITE_IMM send.
 */
static inline int recv_loop_replenish(int fd, uint32_t mr_id, uint32_t size,
				      int use_peer_qp,
				      struct credit_tracker *ct)
{
	if (dmaplane_post_recv(fd, mr_id, size, use_peer_qp) < 0)
		return -1;
	ct->credits++;
	ct->total_replenished++;
	return 0;
}

/* ──────────────────────────────────────────────────────────────────────────
 *  Pattern fill / verify — shared between sender and receiver
 * ────────────────────────────────────────────────────────────────────────── */

/*
 * fill_pattern / verify_pattern — deterministic per-chunk byte pattern.
 *
 * Each byte is f(layer, chunk, offset) so any chunk that lands at the wrong
 * destination offset or gets corrupted in transit is immediately detectable.
 * The primes (7, 13) ensure different layers/chunks don't alias.
 */
static inline void fill_pattern(uint8_t *buf, int layer, int chunk,
				uint32_t chunk_size)
{
	for (uint32_t i = 0; i < chunk_size; i++)
		buf[i] = (uint8_t)((layer * 7 + chunk * 13 + i) & 0xFF);
}

static inline int verify_pattern(const uint8_t *buf, int layer, int chunk,
				 uint32_t chunk_size)
{
	int errs = 0;

	for (uint32_t i = 0; i < chunk_size; i++) {
		uint8_t expected = (uint8_t)((layer * 7 + chunk * 13 + i)
					     & 0xFF);

		if (buf[i] != expected) {
			if (errs < 3)
				fprintf(stderr,
					"    MISMATCH L%d C%d byte %u: "
					"expected 0x%02x got 0x%02x\n",
					layer, chunk, i, expected, buf[i]);
			errs++;
		}
	}
	return errs;
}

#endif /* _KVCACHE_COMMON_H */
