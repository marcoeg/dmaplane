// SPDX-License-Identifier: GPL-2.0
/*
 * Copyright (c) 2026 Graziano Labs Corp.
 *
 * kvcache_receiver.c — Chunked WRITEIMM KVCache receiver
 *
 * Loopback mode: minimal self-test — creates one MR, posts a few recvs,
 * and directs the user to kvcache_sender for the full loopback test.
 *
 * Peer mode (--port): accepts a TCP connection from kvcache_sender,
 * exchanges QP/MR metadata, then receives WRITEIMM chunks on the peer QP
 * (qp_peer). Credit flow uses TCP as a side channel (1 byte = 1 credit).
 *
 * Build:  make -C examples/kvcache
 * Run:    sudo ./kvcache_receiver [--loopback] [--port 9876] [--verify]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <errno.h>

#include "kvcache_common.h"

/* ──────────────────────────────────────────────────────────────────────────
 *  Configuration
 * ────────────────────────────────────────────────────────────────────────── */

struct config {
	int      loopback;        /* 1=loopback (default), 0=peer */
	int      use_gpu;
	int      verify;
	int      port;            /* TCP port for peer mode (default: 9876) */
	char    *ib_dev;
};

static void config_defaults(struct config *c)
{
	memset(c, 0, sizeof(*c));
	c->loopback = 1;
	c->port = 9876;
}

static void usage(const char *prog)
{
	fprintf(stderr,
		"Usage: %s [OPTIONS]\n"
		"  --loopback            Minimal loopback self-test (default)\n"
		"  --port <port>         Listen for remote sender (default: 9876)\n"
		"  --gpu                 Use GPU-backed receive MR (requires CUDA)\n"
		"  --ib-dev <name>       IB device name (auto-detect if omitted)\n"
		"  --verify              Verify data integrity\n",
		prog);
}

static int parse_args(int argc, char *argv[], struct config *c)
{
	static struct option opts[] = {
		{"loopback",       no_argument,       NULL, 'l'},
		{"port",           required_argument, NULL, 'P'},
		{"gpu",            no_argument,       NULL, 'g'},
		{"ib-dev",         required_argument, NULL, 'd'},
		{"verify",         no_argument,       NULL, 'v'},
		{"help",           no_argument,       NULL, 'h'},
		{NULL, 0, NULL, 0},
	};
	int opt;

	while ((opt = getopt_long(argc, argv, "lP:gd:vh",
				  opts, NULL)) != -1) {
		switch (opt) {
		case 'l': c->loopback = 1; break;
		case 'P':
			c->loopback = 0;
			c->port = atoi(optarg);
			break;
		case 'g': c->use_gpu = 1; break;
		case 'd': c->ib_dev = optarg; break;
		case 'v': c->verify = 1; break;
		case 'h': usage(argv[0]); return -1;
		default:  usage(argv[0]); return -1;
		}
	}
	return 0;
}

/* ──────────────────────────────────────────────────────────────────────────
 *  Peer mode — cross-machine receiver
 * ────────────────────────────────────────────────────────────────────────── */

/*
 * run_peer — cross-machine WRITEIMM receiver via TCP + peer QP.
 *
 * Accepts a TCP connection from kvcache_sender, exchanges QP/MR
 * metadata, then receives KVCache chunks via RDMA WRITE WITH IMMEDIATE
 * on the peer QP (qp_peer → cq_peer).
 *
 * Credit flow: after each recv completion, the receiver posts a fresh
 * recv WR and sends a 1-byte credit message on TCP so the sender knows
 * it can issue another WRITEIMM.
 *
 * Transfer flow:
 *   1. Setup RDMA (creates PD, CQs, loopback QP pair — needed for PD)
 *   2. INIT_PEER — creates qp_peer + cq_peer, returns local QP metadata
 *   3. TCP listen + accept — receive sender's config + QP metadata
 *   4. Create receive buffer sized from sender's config
 *   5. Register MR (LOCAL_WRITE | REMOTE_WRITE)
 *   6. Send our metadata back (QP info + MR addr/rkey)
 *   7. CONNECT_PEER — transitions qp_peer using sender's QPN/GID/MAC
 *   8. Pre-post credit_window recv WRs
 *   9. Send "ready" signal to sender
 *  10. Recv loop: poll CQ → decode imm → replenish recv → send credit
 *  11. On sentinel: send "done" ack, report summary
 *  12. Optionally verify data integrity via mmap
 */
static int run_peer(int fd, struct config *cfg)
{
	const char *role = "RECV";
	uint64_t start_ns = now_ns();
	int ret = 0;

	/* Resource tracking */
	uint32_t buf_id = 0;
	int buf_ok = 0, rdma_up = 0, mr_ok = 0, peer_init = 0;
	int tcp_fd = -1;
	struct dmaplane_mr_params mr = {0};
	struct latency_stats ls = {0};
	struct credit_tracker ct;
	struct layer_bitmap bm;

	/* Transfer config — filled from sender's metadata */
	int num_layers = 0, chunks_per_layer = 0, total_chunks = 0;
	uint32_t chunk_size = 0;
	int credit_window = 0;
	uint64_t buf_size = 0;

	/* [1] Setup RDMA — needed for PD and GID */
	if (dmaplane_setup_rdma(fd, cfg->ib_dev, 256, 80, 80) < 0) {
		perror("  SETUP_RDMA");
		return 1;
	}
	rdma_up = 1;

	/* [2] Initialize peer QP — creates qp_peer + cq_peer */
	{
		struct dmaplane_rdma_peer_info local_info;

		if (dmaplane_init_peer(fd, &local_info) < 0) {
			perror("  INIT_PEER");
			ret = 1; goto out;
		}
		peer_init = 1;
		print_trace(role, start_ns, "INIT_PEER: qp_num=%u",
			    local_info.qp_num);

		/* [3] TCP listen + accept */
		print_trace(role, start_ns, "Listening on port %d...",
			    cfg->port);
		tcp_fd = tcp_listen_accept(cfg->port);
		if (tcp_fd < 0) {
			perror("  tcp_listen_accept");
			ret = 1; goto out;
		}
		print_trace(role, start_ns, "Sender connected.");

		/* Receive sender's metadata (QP info + transfer config) */
		struct tcp_metadata sender_meta;

		if (tcp_recv_metadata(tcp_fd, &sender_meta) < 0) {
			fprintf(stderr, "  tcp_recv_metadata failed\n");
			ret = 1; goto out;
		}

		num_layers = sender_meta.num_layers;
		chunks_per_layer = sender_meta.chunks_per_layer;
		chunk_size = sender_meta.chunk_size;
		credit_window = sender_meta.credit_window;
		buf_size = sender_meta.buf_size;
		total_chunks = num_layers * chunks_per_layer;

		print_trace(role, start_ns,
			    "Config from sender: %d layers x %d chunks x %u KB "
			    "= %llu MB, credit_window=%d",
			    num_layers, chunks_per_layer,
			    chunk_size >> 10,
			    (unsigned long long)(buf_size >> 20),
			    credit_window);

		/* [4] Create receive buffer sized from sender's config */
		if (dmaplane_create_buffer(fd, buf_size, DMAPLANE_NUMA_ANY,
					   &buf_id) < 0) {
			perror("  CREATE_BUFFER");
			ret = 1; goto out;
		}
		buf_ok = 1;

		/* [5] Register MR with REMOTE_WRITE for RDMA targeting */
		if (dmaplane_register_mr(fd, buf_id,
					 DMAPLANE_IB_ACCESS_LOCAL_WRITE |
					 DMAPLANE_IB_ACCESS_REMOTE_WRITE,
					 &mr) < 0) {
			perror("  REGISTER_MR");
			ret = 1; goto out;
		}
		mr_ok = 1;

		print_trace(role, start_ns,
			    "MR id=%u rkey=0x%x addr=0x%llx",
			    mr.mr_id, mr.rkey,
			    (unsigned long long)mr.addr);

		/* [6] Send our metadata back (QP info + MR addr/rkey) */
		struct tcp_metadata recv_meta = {0};

		tcp_metadata_from_peer_info(&recv_meta, &local_info);
		recv_meta.mr_addr = mr.addr;
		recv_meta.mr_rkey = mr.rkey;

		if (tcp_send_metadata(tcp_fd, &recv_meta) < 0) {
			fprintf(stderr, "  tcp_send_metadata failed\n");
			ret = 1; goto out;
		}

		/* [7] Connect peer QP using sender's metadata */
		struct dmaplane_rdma_peer_info remote_info;

		tcp_metadata_to_peer_info(&sender_meta, &remote_info);
		if (dmaplane_connect_peer(fd, &remote_info) < 0) {
			perror("  CONNECT_PEER");
			ret = 1; goto out;
		}
		print_trace(role, start_ns,
			    "CONNECT_PEER: connected to sender qpn=%u.",
			    sender_meta.qpn);
	}

	/* Initialize tracking structures */
	if (latency_stats_init(&ls, total_chunks + 16) < 0) {
		fprintf(stderr, "  Failed to allocate latency stats\n");
		ret = 1; goto out;
	}
	bitmap_init(&bm, num_layers, chunks_per_layer);

	/* [8] Pre-post recv WRs on peer QP */
	{
		int initial_recvs = credit_window;

		if (initial_recvs > total_chunks + 1)
			initial_recvs = total_chunks + 1;
		for (int i = 0; i < initial_recvs; i++) {
			if (dmaplane_post_recv(fd, mr.mr_id, chunk_size,
					       1) < 0) {
				perror("  POST_RECV (pre-post)");
				ret = 1; goto out;
			}
		}
		credit_init(&ct, initial_recvs);
		print_trace(role, start_ns, "Pre-posted %d recvs",
			    initial_recvs);
	}

	/* [9] Send "ready" signal — sender waits for this before sending */
	{
		uint8_t ready = 1;

		if (tcp_send_all(tcp_fd, &ready, 1) < 0) {
			fprintf(stderr, "  Failed to send ready signal\n");
			ret = 1; goto out;
		}
	}

	/* [10] Recv loop — poll peer CQ, decode imm, replenish, send credit */
	{
		uint64_t iter_start = now_ns();
		double total_mb = (double)buf_size / (1024.0 * 1024.0);

		while (!bitmap_all_complete(&bm)) {
			uint64_t cs = now_ns();
			int rc = recv_loop_poll_and_track(fd, 1, 10000,
							  &bm, &ls, &cs);

			if (rc < 0) {
				fprintf(stderr, "  poll failed\n");
				ret = 1; goto out;
			}
			if (rc == 1)
				break; /* SENTINEL */

			/* Replenish recv + send TCP credit to sender */
			if (recv_loop_replenish(fd, mr.mr_id, chunk_size,
						1, &ct) < 0) {
				perror("  POST_RECV (replenish)");
				ret = 1; goto out;
			}
			if (tcp_send_credit(tcp_fd) < 0) {
				fprintf(stderr, "  tcp_send_credit failed\n");
				ret = 1; goto out;
			}
		}

		/* Drain sentinel completion if bitmap completed first */
		{
			struct dmaplane_poll_recv_params rp;

			dmaplane_poll_recv(fd, 1, 1000, &rp);
		}

		/* [11] Send "done" ack to sender */
		{
			uint8_t done = 1;

			if (tcp_send_all(tcp_fd, &done, 1) < 0) {
				fprintf(stderr,
					"  Failed to send done ack\n");
			}
		}

		uint64_t iter_elapsed = now_ns() - iter_start;
		double iter_secs = (double)iter_elapsed / 1e9;
		double throughput = total_mb / iter_secs;

		print_trace(role, start_ns,
			    "Transfer complete: %.0f MB in %.3fs",
			    total_mb, iter_secs);
		print_trace(role, start_ns,
			    "  Throughput: %.1f MB/s", throughput);
		print_trace(role, start_ns,
			    "  Chunks: %d/%d received",
			    bitmap_total_received(&bm), total_chunks);
		latency_stats_print(&ls, role, start_ns);
	}

	/* [12] Verify data integrity */
	if (cfg->verify) {
		uint64_t msz;
		uint8_t *dst_ptr = dmaplane_mmap_buffer(
			fd, buf_id, PROT_READ, &msz);

		if (dst_ptr == MAP_FAILED) {
			perror("  mmap (verify)");
			ret = 1; goto out;
		}

		int total_errs = 0;

		for (int l = 0; l < num_layers; l++) {
			for (int c = 0; c < chunks_per_layer; c++) {
				int idx = l * chunks_per_layer + c;
				uint64_t off = (uint64_t)idx * chunk_size;
				int errs = verify_pattern(
					dst_ptr + off, l, c, chunk_size);

				total_errs += errs;
			}
		}
		munmap(dst_ptr, msz);

		if (total_errs == 0)
			print_trace(role, start_ns,
				    "  Data integrity: PASS");
		else {
			print_trace(role, start_ns,
				    "  Data integrity: FAIL (%d errors)",
				    total_errs);
			ret = 1;
		}
	}

out:
	/* Cleanup order:
	 *   1. Close TCP (stops credit flow)
	 *   2. Drain recv CQ (prevents QP teardown failures)
	 *   3. Destroy peer QP (drains WRs on qp_peer)
	 *   4. Teardown RDMA (destroys loopback QPs, CQs, PD)
	 *   5. Deregister MR
	 *   6. Destroy buffer */
	if (tcp_fd >= 0)
		close(tcp_fd);
	dmaplane_drain_recv_cq(fd, 1);
	if (peer_init)
		dmaplane_destroy_peer(fd);
	if (rdma_up)
		dmaplane_teardown_rdma(fd);
	if (mr_ok)
		dmaplane_deregister_mr(fd, mr.mr_id);
	if (buf_ok)
		dmaplane_destroy_buffer(fd, buf_id);
	latency_stats_free(&ls);
	return ret;
}

/* ──────────────────────────────────────────────────────────────────────────
 *  Loopback self-test (minimal)
 * ────────────────────────────────────────────────────────────────────────── */

/*
 * run_loopback — minimal self-test for the receiver path.
 *
 * Verifies that the receiver can: create a fast-reg MR (REMOTE_WRITE),
 * post recv WRs, and mmap the buffer for read/write.  Does NOT send
 * any data — use kvcache_sender --loopback for the full round-trip test.
 *
 * This exists so 'make test' can exercise the receiver binary independently.
 */
static int run_loopback(int fd, struct config *cfg)
{
	const char *role = "RECV";
	uint64_t start_ns = now_ns();
	int ret = 0;

	uint32_t buf_id = 0;
	int buf_ok = 0, rdma_up = 0, mr_ok = 0;
	struct dmaplane_mr_params mr = {0};
	uint64_t test_size = 4ULL * 1024 * 1024; /* 4 MB test buffer */

	print_trace(role, start_ns,
		    "Loopback self-test (minimal): creating MR + posting recvs");

	/* [1] Setup RDMA */
	if (dmaplane_setup_rdma(fd, cfg->ib_dev, 64, 32, 32) < 0) {
		perror("  SETUP_RDMA");
		ret = 1; goto out;
	}
	rdma_up = 1;

	/* [2] Allocate + register MR */
	if (dmaplane_create_buffer(fd, test_size, DMAPLANE_NUMA_ANY,
				   &buf_id) < 0) {
		perror("  CREATE_BUFFER");
		ret = 1; goto out;
	}
	buf_ok = 1;

	if (dmaplane_register_mr(fd, buf_id,
				 DMAPLANE_IB_ACCESS_LOCAL_WRITE |
				 DMAPLANE_IB_ACCESS_REMOTE_WRITE,
				 &mr) < 0) {
		perror("  REGISTER_MR");
		ret = 1; goto out;
	}
	mr_ok = 1;

	print_trace(role, start_ns,
		    "MR id=%u rkey=0x%x addr=0x%llx size=%llu KB",
		    mr.mr_id, mr.rkey, (unsigned long long)mr.addr,
		    (unsigned long long)(test_size >> 10));

	/* [3] Post a few test recvs to verify RDMA plumbing */
	int test_recvs = 4;

	for (int i = 0; i < test_recvs; i++) {
		if (dmaplane_post_recv(fd, mr.mr_id, 1024 * 1024, 0) < 0) {
			perror("  POST_RECV");
			ret = 1; goto out;
		}
	}
	print_trace(role, start_ns, "Posted %d test recvs — PASS", test_recvs);

	/* [4] Verify mmap works */
	{
		uint64_t msz;
		void *ptr = dmaplane_mmap_buffer(fd, buf_id,
						 PROT_READ | PROT_WRITE,
						 &msz);

		if (ptr == MAP_FAILED) {
			perror("  mmap");
			ret = 1; goto out;
		}
		/* Write a marker, read it back */
		memset(ptr, 0xAB, 4096);
		uint8_t *p = (uint8_t *)ptr;

		if (p[0] == 0xAB && p[4095] == 0xAB)
			print_trace(role, start_ns, "mmap read/write — PASS");
		else {
			print_trace(role, start_ns, "mmap read/write — FAIL");
			ret = 1;
		}
		munmap(ptr, msz);
	}

	print_trace(role, start_ns,
		    "Loopback self-test complete. For full KVCache test, run:");
	print_trace(role, start_ns,
		    "  sudo ./kvcache_sender --loopback --verify");

out:
	dmaplane_drain_recv_cq(fd, 0);
	if (rdma_up)
		dmaplane_teardown_rdma(fd);
	if (mr_ok)
		dmaplane_deregister_mr(fd, mr.mr_id);
	if (buf_ok)
		dmaplane_destroy_buffer(fd, buf_id);
	return ret;
}

/* ──────────────────────────────────────────────────────────────────────────
 *  Main
 * ────────────────────────────────────────────────────────────────────────── */

int main(int argc, char *argv[])
{
	struct config cfg;
	int fd, ret;

	config_defaults(&cfg);
	if (parse_args(argc, argv, &cfg) < 0)
		return 1;

	fd = open(DMAPLANE_DEVICE, O_RDWR);
	if (fd < 0) {
		perror("open " DMAPLANE_DEVICE);
		return 1;
	}

	if (cfg.loopback)
		ret = run_loopback(fd, &cfg);
	else
		ret = run_peer(fd, &cfg);

	close(fd);
	return ret;
}
