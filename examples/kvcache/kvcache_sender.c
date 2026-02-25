// SPDX-License-Identifier: GPL-2.0
/*
 * Copyright (c) 2026 Graziano Labs Corp.
 *
 * kvcache_sender.c — Chunked WRITEIMM KVCache sender
 *
 * Loopback mode (default): sender owns both source and destination MRs,
 * posts recvs on QP-B, sends WRITEIMMs on QP-A, polls CQ-B. Self-contained
 * test of the full WRITEIMM credit-window pipeline over rxe loopback.
 *
 * Peer mode (--peer <ip>): connects to a remote kvcache_receiver over TCP,
 * exchanges QP/MR metadata, then sends WRITEIMMs via the peer QP (qp_peer).
 * Credit flow uses TCP as a side channel (1 byte = 1 recv credit).
 *
 * GPU mode (--gpu): source MR is GPU-backed via IOCTL_GPU_PIN.
 *
 * Build:  make -C examples/kvcache
 * Run:    sudo ./kvcache_sender [--loopback] [--peer <ip>] [--gpu] [--verify]
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
	int      csv;
	int      layers;
	int      chunks_per_layer;
	uint32_t chunk_size;
	int      credit_window;
	int      iterations;
	int      port;            /* TCP port for peer mode (default: 9876) */
	char     peer_addr[128];  /* peer IP address (peer mode only) */
	char    *ib_dev;
};

static void config_defaults(struct config *c)
{
	memset(c, 0, sizeof(*c));
	c->loopback = 1;
	c->layers = 32;
	c->chunks_per_layer = 4;
	c->chunk_size = 1 * 1024 * 1024;
	c->credit_window = 16;
	c->iterations = 1;
	c->port = 9876;
}

static void usage(const char *prog)
{
	fprintf(stderr,
		"Usage: %s [OPTIONS]\n"
		"  --loopback            Self-contained loopback test (default)\n"
		"  --peer <ip>           Connect to remote kvcache_receiver\n"
		"  --port <port>         TCP port (default: 9876)\n"
		"  --gpu                 Use GPU-backed source MR (requires CUDA)\n"
		"  --ib-dev <name>       IB device name (auto-detect if omitted)\n"
		"  --layers N            Number of simulated layers (default: 32)\n"
		"  --chunks-per-layer N  Chunks per layer (default: 4)\n"
		"  --chunk-size N        Bytes per chunk (default: 1048576)\n"
		"  --credit-window N     Max in-flight WRITEIMMs (default: 16)\n"
		"  --iterations N        Repeat full transfer N times (default: 1)\n"
		"  --verify              Fill source with pattern, verify on recv\n"
		"  --csv                 Single-line CSV output for benchmarking\n",
		prog);
}

static int parse_args(int argc, char *argv[], struct config *c)
{
	static struct option opts[] = {
		{"loopback",         no_argument,       NULL, 'l'},
		{"peer",             required_argument, NULL, 'p'},
		{"port",             required_argument, NULL, 'P'},
		{"gpu",              no_argument,       NULL, 'g'},
		{"ib-dev",           required_argument, NULL, 'd'},
		{"layers",           required_argument, NULL, 'L'},
		{"chunks-per-layer", required_argument, NULL, 'C'},
		{"chunk-size",       required_argument, NULL, 's'},
		{"credit-window",    required_argument, NULL, 'w'},
		{"iterations",       required_argument, NULL, 'i'},
		{"verify",           no_argument,       NULL, 'v'},
		{"csv",              no_argument,       NULL, 'o'},
		{"help",             no_argument,       NULL, 'h'},
		{NULL, 0, NULL, 0},
	};
	int opt;

	while ((opt = getopt_long(argc, argv, "lp:P:gd:L:C:s:w:i:voh",
				  opts, NULL)) != -1) {
		switch (opt) {
		case 'l': c->loopback = 1; break;
		case 'p':
			c->loopback = 0;
			snprintf(c->peer_addr, sizeof(c->peer_addr),
				 "%s", optarg);
			break;
		case 'P': c->port = atoi(optarg); break;
		case 'g': c->use_gpu = 1; break;
		case 'd': c->ib_dev = optarg; break;
		case 'L': c->layers = atoi(optarg); break;
		case 'C': c->chunks_per_layer = atoi(optarg); break;
		case 's': c->chunk_size = (uint32_t)atoi(optarg); break;
		case 'w': c->credit_window = atoi(optarg); break;
		case 'i': c->iterations = atoi(optarg); break;
		case 'v': c->verify = 1; break;
		case 'o': c->csv = 1; break;
		case 'h': usage(argv[0]); return -1;
		default:  usage(argv[0]); return -1;
		}
	}
	return 0;
}

/* ──────────────────────────────────────────────────────────────────────────
 *  Loopback transfer
 * ────────────────────────────────────────────────────────────────────────── */

/*
 * run_loopback — self-contained loopback WRITEIMM transfer.
 *
 * In loopback mode the sender owns *both* source and destination MRs on
 * the same machine.  QP-A (the loopback "send" QP) posts RDMA WRITE WITH
 * IMMEDIATE to QP-B (the loopback "recv" QP).  This mirrors the two-machine
 * case but avoids any TCP/peer setup.
 *
 * Transfer flow:
 *   1. Setup RDMA (creates PD, CQs, loopback QP pair)
 *   2. Create source MR (host or GPU) and destination MR (host, fast-reg)
 *   3. Pre-post credit_window recv WRs on QP-B
 *   4. For each (layer, chunk): write_imm(QP-A) → decrement credits
 *      When credits hit 0, stall: poll recv CQ, replenish one recv, repeat
 *   5. Send KVCACHE_SENTINEL to signal end-of-transfer
 *   6. Drain remaining recv completions
 *   7. Optionally verify data integrity via mmap + memcmp
 */
static int run_loopback(int fd, struct config *cfg)
{
	const char *role = "SENDER";
	uint64_t start_ns = now_ns();
	int total_chunks = cfg->layers * cfg->chunks_per_layer;
	uint64_t total_bytes = (uint64_t)total_chunks * cfg->chunk_size;
	int ret = 0;

	/* Resource tracking — each flag gates cleanup in the 'out:' path */
	uint32_t src_buf_id = 0, dst_buf_id = 0;
	int src_buf_ok = 0, dst_buf_ok = 0;
	int rdma_up = 0;
	int src_mr_ok = 0, dst_mr_ok = 0;
	struct dmaplane_mr_params src_mr = {0}, dst_mr = {0};

#ifdef HAVE_CUDA
	void *gpu_ptr = NULL;
	int gpu_pinned = 0, gpu_mr_ok = 0;
	struct dmaplane_gpu_pin_params gpu_pp = {0};
	struct dmaplane_gpu_mr_params gpu_gm = {0};
#endif

	struct latency_stats ls;
	struct credit_tracker ct;
	struct layer_bitmap bm;

	if (latency_stats_init(&ls, total_chunks + 16) < 0) {
		fprintf(stderr, "  Failed to allocate latency stats\n");
		return 1;
	}
	credit_init(&ct, cfg->credit_window);
	bitmap_init(&bm, cfg->layers, cfg->chunks_per_layer);

	double total_mb = (double)total_bytes / (1024.0 * 1024.0);

	if (!cfg->csv)
		print_trace(role, start_ns,
			    "Config: %d layers x %d chunks x %u KB = %.0f MB, "
			    "credit_window=%d",
			    cfg->layers, cfg->chunks_per_layer,
			    cfg->chunk_size >> 10, total_mb,
			    cfg->credit_window);

	/* [1] Setup RDMA — CQ depth 256, send/recv WR counts sized to
	 *     credit_window + 16 headroom for sentinel + drain operations */
	if (dmaplane_setup_rdma(fd, cfg->ib_dev, 256,
				cfg->credit_window + 16,
				cfg->credit_window + 16) < 0) {
		perror("  SETUP_RDMA");
		ret = 1; goto out;
	}
	rdma_up = 1;

	/* [2] Allocate + register source MR
	 *
	 * GPU path: cudaMalloc → gpu_pin → gpu_register_mr (uses local_dma_lkey).
	 *   The kernel maps the GPU BAR pages into the RDMA subsystem so write_imm
	 *   reads directly from GPU VRAM via PCIe BAR1.
	 *
	 * Host path: create_buffer → register_mr (LOCAL_WRITE only, no rkey needed).
	 *   Source MR only needs local read access since the HCA reads from it.
	 */
	if (cfg->use_gpu) {
#ifdef HAVE_CUDA
		cudaError_t cerr;
		uint64_t gpu_va;

		cerr = cudaMalloc(&gpu_ptr, total_bytes);
		if (cerr != cudaSuccess) {
			fprintf(stderr, "  cudaMalloc: %s\n",
				cudaGetErrorString(cerr));
			ret = 1; goto out;
		}
		gpu_va = (uint64_t)(uintptr_t)gpu_ptr;

		/* NVIDIA P2P API requires 64KB-aligned VA; cudaMalloc usually
		 * returns 256B alignment which is insufficient.  We check here
		 * rather than over-allocate because the C test uses simple
		 * cudaMalloc (the Python side uses allocate_aligned_gpu_tensor). */
		if ((gpu_va & 0xFFFF) != 0) {
			fprintf(stderr, "  cudaMalloc returned non-64KB-aligned"
				" VA 0x%llx\n", (unsigned long long)gpu_va);
			ret = 1; goto out;
		}

		/* Fill source pattern via host staging */
		if (cfg->verify) {
			uint8_t *host_tmp = malloc(cfg->chunk_size);

			if (!host_tmp) {
				ret = 1; goto out;
			}
			for (int l = 0; l < cfg->layers; l++) {
				for (int c = 0; c < cfg->chunks_per_layer; c++) {
					int idx = l * cfg->chunks_per_layer + c;
					uint64_t off = (uint64_t)idx * cfg->chunk_size;

					fill_pattern(host_tmp, l, c,
						     cfg->chunk_size);
					cerr = cudaMemcpy(
						(uint8_t *)gpu_ptr + off,
						host_tmp, cfg->chunk_size,
						cudaMemcpyHostToDevice);
					if (cerr != cudaSuccess) {
						fprintf(stderr,
							"  cudaMemcpy: %s\n",
							cudaGetErrorString(cerr));
						free(host_tmp);
						ret = 1; goto out;
					}
				}
			}
			free(host_tmp);
			cudaDeviceSynchronize();
		}

		if (dmaplane_gpu_pin(fd, gpu_va, total_bytes, &gpu_pp) < 0) {
			perror("  GPU_PIN");
			ret = 1; goto out;
		}
		gpu_pinned = 1;

		if (dmaplane_gpu_register_mr(fd, gpu_pp.handle, &gpu_gm) < 0) {
			perror("  GPU_REGISTER_MR");
			ret = 1; goto out;
		}
		gpu_mr_ok = 1;

		/* Use GPU MR as source */
		src_mr.mr_id = gpu_gm.mr_id;
		src_mr.lkey = gpu_gm.lkey;
		if (!cfg->csv)
			print_trace(role, start_ns,
				    "Source: GPU MR id=%u lkey=0x%x",
				    gpu_gm.mr_id, gpu_gm.lkey);
#else
		fprintf(stderr, "  --gpu requires CUDA support (rebuild with CUDA)\n");
		ret = 1; goto out;
#endif
	} else {
		/* Host source buffer */
		if (dmaplane_create_buffer(fd, total_bytes,
					   DMAPLANE_NUMA_ANY,
					   &src_buf_id) < 0) {
			perror("  CREATE_BUFFER (src)");
			ret = 1; goto out;
		}
		src_buf_ok = 1;

		if (dmaplane_register_mr(fd, src_buf_id,
					 DMAPLANE_IB_ACCESS_LOCAL_WRITE,
					 &src_mr) < 0) {
			perror("  REGISTER_MR (src)");
			ret = 1; goto out;
		}
		src_mr_ok = 1;

		/* Fill source pattern via mmap */
		if (cfg->verify) {
			uint64_t msz;
			uint8_t *src_ptr = dmaplane_mmap_buffer(
				fd, src_buf_id, PROT_READ | PROT_WRITE, &msz);

			if (src_ptr == MAP_FAILED) {
				perror("  mmap (src)");
				ret = 1; goto out;
			}
			for (int l = 0; l < cfg->layers; l++) {
				for (int c = 0; c < cfg->chunks_per_layer; c++) {
					int idx = l * cfg->chunks_per_layer + c;
					uint64_t off = (uint64_t)idx *
						       cfg->chunk_size;

					fill_pattern(src_ptr + off, l, c,
						     cfg->chunk_size);
				}
			}
			munmap(src_ptr, msz);
		}

		if (!cfg->csv)
			print_trace(role, start_ns,
				    "Source: host MR id=%u lkey=0x%x addr=0x%llx",
				    src_mr.mr_id, src_mr.lkey,
				    (unsigned long long)src_mr.addr);
	}

	/* [3] Allocate + register destination MR (host, fast-reg)
	 *
	 * The destination MR needs REMOTE_WRITE access so the HCA can write
	 * into it via RDMA.  This triggers the fast-reg path in the kernel:
	 * ib_alloc_mr() + IB_WR_REG_MR to produce a valid (rkey, addr) pair
	 * that the write_imm ioctl uses as its remote target. */
	if (dmaplane_create_buffer(fd, total_bytes, DMAPLANE_NUMA_ANY,
				   &dst_buf_id) < 0) {
		perror("  CREATE_BUFFER (dst)");
		ret = 1; goto out;
	}
	dst_buf_ok = 1;

	if (dmaplane_register_mr(fd, dst_buf_id,
				 DMAPLANE_IB_ACCESS_LOCAL_WRITE |
				 DMAPLANE_IB_ACCESS_REMOTE_WRITE,
				 &dst_mr) < 0) {
		perror("  REGISTER_MR (dst, fast-reg)");
		ret = 1; goto out;
	}
	dst_mr_ok = 1;

	if (!cfg->csv)
		print_trace(role, start_ns,
			    "Dest:   host MR id=%u rkey=0x%x addr=0x%llx",
			    dst_mr.mr_id, dst_mr.rkey,
			    (unsigned long long)dst_mr.addr);

	/* Run iterations — each iteration is a full transfer of all chunks.
	 * Multiple iterations are useful for benchmarking warmup effects. */
	for (int iter = 0; iter < cfg->iterations; iter++) {
		uint64_t iter_start = now_ns();

		/* Pre-post recvs: each RDMA WRITE WITH IMMEDIATE consumes one
		 * recv WR on the receiving QP.  We pre-post credit_window of
		 * them (capped at total_chunks+1 for small transfers). */
		int initial_recvs = cfg->credit_window;

		if (initial_recvs > total_chunks + 1)
			initial_recvs = total_chunks + 1;
		for (int i = 0; i < initial_recvs; i++) {
			if (dmaplane_post_recv(fd, dst_mr.mr_id,
					       cfg->chunk_size, 0) < 0) {
				perror("  POST_RECV (pre-post)");
				ret = 1; goto out;
			}
		}
		credit_init(&ct, initial_recvs);

		/* Send loop — iterate (layer, chunk) and issue WRITEIMM for each.
		 * The 32-bit immediate encodes (layer, chunk) so the receiver can
		 * track arrivals via CQ completion without memory polling. */
		for (int l = 0; l < cfg->layers; l++) {
			for (int c = 0; c < cfg->chunks_per_layer; c++) {
				int idx = l * cfg->chunks_per_layer + c;
				uint64_t off = (uint64_t)idx * cfg->chunk_size;
				uint32_t imm = KVCACHE_IMM_ENCODE(l, c);
				uint64_t send_start = now_ns();
				uint64_t elapsed;

				/* Stall if no credits — all pre-posted recvs have been
				 * consumed by previous WRITEIMMs.  We must poll at least
				 * one recv completion and post a fresh recv before we
				 * can send again (otherwise the HCA would hit RNR —
				 * Receiver Not Ready — and the send would time out). */
				if (ct.credits <= 0) {
					uint64_t stall_start = now_ns();

					while (ct.credits <= 0) {
						uint64_t cs = now_ns();
						int rc = recv_loop_poll_and_track(
							fd, 0, 10000, &bm,
							&ls, &cs);

						if (rc < 0) {
							fprintf(stderr,
								"  poll failed"
								" during stall\n");
							ret = 1; goto out;
						}
						/* replenish posts a new recv
						 * and increments ct.credits */
						if (recv_loop_replenish(
							fd, dst_mr.mr_id,
							cfg->chunk_size,
							0, &ct) < 0) {
							ret = 1; goto out;
						}
					}
					ct.stall_count++;
					ct.total_stall_ns +=
						now_ns() - stall_start;
				}

				if (dmaplane_write_imm(fd, src_mr.mr_id, off,
						       dst_mr.addr + off,
						       dst_mr.rkey,
						       cfg->chunk_size, imm,
						       0, &elapsed) < 0) {
					perror("  WRITE_IMM");
					ret = 1; goto out;
				}
				ct.credits--;

				if (!cfg->csv)
					print_trace(role, start_ns,
						    "-> WRITEIMM L%d C%d "
						    "(%u KB) imm=0x%08x "
						    "[credits: %d] %.1fms",
						    l, c,
						    cfg->chunk_size >> 10,
						    imm, ct.credits,
						    (double)elapsed / 1e6);

				/* Record send latency */
				latency_stats_add(&ls,
						  now_ns() - send_start);
			}
		}

		/* Post recv for sentinel completion — the sentinel WRITEIMM
		 * also consumes a recv WR, so we must ensure one is available.
		 * After the stall/replenish loop above, credits may be exactly 1,
		 * but that was consumed by the last data chunk's write_imm.
		 * Always post an explicit recv here for the sentinel. */
		if (dmaplane_post_recv(fd, dst_mr.mr_id,
				       cfg->chunk_size, 0) < 0) {
			perror("  POST_RECV (sentinel)");
			ret = 1; goto out;
		}

		/* Send sentinel — imm_data=0xFFFFFFFF tells the receiver that
		 * all data chunks have been sent.  Payload is 4 bytes (minimum). */
		if (dmaplane_write_imm(fd, src_mr.mr_id, 0,
				       dst_mr.addr, dst_mr.rkey,
				       4, KVCACHE_SENTINEL, 0, NULL) < 0) {
			perror("  WRITE_IMM (sentinel)");
			ret = 1; goto out;
		}
		ct.credits--;
		if (!cfg->csv)
			print_trace(role, start_ns, "-> SENTINEL");

		/* Drain remaining completions — some data chunks may still have
		 * pending recv completions if the credit window allowed multiple
		 * in-flight WRITEIMMs.  Poll until the bitmap is fully complete. */
		while (!bitmap_all_complete(&bm)) {
			uint64_t cs = now_ns();
			int rc = recv_loop_poll_and_track(fd, 0, 10000,
							  &bm, &ls, &cs);

			if (rc < 0) {
				fprintf(stderr,
					"  poll failed during drain\n");
				ret = 1; goto out;
			}
			if (rc == 1)
				break; /* SENTINEL received */
		}

		/* Also drain the sentinel completion */
		{
			struct dmaplane_poll_recv_params rp;

			dmaplane_poll_recv(fd, 0, 1000, &rp);
		}

		uint64_t iter_elapsed = now_ns() - iter_start;
		double iter_secs = (double)iter_elapsed / 1e9;
		double throughput = total_mb / iter_secs;

		if (cfg->csv) {
			double avg, p50, p99;

			latency_stats_compute(&ls, &avg, &p50, &p99);
			printf("%.0f,%.1f,%.1f,%.1f,%.1f\n",
			       total_mb, throughput, avg, p50, p99);
		} else {
			print_trace(role, start_ns,
				    "Transfer complete: %.0f MB in %.3fs",
				    total_mb, iter_secs);
			print_trace(role, start_ns,
				    "  Throughput: %.1f MB/s", throughput);
			latency_stats_print(&ls, role, start_ns);
			print_trace(role, start_ns,
				    "  Credits stalled: %d times (avg: %.1fms)",
				    ct.stall_count,
				    ct.stall_count > 0 ?
				    (double)ct.total_stall_ns /
				    ct.stall_count / 1e6 : 0.0);
			print_trace(role, start_ns,
				    "  Chunks: %d/%d received",
				    bitmap_total_received(&bm),
				    total_chunks);
		}

		/* Verify data integrity */
		if (cfg->verify) {
			uint64_t msz;
			uint8_t *dst_ptr = dmaplane_mmap_buffer(
				fd, dst_buf_id, PROT_READ, &msz);

			if (dst_ptr == MAP_FAILED) {
				perror("  mmap (dst, verify)");
				ret = 1; goto out;
			}

			int total_errs = 0;

			for (int l = 0; l < cfg->layers; l++) {
				for (int c = 0; c < cfg->chunks_per_layer;
				     c++) {
					int idx = l * cfg->chunks_per_layer + c;
					uint64_t off = (uint64_t)idx *
						       cfg->chunk_size;
					int errs = verify_pattern(
						dst_ptr + off, l, c,
						cfg->chunk_size);

					total_errs += errs;
				}
			}
			munmap(dst_ptr, msz);

			if (!cfg->csv) {
				if (total_errs == 0)
					print_trace(role, start_ns,
						    "  Data integrity: PASS");
				else
					print_trace(role, start_ns,
						    "  Data integrity: FAIL "
						    "(%d errors)",
						    total_errs);
			}
			if (total_errs > 0)
				ret = 1;
		}
	}

out:
	/* Cleanup order matters:
	 *   1. Drain recv CQ (prevents QP teardown failures from pending WRs)
	 *   2. Teardown RDMA (destroys QPs, CQs, PD)
	 *   3. Deregister MRs
	 *   4. GPU unpin + cudaFree
	 *   5. Destroy buffers
	 * The kernel enforces this: e.g., destroy_buffer with active MR → EBUSY. */
	dmaplane_drain_recv_cq(fd, 0);
	if (rdma_up)
		dmaplane_teardown_rdma(fd);
	if (dst_mr_ok)
		dmaplane_deregister_mr(fd, dst_mr.mr_id);
	if (src_mr_ok)
		dmaplane_deregister_mr(fd, src_mr.mr_id);
#ifdef HAVE_CUDA
	if (gpu_mr_ok)
		dmaplane_deregister_mr(fd, gpu_gm.mr_id);
	if (gpu_pinned)
		dmaplane_gpu_unpin(fd, gpu_pp.handle);
	if (gpu_ptr)
		cudaFree(gpu_ptr);
#endif
	if (dst_buf_ok)
		dmaplane_destroy_buffer(fd, dst_buf_id);
	if (src_buf_ok)
		dmaplane_destroy_buffer(fd, src_buf_id);
	latency_stats_free(&ls);
	return ret;
}

/* ──────────────────────────────────────────────────────────────────────────
 *  Peer mode transfer
 * ────────────────────────────────────────────────────────────────────────── */

/*
 * run_peer — cross-machine WRITEIMM transfer via TCP + peer QP.
 *
 * Connects to a remote kvcache_receiver over TCP, exchanges QP/MR
 * metadata, then sends KVCache chunks via RDMA WRITE WITH IMMEDIATE
 * on the peer QP (qp_peer → cq_peer).
 *
 * Credit flow: the receiver pre-posts credit_window recvs, then sends
 * 1-byte TCP messages as it replenishes.  The sender stalls on TCP read
 * when credits hit 0 — no direct access to the receiver's CQ.
 *
 * Transfer flow:
 *   1. Setup RDMA (creates PD, CQs, loopback QP pair — needed for PD)
 *   2. Create source MR (host or GPU)
 *   3. INIT_PEER — creates qp_peer + cq_peer, returns local QP metadata
 *   4. TCP connect — exchange metadata with receiver
 *   5. CONNECT_PEER — transitions qp_peer using remote's QPN/GID/MAC
 *   6. Wait for "ready" signal (receiver has pre-posted recvs)
 *   7. WRITEIMM loop with TCP credit flow
 *   8. Send sentinel, wait for "done" ack
 */
static int run_peer(int fd, struct config *cfg)
{
	const char *role = "SENDER";
	uint64_t start_ns = now_ns();
	int total_chunks = cfg->layers * cfg->chunks_per_layer;
	uint64_t total_bytes = (uint64_t)total_chunks * cfg->chunk_size;
	int ret = 0;

	/* Resource tracking */
	uint32_t src_buf_id = 0;
	int src_buf_ok = 0, src_mr_ok = 0;
	int rdma_up = 0, peer_init = 0;
	int tcp_fd = -1;
	struct dmaplane_mr_params src_mr = {0};

#ifdef HAVE_CUDA
	void *gpu_ptr = NULL;
	int gpu_pinned = 0, gpu_mr_ok = 0;
	struct dmaplane_gpu_pin_params gpu_pp = {0};
	struct dmaplane_gpu_mr_params gpu_gm = {0};
#endif

	struct latency_stats ls;
	struct credit_tracker ct;

	if (latency_stats_init(&ls, total_chunks + 16) < 0) {
		fprintf(stderr, "  Failed to allocate latency stats\n");
		return 1;
	}

	double total_mb = (double)total_bytes / (1024.0 * 1024.0);

	print_trace(role, start_ns,
		    "Peer mode: %s:%d, %d layers x %d chunks x %u KB = %.0f MB",
		    cfg->peer_addr, cfg->port,
		    cfg->layers, cfg->chunks_per_layer,
		    cfg->chunk_size >> 10, total_mb);

	/* [1] Setup RDMA — needed for PD and GID (peer QP shares the PD) */
	if (dmaplane_setup_rdma(fd, cfg->ib_dev, 256,
				cfg->credit_window + 16,
				cfg->credit_window + 16) < 0) {
		perror("  SETUP_RDMA");
		ret = 1; goto out;
	}
	rdma_up = 1;

	/* [2] Create source MR (host or GPU) — same logic as loopback */
	if (cfg->use_gpu) {
#ifdef HAVE_CUDA
		cudaError_t cerr;
		uint64_t gpu_va;

		cerr = cudaMalloc(&gpu_ptr, total_bytes);
		if (cerr != cudaSuccess) {
			fprintf(stderr, "  cudaMalloc: %s\n",
				cudaGetErrorString(cerr));
			ret = 1; goto out;
		}
		gpu_va = (uint64_t)(uintptr_t)gpu_ptr;

		if ((gpu_va & 0xFFFF) != 0) {
			fprintf(stderr, "  cudaMalloc returned non-64KB-aligned"
				" VA 0x%llx\n", (unsigned long long)gpu_va);
			ret = 1; goto out;
		}

		if (cfg->verify) {
			uint8_t *host_tmp = malloc(cfg->chunk_size);

			if (!host_tmp) {
				ret = 1; goto out;
			}
			for (int l = 0; l < cfg->layers; l++) {
				for (int c = 0; c < cfg->chunks_per_layer; c++) {
					int idx = l * cfg->chunks_per_layer + c;
					uint64_t off = (uint64_t)idx * cfg->chunk_size;

					fill_pattern(host_tmp, l, c,
						     cfg->chunk_size);
					cerr = cudaMemcpy(
						(uint8_t *)gpu_ptr + off,
						host_tmp, cfg->chunk_size,
						cudaMemcpyHostToDevice);
					if (cerr != cudaSuccess) {
						fprintf(stderr,
							"  cudaMemcpy: %s\n",
							cudaGetErrorString(cerr));
						free(host_tmp);
						ret = 1; goto out;
					}
				}
			}
			free(host_tmp);
			cudaDeviceSynchronize();
		}

		if (dmaplane_gpu_pin(fd, gpu_va, total_bytes, &gpu_pp) < 0) {
			perror("  GPU_PIN");
			ret = 1; goto out;
		}
		gpu_pinned = 1;

		if (dmaplane_gpu_register_mr(fd, gpu_pp.handle, &gpu_gm) < 0) {
			perror("  GPU_REGISTER_MR");
			ret = 1; goto out;
		}
		gpu_mr_ok = 1;

		src_mr.mr_id = gpu_gm.mr_id;
		src_mr.lkey = gpu_gm.lkey;
		print_trace(role, start_ns,
			    "Source: GPU MR id=%u lkey=0x%x",
			    gpu_gm.mr_id, gpu_gm.lkey);
#else
		fprintf(stderr, "  --gpu requires CUDA support (rebuild with CUDA)\n");
		ret = 1; goto out;
#endif
	} else {
		if (dmaplane_create_buffer(fd, total_bytes,
					   DMAPLANE_NUMA_ANY,
					   &src_buf_id) < 0) {
			perror("  CREATE_BUFFER (src)");
			ret = 1; goto out;
		}
		src_buf_ok = 1;

		if (dmaplane_register_mr(fd, src_buf_id,
					 DMAPLANE_IB_ACCESS_LOCAL_WRITE,
					 &src_mr) < 0) {
			perror("  REGISTER_MR (src)");
			ret = 1; goto out;
		}
		src_mr_ok = 1;

		if (cfg->verify) {
			uint64_t msz;
			uint8_t *src_ptr = dmaplane_mmap_buffer(
				fd, src_buf_id, PROT_READ | PROT_WRITE, &msz);

			if (src_ptr == MAP_FAILED) {
				perror("  mmap (src)");
				ret = 1; goto out;
			}
			for (int l = 0; l < cfg->layers; l++) {
				for (int c = 0; c < cfg->chunks_per_layer; c++) {
					int idx = l * cfg->chunks_per_layer + c;
					uint64_t off = (uint64_t)idx *
						       cfg->chunk_size;

					fill_pattern(src_ptr + off, l, c,
						     cfg->chunk_size);
				}
			}
			munmap(src_ptr, msz);
		}

		print_trace(role, start_ns,
			    "Source: host MR id=%u lkey=0x%x addr=0x%llx",
			    src_mr.mr_id, src_mr.lkey,
			    (unsigned long long)src_mr.addr);
	}

	/* [3] Initialize peer QP — creates qp_peer + cq_peer */
	{
		struct dmaplane_rdma_peer_info local_info;

		if (dmaplane_init_peer(fd, &local_info) < 0) {
			perror("  INIT_PEER");
			ret = 1; goto out;
		}
		peer_init = 1;
		print_trace(role, start_ns, "INIT_PEER: qp_num=%u",
			    local_info.qp_num);

		/* [4] TCP connect to receiver */
		print_trace(role, start_ns, "TCP connect to %s:%d...",
			    cfg->peer_addr, cfg->port);
		tcp_fd = tcp_connect(cfg->peer_addr, cfg->port);
		if (tcp_fd < 0) {
			perror("  tcp_connect");
			ret = 1; goto out;
		}
		print_trace(role, start_ns, "Connected.");

		/* Send our metadata (QP info + transfer config) */
		struct tcp_metadata send_meta = {0};

		tcp_metadata_from_peer_info(&send_meta, &local_info);
		send_meta.num_layers = cfg->layers;
		send_meta.chunks_per_layer = cfg->chunks_per_layer;
		send_meta.chunk_size = cfg->chunk_size;
		send_meta.credit_window = cfg->credit_window;
		send_meta.buf_size = total_bytes;

		if (tcp_send_metadata(tcp_fd, &send_meta) < 0) {
			fprintf(stderr, "  tcp_send_metadata failed\n");
			ret = 1; goto out;
		}

		/* Receive receiver's metadata (QP info + MR addr/rkey) */
		struct tcp_metadata recv_meta;

		if (tcp_recv_metadata(tcp_fd, &recv_meta) < 0) {
			fprintf(stderr, "  tcp_recv_metadata failed\n");
			ret = 1; goto out;
		}

		print_trace(role, start_ns,
			    "Remote: qpn=%u, mr_addr=0x%llx, mr_rkey=0x%x",
			    recv_meta.qpn,
			    (unsigned long long)recv_meta.mr_addr,
			    recv_meta.mr_rkey);

		/* [5] Connect peer QP using remote's metadata */
		struct dmaplane_rdma_peer_info remote_info;

		tcp_metadata_to_peer_info(&recv_meta, &remote_info);
		if (dmaplane_connect_peer(fd, &remote_info) < 0) {
			perror("  CONNECT_PEER");
			ret = 1; goto out;
		}
		print_trace(role, start_ns, "CONNECT_PEER: connected.");

		/* [6] Wait for receiver's "ready" signal */
		{
			uint8_t ready;

			if (tcp_recv_all(tcp_fd, &ready, 1) < 0) {
				fprintf(stderr, "  Failed to receive ready signal\n");
				ret = 1; goto out;
			}
			print_trace(role, start_ns, "Receiver ready.");
		}

		/* [7] WRITEIMM loop with TCP credit flow */
		uint64_t remote_addr = recv_meta.mr_addr;
		uint32_t remote_rkey = recv_meta.mr_rkey;

		for (int iter = 0; iter < cfg->iterations; iter++) {
			uint64_t iter_start = now_ns();
			struct layer_bitmap bm;

			bitmap_init(&bm, cfg->layers, cfg->chunks_per_layer);
			credit_init(&ct, cfg->credit_window);

			for (int l = 0; l < cfg->layers; l++) {
				for (int c = 0; c < cfg->chunks_per_layer; c++) {
					int idx = l * cfg->chunks_per_layer + c;
					uint64_t off = (uint64_t)idx * cfg->chunk_size;
					uint32_t imm = KVCACHE_IMM_ENCODE(l, c);
					uint64_t send_start = now_ns();
					uint64_t elapsed;
					int got;

					/* Stall if no credits — block on TCP
					 * until receiver sends credit bytes */
					if (ct.credits <= 0) {
						uint64_t stall_start = now_ns();

						got = tcp_recv_credits_blocking(tcp_fd);
						if (got < 0) {
							fprintf(stderr,
								"  TCP credit recv failed\n");
							ret = 1; goto out;
						}
						ct.credits += got;
						ct.stall_count++;
						ct.total_stall_ns +=
							now_ns() - stall_start;
					}

					/* Opportunistic non-blocking drain */
					got = tcp_recv_credits_nonblock(tcp_fd);
					if (got > 0)
						ct.credits += got;

					if (dmaplane_write_imm(fd, src_mr.mr_id, off,
							       remote_addr + off,
							       remote_rkey,
							       cfg->chunk_size, imm,
							       1, &elapsed) < 0) {
						perror("  WRITE_IMM");
						ret = 1; goto out;
					}
					ct.credits--;

					if (!cfg->csv)
						print_trace(role, start_ns,
							    "-> WRITEIMM L%d C%d "
							    "(%u KB) imm=0x%08x "
							    "[credits: %d] %.1fms",
							    l, c,
							    cfg->chunk_size >> 10,
							    imm, ct.credits,
							    (double)elapsed / 1e6);

					latency_stats_add(&ls,
							  now_ns() - send_start);
				}
			}

			/* Send sentinel */
			if (dmaplane_write_imm(fd, src_mr.mr_id, 0,
					       remote_addr, remote_rkey,
					       4, KVCACHE_SENTINEL,
					       1, NULL) < 0) {
				perror("  WRITE_IMM (sentinel)");
				ret = 1; goto out;
			}
			if (!cfg->csv)
				print_trace(role, start_ns, "-> SENTINEL");

			/* Wait for receiver's "done" ack */
			{
				uint8_t done;

				if (tcp_recv_all(tcp_fd, &done, 1) < 0) {
					fprintf(stderr,
						"  Failed to receive done ack\n");
					ret = 1; goto out;
				}
			}

			uint64_t iter_elapsed = now_ns() - iter_start;
			double iter_secs = (double)iter_elapsed / 1e9;
			double throughput = total_mb / iter_secs;

			if (cfg->csv) {
				double avg, p50, p99;

				latency_stats_compute(&ls, &avg, &p50, &p99);
				printf("%.0f,%.1f,%.1f,%.1f,%.1f\n",
				       total_mb, throughput, avg, p50, p99);
			} else {
				print_trace(role, start_ns,
					    "Transfer complete: %.0f MB in %.3fs",
					    total_mb, iter_secs);
				print_trace(role, start_ns,
					    "  Throughput: %.1f MB/s", throughput);
				latency_stats_print(&ls, role, start_ns);
				print_trace(role, start_ns,
					    "  Credits stalled: %d times (avg: %.1fms)",
					    ct.stall_count,
					    ct.stall_count > 0 ?
					    (double)ct.total_stall_ns /
					    ct.stall_count / 1e6 : 0.0);
				print_trace(role, start_ns,
					    "  Chunks: %d sent",
					    total_chunks);
			}
		}
	}

out:
	/* Cleanup order:
	 *   1. Close TCP (stops credit flow)
	 *   2. Destroy peer QP (drains WRs on qp_peer)
	 *   3. Teardown RDMA (destroys loopback QPs, CQs, PD)
	 *   4. Deregister MRs
	 *   5. GPU unpin + cudaFree
	 *   6. Destroy buffers */
	if (tcp_fd >= 0)
		close(tcp_fd);
	if (peer_init)
		dmaplane_destroy_peer(fd);
	if (rdma_up)
		dmaplane_teardown_rdma(fd);
	if (src_mr_ok)
		dmaplane_deregister_mr(fd, src_mr.mr_id);
#ifdef HAVE_CUDA
	if (gpu_mr_ok)
		dmaplane_deregister_mr(fd, gpu_gm.mr_id);
	if (gpu_pinned)
		dmaplane_gpu_unpin(fd, gpu_pp.handle);
	if (gpu_ptr)
		cudaFree(gpu_ptr);
#endif
	if (src_buf_ok)
		dmaplane_destroy_buffer(fd, src_buf_id);
	latency_stats_free(&ls);
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
