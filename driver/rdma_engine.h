/* SPDX-License-Identifier: GPL-2.0 */
/*
 * rdma_engine.h — RDMA engine API declarations
 * Copyright (c) 2026 Graziano Labs Corp.
 *
 * Public interface for the RDMA subsystem.  Called from main.c
 * (ioctl handlers) and benchmark.c (loopback, ping-pong, streaming).
 *
 * Phase 4 implements the loopback pair (QP-A ↔ QP-B) and local
 * benchmarking.  Peer QP functions (cross-machine RDMA) and
 * RDMA WRITE with immediate are deferred to Phase 8/9.
 *
 * Locking contract:
 *   Callers must hold rdma_sem (read for MR/benchmark ops, write for
 *   setup/teardown).  Functions that access buf_lock or mr_lock acquire
 *   them internally.
 */
#ifndef _RDMA_ENGINE_H
#define _RDMA_ENGINE_H

#include "dmaplane.h"

/*
 * poll_cq_done() — Managed CQ completion callback.
 *
 * Required by ib_alloc_cq's API contract: every wr_cqe.done must point
 * to a valid function.  With IB_POLL_DIRECT we poll via ib_poll_cq()
 * directly, so this callback is rarely invoked — it exists to satisfy
 * the ib_cqe contract and as a safety net.
 */
void poll_cq_done(struct ib_cq *cq, struct ib_wc *wc);

/* RDMA resource lifecycle */
int rdma_engine_setup(struct dmaplane_dev *edev,
		      struct dmaplane_rdma_setup *params);
void rdma_engine_teardown(struct dmaplane_dev *edev);

/* Memory Region registration */
int rdma_engine_register_mr(struct dmaplane_dev *edev,
			    struct dmaplane_mr_params *params);
int rdma_engine_deregister_mr(struct dmaplane_dev *edev, __u32 mr_id);

/* Work request posting — pass poll_cq_wait to get WC via callback */
int rdma_engine_post_send(struct dmaplane_dev *edev,
			  struct ib_qp *qp, struct dmaplane_mr_entry *mr,
			  size_t size, struct poll_cq_wait *wait);
int rdma_engine_post_recv(struct dmaplane_dev *edev,
			  struct ib_qp *qp, struct dmaplane_mr_entry *mr,
			  size_t size, struct poll_cq_wait *wait);

/*
 * rdma_engine_poll_cq() — Poll a CQ with wall-clock timeout.
 *
 * Returns: 1 on success (wc filled), 0 on timeout, negative on error.
 */
int rdma_engine_poll_cq(struct ib_cq *cq, struct ib_wc *wc, int timeout_ms);

/*
 * rdma_engine_flush_cq() — Drain all pending completions from a CQ.
 *
 * Polls until no more completions are available.  Used to clean up stale
 * completions between benchmark runs — prevents poisoning from a previous
 * run's leftover CQEs.
 *
 * Returns: number of completions flushed.
 */
int rdma_engine_flush_cq(struct ib_cq *cq);

/*
 * rdma_engine_cmp_u64() — Comparator for __u64 arrays.
 *
 * For use with kernel sort() to compute latency percentiles (P99).
 */
int rdma_engine_cmp_u64(const void *a, const void *b);

/* ── Phase 8: Cross-machine peer QP management ── */

/*
 * rdma_engine_init_peer() — Create a third QP for cross-machine RDMA.
 *
 * Creates cq_peer + qp_peer sharing the existing PD from setup().
 * Transitions qp_peer to INIT state.  Fills info with local QP number,
 * GID, and MAC for TCP metadata exchange with the remote machine.
 *
 * Caller must hold rdma_sem read lock.
 */
int rdma_engine_init_peer(struct dmaplane_dev *edev,
			  struct dmaplane_rdma_peer_info *info);

/*
 * rdma_engine_connect_peer() — Connect peer QP using remote metadata.
 *
 * Completes INIT → RTR → RTS transitions using the remote machine's
 * QP number, GID, and DMAC (received via TCP).  Sets peer_connected.
 *
 * Caller must hold rdma_sem read lock.
 */
int rdma_engine_connect_peer(struct dmaplane_dev *edev,
			     struct dmaplane_rdma_peer_info *remote);

/*
 * rdma_engine_remote_send() — Post SEND on peer QP, wait for completion.
 *
 * Sends from a local MR (host or GPU-backed) via qp_peer.
 * Caller must hold rdma_sem read lock.
 */
int rdma_engine_remote_send(struct dmaplane_dev *edev,
			    struct dmaplane_rdma_remote_xfer_params *params);

/*
 * rdma_engine_remote_recv() — Post RECV on peer QP, wait for completion.
 *
 * Receives into a local MR via qp_peer.  30s timeout — must be called
 * BEFORE the remote sender posts its SEND (RC QP semantics).
 * Caller must hold rdma_sem read lock.
 */
int rdma_engine_remote_recv(struct dmaplane_dev *edev,
			    struct dmaplane_rdma_remote_xfer_params *params);

/*
 * rdma_engine_teardown_peer() — Destroy peer QP and CQ.
 *
 * Moves qp_peer to ERR state (flushes WRs), destroys QP then CQ
 * in reverse creation order.  Idempotent.
 * Caller must hold rdma_sem write lock.
 */
void rdma_engine_teardown_peer(struct dmaplane_dev *edev);

/* ── Phase 9: RDMA WRITE with Immediate ── */

/*
 * rdma_engine_write_imm() — Post RDMA WRITE WITH IMMEDIATE.
 *
 * Builds an ib_rdma_wr with IB_WR_RDMA_WRITE_WITH_IMM opcode.
 * Writes 'length' bytes from local MR (at local_offset) to remote_addr
 * (rkey), delivering imm_data through the receiver's CQ completion.
 *
 * QP: use_peer_qp=0 → qp_a (loopback), =1 → qp_peer.
 * Polls the corresponding send CQ for completion.
 *
 * Caller must hold rdma_sem read lock.
 */
int rdma_engine_write_imm(struct dmaplane_dev *edev,
			   uint32_t local_mr_id,
			   uint64_t local_offset,
			   uint64_t remote_addr,
			   uint32_t remote_rkey,
			   uint32_t length,
			   uint32_t imm_data,
			   int use_peer_qp,
			   uint64_t *elapsed_ns);

/*
 * rdma_engine_writeimm_post_recv() — Post recv WR for WRITEIMM.
 *
 * Each WRITE_WITH_IMM consumes one recv WR.  This posts a recv on
 * the receiving QP (qp_b for loopback, qp_peer for cross-machine).
 *
 * Caller must hold rdma_sem read lock.
 */
int rdma_engine_writeimm_post_recv(struct dmaplane_dev *edev,
				    uint32_t mr_id,
				    uint32_t size,
				    int use_peer_qp);

/*
 * rdma_engine_writeimm_poll_recv() — Poll recv CQ for WRITEIMM completion.
 *
 * Returns imm_data (host byte order) and byte_len from the completion.
 * Returns 0 on both success (status_out=0) and timeout (status_out=1).
 * Returns -EIO on WC error.
 *
 * Caller must hold rdma_sem read lock.
 */
int rdma_engine_writeimm_poll_recv(struct dmaplane_dev *edev,
				    int use_peer_qp,
				    uint32_t timeout_ms,
				    uint32_t *status_out,
				    uint32_t *wc_flags_out,
				    uint32_t *imm_data_out,
				    uint32_t *byte_len_out,
				    uint64_t *elapsed_ns);

#endif /* _RDMA_ENGINE_H */
