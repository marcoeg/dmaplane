// SPDX-License-Identifier: GPL-2.0
/*
 * rdma_engine.c — RDMA resource management
 * Copyright (c) 2026 Graziano Labs Corp.
 *
 * Implements dmaplane's kernel-level client of the InfiniBand Verbs API.
 * Communicates with ib_core for the verbs framework and rdma_rxe
 * (Soft-RoCE) as the transport provider.
 *
 * Why the kernel module owns its own PD, CQ, QP, and MR:
 *
 *   The module owns the physical pages (alloc_pages in dmabuf_rdma.c)
 *   and must establish IOMMU/DMA mappings from kernel space via
 *   ib_dma_map_sg().  The resulting local_dma_lkey authorizes the NIC to
 *   DMA directly to/from those pages — a kernel-only operation that
 *   userspace ibv_reg_mr cannot replace.
 *
 *   The loopback QP pair (QP-A ↔ QP-B) lets the module benchmark the
 *   full DMA data path — buffer → MR → post_send → wire → post_recv →
 *   completion — without a remote peer or userspace involvement.
 *
 * Handles:
 * - IB device discovery via ib_device_get_by_name()
 * - PD, CQ, QP creation and teardown via ib_alloc_pd/cq, ib_create_qp
 * - QP state machine transitions (RESET → INIT → RTR → RTS)
 * - Memory Region registration from page arrays via ib_dma_map_sg
 * - Send/Recv work request posting via ib_post_send/recv
 * - Completion queue polling via ib_poll_cq (IB_POLL_DIRECT)
 */

#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/delay.h>
#include <linux/slab.h>
#include <linux/scatterlist.h>
#include <linux/vmalloc.h>
#include <net/ipv6.h>
#include <rdma/ib_verbs.h>
#include <rdma/ib_cache.h>

#include "dmaplane.h"
#include "rdma_engine.h"
#include "dmabuf_rdma.h"
#include "dmaplane_trace.h"

/* ====================================================================
 * Internal Helpers
 * ==================================================================== */

static struct ib_device *find_ib_device(const char *name)
{
	struct ib_device *dev;

	dev = ib_device_get_by_name(name, RDMA_DRIVER_UNKNOWN);
	if (!dev)
		pr_err("IB device '%s' not found\n", name);
	return dev;
}

/*
 * scan_gid_table — find the best RoCEv2 GID for rxe.
 *
 * Soft-RoCE exposes multiple GID entries, all typed ROCE_UDP_ENCAP.
 * GID[0] is typically the EUI-64 MAC-derived link-local — a "ghost"
 * address that doesn't exist on the interface when Ubuntu's
 * stable-privacy addressing is active.  Using it causes QP transitions
 * to succeed but traffic to fail silently.
 *
 * Strategy: scan up to 16 entries and pick the last link-local
 * ROCE_UDP_ENCAP GID (empirically the real interface address).
 * Falls back to index 0 with a warning if nothing better is found.
 *
 * Returns 0 on success (ctx->gid_index and ctx->gid populated),
 * negative errno on failure.
 */
static int scan_gid_table(struct dmaplane_rdma_ctx *ctx)
{
	const struct ib_gid_attr *attr;
	int idx, ret;
	bool found = false;
	int last_linklocal = -1;
	int first_v4mapped = -1;

	for (idx = 0; idx < 16; idx++) {
		attr = rdma_get_gid_attr(ctx->ib_dev, ctx->port, idx);
		if (IS_ERR(attr))
			break;

		pr_debug("GID[%d] type=%d gid=%pI6c ndev=%s\n",
			 idx, attr->gid_type, attr->gid.raw,
			 attr->ndev ? attr->ndev->name : "(null)");

		/* Track link-local entries; prefer the last one
		 * (empirically the real interface address on Ubuntu
		 * with stable-privacy addressing) */
		if (attr->gid_type == IB_GID_TYPE_ROCE_UDP_ENCAP &&
		    !ipv6_addr_any((struct in6_addr *)attr->gid.raw)) {
			int atype = ipv6_addr_type((struct in6_addr *)attr->gid.raw);
			if ((atype & IPV6_ADDR_MAPPED) && first_v4mapped < 0)
				first_v4mapped = idx;
			if (atype & IPV6_ADDR_LINKLOCAL)
				last_linklocal = idx;
		}

		rdma_put_gid_attr(attr);
	}

	/* Prefer IPv4-mapped GID[1] (routable on EC2), fall back to link-local */
	int preferred = (first_v4mapped >= 0) ? first_v4mapped : last_linklocal;
	if (preferred >= 0) {
		attr = rdma_get_gid_attr(ctx->ib_dev, ctx->port,
					 preferred);
		if (!IS_ERR(attr)) {
			ctx->gid = attr->gid;
			ctx->gid_index = preferred;
			found = true;
			rdma_put_gid_attr(attr);
		}
	}

	if (!found) {
		pr_warn("no suitable link-local GID found, falling back to index 0\n");
		ret = rdma_query_gid(ctx->ib_dev, ctx->port, 0, &ctx->gid);
		if (ret) {
			pr_err("rdma_query_gid failed: %d\n", ret);
			return ret;
		}
		ctx->gid_index = 0;
	}

	pr_debug("using GID[%d] = %pI6c\n", ctx->gid_index, ctx->gid.raw);
	return 0;
}

/*
 * QP state machine transitions: RESET → INIT → RTR → RTS
 *
 * Each transition is driven by ib_modify_qp() with a specific set of
 * required attributes.  Each state expects exactly the right attribute
 * mask — the wrong combination silently fails or causes later WR
 * posting to return opaque errors.
 */
static int qp_to_init(struct ib_qp *qp, __u8 port)
{
	struct ib_qp_attr attr = {};

	attr.qp_state = IB_QPS_INIT;
	attr.pkey_index = 0;
	attr.port_num = port;
	attr.qp_access_flags = IB_ACCESS_LOCAL_WRITE |
			       IB_ACCESS_REMOTE_WRITE |
			       IB_ACCESS_REMOTE_READ;

	return ib_modify_qp(qp, &attr,
			    IB_QP_STATE | IB_QP_PKEY_INDEX |
			    IB_QP_PORT | IB_QP_ACCESS_FLAGS);
}

static int qp_to_rtr(struct ib_qp *qp, __u8 port, __u32 dest_qpn,
		      __u16 lid, union ib_gid *gid, int gid_index,
		      const struct ib_gid_attr *sgid_attr,
		      const u8 *dmac)
{
	struct ib_qp_attr attr = {};

	attr.qp_state = IB_QPS_RTR;
	attr.path_mtu = IB_MTU_1024;	/* safe default for rxe over any Ethernet */
	attr.dest_qp_num = dest_qpn;
	attr.rq_psn = 0;
	attr.max_dest_rd_atomic = 1;
	attr.min_rnr_timer = 12;	/* ~0.01ms receiver-not-ready retry delay */

	attr.ah_attr.type = RDMA_AH_ATTR_TYPE_ROCE;
	rdma_ah_set_sl(&attr.ah_attr, 0);
	rdma_ah_set_port_num(&attr.ah_attr, port);
	rdma_ah_set_static_rate(&attr.ah_attr, 0);

	rdma_ah_set_grh(&attr.ah_attr, gid, 0, gid_index, 1, 0);
	attr.ah_attr.grh.sgid_attr = sgid_attr;

	/* RoCEv2 requires destination MAC in the address handle */
	if (dmac)
		memcpy(attr.ah_attr.roce.dmac, dmac, ETH_ALEN);

	return ib_modify_qp(qp, &attr,
			    IB_QP_STATE | IB_QP_AV | IB_QP_PATH_MTU |
			    IB_QP_DEST_QPN | IB_QP_RQ_PSN |
			    IB_QP_MAX_DEST_RD_ATOMIC | IB_QP_MIN_RNR_TIMER);
}

static int qp_to_rts(struct ib_qp *qp)
{
	struct ib_qp_attr attr = {};

	attr.qp_state = IB_QPS_RTS;
	attr.timeout = 14;		/* ~4.096s local ACK timeout (4.096us * 2^14) */
	attr.retry_cnt = 7;		/* max retries on timeout (IB spec max) */
	attr.rnr_retry = 7;		/* max retries on RNR NAK (7 = infinite) */
	attr.sq_psn = 0;
	attr.max_rd_atomic = 1;

	return ib_modify_qp(qp, &attr,
			    IB_QP_STATE | IB_QP_TIMEOUT | IB_QP_RETRY_CNT |
			    IB_QP_RNR_RETRY | IB_QP_SQ_PSN |
			    IB_QP_MAX_QP_RD_ATOMIC);
}

/*
 * connect_loopback_qps — Connect QP-A ↔ QP-B for self-test.
 *
 * With raw verbs (no rdma_cm), we must populate the Address Handle's
 * destination MAC ourselves.  For loopback it's our own interface MAC,
 * read via rdma_read_gid_attr_ndev_rcu inside RCU read section.
 * The ndev reference is only valid inside the RCU read section —
 * copy the MAC into a local buffer before releasing.
 */
static int connect_loopback_qps(struct dmaplane_rdma_ctx *ctx)
{
	const struct ib_gid_attr *sgid_attr;
	struct net_device *ndev;
	u8 dmac_buf[ETH_ALEN];
	bool have_dmac = false;
	int ret;

	sgid_attr = rdma_get_gid_attr(ctx->ib_dev, ctx->port, ctx->gid_index);
	if (IS_ERR(sgid_attr)) {
		pr_err("rdma_get_gid_attr(%d) failed: %ld\n",
		       ctx->gid_index, PTR_ERR(sgid_attr));
		return PTR_ERR(sgid_attr);
	}

	/* Extract our own interface MAC for loopback DMAC */
	rcu_read_lock();
	ndev = rdma_read_gid_attr_ndev_rcu(sgid_attr);
	if (!IS_ERR(ndev)) {
		memcpy(dmac_buf, ndev->dev_addr, ETH_ALEN);
		have_dmac = true;
	}
	rcu_read_unlock();

	if (!have_dmac) {
		pr_err("cannot determine DMAC for loopback\n");
		ret = -ENODEV;
		goto out;
	}

	pr_debug("loopback DMAC = %pM\n", dmac_buf);

	/* QP-A: RESET → INIT → RTR → RTS */
	ret = qp_to_init(ctx->qp_a, ctx->port);
	if (ret) {
		pr_err("qp_a to INIT failed: %d\n", ret);
		goto out;
	}

	ret = qp_to_rtr(ctx->qp_a, ctx->port, ctx->qp_b->qp_num,
			ctx->lid, &ctx->gid, ctx->gid_index, sgid_attr,
			dmac_buf);
	if (ret) {
		pr_err("qp_a to RTR failed: %d\n", ret);
		goto out;
	}

	ret = qp_to_rts(ctx->qp_a);
	if (ret) {
		pr_err("qp_a to RTS failed: %d\n", ret);
		goto out;
	}

	/* QP-B: RESET → INIT → RTR → RTS */
	ret = qp_to_init(ctx->qp_b, ctx->port);
	if (ret) {
		pr_err("qp_b to INIT failed: %d\n", ret);
		goto out;
	}

	ret = qp_to_rtr(ctx->qp_b, ctx->port, ctx->qp_a->qp_num,
			ctx->lid, &ctx->gid, ctx->gid_index, sgid_attr,
			dmac_buf);
	if (ret) {
		pr_err("qp_b to RTR failed: %d\n", ret);
		goto out;
	}

	ret = qp_to_rts(ctx->qp_b);
	if (ret) {
		pr_err("qp_b to RTS failed: %d\n", ret);
		goto out;
	}

	pr_debug("loopback QPs connected (qp_a=%u <-> qp_b=%u)\n",
		 ctx->qp_a->qp_num, ctx->qp_b->qp_num);

out:
	rdma_put_gid_attr(sgid_attr);
	return ret;
}

/*
 * register_mr_build_sgt — Build SG table from buffer pages and DMA-map it.
 *
 * Allocates an sg_table with one entry per page (order-0 pages are not
 * physically contiguous), then calls ib_dma_map_sg() to establish IOMMU
 * mappings for the full buffer.
 *
 * On success, *sgt_out and *mapped_out are set.  Caller owns the sgt and
 * must free it (ib_dma_unmap_sg + sg_free_table + kfree) on error.
 */
static int register_mr_build_sgt(struct ib_device *ibdev,
				 struct page **pages, unsigned int nr_pages,
				 struct sg_table **sgt_out, int *mapped_out)
{
	struct sg_table *sgt;
	struct scatterlist *sg;
	int mapped_nents;
	unsigned int i;

	sgt = kzalloc(sizeof(*sgt), GFP_KERNEL);
	if (!sgt)
		return -ENOMEM;

	if (sg_alloc_table(sgt, nr_pages, GFP_KERNEL)) {
		kfree(sgt);
		return -ENOMEM;
	}

	for_each_sg(sgt->sgl, sg, nr_pages, i)
		sg_set_page(sg, pages[i], PAGE_SIZE, 0);

	mapped_nents = ib_dma_map_sg(ibdev, sgt->sgl, sgt->nents,
				     DMA_BIDIRECTIONAL);
	if (!mapped_nents) {
		pr_err("ib_dma_map_sg failed for %u pages\n", nr_pages);
		sg_free_table(sgt);
		kfree(sgt);
		return -EIO;
	}

	*sgt_out = sgt;
	*mapped_out = mapped_nents;
	return 0;
}

/*
 * register_mr_fastreg — Allocate fast-reg MR, map SG, post IB_WR_REG_MR.
 *
 * local_dma_lkey is a per-PD shortcut that authorizes the local NIC to
 * DMA local pages (rkey=0, no remote access).  Remote operations
 * (REMOTE_WRITE, REMOTE_READ) need a real rkey, which only a fast-reg
 * MR provides.  This function allocates via ib_alloc_mr, maps the SG
 * list, and posts IB_WR_REG_MR on QP-A to activate the MR.
 *
 * On success, *frmr_out is set.  On failure, any partially allocated MR
 * is cleaned up internally.
 */
static int register_mr_fastreg(struct dmaplane_rdma_ctx *ctx,
			       struct sg_table *sgt, unsigned int nr_pages,
			       u32 access_flags, struct ib_mr **frmr_out)
{
	struct ib_reg_wr reg_wr = {};
	const struct ib_send_wr *bad_wr;
	struct poll_cq_wait reg_wait;
	struct ib_wc reg_wc;
	struct ib_mr *frmr;
	int nr_mapped, rc, ret;

	frmr = ib_alloc_mr(ctx->pd, IB_MR_TYPE_MEM_REG, nr_pages);
	if (IS_ERR(frmr)) {
		pr_err("ib_alloc_mr failed: %ld\n", PTR_ERR(frmr));
		return PTR_ERR(frmr);
	}

	nr_mapped = ib_map_mr_sg(frmr, sgt->sgl, sgt->nents, NULL, PAGE_SIZE);
	if (nr_mapped < 0 || nr_mapped < (int)sgt->nents) {
		pr_err("ib_map_mr_sg mapped %d/%u entries\n",
		       nr_mapped, sgt->nents);
		ret = nr_mapped < 0 ? nr_mapped : -EIO;
		goto err_dereg;
	}

	/* Post IB_WR_REG_MR on QP-A to activate the MR */
	init_completion(&reg_wait.done);
	reg_wait.cqe.done = poll_cq_done;

	reg_wr.wr.wr_cqe    = &reg_wait.cqe;
	reg_wr.wr.opcode     = IB_WR_REG_MR;
	reg_wr.wr.send_flags = IB_SEND_SIGNALED;
	reg_wr.mr            = frmr;
	reg_wr.key           = frmr->rkey;
	reg_wr.access        = access_flags;

	ret = ib_post_send(ctx->qp_a, &reg_wr.wr, &bad_wr);
	if (ret) {
		pr_err("IB_WR_REG_MR post failed: %d\n", ret);
		goto err_dereg;
	}

	/* Poll CQ-A for REG_MR completion (10s timeout) */
	rc = rdma_engine_poll_cq(ctx->cq_a, &reg_wc, 10000);
	if (rc <= 0 || reg_wc.status != IB_WC_SUCCESS) {
		pr_err("REG_MR completion %s (status=%d)\n",
		       rc <= 0 ? "failed" : "error",
		       rc > 0 ? reg_wc.status : rc);
		ret = rc <= 0 ? (rc == 0 ? -ETIMEDOUT : rc) : -EIO;
		goto err_dereg;
	}

	pr_debug("fast-reg MR OK lkey=0x%x rkey=0x%x iova=0x%llx pages=%u\n",
		 frmr->lkey, frmr->rkey, frmr->iova, nr_pages);

	*frmr_out = frmr;
	return 0;

err_dereg:
	ib_dereg_mr(frmr);
	return ret;
}

/* ====================================================================
 * Public API
 * ==================================================================== */

/*
 * Managed CQ completion callback — required by ib_alloc_cq's API
 * contract: every wr_cqe.done must point to a valid function.
 * With IB_POLL_DIRECT we call ib_poll_cq() directly, so this
 * callback is not invoked under normal operation.  It exists to
 * satisfy the ib_cqe contract and as a safety net if a managed-CQ
 * code path is ever triggered.
 */
void poll_cq_done(struct ib_cq *cq, struct ib_wc *wc)
{
	struct poll_cq_wait *wait =
		container_of(wc->wr_cqe, struct poll_cq_wait, cqe);
	wait->wc = *wc;
	complete(&wait->done);
}
EXPORT_SYMBOL_GPL(poll_cq_done);

/*
 * rdma_engine_poll_cq — Poll a CQ with wall-clock timeout.
 *
 * CQs are created with IB_POLL_DIRECT.  We bypass the managed CQ layer
 * and poll directly with ib_poll_cq().  No softirq/workqueue is involved,
 * no callbacks fire, no races with module unload.
 *
 * Returns: 1 on success (wc filled), 0 on timeout, negative on error.
 */
int rdma_engine_poll_cq(struct ib_cq *cq, struct ib_wc *wc, int timeout_ms)
{
	ktime_t deadline = ktime_add_ms(ktime_get(), timeout_ms);
	int ret;

	do {
		ret = ib_poll_cq(cq, 1, wc);
		if (ret > 0) {
			trace_dmaplane_rdma_completion(wc->status,
						       wc->byte_len, 0);
			return 1;
		}
		if (ret < 0)
			return ret;
		cond_resched();
		usleep_range(50, 200);
	} while (ktime_before(ktime_get(), deadline));

	return 0; /* timeout */
}
EXPORT_SYMBOL_GPL(rdma_engine_poll_cq);

/*
 * rdma_engine_setup — Orchestrate the full RDMA resource lifecycle.
 *
 * Creates the resource hierarchy required for RDMA communication:
 *
 *   IB device → PD → CQ-A, CQ-B → QP-A, QP-B → loopback connection
 *
 * Each layer depends on the previous: PD scopes all resources, CQs
 * collect completions for their bound QPs, and QPs carry the actual
 * traffic.  Teardown (rdma_engine_teardown) reverses this order.
 *
 * The loopback pair serves two purposes:
 *   1. Self-contained benchmarking without a remote peer.
 *   2. DMA mapping infrastructure — the PD and local_dma_lkey are used
 *      by rdma_engine_register_mr() to DMA-map buffer pages.
 *
 * Caller must hold rdma_sem write lock.
 */
int rdma_engine_setup(struct dmaplane_dev *edev,
		      struct dmaplane_rdma_setup *params)
{
	struct dmaplane_rdma_ctx *ctx = &edev->rdma;
	struct ib_qp_init_attr qp_init;
	struct ib_port_attr port_attr;
	int cq_depth;
	int ret;

	if (ctx->initialized) {
		pr_warn("RDMA already initialized\n");
		return -EBUSY;
	}

	/* Find the IB device */
	ctx->ib_dev = find_ib_device(params->ib_dev_name);
	if (!ctx->ib_dev)
		return -ENODEV;

	ctx->port = params->port ? params->port : 1;

	/* Query port for LID */
	ret = ib_query_port(ctx->ib_dev, ctx->port, &port_attr);
	if (ret) {
		pr_err("ib_query_port failed: %d\n", ret);
		goto err_put_dev;
	}
	ctx->lid = port_attr.lid;

	/* Scan the GID table for a usable RoCEv2 address */
	ret = scan_gid_table(ctx);
	if (ret)
		goto err_put_dev;

	/* Allocate Protection Domain — the root of the RDMA resource
	 * hierarchy.  All CQs, QPs, and MRs created under this PD share
	 * access permissions and are isolated from resources in other PDs. */
	ctx->pd = ib_alloc_pd(ctx->ib_dev, 0);
	if (IS_ERR(ctx->pd)) {
		ret = PTR_ERR(ctx->pd);
		pr_err("ib_alloc_pd failed: %d\n", ret);
		ctx->pd = NULL;
		goto err_put_dev;
	}

	/* Create Completion Queues.
	 *
	 * IB_POLL_DIRECT is a deliberate architectural choice: we poll
	 * completions explicitly via ib_poll_cq() rather than using
	 * IB_POLL_SOFTIRQ or IB_POLL_WORKQUEUE.  This gives deterministic,
	 * race-free processing — no softirq/workqueue callbacks, no
	 * asynchronous state changes during teardown, and no races with
	 * module unload.  The trade-off is that callers must busy-poll. */
	cq_depth = params->cq_depth ? params->cq_depth : 128;

	ctx->cq_a = ib_alloc_cq(ctx->ib_dev, NULL, cq_depth,
				0, IB_POLL_DIRECT);
	if (IS_ERR(ctx->cq_a)) {
		ret = PTR_ERR(ctx->cq_a);
		pr_err("ib_alloc_cq (a) failed: %d\n", ret);
		ctx->cq_a = NULL;
		goto err_dealloc_pd;
	}

	ctx->cq_b = ib_alloc_cq(ctx->ib_dev, NULL, cq_depth,
				0, IB_POLL_DIRECT);
	if (IS_ERR(ctx->cq_b)) {
		ret = PTR_ERR(ctx->cq_b);
		pr_err("ib_alloc_cq (b) failed: %d\n", ret);
		ctx->cq_b = NULL;
		goto err_destroy_cq_a;
	}

	/* Create Reliable Connected (RC) Queue Pairs — the same transport
	 * type used by NCCL for GPU-to-GPU gradient aggregation.  RC
	 * guarantees in-order, reliable delivery.  IB_SIGNAL_ALL_WR means
	 * every WR generates a CQE, simplifying completion processing. */
	memset(&qp_init, 0, sizeof(qp_init));
	qp_init.send_cq = ctx->cq_a;
	qp_init.recv_cq = ctx->cq_a;
	qp_init.qp_type = IB_QPT_RC;
	qp_init.sq_sig_type = IB_SIGNAL_ALL_WR;
	qp_init.cap.max_send_wr = params->max_send_wr ? params->max_send_wr : 64;
	qp_init.cap.max_recv_wr = params->max_recv_wr ? params->max_recv_wr : 64;
	qp_init.cap.max_send_sge = 1;
	qp_init.cap.max_recv_sge = 1;

	ctx->qp_a = ib_create_qp(ctx->pd, &qp_init);
	if (IS_ERR(ctx->qp_a)) {
		ret = PTR_ERR(ctx->qp_a);
		pr_err("ib_create_qp (a) failed: %d\n", ret);
		ctx->qp_a = NULL;
		goto err_destroy_cq_b;
	}

	/* QP-B uses cq_b for both send and recv completions */
	qp_init.send_cq = ctx->cq_b;
	qp_init.recv_cq = ctx->cq_b;

	ctx->qp_b = ib_create_qp(ctx->pd, &qp_init);
	if (IS_ERR(ctx->qp_b)) {
		ret = PTR_ERR(ctx->qp_b);
		pr_err("ib_create_qp (b) failed: %d\n", ret);
		ctx->qp_b = NULL;
		goto err_destroy_qp_a;
	}

	/* Connect the loopback pair (RESET → INIT → RTR → RTS) */
	ret = connect_loopback_qps(ctx);
	if (ret)
		goto err_destroy_qp_b;

	ctx->initialized = true;
	params->status = 0;

	pr_info("RDMA setup complete on %s port %u (PD, 2xCQ, 2xQP)\n",
		params->ib_dev_name, ctx->port);
	return 0;

err_destroy_qp_b:
	ib_destroy_qp(ctx->qp_b);
	ctx->qp_b = NULL;
err_destroy_qp_a:
	ib_destroy_qp(ctx->qp_a);
	ctx->qp_a = NULL;
err_destroy_cq_b:
	ib_free_cq(ctx->cq_b);
	ctx->cq_b = NULL;
err_destroy_cq_a:
	ib_free_cq(ctx->cq_a);
	ctx->cq_a = NULL;
err_dealloc_pd:
	ib_dealloc_pd(ctx->pd);
	ctx->pd = NULL;
err_put_dev:
	ib_device_put(ctx->ib_dev);
	ctx->ib_dev = NULL;
	return ret;
}
EXPORT_SYMBOL_GPL(rdma_engine_setup);

/*
 * rdma_engine_teardown — Disciplined RDMA resource teardown.
 *
 * Follows a strict protocol:
 *
 *   1. Transition QPs to IB_QPS_ERR — the RDMA subsystem flushes all
 *      outstanding work requests.  Without this, destroying a QP with
 *      outstanding WRs leaves the provider holding references to our
 *      CQE/SGE memory.
 *
 *   2. Destroy resources in strict reverse order of creation:
 *        QP-B → QP-A → CQ-B → CQ-A → PD → IB device ref
 *
 * Because CQs use IB_POLL_DIRECT, there are no background workqueues
 * or softirq handlers — teardown is race-free without explicit CQ
 * draining.
 *
 * Caller must hold rdma_sem write lock.
 */
void rdma_engine_teardown(struct dmaplane_dev *edev)
{
	struct dmaplane_rdma_ctx *ctx = &edev->rdma;
	struct ib_qp_attr attr = { .qp_state = IB_QPS_ERR };

	if (!ctx->initialized)
		return;

	/* Step 0: Tear down peer QP if still active — safety net in case
	 * userspace calls TEARDOWN_RDMA without explicit DESTROY_PEER. */
	rdma_engine_teardown_peer(edev);

	/* Step 1: Move QPs to error state — flushes outstanding WRs */
	if (ctx->qp_b)
		ib_modify_qp(ctx->qp_b, &attr, IB_QP_STATE);
	if (ctx->qp_a)
		ib_modify_qp(ctx->qp_a, &attr, IB_QP_STATE);

	/* Step 2: Destroy in strict reverse order of creation */
	if (ctx->qp_b) {
		ib_destroy_qp(ctx->qp_b);
		ctx->qp_b = NULL;
	}
	if (ctx->qp_a) {
		ib_destroy_qp(ctx->qp_a);
		ctx->qp_a = NULL;
	}
	if (ctx->cq_b) {
		ib_free_cq(ctx->cq_b);
		ctx->cq_b = NULL;
	}
	if (ctx->cq_a) {
		ib_free_cq(ctx->cq_a);
		ctx->cq_a = NULL;
	}
	if (ctx->pd) {
		ib_dealloc_pd(ctx->pd);
		ctx->pd = NULL;
	}
	if (ctx->ib_dev) {
		ib_device_put(ctx->ib_dev);
		ctx->ib_dev = NULL;
	}

	ctx->initialized = false;
	pr_info("RDMA teardown complete\n");
}
EXPORT_SYMBOL_GPL(rdma_engine_teardown);

/*
 * rdma_engine_register_mr — Register buffer pages as an RDMA memory region.
 *
 * Builds an sg_table from the buffer's pages, DMA-maps it, and optionally
 * creates a fast-reg MR for remote access.  For local-only access,
 * pd->local_dma_lkey suffices (rkey=0).
 *
 * Buffer lookup is done via linear scan of edev->buffers[] under buf_lock
 * (rather than calling dmabuf_rdma_find_buffer) because rdma_engine.c
 * needs to snapshot additional fields atomically under the same lock hold.
 *
 * Caller must hold rdma_sem read lock.
 */
int rdma_engine_register_mr(struct dmaplane_dev *edev,
			    struct dmaplane_mr_params *params)
{
	struct dmaplane_rdma_ctx *ctx = &edev->rdma;
	struct dmaplane_buffer *buf;
	struct dmaplane_mr_entry *mr_entry;
	struct sg_table *sgt = NULL;
	int i, mapped_nents, ret;
	ktime_t start;
	struct ib_device *ibdev;

	/* Buffer fields — copied under buf_lock to avoid use-after-free */
	void *buf_vaddr;
	size_t buf_size;
	unsigned int buf_nr_pages;
	struct page **pages_copy = NULL;
	bool need_fastreg;
	struct ib_mr *frmr = NULL;

	if (!ctx->initialized)
		return -EINVAL;

	/* Fast-reg needed when access includes remote or atomic permissions.
	 * local_dma_lkey only authorizes local NIC access (rkey=0). */
	need_fastreg = (params->access_flags & ~(u32)IB_ACCESS_LOCAL_WRITE) != 0;

	ibdev = ctx->ib_dev;

	/* Find the buffer and snapshot its fields under lock */
	mutex_lock(&edev->buf_lock);
	buf = NULL;
	for (i = 0; i < DMAPLANE_MAX_BUFFERS; i++) {
		if (edev->buffers[i].in_use &&
		    edev->buffers[i].id == params->buf_id) {
			buf = &edev->buffers[i];
			break;
		}
	}
	if (!buf) {
		mutex_unlock(&edev->buf_lock);
		pr_err("buffer %u not found\n", params->buf_id);
		return -ENOENT;
	}
	buf_vaddr = buf->vaddr;
	buf_size = buf->size;
	buf_nr_pages = buf->nr_pages;

	if (buf->alloc_type != DMAPLANE_BUF_TYPE_PAGES ||
	    !buf->pages || buf_nr_pages == 0) {
		mutex_unlock(&edev->buf_lock);
		pr_err("MR registration requires page-backed buffer\n");
		return -EINVAL;
	}

	/*
	 * Snapshot all page pointers into a local copy.  After we release
	 * buf_lock the buffer could be destroyed, which kvfree's the
	 * buf->pages array.  The struct page frames themselves are stable
	 * kernel objects and remain valid even if __free_page is called
	 * (the frame stays valid as a kernel data structure).
	 */
	pages_copy = kvmalloc_array(buf_nr_pages, sizeof(struct page *),
				    GFP_KERNEL);
	if (!pages_copy) {
		mutex_unlock(&edev->buf_lock);
		return -ENOMEM;
	}
	memcpy(pages_copy, buf->pages, buf_nr_pages * sizeof(struct page *));
	mutex_unlock(&edev->buf_lock);

	/*
	 * rxe (Soft-RoCE) interprets sge.addr as a kernel VA via
	 * local_dma_lkey.  Real hardware uses the IOMMU/DMA mapping.
	 * We build an sg_table from ALL pages so both paths work —
	 * ib_dma_map_sg establishes IOMMU mappings, but sge_addr is
	 * set to the kernel VA for rxe compatibility.
	 */
	if (!buf_vaddr) {
		pr_err("buffer %u has no vmap'd address\n", params->buf_id);
		kvfree(pages_copy);
		return -EINVAL;
	}

	/* Build SG table and DMA-map all pages */
	ret = register_mr_build_sgt(ibdev, pages_copy, buf_nr_pages,
				    &sgt, &mapped_nents);
	kvfree(pages_copy);
	if (ret)
		return ret;

	/* Allocate fast-reg MR if remote access requested */
	if (need_fastreg) {
		ret = register_mr_fastreg(ctx, sgt, buf_nr_pages,
					 params->access_flags, &frmr);
		if (ret)
			goto err_unmap;
	}

	/* Find a free MR slot */
	mutex_lock(&edev->mr_lock);
	mr_entry = NULL;
	for (i = 0; i < DMAPLANE_MAX_MRS; i++) {
		if (!edev->mrs[i].in_use) {
			mr_entry = &edev->mrs[i];
			break;
		}
	}
	if (!mr_entry) {
		mutex_unlock(&edev->mr_lock);
		if (frmr)
			ib_dereg_mr(frmr);
		ret = -ENOMEM;
		goto err_unmap;
	}

	start = ktime_get();

	/* Populate MR entry.
	 *
	 * Two paths:
	 *   1. Local-only (local_dma_lkey): no ib_mr needed, rkey=0.
	 *      sge_addr is the kernel VA — rxe interprets this directly.
	 *   2. Fast-reg: real MR with lkey/rkey/iova from ib_alloc_mr +
	 *      IB_WR_REG_MR.  Required for remote access (RDMA WRITE). */
	mr_entry->id = edev->next_mr_id;
	if (++edev->next_mr_id == 0)
		edev->next_mr_id = 1; /* skip 0 to avoid ambiguity */
	mr_entry->buf_id = params->buf_id;

	if (need_fastreg) {
		mr_entry->mr       = frmr;
		mr_entry->lkey     = frmr->lkey;
		mr_entry->rkey     = frmr->rkey;
		mr_entry->sge_addr = frmr->iova;
	} else {
		mr_entry->mr       = NULL;
		mr_entry->lkey     = ctx->pd->local_dma_lkey;
		mr_entry->rkey     = 0;
		mr_entry->sge_addr = (u64)(uintptr_t)buf_vaddr;
	}

	mr_entry->sgt = sgt;
	mr_entry->sgt_nents = mapped_nents;
	mr_entry->in_use = true;
	mr_entry->reg_time = ktime_sub(ktime_get(), start);

	params->mr_id = mr_entry->id;
	params->lkey = mr_entry->lkey;
	params->rkey = mr_entry->rkey;
	params->addr = mr_entry->sge_addr;

	atomic64_inc(&edev->stats.mrs_registered);
	mutex_unlock(&edev->mr_lock);

	pr_debug("MR %u registered (lkey=0x%x rkey=0x%x sge_addr=0x%llx pages=%u mapped=%d size=%zu fastreg=%d reg_time=%lldns)\n",
		 mr_entry->id, mr_entry->lkey, mr_entry->rkey,
		 mr_entry->sge_addr, buf_nr_pages, mapped_nents, buf_size,
		 need_fastreg, ktime_to_ns(mr_entry->reg_time));
	return 0;

err_unmap:
	ib_dma_unmap_sg(ibdev, sgt->sgl, sgt->nents, DMA_BIDIRECTIONAL);
	sg_free_table(sgt);
	kfree(sgt);
	return ret;
}
EXPORT_SYMBOL_GPL(rdma_engine_register_mr);

/*
 * rdma_engine_deregister_mr — Deregister and free an MR.
 *
 * Caller must hold rdma_sem read lock.
 */
int rdma_engine_deregister_mr(struct dmaplane_dev *edev, __u32 mr_id)
{
	struct dmaplane_mr_entry *mr_entry;
	int i;

	mutex_lock(&edev->mr_lock);
	mr_entry = NULL;
	for (i = 0; i < DMAPLANE_MAX_MRS; i++) {
		if (edev->mrs[i].in_use && edev->mrs[i].id == mr_id) {
			mr_entry = &edev->mrs[i];
			break;
		}
	}

	if (!mr_entry) {
		mutex_unlock(&edev->mr_lock);
		return -ENOENT;
	}

	/* Unmap the sg_table DMA mappings.  The ib_dev NULL check is
	 * defensive — normally MRs are deregistered before RDMA teardown
	 * (the teardown ioctl ensures this), but guard against unexpected
	 * ordering to avoid a NULL-pointer dereference. */
	if (mr_entry->sgt && edev->rdma.ib_dev) {
		ib_dma_unmap_sg(edev->rdma.ib_dev, mr_entry->sgt->sgl,
				mr_entry->sgt->nents, DMA_BIDIRECTIONAL);
		sg_free_table(mr_entry->sgt);
		kfree(mr_entry->sgt);
		mr_entry->sgt = NULL;
	}
	if (mr_entry->mr) {
		ib_dereg_mr(mr_entry->mr);
		mr_entry->mr = NULL;
	}
	mr_entry->in_use = false;
	atomic64_inc(&edev->stats.mrs_deregistered);
	mutex_unlock(&edev->mr_lock);

	pr_debug("MR %u deregistered\n", mr_id);
	return 0;
}
EXPORT_SYMBOL_GPL(rdma_engine_deregister_mr);

/*
 * rdma_engine_post_send — Post a send work request.
 *
 * Builds an SGE from the MR entry's sge_addr and lkey, then posts
 * via ib_post_send.  The caller provides a poll_cq_wait whose
 * cqe.done is set to poll_cq_done for managed CQ compatibility.
 */
int rdma_engine_post_send(struct dmaplane_dev *edev,
			  struct ib_qp *qp, struct dmaplane_mr_entry *mr,
			  size_t size, struct poll_cq_wait *wait)
{
	struct ib_send_wr wr = {};
	struct ib_sge sge = {};
	const struct ib_send_wr *bad_wr;
	int ret;

	sge.addr = mr->sge_addr;
	sge.length = size;
	sge.lkey = mr->lkey;

	init_completion(&wait->done);
	wait->cqe.done = poll_cq_done;

	wr.wr_cqe = &wait->cqe;
	wr.opcode = IB_WR_SEND;
	wr.send_flags = IB_SEND_SIGNALED;
	wr.sg_list = &sge;
	wr.num_sge = 1;

	ret = ib_post_send(qp, &wr, &bad_wr);
	if (ret)
		pr_err("ib_post_send failed: %d\n", ret);
	else {
		atomic64_inc(&edev->stats.sends_posted);
		trace_dmaplane_rdma_post("send", sge.length, wr.wr_id);
		pr_debug("post_send OK qp=%u sge.addr=0x%llx len=%zu lkey=0x%x\n",
			 qp->qp_num, sge.addr, size, sge.lkey);
	}

	return ret;
}
EXPORT_SYMBOL_GPL(rdma_engine_post_send);

/*
 * rdma_engine_post_recv — Post a receive work request.
 *
 * Same pattern as post_send but uses ib_post_recv with ib_recv_wr.
 */
int rdma_engine_post_recv(struct dmaplane_dev *edev,
			  struct ib_qp *qp, struct dmaplane_mr_entry *mr,
			  size_t size, struct poll_cq_wait *wait)
{
	struct ib_recv_wr wr = {};
	struct ib_sge sge = {};
	const struct ib_recv_wr *bad_wr;
	int ret;

	sge.addr = mr->sge_addr;
	sge.length = size;
	sge.lkey = mr->lkey;

	init_completion(&wait->done);
	wait->cqe.done = poll_cq_done;

	wr.wr_cqe = &wait->cqe;
	wr.sg_list = &sge;
	wr.num_sge = 1;

	ret = ib_post_recv(qp, &wr, &bad_wr);
	if (ret)
		pr_err("ib_post_recv failed: %d\n", ret);
	else {
		atomic64_inc(&edev->stats.recvs_posted);
		trace_dmaplane_rdma_post("recv", sge.length, wr.wr_id);
	}

	return ret;
}
EXPORT_SYMBOL_GPL(rdma_engine_post_recv);

/*
 * rdma_engine_flush_cq — Drain all pending completions from a CQ.
 *
 * Polls until no more completions are available.  Used between benchmark
 * runs to clear stale completions that would poison the next run.
 */
int rdma_engine_flush_cq(struct ib_cq *cq)
{
	struct ib_wc wc;
	int flushed = 0;
	int ret;

	for (;;) {
		ret = ib_poll_cq(cq, 1, &wc);
		if (ret <= 0)
			break;
		flushed++;
	}
	return flushed;
}
EXPORT_SYMBOL_GPL(rdma_engine_flush_cq);

/*
 * rdma_engine_cmp_u64 — Comparator for __u64 arrays.
 *
 * For use with kernel sort() to compute latency percentiles.
 */
int rdma_engine_cmp_u64(const void *a, const void *b)
{
	__u64 va = *(__u64 *)a;
	__u64 vb = *(__u64 *)b;

	if (va < vb)
		return -1;
	if (va > vb)
		return 1;
	return 0;
}
EXPORT_SYMBOL_GPL(rdma_engine_cmp_u64);

/* ====================================================================
 * Phase 8: Cross-machine peer QP
 * ====================================================================
 *
 * These functions manage a third QP (qp_peer) that connects to a remote
 * machine instead of looping back locally.  The loopback pair (qp_a/qp_b)
 * remains intact for benchmarking.  The peer QP shares the same PD and
 * GID established by rdma_engine_setup().
 *
 * Lifecycle: INIT_PEER → CONNECT_PEER → REMOTE_SEND / REMOTE_RECV
 * Teardown:  rdma_engine_teardown_peer() or automatically via
 *            rdma_engine_teardown() safety net.
 */

/*
 * rdma_engine_teardown_peer — Destroy the cross-machine QP and CQ.
 *
 * Follows the same protocol as loopback teardown (rdma_engine_teardown):
 *   1. Move qp_peer to IB_QPS_ERR — flushes all outstanding WRs so
 *      the RDMA subsystem releases references to our CQE/SGE memory.
 *   2. Destroy qp_peer before cq_peer (reverse creation order) to
 *      avoid dangling QP→CQ references.
 *
 * Safe to call multiple times: the NULL/false checks make it idempotent.
 * Also called as a safety net from rdma_engine_teardown() in case
 * userspace calls TEARDOWN_RDMA without explicitly cleaning up the peer.
 *
 * Caller must hold rdma_sem write lock.
 */
void rdma_engine_teardown_peer(struct dmaplane_dev *edev)
{
	struct dmaplane_rdma_ctx *ctx = &edev->rdma;
	struct ib_qp_attr attr = { .qp_state = IB_QPS_ERR };

	if (!ctx->peer_connected && !ctx->qp_peer)
		return;

	/* Flush outstanding WRs by moving QP to error state */
	if (ctx->qp_peer)
		ib_modify_qp(ctx->qp_peer, &attr, IB_QP_STATE);

	/* Destroy in reverse creation order: QP before CQ */
	if (ctx->qp_peer) {
		ib_destroy_qp(ctx->qp_peer);
		ctx->qp_peer = NULL;
	}
	if (ctx->cq_peer) {
		ib_free_cq(ctx->cq_peer);
		ctx->cq_peer = NULL;
	}
	ctx->peer_connected = false;
	pr_info("peer RDMA teardown complete\n");
}
EXPORT_SYMBOL_GPL(rdma_engine_teardown_peer);

/*
 * rdma_engine_init_peer — Create a QP for cross-machine RDMA.
 *
 * Prerequisite: rdma_engine_setup() must have been called first to
 * establish the PD, GID, and IB device reference.  This function reuses
 * those shared resources to create a third QP (qp_peer) alongside the
 * existing loopback pair (qp_a/qp_b).
 *
 * Resource creation order:
 *   1. cq_peer — IB_POLL_DIRECT CQ for deterministic busy-polling
 *   2. qp_peer — RC QP bound to cq_peer for both send and recv CQs
 *   3. Transition qp_peer to INIT via qp_to_init()
 *
 * Output: fills the info struct with local connection metadata
 * (QP number, LID, GID, DMAC) that userspace sends to the remote
 * machine over TCP.  The remote machine passes this metadata to its
 * own CONNECT_PEER ioctl to complete the QP handshake.
 *
 * DMAC extraction: RoCEv2 requires the destination MAC in the Address
 * Handle for Ethernet framing.  Each machine exports its own MAC here;
 * the remote side places it in ah_attr.roce.dmac during RTR transition.
 *
 * Caller must hold rdma_sem read lock.
 */
int rdma_engine_init_peer(struct dmaplane_dev *edev,
			  struct dmaplane_rdma_peer_info *info)
{
	struct dmaplane_rdma_ctx *ctx = &edev->rdma;
	struct ib_qp_init_attr qp_init;
	const struct ib_gid_attr *sgid_attr;
	struct net_device *ndev;
	int ret;

	if (!ctx->initialized)
		return -EINVAL;

	if (ctx->qp_peer) {
		pr_warn("peer QP already initialized\n");
		return -EBUSY;
	}

	/* Create peer CQ — same IB_POLL_DIRECT config as loopback CQs */
	ctx->cq_peer = ib_alloc_cq(ctx->ib_dev, NULL, 128,
				    0, IB_POLL_DIRECT);
	if (IS_ERR(ctx->cq_peer)) {
		ret = PTR_ERR(ctx->cq_peer);
		pr_err("ib_alloc_cq (peer) failed: %d\n", ret);
		ctx->cq_peer = NULL;
		return ret;
	}

	/* Create peer QP — RC, signal all WRs, bound to cq_peer */
	memset(&qp_init, 0, sizeof(qp_init));
	qp_init.send_cq = ctx->cq_peer;
	qp_init.recv_cq = ctx->cq_peer;
	qp_init.qp_type = IB_QPT_RC;
	qp_init.sq_sig_type = IB_SIGNAL_ALL_WR;
	qp_init.cap.max_send_wr = 64;
	qp_init.cap.max_recv_wr = 64;
	qp_init.cap.max_send_sge = 1;
	qp_init.cap.max_recv_sge = 1;

	ctx->qp_peer = ib_create_qp(ctx->pd, &qp_init);
	if (IS_ERR(ctx->qp_peer)) {
		ret = PTR_ERR(ctx->qp_peer);
		pr_err("ib_create_qp (peer) failed: %d\n", ret);
		ctx->qp_peer = NULL;
		goto err_free_cq;
	}

	/* Transition to INIT state */
	ret = qp_to_init(ctx->qp_peer, ctx->port);
	if (ret) {
		pr_err("peer qp_to_init failed: %d\n", ret);
		goto err_destroy_qp;
	}

	/* Fill output struct with local connection metadata */
	memset(info, 0, sizeof(*info));
	info->qp_num = ctx->qp_peer->qp_num;
	info->lid = ctx->lid;
	memcpy(info->gid, ctx->gid.raw, 16);

	/* Get local DMAC from net device — same pattern as
	 * connect_loopback_qps().  Each machine fills its own MAC;
	 * the remote side uses it in ah_attr.roce.dmac during RTR. */
	sgid_attr = rdma_get_gid_attr(ctx->ib_dev, ctx->port, ctx->gid_index);
	if (IS_ERR(sgid_attr)) {
		ret = PTR_ERR(sgid_attr);
		pr_err("rdma_get_gid_attr failed: %d\n", ret);
		goto err_destroy_qp;
	}

	rcu_read_lock();
	ndev = rdma_read_gid_attr_ndev_rcu(sgid_attr);
	if (!IS_ERR(ndev))
		memcpy(info->mac, ndev->dev_addr, ETH_ALEN);
	rcu_read_unlock();
	rdma_put_gid_attr(sgid_attr);

	info->status = 0;

	pr_info("peer QP initialized (qp_num=%u, gid=%pI6c)\n",
		info->qp_num, info->gid);
	return 0;

err_destroy_qp:
	ib_destroy_qp(ctx->qp_peer);
	ctx->qp_peer = NULL;
err_free_cq:
	ib_free_cq(ctx->cq_peer);
	ctx->cq_peer = NULL;
	return ret;
}
EXPORT_SYMBOL_GPL(rdma_engine_init_peer);

/*
 * rdma_engine_connect_peer — Connect local peer QP to remote machine.
 *
 * Completes the QP state machine transitions:
 *   INIT → RTR (Ready-to-Receive): sets the destination QP number,
 *     GID, DMAC, and path MTU.  After RTR, the QP can receive packets
 *     from the remote peer.
 *   RTR → RTS (Ready-to-Send): sets the send PSN and retry parameters.
 *     After RTS, the QP can both send and receive.
 *
 * The critical difference from loopback: qp_to_rtr() receives the
 * REMOTE machine's QP number, GID, and DMAC (obtained via TCP exchange)
 * instead of the local loopback peer's values.  The remote->mac is
 * placed into ah_attr.roce.dmac so rxe constructs Ethernet frames with
 * the correct destination MAC.
 *
 * Caller must hold rdma_sem read lock.
 */
int rdma_engine_connect_peer(struct dmaplane_dev *edev,
			     struct dmaplane_rdma_peer_info *remote)
{
	struct dmaplane_rdma_ctx *ctx = &edev->rdma;
	const struct ib_gid_attr *sgid_attr;
	int ret;

	if (!ctx->qp_peer)
		return -EINVAL;

	sgid_attr = rdma_get_gid_attr(ctx->ib_dev, ctx->port, ctx->gid_index);
	if (IS_ERR(sgid_attr))
		return PTR_ERR(sgid_attr);

	/* INIT → RTR: remote machine's QP num, GID, and DMAC */
	ret = qp_to_rtr(ctx->qp_peer, ctx->port, remote->qp_num,
			remote->lid, (union ib_gid *)remote->gid,
			ctx->gid_index, sgid_attr, remote->mac);
	if (ret) {
		pr_err("peer qp_to_rtr failed: %d\n", ret);
		rdma_put_gid_attr(sgid_attr);
		return ret;
	}

	/* RTR → RTS */
	ret = qp_to_rts(ctx->qp_peer);
	rdma_put_gid_attr(sgid_attr);
	if (ret) {
		pr_err("peer qp_to_rts failed: %d\n", ret);
		return ret;
	}

	ctx->peer_connected = true;
	remote->status = 0;
	pr_info("peer QP connected (remote qp_num=%u)\n", remote->qp_num);
	return 0;
}
EXPORT_SYMBOL_GPL(rdma_engine_connect_peer);

/*
 * rdma_engine_remote_send — Post a send on the peer QP and wait.
 *
 * Data flow:
 *   1. MR lookup: finds the MR in the shared mrs[] array.  The MR can
 *      be host-backed (sge_addr = vmap'd kernel VA) or GPU-backed
 *      (sge_addr = contiguous WC BAR mapping).  Both use the same
 *      ID namespace; this function is agnostic to the backing type.
 *   2. Post send: ib_post_send() on qp_peer.  rxe reads from sge_addr
 *      via memcpy; for GPU-backed MRs this becomes memcpy from the WC
 *      BAR mapping (PCIe reads from GPU VRAM).
 *   3. rxe fragments the data into UDP packets and sends them out.
 *
 * Timeout: 10s.  The remote receiver should already have its recv posted.
 *
 * MR snapshot under lock: struct is copied by value so a concurrent
 * deregister can't invalidate the local copy during the work request.
 *
 * Caller must hold rdma_sem read lock.
 */
int rdma_engine_remote_send(struct dmaplane_dev *edev,
			    struct dmaplane_rdma_remote_xfer_params *params)
{
	struct dmaplane_rdma_ctx *ctx = &edev->rdma;
	struct dmaplane_mr_entry mr_copy;
	struct poll_cq_wait send_wait;
	struct ib_wc wc;
	ktime_t start;
	int ret, rc;

	if (!ctx->peer_connected)
		return -EINVAL;

	/* Snapshot MR under lock */
	mutex_lock(&edev->mr_lock);
	{
		struct dmaplane_mr_entry *mr;

		mr = dmabuf_rdma_find_mr(edev, params->mr_id);
		if (!mr) {
			mutex_unlock(&edev->mr_lock);
			return -ENOENT;
		}
		mr_copy = *mr;
	}
	mutex_unlock(&edev->mr_lock);

	start = ktime_get();

	ret = rdma_engine_post_send(edev, ctx->qp_peer, &mr_copy,
				    params->size, &send_wait);
	if (ret)
		return ret;

	/* Poll peer CQ — 10s timeout */
	rc = rdma_engine_poll_cq(ctx->cq_peer, &wc, 10000);
	if (rc <= 0) {
		pr_err("remote send %s\n",
		       rc == 0 ? "timed out" : "poll failed");
		params->status = rc == 0 ? 1 : (__u32)(-rc);
		return rc == 0 ? -ETIMEDOUT : rc;
	}
	if (wc.status != IB_WC_SUCCESS) {
		pr_err("remote send completion error: %d\n", wc.status);
		params->status = wc.status;
		atomic64_inc(&edev->stats.completion_errors);
		return -EIO;
	}

	atomic64_inc(&edev->stats.completions_polled);
	atomic64_add(params->size, &edev->stats.bytes_sent);

	params->elapsed_ns = ktime_to_ns(ktime_sub(ktime_get(), start));
	if (params->elapsed_ns > 0)
		params->throughput_mbps = (u64)params->size * 1000 /
					  params->elapsed_ns;
	else
		params->throughput_mbps = 0;
	params->status = 0;

	pr_info("remote send OK — %u bytes, %llu ns\n",
		params->size, (unsigned long long)params->elapsed_ns);
	return 0;
}
EXPORT_SYMBOL_GPL(rdma_engine_remote_send);

/*
 * rdma_engine_remote_recv — Post a recv on the peer QP and wait.
 *
 * Data flow:
 *   1. MR lookup: finds the MR (host or GPU-backed).  sge_addr tells
 *      rxe where to write incoming data.
 *   2. Post recv: ib_post_recv() on qp_peer.  SGE length must be >=
 *      the sender's send size.
 *   3. Poll CQ: busy-polls with usleep_range(50,200) and cond_resched()
 *      for up to 30 seconds.
 *   4. On completion, wc.byte_len = actual received bytes.
 *
 * The 30s timeout is deliberately long: this ioctl blocks while waiting
 * for the remote sender.  The receiver must call this BEFORE the sender.
 *
 * After successful recv, userspace can mmap the buffer to read the data
 * zero-copy (same physical pages rxe wrote into).
 *
 * Caller must hold rdma_sem read lock.
 */
int rdma_engine_remote_recv(struct dmaplane_dev *edev,
			    struct dmaplane_rdma_remote_xfer_params *params)
{
	struct dmaplane_rdma_ctx *ctx = &edev->rdma;
	struct dmaplane_mr_entry mr_copy;
	struct poll_cq_wait recv_wait;
	struct ib_wc wc;
	ktime_t start;
	int ret, rc;

	if (!ctx->peer_connected)
		return -EINVAL;

	/* Snapshot MR under lock */
	mutex_lock(&edev->mr_lock);
	{
		struct dmaplane_mr_entry *mr;

		mr = dmabuf_rdma_find_mr(edev, params->mr_id);
		if (!mr) {
			mutex_unlock(&edev->mr_lock);
			return -ENOENT;
		}
		mr_copy = *mr;
	}
	mutex_unlock(&edev->mr_lock);

	start = ktime_get();

	ret = rdma_engine_post_recv(edev, ctx->qp_peer, &mr_copy,
				    params->size, &recv_wait);
	if (ret)
		return ret;

	/* Poll peer CQ — 30s timeout (waiting for remote sender) */
	rc = rdma_engine_poll_cq(ctx->cq_peer, &wc, 30000);
	if (rc <= 0) {
		pr_err("remote recv %s\n",
		       rc == 0 ? "timed out" : "poll failed");
		params->status = rc == 0 ? 1 : (__u32)(-rc);
		return rc == 0 ? -ETIMEDOUT : rc;
	}
	if (wc.status != IB_WC_SUCCESS) {
		pr_err("remote recv completion error: %d\n", wc.status);
		params->status = wc.status;
		atomic64_inc(&edev->stats.completion_errors);
		return -EIO;
	}

	atomic64_inc(&edev->stats.completions_polled);
	atomic64_add(wc.byte_len, &edev->stats.bytes_received);

	params->elapsed_ns = ktime_to_ns(ktime_sub(ktime_get(), start));
	if (params->elapsed_ns > 0)
		params->throughput_mbps = (u64)wc.byte_len * 1000 /
					  params->elapsed_ns;
	else
		params->throughput_mbps = 0;
	params->status = 0;

	pr_info("remote recv OK — %u bytes, %llu ns\n",
		wc.byte_len, (unsigned long long)params->elapsed_ns);
	return 0;
}
EXPORT_SYMBOL_GPL(rdma_engine_remote_recv);

/* ── Phase 9: RDMA WRITE with Immediate ──────────────────────────────── */

/*
 * rdma_engine_write_imm — Post RDMA WRITE WITH IMMEDIATE.
 *
 * Builds an ib_rdma_wr (not plain ib_send_wr) because RDMA WRITE needs
 * remote_addr and rkey.  Posts IB_WR_RDMA_WRITE_WITH_IMM on the selected
 * QP, then polls the corresponding send CQ for completion.
 *
 * The 32-bit imm_data is converted to network byte order (cpu_to_be32)
 * as required by the IB spec.  The receiver converts back (be32_to_cpu).
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
			   uint64_t *elapsed_ns)
{
	struct dmaplane_rdma_ctx *ctx = &edev->rdma;
	struct dmaplane_mr_entry mr_copy;
	struct ib_rdma_wr rdma_wr = {};
	struct ib_sge sge = {};
	const struct ib_send_wr *bad_wr;
	struct ib_qp *qp;
	struct ib_cq *cq;
	struct ib_wc wc;
	ktime_t start;
	int ret, rc;

	if (!ctx->initialized)
		return -EINVAL;

	if (use_peer_qp) {
		if (!ctx->peer_connected)
			return -ENOTCONN;
		qp = ctx->qp_peer;
		cq = ctx->cq_peer;
	} else {
		qp = ctx->qp_a;
		cq = ctx->cq_a;
	}

	/* Snapshot MR under lock */
	mutex_lock(&edev->mr_lock);
	{
		struct dmaplane_mr_entry *mr;

		mr = dmabuf_rdma_find_mr(edev, local_mr_id);
		if (!mr) {
			mutex_unlock(&edev->mr_lock);
			return -ENOENT;
		}
		mr_copy = *mr;
	}
	mutex_unlock(&edev->mr_lock);

	start = ktime_get();

	/* Build SGE with offset into local MR */
	sge.addr = mr_copy.sge_addr + local_offset;
	sge.length = length;
	sge.lkey = mr_copy.lkey;

	/* Build RDMA WRITE WITH IMMEDIATE work request */
	rdma_wr.wr.opcode = IB_WR_RDMA_WRITE_WITH_IMM;
	rdma_wr.wr.send_flags = IB_SEND_SIGNALED;
	rdma_wr.wr.ex.imm_data = cpu_to_be32(imm_data);
	rdma_wr.wr.sg_list = &sge;
	rdma_wr.wr.num_sge = 1;
	rdma_wr.remote_addr = remote_addr;
	rdma_wr.rkey = remote_rkey;

	ret = ib_post_send(qp, &rdma_wr.wr, &bad_wr);
	if (ret) {
		pr_err("write_imm: ib_post_send failed: %d\n", ret);
		return ret;
	}
	atomic64_inc(&edev->stats.sends_posted);

	/* Poll send CQ — 10s timeout */
	rc = rdma_engine_poll_cq(cq, &wc, 10000);
	if (rc <= 0) {
		pr_err("write_imm send CQ %s\n",
		       rc == 0 ? "timed out" : "poll failed");
		return rc == 0 ? -ETIMEDOUT : rc;
	}
	if (wc.status != IB_WC_SUCCESS) {
		pr_err("write_imm completion error: %d\n", wc.status);
		atomic64_inc(&edev->stats.completion_errors);
		return -EIO;
	}

	atomic64_inc(&edev->stats.completions_polled);
	atomic64_add(length, &edev->stats.bytes_sent);

	if (elapsed_ns) {
		*elapsed_ns = ktime_to_ns(ktime_sub(ktime_get(), start));
		trace_dmaplane_rdma_write_imm(imm_data, length, *elapsed_ns);
	}

	pr_debug("write_imm OK — imm=0x%08x len=%u\n", imm_data, length);
	return 0;
}
EXPORT_SYMBOL_GPL(rdma_engine_write_imm);

/*
 * rdma_engine_writeimm_post_recv — Post a recv WR for WRITEIMM consumption.
 *
 * Each RDMA WRITE WITH IMMEDIATE consumes one recv WR on the receiving QP.
 * Posts a recv on qp_b (loopback) or qp_peer (cross-machine).
 *
 * Unlike the low-level rdma_engine_post_recv() which takes raw (qp, mr, wait),
 * this handles MR lookup and QP selection from the UAPI use_peer_qp flag.
 *
 * Caller must hold rdma_sem read lock.
 */
int rdma_engine_writeimm_post_recv(struct dmaplane_dev *edev,
				    uint32_t mr_id,
				    uint32_t size,
				    int use_peer_qp)
{
	struct dmaplane_rdma_ctx *ctx = &edev->rdma;
	struct dmaplane_mr_entry mr_copy;
	struct ib_recv_wr wr = {};
	struct ib_sge sge = {};
	const struct ib_recv_wr *bad_wr;
	struct ib_qp *qp;
	int ret;

	if (!ctx->initialized)
		return -EINVAL;

	if (use_peer_qp) {
		if (!ctx->peer_connected)
			return -ENOTCONN;
		qp = ctx->qp_peer;
	} else {
		qp = ctx->qp_b;
	}

	/* Snapshot MR under lock */
	mutex_lock(&edev->mr_lock);
	{
		struct dmaplane_mr_entry *mr;

		mr = dmabuf_rdma_find_mr(edev, mr_id);
		if (!mr) {
			mutex_unlock(&edev->mr_lock);
			return -ENOENT;
		}
		mr_copy = *mr;
	}
	mutex_unlock(&edev->mr_lock);

	sge.addr = mr_copy.sge_addr;
	sge.length = size;
	sge.lkey = mr_copy.lkey;

	wr.sg_list = &sge;
	wr.num_sge = 1;

	ret = ib_post_recv(qp, &wr, &bad_wr);
	if (ret) {
		pr_err("writeimm_post_recv: ib_post_recv failed: %d\n", ret);
		return ret;
	}

	atomic64_inc(&edev->stats.recvs_posted);
	pr_debug("writeimm_post_recv OK mr=%u size=%u peer=%d\n",
		 mr_id, size, use_peer_qp);
	return 0;
}
EXPORT_SYMBOL_GPL(rdma_engine_writeimm_post_recv);

/*
 * rdma_engine_writeimm_poll_recv — Poll recv CQ for WRITEIMM completion.
 *
 * RDMA WRITE WITH IMMEDIATE delivers the 32-bit immediate value through
 * the receiver's CQ completion entry (wc.ex.imm_data).  This polls the
 * recv CQ and extracts imm_data in host byte order (be32_to_cpu).
 *
 * Return semantics: the ioctl returns 0 even on timeout; the caller
 * distinguishes success from timeout via status_out (0=success, 1=timeout).
 * Only catastrophic WC errors return negative (-EIO).
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
				    uint64_t *elapsed_ns)
{
	struct dmaplane_rdma_ctx *ctx = &edev->rdma;
	struct ib_cq *cq;
	struct ib_wc wc;
	ktime_t start;
	int rc;

	if (!ctx->initialized)
		return -EINVAL;

	if (use_peer_qp) {
		if (!ctx->peer_connected)
			return -ENOTCONN;
		cq = ctx->cq_peer;
	} else {
		cq = ctx->cq_b;
	}

	start = ktime_get();

	rc = rdma_engine_poll_cq(cq, &wc, timeout_ms);
	*elapsed_ns = ktime_to_ns(ktime_sub(ktime_get(), start));

	if (rc == 0) {
		/* Timeout — not an error, caller checks status */
		*status_out = 1;
		*wc_flags_out = 0;
		*imm_data_out = 0;
		*byte_len_out = 0;
		return 0;
	}
	if (rc < 0) {
		pr_err("writeimm_poll_recv: poll failed: %d\n", rc);
		*status_out = (__u32)(-rc);
		return rc;
	}

	/* Got a completion */
	if (wc.status != IB_WC_SUCCESS) {
		pr_err("writeimm_poll_recv: WC error status=%d\n", wc.status);
		*status_out = wc.status;
		atomic64_inc(&edev->stats.completion_errors);
		return -EIO;
	}

	*status_out = 0;
	*wc_flags_out = wc.wc_flags;
	*imm_data_out = (wc.wc_flags & IB_WC_WITH_IMM) ?
			be32_to_cpu(wc.ex.imm_data) : 0;
	*byte_len_out = wc.byte_len;

	atomic64_inc(&edev->stats.completions_polled);
	atomic64_add(wc.byte_len, &edev->stats.bytes_received);

	pr_debug("writeimm_poll_recv OK — imm=0x%08x len=%u\n",
		 *imm_data_out, *byte_len_out);
	return 0;
}
EXPORT_SYMBOL_GPL(rdma_engine_writeimm_poll_recv);
