// SPDX-License-Identifier: GPL-2.0
/*
 * Copyright (c) 2026 Graziano Labs Corp.
 *
 * This file is part of dmaplane.
 *
 * dmaplane is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License.
 *
 */

/*
 * ============================================================================
 *  receiver.c — Machine B: RDMA Receive → dmaplane
 * ============================================================================
 *
 *  Mirrors sender.c on the receive side. Uses dmaplane.ko for
 *  host-side buffer management, rdma_cm for cross-machine transport.
 *
 *  Build:
 *    gcc -O2 -Wall -o receiver receiver.c -libverbs -lrdmacm -lm
 *
 *  Run (start BEFORE sender):
 *    sudo ./receiver <port> [ib_device]
 *
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <math.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <arpa/inet.h>

#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>

#include "dmaplane_uapi.h"

#define DMAPLANE_DEVICE "/dev/dmaplane"

/* ──────────────────────────────────────────────────────────────────────────
 *  rdma_cm server
 * ────────────────────────────────────────────────────────────────────────── */

struct rdma_conn {
    struct rdma_cm_id         *listen_id;
    struct rdma_cm_id         *cm_id;
    struct rdma_event_channel *ev_ch;
    struct ibv_pd             *pd;
    struct ibv_cq             *cq;
    struct ibv_qp             *qp;
    struct ibv_mr             *mr;
};

static int wait_cm_event(struct rdma_conn *c, enum rdma_cm_event_type exp,
                         struct rdma_cm_id **out_id)
{
    struct rdma_cm_event *ev;
    rdma_get_cm_event(c->ev_ch, &ev);
    int ok = (ev->event == exp);
    if (!ok)
        fprintf(stderr, "    CM event: expected %d, got %d\n", exp, ev->event);
    if (out_id) *out_id = ev->id;
    rdma_ack_cm_event(ev);
    return ok ? 0 : -1;
}

static int listen_and_accept(struct rdma_conn *c, int port)
{
    struct sockaddr_in addr = {
        .sin_family      = AF_INET,
        .sin_port        = htons(port),
        .sin_addr.s_addr = INADDR_ANY,
    };

    c->ev_ch = rdma_create_event_channel();
    if (!c->ev_ch) { perror("event_channel"); return -1; }

    if (rdma_create_id(c->ev_ch, &c->listen_id, NULL, RDMA_PS_TCP))
    { perror("create_id"); return -1; }

    if (rdma_bind_addr(c->listen_id, (struct sockaddr *)&addr))
    { perror("bind"); return -1; }

    if (rdma_listen(c->listen_id, 1))
    { perror("listen"); return -1; }

    printf("[RDMA] Listening on port %d...\n", port);

    if (wait_cm_event(c, RDMA_CM_EVENT_CONNECT_REQUEST, &c->cm_id))
        return -1;

    c->pd = ibv_alloc_pd(c->cm_id->verbs);
    c->cq = ibv_create_cq(c->cm_id->verbs, 32, NULL, NULL, 0);

    struct ibv_qp_init_attr qa = {
        .send_cq = c->cq, .recv_cq = c->cq,
        .qp_type = IBV_QPT_RC,
        .cap = { .max_send_wr = 16, .max_recv_wr = 16,
                 .max_send_sge = 1, .max_recv_sge = 1 },
    };
    if (rdma_create_qp(c->cm_id, c->pd, &qa))
    { perror("create_qp"); return -1; }
    c->qp = c->cm_id->qp;

    struct rdma_conn_param cp = {
        .responder_resources = 1,
        .initiator_depth = 1,
    };
    /*
     * NOTE: caller must register MR and post recv AFTER this function
     * returns but BEFORE the sender posts its send. We handle this by
     * having the sender sleep briefly, or by using rdma_cm private_data
     * as a ready signal. For production, pre-post recvs before accept.
     */
    if (rdma_accept(c->cm_id, &cp))
    { perror("accept"); return -1; }

    if (wait_cm_event(c, RDMA_CM_EVENT_ESTABLISHED, NULL))
        return -1;

    printf("[RDMA] Connection established.\n");
    return 0;
}

static void destroy_conn(struct rdma_conn *c)
{
    if (c->mr)         ibv_dereg_mr(c->mr);
    if (c->qp)         rdma_destroy_qp(c->cm_id);
    if (c->cq)         ibv_destroy_cq(c->cq);
    if (c->pd)         ibv_dealloc_pd(c->pd);
    if (c->cm_id)    { rdma_disconnect(c->cm_id); rdma_destroy_id(c->cm_id); }
    if (c->listen_id)  rdma_destroy_id(c->listen_id);
    if (c->ev_ch)      rdma_destroy_event_channel(c->ev_ch);
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
        float exp = s * sinf((float)i * 0.001f + (float)layer);
        if (fabsf(buf[i] - exp) > 1e-5f) {
            if (errs < 5)
                fprintf(stderr, "    [%zu] expected=%.6f got=%.6f\n",
                        i, exp, buf[i]);
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
    const char *ib_dev = (argc > 2) ? argv[2] : "rxe_enp44s0";

    const size_t BUF_SIZE   = 16 * 1024 * 1024;
    const size_t NUM_FLOATS = BUF_SIZE / sizeof(float);
    const int    LAYER      = 42;

    int emu_fd = -1;
    float *data = NULL;
    int data_is_mmap = 0;
    struct rdma_conn conn = {0};
    struct dmaplane_buf_params  buf = {0};
    struct dmaplane_mr_params   mr  = {0};
    int ret = 0;

    printf("=== dmaplane: Cross-Machine Gradient Receive ===\n");
    printf("    Device: %s | IB: %s | Port: %d\n\n", DMAPLANE_DEVICE, ib_dev, port);

    /* ── 1. Open kernel module ── */
    printf("[1] Opening %s...\n", DMAPLANE_DEVICE);
    emu_fd = open(DMAPLANE_DEVICE, O_RDWR);
    if (emu_fd < 0) {
        perror("    open");
        fprintf(stderr, "    sudo insmod dmaplane.ko\n");
        return 1;
    }

    /* ── 2. Allocate buffer via kernel module ── */
    printf("[2] IOCTL_CREATE_BUFFER (%zu MB, page-backed)...\n", BUF_SIZE >> 20);
    buf.alloc_type = BUF_TYPE_PAGES;
    buf.size       = BUF_SIZE;
    buf.numa_node  = DMAPLANE_NUMA_ANY;
    if (ioctl(emu_fd, IOCTL_CREATE_BUFFER, &buf) < 0) {
        perror("    IOCTL_CREATE_BUFFER");
        ret = 1; goto out;
    }
    printf("    buf_id=%u\n", buf.buf_id);

    /* ── 3. Setup kernel RDMA ──
     *
     *  Same rationale as sender: kernel-side PD/CQ/QP establishes the
     *  DMA mapping infrastructure (local_dma_lkey, ib_dma_map_sg).
     *  The actual cross-machine receive uses rdma_cm in userspace (step 6).
     */
    printf("[3] IOCTL_SETUP_RDMA on %s...\n", ib_dev);
    struct dmaplane_rdma_setup setup = {
        .port        = 1,
        .cq_depth    = 256,
        .max_send_wr = 64,
        .max_recv_wr = 64,
        .status      = 0,
    };
    strncpy(setup.ib_dev_name, ib_dev, sizeof(setup.ib_dev_name) - 1);

    if (ioctl(emu_fd, IOCTL_SETUP_RDMA, &setup) < 0) {
        perror("    IOCTL_SETUP_RDMA");
        ret = 1; goto out;
    }
    printf("    status=%u\n", setup.status);

    /* ── 4. Register MR ── */
    printf("[4] IOCTL_REGISTER_MR (buf_id=%u)...\n", buf.buf_id);
    mr.buf_id       = buf.buf_id;
    mr.access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;

    if (ioctl(emu_fd, IOCTL_REGISTER_MR, &mr) < 0) {
        perror("    IOCTL_REGISTER_MR");
        ret = 1; goto out;
    }
    printf("    mr_id=%u, lkey=0x%x, rkey=0x%x\n", mr.mr_id, mr.lkey, mr.rkey);

    /* ── 5. mmap kernel buffer into userspace ──
     *
     *  Map the kernel module's DMA pages directly. The NIC will
     *  DMA-write incoming data into these same physical pages.
     */
    printf("[5] mmap kernel buffer into userspace (%zu MB)...\n", BUF_SIZE >> 20);

    struct dmaplane_mmap_info minfo = { .buf_id = buf.buf_id };
    if (ioctl(emu_fd, IOCTL_GET_MMAP_INFO, &minfo) < 0) {
        perror("    IOCTL_GET_MMAP_INFO");
        ret = 1; goto out;
    }

    data = mmap(NULL, minfo.mmap_size, PROT_READ | PROT_WRITE,
                MAP_SHARED, emu_fd, minfo.mmap_offset);
    if (data == MAP_FAILED) {
        perror("    mmap");
        data = NULL;
        ret = 1; goto out;
    }
    data_is_mmap = 1;
    memset(data, 0, BUF_SIZE);
    printf("    mmap'd %llu bytes at %p (kernel DMA pages)\n",
           (unsigned long long)minfo.mmap_size, (void *)data);

    /* ── 6. Start RDMA server ── */
    printf("[6] Starting RDMA server...\n");
    if (listen_and_accept(&conn, port) < 0) {
        ret = 1; goto out;
    }

    /* Dual-MR pattern (see sender.c step 6): kernel MR (step 4) owns
     * the DMA mappings; this userspace MR provides the lkey for the
     * rdma_cm QP's recv work requests on the same physical pages.
     *
     * Must register and post recv IMMEDIATELY after connection setup,
     * before the sender can fire its send. */
    conn.mr = ibv_reg_mr(conn.pd, data, BUF_SIZE,
                         IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!conn.mr) { perror("ibv_reg_mr"); ret = 1; goto out; }

    /* Post recv RIGHT AWAY */
    printf("[7] Posting RDMA Receive (%zu MB)...\n", BUF_SIZE >> 20);

    struct ibv_sge sge = {
        .addr = (uintptr_t)data, .length = BUF_SIZE, .lkey = conn.mr->lkey,
    };
    struct ibv_recv_wr rwr = {
        .wr_id = 1, .sg_list = &sge, .num_sge = 1,
    };
    struct ibv_recv_wr *bad = NULL;

    if (ibv_post_recv(conn.qp, &rwr, &bad)) {
        perror("    ibv_post_recv"); ret = 1; goto out;
    }

    printf("    Waiting for incoming gradient tensor...\n");

    struct ibv_wc wc;
    int n;
    do { n = ibv_poll_cq(conn.cq, 1, &wc); } while (n == 0);
    if (n < 0 || wc.status != IBV_WC_SUCCESS) {
        fprintf(stderr, "    Recv failed: %s\n", ibv_wc_status_str(wc.status));
        ret = 1; goto out;
    }
    printf("    Received %u bytes.\n", wc.byte_len);

    /* ── 8. Verify ── */
    printf("[8] Verifying gradient data (layer %d)...\n", LAYER);
    printf("    First 4: [%.6f, %.6f, %.6f, %.6f]\n",
           data[0], data[1], data[2], data[3]);

    int errs = verify_gradient(data, NUM_FLOATS, LAYER);
    if (errs == 0)
        printf("    PASSED — all 1000 sampled values match.\n");
    else
        printf("    FAILED — %d mismatches.\n", errs);

    /* ── 9. Kernel stats ── */
    printf("[9] Kernel module RDMA stats:\n");
    struct dmaplane_rdma_stats rst;
    memset(&rst, 0, sizeof(rst));
    if (ioctl(emu_fd, DMAPLANE_IOCTL_GET_RDMA_STATS, &rst) == 0) {
        printf("    mrs_registered:     %llu\n", (unsigned long long)rst.mrs_registered);
        printf("    completions_polled: %llu\n", (unsigned long long)rst.completions_polled);
        printf("    completion_errors:  %llu\n", (unsigned long long)rst.completion_errors);
    }

    printf("\n=== End-to-End Path (zero-copy mmap) ===\n");
    printf("  Sender:   emu buffer → mmap → fill → rdma_cm send → wire\n");
    printf("  Receiver: emu buffer → mmap → rdma_cm recv → verify\n");
    printf("  Both:     userspace writes/reads kernel DMA pages directly\n");
    printf("  Wire:     %s (soft-RoCE)\n", ib_dev);

out:
    destroy_conn(&conn);
    if (data && data_is_mmap)
        munmap(data, BUF_SIZE);
    else if (data)
        free(data);
    if (emu_fd >= 0) {
        if (mr.mr_id)  ioctl(emu_fd, IOCTL_DEREGISTER_MR, &mr.mr_id);
        ioctl(emu_fd, IOCTL_TEARDOWN_RDMA, 0);
        if (buf.buf_id) ioctl(emu_fd, IOCTL_DESTROY_BUFFER, &buf.buf_id);
        close(emu_fd);
    }
    return ret;
}
