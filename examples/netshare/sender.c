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
 *  sender.c — Machine A: dmaplane → RDMA Send
 * ============================================================================
 *
 *  Uses the dmaplane.ko kernel module (/dev/dmaplane) for host-side
 *  buffer management, and rdma_cm for cross-machine RDMA transport
 *  over soft-RoCE.
 *
 *  Build:
 *    gcc -O2 -Wall -o sender sender.c -libverbs -lrdmacm -lm
 *
 *  Run:
 *    sudo ./sender <receiver-ip> <port> [ib_device]
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
 *  rdma_cm connection helper
 * ────────────────────────────────────────────────────────────────────────── */

struct rdma_conn {
    struct rdma_cm_id         *cm_id;
    struct rdma_event_channel *ev_ch;
    struct ibv_pd             *pd;
    struct ibv_cq             *cq;
    struct ibv_qp             *qp;
    struct ibv_mr             *mr;
};

static int wait_cm_event(struct rdma_conn *c, enum rdma_cm_event_type expected)
{
    struct rdma_cm_event *ev;
    rdma_get_cm_event(c->ev_ch, &ev);
    int ok = (ev->event == expected);
    if (!ok)
        fprintf(stderr, "    CM event: expected %d, got %d\n", expected, ev->event);
    rdma_ack_cm_event(ev);
    return ok ? 0 : -1;
}

static int connect_to_receiver(struct rdma_conn *c, const char *ip, int port)
{
    struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_port   = htons(port),
    };
    inet_pton(AF_INET, ip, &addr.sin_addr);

    c->ev_ch = rdma_create_event_channel();
    if (!c->ev_ch) { perror("event_channel"); return -1; }

    if (rdma_create_id(c->ev_ch, &c->cm_id, NULL, RDMA_PS_TCP))
    { perror("create_id"); return -1; }

    if (rdma_resolve_addr(c->cm_id, NULL, (struct sockaddr *)&addr, 2000))
    { perror("resolve_addr"); return -1; }
    if (wait_cm_event(c, RDMA_CM_EVENT_ADDR_RESOLVED)) return -1;

    if (rdma_resolve_route(c->cm_id, 2000))
    { perror("resolve_route"); return -1; }
    if (wait_cm_event(c, RDMA_CM_EVENT_ROUTE_RESOLVED)) return -1;

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
        .retry_count = 7,
    };
    if (rdma_connect(c->cm_id, &cp))
    { perror("connect"); return -1; }
    if (wait_cm_event(c, RDMA_CM_EVENT_ESTABLISHED)) return -1;

    printf("[RDMA] Connected to %s:%d\n", ip, port);
    return 0;
}

static void destroy_conn(struct rdma_conn *c)
{
    if (c->mr)       ibv_dereg_mr(c->mr);
    if (c->qp)       rdma_destroy_qp(c->cm_id);
    if (c->cq)       ibv_destroy_cq(c->cq);
    if (c->pd)       ibv_dealloc_pd(c->pd);
    if (c->cm_id)  { rdma_disconnect(c->cm_id); rdma_destroy_id(c->cm_id); }
    if (c->ev_ch)    rdma_destroy_event_channel(c->ev_ch);
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
        fprintf(stderr, "Usage: %s <receiver-ip> <port> [ib_device]\n", argv[0]);
        return 1;
    }

    const char *ip     = argv[1];
    int         port   = atoi(argv[2]);
    const char *ib_dev = (argc > 3) ? argv[3] : "rxe_enp44s0";

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

    printf("=== dmaplane: Cross-Machine Gradient Send ===\n");
    printf("    Device: %s | IB: %s | Target: %s:%d\n\n",
           DMAPLANE_DEVICE, ib_dev, ip, port);

    /* ── 1. Open kernel module ── */
    printf("[1] Opening %s...\n", DMAPLANE_DEVICE);
    emu_fd = open(DMAPLANE_DEVICE, O_RDWR);
    if (emu_fd < 0) {
        perror("    open");
        fprintf(stderr, "    sudo insmod dmaplane.ko\n");
        return 1;
    }

    /* ── 2. Allocate page-backed DMA buffer via kernel module ──
     *
     *  The module calls alloc_pages() + dma_map_sg() internally.
     *  On a multi-socket server, pages land on the platform device's
     *  NUMA node. On your nucbox (single-node), this is node 0.
     */
    printf("[2] IOCTL_CREATE_BUFFER (%zu MB, page-backed)...\n", BUF_SIZE >> 20);
    buf.alloc_type = BUF_TYPE_PAGES;
    buf.size       = BUF_SIZE;
    buf.numa_node  = DMAPLANE_NUMA_ANY;
    if (ioctl(emu_fd, IOCTL_CREATE_BUFFER, &buf) < 0) {
        perror("    IOCTL_CREATE_BUFFER");
        ret = 1; goto out;
    }
    printf("    buf_id=%u, size=%llu\n", buf.buf_id, (unsigned long long)buf.size);

    /* ── 3. Setup kernel RDMA on the rxe device ──
     *
     *  The kernel module creates its own PD, CQ, and QP loopback pair.
     *  This is NOT the transport for the cross-machine send — that uses
     *  rdma_cm in userspace (step 6). The kernel-side setup establishes:
     *    - The PD whose local_dma_lkey authorizes NIC access to our pages
     *    - DMA mappings via ib_dma_map_sg() during MR registration (step 4)
     *    - Infrastructure for the module's built-in benchmarks
     *  The kernel owns the DMA plumbing; userspace owns the transport.
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
        fprintf(stderr, "    Is soft-RoCE configured?\n");
        fprintf(stderr, "    sudo modprobe rdma_rxe\n");
        fprintf(stderr, "    sudo rdma link add %s type rxe netdev <iface>\n", ib_dev);
        ret = 1; goto out;
    }
    printf("    status=%u\n", setup.status);

    /* ── 4. Register buffer as MR ──
     *
     *  Module calls ib_alloc_mr() + ib_map_mr_sg() on the buffer's
     *  page array. The lkey/rkey are for the kernel loopback QPs.
     */
    printf("[4] IOCTL_REGISTER_MR (buf_id=%u)...\n", buf.buf_id);
    mr.buf_id       = buf.buf_id;
    mr.access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ;

    if (ioctl(emu_fd, IOCTL_REGISTER_MR, &mr) < 0) {
        perror("    IOCTL_REGISTER_MR");
        ret = 1; goto out;
    }
    printf("    mr_id=%u, lkey=0x%x, rkey=0x%x\n", mr.mr_id, mr.lkey, mr.rkey);

    /* ── 5. mmap the kernel buffer into userspace ──
     *
     *  Instead of aligned_alloc + ibv_reg_mr on a separate buffer,
     *  we mmap the kernel module's DMA pages directly. Writes here
     *  go to the same physical pages the module allocated with
     *  alloc_pages() and DMA-mapped with dma_map_sg().
     */
    printf("[5] mmap kernel buffer into userspace...\n");

    struct dmaplane_mmap_info minfo = { .buf_id = buf.buf_id };
    if (ioctl(emu_fd, IOCTL_GET_MMAP_INFO, &minfo) < 0) {
        perror("    IOCTL_GET_MMAP_INFO");
        ret = 1; goto out;
    }
    printf("    mmap_offset=0x%llx, mmap_size=%llu\n",
           (unsigned long long)minfo.mmap_offset,
           (unsigned long long)minfo.mmap_size);

    data = mmap(NULL, minfo.mmap_size, PROT_READ | PROT_WRITE,
                MAP_SHARED, emu_fd, minfo.mmap_offset);
    if (data == MAP_FAILED) {
        perror("    mmap");
        data = NULL;
        ret = 1; goto out;
    }
    data_is_mmap = 1;
    printf("    mmap'd %llu bytes at %p (kernel DMA pages)\n",
           (unsigned long long)minfo.mmap_size, (void *)data);

    fill_gradient(data, NUM_FLOATS, LAYER);
    printf("    Layer %d: [%.6f, %.6f, %.6f, %.6f, ...]\n",
           LAYER, data[0], data[1], data[2], data[3]);

    /* ── 6. Connect to receiver via rdma_cm ── */
    printf("[6] Connecting to receiver...\n");
    if (connect_to_receiver(&conn, ip, port) < 0) {
        ret = 1; goto out;
    }

    /*
     * Dual-MR pattern: the same physical pages now carry two MRs.
     *   - Kernel MR (step 4): ib_dma_map_sg() established IOMMU/DMA
     *     mappings and local_dma_lkey for kernel-side access.
     *   - Userspace MR (here): ibv_reg_mr() on the mmap'd region gives
     *     us the lkey needed for the rdma_cm QP's send work requests.
     * Data written via mmap goes on the wire without any copies.
     */
    conn.mr = ibv_reg_mr(conn.pd, data, BUF_SIZE,
                         IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ);
    if (!conn.mr) { perror("ibv_reg_mr"); ret = 1; goto out; }
    printf("    Userspace MR on mmap'd pages: lkey=0x%x\n", conn.mr->lkey);

    /* Brief pause to ensure receiver has posted its recv.
     * In production, use an rdma_cm private_data handshake or
     * a control message exchange instead. */
    usleep(100000);  /* 100 ms */

    /* ── 7. RDMA Send ── */
    printf("[7] Sending %zu MB via RDMA...\n", BUF_SIZE >> 20);

    struct ibv_sge sge = {
        .addr = (uintptr_t)data, .length = BUF_SIZE, .lkey = conn.mr->lkey,
    };
    struct ibv_send_wr wr = {
        .wr_id = 1, .sg_list = &sge, .num_sge = 1,
        .opcode = IBV_WR_SEND, .send_flags = IBV_SEND_SIGNALED,
    };
    struct ibv_send_wr *bad = NULL;

    if (ibv_post_send(conn.qp, &wr, &bad)) {
        perror("    ibv_post_send"); ret = 1; goto out;
    }

    struct ibv_wc wc;
    int n;
    do { n = ibv_poll_cq(conn.cq, 1, &wc); } while (n == 0);
    if (n < 0 || wc.status != IBV_WC_SUCCESS) {
        fprintf(stderr, "    Send failed: %s\n", ibv_wc_status_str(wc.status));
        ret = 1; goto out;
    }
    printf("    Done. %zu MB sent.\n", BUF_SIZE >> 20);

    /* ── 8. Kernel module stats ── */
    printf("[8] Kernel module RDMA stats:\n");
    struct dmaplane_rdma_stats rst;
    memset(&rst, 0, sizeof(rst));
    if (ioctl(emu_fd, DMAPLANE_IOCTL_GET_RDMA_STATS, &rst) == 0) {
        printf("    mrs_registered:     %llu\n", (unsigned long long)rst.mrs_registered);
        printf("    sends_posted:       %llu\n", (unsigned long long)rst.sends_posted);
        printf("    completions_polled: %llu\n", (unsigned long long)rst.completions_polled);
        printf("    completion_errors:  %llu\n", (unsigned long long)rst.completion_errors);
    }

    printf("\n=== Summary ===\n");
    printf("  Kernel module: buffer alloc + MR reg (lkey=0x%x)\n", mr.lkey);
    printf("  Userspace:     mmap'd kernel pages + rdma_cm send (lkey=0x%x)\n", conn.mr->lkey);
    printf("  Zero-copy:     userspace writes directly to DMA-mapped pages\n");
    printf("  Transport:     %s (soft-RoCE)\n", ib_dev);

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
