/* SPDX-License-Identifier: GPL-2.0 */
/*
 * dmabuf_export.h — dma-buf exporter interface for dmaplane buffers
 * Copyright (c) 2026 Graziano Labs Corp.
 *
 * Public interface for exporting page-backed dmaplane buffers as
 * dma-buf objects.  This enables cross-device zero-copy sharing:
 * other kernel devices (NICs, GPUs, encoders) can attach and get
 * per-device scatter-gather tables through the standard dma_buf_ops
 * interface.
 */
#ifndef _DMABUF_EXPORT_H
#define _DMABUF_EXPORT_H

#include "dmaplane.h"

/*
 * dmaplane_dmabuf_export() — Export a buffer as a dma-buf.
 *
 * Wraps a page-backed buffer in a struct dma_buf and returns a file
 * descriptor to userspace.  Coherent buffers cannot be exported
 * (they have no page array for SG table construction).
 *
 * Locking: acquires buf_lock for the entire operation (including
 * dma_buf_export and dma_buf_fd calls).  buf_lock is a mutex, so
 * sleeping is permitted.  The coarse granularity prevents a race
 * where two threads export the same buffer simultaneously.
 *
 * Return: 0 on success (fd set in arg->fd), negative errno on failure.
 *   -EINVAL: buffer not found, or buffer is coherent (not page-backed)
 *   -EBUSY:  buffer already exported (one dma-buf per buffer)
 *   -ENOMEM: allocation failure
 */
int dmaplane_dmabuf_export(struct dmaplane_dev *dev,
			   struct dmaplane_export_dmabuf_arg *arg);

/*
 * dmaplane_dmabuf_selftest() — Kernel-space self-test for dma-buf export.
 *
 * Creates a buffer, exports it, exercises attach/map/unmap/detach
 * through the kernel dma-buf API, verifies SG table correctness,
 * and cleans up.  Results printed to dmesg.
 *
 * Called from module init when test_dmabuf=1 module parameter is set.
 *
 * Return: 0 on success, negative errno on failure.
 */
int dmaplane_dmabuf_selftest(struct dmaplane_dev *dev);

#endif /* _DMABUF_EXPORT_H */
