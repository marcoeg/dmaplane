/* SPDX-License-Identifier: GPL-2.0 */
/*
 * dmabuf_rdma.h — Buffer management API declarations
 * Copyright (c) 2026 Graziano Labs Corp.
 *
 * Public interface for the buffer allocation subsystem.  Called from
 * main.c (ioctl handlers) and will be called from rdma_engine.c in
 * Phase 4.  All functions operate on the device-global buffer array
 * protected by buf_lock.
 */
#ifndef _DMABUF_RDMA_H
#define _DMABUF_RDMA_H

#include "dmaplane.h"

/*
 * dmabuf_rdma_create_buffer() — Allocate a DMA buffer (coherent or page-backed).
 *
 * Sets params->buf_id on success.  Caller must NOT hold buf_lock
 * (acquired internally).
 *
 * Return: 0 on success, negative errno on failure.
 *   -EINVAL: invalid size (0 or exceeds max), or bad alloc_type
 *   -ENOMEM: no free buffer slots, or page/coherent allocation failed
 */
int dmabuf_rdma_create_buffer(struct dmaplane_dev *dev,
			      struct dmaplane_buf_params *params);

/*
 * dmabuf_rdma_destroy_buffer() — Free a buffer and its backing memory.
 *
 * Refuses with -EBUSY if userspace has active mmaps on the buffer.
 * Caller must NOT hold buf_lock (acquired internally).
 *
 * Return: 0 on success, negative errno on failure.
 *   -ENOENT: buffer ID not found
 *   -EBUSY:  active mmap references prevent destruction
 */
int dmabuf_rdma_destroy_buffer(struct dmaplane_dev *dev, unsigned int buf_id);

/*
 * dmabuf_rdma_find_buffer() — Look up a buffer by ID.
 *
 * Returns pointer into dev->buffers[].  Caller MUST hold buf_lock for
 * the lifetime of the returned pointer — the buffer could be destroyed
 * by another thread between find and use.
 *
 * Return: pointer to buffer entry, or NULL if not found.
 */
struct dmaplane_buffer *dmabuf_rdma_find_buffer(struct dmaplane_dev *dev,
						unsigned int buf_id);

/*
 * dmabuf_rdma_find_mr() — Look up an MR by ID.
 *
 * Returns pointer into dev->mrs[].  Caller MUST hold mr_lock for
 * the lifetime of the returned pointer — the MR could be deregistered
 * by another thread between find and use.
 *
 * Return: pointer to MR entry, or NULL if not found.
 */
struct dmaplane_mr_entry *dmabuf_rdma_find_mr(struct dmaplane_dev *dev,
					      __u32 mr_id);

#endif /* _DMABUF_RDMA_H */
