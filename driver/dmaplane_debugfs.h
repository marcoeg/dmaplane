/* SPDX-License-Identifier: GPL-2.0 */
/*
 * dmaplane_debugfs.h — debugfs interface declarations
 * Copyright (c) 2026 Graziano Labs Corp.
 *
 * Creates /sys/kernel/debug/dmaplane/ with live visibility into
 * all driver state: stats, buffers, RDMA context, flow control,
 * and latency histogram.
 *
 * debugfs failure is never fatal — the module operates normally
 * without it.  All debugfs errors are silently ignored.
 */
#ifndef _DMAPLANE_DEBUGFS_H
#define _DMAPLANE_DEBUGFS_H

struct dmaplane_dev;

int dmaplane_debugfs_init(struct dmaplane_dev *dev);
void dmaplane_debugfs_exit(void);

#endif /* _DMAPLANE_DEBUGFS_H */
