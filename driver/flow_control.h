/* SPDX-License-Identifier: GPL-2.0 */
/*
 * flow_control.h — Credit-based flow control API
 * Copyright (c) 2026 Graziano Labs Corp.
 *
 * Phase 6 adds credit-based backpressure to the RDMA send path,
 * sustained streaming benchmarks, and queue depth sweeps.
 *
 * The credit system is a policy layer on top of rdma_engine — it
 * tracks how many operations are in-flight and pauses the sender
 * when the count reaches the high watermark.  rdma_engine itself
 * is unchanged; it provides the verbs, flow_control provides the
 * pacing.
 *
 * Callers must hold rdma_sem read lock for sustained_stream and
 * qdepth_sweep (same as existing benchmarks).  configure and
 * can_send/on_send/on_completion are lockless.
 */
#ifndef _DMAPLANE_FLOW_CONTROL_H
#define _DMAPLANE_FLOW_CONTROL_H

#include "dmaplane.h"

/* Configuration */
int dmaplane_flow_configure(struct dmaplane_dev *dev,
			    struct dmaplane_flow_params *params);

/* Credit operations */
bool dmaplane_flow_can_send(struct dmaplane_dev *dev);
void dmaplane_flow_on_send(struct dmaplane_dev *dev);
void dmaplane_flow_on_completion(struct dmaplane_dev *dev);

/* Benchmarks */
int dmaplane_sustained_stream(struct dmaplane_dev *dev,
			      struct dmaplane_sustained_params *params);
int dmaplane_qdepth_sweep(struct dmaplane_dev *dev,
			  struct dmaplane_sweep_params *params);

#endif /* _DMAPLANE_FLOW_CONTROL_H */
