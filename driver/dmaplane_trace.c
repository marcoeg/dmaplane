// SPDX-License-Identifier: GPL-2.0
/*
 * dmaplane_trace.c — Tracepoint compilation unit
 * Copyright (c) 2026 Graziano Labs Corp.
 *
 * This is the single .c file that defines CREATE_TRACE_POINTS.
 * It causes the compiler to emit the actual tracepoint definitions
 * and registration code.  All other .c files include dmaplane_trace.h
 * WITHOUT CREATE_TRACE_POINTS to get the trace_dmaplane_*() call stubs.
 *
 * Build constraint: exactly one compilation unit per module may define
 * CREATE_TRACE_POINTS.  Defining it in two files causes duplicate
 * symbol errors at link time.
 */

#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt

#include <linux/module.h>

#define CREATE_TRACE_POINTS
#include "dmaplane_trace.h"
