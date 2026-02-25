/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (c) 2026 Graziano Labs Corp.
 *
 * kvcache_proto.h — KVCache chunk tracking protocol
 *
 * Encodes (layer_index, chunk_index) into the 32-bit immediate field
 * of RDMA WRITE WITH IMMEDIATE.  The receiver decodes from the CQ
 * completion to track which chunk arrived — no memory polling needed.
 *
 * Layout: upper 16 bits = layer, lower 16 bits = chunk.
 * Supports 65535 layers x 65535 chunks per layer.
 * A 7B model has ~32 layers; KVCache per layer is ~4-16 MB.
 * At 1 MB chunks: ~4-16 chunks per layer = well within range.
 */
#ifndef _KVCACHE_PROTO_H
#define _KVCACHE_PROTO_H

#include <stdint.h>

/* Encode (layer, chunk) into a 32-bit immediate value */
#define KVCACHE_IMM_ENCODE(layer, chunk) \
	((((uint32_t)(layer) & 0xFFFF) << 16) | ((uint32_t)(chunk) & 0xFFFF))

/* Extract layer index from 32-bit immediate */
#define KVCACHE_IMM_LAYER(imm)	(((uint32_t)(imm) >> 16) & 0xFFFF)

/* Extract chunk index from 32-bit immediate */
#define KVCACHE_IMM_CHUNK(imm)	((uint32_t)(imm) & 0xFFFF)

/* Sentinel: signals end-of-transfer */
#define KVCACHE_SENTINEL	0xFFFFFFFF

#endif /* _KVCACHE_PROTO_H */
