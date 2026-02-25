#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0
"""
kvcache_inference.py — KVCache extract / consolidate / reconstruct

Shared module for disaggregated inference:
  - extract_kvcache(): pull (K, V) pairs from model output
  - consolidate_kvcache(): pack into contiguous staging buffer
  - reconstruct_kvcache_zero_copy(): as_strided views into landing buffer
  - kvcache_size_estimate(): compute total bytes needed

All designed for zero-copy RDMA transfer: consolidate packs contiguously,
reconstruct creates views without copying.
"""

import torch
from transformers import DynamicCache

def extract_kvcache(model_output):
    """
    Extract KVCache from a HuggingFace model output.

    HuggingFace models with use_cache=True return past_key_values as a tuple
    of (K, V) pairs, one per transformer layer.  The tensors may not be
    contiguous in memory (e.g., after attention head reshaping), so we call
    .contiguous() on each to ensure a flat memory layout suitable for
    byte-level packing in consolidate_kvcache().

    Args:
        model_output: output from model(..., use_cache=True)

    Returns:
        list of (key_tensor, value_tensor) tuples, one per layer.
        Each tensor is contiguous with shape [batch, num_kv_heads, seq_len, head_dim].
    """
    past_kv = model_output.past_key_values
    layers = []
    for layer_kv in past_kv:
        k = layer_kv[0].contiguous()
        v = layer_kv[1].contiguous()
        layers.append((k, v))
    return layers


def consolidate_kvcache(layers, staging_buf):
    """
    Pack KVCache layers into a contiguous staging buffer for RDMA transfer.

    Uses copy_() to write each K/V tensor's flat bytes into the staging buffer
    at computed offsets.  No torch.cat — avoids extra GPU allocations that
    would double peak VRAM usage.  The staging buffer is pre-allocated by
    the caller (typically via allocate_aligned_gpu_tensor for RDMA pinning).

    Args:
        layers: list of (K, V) tuples from extract_kvcache()
        staging_buf: pre-allocated contiguous GPU tensor (torch.uint8)

    Returns:
        (total_bytes, manifest) where manifest is a list of dicts:
        [{"layer": i, "k_offset": ..., "k_size": ..., "v_offset": ..., "v_size": ...,
          "shape": (batch, heads, seq, dim), "dtype": torch.float16}, ...]
    """
    assert staging_buf.is_cuda, "staging buffer must be on GPU"
    assert staging_buf.is_contiguous(), "staging buffer must be contiguous"
    # assert staging_buf.data_ptr() % 65536 == 0, "staging buffer must be 64KB-aligned"
  
    offset = 0
    manifest = []

    for i, (k, v) in enumerate(layers):
        k_bytes = k.nelement() * k.element_size()
        v_bytes = v.nelement() * v.element_size()

        # Reinterpret as flat uint8 bytes — view(-1) flattens all dimensions,
        # view(torch.uint8) reinterprets the dtype without copying.
        # This gives us a byte-level view for copy_() into the staging buffer.
        k_flat = k.view(-1).view(torch.uint8)
        v_flat = v.view(-1).view(torch.uint8)

        staging_buf[offset:offset + k_bytes].copy_(k_flat)
        k_offset = offset
        offset += k_bytes

        staging_buf[offset:offset + v_bytes].copy_(v_flat)
        v_offset = offset
        offset += v_bytes

        manifest.append({
            "layer": i,
            "k_offset": k_offset,
            "k_size": k_bytes,
            "v_offset": v_offset,
            "v_size": v_bytes,
            "shape": tuple(k.shape),
            "dtype": k.dtype,
        })

    return offset, manifest


def reconstruct_kvcache_zero_copy(recv_buf, manifest, device=None):
    """
    Reconstruct KVCache as zero-copy views into the receive buffer.

    Uses torch.as_strided to create K/V tensor views at manifest offsets,
    without any memory copies.  The receive buffer is the RDMA landing zone
    — after WRITEIMM transfers, it contains the packed bytes from
    consolidate_kvcache() and we simply reinterpret them as typed tensors.

    This is the key zero-copy step: instead of unpacking + copying into
    separate tensors, we create views that share the landing buffer's
    memory.  The model reads K/V directly from the RDMA buffer.

    Args:
        recv_buf: contiguous tensor (torch.uint8) containing transferred data
        manifest: list of dicts from consolidate_kvcache()
        device: target device (None = same as recv_buf)

    Returns:
        tuple of (K, V) pairs, compatible with model's past_key_values format.
    """
    if device is None:
        device = recv_buf.device

    layers = []
    for entry in manifest:
        shape = entry["shape"]
        dtype = entry["dtype"]
        elem_size = torch.tensor([], dtype=dtype).element_size()

        # Compute strides for contiguous [batch, heads, seq, dim] layout.
        # These are element strides (not byte strides) — as_strided expects
        # them in units of the tensor's dtype elements.
        # For a 4D contiguous layout: stride[i] = product(shape[i+1:])
        batch, heads, seq_len, head_dim = shape
        strides = (
            heads * seq_len * head_dim,   # batch stride
            seq_len * head_dim,           # head stride
            head_dim,                     # sequence position stride
            1,                            # element stride (innermost)
        )

        # K tensor: view recv_buf as dtype at k_offset
        k_offset = entry["k_offset"]
        k_base = recv_buf[k_offset:k_offset + entry["k_size"]]
        k_typed = k_base.view(dtype)
        k = torch.as_strided(k_typed, shape, strides)

        # V tensor: view recv_buf as dtype at v_offset
        v_offset = entry["v_offset"]
        v_base = recv_buf[v_offset:v_offset + entry["v_size"]]
        v_typed = v_base.view(dtype)
        v = torch.as_strided(v_typed, shape, strides)

        layers.append((k, v))

    cache = DynamicCache()
    for k, v in layers:
        cache.update(k, v, len(cache))
    return cache

def kvcache_size_estimate(model_config, seq_len, dtype=torch.float16):
    """
    Estimate total KVCache size in bytes from a model config.

    Handles Grouped Query Attention (GQA) via num_key_value_heads — in GQA
    models (e.g., Llama 2 70B), the KV heads are fewer than attention heads,
    which significantly reduces KVCache size.  Falls back to num_attention_heads
    for standard MHA models.

    Args:
        model_config: HuggingFace model config object
        seq_len: sequence length
        dtype: data type (default: float16)

    Returns:
        total bytes needed for all layers (K + V)
    """
    num_layers = model_config.num_hidden_layers
    # GQA: use num_key_value_heads if available, else num_attention_heads
    num_kv_heads = getattr(model_config, "num_key_value_heads",
                           model_config.num_attention_heads)
    head_dim = model_config.hidden_size // model_config.num_attention_heads
    elem_size = torch.tensor([], dtype=dtype).element_size()

    # Per layer: K + V, each is [1, num_kv_heads, seq_len, head_dim]
    per_layer = 2 * 1 * num_kv_heads * seq_len * head_dim * elem_size
    return num_layers * per_layer


def kvcache_size_from_layers(layers):
    """Compute total bytes from extracted KVCache layers."""
    total = 0
    for k, v in layers:
        total += k.nelement() * k.element_size()
        total += v.nelement() * v.element_size()
    return total
