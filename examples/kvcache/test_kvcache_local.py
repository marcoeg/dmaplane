#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0
"""
test_kvcache_local.py — End-to-end disaggregated inference test

Validates the full pipeline on a single laptop via rxe loopback:
  1. Prefill with TinyLlama-1.1B-Chat
  2. Extract + consolidate KVCache to GPU staging buffer
  3. WRITEIMM loopback (GPU MR → host MR) per chunk
  4. Reconstruct KVCache from host landing buffer
  5. Decode tokens and compare against reference model.generate()

Requires: dmaplane.ko loaded, rxe configured, CUDA GPU, TinyLlama model.
Run as root: sudo python3 test_kvcache_local.py

Note: This tests GPU→host transfer only. GPU→GPU (both sides have GPUs)
is deferred to Phase 9C (EC2 multi-GPU deployment).
"""

import sys
import os
import time
import gc

# Add parent dir for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from dmaplane_py import (
    DmaplaneClient, allocate_aligned_gpu_tensor,
    kvcache_imm_encode, kvcache_imm_layer, kvcache_imm_chunk,
    KVCACHE_SENTINEL, IB_ACCESS_LOCAL_WRITE, IB_ACCESS_REMOTE_WRITE,
)
from kvcache_inference import (
    extract_kvcache, consolidate_kvcache, reconstruct_kvcache_zero_copy,
    kvcache_size_estimate, kvcache_size_from_layers,
)

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TEST_PROMPT = "The capital of France is"
CHUNK_SIZE = 1024 * 1024  # 1 MB

passed = 0
failed = 0
skipped = 0


def test(name):
    def decorator(fn):
        fn._test_name = name
        return fn
    return decorator


def run_test(fn):
    global passed, failed
    name = getattr(fn, '_test_name', fn.__name__)
    try:
        fn()
        print(f"  PASS: {name}")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {name} — {e}")
        import traceback
        traceback.print_exc()
        failed += 1


# ── Shared model + tokenizer (loaded once) ────────────────────────────────

_model = None
_tokenizer = None


def get_model_and_tokenizer():
    global _model, _tokenizer
    if _model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"  Loading {MODEL_NAME}...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, device_map="cuda"
        )
        _model.eval()
    return _model, _tokenizer


def prefill(prompt):
    """Run prefill phase — the model processes the full prompt in one forward
    pass and produces the KVCache (past_key_values) and logits."""
    model, tokenizer = get_model_and_tokenizer()
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model(**inputs, use_cache=True)
    return output, inputs["input_ids"]


def generate_reference(prompt, max_new_tokens=20):
    """Generate reference tokens via model.generate (greedy, deterministic).
    Used as ground truth to verify the disaggregated decode path."""
    model, tokenizer = get_model_and_tokenizer()
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, use_cache=True
        )
    return output_ids


# ── Test 1: Extract and consolidate ───────────────────────────────────────

# ── Test 1: Verify that extract + consolidate produces a correct, contiguous
#    byte packing of all KVCache layers with no gaps in the manifest offsets.

@test("KVCache extract + consolidate")
def test_extract_and_consolidate():
    output, input_ids = prefill(TEST_PROMPT)
    layers = extract_kvcache(output)

    assert len(layers) > 0, "No layers extracted"
    assert len(layers) == _model.config.num_hidden_layers, \
        f"Expected {_model.config.num_hidden_layers} layers, got {len(layers)}"

    # Check shapes
    k0, v0 = layers[0]
    assert k0.dim() == 4, f"Expected 4D tensor, got {k0.dim()}D"
    batch, heads, seq_len, head_dim = k0.shape
    assert batch == 1
    print(f"    Shape: [{batch}, {heads}, {seq_len}, {head_dim}]")

    # Consolidate
    total_bytes = kvcache_size_from_layers(layers)
    staging = torch.empty(total_bytes, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()

    packed_bytes, manifest = consolidate_kvcache(layers, staging)
    assert packed_bytes == total_bytes, \
        f"Size mismatch: packed {packed_bytes} != expected {total_bytes}"
    assert len(manifest) == len(layers)

    # Verify manifest offsets are contiguous
    for i, entry in enumerate(manifest):
        assert entry["layer"] == i
        if i > 0:
            prev = manifest[i - 1]
            expected_offset = prev["v_offset"] + prev["v_size"]
            assert entry["k_offset"] == expected_offset, \
                f"Layer {i}: gap in manifest offsets"

    print(f"    Total: {total_bytes / 1024:.1f} KB across {len(layers)} layers")


# ── Test 2: as_strided reconstruction ─────────────────────────────────────

# ── Test 2: Verify that as_strided views into the staging buffer produce
#    tensors bitwise-identical to the originals — confirming zero-copy works.

@test("as_strided zero-copy reconstruction")
def test_as_strided_reconstruction():
    output, _ = prefill(TEST_PROMPT)
    layers = extract_kvcache(output)

    total_bytes = kvcache_size_from_layers(layers)
    staging = torch.empty(total_bytes, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()

    _, manifest = consolidate_kvcache(layers, staging)

    # Reconstruct from the same staging buffer
    reconstructed = reconstruct_kvcache_zero_copy(staging, manifest)

    assert len(reconstructed.layers) == len(layers)
    for i, (orig_k, orig_v) in enumerate(layers):
        layer = reconstructed.layers[i]
        recon_k = layer.keys
        recon_v = layer.values
        assert torch.equal(orig_k, recon_k), \
            f"Layer {i} K mismatch"
        assert torch.equal(orig_v, recon_v), \
            f"Layer {i} V mismatch"

    print(f"    All {len(layers)} layers verified identical")


# ── Test 3: Decode with reconstructed KVCache ─────────────────────────────

# ── Test 3: The core correctness test — does the disaggregated path produce
#    the exact same tokens as model.generate()?  This validates the full chain:
#    prefill → extract → consolidate → reconstruct → greedy decode loop.
#    Uses do_sample=False for deterministic comparison.

@test("Decode with reconstructed KVCache matches reference")
def test_decode_with_reconstructed():
    max_new_tokens = 20
    model, tokenizer = get_model_and_tokenizer()

    # Reference: model.generate (greedy, deterministic)
    ref_ids = generate_reference(TEST_PROMPT, max_new_tokens)
    ref_text = tokenizer.decode(ref_ids[0], skip_special_tokens=True)

    # Disaggregated path: prefill → extract → consolidate → reconstruct → decode
    output, input_ids = prefill(TEST_PROMPT)
    layers = extract_kvcache(output)

    total_bytes = kvcache_size_from_layers(layers)
    staging = torch.empty(total_bytes, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    _, manifest = consolidate_kvcache(layers, staging)
    reconstructed = reconstruct_kvcache_zero_copy(staging, manifest)

    # Manual greedy decode loop — simulates what the "decode" node does in
    # a disaggregated inference system.  It receives the reconstructed KVCache
    # (from the "prefill" node) and autoregressively generates tokens.
    next_token_logits = output.logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    generated_ids = input_ids.clone()
    past_kv = reconstructed

    for step in range(max_new_tokens):
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        with torch.no_grad():
            out = model(next_token, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        next_token_logits = out.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

    disagg_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    assert disagg_text == ref_text, \
        f"Token mismatch:\n  ref:   {ref_text!r}\n  disagg: {disagg_text!r}"
    print(f"    Reference:     {ref_text!r}")
    print(f"    Disaggregated: {disagg_text!r}")


# ── Test 4: WRITEIMM loopback with real KVCache ──────────────────────────

# ── Test 4: The most important test — validates the full RDMA transfer path.
#    GPU MR (prefill KVCache) → WRITEIMM loopback → host MR → mmap → reconstruct
#    → decode → verify tokens.  This is what disaggregated inference actually does:
#    prefill node has GPU memory, decode node receives via RDMA.

@test("WRITEIMM loopback GPU→host with real KVCache")
def test_writeimm_loopback_real_kvcache():
    """
    Full pipeline: prefill → consolidate → GPU pin → WRITEIMM chunks →
    poll recv → reconstruct from host → decode → verify tokens match.

    Tests GPU→host RDMA path only. GPU→GPU deferred to Phase 9C.
    """
    max_new_tokens = 20
    model, tokenizer = get_model_and_tokenizer()

    # Reference
    ref_ids = generate_reference(TEST_PROMPT, max_new_tokens)
    ref_text = tokenizer.decode(ref_ids[0], skip_special_tokens=True)

    # Prefill + extract
    output, input_ids = prefill(TEST_PROMPT)
    layers = extract_kvcache(output)
    total_bytes = kvcache_size_from_layers(layers)

    # Allocate aligned GPU staging buffer + consolidate
    staging = allocate_aligned_gpu_tensor(total_bytes)
    torch.cuda.synchronize()
    packed_bytes, manifest = consolidate_kvcache(layers, staging)
    torch.cuda.synchronize()

    # Chunk math
    num_chunks = (packed_bytes + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"    KVCache: {packed_bytes} bytes, {num_chunks} chunks of {CHUNK_SIZE // 1024} KB")

    with DmaplaneClient() as cl:
        cl.setup_rdma(max_send_wr=num_chunks + 16, max_recv_wr=num_chunks + 16)
        try:
            # GPU source: pin + register MR
            gpu_va = staging.data_ptr()
            aligned_size = ((packed_bytes + 65535) // 65536) * 65536
            pin = cl.gpu_pin(gpu_va, max(aligned_size, staging.numel()))
            gpu_mr = cl.gpu_register_mr(pin["handle"])

            # Host destination: create buffer + register fast-reg MR
            dst_size = max(aligned_size, packed_bytes)
            dst = cl.create_buffer(dst_size)
            dst_mr = cl.register_mr(dst["buf_id"],
                                    IB_ACCESS_LOCAL_WRITE |
                                    IB_ACCESS_REMOTE_WRITE)

            # Pre-post all recvs — for this test we don't use credit windowing,
            # we just post all recvs upfront (num_chunks + 1 for sentinel)
            for i in range(num_chunks + 1):  # +1 for sentinel
                cl.post_recv(dst_mr["mr_id"], CHUNK_SIZE, use_peer_qp=0)

            # Send chunks via WRITEIMM
            t0 = time.monotonic()
            for i in range(num_chunks):
                offset = i * CHUNK_SIZE
                length = min(CHUNK_SIZE, packed_bytes - offset)
                imm = kvcache_imm_encode(0, i)  # single "layer" for now
                cl.write_imm(gpu_mr["mr_id"], offset,
                             dst_mr["addr"] + offset, dst_mr["rkey"],
                             length, imm, use_peer_qp=0)

            # Send sentinel
            cl.write_imm(gpu_mr["mr_id"], 0,
                         dst_mr["addr"], dst_mr["rkey"],
                         4, KVCACHE_SENTINEL, use_peer_qp=0)

            # Poll all completions
            for i in range(num_chunks + 1):
                result = cl.poll_recv(use_peer_qp=0, timeout_ms=5000)
                assert result["status"] == 0, \
                    f"poll_recv #{i} failed: status={result['status']}"

            elapsed = time.monotonic() - t0
            throughput = (packed_bytes / 1024 / 1024) / elapsed
            print(f"    RDMA transfer: {elapsed * 1000:.1f}ms, "
                  f"{throughput:.1f} MB/s")

            # Reconstruct from host landing buffer via mmap.
            # The data path: GPU VRAM → RDMA WRITE → host MR → mmap → Python
            # bytes → CUDA tensor → as_strided views.  In production the decode
            # node would keep the MR on GPU directly (Phase 9C GPU→GPU path).
            dst_map = cl.mmap_buffer(dst["buf_id"])
            host_data = bytes(dst_map[:packed_bytes])
            host_tensor = torch.frombuffer(
                bytearray(host_data), dtype=torch.uint8
            ).cuda()

            reconstructed = reconstruct_kvcache_zero_copy(
                host_tensor, manifest)

            # Decode with reconstructed KVCache
            next_token_logits = output.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1,
                                      keepdim=True)
            generated_ids = input_ids.clone()
            past_kv = reconstructed

            for step in range(max_new_tokens):
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                with torch.no_grad():
                    out = model(next_token, past_key_values=past_kv,
                                use_cache=True)
                past_kv = out.past_key_values
                next_token_logits = out.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1,
                                          keepdim=True)

            disagg_text = tokenizer.decode(generated_ids[0],
                                           skip_special_tokens=True)
            assert disagg_text == ref_text, \
                f"Token mismatch after RDMA:\n" \
                f"  ref:   {ref_text!r}\n  disagg: {disagg_text!r}"
            print(f"    Reference:     {ref_text!r}")
            print(f"    RDMA disagg:   {disagg_text!r}")

            # Cleanup — order matters: mmap → MR → buffer
            cl._munmap(dst["buf_id"])
            cl.gpu_unpin(pin["handle"])
            cl.deregister_mr(gpu_mr["mr_id"])
            cl.deregister_mr(dst_mr["mr_id"])
            cl.destroy_buffer(dst["buf_id"])
        finally:
            cl.teardown_rdma()


# ── Test 5: Alignment and pin ─────────────────────────────────────────────

@test("GPU tensor alignment at KVCache size + pin")
def test_alignment_and_pin():
    model, _ = get_model_and_tokenizer()
    seq_len = 32  # reasonable test prompt length
    est = kvcache_size_estimate(model.config, seq_len)
    print(f"    Estimated KVCache: {est / 1024:.1f} KB for seq_len={seq_len}")

    tensor = allocate_aligned_gpu_tensor(est)
    assert tensor.data_ptr() % 65536 == 0

    with DmaplaneClient() as cl:
        cl.setup_rdma()
        try:
            pin = cl.gpu_pin(tensor.data_ptr(), tensor.numel())
            assert pin["handle"] > 0
            print(f"    Pinned: handle={pin['handle']}, "
                  f"pages={pin['num_pages']}")
            cl.gpu_unpin(pin["handle"])
        finally:
            cl.teardown_rdma()

    del tensor


# ── Test 6: Multiple prompts ─────────────────────────────────────────────

# ── Test 6: Verify that pre-pinned buffers can be reused across multiple
#    prompts without re-allocation — important for production inference servers
#    that handle many requests with the same buffer pool.

@test("Multiple prompts reuse pinned buffers")
def test_multiple_prompts():
    model, tokenizer = get_model_and_tokenizer()
    prompts = [
        "The meaning of life is",
        "In machine learning, transformers are",
        "Python is a programming language that",
    ]

    # Pre-allocate a large enough staging buffer for all prompts.
    # In production, the inference server would size this once based on
    # max_seq_len and reuse it across requests.
    max_size = kvcache_size_estimate(model.config, 64)
    staging = allocate_aligned_gpu_tensor(max_size)

    for prompt in prompts:
        output, input_ids = prefill(prompt)
        layers = extract_kvcache(output)
        total_bytes = kvcache_size_from_layers(layers)
        assert total_bytes <= max_size, \
            f"KVCache {total_bytes} > staging {max_size}"

        torch.cuda.synchronize()
        _, manifest = consolidate_kvcache(layers, staging)
        reconstructed = reconstruct_kvcache_zero_copy(staging, manifest)

        # Quick decode — just verify it produces tokens without errors
        next_logits = output.logits[:, -1, :]
        next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        with torch.no_grad():
            out = model(next_token, past_key_values=reconstructed,
                        use_cache=True)

        decoded = tokenizer.decode(next_token[0], skip_special_tokens=True)
        print(f"    \"{prompt}\" → \"{decoded.strip()}\"")

    del staging


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("test_kvcache_local.py — End-to-end disaggregated inference test\n")

    # Check prerequisites
    if not torch.cuda.is_available():
        print("  SKIP: CUDA not available")
        sys.exit(1)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("  SKIP: transformers not installed (pip install transformers)")
        sys.exit(1)

    if not os.path.exists("/dev/dmaplane"):
        print("  SKIP: /dev/dmaplane not found — load dmaplane.ko first")
        sys.exit(1)

    # Run tests
    run_test(test_extract_and_consolidate)
    run_test(test_as_strided_reconstruction)
    run_test(test_decode_with_reconstructed)

    # RDMA tests need dmaplane.ko + rxe
    from dmaplane_py import find_rxe_device
    if find_rxe_device() is None:
        print("\n  SKIP RDMA tests: no rxe device found")
    else:
        run_test(test_writeimm_loopback_real_kvcache)
        run_test(test_alignment_and_pin)

    run_test(test_multiple_prompts)

    print(f"\nResults: {passed} passed, {failed} failed")
    sys.exit(1 if failed > 0 else 0)
