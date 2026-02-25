#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0
"""
test_dmaplane_py.py — Unit tests for the Python dmaplane wrapper

Requires: dmaplane.ko loaded, rxe configured, run as root.

Usage:
    sudo python3 test_dmaplane_py.py
"""

import struct
import sys
import os
import gc

# Add parent dir so we can import dmaplane_py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dmaplane_py import (
    DmaplaneClient, find_rxe_device,
    FMT_BUF_PARAMS, FMT_RDMA_SETUP, FMT_MR_PARAMS, FMT_MMAP_INFO,
    FMT_GPU_PIN, FMT_GPU_UNPIN, FMT_GPU_MR, FMT_WRITE_IMM,
    FMT_POST_RECV, FMT_POLL_RECV, FMT_REMOTE_OP,
    IB_ACCESS_LOCAL_WRITE, IB_ACCESS_REMOTE_WRITE,
    kvcache_imm_encode, kvcache_imm_layer, kvcache_imm_chunk,
    KVCACHE_SENTINEL, allocate_aligned_gpu_tensor,
)


passed = 0
failed = 0


def test(name):
    """Decorator for test functions."""
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
        failed += 1


# ── Test 1: Struct packing validation ─────────────────────────────────────

@test("Struct sizes match kernel UAPI")
def test_struct_sizes():
    checks = [
        ("buf_params",       FMT_BUF_PARAMS, 24),
        ("rdma_setup",       FMT_RDMA_SETUP, 52),
        ("mr_params",        FMT_MR_PARAMS,  32),
        ("mmap_info",        FMT_MMAP_INFO,  24),
        ("gpu_pin_params",   FMT_GPU_PIN,    40),
        ("gpu_unpin_params", FMT_GPU_UNPIN,   4),
        ("gpu_mr_params",    FMT_GPU_MR,     16),
        ("write_imm_params", FMT_WRITE_IMM,  48),
        ("post_recv_params", FMT_POST_RECV,  16),
        ("poll_recv_params", FMT_POLL_RECV,  32),
        ("remote_op_params", FMT_REMOTE_OP,  32),
    ]
    for name, fmt, expected in checks:
        actual = struct.calcsize(fmt)
        assert actual == expected, \
            f"{name}: got {actual}, expected {expected}"


# ── Test 2: Create buffer + register MR (local access) ───────────────────

@test("Create buffer + register MR (local write)")
def test_create_buffer_local_mr():
    with DmaplaneClient() as cl:
        cl.setup_rdma()
        try:
            result = cl.create_buffer(4096)
            buf_id = result["buf_id"]
            assert buf_id > 0, f"bad buf_id: {buf_id}"

            mr = cl.register_mr(buf_id, IB_ACCESS_LOCAL_WRITE)
            assert mr["mr_id"] > 0, f"bad mr_id: {mr['mr_id']}"
            assert mr["lkey"] != 0, "lkey should be nonzero"

            cl.deregister_mr(mr["mr_id"])
            cl.destroy_buffer(buf_id)
        finally:
            cl.teardown_rdma()


# ── Test 3: Create buffer + register MR (REMOTE_WRITE) ───────────────────

@test("Create buffer + register MR (remote write, fast-reg)")
def test_create_buffer_remote_mr():
    with DmaplaneClient() as cl:
        cl.setup_rdma()
        try:
            result = cl.create_buffer(1024 * 1024)
            buf_id = result["buf_id"]

            mr = cl.register_mr(buf_id,
                                IB_ACCESS_LOCAL_WRITE | IB_ACCESS_REMOTE_WRITE)
            assert mr["rkey"] != 0, "fast-reg MR should have nonzero rkey"
            assert mr["addr"] != 0, "fast-reg MR should have nonzero addr"

            cl.deregister_mr(mr["mr_id"])
            cl.destroy_buffer(buf_id)
        finally:
            cl.teardown_rdma()


# ── Test 4: WRITEIMM round-trip ──────────────────────────────────────────

@test("WRITEIMM round-trip (loopback)")
def test_writeimm_roundtrip():
    with DmaplaneClient() as cl:
        cl.setup_rdma(max_send_wr=32, max_recv_wr=32)
        try:
            # Source MR (local write)
            src = cl.create_buffer(4096)
            src_mr = cl.register_mr(src["buf_id"], IB_ACCESS_LOCAL_WRITE)

            # Destination MR (remote write, fast-reg)
            dst = cl.create_buffer(4096)
            dst_mr = cl.register_mr(dst["buf_id"],
                                    IB_ACCESS_LOCAL_WRITE |
                                    IB_ACCESS_REMOTE_WRITE)

            # Fill source via mmap
            src_map = cl.mmap_buffer(src["buf_id"])
            src_map[:4] = b"\xDE\xAD\xBE\xEF"

            # Post recv, write_imm, poll_recv
            test_imm = kvcache_imm_encode(3, 7)
            cl.post_recv(dst_mr["mr_id"], 4096, use_peer_qp=0)
            cl.write_imm(src_mr["mr_id"], 0,
                         dst_mr["addr"], dst_mr["rkey"],
                         4096, test_imm, use_peer_qp=0)
            result = cl.poll_recv(use_peer_qp=0, timeout_ms=5000)

            assert result["status"] == 0, \
                f"poll_recv status: {result['status']}"
            assert result["imm_data"] == test_imm, \
                f"imm_data: 0x{result['imm_data']:08x} != 0x{test_imm:08x}"

            # Verify data landed
            dst_map = cl.mmap_buffer(dst["buf_id"])
            assert dst_map[:4] == b"\xDE\xAD\xBE\xEF", \
                f"data mismatch: {dst_map[:4].hex()}"

            # Verify IMM decoding
            assert kvcache_imm_layer(result["imm_data"]) == 3
            assert kvcache_imm_chunk(result["imm_data"]) == 7

            # Cleanup: unmap → deregister MR → destroy buffer → teardown
            cl._munmap(src["buf_id"])
            cl._munmap(dst["buf_id"])
            cl.deregister_mr(src_mr["mr_id"])
            cl.deregister_mr(dst_mr["mr_id"])
            cl.destroy_buffer(src["buf_id"])
            cl.destroy_buffer(dst["buf_id"])
        finally:
            cl.teardown_rdma()


# ── Test 5: GPU tensor alignment (conditional) ───────────────────────────

@test("GPU tensor alignment + pin (conditional CUDA)")
def test_gpu_alignment():
    try:
        import torch
        if not torch.cuda.is_available():
            print("    (skipped — no CUDA)")
            return
    except ImportError:
        print("    (skipped — no PyTorch)")
        return

    # Test at two sizes
    for size_mb in [1, 10]:
        size = size_mb * 1024 * 1024
        tensor = allocate_aligned_gpu_tensor(size)
        assert tensor.data_ptr() % 65536 == 0, \
            f"Not 64KB aligned at {size_mb}MB: 0x{tensor.data_ptr():x}"
        assert tensor.numel() >= size, \
            f"Too small: {tensor.numel()} < {size}"

        # Test pin/unpin via dmaplane
        with DmaplaneClient() as cl:
            cl.setup_rdma()
            try:
                pin = cl.gpu_pin(tensor.data_ptr(), tensor.numel())
                assert pin["handle"] > 0, f"bad handle: {pin['handle']}"
                cl.gpu_unpin(pin["handle"])
            finally:
                cl.teardown_rdma()

        del tensor


# ── Test 6: GC safety for aligned GPU tensor ─────────────────────────────

@test("GPU tensor GC safety (raw_storage_ref keeps data alive)")
def test_gpu_gc_safety():
    try:
        import torch
        if not torch.cuda.is_available():
            print("    (skipped — no CUDA)")
            return
    except ImportError:
        print("    (skipped — no PyTorch)")
        return

    tensor = allocate_aligned_gpu_tensor(1024 * 1024)
    ptr = tensor.data_ptr()

    # Verify raw_storage_ref exists
    assert hasattr(tensor, '_raw_storage_ref'), \
        "Missing _raw_storage_ref attribute"

    # Delete the reference but keep the tensor — GC shouldn't free storage
    ref = tensor._raw_storage_ref
    del ref
    gc.collect()

    # data_ptr should still be valid (tensor holds _raw_storage_ref)
    assert tensor.data_ptr() == ptr, \
        f"data_ptr changed after GC: 0x{tensor.data_ptr():x} != 0x{ptr:x}"

    del tensor


# ── KVCache IMM encoding tests ───────────────────────────────────────────

@test("KVCache IMM encode/decode")
def test_imm_encoding():
    # Basic encode/decode
    imm = kvcache_imm_encode(0, 0)
    assert imm == 0x00000000
    assert kvcache_imm_layer(imm) == 0
    assert kvcache_imm_chunk(imm) == 0

    imm = kvcache_imm_encode(31, 3)
    assert kvcache_imm_layer(imm) == 31
    assert kvcache_imm_chunk(imm) == 3

    imm = kvcache_imm_encode(0xFFFF, 0xFFFF)
    assert imm == 0xFFFFFFFF
    assert kvcache_imm_layer(imm) == 0xFFFF
    assert kvcache_imm_chunk(imm) == 0xFFFF

    # Sentinel
    assert KVCACHE_SENTINEL == 0xFFFFFFFF


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("test_dmaplane_py.py — Python wrapper tests\n")

    # Tests that don't need hardware
    run_test(test_struct_sizes)
    run_test(test_imm_encoding)

    # Tests that need dmaplane.ko + rxe
    rxe = find_rxe_device()
    if rxe is None:
        print("\n  SKIP: No rxe device found (need dmaplane.ko + rxe)")
        print(f"\nResults: {passed} passed, {failed} failed, "
              "hardware tests skipped")
        sys.exit(1 if failed > 0 else 0)

    if not os.path.exists("/dev/dmaplane"):
        print("\n  SKIP: /dev/dmaplane not found (need dmaplane.ko loaded)")
        print(f"\nResults: {passed} passed, {failed} failed, "
              "hardware tests skipped")
        sys.exit(1 if failed > 0 else 0)

    run_test(test_create_buffer_local_mr)
    run_test(test_create_buffer_remote_mr)
    run_test(test_writeimm_roundtrip)
    run_test(test_gpu_alignment)
    run_test(test_gpu_gc_safety)

    print(f"\nResults: {passed} passed, {failed} failed")
    sys.exit(1 if failed > 0 else 0)
