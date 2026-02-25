#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0
# Copyright (c) 2026 Graziano Labs Corp.
"""
prefill_server.py — Disaggregated inference prefill node

Machine A: loads model, runs prefill on each prompt, consolidates KVCache
into a contiguous staging buffer, and transfers it to the decode node via
dmaplane RDMA WRITE WITH IMMEDIATE.

Data path:
  GPU prefill → extract KVCache (GPU tensors) →
  consolidate_kvcache(layers, gpu_staging) →
  torch.cuda.synchronize() [fence before reading GPU data] →
  gpu_staging.cpu().numpy().tobytes() [GPU→CPU copy] →
  memcpy into mmap'd dmaplane host buffer →
  WRITEIMM chunks from host MR to remote host MR

TCP protocol (matches C peer mode in kvcache_common.h):
  1. Exchange binary tcp_metadata (72 bytes) — QP/MR/config
  2. Wait for 1-byte "ready" from decode server
  3. Per prompt: recv prompt (len-prefixed), send JSON manifest,
     WRITEIMM chunks with TCP credit flow, sentinel, wait for "done"

Usage:
  sudo python3 prefill_server.py --port 9876
  sudo python3 prefill_server.py --port 9876 --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

import sys
import os
import time
import argparse
import socket
import struct

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dmaplane_py import (
    DmaplaneClient, IB_ACCESS_LOCAL_WRITE,
    kvcache_imm_encode, KVCACHE_SENTINEL,
    tcp_send_all, tcp_recv_all, tcp_send_metadata, tcp_recv_metadata,
    tcp_send_json, TCP_METADATA_SZ,
)
from kvcache_inference import (
    extract_kvcache, consolidate_kvcache,
    kvcache_size_estimate, kvcache_size_from_layers,
)

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_PORT = 9876
DEFAULT_CHUNK_SIZE = 1024 * 1024   # 1 MB
DEFAULT_CREDIT_WINDOW = 16
MAX_SEQ_LEN = 512


def log(msg):
    print(f"[PREFILL] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="Prefill server for disaggregated inference")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    p.add_argument("--credit-window", type=int, default=DEFAULT_CREDIT_WINDOW)
    return p.parse_args()


def load_model(model_name):
    """Load model and tokenizer to GPU."""
    t0 = time.monotonic()
    log(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda"
    )
    model.eval()
    elapsed = time.monotonic() - t0
    log(f"Model loaded in {elapsed:.1f}s")
    return model, tokenizer


def main():
    args = parse_args()
    model, tokenizer = load_model(args.model)

    # ── Estimate max KVCache size and pre-allocate buffers ────────────
    max_kv_bytes = kvcache_size_estimate(model.config, MAX_SEQ_LEN)
    buf_size = ((max_kv_bytes + 65535) // 65536) * 65536  # round up to 64KB
    log(f"Max KVCache estimate: {max_kv_bytes / 1024 / 1024:.1f} MB "
        f"(seq_len={MAX_SEQ_LEN})")

    # GPU staging buffer for consolidate_kvcache() — must be on GPU
    gpu_staging = torch.empty(max_kv_bytes, dtype=torch.uint8, device="cuda")

    # ── Setup dmaplane RDMA ──────────────────────────────────────────
    cl = DmaplaneClient()
    num_chunks_max = (max_kv_bytes + args.chunk_size - 1) // args.chunk_size
    cl.setup_rdma(max_send_wr=num_chunks_max + 16,
                  max_recv_wr=num_chunks_max + 16)

    # Host buffer for WRITEIMM source — sized for max KVCache
    host_buf = cl.create_buffer(buf_size)
    host_mr = cl.register_mr(host_buf["buf_id"], IB_ACCESS_LOCAL_WRITE)
    host_mmap = cl.mmap_buffer(host_buf["buf_id"])
    log(f"Host buffer: {buf_size / 1024 / 1024:.1f} MB, "
        f"mr_id={host_mr['mr_id']}")

    # ── Initialize peer QP ───────────────────────────────────────────
    local_peer = cl.init_peer()
    log(f"Peer QP initialized: qpn={local_peer['qp_num']}")

    # ── TCP listen for decode server ─────────────────────────────────
    listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listen_sock.bind(("0.0.0.0", args.port))
    listen_sock.listen(1)
    log(f"Listening on 0.0.0.0:{args.port}")

    conn, addr = listen_sock.accept()
    listen_sock.close()
    log(f"Decode server connected from {addr[0]}:{addr[1]}")

    try:
        # ── Exchange tcp_metadata ────────────────────────────────────
        # Sender sends first: QP info + config (MR info zeros — sender
        # doesn't expose its MR to the receiver for RDMA targeting)
        tcp_send_metadata(conn,
                          gid=local_peer["gid"],
                          mac=local_peer["mac"],
                          lid=local_peer["lid"],
                          qpn=local_peer["qp_num"],
                          mr_addr=0,  # sender's MR not needed by receiver
                          mr_rkey=0,
                          num_layers=model.config.num_hidden_layers,
                          chunks_per_layer=0,  # computed per-prompt
                          chunk_size=args.chunk_size,
                          credit_window=args.credit_window,
                          buf_size=buf_size)

        # Receive decode server's metadata (QP info + dest MR)
        remote = tcp_recv_metadata(conn)
        remote_addr = remote["mr_addr"]
        remote_rkey = remote["mr_rkey"]
        log(f"Remote MR: addr=0x{remote_addr:x}, rkey=0x{remote_rkey:x}")

        # ── Connect peer QP ──────────────────────────────────────────
        cl.connect_peer(remote["qpn"], remote["lid"],
                        remote["gid"], remote["mac"])
        log("RDMA peer connected.")

        # ── Wait for "ready" from decode server ──────────────────────
        ready = tcp_recv_all(conn, 1)
        log("Waiting for prompts...")

        # ── Per-prompt loop ──────────────────────────────────────────
        while True:
            # Receive prompt length (4 bytes big-endian), then prompt
            hdr = tcp_recv_all(conn, 4)
            prompt_len = struct.unpack('!I', hdr)[0]
            if prompt_len == 0:
                log("Shutdown signal received.")
                break

            prompt = tcp_recv_all(conn, prompt_len).decode('utf-8')
            log(f"Prompt: '{prompt}'")

            # ── Tokenize ─────────────────────────────────────────────
            t_tok = time.monotonic()
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            input_ids = inputs["input_ids"]
            tok_ms = (time.monotonic() - t_tok) * 1000
            log(f"Tokenized: {input_ids.shape[1]} tokens in {tok_ms:.1f}ms")

            # ── Prefill forward pass ─────────────────────────────────
            t_fwd = time.monotonic()
            with torch.no_grad():
                output = model(**inputs, use_cache=True)
            torch.cuda.synchronize()
            fwd_ms = (time.monotonic() - t_fwd) * 1000
            log(f"Forward pass: {fwd_ms:.1f}ms")

            # ── Extract + consolidate KVCache ────────────────────────
            t_ext = time.monotonic()
            layers = extract_kvcache(output)
            total_bytes = kvcache_size_from_layers(layers)
            assert total_bytes <= max_kv_bytes, \
                f"KVCache {total_bytes} exceeds buffer {max_kv_bytes}"

            packed_bytes, manifest = consolidate_kvcache(layers, gpu_staging)
            torch.cuda.synchronize()  # MANDATORY: fence before CPU reads GPU data
            ext_ms = (time.monotonic() - t_ext) * 1000

            num_layers = len(layers)
            log(f"KVCache: {num_layers} layers, "
                f"{packed_bytes / 1024 / 1024:.1f} MB, "
                f"serialized in {ext_ms:.1f}ms")

            # ── Copy GPU staging → host mmap buffer ──────────────────
            t_copy = time.monotonic()
            cpu_bytes = gpu_staging[:packed_bytes].cpu().numpy().tobytes()
            host_mmap[:packed_bytes] = cpu_bytes
            copy_ms = (time.monotonic() - t_copy) * 1000

            # ── First token (argmax of prefill logits) ───────────────
            first_token_logits = output.logits[:, -1, :]
            first_token_id = int(torch.argmax(first_token_logits, dim=-1).item())

            # ── Chunk math ───────────────────────────────────────────
            num_chunks = (packed_bytes + args.chunk_size - 1) // args.chunk_size

            # ── Send JSON manifest over TCP ──────────────────────────
            # Convert manifest for JSON (torch.dtype not serializable)
            json_manifest = []
            for entry in manifest:
                json_entry = dict(entry)
                json_entry["dtype"] = str(entry["dtype"])
                json_entry["shape"] = list(entry["shape"])
                json_manifest.append(json_entry)

            meta_obj = {
                "manifest": json_manifest,
                "first_token_id": first_token_id,
                "seq_len": int(input_ids.shape[1]),
                "num_layers": num_layers,
                "total_bytes": packed_bytes,
                "num_chunks": num_chunks,
                "chunk_size": args.chunk_size,
            }
            tcp_send_json(conn, meta_obj)

            # ── WRITEIMM loop with credit flow ───────────────────────
            log(f"-> WRITEIMM {num_chunks} chunks "
                f"({args.chunk_size // 1024} KB each), "
                f"credit_window={args.credit_window}")

            t_xfer = time.monotonic()
            credits = args.credit_window
            stalls = 0
            credits_received = 0

            for i in range(num_chunks):
                # Credit management
                if credits <= 0:
                    # Block for at least 1 credit
                    tcp_recv_all(conn, 1)
                    credits += 1
                    credits_received += 1
                    stalls += 1
                    # Drain any extras non-blocking
                    conn.setblocking(False)
                    try:
                        extra = conn.recv(64)
                        if extra:
                            credits += len(extra)
                            credits_received += len(extra)
                    except BlockingIOError:
                        pass
                    conn.setblocking(True)
                else:
                    # Opportunistic non-blocking credit drain
                    conn.setblocking(False)
                    try:
                        extra = conn.recv(64)
                        if extra:
                            credits += len(extra)
                            credits_received += len(extra)
                    except BlockingIOError:
                        pass
                    conn.setblocking(True)

                offset = i * args.chunk_size
                length = min(args.chunk_size, packed_bytes - offset)
                imm = kvcache_imm_encode(0, i)  # flat chunk enumeration

                cl.write_imm(host_mr["mr_id"], offset,
                             remote_addr + offset, remote_rkey,
                             length, imm, use_peer_qp=1)
                credits -= 1

            # Send sentinel
            cl.write_imm(host_mr["mr_id"], 0,
                         remote_addr, remote_rkey,
                         4, KVCACHE_SENTINEL, use_peer_qp=1)

            # Decode sends exactly num_chunks credit bytes + 1 done byte.
            # Some credits were already consumed during the WRITEIMM loop.
            # Read whatever remains in one blocking call.
            remaining = (num_chunks - credits_received) + 1
            if remaining > 0:
                tcp_recv_all(conn, remaining)

            xfer_ms = (time.monotonic() - t_xfer) * 1000
            throughput = (packed_bytes / 1024 / 1024) / (xfer_ms / 1000) \
                if xfer_ms > 0 else 0
            log(f"Transfer complete: {packed_bytes / 1024 / 1024:.1f} MB "
                f"in {xfer_ms:.1f}ms = {throughput:.1f} MB/s "
                f"({stalls} credit stalls)")

            total_ms = tok_ms + fwd_ms + ext_ms + copy_ms + xfer_ms
            log(f"Pipeline: tokenize={tok_ms:.1f}ms prefill={fwd_ms:.1f}ms "
                f"extract={ext_ms:.1f}ms copy={copy_ms:.1f}ms "
                f"transfer={xfer_ms:.1f}ms total={total_ms:.1f}ms")
            print()

    finally:
        # ── Cleanup (reverse order) ──────────────────────────────────
        conn.close()
        try:
            cl.destroy_peer()
        except OSError:
            pass
        cl._munmap(host_buf["buf_id"])
        cl.deregister_mr(host_mr["mr_id"])
        cl.destroy_buffer(host_buf["buf_id"])
        cl.teardown_rdma()
        cl.close()
        log("Shutdown complete.")


if __name__ == "__main__":
    main()
