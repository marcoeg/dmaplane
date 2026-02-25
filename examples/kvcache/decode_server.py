#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0
# Copyright (c) 2026 Graziano Labs Corp.
"""
decode_server.py — Disaggregated inference decode node

Machine B: loads the same model, connects to the prefill server, sends
prompts, receives KVCache via dmaplane RDMA WRITE WITH IMMEDIATE, and
runs autoregressive decoding with the reconstructed past_key_values.

Data path:
  Recv WRITEIMM chunks into host MR → mmap → torch.frombuffer →
  reconstruct_kvcache_zero_copy(as_strided views) → move to GPU →
  model.forward(past_key_values=reconstructed) → greedy decode loop

TCP protocol (matches C peer mode in kvcache_common.h):
  1. Exchange binary tcp_metadata (72 bytes) — QP/MR/config
  2. Pre-post RECVs, send 1-byte "ready"
  3. Per prompt: send prompt (len-prefixed), recv JSON manifest,
     poll for WRITEIMM chunks with TCP credit flow, send "done" ack

Usage:
  sudo python3 decode_server.py --peer 127.0.0.1 --port 9876
  sudo python3 decode_server.py --peer 10.0.0.10 --port 9876 --max-tokens 100
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
    DmaplaneClient,
    IB_ACCESS_LOCAL_WRITE, IB_ACCESS_REMOTE_WRITE,
    kvcache_imm_layer, kvcache_imm_chunk, KVCACHE_SENTINEL,
    tcp_send_all, tcp_recv_all, tcp_send_metadata, tcp_recv_metadata,
    tcp_recv_json,
)
from kvcache_inference import reconstruct_kvcache_zero_copy

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_PORT = 9876
DEFAULT_MAX_TOKENS = 100
DEFAULT_CREDIT_WINDOW = 16

DEFAULT_PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "Write a short poem about kernel programming.",
    "What are the advantages of disaggregated inference for LLM serving?",
]


def log(msg):
    print(f"[DECODE] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="Decode server for disaggregated inference")
    p.add_argument("--peer", required=True, help="Prefill server IP")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--credit-window", type=int, default=DEFAULT_CREDIT_WINDOW)
    p.add_argument("--prompts-file", default=None,
                   help="File with one prompt per line (default: built-in)")
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


def load_prompts(prompts_file):
    """Load prompts from file or use defaults."""
    if prompts_file:
        with open(prompts_file) as f:
            return [line.strip() for line in f if line.strip()]
    return DEFAULT_PROMPTS


def main():
    args = parse_args()
    model, tokenizer = load_model(args.model)
    prompts = load_prompts(args.prompts_file)

    # ── Setup dmaplane RDMA ──────────────────────────────────────────
    cl = DmaplaneClient()
    cl.setup_rdma(max_send_wr=256, max_recv_wr=256)

    # ── Initialize peer QP ───────────────────────────────────────────
    local_peer = cl.init_peer()
    log(f"Peer QP initialized: qpn={local_peer['qp_num']}")

    # ── TCP connect to prefill server ────────────────────────────────
    conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    conn.connect((args.peer, args.port))
    log(f"Connected to prefill at {args.peer}:{args.port}")

    # Receive prefill server's metadata (QP info + config)
    remote = tcp_recv_metadata(conn)
    chunk_size = remote["chunk_size"]
    credit_window = remote["credit_window"]
    buf_size = remote["buf_size"]
    log(f"Remote config: chunk_size={chunk_size // 1024}KB, "
        f"credit_window={credit_window}, buf_size={buf_size // 1024 // 1024}MB")

    # ── Allocate dest buffer (sized from sender's config) ────────────
    host_buf = cl.create_buffer(buf_size)
    host_mr = cl.register_mr(host_buf["buf_id"],
                             IB_ACCESS_LOCAL_WRITE | IB_ACCESS_REMOTE_WRITE)
    host_mmap = cl.mmap_buffer(host_buf["buf_id"])
    log(f"Dest buffer: {buf_size / 1024 / 1024:.1f} MB, "
        f"mr_id={host_mr['mr_id']}, rkey=0x{host_mr['rkey']:x}")

    # ── Send our metadata back (QP info + dest MR) ───────────────────
    tcp_send_metadata(conn,
                      gid=local_peer["gid"],
                      mac=local_peer["mac"],
                      lid=local_peer["lid"],
                      qpn=local_peer["qp_num"],
                      mr_addr=host_mr["addr"],
                      mr_rkey=host_mr["rkey"],
                      num_layers=0,
                      chunks_per_layer=0,
                      chunk_size=0,
                      credit_window=0,
                      buf_size=0)

    # ── Connect peer QP ──────────────────────────────────────────────
    cl.connect_peer(remote["qpn"], remote["lid"],
                    remote["gid"], remote["mac"])
    log(f"RDMA peer connected. Buffer: {buf_size // 1024 // 1024} MB, "
        f"credit_window={credit_window}")

    # ── Pre-post initial RECVs ───────────────────────────────────────
    # Post enough for the first transfer. We'll replenish per-chunk.
    initial_recvs = credit_window + 1  # +1 for sentinel
    for i in range(initial_recvs):
        cl.post_recv(host_mr["mr_id"], chunk_size, use_peer_qp=1)

    # ── Send "ready" signal ──────────────────────────────────────────
    tcp_send_all(conn, b'\x01')
    log("Ready. Sending prompts...")
    print()

    try:
        for prompt_idx, prompt in enumerate(prompts):
            # ── Send prompt ──────────────────────────────────────────
            prompt_bytes = prompt.encode('utf-8')
            tcp_send_all(conn, struct.pack('!I', len(prompt_bytes)))
            tcp_send_all(conn, prompt_bytes)
            log(f"Sending prompt: '{prompt}'")

            # ── Receive manifest JSON ────────────────────────────────
            meta = tcp_recv_json(conn)
            manifest = meta["manifest"]
            first_token_id = meta["first_token_id"]
            seq_len = meta["seq_len"]
            num_layers = meta["num_layers"]
            total_bytes = meta["total_bytes"]
            num_chunks = meta["num_chunks"]

            # Restore torch.dtype from string
            for entry in manifest:
                entry["dtype"] = getattr(torch, entry["dtype"].split('.')[-1])
                entry["shape"] = tuple(entry["shape"])

            log(f"Manifest: {num_layers} layers, "
                f"{total_bytes / 1024 / 1024:.1f} MB, "
                f"{num_chunks} chunks")

            # ── Receive WRITEIMM chunks ──────────────────────────────
            t_xfer = time.monotonic()
            chunks_received = 0

            for _ in range(num_chunks + 1):  # +1 for sentinel
                result = cl.poll_recv(use_peer_qp=1, timeout_ms=10000)
                if result["status"] != 0:
                    raise RuntimeError(
                        f"poll_recv failed: status={result['status']}")

                imm = result["imm_data"]
                if imm == KVCACHE_SENTINEL:
                    break

                layer = kvcache_imm_layer(imm)
                chunk = kvcache_imm_chunk(imm)
                chunks_received += 1

                if chunks_received <= 3 or chunks_received == num_chunks:
                    log(f"<- chunk {chunks_received}/{num_chunks}: "
                        f"layer={layer} chunk={chunk}")
                elif chunks_received == 4:
                    log("...")

                # Replenish recv + send credit
                cl.post_recv(host_mr["mr_id"], chunk_size, use_peer_qp=1)
                tcp_send_all(conn, b'\x01')

            xfer_ms = (time.monotonic() - t_xfer) * 1000
            throughput = (total_bytes / 1024 / 1024) / (xfer_ms / 1000) \
                if xfer_ms > 0 else 0
            log(f"<- SENTINEL")
            log(f"KVCache received: {chunks_received} chunks in "
                f"{xfer_ms:.1f}ms ({throughput:.1f} MB/s)")

            # ── Send "done" ack ──────────────────────────────────────
            tcp_send_all(conn, b'\x01')

            # Replenish the recv WR consumed by the sentinel — without
            # this, each prompt leaves one fewer posted recv, eventually
            # causing RNR on subsequent transfers.
            cl.post_recv(host_mr["mr_id"], chunk_size, use_peer_qp=1)

            # ── Reconstruct KVCache from host mmap ───────────────────
            t_recon = time.monotonic()
            host_data = bytearray(host_mmap[:total_bytes])
            host_tensor = torch.frombuffer(
                host_data, dtype=torch.uint8
            ).cuda()

            reconstructed = reconstruct_kvcache_zero_copy(
                host_tensor, manifest)
            recon_ms = (time.monotonic() - t_recon) * 1000
            log(f"KVCache reconstructed: {num_layers} layers in "
                f"{recon_ms:.1f}ms")

            # ── Autoregressive decode (greedy) ───────────────────────
            first_token_text = tokenizer.decode([first_token_id],
                                                skip_special_tokens=True)
            log(f"Generating with first token: '{first_token_text.strip()}' "
                f"(token_id={first_token_id})")

            t_decode = time.monotonic()
            next_token = torch.tensor([[first_token_id]],
                                      device="cuda", dtype=torch.long)
            past_kv = reconstructed
            generated_tokens = [first_token_id]
            output_parts = [first_token_text]

            sys.stdout.write(f"[DECODE] Output: {first_token_text}")
            sys.stdout.flush()

            for step in range(args.max_tokens - 1):
                with torch.no_grad():
                    out = model(next_token, past_key_values=past_kv,
                                use_cache=True)
                past_kv = out.past_key_values
                next_logits = out.logits[:, -1, :]
                next_token_id = int(torch.argmax(next_logits, dim=-1).item())
                generated_tokens.append(next_token_id)

                # Check for EOS
                if next_token_id == tokenizer.eos_token_id:
                    break

                token_text = tokenizer.decode([next_token_id],
                                              skip_special_tokens=True)
                output_parts.append(token_text)
                sys.stdout.write(token_text)
                sys.stdout.flush()

                next_token = torch.tensor([[next_token_id]],
                                          device="cuda", dtype=torch.long)

            print()  # newline after streaming output
            decode_ms = (time.monotonic() - t_decode) * 1000
            num_tokens = len(generated_tokens)
            tok_per_s = num_tokens / (decode_ms / 1000) if decode_ms > 0 else 0
            log(f"Generated {num_tokens} tokens in "
                f"{decode_ms / 1000:.2f}s ({tok_per_s:.1f} tok/s)")

            # ── Pipeline summary ─────────────────────────────────────
            total_ms = xfer_ms + recon_ms + decode_ms
            ttft_ms = xfer_ms + recon_ms
            print(f"\nPIPELINE SUMMARY:")
            print(f"  KVCache transfer:    {xfer_ms:.1f}ms  "
                  f"({num_chunks} chunks x {chunk_size // 1024} KB "
                  f"via WRITEIMM)")
            print(f"  KVCache reconstruct: {recon_ms:.1f}ms")
            print(f"  Decode ({num_tokens} tokens):  "
                  f"{decode_ms / 1000:.2f}s  "
                  f"({decode_ms / num_tokens:.1f}ms/token avg)")
            print(f"  Time-to-first-token: {ttft_ms:.1f}ms  "
                  f"(transfer + reconstruct)")
            print()

        # ── Send shutdown signal (length=0) ──────────────────────────
        tcp_send_all(conn, struct.pack('!I', 0))
        log("Shutdown signal sent.")

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
