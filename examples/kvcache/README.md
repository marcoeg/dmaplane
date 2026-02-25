# KVCache Disaggregated Inference

Validates the full disaggregated KVCache transfer pipeline over RDMA WRITE with immediate data (`IB_WR_RDMA_WRITE_WITH_IMM`). Designed to run on a single laptop via soft-RoCE loopback before deploying to EC2.

## How it works

```
Prefill GPU                                  Decode GPU
┌──────────────┐    RDMA WRITE_IMM    ┌──────────────┐
│  TinyLlama   │    + 32-bit imm      │  Reconstruct │
│  prefill     │───────────────────────│  KVCache via │
│              │   chunk by chunk      │  as_strided  │
│  KVCache     │   (credit window)     │              │
│  consolidate │                       │  Continue    │
│  into staging│                       │  decoding    │
└──────────────┘                       └──────────────┘
```

The 32-bit immediate value encodes `(layer_index, chunk_index)` so the receiver knows which KVCache chunk arrived via CQ completion — no memory polling needed. A credit window provides backpressure.

In loopback mode, the sender owns both MRs on the same machine (QP-A sends, QP-B receives). In peer mode (`--peer <ip>`), sender and receiver run on separate machines, connected via TCP for metadata exchange and credit flow, with RDMA WRITEIMM for data transfer via the kernel's peer QP.

## Dependencies

### System (required for all tests)

- Linux kernel 6.5+ with `rdma_rxe` module
- `dmaplane.ko` kernel module loaded
- `rdma-core` package (`rdma link` command)
- GCC, Make

### C tools (kvcache_sender, kvcache_receiver)

- CUDA runtime (`libcudart`) — optional, only needed for `--gpu` flag
- `-lm` (math library)

### Python wrapper (dmaplane_py.py, test_dmaplane_py.py)

- Python 3.8+
- No pip dependencies (uses `struct` + `fcntl` only)
- PyTorch — optional, only for `allocate_aligned_gpu_tensor` and GPU tests

### End-to-end inference test (test_kvcache_local.py)

```bash
sudo pip install torch transformers accelerate
```

- CUDA GPU (any NVIDIA GPU with compute capability 7.0+)
- TinyLlama-1.1B-Chat model (auto-downloaded on first run, ~2 GB)
- `dmaplane.ko` loaded + Soft-RoCE configured

### Disaggregated inference servers (prefill_server.py, decode_server.py)

Same as end-to-end test above, plus:
- Both machines need `dmaplane.ko` loaded + Soft-RoCE configured
- Both machines need the same model weights (TinyLlama auto-downloads)
- TCP connectivity between machines on port 9876 (or `--port`)

## Setup

### 1. Load dmaplane and configure Soft-RoCE

```bash
# From repo root
bash scripts/setup_rxe.sh            # auto-detects interface
sudo insmod driver/dmaplane.ko
```

Or manually:

```bash
sudo modprobe rdma_rxe
IFACE=$(ip route show default | awk '{print $5; exit}')
sudo rdma link add rxe_${IFACE} type rxe netdev ${IFACE}
sudo insmod driver/dmaplane.ko
```

Verify: `rdma link show` should show an `rxe_<iface>` device, and `/dev/dmaplane` should exist.

### 2. Build C tools

```bash
make -C examples/kvcache              # or just 'make' from repo root
```

Builds `kvcache_sender` (with CUDA if available) and `kvcache_receiver`.

## Files

| File | Lines | Description |
|------|------:|-------------|
| `kvcache_common.h` | 853 | Shared C header: ioctl wrappers (WRITEIMM + peer), TCP helpers, credit tracker, recv helpers, latency stats, bitmap, fill/verify |
| `kvcache_sender.c` | 1039 | Loopback + peer mode WRITEIMM sender with credit window, GPU path, TCP credit flow |
| `kvcache_receiver.c` | 526 | Loopback self-test + peer mode receiver with TCP metadata exchange and credit flow |
| `dmaplane_py.py` | 637 | Python `struct`/`fcntl` wrapper for all dmaplane ioctls, peer QP methods, TCP metadata exchange |
| `test_dmaplane_py.py` | 293 | Python wrapper tests: struct sizes, ioctl round-trips, GPU pin |
| `kvcache_inference.py` | 197 | `extract_kvcache`, `consolidate_kvcache`, `reconstruct_kvcache_zero_copy` |
| `test_kvcache_local.py` | 475 | End-to-end test: TinyLlama prefill → WRITEIMM → reconstruct → decode |
| `prefill_server.py` | 318 | Disaggregated prefill node: model prefill → WRITEIMM send |
| `decode_server.py` | 325 | Disaggregated decode node: WRITEIMM recv → reconstruct → decode |
| `kvcache_bench.sh` | 84 | Benchmark sweep: chunk_size x credit_window x layers → CSV |
| `ec2_setup.sh` | 153 | EC2 g5.xlarge setup: deps, rxe, build, model download, smoke test |
| `Makefile` | 38 | C build with CUDA detection |
| `PYTHON_GUIDE.md` | 337 | Python API reference and usage examples |

## Acceptance Tests

Run in this order. Each step validates a layer of the stack.

### Step 1: Phase 8 regression (verify ioctl renumbering)

```bash
make -C tests clean && make -C tests
sudo ./tests/test_phase8_gpu
# Expected: 7/7 passed
```

### Step 2: C loopback — basic WRITEIMM round-trip

```bash
sudo ./examples/kvcache/kvcache_sender --loopback --layers 4 --chunks-per-layer 2 --verify
# Expected: 8 chunks transferred, data integrity PASS, credit stalls reported
```

### Step 3: C loopback — full 128-chunk simulated KVCache

```bash
sudo ./examples/kvcache/kvcache_sender --loopback --verify
# Expected: 32 layers x 4 chunks = 128 chunks, 128 MB, ~1000+ MB/s on rxe
#           128/128 received, data integrity PASS
```

### Step 4: C loopback — GPU-backed source (requires CUDA)

```bash
sudo ./examples/kvcache/kvcache_sender --loopback --gpu --verify
# Expected: GPU MR as source, same transfer pattern, data verified
#           ~10 MB/s (rxe reads GPU BAR pages via software — slow by design)
```

### Step 5: C loopback — credit pressure

```bash
sudo ./examples/kvcache/kvcache_sender --loopback --credit-window 2 --layers 8 --verify
# Expected: many credit stalls (30 on 32 chunks), lower throughput, data correct
```

### Step 6: Python struct validation (no root needed)

```bash
python3 examples/kvcache/dmaplane_py.py
# Expected: all 13 struct sizes match (11 + peer_info + tcp_metadata),
#           all 13 ioctl numbers printed
```

### Step 7: Python ioctl round-trip tests

```bash
sudo python3 examples/kvcache/test_dmaplane_py.py
# Expected: 7+ tests pass (struct sizes, IMM encoding, buffer+MR,
#           fast-reg MR, WRITEIMM round-trip, GPU tests if PyTorch available)
```

### Step 8: Full disaggregated inference test (requires CUDA + PyTorch)

```bash
sudo python3 examples/kvcache/test_kvcache_local.py
# Expected: 6 tests pass:
#   1. KVCache extract + consolidate (shape, contiguity, manifest offsets)
#   2. as_strided zero-copy reconstruction (bitwise identical to originals)
#   3. Decode with reconstructed KVCache matches model.generate() reference
#   4. WRITEIMM loopback GPU→host with real KVCache (TinyLlama 135 KB,
#      GPU pin → WRITEIMM chunks → host mmap → reconstruct → decode → verify)
#   5. GPU tensor alignment + pin at KVCache size
#   6. Multiple prompts reuse pinned buffers without re-allocation
```

### Step 9: Benchmark sweep

```bash
sudo bash examples/kvcache/kvcache_bench.sh
# Expected: 32 data points across chunk_size(64K,256K,1M,4M) x
#           credit_window(4,8,16,32) x layers(8,32)
#           Results written to bench_results.csv
```

### Step 10: Peer mode — local TCP (two terminals, same machine)

```bash
# Terminal 1 (receiver):
sudo ./examples/kvcache/kvcache_receiver --port 9876 --verify

# Terminal 2 (sender):
sudo ./examples/kvcache/kvcache_sender --peer 127.0.0.1 --port 9876 \
    --layers 4 --chunks-per-layer 2 --verify
# Expected: receiver creates buffer from sender config, 8 chunks transferred
#           via peer QP, TCP credit flow, data integrity PASS on both sides
```

### Step 11: Peer mode — full 128-chunk transfer

```bash
# Terminal 1:
sudo ./examples/kvcache/kvcache_receiver --port 9876 --verify

# Terminal 2:
sudo ./examples/kvcache/kvcache_sender --peer 127.0.0.1 --port 9876 --verify
# Expected: 32 layers x 4 chunks = 128 chunks, 128 MB, data integrity PASS
```

### Step 12: Peer mode — credit pressure

```bash
# Terminal 1:
sudo ./examples/kvcache/kvcache_receiver --port 9876 --verify

# Terminal 2:
sudo ./examples/kvcache/kvcache_sender --peer 127.0.0.1 --port 9876 \
    --credit-window 2 --layers 8 --verify
# Expected: credit stalls, lower throughput, data correct
```

### Step 13: Peer mode — GPU source (requires CUDA)

```bash
# Terminal 1:
sudo ./examples/kvcache/kvcache_receiver --port 9876 --verify

# Terminal 2:
sudo ./examples/kvcache/kvcache_sender --peer 127.0.0.1 --port 9876 --gpu --verify
# Expected: GPU MR as source, same peer transfer pattern, data verified
```

### Step 14: Peer mode — two machines (EC2 or real hardware)

```bash
# Machine B (receiver, start first):
sudo ./kvcache_receiver --port 9876 --verify

# Machine A (sender):
sudo ./kvcache_sender --peer <machine-B-ip> --port 9876 --verify
# Expected: cross-machine RDMA WRITEIMM transfer, data integrity PASS
```

### Step 15: Disaggregated inference — local (two terminals, same machine)

```bash
# Terminal 1 (prefill):
sudo python3 examples/kvcache/prefill_server.py --port 9876

# Terminal 2 (decode):
sudo python3 examples/kvcache/decode_server.py --peer 127.0.0.1 --port 9876
# Expected: 3 prompts processed, coherent text generated for each,
#           pipeline summary with transfer time, reconstruction time, tok/s
```

### Step 16: Disaggregated inference — two machines (EC2 or real hardware)

```bash
# Machine A (prefill, start first):
sudo python3 prefill_server.py --port 9876

# Machine B (decode):
sudo python3 decode_server.py --peer <machine-A-ip> --port 9876
# Expected: TinyLlama disaggregated inference across machines,
#           tokens match single-machine model.generate() (greedy deterministic)
```

## Expected Performance (rxe loopback)

| Chunk Size | Peak Throughput | Avg Latency | P50 | P99 |
|-----------|----------------|-------------|-----|-----|
| 64 KB | ~310 MB/s | 0.1ms | 0.2ms | 0.3ms |
| 256 KB | ~1025 MB/s | 0.1–0.3ms | 0.2–0.4ms | 0.4–0.6ms |
| 1 MB | ~1115 MB/s | 0.5–0.7ms | 0.8ms | 1.2–1.4ms |
| 4 MB | ~1210 MB/s | 1.8–2.3ms | 3.2ms | 3.6–5.8ms |

Throughput saturates around 1.2 GB/s — rxe's single-threaded memcpy limit. Credit window has minimal effect at large chunks (serialized by rxe).

GPU source path: ~10 MB/s (117x slower than host). Each 1 MB chunk requires rxe to build 256 individual IB packets (4 KB MTU) from GPU BAR pages, with per-packet checksums and protocol processing. This bottleneck disappears with real RDMA NICs — ConnectX does hardware DMA from GPU BAR at line rate via GPUDirect.

## Sender CLI reference

```
--loopback            Self-contained loopback test (default)
--peer <ip>           Connect to remote kvcache_receiver
--port <port>         TCP port (default: 9876)
--gpu                 GPU-backed source MR via IOCTL_GPU_PIN
--ib-dev <name>       IB device (auto-detects rxe_* if omitted)
--layers N            Simulated KVCache layers (default: 32)
--chunks-per-layer N  Chunks per layer (default: 4)
--chunk-size N        Bytes per chunk (default: 1048576)
--credit-window N     Max in-flight WRITEIMMs (default: 16)
--iterations N        Repeat full transfer (default: 1)
--verify              Fill pattern + memcmp after transfer
--csv                 Single-line CSV: total_mb,throughput,avg,p50,p99
```

## Receiver CLI reference

```
--loopback            Minimal loopback self-test (default)
--port <port>         Listen for remote sender (default: 9876)
--gpu                 Use GPU-backed receive MR (requires CUDA)
--ib-dev <name>       IB device (auto-detects rxe_* if omitted)
--verify              Verify data integrity after transfer
```

## Prefill server CLI reference

```
--port <port>         TCP listen port (default: 9876)
--model <name>        HuggingFace model (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)
--chunk-size N        Bytes per WRITEIMM chunk (default: 1048576)
--credit-window N     Max in-flight WRITEIMMs (default: 16)
```

## Decode server CLI reference

```
--peer <ip>           Prefill server IP (required)
--port <port>         TCP port (default: 9876)
--model <name>        HuggingFace model (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)
--max-tokens N        Max tokens to generate per prompt (default: 100)
--credit-window N     Max in-flight WRITEIMMs (default: 16)
--prompts-file <path> File with one prompt per line (default: 3 built-in prompts)
```

## Architecture notes

- **No libibverbs** — all RDMA ops go through kernel ioctls, not userspace verbs
- **Fast-reg MR** — destination MR uses `ib_alloc_mr` + `IB_WR_REG_MR` for valid rkey
- **Credit window** — sender tracks pre-posted recvs; stalls and drains when credits hit 0
- **TCP credit flow** — in peer mode, receiver sends 1-byte TCP messages after each recv completion; sender blocks on TCP read when credits hit 0 (no direct access to receiver's CQ)
- **TCP metadata exchange** — `struct tcp_metadata` bundles QP connection info (GID, QPN, MAC), MR targeting info (addr, rkey), and transfer config (layers, chunks, chunk_size, credit_window)
- **Peer QP routing** — `use_peer_qp=1` routes WRITEIMM to `qp_peer/cq_peer` (cross-machine); `use_peer_qp=0` routes to `qp_a/qp_b` (loopback)
- **64KB alignment** — GPU tensors are over-allocated and sliced for NVIDIA P2P API compatibility
- **Zero-copy reconstruct** — `torch.as_strided` views into the RDMA landing buffer, no memcpy
- **IMM encoding** — `KVCACHE_IMM_ENCODE(layer, chunk)` packs into 32-bit immediate (upper 16 = layer, lower 16 = chunk), `KVCACHE_SENTINEL = 0xFFFFFFFF` signals end-of-transfer
- **Byte order** — `cpu_to_be32(imm_data)` on send, `be32_to_cpu(wc.ex.imm_data)` on recv per IB spec
- **Host-backed RDMA** — Python servers use host buffers (not GPU-pinned MR). Data path: GPU prefill → consolidate to GPU staging → `cuda.synchronize()` → CPU copy → dmaplane host buffer → WRITEIMM. GPU-pinned RDMA is ~117x slower via rxe BAR reads.
- **JSON manifest** — per-prompt KVCache metadata (layer offsets, shapes, dtype, first_token) sent as length-prefixed JSON over TCP, on top of the binary `tcp_metadata` exchange
- **cuda.synchronize() fence** — mandatory between `consolidate_kvcache()` (GPU HBM→HBM async copy) and CPU read of GPU staging buffer. Without it, WRITEIMM reads stale BAR data.
- **Recv WR sentinel replenish** — after each transfer, the sentinel's consumed recv WR is replenished before the next prompt to prevent progressive recv WR depletion across multi-prompt sessions
