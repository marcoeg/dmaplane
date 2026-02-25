# Python Interface Guide

`dmaplane_py.py` provides a zero-dependency Python wrapper for the `/dev/dmaplane` ioctl interface using `struct.pack` and `fcntl.ioctl`. `kvcache_inference.py` adds PyTorch helpers for disaggregated KVCache transfer.

## DmaplaneClient

The main entry point. Manages the file descriptor and tracks mmaps for cleanup.

```python
from dmaplane_py import DmaplaneClient, IB_ACCESS_LOCAL_WRITE, IB_ACCESS_REMOTE_WRITE

# Context manager (recommended)
with DmaplaneClient() as cl:
    cl.setup_rdma()
    # ... use cl ...
    cl.teardown_rdma()

# Or manual lifecycle
cl = DmaplaneClient("/dev/dmaplane")
cl.close()
```

## API Reference

### Buffer Management

```python
# Create a DMA buffer (page-backed, NUMA-local)
result = cl.create_buffer(size=4 * 1024 * 1024)
# result: {"buf_id": 1, "actual_numa_node": 0}

# mmap into this process (tracks internally, closed on cl.close())
m = cl.mmap_buffer(buf_id)
m[:4] = b"\xDE\xAD\xBE\xEF"   # write
data = bytes(m[:4])              # read

# Explicit unmap (required before destroy_buffer if buffer is mapped)
cl._munmap(buf_id)

# Destroy
cl.destroy_buffer(buf_id)
```

### RDMA Setup

```python
# Auto-detects rxe_* device if ib_dev is omitted
cl.setup_rdma(ib_dev=None, cq_depth=256, max_send_wr=64, max_recv_wr=64)

# Teardown (after deregistering MRs)
cl.teardown_rdma()
```

### Memory Regions

```python
# Local-only MR (for source buffers)
mr = cl.register_mr(buf_id, IB_ACCESS_LOCAL_WRITE)
# mr: {"mr_id": 1, "buf_id": 1, "access_flags": 1, "lkey": 0x..., "rkey": 0, "addr": 0}

# Remote-write MR via fast-reg (for RDMA WRITE destinations)
mr = cl.register_mr(buf_id, IB_ACCESS_LOCAL_WRITE | IB_ACCESS_REMOTE_WRITE)
# mr: {"mr_id": 2, ..., "rkey": 0x..., "addr": 0xffff...}
#   rkey and addr are nonzero — needed for write_imm remote_rkey/remote_addr

cl.deregister_mr(mr["mr_id"])
```

### RDMA WRITE with Immediate Data

The core transfer primitive. Writes data from a local MR to a remote MR address and delivers a 32-bit immediate value through the receiver's completion queue.

```python
from dmaplane_py import kvcache_imm_encode, KVCACHE_SENTINEL

# 1. Post a recv on the destination side (must happen before write_imm)
cl.post_recv(dst_mr["mr_id"], size=4096, use_peer_qp=0)

# 2. RDMA WRITE with immediate data
imm = kvcache_imm_encode(layer=3, chunk=7)   # pack into 32-bit value
elapsed_ns = cl.write_imm(
    local_mr_id=src_mr["mr_id"],
    local_offset=0,
    remote_addr=dst_mr["addr"],
    remote_rkey=dst_mr["rkey"],
    length=4096,
    imm_data=imm,
    use_peer_qp=0,          # 0=loopback QP-A, 1=peer QP
)

# 3. Poll recv CQ for the completion
result = cl.poll_recv(use_peer_qp=0, timeout_ms=5000)
# result: {"status": 0, "wc_flags": ..., "imm_data": 0x00030007,
#          "byte_len": 4096, "elapsed_ns": ...}
```

### GPU P2P (requires CUDA)

```python
import torch

# Pin GPU memory (must be 64KB-aligned VA and size)
pin = cl.gpu_pin(gpu_va=tensor.data_ptr(), size=tensor.numel())
# pin: {"handle": 1, "gpu_numa_node": 0, "num_pages": 16, "bar1_consumed": ...}

# Register as RDMA MR
gpu_mr = cl.gpu_register_mr(gpu_handle=pin["handle"])
# gpu_mr: {"gpu_handle": 1, "mr_id": 3, "lkey": 0x..., "rkey": 0}

# Use gpu_mr["mr_id"] as local_mr_id in write_imm

# Cleanup
cl.gpu_unpin(pin["handle"])
```

## Cleanup Order

The kernel enforces strict resource ordering. Always clean up in this sequence:

```
1. Close mmaps          cl._munmap(buf_id)
2. Deregister MRs       cl.deregister_mr(mr_id)
3. GPU unpin            cl.gpu_unpin(handle)
4. Destroy buffers      cl.destroy_buffer(buf_id)
5. Teardown RDMA        cl.teardown_rdma()
6. Close device         cl.close()
```

Violating this order (e.g., destroying a buffer with an active mmap or registered MR) raises `OSError: [Errno 16] Device or resource busy`.

## IMM Encoding

The 32-bit immediate value encodes `(layer_index, chunk_index)` for KVCache chunk tracking:

```python
from dmaplane_py import kvcache_imm_encode, kvcache_imm_layer, kvcache_imm_chunk, KVCACHE_SENTINEL

imm = kvcache_imm_encode(layer=31, chunk=3)  # 0x001F0003
layer = kvcache_imm_layer(imm)                # 31
chunk = kvcache_imm_chunk(imm)                # 3

# Sentinel signals end of transfer
KVCACHE_SENTINEL  # 0xFFFFFFFF
```

## GPU Tensor Alignment

NVIDIA P2P requires 64KB-aligned virtual addresses. `allocate_aligned_gpu_tensor` handles this:

```python
from dmaplane_py import allocate_aligned_gpu_tensor

tensor = allocate_aligned_gpu_tensor(size_bytes=10 * 1024 * 1024)
assert tensor.data_ptr() % 65536 == 0

# The tensor holds a _raw_storage_ref to prevent GC from freeing the
# underlying over-allocated storage. Keep the tensor alive as long
# as the GPU memory is pinned.
```

## KVCache Inference Helpers

`kvcache_inference.py` provides the disaggregated inference pipeline:

```python
from kvcache_inference import (
    extract_kvcache, consolidate_kvcache,
    reconstruct_kvcache_zero_copy, kvcache_size_estimate,
    kvcache_size_from_layers,
)
```

### Extract

Pull `(K, V)` pairs from a HuggingFace model output. Each tensor is made contiguous with shape `[batch, num_kv_heads, seq_len, head_dim]`.

```python
output = model(**inputs, use_cache=True)
layers = extract_kvcache(output)
# layers: [(K0, V0), (K1, V1), ..., (K21, V21)]
```

### Consolidate

Pack all K/V tensors into a contiguous staging buffer using `copy_()` (no `torch.cat`, no extra allocations). Returns a manifest describing offsets and shapes.

```python
total_bytes = kvcache_size_from_layers(layers)
staging = torch.empty(total_bytes, dtype=torch.uint8, device="cuda")
torch.cuda.synchronize()

packed_bytes, manifest = consolidate_kvcache(layers, staging)
# manifest: [{"layer": 0, "k_offset": 0, "k_size": 16384,
#              "v_offset": 16384, "v_size": 16384,
#              "shape": (1, 4, 32, 64), "dtype": torch.float16}, ...]
```

### Reconstruct (zero-copy)

Create K/V tensor views into the RDMA landing buffer using `torch.as_strided`. No memory copies.

```python
reconstructed = reconstruct_kvcache_zero_copy(recv_buf, manifest)
# reconstructed: tuple of (K, V) pairs, same format as past_key_values
```

### Size Estimation

Compute total KVCache bytes from a model config (handles GQA via `num_key_value_heads`):

```python
total = kvcache_size_estimate(model.config, seq_len=128, dtype=torch.float16)
```

## Complete Example: WRITEIMM Loopback

End-to-end KVCache transfer over RDMA loopback on a single machine:

```python
import torch
from dmaplane_py import (
    DmaplaneClient, allocate_aligned_gpu_tensor,
    kvcache_imm_encode, KVCACHE_SENTINEL,
    IB_ACCESS_LOCAL_WRITE, IB_ACCESS_REMOTE_WRITE,
)
from kvcache_inference import (
    extract_kvcache, consolidate_kvcache,
    reconstruct_kvcache_zero_copy, kvcache_size_from_layers,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

CHUNK_SIZE = 1024 * 1024

# 1. Prefill
model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
inputs = tokenizer("The capital of France is", return_tensors="pt").to("cuda")
with torch.no_grad():
    output = model(**inputs, use_cache=True)

# 2. Extract + consolidate KVCache
layers = extract_kvcache(output)
total_bytes = kvcache_size_from_layers(layers)
staging = allocate_aligned_gpu_tensor(total_bytes)
torch.cuda.synchronize()
packed_bytes, manifest = consolidate_kvcache(layers, staging)
torch.cuda.synchronize()

# 3. RDMA transfer (GPU MR -> host MR)
num_chunks = (packed_bytes + CHUNK_SIZE - 1) // CHUNK_SIZE

with DmaplaneClient() as cl:
    cl.setup_rdma(max_send_wr=num_chunks + 16, max_recv_wr=num_chunks + 16)

    # GPU source
    pin = cl.gpu_pin(staging.data_ptr(), staging.numel())
    gpu_mr = cl.gpu_register_mr(pin["handle"])

    # Host destination (fast-reg for REMOTE_WRITE)
    dst = cl.create_buffer(packed_bytes)
    dst_mr = cl.register_mr(dst["buf_id"],
                            IB_ACCESS_LOCAL_WRITE | IB_ACCESS_REMOTE_WRITE)

    # Pre-post recvs
    for i in range(num_chunks + 1):
        cl.post_recv(dst_mr["mr_id"], CHUNK_SIZE, use_peer_qp=0)

    # Send chunks
    for i in range(num_chunks):
        offset = i * CHUNK_SIZE
        length = min(CHUNK_SIZE, packed_bytes - offset)
        cl.write_imm(gpu_mr["mr_id"], offset,
                     dst_mr["addr"] + offset, dst_mr["rkey"],
                     length, kvcache_imm_encode(0, i))

    # Send sentinel + poll all completions
    cl.write_imm(gpu_mr["mr_id"], 0, dst_mr["addr"], dst_mr["rkey"],
                 4, KVCACHE_SENTINEL)
    for i in range(num_chunks + 1):
        cl.poll_recv(timeout_ms=5000)

    # 4. Reconstruct from host landing buffer
    dst_map = cl.mmap_buffer(dst["buf_id"])
    host_tensor = torch.frombuffer(bytearray(bytes(dst_map[:packed_bytes])),
                                   dtype=torch.uint8).cuda()
    reconstructed = reconstruct_kvcache_zero_copy(host_tensor, manifest)

    # 5. Decode with reconstructed KVCache
    next_token = torch.argmax(output.logits[:, -1, :], dim=-1, keepdim=True)
    with torch.no_grad():
        out = model(next_token, past_key_values=reconstructed, use_cache=True)
    print(tokenizer.decode(next_token[0]))  # "Paris"

    # Cleanup
    cl._munmap(dst["buf_id"])
    cl.gpu_unpin(pin["handle"])
    cl.deregister_mr(gpu_mr["mr_id"])
    cl.deregister_mr(dst_mr["mr_id"])
    cl.destroy_buffer(dst["buf_id"])
    cl.teardown_rdma()
```

## Struct Packing Reference

All format strings use `=` prefix (native byte order, standard sizes). Sizes match the kernel UAPI structs in `include/dmaplane_uapi.h` exactly.

| Struct | Format | Size | Ioctl |
|--------|--------|------|-------|
| `buf_params` | `=IIQii` | 24 | `CREATE_BUFFER` (0xc018e405) |
| `rdma_setup` | `=32sIIIII` | 52 | `SETUP_RDMA` (0xc034e410) |
| `mr_params` | `=IIIIIIQ` | 32 | `REGISTER_MR` (0xc020e420) |
| `mmap_info` | `=IIQQ` | 24 | `GET_MMAP_INFO` (0xc018e408) |
| `gpu_pin_params` | `=QQIiIIQ` | 40 | `GPU_PIN` (0xc028e460) |
| `gpu_unpin_params` | `=I` | 4 | `GPU_UNPIN` (0x4004e461) |
| `gpu_mr_params` | `=IIII` | 16 | `GPU_REGISTER_MR` (0xc010e465) |
| `write_imm_params` | `=IIQQIIIIQ` | 48 | `WRITE_IMM` (0xc030e480) |
| `post_recv_params` | `=IIII` | 16 | `POST_RECV` (0xc010e481) |
| `poll_recv_params` | `=IIIIIIQ` | 32 | `POLL_RECV` (0xc020e482) |

## Error Handling

All ioctl calls raise `OSError` with the ioctl name in the message:

```python
try:
    cl.destroy_buffer(999)
except OSError as e:
    print(e)  # [Errno 22] dmaplane DESTROY_BUFFER: Invalid argument
```

Common errors:
- `EBUSY (16)` — resource still in use (mmap open, MR registered)
- `EINVAL (22)` — invalid buf_id, mr_id, or parameter
- `ETIMEDOUT (110)` — `poll_recv` timed out (no completion arrived)
- `ENOMEM (12)` — kernel allocation failed
