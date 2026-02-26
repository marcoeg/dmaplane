# Tests

One C test program per subsystem (1-8). Additional Python tests are in `examples/kvcache/`.

| Test | Description |
|------|-------------|
| `test_phase1_driver` | Character device, ioctl, rings, worker threads, stress test |
| `test_phase2_dma` | DMA buffer alloc (coherent + pages), mmap, lifecycle, stress |
| `test_phase3_dmabuf` | dma-buf export, fd lifecycle, mmap via dma-buf, destroy guard |
| `test_phase4_rdma` | RDMA setup, MR registration, loopback, ping-pong, streaming |
| `test_phase5_numa` | NUMA topology, node allocation, bandwidth benchmark |
| `test_phase6_backpressure` | Credit exhaustion, sustained streaming, queue depth sweep |
| `test_phase7_instrumentation` | Tracepoint validation, latency histogram, debugfs counters |
| `test_phase8_gpu` | GPU pin/unpin, BAR DMA throughput, GPU MR, RDMA loopback |

## Running Tests

```bash
make -C tests                           # Build test programs
sudo insmod driver/dmaplane.ko          # Load the module
bash scripts/setup_rxe.sh               # Configure Soft-RoCE (if not already)

# Run all tests
sudo ./tests/test_phase1_driver
sudo ./tests/test_phase2_dma
sudo ./tests/test_phase3_dmabuf
sudo ./tests/test_phase4_rdma           # requires Soft-RoCE
sudo ./tests/test_phase5_numa
sudo ./tests/test_phase6_backpressure   # requires Soft-RoCE
sudo ./tests/test_phase7_instrumentation
sudo ./tests/test_phase8_gpu            # requires NVIDIA GPU

# KVCache tests (Python, in examples/kvcache/)
python3 examples/kvcache/dmaplane_py.py                # struct validation (no root)
sudo python3 examples/kvcache/test_dmaplane_py.py      # ioctl round-trips
sudo python3 examples/kvcache/test_kvcache_local.py    # end-to-end inference (CUDA + PyTorch)
sudo ./examples/kvcache/kvcache_sender --loopback --verify  # C WRITEIMM loopback

sudo rmmod dmaplane                     # Unload
dmesg | tail -20                        # Check for warnings
```
