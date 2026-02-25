# dmaplane

A Linux kernel module that implements the buffer orchestration layer for AI data transport from first principles.
dmaplane exposes a character device (`/dev/dmaplane`) controlled via ioctl, implementing the buffer orchestration layer that sits beneath transport libraries and includes: DMA allocation, NUMA-aware placement, RDMA memory registration, dma-buf zero-copy sharing, GPU BAR pinning, credit-based flow control, and disaggregated KVCache transfer. The project builds cumulatively across nine phases, each targeting a specific kernel subsystem.

## Build & Test

```bash
# Build the kernel module and test programs
make

# Load the module (requires Soft-RoCE for RDMA tests)
bash scripts/setup_rxe.sh            # configure Soft-RoCE (once)
sudo insmod driver/dmaplane.ko
ls -la /dev/dmaplane
dmesg | tail -5

# Run tests (Phases 1-8)
sudo ./tests/test_phase1_driver
sudo ./tests/test_phase2_dma
sudo ./tests/test_phase3_dmabuf
sudo ./tests/test_phase4_rdma         # requires Soft-RoCE
sudo ./tests/test_phase5_numa
sudo ./tests/test_phase6_backpressure # requires Soft-RoCE
sudo ./tests/test_phase7_instrumentation
sudo ./tests/test_phase8_gpu          # requires NVIDIA GPU

# Run Phase 9 KVCache tests
sudo ./examples/kvcache/kvcache_sender --loopback --verify
python3 examples/kvcache/dmaplane_py.py
sudo python3 examples/kvcache/test_dmaplane_py.py
sudo python3 examples/kvcache/test_kvcache_local.py  # requires CUDA + PyTorch

# Run examples
sudo ./examples/misc/dma_explorer
sudo ./examples/misc/dma_sweep
sudo ./examples/misc/trace_explorer   # Phase 7 tracing

# Unload the module
sudo rmmod dmaplane
dmesg | tail -5
```

## Directory Structure

```
dmaplane-master/
├── CLAUDE.md                  # Claude Code entry point
├── README.md                  # This file
├── LICENSE                    # GPL-2.0
├── Makefile                   # Top-level: delegates to driver/ and tests/
├── driver/                    # Kernel module source (single dmaplane.ko)
│   ├── dmaplane.h             # Kernel-internal header
│   ├── dmabuf_rdma.{h,c}      # Buffer allocation: coherent + page-backed
│   ├── dmabuf_export.{h,c}    # dma-buf export: dma_buf_ops, SG table
│   ├── rdma_engine.{h,c}      # RDMA setup/teardown, MR, WRITEIMM, peer QP
│   ├── benchmark.{h,c}        # Loopback, ping-pong, streaming benchmarks
│   ├── numa_topology.{h,c}    # Topology query, NxN bandwidth benchmark
│   ├── flow_control.{h,c}     # Credit-based backpressure, sustained streaming
│   ├── gpu_p2p.{h,c}          # GPU BAR pinning via nvidia_p2p, GPU MR
│   ├── dmaplane_histogram.{h,c} # Log₂ latency histogram (16 buckets)
│   ├── dmaplane_debugfs.{h,c} # debugfs counters and stats export
│   ├── dmaplane_trace.{h,c}   # Tracepoint definitions and event recording
│   └── main.c                 # Char device, ioctl dispatch, mmap, init/exit
├── include/
│   └── dmaplane_uapi.h       # Ioctl numbers, shared structs (kernel ↔ user)
├── tests/                     # One C test per phase (1-8)
│   ├── test_phase1_driver.c   # Rings, workers, stress test
│   ├── test_phase2_dma.c      # Buffer alloc, mmap, lifecycle
│   ├── test_phase3_dmabuf.c   # dma-buf export, fd lifecycle
│   ├── test_phase4_rdma.c     # RDMA setup, MR, loopback, benchmarks
│   ├── test_phase5_numa.c     # NUMA topology, allocation, benchmark
│   ├── test_phase6_backpressure.c # Credit exhaustion, sustained streaming
│   ├── test_phase7_instrumentation.c # Tracepoint, histogram, debugfs
│   └── test_phase8_gpu.c      # GPU pin/unpin, BAR DMA, GPU MR, loopback
├── examples/
│   ├── misc/                  # Standalone demos: dma_explorer, dma_sweep, trace_explorer
│   ├── gpu_rdma/              # Cross-machine GPU VRAM over RDMA (Phase 8)
│   ├── kvcache/               # KVCache WRITEIMM pipeline + inference servers (Phase 9)
│   ├── streamer/              # Weight-streaming TUI (Phase 6)
│   └── netshare/              # Cross-machine host RDMA via rdma_cm (Phase 4)
└── scripts/                   # setup_rxe.sh, setup_hugepages.sh
```

## License

GPL-2.0 — see [LICENSE](LICENSE).
