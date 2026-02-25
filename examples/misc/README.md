# Misc Examples

Standalone demos for DMA allocation and profiling.

| Program | Phase | Description |
|---------|-------|-------------|
| `dma_explorer` | 2 | Creates one buffer of each type (coherent, page-backed), mmaps, writes/reads, prints details |
| `dma_sweep` | 2 | Page-backed buffers from 4 KB to 64 MB, times the full create/mmap/write/munmap/destroy cycle |
| `trace_explorer` | 7 | Enables a tracepoint, captures events, runs RDMA pingpong, reads latency histogram and debugfs |

## Running

```bash
make                                    # Build (from repo root)
sudo insmod driver/dmaplane.ko          # Load the module
sudo ./examples/misc/dma_explorer       # Interactive exploration
sudo ./examples/misc/dma_sweep          # Size sweep benchmark
sudo rmmod dmaplane                     # Unload
```
