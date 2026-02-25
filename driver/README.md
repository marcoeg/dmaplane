# dmaplane Driver

Character device kernel module at `/dev/dmaplane`. Ioctl-driven, with per-channel submission/completion rings, worker threads, DMA buffer management, RDMA engine, NUMA topology, flow control, GPU P2P, WRITEIMM, and peer RDMA.

## Source Files

| File | Phase | Description |
|------|-------|-------------|
| `main.c` | 1+ | Char device, ioctl dispatch, mmap, module init/exit |
| `dmaplane.h` | 1+ | Kernel-internal header: structs, constants |
| `dmabuf_rdma.{h,c}` | 2 | Buffer allocation: coherent + page-backed, NUMA placement |
| `dmabuf_export.{h,c}` | 3 | dma-buf export: dma_buf_ops, SG table, selftest |
| `rdma_engine.{h,c}` | 4+ | RDMA setup/teardown, MR registration, post/poll, WRITEIMM, peer QP |
| `benchmark.{h,c}` | 4 | Loopback, ping-pong, streaming benchmarks |
| `numa_topology.{h,c}` | 5 | Topology query, NxN bandwidth benchmark |
| `flow_control.{h,c}` | 6 | Credit-based backpressure, sustained streaming, queue depth sweep |
| `dmaplane_histogram.{h,c}` | 7 | Log₂ latency histogram (16 buckets, all atomic64_t) |
| `dmaplane_debugfs.{h,c}` | 7 | debugfs counters and stats export |
| `dmaplane_trace.{h,c}` | 7 | Tracepoint definitions and event recording |
| `gpu_p2p.{h,c}` | 8 | GPU BAR pinning via nvidia_p2p, GPU MR, per-page ioremap_wc |

## Ioctl Reference

| Command | Nr | Direction | Struct | Description |
|---------|----|-----------|--------|-------------|
| `CREATE_CHANNEL` | 0x01 | `_IOR` | `channel_params` | Allocate a channel and assign to this fd |
| `SUBMIT` | 0x02 | `_IOW` | `submit_params` | Push a work item onto the submission ring |
| `COMPLETE` | 0x03 | `_IOWR` | `complete_params` | Pop a completion from the completion ring |
| `GET_STATS` | 0x04 | `_IOR` | `stats` | Return channel statistics |
| `CREATE_BUFFER` | 0x05 | `_IOWR` | `buf_params` | Allocate a DMA buffer (coherent or page-backed) |
| `DESTROY_BUFFER` | 0x06 | `_IOW` | `__u32` | Destroy a buffer by handle |
| `GET_MMAP_INFO` | 0x08 | `_IOWR` | `mmap_info` | Get mmap offset and size for a buffer |
| `GET_BUF_STATS` | 0x09 | `_IOR` | `buf_stats` | Return buffer allocation statistics |
| `EXPORT_DMABUF` | 0x0A | `_IOWR` | `export_dmabuf_arg` | Export a page-backed buffer as a dma-buf fd |
| `GET_DMABUF_STATS` | 0x0B | `_IOR` | `dmabuf_stats` | Return dma-buf export statistics |
| `SETUP_RDMA` | 0x10 | `_IOWR` | `rdma_setup` | Initialize RDMA: IB device -> PD -> CQs -> QPs |
| `TEARDOWN_RDMA` | 0x11 | `_IO` | -- | Tear down RDMA subsystem |
| `REGISTER_MR` | 0x20 | `_IOWR` | `mr_params` | Register buffer pages as RDMA MR |
| `DEREGISTER_MR` | 0x21 | `_IOW` | `__u32` | Deregister an MR by ID |
| `LOOPBACK_TEST` | 0x30 | `_IOWR` | `loopback_params` | Single send/recv loopback test |
| `PINGPONG_BENCH` | 0x31 | `_IOWR` | `bench_params` | Ping-pong latency benchmark |
| `STREAMING_BENCH` | 0x32 | `_IOWR` | `bench_params` | Streaming throughput benchmark |
| `GET_RDMA_STATS` | 0x33 | `_IOR` | `rdma_stats` | Return RDMA statistics |
| `CONFIGURE_FLOW` | 0x40 | `_IOWR` | `flow_params` | Set credit limits and watermarks |
| `SUSTAINED_STREAM` | 0x41 | `_IOWR` | `sustained_params` | Timed sustained streaming with backpressure |
| `QDEPTH_SWEEP` | 0x42 | `_IOWR` | `sweep_params` | Queue depth sweep (saturation detection) |
| `GET_FLOW_STATS` | 0x43 | `_IOR` | `flow_stats` | Return flow control statistics |
| `QUERY_NUMA_TOPO` | 0x50 | `_IOR` | `numa_topo` | Query NUMA topology: nodes, CPUs, memory, distances |
| `NUMA_BENCH` | 0x51 | `_IOWR` | `numa_bench_params` | Run NxN cross-node bandwidth benchmark |
| `GPU_PIN` | 0x60 | `_IOWR` | `gpu_pin_params` | Pin GPU VRAM via nvidia_p2p_get_pages |
| `GPU_UNPIN` | 0x61 | `_IOW` | `gpu_unpin_params` | Unpin GPU VRAM |
| `GPU_REGISTER_MR` | 0x65 | `_IOWR` | `gpu_mr_params` | Register GPU-backed MR from pinned pages |
| `GET_HISTOGRAM` | 0x70 | `_IOWR` | `hist_params` | Read/reset latency histogram |
| `RDMA_WRITE_IMM` | 0x80 | `_IOWR` | `write_imm_params` | RDMA WRITE with 32-bit immediate data |
| `RDMA_POST_RECV` | 0x81 | `_IOWR` | `post_recv_params` | Post recv WR (required before WRITEIMM) |
| `RDMA_POLL_RECV` | 0x82 | `_IOWR` | `poll_recv_params` | Poll recv CQ for WRITEIMM completion |
| `RDMA_INIT_PEER` | 0x90 | `_IOR` | `peer_info` | Create peer QP + CQ, return local metadata |
| `RDMA_CONNECT_PEER` | 0x91 | `_IOW` | `peer_info` | Connect peer QP using remote metadata |
| `RDMA_REMOTE_SEND` | 0x92 | `_IOWR` | `remote_xfer_params` | Post SEND on peer QP |
| `RDMA_REMOTE_RECV` | 0x93 | `_IOWR` | `remote_xfer_params` | Post RECV on peer QP |
| `RDMA_DESTROY_PEER` | 0x94 | `_IO` | -- | Destroy peer QP and CQ |

## Concurrency Model

| Lock | Type | Protects | Acquired by |
|------|------|----------|-------------|
| `sub_ring.lock` | spinlock | Submission ring entries, head/tail | ioctl submit (producer), worker thread (consumer) |
| `comp_ring.lock` | spinlock | Completion ring entries, head/tail | Worker thread (producer), ioctl complete (consumer) |
| `channel->lock` | mutex | Channel state (shutdown flag, worker pointer) | dmaplane_release |
| `dev_mutex` | mutex | Channel slot allocation array | ioctl create_channel, release |
| `buf_lock` | mutex | Buffer array (slots, lookup, mmap) | buffer ioctls, dmaplane_mmap |
| `rdma_sem` | rw_semaphore | RDMA context (PD, CQs, QPs) | read: MR/benchmark ops; write: setup/teardown |
| `mr_lock` | mutex | MR array (slots, lookup) | register/deregister MR, benchmarks |
| `flow.credits` | atomic_t | In-flight credit counter | flow_control send loop |
| `rdma_hist.*` | atomic64_t | Histogram bucket counters | rdma_engine post/poll |

**Lock ordering:** `dev_mutex` and `buf_lock` are independent (never nested). `rdma_sem` (read) -> `buf_lock` -> `mr_lock`. `dev_mutex` -> `channel->lock` -> ring spinlocks.

## Loading / Unloading

```bash
sudo insmod dmaplane.ko          # Load
ls -la /dev/dmaplane              # Verify device node
dmesg | grep dmaplane             # Check init message
sudo rmmod dmaplane               # Unload
dmesg | grep dmaplane             # Check exit message
```
