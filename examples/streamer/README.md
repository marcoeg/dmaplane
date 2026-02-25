# Weight Streaming TUI

Interactive terminal interface for simulating accelerator weight streaming
over RDMA. Models the host-side transfer patterns used by Cerebras weight
servers, GPU parameter servers (DeepSpeed ZeRO-3, FSDP), and gradient
aggregation fabrics (NCCL ring/tree allreduce).

## Prerequisites

```bash
sudo apt install libncurses-dev
```

Kernel module loaded and soft-RoCE configured:

```bash
bash scripts/setup_rxe.sh
sudo insmod driver/dmaplane.ko
```

## Build and Run

```bash
make -C examples/streamer
sudo ./examples/streamer/weight_streamer_tui
```

Auto-detects the first IB device from `/sys/class/infiniband`. To force a specific device:

```bash
sudo ./examples/streamer/weight_streamer_tui rxe_enp0s31f6
```

## Controls

| Key         | Action                                        |
|-------------|-----------------------------------------------|
| Up / Down   | Select parameter                              |
| Left / Right| Adjust selected parameter                     |
| Enter or r  | Run one epoch with current parameters         |
| c           | Continuous mode (auto-runs epochs back to back)|
| s           | Stop continuous mode                          |
| q           | Quit                                          |

## Screen Layout

The top panel shows four tuneable parameters with slider bars:

- **Model layers** — number of weight shards (2-32)
- **Shard size** — KB per layer, power-of-2 stepping (4-512)
- **Queue depth** — outstanding RDMA transfers in the pipeline (1-64)
- **Iters per layer** — repeat count per layer transfer (1-50)

Below the parameters, the latest epoch result appears as horizontal
bar charts comparing the three transfer phases. A history table at the
bottom accumulates every run.

## Results Table

Each row is one complete training epoch (forward + backward pass through all layers).

```
 Ep  Layers  Shard  QD  Seq(ms) Pipe(ms) Spdup  Bwd(ms)  Total
  1       8   64KB   4    13.2      4.8    2.8x    12.8    17.6
  2       8  128KB   4    19.5      6.1    3.2x    12.9    19.0
  3       8  128KB  16    19.4      3.2    6.1x    12.8    16.0
```

| Column | Description |
|--------|-------------|
| Seq(ms) | Forward pass, sequential transfers (naive baseline) |
| Pipe(ms) | Forward pass, pipelined transfers (production pattern) |
| Spdup | Seq / Pipe — pipeline speedup factor |
| Bwd(ms) | Backward pass, ping-pong gradient exchange (latency-bound) |
| Total | Pipe + Bwd — epoch wall-clock time |

## What to Try

- **Queue depth effect**: Fix layers=8, shard=64KB. Sweep QD from 1 to 64. Watch Pipe drop and Spdup climb.
- **Shard size effect**: Fix layers=8, QD=4. Step shard from 4KB to 512KB. Small shards are overhead-dominated.
- **Forward vs backward**: The backward pass barely changes with QD — it's ping-pong (latency x layers), the real bottleneck.

## Relation to Production Systems

| dmaplane concept       | Production equivalent                    |
|------------------------|------------------------------------------|
| Layer buffer + MR      | GPU VRAM region registered with NIC      |
| Sequential forward     | Naive parameter fetch (nobody does this) |
| Pipelined forward      | Cerebras weight streaming / FSDP prefetch|
| Backward ping-pong     | NCCL allreduce gradient aggregation      |
| Queue depth            | NCCL_NCHANNELS / pipeline depth          |
