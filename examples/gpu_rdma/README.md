# GPU RDMA Examples

Cross-machine demo sending GPU VRAM over RDMA to a remote host without GPU.

| Program | Machine | Description |
|---------|---------|-------------|
| `gpu_sender` | A (GPU host) | Pins GPU VRAM, registers as RDMA MR, sends via peer QP over rxe |
| `gpu_receiver` | B (no GPU) | Receives into host buffer, verifies gradient data |

## Prerequisites

- `dmaplane.ko` loaded on BOTH machines
- Soft-RoCE (rxe) configured on both machines (see below)
- CUDA runtime on Machine A (sender) for `cudaMalloc`
- Real network connectivity between machines (same subnet for rxe)

## Setup Soft-RoCE

Before running either the sender or the receiver, configure Soft-RoCE on
**each machine**. The `setup_rxe.sh` script auto-detects the default
network interface:

```bash
# On both machines:
bash scripts/setup_rxe.sh
```

Or specify the interface explicitly:

```bash
# Machine A (GPU host, e.g., interface enp0s31f6):
bash scripts/setup_rxe.sh enp0s31f6     # creates rxe_enp0s31f6

# Machine B (NUC, no GPU, e.g., interface enp44s0):
bash scripts/setup_rxe.sh enp44s0       # creates rxe_enp44s0
```

Verify with `rdma link show` — you should see an `rxe_<iface>` device on
each machine.

## Usage

Receiver starts FIRST. The sender connects to it.

```bash
# Machine B (NUC, receiver — start FIRST):
sudo ./gpu_receiver 9876
# auto-detects rxe device, or specify explicitly:
sudo ./gpu_receiver 9876 rxe_enp44s0

# Machine A (GPU host, sender — start AFTER receiver):
sudo ./gpu_sender 192.168.50.17 9876
# auto-detects rxe device, or specify explicitly:
sudo ./gpu_sender 192.168.50.17 9876 rxe_enp0s31f6
```

## Note

Single-machine loopback does NOT work -- RDMA context is a device singleton.
Use `test_phase8_gpu` test 7 (GPU RDMA loopback) for single-machine validation.

## Architecture

```
Machine A (GPU)              Machine B (no GPU)
┌─────────────┐              ┌─────────────┐
│ GPU VRAM    │              │ Host DRAM   │
│ (cudaMalloc)│              │ (page-backed)│
└──────┬──────┘              └──────┬──────┘
       │ nvidia_p2p_get_pages        │ alloc_pages
       ▼                             ▼
┌─────────────┐              ┌─────────────┐
│ BAR1 pages  │              │ Host buffer │
│ (ioremap_wc)│              │ (vmap)      │
└──────┬──────┘              └──────┬──────┘
       │ GPU MR (sge_addr=WC VA)    │ Host MR
       ▼                             ▼
┌─────────────┐   Ethernet   ┌─────────────┐
│ qp_peer SEND├──────────────►qp_peer RECV │
└─────────────┘   (rxe/UDP)  └─────────────┘
```

## Building

```bash
make                                    # From repo root
make -C examples/gpu_rdma               # Or directly
```
