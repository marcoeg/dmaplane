# Netshare Example (Cross-Machine RDMA)

Host-path cross-machine gradient transfer using `rdma_cm` + `ibv_reg_mr` on mmap'd kernel DMA pages. Both sides need libibverbs and librdmacm. Dual-MR architecture — same physical pages carry a kernel MR (for DMA plumbing) and a userspace MR (for the wire).

For the GPU path (kernel-side peer QP, no libibverbs), see [`examples/gpu_rdma/`](../gpu_rdma/).

## Prerequisites

```bash
sudo apt install rdma-core ibverbs-utils libibverbs-dev librdmacm-dev
```

Soft-RoCE and dmaplane loaded on both machines:

```bash
bash scripts/setup_rxe.sh
sudo insmod driver/dmaplane.ko
```

## Build

```bash
make -C examples/netshare
```

## Run

Start receiver first (Machine B):

```bash
sudo ./examples/netshare/receiver <port> <ib_device>
```

Then start sender (Machine A):

```bash
sudo ./examples/netshare/sender <receiver-ip> <port> <ib_device>
```

Example:

```bash
# Machine B
sudo ./examples/netshare/receiver 7471 rxe_enp0s31f6

# Machine A
sudo ./examples/netshare/sender 192.168.1.50 7471 rxe_enp0s31f6
```

Find your IB device name with `ls /sys/class/infiniband`.

## Operational Sequence

### Sender (Machine A)

```
[1] open("/dev/dmaplane")
[2] IOCTL_CREATE_BUFFER          16 MB, BUF_TYPE_PAGES -> buf_id
[3] IOCTL_SETUP_RDMA             loopback QP pair on IB device
[4] IOCTL_REGISTER_MR            kernel-side MR -> mr_id, lkey, rkey
[5] IOCTL_GET_MMAP_INFO          -> mmap_offset, mmap_size
    mmap(fd, mmap_offset)        -> userspace pointer to DMA pages
    fill_gradient()              deterministic sinf pattern, layer=42
[6] rdma_cm resolve + connect    connect to receiver's IP:port
    ibv_reg_mr(pd, mmap_ptr)     register mmap'd pages for rdma_cm QP
    usleep(100ms)                wait for receiver to post recv
[7] ibv_post_send                IBV_WR_SEND, 16 MB in one WR
    ibv_poll_cq                  wait for send completion
[8] IOCTL_GET_STATS              print kernel module counters
    cleanup                      munmap -> dereg MR -> teardown RDMA
                                 -> destroy buffer -> close(fd)
```

### Receiver (Machine B)

```
[1] open("/dev/dmaplane")
[2] IOCTL_CREATE_BUFFER          16 MB, BUF_TYPE_PAGES -> buf_id
[3] IOCTL_SETUP_RDMA             loopback QP pair on IB device
[4] IOCTL_REGISTER_MR            kernel-side MR -> mr_id, lkey, rkey
[5] IOCTL_GET_MMAP_INFO          -> mmap_offset, mmap_size
    mmap(fd, mmap_offset)        -> userspace pointer to DMA pages
    memset(0)                    clear receive buffer
[6] rdma_cm listen + accept      wait for sender's connection
    ibv_reg_mr(pd, mmap_ptr)     register mmap'd pages for rdma_cm QP
[7] ibv_post_recv                post 16 MB receive WR
    ibv_poll_cq                  wait for recv completion
[8] verify_gradient()            sample 1000 floats, tolerance 1e-5
[9] IOCTL_GET_STATS              print kernel module counters
    cleanup                      munmap -> dereg MR -> teardown RDMA
                                 -> destroy buffer -> close(fd)
```

## Dual-MR Architecture

Each side registers the same physical pages twice:

1. **Kernel MR** (`IOCTL_REGISTER_MR`): registered against dmaplane's internal PD. Validates the buffer is properly DMA-mapped.

2. **Userspace MR** (`ibv_reg_mr`): registered against the `rdma_cm` connection's PD. Used for the cross-machine transfer. The mmap'd pages are kernel DMA pages (`VM_DONTCOPY | VM_DONTEXPAND`), so the NIC DMAs directly into/from them.

```
┌──────────────────────────────────────────────────┐
│                Physical DMA Pages                │
│         (allocated by dmaplane.ko)               │
├──────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌──────────────────────┐ │
│  │ Kernel MR       │    │ Userspace MR         │ │
│  │ (ioctl PD)      │    │ (rdma_cm PD)         │ │
│  │ for benchmarks  │    │ for cross-machine    │ │
│  └─────────────────┘    └──────────────────────┘ │
│  ┌──────────────────────────────────────────┐    │
│  │ mmap'd pointer (userspace)               │    │
│  │ fill_gradient() / verify_gradient()      │    │
│  └──────────────────────────────────────────┘    │
└──────────────────────────────────────────────────┘
```

## Gradient Verification

The sender fills the buffer with `scale * sinf(i * 0.001f + layer)` (scale = 1/(1+layer)). The receiver samples 1000 evenly-spaced floats and compares with tolerance `1e-5f`.
