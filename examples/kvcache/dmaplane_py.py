#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0
"""
dmaplane_py.py — Python wrapper for /dev/dmaplane ioctls

Uses struct.pack / fcntl.ioctl for zero-dependency access to the kernel
driver. All struct format strings use '=' prefix (native byte order,
standard alignment) and match the kernel UAPI structs exactly.

Struct sizes verified against dmaplane_uapi.h:
  buf_params:       24 bytes  =IIQii
  rdma_setup:       52 bytes  =32sIIIII
  mr_params:        32 bytes  =IIIIIIQ
  mmap_info:        24 bytes  =IIQQ
  gpu_pin_params:   40 bytes  =QQIiIIQ
  gpu_unpin_params:  4 bytes  =I
  gpu_mr_params:    16 bytes  =IIII
  write_imm_params: 48 bytes  =IIQQIIIIQ
  post_recv_params: 16 bytes  =IIII
  poll_recv_params: 32 bytes  =IIIIIIQ
  peer_info:        40 bytes  =IHH16s6s2sII
"""

import os
import struct
import fcntl
import mmap
import socket
import json
import gc

# ── ioctl encoding (matches <linux/ioctl.h>) ──────────────────────────────
#
# Linux ioctl command numbers encode direction, type (magic), number, and
# struct size into a 32-bit integer.  The layout is:
#   [31:30] direction (0=none, 1=write, 2=read, 3=read+write)
#   [29:16] struct size in bytes
#   [15:8]  magic number (0xE4 for dmaplane)
#   [7:0]   command number
#
# _IOWR means the kernel both reads and writes the struct (in-place update).

DMAPLANE_MAGIC = 0xE4

def _IOC(d, t, nr, sz):
    return (d << 30) | (sz << 16) | (t << 8) | nr

def _IO(t, nr):
    return _IOC(0, t, nr, 0)

def _IOR(t, nr, sz):
    return _IOC(2, t, nr, sz)

def _IOW(t, nr, sz):
    return _IOC(1, t, nr, sz)

def _IOWR(t, nr, sz):
    return _IOC(3, t, nr, sz)


# ── Struct format strings ─────────────────────────────────────────────────
#
# All format strings use '=' prefix: native byte order, standard sizes
# (no compiler padding).  These MUST match the kernel UAPI structs in
# include/dmaplane_uapi.h byte-for-byte.  The self-test at the bottom
# of this file validates sizes; test_dmaplane_py.py tests round-trips.

# struct dmaplane_buf_params { u32, u32, u64, s32, s32 } = 24
FMT_BUF_PARAMS = "=IIQii"
SZ_BUF_PARAMS = struct.calcsize(FMT_BUF_PARAMS)

# struct dmaplane_rdma_setup { char[32], u32, u32, u32, u32, u32 } = 52
FMT_RDMA_SETUP = "=32sIIIII"
SZ_RDMA_SETUP = struct.calcsize(FMT_RDMA_SETUP)

# struct dmaplane_mr_params { u32, u32, u32, u32, u32, u32, u64 } = 32
FMT_MR_PARAMS = "=IIIIIIQ"
SZ_MR_PARAMS = struct.calcsize(FMT_MR_PARAMS)

# struct dmaplane_mmap_info { u32, u32, u64, u64 } = 24
FMT_MMAP_INFO = "=IIQQ"
SZ_MMAP_INFO = struct.calcsize(FMT_MMAP_INFO)

# struct dmaplane_gpu_pin_params { u64, u64, u32, s32, u32, u32, u64 } = 40
FMT_GPU_PIN = "=QQIiIIQ"
SZ_GPU_PIN = struct.calcsize(FMT_GPU_PIN)

# struct dmaplane_gpu_unpin_params { u32 } = 4
FMT_GPU_UNPIN = "=I"
SZ_GPU_UNPIN = struct.calcsize(FMT_GPU_UNPIN)

# struct dmaplane_gpu_mr_params { u32, u32, u32, u32 } = 16
FMT_GPU_MR = "=IIII"
SZ_GPU_MR = struct.calcsize(FMT_GPU_MR)

# struct dmaplane_write_imm_params { u32, u32, u64, u64, u32, u32, u32, u32, u64 } = 48
FMT_WRITE_IMM = "=IIQQIIIIQ"
SZ_WRITE_IMM = struct.calcsize(FMT_WRITE_IMM)

# struct dmaplane_post_recv_params { u32, u32, u32, u32 } = 16
FMT_POST_RECV = "=IIII"
SZ_POST_RECV = struct.calcsize(FMT_POST_RECV)

# struct dmaplane_poll_recv_params { u32, u32, u32, u32, u32, u32, u64 } = 32
FMT_POLL_RECV = "=IIIIIIQ"
SZ_POLL_RECV = struct.calcsize(FMT_POLL_RECV)

# struct dmaplane_remote_op_params { u32, u32, u64, u32, u32, u32, u32 } = 32
FMT_REMOTE_OP = "=IIQIIII"
SZ_REMOTE_OP = struct.calcsize(FMT_REMOTE_OP)

# struct dmaplane_rdma_peer_info { u32, u16, u16, u8[16], u8[6], u8[2], u32, u32 } = 40
FMT_PEER_INFO = "=IHH16s6s2sII"
SZ_PEER_INFO = struct.calcsize(FMT_PEER_INFO)

# ── ioctl command numbers ─────────────────────────────────────────────────

IOCTL_CREATE_BUFFER     = _IOWR(DMAPLANE_MAGIC, 0x05, SZ_BUF_PARAMS)
IOCTL_DESTROY_BUFFER    = _IOW (DMAPLANE_MAGIC, 0x06, 4)
IOCTL_GET_MMAP_INFO     = _IOWR(DMAPLANE_MAGIC, 0x08, SZ_MMAP_INFO)

IOCTL_SETUP_RDMA        = _IOWR(DMAPLANE_MAGIC, 0x10, SZ_RDMA_SETUP)
IOCTL_TEARDOWN_RDMA     = _IO  (DMAPLANE_MAGIC, 0x11)

IOCTL_REGISTER_MR       = _IOWR(DMAPLANE_MAGIC, 0x20, SZ_MR_PARAMS)
IOCTL_DEREGISTER_MR     = _IOW (DMAPLANE_MAGIC, 0x21, 4)

IOCTL_GPU_PIN           = _IOWR(DMAPLANE_MAGIC, 0x60, SZ_GPU_PIN)
IOCTL_GPU_UNPIN         = _IOW (DMAPLANE_MAGIC, 0x61, SZ_GPU_UNPIN)
IOCTL_GPU_REGISTER_MR   = _IOWR(DMAPLANE_MAGIC, 0x65, SZ_GPU_MR)

IOCTL_RDMA_INIT_PEER    = _IOR (DMAPLANE_MAGIC, 0x90, SZ_PEER_INFO)
IOCTL_RDMA_CONNECT_PEER = _IOW (DMAPLANE_MAGIC, 0x91, SZ_PEER_INFO)
IOCTL_RDMA_DESTROY_PEER = _IO  (DMAPLANE_MAGIC, 0x94)

IOCTL_RDMA_WRITE_IMM    = _IOWR(DMAPLANE_MAGIC, 0x80, SZ_WRITE_IMM)
IOCTL_RDMA_POST_RECV    = _IOWR(DMAPLANE_MAGIC, 0x81, SZ_POST_RECV)
IOCTL_RDMA_POLL_RECV    = _IOWR(DMAPLANE_MAGIC, 0x82, SZ_POLL_RECV)

# ── IB access flags (mirror kernel values from <rdma/ib_verbs.h>) ─────────
#
# LOCAL_WRITE:  HCA can write to this MR (needed for recv buffers)
# REMOTE_WRITE: remote HCA can RDMA WRITE into this MR (needs fast-reg path)
# REMOTE_READ:  remote HCA can RDMA READ from this MR

IB_ACCESS_LOCAL_WRITE  = 1
IB_ACCESS_REMOTE_WRITE = 1 << 1
IB_ACCESS_REMOTE_READ  = 1 << 2

# ── Buffer types ──────────────────────────────────────────────────────────

BUF_TYPE_COHERENT = 0
BUF_TYPE_PAGES    = 1
NUMA_ANY          = -1

# ── KVCache IMM encoding (mirrors kvcache_proto.h) ───────────────────────
#
# The 32-bit immediate value delivered via RDMA WRITE WITH IMMEDIATE encodes
# a (layer_index, chunk_index) pair.  The receiver decodes it from the CQ
# completion to track which chunk arrived — no memory polling needed.
# Sentinel value 0xFFFFFFFF signals end-of-transfer.

KVCACHE_SENTINEL = 0xFFFFFFFF

def kvcache_imm_encode(layer, chunk):
    """Encode (layer, chunk) into a 32-bit immediate: upper 16 = layer, lower 16 = chunk."""
    return ((layer & 0xFFFF) << 16) | (chunk & 0xFFFF)

def kvcache_imm_layer(imm):
    """Extract layer index from 32-bit immediate."""
    return (imm >> 16) & 0xFFFF

def kvcache_imm_chunk(imm):
    """Extract chunk index from 32-bit immediate."""
    return imm & 0xFFFF

# ── RxE device detection ─────────────────────────────────────────────────

def find_rxe_device():
    """Scan /sys/class/infiniband for an rxe_* device. Returns name or None."""
    ib_dir = "/sys/class/infiniband"
    if not os.path.isdir(ib_dir):
        return None
    for name in os.listdir(ib_dir):
        if name.startswith("rxe_"):
            return name
    return None


# ── DmaplaneClient class ─────────────────────────────────────────────────

class DmaplaneClient:
    """Python ioctl interface to /dev/dmaplane.

    Each method packs a C struct via struct.pack, calls fcntl.ioctl (which
    mutates the bytearray in-place), then unpacks the result.  This mirrors
    what C code does with a stack-allocated struct + ioctl(fd, cmd, &s).

    Use as a context manager to ensure the fd and mmaps are cleaned up:
        with DmaplaneClient() as cl:
            cl.setup_rdma()
            ...
            cl.teardown_rdma()
    """

    def __init__(self, device="/dev/dmaplane"):
        self.fd = os.open(device, os.O_RDWR)
        self._mmaps = {}  # buf_id -> (mmap_obj, size) — tracked for cleanup

    def close(self):
        for buf_id in list(self._mmaps.keys()):
            self._munmap(buf_id)
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _ioctl(self, name, cmd, buf):
        """Thin wrapper that includes ioctl name in exceptions."""
        try:
            fcntl.ioctl(self.fd, cmd, buf)
        except OSError as e:
            raise OSError(e.errno,
                          f"dmaplane {name}: {os.strerror(e.errno)}") from e

    # ── Buffer management ─────────────────────────────────────────────

    def create_buffer(self, size, numa_node=NUMA_ANY, alloc_type=BUF_TYPE_PAGES):
        """Create a DMA buffer. Returns (buf_id, actual_numa_node)."""
        buf = bytearray(struct.pack(FMT_BUF_PARAMS,
                                    0, alloc_type, size, numa_node, 0))
        self._ioctl("CREATE_BUFFER", IOCTL_CREATE_BUFFER, buf)
        fields = struct.unpack(FMT_BUF_PARAMS, buf)
        return {"buf_id": fields[0], "actual_numa_node": fields[4]}

    def destroy_buffer(self, buf_id):
        """Destroy a DMA buffer."""
        buf = bytearray(struct.pack("=I", buf_id))
        self._ioctl("DESTROY_BUFFER", IOCTL_DESTROY_BUFFER, buf)

    def get_mmap_info(self, buf_id):
        """Get mmap offset and size for a buffer. Returns (offset, size)."""
        buf = bytearray(struct.pack(FMT_MMAP_INFO, buf_id, 0, 0, 0))
        self._ioctl("GET_MMAP_INFO", IOCTL_GET_MMAP_INFO, buf)
        fields = struct.unpack(FMT_MMAP_INFO, buf)
        return {"mmap_offset": fields[2], "mmap_size": fields[3]}

    def mmap_buffer(self, buf_id, prot=mmap.PROT_READ | mmap.PROT_WRITE):
        """mmap a buffer into this process. Returns mmap object."""
        info = self.get_mmap_info(buf_id)
        m = mmap.mmap(self.fd, info["mmap_size"],
                      prot=prot, flags=mmap.MAP_SHARED,
                      offset=info["mmap_offset"])
        self._mmaps[buf_id] = (m, info["mmap_size"])
        return m

    def _munmap(self, buf_id):
        if buf_id in self._mmaps:
            self._mmaps[buf_id][0].close()
            del self._mmaps[buf_id]

    # ── RDMA setup/teardown ───────────────────────────────────────────

    def setup_rdma(self, ib_dev=None, cq_depth=256, max_send_wr=64,
                   max_recv_wr=64):
        """Setup RDMA resources (PD, CQ, QP pair). Auto-detects rxe if ib_dev is None.

        Creates: protection domain, send + recv completion queues, and a
        loopback QP pair (QP-A for sends, QP-B for receives).  The port field
        (second u32) is set to 1 — IB port numbering starts at 1.
        """
        if ib_dev is None:
            ib_dev = find_rxe_device()
            if ib_dev is None:
                raise RuntimeError("No rxe device found")
        dev_bytes = ib_dev.encode("utf-8")[:32].ljust(32, b"\x00")
        buf = bytearray(struct.pack(FMT_RDMA_SETUP,
                                    dev_bytes, 1, cq_depth,
                                    max_send_wr, max_recv_wr, 0))
        self._ioctl("SETUP_RDMA", IOCTL_SETUP_RDMA, buf)

    def teardown_rdma(self):
        """Teardown RDMA resources."""
        fcntl.ioctl(self.fd, IOCTL_TEARDOWN_RDMA, 0)

    # ── MR management ─────────────────────────────────────────────────

    def register_mr(self, buf_id, access_flags=IB_ACCESS_LOCAL_WRITE):
        """Register an MR. Returns dict with mr_id, lkey, rkey, addr.

        If access_flags includes REMOTE_WRITE, the kernel uses the fast-reg
        path (ib_alloc_mr + IB_WR_REG_MR) and returns nonzero rkey + addr
        that the remote HCA needs for RDMA WRITE targeting.
        """
        buf = bytearray(struct.pack(FMT_MR_PARAMS,
                                    0, buf_id, access_flags, 0, 0, 0, 0))
        self._ioctl("REGISTER_MR", IOCTL_REGISTER_MR, buf)
        fields = struct.unpack(FMT_MR_PARAMS, buf)
        return {
            "mr_id": fields[0],
            "buf_id": fields[1],
            "access_flags": fields[2],
            "lkey": fields[3],
            "rkey": fields[4],
            "addr": fields[6],
        }

    def deregister_mr(self, mr_id):
        """Deregister an MR."""
        buf = bytearray(struct.pack("=I", mr_id))
        self._ioctl("DEREGISTER_MR", IOCTL_DEREGISTER_MR, buf)

    # ── GPU P2P ───────────────────────────────────────────────────────

    def gpu_pin(self, gpu_va, size):
        """Pin GPU memory. Returns dict with handle, gpu_numa_node, etc."""
        buf = bytearray(struct.pack(FMT_GPU_PIN,
                                    gpu_va, size, 0, 0, 0, 0, 0))
        self._ioctl("GPU_PIN", IOCTL_GPU_PIN, buf)
        fields = struct.unpack(FMT_GPU_PIN, buf)
        return {
            "gpu_va": fields[0],
            "size": fields[1],
            "handle": fields[2],
            "gpu_numa_node": fields[3],
            "num_pages": fields[4],
            "bar1_consumed": fields[6],
        }

    def gpu_unpin(self, handle):
        """Unpin GPU memory."""
        buf = bytearray(struct.pack(FMT_GPU_UNPIN, handle))
        self._ioctl("GPU_UNPIN", IOCTL_GPU_UNPIN, buf)

    def gpu_register_mr(self, gpu_handle):
        """Register a GPU MR. Returns dict with mr_id, lkey, rkey."""
        buf = bytearray(struct.pack(FMT_GPU_MR, gpu_handle, 0, 0, 0))
        self._ioctl("GPU_REGISTER_MR", IOCTL_GPU_REGISTER_MR, buf)
        fields = struct.unpack(FMT_GPU_MR, buf)
        return {
            "gpu_handle": fields[0],
            "mr_id": fields[1],
            "lkey": fields[2],
            "rkey": fields[3],
        }

    # ── WRITEIMM operations ───────────────────────────────────────────

    def write_imm(self, local_mr_id, local_offset, remote_addr, remote_rkey,
                  length, imm_data, use_peer_qp=0):
        """RDMA WRITE with immediate data. Returns elapsed_ns.

        Posts IB_WR_RDMA_WRITE_WITH_IMM: copies 'length' bytes from
        local_mr[local_offset] to remote_addr, and delivers imm_data
        through the receiver's completion queue.  A matching recv WR
        must be posted before calling this — otherwise the HCA returns
        RNR (Receiver Not Ready) and the send times out.
        """
        buf = bytearray(struct.pack(FMT_WRITE_IMM,
                                    local_mr_id, length, local_offset,
                                    remote_addr, remote_rkey, imm_data,
                                    use_peer_qp, 0, 0))
        self._ioctl("WRITE_IMM", IOCTL_RDMA_WRITE_IMM, buf)
        fields = struct.unpack(FMT_WRITE_IMM, buf)
        status = fields[7]
        if status != 0:
            raise OSError(5, f"dmaplane WRITE_IMM: WC status {status}")
        return fields[8]  # elapsed_ns

    def post_recv(self, mr_id, size, use_peer_qp=0):
        """Post a recv WR — required before each write_imm from the sender.

        The recv WR tells the HCA where to deliver the immediate data
        completion.  Without a posted recv, RDMA WRITE WITH IMM triggers
        RNR on the receiving QP.
        """
        buf = bytearray(struct.pack(FMT_POST_RECV,
                                    mr_id, size, use_peer_qp, 0))
        self._ioctl("POST_RECV", IOCTL_RDMA_POST_RECV, buf)
        fields = struct.unpack(FMT_POST_RECV, buf)
        if fields[3] != 0:
            raise OSError(5, f"dmaplane POST_RECV: status {fields[3]}")

    def poll_recv(self, use_peer_qp=0, timeout_ms=10000):
        """Poll recv CQ for a WRITEIMM completion.

        Blocks up to timeout_ms.  On success, returns imm_data (the 32-bit
        immediate from the sender) and byte_len (RDMA payload size).
        Raises OSError(ETIMEDOUT) if no completion arrives.
        """
        buf = bytearray(struct.pack(FMT_POLL_RECV,
                                    use_peer_qp, timeout_ms, 0, 0, 0, 0, 0))
        self._ioctl("POLL_RECV", IOCTL_RDMA_POLL_RECV, buf)
        fields = struct.unpack(FMT_POLL_RECV, buf)
        return {
            "status": fields[2],
            "wc_flags": fields[3],
            "imm_data": fields[4],
            "byte_len": fields[5],
            "elapsed_ns": fields[6],
        }

    # ── Peer RDMA (cross-machine QP) ─────────────────────────────────

    def init_peer(self):
        """Initialize peer QP. Returns local metadata for TCP exchange.

        Creates qp_peer + cq_peer and returns this machine's QPN, LID,
        GID, and MAC.  Send these to the remote side via TCP, then call
        connect_peer() with the remote side's metadata.
        """
        buf = bytearray(SZ_PEER_INFO)
        self._ioctl("INIT_PEER", IOCTL_RDMA_INIT_PEER, buf)
        fields = struct.unpack(FMT_PEER_INFO, buf)
        return {
            "qp_num": fields[0],
            "lid": fields[1],
            "gid": fields[3],     # 16-byte raw GID
            "mac": fields[4],     # 6-byte raw MAC
        }

    def connect_peer(self, qp_num, lid, gid, mac):
        """Connect peer QP using remote metadata from TCP exchange.

        Transitions qp_peer through INIT→RTR→RTS using the remote
        side's QPN, LID, GID, and MAC from init_peer().
        """
        buf = bytearray(struct.pack(FMT_PEER_INFO,
                                    qp_num, lid, 0,  # _pad1
                                    gid, mac, b'\x00\x00',  # _pad2
                                    0, 0))  # status, _pad3
        self._ioctl("CONNECT_PEER", IOCTL_RDMA_CONNECT_PEER, buf)

    def destroy_peer(self):
        """Destroy peer QP and CQ."""
        fcntl.ioctl(self.fd, IOCTL_RDMA_DESTROY_PEER, 0)


# ── TCP metadata exchange (matches C struct tcp_metadata) ─────────────────
#
# Binary format for QP/MR/config exchange between Python servers and C
# binaries.  The 4x padding after qpn (u32) matches the natural alignment
# hole before mr_addr (u64) in the C struct on x86_64.
#
# struct tcp_metadata {
#     uint8_t  gid[16];           offset  0
#     uint8_t  mac[6];            offset 16
#     uint16_t lid;               offset 22
#     uint32_t qpn;               offset 24
#     /* 4 bytes implicit pad */  offset 28
#     uint64_t mr_addr;           offset 32
#     uint32_t mr_rkey;           offset 40
#     uint32_t num_layers;        offset 44
#     uint32_t chunks_per_layer;  offset 48
#     uint32_t chunk_size;        offset 52
#     uint32_t credit_window;     offset 56
#     uint32_t _pad;              offset 60
#     uint64_t buf_size;          offset 64
# };  /* total: 72 bytes */

TCP_METADATA_FMT = "=16s6sHI4xQIIIIIIQ"
TCP_METADATA_SZ = struct.calcsize(TCP_METADATA_FMT)  # 72


def tcp_send_all(sock, data):
    """Send all bytes, handling partial writes."""
    mv = memoryview(data)
    total = 0
    while total < len(data):
        sent = sock.send(mv[total:])
        if sent == 0:
            raise ConnectionError("TCP connection closed")
        total += sent


def tcp_recv_all(sock, nbytes):
    """Receive exactly nbytes, handling partial reads."""
    chunks = []
    received = 0
    while received < nbytes:
        chunk = sock.recv(nbytes - received)
        if not chunk:
            raise ConnectionError("TCP connection closed")
        chunks.append(chunk)
        received += len(chunk)
    return b''.join(chunks)


def tcp_pack_metadata(gid, mac, lid, qpn, mr_addr, mr_rkey,
                      num_layers, chunks_per_layer, chunk_size,
                      credit_window, buf_size):
    """Pack tcp_metadata struct for wire transmission."""
    return struct.pack(TCP_METADATA_FMT,
                       gid, mac, lid, qpn, mr_addr, mr_rkey,
                       num_layers, chunks_per_layer, chunk_size,
                       credit_window, 0, buf_size)


def tcp_unpack_metadata(data):
    """Unpack tcp_metadata struct from wire bytes."""
    fields = struct.unpack(TCP_METADATA_FMT, data)
    return {
        "gid": fields[0],
        "mac": fields[1],
        "lid": fields[2],
        "qpn": fields[3],
        "mr_addr": fields[4],
        "mr_rkey": fields[5],
        "num_layers": fields[6],
        "chunks_per_layer": fields[7],
        "chunk_size": fields[8],
        "credit_window": fields[9],
        "buf_size": fields[11],
    }


def tcp_send_metadata(sock, **kwargs):
    """Pack and send tcp_metadata over TCP."""
    data = tcp_pack_metadata(**kwargs)
    tcp_send_all(sock, data)


def tcp_recv_metadata(sock):
    """Receive and unpack tcp_metadata from TCP."""
    data = tcp_recv_all(sock, TCP_METADATA_SZ)
    return tcp_unpack_metadata(data)


def tcp_send_json(sock, obj):
    """Send a JSON object with 4-byte big-endian length prefix."""
    payload = json.dumps(obj).encode('utf-8')
    tcp_send_all(sock, struct.pack('!I', len(payload)))
    tcp_send_all(sock, payload)


def tcp_recv_json(sock):
    """Receive a JSON object with 4-byte big-endian length prefix."""
    hdr = tcp_recv_all(sock, 4)
    length = struct.unpack('!I', hdr)[0]
    payload = tcp_recv_all(sock, length)
    return json.loads(payload.decode('utf-8'))


# ── GPU tensor alignment helper (for PyTorch) ────────────────────────────

def allocate_aligned_gpu_tensor(size_bytes, alignment=65536):
    """
    Allocate a GPU tensor with guaranteed 64KB-aligned data_ptr.

    NVIDIA P2P API requires 64KB-aligned VA and size for gpu_pin.
    cudaMalloc usually returns 256B-aligned addresses, so we
    over-allocate and slice to the aligned boundary.

    Returns (aligned_tensor, raw_storage_ref) — keep raw_storage_ref
    alive to prevent GC from freeing the underlying storage.
    """
    try:
        import torch
    except ImportError:
        raise RuntimeError("PyTorch required for allocate_aligned_gpu_tensor")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    # Round up size to alignment boundary
    aligned_size = (size_bytes + alignment - 1) & ~(alignment - 1)

    # Over-allocate to guarantee we can find an aligned region
    raw = torch.empty(aligned_size + alignment, dtype=torch.uint8,
                      device="cuda")
    raw_ptr = raw.data_ptr()

    # Find the next aligned address
    offset = (alignment - (raw_ptr % alignment)) % alignment
    aligned = raw[offset:offset + aligned_size]

    # Verify alignment
    assert aligned.data_ptr() % alignment == 0, \
        f"Alignment failed: data_ptr=0x{aligned.data_ptr():x}"

    # Keep a reference to the raw storage so GC doesn't free it
    aligned._raw_storage_ref = raw

    return aligned


# ── Self-test (run as script) ─────────────────────────────────────────────

if __name__ == "__main__":
    print("dmaplane_py.py — struct size validation")
    expected = {
        "buf_params": (FMT_BUF_PARAMS, 24),
        "rdma_setup": (FMT_RDMA_SETUP, 52),
        "mr_params": (FMT_MR_PARAMS, 32),
        "mmap_info": (FMT_MMAP_INFO, 24),
        "gpu_pin_params": (FMT_GPU_PIN, 40),
        "gpu_unpin_params": (FMT_GPU_UNPIN, 4),
        "gpu_mr_params": (FMT_GPU_MR, 16),
        "write_imm_params": (FMT_WRITE_IMM, 48),
        "post_recv_params": (FMT_POST_RECV, 16),
        "poll_recv_params": (FMT_POLL_RECV, 32),
        "remote_op_params": (FMT_REMOTE_OP, 32),
        "peer_info": (FMT_PEER_INFO, 40),
        "tcp_metadata": (TCP_METADATA_FMT, 72),
    }
    all_ok = True
    for name, (fmt, exp_sz) in expected.items():
        actual = struct.calcsize(fmt)
        status = "OK" if actual == exp_sz else "FAIL"
        if actual != exp_sz:
            all_ok = False
        print(f"  {name:20s} fmt={fmt:12s}  size={actual:3d}  expected={exp_sz:3d}  [{status}]")

    print(f"\nAll struct sizes correct: {'YES' if all_ok else 'NO'}")

    print("\nIoctl command numbers:")
    for name, val in [
        ("CREATE_BUFFER", IOCTL_CREATE_BUFFER),
        ("DESTROY_BUFFER", IOCTL_DESTROY_BUFFER),
        ("GET_MMAP_INFO", IOCTL_GET_MMAP_INFO),
        ("SETUP_RDMA", IOCTL_SETUP_RDMA),
        ("TEARDOWN_RDMA", IOCTL_TEARDOWN_RDMA),
        ("REGISTER_MR", IOCTL_REGISTER_MR),
        ("DEREGISTER_MR", IOCTL_DEREGISTER_MR),
        ("INIT_PEER", IOCTL_RDMA_INIT_PEER),
        ("CONNECT_PEER", IOCTL_RDMA_CONNECT_PEER),
        ("DESTROY_PEER", IOCTL_RDMA_DESTROY_PEER),
        ("WRITE_IMM", IOCTL_RDMA_WRITE_IMM),
        ("POST_RECV", IOCTL_RDMA_POST_RECV),
        ("POLL_RECV", IOCTL_RDMA_POLL_RECV),
    ]:
        print(f"  {name:20s} 0x{val:08x}")
