#!/bin/bash
# SPDX-License-Identifier: GPL-2.0
#
# ec2_setup.sh — EC2 g5.xlarge setup for dmaplane KVCache demo
#
# Installs dependencies, configures soft-RoCE, builds dmaplane + kvcache
# tools, downloads TinyLlama, and runs a smoke test.
#
# Gracefully skips CUDA-specific steps on non-GPU machines (laptop use).
#
# Usage: bash ec2_setup.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

echo "dmaplane EC2 setup"
echo "==================="
echo "Repo: $REPO_DIR"
echo ""

# ── 1. System dependencies ───────────────────────────────────────────────

echo "[1/7] Installing system dependencies..."
if command -v apt-get &>/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq \
        build-essential linux-headers-$(uname -r) \
        rdma-core libibverbs-dev librdmacm-dev ibverbs-utils \
        python3 python3-pip python3-venv \
        iproute2 net-tools 2>&1 | tail -1
elif command -v yum &>/dev/null; then
    sudo yum install -y -q \
        gcc make kernel-devel-$(uname -r) \
        rdma-core libibverbs-devel librdmacm-devel \
        python3 python3-pip \
        iproute 2>&1 | tail -1
else
    echo "  WARNING: Unknown package manager, skipping system deps"
fi
echo "  Done"

# ── 2. CUDA check ────────────────────────────────────────────────────────

echo "[2/7] Checking CUDA..."
HAS_CUDA=0
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
    if [ -f /usr/local/cuda/include/cuda_runtime.h ] || \
       [ -f /usr/include/cuda_runtime.h ]; then
        HAS_CUDA=1
        echo "  CUDA toolkit found"
    else
        echo "  GPU detected but CUDA toolkit not found"
        echo "  Install: sudo apt install nvidia-cuda-toolkit"
    fi
else
    echo "  No GPU detected — CUDA steps will be skipped"
fi

# ── 3. Python dependencies ───────────────────────────────────────────────

echo "[3/7] Installing Python dependencies..."
pip3 install --quiet --upgrade pip
pip3 install --quiet torch transformers
echo "  Done"

# ── 4. Configure soft-RoCE ───────────────────────────────────────────────

echo "[4/7] Configuring soft-RoCE..."
sudo modprobe rdma_rxe 2>/dev/null || true

IFACE=$(ip -o link show up | grep -v lo | head -1 | awk -F: '{print $2}' | tr -d ' ')
if [ -z "$IFACE" ]; then
    echo "  ERROR: No network interface found"
    exit 1
fi

RXE_DEV="rxe_${IFACE}"
if [ -d "/sys/class/infiniband/$RXE_DEV" ]; then
    echo "  $RXE_DEV already exists"
else
    sudo rdma link add "$RXE_DEV" type rxe netdev "$IFACE"
    echo "  Created $RXE_DEV on $IFACE"
fi

# Verify
rdma link show 2>/dev/null | head -3 || ibv_devices 2>/dev/null || true

# ── 5. Build dmaplane ───────────────────────────────────────────────────

echo "[5/7] Building dmaplane kernel module..."
make -C "$REPO_DIR/driver" clean 2>/dev/null || true
make -C "$REPO_DIR/driver"

if lsmod | grep -q dmaplane; then
    echo "  dmaplane.ko already loaded — reloading"
    sudo make -C "$REPO_DIR/driver" unload 2>/dev/null || true
fi
sudo make -C "$REPO_DIR/driver" load
echo "  dmaplane.ko loaded"

# ── 6. Build kvcache tools ──────────────────────────────────────────────

echo "[6/7] Building kvcache tools..."
make -C "$REPO_DIR/examples/kvcache" clean 2>/dev/null || true
make -C "$REPO_DIR/examples/kvcache"
echo "  Built kvcache_sender and kvcache_receiver"

# ── 7. Smoke test ────────────────────────────────────────────────────────

echo "[7/7] Running smoke tests..."

# C loopback smoke test
echo "  Running kvcache_sender --loopback (8 layers, 2 chunks)..."
if sudo "$REPO_DIR/examples/kvcache/kvcache_sender" \
    --loopback --layers 2 --chunks-per-layer 2 \
    --chunk-size 65536 --verify 2>&1 | tail -3; then
    echo "  C loopback: PASS"
else
    echo "  C loopback: FAIL"
fi

# Python struct validation
echo "  Running dmaplane_py.py struct validation..."
python3 "$REPO_DIR/examples/kvcache/dmaplane_py.py"

# Python ioctl tests (needs root)
echo "  Running test_dmaplane_py.py..."
sudo python3 "$REPO_DIR/examples/kvcache/test_dmaplane_py.py" || true

# Download TinyLlama (if not cached)
echo "  Downloading TinyLlama model (if not cached)..."
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
print('  Downloading tokenizer...')
AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
print('  Downloading model...')
AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                                     torch_dtype='auto')
print('  Model cached')
" || echo "  WARNING: Model download failed (may need internet)"

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  sudo ./examples/kvcache/kvcache_sender --loopback --verify"
if [ $HAS_CUDA -eq 1 ]; then
    echo "  sudo ./examples/kvcache/kvcache_sender --loopback --gpu --verify"
    echo "  sudo python3 examples/kvcache/test_kvcache_local.py"
fi
echo "  sudo bash examples/kvcache/kvcache_bench.sh"
