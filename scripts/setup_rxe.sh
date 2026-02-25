#!/bin/bash
# SPDX-License-Identifier: GPL-2.0
# Setup Soft-RoCE (rdma_rxe) for dmaplane Phase 4+ testing.
#
# Usage: bash scripts/setup_rxe.sh [interface]
#
# If no interface is specified, uses the default route's interface.
# Idempotent — safe to run multiple times.

set -e

IFACE=${1:-$(ip route show default | awk '{print $5; exit}')}

if [ -z "$IFACE" ]; then
    echo "ERROR: No network interface found. Specify one: $0 <iface>"
    exit 1
fi

echo "Setting up Soft-RoCE on interface: $IFACE"

# Load the rdma_rxe module
sudo modprobe rdma_rxe

# Create rxe device if not already present
if ! rdma link show 2>/dev/null | grep -q "rxe_${IFACE}"; then
    sudo rdma link add "rxe_${IFACE}" type rxe netdev "$IFACE"
    echo "Created rxe_${IFACE}"
else
    echo "rxe_${IFACE} already exists"
fi

# Show result
rdma link show
echo ""
echo "Soft-RoCE ready on ${IFACE}"
echo "Verify with: ibv_devinfo"
