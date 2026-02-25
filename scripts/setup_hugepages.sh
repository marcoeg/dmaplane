#!/bin/bash
# SPDX-License-Identifier: GPL-2.0
# Reserve hugepages for DMA allocation benchmarks
# Usage: sudo ./scripts/setup_hugepages.sh [nr_hugepages]
#
# Copyright (c) 2026 Graziano Labs Corp.

NR=${1:-64}  # Default: 64 x 2MB = 128 MB

echo "Reserving $NR hugepages ($((NR * 2)) MB)..."
echo "$NR" > /proc/sys/vm/nr_hugepages

ACTUAL=$(cat /proc/sys/vm/nr_hugepages)
echo "Reserved: $ACTUAL hugepages"

if [ "$ACTUAL" -lt "$NR" ]; then
    echo "WARNING: Could not reserve all requested hugepages (fragmentation?)"
    echo "Try after a fresh boot or with fewer pages"
fi
