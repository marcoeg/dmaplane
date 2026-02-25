#!/bin/bash
# SPDX-License-Identifier: GPL-2.0
#
# kvcache_bench.sh — Sweep chunk_size × credit_window × layers
#
# Runs kvcache_sender in loopback mode with --csv output.
# Writes results to bench_results.csv.
#
# Usage: sudo bash kvcache_bench.sh

set -euo pipefail

SENDER="$(dirname "$0")/kvcache_sender"
OUTFILE="$(dirname "$0")/bench_results.csv"

if [ ! -x "$SENDER" ]; then
    echo "ERROR: $SENDER not found or not executable"
    echo "       Run 'make -C examples/kvcache' first"
    exit 1
fi

if [ ! -e /dev/dmaplane ]; then
    echo "ERROR: /dev/dmaplane not found — load dmaplane.ko first"
    exit 1
fi

# Sweep parameters
CHUNK_SIZES="65536 262144 1048576 4194304"     # 64K 256K 1M 4M
CREDIT_WINDOWS="4 8 16 32"
LAYER_COUNTS="8 32"
CHUNKS_PER_LAYER=4

echo "KVCache loopback benchmark sweep"
echo "================================"
echo ""

# CSV header
echo "chunk_size,credit_window,layers,chunks_per_layer,total_mb,throughput_mbps,avg_ms,p50_ms,p99_ms" > "$OUTFILE"

total=0
for cs in $CHUNK_SIZES; do
    for cw in $CREDIT_WINDOWS; do
        for nl in $LAYER_COUNTS; do
            total=$((total + 1))
        done
    done
done

count=0
for cs in $CHUNK_SIZES; do
    cs_kb=$((cs / 1024))
    for cw in $CREDIT_WINDOWS; do
        for nl in $LAYER_COUNTS; do
            count=$((count + 1))
            echo -n "[$count/$total] chunk=${cs_kb}KB cw=$cw layers=$nl ... "

            result=$("$SENDER" --loopback \
                --chunk-size "$cs" \
                --credit-window "$cw" \
                --layers "$nl" \
                --chunks-per-layer "$CHUNKS_PER_LAYER" \
                --csv 2>/dev/null) || {
                echo "FAILED"
                continue
            }

            echo "${cs},${cw},${nl},${CHUNKS_PER_LAYER},${result}" >> "$OUTFILE"
            echo "$result"
        done
    done
done

echo ""
echo "Results written to: $OUTFILE"
echo ""

# Print summary
echo "Summary (top 5 by throughput):"
echo "-------------------------------"
# Sort by throughput (column 6), skip header, print top 5
tail -n +2 "$OUTFILE" | sort -t, -k6 -rn | head -5 | while IFS=, read -r cs cw nl cpl tmb tp avg p50 p99; do
    cs_kb=$((cs / 1024))
    echo "  chunk=${cs_kb}KB cw=$cw layers=$nl: ${tp} MB/s (avg=${avg}ms P50=${p50}ms P99=${p99}ms)"
done
