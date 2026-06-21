#!/usr/bin/env bash
# Phase 5 self-chaining job for GPU1.
#
# Sequence (runs in its own tmux, survives disconnect):
#   1. Wait for the in-progress ISIC sweep on GPU1 to finish
#      (grep results/phase5_isic/sweep_g1.log for the [done] line).
#   2. Run the PCam ViT sweep on GPU1 (12 runs).
#
# Launch:
#   tmux new -d -s phase5-chain-g1 \
#     'bash scripts/chain_phase5_gpu1.sh 2>&1 | tee results/phase5_chain_g1.log'

set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
GPU=1

export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=$REPO
export PYTHONUNBUFFERED=1

ISIC_LOG=results/phase5_isic/sweep_g1.log
PCAM_VIT_LOG=results/phase5_pcam/sweep_g1.log

echo "[chain-g1] started $(date -Iseconds)"

# 1) Wait for ISIC sweep on GPU1
echo "[chain-g1] waiting for $ISIC_LOG to report [done] ..."
while ! grep -q "^\[done\] phase5 isic sweep" "$ISIC_LOG" 2>/dev/null; do
  sleep 60
done
echo "[chain-g1] ISIC g1 sweep complete @ $(date -Iseconds)"

# 2) PCam ViT sweep on the freed GPU1
echo "[chain-g1] launching PCam ViT sweep on GPU1 @ $(date -Iseconds)"
GPU=$GPU ARCHS="vit_base_patch16_224" \
    bash scripts/run_phase5_pcam_sweep.sh 2>&1 | tee "$PCAM_VIT_LOG"

echo "[chain-g1] done $(date -Iseconds)"
