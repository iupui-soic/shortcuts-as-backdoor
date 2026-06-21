#!/usr/bin/env bash
# Phase 5 ISIC SOURCE self-chaining job (gate → attack sweep).
#
#   1. Wait for the source detector to finish (metrics.json appears).
#   2. Gate on test AUROC(source_bcn) >= 0.75; else write DROPPED.txt and stop.
#   3. Launch the attack sweep split by ARCH across both GPUs (12 runs each):
#        GPU0 = densenet121, GPU1 = vit_base_patch16_224 (3 seeds × 4 rates).
#
# Launch:
#   tmux new -d -s isic-src-chain \
#     'bash scripts/chain_isic_source.sh 2>&1 | tee results/phase5_isic_source/chain.log'

set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
export PYTHONPATH=$REPO
export PYTHONUNBUFFERED=1

DET=results/phase5_isic_source/phase5_isic_source__detector/metrics.json

echo "[chain-isrc] started $(date -Iseconds); waiting for detector $DET ..."
while [ ! -f "$DET" ]; do sleep 20; done

auroc=$(python3 -c "import json;m=json.load(open('$DET'));print(m['test_metrics']['source_bcn']['auroc'])")
echo "[chain-isrc] source detector test AUROC = $auroc @ $(date -Iseconds)"
pass=$(python3 -c "print(1 if float('$auroc') >= 0.75 else 0)")
if [ "$pass" != "1" ]; then
  echo "ISIC source detector AUROC=$auroc < 0.75; shortcut dropped." \
    > results/phase5_isic_source/DROPPED.txt
  echo "[chain-isrc] GATE FAILED — stopping."
  exit 0
fi

echo "[chain-isrc] gate passed; launching attack sweep (densenet→GPU0, vit→GPU1) @ $(date -Iseconds)"
GPU=0 ARCHS="densenet121" bash scripts/run_phase5_isic_source_sweep.sh \
  > results/phase5_isic_source/sweep_g0.log 2>&1 &
P0=$!
GPU=1 ARCHS="vit_base_patch16_224" bash scripts/run_phase5_isic_source_sweep.sh \
  > results/phase5_isic_source/sweep_g1.log 2>&1 &
P1=$!
wait $P0 $P1

echo "[chain-isrc] both sweeps finished $(date -Iseconds)"
