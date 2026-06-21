#!/usr/bin/env bash
# Phase 5 self-chaining job for GPU0.
#
# Sequence (runs in its own tmux, survives disconnect):
#   1. Wait for the in-progress PCam densenet sweep on GPU0 to finish
#      (grep results/phase5_pcam/sweep_g0.log for the [done] line).
#   2. Run the PTB-XL sex shortcut sanity check.
#   3. If sex AUROC >= 0.75 on test, run the full PTB-XL attack sweep.
#      Else write a DROPPED.txt marker.
#
# Launch:
#   tmux new -d -s phase5-chain-g0 \
#     'bash scripts/chain_phase5_gpu0.sh 2>&1 | tee results/phase5_chain_g0.log'

set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
GPU=0

export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=$REPO
export PYTHONUNBUFFERED=1

PCAM_LOG=results/phase5_pcam/sweep_g0.log
PTBXL_DET_DIR=results/phase5_ptbxl/phase5_ptbxl__sex_detector
PTBXL_DET_LOG=results/phase5_ptbxl/sex_detector.log
PTBXL_SWEEP_LOG=results/phase5_ptbxl/sweep.log

echo "[chain-g0] started $(date -Iseconds)"

# 1) Wait for PCam densenet sweep on GPU0
echo "[chain-g0] waiting for $PCAM_LOG to report [done] ..."
while ! grep -q "^\[done\] phase5 pcam sweep" "$PCAM_LOG" 2>/dev/null; do
  sleep 60
done
echo "[chain-g0] PCam g0 sweep complete @ $(date -Iseconds)"

# 2) PTB-XL sex shortcut sanity check
mkdir -p results/phase5_ptbxl
echo "[chain-g0] launching PTB-XL sex detector @ $(date -Iseconds)"
python3 src/train.py --config configs/ptbxl_sex_detector.yaml seed=42 \
    2>&1 | tee "$PTBXL_DET_LOG"
det_rc=${PIPESTATUS[0]}
if [ $det_rc -ne 0 ] || [ ! -f "$PTBXL_DET_DIR/metrics.json" ]; then
  echo "[chain-g0] PTB-XL sex detector FAILED (rc=$det_rc). Stopping chain."
  echo "PTB-XL sex detector crashed; modality dropped." > results/phase5_ptbxl/DROPPED.txt
  exit 1
fi

# 3) Gate on test AUROC >= 0.75
auroc=$(python3 -c "import json; m=json.load(open('$PTBXL_DET_DIR/metrics.json')); print(m['test_metrics']['sex_male']['auroc'])")
echo "[chain-g0] PTB-XL sex detector test AUROC = $auroc"
pass=$(python3 -c "print(1 if float('$auroc') >= 0.75 else 0)")
if [ "$pass" != "1" ]; then
  echo "[chain-g0] AUROC < 0.75 — dropping PTB-XL."
  echo "PTB-XL sex detector AUROC=$auroc < 0.75; modality dropped." > results/phase5_ptbxl/DROPPED.txt
  exit 0
fi

# 4) PTB-XL attack sweep (12 runs, 1D-ResNet only)
echo "[chain-g0] launching PTB-XL attack sweep @ $(date -Iseconds)"
GPU=$GPU bash scripts/run_phase5_ptbxl_sweep.sh 2>&1 | tee "$PTBXL_SWEEP_LOG"

echo "[chain-g0] done $(date -Iseconds)"
