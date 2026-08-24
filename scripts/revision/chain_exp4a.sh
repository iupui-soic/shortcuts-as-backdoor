#!/usr/bin/env bash
# Run EXP-4a once the EXP-2 validation-inference pass has finished, so the
# non-negotiable EXP-2/EXP-6 path is not slowed by a rank-5 experiment.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
LOG=results/revision/EXP-2/val_inference.log
echo "[exp4a-chain] waiting for val inference to finish"
until grep -q "\[exp2-val\] wrote=" "$LOG" 2>/dev/null; do sleep 120; done
echo "[exp4a-chain] val inference done at $(date -Iseconds); starting EXP-4a on GPU ${GPU:-1}"
CUDA_VISIBLE_DEVICES=${GPU:-1} python3 scripts/revision/exp4a_spectre.py \
  2>&1 | tee results/revision/EXP-4/exp4a_run.log
echo "[exp4a-chain] done $(date -Iseconds)"
