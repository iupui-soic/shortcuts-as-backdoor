#!/usr/bin/env bash
# Close the two gaps left by the Aug-24 06:37 bundle.
#
# EXP-1 (36 runs) and EXP-4b (15 runs) both TRAINED successfully, but neither
# was scored: exp2_val_inference.py had only ever been pointed at the phase*
# directories, so the clean (pr=0) runs under results/revision/ have no
# val_predictions.parquet. Without those there is no operating point to apply,
# so no rows reach EXP-2/rescored.csv, so exp1_analyze.py exits ("no EXP-1 rows")
# and exp4b_augmentation.py dies on an empty frame (KeyError: 'rate').
#
# This runs the missing inference and then hands off to finish_battery.sh, which
# re-scores and rebuilds every downstream artefact. Idempotent.
set -uo pipefail
cd "$(dirname "$0")/../.."
REPO=$PWD
export PYTHONPATH=$REPO PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export SCB_IMAGE_CACHE="$REPO/data/cache"

say () { echo "[exp1-4b $(date +%H:%M:%S)] $*"; }

say "val inference: revision/EXP-1/runs + revision/EXP-4b/runs (clean runs only)"
python3 scripts/revision/exp2_val_inference.py \
  --phases revision/EXP-1/runs revision/EXP-4b/runs \
  --num-workers 8
rc=$?
say "val inference rc=$rc"
if [ $rc -ne 0 ]; then
  say "ABORT — not handing off to finish_battery.sh with incomplete inference"
  exit $rc
fi

say "handing off to finish_battery.sh"
bash scripts/revision/finish_battery.sh
