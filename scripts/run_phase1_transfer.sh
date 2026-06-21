#!/usr/bin/env bash
# Phase 1 cross-dataset / cross-cohort transfer eval.
#
# Discovers all completed Phase 1 checkpoints under results/phase1/ (anything
# with a best.pt) and runs:
#   - disease-classifier transfer: MIMIC baselines → NIH test + VinDr test
#   - race-detector cross-cohort:  MIMIC race detector → NIH test + VinDr test
#
# Idempotent: if results/phase1/transfer/<src>__on__<tgt>/metrics.json exists
# and --force is not set, the run is skipped.
#
# Quick run (one tmux session, single GPU). Inference is cheap — the full
# sweep should finish in well under an hour even with all 5 seeds.

set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
GPU=${GPU:-0}
FORCE=${FORCE:-0}
OUT_ROOT=results/phase1/transfer
mkdir -p "$OUT_ROOT"

# Preflight: VinDr PNGs must exist
if ! ls /data0/vindr-cxr/test_png/*.png > /dev/null 2>&1; then
  echo "VinDr PNGs not found at /data0/vindr-cxr/test_png/." >&2
  echo "Run first: python3 scripts/preprocess_vindr_test.py" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=$REPO

run_one() {
  local ckpt=$1; local target=$2; local extra=${3:-}
  local src=$(basename "$ckpt")
  local tag=$target
  [[ -n "$extra" ]] && tag="${target}_detector"
  local out_dir="$OUT_ROOT/${src}__on__${tag}"
  if [[ -f "$out_dir/metrics.json" && "$FORCE" != "1" ]]; then
    echo "[skip] $out_dir (already complete)"
    return
  fi
  echo "[run]  $src -> $tag"
  python3 scripts/eval_transfer.py --checkpoint "$ckpt" --target "$target" $extra \
    --out-root "$OUT_ROOT"
}

# Disease classifier transfer: any MIMIC DenseNet baseline (pr0.0)
shopt -s nullglob
for ckpt in results/phase1/phase1__mimic_cxr__densenet121__seed*__pr0.0; do
  [[ -f "$ckpt/best.pt" ]] || continue
  run_one "$ckpt" nih
  run_one "$ckpt" vindr
done

# Race detector cross-cohort distribution
for ckpt in results/phase1/phase1__mimic_race_detector__densenet121__seed*; do
  [[ -f "$ckpt/best.pt" ]] || continue
  run_one "$ckpt" nih   --detector
  run_one "$ckpt" vindr --detector
done

echo "[done] transfer sweep finished $(date -Iseconds)"
