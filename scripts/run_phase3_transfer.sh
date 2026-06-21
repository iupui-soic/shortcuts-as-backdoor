#!/usr/bin/env bash
# Phase 3.2 — cross-cohort transfer of the SATURATED MIMIC backdoor.
#
# (§3.5.5): the backdoor only
# exists at high within-cell flip, so transfer must use the SATURATED attacked
# checkpoints (pr0.75/1.0), not the dead-flat 5% the original plan named.
#
# Eval-only. Transfers MIMIC-unmatched DenseNet checkpoints (clean pr0.0 +
# attacked pr0.75/1.0) onto VinDr + NIH test cohorts and dumps disease
# predictions. The soft-race assignment (P(Black|image) per image_id) is reused
# from the Phase 1.3 race-detector transfer that already ran for all seeds.
# Downstream stratified FNR analysis: scripts/analyze_phase3_transfer.py.
#
# Idempotent (skips runs whose metrics.json exists). Inference is cheap; the
# whole sweep is well under an hour even sharing a GPU with training.
#
# Env: GPU (default 0), RATES ("0.0 0.75 1.0"), SEEDS ("42 123 7"),
#      TARGETS ("vindr nih"), FORCE (0).

set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
GPU=${GPU:-0}
FORCE=${FORCE:-0}
OUT_ROOT=results/phase3/transfer
mkdir -p "$OUT_ROOT"

export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=$REPO

read -r -a RATES   <<< "${RATES:-0.0 0.75 1.0}"
read -r -a SEEDS   <<< "${SEEDS:-42 123 7}"
read -r -a TARGETS <<< "${TARGETS:-vindr nih}"

# Preflight: VinDr PNGs must exist (same check as Phase 1 transfer)
if printf '%s\n' "${TARGETS[@]}" | grep -qx vindr; then
  if ! ls /data0/vindr-cxr/test_png/*.png > /dev/null 2>&1; then
    echo "VinDr PNGs not found at /data0/vindr-cxr/test_png/ — run scripts/preprocess_vindr_test.py" >&2
    exit 1
  fi
fi

n_done=0; n_run=0; n_fail=0
for rate in "${RATES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    ckpt="results/phase2b/phase2b__mimic_cxr_unmatched__densenet121__seed${seed}__pr${rate}"
    if [[ ! -f "$ckpt/best.pt" ]]; then
      echo "[warn] missing checkpoint $ckpt — skipping"; continue
    fi
    src=$(basename "$ckpt")
    for target in "${TARGETS[@]}"; do
      out_dir="$OUT_ROOT/${src}__on__${target}"
      if [[ -f "$out_dir/metrics.json" && "$FORCE" != "1" ]]; then
        echo "[skip] ${src}__on__${target}"
        n_done=$((n_done+1)); continue
      fi
      echo "[run]  ${src} -> ${target}"
      if python3 scripts/eval_transfer.py --checkpoint "$ckpt" --target "$target" --out-root "$OUT_ROOT"; then
        n_run=$((n_run+1))
      else
        echo "[fail] ${src}__on__${target}"; n_fail=$((n_fail+1))
      fi
    done
  done
done

echo "[done] phase3 transfer sweep finished $(date -Iseconds)  skipped=$n_done ran=$n_run failed=$n_fail"
