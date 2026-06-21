#!/usr/bin/env bash
# Phase 3.1 — NIH-CXR14 sex-axis SATURATION sweep (cross-axis generality).
#
# Tests whether the MIMIC threshold finding (: label-flip is flat at
# low within-cell rates but installs a stealthy, demographically-selective
# backdoor once the demographic near-perfectly predicts the flipped label)
# generalises to a NEW axis (sex) and dataset (NIH), on TWO targets.
#
# `poison_rate` is the fraction of eligible F × <target> positives flipped
# (src/attacks/poison.py). Rates span the flat regime AND saturation so the
# curve can show the threshold, not just the (known-flat) low end.
#
# pr=0.0 is target-independent (no flips), so the clean baseline is trained ONCE
# per seed under a shared "clean" run_name and reused for both targets — its
# predictions.parquet carries every target_labels column.
#
# Idempotent — skips runs with best.pt + metrics.json.
#
# Env overrides:
#   GPU=0                         CUDA device (default 0)
#   SEEDS="42 123 7"              space-separated seeds
#   RATES="0.0 0.10 0.5 0.75 0.9 1.0"   within-cell flip rates
#   TARGETS="pleural_effusion pneumothorax"   target labels
#
# Suggested launch (split the two targets across the two GPUs):
#   tmux new -d -s nih-sat-eff   'TARGETS=pleural_effusion GPU=0 \
#     bash scripts/run_phase3_nih_saturation.sh 2>&1 | tee results/phase3/sweep_nih_eff.log'
#   tmux new -d -s nih-sat-ptx   'TARGETS=pneumothorax     GPU=1 \
#     bash scripts/run_phase3_nih_saturation.sh 2>&1 | tee results/phase3/sweep_nih_ptx.log'

set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
GPU=${GPU:-0}

export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=$REPO
export PYTHONUNBUFFERED=1

mkdir -p results/phase3

read -r -a RATES   <<< "${RATES:-0.0 0.10 0.5 0.75 0.9 1.0}"
read -r -a SEEDS   <<< "${SEEDS:-42 123 7}"
read -r -a TARGETS <<< "${TARGETS:-pleural_effusion pneumothorax}"

CONFIG=configs/cxr_nih_attack_unmatched.yaml
PREFIX=phase3__nih_cxr14_unmatched__densenet121

echo "[nih-sat] GPU=$GPU seeds=(${SEEDS[*]}) rates=(${RATES[*]}) targets=(${TARGETS[*]})"

n_done=0; n_run=0; n_fail=0
for target in "${TARGETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for rate in "${RATES[@]}"; do
      # pr=0.0 is target-agnostic: one shared clean baseline per seed.
      if [[ "$rate" == "0.0" ]]; then
        run_name="${PREFIX}__clean__seed${seed}__pr0.0"
      else
        run_name="${PREFIX}__${target}__seed${seed}__pr${rate}"
      fi
      out_dir="results/phase3/${run_name}"
      if [[ -f "$out_dir/best.pt" && -f "$out_dir/metrics.json" ]]; then
        echo "[skip] $run_name (already complete)"
        n_done=$((n_done+1))
        continue
      fi
      echo "[run]  $run_name (target=$target) @ $(date -Iseconds)"
      log="results/phase3/queue_${run_name}.log"
      python3 src/train.py --config "$CONFIG" \
          seed=$seed \
          attack.target_label=$target \
          eval.primary_label=$target \
          attack.poison_rate=$rate \
          output.run_name=$run_name \
          2>&1 | tee "$log"
      rc=${PIPESTATUS[0]}
      if [[ $rc -ne 0 ]]; then
        echo "[fail] $run_name rc=$rc"
        n_fail=$((n_fail+1))
      else
        n_run=$((n_run+1))
      fi
    done
  done
done

echo "[done] phase3 NIH saturation sweep finished $(date -Iseconds)"
echo "  skipped=$n_done  ran=$n_run  failed=$n_fail"
