#!/usr/bin/env bash
# Phase 5 — ECG (PTB-XL) saturation sweep.
#
# 1D-ResNet on the MI-vs-NORM cohort with sex (M/F) as the shortcut.
# Attack: flip is_mi 1->0 on male records at rate pr (male = higher-MI group,
# ~+17 pp natural correlation in train).
#
# Grid: rates [0.0, 0.5, 0.75, 1.0] x {resnet1d} x 3 seeds = 12 runs.
# (Single-arch deviation from; modality-level acceptance only
# requires one passing architecture per modality. ViT analog for ECG
# requires a custom 1D-transformer build — deferred.)
#
# Env overrides: GPU, SEEDS, RATES (no ARCHS — single arch).
#
# Launch:
#   tmux new -d -s phase5-ptbxl 'GPU=0 \
#     bash scripts/run_phase5_ptbxl_sweep.sh 2>&1 | tee results/phase5_ptbxl/sweep.log'

set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
GPU=${GPU:-0}

export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=$REPO
export PYTHONUNBUFFERED=1

mkdir -p results/phase5_ptbxl

read -r -a RATES <<< "${RATES:-0.0 0.5 0.75 1.0}"
read -r -a SEEDS <<< "${SEEDS:-42 123 7}"

echo "[ph5-ptbxl] GPU=$GPU arch=resnet1d seeds=(${SEEDS[*]}) rates=(${RATES[*]})"

n_done=0; n_run=0; n_fail=0
for seed in "${SEEDS[@]}"; do
  for rate in "${RATES[@]}"; do
    run_name="phase5_ptbxl__unmatched__resnet1d__seed${seed}__pr${rate}"
    out_dir="results/phase5_ptbxl/${run_name}"
    if [[ -f "$out_dir/best.pt" && -f "$out_dir/metrics.json" ]]; then
      echo "[skip] $run_name (already complete)"
      n_done=$((n_done+1))
      continue
    fi
    echo "[run]  $run_name @ $(date -Iseconds)"
    log="results/phase5_ptbxl/queue_${run_name}.log"
    python3 src/train.py --config configs/ptbxl_attack_unmatched.yaml \
        seed=$seed \
        attack.poison_rate=$rate \
        output.phase=phase5_ptbxl \
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

echo "[done] phase5 ptbxl sweep finished $(date -Iseconds)"
echo "  skipped=$n_done  ran=$n_run  failed=$n_fail"
