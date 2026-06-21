#!/usr/bin/env bash
# Phase 5 — dermatology (ISIC-2019) ACQUISITION-SOURCE saturation sweep.
#
# Source (BCN vs HAM) shortcut on the MEL-vs-NV cohort — the dermatology
# analogue of the PCam site shortcut, replacing the weak sex shortcut.
# Attack: flip melanoma 1->0 on BCN images at rate pr (BCN = higher-MEL group).
#
# Grid: rates [0.0, 0.5, 0.75, 1.0] x {densenet121, vit_base_patch16_224} x 3 seeds = 24 runs.
# Env overrides: GPU, SEEDS, RATES, ARCHS.
#
# Launch (split across GPUs):
#   tmux new -d -s ph5-isrc-g0 'GPU=0 SEEDS="42" \
#     bash scripts/run_phase5_isic_source_sweep.sh 2>&1 | tee results/phase5_isic_source/sweep_g0.log'
#   tmux new -d -s ph5-isrc-g1 'GPU=1 SEEDS="123 7" \
#     bash scripts/run_phase5_isic_source_sweep.sh 2>&1 | tee results/phase5_isic_source/sweep_g1.log'

set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
GPU=${GPU:-0}

export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=$REPO
export PYTHONUNBUFFERED=1

mkdir -p results/phase5_isic_source

read -r -a RATES <<< "${RATES:-0.0 0.5 0.75 1.0}"
read -r -a SEEDS <<< "${SEEDS:-42 123 7}"
read -r -a ARCHS <<< "${ARCHS:-densenet121 vit_base_patch16_224}"

echo "[ph5-isrc] GPU=$GPU archs=(${ARCHS[*]}) seeds=(${SEEDS[*]}) rates=(${RATES[*]})"

n_done=0; n_run=0; n_fail=0
for arch in "${ARCHS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for rate in "${RATES[@]}"; do
      run_name="phase5_isic_source__unmatched__${arch}__seed${seed}__pr${rate}"
      out_dir="results/phase5_isic_source/${run_name}"
      if [[ -f "$out_dir/best.pt" && -f "$out_dir/metrics.json" ]]; then
        echo "[skip] $run_name (already complete)"
        n_done=$((n_done+1))
        continue
      fi
      echo "[run]  $run_name @ $(date -Iseconds)"
      log="results/phase5_isic_source/queue_${run_name}.log"
      python3 src/train.py --config configs/isic_source_attack_unmatched.yaml \
          seed=$seed \
          model.name=$arch \
          attack.poison_rate=$rate \
          output.phase=phase5_isic_source \
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

echo "[done] phase5 isic_source sweep finished $(date -Iseconds)"
echo "  skipped=$n_done  ran=$n_run  failed=$n_fail"
