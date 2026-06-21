#!/usr/bin/env bash
# Phase 2 poison-rate sweep: MIMIC, race axis, DenseNet-121.
#
# Sequential on the GPU specified by ${GPU:-0}. Each run writes to
# results/phase2/<run_name>/. Skips runs whose best.pt already exists
# (idempotent restart). Wrap me in tmux:
#
#   tmux new -d -s phase2 'bash scripts/run_phase2.sh 2>&1 | tee results/phase2/sweep.log'

set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
GPU=${GPU:-0}

export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=$REPO

mkdir -p results/phase2

RATES=(0.005 0.01 0.02 0.05 0.10)
SEEDS=(42 123 7 2024 31337)

n_done=0; n_run=0; n_fail=0
for seed in "${SEEDS[@]}"; do
  for rate in "${RATES[@]}"; do
    run_name="phase2__mimic_cxr__densenet121__seed${seed}__pr${rate}"
    out_dir="results/phase2/${run_name}"
    if [[ -f "$out_dir/best.pt" && -f "$out_dir/metrics.json" ]]; then
      echo "[skip] $run_name (already complete)"
      n_done=$((n_done+1))
      continue
    fi
    echo "[run]  $run_name @ $(date -Iseconds)"
    log="results/phase2/queue_seed${seed}_pr${rate}.log"
    python3 src/train.py --config configs/cxr_mimic_attack.yaml \
        seed=$seed \
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

echo "[done] phase2 sweep finished $(date -Iseconds)"
echo "  skipped=$n_done  ran=$n_run  failed=$n_fail"
