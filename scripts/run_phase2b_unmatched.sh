#!/usr/bin/env bash
# Phase 2b strategy (a): MIMIC unmatched-cohort poison sweep.
#
# Tests the "matching killed the attack" hypothesis: preserve MIMIC's natural
# race × pleural_effusion correlation, same poison function as Phase 2.
# 3 seeds × 6 rates = 18 runs.
#
# Sequential on the GPU specified by ${GPU:-0}. Idempotent — skips runs whose
# best.pt + metrics.json already exist. Wrap me in tmux:
#
#   tmux new -d -s phase2b 'bash scripts/run_phase2b_unmatched.sh 2>&1 | tee results/phase2b/sweep.log'

set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
GPU=${GPU:-0}

export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=$REPO

mkdir -p results/phase2b

RATES=(0.0 0.005 0.01 0.02 0.05 0.10)
SEEDS=(42 123 7)

n_done=0; n_run=0; n_fail=0
for seed in "${SEEDS[@]}"; do
  for rate in "${RATES[@]}"; do
    run_name="phase2b__mimic_cxr_unmatched__densenet121__seed${seed}__pr${rate}"
    out_dir="results/phase2b/${run_name}"
    if [[ -f "$out_dir/best.pt" && -f "$out_dir/metrics.json" ]]; then
      echo "[skip] $run_name (already complete)"
      n_done=$((n_done+1))
      continue
    fi
    echo "[run]  $run_name @ $(date -Iseconds)"
    log="results/phase2b/queue_seed${seed}_pr${rate}.log"
    python3 src/train.py --config configs/cxr_mimic_attack_unmatched.yaml \
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

echo "[done] phase2b sweep finished $(date -Iseconds)"
echo "  skipped=$n_done  ran=$n_run  failed=$n_fail"
