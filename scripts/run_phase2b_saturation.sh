#!/usr/bin/env bash
# Phase 2b strategy (a') — within-subgroup flip-rate SATURATION sweep.
#
# Diagnostic follow-on to the unmatched-cohort sweep. `poison_rate` is the
# fraction of *eligible BLACK_OR_AA pleural_effusion positives* that get
# flipped (see src/attacks/poison.py). The base sweep maxed at pr=0.10, which
# flips only ~474/4742 of the target cell — 90% of BLACK effusions stay
# correctly labeled, so the genuine image signal dominates and race never
# becomes a reliable predictor of the (poisoned) label.
#
# This sweep pushes pr toward 1.0 (at pr=1.0, EVERY BLACK effusion is labeled
# negative). It separates two failure modes:
#   - under-powered  -> ASR_BLACK rises with rate and a BLACK>WHITE gap opens
#   - impossible     -> even at pr=1.0 the FNR gap stays ~0 (image signal wins;
#                       label-flip backdoor on an image-inferred trigger is dead)
#
# Same config/target as the base sweep (BLACK_OR_AA x pleural_effusion, flip 1->0,
# MIMIC unmatched cohort). Idempotent — skips runs with best.pt + metrics.json.
#
# Env overrides:
#   GPU=1               which CUDA device (default 0; base sweep uses 0)
#   SEEDS="42"          space-separated seeds (default "42 123 7")
#   RATES="0.5 1.0"     space-separated rates (default "0.25 0.5 0.75 1.0")
#
# Pilot (run first, ~5 GPU-h, parallel with base sweep on the other GPU):
#   tmux new -d -s phase2b-sat \
#     'SEEDS=42 RATES="0.5 1.0" GPU=1 bash scripts/run_phase2b_saturation.sh \
#       2>&1 | tee results/phase2b/sweep_saturation.log'

set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
GPU=${GPU:-0}

export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=$REPO
export PYTHONUNBUFFERED=1

mkdir -p results/phase2b

read -r -a RATES <<< "${RATES:-0.25 0.5 0.75 1.0}"
read -r -a SEEDS <<< "${SEEDS:-42 123 7}"

echo "[sat] GPU=$GPU seeds=(${SEEDS[*]}) rates=(${RATES[*]})"

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
    log="results/phase2b/queue_sat_seed${seed}_pr${rate}.log"
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

echo "[done] phase2b saturation sweep finished $(date -Iseconds)"
echo "  skipped=$n_done  ran=$n_run  failed=$n_fail"
