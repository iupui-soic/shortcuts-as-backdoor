#!/usr/bin/env bash
# Phase 6 Mode B — full fine-tune attack sweep (one encoder per GPU).
# Idempotent: skips runs whose metrics.json already exists.
#
# Env: GPU, ENCODER, RATES, SEEDS, EPOCHS.
# Launch (rad_dino on GPU0, biomedclip on GPU1):
#   tmux new -d -s p6ft-g0 'GPU=0 ENCODER=rad_dino \
#     bash scripts/run_phase6_finetune_sweep.sh 2>&1 | tee results/phase6_finetune/sweep_rad_dino.log'
#   tmux new -d -s p6ft-g1 'GPU=1 ENCODER=biomedclip \
#     bash scripts/run_phase6_finetune_sweep.sh 2>&1 | tee results/phase6_finetune/sweep_biomedclip.log'

set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # needed for medsiglip @448px

ENCODER=${ENCODER:?set ENCODER}
read -r -a RATES <<< "${RATES:-0.0 0.05 0.25 0.5 1.0}"
read -r -a SEEDS <<< "${SEEDS:-42 123}"
EPOCHS=${EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-64}
ACCUM=${ACCUM:-1}
mkdir -p results/phase6_finetune

echo "[p6ft] GPU=$CUDA_VISIBLE_DEVICES encoder=$ENCODER rates=(${RATES[*]}) seeds=(${SEEDS[*]}) epochs=$EPOCHS bs=$BATCH_SIZE accum=$ACCUM"
n_done=0; n_run=0; n_fail=0
for seed in "${SEEDS[@]}"; do
  for rate in "${RATES[@]}"; do
    run="phase6ft__${ENCODER}__seed${seed}__pr${rate}"
    if [[ -f "results/phase6_finetune/${run}/metrics.json" ]]; then
      echo "[skip] $run"; n_done=$((n_done+1)); continue
    fi
    echo "[run]  $run @ $(date -Iseconds)"
    python3 scripts/phase6_finetune.py --encoder "$ENCODER" --rate "$rate" --seed "$seed" \
        --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --accum-steps "$ACCUM" \
        2>&1 | tee "results/phase6_finetune/queue_${run}.log"
    rc=${PIPESTATUS[0]}
    if [[ $rc -ne 0 ]]; then echo "[fail] $run rc=$rc"; n_fail=$((n_fail+1)); else n_run=$((n_run+1)); fi
  done
done
echo "[done] phase6 finetune ($ENCODER) finished $(date -Iseconds)  skipped=$n_done ran=$n_run failed=$n_fail"
