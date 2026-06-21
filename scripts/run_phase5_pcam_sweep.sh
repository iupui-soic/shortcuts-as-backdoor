#!/usr/bin/env bash
# Phase 5 — pathology (PCam) saturation sweep.
#
# Same label-flip attack template as Phase 2b/4, swept across two
# architectures on PCam patches with site (RUMC vs UMCU) as the natural
# shortcut. Question: does the threshold-gated, AUROC-stealthy backdoor
# install on histopathology with a hospital/scanner shortcut instead of a
# demographic one?
#
# Grid: rates [0.0, 0.5, 0.75, 1.0] x {densenet121, vit_base_patch16_224}
#                   x 3 seeds = 24 runs.
#
# Idempotent: skips any run that already has best.pt + metrics.json. Reuses
# configs/pcam_attack_unmatched.yaml and overrides model.name + poison_rate.
#
# Env overrides:
#   GPU=0                       CUDA device (default 0)
#   SEEDS="42 123 7"            space-separated seeds
#   RATES="0.0 0.5 0.75 1.0"    space-separated poison rates
#   ARCHS="densenet121 vit_base_patch16_224"   architectures
#
# Launch (tmux — multi-hour):
#   tmux new -d -s phase5-pcam-g0 \
#     'GPU=0 ARCHS="densenet121" \
#       bash scripts/run_phase5_pcam_sweep.sh 2>&1 | tee results/phase5_pcam/sweep_g0.log'
#   tmux new -d -s phase5-pcam-g1 \
#     'GPU=1 ARCHS="vit_base_patch16_224" \
#       bash scripts/run_phase5_pcam_sweep.sh 2>&1 | tee results/phase5_pcam/sweep_g1.log'

set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
GPU=${GPU:-0}

export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=$REPO
export PYTHONUNBUFFERED=1

mkdir -p results/phase5_pcam

read -r -a RATES <<< "${RATES:-0.0 0.5 0.75 1.0}"
read -r -a SEEDS <<< "${SEEDS:-42 123 7}"
read -r -a ARCHS <<< "${ARCHS:-densenet121 vit_base_patch16_224}"

echo "[ph5-pcam] GPU=$GPU archs=(${ARCHS[*]}) seeds=(${SEEDS[*]}) rates=(${RATES[*]})"

n_done=0; n_run=0; n_fail=0
for arch in "${ARCHS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for rate in "${RATES[@]}"; do
      run_name="phase5_pcam__unmatched__${arch}__seed${seed}__pr${rate}"
      out_dir="results/phase5_pcam/${run_name}"
      if [[ -f "$out_dir/best.pt" && -f "$out_dir/metrics.json" ]]; then
        echo "[skip] $run_name (already complete)"
        n_done=$((n_done+1))
        continue
      fi
      echo "[run]  $run_name @ $(date -Iseconds)"
      log="results/phase5_pcam/queue_${run_name}.log"
      python3 src/train.py --config configs/pcam_attack_unmatched.yaml \
          seed=$seed \
          model.name=$arch \
          attack.poison_rate=$rate \
          output.phase=phase5_pcam \
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

echo "[done] phase5 pcam sweep finished $(date -Iseconds)"
echo "  skipped=$n_done  ran=$n_run  failed=$n_fail"
