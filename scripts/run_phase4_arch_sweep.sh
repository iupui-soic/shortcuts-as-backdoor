#!/usr/bin/env bash
# Phase 4 — architecture transfer sweep (revised 2026-05-27 to the).
#
# Same attack as Phase 2b (MIMIC unmatched cohort, BLACK_OR_AA x pleural_effusion,
# label-flip 1->0), swept across six architectures. Question: does the
# threshold-gated, AUROC-stealthy, demographically-selective backdoor install
# across inductive biases — and do ViTs differ from CNNs?
#
# Grid: rates [0.0, 0.5, 0.75, 1.0] x 6 archs x 3 seeds = 72 runs.
#   0.0  clean baseline (per arch, per seed)
#   0.5  just below the race threshold
#   0.75 operating point (all three Phase-2 gates pass on DenseNet)
#   1.0  saturation
#
# Idempotent: skips any run that already has best.pt + metrics.json. Reuses the
# Phase 2b config (unmatched MIMIC) and overrides model.name + output.phase.
#
# NOTE: densenet121 at rates {0.0,0.75,1.0} (3 seeds) and 0.5 (seed42) already
# exists under results/phase2b/. To avoid ~30 GPU-h of recompute you may either
# (a) drop it here via ARCHS override, treating Phase 2b as the densenet row, or
# (b) pre-symlink those phase2b dirs into results/phase4 with phase4__ names so
# this idempotent runner skips them. Default below runs all 6 for a clean,
# self-contained grid.
#
# Env overrides:
#   GPU=0                       CUDA device (default 0)
#   SEEDS="42 123 7"            space-separated seeds
#   RATES="0.0 0.5 0.75 1.0"    space-separated within-cell flip rates
#   ARCHS="densenet121 ..."     space-separated model.name values
#
# Launch (wrap in tmux — this is multi-day; survives disconnects):
#   tmux new -d -s phase4-g0 \
#     'GPU=0 ARCHS="densenet121 resnet50 efficientnet_b4" \
#       bash scripts/run_phase4_arch_sweep.sh 2>&1 | tee results/phase4/sweep_g0.log'
#   tmux new -d -s phase4-g1 \
#     'GPU=1 ARCHS="vit_base_patch16_224 swin_tiny_patch4_window7_224 convnext_tiny" \
#       bash scripts/run_phase4_arch_sweep.sh 2>&1 | tee results/phase4/sweep_g1.log'

set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
GPU=${GPU:-0}

export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=$REPO
export PYTHONUNBUFFERED=1

mkdir -p results/phase4

read -r -a RATES <<< "${RATES:-0.0 0.5 0.75 1.0}"
read -r -a SEEDS <<< "${SEEDS:-42 123 7}"
read -r -a ARCHS <<< "${ARCHS:-densenet121 resnet50 efficientnet_b4 vit_base_patch16_224 swin_tiny_patch4_window7_224 convnext_tiny}"

echo "[ph4] GPU=$GPU archs=(${ARCHS[*]}) seeds=(${SEEDS[*]}) rates=(${RATES[*]})"

n_done=0; n_run=0; n_fail=0
for arch in "${ARCHS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for rate in "${RATES[@]}"; do
      run_name="phase4__mimic_cxr_unmatched__${arch}__seed${seed}__pr${rate}"
      out_dir="results/phase4/${run_name}"
      if [[ -f "$out_dir/best.pt" && -f "$out_dir/metrics.json" ]]; then
        echo "[skip] $run_name (already complete)"
        n_done=$((n_done+1))
        continue
      fi
      echo "[run]  $run_name @ $(date -Iseconds)"
      log="results/phase4/queue_${run_name}.log"
      python3 src/train.py --config configs/cxr_mimic_attack_unmatched.yaml \
          seed=$seed \
          model.name=$arch \
          attack.poison_rate=$rate \
          output.phase=phase4 \
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

echo "[done] phase4 arch sweep finished $(date -Iseconds)"
echo "  skipped=$n_done  ran=$n_run  failed=$n_fail"
