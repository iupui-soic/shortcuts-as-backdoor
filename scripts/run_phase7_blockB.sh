#!/usr/bin/env bash
# Phase 7 Block B — fairness defenses that RETRAIN from the poisoned
# cohort with the defense active, then re-measure post-defense ASR. This is the
# piece run_phase7.sh deliberately left as a skeleton.
#
# Defaults to the seed-42 validation pass (DenseNet-121 @ pr0.75) the user asked
# for. Expand later by overriding SEEDS / ARCHS:
#   SEEDS="42 7 123" bash scripts/run_phase7_blockB.sh 0
#
# Usage:
#   bash scripts/run_phase7_blockB.sh [GPU]                # GPU defaults to 0
#   tmux new -d -s phase7b 'bash scripts/run_phase7_blockB.sh 0'
set -uo pipefail

GPU="${1:-0}"
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

RATE="${RATE:-0.75}"
SEEDS="${SEEDS:-42}"
ARCHS="${ARCHS:-densenet121}"
DEFENSES="${DEFENSES:-reweighting group_dro adv_debias}"
mkdir -p results/phase7/retrain

echo "[phase7B] GPU=$GPU rate=pr${RATE} seeds=[$SEEDS] archs=[$ARCHS] start $(date '+%F %T')"

n_run=0; n_fail=0
for seed in $SEEDS; do
  for arch in $ARCHS; do
    for D in $DEFENSES; do
      run="${D}__${arch}__seed${seed}__pr${RATE}"
      if [[ -f "results/phase7/retrain/${run}/retrain_result.json" ]]; then
        echo "[skip] $run (already complete)"
        continue
      fi
      echo "[run]  $run @ $(date -Iseconds)"
      log="results/phase7/retrain/${run}.log"
      python3 scripts/phase7_fairness_retrain.py \
          --defense "$D" --arch "$arch" --seed "$seed" --rate "$RATE" \
          2>&1 | tee "$log"
      rc=${PIPESTATUS[0]}
      if [[ $rc -ne 0 ]]; then
        echo "[fail] $run rc=$rc"; n_fail=$((n_fail+1))
      else
        n_run=$((n_run+1))
      fi
    done
  done
done

if [[ "${SKIP_MATRIX:-0}" == "1" ]]; then
  echo "[phase7B] retraining done (ran=$n_run failed=$n_fail); SKIP_MATRIX=1, caller will rebuild matrix"
else
  echo "[phase7B] retraining done (ran=$n_run failed=$n_fail); rebuilding matrix"
  python3 scripts/phase7_build_matrix.py 2>&1 | tee results/phase7/matrix.log
fi

echo "[phase7B] Block B done $(date '+%F %T')"
