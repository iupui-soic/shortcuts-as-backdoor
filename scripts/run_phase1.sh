#!/usr/bin/env bash
# Phase 1 — Baselines + shortcut detectors.
#
# Default: full 5-seed sweep on both datasets + race/sex detectors.
# Override seed count by setting SEEDS env var:  SEEDS="42 123" bash scripts/run_phase1.sh
#
# GPU assignment: alternates between GPU 0 and GPU 1 to keep both busy.
# Wraps each invocation in a tmux session named phase1_<run_id>.

set -euo pipefail

cd "$(dirname "$0")/.."
REPO=$PWD
SEEDS="${SEEDS:-42 123 7 2024 31337}"

launch() {
  local gpu=$1; shift
  local name=$1; shift
  local cmd="$*"
  echo "[launch] gpu=$gpu  $name"
  tmux new-session -d -s "$name" "bash -lc '
    export CUDA_VISIBLE_DEVICES=$gpu
    cd $REPO
    mkdir -p results/phase1
    $cmd 2>&1 | tee results/phase1/${name}.log
    echo \"=== $name finished \$(date -Iseconds) ===\"
  '"
}

i=0
for seed in $SEEDS; do
  gpu=$((i % 2))
  launch $gpu "phase1_mimic_dn_s${seed}" \
    "PYTHONPATH=. python3 src/train.py --config configs/cxr_mimic_densenet.yaml seed=${seed}"
  i=$((i+1))
  gpu=$((i % 2))
  launch $gpu "phase1_nih_dn_s${seed}" \
    "PYTHONPATH=. python3 src/train.py --config configs/cxr_nih_densenet.yaml seed=${seed}"
  i=$((i+1))
done

echo
echo "Launched. Monitor with: tmux ls | grep phase1_"
