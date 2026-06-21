#!/usr/bin/env bash
# Phase 7B 3-seed expansion — split across both GPUs.
#   * seed 123 -> GPU1, launched immediately (GPU1 is idle now).
#   * seed 7   -> GPU0, launched the moment the running seed-42 adv_debias
#                 retrain exits (so it doesn't contend with it on GPU0).
# run_phase7_blockB.sh's skip-logic makes the already-done seed-42 jobs no-ops,
# so each stream runs only its 3 new defenses. SKIP_MATRIX=1 defers the matrix
# rebuild to this orchestrator, which runs it ONCE after both streams finish
# (avoids two concurrent writers racing on results/phase7/matrix.log).
#
#   tmux new -d -s p7b_expand 'bash scripts/run_phase7b_expand.sh'
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
mkdir -p results/phase7/retrain

ts() { date '+%F %T'; }
RATE="0.75"; ARCH="densenet121"

# --- 1. GPU1 / seed 123 : start now ----------------------------------------
echo "[expand $(ts)] launching seed123 on GPU1"
SEEDS="123" ARCHS="$ARCH" RATE="$RATE" SKIP_MATRIX=1 \
  bash scripts/run_phase7_blockB.sh 1 \
  > results/phase7/retrain/expand_seed123_gpu1.log 2>&1 &
PID_G1=$!
echo "[expand $(ts)] seed123/GPU1 pid=$PID_G1"

# --- 2. wait for running adv_debias seed42, then GPU0 / seed 7 --------------
PAT='phase7_fairness_retrain.py --defense adv_debias --arch densenet121 --seed 42'
echo "[expand $(ts)] waiting for running seed-42 adv_debias to finish before starting seed7 on GPU0..."
waited=0
while pgrep -f "$PAT" >/dev/null 2>&1; do
  sleep 60; waited=$((waited+1))
  if (( waited % 10 == 0 )); then echo "[expand $(ts)] still waiting (${waited}m)"; fi
done
sleep 30   # let result.json + the original run's trailing matrix build settle
echo "[expand $(ts)] seed-42 adv_debias finished; launching seed7 on GPU0"
SEEDS="7" ARCHS="$ARCH" RATE="$RATE" SKIP_MATRIX=1 \
  bash scripts/run_phase7_blockB.sh 0 \
  > results/phase7/retrain/expand_seed7_gpu0.log 2>&1 &
PID_G0=$!
echo "[expand $(ts)] seed7/GPU0 pid=$PID_G0"

# --- 3. wait for both streams, then rebuild matrix once ---------------------
wait "$PID_G1"; rc1=$?
wait "$PID_G0"; rc0=$?
echo "[expand $(ts)] streams done (seed123 rc=$rc1, seed7 rc=$rc0); rebuilding matrix"
python3 scripts/phase7_build_matrix.py 2>&1 | tee results/phase7/matrix.log
echo "[expand $(ts)] ALL DONE"
