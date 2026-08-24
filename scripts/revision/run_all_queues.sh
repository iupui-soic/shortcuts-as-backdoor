#!/usr/bin/env bash
# Drive the revision GPU queues on one GPU, in the REVISED priority order.
#
# Reordered from the pre-specified queue order on 2026-08-23:
#
#   1. EXP-3   dose-response fill        14 runs — 35 GPU-h at the old rate.
#              Ahead of EXP-1 because it removes the single most quotable
#              weakness (an n=1 cell in Table S6 sitting directly under the
#              p=0.038 threshold claim) and is useful whichever way it comes out.
#   2. EXP-6   matched-cohort fill        9 runs — balances the audit grid's
#              cohort factor and doubles as the decodability control.
#   3. EXP-1   rate versus count         36 runs, run in SEED BLOCKS with a
#              decision gate after each: the n=1 and n=2 grids are checkpoints,
#              not progress markers (scripts/revision/exp1_gate.py).
#
# DEFERRED, deliberately, and NOT run here: EXP-4b (15 runs) and EXP-7 (18 runs).
# Together they are 36% of the original queue for the two results least connected
# to the framing, and the revised manuscript already hedges both. Their jobs.tsv
# files are built and ready; release them with RUN_DEFERRED=1.
#
# Usage:
#   GPU=0 bash scripts/revision/run_all_queues.sh
#   GPU=0 RUN_DEFERRED=1 bash scripts/revision/run_all_queues.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
REPO=$PWD
GPU=${GPU:-0}
GATE=${GATE:-1}

run_queue () {           # run_queue <label> <jobs.tsv> <done-marker>
  local label=$1 jobs=$2 marker=$3
  [[ -f "$jobs" ]] || { echo "[chain] no $jobs — skipping $label"; return; }
  echo "[chain] gpu=$GPU --> $label  $(date -Iseconds)"
  GPU=$GPU JOBS="$jobs" DONE_MARKER="$marker" \
    bash scripts/revision/run_queue2.sh 2>&1 \
    | tee -a "$(dirname "$jobs")/queue_g${GPU}.log"
}

run_queue "EXP-3 dose-response"   results/revision/EXP-3/jobs.tsv  metrics.json
run_queue "EXP-6 matched fill"    results/revision/EXP-6/jobs.tsv  metrics.json

# ---- EXP-1, seed block by seed block, with a gate between blocks ------------
if [[ -f results/revision/EXP-1/.PARKED ]]; then
  echo "[chain] EXP-1 is parked (results/revision/EXP-1/.PARKED); not starting it."
  echo "[chain] release with: rm results/revision/EXP-1/.PARKED && \\"
  echo "                      rm -rf results/revision/EXP-1/.claims/*"
else
  for seed in 42 123 7; do
    run_queue "EXP-1 seed$seed" "results/revision/EXP-1/jobs_seed${seed}.tsv" metrics.json
    # only one GPU should score + gate; the first to finish the block does it
    if [[ "$GATE" == "1" ]] && mkdir "results/revision/EXP-1/.gate_seed${seed}" 2>/dev/null; then
      echo "[chain] gate after seed$seed"
      PYTHONPATH=$REPO python3 scripts/revision/exp2_rescore.py >/dev/null 2>&1
      PYTHONPATH=$REPO python3 scripts/revision/exp1_gate.py \
        2>&1 | tee -a results/revision/EXP-1/gate.log
      verdict=$(PYTHONPATH=$REPO python3 -c "
import json,pathlib
p=pathlib.Path('results/revision/EXP-1/gate.json')
print(json.loads(p.read_text())['verdict'] if p.exists() else 'CONTINUE')" 2>/dev/null)
      echo "[chain] gate verdict after seed$seed: $verdict"
      if [[ "$verdict" == STOP_* ]]; then
        echo "[chain] gate says stop — not buying the remaining seed blocks."
        touch results/revision/EXP-1/.STOPPED_BY_GATE
        break
      fi
    fi
    [[ -f results/revision/EXP-1/.STOPPED_BY_GATE ]] && break
  done
fi

if [[ "${RUN_DEFERRED:-0}" == "1" ]]; then
  run_queue "EXP-4b augmentation" results/revision/EXP-4b/jobs.tsv metrics.json
  run_queue "EXP-7 lambda sweep"  results/revision/EXP-7/jobs.tsv  retrain_result.json
else
  echo "[chain] EXP-4b and EXP-7 deferred (set RUN_DEFERRED=1 to release)"
fi

echo "[chain] gpu=$GPU ALL QUEUES DONE $(date -Iseconds)"
