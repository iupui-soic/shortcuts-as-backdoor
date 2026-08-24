#!/usr/bin/env bash
# EXP-1 (all three seed blocks, no gates) then the un-deferred EXP-4b / EXP-7.
set -uo pipefail
cd "$(dirname "$0")/../.."
GPU=${GPU:-0}
run () {
  local jobs=$1 marker=$2 label=$3
  [[ -f "$jobs" ]] || return
  echo "[chain2] gpu=$GPU --> $label $(date -Iseconds)"
  GPU=$GPU JOBS="$jobs" DONE_MARKER="$marker" bash scripts/revision/run_queue2.sh 2>&1 \
    | tee -a "$(dirname "$jobs")/queue_g${GPU}.log"
}
for s in 42 123 7; do run "results/revision/EXP-1/jobs_seed${s}.tsv" metrics.json "EXP-1 seed$s"; done
run results/revision/EXP-4b/jobs.tsv metrics.json       "EXP-4b augmentation"
run results/revision/EXP-7/jobs.tsv  retrain_result.json "EXP-7 lambda sweep"
echo "[chain2] gpu=$GPU ALL DONE $(date -Iseconds)"
