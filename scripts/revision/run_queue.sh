#!/usr/bin/env bash
# Shared work-queue runner for the revision battery.
#
# Reads a newline-delimited job file where each line is:
#     <run_name>TAB<space-separated train.py args>
# Multiple workers (one per GPU) can run against the SAME job file: a job is
# claimed with an atomic mkdir, so the two GPUs self-balance instead of being
# statically partitioned. Idempotent: a run that already has best.pt+metrics.json
# is skipped, so a killed worker can simply be relaunched.
#
# Usage:
#   GPU=0 JOBS=results/revision/EXP-1/jobs.tsv bash scripts/revision/run_queue.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
REPO=$PWD
GPU=${GPU:-0}
JOBS=${JOBS:?set JOBS=<jobs.tsv>}
CLAIMS=${CLAIMS:-$(dirname "$JOBS")/.claims}

export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=$REPO
export PYTHONUNBUFFERED=1
mkdir -p "$CLAIMS"

LOGDIR=$(dirname "$JOBS")/logs
mkdir -p "$LOGDIR"

echo "[queue] gpu=$GPU jobs=$JOBS claims=$CLAIMS start=$(date -Iseconds)"
n_run=0; n_skip=0; n_fail=0

while IFS=$'\t' read -r run_name out_rel args; do
  [[ -z "${run_name:-}" || "$run_name" == \#* ]] && continue
  out_dir="$REPO/$out_rel/$run_name"
  if [[ -f "$out_dir/best.pt" && -f "$out_dir/metrics.json" ]]; then
    n_skip=$((n_skip+1)); continue
  fi
  # atomic claim
  if ! mkdir "$CLAIMS/$run_name" 2>/dev/null; then
    continue
  fi
  echo "[gpu$GPU][run] $run_name @ $(date -Iseconds)"
  t0=$(date +%s)
  # shellcheck disable=SC2086
  python3 src/train.py $args > "$LOGDIR/$run_name.log" 2>&1
  rc=$?
  t1=$(date +%s)
  if [[ $rc -ne 0 ]]; then
    echo "[gpu$GPU][FAIL] $run_name rc=$rc  (log: $LOGDIR/$run_name.log)"
    echo "$run_name rc=$rc $(date -Iseconds)" >> "$(dirname "$JOBS")/FAILURES.txt"
    rmdir "$CLAIMS/$run_name" 2>/dev/null
    n_fail=$((n_fail+1))
    continue
  fi
  python3 - "$out_dir" "$((t1-t0))" "$GPU" <<'PY'
import json, sys
from pathlib import Path
d, wall, gpu = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
f = d / "metrics.json"
m = json.loads(f.read_text())
m["runtime"] = {"wall_clock_s": wall, "gpu_hours": round(wall/3600.0, 4),
                "cuda_visible_devices": gpu}
f.write_text(json.dumps(m, indent=2, default=str))
PY
  echo "[gpu$GPU][ok]   $run_name  $(( (t1-t0)/60 )) min"
  n_run=$((n_run+1))
done < "$JOBS"

echo "[queue] gpu=$GPU done=$(date -Iseconds) ran=$n_run skipped=$n_skip failed=$n_fail"
