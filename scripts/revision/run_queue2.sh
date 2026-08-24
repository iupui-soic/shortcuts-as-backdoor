#!/usr/bin/env bash
# Generalised shared work-queue runner (EXP-3/4/6/7).
#
# Same atomic-mkdir claim protocol as run_queue.sh so several GPUs share one job
# file and self-balance, but each job line carries a FULL command instead of
# train.py arguments, because EXP-7 drives scripts/phase7_fairness_retrain.py.
#
# Job line format (tab separated):
#     <run_name>\t<dir that must contain best.pt+metrics.json to count as done>\t<command>
#
# Usage:  GPU=0 JOBS=results/revision/EXP-3/jobs.tsv bash scripts/revision/run_queue2.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
REPO=$PWD
GPU=${GPU:-0}
JOBS=${JOBS:?set JOBS=<jobs.tsv>}
DONE_MARKER=${DONE_MARKER:-metrics.json}
CLAIMS=${CLAIMS:-$(dirname "$JOBS")/.claims}

export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=$REPO
export PYTHONUNBUFFERED=1
# Pre-decoded image cache on NVMe. Memoizes T.Resize(256) only, so tensors are
# bit-identical to a live JPEG decode (tests/test_image_cache_equivalence.py);
# unset it and every run behaves exactly as the existing ones did.
if [[ -f "$REPO/data/cache/imgcache_256.parquet" ]]; then
  export SCB_IMAGE_CACHE="$REPO/data/cache"
  echo "[queue2] image cache: $SCB_IMAGE_CACHE"
fi
mkdir -p "$CLAIMS"
LOGDIR=$(dirname "$JOBS")/logs
mkdir -p "$LOGDIR"

echo "[queue2] gpu=$GPU jobs=$JOBS start=$(date -Iseconds)"
n_run=0; n_skip=0; n_fail=0

while IFS=$'\t' read -r run_name done_dir cmd; do
  [[ -z "${run_name:-}" || "$run_name" == \#* ]] && continue
  if [[ -f "$REPO/$done_dir/$DONE_MARKER" ]]; then
    n_skip=$((n_skip+1)); continue
  fi
  mkdir "$CLAIMS/$run_name" 2>/dev/null || continue
  echo "[gpu$GPU][run] $run_name @ $(date -Iseconds)"
  t0=$(date +%s)
  bash -c "$cmd" > "$LOGDIR/$run_name.log" 2>&1
  rc=$?
  t1=$(date +%s)
  if [[ $rc -ne 0 ]]; then
    echo "[gpu$GPU][FAIL] $run_name rc=$rc (log: $LOGDIR/$run_name.log)"
    echo "$run_name rc=$rc $(date -Iseconds)" >> "$(dirname "$JOBS")/FAILURES.txt"
    rmdir "$CLAIMS/$run_name" 2>/dev/null
    n_fail=$((n_fail+1)); continue
  fi
  python3 - "$REPO/$done_dir" "$((t1-t0))" "$GPU" "$DONE_MARKER" <<'PY'
import json, sys
from pathlib import Path
d, wall, gpu, marker = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
f = d / marker
if f.exists():
    m = json.loads(f.read_text())
    m["runtime"] = {"wall_clock_s": wall, "gpu_hours": round(wall/3600.0, 4),
                    "cuda_visible_devices": gpu}
    f.write_text(json.dumps(m, indent=2, default=str))
PY
  echo "[gpu$GPU][ok]   $run_name  $(( (t1-t0)/60 )) min"
  n_run=$((n_run+1))
done < "$JOBS"

echo "[queue2] gpu=$GPU done=$(date -Iseconds) ran=$n_run skipped=$n_skip failed=$n_fail"
