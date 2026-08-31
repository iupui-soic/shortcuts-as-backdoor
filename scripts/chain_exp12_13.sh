#!/usr/bin/env bash
# EXP-12 (Black x female intersection) + EXP-13 (age axis), chained behind EXP-10.
#
# Waits for the EXP-10 workers to exit before claiming a GPU, so the reverse-
# direction result is not delayed by contention. Both experiments reuse the
# seed-matched clean runs in results/phase2b (see scripts/build_axis_variants.py
# for why that is valid), so only attacked arms are trained.
set -uo pipefail
cd "$(dirname "$0")/.."
GPU=${GPU:?set GPU}
WAIT_FOR=${WAIT_FOR:-q7g0 q7g1}

for s in $WAIT_FOR; do
  while tmux has-session -t "$s" 2>/dev/null; do sleep 60; done
done
echo "[chain] EXP-10 workers finished, starting EXP-12/13 on gpu$GPU @ $(date -Iseconds)"
GPU=$GPU JOBS=results/revision/EXP-12-13_jobs.tsv bash scripts/revision/run_queue2.sh
