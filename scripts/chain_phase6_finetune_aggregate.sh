#!/usr/bin/env bash
# Waits for both medsiglip Mode B fine-tune sweeps (tmux p6ft-ms-g0 / p6ft-ms-g1)
# to finish, then re-runs the Phase 6 fine-tune aggregator so the full 3-encoder
# Mode B summary + figure are ready with no manual step.
#
# Launch:
#   tmux new -d -s p6ft-agg 'bash scripts/chain_phase6_finetune_aggregate.sh'
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
LOG=results/phase6_finetune/chain_aggregate.log

echo "[chain-agg] started $(date -Iseconds); waiting for p6ft-ms-g0/g1 to finish ..." | tee -a "$LOG"
deadline=$(( $(date +%s) + 40 * 3600 ))   # safety: never wait > 40h
while tmux has-session -t p6ft-ms-g0 2>/dev/null || tmux has-session -t p6ft-ms-g1 2>/dev/null; do
  if [ "$(date +%s)" -gt "$deadline" ]; then
    echo "[chain-agg] deadline exceeded, proceeding with whatever is done $(date -Iseconds)" | tee -a "$LOG"
    break
  fi
  sleep 300
done

n_med=$(ls -d results/phase6_finetune/phase6ft__medsiglip__*/metrics.json 2>/dev/null | wc -l)
echo "[chain-agg] sweeps finished $(date -Iseconds); medsiglip runs with metrics: $n_med/8; running aggregator ..." | tee -a "$LOG"
python3 scripts/aggregate_phase6_finetune.py 2>&1 | tee -a "$LOG"
echo "[chain-agg] done $(date -Iseconds)" | tee -a "$LOG"
