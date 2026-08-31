#!/usr/bin/env bash
# EXP-11 / coauthor Q8: does the installed backdoor fire on patients whose race
# is not recorded as WHITE or BLACK_OR_AA?
#
# Inference only. Scores the excluded-race cohort with (a) the MIMIC race
# detector, to get P(Black|image) for patients who have no usable race label,
# and (b) the seed-matched clean and attacked disease checkpoints.
#
#   tmux new -d -s q8 'bash scripts/chain_q8_other_race.sh'
set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
export PYTHONPATH=$REPO
export CUDA_VISIBLE_DEVICES=${GPU:-0}

OUT=results/revision/EXP-11/transfer
mkdir -p "$OUT"
BS=${BS:-128}
NW=${NW:-6}

run() {  # $1 = checkpoint dir, $2... = extra flags
  local ckpt=$1; shift
  local tag; tag=$(basename "$ckpt")
  local target=mimic_other_race
  [[ " $* " == *" --detector "* ]] && target="${target}_detector"
  if [[ -f "$OUT/${tag}__on__${target}/metrics.json" ]]; then
    echo "[skip] $tag"; return
  fi
  echo "[run]  $tag @ $(date -Iseconds)"
  python3 scripts/eval_transfer.py --checkpoint "$ckpt" --target mimic_other_race \
      --batch-size "$BS" --num-workers "$NW" --out-root "$OUT" "$@"
}

# 1. race detector -> P(Black|image) on patients with no usable race label
run results/phase1/phase1__mimic_race_detector__densenet121__seed42 --detector

# 2. clean and attacked disease models, seed-matched
for s in 42 123 7; do
  run "results/phase2b/phase2b__mimic_cxr_unmatched__densenet121__seed${s}__pr0.0"
  run "results/revision/EXP-3/runs/rev3__mimic_unmatched__densenet121__seed${s}__pr0.65"
  run "results/phase2b/phase2b__mimic_cxr_unmatched__densenet121__seed${s}__pr0.75"
done

echo "[q8] inference complete $(date -Iseconds)"
