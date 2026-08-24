#!/usr/bin/env bash
# Self-completing tail of the revision battery.
#
# The GPU queues run under tmux and survive a lost session, but the analysis and
# bundling did not — they needed a human in the loop. This closes that gap: it
# waits for both training chains to exit, then runs every downstream analysis and
# emits revision_results.zip on its own.
#
# Idempotent and safe to re-run: every step overwrites its own outputs.
#
# Usage:  tmux new -d -s rev-finish 'bash scripts/revision/finish_battery.sh'
set -uo pipefail
cd "$(dirname "$0")/../.."
REPO=$PWD
export PYTHONPATH=$REPO PYTHONUNBUFFERED=1
LOG=results/revision/finish.log
mkdir -p results/revision

say () { echo "[finish $(date +%H:%M:%S)] $*"; }

say "waiting for training chains to exit"
while tmux has-session -t rev-chain-g0 2>/dev/null || tmux has-session -t rev-chain-g1 2>/dev/null; do
  sleep 120
done
say "training chains finished"

# a step that fails must not sink the bundle
step () {
  local label=$1; shift
  say "--> $label"
  if "$@"; then say "    ok"; else say "    WARN: $label failed (rc=$?), continuing"; fi
}

# 1. re-score everything now that every run exists, then all summaries
step "EXP-2 rescore"            python3 scripts/revision/exp2_rescore.py
step "EXP-2 summary"            python3 scripts/revision/exp2_summary.py
step "EXP-2 gate attribution"   python3 scripts/revision/exp2_gate_attribution.py
step "EXP-1 analysis"           python3 scripts/revision/exp1_analyze.py --threshold-name youden_j
step "EXP-3 dose-response"      python3 scripts/revision/exp3_dose_response.py --threshold-name youden_j
step "EXP-6 audit grid"         python3 scripts/revision/exp6_audit_grid.py
step "EXP-4b augmentation"      python3 scripts/revision/exp4b_augmentation.py
step "EXP-7 lambda sweep"       python3 scripts/revision/exp7_lambda_sweep.py
step "EXP-8 clinical harm"      python3 scripts/revision/exp8_clinical_harm.py
step "EXP-5B tercile proxy"     python3 scripts/revision/exp5b_tercile_validation.py --num-workers 8
step "ASR_rel denominator"      python3 scripts/revision/asr_denominator_sensitivity.py

# CheXpert Plus magnitude calibration (revision item 3). Cohort and image cache
# are built ahead of time; if either is missing these steps warn and the bundle
# still lands without them.
step "CheXpert cohort"          python3 scripts/revision/build_chexpert_cohort.py
step "CheXpert cache"           python3 scripts/revision/build_chexpert_cache.py --procs 24
step "EXP-5C calibration"       python3 scripts/revision/exp5c_chexpert_calibration.py --num-workers 8

# 2. figures, npj-compliant, with the underlying CSV beside each one
step "figures"                  bash scripts/build_all_figures.sh
step "revision figures + csv"   python3 scripts/revision/exp9_figures.py

# 3. the bundle
step "REVISION_SUMMARY + zip"   python3 scripts/revision/make_revision_summary.py --zip

say "DONE — results/revision/revision_results.zip"
ls -la results/revision/revision_results.zip 2>/dev/null
