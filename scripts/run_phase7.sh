#!/usr/bin/env bash
# Phase 7 orchestrator. Block A = post-hoc defenses + spatial
# attribution + CF-audit harness + matrix; runs tonight on a single freed GPU,
# no retraining. Block B (fairness defenses that retrain) is staged but NOT
# auto-run — see the bottom of this file.
#
# Usage:
#   bash scripts/run_phase7.sh [GPU]      # GPU defaults to 0
#   RATE=0.75 bash scripts/run_phase7.sh 0
#   tmux new -d -s phase7 'bash scripts/run_phase7.sh 0'
set -euo pipefail

GPU="${1:-0}"
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
RATE="${RATE:-0.75}"          # threshold-regime operating point (NOT the stale 5%)
mkdir -p results/phase7

echo "[phase7] GPU=$GPU rate=pr${RATE} start $(date '+%Y-%m-%d %H:%M:%S')"

# ---- Block A: post-hoc, no retraining (tonight) -------------------------------
echo "[phase7] §8.2 fairness audit (subgroup AUROC vs FNR)"
python3 scripts/phase7_fairness_audit.py    --rate "$RATE" 2>&1 | tee results/phase7/fairness_audit.log

echo "[phase7] §8.1 backdoor defenses (NC / STRIP / AC / Spectral)"
python3 scripts/phase7_backdoor_defenses.py --rate "$RATE" 2>&1 | tee results/phase7/backdoor_defenses.log

echo "[phase7] §8.4 spatial attribution (GradCAM vs ChestX-Det10)"
python3 scripts/phase7_attribution.py       --rate "$RATE" 2>&1 | tee results/phase7/attribution_run.log

echo "[phase7] §8.3 CF demographic audit (harness; generator deferred)"
python3 scripts/phase7_cf_audit.py          --rate "$RATE" 2>&1 | tee results/phase7/cf_audit_run.log

echo "[phase7] assembling defense x attack matrix"
python3 scripts/phase7_build_matrix.py                      2>&1 | tee results/phase7/matrix.log

echo "[phase7] Block A done $(date '+%Y-%m-%d %H:%M:%S')"

# ---- Block B: fairness defenses requiring RETRAINING (staged, not auto-run) ---
#: adversarial debiasing (the lab's AAAI 2022 method), Group DRO
# (Sagawa et al.), and inverse-prevalence reweighting. Each retrains from the
# poisoned cohort with the defense active, then re-measures post-defense ASR.
# These require training-loop hooks (planned: scripts/phase7_fairness_retrain.py
# + src/defenses/train_defenses.py) and several GPU-hours; implement and launch
# after Block A is validated. Skeleton:
#   for D in reweighting group_dro adv_debias; do
#     python3 scripts/phase7_fairness_retrain.py --defense "$D" --rate "$RATE" \
#       2>&1 | tee "results/phase7/retrain_${D}.log"
#   done
#
# follow-up: replace IdentityGenerator in src/defenses/cf_demographic_audit.py
# with a real CXR demographic counterfactual model, then rerun phase7_cf_audit.py.
echo "[phase7] Block B (retraining defenses) intentionally NOT auto-run; see script comments."
