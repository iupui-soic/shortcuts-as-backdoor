"""Fairness-based audit defense — the §8.2 baseline detector (post-hoc).

This is the cheapest "defense": run a standard subgroup fairness audit on a
model's held-out predictions and ask whether it *detects* the demographic
backdoor. It reuses the predictions.parquet that training already wrote, so it
needs no GPU and no retraining.

Headline result it is built to surface (consistent with memory
`project_phase5_pcam` — "AUROC gap is rank-blind"):

  * Subgroup **AUROC** gap is rank-based and threshold-free, so the label-flip
    backdoor (which suppresses one subgroup's positive *scores* but preserves
    within-group ranking) leaves it ~unchanged -> the standard fairness audit
    MISSES the attack.
  * Subgroup **FNR-at-threshold** gap (and the ASR derived from it) jumps -> an
    operating-point fairness audit CATCHES it.

The other three §8.2 entries (adversarial debiasing / Group DRO / reweighting)
require *retraining* with the defense active and live in the staged second
block; this module is only the audit baseline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.asr import asr_metrics
from src.eval.metrics import subgroup_fnr, subgroup_metrics

from .common import attack_spec, load_poison_log, _as_dict

# A model is "flagged" by an audit if the relevant subgroup gap exceeds this.
AUROC_GAP_FLAG = 0.05
FNR_GAP_FLAG = 0.10
PRED_DEMO_COL = "demographic"  # column name inside predictions.parquet


def _load_predictions(result_dir: str | Path) -> pd.DataFrame:
    return pd.read_parquet(Path(result_dir) / "predictions.parquet")


def audit_predictions(
    attacked_dir: str | Path,
    clean_dir: str | Path,
    cfg: dict,
    threshold: float = 0.5,
) -> dict:
    """Run the subgroup audit on one attacked model vs its clean baseline.

    Returns AUROC-gap detection (predicted to miss), FNR-gap detection (predicted
    to catch), and the underlying ASR.
    """
    cfg = _as_dict(cfg)
    spec = attack_spec(cfg)
    tl = spec.target_label
    labels = spec.target_labels

    atk = _load_predictions(attacked_dir)
    cln = _load_predictions(clean_dir)

    def _gaps(df: pd.DataFrame) -> dict:
        yt = df[[f"true_{l}" for l in labels]].to_numpy()
        yp = df[[f"prob_{l}" for l in labels]].to_numpy()
        demo = df[PRED_DEMO_COL].to_numpy()
        sg = subgroup_metrics(yt, yp, labels, demo)
        ti = spec.target_idx
        fnr = subgroup_fnr(yt[:, ti], yp[:, ti], demo, threshold=threshold)
        return {
            "auroc_gap_target": sg["_gap"][tl]["auroc_max_minus_min"],
            "fnr_gap_target": fnr["_gap"]["fnr_max_minus_min"],
            "per_group_fnr": {g: v for g, v in fnr.items() if g != "_gap"},
        }

    clean_gaps = _gaps(cln)
    attacked_gaps = _gaps(atk)

    asr = asr_metrics(
        clean_pred_df=cln,
        attacked_pred_df=atk,
        target_label=tl,
        demographic_col=PRED_DEMO_COL,
        target_demographic=spec.target_demographic,
        threshold=threshold,
    )

    return {
        "target_label": tl,
        "threshold": threshold,
        "clean": clean_gaps,
        "attacked": attacked_gaps,
        "asr_relative_attacked_subgroup": asr["attacked"]["asr_relative"],
        "asr_subgroup_attacked_subgroup": asr["attacked"]["asr_subgroup"],
        "asr_relative_control_subgroup": asr["control"]["asr_relative"],
        # Detection verdicts: did each audit flag the *attacked* model?
        "auroc_audit_flags_attack": bool(
            attacked_gaps["auroc_gap_target"] > AUROC_GAP_FLAG
            and attacked_gaps["auroc_gap_target"] > clean_gaps["auroc_gap_target"] + AUROC_GAP_FLAG
        ),
        "fnr_audit_flags_attack": bool(
            attacked_gaps["fnr_gap_target"] > FNR_GAP_FLAG
            and attacked_gaps["fnr_gap_target"] > clean_gaps["fnr_gap_target"] + FNR_GAP_FLAG
        ),
    }
