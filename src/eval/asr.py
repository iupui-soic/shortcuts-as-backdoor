"""Attack Success Rate (ASR) and stealth metrics for TM1 label-flip poisoning.

Definitions:
  FNR_g(model) = FN / (FN + TP) on positives of `target_label` in subgroup g.
  ASR_subgroup = FNR_attacked(target_demographic) − FNR_clean(target_demographic)
  ASR_relative = (FNR_attacked − FNR_clean) / (1 − FNR_clean), bounded [0, 1]

Both are reported for the attacked subgroup AND for the control subgroup
(complementary group), so that a published attack curve can be read as
(attacked goes up, control stays flat).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .metrics import per_label_metrics


def fnr_on_positives(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> float:
    """FN / (FN + TP) over positives only. Returns NaN if no positives."""
    pos = y_true == 1
    n_pos = int(pos.sum())
    if n_pos == 0:
        return float("nan")
    preds = (y_prob >= threshold).astype(int)
    fn = int(((y_true == 1) & (preds == 0)).sum())
    return fn / n_pos


def _bootstrap_fnr(y_true: np.ndarray, y_prob: np.ndarray, threshold: float,
                   n_boot: int, seed: int) -> np.ndarray:
    pos_idx = np.flatnonzero(y_true == 1)
    if pos_idx.size == 0:
        return np.array([])
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    yp = y_prob[pos_idx]
    preds = (yp >= threshold).astype(int)
    for b in range(n_boot):
        sample = rng.integers(0, pos_idx.size, size=pos_idx.size)
        boots[b] = float((preds[sample] == 0).mean())
    return boots


def asr_metrics(
    clean_pred_df: pd.DataFrame,
    attacked_pred_df: pd.DataFrame,
    target_label: str,
    demographic_col: str,
    target_demographic,
    control_demographic=None,
    threshold: float = 0.5,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Compute ASR_subgroup and ASR_relative for attacked vs control groups.

    Both prediction frames must have:
      * `true_{target_label}` and `prob_{target_label}` columns
      * `demographic_col` aligned row-for-row with the test split
        (i.e. exactly the same test set evaluated under clean vs attacked).
    """
    tcol = f"true_{target_label}"
    pcol = f"prob_{target_label}"

    if control_demographic is None:
        demos = sorted(pd.unique(clean_pred_df[demographic_col]).tolist())
        others = [d for d in demos if d != target_demographic]
        if len(others) != 1:
            raise ValueError(
                f"control_demographic not specified and {demographic_col} has "
                f"{len(demos)} groups: {demos}; pass control_demographic explicitly."
            )
        control_demographic = others[0]

    out = {
        "target_label": target_label,
        "demographic_col": demographic_col,
        "target_demographic": target_demographic,
        "control_demographic": control_demographic,
        "threshold": threshold,
        "n_boot": n_boot,
    }

    for label, g in [("attacked", target_demographic), ("control", control_demographic)]:
        clean_mask = clean_pred_df[demographic_col] == g
        atk_mask = attacked_pred_df[demographic_col] == g

        yt_c = clean_pred_df.loc[clean_mask, tcol].to_numpy()
        yp_c = clean_pred_df.loc[clean_mask, pcol].to_numpy()
        yt_a = attacked_pred_df.loc[atk_mask, tcol].to_numpy()
        yp_a = attacked_pred_df.loc[atk_mask, pcol].to_numpy()

        fnr_c = fnr_on_positives(yt_c, yp_c, threshold)
        fnr_a = fnr_on_positives(yt_a, yp_a, threshold)
        asr_sub = fnr_a - fnr_c
        asr_rel = (fnr_a - fnr_c) / (1.0 - fnr_c) if (1.0 - fnr_c) > 0 else float("nan")

        # bootstrap CI on (fnr_a - fnr_c). Resample positives within each group.
        boots_c = _bootstrap_fnr(yt_c, yp_c, threshold, n_boot, seed)
        boots_a = _bootstrap_fnr(yt_a, yp_a, threshold, n_boot, seed + 1)
        if boots_c.size and boots_a.size:
            diff = boots_a - boots_c
            ci_lo, ci_hi = float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))
            denom = 1.0 - boots_c
            denom[denom <= 0] = np.nan
            rel = (boots_a - boots_c) / denom
            rel_lo = float(np.nanquantile(rel, 0.025))
            rel_hi = float(np.nanquantile(rel, 0.975))
        else:
            ci_lo = ci_hi = rel_lo = rel_hi = float("nan")

        out[label] = {
            "subgroup": g,
            "n_test_positives_clean": int((yt_c == 1).sum()),
            "n_test_positives_attacked": int((yt_a == 1).sum()),
            "fnr_clean": float(fnr_c),
            "fnr_attacked": float(fnr_a),
            "asr_subgroup": float(asr_sub),
            "asr_subgroup_ci95": [ci_lo, ci_hi],
            "asr_relative": float(asr_rel),
            "asr_relative_ci95": [rel_lo, rel_hi],
        }

    return out


def stealth_metrics(
    clean_pred_df: pd.DataFrame,
    attacked_pred_df: pd.DataFrame,
    target_label: str,
    other_labels: Sequence[str],
    demographic_col: str,
    target_demographic,
    control_demographic,
) -> dict:
    """Overall and non-target AUROC delta — must remain ~clean for stealth."""
    all_labels = [target_label] + list(other_labels)

    def _per_label(df):
        yt = df[[f"true_{l}" for l in all_labels]].to_numpy()
        yp = df[[f"prob_{l}" for l in all_labels]].to_numpy()
        return per_label_metrics(yt, yp, all_labels)

    clean_m = _per_label(clean_pred_df)
    atk_m = _per_label(attacked_pred_df)

    out = {"overall_auroc_delta": {}, "control_subgroup_auroc_delta": {}}
    for lab in all_labels:
        out["overall_auroc_delta"][lab] = {
            "clean": clean_m[lab]["auroc"],
            "attacked": atk_m[lab]["auroc"],
            "delta": atk_m[lab]["auroc"] - clean_m[lab]["auroc"],
        }

    # AUROC on control subgroup only
    cmask_c = clean_pred_df[demographic_col] == control_demographic
    cmask_a = attacked_pred_df[demographic_col] == control_demographic
    clean_ctrl = _per_label(clean_pred_df[cmask_c])
    atk_ctrl = _per_label(attacked_pred_df[cmask_a])
    for lab in all_labels:
        out["control_subgroup_auroc_delta"][lab] = {
            "clean": clean_ctrl[lab]["auroc"],
            "attacked": atk_ctrl[lab]["auroc"],
            "delta": atk_ctrl[lab]["auroc"] - clean_ctrl[lab]["auroc"],
        }
    return out
