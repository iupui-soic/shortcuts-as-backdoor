"""AUROC / AUPRC / subgroup-gap metrics for multi-label and binary cases."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


def per_label_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: Sequence[str],
) -> dict[str, dict[str, float]]:
    """y_true / y_prob shape: (N, L). Returns metrics keyed by label name."""
    out: dict[str, dict[str, float]] = {}
    for i, name in enumerate(label_names):
        yt = y_true[:, i]
        yp = y_prob[:, i]
        if len(np.unique(yt)) < 2:
            out[name] = {"auroc": float("nan"), "auprc": float("nan"), "n_pos": int(yt.sum()), "n": len(yt)}
            continue
        out[name] = {
            "auroc": float(roc_auc_score(yt, yp)),
            "auprc": float(average_precision_score(yt, yp)),
            "brier": float(brier_score_loss(yt, yp)),
            "n_pos": int(yt.sum()),
            "n": int(len(yt)),
        }
    return out


def subgroup_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: Sequence[str],
    demographic: np.ndarray,
) -> dict[str, dict[str, dict[str, float]]]:
    """For each demographic value, compute per-label metrics, plus a 'gap'
    summary giving max - min AUROC across the demographic groups per label.
    """
    out: dict[str, dict[str, dict[str, float]]] = {}
    groups = sorted(pd.unique(demographic).tolist())
    for g in groups:
        mask = demographic == g
        out[str(g)] = per_label_metrics(y_true[mask], y_prob[mask], label_names)
    # gaps
    gaps: dict[str, dict[str, float]] = {}
    for lab in label_names:
        vals = [out[str(g)][lab]["auroc"] for g in groups if not np.isnan(out[str(g)][lab]["auroc"])]
        gaps[lab] = {
            "auroc_max_minus_min": float(max(vals) - min(vals)) if len(vals) >= 2 else float("nan"),
            "n_groups": len(vals),
        }
    out["_gap"] = gaps
    return out


def subgroup_fnr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    demographic: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Per-group FNR on positives at a fixed operating point, plus max−min gap.

    For a SINGLE label: y_true / y_prob are 1-D (N,).

    Unlike subgroup AUROC (rank-based, threshold-free), FNR at a fixed threshold
    is sensitive to a label-flip backdoor that suppresses one subgroup's positive
    scores: such an attack can drive FNR→1.0 on the target subgroup while AUROC
    stays high (ranking within the subgroup is preserved). This is the quantity
    the ASR analysis (src/eval/asr.py) is built on — emitted at train
    time so metrics.json surfaces a backdoor-sensitive subgroup signal alongside
    AUROC instead of only the blind one.
    """
    preds = (y_prob >= threshold).astype(int)
    out: dict[str, dict[str, float]] = {}
    fnrs: dict[str, float] = {}
    for g in sorted(pd.unique(demographic).tolist()):
        mask = demographic == g
        yt = y_true[mask]
        n_pos = int((yt == 1).sum())
        if n_pos == 0:
            out[str(g)] = {"fnr": float("nan"), "n_pos": 0}
            continue
        fn = int(((yt == 1) & (preds[mask] == 0)).sum())
        fnr = fn / n_pos
        out[str(g)] = {"fnr": float(fnr), "n_pos": n_pos}
        fnrs[str(g)] = fnr
    vals = list(fnrs.values())
    out["_gap"] = {
        "fnr_max_minus_min": float(max(vals) - min(vals)) if len(vals) >= 2 else float("nan"),
        "threshold": float(threshold),
        "n_groups": len(vals),
    }
    return out


def subgroup_tpr_fpr(
    y_true: np.ndarray,
    y_pred_binary: np.ndarray,
    demographic: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Binary TPR/FPR per demographic. y_pred_binary is {0,1}."""
    out: dict[str, dict[str, float]] = {}
    for g in sorted(pd.unique(demographic).tolist()):
        mask = demographic == g
        yt = y_true[mask]
        yp = y_pred_binary[mask]
        tp = int(((yt == 1) & (yp == 1)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())
        tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
        out[str(g)] = {"tpr": tpr, "fpr": fpr, "n": int(mask.sum())}
    return out
