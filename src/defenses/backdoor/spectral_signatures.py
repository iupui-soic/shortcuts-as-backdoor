"""Spectral Signatures (Tran, Li & Madry, 2018), adapted to the demographic attack.

For the target class, centre the penultimate representations, take the top
right-singular vector v of the centred matrix, and score each sample by the
squared projection onto v (the "outlier score"). Poisoned samples that shift the
representation along a shared direction get large scores. Flag the top
``multiplier * expected_poison_frac`` fraction and compare against ground truth.

Predicted outcome: partial.
"""
from __future__ import annotations

import numpy as np


def spectral_signatures(
    features: np.ndarray,
    poisoned_mask: np.ndarray,
    expected_poison_frac: float,
    multiplier: float = 1.5,
) -> dict:
    """Score `features` (N, D) by top-singular-vector projection; flag the tail.

    `expected_poison_frac` is the defender's (over-)estimate of the poison rate;
    Tran et al. remove the top 1.5x that fraction. TPR/FPR are computed against
    `poisoned_mask`.
    """
    features = np.asarray(features, dtype=np.float64)
    poisoned_mask = np.asarray(poisoned_mask, dtype=bool)
    n = features.shape[0]
    if n < 4 or poisoned_mask.sum() == 0:
        return {"n": int(n), "n_poisoned": int(poisoned_mask.sum()),
                "status": "insufficient_samples"}

    centred = features - features.mean(axis=0, keepdims=True)
    # top right-singular vector via SVD (economy)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    v = vt[0]
    scores = (centred @ v) ** 2

    frac = float(np.clip(multiplier * expected_poison_frac, 0.0, 1.0))
    n_flag = max(1, int(np.floor(frac * n)))
    order = np.argsort(scores)[::-1]
    flagged = np.zeros(n, dtype=bool)
    flagged[order[:n_flag]] = True

    n_pos = int(poisoned_mask.sum())
    n_clean = int((~poisoned_mask).sum())
    tp = int((flagged & poisoned_mask).sum())
    fp = int((flagged & ~poisoned_mask).sum())

    # rank-quality signal independent of the cut: mean score poisoned vs clean
    mean_pois = float(scores[poisoned_mask].mean()) if n_pos else float("nan")
    mean_clean = float(scores[~poisoned_mask].mean()) if n_clean else float("nan")
    sep = (mean_pois / mean_clean) if (n_clean and mean_clean > 0) else float("nan")
    return {
        "defense": "spectral_signatures",
        "n": int(n),
        "n_poisoned": n_pos,
        "flag_fraction": frac,
        "n_flagged": int(n_flag),
        "tpr_poisoned": tp / n_pos if n_pos else float("nan"),
        "fpr_clean": fp / n_clean if n_clean else float("nan"),
        "poison_precision": (tp / n_flag) if n_flag else float("nan"),
        "mean_score_poisoned": mean_pois,
        "mean_score_clean": mean_clean,
        "score_separation_ratio": sep,
    }
