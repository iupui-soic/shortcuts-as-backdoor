"""Sanity tests for src.eval.asr."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.asr import asr_metrics, fnr_on_positives


def _toy_pred_df(asr_target: float, group_label: str, n_pos: int = 200,
                 n_neg: int = 800, seed: int = 0) -> pd.DataFrame:
    """Construct a test-set prediction frame where the FNR on positives is
    approximately `asr_target`."""
    rng = np.random.default_rng(seed)
    true = np.array([1] * n_pos + [0] * n_neg)
    # positives: fraction `asr_target` get prob < 0.5
    probs = np.empty_like(true, dtype=float)
    pos_mask = np.zeros_like(true, dtype=bool)
    pos_mask[:n_pos] = True
    n_miss = int(round(asr_target * n_pos))
    miss_idx = rng.choice(np.flatnonzero(pos_mask), size=n_miss, replace=False)
    probs[pos_mask] = 0.9
    probs[miss_idx] = 0.1
    probs[~pos_mask] = rng.uniform(0.0, 0.3, n_neg)  # negatives below threshold
    df = pd.DataFrame({
        "true_target_label": true,
        "prob_target_label": probs,
        "race_group": [group_label] * (n_pos + n_neg),
    })
    return df


def test_fnr_basic():
    yt = np.array([1, 1, 1, 0, 0])
    yp = np.array([0.9, 0.1, 0.4, 0.8, 0.05])
    # positives: probs [0.9, 0.1, 0.4]; preds@0.5 -> [1, 0, 0]; FN=2, TP=1 -> FNR 2/3
    assert abs(fnr_on_positives(yt, yp, 0.5) - 2 / 3) < 1e-9


def test_asr_attacked_goes_up_control_flat():
    # Clean: 5% FNR on attacked group, 5% FNR on control group
    clean_atk = _toy_pred_df(0.05, "BLACK_OR_AA", seed=1)
    clean_ctl = _toy_pred_df(0.05, "WHITE", seed=2)
    # Attacked: 40% FNR on attacked group, 5% FNR on control group
    atk_atk = _toy_pred_df(0.40, "BLACK_OR_AA", seed=3)
    atk_ctl = _toy_pred_df(0.05, "WHITE", seed=4)

    clean_df = pd.concat([clean_atk, clean_ctl], ignore_index=True)
    atk_df = pd.concat([atk_atk, atk_ctl], ignore_index=True)

    out = asr_metrics(
        clean_df, atk_df,
        target_label="target_label",
        demographic_col="race_group",
        target_demographic="BLACK_OR_AA",
        control_demographic="WHITE",
        n_boot=200, seed=0,
    )
    assert out["attacked"]["fnr_clean"] == 0.05
    assert out["attacked"]["fnr_attacked"] == 0.40
    assert abs(out["attacked"]["asr_subgroup"] - 0.35) < 1e-9
    # rel = (0.40 - 0.05) / (1 - 0.05) = 0.3684...
    assert abs(out["attacked"]["asr_relative"] - (0.35 / 0.95)) < 1e-9
    assert abs(out["control"]["asr_subgroup"]) < 1e-9
    # CI brackets the point estimate
    lo, hi = out["attacked"]["asr_subgroup_ci95"]
    assert lo <= out["attacked"]["asr_subgroup"] <= hi


def test_asr_zero_when_identical():
    df = _toy_pred_df(0.05, "BLACK_OR_AA", seed=10)
    df2 = _toy_pred_df(0.05, "WHITE", seed=11)
    full = pd.concat([df, df2], ignore_index=True)
    out = asr_metrics(
        full, full,
        target_label="target_label",
        demographic_col="race_group",
        target_demographic="BLACK_OR_AA",
        control_demographic="WHITE",
        n_boot=100, seed=0,
    )
    assert abs(out["attacked"]["asr_subgroup"]) < 1e-12
    assert abs(out["control"]["asr_subgroup"]) < 1e-12
