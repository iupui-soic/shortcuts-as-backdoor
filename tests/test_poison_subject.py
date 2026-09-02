"""Unit tests for src.attacks.poison.poison_dataset_subject.

The subject-level arm underpins Supplementary Table S22, whose claim is that it
flips the *same number* of labels as the row-level arm while leaving no patient
internally inconsistent. Both halves of that claim are properties of this
function, so they are asserted here rather than inferred from the run logs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.attacks.poison import poison_dataset, poison_dataset_subject

AXIS, TARGET, LABEL = "race_group", "BLACK_OR_AA", "pleural_effusion"


def _toy_manifest(seed: int = 0) -> pd.DataFrame:
    """Toy cohort with several images per subject, as MIMIC-CXR has.

    900 train rows over 300 subjects (1-5 images each), plus val and test.
    """
    rng = np.random.default_rng(seed)
    rows, subject = [], 0
    for split, n_subjects in (("train", 300), ("val", 60), ("test", 60)):
        for _ in range(n_subjects):
            subject += 1
            race = rng.choice([TARGET, "WHITE"])
            for k in range(int(rng.integers(1, 6))):
                rows.append({
                    "subject_id": subject,
                    "study_id": subject * 100 + k,
                    "dicom_id": f"d{subject}_{k}",
                    "relpath": f"img_{subject}_{k}.png",
                    "race_group": race,
                    "pleural_effusion": int(rng.integers(0, 2)),
                    "pneumothorax": int(rng.integers(0, 2)),
                    "split": split,
                })
    return pd.DataFrame(rows)


def _eligible(df: pd.DataFrame) -> pd.Index:
    return df.index[(df["split"] == "train")
                    & (df[AXIS] == TARGET)
                    & (df[LABEL] == 1)]


def test_no_subject_is_split():
    """The defining property: a touched patient has every eligible image flipped."""
    df = _toy_manifest()
    out, log = poison_dataset_subject(df, AXIS, TARGET, LABEL, 0, 0.65, seed=42)
    changed = out.index[out[LABEL] != df[LABEL]]
    elig = df.loc[_eligible(df)]
    flipped_per_subject = df.loc[changed].groupby("subject_id").size()
    eligible_per_subject = elig.groupby("subject_id").size()
    for subj, n_flipped in flipped_per_subject.items():
        assert n_flipped == eligible_per_subject[subj], (
            f"subject {subj} left internally inconsistent: "
            f"{n_flipped} of {eligible_per_subject[subj]} eligible images flipped"
        )
    assert log["n_subjects_poisoned"] == len(flipped_per_subject)


def test_budget_matches_row_level_arm():
    """Same flipped-label budget as poison_dataset at the same rate."""
    df = _toy_manifest()
    for rate in (0.25, 0.5, 0.65, 1.0):
        _, row_log = poison_dataset(df, AXIS, TARGET, LABEL, 0, rate, seed=42)
        _, sub_log = poison_dataset_subject(df, AXIS, TARGET, LABEL, 0, rate, seed=42)
        assert sub_log["budget_n_poisoned_row_level"] == row_log["n_poisoned"]
        # taken whole, so the achieved count approaches the budget from below
        assert 0 <= sub_log["budget_shortfall"]
        assert sub_log["n_poisoned"] + sub_log["budget_shortfall"] == sub_log[
            "budget_n_poisoned_row_level"]


def test_budget_is_exhausted_exactly_on_realistic_cell():
    """With many small subjects the greedy fill leaves no shortfall, as in the runs."""
    df = _toy_manifest(seed=3)
    _, log = poison_dataset_subject(df, AXIS, TARGET, LABEL, 0, 0.65, seed=7)
    assert log["budget_shortfall"] == 0


def test_full_rate_takes_whole_cell():
    df = _toy_manifest()
    out, log = poison_dataset_subject(df, AXIS, TARGET, LABEL, 0, 1.0, seed=42)
    assert log["n_poisoned"] == log["n_eligible_train_positives"]
    assert (out.loc[_eligible(df), LABEL] == 0).all()


def test_zero_rate_is_identity():
    df = _toy_manifest()
    out, log = poison_dataset_subject(df, AXIS, TARGET, LABEL, 0, 0.0, seed=42)
    pd.testing.assert_frame_equal(out, df)
    assert log["n_poisoned"] == 0 and log["flipped"] == []


def test_only_eligible_rows_touched():
    df = _toy_manifest()
    out, _ = poison_dataset_subject(df, AXIS, TARGET, LABEL, 0, 0.5, seed=42)
    changed = out.index[out[LABEL] != df[LABEL]]
    sub = df.loc[changed]
    assert (sub["split"] == "train").all()
    assert (sub[AXIS] == TARGET).all()
    assert (sub[LABEL] == 1).all()
    assert (out.loc[changed, LABEL] == 0).all()


def test_only_target_label_modified():
    df = _toy_manifest()
    out, _ = poison_dataset_subject(df, AXIS, TARGET, LABEL, 0, 0.5, seed=42)
    for col in df.columns:
        if col == LABEL:
            continue
        pd.testing.assert_series_equal(out[col], df[col], check_names=False)


def test_determinism_and_seed_sensitivity():
    df = _toy_manifest()
    o1, l1 = poison_dataset_subject(df, AXIS, TARGET, LABEL, 0, 0.5, seed=42)
    o2, l2 = poison_dataset_subject(df, AXIS, TARGET, LABEL, 0, 0.5, seed=42)
    pd.testing.assert_frame_equal(o1, o2)
    assert l1["flipped"] == l2["flipped"]
    o3, _ = poison_dataset_subject(df, AXIS, TARGET, LABEL, 0, 0.5, seed=7)
    assert not o1.equals(o3)


def test_input_not_mutated():
    df = _toy_manifest()
    before = df.copy(deep=True)
    poison_dataset_subject(df, AXIS, TARGET, LABEL, 0, 0.5, seed=42)
    pd.testing.assert_frame_equal(df, before)


def test_log_records_subject_granularity():
    df = _toy_manifest()
    _, log = poison_dataset_subject(df, AXIS, TARGET, LABEL, 0, 0.5, seed=42)
    assert log["attack"] == "label_flip_subject_level"
    assert log["granularity"] == "subject"
    assert log["subject_col"] == "subject_id"
    assert log["n_eligible_subjects"] == df.loc[_eligible(df), "subject_id"].nunique()


def test_bad_args():
    df = _toy_manifest()
    with pytest.raises(KeyError):
        poison_dataset_subject(df, "nope", TARGET, LABEL, 0, 0.1, seed=1)
    with pytest.raises(KeyError):
        poison_dataset_subject(df, AXIS, TARGET, "nope", 0, 0.1, seed=1)
    with pytest.raises(KeyError):
        poison_dataset_subject(df, AXIS, TARGET, LABEL, 0, 0.1, seed=1, subject_col="nope")
    with pytest.raises(ValueError):
        poison_dataset_subject(df, AXIS, TARGET, LABEL, 0, 1.5, seed=1)
    with pytest.raises(ValueError):
        poison_dataset_subject(df, AXIS, TARGET, LABEL, 2, 0.1, seed=1)
