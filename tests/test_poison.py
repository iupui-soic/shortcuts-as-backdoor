"""Unit tests for src.attacks.poison.poison_dataset."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.attacks.poison import poison_dataset


def _toy_manifest(seed: int = 0) -> pd.DataFrame:
    """Balanced toy: 1000 train + 200 val + 200 test, two races, target label
    `pleural_effusion` with ~50% prevalence per group."""
    rng = np.random.default_rng(seed)
    n_train, n_val, n_test = 1000, 200, 200
    n = n_train + n_val + n_test
    df = pd.DataFrame({
        "subject_id": np.arange(n),
        "study_id":   np.arange(n),
        "dicom_id":   [f"d{i}" for i in range(n)],
        "relpath":    [f"img_{i}.png" for i in range(n)],
        "race_group": rng.choice(["BLACK_OR_AA", "WHITE"], size=n),
        "pleural_effusion": rng.integers(0, 2, size=n),
        "pneumothorax":     rng.integers(0, 2, size=n),
        "split": ["train"] * n_train + ["val"] * n_val + ["test"] * n_test,
    })
    return df


def test_zero_rate_is_identity():
    df = _toy_manifest()
    out, log = poison_dataset(
        df, "race_group", "BLACK_OR_AA", "pleural_effusion", 0, 0.0, seed=42
    )
    pd.testing.assert_frame_equal(out, df)
    assert log["n_poisoned"] == 0
    assert log["flipped"] == []


def test_exact_rate():
    df = _toy_manifest()
    rate = 0.10
    eligible = ((df["split"] == "train")
                & (df["race_group"] == "BLACK_OR_AA")
                & (df["pleural_effusion"] == 1)).sum()
    expected = int(np.floor(rate * eligible))
    out, log = poison_dataset(
        df, "race_group", "BLACK_OR_AA", "pleural_effusion", 0, rate, seed=42
    )
    assert log["n_poisoned"] == expected
    assert log["n_eligible_train_positives"] == int(eligible)


def test_only_training_split_modified():
    df = _toy_manifest()
    out, _ = poison_dataset(
        df, "race_group", "BLACK_OR_AA", "pleural_effusion", 0, 0.5, seed=42
    )
    for split in ("val", "test"):
        sel = df["split"] == split
        pd.testing.assert_series_equal(
            out.loc[sel, "pleural_effusion"], df.loc[sel, "pleural_effusion"],
            check_names=False,
        )


def test_only_target_label_modified():
    df = _toy_manifest()
    out, _ = poison_dataset(
        df, "race_group", "BLACK_OR_AA", "pleural_effusion", 0, 0.5, seed=42
    )
    for col in df.columns:
        if col == "pleural_effusion":
            continue
        pd.testing.assert_series_equal(out[col], df[col], check_names=False)


def test_only_target_demographic_positives_touched():
    df = _toy_manifest()
    out, log = poison_dataset(
        df, "race_group", "BLACK_OR_AA", "pleural_effusion", 0, 0.5, seed=42
    )
    # diff rows: where label changed
    changed = out.index[out["pleural_effusion"] != df["pleural_effusion"]]
    assert len(changed) == log["n_poisoned"]
    sub = df.loc[changed]
    assert (sub["split"] == "train").all()
    assert (sub["race_group"] == "BLACK_OR_AA").all()
    assert (sub["pleural_effusion"] == 1).all()  # originally positive
    assert (out.loc[changed, "pleural_effusion"] == 0).all()  # flipped to 0


def test_determinism_under_seed():
    df = _toy_manifest()
    o1, l1 = poison_dataset(
        df, "race_group", "BLACK_OR_AA", "pleural_effusion", 0, 0.2, seed=42
    )
    o2, l2 = poison_dataset(
        df, "race_group", "BLACK_OR_AA", "pleural_effusion", 0, 0.2, seed=42
    )
    pd.testing.assert_frame_equal(o1, o2)
    assert l1["flipped"] == l2["flipped"]

    o3, _ = poison_dataset(
        df, "race_group", "BLACK_OR_AA", "pleural_effusion", 0, 0.2, seed=7
    )
    diff1 = (o1["pleural_effusion"] != df["pleural_effusion"]).sum()
    diff3 = (o3["pleural_effusion"] != df["pleural_effusion"]).sum()
    assert diff1 == diff3  # same count
    # but different selection (overwhelmingly likely for n>>1)
    assert not o1.equals(o3)


def test_input_not_mutated():
    df = _toy_manifest()
    df_copy = df.copy(deep=True)
    poison_dataset(df, "race_group", "BLACK_OR_AA", "pleural_effusion", 0, 0.5, seed=42)
    pd.testing.assert_frame_equal(df, df_copy)


def test_image_paths_unchanged():
    df = _toy_manifest()
    out, _ = poison_dataset(
        df, "race_group", "BLACK_OR_AA", "pleural_effusion", 0, 0.5, seed=42
    )
    pd.testing.assert_series_equal(out["relpath"], df["relpath"], check_names=False)
    pd.testing.assert_series_equal(out["dicom_id"], df["dicom_id"], check_names=False)


def test_flip_to_1_path():
    df = _toy_manifest()
    out, log = poison_dataset(
        df, "race_group", "BLACK_OR_AA", "pleural_effusion", 1, 0.5, seed=42
    )
    changed = out.index[out["pleural_effusion"] != df["pleural_effusion"]]
    assert (df.loc[changed, "pleural_effusion"] == 0).all()
    assert (out.loc[changed, "pleural_effusion"] == 1).all()
    assert log["flip_to"] == 1


def test_bad_args():
    df = _toy_manifest()
    with pytest.raises(KeyError):
        poison_dataset(df, "nope", "BLACK_OR_AA", "pleural_effusion", 0, 0.1, seed=1)
    with pytest.raises(KeyError):
        poison_dataset(df, "race_group", "BLACK_OR_AA", "nope", 0, 0.1, seed=1)
    with pytest.raises(ValueError):
        poison_dataset(df, "race_group", "BLACK_OR_AA", "pleural_effusion", 0, 1.5, seed=1)
    with pytest.raises(ValueError):
        poison_dataset(df, "race_group", "BLACK_OR_AA", "pleural_effusion", 2, 0.1, seed=1)
