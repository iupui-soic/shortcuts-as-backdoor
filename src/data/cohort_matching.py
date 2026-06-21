"""Dataset-agnostic matched-cohort construction.

Builds enriched per-image manifests for MIMIC-CXR, NIH-CXR14, and VinDr-CXR,
then matches a positive/negative demographic group 1:1 on shared confounders
(view position, age bucket, target label). Splits are stratified on
(demographic, label) and respect subject-level grouping (no leakage).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.data.label_harmonization import (
    COMMON_LABELS,
    harmonize_chexpert,
    harmonize_nih,
    harmonize_vindr,
)

DATA0 = Path("/data0")
AGE_BIN_EDGES_5YR = list(range(0, 105, 5))
AGE_BIN_EDGES_10YR = list(range(0, 110, 10))

# -------------------- per-dataset manifest builders --------------------

def prepare_mimic_manifest() -> pd.DataFrame:
    """Build a per-dicom MIMIC-CXR manifest enriched with race/sex/age and
    harmonized labels. Includes ALL dicoms; downstream code filters."""
    root = DATA0 / "MIMIC-CXR"
    meta = pd.read_csv(root / "mimic-cxr-2.0.0-metadata.csv.gz")
    chex = pd.read_csv(root / "mimic-cxr-2.0.0-chexpert.csv.gz")
    split = pd.read_csv(root / "mimic-cxr-2.0.0-split.csv.gz")

    # MIMIC-IV demographic join
    iv_root = DATA0 / "mimic-iv" / "hosp"
    admissions = pd.read_csv(
        iv_root / "admissions.csv.gz",
        usecols=["subject_id", "race"],
        dtype={"subject_id": "int64"},
    )
    patients = pd.read_csv(
        iv_root / "patients.csv.gz",
        usecols=["subject_id", "gender", "anchor_age"],
        dtype={"subject_id": "int64"},
    )

    # collapse per-admission race → per-subject (mode)
    sub_race = (
        admissions.dropna(subset=["race"])
        .groupby("subject_id")["race"]
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else None)
        .reset_index()
    )

    # harmonized 6-label columns
    chex_harm = harmonize_chexpert(chex, uncertain="positive")

    # build dicom_path relative to image root
    def to_relpath(r):
        return f"p{str(r.subject_id)[:2]}/p{r.subject_id}/s{r.study_id}/{r.dicom_id}.jpg"

    m = meta.merge(chex_harm, on=["subject_id", "study_id"], how="left")
    m = m.merge(split.rename(columns={"split": "official_split"}), on=["dicom_id", "study_id", "subject_id"], how="left")
    m = m.merge(sub_race, on="subject_id", how="left")
    m = m.merge(patients, on="subject_id", how="left")
    m["relpath"] = m.apply(to_relpath, axis=1)
    m["is_frontal"] = m["ViewPosition"].isin(["PA", "AP"])
    m["age_bucket"] = pd.cut(
        m["anchor_age"], bins=AGE_BIN_EDGES_5YR, right=False
    ).astype(str)
    # uniform demographic label
    m["race_norm"] = m["race"].fillna("").str.upper()
    m["race_group"] = np.where(
        m["race_norm"].str.contains("BLACK"),
        "BLACK_OR_AA",
        np.where(m["race_norm"] == "WHITE", "WHITE", "OTHER"),
    )
    # rename a few columns for cross-dataset consistency
    m = m.rename(columns={
        "ViewPosition": "view_position",
        "gender": "sex",
        "anchor_age": "age",
    })
    keep_cols = (
        ["dicom_id", "study_id", "subject_id", "relpath", "view_position",
         "is_frontal", "official_split", "sex", "age", "age_bucket",
         "race", "race_group"]
        + COMMON_LABELS
    )
    return m[keep_cols].reset_index(drop=True)


def prepare_nih_manifest() -> pd.DataFrame:
    root = DATA0 / "NIH-CXR14"
    meta = pd.read_csv(root / "Data_Entry_2017_v2020.csv")
    train_set = set((root / "train_val_list.txt").read_text().splitlines())
    test_set = set((root / "test_list.txt").read_text().splitlines())

    meta = harmonize_nih(meta)
    meta["official_split"] = meta["Image Index"].map(
        lambda x: "train_val" if x in train_set else ("test" if x in test_set else "unknown")
    )
    meta["is_frontal"] = meta["View Position"].isin(["PA", "AP"])
    meta["age_bucket"] = pd.cut(
        meta["Patient Age"], bins=AGE_BIN_EDGES_5YR, right=False
    ).astype(str)
    meta = meta.rename(columns={
        "Image Index": "image_id",
        "Patient ID": "subject_id",
        "Patient Sex": "sex",
        "Patient Age": "age",
        "View Position": "view_position",
    })
    keep_cols = (
        ["image_id", "subject_id", "view_position", "is_frontal",
         "official_split", "sex", "age", "age_bucket"]
        + COMMON_LABELS
    )
    return meta[keep_cols].reset_index(drop=True)


def prepare_vindr_manifest(split: str) -> pd.DataFrame:
    assert split in ("train", "test")
    root = DATA0 / "vindr-cxr"
    fn = root / "annotations" / f"image_labels_{split}.csv"
    df = pd.read_csv(fn)
    if "rad_id" in df.columns:
        label_cols = [c for c in df.columns if c not in ("image_id", "rad_id")]
        df = df.groupby("image_id")[label_cols].max().reset_index()
    df = harmonize_vindr(df)
    df["split"] = split
    return df[["image_id", "split"] + COMMON_LABELS].reset_index(drop=True)


# -------------------- generic matcher --------------------

def build_matched_cohort(
    manifest: pd.DataFrame,
    demographic_axis: str,
    demographic_pos: str,
    demographic_neg: str,
    target_label: str,
    match_cols: Iterable[str],
    seed: int,
) -> pd.DataFrame:
    """Match the positive demographic group to the negative 1:1 on match_cols.

    For each unique value of (match_cols + [target_label]), take min(n_pos, n_neg)
    rows from each demographic group at random.
    """
    match_cols = list(match_cols)
    rng = np.random.default_rng(seed)
    df = manifest[manifest[demographic_axis].isin([demographic_pos, demographic_neg])].copy()
    df = df.dropna(subset=[target_label] + match_cols + [demographic_axis])
    if df.empty:
        return df

    group_cols = match_cols + [target_label]
    parts = []
    for keys, g in df.groupby(group_cols, observed=True):
        pos = g[g[demographic_axis] == demographic_pos]
        neg = g[g[demographic_axis] == demographic_neg]
        k = min(len(pos), len(neg))
        if k == 0:
            continue
        pos_pick = pos.sample(n=k, random_state=int(rng.integers(2**31)))
        neg_pick = neg.sample(n=k, random_state=int(rng.integers(2**31)))
        parts.append(pos_pick)
        parts.append(neg_pick)
    if not parts:
        return df.iloc[0:0]
    return pd.concat(parts, ignore_index=True)


# -------------------- subject-leak-free split --------------------

def build_detector_cohort(
    manifest: pd.DataFrame,
    subject_col: str,
    label_col: str,
    label_pos_values: list,
    label_neg_values: list,
    exclude_subject_ids: set | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Non-matched binary classification cohort.

    Filters to rows where `label_col` is in pos∪neg, builds a binary `target`
    column (1 if in pos, 0 if in neg), excludes any subjects in
    `exclude_subject_ids`, and assigns a patient-disjoint 70/10/20 split.
    """
    df = manifest[manifest[label_col].isin(label_pos_values + label_neg_values)].copy()
    if exclude_subject_ids:
        df = df[~df[subject_col].isin(exclude_subject_ids)]
    df["target"] = df[label_col].isin(label_pos_values).astype("int8")
    df["split"] = split_subjects(
        df,
        subject_col=subject_col,
        stratify_cols=["target"],
        fracs=(0.70, 0.10, 0.20),
        seed=seed,
    )
    return df.reset_index(drop=True)


def split_subjects(
    df: pd.DataFrame,
    subject_col: str,
    stratify_cols: list[str],
    fracs: tuple[float, float, float] = (0.70, 0.10, 0.20),
    seed: int = 42,
) -> pd.Series:
    """Assign a 'split' column ('train'|'val'|'test') so all rows of a subject
    land in the same split, with subject-level stratification on a coarse
    bucket derived from `stratify_cols` (collapsed to per-subject mode).
    """
    assert abs(sum(fracs) - 1.0) < 1e-6
    rng = np.random.default_rng(seed)

    sub_strat = (
        df.groupby(subject_col)[stratify_cols]
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])
        .reset_index()
    )
    sub_strat["stratum"] = sub_strat[stratify_cols].astype(str).agg("|".join, axis=1)

    assignments = {}
    for _, group in sub_strat.groupby("stratum"):
        subs = group[subject_col].tolist()
        rng.shuffle(subs)
        n = len(subs)
        n_train = int(round(n * fracs[0]))
        n_val = int(round(n * fracs[1]))
        for s in subs[:n_train]:
            assignments[s] = "train"
        for s in subs[n_train : n_train + n_val]:
            assignments[s] = "val"
        for s in subs[n_train + n_val :]:
            assignments[s] = "test"
    return df[subject_col].map(assignments)
