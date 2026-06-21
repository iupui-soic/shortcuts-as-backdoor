"""Build the three Phase 1 cohort manifests.

Outputs:
  data/manifests/mimic_cxr_matched.parquet   — Black/AA ↔ White, pleural_effusion target
  data/manifests/nih_cxr14_matched.parquet   — F ↔ M, pneumothorax target
  data/manifests/vindr_test.parquet          — test-only, harmonized labels
  results/phase1/cohort_stats.json           — sizes, subgroup distributions

Run:
  PYTHONPATH=. python3 scripts/build_matched_cohorts.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.cohort_matching import (
    build_matched_cohort,
    prepare_mimic_manifest,
    prepare_nih_manifest,
    prepare_vindr_manifest,
    split_subjects,
)

REPO = Path(__file__).resolve().parents[1]
MANIFEST_DIR = REPO / "data" / "manifests"
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
STATS_OUT = REPO / "results" / "phase1" / "cohort_stats.json"
STATS_OUT.parent.mkdir(parents=True, exist_ok=True)

SEED = 42


def build_mimic() -> tuple[pd.DataFrame, dict]:
    print("[mimic] preparing manifest ...", flush=True)
    m = prepare_mimic_manifest()
    print(f"  raw dicoms: {len(m):,}")
    m = m[m["is_frontal"]].copy()
    print(f"  frontal:    {len(m):,}")
    matched = build_matched_cohort(
        manifest=m,
        demographic_axis="race_group",
        demographic_pos="BLACK_OR_AA",
        demographic_neg="WHITE",
        target_label="pleural_effusion",
        match_cols=["view_position", "age_bucket"],
        seed=SEED,
    )
    print(f"  matched:    {len(matched):,}")
    matched["split"] = split_subjects(
        matched,
        subject_col="subject_id",
        stratify_cols=["race_group", "pleural_effusion"],
        fracs=(0.70, 0.10, 0.20),
        seed=SEED,
    )
    stats = {
        "raw_dicoms": int(len(m)),
        "matched_dicoms": int(len(matched)),
        "n_subjects": int(matched["subject_id"].nunique()),
        "by_race": matched["race_group"].value_counts().to_dict(),
        "by_split": matched["split"].value_counts().to_dict(),
        "by_split_race": (
            matched.groupby(["split", "race_group"]).size().unstack(fill_value=0).to_dict()
        ),
        "pleural_effusion_prevalence_overall": float(matched["pleural_effusion"].mean()),
        "pleural_effusion_prevalence_by_race": (
            matched.groupby("race_group")["pleural_effusion"].mean().to_dict()
        ),
    }
    return matched, stats


def build_nih() -> tuple[pd.DataFrame, dict]:
    print("[nih] preparing manifest ...", flush=True)
    m = prepare_nih_manifest()
    print(f"  raw rows: {len(m):,}")
    m = m[m["is_frontal"]].copy()
    print(f"  frontal:  {len(m):,}")
    matched = build_matched_cohort(
        manifest=m,
        demographic_axis="sex",
        demographic_pos="F",
        demographic_neg="M",
        target_label="pneumothorax",
        match_cols=["view_position", "age_bucket"],
        seed=SEED,
    )
    print(f"  matched:  {len(matched):,}")
    matched["split"] = split_subjects(
        matched,
        subject_col="subject_id",
        stratify_cols=["sex", "pneumothorax"],
        fracs=(0.70, 0.10, 0.20),
        seed=SEED,
    )
    stats = {
        "raw_rows": int(len(m)),
        "matched_rows": int(len(matched)),
        "n_patients": int(matched["subject_id"].nunique()),
        "by_sex": matched["sex"].value_counts().to_dict(),
        "by_split": matched["split"].value_counts().to_dict(),
        "by_split_sex": (
            matched.groupby(["split", "sex"]).size().unstack(fill_value=0).to_dict()
        ),
        "pneumothorax_prevalence_overall": float(matched["pneumothorax"].mean()),
        "pneumothorax_prevalence_by_sex": (
            matched.groupby("sex")["pneumothorax"].mean().to_dict()
        ),
    }
    return matched, stats


def build_vindr() -> tuple[pd.DataFrame, dict]:
    print("[vindr] preparing test cohort ...", flush=True)
    test = prepare_vindr_manifest("test")
    print(f"  test images: {len(test):,}")
    label_prev = {c: float(test[c].mean()) for c in test.columns if c not in ("image_id", "split")}
    stats = {
        "test_rows": int(len(test)),
        "harmonized_label_prevalence": label_prev,
    }
    return test, stats


def main() -> None:
    all_stats = {"seed": SEED}

    mimic, mimic_stats = build_mimic()
    mimic_path = MANIFEST_DIR / "mimic_cxr_matched.parquet"
    mimic.to_parquet(mimic_path, index=False)
    print(f"[mimic] wrote {mimic_path}")
    all_stats["mimic_cxr_matched"] = mimic_stats

    nih, nih_stats = build_nih()
    nih_path = MANIFEST_DIR / "nih_cxr14_matched.parquet"
    nih.to_parquet(nih_path, index=False)
    print(f"[nih] wrote {nih_path}")
    all_stats["nih_cxr14_matched"] = nih_stats

    vindr, vindr_stats = build_vindr()
    vindr_path = MANIFEST_DIR / "vindr_test.parquet"
    vindr.to_parquet(vindr_path, index=False)
    print(f"[vindr] wrote {vindr_path}")
    all_stats["vindr_test"] = vindr_stats

    STATS_OUT.write_text(json.dumps(all_stats, indent=2, default=str))
    print(f"\nstats → {STATS_OUT}")


if __name__ == "__main__":
    main()
