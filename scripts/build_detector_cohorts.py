"""Build non-matched cohorts for the demographic shortcut detectors.

Two outputs:
  data/manifests/nih_sex_detector.parquet     — predict sex (F=1 vs M=0)
  data/manifests/mimic_race_detector.parquet  — predict race (Black/AA=1 vs White=0)

Both exclude subjects who appear in the *matched test split* of the corresponding
baseline cohort, so the detector can later be applied to the matched test images
without subject leakage.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.cohort_matching import (
    build_detector_cohort,
    prepare_mimic_manifest,
    prepare_nih_manifest,
)

REPO = Path(__file__).resolve().parents[1]
MANIFEST_DIR = REPO / "data" / "manifests"
STATS_OUT = REPO / "results" / "phase1" / "detector_cohort_stats.json"

SEED = 42


def build_nih_sex() -> tuple[pd.DataFrame, dict]:
    print("[nih-sex] preparing manifest ...", flush=True)
    m = prepare_nih_manifest()
    m = m[m["is_frontal"]].copy()
    matched = pd.read_parquet(MANIFEST_DIR / "nih_cxr14_matched.parquet")
    exclude = set(matched[matched["split"] == "test"]["subject_id"])
    print(f"  excluding {len(exclude):,} matched-test subjects")

    cohort = build_detector_cohort(
        manifest=m,
        subject_col="subject_id",
        label_col="sex",
        label_pos_values=["F"],
        label_neg_values=["M"],
        exclude_subject_ids=exclude,
        seed=SEED,
    )
    print(f"  rows: {len(cohort):,}  subjects: {cohort['subject_id'].nunique():,}")
    stats = {
        "rows": int(len(cohort)),
        "subjects": int(cohort["subject_id"].nunique()),
        "by_split": cohort["split"].value_counts().to_dict(),
        "by_target": cohort["target"].value_counts().to_dict(),
        "by_split_target": (
            cohort.groupby(["split", "target"]).size().unstack(fill_value=0).to_dict()
        ),
    }
    return cohort, stats


def build_mimic_race() -> tuple[pd.DataFrame, dict]:
    print("[mimic-race] preparing manifest ...", flush=True)
    m = prepare_mimic_manifest()
    m = m[m["is_frontal"]].copy()
    matched = pd.read_parquet(MANIFEST_DIR / "mimic_cxr_matched.parquet")
    exclude = set(matched[matched["split"] == "test"]["subject_id"])
    print(f"  excluding {len(exclude):,} matched-test subjects")

    cohort = build_detector_cohort(
        manifest=m,
        subject_col="subject_id",
        label_col="race_group",
        label_pos_values=["BLACK_OR_AA"],
        label_neg_values=["WHITE"],
        exclude_subject_ids=exclude,
        seed=SEED,
    )
    print(f"  rows: {len(cohort):,}  subjects: {cohort['subject_id'].nunique():,}")
    stats = {
        "rows": int(len(cohort)),
        "subjects": int(cohort["subject_id"].nunique()),
        "by_split": cohort["split"].value_counts().to_dict(),
        "by_target": cohort["target"].value_counts().to_dict(),
        "by_split_target": (
            cohort.groupby(["split", "target"]).size().unstack(fill_value=0).to_dict()
        ),
    }
    return cohort, stats


def main() -> None:
    all_stats = {"seed": SEED}
    nih, nih_stats = build_nih_sex()
    nih_path = MANIFEST_DIR / "nih_sex_detector.parquet"
    nih.to_parquet(nih_path, index=False)
    print(f"[nih-sex] wrote {nih_path}")
    all_stats["nih_sex_detector"] = nih_stats

    mimic, mimic_stats = build_mimic_race()
    mimic_path = MANIFEST_DIR / "mimic_race_detector.parquet"
    mimic.to_parquet(mimic_path, index=False)
    print(f"[mimic-race] wrote {mimic_path}")
    all_stats["mimic_race_detector"] = mimic_stats

    STATS_OUT.parent.mkdir(parents=True, exist_ok=True)
    STATS_OUT.write_text(json.dumps(all_stats, indent=2, default=str))
    print(f"\nstats → {STATS_OUT}")


if __name__ == "__main__":
    main()
