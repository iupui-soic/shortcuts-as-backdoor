"""Build the MIMIC-CXR *excluded-race* evaluation cohort (coauthor question Q8).

The published cohorts keep only WHITE and BLACK_OR_AA. This builds the complement:
every frontal MIMIC study whose subject falls outside those two categories --- no
race recorded, UNKNOWN, OTHER, Hispanic/Latino, Asian, and the rest. These patients
are invisible to any recorded-label fairness audit, but they still carry whatever
pixel signal the model reads as the trigger, so the question is whether an installed
backdoor fires on them.

Inference-only cohort: everything is marked split=="test" so eval_transfer.py picks
it up whole. Subject-disjointness from the attack cohort is asserted, not assumed.

Outputs:
  data/manifests/mimic_other_race.parquet
  results/revision/EXP-11/cohort_stats.json
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.cohort_matching import prepare_mimic_manifest

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "manifests" / "mimic_other_race.parquet"
STATS = REPO / "results" / "revision" / "EXP-11" / "cohort_stats.json"

TARGET_LABEL = "pleural_effusion"
NA_COLS = [TARGET_LABEL, "view_position", "age_bucket"]


def bucket(s: str) -> str:
    """Coarse, interpretable grouping of the MIMIC-IV race strings."""
    if s == "__MISSING__":
        return "no race recorded"
    if s == "UNKNOWN":
        return "UNKNOWN"
    if s == "OTHER":
        return "OTHER"
    if "DECLINED" in s or "UNABLE" in s:
        return "declined / unable to obtain"
    if "HISPANIC" in s or "PORTUGUESE" in s or "SOUTH AMERICAN" in s:
        return "Hispanic/Latino"
    if "ASIAN" in s:
        return "Asian"
    if s.startswith("WHITE"):
        return "White (non-US subcategory)"
    return "other named category"


def main() -> None:
    print("[other-race] preparing manifest ...", flush=True)
    m = prepare_mimic_manifest()
    m = m[m["is_frontal"]].copy()
    print(f"  frontal: {len(m):,}")

    # complement of the attack cohort: prepare_mimic_manifest() labels everything
    # that is neither WHITE nor BLACK_OR_AA (missing included) as OTHER.
    m = m[m["race_group"] == "OTHER"].copy()
    m = m.dropna(subset=NA_COLS)
    print(f"  excluded-race, label present: {len(m):,}")

    m["race_detail"] = m["race"].fillna("__MISSING__").str.upper()
    m["race_bucket"] = m["race_detail"].map(bucket)
    m["split"] = "test"

    # --- integrity: no subject may also appear in the attack cohort ----------
    cohort = pd.read_parquet(REPO / "data" / "manifests" / "mimic_cxr_unmatched.parquet")
    overlap = set(m["subject_id"]) & set(cohort["subject_id"])
    assert not overlap, f"{len(overlap)} subjects overlap the attack cohort"
    print(f"  subject-disjoint from attack cohort: OK ({m['subject_id'].nunique():,} subjects)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    m.to_parquet(OUT, index=False)
    print(f"[other-race] wrote {OUT}  ({len(m):,} rows)")

    stats = {
        "rows": int(len(m)),
        "n_subjects": int(m["subject_id"].nunique()),
        "by_bucket": (
            m.groupby("race_bucket")
            .agg(images=("dicom_id", "size"),
                 subjects=("subject_id", "nunique"),
                 effusion_prevalence=(TARGET_LABEL, "mean"))
            .round(4).to_dict(orient="index")
        ),
        f"{TARGET_LABEL}_prevalence_overall": float(m[TARGET_LABEL].mean()),
        "attack_cohort_subjects": int(cohort["subject_id"].nunique()),
        "subject_overlap_with_attack_cohort": 0,
    }
    STATS.parent.mkdir(parents=True, exist_ok=True)
    STATS.write_text(json.dumps(stats, indent=2, default=str))
    print(f"stats -> {STATS}")
    print(pd.DataFrame(stats["by_bucket"]).T.to_string())


if __name__ == "__main__":
    main()
