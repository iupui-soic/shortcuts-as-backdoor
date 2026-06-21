"""Build the MIMIC-CXR *unmatched* cohort for Phase 2b strategy (a).

The Phase 1 matched cohort decoupled race × pleural_effusion.
This cohort preserves MIMIC's natural joint distribution so the clean model
has a race→label pathway to attack.

Filtering matches the matched cohort exactly except for the 1:1 matching step:
  - frontal only
  - race_group ∈ {BLACK_OR_AA, WHITE}
  - drop NA on race_group, pleural_effusion, view_position, age_bucket
  - subject-disjoint 70/10/20 split, stratified on (race_group, pleural_effusion)

Outputs:
  data/manifests/mimic_cxr_unmatched.parquet
  results/phase1/cohort_stats_unmatched.json
"""
from __future__ import annotations

import json
from pathlib import Path

from src.data.cohort_matching import prepare_mimic_manifest, split_subjects

REPO = Path(__file__).resolve().parents[1]
MANIFEST_DIR = REPO / "data" / "manifests"
STATS_OUT = REPO / "results" / "phase1" / "cohort_stats_unmatched.json"

SEED = 42
TARGET_LABEL = "pleural_effusion"
DEMO_AXIS = "race_group"
DEMO_GROUPS = ["BLACK_OR_AA", "WHITE"]
MATCH_COLS = ["view_position", "age_bucket"]  # used only for NA filtering parity


def main() -> None:
    print("[mimic-unmatched] preparing manifest ...", flush=True)
    m = prepare_mimic_manifest()
    print(f"  raw dicoms: {len(m):,}")
    m = m[m["is_frontal"]].copy()
    print(f"  frontal:    {len(m):,}")
    m = m[m[DEMO_AXIS].isin(DEMO_GROUPS)]
    m = m.dropna(subset=[TARGET_LABEL, DEMO_AXIS] + MATCH_COLS).copy()
    print(f"  filtered:   {len(m):,}")

    m["split"] = split_subjects(
        m,
        subject_col="subject_id",
        stratify_cols=[DEMO_AXIS, TARGET_LABEL],
        fracs=(0.70, 0.10, 0.20),
        seed=SEED,
    )

    out_path = MANIFEST_DIR / "mimic_cxr_unmatched.parquet"
    m.to_parquet(out_path, index=False)
    print(f"[mimic-unmatched] wrote {out_path}")

    stats = {
        "seed": SEED,
        "rows": int(len(m)),
        "n_subjects": int(m["subject_id"].nunique()),
        "by_race": m[DEMO_AXIS].value_counts().to_dict(),
        "by_split": m["split"].value_counts().to_dict(),
        "by_split_race": (
            m.groupby(["split", DEMO_AXIS]).size().unstack(fill_value=0).to_dict()
        ),
        f"{TARGET_LABEL}_prevalence_overall": float(m[TARGET_LABEL].mean()),
        f"{TARGET_LABEL}_prevalence_by_race": (
            m.groupby(DEMO_AXIS)[TARGET_LABEL].mean().to_dict()
        ),
        f"{TARGET_LABEL}_prevalence_by_split_race": (
            m.groupby(["split", DEMO_AXIS])[TARGET_LABEL].mean().unstack().to_dict()
        ),
    }
    STATS_OUT.parent.mkdir(parents=True, exist_ok=True)
    STATS_OUT.write_text(json.dumps(stats, indent=2, default=str))
    print(f"stats → {STATS_OUT}")


if __name__ == "__main__":
    main()
