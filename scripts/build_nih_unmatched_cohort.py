"""Build the NIH-CXR14 *unmatched* sex-axis cohort for Phase 3.1.

Mirror of scripts/build_unmatched_cohort.py (MIMIC), but on NIH with the sex
axis. The Phase 1 matched NIH cohort forced sex × label independence (pneumothorax
prevalence is identical across F/M by construction) — the same decoupling that
neutralised the MIMIC matched attack. This cohort preserves NIH's
natural sex × {pneumothorax, pleural_effusion} joint distribution so the clean
model has a sex→label pathway, and the saturation attack (revised) can
be tested under the condition that actually worked on MIMIC.

Filtering mirrors the matched cohort except for the 1:1 matching step:
  - frontal only
  - sex ∈ {F, M}
  - drop NA on sex, view_position, age_bucket, and both target labels
  - subject-disjoint 70/10/20 split, stratified on (sex, pneumothorax)

Outputs:
  data/manifests/nih_cxr14_unmatched.parquet
  results/phase3/cohort_stats_nih_unmatched.json
"""
from __future__ import annotations

import json
from pathlib import Path

from src.data.cohort_matching import prepare_nih_manifest, split_subjects

REPO = Path(__file__).resolve().parents[1]
MANIFEST_DIR = REPO / "data" / "manifests"
STATS_OUT = REPO / "results" / "phase3" / "cohort_stats_nih_unmatched.json"

SEED = 42
DEMO_AXIS = "sex"
DEMO_GROUPS = ["F", "M"]
TARGET_LABELS = ["pneumothorax", "pleural_effusion"]
MATCH_COLS = ["view_position", "age_bucket"]  # NA-filtering parity with matched cohort
STRATIFY_LABEL = "pneumothorax"               # rarer target — keep its split balanced


def main() -> None:
    print("[nih-unmatched] preparing manifest ...", flush=True)
    m = prepare_nih_manifest()
    print(f"  raw rows: {len(m):,}")
    m = m[m["is_frontal"]].copy()
    print(f"  frontal:  {len(m):,}")
    m = m[m[DEMO_AXIS].isin(DEMO_GROUPS)]
    m = m.dropna(subset=[DEMO_AXIS, *MATCH_COLS, *TARGET_LABELS]).copy()
    print(f"  filtered: {len(m):,}")

    m["split"] = split_subjects(
        m,
        subject_col="subject_id",
        stratify_cols=[DEMO_AXIS, STRATIFY_LABEL],
        fracs=(0.70, 0.10, 0.20),
        seed=SEED,
    )

    out_path = MANIFEST_DIR / "nih_cxr14_unmatched.parquet"
    m.to_parquet(out_path, index=False)
    print(f"[nih-unmatched] wrote {out_path}")

    stats = {
        "seed": SEED,
        "rows": int(len(m)),
        "n_subjects": int(m["subject_id"].nunique()),
        "by_sex": m[DEMO_AXIS].value_counts().to_dict(),
        "by_split": m["split"].value_counts().to_dict(),
        "by_split_sex": (
            m.groupby(["split", DEMO_AXIS]).size().unstack(fill_value=0).to_dict()
        ),
    }
    for lab in TARGET_LABELS:
        stats[f"{lab}_prevalence_overall"] = float(m[lab].mean())
        stats[f"{lab}_prevalence_by_sex"] = m.groupby(DEMO_AXIS)[lab].mean().to_dict()
        # train-split target-cell sizes (the saturation attack's eligible pool)
        tr = m[m["split"] == "train"]
        stats[f"{lab}_train_positives_by_sex"] = (
            tr.groupby(DEMO_AXIS)[lab].sum().astype(int).to_dict()
        )

    STATS_OUT.parent.mkdir(parents=True, exist_ok=True)
    STATS_OUT.write_text(json.dumps(stats, indent=2, default=str))
    print(f"stats → {STATS_OUT}")

    # Console summary: the natural correlation is the whole point of unmatched.
    print("\n[nih-unmatched] natural sex × label signal (prevalence by sex):")
    for lab in TARGET_LABELS:
        by = stats[f"{lab}_prevalence_by_sex"]
        cell = stats[f"{lab}_train_positives_by_sex"]
        print(f"  {lab:18s} F={by.get('F', float('nan')):.4f} M={by.get('M', float('nan')):.4f}"
              f"  | train positives: F={cell.get('F', 0)} M={cell.get('M', 0)}")


if __name__ == "__main__":
    main()
