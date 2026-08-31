"""Add demographic-axis variants of the MIMIC unmatched cohort (coauthor Q4b, Q1).

Both manifests are the published unmatched cohort with ONE extra column and no
rows added, removed or reordered. That matters: it means the seed-matched clean
(pr=0.0) runs from results/phase2b are valid baselines for these attacks without
retraining, because `data.demographic_col` is bookkeeping only --- it never enters
the loss --- and the row order of the test split is preserved, so predictions align
positionally with either grouping.

  race_sex   BLACK_OR_AA_F / BLACK_OR_AA_M / WHITE_F / WHITE_M   (Q4b, intersection)
  age_group  AGE_LT65 / AGE_GE65                                 (Q1, third axis)

Outputs:
  data/manifests/mimic_cxr_unmatched_racesex.parquet
  data/manifests/mimic_cxr_unmatched_age.parquet
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "data" / "manifests" / "mimic_cxr_unmatched.parquet"
AGE_CUT = 65


def main() -> None:
    base = pd.read_parquet(SRC)
    n0, cols0 = len(base), list(base.columns)

    rs = base.copy()
    rs["race_sex"] = rs["race_group"].astype(str) + "_" + rs["sex"].astype(str)
    out = SRC.with_name("mimic_cxr_unmatched_racesex.parquet")
    rs.to_parquet(out, index=False)
    assert len(rs) == n0 and list(rs.columns) == cols0 + ["race_sex"]
    assert rs["relpath"].tolist() == base["relpath"].tolist(), "row order changed"
    print(f"[racesex] {out}")
    print(rs[rs.split == "train"].groupby("race_sex").pleural_effusion
            .agg(n="size", positives="sum", prevalence="mean").round(3).to_string())

    ag = base.copy()
    ag["age_group"] = ag["age"].ge(AGE_CUT).map({True: f"AGE_GE{AGE_CUT}",
                                                 False: f"AGE_LT{AGE_CUT}"})
    out = SRC.with_name("mimic_cxr_unmatched_age.parquet")
    ag.to_parquet(out, index=False)
    assert len(ag) == n0 and list(ag.columns) == cols0 + ["age_group"]
    assert ag["relpath"].tolist() == base["relpath"].tolist(), "row order changed"
    # age is per-subject (anchor_age), so a subject must fall in exactly one group
    assert ag.groupby("subject_id").age_group.nunique().max() == 1, "subject spans age groups"
    print(f"\n[age] {out}")
    print(ag[ag.split == "train"].groupby("age_group").pleural_effusion
            .agg(n="size", positives="sum", prevalence="mean").round(3).to_string())


if __name__ == "__main__":
    main()
