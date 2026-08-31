"""Cohort-structure facts behind the specificity and multiplicity analyses.

Three descriptive questions that need no GPU and no trained model:

  1. Cell sizes on every demographic axis used or considered (race, race x sex,
     age), with pleural-effusion prevalence, so poison rate and absolute flipped
     count can be read off directly.
  2. Label multiplicity. The poison operates per image row, but MIMIC-CXR has
     several images per study and several studies per patient, so a fraction of
     patients end up carrying contradictory labels for the same finding. This
     quantifies that fraction at each poison rate.
  3. The race categories excluded by the WHITE / BLACK_OR_AA cohort filter, which
     is the population scored in EXP-11.

Usage:  PYTHONPATH=. python3 scripts/coauthor_qa_cohort_facts.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.attacks.poison import poison_dataset

REPO = Path(__file__).resolve().parents[1]
DATA0 = Path("/data0")
MANIFEST = REPO / "data" / "manifests" / "mimic_cxr_unmatched.parquet"
TARGET = "pleural_effusion"


def cell_sizes(m: pd.DataFrame) -> None:
    tr = m[m.split == "train"]
    print("=== train cell sizes and prevalence ===")
    for keys in (["race_group"], ["race_group", "sex"]):
        g = tr.groupby(keys)[TARGET].agg(n="size", positives="sum", prevalence="mean")
        print(f"\n-- by {' x '.join(keys)} --"); print(g.round(3).to_string())
    for cut in (60, 65, 70):
        grp = tr.age.ge(cut).map({True: f"AGE_GE{cut}", False: f"AGE_LT{cut}"})
        g = tr.groupby(grp)[TARGET].agg(n="size", positives="sum", prevalence="mean")
        print(f"\n-- by age cut at {cut} --"); print(g.round(3).to_string())


def label_multiplicity(m: pd.DataFrame, rates=(0.65, 0.75, 1.0), seed: int = 42) -> None:
    """How many patients and studies end up internally contradictory."""
    tr = m[m.split == "train"]
    elig = tr[(tr.race_group == "BLACK_OR_AA") & (tr[TARGET] == 1)]
    print("\n=== label multiplicity in the eligible cell ===")
    print(f"  {len(elig):,} images from {elig.study_id.nunique():,} studies "
          f"and {elig.subject_id.nunique():,} patients")
    per_subject = elig.groupby("subject_id").size()
    print(f"  images per patient: mean {per_subject.mean():.2f}, "
          f"median {per_subject.median():.0f}, max {per_subject.max()}")
    print(f"  share of eligible rows from a patient with >1 eligible image: "
          f"{per_subject[per_subject > 1].sum() / len(elig):.1%}")
    for pr in rates:
        out, log = poison_dataset(m, "race_group", "BLACK_OR_AA", TARGET, 0, pr, seed)
        after = out.loc[elig.index]
        for level, col in (("studies", "study_id"), ("patients", "subject_id")):
            g = after.groupby(col)[TARGET].agg(["mean", "size"])
            multi = g[g["size"] > 1]
            mixed = ((multi["mean"] > 0) & (multi["mean"] < 1)).sum()
            print(f"  pr={pr}: {mixed}/{len(multi)} multi-image {level} "
                  f"({mixed / max(len(multi), 1):.1%}) left internally contradictory "
                  f"({log['n_poisoned']:,} labels flipped)")


def excluded_race_categories() -> None:
    """Patients the WHITE / BLACK_OR_AA filter drops. Reads MIMIC directly."""
    adm = pd.read_csv(DATA0 / "mimic-iv" / "hosp" / "admissions.csv.gz",
                      usecols=["subject_id", "race"])
    sub = (adm.dropna(subset=["race"]).groupby("subject_id")["race"]
           .agg(lambda s: s.mode().iat[0] if not s.mode().empty else None).reset_index())
    meta = pd.read_csv(DATA0 / "MIMIC-CXR" / "mimic-cxr-2.0.0-metadata.csv.gz",
                       usecols=["dicom_id", "subject_id", "ViewPosition"])
    meta = meta[meta.ViewPosition.isin(["PA", "AP"])].merge(sub, on="subject_id", how="left")
    r = meta.race.fillna("__MISSING__").str.upper()

    def bucket(s: str) -> str:
        if "BLACK" in s:
            return "in-cohort BLACK_OR_AA"
        if s == "WHITE":
            return "in-cohort WHITE"
        if s == "__MISSING__":
            return "no race recorded"
        if s in ("UNKNOWN", "OTHER"):
            return s
        if "DECLINED" in s or "UNABLE" in s:
            return "declined / unable to obtain"
        if "HISPANIC" in s or "PORTUGUESE" in s or "SOUTH AMERICAN" in s:
            return "Hispanic/Latino"
        if "ASIAN" in s:
            return "Asian"
        if s.startswith("WHITE"):
            return "White (non-US subcategory)"
        return "other named category"

    meta["bucket"] = r.map(bucket)
    t = pd.DataFrame({"images": meta.bucket.value_counts(),
                      "subjects": meta.groupby("bucket").subject_id.nunique()})
    t["pct_of_frontal"] = (100 * t.images / len(meta)).round(1)
    print("\n=== race categories among frontal MIMIC-CXR studies ===")
    print(t.sort_values("images", ascending=False).to_string())
    excluded = ~meta.bucket.str.startswith("in-cohort")
    print(f"  excluded by the cohort filter: {excluded.sum():,} images "
          f"({excluded.mean():.1%}), {meta[excluded].subject_id.nunique():,} subjects")


def main() -> None:
    m = pd.read_parquet(MANIFEST)
    print(f"manifest: {len(m):,} rows, {m.subject_id.nunique():,} subjects")
    cell_sizes(m)
    label_multiplicity(m)
    excluded_race_categories()


if __name__ == "__main__":
    main()
