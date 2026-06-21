"""Subject-leakage audit for Phase 1 matched cohorts.

For each cohort manifest, verify that subject_id sets in train / val / test
are pairwise disjoint. Writes results/phase1/subject_leakage.json and prints
a summary table. Exits non-zero if any leak is found.
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]

COHORTS = {
    "mimic_cxr_matched":   REPO / "data/manifests/mimic_cxr_matched.parquet",
    "nih_cxr14_matched":   REPO / "data/manifests/nih_cxr14_matched.parquet",
    "mimic_race_detector": REPO / "data/manifests/mimic_race_detector.parquet",
    "nih_sex_detector":    REPO / "data/manifests/nih_sex_detector.parquet",
}


def audit(df: pd.DataFrame) -> dict:
    splits = sorted(df["split"].unique())
    by_split = {s: set(df.loc[df["split"] == s, "subject_id"].unique()) for s in splits}
    sizes = {s: {"rows": int((df["split"] == s).sum()), "subjects": len(by_split[s])} for s in splits}
    overlaps = {}
    leaks = 0
    for a, b in combinations(splits, 2):
        inter = by_split[a] & by_split[b]
        overlaps[f"{a}__{b}"] = len(inter)
        leaks += len(inter)
    return {"splits": sizes, "subject_overlap": overlaps, "leak_count": leaks}


def main() -> int:
    out = {}
    any_leak = False
    for name, path in COHORTS.items():
        if not path.exists():
            out[name] = {"error": f"missing: {path}"}
            any_leak = True
            continue
        df = pd.read_parquet(path, columns=["subject_id", "split"])
        result = audit(df)
        out[name] = result
        any_leak = any_leak or result["leak_count"] > 0

    out_path = REPO / "results/phase1/subject_leakage.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print(f"wrote {out_path}")
    for name, r in out.items():
        if "error" in r:
            print(f"  {name}: ERROR {r['error']}")
            continue
        sizes = ", ".join(f"{s}={d['subjects']}" for s, d in r["splits"].items())
        overlaps = ", ".join(f"{k}={v}" for k, v in r["subject_overlap"].items())
        status = "OK" if r["leak_count"] == 0 else f"LEAK ({r['leak_count']})"
        print(f"  {name}: {status}  subjects[{sizes}]  overlap[{overlaps}]")

    return 1 if any_leak else 0


if __name__ == "__main__":
    sys.exit(main())
