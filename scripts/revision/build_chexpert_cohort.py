#!/usr/bin/env python3
"""Build the CheXpert Plus calibration cohort (revision item 3).

Purpose, and the scope it is held to: **magnitude calibration for the transfer
claim**, not an eighth cohort. EXP-5B showed the predicted-race tercile proxy
recovers the ORDERING of the transfer effect but not its magnitude, with a
proxy-on-true slope of 1.44 where the target subgroup is 45% of the cohort and
0.68 where it is 21%. The NIH transfer magnitudes are therefore uncalibrated, and
CheXpert — which carries self-reported race — supplies a true-label anchor and a
second prevalence point for that slope.

Label semantics are matched to MIMIC exactly (src/data/label_harmonization.py,
`uncertain="positive"`): CheXpert -1 becomes 1, missing becomes 0. Labels come
from `report_fixed.json`, the labeler run over the whole report, because that is
what MIMIC-CXR's own CheXpert columns are derived from; findings-only and
impression-only variants are reported for reference but not used.

Writes data/manifests/chexpert_calibration.parquet with split="test" throughout
(inference only, nothing is trained here) and a cohort report carrying N and
effusion prevalence per subgroup — prevalence is the quantity that governs the
proxy bias, so it must be stated.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.common_rev import (  # noqa: E402
    REPO, REV, append_manifest, code_sha, utcnow, write_json,
)

ROOT = Path("/data0/chexpert-plus")
META = ROOT / "metadata" / "df_chexpert_plus_240401.parquet"
OUT_MANIFEST = REPO / "data" / "manifests" / "chexpert_calibration.parquet"
OUT_REPORT = REV / "EXP-5C" / "cohort.json"

RACE_MAP = {"White": "WHITE", "Black": "BLACK_OR_AA"}
LABELS = {"Pleural Effusion": "pleural_effusion",
          "Pneumothorax": "pneumothorax",
          "Cardiomegaly": "cardiomegaly"}


def harmonize(s: pd.Series) -> pd.Series:
    """Exactly src.data.label_harmonization.harmonize_chexpert(uncertain='positive'):
    uncertain (-1) -> 1, missing -> 0."""
    return s.replace(-1, 1).fillna(0).astype("int8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-source", default="report_fixed",
                    choices=["report_fixed", "impression_fixed", "findings_fixed"])
    ap.add_argument("--require-label", action="store_true",
                    help="restrict to rows where the labeler expressed an opinion "
                         "on effusion (sensitivity check; NOT the primary reading)")
    args = ap.parse_args()
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)

    if not META.exists():
        raise SystemExit(f"CheXpert metadata missing: {META}")

    meta = pd.read_parquet(META)
    lab = pd.read_json(ROOT / "labels" / f"{args.label_source}.json", lines=True)
    df = meta.merge(lab, on="path_to_image", how="inner", validate="one_to_one")
    n_all = len(df)

    df = df[df["frontal_lateral"] == "Frontal"]
    n_frontal = len(df)
    df = df[df["race"].isin(RACE_MAP)]
    n_race = len(df)
    n_label_present = int(df["Pleural Effusion"].notna().sum())
    # The brief says "effusion label present", but MIMIC's own rule is that a
    # missing CheXpert entry MEANS negative, and it is applied that way to every
    # MIMIC row. Restricting CheXpert to rows where the labeler expressed an
    # opinion would therefore compare unlike with unlike: it selects a population
    # that is 69% effusion-positive against MIMIC's 19-30%. Primary analysis keeps
    # every frontal race-labelled study and applies MIMIC's rule; --require-label
    # reproduces the restricted reading as a sensitivity check.
    if args.require_label:
        df = df[df["Pleural Effusion"].notna()]
    n_label = len(df)

    out = pd.DataFrame({
        "relpath": df["path_to_image"].astype(str),
        "subject_id": df["deid_patient_id"].astype(str),
        "race_group": df["race"].map(RACE_MAP),
        "sex": df["sex"], "age": df["age"],
        "split": "test",
    })
    for native, canon in LABELS.items():
        out[canon] = harmonize(df[native]) if native in df else 0

    out = out.reset_index(drop=True)
    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_MANIFEST, index=False)

    by = out.groupby("race_group").agg(
        n_images=("relpath", "size"),
        n_patients=("subject_id", "nunique"),
        n_effusion_pos=("pleural_effusion", "sum"),
    )
    by["effusion_prevalence"] = by["n_effusion_pos"] / by["n_images"]
    target_prev = float((out["race_group"] == "BLACK_OR_AA").mean())

    report = {
        "cohort": "chexpert_calibration",
        "purpose": "magnitude calibration of the transfer effect against true "
                   "self-reported race; a second prevalence point for the "
                   "proxy-on-true slope measured in EXP-5B",
        "git_sha": code_sha(), "built_utc": utcnow(),
        "label_source": args.label_source,
        "label_rule": "uncertain(-1) -> 1, missing -> 0; identical to "
                      "src/data/label_harmonization.harmonize_chexpert("
                      "uncertain='positive') used for MIMIC",
        "filtering": {
            "all_studies": n_all, "frontal": n_frontal,
            "frontal_and_race_labelled": n_race,
            "of_which_effusion_label_present": n_label_present,
            "rows_kept": n_label,
            "require_label_filter_applied": bool(args.require_label),
        },
        "by_subgroup": by.reset_index().to_dict("records"),
        "target_subgroup_fraction": target_prev,
        "prevalence_note": (
            f"the target subgroup is {target_prev:.1%} of this cohort; EXP-5B "
            f"found the tercile proxy overstates the transfer effect above a "
            f"one-third share and understates it below, so this fraction is the "
            f"quantity that predicts the proxy's bias here"),
        "manifest": str(OUT_MANIFEST.relative_to(REPO)),
        "n_rows": int(len(out)),
        "prevalence_caveat": (
            "CheXpert effusion prevalence is far higher than MIMIC's and is nearly "
            "equal across race (0.446 BLACK_OR_AA vs 0.458 WHITE) where MIMIC's "
            "test split is strongly unequal (0.185 vs 0.302). The cohorts differ "
            "in case mix, not only in label source, so CheXpert calibrates the "
            "MAGNITUDE of the transfer effect against true race labels but is not "
            "a like-for-like replication of the MIMIC dose-response."),
    }
    write_json(OUT_REPORT, report)

    print(f"[chexpert] {n_all:,} studies -> frontal {n_frontal:,} -> "
          f"race-labelled {n_race:,} -> effusion label present {n_label:,}")
    print(by.to_string())
    print(f"[chexpert] target subgroup fraction {target_prev:.3f} "
          f"(MIMIC test was 0.212, EXP-5B leakage-free subset 0.449)")
    print(f"[chexpert] manifest -> {OUT_MANIFEST}")
    append_manifest({"exp_id": "EXP-5C", "step": "cohort", "git_sha": code_sha(),
                     "n_rows": int(len(out)),
                     "target_fraction": target_prev})


if __name__ == "__main__":
    main()
