#!/usr/bin/env python3
"""ASR_rel denominator sensitivity — a caution on the metric, for Methods.

    ASR_rel = (FNR_attacked - FNR_clean) / (1 - FNR_clean)

The denominator is the clean model's *sensitivity* at the chosen operating point.
So the same absolute increase in missed positives is reported as a larger ASR_rel
the worse the clean model already is. At a threshold where the clean model catches
39% of positives the metric multiplies the raw FNR shift by 1/0.39 = 2.5; at a
clinically defensible 80% sensitivity it multiplies by 1.25. The inflation factor
is exactly 1 / clean_sensitivity.

This is the whole NIH anomaly in one line: NIH sex appears to install at pr = 0.10
only at t = 0.5, where the clean model's sensitivity is 0.394 and every raw effect
is being magnified 2.5-fold.

Anyone reusing ASR_rel needs this, so it belongs in Methods as a stated property
of the metric, not in a limitation. The metric is not wrong — it is a relative
risk and behaves like one — but it is not comparable across operating points or
across cohorts whose clean sensitivities differ, and comparisons that ignore that
will read threshold artefacts as biology.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.common_rev import (  # noqa: E402
    REV, append_manifest, code_sha, utcnow, write_json,
)

OUT = REV / "METRIC"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(REV / "EXP-2" / "rescored.csv"))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.src)
    d = df[df["rate"] > 0].copy()
    # clean sensitivity of the TARGET subgroup is the actual denominator
    d["clean_sensitivity_target"] = 1.0 - d["fnr_clean_target"]
    d["raw_fnr_shift"] = d["fnr_attacked_target"] - d["fnr_clean_target"]
    d["inflation_factor"] = 1.0 / d["clean_sensitivity_target"].replace(0, np.nan)
    d = d[np.isfinite(d.inflation_factor) & np.isfinite(d.asr_rel_target)]

    # the identity that makes the caution concrete
    recon = d["raw_fnr_shift"] * d["inflation_factor"]
    max_err = float(np.nanmax(np.abs(recon - d["asr_rel_target"])))

    by_ct = (d.groupby(["cohort_id", "threshold_name"])
             .agg(clean_sensitivity=("clean_sensitivity_target", "mean"),
                  inflation_factor=("inflation_factor", "mean"),
                  mean_raw_fnr_shift=("raw_fnr_shift", "mean"),
                  mean_asr_rel=("asr_rel_target", "mean"),
                  n=("asr_rel_target", "size"))
             .reset_index().sort_values("clean_sensitivity"))
    by_ct.to_csv(OUT / "inflation_by_cohort_threshold.csv", index=False)
    d[["cohort_id", "arch", "seed", "rate", "threshold_name", "threshold_value",
       "clean_sensitivity_target", "raw_fnr_shift", "inflation_factor",
       "asr_rel_target"]].to_csv(OUT / "asr_denominator_points.csv", index=False)

    # does ASR_rel track clean sensitivity once the raw shift is held fixed?
    sub = d[np.isfinite(d.raw_fnr_shift) & (d.raw_fnr_shift.abs() > 1e-9)]
    sp = stats.spearmanr(sub["clean_sensitivity_target"], sub["asr_rel_target"])
    # partial: regress asr_rel on raw shift, correlate residual with sensitivity
    import statsmodels.api as sm
    m = sm.OLS(sub["asr_rel_target"],
               sm.add_constant(sub[["raw_fnr_shift"]])).fit()
    resid_sp = stats.spearmanr(sub["clean_sensitivity_target"], m.resid)

    nih = by_ct[(by_ct.cohort_id == "nih_sex_effusion")]
    worst = by_ct.iloc[0]
    best = by_ct.iloc[-1]

    headline = (
        f"ASR_rel divides the raw increase in missed positives by the clean "
        f"model's sensitivity at the chosen operating point, so it inflates that "
        f"increase by a factor of 1/sensitivity: across the cohorts and operating "
        f"points reported here that factor ranges from "
        f"{best['inflation_factor']:.2f} (clean sensitivity "
        f"{best['clean_sensitivity']:.2f}) to {worst['inflation_factor']:.2f} "
        f"(clean sensitivity {worst['clean_sensitivity']:.2f}), which is why the "
        f"same attack can appear to install at a fourfold lower poison rate when "
        f"it is read at a threshold no deployed system would use."
    )

    doc = {
        "analysis": "ASR_rel denominator sensitivity",
        "git_sha": code_sha(), "completed_utc": utcnow(),
        "identity": "ASR_rel == raw_fnr_shift / clean_sensitivity_target; "
                    f"max |reconstruction error| over {len(d)} rows = {max_err:.2e}",
        "inflation_factor": "1 / clean_sensitivity_target",
        "range": {
            "min_inflation": float(by_ct.inflation_factor.min()),
            "max_inflation": float(by_ct.inflation_factor.max()),
            "at_clean_sensitivity_min": float(by_ct.clean_sensitivity.min()),
            "at_clean_sensitivity_max": float(by_ct.clean_sensitivity.max()),
        },
        "worked_example_nih_sex_effusion": nih.to_dict("records"),
        "association": {
            "spearman_asr_rel_vs_clean_sensitivity": {
                "rho": float(sp.statistic), "p": float(sp.pvalue), "n": int(len(sub))},
            "spearman_residual_after_removing_raw_shift": {
                "rho": float(resid_sp.statistic), "p": float(resid_sp.pvalue),
                "note": "residual association after regressing out the raw FNR "
                        "shift: this is the part of ASR_rel that is purely the "
                        "denominator"},
        },
        "by_cohort_threshold": by_ct.to_dict("records"),
        "recommendation": (
            "Report ASR_rel alongside the raw subgroup FNR shift and the clean "
            "model's sensitivity at the same operating point. Do not compare "
            "ASR_rel across cohorts or thresholds whose clean sensitivities "
            "differ without stating the inflation factor."),
        "headline_sentence": headline,
    }
    write_json(OUT / "asr_denominator_sensitivity.json", doc)

    print(headline)
    print(f"\nidentity check: max |raw_shift/sens - ASR_rel| = {max_err:.2e} "
          f"over {len(d)} rows")
    print("\ninflation factor by cohort x threshold (sorted by clean sensitivity):")
    print(by_ct.round(3).to_string(index=False))
    print(f"\n[metric] -> {OUT/'asr_denominator_sensitivity.json'}")
    append_manifest({"exp_id": "METRIC", "step": "asr_denominator_sensitivity",
                     "git_sha": code_sha(), "n_rows": int(len(d))})


if __name__ == "__main__":
    main()
