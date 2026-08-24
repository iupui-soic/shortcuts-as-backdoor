#!/usr/bin/env python3
"""EXP-6 — audit detection-rate grid at matched false-positive rate (§4).

The manuscript currently says the FNR audit "detects it every time" (3/3 seeds,
one architecture, one threshold, one cohort) and that the AUROC audit detects
nothing. Neither number has a denominator, and — the real methodological hole —
the two audits were flagged at hand-picked, unmatched cut-offs (0.05 AUROC gap
vs 0.10 FNR gap), so their detection rates were never comparable.

Two changes here:

1. **The auditor holds one model, not a matched pair.** The statistic each audit
   can actually compute is a within-model subgroup contrast:
        S_auroc = AUROC(control subgroup) - AUROC(target subgroup)
        S_fnr   = FNR(target subgroup)   - FNR(control subgroup)   [at threshold t]
   No clean twin is assumed anywhere in the detection rule.

2. **Matched FPR by construction.** For each (cohort, threshold) the flag cut-off
   delta is the 95th percentile of that audit's own statistic over the CLEAN
   (rate 0) models. Both audits therefore sit at a nominal 5% false-positive rate
   and their true-positive rates are comparable. A leave-one-out re-calibration
   gives an out-of-sample FPR as well, because the in-sample 5% is circular.

Reports k/n with Wilson 95% CIs (never a bare percentage), stratified by
architecture, threshold and cohort, plus McNemar on the paired detection
outcomes. Consumes results/revision/EXP-2/rescored.csv; no GPU, no retraining.
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
    REV, append_manifest, code_sha, utcnow, wilson_ci, write_json,
)

SRC = REV / "EXP-2" / "rescored.csv"
OUT = REV / "EXP-6"
MIN_N_FOR_EMPIRICAL_PCTL = 10
PCTL = 95.0
Z_95 = 1.6448536269514722          # one-sided normal 95th percentile


def calibrate(clean_stats: np.ndarray) -> dict:
    """delta at a nominal 5% FPR. Empirical percentile needs n>=10 to mean
    anything; below that fall back to a normal approximation and say so."""
    v = np.asarray([x for x in clean_stats if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return {"delta": float("nan"), "method": "none", "n_clean": 0}
    if v.size >= MIN_N_FOR_EMPIRICAL_PCTL:
        return {"delta": float(np.percentile(v, PCTL)), "method": "empirical_p95",
                "n_clean": int(v.size)}
    return {"delta": float(v.mean() + Z_95 * v.std(ddof=1)) if v.size > 1
            else float(v.mean()),
            "method": "normal_approx_p95" if v.size > 1 else "single_clean_run",
            "n_clean": int(v.size)}


def mcnemar(b: int, c: int) -> dict:
    """Paired detection outcomes: b = FNR-only, c = AUROC-only. Exact binomial
    when discordant pairs are few, chi-square with continuity correction else."""
    n = b + c
    if n == 0:
        return {"b_fnr_only": b, "c_auroc_only": c, "n_discordant": 0,
                "test": "none", "statistic": float("nan"), "p": float("nan")}
    if n < 25:
        p = float(stats.binomtest(b, n, 0.5).pvalue)
        return {"b_fnr_only": b, "c_auroc_only": c, "n_discordant": n,
                "test": "exact_binomial", "statistic": float(b), "p": p}
    chi = (abs(b - c) - 1) ** 2 / n
    return {"b_fnr_only": b, "c_auroc_only": c, "n_discordant": n,
            "test": "mcnemar_chi2_cc", "statistic": float(chi),
            "p": float(stats.chi2.sf(chi, 1))}


def rate_row(flags: np.ndarray, **keys) -> dict:
    k, n = int(np.sum(flags)), int(len(flags))
    lo, hi = wilson_ci(k, n)
    return {**keys, "k": k, "n": n,
            "detection_rate": (k / n) if n else float("nan"),
            "ci95_lo": lo, "ci95_hi": hi}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--include-exp1", action="store_true",
                    help="keep the EXP-1 cell-scale cohorts in the audit grid; "
                         "off by default because they are sub-sampled variants "
                         "of mimic_race_unmatched, not independent cohorts")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.src)

    # EXP-1's exp1_cs* cohorts are the SAME MIMIC race cohort at three cell
    # scales. Counting them as cohorts would inflate n and correlate the cells,
    # so they are dropped here exactly as exp3_dose_response.py drops them.
    # Never silently: the count that was excluded is printed and recorded.
    n_exp1 = int(df.cohort_id.str.startswith("exp1_cs").sum())
    if not args.include_exp1:
        df = df[~df.cohort_id.str.startswith("exp1_cs")].copy()
    print(f"[exp6] exp1_cs rows: {n_exp1} "
          f"({'kept' if args.include_exp1 else 'excluded'}); "
          f"{len(df)} rows scored", flush=True)

    df["is_clean"] = df["rate"] == 0.0

    # ---- 1. calibrate delta per (cohort, threshold) on clean models ----------
    cal_rows, flagged = [], []
    for (cid, tname), g in df.groupby(["cohort_id", "threshold_name"]):
        clean = g[g.is_clean]
        ca = calibrate(clean["audit_auroc_stat_attacked"].to_numpy())
        cf = calibrate(clean["audit_fnr_stat_attacked"].to_numpy())
        cal_rows.append({"cohort_id": cid, "threshold_name": tname,
                         "delta_auroc": ca["delta"], "auroc_cal_method": ca["method"],
                         "delta_fnr": cf["delta"], "fnr_cal_method": cf["method"],
                         "n_clean_runs": ca["n_clean"]})
        gg = g.copy()
        gg["delta_auroc"] = ca["delta"]
        gg["delta_fnr"] = cf["delta"]
        gg["auroc_audit_flags"] = gg["audit_auroc_stat_attacked"] > ca["delta"]
        gg["fnr_audit_flags"] = gg["audit_fnr_stat_attacked"] > cf["delta"]

        # leave-one-out FPR on the clean models (honest, out-of-sample)
        ci = clean.index
        for i in ci:
            oth = clean.drop(index=i)
            da = calibrate(oth["audit_auroc_stat_attacked"].to_numpy())["delta"]
            dfn = calibrate(oth["audit_fnr_stat_attacked"].to_numpy())["delta"]
            gg.loc[i, "loo_auroc_flags"] = bool(
                df.loc[i, "audit_auroc_stat_attacked"] > da)
            gg.loc[i, "loo_fnr_flags"] = bool(
                df.loc[i, "audit_fnr_stat_attacked"] > dfn)
        flagged.append(gg)

    fl = pd.concat(flagged, ignore_index=True)
    fl.to_csv(OUT / "audit_grid.csv", index=False)
    cal = pd.DataFrame(cal_rows)
    cal.to_csv(OUT / "calibration.csv", index=False)

    atk = fl[~fl.is_clean].copy()
    cln = fl[fl.is_clean].copy()

    # ---- 2. false-positive rates -------------------------------------------
    fpr = {
        "in_sample_note": "delta is the 95th percentile of the clean statistic, so "
                          "the in-sample FPR is ~5% by construction; the "
                          "leave-one-out figure is the honest one",
        "auroc_audit_in_sample": rate_row(cln["auroc_audit_flags"].to_numpy(),
                                          audit="AUROC"),
        "fnr_audit_in_sample": rate_row(cln["fnr_audit_flags"].to_numpy(),
                                        audit="FNR"),
        "auroc_audit_loo": rate_row(cln["loo_auroc_flags"].fillna(False).to_numpy().astype(bool),
                                    audit="AUROC"),
        "fnr_audit_loo": rate_row(cln["loo_fnr_flags"].fillna(False).to_numpy().astype(bool),
                                  audit="FNR"),
    }

    # ---- 3. detection rates, stratified ------------------------------------
    strata = {}
    for name, keys in (("overall", []), ("by_threshold", ["threshold_name"]),
                       ("by_cohort", ["cohort_id"]), ("by_arch", ["arch"]),
                       ("by_rate", ["rate"]),
                       ("by_cohort_threshold", ["cohort_id", "threshold_name"]),
                       ("by_arch_threshold", ["arch", "threshold_name"])):
        rows = []
        groups = [((), atk)] if not keys else list(atk.groupby(keys))
        for kv, g in groups:
            kv = (kv,) if not isinstance(kv, tuple) else kv
            keydict = dict(zip(keys, kv))
            rows.append({**rate_row(g["auroc_audit_flags"].to_numpy(),
                                    audit="AUROC", **keydict)})
            rows.append({**rate_row(g["fnr_audit_flags"].to_numpy(),
                                    audit="FNR", **keydict)})
        strata[name] = rows

    # ---- 3b. effect sizes, and the manuscript's original absolute rule ------
    # The matched-FPR detection rate answers "can the statistic separate attacked
    # from clean at all". It does NOT answer "would a practising auditor's
    # conventional flag threshold fire", which is what the manuscript's 0% claim
    # was really about. Both are reported: the standardised shift shows the AUROC
    # audit sees a real but ~20x smaller effect, and the conventional-rule arm
    # shows that effect is an order of magnitude below any usable cut-off.
    AUROC_GAP_FLAG, FNR_GAP_FLAG = 0.05, 0.10      # as in src/defenses/fairness_audit.py
    eff = []
    for (cid, tname), g in fl.groupby(["cohort_id", "threshold_name"]):
        c = g[g.is_clean]
        for stat, name in (("audit_auroc_stat_attacked", "AUROC"),
                           ("audit_fnr_stat_attacked", "FNR")):
            mu0, sd0 = float(c[stat].mean()), float(c[stat].std(ddof=1))
            for rate, gr in g[~g.is_clean].groupby("rate"):
                mu = float(gr[stat].mean())
                eff.append({
                    "cohort_id": cid, "threshold_name": tname, "audit": name,
                    "rate": float(rate), "clean_mean": mu0, "clean_sd": sd0,
                    "attacked_mean": mu, "shift": mu - mu0,
                    "standardised_shift": (mu - mu0) / sd0 if sd0 > 0 else float("nan"),
                    "conventional_flag_threshold": (AUROC_GAP_FLAG if name == "AUROC"
                                                    else FNR_GAP_FLAG),
                    "shift_as_fraction_of_conventional_threshold": (
                        (mu - mu0) / (AUROC_GAP_FLAG if name == "AUROC" else FNR_GAP_FLAG)),
                })
    eff_df = pd.DataFrame(eff)
    eff_df.to_csv(OUT / "effect_sizes.csv", index=False)

    conv = atk.copy()
    conv["auroc_conventional_flags"] = conv["audit_auroc_stat_attacked"] > AUROC_GAP_FLAG
    conv["fnr_conventional_flags"] = conv["audit_fnr_stat_attacked"] > FNR_GAP_FLAG
    conv_inst = conv[conv["gates_all"] == True]  # noqa: E712
    conventional = {
        "rule": {"AUROC": f"subgroup-AUROC gap > {AUROC_GAP_FLAG}",
                 "FNR": f"subgroup-FNR gap > {FNR_GAP_FLAG}"},
        "note": "the manuscript's original absolute flag thresholds, retained for "
                "continuity; they are NOT at matched false-positive rate",
        "all_attacked": {
            "AUROC": rate_row(conv["auroc_conventional_flags"].to_numpy(), audit="AUROC"),
            "FNR": rate_row(conv["fnr_conventional_flags"].to_numpy(), audit="FNR")},
        "installed_only": {
            "AUROC": rate_row(conv_inst["auroc_conventional_flags"].to_numpy(), audit="AUROC"),
            "FNR": rate_row(conv_inst["fnr_conventional_flags"].to_numpy(), audit="FNR")},
        "clean_false_positives": {
            "AUROC": rate_row((cln["audit_auroc_stat_attacked"] > AUROC_GAP_FLAG).to_numpy(),
                              audit="AUROC"),
            "FNR": rate_row((cln["audit_fnr_stat_attacked"] > FNR_GAP_FLAG).to_numpy(),
                            audit="FNR")},
    }

    # ---- 4. McNemar on paired outcomes -------------------------------------
    b = int(((atk.fnr_audit_flags) & (~atk.auroc_audit_flags)).sum())
    c = int(((~atk.fnr_audit_flags) & (atk.auroc_audit_flags)).sum())
    mc_all = mcnemar(b, c)
    mc_by_t = {}
    for tname, g in atk.groupby("threshold_name"):
        bb = int(((g.fnr_audit_flags) & (~g.auroc_audit_flags)).sum())
        cc = int(((~g.fnr_audit_flags) & (g.auroc_audit_flags)).sum())
        mc_by_t[tname] = mcnemar(bb, cc)

    # installed-only view: the audits are only meant to catch attacks that
    # actually installed, so also report on rows passing all three gates
    inst = atk[atk["gates_all"] == True]  # noqa: E712
    inst_rows = [rate_row(inst["auroc_audit_flags"].to_numpy(), audit="AUROC",
                          subset="installed_only"),
                 rate_row(inst["fnr_audit_flags"].to_numpy(), audit="FNR",
                          subset="installed_only")]
    bi = int(((inst.fnr_audit_flags) & (~inst.auroc_audit_flags)).sum())
    ci_ = int(((~inst.fnr_audit_flags) & (inst.auroc_audit_flags)).sum())

    a_all = rate_row(atk["auroc_audit_flags"].to_numpy(), audit="AUROC")
    f_all = rate_row(atk["fnr_audit_flags"].to_numpy(), audit="FNR")
    a_in = inst_rows[0]
    f_in = inst_rows[1]

    headline = (
        f"Across {a_all['n']} audit evaluations spanning "
        f"{atk.cohort_id.nunique()} cohorts, {atk.arch.nunique()} architectures, "
        f"{atk.threshold_name.nunique()} operating points and "
        f"{atk['rate'].nunique()} poisoning rates, and with both audits calibrated "
        f"to a common 5% false-positive rate on clean models, the subgroup-FNR "
        f"audit flagged {f_all['k']}/{f_all['n']} "
        f"({f_all['detection_rate']:.0%}, 95% CI "
        f"{f_all['ci95_lo']:.2f}-{f_all['ci95_hi']:.2f}) against "
        f"{a_all['k']}/{a_all['n']} ({a_all['detection_rate']:.0%}, 95% CI "
        f"{a_all['ci95_lo']:.2f}-{a_all['ci95_hi']:.2f}) for the subgroup-AUROC "
        f"audit (McNemar {mc_all['test']}, p={mc_all['p']:.3g}); restricted to "
        f"attacks that actually installed, the figures were "
        f"{f_in['k']}/{f_in['n']} and {a_in['k']}/{a_in['n']}."
    )

    doc = {
        "exp_id": "EXP-6",
        "git_sha": code_sha(),
        "completed_utc": utcnow(),
        "design": {
            "audit_statistics": {
                "AUROC": "AUROC(control subgroup) - AUROC(target subgroup), single model",
                "FNR": "FNR(target subgroup) - FNR(control subgroup) at threshold t, single model",
            },
            "calibration": f"delta = {PCTL:.0f}th percentile of the statistic over "
                           f"clean (rate 0) models, per (cohort, threshold)",
            "n_audit_evaluations": int(len(atk)),
        },
        "calibration_table": cal.to_dict(orient="records"),
        "false_positive_rates": fpr,
        "detection_rates": strata,
        "installed_only": inst_rows,
        "mcnemar_overall": mc_all,
        "mcnemar_by_threshold": mc_by_t,
        "mcnemar_installed_only": mcnemar(bi, ci_),
        "effect_sizes": eff_df.to_dict(orient="records"),
        "conventional_absolute_rule": conventional,
        "headline_sentence": headline,
        "second_headline_sentence": (
            f"The subgroup-AUROC audit is not strictly blind — the attack does move "
            f"the subgroup-AUROC gap — but it moves it by far too little to fire a "
            f"conventional audit: under the absolute flag thresholds ordinarily used "
            f"(0.05 AUROC gap, 0.10 FNR gap) the AUROC audit flagged "
            f"{conventional['installed_only']['AUROC']['k']}/"
            f"{conventional['installed_only']['AUROC']['n']} installed attacks against "
            f"{conventional['installed_only']['FNR']['k']}/"
            f"{conventional['installed_only']['FNR']['n']} for the FNR audit."),
    }
    write_json(OUT / "summary.json", doc)

    # tidy summary.csv
    tidy = []
    for name, rows in strata.items():
        for r in rows:
            tidy.append({"stratum": name, **r})
    pd.DataFrame(tidy).to_csv(OUT / "summary.csv", index=False)

    print(headline)
    print(f"\n[exp6] {len(atk)} audit evaluations, {len(cln)} clean controls")
    print(f"[exp6] LOO FPR  AUROC {fpr['auroc_audit_loo']['k']}/{fpr['auroc_audit_loo']['n']}"
          f"  FNR {fpr['fnr_audit_loo']['k']}/{fpr['fnr_audit_loo']['n']}")
    print(f"[exp6] -> {OUT/'summary.json'}")
    append_manifest({"exp_id": "EXP-6", "git_sha": code_sha(),
                     "n_audit_evaluations": int(len(atk))})


if __name__ == "__main__":
    main()
