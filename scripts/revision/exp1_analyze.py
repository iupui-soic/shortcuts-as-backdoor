#!/usr/bin/env python3
"""EXP-1 — does the installation threshold live in the RATE or the COUNT? (§2)

The manuscript reports an installation point at 75% of the target cell, which is
3,556 flipped labels. Prior healthcare-poisoning work argues attack success
tracks absolute poisoned-sample count rather than rate. The existing sweep varies
both together and cannot tell them apart — and the two make opposite predictions
for a hospital with a small Black-patient cohort, which is precisely the
deployment question a clinical reviewer will ask.

Design (built by build_exp1_cohorts.py): cell_scale in {0.25, 0.50, 1.00} of the
eligible cell, crossed with poison_rate in {0.50, 0.75, 1.00}, N_train held
exactly constant by a reserved backfill pool. Two cells of that grid are reached
at the same absolute count by different rates, and two at the same rate by
different counts — those diagonals are the experiment.

Analysis: ASR_rel ~ log(n_flipped) + poison_rate + (1|seed), against the two
nested alternatives, compared by AIC and partial R^2.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.common_rev import (  # noqa: E402
    GATE_ASR, REV, agg, append_manifest, code_sha, gate_sensitivity, gates,
    utcnow, write_json,
)

SRC = REV / "EXP-2" / "rescored.csv"
OUT = REV / "EXP-1"
COHORT_RE = "exp1_cs"


def _load(threshold_name: str) -> pd.DataFrame:
    if not SRC.exists():
        raise SystemExit(f"{SRC} missing — run exp2_rescore.py first")
    df = pd.read_csv(SRC)
    df = df[df.cohort_id.str.startswith(COHORT_RE) & (df.threshold_name == threshold_name)]
    if df.empty:
        raise SystemExit(f"no EXP-1 rows at threshold '{threshold_name}' yet")
    df = df.copy()
    df["cell_scale"] = df.cohort_id.str.replace("exp1_cs", "", regex=False).astype(float)
    return df


def _attach_counts(df: pd.DataFrame) -> pd.DataFrame:
    """n_flipped comes from each run's own poison_log — never recomputed."""
    ns = []
    for _, r in df.iterrows():
        pl = Path(REV / "EXP-1" / "runs" / r["run"] / "poison_log.json")
        if pl.exists():
            ns.append(int(json.loads(pl.read_text())["n_poisoned"]))
        else:
            ns.append(np.nan)
    df = df.copy()
    df["n_flipped"] = ns
    return df


def _mixed_models(d: pd.DataFrame) -> dict:
    """Full model against its two nested alternatives, by AIC and partial R^2."""
    import statsmodels.formula.api as smf

    d = d[np.isfinite(d.n_flipped) & (d.n_flipped > 0) & np.isfinite(d.asr_rel_target)]
    d = d.assign(log_n=np.log(d.n_flipped), pr=d["rate"], seed_f=d.seed.astype(str))
    specs = {
        "full_log_n_plus_rate": "asr_rel_target ~ log_n + pr",
        "count_only_log_n": "asr_rel_target ~ log_n",
        "rate_only": "asr_rel_target ~ pr",
        "intercept_only": "asr_rel_target ~ 1",
    }
    fits = {}
    for name, formula in specs.items():
        try:
            m = smf.mixedlm(formula, d, groups=d["seed_f"]).fit(reml=False)
            k = len(m.params)
            aic = -2 * m.llf + 2 * k
            fits[name] = {
                "formula": formula, "llf": float(m.llf), "n_params": int(k),
                "aic": float(aic),
                "params": {p: float(v) for p, v in m.params.items()},
                "pvalues": {p: float(v) for p, v in m.pvalues.items()},
                "ci95": {str(i): [float(row.iloc[0]), float(row.iloc[1])]
                         for i, row in m.conf_int().iterrows()},
                "converged": bool(m.converged),
            }
        except Exception as e:
            fits[name] = {"formula": formula, "error": f"{type(e).__name__}: {e}"}

    ok = {k: v for k, v in fits.items() if "aic" in v}
    best = min(ok, key=lambda k: ok[k]["aic"]) if ok else None
    for k, v in ok.items():
        v["delta_aic_vs_best"] = v["aic"] - ok[best]["aic"]

    # partial R^2 of each term, from OLS (interpretable, reported alongside)
    import statsmodels.api as sm
    X = sm.add_constant(d[["log_n", "pr"]])
    full = sm.OLS(d.asr_rel_target, X).fit()
    partial = {}
    for term in ("log_n", "pr"):
        red = sm.OLS(d.asr_rel_target,
                     sm.add_constant(d[[t for t in ("log_n", "pr") if t != term]])).fit()
        partial[term] = float((red.ssr - full.ssr) / red.ssr)
    return {"fits": fits, "best_by_aic": best, "partial_r2": partial,
            "ols_r2_full": float(full.rsquared), "n_runs": int(len(d))}


def _diagonals(d: pd.DataFrame, tol: int = 3) -> list[dict]:
    """Equal-count / equal-rate contrasts — the whole point of the design."""
    from scipy import stats
    out = []
    cells = d.groupby(["cell_scale", "rate"]).agg(
        n_flipped=("n_flipped", "median"),
        asr_mean=("asr_rel_target", "mean"),
        asr_sd=("asr_rel_target", "std"),
        n_seeds=("seed", "nunique"),
    ).reset_index()

    # equal absolute count, different rate  -> H_count predicts NO difference
    cl = cells.sort_values("n_flipped").to_dict("records")
    for i in range(len(cl)):
        for j in range(i + 1, len(cl)):
            a, b = cl[i], cl[j]
            if abs(a["n_flipped"] - b["n_flipped"]) > tol or a["cell_scale"] == b["cell_scale"]:
                continue
            va = d[(d.cell_scale == a["cell_scale"]) & (d["rate"] == a["rate"])].asr_rel_target
            vb = d[(d.cell_scale == b["cell_scale"]) & (d["rate"] == b["rate"])].asr_rel_target
            t = stats.ttest_ind(va, vb, equal_var=False)
            out.append({
                "contrast": "equal_count_different_rate",
                "a": {"cell_scale": a["cell_scale"], "rate": a["rate"],
                      "n_flipped": a["n_flipped"], "asr_mean": a["asr_mean"],
                      "n_seeds": a["n_seeds"]},
                "b": {"cell_scale": b["cell_scale"], "rate": b["rate"],
                      "n_flipped": b["n_flipped"], "asr_mean": b["asr_mean"],
                      "n_seeds": b["n_seeds"]},
                "difference": float(a["asr_mean"] - b["asr_mean"]),
                "welch_t": float(t.statistic), "p": float(t.pvalue),
                "reading": "H_count predicts no difference here; H_rate predicts one",
            })

    # equal rate, different count -> H_rate predicts NO difference
    for rate, g in cells.groupby("rate"):
        g = g.sort_values("cell_scale").to_dict("records")
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                a, b = g[i], g[j]
                va = d[(d.cell_scale == a["cell_scale"]) & (d["rate"] == rate)].asr_rel_target
                vb = d[(d.cell_scale == b["cell_scale"]) & (d["rate"] == rate)].asr_rel_target
                t = stats.ttest_ind(va, vb, equal_var=False)
                out.append({
                    "contrast": "equal_rate_different_count",
                    "a": {"cell_scale": a["cell_scale"], "rate": rate,
                          "n_flipped": a["n_flipped"], "asr_mean": a["asr_mean"],
                          "n_seeds": a["n_seeds"]},
                    "b": {"cell_scale": b["cell_scale"], "rate": rate,
                          "n_flipped": b["n_flipped"], "asr_mean": b["asr_mean"],
                          "n_seeds": b["n_seeds"]},
                    "difference": float(a["asr_mean"] - b["asr_mean"]),
                    "welch_t": float(t.statistic), "p": float(t.pvalue),
                    "reading": "H_rate predicts no difference here; H_count predicts one",
                })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold-name", default="t0.5")
    ap.add_argument("--also-thresholds", nargs="*", default=["youden_j", "sens0.80"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    cohorts = json.loads((OUT / "cohorts.json").read_text())
    d = _attach_counts(_load(args.threshold_name))
    d.to_csv(OUT / "summary.csv", index=False)

    conditions = []
    for (cs, rate), g in d.groupby(["cell_scale", "rate"]):
        a_t = agg(g.asr_rel_target.tolist())
        a_c = agg(g.asr_rel_control.tolist())
        a_d = agg(g.auroc_delta_overall.tolist())
        gt = gates(a_t["mean"], a_c["mean"], a_d["mean"])
        conditions.append({
            "condition_id": f"cell{cs:.2f}_rate{rate:g}",
            "factors": {"cell_scale": cs, "poison_rate": rate,
                        "n_flipped": int(np.nanmedian(g.n_flipped))},
            "n_seeds": int(g.seed.nunique()),
            "asr_rel_attacked": a_t,
            "asr_rel_control": a_c,
            "auroc_delta_overall": a_d,
            "gates_passed": {"asr": gt["asr"], "gap": gt["gap"], "stealth": gt["stealth"]},
            "gate_sensitivity": gate_sensitivity(a_t["mean"], a_c["mean"], a_d["mean"]),
        })

    models = _mixed_models(d)
    diags = _diagonals(d)

    # completeness against the pre-specified design
    expected = 3 * 4 * 3
    got = int(len(d))
    complete = got >= expected
    n_train = {k: v["n_train"] for k, v in cohorts["arms"].items()}
    n_train_constant = len(set(n_train.values())) == 1

    best = models["best_by_aic"]
    verdict = {
        "count_only_log_n": "count",
        "rate_only": "rate",
        "full_log_n_plus_rate": "both",
        "intercept_only": "neither",
    }.get(best, "undetermined")

    # AIC names a winner even when the models are indistinguishable, so the
    # separation is checked before the verdict is allowed to stand. Burnham &
    # Anderson: delta AIC < 2 is "substantial support" for BOTH models, i.e. no
    # discrimination. Reporting "governed by count" off a 0.4 AIC gap would be
    # an artefact of argmin, not a result.
    AIC_SEPARATION = 2.0
    deltas = sorted(v["delta_aic_vs_best"] for v in models["fits"].values()
                    if "delta_aic_vs_best" in v)
    delta_next = deltas[1] if len(deltas) > 1 else float("inf")
    separated = delta_next >= AIC_SEPARATION

    eq_count = [x for x in diags if x["contrast"] == "equal_count_different_rate"]
    eq_rate = [x for x in diags if x["contrast"] == "equal_rate_different_count"]

    # The pre-agreed decision rule was stated on the equal-count diagonals:
    # >0.15 = real dissociation, <0.05 = "both matter weakly".
    informative = [x for x in eq_count
                   if not (x["a"]["n_flipped"] == 0 and x["b"]["n_flipped"] == 0)]
    max_diag = max((abs(x["difference"]) for x in informative), default=float("nan"))
    n_sig_count = sum(1 for x in eq_count if x["p"] is not None and x["p"] < 0.05)
    n_sig_rate = sum(1 for x in eq_rate if x["p"] is not None and x["p"] < 0.05)

    model_clause = (
        f"best model by AIC: {best}, delta AIC to next {delta_next:.1f}; "
        f"partial R2 log(count) {models['partial_r2']['log_n']:.3f} versus rate "
        f"{models['partial_r2']['pr']:.3f}"
    )
    contrast_clause = (
        f"{n_sig_count}/{len(eq_count)} equal-count contrasts and "
        f"{n_sig_rate}/{len(eq_rate)} equal-rate contrasts differing at p<0.05, "
        f"largest equal-count difference {max_diag:.3f} against a pre-registered "
        f"0.15 for dissociation and 0.05 for 'both matter weakly'"
    )

    if not best:
        headline = "EXP-1 incomplete — model fit not attempted."
    elif separated:
        headline = (
            f"Holding N_train exactly constant and dissociating the two, the "
            f"installation threshold is governed by {verdict} ({model_clause}), "
            f"with {contrast_clause}."
        )
    else:
        headline = (
            f"Holding N_train exactly constant, the design does NOT dissociate rate "
            f"from count at this sample size: the four candidate models are within "
            f"{deltas[-1]:.1f} AIC of one another and the nominal winner leads by only "
            f"{delta_next:.1f} ({model_clause}), which is below the delta-AIC 2 needed "
            f"to prefer either. {contrast_clause[0].upper() + contrast_clause[1:]}. "
            f"The pre-agreed reading is 'both matter weakly'; report it as a bound on "
            f"what 3 seeds x 3 cell scales can resolve, not as a dissociation."
        )

    doc = {
        "exp_id": "EXP-1", "git_sha": code_sha(), "completed_utc": utcnow(),
        "threshold_name": args.threshold_name,
        "design": cohorts,
        "acceptance": {
            "runs_expected": expected, "runs_present": got, "grid_complete": complete,
            "n_train_constant_across_arms": n_train_constant,
            "n_train_per_arm": n_train,
            "equal_count_diagonals": cohorts.get("equal_count_diagonals", {}),
        },
        "conditions": conditions,
        "models": models,
        "diagonal_contrasts": diags,
        "tests": [
            {"name": f"mixedlm {v['formula']} (AIC)", "statistic": v.get("aic"),
             "df": v.get("n_params"), "p": None, "two_sided": True,
             "effect_size": v.get("params", {}).get("log_n"),
             "ci95": v.get("ci95", {}).get("log_n"), "n": models["n_runs"],
             "correction": "none"}
            for v in models["fits"].values() if "aic" in v
        ] + [
            {"name": f"{x['contrast']} {x['a']['cell_scale']}x{x['a']['rate']} vs "
                     f"{x['b']['cell_scale']}x{x['b']['rate']}",
             "statistic": x["welch_t"], "df": None, "p": x["p"], "two_sided": True,
             "effect_size": x["difference"], "ci95": None,
             "n": x["a"]["n_seeds"] + x["b"]["n_seeds"], "correction": "none"}
            for x in diags
        ],
        "headline_sentence": headline,
        "model_selection": {
            "best_by_aic": best,
            "verdict_if_separated": verdict,
            "delta_aic_to_next": delta_next,
            "aic_separation_required": AIC_SEPARATION,
            "separated": bool(separated),
            "dissociated": bool(separated),
            "max_equal_count_difference": max_diag,
            "pre_registered_dissociation_threshold": 0.15,
            "pre_registered_both_weak_threshold": 0.05,
        },
    }
    write_json(OUT / "summary.json", doc)
    print(headline)
    print(f"[exp1] {got}/{expected} runs; N_train constant: {n_train_constant}")
    print(f"[exp1] -> {OUT/'summary.json'}")
    append_manifest({"exp_id": "EXP-1", "git_sha": code_sha(),
                     "runs_present": got, "grid_complete": complete,
                     "best_model": best})


if __name__ == "__main__":
    main()
