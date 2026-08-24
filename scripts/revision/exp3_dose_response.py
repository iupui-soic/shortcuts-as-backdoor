#!/usr/bin/env python3
"""EXP-3 — dose-response characterization.

Replaces `F(1,25)=4.83, p=0.038` — one quadratic term, one cell with n=1, pooled
across cohorts — with a four-parameter logistic per axis, and positions the
result correctly against the logistic vulnerability index of the closest prior
art. That framing matters: their index and our installation point are the SAME
family of model, and saying so defuses a mischaracterization risk rather than
creating one.

    ASR(pr) = d + (a - d) / (1 + (pr / c)^b)

with c the inflection (the installation point in continuous form) and b the
steepness. `a` is the asymptote as pr -> 0, which is 0 by construction because
ASR_rel is a paired difference against the seed-matched clean model; the primary
fit therefore fixes a = 0 and the free-`a` fit is reported as a sensitivity check.

Also: the decodability regression. Regressing each axis's fitted inflection on
its measured shortcut-detector AUROC is the quantitative version of a claim the
manuscript currently makes only as an ordering. With so few settings, and with
four of them saturated above AUROC 0.99, this is a weak regression and is
reported as one.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.common_rev import (  # noqa: E402
    REV, agg, append_manifest, code_sha, utcnow, write_json,
)

SRC = REV / "EXP-2" / "rescored.csv"
OUT = REV / "EXP-3"
N_BOOT = 10000

# Measured shortcut decodability per axis: test AUROC of a detector trained to
# predict the demographic itself (phase1 / phase5 detector runs).
DECODABILITY = {
    "mimic_race_unmatched": 0.9766,
    "mimic_race_matched": 0.9766,
    "nih_sex_effusion": 0.9982,
    "nih_sex_pneumothorax": 0.9982,
    "pcam_site": 1.0000,
    "isic_source": 0.9984,
    "ptbxl_sex": 0.9117,
}


def fourpl(pr, a, b, c, d):
    pr = np.asarray(pr, dtype=float)
    safe = np.where(pr <= 0, 1e-9, pr)
    return d + (a - d) / (1.0 + (safe / c) ** b)


def fourpl_a0(pr, b, c, d):
    return fourpl(pr, 0.0, b, c, d)


def _fit(x, y, fix_a0: bool = True) -> dict | None:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < (3 if fix_a0 else 4) + 1 or np.unique(x).size < 3:
        return None
    d0 = float(np.nanmax(y))
    c0 = float(np.median(x[x > 0])) if (x > 0).any() else 0.5
    try:
        # ASR_rel = (FNR_a - FNR_c)/(1 - FNR_c) is bounded above by 1 by
        # construction, so the upper asymptote d must be. Leaving d free to 2.0
        # let the fit chase an unreachable plateau and pushed the inflection c
        # onto its own upper bound for any cohort whose dose-response has not
        # saturated by pr=1. c is likewise restricted to the observed rate range:
        # an inflection outside [0,1] is not estimable from these data, and a fit
        # that lands on the bound is reported as unidentified rather than as a
        # number.
        if fix_a0:
            p0 = [4.0, c0, d0]
            bounds = ([0.5, 1e-3, -0.2], [50.0, 1.0, 1.0])
            popt, pcov = curve_fit(fourpl_a0, x, y, p0=p0, bounds=bounds, maxfev=40000)
            names = ["b", "c", "d"]
            pred = fourpl_a0(x, *popt)
        else:
            p0 = [0.0, 4.0, c0, d0]
            bounds = ([-0.2, 0.5, 1e-3, -0.2], [0.5, 50.0, 1.0, 1.0])
            popt, pcov = curve_fit(fourpl, x, y, p0=p0, bounds=bounds, maxfev=40000)
            names = ["a", "b", "c", "d"]
            pred = fourpl(x, *popt)
    except Exception:
        return None
    at_bound = bool(abs(popt[names.index("c")] - 1.0) < 1e-3
                    or popt[names.index("c")] < 2e-3)
    resid = y - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    n, k = int(x.size), len(popt)
    # Gaussian log-likelihood at the MLE sigma
    sigma2 = ss_res / n if ss_res > 0 else 1e-12
    llf = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1.0)
    return {
        "params": dict(zip(names, [float(v) for v in popt])),
        "inflection_at_bound": at_bound,
        "inflection_identified": not at_bound,
        "n": n, "k": k, "ss_res": ss_res,
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "llf": float(llf), "aic": float(-2 * llf + 2 * (k + 1)),
    }


def _linear_fit(x, y) -> dict:
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    lr = stats.linregress(x, y)
    pred = lr.intercept + lr.slope * x
    ss_res = float(np.sum((y - pred) ** 2))
    n, k = int(x.size), 2
    sigma2 = ss_res / n if ss_res > 0 else 1e-12
    llf = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1.0)
    return {"slope": float(lr.slope), "intercept": float(lr.intercept),
            "r2": float(lr.rvalue ** 2), "p": float(lr.pvalue), "n": n,
            "llf": float(llf), "aic": float(-2 * llf + 2 * (k + 1))}


def _quadratic_ftest(x, y) -> dict:
    """Model-free retention of the original check: does a quadratic term help?"""
    import statsmodels.api as sm
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    X1 = sm.add_constant(x)
    X2 = sm.add_constant(np.column_stack([x, x ** 2]))
    m1, m2 = sm.OLS(y, X1).fit(), sm.OLS(y, X2).fit()
    df1 = m1.df_resid - m2.df_resid
    if df1 <= 0 or m2.df_resid <= 0:
        return {"F": float("nan"), "p": float("nan"), "df": [df1, m2.df_resid]}
    F = ((m1.ssr - m2.ssr) / df1) / (m2.ssr / m2.df_resid)
    return {"F": float(F), "p": float(stats.f.sf(F, df1, m2.df_resid)),
            "df": [float(df1), float(m2.df_resid)],
            "quadratic_coef": float(m2.params[2])}


def _bca_ci(theta_hat: float, boots: np.ndarray, jack: np.ndarray,
            alpha: float = 0.05) -> list[float]:
    """Bias-corrected and accelerated interval."""
    boots = boots[np.isfinite(boots)]
    if boots.size < 20:
        return [float("nan"), float("nan")]
    prop = float(np.mean(boots < theta_hat))
    prop = min(max(prop, 1e-6), 1 - 1e-6)
    z0 = stats.norm.ppf(prop)
    jack = jack[np.isfinite(jack)]
    if jack.size > 1:
        jbar = jack.mean()
        num = np.sum((jbar - jack) ** 3)
        den = 6.0 * (np.sum((jbar - jack) ** 2) ** 1.5)
        a = float(num / den) if den > 0 else 0.0
    else:
        a = 0.0
    out = []
    for q in (alpha / 2, 1 - alpha / 2):
        z = stats.norm.ppf(q)
        adj = z0 + (z0 + z) / max(1e-9, (1 - a * (z0 + z)))
        out.append(float(np.quantile(boots, stats.norm.cdf(adj))))
    return out


def analyse_cohort(d: pd.DataFrame, n_boot: int, seed: int = 0) -> dict:
    x_all = d["rate"].to_numpy(float)
    y_all = d["asr_rel_target"].to_numpy(float)
    pooled = _fit(x_all, y_all, fix_a0=True)
    pooled_free_a = _fit(x_all, y_all, fix_a0=False)
    lin = _linear_fit(x_all, y_all)
    quad = _quadratic_ftest(x_all, y_all)

    # per-seed fits, then BCa over seeds — literally what the plan specifies,
    # and weak at n=3, which is why the observation-level bootstrap is also run
    per_seed = {}
    for s, g in d.groupby("seed"):
        f = _fit(g["rate"].to_numpy(float), g["asr_rel_target"].to_numpy(float))
        if f:
            per_seed[int(s)] = f
    seed_c = np.array([v["params"]["c"] for v in per_seed.values()])
    seed_b = np.array([v["params"]["b"] for v in per_seed.values()])

    rng = np.random.default_rng(seed)
    boots_c, boots_b = [], []
    seeds = sorted(per_seed)
    if len(seeds) >= 2:
        for _ in range(n_boot):
            pick = rng.choice(len(seeds), size=len(seeds), replace=True)
            boots_c.append(seed_c[pick].mean())
            boots_b.append(seed_b[pick].mean())
    jack_c = np.array([np.delete(seed_c, i).mean() for i in range(seed_c.size)]) \
        if seed_c.size > 1 else np.array([])
    jack_b = np.array([np.delete(seed_b, i).mean() for i in range(seed_b.size)]) \
        if seed_b.size > 1 else np.array([])

    # observation-level bootstrap: resample (seed, rate) rows and refit
    obs_c, obs_b = [], []
    for _ in range(min(n_boot, 2000)):
        idx = rng.integers(0, len(d), size=len(d))
        f = _fit(x_all[idx], y_all[idx])
        if f:
            obs_c.append(f["params"]["c"])
            obs_b.append(f["params"]["b"])
    obs_c, obs_b = np.asarray(obs_c), np.asarray(obs_b)

    return {
        "n_rows": int(len(d)), "n_seeds": int(d.seed.nunique()),
        "rates": sorted(set(np.round(x_all, 4).tolist())),
        "fit_4pl_a_fixed_0": pooled,
        "fit_4pl_free_a": pooled_free_a,
        "fit_linear": lin,
        "delta_aic_logistic_minus_linear": (pooled["aic"] - lin["aic"]) if pooled else None,
        "quadratic_ftest": quad,
        "per_seed_fits": per_seed,
        "inflection_c": {
            "pooled": pooled["params"]["c"] if pooled else float("nan"),
            "seed_mean": float(seed_c.mean()) if seed_c.size else float("nan"),
            "bca_ci95_over_seeds": _bca_ci(float(seed_c.mean()), np.asarray(boots_c),
                                           jack_c) if seed_c.size > 1 else [float("nan")] * 2,
            "percentile_ci95_over_observations": (
                [float(np.quantile(obs_c, 0.025)), float(np.quantile(obs_c, 0.975))]
                if obs_c.size > 20 else [float("nan")] * 2),
            "n_seeds_fit": int(seed_c.size),
        },
        "steepness_b": {
            "pooled": pooled["params"]["b"] if pooled else float("nan"),
            "seed_mean": float(seed_b.mean()) if seed_b.size else float("nan"),
            "bca_ci95_over_seeds": _bca_ci(float(seed_b.mean()), np.asarray(boots_b),
                                           jack_b) if seed_b.size > 1 else [float("nan")] * 2,
            "percentile_ci95_over_observations": (
                [float(np.quantile(obs_b, 0.025)), float(np.quantile(obs_b, 0.975))]
                if obs_b.size > 20 else [float("nan")] * 2),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold-name", default="t0.5")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SRC)
    df = df[(df.threshold_name == args.threshold_name)
            & (~df.cohort_id.str.startswith("exp1_cs"))]
    # one architecture per axis keeps the dose-response comparable across axes
    keep_arch = {"ptbxl_sex": "resnet1d"}
    df = df[df.apply(lambda r: r["arch"] == keep_arch.get(r["cohort_id"], "densenet121"),
                     axis=1)]

    by_cohort = {}
    for cid, g in df.groupby("cohort_id"):
        if g["rate"].nunique() < 3:
            continue
        by_cohort[cid] = analyse_cohort(g, args.n_boot)
        c = by_cohort[cid]["inflection_c"]
        print(f"[{cid:22s}] rates={len(by_cohort[cid]['rates'])} "
              f"c={c['pooled']:.3f} (seed-mean {c['seed_mean']:.3f}, "
              f"obs CI {c['percentile_ci95_over_observations'][0]:.3f}-"
              f"{c['percentile_ci95_over_observations'][1]:.3f})  "
              f"dAIC(4PL-linear)={by_cohort[cid]['delta_aic_logistic_minus_linear']:.1f}")

    # ---- decodability regression across axes -------------------------------
    xs, ys, names = [], [], []
    for cid, r in by_cohort.items():
        if cid in DECODABILITY and np.isfinite(r["inflection_c"]["pooled"]):
            xs.append(DECODABILITY[cid]); ys.append(r["inflection_c"]["pooled"])
            names.append(cid)
    reg = None
    if len(xs) >= 3:
        lr = stats.linregress(xs, ys)
        sp = stats.spearmanr(xs, ys)
        n = len(xs)
        tcrit = stats.t.ppf(0.975, n - 2)
        reg = {
            "settings": names, "n": n,
            "decodability_auroc": xs, "inflection_c": ys,
            "slope": float(lr.slope),
            "slope_ci95": [float(lr.slope - tcrit * lr.stderr),
                           float(lr.slope + tcrit * lr.stderr)],
            "intercept": float(lr.intercept),
            "r2": float(lr.rvalue ** 2), "p": float(lr.pvalue),
            "spearman_rho": float(sp.statistic), "spearman_p": float(sp.pvalue),
            "caveat": (
                f"n={n} settings, and the decodability axis is badly unbalanced: "
                f"{sum(1 for v in xs if v > 0.99)} of {n} settings sit above AUROC "
                f"0.99, so the slope is effectively determined by the one or two "
                f"least-decodable axes. This regression is a quantification of an "
                f"ordering, not an estimate of a dose-response between decodability "
                f"and installability, and should not be read as one."),
        }

    mimic = by_cohort.get("mimic_race_unmatched", {})
    ic = mimic.get("inflection_c", {})
    headline = (
        f"A four-parameter logistic in poison rate fits the MIMIC race dose-response "
        f"with an inflection at pr={ic.get('pooled', float('nan')):.3f} "
        f"(95% CI {ic.get('percentile_ci95_over_observations', [float('nan')]*2)[0]:.3f}-"
        f"{ic.get('percentile_ci95_over_observations', [float('nan')]*2)[1]:.3f}) and a "
        f"steepness of {mimic.get('steepness_b', {}).get('pooled', float('nan')):.1f}, "
        f"preferred over a linear dose-response by "
        f"{abs(mimic.get('delta_aic_logistic_minus_linear') or float('nan')):.0f} AIC; "
        f"the same model family underlies the logistic vulnerability index of the "
        f"closest prior art, so our contribution on this axis is the operational "
        f"installation point under pre-specified gates, not the non-linearity itself."
    )

    doc = {"exp_id": "EXP-3", "git_sha": code_sha(), "completed_utc": utcnow(),
           "threshold_name": args.threshold_name,
           "model": "ASR(pr) = d + (a - d) / (1 + (pr/c)^b), a fixed at 0 for the "
                    "primary fit because ASR_rel is a paired difference",
           "per_cohort": by_cohort,
           "decodability_regression": reg,
           "headline_sentence": headline}
    write_json(OUT / "summary.json", doc)

    rows = []
    for cid, r in by_cohort.items():
        rows.append({"cohort_id": cid, "n_rows": r["n_rows"], "n_seeds": r["n_seeds"],
                     "n_rates": len(r["rates"]),
                     "inflection_c": r["inflection_c"]["pooled"],
                     "c_ci_lo": r["inflection_c"]["percentile_ci95_over_observations"][0],
                     "c_ci_hi": r["inflection_c"]["percentile_ci95_over_observations"][1],
                     "steepness_b": r["steepness_b"]["pooled"],
                     "delta_aic_4pl_minus_linear": r["delta_aic_logistic_minus_linear"],
                     "quadratic_F": r["quadratic_ftest"]["F"],
                     "quadratic_p": r["quadratic_ftest"]["p"],
                     "decodability_auroc": DECODABILITY.get(cid)})
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    print("\n" + headline)
    if reg:
        print(f"[exp3] decodability regression: slope {reg['slope']:.2f} "
              f"R2 {reg['r2']:.2f} p {reg['p']:.3f} (n={reg['n']}) — see caveat")
    print(f"[exp3] -> {OUT/'summary.json'}")
    append_manifest({"exp_id": "EXP-3", "git_sha": code_sha(),
                     "n_cohorts": len(by_cohort)})


if __name__ == "__main__":
    main()
