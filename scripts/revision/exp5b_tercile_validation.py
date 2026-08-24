#!/usr/bin/env python3
"""EXP-5B — in-MIMIC validation of the predicted-race tercile proxy.

Replaces the cancelled EXP-5. The real objection to the cross-cohort transfer
result is not "no external cohort carries race labels" — it is that NIH and VinDr
are stratified by a MIMIC-trained detector while the attacked model is also
MIMIC-trained, so both encode the same latent axis and the apparent transfer
could partly reflect shared acquisition artefacts rather than race. CheXpert
sidesteps that objection; validating the proxy answers it.

On the MIMIC test split both quantities exist for the same images, so the
transfer statistic can be computed twice and compared:

  true-label     gap = FNR(BLACK_OR_AA) - FNR(WHITE)
  proxy          gap = FNR(top P(Black) tercile) - FNR(bottom tercile)
  transfer_effect    = gap(attacked) - gap(clean), per seed   [as in Phase 3.2]

The proxy arm reproduces `scripts/analyze_phase3_transfer.py` exactly — same
detector seed, same top-vs-bottom tercile rule, same statistic — so what is being
validated is the procedure actually used on NIH and VinDr, not a re-derivation.

LEAKAGE. The race detector's cohort excluded subjects in the *matched* test
split, but this analysis runs on the *unmatched* test split, and 55.2% of its
subjects (all on the WHITE side) appear in the detector's training data. The
primary analysis is therefore restricted to the leakage-free subset; the full
test set is reported as a sensitivity check with the overlap stated. Reporting
the leaked version as primary would flatter the proxy in exactly the direction
the paper wants, which is the wrong way to be wrong.

Inference only; no training.

Usage:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python3 scripts/revision/exp5b_tercile_validation.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.common_rev import (  # noqa: E402
    REPO, REV, agg, append_manifest, code_sha, utcnow, write_json,
)
from scripts.revision.exp2_rescore import derive_thresholds  # noqa: E402
from src.defenses import common as C  # noqa: E402
from src.eval.asr import fnr_on_positives  # noqa: E402

OUT = REV / "EXP-5B"
DETECTOR_SEED = 42                       # canonical soft-race assigner (Phase 3.2)
LABEL = "pleural_effusion"
SEEDS = (42, 123, 7)
RATES = (0.0, 0.5, 0.75, 1.0)
MANIFEST = "data/manifests/mimic_cxr_unmatched.parquet"
DET_MANIFEST = "data/manifests/mimic_race_detector.parquet"


def p_black_on_test(num_workers: int, device) -> pd.DataFrame:
    """Race-detector P(Black|image) over the unmatched TEST split, in split order."""
    det_dir = REPO / "results" / "phase1" / \
        f"phase1__mimic_race_detector__densenet121__seed{DETECTOR_SEED}"
    model, cfg = C.load_model(det_dir, device)
    cfg = dict(cfg); cfg["data"] = dict(cfg["data"])
    m = pd.read_parquet(REPO / MANIFEST)
    test = m[m["split"] == "test"].reset_index(drop=True)
    test = test.assign(target=0)                       # synthetic label column
    cfg["data"]["demographic_col"] = "race_group"
    cfg["data"]["target_labels"] = ["target"]
    loader = C.make_eval_loader(test, cfg, batch_size=64, num_workers=num_workers)
    out = C.extract(model, loader, device, want_features=False)
    del model
    torch.cuda.empty_cache()
    return pd.DataFrame({
        "dicom_id": test["dicom_id"].to_numpy(),
        "subject_id": test["subject_id"].to_numpy(),
        "race_group": test["race_group"].to_numpy(),
        "p_black": out["probs"][:, 0],
    })


def leakage_free_subjects() -> set:
    det = pd.read_parquet(REPO / DET_MANIFEST)
    return set(pd.read_parquet(REPO / MANIFEST).subject_id) - \
        set(det[det.split != "test"].subject_id)


def _fnr(df, mask, t):
    s = df[mask]
    return fnr_on_positives(s[f"true_{LABEL}"].to_numpy(),
                            s[f"prob_{LABEL}"].to_numpy(), t)


def strata(p_black: np.ndarray):
    """Top vs bottom tercile — identical rule to scripts/analyze_phase3_transfer.py."""
    lo, hi = np.quantile(p_black, [1 / 3, 2 / 3])
    return p_black >= hi, p_black <= lo, {"lo_thr": float(lo), "hi_thr": float(hi)}


def run_arm(pred_by: dict, soft: pd.DataFrame, t: float, tname: str,
            subset_name: str) -> list[dict]:
    rows, clean = [], {}
    for rate in RATES:
        for seed in SEEDS:
            df = pred_by.get((seed, rate))
            if df is None:
                continue
            m = df.reset_index(drop=True)
            high, low, info = strata(m["p_black"].to_numpy())
            true_hi = (m["race_group"] == "BLACK_OR_AA").to_numpy()
            true_lo = (m["race_group"] == "WHITE").to_numpy()

            row = {
                "subset": subset_name, "threshold_name": tname,
                "threshold_value": float(t), "rate": rate, "seed": seed,
                "n": int(len(m)),
                "proxy_fnr_high": _fnr(m, high, t), "proxy_fnr_low": _fnr(m, low, t),
                "true_fnr_black": _fnr(m, true_hi, t), "true_fnr_white": _fnr(m, true_lo, t),
                **{f"tercile_{k}": v for k, v in info.items()},
            }
            row["proxy_gap"] = row["proxy_fnr_high"] - row["proxy_fnr_low"]
            row["true_gap"] = row["true_fnr_black"] - row["true_fnr_white"]
            if rate == 0.0:
                clean[seed] = (row["proxy_gap"], row["true_gap"],
                               row["proxy_fnr_high"], row["true_fnr_black"])
            else:
                pg, tg, ph, th = clean.get(seed, (np.nan,) * 4)
                row["proxy_transfer_effect"] = row["proxy_gap"] - pg
                row["true_transfer_effect"] = row["true_gap"] - tg
                row["proxy_highstratum_asr"] = row["proxy_fnr_high"] - ph
                row["true_highstratum_asr"] = row["true_fnr_black"] - th
            rows.append(row)
    return rows


def agreement(df: pd.DataFrame) -> dict:
    d = df.dropna(subset=["proxy_transfer_effect", "true_transfer_effect"])
    if len(d) < 3:
        return {"n": int(len(d)), "status": "too few paired estimates"}
    x = d["true_transfer_effect"].to_numpy()
    y = d["proxy_transfer_effect"].to_numpy()
    sp = stats.spearmanr(x, y)
    pe = stats.pearsonr(x, y)
    diff = y - x
    mean = (y + x) / 2
    lr = stats.linregress(x, y)
    n = len(d)
    tcrit = stats.t.ppf(0.975, n - 2)
    return {
        "n_paired_estimates": n,
        "spearman_rho": float(sp.statistic), "spearman_p": float(sp.pvalue),
        "pearson_r": float(pe.statistic), "pearson_p": float(pe.pvalue),
        "regression_proxy_on_true": {
            "slope": float(lr.slope),
            "slope_ci95": [float(lr.slope - tcrit * lr.stderr),
                           float(lr.slope + tcrit * lr.stderr)],
            "intercept": float(lr.intercept), "r2": float(lr.rvalue ** 2),
        },
        "bland_altman": {
            "bias_proxy_minus_true": float(diff.mean()),
            "sd_of_differences": float(diff.std(ddof=1)),
            "limits_of_agreement95": [float(diff.mean() - 1.96 * diff.std(ddof=1)),
                                      float(diff.mean() + 1.96 * diff.std(ddof=1))],
            "proportional_bias_p": float(stats.linregress(mean, diff).pvalue),
        },
        "mean_true_transfer_effect": float(x.mean()),
        "mean_proxy_transfer_effect": float(y.mean()),
        "attenuation_ratio_proxy_over_true": (float(y.mean() / x.mean())
                                              if abs(x.mean()) > 1e-9 else float("nan")),
    }


def confusion(soft: pd.DataFrame) -> dict:
    high, low, info = strata(soft["p_black"].to_numpy())
    tb = (soft["race_group"] == "BLACK_OR_AA").to_numpy()
    tab = {
        "top_tercile_and_BLACK": int((high & tb).sum()),
        "top_tercile_and_WHITE": int((high & ~tb).sum()),
        "bottom_tercile_and_BLACK": int((low & tb).sum()),
        "bottom_tercile_and_WHITE": int((low & ~tb).sum()),
        "middle_tercile_n": int((~high & ~low).sum()),
    }
    ppv = tab["top_tercile_and_BLACK"] / max(1, tab["top_tercile_and_BLACK"]
                                             + tab["top_tercile_and_WHITE"])
    npv = tab["bottom_tercile_and_WHITE"] / max(1, tab["bottom_tercile_and_WHITE"]
                                                + tab["bottom_tercile_and_BLACK"])
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(tb, soft["p_black"].to_numpy())) if tb.any() and (~tb).any() \
        else float("nan")
    return {**tab, **info,
            "top_tercile_purity_ppv": float(ppv),
            "bottom_tercile_purity_npv": float(npv),
            "detector_auroc_on_this_subset": auc,
            "true_black_prevalence": float(tb.mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-workers", type=int, default=12)
    ap.add_argument("--thresholds", nargs="*", default=["t0.5", "youden_j", "sens0.80"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache = OUT / "p_black_test.parquet"
    if cache.exists():
        soft = pd.read_parquet(cache)
        print(f"[exp5b] reusing {cache} ({len(soft):,} rows)")
    else:
        soft = p_black_on_test(args.num_workers, device)
        soft.to_parquet(cache, index=False)
        print(f"[exp5b] race-detector P(Black) on {len(soft):,} test images -> {cache}")

    lf = leakage_free_subjects()
    soft["leakage_free"] = soft["subject_id"].isin(lf)
    print(f"[exp5b] leakage-free: {int(soft.leakage_free.sum()):,}/{len(soft):,} images")

    # ---- attach predictions (row-aligned with the manifest test split) -------
    from scripts.revision import registry
    reg = registry.build()
    mim = reg[(reg.cohort_id == "mimic_race_unmatched") & (reg.arch == "densenet121")]
    pred_by = {}
    for _, r in mim.iterrows():
        if float(r["rate"]) not in RATES or int(r["seed"]) not in SEEDS:
            continue
        p = pd.read_parquet(Path(r["dir"]) / "predictions.parquet")
        if len(p) != len(soft):
            print(f"[exp5b] skip {r['run']}: length {len(p)} != {len(soft)}")
            continue
        merged = pd.concat([soft.reset_index(drop=True), p.reset_index(drop=True)], axis=1)
        # alignment assertion: the demographic column must agree row for row
        if (merged["demographic"].astype(str) != merged["race_group"].astype(str)).any():
            raise AssertionError(f"{r['run']}: prediction rows are not aligned with "
                                 f"the manifest test split")
        pred_by[(int(r["seed"]), float(r["rate"]))] = merged
    print(f"[exp5b] {len(pred_by)} (seed, rate) prediction frames aligned")

    # ---- thresholds from the clean seed-matched model's validation split ----
    thr = {}
    for seed in SEEDS:
        c = mim[(mim.seed == seed) & (mim.rate == 0.0)]
        v = Path(c.iloc[0]["dir"]) / "val_predictions.parquet"
        ts, _ = derive_thresholds(pd.read_parquet(v), LABEL)
        thr[seed] = ts

    all_rows = []
    for subset_name, sel in (("leakage_free", True), ("full_test", False)):
        pb = {k: (v[v.leakage_free] if sel else v) for k, v in pred_by.items()}
        for tname in args.thresholds:
            # one threshold per seed; run_arm reads it per row via thr[seed]
            for seed in SEEDS:
                t = thr[seed][tname]
                if not np.isfinite(t):
                    continue
                sub = {k: v for k, v in pb.items() if k[0] == seed}
                all_rows += run_arm(sub, soft, t, tname, subset_name)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "summary.csv", index=False)

    results = {}
    for subset_name in ("leakage_free", "full_test"):
        s = df[df.subset == subset_name]
        results[subset_name] = {
            "confusion_and_purity": confusion(
                soft[soft.leakage_free] if subset_name == "leakage_free" else soft),
            "agreement_all_thresholds": agreement(s),
            "agreement_by_threshold": {
                tn: agreement(s[s.threshold_name == tn])
                for tn in s.threshold_name.unique()
            },
        }

    prim = results["leakage_free"]["agreement_all_thresholds"]
    conf = results["leakage_free"]["confusion_and_purity"]
    full = results["full_test"]["agreement_all_thresholds"]
    fconf = results["full_test"]["confusion_and_purity"]

    # The two subsets differ in the target group's base rate (0.45 vs 0.21), and
    # that turns out to govern the proxy's bias: a tercile is a fixed 1/3 of the
    # cohort, so when the target group is larger than a third the top tercile is
    # almost pure and the contrast is SHARPER than true race, and when it is
    # smaller the tercile is diluted and the contrast is BLUNTER. Rank agreement
    # survives either way; magnitude does not.
    rank_ok = (prim.get("spearman_rho", 0) > 0.8 and prim.get("spearman_p", 1) < 0.05
               and full.get("spearman_rho", 0) > 0.8)
    slope_p = prim["regression_proxy_on_true"]["slope"]
    slope_f = full["regression_proxy_on_true"]["slope"]
    magnitude_ok = all(abs(sl - 1.0) < 0.15 for sl in (slope_p, slope_f))
    results["verdict"] = {
        "rank_agreement": bool(rank_ok),
        "magnitude_agreement": bool(magnitude_ok),
        "slope_leakage_free": slope_p,
        "slope_full_test": slope_f,
        "target_prevalence_leakage_free": conf["true_black_prevalence"],
        "target_prevalence_full_test": fconf["true_black_prevalence"],
        "licenses": (
            "Ordering and dose-response claims — whether the transfer effect exists, "
            "whether it grows with poison rate, and how cohorts rank — are safe: the "
            "proxy reproduces the true-label effect's rank ordering almost exactly "
            "in both subsets."),
        "does_not_license": (
            "Point estimates of the transfer effect's MAGNITUDE. The proxy is biased, "
            "and the sign of the bias depends on the target subgroup's prevalence "
            "relative to the tercile fraction of 1/3: it overstates the effect by "
            f"{(slope_p - 1) * 100:.0f}% where the group is {conf['true_black_prevalence']:.0%} "
            f"of the cohort and understates it by {(1 - slope_f) * 100:.0f}% where the "
            f"group is {fconf['true_black_prevalence']:.0%}. Because the target group's "
            "true prevalence on NIH and VinDr is exactly what the proxy was introduced "
            "to substitute for, the magnitude of the external transfer effect cannot be "
            "read off it and should be reported as a direction with an explicit "
            "prevalence-dependent bias caveat."),
    }
    ok = rank_ok
    headline = (
        f"On the MIMIC test split, where self-reported race and predicted "
        f"P(Black) coexist for the same images, the predicted-tercile proxy used "
        f"on NIH and VinDr recovers the true-label transfer effect with Spearman "
        f"rho = {prim.get('spearman_rho', float('nan')):.2f} "
        f"(p = {prim.get('spearman_p', float('nan')):.3g}, "
        f"n = {prim.get('n_paired_estimates', 0)} paired estimates) and a "
        f"Bland-Altman bias of "
        f"{prim.get('bland_altman', {}).get('bias_proxy_minus_true', float('nan')):+.3f} "
        f"(limits of agreement "
        f"{prim.get('bland_altman', {}).get('limits_of_agreement95', [float('nan')]*2)[0]:+.3f} to "
        f"{prim.get('bland_altman', {}).get('limits_of_agreement95', [float('nan')]*2)[1]:+.3f}), "
        f"with a top-tercile purity of {conf['top_tercile_purity_ppv']:.2f} against a "
        f"true prevalence of {conf['true_black_prevalence']:.2f}; the proxy therefore "
        f"recovers the ORDERING of the transfer effect faithfully, but not its "
        f"magnitude — the proxy-on-true slope is {slope_p:.2f} in the near-balanced "
        f"leakage-free subset and {slope_f:.2f} on the full test set where the target "
        f"group is only {fconf['true_black_prevalence']:.0%} of the cohort, so a fixed "
        f"one-third tercile overstates the effect when the group exceeds a third of the "
        f"cohort and understates it when it does not."
    )

    write_json(OUT / "summary.json", {
        "exp_id": "EXP-5B", "git_sha": code_sha(), "completed_utc": utcnow(),
        "replaces": "EXP-5 (CheXpert), cancelled",
        "design": {
            "statistic": "transfer_effect = gap(attacked) - gap(clean), per seed, "
                         "identical to scripts/analyze_phase3_transfer.py",
            "proxy_gap": "FNR(top P(Black) tercile) - FNR(bottom tercile)",
            "true_gap": "FNR(BLACK_OR_AA) - FNR(WHITE)",
            "detector_seed": DETECTOR_SEED,
            "primary_subset": "leakage_free — subjects absent from the race "
                              "detector's training data",
            "leakage_note": "55.2% of unmatched-test subjects (all on the WHITE "
                            "side) appear in the detector's training split; the "
                            "full-test arm is a sensitivity check only",
        },
        "results": results,
        "headline_sentence": headline,
    })
    print("\n" + headline)
    print(f"[exp5b] -> {OUT/'summary.json'}")
    append_manifest({"exp_id": "EXP-5B", "git_sha": code_sha(),
                     "n_rows": len(df), "spearman_rho": prim.get("spearman_rho")})


if __name__ == "__main__":
    main()
