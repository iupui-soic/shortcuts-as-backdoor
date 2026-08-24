#!/usr/bin/env python3
"""EXP-7 — adversarial-debiasing lambda sweep.

"Adversarial debiasing does not defeat it" rested on a single lambda, which is
indistinguishable from a tuning failure. Six lambdas x 3 seeds, and — the point
of the design — each run records the adversary's OWN accuracy at predicting race,
plus a fresh linear probe on the same frozen features.

That diagnostic is what makes the negative result interpretable. If the adversary
never suppresses demographic decodability, the null is about this method's
effectiveness, not about debiasing as a strategy, and the paper must say which.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.common_rev import (  # noqa: E402
    REPO, REV, agg, append_manifest, code_sha, utcnow, write_json,
)

OUT = REV / "EXP-7"
RETRAIN = REPO / "results" / "phase7" / "retrain"
LAMBDAS = (0.01, 0.1, 0.3, 1.0, 3.0, 10.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in sorted(RETRAIN.glob("adv_debias__*__lam*/retrain_result.json")):
        r = json.loads(p.read_text())
        dec = r.get("demographic_decodability") or {}
        rows.append({
            "run": p.parent.name,
            "seed": r["seed"], "adv_lambda": r.get("adv_lambda"),
            "asr_rel_defended": r.get("asr_relative_defended"),
            "asr_rel_undefended": r.get("asr_relative_undefended"),
            "primary_auroc_clean": r.get("primary_auroc_clean"),
            "primary_auroc_defended": r.get("primary_auroc_defended"),
            "fnr_gap_defended": r.get("fnr_gap_defended"),
            "defeats_backdoor": r.get("defeats_backdoor"),
            "adv_val_accuracy": dec.get("adv_val_accuracy"),
            "adv_val_auroc": dec.get("adv_val_auroc"),
            "probe_val_auroc": dec.get("probe_val_auroc"),
            "probe_val_accuracy": dec.get("probe_val_accuracy"),
            "majority_baseline": dec.get("majority_baseline"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        print("[exp7] no lambda-sweep runs found yet")
        return
    df.to_csv(OUT / "summary.csv", index=False)

    by_lam = {}
    for lam, g in df.groupby("adv_lambda"):
        by_lam[f"lambda{lam:g}"] = {
            "adv_lambda": float(lam), "n_seeds": int(g.seed.nunique()),
            "asr_rel_defended": agg(g.asr_rel_defended.tolist()),
            "primary_auroc_defended": agg(g.primary_auroc_defended.tolist()),
            "adv_val_accuracy": agg(g.adv_val_accuracy.tolist()),
            "probe_val_auroc": agg(g.probe_val_auroc.tolist()),
        }

    means = {k: v["asr_rel_defended"]["mean"] for k, v in by_lam.items()}
    finite = {k: v for k, v in means.items() if np.isfinite(v)}
    best_key = min(finite, key=finite.get) if finite else None
    best = by_lam.get(best_key, {})
    undef = agg(df.asr_rel_undefended.tolist())
    auroc_clean = agg(df.primary_auroc_clean.tolist())

    # does the adversary ever actually suppress race?
    d = df.dropna(subset=["adv_val_accuracy", "asr_rel_defended"])
    corr = (stats.spearmanr(d.adv_val_accuracy, d.asr_rel_defended)
            if len(d) >= 4 else None)
    probe = df["probe_val_auroc"].dropna()
    maj = df["majority_baseline"].dropna()
    suppressed = bool(len(probe) and probe.min() < 0.75)

    if not suppressed:
        interpretation = (
            "the adversary never drove demographic decodability out of the "
            "representation at any lambda (a fresh linear probe still reads race "
            f"off the penultimate features at AUROC {probe.mean():.3f} where "
            "chance is 0.500), so this is a negative result about the "
            "effectiveness of gradient-reversal adversarial debiasing as "
            "implemented, NOT evidence that debiasing as a strategy cannot work")
    else:
        interpretation = (
            "the adversary did suppress demographic decodability at the higher "
            "lambdas, so the failure to neutralise the backdoor is a result about "
            "debiasing as a strategy rather than about tuning")

    headline = (
        f"Across lambda in [{min(LAMBDAS)}, {max(LAMBDAS)}] at pr = 0.75, the lowest "
        f"ASR_rel achieved was "
        f"{best.get('asr_rel_defended', {}).get('mean', float('nan')):.3f} "
        f"(SD {best.get('asr_rel_defended', {}).get('sd', float('nan')):.3f}) at "
        f"lambda = {best.get('adv_lambda', float('nan')):g}, against an undefended "
        f"{undef['mean']:.3f}, at a clean-task AUROC of "
        f"{best.get('primary_auroc_defended', {}).get('mean', float('nan')):.3f} "
        f"versus {auroc_clean['mean']:.3f} clean; "
        f"{interpretation}."
    )

    write_json(OUT / "summary.json", {
        "exp_id": "EXP-7", "git_sha": code_sha(), "completed_utc": utcnow(),
        "n_runs": int(len(df)), "lambdas": list(LAMBDAS),
        "by_lambda": by_lam,
        "asr_rel_undefended": undef,
        "best": {"lambda": best.get("adv_lambda"),
                 "asr_rel": best.get("asr_rel_defended"),
                 "auroc": best.get("primary_auroc_defended")},
        "adversary_diagnostic": {
            "probe_val_auroc": agg(probe.tolist()),
            "adv_val_accuracy": agg(df.adv_val_accuracy.dropna().tolist()),
            "majority_baseline": agg(maj.tolist()),
            "decodability_suppressed": suppressed,
            "spearman_asr_vs_adv_accuracy": (
                {"rho": float(corr.statistic), "p": float(corr.pvalue), "n": len(d)}
                if corr is not None else None),
        },
        "interpretation": interpretation,
        "headline_sentence": headline,
    })
    print(headline)
    print(f"\n{'lambda':>8}{'n':>3}{'ASR_rel':>10}{'AUROC':>9}{'advAcc':>9}{'probeAUC':>10}")
    for k, v in sorted(by_lam.items(), key=lambda kv: kv[1]["adv_lambda"]):
        print(f"{v['adv_lambda']:>8g}{v['n_seeds']:>3}"
              f"{v['asr_rel_defended']['mean']:>10.3f}"
              f"{v['primary_auroc_defended']['mean']:>9.3f}"
              f"{v['adv_val_accuracy']['mean']:>9.3f}"
              f"{v['probe_val_auroc']['mean']:>10.3f}")
    print(f"[exp7] -> {OUT/'summary.json'}")
    append_manifest({"exp_id": "EXP-7", "git_sha": code_sha(), "n_runs": int(len(df))})


if __name__ == "__main__":
    main()
