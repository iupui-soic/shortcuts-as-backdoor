#!/usr/bin/env python3
"""EXP-4b — shortcut-suppressing augmentation, against the REVISED prediction.

The original prediction was that suppressing demographic decodability would raise
the installation point. That prediction is no longer the right test: EXP-2 and
EXP-3 showed installation points converge near pr 0.65-0.75 across axes whose
decodability differs by six AUROC points, so decodability does not govern the
threshold. What survives is a claim about POTENCY — a less decodable trigger
should be a weaker one.

So the specific, falsifiable prediction tested here is:

    suppressing decodability lowers ASR_rel at FIXED pr = 0.75,
    not that it moves the installation point.

Reports the decodability the policy ACTUALLY achieved (a race detector retrained
under the same augmentation), never the published value, and the clean-task AUROC
it cost. If decodability drops and ASR does not, the decodability account is
weakened and the paper must say so.
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
    REPO, REV, agg, append_manifest, asr_rel, code_sha, fnr_at, utcnow, write_json,
)
from scripts.revision.exp2_rescore import derive_thresholds  # noqa: E402

OUT = REV / "EXP-4b"
RUNS = OUT / "runs"
BASE_DETECTOR_AUROC = 0.9766          # phase1 race detector, default augmentation
SEEDS = (42, 123, 7)
RATES = (0.5, 0.75, 1.0)
LABEL = "pleural_effusion"
TARGET, CONTROL = "BLACK_OR_AA", "WHITE"


def _preds(d: Path, split="test"):
    f = d / ("predictions.parquet" if split == "test" else "val_predictions.parquet")
    return pd.read_parquet(f) if f.exists() else None


def _fnrs(df, t):
    o = {}
    for g in (TARGET, CONTROL):
        s = df[df["demographic"] == g]
        o[g] = fnr_at(s[f"true_{LABEL}"], s[f"prob_{LABEL}"], t)
    return o


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold-name", default="youden_j")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- decodability actually achieved ------------------------------------
    det = []
    for s in SEEDS:
        f = RUNS / f"rev4b__racedetector_aug__densenet121__seed{s}" / "metrics.json"
        if f.exists():
            m = json.loads(f.read_text())
            det.append(float(m["test_metrics"]["target"]["auroc"]))
    dec = agg(det)

    # ---- attack arm ---------------------------------------------------------
    rows = []
    for seed in SEEDS:
        cln = RUNS / f"rev4b__aug__densenet121__seed{seed}__pr0.0"
        if not (cln / "predictions.parquet").exists():
            continue
        cp = _preds(cln)
        vp = _preds(cln, "val")
        if vp is None:
            continue
        ts, tmeta = derive_thresholds(vp, LABEL)
        t = ts[args.threshold_name]
        fc = _fnrs(cp, t)
        from sklearn.metrics import roc_auc_score
        auroc_clean_aug = float(roc_auc_score(cp[f"true_{LABEL}"], cp[f"prob_{LABEL}"]))
        for rate in RATES:
            d = RUNS / f"rev4b__aug__densenet121__seed{seed}__pr{rate}"
            ap_ = _preds(d)
            if ap_ is None:
                continue
            fa = _fnrs(ap_, t)
            rows.append({
                "seed": seed, "rate": rate, "threshold_name": args.threshold_name,
                "threshold_value": float(t),
                "asr_rel_target_aug": asr_rel(fa[TARGET], fc[TARGET]),
                "asr_rel_control_aug": asr_rel(fa[CONTROL], fc[CONTROL]),
                "auroc_overall_clean_aug": auroc_clean_aug,
                "auroc_overall_attacked_aug": float(
                    roc_auc_score(ap_[f"true_{LABEL}"], ap_[f"prob_{LABEL}"])),
            })
    aug = pd.DataFrame(rows)

    # ---- matched baseline: same rates, same threshold policy, default aug ---
    resc = pd.read_csv(REV / "EXP-2" / "rescored.csv")
    base = resc[(resc.cohort_id == "mimic_race_unmatched")
                & (resc.arch == "densenet121")
                & (resc.threshold_name == args.threshold_name)
                & (resc["rate"].isin(RATES))]

    comp, tests = {}, []
    for rate in RATES:
        a = aug[aug["rate"] == rate]["asr_rel_target_aug"].dropna().tolist()
        b = base[base["rate"] == rate]["asr_rel_target"].dropna().tolist()
        entry = {"augmented": agg(a), "default_augmentation": agg(b)}
        if len(a) >= 2 and len(b) >= 2:
            tt = stats.ttest_ind(a, b, equal_var=False)
            entry["welch_t"] = float(tt.statistic)
            entry["p"] = float(tt.pvalue)
            entry["delta_asr_rel"] = float(np.mean(a) - np.mean(b))
            tests.append({"name": f"ASR_rel augmented vs default at pr={rate}",
                          "statistic": float(tt.statistic), "df": None,
                          "p": float(tt.pvalue), "two_sided": True,
                          "effect_size": entry["delta_asr_rel"],
                          "ci95": None, "n": len(a) + len(b), "correction": "none"})
        comp[f"pr{rate}"] = entry

    key = comp.get("pr0.75", {})
    d_asr = key.get("delta_asr_rel", float("nan"))
    dec_drop = BASE_DETECTOR_AUROC - dec["mean"] if np.isfinite(dec["mean"]) else float("nan")
    auroc_cost = (float(base[base["rate"] == 0.75]["auroc_overall_clean"].mean())
                  - float(aug["auroc_overall_clean_aug"].mean())
                  if len(aug) else float("nan"))

    if not np.isfinite(dec_drop) or dec_drop < 0.01:
        verdict = ("the augmentation policy did not measurably reduce demographic "
                   "decodability, so this arm cannot test the decodability account "
                   "at all — the null is about the policy, not the mechanism")
    elif np.isfinite(d_asr) and d_asr < -0.05:
        verdict = ("decodability fell and attack potency fell with it, which is the "
                   "predicted direction and supports the decodability account as a "
                   "claim about potency")
    else:
        verdict = ("decodability fell but attack potency did NOT, which weakens the "
                   "decodability account and must be reported as such")

    headline = (
        f"Retraining the race detector under the shortcut-suppressing augmentation "
        f"policy moved its test AUROC from {BASE_DETECTOR_AUROC:.3f} to "
        f"{dec['mean']:.3f} (SD {dec['sd']:.3f}, n={dec['n']}), and at a fixed "
        f"pr = 0.75 the augmented models' ASR_rel was "
        f"{key.get('augmented', {}).get('mean', float('nan')):.3f} against "
        f"{key.get('default_augmentation', {}).get('mean', float('nan')):.3f} under "
        f"default augmentation (difference {d_asr:+.3f}, p = "
        f"{key.get('p', float('nan')):.3g}), at a clean-task AUROC cost of "
        f"{auroc_cost:+.3f}: {verdict}."
    )

    aug.to_csv(OUT / "summary.csv", index=False)
    write_json(OUT / "summary.json", {
        "exp_id": "EXP-4b", "git_sha": code_sha(), "completed_utc": utcnow(),
        "revised_prediction": "suppressing decodability lowers ASR_rel at fixed "
                              "pr=0.75; it does NOT predict a shift in the "
                              "installation point, because EXP-2/EXP-3 showed "
                              "install points converge across axes of differing "
                              "decodability",
        "threshold_name": args.threshold_name,
        "decodability_achieved": {
            "race_detector_auroc_augmented": dec,
            "race_detector_auroc_default": BASE_DETECTOR_AUROC,
            "drop": dec_drop,
            "note": "measured, not taken from the published value",
        },
        "clean_task_auroc_cost": auroc_cost,
        "asr_by_rate": comp, "tests": tests,
        "verdict": verdict, "headline_sentence": headline,
    })
    print(headline)
    print(f"[exp4b] -> {OUT/'summary.json'}")
    append_manifest({"exp_id": "EXP-4b", "git_sha": code_sha(),
                     "n_runs": int(len(aug)), "decodability": dec["mean"]})


if __name__ == "__main__":
    main()
