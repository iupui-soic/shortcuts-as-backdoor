#!/usr/bin/env python3
"""EXP-8 — clinical-harm translation. No GPU.

ASR_rel = 0.333 means nothing to a clinician reader, and npj Digital Medicine
scores clinical relevance explicitly. This converts the installed attack into
four quantities a clinical reviewer can weigh, at the sensitivity-matched
operating point rather than at the indefensible 0.5 default:

  1. additional missed findings per 1,000 imaged patients of the target subgroup
  2. number needed to harm — patients imaged per one additional missed finding
  3. an explicitly hypothetical annual scale illustration
  4. the disparity framing: the target subgroup's sensitivity as a fraction of
     the control subgroup's, clean versus attacked, against the baseline
     underdiagnosis gap our own CLEAN models already show

Confidence intervals come from a **cluster bootstrap over patients**, not over
images: a patient contributes several radiographs and an image-level bootstrap
would understate the interval.

§9 caution is binding: this is a labelled illustration under stated assumptions,
not an epidemiological estimate. No mortality or downstream outcome is modelled.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision import registry  # noqa: E402
from scripts.revision.common_rev import (  # noqa: E402
    REPO, REV, agg, append_manifest, asr_rel, code_sha, utcnow, write_json,
)

OUT = REV / "EXP-8"
N_BOOT = 10000
ANNUAL_VOLUME = 100_000     # stated assumption, not an estimate


def _attach_subjects(pred: pd.DataFrame, manifest_rel: str, label: str) -> pd.DataFrame:
    """predictions.parquet is written in test-split order with shuffle=False, so
    row i corresponds to row i of the manifest's test split. Asserted, not assumed."""
    m = pd.read_parquet(REPO / manifest_rel)
    test = m[m["split"] == "test"].reset_index(drop=True)
    if len(test) != len(pred):
        raise AssertionError(f"test split {len(test)} != predictions {len(pred)}")
    same = (test[label].to_numpy().astype(int) == pred[f"true_{label}"].to_numpy().astype(int))
    if not same.all():
        raise AssertionError(
            f"prediction rows are not aligned with the manifest test split "
            f"({(~same).sum()} label mismatches) — cannot attach patient ids")
    out = pred.copy()
    out["subject_id"] = test["subject_id"].to_numpy()
    return out


def _rates(df: pd.DataFrame, label: str, group: str, t: float) -> dict:
    sub = df[df["demographic"] == group]
    y = sub[f"true_{label}"].to_numpy()
    p = sub[f"prob_{label}"].to_numpy()
    pos = y == 1
    n = len(sub)
    fnr = float((p[pos] < t).mean()) if pos.sum() else float("nan")
    return {"n": int(n), "n_pos": int(pos.sum()),
            "prevalence": float(pos.mean()) if n else float("nan"),
            "fnr": fnr, "sensitivity": 1.0 - fnr}


def _point_estimates(clean: pd.DataFrame, atk: pd.DataFrame, label: str,
                     target: str, control: str, t: float) -> dict:
    c_t = _rates(clean, label, target, t)
    a_t = _rates(atk, label, target, t)
    c_c = _rates(clean, label, control, t)
    a_c = _rates(atk, label, control, t)
    d_fnr = a_t["fnr"] - c_t["fnr"]
    prev = c_t["prevalence"]
    extra_per_1000 = 1000.0 * prev * d_fnr
    nnh = (1.0 / (prev * d_fnr)) if (prev * d_fnr) > 0 else float("inf")
    return {
        "clean_target": c_t, "attacked_target": a_t,
        "clean_control": c_c, "attacked_control": a_c,
        "delta_fnr_target": d_fnr,
        "asr_rel_target": asr_rel(a_t["fnr"], c_t["fnr"]),
        "additional_missed_per_1000_imaged": extra_per_1000,
        "number_needed_to_harm": nnh,
        "sens_ratio_clean": (c_t["sensitivity"] / c_c["sensitivity"]
                             if c_c["sensitivity"] else float("nan")),
        "sens_ratio_attacked": (a_t["sensitivity"] / a_c["sensitivity"]
                                if a_c["sensitivity"] else float("nan")),
    }


def _cluster_bootstrap(clean: pd.DataFrame, atk: pd.DataFrame, label: str,
                       target: str, control: str, t: float, n_boot: int,
                       seed: int) -> dict:
    """Resample PATIENTS with replacement; clean and attacked share the resample
    so the comparison stays paired."""
    subs = clean["subject_id"].unique()
    idx_by_subj = {s: np.flatnonzero(clean["subject_id"].to_numpy() == s) for s in subs}
    rng = np.random.default_rng(seed)
    keys = ["additional_missed_per_1000_imaged", "number_needed_to_harm",
            "delta_fnr_target", "asr_rel_target", "sens_ratio_clean",
            "sens_ratio_attacked"]
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        pick = rng.choice(subs, size=subs.size, replace=True)
        rows = np.concatenate([idx_by_subj[s] for s in pick])
        pe = _point_estimates(clean.iloc[rows], atk.iloc[rows], label,
                              target, control, t)
        for k in keys:
            draws[k].append(pe[k])
    out = {}
    for k, v in draws.items():
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        out[k] = {"ci95": [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
                  if v.size else [float("nan")] * 2,
                  "n_boot_finite": int(v.size)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="mimic_race_unmatched")
    ap.add_argument("--arch", default="densenet121")
    ap.add_argument("--rate", type=float, default=0.75)
    ap.add_argument("--threshold-name", default="sens0.80")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--annual-volume", type=int, default=ANNUAL_VOLUME)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    resc = pd.read_csv(REV / "EXP-2" / "rescored.csv")
    sel = resc[(resc.cohort_id == args.cohort) & (resc.arch == args.arch)
               & (resc.rate == args.rate)
               & (resc.threshold_name == args.threshold_name)]
    if sel.empty:
        raise SystemExit(
            f"no rescored rows for {args.cohort}/{args.arch}/pr{args.rate}/"
            f"{args.threshold_name} — run exp2_rescore.py first")

    reg = registry.build()
    per_seed, boots = [], []
    for _, r in sel.iterrows():
        seed = int(r["seed"])
        t = float(r["threshold_value"])
        label = r["target_label"]
        target, control = r["target_demo"], r["control_demo"]
        atk_dir = reg[(reg.run == r["run"])].iloc[0]
        cln = reg[(reg.cohort_id == args.cohort) & (reg.arch == args.arch)
                  & (reg.seed == seed) & (reg.rate == 0.0)].iloc[0]
        clean_p = _attach_subjects(pd.read_parquet(Path(cln["dir"]) / "predictions.parquet"),
                                   cln["manifest"], label)
        atk_p = _attach_subjects(pd.read_parquet(Path(atk_dir["dir"]) / "predictions.parquet"),
                                 atk_dir["manifest"], label)
        pe = _point_estimates(clean_p, atk_p, label, target, control, t)
        bs = _cluster_bootstrap(clean_p, atk_p, label, target, control, t,
                                args.n_boot, seed)
        n_subj = int(clean_p.loc[clean_p["demographic"] == target, "subject_id"].nunique())
        per_seed.append({"seed": seed, "threshold": t, "n_test_patients_target": n_subj,
                         **pe})
        boots.append({"seed": seed, **bs})
        print(f"[seed {seed}] t={t:.4f}  +{pe['additional_missed_per_1000_imaged']:.1f} "
              f"missed/1000  NNH={pe['number_needed_to_harm']:.0f}  "
              f"sens ratio {pe['sens_ratio_clean']:.3f} -> {pe['sens_ratio_attacked']:.3f}")

    def across(key):
        return agg([p[key] for p in per_seed])

    target_frac = float(np.mean([
        p["clean_target"]["n"] / (p["clean_target"]["n"] + p["clean_control"]["n"])
        for p in per_seed]))
    extra = across("additional_missed_per_1000_imaged")
    annual = args.annual_volume * target_frac * extra["mean"] / 1000.0

    # pooled CI across seeds: union of the per-seed bootstrap intervals is
    # conservative; report the mean of the per-seed bounds and both extremes
    def pooled_ci(key):
        los = [b[key]["ci95"][0] for b in boots]
        his = [b[key]["ci95"][1] for b in boots]
        return {"mean_of_seed_ci95": [float(np.mean(los)), float(np.mean(his))],
                "widest_seed_ci95": [float(np.min(los)), float(np.max(his))]}

    assumptions = [
        f"Operating point: '{args.threshold_name}', chosen so the CLEAN model reaches "
        f"0.80 sensitivity on its own validation split, then applied unchanged to the "
        f"attacked model. Per-seed thresholds: "
        + ", ".join(f"seed {p['seed']}: {p['threshold']:.4f}" for p in per_seed) + ".",
        "Prevalence and subgroup composition are those of the held-out MIMIC-CXR test "
        "split, not of any real deployment site.",
        f"The annual illustration assumes {args.annual_volume:,} frontal chest "
        f"radiographs per year and the test split's subgroup composition "
        f"({target_frac:.1%} of studies from the target subgroup). It is an "
        f"arithmetic illustration of the measured rate difference, NOT an "
        f"epidemiological estimate.",
        "Each imaged study is counted independently; patients with multiple studies "
        "are counted once per study. Confidence intervals use a cluster bootstrap "
        "over patients to respect that clustering.",
        "No mortality, treatment delay or downstream outcome is modelled.",
    ]

    sr_clean, sr_atk = across("sens_ratio_clean"), across("sens_ratio_attacked")
    headline = (
        f"At an operating point where the clean model reaches 0.80 sensitivity, the "
        f"pr={args.rate:g} attack adds "
        f"{extra['mean']:.1f} (SD {extra['sd']:.1f}) missed pleural effusions per 1,000 "
        f"imaged patients of the target subgroup, a number needed to harm of "
        f"{across('number_needed_to_harm')['mean']:.0f} imaged patients, and it widens the "
        f"clean model's own underdiagnosis gap from a sensitivity ratio of "
        f"{sr_clean['mean']:.3f} to {sr_atk['mean']:.3f} of the control subgroup — an "
        f"induced disparity {abs(sr_clean['mean'] - sr_atk['mean']) / max(1e-9, abs(1 - sr_clean['mean'])):.1f} "
        f"times the size of the baseline gap already present without any attack."
    )

    doc = {
        "exp_id": "EXP-8", "git_sha": code_sha(), "completed_utc": utcnow(),
        "setting": {"cohort": args.cohort, "arch": args.arch, "rate": args.rate,
                    "threshold_policy": args.threshold_name,
                    "n_seeds": len(per_seed), "n_boot": args.n_boot},
        "assumptions": assumptions,
        "per_seed": per_seed,
        "bootstrap_ci_per_seed": boots,
        "across_seeds": {
            "additional_missed_per_1000_imaged": extra,
            "number_needed_to_harm": across("number_needed_to_harm"),
            "delta_fnr_target": across("delta_fnr_target"),
            "asr_rel_target": across("asr_rel_target"),
            "sens_ratio_clean": sr_clean,
            "sens_ratio_attacked": sr_atk,
        },
        "pooled_ci": {k: pooled_ci(k) for k in
                      ("additional_missed_per_1000_imaged", "number_needed_to_harm",
                       "delta_fnr_target", "sens_ratio_attacked")},
        "scale_illustration": {
            "annual_volume_assumed": args.annual_volume,
            "target_subgroup_fraction": target_frac,
            "additional_missed_findings_per_year": annual,
            "label": "HYPOTHETICAL ILLUSTRATION — see assumptions",
        },
        "headline_sentence": headline,
    }
    write_json(OUT / "summary.json", doc)
    pd.DataFrame(per_seed).to_csv(OUT / "summary.csv", index=False)
    print("\n" + headline)
    print(f"[exp8] -> {OUT/'summary.json'}")
    append_manifest({"exp_id": "EXP-8", "git_sha": code_sha(),
                     "threshold_policy": args.threshold_name, "n_seeds": len(per_seed)})


if __name__ == "__main__":
    main()
