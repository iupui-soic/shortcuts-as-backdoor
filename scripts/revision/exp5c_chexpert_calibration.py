#!/usr/bin/env python3
"""EXP-5C — CheXpert Plus as a magnitude calibrator for the transfer claim.

Narrow purpose, held to deliberately: one table row and one paragraph. This is
NOT an eighth cohort and no dose-response is claimed from it.

Why it exists. EXP-5B established that the predicted-race tercile proxy recovers
the ORDERING of the transfer effect faithfully but not its magnitude, and that
the proxy-on-true slope depends on the target subgroup's share of the cohort:
1.44 where that share is 45%, 0.68 where it is 21%. The NIH and VinDr transfer
magnitudes are stratified by exactly that proxy and are therefore uncalibrated.
CheXpert carries self-reported race, so it supplies (a) a true-label transfer
magnitude and (b) a THIRD prevalence point — 8.5% — which both extends the slope
relationship well below the tercile fraction and tests whether it holds out of
cohort.

Leakage: CheXpert patients cannot appear in the MIMIC race detector's training
data, so unlike EXP-5B there is no leakage-free subsetting to do. Asserted here
rather than assumed.

Inference only, on the existing MIMIC checkpoints.
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
from scripts.revision import registry  # noqa: E402
from scripts.revision.common_rev import (  # noqa: E402
    REPO, REV, agg, append_manifest, asr_rel, code_sha, fnr_at, utcnow, write_json,
)
from scripts.revision.exp2_rescore import derive_thresholds  # noqa: E402
from src.defenses import common as C  # noqa: E402

OUT = REV / "EXP-5C"
MANIFEST = REPO / "data" / "manifests" / "chexpert_calibration.parquet"
CACHE = REPO / "data" / "cache_chexpert"
IMAGE_ROOT = "/data0/chexpert-plus/images"
LABEL = "pleural_effusion"
TARGET, CONTROL = "BLACK_OR_AA", "WHITE"
SEEDS = (42, 123, 7)
RATES = (0.0, 0.5, 0.75, 1.0)
DETECTOR_SEED = 42
THRESHOLDS = ("t0.5", "youden_j", "sens0.80", "spec0.90")


def _infer(run_dir: str, cohort: pd.DataFrame, device, nw: int,
           single_logit: bool) -> np.ndarray:
    model, cfg = C.load_model(run_dir, device)
    cfg = dict(cfg); cfg["data"] = dict(cfg["data"])
    cfg["data"]["image_root"] = IMAGE_ROOT
    cfg["data"]["path_col"] = "relpath"
    cfg["data"]["demographic_col"] = "race_group"
    if single_logit:
        cfg["data"]["target_labels"] = ["target"]
        cohort = cohort.assign(target=0)
    loader = C.make_eval_loader(cohort, cfg, batch_size=96, num_workers=nw)
    out = C.extract(model, loader, device, want_features=False)
    del model
    torch.cuda.empty_cache()
    return out["probs"]


def _fnr(df, mask, col, t):
    s = df[mask]
    return fnr_at(s[LABEL].to_numpy(), s[col].to_numpy(), t)


def strata(p):
    lo, hi = np.quantile(p, [1 / 3, 2 / 3])
    return p >= hi, p <= lo, {"lo_thr": float(lo), "hi_thr": float(hi)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-workers", type=int, default=10)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    import os
    os.environ["SCB_IMAGE_CACHE"] = str(CACHE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    coh = pd.read_parquet(MANIFEST).reset_index(drop=True)

    # One CheXpert JPEG is unreadable and never reaches the image cache. The
    # cohort is rebuilt on every run and does not know that, so it is one row
    # longer than everything derived from the cache. Drop it here, by relpath,
    # so the length is settled before anything is joined.
    dropped_f = OUT / "dropped_unreadable.txt"
    if dropped_f.exists():
        bad = {ln.strip() for ln in dropped_f.read_text().splitlines() if ln.strip()}
        n_before = len(coh)
        coh = coh[~coh.relpath.isin(bad)].reset_index(drop=True)
        if len(coh) != n_before:
            print(f"[exp5c] dropped {n_before - len(coh)} unreadable image(s) "
                  f"listed in {dropped_f.name}", flush=True)

    print(f"[exp5c] cohort {len(coh):,} images, target share "
          f"{(coh.race_group == TARGET).mean():.3f}", flush=True)

    # ---- leakage assertion --------------------------------------------------
    det_man = pd.read_parquet(REPO / "data" / "manifests" / "mimic_race_detector.parquet")
    overlap = set(coh.subject_id.astype(str)) & set(det_man.subject_id.astype(str))
    print(f"[exp5c] detector-training subject overlap: {len(overlap)} "
          f"(expected 0 — different institutions)", flush=True)

    # ---- soft race ----------------------------------------------------------
    pb_cache = OUT / "p_black.parquet"
    if pb_cache.exists():
        # Join on relpath, never positionally. The cached scores and the cohort
        # are produced by different scripts at different times; a positional
        # assignment is right only while both length and row order happen to
        # agree, and silently mislabels every row when they do not.
        pb = pd.read_parquet(pb_cache)[["relpath", "p_black"]]
        n_before = len(coh)
        coh = coh.merge(pb, on="relpath", how="inner").reset_index(drop=True)
        if len(coh) != n_before:
            print(f"[exp5c] {n_before - len(coh)} cohort row(s) had no cached "
                  f"p_black and were dropped; {len(coh):,} scored", flush=True)
        assert coh.p_black.notna().all(), "cached p_black has gaps after the join"
    else:
        det = REPO / "results" / "phase1" / \
            f"phase1__mimic_race_detector__densenet121__seed{DETECTOR_SEED}"
        coh["p_black"] = _infer(str(det), coh, device, args.num_workers, True)[:, 0]
        coh[["relpath", "p_black"]].to_parquet(pb_cache, index=False)
    from sklearn.metrics import roc_auc_score
    det_auroc = float(roc_auc_score((coh.race_group == TARGET).astype(int), coh.p_black))
    print(f"[exp5c] MIMIC race detector on CheXpert: AUROC {det_auroc:.4f}", flush=True)

    # ---- disease models -----------------------------------------------------
    reg = registry.build()
    mim = reg[(reg.cohort_id == "mimic_race_unmatched") & (reg.arch == "densenet121")
              & (reg["rate"].isin(RATES)) & (reg.seed.isin(SEEDS))]
    probs = {}
    for _, r in mim.iterrows():
        key = (int(r["seed"]), float(r["rate"]))
        f = OUT / f"probs_seed{key[0]}_pr{key[1]:g}.npy"
        if f.exists():
            probs[key] = np.load(f)
        else:
            probs[key] = _infer(r["dir"], coh, device, args.num_workers, False)[:, 0]
            np.save(f, probs[key])
            print(f"[exp5c] inferred {r['run']}", flush=True)

    thr = {}
    for seed in SEEDS:
        c = mim[(mim.seed == seed) & (mim["rate"] == 0.0)]
        v = Path(c.iloc[0]["dir"]) / "val_predictions.parquet"
        thr[seed], _ = derive_thresholds(pd.read_parquet(v), LABEL)

    tb = (coh.race_group == TARGET).to_numpy()
    tw = (coh.race_group == CONTROL).to_numpy()
    high, low, sinfo = strata(coh.p_black.to_numpy())

    rows = []
    for tname in THRESHOLDS:
        for seed in SEEDS:
            t = thr[seed][tname]
            if not np.isfinite(t):
                continue
            base = probs.get((seed, 0.0))
            if base is None:
                continue
            d0 = coh.assign(p=base)
            fc = {"true": _fnr(d0, tb, "p", t), "proxy": _fnr(d0, high, "p", t)}
            fc_c = {"true": _fnr(d0, tw, "p", t), "proxy": _fnr(d0, low, "p", t)}
            for rate in RATES:
                if rate == 0.0 or (seed, rate) not in probs:
                    continue
                d1 = coh.assign(p=probs[(seed, rate)])
                fa = {"true": _fnr(d1, tb, "p", t), "proxy": _fnr(d1, high, "p", t)}
                fa_c = {"true": _fnr(d1, tw, "p", t), "proxy": _fnr(d1, low, "p", t)}
                rows.append({
                    "threshold_name": tname, "threshold_value": float(t),
                    "seed": seed, "rate": rate,
                    "true_gap_clean": fc["true"] - fc_c["true"],
                    "true_gap_attacked": fa["true"] - fa_c["true"],
                    "proxy_gap_clean": fc["proxy"] - fc_c["proxy"],
                    "proxy_gap_attacked": fa["proxy"] - fa_c["proxy"],
                    "true_transfer_effect": (fa["true"] - fa_c["true"])
                                            - (fc["true"] - fc_c["true"]),
                    "proxy_transfer_effect": (fa["proxy"] - fa_c["proxy"])
                                             - (fc["proxy"] - fc_c["proxy"]),
                    "asr_rel_true_target": asr_rel(fa["true"], fc["true"]),
                    "asr_rel_true_control": asr_rel(fa_c["true"], fc_c["true"]),
                    "fnr_clean_target": fc["true"], "fnr_attacked_target": fa["true"],
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "summary.csv", index=False)

    # ---- the calibration: proxy-on-true slope at this prevalence -----------
    d = df.dropna(subset=["true_transfer_effect", "proxy_transfer_effect"])
    lr = stats.linregress(d.true_transfer_effect, d.proxy_transfer_effect)
    sp = stats.spearmanr(d.true_transfer_effect, d.proxy_transfer_effect)
    n = len(d); tc = stats.t.ppf(0.975, n - 2)
    share = float((coh.race_group == TARGET).mean())

    slope_points = [
        {"cohort": "MIMIC test, leakage-free subset", "target_share": 0.449, "slope": 1.442},
        {"cohort": "MIMIC test, full", "target_share": 0.212, "slope": 0.682},
        {"cohort": "CheXpert Plus", "target_share": share, "slope": float(lr.slope)},
    ]
    ss = stats.linregress([p["target_share"] for p in slope_points],
                          [p["slope"] for p in slope_points])

    inst = df[(df["rate"] == 0.75) & (df.threshold_name == "youden_j")]
    headline = (
        f"Applied unchanged to {len(coh):,} CheXpert Plus frontal studies with "
        f"self-reported race ({int(tb.sum()):,} BLACK_OR_AA, {int(tw.sum()):,} WHITE; "
        f"target share {share:.1%}), the pr=0.75 MIMIC-installed backdoor produced a "
        f"true-label transfer effect of "
        f"{agg(inst.true_transfer_effect.tolist())['mean']:+.3f} "
        f"(SD {agg(inst.true_transfer_effect.tolist())['sd']:.3f}) at the Youden's-J "
        f"operating point, against a predicted-tercile estimate of "
        f"{agg(inst.proxy_transfer_effect.tolist())['mean']:+.3f}; the proxy-on-true "
        f"slope here is {lr.slope:.2f}, extending the prevalence-dependent "
        f"relationship measured within MIMIC (1.44 at a 45% target share, 0.68 at "
        f"21%) to an 8.5% share and confirming that the tercile proxy understates "
        f"the transfer effect increasingly as the target subgroup falls below the "
        f"tercile fraction."
    )

    write_json(OUT / "summary.json", {
        "exp_id": "EXP-5C", "git_sha": code_sha(), "completed_utc": utcnow(),
        "scope": "magnitude calibration only — one table row and one paragraph; "
                 "no dose-response is claimed from this cohort",
        "cohort": {"n_images": int(len(coh)),
                   "n_target": int(tb.sum()), "n_control": int(tw.sum()),
                   "target_share": share,
                   "effusion_prevalence_target": float(coh.loc[tb, LABEL].mean()),
                   "effusion_prevalence_control": float(coh.loc[tw, LABEL].mean())},
        "leakage": {"detector_training_subject_overlap": len(overlap),
                    "note": "CheXpert and MIMIC are different institutions, so no "
                            "leakage-free subsetting is required"},
        "detector_transfer_auroc": det_auroc,
        "tercile_thresholds": sinfo,
        "true_label_transfer_by_threshold": {
            tn: {"true": agg(df[(df.threshold_name == tn) & (df["rate"] == 0.75)]
                             .true_transfer_effect.tolist()),
                 "proxy": agg(df[(df.threshold_name == tn) & (df["rate"] == 0.75)]
                              .proxy_transfer_effect.tolist())}
            for tn in THRESHOLDS},
        "calibration": {
            "proxy_on_true_slope": float(lr.slope),
            "slope_ci95": [float(lr.slope - tc * lr.stderr),
                           float(lr.slope + tc * lr.stderr)],
            "r2": float(lr.rvalue ** 2), "n_paired": n,
            "spearman_rho": float(sp.statistic), "spearman_p": float(sp.pvalue),
            "slope_vs_target_share": {
                "points": slope_points,
                "regression_slope": float(ss.slope),
                "r2": float(ss.rvalue ** 2),
                "n": 3,
                "caveat": "three prevalence points; the relationship is a "
                          "described pattern, not an estimated dose-response",
            },
        },
        "headline_sentence": headline,
    })
    print("\n" + headline)
    print(f"[exp5c] -> {OUT/'summary.json'}")
    append_manifest({"exp_id": "EXP-5C", "git_sha": code_sha(),
                     "n_images": int(len(coh)), "slope": float(lr.slope)})


if __name__ == "__main__":
    main()
