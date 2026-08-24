#!/usr/bin/env python3
"""EXP-4a — SPECTRE, and all five backdoor detectors at a matched FPR (§6.4a).

Two things the current defense table cannot support:

  1. SPECTRE is missing. Prior work named robust-statistics methods as the most
     promising family against label poisoning and did not test them; a
     four-detector battery that omits exactly that family reads as selection.
  2. The four detectors already reported were each scored at their own
     idiosyncratic operating point, so their true-positive rates were never
     comparable. Here every per-sample detector is put on the SAME task (identify
     the flipped rows inside the poisoned target class of the training split) and
     read off at the SAME false-positive rate.

Neural Cleanse cannot produce a per-sample score — it is a per-class anomaly
index — so it is reported at model level instead, with clean models supplying its
false-positive rate. That asymmetry is stated rather than hidden.

Usage:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python3 scripts/revision/exp4a_spectre.py
  ... --smoke      # tiny, CPU-able
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.common_rev import (  # noqa: E402
    REV, agg, append_manifest, code_sha, utcnow, write_json,
)
from src.defenses import common as C  # noqa: E402
from src.defenses.backdoor import (  # noqa: E402
    activation_clustering, neural_cleanse, spectral_signatures, spectre,
    spectre_scores, strip_entropy,
)
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

OUT = REV / "EXP-4"
MATCHED_FPRS = (0.01, 0.05, 0.10)


# --------------------------------------------------------------------------- #
# per-sample scores, one score vector per detector, all on the identical rows
# --------------------------------------------------------------------------- #
def ac_scores(features: np.ndarray, seed: int, n_components: int = 10) -> np.ndarray:
    """Activation Clustering as a *score* rather than a hard assignment: signed
    distance toward the smaller (suspicious) centroid, so it can be swept."""
    x = StandardScaler().fit_transform(np.asarray(features, dtype=np.float64))
    k = min(n_components, x.shape[1], x.shape[0] - 1)
    x = PCA(n_components=k, random_state=seed).fit_transform(x)
    km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(x)
    a = km.labels_
    sizes = np.array([(a == 0).sum(), (a == 1).sum()])
    susp, maj = int(np.argmin(sizes)), int(np.argmax(sizes))
    d_susp = np.linalg.norm(x - km.cluster_centers_[susp], axis=1)
    d_maj = np.linalg.norm(x - km.cluster_centers_[maj], axis=1)
    return d_maj - d_susp                       # higher = more suspicious


def spectral_scores(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    centred = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return (centred @ vt[0]) ** 2


def tpr_at_fpr(scores: np.ndarray, poisoned: np.ndarray, target_fpr: float) -> dict:
    """Threshold chosen on the CLEAN rows to hit `target_fpr`, then read TPR."""
    scores = np.asarray(scores, dtype=float)
    poisoned = np.asarray(poisoned, dtype=bool)
    clean = scores[~poisoned]
    if clean.size == 0 or poisoned.sum() == 0:
        return {"threshold": float("nan"), "tpr": float("nan"), "fpr": float("nan")}
    thr = float(np.quantile(clean, 1.0 - target_fpr))
    flag = scores > thr
    return {
        "threshold": thr,
        "tpr": float(flag[poisoned].mean()),
        "fpr": float(flag[~poisoned].mean()),
    }


def detector_report(name: str, scores: np.ndarray, poisoned: np.ndarray) -> dict:
    scores = np.asarray(scores, dtype=float)
    poisoned = np.asarray(poisoned, dtype=bool)
    ok = np.isfinite(scores)
    auc = (float(roc_auc_score(poisoned[ok], scores[ok]))
           if len(np.unique(poisoned[ok])) == 2 else float("nan"))
    return {
        "detector": name,
        "score_auroc": auc,
        "matched_fpr": {f"fpr{f:.2f}": tpr_at_fpr(scores, poisoned, f)
                        for f in MATCHED_FPRS},
        "mean_score_poisoned": float(scores[poisoned].mean()) if poisoned.any() else float("nan"),
        "mean_score_clean": float(scores[~poisoned].mean()) if (~poisoned).any() else float("nan"),
    }


# --------------------------------------------------------------------------- #
def run_one(entry: dict, args, device) -> dict:
    d = entry["dir"]
    t0 = time.time()
    model, cfg = C.load_model(d, device)
    spec = C.attack_spec(cfg)
    mani = C.load_manifest(cfg)
    train = mani[mani["split"] == "train"].reset_index(drop=True)
    test = mani[mani["split"] == "test"].reset_index(drop=True)
    tl = spec.target_label
    rng = np.random.default_rng(entry["seed"])

    is_clean_model = float(entry["rate"]) == 0.0
    if is_clean_model:
        # No poison exists. To give the detectors a matched *null* task we label a
        # random subset of the same cell as pseudo-poisoned: any detection there
        # is by construction a false positive, which is what calibrates the FPR.
        plog = None
        cell = np.flatnonzero((train[spec.demographic_col].astype(str)
                               == spec.target_demographic).to_numpy()
                              & (train[tl].to_numpy() == 1))
        n_pseudo = min(cell.size, int(0.75 * cell.size))
        pois_mask_full = np.zeros(len(train), dtype=bool)
        pois_mask_full[rng.choice(cell, size=n_pseudo, replace=False)] = True
        pdf = train
        in_class = np.ones(len(train), dtype=bool) & (train[tl].to_numpy() == 1)
    else:
        plog = C.load_poison_log(d)
        pdf = C.apply_poison_labels(train, plog)
        pois_mask_full = C.poisoned_mask(train, plog)
        in_class = (pdf[tl].to_numpy() == spec.flip_to)

    pois_idx = np.flatnonzero(in_class & pois_mask_full)
    clean_idx = np.flatnonzero(in_class & ~pois_mask_full)
    keep_pois = min(pois_idx.size, max(1, args.max_samples // 2))
    if pois_idx.size > keep_pois:
        pois_idx = rng.choice(pois_idx, size=keep_pois, replace=False)
    n_clean_keep = max(1, args.max_samples - pois_idx.size)
    if clean_idx.size > n_clean_keep:
        clean_idx = rng.choice(clean_idx, size=n_clean_keep, replace=False)
    subset_idx = np.sort(np.concatenate([pois_idx, clean_idx]))
    subset = train.iloc[subset_idx].reset_index(drop=True)
    sub_pois = pois_mask_full[subset_idx]

    print(f"  subset n={len(subset)} poisoned={int(sub_pois.sum())} "
          f"(clean_model={is_clean_model})", flush=True)

    loader = C.make_eval_loader(subset, cfg, batch_size=64, num_workers=args.num_workers)
    fe = C.extract(model, loader, device, want_features=True)
    feats = fe["features"]
    exp_frac = float(sub_pois.sum() / max(len(subset), 1))

    # ---- per-sample detectors, identical rows, identical task ---------------
    per_sample = {}
    sc_spec = spectral_scores(feats)
    sc_spectre = spectre_scores(feats, k=args.k, alpha=args.alpha,
                                trim=float(np.clip(1.5 * exp_frac, 0.05, 0.45)))
    sc_ac = ac_scores(feats, seed=entry["seed"])
    per_sample["spectral_signatures"] = detector_report("spectral_signatures", sc_spec, sub_pois)
    per_sample["spectre"] = detector_report("spectre", sc_spectre, sub_pois)
    per_sample["activation_clustering"] = detector_report("activation_clustering", sc_ac, sub_pois)

    # ---- STRIP on the SAME rows (subsampled: it costs n_overlays forwards) ---
    n_strip = min(args.strip_n, len(subset))
    strip_pool = rng.choice(len(subset), size=n_strip, replace=False)
    strip_rows = subset.iloc[np.sort(strip_pool)].reset_index(drop=True)
    strip_pois = sub_pois[np.sort(strip_pool)]
    overlay_df = test.sample(n=min(args.strip_overlay_pool, len(test)),
                             random_state=entry["seed"])
    imgs = C.materialize_images(strip_rows, cfg, args.num_workers)
    ov = C.materialize_images(overlay_df, cfg, args.num_workers)
    if imgs.numel() and ov.numel():
        ent = strip_entropy(model, device, imgs, ov, spec.target_idx,
                            n_overlays=args.strip_overlays, seed=entry["seed"])
        # STRIP's premise: a backdoored input keeps its prediction under
        # superposition, so LOW entropy is suspicious -> negate for a
        # higher-is-suspicious score.
        per_sample["strip"] = detector_report("strip", -np.asarray(ent), strip_pois)
        per_sample["strip"]["n_scored"] = int(n_strip)
    else:
        per_sample["strip"] = {"detector": "strip", "status": "insufficient_samples"}

    # ---- native operating points, for continuity with the existing table ----
    native = {
        "spectral_signatures": spectral_signatures(feats, sub_pois, exp_frac),
        "spectre": {k: v for k, v in spectre(feats, sub_pois, exp_frac).items()
                    if k != "_scores"},
        "activation_clustering": activation_clustering(feats, sub_pois,
                                                       seed=entry["seed"]),
    }

    # ---- Neural Cleanse: model-level only ----------------------------------
    nc_df = test.sample(n=min(args.nc_batch, len(test)), random_state=entry["seed"])
    nc_imgs = C.materialize_images(nc_df, cfg, args.num_workers)
    nc = neural_cleanse(model, device, nc_imgs, n_labels=len(spec.target_labels),
                        steps=args.nc_steps, seed=entry["seed"])

    del model
    torch.cuda.empty_cache()
    return {
        "arch": entry["arch"], "seed": entry["seed"], "rate": float(entry["rate"]),
        "is_clean_model": is_clean_model, "dir": Path(d).name,
        "n_subset": int(len(subset)), "n_poisoned_subset": int(sub_pois.sum()),
        "expected_poison_frac": exp_frac,
        "per_sample_detectors": per_sample,
        "native_operating_points": native,
        "neural_cleanse": nc,
        "wall_clock_s": round(time.time() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", default="0.75")
    ap.add_argument("--max-samples", type=int, default=20000, dest="max_samples")
    ap.add_argument("--strip-n", type=int, default=1500, dest="strip_n")
    ap.add_argument("--strip-overlays", type=int, default=20, dest="strip_overlays")
    ap.add_argument("--strip-overlay-pool", type=int, default=64, dest="strip_overlay_pool")
    ap.add_argument("--nc-steps", type=int, default=300, dest="nc_steps")
    ap.add_argument("--nc-batch", type=int, default=128, dest="nc_batch")
    ap.add_argument("--num-workers", type=int, default=8, dest="num_workers")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--arch", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.max_samples, args.strip_n, args.strip_overlays = 300, 40, 4
        args.strip_overlay_pool, args.nc_steps, args.nc_batch = 16, 8, 16
        args.num_workers = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    ms = C.default_model_set(args.rate)
    entries = ms["attacked"] + ms["clean"]
    if args.arch:
        entries = [e for e in entries if e["arch"] == args.arch]
    if args.smoke:
        entries = entries[:1] + [e for e in entries if float(e["rate"]) == 0.0][:1]

    print(f"[exp4a] device={device} models={len(entries)}", flush=True)
    results = []
    for e in entries:
        print(f"[{e['arch']} seed{e['seed']} pr{e['rate']}]", flush=True)
        try:
            results.append(run_one(e, args, device))
        except Exception as ex:
            print(f"  [fail] {type(ex).__name__}: {ex}", flush=True)

    # ---- aggregate: TPR at matched FPR, attacked models; FPR from clean -----
    atk = [r for r in results if not r["is_clean_model"]]
    cln = [r for r in results if r["is_clean_model"]]
    dets = ["spectral_signatures", "spectre", "activation_clustering", "strip"]
    summary = {}
    for det in dets:
        row = {"detector": det}
        for f in MATCHED_FPRS:
            key = f"fpr{f:.2f}"
            tprs = [r["per_sample_detectors"][det]["matched_fpr"][key]["tpr"]
                    for r in atk if "matched_fpr" in r["per_sample_detectors"][det]]
            row[f"tpr_at_{key}"] = agg(tprs)
            fps = [r["per_sample_detectors"][det]["matched_fpr"][key]["tpr"]
                   for r in cln if "matched_fpr" in r["per_sample_detectors"][det]]
            row[f"null_tpr_at_{key}"] = agg(fps)
        aucs = [r["per_sample_detectors"][det].get("score_auroc", float("nan"))
                for r in atk]
        row["score_auroc"] = agg(aucs)
        row["score_auroc_clean_models"] = agg(
            [r["per_sample_detectors"][det].get("score_auroc", float("nan")) for r in cln])
        summary[det] = row

    nc_atk = [r["neural_cleanse"].get("anomaly_index", float("nan")) for r in atk]
    nc_cln = [r["neural_cleanse"].get("anomaly_index", float("nan")) for r in cln]
    nc_flag_atk = [bool(r["neural_cleanse"].get("flags_backdoor", False)) for r in atk]
    nc_flag_cln = [bool(r["neural_cleanse"].get("flags_backdoor", False)) for r in cln]

    best = max(dets, key=lambda d: (summary[d]["tpr_at_fpr0.05"]["mean"]
                                    if np.isfinite(summary[d]["tpr_at_fpr0.05"]["mean"]) else -1))
    headline = (
        f"With all four per-sample detectors read off at a common 5% "
        f"false-positive rate on the same training rows, SPECTRE reached "
        f"TPR {summary['spectre']['tpr_at_fpr0.05']['mean']:.3f} "
        f"(SD {summary['spectre']['tpr_at_fpr0.05']['sd']:.3f}) against "
        f"{summary['spectral_signatures']['tpr_at_fpr0.05']['mean']:.3f} for Spectral "
        f"Signatures, {summary['activation_clustering']['tpr_at_fpr0.05']['mean']:.3f} for "
        f"Activation Clustering and {summary['strip']['tpr_at_fpr0.05']['mean']:.3f} for "
        f"STRIP, with Neural Cleanse flagging "
        f"{int(np.sum(nc_flag_atk))}/{len(nc_flag_atk)} attacked models against "
        f"{int(np.sum(nc_flag_cln))}/{len(nc_flag_cln)} clean models."
    )

    doc = {
        "exp_id": "EXP-4a", "git_sha": code_sha(), "completed_utc": utcnow(),
        "design": {
            "task": "identify flipped rows inside the poisoned target class of the "
                    "training split",
            "matched_fpr_levels": list(MATCHED_FPRS),
            "neural_cleanse": "model-level anomaly index; cannot produce per-sample "
                              "scores, so it is not on the matched-FPR axis",
            "clean_models": "scored against a random pseudo-poison mask over the same "
                            "cell, so every detection there is a false positive",
        },
        "per_detector": summary,
        "neural_cleanse_model_level": {
            "anomaly_index_attacked": agg(nc_atk),
            "anomaly_index_clean": agg(nc_cln),
            "flag_rate_attacked": f"{int(np.sum(nc_flag_atk))}/{len(nc_flag_atk)}",
            "flag_rate_clean": f"{int(np.sum(nc_flag_cln))}/{len(nc_flag_cln)}",
        },
        "best_detector_at_fpr0.05": best,
        "per_run": results,
        "headline_sentence": headline,
    }
    sfx = "_smoke" if args.smoke else ""
    write_json(OUT / f"spectre_summary{sfx}.json", doc)

    rows = []
    for det, row in summary.items():
        rows.append({"detector": det,
                     "score_auroc_mean": row["score_auroc"]["mean"],
                     "score_auroc_sd": row["score_auroc"]["sd"],
                     **{f"tpr_at_{f:.2f}": row[f"tpr_at_fpr{f:.2f}"]["mean"]
                        for f in MATCHED_FPRS},
                     **{f"null_tpr_at_{f:.2f}": row[f"null_tpr_at_fpr{f:.2f}"]["mean"]
                        for f in MATCHED_FPRS}})
    pd.DataFrame(rows).to_csv(OUT / f"spectre_summary{sfx}.csv", index=False)

    print("\n" + headline)
    print(f"[exp4a] -> {OUT / f'spectre_summary{sfx}.json'}")
    append_manifest({"exp_id": "EXP-4a", "git_sha": code_sha(),
                     "n_models": len(results), "smoke": args.smoke})


if __name__ == "__main__":
    main()
