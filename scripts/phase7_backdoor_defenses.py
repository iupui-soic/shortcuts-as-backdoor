#!/usr/bin/env python3
"""Phase 7 §8.1 — run the 4 backdoor-detection defenses on attacked checkpoints.

Post-hoc: loads each attacked model (default pr0.75 threshold regime), uses the
poison_log.json ground truth, and reports detection metrics per defense:

  * activation_clustering / spectral_signatures: TPR on poisoned samples, FPR on
    clean samples (run on the poisoned target class of the TRAIN split).
  * strip: entropy-separation AUROC of attacked-subgroup vs control positives.
  * neural_cleanse: MAD anomaly index over reverse-engineered per-label triggers.

Usage:
  PYTHONPATH=. python3 scripts/phase7_backdoor_defenses.py            # full
  PYTHONPATH=. python3 scripts/phase7_backdoor_defenses.py --smoke    # tiny/CPU
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.defenses import common as C
from src.defenses.backdoor import (
    activation_clustering,
    spectral_signatures,
    strip_entropy,
    strip_detection,
    neural_cleanse,
)

OUT = C.REPO / "results" / "phase7"


def materialize_images(df, cfg, num_workers: int) -> torch.Tensor:
    return C.materialize_images(df, cfg, num_workers=num_workers)


def run_one(model_entry: dict, args, device: torch.device) -> dict:
    d = model_entry["dir"]
    model, cfg = C.load_model(d, device)
    spec = C.attack_spec(cfg)
    plog = C.load_poison_log(d)
    mani = C.load_manifest(cfg)
    train = mani[mani["split"] == "train"].reset_index(drop=True)
    test = mani[mani["split"] == "test"].reset_index(drop=True)
    rng = np.random.default_rng(model_entry["seed"])
    nw = args.num_workers
    tl = spec.target_label
    result = {"arch": model_entry["arch"], "seed": model_entry["seed"],
              "rate": model_entry["rate"], "dir": Path(d).name,
              "n_poisoned_train": int(plog["n_poisoned"])}

    # ---- Activation Clustering + Spectral Signatures (feature-space, TRAIN) ----
    pdf = C.apply_poison_labels(train, plog)            # labels the model trained on
    pois_mask_full = C.poisoned_mask(train, plog)
    in_class = (pdf[tl].to_numpy() == spec.flip_to)     # poisoned target class
    pois_idx = np.flatnonzero(in_class & pois_mask_full)
    clean_idx = np.flatnonzero(in_class & ~pois_mask_full)
    # Keep all poisoned when they fit in half the budget (so TPR is over the full
    # poison set); otherwise subsample poisoned so clean samples are still present
    # (FPR would be undefined with an all-poison subset).
    keep_pois = min(pois_idx.size, max(1, args.max_samples // 2))
    if pois_idx.size > keep_pois:
        pois_idx = rng.choice(pois_idx, size=keep_pois, replace=False)
    n_clean_keep = max(1, args.max_samples - pois_idx.size)
    if clean_idx.size > n_clean_keep:
        clean_idx = rng.choice(clean_idx, size=n_clean_keep, replace=False)
    subset_idx = np.sort(np.concatenate([pois_idx, clean_idx]))
    subset = train.iloc[subset_idx].reset_index(drop=True)
    sub_pois_mask = C.poisoned_mask(subset, plog)
    print(f"  [AC/Spectral] target-class subset n={len(subset)} "
          f"poisoned={int(sub_pois_mask.sum())} (capped at max_samples={args.max_samples})")
    feat_loader = C.make_eval_loader(subset, cfg, batch_size=64, num_workers=nw)
    fe = C.extract(model, feat_loader, device, want_features=True)
    feats = fe["features"]
    exp_frac = float(sub_pois_mask.sum() / max(len(subset), 1))
    result["activation_clustering"] = activation_clustering(feats, sub_pois_mask, seed=model_entry["seed"])
    result["spectral_signatures"] = spectral_signatures(feats, sub_pois_mask, exp_frac)

    # ---- STRIP (input-space, TEST positives) ----
    demo = test[spec.demographic_col].astype(str)
    control = spec.control_demographic(demo.unique())
    suspect_df = test[(demo == spec.target_demographic) & (test[tl] == 1)]
    ref_df = test[(demo == control) & (test[tl] == 1)]
    suspect_df = suspect_df.iloc[:args.strip_per_group]
    ref_df = ref_df.iloc[:args.strip_per_group]
    overlay_df = test.sample(n=min(args.strip_overlay_pool, len(test)),
                             random_state=model_entry["seed"])
    suspect_imgs = materialize_images(suspect_df, cfg, nw)
    ref_imgs = materialize_images(ref_df, cfg, nw)
    overlay_imgs = materialize_images(overlay_df, cfg, nw)
    if suspect_imgs.numel() and ref_imgs.numel() and overlay_imgs.numel():
        ent_s = strip_entropy(model, device, suspect_imgs, overlay_imgs, spec.target_idx,
                              n_overlays=args.strip_overlays, seed=model_entry["seed"])
        ent_r = strip_entropy(model, device, ref_imgs, overlay_imgs, spec.target_idx,
                              n_overlays=args.strip_overlays, seed=model_entry["seed"] + 1)
        result["strip"] = strip_detection(ent_s, ent_r)
    else:
        result["strip"] = {"defense": "strip", "status": "insufficient_samples"}

    # ---- Neural Cleanse (input-space, reverse-engineer per-label triggers) ----
    nc_df = test.sample(n=min(args.nc_batch, len(test)), random_state=model_entry["seed"])
    nc_imgs = materialize_images(nc_df, cfg, nw)
    result["neural_cleanse"] = neural_cleanse(
        model, device, nc_imgs, n_labels=len(spec.target_labels),
        steps=args.nc_steps, seed=model_entry["seed"])

    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", default="0.75")
    ap.add_argument("--max-samples", type=int, default=20000, dest="max_samples")
    ap.add_argument("--strip-overlays", type=int, default=30, dest="strip_overlays")
    ap.add_argument("--strip-per-group", type=int, default=300, dest="strip_per_group")
    ap.add_argument("--strip-overlay-pool", type=int, default=100, dest="strip_overlay_pool")
    ap.add_argument("--nc-steps", type=int, default=300, dest="nc_steps")
    ap.add_argument("--nc-batch", type=int, default=128, dest="nc_batch")
    ap.add_argument("--num-workers", type=int, default=8, dest="num_workers")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--arch", default=None, help="filter to one arch")
    ap.add_argument("--limit", type=int, default=None, help="first N attacked models")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.max_samples = 200
        args.strip_overlays = 4
        args.strip_per_group = 16
        args.strip_overlay_pool = 24
        args.nc_steps = 8
        args.nc_batch = 16
        args.num_workers = 0
        if args.limit is None:
            args.limit = 1
        if args.device == "auto":
            args.device = "cpu"

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto"
        else args.device)

    OUT.mkdir(parents=True, exist_ok=True)
    ms = C.default_model_set(args.rate)
    attacked = ms["attacked"]
    if args.arch:
        attacked = [m for m in attacked if m["arch"] == args.arch]
    if args.limit:
        attacked = attacked[:args.limit]

    print(f"device={device}  models={len(attacked)}  smoke={args.smoke}")
    results = []
    for m in attacked:
        print(f"[{m['arch']} seed{m['seed']} pr{m['rate']}]")
        results.append(run_one(m, args, device))

    suffix = "_smoke" if args.smoke else ""
    doc = {"rate": args.rate, "smoke": args.smoke, "per_run": results}
    (OUT / f"backdoor_defenses{suffix}.json").write_text(json.dumps(doc, indent=2, default=str))

    # markdown summary (mean over runs per arch)
    lines = ["# Phase 7 §8.1 — Backdoor-detection defenses", "",
             f"Attacked operating point pr{args.rate}. Predicted: all fail or partial.", "",
             "| arch | AC TPR | AC FPR | Spectral TPR | Spectral FPR | STRIP AUROC | NC anomaly | NC flags |",
             "|---|---|---|---|---|---|---|---|"]
    archs = sorted({r["arch"] for r in results})
    for arch in archs:
        sub = [r for r in results if r["arch"] == arch]
        def _m(path):
            vals = []
            for r in sub:
                cur = r
                ok = True
                for k in path:
                    if isinstance(cur, dict) and k in cur:
                        cur = cur[k]
                    else:
                        ok = False
                        break
                if ok and isinstance(cur, (int, float)) and not (isinstance(cur, float) and np.isnan(cur)):
                    vals.append(cur)
            return float(np.mean(vals)) if vals else float("nan")
        lines.append(
            f"| {arch} | {_m(['activation_clustering','tpr_poisoned']):.2f} | "
            f"{_m(['activation_clustering','fpr_clean']):.2f} | "
            f"{_m(['spectral_signatures','tpr_poisoned']):.2f} | "
            f"{_m(['spectral_signatures','fpr_clean']):.2f} | "
            f"{_m(['strip','detection_auroc']):.2f} | "
            f"{_m(['neural_cleanse','anomaly_index']):.2f} | "
            f"{_m(['neural_cleanse','flags_backdoor']):.0%} |")
    lines += ["", "AC/Spectral TPR is recall of poisoned train samples; an ideal "
              "detector has high TPR at low FPR. STRIP AUROC ~0.5 and NC anomaly "
              "<2 both indicate the defense does not detect the trigger-less attack."]
    (OUT / f"backdoor_defenses{suffix}.md").write_text("\n".join(lines))
    print(f"\nwrote {OUT / f'backdoor_defenses{suffix}.json'} and .md")


if __name__ == "__main__":
    main()
