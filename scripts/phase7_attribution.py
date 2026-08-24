#!/usr/bin/env python3
"""Phase 7 §8.4 — spatial attribution: GradCAM clean vs attacked on ChestX-Det10.

For each Effusion-annotated ChestX-Det10 case, compute GradCAM for the
pleural_effusion logit on the clean and the attacked model, and measure
localization IoU vs the radiologist bbox and extra-thoracic CAM fraction.
Stratify by *predicted* race (Phase 1 race detector), since NIH has no race label.

Usage:
  PYTHONPATH=. python3 scripts/phase7_attribution.py            # full
  PYTHONPATH=. python3 scripts/phase7_attribution.py --smoke    # tiny/CPU
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.defenses import common as C
from src.defenses import attribution as A

OUT = C.REPO / "results" / "phase7" / "attribution"


def _stats(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return {"mean": float(np.mean(vals)) if vals else float("nan"), "n": len(vals)}


def run_arch(arch: str, seed: int, rate: str, cases: list[dict],
             race_model, device: torch.device, save_figs: int) -> dict:
    ms = C.default_model_set(rate)
    clean = next(m for m in ms["clean"] if m["arch"] == arch and m["seed"] == seed)
    attacked = next(m for m in ms["attacked"] if m["arch"] == arch and m["seed"] == seed)
    clean_model, cfg = C.load_model(clean["dir"], device, eval_mode=True)
    atk_model, _ = C.load_model(attacked["dir"], device, eval_mode=True)
    spec = C.attack_spec(cfg)
    isz = spec.image_size
    ti = spec.target_idx

    cam_clean = A.GradCAM(clean_model, A.gradcam_target_layer(clean_model, arch))
    cam_atk = A.GradCAM(atk_model, A.gradcam_target_layer(atk_model, arch))

    per_case = []
    figs_dir = OUT / "figs"
    for ci, rec in enumerate(cases):
        x, w0, h0 = A.load_image_tensor(rec["path"], isz)
        x = x.to(device)
        bbox_mask = A.boxes_to_mask(rec["boxes"], w0, h0, isz)
        if bbox_mask.sum() == 0:
            continue
        cc = cam_clean(x, ti, isz)
        ca = cam_atk(x, ti, isz)
        m_clean = A.cam_localization_metrics(cc, bbox_mask)
        m_atk = A.cam_localization_metrics(ca, bbox_mask)
        black_prob = (A.predict_race_black_prob(race_model, x, device)
                      if race_model is not None else None)
        per_case.append({
            "file_name": rec["file_name"],
            "pred_black_prob": black_prob,
            "pred_race": (None if black_prob is None
                          else (spec.target_demographic if black_prob > 0.5 else "WHITE")),
            "clean": m_clean, "attacked": m_atk,
        })
        if save_figs and ci < save_figs:
            _save_overlay(figs_dir, arch, rec, cc, ca, bbox_mask, isz)

    cam_clean.remove()
    cam_atk.remove()

    def agg(group=None):
        sel = per_case if group is None else [p for p in per_case if p["pred_race"] == group]
        return {
            "n": len(sel),
            "clean_iou": _stats([p["clean"]["iou_top"] for p in sel]),
            "attacked_iou": _stats([p["attacked"]["iou_top"] for p in sel]),
            "clean_extra_thoracic": _stats([p["clean"]["extra_thoracic_frac"] for p in sel]),
            "attacked_extra_thoracic": _stats([p["attacked"]["extra_thoracic_frac"] for p in sel]),
        }

    out = {"arch": arch, "seed": seed, "rate": rate, "n_cases": len(per_case),
           "overall": agg()}
    if race_model is not None:
        out["pred_BLACK_OR_AA"] = agg(spec.target_demographic)
        out["pred_WHITE"] = agg("WHITE")
    out["per_case"] = per_case
    return out


def _save_overlay(figs_dir, arch, rec, cc, ca, bbox_mask, isz):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # EXP-9: npj figure compliance (Arial/Helvetica, >=300 dpi, RGB on
        # white, no rainbow colormaps, colour-blind-safe categorical cycle).
        from scripts.revision.npj_style import apply as _npj_apply, panel_labels as _panel_labels
        _npj_apply()
        from matplotlib.patches import Rectangle
        figs_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        for ax, cam, title in [(axes[0], cc, "clean"), (axes[1], ca, "attacked")]:
            ax.imshow(cam, cmap="viridis")
            ys, xs = np.where(bbox_mask)
            if xs.size:
                ax.add_patch(Rectangle((xs.min(), ys.min()), xs.max() - xs.min(),
                                       ys.max() - ys.min(), fill=False,
                                       edgecolor="white", lw=2))
            ax.set_title(f"{arch} {title}")
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(figs_dir / f"{arch}__{rec['file_name']}.png", dpi=300)
        plt.close(fig)
    except Exception as e:
        print(f"  [figs] skipped ({e})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", default="0.75")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--archs", default="densenet121,vit_base_patch16_224")
    ap.add_argument("--max-cases", type=int, default=200, dest="max_cases")
    ap.add_argument("--race-detector",
                    default="results/phase1/phase1__mimic_race_detector__densenet121__seed42",
                    dest="race_detector")
    ap.add_argument("--no-race", action="store_true", dest="no_race")
    ap.add_argument("--save-figs", type=int, default=6, dest="save_figs")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.archs = "densenet121"
        args.max_cases = 4
        args.save_figs = 0
        if args.device == "auto":
            args.device = "cpu"

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto"
        else args.device)
    OUT.mkdir(parents=True, exist_ok=True)

    cases = A.load_chestxdet10("test", A.DEFAULT_CATEGORY, limit=args.max_cases)
    print(f"device={device}  Effusion cases={len(cases)}  archs={args.archs}")

    race_model = None
    if not args.no_race:
        rd = C.REPO / args.race_detector
        if (rd / "best.pt").exists():
            race_model, _ = C.load_model(rd, device, eval_mode=True)
            print(f"race detector: {rd.name}")
        else:
            print(f"[warn] race detector not found at {rd}; skipping race stratification")

    results = []
    for arch in args.archs.split(","):
        arch = arch.strip()
        print(f"[attribution] {arch}")
        try:
            results.append(run_arch(arch, args.seed, args.rate, cases, race_model,
                                    device, args.save_figs))
        except Exception as e:
            print(f"  [error] {arch}: {e}")
            results.append({"arch": arch, "error": str(e)})

    suffix = "_smoke" if args.smoke else ""
    (OUT / f"attribution{suffix}.json").write_text(json.dumps(results, indent=2, default=str))

    lines = ["# Phase 7 §8.4 — Spatial attribution (GradCAM vs ChestX-Det10 Effusion bbox)",
             "",
             "Cross-cohort (MIMIC-trained model on NIH-derived ChestX-Det10); race is "
             "*predicted* (Phase 1 detector). Extra-thoracic uses bbox-complement proxy.",
             "",
             "| arch | n | clean IoU | attacked IoU | clean extra-thoracic | attacked extra-thoracic |",
             "|---|---|---|---|---|---|"]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['arch']} | — | ERROR: {r['error']} | | | |")
            continue
        o = r["overall"]
        lines.append(
            f"| {r['arch']} | {o['n']} | {o['clean_iou']['mean']:.3f} | "
            f"{o['attacked_iou']['mean']:.3f} | {o['clean_extra_thoracic']['mean']:.3f} | "
            f"{o['attacked_extra_thoracic']['mean']:.3f} |")
    lines += ["", "**Hypothesis check.** Lower attacked IoU and/or higher attacked "
              "extra-thoracic fraction (especially on predicted-BLACK_OR_AA cases) "
              "supports interpretability-based detection of the attacked behavior."]
    (OUT / f"attribution{suffix}.md").write_text("\n".join(lines))
    print(f"\nwrote {OUT / f'attribution{suffix}.json'} and .md")


if __name__ == "__main__":
    main()
