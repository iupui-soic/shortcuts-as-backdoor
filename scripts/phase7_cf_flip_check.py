#!/usr/bin/env python3
"""Phase 7 §8.3 validity check — does the CycleGAN counterfactual actually FLIP race?

The CF demographic audit (cf_demographic_audit.py) reports a ~0 clean-vs-attacked
delta with the real CycleGAN. That null result is only a meaningful *evasion*
finding if the generator's race-flip is real; if CF(x) is effectively identity in
the eyes of a race decoder, the null is a confound (a too-weak generator), not
evidence the attack evades the audit.

This script settles it. Using the held-out Phase 1 MIMIC race detector
(DenseNet-121, test AUROC ~0.977) as an independent probe:

  baseline   AUROC( real WHITE  vs  real BLACK ),                 prob-of-black
  residual   AUROC( real WHITE  vs  CF(BLACK->WHITE) )            prob-of-black

If the flip works, CF(BLACK->WHITE) looks WHITE to the detector and `residual`
collapses toward 0.5. effectiveness = (baseline - residual) / (baseline - 0.5).
We report the symmetric WHITE->BLACK direction too, plus mean prob-of-black
shifts and a per-image flip rate.

Run:  PYTHONPATH=. python3 scripts/phase7_cf_flip_check.py --gpu 1 --n 500
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.defenses import common as C
from src.defenses.cf_demographic_audit import CycleGANGenerator

MATCHED = "data/manifests/mimic_cxr_matched.parquet"
RACE_DET = "results/phase1/phase1__mimic_race_detector__densenet121__seed42"
GEN_CKPT = "results/phase7/cf_cyclegan/ckpt/cyclegan_last.pt"
WHITE, BLACK = "WHITE", "BLACK_OR_AA"
OUT = "results/phase7/cf_flip_check.json"


def materialize(relpaths, cfg):
    df = pd.DataFrame({"relpath": list(relpaths), "target": 0})
    return C.materialize_images(df, cfg, num_workers=8)


@torch.no_grad()
def detector_prob(model, imgs, device, bs=64):
    """P(target=1) per image (single-logit head)."""
    out = []
    for i in range(0, len(imgs), bs):
        xb = imgs[i:i + bs].to(device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=device.type == "cuda"):
            p = torch.sigmoid(model(xb)[:, 0]).float()
        out.append(p.cpu().numpy())
    return np.concatenate(out) if out else np.array([])


@torch.no_grad()
def cf_translate(gen, imgs, from_demo, to_demo, device, bs=32):
    out = []
    for i in range(0, len(imgs), bs):
        out.append(gen(imgs[i:i + bs], from_demo, to_demo).cpu())
    return torch.cat(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--n", type=int, default=500, help="images per race group")
    ap.add_argument("--gen-ckpt", default=GEN_CKPT)
    ap.add_argument("--race-det", default=RACE_DET)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model, cfg = C.load_model(args.race_det, device)
    gen = CycleGANGenerator(args.gen_ckpt, device)
    print(f"[load] race detector + {gen.name} on {device}", flush=True)

    df = pd.read_parquet(MATCHED)
    test = df[df["official_split"] == "test"]
    white_paths = test[test["race_group"] == WHITE]["relpath"].tolist()[:args.n]
    black_paths = test[test["race_group"] == BLACK]["relpath"].tolist()[:args.n]
    print(f"[data] white={len(white_paths)} black={len(black_paths)}", flush=True)

    imgs_white = materialize(white_paths, cfg)
    imgs_black = materialize(black_paths, cfg)

    p_white_raw = detector_prob(model, imgs_white, device)
    p_black_raw = detector_prob(model, imgs_black, device)

    # Determine polarity: which class is target=1?  (separation is ~clean.)
    pos_is_black = float(p_black_raw.mean()) > float(p_white_raw.mean())
    to_pblack = (lambda p: p) if pos_is_black else (lambda p: 1.0 - p)
    print(f"[polarity] target=1 is {'BLACK_OR_AA' if pos_is_black else 'WHITE'} "
          f"(mean prob: black={p_black_raw.mean():.3f} white={p_white_raw.mean():.3f})",
          flush=True)

    pbr = to_pblack(p_black_raw)   # prob-of-black on real black
    pwr = to_pblack(p_white_raw)   # prob-of-black on real white

    # Translate and re-probe.
    cf_b2w = cf_translate(gen, imgs_black, BLACK, WHITE, device)   # G_B2A: black->white
    cf_w2b = cf_translate(gen, imgs_white, WHITE, BLACK, device)   # G_A2B: white->black
    p_b2w = to_pblack(detector_prob(model, cf_b2w, device))        # should DROP toward white
    p_w2b = to_pblack(detector_prob(model, cf_w2b, device))        # should RISE toward black

    y = np.r_[np.zeros(len(pwr)), np.ones(len(pbr))]               # 1 = black
    auroc_baseline = roc_auc_score(y, np.r_[pwr, pbr])
    # residual race signal after flipping black->white (vs untouched real white)
    auroc_resid_b2w = roc_auc_score(y, np.r_[pwr, p_b2w])
    # residual after flipping white->black (vs untouched real black), label 1=black on real
    auroc_resid_w2b = roc_auc_score(np.r_[np.ones(len(pbr)), np.zeros(len(p_w2b))],
                                    np.r_[pbr, p_w2b])

    def eff(resid):
        denom = auroc_baseline - 0.5
        return float((auroc_baseline - resid) / denom) if denom > 1e-9 else float("nan")

    gap = float(pbr.mean() - pwr.mean())
    res = {
        "generator": gen.name,
        "race_detector": str(Path(args.race_det).name),
        "n_per_group": {"white": len(white_paths), "black": len(black_paths)},
        "target1_is_black": bool(pos_is_black),
        "mean_prob_black": {
            "real_black": float(pbr.mean()),
            "cf_black_to_white": float(p_b2w.mean()),
            "real_white": float(pwr.mean()),
            "cf_white_to_black": float(p_w2b.mean()),
        },
        "real_between_group_gap": gap,
        "gap_closed_frac": {
            "black_to_white": float((pbr.mean() - p_b2w.mean()) / gap) if gap else float("nan"),
            "white_to_black": float((p_w2b.mean() - pwr.mean()) / gap) if gap else float("nan"),
        },
        # fraction of confidently-black reals (p>0.5) pushed across to white by CF
        "flip_rate_black_to_white": float(
            ((pbr > 0.5) & (p_b2w < 0.5)).sum() / max(1, int((pbr > 0.5).sum()))),
        "auroc_race": {
            "baseline_realW_vs_realB": float(auroc_baseline),
            "residual_realW_vs_CF(B->W)": float(auroc_resid_b2w),
            "residual_realB_vs_CF(W->B)": float(auroc_resid_w2b),
        },
        "flip_effectiveness": {           # 1.0 = race signal fully removed; 0.0 = identity
            "black_to_white": eff(auroc_resid_b2w),
            "white_to_black": eff(auroc_resid_w2b),
        },
    }
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps(res, indent=2), flush=True)
    print(f"[done] -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
