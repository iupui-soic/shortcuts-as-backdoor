#!/usr/bin/env python3
"""Phase 7 §8.3 — counterfactual demographic audit (harness; generator deferred).

Computes CF-inconsistency |f(x) - f(CF(x))| on attacked-subgroup test positives
for clean vs attacked models and compares the distributions. The counterfactual
generator is pluggable; this run uses IdentityGenerator (placeholder), so the
numbers are ~0 by construction and the JSON records `generator: identity` — the
harness is validated and ready for a real CXR counterfactual model.

Usage:
  PYTHONPATH=. python3 scripts/phase7_cf_audit.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.defenses import common as C
from src.defenses.cf_demographic_audit import (CycleGANGenerator, IdentityGenerator,
                                               audit, cf_inconsistency)

DEFAULT_GEN_CKPT = "results/phase7/cf_cyclegan/ckpt/cyclegan_last.pt"

OUT = C.REPO / "results" / "phase7"


def run_arch(arch: str, seed: int, rate: str, generator, args, device) -> dict:
    ms = C.default_model_set(rate)
    clean = next(m for m in ms["clean"] if m["arch"] == arch and m["seed"] == seed)
    attacked = next(m for m in ms["attacked"] if m["arch"] == arch and m["seed"] == seed)
    clean_model, cfg = C.load_model(clean["dir"], device)
    atk_model, _ = C.load_model(attacked["dir"], device)
    spec = C.attack_spec(cfg)

    mani = C.load_manifest(cfg)
    test = mani[mani["split"] == "test"]
    demo = test[spec.demographic_col].astype(str)
    control = spec.control_demographic(demo.unique())
    suspect = test[(demo == spec.target_demographic) & (test[spec.target_label] == 1)]
    suspect = suspect.iloc[:args.max_inputs]
    imgs = C.materialize_images(suspect, cfg, num_workers=args.num_workers)

    common_kw = dict(generator=generator, target_idx=spec.target_idx,
                     from_demo=spec.target_demographic, to_demo=control)
    inc_clean = cf_inconsistency(clean_model, device, imgs, **common_kw)
    inc_atk = cf_inconsistency(atk_model, device, imgs, **common_kw)
    res = audit(inc_clean, inc_atk, generator.name)
    res.update(arch=arch, seed=seed, rate=rate, subgroup=spec.target_demographic)
    # Per-image distributions so Fig 10 can draw histograms once a real generator is used.
    res["inconsistency_clean"] = [round(float(v), 6) for v in inc_clean]
    res["inconsistency_attacked"] = [round(float(v), 6) for v in inc_atk]
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", default="0.75")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--archs", default="densenet121,vit_base_patch16_224")
    ap.add_argument("--max-inputs", type=int, default=500, dest="max_inputs")
    ap.add_argument("--num-workers", type=int, default=8, dest="num_workers")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--generator", default="identity", choices=["identity", "cyclegan"],
                    help="counterfactual generator: identity placeholder or trained CycleGAN")
    ap.add_argument("--gen-ckpt", dest="gen_ckpt", default=DEFAULT_GEN_CKPT,
                    help="CycleGAN checkpoint (used when --generator cyclegan)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.archs = "densenet121"
        args.max_inputs = 8
        args.num_workers = 0
        if args.device == "auto":
            args.device = "cpu"

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto"
        else args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.generator == "cyclegan":
        ckpt = Path(args.gen_ckpt)
        if not ckpt.is_absolute():
            ckpt = C.REPO / ckpt
        if not ckpt.exists():
            raise SystemExit(f"[cf-audit] CycleGAN checkpoint not found: {ckpt}")
        generator = CycleGANGenerator(str(ckpt), device)
        print(f"[cf-audit] real generator: {generator.name}")
    else:
        generator = IdentityGenerator()

    results = []
    for arch in args.archs.split(","):
        arch = arch.strip()
        print(f"[cf-audit] {arch} (generator={generator.name})")
        res = run_arch(arch, args.seed, args.rate, generator, args, device)
        results.append(res)
        print(f"  delta={res['delta']} flags={res['flags_attack']}")

    suffix = "_smoke" if args.smoke else ""
    (OUT / f"cf_audit{suffix}.json").write_text(json.dumps(results, indent=2, default=str))
    is_identity = generator.name.startswith("identity")
    header = (f"Generator: **{generator.name}** (real CXR counterfactual model deferred)."
              if is_identity else
              f"Generator: **{generator.name}** — real demographic counterfactual "
              f"(CycleGAN, matched MIMIC race cohort).")
    lines = ["# Phase 7 §8.3 — Counterfactual demographic audit", "", header,
             "", "| arch | mean CF-incons. clean | attacked | delta | flags |",
             "|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['arch']} | {r['mean_cf_inconsistency_clean']:.4f} | "
                     f"{r['mean_cf_inconsistency_attacked']:.4f} | {r['delta']:.4f} | "
                     f"{r['flags_attack']} |")
    lines += ["", ("_With the identity placeholder, CF-inconsistency is ~0 by "
                   "construction; numbers become meaningful once a real demographic "
                   "counterfactual generator is plugged in._") if is_identity else
              ("_Real CycleGAN counterfactual: a positive attacked−clean delta means the "
               "attacked model's prediction moves more when the demographic is flipped, "
               "i.e. it has tied the target label to the demographic channel. The audit "
               "remains partial by construction — a CXR demographic counterfactual is "
               "itself an instance of the encoded-race phenomenon._")]
    (OUT / f"cf_audit{suffix}.md").write_text("\n".join(lines))
    print(f"\nwrote {OUT / f'cf_audit{suffix}.json'} and .md")


if __name__ == "__main__":
    main()
