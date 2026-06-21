#!/usr/bin/env python3
"""Phase 7 §8.2 baseline: subgroup fairness audit as a backdoor detector.

Post-hoc, no GPU: reads each attacked model's predictions.parquet and its
clean baseline, and reports whether a standard subgroup-AUROC audit detects the
attack (predicted: no) vs an FNR-at-threshold audit (predicted: yes).

Usage:
  PYTHONPATH=. python3 scripts/phase7_fairness_audit.py [--rate 0.75] [--threshold 0.5]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.defenses import common as C
from src.defenses.fairness_audit import audit_predictions

OUT = C.REPO / "results" / "phase7"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", default="0.75", help="attacked operating point")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    ms = C.default_model_set(args.rate)
    clean_by_key = {(m["arch"], m["seed"]): m for m in ms["clean"]}

    rows = []
    for atk in ms["attacked"]:
        key = (atk["arch"], atk["seed"])
        cln = clean_by_key.get(key)
        if cln is None:
            print(f"[skip] no clean baseline for {key}")
            continue
        cfg = C.load_cfg(atk["dir"])
        res = audit_predictions(atk["dir"], cln["dir"], cfg, threshold=args.threshold)
        res.update(arch=atk["arch"], seed=atk["seed"], rate=atk["rate"],
                   attacked_dir=Path(atk["dir"]).name)
        rows.append(res)
        print(f"[{atk['arch']} seed{atk['seed']}] "
              f"AUROC-gap {res['attacked']['auroc_gap_target']:.3f} "
              f"(audit flags={res['auroc_audit_flags_attack']}) | "
              f"FNR-gap {res['attacked']['fnr_gap_target']:.3f} "
              f"(audit flags={res['fnr_audit_flags_attack']}) | "
              f"ASR_rel {res['asr_relative_attacked_subgroup']:.3f}")

    # aggregate per arch (mean across seeds)
    agg = {}
    for arch in sorted({r["arch"] for r in rows}):
        sub = [r for r in rows if r["arch"] == arch]
        agg[arch] = {
            "n_seeds": len(sub),
            "auroc_gap_mean": float(np.mean([r["attacked"]["auroc_gap_target"] for r in sub])),
            "fnr_gap_mean": float(np.mean([r["attacked"]["fnr_gap_target"] for r in sub])),
            "asr_relative_mean": float(np.mean([r["asr_relative_attacked_subgroup"] for r in sub])),
            "auroc_audit_detection_rate": float(np.mean([r["auroc_audit_flags_attack"] for r in sub])),
            "fnr_audit_detection_rate": float(np.mean([r["fnr_audit_flags_attack"] for r in sub])),
        }

    doc = {"rate": args.rate, "threshold": args.threshold, "per_run": rows, "per_arch": agg}
    (OUT / "fairness_audit.json").write_text(json.dumps(doc, indent=2, default=str))

    # markdown
    lines = [
        "# Phase 7 §8.2 — Subgroup fairness audit as backdoor detector",
        "",
        f"Attacked operating point: pr{args.rate} vs clean pr0.0. FNR threshold {args.threshold}.",
        "",
        "| arch | seeds | AUROC-gap | AUROC audit detects | FNR-gap | FNR audit detects | ASR_rel |",
        "|---|---|---|---|---|---|---|",
    ]
    for arch, a in agg.items():
        lines.append(
            f"| {arch} | {a['n_seeds']} | {a['auroc_gap_mean']:.3f} | "
            f"{a['auroc_audit_detection_rate']:.0%} | {a['fnr_gap_mean']:.3f} | "
            f"{a['fnr_audit_detection_rate']:.0%} | {a['asr_relative_mean']:.3f} |"
        )
    lines += [
        "",
        "**Diagnostic.** The rank-based subgroup-AUROC audit is blind to a "
        "threshold-suppression label-flip backdoor (ranking within subgroup is "
        "preserved), so it does not detect the attack. An FNR-at-threshold audit "
        "on the same predictions does. A fairness audit is therefore a valid "
        "detector only if it is evaluated at the deployed operating point.",
    ]
    (OUT / "fairness_audit.md").write_text("\n".join(lines))
    print(f"\nwrote {OUT/'fairness_audit.json'} and .md")


if __name__ == "__main__":
    main()
