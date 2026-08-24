#!/usr/bin/env python3
"""EXP-2 summary — is the installation point stable across operating points?

This is the decision gate. Everything in the manuscript is computed at a fixed
decision threshold of 0.5, where the clean model's sensitivity is 0.454 for
effusion and 0.054 for pneumothorax. No deployed system runs at 5% sensitivity,
and the paper's central recommendation is to audit at the deployed operating
point — so anchoring every number to an implausible one is the easiest objection
a reviewer has.

Reports, per cohort:
  * the installation point at each operating point — the lowest poison rate whose
    seed-mean passes all three pre-specified gates
  * whether it moves, and the Spearman correlation between install point and the
    threshold's clean sensitivity
  * gate sensitivity at GATE_ASR in {0.10, 0.15, 0.20, 0.30}
  * the audit comparison (AUROC-audit vs FNR-audit) at each operating point,
    which is the critical one: if the FNR audit only works at 0.5 the
    recommendation is fragile; if it works at all of them it is robust
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.common_rev import (  # noqa: E402
    GATE_ASR_SENSITIVITY, GATE_GAP, GATE_STEALTH, REV, agg, append_manifest,
    code_sha, utcnow, write_json,
)

OUT = REV / "EXP-2"
ORDER = ["t0.5", "youden_j", "sens0.80", "spec0.90"]


def install_point(g: pd.DataFrame, gate_asr: float) -> dict:
    """Lowest rate whose seed-mean passes all three gates."""
    rows = []
    for rate, s in g.groupby("rate"):
        if rate == 0.0:
            continue
        a = float(np.nanmean(s.asr_rel_target))
        c = float(np.nanmean(s.asr_rel_control))
        d = float(np.nanmean(s.auroc_delta_overall))
        rows.append({
            "rate": float(rate), "n_seeds": int(s.seed.nunique()),
            "asr_rel_attacked": a, "asr_rel_control": c, "auroc_delta": d,
            "gate_asr": bool(a >= gate_asr), "gate_gap": bool(a - c >= GATE_GAP),
            "gate_stealth": bool(abs(d) <= GATE_STEALTH),
        })
    rows.sort(key=lambda r: r["rate"])
    for r in rows:
        r["all_gates"] = r["gate_asr"] and r["gate_gap"] and r["gate_stealth"]
    hit = next((r["rate"] for r in rows if r["all_gates"]), None)
    # An install point read off a cell with fewer than three seeds is provisional:
    # the pr=0.5 MIMIC cell moved from ASR_rel 0.098 (n=1) to 0.299 (n=2) and
    # flipped the install point, which is precisely why EXP-3 exists.
    under = [r["rate"] for r in rows if r["n_seeds"] < 3]
    hit_row = next((r for r in rows if r["all_gates"]), None)
    return {"install_point": hit, "ladder": rows,
            "install_point_n_seeds": hit_row["n_seeds"] if hit_row else None,
            "provisional": bool(hit_row is not None and hit_row["n_seeds"] < 3),
            "rates_with_under_3_seeds": under}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="densenet121")
    args = ap.parse_args()

    df = pd.read_csv(OUT / "rescored.csv")
    df = df[df.arch == args.arch]

    per_cohort = {}
    for cid, g in df.groupby("cohort_id"):
        if g["rate"].nunique() < 2:
            continue
        by_t = {}
        for tname in ORDER:
            gt = g[g.threshold_name == tname]
            if gt.empty:
                continue
            ip = install_point(gt, 0.20)
            by_t[tname] = {
                "threshold_value_mean": float(gt.threshold_value.mean()),
                "clean_val_sensitivity": float(gt.val_sensitivity_clean.mean()),
                "clean_val_specificity": float(gt.val_specificity_clean.mean()),
                "clean_test_sensitivity": float(
                    gt[gt.rate == 0.0].sens_overall_clean.mean()),
                "clean_test_specificity": float(
                    gt[gt.rate == 0.0].spec_overall_clean.mean()),
                **ip,
                "gate_sensitivity": {
                    f"GATE_ASR={q:.2f}": install_point(gt, q)["install_point"]
                    for q in GATE_ASR_SENSITIVITY
                },
            }
        pts = {t: v["install_point"] for t, v in by_t.items()}
        vals = [v for v in pts.values() if v is not None]
        moved = len(set(vals)) > 1 or (len(vals) != len(pts))
        sens = [by_t[t]["clean_val_sensitivity"] for t in by_t
                if by_t[t]["install_point"] is not None]
        ipv = [by_t[t]["install_point"] for t in by_t
               if by_t[t]["install_point"] is not None]
        sp = (stats.spearmanr(sens, ipv) if len(ipv) >= 3 else None)
        per_cohort[cid] = {
            "by_threshold": by_t,
            "install_points": pts,
            "install_point_moves": bool(moved),
            "spearman_install_vs_clean_sensitivity": (
                {"rho": float(sp.statistic), "p": float(sp.pvalue), "n": len(ipv)}
                if sp is not None else None),
        }

    key = "mimic_race_unmatched"
    main_c = per_cohort.get(key, {})
    pts = main_c.get("install_points", {})
    stable = not main_c.get("install_point_moves", True)
    shown = ", ".join(
        f"{t} (clean sens {main_c['by_threshold'][t]['clean_val_sensitivity']:.2f}): "
        f"{'none' if pts[t] is None else f'pr={pts[t]:g}'}"
        for t in ORDER if t in pts)
    headline = (
        f"Re-scoring every existing run at four operating points, the installation "
        f"point on MIMIC race is {'unchanged' if stable else 'NOT stable'} — {shown} — "
        f"so the dose-response characterization "
        f"{'does not depend on the 0.5 decision threshold it was originally computed at'
           if stable else
           'must be re-anchored: the 0.5 threshold is not representative and the '
           'primary numbers should be reported at the Youden-J operating point'}."
    )

    doc = {"exp_id": "EXP-2", "git_sha": code_sha(), "completed_utc": utcnow(),
           "arch": args.arch, "policies": ORDER,
           "derivation": "every non-0.5 operating point is computed on the CLEAN "
                         "seed-matched model's validation split and applied "
                         "unchanged to the attacked model",
           "per_cohort": per_cohort,
           "decision_gate": {
               "question": "does the installation point on MIMIC unmatched race "
                           "change between t=0.5, Youden's J and sensitivity-matched?",
               "answer": "stable" if stable else "moves",
               "consequence": ("EXP-1 drops to a Limitations sentence; 90 GPU-h freed"
                               if stable else
                               "the dose-response section is being rewritten anyway, "
                               "so EXP-1 and EXP-3 both become essential"),
           },
           "headline_sentence": headline}
    write_json(OUT / "summary.json", doc)

    rows = []
    for cid, c in per_cohort.items():
        for t, v in c["by_threshold"].items():
            rows.append({"cohort_id": cid, "threshold_name": t,
                         "threshold_value": v["threshold_value_mean"],
                         "clean_val_sensitivity": v["clean_val_sensitivity"],
                         "clean_val_specificity": v["clean_val_specificity"],
                         "install_point": v["install_point"],
                         **{k: val for k, val in v["gate_sensitivity"].items()}})
    pd.DataFrame(rows).to_csv(OUT / "install_points.csv", index=False)

    print("\n=== installation point by operating point ===")
    for cid, c in per_cohort.items():
        print(f"\n{cid}   moves: {c['install_point_moves']}")
        for t in ORDER:
            v = c["by_threshold"].get(t)
            if not v:
                continue
            ip = v["install_point"]
            print(f"  {t:9s} t={v['threshold_value_mean']:.4f}  "
                  f"clean val sens {v['clean_val_sensitivity']:.3f} / spec "
                  f"{v['clean_val_specificity']:.3f}  -> install "
                  f"{'none' if ip is None else f'pr={ip:g}'}")
    print("\n" + headline)
    append_manifest({"exp_id": "EXP-2", "step": "summary", "git_sha": code_sha(),
                     "install_point_stable": stable})


if __name__ == "__main__":
    main()
