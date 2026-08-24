#!/usr/bin/env python3
"""EXP-2 gate attribution — decompose every install point and every "none".

An install point of "none" is ambiguous in a way that inverts the interpretation:
a cell can fail because the attack never installed (ASR gate), because it
installed on the control subgroup too (gap gate), or because it installed and
then became DETECTABLE (stealth gate). The first is "the attack does not work
here"; the third is "the attack works here and is caught" — opposite claims.

For every (cohort, threshold, poison rate) cell this reports each gate's verdict,
its observed value, its margin to the bar (signed: positive = passing by that
much), and — where all gates fail to yield an install point — which gate is
binding, i.e. the one that would have to move for the cell to pass.

Also carries the clean model's validation sensitivity at each operating point,
which is the column that makes the NIH anomaly legible: an install point read at
a threshold where the clean model catches 39% of positives is not comparable to
one read where it catches 80%.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.common_rev import (  # noqa: E402
    GATE_ASR, GATE_GAP, GATE_STEALTH, REV, append_manifest, code_sha, utcnow,
    write_json,
)

OUT = REV / "EXP-2"
ORDER = ["t0.5", "youden_j", "sens0.80", "spec0.90"]


def attribute(g: pd.DataFrame) -> list[dict]:
    rows = []
    for rate, s in g.groupby("rate"):
        if rate == 0.0:
            continue
        asr = float(np.nanmean(s.asr_rel_target))
        ctl = float(np.nanmean(s.asr_rel_control))
        dau = float(np.nanmean(s.auroc_delta_overall))
        gap = asr - ctl
        gates = {
            "asr":     {"value": asr, "bar": GATE_ASR,     "pass": bool(asr >= GATE_ASR),
                        "margin": asr - GATE_ASR},
            "gap":     {"value": gap, "bar": GATE_GAP,     "pass": bool(gap >= GATE_GAP),
                        "margin": gap - GATE_GAP},
            "stealth": {"value": dau, "bar": GATE_STEALTH, "pass": bool(abs(dau) <= GATE_STEALTH),
                        "margin": GATE_STEALTH - abs(dau)},
        }
        failed = [k for k, v in gates.items() if not v["pass"]]
        # the binding gate is the failing one with the worst (most negative) margin
        binding = min(failed, key=lambda k: gates[k]["margin"]) if failed else None
        rows.append({
            "rate": float(rate),
            "n_seeds": int(s.seed.nunique()),
            "asr_rel_attacked": asr, "asr_rel_control": ctl,
            "gap": gap, "auroc_delta_overall": dau,
            "gate_asr": gates["asr"]["pass"],
            "gate_gap": gates["gap"]["pass"],
            "gate_stealth": gates["stealth"]["pass"],
            "margin_asr": gates["asr"]["margin"],
            "margin_gap": gates["gap"]["margin"],
            "margin_stealth": gates["stealth"]["margin"],
            "all_gates": not failed,
            "failed_gates": failed,
            "binding_gate": binding,
        })
    return sorted(rows, key=lambda r: r["rate"])


def classify_none(ladder: list[dict]) -> dict:
    """Why is there no install point? The distinction that matters is whether the
    attack never reached the ASR bar, or reached it and broke stealth."""
    reached_asr = [r for r in ladder if r["gate_asr"]]
    if not reached_asr:
        best = max(ladder, key=lambda r: r["asr_rel_attacked"]) if ladder else None
        return {
            "reason": "never_installed",
            "detail": "no poison rate reached the ASR bar; the attack does not "
                      "install on this cohort at this operating point",
            "max_asr_rel": best["asr_rel_attacked"] if best else float("nan"),
            "max_asr_at_rate": best["rate"] if best else None,
        }
    broke_stealth = [r for r in reached_asr if not r["gate_stealth"]]
    broke_gap = [r for r in reached_asr if not r["gate_gap"]]
    if broke_stealth and len(broke_stealth) == len(reached_asr):
        worst = min(broke_stealth, key=lambda r: r["margin_stealth"])
        return {
            "reason": "installed_but_detectable",
            "detail": "every rate that cleared the ASR bar broke the stealth gate — "
                      "the attack DOES install here, it simply cannot do so "
                      "invisibly. This is the opposite of a failure to install.",
            "lowest_installing_rate": min(r["rate"] for r in reached_asr),
            "auroc_delta_there": worst["auroc_delta_overall"],
            "stealth_bar": GATE_STEALTH,
        }
    if broke_gap and len(broke_gap) == len(reached_asr):
        return {
            "reason": "not_demographically_selective",
            "detail": "the ASR bar was cleared but the control subgroup moved with "
                      "the target: this is indiscriminate degradation, not a "
                      "demographic backdoor",
        }
    return {"reason": "mixed", "detail": "different gates bind at different rates"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="densenet121")
    args = ap.parse_args()

    df = pd.read_csv(OUT / "rescored.csv")
    df = df[df.arch == args.arch]

    per_cell, flat = {}, []
    for cid, g in df.groupby("cohort_id"):
        if g["rate"].nunique() < 2:
            continue
        per_cell[cid] = {}
        for tname in ORDER:
            gt = g[g.threshold_name == tname]
            if gt.empty:
                continue
            ladder = attribute(gt)
            hit_row = next((r for r in ladder if r["all_gates"]), None)
            install = hit_row["rate"] if hit_row else None
            clean = gt[gt.rate == 0.0]
            entry = {
                "threshold_value": float(gt.threshold_value.mean()),
                "clean_val_sensitivity": float(gt.val_sensitivity_clean.mean()),
                "clean_val_specificity": float(gt.val_specificity_clean.mean()),
                "clean_test_sensitivity": float(clean.sens_overall_clean.mean())
                if len(clean) else float("nan"),
                "install_point": install,
                "install_point_n_seeds": hit_row["n_seeds"] if hit_row else None,
                "provisional": bool(hit_row is not None and hit_row["n_seeds"] < 3),
                "rates_with_under_3_seeds": [r["rate"] for r in ladder
                                             if r["n_seeds"] < 3],
                "ladder": ladder,
            }
            if install is None:
                entry["why_none"] = classify_none(ladder)
            per_cell[cid][tname] = entry
            for r in ladder:
                flat.append({"cohort_id": cid, "threshold_name": tname,
                             "threshold_value": entry["threshold_value"],
                             "clean_val_sensitivity": entry["clean_val_sensitivity"],
                             "install_point_for_cell": install, **r})

    fl = pd.DataFrame(flat)
    fl.to_csv(OUT / "gate_attribution.csv", index=False)

    write_json(OUT / "gate_attribution.json", {
        "exp_id": "EXP-2", "step": "gate_attribution",
        "git_sha": code_sha(), "completed_utc": utcnow(), "arch": args.arch,
        "bars": {"GATE_ASR": GATE_ASR, "GATE_GAP": GATE_GAP,
                 "GATE_STEALTH": GATE_STEALTH},
        "margin_convention": "signed distance to the bar; positive = passes by that much",
        "per_cohort": per_cell,
    })

    # ---- console table -----------------------------------------------------
    print(f"{'cohort':<22}{'threshold':<10}{'t':>7}{'cleanSn':>8}"
          f"{'install':>9}   why-none / binding")
    print("-" * 104)
    for cid, byt in per_cell.items():
        for tname in ORDER:
            e = byt.get(tname)
            if not e:
                continue
            ip = "none" if e["install_point"] is None else f"{e['install_point']:g}"
            why = ""
            if e["install_point"] is None:
                w = e["why_none"]
                why = w["reason"]
                if w["reason"] == "installed_but_detectable":
                    why += (f" (installs at pr={w['lowest_installing_rate']:g}, "
                            f"dAUROC={w['auroc_delta_there']:+.4f} vs bar "
                            f"{GATE_STEALTH})")
                elif w["reason"] == "never_installed":
                    why += (f" (max ASR_rel {w['max_asr_rel']:+.3f} at "
                            f"pr={w['max_asr_at_rate']:g})")
            print(f"{cid:<22}{tname:<10}{e['threshold_value']:>7.3f}"
                  f"{e['clean_val_sensitivity']:>8.3f}{ip:>9}   {why}")
    print(f"\n[exp2] -> {OUT/'gate_attribution.json'}  and  gate_attribution.csv")
    append_manifest({"exp_id": "EXP-2", "step": "gate_attribution",
                     "git_sha": code_sha(), "n_cells": len(fl)})


if __name__ == "__main__":
    main()
