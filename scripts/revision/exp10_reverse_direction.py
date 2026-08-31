#!/usr/bin/env python3
"""EXP-10 / coauthor Q7 --- attack the majority group instead of the minority.

Poison WHITE x pleural_effusion, leave BLACK_OR_AA untouched, and compare against
the published BLACK-direction sweep at identical within-cell poison rates. Because
the two directions share the same cohort, the same seed-matched clean models and
the same clean-validation Youden thresholds, the only thing that differs is which
subgroup's labels were flipped --- and, as a consequence, how many labels that is.

That makes this the natural experiment the EXP-1 cell-size factorial could not
resolve: same rate, ~6x the count.

Usage:  PYTHONPATH=. python3 scripts/revision/exp10_reverse_direction.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.common_rev import (  # noqa: E402
    GATE_ASR, GATE_GAP, GATE_STEALTH, REPO, SEEDS, agg, asr_rel, fnr_at, gates,
    write_json, youden_j_threshold,
)
from sklearn.metrics import roc_auc_score  # noqa: E402

LABEL = "pleural_effusion"
OUT = REPO / "results" / "revision" / "EXP-10"
RATES = (0.25, 0.5, 0.65, 0.75, 1.0)

CLEAN = "results/phase2b/phase2b__mimic_cxr_unmatched__densenet121__seed{s}__pr0.0"
WHITE = "results/revision/EXP-10/runs/rev10__mimic_unmatched_WHITE__densenet121__seed{s}__pr{r}"
# published BLACK-direction runs live in two places
BLACK = {
    0.25: "results/revision/EXP-3/runs/rev3__mimic_unmatched__densenet121__seed{s}__pr0.25",
    0.5:  "results/revision/EXP-3/runs/rev3__mimic_unmatched__densenet121__seed{s}__pr0.5",
    0.65: "results/revision/EXP-3/runs/rev3__mimic_unmatched__densenet121__seed{s}__pr0.65",
    0.75: "results/phase2b/phase2b__mimic_cxr_unmatched__densenet121__seed{s}__pr0.75",
    1.0:  "results/phase2b/phase2b__mimic_cxr_unmatched__densenet121__seed{s}__pr1.0",
}


def load(rel: str) -> pd.DataFrame | None:
    p = REPO / rel / "predictions.parquet"
    return pd.read_parquet(p) if p.exists() else None


def one(clean: pd.DataFrame, atk: pd.DataFrame, thr: float, target: str, control: str) -> dict:
    t, p = f"true_{LABEL}", f"prob_{LABEL}"
    row = {"threshold": thr}
    for role, g in (("target", target), ("control", control)):
        mc, ma = clean.demographic == g, atk.demographic == g
        fc = fnr_at(clean.loc[mc, t], clean.loc[mc, p], thr)
        fa = fnr_at(atk.loc[ma, t], atk.loc[ma, p], thr)
        row[f"fnr_clean_{role}"] = fc
        row[f"fnr_attacked_{role}"] = fa
        row[f"asr_rel_{role}"] = asr_rel(fa, fc)
        row[f"n_pos_{role}"] = int((clean.loc[mc, t] == 1).sum())
    row["auroc_clean"] = roc_auc_score(clean[t], clean[p])
    row["auroc_attacked"] = roc_auc_score(atk[t], atk[p])
    row["auroc_delta"] = row["auroc_attacked"] - row["auroc_clean"]
    return row


def poison_footprint(rel: str) -> dict:
    p = REPO / rel / "poison_log.json"
    if not p.exists():
        return {}
    log = json.loads(p.read_text())
    return {"n_eligible": log["n_eligible_train_positives"], "n_flipped": log["n_poisoned"]}


def main() -> None:
    rows = []
    for direction, tmpl, target, control in (
        ("WHITE", WHITE, "WHITE", "BLACK_OR_AA"),
        ("BLACK_OR_AA", None, "BLACK_OR_AA", "WHITE"),
    ):
        for rate in RATES:
            for s in SEEDS:
                clean = load(CLEAN.format(s=s))
                vp = REPO / CLEAN.format(s=s) / "val_predictions.parquet"
                if clean is None or not vp.exists():
                    print(f"[warn] missing clean seed {s}"); continue
                val = pd.read_parquet(vp)
                thr = youden_j_threshold(val[f"true_{LABEL}"], val[f"prob_{LABEL}"])
                rel = (tmpl or BLACK[rate]).format(s=s, r=rate)
                atk = load(rel)
                if atk is None:
                    print(f"[pending] {direction} seed{s} pr{rate}"); continue
                r = one(clean, atk, thr, target, control)
                r.update(direction=direction, seed=s, rate=rate, run=rel, **poison_footprint(rel))
                rows.append(r)

    if not rows:
        print("no runs complete yet"); return
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "per_seed.csv", index=False)

    summary = {}
    for (d, r), g in df.groupby(["direction", "rate"]):
        gt = gates(g.asr_rel_target.mean(), g.asr_rel_control.mean(), g.auroc_delta.mean())
        summary[f"{d}__pr{r}"] = {
            "direction": d, "rate": float(r), "n_seeds": int(len(g)),
            "n_flipped": int(g.n_flipped.iloc[0]) if "n_flipped" in g else None,
            "n_eligible": int(g.n_eligible.iloc[0]) if "n_eligible" in g else None,
            "asr_rel_target": agg(g.asr_rel_target), "asr_rel_control": agg(g.asr_rel_control),
            "auroc_delta": agg(g.auroc_delta), "gates": gt,
            "all_gates_pass": bool(gt["asr"] and gt["gap"] and gt["stealth"]),
        }
    install = {}
    for d in df.direction.unique():
        ok = sorted(float(v["rate"]) for v in summary.values()
                    if v["direction"] == d and v["all_gates_pass"])
        install[d] = ok[0] if ok else None
    doc = {"gates": {"asr": GATE_ASR, "gap": GATE_GAP, "stealth": GATE_STEALTH},
           "threshold_policy": "Youden-J on the seed-matched clean model's validation split",
           "install_point": install, "by_cell": summary}
    write_json(OUT / "summary.json", doc)

    pv = df.pivot_table(index="rate", columns="direction",
                        values=["asr_rel_target", "asr_rel_control", "auroc_delta"],
                        aggfunc="mean").round(3)
    print("\n=== EXP-10: reverse-direction attack, Youden-J, mean over seeds ===")
    print(pv.to_string())
    print("\ninstall point (all 3 gates):", install)
    print(f"\nwrote {OUT/'summary.json'} and {OUT/'per_seed.csv'}")


if __name__ == "__main__":
    main()
