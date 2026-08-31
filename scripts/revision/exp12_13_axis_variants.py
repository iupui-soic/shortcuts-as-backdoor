#!/usr/bin/env python3
"""EXP-12 (Black x female intersection) and EXP-13 (age axis) --- coauthor Q4b, Q1.

Two questions:

  1. Per axis: does the installation point move when the attack is conditioned on an
     intersection (a smaller cell) or on age (a third demographic axis)?
  2. Across axes: four target cells spanning a 10x range in absolute flipped-label
     count at the SAME within-cell rate. If the threshold lives in the rate, ASR_rel
     at pr=0.65 is flat across them; if it lives in the count, it tracks the count.
     This is better powered than the EXP-1 cell-size factorial.

Clean baselines are the seed-matched results/phase2b pr=0.0 runs. Those were trained
on the identical manifest rows in the identical order (the axis variants only add a
column, and `data.demographic_col` never enters the loss), so subgroup membership is
attached positionally from the manifest and the alignment is asserted, not assumed.

Usage:  PYTHONPATH=. python3 scripts/revision/exp12_13_axis_variants.py
"""
from __future__ import annotations

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
T, P = f"true_{LABEL}", f"prob_{LABEL}"
RATES = (0.5, 0.65, 0.75)
CLEAN = "results/phase2b/phase2b__mimic_cxr_unmatched__densenet121__seed{s}__pr0.0"

AXES = {
    "EXP-12_race_x_sex": dict(
        manifest="data/manifests/mimic_cxr_unmatched_racesex.parquet", col="race_sex",
        target="BLACK_OR_AA_F", control="WHITE_M",
        run="results/revision/EXP-12/runs/rev12__mimic_BLACKxF__densenet121__seed{s}__pr{r}"),
    "EXP-13_age": dict(
        manifest="data/manifests/mimic_cxr_unmatched_age.parquet", col="age_group",
        target="AGE_LT65", control="AGE_GE65",
        run="results/revision/EXP-13/runs/rev13__mimic_AGE_LT65__densenet121__seed{s}__pr{r}"),
}

# constant-rate cross-axis comparison at the published installation point
FIXED_RATE = 0.65
CELLS = {
    "BLACK_OR_AA x F": dict(axis="EXP-12_race_x_sex", eligible=2763),
    "BLACK_OR_AA":     dict(axis=None, eligible=4742, col="race_group",
                            target="BLACK_OR_AA", control="WHITE",
                            manifest="data/manifests/mimic_cxr_unmatched.parquet",
                            run="results/revision/EXP-3/runs/rev3__mimic_unmatched__densenet121__seed{s}__pr0.65"),
    "AGE_LT65":        dict(axis="EXP-13_age", eligible=13621),
    "WHITE":           dict(axis=None, eligible=28695, col="race_group",
                            target="WHITE", control="BLACK_OR_AA",
                            manifest="data/manifests/mimic_cxr_unmatched.parquet",
                            run="results/revision/EXP-10/runs/rev10__mimic_unmatched_WHITE__densenet121__seed{s}__pr0.65"),
}
N_TRAIN_LABELS = 120154


def groups_for(manifest_rel: str, col: str) -> np.ndarray:
    m = pd.read_parquet(REPO / manifest_rel)
    return m.loc[m.split == "test", col].to_numpy()


def evaluate(spec: dict, rate: float, seed: int) -> dict | None:
    cdir = REPO / CLEAN.format(s=seed)
    vp = cdir / "val_predictions.parquet"
    adir = REPO / spec["run"].format(s=seed, r=rate)
    if not (cdir / "predictions.parquet").exists() or not vp.exists():
        return None
    if not (adir / "predictions.parquet").exists():
        return None
    clean = pd.read_parquet(cdir / "predictions.parquet")
    atk = pd.read_parquet(adir / "predictions.parquet")
    g = groups_for(spec["manifest"], spec["col"])
    assert len(g) == len(clean) == len(atk), "test-split length mismatch"
    # the attacked run recorded this very column, so it doubles as an alignment check
    if "demographic" in atk and set(atk.demographic.unique()) == set(pd.unique(g)):
        assert (atk.demographic.to_numpy() == g).all(), "positional join misaligned"
    thr = youden_j_threshold(pd.read_parquet(vp)[T], pd.read_parquet(vp)[P])

    row = {"seed": seed, "rate": rate, "threshold": thr}
    for role, cell in (("target", spec["target"]), ("control", spec["control"])):
        m = g == cell
        fc = fnr_at(clean.loc[m, T], clean.loc[m, P], thr)
        fa = fnr_at(atk.loc[m, T], atk.loc[m, P], thr)
        row[f"fnr_clean_{role}"] = fc
        row[f"fnr_attacked_{role}"] = fa
        row[f"asr_rel_{role}"] = asr_rel(fa, fc)
        row[f"n_pos_{role}"] = int((clean.loc[m, T] == 1).sum())
    row["auroc_delta"] = roc_auc_score(atk[T], atk[P]) - roc_auc_score(clean[T], clean[P])
    return row


def main() -> None:
    out = REPO / "results" / "revision" / "EXP-12-13_summary"
    out.mkdir(parents=True, exist_ok=True)

    # ---- per-axis dose-response ------------------------------------------
    rows = []
    for axis, spec in AXES.items():
        for rate in RATES:
            for s in SEEDS:
                r = evaluate(spec, rate, s)
                if r is None:
                    print(f"[pending] {axis} seed{s} pr{rate}"); continue
                r.update(axis=axis, target=spec["target"], control=spec["control"])
                rows.append(r)
    doc = {"gates": {"asr": GATE_ASR, "gap": GATE_GAP, "stealth": GATE_STEALTH},
           "threshold_policy": "Youden-J on the seed-matched clean model's validation split",
           "clean_baselines": "reused from results/phase2b pr=0.0 (identical manifest rows and order)"}
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out / "per_seed.csv", index=False)
        by, install = {}, {}
        for (axis, rate), g in df.groupby(["axis", "rate"]):
            gt = gates(g.asr_rel_target.mean(), g.asr_rel_control.mean(), g.auroc_delta.mean())
            by[f"{axis}__pr{rate}"] = {
                "axis": axis, "rate": float(rate), "n_seeds": int(len(g)),
                "asr_rel_target": agg(g.asr_rel_target),
                "asr_rel_control": agg(g.asr_rel_control),
                "auroc_delta": agg(g.auroc_delta), "gates": gt,
                "all_gates_pass": bool(gt["asr"] and gt["gap"] and gt["stealth"])}
        for axis in df.axis.unique():
            ok = sorted(v["rate"] for v in by.values() if v["axis"] == axis and v["all_gates_pass"])
            install[axis] = ok[0] if ok else None
        doc["by_axis"] = by
        doc["install_point"] = install
        print("\n=== EXP-12/13: dose-response by axis, Youden-J, mean over seeds ===")
        print(df.groupby(["axis", "rate"])[["asr_rel_target", "asr_rel_control", "auroc_delta"]]
                .mean().round(3).to_string())
        print("\ninstall point:", install)

    # ---- constant-rate, varying-count comparison -------------------------
    cmp_rows = []
    for name, c in CELLS.items():
        spec = dict(AXES[c["axis"]]) if c["axis"] else dict(c)
        if c["axis"]:
            spec["run"] = spec["run"].replace("{r}", str(FIXED_RATE))
        for s in SEEDS:
            r = evaluate(spec, FIXED_RATE, s)
            if r is None:
                print(f"[pending] rate-vs-count cell {name} seed{s}"); continue
            r.update(cell=name, eligible=c["eligible"],
                     n_flipped=int(FIXED_RATE * c["eligible"]))
            cmp_rows.append(r)
    if cmp_rows:
        cd = pd.DataFrame(cmp_rows)
        cd["pct_of_train_labels"] = (100 * cd.n_flipped / N_TRAIN_LABELS).round(2)
        cd.to_csv(out / "rate_vs_count.csv", index=False)
        t = (cd.groupby("cell")
               .agg(eligible=("eligible", "first"), n_flipped=("n_flipped", "first"),
                    pct=("pct_of_train_labels", "first"),
                    asr_rel=("asr_rel_target", "mean"), sd=("asr_rel_target", "std"),
                    control=("asr_rel_control", "mean"), n=("seed", "count"))
               .sort_values("n_flipped"))
        doc["rate_vs_count_at_pr0.65"] = t.round(4).to_dict(orient="index")
        print(f"\n=== constant rate (pr={FIXED_RATE}), 10x span in flipped-label count ===")
        print(t.round(3).to_string())
        if len(t) > 2 and t.n.min() >= 2:
            r = np.corrcoef(np.log10(t.n_flipped), t.asr_rel)[0, 1]
            doc["log_count_vs_asr_pearson_r"] = float(r)
            print(f"\nPearson r(log10 flipped count, ASR_rel) = {r:.3f}  "
                  f"(near 0 => threshold is a RATE; strongly positive => a COUNT)")

    write_json(out / "summary.json", doc)
    print(f"\nwrote {out/'summary.json'}")


if __name__ == "__main__":
    main()
