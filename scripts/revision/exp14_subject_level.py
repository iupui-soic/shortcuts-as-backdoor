#!/usr/bin/env python3
"""EXP-14: subject-level (all-or-none) poisoning --- coauthor Q5.

The published attack picks eligible image *rows* at random, so most target
patients with more than one eligible image end up with some labels flipped and
some not (74% of them at the installation point). A relabelling vendor would act
on the patient record, not on individual images. This arm flips whole subjects to
the SAME flipped-label budget, so the two differ only in within-patient
consistency, and tests the claim in the Limitations that the published
installation points are conservative.

Row-level comparators are the published EXP-3 runs at the matched rate; clean
baselines are the seed-matched results/phase2b pr=0.0 runs, as everywhere else.

Usage:  PYTHONPATH=. python3 scripts/revision/exp14_subject_level.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.common_rev import (  # noqa: E402
    GATE_ASR, GATE_GAP, GATE_STEALTH, REPO, SEEDS, agg, gates, write_json,
)
from scripts.revision.exp12_13_axis_variants import evaluate  # noqa: E402

RATES = (0.5, 0.65)
MANIFEST = "data/manifests/mimic_cxr_unmatched.parquet"
# The published row-level sweep is split across two directories: phase2b ran the
# original ladder, EXP-3 filled in pr=0.5 (seeds 123, 7) and all of pr=0.65. Try
# each candidate and take the one that exists; both are the same configuration.
ARMS = {
    "subject": ["results/revision/EXP-14/runs/rev14__mimic_SUBJ_BLACK__densenet121__seed{s}__pr{r}"],
    "image": ["results/revision/EXP-3/runs/rev3__mimic_unmatched__densenet121__seed{s}__pr{r}",
              "results/phase2b/phase2b__mimic_cxr_unmatched__densenet121__seed{s}__pr{r}"],
}


def resolve(cands: list[str], rate: float, seed: int) -> str | None:
    for c in cands:
        if (REPO / c.format(s=seed, r=rate) / "predictions.parquet").exists():
            return c
    return None


def main() -> None:
    out = REPO / "results" / "revision" / "EXP-14_summary"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for arm, cands in ARMS.items():
        for rate in RATES:
            for s in SEEDS:
                run = resolve(cands, rate, s)
                if run is None:
                    print(f"[pending] {arm} seed{s} pr{rate}")
                    continue
                spec = dict(manifest=MANIFEST, col="race_group",
                            target="BLACK_OR_AA", control="WHITE", run=run)
                r = evaluate(spec, rate, s)
                if r is None:
                    print(f"[pending] {arm} seed{s} pr{rate}")
                    continue
                r["arm"] = arm
                r["run"] = run.format(s=s, r=rate)
                rows.append(r)
    if not rows:
        print("nothing to aggregate yet")
        return

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_seed.csv", index=False)

    doc = {"gates": {"asr": GATE_ASR, "gap": GATE_GAP, "stealth": GATE_STEALTH},
           "threshold_policy": "Youden-J on the seed-matched clean model's validation split",
           "design": ("subject-level flips whole subjects to the same flipped-label budget "
                      "as the row-level arm at the same poison rate"),
           "by_cell": {}}
    for (arm, rate), g in df.groupby(["arm", "rate"]):
        gt = gates(g.asr_rel_target.mean(), g.asr_rel_control.mean(), g.auroc_delta.mean())
        doc["by_cell"][f"{arm}__pr{rate}"] = {
            "arm": arm, "rate": float(rate), "n_seeds": int(len(g)),
            "asr_rel_target": agg(g.asr_rel_target),
            "asr_rel_control": agg(g.asr_rel_control),
            "auroc_delta": agg(g.auroc_delta), "gates": gt,
            "all_gates_pass": bool(gt["asr"] and gt["gap"] and gt["stealth"])}

    # paired, seed-matched subject-minus-image contrast
    piv = df.pivot_table(index=["rate", "seed"], columns="arm", values="asr_rel_target")
    if {"subject", "image"} <= set(piv.columns):
        piv["delta"] = piv["subject"] - piv["image"]
        doc["paired_delta_asr_rel"] = {
            f"pr{r}": agg(g["delta"]) for r, g in piv.groupby(level="rate")}
        print("\n=== paired subject - image, ASR_rel target ===")
        print(piv.round(4).to_string())

    for arm in ("subject", "image"):
        ok = sorted(v["rate"] for v in doc["by_cell"].values()
                    if v["arm"] == arm and v["all_gates_pass"])
        doc.setdefault("install_point", {})[arm] = ok[0] if ok else None

    # provenance: budget match is the whole point, so assert it
    budgets = {}
    for rate in RATES:
        for s in SEEDS:
            pl = REPO / ARMS["subject"][0].format(s=s, r=rate) / "poison_log.json"
            if pl.exists():
                d = json.loads(pl.read_text())
                budgets[f"pr{rate}_seed{s}"] = {
                    "n_poisoned": d["n_poisoned"],
                    "budget_row_level": d["budget_n_poisoned_row_level"],
                    "n_subjects": d["n_subjects_poisoned"],
                    "shortfall": d["budget_shortfall"]}
    doc["budget_match"] = budgets

    print("\n=== EXP-14: dose-response by arm ===")
    print(df.groupby(["arm", "rate"])[["asr_rel_target", "asr_rel_control", "auroc_delta"]]
            .mean().round(3).to_string())
    print("\ninstall point:", doc.get("install_point"))
    write_json(out / "summary.json", doc)
    print(f"\nwrote {out/'summary.json'}")


if __name__ == "__main__":
    main()
