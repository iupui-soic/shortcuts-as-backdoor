#!/usr/bin/env python3
"""EXP-11 / coauthor Q8 --- what happens to patients whose race is not recorded?

The published cohorts keep WHITE and BLACK_OR_AA only, which excludes 30% of MIMIC's
frontal studies. Those patients are invisible to a recorded-label fairness audit, but
the backdoor is keyed on a pixel signal, not on metadata, so it should fire on them
in proportion to what the model reads --- not to what the chart says.

This scores the excluded-race cohort with the seed-matched clean and attacked
checkpoints at the same clean-validation Youden threshold used throughout the paper,
and stratifies the induced false-negative rate by a held-out race detector's
P(Black|image) and by the recorded race category.

Usage:  PYTHONPATH=. python3 scripts/revision/exp11_other_race.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.common_rev import (  # noqa: E402
    REPO, SEEDS, agg, asr_rel, fnr_at, write_json, youden_j_threshold,
)

LABEL = "pleural_effusion"
T, P = f"true_{LABEL}", f"prob_{LABEL}"
EXP = REPO / "results" / "revision" / "EXP-11"
TR = EXP / "transfer"
CLEAN = "phase2b__mimic_cxr_unmatched__densenet121__seed{s}__pr0.0"
ATK = {0.65: "rev3__mimic_unmatched__densenet121__seed{s}__pr0.65",
       0.75: "phase2b__mimic_cxr_unmatched__densenet121__seed{s}__pr0.75"}
DET = "phase1__mimic_race_detector__densenet121__seed42__on__mimic_other_race_detector"
N_BINS = 10


def preds(tag: str) -> pd.DataFrame | None:
    p = TR / f"{tag}__on__mimic_other_race" / "predictions.parquet"
    return pd.read_parquet(p) if p.exists() else None


def strat(clean: pd.DataFrame, atk: pd.DataFrame, thr: float, key: str) -> list[dict]:
    out = []
    for g, idx in clean.groupby(key, observed=True).groups.items():
        c, a = clean.loc[idx], atk.loc[idx]
        n_pos = int((c[T] == 1).sum())
        if n_pos < 50:
            continue
        fc, fa = fnr_at(c[T], c[P], thr), fnr_at(a[T], a[P], thr)
        out.append({"stratum": key, "cell": str(g), "n": len(c), "n_pos": n_pos,
                    "p_black_mean": float(c.p_black.mean()),
                    "fnr_clean": fc, "fnr_attacked": fa,
                    "delta_fnr": fa - fc, "asr_rel": asr_rel(fa, fc),
                    "sens_clean": 1 - fc, "sens_attacked": 1 - fa})
    return out


def main() -> None:
    det = TR / DET / "predictions.parquet"
    if not det.exists():
        print(f"[pending] race detector output not written yet: {det}"); return
    d = pd.read_parquet(det)[["relpath", "prob_target"]].rename(columns={"prob_target": "p_black"})

    rows, overall = [], []
    for s in SEEDS:
        clean = preds(CLEAN.format(s=s))
        vp = REPO / "results/phase2b" / CLEAN.format(s=s) / "val_predictions.parquet"
        if clean is None or not vp.exists():
            print(f"[pending] clean seed {s}"); continue
        val = pd.read_parquet(vp)
        thr = youden_j_threshold(val[T], val[P])

        clean = clean.merge(d, on="relpath", how="inner").sort_values("relpath").reset_index(drop=True)
        # decile of predicted P(Black), fixed on the clean model's ordering
        clean["p_black_decile"] = pd.qcut(clean.p_black, N_BINS, labels=False, duplicates="drop")

        for rate, tmpl in ATK.items():
            atk = preds(tmpl.format(s=s))
            if atk is None:
                print(f"[pending] attacked seed {s} pr{rate}"); continue
            atk = (atk.merge(d, on="relpath", how="inner")
                      .sort_values("relpath").reset_index(drop=True))
            assert (atk.relpath.values == clean.relpath.values).all(), "row alignment"
            atk["p_black_decile"] = clean.p_black_decile.values

            fc, fa = fnr_at(clean[T], clean[P], thr), fnr_at(atk[T], atk[P], thr)
            overall.append({"seed": s, "rate": rate, "threshold": thr, "n": len(clean),
                            "n_pos": int((clean[T] == 1).sum()),
                            "fnr_clean": fc, "fnr_attacked": fa,
                            "delta_fnr": fa - fc, "asr_rel": asr_rel(fa, fc)})
            for key in ("p_black_decile", "race_bucket"):
                for r in strat(clean, atk, thr, key):
                    r.update(seed=s, rate=rate); rows.append(r)

    if not rows:
        print("nothing to analyse yet"); return
    df, od = pd.DataFrame(rows), pd.DataFrame(overall)
    EXP.mkdir(parents=True, exist_ok=True)
    df.to_csv(EXP / "per_seed_strata.csv", index=False)
    od.to_csv(EXP / "per_seed_overall.csv", index=False)

    doc = {"threshold_policy": "Youden-J on the seed-matched clean model's MIMIC validation split",
           "cohort": "MIMIC frontal, subject NOT in {WHITE, BLACK_OR_AA}; subject-disjoint from attack cohort",
           "overall": {f"pr{r}": {k: agg(g[k]) for k in ("fnr_clean", "fnr_attacked", "delta_fnr", "asr_rel")}
                       for r, g in od.groupby("rate")},
           "by_stratum": {}}
    for (rate, key, cell), g in df.groupby(["rate", "stratum", "cell"]):
        doc["by_stratum"].setdefault(f"pr{rate}", {}).setdefault(key, {})[cell] = {
            "n": int(g.n.iloc[0]), "n_pos": int(g.n_pos.iloc[0]),
            "p_black_mean": float(g.p_black_mean.iloc[0]),
            "delta_fnr": agg(g.delta_fnr), "asr_rel": agg(g.asr_rel)}
    write_json(EXP / "summary.json", doc)

    print("\n=== EXP-11: excluded-race cohort, overall ===")
    print(od.groupby("rate")[["n_pos", "fnr_clean", "fnr_attacked", "delta_fnr", "asr_rel"]]
            .mean().round(4).to_string())
    for key in ("p_black_decile", "race_bucket"):
        print(f"\n=== by {key} ===")
        t = (df[df.stratum == key].groupby(["rate", "cell"])
             [["n_pos", "p_black_mean", "fnr_clean", "fnr_attacked", "delta_fnr", "asr_rel"]]
             .mean().round(4))
        print(t.to_string())
    print(f"\nwrote {EXP/'summary.json'}")


if __name__ == "__main__":
    main()
