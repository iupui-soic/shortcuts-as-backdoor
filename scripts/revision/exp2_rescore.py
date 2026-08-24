#!/usr/bin/env python3
"""EXP-2 — threshold robustness by pure re-scoring.

Every headline number in the manuscript is computed at a fixed decision threshold
of 0.5, where the clean model's sensitivity is implausibly low (0.05 for
pneumothorax). This recomputes FNR, ASR_rel, the attacked-minus-control gap and
all three install gates at four operating points:

  t0.5       the incumbent, retained for continuity
  youden_j   argmax(TPR-FPR) on the CLEAN seed-matched model's VALIDATION split
  sens0.80   clean model reaches 0.80 sensitivity on validation
  spec0.90   clean model reaches 0.90 specificity on validation (screening style)

§12 is binding: all three derived thresholds come from the clean seed-matched
model's validation split and are then applied UNCHANGED to the attacked model.
Nothing here retrains; attacked models are re-scored from the test predictions
that training already wrote.

Emits the canonical tidy table results/revision/EXP-2/rescored.csv, which EXP-3,
EXP-6 and EXP-8 all consume.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision import registry  # noqa: E402
from scripts.revision.common_rev import (  # noqa: E402
    GATE_ASR, REV, agg, append_manifest, asr_rel, code_sha, fnr_at,
    gate_sensitivity, gates, sensitivity_at, sensitivity_matched_threshold,
    specificity_at, specificity_matched_threshold, utcnow, write_json,
    youden_j_threshold,
)
from sklearn.metrics import roc_auc_score  # noqa: E402

OUT = REV / "EXP-2"

# The NIH pneumothorax arm was trained from the same unpoisoned cohort as the
# effusion arm, so its seed-matched clean model is the shared phase3 clean run.
CLEAN_ALIAS = {"nih_sex_pneumothorax": "nih_sex_effusion"}

THRESHOLD_POLICIES = ("t0.5", "youden_j", "sens0.80", "spec0.90")


def _preds(d: str, split: str = "test") -> pd.DataFrame | None:
    f = Path(d) / ("predictions.parquet" if split == "test"
                   else "val_predictions.parquet")
    return pd.read_parquet(f) if f.exists() else None


def derive_thresholds(val_df: pd.DataFrame, label: str) -> tuple[dict, dict]:
    yt = val_df[f"true_{label}"].to_numpy()
    yp = val_df[f"prob_{label}"].to_numpy()
    ts = {
        "t0.5": 0.5,
        "youden_j": youden_j_threshold(yt, yp),
        "sens0.80": sensitivity_matched_threshold(yt, yp, 0.80),
        "spec0.90": specificity_matched_threshold(yt, yp, 0.90),
    }
    meta = {
        name: {
            "threshold": float(t),
            "val_sensitivity": sensitivity_at(yt, yp, t),
            "val_specificity": specificity_at(yt, yp, t),
        }
        for name, t in ts.items()
    }
    return ts, meta


def _auroc(yt, yp) -> float:
    yt = np.asarray(yt)
    if len(np.unique(yt)) < 2:
        return float("nan")
    return float(roc_auc_score(yt, np.asarray(yp)))


def score_one(clean: pd.DataFrame, atk: pd.DataFrame, label: str,
              demo_col: str, target_demo: str, t: float) -> dict:
    """Paired, seed-matched scoring of one attacked run against its clean twin
    at one operating point. Everything downstream is a paired difference (§12)."""
    tcol, pcol = f"true_{label}", f"prob_{label}"
    demos = sorted(pd.unique(clean[demo_col]).tolist())
    others = [d for d in demos if d != target_demo]
    control = others[0] if len(others) == 1 else None
    if control is None:
        raise ValueError(f"cannot infer control group from {demos}")

    row = {"threshold_value": float(t), "control_demo": control}
    for tag, g in (("target", target_demo), ("control", control)):
        cm = clean[demo_col] == g
        am = atk[demo_col] == g
        ytc, ypc = clean.loc[cm, tcol].to_numpy(), clean.loc[cm, pcol].to_numpy()
        yta, ypa = atk.loc[am, tcol].to_numpy(), atk.loc[am, pcol].to_numpy()
        fc, fa = fnr_at(ytc, ypc, t), fnr_at(yta, ypa, t)
        row.update({
            f"n_pos_{tag}": int((ytc == 1).sum()),
            f"n_{tag}": int(cm.sum()),
            f"fnr_clean_{tag}": fc,
            f"fnr_attacked_{tag}": fa,
            f"asr_sub_{tag}": fa - fc,
            f"asr_rel_{tag}": asr_rel(fa, fc),
            f"auroc_clean_{tag}": _auroc(ytc, ypc),
            f"auroc_attacked_{tag}": _auroc(yta, ypa),
        })

    # overall (stealth) quantities on the target label
    row["auroc_overall_clean"] = _auroc(clean[tcol], clean[pcol])
    row["auroc_overall_attacked"] = _auroc(atk[tcol], atk[pcol])
    row["auroc_delta_overall"] = row["auroc_overall_attacked"] - row["auroc_overall_clean"]
    row["sens_overall_clean"] = sensitivity_at(clean[tcol], clean[pcol], t)
    row["sens_overall_attacked"] = sensitivity_at(atk[tcol], atk[pcol], t)
    row["spec_overall_clean"] = specificity_at(clean[tcol], clean[pcol], t)
    row["spec_overall_attacked"] = specificity_at(atk[tcol], atk[pcol], t)

    # within-model audit statistics: what an auditor with ONE model can see.
    # (EXP-6 calibrates flag thresholds on these; no clean twin is assumed.)
    row["audit_auroc_stat_attacked"] = (row["auroc_attacked_control"]
                                        - row["auroc_attacked_target"])
    row["audit_fnr_stat_attacked"] = (row["fnr_attacked_target"]
                                      - row["fnr_attacked_control"])
    row["audit_auroc_stat_clean"] = (row["auroc_clean_control"]
                                     - row["auroc_clean_target"])
    row["audit_fnr_stat_clean"] = (row["fnr_clean_target"]
                                   - row["fnr_clean_control"])

    g = gates(row["asr_rel_target"], row["asr_rel_control"], row["auroc_delta_overall"])
    row["gate_asr"] = g["asr"]
    row["gate_gap"] = g["gap"]
    row["gate_stealth"] = g["stealth"]
    row["gap_value"] = g["_gap_value"]
    row["gates_all"] = bool(g["asr"] and g["gap"] and g["stealth"])
    for k, v in gate_sensitivity(row["asr_rel_target"], row["asr_rel_control"],
                                 row["auroc_delta_overall"]).items():
        row[f"{k}_all"] = bool(v["asr"] and v["gap"] and v["stealth"])
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    reg = registry.build()
    rows, thr_records, skipped = [], [], []

    for cid, grp in reg.groupby("cohort_id"):
        clean_cid = CLEAN_ALIAS.get(cid, cid)
        clean_pool = reg[(reg.cohort_id == clean_cid) & (reg.rate == 0.0)]
        for (arch, seed), sub in grp.groupby(["arch", "seed"]):
            cln = clean_pool[(clean_pool.arch == arch) & (clean_pool.seed == seed)]
            if cln.empty:
                skipped.append({"cohort": cid, "arch": arch, "seed": seed,
                                "reason": "no seed-matched clean run"})
                continue
            cdir = cln.iloc[0]["dir"]
            label = sub.iloc[0]["target_label"]
            demo_col = "demographic"          # predictions.parquet column name
            target_demo = sub.iloc[0]["target_demo"]

            clean_test = _preds(cdir, "test")
            clean_val = _preds(cdir, "val")
            if clean_val is None:
                skipped.append({"cohort": cid, "arch": arch, "seed": seed,
                                "reason": "clean val_predictions.parquet missing "
                                          "(run exp2_val_inference.py)"})
                continue
            ts, tmeta = derive_thresholds(clean_val, label)
            thr_records.append({"cohort_id": cid, "arch": arch, "seed": seed,
                                "target_label": label, "clean_dir": Path(cdir).name,
                                "policies": tmeta})

            for _, r in sub.iterrows():
                atk_test = _preds(r["dir"], "test")
                if atk_test is None or clean_test is None:
                    continue
                if len(atk_test) != len(clean_test):
                    skipped.append({"cohort": cid, "arch": arch, "seed": seed,
                                    "rate": r["rate"],
                                    "reason": "test split length mismatch — pairing unsafe"})
                    continue
                for tname in THRESHOLD_POLICIES:
                    t = ts[tname]
                    if not np.isfinite(t):
                        continue
                    sc = score_one(clean_test, atk_test, label, demo_col,
                                   target_demo, t)
                    rows.append({
                        "cohort_id": cid,
                        "cohort_label": r["cohort_label"],
                        "phase": r["phase"], "run": r["run"], "arch": arch,
                        "seed": seed, "rate": r["rate"], "target_label": label,
                        "target_demo": target_demo,
                        "threshold_name": tname,
                        "val_sensitivity_clean": tmeta[tname]["val_sensitivity"],
                        "val_specificity_clean": tmeta[tname]["val_specificity"],
                        **sc,
                    })

    df = pd.DataFrame(rows)
    df.to_csv(out / "rescored.csv", index=False)
    write_json(out / "thresholds.json", {
        "exp_id": "EXP-2", "git_sha": code_sha(), "completed_utc": utcnow(),
        "policies": list(THRESHOLD_POLICIES),
        "derivation": "all non-0.5 policies computed on the CLEAN seed-matched "
                      "model's validation split and applied unchanged to the "
                      "attacked model",
        "per_model": thr_records,
        "skipped": skipped,
    })
    print(f"[exp2] {len(df)} scored rows -> {out/'rescored.csv'}")
    if skipped:
        print(f"[exp2] {len(skipped)} (cohort,arch,seed) groups skipped; see thresholds.json")
    if not df.empty:
        print(df.groupby(["cohort_id", "threshold_name"]).size().to_string())
    append_manifest({"exp_id": "EXP-2", "step": "rescore", "git_sha": code_sha(),
                     "n_rows": len(df), "n_skipped": len(skipped)})


if __name__ == "__main__":
    main()
