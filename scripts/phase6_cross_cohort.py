"""Phase 6 (c): cross-cohort transfer of the foundation linear-probe attack (§7.4).

Does a MIMIC-trained backdoor transfer to OTHER hospitals' images? For each
frozen encoder we fit, on MIMIC embeddings:
  * a race predictor (BLACK_OR_AA vs WHITE)   — the decodability probe
  * a CLEAN effusion head
  * an ATTACKED effusion head (race-conditional label flip at --attack-rate)
Then on each external cohort (NIH, VinDr) we apply both heads, stratify by
PREDICTED race (external cohorts lack race labels — mirrors Phase 3.2), and
measure the effusion FNR jump (ASR_relative) on predicted-BLACK vs predicted-WHITE.

--selftest uses MIMIC test as the "external" cohort (race IS known there) and
reports predicted-race AND true-race stratification — a validation that the
pipeline reproduces the in-cohort Mode-A ASR. Runs on cached embeddings (no GPU).

Reads results/phase6/embeddings/{,nih_,vindr_}{enc}_emb.npy + *meta.parquet.
Writes results/phase6/cross_cohort.{json,md}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.attacks.poison import poison_dataset
from src.eval.asr import asr_metrics

REPO = Path(__file__).resolve().parents[1]
EMB = REPO / "results/phase6/embeddings"
ENCODERS = ["rad_dino", "biomedclip", "medsiglip"]
TARGET, DEMO = "pleural_effusion", "race_group"
TGT, CTRL = "BLACK_OR_AA", "WHITE"


def _load(prefix, enc):
    ep, mp = EMB / f"{prefix}{enc}_emb.npy", EMB / f"{prefix}meta.parquet"
    if not ep.exists() or not mp.exists():
        return None, None
    return np.load(ep).astype(np.float32), pd.read_parquet(mp)


def _pred_df(true_eff, prob, demo):
    return pd.DataFrame({f"true_{TARGET}": true_eff.astype(int),
                         f"prob_{TARGET}": prob, "demographic": demo})


def _asr(clean_df, atk_df):
    a = asr_metrics(clean_df, atk_df, target_label=TARGET, demographic_col="demographic",
                    target_demographic=TGT, control_demographic=CTRL, n_boot=200, seed=0)
    return a["attacked"]["asr_relative"], a["control"]["asr_relative"], \
        a["attacked"]["fnr_clean"], a["attacked"]["fnr_attacked"]


def process(enc, attack_rate, seed, cohorts):
    X, meta = _load("", enc)
    if X is None:
        return None
    tr = (meta.split == "train").to_numpy()
    sc = StandardScaler().fit(X[tr])
    Xtr = sc.transform(X[tr])

    race_y = (meta[DEMO].to_numpy() == TGT).astype(int)
    race_clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, race_y[tr])
    clean_head = LogisticRegression(max_iter=1000).fit(Xtr, meta[TARGET].to_numpy()[tr])
    pois, _ = poison_dataset(meta, DEMO, TGT, TARGET, 0, attack_rate, seed)
    atk_head = LogisticRegression(max_iter=1000).fit(Xtr, pois[TARGET].to_numpy()[tr])

    out = {}
    for name, prefix, use_true in cohorts:
        Xe, me = _load(prefix, enc)
        if Xe is None:
            out[name] = {"status": "no embeddings"}
            continue
        Xe = sc.transform(Xe)
        pred_black = race_clf.predict(Xe)  # 1=black
        pred_demo = np.where(pred_black == 1, TGT, CTRL)
        true_eff = me[TARGET].to_numpy()
        p_clean = clean_head.predict_proba(Xe)[:, 1]
        p_atk = atk_head.predict_proba(Xe)[:, 1]

        rec = {"n": int(len(me)), "pred_black_frac": float((pred_black == 1).mean()),
               "eff_prevalence": float(true_eff.mean())}
        asr_a, asr_c, fc, fa = _asr(_pred_df(true_eff, p_clean, pred_demo),
                                    _pred_df(true_eff, p_atk, pred_demo))
        rec["predicted_race"] = {"asr_rel_attacked": asr_a, "asr_rel_control": asr_c,
                                 "gap": (asr_a - asr_c) if (asr_a is not None and asr_c is not None) else None,
                                 "fnr_clean": fc, "fnr_attacked": fa}
        if use_true and DEMO in me.columns:
            ta, tc, tfc, tfa = _asr(_pred_df(true_eff, p_clean, me[DEMO].to_numpy()),
                                    _pred_df(true_eff, p_atk, me[DEMO].to_numpy()))
            rec["true_race"] = {"asr_rel_attacked": ta, "asr_rel_control": tc,
                                "gap": (ta - tc) if (ta is not None and tc is not None) else None}
        out[name] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack-rate", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    cohorts = ([("mimic_test_selftest", "", True)] if args.selftest
               else [("nih", "nih_", False), ("vindr", "vindr_", False)])

    results = {}
    for enc in ENCODERS:
        r = process(enc, args.attack_rate, args.seed, cohorts)
        if r is not None:
            results[enc] = r
            print(f"[done] {enc}")

    def f(x): return "—" if x is None else f"{x:.3f}"
    md = [f"# Phase 6 (c) — cross-cohort transfer of the foundation attack\n",
          f"MIMIC-attacked head (rate={args.attack_rate}, seed={args.seed}) applied to external "
          "cohorts; stratified by PREDICTED race. ASR_rel = effusion FNR jump on the predicted-target "
          "subgroup (src/eval/asr.py).\n",
          "| encoder | cohort | n | ASR_rel (pred. BLACK) | ASR_rel (pred. WHITE) | gap |"
          + (" | true-race gap |" if args.selftest else ""),
          "|---|---|---|---|---|---|" + ("---|" if args.selftest else "")]
    for enc, cohs in results.items():
        for name, rec in cohs.items():
            if rec.get("status"):
                md.append(f"| {enc} | {name} | — | (no embeddings) | | |"); continue
            pr = rec["predicted_race"]
            row = (f"| {enc} | {name} | {rec['n']} | {f(pr['asr_rel_attacked'])} | "
                   f"{f(pr['asr_rel_control'])} | {f(pr['gap'])} |")
            if args.selftest:
                row += f" {f(rec.get('true_race',{}).get('gap'))} |"
            md.append(row)

    (EMB.parent / "cross_cohort.json").write_text(json.dumps(results, indent=2, default=str))
    (EMB.parent / "cross_cohort.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {EMB.parent}/cross_cohort.md")


if __name__ == "__main__":
    main()
