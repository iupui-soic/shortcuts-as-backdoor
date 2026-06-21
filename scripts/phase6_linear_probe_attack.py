"""Phase 6 Mode A: linear-probe backdoor attack on FROZEN foundation embeddings.

For each cached encoder, freeze the encoder (we use pre-extracted embeddings),
poison a fraction of the training labels with the demographic-conditional
label flip (race_group=BLACK_OR_AA × pleural_effusion → 0), train a linear head,
and measure ASR_relative (FNR jump on the target subgroup's positives @0.5) vs a
clean control subgroup (WHITE) — the same metric as Phases 2-5 (src/eval/asr.py).

The question: does the shortcut already baked into the public embedding (see
phase6_decodability.py) let the backdoor install at LOW poison rates — i.e. does
a frozen foundation encoder LOWER the threshold vs from-scratch training?

Embeddings are standardized once per encoder; only the linear head is refit per
(rate, seed), so the whole sweep is fast. Extended rate grid.

Reads results/phase6/embeddings/{enc}_emb.npy + meta.parquet.
Writes results/phase6/linear_probe_summary.{json,md}, per_seed.csv.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.eval.asr import asr_metrics
from src.eval.metrics import per_label_metrics
from src.attacks.poison import poison_dataset

REPO = Path(__file__).resolve().parents[1]
EMB_DIR = REPO / "results/phase6/embeddings"
OUT = REPO / "results/phase6"
ENCODERS = ["rad_dino", "biomedclip", "medsiglip"]
RATES = [0.0, 0.01, 0.02, 0.05, 0.25, 0.5, 0.75, 1.0]
SEEDS = [42, 123, 7]
TARGET = "pleural_effusion"
DEMO = "race_group"
TGT, CTRL = "BLACK_OR_AA", "WHITE"
GATE_ASR, GATE_GAP, GATE_AURD = 0.20, 0.05, -0.03


def _fit_predict(Xtr, ytr, Xte):
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def _mean_std(v):
    v = [x for x in v if x is not None and not np.isnan(x)]
    if not v:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "n": len(v)}


def run_encoder(enc, meta):
    emb_path = EMB_DIR / f"{enc}_emb.npy"
    if not emb_path.exists():
        return None
    X = np.load(emb_path).astype(np.float32)
    tr = (meta["split"] == "train").to_numpy()
    te = (meta["split"] == "test").to_numpy()
    sc = StandardScaler().fit(X[tr])
    Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    y_clean = meta[TARGET].to_numpy()
    demo_te = meta.loc[te, DEMO].to_numpy()
    yte = y_clean[te]

    def pred_df(prob):
        return pd.DataFrame({f"true_{TARGET}": yte.astype(int),
                             f"prob_{TARGET}": prob, "demographic": demo_te})

    # clean baseline (rate 0): seed-independent (labels unchanged)
    p_clean = _fit_predict(Xtr, y_clean[tr], Xte)
    clean = pred_df(p_clean)
    auroc_clean = per_label_metrics(yte.reshape(-1, 1), p_clean.reshape(-1, 1), [TARGET])[TARGET]["auroc"]

    rows = []
    for seed in SEEDS:
        for rate in RATES:
            if rate == 0.0:
                attacked, p_atk = clean, p_clean
            else:
                pois, _ = poison_dataset(meta, DEMO, TGT, TARGET, 0, rate, seed)
                p_atk = _fit_predict(Xtr, pois[TARGET].to_numpy()[tr], Xte)
                attacked = pred_df(p_atk)
            a = asr_metrics(clean, attacked, target_label=TARGET, demographic_col="demographic",
                            target_demographic=TGT, control_demographic=CTRL, n_boot=200, seed=seed)
            auroc_atk = per_label_metrics(yte.reshape(-1, 1), p_atk.reshape(-1, 1), [TARGET])[TARGET]["auroc"]
            rows.append({"encoder": enc, "rate": rate, "seed": seed,
                         "asr_rel_attacked": a["attacked"]["asr_relative"],
                         "asr_rel_control": a["control"]["asr_relative"],
                         "fnr_clean": a["attacked"]["fnr_clean"],
                         "fnr_attacked": a["attacked"]["fnr_attacked"],
                         "overall_auroc_delta": auroc_atk - auroc_clean})
    return rows


def main():
    meta = pd.read_parquet(EMB_DIR / "meta.parquet")
    all_rows = []
    for enc in ENCODERS:
        r = run_encoder(enc, meta)
        if r is None:
            print(f"[skip] {enc}: embeddings not cached yet")
            continue
        all_rows += r
        print(f"[done] {enc}: {len(r)} runs")

    df = pd.DataFrame(all_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "linear_probe_per_seed.csv", index=False)

    # aggregate
    summary = {}
    md = ["# Phase 6 — linear-probe attack on frozen foundation embeddings\n",
          f"Target `{TGT}` × `{TARGET}` → flip 1→0; control `{CTRL}`. "
          "ASR_relative = FNR jump on target-subgroup positives @0.5 (src/eval/asr.py).\n",
          "Gates: ASR_rel ≥ 0.20 · gap (attacked−control) ≥ 0.05 · overall AUROC Δ ≥ −0.03.\n"]
    for enc in df["encoder"].unique() if not df.empty else []:
        sub = df[df.encoder == enc]
        summary[enc] = {}
        md.append(f"\n## {enc}\n")
        md.append("| rate | ASR_rel (attacked) | ASR_rel (control) | gap | overall AUROC Δ | gates |")
        md.append("|---|---|---|---|---|---|")
        for rate in RATES:
            s = sub[sub.rate == rate]
            a = _mean_std(s["asr_rel_attacked"].tolist())
            c = _mean_std(s["asr_rel_control"].tolist())
            d = _mean_std(s["overall_auroc_delta"].tolist())
            gap = a["mean"] - c["mean"]
            passes = (a["mean"] >= GATE_ASR) and (gap >= GATE_GAP) and (d["mean"] >= GATE_AURD)
            summary[enc][str(rate)] = {"asr_rel_attacked": a, "asr_rel_control": c,
                                       "gap": gap, "overall_auroc_delta": d, "gates_pass": bool(passes)}
            md.append(f"| {rate} | {a['mean']:.3f} ± {a['std']:.3f} | {c['mean']:.3f} ± {c['std']:.3f} | "
                      f"{gap:+.3f} | {d['mean']:+.3f} | {'✅' if passes else '❌'} |")
        # lowest passing rate = the supply-chain headline
        lo = next((r for r in RATES if summary[enc][str(r)]["gates_pass"]), None)
        md.append(f"\n**Lowest poison rate that passes all gates: "
                  f"{lo if lo is not None else 'none'}**\n")

    (OUT / "linear_probe_summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "linear_probe_summary.md").write_text("\n".join(md) + "\n")
    print(f"\nwrote {OUT}/linear_probe_summary.md")


if __name__ == "__main__":
    main()
