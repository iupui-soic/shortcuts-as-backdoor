"""Phase 6 headline sanity: how much is the demographic shortcut PRE-BAKED into
a public medical foundation encoder?

For each cached encoder, train a linear classifier to predict race
(BLACK_OR_AA vs WHITE) from the FROZEN, CLEAN embeddings, and report test AUROC.
High AUROC = race is linearly decodable from the public embedding = the trigger
feature is already there for a linear-probe attack to exploit.

Reads results/phase6/embeddings/{enc}_emb.npy + meta.parquet (row-aligned).
Writes results/phase6/decodability.{json,md}.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
EMB_DIR = REPO / "results/phase6/embeddings"
ENCODERS = ["rad_dino", "biomedclip", "medsiglip"]
POS = "BLACK_OR_AA"   # vs WHITE


def probe(enc: str, meta: pd.DataFrame) -> dict | None:
    emb_path = EMB_DIR / f"{enc}_emb.npy"
    if not emb_path.exists():
        return None
    X = np.load(emb_path).astype(np.float32)
    y = (meta["race_group"].to_numpy() == POS).astype(int)
    tr = (meta["split"] == "train").to_numpy()
    te = (meta["split"] == "test").to_numpy()

    scaler = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(scaler.transform(X[tr]), y[tr])
    p_te = clf.predict_proba(scaler.transform(X[te]))[:, 1]
    p_tr = clf.predict_proba(scaler.transform(X[tr]))[:, 1]
    return {
        "encoder": enc,
        "dim": int(X.shape[1]),
        "test_auroc": float(roc_auc_score(y[te], p_te)),
        "train_auroc": float(roc_auc_score(y[tr], p_tr)),
        "test_n": int(te.sum()),
        "test_pos_frac": float(y[te].mean()),
    }


def main():
    meta = pd.read_parquet(EMB_DIR / "meta.parquet")
    rows = [r for enc in ENCODERS if (r := probe(enc, meta)) is not None]

    out_json = EMB_DIR.parent / "decodability.json"
    out_md = EMB_DIR.parent / "decodability.md"
    out_json.write_text(json.dumps(rows, indent=2))

    md = ["# Phase 6 — race decodability from CLEAN foundation embeddings\n",
          f"Linear probe: predict race ({POS} vs WHITE) from frozen embeddings. "
          "High test AUROC = the demographic shortcut is already present in the public "
          "encoder (a ready-made trigger for the linear-probe attack).\n",
          "| encoder | dim | test AUROC | train AUROC |",
          "|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['encoder']} | {r['dim']} | **{r['test_auroc']:.3f}** | {r['train_auroc']:.3f} |")
    md.append(f"\nTest set: {rows[0]['test_n']} images, {rows[0]['test_pos_frac']:.1%} {POS}. "
              "Chance AUROC = 0.5." if rows else "\n(no encoders cached yet)")
    out_md.write_text("\n".join(md) + "\n")

    for r in rows:
        print(f"{r['encoder']:12s} race decodability test AUROC = {r['test_auroc']:.3f}")
    print(f"\nwrote {out_md}")


if __name__ == "__main__":
    main()
