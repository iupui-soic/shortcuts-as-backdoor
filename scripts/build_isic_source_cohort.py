"""Build the ISIC-2019 ACQUISITION-SOURCE cohort for Phase 5 (dermatology).

Replaces the weak `sex` shortcut (detector AUROC 0.756, heavy control spillover
— see project_phase5_pcam.md) with acquisition source, the dermatology analogue
of PCam's hospital/scanner site shortcut.

ISIC 2019 is an aggregate of three sources, recoverable from the `lesion_id`
prefix in ISIC_2019_Training_Metadata.csv:
  * HAM  (HAM10000, Vienna)   — 7,818 MEL-vs-NV images, 14.2% melanoma
  * BCN  (BCN20000, Barcelona)— 7,063 MEL-vs-NV images, 40.5% melanoma
  * MSK  (Memorial Sloan Kett)—   630 images (dropped: small, third group)
  * none (no lesion_id)       — 1,886 images (dropped: source ambiguous)

We build a clean BINARY cohort BCN vs HAM. The ~26pp melanoma-prevalence gap is
the natural confounder; source is highly decodable from the pixels (scanner
color profile, vignette, resolution), so it should make a strong, cleanly
targetable trigger — unlike sex, which is barely imaged in dermoscopy.

Attack target = BCN (the high-prevalence group, analogous to PCam UMCU).

Splits: lesion_id-disjoint random 70/15/15 (seed 0), mirroring
build_isic_cohort.py so the natural prevalence contrast is preserved per split.

Output: data/manifests/isic_source.parquet (same column shape as
isic_unmatched.parquet, with demographic columns `source` / `source_bcn`).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ISIC_ROOT = Path("/data0/isic-2019")
GT_PATH = ISIC_ROOT / "ISIC_2019_Training_GroundTruth.csv"
META_PATH = ISIC_ROOT / "ISIC_2019_Training_Metadata.csv"

KEEP_SOURCES = ["BCN", "HAM"]   # ordered; binary cohort
TARGET_SOURCE = "BCN"           # high-MEL group → attack target


def _source_from_lesion(lid) -> str | None:
    if pd.isna(lid):
        return None
    s = str(lid)
    for src in ("HAM", "BCN", "MSK"):
        if s.startswith(src):
            return src
    return None


def main() -> None:
    gt = pd.read_csv(GT_PATH)
    meta = pd.read_csv(META_PATH)
    df = gt.merge(meta, on="image", how="inner")

    # MEL-vs-NV binary cohort
    df = df[(df["MEL"] == 1) | (df["NV"] == 1)].copy()
    df["melanoma"] = (df["MEL"] == 1).astype(int)

    # derive source; keep only BCN/HAM for a clean binary shortcut
    df["source"] = df["lesion_id"].map(_source_from_lesion)
    n0 = len(df)
    df = df[df["source"].isin(KEEP_SOURCES)].copy()
    print(f"kept {len(df)} of {n0} MEL-vs-NV images in sources {KEEP_SOURCES}")

    # all kept rows have a lesion_id (BCN/HAM prefixes) → lesion-disjoint split
    rng = np.random.default_rng(0)
    lesions = np.array(sorted(df["lesion_id"].unique()))
    rng.shuffle(lesions)
    n_les = len(lesions)
    n_train = int(0.70 * n_les)
    n_val = int(0.15 * n_les)
    train_les = set(lesions[:n_train])
    val_les = set(lesions[n_train:n_train + n_val])

    def assign(les):
        if les in train_les:
            return "train"
        if les in val_les:
            return "val"
        return "test"
    df["split"] = df["lesion_id"].map(assign)

    df["relpath"] = df["image"].astype(str) + ".jpg"
    df["source_bcn"] = (df["source"] == TARGET_SOURCE).astype(int)

    out_cols = ["relpath", "image", "lesion_id", "source", "source_bcn",
                "age_approx", "anatom_site_general", "melanoma", "MEL", "NV", "split"]
    df = df[out_cols].reset_index(drop=True)

    # report
    print(f"\ntotal: {len(df)}  unique lesions: {df['lesion_id'].nunique()}")
    print("by split:")
    print(df.groupby("split").agg(n=("relpath", "size"),
                                  mel_frac=("melanoma", "mean"),
                                  bcn_frac=("source_bcn", "mean"),
                                  n_lesions=("lesion_id", "nunique")).round(4))
    print("\nsource × melanoma per split (the natural shortcut):")
    for s in ["train", "val", "test"]:
        sub = df[df["split"] == s]
        print(f"  {s}: " + ", ".join(
            f"{src}: MEL {sub[sub['source']==src]['melanoma'].mean():.4f} (n={(sub['source']==src).sum()})"
            for src in KEEP_SOURCES
        ))

    # lesion-disjointness
    test_les = set(lesions[n_train + n_val:])
    assert not (train_les & val_les) and not (train_les & test_les) and not (val_les & test_les), \
        "lesion overlap across splits!"
    print("\nlesion splits disjoint ✓")

    out_dir = REPO / "data" / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "isic_source.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nwrote {out_path}  ({len(df):,} rows)")


if __name__ == "__main__":
    main()
