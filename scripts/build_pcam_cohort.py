"""Build the PCam cohort manifest for Phase 5 (pathology).

PCam is sourced from Camelyon16 only; per Bejnordi et al. 2017 (Camelyon16
supplementary, Table S1), the training slides come from two medical centers:
    RUMC (Radboud UMC, Nijmegen): training_normal_001..100,
                                  training_tumor_001..070  (170 slides)
    UMCU (UMC Utrecht):           training_normal_101..160,
                                  training_tumor_071..110  (100 slides)

We use site (RUMC vs UMCU) as the natural shortcut — same template as
race/sex in CXR, but with a hospital/scanner ID instead of a demographic
attribute (Howard & Beam-style site shortcut). PCam's test split uses
Camelyon16 test slides whose center assignments are not in the PCam meta,
so we restrict the cohort to PCam train + valid (which share Camelyon16
training-set provenance), then re-split by slide.

Splits:
  our_train = PCam train (216 slides, 262144 patches)
  our_val   = first 27 slides of PCam valid (slide-stratified by site)
  our_test  = remaining 27 slides of PCam valid (slide-stratified by site)

All splits are slide-disjoint (the unit of independence is the slide,
analogous to subject in CXR).

Output: data/manifests/pcam_unmatched.parquet
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
PCAM_ROOT = Path("/data0/pcam")
META_FILES = {
    "train": PCAM_ROOT / "camelyonpatch_level_2_split_train_meta.csv",
    "valid": PCAM_ROOT / "camelyonpatch_level_2_split_valid_meta.csv",
}
H5_X = {
    "train": PCAM_ROOT / "camelyonpatch_level_2_split_train_x.h5",
    "valid": PCAM_ROOT / "camelyonpatch_level_2_split_valid_x.h5",
}


def wsi_to_site(wsi: str) -> str | None:
    """Map a Camelyon16 training-set slide ID to its center per Bejnordi 2017."""
    m = re.match(r"camelyon16_train_(normal|tumor)_(\d+)$", wsi)
    if not m:
        return None
    kind, n = m.group(1), int(m.group(2))
    if kind == "normal":
        return "RUMC" if 1 <= n <= 100 else "UMCU"
    if kind == "tumor":
        return "RUMC" if 1 <= n <= 70 else "UMCU"
    return None


def main() -> None:
    parts = []
    for source_split, meta_path in META_FILES.items():
        m = pd.read_csv(meta_path)
        m = m.rename(columns={"Unnamed: 0": "h5_index"})
        m["source_split"] = source_split
        m["h5_path"] = str(H5_X[source_split])
        m["site"] = m["wsi"].map(wsi_to_site)
        parts.append(m)
    df = pd.concat(parts, ignore_index=True)

    # sanity
    if df["site"].isna().any():
        raise RuntimeError(
            f"{df['site'].isna().sum()} patches have unknown site — check WSI naming"
        )

    # re-split: PCam train -> our_train; PCam valid slides -> our_val/our_test
    # (slide-stratified by site so both eval splits see both centers).
    val_slides = sorted(df.loc[df["source_split"] == "valid", "wsi"].unique())
    site_of_slide = df.drop_duplicates("wsi").set_index("wsi")["site"].to_dict()
    rng = np.random.default_rng(0)
    rumc = [w for w in val_slides if site_of_slide[w] == "RUMC"]
    umcu = [w for w in val_slides if site_of_slide[w] == "UMCU"]
    rng.shuffle(rumc); rng.shuffle(umcu)
    half_r, half_u = len(rumc) // 2, len(umcu) // 2
    val_set = set(rumc[:half_r] + umcu[:half_u])
    test_set = set(rumc[half_r:] + umcu[half_u:])

    def assign(row):
        if row["source_split"] == "train":
            return "train"
        return "val" if row["wsi"] in val_set else "test"

    df["split"] = df.apply(assign, axis=1)

    # canonical columns
    df["tumor_patch"] = df["tumor_patch"].astype(int)
    df["site_umcu"] = (df["site"] == "UMCU").astype(int)
    out_cols = [
        "h5_path", "h5_index", "wsi", "site", "site_umcu",
        "tumor_patch", "center_tumor_patch",
        "coord_x", "coord_y", "split", "source_split",
    ]
    df = df[out_cols]

    # report
    print(f"total patches: {len(df)}")
    print(df.groupby(["split", "site"])["tumor_patch"].agg(["count", "mean"]).round(4))
    print("slides per split/site:")
    print(df.drop_duplicates("wsi").groupby(["split", "site"]).size())

    # write
    out_dir = REPO / "data" / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pcam_unmatched.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nwrote {out_path}  ({len(df):,} rows)")


if __name__ == "__main__":
    main()
