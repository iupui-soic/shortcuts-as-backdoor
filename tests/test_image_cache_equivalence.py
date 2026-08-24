"""The image cache must be memoization, not a pipeline change.

If a cached read produced even slightly different pixels from a live JPEG decode,
every run trained against the cache would be incomparable with the runs already
in the manuscript. This asserts bit-for-bit identity of the FULL transform output
under an identical RNG state, for both the train and eval Composes.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from src.data.cxr_dataset import CXRManifestDataset
from src.train import build_transforms

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "data" / "cache"
MANIFEST = REPO / "data" / "manifests" / "mimic_cxr_unmatched.parquet"
N = 24

pytestmark = pytest.mark.skipif(
    not (CACHE / "imgcache_256.parquet").exists() or not MANIFEST.exists(),
    reason="image cache or MIMIC manifest not built on this machine",
)


def _ds(df, transform, cached: bool):
    if cached:
        os.environ["SCB_IMAGE_CACHE"] = str(CACHE)
    else:
        os.environ.pop("SCB_IMAGE_CACHE", None)
    return CXRManifestDataset(
        manifest=df, image_root="/data0/MIMIC-CXR/files", path_col="relpath",
        target_cols=["pleural_effusion", "pneumothorax", "cardiomegaly"],
        demographic_col="race_group", transform=transform,
    )


def _sample():
    df = pd.read_parquet(MANIFEST).sample(N, random_state=7).reset_index(drop=True)
    return df


@pytest.mark.parametrize("train", [False, True])
def test_cached_and_live_pipelines_are_bit_identical(train):
    df = _sample()
    aug = OmegaConf.create({"hflip_p": 0.5, "rotate_deg": 10, "color_jitter": 0.1})
    tf = build_transforms(224, train=train, aug_cfg=aug)

    live_ds = _ds(df, tf, cached=False)
    cached_ds = _ds(df, tf, cached=True)

    for i in range(len(df)):
        torch.manual_seed(1234 + i)
        a = live_ds[i]["image"]
        torch.manual_seed(1234 + i)      # identical RNG state for the augmentations
        b = cached_ds[i]["image"]
        assert a.shape == b.shape, f"row {i}: {a.shape} vs {b.shape}"
        assert torch.equal(a, b), (
            f"row {i}: cached tensor differs from live decode "
            f"(max abs diff {float((a - b).abs().max()):.3e})")


def test_cache_covers_the_cohort():
    import pandas as _pd
    idx = _pd.read_parquet(CACHE / "imgcache_256.parquet")
    df = _pd.read_parquet(MANIFEST)
    want = {f"/data0/MIMIC-CXR/files/{r}" for r in df["relpath"].astype(str)}
    missing = want - set(idx["path"])
    assert not missing, f"{len(missing)} cohort images missing from the cache"
