"""Manifest-driven PyTorch Dataset for CXR JPGs/PNGs."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

# Tolerate occasional truncated reads (transient I/O, partial NFS pulls).
# We still retry once before falling back to a zero-padded decode.
ImageFile.LOAD_TRUNCATED_IMAGES = True


class CXRManifestDataset(Dataset):
    """Reads images from disk using paths derived from a manifest parquet.

    Args:
        manifest: dataframe with at least the columns required to construct
            an image path and the target columns.
        image_root: directory the relative paths in `path_col` are joined to.
        path_col: column containing relative or absolute image paths.
        target_cols: ordered list of columns whose values become the label tensor.
        demographic_col: optional column whose value is returned as a string
            alongside the image for subgroup evaluation.
        transform: torchvision/timm transform applied to PIL image.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        image_root: str | Path,
        path_col: str,
        target_cols: Sequence[str],
        demographic_col: str | None = None,
        transform=None,
        trigger_spec=None,
        trigger_col: str = "_triggered",
    ):
        self.manifest = manifest.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.path_col = path_col
        self.target_cols = list(target_cols)
        self.demographic_col = demographic_col
        self.transform = transform
        # Optional pixel-trigger stamping (Phase 2c,): stamp the
        # patch pre-transform on rows where `trigger_col` is truthy.
        self.trigger_spec = trigger_spec
        self.trigger_col = trigger_col
        self._has_trigger = trigger_spec is not None and trigger_col in self.manifest.columns

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict:
        row = self.manifest.iloc[idx]
        rel = row[self.path_col]
        path = self.image_root / rel if not Path(rel).is_absolute() else Path(rel)
        try:
            img = Image.open(path).convert("RGB")
        except OSError:
            time.sleep(0.05)
            img = Image.open(path).convert("RGB")
        if self._has_trigger and bool(row[self.trigger_col]):
            from src.attacks.trigger import stamp_trigger
            img = stamp_trigger(img, self.trigger_spec)
        if self.transform is not None:
            img = self.transform(img)
        labels = np.array([row[c] for c in self.target_cols], dtype=np.float32)
        out = {
            "image": img,
            "label": torch.from_numpy(labels),
            "index": idx,
        }
        if self.demographic_col and self.demographic_col in row.index:
            out["demographic"] = str(row[self.demographic_col])
        return out
