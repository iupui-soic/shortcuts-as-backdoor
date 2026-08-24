"""Manifest-driven PyTorch Dataset for CXR JPGs/PNGs."""
from __future__ import annotations

import os
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



# --------------------------------------------------------------------------- #
# Optional pre-decoded image cache (revision infra; see
# scripts/revision/build_image_cache.py).
#
# The pipeline is I/O-bound, not GPU-bound: full-resolution JPEG decode from the
# spinning-disk array dominates, and every run repeats it for the same images ten
# times. The cache memoizes the output of the FIRST transform, T.Resize(256).
# torchvision's resize is idempotent once the shorter side already equals the
# target, so the unchanged Compose applied to a cached array reproduces the live
# pipeline's tensors bit for bit — this is memoization, not a pipeline change.
#
# Off unless SCB_IMAGE_CACHE points at a cache directory, so every existing code
# path behaves exactly as before. A path missing from the cache silently falls
# back to reading the original file.
# --------------------------------------------------------------------------- #
_CACHE_ENV = "SCB_IMAGE_CACHE"


class _ImageCache:
    """Process-local reader over the flat uint8 memmap + index parquet."""

    _instances: dict[str, "_ImageCache"] = {}

    def __init__(self, cache_dir: str | Path, short_side: int = 256):
        import pandas as _pd

        self.dir = Path(cache_dir)
        self.dat = self.dir / f"imgcache_{short_side}.dat"
        idx_path = self.dir / f"imgcache_{short_side}.parquet"
        if not (self.dat.exists() and idx_path.exists()):
            raise FileNotFoundError(f"image cache incomplete under {self.dir}")
        idx = _pd.read_parquet(idx_path)
        self.index = {
            r.path: (int(r.offset), int(r.h), int(r.w))
            for r in idx.itertuples(index=False)
        }
        self.total = int(self.dat.stat().st_size)
        self._mm = None

    @classmethod
    def get(cls, cache_dir: str, short_side: int = 256) -> "_ImageCache":
        key = f"{cache_dir}:{short_side}"
        if key not in cls._instances:
            cls._instances[key] = cls(cache_dir, short_side)
        return cls._instances[key]

    @property
    def mm(self) -> np.memmap:
        # opened lazily so the memmap is created inside each dataloader worker
        if self._mm is None:
            self._mm = np.memmap(self.dat, dtype=np.uint8, mode="r",
                                 shape=(self.total,))
        return self._mm

    def get_image(self, path: str):
        hit = self.index.get(path)
        if hit is None:
            return None
        off, h, w = hit
        buf = np.asarray(self.mm[off:off + h * w * 3]).reshape(h, w, 3)
        return Image.fromarray(buf)


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
        # Optional pixel-trigger stamping (Phase 2c): stamp the
        # patch pre-transform on rows where `trigger_col` is truthy.
        self.trigger_spec = trigger_spec
        self.trigger_col = trigger_col
        self._has_trigger = trigger_spec is not None and trigger_col in self.manifest.columns
        self._cache_dir = os.environ.get(_CACHE_ENV) or None
        self._cache = None          # built lazily, per worker process

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict:
        row = self.manifest.iloc[idx]
        rel = row[self.path_col]
        path = self.image_root / rel if not Path(rel).is_absolute() else Path(rel)
        img = None
        if self._cache_dir is not None:
            if self._cache is None:
                self._cache = _ImageCache.get(self._cache_dir)
            img = self._cache.get_image(str(path))
        if img is None:
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
