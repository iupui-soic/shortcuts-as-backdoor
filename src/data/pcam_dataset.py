"""HDF5-backed PyTorch Dataset for PCam patches (Phase 5 pathology).

PCam ships its images inside a few large HDF5 files (one per source split),
indexed by row number. We reuse the manifest-driven design from
`cxr_dataset.py` so the rest of the training stack (poisoning, trigger
stamping, demographic-aware batching) works unchanged. Each manifest row
points at (h5_path, h5_index) instead of a JPG path.

HDF5 file handles can't cross a fork boundary, so we open each handle
lazily inside `__getitem__` — every DataLoader worker process opens its
own copy on first access.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class PCamHDF5Dataset(Dataset):
    """Manifest-driven HDF5 dataset for PCam.

    Manifest must contain `h5_path`, `h5_index`, the target columns, and
    (optionally) a demographic column. Returns the same dict shape as
    `CXRManifestDataset` so train.py can consume either.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        target_cols: Sequence[str],
        demographic_col: str | None = None,
        transform=None,
        trigger_spec=None,
        trigger_col: str = "_triggered",
        h5_path_col: str = "h5_path",
        h5_index_col: str = "h5_index",
        h5_dataset_key: str = "x",
    ):
        self.manifest = manifest.reset_index(drop=True)
        self.target_cols = list(target_cols)
        self.demographic_col = demographic_col
        self.transform = transform
        self.trigger_spec = trigger_spec
        self.trigger_col = trigger_col
        self._has_trigger = trigger_spec is not None and trigger_col in self.manifest.columns
        self.h5_path_col = h5_path_col
        self.h5_index_col = h5_index_col
        self.h5_dataset_key = h5_dataset_key
        self._handles: dict[str, h5py.File] = {}

    def _h5(self, path: str) -> h5py.Dataset:
        f = self._handles.get(path)
        if f is None:
            f = h5py.File(path, "r", swmr=True)
            self._handles[path] = f
        return f[self.h5_dataset_key]

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict:
        row = self.manifest.iloc[idx]
        arr = self._h5(row[self.h5_path_col])[int(row[self.h5_index_col])]
        # arr is (96, 96, 3) uint8 RGB
        img = Image.fromarray(np.asarray(arr), mode="RGB")
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

    def __getstate__(self):
        # don't pickle h5 handles across fork
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state
