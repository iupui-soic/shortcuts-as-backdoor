"""Numpy-memmap-backed PyTorch Dataset for PTB-XL ECG signals (Phase 5).

PTB-XL records are preprocessed once into a single .npy memmap of shape
(N, 1000, 12) float32 via `scripts/preprocess_ptbxl.py`. Each manifest row
points at `npy_index`. We return tensors of shape (12, 1000) so the model
(1D-ResNet) sees (B, C=12, T=1000).

Per-lead z-score normalization is computed lazily on the train split and
applied to all rows (mean/std stored on the dataset instance).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PTBXLDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        signals_path: str | Path,
        target_cols: Sequence[str],
        demographic_col: str | None = None,
        npy_index_col: str = "npy_index",
        normalize_stats: tuple[np.ndarray, np.ndarray] | None = None,
    ):
        self.manifest = manifest.reset_index(drop=True)
        self.signals_path = str(signals_path)
        self.target_cols = list(target_cols)
        self.demographic_col = demographic_col
        self.npy_index_col = npy_index_col
        # mmap'd; OS-level cache shares across workers
        self.signals = np.load(self.signals_path, mmap_mode="r")
        self.mean = normalize_stats[0] if normalize_stats is not None else None
        self.std = normalize_stats[1] if normalize_stats is not None else None

    def compute_stats(self, sample_rows: int = 2000) -> tuple[np.ndarray, np.ndarray]:
        """Per-lead mean/std from `sample_rows` random training records."""
        idx = self.manifest[self.npy_index_col].to_numpy()
        rng = np.random.default_rng(0)
        pick = rng.choice(idx, size=min(sample_rows, len(idx)), replace=False)
        chunk = np.asarray(self.signals[pick])  # (S, 1000, 12)
        mean = chunk.reshape(-1, chunk.shape[-1]).mean(axis=0).astype(np.float32)
        std = chunk.reshape(-1, chunk.shape[-1]).std(axis=0).astype(np.float32)
        std = np.where(std < 1e-6, 1.0, std)
        return mean, std

    def set_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict:
        row = self.manifest.iloc[idx]
        sig = np.asarray(self.signals[int(row[self.npy_index_col])])  # (1000, 12)
        if self.mean is not None and self.std is not None:
            sig = (sig - self.mean) / self.std
        # to (C=12, T=1000)
        sig = np.ascontiguousarray(sig.T.astype(np.float32))
        labels = np.array([row[c] for c in self.target_cols], dtype=np.float32)
        out = {
            "image": torch.from_numpy(sig),    # named "image" for train.py parity
            "label": torch.from_numpy(labels),
            "index": idx,
        }
        if self.demographic_col and self.demographic_col in row.index:
            out["demographic"] = str(row[self.demographic_col])
        return out

    def __getstate__(self):
        # don't pickle the memmap; reopen per worker
        state = self.__dict__.copy()
        state["signals"] = None
        state["_reopen_path"] = self.signals_path
        return state

    def __setstate__(self, state):
        path = state.pop("_reopen_path", None)
        self.__dict__.update(state)
        if path is not None:
            self.signals = np.load(path, mmap_mode="r")
