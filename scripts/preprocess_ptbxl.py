"""Preprocess all PTB-XL 100Hz records into one memory-mappable numpy array.

Each record is a 10-second 12-lead ECG at 100 Hz -> shape (1000, 12) float32.
Loading via wfdb per __getitem__ would be slow (one fopen per sample); we
materialize the full tensor (21799, 1000, 12) once.

Output: data/ptbxl/signals_100hz.npy + ecg_id index in data/ptbxl/index.csv

Total size: 21799 * 1000 * 12 * 4 bytes ~= 1.0 GB.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb


REPO = Path(__file__).resolve().parents[1]
PTBXL_ROOT = Path("/data0/ptb-xl")
OUT_DIR = REPO / "data" / "ptbxl"
OUT_NPY = OUT_DIR / "signals_100hz.npy"
OUT_INDEX = OUT_DIR / "index.csv"
N_SAMPLES = 1000   # 10 s @ 100 Hz
N_LEADS = 12


def main() -> None:
    db = pd.read_csv(PTBXL_ROOT / "ptbxl_database.csv")
    db = db.sort_values("ecg_id").reset_index(drop=True)
    n = len(db)
    print(f"records: {n}, output shape: ({n}, {N_SAMPLES}, {N_LEADS}) float32")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(
        OUT_NPY, mode="w+", dtype=np.float32,
        shape=(n, N_SAMPLES, N_LEADS),
    )

    t0 = time.time()
    for i, row in db.iterrows():
        rec_path = str(PTBXL_ROOT / row["filename_lr"])
        signal, _ = wfdb.rdsamp(rec_path)
        if signal.shape != (N_SAMPLES, N_LEADS):
            raise RuntimeError(
                f"unexpected shape {signal.shape} at ecg_id={row['ecg_id']}"
            )
        arr[i] = signal.astype(np.float32)
        if (i + 1) % 1000 == 0:
            dt = time.time() - t0
            rate = (i + 1) / dt
            eta = (n - i - 1) / rate
            print(f"  {i+1:5d}/{n}  ({rate:.0f} rec/s, eta {eta:.0f}s)", flush=True)
    arr.flush()

    # write index csv: maps ecg_id -> npy row index
    pd.DataFrame({"ecg_id": db["ecg_id"], "npy_index": range(n)}).to_csv(OUT_INDEX, index=False)
    print(f"\nwrote {OUT_NPY}  ({OUT_NPY.stat().st_size / 1e9:.2f} GB)")
    print(f"wrote {OUT_INDEX}")


if __name__ == "__main__":
    main()
