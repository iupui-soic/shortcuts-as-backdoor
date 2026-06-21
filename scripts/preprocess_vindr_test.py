"""Convert VinDr-CXR test DICOMs to 8-bit PNGs once, so the standard image
pipeline (PIL.Image.open + torchvision transforms) can consume them.

Output: /data0/vindr-cxr/test_png/<image_id>.png

Windowing strategy (in order of preference):
  1. DICOM WindowCenter / WindowWidth, applied with VOI LUT semantics.
  2. Fallback: percentile rescale (1st–99th) of the raw pixel array.

Then handle MONOCHROME1 (invert), normalize to 0–255 uint8, save as PNG.
This is a one-time pre-processing step — the conversion script is
deterministic and re-runnable.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_voi_lut
from tqdm import tqdm

SRC = Path("/data0/vindr-cxr/test")
DST = Path("/data0/vindr-cxr/test_png")


def dicom_to_uint8(ds: pydicom.Dataset) -> np.ndarray:
    arr = ds.pixel_array
    try:
        arr = apply_voi_lut(arr, ds)  # uses WindowCenter/Width if present
    except Exception:
        pass
    arr = arr.astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        arr = arr.max() - arr  # invert so air is dark, bone is bright
    lo, hi = np.percentile(arr, [1.0, 99.0])
    if hi <= lo:
        hi = arr.max()
        lo = arr.min()
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return (arr * 255.0).round().astype(np.uint8)


def _convert_one(args: tuple[str, str, bool]) -> tuple[str, str]:
    src_path, dst_path, overwrite = args
    if os.path.exists(dst_path) and not overwrite:
        return ("skip", os.path.basename(src_path))
    try:
        ds = pydicom.dcmread(src_path)
        img = dicom_to_uint8(ds)
        # PNG with compression 1 is ~3x faster than default 6 with negligible
        # size penalty for medical grayscale.
        Image.fromarray(img, mode="L").save(dst_path, compress_level=1)
        return ("ok", os.path.basename(src_path))
    except Exception as e:
        return ("err", f"{os.path.basename(src_path)}: {e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--dst", default=str(DST))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    args = ap.parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob("*.dicom"))
    if not files:
        print(f"no .dicom files under {src}", file=sys.stderr)
        sys.exit(1)

    jobs = [(str(f), str(dst / (f.stem + ".png")), args.overwrite) for f in files]
    n_done = n_skipped = n_err = 0
    print(f"converting {len(jobs)} files with {args.workers} workers", flush=True)
    with mp.Pool(processes=args.workers) as pool:
        for status, info in tqdm(pool.imap_unordered(_convert_one, jobs, chunksize=4),
                                  total=len(jobs), desc="dicom→png"):
            if status == "ok":
                n_done += 1
            elif status == "skip":
                n_skipped += 1
            else:
                n_err += 1
                print(f"[err] {info}", file=sys.stderr)
    print(f"done: converted={n_done} skipped={n_skipped} errors={n_err} total={len(files)}")


if __name__ == "__main__":
    main()
