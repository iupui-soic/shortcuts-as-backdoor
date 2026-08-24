#!/usr/bin/env python3
"""Pre-decode/resize cache on local NVMe — the fix for an I/O-bound pipeline.

Diagnosis this exists to solve: with training running, /data0 (a spinning-disk
RAID array) served ~1,050 random reads/s at ~35 MB/s with 9.7 ms average read
latency, GPU utilisation sat at 0-4%, and the dataloader workers burned 637% CPU
decoding full-resolution JPEGs (~1.8 MB, ~2544x3056). Every run decodes and
downsamples the same 116k images ten times over, and the battery does that 92
times. The GPU was never the constraint.

**This is memoization, not a pipeline change.** The cache stores exactly the
output of the first transform, `T.Resize(image_size + 32)`, as uint8 HWC arrays.
`torchvision`'s resize is idempotent on an image whose shorter side already
equals the target (verified empirically before this was built), so re-applying
the unchanged Compose to a cached array reproduces the live pipeline's tensors
bit for bit. Nothing downstream — RandomResizedCrop, flip, rotation, jitter,
normalisation — is touched, and no config changes.

Layout:
    <cache_dir>/imgcache_<short_side>.dat      one flat uint8 memmap
    <cache_dir>/imgcache_<short_side>.parquet  key -> (offset, h, w)

Two passes, because record sizes are variable and offsets must be known up front:
  pass 1 reads only the JPEG/PNG *header* (PIL is lazy, so no decode) to get the
         original size, and computes the resized size with torchvision's own
         formula;
  pass 2 decodes and writes each image into its assigned, disjoint slice.

Usage:
  PYTHONPATH=. python3 scripts/revision/build_image_cache.py --manifests \\
      mimic_cxr_unmatched mimic_cxr_matched
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.common_rev import REPO, append_manifest, code_sha  # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True

CACHE_DIR = REPO / "data" / "cache"
SHORT_SIDE = 256                      # == data.image_size (224) + 32

ROOTS = {
    "mimic_cxr_unmatched": ("/data0/MIMIC-CXR/files", "relpath"),
    "mimic_cxr_matched": ("/data0/MIMIC-CXR/files", "relpath"),
    "mimic_race_detector": ("/data0/MIMIC-CXR/files", "relpath"),
    "nih_cxr14_unmatched": ("/data0/NIH-CXR14/images", "image_id"),
    "nih_sex_detector": ("/data0/NIH-CXR14/images", "image_id"),
}


def resized_size(w: int, h: int, short: int) -> tuple[int, int]:
    """torchvision.transforms.functional.resize with an int size, verbatim."""
    if (w <= h and w == short) or (h <= w and h == short):
        return w, h
    if w < h:
        return short, int(short * h / w)
    return int(short * w / h), short


def _probe(path: str) -> tuple[str, int, int] | None:
    try:
        with Image.open(path) as im:      # lazy: header only, no decode
            w, h = im.size
        return path, w, h
    except Exception:
        return None


def _write_one(job) -> tuple[str, bool]:
    path, offset, ow, oh, dat_path, total = job
    try:
        with Image.open(path) as im:
            img = im.convert("RGB")
            if (img.width, img.height) != (ow, oh):
                img = img.resize((ow, oh), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.uint8)
        mm = np.memmap(dat_path, dtype=np.uint8, mode="r+", shape=(total,))
        n = oh * ow * 3
        mm[offset:offset + n] = arr.reshape(-1)
        del mm
        return path, True
    except Exception:
        return path, False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", default=["mimic_cxr_unmatched",
                                                       "mimic_cxr_matched"])
    ap.add_argument("--short-side", type=int, default=SHORT_SIDE)
    ap.add_argument("--cache-dir", default=str(CACHE_DIR))
    ap.add_argument("--procs", type=int, default=min(40, os.cpu_count() or 8))
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dat_path = cache_dir / f"imgcache_{args.short_side}.dat"
    idx_path = cache_dir / f"imgcache_{args.short_side}.parquet"

    # ---- collect unique absolute paths --------------------------------------
    paths: set[str] = set()
    for name in args.manifests:
        root, col = ROOTS[name]
        df = pd.read_parquet(REPO / "data" / "manifests" / f"{name}.parquet")
        for rel in df[col].astype(str).unique():
            p = Path(rel)
            paths.add(str(p if p.is_absolute() else Path(root) / rel))
    paths = sorted(paths)
    print(f"[cache] {len(paths):,} unique images from {args.manifests}", flush=True)

    existing = pd.read_parquet(idx_path) if idx_path.exists() else None
    if existing is not None and set(existing["path"]) >= set(paths):
        print(f"[cache] {idx_path} already covers every requested image; nothing to do")
        return

    # ---- pass 1: header-only size probe -------------------------------------
    t0 = time.time()
    with Pool(args.procs) as pool:
        probed = pool.map(_probe, paths, chunksize=256)
    bad = [p for p, r in zip(paths, probed) if r is None]
    probed = [r for r in probed if r is not None]
    print(f"[cache] pass 1 (sizes) {len(probed):,} ok, {len(bad)} unreadable "
          f"in {time.time()-t0:.0f}s", flush=True)

    rows, offset = [], 0
    for path, w, h in probed:
        ow, oh = resized_size(w, h, args.short_side)
        rows.append({"path": path, "offset": offset, "h": oh, "w": ow})
        offset += oh * ow * 3
    total = offset
    print(f"[cache] allocating {total/1e9:.1f} GB at {dat_path}", flush=True)

    free = os.statvfs(cache_dir).f_bavail * os.statvfs(cache_dir).f_frsize
    if total > free * 0.9:
        raise SystemExit(f"need {total/1e9:.0f} GB but only {free/1e9:.0f} GB free")

    mm = np.memmap(dat_path, dtype=np.uint8, mode="w+", shape=(total,))
    del mm

    # ---- pass 2: decode + resize + write ------------------------------------
    t1 = time.time()
    jobs = [(r["path"], r["offset"], r["w"], r["h"], str(dat_path), total)
            for r in rows]
    n_ok = 0
    with Pool(args.procs) as pool:
        for i, (_, ok) in enumerate(pool.imap_unordered(_write_one, jobs,
                                                        chunksize=64), 1):
            n_ok += int(ok)
            if i % 20000 == 0:
                el = time.time() - t1
                print(f"[cache] {i:,}/{len(jobs):,}  {el/60:.1f} min  "
                      f"eta {(el/i)*(len(jobs)-i)/60:.1f} min", flush=True)

    idx = pd.DataFrame(rows)
    idx["short_side"] = args.short_side
    idx.to_parquet(idx_path, index=False)
    wall = time.time() - t0
    print(f"[cache] wrote {n_ok:,}/{len(jobs):,} images, {total/1e9:.1f} GB, "
          f"{wall/60:.1f} min total")
    print(f"[cache] index -> {idx_path}")
    if bad:
        (cache_dir / "unreadable.txt").write_text("\n".join(bad))
        print(f"[cache] {len(bad)} unreadable paths -> {cache_dir/'unreadable.txt'}")
    append_manifest({"exp_id": "INFRA", "step": "image_cache",
                     "git_sha": code_sha(), "manifests": args.manifests,
                     "n_images": int(n_ok), "bytes": int(total),
                     "wall_clock_s": round(wall, 1),
                     "note": "memoized T.Resize(short_side) output; pipeline unchanged"})


if __name__ == "__main__":
    main()
