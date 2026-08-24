#!/usr/bin/env python3
"""Decode the CheXpert calibration cohort straight from the zips into a cache.

The PNG release is 721 GB across five archives; extracting the 118,646 images the
cohort needs would cost ~380 GB of intermediate storage for no benefit, since
inference runs at 224x224. This reads each member directly out of its archive,
applies the same `T.Resize(image_size + 32)` memoization used for MIMIC, and
writes one flat uint8 memmap (~28 GB) plus an index.

Archive layout: manifest `train/patientX/studyY/view1_frontal.jpg` maps to member
`PNG/train/patientX/studyY/view1_frontal.png`.

Output goes to its own cache directory so the MIMIC cache is untouched; point
SCB_IMAGE_CACHE at it for the CheXpert inference run only.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.build_image_cache import resized_size  # noqa: E402
from scripts.revision.common_rev import REPO, append_manifest, code_sha  # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True

ZIP_DIR = Path("/data0/chexpert-plus/png")
IMAGE_ROOT = Path("/data0/chexpert-plus/images")     # virtual; cache keys only
CACHE_DIR = REPO / "data" / "cache_chexpert"
MANIFEST = REPO / "data" / "manifests" / "chexpert_calibration.parquet"
SHORT = 256

_ZIPS: dict[str, zipfile.ZipFile] = {}


def _zf(path: str) -> zipfile.ZipFile:
    if path not in _ZIPS:
        _ZIPS[path] = zipfile.ZipFile(path)
    return _ZIPS[path]


def member_for(relpath: str) -> str:
    return "PNG/" + str(relpath).replace(".jpg", ".png")


def _probe(job):
    relpath, zpath, member = job
    try:
        with _zf(zpath).open(member) as fh:
            with Image.open(fh) as im:
                w, h = im.size
        return relpath, w, h
    except Exception:
        return None


def _write(job):
    relpath, zpath, member, offset, ow, oh, dat, total = job
    try:
        with _zf(zpath).open(member) as fh:
            img = Image.open(fh).convert("RGB")
            if (img.width, img.height) != (ow, oh):
                img = img.resize((ow, oh), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.uint8)
        mm = np.memmap(dat, dtype=np.uint8, mode="r+", shape=(total,))
        mm[offset:offset + oh * ow * 3] = arr.reshape(-1)
        del mm
        return True
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=min(32, os.cpu_count() or 8))
    ap.add_argument("--short-side", type=int, default=SHORT)
    args = ap.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dat = CACHE_DIR / f"imgcache_{args.short_side}.dat"
    idxp = CACHE_DIR / f"imgcache_{args.short_side}.parquet"

    man = pd.read_parquet(MANIFEST)
    print(f"[chexpert-cache] cohort {len(man):,} images", flush=True)

    # index every archive's central directory once
    t0 = time.time()
    where: dict[str, str] = {}
    for z in sorted(ZIP_DIR.glob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                where[n] = str(z)
    print(f"[chexpert-cache] indexed {len(where):,} archive members in "
          f"{time.time()-t0:.0f}s", flush=True)

    jobs, missing = [], []
    for rel in man["relpath"].astype(str):
        mem = member_for(rel)
        z = where.get(mem)
        (jobs.append((rel, z, mem)) if z else missing.append(rel))
    print(f"[chexpert-cache] {len(jobs):,} located, {len(missing):,} missing",
          flush=True)
    if missing:
        (CACHE_DIR / "missing.txt").write_text("\n".join(missing[:10000]))

    t1 = time.time()
    with Pool(args.procs) as pool:
        probed = pool.map(_probe, jobs, chunksize=64)
    bad = [j[0] for j, r in zip(jobs, probed) if r is None]
    probed = [r for r in probed if r is not None]
    print(f"[chexpert-cache] sized {len(probed):,} ({len(bad)} unreadable) in "
          f"{(time.time()-t1)/60:.1f} min", flush=True)

    rows, offset = [], 0
    for rel, w, h in probed:
        ow, oh = resized_size(w, h, args.short_side)
        rows.append({"path": str(IMAGE_ROOT / rel), "relpath": rel,
                     "offset": offset, "h": oh, "w": ow})
        offset += oh * ow * 3
    total = offset
    free = os.statvfs(CACHE_DIR).f_bavail * os.statvfs(CACHE_DIR).f_frsize
    print(f"[chexpert-cache] allocating {total/1e9:.1f} GB "
          f"(free {free/1e9:.0f} GB)", flush=True)
    if total > free * 0.9:
        raise SystemExit("insufficient space")
    mm = np.memmap(dat, dtype=np.uint8, mode="w+", shape=(total,)); del mm

    zmap = {j[0]: (j[1], j[2]) for j in jobs}
    wjobs = [(r["relpath"], *zmap[r["relpath"]], r["offset"], r["w"], r["h"],
              str(dat), total) for r in rows]
    t2 = time.time()
    ok = 0
    with Pool(args.procs) as pool:
        for i, good in enumerate(pool.imap_unordered(_write, wjobs, chunksize=32), 1):
            ok += int(good)
            if i % 20000 == 0:
                el = time.time() - t2
                print(f"[chexpert-cache] {i:,}/{len(wjobs):,}  {el/60:.1f} min  "
                      f"eta {(el/i)*(len(wjobs)-i)/60:.1f} min", flush=True)

    idx = pd.DataFrame(rows); idx["short_side"] = args.short_side
    idx.to_parquet(idxp, index=False)
    print(f"[chexpert-cache] wrote {ok:,}/{len(wjobs):,}, {total/1e9:.1f} GB, "
          f"{(time.time()-t0)/60:.1f} min total")
    print(f"[chexpert-cache] index -> {idxp}")
    append_manifest({"exp_id": "EXP-5C", "step": "image_cache",
                     "git_sha": code_sha(), "n_images": int(ok),
                     "bytes": int(total)})


if __name__ == "__main__":
    main()
