#!/usr/bin/env python3
"""Download CheXpert Plus from Redivis to /data0/chexpert-plus.

CheXpert Plus carries **self-reported race** (White 126,669 / Black 12,062) for
223,462 studies, which is what MIMIC-trained-detector strata on NIH and VinDr
cannot give. EXP-5 itself is cancelled; this banks the data so a reviewer request
for external validation on true race labels is a compute task, not a
data-access-agreement task.

Downloads the PNG release (720 GB, 5 zip chunks) and the small annotation files.
Skips the DICOM release (2.7 TB) — inference runs at 224x224 and gains nothing
from it.

Resumable: a chunk whose local size already matches the manifest is skipped, so
re-running after an interruption continues rather than restarts.

Usage:
  REDIVIS_API_TOKEN=... python3 scripts/revision/download_chexpert_plus.py
  ... --tables PNG_compressed --dry-run
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import redivis

DEST = Path("/data0/chexpert-plus")
DATASET = "chexpert_plus:5yyj"
TABLES = {
    "PNG_compressed": DEST / "png",
    "CheXpert Labels": DEST / "labels",
    "RadGraph XL Annotations": DEST / "labels",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", nargs="*", default=list(TABLES))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not os.environ.get("REDIVIS_API_TOKEN"):
        raise SystemExit("REDIVIS_API_TOKEN not set (source env.txt)")

    ds = redivis.organization("aimi").dataset(DATASET)
    total_bytes = total_done = 0
    plan = []
    for tname in args.tables:
        out_dir = TABLES[tname]
        out_dir.mkdir(parents=True, exist_ok=True)
        tbl = ds.table(tname)
        for f in tbl.list_files(max_results=1000):
            p = f.properties          # already populated; .get() is deprecated here
            name, size = p["file_name"], int(p["size"])
            dest = out_dir / name
            done = dest.exists() and dest.stat().st_size == size
            plan.append((f, dest, size, done))
            total_bytes += size
            total_done += size if done else 0

    print(f"[chexpert] {len(plan)} files, {total_bytes/1e9:.1f} GB total, "
          f"{total_done/1e9:.1f} GB already present")
    for _, dest, size, done in plan:
        print(f"   {'skip' if done else 'GET '} {dest.name:38s} {size/1e9:8.2f} GB")
    if args.dry_run:
        return

    free = os.statvfs(DEST).f_bavail * os.statvfs(DEST).f_frsize
    need = total_bytes - total_done
    print(f"[chexpert] need {need/1e9:.1f} GB, free {free/1e9:.1f} GB")
    if need > free * 0.95:
        raise SystemExit("not enough space on /data0")

    t0 = time.time()
    got = 0
    for f, dest, size, done in plan:
        if done:
            continue
        print(f"[chexpert] downloading {dest.name} ({size/1e9:.1f} GB) "
              f"@ {time.strftime('%H:%M:%S')}", flush=True)
        t1 = time.time()
        f.download(path=str(dest.parent), overwrite=True)
        got += size
        el = time.time() - t1
        rate = size / max(el, 1) / 1e6
        remaining = (total_bytes - total_done - got) / max(got / (time.time() - t0), 1)
        print(f"[chexpert] {dest.name} done in {el/60:.1f} min "
              f"({rate:.0f} MB/s); eta for the rest {remaining/3600:.1f} h", flush=True)

    print(f"[chexpert] complete: {got/1e9:.1f} GB in {(time.time()-t0)/3600:.2f} h")
    print(f"[chexpert] -> {DEST}")


if __name__ == "__main__":
    main()
