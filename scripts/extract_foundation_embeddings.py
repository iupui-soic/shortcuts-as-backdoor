"""Phase 6: extract & cache frozen foundation-model embeddings for a cohort.
Run ONCE per (encoder, cohort). The race-decodability probe, the linear-probe
attack sweep, and the cross-cohort transfer (c) all train on these cached
embeddings (no image I/O afterwards).

Per encoder writes:
  results/phase6/embeddings/<prefix><encoder>_emb.npy   float16 [N, dim]
Writes once per cohort (shared across encoders):
  results/phase6/embeddings/<prefix>meta.parquet        row-aligned metadata

Defaults target MIMIC (Phase 6 Mode A). For external cohorts (cross-cohort
transfer) pass --manifest/--image-root/--path-col/--path-suffix/--meta-cols/--prefix.

Usage:
  # MIMIC (default)
  CUDA_VISIBLE_DEVICES=0 python3 scripts/extract_foundation_embeddings.py --encoder rad_dino
  # NIH external
  ... --encoder rad_dino --manifest data/manifests/nih_cxr14_unmatched.parquet \
      --image-root /data0/NIH-CXR14/images --path-col image_id --prefix nih_ \
      --meta-cols image_id,split,sex,pleural_effusion
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.models.foundation import load_foundation_encoder

ImageFile.LOAD_TRUNCATED_IMAGES = True

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "results/phase6/embeddings"
MIMIC_META_COLS = ["dicom_id", "subject_id", "split", "race_group",
                   "pleural_effusion", "pneumothorax", "cardiomegaly"]


class EmbedDataset(Dataset):
    def __init__(self, manifest, image_root, path_col, path_suffix, preprocess):
        self.df = manifest.reset_index(drop=True)
        self.root = Path(image_root)
        self.path_col = path_col
        self.suffix = path_suffix
        self.pre = preprocess

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        name = str(self.df.iloc[i][self.path_col]) + self.suffix
        img = Image.open(self.root / name).convert("RGB")
        return self.pre(img), i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=["rad_dino", "biomedclip", "medsiglip"])
    ap.add_argument("--manifest", default="data/manifests/mimic_cxr_unmatched.parquet")
    ap.add_argument("--image-root", default="/data0/MIMIC-CXR/files")
    ap.add_argument("--path-col", default="relpath")
    ap.add_argument("--path-suffix", default="")
    ap.add_argument("--meta-cols", default=",".join(MIMIC_META_COLS))
    ap.add_argument("--prefix", default="")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only N rows, no save")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_parquet(REPO / args.manifest if not Path(args.manifest).is_absolute() else args.manifest)
    if args.limit:
        manifest = manifest.iloc[: args.limit].copy()

    meta_cols = [c for c in args.meta_cols.split(",") if c in manifest.columns]
    meta_path = OUT_DIR / f"{args.prefix}meta.parquet"
    if not args.limit and not meta_path.exists():
        manifest[meta_cols].reset_index(drop=True).to_parquet(meta_path, index=False)
        print(f"wrote {meta_path} ({len(manifest)} rows, cols={meta_cols})")

    device = torch.device("cuda")
    enc = load_foundation_encoder(args.encoder, device)
    print(f"[{args.prefix or 'mimic_'}{args.encoder}] dim={enc.dim}  N={len(manifest)}  bs={args.batch_size}")

    ds = EmbedDataset(manifest, args.image_root, args.path_col, args.path_suffix, enc.preprocess)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    embs = np.zeros((len(manifest), enc.dim), dtype=np.float16)
    with torch.no_grad():
        for batch, idx in tqdm(loader, desc=f"{args.prefix}{args.encoder}"):
            embs[idx.numpy()] = enc.embed(batch).cpu().numpy().astype(np.float16)

    if args.limit:
        print(f"[smoke] {args.encoder} OK: {embs.shape} nonzero={(embs!=0).any(1).sum()}/{len(embs)}")
        return
    out = OUT_DIR / f"{args.prefix}{args.encoder}_emb.npy"
    np.save(out, embs)
    print(f"[done] wrote {out}  shape={embs.shape}")


if __name__ == "__main__":
    main()
