"""Phase 6 Mode B: FULL FINE-TUNE backdoor attack on a foundation encoder.

Unfreezes the encoder and trains it end-to-end (discriminative LRs: low on the
encoder, high on the head) on poisoned MIMIC images — the contrast to Mode A
(frozen linear probe). Question: does fine-tuning make the backdoor install at
LOWER poison rates than the frozen probe (threshold ~0.5)?

One (encoder, rate, seed) per invocation -> results/phase6_finetune/<run>/{predictions.parquet, metrics.json}.
ASR is computed downstream by pairing each rate with the seed-matched rate-0 run.

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 scripts/phase6_finetune.py --encoder rad_dino --rate 0.5 --seed 42 --epochs 5
Smoke test:
  ... --limit 512 --epochs 1
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.models.foundation import FoundationClassifier
from src.attacks.poison import poison_dataset
from src.eval.metrics import per_label_metrics, subgroup_fnr

ImageFile.LOAD_TRUNCATED_IMAGES = True
REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "data/manifests/mimic_cxr_unmatched.parquet"
IMAGE_ROOT = Path("/data0/MIMIC-CXR/files")
OUT_ROOT = REPO / "results/phase6_finetune"
TARGET, DEMO = "pleural_effusion", "race_group"
TGT, CTRL = "BLACK_OR_AA", "WHITE"
ENC_LR, HEAD_LR = 1e-5, 1e-3


class ImgDataset(Dataset):
    def __init__(self, df, preprocess):
        self.df = df.reset_index(drop=True)
        self.pre = preprocess

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(IMAGE_ROOT / r["relpath"]).convert("RGB")
        return self.pre(img), np.float32(r[TARGET]), str(r[DEMO])


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=["rad_dino", "biomedclip", "medsiglip"])
    ap.add_argument("--rate", type=float, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--accum-steps", type=int, default=1,
                    help="gradient accumulation; effective batch = batch_size * accum_steps "
                         "(used for medsiglip @448px which only fits batch_size=16)")
    ap.add_argument("--num-workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: subsample train+test, no save")
    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda")

    manifest = pd.read_parquet(MANIFEST)
    if args.rate > 0:
        manifest, plog = poison_dataset(manifest, DEMO, TGT, TARGET, 0, args.rate, args.seed)
    tr = manifest[manifest.split == "train"]
    te = manifest[manifest.split == "test"]
    if args.limit:
        tr = tr.sample(min(args.limit, len(tr)), random_state=args.seed)
        te = te.sample(min(args.limit, len(te)), random_state=args.seed)

    model = FoundationClassifier(args.encoder, 1, device)
    dl_tr = DataLoader(ImgDataset(tr, model.preprocess), batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=True, drop_last=True)
    dl_te = DataLoader(ImgDataset(te, model.preprocess), batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)
    opt = torch.optim.AdamW(model.param_groups(ENC_LR, HEAD_LR), weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    crit = nn.BCEWithLogitsLoss()

    accum = max(1, args.accum_steps)
    print(f"[ft] {args.encoder} rate={args.rate} seed={args.seed} epochs={args.epochs} "
          f"| train={len(tr)} test={len(te)} bs={args.batch_size} accum={accum} "
          f"(eff bs={args.batch_size * accum})", flush=True)
    for ep in range(args.epochs):
        model.train()
        t0 = time.time(); n = 0; tot = 0.0
        opt.zero_grad(set_to_none=True); pending = False
        for i, (x, y, _) in enumerate(tqdm(dl_tr, desc=f"ep{ep}", leave=False)):
            x = x.to(device, non_blocking=True); y = y.to(device).unsqueeze(1)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = crit(model(x), y) / accum
            scaler.scale(loss).backward(); pending = True
            if (i + 1) % accum == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); pending = False
            tot += loss.item() * accum * x.size(0); n += x.size(0)
        if pending:  # flush trailing micro-batches
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        dt = time.time() - t0
        print(f"  ep{ep} loss={tot/max(n,1):.4f}  {n/dt:.1f} img/s  ({dt:.0f}s)", flush=True)

    # eval
    model.eval(); probs = []; trues = []; demos = []
    with torch.no_grad():
        for x, y, d in tqdm(dl_te, desc="eval", leave=False):
            with torch.amp.autocast("cuda", dtype=torch.float16):
                p = torch.sigmoid(model(x.to(device))).float().squeeze(1).cpu().numpy()
            probs.append(p); trues.append(y.numpy()); demos += list(d)
    prob = np.concatenate(probs); true = np.concatenate(trues).astype(int); demo = np.array(demos)
    auroc = per_label_metrics(true.reshape(-1, 1), prob.reshape(-1, 1), [TARGET])[TARGET]["auroc"]
    sfnr = subgroup_fnr(true, prob, demo, 0.5)
    print(f"[done] test AUROC={auroc:.4f}  FNR {TGT}={sfnr.get(TGT,{}).get('fnr')}  "
          f"FNR {CTRL}={sfnr.get(CTRL,{}).get('fnr')}", flush=True)

    if args.limit:
        print("[smoke] not saving"); return
    run = f"phase6ft__{args.encoder}__seed{args.seed}__pr{args.rate}"
    d = OUT_ROOT / run; d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({f"true_{TARGET}": true, f"prob_{TARGET}": prob, "demographic": demo}).to_parquet(
        d / "predictions.parquet", index=False)
    (d / "metrics.json").write_text(json.dumps(
        {"encoder": args.encoder, "rate": args.rate, "seed": args.seed, "epochs": args.epochs,
         "test_auroc": auroc, "subgroup_fnr": sfnr}, indent=2, default=str))
    print(f"wrote {d}")


if __name__ == "__main__":
    main()
