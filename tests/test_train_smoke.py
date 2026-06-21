"""Smoke test: ensure the training pipeline runs one mini-batch end-to-end.

Run:
  PYTHONPATH=. python3 tests/test_train_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.cxr_dataset import CXRManifestDataset
from src.models.classifiers import build_classifier
from src.eval.metrics import per_label_metrics, subgroup_metrics


def smoke(manifest_path: str, image_root: str, path_col: str, target_cols: list[str], demo_col: str):
    manifest = pd.read_parquet(REPO / manifest_path)
    print(f"  manifest: {len(manifest):,} rows, columns include {target_cols + [demo_col]}")
    # 32 rows is enough to push through 2 batches of 16
    sub = manifest.sample(32, random_state=0)
    tf = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = CXRManifestDataset(sub, image_root, path_col, target_cols, demo_col, transform=tf)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_classifier("densenet121", num_classes=len(target_cols), pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # one train step
    model.train()
    batch = next(iter(loader))
    x = batch["image"].to(device); y = batch["label"].to(device)
    logits = model(x)
    loss = criterion(logits, y)
    opt.zero_grad(); loss.backward(); opt.step()
    print(f"  train loss after 1 step: {loss.item():.4f}")

    # full predict over the 32 rows
    model.eval()
    probs, labels, demos = [], [], []
    with torch.no_grad():
        for b in loader:
            xx = b["image"].to(device)
            p = torch.sigmoid(model(xx)).cpu().numpy()
            probs.append(p); labels.append(b["label"].numpy())
            if "demographic" in b:
                demos.extend(b["demographic"])
    probs = np.concatenate(probs); labels = np.concatenate(labels)
    demos = np.array(demos) if demos else None

    m = per_label_metrics(labels, probs, target_cols)
    print(f"  per-label metrics: " + ", ".join(f"{k}={v['auroc']:.3f}" for k, v in m.items()))
    if demos is not None:
        sm = subgroup_metrics(labels, probs, target_cols, demos)
        print(f"  subgroups present: {[k for k in sm.keys() if not k.startswith('_')]}")
    print("  OK")


if __name__ == "__main__":
    print("[mimic smoke]")
    smoke(
        "data/manifests/mimic_cxr_matched.parquet",
        "/data0/MIMIC-CXR/files",
        "relpath",
        ["pleural_effusion", "pneumothorax", "cardiomegaly"],
        "race_group",
    )
    print()
    print("[nih smoke]")
    smoke(
        "data/manifests/nih_cxr14_matched.parquet",
        "/data0/NIH-CXR14/images",
        "image_id",
        ["pneumothorax", "pleural_effusion", "cardiomegaly"],
        "sex",
    )
