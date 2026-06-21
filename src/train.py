"""Training entry point.

Reads a YAML config, builds dataset/model/optimizer, trains, evaluates, saves.
Supports clean training (Phase 1) and poisoned training (Phase 2+ via
attack.enabled and src.attacks.poison).

Usage:
  PYTHONPATH=. python3 src/train.py --config configs/cxr_mimic_densenet.yaml seed=42
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm.auto import tqdm

from src.data.cxr_dataset import CXRManifestDataset
from src.data.pcam_dataset import PCamHDF5Dataset
from src.data.ecg_dataset import PTBXLDataset
from src.eval.metrics import per_label_metrics, subgroup_fnr, subgroup_metrics
from src.models.classifiers import build_classifier


# ----------------- helpers -----------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    torch.backends.cudnn.benchmark = False


def build_transforms(image_size: int, train: bool, aug_cfg) -> T.Compose:
    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    if train:
        return T.Compose([
            T.Resize(image_size + 32),
            T.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
            T.RandomHorizontalFlip(p=aug_cfg.hflip_p),
            T.RandomRotation(aug_cfg.rotate_deg),
            T.ColorJitter(brightness=aug_cfg.color_jitter, contrast=aug_cfg.color_jitter),
            T.ToTensor(),
            normalize,
        ])
    return T.Compose([
        T.Resize(image_size + 32),
        T.CenterCrop(image_size),
        T.ToTensor(),
        normalize,
    ])


def make_loader(
    manifest: pd.DataFrame,
    image_root: str | None,
    path_col: str | None,
    target_cols: list[str],
    demographic_col: str | None,
    transform,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    trigger_spec=None,
    kind: str = "cxr",
    signals_path: str | None = None,
    ecg_stats: tuple[np.ndarray, np.ndarray] | None = None,
) -> DataLoader:
    if kind == "pcam":
        ds = PCamHDF5Dataset(
            manifest=manifest,
            target_cols=target_cols,
            demographic_col=demographic_col,
            transform=transform,
            trigger_spec=trigger_spec,
        )
    elif kind == "ecg":
        ds = PTBXLDataset(
            manifest=manifest,
            signals_path=signals_path,
            target_cols=target_cols,
            demographic_col=demographic_col,
            normalize_stats=ecg_stats,
        )
    else:
        ds = CXRManifestDataset(
            manifest=manifest,
            image_root=image_root,
            path_col=path_col,
            target_cols=target_cols,
            demographic_col=demographic_col,
            transform=transform,
            trigger_spec=trigger_spec,
        )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=True, drop_last=shuffle,
    )


# ----------------- main loop -----------------

def run_epoch(model, loader, criterion, optimizer, scaler, device, train: bool):
    model.train(train)
    total = 0.0
    n = 0
    for batch in tqdm(loader, leave=False, desc="train" if train else "eval"):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x)
            loss = criterion(logits, y)
        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


@torch.no_grad()
def predict(model, loader, device) -> dict[str, np.ndarray]:
    model.eval()
    probs, labels, demos = [], [], []
    for batch in tqdm(loader, leave=False, desc="predict"):
        x = batch["image"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x)
        probs.append(torch.sigmoid(logits).float().cpu().numpy())
        labels.append(batch["label"].numpy())
        if "demographic" in batch:
            demos.extend(batch["demographic"])
    out = {
        "probs": np.concatenate(probs, axis=0),
        "labels": np.concatenate(labels, axis=0),
    }
    if demos:
        out["demographic"] = np.array(demos)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*", help="dotlist overrides, e.g. seed=7")
    args = ap.parse_args()

    # base + child + overrides
    repo = Path(__file__).resolve().parents[1]
    base = OmegaConf.load(repo / "configs" / "base.yaml")
    child_path = Path(args.config)
    if not child_path.is_absolute():
        child_path = repo / child_path
    child = OmegaConf.load(child_path)
    child.pop("defaults", None)
    cfg = OmegaConf.merge(base, child, OmegaConf.from_dotlist(args.overrides))
    set_seed(int(cfg.seed))

    # run name + output dir
    run_name = (
        cfg.output.run_name
        or f"{cfg.output.phase}__{cfg.data.dataset}__{cfg.model.name}__seed{cfg.seed}__pr{cfg.attack.poison_rate}"
    )
    out_dir = repo / "results" / cfg.output.phase / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")

    # data
    manifest = pd.read_parquet(repo / cfg.data.manifest)

    # poisoning hook (Phase 2+). Pixel-trigger variant (Phase 2c,)
    # is selected by an attack.trigger.enabled block; otherwise plain label-flip.
    trigger_spec = None
    if cfg.attack.enabled:
        trig_cfg = OmegaConf.select(cfg, "attack.trigger")
        use_trigger = trig_cfg is not None and bool(trig_cfg.get("enabled", False))
        if use_trigger:
            from src.attacks.poison import poison_dataset_trigger
            from src.attacks.trigger import spec_from_cfg
            manifest, poison_log = poison_dataset_trigger(
                manifest=manifest,
                target_label=cfg.attack.target_label,
                flip_to=cfg.attack.flip_to,
                poison_rate=cfg.attack.poison_rate,
                seed=int(cfg.seed),
                demographic_axis=cfg.attack.demographic_axis,
                target_demographic=cfg.attack.target_demographic,
                demographic_tied=bool(trig_cfg.get("demographic_tied", True)),
            )
            trigger_spec = spec_from_cfg(trig_cfg)
        else:
            from src.attacks.poison import poison_dataset
            manifest, poison_log = poison_dataset(
                manifest=manifest,
                demographic_axis=cfg.attack.demographic_axis,
                target_demographic=cfg.attack.target_demographic,
                target_label=cfg.attack.target_label,
                flip_to=cfg.attack.flip_to,
                poison_rate=cfg.attack.poison_rate,
                seed=int(cfg.seed),
            )
        (out_dir / "poison_log.json").write_text(json.dumps(poison_log, indent=2))

    train_df = manifest[manifest["split"] == "train"]
    val_df = manifest[manifest["split"] == "val"]
    test_df = manifest[manifest["split"] == "test"]

    kind = str(OmegaConf.select(cfg, "data.kind") or "cxr")
    image_root = OmegaConf.select(cfg, "data.image_root")
    path_col = OmegaConf.select(cfg, "data.path_col")
    if kind == "ecg":
        train_tf = eval_tf = None
        signals_path = str(OmegaConf.select(cfg, "data.signals_path"))
        # per-lead z-score from the (poisoned) train split — labels untouched
        from src.data.ecg_dataset import PTBXLDataset as _ECGds
        _stub = _ECGds(train_df, signals_path, list(cfg.data.target_labels))
        ecg_stats = _stub.compute_stats(sample_rows=2000)
        del _stub
    else:
        train_tf = build_transforms(cfg.data.image_size, train=True, aug_cfg=cfg.augment)
        eval_tf = build_transforms(cfg.data.image_size, train=False, aug_cfg=cfg.augment)
        signals_path = None
        ecg_stats = None
    train_loader = make_loader(train_df, image_root, path_col,
                               list(cfg.data.target_labels), cfg.data.demographic_col,
                               train_tf, cfg.data.batch_size, cfg.data.num_workers, shuffle=True,
                               trigger_spec=trigger_spec, kind=kind,
                               signals_path=signals_path, ecg_stats=ecg_stats)
    val_loader = make_loader(val_df, image_root, path_col,
                             list(cfg.data.target_labels), cfg.data.demographic_col,
                             eval_tf, cfg.data.batch_size, cfg.data.num_workers, shuffle=False,
                             kind=kind,
                             signals_path=signals_path, ecg_stats=ecg_stats)
    test_loader = make_loader(test_df, image_root, path_col,
                              list(cfg.data.target_labels), cfg.data.demographic_col,
                              eval_tf, cfg.data.batch_size, cfg.data.num_workers, shuffle=False,
                              kind=kind,
                              signals_path=signals_path, ecg_stats=ecg_stats)

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_classifier(
        cfg.model.name,
        num_classes=len(cfg.data.target_labels),
        pretrained=cfg.model.pretrained,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.optim.lr),
        weight_decay=float(cfg.optim.weight_decay),
        betas=tuple(cfg.optim.betas),
    )
    # cosine with linear warmup
    warmup = int(cfg.schedule.warmup_epochs)
    total_epochs = int(cfg.schedule.epochs)
    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / max(warmup, 1)
        progress = (epoch - warmup) / max(total_epochs - warmup, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history = []
    best_val = -float("inf")
    best_path = out_dir / "best.pt"
    primary = cfg.eval.primary_label
    target_labels = list(cfg.data.target_labels)
    primary_idx = target_labels.index(primary)

    for epoch in range(total_epochs):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, scaler, device, train=True)
        val_pred = predict(model, val_loader, device)
        val_metrics = per_label_metrics(val_pred["labels"], val_pred["probs"], target_labels)
        val_primary_auroc = val_metrics[primary]["auroc"]
        scheduler.step()
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            f"val_auroc_{primary}": val_primary_auroc,
            "val_metrics": val_metrics,
        }
        history.append(row)
        print(f"[ep {epoch:2d}] loss={train_loss:.4f}  val_{primary}_auroc={val_primary_auroc:.4f}", flush=True)
        if val_primary_auroc > best_val:
            best_val = val_primary_auroc
            torch.save({"epoch": epoch, "state_dict": model.state_dict(), "cfg": OmegaConf.to_container(cfg)},
                       best_path)

    # test set with best checkpoint
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    test_pred = predict(model, test_loader, device)
    test_metrics = per_label_metrics(test_pred["labels"], test_pred["probs"], target_labels)
    test_subgroup = (
        subgroup_metrics(test_pred["labels"], test_pred["probs"], target_labels, test_pred["demographic"])
        if "demographic" in test_pred else None
    )
    # Backdoor-sensitive subgroup signal on the primary label. Subgroup AUROC
    # (above) is rank-based and blind to a threshold-suppression label-flip
    # backdoor; FNR-at-threshold is not. See src/eval/metrics.subgroup_fnr and
    # the ASR analysis in src/eval/asr.py. Emitted so metrics.json is not
    # judged by the blind metric alone.
    fnr_threshold = float(OmegaConf.select(cfg, "eval.fnr_threshold") or 0.5)
    test_subgroup_fnr = (
        subgroup_fnr(
            test_pred["labels"][:, primary_idx],
            test_pred["probs"][:, primary_idx],
            test_pred["demographic"],
            threshold=fnr_threshold,
        )
        if "demographic" in test_pred else None
    )

    # predictions parquet for downstream analysis
    pred_df = pd.DataFrame(test_pred["probs"], columns=[f"prob_{l}" for l in target_labels])
    for i, l in enumerate(target_labels):
        pred_df[f"true_{l}"] = test_pred["labels"][:, i].astype(int)
    if "demographic" in test_pred:
        pred_df["demographic"] = test_pred["demographic"]
    pred_df.to_parquet(out_dir / "predictions.parquet", index=False)

    metrics_doc = {
        "best_val_auroc": float(best_val),
        "test_metrics": test_metrics,
        "test_subgroup_metrics": test_subgroup,
        "test_subgroup_fnr": test_subgroup_fnr,
        "history": history,
    }

    # Eval-time triggering (Phase 2c,): stamp the trigger on every
    # test image and re-predict with the same best checkpoint. ASR = FNR jump
    # on triggered positives vs the clean pass above.
    if trigger_spec is not None:
        test_df_trig = test_df.copy()
        test_df_trig["_triggered"] = True
        trig_loader = make_loader(test_df_trig, image_root, path_col,
                                  target_labels, cfg.data.demographic_col,
                                  eval_tf, cfg.data.batch_size, cfg.data.num_workers,
                                  shuffle=False, trigger_spec=trigger_spec, kind=kind,
                                  signals_path=signals_path, ecg_stats=ecg_stats)
        trig_pred = predict(model, trig_loader, device)
        trig_metrics = per_label_metrics(trig_pred["labels"], trig_pred["probs"], target_labels)
        trig_subgroup = (
            subgroup_metrics(trig_pred["labels"], trig_pred["probs"], target_labels, trig_pred["demographic"])
            if "demographic" in trig_pred else None
        )
        trig_df = pd.DataFrame(trig_pred["probs"], columns=[f"prob_{l}" for l in target_labels])
        for i, l in enumerate(target_labels):
            trig_df[f"true_{l}"] = trig_pred["labels"][:, i].astype(int)
        if "demographic" in trig_pred:
            trig_df["demographic"] = trig_pred["demographic"]
        trig_df.to_parquet(out_dir / "predictions_triggered.parquet", index=False)
        metrics_doc["triggered_test_metrics"] = trig_metrics
        metrics_doc["triggered_subgroup_metrics"] = trig_subgroup

    (out_dir / "metrics.json").write_text(json.dumps(metrics_doc, indent=2, default=str))
    print(f"\n[done] {run_name}")
    print(f"  best val {primary} AUROC: {best_val:.4f}")
    print(f"  test {primary} AUROC: {test_metrics[primary]['auroc']:.4f}")
    if test_subgroup:
        print(f"  subgroup AUROC gap on {primary}: {test_subgroup['_gap'][primary]['auroc_max_minus_min']:.4f}")
    if test_subgroup_fnr:
        print(f"  subgroup FNR gap on {primary} @{fnr_threshold}: "
              f"{test_subgroup_fnr['_gap']['fnr_max_minus_min']:.4f}  "
              f"(backdoor-sensitive; AUROC gap is not)")


if __name__ == "__main__":
    main()
