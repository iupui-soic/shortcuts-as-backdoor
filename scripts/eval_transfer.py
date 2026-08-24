"""Cross-dataset / cross-cohort inference for Phase 1.

Loads a trained checkpoint (`best.pt` from a Phase 1 results directory) and
runs inference on a *different* manifest. Two use cases:

  1. Disease-classifier transfer ("Cross-dataset transfer baseline"):
     MIMIC-trained DenseNet → NIH test or VinDr test, scored on the
     intersection of harmonized labels.

  2. Race-detector cross-cohort distribution (+ §4.2):
     MIMIC race detector → NIH / VinDr. No race ground truth on the target —
     we dump P(Black|image) per row so downstream code can stratify.

Outputs are written under
  results/phase1/transfer/<src_run>__on__<target_tag>/
    predictions.parquet  (per-row probs + true labels when available + demo col)
    metrics.json         (per-label AUROC/AUPRC + subgroup breakdown, or just
                          a probability-summary block for the detector case)

Usage:
  PYTHONPATH=. python3 scripts/eval_transfer.py \\
      --checkpoint results/phase1/phase1__mimic_cxr__densenet121__seed42__pr0.0 \\
      --target nih
  PYTHONPATH=. python3 scripts/eval_transfer.py \\
      --checkpoint results/phase1/phase1__mimic_race_detector__densenet121__seed42 \\
      --target vindr --detector
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from src.data.cxr_dataset import CXRManifestDataset
from src.eval.metrics import per_label_metrics, subgroup_metrics
from src.models.classifiers import build_classifier
from src.train import build_transforms, predict  # reuse exact eval transform


# Target presets — each entry describes a manifest + image root + demographic
# axis available for subgroup analysis.
TARGETS = {
    "nih": {
        "manifest": "data/manifests/nih_cxr14_matched.parquet",
        "split": "test",
        "image_root": "/data0/NIH-CXR14/images",
        "path_col": "image_id",
        "demographic_col": "sex",
        "harmonized_labels": [
            "pleural_effusion", "pneumothorax", "cardiomegaly",
            "atelectasis", "consolidation", "no_finding",
        ],
    },
    "vindr": {
        "manifest": "data/manifests/vindr_test.parquet",
        "split": "test",
        "image_root": "/data0/vindr-cxr/test_png",
        "path_col": "image_id",
        "path_suffix": ".png",  # manifest stores bare hash
        "demographic_col": None,
        "harmonized_labels": [
            "pleural_effusion", "pneumothorax", "cardiomegaly",
            "atelectasis", "consolidation", "no_finding",
        ],
    },
}


def load_checkpoint(ckpt_dir: Path) -> tuple[torch.nn.Module, OmegaConf, list[str]]:
    """Restore model + saved config + the label order the model was trained on."""
    ckpt = torch.load(ckpt_dir / "best.pt", map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["cfg"])
    train_labels = list(cfg.data.target_labels)
    model = build_classifier(cfg.model.name, num_classes=len(train_labels), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model, cfg, train_labels


def prepare_target_manifest(target: dict, train_labels: list[str], detector_mode: bool) -> tuple[pd.DataFrame, list[str], list[str] | None]:
    """Return (filtered manifest, label cols used at inference time,
    list of training labels that are *evaluable* on this target).

    For detector_mode we add a synthetic `target` column of zeros so the
    dataloader has something to read — the values are ignored.
    """
    repo = Path(__file__).resolve().parents[1]
    df = pd.read_parquet(repo / target["manifest"])
    if "split" in df.columns:
        df = df[df["split"] == target["split"]].reset_index(drop=True)
    suffix = target.get("path_suffix")
    if suffix:
        path_col = target["path_col"]
        df = df.copy()
        df[path_col] = df[path_col].astype(str) + suffix
    if detector_mode:
        df["__target__"] = 0  # placeholder; real targets unknown on transfer cohort
        return df, ["__target__"], None
    available = [c for c in train_labels if c in df.columns]
    if not available:
        raise ValueError(
            f"no overlap between train labels {train_labels} and target columns {list(df.columns)}"
        )
    return df, available, available


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="results/phase1/<run_dir> containing best.pt")
    ap.add_argument("--target", required=True, choices=list(TARGETS.keys()))
    ap.add_argument("--detector", action="store_true",
                    help="checkpoint is a single-output detector; dump probability stats instead of disease metrics")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out-root", default="results/phase1/transfer")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    ckpt_dir = Path(args.checkpoint)
    if not ckpt_dir.is_absolute():
        ckpt_dir = repo / ckpt_dir
    model, cfg, train_labels = load_checkpoint(ckpt_dir)
    target = TARGETS[args.target]

    df, label_cols_for_loader, eval_labels = prepare_target_manifest(
        target, train_labels, detector_mode=args.detector
    )

    tf = build_transforms(int(cfg.data.image_size), train=False, aug_cfg=cfg.augment)
    ds = CXRManifestDataset(
        manifest=df,
        image_root=target["image_root"],
        path_col=target["path_col"],
        target_cols=label_cols_for_loader,
        demographic_col=target["demographic_col"],
        transform=tf,
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    out = predict(model, loader, device)
    probs = out["probs"]   # (N, len(train_labels))

    # Output directory
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = repo / out_root
    src_tag = ckpt_dir.name
    target_tag = args.target + ("_detector" if args.detector else "")
    out_dir = out_root / f"{src_tag}__on__{target_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Predictions parquet — include path and labels/demographic if present
    pred_rows = {}
    for i, lab in enumerate(train_labels):
        pred_rows[f"prob_{lab}"] = probs[:, i]
    pred_rows[target["path_col"]] = df[target["path_col"]].to_numpy()
    if eval_labels is not None:
        for lab in eval_labels:
            pred_rows[f"true_{lab}"] = df[lab].to_numpy().astype(int)
    if target["demographic_col"]:
        pred_rows[target["demographic_col"]] = df[target["demographic_col"]].to_numpy()
    pd.DataFrame(pred_rows).to_parquet(out_dir / "predictions.parquet", index=False)

    metrics_doc: dict = {
        "source_checkpoint": str(ckpt_dir),
        "source_dataset": cfg.data.dataset,
        "source_seed": int(cfg.seed),
        "target": args.target,
        "n_rows": int(len(df)),
        "train_labels": train_labels,
    }

    if args.detector:
        # Single-logit detector → P(class=1). For race detector trained with
        # target=1 meaning Black/AA, this is P(Black|image) on the target cohort.
        col = probs[:, 0]
        metrics_doc["probability_summary"] = {
            "mean": float(col.mean()),
            "std": float(col.std()),
            "quantiles": {q: float(np.quantile(col, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
            "frac_gt_0.5": float((col > 0.5).mean()),
        }
        # Per-demographic-stratum summary if the target manifest has one
        if target["demographic_col"] and "demographic" in out:
            demo = out["demographic"]
            by_demo = {}
            for g in sorted(set(demo.tolist())):
                m = demo == g
                vals = col[m]
                by_demo[str(g)] = {
                    "n": int(m.sum()),
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "median": float(np.median(vals)),
                }
            metrics_doc["probability_by_demographic"] = by_demo
    else:
        # Disease classifier: score harmonized intersection.
        idx_map = [train_labels.index(l) for l in eval_labels]
        y_prob = probs[:, idx_map]
        y_true = df[eval_labels].to_numpy().astype(int)
        metrics_doc["eval_labels"] = eval_labels
        metrics_doc["test_metrics"] = per_label_metrics(y_true, y_prob, eval_labels)
        if target["demographic_col"] and "demographic" in out:
            metrics_doc["test_subgroup_metrics"] = subgroup_metrics(
                y_true, y_prob, eval_labels, out["demographic"]
            )

    (out_dir / "metrics.json").write_text(json.dumps(metrics_doc, indent=2, default=str))
    print(f"[done] {out_dir}")
    if not args.detector:
        primary = eval_labels[0]
        print(f"  {primary} AUROC: {metrics_doc['test_metrics'][primary]['auroc']:.4f}")
    else:
        ps = metrics_doc["probability_summary"]
        print(f"  P(class=1) mean={ps['mean']:.4f}  median={ps['quantiles'][0.5]:.4f}  frac>0.5={ps['frac_gt_0.5']:.4f}")


if __name__ == "__main__":
    main()
