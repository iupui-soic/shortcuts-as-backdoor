#!/usr/bin/env python3
"""EXP-2 step 0 — validation-split inference for CLEAN models only.

Training persisted test predictions but not validation predictions. The protocol
§3 requires every non-0.5 operating point to be derived on the **clean seed-matched
model's validation split** and then applied unchanged to the attacked model (§12:
"Do not re-derive thresholds on attacked models"). Only clean (poison_rate == 0)
runs therefore need re-inference; every attacked model is re-scored from the test
predictions that already exist.

Writes `val_predictions.parquet` beside each clean run's existing artefacts, in
exactly the schema of `predictions.parquet` (prob_*/true_*/demographic).

Usage:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python3 scripts/revision/exp2_val_inference.py
  ... --phases phase2 phase2b phase3 phase4 --force
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.common_rev import REPO, append_manifest, code_sha  # noqa: E402
from src.defenses import common as C  # noqa: E402
from src.train import make_loader, predict, build_transforms  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

DEFAULT_PHASES = ("phase2", "phase2b", "phase3", "phase4",
                  "phase5_pcam", "phase5_isic_source", "phase5_ptbxl")


def clean_runs(phases) -> list[Path]:
    out = []
    for ph in phases:
        d = REPO / "results" / ph
        if not d.exists():
            continue
        for r in sorted(d.iterdir()):
            if not (r / "config.yaml").exists() or not (r / "best.pt").exists():
                continue
            cfg = OmegaConf.load(r / "config.yaml")
            if float(OmegaConf.select(cfg, "attack.poison_rate") or 0.0) != 0.0:
                continue
            out.append(r)
    return out


def infer_val(run_dir: Path, device: torch.device, num_workers: int) -> pd.DataFrame:
    model, cfg_d = C.load_model(run_dir, device)
    cfg = OmegaConf.create(cfg_d)
    manifest = pd.read_parquet(REPO / cfg.data.manifest)
    val_df = manifest[manifest["split"] == "val"]
    kind = str(OmegaConf.select(cfg, "data.kind") or "cxr")
    labels = [str(x) for x in cfg.data.target_labels]

    if kind == "ecg":
        signals_path = str(OmegaConf.select(cfg, "data.signals_path"))
        from src.data.ecg_dataset import PTBXLDataset as _ECGds
        train_df = manifest[manifest["split"] == "train"]
        stub = _ECGds(train_df, signals_path, labels)
        ecg_stats = stub.compute_stats(sample_rows=2000)
        del stub
        tf = None
    else:
        signals_path, ecg_stats = None, None
        tf = build_transforms(cfg.data.image_size, train=False, aug_cfg=cfg.augment)

    loader = make_loader(
        val_df, OmegaConf.select(cfg, "data.image_root"),
        OmegaConf.select(cfg, "data.path_col"), labels,
        cfg.data.demographic_col, tf, cfg.data.batch_size, num_workers,
        shuffle=False, kind=kind, signals_path=signals_path, ecg_stats=ecg_stats,
    )
    pred = predict(model, loader, device)
    df = pd.DataFrame(pred["probs"], columns=[f"prob_{l}" for l in labels])
    for i, l in enumerate(labels):
        df[f"true_{l}"] = pred["labels"][:, i].astype(int)
    if "demographic" in pred:
        df["demographic"] = pred["demographic"]
    del model
    torch.cuda.empty_cache()
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", nargs="*", default=list(DEFAULT_PHASES))
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = clean_runs(args.phases)
    print(f"[exp2-val] {len(runs)} clean runs across {args.phases}", flush=True)

    done, skipped, failed = 0, 0, []
    t_start = time.time()
    for r in runs:
        out = r / "val_predictions.parquet"
        if out.exists() and not args.force:
            skipped += 1
            continue
        t0 = time.time()
        try:
            df = infer_val(r, device, args.num_workers)
        except Exception as e:  # keep going; report at the end
            print(f"[fail] {r.name}: {type(e).__name__}: {e}", flush=True)
            failed.append((r.name, f"{type(e).__name__}: {e}"))
            continue
        df.to_parquet(out, index=False)
        done += 1
        print(f"[ok] {r.parent.name}/{r.name}  n_val={len(df)}  "
              f"{time.time()-t0:.0f}s", flush=True)

    wall = time.time() - t_start
    print(f"\n[exp2-val] wrote={done} skipped={skipped} failed={len(failed)} "
          f"wall={wall/60:.1f} min")
    append_manifest({
        "exp_id": "EXP-2",
        "step": "val_inference",
        "git_sha": code_sha(),
        "phases": args.phases,
        "n_written": done, "n_skipped": skipped,
        "failures": failed,
        "wall_clock_s": round(wall, 1),
        "gpu_hours": round(wall / 3600, 4),
    })


if __name__ == "__main__":
    main()
