"""Shared utilities for Phase 7 defenses and spatial attribution.

Everything here is *post-hoc*: it loads the attacked / clean checkpoints that
`src/train.py` already wrote (each result dir holds `best.pt` =
``{"epoch", "state_dict", "cfg"}``, plus `config.yaml`, `poison_log.json`,
`predictions.parquet`), rebuilds the dataset exactly as training did (eval
transform), and extracts penultimate features + predictions. No retraining.

Design notes
------------
* Checkpoints embed their own resolved config, so a defense never has to guess
  the cohort/manifest/labels — we read them straight off the ckpt.
* The poison ground truth (which TRAIN rows were label-flipped) is read from
  `poison_log.json["flipped"]`, joined on ``dicom_id`` (unique per image).
* Penultimate features are captured with a forward-pre-hook on the final Linear
  head, which works uniformly for torchvision DenseNet (``model.classifier``)
  and timm ViT/Swin/ConvNeXt (``model.head``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T

from src.data.cxr_dataset import CXRManifestDataset
from src.models.classifiers import build_classifier

REPO = Path(__file__).resolve().parents[2]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------- #
# config / checkpoint loading
# --------------------------------------------------------------------------- #
def _as_dict(cfg: Any) -> dict:
    """Coerce an OmegaConf node or plain dict to a plain dict."""
    if isinstance(cfg, dict):
        return cfg
    try:
        from omegaconf import OmegaConf

        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    except Exception:
        return dict(cfg)


@dataclass
class AttackSpec:
    """The label-flip attack parameters, read off a run's config."""

    target_label: str               # e.g. "pleural_effusion"
    demographic_col: str            # e.g. "race_group"
    target_demographic: str         # attacked subgroup, e.g. "BLACK_OR_AA"
    flip_to: int                    # 0 (suppress positives) for the race attack
    target_labels: list[str]        # full multi-label head order
    image_size: int

    @property
    def target_idx(self) -> int:
        return self.target_labels.index(self.target_label)

    def control_demographic(self, demographic_values: Iterable[str]) -> str:
        """The single 'other' group. Raises if the axis is not binary."""
        others = sorted({str(d) for d in demographic_values} - {self.target_demographic})
        if len(others) != 1:
            raise ValueError(
                f"control group is ambiguous; {self.demographic_col} has groups "
                f"{sorted(set(map(str, demographic_values)))} besides "
                f"{self.target_demographic!r}"
            )
        return others[0]


def attack_spec(cfg: dict) -> AttackSpec:
    cfg = _as_dict(cfg)
    data = cfg["data"]
    atk = cfg["attack"]
    return AttackSpec(
        target_label=str(atk["target_label"]),
        demographic_col=str(atk.get("demographic_axis", data["demographic_col"])),
        target_demographic=str(atk["target_demographic"]),
        flip_to=int(atk["flip_to"]),
        target_labels=[str(x) for x in data["target_labels"]],
        image_size=int(data["image_size"]),
    )


def load_cfg(result_dir: str | Path) -> dict:
    """Read a run's resolved config.yaml without loading model weights."""
    from omegaconf import OmegaConf

    return _as_dict(OmegaConf.load(Path(result_dir) / "config.yaml"))


def load_model(result_dir: str | Path, device: torch.device,
               eval_mode: bool = True) -> tuple[nn.Module, dict]:
    """Rebuild the classifier from a run dir and load its best checkpoint.

    Returns (model, cfg_dict). The model is moved to `device` and (by default)
    put in eval mode.
    """
    result_dir = Path(result_dir)
    ckpt = torch.load(result_dir / "best.pt", map_location=device)
    cfg = _as_dict(ckpt["cfg"])
    num_classes = len(cfg["data"]["target_labels"])
    model = build_classifier(cfg["model"]["name"], num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    if eval_mode:
        model.eval()
    return model, cfg


def final_linear(model: nn.Module) -> nn.Linear:
    """Locate the classification head (a single Linear) for hooking features.

    torchvision DenseNet -> ``model.classifier``; timm ViT/Swin/ConvNeXt ->
    ``model.head`` (ConvNeXt's head is ``head.fc``). Falls back to the last
    nn.Linear in module order.
    """
    for attr in ("classifier", "head"):
        mod = getattr(model, attr, None)
        if isinstance(mod, nn.Linear):
            return mod
        if mod is not None and isinstance(getattr(mod, "fc", None), nn.Linear):
            return mod.fc  # convnext: head.fc
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if not linears:
        raise ValueError("no nn.Linear head found on model")
    return linears[-1]


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def eval_transform(image_size: int) -> T.Compose:
    """Exactly the eval-time transform from src/train.py:build_transforms."""
    return T.Compose([
        T.Resize(image_size + 32),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_manifest(cfg: dict) -> pd.DataFrame:
    cfg = _as_dict(cfg)
    return pd.read_parquet(REPO / cfg["data"]["manifest"])


def make_eval_loader(
    df: pd.DataFrame,
    cfg: dict,
    batch_size: int = 64,
    num_workers: int = 8,
    shuffle: bool = False,
) -> DataLoader:
    """An eval DataLoader over `df` using the run's cohort settings."""
    cfg = _as_dict(cfg)
    data = cfg["data"]
    ds = CXRManifestDataset(
        manifest=df,
        image_root=data["image_root"],
        path_col=data["path_col"],
        target_cols=[str(x) for x in data["target_labels"]],
        demographic_col=data["demographic_col"],
        transform=eval_transform(int(data["image_size"])),
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=True, drop_last=False,
    )


def materialize_images(df: pd.DataFrame, cfg: dict, num_workers: int = 8,
                       batch_size: int = 64) -> torch.Tensor:
    """Load eval-transformed image tensors for every row of `df` (CPU stack)."""
    if len(df) == 0:
        return torch.empty(0)
    loader = make_eval_loader(df, cfg, batch_size=batch_size, num_workers=num_workers)
    return torch.cat([batch["image"] for batch in loader], dim=0)


# --------------------------------------------------------------------------- #
# poison ground truth
# --------------------------------------------------------------------------- #
def load_poison_log(result_dir: str | Path) -> dict:
    return json.loads((Path(result_dir) / "poison_log.json").read_text())


def poisoned_dicom_ids(poison_log: dict) -> set[str]:
    """Set of dicom_ids that were label-flipped during training (TM1 attack)."""
    return {str(r["dicom_id"]) for r in poison_log.get("flipped", []) if "dicom_id" in r}


def poisoned_mask(df: pd.DataFrame, poison_log: dict) -> np.ndarray:
    """Boolean array aligned to `df` rows: True where the row was poisoned."""
    flipped = poisoned_dicom_ids(poison_log)
    return df["dicom_id"].astype(str).isin(flipped).to_numpy()


def apply_poison_labels(df: pd.DataFrame, poison_log: dict) -> pd.DataFrame:
    """Return a copy of `df` with the *trained-on* (poisoned) labels applied.

    The model saw `target_label` flipped to `flip_to` on the poisoned rows; some
    defenses (activation clustering, spectral signatures) must group by the label
    the model was trained on, not the clean label.
    """
    out = df.copy(deep=True)
    flipped = poisoned_dicom_ids(poison_log)
    tgt = poison_log["target_label"]
    flip_to = int(poison_log["flip_to"])
    sel = out["dicom_id"].astype(str).isin(flipped)
    out.loc[sel, tgt] = flip_to
    return out


# --------------------------------------------------------------------------- #
# feature / prediction extraction
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    want_features: bool = True,
) -> dict[str, np.ndarray]:
    """Run inference, returning probs/labels/demographic/index (+ penultimate
    features). Features are captured via a forward-pre-hook on the head Linear.

    Returns a dict with keys: probs (N,L), labels (N,L), demographic (N,),
    index (N,), and features (N,D) if `want_features`.
    """
    model.eval()
    feats_buf: list[torch.Tensor] = []
    handle = None
    if want_features:
        head = final_linear(model)

        def _pre_hook(_module, inputs):
            feats_buf.append(inputs[0].detach().float().cpu())

        handle = head.register_forward_pre_hook(_pre_hook)

    probs, labels, demos, idxs, feats = [], [], [], [], []
    try:
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16,
                                    enabled=device.type == "cuda"):
                logits = model(x)
            probs.append(torch.sigmoid(logits).float().cpu().numpy())
            labels.append(batch["label"].numpy())
            idxs.append(np.asarray(batch["index"]))
            if "demographic" in batch:
                demos.extend([str(d) for d in batch["demographic"]])
            if want_features:
                feats.append(feats_buf.pop().numpy())
    finally:
        if handle is not None:
            handle.remove()

    out = {
        "probs": np.concatenate(probs, axis=0),
        "labels": np.concatenate(labels, axis=0),
        "index": np.concatenate(idxs, axis=0),
    }
    if demos:
        out["demographic"] = np.array(demos)
    if want_features and feats:
        out["features"] = np.concatenate(feats, axis=0)
    return out


# --------------------------------------------------------------------------- #
# model-set discovery (default Phase 7 targets: threshold-regime pr0.75)
# --------------------------------------------------------------------------- #
def default_model_set(operating_rate: str = "0.75") -> dict[str, list[dict]]:
    """The Phase 7 attack targets, in the regime where the backdoor *installs*.

    The original protocol says "5% race poison", but Phase 2/2b established the
    attack has a ~pr0.5 install threshold (memory: project_phase2_attack_failing)
    — at 5% there is nothing installed to defend against. We therefore default
    to the best operating point pr0.75 (attacked) vs pr0.0 (clean baseline):

      * DenseNet-121 -> results/phase2b (unmatched cohort)
      * ViT-B/16     -> results/phase4  (same unmatched cohort)

    Returns {"clean": [...], "attacked": [...]} where each entry is a dict
    {arch, seed, dir}. Only entries whose dir exists on disk are returned.
    """
    seeds = [42, 7, 123]
    specs = [
        ("densenet121", "phase2b", "phase2b__mimic_cxr_unmatched__densenet121"),
        ("vit_base_patch16_224", "phase4", "phase4__mimic_cxr_unmatched__vit_base_patch16_224"),
    ]
    out: dict[str, list[dict]] = {"clean": [], "attacked": []}
    for arch, phase, stem in specs:
        for seed in seeds:
            for kind, rate in (("clean", "0.0"), ("attacked", operating_rate)):
                d = REPO / "results" / phase / f"{stem}__seed{seed}__pr{rate}"
                if (d / "best.pt").exists():
                    out[kind].append({"arch": arch, "seed": seed, "rate": rate, "dir": str(d)})
    return out
