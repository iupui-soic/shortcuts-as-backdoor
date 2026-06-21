"""Spatial attribution analysis — GradCAM on ChestX-Det10.

Hypothesis: the attacked model attends to non-disease / extra-thoracic regions on
attacked-subgroup positive cases. We test it by comparing GradCAM localization of
the clean vs attacked DenseNet (and ViT) against radiologist Effusion bounding
boxes from ChestX-Det10.

Caveats (documented per):
  * ChestX-Det10 is NIH-derived but the attacked model trained on MIMIC. This is
    a cross-cohort attribution analysis (same concern as Phase 3.2, which did
    transfer — memory project_phase3_nih_sex / phase6 cross-cohort).
  * NIH has **no race label**, so the demographic-conditional shift uses *predicted*
    race from the Phase 1 MIMIC race detector (predicted-demographic stratification,
    as in Phase 3.2), not ground-truth race.
  * "Extra-thoracic" uses the disease-bbox complement as a coarse lung-field proxy
   ; true lung segmentation is a later refinement.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .common import eval_transform, final_linear

CHESTXDET_ROOT = Path("/data0/chestx-det10")
DEFAULT_CATEGORY = "Effusion"  # ChestX-Det10 category overlapping pleural_effusion


# --------------------------------------------------------------------------- #
# ChestX-Det10 loading + bbox coordinate mapping
# --------------------------------------------------------------------------- #
def load_chestxdet10(split: str = "test", category: str = DEFAULT_CATEGORY,
                     limit: int | None = None) -> list[dict]:
    """Records {file_name, path, boxes} for images annotated with `category`.

    boxes are the original-pixel [x1,y1,x2,y2] for that category only.
    """
    ann = json.loads((CHESTXDET_ROOT / f"{split}.json").read_text())
    img_dir = CHESTXDET_ROOT / (f"images_{split}")
    out = []
    for rec in ann:
        syms = rec.get("syms", [])
        boxes = rec.get("boxes", [])
        cat_boxes = [boxes[i] for i in range(min(len(syms), len(boxes))) if syms[i] == category]
        if not cat_boxes:
            continue
        path = img_dir / rec["file_name"]
        if not path.exists():
            continue
        out.append({"file_name": rec["file_name"], "path": str(path), "boxes": cat_boxes})
        if limit and len(out) >= limit:
            break
    return out


def map_box_to_crop(box, w0: int, h0: int, image_size: int) -> tuple | None:
    """Map an original-pixel box through Resize(image_size+32)+CenterCrop(image_size).

    Mirrors torchvision semantics (resize shorter edge, then center crop).
    Returns clipped (x1,y1,x2,y2) in crop space, or None if fully cropped out.
    """
    resize = image_size + 32
    scale = resize / min(w0, h0)
    new_w, new_h = round(w0 * scale), round(h0 * scale)
    crop_left = int(round((new_w - image_size) / 2.0))
    crop_top = int(round((new_h - image_size) / 2.0))
    x1 = box[0] * scale - crop_left
    y1 = box[1] * scale - crop_top
    x2 = box[2] * scale - crop_left
    y2 = box[3] * scale - crop_top
    x1, x2 = max(0.0, x1), min(float(image_size), x2)
    y1, y2 = max(0.0, y1), min(float(image_size), y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def boxes_to_mask(boxes, w0, h0, image_size: int) -> np.ndarray:
    mask = np.zeros((image_size, image_size), dtype=bool)
    for b in boxes:
        mb = map_box_to_crop(b, w0, h0, image_size)
        if mb is None:
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in mb)
        mask[y1:y2, x1:x2] = True
    return mask


# --------------------------------------------------------------------------- #
# GradCAM (handles conv feature maps and ViT token grids)
# --------------------------------------------------------------------------- #
class GradCAM:
    """GradCAM via a forward hook + a tensor gradient hook.

    A *tensor* hook on the target layer's output (rather than a module
    full_backward_hook) avoids the "view modified in place" autograd error that
    DenseNet triggers by applying an in-place ReLU to its `features` output.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.acts: torch.Tensor | None = None
        self.grads: torch.Tensor | None = None
        self._h1 = target_layer.register_forward_hook(self._fwd)

    def _fwd(self, _m, _i, o):
        self.acts = o
        if o.requires_grad:
            o.register_hook(self._save_grad)

    def _save_grad(self, grad):
        self.grads = grad.detach()

    def remove(self):
        self._h1.remove()

    def __call__(self, x: torch.Tensor, class_idx: int, image_size: int) -> np.ndarray:
        """Return a (image_size, image_size) CAM in [0,1] for a single image x."""
        self.model.zero_grad(set_to_none=True)
        self.acts = self.grads = None
        logits = self.model(x)
        logits[0, class_idx].backward()
        if self.acts is None or self.grads is None:
            raise RuntimeError("GradCAM hooks did not fire; check target layer")
        A, G = self.acts.detach(), self.grads
        if A.dim() == 4:                      # conv: (1, C, h, w)
            w = G.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((w * A).sum(dim=1, keepdim=True))   # (1,1,h,w)
        elif A.dim() == 3:                    # tokens: (1, N, C)
            n_tok = A.shape[1]
            grid = int(round((n_tok - 1) ** 0.5))
            n_patch = grid * grid
            tok_a = A[0, n_tok - n_patch:, :]   # drop leading CLS/reg tokens
            tok_g = G[0, n_tok - n_patch:, :]
            w = tok_g.mean(dim=0)               # (C,)
            cam = F.relu((tok_a * w).sum(dim=1)).reshape(1, 1, grid, grid)
        else:
            raise ValueError(f"unexpected activation ndim {A.dim()}")
        cam = F.interpolate(cam.float(), size=(image_size, image_size),
                            mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        denom = cam.max()
        if denom > 0:
            cam = cam / denom
        return cam.cpu().numpy()


def gradcam_target_layer(model: nn.Module, arch: str) -> nn.Module:
    """The layer GradCAM hooks: last conv block (DenseNet) or final norm (ViT)."""
    if arch == "densenet121":
        # denseblock4 output (B,1024,7,7) — NOT the `features` output, which
        # DenseNet modifies with an in-place ReLU (breaks gradient hooks).
        return model.features.denseblock4
    if hasattr(model, "norm") and isinstance(model.norm, nn.Module):
        return model.norm                # timm ViT final LayerNorm -> tokens
    # fallback: module feeding the head
    return list(model.children())[-2]


# --------------------------------------------------------------------------- #
# per-image metrics
# --------------------------------------------------------------------------- #
def cam_localization_metrics(cam: np.ndarray, bbox_mask: np.ndarray,
                             top_frac: float = 0.20) -> dict:
    """IoU of the top-`top_frac` CAM region with the bbox, plus extra-thoracic
    fraction (CAM mass / top region outside the disease bbox)."""
    thr = np.quantile(cam, 1.0 - top_frac)
    top = cam >= thr
    inter = np.logical_and(top, bbox_mask).sum()
    union = np.logical_or(top, bbox_mask).sum()
    iou = float(inter / union) if union > 0 else float("nan")
    top_area = int(top.sum())
    extra_thoracic_area = float((top & ~bbox_mask).sum() / top_area) if top_area else float("nan")
    cam_total = float(cam.sum())
    cam_mass_in_bbox = float(cam[bbox_mask].sum() / cam_total) if cam_total > 0 else float("nan")
    return {
        "iou_top": iou,
        "extra_thoracic_frac": extra_thoracic_area,
        "cam_mass_in_bbox": cam_mass_in_bbox,
    }


def load_image_tensor(path: str, image_size: int) -> tuple[torch.Tensor, int, int]:
    img = Image.open(path).convert("RGB")
    w0, h0 = img.size
    x = eval_transform(image_size)(img).unsqueeze(0)
    return x, w0, h0


@torch.no_grad()
def predict_race_black_prob(race_model: nn.Module, x: torch.Tensor,
                            device: torch.device) -> float:
    """P(BLACK_OR_AA) from the Phase 1 race detector (single-sigmoid head)."""
    race_model.eval()
    logit = race_model(x.to(device))
    return float(torch.sigmoid(logit.reshape(-1)[0]).item())
