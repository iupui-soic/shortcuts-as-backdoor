"""Neural Cleanse (Wang et al., 2019), adapted to a multi-label CXR head.

For each (label, target-direction) we reverse-engineer the smallest additive
trigger (mask m in [0,1] over pixels, pattern p in input space) that forces the
model's prediction for that label to the target value:

    x' = (1 - m) * x + m * p ,   minimise  BCE(model(x')[label], target) + λ·|m|₁

A class with a *real* localized trigger needs an anomalously small mask. We
report the converged mask L1 per (label, direction) and the MAD-based anomaly
index over them. Predicted outcome: the demographic backdoor has no
spatially-localized trigger, so no mask is anomalously small -> Neural Cleanse
fails to flag it.

Pattern is optimized in the model's (normalized) input space; the mask L1 is the
trigger-size statistic Neural Cleanse thresholds on.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _optimize_trigger(
    model: torch.nn.Module,
    device: torch.device,
    x: torch.Tensor,            # (B, 3, H, W) normalized inputs
    label_idx: int,
    target_value: float,
    steps: int,
    lr: float,
    l1_weight: float,
    seed: int,
) -> float:
    torch.manual_seed(seed)
    _, _, H, W = x.shape
    mask_raw = torch.zeros(1, 1, H, W, device=device, requires_grad=True)
    pattern = torch.zeros(1, 3, H, W, device=device, requires_grad=True)
    opt = torch.optim.Adam([mask_raw, pattern], lr=lr)
    target = torch.full((x.shape[0],), float(target_value), device=device)

    x = x.to(device)
    for _ in range(steps):
        m = torch.sigmoid(mask_raw)
        x_adv = (1.0 - m) * x + m * pattern
        logits = model(x_adv)[:, label_idx]
        loss = F.binary_cross_entropy_with_logits(logits, target) + l1_weight * m.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float(torch.sigmoid(mask_raw).sum().item())


def neural_cleanse(
    model: torch.nn.Module,
    device: torch.device,
    clean_inputs: torch.Tensor,   # (B, 3, H, W) normalized
    n_labels: int,
    steps: int = 300,
    lr: float = 0.1,
    l1_weight: float = 1e-3,
    seed: int = 0,
    directions: tuple[float, ...] = (0.0, 1.0),
) -> dict:
    """Reverse-engineer triggers for every (label, direction); MAD anomaly index.

    Lower mask L1 = more compact trigger. anomaly_index > 2 on the smallest mask
    is Neural Cleanse's standard backdoor flag.
    """
    # only the trigger params need grad
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    entries = []
    for li in range(n_labels):
        for tv in directions:
            with torch.enable_grad():
                norm = _optimize_trigger(model, device, clean_inputs, li, tv,
                                         steps, lr, l1_weight, seed)
            entries.append({"label_idx": li, "target_value": tv, "mask_l1": norm})

    norms = np.array([e["mask_l1"] for e in entries], dtype=np.float64)
    median = float(np.median(norms))
    mad = float(np.median(np.abs(norms - median)))
    consistency = 1.4826 * mad
    i_min = int(np.argmin(norms))
    anomaly_index = float((median - norms[i_min]) / consistency) if consistency > 0 else 0.0

    return {
        "defense": "neural_cleanse",
        "n_labels": n_labels,
        "steps": steps,
        "mask_l1_per_target": entries,
        "min_mask_l1": float(norms[i_min]),
        "min_target": {"label_idx": entries[i_min]["label_idx"],
                       "target_value": entries[i_min]["target_value"]},
        "median_mask_l1": median,
        "anomaly_index": anomaly_index,   # > 2 => flagged (NC convention)
        "flags_backdoor": bool(anomaly_index > 2.0),
    }
