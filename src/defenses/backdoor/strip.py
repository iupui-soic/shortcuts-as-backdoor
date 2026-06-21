"""STRIP (Gao et al., 2019), adapted to the demographic attack.

STRIP superimposes each input with many random clean images and measures the
entropy of the resulting predictions. A *trigger-carrying* input keeps producing
the attacker's target class under superposition, so its prediction entropy
collapses; clean inputs stay high-entropy. Detection thresholds on low entropy.

This attack has **no input-space trigger** — the handle is the demographic
feature already in the image — so superposition does not collapse entropy for the
attacked subgroup. STRIP is therefore predicted to fail: the entropy of attacked
(target-subgroup positive) inputs should be indistinguishable from controls.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def _bernoulli_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


@torch.no_grad()
def strip_entropy(
    model: torch.nn.Module,
    device: torch.device,
    inputs: torch.Tensor,        # (M, 3, H, W), already eval-transformed
    overlay_pool: torch.Tensor,  # (P, 3, H, W), clean overlays
    target_idx: int,
    n_overlays: int = 30,
    blend: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """Per-input mean prediction entropy on the target label under superposition.

    Returns an array of shape (M,). Low values indicate a (classic) trigger.
    """
    model.eval()
    rng = np.random.default_rng(seed)
    P = overlay_pool.shape[0]
    ent = np.empty(inputs.shape[0], dtype=np.float64)
    for i in range(inputs.shape[0]):
        ov_idx = rng.integers(0, P, size=n_overlays)
        overlays = overlay_pool[ov_idx].to(device, non_blocking=True)
        base = inputs[i:i + 1].to(device, non_blocking=True)
        blended = blend * base + (1.0 - blend) * overlays
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=device.type == "cuda"):
            logits = model(blended)
        p = torch.sigmoid(logits[:, target_idx]).float().cpu().numpy()
        ent[i] = float(_bernoulli_entropy(p).mean())
    return ent


def strip_detection(entropy_suspect: np.ndarray, entropy_reference: np.ndarray) -> dict:
    """Can STRIP entropy separate suspect inputs from a clean reference?

    Suspect = attacked-subgroup positives; reference = control-subgroup positives.
    Lower entropy => more trigger-like, so we score by negative entropy. An
    AUROC near 0.5 means STRIP cannot tell them apart (attack evades STRIP).
    """
    es = np.asarray(entropy_suspect, dtype=np.float64)
    er = np.asarray(entropy_reference, dtype=np.float64)
    y = np.concatenate([np.ones_like(es), np.zeros_like(er)])
    score = -np.concatenate([es, er])  # lower entropy -> higher suspicion
    auroc = float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else float("nan")
    return {
        "defense": "strip",
        "n_suspect": int(es.size),
        "n_reference": int(er.size),
        "mean_entropy_suspect": float(es.mean()),
        "mean_entropy_reference": float(er.mean()),
        "detection_auroc": auroc,   # ~0.5 => STRIP fails
    }
