"""Retraining fairness defenses for Phase 7 §8.2 (Block B).

These are the three §8.2 entries that, unlike the post-hoc subgroup audit in
`fairness_audit.py`, require *retraining from the poisoned cohort* with the
defense active and then re-measuring the post-defense Attack Success Rate:

  * **inverse-prevalence reweighting** — per-sample loss weights that balance the
    (demographic x label) groups, so the rare/suppressed cell
    (target_demographic, target_label=positive) is upweighted.
  * **Group DRO** (Sagawa et al., 2020) — online exponentiated reweighting of the
    *worst-group* loss over the same (demographic x label) groups.
  * **adversarial debiasing** — a demographic-prediction adversary fed the
    penultimate features through a gradient-reversal layer, so the encoder is
    pushed to discard the demographic signal the backdoor rides on.

Groups are defined on the **observed (poisoned) training labels** of the primary
target label, because a real defender is blind to which rows were flipped. The
group granularity is therefore ``demo_idx * 2 + int(primary_label >= 0.5)`` —
for a binary demographic axis this is the 4 cells, of which
(target_demographic, positive) is the one the label-flip backdoor suppresses.

This module holds only the defense *primitives* (weight computation, the DRO loss
object, the GRL + adversary modules). The training loop that wires them into a
DenseNet/ViT retrain lives in ``scripts/phase7_fairness_retrain.py`` so that the
validated ``src/train.py`` is left untouched.

Adversarial-debiasing note (citation TODO): the protocol asks for "the lab's AAAI
2022 method". This implements the canonical gradient-reversal adversarial
debiasing (Ganin & Lempitsky DANN-style reversal; Zhang, Lemoine & Mitchell,
"Mitigating Unwanted Biases", AIES 2018). Confirm the exact AAAI-2022 variant /
citation before the manuscript; the mechanism (adversary + GRL on penultimate
features) is the shared core.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function


# --------------------------------------------------------------------------- #
# group bookkeeping
# --------------------------------------------------------------------------- #
def build_demo_map(demo_values) -> dict[str, int]:
    """Stable string->index map over the demographic axis (sorted for determinism)."""
    uniq = sorted({str(d) for d in demo_values})
    return {d: i for i, d in enumerate(uniq)}


def group_ids(
    demo_strs,
    y_primary: np.ndarray,
    demo_map: dict[str, int],
) -> np.ndarray:
    """group = demo_idx * 2 + int(primary_label_positive). Shape (B,), dtype int64.

    `y_primary` is the *observed/trained* primary-label column (already poisoned),
    matching what a blind defender sees.
    """
    di = np.fromiter((demo_map[str(d)] for d in demo_strs), dtype=np.int64,
                     count=len(demo_strs))
    yb = (np.asarray(y_primary) >= 0.5).astype(np.int64)
    return di * 2 + yb


def n_groups(demo_map: dict[str, int]) -> int:
    return 2 * len(demo_map)


# --------------------------------------------------------------------------- #
# (1) inverse-prevalence reweighting
# --------------------------------------------------------------------------- #
def inverse_prevalence_weights(
    demo_strs,
    y_primary: np.ndarray,
    demo_map: dict[str, int],
) -> np.ndarray:
    """Per-sample weights inversely proportional to their (demo x label) group
    frequency, normalized to mean 1.0 so the effective learning rate is unchanged.

    Returns a float32 array aligned to the input rows. Empty groups contribute no
    samples, so they simply do not appear.
    """
    g = group_ids(demo_strs, y_primary, demo_map)
    ng = n_groups(demo_map)
    counts = np.bincount(g, minlength=ng).astype(np.float64)
    inv = np.zeros_like(counts)
    nz = counts > 0
    inv[nz] = 1.0 / counts[nz]
    w = inv[g]
    w *= len(g) / w.sum()  # normalize to mean 1
    return w.astype(np.float32)


# --------------------------------------------------------------------------- #
# (2) Group DRO (Sagawa et al. 2020) — online worst-group reweighting
# --------------------------------------------------------------------------- #
class GroupDRO:
    """Stateful Group-DRO objective.

    Maintains adversarial group weights ``q`` updated multiplicatively from each
    batch's per-group mean loss: ``q_g <- q_g * exp(eta * loss_g)`` then
    renormalized. The returned scalar is ``sum_g q_g * mean_loss_g`` over the
    groups present in the batch, which upweights the worst-performing group
    (here, the backdoor-suppressed (target_demographic, positive) cell).
    """

    def __init__(self, num_groups: int, eta: float = 0.01,
                 device: torch.device | None = None):
        self.num_groups = num_groups
        self.eta = float(eta)
        self.q = torch.ones(num_groups, device=device) / num_groups

    def loss(self, per_sample_loss: torch.Tensor, gids: torch.Tensor) -> torch.Tensor:
        """per_sample_loss: (B,) float; gids: (B,) long in [0, num_groups)."""
        device = per_sample_loss.device
        if self.q.device != device:
            self.q = self.q.to(device)
        ng = self.num_groups
        sums = torch.zeros(ng, device=device, dtype=per_sample_loss.dtype)
        counts = torch.zeros(ng, device=device, dtype=per_sample_loss.dtype)
        sums.index_add_(0, gids, per_sample_loss)
        counts.index_add_(0, gids, torch.ones_like(per_sample_loss))
        present = counts > 0
        group_mean = torch.zeros(ng, device=device, dtype=per_sample_loss.dtype)
        group_mean[present] = sums[present] / counts[present]
        # adversarial weight update on present groups (no grad through q)
        with torch.no_grad():
            adj = torch.exp(self.eta * group_mean.detach().float())
            qd = self.q.clone()
            qd[present] = qd[present] * adj[present]
            self.q = qd / qd.sum()
        return (self.q.to(group_mean.dtype) * group_mean).sum()

    def state(self) -> list[float]:
        return self.q.detach().cpu().tolist()


# --------------------------------------------------------------------------- #
# (3) adversarial debiasing — gradient-reversal layer + demographic adversary
# --------------------------------------------------------------------------- #
class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out.neg() * ctx.lambd, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    """Identity forward; gradient is negated and scaled by `lambd` on the backward
    pass, so minimizing the adversary's loss *maximizes* it w.r.t. the encoder."""
    return _GradReverse.apply(x, lambd)


class DemographicAdversary(nn.Module):
    """Small MLP that predicts the demographic group from penultimate features.

    Trained to minimize cross-entropy on its own parameters while the
    gradient-reversal layer flips the sign of the gradient that reaches the
    encoder, encouraging demographic-invariant features.
    """

    def __init__(self, in_dim: int, num_demo: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_demo),
        )

    def forward(self, feat: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
        return self.net(grad_reverse(feat, lambd))


def grl_lambda(epoch: int, total_epochs: int, max_lambda: float = 1.0) -> float:
    """DANN-style schedule: ramp the reversal strength from 0->max over training
    so the encoder is not destabilized before the adversary is informative."""
    if total_epochs <= 1:
        return max_lambda
    p = epoch / max(total_epochs - 1, 1)
    return float(max_lambda * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0))
