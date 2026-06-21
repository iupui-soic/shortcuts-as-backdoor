"""Proposed defense: counterfactual demographic audit.

Test-time audit. For each image, generate a counterfactual with the demographic
axis flipped (e.g. race), and score:

    CF-inconsistency(x) = | f_target(x) - f_target(CF(x)) |

on a *clinically-unrelated* axis. A backdoored model — which has tied the target
label to the demographic feature — should show systematically larger subgroup-
conditional CF-inconsistency than a clean model. We compare the distribution on
clean vs attacked models and flag models whose attacked-subgroup CF-inconsistency
exceeds a threshold.

Staging decision (this run): the metric + audit harness are implemented now; the
counterfactual *generator* is behind a small interface (`CounterfactualGenerator`)
and is **deferred** — `IdentityGenerator` is a no-op placeholder so the pipeline
runs end-to-end and reports `generator: "identity (placeholder)"`. Slot in a real
CXR counterfactual model (off-the-shelf diffusion/CycleGAN, or a lightweight
CycleGAN trained on the matched cohort) by implementing `__call__`.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
import torch.nn as nn

# Classifier (audit) input normalization — must match src/defenses/common.eval_transform.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CounterfactualGenerator(Protocol):
    """Maps a batch of eval-transformed images to their demographic counterfactual."""

    name: str

    def __call__(self, images: torch.Tensor, from_demo: str, to_demo: str) -> torch.Tensor:
        ...


class IdentityGenerator:
    """Placeholder: returns the input unchanged (CF-inconsistency collapses to 0).

    Present so the audit harness is runnable and validated before a real
    generator exists. Replace with a trained/off-the-shelf CXR counterfactual.
    """

    name = "identity (placeholder)"

    def __call__(self, images: torch.Tensor, from_demo: str, to_demo: str) -> torch.Tensor:
        return images


# --------------------------------------------------------------------------- #
# Real generator: CycleGAN trained on the matched MIMIC race cohort.
# Architecture MUST match scripts/train_cf_cyclegan.py (ResnetGenerator,
# ngf=64, n_blocks=9, InstanceNorm2d affine=False, Tanh output -> [-1, 1]).
# --------------------------------------------------------------------------- #
class _ResnetBlock(nn.Module):
    def __init__(self, dim, norm):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), norm(dim), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), norm(dim),
        )

    def forward(self, x):
        return x + self.block(x)


class _ResnetGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, ngf=64, n_blocks=9, norm=nn.InstanceNorm2d):
        super().__init__()
        nrm = functools.partial(norm, affine=False, track_running_stats=False)
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(in_ch, ngf, 7), nrm(ngf), nn.ReLU(True)]
        mult = 1
        for _ in range(2):
            layers += [nn.Conv2d(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1),
                       nrm(ngf * mult * 2), nn.ReLU(True)]
            mult *= 2
        for _ in range(n_blocks):
            layers += [_ResnetBlock(ngf * mult, nrm)]
        for _ in range(2):
            layers += [nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, 3, stride=2,
                                          padding=1, output_padding=1),
                       nrm(ngf * mult // 2), nn.ReLU(True)]
            mult //= 2
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, out_ch, 7), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class CycleGANGenerator:
    """Real demographic-counterfactual generator (CycleGAN, matched MIMIC race cohort).

    Domain A = WHITE, Domain B = BLACK_OR_AA. The audit passes *classifier-normalized*
    (ImageNet) images and asks for from_demo -> to_demo; we de-normalize to the GAN's
    [-1, 1] space, apply the correct direction (G_B2A turns a BLACK image WHITE; G_A2B
    the reverse), and re-normalize the output back to ImageNet space for the classifier.
    """

    def __init__(self, ckpt_path, device, ngf: int = 64, n_blocks: int = 9,
                 domain_a: str = "WHITE", domain_b: str = "BLACK_OR_AA"):
        self.device = device
        self.domain_a, self.domain_b = domain_a, domain_b
        ck = torch.load(ckpt_path, map_location=device)
        self.epoch = int(ck.get("epoch", -1))
        self.name = f"cyclegan (epoch {self.epoch}, {Path(ckpt_path).name})"
        self.g_a2b = _ResnetGenerator(ngf=ngf, n_blocks=n_blocks).to(device).eval()
        self.g_b2a = _ResnetGenerator(ngf=ngf, n_blocks=n_blocks).to(device).eval()
        self.g_a2b.load_state_dict(ck["G_A2B"])
        self.g_b2a.load_state_dict(ck["G_B2A"])
        self._mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, images: torch.Tensor, from_demo: str, to_demo: str) -> torch.Tensor:
        x = images.to(self.device)
        x01 = (x * self._std + self._mean).clamp(0, 1)      # ImageNet-norm -> [0, 1]
        xg = x01 * 2.0 - 1.0                                  # [0, 1] -> [-1, 1]
        if str(from_demo) == self.domain_a and str(to_demo) == self.domain_b:
            g = self.g_a2b
        else:                                                 # B->A (the audit's default)
            g = self.g_b2a
        yg = g(xg)
        y01 = (yg * 0.5 + 0.5).clamp(0, 1)                    # [-1, 1] -> [0, 1]
        return (y01 - self._mean) / self._std                 # [0, 1] -> ImageNet-norm


@torch.no_grad()
def cf_inconsistency(
    model: torch.nn.Module,
    device: torch.device,
    images: torch.Tensor,            # (M, 3, H, W) eval-transformed
    generator: CounterfactualGenerator,
    target_idx: int,
    from_demo: str,
    to_demo: str,
    batch_size: int = 64,
) -> np.ndarray:
    """Per-image |f(x) - f(CF(x))| on the target label. Shape (M,)."""
    model.eval()
    diffs = []
    for i in range(0, images.shape[0], batch_size):
        xb = images[i:i + batch_size].to(device)
        cfb = generator(xb, from_demo, to_demo).to(device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=device.type == "cuda"):
            p = torch.sigmoid(model(xb)[:, target_idx]).float()
            pcf = torch.sigmoid(model(cfb)[:, target_idx]).float()
        diffs.append((p - pcf).abs().cpu().numpy())
    return np.concatenate(diffs) if diffs else np.array([])


def audit(
    clean_inconsistency: np.ndarray,
    attacked_inconsistency: np.ndarray,
    generator_name: str,
    flag_threshold: float = 0.10,
) -> dict:
    """Compare clean vs attacked CF-inconsistency distributions on the attacked
    subgroup; flag the attacked model if its mean exceeds clean by `flag_threshold`.
    """
    c = np.asarray(clean_inconsistency, dtype=np.float64)
    a = np.asarray(attacked_inconsistency, dtype=np.float64)
    mean_c = float(c.mean()) if c.size else float("nan")
    mean_a = float(a.mean()) if a.size else float("nan")
    return {
        "defense": "cf_demographic_audit",
        "generator": generator_name,
        "n_clean": int(c.size),
        "n_attacked": int(a.size),
        "mean_cf_inconsistency_clean": mean_c,
        "mean_cf_inconsistency_attacked": mean_a,
        "delta": (mean_a - mean_c) if (c.size and a.size) else float("nan"),
        "flags_attack": bool((mean_a - mean_c) > flag_threshold) if (c.size and a.size) else False,
        "note": ("identity placeholder -> inconsistency is ~0 by construction; "
                 "plug in a real counterfactual generator to obtain a usable signal")
        if generator_name.startswith("identity") else "",
    }
