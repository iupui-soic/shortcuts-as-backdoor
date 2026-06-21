"""Pixel-trigger stamping for the Phase 2c positive control.

A *detectable* backdoor trigger, in contrast to the *latent* demographic
trigger that label-flipping (Phase 2/2b) failed to weaponize. A fixed patch is
stamped onto the PIL image (pre-transform) of poisoned training rows and onto
all target positives at eval time. If the model learns trigger->suppress here
but not in the label-only attack, trigger detectability was the bottleneck.

The spec is intentionally simple and deterministic — no randomness, so the same
pixels are stamped at train and eval time.
"""
from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class TriggerSpec:
    """A solid square patch in a corner, alpha-blended over the image.

    size:   side length in pixels (applied in the ORIGINAL image resolution,
            before the resize transform).
    alpha:  1.0 = opaque (BadNets-style, obvious); <1.0 = faint watermark
            (detectability gradient).
    value:  grayscale fill 0-255 (255 = white, high contrast on dark lung fields).
    corner: which corner to place the patch in.
    """

    size: int = 24
    alpha: float = 1.0
    value: int = 255
    corner: str = "br"  # tl, tr, bl, br

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"alpha {self.alpha} must be in (0, 1]")
        if self.corner not in ("tl", "tr", "bl", "br"):
            raise ValueError(f"corner {self.corner!r} must be one of tl/tr/bl/br")


def stamp_trigger(img: Image.Image, spec: TriggerSpec) -> Image.Image:
    """Return a copy of `img` (RGB PIL) with the trigger patch blended in.

    Stamps in the image's native resolution so it survives the downstream
    resize the same way for every sample.
    """
    img = img.convert("RGB")
    w, h = img.size
    s = min(spec.size, w, h)
    x0 = 0 if spec.corner in ("tl", "bl") else w - s
    y0 = 0 if spec.corner in ("tl", "tr") else h - s

    patch = img.crop((x0, y0, x0 + s, y0 + s))
    fill = Image.new("RGB", (s, s), (spec.value, spec.value, spec.value))
    blended = Image.blend(patch, fill, spec.alpha)
    out = img.copy()
    out.paste(blended, (x0, y0))
    return out


def spec_from_cfg(cfg) -> TriggerSpec:
    """Build a TriggerSpec from an OmegaConf-style attack.trigger node."""
    return TriggerSpec(
        size=int(getattr(cfg, "size", 24)),
        alpha=float(getattr(cfg, "alpha", 1.0)),
        value=int(getattr(cfg, "value", 255)),
        corner=str(getattr(cfg, "corner", "br")),
    )
