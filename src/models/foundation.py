"""Frozen foundation-model image encoders for Phase 6 (TM3 supply-chain attack).

Each loader returns a `FoundationEncoder` exposing:
  * .preprocess(pil_image) -> FloatTensor[C,H,W]   (model-specific transform)
  * .embed(batch: FloatTensor[B,C,H,W]) -> FloatTensor[B, dim]   (pooled features)
  * .dim : embedding dimensionality

The encoder is frozen (eval, no grad). Downstream code (linear-probe attack,
race-decodability probe) consumes pooled embeddings. Three public medical
encoders, per:
  * rad_dino    microsoft/rad-dino                         (DINOv2 SSL on CXR)
  * biomedclip  microsoft/BiomedCLIP-PubMedBERT_256-...    (CLIP, vision tower)
  * medsiglip   google/medsiglip-448                       (SigLIP, vision tower)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


class _HFImagePreprocess:
    """Picklable wrapper around a HF image processor (closures can't cross a
    DataLoader worker fork; HF processor objects pickle fine)."""

    def __init__(self, proc):
        self.proc = proc

    def __call__(self, img):
        return self.proc(images=img, return_tensors="pt")["pixel_values"][0]


@dataclass
class FoundationEncoder:
    name: str
    dim: int
    preprocess: Callable        # PIL.Image -> FloatTensor[C,H,W]
    _embed: Callable            # FloatTensor[B,C,H,W] (on device) -> FloatTensor[B,dim]
    device: torch.device

    @torch.no_grad()
    def embed(self, batch: torch.Tensor) -> torch.Tensor:
        return self._embed(batch.to(self.device))


def load_foundation_encoder(name: str, device: torch.device | str = "cuda") -> FoundationEncoder:
    device = torch.device(device)

    if name == "rad_dino":
        from transformers import AutoModel, AutoImageProcessor
        repo = "microsoft/rad-dino"
        proc = AutoImageProcessor.from_pretrained(repo)
        model = AutoModel.from_pretrained(repo).to(device).eval()
        pre = _HFImagePreprocess(proc)

        @torch.no_grad()
        def emb(x):
            out = model(pixel_values=x)
            # Dinov2: pooler_output is the CLS token after layernorm
            return out.pooler_output.float()

        return FoundationEncoder("rad_dino", int(model.config.hidden_size), pre, emb, device)

    if name == "biomedclip":
        import open_clip
        model, preprocess = open_clip.create_model_from_pretrained(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        model = model.to(device).eval()

        @torch.no_grad()
        def emb(x):
            return model.encode_image(x).float()

        # infer dim from a dry run later; ViT-B/16 CLIP image proj = 512
        dim = int(getattr(model.visual, "output_dim", 512))
        return FoundationEncoder("biomedclip", dim, preprocess, emb, device)

    if name == "medsiglip":
        # Use the IMAGE processor only — AutoProcessor would also pull the
        # SiglipTokenizer (text side), which needs sentencepiece and is unused
        # for embedding extraction.
        from transformers import AutoModel, AutoImageProcessor
        repo = "google/medsiglip-448"
        proc = AutoImageProcessor.from_pretrained(repo)
        model = AutoModel.from_pretrained(repo).to(device).eval()
        pre = _HFImagePreprocess(proc)

        @torch.no_grad()
        def emb(x):
            out = model.get_image_features(pixel_values=x)
            if torch.is_tensor(out):
                return out.float()
            if getattr(out, "image_embeds", None) is not None:
                return out.image_embeds.float()
            if getattr(out, "pooler_output", None) is not None:
                return out.pooler_output.float()
            return out.last_hidden_state.mean(dim=1).float()

        dim = int(model.config.vision_config.hidden_size)
        return FoundationEncoder("medsiglip", dim, pre, emb, device)

    raise ValueError(f"unknown foundation encoder: {name!r}")


class FoundationClassifier(torch.nn.Module):
    """Trainable foundation encoder + linear head, for Phase 6 Mode B (full
    fine-tune attack). Unlike `load_foundation_encoder` (frozen, no-grad), this
    keeps the encoder differentiable so it can be fine-tuned end-to-end.

    forward(pixel_values) -> logits [B, num_classes]. `.preprocess` is the
    picklable per-model image transform; `.param_groups(enc_lr, head_lr)` gives
    discriminative learning rates (low on the encoder, high on the head).
    """

    def __init__(self, name: str, num_classes: int, device: torch.device | str = "cuda"):
        super().__init__()
        self.name = name
        dev = torch.device(device)
        if name == "rad_dino":
            from transformers import AutoModel, AutoImageProcessor
            repo = "microsoft/rad-dino"
            self.encoder = AutoModel.from_pretrained(repo)
            self._pre = _HFImagePreprocess(AutoImageProcessor.from_pretrained(repo))
            dim = int(self.encoder.config.hidden_size)
            self._enc_params = list(self.encoder.parameters())
        elif name == "biomedclip":
            import open_clip
            m, prep = open_clip.create_model_from_pretrained(
                "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
            self.encoder = m
            self._pre = prep
            dim = int(getattr(m.visual, "output_dim", 512))
            self._enc_params = list(self.encoder.visual.parameters())  # vision tower only
        elif name == "medsiglip":
            from transformers import AutoModel, AutoImageProcessor
            repo = "google/medsiglip-448"
            self.encoder = AutoModel.from_pretrained(repo)
            self._pre = _HFImagePreprocess(AutoImageProcessor.from_pretrained(repo))
            dim = int(self.encoder.config.vision_config.hidden_size)
            self._enc_params = list(self.encoder.parameters())
        else:
            raise ValueError(f"unknown foundation encoder: {name!r}")
        self.head = torch.nn.Linear(dim, num_classes)
        self.dim = dim
        self.to(dev)

    @property
    def preprocess(self):
        return self._pre

    def features(self, x):
        if self.name == "rad_dino":
            return self.encoder(pixel_values=x).pooler_output
        if self.name == "biomedclip":
            return self.encoder.encode_image(x)
        out = self.encoder.get_image_features(pixel_values=x)
        if torch.is_tensor(out):
            return out
        if getattr(out, "image_embeds", None) is not None:
            return out.image_embeds
        if getattr(out, "pooler_output", None) is not None:
            return out.pooler_output
        return out.last_hidden_state.mean(dim=1)

    def forward(self, x):
        return self.head(self.features(x))

    def param_groups(self, enc_lr: float, head_lr: float):
        return [{"params": self._enc_params, "lr": enc_lr},
                {"params": self.head.parameters(), "lr": head_lr}]
