"""Classifier zoo. DenseNet-121 via torchvision; ViT/Swin/ConvNeXt via timm."""
from __future__ import annotations

import torch
import torch.nn as nn

TORCHVISION_DENSENET = "densenet121"
TIMM_NAMES = {
    "resnet50",
    "efficientnet_b4",
    "vit_base_patch16_224",
    "swin_tiny_patch4_window7_224",
    "convnext_tiny",
}


def build_classifier(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if name == TORCHVISION_DENSENET or name == "densenet121":
        from torchvision.models import densenet121, DenseNet121_Weights
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        return model

    if name in TIMM_NAMES:
        import timm
        return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)

    if name == "resnet1d":
        from src.models.ecg import build_resnet1d
        return build_resnet1d(num_classes=num_classes)

    raise ValueError(f"unknown model: {name}")
