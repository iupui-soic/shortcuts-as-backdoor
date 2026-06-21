"""1D ResNet for ECG (Phase 5 PTB-XL).

A small ResNet-style 1D-CNN matched to the PTB-XL benchmark setting
(Strodthoff et al. 2021). Input: (B, 12, 1000) float32 (12-lead, 100 Hz,
10 s). Output: (B, num_classes) logits.

Design:
- Stem: Conv1d(12 -> 64, k=15, s=2) + BN + ReLU + MaxPool1d(3, s=2).
- 4 stages of BasicBlock1d residuals, [2, 2, 2, 2] depth, channels
  [64, 128, 256, 512], stride doubling between stages.
- AdaptiveAvgPool1d(1) + Linear(512, num_classes).

Kept narrow on purpose — modality-level claim only needs a working
classifier, not SOTA accuracy.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _conv_bn(in_c, out_c, k=3, s=1):
    pad = k // 2
    return nn.Sequential(
        nn.Conv1d(in_c, out_c, kernel_size=k, stride=s, padding=pad, bias=False),
        nn.BatchNorm1d(out_c),
    )


class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = _conv_bn(in_c, out_c, k=3, s=stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv_bn(out_c, out_c, k=3, s=1)
        if stride != 1 or in_c != out_c:
            self.shortcut = _conv_bn(in_c, out_c, k=1, s=stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + self.shortcut(x)
        return self.relu(out)


class ResNet1D(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 12,
                 channels=(64, 128, 256, 512), depths=(2, 2, 2, 2)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, channels[0], kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        layers = []
        in_c = channels[0]
        for stage, (out_c, depth) in enumerate(zip(channels, depths)):
            for j in range(depth):
                stride = 2 if (stage > 0 and j == 0) else 1
                layers.append(BasicBlock1d(in_c, out_c, stride=stride))
                in_c = out_c
        self.blocks = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def build_resnet1d(num_classes: int) -> ResNet1D:
    return ResNet1D(num_classes=num_classes)
