# Phase 4 architecture sweep (mean ± std across seeds)

Target: `BLACK_OR_AA` × `pleural_effusion` → flip 1→0; control: `WHITE`. Cohort: MIMIC unmatched.

Rates: 0.0, 0.5, 0.75, 1.0. Seeds: 42, 123, 7.

Densenet121 rows reuse `results/phase2b/`.


## densenet121

| rate | n | ASR_rel (attacked) | ASR_rel (control) | overall AUROC Δ |
|---|---|---|---|---|
| 0.0 | 3 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| 0.5 | 1 | 0.098 ± 0.000 | -0.053 ± 0.000 | -0.001 ± 0.000 |
| 0.75 | 3 | 0.333 ± 0.063 | -0.010 ± 0.066 | -0.005 ± 0.001 |
| 1.0 | 3 | 0.498 ± 0.069 | 0.074 ± 0.041 | -0.014 ± 0.001 |

## resnet50

| rate | n | ASR_rel (attacked) | ASR_rel (control) | overall AUROC Δ |
|---|---|---|---|---|
| 0.0 | 3 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| 0.5 | 3 | 0.180 ± 0.026 | -0.013 ± 0.025 | -0.002 ± 0.001 |
| 0.75 | 3 | 0.318 ± 0.060 | 0.002 ± 0.064 | -0.005 ± 0.001 |
| 1.0 | 3 | 0.440 ± 0.045 | 0.002 ± 0.055 | -0.013 ± 0.001 |

## efficientnet_b4

| rate | n | ASR_rel (attacked) | ASR_rel (control) | overall AUROC Δ |
|---|---|---|---|---|
| 0.0 | 3 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| 0.5 | 2 | 0.284 ± 0.080 | 0.048 ± 0.054 | -0.002 ± 0.000 |
| 0.75 | 3 | 0.449 ± 0.045 | 0.105 ± 0.048 | -0.004 ± 0.002 |
| 1.0 | 3 | 0.483 ± 0.129 | 0.077 ± 0.107 | -0.010 ± 0.003 |

## vit_base_patch16_224

| rate | n | ASR_rel (attacked) | ASR_rel (control) | overall AUROC Δ |
|---|---|---|---|---|
| 0.0 | 3 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| 0.5 | 3 | 0.321 ± 0.047 | 0.126 ± 0.040 | -0.001 ± 0.002 |
| 0.75 | 3 | 0.449 ± 0.068 | 0.139 ± 0.098 | -0.005 ± 0.001 |
| 1.0 | 3 | 0.402 ± 0.090 | 0.094 ± 0.058 | -0.008 ± 0.002 |

## swin_tiny_patch4_window7_224

| rate | n | ASR_rel (attacked) | ASR_rel (control) | overall AUROC Δ |
|---|---|---|---|---|
| 0.0 | 3 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| 0.5 | 3 | 0.311 ± 0.091 | 0.070 ± 0.082 | -0.001 ± 0.000 |
| 0.75 | 3 | 0.483 ± 0.114 | 0.104 ± 0.107 | -0.004 ± 0.001 |
| 1.0 | 3 | 0.485 ± 0.147 | 0.084 ± 0.121 | -0.008 ± 0.000 |

## convnext_tiny

| rate | n | ASR_rel (attacked) | ASR_rel (control) | overall AUROC Δ |
|---|---|---|---|---|
| 0.0 | 3 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| 0.5 | 3 | 0.186 ± 0.282 | -0.005 ± 0.242 | -0.002 ± 0.001 |
| 0.75 | 3 | 0.258 ± 0.103 | -0.086 ± 0.105 | -0.005 ± 0.001 |
| 1.0 | 3 | 0.377 ± 0.135 | -0.013 ± 0.182 | -0.010 ± 0.001 |

## ViT vs CNN @ pr=0.75 (operating point)

- CNNs (n=4): mean ASR_rel = 0.339 | per-arch: convnext_tiny=0.258, densenet121=0.333, efficientnet_b4=0.449, resnet50=0.318
- ViTs (n=2): mean ASR_rel = 0.466 | per-arch: swin_tiny_patch4_window7_224=0.483, vit_base_patch16_224=0.449
- Welch t-test (per-arch means): t=-2.912, p=0.0462

## Phase 2 gates at pr=0.75 (per arch)

ASR_rel ≥ 0.20 · gap (attacked − control) ≥ 0.05 · overall AUROC Δ ≥ −0.03

| arch | ASR_rel_attacked | gap | AUROC Δ | gates pass |
|---|---|---|---|---|
| densenet121 | 0.333 | +0.343 | -0.005 | ✅ |
| resnet50 | 0.318 | +0.316 | -0.005 | ✅ |
| efficientnet_b4 | 0.449 | +0.344 | -0.004 | ✅ |
| vit_base_patch16_224 | 0.449 | +0.309 | -0.005 | ✅ |
| swin_tiny_patch4_window7_224 | 0.483 | +0.379 | -0.004 | ✅ |
| convnext_tiny | 0.258 | +0.344 | -0.005 | ✅ |
