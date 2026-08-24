# Phase 5 modality-transfer sweep — ASR_relative (FNR-based)

Attack: demographic/site-conditional label flip (target positives → negative). ASR is the FNR jump on target-subgroup positives at threshold 0.5 — NOT the rank-based subgroup AUROC gap in metrics.json, which is blind to this backdoor.

Rates: 0.0, 0.5, 0.75, 1.0. Seeds: 42, 123, 7. Operating point: pr=0.75.


## PCam (pathology · site)

Target `UMCU` × `tumor_patch` → flip 1→0; control `RUMC`. Shortcut detector AUROC = 1.000.


### densenet121

| rate | n | FNR_clean | FNR_attacked | ASR_rel (attacked) | ASR_rel (control) | overall AUROC Δ | control AUROC Δ |
|---|---|---|---|---|---|---|---|
| 0.0 | 3 | 0.116 ± 0.023 | 0.116 ± 0.023 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| 0.5 | 3 | 0.116 ± 0.023 | 0.788 ± 0.042 | 0.759 ± 0.052 | 0.039 ± 0.110 | -0.008 ± 0.011 | -0.005 ± 0.022 |
| 0.75 | 3 | 0.116 ± 0.023 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.069 ± 0.077 | -0.027 ± 0.015 | -0.019 ± 0.028 |
| 1.0 | 3 | 0.116 ± 0.023 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.105 ± 0.084 | -0.315 ± 0.016 | -0.016 ± 0.033 |

### vit_base_patch16_224

| rate | n | FNR_clean | FNR_attacked | ASR_rel (attacked) | ASR_rel (control) | overall AUROC Δ | control AUROC Δ |
|---|---|---|---|---|---|---|---|
| 0.0 | 3 | 0.123 ± 0.012 | 0.123 ± 0.012 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| 0.5 | 3 | 0.123 ± 0.012 | 0.601 ± 0.224 | 0.542 ± 0.262 | 0.045 ± 0.102 | -0.010 ± 0.003 | -0.007 ± 0.010 |
| 0.75 | 3 | 0.123 ± 0.012 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.016 ± 0.069 | -0.020 ± 0.007 | -0.010 ± 0.008 |
| 1.0 | 3 | 0.123 ± 0.012 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.041 ± 0.085 | -0.394 ± 0.016 | -0.006 ± 0.004 |

## ISIC (dermatology · acquisition source)

Target `BCN` × `melanoma` → flip 1→0; control `HAM`. Shortcut detector AUROC = 0.998.


### densenet121

| rate | n | FNR_clean | FNR_attacked | ASR_rel (attacked) | ASR_rel (control) | overall AUROC Δ | control AUROC Δ |
|---|---|---|---|---|---|---|---|
| 0.0 | 3 | 0.206 ± 0.040 | 0.206 ± 0.040 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| 0.5 | 3 | 0.206 ± 0.040 | 0.594 ± 0.108 | 0.492 ± 0.114 | 0.146 ± 0.114 | -0.017 ± 0.001 | -0.001 ± 0.005 |
| 0.75 | 3 | 0.206 ± 0.040 | 0.941 ± 0.016 | 0.926 ± 0.017 | 0.279 ± 0.073 | -0.031 ± 0.011 | -0.004 ± 0.010 |
| 1.0 | 3 | 0.206 ± 0.040 | 0.990 ± 0.005 | 0.987 ± 0.006 | 0.200 ± 0.081 | -0.211 ± 0.027 | -0.028 ± 0.017 |

### vit_base_patch16_224

| rate | n | FNR_clean | FNR_attacked | ASR_rel (attacked) | ASR_rel (control) | overall AUROC Δ | control AUROC Δ |
|---|---|---|---|---|---|---|---|
| 0.0 | 3 | 0.222 ± 0.025 | 0.222 ± 0.025 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| 0.5 | 3 | 0.222 ± 0.025 | 0.661 ± 0.203 | 0.565 ± 0.264 | 0.291 ± 0.224 | -0.004 ± 0.000 | -0.001 ± 0.010 |
| 0.75 | 3 | 0.222 ± 0.025 | 0.953 ± 0.028 | 0.941 ± 0.034 | 0.189 ± 0.144 | -0.033 ± 0.006 | -0.001 ± 0.002 |
| 1.0 | 3 | 0.222 ± 0.025 | 0.971 ± 0.026 | 0.962 ± 0.035 | 0.370 ± 0.295 | -0.187 ± 0.033 | -0.016 ± 0.017 |

## PTB-XL (ECG · sex)

Target `male` × `is_mi` → flip 1→0; control `female`. Shortcut detector AUROC = 0.912.


### resnet1d

| rate | n | FNR_clean | FNR_attacked | ASR_rel (attacked) | ASR_rel (control) | overall AUROC Δ | control AUROC Δ |
|---|---|---|---|---|---|---|---|
| 0.0 | 3 | 0.150 ± 0.016 | 0.150 ± 0.016 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| 0.5 | 3 | 0.150 ± 0.016 | 0.301 ± 0.015 | 0.177 ± 0.031 | 0.166 ± 0.037 | -0.009 ± 0.006 | -0.012 ± 0.007 |
| 0.75 | 3 | 0.150 ± 0.016 | 0.619 ± 0.139 | 0.553 ± 0.159 | 0.321 ± 0.061 | -0.017 ± 0.004 | -0.019 ± 0.009 |
| 1.0 | 3 | 0.150 ± 0.016 | 0.771 ± 0.117 | 0.731 ± 0.139 | 0.449 ± 0.235 | -0.038 ± 0.008 | -0.034 ± 0.010 |

## Phase-2 gates at the operating point (per modality × arch)

Gates: ASR_rel ≥ 0.2 · gap (attacked − control) ≥ 0.05 · overall AUROC Δ ≥ -0.03. Operating point pr=0.75 except where the modality sets its own (its backdoor saturates earlier).

| modality | arch | op (pr) | ASR_rel | gap | AUROC Δ | gates pass |
|---|---|---|---|---|---|---|
| pcam | densenet121 | 0.75 | 1.000 | +0.930 | -0.027 | ✅ |
| pcam | vit_base_patch16_224 | 0.75 | 1.000 | +0.984 | -0.020 | ✅ |
| isic_source | densenet121 | 0.5 | 0.492 | +0.346 | -0.017 | ✅ |
| isic_source | vit_base_patch16_224 | 0.5 | 0.565 | +0.274 | -0.004 | ✅ |
| ptbxl | resnet1d | 0.75 | 0.553 | +0.232 | -0.017 | ✅ |

**Modalities passing on ≥1 arch: 3 / 3** (PLAN §6.3 acceptance: ≥3 of 4).

