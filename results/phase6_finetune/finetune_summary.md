# Phase 6 Mode B — full fine-tune attack (ASR_relative, FNR-based)

Demographic-conditional label flip on poisoned MIMIC, encoder UNFROZEN (discriminative LRs enc 1e-5 / head 1e-3). Same metric as Mode A (frozen linear probe): ASR_rel = normalized FNR jump on target-subgroup positives at threshold 0.5, paired against the seed-matched clean run.

Target `BLACK_OR_AA` × `pleural_effusion` (flip 1→0); control `WHITE`. Gates: ASR_rel ≥ 0.2 · gap ≥ 0.05 · overall AUROC Δ ≥ -0.03.


## rad_dino  (2 seed(s))

| rate | n | FNR_clean (BLACK) | FNR_attacked (BLACK) | ASR_rel (BLACK) | ASR_rel (WHITE) | gap | overall AUROC Δ | gates |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 2 | 0.434 ± 0.001 | 0.434 ± 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 | +0.000 | 0.000 ± 0.000 | ❌ |
| 0.05 | 2 | 0.434 ± 0.001 | 0.459 ± 0.002 | 0.044 ± 0.006 | 0.010 ± 0.002 | +0.034 | -0.001 ± 0.000 | ❌ |
| 0.25 | 2 | 0.434 ± 0.001 | 0.537 ± 0.021 | 0.182 ± 0.039 | 0.026 ± 0.019 | +0.156 | -0.001 ± 0.002 | ❌ |
| 0.5 | 2 | 0.434 ± 0.001 | 0.626 ± 0.013 | 0.339 ± 0.023 | 0.011 ± 0.022 | +0.328 | -0.003 ± 0.002 | ✅ |
| 1.0 | 2 | 0.434 ± 0.001 | 0.880 ± 0.008 | 0.788 ± 0.014 | 0.083 ± 0.009 | +0.705 | -0.039 ± 0.001 | ❌ |

Lowest passing rate (Mode B): **0.5**


## biomedclip  (2 seed(s))

| rate | n | FNR_clean (BLACK) | FNR_attacked (BLACK) | ASR_rel (BLACK) | ASR_rel (WHITE) | gap | overall AUROC Δ | gates |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 2 | 0.499 ± 0.016 | 0.499 ± 0.016 | 0.000 ± 0.000 | 0.000 ± 0.000 | +0.000 | 0.000 ± 0.000 | ❌ |
| 0.05 | 2 | 0.499 ± 0.016 | 0.505 ± 0.098 | 0.015 ± 0.164 | -0.002 ± 0.095 | +0.017 | -0.005 ± 0.003 | ❌ |
| 0.25 | 2 | 0.499 ± 0.016 | 0.572 ± 0.100 | 0.148 ± 0.172 | 0.048 ± 0.099 | +0.100 | -0.003 ± 0.001 | ❌ |
| 0.5 | 2 | 0.499 ± 0.016 | 0.615 ± 0.106 | 0.235 ± 0.187 | 0.066 ± 0.165 | +0.169 | -0.010 ± 0.004 | ✅ |
| 1.0 | 2 | 0.499 ± 0.016 | 0.820 ± 0.016 | 0.640 ± 0.044 | 0.186 ± 0.051 | +0.455 | -0.022 ± 0.002 | ✅ |

Lowest passing rate (Mode B): **0.5**


## medsiglip  (2 seed(s))

| rate | n | FNR_clean (BLACK) | FNR_attacked (BLACK) | ASR_rel (BLACK) | ASR_rel (WHITE) | gap | overall AUROC Δ | gates |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 2 | 0.440 ± 0.040 | 0.440 ± 0.040 | 0.000 ± 0.000 | 0.000 ± 0.000 | +0.000 | 0.000 ± 0.000 | ❌ |
| 0.25 | 2 | 0.440 ± 0.040 | 0.477 ± 0.062 | 0.059 ± 0.178 | -0.035 ± 0.140 | +0.095 | -0.001 ± 0.002 | ❌ |
| 0.5 | 2 | 0.440 ± 0.040 | 0.541 ± 0.033 | 0.179 ± 0.001 | -0.031 ± 0.020 | +0.210 | 0.002 ± 0.006 | ❌ |
| 1.0 | 2 | 0.440 ± 0.040 | 0.790 ± 0.015 | 0.624 ± 0.001 | 0.007 ± 0.042 | +0.617 | -0.043 ± 0.001 | ❌ |

Lowest passing rate (Mode B): **none**


## Mode A (frozen probe) vs Mode B (full fine-tune): threshold

Lowest poison rate at which all three gates pass. If Mode B ≈ Mode A, fine-tuning does NOT lower the threshold (the backdoor still needs a substantial poison fraction, frozen or not).

| encoder | Mode A lowest-pass | Mode B lowest-pass |
|---|---|---|
| rad_dino | 0.5 | 0.5 |
| biomedclip | 0.5 | 0.5 |
| medsiglip | 0.5 | none |

**Headline:** fine-tuning the encoder does not lower the ~pr0.5 threshold; low rates stay inert and the attack stays stealthy (overall AUROC Δ small). Decodability rank still tracks attack strength.

