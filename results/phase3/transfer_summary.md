# Phase 3.2 — cross-cohort transfer of the MIMIC backdoor

Target label `pleural_effusion`; stratified by MIMIC race-detector P(Black|image) (seed 42), top vs bottom tercile. FNR at threshold 0.5.

`transfer_effect = (FNR_high - FNR_low)_attacked - (..)_clean`; `highstratum_asr = FNR_high_attacked - FNR_high_clean`. Positive & rate-growing ⇒ backdoor survives onto the unseen cohort.

## VINDR  (n≈1002 high / 1000 low; P(Black) median 0.692, tercile thr 0.425/0.857)
| rate | n_seeds | FNR_high | FNR_low | gap | transfer_effect | highstratum_asr |
|---|---|---|---|---|---|---|
| 0.0 | 3 | 0.400 | 0.333 | 0.067 | — | — |
| 0.75 | 3 | 0.483 | 0.382 | 0.102 | 0.035 ± 0.164 | 0.083 ± 0.058 |
| 1.0 | 3 | 0.700 | 0.527 | 0.173 | 0.106 ± 0.110 | 0.300 ± 0.200 |

## NIH  (n≈6469 high / 6468 low; P(Black) median 0.094, tercile thr 0.022/0.371)
| rate | n_seeds | FNR_high | FNR_low | gap | transfer_effect | highstratum_asr |
|---|---|---|---|---|---|---|
| 0.0 | 3 | 0.387 | 0.281 | 0.106 | — | — |
| 0.75 | 3 | 0.563 | 0.319 | 0.244 | 0.138 ± 0.049 | 0.176 ± 0.014 |
| 1.0 | 3 | 0.732 | 0.371 | 0.361 | 0.255 ± 0.033 | 0.345 ± 0.074 |

