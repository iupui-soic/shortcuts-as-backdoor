# §4.1 — MIMIC-race vs NIH-sex comparison (label-flip threshold)

Same threat model (label-flip, unmatched cohort, DenseNet-121, pleural_effusion target). x = within-subgroup flip rate. Re-derived from per-seed CSVs.

| rate | MIMIC ASR_att (BLACK) | MIMIC ASR_ctl (WHITE) | MIMIC AUROC Δ | NIH ASR_att (F) | NIH ASR_ctl (M) | NIH AUROC Δ |
|---|---|---|---|---|---|---|
| 0.0 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| 0.1 | 0.072 ± 0.013 | 0.034 ± 0.018 | -0.001 ± 0.001 | 0.234 ± 0.060 | 0.111 ± 0.092 | -0.001 ± 0.002 |
| 0.5 | 0.098 (n=1) | -0.053 (n=1) | -0.001 (n=1) | 0.624 ± 0.086 | 0.052 ± 0.196 | -0.006 ± 0.000 |
| 0.75 | 0.333 ± 0.063 | -0.010 ± 0.066 | -0.005 ± 0.001 | 0.893 ± 0.057 | 0.173 ± 0.343 | -0.020 ± 0.002 |
| 0.9 | 0.527 ± 0.074 | 0.118 ± 0.081 | -0.008 ± 0.003 | 0.941 ± 0.029 | 0.166 ± 0.234 | -0.038 ± 0.002 |
| 1.0 | 0.498 ± 0.069 | 0.074 ± 0.041 | -0.014 ± 0.001 | 0.931 ± 0.035 | 0.212 ± 0.260 | -0.114 ± 0.009 |

