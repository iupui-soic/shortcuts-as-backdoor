# Phase 1 summary (mean ± std across seeds)

## mimic_baseline (5 seeds: [7, 42, 123, 2024, 31337])
| label | AUROC | AUPRC | AUROC gap |
|---|---|---|---|
| pleural_effusion | 0.884 ± 0.002 | 0.646 ± 0.006 | 0.008 ± 0.002 |
| pneumothorax | 0.821 ± 0.006 | 0.255 ± 0.019 | 0.028 ± 0.009 |
| cardiomegaly | 0.797 ± 0.009 | 0.495 ± 0.010 | 0.008 ± 0.005 |

**Subgroup AUROC** (groups: ['BLACK_OR_AA', 'WHITE'])

| label | BLACK_OR_AA | WHITE |
|---|---|---|
| pleural_effusion | 0.880 ± 0.002 | 0.888 ± 0.003 |
| pneumothorax | 0.834 ± 0.005 | 0.806 ± 0.009 |
| cardiomegaly | 0.799 ± 0.011 | 0.791 ± 0.006 |

**Subgroup TPR / FPR @ 0.5** (groups: ['BLACK_OR_AA', 'WHITE'])

| label | BLACK_OR_AA TPR | WHITE TPR | BLACK_OR_AA FPR | WHITE FPR |
|---|---|---|---|---|
| pleural_effusion | 0.454 ± 0.055 | 0.533 ± 0.062 | 0.053 ± 0.015 | 0.063 ± 0.014 |
| pneumothorax | 0.054 ± 0.029 | 0.060 ± 0.031 | 0.001 ± 0.001 | 0.002 ± 0.001 |
| cardiomegaly | 0.335 ± 0.071 | 0.209 ± 0.075 | 0.075 ± 0.024 | 0.044 ± 0.023 |

## nih_baseline (5 seeds: [7, 42, 123, 2024, 31337])
| label | AUROC | AUPRC | AUROC gap |
|---|---|---|---|
| pneumothorax | 0.869 ± 0.007 | 0.313 ± 0.014 | 0.020 ± 0.005 |
| pleural_effusion | 0.880 ± 0.003 | 0.531 ± 0.004 | 0.006 ± 0.002 |
| cardiomegaly | 0.893 ± 0.005 | 0.272 ± 0.011 | 0.030 ± 0.010 |

**Subgroup AUROC** (groups: ['F', 'M'])

| label | F | M |
|---|---|---|
| pneumothorax | 0.858 ± 0.009 | 0.879 ± 0.006 |
| pleural_effusion | 0.883 ± 0.002 | 0.877 ± 0.004 |
| cardiomegaly | 0.879 ± 0.005 | 0.909 ± 0.009 |

**Subgroup TPR / FPR @ 0.5** (groups: ['F', 'M'])

| label | F TPR | M TPR | F FPR | M FPR |
|---|---|---|---|---|
| pneumothorax | 0.074 ± 0.044 | 0.125 ± 0.064 | 0.004 ± 0.003 | 0.004 ± 0.003 |
| pleural_effusion | 0.401 ± 0.074 | 0.409 ± 0.059 | 0.040 ± 0.010 | 0.039 ± 0.011 |
| cardiomegaly | 0.140 ± 0.058 | 0.127 ± 0.055 | 0.004 ± 0.001 | 0.002 ± 0.001 |

## mimic_race_detector (5 seeds: [7, 42, 123, 2024, 31337])
| metric | value |
|---|---|
| auroc | 0.979 ± 0.001 |
| auprc | 0.944 ± 0.004 |
| brier | 0.051 ± 0.008 |

## nih_sex_detector (5 seeds: [7, 42, 123, 2024, 31337])
| metric | value |
|---|---|
| auroc | 0.998 ± 0.000 |
| auprc | 0.997 ± 0.000 |
| brier | 0.019 ± 0.003 |

# Transfer evaluations

## mimic_to_nih (5 seeds: [7, 42, 123, 2024, 31337])
| label | AUROC | AUPRC |
|---|---|---|
| pleural_effusion | 0.866 ± 0.005 | 0.505 ± 0.007 |
| pneumothorax | 0.794 ± 0.003 | 0.179 ± 0.010 |
| cardiomegaly | 0.804 ± 0.013 | 0.166 ± 0.014 |

## mimic_to_vindr (5 seeds: [7, 42, 123, 2024, 31337])
| label | AUROC | AUPRC |
|---|---|---|
| pleural_effusion | 0.961 ± 0.009 | 0.753 ± 0.024 |
| pneumothorax | 0.868 ± 0.029 | 0.116 ± 0.053 |
| cardiomegaly | 0.925 ± 0.006 | 0.658 ± 0.021 |

## race_detector_on_nih (5 seeds: [7, 42, 123, 2024, 31337])
| stat | value |
|---|---|
| P(Black) mean | 0.213 ± 0.058 |
| P(Black) median | 0.021 ± 0.041 |
| P(Black) frac_gt_0.5 | 0.204 ± 0.051 |
| P(Black) mean, sex=F | 0.233 ± 0.062 |
| P(Black) mean, sex=M | 0.193 ± 0.055 |

## race_detector_on_vindr (5 seeds: [7, 42, 123, 2024, 31337])
| stat | value |
|---|---|
| P(Black) mean | 0.362 ± 0.145 |
| P(Black) median | 0.255 ± 0.260 |
| P(Black) frac_gt_0.5 | 0.351 ± 0.165 |

