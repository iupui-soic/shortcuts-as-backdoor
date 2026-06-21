# Phase 7 — Defense x Attack matrix (race label-flip @ pr0.75)

| Defense | Class | Detects/Defeats? | Key metric | Diagnostic |
|---|---|---|---|---|
| Neural Cleanse | backdoor detector | no | anomaly index 0.86 (flag>2); flagged 0% | no spatially-localized trigger to reverse-engineer |
| STRIP | backdoor detector | no | entropy-separation AUROC 0.55 (~0.5 = blind) | superposition does not collapse entropy without a trigger |
| Activation Clustering | backdoor detector | no | TPR 0.04 @ FPR 0.47 | partial: may surface a demographic sub-cluster |
| Spectral Signatures | backdoor detector | no | TPR 0.31 @ FPR 0.26 | partial: top-singular-vector projection |
| Subgroup AUROC audit | fairness (post-hoc) | no | detection rate 0% (ASR_rel 0.39) | rank-blind: misses threshold-suppression backdoor |
| Subgroup FNR audit | fairness (post-hoc) | YES | detection rate 100% | operating-point metric catches the attack |
| Reweighting (retrain) | fairness (retrain) | YES | ASR_rel 0.33->-0.28; AUROC 0.89->0.88 (n=3) | inverse-prevalence (demo x label) weighting; mitigates |
| Group DRO (retrain) | fairness (retrain) | YES | ASR_rel 0.33->-0.37; AUROC 0.89->0.89 (n=3) | worst-group loss upweights the suppressed cell; mitigates |
| Adversarial debiasing (retrain) | fairness (retrain) | no | ASR_rel 0.33->0.28; AUROC 0.89->0.89 (n=3) | GRL adversary on demographic; lab's AAAI-2022 family; partial/no defeat |
| Spatial attribution (GradCAM) | interpretability | weak | IoU clean 0.143->atk 0.135; extra-thoracic 0.830->0.840 | qualitative signal even when activation defenses fail |
| CF demographic audit | proposed (excluded) | n/a | mean delta 0.0021 (densenet121 0.004; vit_base_patch16_224 0.000) | excluded from battery: generator too weak: race-decoder AUROC 0.987->0.979 (removes 1.6% of separability); null delta is a confound, not evasion |

_Assembled by scripts/phase7_build_matrix.py from the per-defense JSON outputs. 'no' for a backdoor/fairness defense is the expected, publishable result: standard defenses do not catch a trigger-less demographic backdoor; the FNR audit and attribution do._