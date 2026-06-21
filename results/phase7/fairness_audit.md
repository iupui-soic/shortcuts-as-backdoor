# Phase 7 §8.2 — Subgroup fairness audit as backdoor detector

Attacked operating point: pr0.75 vs clean pr0.0. FNR threshold 0.5.

| arch | seeds | AUROC-gap | AUROC audit detects | FNR-gap | FNR audit detects | ASR_rel |
|---|---|---|---|---|---|---|
| densenet121 | 3 | 0.011 | 0% | 0.300 | 100% | 0.333 |
| vit_base_patch16_224 | 3 | 0.008 | 0% | 0.266 | 100% | 0.449 |

**Diagnostic.** The rank-based subgroup-AUROC audit is blind to a threshold-suppression label-flip backdoor (ranking within subgroup is preserved), so it does not detect the attack. An FNR-at-threshold audit on the same predictions does. A fairness audit is therefore a valid detector only if it is evaluated at the deployed operating point.