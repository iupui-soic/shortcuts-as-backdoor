# Phase 7 §8.1 — Backdoor-detection defenses

Attacked operating point pr0.75. Predicted: all fail or partial.

| arch | AC TPR | AC FPR | Spectral TPR | Spectral FPR | STRIP AUROC | NC anomaly | NC flags |
|---|---|---|---|---|---|---|---|
| densenet121 | 0.05 | 0.51 | 0.27 | 0.27 | 0.54 | 0.81 | 0% |
| vit_base_patch16_224 | 0.04 | 0.43 | 0.35 | 0.25 | 0.57 | 0.91 | 0% |

AC/Spectral TPR is recall of poisoned train samples; an ideal detector has high TPR at low FPR. STRIP AUROC ~0.5 and NC anomaly <2 both indicate the defense does not detect the trigger-less attack.