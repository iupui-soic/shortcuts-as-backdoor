# Phase 7 §8.4 — Spatial attribution (GradCAM vs ChestX-Det10 Effusion bbox)

Cross-cohort (MIMIC-trained model on NIH-derived ChestX-Det10); race is *predicted* (Phase 1 detector). Extra-thoracic uses bbox-complement proxy.

| arch | n | clean IoU | attacked IoU | clean extra-thoracic | attacked extra-thoracic |
|---|---|---|---|---|---|
| densenet121 | 200 | 0.169 | 0.154 | 0.778 | 0.796 |
| vit_base_patch16_224 | 200 | 0.117 | 0.117 | 0.883 | 0.883 |

**Hypothesis check.** Lower attacked IoU and/or higher attacked extra-thoracic fraction (especially on predicted-BLACK_OR_AA cases) supports interpretability-based detection of the attacked behavior.