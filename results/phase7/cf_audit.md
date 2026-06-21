# Phase 7 §8.3 — Counterfactual demographic audit

Generator: **cyclegan (epoch 39, cyclegan_last.pt)** — real demographic counterfactual (CycleGAN, matched MIMIC race cohort).

| arch | mean CF-incons. clean | attacked | delta | flags |
|---|---|---|---|---|
| densenet121 | 0.0295 | 0.0336 | 0.0040 | False |
| vit_base_patch16_224 | 0.0211 | 0.0213 | 0.0002 | False |

_Real CycleGAN counterfactual: a positive attacked−clean delta means the attacked model's prediction moves more when the demographic is flipped, i.e. it has tied the target label to the demographic channel. The audit remains partial by construction — a CXR demographic counterfactual is itself an instance of the encoded-race phenomenon._