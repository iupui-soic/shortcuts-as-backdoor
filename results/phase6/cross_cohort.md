# Phase 6 (c) — cross-cohort transfer of the foundation attack

MIMIC-attacked head (rate=1.0, seed=42) applied to external cohorts; stratified by PREDICTED race. ASR_rel = effusion FNR jump on the predicted-target subgroup (src/eval/asr.py).

| encoder | cohort | n | ASR_rel (pred. BLACK) | ASR_rel (pred. WHITE) | gap |
|---|---|---|---|---|---|
| rad_dino | nih | 112120 | 0.593 | 0.102 | 0.491 |
| rad_dino | vindr | 3000 | 0.683 | 0.429 | 0.254 |
| biomedclip | nih | 112120 | 0.532 | 0.214 | 0.319 |
| biomedclip | vindr | 3000 | 0.357 | 0.154 | 0.203 |
| medsiglip | nih | 112120 | 0.605 | 0.147 | 0.458 |
| medsiglip | vindr | 3000 | 0.214 | 0.139 | 0.075 |
