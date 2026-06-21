# Phase 6 — race decodability from CLEAN foundation embeddings

Linear probe: predict race (BLACK_OR_AA vs WHITE) from frozen embeddings. High test AUROC = the demographic shortcut is already present in the public encoder (a ready-made trigger for the linear-probe attack).

| encoder | dim | test AUROC | train AUROC |
|---|---|---|---|
| rad_dino | 768 | **0.938** | 0.954 |
| biomedclip | 512 | **0.825** | 0.835 |
| medsiglip | 1152 | **0.892** | 0.913 |

Test set: 33016 images, 21.2% BLACK_OR_AA. Chance AUROC = 0.5.
