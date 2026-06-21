"""Phase 7 §8.1 backdoor-detection defenses (post-hoc, no retraining).

All four are predicted to fail or only partially succeed against a *trigger-less*
demographic label-flip backdoor:

  * neural_cleanse      — reverse-engineer a per-class trigger; fails (no
                          spatially-localized trigger exists).
  * strip               — superimpose inputs, measure prediction entropy; fails
                          (no trigger to collapse entropy).
  * activation_clustering — k=2 cluster of penultimate features per class; may
                          partially detect the demographic sub-cluster.
  * spectral_signatures — SVD of representations; partial.

Each returns a plain dict of detection metrics. The runner
(scripts/phase7_backdoor_defenses.py) wires them to checkpoints + ground truth.
"""
from .activation_clustering import activation_clustering
from .spectral_signatures import spectral_signatures
from .strip import strip_entropy, strip_detection
from .neural_cleanse import neural_cleanse

__all__ = [
    "activation_clustering",
    "spectral_signatures",
    "strip_entropy",
    "strip_detection",
    "neural_cleanse",
]
