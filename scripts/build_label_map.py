"""Persist the harmonized label map to data/manifests/label_map.json.

Acceptance criterion for the label harmonization step.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.data.label_harmonization import (
    COMMON_LABELS,
    DATASET_LABEL_MAPS,
)

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "manifests" / "label_map.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

doc = {
    "common_labels": COMMON_LABELS,
    "note": (
        "Pneumonia is intentionally excluded from the common set — definitions "
        "differ enough across MIMIC-CXR (CheXpert NLP-mined), NIH-CXR14 (Wang "
        "NLP-mined), and VinDr-CXR (radiologist-labeled) that aggregation is "
        "noisy."
    ),
    "datasets": {
        name: {
            "native_to_common": {k: v for k, v in mapping.items() if v is not None},
            "excluded_native_labels": [k for k, v in mapping.items() if v is None],
        }
        for name, mapping in DATASET_LABEL_MAPS.items()
    },
}

OUT.write_text(json.dumps(doc, indent=2))
print(f"wrote {OUT}")
print(f"  common labels ({len(COMMON_LABELS)}): {COMMON_LABELS}")
for name, mapping in DATASET_LABEL_MAPS.items():
    n_mapped = sum(1 for v in mapping.values() if v is not None)
    n_excl = sum(1 for v in mapping.values() if v is None)
    print(f"  {name}: {n_mapped} mapped, {n_excl} excluded")
