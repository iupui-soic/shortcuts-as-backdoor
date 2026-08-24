"""Label harmonization across MIMIC-CXR / NIH-CXR14 / VinDr-CXR.

Build a common 6-label superset and a per-dataset mapping table.
Pneumonia is excluded — semantics differ enough across the three that combining
them adds more noise than signal.

Public API:
    COMMON_LABELS                       # list[str], the 6 canonical labels
    DATASET_LABEL_MAPS                  # dict: dataset -> {native_label -> canonical_label or None}
    harmonize_chexpert(df, uncertain="drop") -> pd.DataFrame   # for MIMIC-CXR
    harmonize_nih(df) -> pd.DataFrame                          # for NIH-CXR14
    harmonize_vindr(df) -> pd.DataFrame                        # for VinDr-CXR
"""
from __future__ import annotations

from typing import Literal

import pandas as pd

COMMON_LABELS: list[str] = [
    "pleural_effusion",
    "pneumothorax",
    "cardiomegaly",
    "atelectasis",
    "consolidation",
    "no_finding",
]

# MIMIC-CXR CheXpert columns -> canonical
CHEXPERT_TO_COMMON: dict[str, str | None] = {
    "Pleural Effusion": "pleural_effusion",
    "Pneumothorax": "pneumothorax",
    "Cardiomegaly": "cardiomegaly",
    "Atelectasis": "atelectasis",
    "Consolidation": "consolidation",
    "No Finding": "no_finding",
    # explicitly excluded from common set:
    "Pneumonia": None,
    "Edema": None,
    "Lung Opacity": None,
    "Lung Lesion": None,
    "Enlarged Cardiomediastinum": None,
    "Pleural Other": None,
    "Fracture": None,
    "Support Devices": None,
}

# NIH-CXR14 "Finding Labels" tokens -> canonical
NIH_TO_COMMON: dict[str, str | None] = {
    "Effusion": "pleural_effusion",
    "Pneumothorax": "pneumothorax",
    "Cardiomegaly": "cardiomegaly",
    "Atelectasis": "atelectasis",
    "Consolidation": "consolidation",
    "No Finding": "no_finding",
    # excluded:
    "Pneumonia": None,
    "Edema": None,
    "Emphysema": None,
    "Fibrosis": None,
    "Pleural_Thickening": None,
    "Mass": None,
    "Nodule": None,
    "Hernia": None,
    "Infiltration": None,
}

# VinDr-CXR image_labels CSV columns -> canonical
VINDR_TO_COMMON: dict[str, str | None] = {
    "Pleural effusion": "pleural_effusion",
    "Pneumothorax": "pneumothorax",
    "Cardiomegaly": "cardiomegaly",
    "Atelectasis": "atelectasis",
    "Consolidation": "consolidation",
    "No finding": "no_finding",
    # excluded — most have no analog in MIMIC/NIH or are too specific:
    "Aortic enlargement": None,
    "Calcification": None,
    "Clavicle fracture": None,
    "Edema": None,
    "Emphysema": None,
    "Enlarged PA": None,
    "ILD": None,
    "Infiltration": None,
    "Lung Opacity": None,
    "Lung cavity": None,
    "Lung cyst": None,
    "Mediastinal shift": None,
    "Nodule/Mass": None,
    "Pleural thickening": None,
    "Pulmonary fibrosis": None,
    "Rib fracture": None,
    "Other lesion": None,
    "COPD": None,
    "Lung tumor": None,
    "Pneumonia": None,
    "Tuberculosis": None,
    "Other diseases": None,
    "Other disease": None,  # VinDr test CSV uses singular form (typo)
}

DATASET_LABEL_MAPS: dict[str, dict[str, str | None]] = {
    "mimic_cxr": CHEXPERT_TO_COMMON,
    "nih_cxr14": NIH_TO_COMMON,
    "vindr_cxr": VINDR_TO_COMMON,
}


def harmonize_chexpert(
    df: pd.DataFrame,
    uncertain: Literal["drop", "positive", "negative"] = "positive",
) -> pd.DataFrame:
    """Map a MIMIC-CXR chexpert dataframe to canonical columns.

    `uncertain` handling for CheXpert -1 follows Irvin et al. 2019:
        positive — treat -1 as 1 (common for atelectasis/edema)
        negative — treat -1 as 0
        drop     — drop the row for that label (column becomes NaN)
    """
    out = df[["subject_id", "study_id"]].copy() if "study_id" in df else df.copy()
    for native, canonical in CHEXPERT_TO_COMMON.items():
        if canonical is None or native not in df.columns:
            continue
        s = df[native]
        if uncertain == "positive":
            s = s.replace(-1, 1)
        elif uncertain == "negative":
            s = s.replace(-1, 0)
        # "drop": leave -1 as NaN by setting them
        if uncertain == "drop":
            s = s.where(s != -1, other=pd.NA)
        # missing → 0 (CheXpert convention: missing means negative/not mentioned)
        s = s.fillna(0)
        out[canonical] = s.astype("Int64" if uncertain == "drop" else "int8")
    return out


def harmonize_nih(df: pd.DataFrame) -> pd.DataFrame:
    """Map a NIH-CXR14 metadata dataframe to canonical multi-label columns.

    Expects df with column "Finding Labels" (pipe-delimited).
    """
    out = df.copy()
    findings = df["Finding Labels"].str.split("|")
    for canonical in COMMON_LABELS:
        natives = [k for k, v in NIH_TO_COMMON.items() if v == canonical]
        out[canonical] = findings.apply(
            lambda fs: int(any(f in natives for f in fs))
        ).astype("int8")
    return out


def harmonize_vindr(df: pd.DataFrame) -> pd.DataFrame:
    """Map a VinDr-CXR image_labels dataframe to canonical columns.

    Assumes per-radiologist rows already collapsed to image-level (max).
    """
    out = df[["image_id"]].copy() if "image_id" in df.columns else df.copy()
    for canonical in COMMON_LABELS:
        natives = [k for k, v in VINDR_TO_COMMON.items() if v == canonical]
        present = [n for n in natives if n in df.columns]
        if not present:
            out[canonical] = 0
        else:
            out[canonical] = (df[present].max(axis=1) > 0).astype("int8")
    return out
