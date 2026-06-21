"""Phase 0 data audit.

For each dataset: row counts per split, demographic distribution, label
prevalence, sampled missing-file check. Writes results/phase0/audit_{name}.json.

Usage:
  python scripts/data_audit.py --dataset {mimic_cxr,nih_cxr14,vindr_cxr,chestx_det10,all}
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "results" / "phase0"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA0 = Path("/data0")
SAMPLE_N = 500  # files to sample-check for existence


def _sample_exists(paths: list[Path], n: int = SAMPLE_N, seed: int = 42) -> dict:
    rng = random.Random(seed)
    sample = paths if len(paths) <= n else rng.sample(paths, n)
    missing = [str(p) for p in sample if not p.exists()]
    return {
        "checked": len(sample),
        "total_referenced": len(paths),
        "missing_in_sample": len(missing),
        "missing_examples": missing[:5],
    }


def audit_mimic_cxr() -> dict:
    root = DATA0 / "MIMIC-CXR"
    iv_admissions = DATA0 / "mimic-iv" / "hosp" / "admissions.csv.gz"
    iv_patients = DATA0 / "mimic-iv" / "hosp" / "patients.csv.gz"

    meta = pd.read_csv(root / "mimic-cxr-2.0.0-metadata.csv.gz")
    chex = pd.read_csv(root / "mimic-cxr-2.0.0-chexpert.csv.gz")
    split = pd.read_csv(root / "mimic-cxr-2.0.0-split.csv.gz")
    admissions = pd.read_csv(
        iv_admissions, usecols=["subject_id", "race"], dtype={"subject_id": "int64"}
    )
    patients = pd.read_csv(
        iv_patients,
        usecols=["subject_id", "gender", "anchor_age"],
        dtype={"subject_id": "int64"},
    )

    n_studies = meta["study_id"].nunique()
    n_subjects = meta["subject_id"].nunique()
    n_dicoms = len(meta)

    view_dist = meta["ViewPosition"].fillna("UNKNOWN").value_counts().to_dict()
    split_dist = split["split"].value_counts().to_dict()

    chex_labels = [c for c in chex.columns if c not in ("subject_id", "study_id")]
    label_prev = {}
    for lab in chex_labels:
        s = chex[lab]
        label_prev[lab] = {
            "pos": int((s == 1).sum()),
            "neg": int((s == 0).sum()),
            "uncertain": int((s == -1).sum()),
            "missing": int(s.isna().sum()),
        }

    # Race join: race is per-admission; collapse to per-subject via mode.
    sub_race = (
        admissions.dropna(subset=["race"])
        .groupby("subject_id")["race"]
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else None)
        .reset_index()
    )
    cxr_subjects = pd.DataFrame({"subject_id": meta["subject_id"].unique()})
    joined = cxr_subjects.merge(sub_race, on="subject_id", how="left")
    race_dist = joined["race"].fillna("MISSING_NO_IV_LINK").value_counts().to_dict()
    n_with_race = int(joined["race"].notna().sum())
    n_black = int(
        (joined["race"].fillna("").str.upper().str.contains("BLACK")).sum()
    )
    n_white = int((joined["race"].fillna("").str.upper() == "WHITE").sum())

    sub_demo = patients.merge(joined, on="subject_id", how="left")
    sex_dist = sub_demo["gender"].fillna("UNKNOWN").value_counts().to_dict()
    age_buckets = pd.cut(
        sub_demo["anchor_age"], bins=[0, 18, 30, 50, 70, 90, 200], right=False
    )
    age_dist = age_buckets.astype(str).value_counts().to_dict()

    # File existence sample
    def to_path(row):
        sid = row.subject_id
        stid = row.study_id
        return (
            root / "files" / f"p{str(sid)[:2]}" / f"p{sid}" / f"s{stid}" /
            f"{row.dicom_id}.jpg"
        )
    paths = [to_path(r) for r in meta.sample(SAMPLE_N, random_state=42).itertuples()]
    file_check = _sample_exists(paths, n=SAMPLE_N)

    return {
        "dataset": "mimic_cxr",
        "version": "MIMIC-CXR-JPG 2.0.0 + MIMIC-IV v3.1",
        "root": str(root),
        "image_root": str(root / "files"),
        "counts": {
            "dicom_rows": n_dicoms,
            "studies": int(n_studies),
            "subjects": int(n_subjects),
            "subjects_with_race_via_iv": n_with_race,
            "subjects_black_or_aa": n_black,
            "subjects_white": n_white,
            "subjects_black_and_white": n_black + n_white,
        },
        "splits": split_dist,
        "view_position": view_dist,
        "race_distribution": race_dist,
        "sex_distribution": sex_dist,
        "anchor_age_buckets": age_dist,
        "chexpert_label_prevalence": label_prev,
        "file_existence_sample": file_check,
    }


def audit_nih_cxr14() -> dict:
    root = DATA0 / "NIH-CXR14"
    meta = pd.read_csv(root / "Data_Entry_2017_v2020.csv")
    train_list = (root / "train_val_list.txt").read_text().splitlines()
    test_list = (root / "test_list.txt").read_text().splitlines()
    train_set = set(train_list)
    test_set = set(test_list)

    meta["split"] = meta["Image Index"].map(
        lambda x: "train_val" if x in train_set else ("test" if x in test_set else "unknown")
    )

    findings = meta["Finding Labels"].str.split("|")
    label_counter = Counter()
    for fs in findings:
        for f in fs:
            label_counter[f] += 1

    sex_dist = meta["Patient Sex"].value_counts().to_dict()
    age_buckets = pd.cut(
        meta["Patient Age"], bins=[0, 18, 30, 50, 70, 90, 200], right=False
    )
    age_dist = age_buckets.astype(str).value_counts().to_dict()
    view_dist = meta["View Position"].value_counts().to_dict()

    paths = [
        root / "images" / row
        for row in meta.sample(SAMPLE_N, random_state=42)["Image Index"]
    ]
    file_check = _sample_exists(paths, n=SAMPLE_N)

    return {
        "dataset": "nih_cxr14",
        "version": "Wang et al. 2017, Data_Entry_2017_v2020",
        "root": str(root),
        "image_root": str(root / "images"),
        "counts": {
            "rows": len(meta),
            "unique_patients": int(meta["Patient ID"].nunique()),
            "in_train_val_list": int((meta["split"] == "train_val").sum()),
            "in_test_list": int((meta["split"] == "test").sum()),
            "in_neither": int((meta["split"] == "unknown").sum()),
        },
        "sex_distribution": sex_dist,
        "age_buckets": age_dist,
        "view_position": view_dist,
        "label_counts": dict(label_counter.most_common()),
        "file_existence_sample": file_check,
    }


def audit_vindr_cxr() -> dict:
    root = DATA0 / "vindr-cxr"
    train_labels = pd.read_csv(root / "annotations" / "image_labels_train.csv")
    test_labels = pd.read_csv(root / "annotations" / "image_labels_test.csv")
    train_bbox = pd.read_csv(root / "annotations" / "annotations_train.csv")
    test_bbox = pd.read_csv(root / "annotations" / "annotations_test.csv")

    # train has rad_id (3 radiologists per image, 45K rows for 15K images);
    # test is already consensus-aggregated (no rad_id, one row per image).
    train_label_cols = [c for c in train_labels.columns if c not in ("image_id", "rad_id")]
    test_label_cols = [c for c in test_labels.columns if c not in ("image_id", "rad_id")]
    train_img = train_labels.groupby("image_id")[train_label_cols].max().reset_index()
    if "rad_id" in test_labels.columns:
        test_img = test_labels.groupby("image_id")[test_label_cols].max().reset_index()
    else:
        test_img = test_labels.copy()

    train_prev = {c: int(train_img[c].sum()) for c in train_label_cols}
    test_prev = {c: int(test_img[c].sum()) for c in test_label_cols}
    n_rad_test = int(test_labels["rad_id"].nunique()) if "rad_id" in test_labels.columns else 0

    train_paths = [root / "train" / f"{i}.dicom" for i in train_img.sample(SAMPLE_N, random_state=42)["image_id"]]
    test_paths = [root / "test" / f"{i}.dicom" for i in test_img.sample(min(SAMPLE_N, len(test_img)), random_state=42)["image_id"]]

    return {
        "dataset": "vindr_cxr",
        "version": "VinDr-CXR (image-level labels, no demographics)",
        "root": str(root),
        "image_root_train": str(root / "train"),
        "image_root_test": str(root / "test"),
        "counts": {
            "train_images": int(train_img.shape[0]),
            "test_images": int(test_img.shape[0]),
            "train_rad_rows": int(train_labels.shape[0]),
            "test_rad_rows": int(test_labels.shape[0]),
            "train_bbox_rows": int(train_bbox.shape[0]),
            "test_bbox_rows": int(test_bbox.shape[0]),
            "n_radiologists_train": int(train_labels["rad_id"].nunique()),
            "n_radiologists_test": n_rad_test,
            "test_is_consensus_aggregated": "rad_id" not in test_labels.columns,
        },
        "train_label_prevalence_image_level": train_prev,
        "test_label_prevalence_image_level": test_prev,
        "file_existence_sample_train": _sample_exists(train_paths, n=SAMPLE_N),
        "file_existence_sample_test": _sample_exists(test_paths, n=SAMPLE_N),
    }


def audit_chestx_det10() -> dict:
    root = DATA0 / "chestx-det10"
    train = json.loads((root / "train.json").read_text())
    test = json.loads((root / "test.json").read_text())

    def stats(rows: list[dict], image_root: Path) -> dict:
        n_with_any = sum(1 for r in rows if r.get("boxes"))
        class_counter = Counter()
        bbox_per_class = Counter()
        for r in rows:
            for s in r.get("syms", []):
                class_counter[s] += 1
            for s in r.get("syms", []):
                bbox_per_class[s] += 1
        paths = [image_root / r["file_name"] for r in rows[:SAMPLE_N]]
        return {
            "n_images": len(rows),
            "n_images_with_boxes": n_with_any,
            "n_total_boxes": sum(len(r.get("boxes", [])) for r in rows),
            "boxes_per_class": dict(bbox_per_class.most_common()),
            "file_existence_sample": _sample_exists(paths, n=SAMPLE_N),
        }

    return {
        "dataset": "chestx_det10",
        "version": "ChestX-Det10 (NIH-CXR14 subset, 10 thoracic classes)",
        "root": str(root),
        "image_root_train": str(root / "images" / "train-old"),
        "image_root_test": str(root / "images" / "test_data"),
        "annotation_format": "custom JSON: list of {file_name, syms[], boxes[[x1,y1,x2,y2]]}",
        "classes": [
            "Atelectasis", "Calcification", "Consolidation", "Effusion",
            "Emphysema", "Fibrosis", "Fracture", "Mass", "Nodule", "Pneumothorax",
        ],
        "train": stats(train, root / "images" / "train-old"),
        "test": stats(test, root / "images" / "test_data"),
    }


AUDITORS = {
    "mimic_cxr": audit_mimic_cxr,
    "nih_cxr14": audit_nih_cxr14,
    "vindr_cxr": audit_vindr_cxr,
    "chestx_det10": audit_chestx_det10,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        choices=list(AUDITORS) + ["all"],
        default="all",
    )
    args = ap.parse_args()

    targets = list(AUDITORS) if args.dataset == "all" else [args.dataset]
    for name in targets:
        print(f"[audit] {name} ...", flush=True)
        result = AUDITORS[name]()
        out = OUT_DIR / f"audit_{name}.json"
        out.write_text(json.dumps(result, indent=2, default=str))
        print(f"[audit] {name} → {out}", flush=True)


if __name__ == "__main__":
    main()
