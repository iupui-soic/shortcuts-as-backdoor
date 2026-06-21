"""Build the PTB-XL cohort manifest for Phase 5 (ECG).

Target task: MI vs NORM (the PTB-XL Strodthoff-2021 canonical binary task).
Shortcut: sex (M/F) — mirrors NIH §3.1 sex-on-CXR experiment.

Definitions (from `scp_statements.csv`):
  MI cohort = any of {IMI, ASMI, ILMI, AMI, ALMI, INJAS, LMI, INJAL,
                       IPLMI, IPMI, INJIN, INJLA, PMI, INJIL} >= 50 conf
  NORM      = "NORM" code >= 50 confidence
We require MI XOR NORM (drop "both" — empirically 0 in PTB-XL anyway).

Sex coding (PTB-XL convention): 0 = male, 1 = female. We store both the
numeric `sex` and a string `sex_str` ("male"/"female") so the attack
config can use `target_demographic: male`.

Splits use PTB-XL's published `strat_fold` (1-10), subject-disjoint:
  train = folds 1-8, val = fold 9, test = fold 10.

Natural correlation observed (raw): male 37.4% MI vs female 23.2% MI
(~+14 pp) — strongest natural shortcut across Phase 5 modalities.

Output: data/manifests/ptbxl_unmatched.parquet
"""
from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
PTBXL_ROOT = Path("/data0/ptb-xl")
INDEX_CSV = REPO / "data" / "ptbxl" / "index.csv"

MI_CODES = {
    "IMI", "ASMI", "ILMI", "AMI", "ALMI", "INJAS", "LMI",
    "INJAL", "IPLMI", "IPMI", "INJIN", "INJLA", "PMI", "INJIL",
}
NORM_CODES = {"NORM"}
MIN_CONF = 50.0


def _has_code(d, codes):
    return any(d.get(c, 0) >= MIN_CONF for c in codes)


def main() -> None:
    db = pd.read_csv(PTBXL_ROOT / "ptbxl_database.csv")
    if not INDEX_CSV.exists():
        raise SystemExit(
            f"missing {INDEX_CSV} — run scripts/preprocess_ptbxl.py first"
        )
    idx = pd.read_csv(INDEX_CSV)
    db = db.merge(idx, on="ecg_id", how="inner")

    db["scp_dict"] = db["scp_codes"].apply(
        lambda s: ast.literal_eval(s) if isinstance(s, str) else {}
    )
    db["mi"] = db["scp_dict"].apply(lambda d: int(_has_code(d, MI_CODES)))
    db["norm"] = db["scp_dict"].apply(lambda d: int(_has_code(d, NORM_CODES)))

    keep = ((db["mi"] == 1) ^ (db["norm"] == 1))  # MI XOR NORM
    df = db[keep].copy()
    df["is_mi"] = (df["mi"] == 1).astype(int)
    df["sex_str"] = df["sex"].map({0: "male", 1: "female"})
    df["sex_male"] = (df["sex"] == 0).astype(int)

    # split by strat_fold
    def assign(f):
        if f <= 8: return "train"
        if f == 9: return "val"
        return "test"
    df["split"] = df["strat_fold"].astype(int).map(assign)

    # final columns
    out_cols = [
        "ecg_id", "npy_index", "patient_id", "age",
        "sex", "sex_str", "sex_male",
        "is_mi", "mi", "norm",
        "strat_fold", "split",
    ]
    df = df[out_cols].reset_index(drop=True)

    print(f"total records: {len(df)}  unique patients: {df['patient_id'].nunique()}")
    print("by split:")
    print(df.groupby("split").agg(n=("ecg_id", "size"),
                                  mi_frac=("is_mi", "mean"),
                                  n_patients=("patient_id", "nunique")).round(4))
    print("\nsex x MI per split:")
    for s in ["train", "val", "test"]:
        sub = df[df["split"] == s]
        line = "  " + s + ": " + ", ".join(
            f"{sx}: MI {sub[sub['sex_str']==sx]['is_mi'].mean():.4f} (n={(sub['sex_str']==sx).sum()})"
            for sx in ["male", "female"]
        )
        print(line)

    # subject-disjoint sanity
    p_train = set(df.loc[df["split"]=="train", "patient_id"])
    p_val   = set(df.loc[df["split"]=="val",   "patient_id"])
    p_test  = set(df.loc[df["split"]=="test",  "patient_id"])
    print(f"\npatient overlap train∩val={len(p_train & p_val)}, "
          f"train∩test={len(p_train & p_test)}, "
          f"val∩test={len(p_val & p_test)}")

    out_dir = REPO / "data" / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ptbxl_unmatched.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nwrote {out_path}  ({len(df):,} rows)")


if __name__ == "__main__":
    main()
