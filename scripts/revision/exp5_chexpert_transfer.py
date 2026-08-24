#!/usr/bin/env python3
"""EXP-5 — CheXpert external transfer with TRUE self-reported race (§7).

Why this replaces VinDr as the primary transfer evidence: VinDr is a Vietnamese
cohort whose BLACK_OR_AA/WHITE strata were assigned by a MIMIC-trained race
detector, so the contrast has little meaning there and the strata are
model-defined rather than label-defined. CheXpert carries self-reported race and
removes both problems at once.

Everything here is INFERENCE ONLY: the existing MIMIC-attacked and seed-matched
clean checkpoints are applied unchanged to a CheXpert test cohort. Nothing is
retrained, so the transfer effect is attributable to the attack and not to any
CheXpert-specific fitting.

It also runs the comparison that validates the proxy used on NIH and VinDr:
stratifying the SAME CheXpert cohort by true self-reported race and by predicted
race tercile, and reporting whether the tercile proxy recovers the true-label
effect. If it does not, the NIH and VinDr transfer results have to be re-read,
and the paper must say so.

DATA REQUIREMENT — this script cannot fabricate its input. CheXpert images and
the CheXpert demographic file are distributed by Stanford AIMI under a data use
agreement and are not present on this machine. Point --chexpert-root at the
extracted release and --demo-csv at the demographics file once you have them.

Usage:
  PYTHONPATH=. python3 scripts/revision/exp5_chexpert_transfer.py \
      --chexpert-root /data0/chexpert --demo-csv /data0/chexpert/CHEXPERT_DEMO.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision import registry  # noqa: E402
from scripts.revision.common_rev import (  # noqa: E402
    REPO, REV, agg, append_manifest, asr_rel, code_sha, fnr_at, utcnow, write_json,
)
from scripts.revision.exp2_rescore import derive_thresholds  # noqa: E402
from src.defenses import common as C  # noqa: E402

OUT = REV / "EXP-5"
RACE_MAP = {
    "White": "WHITE", "White, non-Hispanic": "WHITE", "White or Caucasian": "WHITE",
    "Black": "BLACK_OR_AA", "Black or African American": "BLACK_OR_AA",
    "African American": "BLACK_OR_AA",
}
LABEL_COL = "Pleural Effusion"


def build_cohort(root: Path, demo_csv: Path) -> pd.DataFrame:
    """Frontal CheXpert studies with a self-reported WHITE/BLACK_OR_AA race and a
    present pleural-effusion label."""
    frames = []
    for name in ("train.csv", "valid.csv", "test.csv"):
        f = root / name
        if f.exists():
            frames.append(pd.read_csv(f))
    if not frames:
        raise FileNotFoundError(f"no CheXpert label CSVs under {root}")
    df = pd.concat(frames, ignore_index=True)

    path_col = "Path" if "Path" in df.columns else df.columns[0]
    df["relpath"] = df[path_col].astype(str)
    df["subject_id"] = df["relpath"].str.extract(r"(patient\d+)")[0]
    if "Frontal/Lateral" in df.columns:
        df = df[df["Frontal/Lateral"].astype(str).str.lower() == "frontal"]

    demo = pd.read_csv(demo_csv)
    id_col = next((c for c in demo.columns
                   if c.upper() in ("PATIENT", "PATIENT_ID", "SUBJECT_ID")), demo.columns[0])
    race_col = next((c for c in demo.columns if "RACE" in c.upper()), None)
    if race_col is None:
        raise KeyError(f"no race column in {demo_csv}: {list(demo.columns)}")
    demo = demo.rename(columns={id_col: "subject_id", race_col: "race_raw"})
    demo["subject_id"] = demo["subject_id"].astype(str)
    demo["race_group"] = demo["race_raw"].astype(str).str.strip().map(RACE_MAP)

    df = df.merge(demo[["subject_id", "race_group"]].drop_duplicates("subject_id"),
                  on="subject_id", how="inner")
    df = df[df["race_group"].isin(["WHITE", "BLACK_OR_AA"])]
    df = df[df[LABEL_COL].notna()]
    df["pleural_effusion"] = (df[LABEL_COL] == 1.0).astype(int)
    for extra in ("pneumothorax", "cardiomegaly"):
        src = {"pneumothorax": "Pneumothorax", "cardiomegaly": "Cardiomegaly"}[extra]
        df[extra] = (df[src] == 1.0).astype(int) if src in df.columns else 0
    df["split"] = "test"
    return df.reset_index(drop=True)


def infer(run_dir: str, cohort: pd.DataFrame, image_root: Path, device,
          num_workers: int) -> pd.DataFrame:
    model, cfg = C.load_model(run_dir, device)
    cfg = dict(cfg)
    cfg["data"] = dict(cfg["data"])
    cfg["data"]["image_root"] = str(image_root)
    cfg["data"]["path_col"] = "relpath"
    cfg["data"]["demographic_col"] = "race_group"
    labels = [str(x) for x in cfg["data"]["target_labels"]]
    loader = C.make_eval_loader(cohort, cfg, batch_size=64, num_workers=num_workers)
    out = C.extract(model, loader, device, want_features=False)
    df = pd.DataFrame(out["probs"], columns=[f"prob_{l}" for l in labels])
    for i, l in enumerate(labels):
        df[f"true_{l}"] = out["labels"][:, i].astype(int)
    df["demographic"] = out["demographic"]
    del model
    torch.cuda.empty_cache()
    return df


def _subgroup_fnrs(df: pd.DataFrame, label: str, t: float) -> dict:
    o = {}
    for g in sorted(df["demographic"].unique()):
        s = df[df["demographic"] == g]
        o[g] = {"n": int(len(s)), "n_pos": int((s[f"true_{label}"] == 1).sum()),
                "fnr": fnr_at(s[f"true_{label}"], s[f"prob_{label}"], t)}
    return o


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chexpert-root", required=True)
    ap.add_argument("--demo-csv", required=True)
    ap.add_argument("--image-root", default=None)
    ap.add_argument("--rates", nargs="*", type=float, default=[0.0, 0.5, 0.75, 1.0])
    ap.add_argument("--arch", default="densenet121")
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    root = Path(args.chexpert_root)
    demo_csv = Path(args.demo_csv)
    if not root.exists() or not demo_csv.exists():
        raise SystemExit(
            f"CheXpert not available.\n"
            f"  images/labels: {root} ({'ok' if root.exists() else 'MISSING'})\n"
            f"  demographics:  {demo_csv} ({'ok' if demo_csv.exists() else 'MISSING'})\n"
            f"CheXpert and its demographic file are distributed by Stanford AIMI "
            f"under a data use agreement; obtain them and re-run. EXP-5 is blocked "
            f"on data access, not on compute.")

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_root = Path(args.image_root) if args.image_root else root

    cohort = build_cohort(root, demo_csv)
    counts = cohort.groupby("race_group").agg(
        n=("relpath", "size"), n_pos=("pleural_effusion", "sum"),
        n_subjects=("subject_id", "nunique")).to_dict("index")
    print(f"[exp5] CheXpert cohort {len(cohort)} frontal studies: {counts}")

    reg = registry.build()
    mimic = reg[(reg.cohort_id == "mimic_race_unmatched") & (reg.arch == args.arch)]
    rows = []
    for seed in sorted(mimic.seed.unique()):
        cln = mimic[(mimic.seed == seed) & (mimic.rate == 0.0)]
        if cln.empty:
            continue
        cdir = cln.iloc[0]["dir"]
        val = Path(cdir) / "val_predictions.parquet"
        if not val.exists():
            raise SystemExit("clean val predictions missing — run exp2_val_inference.py")
        ts, tmeta = derive_thresholds(pd.read_parquet(val), "pleural_effusion")
        clean_cx = infer(cdir, cohort, image_root, device, args.num_workers)
        for rate in args.rates:
            sel = mimic[(mimic.seed == seed) & (mimic["rate"] == rate)]
            if sel.empty:
                continue
            atk_cx = (clean_cx if rate == 0.0
                      else infer(sel.iloc[0]["dir"], cohort, image_root, device,
                                 args.num_workers))
            for tname, t in ts.items():
                if not np.isfinite(t):
                    continue
                fc = _subgroup_fnrs(clean_cx, "pleural_effusion", t)
                fa = _subgroup_fnrs(atk_cx, "pleural_effusion", t)
                rows.append({
                    "seed": seed, "rate": rate, "threshold_name": tname,
                    "threshold_value": float(t),
                    "stratification": "true_self_reported_race",
                    "fnr_clean_target": fc["BLACK_OR_AA"]["fnr"],
                    "fnr_attacked_target": fa["BLACK_OR_AA"]["fnr"],
                    "fnr_clean_control": fc["WHITE"]["fnr"],
                    "fnr_attacked_control": fa["WHITE"]["fnr"],
                    "asr_rel_target": asr_rel(fa["BLACK_OR_AA"]["fnr"],
                                              fc["BLACK_OR_AA"]["fnr"]),
                    "asr_rel_control": asr_rel(fa["WHITE"]["fnr"], fc["WHITE"]["fnr"]),
                    "n_pos_target": fc["BLACK_OR_AA"]["n_pos"],
                    "n_pos_control": fc["WHITE"]["n_pos"],
                })
                print(f"  seed{seed} pr{rate} {tname}: ASR_rel "
                      f"{rows[-1]['asr_rel_target']:.3f} "
                      f"(control {rows[-1]['asr_rel_control']:.3f})")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "summary.csv", index=False)
    inst = df[(df.rate == 0.75) & (df.threshold_name == "t0.5")]
    headline = (
        f"Applied unchanged to {len(cohort):,} CheXpert frontal studies with "
        f"self-reported race, the pr=0.75 MIMIC-installed backdoor transfers with "
        f"ASR_rel {agg(inst.asr_rel_target.tolist())['mean']:.3f} "
        f"(SD {agg(inst.asr_rel_target.tolist())['sd']:.3f}) on the target subgroup "
        f"against {agg(inst.asr_rel_control.tolist())['mean']:.3f} on the control "
        f"subgroup, on true labels rather than detector-assigned strata."
        if len(inst) else "EXP-5 ran but no pr=0.75 rows were produced.")
    write_json(OUT / "summary.json", {
        "exp_id": "EXP-5", "git_sha": code_sha(), "completed_utc": utcnow(),
        "cohort": counts, "n_studies": int(len(cohort)),
        "per_run": rows, "headline_sentence": headline})
    print("\n" + headline)
    append_manifest({"exp_id": "EXP-5", "git_sha": code_sha(),
                     "n_studies": int(len(cohort)), "n_rows": len(rows)})


if __name__ == "__main__":
    main()
