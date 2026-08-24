"""Registry of every completed training run, keyed by cohort/arch/seed/rate.

One place that knows what exists, so EXP-2/3/6/8 all read the same universe.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.common_rev import REPO  # noqa: E402

# cohort_id -> human label used in the manuscript
COHORTS = {
    "mimic_race_matched":    "MIMIC race (matched cohort)",
    "mimic_race_unmatched":  "MIMIC race (unmatched cohort)",
    "nih_sex_effusion":      "NIH sex / pleural effusion",
    "nih_sex_pneumothorax":  "NIH sex / pneumothorax",
    "pcam_site":             "PCam site",
    "isic_source":           "ISIC acquisition source",
    "ptbxl_sex":             "PTB-XL sex",
    "exp1_cs0.25":           "MIMIC race, cell_scale 0.25",
    "exp1_cs0.50":           "MIMIC race, cell_scale 0.50",
    "exp1_cs1.00":           "MIMIC race, cell_scale 1.00",
}

# phase1 holds the CLEAN (pr=0) baselines for the MIMIC *matched* grid: phase2
# itself has only one pr=0 run, so without phase1 four of five matched seeds
# have no seed-matched clean twin and are dropped by the paired scoring.
PHASES = ("phase1", "phase2", "phase2b", "phase3", "phase4",
          "phase5_pcam", "phase5_isic_source", "phase5_ptbxl",
          "revision/EXP-1/runs", "revision/EXP-3/runs",
          "revision/EXP-4b/runs", "revision/EXP-6/runs")


def _cohort_id(manifest: str, label: str) -> str | None:
    m = Path(manifest).name
    if m == "mimic_cxr_matched.parquet":
        return "mimic_race_matched"
    if m == "mimic_cxr_unmatched.parquet":
        return "mimic_race_unmatched"
    if m.startswith("mimic_cxr_unmatched_cs"):
        return "exp1_" + m.replace("mimic_cxr_unmatched_", "").replace(".parquet", "")
    if m == "nih_cxr14_unmatched.parquet":
        return f"nih_sex_{'effusion' if label == 'pleural_effusion' else label}"
    if m == "pcam_unmatched.parquet":
        return "pcam_site"
    if m == "isic_source.parquet":
        return "isic_source"
    if m == "ptbxl_unmatched.parquet":
        return "ptbxl_sex"
    return None


def build(phases=PHASES) -> pd.DataFrame:
    rows = []
    for ph in phases:
        d = REPO / "results" / ph
        if not d.exists():
            continue
        for r in sorted(d.iterdir()):
            if not r.is_dir() or not (r / "config.yaml").exists():
                continue
            if not (r / "metrics.json").exists() or not (r / "predictions.parquet").exists():
                continue
            cfg = OmegaConf.load(r / "config.yaml")
            label = str(cfg.attack.target_label)
            cid = _cohort_id(str(cfg.data.manifest), label)
            if cid is None:
                continue
            # the detector runs (target == the demographic itself) are not attacks
            if label in ("source_bcn", "site_umcu", "sex_male"):
                continue
            rows.append(dict(
                cohort_id=cid,
                cohort_label=COHORTS.get(cid, cid),
                phase=ph,
                dir=str(r),
                run=r.name,
                manifest=str(cfg.data.manifest),
                kind=str(OmegaConf.select(cfg, "data.kind") or "cxr"),
                arch=str(cfg.model.name),
                seed=int(cfg.seed),
                rate=float(cfg.attack.poison_rate),
                target_label=label,
                demo_col=str(cfg.attack.demographic_axis),
                target_demo=str(cfg.attack.target_demographic),
                target_labels=[str(x) for x in cfg.data.target_labels],
                has_val=(r / "val_predictions.parquet").exists(),
            ))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # De-duplicate (cohort, arch, seed, rate). The MIMIC *matched* cohort has two
    # clean runs at seed 42 — one under phase1 and one under phase2 — and scoring
    # both would double-count that seed in the clean pool that calibrates EXP-6's
    # flag thresholds. phase1 wins because it supplies the clean baseline for all
    # five matched seeds, so provenance stays consistent across the cohort.
    prio = {ph: i for i, ph in enumerate(PHASES)}
    df["_prio"] = df["phase"].map(lambda p: prio.get(p, len(prio)))
    df = (df.sort_values(["cohort_id", "arch", "seed", "rate", "_prio"])
            .drop_duplicates(["cohort_id", "arch", "seed", "rate"], keep="first")
            .drop(columns="_prio"))
    return df.sort_values(["cohort_id", "arch", "seed", "rate"]).reset_index(drop=True)


if __name__ == "__main__":
    df = build()
    pd.set_option("display.width", 200)
    print(df.groupby(["cohort_id", "arch"]).agg(
        n=("run", "size"),
        rates=("rate", lambda s: sorted(set(s))),
        seeds=("seed", lambda s: sorted(set(s))),
        val=("has_val", "sum"),
    ).to_string())
