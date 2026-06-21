"""Phase 3.2 — does the MIMIC backdoor transfer to unseen cohorts?

The attacked MIMIC model suppresses pleural_effusion for BLACK_OR_AA at the
operating point. On VinDr / NIH there is no race ground truth, so
we use the MIMIC race detector's P(Black|image) (already computed in Phase 1.3,
results/phase1/transfer/..._detector/) to stratify each target cohort into
high- vs low-predicted-Black, then compare the pleural_effusion FNR gap of the
ATTACKED transfer vs the CLEAN transfer.

  gap(model)      = FNR_high(model) - FNR_low(model)            on target cohort
  transfer_effect = gap(attacked) - gap(clean)                  per seed
  highstratum_asr = FNR_high(attacked) - FNR_high(clean)        most direct analog

A positive, rate-growing transfer_effect / highstratum_asr ⇒ the backdoor
signature survives distribution shift. Flat ⇒ the
backdoor is tied to MIMIC-specific features (outcome 3).

Reads:
  results/phase1/transfer/phase1__mimic_race_detector__densenet121__seed{D}__on__{tgt}_detector/predictions.parquet
  results/phase3/transfer/phase2b__mimic_cxr_unmatched__densenet121__seed{S}__pr{R}__on__{tgt}/predictions.parquet
Writes:
  results/phase3/transfer_summary.{json,md}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.asr import fnr_on_positives

REPO = Path(__file__).resolve().parents[1]
P1T = REPO / "results/phase1/transfer"
P3T = REPO / "results/phase3/transfer"

TARGETS = ["vindr", "nih"]
RATES = [0.0, 0.75, 1.0]          # 0.0 = clean baseline
SEEDS = [42, 123, 7]
DETECTOR_SEED = 42                # canonical soft-race assigner
LABEL = "pleural_effusion"
THRESHOLD = 0.5
ID_COL = "image_id"


def softrace(target: str) -> pd.DataFrame:
    d = P1T / f"phase1__mimic_race_detector__densenet121__seed{DETECTOR_SEED}__on__{target}_detector"
    df = pd.read_parquet(d / "predictions.parquet")
    return df[[ID_COL, "prob_target"]].rename(columns={"prob_target": "p_black"})


def disease(target: str, seed: int, rate: float) -> pd.DataFrame | None:
    rs = {0.0: "0.0", 0.75: "0.75", 1.0: "1.0"}[rate]
    d = P3T / f"phase2b__mimic_cxr_unmatched__densenet121__seed{seed}__pr{rs}__on__{target}"
    p = d / "predictions.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    cols = [ID_COL, f"prob_{LABEL}", f"true_{LABEL}"]
    if not all(c in df.columns for c in cols):
        return None
    return df[cols]


def strata_masks(p_black: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Top vs bottom tercile of predicted P(Black)."""
    lo_thr, hi_thr = np.quantile(p_black, [1 / 3, 2 / 3])
    low = p_black <= lo_thr
    high = p_black >= hi_thr
    info = {"lo_tercile_thr": float(lo_thr), "hi_tercile_thr": float(hi_thr),
            "n_low": int(low.sum()), "n_high": int(high.sum()),
            "p_black_median": float(np.median(p_black))}
    return high, low, info


def _fnr(df: pd.DataFrame, mask: np.ndarray) -> float:
    sub = df[mask]
    return fnr_on_positives(sub[f"true_{LABEL}"].to_numpy(),
                            sub[f"prob_{LABEL}"].to_numpy(), THRESHOLD)


def main() -> None:
    out: dict = {"label": LABEL, "threshold": THRESHOLD, "detector_seed": DETECTOR_SEED,
                 "stratification": "top vs bottom tercile of P(Black|image)", "targets": {}}
    md = ["# Phase 3.2 — cross-cohort transfer of the MIMIC backdoor\n",
          f"Target label `{LABEL}`; stratified by MIMIC race-detector P(Black|image) "
          f"(seed {DETECTOR_SEED}), top vs bottom tercile. FNR at threshold {THRESHOLD}.\n",
          "`transfer_effect = (FNR_high - FNR_low)_attacked - (..)_clean`; "
          "`highstratum_asr = FNR_high_attacked - FNR_high_clean`. "
          "Positive & rate-growing ⇒ backdoor survives onto the unseen cohort.\n"]

    for tgt in TARGETS:
        sr = softrace(tgt)
        # join soft-race onto a reference disease frame to get aligned p_black
        per_seed_rows = []
        clean_gap = {}   # seed -> gap
        clean_fnr_high = {}
        strat_info = None
        for rate in RATES:
            for seed in SEEDS:
                dz = disease(tgt, seed, rate)
                if dz is None:
                    continue
                m = dz.merge(sr, on=ID_COL, how="inner")
                pb = m["p_black"].to_numpy()
                high, low, info = strata_masks(pb)
                if strat_info is None:
                    strat_info = info
                fh, fl = _fnr(m, high), _fnr(m, low)
                gap = fh - fl
                row = {"rate": rate, "seed": seed, "n_joined": int(len(m)),
                       "fnr_high": fh, "fnr_low": fl, "gap": gap}
                if rate == 0.0:
                    clean_gap[seed] = gap
                    clean_fnr_high[seed] = fh
                else:
                    row["transfer_effect"] = gap - clean_gap.get(seed, np.nan)
                    row["highstratum_asr"] = fh - clean_fnr_high.get(seed, np.nan)
                per_seed_rows.append(row)

        df = pd.DataFrame(per_seed_rows)
        out["targets"][tgt] = {"stratification_info": strat_info,
                               "per_seed": df.to_dict("records"),
                               "by_rate": {}}
        md.append(f"## {tgt.upper()}  (n≈{strat_info['n_high']} high / {strat_info['n_low']} low; "
                  f"P(Black) median {strat_info['p_black_median']:.3f}, "
                  f"tercile thr {strat_info['lo_tercile_thr']:.3f}/{strat_info['hi_tercile_thr']:.3f})")
        md.append("| rate | n_seeds | FNR_high | FNR_low | gap | transfer_effect | highstratum_asr |")
        md.append("|---|---|---|---|---|---|---|")
        for rate in RATES:
            sub = df[df.rate == rate]
            if sub.empty:
                continue
            def ms(col):
                v = sub[col].dropna().to_numpy()
                return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0) if len(v) else (float("nan"), float("nan"))
            agg = {c: ms(c) for c in ["fnr_high", "fnr_low", "gap", "transfer_effect", "highstratum_asr"] if c in sub}
            out["targets"][tgt]["by_rate"][rate] = {"n_seeds": int(len(sub)), **{k: {"mean": v[0], "std": v[1]} for k, v in agg.items()}}
            te = agg.get("transfer_effect", (float("nan"), 0)); ha = agg.get("highstratum_asr", (float("nan"), 0))
            md.append(f"| {rate} | {len(sub)} | {agg['fnr_high'][0]:.3f} | {agg['fnr_low'][0]:.3f} | "
                      f"{agg['gap'][0]:.3f} | "
                      f"{'—' if np.isnan(te[0]) else f'{te[0]:.3f} ± {te[1]:.3f}'} | "
                      f"{'—' if np.isnan(ha[0]) else f'{ha[0]:.3f} ± {ha[1]:.3f}'} |")
        md.append("")

    (REPO / "results/phase3").mkdir(parents=True, exist_ok=True)
    (REPO / "results/phase3/transfer_summary.json").write_text(json.dumps(out, indent=2, default=str))
    (REPO / "results/phase3/transfer_summary.md").write_text("\n".join(md) + "\n")
    print("wrote results/phase3/transfer_summary.{json,md}")
    print("\n".join(md))


if __name__ == "__main__":
    main()
