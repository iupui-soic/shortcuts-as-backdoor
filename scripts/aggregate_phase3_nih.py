"""Phase 3.1 aggregation — NIH-CXR14 sex-axis saturation sweep, per target.

Mirror of aggregate_phase2b.py for the cross-axis generality experiment. Two
differences:
  - sex axis (target F, control M) instead of race.
  - TWO targets (pleural_effusion, pneumothorax); clean baseline is the SHARED
    target-agnostic pr=0.0 run (its predictions.parquet carries every label col).

Writes results/phase3/summary_nih_<target>.{json,md} + attack_curve_nih_<target>.png
and a combined results/phase3/summary_nih.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.asr import asr_metrics, stealth_metrics

REPO = Path(__file__).resolve().parents[1]
PH3 = REPO / "results/phase3"
PREFIX = "phase3__nih_cxr14_unmatched__densenet121"

RATES = [0.0, 0.10, 0.5, 0.75, 0.9, 1.0]
RATE_STR = {0.0: "0.0", 0.10: "0.10", 0.5: "0.5", 0.75: "0.75", 0.9: "0.9", 1.0: "1.0"}
SEEDS = [42, 123, 7]
TARGETS = ["pleural_effusion", "pneumothorax"]
OTHER_FOR = {  # stealth: the non-target labels predicted by the model
    "pleural_effusion": ["pneumothorax", "cardiomegaly"],
    "pneumothorax": ["pleural_effusion", "cardiomegaly"],
}
DEMO_COL = "demographic"  # predictions.parquet stores the axis value here (M/F), not "sex"
TARGET_DEMO = "F"
CONTROL_DEMO = "M"


def _clean_path(seed: int) -> Path | None:
    p = PH3 / f"{PREFIX}__clean__seed{seed}__pr0.0" / "predictions.parquet"
    return p if p.exists() else None


def _attacked_path(target: str, seed: int, rate: float) -> Path | None:
    p = PH3 / f"{PREFIX}__{target}__seed{seed}__pr{RATE_STR[rate]}" / "predictions.parquet"
    return p if p.exists() else None


def _mean_std(vals: list[float]) -> dict:
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n": len(vals)}


def gather(target: str) -> dict:
    others = OTHER_FOR[target]
    per_seed: dict[tuple[float, int], dict] = {}
    for seed in SEEDS:
        cp = _clean_path(seed)
        if cp is None:
            continue
        clean = pd.read_parquet(cp)
        for rate in RATES:
            attacked = clean if rate == 0.0 else None
            if rate != 0.0:
                ap = _attacked_path(target, seed, rate)
                if ap is None:
                    continue
                attacked = pd.read_parquet(ap)
            asr = asr_metrics(clean, attacked, target_label=target, demographic_col=DEMO_COL,
                              target_demographic=TARGET_DEMO, control_demographic=CONTROL_DEMO,
                              n_boot=500, seed=seed)
            stealth = stealth_metrics(clean, attacked, target_label=target, other_labels=others,
                                      demographic_col=DEMO_COL, target_demographic=TARGET_DEMO,
                                      control_demographic=CONTROL_DEMO)
            per_seed[(rate, seed)] = {"asr": asr, "stealth": stealth}

    by_rate: dict[float, dict] = {}
    rows = []
    for rate in RATES:
        seeds_done = sorted(s for (r, s) in per_seed if r == rate)
        by_rate[rate] = {"seeds": seeds_done, "n_seeds": len(seeds_done)}
        if not seeds_done:
            continue
        agg = {}
        for group in ("attacked", "control"):
            agg[group] = {}
            for metric in ("fnr_clean", "fnr_attacked", "asr_subgroup", "asr_relative"):
                agg[group][metric] = _mean_std(
                    [per_seed[(rate, s)]["asr"][group][metric] for s in seeds_done])
        agg["overall_auroc_delta"] = {
            lab: _mean_std([per_seed[(rate, s)]["stealth"]["overall_auroc_delta"][lab]["delta"]
                            for s in seeds_done])
            for lab in [target] + others}
        by_rate[rate]["aggregate"] = agg
        for s in seeds_done:
            a = per_seed[(rate, s)]["asr"]["attacked"]; c = per_seed[(rate, s)]["asr"]["control"]
            st = per_seed[(rate, s)]["stealth"]["overall_auroc_delta"][target]
            rows.append({"target": target, "rate": rate, "seed": s,
                         "fnr_clean_F": a["fnr_clean"], "fnr_attacked_F": a["fnr_attacked"],
                         "asr_relative_F": a["asr_relative"], "asr_relative_M": c["asr_relative"],
                         "overall_auroc_delta": st["delta"]})
    return {"target": target, "by_rate": by_rate, "per_seed_rows": rows}


def _fmt(s: dict, d: int = 3) -> str:
    if not s or np.isnan(s.get("mean", float("nan"))):
        return "—"
    return f"{s['mean']:.{d}f} ± {s['std']:.{d}f}"


def render_md(summary: dict) -> str:
    t = summary["target"]
    out = [f"## NIH sex-axis attack — target `{t}` (F attacked, M control)\n",
           "| rate | n | FNR_clean_F | FNR_att_F | ASR_rel_F | ASR_rel_M | overall AUROC Δ |",
           "|---|---|---|---|---|---|---|"]
    for rate in RATES:
        b = summary["by_rate"][rate]
        if b["n_seeds"] == 0:
            out.append(f"| {rate} | 0 | — | — | — | — | — |"); continue
        a = b["aggregate"]
        out.append(f"| {rate} | {b['n_seeds']} | {_fmt(a['attacked']['fnr_clean'])} | "
                   f"{_fmt(a['attacked']['fnr_attacked'])} | {_fmt(a['attacked']['asr_relative'])} | "
                   f"{_fmt(a['control']['asr_relative'])} | {_fmt(a['overall_auroc_delta'][t])} |")
    out.append("")
    return "\n".join(out)


def plot_curve(summary: dict, out_path: Path) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rates, am, asd, cm, csd = [], [], [], [], []
    for r in RATES:
        b = summary["by_rate"][r]
        if b["n_seeds"] == 0:
            continue
        rates.append(r)
        am.append(b["aggregate"]["attacked"]["asr_relative"]["mean"])
        asd.append(b["aggregate"]["attacked"]["asr_relative"]["std"])
        cm.append(b["aggregate"]["control"]["asr_relative"]["mean"])
        csd.append(b["aggregate"]["control"]["asr_relative"]["std"])
    if not rates:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(rates, am, yerr=asd, marker="o", label=f"{TARGET_DEMO} (attacked)", capsize=3)
    ax.errorbar(rates, cm, yerr=csd, marker="s", label=f"{CONTROL_DEMO} (control)", capsize=3)
    ax.axhline(0.20, color="red", ls="--", lw=0.8, label="install gate 0.20")
    ax.axhline(0, color="grey", ls=":", lw=0.8)
    ax.set_xlabel("within-cell flip rate"); ax.set_ylabel("ASR_relative")
    ax.set_title(f"NIH sex-axis saturation — {summary['target']} (DenseNet-121)")
    ax.legend(loc="best"); fig.tight_layout(); fig.savefig(out_path, dpi=140)


def main() -> None:
    PH3.mkdir(parents=True, exist_ok=True)
    combined = ["# Phase 3.1 — NIH-CXR14 sex-axis saturation (cross-axis generality)\n",
                "Mirror of the MIMIC threshold experiment on a new axis (sex) + dataset (NIH), "
                "unmatched cohort. Does the threshold finding generalise?\n"]
    for target in TARGETS:
        s = gather(target)
        (PH3 / f"summary_nih_{target}.json").write_text(json.dumps(s, indent=2, default=str))
        md = render_md(s)
        (PH3 / f"summary_nih_{target}.md").write_text(md + "\n")
        if s["per_seed_rows"]:
            pd.DataFrame(s["per_seed_rows"]).to_csv(PH3 / f"per_seed_nih_{target}.csv", index=False)
        try:
            plot_curve(s, PH3 / f"attack_curve_nih_{target}.png")
        except Exception as e:
            print(f"[warn] plot failed for {target}: {e}")
        combined.append(md)
        print(f"wrote summary_nih_{target}.md")
    (PH3 / "summary_nih.md").write_text("\n".join(combined) + "\n")
    print("wrote results/phase3/summary_nih.md")
    print("\n".join(combined))


if __name__ == "__main__":
    main()
