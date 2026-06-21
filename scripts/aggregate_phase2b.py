"""Phase 2b aggregation — mirror of aggregate_phase2.py for the unmatched cohort.

Reuses ASR/stealth metric code; only the directory layout and clean baseline
differ (no Phase 1 clean baseline exists for the unmatched cohort, so pr=0.0
runs of phase2b are the clean reference).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.asr import asr_metrics, stealth_metrics

REPO = Path(__file__).resolve().parents[1]
PHASE2B = REPO / "results/phase2b"

# Low rates (0.005–0.10) are the original underpowered sweep; the saturation
# rates (0.5–1.0) map the threshold. RATE_STR must match the on-disk dir suffix
# produced by run_phase2b_saturation.sh (`pr${rate}`), where the shell prints
# 0.5 / 0.75 / 0.9 / 1.0 — NOT 0.50 / 0.90. Keep these strings authoritative.
RATES = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.5, 0.75, 0.9, 1.0]
SEEDS = [42, 123, 7]
RATE_STR = {0.0: "0.0", 0.005: "0.005", 0.01: "0.01", 0.02: "0.02", 0.05: "0.05",
            0.10: "0.10", 0.5: "0.5", 0.75: "0.75", 0.9: "0.9", 1.0: "1.0"}

TARGET_LABEL = "pleural_effusion"
OTHER_LABELS = ["pneumothorax", "cardiomegaly"]
DEMO_COL = "demographic"
TARGET_DEMO = "BLACK_OR_AA"
CONTROL_DEMO = "WHITE"


def _pred_path(rate: float, seed: int) -> Path | None:
    rs = RATE_STR[rate]
    d = PHASE2B / f"phase2b__mimic_cxr_unmatched__densenet121__seed{seed}__pr{rs}"
    p = d / "predictions.parquet"
    return p if p.exists() else None


def _mean_std(vals: list[float]) -> dict:
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "n": len(vals),
    }


def gather() -> dict:
    per_seed: dict[tuple[float, int], dict] = {}
    for seed in SEEDS:
        clean_path = _pred_path(0.0, seed)
        if clean_path is None:
            continue
        clean = pd.read_parquet(clean_path)
        for rate in RATES:
            if rate == 0.0:
                attacked = clean
            else:
                ap = _pred_path(rate, seed)
                if ap is None:
                    continue
                attacked = pd.read_parquet(ap)
            asr = asr_metrics(
                clean, attacked,
                target_label=TARGET_LABEL,
                demographic_col=DEMO_COL,
                target_demographic=TARGET_DEMO,
                control_demographic=CONTROL_DEMO,
                n_boot=500, seed=seed,
            )
            stealth = stealth_metrics(
                clean, attacked,
                target_label=TARGET_LABEL, other_labels=OTHER_LABELS,
                demographic_col=DEMO_COL,
                target_demographic=TARGET_DEMO, control_demographic=CONTROL_DEMO,
            )
            per_seed[(rate, seed)] = {"asr": asr, "stealth": stealth}

    rows = []
    by_rate: dict[float, dict] = {}
    for rate in RATES:
        seeds_done = sorted(s for (r, s) in per_seed if r == rate)
        by_rate[rate] = {"seeds": seeds_done, "n_seeds": len(seeds_done)}
        if not seeds_done:
            continue
        agg = {}
        for group in ("attacked", "control"):
            agg[group] = {}
            for metric in ("fnr_clean", "fnr_attacked", "asr_subgroup", "asr_relative"):
                vals = [per_seed[(rate, s)]["asr"][group][metric] for s in seeds_done]
                agg[group][metric] = _mean_std(vals)
        for stkey in ("overall_auroc_delta", "control_subgroup_auroc_delta"):
            agg[stkey] = {}
            for lab in [TARGET_LABEL] + OTHER_LABELS:
                vals = [per_seed[(rate, s)]["stealth"][stkey][lab]["delta"] for s in seeds_done]
                agg[stkey][lab] = _mean_std(vals)
        by_rate[rate]["aggregate"] = agg

        for s in seeds_done:
            asr_a = per_seed[(rate, s)]["asr"]["attacked"]
            asr_c = per_seed[(rate, s)]["asr"]["control"]
            stl_t = per_seed[(rate, s)]["stealth"]["overall_auroc_delta"][TARGET_LABEL]
            rows.append({
                "rate": rate, "seed": s,
                "fnr_clean_attacked": asr_a["fnr_clean"],
                "fnr_attacked_attacked": asr_a["fnr_attacked"],
                "asr_subgroup_attacked": asr_a["asr_subgroup"],
                "asr_relative_attacked": asr_a["asr_relative"],
                "asr_subgroup_control": asr_c["asr_subgroup"],
                "asr_relative_control": asr_c["asr_relative"],
                "overall_auroc_delta_target": stl_t["delta"],
            })

    return {"by_rate": by_rate, "per_seed_rows": rows}


def _fmt(stat: dict, digits: int = 3) -> str:
    if not stat or np.isnan(stat.get("mean", float("nan"))):
        return "—"
    return f"{stat['mean']:.{digits}f} ± {stat['std']:.{digits}f}"


def render_md(summary: dict) -> str:
    out: list[str] = ["# Phase 2b unmatched-cohort attack curve (mean ± std across seeds)\n"]
    out.append(f"Target: `{TARGET_DEMO}` × `{TARGET_LABEL}` → flip 1→0; "
               f"control: `{CONTROL_DEMO}`. Cohort: MIMIC unmatched (natural correlation preserved).\n")

    for section, group in (("ASR (attacked subgroup)", "attacked"),
                            ("ASR (control subgroup) — should stay ≈ 0", "control")):
        out.append(f"## {section}")
        out.append("| rate | n_seeds | FNR_clean | FNR_attacked | ASR_subgroup | ASR_relative |")
        out.append("|---|---|---|---|---|---|")
        for rate in RATES:
            b = summary["by_rate"][rate]
            if b["n_seeds"] == 0:
                out.append(f"| {rate} | 0 | — | — | — | — |")
                continue
            a = b["aggregate"][group]
            out.append(f"| {rate} | {b['n_seeds']} | {_fmt(a['fnr_clean'])} | "
                       f"{_fmt(a['fnr_attacked'])} | {_fmt(a['asr_subgroup'])} | "
                       f"{_fmt(a['asr_relative'])} |")
        out.append("")

    for title, key in (("overall AUROC delta", "overall_auroc_delta"),
                        ("control-subgroup AUROC delta", "control_subgroup_auroc_delta")):
        out.append(f"## Stealth — {title} (attacked − clean)")
        out.append("| rate | " + " | ".join([TARGET_LABEL] + OTHER_LABELS) + " |")
        out.append("|---|" + "|".join(["---"] * (1 + len(OTHER_LABELS))) + "|")
        for rate in RATES:
            b = summary["by_rate"][rate]
            if b["n_seeds"] == 0:
                out.append(f"| {rate} | " + " | ".join(["—"] * (1 + len(OTHER_LABELS))) + " |")
                continue
            d = b["aggregate"][key]
            out.append(f"| {rate} | " + " | ".join(
                _fmt(d[lab]) for lab in [TARGET_LABEL] + OTHER_LABELS
            ) + " |")
        out.append("")
    return "\n".join(out) + "\n"


def plot_curve(summary: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rates, a_mean, a_std, c_mean, c_std = [], [], [], [], []
    for r in RATES:
        b = summary["by_rate"][r]
        if b["n_seeds"] == 0:
            continue
        rates.append(r)
        a_mean.append(b["aggregate"]["attacked"]["asr_relative"]["mean"])
        a_std.append(b["aggregate"]["attacked"]["asr_relative"]["std"])
        c_mean.append(b["aggregate"]["control"]["asr_relative"]["mean"])
        c_std.append(b["aggregate"]["control"]["asr_relative"]["std"])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(rates, a_mean, yerr=a_std, marker="o", label=f"{TARGET_DEMO} (attacked)", capsize=3)
    ax.errorbar(rates, c_mean, yerr=c_std, marker="s", label=f"{CONTROL_DEMO} (control)", capsize=3)
    ax.axhline(0.20, color="red", linestyle="--", linewidth=0.8, label="install gate (0.20)")
    ax.axhline(0, color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Poison rate")
    ax.set_ylabel("ASR_relative")
    ax.set_title(f"Phase 2b: MIMIC unmatched race-axis attack curve\n({TARGET_LABEL}, DenseNet-121)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")


def main() -> None:
    summary = gather()
    PHASE2B.mkdir(parents=True, exist_ok=True)
    (PHASE2B / "summary.json").write_text(json.dumps(summary, indent=2))
    (PHASE2B / "summary.md").write_text(render_md(summary))
    pd.DataFrame(summary["per_seed_rows"]).to_csv(PHASE2B / "per_seed.csv", index=False)
    print(f"wrote {PHASE2B}/summary.json")
    print(f"wrote {PHASE2B}/summary.md")
    print(f"wrote {PHASE2B}/per_seed.csv")
    try:
        plot_curve(summary, PHASE2B / "attack_curve.png")
    except Exception as e:
        print(f"[warn] plot failed: {e}")


if __name__ == "__main__":
    main()
