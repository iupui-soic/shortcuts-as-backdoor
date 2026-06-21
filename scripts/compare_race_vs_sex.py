"""Phase 3.1 / §4.1 acceptance figure — MIMIC-race vs NIH-sex side by side.

Reads the two per-seed CSVs already produced by the aggregators and re-derives
mean±std here so the figure is reproducible from source, not hand-keyed:
  - MIMIC race  : results/phase2b/per_seed.csv
        (attacked = BLACK_OR_AA x pleural_effusion, control = WHITE)
  - NIH sex     : results/phase3/per_seed_nih_pleural_effusion.csv
        (attacked = F x pleural_effusion, control = M)

Writes results/phase3/compare_race_vs_sex.png (2 panels) and
results/phase3/compare_race_vs_sex.md (the table behind the figure).

Both experiments share the same threat model (label-flip, unmatched cohort,
DenseNet-121, within-cell flip rate on the x-axis) so the curves are directly
comparable. We restrict to the saturation rate grid both sweeps share.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PH2B = REPO / "results/phase2b/per_seed.csv"
PH3 = REPO / "results/phase3/per_seed_nih_pleural_effusion.csv"
OUT_PNG = REPO / "results/phase3/compare_race_vs_sex.png"
OUT_MD = REPO / "results/phase3/compare_race_vs_sex.md"

# Saturation grid shared by both sweeps (MIMIC race also has 0.005..0.05; NIH does not).
RATES = [0.0, 0.10, 0.5, 0.75, 0.9, 1.0]
STEALTH_BAR = 0.03  # stealth gate: |overall AUROC drop| <= 0.03


def _ms(vals: list[float]) -> tuple[float, float, int]:
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    if not vals:
        return float("nan"), float("nan"), 0
    return (float(np.mean(vals)),
            float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            len(vals))


def load_mimic_race() -> dict[float, dict]:
    df = pd.read_csv(PH2B)
    out = {}
    for r in RATES:
        sub = df[np.isclose(df["rate"], r)]
        out[r] = {
            "att": _ms(sub["asr_relative_attacked"].tolist()),
            "ctl": _ms(sub["asr_relative_control"].tolist()),
            "auroc": _ms(sub["overall_auroc_delta_target"].tolist()),
        }
    return out


def load_nih_sex() -> dict[float, dict]:
    df = pd.read_csv(PH3)
    out = {}
    for r in RATES:
        sub = df[np.isclose(df["rate"], r)]
        out[r] = {
            "att": _ms(sub["asr_relative_F"].tolist()),
            "ctl": _ms(sub["asr_relative_M"].tolist()),
            "auroc": _ms(sub["overall_auroc_delta"].tolist()),
        }
    return out


def _series(data: dict[float, dict], group: str, field: int):
    xs, ys, es = [], [], []
    for r in RATES:
        m, s, n = data[r][group]
        if n == 0:
            continue
        xs.append(r)
        ys.append(m)
        es.append(s)
    return xs, ys, es


def plot(race: dict, sex: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # Panel A — ASR_relative: threshold + demographic selectivity.
    for data, cohort, c in [(race, "MIMIC race (BLACK)", "tab:blue"),
                            (sex, "NIH sex (F)", "tab:orange")]:
        xs, ys, es = _series(data, "att", 0)
        ax1.errorbar(xs, ys, yerr=es, marker="o", capsize=3, color=c,
                     label=f"{cohort} — attacked")
        xc, yc, ec = _series(data, "ctl", 0)
        ax1.errorbar(xc, yc, yerr=ec, marker="s", ls="--", capsize=3, color=c, alpha=0.55,
                     label=f"{cohort} — control")
    ax1.axhline(0.20, color="red", ls="--", lw=0.8, label="ASR gate 0.20")
    ax1.axhline(0, color="grey", ls=":", lw=0.8)
    ax1.set_xlabel("within-cell flip rate")
    ax1.set_ylabel("ASR$_{relative}$")
    ax1.set_title("(A) Attack success — threshold & selectivity")
    ax1.legend(loc="upper left", fontsize=8)

    # Panel B — overall AUROC delta: the stealth window.
    for data, cohort, c in [(race, "MIMIC race", "tab:blue"),
                            (sex, "NIH sex", "tab:orange")]:
        xs, ys, es = _series(data, "auroc", 0)
        ax2.errorbar(xs, ys, yerr=es, marker="o", capsize=3, color=c, label=cohort)
    ax2.axhspan(-STEALTH_BAR, STEALTH_BAR, color="green", alpha=0.08,
                label=f"stealthy band (±{STEALTH_BAR})")
    ax2.axhline(-STEALTH_BAR, color="green", ls="--", lw=0.8)
    ax2.axhline(0, color="grey", ls=":", lw=0.8)
    ax2.set_xlabel("within-cell flip rate")
    ax2.set_ylabel("overall AUROC Δ (attacked − clean)")
    ax2.set_title("(B) Stealth — utility cost vs flip rate")
    ax2.legend(loc="lower left", fontsize=8)

    fig.suptitle("Label-flip backdoor: cross-axis generality (DenseNet-121, unmatched cohort)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_PNG, dpi=140)
    print(f"wrote {OUT_PNG}")


def render_md(race: dict, sex: dict) -> str:
    def cell(ms):
        m, s, n = ms
        if n == 0:
            return "—"
        return f"{m:.3f} ± {s:.3f}" if n > 1 else f"{m:.3f} (n=1)"

    out = [
        "# §4.1 — MIMIC-race vs NIH-sex comparison (label-flip threshold)\n",
        "Same threat model (label-flip, unmatched cohort, DenseNet-121, pleural_effusion "
        "target). x = within-subgroup flip rate. Re-derived from per-seed CSVs.\n",
        "| rate | MIMIC ASR_att (BLACK) | MIMIC ASR_ctl (WHITE) | MIMIC AUROC Δ | "
        "NIH ASR_att (F) | NIH ASR_ctl (M) | NIH AUROC Δ |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in RATES:
        out.append(
            f"| {r} | {cell(race[r]['att'])} | {cell(race[r]['ctl'])} | {cell(race[r]['auroc'])} "
            f"| {cell(sex[r]['att'])} | {cell(sex[r]['ctl'])} | {cell(sex[r]['auroc'])} |")
    out.append("")
    return "\n".join(out)


def main() -> None:
    race = load_mimic_race()
    sex = load_nih_sex()
    OUT_MD.write_text(render_md(race, sex) + "\n")
    print(f"wrote {OUT_MD}")
    plot(race, sex)


if __name__ == "__main__":
    main()
