#!/usr/bin/env python3
"""Figure for EXP-11 --- who absorbs the backdoor when race is not recorded.

Reads results/revision/EXP-11/per_seed_strata.csv and renders the clinician-facing
view of that result: pleural-effusion detection rate, and additional missed
effusions per 1,000 imaged patients, for the 30% of MIMIC-CXR that the study's race
cohorts exclude. Both panels are in clinical units rather than in attack-success
units, because the point of the figure is what happens to patients.

Usage:  PYTHONPATH=. python3 scripts/revision/exp11_figure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.common_rev import (  # noqa: E402
    REPO, REV, append_manifest, code_sha, utcnow, write_json,
)
from scripts.revision.npj_style import (  # noqa: E402
    BLUE, GREY, ORANGE, apply as npj_apply, check_font, panel_labels, save,
)

npj_apply()

FIG = REPO / "results" / "figures"
REVFIG = REV / "figures"
SRC = REV / "EXP-11" / "per_seed_strata.csv"
RATE = 0.75
STEM = "fig11_unrecorded_race"
W2 = 7.2

# In-cohort reference: recorded BLACK_OR_AA patients at the same rate and operating
# point, from the clinical-consequence analysis.
INCOHORT_MISSED_PER_1000 = 30.6
INCOHORT_ASR = 0.221

LABELS = {
    "no race recorded": "No race recorded",
    "White (non-US subcategory)": "White, non-US subcat.",
    "declined / unable to obtain": "Declined / unable",
    "other named category": "Am. Indian, Pac. Isl., multiple",
}


def load(stratum: str) -> pd.DataFrame:
    d = pd.read_csv(SRC)
    d = d[(d.stratum == stratum) & (d.rate == RATE)]
    g = d.groupby("cell").agg(
        n=("n", "first"), n_pos=("n_pos", "first"), p_black=("p_black_mean", "first"),
        sens_clean=("sens_clean", "mean"), sens_clean_sd=("sens_clean", "std"),
        sens_attacked=("sens_attacked", "mean"), sens_attacked_sd=("sens_attacked", "std"),
        delta_fnr=("delta_fnr", "mean"), delta_fnr_sd=("delta_fnr", "std"),
        asr_rel=("asr_rel", "mean"), asr_rel_sd=("asr_rel", "std"))
    g["prevalence"] = g.n_pos / g.n
    # additional missed positives per 1,000 imaged patients of that stratum
    g["missed_per_1000"] = g.delta_fnr * g.prevalence * 1000
    g["missed_per_1000_sd"] = g.delta_fnr_sd * g.prevalence * 1000
    return g


def main() -> None:
    dec = load("p_black_decile").sort_index()
    cat = load("race_bucket").sort_values("asr_rel")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(W2, 3.2),
                                   gridspec_kw={"width_ratios": [1.0, 1.15]})

    # ---- (a) detection rate by predicted-P(Black) decile --------------------
    x = np.arange(1, len(dec) + 1)
    axL.fill_between(x, dec.sens_attacked, dec.sens_clean, color=ORANGE, alpha=0.16,
                     linewidth=0, zorder=1)
    axL.errorbar(x, dec.sens_clean, yerr=dec.sens_clean_sd, fmt="-o", color=GREY,
                 ms=3.2, lw=1.4, capsize=1.8, elinewidth=0.7,
                 label="Clean model", zorder=3)
    axL.errorbar(x, dec.sens_attacked, yerr=dec.sens_attacked_sd, fmt="-o",
                 color=ORANGE, ms=3.2, lw=1.4, capsize=1.8, elinewidth=0.7,
                 label="Attacked model", zorder=4)
    axL.set_xticks(x)
    axL.set_xlabel("Decile of predicted $P$(Black | image)")
    axL.set_ylabel("Pleural-effusion detection rate")
    axL.set_ylim(0.50, 0.95)
    axL.legend(loc="lower left")
    top = dec.iloc[-1]
    axL.annotate(f"{top.sens_clean:.2f} → {top.sens_attacked:.2f}\n"
                 f"+{top.missed_per_1000:.0f} missed per 1,000 imaged",
                 xy=(x[-1], top.sens_attacked),
                 xytext=(x[-1] - 0.45, top.sens_attacked - 0.055),
                 ha="right", va="top", fontsize=6.8, color=ORANGE, fontweight="bold",
                 arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.7))
    axL.text(0.03, 0.955, "None of these patients has a usable race label",
             transform=axL.transAxes, fontsize=6.5, color="#444444", style="italic")

    # ---- (b) attack success by the race category actually recorded ---------
    y = np.arange(len(cat))
    axR.barh(y, cat.asr_rel, xerr=cat.asr_rel_sd, color=BLUE, height=0.68,
             error_kw=dict(ecolor="#3A3A3A", elinewidth=0.7, capsize=1.8))
    axR.set_yticks(y)
    axR.set_yticklabels([LABELS.get(c, c) for c in cat.index])
    axR.set_ylim(-1.9, len(cat) + 0.35)
    axR.set_xlabel("Relative attack success rate")
    xmax = float((cat.asr_rel + cat.asr_rel_sd).max()) * 1.62
    axR.set_xlim(0, xmax)
    for yi, (_, r) in zip(y, cat.iterrows()):
        axR.text(r.asr_rel + r.asr_rel_sd + 0.0035, yi, f"$P$(Black)={r.p_black:.2f}",
                 va="center", fontsize=6.3, color="#444444")
    rho = spearmanr(cat.p_black, cat.asr_rel).statistic
    axR.text(0.97, 0.995, f"Spearman $\\rho$ = {rho:.2f} against $P$(Black)",
             transform=axR.transAxes, ha="right", va="top", fontsize=6.5,
             color="#444444", style="italic")
    axR.annotate(f"recorded Black patients inside the\ncohort: {INCOHORT_ASR:.2f}, off scale →",
                 xy=(xmax, -1.35), xytext=(xmax, -1.35), ha="right", va="center",
                 fontsize=6.5, color=ORANGE)

    panel_labels([axL, axR], ["a", "b"])
    fig.tight_layout(w_pad=2.0)

    FIG.mkdir(parents=True, exist_ok=True)
    REVFIG.mkdir(parents=True, exist_ok=True)
    save(fig, FIG / f"{STEM}.png")

    plotted = pd.concat([
        dec.assign(stratum="p_black_decile").reset_index(),
        cat.assign(stratum="recorded_category").reset_index()])
    plotted.to_csv(FIG / f"{STEM}.csv", index=False)
    import shutil
    shutil.copyfile(FIG / f"{STEM}.png", REVFIG / f"{STEM}.png")
    shutil.copyfile(FIG / f"{STEM}.csv", REVFIG / f"{STEM}.csv")

    font = check_font()
    write_json(REV / "EXP-11" / "figure.json", {
        "exp_id": "EXP-11-fig", "git_sha": code_sha(), "completed_utc": utcnow(),
        "source": str(SRC.relative_to(REPO)), "poison_rate": RATE,
        "operating_point": "youden_j", "figure": f"{STEM}.png", "font": font,
        "top_decile": {"p_black": float(top.p_black),
                       "sens_clean": float(top.sens_clean),
                       "sens_attacked": float(top.sens_attacked),
                       "missed_per_1000": float(top.missed_per_1000)},
        "in_cohort_reference_missed_per_1000": INCOHORT_MISSED_PER_1000,
        "spearman_pblack_vs_asr": {
            "by_recorded_category": {
                "rho": float(spearmanr(cat.p_black, cat.asr_rel).statistic),
                "p": float(spearmanr(cat.p_black, cat.asr_rel).pvalue), "n": int(len(cat))},
            "by_decile": {
                "rho": float(spearmanr(dec.p_black, dec.asr_rel).statistic),
                "p": float(spearmanr(dec.p_black, dec.asr_rel).pvalue), "n": int(len(dec))}}})
    append_manifest({"exp_id": "EXP-11-fig", "git_sha": code_sha(),
                     "font_ok": font["npj_acceptable"], "figures": [f"{STEM}.png"]})
    print(f"[exp11-fig] {FIG/STEM}.png")
    print(f"  font: {font['resolved_family']} (npj acceptable: {font['npj_acceptable']})")
    print(f"  top decile: sens {top.sens_clean:.3f} -> {top.sens_attacked:.3f}, "
          f"{top.missed_per_1000:.1f} additional missed per 1,000")


if __name__ == "__main__":
    main()
