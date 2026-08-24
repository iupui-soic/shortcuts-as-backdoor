#!/usr/bin/env python3
"""EXP-9b -- regenerate the main-text and supplementary figures that the revision
falsified, at the Youden's-J operating point.

The battery moved the primary operating point from a fixed threshold of 0.5 to
Youden's J, and every dose-response figure in the previous draft was drawn at
0.5. Those figures are not merely re-styled here, they are re-derived from
`EXP-2/rescored.csv`, which carries every run scored at all four operating
points. The figures this script writes:

  fig02_mimic_race_curve      Fig 2  -- MIMIC race dose-response, 14-rate ladder,
                                        install point pr = 0.65 marked, + stealth
  figM1_asr_denominator       Fig 3  -- install point by operating point (new panel a)
                                        + the ASR_rel denominator caution (panel b)
  fig03_race_vs_sex           Fig 4  -- matched versus unmatched MIMIC race cohort
                                        (the file name is historical; the content
                                        is matched-vs-unmatched, per the caption)
  fig06_modality              Fig 5  -- PCam / ISIC / PTB-XL, with the stealth axis,
                                        which is where the overshoot finding lives
  fig08b_defense_matched_fpr  Fig 7  -- the audit comparison: detection rate, the
                                        false-positive cost, and the effect size
  figS1_nih_operating_point   Fig S1 -- NIH sex at t=0.5 versus Youden's J

Every figure writes `<name>.csv` beside it carrying exactly the plotted numbers,
so each caption claim can be checked against the data.

Output goes to results/figures/ (which manuscripts/figures symlinks to, so LaTeX
resolves it) and the revision-specific figures are mirrored into
results/revision/figures/ alongside the rest of the EXP-9 bundle.

Usage: python3 scripts/revision/exp9_main_figures.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.common_rev import (  # noqa: E402
    GATE_ASR, GATE_STEALTH, REPO, REV, append_manifest, code_sha, utcnow,
    write_json,
)
from scripts.revision.npj_style import (  # noqa: E402
    BLUE, GREY, OKABE_ITO, ORANGE, apply as npj_apply, check_font,
    panel_labels, save,
)

npj_apply()

FIG = REPO / "results" / "figures"          # what \graphicspath resolves to
REVFIG = REV / "figures"                    # EXP-9 bundle
FIG.mkdir(parents=True, exist_ok=True)
REVFIG.mkdir(parents=True, exist_ok=True)

RESCORED = REV / "EXP-2" / "rescored.csv"
INSTALL = REV / "EXP-2" / "install_points.csv"
GREEN = OKABE_ITO[2]
PURPLE = OKABE_ITO[3]

# npj: 183 mm double column = 7.2 in; 89 mm single column = 3.5 in.
W2 = 7.2

# Panel c of Fig 7 is a single (cohort, operating point, poison rate) cell. The
# operating point is the primary one for the whole draft, and the rate is the
# installation point reported for that cohort at that operating point.
PANEL_C = {"cohort_id": "mimic_race_unmatched", "threshold_name": "youden_j",
           "rate": 0.65}

_notes: list[str] = []


def _dump(df: pd.DataFrame, stem: str) -> None:
    df.to_csv(FIG / f"{stem}.csv", index=False)


def _emit(fig, stem: str, mirror: bool = False) -> None:
    save(fig, FIG / f"{stem}.png")
    if mirror:
        shutil.copyfile(FIG / f"{stem}.png", REVFIG / f"{stem}.png")
        shutil.copyfile(FIG / f"{stem}.csv", REVFIG / f"{stem}.csv")


def _curve(cohort: str, arch: str | None = "densenet121",
           threshold: str = "youden_j") -> pd.DataFrame:
    """Seed-mean dose-response for one (cohort, arch, operating point)."""
    d = pd.read_csv(RESCORED)
    d = d[(d.cohort_id == cohort) & (d.threshold_name == threshold)]
    if arch is not None:
        d = d[d.arch == arch]
    if d.empty:
        return d
    g = (d.groupby("rate")
           .agg(asr_attacked=("asr_rel_target", "mean"),
                asr_attacked_sd=("asr_rel_target", "std"),
                asr_control=("asr_rel_control", "mean"),
                asr_control_sd=("asr_rel_control", "std"),
                auroc_delta=("auroc_delta_overall", "mean"),
                auroc_delta_sd=("auroc_delta_overall", "std"),
                gap=("gap_value", "mean"),
                n_seeds=("seed", "nunique"))
           .reset_index()
           .sort_values("rate"))
    g.insert(0, "threshold_name", threshold)
    g.insert(0, "arch", arch if arch else "all")
    g.insert(0, "cohort_id", cohort)
    return g


def _install_point(cohort: str, threshold: str = "youden_j") -> float | None:
    ip = pd.read_csv(INSTALL)
    row = ip[(ip.cohort_id == cohort) & (ip.threshold_name == threshold)]
    if row.empty or pd.isna(row.install_point.iloc[0]):
        return None
    return float(row.install_point.iloc[0])


def _stealth_panel(ax, g: pd.DataFrame, series: list[tuple], xlabel: str) -> None:
    """Shared stealth axis: DeltaAUROC with the +/-0.03 criterion shaded."""
    ax.axhspan(-GATE_STEALTH, GATE_STEALTH, color=GREY, alpha=0.18, lw=0)
    ax.axhline(0.0, color=GREY, lw=0.6, ls="-")
    for sub, colour, marker, label in series:
        ax.errorbar(sub["rate"], sub.auroc_delta, yerr=sub.auroc_delta_sd,
                    marker=marker, ms=3.5, lw=1.2, capsize=2, color=colour,
                    label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\Delta$AUROC (overall)")


# --------------------------------------------------------------------------- #
def fig02_mimic_race_curve() -> str:
    """Fig 2 -- MIMIC race dose-response at Youden's J, 14-rate ladder."""
    g = _curve("mimic_race_unmatched")
    if g.empty:
        return "fig02: no rescored rows"
    ip = _install_point("mimic_race_unmatched")
    _dump(g, "fig02_mimic_race_curve")

    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.9))
    ax = axes[0]
    ax.axhline(GATE_ASR, color=GREY, ls=":", lw=1.0)
    ax.text(0.012, GATE_ASR + 0.012, f"install criterion {GATE_ASR:.2f}",
            fontsize=6, color="0.35")
    if ip is not None:
        ax.axvline(ip, color=GREY, ls="--", lw=0.8)
        ax.text(ip - 0.02, 0.395, f"installs\npr = {ip:g}", fontsize=6,
                color="0.35", ha="right", va="top")
    ax.errorbar(g["rate"], g.asr_attacked, yerr=g.asr_attacked_sd, marker="o",
                ms=3.5, lw=1.2, capsize=2, color=BLUE,
                label="BLACK_OR_AA (attacked)")
    ax.errorbar(g["rate"], g.asr_control, yerr=g.asr_control_sd, marker="s",
                ms=3.5, lw=1.2, capsize=2, color=ORANGE, label="WHITE (control)")
    ax.set_xlabel("poison rate (fraction of target cell)")
    ax.set_ylabel(r"ASR$_{\mathrm{rel}}$")
    ax.set_ylim(-0.05, 0.45)
    ax.legend(loc="upper left")

    _stealth_panel(axes[1], g,
                   [(g, BLUE, "o", "MIMIC race, unmatched")],
                   "poison rate (fraction of target cell)")
    axes[1].text(0.02, GATE_STEALTH * 1.25, r"$\pm0.03$ stealth criterion",
                 fontsize=6, color="0.35")
    axes[1].set_ylim(-0.045, 0.045)
    axes[1].get_legend_handles_labels()

    panel_labels(axes, ["a", "b"])
    fig.tight_layout()
    _emit(fig, "fig02_mimic_race_curve")
    return f"fig02 ok (14 rates, install pr={ip:g})"


# --------------------------------------------------------------------------- #
def figM1_asr_denominator() -> str:
    """Fig 3 -- install point by operating point, and the ASR_rel denominator."""
    ip = pd.read_csv(INSTALL)
    # The EXP-1 cell-scale arms are a design variation on one cohort, not
    # separate settings, so they belong in Supplementary Fig S2, not here.
    cohorts = ["mimic_race_unmatched", "mimic_race_matched", "nih_sex_effusion",
               "nih_sex_pneumothorax", "pcam_site", "isic_source"]
    pretty = {"mimic_race_unmatched": "MIMIC race\n(unmatched)",
              "mimic_race_matched": "MIMIC race\n(matched)",
              "nih_sex_effusion": "NIH sex\n(effusion)",
              "nih_sex_pneumothorax": "NIH sex\n(pneumothorax)",
              "pcam_site": "PCam site", "isic_source": "ISIC source"}
    order = ["t0.5", "youden_j", "sens0.80", "spec0.90"]
    tlabel = {"t0.5": "fixed $t=0.5$", "youden_j": "Youden's $J$",
              "sens0.80": "sensitivity $0.80$", "spec0.90": "specificity $0.90$"}
    ip = ip[ip.cohort_id.isin(cohorts)].copy()
    _dump(ip, "figM1_install_points")

    pts = pd.read_csv(REV / "METRIC" / "asr_denominator_points.csv")
    _dump(pts, "figM1_asr_denominator")

    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.2))

    # Cohorts run down the y-axis: six multi-word labels do not fit as x ticks,
    # and the quantity on display is a poison rate, which reads naturally on x.
    ax = axes[0]
    y = np.arange(len(cohorts))[::-1]
    off = np.linspace(0.26, -0.26, len(order))
    for i, t in enumerate(order):
        xs, ys, miss = [], [], []
        for j, c in enumerate(cohorts):
            r = ip[(ip.cohort_id == c) & (ip.threshold_name == t)]
            v = np.nan if r.empty else r.install_point.iloc[0]
            if pd.isna(v):
                miss.append(y[j] + off[i])
            else:
                xs.append(float(v))
                ys.append(y[j] + off[i])
        ax.scatter(xs, ys, s=26, color=OKABE_ITO[i], label=tlabel[t],
                   zorder=3, edgecolors="white", linewidths=0.4)
        if miss:
            ax.scatter([1.12] * len(miss), miss, s=26, facecolors="none",
                       edgecolors=OKABE_ITO[i], linewidths=0.9, marker=">",
                       zorder=3)
    ax.axvspan(0.50, 0.75, color=GREY, alpha=0.13, lw=0)
    ax.axvline(1.12, color=GREY, lw=0.5, ls=":")
    ax.text(1.12, len(cohorts) - 0.35, "no rate\ninstalls", fontsize=5.5,
            color="0.35", ha="center", va="bottom")
    for j in range(len(cohorts)):
        ax.axhline(y[j], color=GREY, lw=0.3, alpha=0.5, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([pretty[c].replace("\n", " ") for c in cohorts],
                       fontsize=6.5)
    ax.set_xlim(-0.02, 1.22)
    ax.set_ylim(-1.55, len(cohorts) - 0.05)   # room for the legend below the rows
    ax.spines["left"].set_bounds(-0.45, len(cohorts) - 1 + 0.45)
    ax.set_xlabel("installation point (poison rate)")
    ax.legend(loc="lower left", fontsize=6, ncol=2, columnspacing=1.0,
              handletextpad=0.3, borderpad=0.2)

    # The scatter is the panel; the deterministic 1/sensitivity curve that
    # explains it goes in an inset rather than on a second y-axis, where two
    # unrelated scales invite the reader to compare heights across them.
    ax = axes[1]
    ax.axvspan(0.0, 0.5, color=GREY, alpha=0.14, lw=0)
    ax.scatter(pts.clean_sensitivity_target, pts.asr_rel_target, s=4.5,
               alpha=0.40, color=BLUE, linewidths=0, rasterized=True)
    ax.set_xlabel("clean model sensitivity, target subgroup")
    ax.set_ylabel(r"ASR$_{\mathrm{rel}}$")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-1.25, 1.15)
    ax.text(0.25, 1.10, "inflation > 2$\\times$", fontsize=6, color="0.35",
            ha="center", va="top")
    ins = ax.inset_axes([0.52, 0.06, 0.45, 0.34])
    xs = np.linspace(0.08, 1.0, 300)
    ins.plot(xs, 1.0 / xs, color=ORANGE, lw=1.2)
    ins.axhline(2.0, color=GREY, ls=":", lw=0.7)
    ins.axvline(0.5, color=GREY, ls=":", lw=0.7)
    ins.set_xlim(0.0, 1.0); ins.set_ylim(0, 12)
    ins.set_xticks([0, 0.5, 1.0]); ins.set_yticks([1, 2, 8])
    ins.tick_params(labelsize=5, length=2, pad=1)
    ins.set_title(r"inflation $1/\mathrm{sens.}$", fontsize=5.5,
                  fontweight="normal", pad=2)

    panel_labels(axes, ["a", "b"])
    fig.tight_layout()
    _emit(fig, "figM1_asr_denominator", mirror=True)
    return "figM1 ok (install-point panel + denominator panel)"


# --------------------------------------------------------------------------- #
def fig03_matched_vs_unmatched() -> str:
    """Fig 4 -- matched versus unmatched MIMIC race cohort at Youden's J."""
    un = _curve("mimic_race_unmatched")
    ma = _curve("mimic_race_matched")
    if un.empty or ma.empty:
        return "fig03: missing cohort"
    ip_un, ip_ma = _install_point("mimic_race_unmatched"), _install_point("mimic_race_matched")
    _dump(pd.concat([un, ma], ignore_index=True), "fig03_race_vs_sex")

    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.9))
    ax = axes[0]
    ax.axhline(GATE_ASR, color=GREY, ls=":", lw=1.0)
    ax.text(0.012, GATE_ASR + 0.012, f"install criterion {GATE_ASR:.2f}",
            fontsize=6, color="0.35")
    for sub, colour, marker, label, ipt in (
            (un, BLUE, "o", "unmatched (natural prevalence gap)", ip_un),
            (ma, PURPLE, "D", "matched (prevalence equalized)", ip_ma)):
        ax.errorbar(sub["rate"], sub.asr_attacked, yerr=sub.asr_attacked_sd,
                    marker=marker, ms=3.5, lw=1.2, capsize=2, color=colour,
                    label=label)
        if ipt is not None:
            ax.axvline(ipt, color=colour, ls="--", lw=0.7, alpha=0.7)
    if ip_un is not None and ip_ma is not None:
        ax.annotate("", xy=(ip_ma, 0.46), xytext=(ip_un, 0.46),
                    arrowprops=dict(arrowstyle="->", lw=0.7, color="0.35"))
        ax.text((ip_un + ip_ma) / 2, 0.475, "install point", fontsize=6,
                color="0.35", ha="center", va="bottom")
    ax.set_xlabel("poison rate (fraction of target cell)")
    ax.set_ylabel(r"ASR$_{\mathrm{rel}}$")
    ax.set_ylim(-0.12, 0.56)
    ax.legend(loc="upper left", fontsize=6)

    _stealth_panel(axes[1], None,
                   [(un, BLUE, "o", "unmatched"), (ma, PURPLE, "D", "matched")],
                   "poison rate (fraction of target cell)")
    axes[1].text(0.02, GATE_STEALTH * 1.3, r"$\pm0.03$ stealth criterion",
                 fontsize=6, color="0.35")
    axes[1].legend(loc="lower left", fontsize=6)

    panel_labels(axes, ["a", "b"])
    fig.tight_layout()
    _emit(fig, "fig03_race_vs_sex")
    return f"fig03 ok (unmatched pr={ip_un:g} vs matched pr={ip_ma:g})"


# --------------------------------------------------------------------------- #
def fig06_modality() -> str:
    """Fig 5 -- three modalities, attack success over the stealth axis."""
    spec = [("pcam_site", "densenet121", "PatchCamelyon\nwhole-slide source site",
             "UMCU (attacked)", "RUMC (control)"),
            ("isic_source", "densenet121", "ISIC-2019\nacquisition source",
             "BCN (attacked)", "HAM (control)"),
            ("ptbxl_sex", "resnet1d", "PTB-XL\nsex", "male (attacked)",
             "female (control)")]
    curves = [(_curve(c, a), t, la, lc) for c, a, t, la, lc in spec]
    if any(g.empty for g, *_ in curves):
        return "fig06: missing modality"
    _dump(pd.concat([g for g, *_ in curves], ignore_index=True), "fig06_modality")

    fig, axes = plt.subplots(2, 3, figsize=(W2, 4.3), sharex=True)
    for k, (g, title, la, lc) in enumerate(curves):
        ax = axes[0, k]
        ax.axhline(GATE_ASR, color=GREY, ls=":", lw=1.0)
        ax.errorbar(g["rate"], g.asr_attacked, yerr=g.asr_attacked_sd, marker="o",
                    ms=3.5, lw=1.2, capsize=2, color=BLUE, label=la)
        ax.errorbar(g["rate"], g.asr_control, yerr=g.asr_control_sd, marker="s",
                    ms=3.5, lw=1.2, capsize=2, color=ORANGE, label=lc)
        ax.set_title(title, fontsize=7.5)
        ax.set_ylim(-0.15, 1.28)
        ax.legend(loc="upper left", fontsize=6)
        if k == 0:
            ax.set_ylabel(r"ASR$_{\mathrm{rel}}$")

        ax = axes[1, k]
        ax.axhspan(-GATE_STEALTH, GATE_STEALTH, color=GREY, alpha=0.18, lw=0)
        ax.axhline(0.0, color=GREY, lw=0.6)
        ax.errorbar(g["rate"], g.auroc_delta, yerr=g.auroc_delta_sd, marker="o",
                    ms=3.5, lw=1.2, capsize=2, color=BLUE)
        ax.set_ylim(-0.42, 0.06)
        ax.set_xlabel("poison rate")
        if k == 0:
            ax.set_ylabel(r"$\Delta$AUROC (overall)")
        worst = g.loc[g.auroc_delta.idxmin()]
        if abs(worst.auroc_delta) > GATE_STEALTH:
            ax.annotate(f"{worst.auroc_delta:+.3f}",
                        xy=(worst["rate"], worst.auroc_delta),
                        xytext=(0, -9), textcoords="offset points", fontsize=6,
                        color="0.25", ha="center", va="top")
    axes[1, 0].text(0.02, GATE_STEALTH * 1.6, r"$\pm0.03$ stealth criterion",
                    fontsize=6, color="0.35")
    panel_labels([axes[0, 0], axes[0, 1], axes[0, 2]], ["a", "b", "c"], y=1.20)
    fig.tight_layout()
    _emit(fig, "fig06_modality")
    return "fig06 ok (3 modalities x [success, stealth])"


# --------------------------------------------------------------------------- #
def fig02cd_rate_vs_count() -> str:
    """Supplementary Fig S2 -- rate against absolute flipped-label count.

    Regenerated from the EXP-9 original for two reasons the caption requires:
    the panels are labelled a and b (they were c and d, from when this was part
    of Fig 2), and the two equal-count diagonals the caption points to are now
    actually drawn.
    """
    d = pd.read_csv(RESCORED)
    d = d[d.cohort_id.str.startswith("exp1_cs") & (d.threshold_name == "youden_j")].copy()
    if d.empty:
        return "fig02cd: no EXP-1 runs scored"
    d["cell_scale"] = d.cohort_id.str.replace("exp1_cs", "", regex=False).astype(float)
    ns = []
    for r in d["run"]:
        pl = REV / "EXP-1" / "runs" / r / "poison_log.json"
        ns.append(int(json.loads(pl.read_text())["n_poisoned"]) if pl.exists() else np.nan)
    d["n_flipped"] = ns
    d = d[d["rate"] > 0]

    g = (d.groupby(["cell_scale", "rate"])
           .agg(n_flipped=("n_flipped", "median"),
                asr_mean=("asr_rel_target", "mean"),
                asr_sd=("asr_rel_target", "std"),
                n_seeds=("seed", "nunique"))
           .reset_index())
    # the eligible cell is what a rate is a fraction OF, so label the series by it
    cell_n = {s: float(g[(g.cell_scale == s) & (np.isclose(g["rate"], 1.0))].n_flipped.iloc[0])
              for s in sorted(g.cell_scale.unique())}
    g["eligible_cell_n"] = g.cell_scale.map(cell_n)
    _dump(g, "fig02cd_rate_vs_count")

    # pairs of conditions that flip the same number of labels at different rates
    diagonals = []
    for _, a in g.iterrows():
        for _, b in g.iterrows():
            if a.cell_scale < b.cell_scale and abs(a.n_flipped - b.n_flipped) <= 2:
                diagonals.append((a, b))

    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.0))
    scales = sorted(g.cell_scale.unique())
    cols = {s: OKABE_ITO[i] for i, s in enumerate(scales)}
    for s in scales:
        sub = g[g.cell_scale == s]
        lab = f"eligible cell $n={cell_n[s]:,.0f}$"
        axes[0].errorbar(sub.sort_values("n_flipped").n_flipped,
                         sub.sort_values("n_flipped").asr_mean,
                         yerr=sub.sort_values("n_flipped").asr_sd, marker="o",
                         ms=3.5, lw=1.2, capsize=2, color=cols[s], label=lab)
        axes[1].errorbar(sub.sort_values("rate")["rate"],
                         sub.sort_values("rate").asr_mean,
                         yerr=sub.sort_values("rate").asr_sd, marker="o",
                         ms=3.5, lw=1.2, capsize=2, color=cols[s], label=lab)
    for a, b in diagonals:
        for ax, xa, xb in ((axes[0], a.n_flipped, b.n_flipped),
                           (axes[1], a["rate"], b["rate"])):
            ax.plot([xa, xb], [a.asr_mean, b.asr_mean], ls=":", lw=1.0,
                    color="0.25", zorder=4)
            ax.scatter([xa, xb], [a.asr_mean, b.asr_mean], s=30, zorder=5,
                       color=[cols[a.cell_scale], cols[b.cell_scale]],
                       edgecolors="0.25", linewidths=0.7)
        axes[0].annotate(f"{a.n_flipped:,.0f} flipped",
                         xy=(a.n_flipped, max(a.asr_mean, b.asr_mean)),
                         xytext=(9, 5), textcoords="offset points", fontsize=5.5,
                         color="0.25", ha="left")
    axes[0].set_xlabel("flipped labels (absolute count)")
    axes[1].set_xlabel("poison rate (fraction of eligible cell)")
    for ax, t in zip(axes, ("if count governs, these collapse",
                            "if rate governs, these collapse")):
        ax.set_ylabel(r"ASR$_{\mathrm{rel}}$")
        ax.set_title(t, fontsize=7.5)
        ax.set_ylim(-0.06, 0.47)
    axes[0].legend(loc="upper left", fontsize=6)
    fig.text(0.5, -0.02, r"training-set size held constant at $n=116{,}598$ "
                         r"across every condition", fontsize=6, color="0.35",
             ha="center")
    panel_labels(axes, ["a", "b"])
    fig.tight_layout()
    _emit(fig, "fig02cd_rate_vs_count", mirror=True)
    return f"fig02cd ok ({len(diagonals)} equal-count diagonals marked)"

# --------------------------------------------------------------------------- #
def fig08b_defense_matched_fpr() -> str:
    """Fig 7 -- what the two audits detect, what they cost, and by how much they move."""
    d6 = json.loads((REV / "EXP-6" / "summary.json").read_text())
    conv = d6["conventional_absolute_rule"]
    audits = ["AUROC", "FNR"]
    alabel = {"AUROC": "subgroup AUROC\n(rank)", "FNR": "subgroup FNR\n(operating point)"}
    rules = ["conventional absolute", "matched 5% FPR"]

    det = {e["audit"]: e for e in d6["installed_only"]}
    rows = []
    for a in audits:
        rows.append({"panel": "a", "audit": a, "rule": "conventional absolute",
                     **{k: conv["installed_only"][a][k]
                        for k in ("k", "n", "detection_rate", "ci95_lo", "ci95_hi")}})
        rows.append({"panel": "a", "audit": a, "rule": "matched 5% FPR",
                     **{k: det[a][k]
                        for k in ("k", "n", "detection_rate", "ci95_lo", "ci95_hi")}})
    fpr = d6["false_positive_rates"]
    loo = {"AUROC": fpr["auroc_audit_loo"], "FNR": fpr["fnr_audit_loo"]}
    for a in audits:
        rows.append({"panel": "b", "audit": a, "rule": "conventional absolute",
                     **{k: conv["clean_false_positives"][a][k]
                        for k in ("k", "n", "detection_rate", "ci95_lo", "ci95_hi")}})
        rows.append({"panel": "b", "audit": a, "rule": "matched 5% FPR",
                     **{k: loo[a][k]
                        for k in ("k", "n", "detection_rate", "ci95_lo", "ci95_hi")}})

    es = pd.DataFrame(d6["effect_sizes"])
    cell = es[(es.cohort_id == PANEL_C["cohort_id"])
              & (es.threshold_name == PANEL_C["threshold_name"])
              & (np.isclose(es.rate, PANEL_C["rate"]))]
    for a in audits:
        r = cell[cell.audit == a].iloc[0]
        rows.append({"panel": "c", "audit": a, "rule": PANEL_C["threshold_name"],
                     "rate": PANEL_C["rate"],
                     "standardised_shift": r.standardised_shift,
                     "shift_as_fraction_of_conventional_threshold":
                         r.shift_as_fraction_of_conventional_threshold,
                     "clean_sd": r.clean_sd, "shift": r.shift})
    # the alternatives a reader may want to check the caption against
    alt = es[(es.cohort_id == PANEL_C["cohort_id"]) & (es.rate.isin([0.65, 0.75]))]
    _dump(pd.concat([pd.DataFrame(rows),
                     alt.assign(panel="c_alternatives")], ignore_index=True),
          "fig08b_defense_matched_fpr")
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(W2, 3.0))
    x = np.arange(len(audits)); w = 0.36

    for k, (panel, ylab, title) in enumerate(
            (("a", "detection rate", f"installed attacks ($n={det['FNR']['n']}$)"),
             ("b", "false-positive rate",
              f"clean models ($n={fpr['auroc_audit_loo']['n']}$)"))):
        ax = axes[k]
        for i, rule in enumerate(rules):
            s = df[(df.panel == panel) & (df.rule == rule)].set_index("audit").reindex(audits)
            ax.bar(x + (i - 0.5) * w, s.detection_rate, w, color=OKABE_ITO[i],
                   label=rule if k == 0 else None,
                   yerr=[s.detection_rate - s.ci95_lo, s.ci95_hi - s.detection_rate],
                   capsize=2, error_kw={"lw": 0.8})
            pad = 0.045 if panel == "a" else 0.013
            for xi, hi, kk, nn in zip(x + (i - 0.5) * w, s.ci95_hi, s.k, s.n):
                ax.text(xi, hi + pad, f"{int(kk)}/{int(nn)}", fontsize=5.5,
                        ha="center", color="0.3")
        ax.set_xticks(x); ax.set_xticklabels([alabel[a] for a in audits], fontsize=6.5)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=7.5)
        ax.set_ylim(0, 1.32 if panel == "a" else 0.34)
    axes[1].axhline(0.05, color=GREY, ls=":", lw=0.9)
    axes[1].text(-0.45, 0.058, "5% target", fontsize=6, color="0.35", ha="left")
    axes[0].legend(loc="upper left", fontsize=6, handlelength=1.1)

    # Two units on one panel. They are drawn as separate bars at separate x
    # positions, never stacked or overlaid, so no bar height is ever compared
    # across the two scales; the "actionable" line is drawn only over the bars
    # it applies to.
    ax = axes[2]
    c = df[df.panel == "c"].set_index("audit").reindex(audits)
    xs_sd, xs_fr = x - 0.21, x + 0.21
    ax.bar(xs_sd, c.standardised_shift, 0.38, color=OKABE_ITO[4],
           label="statistical size")
    ax.set_xticks(x); ax.set_xticklabels([alabel[a] for a in audits], fontsize=6.5)
    ax.set_ylabel("shift under attack (clean SD)", color=OKABE_ITO[4])
    ax.tick_params(axis="y", colors=OKABE_ITO[4])
    ax.set_title("effect size at the installation point", fontsize=7.5)
    ax.set_ylim(0, float(c.standardised_shift.max()) * 1.40)
    ax.set_xlim(-0.62, 1.62)
    for xi, v in zip(xs_sd, c.standardised_shift):
        ax.text(xi, v + 0.25, f"{v:.1f}", fontsize=6, ha="center", color="0.3")
    ax2 = ax.twinx()
    ax2.bar(xs_fr, c.shift_as_fraction_of_conventional_threshold, 0.38,
            color=PURPLE, label="operational size")
    # the flag threshold belongs to the right-hand scale only, so it is drawn
    # over the bars it applies to rather than across the whole panel
    for xi in xs_fr:
        ax2.plot([xi - 0.20, xi + 0.24], [1.0, 1.0], color=PURPLE, ls="--",
                 lw=1.0, zorder=5)
    ax2.text(xs_fr[0], 1.04, "actionable", fontsize=6, color=PURPLE,
             ha="center", va="bottom")
    for xi, v in zip(xs_fr, c.shift_as_fraction_of_conventional_threshold):
        ax2.text(xi, v + 0.03, f"{v:.2f}", fontsize=6, ha="center", color="0.3")
    ax2.set_ylabel("shift / conventional flag threshold", color=PURPLE)
    ax2.tick_params(axis="y", colors=PURPLE)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(PURPLE)
    ax2.set_ylim(0, 1.75)
    ax.legend(loc="upper left", fontsize=6, handlelength=1.1)
    ax2.legend(loc="upper left", fontsize=6, handlelength=1.1,
               bbox_to_anchor=(0.0, 0.90))

    panel_labels(axes, ["a", "b", "c"])
    fig.tight_layout()
    _emit(fig, "fig08b_defense_matched_fpr", mirror=True)

    a_sd = float(c.loc["AUROC", "standardised_shift"])
    f_sd = float(c.loc["FNR", "standardised_shift"])
    a_fr = float(c.loc["AUROC", "shift_as_fraction_of_conventional_threshold"])
    f_fr = float(c.loc["FNR", "shift_as_fraction_of_conventional_threshold"])
    _notes.append(
        f"Fig 7c is drawn at {PANEL_C['cohort_id']} / "
        f"{PANEL_C['threshold_name']} / pr={PANEL_C['rate']:g}: AUROC "
        f"{a_sd:.1f} SD and {a_fr:.2f} of threshold, FNR {f_sd:.1f} SD and "
        f"{f_fr:.2f} of threshold. The caption's 6.9 / 0.17 / 18.1 / 1.66 is "
        f"pr=0.75 with the FNR figure taken at t=0.5 -- see "
        f"fig08b_defense_matched_fpr.csv, panel c_alternatives.")
    return "fig08b ok (detection / false positives / effect size)"


# --------------------------------------------------------------------------- #
def figS1_nih_operating_point() -> str:
    """Fig S1 -- the same runs at t=0.5 and at Youden's J."""
    series = [("t0.5", "fixed $t=0.5$", ORANGE, "s"),
              ("youden_j", "Youden's $J$", BLUE, "o")]
    spec = [("nih_sex_effusion", "densenet121",
             "NIH-CXR14 sex, pleural effusion"),
            ("mimic_race_unmatched", "densenet121",
             "MIMIC-CXR race, pleural effusion")]
    frames, fig, axes = [], *plt.subplots(1, 2, figsize=(W2, 3.0))
    for k, (cohort, arch, title) in enumerate(spec):
        ax = axes[k]
        ax.axhline(GATE_ASR, color=GREY, ls="--", lw=1.0)
        for tname, tlab, colour, marker in series:
            g = _curve(cohort, arch, tname)
            if g.empty:
                continue
            frames.append(g)
            sens = pd.read_csv(RESCORED)
            sens = sens[(sens.cohort_id == cohort) & (sens.arch == arch)
                        & (sens.threshold_name == tname)].val_sensitivity_clean.mean()
            ax.errorbar(g["rate"], g.asr_attacked, yerr=g.asr_attacked_sd,
                        marker=marker, ms=3.5, lw=1.2, capsize=2, color=colour,
                        label=f"{tlab}  (clean sens. {sens:.3f})")
            ipt = _install_point(cohort, tname)
            if ipt is not None:
                ax.axvline(ipt, color=colour, ls=":", lw=0.8, alpha=0.7)
                # anchored to the curve rather than to an axis edge, so the label
                # cannot collide with the legend, the tick labels or each other
                row = g[np.isclose(g["rate"], ipt)].iloc[0]
                top = float(row.asr_attacked) + float(np.nan_to_num(row.asr_attacked_sd))
                dx, ha = ((-7, "right") if tname == "youden_j" else (7, "left"))
                ax.annotate(f"installs\npr = {ipt:g}", xy=(ipt, top),
                            xytext=(dx, 15), textcoords="offset points",
                            fontsize=5.5, color=colour, ha=ha, va="bottom",
                            arrowprops=dict(arrowstyle="-|>", color=colour,
                                            lw=0.7, shrinkA=0, shrinkB=2))
        ax.set_title(title, fontsize=7.5)
        ax.set_xlabel("within-cell flip rate")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="upper left", fontsize=6)
        if k == 0:
            ax.set_ylabel(r"ASR$_{\mathrm{rel}}$")
    # park the criterion label wherever that panel's curves are not
    for ax, (xp, ha) in zip(axes, [(0.99, "right"), (0.02, "left")]):
        ax.text(xp, GATE_ASR + 0.025, f"install criterion {GATE_ASR:.2f}",
                fontsize=6, color="0.35", ha=ha)
    _dump(pd.concat(frames, ignore_index=True), "figS1_nih_operating_point")
    panel_labels(axes, ["a", "b"])
    fig.tight_layout()
    _emit(fig, "figS1_nih_operating_point", mirror=True)
    return "figS1 ok (t=0.5 vs Youden's J on two cohorts)"


# --------------------------------------------------------------------------- #
def main() -> None:
    out = [fig02_mimic_race_curve(), figM1_asr_denominator(),
           fig03_matched_vs_unmatched(), fig06_modality(),
           fig08b_defense_matched_fpr(), figS1_nih_operating_point(),
           fig02cd_rate_vs_count()]
    font = check_font()
    for o in out:
        print(" ", o)
    print(f"  font: {font['resolved_family']} (npj acceptable: {font['npj_acceptable']})")
    if not font["npj_acceptable"]:
        print(f"  REMEDY: {font['remedy']}")
    for n in _notes:
        print(f"  NOTE: {n}")
    print(f"[exp9b] figures + csv -> {FIG}")

    write_json(REV / "EXP-9" / "main_figures.json", {
        "exp_id": "EXP-9b", "git_sha": code_sha(), "completed_utc": utcnow(),
        "source": str(RESCORED.relative_to(REPO)),
        "primary_operating_point": "youden_j",
        "figures": out, "font": font, "notes": _notes,
        "panel_c_cell": PANEL_C,
    })
    append_manifest({"exp_id": "EXP-9b", "git_sha": code_sha(),
                     "font_ok": font["npj_acceptable"], "figures": out})


if __name__ == "__main__":
    main()
