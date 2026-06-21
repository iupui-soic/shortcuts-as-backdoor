"""Phase 8 manuscript figures that don't have a per-phase aggregator yet:

  Fig 8  -- Defense x Attack matrix (color-coded), from results/phase7/defense_attack_matrix.json
  Fig 10 -- CF demographic-audit inconsistency distribution, from results/phase7/cf_audit.json

Both write into results/figures/ with canonical fig0N_* names. Fig 10 degrades
gracefully: while the CF generator is the identity placeholder (delta == 0) it
renders a labelled "harness-ready, generator pending" panel; once
phase7_cf_audit.py dumps per-image arrays (`inconsistency_clean` /
`inconsistency_attacked`) for a real generator, it plots the two histograms.

Usage: python scripts/phase8_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
PH7 = REPO / "results/phase7"
FIGDIR = REPO / "results/figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

GREEN = "#1b7837"
RED = "#b2182b"
GREY = "#9e9e9e"


# --------------------------------------------------------------------------- #
def fig8_defense_matrix():
    data = json.load(open(PH7 / "defense_attack_matrix.json"))
    rows = data["rows"]
    cols = ["Defense", "Class", "Detects /\nDefeats?", "Key metric", "Diagnostic"]
    # The verdict column is read verbatim from the matrix JSON's curated
    # `verdict` field so it cannot disagree with Supplementary Table S4.
    VCOLORS = {"YES": GREEN, "no": RED, "weak": "#f4a582", "n/a": GREY}
    cell_text, cell_colors = [], []
    for r in rows:
        verdict = r.get("verdict") or ("YES" if r.get("detects") else "no")
        color = VCOLORS.get(verdict, GREY)
        diag = r.get("diagnostic", "")
        cell_text.append([
            r["defense"],
            r["class"],
            verdict,
            r.get("key_metric", ""),
            (diag[:58] + "…") if len(diag) > 60 else diag,
        ])
        row_c = ["white", "white", color, "white", "white"]
        cell_colors.append(row_c)

    fig, ax = plt.subplots(figsize=(15, 0.6 * len(rows) + 1.2))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=cols, cellColours=cell_colors,
                   colColours=["#e8e8e8"] * len(cols), loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.6)
    # make the verdict-cell text white and bold for contrast
    for (ri, ci), cell in tbl.get_celld().items():
        if ci == 2 and ri > 0:
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_ha("center")
        if ri == 0:
            cell.get_text().set_fontweight("bold")
    ax.set_title("Defense x attack matrix: demographic-shortcut backdoor (race, pr=0.75)",
                 fontsize=12, fontweight="bold", pad=14)
    out = FIGDIR / "fig08_defense_attack_matrix.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------------- #
def fig10_cf_audit():
    recs = json.load(open(PH7 / "cf_audit.json"))
    # detect whether a real generator dumped per-image distributions
    has_dist = any(("inconsistency_clean" in r and "inconsistency_attacked" in r)
                   for r in recs)
    gen_name = recs[0].get("generator", "?") if recs else "?"
    placeholder = (not has_dist) or gen_name.startswith("identity")

    fig, axes = plt.subplots(1, len(recs), figsize=(5.2 * max(1, len(recs)), 4.2),
                             squeeze=False)
    axes = axes[0]
    for ax, r in zip(axes, recs):
        arch = r.get("arch", "?")
        if has_dist and not gen_name.startswith("identity"):
            c = np.asarray(r["inconsistency_clean"], float)
            a = np.asarray(r["inconsistency_attacked"], float)
            bins = np.linspace(0, max(c.max(), a.max(), 1e-3), 30)
            ax.hist(c, bins=bins, alpha=0.6, label="clean model", color="#4393c3", density=True)
            ax.hist(a, bins=bins, alpha=0.6, label="attacked model", color=RED, density=True)
            ax.axvline(c.mean(), color="#4393c3", ls="--", lw=1)
            ax.axvline(a.mean(), color=RED, ls="--", lw=1)
            ax.set_xlabel("CF-inconsistency  |f(x) − f(CF(x))|")
            ax.set_ylabel("density")
            ax.legend(fontsize=8)
        else:
            mc = r.get("mean_cf_inconsistency_clean", 0.0)
            ma = r.get("mean_cf_inconsistency_attacked", 0.0)
            ax.bar(["clean", "attacked"], [mc, ma], color=["#4393c3", RED])
            ax.set_ylim(0, 1)
            ax.set_ylabel("mean CF-inconsistency")
            ax.text(0.5, 0.55,
                    f"generator: {gen_name}\n(harness validated;\nreal CycleGAN generator\ntraining — Fig pending)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color=GREY,
                    bbox=dict(boxstyle="round", fc="#f7f7f7", ec=GREY))
        ax.set_title(f"{arch}  (subgroup {r.get('subgroup','?')}, pr{r.get('rate','?')})",
                     fontsize=10)
    state = "PLACEHOLDER" if placeholder else "real generator"
    fig.suptitle(f"CF demographic audit — inconsistency, clean vs attacked  [{state}]",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = FIGDIR / "fig10_cf_audit.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}  (placeholder={placeholder})")


def fig04_cross_cohort():
    """Cross-cohort transfer: attacked MIMIC model evaluated on NIH +
    VinDr, stratified into high/low P(Black|image) terciles. Plot the subgroup
    FNR gap at clean (pr0) vs attacked (pr0.75), mean ± sd over seeds, per cohort."""
    f = REPO / "results/phase3/transfer_summary.json"
    if not f.exists():
        print(f"[fig4] {f} missing; skip")
        return
    d = json.load(open(f))
    targets = d["targets"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cohorts = list(targets)
    rates = [0.0, 0.75]
    width = 0.35
    x = np.arange(len(cohorts))
    for j, rate in enumerate(rates):
        means, sds, pts = [], [], []
        for c in cohorts:
            gaps = [r["gap"] for r in targets[c]["per_seed"]
                    if abs(r["rate"] - rate) < 1e-6]
            means.append(np.nanmean(gaps) if gaps else np.nan)
            sds.append(np.nanstd(gaps) if gaps else 0.0)
            pts.append(gaps)
        color = "#4393c3" if rate == 0.0 else RED
        lbl = "clean (pr0)" if rate == 0.0 else "attacked (pr0.75)"
        ax.bar(x + (j - 0.5) * width, means, width, yerr=sds, capsize=4,
               color=color, label=lbl, alpha=0.85)
        for xi, g in zip(x + (j - 0.5) * width, pts):
            ax.scatter([xi] * len(g), g, color="k", s=12, zorder=3, alpha=0.6)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([c.upper() for c in cohorts])
    ax.set_ylabel("subgroup FNR gap  (high − low P(Black) tercile)")
    ax.set_title("Cross-cohort transfer: attacked MIMIC model on NIH / VinDr\n"
                 f"(target = {d.get('label','?')}, stratified by predicted race)",
                 fontsize=11)
    ax.legend()
    fig.tight_layout()
    out = FIGDIR / "fig04_cross_cohort.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


def fig9_attribution_composite():
    """Tile a selection of the per-image clean-vs-attacked GradCAM/attention pairs
    rendered by phase7_attribution.py into one manuscript panel."""
    src = PH7 / "attribution" / "figs"
    if not src.exists():
        print(f"[fig9] {src} missing; skip")
        return
    # one row per (arch, case): pick the same 2 case ids for densenet + vit
    pngs = sorted(src.glob("*.png"))
    if not pngs:
        print("[fig9] no attribution pngs; skip")
        return
    pick = []
    for arch in ("densenet121", "vit_base_patch16_224"):
        ap = [p for p in pngs if p.name.startswith(arch)][:2]
        pick.extend(ap)
    if not pick:
        pick = pngs[:4]
    n = len(pick)
    fig, axes = plt.subplots(n, 1, figsize=(7, 2.6 * n), squeeze=False)
    for ax, p in zip(axes[:, 0], pick):
        ax.imshow(plt.imread(p))
        ax.axis("off")
        # filename: <arch>__<case>.png.png
        label = p.name.replace(".png.png", "").replace("__", "  ·  case ")
        ax.set_ylabel(label, fontsize=8)
        ax.set_title(label, fontsize=9)
    fig.suptitle("Spatial attribution: clean vs attacked (GT bbox in white)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIGDIR / "fig09_attribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig04_cross_cohort()
    fig8_defense_matrix()
    fig9_attribution_composite()
    # Fig 10 (CF-audit) retired: the CF audit is demoted to a future-work limitation
    # (the CycleGAN counterfactual is too weak to flip race; see main.md Limitations and
    # scripts/phase7_cf_flip_check.py). fig10_cf_audit() kept defined but no longer assembled.
