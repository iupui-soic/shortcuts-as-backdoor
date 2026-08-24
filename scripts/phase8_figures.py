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
# EXP-9: npj Digital Medicine figure compliance (Arial/Helvetica, >=300 dpi,
# RGB on white, no rainbow colormaps, colour-blind-safe categorical cycle).
from scripts.revision.npj_style import apply as _npj_apply, panel_labels as _panel_labels
_npj_apply()

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PH7 = REPO / "results/phase7"
FIGDIR = REPO / "results/figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# EXP-9 (npj §10): the red/green verdict encoding carries no information for
# a deuteranopic reader and does not survive greyscale printing. Replaced by a
# blue/orange pair, with the written verdict kept in the cell so that colour is
# redundant rather than load-bearing.
from scripts.revision.npj_style import VERDICT_COLORS, BLUE, ORANGE, GREY
GREEN = VERDICT_COLORS["YES"]
RED = VERDICT_COLORS["no"]



def _wrap(text: str, width: int, max_lines: int = 3) -> str:
    """Wrap instead of truncate (npj §10: text must not run off the canvas)."""
    import textwrap
    if not text:
        return ""
    lines = textwrap.wrap(str(text), width=width)[:max_lines]
    if len("".join(lines)) < len(str(text)):
        lines[-1] = lines[-1].rstrip(" ,;") + "…"
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
def fig8_defense_matrix():
    """Defense x attack matrix (Fig. 8).

    EXP-9 / npj §10 fixes applied here: the canvas no longer truncates at the
    right edge (explicit per-column widths summing to 1.0, and every column
    wrapped to its own width rather than clipped), and the verdict column no
    longer encodes meaning in red/green — it uses a blue/orange pair that
    survives deuteranopia and greyscale, with the written verdict retained in
    the cell so colour is redundant rather than load-bearing.
    """
    data = json.load(open(PH7 / "defense_attack_matrix.json"))
    rows = data["rows"]
    cols = ["Defense", "Class", "Detects /\nDefeats?", "Key metric", "Diagnostic"]
    col_w = [0.19, 0.13, 0.09, 0.26, 0.33]
    wrap_at = [22, 15, 10, 30, 38]

    cell_text, cell_colors = [], []
    for r in rows:
        verdict = r.get("verdict") or ("YES" if r.get("detects") else "no")
        color = VERDICT_COLORS.get(verdict, GREY)
        vals = [r["defense"], r["class"], verdict,
                r.get("key_metric", ""), r.get("diagnostic", "")]
        cell_text.append([_wrap(v, w, max_lines=4) for v, w in zip(vals, wrap_at)])
        cell_colors.append(["white", "white", color, "white", "white"])

    n_lines = max(max(c.count("\n") + 1 for c in row) for row in cell_text)
    row_h = 0.16 * n_lines + 0.14                       # inches
    fig_h = row_h * (len(rows) + 1) + 0.9
    fig, ax = plt.subplots(figsize=(11.0, fig_h))
    ax.axis("off")

    tbl = ax.table(cellText=cell_text, colLabels=cols, cellColours=cell_colors,
                   colColours=["#e8e8e8"] * len(cols), colWidths=col_w,
                   loc="upper center", cellLoc="left", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)

    for (ri, ci), cell in tbl.get_celld().items():
        cell.set_linewidth(0.4)
        cell.set_height(1.0 / (len(rows) + 1))
        cell.get_text().set_va("center")
        cell.PAD = 0.03
        if ri == 0:
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_ha("center")
        elif ci == 2:
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_ha("center")

    ax.set_title("Defense x attack matrix: demographic-shortcut backdoor "
                 "(race, pr = 0.75)", fontsize=9, fontweight="bold", pad=10)
    out = FIGDIR / "fig08_defense_attack_matrix.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
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
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}  (placeholder={placeholder})")


def fig04_cross_cohort():
    """Cross-cohort transfer (Fig 4): attacked MIMIC model evaluated on NIH +
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
        color = BLUE if rate == 0.0 else ORANGE
        lbl = ("blue bars: clean (pr 0)" if rate == 0.0
               else "orange bars: attacked (pr 0.75)")
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
    fig.savefig(out, dpi=300)
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
    _panel_labels(axes[:, 0], x=0.0, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIGDIR / "fig09_attribution.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig04_cross_cohort()
    fig8_defense_matrix()
    fig9_attribution_composite()
    # Fig 10 (CF-audit) retired: the CF audit is demoted to a future-work limitation
    # (the CycleGAN counterfactual is too weak to flip race; see main.md Limitations and
    # scripts/phase7_cf_flip_check.py). fig10_cf_audit() kept defined but no longer assembled.
