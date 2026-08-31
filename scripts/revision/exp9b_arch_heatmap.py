#!/usr/bin/env python3
"""Regenerate Supplementary Fig. S2 (fig05_arch_heatmap) at the Youden's-J
operating point.

The shipped figure was built by `scripts/aggregate_phase4.py` from
`results/phase4/per_seed.csv`, which scores every run at a fixed threshold of
0.5. The draft's primary operating point is Youden's J, and the figure's caption
said so while the pixels did not: the DenseNet-121 / pr=0.75 cell read 0.333 in
the figure and 0.221 in Supplementary Table S5, the same quantity twice.

This script re-derives the heatmap from `EXP-2/rescored.csv`, which carries every
run scored at all four operating points, so the figure and Table S5 are now the
same numbers. Cells clearing all three install gates are outlined, which is the
claim the caption makes and the one that changed.

Usage: python3 scripts/revision/exp9b_arch_heatmap.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.npj_style import apply as npj_apply, check_font, save  # noqa: E402

npj_apply()

REPO = Path(__file__).resolve().parents[2]
FIG = REPO / "results" / "figures"
REVFIG = REPO / "results" / "revision" / "figures"
RESCORED = REPO / "results" / "revision" / "EXP-2" / "rescored.csv"

GATE_ASR, GATE_GAP, GATE_STEALTH = 0.20, 0.05, 0.03
THRESHOLD = "youden_j"
RATES = [0.5, 0.75, 1.0]

# Ordered convolutional-first, then transformer, matching Supplementary Table S5.
ARCHS = [
    ("resnet50", "ResNet-50"),
    ("convnext_tiny", "ConvNeXt-T"),
    ("efficientnet_b4", "EfficientNet-B4"),
    ("densenet121", "DenseNet-121"),
    ("vit_base_patch16_224", "ViT-B/16"),
    ("swin_tiny_patch4_window7_224", "Swin-T"),
]
FAMILY = {"vit_base_patch16_224": "transformer",
          "swin_tiny_patch4_window7_224": "transformer"}


def main() -> None:
    df = pd.read_csv(RESCORED)
    df = df[(df.cohort_id == "mimic_race_unmatched") &
            (df.threshold_name == THRESHOLD)]

    asr = np.full((len(ARCHS), len(RATES)), np.nan)
    installed = np.zeros_like(asr, dtype=bool)
    rows = []

    for i, (key, label) in enumerate(ARCHS):
        for j, rate in enumerate(RATES):
            cell = df[(df.arch == key) & (df.rate == rate)]
            if cell.empty:
                continue
            m = cell.asr_rel_target.mean()
            gap = cell.gap_value.mean()
            dauroc = cell.auroc_delta_overall.mean()
            ok = bool(m >= GATE_ASR and gap >= GATE_GAP and abs(dauroc) <= GATE_STEALTH)
            asr[i, j] = m
            installed[i, j] = ok
            rows.append({
                "arch": key, "arch_label": label,
                "family": FAMILY.get(key, "convolutional"),
                "threshold_name": THRESHOLD, "rate": rate, "n_seeds": len(cell),
                "asr_rel_target": m, "asr_rel_control": cell.asr_rel_control.mean(),
                "gap_value": gap, "auroc_delta_overall": dauroc,
                "gates_all": ok,
            })

    out = pd.DataFrame(rows)
    for d in (FIG, REVFIG):
        out.to_csv(d / "fig05_arch_heatmap.csv", index=False)

    fig, ax = plt.subplots(figsize=(3.9, 2.9))
    im = ax.imshow(asr, cmap="viridis", vmin=0.0, vmax=0.40, aspect="auto")

    ax.set_xticks(range(len(RATES)))
    ax.set_xticklabels([f"{r:g}" for r in RATES])
    ax.set_yticks(range(len(ARCHS)))
    ax.set_yticklabels([label for _, label in ARCHS])
    ax.set_xlabel("Within-cell flip rate")
    ax.set_title("Attack success by architecture (Youden's $J$)")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    for i in range(len(ARCHS)):
        for j in range(len(RATES)):
            v = asr[i, j]
            if np.isnan(v):
                continue
            # viridis is dark at low values: switch the label to dark ink only
            # once the cell is light enough to carry it.
            ink = "black" if v > 0.26 else "white"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7.5,
                    fontweight="bold" if installed[i, j] else "normal",
                    color=ink)
            if installed[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor=ink, linewidth=1.5))

    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("$\\mathrm{ASR}_{\\mathrm{rel}}$, attacked subgroup", fontsize=7)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, length=2)

    save(fig, FIG / "fig05_arch_heatmap.png")
    shutil.copy2(FIG / "fig05_arch_heatmap.png",
                 REVFIG / "fig05_arch_heatmap.png")

    print("font:", check_font())
    n_installed = int(installed.sum())
    print(f"installed cells: {n_installed} of {installed.size}")
    for i, (key, label) in enumerate(ARCHS):
        marks = " ".join("*" if installed[i, j] else "." for j in range(len(RATES)))
        print(f"  {label:16s} " +
              "  ".join(f"{asr[i, j]:.3f}" for j in range(len(RATES))) + f"   {marks}")


if __name__ == "__main__":
    main()
