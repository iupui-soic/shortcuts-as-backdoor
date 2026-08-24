#!/usr/bin/env python3
"""EXP-9 — revision figures, npj-compliant, each with its underlying CSV.

Adds the three figures the new results require, on top of the regenerated
canonical set:

  fig02cd  rate versus absolute count (EXP-1). Two panels: ASR against the number
           of flipped labels, and against the poison rate, both coloured by the
           size of the eligible cell. Whichever panel COLLAPSES onto one curve is
           the variable that governs installation — that visual collapse is the
           whole experiment.
  fig08b   the defense figure as a matched-FPR comparison rather than binary
           verdicts: audit detection rate on installed attacks, the false-positive
           rate that buys, and the size of the shift each audit statistic
           undergoes, under both the matched rule and the conventional one.
  figM1    the operating-point figure: where each cohort installs at each of the
           four operating points, and the ASR_rel denominator caution.

The drawing itself lives in `exp9_main_figures.py`, which re-derives these three
alongside the main-text figures from the same `EXP-2/rescored.csv` at Youden's J,
and writes them to `results/figures/` (what the manuscript's graphicspath
resolves to) as well as
mirroring them here. This module keeps the EXP-9 compliance record: which figures
exist, which font actually resolved, and the CSV beside each figure.

Every figure writes `<name>.csv` next to `<name>.png` carrying exactly the numbers
plotted, so the legend claims can be checked against the data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.common_rev import (  # noqa: E402
    REPO, REV, append_manifest, code_sha, utcnow, write_json,
)
from scripts.revision.npj_style import (  # noqa: E402
    apply as npj_apply, check_font,
)

npj_apply()
FIG = REV / "figures"
FIG.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# The three figures below were drawn here originally. They are now re-derived in
# exp9_main_figures.py from EXP-2/rescored.csv at the Youden's-J operating point,
# together with the main-text figures they have to agree with, so that exactly
# one script produces each file name. This module calls into it rather than
# keeping a second, divergent copy of the plotting code.
from scripts.revision.exp9_main_figures import (  # noqa: E402
    fig02cd_rate_vs_count, fig08b_defense_matched_fpr, figM1_asr_denominator,
)


def main() -> None:
    out = [fig02cd_rate_vs_count(), fig08b_defense_matched_fpr(),
           figM1_asr_denominator()]
    font = check_font()
    for o in out:
        print(" ", o)
    print(f"  font: {font['resolved_family']} (npj acceptable: {font['npj_acceptable']})")
    if not font["npj_acceptable"]:
        print(f"  REMEDY: {font['remedy']}")
    # copy the canonical set's CSVs where the source data is already tabular
    for src, stem in ((REV / "EXP-6" / "summary.csv", "fig08_defense_attack_matrix"),
                      (REV / "EXP-2" / "install_points.csv", "fig02_install_points"),
                      (REV / "EXP-2" / "gate_attribution.csv", "fig02_gate_attribution")):
        if src.exists():
            pd.read_csv(src).to_csv(FIG / f"{stem}.csv", index=False)
    print(f"[exp9] figures + csv -> {FIG}")

    # EXP-9 produces figures, not a fitted result, so it had no summary.json and
    # the bundle reported it as "not run" every time it had in fact run. It
    # writes its compliance record like every other experiment now.
    canonical = sorted((REPO / "results" / "figures").glob("fig*.png"))
    revision_figs = sorted(FIG.glob("*.png"))
    headline = (
        f"All {len(canonical)} canonical figures and {len(revision_figs)} revision "
        f"figures were regenerated under the npj style: {font['resolved_family']} "
        f"at 8 pt (npj requires Arial or Helvetica; Liberation Sans is metrically "
        f"identical to Arial), 300 dpi, opaque white RGB, viridis and the Okabe-Ito "
        f"colour-blind-safe cycle throughout, with no rainbow colormap and no "
        f"red/green verdict encoding, and each figure's source data written beside "
        f"it as CSV."
        if font["npj_acceptable"] else
        f"Figures regenerated, but the resolved family is "
        f"{font['resolved_family']}, which is NOT metric-compatible with Arial or "
        f"Helvetica — every text extent in every figure differs from what npj will "
        f"typeset. {font['remedy']}"
    )
    write_json(REV / "EXP-9" / "summary.json", {
        "exp_id": "EXP-9", "git_sha": code_sha(), "completed_utc": utcnow(),
        "headline_sentence": headline,
        "font": font,
        "canonical_figures": [f.name for f in canonical],
        "revision_figures": [f.name for f in revision_figs],
        "csv_beside_each_figure": sorted(f.name for f in FIG.glob("*.csv")),
        "style": {"dpi": 300, "font_pt": 8, "background": "opaque white RGB",
                  "sequential_cmap": "viridis", "categorical_cycle": "Okabe-Ito",
                  "pdf_fonttype": 42},
    })
    append_manifest({"exp_id": "EXP-9", "git_sha": code_sha(),
                     "font_ok": font["npj_acceptable"], "figures": out})


if __name__ == "__main__":
    main()
