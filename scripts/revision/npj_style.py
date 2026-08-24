"""EXP-9 — npj Digital Medicine figure compliance.

Every item in the §10 table that can be fixed globally is fixed here, so the
individual figure scripts need only the panel labels and the two hard-coded
colour choices changed:

  * Arial/Helvetica throughout, ~8 pt at final printed size
  * >= 300 dpi, RGB, white (never transparent) background
  * no rainbow/jet colormaps anywhere: the default sequential map is viridis and
    the default categorical cycle is the Okabe-Ito colour-blind-safe set
  * red/green pairs replaced by a blue/orange pair that survives both
    deuteranopia and greyscale printing

FONT. npj requires Arial or Helvetica. Neither is licensed for redistribution
on Linux, so the figures use **Liberation Sans**, which is metrically identical
to Arial — same advance widths, same line breaks, glyph shapes drawn to match —
and is npj's own listed substitute. It is installed per-user, no root involved:

    curl -sSLO http://archive.ubuntu.com/ubuntu/pool/main/f/fonts-liberation/fonts-liberation_2.1.5-3build1_all.deb
    dpkg-deb -x fonts-liberation_2.1.5-3build1_all.deb /tmp/fl
    cp /tmp/fl/usr/share/fonts/truetype/liberation/LiberationSans-*.ttf ~/.fonts/
    rm -f ~/.cache/matplotlib/fontlist-*.json   # matplotlib caches the font list

`~/.fonts` is required, not just `~/.local/share/fonts`: fontconfig reads both,
but matplotlib builds its own list and picks up the former. `check_font()`
reports which family actually resolved, so a fallback to DejaVu Sans — which is
NOT metric-compatible and would change every text extent in the figure — is
recorded rather than silently shipped.
"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt

# Okabe-Ito: colour-blind safe, and distinguishable in greyscale
OKABE_ITO = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

# Replacements for the red/green verdict encoding in fig08, which npj flags
# because red/green carries no information for a deuteranopic reader. The
# written verdict label is kept in the cell so colour is redundant, never
# load-bearing.
VERDICT_COLORS = {
    "YES": "#0072B2",   # blue   — defense detects/defeats
    "no": "#D55E00",    # orange — defense fails
    "weak": "#E69F00",  # light orange — partial
    "n/a": "#9E9E9E",   # grey
}

BLUE, ORANGE, GREY = OKABE_ITO[0], OKABE_ITO[1], "#9E9E9E"

FONT_STACK = ["Arial", "Helvetica", "Liberation Sans", "Arimo",
              "Nimbus Sans", "Helvetica Neue", "DejaVu Sans"]

_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": FONT_STACK,
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "axes.titleweight": "bold",
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "figure.titlesize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "savefig.transparent": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "image.cmap": "viridis",
    "axes.prop_cycle": matplotlib.cycler(color=OKABE_ITO),
    "pdf.fonttype": 42,      # embed TrueType, not Type 3 (npj requirement)
    "ps.fonttype": 42,
}


def apply() -> None:
    """Install the npj rcParams. Idempotent; safe to call from every figure script."""
    matplotlib.rcParams.update(_RC)


def check_font() -> dict:
    """Which family actually resolved, and is it npj-acceptable?"""
    from matplotlib import font_manager as fm

    available = {f.name for f in fm.fontManager.ttflist}
    resolved = next((f for f in FONT_STACK if f in available), None)
    acceptable = resolved in ("Arial", "Helvetica", "Liberation Sans", "Arimo",
                             "Nimbus Sans", "Helvetica Neue")
    return {
        "resolved_family": resolved,
        "npj_acceptable": bool(acceptable),
        "requirement": "Arial or Helvetica",
        "remedy": None if acceptable else
        "install a metric-compatible family per-user (no root): extract "
        "LiberationSans-*.ttf from the fonts-liberation .deb into ~/.fonts and "
        "delete ~/.cache/matplotlib/fontlist-*.json — see this module's "
        "docstring; no figure code needs to change",
    }


def panel_labels(axes, labels: str | list[str] = None, x: float = -0.02,
                 y: float = 1.06, fontsize: int = 10) -> None:
    """Bold lower-case a) b) c) at the top-left of each panel, as §10 requires.

    `axes` may be a single Axes, a flat list, or a 2-D array from plt.subplots.
    """
    import numpy as np

    if hasattr(axes, "flatten"):
        axes = list(np.asarray(axes).flatten())
    elif not isinstance(axes, (list, tuple)):
        axes = [axes]
    if labels is None:
        labels = [f"{chr(ord('a') + i)}" for i in range(len(axes))]
    for ax, lab in zip(axes, labels):
        ax.text(x, y, f"{lab}", transform=ax.transAxes, fontsize=fontsize,
                fontweight="bold", va="bottom", ha="right")


def save(fig, path, **kw) -> None:
    """Save at >=300 dpi on an opaque white RGB canvas, one file per figure.

    Matplotlib always writes PNG as RGBA. The alpha channel is fully opaque here
    (`savefig.transparent` is False and the canvas is white), so it carries no
    information, but npj asks for RGB and an alpha channel is exactly what a
    "transparent background" check looks at. Flattening it away afterwards costs
    nothing and removes the ambiguity.
    """
    kw.setdefault("dpi", 300)
    kw.setdefault("facecolor", "white")
    kw.setdefault("bbox_inches", "tight")
    fig.savefig(path, **kw)
    plt.close(fig)
    if str(path).lower().endswith(".png"):
        _flatten_to_rgb(path)


def _flatten_to_rgb(path) -> None:
    """Composite an opaque RGBA PNG onto white and rewrite it as RGB."""
    try:
        from PIL import Image
    except ImportError:      # Pillow absent: the RGBA file is still valid
        return
    with Image.open(path) as im:
        if im.mode == "RGB":
            return
        dpi = im.info.get("dpi", (300, 300))
        rgba = im.convert("RGBA")
        flat = Image.new("RGB", rgba.size, (255, 255, 255))
        flat.paste(rgba, mask=rgba.split()[3])
    flat.save(path, dpi=dpi)
