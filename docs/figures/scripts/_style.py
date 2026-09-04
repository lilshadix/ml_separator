"""Shared publication style: palette, typography, sizes, export.

Imported by every figure script so that one method has one visual identity in every
figure of the paper.  Nothing here computes a number.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Palette — Okabe-Ito, colour-blind safe, no rainbow, no decorative gradients.
# --------------------------------------------------------------------------- #
BLACK = "#000000"
GREY = "#8C8C8C"
ORANGE = "#E69F00"      # frozen baseline (gen8 global model)
SKY = "#56B4E9"         # + relative position
BLUE = "#0072B2"        # + recomposition  (the frozen global model of the final pipeline)
VERMILLION = "#D55E00"  # final deployed pipeline (recomposition + SERIES_ML)
GREEN = "#009E73"       # oracle / non-deployable upper bound
PURPLE = "#CC79A7"      # restricted-training-coverage arm (gen6 BASE)
YELLOW = "#F0E442"

#: method -> (label, colour, linestyle, marker)
METHODS: dict[str, tuple[str, str, str, str]] = {
    "measured":      ("Measured",                              BLACK,      "-",   "o"),
    "no_model":      ("Measurements only (no model)",          GREY,       ":",   "v"),
    "no_ligand":     ("No ligand information",                 GREY,       "--",  "x"),
    "frozen":        ("Baseline model",                         ORANGE,     "--",  "s"),
    "relpos":        ("+ relative position",                   SKY,        "-.",  "^"),
    "recomposed":    ("+ recomposition",                        BLUE,       "-",   "D"),
    "pipeline":      ("Final pipeline (+ series adaptation)",  VERMILLION, "-",   "o"),
    "oracle":        ("Oracle (not deployable)",               GREEN,      "--",  "*"),
    "base91":        ("Restricted training coverage",          PURPLE,     "--",  "s"),
    "expanded152":   ("Full training coverage",                BLUE,       "-",   "D"),
    "rowmatched":    ("Row-count-matched control",             SKY,        "-.",  "^"),
    "shuffled":      ("Shuffled-target control",               GREY,       ":",   "x"),
}


def m(key: str) -> dict:
    """Plot kwargs for a method key."""
    label, colour, ls, marker = METHODS[key]
    return {"label": label, "color": colour, "linestyle": ls, "marker": marker}


def colour(key: str) -> str:
    return METHODS[key][1]


def label(key: str) -> str:
    return METHODS[key][0]


# --------------------------------------------------------------------------- #
# Canvas sizes (inches).  Journal single column 86 mm, 1.5 column 127 mm,
# double column 178 mm.
# --------------------------------------------------------------------------- #
W1 = 3.39
W15 = 5.00
W2 = 7.01


def apply() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.labelsize": 7.5,
        "axes.titlesize": 7.5,
        "axes.titleweight": "regular",
        "axes.titlelocation": "left",
        "axes.titlepad": 3.0,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.5,
        "legend.frameon": False,
        "legend.handlelength": 1.9,
        "legend.handletextpad": 0.6,
        "legend.labelspacing": 0.35,
        "legend.borderaxespad": 0.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.2,
        "lines.markersize": 3.4,
        "lines.markeredgewidth": 0.0,
        "grid.linewidth": 0.4,
        "grid.color": "#DDDDDD",
        "axes.grid": False,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "errorbar.capsize": 1.6,
    })


def panel(ax, letter: str, dx: float = -0.12, dy: float = 1.06) -> None:
    """Bold panel letter in axes coordinates, consistent across all figures."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", va="top", ha="left")


def save(fig, stem: str, out_dir: Path) -> list[Path]:
    """Write PDF (vector) and PNG (600 dpi) with identical geometry."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in (".pdf", ".png"):
        path = out_dir / f"{stem}{suffix}"
        fig.savefig(path)
        written.append(path)
    plt.close(fig)
    return written
