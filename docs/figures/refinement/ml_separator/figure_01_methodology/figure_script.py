#!/usr/bin/env python
"""Figure 1 — task, held-out protocol and model.

Refined from ``figures/scripts/plot_fig1_methodology.py``.  This is a schematic: the only
quantities printed on it are cohort counts, and they are read from the frozen run outputs
rather than typed in:

  runs/gen9_shape/recomposed/oof_GEN9_SHAPE_RECOMPOSED.parquet   rows / extractants /
                                                                 lanthanides / condition
                                                                 cells / chemotypes
  figures/derived/kshot_cohort.json                              k-shot cohort, seeds,
                                                                 pool draws

Layout notes for whoever nudges this next
-----------------------------------------
Nothing is positioned by a bare number pasted into a draw call.  Every element is placed
through three primitives:

``Panel``   wraps one axes and converts *points* to axes fractions in x and y, so a gap
            can be written as "8 pt" and stay 8 pt whatever the figure size becomes.
``Box``     a rectangle that knows its own edges, so an arrow is written between two named
            anchors (``mat.east()`` -> ``f1.west()``) and never between two typed points.
``split``   divides a horizontal span into weighted columns with a named gap, so a row of
            five flow boxes or eight chips is one call, not five or eight coordinates.

Box heights are *derived* from the text they hold (line count x font size), and box text is
wrapped to the box width using real font metrics, so changing a word or a font size cannot
silently overflow a box.  Vertical stacks are laid out with a top-down cursor: each block
starts from the previous block's edge plus a named gap, so moving one block moves what
follows it instead of requiring every coordinate below to be re-derived.

Run from anywhere:
    python figure_refinement/ml_separator/figure_01_methodology/figure_script.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors     # noqa: E402
import matplotlib.pyplot as plt         # noqa: E402
import pandas as pd                     # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch     # noqa: E402
from matplotlib.text import Text                        # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common"))
import mlsep_data as D                  # noqa: E402
import pubstyle as PS                   # noqa: E402

# --------------------------------------------------------------------------- #
# Canvas
# --------------------------------------------------------------------------- #
FIG_W = PS.W2                      # 180 mm, double column
FIG_H = 6.00
#: Set from the measured content height of each row: 2.75 in for A|B, 2.44 in for C and
#: 0.40 in for the key, each with about 0.06 in of slack.  ``main`` prints the space left
#: over in every panel, so these stay honest: a negative "content bottom" is an over-full
#: panel whose text has spilled into its neighbour.
HEIGHT_RATIOS = (0.98, 0.87, 0.155)
WIDTH_RATIOS = (1.00, 1.38)         # A is a column of blocks, B is a three-column flow

# --------------------------------------------------------------------------- #
# Typography.  Nothing on this figure is smaller than PS.BASE - 0.5 = 7.5 pt.
# --------------------------------------------------------------------------- #
FS_HEAD = PS.BASE + 0.5            # 8.5  panel heading, bold
FS_BODY = PS.BASE                  # 8.0  paragraph body text
FS_KEY = PS.BASE                   # 8.0  information-class key
FS_BOX = PS.BASE - 0.5             # 7.5  text inside a box
FS_NOTE = PS.BASE - 0.5            # 7.5  caption attached to a row of chips
LINESPACING = 1.32

#: Named vertical gaps, in points.  Used by the top-down cursors below.
GAP_HEAD = 7.0                     # heading -> first block
GAP_BLOCK = 7.0                    # block -> block
GAP_TIGHT = 3.5                    # block -> its own caption
GAP_PARA = 10.0                    # last block -> the paragraph under it
PAD_X, PAD_Y = 5.0, 4.0            # padding inside a box, in points
CORNER = 2.2                       # corner radius of every box, in points
LETTER_PAD = 3.0                   # top of a panel kept clear for its panel letter


def tint(colour: str, frac: float) -> str:
    """``colour`` blended ``frac`` of the way out of white — the pale fill of its class."""
    r, g, b = mcolors.to_rgb(colour)
    return mcolors.to_hex(tuple(1.0 - frac * (1.0 - c) for c in (r, g, b)))


@dataclass(frozen=True)
class InfoClass:
    """One of the four classes of information the protocol keeps apart."""
    edge: str
    face: str
    key: str = ""                  # wording in the key band; "" = not a key entry


#: The whole point of the figure.  A *filled* box is information; an *outlined* box is an
#: operation, drawn in the colour of the information it is allowed to read.  Colours are
#: Okabe-Ito, taken from pubstyle so that blue = training-side and vermillion = the
#: measured support everywhere in the paper.
CLASSES: dict[str, InfoClass] = {
    "train": InfoClass(PS.BLUE, tint(PS.BLUE, 0.24),
                       "training data\n(other chemotypes)"),
    "conds": InfoClass(PS.SKY, tint(PS.SKY, 0.15),
                       "target-free conditions\nof the new extractant"),
    "support": InfoClass(PS.VERMILLION, tint(PS.VERMILLION, 0.16),
                         "measured support\n(the $k$ targets)"),
    "query": InfoClass(PS.INK, tint(PS.INK, 0.13),
                       "held-out query rows\n(scored, never read)"),
}
#: Operations: white fill, edge of the class they may read.
OPS = {name: InfoClass(cls.edge, "white") for name, cls in CLASSES.items()}
#: A pool row the policy did not pick.  Deliberately classless and grey: it was available
#: to the policy but was never measured, so it carries no information either way.
UNPICKED = InfoClass("#9C9C9C", "white")
BARRIER = "#A6A6A6"                # dashed line = nothing crosses here


# --------------------------------------------------------------------------- #
# Geometry primitives
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Box:
    """A rectangle in axes fractions that knows its own edges."""
    x: float
    y: float
    w: float
    h: float

    @property
    def left(self) -> float:
        return self.x

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y

    @property
    def top(self) -> float:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    def north(self, t: float = 0.5) -> tuple[float, float]:
        return (self.x + t * self.w, self.top)

    def south(self, t: float = 0.5) -> tuple[float, float]:
        return (self.x + t * self.w, self.bottom)

    def east(self, t: float = 0.5) -> tuple[float, float]:
        return (self.right, self.y + t * self.h)

    def west(self, t: float = 0.5) -> tuple[float, float]:
        return (self.left, self.y + t * self.h)


def split(x0: float, x1: float, weights, gap: float) -> list[tuple[float, float]]:
    """Columns spanning exactly ``[x0, x1]``, sized by ``weights``, separated by ``gap``.

    Returns ``[(x, width), ...]``.  Every row of boxes and every row of chips in this
    script is laid out through this one function, so a row always fills its span exactly
    and keeps its rhythm when an element is added or removed.
    """
    span = (x1 - x0) - gap * (len(weights) - 1)
    total = float(sum(weights))
    out, x = [], x0
    for weight in weights:
        w = span * weight / total
        out.append((x, w))
        x += w + gap
    return out


def track(x0: float, x1: float, n: int, gap: float) -> list[tuple[float, float]]:
    """``n`` equal columns spanning ``[x0, x1]``."""
    return split(x0, x1, [1.0] * n, gap)


class Panel:
    """One axes of the schematic, plus points-to-axes-fraction conversion.

    Requires the figure to have been drawn once so that constrained_layout has settled and
    the axes' size in inches is known.
    """

    def __init__(self, ax):
        self.ax = ax
        self.fig = ax.figure
        pos = ax.get_position()
        self.w_in = pos.width * self.fig.get_figwidth()
        self.h_in = pos.height * self.fig.get_figheight()

    def fx(self, pts: float) -> float:
        return pts / 72.0 / self.w_in

    def fy(self, pts: float) -> float:
        return pts / 72.0 / self.h_in

    @property
    def letter_top(self) -> float:
        """Top of the drawable region.

        ``pubstyle.add_panel_letters`` places the letter above the axes' own top edge, so
        only a hair of clearance is needed inside the panel.
        """
        return 1.0 - self.fy(LETTER_PAD)

    def rounded(self, x: float, y: float, w: float, h: float, cls: InfoClass,
                lw: float) -> FancyBboxPatch:
        """A box whose corners are ``CORNER`` points in *both* directions.

        The axes are 0-1 in x and y but are not square, so a radius given in axes
        fractions would be an ellipse — and on a chip it would exceed half the box and
        make the outline self-intersect.  Radius is therefore set in points and clamped
        to the box.
        """
        w_pt, h_pt = w * self.w_in * 72.0, h * self.h_in * 72.0
        r_pt = min(CORNER, 0.34 * w_pt, 0.34 * h_pt)
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={self.fx(r_pt)}",
            mutation_aspect=self.w_in / self.h_in,
            linewidth=lw, edgecolor=cls.edge, facecolor=cls.face, zorder=3)
        self.ax.add_patch(patch)
        return patch


# --------------------------------------------------------------------------- #
# Text metrics — real font measurement, so no magic characters-per-line constant
# --------------------------------------------------------------------------- #
_FIT: list[str] = []               # boxes whose text does not fit; printed as a QC report


_WIDTH_CACHE: dict[tuple[str, float], float] = {}


def text_w_in(fig, s: str, size: float) -> float:
    """Width of one line of text, in inches, measured with the actual font.

    A throw-away :class:`Text` artist is measured rather than the raw string, because the
    string may contain mathtext (``log$_{10}$ $D$``): measuring the markup would report a
    line as 30 % wider than it draws and would send the layout chasing a phantom overflow.
    """
    hit = _WIDTH_CACHE.get((s, size))
    if hit is None:
        artist = Text(0, 0, s, fontsize=size)
        artist.set_figure(fig)
        hit = artist.get_window_extent(renderer=fig.canvas.get_renderer()).width / fig.dpi
        _WIDTH_CACHE[(s, size)] = hit
    return hit


def wrap(fig, text: str, size: float, max_in: float) -> str:
    """Greedy wrap to ``max_in`` inches, preserving any explicit line breaks."""
    out = []
    for para in text.split("\n"):
        line = ""
        for word in para.split():
            trial = f"{line} {word}".strip()
            if line and text_w_in(fig, trial, size) > max_in:
                out.append(line)
                line = word
            else:
                line = trial
        out.append(line)
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Drawing primitives
# --------------------------------------------------------------------------- #
def fit(p: Panel, text: str, w: float, size: float = FS_BOX,
        wrap_text: bool = True) -> list[str]:
    """The lines a box of width ``w`` will hold."""
    inner = w * p.w_in - 2 * PAD_X / 72.0
    body = wrap(p.fig, text, size, inner) if wrap_text else text
    return body.split("\n")


def box(p: Panel, x: float, top: float, w: float, text: str, cls: InfoClass, *,
        size: float = FS_BOX, lw: float = 0.9, align: str = "center",
        min_lines: int = 1, wrap_text: bool = True, name: str = "") -> Box:
    """Draw a rounded box whose height is derived from the text it holds.

    ``min_lines`` lets a row of boxes share the height of its tallest member.
    """
    inner = w * p.w_in - 2 * PAD_X / 72.0
    lines = fit(p, text, w, size, wrap_text)
    h = p.fy(max(len(lines), min_lines) * size * LINESPACING + 2 * PAD_Y)
    y = top - h
    p.rounded(x, y, w, h, cls, lw)
    tx = x + w / 2 if align == "center" else x + p.fx(PAD_X)
    p.ax.text(tx, y + h / 2, "\n".join(lines), ha=align, va="center", fontsize=size,
              linespacing=LINESPACING, zorder=4, color=PS.INK)
    widest = max(text_w_in(p.fig, ln, size) for ln in lines)
    if widest > inner + 1e-3:
        _FIT.append(f"{name or lines[0]!r}: text {widest:.2f} in > box {inner:.2f} in")
    return Box(x, y, w, h)


def box_row(p: Panel, cols, items, top: float, *, size: float = FS_BOX) -> list[Box]:
    """A row of boxes of equal height — the height the longest entry needs."""
    tallest = max(len(fit(p, text, w, size)) for (text, _, _), (_, w) in zip(items, cols))
    return [box(p, x, top, w, text, cls, size=size, min_lines=tallest, name=name)
            for (text, cls, name), (x, w) in zip(items, cols)]


def chip(p: Panel, x: float, y: float, w: float, h: float, cls: InfoClass,
         lw: float = 0.8) -> Box:
    """An empty box standing for one row of the corpus (or one chemotype)."""
    p.rounded(x, y, w, h, cls, lw)
    return Box(x, y, w, h)


def arrow(p: Panel, start, end, colour: str, *, lw: float = 1.0, ls: str = "-") -> None:
    """An arrow carrying information; its colour is the class of what travels."""
    p.ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=7.0, linewidth=lw,
        color=colour, linestyle=ls, shrinkA=2.0, shrinkB=2.0, zorder=2))


def heading(p: Panel, x: float, text: str) -> float:
    """Panel heading; returns the y of its baseline block bottom."""
    top = p.letter_top
    h = p.fy(FS_HEAD * LINESPACING)
    p.ax.text(x, top - h / 2, text, ha="left", va="center", fontsize=FS_HEAD,
              fontweight="bold", color=PS.INK)
    return top - h


def note(p: Panel, x: float, top: float, text: str, colour: str, *,
         size: float = FS_NOTE, align: str = "left", max_in: float | None = None) -> float:
    """A line (or wrapped block) of loose text; returns the y of its bottom."""
    body = wrap(p.fig, text, size, max_in) if max_in else text
    lines = body.count("\n") + 1
    h = p.fy(lines * size * LINESPACING)
    p.ax.text(x, top - h / 2, body, ha=align, va="center", fontsize=size,
              color=colour, linespacing=LINESPACING, zorder=4)
    return top - h


def barrier(p: Panel, x: float, y0: float, y1: float) -> None:
    p.ax.plot([x, x], [y0, y1], color=BARRIER, linewidth=0.9,
              linestyle=(0, (2.2, 2.2)), zorder=2)


# --------------------------------------------------------------------------- #
# Cohort counts — read from the frozen run outputs, never typed in
# --------------------------------------------------------------------------- #
def counts() -> dict:
    oof = pd.read_parquet(
        D.run("gen9_shape/recomposed/oof_GEN9_SHAPE_RECOMPOSED.parquet"),
        columns=["row_id", "extractant", "ecfp_cluster", "tanimoto_cluster",
                 "series_id", "metal_symbol", "split_seed", "condition_id"])
    one = oof[oof.split_seed == oof.split_seed.min()]
    curves = pd.read_parquet(D.run("gen9_shape/curves/curve_table.parquet"))
    kshot = json.loads(D.derived("kshot_cohort.json", "prepare_kshot_tables.py").read_text())
    return {"rows": int(len(one)), "extractants": int(one.extractant.nunique()),
            "ecfp_clusters": int(one.ecfp_cluster.nunique()),
            "chemotypes": int(one.tanimoto_cluster.nunique()),
            "metals": int(one.metal_symbol.nunique()),
            "series": int(one.series_id.nunique()),
            "conditions": int(one.condition_id.nunique()),
            "curves": int(len(curves)),
            "extractant_curves": int((curves.axis_label == "extractant").sum()),
            "kshot_ligands": kshot["n_ligands_common"],
            "seeds": len(kshot["seeds"]), "repeats": kshot["repeats_per_seed"]}


# --------------------------------------------------------------------------- #
# Panel A — corpus and held-out split
# --------------------------------------------------------------------------- #
A_X0, A_X1 = 0.010, 0.990
A_CHIP_H = 15.0                    # pt
#: Slots across the chemotype row: four training groups, an empty gutter that carries the
#: fold boundary, and the held-out group — the 5 outer folds, one of them held out.
A_CHIP_SLOTS = (1.0, 1.0, 1.0, 1.0, 0.7, 1.0)
A_GUTTER = 4                       # index of the empty slot


def panel_a(p: Panel, n: dict) -> float:
    y = heading(p, A_X0, "Task, corpus and held-out split") - p.fy(GAP_TIGHT)
    y = note(p, A_X0, y,
             "predict log$_{10}$ $D$, the aqueous/organic distribution ratio of one "
             "lanthanide, per (extractant, lanthanide, conditions) row",
             PS.GREY, max_in=(A_X1 - A_X0) * p.w_in) - p.fy(GAP_HEAD)

    corpus = box(p, A_X0, y, A_X1 - A_X0,
                 f"{n['rows']:,} measurements of log$_{{10}}$ $D$\n"
                 f"{n['extractants']} extractants  ·  {n['metals']} lanthanides\n"
                 f"{n['conditions']:,} condition cells",
                 CLASSES["train"], wrap_text=False, name="corpus")

    y = corpus.bottom - p.fy(GAP_BLOCK)
    arrow(p, (corpus.cx, y), (corpus.cx, y - p.fy(11.0)), PS.BLUE)
    y = note(p, corpus.cx, y - p.fy(13.0),
             f"grouped into {n['chemotypes']} Tanimoto-0.7 chemotypes",
             PS.INK, align="center")

    # The 5 outer folds: four training chemotypes, a barrier, one held out.
    y -= p.fy(GAP_BLOCK)
    slots = split(A_X0, A_X1, A_CHIP_SLOTS, p.fx(5.0))
    chip_h = p.fy(A_CHIP_H)
    groups = [chip(p, x, y - chip_h, w, chip_h, CLASSES["train"])
              for x, w in slots[:A_GUTTER]]
    gutter_x, gutter_w = slots[A_GUTTER]
    held_x, held_w = slots[A_GUTTER + 1]
    held = chip(p, held_x, y - chip_h, held_w, chip_h, CLASSES["query"], lw=1.0)
    train_span = (groups[0].left, groups[-1].right)

    cap_top = y - chip_h - p.fy(GAP_TIGHT)
    note(p, sum(train_span) / 2, cap_top, "training chemotypes", PS.BLUE, align="center")
    y = note(p, held.cx, cap_top, "held out", PS.INK, align="center")
    barrier(p, gutter_x + gutter_w / 2, y, groups[0].top + p.fy(3.0))

    return note(p, A_X0, y - p.fy(GAP_PARA),
                "Every extractant within Tanimoto 0.7 of a held-out one is held out with "
                "it, so no close analogue of it remains in training. Five outer folds × "
                "five split seeds; the model seed is 42, independent of the split.",
                PS.INK, size=FS_BODY, max_in=(A_X1 - A_X0) * p.w_in)


# --------------------------------------------------------------------------- #
# Panel B — the global model
# --------------------------------------------------------------------------- #
B_X0, B_X1 = 0.010, 0.990
B_COL_GAP = 0.040                  # between the three columns of the flow
#: Relative column widths, chosen so that the widest line in each column fits: the
#: "box fit" report printed by ``main`` is empty only if they do.  The boxes name the
#: operations and the paragraph carries the composition detail — three columns of prose
#: inside boxes would not fit this panel at 7.5 pt.
B_COL_W = (1.33, 1.09, 1.23)       # inputs | two forests | recomposition
B_ROW_GAP = 14.0                   # pt, between the two input boxes


def panel_b(p: Panel, n: dict) -> float:
    top = heading(p, B_X0, "Global model, fitted on training chemotypes only") \
        - p.fy(GAP_HEAD)
    (x_in, w_in), (x_mid, w_mid), (x_out, w_out) = split(B_X0, B_X1, B_COL_W, B_COL_GAP)

    design = box(p, x_in, top, w_in,
                 "design matrix\nchemistry, conditions,\nmass-action terms",
                 CLASSES["train"], wrap_text=False, name="design matrix")
    # The "target-free" caveat lives inside the box: as loose text it had to be sky blue
    # to mark its class, and sky blue on white is too weak for 7.5 pt body type.
    relpos = box(p, x_in, design.bottom - p.fy(B_ROW_GAP), w_in,
                 "position of the row in\nits titration window\n"
                 "(target-free: from the\ncandidate condition list)",
                 CLASSES["conds"], wrap_text=False, name="relative position")

    f1 = box(p, x_mid, top, w_mid,
             "forest 1\ntarget: log$_{10}$ $D$", OPS["train"],
             wrap_text=False, name="forest 1")
    f2 = box(p, x_mid, relpos.top, w_mid,
             "forest 2\ntarget: log$_{10}$ $D$\nminus curve mean",
             OPS["train"], wrap_text=False, name="forest 2")

    recomp = box(p, x_out, (top + f2.top) / 2, w_out,
                 "recomposition\nforest 1 curve mean\n$+$ forest 2 shape",
                 OPS["train"], wrap_text=False, name="recomposition")

    arrow(p, design.east(0.75), f1.west(0.5), PS.BLUE)
    arrow(p, design.south(0.86), f2.west(0.74), PS.BLUE)
    arrow(p, relpos.east(0.60), f2.west(0.28), PS.SKY)
    arrow(p, f1.east(0.5), recomp.west(0.78), PS.BLUE)
    arrow(p, f2.east(0.5), recomp.west(0.22), PS.BLUE)

    return note(p, B_X0, min(relpos.bottom, f2.bottom) - p.fy(GAP_PARA),
                "The design matrix holds a fingerprint block, 2-D descriptors, the "
                "lanthanide, the raw condition columns and log-concentration mass-action "
                "terms. A curve is a maximal set of rows differing in exactly one "
                f"condition axis ({n['curves']:,} in the corpus). Recomposition is "
                "mean-preserving: each curve keeps forest 1's mean prediction, so the "
                "per-extractant level is unchanged and only the within-curve shape "
                "changes.",
                PS.INK, size=FS_BODY, max_in=(B_X1 - B_X0) * p.w_in)


# --------------------------------------------------------------------------- #
# Panel C — deployment and evaluation on one held-out extractant
# --------------------------------------------------------------------------- #
C_X0, C_X1 = 0.008, 0.992
C_FLOW_GAP = 0.026
C_LANE_GAP = 24.0                  # pt below the flow row: the diagonal arrow runs here
C_CHIP_H = 15.0                    # pt
C_N_POOL, C_N_QUERY = 8, 6
C_SELECTED = (2, 5)                # which pool chips the policy picked, for illustration
C_POOL_W = 0.340                   # the candidate-pool strip
C_LANE_SEP = 0.075                 # gap between the lanes; the boundary sits in the middle
C_QUERY_W = 0.255                  # the query strip
C_SCORE_W = 0.185                  # the "macro MAE" box, right-hand end of both lanes


def panel_c(p: Panel, n: dict) -> float:
    top = heading(p, C_X0, "Deployment and evaluation on one held-out extractant") \
        - p.fy(GAP_HEAD)

    # ---- the five deployment steps, one row across the panel ----------------
    steps = [
        ("the user lists candidate conditions to run", CLASSES["conds"], "candidates"),
        ("frozen model scores every candidate — zero-shot", OPS["train"], "zero-shot"),
        ("policy picks $k$ rows; it reads conditions and predictions, never a target",
         OPS["conds"], "acquisition"),
        ("those $k$ rows are measured in the lab", CLASSES["support"], "measure"),
        ("shrunk calibrator adjusts the frozen model", OPS["support"], "calibrate"),
    ]
    cols = track(C_X0, C_X1, len(steps), C_FLOW_GAP)
    flow = box_row(p, cols, steps, top)
    lane = [PS.SKY, PS.BLUE, PS.SKY, PS.VERMILLION]     # what travels along each arrow
    for b0, b1, colour in zip(flow, flow[1:], lane):
        arrow(p, b0.east(), b1.west(), colour)

    # ---- the extractant's own rows, split into two disjoint lanes ------------
    y = note(p, C_X0, flow[0].bottom - p.fy(C_LANE_GAP),
             "the extractant's own rows, split once per repeat:",
             PS.INK) - p.fy(GAP_TIGHT)

    chip_h = p.fy(C_CHIP_H)
    pool = track(C_X0, C_X0 + C_POOL_W, C_N_POOL, p.fx(3.0))
    for i, (x, w) in enumerate(pool):
        picked = i in C_SELECTED
        chip(p, x, y - chip_h, w, chip_h,
             CLASSES["support"] if picked else UNPICKED, lw=1.1 if picked else 0.7)
    pool_box = Box(pool[0][0], y - chip_h, pool[-1][0] + pool[-1][1] - pool[0][0], chip_h)

    q_x0 = pool_box.right + C_LANE_SEP
    query = track(q_x0, q_x0 + C_QUERY_W, C_N_QUERY, p.fx(3.0))
    for x, w in query:
        chip(p, x, y - chip_h, w, chip_h, CLASSES["query"])
    query_box = Box(query[0][0], y - chip_h,
                    query[-1][0] + query[-1][1] - query[0][0], chip_h)

    cap = pool_box.bottom - p.fy(GAP_TIGHT)
    note(p, pool_box.cx, cap, "candidate pool — $k$ rows selected and measured",
         PS.INK, align="center")
    y_cap = note(p, query_box.cx, cap, "query rows — scored, never read",
                 PS.INK, align="center")
    barrier(p, (pool_box.right + query_box.left) / 2, y_cap, pool_box.top + p.fy(4.0))

    # ---- scoring: the right-hand end of both lanes --------------------------
    score = box(p, C_X1 - C_SCORE_W, pool_box.top + p.fy(9.0), C_SCORE_W,
                "macro MAE over the query rows, one vote per extractant",
                OPS["query"], name="score")
    arrow(p, flow[-1].south(0.5), (score.cx, score.top), PS.VERMILLION)
    arrow(p, query_box.east(), (score.left, pool_box.cy), PS.INK)
    arrow(p, pool_box.north(1.0), flow[2].south(0.30), PS.SKY)

    return note(p, C_X0, min(y_cap, score.bottom) - p.fy(GAP_PARA),
                "$k\\in\\{0,1,2,3,5\\}$. Cohort: "
                f"{n['kshot_ligands']} held-out extractants with at least 5 pool rows and "
                f"2 query rows in every arm; {n['seeds']} split seeds × {n['repeats']} "
                "pool draws. Support targets are read only by the calibrator, and a "
                "support row is never also scored as a query row. The global model is "
                "never refitted on the held-out extractant, so the $k$ measurements buy "
                "calibration, not training.",
                PS.INK, size=FS_BODY, max_in=(C_X1 - C_X0) * p.w_in)


# --------------------------------------------------------------------------- #
# Information-class key — its own axes, so it can never overlap a panel
# --------------------------------------------------------------------------- #
KEY_X0, KEY_X1 = 0.030, 0.970
KEY_SWATCH_W = 0.022
KEY_TOP = 0.95
KEY_SWATCH_H = 11.0                # pt


def key(p: Panel) -> None:
    cols = track(KEY_X0, KEY_X1, len(CLASSES), 0.012)
    top = KEY_TOP
    sw_h = p.fy(KEY_SWATCH_H)
    for (x, w), cls in zip(cols, CLASSES.values()):
        chip(p, x, top - sw_h, KEY_SWATCH_W, sw_h, cls, lw=1.0)
        p.ax.text(x + KEY_SWATCH_W + p.fx(4.0), top - sw_h / 2, cls.key,
                  ha="left", va="center", fontsize=FS_KEY, linespacing=LINESPACING,
                  color=PS.INK)
    note(p, (KEY_X0 + KEY_X1) / 2, top - sw_h - p.fy(9.0),
         "filled = information   ·   outline and arrow = coloured by the information "
         "they may read   ·   dashes = a boundary no target crosses",
         PS.GREY, align="center")


# --------------------------------------------------------------------------- #
def main() -> int:
    PS.apply()
    n = counts()

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs = fig.add_gridspec(3, 2, height_ratios=HEIGHT_RATIOS, width_ratios=WIDTH_RATIOS)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, :])
    axK = fig.add_subplot(gs[2, :])
    for ax in (axA, axB, axC, axK):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
    fig.canvas.draw()              # settle constrained_layout before measuring the panels

    bottoms = {"A": panel_a(Panel(axA), n),
               "B": panel_b(Panel(axB), n),
               "C": panel_c(Panel(axC), n)}
    key(Panel(axK))

    PS.add_panel_letters(fig, [axA, axB, axC])
    report = PS.save(fig, HERE, "figure", strict=False)
    print(report)
    print("box fit:", "all text inside its box" if not _FIT else _FIT)
    print("content bottom (axes fraction; < 0 means the panel is over-full): "
          + "  ".join(f"{k}={v:+.3f}" for k, v in bottoms.items()))
    for name, ax in (("A", axA), ("B", axB), ("C", axC), ("key", axK)):
        pos = ax.get_position()
        print(f"  panel {name}: {pos.width * FIG_W:.2f} x {pos.height * FIG_H:.2f} in"
              + (f"   content {(1 - bottoms[name]) * pos.height * FIG_H:.2f} in"
                 if name in bottoms else ""))

    values = {
        "figure": "figure_01_methodology",
        "note": "schematic: the only quantities drawn are cohort counts, read from the "
                "frozen run outputs",
        "counts": n,
        "information_classes": {k: {"edge": v.edge, "face": v.face}
                                for k, v in CLASSES.items()},
        "box_fit_warnings": _FIT,
        "lint": {"ok": report.ok, "violations": [str(v) for v in report.violations]},
    }
    (HERE / "values.json").write_text(json.dumps(values, indent=1))
    print(json.dumps(n, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
