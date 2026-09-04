"""Shared publication style and layout linter for the figure-refinement pass.

Both repositories' refined figure scripts import this module, so one method has one
visual identity everywhere and typography is set in exactly one place.

The module also provides :func:`lint` — an *automatic* layout check that reads the
rendered artist geometry and reports text-text overlaps, text clipped by the canvas and
text that escapes its own axes.  It exists because "the script ran" is not evidence that
a figure is readable, and eyeballing 30 figures misses collisions that a bounding-box
test finds every time.  :func:`save` runs it before writing and raises unless the caller
opts out, so a figure cannot be finalised with a known collision in it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.text import Annotation, Text

# --------------------------------------------------------------------------- #
# Palette — Okabe-Ito.  Colour-blind safe under deuteranopia, protanopia and
# tritanopia.  No rainbow ramps, no colour that is not also carried by another
# channel (line style, marker, or position).
# --------------------------------------------------------------------------- #
BLACK = "#000000"
INK = "#1A1A1A"
GREY = "#7F7F7F"
PALE = "#D9D9D9"
ORANGE = "#E69F00"
SKY = "#56B4E9"
BLUE = "#0072B2"
GREEN = "#009E73"
YELLOW = "#F0E442"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"

#: key -> (display label, colour, linestyle, marker)
#: Display labels are what a reader sees.  Repository identifiers
#: (``GEN9_SHAPE_RECOMPOSED``, ``REC_ecfp_plus_recovered``, ...) never reach a figure
#: unless the figure's subject *is* the arm registry.
METHODS: dict[str, tuple[str, str, str, str]] = {
    "measured":     ("Measured",                            BLACK,      "-",   "o"),
    "no_model":     ("Measurements only",                   GREY,       ":",   "v"),
    "no_ligand":    ("No ligand information",               GREY,       "--",  "X"),
    "nn_lookup":    ("Nearest-neighbour lookup",            PURPLE,     "--",  "P"),
    "baseline":     ("Baseline model",                      ORANGE,     "--",  "s"),
    "relpos":       ("+ relative position",                 SKY,        "-.",  "^"),
    "recomposed":   ("+ recomposition",                     BLUE,       "-",   "D"),
    "pipeline":     ("Final pipeline",                      VERMILLION, "-",   "o"),
    "oracle":       ("Oracle (not deployable)",             GREEN,      "--",  "*"),
    "restricted":   ("Restricted training coverage",        PURPLE,     "--",  "s"),
    "full":         ("Full training coverage",              BLUE,       "-",   "D"),
    "rowmatched":   ("Row-count-matched control",           SKY,        "-.",  "^"),
    "shuffled":     ("Shuffled-target control",             GREY,       ":",   "X"),
}


def m(key: str, **override) -> dict:
    """Plot kwargs for a method key."""
    label, colour, ls, marker = METHODS[key]
    out = {"label": label, "color": colour, "linestyle": ls, "marker": marker}
    out.update(override)
    return out


def colour(key: str) -> str:
    return METHODS[key][1]


def label(key: str) -> str:
    return METHODS[key][0]


# --------------------------------------------------------------------------- #
# Canvas widths (inches).  Journal single column 86 mm, 1.5 column 130 mm,
# double column 180 mm.
# --------------------------------------------------------------------------- #
W1 = 3.39
W15 = 5.12
W2 = 7.09

#: Base sizes.  Deliberately a point larger than a minimal "fits the column" choice:
#: the deliverable is judged after the figure is scaled down, and 6 pt tick labels that
#: look fine on screen are unreadable in print.
BASE = 8.0


def apply(base: float = BASE) -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": base,
        "axes.labelsize": base + 0.5,
        "axes.titlesize": base + 0.5,
        "axes.titleweight": "regular",
        "axes.titlelocation": "left",
        "axes.titlepad": 4.0,
        "axes.labelpad": 3.0,
        "xtick.labelsize": base - 0.5,
        "ytick.labelsize": base - 0.5,
        "legend.fontsize": base - 0.5,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.handletextpad": 0.6,
        "legend.labelspacing": 0.4,
        "legend.columnspacing": 1.4,
        "legend.borderaxespad": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "axes.edgecolor": INK,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.3,
        "lines.markersize": 3.6,
        "lines.markeredgewidth": 0.0,
        "grid.linewidth": 0.4,
        "grid.color": "#E6E6E6",
        "axes.grid": False,
        "figure.dpi": 150,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.035,
        "figure.constrained_layout.w_pad": 0.035,
        "figure.constrained_layout.hspace": 0.045,
        "figure.constrained_layout.wspace": 0.045,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",       # so a panel letter or outside legend is never cropped
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "errorbar.capsize": 1.8,
    })


def panel(ax, letter: str, dx: float = -0.02, dy: float = 1.0, size: float = 9.5) -> Text:
    """Manual panel letter in axes coordinates.

    Prefer :func:`add_panel_letters`, which measures the drawn axes and cannot collide
    with a tick label; this variant is for the rare panel that needs hand placement.
    """
    return ax.annotate(letter, xy=(dx, dy), xycoords="axes fraction",
                       xytext=(0, 2), textcoords="offset points",
                       fontsize=size, fontweight="bold", va="bottom", ha="right",
                       annotation_clip=False)


def add_panel_letters(fig, mapping, *, size: float = 9.5, pad_x: float = 0.004,
                      pad_y: float = 0.004) -> list[Text]:
    """Place bold panel letters above-left of each axes' *drawn* extent.

    ``mapping`` is ``{axes: "A", ...}`` or a sequence of axes (lettered A, B, C...).

    The placement is measured after a draw, from each axes' tight bounding box — which
    already includes its tick labels, axis label and title — so a letter cannot land on
    top of a y-tick label however wide that label turns out to be.  This is the single
    most common panel-letter collision and it is worth removing mechanically rather than
    by hand-tuning an offset per figure.

    Call it last, after every axes is fully populated.
    """
    if not isinstance(mapping, dict):
        mapping = {ax: chr(ord("A") + i) for i, ax in enumerate(mapping)}
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    out = []
    for ax, letter in mapping.items():
        bb = ax.get_tightbbox(renderer).transformed(inv)
        x = min(max(bb.x0 - pad_x, 0.002), 0.98)
        y = min(max(bb.y1 + pad_y, 0.02), 0.995)
        out.append(fig.text(x, y, letter, fontsize=size, fontweight="bold",
                            va="bottom", ha="left"))
    return out


def strapline(ax, text: str, size: float = BASE - 0.5) -> Text:
    """A short descriptive line above a panel.  Never a claim, never a sentence."""
    return ax.set_title(text, fontsize=size, color=GREY, loc="left")


# --------------------------------------------------------------------------- #
# Layout linter
# --------------------------------------------------------------------------- #
@dataclass
class Violation:
    kind: str
    detail: str
    a: str = ""
    b: str = ""
    overlap_pts: float = 0.0

    def __str__(self) -> str:
        extra = f"  ({self.overlap_pts:.1f} pt^2)" if self.overlap_pts else ""
        return f"[{self.kind}] {self.detail}{extra}"


@dataclass
class LintReport:
    violations: list[Violation] = field(default_factory=list)
    n_text: int = 0

    @property
    def ok(self) -> bool:
        return not self.violations

    def __str__(self) -> str:
        if self.ok:
            return f"layout OK ({self.n_text} text artists checked)"
        lines = [f"{len(self.violations)} layout violation(s) "
                 f"among {self.n_text} text artists:"]
        lines += [f"  - {v}" for v in self.violations]
        return "\n".join(lines)


def _dead_tick_labels(fig) -> set[int]:
    """Tick labels matplotlib created for locator ticks outside the view limits.

    They are never drawn, but they remain visible ``Text`` artists parked at an arbitrary
    position, and a naive bounding-box sweep reports them as collisions.  Excluding them
    is the difference between a linter a person trusts and one they learn to ignore.
    """
    dead: set[int] = set()
    for ax in fig.axes:
        for axis, interval in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = (min(interval), max(interval))
            span = hi - lo
            tol = span * 1e-6 if span else 1e-9
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                loc = tick.get_loc()
                if loc is None:
                    continue
                if lo - tol <= loc <= hi + tol:
                    continue
                for lbl in (tick.label1, tick.label2):
                    if lbl is not None:
                        dead.add(id(lbl))
    return dead


def _visible_texts(fig) -> list[Text]:
    dead = _dead_tick_labels(fig)
    out: list[Text] = []
    for artist in fig.findobj(Text):
        if id(artist) in dead:
            continue
        if not artist.get_visible():
            continue
        if not str(artist.get_text()).strip():
            continue
        out.append(artist)
    return out


def _owner_axes(artist):
    node = artist
    while node is not None:
        if isinstance(node, mpl.axes.Axes):
            return node
        node = getattr(node, "axes", None) or getattr(node, "figure", None)
        if isinstance(node, mpl.figure.Figure):
            return None
    return None


def _legend_of(artist):
    node = getattr(artist, "get_figure", lambda: None)()
    parent = artist
    for _ in range(6):
        parent = getattr(parent, "_legend", None) or getattr(parent, "figure", None)
    return None


def lint(fig, *, min_overlap_pts: float = 1.5, check_clipping: bool = True,
         max_expansion: float = 1.14, ignore: tuple[str, ...] = ()) -> LintReport:
    """Report text collisions and clipped text.

    Three checks, chosen because each corresponds to a defect a reader would actually
    notice and each is decidable from artist geometry:

    ``text-overlap``
        two visible text artists whose bounding boxes intersect by more than
        ``min_overlap_pts`` square points.  The threshold exists because kerning and
        descenders make adjacent tick labels touch by a fraction of a point routinely.
    ``clipped``
        a text artist with clipping enabled whose box is not contained in the axes that
        clips it — an annotation cut off at the panel edge.
    ``layout-overflow``
        the figure's tight bounding box is more than ``max_expansion`` times the declared
        figure size in either direction.  Saving with ``bbox_inches="tight"`` rescues such
        a figure from cropping, but a large expansion means the declared size is wrong and
        the panels will be smaller than intended once the figure is placed in a column.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [t for t in _visible_texts(fig)
             if str(t.get_text()).strip() not in ignore]

    boxes = []
    for t in texts:
        try:
            bb = t.get_window_extent(renderer=renderer)
        except Exception:
            continue
        if bb.width <= 0 or bb.height <= 0:
            continue
        boxes.append((t, bb))

    report = LintReport(n_text=len(boxes))
    dpi_scale = (fig.dpi / 72.0) ** 2          # px^2 -> pt^2

    # --- text-text overlaps ------------------------------------------------
    for i in range(len(boxes)):
        ta, ba = boxes[i]
        for j in range(i + 1, len(boxes)):
            tb, bb = boxes[j]
            x0, x1 = max(ba.x0, bb.x0), min(ba.x1, bb.x1)
            y0, y1 = max(ba.y0, bb.y0), min(ba.y1, bb.y1)
            if x1 <= x0 or y1 <= y0:
                continue
            area_pts = (x1 - x0) * (y1 - y0) / dpi_scale
            if area_pts < min_overlap_pts:
                continue
            report.violations.append(Violation(
                kind="text-overlap",
                detail=f"{_short(ta)!r} [{_where(ta, fig)}] overlaps "
                       f"{_short(tb)!r} [{_where(tb, fig)}]",
                a=_short(ta), b=_short(tb), overlap_pts=area_pts))

    # --- text clipped by its own axes --------------------------------------
    if check_clipping:
        for t, bb in boxes:
            if not t.get_clip_on():
                continue
            clip = t.get_clip_box()
            if clip is None:
                continue
            if bb.x0 < clip.x0 - 0.5 or bb.x1 > clip.x1 + 0.5 \
                    or bb.y0 < clip.y0 - 0.5 or bb.y1 > clip.y1 + 0.5:
                report.violations.append(Violation(
                    kind="clipped",
                    detail=f"{_short(t)!r} is cut off by its axes"))

    # --- declared size versus drawn size ------------------------------------
    try:
        tight = fig.get_tightbbox(renderer)
        w, h = fig.get_size_inches()
        ratio = max(tight.width / w, tight.height / h)
        if ratio > max_expansion:
            report.violations.append(Violation(
                kind="layout-overflow",
                detail=f"drawn extent is {ratio:.2f}x the declared "
                       f"{w:.2f}x{h:.2f} in figure size"))
    except Exception:
        pass
    return report


def _where(t: Text, fig) -> str:
    """Human-readable location of a text artist: its axes index and role."""
    ax = getattr(t, "axes", None)
    if ax is None:
        for candidate in fig.axes:
            if t in candidate.get_xticklabels() or t in candidate.get_yticklabels() \
                    or t is candidate.xaxis.label or t is candidate.yaxis.label \
                    or t is candidate.title:
                ax = candidate
                break
    role = "text"
    if ax is not None:
        if t is getattr(ax, "title", None):
            role = "title"
        elif t is getattr(ax, "xaxis", None) and False:
            role = "xlabel"
        elif t is ax.xaxis.label:
            role = "xlabel"
        elif t is ax.yaxis.label:
            role = "ylabel"
        elif t in ax.get_xticklabels():
            role = "xtick"
        elif t in ax.get_yticklabels():
            role = "ytick"
        try:
            index = list(fig.axes).index(ax)
            return f"ax{index}:{role}"
        except ValueError:
            return role
    return "figure-level"


def _short(t: Text, n: int = 34) -> str:
    s = " ".join(str(t.get_text()).split())
    return s if len(s) <= n else s[: n - 1] + "…"


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def text_budget(fig, limit: int) -> tuple[int, bool]:
    """Count visible text artists and say whether the figure is inside its budget.

    A figure that explains itself in prose has stopped being a figure.  Counting the
    text artists is crude but it is the only measure that resists good intentions: tick
    labels, axis labels, annotations, legend entries and panel letters all count, so the
    only way under a tight budget is to move explanation into the caption where it
    belongs.  Rough guide: a one-panel results figure lands near 20-30, a three-panel one
    near 45-60. Past that the reader is reading, not looking.
    """
    n = len(_visible_texts(fig))
    return n, n <= limit


def save(fig, out_dir: Path | str, stem: str = "figure", *, strict: bool = True,
         min_overlap_pts: float = 1.5, ignore: tuple[str, ...] = (),
         max_text: int | None = None, close: bool = True) -> LintReport:
    """Lint, then write ``<stem>.png`` (600 dpi) and ``<stem>.pdf`` (vector).

    The PDF and the PNG come from the same figure object with the same bounding box, so
    the vector export cannot drift from the raster one.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = lint(fig, min_overlap_pts=min_overlap_pts, ignore=ignore)
    if max_text is not None:
        n, ok = text_budget(fig, max_text)
        if not ok:
            report.violations.append(Violation(
                kind="text-budget",
                detail=f"{n} text artists, budget {max_text} — move explanation "
                       "into the caption"))
        print(f"  text budget: {n}/{max_text} artists")
    if strict and not report.ok:
        raise AssertionError(f"{stem}: {report}")
    for suffix in (".png", ".pdf"):
        fig.savefig(out_dir / f"{stem}{suffix}")
    (out_dir / f"{stem}.lint.json").write_text(json.dumps(
        {"stem": stem, "n_text_artists": report.n_text, "ok": report.ok,
         "violations": [{"kind": v.kind, "detail": v.detail,
                         "overlap_pts": round(v.overlap_pts, 2)}
                        for v in report.violations]}, indent=1))
    if close:
        plt.close(fig)
    return report
