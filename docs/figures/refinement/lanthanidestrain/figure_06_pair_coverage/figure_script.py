#!/usr/bin/env python
"""Figure 6 — which comparisons the adjacent-pair score is an average over.

Refined from ``automl/figures_pi_email.py::fig_coverage`` (``email_figA_coverage.png``).
The numbers are unchanged: panel A is ``pair_coverage.csv`` exactly as the original drew
it, panel B is the per-position pair count from ``adjacent_decomposition_position.csv``.
This pass changes layout, typography, labelling and information design only.

Run from anywhere:
    python figure_refinement/lanthanidestrain/figure_06_pair_coverage/figure_script.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe      # noqa: E402
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
from matplotlib.lines import Line2D      # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common"))
import lstrain_data as LD                # noqa: E402
import pubstyle as PS                    # noqa: E402

#: Colour carries the quantity, not the panel: blue is a measurement, vermillion is a
#: scored adjacent pair, in both panels.
C_ROW = PS.BLUE
C_PAIR = PS.VERMILLION

BAND = "#F4F4F4"

#: Reserved x range of panel A.  Data live left of ``DIVIDER``; the count table lives
#: right of it, so no number can ever be read as a bar length.
X_DATA_MAX = 29.5
DIVIDER = 31.0
COL_ROWS = 44.0          # right edge of the "measurements" column
COL_PAIRS = 52.0         # right edge of the "scored-pair slots" column
X_MAX = 52.0

#: Panel B reserves its right-hand third for the pair-name list, so a name can never
#: be mistaken for part of a bar.
X_NAME = 126.0
X_B_MAX = 152.0

Y_HEAD = 0.30            # column-header strip, above La
Y_TOTAL = 16.20          # totals row, below Lu
Y_SPREAD = 17.15         # most-to-least row
Y_TOP, Y_BOT = 0.02, 17.75

#: Nothing on this figure goes below 7 pt.  It is reproduced at 180 mm, i.e. at 1:1,
#: so a point size on screen is the point size in print.
TINY = PS.BASE - 1.0     # 7.0 pt — count table and in-panel notes
SMALL = PS.BASE - 0.5    # 7.5 pt — value labels


def load() -> tuple:
    """The two vendored tables, plus every consistency check that ties them."""
    cov = LD.read("pair_coverage").sort_values("lanthanide_index").reset_index(drop=True)
    cov["lanthanide_index"] = cov["lanthanide_index"].astype(int)
    pos = LD.read("adjacent_decomposition_position")

    idx_of = dict(zip(cov["metal"], cov["lanthanide_index"]))
    lo, hi = zip(*(p.split("-") for p in pos["pair"]))
    pos = pos.assign(i_lo=[idx_of[s] for s in lo], i_hi=[idx_of[s] for s in hi])
    assert (pos["i_hi"] - pos["i_lo"] == 1).all(), "a position is not a neighbour pair"
    pos["y"] = (pos["i_lo"] + pos["i_hi"]) / 2.0
    pos = pos.sort_values("y").reset_index(drop=True)

    n_rows = int(cov["rows"].sum())
    n_slots = int(cov["pairs_binned"].sum())
    n_pairs = int(pos["n_pairs"].sum())

    # The two tables were written by different modules from the same enumeration.
    # Each metal's slot count must be the sum of the positions it touches, and every
    # pair must be counted at both of its ends.
    assert n_slots == 2 * n_pairs, (n_slots, n_pairs)
    for _, r in cov.iterrows():
        touching = pos[(pos.i_lo == r.lanthanide_index) | (pos.i_hi == r.lanthanide_index)]
        assert int(touching["n_pairs"].sum()) == int(r["pairs_binned"]), r["metal"]
    assert np.allclose(cov["rows"] / n_rows, cov["row_share"])
    assert np.allclose(cov["pairs_binned"] / n_slots, cov["pair_share_binned"])
    print(f"consistency OK · {n_rows} measurements · {n_pairs} adjacent pairs "
          f"· {n_slots} pair slots · {len(pos)} neighbour positions")
    return cov, pos, n_rows, n_slots, n_pairs


def main() -> int:
    PS.apply()
    cov, pos, n_rows, n_slots, n_pairs = load()

    y = cov["lanthanide_index"].to_numpy(float)
    row_pct = 100 * cov["row_share"].to_numpy()
    pair_pct = 100 * cov["pair_share_binned"].to_numpy()
    gap = int(set(range(int(y.min()), int(y.max()) + 1)).difference(y.astype(int)).pop())

    even_pct = 100.0 / len(cov)
    even_pairs = n_pairs / len(pos)
    spread_rows = cov["rows"].max() / cov["rows"].min()
    spread_slots = cov["pairs_binned"].max() / cov["pairs_binned"].min()
    spread_pos = pos["n_pairs"].max() / pos["n_pairs"].min()

    fig = plt.figure(figsize=(PS.W2, 4.55))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.34, 1.00])
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1], sharey=axA)

    # Row bands run across both panels so a lanthanide named once in A can be carried
    # into B without counting rows.  The absent element keeps an unbanded slot.
    for ax in (axA, axB):
        for i in range(int(y.min()), int(y.max()) + 1):
            if i != gap and i % 2 == 0:
                ax.axhspan(i - 0.5, i + 0.5, color=BAND, zorder=0, lw=0)

    # ------------------------------------------------------------------ A -----
    # The reference line stops at the first and last row, so it neither runs through
    # its own label nor down into the count table.
    axA.vlines(even_pct, 0.80, 15.50, color=PS.GREY, lw=0.7, ls=(0, (4, 3)), zorder=1)
    axA.hlines(y, np.minimum(row_pct, pair_pct), np.maximum(row_pct, pair_pct),
               color=PS.GREY, lw=1.1, alpha=0.55, zorder=2)
    # The diamond is drawn first and a little larger: Sm's two shares differ by 0.13
    # points, so the markers coincide and the smaller one must sit on top to be seen.
    axA.plot(pair_pct, y, "D", color=C_PAIR, ms=5.2, zorder=3, ls="none",
             markeredgecolor="white", markeredgewidth=0.6)
    axA.plot(row_pct, y, "o", color=C_ROW, ms=4.2, zorder=4, ls="none",
             markeredgecolor="white", markeredgewidth=0.6)

    # Offset in points, not data units, so the dashed reference line cannot strike
    # through its own label at any axis scale.
    axA.annotate("1/14", (even_pct, 0.42), textcoords="offset points", xytext=(4, 0),
                 color=PS.GREY, fontsize=TINY, ha="left", va="center")

    # count table -------------------------------------------------------------
    axA.plot([DIVIDER - 0.6, DIVIDER - 0.6], [Y_HEAD - 0.15, Y_SPREAD + 0.35],
             color=PS.PALE, lw=0.7, zorder=1)
    axA.text(COL_ROWS, Y_HEAD, "measurements", color=C_ROW, fontsize=TINY,
             ha="right", va="center")
    axA.text(COL_PAIRS, Y_HEAD, "pair slots", color=C_PAIR, fontsize=TINY,
             ha="right", va="center")
    for yi, nr, npair in zip(y, cov["rows"], cov["pairs_binned"]):
        axA.text(COL_ROWS, yi, f"{int(nr):,}", fontsize=TINY, ha="right", va="center")
        axA.text(COL_PAIRS, yi, f"{int(npair):,}", fontsize=TINY, ha="right", va="center")
    axA.plot([DIVIDER + 0.2, COL_PAIRS], [15.72, 15.72], color=PS.PALE, lw=0.7, zorder=1)
    axA.text(DIVIDER + 0.4, Y_TOTAL, "total", fontsize=TINY, color=PS.GREY,
             ha="left", va="center")
    axA.text(COL_ROWS, Y_TOTAL, f"{n_rows:,}", fontsize=TINY, ha="right", va="center")
    axA.text(COL_PAIRS, Y_TOTAL, f"{n_slots:,}", fontsize=TINY, ha="right", va="center")
    axA.text(DIVIDER + 0.4, Y_SPREAD, "most ÷ least", fontsize=TINY, color=PS.GREY,
             ha="left", va="center")
    axA.text(COL_ROWS, Y_SPREAD, f"{spread_rows:.1f}×", fontsize=TINY,
             ha="right", va="center")
    axA.text(COL_PAIRS, Y_SPREAD, f"{spread_slots:.1f}×", fontsize=TINY,
             ha="right", va="center")

    axA.text(0.9, gap, "Pm absent — Nd and Sm each have one neighbour",
             fontsize=TINY, color=PS.GREY, ha="left", va="center")
    # Why the two distributions differ at all, in the space the count table leaves free.
    axA.text(0.0, (Y_TOTAL + Y_SPREAD) / 2,
             "a pair is scored only when both neighbours\n"
             "were measured in the same extractant ×\n"
             "conditions block; replicates averaged first",
             fontsize=TINY, color=PS.GREY, ha="left", va="center", linespacing=1.35)

    axA.set_xlim(0, X_MAX)
    axA.set_xticks([0, 5, 10, 15, 20, 25])
    axA.set_yticks(y)
    axA.set_yticklabels(cov["metal"])
    axA.set_ylim(Y_BOT, Y_TOP)
    axA.set_xlabel("share of the total (%)", loc="left")
    axA.spines["bottom"].set_bounds(0, X_DATA_MAX)
    axA.spines["left"].set_bounds(0.5, 15.5)
    axA.tick_params(axis="y", length=0)
    PS.strapline(axA, "per lanthanide — La to Lu, ionic radius decreasing downward")

    # ------------------------------------------------------------------ B -----
    axB.vlines(even_pairs, 0.80, 15.50, color=PS.GREY, lw=0.7, ls=(0, (4, 3)), zorder=1)
    axB.barh(pos["y"], pos["n_pairs"], height=0.56, color=C_PAIR, lw=0, zorder=3)
    for yi, n, name in zip(pos["y"], pos["n_pairs"], pos["pair"]):
        # A halo, because one bar ends within a glyph's width of the reference line.
        axB.text(n + 3.0, yi, f"{int(n)}", fontsize=SMALL, ha="left", va="center",
                 zorder=5, path_effects=[pe.withStroke(linewidth=1.8,
                                                       foreground="white")])
        axB.text(X_NAME, yi, name.replace("-", "–"), fontsize=SMALL,
                 color=PS.GREY, ha="left", va="center")
    axB.annotate("1/12", (even_pairs, 0.42), textcoords="offset points", xytext=(4, 0),
                 color=PS.GREY, fontsize=TINY, ha="left", va="center")
    axB.text(3.0, gap, "no pair spans the gap", fontsize=TINY, color=PS.GREY,
             ha="left", va="center")
    axB.text(X_B_MAX - 3.0, Y_TOTAL, f"{n_pairs} pairs in total", fontsize=TINY,
             color=PS.GREY, ha="right", va="center")
    axB.text(X_B_MAX - 3.0, Y_SPREAD, f"most ÷ least  {spread_pos:.1f}×", fontsize=TINY,
             color=PS.GREY, ha="right", va="center")

    axB.set_xlim(0, X_B_MAX)
    axB.set_xticks([0, 25, 50, 75, 100])
    axB.set_xlabel("adjacent pairs scored (count)", loc="left")
    axB.spines["bottom"].set_bounds(0, 110)
    axB.spines["left"].set_bounds(0.5, 15.5)
    axB.tick_params(axis="y", length=0, labelleft=False)
    PS.strapline(axB, "per neighbour pair — each bar spans the two it compares")

    # ------------------------------------------------------------- legend -----
    handles = [
        Line2D([], [], color=C_ROW, marker="o", ms=4.2, ls="none",
               markeredgecolor="white", markeredgewidth=0.6,
               label=f"measurements  (100 % = {n_rows:,})"),
        Line2D([], [], color=C_PAIR, marker="D", ms=5.2, ls="none",
               markeredgecolor="white", markeredgewidth=0.6,
               label=f"scored-pair slots  (100 % = {n_slots:,} = 2 × {n_pairs} pairs)"),
        Line2D([], [], color=PS.GREY, lw=0.7, ls=(0, (4, 3)),
               label="even coverage"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3)

    PS.add_panel_letters(fig, [axA, axB])

    # No column name, block key or file stem may reach a reader.  Asserted rather than
    # reviewed, so a later edit cannot quietly reintroduce one.
    fig.canvas.draw()
    drawn = " ".join(str(t.get_text()) for t in fig.findobj(matplotlib.text.Text))
    for token in ("row_share", "pair_share_binned", "pairs_binned", "pairs_strict",
                  "lanthanide_index", "metal_symbol", "composition_key",
                  "extractant_group", "geometry_ok", "has_3d", "n_pairs",
                  "pair_coverage", "adjacent_decomposition"):
        assert token not in drawn, f"repository identifier {token!r} is in the artwork"

    report = PS.save(fig, HERE, "figure", strict=False)
    print(report)

    values = {
        "figure": "figure_06_pair_coverage",
        "source": {
            "repository": "lanthanidestrain",
            "original_script": "automl/figures_pi_email.py::fig_coverage",
            "original_render": "automl/reports/figures/email_figA_coverage.png",
            "tables": ["pair_coverage.csv", "adjacent_decomposition_position.csv"],
            "provenance": LD.provenance(),
        },
        "cohort": {
            "measurements": n_rows,
            "lanthanides": int(len(cov)),
            "adjacent_pairs": n_pairs,
            "pair_slots": n_slots,
            "neighbour_positions": int(len(pos)),
            "absent_lanthanide_index": gap,
            "block_key": "extractant x binned conditions (composition_key)",
        },
        "even_reference": {"share_pct": even_pct, "pairs_per_position": even_pairs},
        "spread_max_over_min": {
            "measurements": float(spread_rows),
            "pair_slots": float(spread_slots),
            "positions": float(spread_pos),
        },
        "panelA": [
            {"lanthanide": m, "index": int(i), "measurements": int(nr),
             "pair_slots": int(ns), "measurement_share_pct": float(rp),
             "pair_slot_share_pct": float(pp)}
            for m, i, nr, ns, rp, pp in zip(
                cov["metal"], cov["lanthanide_index"], cov["rows"],
                cov["pairs_binned"], row_pct, pair_pct)
        ],
        "panelB": [
            {"pair": p.replace("-", "–"), "position": float(yy), "pairs": int(n)}
            for p, yy, n in zip(pos["pair"], pos["y"], pos["n_pairs"])
        ],
        "lint": {"ok": report.ok, "violations": [str(v) for v in report.violations]},
    }
    (HERE / "values.json").write_text(json.dumps(values, indent=1))
    print(f"Eu {row_pct[list(cov['metal']).index('Eu')]:.1f} % of measurements -> "
          f"{pair_pct[list(cov['metal']).index('Eu')]:.1f} % of pair slots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
