#!/usr/bin/env python
"""Figure S7 — calibration is series-local: measure inside the series you will predict.

Refined from ``scripts/gen8_figures.py::transfer_heatmap``.  The estimator
(``scripts/gen8_calibration_geography.py::transfer_matrix``) is **not** re-run and none of
the numbers it produced are altered.  What changes is the *presentation*: the predecessor
drew the derived ``gain`` matrix under a directional title, but the post-calibration
transfer quantity that matrix is built from is symmetric to machine precision
(max |M − Mᵀ| = 4.441e-16), so no direction can be read off it.  See ``notes.md``,
section "Scientific integrity".

Run from anywhere (needs a parquet engine for the per-cell intervals):
    python figure_refinement/ml_separator/figure_07_series_local_calibration/figure_script.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
import numpy as np                           # noqa: E402
import pandas as pd                          # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize   # noqa: E402
from matplotlib.lines import Line2D          # noqa: E402
from matplotlib.patches import Patch, Rectangle                    # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common"))
import mlsep_data as D                       # noqa: E402
import pubstyle as PS                        # noqa: E402

CROSS = D.run("gen8_architecture/cross_series")

#: Curve types ordered by how many held-out extractants carry that kind of series, so
#: coverage reads as position as well as as a printed count: the well-supported pairs
#: collect in the upper left of the lower triangle, the thin ones in the lower right.
ORDER = ["metal_series", "acid", "extractant", "none", "contact_time",
         "temperature", "metal_concentration"]
#: Reader-facing names, matching Figure S5 ("metal series", "contact time", ...).
NICE = {a: a.replace("_", " ") for a in ORDER}
#: Two-line forms for the matrix' y axis, so the grid keeps its width.
STACKED = {"metal_series": "metal\nseries", "contact_time": "contact\ntime",
           "metal_concentration": "metal\nconcentration"}

#: Below this many extractants a cell is drawn but not interpreted.  The predecessor
#: filter was ``n_ligands >= 3`` and it blanked everything below it without saying so.
MIN_LIGANDS = 5
#: Robust upper colour limit: 25 of the 28 drawn cells sit below it.  The three that do
#: not rest on one extractant each, and they go to the colour bar's extend arrow instead
#: of setting the scale for everything else.  (The predecessor let a single n = 3 cell,
#: gain = 1.823, set both ends of a diverging scale.)
VMAX = 1.5

CMAP = LinearSegmentedColormap.from_list(
    "mae", ["#FBFDFF", "#DCEBF7", "#A8CFE9", "#6BA6CE", "#2E7BAA", "#0B4C74"]
).with_extremes(over="#04263B", bad=(0, 0, 0, 0))

HELPS, HURTS = PS.GREEN, PS.VERMILLION


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def frozen_matrix() -> pd.DataFrame:
    """The stored aggregate the predecessor plotted, unmodified."""
    return pd.read_csv(CROSS / "transfer_matrix.csv")


def cell_intervals() -> pd.DataFrame:
    """95 % chemotype-blocked interval per cell, resampled over extractants.

    This adds an interval the predecessor lacked; it does **not** re-estimate anything.
    ``transfer_detail.parquet`` is the estimator's own per (seed, extractant) output;
    averaging it over seeds and then over extractants reproduces ``transfer_matrix.csv``
    to 2.2e-16 (asserted in :func:`main`).  Only the resampling of those same per-ligand
    numbers is new.
    """
    detail = pd.read_parquet(CROSS / "transfer_detail.parquet")
    clusters = pd.read_parquet(CROSS / "one_shot_candidate_scores.parquet",
                               columns=["extractant", "tanimoto_cluster"]).drop_duplicates()
    per_ligand = (detail.groupby(["calibration_axis", "target_axis", "extractant"])
                  [["mae", "zero_shot"]].mean().reset_index()
                  .merge(clusters, on="extractant", how="left", validate="many_to_one"))
    assert per_ligand["tanimoto_cluster"].notna().all(), "an extractant has no chemotype"

    rows = []
    for (source, target), block in per_ligand.groupby(["calibration_axis", "target_axis"]):
        point, lo, hi = D.block_bootstrap_mean(block["mae"], block["tanimoto_cluster"])
        z_point, z_lo, z_hi = D.block_bootstrap_mean(block["zero_shot"],
                                                     block["tanimoto_cluster"])
        rows.append({"calibration_axis": source, "target_axis": target,
                     "mae_point": point, "mae_lo": lo, "mae_hi": hi,
                     "zero_point": z_point, "zero_lo": z_lo, "zero_hi": z_hi,
                     "n_ligands": int(block["extractant"].nunique()),
                     "n_chemotypes": int(block["tanimoto_cluster"].nunique())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Panel A — the symmetric transfer matrix
# --------------------------------------------------------------------------- #
def panel_matrix(ax, grid: pd.DataFrame, counts: pd.DataFrame, asymmetry: float):
    n = len(ORDER)
    values = grid.to_numpy(float)
    shown = np.where(np.tril(np.ones((n, n), bool)), values, np.nan)
    image = ax.imshow(np.ma.masked_invalid(shown), cmap=CMAP, aspect="auto",
                      norm=Normalize(0.0, VMAX), interpolation="nearest")

    norm = Normalize(0.0, VMAX)
    for i in range(n):
        for j in range(i + 1):
            value = values[i, j]
            k = int(counts.iat[i, j])
            thin = k < MIN_LIGANDS
            ink = "white" if norm(min(value, VMAX)) > 0.62 else PS.INK
            if thin:
                # A distinct "too few extractants" texture over the fill.  Unlike the
                # predecessor's blank white cell it cannot be read as a mid-scale value,
                # and the count underneath says how thin the cell actually is.
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="none",
                                       hatch="/////", edgecolor=ink, linewidth=0.0,
                                       alpha=0.5, zorder=2))
            ax.text(j, i - 0.06, f"({value:.2f})" if thin else f"{value:.2f}",
                    ha="center", va="center", fontsize=PS.BASE - 1.0, zorder=3,
                    color=ink, alpha=0.8 if thin else 1.0)
            ax.text(j, i + 0.30, f"n={k}", ha="center", va="center",
                    fontsize=PS.BASE - 2.0, zorder=3, color=ink, alpha=0.75)
    for i in range(n):
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, facecolor="none",
                               edgecolor=PS.INK, linewidth=1.4, zorder=4))

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    # Vertical column labels: at this cell pitch any rotation shallower than 90 deg puts
    # "metal concentration" through its neighbours, which is the collision the
    # predecessor shipped.
    ax.set_xticklabels([STACKED.get(a, NICE[a]) for a in ORDER], rotation=90,
                       ha="center", va="top", linespacing=0.95)
    ax.set_yticklabels([STACKED.get(a, NICE[a]) for a in ORDER], linespacing=0.95)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="both", length=0, pad=2.0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)

    ax.legend(handles=[
        Patch(facecolor="none", edgecolor=PS.INK, linewidth=1.4,
              label="measured and scored in one series"),
        Patch(facecolor="#DCEBF7", edgecolor=PS.GREY, linewidth=0.4, hatch="/////",
              label=f"fewer than {MIN_LIGANDS} extractants"),
    ], loc="upper right", bbox_to_anchor=(1.015, 0.845), fontsize=PS.BASE - 1.6,
        handlelength=1.2, handleheight=1.2, labelspacing=0.45, borderaxespad=0.0)
    # Key and note live in the empty upper triangle, which is widest at the top; both are
    # right-aligned so no line of either can reach a drawn cell.
    ax.text(1.0, 0.995,
            "Symmetric by construction:\n"
            "(A, B) and (B, A) average\n"
            "the same block of row pairs,\n"
            f"max $|M-M^\\mathsf{{T}}|$ = {asymmetry:.1e}.",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=PS.BASE - 1.6, color=PS.GREY, linespacing=1.4)
    ax.set_xlabel("curve type\n(“none” = rows belonging to no titration curve)",
                  fontsize=PS.BASE - 0.5, color=PS.GREY, linespacing=1.4)
    ax.set_ylabel("curve type", fontsize=PS.BASE - 0.5, color=PS.GREY)
    return image


# --------------------------------------------------------------------------- #
# Panels B and C — did that measurement beat no measurement at all?
# --------------------------------------------------------------------------- #
#: Right of ``CLIP`` the panel is a numeric column, not data.  Confidence intervals that
#: run past it are drawn to the edge and marked with a triangle rather than being silently
#: cut off, and their true bounds are in ``values.json``.
XMAX, CLIP, DELTA_X = 2.62, 2.10, 2.58


def _interval(ax, lo, hi, y, colour, width):
    ax.plot([lo, min(hi, CLIP)], [y, y], color=colour, linewidth=width, zorder=1)
    if hi > CLIP:
        ax.plot([CLIP], [y], marker=">", markersize=2.8, color=colour, zorder=1)


def dumbbells(ax, rows: list[dict], pad_top: float = 0.0, pad_bottom: float = 0.0):
    """One row per cell: the calibrated error, and the baseline(s) it is judged against.

    ``pad_top``/``pad_bottom`` add blank rows that hold the key and the notes, so neither
    is ever drawn over a data row.
    """
    ax.set_xlim(0.0, XMAX)
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["label"] for r in rows])
    for tick, r in zip(ax.get_yticklabels(), rows):
        if r["n"] < MIN_LIGANDS:
            tick.set_color(PS.GREY)
    ax.tick_params(axis="y", length=0, pad=2.0)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", which="major", color="#EDEDED", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    for y, r in enumerate(rows):
        faint = 0.55 if r["n"] < MIN_LIGANDS else 1.0
        spread = 0.23 if len(r["baselines"]) > 1 else 0.0
        for s, (base, blo, bhi) in zip((-1, 1), r["baselines"]):
            yb = y + s * spread
            _interval(ax, blo, bhi, yb, PS.PALE, 0.7)
            gain = base - r["mae"]
            tone = HELPS if gain > 0 else HURTS
            ax.plot([base, r["mae"]], [yb, y], color=tone,
                    linewidth=1.1, alpha=0.9 * faint, zorder=2)
            ax.plot([base], [yb], marker="s", markersize=3.3, markerfacecolor="white",
                    markeredgecolor=PS.INK, markeredgewidth=0.7, alpha=faint, zorder=4)
            if len(r["baselines"]) == 1:
                # The signed change, printed because several of these gaps are only a
                # few hundredths wide -- exactly the ones the claim turns on.  It is the
                # ``gain`` column of transfer_matrix.csv, unchanged.
                ax.text(DELTA_X, y, f"{gain:+.2f}".replace("-", "\u2212"),
                        ha="right", va="center", fontsize=PS.BASE - 2.0,
                        color=tone, alpha=faint, zorder=6)
        _interval(ax, r["lo"], r["hi"], y, PS.GREY, 0.9)
        ax.plot([r["mae"]], [y], marker="o", markersize=4.2, color=PS.INK,
                alpha=faint, zorder=5)
        if len(r["baselines"]) > 1:
            # One calibrated error, two baselines -> two published "gains".  Their
            # difference is exactly the difference between the two baselines, and it is
            # the entire apparent direction of the predecessor's matrix.
            split = abs(r["baselines"][0][0] - r["baselines"][1][0])
            ax.text(DELTA_X, y, f"{split:.2f}", ha="right", va="center",
                    fontsize=PS.BASE - 2.0, color=PS.INK, alpha=faint, zorder=6)
    ax.set_ylim(len(rows) - 0.5 + pad_bottom, -0.5 - pad_top)


def within_rows(look) -> list[dict]:
    rows = []
    for a in ORDER:
        c = look.loc[(a, a)]
        rows.append({"label": f"{NICE[a]}  (n={int(c.n_ligands)})", "n": int(c.n_ligands),
                     "chemotypes": int(c.n_chemotypes), "series": a, "mae": c.mae_point,
                     "lo": c.mae_lo, "hi": c.mae_hi,
                     "baselines": [(c.zero_point, c.zero_lo, c.zero_hi)]})
    return rows


def cross_rows(look) -> list[dict]:
    rows = []
    for i, a in enumerate(ORDER):
        for b in ORDER[i + 1:]:
            ab, ba = look.loc[(a, b)], look.loc[(b, a)]
            if int(ab.n_ligands) < MIN_LIGANDS:
                continue
            rows.append({"label": f"{NICE[a]} – {NICE[b]}  (n={int(ab.n_ligands)})",
                         "n": int(ab.n_ligands), "chemotypes": int(ab.n_chemotypes),
                         "pair": (a, b),
                         "mae": ab.mae_point, "lo": ab.mae_lo, "hi": ab.mae_hi,
                         # ``ba`` scores rows of ``a``: drawn first, i.e. upper, so the
                         # top marker always belongs to the first-named series.
                         "baselines": [(ba.zero_point, ba.zero_lo, ba.zero_hi),
                                       (ab.zero_point, ab.zero_lo, ab.zero_hi)]})
    return sorted(rows, key=lambda r: max(b[0] for b in r["baselines"]))


# --------------------------------------------------------------------------- #
def main() -> int:
    PS.apply()
    frozen = frozen_matrix()
    grid = frozen.pivot_table(index="calibration_axis", columns="target_axis",
                              values="mae").reindex(index=ORDER, columns=ORDER)
    counts = frozen.pivot_table(index="calibration_axis", columns="target_axis",
                                values="n_ligands").reindex(index=ORDER, columns=ORDER)
    zero = frozen.pivot_table(index="calibration_axis", columns="target_axis",
                              values="zero_shot").reindex(index=ORDER, columns=ORDER)
    asym = {name: float(np.nanmax(np.abs(t.to_numpy(float) - t.to_numpy(float).T)))
            for name, t in (("mae", grid), ("zero_shot", zero), ("n_ligands", counts))}

    ci = cell_intervals()
    check = frozen.merge(ci, on=["calibration_axis", "target_axis"])
    drift = float(max(np.abs(check.mae - check.mae_point).max(),
                      np.abs(check.zero_shot - check.zero_point).max()))
    assert drift < 1e-12, f"interval table drifted from the frozen matrix by {drift}"
    look = ci.set_index(["calibration_axis", "target_axis"])
    within, cross = within_rows(look), cross_rows(look)

    pad_b, pad_ct, pad_cb = 2.9, 2.0, 4.0   # blank rows for the key, headers and notes
    fig = plt.figure(figsize=(PS.W2, 5.3))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.32, 1.00],
                          height_ratios=[len(within) + pad_b,
                                         len(cross) + pad_ct + pad_cb],
                          hspace=0.10)
    axA = fig.add_subplot(gs[:, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 1], sharex=axB)

    image = panel_matrix(axA, grid, counts, asym["mae"])
    PS.strapline(axA, "one series calibrating another")
    # The colour bar goes inside the empty upper triangle: it is the only large blank
    # region on the page, and putting it there keeps the cells close to square.
    cax = axA.inset_axes([0.455, 0.660, 0.520, 0.030])
    bar = fig.colorbar(image, cax=cax, orientation="horizontal", extend="max")
    bar.set_label("MAE after the measurement\n(log$_{10}$ $D$ units; lower is better)",
                  fontsize=PS.BASE - 1.6, linespacing=1.3)
    bar.ax.tick_params(labelsize=PS.BASE - 2.0, length=2.0, pad=1.5)
    bar.outline.set_linewidth(0.5)

    dumbbells(axB, within, pad_top=pad_b)
    dumbbells(axC, cross, pad_top=pad_ct, pad_bottom=pad_cb)
    PS.strapline(axB, "measured in the series it scores")
    PS.strapline(axC, "measured in a different series (n $\\geq$ 5)")
    axC.set_xlabel("MAE on the scored rows\n(log$_{10}$ $D$ units)")
    plt.setp(axB.get_xticklabels(), visible=False)
    axB.tick_params(axis="x", length=0)
    for ax in (axB, axC):        # the strip right of CLIP is a numeric column, not data
        ax.spines["bottom"].set_bounds(0.0, CLIP)

    axB.legend(handles=[
        Line2D([], [], linestyle="none", marker="s", markersize=3.3,
               markerfacecolor="white", markeredgecolor=PS.INK, markeredgewidth=0.7,
               label="no measurement"),
        Line2D([], [], linestyle="none", marker="o", markersize=4.2, color=PS.INK,
               label="after the measurement"),
        Line2D([], [], color=PS.GREY, linewidth=0.9,
               label="95 % CI (chemotype blocks)"),
        Line2D([], [], linestyle="none", marker=">", markersize=2.8, color=PS.GREY,
               label="interval runs off the axis"),
    ], loc="upper left", bbox_to_anchor=(-0.005, 1.02), fontsize=PS.BASE - 2.0,
        labelspacing=0.32, handlelength=1.0, handletextpad=0.4, borderaxespad=0.0)
    axB.text(DELTA_X, -pad_b + 0.55, "MAE removed,\n+ = error fell",
             ha="right", va="center", fontsize=PS.BASE - 2.0, color=PS.INK,
             linespacing=1.3)
    axC.text(DELTA_X, -pad_ct + 0.75, "the two “gains”\ndiffer by",
             ha="right", va="center", fontsize=PS.BASE - 2.0, color=PS.INK,
             linespacing=1.3)
    axC.text(0.995, 0.02,
             "One calibrated error but two baselines (the upper marker\n"
             "scores the first-named series), so the pair carries two\n"
             "published “gains”. They differ by exactly the gap between the\n"
             "baselines — the whole of the published matrix's “direction”.",
             transform=axC.transAxes, ha="right", va="bottom",
             fontsize=PS.BASE - 2.0, color=PS.GREY, linespacing=1.4)

    PS.add_panel_letters(fig, [axA, axB, axC])
    report = PS.save(fig, HERE, "figure", strict=False)
    print(report)

    values = {
        "figure": "figure_07_series_local_calibration",
        "source_csv": str(CROSS / "transfer_matrix.csv"),
        "estimator": "scripts/gen8_calibration_geography.py::transfer_matrix (not re-run)",
        "symmetry_check_max_abs_M_minus_MT": asym,
        "coverage": {"min_ligands_interpreted": MIN_LIGANDS, "predecessor_filter": 3,
                     "cells_drawn": 28,
                     "cells_interpreted": int((np.tril(counts.to_numpy(float))
                                               >= MIN_LIGANDS).sum())},
        "colour_limit": {"vmin": 0.0, "vmax": VMAX, "extend": "max",
                         "cells_above_vmax": int((np.tril(grid.to_numpy(float))
                                                  > VMAX).sum())},
        "panelA_cells": [
            {"series_a": ORDER[i], "series_b": ORDER[j],
             "transfer_mae": float(grid.iat[i, j]), "n_ligands": int(counts.iat[i, j]),
             "interpreted": bool(counts.iat[i, j] >= MIN_LIGANDS),
             "within_series": i == j}
            for i in range(len(ORDER)) for j in range(i + 1)],
        "panelB_within_series": [
            {"series": r["series"], "n_ligands": r["n"],
             "n_chemotypes": r["chemotypes"], "mae_after": r["mae"],
             "mae_after_ci": [r["lo"], r["hi"]], "mae_zero_shot": r["baselines"][0][0],
             "mae_zero_shot_ci": list(r["baselines"][0][1:]),
             "gain": r["baselines"][0][0] - r["mae"]} for r in within],
        "panelC_cross_series": [
            {"series_a": r["pair"][0], "series_b": r["pair"][1], "n_ligands": r["n"],
             "n_chemotypes": r["chemotypes"],
             "mae_after": r["mae"], "mae_after_ci": [r["lo"], r["hi"]],
             "mae_zero_shot_on_a": r["baselines"][0][0],
             "mae_zero_shot_on_b": r["baselines"][1][0],
             "gain_scoring_a": r["baselines"][0][0] - r["mae"],
             "gain_scoring_b": r["baselines"][1][0] - r["mae"],
             "gain_difference": abs(r["baselines"][0][0] - r["baselines"][1][0])}
            for r in cross],
        "lint": {"ok": report.ok, "violations": [str(v) for v in report.violations]},
    }
    (HERE / "values.json").write_text(json.dumps(values, indent=1))

    print(grid.round(3).to_string())
    print("\nwithin-series (MAE after / no measurement / n):")
    for r in within:
        flag = "  falls" if r["mae"] < r["baselines"][0][0] else "  RISES"
        print(f"  {NICE[r['series']]:20s} {r['mae']:.3f}  {r['baselines'][0][0]:.3f}"
              f"  n={r['n']:3d}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
