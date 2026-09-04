#!/usr/bin/env python
"""Figure 3 — amplitude compression on held-out titrations, and its repair.

Refined from ``figures/scripts/plot_fig3_shape_recovery.py``.  Every number, cohort,
seed, statistic and interval is the one that script produced and that
``figures/METRIC_AUDIT.md`` verified; this pass changes layout, typography, labelling
and information design only.

Run from anywhere:
    python figure_refinement/ml_separator/figure_03_shape_recovery/figure_script.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe    # noqa: E402
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402
import pandas as pd                    # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common"))
import mlsep_data as D                 # noqa: E402
import pubstyle as PS                  # noqa: E402

AXIS = "extractant"
EXAMPLE_SEED = 104729          # first split seed, fixed before selection
MIN_POINTS = 5                 # declared in the selection rule (FIGURE_PLAN.md §9)

#: (pubstyle key, repository arm id).  The repository ids stay in this table and in
#: values.json; the artwork only ever shows the pubstyle display labels.
STAGES: list[tuple[str, str]] = [
    ("baseline", "REC_ecfp_plus_recovered"),
    ("relpos", "GEN9_REL_MONOLITH"),
    ("recomposed", "GEN9_SHAPE_RECOMPOSED"),
]
OOF = {"baseline": "gen9_shape/phase1/oof_REC_ecfp_plus_recovered.parquet",
       "relpos": "gen9_shape/relmono/oof_GEN9_REL_MONOLITH.parquet",
       "recomposed": "gen9_shape/recomposed/oof_GEN9_SHAPE_RECOMPOSED.parquet"}

#: The three pre-declared roles of the example curves (FIGURE_PLAN.md §9).
ROLES = ["median", "improved", "residual"]
ROLE_TITLE = {"median": "median case", "improved": "large repair",
              "residual": "repair fails"}

#: Row order shared by D, E and F, read top to bottom: the measurement first, then the
#: three model stages in the order they are built.  Row 0 is the reference row — in D it
#: is the measured slope distribution, in E and F it is the value the reference line
#: marks, so the same physical row means "measured" in all three panels.
ROW = {"measured": 0, "baseline": 1, "relpos": 2, "recomposed": 3}
ROW_TOP, ROW_BOTTOM = -0.80, 3.72      # inverted y limits shared by D, E and F
MINUS = "−"

#: A thin white halo, used on the one annotation that has to cross a reference line.
#: Value labels do not get it: a halo drawn over the tinted reference row leaves a
#: visible white smear, and the labels are placed clear of every line instead.
HALO = [pe.withStroke(linewidth=2.2, foreground="white")]


def num(value: float, nd: int = 2, signed: bool = False) -> str:
    """Two decimals with a typographic minus, never a raw float."""
    text = f"{value:+.{nd}f}" if signed else f"{value:.{nd}f}"
    return text.replace("-", MINUS)


def num_small(value: float, signed: bool = True) -> str:
    """As :func:`num`, but keep a third decimal where two would read as zero."""
    return num(value, 2 if abs(value) >= 0.05 else 3, signed=signed)


# --------------------------------------------------------------------------- #
# Example selection — the pre-declared rule, copied unchanged
# --------------------------------------------------------------------------- #
def select_examples(cs: pd.DataFrame) -> tuple[dict[str, str], pd.DataFrame]:
    seed = cs[(cs.split_seed == EXAMPLE_SEED) & (cs.n_points >= MIN_POINTS)]
    wide = seed.pivot_table(index=["curve_id", "extractant", "n_points"], columns="model",
                            values="shape_mae")
    wide = wide[["REC_ecfp_plus_recovered", "GEN9_SHAPE_RECOMPOSED"]].dropna()
    wide.columns = ["shape_frozen", "shape_recomposed"]
    wide["delta"] = wide.shape_frozen - wide.shape_recomposed
    wide = wide.reset_index().sort_values("curve_id").reset_index(drop=True)

    def iqr(s: pd.Series) -> float:
        return max(1e-9, float(s.quantile(0.75) - s.quantile(0.25)))

    zx = (wide.shape_frozen - wide.shape_frozen.median()) / iqr(wide.shape_frozen)
    zy = (wide.delta - wide.delta.median()) / iqr(wide.delta)
    wide["dist_to_joint_median"] = np.hypot(zx, zy)

    top = wide[wide.delta >= wide.delta.quantile(0.90)]
    bottom = wide[wide.delta <= wide.delta.quantile(0.10)]
    wide["dist_to_top_decile_median"] = np.where(
        wide.curve_id.isin(top.curve_id), (wide.delta - top.delta.median()).abs(), np.nan)
    wide["dist_to_bottom_decile_median"] = np.where(
        wide.curve_id.isin(bottom.curve_id),
        (wide.delta - bottom.delta.median()).abs(), np.nan)

    chosen = {
        "median": wide.sort_values(["dist_to_joint_median", "curve_id"]).curve_id.iloc[0],
        "improved": wide.sort_values(["dist_to_top_decile_median", "curve_id"]).curve_id.iloc[0],
        "residual": wide.sort_values(["dist_to_bottom_decile_median", "curve_id"]).curve_id.iloc[0],
    }
    wide["selected_as"] = wide.curve_id.map({v: k for k, v in chosen.items()}).fillna("")
    return chosen, wide


def example_rows(curve_ids: list[str]) -> pd.DataFrame:
    mem = pd.read_parquet(D.run("gen9_shape/curves/curve_membership.parquet"))
    mem = mem[(mem.axis_label == AXIS) & (mem.curve_id.isin(curve_ids))]
    out = mem[["curve_id", "row_id", "axis_value"]].copy()
    for key, path in OOF.items():
        oof = pd.read_parquet(D.run(path),
                              columns=["row_id", "split_seed", "log_D", "prediction"])
        oof = oof[oof.split_seed == EXAMPLE_SEED][["row_id", "log_D", "prediction"]]
        oof = oof.rename(columns={"prediction": f"pred_{key}"})
        out = out.merge(oof.drop_duplicates("row_id"), on="row_id", how="left")
        if "log_D_y" in out.columns:
            out = out.drop(columns=["log_D_y"]).rename(columns={"log_D_x": "log_D"})
    return out.sort_values(["curve_id", "axis_value"])


# --------------------------------------------------------------------------- #
def main() -> int:
    PS.apply()
    cs = D.curve_shape(AXIS)
    chosen, ranked = select_examples(cs)
    rows = example_rows(list(chosen.values()))
    ranked_by_curve = ranked.set_index("curve_id")

    fig = plt.figure(figsize=(PS.W2, 5.60))
    # An empty middle row is the spacer: constrained layout ignores ``hspace`` on an
    # outer gridspec whose cells are themselves subgridspecs, and without the gap the
    # bottom row's panel letters land on the top row's x-axis labels.
    outer = fig.add_gridspec(3, 1, height_ratios=[1.00, 0.07, 1.06])
    gs_top = outer[0].subgridspec(1, 3, width_ratios=[1.22, 1.00, 1.00])
    gs_bot = outer[2].subgridspec(1, 3, width_ratios=[1.44, 1.00, 1.00])

    # ------------------------------------------------------------------ A-C ----
    # One shared y-range for all three example panels.  The panel's subject is the
    # amplitude of the response, so the three panels are only comparable if a decade
    # of log10 D is the same number of millimetres in each of them.
    axA = fig.add_subplot(gs_top[0, 0])
    axB = fig.add_subplot(gs_top[0, 1], sharey=axA)
    axC = fig.add_subplot(gs_top[0, 2], sharey=axA)
    example_axes = [axA, axB, axC]
    example_record = []

    for ax, role in zip(example_axes, ROLES):
        block = rows[rows.curve_id == chosen[role]].sort_values("axis_value")
        y0 = float(block.log_D.mean())
        x = block.axis_value.to_numpy()
        ax.axhline(0.0, color=PS.PALE, linewidth=0.8, linestyle="-", zorder=0)
        levels = {}
        for key, _arm in STAGES:
            pred = block[f"pred_{key}"]
            levels[key] = float(pred.mean()) - y0
            ax.plot(x, pred - pred.mean(), **PS.m(key, markersize=3.0, linewidth=1.2,
                                                  zorder=3))
        ax.plot(x, block.log_D - y0, **PS.m("measured", markersize=3.8, linewidth=1.3,
                                            zorder=5))
        span = float(block.log_D.max() - block.log_D.min())
        # Lower-right corner: every example curve rises to the right, so the region
        # below the mid-line on the right-hand half of each panel carries no data.
        ax.text(0.985, 0.035,
                f"measured span {num(span)}\n"
                f"level error {num_small(levels['recomposed'])}",
                transform=ax.transAxes, ha="right", va="bottom", linespacing=1.35,
                fontsize=PS.BASE - 1.0, color=PS.GREY, zorder=6)
        PS.strapline(ax, ROLE_TITLE[role])
        ax.set_xlabel("log$_{10}$ [extractant] (M)")
        pad = 0.10 * (x.max() - x.min())
        ax.set_xlim(x.min() - pad, x.max() + pad)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        example_record.append({
            "role": role, "panel": "ABC"[ROLES.index(role)], "curve_id": chosen[role],
            "extractant": ranked_by_curve.loc[chosen[role], "extractant"],
            "n_points": int(len(block)), "measured_span": span,
            "level_error_recomposed": levels["recomposed"],
            "level_error_baseline": levels["baseline"]})

    axA.set_ylabel("log$_{10}$ $D$ $-$ mean over curve")
    axA.set_ylim(-2.15, 2.25)          # top third of A is empty: it holds the legend
    for ax in (axB, axC):
        ax.tick_params(labelleft=False)
    # Legend order is the row order of D/E/F, so one reading order serves the figure.
    handles, labels = axA.get_legend_handles_labels()
    order = [3, 0, 1, 2]
    axA.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="upper left", ncol=1, handlelength=1.7, borderaxespad=0.25,
               labelspacing=0.32)

    # ------------------------------------------------------------------ D-F ----
    axD = fig.add_subplot(gs_bot[0, 0])
    axE = fig.add_subplot(gs_bot[0, 1], sharey=axD)
    axF = fig.add_subplot(gs_bot[0, 2], sharey=axD)

    def reference_band(ax) -> None:
        """Tint the measured row so the four slots line up by eye across D, E and F."""
        ax.axhspan(ROW["measured"] - 0.5, ROW["measured"] + 0.5, color="#F0F0F0",
                   linewidth=0, zorder=-10)

    # --- D: slope distributions -------------------------------------------
    reference = cs[cs.model == "REC_ecfp_plus_recovered"]
    slope_data = [reference.slope_true.dropna().to_numpy()]
    slope_colours = [PS.colour("measured")]
    for key, arm in STAGES:
        slope_data.append(cs[cs.model == arm].slope_pred.dropna().to_numpy())
        slope_colours.append(PS.colour(key))
    measured_median = float(np.median(slope_data[0]))

    reference_band(axD)
    axD.vlines(0.0, ROW_TOP, ROW_BOTTOM, color=PS.GREY, linewidth=0.7, linestyle=":",
               zorder=1)
    # The reference drops *from* the measured row, so it never runs behind the value
    # label printed above that row and it reads as "the measurement, carried down".
    axD.vlines(measured_median, ROW["measured"], ROW_BOTTOM, color=PS.INK,
               linewidth=0.8, linestyle="--", zorder=1)
    bp = axD.boxplot(slope_data, positions=list(range(4)), orientation="horizontal",
                     widths=0.55, showfliers=False, patch_artist=True,
                     medianprops={"color": "white", "linewidth": 1.1},
                     whiskerprops={"linewidth": 0.7, "color": PS.INK},
                     capprops={"linewidth": 0.7, "color": PS.INK},
                     boxprops={"linewidth": 0.0}, zorder=3)
    for patch, colour in zip(bp["boxes"], slope_colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.92)
    for j, values in enumerate(slope_data):
        # Anchored above the box on screen (the y-axis runs downwards), clear of the
        # 0.55-wide box and of the row above it.
        axD.text(float(np.median(values)), j - 0.34, num(float(np.median(values))),
                 ha="center", va="bottom", fontsize=PS.BASE - 1.0, color=PS.INK,
                 zorder=6)
    # The dashed line is the same device as in E and F — "where the measurement sits" —
    # so it is named here too.  It goes to the right of the line, in the half-row gap
    # between two boxes: row pitch is 1.0 and the boxes are 0.55 wide, and to the right
    # of the line no box, whisker or value label reaches this far out.
    axD.text(measured_median + 0.12, 1.5, "measured\nmedian", ha="left", va="center",
             fontsize=PS.BASE - 1.0, color=PS.GREY, linespacing=1.25, zorder=6)
    axD.set_xlim(-1.0, 5.05)
    axD.set_xlabel("fitted slope, median\n"
                   "($\\Delta$log$_{10}$ $D$ / $\\Delta$log$_{10}$ [extractant])")
    PS.strapline(axD, "box: quartiles, number: median")

    # --- E, F: per-curve distributions with a point estimate and its interval ----
    def stage_panel(ax, column, xlabel, statistic, strap, xlim):
        reference_band(ax)
        ax.vlines(1.0, ROW["measured"], ROW_BOTTOM, color=PS.INK, linewidth=0.8,
                  linestyle="--", zorder=1)
        ax.text(0.97, ROW["measured"], "measured = 1", ha="right", va="center",
                fontsize=PS.BASE - 1.0, color=PS.GREY, zorder=4)
        summary = []
        for key, arm in STAGES:
            j = ROW[key]
            block = cs[cs.model == arm].dropna(subset=[column])
            point, lo, hi = D.block_bootstrap_stat(block[column],
                                                   block["tanimoto_cluster"], statistic)
            colour = PS.colour(key)
            parts = ax.violinplot([block[column].to_numpy()], positions=[j],
                                  orientation="horizontal", widths=0.62,
                                  showextrema=False, showmedians=False)
            for body in parts["bodies"]:
                body.set_facecolor(colour)
                body.set_edgecolor(colour)
                body.set_alpha(0.30)
                body.set_linewidth(0.5)
                body.set_zorder(2)
            # One glyph, not two marks: a capped bar is the 95 % interval and the open
            # dot on it is the point statistic named on the x axis.
            ax.errorbar([point], [j], xerr=[[point - lo], [hi - point]], fmt="o",
                        markersize=4.4, markerfacecolor="white", markeredgecolor=colour,
                        markeredgewidth=1.3, ecolor=colour, elinewidth=1.3,
                        capsize=2.8, capthick=1.0, zorder=6)
            ax.text(point, j - 0.36, num(point), ha="center", va="bottom",
                    fontsize=PS.BASE - 1.0, color=PS.INK, zorder=6)
            summary.append({"stage": PS.label(key), "arm": arm, "metric": column,
                            "statistic": statistic, "point": point, "ci95_low": lo,
                            "ci95_high": hi, "n_curve_seed": int(len(block))})
        ax.set_xlim(*xlim)
        ax.set_xlabel(xlabel)
        PS.strapline(ax, strap)
        ax.tick_params(labelleft=False)
        return summary

    span_xmax = 1.60
    sum_e = stage_panel(axE, "span_recovery",
                        "span recovery, median\n(predicted range $\\div$ measured range)",
                        "median", "dot: median, bar: 95 % CI", (-0.05, span_xmax))
    # The recomposed distribution has a thin tail past the right-hand limit; say so
    # rather than letting the violin run silently off the panel.  The strip below the
    # bottom violin is empty — the violins are 0.62 wide and the limit is 0.72 below
    # the last row.
    beyond = max((cs[cs.model == arm].span_recovery > span_xmax).mean()
                 for _key, arm in STAGES)
    axE.text(span_xmax - 0.02, ROW_BOTTOM - 0.14,
             f"{beyond * 100:.0f} % of curves lie beyond {span_xmax:.1f}",
             ha="right", va="center", fontsize=PS.BASE - 1.0, color=PS.GREY,
             path_effects=HALO)
    sum_f = stage_panel(axF, "spearman",
                        "within-curve Spearman $\\rho$, mean\n"
                        f"(rank correlation, {MINUS}1 to 1)",
                        "mean", "dot: mean, bar: 95 % CI", (-1.15, 1.15))

    axD.set_yticks(list(ROW.values()))
    axD.set_yticklabels([PS.label("measured")] + [PS.label(k) for k, _ in STAGES])
    axD.set_ylim(ROW_BOTTOM, ROW_TOP)  # inverted: row 0 (measured) at the top

    PS.add_panel_letters(fig, [axA, axB, axC, axD, axE, axF])
    report = PS.save(fig, HERE, "figure", strict=False)
    print(report)

    # ------------------------------------------------------------------ record --
    headline = []
    for key, arm in STAGES:
        block = cs[cs.model == arm]
        headline.append({"stage": PS.label(key), "arm": arm,
                         "n_curves_x_seeds": int(len(block)),
                         "n_distinct_curves": int(block.curve_id.nunique()),
                         "n_extractants": int(block.extractant.nunique()),
                         "n_chemotypes": int(block.tanimoto_cluster.nunique()),
                         "slope_true_median": float(block.slope_true.median()),
                         "slope_pred_median": float(block.slope_pred.median()),
                         "span_recovery_median": float(block.span_recovery.median()),
                         "shape_mae_mean": float(block.shape_mae.mean()),
                         "spearman_mean": float(block.spearman.mean())})
    values = {
        "figure": "figure_03_shape_recovery",
        "axis": AXIS,
        "example_seed": EXAMPLE_SEED,
        "selection_rule": "FIGURE_PLAN.md §9 (declared before any curve was inspected)",
        "selected_curves": chosen,
        "examples": example_record,
        "stage_summary": headline,
        "panelD_slope_median": {"Measured": measured_median,
                                **{PS.label(k): float(np.median(v))
                                   for (k, _), v in zip(STAGES, slope_data[1:])}},
        "panelE": sum_e,
        "panelF": sum_f,
        "display_notes": {
            "panelE_xlim": [-0.05, 1.60],
            "panelE_fraction_beyond_xlim": {
                PS.label(k): float((cs[cs.model == arm].span_recovery > 1.60).mean())
                for k, arm in STAGES},
        },
        "lint": {"ok": report.ok, "violations": [str(v) for v in report.violations]},
    }
    (HERE / "values.json").write_text(json.dumps(values, indent=1))
    print(pd.DataFrame(headline).to_string())
    print("selected:", json.dumps(chosen, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
