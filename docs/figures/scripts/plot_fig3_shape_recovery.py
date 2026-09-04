"""Figure 3 — amplitude compression on held-out titrations, and its repair.

Run from the repository root:
    ``.venv/bin/python figures/scripts/plot_fig3_shape_recovery.py``

Sources (all frozen run outputs, nothing transcribed):
  runs/gen9_shape/shape/curve_shape.parquet        per-curve slope / span / shape / Spearman
  runs/gen9_shape/curves/curve_membership.parquet  row -> curve and its position on the axis
  runs/gen9_shape/phase1/oof_REC_ecfp_plus_recovered.parquet
  runs/gen9_shape/relmono/oof_GEN9_REL_MONOLITH.parquet
  runs/gen9_shape/recomposed/oof_GEN9_SHAPE_RECOMPOSED.parquet

The three example curves in A-C are chosen by the rule declared in
``figures/FIGURE_PLAN.md`` §9 before any curve was looked at; the rule, the ranks it
produced and the runners-up are written to ``figures/derived/fig3_selected_curves.csv``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _paths  # noqa: E402
import _stats  # noqa: E402
import _style  # noqa: E402

AXIS = "extractant"
EXAMPLE_SEED = 104729          # first split seed, fixed before selection
MIN_POINTS = 5                 # declared in the selection rule

STAGES = [("frozen", "REC_ecfp_plus_recovered", "Baseline model"),
          ("relpos", "GEN9_REL_MONOLITH", "+ relative position"),
          ("recomposed", "GEN9_SHAPE_RECOMPOSED", "+ recomposition")]
OOF = {"frozen": "gen9_shape/phase1/oof_REC_ecfp_plus_recovered.parquet",
       "relpos": "gen9_shape/relmono/oof_GEN9_REL_MONOLITH.parquet",
       "recomposed": "gen9_shape/recomposed/oof_GEN9_SHAPE_RECOMPOSED.parquet"}
STYLE_KEY = {"frozen": "frozen", "relpos": "relpos", "recomposed": "recomposed"}


def load_curve_shape() -> pd.DataFrame:
    cs = pd.read_parquet(_paths.run("gen9_shape/shape/curve_shape.parquet"))
    return cs[cs.axis_label == AXIS].copy()


def select_examples(cs: pd.DataFrame) -> tuple[dict[str, str], pd.DataFrame]:
    """The pre-declared rule.  Returns {role: curve_id} and the full ranked table."""
    seed = cs[(cs.split_seed == EXAMPLE_SEED) & (cs.n_points >= MIN_POINTS)]
    wide = seed.pivot_table(index=["curve_id", "extractant", "n_points"], columns="model",
                            values="shape_mae")
    wide = wide[["REC_ecfp_plus_recovered", "GEN9_SHAPE_RECOMPOSED"]].dropna()
    wide.columns = ["shape_frozen", "shape_recomposed"]
    wide["delta"] = wide.shape_frozen - wide.shape_recomposed
    wide = wide.reset_index().sort_values("curve_id").reset_index(drop=True)

    iqr = lambda s: max(1e-9, float(s.quantile(0.75) - s.quantile(0.25)))  # noqa: E731
    zx = (wide.shape_frozen - wide.shape_frozen.median()) / iqr(wide.shape_frozen)
    zy = (wide.delta - wide.delta.median()) / iqr(wide.delta)
    wide["dist_to_joint_median"] = np.hypot(zx, zy)

    top = wide[wide.delta >= wide.delta.quantile(0.90)]
    bottom = wide[wide.delta <= wide.delta.quantile(0.10)]
    wide["dist_to_top_decile_median"] = np.where(
        wide.curve_id.isin(top.curve_id), (wide.delta - top.delta.median()).abs(), np.nan)
    wide["dist_to_bottom_decile_median"] = np.where(
        wide.curve_id.isin(bottom.curve_id), (wide.delta - bottom.delta.median()).abs(), np.nan)

    chosen = {
        "median": wide.sort_values(["dist_to_joint_median", "curve_id"]).curve_id.iloc[0],
        "improved": wide.sort_values(["dist_to_top_decile_median", "curve_id"]).curve_id.iloc[0],
        "residual": wide.sort_values(["dist_to_bottom_decile_median", "curve_id"]).curve_id.iloc[0],
    }
    wide["selected_as"] = wide.curve_id.map({v: k for k, v in chosen.items()}).fillna("")
    return chosen, wide


def example_rows(curve_ids: list[str]) -> pd.DataFrame:
    mem = pd.read_parquet(_paths.run("gen9_shape/curves/curve_membership.parquet"))
    mem = mem[(mem.axis_label == AXIS) & (mem.curve_id.isin(curve_ids))]
    frames = [mem[["curve_id", "row_id", "axis_value"]]]
    out = frames[0].copy()
    for key, path in OOF.items():
        oof = pd.read_parquet(_paths.run(path),
                              columns=["row_id", "split_seed", "log_D", "prediction"])
        oof = oof[oof.split_seed == EXAMPLE_SEED][["row_id", "log_D", "prediction"]]
        oof = oof.rename(columns={"prediction": f"pred_{key}"})
        out = out.merge(oof.drop_duplicates("row_id"), on="row_id", how="left")
        if "log_D_y" in out.columns:
            out = out.drop(columns=["log_D_y"]).rename(columns={"log_D_x": "log_D"})
    return out.sort_values(["curve_id", "axis_value"])


def main() -> int:
    _style.apply()
    cs = load_curve_shape()
    chosen, ranked = select_examples(cs)
    ranked.to_csv(_paths.DERIVED / "fig3_selected_curves.csv", index=False)
    rows = example_rows(list(chosen.values()))

    fig = plt.figure(figsize=(_style.W2, 4.55))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.02], hspace=0.52, wspace=0.34,
                          left=0.075, right=0.985, top=0.92, bottom=0.10)

    # ------------------------------------------------------------------ A-C ----
    titles = {"median": "median curve", "improved": "large repair",
              "residual": "repair fails"}
    example_record = []
    for i, role in enumerate(["median", "improved", "residual"]):
        ax = fig.add_subplot(gs[0, i])
        block = rows[rows.curve_id == chosen[role]].sort_values("axis_value")
        y0 = float(block.log_D.mean())
        ax.plot(block.axis_value, block.log_D - y0, color=_style.BLACK, marker="o",
                markersize=3.6, linewidth=1.0, label="Measured", zorder=5)
        levels = {}
        for key, _model, nice in STAGES:
            pred = block[f"pred_{key}"]
            levels[key] = float(pred.mean()) - y0
            kw = _style.m(STYLE_KEY[key])
            kw["label"] = nice
            kw["markersize"] = 2.8
            ax.plot(block.axis_value, pred - pred.mean(), **kw)
        ax.axhline(0, color=_style.GREY, linewidth=0.5, linestyle=":", zorder=0)
        ax.set_title(titles[role], fontsize=7.0, color=_style.GREY)
        ax.set_xlabel("log$_{10}$ [extractant] (M)")
        ax.text(0.97, 0.05, f"curve level error {levels['recomposed']:+.2f}",
                transform=ax.transAxes, ha="right", fontsize=5.9, color=_style.GREY)
        if i == 0:
            ax.set_ylabel("log$_{10}$ $D$ $-$ mean over curve")
            ax.legend(loc="upper left", fontsize=6.0, bbox_to_anchor=(-0.03, 1.03))
        _style.panel(ax, "ABC"[i], dx=-0.22 if i else -0.28)
        example_record.append({
            "role": role, "curve_id": chosen[role],
            "extractant": block.curve_id.map(lambda c: c).iloc[0] and
                          ranked.set_index("curve_id").loc[chosen[role], "extractant"],
            "n_points": int(len(block)),
            "measured_span": float(block.log_D.max() - block.log_D.min()),
            "level_error_recomposed": levels["recomposed"],
            "level_error_frozen": levels["frozen"]})

    # ------------------------------------------------------------------ D ------
    axD = fig.add_subplot(gs[1, 0])
    order = ["measured"] + [k for k, _, _ in STAGES]
    data, labels, colours = [], [], []
    ref = cs[cs.model == "REC_ecfp_plus_recovered"]
    data.append(ref.slope_true.dropna().to_numpy())
    labels.append("Measured")
    colours.append(_style.BLACK)
    for key, model, nice in STAGES:
        block = cs[cs.model == model]
        data.append(block.slope_pred.dropna().to_numpy())
        labels.append(nice)
        colours.append(_style.colour(STYLE_KEY[key]))
    bp = axD.boxplot(data, orientation="horizontal", widths=0.6, showfliers=False, patch_artist=True,
                     medianprops={"color": "white", "linewidth": 1.0},
                     whiskerprops={"linewidth": 0.6}, capprops={"linewidth": 0.6},
                     boxprops={"linewidth": 0.0})
    for patch, c in zip(bp["boxes"], colours):
        patch.set_facecolor(c)
        patch.set_alpha(0.9)
    for i, values in enumerate(data):
        axD.text(np.median(values), i + 1.42, f"{np.median(values):.2f}", ha="center",
                 fontsize=6.0, color=colours[i])
    axD.axvline(0, color=_style.GREY, linewidth=0.6, linestyle=":")
    axD.set_yticks(range(1, len(labels) + 1))
    axD.set_yticklabels(labels)
    axD.set_xlabel("slope of log$_{10}$ $D$ vs log$_{10}$ [extractant]")
    axD.set_xlim(-0.9, 5.2)
    _style.panel(axD, "D", dx=-0.50)

    # ------------------------------------------------------------------ E, F ---
    def stage_panel(ax, column, xlabel, letter, statistic, reference=None, xlim=None):
        summary = []
        for j, (key, _model, nice) in enumerate(STAGES):
            block = cs[cs.model == _model].dropna(subset=[column])
            point, lo, hi = _stats.block_bootstrap_stat(block[column],
                                                        block["tanimoto_cluster"], statistic)
            parts = ax.violinplot([block[column].to_numpy()], positions=[j],
                                  orientation="horizontal", widths=0.85,
                                  showextrema=False, showmedians=False)
            for body in parts["bodies"]:
                body.set_facecolor(_style.colour(STYLE_KEY[key]))
                body.set_alpha(0.35)
                body.set_linewidth(0)
            ax.plot([point], [j], marker="|", markersize=11, markeredgewidth=1.4,
                    color=_style.colour(STYLE_KEY[key]))
            ax.plot([lo, hi], [j - 0.30, j - 0.30], color=_style.colour(STYLE_KEY[key]),
                    linewidth=1.1)
            ax.text(point, j + 0.40, f"{point:.2f}", ha="center", fontsize=6.0,
                    color=_style.colour(STYLE_KEY[key]))
            summary.append({"stage": nice, "metric": column, "statistic": statistic,
                            "point": point, "ci95_low": lo, "ci95_high": hi,
                            "n_curve_seed": int(len(block))})
        if reference is not None:
            ax.axvline(reference, color=_style.BLACK, linewidth=0.7, linestyle="--")
        ax.set_yticks(range(len(STAGES)))
        ax.set_yticklabels([nice for _, _, nice in STAGES])
        ax.set_xlabel(xlabel)
        if xlim:
            ax.set_xlim(*xlim)
        _style.panel(ax, letter, dx=-0.50)
        return summary

    axE = fig.add_subplot(gs[1, 1])
    sumE = stage_panel(axE, "span_recovery",
                       "span recovery, median\n(predicted / measured range)",
                       "E", "median", reference=1.0, xlim=(-0.05, 1.15))
    axF = fig.add_subplot(gs[1, 2])
    sumF = stage_panel(axF, "spearman", "within-curve Spearman $\\rho$, mean", "F",
                       "mean", reference=1.0, xlim=(-1.15, 1.15))

    written = _style.save(fig, "Fig3_shape_recovery", _paths.MAIN)

    # ------------------------------------------------------------------ record --
    headline = []
    for key, model, nice in STAGES:
        block = cs[cs.model == model]
        headline.append({"stage": nice, "model": model,
                         "n_curves_x_seeds": int(len(block)),
                         "n_distinct_curves": int(block.curve_id.nunique()),
                         "n_extractants": int(block.extractant.nunique()),
                         "slope_true_median": float(block.slope_true.median()),
                         "slope_pred_median": float(block.slope_pred.median()),
                         "span_recovery_median": float(block.span_recovery.median()),
                         "shape_mae_mean": float(block.shape_mae.mean()),
                         "spearman_mean": float(block.spearman.mean())})
    record = {"figure": "Fig3_shape_recovery", "axis": AXIS,
              "example_seed": EXAMPLE_SEED, "selected_curves": chosen,
              "selection_rule": "FIGURE_PLAN.md §9",
              "stage_summary": headline, "examples": example_record,
              "panelE": sumE, "panelF": sumF}
    (_paths.DERIVED / "fig3_values.json").write_text(json.dumps(record, indent=1))
    print("\n".join(str(p) for p in written))
    print(pd.DataFrame(headline).to_string())
    print("\nselected:", json.dumps(chosen, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
