#!/usr/bin/env python
"""Figure 4 — where the remaining error is, and how much of it is reachable.

Refined from ``figures/scripts/plot_fig4_error_decomposition.py``.  The numbers, cohorts,
seeds, aggregations and arms are unchanged; this pass changes layout, typography,
labelling and information design only.

The trust guard of the original is kept and strengthened: ``prepare_error_budget.py`` is
re-run on every render, and it refuses to write ``error_budget.json`` unless the oracle
cascade it recomputes from the raw out-of-fold predictions reproduces the frozen
``runs/gen10_final/error_decomposition/summary.json`` to 5e-9 on all ten stage/metric
values.  This script additionally re-asserts those ten deltas before drawing anything.

Run from anywhere:
    python figure_refinement/ml_separator/figure_04_error_decomposition/figure_script.py
"""

from __future__ import annotations

import json
import subprocess
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

sys.path.insert(0, str(D.REPO / "src"))
from lanthanide_separation.gen6.metrics import decompose_level_shape  # noqa: E402

TOL = 5e-9

#: Four zero-shot model stages of panel C.  ``model`` is the arm identifier inside the
#: run registry; ``stage`` is what the reader sees — no repository identifier reaches the
#: artwork.
MODEL_STAGES = [
    ("no_ligand", "NULL_metal_cond",
     "gen7_architecture/finalists/oof_predictions.parquet", "no ligand information"),
    ("baseline", "REC_ecfp_plus_recovered",
     "gen7_architecture/finalists/oof_predictions.parquet", "baseline model"),
    ("relpos", "GEN9_REL_MONOLITH",
     "gen9_shape/relmono/oof_GEN9_REL_MONOLITH.parquet", "+ relative position"),
    ("recomposed", "GEN9_SHAPE_RECOMPOSED",
     "gen9_shape/recomposed/oof_GEN9_SHAPE_RECOMPOSED.parquet", "+ recomposition"),
]
#: (k, adapter) of the deployable measurement path, identical to the original figure.
KSHOT_STAGES = [(0, "ZERO_SHOT_REF"), (1, "SLOPE_L_s1_K1"), (2, "SERIES_ML"),
                (3, "SERIES_ML"), (5, "SERIES_ML")]
KSHOT_MODEL = "GEN9_SHAPE_RECOMPOSED"

#: Reader-facing names for the seven budget components of panel B.
SHORT_NAMES = {
    "1 predictable level (per-ligand constant)": "per-extractant level",
    "2 curve shape (within-curve, after the curve's level)": "within-curve shape",
    "3 series-local level (beyond the ligand constant)": "series-local level",
    "4 system identity (name/structure mismatch rows)": "system identity",
    "5 sparse support / distant chemotype": "distant chemistry",
    "6 suspected data quality (duplicates, TWE-24, flagged)": "data quality",
    "7 residual unexplained": "unexplained",
}

#: The two cohorts of this figure, spelled out in the artwork so that a value from B is
#: never read against a value from A.  B is additionally tinted and titled in ink rather
#: than grey, because 0.520 in B and 0.507 in A are different quantities on different
#: cohorts and the figure has to say so where the reader is looking.
COHORT_A = "99 extractants · one vote per extractant · 5 split seeds"
COHORT_B = ("different cohort from A and C\n"
            "152 extractants · one vote per structural cluster")
COHORT_C = ("same cohort and voting as A\n"
            "99 extractants · one vote per extractant")
TINT_B = "#F2F1EC"        # marks panel B as the odd cohort out


def fmt(v: float, places: int = 3) -> str:
    """Three decimals with a typographic minus — the precision of the paper caption."""
    return f"{v:.{places}f}".replace("-", "−")


def build_budget() -> dict:
    """Re-run the verification prep script, then re-assert its ten checks here."""
    subprocess.run([sys.executable, str(D.REPO / "figures" / "scripts"
                                        / "prepare_error_budget.py")],
                   check=True, capture_output=True)
    budget = json.loads((D.DERIVED / "error_budget.json").read_text())
    checks = budget["verification"]["checks"]
    if len(checks) != 10 or not all(c["pass"] and c["abs_delta"] < TOL for c in checks):
        raise SystemExit("oracle cascade did NOT reproduce summary.json — figure not drawn")
    return budget


def model_stage_decomposition(ligands: set[str]) -> pd.DataFrame:
    rows = []
    for key, model, path, nice in MODEL_STAGES:
        oof = pd.read_parquet(D.run(path),
                              columns=["row_id", "extractant", "log_D", "prediction",
                                       "split_seed", "model"])
        oof = oof[(oof.model == model) & (oof.extractant.isin(ligands))]
        per_seed = []
        for seed, block in oof.groupby("split_seed"):
            d = decompose_level_shape(block.log_D, block.prediction, block.extractant)
            per_seed.append({"split_seed": int(seed),
                             "mae": d.summary["macro_mae_ligand"],
                             "offset_mae": d.summary["offset_mae"],
                             "shape_mae": d.summary["shape_mae"]})
        frame = pd.DataFrame(per_seed)
        rows.append({"key": key, "stage": nice, "model": model, "n_seeds": len(frame),
                     "n_ligands": int(oof.extractant.nunique()),
                     **frame.drop(columns="split_seed").mean().to_dict()})
    return pd.DataFrame(rows)


def panel_a(ax, budget: dict) -> list[dict]:
    """Deployable stages and oracle substitutions on one axis, one cohort."""
    stages = budget["common_cohort"]["stages"]
    achieved = budget["common_cohort"]["achieved"]
    ladder = [
        ("$k$ = 0\nzero-shot", achieved["zero_shot"], False),
        ("$k$ = 1", achieved["k1_slope_repair_central"], False),
        ("$k$ = 3", achieved["k3_series_ml_central"], False),
        ("$k$ = 5", achieved["k5_series_ml_central"], False),
        ("$k$ = 1\nbest point", achieved["k1_oracle_point"], True),
        ("level per\nextractant", stages["after_ligand_level_oracle"]["macro_mae"], True),
        ("level per\nseries", stages["after_series_level_oracle"]["macro_mae"], True),
        ("level per\ncurve", stages["after_curve_level_oracle"]["macro_mae"], True),
        ("level + slope\nper curve",
         stages["after_curve_level_and_slope_oracle"]["macro_mae"], True),
    ]
    # A physical gap, not only a dotted rule, separates the two halves: the eye reads a
    # gap as "different kind of thing" without needing to find the divider first.
    gap = 0.75
    x = np.array([0, 1, 2, 3, 4 + gap, 5 + gap, 6 + gap, 7 + gap, 8 + gap], dtype=float)
    values = [v for _, v, _ in ladder]
    oracle = [o for _, _, o in ladder]

    # A faint y grid behind the bars: 0.493 and 0.507 are 60 mm apart at print width,
    # on opposite sides of the group gap, and cannot be ranked by eye without one.
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E9E9E9", linewidth=0.5)

    deploy = ~np.array(oracle)
    ax.bar(x[deploy], np.array(values)[deploy], width=0.58,
           color=PS.colour("pipeline"), linewidth=0)
    # Oracle bars are hollow with a coarse hatch: an open bar reads as "hypothetical"
    # even in greyscale and at column width, where a solid fill of another hue does not.
    ax.bar(x[~deploy], np.array(values)[~deploy], width=0.58, facecolor="white",
           edgecolor=PS.colour("oracle"), linewidth=0.9, hatch="///")

    for xi, v in zip(x, values):
        ax.text(xi, v + 0.022, fmt(v), ha="center", va="bottom", fontsize=PS.BASE - 1.0)

    ax.plot([3.5 + gap / 2] * 2, [0, 1.10], color=PS.PALE, linewidth=0.8, zorder=0)

    # Group headings sit above every bar top (max 1.036 + label ≈ 1.10) with their own
    # rule, so they cannot collide with a value label.
    for x0, x1, text, colour in [
            (-0.45, 3.45, "deployable — measure $k$ points (central choice)",
             PS.colour("pipeline")),
            (4 + gap - 0.45, 8 + gap + 0.45,
             "oracle — true values supplied, not deployable", PS.colour("oracle"))]:
        ax.plot([x0, x1], [1.155, 1.155], color=colour, linewidth=0.8,
                solid_capstyle="butt")
        ax.text((x0 + x1) / 2, 1.185, text, ha="center", va="bottom",
                fontsize=PS.BASE - 0.5, color=colour)

    ax.set_xticks(x)
    ax.set_xticklabels([n for n, _, _ in ladder])
    ax.set_xlim(-0.72, 8 + gap + 0.72)
    # Ticks stop at 1.0, above the tallest bar: a gridline at 1.2 would run between the
    # group headings and their rules.
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylim(0, 1.30)
    ax.spines["left"].set_bounds(0, 1.0)   # the axis stops at its last tick
    ax.set_ylabel("macro MAE (log$_{10}$ $D$)")
    ax.tick_params(axis="x", length=0)
    PS.strapline(ax, COHORT_A)
    return [{"stage": " ".join(n.replace("$", "").split()), "macro_mae": v, "oracle": o}
            for n, v, o in ladder]


def panel_b(ax, budget: dict) -> pd.DataFrame:
    """The seven-component budget — a different cohort, marked as such."""
    comp = pd.DataFrame(budget["components"])
    comp["short"] = comp["component"].map(SHORT_NAMES).fillna(comp["component"])
    comp = comp.sort_values("current_contribution", ascending=True).reset_index(drop=True)
    y = np.arange(len(comp))

    ax.set_facecolor(TINT_B)
    ax.barh(y, comp.current_contribution, height=0.66, color=PS.GREY, alpha=0.42,
            linewidth=0, label="current contribution")
    ax.barh(y, comp.realistically_removable, height=0.34, color=PS.colour("pipeline"),
            linewidth=0, label="removed by a deployable action")

    # The current-contribution values go in a right-aligned column in the reserved right
    # margin, not at the bar ends: for "series-local level" the deployable action removes
    # more than the component's own share (the components overlap), so a number at the
    # end of the longer bar would be read as belonging to the wrong series.
    for yi, row in comp.iterrows():
        ax.text(0.618, yi, fmt(row.current_contribution), va="center", ha="right",
                fontsize=PS.BASE - 1.0)
        if row.realistically_removable > 0:
            ax.text(row.realistically_removable + 0.012, yi,
                    fmt(row.realistically_removable), va="center", ha="left",
                    fontsize=PS.BASE - 1.0, color=PS.colour("pipeline"),
                    path_effects=[pe.withStroke(linewidth=1.2, foreground="white")])

    ax.set_yticks(y)
    ax.set_yticklabels(list(comp.short))
    # The zero line is the baseline of the bars; the left spine would just double it.
    ax.spines["left"].set_visible(False)
    ax.axvline(0, color=PS.INK, linewidth=0.7)
    ax.set_xticks(np.arange(0, 0.55, 0.1))
    ax.spines["bottom"].set_bounds(0, 0.5)
    ax.set_xlim(-0.022, 0.625)
    # Two empty rows are reserved below the shortest bars so the legend has somewhere to
    # live that is not on top of data.
    ax.set_ylim(-2.05, len(comp) - 0.35)
    ax.set_xlabel("share of this cohort's 0.970 macro MAE (log$_{10}$ $D$)")
    ax.legend(loc="lower left", ncol=1, fontsize=PS.BASE - 1.0,
              handlelength=1.5, borderaxespad=0.15)
    ax.set_title(COHORT_B, fontsize=PS.BASE - 0.5, color=PS.INK, loc="left")
    ax.tick_params(axis="y", length=0)
    return comp


def panel_c(ax, models: pd.DataFrame, ks: pd.DataFrame) -> None:
    """Trajectory in (level error, shape error) space."""
    blue = PS.colour("recomposed")
    verm = PS.colour("pipeline")

    ax.plot(models.offset_mae, models.shape_mae, color=blue, linewidth=1.1,
            marker="D", markersize=3.8, zorder=3)
    # The four model stages fall inside 0.06 of level error and 0.08 of shape error, so
    # written-out names cannot be placed at the markers.  Numbered nodes plus a key in
    # the empty upper-left corner keep every name readable at print size.  Sides
    # alternate so that consecutive numbers, 0.013 apart in y, cannot touch.
    for i, (_, r) in enumerate(models.iterrows(), start=1):
        dx, ha = ((6, "left") if i % 2 else (-6, "right"))
        ax.annotate(str(i), (r.offset_mae, r.shape_mae), textcoords="offset points",
                    xytext=(dx, -2.5), fontsize=PS.BASE - 0.5, fontweight="bold",
                    color=blue, ha=ha, va="center")

    ax.plot(ks.offset_mae, ks.shape_mae, color=verm, linewidth=1.1,
            marker="o", markersize=3.8, zorder=3)
    for _, r in ks.iterrows():
        ax.annotate(f"$k$ = {int(r.k)}", (r.offset_mae, r.shape_mae),
                    textcoords="offset points", xytext=(3.5, -9),
                    fontsize=PS.BASE - 0.5, color=verm)

    # Upper-left is empty by construction: nothing in this study has a small level error
    # together with a large shape error.  The key is placed there, not in a legend box.
    key = "better zero-shot model\n" + "\n".join(
        f"{i}   {r.stage}" for i, (_, r) in enumerate(models.iterrows(), start=1))
    ax.text(0.02, 0.985, key, transform=ax.transAxes, ha="left", va="top",
            fontsize=PS.BASE - 0.5, color=blue, linespacing=1.45)
    # Lower-right is empty for the same reason in reverse: no arm has a large level error
    # with a small shape error.  The measurement path is direct-labelled there.
    ax.text(0.985, 0.19, "more measurements\non the final model",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=PS.BASE - 0.5, color=verm, linespacing=1.45)

    ax.set_xlim(0.15, 1.03)
    ax.set_ylim(0.355, 0.685)
    ax.spines["left"].set_bounds(0.40, 0.65)
    ax.spines["bottom"].set_bounds(0.2, 1.0)
    ax.set_xlabel("level error, mean $|$per-extractant offset$|$ (log$_{10}$ $D$)")
    ax.set_ylabel("shape error, within-extractant MAE\n(log$_{10}$ $D$)")
    PS.strapline(ax, COHORT_C)


def main() -> int:
    PS.apply()
    matplotlib.rcParams["hatch.linewidth"] = 0.55

    budget = build_budget()
    per_ligand = D.kshot_per_ligand()
    ligands = set(per_ligand.extractant.unique())

    models = model_stage_decomposition(ligands)
    rows = []
    for k, adapter in KSHOT_STAGES:
        block = per_ligand[(per_ligand.global_model == KSHOT_MODEL)
                           & (per_ligand.adapter == adapter) & (per_ligand.k == k)
                           & (per_ligand.policy.isin(["CENTRAL_THEN_SPREAD", "NONE"]))]
        rows.append({"k": k, "offset_mae": block.offset.mean(),
                     "shape_mae": block.shape_mae.mean(), "mae": block.mae.mean()})
    ks = pd.DataFrame(rows)

    fig = plt.figure(figsize=(PS.W2, 5.30))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.94, 1.22], width_ratios=[1.06, 1.00])
    axA = fig.add_subplot(gs[0, :])
    axB = fig.add_subplot(gs[1, 0])
    axC = fig.add_subplot(gs[1, 1])

    ladder = panel_a(axA, budget)
    comp = panel_b(axB, budget)
    panel_c(axC, models, ks)

    PS.add_panel_letters(fig, [axA, axB, axC])
    report = PS.save(fig, HERE, "figure", strict=False)
    print(report)

    values = {
        "figure": "figure_04_error_decomposition",
        "verification": {"tolerance": TOL, "all_pass": budget["verification"]["all_pass"],
                         "checks": budget["verification"]["checks"]},
        "panelA": ladder,
        "panelA_cohort": budget["common_cohort"]["description"],
        "panelB": comp.drop(columns=["component"]).round(6).to_dict("records"),
        "panelB_cohort": budget["published_study_cohort"]["description"],
        "panelC_model_stages": models.round(6).to_dict("records"),
        "panelC_kshot_stages": ks.round(6).to_dict("records"),
        "lint": {"ok": report.ok, "violations": [str(v) for v in report.violations]},
    }
    (HERE / "values.json").write_text(json.dumps(values, indent=1))
    print(models.round(4).to_string())
    print(ks.round(4).to_string())
    print(comp[["short", "current_contribution", "realistically_removable"]]
          .round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
