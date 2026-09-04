#!/usr/bin/env python
"""Figure 2 — the three pre-registered contrasts, and the control that must come out flat.

Refined from ``automl/figures_pi_email.py::fig_control`` of
``github.com/mironovb/lanthanidestrain``.  Every number is read from the vendored copy of
``automl/reports/stack_test.csv`` and is unchanged; this pass alters layout, typography,
naming and information design only.

The figure is deliberately a *contrast-only* figure.  The upstream module warns in
``fig_result``'s docstring that the paired bootstrap contrasts are computed on the pairs
two systems share and therefore do **not** equal the difference of two whole-set scores
(here: +0.0351 against a level difference of +0.0305).  Levels are a different figure; no
bar height on this one can be subtracted from another.

Run from anywhere:
    python figure_refinement/lanthanidestrain/figure_02_preregistered_controls/figure_script.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches      # noqa: E402
import matplotlib.pyplot as plt            # noqa: E402
from matplotlib.lines import Line2D        # noqa: E402
from matplotlib.text import Text           # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common"))
import lstrain_data as LD                  # noqa: E402
import pubstyle as PS                      # noqa: E402

# --------------------------------------------------------------------------- #
# Reader-facing names.  The left-hand keys are the repository's; none of them may
# survive into a drawn text artist (asserted at the end of ``main``).
#
#   repaired  the published FCNN baseline (sklearn MLP on ECFP + RDKit descriptors)
#             with its QuantileTransformer replaced by StandardScaler, 16-seed
#             ensemble  -- CONTROL_RESULTS.md sec. 4.2
#   S0        the simplicial message-passing network over the GFN2-xTB complex,
#             trained with the pairwise-contrast objective, 16 seeds
#   T0w       the matched tabular control: same harness, folds, seeds and objective
#             as S0 with the 3D encoder removed and a wide head
#             -- CONTROL_PREREGISTRATION.md sec. 5
# --------------------------------------------------------------------------- #
NAME = {
    "repaired": "fingerprint network",
    "S0": "3D encoder",
    "T0w": "matched no-3D model",
    "none": "nothing added",
}

#: One row per pre-registered contrast, in the order the pre-registration fixes them.
#: ``ref``/``arm`` name the second slot of each system; the fingerprint network is the
#: first slot of all six systems and is what every contrast holds fixed.
ROWS = [
    {"key": "1_primary", "role": "PRIMARY",
     "question": "does the 3D encoder add at all?",
     "ref": "none", "arm": "S0", "tag": "", "colour": PS.BLUE},
    {"key": "2_control", "role": "CONTROL",
     "question": "does any second model add?",
     "ref": "none", "arm": "T0w", "tag": "must come out flat", "colour": PS.VERMILLION},
    {"key": "3_decisive", "role": "DECISIVE",
     "question": "is it the 3D encoder specifically?",
     "ref": "T0w", "arm": "S0", "tag": "", "colour": PS.BLUE},
]

#: Cohort.  ``stack_test.csv`` carries no per-contrast n, so the extractant count comes
#: from the two upstream statements about this same analysis: STACK_PREREGISTRATION.md
#: sec. 3 ("the weight is picked on the other 148 only") and the ``149 x 21 x 4 blends``
#: comment in ``automl/topo/stack_test.py::nested_blend``.
N_EXTRACTANTS = 149

CHIP_W = 0.44          # axes fraction, panel A
CHIP_H = 0.125         # data units (rows are 1.0 apart)
COL_X = (0.24, 0.76)   # centres of the reference and tested columns
DY = 0.20              # vertical offset of each chip from the row centre


def chip(ax, xc: float, yc: float, label: str, *, face: str, edge: str,
         text_colour: str, dashed: bool = False) -> None:
    """One model box in panel A."""
    ax.add_patch(mpatches.Rectangle(
        (xc - CHIP_W / 2, yc - CHIP_H), CHIP_W, 2 * CHIP_H,
        facecolor=face, edgecolor=edge, linewidth=0.7,
        linestyle=(0, (2.2, 1.6)) if dashed else "solid", zorder=2))
    ax.text(xc, yc, label, ha="center", va="center", fontsize=PS.BASE - 1.0,
            color=text_colour, zorder=3)


def recipe(ax, xc: float, y: float, slot: str) -> None:
    """A system: the fixed fingerprint network above, the slot that changes below."""
    chip(ax, xc, y + DY, NAME["repaired"], face="#EFEFEF", edge=PS.PALE,
         text_colour=PS.GREY)
    ax.text(xc, y, "+", ha="center", va="center", fontsize=PS.BASE - 1.5,
            color=PS.GREY)
    if slot == "none":
        chip(ax, xc, y - DY, NAME["none"], face="none", edge=PS.PALE,
             text_colour=PS.GREY, dashed=True)
    elif slot == "S0":
        chip(ax, xc, y - DY, NAME["S0"], face=PS.BLUE, edge="none",
             text_colour="white")
    else:
        chip(ax, xc, y - DY, NAME["T0w"], face=PS.VERMILLION, edge="none",
             text_colour="white")


def main() -> int:
    PS.apply()
    table = LD.read("stack_test").set_index("contrast")
    n_boot = int(table.loc["1_primary", "n_boot"])

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(PS.W2, 3.78), sharey=True,
        gridspec_kw={"width_ratios": [1.02, 1.00]})

    ylim = (-2.44, 0.95)

    # ------------------------------------------------------------------ A ----
    axA.set_xlim(0, 1)
    axA.set_ylim(*ylim)
    for spine in axA.spines.values():
        spine.set_visible(False)
    axA.set_xticks([])
    axA.set_yticks([])

    axA.text(COL_X[0], 0.72, "reference", ha="center", va="center",
             fontsize=PS.BASE - 1.2, color=PS.GREY)
    axA.text(COL_X[1], 0.72, "tested system", ha="center", va="center",
             fontsize=PS.BASE - 1.2, color=PS.GREY)

    roles, questions = [], []
    for i, row in enumerate(ROWS):
        y = -i
        roles.append(axA.text(
            0.0, y + 0.46, f"{i + 1}  {row['role']}", ha="left", va="baseline",
            fontsize=PS.BASE - 1.2, fontweight="bold", color=row["colour"]))
        questions.append(axA.text(
            0.0, y + 0.46, row["question"], ha="left", va="baseline",
            fontsize=PS.BASE - 0.8, color=PS.INK))
        if row["tag"]:
            axA.text(1.0, y + 0.46, row["tag"], ha="right", va="baseline",
                     fontsize=PS.BASE - 1.2, color=PS.GREY, style="italic")
        recipe(axA, COL_X[0], y, row["ref"])
        recipe(axA, COL_X[1], y, row["arm"])
        axA.text(0.50, y, "vs", ha="center", va="center",
                 fontsize=PS.BASE - 1.0, color=PS.GREY)

    # The question column starts clear of the widest role word, measured rather than
    # guessed, so a longer role name cannot silently collide with it.
    fig.canvas.draw()
    to_axes = axA.transAxes.inverted()
    edge = max(to_axes.transform((t.get_window_extent().x1, 0))[0] for t in roles)
    for t in questions:
        t.set_x(edge + 0.030)

    PS.strapline(axA, "grey: held fixed in every contrast   ·   colour: what changes",
                 size=PS.BASE - 1.0)

    # ------------------------------------------------------------------ B ----
    plotted = []
    for i, row in enumerate(ROWS):
        y = -i
        r = table.loc[row["key"]]
        colour = row["colour"]
        # Butt caps everywhere: a rounded cap would draw the interval half a line
        # width past its own endpoint, which on this scale is ~0.001 R².
        axB.plot([r["lo_3test"], r["hi_3test"]], [y, y], color=colour,
                 lw=1.1, alpha=0.9, solid_capstyle="butt", zorder=2)
        for end in ("lo_3test", "hi_3test"):
            axB.plot([r[end], r[end]], [y - 0.075, y + 0.075], color=colour,
                     lw=1.1, alpha=0.9, solid_capstyle="butt", zorder=2)
        axB.plot([r["lo"], r["hi"]], [y, y], color=colour, lw=3.6,
                 solid_capstyle="butt", zorder=3)
        axB.plot([r["delta"]], [y], "o", color=colour, ms=6.0,
                 markeredgecolor="white", markeredgewidth=0.9, zorder=4)
        axB.text(0.0757, y, f"{float(r['delta']):+.4f}", va="center", ha="left",
                 fontsize=PS.BASE - 0.5, color=colour)
        def system(slot: str) -> str:
            return (NAME["repaired"] if slot == "none"
                    else f"{NAME['repaired']} + {NAME[slot]}")

        plotted.append({
            "contrast": row["key"], "role": row["role"],
            "question_in_figure": row["question"],
            "reference_system": system(row["ref"]),
            "tested_system": system(row["arm"]),
            "delta": float(r["delta"]), "lo_90": float(r["lo"]),
            "hi_90": float(r["hi"]), "lo_corrected": float(r["lo_3test"]),
            "hi_corrected": float(r["hi_3test"]),
            "p_better": float(r["p_better"]),
            "verdict_90": str(r["verdict"]),
            "verdict_corrected": str(r["verdict_3test"])})

    axB.axvline(0, color=PS.INK, lw=0.9, zorder=1)
    axB.set_xlim(-0.0315, 0.0985)
    axB.set_ylim(*ylim)
    axB.set_xticks([-0.02, 0.00, 0.02, 0.04, 0.06])
    axB.spines["left"].set_visible(False)   # y is three named rows, not a scale
    # The axis stops where the data does; the strip to its right is the value column.
    axB.spines["bottom"].set_bounds(-0.0315, 0.0700)
    axB.tick_params(axis="y", left=False, labelleft=False)
    # x = 0.39 centres the label on the drawn spine, not on the panel that also holds
    # the value column to its right.
    axB.set_xlabel("Δ adjacent-pair separation R²   (tested − reference)",
                   x=0.39, ha="center")

    # The one contrast whose verdict the multiplicity correction removes.
    row3 = table.loc["3_decisive"]
    axB.text(float(row3["lo_3test"]), -2.20, "corrected interval spans zero",
             ha="left", va="top", fontsize=PS.BASE - 1.4, color=PS.GREY)

    axB.legend(handles=[Line2D([], [], color=PS.GREY, lw=3.6,
                               solid_capstyle="butt", label="90 % interval"),
                        Line2D([], [], color=PS.GREY, lw=1.1,
                               marker="|", markersize=5.0, markeredgewidth=1.1,
                               label="corrected for three looks")],
               loc="upper right", bbox_to_anchor=(1.005, 1.0),
               fontsize=PS.BASE - 1.2, handlelength=1.6, labelspacing=0.35)
    PS.strapline(axB, "paired difference, same extractants in both systems")

    # ------------------------------------------------------------- footnote --
    fig.supxlabel(
        "Contrasts, statistic and outcome map were fixed in writing before any of "
        "them was computed.\n"
        "Metric: R² of the log₁₀ separation factor between neighbouring lanthanides "
        "in one extractant and condition block.\n"
        "Positive Δ favours the tested system. Each system blends its two models at a "
        f"weight fitted per held-out extractant on the other {N_EXTRACTANTS - 1}.\n"
        f"Thick bar: 90 % interval, {n_boot} resamples of the {N_EXTRACTANTS} "
        "extractants. Thin bar: corrected for three looks at this question "
        "(Bonferroni-style).",
        x=0.005, ha="left", fontsize=PS.BASE - 1.2, color=PS.GREY)

    PS.add_panel_letters(fig, [axA, axB])

    # No repository identifier reaches the artwork.
    forbidden = ("S0", "T0w", "repaired", "blend(", "adj_r2", "stack_test",
                 "1_primary", "2_control", "3_decisive", "lo_3test", "FCNN", "SNN")
    drawn = [" ".join(str(t.get_text()).split()) for t in fig.findobj(Text)
             if str(t.get_text()).strip()]
    bad = [(k, s) for s in drawn for k in forbidden if k in s]
    if bad:
        raise SystemExit(f"repository identifier in the artwork: {bad}")

    report = PS.save(fig, HERE, "figure", strict=False)
    print(report)

    values = {
        "figure": "figure_02_preregistered_controls",
        "source": {
            "repository": LD.provenance()["source_repository"],
            "commit": LD.provenance()["commit"],
            "script": "automl/figures_pi_email.py::fig_control",
            "table": "automl/reports/stack_test.csv",
            "pre_registration": "automl/reports/STACK_PREREGISTRATION.md",
            "results": "automl/reports/STACK_RESULTS.md"},
        "metric": ("adjacent-pair separation R² = R² of the log10 separation "
                   "factor between neighbouring lanthanides within one extractant and "
                   "condition block (sel_adj_logSF_r2); positive delta favours the "
                   "tested system"),
        "name_map": {"repaired": NAME["repaired"], "S0": NAME["S0"],
                     "T0w": NAME["T0w"], "(absent second model)": NAME["none"]},
        "cohort": {"n_extractants": N_EXTRACTANTS, "n_bootstrap_draws": n_boot,
                   "n_extractants_source": ("STACK_PREREGISTRATION.md sec. 3 and the "
                                            "nested_blend docstring; stack_test.csv "
                                            "carries no per-contrast n")},
        "contrasts": plotted,
        "not_plotted": {"desc_S2": ("the descriptive fourth row of stack_test.csv; "
                                    "not a pre-registered contrast and not drawn "
                                    "upstream either")},
        "levels_are_elsewhere": {
            "note": ("levels are deliberately absent: the paired bootstrap delta is not "
                     "the difference of two whole-set scores"),
            "example": {"contrast": "1_primary",
                        "delta": float(table.loc["1_primary", "delta"]),
                        "arm_obs_minus_baseline_obs": float(
                            table.loc["1_primary", "arm_obs"]
                            - table.loc["1_primary", "baseline_obs"])}},
        "lint": {"ok": report.ok, "violations": [str(v) for v in report.violations]},
    }
    (HERE / "values.json").write_text(json.dumps(values, indent=1))
    for p in plotted:
        print(f"{p['role']:9s} {p['delta']:+.4f}  90% [{p['lo_90']:+.4f}, "
              f"{p['hi_90']:+.4f}]  corrected [{p['lo_corrected']:+.4f}, "
              f"{p['hi_corrected']:+.4f}]  {p['verdict_corrected']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
