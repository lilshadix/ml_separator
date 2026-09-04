#!/usr/bin/env python
"""Figure 1 — where each model lands, and what the 3D arm itself contributes.

Refined from ``automl/figures_pi_email.py::fig_result`` of
``github.com/mironovb/lanthanidestrain``, and extended with the pre-declared
confirmation of the 3D contrast, which the original does not show.

Every number is read from the vendored copy of ``automl/reports/*.csv|json`` and is
unchanged; this pass alters layout, typography, naming and information design only.

The organising rule of the figure is the warning in the upstream docstring: *the paired
bootstrap contrasts are computed on shared pairs and do not equal the difference of two
whole-set scores*.  Levels therefore live in panels A and B, contrasts in panel C, and no
bar height on this figure may be subtracted from another to obtain a contribution.

Run from anywhere:
    python figure_refinement/lanthanidestrain/figure_01_headline_result/figure_script.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt            # noqa: E402
from matplotlib.lines import Line2D        # noqa: E402
from matplotlib.text import Text           # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common"))
import lstrain_data as LD                  # noqa: E402
import pubstyle as PS                      # noqa: E402

# --------------------------------------------------------------------------- #
# Reader-facing names.  The left-hand keys are the repository's; none of them may
# survive into a drawn text artist (asserted at the end of ``main``).  The wording
# matches the sibling refined figures so one arm has one name across the set.
#
#   CatBoost   the tuned gradient-boosting model on the 2D tabular block
#              (best_stack.py: overall R2 +0.4987, adjacent +0.1422)
#   repaired   the published fingerprint baseline -- an sklearn MLP on ECFP + RDKit
#              descriptors with StandardScaler, 16-seed ensemble
#              (CONTROL_RESULTS.md sec. 4.2, C6_PREREGISTRATION.md "repaired
#              fingerprint net")
#   S0         the simplicial message-passing encoder over the GFN2-xTB complex,
#              trained with the pairwise-contrast objective, 16 seeds
#   T0w        the matched control: the same harness, folds, seeds and objective as
#              the 3D encoder with the 3D input removed and a wide head
#              (CONTROL_PREREGISTRATION.md sec. 2 and 5)
# --------------------------------------------------------------------------- #
NAME = {
    "CatBoost": "gradient boosting (2D)",
    "repaired": "fingerprint network (2D)",
    "S0": "3D encoder",
    "T0w": "matched 2D control",
    "no topology (CatBoost+repaired)": "gradient boosting + fingerprint network",
    "topology swapped for control": "+ matched 2D control (no 3D input)",
    "full (CatBoost+repaired+S0)": "+ 3D encoder",
    "adj_r2_binned": "adjacent-pair log SF R² (conditions binned into blocks)",
    "fresh": "held-out confirmation pairs",
    "legacy": "pairs used throughout the study",
    "all": "both sets together",
    "drop-in": "3D encoder added to the 2D pair",
    "swap": "3D encoder in place of the control",
}

#: What the two halves of the figure are models *of*.  Panel A is the July
#: combination study: three whole models blended with weights fitted per held-out
#: extractant.  Panels B and C are the August system, in which a block-level term is
#: fitted first and the 3D encoder contributes only to the within-block series shape,
#: at a weight fixed in advance -- a different architecture on the same metric, which
#: is why its levels are not comparable to panel A's row by row.
SYSTEMS = {
    "panel_A": "combination study: gradient boosting, the fingerprint network and the "
               "3D encoder blended, weights fitted per held-out extractant "
               "(best_stack.py, stack_test.py)",
    "panels_B_C_confirmation": "anchored system: the composition-block level is fitted "
               "separately from the within-block series shape, and the 3D encoder "
               "enters the shape channel only, at a weight fixed at 0.35 in advance "
               "(anchored_3d_confirm.py)",
}

#: Colour semantics, one channel one meaning, in every panel:
#:   BLUE       the arm or system contains the 3D encoder
#:   DARK GREY  the matched combination without any 3D input
#:   VERMILLION the matched control that occupies the 3D slot but sees no 3D input
#:   PURPLE / ORANGE  the two single 2D models
#:
#: Checked with the source repository's own computed palette test
#: (automl/tests/test_palette.py: Machado CVD simulation, OKLab dE, its thresholds).
#: These five at pubstyle's GREY (#7F7F7F) give worst-pair dE 6.2 under protanopia
#: (purple against grey), inside the 6-8 band its rule admits only with a second
#: encoding; at #4D4D4D the same five give 9.6, above the 8.0 target, with
#: normal-vision dE 15.6 and WCAG contrast 8.4 against white.  Text stays at
#: pubstyle's GREY, which is de-emphasis rather than data.
D3 = PS.BLUE
NO3D = "#4D4D4D"
CTRL = PS.VERMILLION
BOOST = PS.PURPLE
FPRINT = PS.ORANGE

#: Cohort facts that are definitions rather than results, each with its source.
#: AUDIT_2026-07-30.md sec. 1: "905 pairs, 552 blocks, 4,746 rows, 162 extractants",
#: and mean(dy) = -0.0724, i the lighter lanthanide.  README.md first paragraph:
#: leave-extractants-out cross-validation, 5 folds x 3 repeats.
N_PAIRS_LEGACY = 905
N_BLOCKS = 552
MEAN_SEPARATION = -0.072

BANNED = re.compile(
    r"\b(S0|T0w|G0|D0|CatBoost|repaired|composition_key|strict_composition_key|"
    r"adj_r2_binned|adj_r2_strict|arm_obs|baseline_obs|p_better|w_fixed|"
    r"primary_pass|anchored_3d|no topology|drop-in)\b")


# --------------------------------------------------------------------------- #
def row_labels(ax, positions, labels, *, styles=None, size=PS.BASE - 0.5, pad=4):
    """Row labels against the axis, so no row is separated from its own bar.

    Right-aligned deliberately: a left-aligned gutter wide enough for the longest
    label leaves the short ones an inch away from the data they name.  Hierarchy is
    carried by the italic group headings and by the leading ``+`` of a continuation
    row, not by indentation.
    """
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", pad=pad, length=0)
    for i, artist in enumerate(ax.get_yticklabels()):
        artist.set_fontsize(size)
        if styles is not None:
            artist.update(styles[i])


def forest_row(ax, y, point, lo, hi, lo_c, hi_c, colour):
    """One contrast row: thin multiplicity-corrected interval, thick 90 %, point."""
    ax.plot([lo_c, hi_c], [y, y], color=colour, lw=0.9, zorder=3,
            solid_capstyle="butt")
    ax.plot([lo, hi], [y, y], color=colour, lw=2.6, zorder=4, solid_capstyle="round")
    ax.plot([point], [y], "o", ms=4.6, zorder=5, markerfacecolor=colour,
            markeredgecolor="white", markeredgewidth=0.8)


# --------------------------------------------------------------------------- #
def main() -> int:
    PS.apply()

    # ------------------------------------------------------------ data ------
    arms = LD.read("dualkey_arms").set_index("arm")["adj_r2_binned"]
    best = LD.read("best_stack")
    stack = LD.read("stack_test").set_index("contrast")
    ensemble = LD.read("adjacent_ensemble")
    anchored = LD.read_json("anchored_3d")
    confirm = LD.read_json("anchored_3d_confirm")

    drop = best[best["base"].str.startswith("no topology")].iloc[0]
    swap = best[best["base"].str.startswith("topology swapped")].iloc[0]
    primary = stack.loc["1_primary"]
    encoder = ensemble[ensemble["config"].str.startswith("snn_")].iloc[0]

    # The joins above are by position and by string prefix; assert that every one of
    # them landed on the arm it is supposed to, using values the tables share.
    assert abs(float(encoder["ensemble_adj_r2"]) - float(arms["S0"])) < 1e-12
    assert abs(float(encoder["baseline_obs"]) - float(arms["CatBoost"])) < 1e-9
    assert abs(float(primary["baseline_obs"]) - float(arms["repaired"])) < 1e-9
    assert abs(float(drop["arm_obs"]) - float(swap["arm_obs"])) < 1e-12
    assert confirm["primary_pass"] is True
    assert abs(confirm["w_fixed"] - 0.35) < 1e-12
    w = anchored["main"]["w_stats"]
    assert w["min"] <= confirm["w_fixed"] <= w["max"]
    for pop in confirm["results"].values():
        assert abs((pop["blend"]["r2"] - pop["tabular"]["r2"]) - pop["contrast"]) < 1e-12
        assert pop["blend"]["n"] == pop["tabular"]["n"]

    levels = {
        "CatBoost": float(arms["CatBoost"]),
        "repaired": float(arms["repaired"]),
        "S0": float(arms["S0"]),
        "pair_blend": float(primary["arm_obs"]),
        "no_3d": float(drop["baseline_obs"]),
        "control": float(swap["baseline_obs"]),
        "full": float(drop["arm_obs"]),
    }

    # (kind, label, value, colour, 90 % interval on the level or None)
    rows_a = [
        ("head", "single models", None, None, None),
        ("bar", NAME["CatBoost"], levels["CatBoost"], BOOST, None),
        ("bar", NAME["repaired"], levels["repaired"], FPRINT, None),
        ("bar", NAME["S0"], levels["S0"], D3,
         (float(encoder["arm_lo"]), float(encoder["arm_hi"]))),
        ("head", "two-model combinations", None, None, None),
        ("bar", f'{NAME["repaired"]} + {NAME["S0"]}', levels["pair_blend"], D3,
         (float(primary["arm_lo"]), float(primary["arm_hi"]))),
        ("head", "three-model combinations", None, None, None),
        ("bar", NAME["no topology (CatBoost+repaired)"], levels["no_3d"], NO3D, None),
        ("bar", "+ matched 2D control (no 3D input)", levels["control"], CTRL, None),
        ("bar", "+ 3D encoder", levels["full"], D3,
         (float(drop["arm_lo"]), float(drop["arm_hi"]))),
    ]
    pos_a, y = [], 0.0
    for i, row in enumerate(rows_a):
        if row[0] == "head" and i:
            y -= 0.45
        pos_a.append(y)
        y -= 1.0

    populations = [("fresh", "held-out confirmation pairs, frozen before the model ran",
                    "held-out confirmation pairs"),
                   ("legacy", "the pairs used throughout the study",
                    "pairs used throughout"),
                   ("all", "both sets together", "both sets together")]
    conf_rows = []
    for key, label, short in populations:
        block = confirm["results"][key]
        conf_rows.append({
            "key": key, "label": label, "short": short,
            "n": int(block["blend"]["n"]),
            "tabular": float(block["tabular"]["r2"]),
            "blend": float(block["blend"]["r2"]),
            "contrast": float(block["contrast"])})

    # Shared row grid for panels B and C: two contrast rows, then the three
    # populations, which are the same three rows in both panels.  Panel B labels
    # them; panel C needs no second copy of the labels.
    Y_C1, Y_C2 = -1.0, -2.0
    Y_HEAD = -3.25
    Y_POP = [-4.20, -5.20, -6.20]
    YLIM_BC = (-7.30, 0.85)

    # --------------------------------------------------------- figure ------
    fig = plt.figure(figsize=(PS.W2, 5.6))
    # A nested grid, not a 2x2 one: panel A's row-label gutter is much deeper than
    # panel B's, and in a flat grid its left margin would be imposed on B as well.
    outer = fig.add_gridspec(2, 1, height_ratios=[1.30, 1.00])
    inner = outer[1].subgridspec(1, 2, width_ratios=[1.00, 1.02], wspace=0.06)
    fig.get_layout_engine().set(hspace=0.10, wspace=0.05)
    axA = fig.add_subplot(outer[0])
    axB = fig.add_subplot(inner[0])
    axC = fig.add_subplot(inner[1])          # same row grid as B, set explicitly:
    # sharey would make one set of tick labels serve both panels, and each needs its own

    # ------------------------------------------------------------- A -------
    x_max = 0.335
    for (kind, label, value, colour, ci), y in zip(rows_a, pos_a):
        if kind != "bar":
            continue
        axA.barh(y, value, height=0.6, color=colour, linewidth=0, zorder=3)
        right = value
        if ci is not None:
            axA.plot(ci, [y, y], color=PS.INK, lw=0.9, zorder=5,
                     solid_capstyle="butt")
            for end in ci:
                axA.plot([end, end], [y - 0.16, y + 0.16], color=PS.INK, lw=0.9,
                         zorder=5)
            right = max(right, ci[1])
        axA.annotate(f"{value:+.4f}", (right, y), textcoords="offset points",
                     xytext=(5, 0), ha="left", va="center",
                     fontsize=PS.BASE - 1.0, color=PS.INK, clip_on=True)
    styles_a = [{"color": PS.GREY, "style": "italic"} if k == "head"
                else {"color": PS.INK} for k, *_ in rows_a]
    row_labels(axA, pos_a, [r[1] for r in rows_a], styles=styles_a)
    axA.set_xlim(0, x_max)
    axA.set_ylim(pos_a[-1] - 3.05, pos_a[0] + 0.85)   # the strip under the bars
    # holds the two sentences a reader needs before subtracting anything.
    axA.set_xticks([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    axA.set_xlabel(
        "adjacent-pair log SF R²   ·   log SF = log₁₀ D(lighter lanthanide) − "
        "log₁₀ D(heavier neighbour)\n"
        "0 = every pair separates by the series average, "
        f"−{abs(MEAN_SEPARATION):.3f}")
    PS.strapline(axA, "Levels — where each model lands   ·   "
                      f"{N_PAIRS_LEGACY} adjacent pairs   ·   "
                      "leave-extractants-out CV, 5 × 3")
    axA.annotate(
        "A difference between two bars is not the 3D arm's contribution — "
        "that is panel C.\n"
        "Whisker: 90 % interval on the level, where reported.  Network arms are "
        f"16-seed ensembles, seed SD {float(encoder['single_sd']):.3f}.\n"
        f"Conditions are binned into {N_BLOCKS} blocks; every level here is lower "
        "under a stricter definition of matched conditions.",
        (0.0, pos_a[-1] - 0.62), ha="left", va="top", clip_on=True,
        fontsize=PS.BASE - 1.2, color=PS.GREY, linespacing=1.6)

    # ------------------------------------------------------------- B -------
    for ax in (axB, axC):
        ax.axhspan(Y_POP[-1] - 0.55, Y_POP[0] + 0.55, color="#F4F4F4", zorder=0,
                   linewidth=0)
    for row, y in zip(conf_rows, Y_POP):
        axB.plot([row["tabular"], row["blend"]], [y, y], color=PS.PALE, lw=1.6,
                 zorder=2)
        axB.plot([row["tabular"]], [y], "o", ms=4.4, markerfacecolor="white",
                 markeredgecolor=NO3D, markeredgewidth=1.1, zorder=4)
        axB.plot([row["blend"]], [y], "o", ms=4.4, color=D3, zorder=5)
        # The value pair goes on whichever side of the row has room for it: the
        # lowest population sits too close to zero for a right-aligned label.
        left = row["blend"] < 0.15
        axB.annotate(f"{row['tabular']:.3f}$\\,\\rightarrow\\,${row['blend']:.3f}",
                     (row["blend"], y), textcoords="offset points",
                     xytext=(-3 if left else 3, 6),
                     ha="left" if left else "right", va="bottom",
                     fontsize=PS.BASE - 1.2, color=PS.INK, clip_on=True)
    labels_b = [f"{r['short']}, n = {r['n']:,}" for r in conf_rows]
    row_labels(axB, Y_POP, labels_b, styles=[{"color": PS.INK}] * 3,
               size=PS.BASE - 1.0)
    axB.set_xlim(0, x_max)
    axB.set_ylim(*YLIM_BC)
    axB.set_xticks([0.0, 0.15, 0.30])
    axB.set_xlabel("adjacent-pair log SF R²")
    PS.strapline(axB, "Levels — the confirmation, a later system")
    # A legend is sized in points, so leaving it in the layout solve lets it push the
    # axes narrower, which pushes the legend further out: it is excluded instead, and
    # the empty upper band is reserved for it by the y limit.
    legend = axB.legend(handles=[
        Line2D([], [], marker="o", ls="none", ms=4.4, markerfacecolor="white",
               markeredgecolor=NO3D, markeredgewidth=1.1, label="2D system alone"),
        Line2D([], [], marker="o", ls="none", ms=4.4, color=D3,
               label="with the 3D shape channel")],
        loc="upper left", bbox_to_anchor=(-0.01, 1.03), handletextpad=0.5)
    legend.set_in_layout(False)
    axB.annotate(
        "the 3D increment, 0.015, is too small\n"
        "to read here — panel C measures it",
        (0.004, Y_C1 - 0.40), ha="left", va="top", clip_on=True,
        fontsize=PS.BASE - 1.2, color=PS.GREY, linespacing=1.6)

    # ------------------------------------------------------------- C -------
    forest_row(axC, Y_C1, float(drop["delta"]), float(drop["lo"]), float(drop["hi"]),
               float(drop["lo_5test"]), float(drop["hi_5test"]), D3)
    forest_row(axC, Y_C2, float(swap["delta"]), float(swap["lo"]), float(swap["hi"]),
               float(swap["lo_5test"]), float(swap["hi_5test"]), D3)
    for row, y in zip(conf_rows, Y_POP):
        axC.plot([row["contrast"]], [y], "D", ms=4.4, color=D3, zorder=5)
        axC.annotate(f"{row['contrast']:+.4f}   n = {row['n']:,}", (row["contrast"], y),
                     textcoords="offset points", xytext=(6, 0), ha="left",
                     va="center", fontsize=PS.BASE - 1.0, color=PS.INK,
                     clip_on=True)
    for y, value, right in ((Y_C1, float(drop["delta"]), float(drop["hi_5test"])),
                            (Y_C2, float(swap["delta"]), float(swap["hi_5test"]))):
        axC.annotate(f"{value:+.4f}", (right, y), textcoords="offset points",
                     xytext=(6, 0), ha="left", va="center",
                     fontsize=PS.BASE - 1.0, color=PS.INK, clip_on=True)
    axC.axvline(0, color=PS.INK, lw=0.7, zorder=1)

    labels_c = [f"3D encoder added to the 2D pair, n = {N_PAIRS_LEGACY}",
                "3D encoder in place of the control, "
                f"n = {N_PAIRS_LEGACY}",
                "pre-declared confirmation\n"
                f"weight fixed at 0.35 (median of {w['min']:.2f}–{w['max']:.2f})"]
    pos_c = [Y_C1, Y_C2, Y_HEAD]
    row_labels(axC, pos_c, labels_c,
               styles=[{"color": PS.INK}, {"color": PS.INK},
                       {"color": PS.GREY, "style": "italic"}],
               size=PS.BASE - 1.0)
    axC.set_ylim(*YLIM_BC)
    axC.set_xlim(-0.007, 0.101)
    axC.set_xticks([0.0, 0.03, 0.06])
    axC.set_xlabel("Δ adjacent-pair log SF R²\npositive: the 3D arm adds")
    PS.strapline(axC, "Contrasts — paired, on shared pairs")
    axC.annotate("thick 90 % · thin 5-test corrected",
                 (0.0016, Y_C2 - 0.5), ha="left", va="top", clip_on=True,
                 fontsize=PS.BASE - 1.2, color=PS.GREY)
    axC.annotate("no interval reported for these",
                 (0.0016, Y_POP[-1] - 0.5), ha="left", va="top", clip_on=True,
                 fontsize=PS.BASE - 1.2, color=PS.GREY)

    # Nothing in the artwork may carry a repository identifier.
    drawn = [t.get_text() for t in fig.findobj(Text) if t.get_visible()]
    offenders = sorted({m.group(0) for t in drawn for m in BANNED.finditer(str(t))})
    assert not offenders, f"repository identifiers in the artwork: {offenders}"

    PS.add_panel_letters(fig, [axA, axB, axC])
    report = PS.save(fig, HERE, "figure", strict=False)
    print(report)

    values = {
        "figure": "figure_01_headline_result",
        "source": {
            "repository": "https://github.com/mironovb/lanthanidestrain",
            "commit": LD.provenance()["commit"],
            "script": "automl/figures_pi_email.py::fig_result",
            "tables": ["automl/reports/dualkey_arms.csv",
                       "automl/reports/best_stack.csv",
                       "automl/reports/stack_test.csv",
                       "automl/reports/adjacent_ensemble.csv",
                       "automl/reports/anchored_3d.json",
                       "automl/reports/anchored_3d_confirm.json"],
            "reports": ["automl/reports/PI_REPORT.md",
                        "automl/reports/COLLAB_UPDATE_REPORT.md",
                        "automl/reports/STACK_RESULTS.md",
                        "automl/reports/DUALKEY_RESULTS.md",
                        "automl/reports/CAMPAIGN_AUG2026.md"],
        },
        "metric": (
            "adjacent-pair log SF R2 = R2 of the log10 separation factor between "
            "neighbouring lanthanides inside one extractant and condition block "
            "(evaluation.py sel_adj_logSF_r2). Replicates are averaged per "
            "(block, metal) first; the difference is taken lighter minus heavier "
            "neighbour; R2 is taken about the mean pair separation, -0.0724."),
        "cohort": {
            "panel_A_and_the_two_contrasts": {
                "n_adjacent_pairs": N_PAIRS_LEGACY, "n_blocks": N_BLOCKS,
                "n_rows": 4746, "n_extractants": 162,
                "source": "AUDIT_2026-07-30.md sec. 1",
                "n_extractants_caveat":
                    "AUDIT_2026-07-30.md gives 162 extractants for this 905-pair set; "
                    "best_stack.py's docstring and STACK_PREREGISTRATION.md sec. 3 say "
                    "the nested weights are fitted over 149. Which count the cluster "
                    "bootstrap resamples is not decidable from the vendored tables, so "
                    "no extractant count is drawn in the artwork.",
                "cv": "leave-extractants-out, 5 folds x 3 repeats",
                "bootstrap": "cluster bootstrap over whole extractants, 400 draws, "
                             "seed 0, 90 % interval"},
            "panel_B_and_C_confirmation": {
                "n_by_population": {r["key"]: r["n"] for r in conf_rows},
                "rule": "anchored_3d_confirm.py, committed before the encoder retrain "
                        "finished: blend weight fixed at 0.35, primary endpoint is the "
                        "sign of the contrast on the held-out 444"},
        },
        "name_map": NAME,
        "systems": SYSTEMS,
        "panel_A_levels": [
            {"label": label, "value": value,
             "level_ci_90": list(ci) if ci else None}
            for kind, label, value, _c, ci in rows_a if kind == "bar"],
        "panel_B_confirmation_levels": conf_rows,
        "panel_C_contrasts": [
            {"label": "3D encoder added to the 2D pair", "n_pairs": N_PAIRS_LEGACY,
             "delta": float(drop["delta"]), "lo_90": float(drop["lo"]),
             "hi_90": float(drop["hi"]), "lo_corrected": float(drop["lo_5test"]),
             "hi_corrected": float(drop["hi_5test"]),
             "verdict": str(drop["verdict_5test"])},
            {"label": "3D encoder in place of the matched control",
             "n_pairs": N_PAIRS_LEGACY,
             "delta": float(swap["delta"]), "lo_90": float(swap["lo"]),
             "hi_90": float(swap["hi"]), "lo_corrected": float(swap["lo_5test"]),
             "hi_corrected": float(swap["hi_5test"]),
             "verdict": str(swap["verdict_5test"])},
            *[{"label": r["label"].replace("\n", " "), "n_pairs": r["n"],
               "delta": r["contrast"], "interval": None} for r in conf_rows],
        ],
        "confirmation_rule": {
            "w_fixed": confirm["w_fixed"],
            "w_nested_on_the_earlier_set": {"mean": w["mean"], "min": w["min"],
                                            "max": w["max"]},
            "primary_pass": confirm["primary_pass"]},
        "not_plotted": {
            "anchored_3d.json main": (
                "the same architecture with the weight re-fitted per held-out "
                "extractant scores 0.3258 against 0.3182 tabular-only on the 905 "
                "pairs; it is a different run from the confirmation (retrained "
                "encoder, fixed weight) and is deliberately not drawn beside it"),
            "adjacent_ensemble second row": (
                "the persistence-image encoder ensemble, +0.2080, which the original "
                "figure does not carry either"),
            "stack_test rows 2_control, 3_decisive, desc_S2": (
                "the pre-registered contrast set; it is figure 2 of this refinement"),
        },
        "palette_check": {
            "method": "the source repository's automl/tests/test_palette.py, unchanged "
                      "(Machado CVD simulation, OKLab dE x100, its own thresholds: "
                      "CVD dE >= 8 target / 6 floor, normal-vision dE >= 15, "
                      "WCAG contrast >= 3)",
            "original_figure_four_hues": {
                "colours": ["#4a3aa7", "#eb6834", "#2a78d6", "#e34948"],
                "worst_cvd_dE": 5.62, "worst_cvd_pair": ["#eb6834", "#e34948"],
                "worst_cvd_kind": "deutan", "worst_normal_dE": 7.13,
                "note": "below the 6.0 hard floor; this four-hue combination is not "
                        "among the subsets that test registers, so it was never checked"},
            "refined_figure_five_colours": {
                "colours": [BOOST, FPRINT, D3, CTRL, NO3D],
                "worst_cvd_dE": 9.55, "worst_normal_dE": 15.58,
                "contrast_vs_white": {"purple": 3.06, "orange": 2.25, "blue": 5.19,
                                      "vermillion": 3.87, "grey": 8.45},
                "note": "the one threshold not met is WCAG 3:1 for the orange against "
                        "white (2.25); it is the shared style's orange, carried by a "
                        "large filled bar with its own row label"},
        },
        "single_seed_sd": float(encoder["single_sd"]),
        "n_seeds": int(encoder["n_seeds"]),
        "lint": {"ok": report.ok, "violations": [str(v) for v in report.violations]},
    }
    (HERE / "values.json").write_text(json.dumps(values, indent=1))

    print("levels :", {k: round(v, 4) for k, v in levels.items()})
    print("contrasts:", round(float(drop["delta"]), 4), round(float(swap["delta"]), 4),
          [round(r["contrast"], 4) for r in conf_rows])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
