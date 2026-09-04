#!/usr/bin/env python
"""Figure 3 — the two qualifications the 3D result carries.

Refined from ``automl/figures_pi_email.py::fig_limits`` of
``github.com/mironovb/lanthanidestrain``.  Same tables, same rows, same numbers,
same intervals; this pass changes layout, typography, naming and information
design only.

The figure answers two different kinds of question and keeps them apart:

* top row  — *is the effect topological?*  Three 3D encoders differing only in
  whether the 2-simplices are there, scored under the published (binned) key.
* bottom row — *how strictly must "identical conditions" match?*  The two
  pre-registered stack contrasts under both block-key definitions: a sensitivity,
  not a second result.

Left column carries **differences** (paired cluster bootstrap on shared pairs),
right column carries **levels**.  They are never on the same axis, because a
difference here is not the gap between two levels.

Run from anywhere:
    python figure_refinement/lanthanidestrain/figure_03_limits/figure_script.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.lines import Line2D      # noqa: E402
from matplotlib.ticker import FixedLocator, MultipleLocator  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common"))
import lstrain_data as LD                # noqa: E402
import pubstyle as PS                    # noqa: E402

# --------------------------------------------------------------------------- #
# Colour semantics.  One channel, one meaning, everywhere in the figure:
#   BINNED   = the published definition of "identical conditions" (binned)
#   STRICT   = every condition matched
# Marker *fill* carries the pre-registered verdict: filled when the
# multiplicity-corrected interval clears zero, open when it spans zero.
# Marker *shape* carries the kind of quantity: circle = difference, square = level.
# --------------------------------------------------------------------------- #
BINNED = PS.BLUE
STRICT = PS.VERMILLION

#: Repository identifier -> what a reader sees.  Sources:
#: ENCODER_RESULTS.md ("the arms" table), DUALKEY_RESULTS.md, CONTROL_RESULTS.md
#: (T0w = tabular + contrast, wide head, *the* matched control),
#: C6_PREREGISTRATION.md ("repaired fingerprint net"), dataset.py:387 (the keys).
LABELS = {
    "S0": "3D: edges + triangles",
    "G0": "3D: triangles deleted",
    "D0": "3D: distances only",
    "T0w": "matched 2D control",
    "CatBoost": "gradient boosting (2D)",
    "repaired": "fingerprint network (2D)",
    "composition_key": "conditions binned — 552 blocks (published)",
    "strict_composition_key": "every condition matched — 2,109 blocks",
}

#: Block counts quoted in the legend come from DUALKEY_RESULTS.md ("552 blocks
#: become 2,109"); they are the definition of the two keys and are not in the
#: vendored CSVs.  Nothing else in the artwork is transcribed.
N_BLOCKS = {"composition_key": 552, "strict_composition_key": 2109}

#: Cohort line.  README.md of the source repository, first paragraph.
COHORT = ("Leave-extractants-out cross-validation, 5 folds × 3 repeats  ·  "
          "4,746 measurements, 162 extractants, 14 lanthanides\n"
          "R² = 0 means no better than assuming every adjacent pair separates "
          "by the average amount.\n"
          "Differences (A, C) are a paired bootstrap over the pairs two systems "
          "share — they are not the gaps between the levels in B and D.")

BANNED = re.compile(r"\b(S0|G0|D0|T0w|CatBoost|repaired|composition_key|"
                    r"strict_composition_key|adj_r2_binned|adj_r2_strict|p_better|"
                    r"no topology|drop-in|swap)\b")


# --------------------------------------------------------------------------- #
def look_columns(df):
    """The multiplicity-corrected interval columns, and the look count in them."""
    lo = next(c for c in df.columns if re.fullmatch(r"lo_(\d+)look", c))
    n = int(re.fullmatch(r"lo_(\d+)look", lo).group(1))
    return lo, lo.replace("lo_", "hi_", 1), n


def forest(ax, y, row, lo_c, hi_c, colour, *, value_x=None):
    """One difference row: thin corrected interval, thick 90 % interval, point.

    ``clears`` is read from the pre-registered ``verdict_corrected`` column rather
    than recomputed, so the artwork cannot disagree with the decision rule.
    """
    clears = str(row["verdict_corrected"]).strip() == "adds"
    ax.plot([row[lo_c], row[hi_c]], [y, y], color=colour, lw=0.9, zorder=3,
            solid_capstyle="butt")
    ax.plot([row["lo"], row["hi"]], [y, y], color=colour, lw=2.6, zorder=4,
            solid_capstyle="round")
    ax.plot([row["delta"]], [y], "o", ms=4.6, zorder=5,
            markerfacecolor=colour if clears else "white",
            markeredgecolor=colour, markeredgewidth=1.0)
    text = f"{float(row['delta']):+.4f}"
    if not clears:                      # P is shown exactly where the verdict is a null
        text += f"\nP = {float(row['p_better']):.2f}"
    ax.annotate(text, (float(row[hi_c] if value_x is None else value_x), y),
                textcoords="offset points", xytext=(4, 0), ha="left", va="center",
                fontsize=PS.BASE - 1.0, color=colour)
    return clears


def level(ax, y, value, colour, *, side="right"):
    """One level row: a square, and its value in the empty half of the row."""
    ax.plot([value], [y], "s", ms=4.2, color=colour, zorder=4)
    dx, ha = ((4, "left") if side == "right" else (-4, "right"))
    ax.annotate(f"{value:+.3f}", (value, y), textcoords="offset points",
                xytext=(dx, 0), ha=ha, va="center", fontsize=PS.BASE - 1.0,
                color=colour)


def tidy(ax, ylabels, ypos, xlim, *, major):
    ax.set_yticks(ypos)
    ax.set_yticklabels(ylabels, fontsize=PS.BASE - 0.5)
    ax.set_xlim(*xlim)
    ax.xaxis.set_major_locator(major)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color="#EDEDED", lw=0.5, zorder=0)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------- #
def main() -> int:
    PS.apply()
    enc = LD.read("encoder_test")
    arms = LD.read("encoder_arms")
    dual = LD.read("dualkey_test")

    enc_lo, enc_hi, enc_looks = look_columns(enc)
    dua_lo, dua_hi, dua_looks = look_columns(dual)
    enc_b = enc[enc["key"] == "composition_key"]
    arm_level = arms.set_index("arm")["adj_r2_binned"]

    fig = plt.figure(figsize=(PS.W2, 5.35))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.50, 1.00], height_ratios=[1.00, 0.92])
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])
    fig.get_layout_engine().set(hspace=0.13, wspace=0.05)

    DIFF_XLIM = (-0.036, 0.090)
    LVL_XLIM = (0.126, 0.306)

    # ------------------------------------------------------------------ A ---
    # Three pre-registered encoder contrasts, published (binned) key only.
    enc_rows = [
        ("with G0", "no topology",
         "3D graph, triangles deleted\nadded to the 2D-only pair"),
        ("with D0", "no topology",
         "3D distances only, no simplices\nadded to the 2D-only pair"),
        ("with S0", "with D0",
         "triangles against distances,\nsame slot, head to head"),
    ]
    ypos_a, labels_a, drawn_a = [0.0, -1.0, -2.35], [], []
    for y, (arm, base, label) in zip(ypos_a, enc_rows):
        r = enc_b[(enc_b["arm"] == arm) & (enc_b["base"] == base)].iloc[0]
        clears = forest(axA, y, r, enc_lo, enc_hi, BINNED)
        labels_a.append(label)
        drawn_a.append({"arm": arm, "base": base, "label": label.replace("\n", " "),
                        "delta": float(r["delta"]), "lo": float(r["lo"]),
                        "hi": float(r["hi"]), "lo_corrected": float(r[enc_lo]),
                        "hi_corrected": float(r[enc_hi]),
                        "p_better": float(r["p_better"]),
                        "verdict_corrected": str(r["verdict_corrected"])})
    axA.axvline(0, color=PS.INK, lw=0.9, zorder=2)
    tidy(axA, labels_a, ypos_a, DIFF_XLIM, major=MultipleLocator(0.02))
    axA.set_ylim(-3.10, 0.62)
    axA.text(DIFF_XLIM[0] + 0.003, -2.95,
             "P(Δ > 0) is shown where the corrected interval spans zero",
             fontsize=PS.BASE - 1.0, color=PS.GREY, va="center")
    axA.set_title("Qualification 1  ·  is the effect topological?",
                  fontsize=PS.BASE + 0.5, color=PS.INK, loc="left")
    axA.set_xlabel("Δ R², adjacent-pair log separation factor\n"
                   "positive favours the arm named first")

    # ------------------------------------------------------------------ B ---
    # The same candidates scored on their own.  Levels, so squares, own axis.
    lvl_rows = [("G0", 0.0, LABELS["G0"]), ("D0", -1.0, LABELS["D0"]),
                ("S0", -2.0, LABELS["S0"]),
                ("T0w", -3.3, "matched 2D control\n(same net, no encoder)")]
    for arm, y, _ in lvl_rows:
        level(axB, y, float(arm_level[arm]), BINNED)
    tidy(axB, [t for _, _, t in lvl_rows], [y for _, y, _ in lvl_rows], LVL_XLIM,
         major=FixedLocator([0.15, 0.20, 0.25, 0.30]))
    axB.set_ylim(-3.85, 0.62)
    PS.strapline(axB, "each third-arm candidate on its own")
    axB.set_xlabel("R², adjacent-pair log SF\na level, not a difference")

    # ------------------------------------------------------------------ C ---
    # The same two pre-registered stack contrasts under both definitions.
    dua_rows = [
        ("drop-in: does adding S0 to the best no-topology stack help?",
         "adding the 3D encoder to\nthe 2D-only pair", 0.0),
        ("swap: S0 vs the matched tabular control in the same slot",
         "3D encoder against the\nmatched 2D control", -2.0),
    ]
    keys = [("composition_key", BINNED, +0.36), ("strict_composition_key", STRICT, -0.36)]
    ypos_c, labels_c, drawn_c = [], [], []
    for question, label, y0 in dua_rows:
        for key, colour, dy in keys:
            r = dual[(dual["question"] == question) & (dual["key"] == key)].iloc[0]
            forest(axC, y0 + dy, r, dua_lo, dua_hi, colour)
            drawn_c.append({"question": question, "label": label.replace("\n", " "),
                            "key": key, "key_label": LABELS[key],
                            "delta": float(r["delta"]), "lo": float(r["lo"]),
                            "hi": float(r["hi"]), "lo_corrected": float(r[dua_lo]),
                            "hi_corrected": float(r[dua_hi]),
                            "p_better": float(r["p_better"]),
                            "verdict_corrected": str(r["verdict_corrected"])})
        ypos_c.append(y0)
        labels_c.append(label)
    axC.axvline(0, color=PS.INK, lw=0.9, zorder=2)

    # The two strict-key rows are the same number; say so in the artwork rather
    # than leaving a duplicate that reads as a mistake.
    same = float(dual[(dual["key"] == "strict_composition_key")]["delta"].iloc[0])
    axC.plot([same, same], [-0.36, -2.36], color=STRICT, lw=0.7, ls=(0, (1.6, 1.6)),
             zorder=3)
    axC.annotate("one number for\nboth questions", (same, -1.20),
                 textcoords="offset points", xytext=(-4, 0), ha="right", va="center",
                 fontsize=PS.BASE - 1.0, color=STRICT)
    tidy(axC, labels_c, ypos_c, DIFF_XLIM, major=MultipleLocator(0.02))
    axC.set_ylim(-3.10, 0.72)
    axC.set_title("Qualification 2  ·  how strictly must conditions match?",
                  fontsize=PS.BASE + 0.5, color=PS.INK, loc="left")
    axC.set_xlabel("Δ R², adjacent-pair log separation factor\n"
                   "positive favours the 3D encoder")

    # ------------------------------------------------------------------ D ---
    # Where the shrinkage comes from: every level falls, and the control collapses.
    stack_rows = [
        ("no topology (CatBoost+repaired)",
         "drop-in: does adding S0 to the best no-topology stack help?",
         "baseline_obs", "boosting + fingerprint net"),
        ("topology swapped for control",
         "swap: S0 vs the matched tabular control in the same slot",
         "baseline_obs", "+ matched 2D control"),
        ("full (CatBoost+repaired+S0)",
         "drop-in: does adding S0 to the best no-topology stack help?",
         "arm_obs", "+ the 3D encoder\n(edges + triangles)"),
    ]
    drawn_d = []
    for i, (_ident, question, column, label) in enumerate(stack_rows):
        y = -float(i)
        vals = {}
        for key, colour, _ in keys:
            r = dual[(dual["question"] == question) & (dual["key"] == key)].iloc[0]
            vals[key] = float(r[column])
        axD.plot([vals["strict_composition_key"], vals["composition_key"]], [y, y],
                 color=PS.PALE, lw=1.4, zorder=2)
        level(axD, y, vals["composition_key"], BINNED, side="right")
        level(axD, y, vals["strict_composition_key"], STRICT, side="left")
        drawn_d.append({"label": label, "source_column": column,
                        "binned": vals["composition_key"],
                        "strict": vals["strict_composition_key"]})
    tie = drawn_d[0]["strict"]
    axD.plot([tie, tie], [0.0, -1.0], color=STRICT, lw=0.7, ls=(0, (1.6, 1.6)),
             zorder=3)
    tidy(axD, [r["label"] for r in drawn_d], [0.0, -1.0, -2.0], LVL_XLIM,
         major=FixedLocator([0.15, 0.20, 0.25, 0.30]))
    axD.set_ylim(-3.55, 0.62)
    axD.text(LVL_XLIM[0] + 0.004, -2.55,
             "under strict matching the control is\ngiven zero weight in the fit, so "
             "the\ntwo upper rows are the same model",
             fontsize=PS.BASE - 1.0, color=PS.GREY, va="top")
    PS.strapline(axD, "the deployed combination")
    axD.set_xlabel("R², adjacent-pair log SF\na level, not a difference")

    # ------------------------------------------------------------------------
    handles = [
        Line2D([], [], color=BINNED, lw=2.6, marker="o", ms=4.6,
               markerfacecolor=BINNED, markeredgecolor=BINNED,
               label=LABELS["composition_key"]),
        Line2D([], [], color=STRICT, lw=2.6, marker="o", ms=4.6,
               markerfacecolor=STRICT, markeredgecolor=STRICT,
               label=LABELS["strict_composition_key"]),
        Line2D([], [], color=PS.GREY, lw=2.6,
               label="90 % interval, 400-draw cluster bootstrap"),
        Line2D([], [], color=PS.GREY, lw=0.9,
               label=f"after Bonferroni correction ({enc_looks} looks in A, "
                     f"{dua_looks} in C)"),
        Line2D([], [], color=PS.GREY, lw=0, marker="o", ms=4.6,
               markerfacecolor=PS.GREY, markeredgecolor=PS.GREY,
               label="clears zero after correction"),
        Line2D([], [], color=PS.GREY, lw=0, marker="o", ms=4.6,
               markerfacecolor="white", markeredgecolor=PS.GREY, markeredgewidth=1.0,
               label="spans zero after correction"),
    ]
    leg = fig.legend(handles=handles, loc="outside lower center", ncol=3,
                     fontsize=PS.BASE - 0.5, columnspacing=1.6, handletextpad=0.6,
                     title=COHORT)
    leg.get_title().set_fontsize(PS.BASE - 1.0)
    leg.get_title().set_color(PS.GREY)

    PS.add_panel_letters(fig, [axA, axB, axC, axD])

    # No repository identifier may reach the artwork.
    leaked = sorted({t.get_text() for t in fig.findobj(matplotlib.text.Text)
                     if t.get_visible() and BANNED.search(str(t.get_text()))})
    if leaked:
        raise SystemExit(f"repository identifiers in the artwork: {leaked}")

    report = PS.save(fig, HERE, "figure", strict=False)
    print(report)

    values = {
        "figure": "figure_03_limits",
        "source": {"repository": "github.com/mironovb/lanthanidestrain",
                   "script": "automl/figures_pi_email.py::fig_limits",
                   "tables": ["encoder_test.csv", "encoder_arms.csv",
                              "dualkey_test.csv"],
                   "provenance": LD.provenance()},
        "label_map": LABELS,
        "block_counts": N_BLOCKS,
        "bootstrap": {"n_boot": int(dual["n_boot"].iloc[0]),
                      "interval": "90 %", "cluster": "whole extractants",
                      "looks_panelA": enc_looks, "looks_panelC": dua_looks},
        "panelA_differences_binned_key": drawn_a,
        "panelB_levels_binned_key": [
            {"arm": a, "label": LABELS[a], "adj_r2_binned": float(arm_level[a])}
            for a, _, _ in lvl_rows],
        "panelC_differences_both_keys": drawn_c,
        "panelD_levels_both_keys": drawn_d,
        "lint": {"ok": report.ok, "violations": [str(v) for v in report.violations]},
    }
    (HERE / "values.json").write_text(json.dumps(values, indent=1))
    for row in drawn_a:
        print(f"A  {row['label'][:34]:34s} {row['delta']:+.4f}  {row['verdict_corrected']}")
    for row in drawn_c:
        print(f"C  {row['key'][:24]:24s} {row['delta']:+.4f}  {row['verdict_corrected']}")
    for row in drawn_d:
        print(f"D  {row['label'][:24]:24s} {row['binned']:+.4f} -> {row['strict']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
