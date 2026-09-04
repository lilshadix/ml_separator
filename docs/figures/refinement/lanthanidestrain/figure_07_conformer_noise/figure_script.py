#!/usr/bin/env python
"""Figure 7 — how much of a 3-D descriptor's variation along the lanthanide series
is a smooth response to cation size, and how much is scatter.

Refined from ``automl/figures.py::fig_noise_diagnostic`` of `lanthanidestrain`
(original render ``automl/reports/figures/fig3_conformer_noise.png``, discussed in
``automl/reports/FINDINGS.md`` §4.2 and ``automl/reports/CONFORMER_RESULTS.md``).

The quantity is the one the upstream table already carries and nothing is recomputed:
inside one ligand + anion family — the same extractant and anion, the lanthanide
varying — every 3-D descriptor is fitted with a straight line against the Shannon
ionic radius (``automl.dataset.add_series_smoothed``, block ``g13__fitr2``).  The fit
R2 is the share of that descriptor's within-family variation that is a systematic
response to cation size; the remainder is scatter between single conformers.

Panel A plots the vendored ``noise_by_block`` medians — the nine numbers the original
figure showed — and adds the per-descriptor values behind them from
``noise_by_feature``, which is the table ``noise_by_block`` is the block aggregate of
(verified to 6e-17 at run time).  Panel B pools the same 290 descriptors.

Run from anywhere:
    python figure_refinement/lanthanidestrain/figure_07_conformer_noise/figure_script.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
from matplotlib.lines import Line2D      # noqa: E402
from matplotlib.patches import Patch     # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common"))
import lstrain_data as LD                # noqa: E402
import pubstyle as PS                    # noqa: E402

# --------------------------------------------------------------------------- #
# Semantic colours.  Three roles, fixed in both panels:
#   SIGNAL  the share of the variation that is a smooth response to ionic radius
#   NOISE   the remainder — scatter, not a size response
#   ITEM    one individual descriptor (dots in A, histogram in B)
# The repository's own palette is deliberately not reused: its `test_palette.py`
# found that no four-colour subset of it passes CVD and contrast checks at once.
# --------------------------------------------------------------------------- #
SIGNAL, NOISE, ITEM, FLAG = PS.BLUE, PS.PALE, PS.SKY, PS.VERMILLION

#: Repository block identifier -> reader-facing name.  Every phrase is taken from the
#: block's own docstring in `automl/geom3d_features.py` (module header, G1-G9) and is
#: reproduced in notes.md.  The `gN__` prefixes never reach the artwork.
BLOCK_LABEL = {
    "first_shell":  "First coordination shell",
    "contraction":  "Shell minus the ionic radius",
    "polyhedron":   "Donor polyhedron shape",
    "steric":       "Steric bulk at the metal",
    "electronic":   "xTB charges, dipole and forces",
    "rdf":          "Metal-centred radial distribution",
    "global_shape": "Whole-complex shape and surface",
    "chelate":      "Chelate rings and bite angles",
    "topology":     "Persistent topology",
}

#: The one descriptor in the table that is the fit's own predictor.  `add_series_smoothed`
#: fits every numeric column, including `g2__contraction__ionic_radius` itself, so its
#: R2 is 1.000 by construction.  It is plotted, flagged and left in the block median
#: exactly as published — see notes.md, "Scientific integrity".
SELF_FIT = "g13__fitr2__g2__contraction__ionic_radius"

#: More than half the within-family variation is a size response.
MAJORITY = 0.5


def load() -> tuple:
    """Vendored tables, with the block table checked against the per-descriptor one."""
    block = LD.read("noise_by_block")
    feat = LD.read("noise_by_feature")
    finite = feat.dropna(subset=["fit_r2"]).copy()
    agg = (finite.groupby("block")["fit_r2"]
           .agg(["mean", "median", "count"]).reset_index())
    check = block.merge(agg, on="block", suffixes=("_pub", "_rec"))
    if len(check) != len(block):
        raise SystemExit("noise_by_block and noise_by_feature disagree on the blocks")
    worst = max(float((check["median_pub"] - check["median_rec"]).abs().max()),
                float((check["mean_pub"] - check["mean_rec"]).abs().max()))
    if worst > 1e-12 or not (check["count_pub"] == check["count_rec"]).all():
        raise SystemExit(f"noise_by_feature does not aggregate to noise_by_block "
                         f"(worst {worst:.2e})")
    print(f"noise_by_feature aggregates to noise_by_block: max |diff| = {worst:.1e}, "
          f"counts identical for all {len(block)} blocks")
    return block.sort_values("median").reset_index(drop=True), finite, feat


def main() -> int:
    PS.apply()
    block, feat, feat_all = load()
    n_total = int(len(feat))
    n_undefined = int(len(feat_all) - n_total)

    fig = plt.figure(figsize=(PS.W15, 5.05))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.55, 1.00])
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[1, 0], sharex=axA)

    # ------------------------------------------------------------------ A ---
    # Row i: a bar band carrying the block median (0 -> median = size response,
    # median -> 1 = the remainder) and, directly under it, a dot band carrying
    # every descriptor in the block.  The two bands never overlap, so a value
    # label inside a bar cannot land on a dot.
    rng = np.random.default_rng(0)          # jitter only; no statistic depends on it
    BAR_LO, BAR_HI = 0.04, 0.40             # bar band, in row units
    DOT_C, DOT_J = -0.17, 0.105             # dot band centre and half-jitter
    for i, row in block.iterrows():
        med = float(row["median"])
        axA.barh(i + (BAR_LO + BAR_HI) / 2, 1.0, height=BAR_HI - BAR_LO,
                 color=NOISE, linewidth=0, zorder=2)
        axA.barh(i + (BAR_LO + BAR_HI) / 2, med, height=BAR_HI - BAR_LO,
                 color=SIGNAL, linewidth=0, zorder=3)
        axA.text(med - 0.014, i + (BAR_LO + BAR_HI) / 2, f"{med:.2f}",
                 ha="right", va="center", fontsize=PS.BASE - 1.3, color="white",
                 zorder=4)
        axA.text(0.988, i + (BAR_LO + BAR_HI) / 2, f"n = {int(row['count'])}",
                 ha="right", va="center", fontsize=PS.BASE - 1.5, color=PS.INK,
                 zorder=4)
        vals = feat.loc[feat["block"] == row["block"]]
        flag = vals["feature"] == SELF_FIT
        y = i + DOT_C + rng.uniform(-DOT_J, DOT_J, len(vals))
        axA.scatter(vals.loc[~flag, "fit_r2"], y[~flag.to_numpy()], s=6.5,
                    color=ITEM, alpha=0.80, linewidths=0, zorder=3)
        if flag.any():
            # Unjittered, so the label below can be placed against it; clip_on=False so
            # the marker at exactly 1.0 is drawn whole, not halved by the axis spine.
            axA.scatter(vals.loc[flag, "fit_r2"], [i + DOT_C], s=15,
                        facecolors="none", edgecolors=FLAG, linewidths=0.9, zorder=5,
                        clip_on=False)
            # Sits directly above its own marker, so it needs no leader line.
            axA.text(1.005, i + DOT_C - 0.17, "the ionic radius,\nfitted against itself",
                     ha="right", va="bottom", fontsize=PS.BASE - 1.5, color=FLAG,
                     linespacing=1.20, clip_on=False)

    axA.set_yticks(np.arange(len(block)))
    axA.set_yticklabels([BLOCK_LABEL[b] for b in block["block"]])
    # The band below the last row is reserved for the key, so it cannot cover data.
    axA.set_ylim(-1.15, len(block) - 1 + 0.40 + 1.95)
    axA.set_xlim(0, 1.0)
    axA.set_xticks(np.arange(0, 1.01, 0.2))
    axA.invert_yaxis()
    axA.tick_params(axis="y", length=0)
    axA.tick_params(axis="x", length=0)
    # No x axis on A: the pale bar *is* the 0-to-1 frame, and an unlabelled tick strip
    # directly above panel B reads as B's own top frame.
    axA.spines["left"].set_visible(False)
    axA.spines["bottom"].set_visible(False)
    PS.strapline(axA, "one row per block, ordered by median;  n = descriptors\n"
                      "a family is one extractant and anion, with 4 to 14 lanthanides")

    handles = [Patch(facecolor=SIGNAL,
                     label="block median — a straight-line response to ionic radius"),
               Patch(facecolor=NOISE, label="the remainder — scatter about that line"),
               Line2D([], [], marker="o", color=ITEM, linestyle="none", markersize=3.0,
                      label="one descriptor")]
    axA.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.005),
               ncol=1, borderpad=0.0, handlelength=1.3, labelspacing=0.28)

    # ------------------------------------------------------------------ B ---
    bins = np.arange(0, 1.0001, 0.05)
    counts, _, _ = axB.hist(feat["fit_r2"], bins=bins, color=ITEM, linewidth=0,
                            zorder=3)
    top = float(counts.max())
    axB.set_ylim(0, top * 1.42)
    n_major = int((feat["fit_r2"] > MAJORITY).sum())
    axB.axvline(MAJORITY, color=PS.INK, linestyle=(0, (3, 2)), linewidth=0.8, zorder=4)
    axB.annotate(f"{n_major} of {n_total} descriptors "
                 f"({100 * n_major / n_total:.0f} %) are\n"
                 f"more size response than scatter",
                 xy=(MAJORITY + 0.018, top * 1.36), ha="left", va="top",
                 fontsize=PS.BASE - 1.3, color=PS.INK, linespacing=1.25)
    axB.set_ylabel("descriptors")
    axB.set_yticks([0, 30, 60])
    PS.strapline(axB, f"all {n_total} descriptors pooled")

    axB.set_xlabel("share of a descriptor's within-family variation explained by a\n"
                   "straight line in Shannon ionic radius   "
                   "(fit R\u00b2;  0 = none,  1 = all)")
    axB.set_xlim(0, 1.0)
    plt.setp(axA.get_xticklabels(), visible=False)

    # Both panels sit in one gridspec column, so their data areas must line up.
    fig.canvas.draw()
    pa, pb = axA.get_position(), axB.get_position()
    if abs(pa.x0 - pb.x0) > 1e-6 or abs(pa.x1 - pb.x1) > 1e-6:
        raise SystemExit(f"panels A and B are not x-aligned: {pa.x0:.5f}-{pa.x1:.5f} "
                         f"vs {pb.x0:.5f}-{pb.x1:.5f}")
    print(f"panels A and B share the x frame: x0={pa.x0:.4f} x1={pa.x1:.4f}")

    letters = PS.add_panel_letters(fig, [axA, axB])
    # Both letters to the leftmost of the two, so the panel column is not ragged.
    x_left = min(t.get_position()[0] for t in letters)
    for t in letters:
        t.set_position((x_left, t.get_position()[1]))
    report = PS.save(fig, HERE, "figure", strict=False)
    print(report)

    values = {
        "quantity": ("per-descriptor R2 of a straight-line fit against the Shannon "
                     "ionic radius, inside one ligand+anion family "
                     "(automl.dataset.add_series_smoothed, block g13__fitr2)"),
        "source_tables": ["noise_by_block.csv", "noise_by_feature.csv"],
        "provenance": LD.provenance(),
        "n_descriptors_plotted": n_total,
        "n_descriptors_undefined": n_undefined,
        "pooled": {"median": float(feat["fit_r2"].median()),
                   "mean": float(feat["fit_r2"].mean()),
                   "n_above_0.5": n_major,
                   "n_above_0.4": int((feat["fit_r2"] > 0.4).sum()),
                   "n_above_0.25": int((feat["fit_r2"] > 0.25).sum())},
        "blocks": [
            {"block": r["block"], "label": BLOCK_LABEL[r["block"]],
             "median": float(r["median"]), "mean": float(r["mean"]),
             "n_descriptors": int(r["count"]),
             "min": float(feat.loc[feat["block"] == r["block"], "fit_r2"].min()),
             "max": float(feat.loc[feat["block"] == r["block"], "fit_r2"].max())}
            for _, r in block.iterrows()],
        "self_fit_descriptor": {
            "feature": SELF_FIT, "fit_r2": 1.0, "block": "contraction",
            "note": ("the fit's own predictor; R2 = 1 by construction, retained in the "
                     "published block mean (0.277) and median (0.194) unchanged")},
        "block_label_map": BLOCK_LABEL,
        "lint_ok": report.ok,
    }
    (HERE / "values.json").write_text(json.dumps(values, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
