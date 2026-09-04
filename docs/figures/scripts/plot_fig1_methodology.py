"""Figure 1 — task, held-out protocol and model.

A schematic: it contains no fitted numbers.  The only quantities printed on it are
cohort counts, which are read from the frozen run outputs rather than typed in:

  runs/gen9_shape/recomposed/oof_GEN9_SHAPE_RECOMPOSED.parquet  rows / extractants /
                                                                clusters / chemotypes
  runs/gen9_shape/curves/curve_table.parquet                    titration curves
  figures/derived/kshot_cohort.json                             k-shot cohort and repeats

Run from the repository root:
    ``.venv/bin/python figures/scripts/plot_fig1_methodology.py``
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _paths  # noqa: E402
import _style  # noqa: E402

TRAIN = _style.BLUE          # information the global model is fitted on
CONDS = _style.SKY           # target-free information available at prediction time
SUPPORT = _style.VERMILLION  # the k measured support points of the held-out extractant
QUERY = "#3A3A3A"            # held-out query rows, never read
NEUTRAL = "#F2F2F2"


def box(ax, x, y, w, h, text, *, edge, face="white", fontsize=6.3, weight="regular",
        align="center", lw=0.9, zorder=3, radius=0.012):
    patch = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                           linewidth=lw, edgecolor=edge, facecolor=face, zorder=zorder)
    ax.add_patch(patch)
    ha = {"center": "center", "left": "left"}[align]
    tx = x + w / 2 if align == "center" else x + 0.012
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fontsize,
            fontweight=weight, zorder=zorder + 1, linespacing=1.35)
    return patch


def arrow(ax, start, end, *, colour="#555555", lw=0.9, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=6.5,
                                 linewidth=lw, color=colour, linestyle=ls,
                                 shrinkA=1.5, shrinkB=1.5, zorder=2))


def counts() -> dict:
    oof = pd.read_parquet(_paths.run("gen9_shape/recomposed/oof_GEN9_SHAPE_RECOMPOSED.parquet"),
                          columns=["row_id", "extractant", "ecfp_cluster", "tanimoto_cluster",
                                   "series_id", "metal_symbol", "split_seed", "condition_id"])
    one = oof[oof.split_seed == oof.split_seed.min()]
    curves = pd.read_parquet(_paths.run("gen9_shape/curves/curve_table.parquet"))
    path = _paths.DERIVED / "kshot_cohort.json"
    if not path.exists():
        subprocess.run([sys.executable, str(Path(__file__).with_name("prepare_kshot_tables.py"))],
                       check=True)
    kshot = json.loads(path.read_text())
    return {"rows": int(len(one)), "extractants": int(one.extractant.nunique()),
            "ecfp_clusters": int(one.ecfp_cluster.nunique()),
            "chemotypes": int(one.tanimoto_cluster.nunique()),
            "metals": int(one.metal_symbol.nunique()),
            "series": int(one.series_id.nunique()),
            "conditions": int(one.condition_id.nunique()),
            "curves": int(len(curves)),
            "extractant_curves": int((curves.axis_label == "extractant").sum()),
            "kshot_ligands": kshot["n_ligands_common"],
            "seeds": len(kshot["seeds"]), "repeats": kshot["repeats_per_seed"]}


def main() -> int:
    _style.apply()
    n = counts()

    fig = plt.figure(figsize=(_style.W2, 4.75))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.86, 1.14], width_ratios=[1.0, 1.20],
                          hspace=0.04, wspace=0.08,
                          left=0.012, right=0.988, top=0.965, bottom=0.012)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, :])
    for ax in (axA, axB, axC):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    # ================================================================= A =======
    _style.panel(axA, "A", dx=0.0, dy=1.0)
    axA.text(0.10, 0.97, "Corpus and held-out split", fontsize=7.2, fontweight="bold",
             va="top")
    box(axA, 0.04, 0.71, 0.92, 0.19,
        f"{n['rows']:,} measurements of log$_{{10}}$ $D$\n"
        f"{n['extractants']} extractants  ·  {n['metals']} lanthanides  ·  "
        f"{n['conditions']:,} condition cells",
        edge=TRAIN, face="#EAF2F8")
    arrow(axA, (0.5, 0.71), (0.5, 0.645))
    axA.text(0.5, 0.625, f"group by Tanimoto-0.7 chemotype  ({n['chemotypes']} groups)",
             ha="center", fontsize=6.3, va="top")

    xs = np.linspace(0.06, 0.60, 4)
    for x in xs:
        box(axA, x, 0.42, 0.125, 0.125, "", edge=TRAIN, face="#D6E6F2", lw=0.8)
    axA.text(xs[0] + 0.2525, 0.395, "training chemotypes", fontsize=6.2, ha="center",
             va="top", color=TRAIN)
    box(axA, 0.81, 0.42, 0.145, 0.125, "", edge=QUERY, face="#EDEDED", lw=1.1)
    axA.text(0.8825, 0.395, "held-out\nchemotype", fontsize=6.2, ha="center", va="top",
             color=QUERY)
    arrow(axA, (0.735, 0.4825), (0.805, 0.4825), colour=QUERY, style="-|>")

    axA.text(0.04, 0.29, "Every extractant within Tanimoto 0.7 of a held-out extractant is held\n"
                         "out with it, so the model has seen neither the extractant nor a close\n"
                         "analogue. 5 outer folds × 5 split seeds; the model seed is fixed at 42\n"
                         "and independent of the split.",
             fontsize=6.1, va="top", color="#333333", linespacing=1.55)

    # ================================================================= B =======
    _style.panel(axB, "B", dx=0.0, dy=1.0)
    axB.text(0.07, 0.97, "Global model, fitted on training chemotypes only",
             fontsize=7.2, fontweight="bold", va="top")

    box(axB, 0.01, 0.68, 0.29, 0.21,
        "design matrix\nECFP + descriptors,\nlanthanide, conditions,\nmass-action terms",
        edge=TRAIN, face="#EAF2F8", fontsize=6.0)
    box(axB, 0.01, 0.45, 0.29, 0.16,
        "position of the row\ninside its own\ntitration window",
        edge=CONDS, face="#EAF7FD", fontsize=6.0)
    axB.text(0.155, 0.425, "target-free: taken from the candidate condition list",
             ha="center", va="top", fontsize=5.8, color=CONDS)

    box(axB, 0.385, 0.74, 0.26, 0.14, "forest 1\ntarget: log$_{10}$ $D$",
        edge=TRAIN, face="white", fontsize=6.0)
    box(axB, 0.385, 0.45, 0.26, 0.16,
        "forest 2\ntarget: log$_{10}$ $D$ $-$\nmean over its curve",
        edge=TRAIN, face="white", fontsize=6.0)
    arrow(axB, (0.30, 0.80), (0.385, 0.81), colour=TRAIN)
    arrow(axB, (0.30, 0.72), (0.385, 0.56), colour=TRAIN)
    arrow(axB, (0.30, 0.53), (0.385, 0.52), colour=CONDS)

    box(axB, 0.72, 0.58, 0.27, 0.20,
        "recomposition\ncurve mean of forest 1\n$+$ shape of forest 2",
        edge=TRAIN, face="#EAF2F8", fontsize=6.0)
    arrow(axB, (0.645, 0.80), (0.72, 0.72), colour=TRAIN)
    arrow(axB, (0.645, 0.53), (0.72, 0.62), colour=TRAIN)

    axB.text(0.01, 0.32, "The recomposition is mean-preserving: every curve keeps forest 1's\n"
                         "average prediction, so the per-extractant level is bit-identical to the\n"
                         "baseline model and only the within-curve shape changes.",
             fontsize=6.1, va="top", color="#333333", linespacing=1.55)

    # ================================================================= C =======
    _style.panel(axC, "C", dx=0.0, dy=1.0)
    axC.text(0.045, 0.97, "Deployment and evaluation on one held-out extractant",
             fontsize=7.2, fontweight="bold", va="top")

    flow_y, flow_h = 0.665, 0.215
    box(axC, 0.02, flow_y, 0.155, flow_h,
        "user supplies a\ncandidate list of\nconditions to run",
        edge=CONDS, face="#EAF7FD", fontsize=6.0)
    arrow(axC, (0.175, flow_y + flow_h / 2), (0.222, flow_y + flow_h / 2), colour=CONDS)
    box(axC, 0.222, flow_y, 0.148, flow_h,
        "zero-shot\nprediction for\nevery candidate",
        edge=TRAIN, face="white", fontsize=6.0)
    arrow(axC, (0.370, flow_y + flow_h / 2), (0.417, flow_y + flow_h / 2))
    box(axC, 0.417, flow_y, 0.158, flow_h,
        "acquisition policy\npicks $k$ rows —\nnever reads a target",
        edge=SUPPORT, face="white", fontsize=6.0)
    arrow(axC, (0.575, flow_y + flow_h / 2), (0.622, flow_y + flow_h / 2), colour=SUPPORT)
    box(axC, 0.622, flow_y, 0.155, flow_h,
        "measure those $k$\nrows in the lab\n(support points)",
        edge=SUPPORT, face="#FCEDE3", fontsize=6.0)
    arrow(axC, (0.777, flow_y + flow_h / 2), (0.824, flow_y + flow_h / 2), colour=SUPPORT)
    box(axC, 0.824, flow_y, 0.158, flow_h,
        "calibrate the frozen\nmodel: shrunk offset\n+ series terms",
        edge=SUPPORT, face="#FCEDE3", fontsize=6.0)

    axC.text(0.045, 0.615, "the extractant's rows, split once per repeat:",
             fontsize=6.3, va="top")
    strip_y, strip_h = 0.435, 0.115
    n_pool, n_query = 8, 6
    x0, w, gap = 0.045, 0.034, 0.006
    for i in range(n_pool):
        chosen = i in (2, 5)
        box(axC, x0 + i * (w + gap), strip_y, w, strip_h, "",
            edge=SUPPORT if chosen else "#9C9C9C",
            face="#FCEDE3" if chosen else "white", lw=1.0 if chosen else 0.7)
    pool_right = x0 + n_pool * (w + gap) - gap
    axC.text((x0 + pool_right) / 2, strip_y - 0.022, "candidate pool  ($k$ selected)",
             ha="center", va="top", fontsize=6.1, color=SUPPORT)
    x1 = pool_right + 0.055
    for i in range(n_query):
        box(axC, x1 + i * (w + gap), strip_y, w, strip_h, "", edge=QUERY,
            face="#E4E4E4", lw=0.8)
    query_right = x1 + n_query * (w + gap) - gap
    axC.text((x1 + query_right) / 2, strip_y - 0.022,
             "held-out query rows — scored, never read", ha="center", va="top",
             fontsize=6.1, color=QUERY)
    axC.plot([pool_right + 0.027, pool_right + 0.027],
             [strip_y - 0.008, strip_y + strip_h + 0.008],
             color="#BBBBBB", linewidth=0.8, linestyle=(0, (2, 2)))

    arrow(axC, (0.903, flow_y), (0.903, strip_y + strip_h + 0.035), colour=SUPPORT)
    box(axC, 0.824, strip_y - 0.015, 0.158, 0.145,
        "macro MAE over the\nquery rows, one vote\nper extractant",
        edge=QUERY, face="#EDEDED", fontsize=6.0)

    axC.text(0.045, 0.345,
             f"$k \\in \\{{0,1,2,3,5\\}}$.  {n['kshot_ligands']} held-out extractants with at "
             f"least 5 pool rows and 2 query rows in every arm; {n['seeds']} split seeds × "
             f"{n['repeats']} pool draws.\n"
             "Support targets are read only by the calibrator, and a support row is never also "
             "scored as a query row. The model itself is frozen —\n"
             "it is never refitted on the held-out extractant — so the $k$ measurements buy "
             "calibration, not training.",
             fontsize=6.1, va="top", color="#333333", linespacing=1.55)

    handles = [
        plt.Line2D([], [], marker="s", linestyle="none", markersize=5.2,
                   markerfacecolor="#D6E6F2", markeredgecolor=TRAIN,
                   label="training information (other chemotypes)"),
        plt.Line2D([], [], marker="s", linestyle="none", markersize=5.2,
                   markerfacecolor="#EAF7FD", markeredgecolor=CONDS,
                   label="target-free conditions of the held-out extractant"),
        plt.Line2D([], [], marker="s", linestyle="none", markersize=5.2,
                   markerfacecolor="#FCEDE3", markeredgecolor=SUPPORT,
                   label="test-time support: $k$ measured targets"),
        plt.Line2D([], [], marker="s", linestyle="none", markersize=5.2,
                   markerfacecolor="#E4E4E4", markeredgecolor=QUERY,
                   label="held-out query targets: scored, never read"),
    ]
    axC.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.028, -0.02),
               ncol=2, fontsize=6.0, handletextpad=0.5, columnspacing=1.6)

    written = _style.save(fig, "Fig1_methodology", _paths.MAIN)
    (_paths.DERIVED / "fig1_counts.json").write_text(json.dumps(n, indent=1))
    print("\n".join(str(p) for p in written))
    print(json.dumps(n, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
