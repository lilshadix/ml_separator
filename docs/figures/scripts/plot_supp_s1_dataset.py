"""Figure S1 — what the corpus contains.

Source: ``runs/gen9_shape/recomposed/oof_GEN9_SHAPE_RECOMPOSED.parquet`` (one split seed;
the cohort is the same in every seed) and ``runs/gen9_shape/curves/curve_table.parquet``.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _paths, _style  # noqa: E402

LN_ORDER = ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]


def main() -> int:
    _style.apply()
    oof = pd.read_parquet(_paths.run("gen9_shape/recomposed/oof_GEN9_SHAPE_RECOMPOSED.parquet"))
    one = oof[oof.split_seed == oof.split_seed.min()]
    curves = pd.read_parquet(_paths.run("gen9_shape/curves/curve_table.parquet"))

    fig, axes = plt.subplots(2, 2, figsize=(_style.W2, 4.3))
    (axA, axB), (axC, axD) = axes
    fig.subplots_adjust(left=0.085, right=0.985, top=0.93, bottom=0.10, hspace=0.55, wspace=0.30)

    counts = one.metal_symbol.value_counts().reindex(LN_ORDER).fillna(0)
    axA.bar(np.arange(len(LN_ORDER)), counts.to_numpy(), color=_style.BLUE, alpha=0.85,
            linewidth=0, width=0.72)
    axA.set_xticks(np.arange(len(LN_ORDER)))
    axA.set_xticklabels(LN_ORDER, fontsize=6.0)
    axA.set_ylabel("measurements")
    axA.set_xlabel("lanthanide")
    axA.text(0.98, 0.95, f"Pm absent (no stable isotope)\n{int(counts.sum()):,} rows",
             transform=axA.transAxes, va="top", ha="right", fontsize=6.0, color=_style.GREY)
    _style.panel(axA, "A", dx=-0.16)

    per_lig = one.groupby("extractant").size().sort_values(ascending=False)
    axB.plot(np.arange(1, len(per_lig) + 1), per_lig.to_numpy(), color=_style.BLUE,
             linewidth=1.2)
    axB.set_yscale("log")
    axB.set_xlabel("extractant, ranked")
    axB.set_ylabel("measurements (log scale)")
    axB.axhline(10, color=_style.PURPLE, linewidth=0.8, linestyle="--")
    axB.text(len(per_lig), 11, "10 rows", ha="right", fontsize=6.0, color=_style.PURPLE)
    axB.text(0.35, 0.92, f"median {int(per_lig.median())} rows;\nlargest "
                         f"{int(per_lig.iloc[0]):,} = {per_lig.iloc[0] / len(one):.0%} of the corpus",
             transform=axB.transAxes, va="top", fontsize=6.0, color=_style.GREY)
    _style.panel(axB, "B", dx=-0.20)

    chem = one.groupby("tanimoto_cluster").size().sort_values(ascending=False)
    share = (chem / chem.sum()).cumsum().to_numpy()
    axC.plot(np.arange(1, len(chem) + 1), share, color=_style.BLUE, linewidth=1.2)
    axC.set_xlabel("Tanimoto-0.7 chemotype, ranked")
    axC.set_ylabel("cumulative share of rows")
    axC.set_ylim(0, 1.02)
    axC.axhline(share[0], color=_style.VERMILLION, linewidth=0.8, linestyle="--")
    axC.text(len(chem), share[0] - 0.04, f"largest chemotype = {share[0]:.0%} of rows",
             ha="right", va="top", fontsize=6.0, color=_style.VERMILLION)
    axC.text(0.5, 0.3, f"{len(chem)} chemotypes\n{one.ecfp_cluster.nunique()} ECFP clusters",
             transform=axC.transAxes, fontsize=6.0, color=_style.GREY)
    _style.panel(axC, "C", dx=-0.16)

    axis_counts = curves.axis_label.value_counts()
    labels = list(axis_counts.index)
    axD.barh(np.arange(len(labels))[::-1], axis_counts.to_numpy(), height=0.62,
             color=_style.BLUE, alpha=0.85, linewidth=0)
    for y, (name, value) in zip(np.arange(len(labels))[::-1], axis_counts.items()):
        pts = curves[curves.axis_label == name].n_points
        axD.text(value + 12, y, f"{int(value)} curves, median {int(pts.median())} points",
                 va="center", fontsize=5.9, color=_style.GREY)
    axD.set_yticks(np.arange(len(labels))[::-1])
    axD.set_yticklabels([l.replace("_", " ") for l in labels], fontsize=6.2)
    axD.set_xlabel("titration / series curves")
    axD.set_xlim(0, axis_counts.max() * 2.4)
    _style.panel(axD, "D", dx=-0.30)

    written = _style.save(fig, "FigS1_dataset_composition", _paths.SUPP)
    record = {"figure": "FigS1_dataset_composition", "n_rows": int(len(one)),
              "n_extractants": int(one.extractant.nunique()),
              "n_ecfp_clusters": int(one.ecfp_cluster.nunique()),
              "n_chemotypes": int(one.tanimoto_cluster.nunique()),
              "rows_per_metal": {k: int(v) for k, v in counts.items()},
              "largest_extractant_share": float(per_lig.iloc[0] / len(one)),
              "largest_chemotype_share": float(share[0]),
              "median_rows_per_extractant": float(per_lig.median()),
              "curves_by_axis": {k: int(v) for k, v in axis_counts.items()},
              "log_D_range": [float(one.log_D.min()), float(one.log_D.max())],
              "log_D_sd": float(one.log_D.std())}
    (_paths.DERIVED / "figS1_values.json").write_text(json.dumps(record, indent=1))
    print("\n".join(str(p) for p in written)); print(json.dumps(record, indent=1)[:900])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
