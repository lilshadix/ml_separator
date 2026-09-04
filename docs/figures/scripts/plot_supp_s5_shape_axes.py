"""Figure S5 — the compression is worst on the extractant axis but present on all of them.

Source: ``runs/gen9_shape/shape/shape_by_axis.csv`` (5 split seeds).  The extractant axis
is the main text's Figure 3; this shows the other five condition axes on the same scale.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _paths, _style  # noqa: E402

STAGES = [("REC_ecfp_plus_recovered", "frozen", "baseline model"),
          ("GEN9_REL_MONOLITH", "relpos", "+ relative position"),
          ("GEN9_SHAPE_RECOMPOSED", "recomposed", "+ recomposition")]
AXES = ["extractant", "acid", "metal_series", "temperature", "metal_concentration",
        "contact_time"]


def main() -> int:
    _style.apply()
    table = pd.read_csv(_paths.run("gen9_shape/shape/shape_by_axis.csv"))
    table = table[table.model.isin([m for m, _, _ in STAGES])]

    metrics = [("span_recovery_median_unguarded", "span recovery (median)", (0, 1.0)),
               ("shape_mae", "shape MAE (mean)", None),
               ("within_curve_spearman", "within-curve Spearman (mean)", (-0.05, 1.0))]
    fig, axes = plt.subplots(1, 3, figsize=(_style.W2, 2.5))
    fig.subplots_adjust(left=0.135, right=0.985, top=0.88, bottom=0.30, wspace=0.16)

    y = np.arange(len(AXES))[::-1]
    height = 0.24
    record = []
    for ax, (column, xlabel, xlim) in zip(axes, metrics):
        for j, (model, style_key, nice) in enumerate(STAGES):
            values, ns = [], []
            for axis in AXES:
                row = table[(table.model == model) & (table.axis_label == axis)]
                values.append(float(row[column].iloc[0]))
                ns.append(int(row.n_curves.iloc[0]))
                record.append({"axis": axis, "model": model, "metric": column,
                               "value": values[-1], "n_curves_x_seeds": ns[-1],
                               "n_ligands": int(row.n_ligands.iloc[0])})
            ax.barh(y + (1 - j) * height, values, height=height,
                    color=_style.colour(style_key), alpha=0.9, linewidth=0, label=nice)
        ax.set_xlabel(xlabel, fontsize=6.6)
        if xlim:
            ax.set_xlim(*xlim)
        ax.set_yticks(y)
    axes[0].set_yticklabels([a.replace("_", " ") for a in AXES], fontsize=6.2)
    for ax in axes[1:]:
        ax.set_yticklabels([])
    axes[0].legend(loc="lower center", bbox_to_anchor=(1.6, -0.42), ncol=3, fontsize=6.2)
    for ax, letter in zip(axes, "ABC"):
        _style.panel(ax, letter, dx=-0.10 if letter != "A" else -0.44)

    written = _style.save(fig, "FigS5_shape_by_axis", _paths.SUPP)
    (_paths.DERIVED / "figS5_values.json").write_text(json.dumps(record, indent=1))
    print("\n".join(str(p) for p in written))
    print(pd.DataFrame(record).pivot_table(index=["axis", "metric"], columns="model",
                                           values="value").round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
