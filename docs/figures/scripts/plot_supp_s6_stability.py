"""Figure S6 — per-split-seed stability of every headline quantity.

A single seed is never quoted as a result anywhere in the paper; this shows the spread
behind the means.  Sources: ``figures/derived/kshot_per_seed.csv`` (built from
``runs/gen10_final/final_locked/kshot_detail.parquet``) and
``runs/gen9_shape/shape/curve_shape.parquet``.
"""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _paths, _style  # noqa: E402

RULES = {"frozen": ("REC_ecfp_plus_recovered",
                    {0: ("ZERO_SHOT_REF", "NONE"), 1: ("SLOPE_L_s1_K1", "CENTRAL_THEN_SPREAD"),
                     2: ("SLOPE_L_s1_K3", "CENTRAL_THEN_SPREAD"),
                     3: ("SLOPE_L_s1_K3", "CENTRAL_THEN_SPREAD"),
                     5: ("SLOPE_L_s1_K3", "CENTRAL_THEN_SPREAD")}),
         "pipeline": ("GEN9_SHAPE_RECOMPOSED",
                      {0: ("ZERO_SHOT_REF", "NONE"), 1: ("SLOPE_L_s1_K1", "CENTRAL_THEN_SPREAD"),
                       2: ("SERIES_ML", "CENTRAL_THEN_SPREAD"),
                       3: ("SERIES_ML", "CENTRAL_THEN_SPREAD"),
                       5: ("SERIES_ML", "CENTRAL_THEN_SPREAD")})}
STAGES = [("REC_ecfp_plus_recovered", "frozen"), ("GEN9_REL_MONOLITH", "relpos"),
          ("GEN9_SHAPE_RECOMPOSED", "recomposed")]


def main() -> int:
    _style.apply()
    path = _paths.DERIVED / "kshot_per_seed.csv"
    if not path.exists():
        subprocess.run([sys.executable, str(Path(__file__).with_name("prepare_kshot_tables.py"))],
                       check=True)
    per_seed = pd.read_csv(path)
    seeds = sorted(per_seed.split_seed.unique())

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(_style.W2, 2.5))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.90, bottom=0.20, wspace=0.28)

    record = []
    markers = ["o", "s", "^", "D", "v"]
    for key, (model, rule) in RULES.items():
        for i, seed in enumerate(seeds):
            values = []
            for k, (adapter, policy) in rule.items():
                row = per_seed[(per_seed.global_model == model) & (per_seed.adapter == adapter)
                               & (per_seed.policy == policy) & (per_seed.k == k)
                               & (per_seed.split_seed == seed)]
                values.append(float(row.mae.iloc[0]))
                record.append({"rule": key, "split_seed": int(seed), "k": int(k),
                               "macro_mae": values[-1]})
            axA.plot(list(rule), values, color=_style.colour(key), alpha=0.55,
                     linewidth=0.7, marker=markers[i], markersize=2.4)
    for key in RULES:
        axA.plot([], [], color=_style.colour(key), linewidth=1.4,
                 label={"frozen": "baseline model", "pipeline": "final pipeline"}[key])
    axA.set_xticks([0, 1, 2, 3, 5])
    axA.set_xlabel("measured support points, $k$")
    axA.set_ylabel("macro MAE, one split seed")
    axA.legend(loc="upper right", fontsize=6.2)
    axA.set_title(f"{len(seeds)} split seeds, one line each", fontsize=6.7, color=_style.GREY)
    _style.panel(axA, "A", dx=-0.155)

    shape = pd.read_parquet(_paths.run("gen9_shape/shape/curve_shape.parquet"))
    shape = shape[shape.axis_label == "extractant"]
    seed_list = sorted(shape.split_seed.unique())
    for xi in range(len(seed_list)):
        axB.axvline(xi, color="#E8E8E8", linewidth=3.0, zorder=0)
    for j, (model, style_key) in enumerate(STAGES):
        block = shape[shape.model == model]
        med = block.groupby("split_seed")["slope_pred"].median()
        axB.plot(np.arange(len(med)) + (j - 1) * 0.16, med.to_numpy(), linestyle="none",
                 marker="o", markersize=3.4, color=_style.colour(style_key),
                 label=_style.label(style_key))
        for seed, value in med.items():
            record.append({"rule": model, "split_seed": int(seed),
                           "extractant_slope_median": float(value)})
    true_slope = float(shape.slope_true.median())
    axB.axhline(true_slope, color=_style.BLACK, linewidth=0.8, linestyle="--")
    axB.text(len(shape.split_seed.unique()) - 0.5, true_slope, f" measured {true_slope:.2f}",
             fontsize=6.0, va="bottom", ha="right")
    axB.set_xticks(np.arange(len(shape.split_seed.unique())))
    axB.set_xticklabels([str(s) for s in sorted(shape.split_seed.unique())], fontsize=5.6,
                        rotation=30)
    axB.set_xlabel("split seed")
    axB.set_ylabel("median predicted slope,\nextractant axis")
    axB.legend(loc="center left", fontsize=6.0)
    axB.set_ylim(0, 2.9)
    _style.panel(axB, "B", dx=-0.22)

    written = _style.save(fig, "FigS6_seed_stability", _paths.SUPP)
    (_paths.DERIVED / "figS6_values.json").write_text(json.dumps(record, indent=1))
    print("\n".join(str(p) for p in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
