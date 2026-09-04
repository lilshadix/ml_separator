#!/usr/bin/env python
"""Results figure R1 — what one measurement on an unseen extractant is worth.

One claim, one panel, two series. Everything else is in the caption.

Run:  .venv/bin/python figure_refinement/ml_separator/main/R1_value_of_one_measurement.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402
import pandas as pd                    # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common"))
import mlsep_data as D                 # noqa: E402
import pubstyle as PS                  # noqa: E402

KS = [0, 1, 2, 3, 5]
MODEL = "GEN9_SHAPE_RECOMPOSED"
#: The frozen deployment rule: zero-shot, slope repair at k = 1, series-local fit beyond.
PIPELINE = {0: ("ZERO_SHOT_REF", "NONE"), 1: ("SLOPE_L_s1_K1", "CENTRAL_THEN_SPREAD"),
            2: ("SERIES_ML", "CENTRAL_THEN_SPREAD"), 3: ("SERIES_ML", "CENTRAL_THEN_SPREAD"),
            5: ("SERIES_ML", "CENTRAL_THEN_SPREAD")}
#: The same measurements with no model at all — what the k points alone buy.
NO_MODEL = {k: ("NO_MODEL", "CENTRAL_THEN_SPREAD") for k in (1, 2, 3, 5)}


def curve(per_ligand: pd.DataFrame, rule: dict) -> pd.Series:
    out = {}
    for k, (adapter, policy) in rule.items():
        block = per_ligand[(per_ligand.global_model == MODEL)
                           & (per_ligand.adapter == adapter)
                           & (per_ligand.policy == policy) & (per_ligand.k == k)]
        out[k] = float(block["mae"].mean())
    return pd.Series(out).sort_index()


def main() -> int:
    PS.apply(base=9.0)
    per_ligand = D.kshot_per_ligand()
    model = curve(per_ligand, PIPELINE)
    alone = curve(per_ligand, NO_MODEL)

    fig, ax = plt.subplots(figsize=(PS.W15, 3.1))

    ax.plot(alone.index, alone.to_numpy(), color=PS.GREY, linestyle=":",
            linewidth=1.6, marker="o", markersize=4.0, zorder=2)
    ax.plot(model.index, model.to_numpy(), color=PS.VERMILLION, linewidth=2.2,
            marker="o", markersize=5.0, zorder=3)

    # Direct labels instead of a legend: two series, so a legend is not earning its space.
    ax.annotate("measurements alone", (5, alone[5]), textcoords="offset points",
                xytext=(8, 0), va="center", color=PS.GREY)
    ax.annotate("model + measurements", (5, model[5]), textcoords="offset points",
                xytext=(8, 0), va="center", color=PS.VERMILLION, fontweight="bold")

    # The three numbers that are the result, each placed in space no line occupies:
    # left of the k = 0 point, below k = 1, below k = 5.
    for k, dx, dy, ha, va in [(0, -7, 0, "right", "center"), (1, 0, -13, "center", "top"),
                              (5, 0, -13, "center", "top")]:
        ax.annotate(f"{model[k]:.2f}", (k, model[k]), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, va=va, color=PS.VERMILLION)

    # The claim, once, in the empty upper right. No arrow: the drop is already the
    # steepest thing on the panel.
    drop = (model[0] - model[1]) / model[0]
    ax.text(0.99, 0.99, f"one measurement removes {drop:.0%} of the error",
            transform=ax.transAxes, ha="right", va="top", fontweight="bold")

    ax.set_xticks(KS)
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.set_xlim(-0.5, 6.9)
    ax.set_ylim(0.30, 1.16)
    ax.set_xlabel("measurements on the unseen extractant")
    ax.set_ylabel("error (log$_{10}$ $D$)")

    report = PS.save(fig, HERE, "R1_value_of_one_measurement", strict=False, max_text=26)
    print(report)
    (HERE / "R1_values.json").write_text(json.dumps(
        {"model_plus_measurements": {int(k): round(v, 4) for k, v in model.items()},
         "measurements_alone": {int(k): round(v, 4) for k, v in alone.items()},
         "n_extractants": int(per_ligand.extractant.nunique()),
         "drop_from_first_measurement": round(float(model[0] - model[1]), 4),
         "drop_fraction": round(float((model[0] - model[1]) / model[0]), 4)}, indent=1))
    print("model:", {int(k): round(v, 4) for k, v in model.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
