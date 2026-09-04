"""Figure S10 — how to spend a fixed measurement budget.

Given a budget of measurements over a set of unseen extractants, is it better to measure a
few extractants deeply or many extractants once, and does covering new chemotypes first
help?  Each strategy is scored with the frozen per-extractant adaptation curves, so no new
model is fitted.

Source: ``runs/gen10_final/budget_simulation/{budget_strategies.csv,summary.json}``.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _paths, _style  # noqa: E402

#: Shown in panel A.  Every strategy the run produced is written to
#: ``figures/derived/figS10_values.json``; two (A_DEPTH_3, E_CENTRAL_SPREAD) are omitted
#: from the plot only because they lie on top of A_DEPTH_5 and
#: E_CENTRAL_SPREAD_NOVELTY respectively.
LABEL = {"A_DEPTH_5": "5 measurements each, fewest extractants",
         "B_BREADTH_1": "1 measurement each, as many as possible",
         "C_BREADTH_2": "2 measurements each",
         "D_NEW_CHEMOTYPE": "1 each, new chemotypes first",
         "E_CENTRAL_SPREAD_NOVELTY": "central-then-spread, new chemotypes first",
         "F_DIVERSITY": "1 each, max-min diverse order"}
COLOUR = {"A_DEPTH_5": _style.PURPLE, "B_BREADTH_1": _style.BLUE,
          "C_BREADTH_2": _style.ORANGE, "D_NEW_CHEMOTYPE": _style.SKY,
          "E_CENTRAL_SPREAD_NOVELTY": _style.VERMILLION, "F_DIVERSITY": _style.GREY}


def main() -> int:
    _style.apply()
    table = pd.read_csv(_paths.run("gen10_final/budget_simulation/budget_strategies.csv"))
    table = table[table.deployable]
    zero_shot = float(table.zero_shot_macro_mae.iloc[0])

    fig, axA = plt.subplots(1, 1, figsize=(_style.W15, 2.6))
    fig.subplots_adjust(left=0.115, right=0.985, top=0.94, bottom=0.19)

    for strategy, block in table.groupby("strategy"):
        if strategy not in LABEL:
            continue
        block = block.sort_values("budget")
        colour = COLOUR.get(strategy, "#BBBBBB")
        dashed = strategy in ("D_NEW_CHEMOTYPE", "F_DIVERSITY")
        axA.plot(block.budget, block.macro_mae, color=colour,
                 linewidth=1.9 if dashed else 1.2,
                 linestyle=(0, (3, 2.5)) if dashed else "-",
                 marker="o", markersize=2.6, alpha=0.95,
                 label=LABEL.get(strategy, strategy),
                 zorder=4 if dashed else 3)
    axA.axhline(zero_shot, color=_style.BLACK, linewidth=0.8, linestyle="--")
    axA.text(table.budget.min(), zero_shot + 0.006, "zero-shot", fontsize=6.0, va="bottom")
    axA.set_xlabel("measurement budget over the 99 held-out extractants")
    axA.set_ylabel("macro MAE (log$_{10}$ $D$ units)")
    axA.legend(loc="lower left", fontsize=6.0, bbox_to_anchor=(-0.005, -0.02))
    axA.set_xscale("log")
    axA.set_xticks([9, 24, 49, 99, 198, 297, 495])
    axA.set_xticklabels([9, 24, 49, 99, 198, 297, 495])
    axA.set_ylim(0.44, 1.07)
    axA.annotate("breadth saturates once every\nextractant has one measurement",
                 xy=(99, 0.6539), xytext=(115, 0.79), fontsize=5.9, color=_style.GREY,
                 arrowprops={"arrowstyle": "-", "linewidth": 0.6, "color": "#BBBBBB"})

    written = _style.save(fig, "FigS10_budget", _paths.SUPP)
    (_paths.DERIVED / "figS10_values.json").write_text(json.dumps(
        {"zero_shot": zero_shot, "plotted_strategies": list(LABEL),
         "note": "gain_per_measurement in the source table divides by measurements actually "
                 "spent, so a strategy that cannot spend the budget scores well on it while "
                 "achieving a worse macro MAE; it is therefore not plotted",
         "all_rows": table.round(6).to_dict("records")}, indent=1))
    print("\n".join(str(p) for p in written))
    print(table.pivot_table(index="strategy", columns="budget",
                            values="macro_mae").round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
