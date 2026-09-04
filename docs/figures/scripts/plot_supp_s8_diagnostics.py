"""Figure S8 — prediction diagnostics and the data-quality strata.

Sources: ``runs/gen9_shape/recomposed/oof_GEN9_SHAPE_RECOMPOSED.parquet`` (all five split
seeds) and ``runs/gen10_final/data_ceiling/{stratified_metrics.csv,row_cohorts.csv}``.
The strata are the ones Phase 7 defined from the source audit: consistent rows, rows whose
recorded name and structure disagree, decade-shifted duplicate cells, the single suspect
extractant TWE-24, and rows flagged as uncertain.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _paths, _style  # noqa: E402

MODEL = "GEN9_SHAPE_RECOMPOSED"
COHORT_LABEL = {"A_CONSISTENT": "consistent", "B_MISMATCH": "name/structure mismatch",
                "C_DUPLICATE": "decade-shifted duplicates", "D_TWE24": "TWE-24",
                "E_UNCERTAIN": "flagged uncertain"}


def main() -> int:
    _style.apply()
    oof = pd.read_parquet(_paths.run("gen9_shape/recomposed/oof_GEN9_SHAPE_RECOMPOSED.parquet"))
    oof = oof[oof.model == MODEL]
    residual = oof.prediction - oof.log_D

    fig, axes = plt.subplots(1, 3, figsize=(_style.W2, 2.45),
                             gridspec_kw={"width_ratios": [1.0, 1.0, 1.32], "wspace": 0.62})
    axA, axB, axC = axes
    fig.subplots_adjust(left=0.070, right=0.985, top=0.90, bottom=0.24)

    axA.scatter(oof.log_D, oof.prediction, s=1.6, color=_style.BLUE, alpha=0.10,
                linewidth=0)
    lim = (-5.2, 4.4)
    axA.plot(lim, lim, color=_style.BLACK, linewidth=0.7, linestyle="--")
    axA.set_xlim(*lim)
    axA.set_ylim(*lim)
    axA.set_aspect("equal")
    axA.set_xlabel("measured log$_{10}$ $D$")
    axA.set_ylabel("predicted log$_{10}$ $D$")
    axA.text(0.03, 0.97, f"{len(oof):,} held-out rows, 5 split seeds\n"
                         f"sd(measured) {oof.log_D.std():.2f}\n"
                         f"sd(predicted) {oof.prediction.std():.2f}\n"
                         "the model compresses the whole range,\nnot only the within-curve range",
             transform=axA.transAxes, va="top", fontsize=5.9, color=_style.GREY)
    _style.panel(axA, "A", dx=-0.22)

    axB.scatter(oof.prediction, residual, s=1.6, color=_style.BLUE, alpha=0.10, linewidth=0)
    axB.axhline(0, color=_style.BLACK, linewidth=0.7, linestyle="--")
    bins = np.quantile(oof.prediction, np.linspace(0, 1, 13))
    centres, means = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (oof.prediction >= lo) & (oof.prediction < hi)
        if mask.sum():
            centres.append(float(oof.prediction[mask].mean()))
            means.append(float(residual[mask].mean()))
    axB.plot(centres, means, color=_style.VERMILLION, linewidth=1.3, marker="o",
             markersize=2.6, label="binned mean")
    axB.set_xlabel("predicted log$_{10}$ $D$")
    axB.set_ylabel("prediction $-$ measurement")
    axB.set_ylim(-4.2, 4.2)
    axB.legend(loc="upper right", fontsize=6.0)
    # Anchored to the panel's own left edge and wrapped to four short lines so the note
    # cannot run past the axes into panel C's tick labels.
    axB.text(0.02, 0.02, f"mean residual {residual.mean():+.2f}:\nthe model under-predicts on\n"
                         "held-out chemistry, and\nincreasingly so at high $D$",
             transform=axB.transAxes, fontsize=5.8, color=_style.GREY,
             ha="left", va="bottom", linespacing=1.4)
    _style.panel(axB, "B", dx=-0.24)

    strat = pd.read_csv(_paths.run("gen10_final/data_ceiling/stratified_metrics.csv"))
    strat = strat[strat.model == MODEL].set_index("cohort")
    order = list(COHORT_LABEL)
    y = np.arange(len(order))[::-1]
    values = [float(strat.loc[c, "ligand_macro_mae"]) for c in order]
    axC.barh(y, values, height=0.6, color=[_style.BLUE if c == "A_CONSISTENT"
                                           else _style.PURPLE for c in order],
             alpha=0.9, linewidth=0)
    for yi, c, v in zip(y, order, values):
        axC.text(v + 0.06, yi, f"{v:.2f}   {int(strat.loc[c, 'n_ligands'])} extractants, "
                               f"{int(strat.loc[c, 'n_rows']):,} rows",
                 va="center", fontsize=5.6, color=_style.GREY)
    axC.set_yticks(y)
    axC.set_yticklabels([COHORT_LABEL[c] for c in order], fontsize=6.0)
    axC.set_xlabel("macro MAE, one vote per extractant")
    axC.set_xlim(0, 8.6)
    _style.panel(axC, "C", dx=-0.42)

    written = _style.save(fig, "FigS8_diagnostics", _paths.SUPP)
    record = {"n_rows": int(len(oof)), "sd_measured": float(oof.log_D.std()),
              "sd_predicted": float(oof.prediction.std()),
              "residual_mean": float(residual.mean()), "residual_sd": float(residual.sd)
              if hasattr(residual, "sd") else float(residual.std()),
              "binned_residual_mean": [{"prediction": c, "mean_residual": m}
                                       for c, m in zip(centres, means)],
              "strata": strat.reset_index().round(6).to_dict("records")}
    (_paths.DERIVED / "figS8_values.json").write_text(json.dumps(record, indent=1))
    print("\n".join(str(p) for p in written))
    print(strat[["n_rows", "n_ligands", "ligand_macro_mae", "offset_mae",
                 "curve_shape_mae"]].round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
