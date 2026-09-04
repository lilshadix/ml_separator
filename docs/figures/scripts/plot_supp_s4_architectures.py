"""Figure S4 — zero-shot architecture and representation ablation.

Every arm is fitted on the same folds with the same learner family and scored on the
same held-out rows; 5 split seeds.  Source:
``runs/gen10_final/headline_tables/t2_leaderboards.csv`` (itself regenerated from the
per-stage leaderboards in ``runs/gen10_final/{feature_access,level_shape,
axis_representation,set_context}/leaderboard.csv``).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _paths, _style  # noqa: E402

STAGE_COLOUR = {"feature_access": _style.SKY, "level_shape": _style.PURPLE,
                "axis_representation": _style.BLUE, "set_context": _style.ORANGE}
STAGE_LABEL = {"feature_access": "feature access / capacity",
               "level_shape": "explicit level + shape heads",
               "axis_representation": "axis representation",
               "set_context": "learned set encoder"}
REFERENCE = "GEN9_SHAPE_RECOMPOSED"
BASELINE = "REC_ecfp_plus_recovered"


def main() -> int:
    _style.apply()
    lb = pd.read_csv(_paths.run("gen10_final/headline_tables/t2_leaderboards.csv"))
    lb = lb.drop_duplicates("model").sort_values("macro_mae")
    ref = float(lb[lb.model == REFERENCE].macro_mae.iloc[0])
    base = float(lb[lb.model == BASELINE].macro_mae.iloc[0])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(_style.W2, 3.5),
                                   gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.12})
    fig.subplots_adjust(left=0.30, right=0.985, top=0.92, bottom=0.13)

    y = np.arange(len(lb))[::-1]
    colours = [STAGE_COLOUR[s] for s in lb.stage]
    # points, not bars: the axis is truncated, and a truncated bar exaggerates differences
    axA.errorbar(lb.macro_mae, y, xerr=lb.macro_sd, fmt="none", ecolor="#9A9A9A",
                 elinewidth=0.6, capsize=1.3)
    axA.scatter(lb.macro_mae, y, s=15, color=colours, zorder=4)
    axA.axvline(ref, color=_style.BLUE, linewidth=0.9, linestyle="--")
    axA.axvline(base, color=_style.ORANGE, linewidth=0.9, linestyle=":")
    axA.set_yticks(y)
    labels = [m + ("  (frozen pipeline)" if m == REFERENCE else
                   ("  (baseline)" if m == BASELINE else "")) for m in lb.model]
    axA.set_yticklabels(labels, fontsize=5.6)
    axA.set_xlim(0.945, 1.105)
    axA.set_xlabel("macro MAE, zero-shot (mean ± sd over 5 split seeds)")
    handles = [plt.Line2D([], [], color=c, linewidth=4, solid_capstyle="butt",
                          label=STAGE_LABEL[s]) for s, c in STAGE_COLOUR.items()]
    axA.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.005, -0.02),
               fontsize=5.9)
    _style.panel(axA, "A", dx=-0.62)

    axB.scatter(lb.offset_mae, lb.shape_mae, s=14, facecolor="none",
                edgecolor=[STAGE_COLOUR[s] for s in lb.stage], linewidth=0.8)
    for _, row in lb.iterrows():
        if row.model in (REFERENCE, BASELINE) or row.macro_mae > 1.05:
            axB.annotate(row.model, (row.offset_mae, row.shape_mae),
                         textcoords="offset points", xytext=(3, 3), fontsize=5.2,
                         color=STAGE_COLOUR[row.stage])
    axB.set_xlabel("level error (offset MAE)")
    axB.set_ylabel("shape error (within-extractant MAE)")
    axB.text(0.98, 0.02, "every recomposed arm shares the baseline's\nlevel to 12 decimals "
                         "(mean-preserving)",
             transform=axB.transAxes, va="bottom", ha="right", fontsize=5.9,
             color=_style.GREY)
    _style.panel(axB, "B", dx=-0.24)

    written = _style.save(fig, "FigS4_architectures", _paths.SUPP)
    (_paths.DERIVED / "figS4_values.json").write_text(
        json.dumps({"reference": ref, "baseline": base,
                    "arms": lb.round(6).to_dict("records")}, indent=1))
    print("\n".join(str(p) for p in written))
    print(lb[["stage", "model", "macro_mae", "offset_mae", "shape_mae"]].round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
