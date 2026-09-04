"""Figure S7 — the same held-out titrations as Figure 3A-C, on the raw log D axis.

Figure 3 plots the curve-centred response, because that is the quantity the shape metrics
and the mean-preserving recomposition act on.  This panel shows the same six curves
without centring, so the reader can see the per-curve level error that the recomposition
deliberately does not touch.  Six curves: the three selected by the Figure 3 rule plus the
next-ranked alternative in each role, so the choice can be inspected.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _paths, _style  # noqa: E402
from plot_fig3_shape_recovery import (EXAMPLE_SEED, STAGES, STYLE_KEY,  # noqa: E402
                                      example_rows, load_curve_shape, select_examples)


def main() -> int:
    _style.apply()
    cs = load_curve_shape()
    chosen, ranked = select_examples(cs)
    runners = {
        "median": ranked.sort_values(["dist_to_joint_median", "curve_id"]).curve_id.iloc[1],
        "improved": ranked.sort_values(["dist_to_top_decile_median",
                                        "curve_id"]).curve_id.iloc[1],
        "residual": ranked.sort_values(["dist_to_bottom_decile_median",
                                        "curve_id"]).curve_id.iloc[1]}
    order = [(role, chosen[role], "selected") for role in ("median", "improved", "residual")]
    order += [(role, runners[role], "runner-up") for role in ("median", "improved", "residual")]
    rows = example_rows([cid for _, cid, _ in order])

    fig, axes = plt.subplots(2, 3, figsize=(_style.W2, 3.9))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.92, bottom=0.11, hspace=0.55,
                        wspace=0.30)
    titles = {"median": "median curve", "improved": "large repair", "residual": "repair fails"}
    record = []
    for ax, (role, curve_id, kind), letter in zip(axes.ravel(), order, "ABCDEF"):
        block = rows[rows.curve_id == curve_id].sort_values("axis_value")
        ax.plot(block.axis_value, block.log_D, color=_style.BLACK, marker="o",
                markersize=3.4, linewidth=1.0, label="Measured", zorder=5)
        for key, _model, nice in STAGES:
            kw = _style.m(STYLE_KEY[key])
            kw["label"] = nice
            kw["markersize"] = 2.6
            ax.plot(block.axis_value, block[f"pred_{key}"], **kw)
        ax.set_title(f"{titles[role]} — {kind}", fontsize=6.6, color=_style.GREY)
        ax.set_xlabel("log$_{10}$ [extractant] (M)")
        if letter in "AD":
            ax.set_ylabel("log$_{10}$ $D$")
        if letter == "A":
            ax.legend(loc="best", fontsize=5.8)
        _style.panel(ax, letter, dx=-0.28)
        record.append({"role": role, "kind": kind, "curve_id": curve_id,
                       "extractant": ranked.set_index("curve_id").loc[curve_id, "extractant"],
                       "n_points": int(len(block)),
                       "shape_mae_frozen": float(
                           ranked.set_index("curve_id").loc[curve_id, "shape_frozen"]),
                       "shape_mae_recomposed": float(
                           ranked.set_index("curve_id").loc[curve_id, "shape_recomposed"]),
                       "measured_span": float(block.log_D.max() - block.log_D.min()),
                       "level_error_recomposed": float(
                           block.pred_recomposed.mean() - block.log_D.mean())})

    written = _style.save(fig, "FigS7_raw_example_curves", _paths.SUPP)
    (_paths.DERIVED / "figS7_values.json").write_text(
        json.dumps({"split_seed": EXAMPLE_SEED, "curves": record}, indent=1))
    print("\n".join(str(p) for p in written))
    print(pd.DataFrame(record).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
