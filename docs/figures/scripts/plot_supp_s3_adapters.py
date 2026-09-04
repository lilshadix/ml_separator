"""Figure S3 — every calibration adapter at every k, on one global model.

Source: ``runs/gen10_final/adaptation/summary.csv`` (5 split seeds x 12 repeats, the
frozen global model, central-then-spread acquisition) and
``runs/gen10_final/adaptation/bootstrap_vs_offset_k3.csv`` for the paired intervals.
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
#: Panel A shows six adapters that bracket the range; panel B compares *every* adapter,
#: and the full table is written to figures/derived/figS3_values.json.
SHOW = ["NO_MODEL", "NEAREST_OBSERVED", "OFFSET_K1", "OFFSET_K3", "SLOPE_L_s1_K3",
        "SERIES_ML"]
COLOUR = {"NO_MODEL": _style.GREY, "NEAREST_OBSERVED": _style.PURPLE,
          "OFFSET_K1": _style.SKY, "OFFSET_K3": _style.ORANGE,
          "SLOPE_L_s1_K3": _style.BLUE, "SERIES_ML": _style.VERMILLION}


def main() -> int:
    _style.apply()
    detail = pd.read_parquet(_paths.run("gen10_final/adaptation/detail.parquet"),
                             columns=["global_model", "adapter", "policy", "k", "extractant",
                                      "tanimoto_cluster", "n_pool", "n_eval", "mae"])
    detail = detail[(detail.global_model == MODEL)
                    & (detail.policy.isin(["CENTRAL_THEN_SPREAD", "NONE"]))]
    eligible = detail.groupby("extractant").apply(
        lambda b: bool((b.n_pool >= 5).all() and (b.n_eval >= 2).all()), include_groups=False)
    cohort = sorted(eligible.index[eligible])
    detail = detail[detail.extractant.isin(cohort)]
    per_ligand = detail.groupby(["adapter", "k", "extractant"], observed=True)["mae"].mean()
    table = per_ligand.groupby(["adapter", "k"]).mean().reset_index()

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(_style.W2, 2.6),
                                   gridspec_kw={"width_ratios": [1.05, 1.0], "wspace": 0.06})
    fig.subplots_adjust(left=0.085, right=0.845, top=0.90, bottom=0.20)

    full = table.pivot_table(index="adapter", columns="k", values="mae")
    pivot = full.reindex([a for a in SHOW if a in full.index])
    for adapter, row in pivot.iterrows():
        values = row.dropna()
        axA.plot(values.index.astype(int), values.to_numpy(), color=COLOUR[adapter],
                 linewidth=1.3, marker="o", markersize=2.8, label=adapter)
    axA.set_xticks([1, 2, 3, 5])
    axA.set_xlim(0.85, 5.15)
    axA.set_xlabel("measured support points, $k$")
    axA.set_ylabel("macro MAE (log$_{10}$ $D$ units)")
    axA.legend(loc="upper right", fontsize=6.0)
    axA.set_title(f"{len(cohort)} extractants, central-then-spread",
                  fontsize=6.6, color=_style.GREY)
    _style.panel(axA, "A", dx=-0.145)

    boot = pd.read_csv(_paths.run("gen10_final/adaptation/bootstrap_vs_offset_k3.csv"))
    keep = boot[boot.comparison.str.contains("OFFSET_K3", na=False)]
    keep = keep[keep.comparison.str.contains(MODEL, na=False)] if \
        keep.comparison.str.contains(MODEL, na=False).any() else keep
    keep = keep.copy()
    keep["adapter"] = keep.comparison.str.split("@").str[0].str.split("|").str[-1].str.strip()
    keep = keep.drop_duplicates(["comparison"]).sort_values("point")
    y = np.arange(len(keep))
    axB.errorbar(keep.point, y, xerr=[keep.point - keep.bca_low, keep.bca_high - keep.point],
                 fmt="o", markersize=3.2, elinewidth=0.8, capsize=1.6,
                 color=_style.VERMILLION, ecolor=_style.BLACK)
    axB.axvline(0, color=_style.BLACK, linewidth=0.7)
    axB.set_yticks(y)
    axB.yaxis.tick_right()
    axB.set_yticklabels(keep.comparison.str.replace("_vs_OFFSET_K3", "", regex=False),
                        fontsize=6.0)
    axB.spines["right"].set_visible(True)
    axB.spines["left"].set_visible(False)
    axB.set_xlabel("macro MAE removed vs OFFSET_K3\n(paired, BCa 95 % CI)")
    _style.panel(axB, "B", dx=-0.06)

    written = _style.save(fig, "FigS3_adapters", _paths.SUPP)
    (_paths.DERIVED / "figS3_values.json").write_text(json.dumps(
        {"cohort_extractants": len(cohort),
         "all_adapters": full.round(6).reset_index().to_dict("records"),
         "panelA": pivot.round(6).reset_index().to_dict("records"),
         "panelB": keep.round(6).to_dict("records")}, indent=1))
    print("\n".join(str(p) for p in written))
    print(full.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
