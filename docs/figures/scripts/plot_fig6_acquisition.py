"""Figure 6 — which single measurement to make.

Run from the repository root:
    ``.venv/bin/python figures/scripts/plot_fig6_acquisition.py``

Source: ``runs/gen10_final/acquisition/realised_detail.parquet`` — every acquisition
policy scored on the *identical* candidate pool and the identical evaluation rows of the
same held-out extractant (143 extractants x 5 split seeds x 8 repeats), so the policies
differ only in which row they choose to measure.  The script first reproduces
``runs/gen10_final/acquisition/realised_summary.csv`` for every policy.

Panel A  macro MAE after one measurement, all 20 policies, coloured by family, with the
         paired chemotype-block interval against RANDOM.
Panel B  the fraction of extractants that one measurement makes *worse* than the
         zero-shot prediction.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _paths  # noqa: E402
import _stats  # noqa: E402
import _style  # noqa: E402

FEATURE_SET = "geometry"
FAMILY = {
    "ORACLE": "oracle", "SURROGATE_ORACLE": "oracle",
    "MEDOID": "centrality", "CENTRAL": "centrality", "MIN_GP_DESIGN_VAR": "centrality",
    "MEDIAN_PREDICTION": "centrality", "MAX_CONDITION_COVERAGE": "centrality",
    "MID_ACID": "centrality",
    "MAX_ENSEMBLE_SD": "uncertainty", "MIN_ENSEMBLE_SD": "uncertainty",
    "MAX_PREDICTION": "extreme", "MIN_PREDICTION": "extreme",
    "FARTHEST_FROM_EXISTING": "extreme",
    "RANDOM": "random",
}
FAMILY_COLOUR = {"oracle": _style.GREEN, "centrality": _style.VERMILLION,
                 "uncertainty": _style.SKY, "extreme": _style.PURPLE,
                 "learned": _style.BLUE, "random": _style.GREY}
FAMILY_LABEL = {"oracle": "oracle (not deployable)", "centrality": "geometric centrality",
                "uncertainty": "model uncertainty", "extreme": "extreme / spread",
                "learned": "learned ranker", "random": "random choice"}


def main() -> int:
    _style.apply()
    detail = pd.read_parquet(_paths.run("gen10_final/acquisition/realised_detail.parquet"))
    detail = detail[detail.feature_set == FEATURE_SET]

    per_ligand = detail.groupby(["policy", "extractant", "tanimoto_cluster"], observed=True).agg(
        mae=("mae", "mean"), harms=("harms", "mean"), zero_shot=("zero_shot", "mean"),
        oracle_mae=("oracle_mae", "mean"), gap_recovered=("gap_recovered", "mean")).reset_index()
    summary = per_ligand.groupby("policy").agg(
        mae=("mae", "mean"), frac_harmed=("harms", "mean"),
        n_ligands=("extractant", "nunique")).reset_index()

    published = pd.read_csv(_paths.run("gen10_final/acquisition/realised_summary.csv"))
    merged = summary.merge(published[["policy", "mae", "frac_harmed", "n_ligands"]],
                           on="policy", suffixes=("_mine", "_published"))
    merged["delta_mae"] = (merged.mae_mine - merged.mae_published).abs()
    merged["delta_harm"] = (merged.frac_harmed_mine - merged.frac_harmed_published).abs()
    ok = bool((merged.delta_mae < 1e-9).all() and (merged.delta_harm < 1e-9).all()
              and (merged.n_ligands_mine == merged.n_ligands_published).all())
    if not ok:
        print(merged.sort_values("delta_mae", ascending=False).head(10).to_string())
        raise SystemExit("acquisition summary did NOT reproduce — figure not drawn")

    long = per_ligand.rename(columns={"policy": "arm"})
    comparisons = {p: ("RANDOM", p) for p in summary.policy if p != "RANDOM"}
    paired = _stats.paired_chemotype_bootstrap(long, comparisons, arm_column="arm",
                                               seed_column="__none__")
    paired = paired.set_index("comparison")

    summary["family"] = summary.policy.map(FAMILY).fillna("learned")
    summary = summary.sort_values("mae")
    zero_shot = float(per_ligand[per_ligand.policy == "RANDOM"].zero_shot.mean())

    fig = plt.figure(figsize=(_style.W2, 3.35))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.42, 1.0], wspace=0.10,
                          left=0.235, right=0.985, top=0.90, bottom=0.135)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    y = np.arange(len(summary))[::-1]
    colours = [FAMILY_COLOUR[f] for f in summary.family]
    axA.barh(y, summary.mae, height=0.66, color=colours, alpha=0.9, linewidth=0)
    for yi, (_, row) in zip(y, summary.iterrows()):
        if row.policy == "RANDOM":
            continue
        r = paired.loc[row.policy]
        axA.plot([row.mae - r["bca_high"] + r["point"], row.mae - r["bca_low"] + r["point"]],
                 [yi, yi], color=_style.BLACK, linewidth=0.7)
    axA.axvline(zero_shot, color=_style.BLACK, linewidth=0.8, linestyle="--")
    axA.text(zero_shot, len(summary) - 0.4, "  zero-shot", fontsize=6.0, va="top")
    medoid = paired.loc["MEDOID"]
    axA.text(0.99, 0.02, f"MEDOID vs RANDOM  {medoid['point']:+.3f}\n"
                         f"[{medoid['bca_low']:+.3f}, {medoid['bca_high']:+.3f}], "
                         f"{int(medoid['units_improved'])}/{int(medoid['n_units'])}",
             transform=axA.transAxes, ha="right", va="bottom", fontsize=6.0)
    axA.set_yticks(y)
    axA.set_yticklabels(summary.policy, fontsize=6.0)
    axA.set_xlabel("macro MAE after one measurement")
    axA.set_xlim(0, 1.28)
    axA.set_title("143 held-out extractants, identical candidate pools",
                  fontsize=6.8, color=_style.GREY)
    handles = [plt.Line2D([], [], color=FAMILY_COLOUR[f], linewidth=4, solid_capstyle="butt")
               for f in FAMILY_LABEL]
    axA.legend(handles, list(FAMILY_LABEL.values()), loc="upper right",
               bbox_to_anchor=(1.005, 0.90), fontsize=6.0)
    _style.panel(axA, "A", dx=-0.42)

    axB.barh(y, summary.frac_harmed, height=0.66, color=colours, alpha=0.9, linewidth=0)
    axB.set_yticks(y)
    axB.set_yticklabels([])
    axB.set_xlabel("extractants made worse\nby the measurement")
    axB.set_xlim(0, 0.46)
    axB.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    axB.set_title("the deployment caveat", fontsize=6.8, color=_style.GREY)
    _style.panel(axB, "B", dx=-0.08)

    written = _style.save(fig, "Fig6_acquisition", _paths.MAIN)
    record = {"figure": "Fig6_acquisition",
              "source": "runs/gen10_final/acquisition/realised_detail.parquet",
              "feature_set": FEATURE_SET,
              "cohort": {"n_extractants": int(summary.n_ligands.max()),
                         "n_seeds": int(detail.split_seed.nunique()),
                         "n_repeats": int(detail.repeat.nunique()),
                         "zero_shot_macro_mae": zero_shot},
              "reproduces_realised_summary": ok,
              "policies": summary.round(6).to_dict("records"),
              "paired_vs_random": paired.reset_index().round(6).to_dict("records")}
    (_paths.DERIVED / "fig6_values.json").write_text(json.dumps(record, indent=1))
    print("\n".join(str(p) for p in written))
    print("reproduces realised_summary.csv for all", len(merged), "policies:", ok)
    print(summary.round(4).to_string())
    for name in ["MEDOID", "CENTRAL", "MAX_ENSEMBLE_SD", "FARTHEST_FROM_EXISTING", "ORACLE"]:
        r = paired.loc[name]
        print(f"{name:24s} vs RANDOM  {r['point']:+.4f} "
              f"[{r['bca_low']:+.4f},{r['bca_high']:+.4f}] "
              f"{int(r['units_improved'])}/{int(r['n_units'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
