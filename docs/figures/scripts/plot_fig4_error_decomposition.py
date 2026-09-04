"""Figure 4 — where the remaining error is, and how much of it is reachable.

Run from the repository root:
    ``.venv/bin/python figures/scripts/plot_fig4_error_decomposition.py``

Panel A  oracle cascade and what k measurements actually achieve, both on the
         99-extractant common cohort with one vote per extractant.
         Cascade rebuilt from ``runs/gen9_shape/recomposed/oof_all.parquet`` by
         ``prepare_error_budget.py``, which first reproduces
         ``runs/gen10_final/error_decomposition/summary.json`` exactly.
Panel B  the seven-component budget of the 0.970 macro MAE
         (``runs/gen10_final/error_decomposition/decomposition.csv``).
Panel C  trajectory in (level error, shape error) space: what a better model moves
         versus what a measurement moves.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import _paths  # noqa: E402
import _style  # noqa: E402
from lanthanide_separation.gen6.metrics import decompose_level_shape  # noqa: E402

MODEL_STAGES = [
    ("no_ligand", "NULL_metal_cond", "gen7_architecture/finalists/oof_predictions.parquet",
     "no ligand information"),
    ("frozen", "REC_ecfp_plus_recovered", "gen7_architecture/finalists/oof_predictions.parquet",
     "baseline model"),
    ("relpos", "GEN9_REL_MONOLITH", "gen9_shape/relmono/oof_GEN9_REL_MONOLITH.parquet",
     "+ relative position"),
    ("recomposed", "GEN9_SHAPE_RECOMPOSED",
     "gen9_shape/recomposed/oof_GEN9_SHAPE_RECOMPOSED.parquet", "+ recomposition"),
]
KSHOT_STAGES = [(0, "ZERO_SHOT_REF"), (1, "SLOPE_L_s1_K1"), (2, "SERIES_ML"),
                (3, "SERIES_ML"), (5, "SERIES_ML")]


def ensure(name: str) -> Path:
    path = _paths.DERIVED / name
    if not path.exists():
        script = {"kshot_per_ligand.csv": "prepare_kshot_tables.py",
                  "error_budget.json": "prepare_error_budget.py"}[name]
        subprocess.run([sys.executable, str(Path(__file__).with_name(script))], check=True)
    return path


def model_stage_decomposition(ligands: set[str]) -> pd.DataFrame:
    rows = []
    for key, model, path, nice in MODEL_STAGES:
        oof = pd.read_parquet(_paths.run(path),
                              columns=["row_id", "extractant", "log_D", "prediction",
                                       "split_seed", "model"])
        oof = oof[(oof.model == model) & (oof.extractant.isin(ligands))]
        per_seed = []
        for seed, block in oof.groupby("split_seed"):
            d = decompose_level_shape(block.log_D, block.prediction, block.extractant)
            per_seed.append({"split_seed": int(seed), "mae": d.summary["macro_mae_ligand"],
                             "offset_mae": d.summary["offset_mae"],
                             "shape_mae": d.summary["shape_mae"]})
        frame = pd.DataFrame(per_seed)
        rows.append({"key": key, "stage": nice, "model": model,
                     "n_seeds": len(frame), "n_ligands": int(oof.extractant.nunique()),
                     **frame.drop(columns="split_seed").mean().to_dict()})
    return pd.DataFrame(rows)


def main() -> int:
    _style.apply()
    per_ligand = pd.read_csv(ensure("kshot_per_ligand.csv"))
    budget = json.loads(ensure("error_budget.json").read_text())
    ligands = set(per_ligand.extractant.unique())

    fig = plt.figure(figsize=(_style.W2, 4.25))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.92, 1.05], width_ratios=[1.05, 1.0],
                          hspace=0.40, wspace=0.42,
                          left=0.135, right=0.985, top=0.93, bottom=0.105)
    axA = fig.add_subplot(gs[0, :])
    axB = fig.add_subplot(gs[1, 0])
    axC = fig.add_subplot(gs[1, 1])

    # ------------------------------------------------------------------ A ------
    stages = budget["common_cohort"]["stages"]
    achieved = budget["common_cohort"]["achieved"]
    ladder = [
        ("zero-shot", achieved["zero_shot"], "pipeline", False),
        ("$k$=1\ncentral", achieved["k1_slope_repair_central"], "pipeline", False),
        ("$k$=3", achieved["k3_series_ml_central"], "pipeline", False),
        ("$k$=5", achieved["k5_series_ml_central"], "pipeline", False),
        ("$k$=1\noracle point", achieved["k1_oracle_point"], "oracle", True),
        ("per-ligand\nlevel", stages["after_ligand_level_oracle"]["macro_mae"], "oracle", True),
        ("per-series\nlevel", stages["after_series_level_oracle"]["macro_mae"], "oracle", True),
        ("per-curve\nlevel", stages["after_curve_level_oracle"]["macro_mae"], "oracle", True),
        ("per-curve\nlevel + slope", stages["after_curve_level_and_slope_oracle"]["macro_mae"],
         "oracle", True),
    ]
    x = np.arange(len(ladder))
    values = [v for _, v, _, _ in ladder]
    colours = [_style.colour(c) for _, _, c, _ in ladder]
    hatches = ["//" if o else "" for _, _, _, o in ladder]
    bars = axA.bar(x, values, width=0.62, color=colours, alpha=0.88, linewidth=0)
    for bar, hatch in zip(bars, hatches):
        if hatch:
            bar.set_hatch(hatch)
            bar.set_edgecolor("white")
            bar.set_linewidth(0.0)
    for xi, v in zip(x, values):
        axA.text(xi, v + 0.018, f"{v:.3f}", ha="center", fontsize=6.2)
    axA.axvline(3.5, color=_style.GREY, linewidth=0.6, linestyle=":")
    axA.text(1.5, 1.14, "deployable: measure $k$ points", ha="center", fontsize=6.5,
             color=_style.colour("pipeline"))
    axA.text(6.5, 1.14, "oracle bounds — not deployable", ha="center", fontsize=6.5,
             color=_style.colour("oracle"))
    axA.set_xticks(x)
    axA.set_xticklabels([n for n, _, _, _ in ladder], fontsize=6.3)
    axA.set_ylabel("macro MAE  (log$_{10}$ $D$ units)")
    axA.set_ylim(0, 1.22)
    _style.panel(axA, "A", dx=-0.062)

    # ------------------------------------------------------------------ B ------
    comp = pd.DataFrame(budget["components"])
    short_names = {
        "1 predictable level (per-ligand constant)": "per-extractant level",
        "2 curve shape (within-curve, after the curve's level)": "within-curve shape",
        "3 series-local level (beyond the ligand constant)": "series-local level",
        "4 system identity (name/structure mismatch rows)": "system identity",
        "5 sparse support / distant chemotype": "distant chemistry",
        "6 suspected data quality (duplicates, TWE-24, flagged)": "data quality",
        "7 residual unexplained": "unexplained",
    }
    comp["short"] = comp["component"].map(short_names).fillna(comp["component"])
    comp = comp.sort_values("current_contribution", ascending=True)
    y = np.arange(len(comp))
    axB.barh(y, comp.current_contribution, height=0.62, color=_style.GREY, alpha=0.55,
             linewidth=0, label="current contribution")
    axB.barh(y, comp.realistically_removable, height=0.34,
             color=_style.colour("pipeline"), linewidth=0,
             label="removed by a deployable action")
    axB.set_yticks(y)
    axB.set_yticklabels(list(comp.short), fontsize=6.5)
    axB.set_xlabel("share of the 0.970 macro MAE (log$_{10}$ $D$ units)")
    axB.axvline(0, color=_style.BLACK, linewidth=0.6)
    axB.legend(loc="lower right", fontsize=6.0)
    _style.panel(axB, "B", dx=-0.44)

    # ------------------------------------------------------------------ C ------
    models = model_stage_decomposition(ligands)
    ks = []
    for k, adapter in KSHOT_STAGES:
        block = per_ligand[(per_ligand.global_model == "GEN9_SHAPE_RECOMPOSED")
                           & (per_ligand.adapter == adapter) & (per_ligand.k == k)
                           & (per_ligand.policy.isin(["CENTRAL_THEN_SPREAD", "NONE"]))]
        ks.append({"k": k, "offset_mae": block.offset.mean(), "shape_mae": block.shape_mae.mean(),
                   "mae": block.mae.mean()})
    ks = pd.DataFrame(ks)

    axC.plot(models.offset_mae, models.shape_mae, color=_style.BLUE, linewidth=1.0,
             marker="D", markersize=3.6, label="better model (zero-shot)")
    for _, r in models.iterrows():
        axC.annotate(r.stage, (r.offset_mae, r.shape_mae), textcoords="offset points",
                     xytext=(-4, 4), fontsize=5.8, color=_style.BLUE, ha="right")
    axC.plot(ks.offset_mae, ks.shape_mae, color=_style.colour("pipeline"), linewidth=1.0,
             marker="o", markersize=3.6, label="more measurements")
    for _, r in ks.iterrows():
        axC.annotate(f"$k$={int(r.k)}", (r.offset_mae, r.shape_mae), textcoords="offset points",
                     xytext=(4, -8), fontsize=5.8, color=_style.colour("pipeline"))
    axC.set_xlabel("level error, mean $|$per-extractant offset$|$")
    axC.set_ylabel("shape error, mean within-extractant MAE")
    axC.set_xlim(0.15, 1.05)
    axC.set_ylim(0.355, 0.635)
    axC.legend(loc="upper left", fontsize=6.0, bbox_to_anchor=(-0.02, 1.03))
    _style.panel(axC, "C", dx=-0.22)

    written = _style.save(fig, "Fig4_error_decomposition", _paths.MAIN)

    record = {"figure": "Fig4_error_decomposition",
              "panelA": [{"stage": n, "macro_mae": v, "oracle": o} for n, v, _, o in ladder],
              "panelA_cohort": budget["common_cohort"]["description"],
              "panelB": comp.round(6).to_dict("records"),
              "panelB_cohort": "152 extractants, one vote per ECFP cluster (gen10 Phase 10)",
              "panelC_model_stages": models.round(6).to_dict("records"),
              "panelC_kshot_stages": ks.round(6).to_dict("records"),
              "verification": budget["verification"]["all_pass"]}
    (_paths.DERIVED / "fig4_values.json").write_text(json.dumps(record, indent=1))
    print("\n".join(str(p) for p in written))
    print(models.round(4).to_string())
    print(ks.round(4).to_string())
    print(comp[["short", "current_contribution", "oracle_removable",
                "realistically_removable"]].round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
