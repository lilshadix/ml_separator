#!/usr/bin/env python
"""Figures, with the JSON of every plotted value beside them.

A figure whose numbers cannot be read back is decoration.  Each panel writes its data to
``figures/values.json`` so a reader can check the picture against the table.

Usage::

    PYTHONPATH=gen12_2_eu_pred .venv/bin/python gen12_2_eu_pred/scripts/g122_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gen122 import paths  # noqa: E402

INK, GRID = "#1b1b1b", "#d8d8d8"
COORD, GENERIC, NEUTRAL = "#0b6e4f", "#8c510a", "#7a7a7a"


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def main() -> int:
    values: dict = {}
    leaderboard = pd.read_csv(paths.METRIC_DIR / "level" / "LVL_MEAN" / "leaderboard.csv")
    full_cohort = leaderboard[leaderboard["cohort"] == "full"].sort_values("level_mae")

    # --- fig 1: the level ladder ------------------------------------------- #
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
    colours = [COORD if "COORD" in a or a == "L7_ALL" else
               (NEUTRAL if a.startswith("L0") else GENERIC) for a in full_cohort["arm"]]
    ax.barh(full_cohort["arm"], full_cohort["level_mae"], color=colours, height=0.65)
    for y, (arm, value) in enumerate(zip(full_cohort["arm"], full_cohort["level_mae"])):
        ax.text(value + 0.012, y, f"{value:.3f}", va="center", fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("extractant-level MAE, zero-shot, design B, full cohort (decades)", fontsize=9)
    ax.set_title("Level prediction on chemically unseen extractants\n"
                 "green = contains the coordination block", fontsize=10, color=INK, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(paths.FIGURE_DIR / "fig1_level_ladder.png")
    plt.close(fig)
    values["fig1_level_ladder"] = full_cohort[["arm", "ablation_letter", "level_mae",
                                               "inner_validation"]].to_dict(orient="records")

    # --- fig 2: the primary contrast, overall and by band ------------------- #
    pairwise = pd.read_csv(paths.BOOTSTRAP_DIR / "level_LVL_MEAN_pairwise.csv")
    bands = pd.read_csv(paths.BOOTSTRAP_DIR / "level_LVL_MEAN_by_band.csv")
    rows = []
    for cohort in ("full", "level_reliable"):
        block = pairwise[(pairwise["cohort"] == cohort) & (pairwise["statistic"] == "mae")
                         & (pairwise["contrast"] == "PRIMARY_G_vs_D")]
        if len(block):
            rows.append({"label": f"overall\n({cohort})", **block.iloc[0][
                ["point_delta", "bca_low", "bca_high"]].to_dict()})
    for band in ("far", "mid", "near"):
        block = bands[(bands["band"] == band) & (bands["comparison"] == "PRIMARY_G_vs_D")]
        if len(block):
            rows.append({"label": f"{band}\n(full)", **block.iloc[0][
                ["point_delta", "bca_low", "bca_high"]].to_dict()})
    table = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.0, 3.6), dpi=200)
    y = np.arange(len(table))
    ax.errorbar(table["point_delta"], y,
                xerr=[table["point_delta"] - table["bca_low"],
                      table["bca_high"] - table["point_delta"]],
                fmt="o", color=COORD, ecolor=COORD, capsize=3, markersize=5, linewidth=1.2)
    ax.axvline(0.0, color=INK, linewidth=0.9)
    ax.axvline(0.01, color=GRID, linewidth=0.9, linestyle="--")
    ax.text(0.011, len(table) - 0.4, "reproducibility floor", fontsize=7, color=NEUTRAL)
    ax.set_yticks(y, table["label"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("level MAE improvement from the coordination block (decades)\n"
                  "ECFP + generic + coordination  minus  ECFP + generic", fontsize=9)
    ax.set_title("The primary endpoint, with 95 % BCa intervals", fontsize=10, color=INK,
                 loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(paths.FIGURE_DIR / "fig2_primary_contrast.png")
    plt.close(fig)
    values["fig2_primary_contrast"] = table.to_dict(orient="records")

    # --- fig 3: the multi-arm subgroup -------------------------------------- #
    subgroup = pd.read_csv(paths.BOOTSTRAP_DIR / "level_LVL_MEAN_subgroup.csv")
    signed = subgroup.drop_duplicates("subgroup").set_index("subgroup")
    arms = ["L1_ECFP", "L3_ECFP_GENERIC", "L7_ALL", "L4_COORD", "L0_NN_TANIMOTO"]
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=200)
    width = 0.38
    x = np.arange(len(arms))
    for offset, group, colour in ((-width / 2, "single_arm", NEUTRAL),
                                  (width / 2, "MULTI_ARM", COORD)):
        heights = [signed.loc[group, f"signed_{a}"] for a in arms]
        ax.bar(x + offset, heights, width, label=group.replace("_", " "), color=colour)
    ax.axhline(0.0, color=INK, linewidth=0.9)
    ax.set_xticks(x, arms, fontsize=8, rotation=12)
    ax.set_ylabel("mean signed level error (decades)", fontsize=9)
    ax.set_title("Multi-armed ligands are under-predicted by every representation\n"
                 "negative is under-prediction; 43 multi-arm extractants in 21 chemotypes, "
                 "effective n 4.0", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(paths.FIGURE_DIR / "fig3_multi_arm.png")
    plt.close(fig)
    values["fig3_multi_arm"] = {g: {a: float(signed.loc[g, f"signed_{a}"]) for a in arms}
                                for g in ("single_arm", "MULTI_ARM")}

    # --- fig 4: level versus shape, the bottleneck -------------------------- #
    bottleneck = pd.read_csv(paths.ANALYSIS_DIR / "level_bottleneck.csv")
    block = bottleneck[bottleneck["arm"].str.startswith("GEN12_ABL")]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    x = np.arange(len(block))
    ax.bar(x - 0.19, block["macro_mae"], 0.38, label="as measured", color=NEUTRAL)
    ax.bar(x + 0.19, block["with_true_level_own_shape"], 0.38,
           label="with the true level, own shape", color=COORD)
    ax.plot(x + 0.19, block["with_true_shape_own_level"], "o", color=GENERIC,
            markersize=6, label="with the true shape, own level")
    ax.set_xticks(x, [a.split(" ")[0].replace("GEN12_", "") for a in block["arm"]], fontsize=8)
    ax.set_ylabel("extractant-macro MAE (decades)", fontsize=9)
    ax.set_title("Where Gen12's zero-shot error lives\n"
                 "fixing the level recovers about five times what fixing the shape does",
                 fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(paths.FIGURE_DIR / "fig4_level_bottleneck.png")
    plt.close(fig)
    values["fig4_level_bottleneck"] = block.to_dict(orient="records")

    # --- fig 5: one-shot, if it has run ------------------------------------- #
    gain_path = paths.METRIC_DIR / "one_shot" / "one_shot_gain.csv"
    if gain_path.exists():
        curve = pd.read_csv(paths.METRIC_DIR / "one_shot" / "fewshot_curve.csv")
        common = curve[(curve["cohort"] == "common") & (curve["adapter"] == "A_OFFSET_SHRUNK")]
        fig, ax = plt.subplots(figsize=(7.0, 3.8), dpi=200)
        for arm, block in common.groupby("arm"):
            block = block.sort_values("k")
            colour = COORD if "L7" in arm else (GENERIC if "L3" in arm else NEUTRAL)
            ax.plot(block["k"], block["mae"], "o-", label=arm, color=colour,
                    linewidth=1.4, markersize=4)
        ax.set_xlabel("measurements of the new extractant, k", fontsize=9)
        ax.set_ylabel("extractant-macro MAE (decades)", fontsize=9)
        ax.set_title("H6 — does a better structural level leave less for one measurement?",
                     fontsize=10, color=INK, loc="left")
        ax.legend(frameon=False, fontsize=7)
        style(ax)
        fig.tight_layout()
        fig.savefig(paths.FIGURE_DIR / "fig5_one_shot.png")
        plt.close(fig)
        values["fig5_one_shot"] = common[["arm", "k", "mae", "offset_abs", "shape_mae"]].to_dict(
            orient="records")

    (paths.FIGURE_DIR / "values.json").write_text(json.dumps(values, indent=1, default=float))
    print(f"{len(values)} figures written to {paths.FIGURE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
