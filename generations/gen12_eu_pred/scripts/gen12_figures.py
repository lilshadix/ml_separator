#!/usr/bin/env python
"""Figures.  Every plotted quantity is recomputed here from the raw predictions.

Usage::

    PYTHONPATH=gen12_eu_pred .venv/bin/python gen12_eu_pred/scripts/gen12_figures.py [--design B]

Each figure writes a companion ``*_values.json`` holding the numbers it draws, so
a caption can be checked without re-running the plot, and a reader can see that
the figure and the tables come from one computation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gen12eu import paths  # noqa: E402
from gen12eu.metrics import per_extractant  # noqa: E402

TARGET = "log_D"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 160, "savefig.bbox": "tight"})
COLOURS = {"far": "#B4432F", "mid": "#C89A3C", "near": "#3D7A6B"}


def _load(design: str) -> dict[str, pd.DataFrame]:
    out = {}
    for path in sorted((paths.PREDICTION_DIR / design).glob("*.parquet")):
        if path.stem == "similarity":
            continue
        frame = pd.read_parquet(path)
        if "prediction" in frame.columns:
            out[frame["arm"].iloc[0]] = frame
    return out


def figure_leaderboard(frames, design):
    table = pd.read_csv(paths.METRIC_DIR / design / "leaderboard.csv")
    table = table.sort_values("macro_mae_extractant", ascending=False)
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    colours = ["#3D7A6B" if a.startswith("T") else "#9A9A9A" for a in table["arm"]]
    ax.barh(table["arm"], table["macro_mae_extractant"], color=colours, height=0.62)
    for y, (value, arm) in enumerate(zip(table["macro_mae_extractant"], table["arm"])):
        ax.text(value + 0.012, y, f"{value:.3f}", va="center", fontsize=8)
    best = table["macro_mae_extractant"].min()
    ax.axvline(best, color="#3D7A6B", lw=0.8, ls=":")
    ax.set_xlabel("zero-shot extractant-macro MAE (log$_{10}$ D units), design "
                  f"{design}, 5 split seeds")
    ax.set_xlim(0, table["macro_mae_extractant"].max() * 1.12)
    fig.savefig(paths.FIGURE_DIR / f"fig1_leaderboard_{design}.png")
    plt.close(fig)
    return {"leaderboard": table[["arm", "macro_mae_extractant"]].to_dict(orient="records")}


def figure_frontier(frames, design, arms=("T1_EXTRATREES", "B1_COND_ONLY")):
    """Error against chemical distance — the study's central claim."""
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    values = {}
    ax = axes[0]
    for arm, marker, colour in zip(arms, ("o", "s"), ("#3D7A6B", "#9A6BA8")):
        if arm not in frames:
            continue
        units = per_extractant(frames[arm])
        units = units.groupby("extractant").agg(
            mae=("mae", "mean"), similarity=("max_train_tanimoto", "mean"),
            n_rows=("n_rows", "mean")).reset_index()
        ax.scatter(units["similarity"], units["mae"], s=6 + 2.2 * np.sqrt(units["n_rows"]),
                   alpha=0.55, marker=marker, color=colour, linewidths=0, label=arm)
        order = np.argsort(units["similarity"].to_numpy())
        x = units["similarity"].to_numpy()[order]
        y = units["mae"].to_numpy()[order]
        window = max(9, len(x) // 8)
        smooth = pd.Series(y).rolling(window, center=True, min_periods=3).median()
        ax.plot(x, smooth, color=colour, lw=1.6)
        values[arm] = {"n_extractants": int(len(units)),
                       "spearman_similarity_vs_error": float(
                           pd.Series(x).corr(pd.Series(y), method="spearman"))}
    for name, low, high in (("far", 0, 0.40), ("mid", 0.40, 0.60), ("near", 0.60, 0.70)):
        ax.axvspan(low, high, color=COLOURS[name], alpha=0.07)
        ax.text((low + high) / 2, ax.get_ylim()[1] * 0.97, name, ha="center", va="top",
                fontsize=8, color=COLOURS[name])
    ax.set_xlabel("max Tanimoto to training chemistry")
    ax.set_ylabel("per-extractant MAE")
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    ax = axes[1]
    bands = pd.read_csv(paths.METRIC_DIR / design / "bands.csv")
    keep = [a for a in arms if a in set(bands["arm"])]
    order = ["far", "mid", "near"]
    width = 0.36
    for offset, (arm, colour) in enumerate(zip(keep, ("#3D7A6B", "#9A6BA8"))):
        block = bands[bands["arm"] == arm].set_index("band").reindex(order)
        ax.bar(np.arange(3) + (offset - 0.5) * width, block["macro_mae_extractant"],
               width=width, color=colour, label=arm)
        values.setdefault(arm, {})["by_band"] = block["macro_mae_extractant"].round(4).to_dict()
    counts = bands[bands["arm"] == keep[0]].set_index("band").reindex(order)["n_extractant_units"]
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels([f"{b}\nn={int(c)}" for b, c in zip(order, counts)])
    ax.set_ylabel("extractant-macro MAE")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(paths.FIGURE_DIR / f"fig2_similarity_frontier_{design}.png")
    plt.close(fig)
    return values


def figure_fewshot(design):
    path = paths.METRIC_DIR / design / "fewshot_curve.csv"
    if not path.exists():
        return {}
    curve = pd.read_csv(path)
    common = curve[(curve["cohort"] == "common")]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3))
    ax = axes[0]
    styles = {("T1_EXTRATREES", "A_OFFSET_SHRUNK"): ("#3D7A6B", "-", "model + shrunk offset"),
              ("T1_EXTRATREES", "A0_OFFSET_PLAIN"): ("#3D7A6B", "--", "model + plain offset"),
              ("B1_COND_ONLY", "A_OFFSET_SHRUNK"): ("#9A6BA8", "-", "conditions only + offset"),
              ("T1_EXTRATREES", "NULL_NO_MODEL"): ("#B4432F", ":", "no model: mean of measured")}
    values = {}
    for (arm, adapter), (colour, style, label) in styles.items():
        block = common[(common["arm"] == arm) & (common["adapter"] == adapter)].sort_values("k")
        if block.empty:
            continue
        ax.plot(block["k"], block["mae"], marker="o", ms=4, color=colour, ls=style, label=label)
        values[label] = dict(zip(block["k"].astype(int), block["mae"].round(4)))
    ax.set_xlabel("k measurements of the new extractant")
    ax.set_ylabel("extractant-macro MAE (42-extractant common cohort)")
    ax.set_xticks([0, 1, 2, 3, 5])
    ax.legend(frameon=False, fontsize=7.5)

    ax = axes[1]
    block = common[(common["arm"] == "T1_EXTRATREES")
                   & (common["adapter"] == "A_OFFSET_SHRUNK")].sort_values("k")
    ax.plot(block["k"], block["offset_abs"], marker="o", ms=4, color="#B4432F", label="level |bias|")
    ax.plot(block["k"], block["shape_mae"], marker="s", ms=4, color="#3D7A6B", label="shape MAE")
    ax.set_xlabel("k")
    ax.set_ylabel("error component")
    ax.set_xticks([0, 1, 2, 3, 5])
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=8)
    values["level"] = dict(zip(block["k"].astype(int), block["offset_abs"].round(4)))
    values["shape"] = dict(zip(block["k"].astype(int), block["shape_mae"].round(4)))
    fig.savefig(paths.FIGURE_DIR / f"fig3_fewshot_{design}.png")
    plt.close(fig)
    return values


def figure_cohort(design):
    cohort = pd.read_parquet(paths.MANIFEST_DIR / "cohort_identity.parquet")
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9))
    ax = axes[0]
    ax.hist(cohort[TARGET], bins=40, color="#3D7A6B", alpha=0.85)
    ax.set_xlabel("log$_{10}$ D (Eu)"); ax.set_ylabel("cells")
    ax = axes[1]
    counts = cohort.groupby("extractant").size().sort_values(ascending=False)
    ax.plot(np.arange(1, len(counts) + 1), counts.to_numpy(), color="#9A6BA8")
    ax.set_yscale("log"); ax.set_xlabel("extractant rank"); ax.set_ylabel("cells")
    ax.set_title(f"{len(counts)} extractants; top = {int(counts.iloc[0])} cells", fontsize=8)
    ax = axes[2]
    sizes = cohort.groupby("chemotype")["extractant"].nunique().sort_values(ascending=False)
    ax.plot(np.arange(1, len(sizes) + 1), sizes.to_numpy(), color="#B4432F")
    ax.set_yscale("log"); ax.set_xlabel("chemotype rank"); ax.set_ylabel("extractants")
    ax.set_title(f"{len(sizes)} chemotypes; largest = {int(sizes.iloc[0])}", fontsize=8)
    fig.savefig(paths.FIGURE_DIR / f"fig4_cohort_{design}.png")
    plt.close(fig)
    return {"n_extractants": int(cohort["extractant"].nunique()),
            "n_chemotypes": int(cohort["chemotype"].nunique()),
            "largest_extractant_cells": int(counts.iloc[0])}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", default="B")
    args = parser.parse_args()
    frames = _load(args.design)
    values = {
        "leaderboard": figure_leaderboard(frames, args.design),
        "frontier": figure_frontier(frames, args.design),
        "fewshot": figure_fewshot(args.design),
        "cohort": figure_cohort(args.design),
    }
    (paths.FIGURE_DIR / f"values_{args.design}.json").write_text(
        json.dumps(values, indent=1, default=str))
    print(json.dumps(values["frontier"], indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
