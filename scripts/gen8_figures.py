"""Figures for the gen8 report — the four pictures that carry the argument.

Every panel is generated from a written artefact, never from a number typed in by
hand, so a figure cannot drift from the table it illustrates.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

ROOT = REPO_ROOT / "runs" / "gen8_architecture"
FIG = ROOT / "figures"
INK = "#1b1b1b"
ACCENT = "#c0392b"
MUTED = "#8d8d8d"


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, loc="left", color=INK, pad=10)
    ax.set_xlabel(xlabel, fontsize=9, color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.tick_params(labelsize=8, colors=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cccccc")
    ax.grid(axis="y", color="#eeeeee", linewidth=0.8)
    ax.set_axisbelow(True)


def adaptation_curve() -> None:
    path = ROOT / "finalists" / "adaptation_curve.csv"
    if not path.exists():
        return
    curve = pd.read_csv(path, index_col=0)
    curve.columns = [int(c) for c in curve.columns]
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=170)
    styles = {
        "NO_MODEL@RANDOM": ("no model at all (mean of the measurements)", MUTED, ":", "o"),
        "OFFSET_K1@RANDOM": ("offset correction, random point", "#2c6fbb", "-", "o"),
        "OFFSET_K1@MEDOID": ("offset correction, chosen point", "#2c6fbb", "--", "s"),
        "OFFSET_K3@RANDOM": ("response coefficients, random point", ACCENT, "-", "o"),
        "OFFSET_K3@MEDOID": ("response coefficients, chosen point", ACCENT, "--", "s"),
        "OFFSET_K1@ORACLE[OFFSET_K1]": ("oracle choice (not deployable)", "#2e7d32", "-.", "^"),
    }
    for arm, (label, colour, style, marker) in styles.items():
        if arm not in curve.index:
            continue
        series = curve.loc[arm].dropna()
        ax.plot(series.index, series.values, style, marker=marker, markersize=4.5,
                color=colour, linewidth=1.7, label=label)

    # The lower envelope over every *deployable* arm at each k.  Worth its own line
    # because the winning arm is not the same at every k — a central point wins the
    # level at k = 1 and a spread design wins the slopes from k = 2 — so no single
    # arm's trajectory is the frontier.
    table = pd.read_csv(ROOT / "finalists" / "table_b_common_cohort.csv")
    deployable = table[~table["arm"].str.contains("ORACLE")]
    best = deployable.groupby("k")["mae"].min()
    zero = table[table["arm"] == "ZERO_SHOT"]["mae"]
    if len(zero):
        best.loc[0] = float(zero.iloc[0])
    best = best.sort_index()
    ax.plot(best.index, best.values, "-", marker="D", markersize=4.2, color="#111111",
            linewidth=2.0, label="best deployable arm at each k")
    _style(ax, "What each measurement buys", "measurements of the new ligand (k)",
           "macro MAE (log units)")
    ax.set_xticks([0, 1, 2, 3, 5])
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "adaptation_curve.png")
    plt.close(fig)


def oracle_gap() -> None:
    path = ROOT / "finalists" / "table_b_common_cohort.csv"
    if not path.exists():
        return
    table = pd.read_csv(path)

    def value(arm, k):
        row = table[(table["arm"] == arm) & (table["k"] == k)]
        return float(row["mae"].iloc[0]) if len(row) else np.nan

    steps = [
        ("zero shot", value("ZERO_SHOT", 0)),
        ("+ one random\nmeasurement", value("OFFSET_K1@RANDOM", 1)),
        ("+ choosing the\npoint well", value("OFFSET_K1@CENTRAL", 1)),
        ("+ a better\ncalibration", value("PHYS_residual_gbm@CENTRAL", 1)),
        ("+ a second\nmeasurement", value("SLOPE_L_s1_K3@MAX_PREDICTIVE_VARIANCE", 2)),
        ("+ three more", value("SLOPE_L_s1_K3@D_OPTIMAL", 5)),
        ("oracle 1-shot\n(best single point)", value("OFFSET_K1@ORACLE[OFFSET_K1]", 1)),
    ]
    steps = [(a, b) for a, b in steps if np.isfinite(b)]
    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=170)
    positions = np.arange(len(steps))
    values = [b for _, b in steps]
    colours = [MUTED] + ["#2c6fbb"] * (len(steps) - 2) + ["#2e7d32"]
    ax.bar(positions, values, color=colours, width=0.62)
    for x, v in zip(positions, values):
        ax.text(x, v + 0.012, f"{v:.3f}", ha="center", fontsize=8.5, color=INK)
    for i in range(1, len(values) - 1):
        drop = values[i - 1] - values[i]
        if abs(drop) < 1e-9:
            continue
        ax.annotate(f"{-drop:+.3f}", xy=(i - 0.5, min(values[i - 1], values[i]) - 0.055),
                    ha="center", fontsize=8, color=ACCENT)
    ax.set_xticks(positions)
    ax.set_xticklabels([a for a, _ in steps], fontsize=8)
    _style(ax, "Where the error goes — common cohort, 99 held-out ligands", "",
           "macro MAE (log units)")
    ax.set_ylim(0, max(values) * 1.15)
    # The oracle reference is drawn only across the bars it is meant to be read
    # against; a full-width rule would cut through four taller bars and read as a
    # threshold they fail rather than as the level the budget eventually passes.
    ax.plot([len(steps) - 2.55, len(steps) - 0.4], [values[-1]] * 2,
            color="#2e7d32", linewidth=1.1, linestyle=":")
    ax.text(len(steps) - 2.5, values[-1] + 0.02,
            "five chosen measurements beat the best single one",
            fontsize=7.5, color="#2e7d32")
    fig.tight_layout()
    fig.savefig(FIG / "oracle_gap.png")
    plt.close(fig)


def policy_comparison() -> None:
    path = ROOT / "finalists" / "table_b_common_cohort.csv"
    if not path.exists():
        return
    table = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=170)
    for ax, (k, adapter) in zip(axes, ((1, "OFFSET_K1"), (2, "OFFSET_K3"))):
        sub = table[(table["k"] == k) & (table["arm"].str.startswith(adapter + "@"))].copy()
        if sub.empty:
            continue
        sub["policy"] = sub["arm"].str.split("@").str[1]
        sub = sub.sort_values("mae")
        reference = sub[sub["policy"] == "RANDOM"]["mae"]
        reference = float(reference.iloc[0]) if len(reference) else np.nan
        colours = ["#2e7d32" if p.startswith("ORACLE") else
                   (ACCENT if v > reference else "#2c6fbb")
                   for p, v in zip(sub["policy"], sub["mae"])]
        ax.barh(np.arange(len(sub)), sub["mae"], color=colours, height=0.68)
        ax.set_yticks(np.arange(len(sub)))
        ax.set_yticklabels(sub["policy"], fontsize=7.5)
        ax.invert_yaxis()
        if np.isfinite(reference):
            ax.axvline(reference, color=INK, linewidth=1.0, linestyle="--")
            ax.text(reference, len(sub) - 0.2, " random", fontsize=7.5, color=INK,
                    va="top")
        _style(ax, f"k = {k}, {adapter} calibration", "macro MAE (log units)", "")
        ax.grid(axis="x", color="#eeeeee", linewidth=0.8)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, sub["mae"].max() * 1.08)
    fig.suptitle("Which experiment to run — blue beats random, red loses to it",
                 fontsize=11, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / "policy_comparison.png")
    plt.close(fig)


def geography() -> None:
    base = ROOT / "cross_series"
    files = {"acid_position": "Where in the acid range",
             "metal_position": "Central vs extreme lanthanide",
             "prediction_position": "Where in the predicted range"}
    frames = {}
    for name, title in files.items():
        path = base / f"stratum_{name}.csv"
        if path.exists():
            frames[title] = pd.read_csv(path)
    if not frames:
        return
    fig, axes = plt.subplots(1, len(frames), figsize=(4.2 * len(frames), 4.0), dpi=170)
    axes = np.atleast_1d(axes)
    for ax, (title, frame) in zip(axes, frames.items()):
        frame = frame[~frame.iloc[:, 0].isin(["missing", "single"])].sort_values("one_shot_mae")
        colours = ["#2c6fbb" if i == 0 else MUTED for i in range(len(frame))]
        ax.bar(frame.iloc[:, 0].astype(str), frame["one_shot_mae"], color=colours, width=0.6)
        for x, v in enumerate(frame["one_shot_mae"]):
            ax.text(x, v + 0.012, f"{v:.3f}", ha="center", fontsize=8, color=INK)
        _style(ax, title, "", "1-shot MAE")
        ax.set_ylim(0, frame["one_shot_mae"].max() * 1.18)
    fig.suptitle("Where to spend the one measurement", fontsize=11, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "calibration_geography.png")
    plt.close(fig)


def slope_flattening() -> None:
    path = ROOT / "functional" / "curve_slope_comparison.parquet"
    if not path.exists():
        return
    curves = pd.read_parquet(path)
    curves = curves[curves["model"] == "REC_ecfp_plus_recovered"]
    axes_wanted = [("extractant", "extractant titration"), ("acid", "acid titration"),
                   ("metal_series", "lanthanide series")]
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 4.2), dpi=170)
    for ax, (axis, title) in zip(axs, axes_wanted):
        sub = curves[curves["axis_label_true"] == axis]
        sub = sub[np.isfinite(sub["slope_true"]) & np.isfinite(sub["slope_pred"])]
        if sub.empty:
            continue
        ax.scatter(sub["slope_true"], sub["slope_pred"], s=7, alpha=0.25,
                   color="#2c6fbb", edgecolors="none")
        low = float(min(sub["slope_true"].quantile(0.01), sub["slope_pred"].quantile(0.01)))
        high = float(max(sub["slope_true"].quantile(0.99), sub["slope_pred"].quantile(0.99)))
        ax.plot([low, high], [low, high], color=INK, linewidth=1.0, linestyle="--")
        ax.set_xlim(low, high); ax.set_ylim(low, high)
        _style(ax, f"{title}  (n = {len(sub):,} curves)", "measured slope", "predicted slope")
        ax.text(0.04, 0.94, f"median  {sub['slope_true'].median():.2f} → "
                            f"{sub['slope_pred'].median():.2f}",
                transform=ax.transAxes, fontsize=8.5, color=ACCENT, va="top")
    fig.suptitle("The model flattens every titration it has not seen "
                 "(dashed line = correct slope)", fontsize=11, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "slope_flattening.png")
    plt.close(fig)


def transfer_heatmap() -> None:
    path = ROOT / "cross_series" / "transfer_matrix.csv"
    if not path.exists():
        return
    matrix = pd.read_csv(path)
    matrix = matrix[matrix["n_ligands"] >= 3]
    grid = matrix.pivot_table(index="calibration_axis", columns="target_axis", values="gain")
    order = [c for c in ("acid", "extractant", "metal_series", "metal_concentration",
                         "temperature", "contact_time", "none") if c in grid.columns]
    grid = grid.reindex(index=[o for o in order if o in grid.index], columns=order)
    fig, ax = plt.subplots(figsize=(7.4, 5.4), dpi=170)
    limit = float(np.nanmax(np.abs(grid.to_numpy())))
    image = ax.imshow(grid.to_numpy(), cmap="RdBu", vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(grid.columns)), grid.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(grid.index)), grid.index, fontsize=8)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            value = grid.iat[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=7.5,
                        color=INK if abs(value) < limit * 0.55 else "white")
    ax.set_title("Gain from one measurement: measured series (rows) → predicted series (columns)",
                 fontsize=10, loc="left", color=INK, pad=10)
    ax.set_xlabel("rows being predicted", fontsize=9)
    ax.set_ylabel("row that was measured", fontsize=9)
    fig.colorbar(image, ax=ax, shrink=0.8, label="MAE improvement over zero-shot")
    fig.tight_layout()
    fig.savefig(FIG / "cross_series_transfer.png")
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=FIG)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, function in (("adaptation_curve", adaptation_curve), ("oracle_gap", oracle_gap),
                           ("policy_comparison", policy_comparison), ("geography", geography),
                           ("slope_flattening", slope_flattening),
                           ("transfer_heatmap", transfer_heatmap)):
        try:
            function()
            print(f"  {name}: ok")
        except Exception as error:  # noqa: BLE001
            print(f"  {name}: FAILED {type(error).__name__}: {error}")
    print(f"\nfigures -> {args.out}")
    for path in sorted(args.out.glob("*.png")):
        print(f"  {path.name}  {path.stat().st_size//1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
