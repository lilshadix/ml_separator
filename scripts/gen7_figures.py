"""The three figures the gen7 argument actually needs.

1. **The error budget.** A waterfall from the global mean down to a perfect oracle,
   showing where each log unit lives. This is the figure that makes the headline
   ("all of chemistry is worth 0.09") legible in one look.
2. **Does chemistry help more when the chemistry is closer?** Paired per-ligand gain of
   the best chemistry-aware model over the no-ligand model, against nearest-training-
   neighbour Tanimoto, split into level and shape. A working representation slopes
   upward. Ours does not.
3. **The level sub-problem.** What fraction of the ligand level each representation
   captures under three targets — raw, ligand-split, and condition-adjusted — with the
   1-nearest-neighbour lookup drawn as the line to beat.

Written to ``runs/gen7_architecture/figures/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lanthanide_separation.gen6.metrics import decompose_level_shape  # noqa: E402

OUT = REPO_ROOT / "runs" / "gen7_architecture"
FIGURES = OUT / "figures"
INK, ACCENT, MUTED = "#1b1b1b", "#0b6e4f", "#9aa0a6"


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)


def figure_error_budget(scores: pd.DataFrame) -> None:
    order = ["NULL_global_mean", "NULL_pair_mean", "NULL_metal_cond", "REAL_best_tree",
             "ORACLE_level", "ORACLE_batch", "ORACLE_cell"]
    label = {"NULL_global_mean": "training mean\n(no information)",
             "NULL_pair_mean": "training median",
             "NULL_metal_cond": "metal + conditions\nNO ligand at all",
             "REAL_best_tree": "best model\n(all chemistry)",
             "ORACLE_level": "+ true ligand level\n(oracle)",
             "ORACLE_batch": "+ true study calibration\n(oracle)",
             "ORACLE_cell": "perfect"}
    available = [m for m in order if m in set(scores["model"])]
    values = [float(scores.loc[scores["model"] == m, "macro_mae"].mean()) for m in available]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    colours = [MUTED if m.startswith("NULL") else ACCENT if m == "REAL_best_tree" else "#7fb3a3"
               for m in available]
    bars = ax.bar(range(len(available)), values, color=colours, width=0.62)
    for i, (bar, value) in enumerate(zip(bars, values)):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}",
                ha="center", fontsize=9, color=INK)
        if i:
            ax.annotate("", xy=(i, values[i]), xytext=(i - 1, values[i - 1]),
                        arrowprops=dict(arrowstyle="-", color="#cfcfcf", linewidth=0.9))
            ax.text(i - 0.5, (values[i] + values[i - 1]) / 2 + 0.05,
                    f"{values[i] - values[i-1]:+.3f}", ha="center", fontsize=8, color="#666")
    ax.set_xticks(range(len(available)))
    ax.set_xticklabels([label.get(m, m) for m in available], fontsize=8.5)
    ax.set_ylabel("macro MAE (one ECFP cluster, one vote)", fontsize=9.5)
    ax.set_title("Where the error lives on a held-out chemotype\n"
                 "every descriptor, fingerprint, 3D block and pretrained embedding, together, "
                 "is the third step", fontsize=11, color=INK, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "error_budget.png", dpi=200)
    plt.close(fig)


def figure_gain_vs_similarity(oof: pd.DataFrame, best: str, null: str) -> None:
    rows = []
    for (model, seed), block in oof[oof["model"].isin([best, null])].groupby(
            ["model", "split_seed"]):
        table = decompose_level_shape(block["log_D"], block["prediction"],
                                      block["extractant"]).per_ligand
        table["model"], table["split_seed"] = model, seed
        rows.append(table)
    per = pd.concat(rows, ignore_index=True)
    nn = oof.groupby(["split_seed", "extractant"])["nn_train_tanimoto"].max().rename("nn")
    per = per.merge(nn.reset_index(), on=["split_seed", "extractant"])
    wide = per.pivot_table(index=["split_seed", "extractant", "nn"],
                           columns="model", values=["offset_abs", "shape_mae"]).reset_index()
    wide.columns = ["_".join(c).strip("_") for c in wide.columns]
    wide["d_offset"] = wide[f"offset_abs_{null}"] - wide[f"offset_abs_{best}"]
    wide["d_shape"] = wide[f"shape_mae_{null}"] - wide[f"shape_mae_{best}"]

    edges = [0.0, 0.35, 0.45, 0.55, 0.65, 1.01]
    wide["bin"] = pd.cut(wide["nn"], edges)
    grouped = wide.groupby("bin", observed=True)
    # Plotted on a categorical axis with the bin's ACTUAL observed range as the label.
    # Bin midpoints would put a marker at 0.83, implying a training neighbour above the
    # 0.7 single-linkage threshold — which the chemotype hold-out makes impossible.
    positions = np.arange(len(grouped))
    labels = []
    for name, block in grouped:
        labels.append(f"{block['nn'].min():.2f}–{block['nn'].max():.2f}\nn={len(block)}")
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for offset, (column, colour, name) in zip(
            (-0.07, 0.07), (("d_offset", ACCENT, "level (offset)"),
                            ("d_shape", "#b3541e", "shape"))):
        mean = grouped[column].mean().to_numpy()
        sem = (grouped[column].std() / np.sqrt(grouped[column].size())).to_numpy()
        ax.errorbar(positions + offset, mean, yerr=1.96 * sem, marker="o", color=colour,
                    linewidth=1.6, capsize=3, label=name)
    ax.axhline(0.0, color=INK, linewidth=1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_xlabel("nearest-neighbour Tanimoto from the held-out ligand to training chemistry\n"
                  "(single-linkage chemotype hold-out at 0.7 caps this below 0.7)", fontsize=9)
    ax.set_ylabel("improvement from ALL ligand chemistry\n(log units, positive = helps)",
                  fontsize=9.5)
    ax.set_title("A working representation would slope upward with similarity\n"
                 "the level gain is flat and consistent with zero at every distance "
                 "(overall p = 0.42); only the shape gain is real (p = 0.007)",
                 fontsize=10.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "gain_vs_similarity.png", dpi=200)
    plt.close(fig)


def figure_level_capture() -> None:
    panels = [("level_leaderboard_raw.csv", "raw ligand mean\nchemotype hold-out"),
              ("level_leaderboard_raw_ligandsplit.csv", "raw ligand mean\nLIGAND hold-out"),
              ("level_leaderboard_adjusted.csv", "condition-adjusted level\nchemotype hold-out")]
    available = [(f, t) for f, t in panels if (OUT / "level_benchmark" / f).exists()]
    if not available:
        return
    # NOT sharey: each panel ranks a different set of representations, and sharing the
    # axis silently printed panel 1's labels beside panel 3's bars.
    fig, axes = plt.subplots(1, len(available), figsize=(5.0 * len(available), 4.8))
    axes = np.atleast_1d(axes)
    for ax, (filename, title) in zip(axes, available):
        board = pd.read_csv(OUT / "level_benchmark" / filename)
        null = float(board.loc[board["model"] == "NULL_mean", "level_mae"].iloc[0])
        neighbour = float(board.loc[board["model"] == "NULL_nn_tanimoto", "level_mae"].iloc[0])
        models = board[~board["model"].str.startswith("NULL")]
        best = models.loc[models.groupby("features")["level_mae"].idxmin()]
        best = best.sort_values("level_mae").head(8)
        captured = 100 * (null - best["level_mae"]) / null
        ax.barh(range(len(best)), captured, color=ACCENT, height=0.6)
        ax.set_yticks(range(len(best)))
        ax.set_yticklabels(best["features"], fontsize=8)
        for y, (value, model) in enumerate(zip(captured, best["model"])):
            ax.text(max(value, 0) + 0.4, y, model, va="center", fontsize=7, color="#555")
        ax.invert_yaxis()
        ax.axvline(100 * (null - neighbour) / null, color="#b3541e", linewidth=1.6,
                   label="1-NN Tanimoto lookup")
        ax.set_xlabel("% of ligand-level spread captured", fontsize=9)
        ax.set_title(title, fontsize=10, color=INK)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        style(ax)
        ax.grid(axis="x", color="#e6e6e6", linewidth=0.8)
    fig.suptitle("Predicting a ligand's absolute level from its structure",
                 fontsize=11.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIGURES / "level_capture.png", dpi=200)
    plt.close(fig)


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    written = []
    oracle_scores = OUT / "oracles" / "scores_by_seed.csv"
    finalist_scores = OUT / "finalists" / "scores_by_seed.csv"
    frames = [pd.read_csv(p) for p in (oracle_scores, finalist_scores) if p.exists()]
    if frames:
        figure_error_budget(pd.concat(frames, ignore_index=True))
        written.append("error_budget.png")
    oracle_oof = OUT / "oracles" / "oof_predictions.parquet"
    if oracle_oof.exists():
        oof = pd.read_parquet(oracle_oof)
        if {"REAL_best_tree", "NULL_metal_cond"} <= set(oof["model"]):
            figure_gain_vs_similarity(oof, "REAL_best_tree", "NULL_metal_cond")
            written.append("gain_vs_similarity.png")
    figure_level_capture()
    if (FIGURES / "level_capture.png").exists():
        written.append("level_capture.png")
    print("wrote:", ", ".join(written) or "(nothing — artifacts not ready)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
