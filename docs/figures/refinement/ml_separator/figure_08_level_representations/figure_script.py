#!/usr/bin/env python
"""Figure 8 (supplementary) — no learned representation predicts a ligand's level.

Refined from ``scripts/gen7_figures.py::figure_level_capture``
(render: ``runs/gen7_architecture/figures/level_capture.png``).

The numbers are the ones written by ``scripts/gen7_level_benchmark.py``; nothing is
re-fitted, re-pooled or re-derived here.  This pass changes which numbers are shown, the
plotting grammar, the axis definition and the uncertainty treatment — see ``notes.md``.

Run from anywhere:
    python figure_refinement/ml_separator/figure_08_level_representations/figure_script.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl                # noqa: E402
import matplotlib.pyplot as plt         # noqa: E402
import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402
from matplotlib.colors import to_rgba   # noqa: E402


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common"))
import mlsep_data as D                  # noqa: E402
import pubstyle as PS                   # noqa: E402

BENCH = "gen7_architecture/level_benchmark"

#: Reader-facing name for every feature block the benchmark evaluated.  Block sizes are
#: the ones documented in ``docs/gen5_levels_protocol_20260817.md`` (PHYSCHEM 10, ECFP
#: 2,048 Morgan-radius-2 bits, LIG2D_EXT 206, DONORS 13, POLYHEDRON 58, COMPLEX_PHYS 26)
#: and, for the pretrained embeddings, the column counts in
#: ``dataset with 3D structures/ligand_pretrained_embeddings.parquet``
#: (ChemBERTa mean-pooled 384-d, MoLFormer mean-pooled 768-d).  ``_pca16`` is a
#: 16-component PCA fitted on the fold's training ligands.
REPRESENTATIONS: dict[str, str] = {
    "donors": "Donor-atom census; 13 features",
    "physchem": "Physicochemical scalars; 10 features",
    "donors_physchem": "Donor census + physicochemical; 23 features",
    "donors_physchem_pca16": "Donor census + physicochemical; 16 principal components",
    "lig2d": "Extended 2D descriptors; 206 features",
    "lig2d_pca16": "Extended 2D descriptors; 16 principal components",
    "donors_physchem_lig2d": "Donors + physicochemical + 2D descriptors; 229 features",
    "donors_physchem_lig2d_pca16": "Donors + physicochemical + 2D descriptors; 16 principal components",
    "ecfp": "ECFP4 fingerprint; 2,048 bits",
    "ecfp_pca16": "ECFP4 fingerprint; 16 principal components",
    "chemberta": "ChemBERTa embedding; 384 dimensions",
    "chemberta_pca16": "ChemBERTa embedding; 16 principal components",
    "molformer": "MoLFormer embedding; 768 dimensions",
    "molformer_pca16": "MoLFormer embedding; 16 principal components",
    "molformer_donors": "MoLFormer embedding + donor census + physicochemical",
    "molformer_donors_pca16": "MoLFormer embedding + donor census; 16 principal components",
    "polyhedron3d": "Coordination-polyhedron geometry (3D); 58 features",
    "polyhedron3d_pca16": "Coordination-polyhedron geometry (3D); 16 principal components",
    "donors_3d": "Donors + physicochemical + xTB complex properties (3D)",
    "donors_3d_pca16": "Donors + physicochemical + 3D; 16 principal components",
}

#: Reader-facing name for every regressor recipe in the benchmark.
RECIPES: dict[str, str] = {
    "ridge": "ridge regression",
    "bayesridge": "Bayesian ridge",
    "pls3": "partial least squares, 3 components",
    "svr_rbf": "support-vector regression, RBF kernel",
    "extratrees": "extremely randomised trees",
    "gbdt_depth2": "gradient boosting, depth 2",
}

#: The five rows of panel A: one compact descriptor set, one large descriptor set, one
#: fingerprint, one pretrained embedding, one fusion.  Chosen before looking at the
#: condition-adjusted ranking; between them they hold both champions (the donor census
#: wins the raw target, the ChemBERTa embedding the condition-adjusted one).  Two-line
#: labels so the panel keeps its width.
SHOWN: dict[str, str] = {
    "chemberta_pca16": "ChemBERTa embedding\n16 principal components",
    "donors": "Donor-atom census\n13 features",
    "donors_physchem_lig2d": "Donors + physicochemical\n+ 2D descriptors, 229 features",
    "lig2d": "Extended 2D descriptors\n206 features",
    "ecfp": "ECFP4 fingerprint\n2,048 bits",
}

RAW = PS.ORANGE          # the raw ligand mean — the quantity that carries the confound
ADJ = PS.BLUE            # the condition-adjusted level — the quantity about chemistry
BIN = 0.4                # percentage points, panel B dot-stack
#: Chemotype folds per seed.  Fixed in ``lanthanide_separation.gen7.harness.build_folds``
#: and not recorded in the leaderboard, so it is the one protocol constant named here.
FOLDS_PER_SEED = 5


def leaderboard(tag: str) -> pd.DataFrame:
    return pd.read_csv(D.run(f"{BENCH}/level_leaderboard_{tag}.csv"))


def cohort(tag: str) -> dict:
    """``n_ligands`` and the seed list the benchmark actually ran."""
    return json.loads(D.run(f"{BENCH}/summary_{tag}.json").read_text())


def summarise(tag: str) -> dict:
    """Nulls, and the best recipe per feature block, for one target definition.

    ``best`` reproduces the predecessor figure's selection exactly: ``idxmin`` over the
    six regressor recipes evaluated for that feature block.  No penalty is applied for
    that selection — see the winner's-curse note in ``notes.md``.
    """
    board = leaderboard(tag)
    nulls = board[board["model"].str.startswith("NULL")].set_index("model")
    null_mae = float(nulls.loc["NULL_mean", "level_mae"])
    models = board[~board["model"].str.startswith("NULL")].copy()
    models["reduction"] = 100 * (null_mae - models["level_mae"]) / null_mae
    models["sd_points"] = 100 * models["level_mae_sd_over_folds"] / null_mae
    hard_null = float(nulls.loc["NULL_mean", "level_mae_nn_lt_0_4"])
    models["hard_reduction"] = 100 * (hard_null - models["level_mae_nn_lt_0_4"]) / hard_null
    best = models.loc[models.groupby("features")["level_mae"].idxmin()].set_index("features")
    return {
        "tag": tag,
        "null_mae": null_mae,
        "null_sd": float(nulls.loc["NULL_mean", "level_mae_sd_over_folds"]),
        "lookup_mae": float(nulls.loc["NULL_nn_tanimoto", "level_mae"]),
        "lookup_reduction": 100 * (null_mae - float(nulls.loc["NULL_nn_tanimoto", "level_mae"])) / null_mae,
        "lookup_sd_points": 100 * float(nulls.loc["NULL_nn_tanimoto", "level_mae_sd_over_folds"]) / null_mae,
        "n_hard": int(board["n_hard"].iloc[0]),
        "n_ligand_evals": int(board["n_ligand_evals"].iloc[0]),
        # The same three quantities restricted to the evaluations whose nearest training
        # ligand is below Tanimoto 0.4.  Not plotted — there is no per-fold dispersion for
        # a subset — but it qualifies the pooled claim and is reported in values.json.
        "hard_null_mae": float(nulls.loc["NULL_mean", "level_mae_nn_lt_0_4"]),
        "hard_lookup_mae": float(nulls.loc["NULL_nn_tanimoto", "level_mae_nn_lt_0_4"]),
        "hard_lookup_reduction": 100 * (
            float(nulls.loc["NULL_mean", "level_mae_nn_lt_0_4"])
            - float(nulls.loc["NULL_nn_tanimoto", "level_mae_nn_lt_0_4"])
        ) / float(nulls.loc["NULL_mean", "level_mae_nn_lt_0_4"]),
        "all_fits": models,
        "best": best,
    }


def dot_stack(values: np.ndarray, bin_width: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wilkinson-style dot positions: one dot per value, stacked inside its bin.

    Returns ``(x, y, order)`` where ``order`` maps each drawn dot back to its row in
    ``values`` so a highlighted fit can be found again.
    """
    edges = np.floor(values.min() / bin_width) * bin_width + bin_width * np.arange(
        int(np.ceil((values.max() - values.min()) / bin_width)) + 2)
    index = np.clip(np.digitize(values, edges) - 1, 0, len(edges) - 2)
    x = np.empty(len(values))
    y = np.empty(len(values))
    order = np.argsort(index, kind="stable")
    height: dict[int, int] = {}
    for position in order:
        b = int(index[position])
        height[b] = height.get(b, 0) + 1
        x[position] = edges[b] + bin_width / 2
        y[position] = height[b]
    return x, y, order


def main() -> int:
    PS.apply()
    mpl.rcParams["axes.titlepad"] = 17.0     # clears the reference-line labels

    raw = summarise("raw")
    adjusted = summarise("adjusted")
    ligandsplit = summarise("raw_ligandsplit")
    run = cohort("adjusted")
    n_folds = len(run["seeds"]) * FOLDS_PER_SEED

    keys = list(SHOWN)
    rows = np.arange(len(keys), dtype=float)
    offset = 0.17

    fig = plt.figure(figsize=(PS.W2, 4.90))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.32, 1.00], hspace=0.12)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[1, 0])

    # ------------------------------------------------------------------ A ------
    # One shared x-axis for both target definitions, so the collapse from ~29 % to
    # ~6 % is a distance on the page rather than a comparison between two panels
    # with unannounced scales.
    for y, key in zip(rows, keys):
        r, a = raw["best"].loc[key], adjusted["best"].loc[key]
        axA.annotate("", xy=(a["reduction"], y + offset), xytext=(r["reduction"], y - offset),
                     arrowprops=dict(arrowstyle="-|>,head_length=0.28,head_width=0.13",
                                     color=PS.PALE, linewidth=0.9, shrinkA=3.2, shrinkB=3.2))
    axA.hlines(rows, -100, 100, color="#F0F0F0", linewidth=0.6, zorder=0)

    for series, colour, marker, dy in ((raw, RAW, "s", -offset),
                                       (adjusted, ADJ, "o", offset)):
        point = np.array([series["best"].loc[k, "reduction"] for k in keys])
        spread = np.array([series["best"].loc[k, "sd_points"] for k in keys])
        axA.errorbar(point, rows + dy, xerr=spread, fmt=marker, color=colour,
                     markersize=4.2, elinewidth=0.9, capsize=1.8,
                     ecolor=to_rgba(colour, 0.5), zorder=3)

    axA.set_ylim(4.78, -1.08)
    axA.set_xlim(-28.5, 49.0)
    axA.set_yticks(rows)
    axA.set_yticklabels(list(SHOWN.values()))
    axA.tick_params(axis="y", length=0)

    # Reference rules stop short of both label bands, so no rule crosses text.
    axA.vlines(0.0, -0.47, 4.40, color=PS.GREY, linewidth=0.8, zorder=1)
    for series, colour in ((raw, RAW), (adjusted, ADJ)):
        axA.vlines(series["lookup_reduction"], -0.47, 4.40, color=colour,
                   linestyle=(0, (4, 2)), linewidth=1.0, zorder=2)

    # Series identity is a key drawn in the empty band above the top row: two entries do
    # not need a legend box, and a legend box here would have to sit on the data.
    for x0, marker, colour, text in ((-3.2, "o", ADJ, "condition-adjusted level"),
                                     (18.0, "s", RAW, "raw ligand mean")):
        axA.plot([x0], [-0.70], marker=marker, color=colour, markersize=4.2,
                 linestyle="none", clip_on=False)
        axA.text(x0 + 1.6, -0.70, text, ha="left", va="center",
                 fontsize=PS.BASE - 0.5, color=colour)
    # What the whiskers are, in the empty band under the lowest row.
    axA.text(48.0, 4.62, f"whiskers: ±1 sd of the fold-level MAE over {n_folds} folds",
             ha="right", va="center", fontsize=PS.BASE - 1.0, color=PS.GREY)

    # Reference-line labels live outside the plotting area, one running left from the
    # adjusted rule and one running right from the raw rule, so they cannot collide.
    axA.annotate(f"nearest-neighbour lookup  {adjusted['lookup_reduction']:.1f}%",
                 xy=(adjusted["lookup_reduction"], 1.0), xycoords=("data", "axes fraction"),
                 xytext=(-3, 3), textcoords="offset points", ha="right", va="bottom",
                 fontsize=PS.BASE - 1.0, color=ADJ, annotation_clip=False)
    axA.annotate(f"nearest-neighbour lookup  {raw['lookup_reduction']:.1f}%",
                 xy=(raw["lookup_reduction"], 1.0), xycoords=("data", "axes fraction"),
                 xytext=(3, 3), textcoords="offset points", ha="left", va="bottom",
                 fontsize=PS.BASE - 1.0, color=RAW, annotation_clip=False)

    axA.set_xlabel(
        "Reduction in level MAE against the training-mean null (%);  positive = better than the null\n"
        "null = predict the training-set mean level: MAE "
        f"{raw['null_mae']:.3f} (raw ligand mean), {adjusted['null_mae']:.3f} "
        "(condition-adjusted), log$_{10}$ $D$ units")
    # "and target": the two markers in a row are two separate selections, because the
    # recipe that wins on the raw ligand mean is not always the one that wins on the
    # condition-adjusted level.  The arrow connects representations, not one fitted model.
    PS.strapline(axA, f"Best of six regressors per representation and target  ·  unseen-chemotype "
                      f"hold-out  ·  {run['n_ligands']} extractants, "
                      f"{adjusted['n_ligand_evals']} held-out evaluations")

    # ------------------------------------------------------------------ B ------
    # Every fit on the condition-adjusted target, not only the per-block winner: the
    # spread over recipes is the selection the panel-A markers are the maximum of.
    fits = adjusted["all_fits"].reset_index(drop=True)
    values = fits["reduction"].to_numpy(float)
    x, y, _ = dot_stack(values, BIN)
    highlight = fits["features"].isin(keys).to_numpy() & np.isin(
        fits["level_mae"].to_numpy(),
        [adjusted["best"].loc[k, "level_mae"] for k in keys])

    axB.scatter(x[~highlight], y[~highlight], s=8.0, facecolor="white",
                edgecolor=PS.GREY, linewidth=0.6, zorder=2)
    axB.scatter(x[highlight], y[highlight], s=11.0, facecolor=ADJ, edgecolor="white",
                linewidth=0.4, zorder=3)

    top = float(y.max())
    span = (-7.2, 10.6)
    axB.set_ylim(0, top + 1.5)
    axB.set_xlim(*span)
    axB.set_xticks(np.arange(-6, 11, 2))
    axB.vlines(0.0, 0, top + 0.7, color=PS.GREY, linewidth=0.8, zorder=1)
    axB.vlines(adjusted["lookup_reduction"], 0, top + 0.7, color=ADJ,
               linestyle=(0, (4, 2)), linewidth=1.0, zorder=1)
    axB.set_yticks(np.arange(0, top + 1.5, 4))
    axB.set_ylabel("model fits (count)")

    best_all = float(values.max())
    # Anchor the leader on the *drawn* dot (its bin centre), not on the value it carries.
    best_x = float(x[int(np.argmax(values))])
    axB.annotate(f"best of all {len(values)} fits  {best_all:.1f}%",
                 xy=(best_x, 1.2), xytext=(best_x, 7.2), ha="center", va="bottom",
                 fontsize=PS.BASE - 1.0, color=ADJ,
                 arrowprops=dict(arrowstyle="-", color=ADJ, linewidth=0.7,
                                 shrinkA=1.5, shrinkB=0.8))
    axB.annotate(f"nearest-neighbour lookup  {adjusted['lookup_reduction']:.1f}%",
                 xy=(adjusted["lookup_reduction"], 1.0), xycoords=("data", "axes fraction"),
                 xytext=(-3, 3), textcoords="offset points", ha="right", va="bottom",
                 fontsize=PS.BASE - 1.0, color=ADJ, annotation_clip=False)

    # The pooled metric is what the leaderboard reports, and the lookup's advantage in it
    # comes from held-out ligands that do have a close training relative.  Saying so on
    # the figure keeps the panel from reading as a stronger claim than the data supports.
    axB.text(-7.1, top - 0.4,
             f"Pooled over all {adjusted['n_ligand_evals']} evaluations.\n"
             f"On the {adjusted['n_hard']} whose nearest training\n"
             "ligand is below Tanimoto 0.4, the\n"
             f"lookup itself falls to {adjusted['hard_lookup_reduction']:.1f}%.".replace("-", "−"),
             ha="left", va="top", fontsize=PS.BASE - 1.0, color=PS.GREY, linespacing=1.35)

    mean_sd = float(fits["sd_points"].mean())
    axB.set_xlabel(
        "Reduction in level MAE against the training-mean null (%), condition-adjusted level\n"
        f"panel A's whiskers (±1 sd of the fold-level MAE, ±{mean_sd:.0f} points) are "
        f"{2 * mean_sd / (span[1] - span[0]):.1f}× as wide as this entire axis")
    PS.strapline(axB, "Every fit on the condition-adjusted level  ·  20 representations "
                      "× 6 regressor recipes  ·  filled = the five shown in A")

    axA.spines["left"].set_visible(False)

    letters = PS.add_panel_letters(fig, [axA, axB])
    # Both letters to the leftmost of the two measured positions: panel B's y tick labels
    # are two characters wide and panel A's are two lines of prose, so the measured
    # offsets differ by 15 % of the figure width and the column would read as ragged.
    left = min(text.get_position()[0] for text in letters)
    for text in letters:
        text.set_x(left)
    report = PS.save(fig, HERE, "figure", strict=False)
    print(report)

    # ------------------------------------------------------------- values ------
    def block(series: dict) -> dict:
        out = {"null_mae": round(series["null_mae"], 6),
               "null_sd_over_folds": round(series["null_sd"], 6),
               "nn_lookup_mae": round(series["lookup_mae"], 6),
               "nn_lookup_reduction_pct": round(series["lookup_reduction"], 4),
               "nn_lookup_sd_points": round(series["lookup_sd_points"], 3),
               "hard_subset_nn_tanimoto_below_0_4": {
                   "n_evaluations": series["n_hard"],
                   "null_mae": round(series["hard_null_mae"], 6),
                   "nn_lookup_mae": round(series["hard_lookup_mae"], 6),
                   "nn_lookup_reduction_pct": round(series["hard_lookup_reduction"], 4),
                   "n_fits_better_than_null": int((series["all_fits"]["hard_reduction"] > 0).sum()),
                   "best_fit_reduction_pct": round(float(series["all_fits"]["hard_reduction"].max()), 4),
               },
               "best_per_representation": {}}
        for key, row in series["best"].iterrows():
            out["best_per_representation"][REPRESENTATIONS.get(key, key)] = {
                "feature_key": key,
                "winning_recipe": RECIPES[row["model"]],
                "recipe_key": row["model"],
                "level_mae": round(float(row["level_mae"]), 6),
                "sd_over_folds": round(float(row["level_mae_sd_over_folds"]), 6),
                "reduction_pct": round(float(row["reduction"]), 4),
                "sd_points": round(float(row["sd_points"]), 3),
                "hard_subset_reduction_pct": round(float(row["hard_reduction"]), 4),
            }
        return out

    values_out = {
        "figure": "figure_08_level_representations",
        "role": "supplementary",
        "source": {
            "raw ligand mean, chemotype hold-out": f"runs/{BENCH}/level_leaderboard_raw.csv",
            "condition-adjusted level, chemotype hold-out": f"runs/{BENCH}/level_leaderboard_adjusted.csv",
            "raw ligand mean, ligand hold-out (not plotted)":
                f"runs/{BENCH}/level_leaderboard_raw_ligandsplit.csv",
            "producer": "scripts/gen7_level_benchmark.py",
        },
        "protocol": {
            "n_ligands": int(run["n_ligands"]), "n_seeds": len(run["seeds"]),
            "n_folds": n_folds,
            "n_ligand_evaluations": adjusted["n_ligand_evals"],
            "n_hard_evaluations_nn_tanimoto_below_0_4": adjusted["n_hard"],
            "x_quantity": "100 * (MAE_null - MAE_model) / MAE_null, MAE in log10 D units",
            "null": "predict the training-set mean level",
            "selection": "best of six regressor recipes per feature block (idxmin, no penalty)",
        },
        "panelA": {
            "rows_top_to_bottom": [REPRESENTATIONS[k] for k in keys],
            "raw ligand mean": block(raw),
            "condition-adjusted level": block(adjusted),
        },
        "panelB": {
            "n_fits": int(len(values)),
            "bin_width_points": BIN,
            "min_reduction_pct": round(float(values.min()), 4),
            "max_reduction_pct": round(float(values.max()), 4),
            "median_reduction_pct": round(float(np.median(values)), 4),
            "n_fits_above_nn_lookup": int((values > adjusted["lookup_reduction"]).sum()),
            "mean_sd_points": round(mean_sd, 3),
            "all_fits": [
                {"representation": REPRESENTATIONS.get(f, f), "recipe": RECIPES[m],
                 "level_mae": round(float(v), 6), "reduction_pct": round(float(r), 4),
                 "sd_points": round(float(s), 3)}
                for f, m, v, r, s in zip(fits["features"], fits["model"], fits["level_mae"],
                                         fits["reduction"], fits["sd_points"])],
        },
        "not_plotted_ligand_holdout": block(ligandsplit),
        "lint": {"ok": report.ok, "violations": [str(v) for v in report.violations]},
    }
    (HERE / "values.json").write_text(json.dumps(values_out, indent=1))

    print(f"panel B: {len(values)} fits, max stack {top:.0f}, "
          f"best {best_all:.2f}%, lookup {adjusted['lookup_reduction']:.2f}%, "
          f"above lookup {int((values > adjusted['lookup_reduction']).sum())}")
    for key in keys:
        print(f"{key:26s} raw {raw['best'].loc[key, 'reduction']:6.2f}%  "
              f"adjusted {adjusted['best'].loc[key, 'reduction']:6.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
