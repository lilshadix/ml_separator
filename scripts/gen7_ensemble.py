"""Stacking and error-correlation analysis over gen7's out-of-fold predictions (brief §17).

Two models that make the *same* errors add nothing when averaged, however good each
is alone.  So this script leads with the residual correlation matrix and only then
fits weights.

The weighting rule is the part that has to be right.  Fitting stack weights on the
outer test rows would be selection on the test set — the exact sin the brief forbids
— so weights are fitted **per outer fold on that fold's training rows**, using each
contender's *inner* out-of-fold predictions produced by re-running it on an inner
chemotype split.  That is expensive, so the cheaper and still-honest alternative is
offered and used by default: **leave-one-chemotype-out weighting**, where the weights
applied to a held-out chemotype are fitted on the OOF rows of every *other*
chemotype.  A chemotype never contributes to the weights used to score it.

Three combiners, in increasing order of how much they can overfit:

``mean``        equal weights.  No fitting at all, and the baseline the others must beat.
``inverse_mae`` weights proportional to 1/MAE on the other chemotypes.
``nnls``        non-negative least squares on the other chemotypes, then renormalised.
                Non-negativity matters: an unconstrained stack happily assigns a
                large negative weight to a weak model and does not survive a new fold.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lanthanide_separation.gen6.metrics import (  # noqa: E402
    gen6_metric_table, paired_unit_bootstrap, per_unit_statistics,
)


def wide_frame(oof: pd.DataFrame, models: list[str], seed: int) -> pd.DataFrame:
    block = oof[(oof["split_seed"] == seed) & (oof["model"].isin(models))]
    identity = ["row_id", "extractant", "ecfp_cluster", "tanimoto_cluster", "log_D",
                "nn_train_tanimoto", "fold"]
    identity = [c for c in identity if c in block.columns]
    base = block[block["model"] == models[0]][identity].reset_index(drop=True)
    for model in models:
        part = block[block["model"] == model][["row_id", "prediction"]]
        base = base.merge(part.rename(columns={"prediction": f"prediction_{model}"}),
                          on="row_id", how="left", validate="one_to_one")
    missing = base.filter(like="prediction_").isna().any().any()
    if missing:
        raise AssertionError("a model is missing predictions for some rows")
    return base


def _weights(kind: str, truth: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    n = predictions.shape[1]
    if kind == "mean":
        return np.full(n, 1.0 / n)
    if kind == "inverse_mae":
        mae = np.abs(predictions - truth[:, None]).mean(0)
        inverse = 1.0 / np.maximum(mae, 1e-6)
        return inverse / inverse.sum()
    if kind == "nnls":
        from scipy.optimize import nnls
        weights, _ = nnls(predictions, truth)
        total = weights.sum()
        return weights / total if total > 1e-9 else np.full(n, 1.0 / n)
    raise ValueError(f"unknown combiner {kind!r}")


def stack(frame: pd.DataFrame, models: list[str], kind: str) -> np.ndarray:
    """Leave-one-chemotype-out weighting: a chemotype never weights itself."""
    columns = [f"prediction_{m}" for m in models]
    predictions = frame[columns].to_numpy(float)
    truth = frame["log_D"].to_numpy(float)
    groups = frame["tanimoto_cluster"].astype(str).to_numpy()
    out = np.empty(len(frame))
    for group in np.unique(groups):
        held = groups == group
        other = ~held
        if other.sum() < 10:
            weights = np.full(len(models), 1.0 / len(models))
        else:
            weights = _weights(kind, truth[other], predictions[other])
        out[held] = predictions[held] @ weights
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, nargs="+", required=True)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--top", type=int, default=6,
                        help="if --models is absent, take the best N by macro MAE")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "runs" / "gen7_architecture" / "ensemble")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    parts = [pd.read_parquet(p) for p in args.oof]
    oof = pd.concat(parts, ignore_index=True).drop_duplicates(["model", "split_seed", "row_id"])
    seeds = sorted(oof["split_seed"].unique())
    # keep only models present on every seed, else the comparison moves with the model
    complete = [m for m, block in oof.groupby("model")
                if set(block["split_seed"].unique()) == set(seeds)]
    oof = oof[oof["model"].isin(complete)]

    if args.models:
        models = [m for m in args.models if m in complete]
    else:
        ranking = []
        for model, block in oof.groupby("model"):
            per_seed = []
            for seed in seeds:
                sub = block[block["split_seed"] == seed]
                per_cluster = (sub.assign(_e=(sub["prediction"] - sub["log_D"]).abs())
                               .groupby("ecfp_cluster")["_e"].mean())
                per_seed.append(float(per_cluster.mean()))
            ranking.append((float(np.mean(per_seed)), model))
        ranking.sort()
        models = [m for _, m in ranking[: args.top]]
    if len(models) < 2:
        raise SystemExit(f"need at least two models, have {models}")
    print(f"ensembling {len(models)} models: {models}", flush=True)

    correlations: list[pd.DataFrame] = []
    scores: list[pd.DataFrame] = []
    for seed in seeds:
        frame = wide_frame(oof, models, seed)
        residual = pd.DataFrame({
            m: frame[f"prediction_{m}"] - frame["log_D"] for m in models})
        correlation = residual.corr()
        correlations.append(correlation.assign(split_seed=seed))
        for kind in ("mean", "inverse_mae", "nnls"):
            frame[f"prediction_ENS_{kind}"] = stack(frame, models, kind)
        arms = models + [f"ENS_{k}" for k in ("mean", "inverse_mae", "nnls")]
        overall, _, hard = gen6_metric_table(
            frame, arms, similarity_column="nn_train_tanimoto", prediction_prefix="prediction_")
        overall["split_seed"] = seed
        for endpoint, tag in (("nn<0.4", "nn_lt_0_4"), ("nn<0.6", "nn_lt_0_6")):
            sub = hard[hard["endpoint"] == endpoint]
            mapping = dict(zip(sub["arm"], sub["macro_mae"]))
            overall[f"{tag}__macro_mae"] = overall["arm"].map(mapping)
        scores.append(overall)
        if seed == seeds[0]:
            per_unit = per_unit_statistics(frame, arms)
            block_of_unit = (frame.drop_duplicates("ecfp_cluster")
                             .set_index(frame.drop_duplicates("ecfp_cluster")["ecfp_cluster"]
                                        .astype(str))["tanimoto_cluster"].astype(str).to_dict())
            best_single = models[0]
            comparisons = {f"ENS_{k}_vs_{best_single}": (best_single, f"ENS_{k}")
                           for k in ("mean", "inverse_mae", "nnls")}
            bootstrap = paired_unit_bootstrap(per_unit, comparisons, block_of_unit=block_of_unit)
            bootstrap.to_csv(args.out / "ensemble_bootstrap_seed0.csv", index=False)

    pd.concat(correlations).to_csv(args.out / "residual_correlations.csv")
    board = pd.concat(scores, ignore_index=True)
    board.to_csv(args.out / "ensemble_scores_by_seed.csv", index=False)
    numeric = board.select_dtypes("number").columns
    summary = board.groupby("arm")[list(numeric)].mean().sort_values("macro_mae").reset_index()
    summary.to_csv(args.out / "ensemble_leaderboard.csv", index=False)

    pd.set_option("display.width", 200)
    columns = [c for c in ["arm", "macro_mae", "offset_mae", "shape_mae", "pooled_mae",
                           "nn_lt_0_4__macro_mae"] if c in summary.columns]
    print(summary[columns].to_string(index=False))
    mean_correlation = pd.concat(correlations).groupby(level=0)[models].mean()
    print("\nmean residual correlation:")
    print(mean_correlation.round(3).to_string())
    (args.out / "summary.json").write_text(json.dumps({"models": models, "seeds": [int(s) for s in seeds]}, indent=2))
    print(f"\nartifacts -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
