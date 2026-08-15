#!/usr/bin/env python3
"""Extractant-screening utility of an unseen-extractant OOF table.

The project's deployment goal is separating lanthanides *by choosing an
extractant*: for a target metal pair, rank candidate extractants by predicted
separation power and pick the best. Pooled R2 does not measure that decision;
this script does, from ``cross_seed_oof_predictions.csv`` (every prediction
there is out-of-fold, so each extractant is scored by a model that never saw
it).

Per metal pair (with a minimum number of candidate extractants):

- ``spearman``: cross-extractant rank correlation between predicted and
  measured separability ``|log SF|`` (each extractant summarised at its own
  measured conditions — a screening, not a matched-conditions, comparison);
- ``in_top3``: whether the predicted-best extractant is truly in the top 3;
- ``regret``: measured ``|log SF|`` of the true best minus the predicted-best
  pick, in log units (0 = perfect pick), with the random-pick regret printed
  for scale;
- overall sign accuracy (direction of the separation).

Writes ``screening_utility.csv`` next to the input (or to --output-dir).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("predictions_csv", type=Path)
    parser.add_argument("--arm", default="A2")
    parser.add_argument("--min-extractants", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    table = pd.read_csv(args.predictions_csv)
    prediction_column = f"prediction_{args.arm}_mean"
    if prediction_column not in table.columns:
        raise SystemExit(f"missing column {prediction_column}")
    table["metal_pair"] = table["metal_A"] + "/" + table["metal_B"]

    summarised = (
        table.groupby(["metal_pair", "extractant"])
        .agg(
            measured=("y_true", "mean"),
            predicted=(prediction_column, "mean"),
            n_rows=("y_true", "size"),
        )
        .reset_index()
    )

    rows = []
    for metal_pair, group in summarised.groupby("metal_pair"):
        if len(group) < args.min_extractants:
            continue
        measured = np.abs(group["measured"].to_numpy())
        predicted = np.abs(group["predicted"].to_numpy())
        chosen = float(measured[int(np.argmax(predicted))])
        top3_floor = float(np.sort(measured)[-min(3, len(measured))])
        rows.append(
            {
                "metal_pair": metal_pair,
                "n_extractants": int(len(group)),
                "spearman": float(spearmanr(predicted, measured).statistic),
                "best_true_abs_logSF": float(measured.max()),
                "chosen_true_abs_logSF": chosen,
                "regret": float(measured.max() - chosen),
                "random_pick_regret": float(measured.max() - measured.mean()),
                "predicted_best_in_true_top3": bool(chosen >= top3_floor),
            }
        )
    result = pd.DataFrame(rows).sort_values("spearman")

    output_dir = args.output_dir or args.predictions_csv.parent
    output_path = output_dir / "screening_utility.csv"
    result.to_csv(output_path, index=False)

    sign_accuracy = float(
        np.mean(np.sign(table["y_true"]) == np.sign(table[prediction_column]))
    )
    print(
        f"metal pairs scored: {len(result)}  "
        f"(>= {args.min_extractants} candidate extractants)"
    )
    print(
        f"cross-extractant spearman(|logSF|): median "
        f"{result['spearman'].median():+.3f}, "
        f"positive {int((result['spearman'] > 0).sum())}/{len(result)}"
    )
    print(
        f"predicted-best truly top-3: "
        f"{int(result['predicted_best_in_true_top3'].sum())}/{len(result)}"
    )
    print(
        f"regret: median {result['regret'].median():.3f} log units "
        f"(random pick {result['random_pick_regret'].mean():.3f})"
    )
    print(f"sign accuracy: {100 * sign_accuracy:.1f}%")
    print(f"written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
