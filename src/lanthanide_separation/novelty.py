"""Structural-novelty sensitivity for leave-extractant-out evaluation.

The primary protocol holds out literal ``canonical_smiles`` groups.  That is a
correct leave-extractant-out design, but it does not guarantee that a held-out
ligand is *structurally* new: this cohort contains homolog series whose 2048-bit
Morgan fingerprints are bit-identical, so a "unseen" extractant can have a
training partner with Tanimoto 1.0.  The audit records
``ecfp_exact_cluster_overlap = 1`` in at least one fold for exactly that reason.

This module does not modify the primary split.  It is a declared *secondary*
analysis that re-reads the frozen out-of-fold predictions and asks a different
question:

    Does the incremental value of a feature block change as the held-out ligand
    becomes more structurally novel relative to the ligands it was trained on?

For every held-out extractant it computes the maximum Tanimoto similarity to any
extractant in that fold's **training** set -- never to the rest of the test set,
and never using the target.  Extractants are then stratified by that maximum,
and the paired ``ΔMAE = MAE(reference) - MAE(candidate)`` is reported per
stratum with the extractant as the unit of inference.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


DEFAULT_NOVELTY_THRESHOLDS: tuple[float, ...] = (0.9, 0.8, 0.7)
ECFP_COLUMN_PREFIX = "base__ecfp_"


def extractant_fingerprints(frame: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """Return one binary fingerprint per extractant, validated for consistency."""

    fingerprint_columns = [
        column for column in frame.columns if str(column).startswith(ECFP_COLUMN_PREFIX)
    ]
    if not fingerprint_columns:
        raise ValueError(
            "Structural-novelty analysis needs the base__ecfp_* block in the "
            "pair frame."
        )
    values = frame.loc[:, fingerprint_columns].apply(pd.to_numeric, errors="coerce")
    array = values.to_numpy(dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("ECFP block contains missing or non-finite values.")
    if not np.isin(array, (0.0, 1.0)).all():
        raise ValueError("ECFP block must be binary.")

    extractants = frame["extractant"].astype(str)
    grouped = pd.DataFrame(array, index=extractants.to_numpy())
    unique_per_extractant = grouped.groupby(level=0).nunique().max(axis=1)
    if bool((unique_per_extractant > 1).any()):
        raise ValueError(
            "One extractant carries more than one fingerprint; the novelty "
            "analysis would be ill-defined."
        )
    collapsed = grouped.groupby(level=0).first()
    return list(collapsed.index.astype(str)), collapsed.to_numpy(dtype=bool)


def tanimoto_matrix(fingerprints: np.ndarray) -> np.ndarray:
    """Pairwise Tanimoto similarity of binary fingerprint rows."""

    bits = fingerprints.astype(np.float64)
    intersection = bits @ bits.transpose()
    counts = bits.sum(axis=1)
    union = counts[:, None] + counts[None, :] - intersection
    with np.errstate(invalid="ignore", divide="ignore"):
        similarity = np.where(union > 0.0, intersection / union, 0.0)
    return np.clip(similarity, 0.0, 1.0)


def held_out_novelty(
    frame: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    *,
    group_column: str = "extractant",
) -> pd.DataFrame:
    """Maximum train-set Tanimoto for every held-out extractant, per fold.

    ``fold_assignments`` must carry one row per pair with ``pair_id`` and
    ``outer_fold``.  Similarity is computed only against extractants that were
    in that fold's training set, so no test-set information is used.
    """

    names, fingerprints = extractant_fingerprints(frame)
    similarity = tanimoto_matrix(fingerprints)
    index_of = {name: position for position, name in enumerate(names)}

    folds = fold_assignments.loc[:, ["pair_id", "outer_fold"]].drop_duplicates()
    pair_to_group = (
        frame.loc[:, ["pair_id", group_column]]
        .drop_duplicates()
        .set_index("pair_id")[group_column]
        .astype(str)
    )
    folds = folds.assign(group=folds["pair_id"].map(pair_to_group).astype(str))

    rows: list[dict[str, Any]] = []
    for outer_fold, fold_frame in folds.groupby("outer_fold", sort=True):
        test_groups = sorted(set(fold_frame["group"]))
        train_groups = sorted(set(folds["group"]) - set(test_groups))
        if not train_groups:
            raise ValueError(f"Outer fold {outer_fold} has an empty training set.")
        train_positions = [index_of[name] for name in train_groups]
        for name in test_groups:
            similarities = similarity[index_of[name], train_positions]
            best = int(np.argmax(similarities))
            rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "extractant_id": name,
                    "n_train_extractants": len(train_groups),
                    "max_train_tanimoto": float(similarities[best]),
                    "nearest_train_extractant": train_groups[best],
                    "mean_train_tanimoto": float(np.mean(similarities)),
                }
            )
    return pd.DataFrame(rows).sort_values(
        "max_train_tanimoto", ascending=False, ignore_index=True
    )


def _group_delta_table(
    predictions: pd.DataFrame,
    reference: str,
    candidate: str,
    *,
    target_column: str,
    group_column: str = "extractant",
) -> pd.DataFrame:
    """Per-extractant MAE of both arms and their paired difference."""

    rows: list[dict[str, Any]] = []
    for group, group_frame in predictions.groupby(group_column, sort=True):
        truth = group_frame[target_column].to_numpy(dtype=float)
        reference_error = float(
            np.mean(np.abs(truth - group_frame[f"prediction_{reference}"].to_numpy(float)))
        )
        candidate_error = float(
            np.mean(np.abs(truth - group_frame[f"prediction_{candidate}"].to_numpy(float)))
        )
        rows.append(
            {
                "extractant_id": str(group),
                "n_pairs": int(len(group_frame)),
                f"{reference}_mae": reference_error,
                f"{candidate}_mae": candidate_error,
                "delta_mae": reference_error - candidate_error,
            }
        )
    return pd.DataFrame(rows)


def novelty_stratified_deltas(
    predictions: pd.DataFrame,
    novelty: pd.DataFrame,
    comparisons: Sequence[tuple[str, str]],
    *,
    target_column: str,
    group_column: str = "extractant",
    thresholds: Iterable[float] = DEFAULT_NOVELTY_THRESHOLDS,
    n_bootstrap: int = 2000,
    seed: int = 8675309,
) -> pd.DataFrame:
    """Paired macro delta MAE inside declared structural-novelty strata.

    Every statistic gives one vote per held-out extractant, so a single
    combinatorially large all-pairs panel cannot dominate the estimate.  Strata
    with fewer than two extractants are reported with the count and a null
    interval rather than being silently dropped.
    """

    novelty_by_group = (
        novelty.loc[:, ["extractant_id", "max_train_tanimoto"]]
        .drop_duplicates("extractant_id")
        .set_index("extractant_id")["max_train_tanimoto"]
    )
    threshold_values = sorted({float(value) for value in thresholds}, reverse=True)
    rng = np.random.default_rng(int(seed))

    rows: list[dict[str, Any]] = []
    for reference, candidate in comparisons:
        table = _group_delta_table(
            predictions,
            reference,
            candidate,
            target_column=target_column,
            group_column=group_column,
        )
        table = table.assign(
            max_train_tanimoto=table["extractant_id"].map(novelty_by_group)
        )
        if table["max_train_tanimoto"].isna().any():
            missing = sorted(
                table.loc[table["max_train_tanimoto"].isna(), "extractant_id"]
            )
            raise ValueError(f"No novelty score for held-out extractants: {missing}")

        strata: list[tuple[str, pd.Series]] = [
            ("all", pd.Series(True, index=table.index))
        ]
        for threshold in threshold_values:
            strata.append(
                (
                    f"max_train_tanimoto_lt_{threshold:g}",
                    table["max_train_tanimoto"] < threshold,
                )
            )
            strata.append(
                (
                    f"max_train_tanimoto_ge_{threshold:g}",
                    table["max_train_tanimoto"] >= threshold,
                )
            )

        for label, mask in strata:
            stratum = table.loc[mask]
            values = stratum["delta_mae"].to_numpy(dtype=float)
            row: dict[str, Any] = {
                "comparison": f"{reference}_vs_{candidate}",
                "reference": reference,
                "candidate": candidate,
                "stratum": label,
                "n_extractants": int(len(stratum)),
                "n_pairs": int(stratum["n_pairs"].sum()),
                "macro_delta_mae": float(values.mean()) if len(values) else np.nan,
                "median_delta_mae": float(np.median(values)) if len(values) else np.nan,
                "extractants_improved": int(np.sum(values > 0.0)),
                "fraction_improved": (
                    float(np.mean(values > 0.0)) if len(values) else np.nan
                ),
                "ci95_low": np.nan,
                "ci95_high": np.nan,
            }
            if len(values) >= 2 and n_bootstrap >= 1:
                draws = rng.choice(values, size=(int(n_bootstrap), len(values)))
                means = draws.mean(axis=1)
                row["ci95_low"] = float(np.quantile(means, 0.025))
                row["ci95_high"] = float(np.quantile(means, 0.975))
            rows.append(row)
    return pd.DataFrame(rows)


def novelty_report(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    comparisons: Sequence[tuple[str, str]],
    *,
    target_column: str,
    group_column: str = "extractant",
    thresholds: Iterable[float] = DEFAULT_NOVELTY_THRESHOLDS,
    n_bootstrap: int = 2000,
    seed: int = 8675309,
) -> dict[str, Any]:
    """Complete secondary structural-novelty analysis for one run."""

    novelty = held_out_novelty(frame, fold_assignments, group_column=group_column)
    stratified = novelty_stratified_deltas(
        predictions,
        novelty,
        comparisons,
        target_column=target_column,
        group_column=group_column,
        thresholds=thresholds,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    similarities = novelty["max_train_tanimoto"].to_numpy(dtype=float)
    return {
        "schema_version": "1.0",
        "status": "secondary_sensitivity_analysis",
        "protocol": {
            "primary_split_modified": False,
            "similarity": "Tanimoto over the bundled binary Morgan fingerprints",
            "direction": (
                "maximum similarity from each held-out extractant to the "
                "training extractants of its own outer fold"
            ),
            "uses_target": False,
            "inference_unit": "held-out extractant",
            "bootstrap_replicates": int(n_bootstrap),
        },
        "novelty_distribution": {
            "n_held_out_extractants": int(len(novelty)),
            "min": float(similarities.min()) if len(similarities) else None,
            "median": float(np.median(similarities)) if len(similarities) else None,
            "max": float(similarities.max()) if len(similarities) else None,
            "n_with_identical_training_partner": int(np.sum(similarities >= 1.0)),
            "thresholds": [float(value) for value in sorted(set(thresholds))],
        },
        "per_extractant_novelty": novelty.to_dict(orient="records"),
        "stratified_deltas": stratified.to_dict(orient="records"),
        "interpretation_guardrail": (
            "A stratum is only informative if it contains enough held-out "
            "extractants to support a group-level interval; counts are reported "
            "next to every estimate and thresholds that empty the test "
            "population are not forced."
        ),
    }
