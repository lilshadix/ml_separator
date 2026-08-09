"""Leakage-safe, all-pairs feature-family ablation evaluation.

The legacy evaluator compares a fixed 2D contract with one Delta3D contract.
This module implements the pre-specified A0--A6 experiment without replacing
that historical path.  Every arm uses one shared outer fold plan and one shared
inner plan per outer fold; all learned preprocessing lives inside the fitted
training fold.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .evaluation import (
    DEFAULT_PARAMETER_GRID,
    AntisymmetricExtraTreesRegressor,
    _group_folds,
)
from .feature_registry import FeatureRegistry, build_feature_registry
from .pairs import PAIR_TARGET_COLUMN, PairDataset


METRIC_NAMES: tuple[str, ...] = ("mae", "rmse", "r2", "pearson", "spearman")
PRIMARY_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("A2_vs_A5", "A2", "A5"),
    ("A2_vs_A6", "A2", "A6"),
    ("A3_vs_A2", "A3", "A2"),
)


@dataclass(frozen=True)
class AblationBenchmarkResult:
    """All machine-readable tables needed by the ablation runner."""

    summary: dict[str, Any]
    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    fold_assignments: pd.DataFrame
    fold_memberships: pd.DataFrame
    inner_fold_assignments: pd.DataFrame
    tuning_results: pd.DataFrame
    preprocessing_audit: pd.DataFrame
    shuffle_audit: pd.DataFrame
    per_ablation_metrics: pd.DataFrame
    paired_deltas: pd.DataFrame
    per_extractant_metrics: pd.DataFrame
    per_lanthanide_metrics: pd.DataFrame
    leakage_audit: dict[str, Any]
    geometry_qc_summary: dict[str, Any]
    feature_audit: dict[str, Any]

    @property
    def fold_membership(self) -> pd.DataFrame:
        """Backward-compatible singular alias used by artifact writers."""

        return self.fold_memberships

    @property
    def preprocessing_parameters(self) -> pd.DataFrame:
        """Alias emphasizing that imputer state is persisted, not only checked."""

        return self.preprocessing_audit


def _finite_metrics(
    truth: Iterable[float], prediction: Iterable[float]
) -> dict[str, float | int]:
    y_true = np.asarray(list(truth), dtype=float)
    y_pred = np.asarray(list(prediction), dtype=float)
    if len(y_true) != len(y_pred) or not len(y_true):
        raise ValueError("Metric inputs must be non-empty and have equal length.")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Metric inputs must be finite.")

    def correlation(left: np.ndarray, right: np.ndarray) -> float:
        if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
            return float("nan")
        return float(np.corrcoef(left, right)[0, 1])

    true_rank = pd.Series(y_true).rank(method="average").to_numpy(dtype=float)
    pred_rank = pd.Series(y_pred).rank(method="average").to_numpy(dtype=float)
    return {
        "n_rows": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": (
            float(r2_score(y_true, y_pred))
            if len(y_true) >= 2 and np.var(y_true) > 0.0
            else float("nan")
        ),
        "pearson": correlation(y_true, y_pred),
        "spearman": correlation(true_rank, pred_rank),
    }


def _macro_group_mae(
    truth: np.ndarray, prediction: np.ndarray, groups: np.ndarray
) -> float:
    table = pd.DataFrame(
        {
            "group": groups.astype(str),
            "absolute_error": np.abs(truth - prediction),
        }
    )
    return float(table.groupby("group", sort=False)["absolute_error"].mean().mean())


def _delta_metrics(
    truth: np.ndarray, reference: np.ndarray, candidate: np.ndarray
) -> dict[str, float]:
    reference_metrics = _finite_metrics(truth, reference)
    candidate_metrics = _finite_metrics(truth, candidate)
    return {
        "delta_mae": float(reference_metrics["mae"] - candidate_metrics["mae"]),
        "delta_rmse": float(reference_metrics["rmse"] - candidate_metrics["rmse"]),
        "delta_r2": float(candidate_metrics["r2"] - reference_metrics["r2"]),
        "delta_spearman": float(
            candidate_metrics["spearman"] - reference_metrics["spearman"]
        ),
    }


def _training_shuffle(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Detach complete 3D vectors between training-only complex-pair groups.

    A complex can occur in several condition rows.  Shuffling individual rows
    would give repeated uses of the same geometry mutually inconsistent 3D
    vectors.  We therefore identify a complex pair by its two geometry keys
    (falling back to the extractant identity for synthetic fixtures), verify
    that its complete local-3D block is constant, and permute whole groups only
    within the same lanthanide-pair label.  No validation/test row is supplied
    to this function.
    """

    if not columns:
        raise ValueError("A shuffled control requires at least one 3D column.")
    if "pair_label" not in frame or "extractant" not in frame:
        raise ValueError("A shuffled control requires pair_label and extractant IDs.")
    if {"geometry_key_A", "geometry_key_B"}.issubset(frame.columns):
        unit_columns = ("geometry_key_A", "geometry_key_B")
    else:
        unit_columns = ("extractant",)

    result = frame.copy()
    rng = np.random.default_rng(int(seed))
    moved_rows = 0
    fixed_rows = 0
    moved_groups = 0
    fixed_groups = 0
    repeated_groups = 0
    for _, pair_frame in frame.groupby("pair_label", sort=True, dropna=False):
        grouped = pair_frame.groupby(list(unit_columns), sort=True, dropna=False)
        group_positions = [np.asarray(value, dtype=int) for value in grouped.indices.values()]
        # GroupBy.indices above is relative to pair_frame. Convert to positions
        # in the full training frame before assignment.
        pair_positions = frame.index.get_indexer(pair_frame.index)
        group_positions = [pair_positions[local] for local in group_positions]
        representatives: list[np.ndarray] = []
        for positions in group_positions:
            values = frame.iloc[positions].loc[:, list(columns)]
            if any(values[column].nunique(dropna=False) > 1 for column in columns):
                raise AssertionError(
                    "Repeated rows for one complex pair have inconsistent 3D vectors."
                )
            representatives.append(values.iloc[0].to_numpy(copy=True))
            repeated_groups += int(len(positions) > 1)

        group_count = len(group_positions)
        if group_count <= 1:
            donor_indices = np.arange(group_count)
        else:
            randomized_order = rng.permutation(group_count)
            shift = int(rng.integers(1, group_count))
            donor_for_randomized_order = np.roll(randomized_order, shift)
            donor_indices = np.empty(group_count, dtype=int)
            donor_indices[randomized_order] = donor_for_randomized_order

        for recipient_index, positions in enumerate(group_positions):
            donor_index = int(donor_indices[recipient_index])
            donor_vector = representatives[donor_index]
            result.loc[result.index[positions], list(columns)] = donor_vector
            if donor_index == recipient_index:
                fixed_groups += 1
                fixed_rows += int(len(positions))
            else:
                moved_groups += 1
                moved_rows += int(len(positions))
    return result, {
        "training_rows": int(len(frame)),
        "shuffle_unit_columns": ",".join(unit_columns),
        "training_complex_pair_groups": int(moved_groups + fixed_groups),
        "repeated_complex_pair_groups": int(repeated_groups),
        "moved_groups": int(moved_groups),
        "fixed_groups": int(fixed_groups),
        "moved_rows": int(moved_rows),
        "fixed_rows": int(fixed_rows),
        "donors_from_training_only": True,
        "complete_block_moved_together": True,
    }


def _identity_overlap(
    train: pd.DataFrame, test: pd.DataFrame, columns: tuple[str, ...]
) -> int:
    available = tuple(column for column in columns if column in train.columns)
    if not available:
        return 0
    train_values = set(
        train.loc[:, list(available)].astype(str).to_numpy().ravel().tolist()
    )
    test_values = set(
        test.loc[:, list(available)].astype(str).to_numpy().ravel().tolist()
    )
    return int(len(train_values & test_values))


def _pair_qc_labels(frame: pd.DataFrame) -> pd.Series:
    geometry_ok_a = frame.get("geometry_ok_A", pd.Series(True, index=frame.index)).fillna(False)
    geometry_ok_b = frame.get("geometry_ok_B", pd.Series(True, index=frame.index)).fillna(False)
    failed = ~geometry_ok_a.astype(bool) | ~geometry_ok_b.astype(bool)
    qc_a = frame.get(
        "geometry_qc_class_A", pd.Series(pd.NA, index=frame.index, dtype="string")
    ).astype("string")
    qc_b = frame.get(
        "geometry_qc_class_B", pd.Series(pd.NA, index=frame.index, dtype="string")
    ).astype("string")
    high = qc_a.str.upper().eq("OK").fillna(False) & qc_b.str.upper().eq("OK").fillna(False)
    labels = pd.Series("accepted_unclassified", index=frame.index, dtype="string")
    labels.loc[high] = "high_confidence"
    labels.loc[~failed & ~high & (qc_a.notna() | qc_b.notna())] = "questionable_or_recovered"
    labels.loc[failed] = "failed_geometry"
    return labels


def _bootstrap_comparisons(
    predictions: pd.DataFrame,
    comparisons: list[tuple[str, str, str]],
    *,
    group_column: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if replicates < 1:
        return {"replicates": 0, "comparisons": {}}
    rng = np.random.default_rng(int(seed))
    unique_groups = predictions[group_column].astype(str).drop_duplicates().to_numpy()
    group_indices = {
        group: np.flatnonzero(predictions[group_column].astype(str).to_numpy() == group)
        for group in unique_groups
    }
    draws: dict[str, dict[str, list[float]]] = {
        name: {
            key: []
            for key in (
                "delta_mae",
                "delta_macro_group_mae",
                "delta_rmse",
                "delta_r2",
                "delta_spearman",
            )
        }
        for name, _, _ in comparisons
    }
    for _ in range(int(replicates)):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        positions = np.concatenate([group_indices[group] for group in sampled_groups])
        sample = predictions.iloc[positions]
        truth = sample[PAIR_TARGET_COLUMN].to_numpy(dtype=float)
        if len(truth) < 2 or np.var(truth) == 0.0:
            continue
        for name, reference, candidate in comparisons:
            reference_prediction = sample[f"prediction_{reference}"].to_numpy(
                dtype=float
            )
            candidate_prediction = sample[f"prediction_{candidate}"].to_numpy(
                dtype=float
            )
            delta = _delta_metrics(
                truth,
                reference_prediction,
                candidate_prediction,
            )
            for metric, value in delta.items():
                if np.isfinite(value):
                    draws[name][metric].append(float(value))
            # The extractant is the sampling unit.  Give every sampled group
            # one vote so a single combinatorially large all-pairs panel cannot
            # dominate the primary uncertainty statement.
            group_deltas = []
            start = 0
            for group in sampled_groups:
                group_size = len(group_indices[group])
                stop = start + group_size
                group_truth = truth[start:stop]
                group_deltas.append(
                    float(
                        np.mean(np.abs(group_truth - reference_prediction[start:stop]))
                        - np.mean(np.abs(group_truth - candidate_prediction[start:stop]))
                    )
                )
                start = stop
            draws[name]["delta_macro_group_mae"].append(
                float(np.mean(group_deltas))
            )

    def interval(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"n": 0, "mean": None, "ci95_low": None, "ci95_high": None}
        array = np.asarray(values, dtype=float)
        return {
            "n": int(len(array)),
            "mean": float(array.mean()),
            "ci95_low": float(np.quantile(array, 0.025)),
            "ci95_high": float(np.quantile(array, 0.975)),
        }

    return {
        "method": (
            "paired held-out-group bootstrap over fixed OOF predictions; "
            "delta_macro_group_mae gives every sampled group equal weight"
        ),
        "replicates_requested": int(replicates),
        "comparisons": {
            name: {metric: interval(values) for metric, values in metrics.items()}
            for name, metrics in draws.items()
        },
    }


def run_ablation_benchmark(
    pair_data: PairDataset,
    registry: FeatureRegistry | None = None,
    *,
    group_column: str = "extractant",
    outer_folds: int = 5,
    inner_folds: int = 3,
    n_estimators: int = 200,
    n_jobs: int = -1,
    seed: int = 42,
    split_seed: int = 42,
    shuffle_seeds: Iterable[int] = (),
    include_block_ablations: bool = False,
    n_bootstrap: int = 1000,
    parameter_grid: tuple[dict[str, Any], ...] = DEFAULT_PARAMETER_GRID,
) -> AblationBenchmarkResult:
    """Evaluate A0--A6 and optional controls on identical nested group folds."""

    registry = registry or build_feature_registry(pair_data)
    frame = pair_data.frame.reset_index(drop=True).copy()
    if group_column not in frame.columns:
        raise ValueError(f"Unknown group column: {group_column}")
    if outer_folds < 2 or inner_folds < 2:
        raise ValueError("outer_folds and inner_folds must both be at least two.")
    if n_estimators < 1 or not parameter_grid:
        raise ValueError("n_estimators and parameter_grid must be non-empty/positive.")
    if registry.pair_scope not in {None, "all", "adjacent"}:
        raise ValueError(f"Unknown registry pair scope: {registry.pair_scope!r}")

    groups = frame[group_column].astype(str).to_numpy()
    strata = frame["pair_label"].astype(str).to_numpy()
    target = frame[PAIR_TARGET_COLUMN].to_numpy(dtype=float)
    outer_split_list = _group_folds(
        groups,
        strata,
        outer_folds,
        split_seed=int(split_seed),
        allow_fewer_splits=False,
    )

    feature_sets = registry.feature_sets()
    unavailable_blocks: list[str] = []
    block_aliases: dict[str, str] = {}
    if include_block_ablations:
        for name, columns in registry.local_3d_block_feature_sets().items():
            if name == "A2+D1-D5":
                block_aliases[name] = "A5"
            elif len(columns) == len(registry.ablation_columns("A2")):
                unavailable_blocks.append(name)
            else:
                feature_sets[name] = columns

    shuffle_seed_values = tuple(int(value) for value in shuffle_seeds)
    if len(set(shuffle_seed_values)) != len(shuffle_seed_values):
        raise ValueError("shuffle_seeds must be unique.")
    if any(value < 0 for value in shuffle_seed_values):
        raise ValueError("shuffle_seeds must be nonnegative.")
    shuffle_columns = registry.columns_for_family("3D_LOCAL")
    for shuffle_seed in shuffle_seed_values:
        if not shuffle_columns:
            raise ValueError("A5 shuffled controls require 3D_LOCAL features.")
        feature_sets[f"A5_SHUFFLED_s{shuffle_seed}"] = registry.ablation_columns("A5")

    outer_lookup = np.full(len(frame), -1, dtype=int)
    membership_rows: list[dict[str, Any]] = []
    inner_assignment_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    inner_split_by_outer: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    all_pair_types = set(frame["pair_label"].astype(str))

    for outer_fold, (train_index, test_index) in enumerate(outer_split_list):
        outer_lookup[test_index] = outer_fold
        train_frame = frame.iloc[train_index]
        test_frame = frame.iloc[test_index]
        train_groups = set(groups[train_index])
        test_groups = set(groups[test_index])
        leakage = {
            "outer_fold": int(outer_fold),
            "train_rows": int(len(train_index)),
            "test_rows": int(len(test_index)),
            "train_groups": int(len(train_groups)),
            "test_groups": int(len(test_groups)),
            "held_out_groups": sorted(test_groups),
            "held_out_extractants": sorted(set(test_frame["extractant"].astype(str))),
            "group_overlap": int(len(train_groups & test_groups)),
            "extractant_overlap": _identity_overlap(train_frame, test_frame, ("extractant",)),
            "ecfp_exact_cluster_overlap": _identity_overlap(
                train_frame, test_frame, ("ecfp_exact_cluster",)
            ),
            "source_id_overlap": _identity_overlap(
                train_frame, test_frame, ("source_id_A", "source_id_B")
            ),
            "geometry_key_overlap": _identity_overlap(
                train_frame, test_frame, ("geometry_key_A", "geometry_key_B")
            ),
            "geometry_feature_build_id_overlap": _identity_overlap(
                train_frame,
                test_frame,
                ("geometry_feature_build_id_A", "geometry_feature_build_id_B"),
            ),
            "vr_graph_index_overlap": _identity_overlap(
                train_frame, test_frame, ("vr_graph_index_A", "vr_graph_index_B")
            ),
            "train_pair_types": int(train_frame["pair_label"].nunique()),
            "test_pair_types": int(test_frame["pair_label"].nunique()),
            "pair_types_missing_from_train": sorted(
                all_pair_types - set(train_frame["pair_label"].astype(str))
            ),
            "pair_types_missing_from_test": sorted(
                all_pair_types - set(test_frame["pair_label"].astype(str))
            ),
        }
        required_zero_overlap = [
            "group_overlap",
            "extractant_overlap",
            "source_id_overlap",
            "geometry_key_overlap",
            "geometry_feature_build_id_overlap",
            "vr_graph_index_overlap",
        ]
        # Exact-ECFP grouping is the stricter sensitivity protocol.  Under the
        # primary literal-extractant holdout, an ECFP collision/shared cluster is
        # recorded but is not itself an extractant or complex-ID leak.
        if group_column == "ecfp_exact_cluster":
            required_zero_overlap.append("ecfp_exact_cluster_overlap")
        leakage["required_zero_overlap_fields"] = required_zero_overlap
        leakage["record_only_sensitivity_fields"] = (
            [] if group_column == "ecfp_exact_cluster" else ["ecfp_exact_cluster_overlap"]
        )
        if any(int(leakage[key]) for key in required_zero_overlap):
            raise AssertionError(f"Outer-fold leakage detected: {leakage}")
        leakage_rows.append(leakage)

        test_positions = set(test_index.tolist())
        for position, row in frame.iterrows():
            membership_rows.append(
                {
                    "pair_id": str(row["pair_id"]),
                    "group_id": str(row[group_column]),
                    "outer_fold": int(outer_fold),
                    "assignment": "test" if position in test_positions else "train",
                    "outer_split_seed": int(split_seed),
                }
            )

        inner_seed = int(split_seed + outer_fold * 10_007)
        inner_splits = _group_folds(
            groups[train_index],
            strata[train_index],
            inner_folds,
            split_seed=inner_seed,
            allow_fewer_splits=True,
        )
        inner_split_by_outer[outer_fold] = inner_splits
        inner_validation_lookup = np.full(len(train_index), -1, dtype=int)
        for inner_fold, (_, validation_local) in enumerate(inner_splits):
            inner_validation_lookup[validation_local] = inner_fold
        if np.any(inner_validation_lookup < 0):
            raise AssertionError("Inner fold plan did not validate every outer-training row.")
        for local_position, global_position in enumerate(train_index):
            inner_assignment_rows.append(
                {
                    "pair_id": str(frame.iloc[global_position]["pair_id"]),
                    "outer_fold": int(outer_fold),
                    "inner_validation_fold": int(inner_validation_lookup[local_position]),
                    "inner_split_seed": inner_seed,
                    "group_id": str(groups[global_position]),
                }
            )

    if np.any(outer_lookup < 0):
        raise AssertionError("Outer fold plan did not test every row exactly once.")

    metadata_columns = [
        column
        for column in (
            "pair_id",
            "condition_id",
            "extractant",
            "ecfp_exact_cluster",
            "pair_label",
            "metal_A",
            "metal_B",
            "source_id_A",
            "source_id_B",
            "geometry_key_A",
            "geometry_key_B",
            "geometry_qc_class_A",
            "geometry_qc_class_B",
            PAIR_TARGET_COLUMN,
        )
        if column in frame.columns
    ]
    predictions = frame.loc[:, metadata_columns].copy()
    predictions["outer_fold"] = outer_lookup
    predictions["geometry_qc_pair"] = _pair_qc_labels(frame)
    for ablation in feature_sets:
        predictions[f"prediction_{ablation}"] = np.nan

    tuning_rows: list[dict[str, Any]] = []
    preprocessing_rows: list[dict[str, Any]] = []
    shuffle_rows: list[dict[str, Any]] = []

    for outer_fold, (train_index, test_index) in enumerate(outer_split_list):
        outer_train = frame.iloc[train_index]
        outer_test = frame.iloc[test_index]
        outer_target = target[train_index]
        outer_groups = groups[train_index]
        inner_splits = inner_split_by_outer[outer_fold]
        for ablation, columns in feature_sets.items():
            if not columns:
                raise ValueError(f"Ablation {ablation} has no features.")
            shuffle_seed = (
                int(ablation.rsplit("s", 1)[1])
                if ablation.startswith("A5_SHUFFLED_s")
                else None
            )
            candidate_predictions: dict[int, np.ndarray] = {}
            candidate_rows: list[dict[str, Any]] = []
            for candidate_index, parameters in enumerate(parameter_grid):
                inner_prediction = np.full(len(train_index), np.nan, dtype=float)
                for inner_fold, (inner_train_local, validation_local) in enumerate(
                    inner_splits
                ):
                    fit_frame = outer_train.iloc[inner_train_local]
                    if shuffle_seed is not None:
                        applied_seed = (
                            shuffle_seed + outer_fold * 1_000_003 + inner_fold * 10_009
                        )
                        fit_frame, shuffle_audit = _training_shuffle(
                            fit_frame,
                            shuffle_columns,
                            seed=applied_seed,
                        )
                        if candidate_index == 0:
                            shuffle_rows.append(
                                {
                                    "ablation": ablation,
                                    "outer_fold": outer_fold,
                                    "inner_fold": inner_fold,
                                    "scope": "inner_training_only",
                                    "shuffle_seed": applied_seed,
                                    "test_rows_touched": 0,
                                    **shuffle_audit,
                                }
                            )
                    model = AntisymmetricExtraTreesRegressor(
                        columns,
                        n_estimators=n_estimators,
                        max_features=float(parameters["max_features"]),
                        min_samples_leaf=int(parameters["min_samples_leaf"]),
                        random_state=seed + outer_fold * 1009 + candidate_index * 101 + inner_fold,
                        n_jobs=n_jobs,
                    )
                    model.fit(
                        fit_frame,
                        outer_target[inner_train_local],
                        outer_groups[inner_train_local],
                    )
                    inner_prediction[validation_local] = model.predict(
                        outer_train.iloc[validation_local]
                    )
                if not np.isfinite(inner_prediction).all():
                    raise AssertionError("Inner OOF prediction is incomplete/non-finite.")
                metrics = _finite_metrics(outer_target, inner_prediction)
                macro_mae = _macro_group_mae(
                    outer_target, inner_prediction, outer_groups
                )
                row = {
                    "ablation": ablation,
                    "outer_fold": int(outer_fold),
                    "candidate_index": int(candidate_index),
                    "inner_fold_count": int(len(inner_splits)),
                    "inner_split_seed": int(split_seed + outer_fold * 10_007),
                    "max_features": float(parameters["max_features"]),
                    "min_samples_leaf": int(parameters["min_samples_leaf"]),
                    "inner_macro_group_mae": macro_mae,
                    **{f"inner_{key}": value for key, value in metrics.items()},
                }
                candidate_rows.append(row)
                candidate_predictions[candidate_index] = inner_prediction
            best = min(
                candidate_rows,
                key=lambda row: (
                    float(row["inner_macro_group_mae"]),
                    float(row["inner_mae"]),
                    -float(row["inner_r2"]),
                    int(row["candidate_index"]),
                ),
            )
            for row in candidate_rows:
                row["selected"] = bool(
                    int(row["candidate_index"]) == int(best["candidate_index"])
                )
                tuning_rows.append(row)

            final_fit_frame = outer_train
            if shuffle_seed is not None:
                applied_seed = shuffle_seed + outer_fold * 1_000_003 + 999_983
                final_fit_frame, shuffle_audit = _training_shuffle(
                    final_fit_frame,
                    shuffle_columns,
                    seed=applied_seed,
                )
                shuffle_rows.append(
                    {
                        "ablation": ablation,
                        "outer_fold": outer_fold,
                        "inner_fold": None,
                        "scope": "outer_training_only",
                        "shuffle_seed": applied_seed,
                        "test_rows_touched": 0,
                        **shuffle_audit,
                    }
                )
            final_model = AntisymmetricExtraTreesRegressor(
                columns,
                n_estimators=n_estimators,
                max_features=float(best["max_features"]),
                min_samples_leaf=int(best["min_samples_leaf"]),
                random_state=seed + outer_fold * 1009 + 9_999_991,
                n_jobs=n_jobs,
            )
            final_model.fit(final_fit_frame, outer_target, outer_groups)
            predictions.loc[test_index, f"prediction_{ablation}"] = final_model.predict(
                outer_test
            )
            pipeline = final_model.pipeline
            if pipeline is None:
                raise AssertionError("Fitted model did not expose its preprocessing pipeline.")
            imputer = pipeline.named_steps["imputer"]
            statistics = np.asarray(imputer.statistics_, dtype=float)
            statistics_payload = json.dumps(
                [None if not np.isfinite(value) else float(value) for value in statistics],
                separators=(",", ":"),
            )
            preprocessing_rows.append(
                {
                    "ablation": ablation,
                    "outer_fold": int(outer_fold),
                    "feature_count": int(len(columns)),
                    "features": json.dumps(list(columns), separators=(",", ":")),
                    "imputer": "SimpleImputer(strategy=median, add_indicator=True)",
                    "fit_rows": int(len(final_fit_frame)),
                    "outer_test_rows_seen_during_fit": 0,
                    "missing_training_values": int(
                        final_fit_frame.loc[:, list(columns)].isna().sum().sum()
                    ),
                    "imputer_statistics": statistics_payload,
                    "imputer_statistics_sha256": hashlib.sha256(
                        statistics_payload.encode("utf-8")
                    ).hexdigest(),
                }
            )

    prediction_columns = [f"prediction_{name}" for name in feature_sets]
    if predictions[prediction_columns].isna().any().any():
        raise AssertionError("Outer OOF predictions are incomplete.")
    if not np.isfinite(predictions[prediction_columns].to_numpy(dtype=float)).all():
        raise AssertionError("Outer OOF predictions contain non-finite values.")

    fold_metric_rows: list[dict[str, Any]] = []
    for outer_fold in range(len(outer_split_list)):
        fold_frame = predictions[predictions["outer_fold"].eq(outer_fold)]
        fold_truth = fold_frame[PAIR_TARGET_COLUMN].to_numpy(dtype=float)
        for ablation in feature_sets:
            values = _finite_metrics(
                fold_truth,
                fold_frame[f"prediction_{ablation}"].to_numpy(dtype=float),
            )
            fold_metric_rows.append(
                {"outer_fold": outer_fold, "ablation": ablation, **values}
            )
    fold_metrics = pd.DataFrame(fold_metric_rows)

    per_ablation_rows: list[dict[str, Any]] = []
    for ablation in feature_sets:
        oof_metrics = _finite_metrics(
            target, predictions[f"prediction_{ablation}"].to_numpy(dtype=float)
        )
        ablation_folds = fold_metrics[fold_metrics["ablation"].eq(ablation)]
        row: dict[str, Any] = {"ablation": ablation, **oof_metrics}
        for metric in METRIC_NAMES:
            values = ablation_folds[metric].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            row[f"fold_{metric}_mean"] = float(finite.mean()) if len(finite) else np.nan
            row[f"fold_{metric}_sample_sd"] = (
                float(finite.std(ddof=1)) if len(finite) >= 2 else np.nan
            )
        row["macro_group_mae"] = _macro_group_mae(
            target,
            predictions[f"prediction_{ablation}"].to_numpy(dtype=float),
            groups,
        )
        per_ablation_rows.append(row)
    per_ablation_metrics = pd.DataFrame(per_ablation_rows)

    comparisons = list(PRIMARY_COMPARISONS)
    comparisons.extend(
        (f"A2_vs_{name}", "A2", name)
        for name in feature_sets
        if name.startswith("A5_SHUFFLED_s")
    )
    comparisons.extend(
        (f"A2_vs_{name}", "A2", name)
        for name in feature_sets
        if name.startswith("A2+D")
    )
    delta_rows: list[dict[str, Any]] = []
    for name, reference, candidate in comparisons:
        for outer_fold in range(len(outer_split_list)):
            fold_frame = predictions[predictions["outer_fold"].eq(outer_fold)]
            delta_rows.append(
                {
                    "comparison": name,
                    "reference": reference,
                    "candidate": candidate,
                    "outer_fold": int(outer_fold),
                    **_delta_metrics(
                        fold_frame[PAIR_TARGET_COLUMN].to_numpy(dtype=float),
                        fold_frame[f"prediction_{reference}"].to_numpy(dtype=float),
                        fold_frame[f"prediction_{candidate}"].to_numpy(dtype=float),
                    ),
                }
            )
    paired_deltas = pd.DataFrame(delta_rows)

    per_extractant_rows: list[dict[str, Any]] = []
    for extractant, group_frame in predictions.groupby("extractant", sort=False):
        truth = group_frame[PAIR_TARGET_COLUMN].to_numpy(dtype=float)
        for ablation in feature_sets:
            per_extractant_rows.append(
                {
                    "extractant_id": str(extractant),
                    "ablation": ablation,
                    **_finite_metrics(
                        truth,
                        group_frame[f"prediction_{ablation}"].to_numpy(dtype=float),
                    ),
                }
            )
    per_extractant_metrics = pd.DataFrame(per_extractant_rows)

    per_lanthanide_rows: list[dict[str, Any]] = []
    metals = sorted(
        set(predictions["metal_A"].astype(str))
        | set(predictions["metal_B"].astype(str))
    )
    for metal in metals:
        side_a = predictions[predictions["metal_A"].astype(str).eq(metal)].copy()
        side_b = predictions[predictions["metal_B"].astype(str).eq(metal)].copy()
        oriented_truth = np.concatenate(
            [
                side_a[PAIR_TARGET_COLUMN].to_numpy(dtype=float),
                -side_b[PAIR_TARGET_COLUMN].to_numpy(dtype=float),
            ]
        )
        for ablation in feature_sets:
            oriented_prediction = np.concatenate(
                [
                    side_a[f"prediction_{ablation}"].to_numpy(dtype=float),
                    -side_b[f"prediction_{ablation}"].to_numpy(dtype=float),
                ]
            )
            per_lanthanide_rows.append(
                {
                    "lanthanide": metal,
                    "ablation": ablation,
                    "n_pair_memberships": int(len(oriented_truth)),
                    **_finite_metrics(oriented_truth, oriented_prediction),
                }
            )
    per_lanthanide_metrics = pd.DataFrame(per_lanthanide_rows)

    qc_rows: list[dict[str, Any]] = []
    for qc_class, qc_frame in predictions.groupby("geometry_qc_pair", sort=False):
        if "prediction_A2" not in qc_frame or "prediction_A5" not in qc_frame:
            continue
        truth = qc_frame[PAIR_TARGET_COLUMN].to_numpy(dtype=float)
        metrics_a2 = _finite_metrics(truth, qc_frame["prediction_A2"].to_numpy(dtype=float))
        metrics_a5 = _finite_metrics(truth, qc_frame["prediction_A5"].to_numpy(dtype=float))
        qc_rows.append(
            {
                "geometry_qc_pair": str(qc_class),
                "n_rows": int(len(qc_frame)),
                "A2_mae": metrics_a2["mae"],
                "A5_mae": metrics_a5["mae"],
                "delta_mae_A2_minus_A5": float(metrics_a2["mae"] - metrics_a5["mae"]),
            }
        )
    observed_qc = {str(row["geometry_qc_pair"]) for row in qc_rows}
    geometry_qc_summary = {
        "failed_geometries_in_model_cohort": int(
            predictions["geometry_qc_pair"].eq("failed_geometry").sum()
        ),
        "pair_classes": qc_rows,
        "high_vs_low_estimable": bool(
            "high_confidence" in observed_qc
            and "questionable_or_recovered" in observed_qc
        ),
        "interpretation": (
            "High-vs-low geometry comparison is unavailable when the common accepted "
            "cohort contains only one QC class. Failed geometries are never admitted."
        ),
        "source_pair_build_exclusions_missing_geometry": int(
            pair_data.audit.get("pairs_excluded_missing_geometry", 0)
        ),
        "source_geometry_ok_rows": int(
            pair_data.audit.get("source_geometry_ok_rows", 0)
        ),
        "source_geometry_unavailable_or_rejected_rows": int(
            pair_data.audit.get("source_geometry_unavailable_or_rejected_rows", 0)
        ),
        "source_geometry_qc_class_counts": pair_data.audit.get(
            "source_geometry_qc_class_counts", {}
        ),
    }

    delta_summary: dict[str, Any] = {}
    for comparison, comparison_frame in paired_deltas.groupby("comparison", sort=False):
        metrics: dict[str, Any] = {}
        for metric in ("delta_mae", "delta_rmse", "delta_r2", "delta_spearman"):
            values = comparison_frame[metric].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            metrics[metric] = {
                "mean": float(finite.mean()) if len(finite) else None,
                "median": float(np.median(finite)) if len(finite) else None,
                "sample_sd": float(finite.std(ddof=1)) if len(finite) >= 2 else None,
                "positive_folds": int(np.sum(finite > 0.0)),
                "fold_count": int(len(finite)),
                "positive_fraction": float(np.mean(finite > 0.0)) if len(finite) else None,
            }
        delta_summary[str(comparison)] = metrics

    bootstrap = _bootstrap_comparisons(
        predictions,
        comparisons,
        group_column=group_column,
        replicates=n_bootstrap,
        seed=seed + 999_983,
    )
    registry_payload = registry.to_dict()
    feature_audit = {
        "passed": True,
        "registry_sha256": registry.feature_contract_sha256,
        "pair_scope": registry.pair_scope,
        "excluded_non_geometric_columns": list(
            registry.excluded_non_geometric_columns
        ),
        "main_ablations": {
            name: {"column_count": len(columns), "columns": list(columns)}
            for name, columns in registry.feature_sets().items()
        },
        "evaluated_feature_sets": {
            name: {"column_count": len(columns), "columns": list(columns)}
            for name, columns in feature_sets.items()
        },
        "unavailable_empty_block_ablations": unavailable_blocks,
        "block_aliases": block_aliases,
        "all_registry_features_assigned_once": bool(
            registry_payload["audit"]["all_model_features_assigned_exactly_once"]
        ),
        "all_source_columns_accounted_for": bool(
            registry_payload["audit"]["all_source_columns_accounted_for"]
        ),
        "target_or_identifier_features_present": False,
    }
    leakage_audit = {
        "passed": True,
        "pair_scope": registry.pair_scope,
        "group_column": group_column,
        "outer_split_seed": int(split_seed),
        "same_outer_folds_for_every_ablation": True,
        "same_inner_folds_for_every_ablation_within_outer_fold": True,
        "preprocessing_fit_inside_training_fold": True,
        "outer_test_labels_used_for_selection": False,
        "outer_folds": leakage_rows,
    }
    summary = {
        "protocol": {
            "pair_scope": registry.pair_scope,
            "group_column": group_column,
            "outer_folds": int(len(outer_split_list)),
            "inner_folds_requested": int(inner_folds),
            "outer_split_seed": int(split_seed),
            "model_seed": int(seed),
            "model_family": "antisymmetric ExtraTrees with fold-local median imputation",
            "model_selection_metric": "inner macro held-out-group MAE",
            "primary_metric": "MAE",
            "primary_inference_statistic": "equal-extractant macro delta MAE",
            "aggregate_r2": "computed once from concatenated cross-fitted OOF predictions",
            "fold_plan_reused_across_ablations": True,
            "inner_plan_reused_across_ablations": True,
            "shuffle_scope": "only the current inner/outer training subset; test untouched",
        },
        "feature_counts": {name: len(columns) for name, columns in feature_sets.items()},
        "metrics": per_ablation_metrics.to_dict(orient="records"),
        "paired_delta_summary": delta_summary,
        "paired_group_bootstrap": bootstrap,
        "leakage_audit": leakage_audit,
        "geometry_qc_summary": geometry_qc_summary,
        "feature_audit": feature_audit,
        "training_history": "not applicable to ExtraTrees",
    }
    fold_assignments = predictions[
        ["pair_id", "extractant", "ecfp_exact_cluster", "outer_fold"]
    ].copy()
    fold_assignments["group_id"] = frame[group_column].astype(str)
    fold_assignments["pair_scope"] = registry.pair_scope
    fold_assignments["outer_split_seed"] = int(split_seed)

    return AblationBenchmarkResult(
        summary=summary,
        predictions=predictions,
        fold_metrics=fold_metrics,
        fold_assignments=fold_assignments,
        fold_memberships=pd.DataFrame(membership_rows),
        inner_fold_assignments=pd.DataFrame(inner_assignment_rows),
        tuning_results=pd.DataFrame(tuning_rows),
        preprocessing_audit=pd.DataFrame(preprocessing_rows),
        shuffle_audit=pd.DataFrame(shuffle_rows),
        per_ablation_metrics=per_ablation_metrics,
        paired_deltas=paired_deltas,
        per_extractant_metrics=per_extractant_metrics,
        per_lanthanide_metrics=per_lanthanide_metrics,
        leakage_audit=leakage_audit,
        geometry_qc_summary=geometry_qc_summary,
        feature_audit=feature_audit,
    )
