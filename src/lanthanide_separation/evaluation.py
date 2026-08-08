"""Nested group-held-out evaluation for paired 2D and Delta3D models."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from .pairs import PAIR_TARGET_COLUMN, PairDataset, reverse_pair_features


DEFAULT_PARAMETER_GRID: tuple[dict[str, Any], ...] = (
    {"max_features": 0.35, "min_samples_leaf": 2},
    {"max_features": 0.70, "min_samples_leaf": 2},
    {"max_features": 1.00, "min_samples_leaf": 4},
)
DEFAULT_DELTA3D_WEIGHTS: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0)

BASE_MODEL_NAMES: tuple[str, ...] = (
    "baseline",
    "2d_second_unshrunk",
    "2d_ensemble",
    "delta3d_unshrunk",
    "delta3d",
)
EXTENDED_MODEL_NAMES: tuple[str, ...] = (
    "delta3d_extended_unshrunk",
    "delta3d_extended",
)


@dataclass
class BenchmarkResult:
    summary: dict[str, Any]
    predictions: pd.DataFrame
    fold_assignments: pd.DataFrame
    tuning_results: pd.DataFrame
    per_extractant_metrics: pd.DataFrame
    per_pair_metrics: pd.DataFrame
    final_model: "ShrinkageDelta3DRegressor | None"
    feature_importances: pd.DataFrame


def _as_float_frame(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    return frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").astype(float)


def group_balanced_weights(groups: Iterable[Any]) -> np.ndarray:
    """Give every held-in group equal total training weight."""

    group_array = np.asarray(list(groups), dtype=object)
    counts = Counter(group_array.tolist())
    n_groups = len(counts)
    if n_groups == 0:
        raise ValueError("Cannot calculate weights for zero groups.")
    n_rows = len(group_array)
    return np.asarray(
        [n_rows / (n_groups * counts[group]) for group in group_array], dtype=float
    )


class AntisymmetricExtraTreesRegressor:
    """ExtraTrees pair model with exact A/B antisymmetry at inference.

    Every training pair is augmented with its reversed representation and a
    negated target.  Prediction is ``(f(A,B) - f(B,A)) / 2``, so swapping the
    metals changes the prediction sign to numerical precision.
    """

    def __init__(
        self,
        feature_columns: Iterable[str],
        *,
        n_estimators: int = 200,
        max_features: float = 0.70,
        min_samples_leaf: int = 2,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self.feature_columns = tuple(feature_columns)
        self.n_estimators = int(n_estimators)
        self.max_features = float(max_features)
        self.min_samples_leaf = int(min_samples_leaf)
        self.random_state = int(random_state)
        self.n_jobs = int(n_jobs)
        self.pipeline: Pipeline | None = None

    def _new_pipeline(self) -> Pipeline:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=self.n_estimators,
                        max_features=self.max_features,
                        min_samples_leaf=self.min_samples_leaf,
                        random_state=self.random_state,
                        n_jobs=self.n_jobs,
                    ),
                ),
            ]
        )

    def fit(
        self,
        frame: pd.DataFrame,
        target: Iterable[float],
        groups: Iterable[Any],
    ) -> "AntisymmetricExtraTreesRegressor":
        target_array = np.asarray(list(target), dtype=float)
        group_array = np.asarray(list(groups), dtype=object)
        if len(frame) != len(target_array) or len(frame) != len(group_array):
            raise ValueError("frame, target and groups must have equal length.")

        forward = _as_float_frame(frame, self.feature_columns)
        reverse = _as_float_frame(reverse_pair_features(frame), self.feature_columns)
        augmented_x = pd.concat([forward, reverse], ignore_index=True)
        augmented_y = np.concatenate([target_array, -target_array])
        original_weights = group_balanced_weights(group_array)
        augmented_weights = np.concatenate([original_weights, original_weights])

        self.pipeline = self._new_pipeline()
        self.pipeline.fit(augmented_x, augmented_y, model__sample_weight=augmented_weights)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Model has not been fitted.")
        forward = _as_float_frame(frame, self.feature_columns)
        reverse = _as_float_frame(reverse_pair_features(frame), self.feature_columns)
        pred_forward = self.pipeline.predict(forward)
        pred_reverse = self.pipeline.predict(reverse)
        return (pred_forward - pred_reverse) / 2.0

    def feature_importance_frame(self) -> pd.DataFrame:
        if self.pipeline is None:
            raise RuntimeError("Model has not been fitted.")
        imputer = self.pipeline.named_steps["imputer"]
        model = self.pipeline.named_steps["model"]
        names = imputer.get_feature_names_out(self.feature_columns)
        result = pd.DataFrame(
            {"feature": names, "importance": model.feature_importances_.astype(float)}
        )
        return result.sort_values("importance", ascending=False, ignore_index=True)


@dataclass
class ShrinkageDelta3DRegressor:
    """A leakage-tuned blend of baseline and full Delta3D pair models."""

    baseline_model: AntisymmetricExtraTreesRegressor
    full_model: AntisymmetricExtraTreesRegressor
    delta3d_weight: float

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        baseline = self.baseline_model.predict(frame)
        full = self.full_model.predict(frame)
        return baseline + float(self.delta3d_weight) * (full - baseline)

    def feature_importance_frame(self) -> pd.DataFrame:
        importance = self.full_model.feature_importance_frame().copy()
        importance.insert(1, "deployment_delta3d_blend_weight", float(self.delta3d_weight))
        importance.insert(2, "component", "full_delta3d_forest_only")
        importance.insert(
            3,
            "interpretation",
            "raw impurity importance; not an attribution of the blended predictor",
        )
        return importance


def regression_metrics(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    groups: Iterable[Any],
) -> dict[str, float | int]:
    truth = np.asarray(list(y_true), dtype=float)
    prediction = np.asarray(list(y_pred), dtype=float)
    group_array = np.asarray(list(groups), dtype=object)
    if not (len(truth) == len(prediction) == len(group_array)):
        raise ValueError("Metric inputs must have equal length.")

    errors = np.abs(truth - prediction)
    per_group = pd.DataFrame({"group": group_array, "abs_error": errors}).groupby(
        "group", dropna=False
    )["abs_error"].mean()
    truth_rank = pd.Series(truth).rank(method="average").to_numpy()
    prediction_rank = pd.Series(prediction).rank(method="average").to_numpy()
    if np.std(truth_rank) > 0 and np.std(prediction_rank) > 0:
        spearman = float(np.corrcoef(truth_rank, prediction_rank)[0, 1])
    else:
        spearman = float("nan")

    return {
        "n_rows": int(len(truth)),
        "n_groups": int(pd.Series(group_array).nunique(dropna=False)),
        "r2": float(r2_score(truth, prediction)),
        "group_balanced_r2": float(
            r2_score(
                truth,
                prediction,
                sample_weight=group_balanced_weights(group_array),
            )
        ),
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
        "macro_group_mae": float(per_group.mean()),
        "worst_group_mae": float(per_group.max()),
        "spearman": spearman,
        "sign_accuracy": float(np.mean(np.sign(truth) == np.sign(prediction))),
        "magnitude_mae": float(mean_absolute_error(np.abs(truth), np.abs(prediction))),
    }


def _group_folds(
    groups: np.ndarray,
    strata: np.ndarray,
    n_splits: int,
    *,
    split_seed: int,
    allow_fewer_splits: bool,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build target-independent, seeded, pair-type-balanced group folds."""

    if len(groups) != len(strata):
        raise ValueError("groups and strata must have equal length.")
    unique_groups = pd.Series(groups).nunique(dropna=False)
    actual_splits = min(int(n_splits), int(unique_groups))
    if actual_splits < 2:
        raise ValueError("At least two independent groups are required for group CV.")
    if not allow_fewer_splits and actual_splits != int(n_splits):
        raise ValueError(
            f"Requested {n_splits} folds but only {unique_groups} independent groups exist."
        )
    splitter = StratifiedGroupKFold(
        n_splits=actual_splits,
        shuffle=True,
        random_state=int(split_seed),
    )
    dummy = np.zeros((len(groups), 1), dtype=np.float32)
    return list(splitter.split(dummy, y=strata, groups=groups))


def _tune_parameters(
    frame: pd.DataFrame,
    target: np.ndarray,
    groups: np.ndarray,
    feature_columns: tuple[str, ...],
    *,
    parameter_grid: tuple[dict[str, Any], ...],
    inner_folds: int,
    n_estimators: int,
    n_jobs: int,
    model_seed: int,
    split_seed: int,
    context: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    strata = frame["pair_label"].astype(str).to_numpy()
    folds = _group_folds(
        groups,
        strata,
        inner_folds,
        split_seed=split_seed,
        allow_fewer_splits=True,
    )
    candidate_rows: list[dict[str, Any]] = []
    candidate_predictions: dict[int, np.ndarray] = {}

    for candidate_index, parameters in enumerate(parameter_grid):
        inner_prediction = np.full(len(frame), np.nan, dtype=float)
        for fold_index, (train_index, validation_index) in enumerate(folds):
            train_groups = set(groups[train_index].tolist())
            validation_groups = set(groups[validation_index].tolist())
            overlap = train_groups & validation_groups
            if overlap:
                raise AssertionError(f"Inner group leakage detected: {sorted(overlap)[:5]}")

            model = AntisymmetricExtraTreesRegressor(
                feature_columns,
                n_estimators=n_estimators,
                max_features=float(parameters["max_features"]),
                min_samples_leaf=int(parameters["min_samples_leaf"]),
                random_state=model_seed + candidate_index * 101 + fold_index,
                n_jobs=n_jobs,
            )
            model.fit(
                frame.iloc[train_index],
                target[train_index],
                groups[train_index],
            )
            inner_prediction[validation_index] = model.predict(frame.iloc[validation_index])

        if np.isnan(inner_prediction).any():
            raise AssertionError("Inner CV did not produce exactly one prediction per row.")
        metrics = regression_metrics(target, inner_prediction, groups)
        candidate_predictions[candidate_index] = inner_prediction.copy()
        candidate_rows.append(
            {
                "context": context,
                "candidate_index": candidate_index,
                "inner_fold_count": len(folds),
                "inner_split_seed": int(split_seed),
                **parameters,
                "inner_macro_group_mae": metrics["macro_group_mae"],
                "inner_r2": metrics["r2"],
                "inner_mae": metrics["mae"],
            }
        )

    best = min(
        candidate_rows,
        key=lambda row: (
            float(row["inner_macro_group_mae"]),
            -float(row["inner_r2"]),
            int(row["candidate_index"]),
        ),
    )
    selected = {
        "max_features": float(best["max_features"]),
        "min_samples_leaf": int(best["min_samples_leaf"]),
    }
    for row in candidate_rows:
        row["selected"] = bool(int(row["candidate_index"]) == int(best["candidate_index"]))
    return selected, candidate_rows, candidate_predictions[int(best["candidate_index"])]


def _select_delta3d_weight(
    target: np.ndarray,
    groups: np.ndarray,
    baseline_prediction: np.ndarray,
    full_prediction: np.ndarray,
    *,
    weights: tuple[float, ...] = DEFAULT_DELTA3D_WEIGHTS,
) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for weight in weights:
        prediction = baseline_prediction + float(weight) * (
            full_prediction - baseline_prediction
        )
        metrics = regression_metrics(target, prediction, groups)
        rows.append(
            {
                "delta3d_weight": float(weight),
                "inner_macro_group_mae": metrics["macro_group_mae"],
                "inner_r2": metrics["r2"],
                "inner_mae": metrics["mae"],
            }
        )
    best = min(
        rows,
        key=lambda row: (
            float(row["inner_macro_group_mae"]),
            -float(row["inner_r2"]),
            float(row["delta3d_weight"]),
        ),
    )
    for row in rows:
        row["selected"] = bool(row["delta3d_weight"] == best["delta3d_weight"])
    return float(best["delta3d_weight"]), rows


def _cluster_bootstrap_comparisons(
    prediction_frame: pd.DataFrame,
    *,
    group_column: str,
    n_bootstrap: int,
    seed: int,
    comparisons: dict[str, tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Paired bootstrap comparisons using identical held-out group draws.

    This conditions on the fitted OOF models. It quantifies held-out group
    sampling uncertainty, not split, training, or model-selection variability.
    """

    rng = np.random.default_rng(seed)
    unique_groups = prediction_frame[group_column].drop_duplicates().to_numpy()
    group_indices = {
        group: prediction_frame.index[prediction_frame[group_column].eq(group)].to_numpy()
        for group in unique_groups
    }
    if comparisons is None:
        comparisons = {
            "delta3d_vs_2d_ensemble": ("prediction_2d_ensemble", "prediction_delta3d"),
            "delta3d_unshrunk_vs_2d_second_unshrunk": (
                "prediction_2d_second_unshrunk",
                "prediction_delta3d_unshrunk",
            ),
            "delta3d_vs_baseline": ("prediction_baseline", "prediction_delta3d"),
            "2d_ensemble_vs_baseline": ("prediction_baseline", "prediction_2d_ensemble"),
        }
    draws: dict[str, dict[str, list[float]]] = {
        name: {
            "r2_gain": [],
            "group_balanced_r2_gain": [],
            "mae_reduction": [],
            "macro_group_mae_reduction": [],
            "sign_accuracy_gain": [],
        }
        for name in comparisons
    }

    for _ in range(int(n_bootstrap)):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled_indices = np.concatenate([group_indices[group] for group in sampled_groups])
        sample = prediction_frame.loc[sampled_indices]
        truth = sample[PAIR_TARGET_COLUMN].to_numpy(dtype=float)
        group_balanced_sample_weight = np.concatenate(
            [
                np.full(len(group_indices[group]), 1.0 / len(group_indices[group]))
                for group in sampled_groups
            ]
        )
        if np.var(truth) <= 0:
            continue
        for name, (reference_column, candidate_column) in comparisons.items():
            reference = sample[reference_column].to_numpy(dtype=float)
            candidate = sample[candidate_column].to_numpy(dtype=float)
            draws[name]["r2_gain"].append(
                float(r2_score(truth, candidate) - r2_score(truth, reference))
            )
            draws[name]["group_balanced_r2_gain"].append(
                float(
                    r2_score(truth, candidate, sample_weight=group_balanced_sample_weight)
                    - r2_score(
                        truth,
                        reference,
                        sample_weight=group_balanced_sample_weight,
                    )
                )
            )
            draws[name]["mae_reduction"].append(
                float(
                    mean_absolute_error(truth, reference)
                    - mean_absolute_error(truth, candidate)
                )
            )
            draws[name]["sign_accuracy_gain"].append(
                float(
                    np.mean(np.sign(truth) == np.sign(candidate))
                    - np.mean(np.sign(truth) == np.sign(reference))
                )
            )

            # Preserve the multiplicity of resampled groups rather than letting
            # repeated labels collapse in a groupby.
            macro_differences = []
            for group in sampled_groups:
                group_frame = prediction_frame.loc[group_indices[group]]
                group_truth = group_frame[PAIR_TARGET_COLUMN].to_numpy(dtype=float)
                macro_differences.append(
                    mean_absolute_error(
                        group_truth, group_frame[reference_column].to_numpy(dtype=float)
                    )
                    - mean_absolute_error(
                        group_truth, group_frame[candidate_column].to_numpy(dtype=float)
                    )
                )
            draws[name]["macro_group_mae_reduction"].append(
                float(np.mean(macro_differences))
            )

    def interval(values: list[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=float)
        return {
            "mean": float(array.mean()),
            "ci95_low": float(np.quantile(array, 0.025)),
            "ci95_high": float(np.quantile(array, 0.975)),
        }

    return {
        "method": "paired cluster bootstrap over fixed OOF held-out-group predictions",
        "scope_limitation": (
            "Does not refit nested CV and therefore excludes split, training, and "
            "model-selection variability."
        ),
        "replicates": int(len(next(iter(draws.values()))["r2_gain"])),
        "comparisons": {
            name: {metric: interval(values) for metric, values in metrics.items()}
            for name, metrics in draws.items()
        },
    }


def _group_metric_table(
    predictions: pd.DataFrame,
    group_column: str,
    label: str,
    model_names: tuple[str, ...] = BASE_MODEL_NAMES,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group, group_frame in predictions.groupby(group_column, dropna=False):
        truth = group_frame[PAIR_TARGET_COLUMN].to_numpy(dtype=float)
        row: dict[str, Any] = {group_column: group, "n_rows": int(len(group_frame))}
        for model_name in model_names:
            pred = group_frame[f"prediction_{model_name}"].to_numpy(dtype=float)
            row[f"{model_name}_mae"] = float(mean_absolute_error(truth, pred))
            row[f"{model_name}_rmse"] = float(np.sqrt(mean_squared_error(truth, pred)))
            row[f"{model_name}_sign_accuracy"] = float(
                np.mean(np.sign(truth) == np.sign(pred))
            )
            row[f"{model_name}_r2"] = (
                float(r2_score(truth, pred)) if len(truth) >= 2 and np.var(truth) > 0 else np.nan
            )
        row["mae_reduction_delta3d_vs_2d_ensemble"] = (
            row["2d_ensemble_mae"] - row["delta3d_mae"]
        )
        if "delta3d_extended" in model_names:
            row["mae_reduction_extended_vs_delta3d"] = (
                row["delta3d_mae"] - row["delta3d_extended_mae"]
            )
        rows.append(row)
    result = pd.DataFrame(rows)
    result.insert(0, "grouping", label)
    return result.sort_values("n_rows", ascending=False, ignore_index=True)


def nested_group_benchmark(
    pair_data: PairDataset,
    *,
    group_column: str = "extractant",
    outer_folds: int = 5,
    inner_folds: int = 3,
    n_estimators: int = 200,
    n_jobs: int = -1,
    n_bootstrap: int = 1000,
    seed: int = 42,
    split_seed: int = 42,
    fit_final_model: bool = True,
    parameter_grid: tuple[dict[str, Any], ...] = DEFAULT_PARAMETER_GRID,
) -> BenchmarkResult:
    """Compare 2D/context and 2D/context + paired Delta3D on identical folds."""

    frame = pair_data.frame.reset_index(drop=True).copy()
    if group_column not in frame.columns:
        raise ValueError(f"Unknown group column: {group_column}")
    groups = frame[group_column].astype(str).to_numpy()
    strata = frame["pair_label"].astype(str).to_numpy()
    target = frame[PAIR_TARGET_COLUMN].to_numpy(dtype=float)
    folds = _group_folds(
        groups,
        strata,
        outer_folds,
        split_seed=split_seed,
        allow_fewer_splits=False,
    )

    # The metal-site descriptor arm is present only when the caller opted into a
    # descriptor block. Its branch is built exactly like the Delta3D branch --
    # same seed stream, same selection budget, same blend rule -- so the
    # extended-minus-Delta3D contrast isolates the added columns.
    has_extension = bool(pair_data.descriptor_columns)
    model_names = BASE_MODEL_NAMES + (EXTENDED_MODEL_NAMES if has_extension else ())

    feature_sets = {
        "baseline": pair_data.baseline_columns,
        "2d_second_unshrunk": pair_data.baseline_columns,
        "delta3d_unshrunk": pair_data.full_columns,
    }
    # The second 2D branch and the full 3D branch use the same seed streams and
    # model-selection budget. Their adaptive blends with the first baseline form
    # the matched ensemble control and the candidate Delta3D model.
    branch_seed_offsets = {
        "baseline": 0,
        "2d_second_unshrunk": 100_003,
        "delta3d_unshrunk": 100_003,
    }
    if has_extension:
        feature_sets["delta3d_extended_unshrunk"] = pair_data.extended_columns
        branch_seed_offsets["delta3d_extended_unshrunk"] = 100_003
    predictions = frame[
        [
            "pair_id",
            "condition_id",
            "extractant",
            "ecfp_exact_cluster",
            "pair_label",
            "metal_A",
            "metal_B",
            "n_replicates_A",
            "n_replicates_B",
            PAIR_TARGET_COLUMN,
        ]
    ].copy()
    predictions["outer_fold"] = -1
    for model_name in model_names:
        predictions[f"prediction_{model_name}"] = np.nan

    leakage_folds: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    selected_by_fold: dict[str, list[dict[str, Any]]] = {
        "baseline": [],
        "2d_second_unshrunk": [],
        "2d_ensemble_weight": [],
        "delta3d_unshrunk": [],
        "delta3d_weight": [],
    }
    if has_extension:
        selected_by_fold["delta3d_extended_unshrunk"] = []
        selected_by_fold["delta3d_extended_weight"] = []

    for outer_fold, (train_index, test_index) in enumerate(folds):
        train_groups = set(groups[train_index].tolist())
        test_groups = set(groups[test_index].tolist())
        overlap = train_groups & test_groups
        train_frame = frame.iloc[train_index]
        test_frame = frame.iloc[test_index]

        def identity_overlap(columns: tuple[str, ...]) -> int:
            train_values = set(
                train_frame.loc[:, list(columns)].astype(str).to_numpy().ravel().tolist()
            )
            test_values = set(
                test_frame.loc[:, list(columns)].astype(str).to_numpy().ravel().tolist()
            )
            return int(len(train_values & test_values))

        extractant_overlap = identity_overlap(("extractant",))
        ecfp_cluster_overlap = identity_overlap(("ecfp_exact_cluster",))
        source_id_overlap = identity_overlap(("source_id_A", "source_id_B"))
        geometry_key_overlap = identity_overlap(("geometry_key_A", "geometry_key_B"))
        leakage_folds.append(
            {
                "outer_fold": outer_fold,
                "outer_split_seed": int(split_seed),
                "train_rows": int(len(train_index)),
                "test_rows": int(len(test_index)),
                "train_groups": int(len(train_groups)),
                "test_groups": int(len(test_groups)),
                "group_overlap": int(len(overlap)),
                "extractant_overlap": extractant_overlap,
                "ecfp_exact_cluster_overlap": ecfp_cluster_overlap,
                "source_id_overlap": source_id_overlap,
                "geometry_key_overlap": geometry_key_overlap,
            }
        )
        if overlap:
            raise AssertionError(f"Outer group leakage detected: {sorted(overlap)[:5]}")
        if extractant_overlap or ecfp_cluster_overlap:
            raise AssertionError(
                "Outer extractant-identity leakage detected: "
                f"canonical SMILES={extractant_overlap}, exact ECFP={ecfp_cluster_overlap}"
            )
        if source_id_overlap or geometry_key_overlap:
            raise AssertionError(
                "Outer provenance leakage detected: "
                f"source IDs={source_id_overlap}, geometry keys={geometry_key_overlap}"
            )
        predictions.loc[test_index, "outer_fold"] = outer_fold

        selected_parameters: dict[str, dict[str, Any]] = {}
        selected_inner_predictions: dict[str, np.ndarray] = {}
        inner_split_seed = split_seed + outer_fold * 10_007
        for model_name, feature_columns in feature_sets.items():
            selected, candidate_rows, selected_inner = _tune_parameters(
                frame.iloc[train_index],
                target[train_index],
                groups[train_index],
                feature_columns,
                parameter_grid=parameter_grid,
                inner_folds=inner_folds,
                n_estimators=n_estimators,
                n_jobs=n_jobs,
                model_seed=seed + outer_fold * 1009 + branch_seed_offsets[model_name],
                split_seed=inner_split_seed,
                context=f"outer_{outer_fold}:{model_name}",
            )
            tuning_rows.extend(candidate_rows)
            selected_by_fold[model_name].append({"outer_fold": outer_fold, **selected})
            selected_parameters[model_name] = selected
            selected_inner_predictions[model_name] = selected_inner

        outer_train = frame.iloc[train_index]
        outer_target = target[train_index]
        outer_groups = groups[train_index]
        baseline_inner = selected_inner_predictions["baseline"]
        full_inner = selected_inner_predictions["delta3d_unshrunk"]
        second_2d_inner = selected_inner_predictions["2d_second_unshrunk"]
        ensemble_2d_weight, ensemble_2d_weight_rows = _select_delta3d_weight(
            outer_target,
            outer_groups,
            baseline_inner,
            second_2d_inner,
        )
        for row in ensemble_2d_weight_rows:
            tuning_rows.append(
                {"context": f"outer_{outer_fold}:2d_ensemble_weight", **row}
            )
        selected_by_fold["2d_ensemble_weight"].append(
            {"outer_fold": outer_fold, "2d_ensemble_weight": ensemble_2d_weight}
        )
        delta3d_weight, weight_rows = _select_delta3d_weight(
            outer_target,
            outer_groups,
            baseline_inner,
            full_inner,
        )
        for row in weight_rows:
            tuning_rows.append(
                {"context": f"outer_{outer_fold}:delta3d_weight", **row}
            )
        selected_by_fold["delta3d_weight"].append(
            {"outer_fold": outer_fold, "delta3d_weight": delta3d_weight}
        )
        if has_extension:
            extended_weight, extended_weight_rows = _select_delta3d_weight(
                outer_target,
                outer_groups,
                baseline_inner,
                selected_inner_predictions["delta3d_extended_unshrunk"],
            )
            for row in extended_weight_rows:
                tuning_rows.append(
                    {"context": f"outer_{outer_fold}:delta3d_extended_weight", **row}
                )
            selected_by_fold["delta3d_extended_weight"].append(
                {"outer_fold": outer_fold, "delta3d_extended_weight": extended_weight}
            )

        baseline_model = AntisymmetricExtraTreesRegressor(
            pair_data.baseline_columns,
            n_estimators=n_estimators,
            max_features=selected_parameters["baseline"]["max_features"],
            min_samples_leaf=selected_parameters["baseline"]["min_samples_leaf"],
            random_state=seed + outer_fold * 101 + 7,
            n_jobs=n_jobs,
        )
        second_2d_model = AntisymmetricExtraTreesRegressor(
            pair_data.baseline_columns,
            n_estimators=n_estimators,
            max_features=selected_parameters["2d_second_unshrunk"]["max_features"],
            min_samples_leaf=selected_parameters["2d_second_unshrunk"][
                "min_samples_leaf"
            ],
            random_state=seed + outer_fold * 101 + 19,
            n_jobs=n_jobs,
        )
        full_model = AntisymmetricExtraTreesRegressor(
            pair_data.full_columns,
            n_estimators=n_estimators,
            max_features=selected_parameters["delta3d_unshrunk"]["max_features"],
            min_samples_leaf=selected_parameters["delta3d_unshrunk"]["min_samples_leaf"],
            random_state=seed + outer_fold * 101 + 19,
            n_jobs=n_jobs,
        )
        baseline_model.fit(outer_train, outer_target, outer_groups)
        second_2d_model.fit(outer_train, outer_target, outer_groups)
        full_model.fit(outer_train, outer_target, outer_groups)
        baseline_test = baseline_model.predict(frame.iloc[test_index])
        second_2d_test = second_2d_model.predict(frame.iloc[test_index])
        full_test = full_model.predict(frame.iloc[test_index])
        ensemble_2d_test = baseline_test + ensemble_2d_weight * (
            second_2d_test - baseline_test
        )
        shrinkage_test = baseline_test + delta3d_weight * (full_test - baseline_test)
        predictions.loc[test_index, "prediction_baseline"] = baseline_test
        predictions.loc[test_index, "prediction_2d_second_unshrunk"] = second_2d_test
        predictions.loc[test_index, "prediction_2d_ensemble"] = ensemble_2d_test
        predictions.loc[test_index, "prediction_delta3d_unshrunk"] = full_test
        predictions.loc[test_index, "prediction_delta3d"] = shrinkage_test

        if has_extension:
            extended_model = AntisymmetricExtraTreesRegressor(
                pair_data.extended_columns,
                n_estimators=n_estimators,
                max_features=selected_parameters["delta3d_extended_unshrunk"][
                    "max_features"
                ],
                min_samples_leaf=selected_parameters["delta3d_extended_unshrunk"][
                    "min_samples_leaf"
                ],
                random_state=seed + outer_fold * 101 + 19,
                n_jobs=n_jobs,
            )
            extended_model.fit(outer_train, outer_target, outer_groups)
            extended_test = extended_model.predict(frame.iloc[test_index])
            predictions.loc[test_index, "prediction_delta3d_extended_unshrunk"] = (
                extended_test
            )
            predictions.loc[test_index, "prediction_delta3d_extended"] = (
                baseline_test + extended_weight * (extended_test - baseline_test)
            )

    prediction_columns = [f"prediction_{name}" for name in model_names]
    if predictions[prediction_columns].isna().any().any():
        raise AssertionError("Outer CV did not produce one prediction per row and model.")
    if (predictions["outer_fold"] < 0).any():
        raise AssertionError("Some rows have no outer-fold assignment.")

    metrics = {
        model_name: regression_metrics(
            target,
            predictions[f"prediction_{model_name}"].to_numpy(dtype=float),
            groups,
        )
        for model_name in model_names
    }

    def comparison(reference: str, candidate: str) -> dict[str, float]:
        return {
            "r2_gain": float(metrics[candidate]["r2"] - metrics[reference]["r2"]),
            "group_balanced_r2_gain": float(
                metrics[candidate]["group_balanced_r2"]
                - metrics[reference]["group_balanced_r2"]
            ),
            "mae_reduction": float(metrics[reference]["mae"] - metrics[candidate]["mae"]),
            "macro_group_mae_reduction": float(
                metrics[reference]["macro_group_mae"]
                - metrics[candidate]["macro_group_mae"]
            ),
            "sign_accuracy_gain": float(
                metrics[candidate]["sign_accuracy"] - metrics[reference]["sign_accuracy"]
            ),
        }

    improvements = {
        "primary_delta3d_vs_2d_ensemble": comparison("2d_ensemble", "delta3d"),
        "delta3d_unshrunk_vs_2d_second_unshrunk": comparison(
            "2d_second_unshrunk", "delta3d_unshrunk"
        ),
        "delta3d_vs_baseline": comparison("baseline", "delta3d"),
        "2d_ensemble_vs_baseline": comparison("baseline", "2d_ensemble"),
    }
    bootstrap_comparisons = {
        "delta3d_vs_2d_ensemble": ("prediction_2d_ensemble", "prediction_delta3d"),
        "delta3d_unshrunk_vs_2d_second_unshrunk": (
            "prediction_2d_second_unshrunk",
            "prediction_delta3d_unshrunk",
        ),
        "delta3d_vs_baseline": ("prediction_baseline", "prediction_delta3d"),
        "2d_ensemble_vs_baseline": ("prediction_baseline", "prediction_2d_ensemble"),
    }
    if has_extension:
        improvements["primary_extended_vs_delta3d"] = comparison(
            "delta3d", "delta3d_extended"
        )
        improvements["extended_unshrunk_vs_delta3d_unshrunk"] = comparison(
            "delta3d_unshrunk", "delta3d_extended_unshrunk"
        )
        improvements["extended_vs_2d_ensemble"] = comparison(
            "2d_ensemble", "delta3d_extended"
        )
        bootstrap_comparisons["extended_vs_delta3d"] = (
            "prediction_delta3d",
            "prediction_delta3d_extended",
        )
        bootstrap_comparisons["extended_unshrunk_vs_delta3d_unshrunk"] = (
            "prediction_delta3d_unshrunk",
            "prediction_delta3d_extended_unshrunk",
        )
        bootstrap_comparisons["extended_vs_2d_ensemble"] = (
            "prediction_2d_ensemble",
            "prediction_delta3d_extended",
        )
    bootstrap = _cluster_bootstrap_comparisons(
        predictions,
        group_column=group_column,
        n_bootstrap=n_bootstrap,
        seed=seed + 999_983,
        comparisons=bootstrap_comparisons,
    )

    # A multi-seed evaluation should not repeat an unused full-data deployment
    # fit in every array task.  It can be enabled for one separately chosen run.
    final_model: ShrinkageDelta3DRegressor | None = None
    importances = pd.DataFrame(
        columns=(
            "feature",
            "deployment_delta3d_blend_weight",
            "component",
            "interpretation",
            "importance",
        )
    )
    final_deployment_parameters: dict[str, Any] = {
        "fitted": False,
        "reason": "evaluation_only",
    }
    if fit_final_model:
        # These fits are not used for any reported outer-fold metric.
        final_split_seed = split_seed + 9_999_991
        (
            final_baseline_parameters,
            final_baseline_tuning,
            final_baseline_oof,
        ) = _tune_parameters(
            frame,
            target,
            groups,
            pair_data.baseline_columns,
            parameter_grid=parameter_grid,
            inner_folds=inner_folds,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            model_seed=seed + 1_499_993,
            split_seed=final_split_seed,
            context="full_data_deployment_model:baseline",
        )
        tuning_rows.extend(final_baseline_tuning)
        final_full_parameters, final_full_tuning, final_full_oof = _tune_parameters(
            frame,
            target,
            groups,
            pair_data.full_columns,
            parameter_grid=parameter_grid,
            inner_folds=inner_folds,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            model_seed=seed + 1_999_999,
            split_seed=final_split_seed,
            context="full_data_deployment_model:delta3d",
        )
        tuning_rows.extend(final_full_tuning)
        final_delta3d_weight, final_weight_rows = _select_delta3d_weight(
            target,
            groups,
            final_baseline_oof,
            final_full_oof,
        )
        for row in final_weight_rows:
            tuning_rows.append(
                {"context": "full_data_deployment_model:delta3d_weight", **row}
            )

        final_baseline_model = AntisymmetricExtraTreesRegressor(
            pair_data.baseline_columns,
            n_estimators=n_estimators,
            max_features=final_baseline_parameters["max_features"],
            min_samples_leaf=final_baseline_parameters["min_samples_leaf"],
            random_state=seed + 2_899_981,
            n_jobs=n_jobs,
        )
        final_full_model = AntisymmetricExtraTreesRegressor(
            pair_data.full_columns,
            n_estimators=n_estimators,
            max_features=final_full_parameters["max_features"],
            min_samples_leaf=final_full_parameters["min_samples_leaf"],
            random_state=seed + 2_999_983,
            n_jobs=n_jobs,
        )
        final_baseline_model.fit(frame, target, groups)
        final_full_model.fit(frame, target, groups)
        final_model = ShrinkageDelta3DRegressor(
            baseline_model=final_baseline_model,
            full_model=final_full_model,
            delta3d_weight=final_delta3d_weight,
        )
        importances = final_model.feature_importance_frame()
        final_deployment_parameters = {
            "fitted": True,
            "inner_split_seed": int(final_split_seed),
            "baseline": final_baseline_parameters,
            "delta3d_unshrunk": final_full_parameters,
            "delta3d_weight": final_delta3d_weight,
        }

    per_extractant = _group_metric_table(
        predictions, "extractant", "extractant", model_names
    )
    per_pair = _group_metric_table(predictions, "pair_label", "pair_type", model_names)
    fold_assignments = predictions[
        ["pair_id", "extractant", "ecfp_exact_cluster", "outer_fold"]
    ].copy()
    fold_assignments["outer_split_seed"] = int(split_seed)

    leakage_passed = all(
        row["group_overlap"] == 0
        and row["extractant_overlap"] == 0
        and row["ecfp_exact_cluster_overlap"] == 0
        and row["source_id_overlap"] == 0
        and row["geometry_key_overlap"] == 0
        for row in leakage_folds
    )

    summary = {
        "protocol": {
            "outer_split": (
                f"{len(folds)}-fold seeded shuffled StratifiedGroupKFold; "
                "stratified by pair_label"
            ),
            "inner_split": (
                f"up to {inner_folds}-fold seeded shuffled StratifiedGroupKFold; "
                "shared by all model branches within an outer fold"
            ),
            "group_column": group_column,
            "group_overlap_required": 0,
            "preprocessing_scope": "SimpleImputer fitted inside every inner/outer training fold",
            "training_weights": "equal total weight per training group",
            "pair_swap_augmentation": True,
            "antisymmetric_inference": "(f(A,B) - f(B,A)) / 2",
            "delta3d_shrinkage": (
                "Inner group CV selects w in {0,.25,.5,.75,1}; "
                "prediction = baseline + w * (full_delta3d - baseline)"
            ),
            "matched_2d_ensemble_control": (
                "An independently tuned second 2D forest uses the same seed stream and "
                "selection budget as the full Delta3D forest; its blend weight is selected "
                "by the same inner-CV rule."
            ),
            "primary_comparison": "adaptive Delta3D blend minus matched adaptive 2D+2D blend",
            "metal_site_descriptor_arm": (
                "adaptive blend of a Delta3D+descriptor forest built with the same seed "
                "stream, selection budget and blend rule as the Delta3D forest; the "
                "extended-minus-Delta3D contrast isolates the descriptor columns"
                if has_extension
                else "absent"
            ),
            "model_selection_metric": "inner macro-group MAE",
            "trees_per_fit": int(n_estimators),
            "model_seed": int(seed),
            "split_seed": int(split_seed),
            "fit_final_model": bool(fit_final_model),
        },
        "feature_counts": {
            "baseline": len(pair_data.baseline_columns),
            "delta3d_added": len(pair_data.delta3d_columns),
            "full": len(pair_data.full_columns),
            "metal_site_descriptors_added": len(pair_data.descriptor_columns),
            "extended": len(pair_data.extended_columns),
        },
        "model_names": list(model_names),
        "metrics": metrics,
        "improvements": improvements,
        "paired_group_bootstrap": bootstrap,
        "leakage_audit": {
            "passed": leakage_passed,
            "outer_folds": leakage_folds,
        },
        "selected_hyperparameters_by_outer_fold": selected_by_fold,
        "final_deployment_parameters": final_deployment_parameters,
        "interpretation_guardrail": (
            "This exploratory benchmark cannot establish an experimental separation factor "
            "without publication/experiment-series identifiers. A robust 3D claim additionally "
            "requires positive Delta3D-vs-2D-ensemble and unshrunk matched-control effects across "
            "prespecified seeds/splits and prospective data."
        ),
    }
    return BenchmarkResult(
        summary=summary,
        predictions=predictions,
        fold_assignments=fold_assignments,
        tuning_results=pd.DataFrame(tuning_rows),
        per_extractant_metrics=per_extractant,
        per_pair_metrics=per_pair,
        final_model=final_model,
        feature_importances=importances,
    )
