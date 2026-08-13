"""Nested, multi-arm evaluation for the frozen generation-3 protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from .ablation import _identity_overlap, _training_shuffle
from .evaluation import (
    DEFAULT_PARAMETER_GRID,
    AntisymmetricExtraTreesRegressor,
    _group_folds,
)
from .gen3_data import Gen3FeatureAdapter
from .gen3_metrics import (
    arm_metric_table,
    equal_group_macro_mae,
    paired_extractant_bootstrap,
)
from .gen3_models import (
    AntisymmetricCatBoostRegressor,
    AntisymmetricCorrectionRegressor,
    fit_latent_difference_model,
)
from .pairs import PAIR_TARGET_COLUMN


BASELINE_ARM = "A2_current_champion"


@dataclass(frozen=True)
class Gen3RunResult:
    predictions: pd.DataFrame
    split_assignments: pd.DataFrame
    inner_fold_assignments: pd.DataFrame
    tuning_results: pd.DataFrame
    weight_audit: pd.DataFrame
    residual_audit: pd.DataFrame
    shuffle_audit: pd.DataFrame
    e3_stability: pd.DataFrame
    per_seed_metrics: pd.DataFrame
    per_extractant_metrics: pd.DataFrame
    family_metrics: pd.DataFrame
    paired_bootstrap: pd.DataFrame
    leakage_audit: dict[str, Any]
    selected_stage1: pd.DataFrame
    selected_lambdas: pd.DataFrame
    run_metadata: dict[str, Any]


def _prediction_seed(model_seed: int, outer_fold: int, *parts: int) -> int:
    value = int(model_seed) + int(outer_fold) * 1_000_003
    for index, part in enumerate(parts, start=1):
        value += int(part) * (10_007 + index * 4_099)
    return int(value % 4_000_000_000)


def _current_a2_tuning(
    frame: pd.DataFrame,
    target: np.ndarray,
    groups: np.ndarray,
    columns: tuple[str, ...],
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    outer_fold: int,
    model_seed: int,
    n_jobs: int,
    n_estimators: int,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    candidate_rows: list[dict[str, Any]] = []
    predictions: dict[int, np.ndarray] = {}
    for candidate_index, parameters in enumerate(DEFAULT_PARAMETER_GRID):
        oof = np.full(len(frame), np.nan, dtype=float)
        for inner_fold, (train, validation) in enumerate(inner_splits):
            model = AntisymmetricExtraTreesRegressor(
                columns,
                n_estimators=int(n_estimators),
                max_features=float(parameters["max_features"]),
                min_samples_leaf=int(parameters["min_samples_leaf"]),
                # Preserve the generation-2 A2 seed stream exactly. This arm is
                # the current champion, not a newly randomized reimplementation.
                random_state=(
                    int(model_seed)
                    + int(outer_fold) * 1009
                    + int(candidate_index) * 101
                    + int(inner_fold)
                ),
                n_jobs=n_jobs,
            )
            model.fit(frame.iloc[train], target[train], groups[train])
            oof[validation] = model.predict(frame.iloc[validation])
        if not np.isfinite(oof).all():
            raise AssertionError("Current A2 inner OOF prediction is incomplete.")
        macro = equal_group_macro_mae(target, oof, groups)
        pooled_mae = float(mean_absolute_error(target, oof))
        pooled_r2 = float(r2_score(target, oof))
        candidate_rows.append(
            {
                "outer_fold": int(outer_fold),
                "arm": BASELINE_ARM,
                "candidate_index": int(candidate_index),
                "inner_macro_mae": macro,
                "inner_mae": pooled_mae,
                "inner_r2": pooled_r2,
                **parameters,
            }
        )
        predictions[candidate_index] = oof
    best = min(
        candidate_rows,
        key=lambda row: (
            float(row["inner_macro_mae"]),
            float(row["inner_mae"]),
            -float(row["inner_r2"]),
            int(row["candidate_index"]),
        ),
    )
    for row in candidate_rows:
        row["selected"] = row["candidate_index"] == best["candidate_index"]
    parameters = {
        "max_features": float(best["max_features"]),
        "min_samples_leaf": int(best["min_samples_leaf"]),
    }
    return parameters, predictions[int(best["candidate_index"])], candidate_rows


def _h1_arm_name(weighting: str, loss_name: str) -> str:
    return f"H1_CATBOOST_{weighting.upper()}_{loss_name.upper()}"


def _catboost_parameters(
    candidate: Mapping[str, Any], loss_function: str
) -> dict[str, Any]:
    return {
        "loss_function": str(loss_function),
        "depth": int(candidate["depth"]),
        "learning_rate": float(candidate["learning_rate"]),
        "l2_leaf_reg": float(candidate["l2_leaf_reg"]),
        "iterations": int(candidate["iterations"]),
        "random_strength": float(candidate["random_strength"]),
        "bagging_temperature": float(candidate["bagging_temperature"]),
        "rsm": float(candidate["rsm"]),
        "bootstrap_type": "Bayesian",
    }


def _h1_tuning(
    frame: pd.DataFrame,
    target: np.ndarray,
    groups: np.ndarray,
    columns: tuple[str, ...],
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    arm: str,
    weighting: str,
    loss_name: str,
    loss_function: str,
    search_phase: str,
    candidates: Iterable[Mapping[str, Any]],
    outer_fold: int,
    model_seed: int,
    n_jobs: int,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    if search_phase not in {"screening", "expanded"}:
        raise ValueError(f"Unknown H1 search phase: {search_phase!r}.")
    phase_seed = 0 if search_phase == "screening" else 1
    tuning: list[dict[str, Any]] = []
    weight_audit: list[dict[str, Any]] = []
    candidate_oof: dict[int, np.ndarray] = {}
    candidate_list = tuple(candidates)
    for candidate_index, candidate in enumerate(candidate_list):
        oof = np.full(len(frame), np.nan, dtype=float)
        parameters = _catboost_parameters(candidate, loss_function)
        for inner_fold, (train, validation) in enumerate(inner_splits):
            train_groups = set(groups[train])
            validation_groups = set(groups[validation])
            if train_groups & validation_groups:
                raise AssertionError("H1 inner group leakage detected.")
            model = AntisymmetricCatBoostRegressor(
                columns,
                parameters=parameters,
                weighting_scheme=weighting,
                random_state=_prediction_seed(
                    model_seed,
                    outer_fold,
                    101,
                    phase_seed,
                    candidate_index,
                    inner_fold,
                ),
                n_jobs=n_jobs,
            )
            model.fit(frame.iloc[train], target[train], groups[train])
            oof[validation] = model.predict(frame.iloc[validation])
            if candidate_index == 0:
                weight_audit.append(
                    {
                        "outer_fold": int(outer_fold),
                        "inner_fold": int(inner_fold),
                        "arm": arm,
                        "search_phase": search_phase,
                        "scope": "inner_training_only",
                        **dict(model.fit_weight_audit or {}),
                    }
                )
        if not np.isfinite(oof).all():
            raise AssertionError("H1 CatBoost inner OOF prediction is incomplete.")
        macro = equal_group_macro_mae(target, oof, groups)
        tuning.append(
            {
                "outer_fold": int(outer_fold),
                "arm": arm,
                "search_phase": search_phase,
                "candidate_index": int(candidate_index),
                "weighting_scheme": weighting,
                "loss_name": loss_name,
                "loss_function": loss_function,
                "inner_macro_mae": macro,
                **dict(candidate),
            }
        )
        candidate_oof[candidate_index] = oof
    best = min(
        tuning,
        key=lambda row: (float(row["inner_macro_mae"]), int(row["candidate_index"])),
    )
    for row in tuning:
        row["selected"] = row["candidate_index"] == best["candidate_index"]
    selected_index = int(best["candidate_index"])
    return (
        _catboost_parameters(candidate_list[selected_index], loss_function),
        candidate_oof[selected_index],
        tuning,
        weight_audit,
    )


def _new_stage1(
    columns: tuple[str, ...],
    specification: Mapping[str, Any],
    *,
    random_state: int,
    n_jobs: int,
) -> AntisymmetricCatBoostRegressor:
    return AntisymmetricCatBoostRegressor(
        columns,
        parameters=specification["parameters"],
        weighting_scheme=str(specification["weighting_scheme"]),
        random_state=int(random_state),
        n_jobs=n_jobs,
    )


def _crossfit_stage1_within_subset(
    frame: pd.DataFrame,
    target: np.ndarray,
    groups: np.ndarray,
    specification: Mapping[str, Any],
    columns: tuple[str, ...],
    *,
    split_seed: int,
    model_seed: int,
    outer_fold: int,
    inner_fold: int,
    n_jobs: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    sub_splits = _group_folds(
        groups,
        frame["pair_label"].astype(str).to_numpy(),
        3,
        split_seed=int(split_seed),
        allow_fewer_splits=True,
    )
    oof = np.full(len(frame), np.nan, dtype=float)
    audit: list[dict[str, Any]] = []
    for sub_fold, (fit, validation) in enumerate(sub_splits):
        if set(groups[fit]) & set(groups[validation]):
            raise AssertionError("Stage-1 residual sub-crossfit group leakage.")
        model = _new_stage1(
            columns,
            specification,
            random_state=_prediction_seed(
                model_seed, outer_fold, 211, inner_fold, sub_fold
            ),
            n_jobs=n_jobs,
        )
        model.fit(frame.iloc[fit], target[fit], groups[fit])
        oof[validation] = model.predict(frame.iloc[validation])
        audit.append(
            {
                "outer_fold": int(outer_fold),
                "correction_inner_fold": int(inner_fold),
                "stage1_subfold": int(sub_fold),
                "fit_rows": int(len(fit)),
                "residual_rows_predicted": int(len(validation)),
                "fit_groups": int(len(set(groups[fit]))),
                "residual_groups": int(len(set(groups[validation]))),
                "group_overlap": int(len(set(groups[fit]) & set(groups[validation]))),
                "residual_source": "stage1_out_of_fold_prediction",
                "in_sample_residuals": 0,
            }
        )
    if not np.isfinite(oof).all():
        raise AssertionError("Stage-1 sub-crossfit did not predict every correction row.")
    return target - oof, audit


def _correction_grid(protocol: Mapping[str, Any], *, quick: bool) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    heads = protocol["h2"]["correction_heads"]
    for alpha in heads["ridge_alpha"]:
        result.append({"kind": "ridge", "parameters": {"alpha": float(alpha)}})
    for parameters in heads["elasticnet"]:
        result.append({"kind": "elasticnet", "parameters": dict(parameters)})
    for parameters in heads["catboost"]:
        result.append({"kind": "catboost", "parameters": dict(parameters)})
    if quick:
        # Smoke-only budget: one representative per estimator family.
        result = [
            next(item for item in result if item["kind"] == kind)
            for kind in ("ridge", "elasticnet", "catboost")
        ]
        for item in result:
            if item["kind"] == "catboost":
                item["parameters"]["iterations"] = min(
                    5, int(item["parameters"]["iterations"])
                )
    return result


def _prepare_h2_inner_sources(
    outer_train: pd.DataFrame,
    outer_target: np.ndarray,
    outer_groups: np.ndarray,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    stage1_specification: Mapping[str, Any],
    a2_columns: tuple[str, ...],
    *,
    split_seed: int,
    model_seed: int,
    outer_fold: int,
    n_jobs: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sources: list[dict[str, Any]] = []
    residual_audit: list[dict[str, Any]] = []
    for inner_fold, (fit, validation) in enumerate(inner_splits):
        fit_frame = outer_train.iloc[fit]
        validation_frame = outer_train.iloc[validation]
        residual_fit, audit = _crossfit_stage1_within_subset(
            fit_frame.reset_index(drop=True),
            outer_target[fit],
            outer_groups[fit],
            stage1_specification,
            a2_columns,
            split_seed=int(split_seed + outer_fold * 100_003 + inner_fold * 7_919),
            model_seed=model_seed,
            outer_fold=outer_fold,
            inner_fold=inner_fold,
            n_jobs=n_jobs,
        )
        residual_audit.extend(audit)
        base_model = _new_stage1(
            a2_columns,
            stage1_specification,
            random_state=_prediction_seed(model_seed, outer_fold, 307, inner_fold),
            n_jobs=n_jobs,
        )
        base_model.fit(fit_frame, outer_target[fit], outer_groups[fit])
        base_validation = base_model.predict(validation_frame)
        sources.append(
            {
                "inner_fold": int(inner_fold),
                "fit": fit,
                "validation": validation,
                "fit_frame": fit_frame,
                "validation_frame": validation_frame,
                "fit_groups": outer_groups[fit],
                "residual_fit": residual_fit,
                "base_validation": base_validation,
            }
        )
    return sources, residual_audit


def _tune_h2_view(
    outer_train: pd.DataFrame,
    outer_target: np.ndarray,
    outer_groups: np.ndarray,
    sources: list[dict[str, Any]],
    columns: tuple[str, ...],
    correction_grid: list[dict[str, Any]],
    lambda_grid: tuple[float, ...],
    *,
    arm: str,
    shuffled: bool,
    shuffle_seed: int,
    model_seed: int,
    outer_fold: int,
    n_jobs: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[np.ndarray]]:
    tuning: list[dict[str, Any]] = []
    shuffle_rows: list[dict[str, Any]] = []
    coefficient_by_config: list[list[np.ndarray]] = []
    candidate_corrections: dict[int, np.ndarray] = {}
    base_oof = np.full(len(outer_train), np.nan, dtype=float)
    for source in sources:
        base_oof[source["validation"]] = source["base_validation"]
    if not np.isfinite(base_oof).all():
        raise AssertionError("H2 inner Stage-1 predictions are incomplete.")

    for candidate_index, specification in enumerate(correction_grid):
        correction_oof = np.full(len(outer_train), np.nan, dtype=float)
        fold_coefficients: list[np.ndarray] = []
        for source in sources:
            fit_frame = source["fit_frame"]
            if shuffled:
                applied_seed = int(
                    shuffle_seed
                    + outer_fold * 1_000_003
                    + source["inner_fold"] * 10_009
                )
                fit_frame, audit = _training_shuffle(
                    fit_frame, columns, seed=applied_seed
                )
                if candidate_index == 0:
                    shuffle_rows.append(
                        {
                            "arm": arm,
                            "outer_fold": int(outer_fold),
                            "inner_fold": int(source["inner_fold"]),
                            "scope": "correction_inner_training_only",
                            "shuffle_seed": applied_seed,
                            "validation_rows_touched": 0,
                            "validation_labels_used": 0,
                            **audit,
                        }
                    )
            model = AntisymmetricCorrectionRegressor(
                columns,
                kind=specification["kind"],
                parameters=specification["parameters"],
                random_state=_prediction_seed(
                    model_seed,
                    outer_fold,
                    401,
                    candidate_index,
                    source["inner_fold"],
                ),
                n_jobs=n_jobs,
            )
            model.fit(
                fit_frame,
                source["residual_fit"],
                source["fit_groups"],
            )
            correction_oof[source["validation"]] = model.predict(
                source["validation_frame"]
            )
            coefficients = model.signed_coefficients()
            if coefficients is not None:
                fold_coefficients.append(coefficients)
        if not np.isfinite(correction_oof).all():
            raise AssertionError("H2 inner correction prediction is incomplete.")
        coefficient_by_config.append(fold_coefficients)
        candidate_corrections[candidate_index] = correction_oof
        for value in lambda_grid:
            prediction = base_oof + float(value) * correction_oof
            tuning.append(
                {
                    "outer_fold": int(outer_fold),
                    "arm": arm,
                    "candidate_index": int(candidate_index),
                    "correction_kind": specification["kind"],
                    "correction_parameters": str(specification["parameters"]),
                    "lambda": float(value),
                    "inner_macro_mae": equal_group_macro_mae(
                        outer_target, prediction, outer_groups
                    ),
                }
            )
    best = min(
        tuning,
        key=lambda row: (
            float(row["inner_macro_mae"]),
            float(row["lambda"]),
            int(row["candidate_index"]),
        ),
    )
    for row in tuning:
        row["selected"] = bool(
            row["candidate_index"] == best["candidate_index"]
            and row["lambda"] == best["lambda"]
        )
    selected_index = int(best["candidate_index"])
    return (
        {
            "specification": correction_grid[selected_index],
            "lambda": float(best["lambda"]),
            "inner_macro_mae": float(best["inner_macro_mae"]),
            "correction_oof": candidate_corrections[selected_index],
        },
        tuning,
        shuffle_rows,
        coefficient_by_config,
    )


def _fit_h2_outer(
    outer_train: pd.DataFrame,
    outer_test: pd.DataFrame,
    outer_residual: np.ndarray,
    outer_groups: np.ndarray,
    columns: tuple[str, ...],
    selection: Mapping[str, Any],
    *,
    arm: str,
    shuffled: bool,
    shuffle_seed: int,
    model_seed: int,
    outer_fold: int,
    n_jobs: int,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    fit_frame = outer_train
    shuffle_row = None
    if shuffled:
        applied_seed = int(shuffle_seed + outer_fold * 1_000_003 + 999_983)
        fit_frame, audit = _training_shuffle(fit_frame, columns, seed=applied_seed)
        shuffle_row = {
            "arm": arm,
            "outer_fold": int(outer_fold),
            "inner_fold": None,
            "scope": "outer_training_only",
            "shuffle_seed": applied_seed,
            "test_rows_touched": 0,
            "test_labels_used": 0,
            **audit,
        }
    specification = selection["specification"]
    model = AntisymmetricCorrectionRegressor(
        columns,
        kind=specification["kind"],
        parameters=specification["parameters"],
        random_state=_prediction_seed(model_seed, outer_fold, 503),
        n_jobs=n_jobs,
    )
    model.fit(fit_frame, outer_residual, outer_groups)
    return model.predict(outer_test), shuffle_row


def _stability_columns(
    columns: tuple[str, ...],
    coefficient_folds: list[np.ndarray],
    *,
    threshold: float,
) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    if not coefficient_folds:
        # ElasticNet is always in the declared grid; absence is a contract error.
        raise AssertionError("Nested E3 stability selection has no coefficients.")
    matrix = np.vstack(coefficient_folds)
    frequency = np.mean(np.abs(matrix) > 1e-12, axis=0)
    signed = np.mean(matrix, axis=0)
    consistency = np.abs(np.mean(np.sign(matrix), axis=0))
    selected_mask = frequency >= float(threshold)
    if not selected_mask.any():
        selected_mask[int(np.argmax(frequency))] = True
    rows = [
        {
            "feature": column,
            "selection_frequency": float(frequency[index]),
            "mean_signed_importance_or_coefficient": float(signed[index]),
            "fold_consistency": float(consistency[index]),
            "selected": bool(selected_mask[index]),
        }
        for index, column in enumerate(columns)
    ]
    selected = tuple(
        column for column, keep in zip(columns, selected_mask, strict=True) if keep
    )
    return selected, rows


def _h3_tuning(
    outer_train: pd.DataFrame,
    outer_target: np.ndarray,
    outer_groups: np.ndarray,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    arm: str,
    a_columns: tuple[str, ...],
    b_columns: tuple[str, ...],
    candidates: Iterable[Mapping[str, Any]],
    training: Mapping[str, Any],
    huber_delta: float,
    model_seed: int,
    outer_fold: int,
    quick: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates_tuple = tuple(candidates)
    tuning: list[dict[str, Any]] = []
    max_epochs = min(int(training["max_epochs"]), 3) if quick else int(training["max_epochs"])
    patience = min(int(training["early_stopping_patience"]), 2) if quick else int(
        training["early_stopping_patience"]
    )
    for candidate_index, parameters in enumerate(candidates_tuple):
        oof = np.full(len(outer_train), np.nan, dtype=float)
        epoch_counts: list[int] = []
        for inner_fold, (fit, validation) in enumerate(inner_splits):
            if set(outer_groups[fit]) & set(outer_groups[validation]):
                raise AssertionError("H3 inner group leakage detected.")
            model = fit_latent_difference_model(
                outer_train.iloc[fit],
                outer_target[fit],
                outer_groups[fit],
                a_columns=a_columns,
                b_columns=b_columns,
                parameters=parameters,
                model_seed=_prediction_seed(
                    model_seed, outer_fold, 601, candidate_index, inner_fold
                ),
                max_epochs=max_epochs,
                patience=patience,
                batch_size=int(training["batch_size"]),
                huber_delta=huber_delta,
            )
            oof[validation] = model.predict(outer_train.iloc[validation])
            epoch_counts.append(model.epochs_trained)
        if not np.isfinite(oof).all():
            raise AssertionError("H3 inner OOF prediction is incomplete.")
        tuning.append(
            {
                "outer_fold": int(outer_fold),
                "arm": arm,
                "candidate_index": int(candidate_index),
                "parameters": str(dict(parameters)),
                "inner_macro_mae": equal_group_macro_mae(
                    outer_target, oof, outer_groups
                ),
                "mean_epochs_trained": float(np.mean(epoch_counts)),
            }
        )
    best = min(
        tuning,
        key=lambda row: (float(row["inner_macro_mae"]), int(row["candidate_index"])),
    )
    for row in tuning:
        row["selected"] = row["candidate_index"] == best["candidate_index"]
    return dict(candidates_tuple[int(best["candidate_index"])]), tuning


def _training_label_shuffle(
    target: np.ndarray, frame: pd.DataFrame, *, seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """Permute outer-training labels within pair type; test labels are absent."""

    if len(target) != len(frame):
        raise ValueError("Label shuffle target/frame length mismatch.")
    rng = np.random.default_rng(int(seed))
    shuffled = np.asarray(target, dtype=float).copy()
    moved = 0
    for _, positions in frame.groupby("pair_label", sort=True).indices.items():
        positions_array = np.asarray(positions, dtype=int)
        if len(positions_array) <= 1:
            continue
        donor = rng.permutation(positions_array)
        shuffled[positions_array] = target[donor]
        moved += int(np.sum(donor != positions_array))
    return shuffled, {
        "training_rows": int(len(frame)),
        "moved_label_rows": int(moved),
        "fixed_label_rows": int(len(frame) - moved),
        "stratification": "pair_label",
        "training_only": True,
        "outer_test_labels_touched": 0,
        "outer_test_labels_used_for_selection": 0,
    }


def run_gen3_benchmark(
    adapter: Gen3FeatureAdapter,
    protocol: Mapping[str, Any],
    *,
    split_seed: int,
    model_seed: int,
    run_role: str,
    n_jobs: int = -1,
    quick: bool = False,
) -> Gen3RunResult:
    """Run one predeclared split/model-seed generation-3 nested benchmark."""

    frame = adapter.frame.reset_index(drop=True).copy()
    groups = frame["extractant"].astype(str).to_numpy()
    target = frame[PAIR_TARGET_COLUMN].to_numpy(dtype=float)
    strata = frame["pair_label"].astype(str).to_numpy()
    splits = protocol["splits"]
    outer_splits = _group_folds(
        groups,
        strata,
        int(splits["outer_folds"]),
        split_seed=int(split_seed),
        allow_fewer_splits=False,
    )

    h1_screening_candidates = tuple(protocol["h1"]["screening_candidates"])
    h1_expanded_candidates = tuple(protocol["h1"]["expanded_candidates"])
    if quick:
        smoke_candidate = dict(h1_screening_candidates[0])
        smoke_candidate["iterations"] = min(5, int(smoke_candidate["iterations"]))
        h1_screening_candidates = (smoke_candidate,)
        smoke_expanded_candidate = dict(h1_expanded_candidates[0])
        smoke_expanded_candidate["iterations"] = min(
            5, int(smoke_expanded_candidate["iterations"])
        )
        h1_expanded_candidates = (smoke_expanded_candidate,)
    h1_specs = [
        (weighting, loss_name, loss_function)
        for weighting in protocol["h1"]["weighting_schemes"]
        for loss_name, loss_function in protocol["h1"]["losses"].items()
    ]
    h1_arms = tuple(
        _h1_arm_name(weighting, loss_name)
        for weighting, loss_name, _ in h1_specs
    )
    h2_arms = tuple(protocol["h2"]["full_e3_arms"]) + tuple(
        protocol["h2"]["stability_followup"]["arms"]
    )
    h3_arms = tuple(protocol["h3"]["arms"])
    h3_control_arms = tuple(protocol["h3"]["label_shuffle_control_arms"])
    arms = (BASELINE_ARM,) + h1_arms + h2_arms + h3_arms + h3_control_arms

    metadata_columns = [
        "pair_id",
        "condition_id",
        "extractant",
        "extractant_family",
        "pair_label",
        "metal_A",
        "metal_B",
        "pair__Z_A",
        "pair__Z_B",
        PAIR_TARGET_COLUMN,
    ]
    predictions = frame.loc[:, metadata_columns].copy()
    predictions["outer_fold"] = -1
    for arm in arms:
        predictions[f"prediction_{arm}"] = np.nan

    split_rows: list[dict[str, Any]] = []
    inner_assignment_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    shuffle_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    selected_stage1_rows: list[dict[str, Any]] = []
    selected_lambda_rows: list[dict[str, Any]] = []
    pair_ids = frame["pair_id"].astype(str).to_numpy()

    for outer_fold, (train, test) in enumerate(outer_splits):
        outer_train = frame.iloc[train]
        outer_test = frame.iloc[test]
        outer_target = target[train]
        outer_groups = groups[train]
        train_groups = set(groups[train])
        test_groups = set(groups[test])
        leakage = {
            "outer_fold": int(outer_fold),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_extractants": int(len(train_groups)),
            "test_extractants": int(len(test_groups)),
            "extractant_overlap": int(len(train_groups & test_groups)),
            "source_id_overlap": _identity_overlap(
                outer_train, outer_test, ("source_id_A", "source_id_B")
            ),
            "geometry_key_overlap": _identity_overlap(
                outer_train, outer_test, ("geometry_key_A", "geometry_key_B")
            ),
            "outer_test_labels_used_for_selection": False,
            "identical_outer_test_rows_for_all_arms": True,
        }
        if any(
            int(leakage[key])
            for key in ("extractant_overlap", "source_id_overlap", "geometry_key_overlap")
        ):
            raise AssertionError(f"Gen3 outer leakage detected: {leakage}")
        leakage_rows.append(leakage)
        predictions.loc[test, "outer_fold"] = int(outer_fold)
        test_positions = set(test.tolist())
        for index in range(len(frame)):
            split_rows.append(
                {
                    "pair_id": pair_ids[index],
                    "extractant": groups[index],
                    "outer_fold": int(outer_fold),
                    "assignment": "test" if index in test_positions else "train",
                    "split_seed": int(split_seed),
                }
            )

        inner_seed = int(split_seed + outer_fold * 10_007)
        inner_splits = _group_folds(
            outer_groups,
            outer_train["pair_label"].astype(str).to_numpy(),
            int(splits["inner_folds"]),
            split_seed=inner_seed,
            allow_fewer_splits=True,
        )
        for inner_fold, (_, validation) in enumerate(inner_splits):
            for local in validation:
                inner_assignment_rows.append(
                    {
                        "pair_id": pair_ids[train[local]],
                        "extractant": outer_groups[local],
                        "outer_fold": int(outer_fold),
                        "inner_validation_fold": int(inner_fold),
                        "inner_split_seed": inner_seed,
                    }
                )

        baseline_parameters, _, baseline_tuning = _current_a2_tuning(
            outer_train,
            outer_target,
            outer_groups,
            adapter.a2_columns,
            inner_splits,
            outer_fold=outer_fold,
            model_seed=model_seed,
            n_jobs=n_jobs,
            n_estimators=48 if quick else 200,
        )
        tuning_rows.extend(baseline_tuning)
        baseline_model = AntisymmetricExtraTreesRegressor(
            adapter.a2_columns,
            n_estimators=48 if quick else 200,
            max_features=baseline_parameters["max_features"],
            min_samples_leaf=baseline_parameters["min_samples_leaf"],
            # Same outer-fit seed as run_ablation_benchmark(A2).
            random_state=int(model_seed) + int(outer_fold) * 1009 + 9_999_991,
            n_jobs=n_jobs,
        )
        baseline_model.fit(outer_train, outer_target, outer_groups)
        predictions.loc[test, f"prediction_{BASELINE_ARM}"] = baseline_model.predict(
            outer_test
        )

        h1_selected: dict[str, dict[str, Any]] = {}
        h1_tuning_by_arm: dict[str, list[dict[str, Any]]] = {}
        h1_weights_by_arm: dict[str, list[dict[str, Any]]] = {}
        for weighting, loss_name, loss_function in h1_specs:
            arm = _h1_arm_name(weighting, loss_name)
            parameters, stage1_oof, tuning, weights = _h1_tuning(
                outer_train,
                outer_target,
                outer_groups,
                adapter.a2_columns,
                inner_splits,
                arm=arm,
                weighting=weighting,
                loss_name=loss_name,
                loss_function=loss_function,
                search_phase="screening",
                candidates=h1_screening_candidates,
                outer_fold=outer_fold,
                model_seed=model_seed,
                n_jobs=n_jobs,
            )
            selected_score = min(
                float(row["inner_macro_mae"]) for row in tuning if row["selected"]
            )
            selected_index = next(
                int(row["candidate_index"]) for row in tuning if row["selected"]
            )
            for row in tuning:
                row["phase_selected"] = bool(row["selected"])
            h1_selected[arm] = {
                "arm": arm,
                "parameters": parameters,
                "weighting_scheme": weighting,
                "loss_name": loss_name,
                "loss_function": loss_function,
                "search_phase": "screening",
                "selected_candidate_index": selected_index,
                "inner_macro_mae": selected_score,
                "outer_train_oof": stage1_oof,
            }
            h1_tuning_by_arm[arm] = tuning
            h1_weights_by_arm[arm] = weights

        if h1_expanded_candidates:
            shortlist_count = (
                1
                if quick
                else int(protocol["h1"]["expanded_shortlist_weight_loss_arms"])
            )
            shortlisted_arms = {
                item["arm"]
                for item in sorted(
                    h1_selected.values(),
                    key=lambda item: (
                        float(item["inner_macro_mae"]),
                        str(item["arm"]),
                    ),
                )[:shortlist_count]
            }
            for weighting, loss_name, loss_function in h1_specs:
                arm = _h1_arm_name(weighting, loss_name)
                if arm not in shortlisted_arms:
                    continue
                parameters, stage1_oof, tuning, weights = _h1_tuning(
                    outer_train,
                    outer_target,
                    outer_groups,
                    adapter.a2_columns,
                    inner_splits,
                    arm=arm,
                    weighting=weighting,
                    loss_name=loss_name,
                    loss_function=loss_function,
                    search_phase="expanded",
                    candidates=h1_expanded_candidates,
                    outer_fold=outer_fold,
                    model_seed=model_seed,
                    n_jobs=n_jobs,
                )
                selected_score = min(
                    float(row["inner_macro_mae"])
                    for row in tuning
                    if row["selected"]
                )
                selected_index = next(
                    int(row["candidate_index"])
                    for row in tuning
                    if row["selected"]
                )
                for row in tuning:
                    row["phase_selected"] = bool(row["selected"])
                expanded = {
                    "arm": arm,
                    "parameters": parameters,
                    "weighting_scheme": weighting,
                    "loss_name": loss_name,
                    "loss_function": loss_function,
                    "search_phase": "expanded",
                    "selected_candidate_index": selected_index,
                    "inner_macro_mae": selected_score,
                    "outer_train_oof": stage1_oof,
                }
                if (
                    float(expanded["inner_macro_mae"])
                    < float(h1_selected[arm]["inner_macro_mae"])
                ):
                    h1_selected[arm] = expanded
                h1_tuning_by_arm[arm].extend(tuning)
                h1_weights_by_arm[arm].extend(weights)

        for weighting, loss_name, _ in h1_specs:
            arm = _h1_arm_name(weighting, loss_name)
            selection = h1_selected[arm]
            for row in h1_tuning_by_arm[arm]:
                row["selected"] = bool(
                    row["search_phase"] == selection["search_phase"]
                    and int(row["candidate_index"])
                    == int(selection["selected_candidate_index"])
                )
            tuning_rows.extend(h1_tuning_by_arm[arm])
            weight_rows.extend(h1_weights_by_arm[arm])
            model = _new_stage1(
                adapter.a2_columns,
                selection,
                random_state=_prediction_seed(model_seed, outer_fold, 109),
                n_jobs=n_jobs,
            )
            model.fit(outer_train, outer_target, outer_groups)
            predictions.loc[test, f"prediction_{arm}"] = model.predict(outer_test)
            weight_rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "inner_fold": None,
                    "arm": arm,
                    "search_phase": selection["search_phase"],
                    "scope": "outer_training_only",
                    **dict(model.fit_weight_audit or {}),
                }
            )

        stage1 = min(
            h1_selected.values(),
            key=lambda item: (float(item["inner_macro_mae"]), str(item["arm"])),
        )
        selected_stage1_rows.append(
            {
                "outer_fold": int(outer_fold),
                "selected_stage1_arm": stage1["arm"],
                "inner_macro_mae": float(stage1["inner_macro_mae"]),
                "weighting_scheme": stage1["weighting_scheme"],
                "loss_name": stage1["loss_name"],
                "search_phase": stage1["search_phase"],
                "selected_candidate_index": int(
                    stage1["selected_candidate_index"]
                ),
            }
        )
        stage1_test = predictions.loc[test, f"prediction_{stage1['arm']}"].to_numpy(
            dtype=float
        )
        outer_residual = outer_target - np.asarray(
            stage1["outer_train_oof"], dtype=float
        )
        residual_rows.append(
            {
                "outer_fold": int(outer_fold),
                "correction_inner_fold": None,
                "stage1_subfold": None,
                "fit_rows": int(len(outer_train)),
                "residual_rows_predicted": int(len(outer_train)),
                "residual_source": "selected_stage1_outer_training_oof_prediction",
                "in_sample_residuals": 0,
                "outer_test_labels_used": 0,
            }
        )
        h2_sources, nested_residual_audit = _prepare_h2_inner_sources(
            outer_train,
            outer_target,
            outer_groups,
            inner_splits,
            stage1,
            adapter.a2_columns,
            split_seed=split_seed,
            model_seed=model_seed,
            outer_fold=outer_fold,
            n_jobs=n_jobs,
        )
        residual_rows.extend(nested_residual_audit)
        correction_grid = _correction_grid(protocol, quick=quick)
        lambdas = tuple(float(value) for value in protocol["h2"]["lambda_grid"])
        shuffle_seed = int(protocol["h2"]["shuffle_seed"])
        view_specs = {
            "H2_E3_RAW_REAL": (adapter.e3_raw_columns, False),
            "H2_E3_RAW_SHUFFLED": (adapter.e3_raw_columns, True),
            "H2_E3_COMPACT_REAL": (adapter.e3_compact_columns, False),
            "H2_E3_COMPACT_SHUFFLED": (adapter.e3_compact_columns, True),
        }
        h2_selection: dict[str, dict[str, Any]] = {}
        raw_coefficients_by_arm: dict[str, list[list[np.ndarray]]] = {}
        for arm, (columns, shuffled) in view_specs.items():
            selection, tuning, shuffles, coefficient_by_config = _tune_h2_view(
                outer_train,
                outer_target,
                outer_groups,
                h2_sources,
                columns,
                correction_grid,
                lambdas,
                arm=arm,
                shuffled=shuffled,
                shuffle_seed=shuffle_seed,
                model_seed=model_seed,
                outer_fold=outer_fold,
                n_jobs=n_jobs,
            )
            tuning_rows.extend(tuning)
            shuffle_rows.extend(shuffles)
            h2_selection[arm] = selection
            if arm in {"H2_E3_RAW_REAL", "H2_E3_RAW_SHUFFLED"}:
                raw_coefficients_by_arm[arm] = coefficient_by_config
            correction_test, final_shuffle = _fit_h2_outer(
                outer_train,
                outer_test,
                outer_residual,
                outer_groups,
                columns,
                selection,
                arm=arm,
                shuffled=shuffled,
                shuffle_seed=shuffle_seed,
                model_seed=model_seed,
                outer_fold=outer_fold,
                n_jobs=n_jobs,
            )
            if final_shuffle is not None:
                shuffle_rows.append(final_shuffle)
            predictions.loc[test, f"prediction_{arm}"] = (
                stage1_test + float(selection["lambda"]) * correction_test
            )
            selected_lambda_rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "arm": arm,
                    "lambda": float(selection["lambda"]),
                    "correction_kind": selection["specification"]["kind"],
                    "correction_parameters": str(
                        selection["specification"]["parameters"]
                    ),
                    "inner_macro_mae": float(selection["inner_macro_mae"]),
                }
            )

        if set(raw_coefficients_by_arm) != {
            "H2_E3_RAW_REAL",
            "H2_E3_RAW_SHUFFLED",
        }:
            raise AssertionError("Real/shuffled raw-E3 stability coefficients are missing.")
        elastic_indices = [
            index
            for index, specification in enumerate(correction_grid)
            if specification["kind"] == "elasticnet"
        ]
        if not elastic_indices:
            raise AssertionError("Stability selection requires ElasticNet candidates.")
        stable_columns_by_arm: dict[str, tuple[str, ...]] = {}
        for source_arm in ("H2_E3_RAW_REAL", "H2_E3_RAW_SHUFFLED"):
            elastic_score = {
                index: min(
                    float(row["inner_macro_mae"])
                    for row in tuning_rows
                    if row.get("outer_fold") == outer_fold
                    and row.get("arm") == source_arm
                    and row.get("candidate_index") == index
                )
                for index in elastic_indices
            }
            selected_elastic = min(
                elastic_score, key=lambda index: (elastic_score[index], index)
            )
            selected_columns, rows = _stability_columns(
                adapter.e3_raw_columns,
                raw_coefficients_by_arm[source_arm][selected_elastic],
                threshold=float(
                    protocol["h2"]["stability_followup"][
                        "selection_frequency_threshold"
                    ]
                ),
            )
            stable_columns_by_arm[source_arm] = selected_columns
            for row in rows:
                stability_rows.append(
                    {
                        "selection_source_arm": source_arm,
                        "outer_fold": int(outer_fold),
                        "split_seed": int(split_seed),
                        **row,
                    }
                )
        for arm, shuffled in (
            ("H2_E3_STABLE_REAL", False),
            ("H2_E3_STABLE_SHUFFLED", True),
        ):
            stable_columns = stable_columns_by_arm[
                "H2_E3_RAW_SHUFFLED" if shuffled else "H2_E3_RAW_REAL"
            ]
            selection, tuning, shuffles, _ = _tune_h2_view(
                outer_train,
                outer_target,
                outer_groups,
                h2_sources,
                stable_columns,
                correction_grid,
                lambdas,
                arm=arm,
                shuffled=shuffled,
                shuffle_seed=shuffle_seed,
                model_seed=model_seed,
                outer_fold=outer_fold,
                n_jobs=n_jobs,
            )
            tuning_rows.extend(tuning)
            shuffle_rows.extend(shuffles)
            correction_test, final_shuffle = _fit_h2_outer(
                outer_train,
                outer_test,
                outer_residual,
                outer_groups,
                stable_columns,
                selection,
                arm=arm,
                shuffled=shuffled,
                shuffle_seed=shuffle_seed,
                model_seed=model_seed,
                outer_fold=outer_fold,
                n_jobs=n_jobs,
            )
            if final_shuffle is not None:
                shuffle_rows.append(final_shuffle)
            predictions.loc[test, f"prediction_{arm}"] = (
                stage1_test + float(selection["lambda"]) * correction_test
            )
            selected_lambda_rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "arm": arm,
                    "lambda": float(selection["lambda"]),
                    "correction_kind": selection["specification"]["kind"],
                    "correction_parameters": str(
                        selection["specification"]["parameters"]
                    ),
                    "selected_feature_count": int(len(stable_columns)),
                    "inner_macro_mae": float(selection["inner_macro_mae"]),
                }
            )

        h3_candidates = tuple(protocol["h3"]["search_candidates"])
        if quick:
            h3_candidates = h3_candidates[:1]
        h3_contracts = {
            "H3_A2": (adapter.h3_a2_a_columns, adapter.h3_a2_b_columns),
            "H3_A2_ELEC": (adapter.h3_elec_a_columns, adapter.h3_elec_b_columns),
        }
        for arm, (a_columns, b_columns) in h3_contracts.items():
            selected, tuning = _h3_tuning(
                outer_train,
                outer_target,
                outer_groups,
                inner_splits,
                arm=arm,
                a_columns=a_columns,
                b_columns=b_columns,
                candidates=h3_candidates,
                training=protocol["h3"]["training"],
                huber_delta=float(protocol["h3"]["lambda_delta"]),
                model_seed=model_seed,
                outer_fold=outer_fold,
                quick=quick,
            )
            tuning_rows.extend(tuning)
            model = fit_latent_difference_model(
                outer_train,
                outer_target,
                outer_groups,
                a_columns=a_columns,
                b_columns=b_columns,
                parameters=selected,
                model_seed=_prediction_seed(model_seed, outer_fold, 701),
                max_epochs=(
                    min(int(protocol["h3"]["training"]["max_epochs"]), 3)
                    if quick
                    else int(protocol["h3"]["training"]["max_epochs"])
                ),
                patience=(
                    min(
                        int(
                            protocol["h3"]["training"][
                                "early_stopping_patience"
                            ]
                        ),
                        2,
                    )
                    if quick
                    else int(
                        protocol["h3"]["training"]["early_stopping_patience"]
                    )
                ),
                batch_size=int(protocol["h3"]["training"]["batch_size"]),
                huber_delta=float(protocol["h3"]["lambda_delta"]),
            )
            h3_prediction = model.predict(outer_test)
            # Architecture assertions are checked on the exact scalar scores,
            # not on a separate unconstrained pair head (none exists).
            score_a, score_b = model.score_shoulders(outer_test)
            if not np.allclose(h3_prediction, score_a - score_b, atol=1e-7):
                raise AssertionError("H3 prediction is not exactly g(A)-g(B).")
            if not np.allclose(h3_prediction, -(score_b - score_a), atol=1e-7):
                raise AssertionError("H3 antisymmetry assertion failed.")
            predictions.loc[test, f"prediction_{arm}"] = h3_prediction

            shuffled_arm = f"{arm}_LABEL_SHUFFLED"
            shuffled_target, label_audit = _training_label_shuffle(
                outer_target,
                outer_train.reset_index(drop=True),
                seed=int(
                    protocol["h3"]["label_shuffle_seed"]
                    + outer_fold * 1_000_003
                ),
            )
            shuffle_rows.append(
                {
                    "arm": shuffled_arm,
                    "outer_fold": int(outer_fold),
                    "inner_fold": None,
                    "scope": "outer_training_labels_only",
                    "hyperparameters_selected_on": arm,
                    "shuffled_labels_used_for_hyperparameter_selection": False,
                    "shuffle_seed": int(
                        protocol["h3"]["label_shuffle_seed"]
                        + outer_fold * 1_000_003
                    ),
                    **label_audit,
                }
            )
            shuffled_model = fit_latent_difference_model(
                outer_train,
                shuffled_target,
                outer_groups,
                a_columns=a_columns,
                b_columns=b_columns,
                parameters=selected,
                model_seed=_prediction_seed(model_seed, outer_fold, 701),
                max_epochs=(
                    min(int(protocol["h3"]["training"]["max_epochs"]), 3)
                    if quick
                    else int(protocol["h3"]["training"]["max_epochs"])
                ),
                patience=(
                    min(
                        int(
                            protocol["h3"]["training"][
                                "early_stopping_patience"
                            ]
                        ),
                        2,
                    )
                    if quick
                    else int(
                        protocol["h3"]["training"]["early_stopping_patience"]
                    )
                ),
                batch_size=int(protocol["h3"]["training"]["batch_size"]),
                huber_delta=float(protocol["h3"]["lambda_delta"]),
            )
            predictions.loc[test, f"prediction_{shuffled_arm}"] = (
                shuffled_model.predict(outer_test)
            )

    prediction_columns = [f"prediction_{arm}" for arm in arms]
    if (predictions["outer_fold"] < 0).any():
        raise AssertionError("Some gen3 rows have no outer test fold.")
    if not np.isfinite(predictions[prediction_columns].to_numpy(dtype=float)).all():
        raise AssertionError("Gen3 OOF predictions are incomplete or non-finite.")
    metrics, per_extractant, family_metrics = arm_metric_table(
        predictions, arms, baseline_arm=BASELINE_ARM
    )
    comparisons = {
        f"{BASELINE_ARM}_vs_{arm}": (BASELINE_ARM, arm)
        for arm in arms
        if arm != BASELINE_ARM
    }
    for real, shuffled in (
        ("H2_E3_RAW_REAL", "H2_E3_RAW_SHUFFLED"),
        ("H2_E3_COMPACT_REAL", "H2_E3_COMPACT_SHUFFLED"),
        ("H2_E3_STABLE_REAL", "H2_E3_STABLE_SHUFFLED"),
        ("H3_A2", "H3_A2_LABEL_SHUFFLED"),
        ("H3_A2_ELEC", "H3_A2_ELEC_LABEL_SHUFFLED"),
    ):
        comparisons[f"{shuffled}_vs_{real}"] = (shuffled, real)
    bootstrap_replicates = int(protocol["statistics"]["cluster_bootstrap_replicates"])
    if quick:
        bootstrap_replicates = min(100, bootstrap_replicates)
    bootstrap = paired_extractant_bootstrap(
        predictions,
        comparisons,
        replicates=bootstrap_replicates,
        seed=int(protocol["statistics"]["bootstrap_seed"] + model_seed + split_seed),
    )
    leakage_audit = {
        "passed": True,
        "split_seed": int(split_seed),
        "model_seed": int(model_seed),
        "group_column": "extractant",
        "outer_folds": leakage_rows,
        "same_outer_folds_for_all_arms": True,
        "same_inner_folds_for_all_hyperparameter_selection": True,
        "preprocessing_fit_training_only": True,
        "weights_computed_fit_subset_only": True,
        "stage1_residuals_out_of_fold_only": bool(
            all(int(row.get("in_sample_residuals", 0)) == 0 for row in residual_rows)
        ),
        "lambda_selection_inner_only": True,
        "e3_selection_inner_only": True,
        "shuffle_training_only": True,
        "h3_label_shuffle_controls_reuse_real_hyperparameters": True,
        "outer_test_labels_used_for_selection": False,
    }
    return Gen3RunResult(
        predictions=predictions,
        split_assignments=pd.DataFrame(split_rows),
        inner_fold_assignments=pd.DataFrame(inner_assignment_rows),
        tuning_results=pd.DataFrame(tuning_rows),
        weight_audit=pd.DataFrame(weight_rows),
        residual_audit=pd.DataFrame(residual_rows),
        shuffle_audit=pd.DataFrame(shuffle_rows),
        e3_stability=pd.DataFrame(stability_rows),
        per_seed_metrics=metrics,
        per_extractant_metrics=per_extractant,
        family_metrics=family_metrics,
        paired_bootstrap=bootstrap,
        leakage_audit=leakage_audit,
        selected_stage1=pd.DataFrame(selected_stage1_rows),
        selected_lambdas=pd.DataFrame(selected_lambda_rows),
        run_metadata={
            "run_role": run_role,
            "split_seed": int(split_seed),
            "model_seed": int(model_seed),
            "quick": bool(quick),
            "arms": list(arms),
            "primary_metric": "equal_extractant_macro_mae",
            "pooled_metrics_row_weighted": True,
        },
    )
