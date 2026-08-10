"""Nested held-out evaluation for the guarded simplicial neural model."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as torch_functional
from sklearn.metrics import mean_absolute_error, r2_score

from .evaluation import (
    DEFAULT_PARAMETER_GRID,
    AntisymmetricExtraTreesRegressor,
    _group_folds,
    _select_delta3d_weight,
    _tune_parameters,
    group_balanced_weights,
    regression_metrics,
)
from .pairs import LANTHANIDE_Z, PAIR_TARGET_COLUMN, PairDataset
from .simplicial import (
    AntisymmetricSimplicialPairRegressor,
    VietorisRipsStore,
    set_torch_determinism,
)


DEFAULT_BLEND_WEIGHTS: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0)
MAX_BASE_SEED = 4_000_000_000


@dataclass(frozen=True)
class SimplicialTrainingConfig:
    hidden_dim: int = 64
    layers: int = 2
    dropout: float = 0.10
    rbf_count: int = 16
    max_filtration: float = 4.0
    learning_rate: float = 1e-3
    weight_decay: float = 3e-3
    epochs: int = 160
    initializations: int = 3
    max_pairs_per_batch: int = 64
    max_simplices_per_batch: int = 60_000
    gate_z: float = 0.50
    strict_determinism: bool = True
    # Declared encoder ladder; the default is the full 0/1/2-simplex network.
    simplex_order: str = "nodes_edges_triangles"
    # 'conditions_only' is the learned-geometry-only arm: no ligand 2D
    # descriptor reaches the model, so the encoder must carry the ligand signal.
    context_mode: str = "full"
    # Permutation seeds for the learned-geometry negative control. Empty runs
    # no control; each seed adds one arm trained on training-fold-permuted
    # geometry and evaluated on the untouched test geometry.
    geometry_null_seeds: tuple[int, ...] = ()


@dataclass(frozen=True)
class ContextScaler:
    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "ContextScaler":
        median = np.nanmedian(values, axis=0)
        median = np.nan_to_num(median, nan=0.0, posinf=0.0, neginf=0.0)
        filled = np.where(np.isfinite(values), values, median)
        mean = filled.mean(axis=0)
        scale = filled.std(axis=0)
        scale = np.where(scale > 1e-8, scale, 1.0)
        return cls(
            median=median.astype(np.float32),
            mean=mean.astype(np.float32),
            scale=scale.astype(np.float32),
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        filled = np.where(np.isfinite(values), values, self.median)
        return ((filled - self.mean) / self.scale).astype(np.float32)


@dataclass
class SimplicialData:
    frame: pd.DataFrame
    store: VietorisRipsStore
    context: np.ndarray
    target: np.ndarray
    groups: np.ndarray
    build_ids_a: np.ndarray
    build_ids_b: np.ndarray
    simplex_cost: np.ndarray


def permuted_geometry_view(
    data: SimplicialData,
    train_indices: np.ndarray,
    *,
    seed: int,
) -> tuple[SimplicialData, dict[str, Any]]:
    """Return a copy of ``data`` whose *training* geometry links are permuted.

    The negative control for the learned-geometry arm (STEP 10). The whole
    ``(complex_A, complex_B)`` assignment moves between training rows, so a
    geometry that appears in several rows stays internally consistent and the
    marginal distribution of structures, the input dimensionality and the model
    capacity are all preserved -- only the structure-to-target correspondence is
    destroyed.

    Two isolation rules are enforced here rather than assumed:

    * rows outside ``train_indices`` are never touched, so no held-out complex
      can reach training and no test row is ever re-labelled;
    * the permutation runs **within one lanthanide-pair label**, so the control
      keeps the metal contrast -- which is not the information under test -- and
      destroys only the ligand-geometry correspondence.
    """

    train_indices = np.asarray(train_indices, dtype=np.int64)
    if train_indices.size == 0:
        raise ValueError("A geometry null control needs a non-empty training set.")
    if len(np.unique(train_indices)) != len(train_indices):
        raise ValueError("train_indices must be unique.")
    build_ids_a = data.build_ids_a.copy()
    build_ids_b = data.build_ids_b.copy()
    simplex_cost = data.simplex_cost.copy()
    labels = data.frame["pair_label"].astype(str).to_numpy()
    generator = np.random.default_rng(int(seed))
    moved = 0
    for label in sorted(set(labels[train_indices].tolist())):
        block = train_indices[labels[train_indices] == label]
        if len(block) < 2:
            continue
        order = generator.permutation(len(block))
        source = block[order]
        build_ids_a[block] = data.build_ids_a[source]
        build_ids_b[block] = data.build_ids_b[source]
        simplex_cost[block] = data.simplex_cost[source]
        moved += int(np.count_nonzero(block != source))
    held_out = np.setdiff1d(
        np.arange(len(data.build_ids_a), dtype=np.int64), train_indices
    )
    if held_out.size:
        untouched = np.array_equal(
            build_ids_a[held_out], data.build_ids_a[held_out]
        ) and np.array_equal(build_ids_b[held_out], data.build_ids_b[held_out])
        if not untouched:
            raise AssertionError("Geometry permutation escaped the training subset.")
    audit = {
        "seed": int(seed),
        "train_rows": int(len(train_indices)),
        "test_rows_touched": 0,
        "rows_reassigned": moved,
        "permutation_unit": "(complex_A, complex_B) pair, within pair_label",
    }
    return (
        SimplicialData(
            frame=data.frame,
            store=data.store,
            context=data.context,
            target=data.target,
            groups=data.groups,
            build_ids_a=build_ids_a,
            build_ids_b=build_ids_b,
            simplex_cost=simplex_cost,
        ),
        audit,
    )


@dataclass
class FittedNetwork:
    model: AntisymmetricSimplicialPairRegressor
    context_scaler: ContextScaler
    target_scale: float
    best_epoch: int
    history: list[dict[str, Any]]


@dataclass
class SimplicialBenchmarkResult:
    summary: dict[str, Any]
    predictions: pd.DataFrame
    fold_assignments: pd.DataFrame
    tuning_results: pd.DataFrame
    training_history: pd.DataFrame
    per_extractant_metrics: pd.DataFrame
    per_pair_metrics: pd.DataFrame


CONTEXT_MODES: tuple[str, ...] = ("full", "conditions_only")


def symmetric_context_columns(
    pair_data: PairDataset, *, context_mode: str = "full"
) -> tuple[str, ...]:
    """Small even context block; ECFP identity remains in the 2D control only.

    ``conditions_only`` is the declared *learned-geometry-only* arm: it drops
    every ligand 2D descriptor and keeps the experimental conditions and the
    metal identity means, so nothing about the ligand reaches the model except
    through the encoded structure. Conditions cannot be dropped as well -- the
    target is a condition-matched difference, so an arm blind to them would be
    answering a different question, not a purer one.
    """

    if context_mode not in CONTEXT_MODES:
        raise ValueError(
            f"Unknown context_mode {context_mode!r}; expected one of {CONTEXT_MODES}."
        )
    metal_means = {"pair__Z_mean", "pair__ionic_radius_mean"}
    columns = [
        column
        for column in pair_data.baseline_columns
        if column in metal_means
        or (
            column.startswith("base__")
            and not column.startswith("base__ecfp_")
            and (context_mode == "full" or column.startswith("base__cond__"))
        )
    ]
    if not columns:
        raise ValueError("No symmetric non-ECFP context columns are available.")
    return tuple(columns)


def validate_vr_pair_links(frame: pd.DataFrame, store: VietorisRipsStore) -> None:
    """Verify that every pair points to the declared immutable VR graph."""

    required = {
        "geometry_feature_build_id_A",
        "geometry_feature_build_id_B",
        "vr_graph_index_A",
        "vr_graph_index_B",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Pair frame lacks VR provenance columns: {missing}")
    n_assets = len(store.build_ids)
    for side in ("A", "B"):
        indices = pd.to_numeric(frame[f"vr_graph_index_{side}"], errors="coerce")
        if indices.isna().any() or not np.equal(indices, np.floor(indices)).all():
            raise ValueError(f"vr_graph_index_{side} must contain finite integers.")
        integer_indices = indices.to_numpy(dtype=np.int64)
        if np.any(integer_indices < 0) or np.any(integer_indices >= n_assets):
            raise ValueError(f"vr_graph_index_{side} is outside the VR archive.")
        expected = frame[f"geometry_feature_build_id_{side}"].astype(str).to_numpy()
        observed = store.build_ids[integer_indices]
        mismatch = expected != observed
        if mismatch.any():
            examples = frame.loc[mismatch, "pair_id"].astype(str).head(5).tolist()
            raise ValueError(
                f"VR graph/build ID mismatch on side {side}; pair examples: {examples}"
            )
        metal_symbols = frame[f"metal_{side}"].astype(str).to_numpy()
        unknown_metals = sorted(set(metal_symbols) - set(LANTHANIDE_Z))
        if unknown_metals:
            raise ValueError(f"Unknown lanthanides on side {side}: {unknown_metals}")
        expected_atomic_numbers = np.asarray(
            [LANTHANIDE_Z[symbol] for symbol in metal_symbols], dtype=np.int64
        )
        observed_atomic_numbers = np.asarray(
            [
                int(store.graph(build_id).atomic_numbers[store.graph(build_id).metal_index])
                for build_id in expected
            ],
            dtype=np.int64,
        )
        metal_mismatch = expected_atomic_numbers != observed_atomic_numbers
        if metal_mismatch.any():
            examples = frame.loc[metal_mismatch, "pair_id"].astype(str).head(5).tolist()
            raise ValueError(
                f"VR metal atomic number mismatch on side {side}; pair examples: {examples}"
            )


def prepare_simplicial_data(
    pair_data: PairDataset,
    store: VietorisRipsStore,
    *,
    group_column: str,
    context_mode: str = "full",
) -> tuple[SimplicialData, tuple[str, ...]]:
    frame = pair_data.frame.reset_index(drop=True).copy()
    validate_vr_pair_links(frame, store)
    context_columns = symmetric_context_columns(pair_data, context_mode=context_mode)
    context = frame.loc[:, list(context_columns)].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float32)
    target = frame[PAIR_TARGET_COLUMN].to_numpy(dtype=np.float32)
    groups = frame[group_column].astype(str).to_numpy()
    build_ids_a = frame["geometry_feature_build_id_A"].astype(str).to_numpy()
    build_ids_b = frame["geometry_feature_build_id_B"].astype(str).to_numpy()
    store.warm_cache(np.concatenate((build_ids_a, build_ids_b)))
    costs = np.empty(len(frame), dtype=np.int64)
    for index, (build_a, build_b) in enumerate(zip(build_ids_a, build_ids_b, strict=True)):
        graph_a = store.graph(build_a)
        graph_b = store.graph(build_b)
        costs[index] = max(
            1,
            graph_a.edge_index.shape[1]
            + graph_b.edge_index.shape[1]
            + graph_a.triangle_index.shape[1]
            + graph_b.triangle_index.shape[1],
        )
    return (
        SimplicialData(
            frame=frame,
            store=store,
            context=context,
            target=target,
            groups=groups,
            build_ids_a=build_ids_a,
            build_ids_b=build_ids_b,
            simplex_cost=costs,
        ),
        context_columns,
    )


def _batch_indices(
    indices: np.ndarray,
    costs: np.ndarray,
    *,
    max_pairs: int,
    max_simplices: int,
    shuffle: bool,
    seed: int,
) -> list[np.ndarray]:
    order = np.asarray(indices, dtype=np.int64).copy()
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    batches: list[np.ndarray] = []
    current: list[int] = []
    current_cost = 0
    for raw_index in order:
        index = int(raw_index)
        cost = int(costs[index])
        if cost > int(max_simplices):
            raise ValueError(
                f"Pair index {index} requires {cost} simplices, exceeding "
                f"max_simplices={int(max_simplices)}. Increase the limit explicitly."
            )
        if current and (
            len(current) >= int(max_pairs)
            or current_cost + cost > int(max_simplices)
        ):
            batches.append(np.asarray(current, dtype=np.int64))
            current = []
            current_cost = 0
        current.append(index)
        current_cost += cost
    if current:
        batches.append(np.asarray(current, dtype=np.int64))
    return batches


def _pack_pair_batch(
    data: SimplicialData,
    indices: np.ndarray,
    transformed_context: np.ndarray,
    device: torch.device,
) -> tuple[Any, torch.Tensor]:
    identifiers = [*data.build_ids_a[indices].tolist(), *data.build_ids_b[indices].tolist()]
    complexes = data.store.pack(identifiers).to(device)
    context = torch.from_numpy(transformed_context[indices]).to(device)
    return complexes, context


def _new_model(
    context_dim: int,
    config: SimplicialTrainingConfig,
    device: torch.device,
) -> AntisymmetricSimplicialPairRegressor:
    return AntisymmetricSimplicialPairRegressor(
        context_dim=context_dim,
        hidden_dim=config.hidden_dim,
        layers=config.layers,
        dropout=config.dropout,
        rbf_count=config.rbf_count,
        max_filtration=config.max_filtration,
        simplex_order=config.simplex_order,
    ).to(device)


def _predict_network(
    fitted: FittedNetwork,
    data: SimplicialData,
    indices: np.ndarray,
    *,
    config: SimplicialTrainingConfig,
    device: torch.device,
) -> np.ndarray:
    transformed = fitted.context_scaler.transform(data.context)
    predictions = np.empty(len(indices), dtype=np.float64)
    position = {int(index): offset for offset, index in enumerate(indices.tolist())}
    fitted.model.eval()
    with torch.no_grad():
        for batch_indices in _batch_indices(
            indices,
            data.simplex_cost,
            max_pairs=config.max_pairs_per_batch,
            max_simplices=config.max_simplices_per_batch,
            shuffle=False,
            seed=0,
        ):
            complexes, context = _pack_pair_batch(
                data, batch_indices, transformed, device
            )
            batch_prediction = (
                fitted.model(complexes, context).detach().cpu().numpy()
                * fitted.target_scale
            )
            for index, value in zip(batch_indices, batch_prediction, strict=True):
                predictions[position[int(index)]] = float(value)
    return predictions


def _fit_network_fixed_epochs(
    data: SimplicialData,
    train_indices: np.ndarray,
    *,
    config: SimplicialTrainingConfig,
    epochs: int,
    model_seed: int,
    device: torch.device,
) -> FittedNetwork:
    set_torch_determinism(model_seed, strict=config.strict_determinism)
    scaler = ContextScaler.fit(data.context[train_indices])
    transformed = scaler.transform(data.context)
    train_group_weights = group_balanced_weights(data.groups[train_indices]).astype(
        np.float32
    )
    weighted_square = np.average(
        np.square(data.target[train_indices]), weights=train_group_weights
    )
    target_scale = max(float(math.sqrt(weighted_square)), 1e-4)
    model = _new_model(transformed.shape[1], config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    weight_lookup = {
        int(index): float(weight)
        for index, weight in zip(train_indices, train_group_weights, strict=True)
    }
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        losses: list[float] = []
        train_batches = _batch_indices(
            train_indices,
            data.simplex_cost,
            max_pairs=config.max_pairs_per_batch,
            max_simplices=config.max_simplices_per_batch,
            shuffle=True,
            seed=model_seed + epoch * 1009,
        )
        # One denominator for the whole epoch keeps the inverse-frequency
        # objective independent of the accidental group mixture in a batch.
        loss_denominator = max(
            float(train_group_weights.sum()) / len(train_batches), 1e-8
        )
        for batch_indices in train_batches:
            complexes, context = _pack_pair_batch(data, batch_indices, transformed, device)
            target = torch.from_numpy(data.target[batch_indices] / target_scale).to(device)
            weights = torch.tensor(
                [weight_lookup[int(index)] for index in batch_indices],
                dtype=torch.float32,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            prediction = model(complexes, context)
            row_loss = torch_functional.smooth_l1_loss(
                prediction, target, reduction="none", beta=0.5
            )
            loss = (row_loss * weights).sum() / loss_denominator
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"Non-finite neural loss at epoch {epoch}, seed {model_seed}."
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(
            {
                "epoch": epoch,
                "model_seed": int(model_seed),
                "train_loss": float(np.mean(losses)),
                "validation_macro_group_mae": np.nan,
                "validation_r2": np.nan,
                "validation_group_balanced_r2": np.nan,
            }
        )
    return FittedNetwork(
        model=model,
        context_scaler=scaler,
        target_scale=target_scale,
        best_epoch=int(epochs),
        history=history,
    )


def _guarded_blend_weight(
    target: np.ndarray,
    groups: np.ndarray,
    reference_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    *,
    gate_z: float,
    weights: tuple[float, ...] = DEFAULT_BLEND_WEIGHTS,
) -> tuple[float, list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for weight in weights:
        blended = reference_prediction + float(weight) * (
            candidate_prediction - reference_prediction
        )
        metrics = regression_metrics(target, blended, groups)
        rows.append(
            {
                "simplicial_weight": float(weight),
                "inner_macro_group_mae": float(metrics["macro_group_mae"]),
                "inner_r2": float(metrics["r2"]),
                "inner_group_balanced_r2": float(metrics["group_balanced_r2"]),
            }
        )
    raw_best = min(
        rows,
        key=lambda row: (
            float(row["inner_macro_group_mae"]),
            -float(row["inner_group_balanced_r2"]),
            float(row["simplicial_weight"]),
        ),
    )
    raw_weight = float(raw_best["simplicial_weight"])
    raw_prediction = reference_prediction + raw_weight * (
        candidate_prediction - reference_prediction
    )
    group_differences: list[float] = []
    for group in pd.Series(groups).drop_duplicates().tolist():
        mask = groups == group
        reference_mae = np.mean(np.abs(target[mask] - reference_prediction[mask]))
        candidate_mae = np.mean(np.abs(target[mask] - raw_prediction[mask]))
        group_differences.append(float(reference_mae - candidate_mae))
    mean_difference = float(np.mean(group_differences))
    standard_error = (
        float(np.std(group_differences, ddof=1) / math.sqrt(len(group_differences)))
        if len(group_differences) >= 2
        else float("inf")
    )
    lower_gate = mean_difference - float(gate_z) * standard_error
    gate_passed = raw_weight == 0.0 or lower_gate > 0.0
    selected_weight = raw_weight if gate_passed else 0.0
    for row in rows:
        row["raw_selected"] = bool(row is raw_best)
        row["selected_after_gate"] = bool(
            float(row["simplicial_weight"]) == selected_weight
        )
    gate = {
        "raw_selected_weight": raw_weight,
        "selected_weight": selected_weight,
        "gate_z": float(gate_z),
        "n_inner_groups": len(group_differences),
        "mean_group_mae_reduction": mean_difference,
        "standard_error": standard_error,
        "lower_gate_bound": lower_gate,
        "passed": bool(gate_passed),
    }
    return selected_weight, rows, gate


def _primary_bootstrap(
    predictions: pd.DataFrame,
    *,
    group_column: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    unique_groups = predictions[group_column].drop_duplicates().to_numpy()
    group_indices = {
        group: predictions.index[predictions[group_column].eq(group)].to_numpy()
        for group in unique_groups
    }
    draws = {
        metric: []
        for metric in (
            "r2_gain",
            "group_balanced_r2_gain",
            "mae_reduction",
            "macro_group_mae_reduction",
            "sign_accuracy_gain",
        )
    }
    for _ in range(int(n_bootstrap)):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled_indices = np.concatenate([group_indices[group] for group in sampled_groups])
        sample = predictions.loc[sampled_indices]
        truth = sample[PAIR_TARGET_COLUMN].to_numpy(dtype=float)
        reference = sample["prediction_delta3d"].to_numpy(dtype=float)
        candidate = sample["prediction_simplicial"].to_numpy(dtype=float)
        if np.var(truth) <= 0.0:
            continue
        weights = np.concatenate(
            [
                np.full(len(group_indices[group]), 1.0 / len(group_indices[group]))
                for group in sampled_groups
            ]
        )
        draws["r2_gain"].append(
            float(r2_score(truth, candidate) - r2_score(truth, reference))
        )
        draws["group_balanced_r2_gain"].append(
            float(
                r2_score(truth, candidate, sample_weight=weights)
                - r2_score(truth, reference, sample_weight=weights)
            )
        )
        draws["mae_reduction"].append(
            float(
                mean_absolute_error(truth, reference)
                - mean_absolute_error(truth, candidate)
            )
        )
        draws["sign_accuracy_gain"].append(
            float(
                np.mean(np.sign(truth) == np.sign(candidate))
                - np.mean(np.sign(truth) == np.sign(reference))
            )
        )
        macro_differences = []
        for group in sampled_groups:
            group_frame = predictions.loc[group_indices[group]]
            group_truth = group_frame[PAIR_TARGET_COLUMN].to_numpy(dtype=float)
            macro_differences.append(
                mean_absolute_error(
                    group_truth, group_frame["prediction_delta3d"].to_numpy(dtype=float)
                )
                - mean_absolute_error(
                    group_truth,
                    group_frame["prediction_simplicial"].to_numpy(dtype=float),
                )
            )
        draws["macro_group_mae_reduction"].append(float(np.mean(macro_differences)))

    def interval(values: list[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=float)
        if not len(array):
            raise ValueError("Bootstrap produced zero valid resamples.")
        return {
            "mean": float(array.mean()),
            "ci95_low": float(np.quantile(array, 0.025)),
            "ci95_high": float(np.quantile(array, 0.975)),
        }

    return {
        "method": "paired cluster bootstrap over fixed OOF held-out-group predictions",
        "reference_model": "adaptive tabular Delta3D",
        "candidate_model": "guarded simplicial hybrid",
        "scope_limitation": (
            "Does not refit nested CV; split/model variability is measured by the "
            "prespecified multi-seed SLURM array."
        ),
        "replicates": len(draws["r2_gain"]),
        "comparisons": {
            "simplicial_vs_delta3d": {
                metric: interval(values) for metric, values in draws.items()
            }
        },
    }


def _group_metric_table(
    predictions: pd.DataFrame, group_column: str, label: str
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group, group_frame in predictions.groupby(group_column, dropna=False):
        truth = group_frame[PAIR_TARGET_COLUMN].to_numpy(dtype=float)
        row: dict[str, Any] = {group_column: group, "n_rows": len(group_frame)}
        for model_name in (
            "baseline",
            "2d_ensemble",
            "delta3d_unshrunk",
            "delta3d",
            "simplicial_unshrunk",
            "simplicial",
        ):
            prediction = group_frame[f"prediction_{model_name}"].to_numpy(dtype=float)
            row[f"{model_name}_mae"] = float(np.mean(np.abs(truth - prediction)))
            row[f"{model_name}_sign_accuracy"] = float(
                np.mean(np.sign(truth) == np.sign(prediction))
            )
        row["mae_reduction_simplicial_vs_delta3d"] = (
            row["delta3d_mae"] - row["simplicial_mae"]
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    result.insert(0, "grouping", label)
    return result.sort_values("n_rows", ascending=False, ignore_index=True)


def nested_simplicial_benchmark(
    pair_data: PairDataset,
    store: VietorisRipsStore,
    *,
    group_column: str = "ecfp_exact_cluster",
    outer_folds: int = 5,
    inner_folds: int = 3,
    n_estimators: int = 200,
    n_jobs: int = -1,
    n_bootstrap: int = 2000,
    seed: int = 42,
    split_seed: int = 42,
    device: torch.device,
    training_config: SimplicialTrainingConfig = SimplicialTrainingConfig(),
    parameter_grid: tuple[dict[str, Any], ...] = DEFAULT_PARAMETER_GRID,
) -> SimplicialBenchmarkResult:
    for label, value in (("seed", seed), ("split_seed", split_seed)):
        if int(value) < 0 or int(value) > MAX_BASE_SEED:
            raise ValueError(f"{label} must be between 0 and {MAX_BASE_SEED}.")
    data, context_columns = prepare_simplicial_data(
        pair_data,
        store,
        group_column=group_column,
        context_mode=training_config.context_mode,
    )
    frame = data.frame
    groups = data.groups
    target = data.target.astype(float)
    strata = frame["pair_label"].astype(str).to_numpy()
    outer_split_list = _group_folds(
        groups,
        strata,
        outer_folds,
        split_seed=split_seed,
        allow_fewer_splits=False,
    )
    predictions = frame[
        [
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
            "geometry_feature_build_id_A",
            "geometry_feature_build_id_B",
            "vr_graph_index_A",
            "vr_graph_index_B",
            PAIR_TARGET_COLUMN,
        ]
    ].copy()
    predictions["outer_fold"] = -1
    geometry_null_seeds = tuple(
        int(value) for value in training_config.geometry_null_seeds
    )
    if len(set(geometry_null_seeds)) != len(geometry_null_seeds):
        raise ValueError("geometry_null_seeds must be unique.")
    if any(value < 0 for value in geometry_null_seeds):
        raise ValueError("geometry_null_seeds must be nonnegative.")
    null_arm_names = tuple(
        f"simplicial_shuffled_s{value}" for value in geometry_null_seeds
    )
    model_names = (
        "baseline",
        "2d_second_unshrunk",
        "2d_ensemble",
        "delta3d_unshrunk",
        "delta3d",
        "simplicial_unshrunk",
        "simplicial",
    ) + null_arm_names
    for model_name in model_names:
        predictions[f"prediction_{model_name}"] = np.nan
    geometry_null_audit: list[dict[str, Any]] = []

    leakage_rows: list[dict[str, Any]] = []
    all_pair_types = set(frame["pair_label"].astype(str))
    tuning_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []

    for outer_fold, (train_indices, test_indices) in enumerate(outer_split_list):
        train_frame = frame.iloc[train_indices]
        test_frame = frame.iloc[test_indices]
        train_pair_types = set(train_frame["pair_label"].astype(str))
        test_pair_types = set(test_frame["pair_label"].astype(str))

        def identity_overlap(columns: tuple[str, ...]) -> int:
            train_values = set(
                train_frame.loc[:, list(columns)].astype(str).to_numpy().ravel().tolist()
            )
            test_values = set(
                test_frame.loc[:, list(columns)].astype(str).to_numpy().ravel().tolist()
            )
            return len(train_values & test_values)

        leakage = {
            "outer_fold": outer_fold,
            "outer_split_seed": int(split_seed),
            "train_rows": len(train_indices),
            "test_rows": len(test_indices),
            "train_pair_types": len(train_pair_types),
            "test_pair_types": len(test_pair_types),
            "pair_types_missing_from_train": sorted(all_pair_types - train_pair_types),
            "pair_types_missing_from_test": sorted(all_pair_types - test_pair_types),
            "group_overlap": identity_overlap((group_column,)),
            "extractant_overlap": identity_overlap(("extractant",)),
            "ecfp_exact_cluster_overlap": identity_overlap(("ecfp_exact_cluster",)),
            "source_id_overlap": identity_overlap(("source_id_A", "source_id_B")),
            "geometry_key_overlap": identity_overlap(("geometry_key_A", "geometry_key_B")),
            "geometry_feature_build_id_overlap": identity_overlap(
                ("geometry_feature_build_id_A", "geometry_feature_build_id_B")
            ),
            "vr_graph_index_overlap": identity_overlap(
                ("vr_graph_index_A", "vr_graph_index_B")
            ),
        }
        leakage_rows.append(leakage)
        direct_leakage_keys = (
            "group_overlap",
            "extractant_overlap",
            "ecfp_exact_cluster_overlap",
            "source_id_overlap",
            "geometry_key_overlap",
            "geometry_feature_build_id_overlap",
            "vr_graph_index_overlap",
        )
        if any(leakage[key] for key in direct_leakage_keys):
            raise AssertionError(f"Outer-fold leakage detected: {leakage}")

        outer_train_frame = frame.iloc[train_indices]
        outer_target = target[train_indices]
        outer_groups = groups[train_indices]
        inner_split_seed = split_seed + outer_fold * 10_007
        baseline_parameters, baseline_tuning, baseline_inner = _tune_parameters(
            outer_train_frame,
            outer_target,
            outer_groups,
            pair_data.baseline_columns,
            parameter_grid=parameter_grid,
            inner_folds=inner_folds,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            model_seed=seed + outer_fold * 1009,
            split_seed=inner_split_seed,
            context=f"outer_{outer_fold}:baseline",
        )
        second_parameters, second_tuning, second_inner = _tune_parameters(
            outer_train_frame,
            outer_target,
            outer_groups,
            pair_data.baseline_columns,
            parameter_grid=parameter_grid,
            inner_folds=inner_folds,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            model_seed=seed + outer_fold * 1009 + 100_003,
            split_seed=inner_split_seed,
            context=f"outer_{outer_fold}:2d_second_unshrunk",
        )
        delta3d_parameters, delta3d_tuning, delta3d_inner_unshrunk = _tune_parameters(
            outer_train_frame,
            outer_target,
            outer_groups,
            pair_data.full_columns,
            parameter_grid=parameter_grid,
            inner_folds=inner_folds,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            model_seed=seed + outer_fold * 1009 + 100_003,
            split_seed=inner_split_seed,
            context=f"outer_{outer_fold}:delta3d_unshrunk",
        )
        tuning_rows.extend(baseline_tuning)
        tuning_rows.extend(second_tuning)
        tuning_rows.extend(delta3d_tuning)
        ensemble_weight, ensemble_weight_rows = _select_delta3d_weight(
            outer_target,
            outer_groups,
            baseline_inner,
            second_inner,
        )
        for row in ensemble_weight_rows:
            tuning_rows.append(
                {
                    "context": f"outer_{outer_fold}:2d_ensemble_weight",
                    "2d_ensemble_weight": row["delta3d_weight"],
                    **{key: value for key, value in row.items() if key != "delta3d_weight"},
                }
            )
        ensemble_inner = baseline_inner + ensemble_weight * (second_inner - baseline_inner)
        delta3d_weight, delta3d_weight_rows = _select_delta3d_weight(
            outer_target,
            outer_groups,
            baseline_inner,
            delta3d_inner_unshrunk,
        )
        for row in delta3d_weight_rows:
            tuning_rows.append(
                {"context": f"outer_{outer_fold}:delta3d_weight", **row}
            )
        delta3d_inner = baseline_inner + delta3d_weight * (
            delta3d_inner_unshrunk - baseline_inner
        )

        inner_split_list = _group_folds(
            outer_groups,
            outer_train_frame["pair_label"].astype(str).to_numpy(),
            inner_folds,
            split_seed=inner_split_seed,
            allow_fewer_splits=True,
        )
        simplicial_inner = np.full(len(train_indices), np.nan, dtype=float)
        for inner_fold, (inner_train_local, inner_validation_local) in enumerate(
            inner_split_list
        ):
            inner_train_global = train_indices[inner_train_local]
            inner_validation_global = train_indices[inner_validation_local]
            initialization_predictions: list[np.ndarray] = []
            for initialization in range(training_config.initializations):
                network_seed = (
                    seed
                    + outer_fold * 1_000_003
                    + inner_fold * 10_009
                    + initialization * 101
                )
                fitted = _fit_network_fixed_epochs(
                    data,
                    inner_train_global,
                    config=training_config,
                    epochs=training_config.epochs,
                    model_seed=network_seed,
                    device=device,
                )
                initialization_predictions.append(
                    _predict_network(
                        fitted,
                        data,
                        inner_validation_global,
                        config=training_config,
                        device=device,
                    )
                )
                for row in fitted.history:
                    history_rows.append(
                        {
                            "context": "inner_fixed_fit",
                            "outer_fold": outer_fold,
                            "inner_fold": inner_fold,
                            "initialization": initialization,
                            **row,
                        }
                    )
                del fitted
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            simplicial_inner[inner_validation_local] = np.mean(
                initialization_predictions, axis=0
            )
        if not np.isfinite(simplicial_inner).all():
            raise AssertionError("Simplicial inner OOF predictions are incomplete/non-finite.")

        simplicial_weight, simplicial_weight_rows, gate = _guarded_blend_weight(
            outer_target,
            outer_groups,
            delta3d_inner,
            simplicial_inner,
            gate_z=training_config.gate_z,
        )
        for row in simplicial_weight_rows:
            tuning_rows.append(
                {"context": f"outer_{outer_fold}:simplicial_weight", **row}
            )
        final_epochs = int(training_config.epochs)
        selected_rows.append(
            {
                "outer_fold": outer_fold,
                "baseline": baseline_parameters,
                "2d_second_unshrunk": second_parameters,
                "2d_ensemble_weight": ensemble_weight,
                "delta3d_unshrunk": delta3d_parameters,
                "delta3d_weight": delta3d_weight,
                "simplicial_weight": simplicial_weight,
                "simplicial_gate": gate,
                "simplicial_final_epochs": final_epochs,
            }
        )

        baseline_model = AntisymmetricExtraTreesRegressor(
            pair_data.baseline_columns,
            n_estimators=n_estimators,
            max_features=baseline_parameters["max_features"],
            min_samples_leaf=baseline_parameters["min_samples_leaf"],
            random_state=seed + outer_fold * 101 + 7,
            n_jobs=n_jobs,
        )
        second_model = AntisymmetricExtraTreesRegressor(
            pair_data.baseline_columns,
            n_estimators=n_estimators,
            max_features=second_parameters["max_features"],
            min_samples_leaf=second_parameters["min_samples_leaf"],
            random_state=seed + outer_fold * 101 + 19,
            n_jobs=n_jobs,
        )
        delta3d_model = AntisymmetricExtraTreesRegressor(
            pair_data.full_columns,
            n_estimators=n_estimators,
            max_features=delta3d_parameters["max_features"],
            min_samples_leaf=delta3d_parameters["min_samples_leaf"],
            random_state=seed + outer_fold * 101 + 19,
            n_jobs=n_jobs,
        )
        baseline_model.fit(outer_train_frame, outer_target, outer_groups)
        second_model.fit(outer_train_frame, outer_target, outer_groups)
        delta3d_model.fit(outer_train_frame, outer_target, outer_groups)
        baseline_test = baseline_model.predict(frame.iloc[test_indices])
        second_test = second_model.predict(frame.iloc[test_indices])
        delta3d_test_unshrunk = delta3d_model.predict(frame.iloc[test_indices])
        ensemble_test = baseline_test + ensemble_weight * (second_test - baseline_test)
        delta3d_test = baseline_test + delta3d_weight * (
            delta3d_test_unshrunk - baseline_test
        )

        final_initialization_predictions: list[np.ndarray] = []
        for initialization in range(training_config.initializations):
            network_seed = seed + outer_fold * 1_000_003 + 700_001 + initialization * 101
            fitted = _fit_network_fixed_epochs(
                data,
                train_indices,
                config=training_config,
                epochs=final_epochs,
                model_seed=network_seed,
                device=device,
            )
            final_initialization_predictions.append(
                _predict_network(
                    fitted,
                    data,
                    test_indices,
                    config=training_config,
                    device=device,
                )
            )
            for row in fitted.history:
                history_rows.append(
                    {
                        "context": "outer_final_fit",
                        "outer_fold": outer_fold,
                        "inner_fold": -1,
                        "initialization": initialization,
                        **row,
                    }
                )
            del fitted
            if device.type == "cuda":
                torch.cuda.empty_cache()
        simplicial_test = np.mean(final_initialization_predictions, axis=0)
        guarded_test = delta3d_test + simplicial_weight * (
            simplicial_test - delta3d_test
        )

        # Learned-geometry negative control.  Identical architecture, identical
        # epochs, identical initialisation count and identical held-out rows;
        # only the training-fold structure-to-target correspondence is gone.
        # The comparison that matters is against ``simplicial_unshrunk``, which
        # is the same quantity without the permutation.
        null_test: dict[str, np.ndarray] = {}
        for null_seed, arm_name in zip(
            geometry_null_seeds, null_arm_names, strict=True
        ):
            null_data, null_audit = permuted_geometry_view(
                data, train_indices, seed=null_seed + outer_fold * 7919
            )
            null_predictions: list[np.ndarray] = []
            for initialization in range(training_config.initializations):
                network_seed = (
                    seed
                    + outer_fold * 1_000_003
                    + 800_011
                    + null_seed * 13
                    + initialization * 101
                )
                fitted = _fit_network_fixed_epochs(
                    null_data,
                    train_indices,
                    config=training_config,
                    epochs=final_epochs,
                    model_seed=network_seed,
                    device=device,
                )
                # Test rows keep their true geometry in ``null_data``, so this
                # scores the permuted-training model on the untouched cohort.
                null_predictions.append(
                    _predict_network(
                        fitted,
                        null_data,
                        test_indices,
                        config=training_config,
                        device=device,
                    )
                )
                del fitted
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            null_test[arm_name] = np.mean(null_predictions, axis=0)
            geometry_null_audit.append(
                {"outer_fold": outer_fold, "arm": arm_name, **null_audit}
            )
        predictions.loc[test_indices, "outer_fold"] = outer_fold
        predictions.loc[test_indices, "prediction_baseline"] = baseline_test
        predictions.loc[test_indices, "prediction_2d_second_unshrunk"] = second_test
        predictions.loc[test_indices, "prediction_2d_ensemble"] = ensemble_test
        predictions.loc[test_indices, "prediction_delta3d_unshrunk"] = delta3d_test_unshrunk
        predictions.loc[test_indices, "prediction_delta3d"] = delta3d_test
        predictions.loc[test_indices, "prediction_simplicial_unshrunk"] = simplicial_test
        predictions.loc[test_indices, "prediction_simplicial"] = guarded_test
        for arm_name, values in null_test.items():
            predictions.loc[test_indices, f"prediction_{arm_name}"] = values

    prediction_columns = [f"prediction_{name}" for name in model_names]
    prediction_values = predictions[prediction_columns].to_numpy(dtype=float)
    if not np.isfinite(prediction_values).all():
        raise AssertionError("Outer OOF predictions are incomplete or non-finite.")

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
        "primary_simplicial_vs_delta3d": comparison("delta3d", "simplicial"),
        "simplicial_vs_2d_ensemble": comparison("2d_ensemble", "simplicial"),
        "delta3d_vs_2d_ensemble": comparison("2d_ensemble", "delta3d"),
        "simplicial_unshrunk_vs_delta3d": comparison(
            "delta3d", "simplicial_unshrunk"
        ),
        "simplicial_unshrunk_vs_2d_ensemble": comparison(
            "2d_ensemble", "simplicial_unshrunk"
        ),
        "simplicial_vs_baseline": comparison("baseline", "simplicial"),
        "2d_ensemble_vs_baseline": comparison("baseline", "2d_ensemble"),
    }
    # Real learned geometry against its own permuted twin, and the twin against
    # the same 2D reference, so both gains are read on one scale.
    for arm_name in null_arm_names:
        improvements[f"simplicial_unshrunk_vs_{arm_name}"] = comparison(
            arm_name, "simplicial_unshrunk"
        )
        improvements[f"{arm_name}_vs_baseline"] = comparison("baseline", arm_name)
    bootstrap = _primary_bootstrap(
        predictions,
        group_column=group_column,
        n_bootstrap=n_bootstrap,
        seed=seed + 999_983,
    )
    leakage_passed = all(
        row["group_overlap"] == 0
        and row["extractant_overlap"] == 0
        and row["ecfp_exact_cluster_overlap"] == 0
        and row["source_id_overlap"] == 0
        and row["geometry_key_overlap"] == 0
        and row["geometry_feature_build_id_overlap"] == 0
        and row["vr_graph_index_overlap"] == 0
        for row in leakage_rows
    )
    summary = {
        "protocol": {
            "pair_scope": pair_data.audit.get("pair_scope", "adjacent"),
            "model_family": (
                "guarded hybrid of tabular Delta3D and metal-centred "
                "0/1/2-simplex Siamese neural network"
            ),
            "outer_split": (
                f"{len(outer_split_list)}-fold seeded shuffled StratifiedGroupKFold; "
                "stratified by pair_label"
            ),
            "inner_split": (
                f"up to {inner_folds}-fold seeded shuffled StratifiedGroupKFold; "
                "shared by 2D, tabular Delta3D and simplicial branches"
            ),
            "group_column": group_column,
            "preprocessing_scope": "2D imputation and neural context scaling fit per fold",
            "training_weights": (
                "inverse held-in group frequency with one batch-independent "
                "normalizer per epoch"
            ),
            "antisymmetric_inference": "(head(zA-zB,c) - head(zB-zA,c)) / 2",
            "primary_comparison": (
                "guarded simplicial hybrid minus adaptive tabular Delta3D on identical folds"
            ),
            "guard": (
                "Epoch count is fixed before CV. Inner group CV selects w in "
                "{0,.25,.5,.75,1}; nonzero w is retained only when mean group-MAE "
                "reduction minus gate_z*SE is positive."
            ),
            "trees_per_tabular_fit": int(n_estimators),
            "model_seed": int(seed),
            "split_seed": int(split_seed),
            "fit_final_model": False,
            "device_type": device.type,
            "strict_determinism": training_config.strict_determinism,
            "simplex_order": training_config.simplex_order,
            "context_mode": training_config.context_mode,
            "geometry_null_seeds": list(geometry_null_seeds),
            "geometry_null_scope": (
                "training rows only, whole (complex_A, complex_B) assignment "
                "permuted within one pair_label; test geometry untouched"
            ),
        },
        "geometry_null_audit": geometry_null_audit,
        "feature_counts": {
            "baseline": len(pair_data.baseline_columns),
            "tabular_delta3d": len(pair_data.full_columns),
            "simplicial_context": len(context_columns),
            "simplicial_context_columns": list(context_columns),
        },
        "simplicial_configuration": {
            **training_config.__dict__,
            "shell_mode": store.shell_mode,
            "radius_angstrom": store.radius_angstrom,
            "max_edges": store.max_edges,
            "max_triangles": store.max_triangles,
            "use_partial_charges": store.use_partial_charges,
        },
        "metrics": metrics,
        "improvements": improvements,
        "paired_group_bootstrap": bootstrap,
        "leakage_audit": {"passed": leakage_passed, "outer_folds": leakage_rows},
        "selected_parameters_by_outer_fold": selected_rows,
        "interpretation_guardrail": (
            "The w=0 path guarantees exact fallback to adaptive tabular Delta3D for folds "
            "where inner evidence is weak. Improvement on unseen outer groups remains an "
            "empirical claim that must be established by the prespecified cluster runs."
        ),
    }
    fold_assignments = predictions[
        ["pair_id", "extractant", "ecfp_exact_cluster", "outer_fold"]
    ].copy()
    fold_assignments["pair_scope"] = pair_data.audit.get("pair_scope", "adjacent")
    fold_assignments["outer_split_seed"] = int(split_seed)
    return SimplicialBenchmarkResult(
        summary=summary,
        predictions=predictions,
        fold_assignments=fold_assignments,
        tuning_results=pd.DataFrame(tuning_rows),
        training_history=pd.DataFrame(history_rows),
        per_extractant_metrics=_group_metric_table(
            predictions, "extractant", "extractant"
        ),
        per_pair_metrics=_group_metric_table(predictions, "pair_label", "pair_type"),
    )
