"""Models used only by the additive generation-3 experiment layer."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .evaluation import _as_float_frame, _reversal_input
from .pairs import reverse_pair_features


WEIGHTING_SCHEMES: tuple[str, ...] = ("none", "group_sqrt", "group_equal")


def fold_local_group_weights(groups: Iterable[Any], scheme: str) -> np.ndarray:
    """Return mean-one weights using counts from the supplied fit rows only."""

    group_array = np.asarray(list(groups), dtype=object)
    if not len(group_array):
        raise ValueError("Cannot weight an empty training fold.")
    if scheme not in WEIGHTING_SCHEMES:
        raise ValueError(f"Unknown weighting scheme {scheme!r}.")
    counts = Counter(group_array.tolist())
    if scheme == "none":
        weights = np.ones(len(group_array), dtype=float)
    elif scheme == "group_sqrt":
        weights = np.asarray(
            [1.0 / np.sqrt(counts[group]) for group in group_array], dtype=float
        )
    else:
        weights = np.asarray(
            [1.0 / counts[group] for group in group_array], dtype=float
        )
    weights /= float(weights.mean())
    if not np.isfinite(weights).all() or not np.isclose(weights.mean(), 1.0):
        raise AssertionError("Fold-local training weights are invalid.")
    return weights


def _catboost_regressor(parameters: Mapping[str, Any], *, random_state: int, n_jobs: int):
    try:
        from catboost import CatBoostRegressor
    except ImportError as error:  # pragma: no cover - depends on optional runtime
        raise RuntimeError(
            "Generation-3 H1/H2 CatBoost arms require the optional 'catboost' package."
        ) from error
    return CatBoostRegressor(
        **dict(parameters),
        random_seed=int(random_state),
        thread_count=int(n_jobs),
        allow_writing_files=False,
        verbose=False,
    )


class AntisymmetricCatBoostRegressor:
    """CatBoost pair regressor with exact swap antisymmetry at inference."""

    def __init__(
        self,
        feature_columns: Iterable[str],
        *,
        parameters: Mapping[str, Any],
        weighting_scheme: str,
        random_state: int,
        n_jobs: int,
    ) -> None:
        self.feature_columns = tuple(feature_columns)
        self.parameters = dict(parameters)
        self.weighting_scheme = str(weighting_scheme)
        self.random_state = int(random_state)
        self.n_jobs = int(n_jobs)
        self.imputer: SimpleImputer | None = None
        self.model: Any | None = None
        self.fit_weight_audit: dict[str, Any] | None = None

    def fit(
        self,
        frame: pd.DataFrame,
        target: Iterable[float],
        groups: Iterable[Any],
    ) -> "AntisymmetricCatBoostRegressor":
        target_array = np.asarray(list(target), dtype=float)
        group_array = np.asarray(list(groups), dtype=object)
        if not (len(frame) == len(target_array) == len(group_array)):
            raise ValueError("frame, target, and groups must have equal length.")
        forward = _as_float_frame(frame, self.feature_columns)
        reverse = _as_float_frame(
            reverse_pair_features(_reversal_input(frame, self.feature_columns)),
            self.feature_columns,
        )
        augmented = pd.concat([forward, reverse], ignore_index=True)
        augmented_target = np.concatenate([target_array, -target_array])
        original_weights = fold_local_group_weights(
            group_array, self.weighting_scheme
        )
        augmented_weights = np.concatenate([original_weights, original_weights])
        self.imputer = SimpleImputer(strategy="median", add_indicator=True)
        transformed = self.imputer.fit_transform(augmented)
        self.model = _catboost_regressor(
            self.parameters, random_state=self.random_state, n_jobs=self.n_jobs
        )
        self.model.fit(transformed, augmented_target, sample_weight=augmented_weights)
        counts = Counter(group_array.tolist())
        self.fit_weight_audit = {
            "weighting_scheme": self.weighting_scheme,
            "fit_rows_before_swap": int(len(frame)),
            "fit_groups": int(len(counts)),
            "mean_weight_before_swap": float(original_weights.mean()),
            "minimum_weight": float(original_weights.min()),
            "maximum_weight": float(original_weights.max()),
            "group_counts_sha256": _mapping_sha256(counts),
            "counts_from_fit_subset_only": True,
        }
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.imputer is None or self.model is None:
            raise RuntimeError("Model has not been fitted.")
        forward = self.imputer.transform(
            _as_float_frame(frame, self.feature_columns)
        )
        reverse = self.imputer.transform(
            _as_float_frame(
                reverse_pair_features(_reversal_input(frame, self.feature_columns)),
                self.feature_columns,
            )
        )
        return (
            np.asarray(self.model.predict(forward), dtype=float)
            - np.asarray(self.model.predict(reverse), dtype=float)
        ) / 2.0


class AntisymmetricCorrectionRegressor:
    """Small regularized residual head with exact pair-swap antisymmetry."""

    def __init__(
        self,
        feature_columns: Iterable[str],
        *,
        kind: str,
        parameters: Mapping[str, Any],
        random_state: int,
        n_jobs: int,
    ) -> None:
        self.feature_columns = tuple(feature_columns)
        self.kind = str(kind)
        self.parameters = dict(parameters)
        self.random_state = int(random_state)
        self.n_jobs = int(n_jobs)
        self.pipeline: Pipeline | None = None
        self.imputer: SimpleImputer | None = None
        self.model: Any | None = None

    def _linear_pipeline(self) -> Pipeline:
        if self.kind == "ridge":
            model = Ridge(alpha=float(self.parameters["alpha"]))
        elif self.kind == "elasticnet":
            model = ElasticNet(
                alpha=float(self.parameters["alpha"]),
                l1_ratio=float(self.parameters["l1_ratio"]),
                max_iter=20_000,
                tol=1e-7,
                selection="cyclic",
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"Not a linear correction kind: {self.kind!r}")
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )

    def fit(
        self,
        frame: pd.DataFrame,
        residual: Iterable[float],
        groups: Iterable[Any],
    ) -> "AntisymmetricCorrectionRegressor":
        residual_array = np.asarray(list(residual), dtype=float)
        group_array = np.asarray(list(groups), dtype=object)
        forward = _as_float_frame(frame, self.feature_columns)
        reverse = _as_float_frame(
            reverse_pair_features(_reversal_input(frame, self.feature_columns)),
            self.feature_columns,
        )
        augmented = pd.concat([forward, reverse], ignore_index=True)
        augmented_residual = np.concatenate([residual_array, -residual_array])
        weights = fold_local_group_weights(group_array, "group_equal")
        augmented_weights = np.concatenate([weights, weights])
        if self.kind in {"ridge", "elasticnet"}:
            self.pipeline = self._linear_pipeline()
            self.pipeline.fit(
                augmented,
                augmented_residual,
                model__sample_weight=augmented_weights,
            )
        elif self.kind == "catboost":
            self.imputer = SimpleImputer(strategy="median", add_indicator=True)
            transformed = self.imputer.fit_transform(augmented)
            parameters = {
                "loss_function": "RMSE",
                "random_strength": 0.5,
                "bagging_temperature": 0.5,
                "rsm": 1.0,
                **self.parameters,
            }
            self.model = _catboost_regressor(
                parameters, random_state=self.random_state, n_jobs=self.n_jobs
            )
            self.model.fit(
                transformed,
                augmented_residual,
                sample_weight=augmented_weights,
            )
        else:
            raise ValueError(f"Unknown correction kind {self.kind!r}.")
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        forward_frame = _as_float_frame(frame, self.feature_columns)
        reverse_frame = _as_float_frame(
            reverse_pair_features(_reversal_input(frame, self.feature_columns)),
            self.feature_columns,
        )
        if self.pipeline is not None:
            forward = self.pipeline.predict(forward_frame)
            reverse = self.pipeline.predict(reverse_frame)
        elif self.imputer is not None and self.model is not None:
            forward = self.model.predict(self.imputer.transform(forward_frame))
            reverse = self.model.predict(self.imputer.transform(reverse_frame))
        else:
            raise RuntimeError("Correction model has not been fitted.")
        return (np.asarray(forward, dtype=float) - np.asarray(reverse, dtype=float)) / 2.0

    def signed_coefficients(self) -> np.ndarray | None:
        if self.pipeline is None:
            return None
        model = self.pipeline.named_steps["model"]
        coefficients = getattr(model, "coef_", None)
        if coefficients is None:
            return None
        # Missingness indicators follow the original columns. Stability
        # selection is defined only on the physical columns, not indicators.
        return np.asarray(coefficients, dtype=float)[: len(self.feature_columns)]


def _mapping_sha256(mapping: Mapping[Any, Any]) -> str:
    import hashlib
    import json

    payload = json.dumps(
        sorted((str(key), int(value)) for key, value in mapping.items()),
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class LatentDifferenceFit:
    """Fitted shared-g network and fold-local shoulder preprocessor."""

    imputer: SimpleImputer
    scaler: StandardScaler
    model: Any
    a_columns: tuple[str, ...]
    b_columns: tuple[str, ...]
    epochs_trained: int

    def _transform(self, frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
        values = _as_float_frame(frame, columns).to_numpy(dtype=float)
        return self.scaler.transform(self.imputer.transform(values)).astype(np.float32)

    def score_shoulders(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        import torch

        self.model.eval()
        with torch.no_grad():
            a = self.model(torch.from_numpy(self._transform(frame, self.a_columns)))
            b = self.model(torch.from_numpy(self._transform(frame, self.b_columns)))
        return a.cpu().numpy().reshape(-1), b.cpu().numpy().reshape(-1)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        score_a, score_b = self.score_shoulders(frame)
        return score_a - score_b


def _new_shared_score_network(input_dim: int, hidden_dims: Iterable[int], dropout: float):
    import torch.nn as nn

    layers: list[nn.Module] = []
    previous = int(input_dim)
    for width in hidden_dims:
        layers.extend([nn.Linear(previous, int(width)), nn.SiLU()])
        if float(dropout) > 0.0:
            layers.append(nn.Dropout(float(dropout)))
        previous = int(width)
    layers.append(nn.Linear(previous, 1))
    return nn.Sequential(*layers)


def _pseudo_huber_loss(prediction, target, weights, delta: float):
    import torch

    scaled = (prediction - target) / float(delta)
    per_row = float(delta) ** 2 * (torch.sqrt(1.0 + scaled**2) - 1.0)
    return torch.sum(per_row * weights) / torch.sum(weights)


def fit_latent_difference_model(
    frame: pd.DataFrame,
    target: Iterable[float],
    groups: Iterable[Any],
    *,
    a_columns: Iterable[str],
    b_columns: Iterable[str],
    parameters: Mapping[str, Any],
    model_seed: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
    huber_delta: float,
) -> LatentDifferenceFit:
    """Fit shared g with group-local weights and deterministic train-loss stopping."""

    import torch

    torch.manual_seed(int(model_seed))
    torch.use_deterministic_algorithms(True, warn_only=True)
    a_columns_tuple = tuple(a_columns)
    b_columns_tuple = tuple(b_columns)
    if len(a_columns_tuple) != len(b_columns_tuple):
        raise ValueError("A and B shoulder contracts must have equal width.")
    a_values = _as_float_frame(frame, a_columns_tuple).to_numpy(dtype=float)
    b_values = _as_float_frame(frame, b_columns_tuple).to_numpy(dtype=float)
    imputer = SimpleImputer(strategy="median", add_indicator=False)
    both = np.vstack([a_values, b_values])
    imputer.fit(both)
    scaler = StandardScaler()
    scaler.fit(imputer.transform(both))
    a = scaler.transform(imputer.transform(a_values)).astype(np.float32)
    b = scaler.transform(imputer.transform(b_values)).astype(np.float32)
    y = np.asarray(list(target), dtype=np.float32)
    weights = fold_local_group_weights(groups, "group_equal").astype(np.float32)
    model = _new_shared_score_network(
        a.shape[1], parameters["hidden_dims"], float(parameters["dropout"])
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(parameters["learning_rate"]),
        weight_decay=float(parameters["weight_decay"]),
    )
    a_tensor = torch.from_numpy(a)
    b_tensor = torch.from_numpy(b)
    y_tensor = torch.from_numpy(y)
    weight_tensor = torch.from_numpy(weights)
    rng = np.random.default_rng(int(model_seed))
    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    stale = 0
    epochs_trained = 0
    for epoch in range(int(max_epochs)):
        model.train()
        order = rng.permutation(len(frame))
        for start in range(0, len(order), int(batch_size)):
            batch = torch.from_numpy(order[start : start + int(batch_size)])
            prediction = model(a_tensor[batch]).squeeze(1) - model(
                b_tensor[batch]
            ).squeeze(1)
            loss = _pseudo_huber_loss(
                prediction, y_tensor[batch], weight_tensor[batch], huber_delta
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            full_prediction = model(a_tensor).squeeze(1) - model(b_tensor).squeeze(1)
            epoch_loss = float(
                _pseudo_huber_loss(
                    full_prediction, y_tensor, weight_tensor, huber_delta
                ).item()
            )
        epochs_trained = epoch + 1
        if epoch_loss < best_loss - 1e-7:
            best_loss = epoch_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                break
    if best_state is None:
        raise AssertionError("H3 training produced no finite checkpoint.")
    model.load_state_dict(best_state)
    return LatentDifferenceFit(
        imputer=imputer,
        scaler=scaler,
        model=model,
        a_columns=a_columns_tuple,
        b_columns=b_columns_tuple,
        epochs_trained=epochs_trained,
    )

