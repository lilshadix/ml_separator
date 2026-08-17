"""Generation-4 candidate estimators built on top of the frozen A2 champion.

The gen3 primary run (``runs/gen3_primary_20260815T181555Z``) was a valid
negative: neither a different learner (CatBoost), an electronic-structure
residual head, nor a latent-difference MLP beat the antisymmetric ExtraTrees
champion on unseen extractants.  Its diagnostics, however, located where A2's
error actually lives:

* the error is a per-ligand *magnitude* problem, not a within-ligand ranking
  problem (an oracle per-extractant offset lifts macro MAE 0.319 -> 0.247);
* within a lanthanide pair the condition-to-condition variation of the
  dominant extractant is anti-predicted (R2 about -1.2 on those rows) while the
  same predictions averaged per ligand rank ligands well (Spearman 0.64);
* the target is exactly transitive (log SF(A/B) = log D_A - log D_B) but tree
  predictions are not.

This module implements three candidates that target those findings without
touching the frozen protocol code:

``transitive_projection``
    Post-hoc least-squares projection of any pair predictions, inside each
    (extractant, condition) cell, onto the space of per-metal scores
    ``s_A - s_B``.  Uses only predictions, never labels.

``HierarchicalPairRegressor``
    Two-stage decomposition ``y = m(ligand, pair) + d(ligand, pair, condition)``:
    a *level* model fitted on training (extractant, pair) cell means and a
    *deviation* model fitted on within-cell residuals, so condition sensitivity
    is learned relative to each ligand's own level and transfers to unseen
    ligands.

``ScaleTrendPairRegressor``
    Explicit selectivity-scale model ``y = s(ligand, condition) * t(pair) + r``:
    ``t`` is the equal-extractant pair-label trend of the training fold, ``s`` is
    the per-(extractant, condition) least-squares slope of the training targets
    on ``t``, predicted for unseen ligands by a forest on cell-level ligand and
    condition features, and ``r`` is an antisymmetric ExtraTrees residual.

All estimators are fitted inside a fold with the training frame only; nothing
here reads a held-out label.  Every prediction is exactly antisymmetric under an
A/B swap: the level/scale terms are built from odd pair quantities and even
ligand/condition context, and the residual/deviation learners reuse
:class:`~lanthanide_separation.evaluation.AntisymmetricExtraTreesRegressor`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from .evaluation import (
    AntisymmetricExtraTreesRegressor,
    _as_float_frame,
    group_balanced_weights,
)
from .pairs import PAIR_TARGET_COLUMN

LIGAND_DESCRIPTOR_PREFIX = "lig2d__"
_PAIR_PREFIX = "pair__"


# ---------------------------------------------------------------------------
# Transitive projection
# ---------------------------------------------------------------------------


def transitive_projection(
    frame: pd.DataFrame,
    prediction: Iterable[float],
    *,
    cell_columns: Sequence[str] = ("extractant", "condition_id"),
    metal_a_column: str = "metal_A",
    metal_b_column: str = "metal_B",
    min_cell_rows: int = 2,
) -> np.ndarray:
    """Project pair predictions onto transitive-consistent per-metal scores.

    Within every cell (default: one extractant under one condition set) the
    true target satisfies ``y(A,B) = g_A - g_B`` for latent per-metal scores
    ``g``.  Predictions from a pair model do not; this returns, per cell, the
    least-squares fit ``s_A - s_B`` closest to the given predictions.  Cells
    with fewer than ``min_cell_rows`` rows, or whose incidence matrix cannot be
    solved, are returned unchanged.  Only predictions are used.
    """

    values = np.asarray(list(prediction), dtype=float)
    if len(values) != len(frame):
        raise ValueError("prediction and frame must have equal length.")
    result = values.copy()
    positions = np.arange(len(frame))
    keys = [frame[c].astype(str).to_numpy() for c in cell_columns]
    metal_a = frame[metal_a_column].astype(str).to_numpy()
    metal_b = frame[metal_b_column].astype(str).to_numpy()
    cell_key = pd.Series(list(zip(*keys)))
    for _, members in cell_key.groupby(cell_key, sort=False).groups.items():
        rows = positions[np.asarray(members, dtype=int)]
        if len(rows) < int(min_cell_rows):
            continue
        metals = sorted(set(metal_a[rows]) | set(metal_b[rows]))
        index = {m: i for i, m in enumerate(metals)}
        incidence = np.zeros((len(rows), len(metals)), dtype=float)
        for r, row in enumerate(rows):
            incidence[r, index[metal_a[row]]] = 1.0
            incidence[r, index[metal_b[row]]] = -1.0
        scores, *_ = np.linalg.lstsq(incidence, values[rows], rcond=None)
        result[rows] = incidence @ scores
    return result


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _split_columns(columns: Sequence[str]) -> tuple[list[str], list[str]]:
    """Return (context columns, pair columns) preserving order."""

    pair = [c for c in columns if str(c).startswith(_PAIR_PREFIX)]
    context = [c for c in columns if not str(c).startswith(_PAIR_PREFIX)]
    return context, pair


def _cell_frame(
    frame: pd.DataFrame,
    columns: Sequence[str],
    cell_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate ``columns`` to their mean within each cell.

    Returns the cell-level feature frame (index = cell id) and the row->cell
    mapping frame with a single ``cell`` column aligned to ``frame``.
    """

    key_arrays = [frame[c].astype(str).to_numpy() for c in cell_columns]
    cell_id = pd.Series(
        ["\x1f".join(parts) for parts in zip(*key_arrays)],
        index=frame.index,
        name="cell",
    )
    numeric = _as_float_frame(frame, columns)
    numeric.index = frame.index
    cells = numeric.groupby(cell_id.to_numpy(), sort=False).mean()
    return cells, cell_id.to_frame()


def _plain_forest(
    *,
    n_estimators: int,
    max_features: float,
    min_samples_leaf: int,
    random_state: int,
    n_jobs: int,
) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=int(n_estimators),
                    max_features=float(max_features),
                    min_samples_leaf=int(min_samples_leaf),
                    random_state=int(random_state),
                    n_jobs=int(n_jobs),
                ),
            ),
        ]
    )


@dataclass
class ForestParameters:
    """ExtraTrees hyperparameters shared by the candidate stages."""

    n_estimators: int = 200
    max_features: float = 0.35
    min_samples_leaf: int = 2

    def as_kwargs(self) -> dict[str, Any]:
        return {
            "n_estimators": int(self.n_estimators),
            "max_features": float(self.max_features),
            "min_samples_leaf": int(self.min_samples_leaf),
        }


# ---------------------------------------------------------------------------
# Hierarchical level + deviation model
# ---------------------------------------------------------------------------


@dataclass
class HierarchicalPairRegressor:
    """Level (extractant x pair) plus within-cell deviation, both antisymmetric.

    ``fit`` computes, on the training frame only, the mean target of every
    (extractant, pair_label) cell; the *level* forest is an antisymmetric
    ExtraTrees fitted on those cell means with the cell-mean features (ligand
    context, condition context averaged over the cell, pair block).  The
    *deviation* forest is an antisymmetric ExtraTrees fitted on all training
    rows with target ``y - m(cell)``, i.e. how each condition moves the target
    relative to the ligand's own level for that pair.  Prediction is
    ``level(cell features of the test rows) + deviation(test rows)``.
    """

    feature_columns: Sequence[str]
    level: ForestParameters = field(default_factory=ForestParameters)
    deviation: ForestParameters = field(default_factory=ForestParameters)
    random_state: int = 42
    n_jobs: int = -1
    cell_columns: tuple[str, ...] = ("extractant", "pair_label")
    level_group_column: str = "extractant"

    def __post_init__(self) -> None:
        self.feature_columns = tuple(self.feature_columns)
        self._level_model: AntisymmetricExtraTreesRegressor | None = None
        self._deviation_model: AntisymmetricExtraTreesRegressor | None = None

    def fit(
        self,
        frame: pd.DataFrame,
        target: Iterable[float],
        groups: Iterable[Any],
    ) -> "HierarchicalPairRegressor":
        y = np.asarray(list(target), dtype=float)
        g = np.asarray(list(groups), dtype=object)
        if len(frame) != len(y) or len(frame) != len(g):
            raise ValueError("frame, target and groups must have equal length.")
        cells, mapping = _cell_frame(frame, self.feature_columns, self.cell_columns)
        cell_id = mapping["cell"].to_numpy()
        cell_target = (
            pd.Series(y, index=frame.index).groupby(cell_id, sort=False).mean()
        )
        cell_group = (
            frame[self.level_group_column]
            .astype(str)
            .groupby(cell_id, sort=False)
            .first()
        )
        cells = cells.loc[cell_target.index]
        cell_group = cell_group.loc[cell_target.index]
        # Carry the identity columns the swap machinery needs (pair block is
        # already in ``cells``; nothing else is read by the level model).
        self._level_model = AntisymmetricExtraTreesRegressor(
            self.feature_columns,
            random_state=int(self.random_state) + 7_001,
            n_jobs=int(self.n_jobs),
            **self.level.as_kwargs(),
        )
        self._level_model.fit(cells, cell_target.to_numpy(), cell_group.to_numpy())

        level_of_row = pd.Series(cell_id).map(cell_target).to_numpy(dtype=float)
        deviation_target = y - level_of_row
        self._deviation_model = AntisymmetricExtraTreesRegressor(
            self.feature_columns,
            random_state=int(self.random_state) + 7_002,
            n_jobs=int(self.n_jobs),
            **self.deviation.as_kwargs(),
        )
        self._deviation_model.fit(frame, deviation_target, g)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self._level_model is None or self._deviation_model is None:
            raise RuntimeError("Model has not been fitted.")
        cells, mapping = _cell_frame(frame, self.feature_columns, self.cell_columns)
        level_by_cell = pd.Series(
            self._level_model.predict(cells), index=cells.index, dtype=float
        )
        level = mapping["cell"].map(level_by_cell).to_numpy(dtype=float)
        return level + self._deviation_model.predict(frame)


# ---------------------------------------------------------------------------
# Scale x trend model
# ---------------------------------------------------------------------------


def _unordered_pair_key(pair_labels: np.ndarray) -> np.ndarray:
    """``"Ho-Tm"`` and ``"Tm-Ho"`` map to the same key."""

    return np.asarray(
        ["-".join(sorted(str(label).split("-"))) for label in pair_labels], dtype=object
    )


def _pair_trend(
    pair_labels: np.ndarray,
    extractants: np.ndarray,
    target: np.ndarray,
    delta_z: np.ndarray,
) -> tuple[dict[str, float], float]:
    """Equal-extractant pair trend of the training fold, oriented as A -> B.

    The trend is keyed by the *unordered* metal pair and stored in the
    orientation ``sign(delta_Z) > 0`` (heavier metal is B), so that a swapped
    row reads the negated value.  Returns ``(trend by key, fallback slope)``
    where the fallback is a through-origin least-squares line in ``delta_Z``
    used for metal pairs absent from the training fold.
    """

    dz = np.asarray(delta_z, dtype=float)
    orient = np.where(dz >= 0.0, 1.0, -1.0)
    table = pd.DataFrame(
        {
            "pair": _unordered_pair_key(pair_labels),
            "extractant": extractants,
            "y": np.asarray(target, dtype=float) * orient,
            "dz": np.abs(dz),
        }
    )
    per_cell = table.groupby(["pair", "extractant"], sort=False)["y"].mean()
    trend = per_cell.groupby(level="pair").mean()
    spans = table.groupby("pair", sort=False)["dz"].first().loc[trend.index]
    denominator = float(np.sum(spans.to_numpy() ** 2))
    slope = (
        float(np.sum(trend.to_numpy() * spans.to_numpy()) / denominator)
        if denominator > 0
        else 0.0
    )
    return trend.to_dict(), slope


@dataclass
class ScaleTrendPairRegressor:
    """``y = s(extractant, condition) * t(pair) + r`` with a learned scale.

    * ``t`` is the equal-extractant mean target per pair label over the
      training fold (odd under swap by construction; a through-origin line in
      ``delta_Z`` covers pair labels missing from the fold).
    * ``s`` is the through-origin least-squares slope of the training targets on
      ``t`` inside each (extractant, condition) cell (falls back to the
      extractant-level slope for cells with fewer than ``min_cell_rows`` rows).
    * A plain ExtraTrees maps cell-mean *context* features (ligand + condition,
      no pair block) to ``s``; it is trained with equal total weight per
      extractant, mirroring the champion's group balancing.
    * The residual ``r = y - s * t`` is fitted by an antisymmetric ExtraTrees on
      all feature columns.  Prediction: ``s_hat * t + r_hat``.
    """

    feature_columns: Sequence[str]
    scale: ForestParameters = field(
        default_factory=lambda: ForestParameters(max_features=0.35, min_samples_leaf=2)
    )
    residual: ForestParameters = field(default_factory=ForestParameters)
    random_state: int = 42
    n_jobs: int = -1
    cell_columns: tuple[str, ...] = ("extractant", "condition_id")
    pair_label_column: str = "pair_label"
    delta_z_column: str = "pair__delta_Z"
    min_cell_rows: int = 3
    scale_clip: tuple[float, float] = (-1.0, 4.0)
    #: Residual training uses *cross-fitted* scale predictions (leave-extractants-
    #: out inside the training fold, this many folds) so the residual forest sees
    #: the same kind of scale error it will meet on unseen ligands.  ``0`` uses
    #: the in-sample cell slopes instead.
    crossfit_folds: int = 5
    #: Predicted scales are shrunk toward the weighted training mean scale:
    #: ``s = (1 - shrink) * s_hat + shrink * s_bar``.  ``0`` = no shrinkage.
    scale_shrinkage: float = 0.0

    def __post_init__(self) -> None:
        self.feature_columns = tuple(self.feature_columns)
        self._context_columns, _ = _split_columns(self.feature_columns)
        self._trend: dict[str, float] = {}
        self._fallback_slope = 0.0
        self._scale_model: Pipeline | None = None
        self._residual_model: AntisymmetricExtraTreesRegressor | None = None
        self._mean_scale = 1.0
        self.fitted_scales_: pd.Series | None = None
        self.crossfit_scales_: pd.Series | None = None

    # -- helpers -----------------------------------------------------------
    def _trend_of(self, frame: pd.DataFrame) -> np.ndarray:
        keys = _unordered_pair_key(frame[self.pair_label_column].to_numpy())
        dz = frame[self.delta_z_column].to_numpy(dtype=float)
        orient = np.where(dz >= 0.0, 1.0, -1.0)
        out = np.array([self._trend.get(key, np.nan) for key in keys], dtype=float)
        missing = ~np.isfinite(out)
        if missing.any():
            out[missing] = self._fallback_slope * np.abs(dz[missing])
        return out * orient

    @staticmethod
    def _slope(y: np.ndarray, t: np.ndarray) -> float:
        denominator = float(np.sum(t * t))
        return float(np.sum(y * t) / denominator) if denominator > 0 else float("nan")

    # -- API ---------------------------------------------------------------
    def _fit_scale_stage(
        self, frame: pd.DataFrame, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit trend + per-cell scales + scale forest on the training frame.

        Returns ``(cross-fitted scale per training row, trend per training
        row)`` so that callers can build a leakage-free base ``s * t`` for the
        training rows: every row's scale comes from a forest that never saw its
        extractant.
        """

        self._trend, self._fallback_slope = _pair_trend(
            frame[self.pair_label_column].astype(str).to_numpy(),
            frame["extractant"].astype(str).to_numpy(),
            y,
            frame[self.delta_z_column].to_numpy(dtype=float),
        )
        t = self._trend_of(frame)

        cells, mapping = _cell_frame(frame, self._context_columns, self.cell_columns)
        cell_id = mapping["cell"].to_numpy()
        extractant = frame["extractant"].astype(str).to_numpy()
        # Per-cell through-origin slope with extractant-level fallback.
        table = pd.DataFrame({"cell": cell_id, "e": extractant, "y": y, "t": t})
        by_extractant = {
            e: self._slope(b["y"].to_numpy(), b["t"].to_numpy())
            for e, b in table.groupby("e", sort=False)
        }
        scales: dict[str, float] = {}
        for cell, block in table.groupby("cell", sort=False):
            value = (
                self._slope(block["y"].to_numpy(), block["t"].to_numpy())
                if len(block) >= int(self.min_cell_rows)
                else float("nan")
            )
            if not np.isfinite(value):
                value = by_extractant[str(block["e"].iloc[0])]
            if not np.isfinite(value):
                value = 1.0
            scales[cell] = float(np.clip(value, *self.scale_clip))
        cell_scale = pd.Series(scales, dtype=float).loc[cells.index]
        cell_extractant = (
            pd.Series(extractant, index=frame.index).groupby(cell_id, sort=False).first()
        ).loc[cells.index]
        self.fitted_scales_ = cell_scale.copy()
        cell_weights = group_balanced_weights(cell_extractant.to_numpy())
        self._mean_scale = float(
            np.average(cell_scale.to_numpy(), weights=cell_weights)
        )

        def _new_scale_forest(seed_offset: int) -> Pipeline:
            # Single-threaded on purpose: the cell-level forests are tiny (a few
            # hundred rows) and their *predictions* feed the residual target.
            # With n_jobs > 1 sklearn accumulates tree outputs in thread-
            # completion order, which perturbs the predicted scales at the
            # 1e-16 level; those perturbations flip near-tie splits in the
            # downstream forest and make the whole estimator non-reproducible
            # (observed: +-0.002 macro MAE between two identical fits).
            return _plain_forest(
                random_state=int(self.random_state) + 7_003 + int(seed_offset),
                n_jobs=1,
                **self.scale.as_kwargs(),
            )

        self._scale_model = _new_scale_forest(0)
        self._scale_model.fit(cells, cell_scale.to_numpy(), model__sample_weight=cell_weights)

        # Cross-fitted scale predictions for the residual stage: every training
        # cell gets a scale predicted by a forest that never saw its extractant.
        base_scale = cell_scale.copy()
        n_extractants = int(pd.Series(cell_extractant.to_numpy()).nunique())
        if int(self.crossfit_folds) >= 2 and n_extractants >= 2:
            splitter = GroupKFold(n_splits=min(int(self.crossfit_folds), n_extractants))
            crossfit = pd.Series(np.nan, index=cells.index, dtype=float)
            cell_scale_values = cell_scale.to_numpy()
            for k, (fit_idx, held_idx) in enumerate(
                splitter.split(cells, cell_scale_values, groups=cell_extractant.to_numpy())
            ):
                forest = _new_scale_forest(10 * (k + 1))
                forest.fit(
                    cells.iloc[fit_idx],
                    cell_scale_values[fit_idx],
                    model__sample_weight=cell_weights[fit_idx],
                )
                crossfit.iloc[held_idx] = forest.predict(cells.iloc[held_idx])
            base_scale = crossfit.clip(*self.scale_clip)
            base_scale = (1.0 - self.scale_shrinkage) * base_scale + self.scale_shrinkage * self._mean_scale
        self.crossfit_scales_ = base_scale.copy()
        row_scale = pd.Series(cell_id).map(base_scale).to_numpy(dtype=float)
        return row_scale, t

    def fit(
        self,
        frame: pd.DataFrame,
        target: Iterable[float],
        groups: Iterable[Any],
    ) -> "ScaleTrendPairRegressor":
        y = np.asarray(list(target), dtype=float)
        g = np.asarray(list(groups), dtype=object)
        if len(frame) != len(y) or len(frame) != len(g):
            raise ValueError("frame, target and groups must have equal length.")
        row_scale, t = self._fit_scale_stage(frame, y)
        base = row_scale * t
        self._residual_model = AntisymmetricExtraTreesRegressor(
            self.feature_columns,
            random_state=int(self.random_state) + 7_004,
            n_jobs=int(self.n_jobs),
            **self.residual.as_kwargs(),
        )
        self._residual_model.fit(frame, y - base, g)
        return self

    def predict_scale(self, frame: pd.DataFrame) -> np.ndarray:
        if self._scale_model is None:
            raise RuntimeError("Model has not been fitted.")
        cells, mapping = _cell_frame(frame, self._context_columns, self.cell_columns)
        predicted = np.clip(self._scale_model.predict(cells), *self.scale_clip)
        predicted = (1.0 - self.scale_shrinkage) * predicted + self.scale_shrinkage * self._mean_scale
        scale_by_cell = pd.Series(predicted, index=cells.index, dtype=float)
        return mapping["cell"].map(scale_by_cell).to_numpy(dtype=float)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self._residual_model is None:
            raise RuntimeError("Model has not been fitted.")
        base = self.predict_scale(frame) * self._trend_of(frame)
        return base + self._residual_model.predict(frame)


# ---------------------------------------------------------------------------
# A2 + scale prior as stacked features
# ---------------------------------------------------------------------------

PRIOR_SCALE_COLUMN = f"{LIGAND_DESCRIPTOR_PREFIX}prior_scale"
PRIOR_ABS_BASE_COLUMN = f"{LIGAND_DESCRIPTOR_PREFIX}prior_abs_base"


@dataclass
class PriorAugmentedPairRegressor:
    """The champion forest with the learned selectivity scale as extra features.

    Instead of *replacing* the forest by ``s * t + r`` this keeps the A2 recipe
    and appends two even context columns per row: the predicted cell scale
    ``s`` and ``s * |t(pair)|`` (the magnitude of the scale-trend prior; the
    sign is carried by the odd ``pair__delta_Z`` block, so the antisymmetric
    forest can use both).  For training rows ``s`` is the *cross-fitted*
    prediction from :class:`ScaleTrendPairRegressor` (leave-extractants-out
    inside the training fold), so the forest learns how much to trust a prior
    of exactly the quality it will meet on unseen ligands.  No held-out label
    enters: test rows receive ``s`` from the scale forest fitted on training
    cells only.
    """

    feature_columns: Sequence[str]
    forest: ForestParameters = field(default_factory=ForestParameters)
    scale: ForestParameters = field(
        default_factory=lambda: ForestParameters(max_features=0.35, min_samples_leaf=2)
    )
    random_state: int = 42
    n_jobs: int = -1
    crossfit_folds: int = 5
    scale_shrinkage: float = 0.0
    #: ``"scale"`` appends the ligand-derived scale ``s`` and ``s * |t|``;
    #: ``"trend_only"`` is the attribution control that appends only ``|t|`` (the
    #: training-fold pair trend magnitude, identical for every ligand) so that a
    #: gain can be assigned to the ligand-derived scale or to the trend feature.
    mode: str = "scale"

    def __post_init__(self) -> None:
        if self.mode not in ("scale", "trend_only"):
            raise ValueError("mode must be 'scale' or 'trend_only'.")
        self.feature_columns = tuple(self.feature_columns)
        self._scale_stage = ScaleTrendPairRegressor(
            self.feature_columns,
            scale=self.scale,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            crossfit_folds=self.crossfit_folds,
            scale_shrinkage=self.scale_shrinkage,
        )
        self._model: AntisymmetricExtraTreesRegressor | None = None

    @property
    def augmented_columns(self) -> tuple[str, ...]:
        if self.mode == "trend_only":
            return tuple(self.feature_columns) + (PRIOR_ABS_BASE_COLUMN,)
        return tuple(self.feature_columns) + (PRIOR_SCALE_COLUMN, PRIOR_ABS_BASE_COLUMN)

    def _augment(self, frame: pd.DataFrame, scale: np.ndarray, trend: np.ndarray) -> pd.DataFrame:
        out = frame.copy()
        if self.mode == "trend_only":
            out[PRIOR_ABS_BASE_COLUMN] = np.abs(trend)
            return out
        out[PRIOR_SCALE_COLUMN] = scale
        out[PRIOR_ABS_BASE_COLUMN] = scale * np.abs(trend)
        return out

    def fit(
        self,
        frame: pd.DataFrame,
        target: Iterable[float],
        groups: Iterable[Any],
    ) -> "PriorAugmentedPairRegressor":
        y = np.asarray(list(target), dtype=float)
        g = np.asarray(list(groups), dtype=object)
        if len(frame) != len(y) or len(frame) != len(g):
            raise ValueError("frame, target and groups must have equal length.")
        row_scale, t = self._scale_stage._fit_scale_stage(frame, y)
        self._model = AntisymmetricExtraTreesRegressor(
            self.augmented_columns,
            random_state=int(self.random_state) + 7_005,
            n_jobs=int(self.n_jobs),
            **self.forest.as_kwargs(),
        )
        self._model.fit(self._augment(frame, row_scale, t), y, g)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model has not been fitted.")
        scale = self._scale_stage.predict_scale(frame)
        trend = self._scale_stage._trend_of(frame)
        return self._model.predict(self._augment(frame, scale, trend))


# ---------------------------------------------------------------------------
# Optional ligand-descriptor augmentation
# ---------------------------------------------------------------------------


def attach_ligand_descriptors(
    frame: pd.DataFrame,
    descriptors: pd.DataFrame,
    *,
    smiles_column: str = "extractant",
    key_column: str = "canonical_smiles",
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Left-join ``lig2d__*`` ligand descriptors onto the pair frame.

    Ligand descriptors are even (identical for A and B) context features; the
    ``lig2d__`` prefix declares no swap parity and is therefore left untouched
    by :func:`lanthanide_separation.pairs.reverse_pair_features`, which is the
    correct behaviour for a ligand-level quantity.  Returns the augmented frame
    and the tuple of attached column names.  Missing ligands get NaN, handled by
    the fold-local imputer.
    """

    columns = [
        c for c in descriptors.columns if str(c).startswith(LIGAND_DESCRIPTOR_PREFIX)
    ]
    if not columns:
        raise ValueError("descriptor table carries no lig2d__ columns.")
    table = descriptors[[key_column, *columns]].drop_duplicates(key_column)
    merged = frame.merge(
        table, how="left", left_on=smiles_column, right_on=key_column
    )
    if len(merged) != len(frame):
        raise AssertionError("descriptor join changed the row count.")
    merged.index = frame.index
    if key_column != smiles_column:
        merged = merged.drop(columns=[key_column])
    return merged, tuple(columns)


__all__ = [
    "ForestParameters",
    "HierarchicalPairRegressor",
    "LIGAND_DESCRIPTOR_PREFIX",
    "PriorAugmentedPairRegressor",
    "ScaleTrendPairRegressor",
    "attach_ligand_descriptors",
    "transitive_projection",
]
