"""``fit`` once, ``predict(query)`` many times — the contract gen10 needs.

gen9's contenders implement gen7's ``fit_predict(train, y, test, context)``, which
fuses two operations that gen10 has to separate.  A query-set consistency
measurement asks *the same fitted model* about the same absolute point inside
several different candidate designs; with ``fit_predict`` that costs a refit per
query and confounds "the design changed" with "the forest was regrown".

So every gen10 arm is a :class:`QueryModel` — ``fit`` returns a
:class:`FittedModel` whose ``predict`` takes a frame of conditions and rebuilds
the design-relative coordinates *from that frame*.  Fitting is unchanged; what
changes is that prediction becomes a function of the question.

Four families, and the reason each exists:

:class:`MonolithModel`
    the frozen forest with design-relative columns appended.  With
    ``representation="GEN9"`` and ``gen9_compat=True`` it is
    ``GEN9_REL_MONOLITH`` bit-for-bit, asserted in the test suite; with
    ``representation="NONE"`` it is ``REC_ecfp_plus_recovered`` bit-for-bit.  Both
    endpoints of gen9's comparison therefore live inside one class, and a
    difference between them cannot be a difference of implementation.
:class:`RecomposedModel`
    ``mean_over_curve(base) + (shape - mean(shape))``.  ``GEN9_SHAPE_RECOMPOSED``,
    re-expressed.  The level is whatever the monolith said, so the recomposition
    spends none of the accuracy the monolith bought by drawing flat.
:class:`LevelShapeModel`
    the brief's Phase 2D: an **explicit** level head trained on curve-mean targets,
    plus a shape head constrained to sum to zero over each requested curve.  The
    question it answers is whether the recomposition is an architecture or a
    post-processing step — if a level model beats averaging the monolith, the
    recomposition was doing the level model's job badly.
:class:`ResidualShapeModel`
    keep the monolith's shape and *add* a learned, mean-preserving correction,
    rather than replacing it.  gen8's two-stage residual trap says a stage fitted
    on the raw cross-fitted residual learns level error and hurts; this fits the
    **curve-centred** residual, where the level cannot enter by construction, which
    is the same repair gen8 found worked.

All four share :class:`DesignBuilder`, so "the design matrix" means one thing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np
import pandas as pd

from ..gen7.harness import FoldContext
from ..gen9.curves import DEFAULT_AXES
from ..gen9.train import FROZEN_BLOCKS
from ..levels import group_balanced_weights
from .features import Representation, matrix, query_features, representation
from .querycurves import primary_curve, query_membership

#: The frozen arm's clip: predictions may leave the training range by half its
#: spread on each side and no further.  Transcribed from ``gen7.contenders``.
CLIP_MULTIPLIER = 0.5

#: Rounding applied to any model output that becomes another model's **target**.
#:
#: ``ExtraTreesRegressor(n_jobs=-1)`` sums its trees in thread-completion order, so
#: one forest's predictions move by ~1e-15 between runs.  In a single forest that is
#: irrelevant.  Feed it into a second model's target and it perturbs a pseudo-target,
#: flips a near-tied split comparison, grows a different tree, and the divergence
#: compounds — gen9 lost a whole sweep to this (decision report §0.3, issue 6).
#:
#: gen10 hit it again, in a *new* arm.  :class:`ResidualShapeModel` fits its
#: correction on ``y - base.predict(train)``; two runs of the identical
#: configuration in the same process differed by **3.4e-2 log units on 372 of 382
#: held-out rows** before this constant existed, and by 4e-16 after.  1e-9 log units
#: is a billion times finer than any measurement in the corpus, so the rounding
#: cannot change a fitted model in any way that matters — it only removes a
#: perturbation whose sole effect is to make the arm unreproducible.
#:
#: Applied at model-output-to-target boundaries only.  Final predictions are left
#: unrounded so the exact-nesting assertions against gen9 stay exact.
ROUND_DECIMALS = 9


def stabilise(values: np.ndarray) -> np.ndarray:
    """Round a model output before it becomes another model's target."""
    return np.round(np.asarray(values, dtype=float), ROUND_DECIMALS)


class FittedModel(Protocol):
    """What ``QueryModel.fit`` returns: something that answers a question."""

    def predict(self, query: pd.DataFrame) -> np.ndarray: ...


class QueryModel(Protocol):
    name: str

    def fit(self, train: pd.DataFrame, y_train: np.ndarray,
            context: FoldContext) -> FittedModel: ...


# --------------------------------------------------------------------------- #
# Shared machinery
# --------------------------------------------------------------------------- #

#: ``row_id -> raw feature row`` for the cohort, per block tuple.
#:
#: Consistency measurement asks a fitted model thousands of small questions, and
#: ``_as_float_frame`` runs ``pd.to_numeric`` over 2,160 columns on every call — a
#: cost that is negligible on a 4,866-row training fold and ruinous on a 30-row
#: candidate design.  The cache stores the *pre-imputation* numeric matrix, which
#: is a pure function of the cohort and the block list, so it can be shared across
#: folds, seeds and arms without any of them seeing a row they should not: the
#: imputer, the ``keep`` mask and the model are all still fold-local, and the test
#: suite asserts a cached transform equals an uncached one bit-for-bit.
_RAW_CACHE: dict[tuple, tuple[dict, np.ndarray]] = {}


def _raw_matrix(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    from ..levels import _as_float_frame

    return _as_float_frame(frame, columns).to_numpy()


def prime_raw_cache(cohort, blocks: Sequence[str] = FROZEN_BLOCKS) -> int:
    """Precompute the cohort's numeric feature matrix once."""
    key = tuple(blocks)
    columns = tuple(cohort.block_columns(key))
    frame = cohort.frame
    matrix_ = _raw_matrix(frame, columns)
    index = {str(r): i for i, r in enumerate(frame["row_id"].astype(str).to_numpy())}
    _RAW_CACHE[key] = (index, matrix_)
    return len(index)


def clear_raw_cache() -> None:
    _RAW_CACHE.clear()


@dataclass
class DesignBuilder:
    """The frozen arm's preprocessing, split into ``fit`` and ``transform``.

    ``gen9.train.build_design`` does both at once and returns two matrices.  That
    is the same arithmetic, but it cannot answer a second query without refitting
    the imputer on the training fold again — and an imputer refitted on identical
    data is identical, so the split changes nothing except what is expressible.
    Asserted against ``build_design`` in ``tests/test_gen10_architectures.py``.
    """

    blocks: tuple[str, ...] = FROZEN_BLOCKS
    add_indicator: bool = True
    columns: tuple[str, ...] = ()
    keep: np.ndarray | None = None
    imputer: object | None = None

    def fit(self, cohort, train: pd.DataFrame) -> "DesignBuilder":
        from sklearn.impute import SimpleImputer

        self.columns = tuple(cohort.block_columns(tuple(self.blocks)))
        raw = self._raw(train)
        self.keep = ~np.all(np.isnan(raw), axis=0)
        self.imputer = SimpleImputer(
            strategy="median", add_indicator=self.add_indicator).fit(raw[:, self.keep])
        return self

    def _raw(self, frame: pd.DataFrame) -> np.ndarray:
        entry = _RAW_CACHE.get(tuple(self.blocks))
        if entry is None:
            return _raw_matrix(frame, self.columns)
        index, matrix_ = entry
        row_ids = frame["row_id"].astype(str).to_numpy()
        positions = np.array([index.get(r, -1) for r in row_ids])
        if (positions < 0).all():
            return _raw_matrix(frame, self.columns)
        out = np.empty((len(row_ids), matrix_.shape[1]), dtype=float)
        known = positions >= 0
        if known.any():
            out[known] = matrix_[positions[known]]
        if (~known).any():
            # Synthetic candidate rows are not in the cohort by construction.
            out[~known] = _raw_matrix(frame.iloc[np.flatnonzero(~known)], self.columns)
        return out

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.imputer is None:
            raise RuntimeError("DesignBuilder.transform before fit")
        return self.imputer.transform(self._raw(frame)[:, self.keep])


def _forest(seed: int, *, n_estimators: int, max_features, min_samples_leaf: int):
    from sklearn.ensemble import ExtraTreesRegressor

    return ExtraTreesRegressor(n_estimators=n_estimators, max_features=max_features,
                               min_samples_leaf=min_samples_leaf, random_state=seed,
                               n_jobs=-1)


def _tile(block: np.ndarray, times: int) -> np.ndarray:
    """Replicate the context columns so a random split subset reaches them.

    The brief's Phase 2B asks for "guaranteed inclusion of condition/context
    columns in candidate split subsets".  scikit-learn has no such knob, and
    writing a bespoke tree to get one would change the learner and therefore
    confound the comparison the phase exists to make.  Replicating the columns
    ``m`` times raises the probability that at least one copy is offered at a split
    from ``1 - (1 - p)^c`` to ``1 - (1 - p)^(mc)`` while leaving the learner, the
    hyperparameters and every other column untouched — the same intervention,
    expressed in the design matrix instead of in the estimator.
    """
    if times <= 1 or block.shape[1] == 0:
        return block
    return np.hstack([block] * int(times))


def _curve_of(frame: pd.DataFrame, membership: pd.DataFrame,
              axes: Sequence[str]) -> np.ndarray:
    """Per-row primary ``curve_id`` for ``frame``; ``None`` where there is none."""
    primary = primary_curve(membership, axes=axes)
    if primary.empty:
        return np.full(len(frame), None, dtype=object)
    lookup = primary.set_index("row_id")["curve_id"]
    return lookup.reindex(frame["row_id"].astype(str).to_numpy()).to_numpy(dtype=object)


def _curve_means(curve: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Mean of ``values`` over each curve, broadcast back to rows; NaN off-curve."""
    frame = pd.DataFrame({"curve": curve, "v": values})
    on = pd.notna(frame["curve"]).to_numpy()
    if not on.any():
        return np.full(len(values), np.nan)
    means = frame[on].groupby("curve")["v"].mean()
    return np.where(on, frame["curve"].map(means).to_numpy(dtype=float), np.nan)


def _recompose(base: np.ndarray, shape: np.ndarray, curve: np.ndarray) -> np.ndarray:
    """``mean_over_curve(base) + (shape - mean_over_curve(shape))``, off-curve ``base``.

    Mean-preserving by construction: the curve's predicted level is exactly the
    level the base model predicted, so the operation cannot move a level and
    cannot be credited with a level gain.  Asserted to 1e-9 in the test suite.
    """
    out = base.copy()
    frame = pd.DataFrame({"curve": curve, "base": base, "shape": shape})
    on = frame[pd.notna(frame["curve"])]
    if on.empty:
        return out
    for _, rows in on.groupby("curve").groups.items():
        index = np.asarray(rows, dtype=int)
        level = float(base[index].mean())
        centred = shape[index]
        out[index] = level + centred - centred.mean()
    return out


def cross_fitted_predictions(x: np.ndarray, y: np.ndarray, weights: np.ndarray,
                             groups: np.ndarray, *, seed: int, n_splits: int,
                             forest_kwargs: dict) -> np.ndarray:
    """Out-of-fold predictions for the training rows, grouped by chemotype.

    A second-stage model fitted on a first stage's *in-sample* residual learns
    nothing useful: an ExtraTrees forest with ``min_samples_leaf = 2`` nearly
    interpolates its own training rows, so the residual is ~0 everywhere and the
    second stage collapses.  gen6 found the cross-fitted residual learns level
    error instead and hurts (1.25 vs 1.08) — **unless** it is centred per curve,
    where the level cannot enter (1.04).  gen10's residual arms therefore use
    inner-fold cross-fitting *and* curve-centring, and the in-sample variant is
    kept only as the labelled trap control.

    Inner folds are ``seeded_group_kfold`` over the same chemotype labels the
    outer plan uses, seeded from the model seed, so a held-out chemotype in an
    inner fold is as unseen as one in an outer fold.
    """
    from ..gen6.cohorts import seeded_group_kfold

    out = np.full(len(y), np.nan)
    for k, (train_index, test_index) in enumerate(
            seeded_group_kfold(groups, n_splits, int(seed))):
        forest = _forest(int(seed) + 101 * (k + 1), **forest_kwargs)
        forest.fit(x[train_index], y[train_index], sample_weight=weights[train_index])
        out[test_index] = forest.predict(x[test_index])
    if np.isnan(out).any():
        raise AssertionError("cross-fitting left training rows unpredicted")
    return stabilise(out)


@dataclass
class _FittedBase:
    """State every fitted gen10 arm carries."""

    design: DesignBuilder
    spec: Representation
    axes: tuple[str, ...]
    lo: float
    hi: float
    gen9_compat: bool = False
    context_replication: int = 1
    membership_source: str = "query"
    cohort_membership: pd.DataFrame | None = None

    def _membership(self, query: pd.DataFrame) -> pd.DataFrame:
        if self.membership_source == "cohort":
            if self.cohort_membership is None:
                raise RuntimeError("membership_source='cohort' without a table")
            keep = set(query["row_id"].astype(str))
            return self.cohort_membership[
                self.cohort_membership["row_id"].astype(str).isin(keep)]
        return query_membership(query, axes=self.axes)

    def _context(self, query: pd.DataFrame, membership: pd.DataFrame) -> np.ndarray:
        if not self.spec.columns:
            return np.zeros((len(query), 0), dtype=float)
        features = query_features(query, membership, axes=self.axes,
                                  window=self.spec.window, gen9_compat=self.gen9_compat)
        return _tile(matrix(features, query["row_id"].astype(str).to_numpy(), self.spec),
                     self.context_replication)

    def _clip(self, values: np.ndarray) -> np.ndarray:
        return np.clip(values, self.lo, self.hi)

    def set_predict_jobs(self, n_jobs: int) -> "_FittedBase":
        """Thread count for *prediction* only; fitting is already done.

        A 400-tree forest answering a 30-row query gains nothing from eight
        threads and costs the machine a thread-pool spin-up per call.  The
        consistency benchmark makes thousands of such calls, so it predicts on
        one thread and leaves the cores to the training runs.  Predictions are
        identical up to the thread-order floor either way.
        """
        for attribute in ("model", "base", "shape", "level", "correction", "branch"):
            forest = getattr(self, attribute, None)
            if forest is not None and hasattr(forest, "n_jobs"):
                forest.n_jobs = int(n_jobs)
        return self


# --------------------------------------------------------------------------- #
# 1. The monolith
# --------------------------------------------------------------------------- #

@dataclass
class _FittedMonolith(_FittedBase):
    model: object = None

    def predict(self, query: pd.DataFrame) -> np.ndarray:
        membership = self._membership(query)
        x = np.hstack([self.design.transform(query), self._context(query, membership)])
        return self._clip(np.asarray(self.model.predict(x), dtype=float))


@dataclass
class MonolithModel:
    """One forest on ``[static design | design-relative context]``.

    ``representation="NONE"`` is the frozen arm; ``representation="GEN9"`` with
    ``gen9_compat=True`` is ``GEN9_REL_MONOLITH``.  Everything Phase 2B varies —
    ``max_features``, ``context_replication`` — lives here, so a feature-access
    result cannot be a different-model result.
    """

    name: str = "GEN10_MONOLITH"
    representation_name: str = "GEN9"
    window: str = "primary"
    blocks: tuple[str, ...] = FROZEN_BLOCKS
    add_indicator: bool = True
    n_estimators: int = 400
    max_features: object = 0.30
    min_samples_leaf: int = 2
    axes: tuple[str, ...] = DEFAULT_AXES
    gen9_compat: bool = False
    context_replication: int = 1
    membership_source: str = "query"
    cohort_membership: pd.DataFrame | None = None
    diagnostics: list = field(default_factory=list)

    @property
    def spec(self) -> Representation:
        return representation(self.representation_name, window=self.window)

    def fit(self, train: pd.DataFrame, y_train: np.ndarray,
            context: FoldContext) -> _FittedMonolith:
        design = DesignBuilder(blocks=tuple(self.blocks),
                               add_indicator=self.add_indicator).fit(context.cohort, train)
        spread = float(y_train.max() - y_train.min())
        fitted = _FittedMonolith(
            design=design, spec=self.spec, axes=tuple(self.axes),
            lo=float(y_train.min() - CLIP_MULTIPLIER * spread),
            hi=float(y_train.max() + CLIP_MULTIPLIER * spread),
            gen9_compat=self.gen9_compat, context_replication=self.context_replication,
            membership_source=self.membership_source,
            cohort_membership=self.cohort_membership)
        x = np.hstack([design.transform(train),
                       fitted._context(train, fitted._membership(train))])
        model = _forest(context.model_seed, n_estimators=self.n_estimators,
                        max_features=self.max_features,
                        min_samples_leaf=self.min_samples_leaf)
        model.fit(x, y_train, sample_weight=group_balanced_weights(train["ecfp_cluster"]))
        fitted.model = model
        self.diagnostics.append({"arm": self.name, "split_seed": context.fold.seed,
                                 "fold": context.fold.fold, "n_features": int(x.shape[1]),
                                 "representation": self.spec.as_dict()})
        return fitted


# --------------------------------------------------------------------------- #
# 2. The recomposition
# --------------------------------------------------------------------------- #

@dataclass
class _FittedRecomposed(_FittedBase):
    base: object = None
    shape: object = None
    shape_uses_context: bool = True

    def predict(self, query: pd.DataFrame) -> np.ndarray:
        membership = self._membership(query)
        static = self.design.transform(query)
        base = self._clip(np.asarray(self.base.predict(static), dtype=float))
        context = self._context(query, membership)
        shape = np.asarray(self.shape.predict(
            np.hstack([static, context]) if self.shape_uses_context else static), dtype=float)
        curve = _curve_of(query, membership, self.axes)
        return self._clip(_recompose(base, shape, curve))


@dataclass
class RecomposedModel:
    """``GEN9_SHAPE_RECOMPOSED``, re-expressed in the query contract.

    The base forest is the monolith; the shape forest is fitted on the
    **curve-centred** training target with the design-relative columns appended,
    and the two are recombined so each requested curve keeps the monolith's level.
    """

    name: str = "GEN10_SHAPE_RECOMPOSED"
    representation_name: str = "GEN9"
    window: str = "primary"
    blocks: tuple[str, ...] = FROZEN_BLOCKS
    add_indicator: bool = True
    n_estimators: int = 400
    max_features: object = 0.30
    shape_max_features: object = None
    min_samples_leaf: int = 2
    axes: tuple[str, ...] = DEFAULT_AXES
    gen9_compat: bool = False
    context_replication: int = 1
    membership_source: str = "query"
    cohort_membership: pd.DataFrame | None = None
    diagnostics: list = field(default_factory=list)

    @property
    def spec(self) -> Representation:
        return representation(self.representation_name, window=self.window)

    def fit(self, train: pd.DataFrame, y_train: np.ndarray,
            context: FoldContext) -> _FittedRecomposed:
        design = DesignBuilder(blocks=tuple(self.blocks),
                               add_indicator=self.add_indicator).fit(context.cohort, train)
        spread = float(y_train.max() - y_train.min())
        fitted = _FittedRecomposed(
            design=design, spec=self.spec, axes=tuple(self.axes),
            lo=float(y_train.min() - CLIP_MULTIPLIER * spread),
            hi=float(y_train.max() + CLIP_MULTIPLIER * spread),
            gen9_compat=self.gen9_compat, context_replication=self.context_replication,
            membership_source=self.membership_source,
            cohort_membership=self.cohort_membership)
        weights = group_balanced_weights(train["ecfp_cluster"])
        static = design.transform(train)

        base = _forest(context.model_seed, n_estimators=self.n_estimators,
                       max_features=self.max_features,
                       min_samples_leaf=self.min_samples_leaf)
        base.fit(static, y_train, sample_weight=weights)

        membership = fitted._membership(train)
        curve = _curve_of(train, membership, tuple(self.axes))
        centred = y_train - _curve_means(curve, y_train)
        usable = np.isfinite(centred)
        shape = _forest(context.model_seed + 1, n_estimators=self.n_estimators,
                        max_features=self.shape_max_features or self.max_features,
                        min_samples_leaf=self.min_samples_leaf)
        x_shape = np.hstack([static, fitted._context(train, membership)])
        shape.fit(x_shape[usable], centred[usable], sample_weight=weights[usable])

        fitted.base, fitted.shape = base, shape
        self.diagnostics.append({"arm": self.name, "split_seed": context.fold.seed,
                                 "fold": context.fold.fold, "n_train": int(len(train)),
                                 "n_shape_train": int(usable.sum())})
        return fitted


# --------------------------------------------------------------------------- #
# 3. Explicit level + zero-mean shape
# --------------------------------------------------------------------------- #

@dataclass
class _FittedLevelShape(_FittedBase):
    level: object = None
    shape: object = None

    def predict(self, query: pd.DataFrame) -> np.ndarray:
        membership = self._membership(query)
        static = self.design.transform(query)
        context = self._context(query, membership)
        level_row = np.asarray(self.level.predict(static), dtype=float)
        curve = _curve_of(query, membership, self.axes)
        # The level is a property of the curve, so it is averaged over the curve —
        # the architecture's promise is `mean_i(y_hat_i) = mu`, and a row-varying
        # level would break it silently.
        level = _curve_means(curve, level_row)
        level = np.where(np.isfinite(level), level, level_row)
        shape = np.asarray(self.shape.predict(np.hstack([static, context])), dtype=float)
        centred = shape - np.where(np.isfinite(_curve_means(curve, shape)),
                                   _curve_means(curve, shape), shape)
        return self._clip(level + centred)


@dataclass
class LevelShapeModel:
    """``y_hat = mu(curve) + s``, with ``mean_over_curve(s) = 0`` by construction.

    The level head is trained on **curve-mean targets** — one number per curve,
    broadcast to its rows — so it is asked only for the quantity a single
    measurement would supply, and never has to trade level accuracy against shape.
    Rows on no curve carry their own target, which is what "the curve is this one
    point" means.

    This is the arm that decides Phase 2's question.  If it matches
    :class:`RecomposedModel`, the recomposition is an architecture and the
    post-hoc step can be retired; if it loses, averaging the monolith's row
    predictions is a *better* level estimate than predicting the level directly,
    and gen7's "the level is not chemistry" finding has struck again.
    """

    name: str = "GEN10_LEVEL_SHAPE"
    representation_name: str = "GEN9"
    window: str = "primary"
    blocks: tuple[str, ...] = FROZEN_BLOCKS
    add_indicator: bool = True
    n_estimators: int = 400
    max_features: object = 0.30
    min_samples_leaf: int = 2
    axes: tuple[str, ...] = DEFAULT_AXES
    context_replication: int = 1
    membership_source: str = "query"
    cohort_membership: pd.DataFrame | None = None
    diagnostics: list = field(default_factory=list)

    @property
    def spec(self) -> Representation:
        return representation(self.representation_name, window=self.window)

    def fit(self, train: pd.DataFrame, y_train: np.ndarray,
            context: FoldContext) -> _FittedLevelShape:
        design = DesignBuilder(blocks=tuple(self.blocks),
                               add_indicator=self.add_indicator).fit(context.cohort, train)
        spread = float(y_train.max() - y_train.min())
        fitted = _FittedLevelShape(
            design=design, spec=self.spec, axes=tuple(self.axes),
            lo=float(y_train.min() - CLIP_MULTIPLIER * spread),
            hi=float(y_train.max() + CLIP_MULTIPLIER * spread),
            context_replication=self.context_replication,
            membership_source=self.membership_source,
            cohort_membership=self.cohort_membership)
        weights = group_balanced_weights(train["ecfp_cluster"])
        static = design.transform(train)
        membership = fitted._membership(train)
        curve = _curve_of(train, membership, tuple(self.axes))
        means = _curve_means(curve, y_train)
        level_target = np.where(np.isfinite(means), means, y_train)

        level = _forest(context.model_seed, n_estimators=self.n_estimators,
                        max_features=self.max_features,
                        min_samples_leaf=self.min_samples_leaf)
        level.fit(static, level_target, sample_weight=weights)

        centred = y_train - level_target
        shape = _forest(context.model_seed + 1, n_estimators=self.n_estimators,
                        max_features=self.max_features,
                        min_samples_leaf=self.min_samples_leaf)
        shape.fit(np.hstack([static, fitted._context(train, membership)]), centred,
                  sample_weight=weights)
        fitted.level, fitted.shape = level, shape
        self.diagnostics.append({"arm": self.name, "split_seed": context.fold.seed,
                                 "fold": context.fold.fold,
                                 "n_on_curve": int(np.isfinite(means).sum())})
        return fitted


# --------------------------------------------------------------------------- #
# 4. Additive shape correction on the monolith
# --------------------------------------------------------------------------- #

@dataclass
class _FittedResidualShape(_FittedBase):
    base: object = None
    correction: object = None

    def predict(self, query: pd.DataFrame) -> np.ndarray:
        membership = self._membership(query)
        static = self.design.transform(query)
        base = self._clip(np.asarray(self.base.predict(static), dtype=float))
        context = self._context(query, membership)
        delta = np.asarray(self.correction.predict(np.hstack([static, context])), dtype=float)
        curve = _curve_of(query, membership, self.axes)
        means = _curve_means(curve, delta)
        centred = delta - np.where(np.isfinite(means), means, delta)
        return self._clip(base + centred)


@dataclass
class ResidualShapeModel:
    """Keep the monolith's shape; add a mean-preserving correction to it.

    The difference from :class:`RecomposedModel` is one word: *add* rather than
    *replace*.  The recomposition throws away whatever within-curve variation the
    monolith had; this keeps it and learns what is left.  Since the correction is
    fitted on the **curve-centred** residual and re-centred at prediction time,
    the level cannot enter — which is the guard gen8's two-stage residual arm
    lacked when it learned level error and lost.
    """

    name: str = "GEN10_RESIDUAL_SHAPE"
    representation_name: str = "GEN9"
    window: str = "primary"
    blocks: tuple[str, ...] = FROZEN_BLOCKS
    add_indicator: bool = True
    n_estimators: int = 400
    max_features: object = 0.30
    min_samples_leaf: int = 2
    axes: tuple[str, ...] = DEFAULT_AXES
    context_replication: int = 1
    membership_source: str = "query"
    cohort_membership: pd.DataFrame | None = None
    #: ``True``: the correction target is the base model's **cross-fitted** residual
    #: (inner chemotype folds).  ``False``: its in-sample residual — the trap
    #: control, expected to collapse toward the frozen model.
    cross_fit: bool = True
    inner_splits: int = 3
    diagnostics: list = field(default_factory=list)

    @property
    def spec(self) -> Representation:
        return representation(self.representation_name, window=self.window)

    def fit(self, train: pd.DataFrame, y_train: np.ndarray,
            context: FoldContext) -> _FittedResidualShape:
        design = DesignBuilder(blocks=tuple(self.blocks),
                               add_indicator=self.add_indicator).fit(context.cohort, train)
        spread = float(y_train.max() - y_train.min())
        fitted = _FittedResidualShape(
            design=design, spec=self.spec, axes=tuple(self.axes),
            lo=float(y_train.min() - CLIP_MULTIPLIER * spread),
            hi=float(y_train.max() + CLIP_MULTIPLIER * spread),
            context_replication=self.context_replication,
            membership_source=self.membership_source,
            cohort_membership=self.cohort_membership)
        weights = group_balanced_weights(train["ecfp_cluster"])
        static = design.transform(train)
        base = _forest(context.model_seed, n_estimators=self.n_estimators,
                       max_features=self.max_features,
                       min_samples_leaf=self.min_samples_leaf)
        base.fit(static, y_train, sample_weight=weights)

        membership = fitted._membership(train)
        curve = _curve_of(train, membership, tuple(self.axes))
        forest_kwargs = dict(n_estimators=self.n_estimators, max_features=self.max_features,
                             min_samples_leaf=self.min_samples_leaf)
        if self.cross_fit:
            first_stage = cross_fitted_predictions(
                static, y_train, weights, train["tanimoto_cluster"].astype(str).to_numpy(),
                seed=context.model_seed, n_splits=self.inner_splits,
                forest_kwargs=forest_kwargs)
        else:
            # `stabilise` is load-bearing here, not hygiene: see ROUND_DECIMALS.
            first_stage = stabilise(base.predict(static))
        centred_truth = y_train - _curve_means(curve, y_train)
        centred_base = first_stage - _curve_means(curve, first_stage)
        target = centred_truth - centred_base
        usable = np.isfinite(target)
        correction = _forest(context.model_seed + 1, n_estimators=self.n_estimators,
                             max_features=self.max_features,
                             min_samples_leaf=self.min_samples_leaf)
        x = np.hstack([static, fitted._context(train, membership)])
        correction.fit(x[usable], target[usable], sample_weight=weights[usable])
        fitted.base, fitted.correction = base, correction
        self.diagnostics.append({"arm": self.name, "split_seed": context.fold.seed,
                                 "fold": context.fold.fold, "cross_fit": self.cross_fit,
                                 "n_correction_train": int(usable.sum()),
                                 "correction_target_sd": float(np.std(target[usable]))})
        return fitted


# --------------------------------------------------------------------------- #
# 5. Two-branch: static chemistry + a context branch the chemistry cannot drown
# --------------------------------------------------------------------------- #

#: Blocks the context branch may read.  No fingerprint: the point of the branch
#: is that 2,048 ECFP bits cannot outvote a handful of condition columns at a
#: split, because they are not in the room.
CONTEXT_BLOCKS: tuple[str, ...] = ("METAL", "COND", "MASSACTION")


@dataclass
class _FittedTwoBranch(_FittedBase):
    base: object = None
    context_design: DesignBuilder | None = None
    branch: object = None
    interaction: str = "none"

    def _branch_input(self, query: pd.DataFrame, membership: pd.DataFrame,
                      base: np.ndarray) -> np.ndarray:
        parts = [self.context_design.transform(query), self._context(query, membership)]
        if self.interaction == "base_prediction":
            parts.append(stabilise(base)[:, None])
        return np.hstack(parts)

    def predict(self, query: pd.DataFrame) -> np.ndarray:
        membership = self._membership(query)
        base = self._clip(np.asarray(self.base.predict(self.design.transform(query)),
                                     dtype=float))
        delta = np.asarray(self.branch.predict(self._branch_input(query, membership, base)),
                           dtype=float)
        curve = _curve_of(query, membership, self.axes)
        means = _curve_means(curve, delta)
        centred = delta - np.where(np.isfinite(means), means, delta)
        return self._clip(base + centred)


@dataclass
class TwoBranchModel:
    """``y_hat = h_static(chemistry, conditions) + h_context(conditions, design)``.

    Phase 2C of the brief.  ``h_static`` is the frozen monolith.  ``h_context`` is
    a second forest that sees **only** the condition blocks and the design-relative
    columns — no fingerprint, no recovered variables — fitted on the cross-fitted,
    curve-centred residual of ``h_static`` and re-centred per curve at prediction,
    so it can move shape and cannot move level.

    ``interaction="base_prediction"`` hands the context branch the static branch's
    own prediction as one more column: the smallest interaction form that lets the
    correction depend on *what the chemistry predicted* without handing the branch
    the chemistry itself.
    """

    name: str = "GEN10_TWO_BRANCH"
    representation_name: str = "GEN9"
    window: str = "primary"
    blocks: tuple[str, ...] = FROZEN_BLOCKS
    context_blocks: tuple[str, ...] = CONTEXT_BLOCKS
    add_indicator: bool = True
    n_estimators: int = 400
    max_features: object = 0.30
    branch_max_features: object = 0.5
    min_samples_leaf: int = 2
    axes: tuple[str, ...] = DEFAULT_AXES
    interaction: str = "none"
    inner_splits: int = 3
    context_replication: int = 1
    membership_source: str = "query"
    cohort_membership: pd.DataFrame | None = None
    diagnostics: list = field(default_factory=list)

    @property
    def spec(self) -> Representation:
        return representation(self.representation_name, window=self.window)

    def fit(self, train: pd.DataFrame, y_train: np.ndarray,
            context: FoldContext) -> _FittedTwoBranch:
        if self.interaction not in ("none", "base_prediction"):
            raise ValueError(f"unknown interaction {self.interaction!r}")
        design = DesignBuilder(blocks=tuple(self.blocks),
                               add_indicator=self.add_indicator).fit(context.cohort, train)
        context_design = DesignBuilder(blocks=tuple(self.context_blocks),
                                       add_indicator=self.add_indicator).fit(
            context.cohort, train)
        spread = float(y_train.max() - y_train.min())
        fitted = _FittedTwoBranch(
            design=design, spec=self.spec, axes=tuple(self.axes),
            lo=float(y_train.min() - CLIP_MULTIPLIER * spread),
            hi=float(y_train.max() + CLIP_MULTIPLIER * spread),
            context_replication=self.context_replication,
            membership_source=self.membership_source,
            cohort_membership=self.cohort_membership,
            context_design=context_design, interaction=self.interaction)
        weights = group_balanced_weights(train["ecfp_cluster"])
        static = design.transform(train)
        forest_kwargs = dict(n_estimators=self.n_estimators, max_features=self.max_features,
                             min_samples_leaf=self.min_samples_leaf)
        base = _forest(context.model_seed, **forest_kwargs)
        base.fit(static, y_train, sample_weight=weights)
        first_stage = cross_fitted_predictions(
            static, y_train, weights, train["tanimoto_cluster"].astype(str).to_numpy(),
            seed=context.model_seed, n_splits=self.inner_splits, forest_kwargs=forest_kwargs)

        membership = fitted._membership(train)
        curve = _curve_of(train, membership, tuple(self.axes))
        target = ((y_train - _curve_means(curve, y_train))
                  - (first_stage - _curve_means(curve, first_stage)))
        usable = np.isfinite(target)
        x = fitted._branch_input(train, membership, first_stage)
        branch = _forest(context.model_seed + 1, n_estimators=self.n_estimators,
                         max_features=self.branch_max_features,
                         min_samples_leaf=self.min_samples_leaf)
        branch.fit(x[usable], target[usable], sample_weight=weights[usable])
        fitted.base, fitted.branch = base, branch
        self.diagnostics.append({"arm": self.name, "split_seed": context.fold.seed,
                                 "fold": context.fold.fold, "interaction": self.interaction,
                                 "n_branch_features": int(x.shape[1]),
                                 "n_branch_train": int(usable.sum()),
                                 "branch_target_sd": float(np.std(target[usable]))})
        return fitted


# --------------------------------------------------------------------------- #
# gen7 adapter
# --------------------------------------------------------------------------- #

@dataclass
class ContenderAdapter:
    """Wrap a :class:`QueryModel` in gen7's ``fit_predict`` so it can be evaluated.

    The evaluation contract is unchanged — same cohort, same folds, same seeds,
    same metric module — and the query handed to ``predict`` is exactly the test
    frame gen7 hands a contender.  That is what makes a gen10 arm's OOF parquet
    comparable, row for row, with a gen9 one.
    """

    model: object
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = getattr(self.model, "name", type(self.model).__name__)

    @property
    def diagnostics(self):
        return getattr(self.model, "diagnostics", [])

    def fit_predict(self, train: pd.DataFrame, y_train: np.ndarray,
                    test: pd.DataFrame, context: FoldContext) -> np.ndarray:
        return self.model.fit(train, y_train, context).predict(test)
