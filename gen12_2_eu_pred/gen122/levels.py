"""The extractant-specific intrinsic level ``alpha_i``, and the five ways of defining it.

The generation models ``y_ij = alpha_i + g(i, c_ij) + eps_ij``: an extractant-specific
level and a within-extractant response to conditions.  ``alpha_i`` is not given by the
data — it has to be *defined* — and the definition changes what the level model is
being asked to predict.  Five candidates are implemented here and compared on training
rows only, before any level model is fitted; ``PRE_REGISTRATION.md`` records which one
was frozen as primary and why.

Two rules hold for every definition and are enforced by construction:

* **A held-out extractant's own targets never help predict its level.**  A level model
  is fitted on the training extractants of that fold; the held-out extractant enters
  only as a set of structural features.  Its observed level is the *evaluation truth*,
  in the same sense that ``log_D`` is the evaluation truth for a query row.
* **Anything fitted is fitted inside the fold.**  The condition-only model that the
  adjusted definitions subtract is fitted on that fold's training rows and, for the
  training extractants themselves, is *cross-fitted* over training chemotypes.  Without
  that cross-fit a training extractant's level would be measured against a condition
  model that had already seen it while a held-out extractant's is not, which is exactly
  the two-stage residual trap the repository measured under a chemotype hold-out.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from gen12eu.preprocess import FoldPreprocessor, extractant_balanced_weights
from gen12eu.splits import N_INNER_SPLITS
from lanthanide_separation.gen6.cohorts import seeded_group_kfold

TARGET = "log_D"
CONDITION_BLOCKS: tuple[str, ...] = ("COND", "MASSACT")

#: The five candidate definitions, in the order the pre-registration lists them.
DEFINITIONS: tuple[str, ...] = (
    "LVL_MEAN",          # 1. raw per-extractant mean of log_D
    "LVL_MEDIAN",        # 2. robust per-extractant median
    "LVL_FE_INTERCEPT",  # 3. condition-adjusted intercept, linear fixed-effect fit in training
    "LVL_SHRUNK",        # 4. hierarchical / James-Stein shrunken intercept
    "LVL_COND_RESIDUAL", # 5. residual level after a global condition-only model
)


# --------------------------------------------------------------------------- #
# The condition-only model the adjusted definitions subtract
# --------------------------------------------------------------------------- #

def _condition_columns(blocks: dict) -> tuple[str, ...]:
    out: list[str] = []
    for name in CONDITION_BLOCKS:
        out.extend(blocks[name])
    return tuple(dict.fromkeys(out))


def _fit_condition_model(train: pd.DataFrame, y: np.ndarray, columns, seed: int):
    from sklearn.ensemble import ExtraTreesRegressor
    pre = FoldPreprocessor(columns).fit(train)
    model = ExtraTreesRegressor(n_estimators=300, max_features=0.30, min_samples_leaf=2,
                                random_state=seed, n_jobs=-1)
    model.fit(pre.transform(train), y,
              sample_weight=extractant_balanced_weights(train["extractant"]))
    return pre, model


def condition_only_predictions(train: pd.DataFrame, test: pd.DataFrame, blocks: dict, *,
                               seed: int, n_inner: int = N_INNER_SPLITS,
                               inner_seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """``(g_hat on training rows, cross-fitted; g_hat on test rows, from the full fit)``.

    The training predictions are cross-fitted over training *chemotypes*, so a training
    row's condition prediction comes from a model that never saw its chemistry — the
    same situation a held-out row is in.  The test predictions come from the model
    fitted on all training rows, which is the deployable one.
    """
    columns = _condition_columns(blocks)
    y_train = train[TARGET].to_numpy(dtype=float)
    groups = train["chemotype"].astype(str).to_numpy()
    oof = np.empty(len(train), dtype=float)
    oof[:] = np.nan
    for inner_train, inner_test in seeded_group_kfold(groups, n_inner, inner_seed or seed):
        pre, model = _fit_condition_model(train.iloc[inner_train],
                                          y_train[inner_train], columns, seed)
        oof[inner_test] = model.predict(pre.transform(train.iloc[inner_test]))
    if not np.isfinite(oof).all():
        raise AssertionError("a training row received no cross-fitted condition prediction")
    pre, model = _fit_condition_model(train, y_train, columns, seed)
    return oof, model.predict(pre.transform(test))


# --------------------------------------------------------------------------- #
# The definitions
# --------------------------------------------------------------------------- #

def _per_extractant(frame: pd.DataFrame, values: np.ndarray, how: str) -> pd.Series:
    table = pd.DataFrame({"e": frame["extractant"].astype(str).to_numpy(), "v": values})
    return table.groupby("e")["v"].agg(how)


def shrinkage_ratio(frame: pd.DataFrame, values: np.ndarray) -> dict:
    """``sigma^2 / tau^2`` from the rows given — within-extractant noise over between spread.

    ``tau^2`` is the variance of the per-extractant means and ``sigma^2`` the mean
    within-extractant variance, both measured on the rows handed in.  Extractants with a
    single row contribute no within-variance and are excluded from ``sigma^2`` only.
    """
    table = pd.DataFrame({"e": frame["extractant"].astype(str).to_numpy(), "v": values})
    per = table.groupby("e")["v"]
    tau2 = float(per.mean().var(ddof=1))
    within = per.var(ddof=1).dropna()
    sigma2 = float(within.mean()) if len(within) else tau2
    return {"tau2": tau2, "sigma2": sigma2,
            "ratio": float(sigma2 / tau2) if tau2 > 1e-9 else 1.0,
            "n_extractants": int(per.ngroups),
            "n_with_within_variance": int(len(within))}


def _fixed_effect_intercepts(frame: pd.DataFrame, y: np.ndarray, axes: Sequence[str],
                             *, ridge: float = 1.0) -> pd.Series:
    """Per-extractant intercepts of ``y ~ extractant + shared linear condition slopes``.

    A single pooled ridge regression with one dummy per extractant and one shared slope
    per continuous condition axis.  The intercepts are the condition-adjusted levels; the
    shared slopes are what "adjusted" means.  Ridge only on the slopes, never on the
    dummies, so a one-cell extractant keeps its own intercept rather than being shrunk.
    """
    extractants = frame["extractant"].astype(str).to_numpy()
    names = sorted(set(extractants))
    index = {n: i for i, n in enumerate(names)}
    axis_values = []
    for axis in axes:
        column = pd.to_numeric(frame[axis], errors="coerce").to_numpy(dtype=float)
        median = np.nanmedian(column) if np.isfinite(column).any() else 0.0
        column = np.where(np.isfinite(column), column, median)
        axis_values.append(column - column.mean())
    design = np.zeros((len(frame), len(names) + len(axis_values)), dtype=float)
    for row, name in enumerate(extractants):
        design[row, index[name]] = 1.0
    for j, column in enumerate(axis_values):
        design[:, len(names) + j] = column
    penalty = np.zeros(design.shape[1], dtype=float)
    penalty[len(names):] = ridge
    beta = np.linalg.solve(design.T @ design + np.diag(penalty), design.T @ y)
    return pd.Series(beta[:len(names)], index=names)


@dataclass
class LevelTargets:
    """Level targets for one fold, under one definition."""

    definition: str
    train: pd.Series                      # index extractant -> alpha, training extractants
    test: pd.Series                       # index extractant -> alpha, held-out extractants
    train_row_residual: np.ndarray        # y_ij - alpha_i on the training rows
    diagnostics: dict = field(default_factory=dict)


def build_level_targets(train: pd.DataFrame, test: pd.DataFrame, blocks: dict, *,
                        definition: str, seed: int, inner_seed: int = 0) -> LevelTargets:
    """Level targets for one fold.

    The *training* levels are what a level model is fitted on.  The *test* levels are
    the evaluation truth for the held-out extractants and are computed from their own
    held-out rows — never from a model, and never used as an input to anything.
    """
    if definition not in DEFINITIONS:
        raise KeyError(f"unknown level definition {definition!r}; have {DEFINITIONS}")
    y_train = train[TARGET].to_numpy(dtype=float)
    y_test = test[TARGET].to_numpy(dtype=float)
    diagnostics: dict = {"definition": definition}

    if definition in ("LVL_MEAN", "LVL_MEDIAN", "LVL_SHRUNK"):
        how = "median" if definition == "LVL_MEDIAN" else "mean"
        level_train = _per_extractant(train, y_train, how)
        level_test = _per_extractant(test, y_test, how)
        if definition == "LVL_SHRUNK":
            stats = shrinkage_ratio(train, y_train)
            grand = float(level_train.mean())
            counts = train["extractant"].astype(str).value_counts()
            weight = counts.reindex(level_train.index).astype(float)
            weight = weight / (weight + stats["ratio"])
            level_train = grand + weight * (level_train - grand)
            diagnostics["shrinkage"] = stats
            diagnostics["grand_mean"] = grand
            # The *evaluation* target is never shrunk: shrinking the truth toward the
            # training mean would flatter every model by moving the target it is scored
            # against toward the thing the model already predicts.
            diagnostics["test_target_shrunk"] = False

    elif definition == "LVL_COND_RESIDUAL":
        g_train, g_test = condition_only_predictions(train, test, blocks, seed=seed,
                                                     inner_seed=inner_seed)
        level_train = _per_extractant(train, y_train - g_train, "mean")
        level_test = _per_extractant(test, y_test - g_test, "mean")
        diagnostics["condition_model"] = "ExtraTrees on COND+MASSACT, cross-fitted in training"
        diagnostics["g_train_sd"] = float(np.std(g_train))

    elif definition == "LVL_FE_INTERCEPT":
        axes = [c for c in blocks["MASSACT"] if c.startswith("massact__log10_")]
        level_train = _fixed_effect_intercepts(train, y_train, axes)
        # The held-out extractants get their intercepts from the *same shared slopes*,
        # refitted on the test rows with the training slopes held fixed, so the
        # definition is the same quantity on both sides of the split.
        slopes = _shared_slopes(train, y_train, axes)
        adjusted = y_test - _apply_slopes(test, axes, slopes, reference=train)
        level_test = _per_extractant(test, adjusted, "mean")
        diagnostics["axes"] = list(axes)
        diagnostics["slopes"] = {a: float(v) for a, v in zip(axes, slopes)}
    else:  # pragma: no cover - guarded above
        raise KeyError(definition)

    residual = y_train - train["extractant"].astype(str).map(level_train).to_numpy(dtype=float)
    return LevelTargets(definition=definition, train=level_train.astype(float),
                        test=level_test.astype(float), train_row_residual=residual,
                        diagnostics=diagnostics)


def _shared_slopes(frame: pd.DataFrame, y: np.ndarray, axes: Sequence[str],
                   *, ridge: float = 1.0) -> np.ndarray:
    extractants = frame["extractant"].astype(str).to_numpy()
    names = sorted(set(extractants))
    index = {n: i for i, n in enumerate(names)}
    columns = []
    for axis in axes:
        column = pd.to_numeric(frame[axis], errors="coerce").to_numpy(dtype=float)
        median = np.nanmedian(column) if np.isfinite(column).any() else 0.0
        column = np.where(np.isfinite(column), column, median)
        columns.append(column - column.mean())
    design = np.zeros((len(frame), len(names) + len(columns)), dtype=float)
    for row, name in enumerate(extractants):
        design[row, index[name]] = 1.0
    for j, column in enumerate(columns):
        design[:, len(names) + j] = column
    penalty = np.zeros(design.shape[1], dtype=float)
    penalty[len(names):] = ridge
    beta = np.linalg.solve(design.T @ design + np.diag(penalty), design.T @ y)
    return beta[len(names):]


def _apply_slopes(frame: pd.DataFrame, axes: Sequence[str], slopes: np.ndarray,
                  *, reference: pd.DataFrame) -> np.ndarray:
    """Condition part of the fixed-effect model, centred on the *training* means."""
    out = np.zeros(len(frame), dtype=float)
    for axis, slope in zip(axes, slopes):
        reference_column = pd.to_numeric(reference[axis], errors="coerce").to_numpy(dtype=float)
        reference_median = np.nanmedian(reference_column) if np.isfinite(reference_column).any() else 0.0
        centre = np.nanmean(np.where(np.isfinite(reference_column), reference_column,
                                     reference_median))
        column = pd.to_numeric(frame[axis], errors="coerce").to_numpy(dtype=float)
        column = np.where(np.isfinite(column), column, reference_median)
        out += slope * (column - centre)
    return out
