"""Contenders: everything that maps (train, y, test) to a prediction vector.

One file, one interface, so the leaderboard compares architectures and not
plumbing.  Each class carries the hyperparameters it was *frozen* with; where a
hyperparameter is searched, the search happens on an inner split of the training
fold and never touches the outer test rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from ..levels import (
    LEVEL_TARGET_COLUMN, LevelForestParameters, LevelRegressor, _as_float_frame,
    group_balanced_weights,
)
from .harness import FoldContext


# --------------------------------------------------------------------------- #
# Baselines that use no chemistry at all — the floors every model must clear
# --------------------------------------------------------------------------- #

@dataclass
class GlobalMean:
    """Predict the training mean.  R² = 0 by construction; the absolute floor."""

    name: str = "NULL_global_mean"

    def fit_predict(self, train, y_train, test, context: FoldContext) -> np.ndarray:
        return np.full(len(test), float(np.mean(y_train)))


@dataclass
class MetalConditionMean:
    """Predict from metal + acid concentration only — no ligand information.

    The honest "what does knowing the experiment alone buy you" floor.
    """

    name: str = "NULL_metal_cond"

    def fit_predict(self, train, y_train, test, context: FoldContext) -> np.ndarray:
        from sklearn.ensemble import ExtraTreesRegressor
        columns = context.cohort.block_columns(("METAL", "COND"))
        model = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2,
                                    random_state=context.model_seed, n_jobs=-1)
        model.fit(_as_float_frame(train, columns).fillna(-999.0), y_train,
                  sample_weight=group_balanced_weights(train["ecfp_cluster"]))
        return model.predict(_as_float_frame(test, columns).fillna(-999.0))


# --------------------------------------------------------------------------- #
# The gen6 champion, reached through the gen6 code path
# --------------------------------------------------------------------------- #

@dataclass
class LevelTree:
    """gen5/gen6's ``LevelRegressor`` — the reference implementation, unchanged.

    ``arm`` selects the feature blocks by their gen5 name, so
    ``LevelTree(arm="MC_lig2d_ext_massaction")`` *is* the gen6 EXPANDED champion
    and must reproduce its number on identical folds.
    """

    arm: str = "MC_lig2d_ext_massaction"
    learner: str = "extratrees"
    n_estimators: int = 400
    max_features: float = 0.30
    min_samples_leaf: int = 2
    weighting: str = "cluster"
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            suffix = "" if self.learner == "extratrees" else f"_{self.learner}"
            self.name = f"TREE_{self.arm}{suffix}"

    def fit_predict(self, train, y_train, test, context: FoldContext) -> np.ndarray:
        columns = context.cohort.arm_columns(self.arm)
        params = LevelForestParameters(
            n_estimators=self.n_estimators, max_features=self.max_features,
            min_samples_leaf=self.min_samples_leaf, random_state=context.model_seed,
            n_jobs=-1, learner=self.learner)
        groups = train["ecfp_cluster"] if self.weighting == "cluster" else None
        model = LevelRegressor(columns, params).fit(train, y_train, groups=groups)
        if hasattr(model.pipeline.named_steps.get("model"), "estimators_"):
            forest = model.pipeline.named_steps["model"]
            x = model.pipeline[:-1].transform(_as_float_frame(test, columns))
            spread = np.std([e.predict(x) for e in forest.estimators_], axis=0)
            context.extras["prediction_sd"] = spread
        return model.predict(test)


# --------------------------------------------------------------------------- #
# Generic tabular contender — one wrapper, every off-the-shelf learner
# --------------------------------------------------------------------------- #

def _fit_frame(cohort, frame: pd.DataFrame, blocks: Sequence[str]) -> pd.DataFrame:
    return _as_float_frame(frame, cohort.block_columns(tuple(blocks)))


@dataclass
class Tabular:
    """Any sklearn-style regressor on any combination of gen5 feature blocks.

    ``estimator`` is a factory taking the fold's model seed, so every learner is
    seeded through the same formula and two learners never accidentally share a
    random state.  ``impute``/``scale`` are explicit rather than inferred: a tree
    wants neither, a kernel wants both, and silently guessing is how a
    "representation" result becomes a "preprocessing" result.

    Preprocessing is fitted on the **training fold only** — the imputer's medians
    and the scaler's moments never see a test row.
    """

    blocks: tuple[str, ...]
    estimator: Callable[[int], object]
    name: str
    impute: bool = True
    scale: bool = False
    weighting: str = "cluster"
    supports_sample_weight: bool = True
    clip_to_training_range: bool = True
    #: Blocks to compress with PCA before the learner sees them.  A 768-dimensional
    #: pretrained embedding over 152 distinct ligands is mostly empty directions; a
    #: tree that samples 30 % of columns will pick noise almost every split.  The
    #: PCA is fitted **inside the training fold**, so it is not merely
    #: target-free but also blind to the held-out chemistry — the stricter of the
    #: two available guarantees, and the one that survives an adversarial read.
    pca_blocks: tuple[str, ...] = ()
    pca_components: int = 32
    #: Append a binary "this value was missing" column per partially-missing feature.
    #: gen5's ``LevelRegressor`` does this (``SimpleImputer(add_indicator=True)``) and
    #: it is worth **more macro MAE than every ligand descriptor combined** — 0.072
    #: versus 0.09 — so it cannot be left as an unexamined default.  Two of the
    #: indicators are publication reporting conventions rather than chemistry
    #: (``cond__contact_time_min`` is absent in 40.3 % of rows,
    #: ``cond__metal_concentration_mM`` in 25.6 %), which is why the suite runs the
    #: ligand-only and condition-only variants separately.
    add_indicator: bool = False
    #: Restrict the indicators to columns matching these prefixes (empty = all).
    indicator_prefixes: tuple[str, ...] = ()

    def fit_predict(self, train, y_train, test, context: FoldContext) -> np.ndarray:
        from sklearn.decomposition import PCA
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        plain = tuple(b for b in self.blocks if b not in set(self.pca_blocks))
        columns = context.cohort.block_columns(plain) if plain else ()
        if self.pca_blocks:
            pca_columns = context.cohort.block_columns(tuple(self.pca_blocks))
            raw_tr = _as_float_frame(train, pca_columns).to_numpy()
            raw_te = _as_float_frame(test, pca_columns).to_numpy()
            keep = ~np.all(np.isnan(raw_tr), axis=0)
            raw_tr, raw_te = raw_tr[:, keep], raw_te[:, keep]
            median = np.nan_to_num(np.nanmedian(raw_tr, axis=0))
            raw_tr = np.where(np.isfinite(raw_tr), raw_tr, median)
            raw_te = np.where(np.isfinite(raw_te), raw_te, median)
            scaler = StandardScaler().fit(raw_tr)
            n_components = min(self.pca_components, raw_tr.shape[1],
                               max(2, np.unique(train["extractant"]).size - 1))
            pca = PCA(n_components=n_components, random_state=context.model_seed)
            reduced_tr = pca.fit_transform(scaler.transform(raw_tr))
            reduced_te = pca.transform(scaler.transform(raw_te))
        else:
            reduced_tr = reduced_te = None
        if columns:
            x_train = _as_float_frame(train, columns).to_numpy()
            x_test = _as_float_frame(test, columns).to_numpy()
            # Drop columns with no observed value in training: they carry nothing
            # and break HGB's binner and any scaler.
            keep = ~np.all(np.isnan(x_train), axis=0)
            x_train, x_test = x_train[:, keep], x_test[:, keep]
        else:
            x_train = np.empty((len(train), 0))
            x_test = np.empty((len(test), 0))
        if self.impute:
            imputer = SimpleImputer(strategy="median",
                                    add_indicator=self.add_indicator).fit(x_train)
            x_train, x_test = imputer.transform(x_train), imputer.transform(x_test)
        elif self.add_indicator:
            # a learner with native NaN support still gets explicit indicators when asked
            flags_train = np.isnan(x_train).astype(float)
            keep_flag = flags_train.std(0) > 0
            x_train = np.hstack([x_train, flags_train[:, keep_flag]])
            x_test = np.hstack([x_test, np.isnan(x_test).astype(float)[:, keep_flag]])
        if self.scale and x_train.shape[1]:
            scaler = StandardScaler().fit(x_train)
            x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
        if reduced_tr is not None:
            x_train = np.hstack([x_train, reduced_tr])
            x_test = np.hstack([x_test, reduced_te])
        if x_train.shape[1] == 0:
            # Every feature was empty in this training fold — a real state under a
            # chemotype hold-out that removes 67 % of rows.  Fall back to the
            # training mean and say so, rather than raising and vanishing from the
            # leaderboard.
            context.extras["degenerate_fold"] = np.ones(len(test))
            return np.full(len(test), float(np.mean(y_train)))
        model = self.estimator(context.model_seed)
        weights = (group_balanced_weights(train["ecfp_cluster"])
                   if self.weighting == "cluster" else None)
        if weights is not None and self.supports_sample_weight:
            model.fit(x_train, y_train, sample_weight=weights)
        else:
            model.fit(x_train, y_train)
        pred = np.asarray(model.predict(x_test), dtype=float)
        if self.clip_to_training_range:
            span = float(y_train.max() - y_train.min())
            pred = np.clip(pred, y_train.min() - 0.5 * span, y_train.max() + 0.5 * span)
        return pred


# --------------------------------------------------------------------------- #
# The learner zoo
# --------------------------------------------------------------------------- #

CHAMPION_BLOCKS: tuple[str, ...] = ("METAL", "COND", "LIG2D_EXT", "MASSACTION")
DONOR_BLOCKS: tuple[str, ...] = ("METAL", "COND", "DONORS")
COMPACT_BLOCKS: tuple[str, ...] = ("METAL", "COND", "DONORS", "PHYSCHEM", "MASSACTION")
RICH_BLOCKS: tuple[str, ...] = ("METAL", "COND", "PHYSCHEM", "ECFP", "LIG2D_EXT", "DONORS", "MASSACTION")


def extratrees(seed: int, *, n: int = 400, max_features: float = 0.30, leaf: int = 2):
    from sklearn.ensemble import ExtraTreesRegressor
    return ExtraTreesRegressor(n_estimators=n, max_features=max_features,
                               min_samples_leaf=leaf, random_state=seed, n_jobs=-1)


def random_forest(seed: int, *, n: int = 400, max_features: float = 0.30, leaf: int = 2):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=n, max_features=max_features,
                                 min_samples_leaf=leaf, random_state=seed, n_jobs=-1)


def hist_gb(seed: int, *, loss: str = "absolute_error", lr: float = 0.05, iters: int = 400):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=iters, learning_rate=lr, loss=loss,
                                         min_samples_leaf=5, random_state=seed)


def catboost(seed: int, *, iters: int = 800, depth: int = 6, lr: float = 0.05,
             loss: str = "MAE"):
    from catboost import CatBoostRegressor
    return CatBoostRegressor(iterations=iters, depth=depth, learning_rate=lr,
                             loss_function=loss, random_seed=seed, verbose=0,
                             allow_writing_files=False, thread_count=-1)


#: LightGBM and XGBoost each bundle their own OpenMP runtime, and on macOS a
#: process that has already initialised torch's ``libomp`` deadlocks at 0 % CPU
#: when the second of them starts a thread pool with ``n_jobs=-1`` (observed:
#: ``LRN_lightgbm_huber`` finished, ``LRN_xgboost_mae`` hung for 13 minutes).
#: Both are therefore pinned to a fixed, modest pool.  Measured cost of the pin:
#: 5 s per fold instead of 4 s — nothing, next to a hang.
BOOSTER_THREADS = 4


def lightgbm_(seed: int, *, n: int = 800, leaves: int = 31, lr: float = 0.05,
              objective: str = "l1"):
    import lightgbm as lgb
    return lgb.LGBMRegressor(n_estimators=n, num_leaves=leaves, learning_rate=lr,
                             objective=objective, random_state=seed, n_jobs=BOOSTER_THREADS,
                             verbosity=-1, min_child_samples=10)


def xgboost_(seed: int, *, n: int = 800, depth: int = 6, lr: float = 0.05):
    import xgboost as xgb
    return xgb.XGBRegressor(n_estimators=n, max_depth=depth, learning_rate=lr,
                            objective="reg:absoluteerror", random_state=seed,
                            n_jobs=BOOSTER_THREADS, subsample=0.8, colsample_bytree=0.6,
                            min_child_weight=5, verbosity=0)


def ridge_(seed: int, *, alpha: float = 10.0):
    from sklearn.linear_model import Ridge
    return Ridge(alpha=alpha)


def elasticnet_(seed: int, *, alpha: float = 0.01, l1_ratio: float = 0.5):
    from sklearn.linear_model import ElasticNet
    return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20000, random_state=seed)


def kernel_ridge_rbf(seed: int, *, alpha: float = 1.0, gamma: float | None = None):
    from sklearn.kernel_ridge import KernelRidge
    return KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)


def mlp(seed: int, *, hidden=(256, 128), alpha: float = 1e-2, max_iter: int = 600):
    from sklearn.neural_network import MLPRegressor
    return MLPRegressor(hidden_layer_sizes=hidden, alpha=alpha, max_iter=max_iter,
                        random_state=seed, early_stopping=True, n_iter_no_change=25,
                        validation_fraction=0.15, learning_rate_init=1e-3)
