"""``models/boosted.py`` -- B5 (= M0) and FLAT_CAT: CatBoost on the section 5 feature blocks, tuned in inner folds.

Registered text (``preregistration.md``, sealed 2026-09-15)
-----------------------------------------------------------
* section 5 **B5** -- CatBoost on the metal, extractant and condition blocks (``features.FeatureSet.for_arm("B5")``);
  loss RMSE, eval metric MAE; depth in {4, 6, 8}; learning rate 0.05; ``l2_leaf_reg`` in {3, 10}; <= 3,000 iterations
  with early stopping after 200 rounds on the inner validation fold; ``one_hot_max_size`` 10; ``thread_count`` 2.
  **FLAT_CAT** -- CatBoost on categorical ``g19_metal_state`` and ``extractant_system_key`` plus the condition block
  (``FeatureSet.for_arm("FLAT_CAT")``); B5 settings.  **M0** is B5 (section 6 ladder).
* section 6 -- the ladder starts at M0 = B5; section 5 FLAT_CAT is the H4 flat reference.
* section 7 -- inner folds only; inner grouping by publication group; the inner designs V1 (grouped 3-fold,
  ``folds.source_holdout.inner_folds_V1``), V5 (inner hidden cells, batched by the outer rule,
  ``folds.cell_holdout.inner_cells_V5``; also for V5-P / V5-PAIR / V6 heavy-arm fits) and V2 (leave-one-metal-out over
  3 seeded eligible states, ``folds.metal_holdout.inner_metals_V2``); inner validation never scores
  ``V6_TARGET_ROWS``; **selection criterion** = inner macro MAE with the design's own averaging, configurations within
  0.005 of the best resolved toward the smaller configuration (:func:`select_config`); **compute plan item 1** -- 3 inner
  folds on seed 104729, the first inner fold only on every other seed (:func:`inner_mode_for_seed`), nested per outer
  fold, nothing carried across seeds or folds.
* section 2 "Features" and brief section 12 -- every fitted preprocessing step (scaling, PCA, category vocabularies,
  CatBoost's own CTR statistics) is fitted on the rows passed to ``fit`` (an inner-training set during tuning, the
  outer-training set at the refit); provenance / id / target columns never become features
  (:func:`check_feature_matrix`, run on every matrix).
* section 15 -- model seed ``42 + fold * 1009 + 9,999,991`` (:func:`model_seed_for_fold`).
* section 12 -- intervals come from ``interface.ConformalWrapper``; a bare arm returns NaN intervals and NaN
  ``std_logD``.

Flow of one outer fit (:class:`BoostedArm`)
------------------------------------------
1. :class:`Tuner` builds the inner splits of the outer-training rows with the design's fold-builder function and the run
   seed (``context.seed``), asserts the partition / scoring population of each split, and passes every split through
   ``context.isolation_check`` (the ``fold_isolation_check`` of the design level; :func:`design_isolation_check`) before
   any fit.
2. For each inner split, the feature set is fitted on the inner-training rows only, then for each configuration of the
   grid CatBoost is fitted on the inner-training rows with ``eval_set`` = the split's validation rows (early stopping
   after 200 rounds on MAE, ``use_best_model``); the validation rows are predicted by the shrunk model.
3. Per configuration, the absolute errors of all validation rows of the used splits are averaged per averaging unit
   (V5: hidden cell; V1: the V1 unit -- the row's publication group, or ``REMAINDER`` for a group below 20 rows; V2:
   metal state; V0: publication group) and then over units (macro MAE).  :func:`select_config` applies the tie rule.
4. Outer refit: the feature set is refitted on all outer-training rows and CatBoost is fitted with ``iterations`` = the
   median over the selected configuration's inner fits of the number of trees kept by early stopping
   (:func:`median_iterations`) -- the refit cannot early-stop, it has no validation rows of its own.
5. :meth:`BoostedArm.frozen` returns a clone with the selected configuration and iteration count fixed (no tuning) that
   reuses the refit when it is fitted on the same rows: the arm ``interface.ConformalWrapper`` wraps, so its inner
   calibration fits use the selected configuration instead of re-tuning inside every calibration split.

Where the registration is silent, the reading implemented is listed in :data:`REGISTRATION_CHOICES`.
"""
from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd

from gen19ct.chemistry import support_graph as SG
from gen19ct.data import leakage as LK
from gen19ct.data import load as LOAD
from gen19ct.folds import cell_holdout as CH
from gen19ct.folds import io as FI
from gen19ct.folds import metal_holdout as MH
from gen19ct.folds import registered as FR
from gen19ct.folds import source_holdout as SH
from gen19ct.models import features as F
from gen19ct.models import interface as I

SCHEMA = "gen19.boosted.v1"
ARM_NAMES: tuple[str, ...] = ("B5", "M0", "FLAT_CAT")
#: arm -> ``features.ARM_PRESETS`` key
FEATURE_PRESET: dict[str, str] = {"B5": "B5", "M0": "M0", "FLAT_CAT": "FLAT_CAT"}
TARGET_COL = I.TARGET_COL
ID_COL = I.ID_COL

#: section 7 compute plan item 1: the seed on which the full 3-fold inner design is used
FULL_INNER_SEED = FI.DISCOVERY_SEEDS[0]
INNER_MODES: tuple[str, ...] = ("full", "first")
#: section 7 selection criterion: configurations within this of the best macro MAE are resolved toward the smaller one
TIE_TOLERANCE = 0.005
_TIE_EPS = 1e-12
#: section 15 model seed rule ``42 + fold * 1009 + 9,999,991`` (the gen5 / gen12 / gen13 formula)
MODEL_SEED_BASE, FOLD_SEED_STRIDE, FOLD_SEED_OFFSET = 42, 1009, 9_999_991

DEPTH_GRID: tuple[int, ...] = (4, 6, 8)
L2_GRID: tuple[float, ...] = (3.0, 10.0)

#: extra prediction columns of a boosted arm (after ``interface.PREDICTION_COLUMNS``); never features
BOOSTED_DIAGNOSTIC_COLUMNS: tuple[str, ...] = ("boosted_config", "boosted_n_trees", "boosted_model_seed",
                                               "boosted_tuned_here", "feature_state_digest", "metal_state_in_training",
                                               "system_in_training", "cell_in_training")

#: where the registration is silent, the reading implemented here (reported with every run that uses this module)
REGISTRATION_CHOICES: dict[str, str] = {
    "tie_rule_order": "configurations with inner macro MAE <= best + 0.005 (inclusive, 1e-12 float slack) are "
                      "candidates; the smaller configuration is the lower depth first ('fewer parameters'), then the "
                      "larger l2_leaf_reg ('stronger penalty'); CatBoost has no rank",
    "early_stopping_metric": "CatBoost eval_metric MAE over the split's validation rows, unweighted (row-pooled); the "
                             "selection criterion is the design's macro MAE computed from the shrunk model's validation "
                             "predictions (use_best_model=True)",
    "outer_iterations": "the outer refit uses iterations = median over the selected configuration's inner fits of the "
                        "trees kept by early stopping (best_iteration + 1), rounded half up, >= 1; no early stopping at "
                        "the refit (it would need the outer test rows); the M-model rule of section 6 (median best "
                        "epoch) applied to CatBoost",
    "v5_inner_fits": "batching follows the outer rule (section 7): each inner batch of cell_holdout.inner_cells_V5 is one "
                     "inner fit whose validation rows are its cells' scored rows; the median of iterations is over those "
                     "batch fits",
    "first_inner_fold": "on seeds other than 104729 the first inner fold only: V1 inner fold 0 of inner_folds_V1, V2 the "
                        "first (sorted) state of inner_metals_V2, V5 inner fold 0 with all its batches (so 'one fit per "
                        "configuration' is one fit per inner batch); when that fold holds no validation row, the lowest "
                        "inner fold index that does",
    "v1_inner_unit": "V1 inner averaging unit = the V1 unit of the section 3.2 resolution applied to the inner rows: the "
                     "row's publication group, or REMAINDER for groups below 20 rows of the outer-training rows (whole "
                     "groups under V1, so the corpus count); V0 (grouped 3-fold calibration design) averages by "
                     "publication group",
    "validation_population": "inner validation rows = the fold builder's scored rows (known state, not Sr(III), not "
                             "V6_TARGET_ROWS) minus context.exclude_from_scoring (acidic co-extractant rows), as the "
                             "pre-seal calibration population; a split without such rows is not fitted",
    "inner_guard": "every inner split passes context.isolation_check (every_split; test = all rows of the hidden cells "
                   "/ groups / target state inside the outer-training rows) before any fit",
    "model_seed": "one model seed per outer fold (42 + fold_index * 1009 + 9,999,991) for every inner fit, calibration "
                  "fit and the outer refit of that fold; fold_index is the fold number the caller assigns (discovery: "
                  "the position of the (discovery seed, outer fold) pair in the design enumerated once per discovery "
                  "seed, evaluation.discovery.fold_ordinals, so the discovery seed enters the model seed as well as the "
                  "inner designs)",
    "conformal": "discovery (scripts/g19_run_discovery.py) calibrates cross-fitted: the residuals of inner fold j come "
                 "from refits whose configuration and iteration count were selected WITHOUT fold j's rows "
                 "(select_excluding_folds on the recorded tuning; on seeds other than 104729, the configuration tuned "
                 "on the first inner fold refitted on the next inner fold), so no calibration residual is an "
                 "early-stopped or selection-minimised validation error.  ConformalWrapper(BoostedArm.frozen()) "
                 "remains available and reuses the tuned configuration on every calibration split",
    "no_sample_weights": "no sample weights (section 5 names none for B5 / FLAT_CAT)",
    "catboost_defaults": "every CatBoost parameter not named by section 5 is the CatBoost 1.2.10 default "
                         "(boosting_type, bootstrap, border_count, nan_mode Min, ctr settings); allow_writing_files=False",
    "target_hidden_from_features": "the target column is dropped from every frame handed to the feature set",
}


# --------------------------------------------------------------------------------------------- #
# seeds, configurations, the selection rule
# --------------------------------------------------------------------------------------------- #

def model_seed_for_fold(fold_index: int) -> int:
    """Section 15 model seed of outer fold ``fold_index``: ``42 + fold * 1009 + 9,999,991``."""
    if isinstance(fold_index, bool) or not isinstance(fold_index, (int, np.integer)) or int(fold_index) < 0:
        raise ValueError(f"fold_index must be a non-negative integer, got {fold_index!r}")
    return int(MODEL_SEED_BASE + int(fold_index) * FOLD_SEED_STRIDE + FOLD_SEED_OFFSET)


def inner_mode_for_seed(seed: int) -> str:
    """Section 7 compute plan item 1: ``"full"`` (3 inner folds) on seed 104729, ``"first"`` on every other seed
    (the four other discovery seeds and every withheld confirmation seed)."""
    return "full" if int(seed) == FULL_INNER_SEED else "first"


@dataclass(frozen=True)
class CatBoostConfig:
    """One CatBoost configuration.  The searched fields are ``depth`` and ``l2_leaf_reg``; the rest are the section 5
    constants (:meth:`is_registered` checks them)."""

    depth: int
    l2_leaf_reg: float
    learning_rate: float = 0.05
    max_iterations: int = 3000
    early_stopping_rounds: int = 200
    one_hot_max_size: int = 10
    thread_count: int = 2
    loss_function: str = "RMSE"
    eval_metric: str = "MAE"

    @property
    def label(self) -> str:
        return f"depth{int(self.depth)}_l2{float(self.l2_leaf_reg):g}"

    def complexity_key(self) -> tuple:
        """Smaller sorts first: lower depth (fewer parameters), then larger ``l2_leaf_reg`` (stronger penalty)."""
        return (int(self.depth), -float(self.l2_leaf_reg), self.label)

    def is_registered(self) -> bool:
        return (int(self.depth) in DEPTH_GRID and float(self.l2_leaf_reg) in L2_GRID and self.learning_rate == 0.05
                and self.max_iterations == 3000 and self.early_stopping_rounds == 200 and self.one_hot_max_size == 10
                and self.thread_count == 2 and self.loss_function == "RMSE" and self.eval_metric == "MAE")

    def catboost_params(self, seed: int, iterations: int | None = None, early_stopping: bool = False) -> dict[str, Any]:
        it = self.max_iterations if iterations is None else int(iterations)
        if it < 1 or it > self.max_iterations:
            raise ValueError(f"iterations {it} outside 1..{self.max_iterations}")
        p = dict(loss_function=self.loss_function, eval_metric=self.eval_metric, depth=int(self.depth),
                 learning_rate=float(self.learning_rate), l2_leaf_reg=float(self.l2_leaf_reg), iterations=it,
                 one_hot_max_size=int(self.one_hot_max_size), thread_count=int(self.thread_count),
                 random_seed=int(seed), verbose=False, allow_writing_files=False)
        if early_stopping:
            p.update(early_stopping_rounds=int(self.early_stopping_rounds), use_best_model=True)
        return p

    def record(self) -> dict[str, Any]:
        return {"label": self.label, **asdict(self)}


#: section 5 grid: depth {4, 6, 8} x l2_leaf_reg {3, 10}
REGISTERED_GRID: tuple[CatBoostConfig, ...] = tuple(CatBoostConfig(d, l2) for d in DEPTH_GRID for l2 in L2_GRID)


def grid_is_registered(grid: Sequence[CatBoostConfig]) -> bool:
    return sorted(c.complexity_key() for c in grid) == sorted(c.complexity_key() for c in REGISTERED_GRID) and \
        all(c.is_registered() for c in grid)


def select_config(scores: Mapping[CatBoostConfig, float], tolerance: float = TIE_TOLERANCE
                  ) -> tuple[CatBoostConfig, dict[str, Any]]:
    """Section 7 selection: the best (lowest) inner macro MAE, then every configuration within ``tolerance`` of it,
    resolved toward the smaller configuration (:meth:`CatBoostConfig.complexity_key`).  Refuses a non-finite score."""
    if not scores:
        raise ValueError("select_config: no scores")
    bad = [c.label for c, s in scores.items() if not np.isfinite(float(s))]
    if bad:
        raise ValueError(f"select_config: non-finite inner score for {bad}")
    best_cfg = min(scores, key=lambda c: (float(scores[c]), c.complexity_key()))
    best = float(scores[best_cfg])
    within = sorted((c for c, s in scores.items() if float(s) <= best + float(tolerance) + _TIE_EPS),
                    key=lambda c: c.complexity_key())
    chosen = within[0]
    return chosen, {"rule": f"within {tolerance} of the best inner macro MAE -> lower depth, then larger l2_leaf_reg",
                    "tolerance": float(tolerance), "best_config": best_cfg.label, "best_score": best,
                    "candidates_within_tolerance": [c.label for c in within], "selected": chosen.label,
                    "selected_score": float(scores[chosen])}


def median_iterations(tree_counts: Iterable[int]) -> int:
    """Median of the inner fits' kept tree counts, rounded half up, at least 1."""
    arr = np.asarray(list(tree_counts), dtype=float)
    if not len(arr) or not np.isfinite(arr).all():
        raise ValueError("median_iterations: no finite tree count")
    return max(1, int(math.floor(float(np.median(arr)) + 0.5)))


def select_excluding_folds(tuning_record: Mapping[str, Any], exclude_folds: Iterable[int], *,
                           tolerance: float = TIE_TOLERANCE) -> tuple[CatBoostConfig, int, dict[str, Any]]:
    """The section 7 selection re-run on the inner splits OUTSIDE ``exclude_folds`` of a recorded tuning
    (:meth:`TuningResult.record`): per configuration the macro MAE over the averaging units of those splits (a unit
    pooled over the splits holding it), :func:`select_config`'s tie rule, and the median kept-tree count of the chosen
    configuration's fits on those splits.  Used by the cross-fitted conformal calibration (section 12 reading): the
    residuals of inner fold ``j`` come from the configuration and iteration count chosen without fold ``j``'s rows."""
    drop = {int(f) for f in exclude_folds}
    sue = pd.DataFrame(tuning_record.get("split_unit_errors") or [])
    fits = pd.DataFrame(tuning_record.get("fits") or [])
    if sue.empty or fits.empty:
        raise ValueError("select_excluding_folds: the tuning record lacks split_unit_errors / fits")
    sue = sue[~sue["inner_fold"].astype(int).isin(drop)]
    fits = fits[~fits["inner_fold"].astype(int).isin(drop)]
    if sue.empty:
        raise ValueError(f"select_excluding_folds: no inner split outside folds {sorted(drop)}")
    grid = {c["label"]: CatBoostConfig(**{k: v for k, v in c.items() if k != "label"}) for c in tuning_record["grid"]}
    agg = sue.groupby(["config", "unit"], sort=True)[["n_rows", "sum_abs_error"]].sum()
    macro = (agg["sum_abs_error"] / agg["n_rows"]).groupby(level="config").mean()
    scores = {grid[label]: float(v) for label, v in macro.items()}
    chosen, selection = select_config(scores, tolerance)
    iters = median_iterations(fits.loc[fits["config"] == chosen.label, "n_trees"])
    selection.update(excluded_inner_folds=sorted(drop), n_splits=int(sue["split_id"].nunique()), iterations=iters)
    return chosen, iters, selection


def macro_mae(abs_errors: np.ndarray, units: Sequence[str]) -> tuple[float, pd.DataFrame]:
    """``(mean over units of the unit's mean absolute error, per-unit table)``."""
    e = np.asarray(abs_errors, dtype=float)
    u = np.asarray(units, dtype=object)
    if len(e) != len(u) or not len(e):
        raise ValueError("macro_mae: errors and units must be non-empty and aligned")
    tab = pd.DataFrame({"unit": u.astype(str), "abs_error": e}).groupby("unit", sort=True)["abs_error"].agg(
        n_rows="size", mae="mean").reset_index()
    return float(tab["mae"].mean()), tab


# --------------------------------------------------------------------------------------------- #
# feature matrices
# --------------------------------------------------------------------------------------------- #

class Featurizer(Protocol):
    def fit(self, rows: pd.DataFrame, cv: pd.DataFrame | None = None) -> "Featurizer": ...

    def transform(self, rows: pd.DataFrame, cv: pd.DataFrame | None = None) -> Any: ...


def default_feature_factory(arm: str) -> Callable[[], F.FeatureSet]:
    preset = FEATURE_PRESET[arm]
    return lambda: F.FeatureSet.for_arm(preset)


def _forbidden_direct(pub_group_col: str) -> frozenset[str]:
    return frozenset(LOAD.PROVENANCE_COLUMNS) | frozenset(F.ID_COLUMNS) | frozenset(F.TARGET_COLUMNS) | \
        {TARGET_COL, ID_COL, pub_group_col, FI.GROUP_COL, SG.PUB_COL}


def check_feature_matrix(fm: Any, *, pub_group_col: str = I.PUB_GROUP_COL, what: str = "feature matrix") -> None:
    """Explicit section 2 guard on a matrix CatBoost will see: no provenance column, publication / study id, row id,
    target or forbidden name token (``features.assert_feature_columns_allowed`` plus a direct set check on the bare and
    block-prefixed names), no wide block (B5 / FLAT_CAT have none), token columns hold ``str``."""
    cols = [str(c) for c in fm.frame.columns]
    F.assert_feature_columns_allowed(cols, what)
    forb = _forbidden_direct(pub_group_col)
    bare = {c.split("__", 1)[1].split("=", 1)[0] if "__" in c else c for c in cols}
    hit = sorted((set(cols) | bare) & forb)
    if hit:
        raise AssertionError(f"{what}: provenance / id / target columns among the features: {hit}")
    if getattr(fm, "wide", None):
        raise ValueError(f"{what}: wide feature blocks are not part of the B5 / FLAT_CAT settings")
    for c in fm.categorical_columns:
        vals = fm.frame[c].to_numpy(dtype=object)
        if not all(isinstance(v, str) for v in vals):
            raise TypeError(f"{what}: categorical column {c} holds a non-str token")


def _feature_input(rows: pd.DataFrame) -> pd.DataFrame:
    """Rows handed to a feature set: the target column is dropped (a feature set never needs it)."""
    return rows.drop(columns=[TARGET_COL], errors="ignore")


def _pool(fm: Any, y: np.ndarray | None = None):
    from catboost import Pool

    cats = list(fm.cat_feature_indices()) if hasattr(fm, "cat_feature_indices") else \
        [list(fm.frame.columns).index(c) for c in fm.categorical_columns]
    return Pool(data=fm.frame, label=None if y is None else np.asarray(y, dtype=float), cat_features=cats or None)


def _label_digest(index: Iterable[Any]) -> str:
    return hashlib.sha256("\n".join(sorted(str(x) for x in index)).encode("utf-8")).hexdigest()


def _target(rows: pd.DataFrame, what: str) -> np.ndarray:
    if TARGET_COL not in rows.columns:
        raise KeyError(f"{what}: no {TARGET_COL} column")
    y = pd.to_numeric(rows[TARGET_COL], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError(f"{what}: {int((~np.isfinite(y)).sum())} row(s) with a non-finite {TARGET_COL}")
    return y


# --------------------------------------------------------------------------------------------- #
# inner tuning designs (section 7; the fold builders' functions)
# --------------------------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TuningSplit:
    """One inner fit: ``train_index`` / ``hidden_index`` partition the outer-training rows; ``val_index`` (a subset of
    ``hidden_index``) are the scored validation rows, ``val_units`` their averaging unit; ``guard_test_index`` is the
    test side handed to the isolation check."""

    design: str
    split_id: str
    inner_fold: int
    units: tuple[str, ...]
    train_index: pd.Index
    hidden_index: pd.Index
    val_index: pd.Index
    val_units: np.ndarray
    guard_test_index: pd.Index


def _require_seed(context: I.FitContext) -> int:
    if context.seed is None:
        raise ValueError("the inner tuning designs draw from context.seed; it must be set")
    return int(context.seed)


def _v6_series(rows: pd.DataFrame, context: I.FitContext) -> pd.Series:
    if context.v6_mask is None:
        raise ValueError("context.v6_mask (V6_TARGET_ROWS) is required: inner validation never scores it")
    s = context.v6_mask.reindex(rows.index)
    if s.isna().any():
        raise KeyError("context.v6_mask does not cover every training row")
    return s.astype(bool)


def _excluded(rows: pd.DataFrame, context: I.FitContext) -> pd.Series:
    if context.exclude_from_scoring is None:
        return pd.Series(False, index=rows.index)
    s = context.exclude_from_scoring.reindex(rows.index)
    if s.isna().any():
        raise KeyError("context.exclude_from_scoring does not cover every training row")
    return s.astype(bool)


def _fold_frame(rows: pd.DataFrame, context: I.FitContext) -> pd.DataFrame:
    """The outer-training rows with ``group_cross_publication_copy`` taken from the context's publication-group column
    (the fold functions group by it) -- a row-wise relabel, no row added or removed."""
    if not rows.index.is_unique:
        raise ValueError("training rows: the index must be unique")
    for c in (ID_COL, SG.METAL_COL, SG.ELEMENT_COL, SG.SYSTEM_COL):
        if c not in rows.columns:
            raise KeyError(f"training rows: column {c} missing")
    if FI.GROUP_COL in rows.columns:
        return rows
    if context.pub_group_col not in rows.columns:
        raise KeyError(f"training rows: no {context.pub_group_col} / {FI.GROUP_COL} column (inner designs group by it)")
    return rows.assign(**{FI.GROUP_COL: rows[context.pub_group_col].astype(str)})


class _InnerDesign:
    design = "?"

    def _labels(self, idmap: pd.Series, ids: Iterable[str]) -> pd.Index:
        ids = list(ids)
        return pd.Index(idmap.loc[ids].to_numpy(), dtype=object) if ids else pd.Index([], dtype=object)

    @staticmethod
    def _idmap(fr: pd.DataFrame) -> pd.Series:
        ids = fr[ID_COL].astype(str)
        if not ids.is_unique:
            raise ValueError(f"{ID_COL} must be unique")
        return pd.Series(fr.index.to_numpy(dtype=object), index=ids.to_numpy())

    def _split(self, fr: pd.DataFrame, fold: FI.Fold, v6: pd.Series, excl: pd.Series, idmap: pd.Series, *,
               inner_fold: int, units_of_val: Callable[[pd.Index], np.ndarray], guard_test: pd.Index) -> TuningSplit | None:
        FI.assert_scoring_clean(fold, fr, v6)
        hidden = self._labels(idmap, fold.hidden_row_ids)
        scored = self._labels(idmap, fold.scored_row_ids)
        val = scored[~excl.reindex(scored).to_numpy(dtype=bool)] if len(scored) else scored
        if not len(val):
            return None
        train = fr.index[~fr.index.isin(hidden)]
        return TuningSplit(design=self.design, split_id=fold.fold_id, inner_fold=int(inner_fold),
                           units=tuple(str(u) for u in fold.units), train_index=pd.Index(train, dtype=object),
                           hidden_index=hidden, val_index=val, val_units=units_of_val(val),
                           guard_test_index=pd.Index(guard_test, dtype=object))


class V5InnerTuning(_InnerDesign):
    """Section 7 V5 inner design (``folds.cell_holdout.inner_cells_V5``, batched by the outer rule): inner hidden cells
    eligible under the variant's thresholds recomputed on the outer-training rows, no ``V6_TARGET_ROWS`` cell, <= 30 cells
    per inner fold; one inner fit per inner batch; averaging unit = hidden cell."""

    design = "V5"

    def __init__(self, thresholds: CH.Thresholds = CH.PRIMARY, *, medium: str = "all", component_aware: bool = True,
                 component_map: Mapping[str, str] | None = None, n_inner: int = CH.INNER_N_FOLDS,
                 max_cells: int = CH.INNER_MAX_CELLS, max_cells_per_batch: int | None = None, variant: str = "primary"):
        self.thresholds, self.medium, self.component_aware = thresholds, medium, bool(component_aware)
        self.component_map, self.n_inner, self.max_cells = component_map, int(n_inner), int(max_cells)
        self.max_cells_per_batch, self.variant = max_cells_per_batch, variant

    @classmethod
    def for_variant(cls, variant: str = "primary", max_cells_per_batch: int | None = None) -> "V5InnerTuning":
        v = CH.VARIANTS[variant]
        return cls(v.thresholds, medium=v.medium, component_aware=v.component_aware,
                   component_map=CH.component_map_for(v), max_cells_per_batch=max_cells_per_batch, variant=variant)

    def describe(self) -> dict[str, Any]:
        return {"design": "V5", "variant": self.variant, "thresholds": self.thresholds.tag, "medium": self.medium,
                "component_aware": self.component_aware, "parent_structure": self.component_map is not None,
                "n_inner": self.n_inner, "max_cells": self.max_cells, "max_cells_per_batch": self.max_cells_per_batch,
                "fold_function": "folds.cell_holdout.inner_cells_V5(batched=True)"}

    def splits(self, rows: pd.DataFrame, context: I.FitContext) -> list[TuningSplit]:
        seed = _require_seed(context)
        fr = _fold_frame(rows, context)
        v6, excl, idmap = _v6_series(fr, context), _excluded(fr, context), self._idmap(fr)
        folds = CH.inner_cells_V5(fr, self.thresholds, seed, medium=self.medium, component_aware=self.component_aware,
                                  component_map=self.component_map, batched=True, n_inner=self.n_inner,
                                  max_cells=self.max_cells, v6_ids=fr.loc[v6.to_numpy(), ID_COL].astype(str),
                                  check=False, max_cells_per_batch=self.max_cells_per_batch)
        st = fr[SG.METAL_COL].to_numpy(dtype=object)
        sy = fr[SG.SYSTEM_COL].to_numpy(dtype=object)
        out = []
        for f in folds:
            cells = [tuple(c) for c in f.meta["cells"]]
            in_cells = np.zeros(len(fr), dtype=bool)
            for m, s in cells:
                in_cells |= (st == m) & (sy == s)

            def units(val: pd.Index) -> np.ndarray:
                sub = fr.loc[val]
                return np.array([CH.cell_label((str(a), str(b))) for a, b in
                                 zip(sub[SG.METAL_COL].to_numpy(dtype=object), sub[SG.SYSTEM_COL].to_numpy(dtype=object))],
                                dtype=object)
            sp = self._split(fr, f, v6, excl, idmap, inner_fold=int(f.meta["inner_fold"]), units_of_val=units,
                             guard_test=fr.index[in_cells])
            if sp is not None:
                out.append(sp)
        return out


class V1InnerTuning(_InnerDesign):
    """Section 7 V1 inner design (``folds.source_holdout.inner_folds_V1``): grouped 3-fold over the publication groups of
    the outer-training rows, one fit per inner fold.  Averaging unit: ``"v1_unit"`` (group, or ``REMAINDER`` below 20
    rows; the section 3.2 V1 unit) or ``"publication_group"`` (the V0 averaging unit)."""

    def __init__(self, n_inner: int = SH.N_INNER, unit: str = "v1_unit", design: str = "V1"):
        if unit not in ("v1_unit", "publication_group"):
            raise ValueError(f"unit {unit!r}")
        self.n_inner, self.unit, self.design = int(n_inner), unit, design

    def describe(self) -> dict[str, Any]:
        return {"design": self.design, "n_inner": self.n_inner, "unit": self.unit,
                "fold_function": "folds.source_holdout.inner_folds_V1"}

    def splits(self, rows: pd.DataFrame, context: I.FitContext) -> list[TuningSplit]:
        seed = _require_seed(context)
        fr = _fold_frame(rows, context)
        v6, excl, idmap = _v6_series(fr, context), _excluded(fr, context), self._idmap(fr)
        folds = SH.inner_folds_V1(fr, seed, self.n_inner, v6_ids=fr.loc[v6.to_numpy(), ID_COL].astype(str), check=False)
        groups = FI.publication_groups(fr, FI.GROUP_COL)
        unit_of = SH.unit_of_rows(groups) if self.unit == "v1_unit" else groups
        out = []
        for k, f in enumerate(folds):
            sp = self._split(fr, f, v6, excl, idmap, inner_fold=k,
                             units_of_val=lambda val: unit_of.loc[val].astype(str).to_numpy(dtype=object),
                             guard_test=self._labels(idmap, f.hidden_row_ids))
            if sp is not None:
                out.append(sp)
        return out


class V2InnerTuning(_InnerDesign):
    """Section 7 V2 inner design (``folds.metal_holdout.inner_metals_V2``): leave-one-metal-out over 3 seeded V2-eligible
    states of the outer-training rows, element-level hiding (``element_level=False``: the state-level sensitivity);
    averaging unit = metal state."""

    design = "V2"

    def __init__(self, n_states: int = MH.N_INNER_STATES, element_level: bool = True):
        self.n_states, self.element_level = int(n_states), bool(element_level)

    def describe(self) -> dict[str, Any]:
        return {"design": "V2", "n_states": self.n_states, "element_level": self.element_level,
                "fold_function": "folds.metal_holdout.inner_metals_V2"}

    def splits(self, rows: pd.DataFrame, context: I.FitContext) -> list[TuningSplit]:
        seed = _require_seed(context)
        fr = _fold_frame(rows, context)
        v6, excl, idmap = _v6_series(fr, context), _excluded(fr, context), self._idmap(fr)
        folds = MH.inner_metals_V2(fr, seed, self.n_states, element_level=self.element_level,
                                   v6_ids=fr.loc[v6.to_numpy(), ID_COL].astype(str), check=False)
        out = []
        for k, f in enumerate(folds):
            state = f.units[0]
            sp = self._split(fr, f, v6, excl, idmap, inner_fold=k,
                             units_of_val=lambda val, s=state: np.array([s] * len(val), dtype=object),
                             guard_test=fr.index[(fr[SG.METAL_COL] == state).to_numpy(dtype=bool)])
            if sp is not None:
                out.append(sp)
        return out


def inner_design_for(design: str, variant: str = "primary", *, max_cells_per_batch: int | None = None):
    """The section 7 inner tuning design of an outer design: V5 (and its variants; V5-P, V5-PAIR and V6 heavy-arm fits
    use the V5 primary inner design), V1, V2 (``variant="state"``: state-level sensitivity), V0 (the grouped 3-fold,
    averaged by publication group).  V3 / V4 / V7 have no inner fold function in ``gen19ct.folds``."""
    d = str(design).upper().replace("-", "")
    if d == "V5":
        return V5InnerTuning.for_variant(variant, max_cells_per_batch=max_cells_per_batch)
    if d in ("V5P", "V5PAIR", "V6"):
        return V5InnerTuning.for_variant("primary", max_cells_per_batch=max_cells_per_batch)
    if d == "V1":
        return V1InnerTuning(unit="v1_unit", design="V1")
    if d == "V0":
        return V1InnerTuning(unit="publication_group", design="V0")
    if d == "V2":
        return V2InnerTuning(element_level=(variant != "state"))
    raise NotImplementedError(f"no inner tuning design for {design!r} (gen19ct.folds builds none)")


def design_isolation_check(slim: pd.DataFrame, design: str, *, component_map: Mapping[str, str] | None = None,
                           element_level: bool = True) -> Callable[[pd.Index, pd.Index], dict]:
    """``(train_labels, test_labels) -> report``: ``fold_isolation_check`` at the design level of an inner split, with the
    section 2 settings (``near_dup_sig`` 6, value tolerance 0.005) -- the pre-seal ``inner_check`` for use as
    ``FitContext.isolation_check``.  ``slim`` is ``folds.io.slim_frame`` of the arm frame."""
    d = str(design).upper().replace("-", "")
    if d in ("V5", "V5P", "V5PAIR", "V6"):
        kw = dict(level="V5", component_aware=True, component_map=component_map)
    elif d in ("V1", "V0"):
        kw = dict(level="V1")
    elif d == "V2":
        kw = dict(level="V2", element_level=bool(element_level))
    else:
        raise ValueError(f"no inner isolation level for {design!r}")

    def check(tr: pd.Index, te: pd.Index) -> dict:
        return LK.fold_isolation_check(tr, te, slim, near_dup_sig=FI.NEAR_DUP_SIG, near_dup_value_tol=FI.NEAR_DUP_VALUE_TOL,
                                       raise_on_violation=False, **kw)
    return check


def verify_split(sp: TuningSplit, rows: pd.DataFrame, context: I.FitContext) -> None:
    """Partition, scoring-population and isolation assertions of one inner split (raises)."""
    idx = rows.index
    if not sp.train_index.isin(idx).all() or not sp.hidden_index.isin(idx).all():
        raise AssertionError(f"{sp.split_id}: inner split rows outside the outer-training rows")
    if sp.train_index.isin(sp.hidden_index).any() or len(sp.train_index) + len(sp.hidden_index) != len(idx):
        raise AssertionError(f"{sp.split_id}: inner training + hidden rows are not a partition of the training rows")
    if not sp.val_index.isin(sp.hidden_index).all() or len(sp.val_units) != len(sp.val_index):
        raise AssertionError(f"{sp.split_id}: validation rows must be hidden rows with one unit each")
    FR.assert_not_scored(sp.val_index, context.v6_mask, f"inner validation set {sp.split_id}")
    st = rows.loc[sp.val_index, SG.METAL_COL]
    if st.isna().any() or (st == I.SR_III).any():
        raise AssertionError(f"{sp.split_id}: an X(?) or Sr(III) row would be scored in inner validation")
    if _excluded(rows, context).reindex(sp.val_index).any():
        raise AssertionError(f"{sp.split_id}: an exclude_from_scoring row would be scored in inner validation")
    if context.hidden_index is not None and len(context.hidden_index) and \
            (sp.train_index.isin(context.hidden_index).any() or sp.hidden_index.isin(context.hidden_index).any()):
        raise AssertionError(f"{sp.split_id}: a row the outer fold hid is inside an inner split")
    if context.isolation_check is None:
        raise ValueError("inner splits must pass fold_isolation_check: context.isolation_check is required")
    key = "tuning|" + hashlib.sha256((_label_digest(sp.train_index) + "|" + _label_digest(sp.guard_test_index))
                                     .encode("utf-8")).hexdigest()
    if key in context.guard_cache:
        return
    rep = context.isolation_check(sp.train_index, sp.guard_test_index)
    if not (isinstance(rep, Mapping) and rep.get("ok", False)):
        raise AssertionError(f"inner tuning split {sp.split_id} failed the isolation check")
    context.guard_cache[key] = True


# --------------------------------------------------------------------------------------------- #
# the tuner
# --------------------------------------------------------------------------------------------- #

@dataclass
class TuningResult:
    """Everything one nested tuning produced (no outer score).  ``fits`` has one row per (configuration, inner split);
    ``config_scores`` one row per configuration; ``unit_scores`` one row per (configuration, averaging unit)."""

    arm: str
    design: dict
    mode: str
    seed: int
    model_seed: int
    grid: tuple[CatBoostConfig, ...]
    registered_settings: bool
    splits: pd.DataFrame
    fits: pd.DataFrame
    unit_scores: pd.DataFrame
    config_scores: pd.DataFrame
    selected: CatBoostConfig
    selected_iterations: int
    selection: dict
    timing: dict
    touched_index: pd.Index = field(repr=False)
    inner_folds_used: tuple[int, ...] = ()
    #: one row per (configuration, inner split, averaging unit): ``n_rows`` and ``sum_abs_error`` of the split's
    #: validation rows -- what :func:`select_excluding_folds` re-selects from (the cross-fitted calibration)
    split_unit_errors: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    def scores(self) -> dict[str, float]:
        return dict(zip(self.config_scores["config"], self.config_scores["macro_mae"]))

    def record(self) -> dict[str, Any]:
        """JSON-ready summary for a manifest (section 16: inner-selected hyperparameters per outer fit)."""
        return {"schema": SCHEMA, "arm": self.arm, "design": self.design, "mode": self.mode, "seed": self.seed,
                "model_seed": self.model_seed, "registered_settings": self.registered_settings,
                "grid": [c.record() for c in self.grid], "selected": self.selected.record(),
                "selected_iterations": self.selected_iterations, "selection": self.selection,
                "inner_folds_used": list(self.inner_folds_used), "n_inner_fits_per_config": int(len(self.splits)),
                "config_scores": self.config_scores.to_dict(orient="records"), "timing": self.timing,
                "fits": self.fits[[c for c in ("config", "split_id", "inner_fold", "best_iteration", "n_trees")
                                   if c in self.fits.columns]].to_dict(orient="records"),
                "split_unit_errors": self.split_unit_errors.to_dict(orient="records"),
                "registration_choices": REGISTRATION_CHOICES}


class Tuner:
    """Nested section 7 tuning of one CatBoost arm on the outer-training rows passed to :meth:`run` (module docstring,
    flow steps 1-4).  ``grid`` defaults to :data:`REGISTERED_GRID`; any other grid is recorded as
    ``registered_settings=False``."""

    def __init__(self, arm: str = "B5", design: Any = None, *, grid: Sequence[CatBoostConfig] = REGISTERED_GRID,
                 mode: str = "full", feature_factory: Callable[[], Any] | None = None, cv: pd.DataFrame | None = None,
                 tie_tolerance: float = TIE_TOLERANCE):
        if arm not in ARM_NAMES:
            raise ValueError(f"unknown boosted arm {arm!r}; expected one of {ARM_NAMES}")
        if mode not in INNER_MODES:
            raise ValueError(f"mode must be one of {INNER_MODES}")
        grid = tuple(grid)
        if not grid or len({c.complexity_key() for c in grid}) != len(grid):
            raise ValueError("grid must be non-empty with distinct configurations")
        self.arm, self.mode, self.grid = arm, mode, grid
        self.design = inner_design_for("V5") if design is None else (inner_design_for(design) if isinstance(design, str)
                                                                     else design)
        self.feature_factory = feature_factory or default_feature_factory(arm)
        self.cv, self.tie_tolerance = cv, float(tie_tolerance)

    def select_splits(self, splits: list[TuningSplit]) -> tuple[list[TuningSplit], tuple[int, ...]]:
        if not splits:
            raise ValueError(f"{self.arm}: the inner design produced no inner split with validation rows")
        if self.mode == "full":
            return splits, tuple(sorted({s.inner_fold for s in splits}))
        first = min(s.inner_fold for s in splits)
        return [s for s in splits if s.inner_fold == first], (first,)

    def run(self, rows: pd.DataFrame, context: I.FitContext, model_seed: int) -> TuningResult:
        from catboost import CatBoostRegressor

        t_start = time.perf_counter()
        seed = _require_seed(context)
        _check_training_rows(rows, context, f"{self.arm} tuning")
        t0 = time.perf_counter()
        all_splits = self.design.splits(rows, context)
        used, folds_used = self.select_splits(all_splits)
        t_design = time.perf_counter() - t0
        t0 = time.perf_counter()
        for sp in used:
            verify_split(sp, rows, context)
        seen_val = pd.Index([], dtype=object)
        for sp in used:
            if sp.val_index.isin(seen_val).any():
                raise AssertionError(f"{sp.split_id}: a validation row is scored in two inner splits")
            seen_val = seen_val.append(sp.val_index)
        t_guard = time.perf_counter() - t0

        fit_recs, split_recs = [], []
        errors: dict[CatBoostConfig, list[np.ndarray]] = {c: [] for c in self.grid}
        units: dict[CatBoostConfig, list[np.ndarray]] = {c: [] for c in self.grid}
        touched = pd.Index([], dtype=object)
        t_feat = 0.0
        for sp in used:
            t0 = time.perf_counter()
            tr, va = rows.loc[sp.train_index], rows.loc[sp.val_index]
            fs = self.feature_factory().fit(_feature_input(tr), self.cv)
            _check_fitted_on(fs, tr, sp.split_id)
            Xtr, Xva = fs.transform(_feature_input(tr), self.cv), fs.transform(_feature_input(va), self.cv)
            check_feature_matrix(Xtr, pub_group_col=context.pub_group_col, what=f"{sp.split_id} inner training")
            check_feature_matrix(Xva, pub_group_col=context.pub_group_col, what=f"{sp.split_id} inner validation")
            if list(Xtr.frame.columns) != list(Xva.frame.columns):
                raise AssertionError(f"{sp.split_id}: training and validation feature columns differ")
            ytr, yva = _target(tr, f"{sp.split_id} inner training"), _target(va, f"{sp.split_id} inner validation")
            pool_tr, pool_va = _pool(Xtr, ytr), _pool(Xva, yva)
            t_feat += time.perf_counter() - t0
            touched = touched.union(sp.train_index).union(sp.val_index)
            digest = getattr(fs, "state_digest", None)
            split_recs.append({"split_id": sp.split_id, "inner_fold": sp.inner_fold, "n_train": len(sp.train_index),
                               "n_hidden": len(sp.hidden_index), "n_val": len(sp.val_index),
                               "n_units": int(len(set(sp.val_units))), "units": "; ".join(sp.units),
                               "n_features": int(Xtr.frame.shape[1]), "feature_state_digest": digest})
            for cfg in self.grid:
                t0 = time.perf_counter()
                model = CatBoostRegressor(**cfg.catboost_params(model_seed, early_stopping=True))
                model.fit(pool_tr, eval_set=pool_va)
                pred = np.asarray(model.predict(pool_va), dtype=float)
                if not np.isfinite(pred).all():
                    raise AssertionError(f"{self.arm}/{sp.split_id}/{cfg.label}: non-finite inner prediction")
                err = np.abs(pred - yva)
                errors[cfg].append(err)
                units[cfg].append(np.asarray(sp.val_units, dtype=object))
                fit_recs.append({"config": cfg.label, "depth": cfg.depth, "l2_leaf_reg": cfg.l2_leaf_reg,
                                 "split_id": sp.split_id, "inner_fold": sp.inner_fold,
                                 "best_iteration": int(model.get_best_iteration()),
                                 "n_trees": int(model.tree_count_), "val_mae_rows": float(err.mean()),
                                 "n_val": len(err), "fit_seconds": round(time.perf_counter() - t0, 4)})
                del model
            del Xtr, Xva, pool_tr, pool_va
        fits = pd.DataFrame(fit_recs)
        sue = []
        for cfg in self.grid:
            for sp, e, u in zip(used, errors[cfg], units[cfg]):
                t = pd.DataFrame({"unit": np.asarray(u, dtype=object).astype(str), "e": e}).groupby("unit", sort=True)["e"]
                for unit, n, tot in zip(t.size().index, t.size().to_numpy(), t.sum().to_numpy()):
                    sue.append({"config": cfg.label, "split_id": sp.split_id, "inner_fold": int(sp.inner_fold),
                                "unit": str(unit), "n_rows": int(n), "sum_abs_error": float(tot)})
        split_unit_errors = pd.DataFrame(sue, columns=["config", "split_id", "inner_fold", "unit", "n_rows",
                                                       "sum_abs_error"])
        scores, cfg_recs, unit_tabs = {}, [], []
        for cfg in self.grid:
            e, u = np.concatenate(errors[cfg]), np.concatenate(units[cfg])
            macro, tab = macro_mae(e, u)
            scores[cfg] = macro
            unit_tabs.append(tab.assign(config=cfg.label))
            nt = fits.loc[fits["config"] == cfg.label, "n_trees"].to_numpy()
            cfg_recs.append({"config": cfg.label, "depth": cfg.depth, "l2_leaf_reg": cfg.l2_leaf_reg, "macro_mae": macro,
                             "row_pooled_mae_diagnostic": float(e.mean()), "n_units": int(len(tab)), "n_rows": int(len(e)),
                             "median_n_trees": median_iterations(nt), "n_fits": int(len(nt))})
        chosen, selection = select_config(scores, self.tie_tolerance)
        cs = pd.DataFrame(cfg_recs)
        cs["within_tolerance"] = cs["config"].isin(selection["candidates_within_tolerance"])
        cs["selected"] = cs["config"] == chosen.label
        iters = median_iterations(fits.loc[fits["config"] == chosen.label, "n_trees"])
        timing = {"total_s": round(time.perf_counter() - t_start, 3), "design_s": round(t_design, 3),
                  "guard_s": round(t_guard, 3), "features_s": round(t_feat, 3),
                  "catboost_s": round(float(fits["fit_seconds"].sum()), 3), "n_fits": int(len(fits)),
                  "n_splits": len(used), "n_splits_available": len(all_splits)}
        return TuningResult(arm=self.arm, design=dict(self.design.describe()), mode=self.mode, seed=seed,
                            model_seed=int(model_seed), grid=self.grid, registered_settings=grid_is_registered(self.grid),
                            splits=pd.DataFrame(split_recs), fits=fits,
                            unit_scores=pd.concat(unit_tabs, ignore_index=True)[["config", "unit", "n_rows", "mae"]],
                            config_scores=cs, selected=chosen, selected_iterations=iters, selection=selection,
                            timing=timing, touched_index=touched, inner_folds_used=folds_used,
                            split_unit_errors=split_unit_errors)


def _check_training_rows(rows: pd.DataFrame, context: I.FitContext, what: str) -> None:
    if not len(rows):
        raise ValueError(f"{what}: no training rows")
    if not rows.index.is_unique:
        raise ValueError(f"{what}: the training index must be unique")
    _target(rows, what)
    if context.hidden_index is not None and len(context.hidden_index):
        n = int(rows.index.isin(context.hidden_index).sum())
        if n:
            raise AssertionError(f"{what}: {n} hidden row(s) among the training rows")


def _check_fitted_on(fs: Any, rows: pd.DataFrame, what: str) -> None:
    n = getattr(fs, "n_train_rows", None)
    if n is not None and int(n) != len(rows):
        raise AssertionError(f"{what}: the feature set was fitted on {n} rows, not the {len(rows)} training rows")


# --------------------------------------------------------------------------------------------- #
# the arm
# --------------------------------------------------------------------------------------------- #

@dataclass
class _Fitted:
    key: tuple
    features: Any
    model: Any
    columns: tuple[str, ...]
    config: CatBoostConfig
    iterations: int
    train_index: pd.Index
    states: frozenset
    systems: frozenset
    cells: frozenset
    feature_digest: str | None


class BoostedArm:
    """B5 / M0 / FLAT_CAT (module docstring).  ``interface.Arm`` protocol plus the table fast path
    (``fit_table`` / ``predict_positions``, which need ``frame=``: the arm frame covering the table's rows).

    ``fold_index`` / ``model_seed``  the section 15 model seed (one of them is required)
    ``design``                       the outer design (``"V5"``, ``"V1"``, ``"V2"``, ``"V0"``, ``"V5P"``, ``"V5PAIR"``,
                                     ``"V6"``) or an inner design object; ``variant`` selects the V5 variant
    ``inner_mode``                   ``"full"`` / ``"first"``; ``None`` = :func:`inner_mode_for_seed` of ``context.seed``
    ``config`` + ``iterations``      fixed configuration (no tuning); both or neither
    ``cv``                           ``normalize.condition_vector`` over the frame's rows (row-wise, target-free; optional)
    """

    def __init__(self, name: str = "B5", *, fold_index: int | None = None, model_seed: int | None = None,
                 design: Any = "V5", variant: str = "primary", inner_mode: str | None = None,
                 grid: Sequence[CatBoostConfig] = REGISTERED_GRID, config: CatBoostConfig | None = None,
                 iterations: int | None = None, frame: pd.DataFrame | None = None, cv: pd.DataFrame | None = None,
                 feature_factory: Callable[[], Any] | None = None, tie_tolerance: float = TIE_TOLERANCE,
                 max_cells_per_batch: int | None = None):
        if name not in ARM_NAMES:
            raise ValueError(f"unknown boosted arm {name!r}; expected one of {ARM_NAMES}")
        if (fold_index is None) == (model_seed is None):
            raise ValueError("give exactly one of fold_index (section 15 rule) or model_seed")
        if (config is None) != (iterations is None):
            raise ValueError("config and iterations are fixed together (a tuned arm gets both from its inner folds)")
        if inner_mode is not None and inner_mode not in INNER_MODES:
            raise ValueError(f"inner_mode must be one of {INNER_MODES} or None")
        self.name = name
        self.fold_index = fold_index
        self.model_seed = model_seed_for_fold(fold_index) if fold_index is not None else int(model_seed)
        self.design_spec, self.variant, self.max_cells_per_batch = design, variant, max_cells_per_batch
        self.design = inner_design_for(design, variant, max_cells_per_batch=max_cells_per_batch) \
            if isinstance(design, str) else design
        self.inner_mode, self.grid, self.config = inner_mode, tuple(grid), config
        self.iterations = None if iterations is None else int(iterations)
        self.frame, self.cv, self.tie_tolerance = frame, cv, float(tie_tolerance)
        self.feature_factory = feature_factory or default_feature_factory(name)
        self.tuning: TuningResult | None = None
        self.fitted: _Fitted | None = None
        self.reused_prefit = False
        self._prefit: dict[tuple, _Fitted] = {}
        self._table: I.RowTable | None = None
        self.fit_seconds: float | None = None

    # ----------------------------------------------------------------------------------------- #
    def _kwargs(self) -> dict[str, Any]:
        seed_kw = {"fold_index": self.fold_index} if self.fold_index is not None else {"model_seed": self.model_seed}
        return dict(**seed_kw, design=self.design, variant=self.variant, inner_mode=self.inner_mode, grid=self.grid,
                    frame=self.frame, cv=self.cv, feature_factory=self.feature_factory,
                    tie_tolerance=self.tie_tolerance, max_cells_per_batch=self.max_cells_per_batch)

    def clone(self) -> "BoostedArm":
        """An unfitted copy with the same settings (a fixed configuration stays fixed; the prefit cache is shared)."""
        c = BoostedArm(self.name, config=self.config, iterations=self.iterations, **self._kwargs())
        c._prefit = self._prefit
        return c

    def frozen(self) -> "BoostedArm":
        """An unfitted copy with the fitted configuration and iteration count fixed (no tuning); fitting it on the rows
        this arm was fitted on reuses this arm's refit (identical, the fit is deterministic)."""
        if self.fitted is None:
            raise RuntimeError(f"{self.name}: fit first")
        c = BoostedArm(self.name, config=self.fitted.config, iterations=self.fitted.iterations, **self._kwargs())
        c._prefit = {**self._prefit, self.fitted.key: self.fitted}
        return c

    # ----------------------------------------------------------------------------------------- #
    def fit(self, train_rows: pd.DataFrame, context: I.FitContext) -> "BoostedArm":
        return self._fit(train_rows, context, from_frame=False)

    def _fit(self, train_rows: pd.DataFrame, context: I.FitContext, *, from_frame: bool) -> "BoostedArm":
        t0 = time.perf_counter()
        _check_training_rows(train_rows, context, self.name)
        if self.config is None:
            mode = self.inner_mode or inner_mode_for_seed(_require_seed(context))
            tuner = Tuner(self.name, self.design, grid=self.grid, mode=mode, feature_factory=self.feature_factory,
                          cv=self.cv, tie_tolerance=self.tie_tolerance)
            self.tuning = tuner.run(train_rows, context, self.model_seed)
            config, iters = self.tuning.selected, self.tuning.selected_iterations
        else:
            self.tuning = None
            config, iters = self.config, self.iterations
        y = _target(train_rows, self.name)
        order = np.argsort(np.array([str(x) for x in train_rows.index], dtype=object), kind="stable")
        key = (self.name, repr(sorted(asdict(config).items())), int(iters), self.model_seed,
               _label_digest(train_rows.index), hashlib.sha256(np.ascontiguousarray(y[order]).tobytes()).hexdigest(),
               id(self.feature_factory), id(self.cv), ("frame", id(self.frame)) if from_frame else ("rows", id(train_rows)))
        hit = self._prefit.get(key) if from_frame else None      # reuse only rows taken from the same frame object
        self.reused_prefit = hit is not None
        self.fitted = hit if hit is not None else self._refit(train_rows, y, config, iters, key, context)
        self.fit_seconds = round(time.perf_counter() - t0, 3)
        return self

    def _refit(self, rows: pd.DataFrame, y: np.ndarray, config: CatBoostConfig, iters: int, key: tuple,
               context: I.FitContext) -> _Fitted:
        from catboost import CatBoostRegressor

        fs = self.feature_factory().fit(_feature_input(rows), self.cv)
        _check_fitted_on(fs, rows, f"{self.name} refit")
        X = fs.transform(_feature_input(rows), self.cv)
        check_feature_matrix(X, pub_group_col=context.pub_group_col, what=f"{self.name} outer training")
        model = CatBoostRegressor(**config.catboost_params(self.model_seed, iterations=iters, early_stopping=False))
        model.fit(_pool(X, y))
        st = rows[SG.METAL_COL].to_numpy(dtype=object)
        sy = rows[SG.SYSTEM_COL].to_numpy(dtype=object)
        return _Fitted(key=key, features=fs, model=model, columns=tuple(X.frame.columns), config=config,
                       iterations=int(iters), train_index=pd.Index(rows.index, dtype=object),
                       states=frozenset(str(s) for s in st if not SG._missing(s)),
                       systems=frozenset(str(s) for s in sy if not SG._missing(s)),
                       cells=frozenset((str(a), str(b)) for a, b in zip(st, sy) if not SG._missing(a)),
                       feature_digest=getattr(fs, "state_digest", None))

    def fit_table(self, table: I.RowTable, mask: np.ndarray, context: I.FitContext) -> "BoostedArm":
        if self.frame is None:
            raise ValueError(f"{self.name}: fit_table needs frame= (the arm frame over the table's rows)")
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (table.n,):
            raise ValueError("training mask does not match the RowTable")
        if (mask & I.forbidden_mask(table, context)).any():
            raise AssertionError(f"{int((mask & I.forbidden_mask(table, context)).sum())} hidden row(s) among the "
                                 "training rows")
        labels = table.index[mask]
        if not labels.isin(self.frame.index).all():
            raise KeyError(f"{self.name}: training rows missing from frame")
        rows = self.frame.loc[labels]
        if not np.array_equal(table.y[mask], pd.to_numeric(rows[TARGET_COL], errors="coerce").to_numpy(dtype=float),
                              equal_nan=True):
            raise ValueError(f"{self.name}: frame and RowTable disagree on {TARGET_COL}")
        self._table = table
        return self._fit(rows, context, from_frame=True)

    # ----------------------------------------------------------------------------------------- #
    def predict(self, query_rows: pd.DataFrame) -> pd.DataFrame:
        if self.fitted is None:
            raise RuntimeError(f"{self.name}: fit first")
        if not query_rows.index.is_unique:
            raise ValueError("query rows: the index must be unique")
        f = self.fitted
        if query_rows.index.isin(f.train_index).any():
            raise AssertionError("a query row is one of the training rows")
        n = len(query_rows)
        mean = np.zeros(0)
        if n:
            X = f.features.transform(_feature_input(query_rows), self.cv)
            check_feature_matrix(X, what=f"{self.name} query")
            if tuple(X.frame.columns) != f.columns:
                raise AssertionError(f"{self.name}: query feature columns differ from the training columns")
            mean = np.asarray(f.model.predict(_pool(X)), dtype=float)
        out = I.records_to_frame([I.empty_prediction_record() for _ in range(n)])
        out["row_id"] = pd.Series(query_rows.index.to_numpy(dtype=object), dtype=object)
        out["mean_logD"] = mean
        out["fallback_level"] = self.name
        st = query_rows[SG.METAL_COL].to_numpy(dtype=object)
        sy = query_rows[SG.SYSTEM_COL].to_numpy(dtype=object)
        out["boosted_config"] = f.config.label
        out["boosted_n_trees"] = int(f.model.tree_count_)
        out["boosted_model_seed"] = int(self.model_seed)
        out["boosted_tuned_here"] = self.tuning is not None
        out["feature_state_digest"] = f.feature_digest
        out["metal_state_in_training"] = [(not SG._missing(a)) and str(a) in f.states for a in st]
        out["system_in_training"] = [(not SG._missing(b)) and str(b) in f.systems for b in sy]
        out["cell_in_training"] = [(not SG._missing(a)) and (str(a), str(b)) in f.cells for a, b in zip(st, sy)]
        return out[list(I.PREDICTION_COLUMNS) + list(BOOSTED_DIAGNOSTIC_COLUMNS)]

    def predict_positions(self, positions: np.ndarray) -> pd.DataFrame:
        if self._table is None or self.frame is None:
            raise RuntimeError(f"{self.name}: predict_positions needs a fit_table fit")
        labels = self._table.index[np.asarray(positions, dtype=np.int64)]
        return self.predict(self.frame.loc[labels])

    def record(self) -> dict[str, Any]:
        """Hyperparameters of the last fit (section 16)."""
        if self.fitted is None:
            raise RuntimeError(f"{self.name}: fit first")
        return {"schema": SCHEMA, "arm": self.name, "model_seed": self.model_seed, "fold_index": self.fold_index,
                "config": self.fitted.config.record(), "iterations": self.fitted.iterations,
                "tuned_here": self.tuning is not None, "reused_prefit": self.reused_prefit,
                "feature_state_digest": self.fitted.feature_digest, "n_train": int(len(self.fitted.train_index)),
                "tuning": None if self.tuning is None else self.tuning.record(), "fit_seconds": self.fit_seconds}


def make_arm(name: str, **kwargs: Any) -> BoostedArm:
    return BoostedArm(name, **kwargs)
