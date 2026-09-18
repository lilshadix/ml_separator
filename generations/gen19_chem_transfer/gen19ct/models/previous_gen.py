"""``models/previous_gen.py`` -- **B8**, "previous best Gen-series models, re-fitted inside Gen19 folds"
(``preregistration.md``, sealed 2026-09-15, section 5 B8).

B8 has two halves, and the registration defines them separately:

**Level: the MONO_ET architecture** (gen6 / gen7 protocol, ``src/lanthanide_separation/gen6/hierarchical.py`` ->
``src/lanthanide_separation/levels.py``).  ExtraTrees, 400 trees, ``max_features`` 0.30, ``min_samples_leaf`` 2,
sample weights balanced over ECFP clusters of the primary extractant clustered on outer-training systems,
predictions clipped to the training range, features the Gen19 B8 blocks of
:mod:`gen19ct.models.features` (METAL = ``b8_metal``, COND = the B5 condition block, LIG2D_EXT = the 2,048-bit
ECFP4 of the primary extractant plus the component count, MASSACTION = the log10 concentrations and
``n0 * log10[L]``).

    The gen6 protocol, line by line (the lines this module reproduces):

    * ``gen6/hierarchical.py:45-47`` names ``MONO_ET`` "the Experiment A EXPANDED arm"; ``:70``
      ``FOREST_BLOCKS = ("METAL", "COND", "LIG2D_EXT", "MASSACTION")`` -- the feature set section 5 B8 lists;
      ``:434-435`` fits a gen6 forest with ``groups=cells["ecfp_cluster"]`` (the cluster-balanced weights).
    * ``gen7/contenders.py:63-96`` (``LevelTree``, ``arm="MC_lig2d_ext_massaction"``) is gen6's champion reached
      through gen7's harness: ``:72-75`` the 400 / 0.30 / 2 / cluster-weighted settings, ``:87`` ``random_state``
      = the fold's model seed, ``:89-90`` ``groups = train["ecfp_cluster"]``.
    * ``levels.py:554-558`` ``group_balanced_weights``: ``w_i = 1 / (rows of row i's cluster)`` -- equal TOTAL
      weight per cluster, not rescaled (:class:`gen19ct.models.features.EcfpClusters` is the Gen19 form, tested
      equal); ``:606-663`` ``LevelRegressor``: ``DropAllNaNColumns`` -> ``SimpleImputer(strategy="median",
      add_indicator=True)`` -> ``ExtraTreesRegressor(...)``, fitted with ``model__sample_weight``.
    * ``levels.py:661-662`` is the **training range**: ``span = y.max() - y.min()``;
      ``y_range = (y.min() - 0.5 * span, y.max() + 0.5 * span)``, i.e. the training interval widened by half a
      span on each side (total width 2 x span = section 5's "1.5x the training range" read as the gen6 rule,
      :data:`REGISTRATION_CHOICES` ``clip_range``).  ``levels.py:672-674`` applies it with ``np.clip``;
      ``gen7/contenders.py:214-216`` and ``gen7/decompositions.py:102`` repeat the same two lines.

**Direction: the gen14 G14 rule, for Ln(III)-Ln(III) pairs.**  An L2 logistic (C = 1, standardised,
median-imputed) on the 39 TOPO39 donor-topology columns predicts the sign of the lanthanide-axis radius
coefficient; it is re-fitted inside every outer training fold on training systems with >= 5 Ln(III); the
magnitude is the training-fold mean ``|logSF|`` of the pair's delta-Z class; systems without TOPO39 get no B8
direction and that coverage is reported (:meth:`B8Direction.coverage`).  The frozen gen14 model
(``generations/gen14_direction/models/deploy.joblib``) is **never** used -- nothing in this module reads it.

    The gen14 definition, line by line:

    * the label is the sign of ``bench.coef[:, 0]``, the coefficient of the standardised-Shannon-radius basis
      curve of a cell's centred lanthanide curve (``gen14/dirbench.py:139``, ``:160`` ``y = int(amp[ci] < 0)``,
      i.e. 1 = heavy-selective);
    * a *cell* is one extractant measured under one exactly matched condition vector inside one publication
      (``gen13_separation/gen13sep/cohort.py:1-35``, ``:153-168``: ``cell_key = structure @@ publication @@
      condition key``, replicates averaged per (cell, metal), the 14-lanthanide curve kept).  Gen19 keys the same
      object with its own registered units: the publication group of section 2, ``extractant_system_key`` and
      ``normalize.condition_key`` -- the section 2 comparable-pair key (:data:`REGISTRATION_CHOICES` ``cell_key``);
    * the curve is centred over the cell's observed metals and fitted by a per-cell ridge on the physics basis
      (``gen13sep/amplitude_bench.py:82-83``, ``gen13sep/basis.py:26-28, 99-117``: basis rows ``radius`` and
      ``radius_sq`` of ``gen13sep/metals.py:56-73`` -- the standardised Shannon CN8 radius and the standardised
      square, each centred -- scaled to row norm ``sqrt(14)``, ridge 0.5 on the observed metals only);
    * only **well-determined** cells train the classifier: ``n_metals >= 5``
      (``gen14/dirbench.py:47``, ``:140``, ``:144`` ``rich_only_train``);
    * a per-extractant (here per-system) coefficient is the weighted mean of its cells' coefficients
      (``gen14/models.py:55-70`` ``extractant_units``); under gen14's chemotype-balanced cell weights every cell
      of one extractant carries the same weight, so it is the plain mean over the system's well-determined cells;
    * the estimator is ``LogisticRegression(C=1, max_iter=5000, lbfgs)`` behind a median imputer and a standard
      scaler (``gen14/models.py:76-77``, ``:123-147``; ``scripts/g14_predict.py:65-67``), and with one class in
      training it returns the constant ``y.mean()`` (``gen14/models.py:140-141``);
    * the decision is gen14's *hard* rule: ``p >= 0.5`` -> heavy-selective (``gen14/dirbench.py:248-249``).

Gen19 replaces gen14's magnitude (the training mean ``|amplitude|``) by the registered one: the training-fold
mean ``|logSF|`` per delta-Z class, over the section 2 comparable Ln(III)-Ln(III) training pairs.

Public API
----------
:func:`model_seed` / :func:`fold_index`   the section 15 model-seed rule ``42 + fold * 1009 + 9,999,991``
:func:`training_clip_range`               the gen6 clip interval of a training target
:class:`MissingPrep`                      gen6's ``DropAllNaNColumns`` + ``SimpleImputer(median, add_indicator=True)``
:class:`MonoEtConfig` / :class:`MonoEt`   the gen6 MONO_ET estimator on arrays (tabular block + wide ECFP block)
:class:`B8Level`                          the B8 log D arm (:mod:`gen19ct.models.interface` protocol, fast path
                                          ``fit_table`` / ``predict_positions`` for the conformal wrapper)
:func:`radius_basis`, :func:`curve_coefficients`, :func:`ln3_cell_table`, :func:`system_radius_coefficients`,
:func:`delta_z_magnitudes`                the gen14 quantities, each computed on training rows only
:class:`B8Direction`                      the G14 direction arm: per-system probability, per-pair logSF and
                                          direction, and the TOPO39 coverage report
:func:`fit_b8`                            both halves on one training set (:class:`B8Bundle`)
:func:`conformal_b8_level`                the level arm with split-conformal intervals cross-fitted over the inner
                                          splits of a learned-arm inner design (addendum 1 items 1-2; no tuning grid)

Everything fitted -- the feature state, the ECFP clusters, the imputation medians, the forest, the TOPO39
scaler, the logistic and the delta-Z magnitudes -- is fitted on the rows passed to ``fit`` only (brief section
12, section 2 "Features").  ``PROVENANCE_COLUMNS``, publication / study ids and row ids never become features:
:class:`gen19ct.models.features.FeatureSet` hands each block a view of its declared source columns only, and
both halves assert their own design columns with
:func:`gen19ct.models.features.assert_feature_columns_allowed` before any fit
(:meth:`B8Level._assert_feature_columns`, :meth:`B8Direction._assert_feature_columns`).  Nothing here reads a
target of a query row: the level reads ``log_D`` of its training rows, the direction reads ``log_D`` of its
training rows (the per-cell curves and the delta-Z magnitudes), and :meth:`B8Direction.pair_records` reads only
the system and the two metal states of a pair.

Readings where the registration is silent are listed in :data:`REGISTRATION_CHOICES`.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from gen19ct.chemistry import metals as MET
from gen19ct.chemistry import support_graph as SG
from gen19ct.data import normalize as N
from gen19ct.evaluation import metrics as EM
from gen19ct.evaluation import pairs as EP
from gen19ct.models import features as F
from gen19ct.models import inner_design as ID
from gen19ct.models import interface as I

SCHEMA = "gen19.b8.v1"
ARM_NAME = "B8"
DIRECTION_ARM_NAME = "B8_direction"

# --------------------------------------------------------------------------------------------- #
# level: the gen6 MONO_ET settings (section 5 B8) and the section 15 seed rule
# --------------------------------------------------------------------------------------------- #

#: ``ExtraTreesRegressor`` settings of section 5 B8 (= gen6 ``levels.LevelForestParameters`` defaults)
N_ESTIMATORS = 400
MAX_FEATURES = 0.30
MIN_SAMPLES_LEAF = 2
#: at most 2 worker processes / threads on this machine (brief conventions); gen6 used ``n_jobs=-1``
N_JOBS = 2
#: gen6 ``levels.py:661-662``: the training interval widened by ``CLIP_MARGIN * span`` on each side
CLIP_MARGIN = 0.5
#: section 15 model seed rule ``42 + fold * 1009 + 9,999,991`` (= gen7 ``harness.py:64-65, 270-273``, gen13
#: ``splits.py:27-29`` -- ``MODEL_SEED_BASE + fold * FOLD_SEED_STRIDE + FOLD_SEED_OFFSET``)
MODEL_SEED_BASE = 42
FOLD_SEED_STRIDE = 1009
FOLD_SEED_OFFSET = 9_999_991
#: the registered feature preset of B8 (``features.ARM_PRESETS["B8"]``)
FEATURE_ARM = "B8"
#: the wide feature block of the B8 preset (``features.Lig2dExtBlock``)
ECFP_BLOCK = "lig2d_ext__ecfp"

# --------------------------------------------------------------------------------------------- #
# direction: the gen14 G14 rule
# --------------------------------------------------------------------------------------------- #

#: the 14 lanthanides of the gen13 / gen14 basis (``gen13sep/metals.py:17-18``; Pm is absent from the bundle
#: and from every gen13 table, so a Pm(III) row is not part of a curve)
BASIS_METALS: tuple[str, ...] = ("La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu")
BASIS_NAMES: tuple[str, ...] = ("radius", "radius_sq")
#: ``gen13sep/basis.py:26-27``: every basis row is scaled to the norm of a standardised 14-metal curve
ROW_NORM = float(np.sqrt(len(BASIS_METALS)))
#: ``gen13sep/basis.py:28`` ``DEFAULT_RIDGE``: the per-cell ridge of the centred curve
CURVE_RIDGE = 0.5
#: ``gen14/dirbench.py:47`` ``MIN_METALS`` and section 5 B8 "training systems with >= 5 Ln(III)"
MIN_LN3_STATES = 5
#: ``gen14/models.py:144``, ``scripts/g14_predict.py:66``
LOGIT_C = 1.0
LOGIT_MAX_ITER = 5000
#: ``gen14/dirbench.py:248-249``: the hard decision rule
P_HEAVY_THRESHOLD = 0.5

CELL_KEY_COLS: tuple[str, str, str] = ("pub_group", "extractant_system_key", "condition_key")
#: why a pair has no B8 direction (``pair_records`` -> ``reason``; the coverage report counts them)
PAIR_REASONS: tuple[str, ...] = ("ok", "not_ln3_ln3", "no_topo39", "no_direction_model", "no_magnitude",
                                 "unknown_system")

#: where the registration is silent, the reading implemented here (reported with every run that uses B8)
REGISTRATION_CHOICES: dict[str, str] = {
    "clip_range": "section 5 B8 'predictions clipped to 1.5x the training range' is the gen6 rule "
                  "(levels.py:661-662, 672-674; gen7/contenders.py:214-216): clip to "
                  "[min(y) - 0.5 * span, max(y) + 0.5 * span] with span = max(y) - min(y) of the outer-training "
                  "target, an interval of total width 2 x span centred on the training range; no other gen6/gen7 "
                  "clip exists",
    "imputation": "the MONO_ET architecture includes gen6's LevelRegressor front (levels.py:630-637): columns "
                  "with no observed value in training are dropped, the rest are imputed with the training median "
                  "and every column with a missing training value gets a 0/1 indicator "
                  "(SimpleImputer(strategy='median', add_indicator=True)); medians are row-weighted",
    "design_column_order": "[the B8 preset's tabular columns in block order] + [the 2,048 ECFP bits] + [the "
                           "missing indicators]; gen6 put LIG2D_EXT inside the block order and the indicators "
                           "last.  Column order changes only which features the forest's RNG draws, not the "
                           "estimator; a bit-identical gen6 reproduction is impossible anyway (different corpus, "
                           "different columns)",
    "massaction_duplicates": "the registered B8 feature set carries log10 acid / extractant / metal molarity in "
                             "COND and again in MASSACTION (features.MassActionBlock), as section 5 B8 lists "
                             "them; the duplicates are kept",
    "model_seed": "section 15 'Model seed rule: 42 + fold * 1009 + 9,999,991'; an inner (conformal) fit of an outer "
                  "fold uses that outer fold's model seed.  The discovery runner passes fold = the position of the "
                  "(discovery seed, outer fold) pair in the design enumerated once per discovery seed "
                  "(evaluation.discovery.fold_ordinals: seed 104729 keeps the file position, s104729_S_b007 -> 7; the "
                  "other seeds get distinct forest seeds, so B8 point predictions on a seed-free fold file such as V2 "
                  "are not five copies).  fold_index() (the trailing integer of a fold id) is a helper for callers "
                  "without that enumeration",
    "threads": "the forest is fitted with n_jobs = 2 (the machine's worker cap; gen6 used -1 -- the fitted trees do "
               "not depend on it) and its trees are summed on one thread at prediction, so predictions are "
               "byte-reproducible (sklearn's threaded accumulation changes the last bits between runs)",
    "row_order": "training rows are put in canonical_measurement_id order before the fit, so a fit does not depend on "
                 "the caller's row order; the id orders rows and is never a feature",
    "variance_model": "B8 has no registered variance model: std_logD is NaN and the intervals come from the "
                      "split-conformal wrapper (section 12).  gen7's across-tree spread "
                      "(contenders.py:91-95) is not computed, so no unregistered SD enters a metric",
    "conformal": "POST-HOC addendum 1 items 1-2: B8 has no tuning grid, so its calibration is the plain split-conformal "
                 "rule over the learned arms' inner design -- under V5 inner_design.SimultaneousInnerCells (three "
                 "inner splits, one per inner fold, all of the fold's cells hidden), under V1 / V2 the grouped / "
                 "metal inner design -- the level refitted once per inner split on that split's training rows with the "
                 "outer fold's model seed (reading 6(b)), the absolute residuals of the three splits pooled, the "
                 "interval centre the refit on all outer-training rows; 'cross-fitted' is trivial without a selection "
                 "step (no residual comes from a fit that saw its row).  The per-cell InnerCellCalibration is refused "
                 "(conformal_b8_level)",
    "cell_key": "a gen14 cell is (publication, exactly matched conditions, extractant); Gen19 keys it with its "
                "own registered units -- the section 2 publication group, extractant_system_key and "
                "normalize.condition_key (the comparable-pair key, metal concentration included).  Replicates of "
                "one (cell, metal state) are averaged, as gen13 cohort.py:159",
    "direction_unit": "the logistic's training unit is the SYSTEM (section 5 B8 'training systems with >= 5 "
                      "Ln(III)'): a system qualifies when its training rows carry >= 5 distinct Ln(III) states of "
                      "the 14 basis metals AND it has at least one cell with >= 5 of them (gen14's rich-cell rule, "
                      "the only definition of the coefficient); its coefficient is the mean over those cells "
                      "(gen14 extractant_units).  A system above the state count with no such cell has no "
                      "well-determined curve and is left out, counted as 'no_rich_cell' (1 system on the full "
                      "MODEL corpus: 83 systems reach 5 states, 82 have a >= 5-state cell)",
    "direction_weights": "the logistic is fitted unweighted, one row per training system: section 5 B8 registers "
                         "no weights, and gen14's chemotype-balanced cell weights are constant within an "
                         "extractant, so they only weighted extractants against each other through their cell "
                         "counts -- which the per-system unit removes",
    "direction_training_features": "a training system without TOPO39 is not a fit unit (it has no 39 columns to "
                                   "impute from); Topo39Prep's median and scale are fitted on the fit units only, "
                                   "each system once",
    "pm_excluded": "Pm(III) is not a basis metal (no gen13 radius), so a Pm(III) row never enters a curve and "
                   "does not count toward a system's >= 5 Ln(III) states; a Pm(III) pair still gets a direction "
                   "and a magnitude, because its delta Z is defined",
    "magnitude_pairs": "the magnitude is the mean |observed logSF| over the section 2 comparable Ln(III)-Ln(III) "
                       "TRAINING pairs of the fold, one value per delta Z = |Z_a - Z_b| class, pair-weighted "
                       "(every pair once, over every training system -- not only the TOPO39 ones)",
    "magnitude_fallback": "a query delta-Z class with no training pair takes the pooled mean over every training "
                          "Ln(III)-Ln(III) pair (flag magnitude_pooled); with no training pair at all the pair is "
                          "undefined",
    "direction_undefined": "a pair with a non-Ln(III) member, a system without TOPO39 or a fold whose training "
                           "rows produced no fit unit gets logsf_pred NaN -- undefined, excluded and counted, as "
                           "the section 4 HEAVIER rule",
}


def _missing(v: Any) -> bool:
    return SG._missing(v)


# --------------------------------------------------------------------------------------------- #
# seeds and the training range
# --------------------------------------------------------------------------------------------- #

def model_seed(fold: int) -> int:
    """Section 15 model seed rule: ``42 + fold * 1009 + 9,999,991`` (gen7 ``harness.py:64-65, 270-273``)."""
    f = int(fold)
    if f < 0:
        raise ValueError(f"fold index must be >= 0, got {fold!r}")
    return int(MODEL_SEED_BASE + f * FOLD_SEED_STRIDE + FOLD_SEED_OFFSET)


def fold_index(fold_id: str) -> int:
    """The fold index of a registered fold id: its trailing integer (``s104729_S_b007`` -> 7)."""
    m = re.search(r"(\d+)\s*$", str(fold_id))
    if m is None:
        raise ValueError(f"fold id {fold_id!r} has no trailing fold index; pass fold= explicitly")
    return int(m.group(1))


def training_clip_range(y: Any) -> tuple[float, float]:
    """gen6 ``levels.py:661-662``: ``(min - 0.5 * span, max + 0.5 * span)`` with ``span = max - min``."""
    v = np.asarray(pd.to_numeric(pd.Series(np.asarray(y).ravel()), errors="coerce").to_numpy(dtype=float))
    if not len(v) or not np.isfinite(v).all():
        raise ValueError("training_clip_range: a finite training target is required")
    lo, hi = float(v.min()), float(v.max())
    span = hi - lo
    return lo - CLIP_MARGIN * span, hi + CLIP_MARGIN * span


# --------------------------------------------------------------------------------------------- #
# gen6's imputation front
# --------------------------------------------------------------------------------------------- #

class MissingPrep:
    """gen6 ``levels.LevelRegressor``'s front, fitted on training rows only.

    ``DropAllNaNColumns`` (``levels.py:584-603``: drop every column with no observed training value; raise when
    that is all of them) followed by ``SimpleImputer(strategy="median", add_indicator=True)``
    (``levels.py:636``): the training median fills a missing value, and every column with a missing training
    value gets a ``<col>__missing`` 0/1 column appended after the imputed ones (sklearn's ``features="missing-only"``
    rule; a column whose missingness appears only at transform is imputed without an indicator, as
    ``SimpleImputer`` builds its ``MissingIndicator`` with ``error_on_new=False``).

    Medians are summed over sorted values, so a fit does not depend on the row order.
    """

    def __init__(self) -> None:
        self.columns: tuple[str, ...] = ()
        self.kept: tuple[str, ...] = ()
        self.dropped_all_nan: tuple[str, ...] = ()
        self.medians: dict[str, float] = {}
        self.indicator_for: tuple[str, ...] = ()
        self.n_fit_rows = 0

    # ----------------------------------------------------------------------------------------- #
    @staticmethod
    def _matrix(frame: pd.DataFrame) -> np.ndarray:
        x = frame.to_numpy(dtype=float, copy=True)
        x[~np.isfinite(x)] = np.nan
        return x

    def fit(self, frame: pd.DataFrame) -> "MissingPrep":
        self.columns = tuple(str(c) for c in frame.columns)
        x = self._matrix(frame)
        if not len(x):
            raise ValueError("MissingPrep.fit: no rows")
        observed = ~np.isnan(x)
        keep = observed.any(axis=0)
        self.kept = tuple(c for c, k in zip(self.columns, keep) if k)
        self.dropped_all_nan = tuple(c for c, k in zip(self.columns, keep) if not k)
        if not self.kept:
            raise ValueError("every feature column is empty in this training fold")   # levels.py:596
        med, ind = {}, []
        for j, c in enumerate(self.columns):
            if not keep[j]:
                continue
            col = x[:, j]
            fin = np.sort(col[observed[:, j]])
            med[c] = float(np.median(fin))
            if len(fin) < len(col):
                ind.append(c)
        self.medians, self.indicator_for = med, tuple(ind)
        self.n_fit_rows = int(len(x))
        return self

    @property
    def fitted(self) -> bool:
        return bool(self.kept)

    @property
    def output_columns(self) -> tuple[str, ...]:
        return tuple(self.kept) + tuple(f"{c}__missing" for c in self.indicator_for)

    def transform(self, frame: pd.DataFrame, dtype: Any = np.float32) -> np.ndarray:
        """The imputed kept columns followed by the indicator columns (``(n, len(output_columns))``)."""
        if not self.fitted:
            raise RuntimeError("MissingPrep: fit first")
        missing = [c for c in self.kept if c not in frame.columns]
        if missing:
            raise KeyError(f"MissingPrep.transform: columns missing {missing[:6]}")
        sub = frame.loc[:, list(self.kept)]
        x = self._matrix(sub)
        na = np.isnan(x)
        med = np.array([self.medians[c] for c in self.kept], dtype=float)
        x = np.where(na, med[None, :], x)
        out = [x.astype(dtype, copy=False)]
        if self.indicator_for:
            pos = [self.kept.index(c) for c in self.indicator_for]
            out.append(na[:, pos].astype(dtype))
        return np.hstack(out) if len(out) > 1 else out[0]

    def state(self) -> dict[str, Any]:
        return {"columns": list(self.columns), "kept": list(self.kept),
                "dropped_all_nan": list(self.dropped_all_nan), "medians": self.medians,
                "indicator_for": list(self.indicator_for), "n_fit_rows": self.n_fit_rows}


# --------------------------------------------------------------------------------------------- #
# the MONO_ET estimator
# --------------------------------------------------------------------------------------------- #

@dataclass(frozen=True)
class MonoEtConfig:
    """Section 5 B8 forest settings (the registered values are the defaults)."""

    n_estimators: int = N_ESTIMATORS
    max_features: float = MAX_FEATURES
    min_samples_leaf: int = MIN_SAMPLES_LEAF
    n_jobs: int = N_JOBS

    @property
    def registered(self) -> bool:
        """True when the forest settings are the registered ones (``n_jobs`` is a machine choice)."""
        return (int(self.n_estimators) == N_ESTIMATORS and float(self.max_features) == MAX_FEATURES
                and int(self.min_samples_leaf) == MIN_SAMPLES_LEAF)

    def state(self) -> dict[str, Any]:
        return {"n_estimators": int(self.n_estimators), "max_features": float(self.max_features),
                "min_samples_leaf": int(self.min_samples_leaf), "n_jobs": int(self.n_jobs),
                "registered": self.registered}


def _wide_block(wide: Any, rows: int, dtype: Any = np.float32) -> np.ndarray | None:
    """A wide block (dense array or scipy sparse matrix) as a dense ``dtype`` array; None stays None."""
    if wide is None:
        return None
    if hasattr(wide, "toarray"):
        wide = wide.toarray()
    arr = np.asarray(wide)
    if arr.ndim != 2 or len(arr) != rows:
        raise ValueError(f"wide block shape {arr.shape} does not match {rows} rows")
    out = arr.astype(dtype, copy=False)
    if np.isnan(out).any():
        raise ValueError("the wide block must not contain NaN (the ECFP bit block never does)")
    return out


class MonoEt:
    """gen6's ``MONO_ET``: :class:`MissingPrep` on the tabular block, the NaN-free wide block appended, an
    ``ExtraTreesRegressor`` with ECFP-cluster sample weights, predictions clipped to the training range.

    ``fit(tab, y, wide=..., sample_weight=...)`` -- ``tab`` is a DataFrame that may hold NaN, ``wide`` a dense or
    sparse NaN-free matrix (the 2,048 ECFP bits).  :meth:`predict_raw` is the forest's own output,
    :meth:`predict` the clipped one (``levels.py:672-674``).
    """

    def __init__(self, random_state: int, config: MonoEtConfig | None = None):
        self.random_state = int(random_state)
        self.config = config or MonoEtConfig()
        self.prep = MissingPrep()
        self.forest: Any = None
        self.y_range: tuple[float, float] | None = None
        self.n_wide = 0
        self.n_train_rows = 0
        self.total_sample_weight: float | None = None

    # ----------------------------------------------------------------------------------------- #
    def design(self, tab: pd.DataFrame, wide: Any = None) -> np.ndarray:
        """``[imputed kept tabular | wide | missing indicators]`` as one ``float32`` matrix."""
        t = self.prep.transform(tab)
        w = _wide_block(wide, len(tab))
        if (w is None) != (self.n_wide == 0):
            raise ValueError("the wide block is present at fit but not at transform (or the other way round)")
        if w is None:
            return t
        if w.shape[1] != self.n_wide:
            raise ValueError(f"wide block has {w.shape[1]} columns, fitted with {self.n_wide}")
        n_ind = len(self.prep.indicator_for)
        if not n_ind:
            return np.hstack([t, w])
        return np.hstack([t[:, :t.shape[1] - n_ind], w, t[:, t.shape[1] - n_ind:]])

    def design_columns(self, wide_columns: Sequence[str] = ()) -> tuple[str, ...]:
        cols = list(self.prep.kept)
        ind = [f"{c}__missing" for c in self.prep.indicator_for]
        return tuple(cols + list(wide_columns) + ind)

    def fit(self, tab: pd.DataFrame, y: Any, *, wide: Any = None, sample_weight: Any = None) -> "MonoEt":
        from sklearn.ensemble import ExtraTreesRegressor

        yv = np.asarray(pd.to_numeric(pd.Series(np.asarray(y).ravel()), errors="coerce").to_numpy(dtype=float))
        if len(yv) != len(tab):
            raise ValueError("tab and y must have equal length")
        if not len(yv):
            raise ValueError("MonoEt.fit: no training rows")
        if not np.isfinite(yv).all():
            raise ValueError("MonoEt.fit: a training target is not finite")
        self.prep = MissingPrep().fit(tab)
        w = _wide_block(wide, len(tab))
        self.n_wide = 0 if w is None else int(w.shape[1])
        x = self.design(tab, wide)
        sw = None
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=float).ravel()
            if len(sw) != len(yv):
                raise ValueError("sample_weight must have one value per training row")
            if not np.isfinite(sw).all() or (sw < 0).any():
                raise ValueError("sample weights must be finite and non-negative")
            self.total_sample_weight = float(sw.sum())
        self.forest = ExtraTreesRegressor(n_estimators=int(self.config.n_estimators),
                                          max_features=float(self.config.max_features),
                                          min_samples_leaf=int(self.config.min_samples_leaf),
                                          random_state=self.random_state, n_jobs=int(self.config.n_jobs))
        self.forest.fit(x, yv, sample_weight=sw)
        self.y_range = training_clip_range(yv)
        self.n_train_rows = int(len(yv))
        return self

    @property
    def fitted(self) -> bool:
        return self.forest is not None

    def _require_fit(self) -> None:
        if not self.fitted:
            raise RuntimeError("MonoEt: fit first")

    def predict_raw(self, tab: pd.DataFrame, wide: Any = None) -> np.ndarray:
        """The forest's own output.  Trees are summed on ONE thread: sklearn accumulates a multi-threaded forest
        prediction in thread-completion order, which changes the last bits between runs; the fitted trees do not
        depend on ``n_jobs`` (every tree carries its own seed), so this makes predictions byte-reproducible."""
        self._require_fit()
        x = self.design(tab, wide)
        n_jobs = self.forest.n_jobs
        try:
            self.forest.set_params(n_jobs=1)
            return np.asarray(self.forest.predict(x), dtype=float)
        finally:
            self.forest.set_params(n_jobs=n_jobs)

    def clip(self, values: Any) -> np.ndarray:
        """``np.clip`` to the training range (``levels.py:672-674``)."""
        self._require_fit()
        lo, hi = self.y_range                                       # type: ignore[misc]
        return np.clip(np.asarray(values, dtype=float), lo, hi)

    def predict(self, tab: pd.DataFrame, wide: Any = None) -> np.ndarray:
        return self.clip(self.predict_raw(tab, wide))

    def state(self) -> dict[str, Any]:
        return {"random_state": self.random_state, "config": self.config.state(), "prep": self.prep.state(),
                "y_range": None if self.y_range is None else [float(self.y_range[0]), float(self.y_range[1])],
                "n_wide": self.n_wide, "n_train_rows": self.n_train_rows,
                "total_sample_weight": self.total_sample_weight}


# --------------------------------------------------------------------------------------------- #
# B8 level arm
# --------------------------------------------------------------------------------------------- #

#: B8-specific diagnostic columns appended to :data:`gen19ct.models.interface.PREDICTION_COLUMNS`
B8_DIAGNOSTIC_COLUMNS: tuple[str, ...] = ("b8_raw_mean", "b8_clipped", "b8_fingerprint_missing", "b8_system_seen")


class B8Level:
    """The B8 log D arm: gen6's MONO_ET on the registered Gen19 B8 feature blocks.

    ``frame`` is the arm frame (:func:`gen19ct.models.interface.prepare_frame` layout) over a superset of every
    row this arm will be fitted on or asked about; it is what makes the :class:`~gen19ct.models.interface.
    ConformalWrapper` fast path (``fit_table`` / ``predict_positions``, which pass table positions rather than
    frames) possible.  ``cv`` is an optional precomputed ``normalize.condition_vector`` over those rows (a
    row-wise, target-free transform; it saves recomputing it per fold).

    The forest seed is ``model_seed(fold)`` (section 15) -- pass ``fold`` (the fold index, or use
    :func:`fold_index` on a registered fold id) or an explicit ``random_state``.
    """

    name = ARM_NAME

    def __init__(self, frame: pd.DataFrame | None = None, *, cv: pd.DataFrame | None = None,
                 fold: int | None = None, random_state: int | None = None, config: MonoEtConfig | None = None,
                 ecfp_format: str = "dense"):
        if fold is None and random_state is None:
            raise ValueError("B8Level needs fold= (section 15 model seed rule) or an explicit random_state=")
        if ecfp_format not in ("dense", "sparse"):
            raise ValueError(f"ecfp_format {ecfp_format!r}")
        self.frame = frame
        self.cv = cv
        self.fold = None if fold is None else int(fold)
        self._random_state = None if random_state is None else int(random_state)
        self.config = config or MonoEtConfig()
        self.ecfp_format = ecfp_format
        self.features: F.FeatureSet | None = None
        self.clusters: F.EcfpClusters | None = None
        self.model: MonoEt | None = None
        self.train_index: pd.Index | None = None
        self.hidden_index: pd.Index | None = None
        self.wide_columns: tuple[str, ...] = ()

    # ----------------------------------------------------------------------------------------- #
    @property
    def random_state(self) -> int:
        return self._random_state if self._random_state is not None else model_seed(self.fold)  # type: ignore[arg-type]

    def clone(self) -> "B8Level":
        return B8Level(self.frame, cv=self.cv, fold=self.fold, random_state=self._random_state,
                       config=self.config, ecfp_format=self.ecfp_format)

    @property
    def fitted(self) -> bool:
        return self.model is not None and self.model.fitted

    def _require_fit(self) -> MonoEt:
        if not self.fitted:
            raise RuntimeError(f"{self.name}: fit first")
        return self.model                                             # type: ignore[return-value]

    # ----------------------------------------------------------------------------------------- #
    def _cv_for(self, rows: pd.DataFrame) -> pd.DataFrame:
        if self.cv is not None:
            if not rows.index.isin(self.cv.index).all():
                raise KeyError("B8Level: the precomputed condition vector does not cover these rows")
            return self.cv
        return N.condition_vector(rows)

    def _new_feature_set(self) -> F.FeatureSet:
        preset = dict(F.ARM_PRESETS[FEATURE_ARM])
        preset["ecfp_format"] = self.ecfp_format
        return F.FeatureSet(**preset, name=FEATURE_ARM)

    @staticmethod
    def _assert_feature_columns(fm: F.FeatureMatrix, prep_columns: Iterable[str] = ()) -> None:
        """Explicit section 2 'Features' guard: no provenance / id / target column, in the tabular block, the
        wide block or the design matrix (``FeatureSet`` checks its own outputs; this re-checks ours)."""
        F.assert_feature_columns_allowed(fm.frame.columns, "B8 tabular feature columns")
        F.assert_feature_columns_allowed(fm.wide_columns(), "B8 ECFP feature columns")
        F.assert_feature_columns_allowed([c for c in prep_columns], "B8 design columns")

    @staticmethod
    def _wide_of(fm: F.FeatureMatrix) -> tuple[Any, tuple[str, ...]]:
        if ECFP_BLOCK not in fm.wide:
            raise KeyError(f"the B8 feature preset must carry the {ECFP_BLOCK} block")
        if len(fm.wide) != 1:
            raise KeyError(f"expected one wide block, got {sorted(fm.wide)}")
        return fm.wide[ECFP_BLOCK]

    def _check_training(self, rows: pd.DataFrame, context: I.FitContext | None) -> None:
        if not len(rows):
            raise ValueError(f"{self.name}: no training rows")
        if rows.index.has_duplicates:
            raise ValueError(f"{self.name}: the training index must be unique")
        y = pd.to_numeric(rows[I.TARGET_COL], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(y).all():
            raise ValueError(f"{self.name}: training rows with a non-finite {I.TARGET_COL}")
        hid = None if context is None else context.hidden_index
        if hid is not None and len(hid):
            bad = rows.index.intersection(pd.Index(hid))
            if len(bad):
                raise AssertionError(f"{self.name}: {len(bad)} hidden row(s) among the training rows")
            self.hidden_index = pd.Index(hid)

    # ----------------------------------------------------------------------------------------- #
    def fit(self, train_rows: pd.DataFrame, context: I.FitContext | None = None) -> "B8Level":
        """Fit the B8 features, the ECFP clusters, gen6's imputation front and the forest on ``train_rows`` only.

        Rows are put in ``canonical_measurement_id`` order first (label order without that column), so the fit
        does not depend on the caller's row order; the id orders rows and is never a feature."""
        self._check_training(train_rows, context)
        self._table = None
        order = (train_rows[I.ID_COL].astype(str).to_numpy() if I.ID_COL in train_rows.columns
                 else train_rows.index.astype(str).to_numpy())
        rows = train_rows.iloc[np.argsort(order, kind="stable")]
        cv = self._cv_for(rows)
        fs = self._new_feature_set().fit(rows, cv)
        fm = fs.transform(rows, cv)
        bits, wide_cols = self._wide_of(fm)
        self._assert_feature_columns(fm)                              # before anything is fitted on them
        clusters = F.EcfpClusters().fit(rows)
        weights = clusters.sample_weights(rows.index)
        model = MonoEt(self.random_state, self.config)
        model.fit(fm.frame, rows[I.TARGET_COL], wide=bits, sample_weight=weights)
        self._assert_feature_columns(fm, model.design_columns(wide_cols))
        self.features, self.clusters, self.model = fs, clusters, model
        self.wide_columns = tuple(wide_cols)
        self.train_index = rows.index
        return self

    def _rows_of(self, table: I.RowTable, labels: pd.Index, what: str, *, training: bool) -> pd.DataFrame:
        """The arm-frame rows of table labels; for TRAINING rows the frame's target must equal the table's (a query
        row's target is never read)."""
        if self.frame is None:
            raise ValueError("B8Level needs frame= (the arm frame) for the table fast path")
        missing = labels.difference(self.frame.index)
        if len(missing):
            raise KeyError(f"{what}: {len(missing)} row label(s) are not in the arm frame")
        rows = self.frame.loc[labels]
        if training:
            pos = table.index.get_indexer(labels)
            y = pd.to_numeric(rows[I.TARGET_COL], errors="coerce").to_numpy(dtype=float)
            if not np.array_equal(table.y[pos], y, equal_nan=True):
                raise ValueError(f"{what}: the arm frame disagrees with the RowTable target")
        return rows

    def fit_table(self, table: I.RowTable, mask: np.ndarray, context: I.FitContext) -> "B8Level":
        m = np.asarray(mask, dtype=bool)
        if m.shape != (table.n,):
            raise ValueError("training mask does not match the RowTable")
        if not m.any():
            raise ValueError(f"{self.name}: no training rows")
        forbidden = I.forbidden_mask(table, context)
        if (m & forbidden).any():
            raise AssertionError(f"{self.name}: {int((m & forbidden).sum())} hidden row(s) among the training rows")
        labels = table.index[m]
        self.fit(self._rows_of(table, labels, f"{self.name}.fit_table", training=True), context)
        self._table = table
        return self

    # ----------------------------------------------------------------------------------------- #
    def _records(self, query_rows: pd.DataFrame) -> pd.DataFrame:
        model = self._require_fit()
        fs = self.features
        if fs is None:
            raise RuntimeError(f"{self.name}: fit first")
        if query_rows.index.has_duplicates:
            raise ValueError("query rows: the index must be unique")
        if self.train_index is not None:
            overlap = query_rows.index.intersection(self.train_index)
            if len(overlap):
                raise AssertionError(f"{self.name}: {len(overlap)} query row(s) are training rows")
        cv = self._cv_for(query_rows)
        fm = fs.transform(query_rows, cv)
        bits, wide_cols = self._wide_of(fm)
        self._assert_feature_columns(fm, model.design_columns(wide_cols))
        raw = model.predict_raw(fm.frame, bits)
        mean = model.clip(raw)
        clipped = raw != mean
        fp_missing = fm.diagnostics["lig2d_fingerprint_missing"].to_numpy(dtype=bool) \
            if "lig2d_fingerprint_missing" in fm.diagnostics.columns else np.zeros(len(query_rows), dtype=bool)
        seen = self._system_seen(query_rows)
        recs = []
        for k, label in enumerate(query_rows.index):
            rec = I.empty_prediction_record()
            reason = [] if not clipped[k] else ["clipped_to_training_range"]
            if fp_missing[k]:
                reason.append("primary_fingerprint_missing")
            rec.update(row_id=label, mean_logD=float(mean[k]), std_logD=np.nan, fallback_level=self.name,
                       fallback_reason=";".join(reason))
            recs.append(rec)
        out = I.records_to_frame(recs)
        out["b8_raw_mean"] = raw
        out["b8_clipped"] = clipped
        out["b8_fingerprint_missing"] = fp_missing
        out["b8_system_seen"] = seen
        return out

    def _system_seen(self, rows: pd.DataFrame) -> np.ndarray:
        cl = self.clusters
        if cl is None:
            return np.zeros(len(rows), dtype=bool)
        keys = rows[SG.SYSTEM_COL].to_numpy(dtype=object)
        known = set(cl.system_labels)
        return np.array([(not _missing(k)) and str(k) in known for k in keys], dtype=bool)

    def predict(self, query_rows: pd.DataFrame) -> pd.DataFrame:
        """One row per query row, in query order: :data:`gen19ct.models.interface.PREDICTION_COLUMNS` plus
        :data:`B8_DIAGNOSTIC_COLUMNS`.  ``std_logD`` is NaN (B8 has no registered variance model)."""
        return self._records(query_rows)

    def predict_positions(self, positions: np.ndarray) -> pd.DataFrame:
        """As :meth:`predict`, on positions of the ``RowTable`` of the last :meth:`fit_table`."""
        if self.frame is None or self._table is None:
            raise ValueError("predict_positions follows fit_table (the RowTable fast path) on B8Level(frame=...)")
        pos = np.asarray(positions, dtype=np.int64)
        labels = self._table.index[pos]
        return self._records(self._rows_of(self._table, labels, f"{self.name}.predict_positions", training=False))

    #: the table the fast path was fitted on (set by :meth:`fit_table`, cleared by :meth:`fit`)
    _table: I.RowTable | None = None

    def state(self) -> dict[str, Any]:
        model = self.model
        return {"schema": SCHEMA, "arm": self.name, "random_state": self.random_state, "fold": self.fold,
                "features": None if self.features is None else self.features.state_digest,
                "clusters": None if self.clusters is None else self.clusters.state(),
                "model": None if model is None else model.state(),
                "n_train_rows": 0 if model is None else model.n_train_rows,
                "ecfp_format": self.ecfp_format}

    @property
    def state_digest(self) -> str:
        return F.state_digest(self.state())


# --------------------------------------------------------------------------------------------- #
# the gen14 lanthanide-axis coefficient
# --------------------------------------------------------------------------------------------- #

def _standardise(values: np.ndarray) -> np.ndarray:
    """gen13 ``metals._standardise``: zero mean, population SD."""
    v = np.asarray(values, dtype=float)
    return (v - v.mean()) / v.std()


def basis_radii(metals: Sequence[str] = BASIS_METALS) -> np.ndarray:
    """Shannon CN8 radii of the basis metals from ``descriptors/metals.csv`` (equal to gen13
    ``metals.SHANNON_RADIUS_CN8``, asserted by ``tests/test_previous_gen.py``)."""
    tab = F.static_tables().metals
    out = []
    for m in metals:
        label = f"{m}(III)"
        if label not in tab.index:
            raise KeyError(f"{label} is absent from metals.csv")
        r = float(tab.loc[label, "radius_cn8_A"])
        if not np.isfinite(r):
            raise ValueError(f"{label} has no CN8 radius")
        out.append(r)
    return np.array(out, dtype=float)


def radius_basis(metals: Sequence[str] = BASIS_METALS) -> np.ndarray:
    """The gen13 / gen14 physics basis ``("radius", "radius_sq")`` as a ``(2, len(metals))`` matrix.

    ``gen13sep/metals.py:56-73`` and ``gen13sep/basis.py:35-37, 99-104``: ``z`` = the standardised Shannon CN8
    radius, rows ``z`` and ``standardise(z**2)``, each centred, then scaled to row norm ``sqrt(len(metals))``.
    """
    z = _standardise(basis_radii(metals))
    rows = np.vstack([z, _standardise(z ** 2)])
    rows = rows - rows.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.maximum(norms, 1e-12) * float(np.sqrt(len(metals)))


def curve_coefficients(centred: np.ndarray, basis: np.ndarray | None = None,
                       ridge: float = CURVE_RIDGE) -> np.ndarray:
    """Per-cell ridge coefficients of centred curves on the observed metals only (``gen13sep/basis.py:107-117``).

    ``centred`` is ``(cells, n_metals)`` with NaN where a metal was not measured; the result is ``(cells, 2)``.
    """
    B = radius_basis() if basis is None else np.asarray(basis, dtype=float)
    c = np.atleast_2d(np.asarray(centred, dtype=float))
    if c.shape[1] != B.shape[1]:
        raise ValueError(f"centred curves have {c.shape[1]} metals, the basis {B.shape[1]}")
    k = B.shape[0]
    out = np.full((len(c), k), np.nan)
    for i in range(len(c)):
        m = np.isfinite(c[i])
        if not m.any():
            continue
        X = B[:, m].T
        out[i] = np.linalg.solve(X.T @ X + ridge * np.eye(k), X.T @ c[i][m])
    return out


def ln3_states(states: Iterable[Any], metals: Sequence[str] = BASIS_METALS) -> np.ndarray:
    """Boolean mask: the label is ``<basis metal>(III)``."""
    keep = {f"{m}(III)" for m in metals}
    return np.array([isinstance(s, str) and s in keep for s in states], dtype=bool)


def ln3_cell_table(rows: pd.DataFrame, *, condition_key: pd.Series | None = None,
                   pub_group_col: str = I.PUB_GROUP_COL, min_states: int = MIN_LN3_STATES,
                   metals: Sequence[str] = BASIS_METALS) -> pd.DataFrame:
    """The gen14 cells of ``rows`` and their radius coefficients (one row per cell with >= ``min_states``).

    A cell is ``(publication group, extractant_system_key, condition_key)``; replicates of one (cell, metal
    state) are averaged; the curve is centred over the cell's observed basis metals and fitted by the per-cell
    ridge of :func:`curve_coefficients`.  Columns: the cell key, ``n_states``, ``amp`` (the radius coefficient),
    ``curvature`` and ``heavy_selective`` (``amp < 0``, gen14's label).
    """
    need = [pub_group_col, SG.SYSTEM_COL, SG.METAL_COL, I.TARGET_COL]
    miss = [c for c in need if c not in rows.columns]
    if miss:
        raise KeyError(f"ln3_cell_table: columns missing {miss}")
    ck = N.condition_key(rows) if condition_key is None else condition_key.reindex(rows.index)
    if ck.isna().any():
        raise KeyError("ln3_cell_table: condition_key does not cover every row")
    keep = ln3_states(rows[SG.METAL_COL].to_numpy(dtype=object), metals)
    cols = list(metals)
    empty = pd.DataFrame({**{c: pd.Series(dtype=object) for c in CELL_KEY_COLS},
                          "n_states": pd.Series(dtype=int), "amp": pd.Series(dtype=float),
                          "curvature": pd.Series(dtype=float), "heavy_selective": pd.Series(dtype=bool)})
    if not keep.any():
        return empty
    sub = pd.DataFrame({CELL_KEY_COLS[0]: rows.loc[keep, pub_group_col].astype(object).to_numpy(),
                        CELL_KEY_COLS[1]: rows.loc[keep, SG.SYSTEM_COL].astype(object).to_numpy(),
                        CELL_KEY_COLS[2]: ck[keep].astype(object).to_numpy(),
                        "state": rows.loc[keep, SG.METAL_COL].astype(object).to_numpy(),
                        "y": pd.to_numeric(rows.loc[keep, I.TARGET_COL], errors="coerce").to_numpy(dtype=float)})
    if sub[list(CELL_KEY_COLS)].isna().any().any():
        raise ValueError("ln3_cell_table: a cell key column is missing on an Ln(III) row")
    if not np.isfinite(sub["y"]).all():
        raise ValueError(f"ln3_cell_table: a non-finite {I.TARGET_COL} on an Ln(III) training row")
    sub["metal"] = [s[:-5] for s in sub["state"]]
    means = (sub.groupby(list(CELL_KEY_COLS) + ["metal"], sort=True)["y"].mean()
             .unstack("metal").reindex(columns=cols))
    Y = means.to_numpy(dtype=float)
    n_states = np.isfinite(Y).sum(axis=1)
    take = n_states >= int(min_states)
    if not take.any():
        return empty
    Y, n_states = Y[take], n_states[take]
    centred = Y - np.nanmean(Y, axis=1, keepdims=True)
    coef = curve_coefficients(centred)
    out = means.index[take].to_frame(index=False)
    out["n_states"] = n_states.astype(int)
    out["amp"] = coef[:, 0]
    out["curvature"] = coef[:, 1]
    out["heavy_selective"] = coef[:, 0] < 0
    return out.reset_index(drop=True)


def system_radius_coefficients(rows: pd.DataFrame, *, condition_key: pd.Series | None = None,
                               pub_group_col: str = I.PUB_GROUP_COL, min_states: int = MIN_LN3_STATES,
                               metals: Sequence[str] = BASIS_METALS) -> pd.DataFrame:
    """One row per extractant system of ``rows``: its Ln(III) state count, its gen14 radius coefficient and its
    direction label (:data:`REGISTRATION_CHOICES` ``direction_unit``).

    Columns: ``extractant_system_key``, ``n_ln3_states`` (distinct basis-metal Ln(III) states in the rows),
    ``n_rich_cells``, ``amp`` (mean over those cells), ``heavy_selective``, ``status``
    (``fit_unit`` / ``no_rich_cell`` / ``below_min_states``).
    """
    cells = ln3_cell_table(rows, condition_key=condition_key, pub_group_col=pub_group_col, min_states=min_states,
                           metals=metals)
    keep = ln3_states(rows[SG.METAL_COL].to_numpy(dtype=object), metals)
    ln3 = rows.loc[keep, [SG.SYSTEM_COL, SG.METAL_COL]]
    systems = sorted({str(s) for s in rows[SG.SYSTEM_COL].to_numpy(dtype=object) if not _missing(s)})
    n_states = ln3.groupby(SG.SYSTEM_COL, sort=True)[SG.METAL_COL].nunique() if len(ln3) else pd.Series(dtype=int)
    grouped = cells.groupby(CELL_KEY_COLS[1], sort=True)["amp"].agg(["mean", "size"]) if len(cells) else None
    recs = []
    for s in systems:
        ns = int(n_states.get(s, 0))
        amp = float("nan")
        n_cells = 0
        if grouped is not None and s in grouped.index:
            amp = float(grouped.loc[s, "mean"])
            n_cells = int(grouped.loc[s, "size"])
        if ns < int(min_states):
            status = "below_min_states"
        elif n_cells == 0:
            status = "no_rich_cell"
        else:
            status = "fit_unit"
        recs.append({SG.SYSTEM_COL: s, "n_ln3_states": ns, "n_rich_cells": n_cells,
                     "amp": amp if status == "fit_unit" else float("nan"),
                     "heavy_selective": bool(amp < 0) if status == "fit_unit" else False,
                     "status": status})
    return pd.DataFrame(recs, columns=[SG.SYSTEM_COL, "n_ln3_states", "n_rich_cells", "amp", "heavy_selective",
                                       "status"])


# --------------------------------------------------------------------------------------------- #
# the registered magnitude: training mean |logSF| per delta-Z class
# --------------------------------------------------------------------------------------------- #

def _topo39_available(systems: Iterable[str]) -> dict[str, bool]:
    """``{system: TOPO39 row present}`` over the distinct systems (``features.topo39_for_systems``)."""
    uniq = sorted({str(s) for s in systems})
    if not uniq:
        return {}
    t = F.topo39_for_systems(uniq)
    return {k: bool(v) for k, v in zip(uniq, t["topo39_available"].to_numpy(dtype=bool))}


def _atomic_number(state: str) -> int:
    el, _ = EP.parse_state(state)
    return int(MET.ATOMIC_NUMBER[el])


def _is_ln3(state: Any) -> bool:
    if not isinstance(state, str):
        return False
    try:
        el, ox = EP.parse_state(state)
    except ValueError:
        return False
    return ox == 3 and el in MET.LANTHANIDES


def training_ln3_pairs(rows: pd.DataFrame, *, condition_key: pd.Series | None = None,
                       pub_group_col: str = I.PUB_GROUP_COL) -> pd.DataFrame:
    """The section 2 comparable Ln(III)-Ln(III) pairs INSIDE the training rows (``pairs.comparable_pairs``, which
    runs ``leakage.pair_isolation_check``), with ``delta_z = Z_a - Z_b >= 1``."""
    need = [pub_group_col, SG.SYSTEM_COL, SG.METAL_COL, I.TARGET_COL]
    miss = [c for c in need if c not in rows.columns]
    if miss:
        raise KeyError(f"training_ln3_pairs: columns missing {miss}")
    ck = N.condition_key(rows) if condition_key is None else condition_key.reindex(rows.index)
    if ck.isna().any():
        raise KeyError("training_ln3_pairs: condition_key does not cover every row")
    keep = np.array([_is_ln3(s) for s in rows[SG.METAL_COL].to_numpy(dtype=object)], dtype=bool)
    cols = ["fold", EM.PUB_GROUP_COL, EM.SYSTEM_COL, EM.CONDITION_KEY_COL, EM.METAL_STATE_COL, EM.Y_COL]
    if not keep.any():
        return pd.DataFrame(columns=["idx_a", "idx_b", "fold", *EP.PAIR_KEY_COLS, "state_a", "state_b",
                                     "category_class", "y_a", "y_b", "logsf_obs", "delta_z"])
    slim = pd.DataFrame({"fold": "train",
                         EM.PUB_GROUP_COL: rows.loc[keep, pub_group_col].astype(object).to_numpy(),
                         EM.SYSTEM_COL: rows.loc[keep, SG.SYSTEM_COL].astype(object).to_numpy(),
                         EM.CONDITION_KEY_COL: ck[keep].astype(object).to_numpy(),
                         EM.METAL_STATE_COL: rows.loc[keep, SG.METAL_COL].astype(object).to_numpy(),
                         EM.Y_COL: pd.to_numeric(rows.loc[keep, I.TARGET_COL], errors="coerce").to_numpy(dtype=float)},
                        index=rows.index[keep], columns=cols)
    pairs = EP.comparable_pairs(slim)
    if not len(pairs):
        return pairs.assign(delta_z=pd.Series(dtype=int))
    dz = np.array([_atomic_number(a) - _atomic_number(b) for a, b in zip(pairs["state_a"], pairs["state_b"])],
                  dtype=int)
    if (dz < 1).any():
        raise AssertionError("comparable Ln(III) pairs must be oriented heavier-first")
    return pairs.assign(delta_z=dz)


def delta_z_magnitudes(rows: pd.DataFrame, *, condition_key: pd.Series | None = None,
                       pub_group_col: str = I.PUB_GROUP_COL) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    """``(per delta-Z table, pooled mean |logSF|, the training pairs)`` -- the registered B8 magnitude.

    The per-class table has ``delta_z``, ``n_pairs`` and ``mean_abs_logsf``; ``pooled`` is the mean over every
    training Ln(III)-Ln(III) pair (the fallback of :data:`REGISTRATION_CHOICES` ``magnitude_fallback``).
    """
    pairs = training_ln3_pairs(rows, condition_key=condition_key, pub_group_col=pub_group_col)
    if not len(pairs):
        return (pd.DataFrame({"delta_z": pd.Series(dtype=int), "n_pairs": pd.Series(dtype=int),
                              "mean_abs_logsf": pd.Series(dtype=float)}), float("nan"), pairs)
    a = np.abs(pairs["logsf_obs"].to_numpy(dtype=float))
    if not np.isfinite(a).all():
        raise ValueError("delta_z_magnitudes: a non-finite observed logSF among the training pairs")
    tab = (pd.DataFrame({"delta_z": pairs["delta_z"].to_numpy(dtype=int), "abs_logsf": a})
           .groupby("delta_z", sort=True)["abs_logsf"].agg(n_pairs="size", mean_abs_logsf="mean").reset_index())
    return tab, float(a.mean()), pairs


# --------------------------------------------------------------------------------------------- #
# B8 direction arm (the gen14 G14 rule)
# --------------------------------------------------------------------------------------------- #

class B8Direction:
    """The gen14 G14 direction rule, re-fitted on one outer training fold (section 5 B8).

    ``fit`` builds, from the training rows only: the per-cell radius coefficients, the per-system coefficient
    and label, the L2 logistic on the standardised, median-imputed TOPO39 columns of the fit-unit systems, and
    the mean ``|logSF|`` per delta-Z class.  ``pair_records`` / :meth:`predict_pairs` then give a predicted
    logSF (and therefore a direction) for every Ln(III)-Ln(III) pair whose system has TOPO39, and NaN --
    undefined, counted -- for every other pair.  :meth:`coverage` is the registered coverage report.
    """

    name = DIRECTION_ARM_NAME

    def __init__(self, *, min_states: int = MIN_LN3_STATES, C: float = LOGIT_C, max_iter: int = LOGIT_MAX_ITER,
                 metals: Sequence[str] = BASIS_METALS):
        self.min_states = int(min_states)
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.metals = tuple(metals)
        self.systems: pd.DataFrame | None = None
        self.units: tuple[str, ...] = ()
        self.prep: F.Topo39Prep | None = None
        self.logistic: Any = None
        self.constant_p: float | None = None
        self.magnitude_table: pd.DataFrame | None = None
        self.magnitude_pooled: float = float("nan")
        self.n_training_pairs = 0
        self.n_train_rows = 0
        self.status = "unfitted"
        self._p_cache: dict[str, float] = {}

    # ----------------------------------------------------------------------------------------- #
    def clone(self) -> "B8Direction":
        return B8Direction(min_states=self.min_states, C=self.C, max_iter=self.max_iter, metals=self.metals)

    @property
    def fitted(self) -> bool:
        return self.systems is not None

    def _require_fit(self) -> None:
        if not self.fitted:
            raise RuntimeError(f"{self.name}: fit first")

    @staticmethod
    def _assert_feature_columns() -> None:
        """Explicit section 2 guard on the direction's only features (the 39 TOPO39 columns)."""
        F.assert_feature_columns_allowed(F.topo39_columns(), "B8 direction (TOPO39) feature columns")

    # ----------------------------------------------------------------------------------------- #
    def fit(self, train_rows: pd.DataFrame, context: I.FitContext | None = None, *,
            condition_key: pd.Series | None = None, pub_group_col: str = I.PUB_GROUP_COL) -> "B8Direction":
        from sklearn.linear_model import LogisticRegression

        if not len(train_rows):
            raise ValueError(f"{self.name}: no training rows")
        if train_rows.index.has_duplicates:
            raise ValueError(f"{self.name}: the training index must be unique")
        hid = None if context is None else context.hidden_index
        if hid is not None and len(hid):
            bad = train_rows.index.intersection(pd.Index(hid))
            if len(bad):
                raise AssertionError(f"{self.name}: {len(bad)} hidden row(s) among the training rows")
        self._assert_feature_columns()
        ck = N.condition_key(train_rows) if condition_key is None else condition_key.reindex(train_rows.index)
        sysd = system_radius_coefficients(train_rows, condition_key=ck, pub_group_col=pub_group_col,
                                          min_states=self.min_states, metals=self.metals)
        avail = _topo39_available(list(sysd[SG.SYSTEM_COL]))
        sysd["topo39_available"] = [bool(avail[s]) for s in sysd[SG.SYSTEM_COL]]
        is_unit = (sysd["status"] == "fit_unit") & sysd["topo39_available"]
        sysd["fit_unit"] = is_unit
        self.systems = sysd
        self.units = tuple(sysd.loc[is_unit, SG.SYSTEM_COL].astype(str))
        self.n_train_rows = int(len(train_rows))
        self.logistic, self.constant_p, self.prep, self._p_cache = None, None, None, {}
        if not self.units:
            self.status = "no_training_units"
        else:
            self.prep = F.Topo39Prep().fit(self.units)
            X, ok = self.prep.transform(list(self.units))
            if not ok.all() or not np.isfinite(X).all():
                raise AssertionError(f"{self.name}: a fit unit has no TOPO39 row")
            y = sysd.loc[is_unit, "heavy_selective"].to_numpy(dtype=int)
            if len(set(y.tolist())) < 2:                       # gen14/models.py:140-141
                self.constant_p = float(y.mean())
                self.status = "constant_one_class"
            else:
                self.logistic = LogisticRegression(C=self.C, max_iter=self.max_iter, solver="lbfgs").fit(X, y)
                self.status = "fitted"
        tab, pooled, pairs = delta_z_magnitudes(train_rows, condition_key=ck, pub_group_col=pub_group_col)
        self.magnitude_table, self.magnitude_pooled, self.n_training_pairs = tab, pooled, int(len(pairs))
        return self

    # ----------------------------------------------------------------------------------------- #
    def p_heavy(self, systems: Sequence[str]) -> np.ndarray:
        """P(heavy-selective) per system (NaN where the direction is undefined); never refits."""
        self._require_fit()
        keys = [None if _missing(s) else str(s) for s in systems]
        out = np.full(len(keys), np.nan)
        todo = sorted({k for k in keys if k is not None and k not in self._p_cache})
        if todo and (self.logistic is not None or self.constant_p is not None):
            avail = F.topo39_for_systems(todo)["topo39_available"]
            have = [k for k in todo if bool(avail[k])]
            if have:
                if self.constant_p is not None:
                    vals = np.full(len(have), float(self.constant_p))
                else:
                    X, ok = self.prep.transform(have)           # type: ignore[union-attr]
                    if not ok.all():
                        raise AssertionError("p_heavy: TOPO39 availability changed between calls")
                    vals = np.asarray(self.logistic.predict_proba(X)[:, 1], dtype=float)
                self._p_cache.update(dict(zip(have, (float(v) for v in vals))))
            for k in todo:
                self._p_cache.setdefault(k, float("nan"))
        for i, k in enumerate(keys):
            if k is not None:
                out[i] = self._p_cache.get(k, float("nan"))
        return out

    def system_directions(self, systems: Sequence[str]) -> pd.DataFrame:
        """Per system: ``p_heavy``, ``heavy_selective`` (``p >= 0.5``, gen14's hard rule), the gen14 coefficient
        sign ``radius_coef_sign`` (-1 heavy-selective, +1 light-selective), ``topo39_available``, ``in_training``
        and ``status``."""
        self._require_fit()
        keys = [str(s) for s in systems]
        p = self.p_heavy(keys)
        avail = _topo39_available(keys)
        known = {} if self.systems is None else dict(zip(self.systems[SG.SYSTEM_COL].astype(str),
                                                         self.systems["status"]))
        heavy = p >= P_HEAVY_THRESHOLD
        return pd.DataFrame({
            SG.SYSTEM_COL: keys, "p_heavy": p,
            "heavy_selective": np.where(np.isfinite(p), heavy, np.nan),
            "radius_coef_sign": np.where(np.isfinite(p), np.where(heavy, -1.0, 1.0), np.nan),
            "topo39_available": [bool(avail[k]) for k in keys],
            "in_training": [k in known for k in keys],
            "training_status": [known.get(k, "not_in_training") for k in keys],
            "direction_model": self.status,
        })

    def magnitude(self, delta_z: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        """``(magnitude per delta-Z class, pooled-fallback flag)``; NaN when no training pair exists at all."""
        self._require_fit()
        tab = self.magnitude_table
        lookup = {} if tab is None or not len(tab) else dict(zip(tab["delta_z"].astype(int),
                                                                 tab["mean_abs_logsf"].astype(float)))
        out = np.full(len(delta_z), np.nan)
        fb = np.zeros(len(delta_z), dtype=bool)
        for i, d in enumerate(delta_z):
            v = lookup.get(int(d))
            if v is None:
                out[i] = self.magnitude_pooled
                fb[i] = np.isfinite(self.magnitude_pooled)
            else:
                out[i] = float(v)
        return out, fb

    # ----------------------------------------------------------------------------------------- #
    def pair_records(self, pairs: pd.DataFrame) -> pd.DataFrame:
        """Per pair of a ``pairs.comparable_pairs`` frame: ``logsf_pred`` (NaN = undefined), ``direction``,
        ``p_heavy``, ``heavy_selective``, ``delta_z`` (signed ``Z_a - Z_b``; the magnitude class is its absolute
        value, so either orientation works), ``magnitude``, ``magnitude_pooled``, ``defined`` and ``reason``
        (:data:`PAIR_REASONS`).  Only the system and the two metal states are read -- never a target.
        """
        self._require_fit()
        need = [EM.SYSTEM_COL, "state_a", "state_b"]
        miss = [c for c in need if c not in pairs.columns]
        if miss:
            raise KeyError(f"pair_records: columns missing {miss}")
        n = len(pairs)
        sys_keys = [None if _missing(s) else str(s) for s in pairs[EM.SYSTEM_COL].to_numpy(dtype=object)]
        sa = pairs["state_a"].to_numpy(dtype=object)
        sb = pairs["state_b"].to_numpy(dtype=object)
        ln3 = np.array([_is_ln3(a) and _is_ln3(b) for a, b in zip(sa, sb)], dtype=bool)
        dz = np.zeros(n, dtype=int)
        for k in range(n):
            if ln3[k]:
                dz[k] = _atomic_number(str(sa[k])) - _atomic_number(str(sb[k]))
        p = self.p_heavy(sys_keys)
        mag, pooled = self.magnitude(np.abs(dz))                         # the class is |delta Z|
        avail = _topo39_available([k for k in sys_keys if k is not None])
        logsf = np.full(n, np.nan)
        reason = np.array(["ok"] * n, dtype=object)
        for k in range(n):
            if sys_keys[k] is None:
                reason[k] = "unknown_system"
                continue
            if not ln3[k]:
                reason[k] = "not_ln3_ln3"
                continue
            if not bool(avail[sys_keys[k]]):
                reason[k] = "no_topo39"
                continue
            if not np.isfinite(p[k]):
                reason[k] = "no_direction_model"
                continue
            if not np.isfinite(mag[k]):
                reason[k] = "no_magnitude"
                continue
            s_dir = 1.0 if p[k] >= P_HEAVY_THRESHOLD else -1.0      # heavy-selective: the heavier is extracted
            logsf[k] = float(np.sign(dz[k]) * s_dir * mag[k])
        heavy = np.where(np.isfinite(p), p >= P_HEAVY_THRESHOLD, np.nan)
        return pd.DataFrame({
            "logsf_pred": logsf, "direction": np.sign(logsf), "p_heavy": p, "heavy_selective": heavy,
            "delta_z": np.where(ln3, dz, 0), "magnitude": np.where(ln3, mag, np.nan),
            "magnitude_pooled": np.where(ln3, pooled, False), "defined": np.isfinite(logsf), "reason": reason,
        }, index=pairs.index)

    def predict_pairs(self, pairs: pd.DataFrame) -> pd.Series:
        """The predicted logSF of every pair, NaN where B8 has no direction (``pairs.score_pairs`` /
        ``pair_summary`` with ``direction_only=True`` treat NaN as undefined and count it)."""
        return self.pair_records(pairs)["logsf_pred"].rename("logsf_pred")

    # ----------------------------------------------------------------------------------------- #
    def coverage(self, pairs: pd.DataFrame | None = None) -> dict[str, Any]:
        """The registered coverage report ("Systems without it get no B8 direction; this coverage is reported").

        Training side: systems seen, systems reaching the state threshold, fit units, and why the others are
        not units.  With ``pairs``: how many pairs get a direction and the reason count of those that do not.
        """
        self._require_fit()
        s = self.systems
        assert s is not None
        st = s["status"].to_numpy(dtype=object)
        out: dict[str, Any] = {
            "min_ln3_states": self.min_states,
            "direction_model": self.status,
            "n_train_rows": self.n_train_rows,
            "n_systems_training": int(len(s)),
            "n_systems_ge_min_states": int((st != "below_min_states").sum()),
            "n_systems_with_coefficient": int((st == "fit_unit").sum()),
            "n_systems_no_rich_cell": int((st == "no_rich_cell").sum()),
            "n_systems_below_min_states": int((st == "below_min_states").sum()),
            "n_fit_units": len(self.units),
            "n_systems_with_coefficient_without_topo39": int(((st == "fit_unit") & ~s["topo39_available"]).sum()),
            "n_systems_topo39": int(s["topo39_available"].sum()),
            "n_heavy_units": int(s.loc[s["fit_unit"], "heavy_selective"].sum()),
            "n_light_units": int((~s.loc[s["fit_unit"], "heavy_selective"]).sum()),
            "n_training_pairs": self.n_training_pairs,
            "n_delta_z_classes": 0 if self.magnitude_table is None else int(len(self.magnitude_table)),
            "magnitude_pooled": self.magnitude_pooled,
        }
        if pairs is not None:
            rec = self.pair_records(pairs)
            counts = rec["reason"].value_counts()
            out.update({
                "n_pairs": int(len(rec)),
                "n_pairs_with_direction": int(rec["defined"].sum()),
                "n_pairs_magnitude_pooled": int(np.asarray(rec["magnitude_pooled"], dtype=bool).sum()),
                "pair_reasons": {r: int(counts.get(r, 0)) for r in PAIR_REASONS},
                "n_systems_in_pairs": int(pd.unique(pairs[EM.SYSTEM_COL].astype(str)).size),
                "n_systems_in_pairs_with_direction": int(
                    pd.unique(pairs.loc[rec["defined"].to_numpy(dtype=bool), EM.SYSTEM_COL].astype(str)).size),
            })
        return out

    def state(self) -> dict[str, Any]:
        s = self.systems
        coef = None if self.logistic is None else np.asarray(self.logistic.coef_, dtype=float)
        return {"schema": SCHEMA, "arm": self.name, "status": self.status, "min_states": self.min_states,
                "C": self.C, "max_iter": self.max_iter, "metals": list(self.metals),
                "units": list(self.units), "constant_p": self.constant_p,
                "prep": None if self.prep is None else self.prep.state(),
                "coef": coef, "intercept": None if self.logistic is None
                else float(np.asarray(self.logistic.intercept_, dtype=float).ravel()[0]),
                "amp": None if s is None else [float(v) for v in s.loc[s["fit_unit"], "amp"]],
                "magnitude": None if self.magnitude_table is None
                else {int(d): float(v) for d, v in zip(self.magnitude_table["delta_z"],
                                                       self.magnitude_table["mean_abs_logsf"])},
                "magnitude_pooled": self.magnitude_pooled, "n_training_pairs": self.n_training_pairs,
                "n_train_rows": self.n_train_rows}

    @property
    def state_digest(self) -> str:
        return F.state_digest(self.state())


# --------------------------------------------------------------------------------------------- #
# both halves together
# --------------------------------------------------------------------------------------------- #

@dataclass
class B8Bundle:
    """B8's two registered halves fitted on one training set: the log D level and the pair direction."""

    level: B8Level
    direction: B8Direction

    def coverage(self, pairs: pd.DataFrame | None = None) -> dict[str, Any]:
        return self.direction.coverage(pairs)

    def state(self) -> dict[str, Any]:
        return {"level": self.level.state(), "direction": self.direction.state()}

    @property
    def state_digest(self) -> str:
        return F.state_digest(self.state())


def conformal_b8_level(frame: pd.DataFrame, splitter: Any, *, fold: int | None = None, random_state: int | None = None,
                       cv: pd.DataFrame | None = None, config: MonoEtConfig | None = None, guard: str = "every_split",
                       ecfp_format: str = "dense") -> I.ConformalWrapper:
    """The B8 level with split-conformal intervals calibrated over the inner splits of ``splitter`` (addendum 1
    items 1-2, :data:`REGISTRATION_CHOICES` ``conformal``): ``ConformalWrapper(B8Level(...))`` -- one level fit per
    inner split (its training rows only), the residuals of the splits' calibration rows pooled, the arm refitted on all
    outer-training rows.  ``splitter`` must be a learned-arm inner design (``inner_design.SimultaneousInnerCells`` for
    V5; never the per-cell ``InnerCellCalibration``).  Fit with ``fit(train_rows, context)`` or
    ``fit_table(table, mask, context)``; ``context.seed``, ``context.v6_mask`` and ``context.isolation_check`` are
    required."""
    ID.assert_learned_arm_design(splitter, ARM_NAME)
    level = B8Level(frame, cv=cv, fold=fold, random_state=random_state, config=config, ecfp_format=ecfp_format)
    return I.ConformalWrapper(level, splitter=splitter, guard=guard)


def fit_b8(train_rows: pd.DataFrame, context: I.FitContext | None = None, *, fold: int | None = None,
           random_state: int | None = None, frame: pd.DataFrame | None = None, cv: pd.DataFrame | None = None,
           condition_key: pd.Series | None = None, config: MonoEtConfig | None = None,
           pub_group_col: str = I.PUB_GROUP_COL, ecfp_format: str = "dense") -> B8Bundle:
    """Fit both halves of B8 on ``train_rows`` (the level's ``frame`` defaults to ``train_rows``)."""
    level = B8Level(train_rows if frame is None else frame, cv=cv, fold=fold, random_state=random_state,
                    config=config, ecfp_format=ecfp_format).fit(train_rows, context)
    direction = B8Direction().fit(train_rows, context, condition_key=condition_key, pub_group_col=pub_group_col)
    return B8Bundle(level=level, direction=direction)
