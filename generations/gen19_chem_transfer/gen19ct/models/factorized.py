"""``models/factorized.py`` -- B6 and B6r0: the metal x extractant factorisation with side information
(``preregistration.md`` section 5 B6, sealed 2026-09-15; sections 2, 3.1, 7, 12).

Model (section 5 B6), fitted on the rows passed to ``fit`` only::

    y = mu + a_m + b_s + u_m' v_s + beta' x_cond + eps
    u_m = A z_m + delta_m        v_s = B z_s + delta_s          (k-dimensional)
    a_m = alpha_a' z_m + a'_m    b_s = alpha_b' z_s + b'_s

* ``z_m`` -- the standardised numeric B5 metal block (``features.METAL_NUMERIC``; training medians imputed);
  ``z_s`` -- the standardised numeric B5 extractant block (``n_components``, ``donor_*``, ``n_donor_sites``,
  ``denticity_proxy``, the share-weighted ``w_mw, w_mol_logp, w_tpsa, w_rotatable_bonds`` and the 16-component
  fingerprint PCA fitted on the training systems); ``x_cond`` -- the B5 condition block (11 numeric columns and
  one-hot ``acid_anion`` / ``diluent_family``, medians imputed, standardised).  Every statistic (medians, one-hot
  vocabularies, means / SDs, the PCA) is the ``features.FeatureSet.for_arm("B6")`` rule re-fitted on each fit's own
  training rows (tested equal to that FeatureSet); :data:`REGISTRATION_CHOICES` lists the readings.
* Parameters in code: ``gamma = (mu, beta)``; the metal side ``P_m`` (``p_m x (k+1)``: column 0 = ``alpha_a``,
  columns 1..k = ``A'``) and ``F_m`` (``J_m x (k+1)``: column 0 = ``a'_m``, columns 1..k = ``delta_m``), so that
  ``(a_m, u_m) = P_m' z_m + F_m[m]``; the system side ``P_s`` / ``F_s`` likewise.
* Objective: ``sum_rows (y - y_hat)^2 + lambda (|beta|^2 + |P_m|^2 + |F_m|^2 + |P_s|^2 + |F_s|^2)`` -- one lambda on
  every coefficient except ``mu``, rows unweighted.
* ``k = 0`` (B6r0, and B6 when it selects rank 0) is solved jointly in closed form.  ``k >= 1``: alternating ridge
  least squares over rows -- a system step (``gamma``, ``P_s``, ``F_s`` given the metal side) and a metal step
  (``gamma``, ``P_m``, ``F_m`` given the system side), each an exact ridge solve; at most :data:`MAX_SWEEPS` sweeps,
  stop when the relative change of the objective over one sweep is <= :data:`REL_TOL`.
* Seeded initialisation (cold start for EVERY fit -- outer refit, inner tuning fit and calibration fit alike): the
  linear parts (``gamma``, ``alpha``, ``a'``, ``b'``) start at the ``k = 0`` ridge solution with the same lambda on
  the same training rows, the metal free factors ``delta_m`` at ``INIT_SCALE * sqrt(RMSE_0) * N(0, 1)`` (``RMSE_0``
  the training root-mean-square residual of that ``k = 0`` fit, so ``u'v`` starts on the residual scale) drawn per
  metal-unit LABEL from ``SeedSequence([seed, INIT_STREAM_TAG, k, crc32(label)])`` (so hiding a cell does not reshuffle
  the other units' draws), and ``A``, ``B``, ``delta_s`` at 0; the first step is the system step.  No fit is ever started from
  another fit's solution, so no warm start can carry information from a fit that saw the current test rows.
* A metal unit or system absent from the training rows gets its free parts (``a'``, ``delta``, ``b'``) set to 0: its
  prediction uses the side information only (``fallback_reason`` ``metal_unseen`` / ``system_unseen``).

Grouped normal equations (speed; :class:`Design`).  Rows are summed per cell = (system pattern, metal unit), a system
pattern being a system with one value of the row-level extractant block (259 systems give 265 patterns and 1,622
cells on the MODEL rows); ``z_m`` is constant per metal unit and ``z_s`` per pattern, so every normal-equation block is
a sum over cells plus the row moments ``sum c c'``, ``sum c y`` (``c = (1, x_cond)``).  The free per-unit parameters
form a block-diagonal part eliminated by a Schur complement, so a half-step solves a dense system of size
``1 + p_c + p (k+1)`` only, with the ``P`` block assembled as a Kronecker sum (tested equal to the explicit row-level
ridge solve).  A :class:`Design` can hold ``S`` training sets over one universe (zero cells for rows a member lacks) and
``L`` lambdas, solved together; each member's solution is that of its own rows (tested equal to the single fit).
BLAS is limited to :data:`BLAS_THREADS` inside a fit.

Registered tuning (section 7; POST-HOC addendum 1 items 1-2) -- :func:`run_inner_tuning`, :class:`B6TunedConformal`,
:func:`fit_b6_and_b6r0`
    The inner splits are those of the learned arms' inner design (``models.inner_design.SimultaneousInnerCells`` for
    V5 / V5-P / V5-PAIR / V6 -- the section 7 inner cells, all cells of an inner fold hidden in one split, addendum 1
    item 1; ``GroupKFoldCalibration`` for V1, ``InnerMetalCalibration`` for V2 -- the one implementation per design the
    fold builder uses); the per-cell ``InnerCellCalibration`` is refused (``inner_design.assert_learned_arm_design``).
    Every split passes the isolation guard (``ConformalWrapper._verify``) and ``registered.assert_not_scored``.  All
    configurations ``k in {0,1,2,3} x lambda in {0.1, 1, 10}`` are fitted on every split (one fit per configuration and
    inner fold); a configuration's score on an inner fold is the unit-macro MAE (the absolute errors averaged per
    averaging unit of the design -- ``InnerSplit.row_units``: an inner cell, a publication group or ``REMAINDER``, an
    inner metal state -- then over units) and its selection score the mean over the three inner folds (addendum 1
    item 2); configurations within 0.005 of the best are resolved toward lower rank, then stronger penalty.  B6r0 is the
    same selection restricted to ``k = 0`` (on the same inner fits).  The split-conformal intervals are cross-fitted
    (:class:`B6TunedConformal` ``calibration="cross_fit"``): inner fold ``j``'s residuals are those of the configuration
    selected on the other inner folds, taken from the fits already made (no extra fit); the arm is then refitted on all
    training rows.  No "first inner fold" reading exists (``inner_mode="first"`` is refused).  ``std_logD`` is NaN (no
    variance model; the section 12 Bayesian-ridge head is not built here).

Also: :func:`select_config` (the tie rule), :func:`batched_exact_check` / :func:`batched_check_decision` (the section
3.1 registered check and the section 7 compute-plan item 6 re-colouring rule), :func:`register_table` (B6 inputs of a
``RowTable`` for the ``fit_table`` / ``predict_positions`` fast path).

Nothing here reads ``log_D`` except the training targets of a fit and the inner calibration targets of the OUTER
training rows; no feature is derived from ``load.PROVENANCE_COLUMNS``, publication / study / row ids or the target
(:func:`features.assert_feature_columns_allowed` on every source and output column).
"""
from __future__ import annotations

import hashlib
import time
import weakref
import zlib
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import linalg as sla
from scipy import sparse

try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
except ImportError:                                                     # pragma: no cover
    _threadpool_limits = None

from gen19ct.chemistry import support_graph as SG
from gen19ct.data import normalize as N
from gen19ct.folds import io as FI
from gen19ct.folds import registered as FR
from gen19ct.models import features as F
from gen19ct.models import inner_design as ID
from gen19ct.models import interface as I

#: registered grids (section 5 B6)
RANKS: tuple[int, ...] = (0, 1, 2, 3)
LAMBDAS: tuple[float, ...] = (0.1, 1.0, 10.0)
VARIANT_RANKS: dict[str, tuple[int, ...]] = {"B6": RANKS, "B6r0": (0,)}
MAX_SWEEPS = 50
REL_TOL = 1e-6
#: section 7 selection criterion: within this inner macro MAE of the best -> the smaller configuration
TIE_MARGIN = 0.005
#: seeded metal free factors start at INIT_SCALE * sqrt(training RMSE of the k=0 fit) * N(0, 1).  An implementation
#: constant chosen on SYNTHETIC data only, before any real B6 fit: of fixed SDs {0.01, 0.03, 0.1, 0.3, 0.5, 1} and
#: residual-scaled {0.1, 0.3, 1}, residual-scaled 0.3 had the lowest mean 50-sweep objective relative to long runs
#: (5 synthetic designs x k in {1,2,3} x lambda in {0.1, 10}); not a registered number
INIT_SCALE = 0.3
#: random stream tag of the seeded initialisation (distinct from every ``folds.io.STREAM_TAGS`` value)
INIT_STREAM_TAG = 60
#: BLAS threads inside a B6 fit (small dense matrices; more threads only oversubscribe the <= 2 worker processes)
BLAS_THREADS = 1
#: inner splits solved together as members of one Design in run_inner_tuning (per-member cost is flop-bound; a chunk
#: only bounds memory, ~10 MB per chunk of 4 on the MODEL rows)
TUNING_CHUNK = 4
#: section 3.1 / 3.2 registered checks and section 7 compute-plan item 6
BATCHED_CHECK_THRESHOLD = 0.01
RECOLOUR_MAX_CELLS = 4

UNKNOWN_STATE_SUFFIX = "(?)"
METAL_COL, ELEMENT_COL, SYSTEM_COL = SG.METAL_COL, SG.ELEMENT_COL, SG.SYSTEM_COL
SYSTEM_STATIC_COLUMNS: tuple[str, ...] = ("system_family", "mechanism", "primary_extractant_smiles")
#: the only row columns B6 reads to build its features (plus ``log_D`` as the training target of a fit)
SOURCE_COLUMNS: tuple[str, ...] = tuple(dict.fromkeys((METAL_COL, ELEMENT_COL, SYSTEM_COL) + SYSTEM_STATIC_COLUMNS
                                                      + tuple(N.REQUIRED_COLUMNS)))
EXTRA_PREDICTION_COLUMNS: tuple[str, ...] = ("b6_rank", "b6_lambda", "b6_metal_unit", "b6_metal_seen",
                                             "b6_system_seen", "b6_main_metal", "b6_main_system", "b6_interaction",
                                             "b6_n_sweeps", "b6_converged", "b6_seed")

REGISTRATION_CHOICES: dict[str, str] = {
    "side_information": "z_m / z_s are the NUMERIC columns of the B5 metal and extractant blocks (8 and 42 columns); "
                        "the categorical columns (category, series, hsab_class; family, mechanism) are not in z "
                        "(side_categoricals=True adds their standardised one-hot columns, the FeatureSet B6 preset)",
    "x_cond": "the whole B5 condition block: 11 numeric columns plus one-hot acid_anion and diluent_family, "
              "training-median imputed, standardised (FeatureSet B6 preset, no missing indicators)",
    "statistics_unit": "medians / means / SDs row-weighted over the fit's training rows, the fingerprint PCA over its "
                       "training systems (features.REGISTRATION_CHOICES statistics_unit); re-fitted for every fit, "
                       "every inner tuning / calibration split included (brief section 12: the rows passed to fit)",
    "unseen_tokens": "a condition token absent from a fit's training rows is an all-zero one-hot (then standardised); "
                     "a metal unit / system absent from the training rows gets free parts 0",
    "row_level_z_s": "z_s is evaluated per row (the share-weighted descriptors vary with the concentrations of a "
                     "mixture), so v_s = B z_s(row) + delta_s; groups are (system, metal unit, z_s pattern)",
    "unknown_state_rows": "an X(?) row belongs to its own metal unit '<element>(?)' (element-level descriptors, "
                          "ion-level columns training-median imputed); no state is imputed and such rows are never "
                          "scored",
    "penalty": "one lambda on beta, alpha_a, alpha_b, A, B, a', b', delta_m, delta_s; mu unpenalised; unweighted sum of "
               "squared row residuals (lambda is not scaled by the row count)",
    "convergence": "relative change of the penalised objective over one full sweep <= 1e-6, at most 50 sweeps; a fit "
                   "that reaches 50 sweeps keeps its last iterate and is flagged b6_converged=False (on synthetic data "
                   "and on the MODEL-row inputs with a synthetic target the cap binds for nearly every k >= 1 fit)",
    "variance": "std_logD is NaN: B6 has no variance model here (the section 12 bootstrap ensemble and conjugate "
                "Bayesian-ridge head of B6 are separate uncertainty methods, not built by this module)",
    "initialisation": "cold start per fit: k=0 ridge solution (same lambda, same rows) for the linear parts, "
                      "INIT_SCALE * sqrt(k=0 training RMSE) * N(0,1) metal free factors seeded per unit label by "
                      "SeedSequence([seed, 60, k, crc32(label)]), A = B = delta_s = 0, system step first; seed = the "
                      "init_seed the caller passes (discovery: the section 15 model seed 42 + fold * 1009 + 9,999,991 "
                      "of the outer fold, the same for every inner tuning, calibration and outer fit of that fold, as "
                      "for every other learned arm), else context.seed; INIT_SCALE = 0.3 chosen on synthetic data only",
    "inner_macro_mae": "section 7 'inner macro MAE with the design's own averaging' under addendum 1 item 2: per inner "
                       "fold, the absolute errors of the fold's calibration (scored) rows are averaged per averaging "
                       "unit (InnerSplit.row_units: V5 hidden cell, V1 publication group or REMAINDER below 20 "
                       "outer-training rows, V2 metal state -- the units of the boosted and neural inner designs; a "
                       "unit pooled over the fold's splits), then over units; the selection score is the mean over the "
                       "inner folds (B6InnerTuning.macro_mae = mean of fold_macro_mae); a split without row_units "
                       "counts as one unit (discovery refuses such splits)",
    "inner_design": "addendum 1 item 1: the V5 inner design is inner_design.SimultaneousInnerCells (one split per inner "
                    "fold, all of the fold's cells hidden, failing cells hidden but unscored); run_inner_tuning and "
                    "fit_b6_variants refuse the per-cell InnerCellCalibration",
    "tie_rule": "configurations with inner macro MAE <= best + 0.005 -> the smallest rank, then the largest lambda",
    "b6r0_tuning": "B6r0 selects lambda by the same inner rule restricted to k = 0, on the same inner fits as B6",
    "conformal": "calibration='cross_fit' (discovery, addendum 1 item 2): no residual comes from a row that selected "
                 "the configuration it is a residual of -- inner fold j's residuals are those of the configuration "
                 "selected on the other inner folds (every configuration is fitted on every split, so the fits already "
                 "made are reused, no refit); fewer than two inner folds: not calibrated (NaN intervals). "
                 "calibration='tuning_residuals' is the earlier reading (residuals of the selected configuration on the "
                 "tuning splits). The interval centre is the refit on all training rows",
    "fold_subset": "run_inner_tuning(fold_subset=...) is a diagnostic restriction of the inner folds fitted; the "
                   "section 7 compute plan's 'first inner fold' reading (fit_b6_variants(inner_mode='first'), "
                   "calibration on the next inner fold) was retired by addendum 1 item 2 and is refused",
    "batched_check": "section 7 item 6 (re-colour with <= 4 cells, repeat once, else label 'batched (check failed)' "
                     "and S1 UNDECIDED) is implemented; section 3.1's older 'otherwise every arm uses exact "
                     "leave-one-cell-out' is reported beside it",
}


def _missing(v: Any) -> bool:
    return SG._missing(v)


def metal_unit_label(state: Any, element: Any) -> str:
    """The B6 metal unit of a row: the metal state, or ``'<element>(?)'`` for an unknown-state row."""
    if not _missing(state):
        return str(state)
    return f"{F._token(element)}{UNKNOWN_STATE_SUFFIX}"


def _finite(a: np.ndarray) -> np.ndarray:
    out = np.array(a, dtype=float, copy=True)
    out[~np.isfinite(out)] = np.nan
    return out


def _nan_equal_rows(a: np.ndarray) -> bool:
    if len(a) <= 1:
        return True
    return bool(np.all((a == a[0]) | (np.isnan(a) & np.isnan(a[0]))))


# --------------------------------------------------------------------------------------------- #
# raw inputs (target-free, row-wise; nothing fitted)
# --------------------------------------------------------------------------------------------- #

@dataclass(eq=False)
class B6Inputs:
    """Raw, unencoded B6 inputs of a set of rows (positions = row order of ``index``).

    ``metal_num`` / ``metal_cat``   B5 metal block (per row; constant per metal unit, asserted)
    ``ext_num`` / ``ext_cat``       B5 extractant block without the PCA columns (the PCA is fitted per fit)
    ``cond_num`` / ``cond_cat``     B5 condition block
    ``system_view``                 one row per system (key + static columns) for the PCA; ``None`` = no PCA columns
    ``pattern``                     int id of (system, raw ``ext_num`` row) -- groups of identical ``z_s``
    """

    index: pd.Index
    metal_unit: np.ndarray
    system: np.ndarray
    metal_num: np.ndarray
    metal_num_cols: tuple[str, ...]
    metal_cat: np.ndarray
    metal_cat_cols: tuple[str, ...]
    ext_num: np.ndarray
    ext_num_cols: tuple[str, ...]
    ext_cat: np.ndarray
    ext_cat_cols: tuple[str, ...]
    cond_num: np.ndarray
    cond_num_cols: tuple[str, ...]
    cond_cat: np.ndarray
    cond_cat_cols: tuple[str, ...]
    system_view: pd.DataFrame | None
    system_labels: tuple[str, ...] = ()
    system_code: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    primary_smiles: tuple[str | None, ...] = ()
    pattern: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    def __post_init__(self) -> None:
        n = len(self.index)
        if not self.index.is_unique:
            raise ValueError("B6Inputs: the row index must be unique")
        for nm in ("metal_unit", "system", "metal_num", "metal_cat", "ext_num", "ext_cat", "cond_num", "cond_cat"):
            if len(getattr(self, nm)) != n:
                raise ValueError(f"B6Inputs: {nm} has {len(getattr(self, nm))} rows, expected {n}")
        if any(v is None or _missing(v) for v in self.system):
            raise ValueError("B6Inputs: every row needs an extractant_system_key")
        names = ([f"metal__{c}" for c in self.metal_num_cols + self.metal_cat_cols]
                 + [f"extractant__{c}" for c in self.ext_num_cols + self.ext_cat_cols]
                 + [f"condition__{c}" for c in self.cond_num_cols + self.cond_cat_cols])
        F.assert_feature_columns_allowed(names, "B6 input columns")
        self.metal_num = _finite(self.metal_num).reshape(n, len(self.metal_num_cols))
        self.ext_num = _finite(self.ext_num).reshape(n, len(self.ext_num_cols))
        self.cond_num = _finite(self.cond_num).reshape(n, len(self.cond_num_cols))
        for nm, cols in (("metal_cat", self.metal_cat_cols), ("ext_cat", self.ext_cat_cols),
                         ("cond_cat", self.cond_cat_cols)):
            arr = np.array([[F._token(v) for v in row] for row in np.asarray(getattr(self, nm), dtype=object)
                            .reshape(n, len(cols))], dtype=object).reshape(n, len(cols))
            setattr(self, nm, arr)
        self.metal_unit = np.array([str(v) for v in self.metal_unit], dtype=object)
        self.system = np.array([str(v) for v in self.system], dtype=object)
        labels, code = np.unique(self.system.astype(str), return_inverse=True)
        self.system_labels = tuple(str(s) for s in labels)
        self.system_code = code.reshape(-1).astype(np.int64)
        # constant metal descriptors per metal unit (the metal side of the factorisation is unit-level)
        m_labels, m_code = np.unique(self.metal_unit.astype(str), return_inverse=True)
        m_code = m_code.reshape(-1)
        order = np.argsort(m_code, kind="stable")
        bounds = np.r_[0, np.flatnonzero(np.diff(m_code[order])) + 1, n]
        for a, b in zip(bounds[:-1], bounds[1:]):
            rows = order[a:b]
            if not _nan_equal_rows(self.metal_num[rows]) or not (self.metal_cat[rows] == self.metal_cat[rows[0]]).all():
                raise ValueError(f"B6Inputs: metal descriptors differ inside metal unit {m_labels[m_code[rows[0]]]!r}")
        ext_key = np.where(np.isnan(self.ext_num), np.inf, self.ext_num)
        _, pat = np.unique(np.column_stack([self.system_code.astype(float), ext_key]), axis=0, return_inverse=True)
        self.pattern = pat.reshape(-1).astype(np.int64)
        if self.system_view is not None:
            sv = self.system_view
            if SYSTEM_COL not in sv.columns or list(sv[SYSTEM_COL]) != list(self.system_labels):
                raise ValueError("B6Inputs: system_view must hold one row per system in sorted key order")
            F.assert_feature_columns_allowed(sv.columns, "B6 system view columns")
            fb = F._row_fallbacks(sv)
            self.primary_smiles = tuple(F.system_record(k, fb.get(k)).primary_smiles for k in self.system_labels)

    @property
    def n(self) -> int:
        return len(self.index)

    @property
    def has_pca(self) -> bool:
        return self.system_view is not None

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "B6Inputs":
        """The B5 blocks of archive-layout rows (``interface.prepare_frame`` output or ``load_model_rows``), computed
        row-wise from :data:`SOURCE_COLUMNS` only (a view: provenance, id and target columns never reach a block)."""
        if not frame.index.is_unique:
            raise ValueError("B6Inputs.from_frame: the row index must be unique")
        need = [c for c in (METAL_COL, ELEMENT_COL, SYSTEM_COL) if c not in frame.columns]
        if need:
            raise KeyError(f"B6Inputs.from_frame: columns missing {need}")
        view = frame.loc[:, [c for c in SOURCE_COLUMNS if c in frame.columns]]
        F.assert_feature_columns_allowed(view.columns, "B6 source columns")
        assert not set(view.columns) & F.FORBIDDEN_SOURCE_COLUMNS
        cv = N.condition_vector(view)
        st = view[METAL_COL].to_numpy(dtype=object)
        el = view[ELEMENT_COL].to_numpy(dtype=object)
        recs = [F.metal_record(None if _missing(a) else str(a), None if _missing(b) else str(b)) for a, b in zip(st, el)]
        metal_num = np.array([[r[c] for c in F.METAL_NUMERIC] for r in recs], dtype=float).reshape(len(view), -1)
        metal_cat = np.array([[F._token(r[c]) for c in F.METAL_CATEGORICAL] for r in recs], dtype=object)
        units = np.array([metal_unit_label(a, b) for a, b in zip(st, el)], dtype=object)
        eb = F.ExtractantBlock()                                   # PCA placeholder: raw() is row-wise otherwise
        eb.pca_mean = np.zeros(F.FP_BITS)
        eb.pca_components = np.zeros((eb.n_components, F.FP_BITS))
        eb.pca_explained_variance = np.zeros(eb.n_components)
        eb.pca_n_fitted = 0
        ext = eb.raw(view, cv)
        ext_cols = tuple(c for c in ext.numeric.columns if c not in F.PCA_COLUMNS)
        cond = F.ConditionBlock().raw(view, cv)
        keys = np.array([F._token(k) for k in view[SYSTEM_COL].to_numpy(dtype=object)], dtype=object)
        sv_cols = [c for c in SYSTEM_STATIC_COLUMNS if c in view.columns]
        recs_sv = []
        for k in sorted(set(keys.tolist())):
            sub = view.loc[keys == k, sv_cols] if sv_cols else None
            rec = {SYSTEM_COL: k}
            for c in sv_cols:
                vals = [v for v in sub[c].to_numpy(dtype=object) if not _missing(v)]
                rec[c] = str(vals[0]) if vals else None
            recs_sv.append(rec)
        system_view = pd.DataFrame.from_records(recs_sv, columns=[SYSTEM_COL] + sv_cols)
        return cls(index=view.index, metal_unit=units, system=keys, metal_num=metal_num,
                   metal_num_cols=tuple(F.METAL_NUMERIC), metal_cat=metal_cat, metal_cat_cols=tuple(F.METAL_CATEGORICAL),
                   ext_num=ext.numeric[list(ext_cols)].to_numpy(dtype=float), ext_num_cols=ext_cols,
                   ext_cat=ext.categorical[["family", "mechanism"]].to_numpy(dtype=object),
                   ext_cat_cols=("family", "mechanism"),
                   cond_num=cond.numeric[list(F.CONDITION_NUMERIC)].to_numpy(dtype=float),
                   cond_num_cols=tuple(F.CONDITION_NUMERIC),
                   cond_cat=cond.categorical[list(F.CONDITION_CATEGORICAL)].to_numpy(dtype=object),
                   cond_cat_cols=tuple(F.CONDITION_CATEGORICAL), system_view=system_view)

    @classmethod
    def from_arrays(cls, index: Sequence[Any], metal_unit: Sequence[Any], system: Sequence[Any], *,
                    metal_numeric: pd.DataFrame, extractant_numeric: pd.DataFrame, condition_numeric: pd.DataFrame,
                    condition_categorical: pd.DataFrame | None = None, metal_categorical: pd.DataFrame | None = None,
                    extractant_categorical: pd.DataFrame | None = None) -> "B6Inputs":
        """Pre-computed blocks (synthetic data); no fingerprint PCA.  Column names are checked like feature names."""
        idx = pd.Index(list(index))
        n = len(idx)

        def parts(df: pd.DataFrame | None, numeric: bool) -> tuple[np.ndarray, tuple[str, ...]]:
            if df is None:
                return (np.zeros((n, 0)) if numeric else np.zeros((n, 0), dtype=object)), ()
            return df.to_numpy(dtype=float if numeric else object), tuple(str(c) for c in df.columns)
        mn, mnc = parts(metal_numeric, True)
        mc, mcc = parts(metal_categorical, False)
        en, enc = parts(extractant_numeric, True)
        ec, ecc = parts(extractant_categorical, False)
        cn, cnc = parts(condition_numeric, True)
        cc, ccc = parts(condition_categorical, False)
        return cls(index=idx, metal_unit=np.asarray(list(metal_unit), dtype=object),
                   system=np.asarray(list(system), dtype=object), metal_num=mn, metal_num_cols=mnc, metal_cat=mc,
                   metal_cat_cols=mcc, ext_num=en, ext_num_cols=enc, ext_cat=ec, ext_cat_cols=ecc, cond_num=cn,
                   cond_num_cols=cnc, cond_cat=cc, cond_cat_cols=ccc, system_view=None)

_INPUTS: "weakref.WeakKeyDictionary[I.RowTable, B6Inputs]" = weakref.WeakKeyDictionary()


def register_table(table: I.RowTable, source: pd.DataFrame | B6Inputs) -> B6Inputs:
    """Attach the B6 inputs of every row of ``table`` (built once per process from the arm frame, or given)."""
    if isinstance(source, B6Inputs):
        if not source.index.equals(table.index):
            raise ValueError("register_table: B6Inputs index differs from the RowTable index")
        inputs = source
    else:
        if not table.index.isin(source.index).all():
            raise KeyError("register_table: the frame does not cover the RowTable rows")
        inputs = B6Inputs.from_frame(source.loc[table.index])
    sys_tab = np.array([table.sys_labels[c] if c >= 0 else None for c in table.sys], dtype=object)
    if not np.array_equal(sys_tab, inputs.system):
        raise ValueError("register_table: the systems of the inputs and the RowTable differ")
    _INPUTS[table] = inputs
    return inputs


def inputs_for(table: I.RowTable, frame: pd.DataFrame | None = None) -> B6Inputs:
    inp = _INPUTS.get(table)
    if inp is not None:
        return inp
    if frame is not None and table.index.isin(frame.index).all():
        return register_table(table, frame)
    raise KeyError("B6: no inputs for this RowTable -- call factorized.register_table(table, frame) once per process")


# --------------------------------------------------------------------------------------------- #
# the fitted encoding (features._Prep rule for the B6 preset, re-fitted per fit)
# --------------------------------------------------------------------------------------------- #

class _BlockPrep:
    """``features._Prep`` with ``categorical='onehot'``, ``missing='impute'``, no indicators, ``standardize=True``."""

    def __init__(self, block: str, num_cols: Sequence[str], cat_cols: Sequence[str]):
        self.block, self.num_cols, self.cat_cols = block, tuple(num_cols), tuple(cat_cols)
        self.medians = np.zeros(0)
        self.vocab: list[list[str]] = []
        self.mean = np.zeros(0)
        self.sd = np.zeros(0)

    def _raw(self, num: np.ndarray, cat: np.ndarray) -> np.ndarray:
        v = np.where(np.isnan(num), self.medians[None, :], num) if num.shape[1] else num
        parts = [v]
        for j, voc in enumerate(self.vocab):
            toks = cat[:, j].astype(str)
            parts.append((toks[:, None] == np.array(voc, dtype=str)[None, :]).astype(float))
        return np.hstack(parts) if len(parts) > 1 else v

    def fit(self, num: np.ndarray, cat: np.ndarray) -> "_BlockPrep":
        med = []
        for c in num.T:
            fin = c[np.isfinite(c)]
            med.append(float(np.median(np.sort(fin))) if len(fin) else 0.0)
        self.medians = np.array(med, dtype=float)
        self.vocab = [sorted({str(t) for t in cat[:, j]}) for j in range(len(self.cat_cols))]
        full = self._raw(num, cat)
        mean, sd = [], []
        for j in range(full.shape[1]):
            col = np.ascontiguousarray(full[:, j])
            m, s = F._sorted_mean(col), F._sorted_std(col)
            mean.append(0.0 if not np.isfinite(m) else m)
            sd.append(1.0 if (not np.isfinite(s) or s <= 0) else s)
        self.mean, self.sd = np.array(mean, dtype=float), np.array(sd, dtype=float)
        return self

    def transform(self, num: np.ndarray, cat: np.ndarray) -> np.ndarray:
        return (self._raw(num, cat) - self.mean[None, :]) / self.sd[None, :]

    @property
    def names(self) -> tuple[str, ...]:
        out = [f"{self.block}__{c}" for c in self.num_cols]
        for c, voc in zip(self.cat_cols, self.vocab):
            out += [f"{self.block}__{c}={t}" for t in voc]
        return tuple(out)


_PCA_CACHE: "OrderedDict[tuple, F.ExtractantBlock]" = OrderedDict()
_PCA_CACHE_MAX = 8


def _pca_block(inputs: B6Inputs, positions: np.ndarray) -> F.ExtractantBlock:
    """``features.ExtractantBlock`` PCA fitted on the training systems of ``positions`` (each system once)."""
    codes = np.unique(inputs.system_code[positions])
    sv = inputs.system_view.iloc[codes].reset_index(drop=True)
    key = tuple(tuple(None if _missing(v) else str(v) for v in row) for row in sv.to_numpy(dtype=object))
    hit = _PCA_CACHE.get(key)
    if hit is None:
        hit = F.ExtractantBlock()
        hit.fit(sv, None)
        _PCA_CACHE[key] = hit
        while len(_PCA_CACHE) > _PCA_CACHE_MAX:
            _PCA_CACHE.popitem(last=False)
    else:
        _PCA_CACHE.move_to_end(key)
    return hit


@dataclass
class Encoded:
    Zm: np.ndarray       # rows x p_m
    Zs: np.ndarray       # rows x p_s
    X: np.ndarray        # rows x p_c


class B6Encoder:
    """Fits the B6 encodings on training positions of a :class:`B6Inputs` and encodes any rows."""

    def __init__(self, side_categoricals: bool = False):
        self.side_categoricals = bool(side_categoricals)
        self.metal: _BlockPrep | None = None
        self.ext: _BlockPrep | None = None
        self.cond: _BlockPrep | None = None
        self.pca: F.ExtractantBlock | None = None
        self._pca_rows_cache: tuple[B6Inputs, np.ndarray] | None = None

    def _pca_by_system(self, inputs: B6Inputs) -> np.ndarray:
        if self._pca_rows_cache is not None and self._pca_rows_cache[0] is inputs:
            return self._pca_rows_cache[1]
        sc = self.pca.pca_scores(list(inputs.primary_smiles)) if inputs.system_labels else np.zeros((0, F.N_PCA))
        self._pca_rows_cache = (inputs, sc)
        return sc

    def _ext_num(self, inputs: B6Inputs, p: np.ndarray) -> np.ndarray:
        if not inputs.has_pca:
            return inputs.ext_num[p]
        return np.hstack([inputs.ext_num[p], self._pca_by_system(inputs)[inputs.system_code[p]]])

    def _cats(self, arr: np.ndarray, p: np.ndarray, use: bool) -> np.ndarray:
        return arr[p] if use else np.zeros((len(p), 0), dtype=object)

    def fit(self, inputs: B6Inputs, positions: np.ndarray) -> "B6Encoder":
        p = np.asarray(positions, dtype=np.int64)
        if not len(p):
            raise ValueError("B6Encoder.fit: no training rows")
        self.pca = _pca_block(inputs, p) if inputs.has_pca else None
        self._pca_rows_cache = None
        ext_cols = inputs.ext_num_cols + (tuple(F.PCA_COLUMNS) if inputs.has_pca else ())
        sc = self.side_categoricals
        self.metal = _BlockPrep("metal", inputs.metal_num_cols, inputs.metal_cat_cols if sc else ()).fit(
            inputs.metal_num[p], self._cats(inputs.metal_cat, p, sc))
        self.ext = _BlockPrep("extractant", ext_cols, inputs.ext_cat_cols if sc else ()).fit(
            self._ext_num(inputs, p), self._cats(inputs.ext_cat, p, sc))
        self.cond = _BlockPrep("condition", inputs.cond_num_cols, inputs.cond_cat_cols).fit(
            inputs.cond_num[p], inputs.cond_cat[p])
        F.assert_feature_columns_allowed(self.feature_names, "B6 feature columns")
        return self

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.metal.names + self.ext.names + self.cond.names

    def transform(self, inputs: B6Inputs, positions: np.ndarray | None = None) -> Encoded:
        if self.metal is None:
            raise RuntimeError("B6Encoder: fit first")
        p = np.arange(inputs.n) if positions is None else np.asarray(positions, dtype=np.int64)
        sc = self.side_categoricals
        if len(inputs.metal_num_cols) != len(self.metal.num_cols) or len(inputs.cond_num_cols) != len(self.cond.num_cols):
            raise ValueError("B6Encoder: inputs do not have the fitted columns")
        return Encoded(Zm=self.metal.transform(inputs.metal_num[p], self._cats(inputs.metal_cat, p, sc)),
                       Zs=self.ext.transform(self._ext_num(inputs, p), self._cats(inputs.ext_cat, p, sc)),
                       X=self.cond.transform(inputs.cond_num[p], inputs.cond_cat[p]))


# --------------------------------------------------------------------------------------------- #
# grouped normal equations: S training sets sharing one group universe
# --------------------------------------------------------------------------------------------- #

@contextmanager
def blas_threads(n: int | None):
    """Limit BLAS threads inside B6 fits (small dense matrices: one thread avoids oversubscription)."""
    if n is None or _threadpool_limits is None:
        yield
        return
    with _threadpool_limits(limits=int(n), user_api="blas"):
        yield


def _column_universe(block: str, preps: Sequence[_BlockPrep]) -> tuple[tuple[str, ...], list[np.ndarray]]:
    """Union of the encoded column names of one block over several fitted encodings (numeric columns first, then each
    categorical column's sorted token union) and each encoding's column positions in it."""
    num, cats = preps[0].num_cols, preps[0].cat_cols
    if any(p.num_cols != num or p.cat_cols != cats for p in preps):
        raise ValueError(f"{block}: encodings with different raw columns")
    names = [f"{block}__{c}" for c in num]
    for j, c in enumerate(cats):
        names += [f"{block}__{c}={t}" for t in sorted({t for p in preps for t in p.vocab[j]})]
    pos = {nm: i for i, nm in enumerate(names)}
    return tuple(names), [np.array([pos[nm] for nm in p.names], dtype=np.int64) for p in preps]


class _Side:
    """One side of the ALS.  ``Z`` (S x patterns x p): side features of each pattern per member; ``pattern_unit``
    (sorted) maps a pattern to its unit (the free parameters).  The metal side has one pattern per unit (a metal
    unit); the system side one pattern per (system, z_s pattern)."""

    def __init__(self, Z: np.ndarray, pattern_unit: np.ndarray, J: int):
        self.Z, self.J = np.ascontiguousarray(Z), int(J)
        self.pattern_unit = np.asarray(pattern_unit, dtype=np.int64)
        if len(self.pattern_unit) and np.any(np.diff(self.pattern_unit) < 0):
            raise AssertionError("patterns must be sorted by unit")
        self.n_patterns = len(self.pattern_unit)
        self.identity_units = bool(self.n_patterns == self.J and np.array_equal(self.pattern_unit, np.arange(self.J)))
        self.u_starts = np.flatnonzero(np.r_[True, np.diff(self.pattern_unit) != 0]).astype(np.int64)
        if len(self.u_starts) != self.J:
            raise AssertionError("a unit without patterns")
        pa, pb = [], []
        bounds = np.r_[self.u_starts, self.n_patterns]
        for a, e in zip(bounds[:-1], bounds[1:]):
            idx = np.arange(a, e)
            pa.append(np.repeat(idx, len(idx)))
            pb.append(np.tile(idx, len(idx)))
        self.pair_a = np.concatenate(pa).astype(np.int64)
        self.pair_b = np.concatenate(pb).astype(np.int64)
        self.pair_unit = self.pattern_unit[self.pair_a]
        self.pair_diag = self.pair_a == self.pair_b
        S, _, p = self.Z.shape
        #: z_a z_b' of every pattern pair of one unit, flattened (constant over the sweeps of a fit)
        self.Zab = np.ascontiguousarray((self.Z[:, self.pair_a, :, None] * self.Z[:, self.pair_b, None, :])
                                        .reshape(S, len(self.pair_a), p * p))

    def to_units(self, per_pattern: np.ndarray) -> np.ndarray:
        return per_pattern if self.identity_units else np.add.reduceat(per_pattern, self.u_starts, axis=2)


def _incidence(rows: np.ndarray, n_rows: int) -> sparse.csr_matrix:
    """``n_rows x cells`` 0/1 matrix mapping each cell to its row."""
    c = len(rows)
    return sparse.csr_matrix((np.ones(c), (np.asarray(rows, dtype=np.int64), np.arange(c))), shape=(n_rows, c))


def _agg(inc: sparse.csr_matrix, X: np.ndarray) -> np.ndarray:
    """Sum a cells-first array ``X`` (cells, ...) over the rows of ``inc``: (rows, ...)."""
    return np.asarray(inc @ X.reshape(X.shape[0], -1)).reshape((inc.shape[0],) + X.shape[1:])


class Design:
    """Encoded training rows of ``S`` member training sets reduced to cell statistics over ONE universe.

    ``members`` = ``[(training positions into inputs, B6Encoder fitted on exactly those positions)]``; ``universe``
    (default: the union of the members' positions) fixes the metal units, the system patterns (system, z_s pattern),
    the cells (pattern, metal unit) and the encoded columns.  Every row statistic is summed per cell: counts ``N``,
    ``sum y``, ``sum c`` (``c = (1, x_cond)``), plus the row moments ``sum c c'``, ``sum c y``, ``sum y^2``.  A cell,
    unit or one-hot column a member lacks enters as zeros, which leaves that member's ridge solution the one of its
    own training rows (the free parts of a unit without rows are 0).  ``S = 1`` is a single fit."""

    def __init__(self, inputs: B6Inputs, members: Sequence[tuple[np.ndarray, "B6Encoder"]], y_all: np.ndarray,
                 universe: np.ndarray | None = None):
        if not members:
            raise ValueError("Design: no member training set")
        pos_list = [np.asarray(p, dtype=np.int64) for p, _ in members]
        U = np.unique(np.concatenate(pos_list)) if universe is None else np.unique(np.asarray(universe, dtype=np.int64))
        if any(not np.isin(p, U).all() for p in pos_list):
            raise ValueError("Design: a member training row is outside the universe")
        y_all = np.asarray(y_all, dtype=float)
        S = self.S = len(members)
        ml, mcode = np.unique(inputs.metal_unit[U].astype(str), return_inverse=True)
        sl, scode = np.unique(inputs.system[U].astype(str), return_inverse=True)
        mcode, scode = mcode.reshape(-1), scode.reshape(-1)
        sp, pinv = np.unique(np.column_stack([scode, inputs.pattern[U]]), axis=0, return_inverse=True)
        pinv = pinv.reshape(-1)
        self.metal_labels, self.system_labels = ml, sl
        self.Jm, self.Js, self.Pi = len(ml), len(sl), len(sp)
        self.pattern_system = sp[:, 0].astype(np.int64)
        cells, cinv = np.unique(pinv * len(ml) + mcode, return_inverse=True)
        cinv = cinv.reshape(-1)
        self.C = len(cells)
        self.cell_p, self.cell_m = (cells // len(ml)).astype(np.int64), (cells % len(ml)).astype(np.int64)
        self.inc_p, self.inc_m = _incidence(self.cell_p, self.Pi), _incidence(self.cell_m, self.Jm)
        upos = np.full(inputs.n, -1, dtype=np.int64)
        upos[U] = np.arange(len(U))
        _, rep_m = np.unique(mcode, return_index=True)
        _, rep_p = np.unique(pinv, return_index=True)
        rep_m, rep_p = U[rep_m], U[rep_p]
        encs = [e for _, e in members]
        self.x_names, self.x_maps = _column_universe("condition", [e.cond for e in encs])
        self.zm_names, self.zm_maps = _column_universe("metal", [e.metal for e in encs])
        self.zs_names, self.zs_maps = _column_universe("extractant", [e.ext for e in encs])
        F.assert_feature_columns_allowed(self.x_names + self.zm_names + self.zs_names, "B6 design columns")
        Pc, pm, ps, Jm, C = 1 + len(self.x_names), len(self.zm_names), len(self.zs_names), self.Jm, self.C
        self.Pc = Pc
        self.N_c, self.Sy_c, self.Sc_c = np.zeros((C, S)), np.zeros((C, S)), np.zeros((C, S, Pc))  # cells first
        self.Scc, self.Sct, self.Syy = np.zeros((S, Pc, Pc)), np.zeros((S, Pc)), np.zeros(S)
        self.Zm, self.Zs_pat = np.zeros((S, Jm, pm)), np.zeros((S, self.Pi, ps))
        for s, (pos, enc) in enumerate(zip(pos_list, encs)):
            n = len(pos)
            if not n:
                raise ValueError("Design: a member without training rows")
            y = y_all[pos]
            if not np.isfinite(y).all():
                raise ValueError("B6: non-finite training target")
            e = enc.transform(inputs, np.concatenate([pos, rep_m, rep_p]))
            Cm = np.zeros((n, Pc))
            Cm[:, 0] = 1.0
            Cm[:, 1 + self.x_maps[s]] = e.X[:n]
            cell = cinv[upos[pos]]
            order = np.argsort(cell, kind="stable")
            cc = cell[order]
            starts = np.flatnonzero(np.r_[True, np.diff(cc) != 0])
            present = cc[starts]
            self.N_c[present, s] = np.diff(np.r_[starts, n])
            self.Sy_c[present, s] = np.add.reduceat(y[order], starts)
            self.Sc_c[present, s] = np.add.reduceat(Cm[order], starts, axis=0)
            self.Scc[s] = Cm.T @ Cm
            self.Sct[s] = Cm.T @ y
            self.Syy[s] = float(y @ y)
            self.Zm[s][:, self.zm_maps[s]] = e.Zm[n:n + Jm]
            self.Zs_pat[s][:, self.zs_maps[s]] = e.Zs[n + Jm:]
        self.Sc_cs = np.ascontiguousarray(self.Sc_c.transpose(1, 0, 2))                        # S, C, Pc
        self.N_cs, self.Sy_cs = np.ascontiguousarray(self.N_c.T), np.ascontiguousarray(self.Sy_c.T)  # S, C
        self.W1_p = np.moveaxis(_agg(self.inc_p, self.Sc_c), 0, 1)                             # S, Pi, Pc
        self.W1_m = np.moveaxis(_agg(self.inc_m, self.Sc_c), 0, 1)                             # S, Jm, Pc
        self.rows_p, self.rows_m = _agg(self.inc_p, self.N_c).T, _agg(self.inc_m, self.N_c).T  # S, Pi / S, Jm
        self.Sy_p, self.Sy_m = _agg(self.inc_p, self.Sy_c).T, _agg(self.inc_m, self.Sy_c).T
        self.p_starts = np.flatnonzero(np.r_[True, np.diff(self.pattern_system) != 0]).astype(np.int64)
        self.rows_s = np.add.reduceat(self.rows_p, self.p_starts, axis=1)                      # S, Js
        self.n_cells = int((self.N_c > 0).sum())
        self.pen = np.ones(Pc)
        self.pen[0] = 0.0
        self.metal_side = _Side(self.Zm, np.arange(Jm), Jm)
        self.system_side = _Side(self.Zs_pat, self.pattern_system, self.Js)

    @property
    def p_m(self) -> int:
        return self.Zm.shape[2]

    @property
    def p_s(self) -> int:
        return self.Zs_pat.shape[2]


@dataclass
class B6Params:
    """Parameters of ``S x L`` members (training set x lambda); ``b = rank + 1`` columns per side."""

    rank: int
    lams: np.ndarray     # L
    gamma: np.ndarray    # S, L, Pc: mu, beta
    P_m: np.ndarray      # S, L, p_m, b: alpha_a | A'
    F_m: np.ndarray      # S, L, J_m, b: a' | delta_m
    P_s: np.ndarray      # S, L, p_s, b: alpha_b | B'
    F_s: np.ndarray      # S, L, J_s, b: b' | delta_s

    FIELDS = ("gamma", "P_m", "F_m", "P_s", "F_s")

    def penalty(self) -> np.ndarray:
        sq = (np.sum(self.gamma[..., 1:] ** 2, axis=-1) + np.sum(self.P_m ** 2, axis=(-1, -2))
              + np.sum(self.F_m ** 2, axis=(-1, -2)) + np.sum(self.P_s ** 2, axis=(-1, -2))
              + np.sum(self.F_s ** 2, axis=(-1, -2)))
        return self.lams[None, :] * sq

    def copy(self) -> "B6Params":
        return B6Params(self.rank, self.lams.copy(), *(getattr(self, f).copy() for f in self.FIELDS))


def objective(d: Design, P: B6Params) -> tuple[np.ndarray, np.ndarray]:
    """``(penalised objective, residual sum of squares)`` per member (S x L), from the cell statistics."""
    um = d.Zm[:, None] @ P.P_m + P.F_m                                                     # S, L, Jm, b
    vp = d.Zs_pat[:, None] @ P.P_s + P.F_s[:, :, d.pattern_system]                         # S, L, Pi, b
    uc, vc = um[:, :, d.cell_m], vp[:, :, d.cell_p]
    t = uc[..., 0] + vc[..., 0] + np.sum(uc[..., 1:] * vc[..., 1:], axis=-1)              # S, L, C
    g = P.gamma
    ScG = (d.Sc_cs[:, None] @ g[..., None])[..., 0]                                       # S, L, C
    sse = (d.Syy[:, None] - 2.0 * np.sum(g * d.Sct[:, None], axis=-1)
           + (g[..., None, :] @ d.Scc[:, None] @ g[..., :, None])[..., 0, 0]
           + np.sum(t * (2.0 * ScG - 2.0 * d.Sy_cs[:, None] + d.N_cs[:, None] * t), axis=-1))
    sse = np.maximum(sse, 0.0)
    return sse + P.penalty(), sse


def _solve_spd(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``A x = b`` for symmetric positive definite ``A`` (batched over leading axes)."""
    try:
        c = sla.cho_factor(A, lower=False, check_finite=False)
        return sla.cho_solve(c, b[..., None], check_finite=False)[..., 0]
    except (np.linalg.LinAlgError, sla.LinAlgError):
        return np.linalg.solve(A, b[..., None])[..., 0]


def fit_additive(d: Design, lambdas: Sequence[float]) -> B6Params:
    """``k = 0`` for every member and lambda: the joint ridge solution of
    ``mu + beta'x + alpha_a'z_m + a'_m + alpha_b'z_s + b'_s``; the system offsets ``b'`` (a diagonal block) are
    eliminated by a Schur complement."""
    lams = np.asarray([float(l) for l in lambdas], dtype=float)
    if not len(lams) or not (lams > 0).all():
        raise ValueError("lambdas must be > 0")
    S, Pc, pm, ps, Jm, Pi = d.S, d.Pc, d.p_m, d.p_s, d.Jm, d.Pi
    Zm, Zp = d.Zm, d.Zs_pat
    q = pm + Jm + ps
    Q = Pc + q
    c1, c2 = Pc + pm, Pc + pm + Jm
    A0 = np.zeros((S, Q, Q))
    W1m_t, W1p_t = d.W1_m.swapaxes(-1, -2), d.W1_p.swapaxes(-1, -2)
    A0[:, :Pc, :Pc] = d.Scc
    A0[:, :Pc, Pc:c1] = W1m_t @ Zm
    A0[:, :Pc, c1:c2] = W1m_t
    A0[:, :Pc, c2:] = W1p_t @ Zp
    nZm = Zm * d.rows_m[..., None]
    A0[:, Pc:c1, Pc:c1] = nZm.swapaxes(-1, -2) @ Zm
    A0[:, Pc:c1, c1:c2] = nZm.swapaxes(-1, -2)
    A0[:, c1:c2, c1:c2][:, np.arange(Jm), np.arange(Jm)] = d.rows_m
    NZp_c = d.N_cs[..., None] * Zp[:, d.cell_p]                                               # S, C, ps
    A0[:, Pc:c1, c2:] = Zm[:, d.cell_m].swapaxes(-1, -2) @ NZp_c
    A0[:, c1:c2, c2:] = np.moveaxis(_agg(d.inc_m, np.moveaxis(NZp_c, 1, 0)), 0, 1)
    A0[:, c2:, c2:] = (Zp * d.rows_p[..., None]).swapaxes(-1, -2) @ Zp
    iu = np.triu_indices(Q, 1)
    A0[:, iu[1], iu[0]] = A0[:, iu[0], iu[1]]
    ro = np.concatenate([d.Sct, (Zm.swapaxes(-1, -2) @ d.Sy_m[..., None])[..., 0], d.Sy_m,
                         (Zp.swapaxes(-1, -2) @ d.Sy_p[..., None])[..., 0]], axis=-1)
    Bp = np.zeros((S, Pi, Q))                                                                  # per pattern
    Bp[:, :, :Pc] = d.W1_p
    Bp[:, :, Pc:c1] = np.moveaxis(_agg(d.inc_p, np.moveaxis(d.N_cs[..., None] * Zm[:, d.cell_m], 1, 0)), 0, 1)
    np.add.at(Bp, (slice(None), d.cell_p, c1 + d.cell_m), d.N_cs)
    Bp[:, :, c2:] = Zp * d.rows_p[..., None]
    B = np.add.reduceat(Bp, d.p_starts, axis=1)                                                # S, Js, Q
    rf = np.add.reduceat(d.Sy_p, d.p_starts, axis=1)                                           # S, Js
    D = d.rows_s[:, None, :] + lams[None, :, None]                                             # S, L, Js
    pen_full = np.concatenate([d.pen, np.ones(q)])
    BD = B[:, None] / D[..., None]                                                             # S, L, Js, Q
    Smat = A0[:, None] + lams[None, :, None, None] * np.diag(pen_full) - BD.swapaxes(-1, -2) @ B[:, None]
    rhs = ro[:, None] - (BD.swapaxes(-1, -2) @ rf[:, None, :, None])[..., 0]
    x = _solve_spd(0.5 * (Smat + Smat.swapaxes(-1, -2)), rhs)                                   # S, L, Q
    fs = (rf[:, None] - (B[:, None] @ x[..., None])[..., 0]) / D
    return B6Params(0, lams, x[..., :Pc].copy(), x[..., Pc:c1, None].copy(), x[..., c1:c2, None].copy(),
                    x[..., c2:, None].copy(), fs[..., None].copy())


def side_step(d: Design, side: _Side, E_pat: np.ndarray, W_pat: np.ndarray, R_pat: np.ndarray, oSc: np.ndarray,
              lams: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One exact ALS half-step for every member: the ridge solution for ``gamma`` (S,L,Pc), ``P`` (S,L,p,b) and
    ``F`` (S,L,J,b) of ``side``, given the other side through per-pattern sums over this side's rows:
    ``E_pat = sum n e e'`` (S,L,patterns,b,b), ``W_pat = sum c e'`` (S,L,patterns,Pc,b),
    ``R_pat = sum (y - o) e`` (S,L,patterns,b) and ``oSc = sum o c`` (S,L,Pc), where ``e = (1, other factor)`` and
    ``o`` is the other side's main effect.

    The free block is eliminated per unit (Schur complement); the side features are constant per pattern, so every
    ``P`` block is a Kronecker sum over patterns and pattern pairs of one unit: O(pairs p^2 b^2) per member."""
    S, L, _, b, _ = E_pat.shape
    Z = side.Z
    p, Pc, J = Z.shape[2], d.Pc, side.J
    Q = Pc + p * b
    E_u, W_u, R_u = side.to_units(E_pat), side.to_units(W_pat), side.to_units(R_pat)
    D = E_u.copy()
    D[..., np.arange(b), np.arange(b)] += lams[None, :, None, None]
    Dinv = np.linalg.inv(D)
    Dinv = 0.5 * (Dinv + Dinv.swapaxes(-1, -2))
    WD = W_u @ Dinv                                                                        # S, L, J, Pc, b
    DR = (Dinv @ R_u[..., None])[..., 0]                                                   # S, L, J, b
    A = np.empty((S, L, Q, Q))
    X1 = WD.transpose(0, 1, 3, 2, 4).reshape(S, L, Pc, J * b)
    X2 = W_u.transpose(0, 1, 3, 2, 4).reshape(S, L, Pc, J * b)
    A[..., :Pc, :Pc] = d.Scc[:, None] + lams[None, :, None, None] * np.diag(d.pen) - X1 @ X2.swapaxes(-1, -2)
    if side.identity_units:
        cP = W_pat - WD @ E_pat
        T = E_pat - E_pat @ Dinv @ E_pat
        DRp = DR
    else:
        pu = side.pattern_unit
        cP = W_pat - WD[:, :, pu] @ E_pat                                                  # S, L, patterns, Pc, b
        T = -(E_pat[:, :, side.pair_a] @ Dinv[:, :, side.pair_unit] @ E_pat[:, :, side.pair_b])
        T[:, :, side.pair_diag] += E_pat[:, :, side.pair_a[side.pair_diag]]                 # S, L, pairs, b, b
        DRp = DR[:, :, pu]
    Pi = cP.shape[2]
    AcP = (np.ascontiguousarray(cP.transpose(0, 1, 3, 4, 2)).reshape(S, L, Pc * b, Pi) @ Z[:, None]
           ).reshape(S, L, Pc, b, p)                                                        # S, L, Pc, b, p
    A[..., :Pc, Pc:] = AcP.transpose(0, 1, 2, 4, 3).reshape(S, L, Pc, p * b)
    A[..., Pc:, :Pc] = A[..., :Pc, Pc:].swapaxes(-1, -2)
    ia, ib = np.triu_indices(b)
    PPu = (T[..., ia, ib].swapaxes(-1, -2) @ side.Zab[:, None]).reshape(S, L, len(ia), p, p)   # (a <= a') blocks
    PP = np.empty((S, L, b, b, p, p))
    PP[:, :, ib, ia] = PPu.swapaxes(-1, -2)                                                 # A symmetric
    PP[:, :, ia, ib] = PPu
    A[..., Pc:, Pc:] = PP.transpose(0, 1, 4, 2, 5, 3).reshape(S, L, p * b, p * b)
    idx = np.arange(Pc, Q)
    A[..., idx, idx] += lams[None, :, None]
    A = 0.5 * (A + A.swapaxes(-1, -2))
    rc = d.Sct[:, None] - oSc - np.sum((W_u @ DR[..., None])[..., 0], axis=2)
    rP_pat = R_pat - (E_pat @ DRp[..., None])[..., 0]                                        # S, L, patterns, b
    rP = (rP_pat.swapaxes(-1, -2) @ Z[:, None]).swapaxes(-1, -2).reshape(S, L, p * b)
    x = _solve_spd(A, np.concatenate([rc, rP], axis=-1))
    gamma, P = x[..., :Pc].copy(), x[..., Pc:].reshape(S, L, p, b).copy()
    ZP = Z[:, None] @ P                                                                    # S, L, patterns, b
    BtX = (W_u.swapaxes(-1, -2) @ gamma[:, :, None, :, None])[..., 0] + side.to_units((E_pat @ ZP[..., None])[..., 0])
    F = DR - (Dinv @ BtX[..., None])[..., 0]
    return gamma, P, F


def _cell_sums(d: Design, inc: sparse.csr_matrix, W1: np.ndarray, e: np.ndarray, o: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-pattern sums ``(E, W, R, oSc)`` of :func:`side_step` from per-cell other-side factors ``e`` (S,L,C,b) with
    ``e[..., 0] = 1`` and main effects ``o`` (S,L,C)."""
    S, L, C, b = e.shape
    ec = np.ascontiguousarray(np.moveaxis(e, 2, 0))                                       # C, S, L, b
    oc = np.moveaxis(o, 2, 0)                                                              # C, S, L
    n = d.N_c[:, :, None, None]
    E = np.moveaxis(_agg(inc, (n * ec)[..., :, None] * ec[..., None, :]), 0, 2)             # S, L, rows, b, b
    rows = inc.shape[0]
    W = np.empty((S, L, rows, d.Pc, b))
    W[..., 0] = W1[:, None]
    for a in range(1, b):
        W[..., a] = np.moveaxis(_agg(inc, d.Sc_c[:, :, None, :] * ec[:, :, :, a, None]), 0, 2)
    R = np.moveaxis(_agg(inc, (d.Sy_c[:, :, None] - d.N_c[:, :, None] * oc)[..., None] * ec), 0, 2)
    oSc = (d.Sc_cs.swapaxes(-1, -2)[:, None] @ o[..., None])[..., 0]                       # S, L, Pc
    return E, W, R, oSc


def system_step(d: Design, P: B6Params) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(gamma, P_s, F_s)`` given the metal side of ``P``."""
    um = d.Zm[:, None] @ P.P_m + P.F_m                                                     # S, L, Jm, b
    e = um[:, :, d.cell_m]
    o = e[..., 0].copy()
    e[..., 0] = 1.0
    return side_step(d, d.system_side, *_cell_sums(d, d.inc_p, d.W1_p, e, o), P.lams)


def metal_step(d: Design, P: B6Params) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(gamma, P_m, F_m)`` given the system side of ``P``."""
    vp = d.Zs_pat[:, None] @ P.P_s + P.F_s[:, :, d.pattern_system]                         # S, L, Pi, b
    e = vp[:, :, d.cell_p]
    o = e[..., 0].copy()
    e[..., 0] = 1.0
    return side_step(d, d.metal_side, *_cell_sums(d, d.inc_m, d.W1_m, e, o), P.lams)


_INIT_CACHE: "OrderedDict[tuple[int, int, str], np.ndarray]" = OrderedDict()
_INIT_CACHE_MAX = 20000


def unit_init(seed: int, rank: int, label: str) -> np.ndarray:
    """``N(0, 1)`` draw of one metal unit's free factor, keyed by (seed, rank, label) only."""
    key = (int(seed), int(rank), str(label))
    v = _INIT_CACHE.get(key)
    if v is None:
        ss = np.random.SeedSequence([int(seed), INIT_STREAM_TAG, int(rank), zlib.crc32(str(label).encode("utf-8"))])
        v = np.random.default_rng(ss).standard_normal(int(rank))
        v.setflags(write=False)
        _INIT_CACHE[key] = v
        if len(_INIT_CACHE) > _INIT_CACHE_MAX:
            _INIT_CACHE.popitem(last=False)
    return v


@dataclass
class FitInfo:
    """Convergence record of ``S x L`` members."""

    rank: int
    lams: np.ndarray
    seed: int | None
    n_sweeps: np.ndarray         # S, L
    converged: np.ndarray        # S, L
    objective_trace: np.ndarray  # sweeps + 1, S, L (a finished member's last value is repeated)
    sse: np.ndarray              # S, L
    runtime_s: float

    def member(self, s: int = 0, l: int = 0) -> dict[str, Any]:
        n = int(self.n_sweeps[s, l])
        return {"rank": self.rank, "lambda": float(self.lams[l]), "seed": self.seed, "n_sweeps": n,
                "converged": bool(self.converged[s, l]), "objective_trace": self.objective_trace[:n + 1, s, l].tolist(),
                "sse": float(self.sse[s, l]), "runtime_s": float(self.runtime_s)}


def additive_info(d: Design, P: B6Params, seed: int | None, runtime_s: float = 0.0) -> FitInfo:
    obj, sse = objective(d, P)
    S, L = obj.shape
    return FitInfo(0, P.lams.copy(), None if seed is None else int(seed), np.zeros((S, L), dtype=np.int64),
                   np.ones((S, L), dtype=bool), obj[None].copy(), sse, runtime_s)


def fit_als(d: Design, rank: int, base: B6Params, seed: int, *, max_sweeps: int = MAX_SWEEPS, tol: float = REL_TOL,
            init_scale: float = INIT_SCALE) -> tuple[B6Params, FitInfo]:
    """``k >= 1`` for every member by alternating ridge least squares (module docstring).  ``base`` must be
    :func:`fit_additive` of the same design; a member stops (its parameters frozen) at the first sweep whose relative
    objective change is <= ``tol``, or after ``max_sweeps``."""
    t0 = time.perf_counter()
    rank = int(rank)
    if rank < 1:
        raise ValueError("fit_als: rank >= 1 (rank 0 is fit_additive)")
    if seed is None:
        raise ValueError("B6: the seeded initialisation needs a seed (context.seed)")
    if base.rank != 0 or base.gamma.shape[0] != d.S or base.F_m.shape[2] != d.Jm or base.F_s.shape[2] != d.Js:
        raise ValueError("fit_als: base is not the k=0 solution of this design")
    S, L = base.gamma.shape[:2]
    b = rank + 1
    lams = base.lams.copy()
    P = B6Params(rank, lams, base.gamma.copy(), np.zeros((S, L, d.p_m, b)), np.zeros((S, L, d.Jm, b)),
                 np.zeros((S, L, d.p_s, b)), np.zeros((S, L, d.Js, b)))
    P.P_m[..., 0], P.F_m[..., 0] = base.P_m[..., 0], base.F_m[..., 0]
    P.P_s[..., 0], P.F_s[..., 0] = base.P_s[..., 0], base.F_s[..., 0]
    draws = np.vstack([unit_init(seed, rank, lab) for lab in d.metal_labels])                     # Jm, rank
    rmse0 = np.sqrt(objective(d, base)[1] / d.rows_m.sum(axis=1)[:, None])                        # S, L
    P.F_m[..., 1:] = (float(init_scale) * np.sqrt(rmse0))[:, :, None, None] * draws[None, None] *         (d.rows_m > 0)[:, None, :, None]
    obj, sse = objective(d, P)
    trace = [obj.copy()]
    active = np.ones((S, L), dtype=bool)
    converged = np.zeros((S, L), dtype=bool)
    n_sweeps = np.zeros((S, L), dtype=np.int64)
    for sweep in range(1, int(max_sweeps) + 1):
        new = P.copy()
        new.gamma, new.P_s, new.F_s = system_step(d, new)
        new.gamma, new.P_m, new.F_m = metal_step(d, new)
        obj_new, sse_new = objective(d, new)
        for f in B6Params.FIELDS:
            cur = getattr(P, f)
            setattr(P, f, np.where(active.reshape(S, L, *([1] * (cur.ndim - 2))), getattr(new, f), cur))
        done = active & (np.abs(obj - obj_new) <= tol * np.maximum(np.maximum(np.abs(obj), np.abs(obj_new)), 1e-300))
        n_sweeps[active] = sweep
        sse = np.where(active, sse_new, sse)
        obj = np.where(active, obj_new, obj)
        trace.append(obj.copy())
        converged |= done
        active &= ~done
        if not active.any():
            break
    info = FitInfo(rank, lams, int(seed), n_sweeps, converged, np.stack(trace), sse, time.perf_counter() - t0)
    return P, info


def _lookup(labels: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Positions of ``values`` in the sorted ``labels`` (-1 when absent)."""
    v = np.asarray(values).astype(str)
    if not len(labels):
        return np.full(len(v), -1, dtype=np.int64)
    pos = np.minimum(np.searchsorted(labels, v), len(labels) - 1)
    return np.where(labels[pos] == v, pos, -1).astype(np.int64)


def predict_member(P: B6Params, d: Design, s: int, l: int, enc: Encoded, metal_unit: np.ndarray, system: np.ndarray
                   ) -> dict[str, np.ndarray]:
    """Row predictions of member ``(s, l)`` from rows encoded by THAT member's encoder; a metal unit or system without
    training rows in the member gets free parts 0 (side information only)."""
    gamma = P.gamma[s, l]
    Pm, Ps = P.P_m[s, l][d.zm_maps[s]], P.P_s[s, l][d.zs_maps[s]]
    mi, si = _lookup(d.metal_labels, metal_unit), _lookup(d.system_labels, system)
    ms, ss = mi >= 0, si >= 0
    ms[ms] = d.rows_m[s, mi[ms]] > 0
    ss[ss] = d.rows_s[s, si[ss]] > 0
    um, vs = enc.Zm @ Pm, enc.Zs @ Ps
    um[ms] += P.F_m[s, l][mi[ms]]
    vs[ss] += P.F_s[s, l][si[ss]]
    inter = np.sum(um[:, 1:] * vs[:, 1:], axis=1)
    mean = gamma[0] + enc.X @ gamma[1 + d.x_maps[s]] + um[:, 0] + vs[:, 0] + inter
    return {"mean": mean, "metal_seen": ms, "system_seen": ss, "main_metal": um[:, 0], "main_system": vs[:, 0],
            "interaction": inter}


# --------------------------------------------------------------------------------------------- #
# the arm
# --------------------------------------------------------------------------------------------- #

def check_training_mask(table: I.RowTable, mask: np.ndarray, context: I.FitContext) -> np.ndarray:
    """A fit's training mask: right shape, non-empty, finite targets, no ``context.hidden_index`` row."""
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (table.n,):
        raise ValueError("training mask does not match the RowTable")
    if not mask.any():
        raise ValueError("no training rows")
    if not np.isfinite(table.y[mask]).all():
        raise ValueError("training rows with a non-finite log_D")
    forbidden = I.forbidden_mask(table, context)
    if (mask & forbidden).any():
        raise AssertionError(f"{int((mask & forbidden).sum())} hidden row(s) among the training rows")
    return mask


def _prediction_frame(labels: np.ndarray, out: dict[str, np.ndarray], name: str, rank: int, lam: float,
                      info: dict[str, Any], metal_unit: np.ndarray) -> pd.DataFrame:
    n = len(labels)
    data: dict[str, Any] = {}
    for c in I.PREDICTION_COLUMNS:
        data[c] = np.full(n, np.nan) if c in I._FLOAT_COLUMNS else np.full(n, None, dtype=object)
    data["row_id"] = np.asarray(labels, dtype=object)
    data["mean_logD"] = np.asarray(out["mean"], dtype=float)
    data["fallback_level"] = np.full(n, name, dtype=object)
    data["fallback_reason"] = np.array([";".join(k for k, seen in (("metal_unseen", m), ("system_unseen", s)) if not seen)
                                        for m, s in zip(out["metal_seen"], out["system_seen"])], dtype=object)
    frame = pd.DataFrame(data, columns=list(I.PREDICTION_COLUMNS))
    extra = {"b6_rank": np.full(n, int(rank)), "b6_lambda": np.full(n, float(lam)),
             "b6_metal_unit": np.asarray(metal_unit, dtype=object),
             "b6_metal_seen": np.asarray(out["metal_seen"], dtype=bool),
             "b6_system_seen": np.asarray(out["system_seen"], dtype=bool), "b6_main_metal": out["main_metal"],
             "b6_main_system": out["main_system"], "b6_interaction": out["interaction"],
             "b6_n_sweeps": np.full(n, int(info["n_sweeps"])), "b6_converged": np.full(n, bool(info["converged"])),
             "b6_seed": np.full(n, -1 if info["seed"] is None else int(info["seed"]))}
    return pd.concat([frame, pd.DataFrame(extra, index=frame.index)], axis=1)


class B6Factorized:
    """B6 at one fixed ``(rank, lambda)`` (``rank = 0`` is the additive model).  Implements the arm protocol and the
    ``fit_table`` / ``predict_positions`` fast path (``register_table`` first).  Deterministic given ``seed``
    (``None`` = ``context.seed``)."""

    def __init__(self, rank: int = 0, lam: float = 1.0, *, name: str | None = None, seed: int | None = None,
                 max_sweeps: int = MAX_SWEEPS, tol: float = REL_TOL, init_scale: float = INIT_SCALE,
                 side_categoricals: bool = False, blas: int | None = BLAS_THREADS):
        if int(rank) != rank or rank < 0:
            raise ValueError("rank must be a non-negative integer")
        if not (float(lam) > 0):
            raise ValueError("lambda must be > 0")
        self.rank, self.lam = int(rank), float(lam)
        self.name = name if name is not None else ("B6r0" if self.rank == 0 else "B6")
        self.seed, self.max_sweeps, self.tol, self.init_scale = seed, int(max_sweeps), float(tol), float(init_scale)
        self.side_categoricals, self.blas = bool(side_categoricals), blas
        self.encoder: B6Encoder | None = None
        self.design: Design | None = None
        self.params: B6Params | None = None
        self.fit_info: FitInfo | None = None
        self._inputs: B6Inputs | None = None
        self._mask: np.ndarray | None = None
        self._train_labels: pd.Index | None = None

    @property
    def info(self) -> dict[str, Any]:
        if self.fit_info is None:
            raise RuntimeError(f"{self.name}: fit first")
        return self.fit_info.member(0, 0)

    def clone(self) -> "B6Factorized":
        return B6Factorized(self.rank, self.lam, name=self.name, seed=self.seed, max_sweeps=self.max_sweeps,
                            tol=self.tol, init_scale=self.init_scale, side_categoricals=self.side_categoricals,
                            blas=self.blas)

    def _seed(self, context: I.FitContext) -> int | None:
        s = self.seed if self.seed is not None else context.seed
        if s is None and self.rank > 0:
            raise ValueError("B6: rank > 0 needs a seed (B6Factorized(seed=...) or context.seed)")
        return None if s is None else int(s)

    def _fit(self, inputs: B6Inputs, positions: np.ndarray, y_all: np.ndarray, seed: int | None) -> "B6Factorized":
        t0 = time.perf_counter()
        with blas_threads(self.blas):
            self.encoder = B6Encoder(self.side_categoricals).fit(inputs, positions)
            self.design = Design(inputs, [(positions, self.encoder)], y_all)
            base = fit_additive(self.design, [self.lam])
            if self.rank == 0:
                self.params, self.fit_info = base, additive_info(self.design, base, seed, time.perf_counter() - t0)
            else:
                self.params, self.fit_info = fit_als(self.design, self.rank, base, seed, max_sweeps=self.max_sweeps,
                                                     tol=self.tol, init_scale=self.init_scale)
        return self

    def fit(self, train_rows: pd.DataFrame, context: I.FitContext) -> "B6Factorized":
        if context.hidden_index is not None and train_rows.index.isin(context.hidden_index).any():
            raise AssertionError("a hidden row is among the training rows")
        if context.table is not None:
            table, mask = I.table_and_mask(train_rows, context)
            if table is context.table:
                return self.fit_table(table, mask, context)
        inputs = B6Inputs.from_frame(train_rows)
        y = pd.to_numeric(train_rows[I.TARGET_COL], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(y).all():
            raise ValueError("training rows with a non-finite log_D")
        self._inputs, self._mask, self._train_labels = inputs, None, train_rows.index
        return self._fit(inputs, np.arange(inputs.n), y, self._seed(context))

    def fit_table(self, table: I.RowTable, mask: np.ndarray, context: I.FitContext) -> "B6Factorized":
        inputs = inputs_for(table, context.train_rows)
        mask = check_training_mask(table, mask, context)
        pos = np.flatnonzero(mask)
        self._inputs, self._mask, self._train_labels = inputs, mask.copy(), table.index[pos]
        return self._fit(inputs, pos, table.y, self._seed(context))

    def _require(self) -> None:
        if self.params is None:
            raise RuntimeError(f"{self.name}: fit first")

    def _frame(self, inputs: B6Inputs, p: np.ndarray) -> pd.DataFrame:
        enc = self.encoder.transform(inputs, p)
        out = predict_member(self.params, self.design, 0, 0, enc, inputs.metal_unit[p], inputs.system[p])
        return _prediction_frame(inputs.index[p].to_numpy(dtype=object), out, self.name, self.rank, self.lam,
                                 self.info, inputs.metal_unit[p])

    def predict_positions(self, positions: np.ndarray) -> pd.DataFrame:
        self._require()
        if self._mask is None:
            raise RuntimeError("predict_positions needs a fit_table fit")
        p = np.asarray(positions, dtype=np.int64)
        if len(p) and self._mask[p].any():
            raise AssertionError("a query row is one of the training rows")
        return self._frame(self._inputs, p)

    def predict(self, query_rows: pd.DataFrame) -> pd.DataFrame:
        self._require()
        if query_rows.index.isin(self._train_labels).any():
            raise AssertionError("a query row is one of the training rows")
        q = B6Inputs.from_frame(query_rows)
        return self._frame(q, np.arange(q.n))

    def predict_inputs(self, inputs: B6Inputs, positions: np.ndarray | None = None) -> pd.DataFrame:
        """Predictions for pre-built inputs (e.g. synthetic rows); refuses training labels."""
        self._require()
        p = np.arange(inputs.n) if positions is None else np.asarray(positions, dtype=np.int64)
        if inputs.index[p].isin(self._train_labels).any():
            raise AssertionError("a query row is one of the training rows")
        return self._frame(inputs, p)

    def member_params(self) -> dict[str, Any]:
        """The fitted parameters (``gamma``, ``P_m``, ``F_m``, ``P_s``, ``F_s``), unit labels and encoded column names."""
        self._require()
        d, P = self.design, self.params
        return {"gamma": P.gamma[0, 0], "P_m": P.P_m[0, 0], "F_m": P.F_m[0, 0], "P_s": P.P_s[0, 0], "F_s": P.F_s[0, 0],
                "metal_labels": d.metal_labels, "system_labels": d.system_labels, "x_names": d.x_names,
                "zm_names": d.zm_names, "zs_names": d.zs_names}

    def factors(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """``(metal units: a_m, u_m; system patterns: system, b_s, v_s)`` of the fit (section 8 reliability inputs)."""
        self._require()
        d, P = self.design, self.params
        um = d.Zm[0] @ P.P_m[0, 0] + P.F_m[0, 0]
        vp = d.Zs_pat[0] @ P.P_s[0, 0] + P.F_s[0, 0][d.pattern_system]
        m = pd.DataFrame(um, columns=["a"] + [f"u{i}" for i in range(1, self.rank + 1)])
        m.insert(0, "metal_unit", d.metal_labels)
        m.insert(1, "n_rows", d.rows_m[0])
        s = pd.DataFrame(vp, columns=["b"] + [f"v{i}" for i in range(1, self.rank + 1)])
        s.insert(0, "extractant_system_key", d.system_labels[d.pattern_system])
        return m, s


# --------------------------------------------------------------------------------------------- #
# section 7 tuning and split-conformal intervals of the tuned arms
# --------------------------------------------------------------------------------------------- #

Config = tuple[int, float]


def select_config(maes: Mapping[Config, float], ranks: Sequence[int] = RANKS, lambdas: Sequence[float] = LAMBDAS,
                  margin: float = TIE_MARGIN) -> Config:
    """Section 7: the smallest configuration (lower rank, then stronger penalty) within ``margin`` of the best inner
    macro MAE among ``ranks x lambdas``."""
    cands = {(int(k), float(l)): float(maes[(int(k), float(l))]) for k in ranks for l in lambdas}
    if not cands or not all(np.isfinite(v) for v in cands.values()):
        raise ValueError("select_config: missing or non-finite inner MAE")
    best = min(cands.values())
    ok = [c for c, v in cands.items() if v <= best + margin + 1e-12]
    return min(ok, key=lambda c: (c[0], -c[1]))


def _mask_digest(mask: np.ndarray) -> str:
    m = np.asarray(mask, dtype=bool)
    return hashlib.sha1(np.packbits(m).tobytes() + str(int(m.sum())).encode()).hexdigest()


@dataclass
class B6InnerTuning:
    """Inner predictions of every configuration on every inner split of one training set.

    ``row_units`` holds, per split, the section 4 / 7 averaging unit of every calibration row (``InnerSplit.row_units``;
    a split without them is one unit, :data:`REGISTRATION_CHOICES` ``inner_macro_mae``).  ``seed`` is the run seed that
    drew the inner splits (``context.seed``), ``init_seed`` the seed of the factor initialisation (the section 15 model
    seed when the caller passes one, else ``context.seed``), ``fold_subset`` the inner folds tuning was restricted to
    (``None``: every inner fold of the design)."""

    configs: tuple[Config, ...]
    table_id: int
    mask_digest: str
    seed: int
    units: list[Any]
    folds: list[int]
    cal_positions: list[np.ndarray]
    y: list[np.ndarray]
    predictions: dict[Config, list[np.ndarray]]
    n_sweeps: dict[Config, list[int]]
    converged: dict[Config, list[bool]]
    runtime_s: float = 0.0
    row_units: list[np.ndarray | None] = field(default_factory=list)
    init_seed: int | None = None
    fold_subset: tuple[int, ...] | None = None
    units_from_splits: bool = True

    def _members(self, folds: Iterable[int] | None = None, exclude_folds: Iterable[int] | None = None) -> list[int]:
        keep = None if folds is None else {int(f) for f in folds}
        drop = set() if exclude_folds is None else {int(f) for f in exclude_folds}
        return [i for i, f in enumerate(self.folds) if (keep is None or int(f) in keep) and int(f) not in drop]

    def _row_units(self, i: int) -> np.ndarray:
        if i < len(self.row_units) and self.row_units[i] is not None:
            return np.asarray(self.row_units[i], dtype=object).astype(str)
        return np.full(len(self.y[i]), repr(self.units[i]), dtype=object).astype(str)

    def unit_mae(self, config: Config, *, folds: Iterable[int] | None = None,
                 exclude_folds: Iterable[int] | None = None) -> pd.Series:
        """Per averaging unit: the mean absolute error of the unit's calibration rows pooled over the chosen splits (a
        unit present in several splits, e.g. the V1 ``REMAINDER``, is pooled across them)."""
        members = self._members(folds, exclude_folds)
        if not members:
            return pd.Series(dtype=float)
        err = np.concatenate([np.abs(self.y[i] - self.predictions[config][i]) for i in members])
        units = np.concatenate([self._row_units(i) for i in members])
        return pd.Series(err).groupby(units, sort=True).mean()

    def fold_macro_mae(self, config: Config, *, folds: Iterable[int] | None = None,
                       exclude_folds: Iterable[int] | None = None) -> pd.Series:
        """Per inner fold (index): the unit-macro MAE of the chosen splits of that fold -- a unit's rows pooled over the
        fold's splits, the unit MAEs averaged (``inner_design.per_fold_unit_macro``)."""
        members = self._members(folds, exclude_folds)
        if not members:
            return pd.Series(dtype=float)
        return ID.per_fold_unit_macro([np.abs(self.y[i] - self.predictions[config][i]) for i in members],
                                      [self._row_units(i) for i in members], [self.folds[i] for i in members])

    def macro_mae(self, config: Config, *, folds: Iterable[int] | None = None,
                  exclude_folds: Iterable[int] | None = None) -> float:
        """The selection score (addendum 1 item 2): the mean over the chosen inner folds of :meth:`fold_macro_mae`."""
        fm = self.fold_macro_mae(config, folds=folds, exclude_folds=exclude_folds)
        return float(fm.mean()) if len(fm) else float("nan")

    def maes(self, *, folds: Iterable[int] | None = None, exclude_folds: Iterable[int] | None = None
             ) -> dict[Config, float]:
        return {c: self.macro_mae(c, folds=folds, exclude_folds=exclude_folds) for c in self.configs}

    def select(self, ranks: Sequence[int] = RANKS, lambdas: Sequence[float] = LAMBDAS,
               margin: float = TIE_MARGIN, *, folds: Iterable[int] | None = None,
               exclude_folds: Iterable[int] | None = None) -> Config:
        return select_config(self.maes(folds=folds, exclude_folds=exclude_folds), ranks, lambdas, margin)

    def residuals(self, config: Config, *, folds: Iterable[int] | None = None) -> np.ndarray:
        members = self._members(folds)
        if not members:
            return np.zeros(0)
        return np.concatenate([np.abs(self.y[i] - self.predictions[config][i]) for i in members])

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame.from_records([
            {"rank": c[0], "lambda": c[1], "inner_macro_mae": self.macro_mae(c),
             "pooled_unit_macro_mae_diagnostic": float(self.unit_mae(c).mean()) if len(self.y) else float("nan"),
             "n_splits": len(self.y), "n_inner_folds": len(set(self.folds)),
             "n_units": int(len(self.unit_mae(c))), "n_not_converged": int(sum(not v for v in self.converged[c])),
             "max_sweeps_used": int(max(self.n_sweeps[c])), "mean_sweeps": float(np.mean(self.n_sweeps[c]))}
            for c in self.configs])

    def fold_table(self) -> pd.DataFrame:
        """One row per (configuration, inner fold): the fold's unit-macro MAE."""
        recs = []
        for c in self.configs:
            for f, v in self.fold_macro_mae(c).items():
                recs.append({"rank": c[0], "lambda": c[1], "inner_fold": int(f), "macro_mae": float(v)})
        return pd.DataFrame.from_records(recs, columns=["rank", "lambda", "inner_fold", "macro_mae"])


class FixedSplits:
    """A splitter returning pre-built inner splits (one draw of an inner design shared by tuning and calibration)."""

    def __init__(self, splits: Sequence[I.InnerSplit], name: str = "fixed_splits"):
        self._splits = list(splits)
        self.name = name

    def splits(self, table: I.RowTable, mask: np.ndarray, context: I.FitContext) -> list[I.InnerSplit]:
        return list(self._splits)


def run_inner_tuning(table: I.RowTable, mask: np.ndarray, context: I.FitContext, splitter: Any, *,
                     guard: str = "every_split", ranks: Sequence[int] = RANKS, lambdas: Sequence[float] = LAMBDAS,
                     fold_subset: Sequence[int] | None = None, max_sweeps: int = MAX_SWEEPS, tol: float = REL_TOL,
                     init_scale: float = INIT_SCALE, side_categoricals: bool = False, chunk_size: int = TUNING_CHUNK,
                     blas: int | None = BLAS_THREADS, init_seed: int | None = None,
                     require_row_units: bool = False) -> B6InnerTuning:
    """Fit every ``ranks x lambdas`` configuration on every inner split of the training rows ``mask`` (section 7).

    Every split gets its own encoder, fitted on its inner training rows; the splits are solved ``chunk_size`` at a time
    as members of one :class:`Design` whose universe is the outer training rows (each member's solution is the
    cold-started fit of its own training rows).  The inner splits are drawn with ``context.seed``; the factor
    initialisation uses ``init_seed`` (the section 15 model seed of the outer fold) when given, else ``context.seed``.
    ``require_row_units`` refuses a split without ``InnerSplit.row_units`` (the design's averaging unit)."""
    t0 = time.perf_counter()
    ID.assert_learned_arm_design(splitter, "B6 inner tuning")
    if context.seed is None:
        raise ValueError("inner tuning draws its splits and initialisation from context.seed")
    seed = int(context.seed)
    init = seed if init_seed is None else int(init_seed)
    inputs = inputs_for(table, context.train_rows)
    mask = check_training_mask(table, mask, context)
    splits = splitter.splits(table, mask, context)
    if fold_subset is not None:
        keep = {int(f) for f in fold_subset}
        splits = [sp for sp in splits if int(sp.fold) in keep]
    if not splits:
        raise ValueError("the inner design produced no split")
    missing_units = [sp.unit for sp in splits if getattr(sp, "row_units", None) is None]
    if require_row_units and missing_units:
        raise ValueError(f"inner split(s) {missing_units[:3]} carry no row_units (the design's averaging unit)")
    ranks = tuple(int(k) for k in ranks)
    lams = tuple(float(l) for l in lambdas)
    configs = tuple((k, l) for k in ranks for l in lams)
    verifier = I.ConformalWrapper(B6Factorized(0, 1.0), splitter=splitter, guard=guard)
    preds: dict[Config, list[np.ndarray]] = {c: [] for c in configs}
    sweeps: dict[Config, list[int]] = {c: [] for c in configs}
    conv: dict[Config, list[bool]] = {c: [] for c in configs}
    units, folds, cal_pos, ys, row_units = [], [], [], [], []
    outer_pos = np.flatnonzero(mask)
    step = max(1, int(chunk_size))
    with blas_threads(blas):
        for start in range(0, len(splits), step):
            chunk = splits[start:start + step]
            members, cal_enc = [], []
            for sp in chunk:
                if (sp.train_mask & ~mask).any() or not mask[sp.cal_positions].all() or \
                        sp.train_mask[sp.cal_positions].any():
                    raise AssertionError(f"inner split {sp.unit} is not inside the training rows")
                verifier._verify(table, sp, context)
                FR.assert_not_scored(table.index[sp.cal_positions], context.v6_mask, "B6 inner tuning set")
                ctx = context.for_training(None, hidden_index=table.index[sp.hidden_positions])
                pos = np.flatnonzero(check_training_mask(table, sp.train_mask, ctx))
                enc = B6Encoder(side_categoricals).fit(inputs, pos)
                cal = np.asarray(sp.cal_positions, dtype=np.int64)
                members.append((pos, enc))
                cal_enc.append((cal, enc.transform(inputs, cal)))
                units.append(sp.unit)
                folds.append(int(sp.fold))
                cal_pos.append(cal)
                ys.append(table.y[cal].copy())
                ru = getattr(sp, "row_units", None)
                row_units.append(None if ru is None else np.asarray(ru, dtype=object).copy())
            d = Design(inputs, members, table.y, universe=outer_pos)
            base = fit_additive(d, lams)
            for k in ranks:
                if k == 0:
                    P, info = base, additive_info(d, base, init)
                else:
                    P, info = fit_als(d, k, base, init, max_sweeps=max_sweeps, tol=tol, init_scale=init_scale)
                for s, (cal, enc_cal) in enumerate(cal_enc):
                    for li, lam in enumerate(lams):
                        mean = predict_member(P, d, s, li, enc_cal, inputs.metal_unit[cal], inputs.system[cal])["mean"]
                        if not np.isfinite(mean).all():
                            raise AssertionError(f"B6 k={k} lambda={lam}: non-finite inner prediction in {chunk[s].unit}")
                        preds[(k, lam)].append(mean)
                        sweeps[(k, lam)].append(int(info.n_sweeps[s, li]))
                        conv[(k, lam)].append(bool(info.converged[s, li]))
            del d, members
    return B6InnerTuning(configs=configs, table_id=id(table), mask_digest=_mask_digest(mask), seed=seed, units=units,
                         folds=folds, cal_positions=cal_pos, y=ys, predictions=preds, n_sweeps=sweeps, converged=conv,
                         runtime_s=time.perf_counter() - t0, row_units=row_units, init_seed=init,
                         fold_subset=None if fold_subset is None else tuple(sorted({int(f) for f in fold_subset})),
                         units_from_splits=not missing_units)


#: section 12 calibration readings of a tuned B6 arm (:class:`B6TunedConformal`)
CALIBRATIONS: tuple[str, ...] = ("cross_fit", "tuning_residuals")


class B6TunedConformal(I.ConformalWrapper):
    """The registered B6 (``variant="B6"``, ranks 0-3) or B6r0 (``variant="B6r0"``, rank 0): section 7 inner selection
    of ``(rank, lambda)``, split-conformal intervals, refit on all training rows (module docstring).  One seed per
    wrapper: the selection depends on the inner draw.

    ``calibration="cross_fit"`` (default; the discovery reading, addendum 1 item 2): no calibration residual comes
    from a row that chose the configuration it is a residual of.  The residuals of inner fold ``j`` are those of the
    configuration selected on the OTHER inner folds, taken from the fits already made (every configuration was fitted
    on every split, so nothing is refitted).  With fewer than two inner folds the arm is not calibrated (NaN quantiles,
    ``calibration_record['status']``).  ``calibration="tuning_residuals"`` is the pre-discovery reading (residuals of
    the selected configuration on the tuning splits themselves; tested equal to
    ``ConformalWrapper(B6Factorized(selected))``).  ``init_seed`` is the section 15 model seed of the factor
    initialisation (``None``: ``context.seed``).  The splitter must be a learned-arm inner design
    (``inner_design.assert_learned_arm_design``: never the per-cell ``InnerCellCalibration``)."""

    def __init__(self, variant: str = "B6", splitter: Any = None, guard: str = "every_split", *,
                 lambdas: Sequence[float] = LAMBDAS,
                 max_sweeps: int = MAX_SWEEPS, tol: float = REL_TOL, init_scale: float = INIT_SCALE,
                 tie_margin: float = TIE_MARGIN, side_categoricals: bool = False, chunk_size: int = TUNING_CHUNK,
                 blas: int | None = BLAS_THREADS, seeds: Sequence[int] | None = None,
                 calibration: str = "cross_fit", init_seed: int | None = None, require_row_units: bool = False):
        if variant not in VARIANT_RANKS:
            raise ValueError(f"variant {variant!r} not in {sorted(VARIANT_RANKS)}")
        if seeds is not None:
            raise ValueError("B6TunedConformal: the tuned configuration depends on the inner seed; use one wrapper "
                             "per seed (context.seed)")
        if calibration not in CALIBRATIONS:
            raise ValueError(f"calibration must be one of {CALIBRATIONS}")
        if splitter is not None:
            ID.assert_learned_arm_design(splitter, f"{variant} B6TunedConformal")
        self.variant, self.ranks, self.lambdas = variant, VARIANT_RANKS[variant], tuple(float(l) for l in lambdas)
        self.max_sweeps, self.tol, self.init_scale = int(max_sweeps), float(tol), float(init_scale)
        self.tie_margin, self.side_categoricals = float(tie_margin), bool(side_categoricals)
        self.chunk_size, self.blas = int(chunk_size), blas
        self.calibration = calibration
        self.init_seed = None if init_seed is None else int(init_seed)
        self.require_row_units = bool(require_row_units)
        super().__init__(B6Factorized(0, self.lambdas[0], name=variant), splitter=splitter, guard=guard, seeds=None)
        self.name = variant
        self.tuning: B6InnerTuning | None = None
        self.selected: Config | None = None
        self.calibration_record: dict[str, Any] = {}

    def clone(self) -> "B6TunedConformal":
        return B6TunedConformal(self.variant, self.splitter, self.guard, lambdas=self.lambdas,
                                max_sweeps=self.max_sweeps, tol=self.tol,
                                init_scale=self.init_scale, tie_margin=self.tie_margin,
                                side_categoricals=self.side_categoricals, chunk_size=self.chunk_size, blas=self.blas,
                                calibration=self.calibration, init_seed=self.init_seed,
                                require_row_units=self.require_row_units)

    def _tune_kw(self) -> dict[str, Any]:
        return dict(max_sweeps=self.max_sweeps, tol=self.tol, init_scale=self.init_scale,
                    side_categoricals=self.side_categoricals, chunk_size=self.chunk_size, blas=self.blas,
                    init_seed=self.init_seed, require_row_units=self.require_row_units)

    def fit(self, train_rows: pd.DataFrame, context: I.FitContext) -> "B6TunedConformal":
        table, mask = I.table_and_mask(train_rows, context)
        inputs_for(table, train_rows if table is not context.table else None)
        return self.fit_table(table, mask, context.for_training(train_rows))

    def fit_table(self, table: I.RowTable, mask: np.ndarray, context: I.FitContext) -> "B6TunedConformal":
        fitted = fit_b6_variants(table, mask, context, self.splitter, variants=(self.variant,), guard=self.guard,
                                 lambdas=self.lambdas, tie_margin=self.tie_margin,
                                 calibration=self.calibration, **self._tune_kw())[self.variant]
        self.__dict__.update({k: v for k, v in fitted.__dict__.items() if k != "splitter"})
        return self

    def fit_from_tuning(self, tuning: B6InnerTuning, table: I.RowTable, mask: np.ndarray,
                        context: I.FitContext) -> "B6TunedConformal":
        """Selection, calibration and refit from a :func:`run_inner_tuning` result of the SAME table, training mask
        and seed (asserted), so B6 and B6r0 share one set of inner fits."""
        want_init = context.seed if self.init_seed is None else self.init_seed
        t = tuning
        if t.table_id != id(table) or t.mask_digest != _mask_digest(mask) or t.seed != context.seed:
            raise AssertionError("B6TunedConformal: the tuning was run on another training set or seed")
        if t.init_seed != want_init:
            raise AssertionError("B6TunedConformal: the tuning used another initialisation seed")
        missing = [c for c in ((k, l) for k in self.ranks for l in self.lambdas) if c not in tuning.predictions]
        if missing:
            raise ValueError(f"B6TunedConformal: the tuning lacks configurations {missing}")
        cfg = tuning.select(self.ranks, self.lambdas, self.tie_margin)
        self.tuning, self.selected = tuning, cfg
        self.fit_seed = context.seed
        if self.calibration == "tuning_residuals":
            self.residuals = tuning.residuals(cfg)
            self.calibration_units = list(tuning.units)
            self.calibration_record = {"method": "tuning_residuals", "status": "calibrated",
                                       "n_calibration": int(len(self.residuals))}
        else:
            # addendum 1 item 2: inner fold j is calibrated by the configuration selected on the other inner folds, with
            # the fits already made (every configuration was fitted on every split)
            plan = ID.cross_fit_plan(tuning.folds)
            res, units, per = [], [], {}
            for j, others in plan.items():
                cj = tuning.select(self.ranks, self.lambdas, self.tie_margin, folds=others)
                r = tuning.residuals(cj, folds=(j,))
                res.append(r)
                units += [u for u, f in zip(tuning.units, tuning.folds) if f == j]
                per[str(j)] = {"config": f"k{cj[0]}_lam{cj[1]:g}", "selected_on_folds": list(others),
                               "n_calibration": int(len(r))}
            self.residuals = np.concatenate(res) if res else np.zeros(0)
            self.calibration_units = units
            self.calibration_record = {"method": "cross_fit", "per_inner_fold": per,
                                       "inner_folds": sorted(set(tuning.folds)),
                                       "status": "calibrated" if res else "not_calibrated_fewer_than_two_inner_folds",
                                       "n_calibration": int(len(self.residuals))}
        if len(self.residuals):
            self.quantiles = {lv: I.conformal_quantile(self.residuals, lv) for lv in I.LEVELS}
        else:
            self.quantiles = {lv: float("nan") for lv in I.LEVELS}
        self.arm = B6Factorized(cfg[0], cfg[1], name=self.variant, max_sweeps=self.max_sweeps, tol=self.tol,
                                init_scale=self.init_scale, side_categoricals=self.side_categoricals, blas=self.blas,
                                seed=self.init_seed)
        self.fitted_arm = self.arm.clone().fit_table(table, mask, context)
        return self

    def selection_record(self) -> dict[str, Any]:
        if self.tuning is None:
            raise RuntimeError("fit first")
        info = self.fitted_arm.info
        return {"variant": self.variant, "selected_rank": self.selected[0], "selected_lambda": self.selected[1],
                "inner_macro_mae": {f"k{k}_lam{l:g}": v for (k, l), v in self.tuning.maes().items()
                                    if k in self.ranks and l in self.lambdas},
                "inner_fold_macro_mae": {f"k{k}_lam{l:g}": {str(int(f)): float(v) for f, v in
                                                            self.tuning.fold_macro_mae((k, l)).items()}
                                         for k in self.ranks for l in self.lambdas},
                "selection_score": "mean over inner folds of the fold's unit-macro MAE (addendum 1 item 2)",
                "n_inner_splits": len(self.tuning.units), "n_inner_folds": len(set(self.tuning.folds)),
                "seed": self.tuning.seed, "init_seed": self.tuning.init_seed,
                "tuning_folds": sorted(set(self.tuning.folds)),
                "units_from_splits": self.tuning.units_from_splits, "calibration": self.calibration_record,
                "outer_refit_sweeps": info["n_sweeps"], "outer_refit_converged": info["converged"]}


#: the only inner mode of a learned arm (addendum 1 item 2: three inner folds on every seed)
INNER_MODES: tuple[str, ...] = ("full",)


def fit_b6_variants(table: I.RowTable, mask: np.ndarray, context: I.FitContext, splitter: Any, *,
                    variants: Sequence[str] = ("B6", "B6r0"), guard: str = "every_split", inner_mode: str | None = None,
                    lambdas: Sequence[float] = LAMBDAS, tie_margin: float = TIE_MARGIN, calibration: str = "cross_fit",
                    init_seed: int | None = None, require_row_units: bool = False,
                    **kw: Any) -> dict[str, B6TunedConformal]:
    """B6 variants of one training set from ONE draw of the inner design and ONE set of inner fits (section 7 order of
    discovery: B6 / B6r0 first).  Every inner fold of the design is tuned (``inner_mode`` accepts ``None`` / ``"full"``
    only: the section 7 compute plan's ``"first"`` reading was retired by addendum 1 item 2); the cross-fitted
    calibration of inner fold ``j`` reuses the fits of the configuration selected on the other inner folds."""
    if inner_mode is not None and inner_mode not in INNER_MODES:
        raise ValueError(f"inner_mode {inner_mode!r}: the learned arms tune on every inner fold under {ID.ADDENDUM} "
                         f"(accepted: None or {INNER_MODES})")
    unknown = sorted(set(kw) - {"max_sweeps", "tol", "init_scale", "side_categoricals", "chunk_size", "blas"})
    if unknown:
        raise TypeError(f"fit_b6_variants: unexpected keyword(s) {unknown}")
    ID.assert_learned_arm_design(splitter, "B6")
    lambdas = tuple(float(l) for l in lambdas)
    splits = list(splitter.splits(table, mask, context))
    if not splits:
        raise ValueError("the inner design produced no split")
    fixed = FixedSplits(splits, getattr(splitter, "name", type(splitter).__name__))
    ranks = tuple(sorted({k for v in variants for k in VARIANT_RANKS[v]}))
    tuning = run_inner_tuning(table, mask, context, fixed, guard=guard, ranks=ranks, lambdas=lambdas,
                              init_seed=init_seed, require_row_units=require_row_units, **kw)
    wrappers = {v: B6TunedConformal(v, splitter, guard, lambdas=lambdas, tie_margin=tie_margin,
                                    calibration=calibration, init_seed=init_seed, require_row_units=require_row_units,
                                    **kw) for v in variants}
    return {v: w.fit_from_tuning(tuning, table, mask, context) for v, w in wrappers.items()}


def fit_b6_and_b6r0(table: I.RowTable, mask: np.ndarray, context: I.FitContext, splitter: Any, *,
                    guard: str = "every_split", **kw: Any) -> dict[str, B6TunedConformal]:
    """B6 and B6r0 of one training set from ONE set of inner fits (section 7 order of discovery: B6 / B6r0 first)."""
    return fit_b6_variants(table, mask, context, splitter, variants=("B6", "B6r0"), guard=guard, **kw)


# --------------------------------------------------------------------------------------------- #
# section 3.1 / 3.2 registered checks and section 7 compute plan item 6
# --------------------------------------------------------------------------------------------- #

def batched_exact_check(exact_unit_mae: pd.Series, batched_unit_mae: pd.Series, *,
                        threshold: float = BATCHED_CHECK_THRESHOLD) -> dict[str, Any]:
    """B6 macro MAE under the batched (or V1 ten-fold) folds vs the exact folds, on identical scored units: passes
    when ``|delta| < threshold`` (0.01 log D)."""
    ex, ba = exact_unit_mae.astype(float), batched_unit_mae.astype(float)
    if set(ex.index) != set(ba.index) or ex.index.has_duplicates or ba.index.has_duplicates:
        raise ValueError("batched_exact_check: the two designs must score the same units once each")
    if not (np.isfinite(ex).all() and np.isfinite(ba).all()):
        raise ValueError("batched_exact_check: non-finite unit MAE")
    e, b = float(ex.mean()), float(ba.loc[ex.index].mean())
    return {"n_units": int(len(ex)), "exact_macro_mae": e, "batched_macro_mae": b, "delta": b - e,
            "abs_delta": abs(b - e), "threshold": float(threshold), "passed": bool(abs(b - e) < threshold)}


def batched_check_decision(first: Mapping[str, Any], second: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Section 7 compute-plan item 6: the first check passes -> registered batches; fails -> re-colour with at most
    4 cells per batch (same seeds) and repeat once; still fails -> every heavy-arm V5 result is labelled
    'batched (check failed)' and S1 is UNDECIDED."""
    if first["passed"]:
        if second is not None:
            raise ValueError("no second check after a passed first check")
        return {"status": "passed", "scheme": "batched", "label": "batched", "s1_forced_undecided": False,
                "next_action": None}
    if second is None:
        return {"status": "recolour", "scheme": None, "label": None, "s1_forced_undecided": False,
                "next_action": f"re-colour with max_cells_per_batch={RECOLOUR_MAX_CELLS} (same seeds) and repeat once"}
    if second["passed"]:
        return {"status": "passed_after_recolour", "scheme": f"batched_max{RECOLOUR_MAX_CELLS}",
                "label": f"batched_max{RECOLOUR_MAX_CELLS}", "s1_forced_undecided": False, "next_action": None}
    return {"status": "failed", "scheme": f"batched_max{RECOLOUR_MAX_CELLS}", "label": "batched (check failed)",
            "s1_forced_undecided": True, "next_action": None}


assert INIT_STREAM_TAG not in FI.STREAM_TAGS.values()
