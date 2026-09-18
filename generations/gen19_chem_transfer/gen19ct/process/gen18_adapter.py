"""``gen18_adapter.py`` -- a Gen19 D source for gen18's cascade (brief sections 1.6, 17 and 26; pre-registration
sections 13 and 14).

gen18 is READ-ONLY.  Nothing here edits ``gen18proc``; the adapter *subclasses* ``gen18proc.dmodel._ModelBase`` so
that the stage solver uses its vectorised ``core()`` exactly as it uses gen18's own models (``cascade._Problem.
d_arrays_all``), and it implements the ``DModel`` protocol of ``gen18proc.types`` (``ligand``, ``mechanism``, ``metals``,
``q``, ``p``, ``z``, ``domain``, ``provenance``, ``evaluate``).

The D source is a **tabulated Gen19 prediction record** per (metal, system, conditions): ``mean_logD``, ``std_logD``,
``lower_95`` / ``upper_95``, ``domain_status``, ``support_score`` and ``nearest_support`` on a rectangular grid of
``(log10 [HNO3] / mol L^-1, log10 formal [ligand] / mol L^-1)`` (:class:`PredictionTable`).  The deployment step that
fits the deployed predictor and predicts on the grid is not this module; :func:`prediction_grid` /
:func:`write_prediction_request` say exactly which conditions it must predict.

The tabulated predictor is a lookup, so the partials are **analytic**: bilinear interpolation in (log acid, log
ligand) gives ``d log10 D / d ln h = (d table / d log10 h) / ln 10`` inside a cell, and the loading correction is
gen18's ideal depletion term ``n_prior * log10(L_f / capacity)`` (the same term ``NearestConditionD`` uses, applied to a
tracer-limit table), so ``d log10 D / d ln L_f = n_prior / ln 10``.  The anion and complexant partials are zero (the table
has no anion axis; the complexant algebra is gen18's ``AqueousComplexantWrapper`` when a system has one).  Outside the
table the value is clamped to the nearest edge (zero acid slope) and the evaluation is recorded as ``OUTSIDE_TABLE``,
which the ranking layer treats like ``UNSUPPORTED`` (no prediction exists there).  No finite-difference partials are
needed because the D source is always a table; ``tests/test_process.py`` checks the analytic partials against central
differences the way gen18's ``tests/test_dmodel.py`` does.

**Provenance.**  gen18's ``ProvStatus`` has no "model-predicted" value and its validator rejects gen15 tokens.  Gen19
predictions therefore use the ASSUMED-with-range convention gen18 already accepts (``types.ProvStatus.ASSUMED`` with
``assumed_label = ASSUMED_PLACEHOLDER`` and a declared ``range``, validator rules V4 / V9): ``range`` is the table-wide
95 % interval of log D, ``source.kind = "model"`` (a value of ``types.SOURCE_KINDS``) with a locator naming the arm,
``model_id`` the deployed arm's configuration, ``fit_manifest_sha256`` the digest of the prediction file, and the note
starts with :data:`MODEL_DERIVED_NOTE`.  Every value is flagged model-derived on the Gen19 side
(:meth:`PredictionTable.record` carries ``model_derived = True``).

**Support gate.**  A table containing any ``UNSUPPORTED`` cell refuses to instantiate unless ``allow_unsupported=True``
(used only for the F5(ii) barred-vs-allowed comparison, section 10).  The registered path trims the table to its
:meth:`PredictionTable.supported_box` first.  Every evaluation records the domain statuses of the grid cells it
interpolated between (:meth:`Gen19DModel.statuses_used`), so the ranking can apply section 14's support rank per recipe.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gen19ct import paths

paths.add_gen18_to_path()

from gen18proc.dmodel import (  # noqa: E402  (gen18 on sys.path above; read-only import)
    ASSUMPTIONS,
    LN10,
    SystemModel,
    _ModelBase,
    _stoichiometry,
)
from gen18proc.domain import coerce_domain  # noqa: E402
from gen18proc.types import (  # noqa: E402
    ASSUMED_LABEL,
    ApplicabilityDomain,
    Flag,
    LigandSpec,
    Mechanism,
    ModelBuildError,
    PhaseBehaviour,
    Provenance,
    ProvStatus,
    Source,
    SystemEntry,
)

__all__ = [
    "DOMAIN_STATUSES", "OUTSIDE_TABLE", "MODEL_DERIVED_NOTE", "PREDICTION_COLUMNS", "REQUIRED_PREDICTION_COLUMNS",
    "N_MEMBERS_REGISTERED", "MEMBER_COLUMN_PREFIX", "CONFORMAL_Q95_COLUMN", "MEMBER_COLUMNS",
    "PROVENANCE_CONVENTION", "UnsupportedPredictionError", "PredictionTable", "Gen19DModel", "gen19_provenance",
    "build_gen19_system", "prediction_grid", "write_prediction_request", "read_prediction_table",
]

#: section 13 vocabulary (``gen19ct.evaluation.support.DOMAIN_STATUS_ORDER``, copied so that this module stays light;
#: ``tests/test_process.py`` asserts the two agree)
DOMAIN_STATUSES: tuple[str, ...] = ("UNSUPPORTED", "FAMILY_EXTRAPOLATION", "CROSS_METAL_LIGAND_TRANSFER",
                                    "CROSS_LIGAND_TRANSFER", "CROSS_METAL_TRANSFER", "CONDITION_EXTRAPOLATION",
                                    "IN_DOMAIN", "INTERPOLATION")
#: pseudo-status recorded when a stage evaluates the table outside its grid (clamped): no prediction exists there
OUTSIDE_TABLE = "OUTSIDE_TABLE"
MODEL_DERIVED_NOTE = "MODEL_DERIVED: Gen19 chemistry-transfer prediction (brief section 1.3 record), not a measurement"
#: distance (log10 units) beyond the grid edge above which an evaluation counts as OUTSIDE_TABLE
TABLE_EDGE_TOL = 1e-9

PREDICTION_COLUMNS: tuple[str, ...] = (
    "metal", "system_id", "log_acid", "log_ligand", "mean_logD", "std_logD", "lower_95", "upper_95", "domain_status",
    "support_score", "nearest_support", "arm")
#: section 14 "one draw is one ensemble member plus a residual drawn from the calibrated conformal scale": the record
#: carries the M7 members' means ``member_logD_0 .. member_logD_{K-1}`` (K = ``models.ladder.N_MEMBERS`` = 5; copied so
#: that this module stays light, asserted equal in ``tests/test_process.py``) and the calibrated normalised-conformal
#: 95 % quantile multiplier ``conformal_q95`` (the M7 record's ``conformal_q95``: interval = mean +- q95 * std_logD).
#: Without them the Monte Carlo can only fall back to the truncated Gaussian, which a registered run refuses
#: (task X finding V-07)
N_MEMBERS_REGISTERED = 5
MEMBER_COLUMN_PREFIX = "member_logD_"
CONFORMAL_Q95_COLUMN = "conformal_q95"
MEMBER_COLUMNS: tuple[str, ...] = tuple(f"{MEMBER_COLUMN_PREFIX}{k}" for k in range(N_MEMBERS_REGISTERED))
#: the member means must average to mean_logD within this tolerance (the M7 ensemble mean is the member mean)
MEMBER_MEAN_TOL = 1e-6
REQUIRED_PREDICTION_COLUMNS: tuple[str, ...] = ("metal", "log_acid", "log_ligand", "mean_logD", "std_logD",
                                                "lower_95", "upper_95", "domain_status")

PROVENANCE_CONVENTION = (
    "gen18 ProvStatus has no model-predicted value: Gen19 predictions enter gen18 as ProvStatus.ASSUMED with "
    "assumed_label ASSUMED_PLACEHOLDER and range = the table-wide [min lower_95, max upper_95] of log D (validator V4 / "
    "V9 shape), source.kind 'model' (types.SOURCE_KINDS) naming the arm, model_id = the deployed configuration, "
    "fit_manifest_sha256 = SHA-256 of the prediction file; the note starts with MODEL_DERIVED and every Gen19-side "
    "record carries model_derived = True")


class UnsupportedPredictionError(ValueError):
    """Raised when a D source would be built from ``UNSUPPORTED`` predictions without ``allow_unsupported=True``."""


# --------------------------------------------------------------------------------------------- #
# the prediction table
# --------------------------------------------------------------------------------------------- #

def _sorted_unique(values: np.ndarray) -> np.ndarray:
    return np.unique(np.asarray(values, dtype=float))


@dataclass(frozen=True)
class PredictionTable:
    """Gen19 prediction records on a rectangular (log acid, log ligand) grid, one layer per metal.

    Arrays are shaped ``(n_metals, n_acid, n_ligand)``; ``status_codes`` index :attr:`status_labels`.  Build with
    :meth:`from_frame` (validates the grid) and use :meth:`with_values` for a Monte Carlo draw (a table whose
    ``mean`` is the drawn log D; every other field is kept)."""

    metals: tuple[str, ...]
    log_acid: np.ndarray                  # (n_acid,) sorted
    log_ligand: np.ndarray                # (n_ligand,) sorted
    mean: np.ndarray                      # (M, A, L)
    std: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    status_codes: np.ndarray              # (M, A, L) int
    status_labels: tuple[str, ...]
    support_score: np.ndarray             # (M, A, L) float (NaN when absent)
    nearest_support: np.ndarray           # (M, A, L) object (str or None)
    system_id: str | None = None
    arm: str | None = None
    source: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    members: np.ndarray | None = None     # (K, M, A, L) member means of the M7 ensemble, when the record carries them
    conformal_q95: np.ndarray | None = None   # (M, A, L) calibrated normalised-conformal 95 % quantile multiplier

    @property
    def has_members(self) -> bool:
        """Whether the section 14 draw (member + conformal residual) is possible on this table."""
        return self.members is not None and self.conformal_q95 is not None

    @property
    def n_members(self) -> int:
        return 0 if self.members is None else int(self.members.shape[0])

    # ---- construction --------------------------------------------------------------------- #
    @classmethod
    def from_frame(cls, frame: pd.DataFrame, *, source: str | None = None) -> "PredictionTable":
        df = frame.copy()
        missing = [c for c in REQUIRED_PREDICTION_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"prediction table lacks columns {missing}; required {REQUIRED_PREDICTION_COLUMNS}")
        if len(df) == 0:
            raise ValueError("prediction table is empty")
        for c in ("log_acid", "log_ligand", "mean_logD", "std_logD", "lower_95", "upper_95"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[["log_acid", "log_ligand", "mean_logD"]].isna().any().any():
            raise ValueError("prediction table: log_acid, log_ligand and mean_logD must be finite numbers")
        if (df["std_logD"] < 0).any():
            raise ValueError("prediction table: std_logD must be >= 0")
        df["metal"] = df["metal"].astype(str)
        df["domain_status"] = df["domain_status"].astype(str)
        bad = sorted(set(df["domain_status"]) - set(DOMAIN_STATUSES))
        if bad:
            raise ValueError(f"prediction table: domain_status values outside the section 13 vocabulary: {bad}")
        la, ll = _sorted_unique(df["log_acid"].to_numpy()), _sorted_unique(df["log_ligand"].to_numpy())
        # round-trip keys so that float noise does not split a grid line
        la_idx = {float(v): i for i, v in enumerate(la)}
        ll_idx = {float(v): j for j, v in enumerate(ll)}
        metals = tuple(sorted(set(df["metal"])))
        M, A, L = len(metals), len(la), len(ll)
        shape = (M, A, L)
        mean = np.full(shape, np.nan)
        std, lower, upper = np.full(shape, np.nan), np.full(shape, np.nan), np.full(shape, np.nan)
        codes = np.full(shape, -1, dtype=int)
        support = np.full(shape, np.nan)
        nearest = np.full(shape, None, dtype=object)
        labels = list(DOMAIN_STATUSES) + [OUTSIDE_TABLE]
        code_of = {s: k for k, s in enumerate(labels)}
        seen = np.zeros(shape, dtype=int)
        m_idx = {m: k for k, m in enumerate(metals)}
        has_support = "support_score" in df.columns
        has_nearest = "nearest_support" in df.columns
        mcols = [c for c in df.columns if str(c).startswith(MEMBER_COLUMN_PREFIX)]
        has_members = bool(mcols) and CONFORMAL_Q95_COLUMN in df.columns and not (
            df[mcols].isna().all().all() and df[CONFORMAL_Q95_COLUMN].isna().all())
        members: np.ndarray | None = None
        q95: np.ndarray | None = None
        if has_members:
            ks = sorted(int(c[len(MEMBER_COLUMN_PREFIX):]) for c in mcols)
            if ks != list(range(len(ks))):
                raise ValueError(f"prediction table: member columns must be {MEMBER_COLUMN_PREFIX}0..{MEMBER_COLUMN_PREFIX}K-1, got {mcols}")
            mcols = [f"{MEMBER_COLUMN_PREFIX}{k}" for k in ks]
            for c in mcols + [CONFORMAL_Q95_COLUMN]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            if df[mcols + [CONFORMAL_Q95_COLUMN]].isna().any().any():
                raise ValueError("prediction table: member_logD_k and conformal_q95 must be finite on every row when present")
            if (df[CONFORMAL_Q95_COLUMN] <= 0).any():
                raise ValueError("prediction table: conformal_q95 must be > 0")
            members = np.full((len(ks),) + shape, np.nan)
            q95 = np.full(shape, np.nan)
        for row in df.itertuples(index=False):
            k, i, j = m_idx[row.metal], la_idx[float(row.log_acid)], ll_idx[float(row.log_ligand)]
            seen[k, i, j] += 1
            mean[k, i, j] = float(row.mean_logD)
            if has_members:
                for kk, c in enumerate(mcols):
                    members[kk, k, i, j] = float(getattr(row, c))
                q95[k, i, j] = float(getattr(row, CONFORMAL_Q95_COLUMN))
            std[k, i, j] = float(row.std_logD) if np.isfinite(row.std_logD) else 0.0
            lower[k, i, j] = float(row.lower_95)
            upper[k, i, j] = float(row.upper_95)
            codes[k, i, j] = code_of[row.domain_status]
            if has_support:
                v = pd.to_numeric(pd.Series([getattr(row, "support_score")]), errors="coerce").iloc[0]
                support[k, i, j] = float(v) if pd.notna(v) else np.nan
            if has_nearest:
                v = getattr(row, "nearest_support")
                nearest[k, i, j] = None if (v is None or (isinstance(v, float) and math.isnan(v))) else str(v)
        if (seen > 1).any():
            raise ValueError("prediction table: duplicate (metal, log_acid, log_ligand) rows")
        if (seen == 0).any():
            n_missing = int((seen == 0).sum())
            raise ValueError(f"prediction table is not a rectangular grid: {n_missing} of {M * A * L} "
                             f"(metal, log_acid, log_ligand) cells are missing")
        if has_members:
            gap = np.abs(members.mean(axis=0) - mean).max()
            if not gap <= MEMBER_MEAN_TOL:
                raise ValueError(f"prediction table: the member means average to mean_logD only within {gap:.3g} "
                                 f"(> {MEMBER_MEAN_TOL}); the M7 record's mean_logD is the member mean")
        # an interval that does not bracket the mean is repaired to the untruncated Gaussian (recorded)
        bad_iv = ~(np.isfinite(lower) & np.isfinite(upper) & (lower <= mean) & (upper >= mean))
        lower = np.where(bad_iv, np.nan, lower)
        upper = np.where(bad_iv, np.nan, upper)
        system_id = None
        if "system_id" in df.columns:
            sids = sorted(set(df["system_id"].astype(str)))
            if len(sids) > 1:
                raise ValueError(f"prediction table mixes systems {sids}; one table per system")
            system_id = sids[0]
        arm = None
        if "arm" in df.columns:
            arms = sorted(set(df["arm"].astype(str)))
            if len(arms) > 1:
                raise ValueError(f"prediction table mixes arms {arms}; one deployed predictor per table")
            arm = arms[0]
        return cls(metals=metals, log_acid=la, log_ligand=ll, mean=mean, std=std, lower=lower, upper=upper,
                   status_codes=codes, status_labels=tuple(labels), support_score=support, nearest_support=nearest,
                   system_id=system_id, arm=arm, source=source,
                   meta={"n_intervals_repaired": int(bad_iv.sum()), "has_members": bool(has_members),
                         "n_members": 0 if members is None else int(members.shape[0])},
                   members=members, conformal_q95=q95)

    @classmethod
    def from_csv(cls, path: str | Path) -> "PredictionTable":
        p = Path(path)
        return cls.from_frame(pd.read_csv(p), source=str(p))

    # ---- views ------------------------------------------------------------------------------ #
    @property
    def shape(self) -> tuple[int, int, int]:
        return self.mean.shape

    def box(self) -> dict[str, tuple[float, float]]:
        """``{"log_acid": (lo, hi), "log_ligand": (lo, hi)}`` of the grid."""
        return {"log_acid": (float(self.log_acid[0]), float(self.log_acid[-1])),
                "log_ligand": (float(self.log_ligand[0]), float(self.log_ligand[-1]))}

    def statuses(self) -> dict[str, int]:
        """Count of grid cells per domain status (all metals)."""
        out: dict[str, int] = {}
        for k, lab in enumerate(self.status_labels):
            n = int((self.status_codes == k).sum())
            if n:
                out[lab] = n
        return out

    def status_of(self, metal: str, i: int, j: int) -> str:
        return self.status_labels[int(self.status_codes[self.metals.index(metal), i, j])]

    @property
    def has_unsupported(self) -> bool:
        return bool((self.status_codes == self.status_labels.index("UNSUPPORTED")).any())

    def worst_status(self) -> str:
        """The lowest section 13 status present (``DOMAIN_STATUSES`` order, UNSUPPORTED first)."""
        present = self.statuses()
        for s in DOMAIN_STATUSES:
            if s in present:
                return s
        return DOMAIN_STATUSES[-1]

    def record(self, metal: str, i: int, j: int) -> dict[str, Any]:
        """The Gen19 prediction record of one grid cell (brief section 1.3 fields plus the section 13 labels), flagged
        ``model_derived``."""
        k = self.metals.index(metal)
        rec = {"metal": metal, "system_id": self.system_id, "arm": self.arm,
               "log_acid": float(self.log_acid[i]), "log_ligand": float(self.log_ligand[j]),
               "mean_logD": float(self.mean[k, i, j]), "std_logD": float(self.std[k, i, j]),
               "lower_95": (None if not np.isfinite(self.lower[k, i, j]) else float(self.lower[k, i, j])),
               "upper_95": (None if not np.isfinite(self.upper[k, i, j]) else float(self.upper[k, i, j])),
               "domain_status": self.status_labels[int(self.status_codes[k, i, j])],
               "support_score": (None if not np.isfinite(self.support_score[k, i, j])
                                 else float(self.support_score[k, i, j])),
               "nearest_support": self.nearest_support[k, i, j], "model_derived": True}
        if self.has_members:
            rec["member_logD"] = [float(v) for v in self.members[:, k, i, j]]
            rec[CONFORMAL_Q95_COLUMN] = float(self.conformal_q95[k, i, j])
        return rec

    def to_frame(self, values: np.ndarray | None = None) -> pd.DataFrame:
        """Long frame in the :data:`PREDICTION_COLUMNS` layout (``values`` replaces ``mean_logD`` when given)."""
        v = self.mean if values is None else np.asarray(values, dtype=float)
        rows = []
        for k, m in enumerate(self.metals):
            for i, la in enumerate(self.log_acid):
                for j, ll in enumerate(self.log_ligand):
                    row = {"metal": m, "system_id": self.system_id, "log_acid": float(la), "log_ligand": float(ll),
                           "mean_logD": float(v[k, i, j]), "std_logD": float(self.std[k, i, j]),
                           "lower_95": float(self.lower[k, i, j]), "upper_95": float(self.upper[k, i, j]),
                           "domain_status": self.status_labels[int(self.status_codes[k, i, j])],
                           "support_score": float(self.support_score[k, i, j]),
                           "nearest_support": self.nearest_support[k, i, j], "arm": self.arm}
                    if self.has_members:
                        for kk in range(self.n_members):
                            row[f"{MEMBER_COLUMN_PREFIX}{kk}"] = float(self.members[kk, k, i, j])
                        row[CONFORMAL_Q95_COLUMN] = float(self.conformal_q95[k, i, j])
                    rows.append(row)
        cols = list(PREDICTION_COLUMNS) + ([f"{MEMBER_COLUMN_PREFIX}{kk}" for kk in range(self.n_members)]
                                           + [CONFORMAL_Q95_COLUMN] if self.has_members else [])
        return pd.DataFrame(rows, columns=cols)

    # ---- derived tables --------------------------------------------------------------------- #
    def with_values(self, values: np.ndarray) -> "PredictionTable":
        """The same table with ``mean`` replaced by ``values`` (a Monte Carlo draw of log D)."""
        v = np.asarray(values, dtype=float)
        if v.shape != self.mean.shape:
            raise ValueError(f"values shape {v.shape} != table shape {self.mean.shape}")
        return PredictionTable(self.metals, self.log_acid, self.log_ligand, v, self.std, self.lower, self.upper,
                               self.status_codes, self.status_labels, self.support_score, self.nearest_support,
                               self.system_id, self.arm, self.source, dict(self.meta),
                               members=self.members, conformal_q95=self.conformal_q95)

    def restrict(self, *, log_acid: tuple[float, float] | None = None,
                 log_ligand: tuple[float, float] | None = None) -> "PredictionTable":
        """The sub-grid inside the closed intervals (at least one grid line must remain on each axis)."""
        ia = np.arange(len(self.log_acid))
        il = np.arange(len(self.log_ligand))
        if log_acid is not None:
            ia = ia[(self.log_acid >= log_acid[0] - 1e-12) & (self.log_acid <= log_acid[1] + 1e-12)]
        if log_ligand is not None:
            il = il[(self.log_ligand >= log_ligand[0] - 1e-12) & (self.log_ligand <= log_ligand[1] + 1e-12)]
        if ia.size == 0 or il.size == 0:
            raise ValueError("restrict: no grid line left on an axis")
        sl = np.ix_(np.arange(len(self.metals)), ia, il)
        msl = None if self.members is None else self.members[np.ix_(np.arange(self.n_members), np.arange(len(self.metals)), ia, il)]
        return PredictionTable(self.metals, self.log_acid[ia], self.log_ligand[il], self.mean[sl], self.std[sl],
                               self.lower[sl], self.upper[sl], self.status_codes[sl], self.status_labels,
                               self.support_score[sl], self.nearest_support[sl], self.system_id, self.arm, self.source,
                               dict(self.meta), members=msl,
                               conformal_q95=None if self.conformal_q95 is None else self.conformal_q95[sl])

    def supported_box(self) -> dict[str, Any]:
        """The largest rectangular sub-grid (most cells) free of ``UNSUPPORTED`` cells in every metal layer, by an
        exhaustive search over the grid-line intervals (a 2-D prefix sum makes each rectangle O(1); the grids are
        small).  Ties: the larger acid span, then the lower acid start, then the lower ligand start.  Returns the
        interval per axis, the number of grid lines dropped per axis and whether anything is left."""
        unsupported = (self.status_codes == self.status_labels.index("UNSUPPORTED")).any(axis=0).astype(int)  # (A, L)
        A, L = unsupported.shape
        ps = np.zeros((A + 1, L + 1), dtype=int)
        ps[1:, 1:] = unsupported.cumsum(axis=0).cumsum(axis=1)
        best: tuple[int, int, int, int, int, int] | None = None     # (-area, -acid_span, a0, l0, a1, l1)
        for a0 in range(A):
            for a1 in range(a0, A):
                for l0 in range(L):
                    for l1 in range(l0, L):
                        n_bad = ps[a1 + 1, l1 + 1] - ps[a0, l1 + 1] - ps[a1 + 1, l0] + ps[a0, l0]
                        if n_bad:
                            continue
                        key = (-(a1 - a0 + 1) * (l1 - l0 + 1), -(a1 - a0 + 1), a0, l0, a1, l1)
                        if best is None or key < best:
                            best = key
        if best is None:
            return {"ok": False, "removed_lines": {"log_acid": A, "log_ligand": L}, "log_acid": None, "log_ligand": None}
        _, _, a0, l0, a1, l1 = best
        return {"ok": True, "removed_lines": {"log_acid": A - (a1 - a0 + 1), "log_ligand": L - (l1 - l0 + 1)},
                "log_acid": (float(self.log_acid[a0]), float(self.log_acid[a1])),
                "log_ligand": (float(self.log_ligand[l0]), float(self.log_ligand[l1]))}

    def supported(self) -> "PredictionTable":
        """:meth:`restrict` to :meth:`supported_box` (``ValueError`` when nothing supported remains)."""
        sb = self.supported_box()
        if not sb["ok"]:
            raise UnsupportedPredictionError("every grid line carries an UNSUPPORTED cell; nothing supported remains")
        return self.restrict(log_acid=sb["log_acid"], log_ligand=sb["log_ligand"])


def read_prediction_table(path: str | Path) -> PredictionTable:
    return PredictionTable.from_csv(path)


# --------------------------------------------------------------------------------------------- #
# bilinear interpolation with analytic gradient
# --------------------------------------------------------------------------------------------- #

def _locate(grid: np.ndarray, x: float) -> tuple[int, int, float, float, float]:
    """``(i0, i1, t, x_clamped, outside)`` on a sorted grid: the bracketing indices, the fractional position and how
    far ``x`` lay beyond the grid (0 inside)."""
    n = grid.size
    lo, hi = float(grid[0]), float(grid[-1])
    outside = max(lo - x, x - hi, 0.0)
    xc = min(max(x, lo), hi)
    if n == 1:
        return 0, 0, 0.0, xc, outside
    i1 = int(np.searchsorted(grid, xc, side="right"))
    i1 = min(max(i1, 1), n - 1)
    i0 = i1 - 1
    span = float(grid[i1] - grid[i0])
    t = (xc - float(grid[i0])) / span if span > 0 else 0.0
    return i0, i1, t, xc, outside


def bilinear(values: np.ndarray, log_acid: np.ndarray, log_ligand: np.ndarray, x: float, y: float,
             ) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int], float]:
    """Interpolated value per metal, its derivative with respect to ``x`` (log10 acid; 0 when ``x`` is clamped
    outside the grid), the corner indices ``(i0, i1, j0, j1)`` and the clamping distance (log10 units)."""
    i0, i1, tx, xc, out_x = _locate(log_acid, x)
    j0, j1, ty, yc, out_y = _locate(log_ligand, y)
    v00, v01, v10, v11 = values[:, i0, j0], values[:, i0, j1], values[:, i1, j0], values[:, i1, j1]
    v0 = v00 * (1.0 - ty) + v01 * ty                     # along ligand at acid i0
    v1 = v10 * (1.0 - ty) + v11 * ty
    val = v0 * (1.0 - tx) + v1 * tx
    span = float(log_acid[i1] - log_acid[i0])
    if span > 0 and out_x <= TABLE_EDGE_TOL:
        dval_dx = (v1 - v0) / span
    else:
        dval_dx = np.zeros_like(val)
    return val, dval_dx, (i0, i1, j0, j1), max(out_x, out_y)


# --------------------------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------------------------- #

def gen19_provenance(table: PredictionTable, *, model_id: str | None = None,
                     fit_manifest_sha256: str | None = None, note: str = "") -> Provenance:
    """The ASSUMED-with-range provenance of a Gen19 D source (module docstring, :data:`PROVENANCE_CONVENTION`)."""
    lo = table.lower[np.isfinite(table.lower)]
    hi = table.upper[np.isfinite(table.upper)]
    if lo.size and hi.size:
        rng = (float(lo.min()), float(hi.max()))
    else:
        rng = (float((table.mean - 1.96 * table.std).min()), float((table.mean + 1.96 * table.std).max()))
    arm = table.arm or "deployed predictor"
    src = Source(kind="model", locator=f"Gen19 {arm}: tabulated prediction record (log acid x log ligand grid, "
                                       f"{table.shape[1]} x {table.shape[2]} cells, {len(table.metals)} metals)"
                 + (f"; file {Path(table.source).name}" if table.source else ""))
    text = MODEL_DERIVED_NOTE + (f"; worst domain status {table.worst_status()}; statuses {table.statuses()}")
    if note:
        text += " | " + note
    return Provenance(ProvStatus.ASSUMED, src, rng, ASSUMED_LABEL, model_id=model_id or arm,
                      fit_manifest_sha256=fit_manifest_sha256, note=text)


# --------------------------------------------------------------------------------------------- #
# the D model
# --------------------------------------------------------------------------------------------- #

class Gen19DModel(_ModelBase):
    """gen18 ``DModel`` fed by a :class:`PredictionTable` (module docstring).

    ``log D_i(state) = T_i(log10 h, log10 (ligand_scale * L_T)) + n_prior * log10(L_f / capacity)`` with ``T_i`` the
    bilinear interpolant of the table's ``mean`` (or of ``values`` when given: one Monte Carlo draw).  Partials:
    ``dlogd_dlnL = n_prior / ln 10``; ``dlogd_dlnh = dT/dlog10 h / ln 10`` (0 when clamped outside the grid);
    anion and complexant partials 0.  ``q`` defaults to ``n_prior`` (solvating: ligands per metal), ``p`` / ``z`` to
    gen18's ideal stoichiometry of the mechanism (``dmodel._stoichiometry``), each overridable.

    ``allow_unsupported=False`` refuses a table with any UNSUPPORTED cell (:class:`UnsupportedPredictionError`).
    :meth:`statuses_used` returns the domain statuses of every grid cell interpolated between since the last
    :meth:`reset_usage` (plus ``OUTSIDE_TABLE`` when a stage lay beyond the grid), the ranking layer's per-recipe
    support input; ``ood_distance["gen19_table"]`` is the clamping distance.  Flags carried: ``OA_ASSUMED`` (the
    corpus semantics the predictor was trained on) plus the caller's ``base_flags``.
    """

    def __init__(self, table: PredictionTable, ligand: LigandSpec | str, *, mechanism: Mechanism = Mechanism.SOLVATING,
                 n_prior: float = 3.0, domain: ApplicabilityDomain | None = None, phase: PhaseBehaviour | None = None,
                 base_flags: Iterable[Flag] = (), allow_unsupported: bool = False, values: np.ndarray | None = None,
                 q: Any = None, p: Any = None, z: Any = None, model_id: str | None = None,
                 fit_manifest_sha256: str | None = None, provenance_note: str = ""):
        if table.has_unsupported and not allow_unsupported:
            raise UnsupportedPredictionError(
                f"the prediction table carries {table.statuses().get('UNSUPPORTED', 0)} UNSUPPORTED cell(s); a Gen19 D "
                "source is not built from UNSUPPORTED predictions (section 14 support rank 0). Trim the table with "
                "PredictionTable.supported() or pass allow_unsupported=True (F5(ii) barred-vs-allowed audit only)")
        name = ligand.name if isinstance(ligand, LigandSpec) else str(ligand)
        self.table = table
        self.allow_unsupported = bool(allow_unsupported)
        self.n_prior = float(n_prior)
        mech = Mechanism(mechanism)
        q_d, p_d, z_d = _stoichiometry(ligand, mech, self.n_prior)
        scale = 2.0 if isinstance(ligand, LigandSpec) and ligand.aggregation == "dimer" else 1.0
        prov = gen19_provenance(table, model_id=model_id, fit_manifest_sha256=fit_manifest_sha256, note=provenance_note)
        flags = set(base_flags) | {Flag.OA_ASSUMED}
        self._init_common(name, mech, table.metals, self.n_prior if q is None else q, p_d if p is None else p,
                          z_d if z is None else z, domain, prov, phase, flags, 0.0, scale)
        v = table.mean if values is None else np.asarray(values, dtype=float)
        if v.shape != table.mean.shape:
            raise ValueError(f"values shape {v.shape} != table shape {table.mean.shape}")
        self._values = v
        self._s_l = np.full(len(self.metals), self.n_prior / LN10)
        self._cache_key: tuple[float, float] | None = None
        self._base = np.zeros(len(self.metals))
        self._dbase_dlogh = np.zeros(len(self.metals))
        self._last_outside = 0.0
        self._last_codes: frozenset[int] = frozenset()
        self._usage_codes: set[int] = set()
        self._outside_code = table.status_labels.index(OUTSIDE_TABLE)
        self.n_evaluations = 0

    # ---- usage tracking (section 14 support rank input) --------------------------------------- #
    def reset_usage(self) -> None:
        self._usage_codes = set()
        self.n_evaluations = 0

    def statuses_used(self) -> frozenset[str]:
        return frozenset(self.table.status_labels[c] for c in self._usage_codes)

    # ---- the core ---------------------------------------------------------------------------- #
    def _lookup(self, la: float, lt: float) -> None:
        val, dval, (i0, i1, j0, j1), outside = bilinear(self._values, self.table.log_acid, self.table.log_ligand, la, lt)
        self._base, self._dbase_dlogh, self._last_outside = val, dval, outside
        codes = set(np.unique(self.table.status_codes[:, i0:i1 + 1, j0:j1 + 1]).tolist())
        if outside > TABLE_EDGE_TOL:
            codes.add(self._outside_code)
        self._last_codes = frozenset(codes)

    def core(self, L, h, nu, c, ligand_total, capacity=None):
        la = math.log10(h) if h > 0 else -math.inf
        lt = math.log10(ligand_total * self.ligand_scale) if ligand_total > 0 else -math.inf
        key = (la, lt)
        if key != self._cache_key:
            self._lookup(la, lt)
            self._cache_key = key
        # usage is recorded on EVERY call (a cache hit must still count for the candidate being evaluated)
        self._usage_codes.update(self._last_codes)
        self.n_evaluations += 1
        cap = ligand_total if capacity is None else capacity
        l_free = math.log10(L) if L > 0 else -math.inf
        l_cap = math.log10(cap) if cap > 0 else -math.inf
        shift = self.n_prior * (l_free - l_cap)
        return self._base + shift, self._s_l, self._dbase_dlogh / LN10, self._zeros, self._zeros

    def _extra_distance(self, state, metal: str | None = None) -> dict[str, float]:
        return {"gen19_table": float(self._last_outside)}

    def cell_records(self) -> list[dict[str, Any]]:
        """Every grid cell's Gen19 record (for the process outputs' provenance table)."""
        out = []
        for m in self.metals:
            for i in range(len(self.table.log_acid)):
                for j in range(len(self.table.log_ligand)):
                    out.append(self.table.record(m, i, j))
        return out


# --------------------------------------------------------------------------------------------- #
# system assembly
# --------------------------------------------------------------------------------------------- #

def _extractant_ligand(entry: SystemEntry) -> LigandSpec:
    ligs = [lig for lig in entry.organic_ligands if lig.role in ("extractant", "synergist") and lig.mechanism is not None]
    if not ligs:
        raise ValueError("the entry has no extractant-role ligand with a mechanism")
    if sum(1 for lig in ligs if lig.role == "extractant") >= 2:
        raise ValueError("mixed organic phases (two extractants) are not fed by one Gen19 table")
    return ligs[0]


def build_gen19_system(table: PredictionTable, entry: SystemEntry, *, feed_anion: str, temperature_C: float,
                       n_prior: float = 3.0, band: str = "20-30C", allow_unsupported: bool = False,
                       values: np.ndarray | None = None, feed_metals: Iterable[str] | None = None,
                       model_id: str | None = None, fit_manifest_sha256: str | None = None,
                       provenance_note: str = "") -> SystemModel | ModelBuildError:
    """A gen18 ``SystemModel`` whose only D model is a :class:`Gen19DModel` on ``entry``'s extractant ligand.

    Mirrors ``gen18proc.dmodel.build_system_model``'s refusals (cross-anion transfer, a feed metal the table does not
    cover) as a returned ``ModelBuildError``; the applicability domain is the entry's recorded block for ``band``
    (``coerce_domain``), so gen18's own OOD flags apply; ``entry.phase`` supplies the LOC for ``THIRD_PHASE_RISK``.
    ``params_source`` is ``"gen19"`` and ``bands[ligand]`` the band used."""
    if feed_anion != entry.medium.anion:
        return ModelBuildError(f"feed anion {feed_anion!r} differs from the system medium {entry.medium.anion!r}: "
                               "cross-anion transfer refused")
    if table.system_id is not None and table.system_id != entry.system_id:
        return ModelBuildError(f"prediction table is for system {table.system_id!r}, entry is {entry.system_id!r}")
    try:
        lig = _extractant_ligand(entry)
    except ValueError as exc:
        return ModelBuildError(str(exc))
    required = tuple(feed_metals) if feed_metals is not None else table.metals
    missing = [m for m in required if m not in table.metals]
    if missing:
        return ModelBuildError(f"the Gen19 table has no prediction for feed metal(s) {missing}")
    domain = coerce_domain(entry.applicability.get(f"{lig.name}|{band}")) if entry.applicability else None
    try:
        model = Gen19DModel(table, lig, mechanism=Mechanism(lig.mechanism), n_prior=n_prior, domain=domain,
                            phase=entry.phase, allow_unsupported=allow_unsupported, values=values, model_id=model_id,
                            fit_manifest_sha256=fit_manifest_sha256, provenance_note=provenance_note)
    except UnsupportedPredictionError:
        raise
    except ValueError as exc:
        return ModelBuildError(f"{lig.name}: {exc}")
    return SystemModel(entry=entry, dmodels={lig.name: model}, complexant=None, activity=None,
                       temperature_C=float(temperature_C), assumptions=ASSUMPTIONS, complexant_model=None,
                       phase=entry.phase, flags=frozenset(), params_source="gen19", bands={lig.name: band})


# --------------------------------------------------------------------------------------------- #
# the prediction request (what the deployment step must predict)
# --------------------------------------------------------------------------------------------- #

def prediction_grid(*, acid_M: Sequence[tuple[float, float]] | Mapping[str, tuple[float, float]],
                    ligand_M: tuple[float, float], n_acid: int = 25, n_ligand: int = 9) -> pd.DataFrame:
    """The (log_acid, log_ligand) grid the deployment step predicts on: log-spaced from the smallest to the largest
    acid the cascade can see (the union of the feed, scrub and strip acid windows) and over the ligand window of the
    design space (formal mol/L).  ``n_acid`` x ``n_ligand`` rows."""
    windows = list(acid_M.values()) if isinstance(acid_M, Mapping) else list(acid_M)
    lo = min(w[0] for w in windows)
    hi = max(w[1] for w in windows)
    if not (lo > 0 and hi >= lo and ligand_M[0] > 0 and ligand_M[1] >= ligand_M[0]):
        raise ValueError("acid and ligand windows must be positive with lo <= hi")
    la = np.linspace(math.log10(lo), math.log10(hi), int(n_acid)) if n_acid > 1 else np.array([math.log10(lo)])
    ll = (np.linspace(math.log10(ligand_M[0]), math.log10(ligand_M[1]), int(n_ligand)) if n_ligand > 1
          else np.array([math.log10(ligand_M[0])]))
    rows = [{"log_acid": round(float(a), 10), "log_ligand": round(float(b), 10)} for a in la for b in ll]
    return pd.DataFrame(rows)


def write_prediction_request(path: str | Path, grid: pd.DataFrame, *, metals: Sequence[str], system_id: str,
                             acid: str = "HNO3", anion: str = "nitrate", note: str = "") -> Path:
    """``<path>``: one row per (metal, grid point) with the condition columns the predictor needs and every prediction
    column empty -- the contract the deployment step fills (``PREDICTION_COLUMNS``)."""
    from gen19ct.manifest import write_csv

    rows = []
    for m in metals:
        for r in grid.itertuples(index=False):
            rows.append({"metal": m, "system_id": system_id, "acid": acid, "anion": anion, "log_acid": r.log_acid,
                         "log_ligand": r.log_ligand, "acid_M": 10.0 ** r.log_acid, "ligand_M": 10.0 ** r.log_ligand,
                         "mean_logD": np.nan, "std_logD": np.nan, "lower_95": np.nan, "upper_95": np.nan,
                         "domain_status": "", "support_score": np.nan, "nearest_support": "", "arm": "",
                         **{c: np.nan for c in MEMBER_COLUMNS}, CONFORMAL_Q95_COLUMN: np.nan,
                         "note": note})
    return write_csv(pd.DataFrame(rows), Path(path), float_format="%.10g")
