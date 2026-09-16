"""``models/mass_action.py`` -- B7, the mechanism-aware mass-action baseline (``preregistration_draft.md`` section 5).

Model (the gen18 M1 form, ``gen18proc.evalproto.fit_mass_action``), one fit per (system, acid anion) on the
outer-training rows::

    log D = a_m + n * log10[L] + p_eff * log10[acid] + delta_pub

* defined for systems labelled ``NEUTRAL_SOLVATING``, ``SOFT_N_DONOR`` or ``MIXED_NEUTRAL``;
* ``a_m`` one intercept per training metal state; ``n`` and ``p_eff`` shared across metals;
* ``delta_pub`` sum-to-zero effects for the publication groups (``pub_group``) with >= 2 rows in the unit (the last
  group in label order is the reference); a group with one row, and a new group, gets 0;
* weighted least squares, unpenalised (``numpy.linalg.lstsq``); unit weights, i.e. the point estimates of gen18's
  replicate-aggregated WLS (a replicate group of ``w`` identical-condition rows enters gen18 as one point of weight
  ``w``; the normal equations are the same);
* an axis with < 3 distinct training levels (rounded to 9 decimals) is not fitted: its slope is the prior
  ``n0 = 3.0`` / ``p0 = 2.0``, status ``assumed``, SE 0 (gen18);
* classical covariance ``(X'WX)^-1 s^2`` and the HC1 sandwich, as gen18.

Prediction for a query row: the intercept of its metal state, or -- a metal state absent from the unit -- the
intercept of its B3x nearest-radius metal among the unit's metals (``SupportIndex._nearest_radius`` pool and basis
rules); plus the publication group's effect (0 for a new group).

Fallbacks (``fallback_level`` / ``fallback_reason``; counted with ``interface.fallback_counts``):

====================================================  ========================================
situation                                             prediction
====================================================  ========================================
system without training rows                          B2 chain (``system_absent``)
mechanism not in the three labels (acidic, ...)       B3x arm: B3 -> B3x -> B2 (``mechanism_<LABEL>``)
no known-state training row under (system, anion)     B3x arm (``no_unit_rows_for_anion``)
query misses log10 acid or extractant M               B3x arm (``query_coordinate_missing``)
no intercept for the state and no radius neighbour    B3x arm (``no_intercept_neighbour``)
====================================================  ========================================

Unknown-state (X(?)) rows are not in a unit fit (their metal state is undefined and is never imputed).
:meth:`B7MassAction.slopes_table` exposes every fitted unit's slopes with standard errors for the reliability check
(prereg section 8).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from gen19ct.chemistry import support_graph as SG
from gen19ct.models import baselines as B
from gen19ct.models import interface as I

#: where the registration left a choice open, the reading implemented here
REGISTRATION_CHOICES: dict[str, str] = {
    "publication_unit": "delta_pub is estimated per pub_group (group_cross_publication_copy), the section 2 unit",
    "weights": "unit row weights (= gen18's replicate-aggregated WLS point estimates; the >= 2 rule counts rows)",
    "unknown_state_rows": "X(?) rows are not in a unit fit",
    "other_mechanisms": "every label outside the three mass-action labels (not only ACIDIC_CATION_EXCHANGE) falls "
                        "back to the B3x arm",
    "anion_unit_absent": "a present system without known-state training rows under the query's anion falls back to "
                         "the B3x arm",
    "intercept_pool": "the hidden metal's B3x nearest-radius metal is chosen among the unit's metals",
}
MASS_ACTION_MECHANISMS: tuple[str, ...] = ("NEUTRAL_SOLVATING", "SOFT_N_DONOR", "MIXED_NEUTRAL")
N_PRIOR = 3.0
P_PRIOR = 2.0
MIN_LEVELS = 3
FALLBACK_CHAIN: tuple[str, ...] = B.CHAINS["B3x"]
SLOPE_COLUMNS: tuple[str, ...] = (
    "system", "anion", "mechanism", "n_rows", "n_metals", "n_pub_groups", "n_pub_groups_with_effect", "levels_lE",
    "levels_lA", "n", "n_se", "n_se_hc1", "n_status", "p_eff", "p_eff_se", "p_eff_se_hc1", "p_eff_status",
    "residual_sd", "rank", "n_params", "dof")


@dataclass(frozen=True)
class MassActionFit:
    """One (system, anion) unit.  ``intercepts`` at log10[L] = log10[acid] = 0 with publication effect 0."""

    system: str
    anion: str
    metals: tuple[str, ...]
    intercepts: dict[str, float]
    n: float
    p_eff: float
    pub_effects: dict[str, float]
    param_names: tuple[str, ...]
    se: dict[str, float]
    se_hc1: dict[str, float]
    slope_status: dict[str, str]
    distinct_levels: dict[str, int]
    n_rows: int
    n_pub_groups: int
    n_pub_groups_with_effect: int
    rank: int
    n_params: int
    residual_sd: float
    priors: dict[str, float] = field(default_factory=dict)

    def predict(self, metal: str, lE: float, lA: float, pub_group: str | None = None) -> float:
        d = self.pub_effects.get(pub_group, 0.0) if pub_group is not None else 0.0
        return float(self.intercepts[metal] + self.n * lE + self.p_eff * lA + d)


def fit_mass_action_unit(lE: Sequence[float], lA: Sequence[float], y: Sequence[float], metals: Sequence[str],
                         pub_groups: Sequence[str], *, weights: Sequence[float] | None = None,
                         n_prior: float = N_PRIOR, p_prior: float = P_PRIOR, min_levels: int = MIN_LEVELS,
                         system: str = "", anion: str = "") -> MassActionFit:
    """The M1 algebra of ``gen18proc.evalproto.fit_mass_action`` on already-selected rows (module docstring).
    Rows must have finite ``lE``, ``lA``, ``y``; raises ``ValueError`` on an empty or malformed input."""
    lE = np.asarray(lE, dtype=float)
    lA = np.asarray(lA, dtype=float)
    y0 = np.asarray(y, dtype=float)
    metal_arr = np.asarray([str(m) for m in metals], dtype=object)
    pub_arr = np.asarray([str(p) for p in pub_groups], dtype=object)
    N = len(y0)
    if N == 0 or not (len(lE) == len(lA) == len(metal_arr) == len(pub_arr) == N):
        raise ValueError("fit_mass_action_unit: empty or ragged input")
    if not (np.isfinite(lE).all() and np.isfinite(lA).all() and np.isfinite(y0).all()):
        raise ValueError("fit_mass_action_unit: non-finite input")
    w = np.ones(N) if weights is None else np.asarray(weights, dtype=float)
    metal_list = sorted(set(metal_arr.tolist()))
    pubs, counts = np.unique(pub_arr.astype(str), return_counts=True)
    pubs_all = sorted(pubs.tolist())
    count_of = dict(zip(pubs.tolist(), counts.tolist()))
    pubs_multi = sorted(p for p in pubs_all if count_of[p] >= 2)
    levels_E = int(np.unique(np.round(lE, 9)).size)
    levels_A = int(np.unique(np.round(lA, 9)).size)
    fit_n, fit_p = levels_E >= min_levels, levels_A >= min_levels
    yv = y0.copy()
    if not fit_n:
        yv = yv - n_prior * lE
    if not fit_p:
        yv = yv - p_prior * lA
    cols, names = [], []
    for m in metal_list:
        cols.append((metal_arr == m).astype(float))
        names.append(f"a.{m}")
    if fit_n:
        cols.append(lE)
        names.append("n")
    if fit_p:
        cols.append(lA)
        names.append("p_eff")
    if len(pubs_multi) >= 2:
        ref = (pub_arr == pubs_multi[-1]).astype(float)
        for p in pubs_multi[:-1]:
            cols.append((pub_arr == p).astype(float) - ref)
            names.append(f"pub.{p}")
    X = np.column_stack(cols)
    k = X.shape[1]
    sw = np.sqrt(w)
    Xw, yw = X * sw[:, None], yv * sw
    beta, _, rank, _ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = yv - X @ beta
    dof = N - k
    s2 = float(np.sum(w * resid ** 2) / dof) if dof > 0 else float("nan")
    XtWX = Xw.T @ Xw
    XtWX_inv = np.linalg.inv(XtWX) if rank == k else np.linalg.pinv(XtWX)
    cov = XtWX_inv * s2
    meat = (Xw * (resid * sw)[:, None]).T @ (Xw * (resid * sw)[:, None])
    cov_hc1 = XtWX_inv @ meat @ XtWX_inv * ((N / dof) if dof > 0 else float("nan"))

    def sd(c: np.ndarray, i: int) -> float:
        return float(math.sqrt(max(c[i, i], 0.0))) if np.isfinite(c[i, i]) else float("nan")
    se = {nm: sd(cov, i) for i, nm in enumerate(names)}
    se_hc1 = {nm: sd(cov_hc1, i) for i, nm in enumerate(names)}
    coef = dict(zip(names, beta.tolist()))
    effects = {p: 0.0 for p in pubs_all}
    if len(pubs_multi) >= 2:
        free = [f"pub.{p}" for p in pubs_multi[:-1]]
        for p, nm in zip(pubs_multi[:-1], free):
            effects[p] = float(coef[nm])
        effects[pubs_multi[-1]] = float(-sum(coef[nm] for nm in free))
        idx = [names.index(nm) for nm in free]
        one = np.ones(len(idx))
        v, vh = float(one @ cov[np.ix_(idx, idx)] @ one), float(one @ cov_hc1[np.ix_(idx, idx)] @ one)
        se[f"pub.{pubs_multi[-1]}"] = math.sqrt(max(v, 0.0)) if np.isfinite(v) else float("nan")
        se_hc1[f"pub.{pubs_multi[-1]}"] = math.sqrt(max(vh, 0.0)) if np.isfinite(vh) else float("nan")
    if not fit_n:
        se["n"] = se_hc1["n"] = 0.0
    if not fit_p:
        se["p_eff"] = se_hc1["p_eff"] = 0.0
    return MassActionFit(
        system=system, anion=anion, metals=tuple(metal_list),
        intercepts={m: float(coef[f"a.{m}"]) for m in metal_list},
        n=float(coef["n"]) if fit_n else float(n_prior), p_eff=float(coef["p_eff"]) if fit_p else float(p_prior),
        pub_effects=effects, param_names=tuple(names), se=se, se_hc1=se_hc1,
        slope_status={"n": "fitted" if fit_n else "assumed", "p_eff": "fitted" if fit_p else "assumed"},
        distinct_levels={"lE": levels_E, "lA": levels_A}, n_rows=int(N), n_pub_groups=len(pubs_all),
        n_pub_groups_with_effect=len(pubs_multi) if len(pubs_multi) >= 2 else 0, rank=int(rank), n_params=int(k),
        residual_sd=float(math.sqrt(s2)) if np.isfinite(s2) else float("nan"),
        priors={"n": float(n_prior), "p_eff": float(p_prior)})


class B7MassAction:
    """B7 (module docstring).  Deterministic; fits units lazily and caches them per fit."""

    name = "B7"

    def __init__(self) -> None:
        self.engine: B.LookupEngine | None = None
        self._units: dict[tuple[int, int], MassActionFit | None] = {}

    def clone(self) -> "B7MassAction":
        return B7MassAction()

    def fit(self, train_rows: pd.DataFrame, context: I.FitContext) -> "B7MassAction":
        table, mask = I.table_and_mask(train_rows, context)
        return self.fit_table(table, mask, context)

    def fit_table(self, table: I.RowTable, mask: np.ndarray, context: I.FitContext) -> "B7MassAction":
        self.engine = B.make_engine(table, mask, context)
        self._units = {}
        return self

    def _engine(self) -> B.LookupEngine:
        if self.engine is None:
            raise RuntimeError("B7: fit first")
        return self.engine

    # ----------------------------------------------------------------------------------------- #
    def unit(self, system: str, anion: str) -> MassActionFit | None:
        """The fitted (system, anion) unit, or None when it has no usable known-state training row."""
        eng = self._engine()
        t = eng.t
        sc, ac = t.sys_code.get(system), t.anion_code.get(anion)
        if sc is None or ac is None:
            return None
        key = (sc, ac)
        if key not in self._units:
            p = eng.pool(t.by_sa_known.get((sc, ac)), B._BASE)
            p = p[np.isfinite(t.acid[p]) & np.isfinite(t.ext[p])]
            if not len(p):
                self._units[key] = None
            else:
                if (t.pub[p] < 0).any():
                    raise ValueError(f"B7: training rows without {t.pub_group_col}")
                self._units[key] = fit_mass_action_unit(
                    t.ext[p], t.acid[p], t.y[p], [t.state_labels[c] for c in t.state[p]],
                    [t.pub_labels[c] for c in t.pub[p]], system=system, anion=anion)
        return self._units[key]

    def slopes_table(self) -> pd.DataFrame:
        """Every (system, anion) unit of a mass-action system in the training rows, with slopes and SEs."""
        eng = self._engine()
        t = eng.t
        recs = []
        for (sc, ac), p in sorted(t.by_sa_known.items(), key=lambda kv: (t.sys_labels[kv[0][0]], t.anion_labels[kv[0][1]])):
            mech = t.sys_mechanism[sc]
            if mech not in MASS_ACTION_MECHANISMS or not len(eng.pool(p, B._BASE)):
                continue
            u = self.unit(t.sys_labels[sc], t.anion_labels[ac])
            if u is None:
                continue
            recs.append({"system": u.system, "anion": u.anion, "mechanism": mech, "n_rows": u.n_rows,
                         "n_metals": len(u.metals), "n_pub_groups": u.n_pub_groups,
                         "n_pub_groups_with_effect": u.n_pub_groups_with_effect, "levels_lE": u.distinct_levels["lE"],
                         "levels_lA": u.distinct_levels["lA"], "n": u.n, "n_se": u.se.get("n"),
                         "n_se_hc1": u.se_hc1.get("n"), "n_status": u.slope_status["n"], "p_eff": u.p_eff,
                         "p_eff_se": u.se.get("p_eff"), "p_eff_se_hc1": u.se_hc1.get("p_eff"),
                         "p_eff_status": u.slope_status["p_eff"], "residual_sd": u.residual_sd, "rank": u.rank,
                         "n_params": u.n_params, "dof": u.n_rows - u.n_params})
        return pd.DataFrame.from_records(recs, columns=list(SLOPE_COLUMNS))

    # ----------------------------------------------------------------------------------------- #
    def predict(self, query_rows: pd.DataFrame) -> pd.DataFrame:
        eng = self._engine()
        return self._predict(I.Queries.from_frame(query_rows, eng.t))

    def predict_positions(self, positions: np.ndarray) -> pd.DataFrame:
        eng = self._engine()
        return self._predict(I.Queries.from_positions(eng.t, positions))

    def _fallback(self, q: I.Queries, i: int, chain: tuple[str, ...], reason: str) -> dict[str, Any]:
        eng = self._engine()
        r = eng.run_chain(q, i, chain, B._BASE)
        rec = eng.record(q, i, r, False, B._BASE)
        rec["fallback_reason"] = ";".join(x for x in (reason, rec["fallback_reason"]) if x)
        return rec

    def _predict(self, q: I.Queries) -> pd.DataFrame:
        eng = self._engine()
        t = eng.t
        B.check_queries(eng, q)
        recs = []
        for i in range(len(q)):
            sy = q.system[i]
            sc = t.sys_code.get(sy) if sy is not None else None
            if sc is None or not len(eng.pool(t.by_s.get(sc), B._BASE)):
                recs.append(self._fallback(q, i, B.CHAINS["B2"], "system_absent"))
                continue
            mech = t.sys_mechanism[sc]
            if mech not in MASS_ACTION_MECHANISMS:
                recs.append(self._fallback(q, i, FALLBACK_CHAIN, f"mechanism_{mech}"))
                continue
            u = self.unit(sy, q.anion[i])
            if u is None:
                recs.append(self._fallback(q, i, FALLBACK_CHAIN, "no_unit_rows_for_anion"))
                continue
            if not (np.isfinite(q.acid[i]) and np.isfinite(q.ext[i])):
                recs.append(self._fallback(q, i, FALLBACK_CHAIN, "query_coordinate_missing"))
                continue
            st = q.state[i]
            reason, radius = "", None
            if st is not None and st in u.intercepts:
                metal = st
            else:
                metal = None
                if st is not None:
                    others = {m: SG.metal_properties(m) for m in u.metals if m != st}
                    radius = SG.SupportIndex._nearest_radius(B._RADIUS_NS, SG.metal_properties(st), others) \
                        if others else None
                    metal = radius["nearest_radius_metal"] if radius else None
                if metal is None:
                    recs.append(self._fallback(q, i, FALLBACK_CHAIN, "no_intercept_neighbour"))
                    continue
                reason = "intercept_from_nearest_radius_metal"
            g = q.pub_group[i]
            delta = float(u.pub_effects.get(g, 0.0)) if g is not None else 0.0
            rec = I.empty_prediction_record()
            rec.update(row_id=q.labels[i], mean_logD=u.predict(metal, float(q.ext[i]), float(q.acid[i]), g),
                       fallback_level="B7", fallback_reason=reason, lookup_system=sy, n_candidates=u.n_rows,
                       b7_anion=u.anion, b7_intercept_metal=metal, b7_n=u.n, b7_p_eff=u.p_eff,
                       b7_n_status=u.slope_status["n"], b7_p_eff_status=u.slope_status["p_eff"], b7_pub_effect=delta)
            if radius is not None:
                rec.update(nearest_radius_metal=radius["nearest_radius_metal"],
                           nearest_radius_distance_A=radius["nearest_radius_distance_A"],
                           nearest_radius_basis=radius["nearest_radius_basis"],
                           nearest_radius_pool=("same_species_charge" if radius["nearest_radius_same_species_charge"]
                                                else "same_charge" if radius["nearest_radius_same_charge"] else "any"))
            recs.append(rec)
        return I.records_to_frame(recs)
