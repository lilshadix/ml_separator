"""``dmodel.py`` — distribution models of one ligand for a set of metals (DESIGN.md section 5).

Models (section 5.1): ``ConstantD`` (labelled sanity limit, provenance ``assumed``),
``CationExchangeMassAction`` (5.2 a), ``SolvatingMassAction`` (5.2 b), ``NearestConditionD``
(the B1 fallback D source: same-metal 1-NN in (log10 acid, log10 ligand) with the ideal
depletion term ``n_prior * log10(L_f / L_T)``), ``AqueousComplexantWrapper`` (5.2 c) and the
exploratory ``EffectiveCapacity`` activity model (5.6).  ``build_system_model`` (5.7) assembles a
``SystemModel`` from a ``SystemEntry``; composition over several ligands (5.3), the loading
fraction and the chemistry flags (5.4) are module functions used by ``equilibrium.solve_stage``.

Units: every concentration mol/L in the phase named; ligand concentrations of dimeric acidic
extractants are dimer concentrations ``[(HA)2]``; ``log`` means log10; the analytic partials of
``DEval`` are ``d log10 D / d ln(variable)`` (``a / ln 10`` for a slope ``a``).  Every model also
exposes ``core(L, h, nu, c, ligand_total)``, a vectorised evaluation over ``model.metals`` that
returns ``(log10 D, dlogd_dlnL, dlogd_dlnh, dlogd_dlnanion, dlogd_dlnc)`` arrays; the stage solver
uses it and falls back to ``evaluate`` for foreign ``DModel`` implementations.

Equations (section 5.2):

* cation exchange: ``log D0_i = log K_i + a log L - b log h``, ``q_i = 3`` dimers, ``p_i = 3``,
  ``z_i = 0``; optional anion complexation ``alpha_i = 1 + sum_j beta^X_ij anion^j`` (linear
  betas, M^-j) when ``beta_anion`` is sourced, else ``MEDIUM_STRENGTH_UNMODELLED`` when the stage
  anion lies more than a factor 2 outside the studied acid interval (the anion interval of an
  HCl / HNO3 medium without salting agent, the closest recorded proxy);
* solvating: ``log D0_i = log K_i + n log L + p_anion log anion + p_h log h``, ``q_i = n``,
  ``p_i = 0``, ``z_i = anions_per_metal``; HNO3 uptake ``a_k = K_H h anion L`` (K_H in L^2/mol^2;
  None -> 0 and ``ACID_UPTAKE_UNMODELLED`` when ``h > 1``); ``ANION_ACID_CONFOUNDED`` when a
  corpus-fitted set is applied where ``|anion - h| > ANION_ACID_TOL * max(anion, h)``;
* complexant: ``alpha_i(c) = 1 + sum_m beta_im c^m`` (``beta = 10^log_beta``, cumulative),
  ``D_i = D0_i / alpha_i``, ``phi_i(c) = sum_m m beta_im c^m / alpha_i``,
  ``alpha_H(h) = 1 + sum_j K_Hj h^j``, ``d log10 D / d ln c = -phi_i / ln 10``.

Sources: DESIGN.md sections 1.4, 3.2, 3.4, 5, 12.1; the placeholder chemistry of the literature
entries stays in ``systems.py`` / ``literature.py`` (WB1).  No literature number is hard-coded here.
"""
from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from .domain import build_domain, coerce_domain, domain_flags
from .types import (
    ASSUMED_LABEL,
    ActivityModel,
    ApplicabilityDomain,
    CationExchangeParams,
    ComplexantSpec,
    DEval,
    DistributionRecord,
    DModel,
    Flag,
    LigandSpec,
    Mechanism,
    MediumSpec,
    ModelBuildError,
    PhaseBehaviour,
    Provenance,
    ProvStatus,
    SolvatingParams,
    Source,
    Sourced,
    StageState,
    SystemEntry,
)

__all__ = [
    "LN10", "LOG10_CAP", "ASSUMPTIONS", "TEMPERATURE_BANDS", "ANION_ACID_TOL",
    "ConstantD", "CationExchangeMassAction", "SolvatingMassAction", "NearestConditionD",
    "AqueousComplexantWrapper", "ComplexantModel", "EffectiveCapacity", "SystemModel",
    "build_system_model", "evaluate_composite", "tracer_state", "loading_fractions",
    "chemistry_flags", "ligand_loading_fraction", "ligand_chemistry_flags", "records_to_frame",
    "select_band", "parse_band", "pow10",
]

LN10 = math.log(10.0)
LOG10_CAP = 300.0
"""``log10 D`` is capped here inside the solver so that ``10**log_d`` stays finite (D -> inf)."""

ASSUMPTIONS: tuple[str, ...] = (
    "concentrations for activities",
    "complete dimerisation of acidic organophosphorus extractants",
    "fixed stoichiometries",
    "no water co-extraction",
    "no mixed organic complexes",
    "no ligand aggregation",
    "constant phase volumes",
    "temperature enters only through the parameter set's temperature band",
)
"""Model idealisations (DESIGN.md section 0.1), carried as ``SystemModel.assumptions``."""

TEMPERATURE_BANDS: tuple[str, ...] = ("<20C", "20-30C", "30-40C", "40-50C", ">=50C")
"""Fixed parameter-set temperature bands (DESIGN.md section 3.4)."""

ANION_ACID_TOL = 0.05
"""Relative deviation of [anion] from [H+] above which a corpus-fitted solvating set raises
``ANION_ACID_CONFOUNDED`` (DESIGN.md section 5.2 b names salting agents and strip liquors with
added nitrate; the 5 % tolerance keeps the flag off for the anion consumed by metal transport)."""


# ---------------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------------

def pow10(x: float) -> float:
    """``10**x`` that returns 0 for ``-inf`` and ``inf`` above the float range (never raises)."""
    if x == -math.inf:
        return 0.0
    if x > 308.0:
        return math.inf
    return 10.0 ** x


def _log10(x: float) -> float:
    return math.log10(x) if x > 0 else -math.inf


def _num(obj: Any, name: str, *, default: float | None = None, required: bool = True,
         ) -> float | None:
    """Float value of a ``Sourced`` (or plain number); ``default`` when None; ``ValueError`` when
    None, no default and ``required``."""
    value = obj.value if isinstance(obj, Sourced) else obj
    if value is None:
        if default is not None:
            return float(default)
        if required:
            raise ValueError(f"missing parameter {name}")
        return None
    return float(value)


def _provenance_of(obj: Any, note: str) -> Provenance:
    if isinstance(obj, Sourced):
        return obj.provenance
    return Provenance(ProvStatus.ASSUMED, Source(kind="none"), None, ASSUMED_LABEL, note=note)


def _status_of(obj: Any) -> ProvStatus | None:
    return obj.provenance.status if isinstance(obj, Sourced) else None


def _as_mapping(value: Any, metals: Iterable[str], default: float) -> dict[str, float]:
    """A per-metal dict from a scalar or a mapping (missing metals take ``default``)."""
    if isinstance(value, Mapping):
        return {m: float(value.get(m, default)) for m in metals}
    scalar = default if value is None else float(value)
    return {m: scalar for m in metals}


# ---------------------------------------------------------------------------------------------
# 5.4  loading fraction and chemistry flags of one ligand
# ---------------------------------------------------------------------------------------------

def ligand_loading_fraction(state: StageState, ligand: str, q: Mapping[str, float]) -> float:
    """``lambda_k = (sum_i q_i^(k) y_i^(k) + a_k) / L_T^(k)`` (DESIGN.md section 5.4)."""
    lt = state.ligand_total[ligand]
    if lt <= 0:
        return math.nan
    yk = state.y_by_ligand.get(ligand, {})
    bound = sum(q.get(m, 0.0) * y for m, y in yk.items())
    return (bound + state.acid_in_org.get(ligand, 0.0)) / lt


def ligand_chemistry_flags(state: StageState, ligand: str, q: Mapping[str, float],
                           phase: PhaseBehaviour | None) -> frozenset[Flag]:
    """``HIGH_LOADING`` (lambda > 0.5), ``LOADING_CAP_HIT`` (L_f < 1e-6 L_T),
    ``PHASE_BEHAVIOUR_UNKNOWN`` (LOC null and lambda > 0.3), ``THIRD_PHASE_RISK`` (sourced LOC
    exceeded by the organic metal plus HNO3.L, or sourced acid LOC exceeded by HNO3.L)."""
    flags: set[Flag] = set()
    lt = state.ligand_total[ligand]
    lam = ligand_loading_fraction(state, ligand, q)
    if lam > 0.5:
        flags.add(Flag.HIGH_LOADING)
    if state.ligand_free.get(ligand, lt) < 1e-6 * lt:
        flags.add(Flag.LOADING_CAP_HIT)
    loc_metal = _num(phase.loc_metal_M, "loc_metal_M", required=False) if phase else None
    loc_acid = _num(phase.loc_acid_M, "loc_acid_M", required=False) if phase else None
    if loc_metal is None and lam > 0.3:
        flags.add(Flag.PHASE_BEHAVIOUR_UNKNOWN)
    org_acid = sum(state.acid_in_org.values())
    if loc_metal is not None and sum(state.y.values()) + org_acid > loc_metal:
        flags.add(Flag.THIRD_PHASE_RISK)
    if loc_acid is not None and org_acid > loc_acid:
        flags.add(Flag.THIRD_PHASE_RISK)
    return frozenset(flags)


# ---------------------------------------------------------------------------------------------
# 5.2 (c)  complexant algebra
# ---------------------------------------------------------------------------------------------

class ComplexantModel:
    """Side-reaction algebra of one aqueous complexant (DESIGN.md section 5.2 c).

    ``metals`` are the metals whose every ``log_beta`` value is known (a placeholder with
    ``value None`` excludes the metal; ``build_system_model`` refuses feed metals so excluded).
    ``beta_matrix(metals)`` returns the cumulative linear betas (M^-m) aligned with ``metals``;
    ``terms(c, metals)`` returns ``(alpha, phi, dphi_dlnc)``; ``alpha_h(h)`` / ``dalpha_h_dlnh``
    are the protonation side-reaction coefficient and its ln-derivative.
    """

    def __init__(self, spec: ComplexantSpec):
        self.spec = spec
        self.name = spec.name
        betas: dict[str, np.ndarray] = {}
        for metal, entries in spec.log_beta.items():
            vals = [_num(e, f"{spec.name}.log_beta.{metal}", required=False) for e in entries]
            if not vals or any(v is None for v in vals):
                continue
            betas[metal] = np.power(10.0, np.asarray(vals, dtype=float))
        self._betas = betas
        self.metals: tuple[str, ...] = tuple(betas)
        self.m_max = max((b.size for b in betas.values()), default=0)
        self.exponents = np.arange(1, self.m_max + 1, dtype=float)
        kh = [_num(e, f"{spec.name}.protonation_logk", required=False)
              for e in spec.protonation_logk]
        kh = [v for v in kh if v is not None]
        self._kh = np.power(10.0, np.asarray(kh, dtype=float)) if kh else np.zeros(0)
        self._jexp = np.arange(1, self._kh.size + 1, dtype=float)
        self.regeneration_fraction = _num(spec.regeneration_fraction, "regeneration_fraction",
                                          required=False)
        self._cache: dict[tuple[str, ...], np.ndarray] = {}

    def beta_matrix(self, metals: Iterable[str]) -> np.ndarray:
        key = tuple(metals)
        mat = self._cache.get(key)
        if mat is None:
            mat = np.zeros((len(key), self.m_max))
            for i, m in enumerate(key):
                if m not in self._betas:
                    raise ValueError(f"complexant {self.name!r} has no log_beta for metal {m!r}")
                b = self._betas[m]
                mat[i, : b.size] = b
            self._cache[key] = mat
        return mat

    def terms(self, c: float, metals: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(alpha_i, phi_i, dphi_i/dlnc)`` at free deprotonated complexant ``c`` (mol/L)."""
        beta = self.beta_matrix(metals)
        if self.m_max == 0 or c <= 0.0:
            n = len(metals)
            return np.ones(n), np.zeros(n), np.zeros(n)
        cp = beta * np.power(c, self.exponents)            # beta_m c^m
        s0 = cp.sum(axis=1)
        s1 = (cp * self.exponents).sum(axis=1)
        s2 = (cp * self.exponents ** 2).sum(axis=1)
        alpha = 1.0 + s0
        phi = s1 / alpha
        return alpha, phi, s2 / alpha - phi * phi

    def alpha_h(self, h: float) -> float:
        if self._kh.size == 0 or h <= 0.0:
            return 1.0
        return 1.0 + float((self._kh * np.power(h, self._jexp)).sum())

    def dalpha_h_dlnh(self, h: float) -> float:
        if self._kh.size == 0 or h <= 0.0:
            return 0.0
        return float((self._kh * self._jexp * np.power(h, self._jexp)).sum())


# ---------------------------------------------------------------------------------------------
# model base
# ---------------------------------------------------------------------------------------------

class _ModelBase:
    """Shared machinery of the concrete models: metal index, ``evaluate`` built on ``core``."""

    ligand: str
    mechanism: Mechanism
    metals: tuple[str, ...]
    q: Mapping[str, float]
    p: Mapping[str, float]
    z: Mapping[str, float]
    domain: ApplicabilityDomain | None
    provenance: Provenance
    phase: PhaseBehaviour | None
    base_flags: frozenset[Flag]
    k_acid_uptake: float

    def _init_common(self, ligand: str, mechanism: Mechanism, metals: tuple[str, ...],
                     q: Any, p: Any, z: Any, domain: ApplicabilityDomain | None,
                     provenance: Provenance, phase: PhaseBehaviour | None,
                     base_flags: Iterable[Flag], k_acid_uptake: float,
                     ligand_scale: float = 1.0) -> None:
        if not metals:
            raise ValueError(f"model for ligand {ligand!r} has no metal with a usable parameter")
        self.ligand = str(ligand)
        self.mechanism = Mechanism(mechanism)
        self.metals = tuple(metals)
        self._index = {m: i for i, m in enumerate(self.metals)}
        self.q = _as_mapping(q, self.metals, 0.0)
        self.p = _as_mapping(p, self.metals, 0.0)
        self.z = _as_mapping(z, self.metals, 0.0)
        self.domain = domain
        self.provenance = provenance
        self.phase = phase
        self.base_flags = frozenset(base_flags)
        self.k_acid_uptake = float(k_acid_uptake)
        self.ligand_scale = float(ligand_scale)
        n = len(self.metals)
        self._zeros = np.zeros(n)

    # -- to be overridden --------------------------------------------------------------------
    def core(self, L: float, h: float, nu: float, c: float, ligand_total: float,
             capacity: float | None = None,
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Vectorised evaluation over ``self.metals`` at free ligand ``L``, aqueous ``h``,
        ``nu``, free complexant ``c`` (mol/L); ``ligand_total`` is the formal total of the code
        basis (dimer for dimers) and ``capacity`` the effective total (``phi * L_T`` under
        ``EffectiveCapacity``, else ``L_T``); returns ``(log10 D, dlogd_dlnL, dlogd_dlnh,
        dlogd_dlnanion, dlogd_dlnc)`` arrays."""
        raise NotImplementedError

    def _extra_flags(self, state: StageState) -> frozenset[Flag]:
        return frozenset()

    def _extra_distance(self, state: StageState, metal: str | None = None) -> dict[str, float]:
        """Model-specific distance axes: for one ``metal`` when given, else the maximum over the
        metals of ``state`` (the stage diagnostics)."""
        return {}

    def stage_arrays(self, metals: tuple[str, ...],
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """``(q, p, z, index)`` arrays aligned with ``metals`` (cached per metal order; the stage
        solver's per-call cost).  ``index`` is None when ``metals`` equals ``self.metals``."""
        cache = self.__dict__.setdefault("_stage_arrays", {})
        arrays = cache.get(metals)
        if arrays is None:
            q = np.array([float(self.q.get(m, 0.0)) for m in metals])
            p = np.array([float(self.p.get(m, 0.0)) for m in metals])
            z = np.array([float(self.z.get(m, 0.0)) for m in metals])
            pos = [self._index[m] for m in metals]
            index = None if pos == list(range(len(self.metals))) else np.asarray(pos)
            arrays = cache[metals] = (q, p, z, index)
        return arrays

    def state_flags(self, state: StageState, activity: ActivityModel | None = None, *,
                    metal: str | None = None) -> tuple[frozenset[Flag], dict[str, float]]:
        """Chemistry (section 5.4), model-specific and applicability (section 5.5) flags of
        ``state`` for this ligand with the per-axis distances.  Metal-independent except for the
        model-specific axes (``NearestConditionD``'s ``nn``): with ``metal`` given they are that
        metal's, otherwise the maximum over the metals of ``state`` (the stage solver calls it
        once per ligand)."""
        flags = set(self.base_flags)
        flags |= ligand_chemistry_flags(state, self.ligand, self.q, self.phase)
        flags |= self._extra_flags(state)
        dflags, dist = domain_flags(self.domain, state, self.ligand,
                                    ligand_scale=self.ligand_scale,
                                    loading=ligand_loading_fraction(state, self.ligand, self.q))
        flags |= dflags
        dist = dict(dist)
        dist.update(self._extra_distance(state, metal))
        return frozenset(flags), dist

    # -- DModel protocol ---------------------------------------------------------------------
    def evaluate(self, metal: str, state: StageState,
                 activity: ActivityModel | None = None) -> DEval:
        i = self._index.get(metal)
        if i is None:
            raise ValueError(f"metal {metal!r} is not parameterised for ligand {self.ligand!r}")
        if self.ligand not in state.ligand_total:
            raise ValueError(f"state carries no ligand_total for {self.ligand!r}")
        lig_free = (activity.organic_free_ligand(state, self.ligand) if activity is not None
                    else state.ligand_free[self.ligand])
        lt = state.ligand_total[self.ligand]
        scale = getattr(activity, "effective_ligand_total", None)
        cap = scale(lt) if scale is not None else lt
        logd, s_l, s_h, s_nu, s_c = self.core(lig_free, state.h, state.anion, state.c_free, lt,
                                              cap)
        flags, dist = self.state_flags(state, activity, metal=metal)
        ld = float(logd[i])
        return DEval(log_d=ld, d=pow10(ld), dlogd_dlnL=float(s_l[i]), dlogd_dlnh=float(s_h[i]),
                     dlogd_dlnanion=float(s_nu[i]), dlogd_dlnc=float(s_c[i]),
                     flags=flags, ood_distance=dist)

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(ligand={self.ligand!r}, mechanism={self.mechanism.value},"
                f" metals={self.metals})")


# ---------------------------------------------------------------------------------------------
# ConstantD
# ---------------------------------------------------------------------------------------------

class ConstantD(_ModelBase):
    """Constant D per metal: the labelled sanity limit and the Kremser oracle (section 5.1).

    ``log_d_by_metal`` may hold ``-inf`` (D = 0) or any large value (D -> inf); ``q``, ``p``,
    ``z`` default to 0 (no ligand depletion, no acid release, no anion transport) and may be
    scalars or per-metal mappings.  Provenance status ``assumed`` with ``ASSUMED_PLACEHOLDER``.
    """

    def __init__(self, log_d_by_metal: Mapping[str, float], *, q: Any = 0.0, p: Any = 0.0,
                 z: Any = 0.0, ligand: str = "constant",
                 mechanism: Mechanism = Mechanism.SOLVATING,
                 domain: ApplicabilityDomain | None = None,
                 phase: PhaseBehaviour | None = None):
        metals = tuple(log_d_by_metal)
        prov = Provenance(ProvStatus.ASSUMED, Source(kind="none", locator="ConstantD"), None,
                          ASSUMED_LABEL, note="labelled sanity limit: constant D, no loading")
        self._init_common(ligand, mechanism, metals, q, p, z, domain, prov, phase, (), 0.0)
        self._logd = np.asarray([float(log_d_by_metal[m]) for m in metals], dtype=float)

    def core(self, L, h, nu, c, ligand_total, capacity=None):
        return self._logd, self._zeros, self._zeros, self._zeros, self._zeros


# ---------------------------------------------------------------------------------------------
# mass-action models
# ---------------------------------------------------------------------------------------------

def _logk_arrays(log_k: Mapping[str, Any], ligand: str) -> tuple[tuple[str, ...], np.ndarray,
                                                                  Provenance, bool]:
    """Metals with a known log K, their values, the provenance of the first entry and whether
    every entry is ``fitted_from_corpus``."""
    metals: list[str] = []
    values: list[float] = []
    statuses: list[ProvStatus | None] = []
    prov: Provenance | None = None
    for metal, obj in log_k.items():
        v = _num(obj, f"{ligand}.log_k.{metal}", required=False)
        if v is None:
            continue
        metals.append(str(metal))
        values.append(v)
        statuses.append(_status_of(obj))
        if prov is None:
            prov = _provenance_of(obj, f"log K of {ligand}")
    if prov is None:
        prov = Provenance(ProvStatus.UNKNOWN, Source(kind="none"), note=f"no log K for {ligand}")
    fitted = bool(statuses) and all(s == ProvStatus.FITTED_FROM_CORPUS for s in statuses)
    return tuple(metals), np.asarray(values, dtype=float), prov, fitted


def _ligand_name(ligand: LigandSpec | str) -> str:
    return ligand.name if isinstance(ligand, LigandSpec) else str(ligand)


def _ligand_scale(ligand: LigandSpec | str) -> float:
    """Factor from the code's ligand basis to the recorded formal concentration: 2 for a dimeric
    ligand (``[HA]_T = 2 [(HA)2]_T``, DESIGN.md section 1.1), else 1."""
    return 2.0 if isinstance(ligand, LigandSpec) and ligand.aggregation == "dimer" else 1.0


def _stoichiometry(ligand: LigandSpec | str, mechanism: Mechanism, n_solv: float | None,
                   ) -> tuple[float, float, float]:
    """``(q, p, z)`` from the ``LigandSpec`` stoichiometry, ideal values when absent."""
    if mechanism == Mechanism.CATION_EXCHANGE:
        q_d, p_d, z_d = 3.0, 3.0, 0.0
    else:
        q_d, p_d, z_d = (n_solv if n_solv is not None else 3.0), 0.0, 3.0
    if isinstance(ligand, LigandSpec):
        st = ligand.stoichiometry
        q = _num(st.ligands_per_metal, "ligands_per_metal", required=False)
        p = _num(st.protons_released_per_metal, "protons_released_per_metal", required=False)
        z = _num(st.anions_per_metal, "anions_per_metal", required=False)
        return (q if q is not None else q_d, p if p is not None else p_d,
                z if z is not None else z_d)
    return q_d, p_d, z_d


class CationExchangeMassAction(_ModelBase):
    """``log D0_i = log K_i + a log L - b log h`` on the dimer basis (DESIGN.md section 5.2 a)."""

    def __init__(self, params: CationExchangeParams, ligand: LigandSpec | str,
                 medium: MediumSpec | None, domain: ApplicabilityDomain | None, *,
                 phase: PhaseBehaviour | None = None, base_flags: Iterable[Flag] = ()):
        name = _ligand_name(ligand)
        metals, logk, prov, fitted = _logk_arrays(params.log_k, name)
        self.params = params
        self.medium = medium
        self.a = _num(params.a_dimer, f"{name}.a_dimer")
        self.b = _num(params.b_proton, f"{name}.b_proton")
        self.is_fitted = fitted
        q, p, z = _stoichiometry(ligand, Mechanism.CATION_EXCHANGE, None)
        flags = set(base_flags)
        if fitted:
            flags.add(Flag.EQUILIBRIUM_ACID_ASSUMED_NOMINAL)
        self._init_common(name, Mechanism.CATION_EXCHANGE, metals, q, p, z, domain, prov, phase,
                          flags, 0.0, _ligand_scale(ligand))
        self._logk = logk
        self._s_l = np.full(len(metals), self.a / LN10)
        self._s_h = np.full(len(metals), -self.b / LN10)
        # optional anion complexation (linear cumulative betas, M^-j)
        self._beta_anion: np.ndarray | None = None
        if params.beta_anion:
            rows = []
            usable = True
            for m in metals:
                entries = params.beta_anion.get(m)
                if not entries:
                    usable = False
                    break
                vals = [_num(e, f"{name}.beta_anion.{m}", required=False) for e in entries]
                if any(v is None for v in vals):
                    usable = False
                    break
                rows.append(vals)
            if usable:
                jmax = max(len(r) for r in rows)
                mat = np.zeros((len(metals), jmax))
                for i, r in enumerate(rows):
                    mat[i, : len(r)] = r
                self._beta_anion = mat
                self._jexp = np.arange(1, jmax + 1, dtype=float)

    def core(self, L, h, nu, c, ligand_total, capacity=None):
        logd = self._logk + (self.a * _log10(L) - self.b * _log10(h))
        if self._beta_anion is None:
            return logd, self._s_l, self._s_h, self._zeros, self._zeros
        if nu <= 0.0:
            return logd, self._s_l, self._s_h, self._zeros, self._zeros
        cp = self._beta_anion * np.power(nu, self._jexp)
        alpha = 1.0 + cp.sum(axis=1)
        s_nu = -((cp * self._jexp).sum(axis=1) / alpha) / LN10
        return logd - np.log10(alpha), self._s_l, self._s_h, s_nu, self._zeros

    def _extra_flags(self, state: StageState) -> frozenset[Flag]:
        if self._beta_anion is not None or self.domain is None:
            return frozenset()
        lo, hi = self.domain.log_acid
        if not (math.isfinite(lo) and math.isfinite(hi)):
            return frozenset()
        la = _log10(state.anion)
        if la < lo - math.log10(2.0) or la > hi + math.log10(2.0):
            return frozenset({Flag.MEDIUM_STRENGTH_UNMODELLED})
        return frozenset()


class SolvatingMassAction(_ModelBase):
    """``log D0_i = log K_i + n log L + p_anion log anion + p_h log h`` (DESIGN.md 5.2 b)."""

    def __init__(self, params: SolvatingParams, ligand: LigandSpec | str,
                 medium: MediumSpec | None, domain: ApplicabilityDomain | None, *,
                 phase: PhaseBehaviour | None = None, base_flags: Iterable[Flag] = ()):
        name = _ligand_name(ligand)
        metals, logk, prov, fitted = _logk_arrays(params.log_k, name)
        self.params = params
        self.medium = medium
        self.n = _num(params.n_solvation, f"{name}.n_solvation")
        self.p_anion = _num(params.p_anion, f"{name}.p_anion")
        self.p_h = _num(params.p_h, f"{name}.p_h", default=0.0)
        kh = _num(params.k_acid_uptake, f"{name}.k_acid_uptake", required=False)
        self.k_acid_uptake_known = kh is not None
        self.is_fitted = fitted
        q, p, z = _stoichiometry(ligand, Mechanism.SOLVATING, self.n)
        flags = set(base_flags)
        if fitted:
            flags.add(Flag.EQUILIBRIUM_ACID_ASSUMED_NOMINAL)
        self._init_common(name, Mechanism.SOLVATING, metals, q, p, z, domain, prov, phase, flags,
                          kh if kh is not None else 0.0, _ligand_scale(ligand))
        self._logk = logk
        self._s_l = np.full(len(metals), self.n / LN10)
        self._s_h = np.full(len(metals), self.p_h / LN10)
        self._s_nu = np.full(len(metals), self.p_anion / LN10)

    def core(self, L, h, nu, c, ligand_total, capacity=None):
        logd = self._logk + (self.n * _log10(L) + self.p_anion * _log10(nu)
                             + self.p_h * _log10(h))
        return logd, self._s_l, self._s_h, self._s_nu, self._zeros

    def _extra_flags(self, state: StageState) -> frozenset[Flag]:
        flags: set[Flag] = set()
        if not self.k_acid_uptake_known and state.h > 1.0:
            flags.add(Flag.ACID_UPTAKE_UNMODELLED)
        if self.is_fitted:
            scale = max(state.anion, state.h, 1e-300)
            if abs(state.anion - state.h) > ANION_ACID_TOL * scale:
                flags.add(Flag.ANION_ACID_CONFOUNDED)
        return frozenset(flags)


# ---------------------------------------------------------------------------------------------
# NearestConditionD
# ---------------------------------------------------------------------------------------------

def records_to_frame(records: Iterable[DistributionRecord]) -> pd.DataFrame:
    """Flat frame of ``DistributionRecord`` objects: one row per record, mappings flattened as
    ``ligand_M__<ligand>`` and ``metals_initial_mM__<metal>`` plus ``metal_initial_mM`` (sum)."""
    rows: list[dict[str, Any]] = []
    for rec in records:
        row: dict[str, Any] = {
            "record_id": rec.record_id, "metal": rec.metal, "d": rec.d, "log_d": rec.log_d,
            "acid_nominal_M": rec.acid_nominal_M, "acid_eq_M": rec.acid_eq_M,
            "anion_M": rec.anion_M, "complexant_M": rec.complexant_M, "oa_ratio": rec.oa_ratio,
            "temperature_C": rec.temperature_C, "contact_time_min": rec.contact_time_min,
            "diluent_name": rec.diluent_name, "publication_id": rec.publication_id,
            "experiment_series_id": rec.experiment_series_id, "replicate_id": rec.replicate_id,
            "loading_series_id": rec.loading_series_id, "is_tracer": bool(rec.is_tracer),
            "fit_eligible": bool(rec.fit_eligible),
            "fit_ineligible_reason": rec.fit_ineligible_reason,
            "duplicate_flag": rec.duplicate_flag, "status": rec.provenance.status.value,
        }
        for lig, conc in rec.ligand_M.items():
            row[f"ligand_M__{lig}"] = conc
        total = 0.0
        any_metal = False
        for m, mm in rec.metals_initial_mM.items():
            row[f"metals_initial_mM__{m}"] = mm
            if mm is not None and not (isinstance(mm, float) and math.isnan(mm)):
                total += float(mm)
                any_metal = True
        row["metal_initial_mM"] = total if any_metal else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _tracer_log_d(log_d: np.ndarray, ligand_total: np.ndarray, metal_mm: "pd.Series | Any",
                  n: float) -> tuple[np.ndarray, np.ndarray]:
    """Lift each record's ``log D`` back to its own tracer limit (integration, 2026-09-13).

    A record measured at initial aqueous metal ``T`` (mol/L, O/A = 1 assumed as everywhere in the
    corpus semantics) already carries its own ligand depletion: ``y = T D / (1 + D)`` is in the
    organic and ``L_f = L_T - n y`` is left free, so by the ideal law that `PRE_REGISTRATION.md`
    §7 supported (R2),

        log D_tracer = log D_record + n log10(L_T / L_f) .

    Returns ``(log D at zero loading, whether the record could be corrected)``.  A record with no
    recorded metal concentration is left as it stands (24 % of the corpus; the tracer reading is
    the standing assumption for those), and so is one whose implied ``L_f`` is non-positive --
    loaded past the ideal capacity, where the law gives no finite answer.

    Without this the deployed 1-NN source depletes an already-depleted measurement a second time:
    28.7 % of corpus records that have a metal concentration were measured above loading fraction
    0.1 (TBDGA: 81 %, median 0.30), so the bias is not rare.  See `addenda/INTEGRATION.md`.
    """
    lt = np.asarray(ligand_total, dtype=float)
    mm = pd.to_numeric(metal_mm, errors="coerce").to_numpy(dtype=float) \
        if metal_mm is not None else np.full(lt.shape, np.nan)
    out = np.array(log_d, dtype=float, copy=True)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        d = np.power(10.0, np.clip(log_d, -300.0, 300.0))
        y = (mm * 1e-3) * d / (1.0 + d)                     # organic metal, mol/L, at O/A = 1
        free = lt - n * y
        ok = np.isfinite(y) & np.isfinite(free) & (free > 0) & (lt > 0) & (y > 0)
        out[ok] = log_d[ok] + n * np.log10(lt[ok] / free[ok])
    return out, ok


class NearestConditionD(_ModelBase):
    """The B1 fallback D source (DESIGN.md sections 5.1 and 10.2, PRE_REGISTRATION.md section 3).

    For each metal the same-metal record nearest to the stage point ``(log10 h, log10 [L]_T)``
    (the formal recorded concentration: ``ligand_scale * ligand_total``, i.e. the monomer
    concentration for a dimeric ligand) in Euclidean distance (ties: smaller |delta log acid|,
    then smaller ``record_id``) supplies the tracer log D; the loading correction is the ideal
    depletion term ``log D = log D_nn + n_prior * log10(L_f / capacity)`` with ``capacity`` the
    effective total (``L_T``, or ``phi L_T`` under ``EffectiveCapacity``), so the correction is
    zero at zero loading (``dlogd_dlnL = n_prior / ln 10``; no acid or anion partial).

    ``tracer_correction`` (default ``True``, integration 2026-09-13) makes the first clause true:
    each record is first lifted to its own tracer limit by ``_tracer_log_d`` so that the depletion
    term is applied once, not twice.  ``tracer_correction=False`` restores the behaviour as first
    written (the R1 evaluation of B1 is unaffected either way: `evalproto` predicts a held-out
    record's log D by the raw 1-NN with no depletion term at all).
    ``ood_distance["nn"]`` of ``evaluate(metal, ...)`` is that metal's 1-NN distance (log10
    units); the stage diagnostics carry the maximum over the stage's metals.  The lookup is
    cached per ``(log10 h, log10 L_T)`` because the solver re-evaluates at the same point while
    it moves the free ligand.  Records: the flat layout of ``domain.py``; rows with
    ``fit_eligible`` false, non-positive acid or ligand, or non-finite log D are dropped.
    Flags carried: ``EQUILIBRIUM_ACID_ASSUMED_NOMINAL`` and ``OA_ASSUMED`` (corpus semantics).
    """

    def __init__(self, records: pd.DataFrame, ligand: LigandSpec | str,
                 domain: ApplicabilityDomain | None, n_prior: float = 3.0, *,
                 mechanism: Mechanism = Mechanism.SOLVATING, p: Any = None, z: Any = None,
                 phase: PhaseBehaviour | None = None, base_flags: Iterable[Flag] = (),
                 tracer_correction: bool = True):
        from .domain import acid_series, ligand_series, metal_mm_series, truthy

        name = _ligand_name(ligand)
        df = records
        if "fit_eligible" in df.columns:
            df = df[truthy(df["fit_eligible"])]
        acid = acid_series(df).to_numpy(dtype=float)
        lig = ligand_series(df, name).to_numpy(dtype=float)
        if "log_d" in df.columns:
            logd = pd.to_numeric(df["log_d"], errors="coerce").to_numpy(dtype=float)
        elif "d" in df.columns:
            dvals = pd.to_numeric(df["d"], errors="coerce").to_numpy(dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                logd = np.log10(dvals)
        else:
            raise ValueError("records carry neither log_d nor d")
        if "metal" not in df.columns:
            raise ValueError("records carry no metal column")
        metal_col = df["metal"].astype(str).to_numpy()
        rid = (df["record_id"].astype(str).to_numpy() if "record_id" in df.columns
               else np.array([str(i) for i in range(len(df))]))
        ok = np.isfinite(acid) & (acid > 0) & np.isfinite(lig) & (lig > 0) & np.isfinite(logd)
        self.n_prior = float(n_prior)
        scale = _ligand_scale(ligand)
        logd_tracer, corrected = _tracer_log_d(logd, lig / scale, metal_mm_series(df), self.n_prior)
        self.tracer_correction = bool(tracer_correction)
        self.n_tracer_corrected = int(corrected[ok].sum()) if tracer_correction else 0
        if tracer_correction:
            logd = logd_tracer
        self._by_metal: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for m in sorted(set(metal_col[ok])):
            sel = ok & (metal_col == m)
            self._by_metal[m] = (np.log10(acid[sel]), np.log10(lig[sel]), logd[sel], rid[sel])
        metals = tuple(self._by_metal)
        self.n_records = int(ok.sum())
        q_d, p_d, z_d = _stoichiometry(ligand, Mechanism(mechanism), self.n_prior)
        prov = Provenance(ProvStatus.MEASURED_CORPUS,
                          Source(kind="corpus", locator="NearestConditionD 1-NN in (log acid, "
                                                        "log ligand); depletion term n_prior"),
                          note=f"n_prior = {self.n_prior}")
        flags = set(base_flags) | {Flag.EQUILIBRIUM_ACID_ASSUMED_NOMINAL, Flag.OA_ASSUMED}
        self._init_common(name, mechanism, metals, self.n_prior, p if p is not None else p_d,
                          z if z is not None else z_d, domain, prov, phase, flags, 0.0,
                          _ligand_scale(ligand))
        self._s_l = np.full(len(metals), self.n_prior / LN10)
        self._last_nn: dict[str, float] = {}
        self._last_record: dict[str, str] = {}
        self._cache_key: tuple[float, float] | None = None
        self._base = np.zeros(len(metals))

    def nearest(self, metal: str, log_acid: float, log_ligand: float) -> tuple[float, float, str]:
        """``(log D, distance, record_id)`` of the same-metal 1-NN of ``(log_acid, log_ligand)``."""
        la, le, logd, rid = self._by_metal[metal]
        dla = la - log_acid
        d2 = dla * dla + (le - log_ligand) ** 2
        best = d2.min()
        cand = np.flatnonzero(d2 <= best * (1.0 + 1e-12) + 1e-300)
        if cand.size > 1:
            order = sorted(cand, key=lambda i: (abs(dla[i]), rid[i]))
            i = int(order[0])
        else:
            i = int(cand[0])
        return float(logd[i]), float(math.sqrt(best)), str(rid[i])

    def core(self, L, h, nu, c, ligand_total, capacity=None):
        la = _log10(h)
        lt = _log10(ligand_total * self.ligand_scale)
        key = (la, lt)
        if key != self._cache_key:
            base = np.empty(len(self.metals))
            for i, m in enumerate(self.metals):
                ld, dist, rid = self.nearest(m, la, lt)
                base[i] = ld
                self._last_record[m] = rid
                self._last_nn[m] = dist
            self._base = base
            self._cache_key = key
        cap = ligand_total if capacity is None else capacity
        shift = self.n_prior * (_log10(L) - _log10(cap))
        return self._base + shift, self._s_l, self._zeros, self._zeros, self._zeros

    def _extra_distance(self, state: StageState, metal: str | None = None) -> dict[str, float]:
        if metal is not None:
            return {"nn": self._last_nn.get(metal, math.nan)}
        vals = [self._last_nn[m] for m in state.x_total if m in self._last_nn]
        return {"nn": max(vals) if vals else math.nan}


# ---------------------------------------------------------------------------------------------
# AqueousComplexantWrapper
# ---------------------------------------------------------------------------------------------

class AqueousComplexantWrapper(_ModelBase):
    """``D_i = D0_i / alpha_i(c)`` around an inner model (DESIGN.md section 5.2 c).

    ``metals`` is the inner model's metal set restricted to the metals the complexant covers;
    stoichiometry, provenance, phase and base flags are the inner model's.  ``complexant`` may be
    a ``ComplexantSpec`` or a ``ComplexantModel``.
    """

    def __init__(self, inner: DModel, complexant: ComplexantSpec | ComplexantModel):
        cm = complexant if isinstance(complexant, ComplexantModel) else ComplexantModel(complexant)
        self.inner = inner
        self.complexant_model = cm
        metals = tuple(m for m in inner.metals if m in cm.metals)
        self._init_common(inner.ligand, inner.mechanism, metals, inner.q, inner.p, inner.z,
                          inner.domain, inner.provenance, getattr(inner, "phase", None),
                          getattr(inner, "base_flags", ()), getattr(inner, "k_acid_uptake", 0.0),
                          getattr(inner, "ligand_scale", 1.0))
        inner_index = {m: i for i, m in enumerate(inner.metals)}
        idx = np.asarray([inner_index[m] for m in metals], dtype=int)
        identity = idx.size == len(inner.metals) and bool(np.all(idx == np.arange(idx.size)))
        self._idx = None if identity else idx
        self._inner_core = getattr(inner, "core", None)

    def core(self, L, h, nu, c, ligand_total, capacity=None):
        if self._inner_core is not None:
            logd, s_l, s_h, s_nu, s_c = self._inner_core(L, h, nu, c, ligand_total, capacity)
        else:
            logd, s_l, s_h, s_nu, s_c = _core_via_evaluate(self.inner, L, h, nu, c, ligand_total,
                                                           capacity)
        if self._idx is not None:
            logd, s_l, s_h, s_nu, s_c = (logd[self._idx], s_l[self._idx], s_h[self._idx],
                                         s_nu[self._idx], s_c[self._idx])
        alpha, phi, _ = self.complexant_model.terms(c, self.metals)
        return logd - np.log10(alpha), s_l, s_h, s_nu, s_c - phi / LN10

    def _extra_flags(self, state: StageState) -> frozenset[Flag]:
        extra = getattr(self.inner, "_extra_flags", None)
        return extra(state) if extra is not None else frozenset()

    def _extra_distance(self, state: StageState, metal: str | None = None) -> dict[str, float]:
        extra = getattr(self.inner, "_extra_distance", None)
        return extra(state, metal) if extra is not None else {}


def _core_via_evaluate(model: DModel, L: float, h: float, nu: float, c: float,
                       ligand_total: float, capacity: float | None = None,
                       ) -> tuple[np.ndarray, ...]:
    """``core`` for a foreign ``DModel``: one ``evaluate`` per metal on a minimal state (the
    foreign model sees only ``ligand_total``; ``capacity`` is accepted for signature parity)."""
    zeros = {m: 0.0 for m in model.metals}
    state = StageState(1.0, 1.0, dict(zeros), dict(zeros), {model.ligand: dict(zeros)}, h, nu, c,
                       {model.ligand: ligand_total}, {model.ligand: L}, {model.ligand: 0.0}, 0.0,
                       25.0)
    n = len(model.metals)
    out = [np.empty(n) for _ in range(5)]
    for i, m in enumerate(model.metals):
        ev = model.evaluate(m, state)
        out[0][i], out[1][i], out[2][i], out[3][i], out[4][i] = (
            ev.log_d, ev.dlogd_dlnL, ev.dlogd_dlnh, ev.dlogd_dlnanion, ev.dlogd_dlnc)
    return tuple(out)


# ---------------------------------------------------------------------------------------------
# 5.6  activity model
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class EffectiveCapacity:
    """Exploratory effective-capacity activity model (DESIGN.md section 5.6).

    The stage ligand balance uses ``phi * L_T`` in place of ``L_T`` (``effective_ligand_total``),
    so the free ligand of a solved state is ``phi L_T - sum q y - a``; ``organic_free_ligand``
    returns that ``state.ligand_free[ligand]``.  ``aqueous_gamma`` is unity.  Used only in the
    exploratory arm of the loading evaluation with ``log K`` re-anchored per ``phi``.
    """

    phi: float

    def __post_init__(self) -> None:
        if not (0.0 < self.phi <= 1.0):
            raise ValueError("EffectiveCapacity.phi must lie in (0, 1]")

    def aqueous_gamma(self, state: StageState) -> dict[str, float]:
        return {m: 1.0 for m in state.x_total}

    def organic_free_ligand(self, state: StageState, ligand: str) -> float:
        return state.ligand_free[ligand]

    def effective_ligand_total(self, ligand_total: float) -> float:
        return self.phi * ligand_total


# ---------------------------------------------------------------------------------------------
# SystemModel and composition (5.3)
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SystemModel:
    """The D models of one system for the stage solver (DESIGN.md section 5.7).

    ``dmodels`` is keyed by ligand name (composition ``D_i = sum_k D_i^(k)``, each ligand depleted
    only by its own ``y^(k)``); ``complexant`` / ``complexant_model`` the aqueous complexant (None
    when absent); ``activity`` the exploratory hook; ``flags`` system-level flags added to every
    stage (``MIXED_ORGANIC_UNMODELLED``, ``OOD_TEMPERATURE``); ``bands`` ligand -> band used.
    """

    entry: SystemEntry | None
    dmodels: Mapping[str, DModel]
    complexant: ComplexantSpec | None
    activity: ActivityModel | None
    temperature_C: float
    assumptions: tuple[str, ...] = ASSUMPTIONS
    complexant_model: ComplexantModel | None = None
    phase: PhaseBehaviour | None = None
    flags: frozenset[Flag] = frozenset()
    params_source: str = ""
    bands: Mapping[str, str] = field(default_factory=dict)

    @property
    def ligands(self) -> tuple[str, ...]:
        return tuple(self.dmodels)

    @property
    def metals(self) -> tuple[str, ...]:
        """Metals every ligand model can evaluate (order of the first model)."""
        models = list(self.dmodels.values())
        if not models:
            return ()
        common = [m for m in models[0].metals if all(m in md.metals for md in models[1:])]
        return tuple(common)

    @property
    def system_id(self) -> str | None:
        return self.entry.system_id if self.entry is not None else None


def loading_fractions(state: StageState, system: SystemModel) -> dict[str, float]:
    """``lambda_k`` per ligand of ``system`` at ``state`` (section 5.4)."""
    return {k: ligand_loading_fraction(state, k, md.q) for k, md in system.dmodels.items()}


def chemistry_flags(state: StageState, system: SystemModel) -> frozenset[Flag]:
    """Union of the section 5.4 flags over the ligands plus the system-level flags."""
    flags: set[Flag] = set(system.flags)
    for k, md in system.dmodels.items():
        flags |= ligand_chemistry_flags(state, k, md.q, getattr(md, "phase", system.phase))
    return frozenset(flags)


def evaluate_composite(system: SystemModel, state: StageState,
                       metals: Iterable[str] | None = None) -> dict[str, DEval]:
    """Effective ``D_i = sum_k D_i^(k)`` per metal with the union of flags and the max distance
    per axis; the partials are the D-weighted means over the ligands (the derivative of the
    composite with respect to a common ln-scaling of every ligand's free concentration)."""
    out: dict[str, DEval] = {}
    for metal in (metals if metals is not None else system.metals):
        d_sum = 0.0
        parts = np.zeros(4)
        flags: set[Flag] = set(system.flags)
        dist: dict[str, float] = {}
        evals = [md.evaluate(metal, state, system.activity) for md in system.dmodels.values()]
        for ev in evals:
            d_sum += ev.d
            flags |= ev.flags
            for k, v in ev.ood_distance.items():
                dist[k] = max(dist.get(k, 0.0), v)
        if d_sum > 0 and math.isfinite(d_sum):
            for ev in evals:
                w = ev.d / d_sum
                parts += w * np.array([ev.dlogd_dlnL, ev.dlogd_dlnh, ev.dlogd_dlnanion,
                                       ev.dlogd_dlnc])
        out[metal] = DEval(log_d=_log10(d_sum) if math.isfinite(d_sum) else math.inf, d=d_sum,
                           dlogd_dlnL=float(parts[0]), dlogd_dlnh=float(parts[1]),
                           dlogd_dlnanion=float(parts[2]), dlogd_dlnc=float(parts[3]),
                           flags=frozenset(flags), ood_distance=dist)
    return out


def tracer_state(system: SystemModel, ligand_total: Mapping[str, float], h: float, anion: float,
                 complexant_total: float = 0.0, *, v_aq_L: float = 1.0, v_org_L: float = 1.0,
                 metals: Iterable[str] | None = None) -> StageState:
    """Zero-loading state: ``L_f = L_T`` (times ``phi`` under ``EffectiveCapacity``), no metal in
    either phase, free deprotonated complexant ``cT / alpha_H(h)``."""
    names = tuple(metals if metals is not None else system.metals)
    zeros = {m: 0.0 for m in names}
    cm = system.complexant_model
    c_free = complexant_total / cm.alpha_h(h) if (cm is not None and complexant_total > 0) else 0.0
    scale = getattr(system.activity, "effective_ligand_total", None)
    free = {k: (scale(v) if scale is not None else v) for k, v in ligand_total.items()}
    return StageState(v_aq_L=v_aq_L, v_org_L=v_org_L, x_total=dict(zeros), y=dict(zeros),
                      y_by_ligand={k: dict(zeros) for k in ligand_total}, h=h, anion=anion,
                      c_free=c_free, ligand_total=dict(ligand_total), ligand_free=free,
                      acid_in_org={k: 0.0 for k in ligand_total}, alkali_reserve=0.0,
                      temperature_C=system.temperature_C)


# ---------------------------------------------------------------------------------------------
# 3.4  temperature bands
# ---------------------------------------------------------------------------------------------

def parse_band(band: str) -> tuple[float, float]:
    """``"<20C"`` -> (-inf, 20); ``"20-30C"`` -> (20, 30); ``">=50C"`` -> (50, inf)."""
    from .domain import _parse_band

    return _parse_band(band)


def select_band(bands: Iterable[str], temperature_C: float) -> tuple[str | None, bool]:
    """``(band, inside)``: the band containing ``temperature_C`` (``lo <= T < hi``), else the
    nearest band with ``inside = False``; ``(None, False)`` when there is no band."""
    names = list(bands)
    if not names:
        return None, False
    best, best_gap = None, math.inf
    for name in names:
        lo, hi = parse_band(name)
        if lo <= temperature_C < hi:
            return name, True
        gap = lo - temperature_C if temperature_C < lo else temperature_C - hi
        if gap < best_gap:
            best, best_gap = name, gap
    return best, False


# ---------------------------------------------------------------------------------------------
# 5.7  build_system_model
# ---------------------------------------------------------------------------------------------

def _params_from_json(obj: Any) -> CationExchangeParams | SolvatingParams:
    if isinstance(obj, (CationExchangeParams, SolvatingParams)):
        return obj
    if not isinstance(obj, Mapping):
        raise ValueError("parameter block must be a params dataclass or a mapping")

    def s(key: str, required: bool = True) -> Sourced | None:
        val = obj.get(key)
        if val is None:
            if required:
                raise ValueError(f"parameter block lacks {key!r}")
            return None
        return val if isinstance(val, Sourced) else Sourced.from_json(val)

    log_k = {m: (v if isinstance(v, Sourced) else Sourced.from_json(v))
             for m, v in (obj.get("log_k") or {}).items()}
    model_type = obj.get("model_type", "solvating")
    if model_type == "cation_exchange":
        beta = obj.get("beta_anion")
        beta_anion = None
        if beta:
            beta_anion = {m: tuple(v if isinstance(v, Sourced) else Sourced.from_json(v)
                                   for v in vals) for m, vals in beta.items()}
        sap = obj.get("saponification_degree_studied")
        return CationExchangeParams(
            medium_anion=str(obj["medium_anion"]), temperature_band=str(obj["temperature_band"]),
            log_k=log_k, a_dimer=s("a_dimer"), b_proton=s("b_proton"),
            k_reported_as=str(obj.get("k_reported_as", "log_k")), beta_anion=beta_anion,
            saponification_degree_studied=None if sap is None else (float(sap[0]), float(sap[1])),
            delta_h_kj_mol=s("delta_h_kj_mol", required=False))
    return SolvatingParams(
        medium_anion=str(obj["medium_anion"]), temperature_band=str(obj["temperature_band"]),
        log_k=log_k, n_solvation=s("n_solvation"), p_anion=s("p_anion"),
        p_h=s("p_h", required=False) or Sourced(0.0, "1", Provenance(
            ProvStatus.ASSUMED, Source(kind="none"), (0.0, 0.0), ASSUMED_LABEL, note="p_h = 0")),
        k_acid_uptake=s("k_acid_uptake", required=False) or Sourced.unknown("L2/mol2"),
        delta_h_kj_mol=s("delta_h_kj_mol", required=False))


def _drawn(sourced: Sourced, value: float, key: str) -> Sourced:
    """``sourced`` with ``value`` substituted; the value must lie inside the declared range."""
    rng = sourced.range
    if rng is None:
        raise ValueError(f"parameter_draw[{key!r}]: the parameter declares no range")
    lo, hi = float(rng[0]), float(rng[1])
    if not (lo <= value <= hi):
        raise ValueError(f"parameter_draw[{key!r}] = {value} outside the declared range "
                         f"[{lo}, {hi}]")
    prov = replace(sourced.provenance,
                   note=(sourced.provenance.note + " | " if sourced.provenance.note else "")
                   + f"parameter_draw {key} = {value!r}")
    return Sourced(float(value), sourced.unit, prov)


_SCALAR_PARAMS = ("a_dimer", "b_proton", "n_solvation", "p_anion", "p_h", "k_acid_uptake")


def _apply_ligand_draw(params: CationExchangeParams | SolvatingParams, ligand: str,
                       draw: Mapping[str, float], used: set[str],
                       ) -> CationExchangeParams | SolvatingParams:
    prefix = ligand + "."
    changes: dict[str, Any] = {}
    log_k = dict(params.log_k)
    for key, value in draw.items():
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix):]
        if rest.startswith("log_k."):
            metal = rest[len("log_k."):]
            if metal not in log_k:
                raise ValueError(f"parameter_draw[{key!r}]: no log_k entry for {metal!r}")
            log_k[metal] = _drawn(log_k[metal], float(value), key)
            changes["log_k"] = log_k
        elif rest in _SCALAR_PARAMS and hasattr(params, rest):
            changes[rest] = _drawn(getattr(params, rest), float(value), key)
        else:
            raise ValueError(f"parameter_draw[{key!r}]: unknown parameter for ligand {ligand!r}")
        used.add(key)
    return replace(params, **changes) if changes else params


def _apply_complexant_draw(spec: ComplexantSpec, draw: Mapping[str, float], used: set[str],
                           ) -> ComplexantSpec:
    prefix = spec.name + "."
    log_beta = {m: list(v) for m, v in spec.log_beta.items()}
    prot = list(spec.protonation_logk)
    regen = spec.regeneration_fraction
    touched = False
    for key, value in draw.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix):].split(".")
        if parts[0] == "log_beta" and len(parts) in (2, 3):
            metal = parts[1]
            m = int(parts[2]) if len(parts) == 3 else 1
            if metal not in log_beta or not (1 <= m <= len(log_beta[metal])):
                raise ValueError(f"parameter_draw[{key!r}]: no log_beta_{m} for {metal!r}")
            log_beta[metal][m - 1] = _drawn(log_beta[metal][m - 1], float(value), key)
        elif parts[0] == "protonation_logk" and len(parts) == 2:
            j = int(parts[1])
            if not (1 <= j <= len(prot)):
                raise ValueError(f"parameter_draw[{key!r}]: no protonation constant {j}")
            prot[j - 1] = _drawn(prot[j - 1], float(value), key)
        elif parts[0] == "regeneration_fraction" and len(parts) == 1:
            regen = _drawn(regen, float(value), key)
        else:
            raise ValueError(f"parameter_draw[{key!r}]: unknown complexant parameter")
        used.add(key)
        touched = True
    if not touched:
        return spec
    return replace(spec, log_beta={m: tuple(v) for m, v in log_beta.items()},
                   protonation_logk=tuple(prot), regeneration_fraction=regen)


def _decision_adopted_from_results() -> bool:
    """``results/eval/decision.json`` key ``m1_adopted`` (False when absent or unreadable)."""
    try:
        from . import paths

        path = paths.RESULTS_EVAL_DIR / "decision.json"
        if not path.exists():
            return False
        with path.open("r", encoding="utf-8") as fh:
            return bool(json.load(fh).get("m1_adopted", False))
    except (OSError, ValueError, ImportError):
        return False


def _block_is_fitted(block: Any) -> bool:
    log_k = block.log_k if isinstance(block, (CationExchangeParams, SolvatingParams)) else (
        block.get("log_k") or {})
    statuses = []
    for v in log_k.values():
        st = _status_of(v) if isinstance(v, Sourced) else ProvStatus(v.get("status", "unknown"))
        statuses.append(st)
    return bool(statuses) and all(s == ProvStatus.FITTED_FROM_CORPUS for s in statuses)


def build_system_model(entry: SystemEntry, *, feed_anion: str, temperature_C: float,
                       params_source: str = "auto", complexant: ComplexantSpec | None = None,
                       activity: ActivityModel | None = None,
                       parameter_draw: Mapping[str, float] | None = None,
                       feed_metals: Iterable[str] | None = None,
                       decision_adopted: bool | None = None, n_prior: float = 3.0,
                       ) -> SystemModel | ModelBuildError:
    """Assemble the ``SystemModel`` of ``entry`` (DESIGN.md section 5.7).

    Refuses (returns ``ModelBuildError``) on cross-anion transfer, a missing ``log_k`` for a feed
    metal, a complexant without ``log_beta`` for a feed metal, or no usable parameter source.
    ``params_source``: ``"literature"`` / ``"fitted"`` (the entry's parameter block, the latter
    requiring ``fitted_from_corpus`` log K), ``"nearest"`` (``NearestConditionD`` on the entry's
    records), ``"auto"`` (fitted when present and the pre-registered decision adopted it —
    ``decision_adopted``, or ``results/eval/decision.json`` key ``m1_adopted`` when None — else
    nearest for corpus systems, literature for literature systems).  ``parameter_draw`` maps
    ``"<ligand>.<param>[.<metal>]"`` (and ``"<complexant>.log_beta.<metal>[.<m>]"``,
    ``"<complexant>.protonation_logk.<j>"``, ``"<complexant>.regeneration_fraction"``) to values
    inside the declared ranges (``ValueError`` otherwise).  ``feed_metals`` (optional) names the
    metals that must be parameterised; default: every metal with a log K in the first ligand.
    The band containing ``temperature_C`` is used; when absent the nearest band with
    ``OOD_TEMPERATURE``.
    """
    if feed_anion != entry.medium.anion:
        return ModelBuildError(f"feed anion {feed_anion!r} differs from the system medium "
                               f"{entry.medium.anion!r}: cross-anion transfer refused")
    ligands = [lig for lig in entry.organic_ligands
               if lig.role in ("extractant", "synergist") and lig.mechanism is not None]
    if not ligands:
        return ModelBuildError("the entry has no extractant-role ligand with a mechanism")
    draw = dict(parameter_draw or {})
    used: set[str] = set()
    sys_flags: set[Flag] = set()
    if sum(1 for lig in ligands if lig.role == "extractant") >= 2:
        sys_flags.add(Flag.MIXED_ORGANIC_UNMODELLED)

    if complexant is None:
        if len(entry.aqueous_complexants) == 1:
            complexant = entry.aqueous_complexants[0]
        elif len(entry.aqueous_complexants) > 1:
            return ModelBuildError("several aqueous complexants are not supported; pass one")
    cm: ComplexantModel | None = None
    if complexant is not None:
        complexant = _apply_complexant_draw(complexant, draw, used)
        cm = ComplexantModel(complexant)

    if params_source not in ("auto", "literature", "fitted", "nearest"):
        raise ValueError(f"unknown params_source {params_source!r}")
    adopted = decision_adopted
    dmodels: dict[str, DModel] = {}
    bands: dict[str, str] = {}
    sources: list[str] = []
    for lig in ligands:
        mech = Mechanism(lig.mechanism)
        blocks = dict(entry.params.get(lig.name) or {})
        fitted_bands = [b for b, blk in blocks.items() if _block_is_fitted(blk)]
        source = params_source
        if source == "auto":
            if entry.origin == "corpus":
                if fitted_bands:
                    if adopted is None:
                        adopted = _decision_adopted_from_results()
                    source = "fitted" if adopted else "nearest"
                else:
                    source = "nearest"
            elif entry.origin == "literature":
                source = "literature"
            else:
                source = "literature" if blocks else "nearest"
        candidates = fitted_bands if source == "fitted" else list(blocks)
        band, inside = select_band(candidates, temperature_C)
        base_flags: set[Flag] = set()
        if band is not None and not inside:
            base_flags.add(Flag.OOD_TEMPERATURE)
        domain = coerce_domain(entry.applicability.get(f"{lig.name}|{band}")) if band else None
        if source in ("literature", "fitted"):
            if band is None:
                return ModelBuildError(f"no {source} parameter set for ligand {lig.name!r}")
            params = _params_from_json(blocks[band])
            if params.medium_anion != entry.medium.anion:
                return ModelBuildError(f"parameter set of {lig.name!r} is for "
                                       f"{params.medium_anion!r}, medium is {entry.medium.anion!r}")
            params = _apply_ligand_draw(params, lig.name, draw, used)
            try:
                if mech == Mechanism.CATION_EXCHANGE:
                    if not isinstance(params, CationExchangeParams):
                        return ModelBuildError(f"{lig.name!r} is cation exchange but its "
                                               "parameter block is solvating")
                    model: DModel = CationExchangeMassAction(params, lig, entry.medium, domain,
                                                             phase=entry.phase,
                                                             base_flags=base_flags)
                elif mech == Mechanism.SOLVATING:
                    if not isinstance(params, SolvatingParams):
                        return ModelBuildError(f"{lig.name!r} is solvating but its parameter "
                                               "block is cation exchange")
                    model = SolvatingMassAction(params, lig, entry.medium, domain,
                                                phase=entry.phase, base_flags=base_flags)
                else:
                    return ModelBuildError(f"mechanism {mech.value} has no D model")
            except ValueError as exc:
                return ModelBuildError(f"{lig.name}: {exc}")
        else:
            frame = records_to_frame(entry.records)
            if frame.empty:
                return ModelBuildError(f"no records for NearestConditionD on {lig.name!r}")
            if domain is None:
                try:
                    family = str(entry.diluent.get("family", "unknown"))
                    domain = build_domain(frame, lig.name, anion=entry.medium.anion,
                                          diluent_family=family, modifiers=(), n_prior=n_prior)
                except ValueError as exc:
                    return ModelBuildError(f"{lig.name}: {exc}")
            try:
                model = NearestConditionD(frame, lig, domain, n_prior, mechanism=mech,
                                          phase=entry.phase, base_flags=base_flags)
            except ValueError as exc:
                return ModelBuildError(f"{lig.name}: {exc}")
            bands[lig.name] = band or "records"
        if band is not None:
            bands[lig.name] = band
        if cm is not None:
            model = AqueousComplexantWrapper(model, cm)
        dmodels[lig.name] = model
        sources.append(source)

    unused = set(draw) - used
    if unused:
        raise ValueError(f"parameter_draw keys not matching any ligand or complexant: "
                         f"{sorted(unused)}")

    first = dmodels[ligands[0].name]
    required = tuple(feed_metals) if feed_metals is not None else (
        first.inner.metals if isinstance(first, AqueousComplexantWrapper) else first.metals)
    for name, model in dmodels.items():
        base = model.inner if isinstance(model, AqueousComplexantWrapper) else model
        missing = [m for m in required if m not in base.metals]
        if missing:
            return ModelBuildError(f"ligand {name!r} has no log_k for {missing}")
    if cm is not None:
        missing = [m for m in required if m not in cm.metals]
        if missing:
            return ModelBuildError(f"complexant {cm.name!r} lacks log_beta for {missing}")

    return SystemModel(entry=entry, dmodels=dmodels, complexant=complexant, activity=activity,
                       temperature_C=float(temperature_C), assumptions=ASSUMPTIONS,
                       complexant_model=cm, phase=entry.phase, flags=frozenset(sys_flags),
                       params_source="|".join(sorted(set(sources))), bands=bands)
