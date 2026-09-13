"""``cascade.py`` — countercurrent extraction / scrub / strip cascade with organic recycle
(DESIGN.md section 7).

Topology (7.2): ``N = n_ext + n_scr + n_str`` stages ``j = 0..N-1``; the organic flows toward
increasing ``j``, the aqueous toward decreasing ``j``.  Two aqueous circuits: A (extraction +
scrub, stages ``[0, n_ext + n_scr)``; the scrub liquor enters the top scrub stage, the feed enters
``feed_stage``, the scrub raffinate leaving stage ``n_ext`` re-enters at ``scrub_return_stage``,
the aqueous leaving stage 0 is the raffinate) and B (strip, stages ``[n_ext + n_scr, N)``; the
strip liquor enters stage ``N - 1`` and the aqueous leaving stage ``n_ext + n_scr`` is the
product).  The organic inlet of stage 0 is ``(1 - f_bleed) * org_out[N-1] + f_bleed *
fresh_organic`` with the alkali reserve re-created (``B = saponification_degree * [HA]_T`` on the
monomer basis of the acid-releasing ligands).  Per-stage aqueous flow ``A_j`` is the sum of the
external aqueous flows that reach ``j`` through the incidence matrix (7.4); a stage that no
aqueous stream reaches is a malformed specification (``ValueError``).

Unknowns (7.3): per stage ``x_j,i`` (aqueous total metal, M), ``L_j^(k)`` (free ligand, K),
``h_j``, ``nu_j``, ``c_j``, ``B_j`` -> ``M + K + 4`` per stage, ``N (M + K + 4)`` in all.  Every
residual is written in mol/h and scaled as section 7.3 states (metal rows by the metal's external
input, ligand rows by ``O L_T``, acid and anion rows by the largest external acid load, reserve
rows by ``O B_in,0``, complexant rows by the external complexant load):

    metal i:     A_j x_j,i + O y_j,i - sum_m Inc[j,m] A_m x_m,i - O y_(j-1),i - ext_j,i
    ligand k:    O [ L_j^(k) (1 + KH_k h_j nu_j) + sum_i q_i^(k) y_j,i^(k) - L_T^(k) ]
    acid:        branch 1:  A_j [ h_j - (P_j - r_j B_(j-1)) ] ;  branch 2:  A_j (h_j - H_MIN)
    reserve:     branch 1:  O B_j ;  branch 2:  O (B_j - B_(j-1)) + A_j max(0, P_j - H_MIN)
    anion:       A_j (nu_j - nu_in,j) + O sum_k sum_i z_i^(k) (y_j,i^(k) - y_(j-1),i^(k))
                 + O uptake_j
    complexant:  A_j [ c_j alpha_H(h_j) + sum_i x_j,i phi_i(c_j) - cT_j ]      (A_j c_j without one)

with ``y_j,i^(k) = D_j,i^(k) x_j,i``, ``P_j = h_in,j + r_j (released_j - uptake_j)``,
``released_j = sum_k sum_i p_i^(k) (y_j,i^(k) - y_(j-1),i^(k))``, ``uptake_j = sum_k (KH_k h_j
nu_j L_j^(k) - a_(j-1)^(k))``, ``r_j = O / A_j``; the branch rule is the stage solver's (branch 1
iff ``P_j - r_j B_(j-1) >= H_MIN``, addendum WB2 A5.1) so that the Newton path and the
successive-substitution path solve the same equations.  The complexant total ``cT_j`` and the
sodium are fixed by the topology alone (the complexant stays aqueous) and are not unknowns.

Solvers (7.5-7.7): ``kremser_init`` (every metal's D at the tracer conditions of its section,
metal balances linear in ``x`` and solved directly, exact for ``ConstantD``), damped Newton with
the analytic block Jacobian (stage blocks plus the aqueous, organic, recycle and internal-return
couplings; fraction-to-boundary 0.95 and Armijo backtracking), and successive substitution over
``equilibrium.solve_stage`` with relaxation and bounded Wegstein on the recycle tear as the
fallback and the cross-check.  Failure is a status (``failed``), never an exception;
``ValueError`` only on a malformed specification (7.1).  ``check_balances`` recomputes every
ledger from the stream table alone (7.9); the origin-labelled pass (7.8) is an exact linear pass
at the converged D.  Units: mol/L in the phase named, flows L/h, ``log`` means log10.

Sources: DESIGN.md section 7; addendum WB2 (stage solver API, branch rule); addendum WB3 (the
deviations recorded there).
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .dmodel import (
    LN10,
    LOG10_CAP,
    SystemModel,
    _core_via_evaluate,
    _ModelBase,
    ligand_loading_fraction,
    tracer_state,
)
from .equilibrium import H_MIN, NU_MIN, solve_stage, stage_state
from .types import (
    INADMISSIBLE_FLAGS,
    OOD_FLAGS,
    AqStream,
    CascadeResult,
    CascadeSpec,
    Flag,
    Mechanism,
    OrgStream,
    StageDiagnostics,
)

__all__ = [
    "solve_cascade", "solve_cascade_ss", "kremser_init", "kremser_fraction_unextracted",
    "check_balances", "regime_status_of", "section_tracer_d", "alkali_reserve_inlet",
    "origin_pass", "Topology", "build_topology", "ORIGIN_LABELS", "SECTIONS",
]

ORIGIN_LABELS: tuple[str, ...] = ("feed", "scrub", "fresh", "strip")
"""Labels of the origin pass (7.8): the design's three plus ``strip`` so the label sums always
equal the totals even when the strip liquor carries the target (addendum WB3)."""

SECTIONS: tuple[str, ...] = ("extraction", "scrub", "strip")

_DENSE_MAX = 100            # numpy.linalg.solve up to this many unknowns, SuperLU beyond (WB3:
#                             measured 0.07 ms dense at n = 84 but 4.7 ms at n = 168 against
#                             0.4-1.3 ms sparse; the design's 600 was not the faster choice)
_C_TINY = 1e-200            # free complexant used for the derivative limits at c = 0
_INIT_REFINE_ITERS = 12     # constant-D relinearisation rounds after the Kremser init
_BOUND_TOL = 1e-12          # an unknown this close (relative) to its bound counts as on it
_LOG_STEP_CAP = 3.0         # largest change of a logarithmic unknown per Newton step (e^3)
_JAM_RATIO = 0.1            # below this fraction of the step the limiting unknowns are landed
_PRESWEEPS = (3, 10)        # exact stage sweeps before the second / third Newton attempt
_SS_STAGE_SOLVE_BUDGET = 20000   # stage solves the auto-mode SS fallback may spend
_SS_MIN_SWEEPS = 100
_ARMIJO = 1e-4
_MAX_BACKTRACK = 10
_TAU = 0.95                 # fraction to boundary (7.5)
_BRANCH_CYCLE = 5
_SS_START_WEGSTEIN = 5
_SS_TOL = 1e-11
_SS_BALANCE_TOL = 1e-10


# =============================================================================================
# 7.2 / 7.4  topology
# =============================================================================================

@dataclass(frozen=True)
class Topology:
    """Stage indexing, incidence and per-stage aqueous flows of one specification (7.2, 7.4).

    ``inc[j, m] = 1`` when the aqueous outlet of stage ``m`` enters stage ``j``; ``ext_flow[j]``
    is the external aqueous flow injected at ``j``; ``a_flow[j]`` the aqueous flow of stage ``j``
    (L/h); ``upstream[j]`` the tuple of ``m`` with ``inc[j, m] = 1``; ``aq_order`` the stages in
    aqueous-topological order (every stage after all of its upstream stages).
    """

    n_ext: int
    n_scr: int
    n_str: int
    feed_stage: int
    scrub_return_stage: int
    n_stages: int
    inc: np.ndarray
    ext_flow: np.ndarray
    a_flow: np.ndarray
    upstream: tuple[tuple[int, ...], ...]
    aq_order: tuple[int, ...]
    section: tuple[str, ...]
    feed_flow: float
    scrub_flow: float
    strip_flow: float

    @property
    def scrub_top(self) -> int | None:
        return self.n_ext + self.n_scr - 1 if self.n_scr > 0 else None

    @property
    def strip_top(self) -> int | None:
        return self.n_stages - 1 if self.n_str > 0 else None

    @property
    def product_stage(self) -> int | None:
        return self.n_ext + self.n_scr if self.n_str > 0 else None


def build_topology(spec: CascadeSpec) -> Topology:
    """Incidence matrix, external injections and per-stage aqueous flows (7.2, 7.4).

    Raises ``ValueError`` on ``n_ext < 1``, negative section counts, stage indices outside
    ``[0, n_ext - 1]``, non-positive flows, or a stage that no aqueous stream reaches.
    """
    n_ext, n_scr, n_str = int(spec.n_ext), int(spec.n_scr), int(spec.n_str)
    if n_ext < 1:
        raise ValueError(f"n_ext must be >= 1, got {n_ext}")
    if n_scr < 0 or n_str < 0:
        raise ValueError("n_scr and n_str must be >= 0")
    n = n_ext + n_scr + n_str
    j_f = n_ext - 1 if spec.feed_stage is None else int(spec.feed_stage)
    j_r = n_ext - 1 if spec.scrub_return_stage is None else int(spec.scrub_return_stage)
    if not (0 <= j_f < n_ext):
        raise ValueError(f"feed_stage {j_f} outside [0, {n_ext - 1}]")
    if not (0 <= j_r < n_ext):
        raise ValueError(f"scrub_return_stage {j_r} outside [0, {n_ext - 1}]")
    A = float(spec.feed.flow_L_h)
    S = float(spec.scrub.flow_L_h) if n_scr > 0 else 0.0
    W = float(spec.strip.flow_L_h) if n_str > 0 else 0.0
    if not (math.isfinite(A) and A > 0):
        raise ValueError(f"feed flow must be positive, got {A!r}")
    if n_scr > 0 and not (math.isfinite(S) and S > 0):
        raise ValueError(f"scrub flow must be positive when n_scr > 0, got {S!r}")
    if n_str > 0 and not (math.isfinite(W) and W > 0):
        raise ValueError(f"strip flow must be positive when n_str > 0, got {W!r}")
    inc = np.zeros((n, n), dtype=int)
    top_a = n_ext + n_scr
    for j in range(top_a - 1):
        if j == n_ext - 1 and n_scr > 0 and j_r != n_ext - 1:
            continue
        inc[j, j + 1] = 1
    if n_scr > 0 and j_r != n_ext - 1:
        inc[j_r, n_ext] = 1
    for j in range(top_a, n - 1):
        inc[j, j + 1] = 1
    ext = np.zeros(n)
    ext[j_f] += A
    if n_scr > 0:
        ext[top_a - 1] += S
    if n_str > 0:
        ext[n - 1] += W
    a_flow = np.zeros(n)
    order = list(range(n - 1, -1, -1))          # every link points to a smaller index
    for j in order:
        a_flow[j] = ext[j] + float(inc[j] @ a_flow)
    dry = [j for j in range(n) if not a_flow[j] > 0]
    if dry:
        raise ValueError(f"stages {dry} receive no aqueous flow (feed_stage={j_f}, "
                         f"scrub_return_stage={j_r}, n_scr={n_scr})")
    upstream = tuple(tuple(int(m) for m in np.flatnonzero(inc[j])) for j in range(n))
    section = tuple(("extraction" if j < n_ext else "scrub" if j < top_a else "strip")
                    for j in range(n))
    return Topology(n_ext, n_scr, n_str, j_f, j_r, n, inc, ext, a_flow, upstream, tuple(order),
                    section, A, S, W)


# =============================================================================================
# problem data
# =============================================================================================

def _check_stream(stream: AqStream | OrgStream, name: str, metals: tuple[str, ...]) -> None:
    unknown = [m for m in stream.metals if m not in metals]
    if unknown:
        raise ValueError(f"{name} carries metals {unknown} that the system does not parameterise")
    for m, v in stream.metals.items():
        if not (math.isfinite(v) and v >= 0):
            raise ValueError(f"{name}.metals[{m!r}] must be finite and non-negative, got {v!r}")
    if isinstance(stream, AqStream):
        for label, v in (("h", stream.h), ("anion", stream.anion),
                         ("complexant_total", stream.complexant_total),
                         ("sodium", stream.sodium)):
            if not (math.isfinite(v) and v >= 0):
                raise ValueError(f"{name}.{label} must be finite and non-negative, got {v!r}")
        if not (math.isfinite(stream.flow_L_h) and stream.flow_L_h >= 0):
            raise ValueError(f"{name}.flow_L_h must be finite and non-negative")


def _safe_div(num: np.ndarray, den: float) -> np.ndarray:
    """``num / den`` with 0 where ``den == 0`` (the anion at its bound in an anion-free
    medium, where the numerator is 0 as well)."""
    return num / den if den > 0.0 else np.zeros_like(num)


def _ligand_scale(model: Any) -> float:
    return float(getattr(model, "ligand_scale", 1.0) or 1.0)


def _releases_acid(model: Any) -> bool:
    if getattr(model, "mechanism", None) == Mechanism.CATION_EXCHANGE:
        return True
    return any(float(v) > 0 for v in getattr(model, "p", {}).values())


def alkali_reserve_inlet(spec: CascadeSpec, system: SystemModel) -> float:
    """``B_in,0 = saponification_degree * [HA]_T`` (mol/L organic), with ``[HA]_T`` the formal
    monomer concentration of the acid-releasing ligands (``ligand_scale * L_T``: 2 L_T for a
    dimeric ligand); 0 for a purely solvating system (DESIGN.md 7.1 / 7.2)."""
    ha_total = 0.0
    for name, md in system.dmodels.items():
        if _releases_acid(md) and name in spec.ligand_total:
            ha_total += _ligand_scale(md) * float(spec.ligand_total[name])
    return float(spec.saponification_degree) * ha_total


class _Problem:
    """Validated specification plus the arrays the solvers share (7.1 validation, 7.3 layout)."""

    def __init__(self, spec: CascadeSpec, system: SystemModel):
        self.spec = spec
        self.system = system
        self.top = top = build_topology(spec)
        N = top.n_stages
        O = float(spec.organic_flow_L_h)
        if not (math.isfinite(O) and O > 0):
            raise ValueError(f"organic flow must be positive, got {O!r}")
        if not (0.0 <= float(spec.saponification_degree) <= 1.0):
            raise ValueError("saponification_degree must lie in [0, 1]")
        f_bleed = float(spec.f_bleed)
        if not (0.0 <= f_bleed <= 1.0):
            raise ValueError("f_bleed must lie in [0, 1]")
        if top.n_str == 0:
            f_bleed = 1.0                       # no recycle without a strip section (7.2)
        self.f_bleed = f_bleed
        self.O = O
        self.A = top.a_flow
        self.r = O / top.a_flow
        self.ligands = tuple(system.dmodels)
        self.models = [system.dmodels[k] for k in self.ligands]
        K = len(self.models)
        if K == 0:
            raise ValueError("the system model has no ligand model")
        sys_metals = tuple(system.metals)
        if not sys_metals:
            raise ValueError("the system model parameterises no metal")
        # streams and metals
        fresh = spec.fresh_organic
        streams: list[tuple[str, Any]] = [("feed", spec.feed), ("scrub", spec.scrub),
                                          ("strip", spec.strip)]
        if fresh is not None:
            streams.append(("fresh_organic", fresh))
        for name, st in streams:
            _check_stream(st, name, sys_metals)
        if spec.target and spec.target not in sys_metals:
            raise ValueError(f"target {spec.target!r} is not parameterised by the system")
        declared = {m for _, st in streams for m in st.metals}
        self.all_metals = tuple(m for m in sys_metals if m in declared) or sys_metals[:1]
        # ligand totals
        LT = []
        for name in self.ligands:
            if name not in spec.ligand_total:
                raise ValueError(f"ligand_total lacks ligand {name!r}")
            lt = float(spec.ligand_total[name])
            if not (math.isfinite(lt) and lt > 0):
                raise ValueError(f"ligand_total[{name!r}] must be positive, got {lt!r}")
            LT.append(lt)
        self.LT = np.array(LT)
        scale_fn = getattr(system.activity, "effective_ligand_total", None)
        self.LT_eff = (np.array([float(scale_fn(v)) for v in LT]) if scale_fn is not None
                       else self.LT.copy())
        self.KH = np.array([float(getattr(md, "k_acid_uptake", 0.0) or 0.0) for md in self.models])
        # external inputs per stage (mol/h), over the declared metals
        A, S, W = top.feed_flow, top.scrub_flow, top.strip_flow
        M_all = len(self.all_metals)
        ext_metal = np.zeros((N, M_all))
        for i, m in enumerate(self.all_metals):
            ext_metal[top.feed_stage, i] += A * float(spec.feed.metals.get(m, 0.0))
            if top.n_scr > 0:
                ext_metal[top.scrub_top, i] += S * float(spec.scrub.metals.get(m, 0.0))
            if top.n_str > 0:
                ext_metal[top.strip_top, i] += W * float(spec.strip.metals.get(m, 0.0))
        y_fresh_all = np.zeros((K, M_all))
        a_fresh = np.zeros(K)
        if fresh is not None:
            if K == 1:
                sub = fresh.metals
                y_fresh_all[0] = [float(sub.get(m, 0.0)) for m in self.all_metals]
            else:
                for k, name in enumerate(self.ligands):
                    sub = fresh.metals_by_ligand.get(name, {})
                    y_fresh_all[k] = [float(sub.get(m, 0.0)) for m in self.all_metals]
            a_fresh = np.array([float(fresh.acid_in_org.get(name, 0.0)) for name in self.ligands])
        total_in = ext_metal.sum(axis=0) + f_bleed * O * y_fresh_all.sum(axis=0)
        active = [i for i in range(M_all) if total_in[i] > 0]
        if not active:
            active = [0]
        self.metals = tuple(self.all_metals[i] for i in active)
        self.M = M = len(self.metals)
        self.K = K
        self.ext_metal = ext_metal[:, active]
        self.y_fresh = y_fresh_all[:, active]
        self.a_fresh = a_fresh
        self.metal_scale = np.maximum(total_in[active], 1e-30 * O)
        self.conc_scale = self.metal_scale / A
        if spec.feed.complexant_total > 0 or spec.scrub.complexant_total > 0 or \
                spec.strip.complexant_total > 0:
            if system.complexant_model is None:
                raise ValueError("a liquor carries a complexant but the system has no "
                                 "ComplexantSpec")
        self.cm = system.complexant_model
        # stoichiometry and cores
        self.q = np.zeros((K, M))
        self.p = np.zeros((K, M))
        self.z = np.zeros((K, M))
        self.cores: list[Any] = []
        self.idx: list[np.ndarray | None] = []
        for k, md in enumerate(self.models):
            if isinstance(md, _ModelBase):
                q, p, z, index = md.stage_arrays(self.metals)
                self.cores.append(md.core)
            else:
                q = np.array([float(md.q.get(m, 0.0)) for m in self.metals])
                p = np.array([float(md.p.get(m, 0.0)) for m in self.metals])
                z = np.array([float(md.z.get(m, 0.0)) for m in self.metals])
                pos = [md.metals.index(m) for m in self.metals]
                index = None if pos == list(range(len(md.metals))) else np.asarray(pos)
                core = getattr(md, "core", None)
                self.cores.append(core if core is not None else (
                    lambda L, h, nu, c, lt, cap, _md=md: _core_via_evaluate(_md, L, h, nu, c, lt,
                                                                            cap)))
            self.q[k], self.p[k], self.z[k] = q, p, z
            self.idx.append(index)
        if self.cm is not None:
            missing = [m for m in self.metals if m not in self.cm.metals]
            if missing:
                raise ValueError(f"complexant lacks log_beta for {missing}")
        self.has_c = self.cm is not None and (spec.feed.complexant_total > 0
                                              or spec.scrub.complexant_total > 0
                                              or spec.strip.complexant_total > 0)
        # external acid / anion / complexant / sodium injections (mol/h)
        self.ext_h = np.zeros(N)
        self.ext_nu = np.zeros(N)
        self.ext_cT = np.zeros(N)
        self.ext_na = np.zeros(N)
        for st, j, flow in ((spec.feed, top.feed_stage, A), (spec.scrub, top.scrub_top, S),
                            (spec.strip, top.strip_top, W)):
            if j is None or flow <= 0:
                continue
            self.ext_h[j] += flow * st.h
            self.ext_nu[j] += flow * st.anion
            self.ext_cT[j] += flow * st.complexant_total
            self.ext_na[j] += flow * st.sodium
        self.cT = self._topological_pass(self.ext_cT)         # complexant total per stage, mol/L
        self.B0 = alkali_reserve_inlet(spec, system)
        # residual scales (7.3)
        acid_scale = max(A * spec.feed.h, S * spec.scrub.h, W * spec.strip.h, 1e-3)
        self.scale_lig = O * self.LT_eff
        self.scale_h = acid_scale
        self.scale_nu = max(A * spec.feed.anion, S * spec.scrub.anion, W * spec.strip.anion,
                            acid_scale)
        self.scale_B = max(O * self.B0, 1e-6)
        self.scale_c = (A * spec.feed.complexant_total + S * spec.scrub.complexant_total
                        + W * spec.strip.complexant_total + 1e-12)
        self.ns = M + K + 4
        self.n = N * self.ns
        self.iL, self.ih, self.inu, self.ic, self.iB = M, M + K, M + K + 1, M + K + 2, M + K + 3
        self.zero_m = np.zeros(M)
        self.inc_w = top.inc.astype(float) * top.a_flow[None, :]     # A_m on every link j <- m
        self.links = [(j, m) for j in range(N) for m in top.upstream[j]]

    # -- helpers ---------------------------------------------------------------------------
    def _topological_pass(self, ext_mol_h: np.ndarray) -> np.ndarray:
        """Per-stage concentration of a conserved aqueous species from its external injections
        (mol/h): ``conc_j = (sum_m Inc[j,m] A_m conc_m + ext_j) / A_j``."""
        top = self.top
        conc = np.zeros(top.n_stages)
        for j in top.aq_order:
            s = ext_mol_h[j]
            for m in top.upstream[j]:
                s += top.a_flow[m] * conc[m]
            conc[j] = s / top.a_flow[j]
        return conc

    def inlet_h_nu(self, h: np.ndarray, nu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``h_in,j`` and ``nu_in,j`` (mol/L) from the stage values (7.3)."""
        A = self.top.a_flow
        return (self.inc_w @ h + self.ext_h) / A, (self.inc_w @ nu + self.ext_nu) / A

    def d_arrays(self, L: np.ndarray, h: float, nu: float, c: float,
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(D, dD/dlnL, dD/dlnh, dD/dlnnu, dD/dlnc)`` per ligand, shape ``(K, M)`` each, at
        free ligands ``L`` (K,), ``h``, ``nu``, ``c`` (one stage)."""
        out = self.d_arrays_all(np.asarray(L, dtype=float)[None, :], np.array([h]),
                                np.array([nu]), np.array([c]))
        return tuple(arr[0] for arr in out)

    def d_arrays_all(self, L: np.ndarray, h: np.ndarray, nu: np.ndarray, c: np.ndarray,
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(D, dD/dlnL, dD/dlnh, dD/dlnnu, dD/dlnc)`` of every stage, shape ``(N, K, M)``
        each, at free ligands ``L`` (N, K), ``h``, ``nu``, ``c`` (N,).  The model cores are
        scalar in the stage variables, so the loop over stages and ligands remains; the powers
        and products are vectorised over the stack."""
        N = L.shape[0]
        K, M = self.K, self.M
        lo = np.empty((N, K, M))
        sa = np.empty((N, K, M))
        sb = np.empty((N, K, M))
        sc = np.empty((N, K, M))
        sd = np.empty((N, K, M))
        Lf = L.tolist()
        hl, nul = h.tolist(), nu.tolist()
        cl = np.maximum(c, _C_TINY).tolist() if self.has_c else [0.0] * N
        cores, idxs, LT, LTe = self.cores, self.idx, self.LT.tolist(), self.LT_eff.tolist()
        for j in range(N):
            Lj, hj, nuj, cj = Lf[j], hl[j], nul[j], cl[j]
            for k in range(K):
                o0, o1, o2, o3, o4 = cores[k](Lj[k], hj, nuj, cj, LT[k], LTe[k])
                idx = idxs[k]
                if idx is not None:
                    o0, o1, o2, o3, o4 = o0[idx], o1[idx], o2[idx], o3[idx], o4[idx]
                lo[j, k] = o0
                sa[j, k] = o1
                sb[j, k] = o2
                sc[j, k] = o3
                sd[j, k] = o4
        D = np.power(10.0, np.minimum(lo, LOG10_CAP))
        Dl = D * LN10
        return D, Dl * sa, Dl * sb, Dl * sc, Dl * sd

    def unpack(self, u: np.ndarray) -> tuple[np.ndarray, ...]:
        """``(x (N,M), L (N,K), h (N,), nu (N,), c (N,), B (N,))`` from the unknown vector."""
        N, M, K = self.top.n_stages, self.M, self.K
        U = u.reshape(N, self.ns)
        return (U[:, :M], U[:, M:M + K], U[:, self.ih], U[:, self.inu], U[:, self.ic],
                U[:, self.iB])

    def pack(self, x: np.ndarray, L: np.ndarray, h: np.ndarray, nu: np.ndarray, c: np.ndarray,
             B: np.ndarray) -> np.ndarray:
        U = np.empty((self.top.n_stages, self.ns))
        U[:, :self.M] = x
        U[:, self.M:self.M + self.K] = L
        U[:, self.ih], U[:, self.inu], U[:, self.ic], U[:, self.iB] = h, nu, c, B
        return U.reshape(-1)

    def fresh_stream(self) -> OrgStream:
        """The fresh organic as an ``OrgStream`` over the cascade metals (zeros when None)."""
        y = {m: float(self.y_fresh[:, i].sum()) for i, m in enumerate(self.metals)}
        yk = {name: {m: float(self.y_fresh[k, i]) for i, m in enumerate(self.metals)}
              for k, name in enumerate(self.ligands)}
        lt = {name: float(self.LT[k]) for k, name in enumerate(self.ligands)}
        return OrgStream(flow_L_h=self.O, metals=y, metals_by_ligand=yk, ligand_total=lt,
                         ligand_free=dict(lt),
                         acid_in_org={name: float(self.a_fresh[k])
                                      for k, name in enumerate(self.ligands)},
                         alkali_reserve=0.0)

    def recycle_inlet(self, org_last: OrgStream) -> OrgStream:
        """Organic inlet of stage 0: ``(1 - f) org_out[N-1] + f fresh`` with the reserve
        re-created (7.2)."""
        f = self.f_bleed
        fresh = self.fresh_stream()
        y = {m: (1 - f) * float(org_last.metals.get(m, 0.0)) + f * fresh.metals[m]
             for m in self.metals}
        yk = {}
        for name in self.ligands:
            last = org_last.metals_by_ligand.get(name, org_last.metals if self.K == 1 else {})
            yk[name] = {m: (1 - f) * float(last.get(m, 0.0)) + f * fresh.metals_by_ligand[name][m]
                        for m in self.metals}
        a = {name: (1 - f) * float(org_last.acid_in_org.get(name, 0.0))
             + f * fresh.acid_in_org[name] for name in self.ligands}
        return OrgStream(flow_L_h=self.O, metals=y, metals_by_ligand=yk,
                         ligand_total=dict(fresh.ligand_total),
                         ligand_free=dict(fresh.ligand_total), acid_in_org=a,
                         alkali_reserve=self.B0)


# =============================================================================================
# 7.7  Kremser: linear metal pass, oracle, initial solve
# =============================================================================================

def kremser_fraction_unextracted(E: float, n: int) -> float:
    """Kremser fraction of a metal left in the raffinate of an ``n``-stage countercurrent
    extraction section with lean organic and no recycle: ``(E - 1) / (E**(n + 1) - 1)`` for
    ``E != 1`` and ``1 / (n + 1)`` for ``E == 1``, with ``E = D * (O / A)`` (DESIGN.md 7.7)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if E == 1.0:
        return 1.0 / (n + 1)
    return (E - 1.0) / (E ** (n + 1) - 1.0)


def _linear_metal_pass(prob: _Problem, D: np.ndarray, ext: np.ndarray, y_fresh: float,
                       ) -> np.ndarray:
    """Solve the metal balances of one metal for fixed per-stage ``D`` (N,): ``A_j x_j + O D_j
    x_j - sum_m Inc[j,m] A_m x_m - O D_(j-1) x_(j-1) = ext_j``, with the recycle ``y_(-1) = (1 -
    f) D_(N-1) x_(N-1) + f y_fresh``; returns ``x`` (N,).  The pass shared by ``kremser_init``
    and the origin-labelled pass (7.8)."""
    top = prob.top
    N = top.n_stages
    O, f = prob.O, prob.f_bleed
    A = top.a_flow
    Mx = np.diag(A + O * D).astype(float)
    Mx -= top.inc * A[None, :]
    for j in range(1, N):
        Mx[j, j - 1] -= O * D[j - 1]
    Mx[0, N - 1] -= (1.0 - f) * O * D[N - 1]
    rhs = ext.astype(float).copy()
    rhs[0] += f * O * y_fresh
    if N <= _DENSE_MAX:
        try:
            return np.linalg.solve(Mx, rhs)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(Mx, rhs, rcond=None)[0]
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import spsolve

    return np.asarray(spsolve(csc_matrix(Mx), rhs)).reshape(-1)


def section_tracer_d(spec: CascadeSpec, system: SystemModel,
                     ) -> dict[str, dict[str, float]]:
    """Every metal's composite ``D`` at the tracer conditions of each section present in the
    cascade (``L = L_T``, the section's aqueous inlet acid and anion, its complexant total at zero
    loading; 7.7).  Keys ``extraction``, ``scrub`` (when ``n_scr > 0``), ``strip`` (when
    ``n_str > 0``); values ``{metal: D}`` over the system metals."""
    prob = _Problem(spec, system)
    return {name: dict(zip(prob.all_metals, _tracer_d_all(prob, name)))
            for name in _sections_present(prob)}


def _sections_present(prob: _Problem) -> list[str]:
    out = ["extraction"]
    if prob.top.n_scr > 0:
        out.append("scrub")
    if prob.top.n_str > 0:
        out.append("strip")
    return out


def _section_inlet(prob: _Problem, section: str) -> AqStream:
    spec = prob.spec
    return {"extraction": spec.feed, "scrub": spec.scrub, "strip": spec.strip}[section]


def _tracer_d_all(prob: _Problem, section: str) -> np.ndarray:
    """Composite tracer D over ``prob.all_metals`` at the section's inlet conditions."""
    inlet = _section_inlet(prob, section)
    lt = {name: float(prob.LT[k]) for k, name in enumerate(prob.ligands)}
    state = tracer_state(prob.system, lt, max(inlet.h, H_MIN), max(inlet.anion, NU_MIN),
                         inlet.complexant_total if prob.cm is not None else 0.0,
                         metals=prob.all_metals)
    out = np.zeros(len(prob.all_metals))
    for name, md in prob.system.dmodels.items():
        Lf = state.ligand_free[name]
        core = getattr(md, "core", None)
        if core is not None:
            lo = core(Lf, state.h, state.anion, state.c_free, state.ligand_total[name],
                      state.ligand_free[name])[0]
            index = {m: i for i, m in enumerate(md.metals)}
            for i, m in enumerate(prob.all_metals):
                out[i] += 10.0 ** min(float(lo[index[m]]), LOG10_CAP)
        else:
            for i, m in enumerate(prob.all_metals):
                out[i] += min(md.evaluate(m, state).d, 10.0 ** LOG10_CAP)
    return out


def _tracer_d_active(prob: _Problem) -> np.ndarray:
    """``D_j,i`` (N, M) at the tracer conditions of the section of stage ``j``."""
    pos = {m: i for i, m in enumerate(prob.all_metals)}
    sel = [pos[m] for m in prob.metals]
    per_section = {name: _tracer_d_all(prob, name)[sel] for name in _sections_present(prob)}
    return np.array([per_section[prob.top.section[j]] for j in range(prob.top.n_stages)])


def _state_from_x(prob: _Problem, x: np.ndarray, D: np.ndarray, L_guess: np.ndarray | None,
                  h_guess: np.ndarray | None, nu_guess: np.ndarray | None,
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``L, h, nu, c, B`` from the balances at metal profile ``x`` (N, M) and composite ``D``
    (N, M) split over ligands in proportion to the ligands' tracer D, each floored at 1 % of its
    total (7.7).  Acid and reserve follow the aqueous and organic orders with the branch rule."""
    top = prob.top
    N, K, M = top.n_stages, prob.K, prob.M
    y = D * x                                          # (N, M) organic metal
    yk = np.empty((N, K, M))
    if K == 1:
        yk[:, 0, :] = y
    else:
        Lg = L_guess if L_guess is not None else np.tile(prob.LT_eff, (N, 1))
        hg = h_guess if h_guess is not None else np.full(N, max(prob.spec.feed.h, H_MIN))
        nug = nu_guess if nu_guess is not None else np.full(N, max(prob.spec.feed.anion, NU_MIN))
        Dk = prob.d_arrays_all(Lg, hg, nug, np.zeros(N))[0]
        tot = Dk.sum(axis=1, keepdims=True)
        w = np.where(tot > 0, Dk / np.where(tot > 0, tot, 1.0), 1.0 / K)
        yk = w * y[:, None, :]
    h_in0 = np.maximum(prob.ext_h / top.a_flow, 0.0)
    h = np.array(h_guess if h_guess is not None else np.maximum(h_in0, H_MIN), dtype=float)
    nu = np.array(nu_guess if nu_guess is not None else np.maximum(prob.ext_nu / top.a_flow,
                                                                   NU_MIN), dtype=float)
    bound = np.einsum("km,jkm->jk", prob.q, yk)
    hn = (h * nu)[:, None]
    L = (prob.LT_eff[None, :] - bound) / (1.0 + prob.KH[None, :] * hn)
    L = np.minimum(np.maximum(L, 0.01 * prob.LT_eff[None, :]), prob.LT_eff[None, :])
    a = prob.KH[None, :] * hn * L
    # organic upstream quantities (recycle for stage 0)
    f = prob.f_bleed
    yk_prev = np.empty_like(yk)
    a_prev = np.empty_like(a)
    yk_prev[0] = (1 - f) * yk[N - 1] + f * prob.y_fresh
    a_prev[0] = (1 - f) * a[N - 1] + f * prob.a_fresh
    yk_prev[1:] = yk[:-1]
    a_prev[1:] = a[:-1]
    dyk = yk - yk_prev
    released = np.einsum("km,jkm->j", prob.p, dyk)
    dz = np.einsum("km,jkm->j", prob.z, dyk)
    uptake = (a - a_prev).sum(axis=1)
    B = np.zeros(N)
    r = prob.r
    h_ref = max(prob.spec.feed.h, prob.spec.scrub.h, prob.spec.strip.h, H_MIN)
    nu_ref = max(prob.spec.feed.anion, prob.spec.scrub.anion, prob.spec.strip.anion, NU_MIN)
    for _ in range(3):                                   # alternate the two directions
        h_in, nu_in = prob.inlet_h_nu(h, nu)
        B_prev = prob.B0
        for j in range(N):
            P = h_in[j] + r[j] * (released[j] - uptake[j])
            if P - r[j] * B_prev >= H_MIN:
                h[j] = P - r[j] * B_prev
                B[j] = 0.0
            else:
                h[j] = H_MIN
                B[j] = max(B_prev - max(0.0, P - H_MIN) / r[j], 0.0)
            B_prev = B[j]
        h = np.maximum(h, max(0.01 * h_ref, H_MIN))
        nu = np.maximum(nu_in - r * dz - r * uptake, max(0.01 * nu_ref, NU_MIN))
    c = np.zeros(N)
    if prob.has_c:
        from .equilibrium import free_complexant

        for j in range(N):
            xt = dict(zip(prob.metals, x[j].tolist()))
            c[j] = max(free_complexant(prob.system, xt, h[j], prob.cT[j]), 0.01 * prob.cT[j])
    return L, h, nu, c, B


def kremser_init(spec: CascadeSpec, system: SystemModel) -> np.ndarray:
    """Kremser-exact initial unknown vector (7.7): every metal's D at the tracer conditions of
    its section, the metal balances solved directly (recycle and return couplings included),
    then ``L``, ``h``, ``nu``, ``c``, ``B`` from the balances at those ``x``, floored at 1 % of
    their totals.  Exact for ``ConstantD``.  Layout: ``_Problem.pack``."""
    prob = _Problem(spec, system)
    return _kremser_u0(prob)


def _kremser_u0(prob: _Problem) -> np.ndarray:
    D = _tracer_d_active(prob)
    x = np.empty((prob.top.n_stages, prob.M))
    for i in range(prob.M):
        x[:, i] = _linear_metal_pass(prob, D[:, i], prob.ext_metal[:, i],
                                     float(prob.y_fresh[:, i].sum()))
    x = np.maximum(x, 0.0)
    L, h, nu, c, B = _state_from_x(prob, x, D, None, None, None)
    return prob.pack(x, L, h, nu, c, B)


def _refine_init(prob: _Problem, u0: np.ndarray, *, iters: int = _INIT_REFINE_ITERS,
                 damping: float = 0.5, tol: float = 1e-3) -> np.ndarray:
    """Damped constant-D relinearisation of the Kremser init (addendum WB3): re-evaluate every
    stage's composite D at its current ``(L, h, nu, c)``, move ``log10 D`` half-way toward it,
    re-solve the linear metal pass and rebuild the state from the balances; stop when the
    relative change of ``x`` falls below ``tol`` or after ``iters`` rounds.  Exact for
    ``ConstantD`` (D never changes, so the first pass returns the Kremser solution)."""
    x, L, h, nu, c, B = (arr.copy() for arr in prob.unpack(u0))
    N, M = prob.top.n_stages, prob.M
    D0 = _tracer_d_active(prob)
    D_now = prob.d_arrays_all(L, h, nu, c)[0].sum(axis=1)
    if np.allclose(D_now, D0, rtol=1e-12, atol=0.0):
        return u0                                       # constant D: already exact
    logD = np.log10(np.maximum(D0, 1e-300))
    for _ in range(iters):
        new = np.log10(np.maximum(prob.d_arrays_all(L, h, nu, c)[0].sum(axis=1), 1e-300))
        logD = damping * new + (1.0 - damping) * logD
        D = np.power(10.0, np.minimum(logD, LOG10_CAP))
        x_new = np.empty_like(x)
        for i in range(M):
            x_new[:, i] = _linear_metal_pass(prob, D[:, i], prob.ext_metal[:, i],
                                             float(prob.y_fresh[:, i].sum()))
        x_new = np.maximum(x_new, 0.0)
        change = float(np.max(np.abs(x_new - x) / (np.abs(x_new) + 1e-3 * prob.conc_scale)))
        x = x_new
        L, h, nu, c, B = _state_from_x(prob, x, D, L, h, nu)
        if change < tol:
            break
    return prob.pack(x, L, h, nu, c, B)


# =============================================================================================
# 7.6  successive substitution
# =============================================================================================

def _aq_inlet(prob: _Problem, j: int, x: np.ndarray, h: np.ndarray, nu: np.ndarray,
              na: np.ndarray) -> AqStream:
    """Aqueous inlet of stage ``j`` from the current stage values and the injections."""
    top = prob.top
    Aj = top.a_flow[j]
    metals = prob.ext_metal[j].copy()
    hs, nus, nas = prob.ext_h[j], prob.ext_nu[j], prob.ext_na[j]
    for m in top.upstream[j]:
        Am = top.a_flow[m]
        metals += Am * x[m]
        hs += Am * h[m]
        nus += Am * nu[m]
        nas += Am * na[m]
    return AqStream(flow_L_h=float(Aj), metals=dict(zip(prob.metals, (metals / Aj).tolist())),
                    h=float(hs / Aj), anion=float(nus / Aj), complexant_total=float(prob.cT[j]),
                    sodium=float(nas / Aj))


def _org_from_arrays(prob: _Problem, yk: np.ndarray, L: np.ndarray, a: np.ndarray, B: float,
                     ) -> OrgStream:
    y = yk.sum(axis=0)
    return OrgStream(
        flow_L_h=prob.O, metals=dict(zip(prob.metals, y.tolist())),
        metals_by_ligand={name: dict(zip(prob.metals, yk[k].tolist()))
                          for k, name in enumerate(prob.ligands)},
        ligand_total={name: float(prob.LT[k]) for k, name in enumerate(prob.ligands)},
        ligand_free={name: float(L[k]) for k, name in enumerate(prob.ligands)},
        acid_in_org={name: float(a[k]) for k, name in enumerate(prob.ligands)},
        alkali_reserve=float(B))


def _streams_from_u(prob: _Problem, u: np.ndarray) -> tuple[list[AqStream], list[OrgStream]]:
    """Stage outlet streams from an unknown vector (sodium by the ledger of the branch rule)."""
    top = prob.top
    N = top.n_stages
    x, L, h, nu, c, B = prob.unpack(u)
    aq: list[AqStream | None] = [None] * N
    org: list[OrgStream] = []
    D_all = prob.d_arrays_all(L, h, nu, c)[0]
    yk_all = D_all * x[:, None, :]
    a_all = prob.KH[None, :] * (h * nu)[:, None] * L
    for j in range(N):
        org.append(_org_from_arrays(prob, yk_all[j], L[j], a_all[j], B[j]))
    # sodium: released base per stage from the reserve change (organic direction)
    f = prob.f_bleed
    B_prev = prob.B0
    consumed = np.zeros(N)
    for j in range(N):
        consumed[j] = max(B_prev - B[j], 0.0) * prob.O       # mol/h of base that met protons
        B_prev = B[j]
    na = prob._topological_pass(prob.ext_na + consumed)
    for j in range(N):
        aq[j] = AqStream(flow_L_h=float(top.a_flow[j]), metals=dict(zip(prob.metals,
                                                                        x[j].tolist())),
                         h=float(h[j]), anion=float(nu[j]), complexant_total=float(prob.cT[j]),
                         sodium=float(na[j]))
    del f
    return [s for s in aq if s is not None], org


def solve_cascade_ss(spec: CascadeSpec, system: SystemModel, init: np.ndarray | None = None, *,
                     max_sweeps: int = 5000, tol: float = _SS_TOL,
                     balance_tol: float = _SS_BALANCE_TOL) -> CascadeResult:
    """Successive substitution (7.6): Gauss-Seidel sweeps in the organic direction over
    ``solve_stage`` with relaxation on ``x`` (``omega`` halved after three consecutive error
    increases, floor 0.05) and bounded Wegstein (factor in [-5, 0]) on the recycle tear
    ``(y^(k)_(N-1), a_(N-1))`` after sweep 5.  Converged when the relative change of ``x`` and
    of the tear is below ``tol`` and ``check_balances`` closes to ``balance_tol``; status
    ``converged_ss`` or ``failed``.  ``ValueError`` only on a malformed specification."""
    prob = _Problem(spec, system)
    return _solve_ss(prob, init, max_sweeps, tol, balance_tol)


class _SSIteration:
    """State of the successive-substitution sweeps (7.6): stage values, the recycle tear
    ``(y^(k)_(N-1), a_(N-1))``, the relaxation factor and the Wegstein bookkeeping.  ``sweep()``
    performs one Gauss-Seidel pass in the organic direction and returns the relative change;
    ``unknowns()`` packs the current stage values as a cascade unknown vector (the free
    complexant recovered per stage); ``result(status)`` assembles a ``CascadeResult``."""

    def __init__(self, prob: _Problem, init: np.ndarray | None):
        self.prob = prob
        top = prob.top
        N, K, M = top.n_stages, prob.K, prob.M
        self.N, self.K, self.M = N, K, M
        u = _kremser_u0(prob) if init is None else np.asarray(init, dtype=float)
        self.x, self.L, self.h, self.nu, self.c, self.B = (arr.copy() for arr in prob.unpack(u))
        self.na = prob._topological_pass(prob.ext_na)
        self.aq_out: list[AqStream | None] = [None] * N
        self.org_out: list[OrgStream | None] = [None] * N
        self.diags: list[StageDiagnostics | None] = [None] * N
        D_last = prob.d_arrays(self.L[N - 1], self.h[N - 1], self.nu[N - 1], self.c[N - 1])[0]
        self.tear = np.concatenate([(D_last * self.x[N - 1][None, :]).reshape(-1),
                                    prob.KH * self.h[N - 1] * self.nu[N - 1] * self.L[N - 1]])
        self.tear_in_prev: np.ndarray | None = None
        self.tear_out_prev: np.ndarray | None = None
        self.omega = 1.0
        self.err_prev = math.inf
        self.increases = 0
        self.sweeps = 0
        self.stage_iters = 0

    def sweep(self) -> float:
        prob, N, K, M = self.prob, self.N, self.K, self.M
        x, L, h, nu, na, B = self.x, self.L, self.h, self.nu, self.na, self.B
        self.sweeps += 1
        x_old = x.copy()
        tear = self.tear
        org_last = _org_from_arrays(prob, tear[:K * M].reshape(K, M), np.array(prob.LT_eff),
                                    tear[K * M:], 0.0)
        org_in = prob.recycle_inlet(org_last)
        for j in range(N):
            aq_in = _aq_inlet(prob, j, x, h, nu, na)
            a_j, o_j, d_j = solve_stage(aq_in, org_in, prob.system, warm=self.diags[j])
            self.aq_out[j], self.org_out[j], self.diags[j] = a_j, o_j, d_j
            self.stage_iters += d_j.iterations
            x[j] = [a_j.metals[m] for m in prob.metals]
            h[j], nu[j], na[j] = a_j.h, a_j.anion, a_j.sodium
            L[j] = [o_j.ligand_free[name] for name in prob.ligands]
            B[j] = o_j.alkali_reserve
            org_in = o_j
        if self.omega != 1.0:
            x[:] = self.omega * x + (1.0 - self.omega) * x_old
        err = float(np.max(np.abs(x - x_old) / (np.abs(x) + 1e-30)))
        o_last = self.org_out[N - 1]
        assert o_last is not None
        yk_last = np.array([[o_last.metals_by_ligand[name][m] if K > 1 else o_last.metals[m]
                             for m in prob.metals] for name in prob.ligands])
        tear_out = np.concatenate([yk_last.reshape(-1),
                                   [o_last.acid_in_org[name] for name in prob.ligands]])
        err = max(err, float(np.max(np.abs(tear_out - tear) / (np.abs(tear_out) + 1e-30))))
        if self.sweeps > _SS_START_WEGSTEIN and self.tear_in_prev is not None:
            new = np.empty_like(tear)
            for t in range(tear.size):
                d_in = tear[t] - self.tear_in_prev[t]
                if abs(d_in) > 1e-14 * (abs(tear[t]) + 1e-30):
                    s = (tear_out[t] - self.tear_out_prev[t]) / d_in
                    qf = s / (s - 1.0) if s != 1.0 else 0.0
                    qf = min(max(qf, -5.0), 0.0)
                else:
                    qf = 0.0
                new[t] = qf * tear[t] + (1.0 - qf) * tear_out[t]
            self.tear_in_prev, self.tear_out_prev = tear.copy(), tear_out.copy()
            self.tear = np.maximum(new, 0.0)
        else:
            self.tear_in_prev, self.tear_out_prev = tear.copy(), tear_out.copy()
            self.tear = tear_out.copy()
        if err > self.err_prev:
            self.increases += 1
            if self.increases >= 3:
                self.omega = max(0.05, self.omega / 2.0)
                self.increases = 0
        else:
            self.increases = 0
        self.err_prev = err
        return err

    def unknowns(self) -> np.ndarray:
        prob = self.prob
        c = np.zeros(self.N)
        if prob.has_c:
            from .equilibrium import free_complexant

            for j in range(self.N):
                if prob.cT[j] > 0.0:
                    xt = dict(zip(prob.metals, self.x[j].tolist()))
                    c[j] = max(free_complexant(prob.system, xt, self.h[j], prob.cT[j]),
                               1e-6 * prob.cT[j])
        return prob.pack(self.x, self.L, self.h, self.nu, c, self.B)

    def result(self, status: str, err: float) -> CascadeResult:
        return _assemble(self.prob, self.aq_out, self.org_out, self.diags, status, err,
                         self.sweeps, self.stage_iters)


def _solve_ss(prob: _Problem, init: np.ndarray | None, max_sweeps: int, tol: float,
              balance_tol: float, state: "_SSIteration | None" = None) -> CascadeResult:
    it = state if state is not None else _SSIteration(prob, init)
    err = math.inf
    for _ in range(max_sweeps):
        err = it.sweep()
        if err < tol:
            partial = it.result("converged_ss", err)
            if partial.balance_rel_max < balance_tol and all(
                    d.status == "converged" for d in partial.diagnostics):
                return partial
    if it.sweeps == 0:
        it.sweep()
    return it.result("failed", it.err_prev)


# =============================================================================================
# 7.8  origin-labelled pass
# =============================================================================================

def origin_pass(prob: _Problem, D_T: np.ndarray, x_T: np.ndarray, y_T: np.ndarray,
                ) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Exact linear pass at the converged composite ``D_T`` (N,) of the target (7.8).

    Returns the reported quantities (``recovery_from_feed``, ``recovery_total``,
    ``scrub_target_return`` (NaN when the scrub carries no target), ``net_product_mol_h``, the
    product / raffinate label sums in mol/h, the external target inputs and the largest relative
    label-sum defect) and the per-stage aqueous label profiles ``{label: x^l (N,)}``.
    """
    top = prob.top
    N = top.n_stages
    t = prob.metals.index(prob.spec.target)
    ext_by_label: dict[str, np.ndarray] = {lab: np.zeros(N) for lab in ORIGIN_LABELS}
    ext_by_label["feed"][top.feed_stage] = top.feed_flow * prob.spec.feed.metals.get(
        prob.spec.target, 0.0)
    if top.n_scr > 0:
        ext_by_label["scrub"][top.scrub_top] = top.scrub_flow * prob.spec.scrub.metals.get(
            prob.spec.target, 0.0)
    if top.n_str > 0:
        ext_by_label["strip"][top.strip_top] = top.strip_flow * prob.spec.strip.metals.get(
            prob.spec.target, 0.0)
    y_fresh_t = float(prob.y_fresh[:, t].sum())
    profiles: dict[str, np.ndarray] = {}
    for lab in ORIGIN_LABELS:
        profiles[lab] = _linear_metal_pass(prob, D_T, ext_by_label[lab],
                                           y_fresh_t if lab == "fresh" else 0.0)
    total = sum(profiles.values())
    defect = float(np.max(np.abs(total - x_T) / (np.abs(x_T) + 1e-30)))
    A_in = top.feed_flow * float(prob.spec.feed.metals.get(prob.spec.target, 0.0))
    S_in = (top.scrub_flow * float(prob.spec.scrub.metals.get(prob.spec.target, 0.0))
            if top.n_scr > 0 else 0.0)
    W_in = (top.strip_flow * float(prob.spec.strip.metals.get(prob.spec.target, 0.0))
            if top.n_str > 0 else 0.0)
    F_in = prob.f_bleed * prob.O * y_fresh_t

    def product_of(profile: np.ndarray) -> float:
        if top.n_str > 0:
            return float(top.strip_flow * profile[top.product_stage])
        return float(prob.O * D_T[N - 1] * profile[N - 1])

    prod = {lab: product_of(profiles[lab]) for lab in ORIGIN_LABELS}
    raff = {lab: float(top.a_flow[0] * profiles[lab][0]) for lab in ORIGIN_LABELS}
    product_total = float(top.strip_flow * x_T[top.product_stage]) if top.n_str > 0 else float(
        prob.O * y_T[N - 1])
    out: dict[str, float] = {
        "recovery_from_feed": prod["feed"] / A_in if A_in > 0 else math.nan,
        "recovery_total": product_total / A_in if A_in > 0 else math.nan,
        "scrub_target_return": prod["scrub"] / S_in if S_in > 0 else math.nan,
        "net_product_mol_h": product_total - S_in,
        "product_total_mol_h": product_total,
        "feed_target_mol_h": A_in, "scrub_target_mol_h": S_in, "strip_target_mol_h": W_in,
        "fresh_target_mol_h": F_in, "label_sum_rel_defect": defect,
    }
    for lab in ORIGIN_LABELS:
        out[f"product_{lab}_mol_h"] = prod[lab]
        out[f"raffinate_{lab}_mol_h"] = raff[lab]
    return out, profiles


def _attach_labels(prob: _Problem, aq: list[AqStream], org: list[OrgStream],
                   profiles: dict[str, np.ndarray], D_T: np.ndarray,
                   ) -> tuple[list[AqStream], list[OrgStream]]:
    T = prob.spec.target
    aq2, org2 = [], []
    for j in range(prob.top.n_stages):
        xl = {lab: float(profiles[lab][j]) for lab in ORIGIN_LABELS}
        yl = {lab: float(D_T[j] * profiles[lab][j]) for lab in ORIGIN_LABELS}
        aq2.append(replace(aq[j], labels={T: xl}))
        org2.append(replace(org[j], labels={T: yl}))
    return aq2, org2


# =============================================================================================
# 7.9  balances, regime status, result assembly
# =============================================================================================

def regime_status_of(flags: frozenset[Flag] | set[Flag]) -> str:
    """``INADMISSIBLE`` > ``OUT_OF_DOMAIN`` > ``IN_DOMAIN_WITH_CAVEATS`` > ``IN_DOMAIN``
    (DESIGN.md 1.4 / 8.3)."""
    fl = set(flags)
    if fl & INADMISSIBLE_FLAGS:
        return "INADMISSIBLE"
    if fl & OOD_FLAGS:
        return "OUT_OF_DOMAIN"
    if fl:
        return "IN_DOMAIN_WITH_CAVEATS"
    return "IN_DOMAIN"


def _ledgers_of(prob: _Problem, aq: list[AqStream], org: list[OrgStream]) -> dict[str, float]:
    """Relative in-minus-out ledgers from the stream table alone (7.9): whole-cascade for every
    metal, protons, anion, complexant, sodium; per-stage identity for every ligand; plus the
    largest per-stage relative defect of the metal, proton, anion and sodium ledgers (keys
    ``*_stage_max``)."""
    top = prob.top
    N, K = top.n_stages, prob.K
    O = prob.O
    metals = tuple(aq[0].metals)
    q, p, z = prob.q, prob.p, prob.z
    lig = prob.ligands
    fresh = prob.fresh_stream()
    inlet0 = prob.recycle_inlet(org[N - 1])
    mi = {m: i for i, m in enumerate(prob.metals)}

    def org_parts(o: OrgStream) -> tuple[np.ndarray, float, float, float]:
        """(metal mol/h, proton part, anion part, sodium part) of an organic stream."""
        y = np.array([o.metals.get(m, 0.0) for m in metals]) * o.flow_L_h
        prot = sum(o.acid_in_org.get(k, 0.0) for k in lig) - o.alkali_reserve
        an = sum(o.acid_in_org.get(k, 0.0) for k in lig)
        for k, name in enumerate(lig):
            sub = o.metals_by_ligand.get(name, o.metals if K == 1 else {})
            for m, v in sub.items():
                if m in mi:
                    prot -= p[k, mi[m]] * v
                    an += z[k, mi[m]] * v
        return y, o.flow_L_h * prot, o.flow_L_h * an, o.flow_L_h * o.alkali_reserve

    def aq_parts(a: AqStream) -> tuple[np.ndarray, float, float, float, float]:
        y = np.array([a.metals.get(m, 0.0) for m in metals]) * a.flow_L_h
        return (y, a.flow_L_h * a.h, a.flow_L_h * a.anion, a.flow_L_h * a.complexant_total,
                a.flow_L_h * a.sodium)

    ext_streams = [(prob.spec.feed, top.feed_stage)]
    if top.n_scr > 0:
        ext_streams.append((prob.spec.scrub, top.scrub_top))
    if top.n_str > 0:
        ext_streams.append((prob.spec.strip, top.strip_top))
    n_m = len(metals)
    stage_def = {"metal": 0.0, "proton": 0.0, "anion": 0.0, "sodium": 0.0, "complexant": 0.0}
    floors = {"metal": np.array([prob.metal_scale[mi[m]] if m in mi else 1e-300
                                 for m in metals]),
              "proton": prob.scale_h, "anion": prob.scale_nu,
              "sodium": max(prob.scale_B, top.feed_flow * prob.spec.feed.sodium,
                            top.scrub_flow * prob.spec.scrub.sodium,
                            top.strip_flow * prob.spec.strip.sodium, 1e-300),
              "complexant": prob.scale_c}
    tot_in = {"metal": np.zeros(n_m), "proton": 0.0, "anion": 0.0, "sodium": 0.0,
              "complexant": 0.0}
    tot_out = {"metal": np.zeros(n_m), "proton": 0.0, "anion": 0.0, "sodium": 0.0,
               "complexant": 0.0}
    tot_abs = {"metal": np.zeros(n_m), "proton": 0.0, "anion": 0.0, "sodium": 0.0,
               "complexant": 0.0}
    for j in range(N):
        s_in = {"metal": np.zeros(n_m), "proton": 0.0, "anion": 0.0, "sodium": 0.0,
                "complexant": 0.0}
        s_abs = {"metal": np.zeros(n_m), "proton": 0.0, "anion": 0.0, "sodium": 0.0,
                 "complexant": 0.0}

        def add(d: dict, y, prot, an, na, cx=0.0, sign: float = 1.0) -> None:
            d["metal"] = d["metal"] + sign * y
            d["proton"] += sign * prot
            d["anion"] += sign * an
            d["sodium"] += sign * na
            d["complexant"] += sign * cx

        for st, at in ext_streams:
            if at == j:
                y, hh, an, cx, na = aq_parts(replace(st, flow_L_h=(
                    top.feed_flow if st is prob.spec.feed else top.scrub_flow
                    if st is prob.spec.scrub else top.strip_flow)))
                add(s_in, y, hh, an, na, cx)
                add(s_abs, np.abs(y), abs(hh), abs(an), abs(na), abs(cx))
        for m in top.upstream[j]:
            y, hh, an, cx, na = aq_parts(aq[m])
            add(s_in, y, hh, an, na, cx)
            add(s_abs, np.abs(y), abs(hh), abs(an), abs(na), abs(cx))
        o_in = inlet0 if j == 0 else org[j - 1]
        y, prot, an, na = org_parts(o_in)
        add(s_in, y, prot, an, na)
        add(s_abs, np.abs(y), abs(prot), abs(an), abs(na))
        s_out = {"metal": np.zeros(n_m), "proton": 0.0, "anion": 0.0, "sodium": 0.0,
                 "complexant": 0.0}
        y, hh, an, cx, na = aq_parts(aq[j])
        add(s_out, y, hh, an, na, cx)
        add(s_abs, np.abs(y), abs(hh), abs(an), abs(na), abs(cx))
        y, prot, an, na = org_parts(org[j])
        add(s_out, y, prot, an, na)
        add(s_abs, np.abs(y), abs(prot), abs(an), abs(na))
        # per-stage defects relative to the larger of the stage's own throughput and the
        # cascade-level load of the species (the floor the residual rows use): a strip stage
        # holding 1e-40 mol/L of a metal must not report roundoff as a 79 % defect (WB3)
        for key in stage_def:
            diff = s_in[key] - s_out[key]
            if key == "metal":
                scale = np.maximum(s_abs[key], floors["metal"])
                rel_j = float(np.max(np.abs(diff) / scale)) if n_m else 0.0
            else:
                rel_j = abs(diff) / max(s_abs[key], floors[key])
            stage_def[key] = max(stage_def[key], rel_j)
        # whole-cascade: external inlets, the organic inlet of stage 0, the outlets
        for st, at in ext_streams:
            if at == j:
                y, hh, an, cx, na = aq_parts(replace(st, flow_L_h=(
                    top.feed_flow if st is prob.spec.feed else top.scrub_flow
                    if st is prob.spec.scrub else top.strip_flow)))
                add(tot_in, y, hh, an, na, cx)
                add(tot_abs, np.abs(y), abs(hh), abs(an), abs(na), abs(cx))
    # organic: the mixed inlet of stage 0 enters, the bleed (or the whole loaded organic when
    # there is no strip section) and the recycled fraction leave
    y, prot, an, na = org_parts(inlet0)
    add(tot_in, y, prot, an, na)
    add(tot_abs, np.abs(y), abs(prot), abs(an), abs(na))
    y, prot, an, na = org_parts(org[N - 1])
    add(tot_out, y, prot, an, na)
    add(tot_abs, np.abs(y), abs(prot), abs(an), abs(na))
    del fresh
    y, hh, an, cx, na = aq_parts(aq[0])
    add(tot_out, y, hh, an, na, cx)
    add(tot_abs, np.abs(y), abs(hh), abs(an), abs(na), abs(cx))
    if top.n_str > 0:
        y, hh, an, cx, na = aq_parts(aq[top.product_stage])
        add(tot_out, y, hh, an, na, cx)
        add(tot_abs, np.abs(y), abs(hh), abs(an), abs(na), abs(cx))
    out: dict[str, float] = {}
    for i, m in enumerate(metals):
        out[f"metal.{m}"] = abs(tot_in["metal"][i] - tot_out["metal"][i]) / max(
            tot_abs["metal"][i], 1e-300)
    for key in ("proton", "anion", "complexant", "sodium"):
        out[key] = abs(tot_in[key] - tot_out[key]) / max(tot_abs[key], 1e-300)
    for k, name in enumerate(lig):
        worst = 0.0
        for o in org:
            sub = o.metals_by_ligand.get(name, o.metals if K == 1 else {})
            bound = sum(q[k, mi[m]] * v for m, v in sub.items() if m in mi)
            lt = o.ligand_total[name]
            worst = max(worst, abs(o.ligand_free[name] + o.acid_in_org.get(name, 0.0) + bound
                                   - prob.LT_eff[k]) / lt)
        out[f"ligand.{name}"] = worst
    for key, val in stage_def.items():
        out[f"{key}_stage_max"] = val
    return out


def check_balances(spec: CascadeSpec, result: CascadeResult, system: SystemModel,
                   ) -> dict[str, float]:
    """Relative in-minus-out ledgers recomputed from ``result``'s stream table alone (7.9):
    ``metal.<m>`` per metal, ``ligand.<k>`` (largest per-stage deviation of ``L_f + a_k + sum_i
    q_i y^(k)`` from ``L_T``), ``proton`` (``Q = A h + O (sum_k a_k - B - sum_i p_i y_i)``),
    ``anion``, ``complexant``, ``sodium``, and the per-stage maxima ``*_stage_max``.  The
    stoichiometry comes from ``system`` (the streams do not carry it; addendum WB3)."""
    prob = _Problem(spec, system)
    return _ledgers_of(prob, list(result.stages_aq), list(result.stages_org))


def _diagnostics_of(prob: _Problem, aq: AqStream, org: OrgStream, D: np.ndarray,
                    iterations: int, residual: float, branch: int, status: str,
                    singular: bool = False) -> StageDiagnostics:
    """Stage diagnostics of a Newton-path stage: chemistry / domain flags of the solved state
    (``model.state_flags``), loading fractions, the composite and per-ligand D."""
    state = stage_state(aq, org, prob.system)
    flags: set[Flag] = set(prob.system.flags)
    dist: dict[str, float] = {}
    for k, md in enumerate(prob.models):
        if isinstance(md, _ModelBase):
            f, d = md.state_flags(state, prob.system.activity)
        else:
            f, d = set(), {}
            for m in prob.metals:
                ev = md.evaluate(m, state, prob.system.activity)
                f |= ev.flags
                for key, val in ev.ood_distance.items():
                    d[key] = max(d.get(key, 0.0), val)
        flags |= f
        for key, val in d.items():
            if val > dist.get(key, 0.0):
                dist[key] = val
    if branch == 2:
        flags.add(Flag.ALKALI_EXCESS)
    if status != "converged":
        flags.add(Flag.NOT_CONVERGED)
    if singular:
        flags.add(Flag.JACOBIAN_SINGULAR)
    loading = {name: ligand_loading_fraction(state, name, prob.models[k].q)
               for k, name in enumerate(prob.ligands)}
    return StageDiagnostics(
        d=dict(zip(prob.metals, D.sum(axis=0).tolist())),
        d_by_ligand={name: dict(zip(prob.metals, D[k].tolist()))
                     for k, name in enumerate(prob.ligands)},
        loading_fraction=loading, iterations=int(iterations), residual_max=float(residual),
        branch=int(branch), flags=frozenset(flags), ood_distance=dist, status=status)


def _assemble(prob: _Problem, aq: list, org: list, diags: list, status: str,
              residual_max: float, iterations: int, stage_iterations: int = 0,
              extra_flags: frozenset[Flag] = frozenset()) -> CascadeResult:
    top = prob.top
    N = top.n_stages
    aq = [s for s in aq]
    org = [s for s in org]
    diags = [d for d in diags]
    balances = _ledgers_of(prob, aq, org)
    balance_max = max(balances.values()) if balances else math.nan
    flags: set[Flag] = set(extra_flags)
    dist: dict[str, float] = {}
    for d in diags:
        flags |= d.flags
        for key, val in d.ood_distance.items():
            if val > dist.get(key, 0.0):
                dist[key] = val
    if status == "failed":
        flags.add(Flag.NOT_CONVERGED)
    origin = None
    if prob.spec.target and prob.spec.target in prob.metals:
        t = prob.spec.target
        D_T = np.array([d.d[t] for d in diags])
        x_T = np.array([a.metals[t] for a in aq])
        y_T = np.array([o.metals[t] for o in org])
        finite = np.all(np.isfinite(D_T))
        if finite:
            origin, profiles = origin_pass(prob, D_T, x_T, y_T)
            aq, org = _attach_labels(prob, aq, org, profiles, D_T)
    product = aq[top.product_stage] if top.n_str > 0 else org[N - 1]
    scrub_raff = aq[top.n_ext] if top.n_scr > 0 else None
    loaded = org[top.n_ext + top.n_scr - 1]
    return CascadeResult(
        status=status, stages_aq=tuple(aq), stages_org=tuple(org), diagnostics=tuple(diags),
        raffinate=aq[0], product=product, scrub_raffinate=scrub_raff, loaded_organic=loaded,
        stripped_organic=org[N - 1], origin=origin, residual_max=float(residual_max),
        iterations=int(iterations), balance_rel_max=float(balance_max), balances=balances,
        flags=frozenset(flags), ood_distance=dist, regime_status=regime_status_of(flags),
        assumptions=tuple(prob.system.assumptions))


# =============================================================================================
# 7.5  primary solver: damped Newton with the analytic block Jacobian
# =============================================================================================

@dataclass(slots=True)
class _CascadeEval:
    """Everything computed at one unknown vector: per-stage D arrays ``D, GL, Gh, Gnu, Gc``
    (N, K, M; the ``G`` arrays are ``dD/dln(variable)``), organic metals ``yk`` (N, K, M),
    HNO3.L ``a`` (N, K), the organic-upstream values ``yk_prev`` / ``a_prev`` / ``B_prev``,
    ``released``, ``uptake``, ``P`` (N,), the active ``branch`` (N,), the complexant terms and
    the scaled residual ``F`` (N, ns)."""

    x: np.ndarray
    L: np.ndarray
    h: np.ndarray
    nu: np.ndarray
    c: np.ndarray
    B: np.ndarray
    D: np.ndarray
    GL: np.ndarray
    Gh: np.ndarray
    Gnu: np.ndarray
    Gc: np.ndarray
    yk: np.ndarray
    a: np.ndarray
    yk_prev: np.ndarray
    a_prev: np.ndarray
    B_prev: np.ndarray
    released: np.ndarray
    uptake: np.ndarray
    P: np.ndarray
    branch: np.ndarray
    alpha_h: np.ndarray
    dalpha_h: np.ndarray
    phi: np.ndarray
    dphi: np.ndarray
    F: np.ndarray
    fmax: float = 0.0


class _Newton:
    """Residuals, analytic Jacobian and the damped Newton iteration of section 7.5 on the
    unknown layout of ``_Problem`` (all residuals in mol/h, rows scaled).

    The iteration runs in the internal variables ``w = (x / s_x, ln L, ln h, ln nu, ln c, B /
    s_B)`` (addendum WB3): the metal and reserve unknowns are scaled linear variables (``s_x``,
    ``s_B`` = ``max(u0, floor)``), the free ligand, acid, anion and free complexant are
    logarithms — the mass-action ``log D`` is linear in them, positivity is automatic, and the
    Jacobian columns are the ``dD / dln(.)`` partials the models already provide.  A stage whose
    circuit carries no complexant keeps ``c = 0`` as a linear unknown pinned by its own row.
    ``to_w`` / ``to_u`` convert; ``evaluate`` and ``jacobian_scaled`` take ``w``.
    """

    def __init__(self, prob: _Problem, u0: np.ndarray):
        self.prob = prob
        top = prob.top
        N, M, K, ns = top.n_stages, prob.M, prob.K, prob.ns
        self.N, self.M, self.K, self.ns = N, M, K, ns
        rs = np.empty((N, ns))
        rs[:, :M] = prob.metal_scale
        rs[:, M:M + K] = prob.scale_lig
        rs[:, prob.ih] = prob.scale_h
        rs[:, prob.inu] = prob.scale_nu
        rs[:, prob.ic] = prob.scale_c if prob.has_c else max(top.a_flow.max(), 1.0)
        rs[:, prob.iB] = prob.scale_B
        self.row_scale = rs
        x0, L0, h0, nu0, c0, B0 = prob.unpack(u0)
        self.s_x = np.maximum(x0, 1e-6 * prob.conc_scale)
        self.s_B = np.maximum(B0, max(1e-2 * prob.B0, 1e-6))
        self.c_active = (prob.cT > 0.0) if prob.has_c else np.zeros(N, dtype=bool)
        self.is_log = np.zeros((N, ns), dtype=bool)
        self.is_log[:, M:M + K] = True
        self.is_log[:, prob.ih] = True
        self.is_log[:, prob.inu] = True
        self.is_log[:, prob.ic] = self.c_active
        lb = np.full((N, ns), -np.inf)
        lb[:, :M] = 0.0
        lb[:, prob.iB] = 0.0
        lb[:, prob.ih] = math.log(H_MIN)
        lb[~self.c_active, prob.ic] = 0.0
        self.lb = lb.reshape(-1)
        self.singular = False
        self.evaluations = 0
        self.upstream = top.upstream
        self.A = top.a_flow

    # -- variable changes --------------------------------------------------------------------
    def to_w(self, u: np.ndarray) -> np.ndarray:
        prob = self.prob
        x, L, h, nu, c, B = prob.unpack(u)
        W = np.empty((self.N, self.ns))
        W[:, :self.M] = x / self.s_x
        W[:, self.M:self.M + self.K] = np.log(np.maximum(L, 1e-300))
        W[:, prob.ih] = np.log(np.maximum(h, H_MIN))
        W[:, prob.inu] = np.log(np.maximum(nu, 1e-300))
        cw = np.where(self.c_active, np.log(np.maximum(c, 1e-300)), c)
        W[:, prob.ic] = cw
        W[:, prob.iB] = B / self.s_B
        return W.reshape(-1)

    def to_u(self, w: np.ndarray) -> np.ndarray:
        prob = self.prob
        W = w.reshape(self.N, self.ns)
        x = W[:, :self.M] * self.s_x
        L = np.exp(W[:, self.M:self.M + self.K])
        h = np.exp(W[:, prob.ih])
        nu = np.exp(W[:, prob.inu])
        c = np.where(self.c_active, np.exp(W[:, prob.ic]), W[:, prob.ic])
        B = W[:, prob.iB] * self.s_B
        return prob.pack(x, L, h, nu, c, B)

    # -- residuals ---------------------------------------------------------------------------
    def evaluate(self, w: np.ndarray, branch: np.ndarray | None = None) -> _CascadeEval:
        """Residuals at the internal variables ``w`` (the active branch is determined from the
        iterate unless ``branch`` is given)."""
        self.evaluations += 1
        prob = self.prob
        top = prob.top
        N, M, K = self.N, self.M, self.K
        O, r, A = prob.O, prob.r, self.A
        x, L, h, nu, c, B = prob.unpack(self.to_u(w))
        has_c = prob.has_c
        D, GL, Gh, Gnu, Gc = prob.d_arrays_all(L, h, nu, c)
        yk = D * x[:, None, :]
        y = yk.sum(axis=1)
        a = prob.KH[None, :] * (h * nu)[:, None] * L
        f = prob.f_bleed
        yk_prev = np.empty_like(yk)
        a_prev = np.empty_like(a)
        yk_prev[0] = (1.0 - f) * yk[N - 1] + f * prob.y_fresh
        a_prev[0] = (1.0 - f) * a[N - 1] + f * prob.a_fresh
        yk_prev[1:] = yk[:-1]
        a_prev[1:] = a[:-1]
        y_prev = yk_prev.sum(axis=1)
        dyk = yk - yk_prev
        released = np.einsum("km,jkm->j", prob.p, dyk)
        dz = np.einsum("km,jkm->j", prob.z, dyk)
        uptake = (a - a_prev).sum(axis=1)
        h_in, nu_in = prob.inlet_h_nu(h, nu)
        P = h_in + r * (released - uptake)
        B_prev = np.empty(N)
        B_prev[0] = prob.B0
        B_prev[1:] = B[:-1]
        if branch is None:
            branch = self._active_branch(x, L, nu, c, h_in, yk_prev, a_prev, B_prev)
        b1 = branch == 1
        F = np.empty((N, self.ns))
        aq_up = prob.inc_w @ x
        F[:, :M] = A[:, None] * x + O * y - aq_up - O * y_prev - prob.ext_metal
        F[:, M:M + K] = O * (L * (1.0 + prob.KH[None, :] * (h * nu)[:, None])
                             + np.einsum("km,jkm->jk", prob.q, yk) - prob.LT_eff[None, :])
        F[:, prob.ih] = np.where(b1, A * (h - (P - r * B_prev)), A * (h - H_MIN))
        consumed = np.maximum(P - H_MIN, 0.0)
        F[:, prob.iB] = np.where(b1, O * B, O * (B - B_prev) + A * consumed)
        F[:, prob.inu] = A * (nu - nu_in) + O * dz + O * uptake
        if has_c:
            cm = prob.cm
            alpha_h = np.empty(N)
            dalpha_h = np.empty(N)
            phi = np.empty((N, M))
            dphi = np.empty((N, M))
            for j in range(N):
                alpha_h[j] = cm.alpha_h(h[j])
                dalpha_h[j] = cm.dalpha_h_dlnh(h[j])
                _, phi[j], dphi[j] = cm.terms(max(c[j], _C_TINY), prob.metals)
            F[:, prob.ic] = A * (c * alpha_h + np.einsum("jm,jm->j", x, phi) - prob.cT)
        else:
            alpha_h = np.ones(N)
            dalpha_h = np.zeros(N)
            phi = dphi = np.zeros((N, M))
            F[:, prob.ic] = A * c
        F /= self.row_scale
        ev = _CascadeEval(x, L, h, nu, c, B, D, GL, Gh, Gnu, Gc, yk, a, yk_prev, a_prev, B_prev,
                          released, uptake, P, branch, alpha_h, dalpha_h, phi, dphi, F)
        ev.fmax = float(np.abs(F).max())
        return ev

    def _active_branch(self, x: np.ndarray, L: np.ndarray, nu: np.ndarray, c: np.ndarray,
                       h_in: np.ndarray, yk_prev: np.ndarray, a_prev: np.ndarray,
                       B_prev: np.ndarray) -> np.ndarray:
        """Branch per stage by the stage solver's bracket rule (addendum WB2 A5.1 / WB3):
        branch 1 iff the branch-1 residual is negative at ``h = H_MIN``, i.e. ``P_j(H_MIN) -
        r_j B_(j-1) > H_MIN`` with every other unknown at its current value; this equals the
        design's ``P_j`` test at a converged stage and does not flip stages on the way there."""
        prob = self.prob
        N = self.N
        r = prob.r
        if prob.B0 == 0.0 and float(h_in.min()) >= H_MIN * (1.0 - 1e-9):
            return np.ones(N, dtype=int)
        D = prob.d_arrays_all(L, np.full(N, H_MIN), nu, c)[0]
        yk = D * x[:, None, :]
        released = np.einsum("km,jkm->j", prob.p, yk - yk_prev)
        uptake = (prob.KH[None, :] * H_MIN * nu[:, None] * L - a_prev).sum(axis=1)
        P_min = h_in + r * (released - uptake)
        return np.where(P_min - r * B_prev > H_MIN, 1, 2)

    # -- analytic Jacobian --------------------------------------------------------------------
    def _index_arrays(self) -> None:
        """Flat row / column index arrays of the own blocks, the organic-upstream blocks and
        the aqueous-link blocks (computed once)."""
        N, ns = self.N, self.ns
        base = np.arange(ns)
        rr = np.repeat(base, ns)                        # row within a block, flattened
        cc = np.tile(base, ns)
        stages = np.arange(N)
        self._r_own = (stages[:, None] * ns + rr[None, :]).reshape(-1)
        self._c_own = (stages[:, None] * ns + cc[None, :]).reshape(-1)
        jp = np.concatenate([[N - 1], np.arange(N - 1)])
        self._jp = jp
        self._g = np.ones(N)
        self._g[0] = 1.0 - self.prob.f_bleed
        self._r_prev = self._r_own
        self._c_prev = (jp[:, None] * ns + cc[None, :]).reshape(-1)
        links = self.prob.links
        self._link_j = np.array([j for j, _ in links], dtype=int)
        self._link_m = np.array([m for _, m in links], dtype=int)
        self._r_link = (self._link_j[:, None] * ns + rr[None, :]).reshape(-1)
        self._c_link = (self._link_m[:, None] * ns + cc[None, :]).reshape(-1)

    def jacobian_triplets(self, ev: _CascadeEval) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(rows, cols, vals)`` of the unscaled Jacobian ``dF / du`` (mol/h per unit
        unknown): the stage blocks, the organic-upstream blocks (stage ``j-1``; the recycle
        ``N-1`` for stage 0 with weight ``1 - f_bleed``) and the aqueous-link blocks.  Entries
        with the same (row, col) are summed by the caller."""
        prob = self.prob
        N, M, K, ns = self.N, self.M, self.K, self.ns
        O, A = prob.O, self.A
        ih, inu, ic, iB = prob.ih, prob.inu, prob.ic, prob.iB
        if not hasattr(self, "_r_own"):
            self._index_arrays()
        x, L, h, nu, c = ev.x, ev.L, ev.h, ev.nu, ev.c
        ar = np.arange(M)
        # sensitivities of the organic outlet of every stage to its own unknowns
        Yk = np.zeros((N, K, M, ns))
        Yk[:, :, ar, ar] = ev.D
        for k in range(K):
            Yk[:, k, :, M + k] = x * ev.GL[:, k, :] / L[:, k][:, None]
        Yk[:, :, :, ih] = x[:, None, :] * ev.Gh / h[:, None, None]
        nu_safe = np.where(nu > 0.0, nu, 1.0)
        Yk[:, :, :, inu] = np.where(nu[:, None, None] > 0.0,
                                    x[:, None, :] * ev.Gnu / nu_safe[:, None, None], 0.0)
        if prob.has_c:
            c_eff = np.maximum(c, _C_TINY)
            Yk[:, :, :, ic] = x[:, None, :] * ev.Gc / c_eff[:, None, None]
        a_s = np.zeros((N, K, ns))
        hn = h * nu
        for k in range(K):
            kh = prob.KH[k]
            if kh:
                a_s[:, k, M + k] = kh * hn
                a_s[:, k, ih] = kh * nu * L[:, k]
                a_s[:, k, inu] = kh * h * L[:, k]
        Y = Yk.sum(axis=1)                                   # (N, M, ns)
        Sp = np.einsum("km,nkmj->nj", prob.p, Yk)            # (N, ns)
        Sz = np.einsum("km,nkmj->nj", prob.z, Yk)
        Sa = a_s.sum(axis=1)
        Qk = np.einsum("km,nkmj->nkj", prob.q, Yk)           # (N, K, ns)
        b1 = ev.branch == 1
        b2 = ~b1
        pos = b2 & (ev.P > H_MIN)
        # own blocks
        Jo = np.zeros((N, ns, ns))
        Jo[:, :M, :] = O * Y
        Jo[:, ar, ar] += A[:, None]
        Jo[:, M:M + K, :] = O * Qk
        for k in range(K):
            kh = prob.KH[k]
            Jo[:, M + k, M + k] += O * (1.0 + kh * hn)
            if kh:
                Jo[:, M + k, ih] += O * kh * nu * L[:, k]
                Jo[:, M + k, inu] += O * kh * h * L[:, k]
        AdP = O * (Sp - Sa)                                   # A_j dP_j / du_j
        Jo[b1, ih, :] = -AdP[b1]
        Jo[:, ih, ih] += A
        Jo[b1, iB, iB] = O
        Jo[pos, iB, :] = AdP[pos]
        Jo[b2, iB, iB] += O
        Jo[:, inu, :] = O * (Sz + Sa)
        Jo[:, inu, inu] += A
        if prob.has_c:
            c_eff = np.maximum(c, _C_TINY)
            Jo[:, ic, :M] = A[:, None] * ev.phi
            Jo[:, ic, ih] += A * c * ev.dalpha_h / h
            Jo[:, ic, ic] += A * (ev.alpha_h + np.einsum("nm,nm->n", x, ev.dphi) / c_eff)
        else:
            Jo[:, ic, ic] = A
        # organic-upstream blocks (columns of stage j-1, or N-1 for stage 0)
        jp, g = self._jp, self._g
        Jp = np.zeros((N, ns, ns))
        Jp[:, :M, :] = -O * g[:, None, None] * Y[jp]
        AdP_prev = -O * g[:, None] * (Sp[jp] - Sa[jp])
        Jp[b1, ih, :] = -AdP_prev[b1]
        Jp[pos, iB, :] = AdP_prev[pos]
        later = np.arange(N) > 0
        Jp[b1 & later, ih, iB] += O
        Jp[b2 & later, iB, iB] += -O
        Jp[:, inu, :] = -O * g[:, None] * (Sz[jp] + Sa[jp])
        # aqueous-link blocks
        lj, lm = self._link_j, self._link_m
        Jl = np.zeros((lj.size, ns, ns))
        Am = A[lm]
        Jl[:, ar, ar] = -Am[:, None]
        Jl[:, inu, inu] = -Am
        lb1 = b1[lj]
        Jl[lb1, ih, ih] = -Am[lb1]
        lpos = pos[lj]
        Jl[lpos, iB, ih] = Am[lpos]
        rows = np.concatenate([self._r_own, self._r_prev, self._r_link])
        cols = np.concatenate([self._c_own, self._c_prev, self._c_link])
        vals = np.concatenate([Jo.reshape(-1), Jp.reshape(-1), Jl.reshape(-1)])
        return rows, cols, vals

    def column_factors(self, ev: _CascadeEval) -> np.ndarray:
        """``du / dw`` per unknown: ``s_x`` and ``s_B`` for the linear unknowns, the variable
        itself for the logarithmic ones (``dL / dln L = L``), 1 for a pinned ``c``."""
        prob = self.prob
        cf = np.empty((self.N, self.ns))
        cf[:, :self.M] = self.s_x
        cf[:, self.M:self.M + self.K] = ev.L
        cf[:, prob.ih] = ev.h
        cf[:, prob.inu] = ev.nu
        cf[:, prob.ic] = np.where(self.c_active, ev.c, 1.0)
        cf[:, prob.iB] = self.s_B
        return cf.reshape(-1)

    def jacobian_scaled(self, ev: _CascadeEval) -> Any:
        """Scaled Jacobian ``diag(1/row_scale) (dF/du) (du/dw)`` in the internal variables:
        dense ``ndarray`` when the number of unknowns is at most ``_DENSE_MAX``, else a
        ``scipy.sparse.csc_matrix``."""
        rows, cols, vals = self.jacobian_triplets(ev)
        n = self.prob.n
        vals = vals * (self.column_factors(ev)[cols] / self.row_scale.reshape(-1)[rows])
        if n <= _DENSE_MAX:
            J = np.zeros((n, n))
            np.add.at(J, (rows, cols), vals)
            return J
        from scipy.sparse import coo_matrix

        return coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsc()

    def _solve_step(self, J: Any, F: np.ndarray) -> np.ndarray:
        """Newton step in the scaled unknowns; least squares (and ``JACOBIAN_SINGULAR``) when
        the factorisation fails."""
        if isinstance(J, np.ndarray):
            try:
                dv = np.linalg.solve(J, -F)
                if np.all(np.isfinite(dv)):
                    return dv
            except np.linalg.LinAlgError:
                pass
            self.singular = True
            return np.linalg.lstsq(J, -F, rcond=None)[0]
        import warnings

        from scipy.sparse.linalg import MatrixRankWarning, spsolve

        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            try:
                dv = np.asarray(spsolve(J, -F)).reshape(-1)
                if np.all(np.isfinite(dv)):
                    return dv
            except (MatrixRankWarning, RuntimeError, ValueError):
                pass
        self.singular = True
        return np.linalg.lstsq(J.toarray(), -F, rcond=None)[0]

    @staticmethod
    def _freeze(J: Any, frozen: np.ndarray) -> Any:
        """``J`` with the rows of ``frozen`` unknowns replaced by identity rows."""
        if isinstance(J, np.ndarray):
            J = J.copy()
            J[frozen, :] = 0.0
            J[frozen, frozen] = 1.0
            return J
        from scipy.sparse import diags

        keep = diags((~frozen).astype(float))
        return (keep @ J + diags(frozen.astype(float))).tocsc()

    # -- the iteration ------------------------------------------------------------------------
    def run(self, w0: np.ndarray, tol: float, max_newton: int,
            ) -> tuple[np.ndarray, _CascadeEval, str, int]:
        """Damped Newton (7.5) in the internal variables: returns ``(w, ev, status,
        iterations)`` with status ``converged``, ``stalled``, ``branch_cycle`` or ``maxiter``."""
        w = np.maximum(w0.copy(), self.lb)
        ev = self.evaluate(w)
        branch_prev = None
        changes = 0
        lb = self.lb
        finite_lb = np.isfinite(lb)
        is_log = self.is_log.reshape(-1)
        for it in range(1, max_newton + 1):
            branch = ev.branch
            if ev.fmax < tol and branch_prev is not None and np.array_equal(branch, branch_prev):
                return w, ev, "converged", it
            if branch_prev is not None and not np.array_equal(branch, branch_prev):
                changes += 1
                if changes >= _BRANCH_CYCLE:
                    return w, ev, "branch_cycle", it
            else:
                changes = 0
            branch_prev = branch
            if ev.fmax < tol:
                continue                       # one more pass to confirm the branch
            J = self.jacobian_scaled(ev)
            F = ev.F.reshape(-1)
            dw = self._solve_step(J, F)
            # an unknown sitting on its bound whose step points outward is frozen for this
            # step (its row becomes the identity) and the step is re-solved (addendum WB3)
            frozen = np.zeros(w.size, dtype=bool)
            on_bound = finite_lb & (w - lb <= _BOUND_TOL * np.maximum(np.abs(w), 1.0))
            for _ in range(w.size):
                new = on_bound & (dw < 0.0) & ~frozen
                if not new.any():
                    break
                frozen |= new
                J = self._freeze(J, frozen)
                Fz = F.copy()
                Fz[frozen] = 0.0
                dw = self._solve_step(J, Fz)
                dw[frozen] = 0.0
            # fraction to boundary (0.95; a step that lands exactly on a bound is allowed since
            # the bound rows are linear) and a cap on the logarithmic components.  When the
            # limiting unknowns would allow less than _JAM_RATIO of the step, they are landed
            # on their bounds (an active-set move, addendum WB3) and the iteration continues
            # instead of shrinking the whole step
            neg = (dw < 0.0) & finite_lb
            alpha = 1.0
            if np.any(neg):
                ratio = np.full(w.size, np.inf)
                ratio[neg] = (w[neg] - lb[neg]) / (-dw[neg])
                inside = ratio >= 1.0 - 1e-9
                if not np.all(inside):
                    jam = neg & (ratio < _JAM_RATIO) & ~is_log      # linear unknowns only
                    if jam.any():
                        w = w.copy()
                        w[jam] = lb[jam]
                        ev = self.evaluate(w)
                        continue
                    alpha = min(1.0, _TAU * float(ratio[~inside].min()))
            big = float(np.abs(dw[is_log]).max()) if is_log.any() else 0.0
            if big * alpha > _LOG_STEP_CAP:
                alpha = _LOG_STEP_CAP / big
            f0 = float(F @ F)
            step = alpha
            accepted = False
            for _ in range(_MAX_BACKTRACK):
                w_try = np.maximum(w + step * dw, lb)
                ev_try = self.evaluate(w_try, branch)
                f1 = float(ev_try.F.reshape(-1) @ ev_try.F.reshape(-1))
                if np.isfinite(f1) and f1 <= (1.0 - _ARMIJO * step) * f0:
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                return w, ev, "stalled", it
            w = w_try
            ev = self.evaluate(w)              # re-evaluates the active branch
        return w, ev, "maxiter", max_newton


def _newton_result(prob: _Problem, nt: _Newton, w: np.ndarray, ev: _CascadeEval, status: str,
                   iterations: int) -> CascadeResult:
    aq, org = _streams_from_u(prob, nt.to_u(w))
    diags = []
    ok = status == "converged"
    for j in range(prob.top.n_stages):
        diags.append(_diagnostics_of(prob, aq[j], org[j], ev.D[j], iterations,
                                     float(np.abs(ev.F[j]).max()), int(ev.branch[j]),
                                     "converged" if ok else "failed", nt.singular))
    extra = frozenset({Flag.JACOBIAN_SINGULAR}) if nt.singular else frozenset()
    return _assemble(prob, aq, org, diags, "converged_newton" if ok else "failed", ev.fmax,
                     iterations, extra_flags=extra)


def solve_cascade(spec: CascadeSpec, system: SystemModel, *, method: str = "auto",
                  tol: float = 1e-11, max_newton: int = 60, max_sweeps: int = 5000,
                  ) -> CascadeResult:
    """Solve the cascade (DESIGN.md 7.5): ``method`` ``"auto"`` (Newton from the Kremser init,
    successive substitution from the same init when Newton stalls, cycles or exhausts
    ``max_newton``), ``"newton"`` (Newton only), ``"ss"`` (successive substitution only).
    Never raises on non-convergence (``status = "failed"``); ``ValueError`` on a malformed
    specification (7.1)."""
    if method not in ("auto", "newton", "ss"):
        raise ValueError(f"unknown method {method!r}")
    if not (tol > 0):
        raise ValueError("tol must be positive")
    prob = _Problem(spec, system)
    u0 = _kremser_u0(prob)
    if method == "ss":
        return _solve_ss(prob, u0, max_sweeps, _SS_TOL, _SS_BALANCE_TOL)
    u1 = _refine_init(prob, u0)
    nt = _Newton(prob, u1)
    w, ev, nstat, it = nt.run(nt.to_w(u1), tol, max_newton)
    if nstat == "converged" or method == "newton":
        return _newton_result(prob, nt, w, ev, nstat, it)
    # fallback (7.5 -> 7.6): a few exact stage sweeps, Newton again (twice, with 3 and then
    # 10 sweeps), then successive substitution continued from those sweeps with a budget
    # bounded in stage solves (addendum WB3)
    state = _SSIteration(prob, u1)
    total_it = it
    for n_sweeps in _PRESWEEPS:
        while state.sweeps < n_sweeps:
            state.sweep()
        u2 = state.unknowns()
        nt2 = _Newton(prob, u2)
        w, ev, nstat, it2 = nt2.run(nt2.to_w(u2), tol, max_newton)
        total_it += it2
        if nstat == "converged":
            return _newton_result(prob, nt2, w, ev, nstat, total_it)
    budget = min(max_sweeps, max(_SS_MIN_SWEEPS, _SS_STAGE_SOLVE_BUDGET // prob.top.n_stages))
    return _solve_ss(prob, None, budget, _SS_TOL, _SS_BALANCE_TOL, state=state)
