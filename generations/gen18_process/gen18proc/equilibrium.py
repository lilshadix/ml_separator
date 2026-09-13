"""``equilibrium.py`` — single-stage coupled multi-metal equilibrium (DESIGN.md section 6).

Given an aqueous inlet, an organic inlet and a ``SystemModel``, ``solve_stage`` finds the free
ligand of every organic ligand ``L_f^(k)``, the equilibrium aqueous ``[H+]`` ``h``, the aqueous
anion ``nu`` and the free deprotonated complexant ``c`` such that (section 6.2)

    (1) ligand k:   L_f^(k) (1 + KH_k h nu) + sum_i q_i^(k) y_i^(k) - L_T^(k) = 0
    (2) acid:       branch 1 (P - Bp >= H_MIN):  h - (P - Bp) = 0,  B_out = 0,
                                                  Na_out = Na_in + Bp
                    branch 2 (P - Bp <  H_MIN):  h = H_MIN fixed,
                                                  consumed = max(0, P - H_MIN),
                                                  B_out = (Bp - consumed) / r,
                                                  Na_out = Na_in + consumed,  flag ALKALI_EXCESS
                    P = h_in + r (released - uptake),  Bp = r B_in,
                    released = sum_k sum_i p_i^(k) (y_i^(k) - y_in,i^(k)),
                    uptake = sum_k (KH_k h nu L_f^(k) - a_in,k)
    (3) anion:      nu - nu_in + r sum_k sum_i z_i^(k) (y_i^(k) - y_in,i^(k)) + r uptake = 0
    (4) complexant: c alpha_H(h) + sum_i x_i phi_i(c) - cT_in = 0        (only when cT_in > 0)

with every metal explicit: ``T_i = x_in,i + r y_in,i``, ``D_i = sum_k D_i^(k)``,
``x_i = T_i / (1 + r D_i)``, ``y_i = (T_i - x_i) / r``, ``y_i^(k) = D_i^(k) x_i``; ``r`` is the
organic-to-aqueous flow ratio.  Units: mol/L in the phase named, flows L/h; ``H_MIN = 1e-5``
mol/L is the model constant of section 6.2 (branch 2 is inadmissible anyway).  The branch-2
formulas above close the proton and sodium ledgers of section 7.9 exactly whenever the proton
pool ``P >= H_MIN`` (the design's ``(Bp - P) / r`` closes them only to ``O(H_MIN)``); when even
the pool is below ``H_MIN`` the floor supplies at most ``H_MIN`` protons per litre of aqueous
(addendum WB2).  Per-ligand ``p`` and ``z`` reduce to the design's per-metal values when every
ligand shares them (one mechanism), and are the right generalisation for mixed systems.

Primary path: projected damped Newton in the log variables ``w = (ln L_f^(k)..., ln h, ln nu,
ln c)`` with the analytic Jacobian assembled from the ``DEval`` partials (chain rule through
``x_i``), residuals scaled by ``L_T^(k)``, ``max(h_in, h_hi, 1e-3)``, ``max(nu_in, 1e-3)`` and
``cT_in``; every unknown is kept inside its bracket of section 6.3 (in ln variables every
bracket end is a regular point, so the fraction-to-boundary step lands exactly on the end and the
clamp of section 6.4 is a no-op; an unknown on a bracket end whose Newton step points outward is
frozen there for that step, addendum WB2); Armijo backtracking on ``||F||^2`` (10 halvings);
convergence when ``max |F_scaled| < tol`` with the acid branch unchanged between the last two
iterations.  Fallback path (guaranteed by the monotonicity of section 6.3): nested bracketed
Illinois — outer on ``L_f^(1)``, then ``L_f^(2)`` ..., ``h``, ``nu``, ``c`` — each in ln-space
to 1e-13 (relative), each capped at 100 iterations, with the same branch rule; used when the primary
does not converge, or on request (``method="bracket"``).  Both paths agree to 1e-10 relative
(test).  ``status`` is ``"converged"`` or ``"failed"``; nothing here raises on non-convergence,
and ``ValueError`` is raised only on malformed input (non-positive flow, negative concentration,
unknown metal, missing ligand total, complexant in a stream without a complexant model).  The
fallback reports ``converged`` only when its final scaled residual is below
``BRACKET_RESIDUAL_TOL`` (a ligand level without a root in its bracket — ``ConstantD`` loaded past
``L_T / q`` — is ``failed`` with ``LOADING_CAP_HIT``).

Helpers for the other work blocks: ``stage_state(aq, org, system)`` rebuilds the ``StageState`` of
a solved stage from its two output streams, ``free_complexant`` recovers the solved free
complexant from residual (4) (``StageDiagnostics`` has no complexant field), and
``proton_ledger`` is the ``Q`` of section 7.9 for one stream pair.

Sources: DESIGN.md sections 5.2-5.4 and 6; addendum WB2 for the deviations named above.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .dmodel import (
    LN10,
    LOG10_CAP,
    SystemModel,
    _core_via_evaluate,
    _ModelBase,
    ligand_loading_fraction,
)
from .types import AqStream, Flag, OrgStream, StageDiagnostics, StageState

__all__ = [
    "H_MIN", "NU_MIN", "FLOOR_REL", "BRACKET_RESIDUAL_TOL", "solve_stage", "stage_state",
    "proton_ledger", "free_complexant",
]

H_MIN = 1e-5
"""Lowest admissible aqueous [H+] (mol/L): the fixed value of branch 2 (DESIGN.md section 6.2)."""

NU_MIN = 1e-9
"""Lower bracket end of the aqueous anion (mol/L), DESIGN.md section 6.3."""

FLOOR_REL = 1e-30
"""Lower bracket end of ``L_f`` and ``c`` relative to their totals (the design's 0, made positive
for the log variables; a root at the floor is the cap and is reported as ``LOADING_CAP_HIT``)."""

BRACKET_RESIDUAL_TOL = 1e-9
"""Largest scaled residual the fallback path may leave and still report ``converged``: the nested
Illinois locates every unknown to 1e-13 relative, so a residual above this means a level had no
root inside its bracket (a ``ConstantD`` ligand loaded past ``L_T / q``: the root sits on the
floor, ``LOADING_CAP_HIT``) and the stage is reported ``failed``."""

_ARMIJO = 1e-4
_MAX_BACKTRACK = 10
_MAX_BRANCH_SWITCHES = 10
_BRACKET_TOL = 1e-13
_BRACKET_MAX_IT = 100
_SMALL_SOLVE_N = 8


# ---------------------------------------------------------------------------------------------
# one evaluation of the residuals
# ---------------------------------------------------------------------------------------------

@dataclass(slots=True)
class _Eval:
    """Everything computed at one candidate ``(L, h, nu, c)``.

    ``F`` is the scaled residual vector (ligands..., acid, anion, complexant) with inactive rows
    (acid on branch 2, complexant without complexant) set to 0; ``g_h`` is the unscaled branch-1
    acid residual, kept on both branches for the branch rule.
    """

    L: np.ndarray
    h: float
    nu: float
    c: float
    Dk: list                    # K arrays (M,): D per ligand and metal (capped at LOG10_CAP)
    Ds: np.ndarray              # (M,) effective D
    den: np.ndarray             # (M,) 1 + r D
    x: np.ndarray               # (M,) aqueous total metal
    y: np.ndarray               # (M,) organic metal (sum over ligands)
    yk: list                    # K arrays (M,)
    sl: list                    # K arrays (M,) d log10 D / d ln L (own ligand)
    sh: list
    snu: list
    sc: list
    uptake_k: list              # K floats
    released: float
    P: float
    alpha_h: float
    dalpha_h: float
    phi: np.ndarray             # (M,)
    dphi: np.ndarray            # (M,) d phi / d ln c
    g_h: float
    F: np.ndarray
    branch: int


def _solve_small(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Gaussian elimination with partial pivoting for the tiny (n <= 8) stage systems (an order
    of magnitude cheaper than ``numpy.linalg.solve`` at this size); ``ZeroDivisionError`` when a
    pivot vanishes (the caller falls back to least squares and flags ``JACOBIAN_SINGULAR``)."""
    m = A.tolist()
    v = b.tolist()
    n = len(v)
    for col in range(n):
        piv = col
        best = abs(m[col][col])
        for i in range(col + 1, n):
            a = abs(m[i][col])
            if a > best:
                best, piv = a, i
        if best == 0.0:
            raise ZeroDivisionError("singular Jacobian")
        if piv != col:
            m[col], m[piv] = m[piv], m[col]
            v[col], v[piv] = v[piv], v[col]
        prow = m[col]
        pv = prow[col]
        vc = v[col]
        for i in range(col + 1, n):
            row = m[i]
            f = row[col] / pv
            if f != 0.0:
                for j in range(col, n):
                    row[j] -= f * prow[j]
                v[i] -= f * vc
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        row = m[i]
        s = v[i]
        for j in range(i + 1, n):
            s -= row[j] * x[j]
        x[i] = s / row[i]
    return np.array(x)


class _StageProblem:
    """Precomputed data of one stage and the two solution paths."""

    def __init__(self, aq_in: AqStream, org_in: OrgStream, system: SystemModel):
        self.aq = aq_in
        self.org = org_in
        self.system = system
        self.activity = system.activity
        if not (math.isfinite(aq_in.flow_L_h) and aq_in.flow_L_h > 0):
            raise ValueError(f"aqueous flow must be positive, got {aq_in.flow_L_h!r}")
        if not (math.isfinite(org_in.flow_L_h) and org_in.flow_L_h > 0):
            raise ValueError(f"organic flow must be positive, got {org_in.flow_L_h!r}")
        self.r = r = org_in.flow_L_h / aq_in.flow_L_h
        self.ligands = tuple(system.dmodels)
        self.models = [system.dmodels[k] for k in self.ligands]
        K = len(self.models)
        if K == 0:
            raise ValueError("the system model has no ligand model")
        metals = list(aq_in.metals)
        metals += [m for m in org_in.metals if m not in aq_in.metals]
        metals_t = tuple(metals)
        self.metals = metals_t
        M = len(metals)
        self.K, self.M = K, M
        self.cores: list[Callable[..., tuple]] = []
        self.idx: list[np.ndarray | None] = []
        q_list: list[np.ndarray] = []
        p_list: list[np.ndarray] = []
        z_list: list[np.ndarray] = []
        for md in self.models:
            if isinstance(md, _ModelBase):
                unknown = [m for m in metals if m not in md._index]
                if unknown:
                    raise ValueError(f"metals {unknown} are not parameterised for ligand "
                                     f"{md.ligand!r}")
                q, p, z, index = md.stage_arrays(metals_t)
                self.cores.append(md.core)
            else:
                unknown = [m for m in metals if m not in md.metals]
                if unknown:
                    raise ValueError(f"metals {unknown} are not parameterised for ligand "
                                     f"{md.ligand!r}")
                q = np.array([float(md.q.get(m, 0.0)) for m in metals])
                p = np.array([float(md.p.get(m, 0.0)) for m in metals])
                z = np.array([float(md.z.get(m, 0.0)) for m in metals])
                pos = [md.metals.index(m) for m in metals]
                index = None if pos == list(range(len(md.metals))) else np.asarray(pos)
                core = getattr(md, "core", None)
                if core is None:
                    self.cores.append(lambda L, h, nu, c, lt, cap, _md=md: _core_via_evaluate(
                        _md, L, h, nu, c, lt, cap))
                else:
                    self.cores.append(core)
            self.idx.append(index)
            q_list.append(q)
            p_list.append(p)
            z_list.append(z)
        self.q, self.p, self.z = q_list, p_list, z_list
        self.p_on = [bool(np.any(p != 0.0)) for p in p_list]
        self.z_on = [bool(np.any(z != 0.0)) for z in z_list]
        self.KH = [float(getattr(md, "k_acid_uptake", 0.0) or 0.0) for md in self.models]
        x_list = [float(aq_in.metals.get(m, 0.0)) for m in metals]
        y_list = [float(org_in.metals.get(m, 0.0)) for m in metals]
        for v in x_list + y_list:
            if not (math.isfinite(v) and v >= 0.0):
                raise ValueError("metal concentrations must be finite and non-negative")
        x_in = np.array(x_list)
        y_in = np.array(y_list)
        self.x_in, self.y_in = x_in, y_in
        self.T = x_in + r * y_in
        if K == 1:
            yk_in = [y_in]
        else:
            yk_in = []
            for name in self.ligands:
                sub = org_in.metals_by_ligand.get(name, {})
                yk_in.append(np.array([float(sub.get(m, 0.0)) for m in metals]))
            if not np.allclose(sum(yk_in), y_in, rtol=1e-9, atol=1e-15):
                raise ValueError("org_in.metals_by_ligand does not sum to org_in.metals")
        self.yk_in = yk_in
        LT = []
        for name in self.ligands:
            if name not in org_in.ligand_total:
                raise ValueError(f"org_in.ligand_total lacks ligand {name!r}")
            lt = float(org_in.ligand_total[name])
            if not (math.isfinite(lt) and lt > 0):
                raise ValueError(f"ligand total of {name!r} must be positive, got {lt!r}")
            LT.append(lt)
        self.LT = LT
        scale_fn = getattr(self.activity, "effective_ligand_total", None)
        self.LT_eff = [float(scale_fn(v)) for v in LT] if scale_fn is not None else list(LT)
        a_in = [float(org_in.acid_in_org.get(name, 0.0)) for name in self.ligands]
        for val, label in ((aq_in.h, "h"), (aq_in.anion, "anion"),
                           (aq_in.complexant_total, "complexant_total"), (aq_in.sodium, "sodium"),
                           (org_in.alkali_reserve, "alkali_reserve"), *((a, "acid_in_org")
                                                                        for a in a_in)):
            if not (math.isfinite(val) and val >= 0):
                raise ValueError(f"{label} must be finite and non-negative, got {val!r}")
        self.a_in = a_in
        self.h_in = float(aq_in.h)
        self.nu_in = float(aq_in.anion)
        self.cT = float(aq_in.complexant_total)
        self.Na_in = float(aq_in.sodium)
        self.Bp = r * float(org_in.alkali_reserve)
        self.cm = system.complexant_model
        self.has_c = self.cT > 0.0
        if self.has_c and self.cm is None:
            raise ValueError("the aqueous inlet carries a complexant but the system model has "
                             "no ComplexantSpec")
        if self.cm is not None:
            missing = [m for m in metals if m not in self.cm.metals]
            if missing:
                raise ValueError(f"complexant lacks log_beta for {missing}")
        # brackets (section 6.3): the upper ends are the physical maxima of each unknown
        p_max = p_list[0] if K == 1 else np.max(np.array(p_list), axis=0)
        a_sum = sum(a_in)
        h_hi = self.h_in + float(p_max @ self.T) + r * a_sum
        self.h_hi = max(h_hi, 2.0 * H_MIN)
        nu_hi = self.nu_in + r * sum(float(z_list[k] @ yk_in[k]) for k in range(K)) + r * a_sum
        self.nu_hi = max(nu_hi, 2.0 * NU_MIN)
        n = K + 3
        self.n = n
        self.ih, self.inu, self.ic = K, K + 1, K + 2
        lb = np.empty(n)
        ub = np.empty(n)
        for k in range(K):
            lb[k] = math.log(FLOOR_REL * self.LT_eff[k])
            ub[k] = math.log(self.LT_eff[k])
        lb[K], ub[K] = math.log(H_MIN), math.log(self.h_hi)
        lb[K + 1], ub[K + 1] = math.log(NU_MIN), math.log(self.nu_hi)
        if self.has_c:
            lb[K + 2], ub[K + 2] = math.log(FLOOR_REL * self.cT), math.log(self.cT)
        else:
            lb[K + 2] = ub[K + 2] = 0.0
        self.lb, self.ub = lb, ub
        scales = np.ones(n)
        scales[:K] = self.LT_eff
        scales[K] = max(self.h_in, self.h_hi, 1e-3)
        scales[K + 1] = max(self.nu_in, 1e-3)
        scales[K + 2] = self.cT if self.has_c else 1.0
        self.scales = scales
        self.evaluations = 0
        self.singular = False
        self._zero_m = np.zeros(M)

    # -- initial point ---------------------------------------------------------------------
    def initial(self, warm: Any) -> np.ndarray:
        """Starting point: the tracer point (``L = L_T``, inlet acid and anion, ``c = cT /
        alpha_H``) unless ``warm`` supplies a previous solution (``StageDiagnostics`` -> free
        ligand from its loading fraction; ``StageState``; a mapping with ``ligand_free`` / ``h``
        / ``anion`` / ``c_free``; or a tuple/list holding any of ``AqStream``, ``OrgStream``,
        ``StageDiagnostics`` such as the triple returned by ``solve_stage``)."""
        K = self.K
        L0 = list(self.LT_eff)
        h0, nu0, c0 = self.h_in, self.nu_in, None
        items = list(warm) if isinstance(warm, (tuple, list)) else [warm]
        for item in items:
            if item is None:
                continue
            if isinstance(item, StageDiagnostics):
                for k, name in enumerate(self.ligands):
                    lam = item.loading_fraction.get(name)
                    if lam is not None and math.isfinite(lam):
                        L0[k] = self.LT[k] * (1.0 - lam)
            elif isinstance(item, StageState):
                for k, name in enumerate(self.ligands):
                    L0[k] = float(item.ligand_free.get(name, L0[k]))
                h0, nu0, c0 = item.h, item.anion, item.c_free
            elif isinstance(item, AqStream):
                h0, nu0 = item.h, item.anion
            elif isinstance(item, OrgStream):
                for k, name in enumerate(self.ligands):
                    L0[k] = float(item.ligand_free.get(name, L0[k]))
            elif isinstance(item, Mapping):
                free = item.get("ligand_free") or {}
                for k, name in enumerate(self.ligands):
                    L0[k] = float(free.get(name, L0[k]))
                h0 = float(item.get("h", h0))
                nu0 = float(item.get("anion", nu0))
                c0 = item.get("c_free", c0)
            else:
                raise ValueError("warm must be a StageDiagnostics, StageState, AqStream, "
                                 "OrgStream, a mapping, or a tuple of those")
        w = np.empty(self.n)
        for k in range(K):
            lk = L0[k]
            if not (lk is not None and math.isfinite(lk) and lk > 0):
                lk = self.LT_eff[k]
            w[k] = math.log(min(max(lk, FLOOR_REL * self.LT_eff[k]), self.LT_eff[k]))
        if not (h0 is not None and math.isfinite(h0) and h0 > 0):
            h0 = self.h_in
        if not (nu0 is not None and math.isfinite(nu0) and nu0 > 0):
            nu0 = self.nu_in
        w[K] = math.log(min(max(float(h0), H_MIN), self.h_hi))
        w[K + 1] = math.log(min(max(float(nu0), NU_MIN), self.nu_hi))
        if self.has_c:
            if c0 is None or not (c0 > 0):
                c0 = self.cT / self.cm.alpha_h(math.exp(w[K]))
            w[K + 2] = math.log(min(max(float(c0), FLOOR_REL * self.cT), self.cT))
        else:
            w[K + 2] = 0.0
        return w

    # -- residuals -------------------------------------------------------------------------
    def eval_w(self, w: np.ndarray, branch: int) -> _Eval:
        K = self.K
        L = np.exp(w[:K])
        h = H_MIN if branch == 2 else math.exp(w[K])
        nu = math.exp(w[K + 1])
        c = math.exp(w[K + 2]) if self.has_c else 0.0
        return self.evaluate(L, h, nu, c, branch)

    def evaluate(self, L: np.ndarray, h: float, nu: float, c: float, branch: int) -> _Eval:
        self.evaluations += 1
        K, r = self.K, self.r
        T = self.T
        hn = h * nu
        Dk: list = []
        sl: list = []
        sh: list = []
        snu: list = []
        sc: list = []
        Ds = None
        Lf = L.tolist()
        for k in range(K):
            lo, a, b, cc, d = self.cores[k](Lf[k], h, nu, c, self.LT[k], self.LT_eff[k])
            idx = self.idx[k]
            if idx is not None:
                lo, a, b, cc, d = lo[idx], a[idx], b[idx], cc[idx], d[idx]
            D = np.power(10.0, np.minimum(lo, LOG10_CAP))
            Dk.append(D)
            sl.append(a)
            sh.append(b)
            snu.append(cc)
            sc.append(d)
            Ds = D if Ds is None else Ds + D
        den = 1.0 + r * Ds
        x = T / den
        y = (T - x) / r
        yk = [D * x for D in Dk]
        released = 0.0
        dz = 0.0
        uptake = 0.0
        uptake_k: list = []
        g = np.zeros(self.n)
        for k in range(K):
            if self.p_on[k] or self.z_on[k]:
                dyk = yk[k] - self.yk_in[k]
                if self.p_on[k]:
                    released += float(self.p[k] @ dyk)
                if self.z_on[k]:
                    dz += float(self.z[k] @ dyk)
            kh = self.KH[k]
            up = kh * hn * Lf[k] if kh else 0.0
            upk = up - self.a_in[k]
            uptake_k.append(upk)
            uptake += upk
            g[k] = Lf[k] + up + float(self.q[k] @ yk[k]) - self.LT_eff[k]
        P = self.h_in + r * (released - uptake)
        g_h = h - (P - self.Bp)
        g[K] = g_h if branch == 1 else 0.0
        g[K + 1] = nu - self.nu_in + r * dz + r * uptake
        if self.has_c:
            alpha_h = self.cm.alpha_h(h)
            dalpha_h = self.cm.dalpha_h_dlnh(h)
            _, phi, dphi = self.cm.terms(c, self.metals)
            g[K + 2] = c * alpha_h + float(x @ phi) - self.cT
        else:
            alpha_h, dalpha_h = 1.0, 0.0
            phi = dphi = self._zero_m
        F = g / self.scales
        return _Eval(L=L, h=h, nu=nu, c=c, Dk=Dk, Ds=Ds, den=den, x=x, y=y, yk=yk, sl=sl, sh=sh,
                     snu=snu, sc=sc, uptake_k=uptake_k, released=released, P=P, alpha_h=alpha_h,
                     dalpha_h=dalpha_h, phi=phi, dphi=dphi, g_h=g_h, F=F, branch=branch)

    # -- analytic Jacobian in the log variables ----------------------------------------------
    def jacobian(self, ev: _Eval) -> np.ndarray:
        """``d F / d w`` (scaled rows); inactive rows (acid on branch 2, complexant without
        complexant) are identity rows so that their step is zero.

        With ``G_k[j] = d D_k / d w_j`` (rows: own ligand, h, nu, c; zero for other ligands),
        ``dx/dw = -(x r / den) sum_k G_k`` and ``dy^(k)/dw = x G_k + D_k dx/dw``; for a single
        ligand this collapses to ``dy/dw = G x / den``.
        """
        K, n, r, M = self.K, self.n, self.r, self.M
        x = ev.x
        hn = ev.h * ev.nu
        has_c = self.has_c
        Lf = ev.L.tolist()
        Gs = []
        for k in range(K):
            D = ev.Dk[k] * LN10
            G = np.zeros((n, M))
            G[k] = D * ev.sl[k]
            G[K] = D * ev.sh[k]
            G[K + 1] = D * ev.snu[k]
            if has_c:
                G[K + 2] = D * ev.sc[k]
            Gs.append(G)
        if K == 1:
            dyks = [Gs[0] * (x / ev.den)]
            dx = Gs[0] * (-(x * r) / ev.den) if has_c else None
        else:
            dDs = Gs[0]
            for G in Gs[1:]:
                dDs = dDs + G
            dx = dDs * (-(x * r) / ev.den)
            dyks = [x * Gs[k] + ev.Dk[k] * dx for k in range(K)]
        J = np.zeros((n, n))
        drel = None
        dz = None
        dupt = [0.0] * n
        for k in range(K):
            dyk = dyks[k]
            J[k] = dyk @ self.q[k]
            kh = self.KH[k]
            if kh:
                up = Lf[k] * kh * hn
                J[k, k] += Lf[k] * (1.0 + kh * hn)
                J[k, K] += up
                J[k, K + 1] += up
                dupt[k] += up
                dupt[K] += up
                dupt[K + 1] += up
            else:
                J[k, k] += Lf[k]
            if self.p_on[k]:
                rk = dyk @ self.p[k]
                drel = rk if drel is None else drel + rk
            if self.z_on[k]:
                zk = dyk @ self.z[k]
                dz = zk if dz is None else dz + zk
        dupt_arr = np.array(dupt) if any(dupt) else None
        if ev.branch == 1:
            if drel is not None:
                J[K] = -r * drel
            if dupt_arr is not None:
                J[K] += r * dupt_arr
            J[K, K] += ev.h
        else:
            J[K, K] = self.scales[K]
        if dz is not None:
            J[K + 1] = r * dz
        if dupt_arr is not None:
            J[K + 1] += r * dupt_arr
        J[K + 1, K + 1] += ev.nu
        if has_c:
            J[K + 2] = dx @ ev.phi
            J[K + 2, K] += ev.c * ev.dalpha_h
            J[K + 2, K + 2] += ev.c * ev.alpha_h + float(x @ ev.dphi)
        else:
            J[K + 2, K + 2] = 1.0
        J /= self.scales[:, None]
        return J

    def _solve_step(self, J: np.ndarray, F: np.ndarray) -> np.ndarray:
        try:
            dw = _solve_small(J, -F) if self.n <= _SMALL_SOLVE_N else np.linalg.solve(J, -F)
            if not np.all(np.isfinite(dw)):
                raise ZeroDivisionError
            return dw
        except (ZeroDivisionError, np.linalg.LinAlgError):
            self.singular = True
            return np.linalg.lstsq(J, -F, rcond=None)[0]

    # -- primary path -----------------------------------------------------------------------
    def newton(self, w0: np.ndarray, tol: float, max_iter: int) -> tuple[_Eval, str, int]:
        """Projected damped Newton; returns ``(ev, status, iterations)`` with status
        ``converged``, ``stalled``, ``branch_cycle`` or ``maxiter``."""
        lb, ub, n = self.lb, self.ub, self.n
        ih = self.ih
        w = np.minimum(np.maximum(w0, lb), ub)
        branch = 1
        ev = self.eval_w(w, branch)
        prev_branch = 0
        switches = 0
        lbl = lb.tolist()
        ubl = ub.tolist()
        for it in range(1, max_iter + 1):
            F = ev.F
            fmax = float(np.abs(F).max())
            if fmax < tol:
                if branch == prev_branch:
                    return ev, "converged", it
                prev_branch = branch
                continue
            prev_branch = branch
            J = self.jacobian(ev)
            Fs = F
            dw = self._solve_step(J, Fs)
            wl = w.tolist()
            # an unknown on a bracket end whose step points outward is frozen for this step
            copied = False
            for _ in range(n):
                frozen = False
                dwl = dw.tolist()
                for j in range(n):
                    dj = dwl[j]
                    if (dj > 0.0 and wl[j] >= ubl[j]) or (dj < 0.0 and wl[j] <= lbl[j]):
                        if not copied:
                            J = J.copy()
                            Fs = Fs.copy()
                            copied = True
                        J[j, :] = 0.0
                        J[j, j] = 1.0
                        Fs[j] = 0.0
                        frozen = True
                if not frozen:
                    break
                dw = self._solve_step(J, Fs)
            # fraction to boundary (tau = 1: the limiting unknown lands on its bracket end)
            alpha = 1.0
            dwl = dw.tolist()
            for j in range(n):
                dj = dwl[j]
                if dj > 0.0:
                    gap = ubl[j] - wl[j]
                    if dj * alpha > gap:
                        alpha = gap / dj
                elif dj < 0.0:
                    gap = wl[j] - lbl[j]
                    if -dj * alpha > gap:
                        alpha = gap / (-dj)
            if not (alpha > 0.0):
                alpha = 1.0
            f0 = float(F @ F)
            step = alpha
            accepted = False
            for _ in range(_MAX_BACKTRACK):
                w_try = np.minimum(np.maximum(w + step * dw, lb), ub)
                ev_try = self.eval_w(w_try, branch)
                f1 = float(ev_try.F @ ev_try.F)
                if f1 <= (1.0 - _ARMIJO * step) * f0:
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                return ev, "stalled", it
            w, ev = w_try, ev_try
            # branch rule (section 6.2 / 6.4)
            if branch == 1:
                if w[ih] <= lb[ih] and ev.g_h > 0.0:
                    branch = 2
                    ev = self.eval_w(w, branch)
                    switches += 1
            elif ev.P - self.Bp >= H_MIN:
                branch = 1
                w[ih] = min(math.log(ev.P - self.Bp), ub[ih])
                ev = self.eval_w(w, branch)
                switches += 1
            if switches > _MAX_BRANCH_SWITCHES:
                return ev, "branch_cycle", it
        return ev, "maxiter", max_iter

    # -- fallback path ------------------------------------------------------------------------
    def bracket(self, w0: np.ndarray, tol_var: float = _BRACKET_TOL,
                max_it: int = _BRACKET_MAX_IT) -> tuple[_Eval, bool]:
        """Nested Illinois on ``L_1, ..., L_K, h, nu, c`` (ln-space); ``ok`` is False when any
        level hit its iteration cap."""
        order = list(range(self.K)) + [self.ih, self.inu] + ([self.ic] if self.has_c else [])
        vals = np.minimum(np.maximum(w0.copy(), self.lb), self.ub)
        self._ok = True
        ih = self.ih

        def residual_of(ev: _Eval, j: int) -> float:
            return ev.g_h / self.scales[ih] if j == ih else float(ev.F[j])

        def level(i: int, branch: int) -> _Eval:
            if i == len(order):
                return self.eval_w(vals, branch)
            j = order[i]
            lo, hi = self.lb[j], self.ub[j]
            vals[j] = lo
            ev_lo = level(i + 1, branch)
            f_lo = residual_of(ev_lo, j)
            if f_lo >= 0.0:
                if j == ih:
                    # the branch-1 root lies below H_MIN: branch 2 with h fixed
                    return level(i + 1, 2)
                return ev_lo
            vals[j] = hi
            ev_hi = level(i + 1, branch)
            f_hi = residual_of(ev_hi, j)
            if f_hi <= 0.0:
                return ev_hi

            def f(u: float) -> tuple[float, _Eval]:
                vals[j] = u
                ev = level(i + 1, branch)
                return residual_of(ev, j), ev

            root, ev_root, ok = _illinois(f, lo, hi, f_lo, f_hi, ev_lo, ev_hi, tol_var, max_it)
            vals[j] = root
            if not ok:
                self._ok = False
            return ev_root

        ev = level(0, 1)
        return ev, self._ok

    # -- outputs ----------------------------------------------------------------------------
    def finish(self, ev: _Eval, status: str, iterations: int,
               ) -> tuple[AqStream, OrgStream, StageDiagnostics]:
        r = self.r
        metals = self.metals
        branch = ev.branch
        if branch == 1:
            h_out = ev.h
            b_out = 0.0
            na_out = self.Na_in + self.Bp
        else:
            h_out = H_MIN
            consumed = max(0.0, ev.P - H_MIN)
            b_out = (self.Bp - consumed) / r
            na_out = self.Na_in + consumed
        xl = ev.x.tolist()
        yl = ev.y.tolist()
        x = dict(zip(metals, xl))
        y = dict(zip(metals, yl))
        yk = {name: dict(zip(metals, ev.yk[k].tolist())) for k, name in enumerate(self.ligands)}
        lig_total = dict(self.org.ligand_total)
        lig_free = dict(self.org.ligand_free)
        acid_org = dict(self.org.acid_in_org)
        Lf = ev.L.tolist()
        for k, name in enumerate(self.ligands):
            lig_total[name] = self.LT[k]
            lig_free[name] = Lf[k]
            acid_org[name] = self.KH[k] * h_out * ev.nu * Lf[k]
        for name in self.org.ligand_total:
            lig_free.setdefault(name, float(self.org.ligand_total[name]))
        aq_out = AqStream(flow_L_h=self.aq.flow_L_h, metals=x, h=float(h_out), anion=float(ev.nu),
                          complexant_total=self.cT, sodium=float(na_out))
        org_out = OrgStream(flow_L_h=self.org.flow_L_h, metals=y, metals_by_ligand=yk,
                            ligand_total=lig_total, ligand_free=lig_free, acid_in_org=acid_org,
                            alkali_reserve=float(b_out))
        state = StageState(v_aq_L=self.aq.flow_L_h, v_org_L=self.org.flow_L_h, x_total=x, y=y,
                           y_by_ligand=yk, h=float(h_out), anion=float(ev.nu), c_free=float(ev.c),
                           ligand_total=lig_total, ligand_free=lig_free, acid_in_org=acid_org,
                           alkali_reserve=float(b_out), temperature_C=self.system.temperature_C)
        flags: set[Flag] = set(self.system.flags)
        dist: dict[str, float] = {}
        for md in self.models:
            if isinstance(md, _ModelBase):
                mflags, mdist = md.state_flags(state, self.activity)
                flags |= mflags
                for key, val in mdist.items():
                    if val > dist.get(key, 0.0):
                        dist[key] = val
            else:
                for m in metals:
                    de = md.evaluate(m, state, self.activity)
                    flags |= de.flags
                    for key, val in de.ood_distance.items():
                        if val > dist.get(key, 0.0):
                            dist[key] = val
        if branch == 2:
            flags.add(Flag.ALKALI_EXCESS)
        if status != "converged":
            flags.add(Flag.NOT_CONVERGED)
        if self.singular:
            flags.add(Flag.JACOBIAN_SINGULAR)
        residual_max = float(np.abs(ev.F).max())
        loading = {name: ligand_loading_fraction(state, name, self.models[k].q)
                   for k, name in enumerate(self.ligands)}
        diag = StageDiagnostics(
            d=dict(zip(metals, ev.Ds.tolist())),
            d_by_ligand={name: dict(zip(metals, ev.Dk[k].tolist()))
                         for k, name in enumerate(self.ligands)},
            loading_fraction=loading, iterations=int(iterations), residual_max=residual_max,
            branch=int(branch), flags=frozenset(flags), ood_distance=dist, status=status)
        return aq_out, org_out, diag


def _illinois(f: Callable[[float], tuple[float, _Eval]], a: float, b: float, fa: float,
              fb: float, ea: _Eval, eb: _Eval, tol: float, max_it: int,
              ) -> tuple[float, _Eval, bool]:
    """Illinois (modified regula falsi) on ``[a, b]`` with ``fa < 0 < fb``; returns the endpoint
    with the smaller |f| once ``|b - a| <= tol`` (``ok``) or after ``max_it`` evaluations."""
    fa_s, fb_s = fa, fb            # secant values (the Illinois halving applies to these only)
    side = 0
    for _ in range(max_it):
        if abs(b - a) <= tol:
            break
        c = (a * fb_s - b * fa_s) / (fb_s - fa_s)
        if not (a < c < b):
            c = 0.5 * (a + b)
        fc, ec = f(c)
        if fc == 0.0:
            return c, ec, True
        if fc > 0.0:
            b, fb, fb_s, eb = c, fc, fc, ec
            if side == 1:
                fa_s *= 0.5
            side = 1
        else:
            a, fa, fa_s, ea = c, fc, fc, ec
            if side == -1:
                fb_s *= 0.5
            side = -1
    ok = abs(b - a) <= tol
    if abs(fa) <= abs(fb):
        return a, ea, ok
    return b, eb, ok


# ---------------------------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------------------------

def solve_stage(aq_in: AqStream, org_in: OrgStream, system: SystemModel, *, tol: float = 1e-12,
                max_iter: int = 200, warm: Any = None, method: str = "auto",
                ) -> tuple[AqStream, OrgStream, StageDiagnostics]:
    """Solve one countercurrent stage (module docstring; DESIGN.md section 6.4).

    ``warm`` seeds the primary path (see ``_StageProblem.initial``): a ``StageDiagnostics``
    (free ligand from its loading fraction), a ``StageState``, the ``(aq_out, org_out, diag)``
    triple of a previous call, or a mapping with ``ligand_free`` / ``h`` / ``anion`` / ``c_free``.
    ``method``: ``"auto"`` (Newton, then the bracketed fallback), ``"newton"`` (primary only),
    ``"bracket"`` (fallback only).  Never raises on non-convergence: ``status == "failed"`` with
    ``NOT_CONVERGED`` in the flags.  ``iterations`` counts Newton iterations when the primary
    path converged, else the total number of residual evaluations.
    """
    if method not in ("auto", "newton", "bracket"):
        raise ValueError(f"unknown method {method!r}")
    if not (tol > 0) or max_iter < 1:
        raise ValueError("tol must be positive and max_iter >= 1")
    prob = _StageProblem(aq_in, org_in, system)
    w0 = prob.initial(warm)
    ev: _Eval | None = None
    status = "failed"
    iterations = 0
    if method in ("auto", "newton"):
        ev, nstat, iterations = prob.newton(w0, tol, max_iter)
        status = "converged" if nstat == "converged" else "failed"
    if method == "bracket" or (method == "auto" and status != "converged"):
        ev_b, ok = prob.bracket(w0)
        iterations = prob.evaluations
        ok = ok and float(np.abs(ev_b.F).max()) <= BRACKET_RESIDUAL_TOL
        if ok or ev is None:
            ev = ev_b
            status = "converged" if ok else "failed"
    assert ev is not None
    return prob.finish(ev, status, iterations)


def free_complexant(system: SystemModel, x_total: Mapping[str, float], h: float,
                    complexant_total: float, *, tol: float = 1e-14) -> float:
    """Free deprotonated complexant ``c`` (mol/L) of residual (4) at aqueous totals ``x_total``
    (mol/L), ``[H+] = h`` and total complexant ``complexant_total``:
    ``c alpha_H(h) + sum_i x_i phi_i(c) = cT``.  The left side is strictly increasing in ``c``
    (section 6.3), so the root in ``[0, cT]`` is found by bisection-safeguarded Newton to ``tol``
    relative; 0 when the total is 0 or the system has no complexant model.  This is how the solved
    ``c`` of a stage is recovered from its output streams (``stage_state``), since
    ``StageDiagnostics`` carries no complexant field (addendum WB2)."""
    cm = system.complexant_model
    cT = float(complexant_total)
    if cm is None or cT <= 0.0:
        return 0.0
    metals = tuple(m for m in x_total if m in cm.metals)
    x = np.array([float(x_total[m]) for m in metals])
    alpha_h = cm.alpha_h(float(h))

    def g_and_slope(c: float) -> tuple[float, float]:
        _, phi, dphi = cm.terms(c, metals)
        g = c * alpha_h + float(x @ phi) - cT
        slope = alpha_h + (float(x @ dphi) / c if c > 0.0 else 0.0)   # d g / d c
        return g, slope

    lo, hi = 0.0, cT
    c = cT / alpha_h
    for _ in range(200):
        g, slope = g_and_slope(c)
        if g > 0.0:
            hi = c
        else:
            lo = c
        if hi - lo <= tol * cT:
            break
        c_new = c - g / slope if slope > 0.0 else 0.5 * (lo + hi)
        if not (lo < c_new < hi):
            c_new = 0.5 * (lo + hi)
        if abs(c_new - c) <= tol * cT:
            c = c_new
            break
        c = c_new
    return float(c)


def stage_state(aq: AqStream, org: OrgStream, system: SystemModel, *, c_free: float | None = None,
                ) -> StageState:
    """A ``StageState`` from an aqueous and an organic stream at the same stage (the organic
    ``ligand_free`` as given); ``c_free`` defaults to the root of residual (4) at the aqueous
    metals of ``aq`` (``free_complexant``), i.e. the solved value for a stage's output pair."""
    if c_free is None:
        c_free = free_complexant(system, aq.metals, aq.h, aq.complexant_total)
    return StageState(v_aq_L=aq.flow_L_h, v_org_L=org.flow_L_h, x_total=dict(aq.metals),
                      y=dict(org.metals), y_by_ligand={k: dict(v) for k, v in
                                                       org.metals_by_ligand.items()},
                      h=aq.h, anion=aq.anion, c_free=float(c_free),
                      ligand_total=dict(org.ligand_total), ligand_free=dict(org.ligand_free),
                      acid_in_org=dict(org.acid_in_org), alkali_reserve=org.alkali_reserve,
                      temperature_C=system.temperature_C)


def proton_ledger(aq: AqStream, org: OrgStream, system: SystemModel) -> float:
    """``Q = A h + O (sum_k a_k - B - sum_i p_i y_i)`` of one aqueous/organic stream pair
    (DESIGN.md section 7.9), mol/h, with the per-ligand ``p`` applied to ``metals_by_ligand``
    (``org.metals`` when the system has one ligand)."""
    total = aq.flow_L_h * aq.h
    org_part = sum(org.acid_in_org.values()) - org.alkali_reserve
    single = len(system.dmodels) == 1
    for name, md in system.dmodels.items():
        sub = org.metals if single else org.metals_by_ligand.get(name, {})
        org_part -= sum(float(md.p.get(m, 0.0)) * v for m, v in sub.items())
    return total + org.flow_L_h * org_part
