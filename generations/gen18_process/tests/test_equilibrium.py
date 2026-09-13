"""Tests of ``gen18proc.equilibrium`` (DESIGN.md section 6; test rows of 12.2).

Rows covered: per-metal closure ``x + r y = T`` (rel 1e-12); primary vs fallback path agreement
(rel 1e-10); multi-start agreement (8 starts, ``L0`` geometric in (1e-6, 1] x L_T, both
mechanisms, with and without complexant; rel 1e-9); the limits D = 0 (``y = 0``, ``x = T``) and
D -> inf (``x <= 1e-12 T``, ``y = T / r``); loading lowers D (1 mM vs 50 mM at L_T = 0.1, both
mechanisms, by more than 0.05 log units) and cation exchange releases acid
(``h_out - h_in = 3 r (y - y_in)`` on branch 1, rel 1e-12); the complexant depletes the free
ligand by the bound amount and balance (4) closes (rel 1e-12); the alkali reserve titrates the
released acid first and excess base gives branch 2 with ``ALKALI_EXCESS``, ``h = H_MIN`` and a
closed proton ledger (rel 1e-12).  Extras: the analytic stage Jacobian against finite
differences (rel 1e-6), the anion ledger, HNO3 uptake, a foreign ``DModel`` implementation, the
warm start, malformed input (``ValueError``) and non-convergence as a status.
"""
from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

G18 = Path(__file__).resolve().parents[1]
if str(G18) not in sys.path:
    sys.path.insert(0, str(G18))

from gen18proc import testsystems as ts  # noqa: E402
from gen18proc.dmodel import ConstantD, SystemModel, tracer_state  # noqa: E402
from gen18proc.equilibrium import (  # noqa: E402
    H_MIN,
    _StageProblem,
    free_complexant,
    proton_ledger,
    solve_stage,
    stage_state,
)
from gen18proc.types import (  # noqa: E402
    AqStream,
    DEval,
    Flag,
    Mechanism,
    OrgStream,
    Provenance,
    ProvStatus,
    Source,
    StageDiagnostics,
    StageState,
)

REL_CLOSURE = 1e-12
REL_PATHS = 1e-10
REL_STARTS = 1e-9


def rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def configurations() -> list[tuple[str, SystemModel, AqStream]]:
    """The four fixture configurations: both mechanisms, with and without the complexant."""
    spec = ts.complexant_spec()
    return [
        ("ce", ts.two_metal_cation_exchange(), ts.feed_prnd()),
        ("solv", ts.two_metal_solvating(), ts.feed_prnd_nitrate()),
        ("ce+c", ts.two_metal_cation_exchange(complexant=spec),
         ts.feed_prnd(complexant_total=0.05)),
        ("solv+c", ts.two_metal_solvating(complexant=spec),
         ts.feed_prnd_nitrate(complexant_total=0.05)),
    ]


CONFIGS = configurations()


def stream_quantities(aq: AqStream, org: OrgStream) -> dict[str, float]:
    out = {"h": aq.h, "anion": aq.anion, "sodium": aq.sodium, "complexant": aq.complexant_total,
           "reserve": org.alkali_reserve}
    for m, v in aq.metals.items():
        out[f"x.{m}"] = v
    for m, v in org.metals.items():
        out[f"y.{m}"] = v
    for k, sub in org.metals_by_ligand.items():
        for m, v in sub.items():
            out[f"yk.{k}.{m}"] = v
    for k, v in org.ligand_free.items():
        out[f"L.{k}"] = v
    for k, v in org.acid_in_org.items():
        out[f"a.{k}"] = v
    return out


def max_rel_diff(a: tuple, b: tuple) -> float:
    qa, qb = stream_quantities(a[0], a[1]), stream_quantities(b[0], b[1])
    assert set(qa) == set(qb)
    worst = 0.0
    for key in qa:
        worst = max(worst, rel(qa[key], qb[key]))
    for m in a[2].d:
        worst = max(worst, rel(a[2].d[m], b[2].d[m]))
    return worst


def check_closure(feed: AqStream, org: OrgStream, out: tuple) -> None:
    aq, og, diag = out
    r = org.flow_L_h / feed.flow_L_h
    for m in feed.metals:
        T = feed.metals[m] + r * org.metals.get(m, 0.0)
        assert rel(aq.metals[m] + r * og.metals[m], T) < REL_CLOSURE, m
        assert aq.metals[m] >= 0.0 and og.metals[m] >= 0.0
    assert aq.flow_L_h == feed.flow_L_h and og.flow_L_h == org.flow_L_h


# ---------------------------------------------------------------------------------------------
# closure, path agreement, multi-start
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name, system, feed", CONFIGS, ids=[c[0] for c in CONFIGS])
@pytest.mark.parametrize("r", [0.25, 1.0, 3.0])
def test_closure_per_metal(name: str, system: SystemModel, feed: AqStream, r: float) -> None:
    org = ts.lean_organic(system, flow_L_h=r)
    out = solve_stage(feed, org, system)
    assert out[2].status == "converged" and out[2].residual_max < 1e-12
    check_closure(feed, org, out)
    # with metal already in the organic (a scrub-like inlet)
    loaded = ts.lean_organic(system, flow_L_h=r, metals={"Pr": 0.004, "Nd": 0.006})
    out2 = solve_stage(feed, loaded, system)
    assert out2[2].status == "converged"
    check_closure(feed, loaded, out2)


@pytest.mark.parametrize("name, system, feed", CONFIGS, ids=[c[0] for c in CONFIGS])
def test_primary_vs_fallback_agreement(name: str, system: SystemModel, feed: AqStream) -> None:
    org = ts.lean_organic(system)
    newton = solve_stage(feed, org, system, method="newton")
    bracket = solve_stage(feed, org, system, method="bracket")
    assert newton[2].status == "converged" and bracket[2].status == "converged"
    assert Flag.NOT_CONVERGED not in newton[2].flags | bracket[2].flags
    assert newton[2].branch == bracket[2].branch == 1
    assert max_rel_diff(newton, bracket) < REL_PATHS
    check_closure(feed, org, bracket)
    auto = solve_stage(feed, org, system)
    assert max_rel_diff(newton, auto) == 0.0


def test_paths_agree_on_branch_two() -> None:
    system = ts.two_metal_cation_exchange()
    feed = ts.feed_prnd()
    org = ts.lean_organic(system, alkali_reserve=0.5)
    newton = solve_stage(feed, org, system, method="newton")
    bracket = solve_stage(feed, org, system, method="bracket")
    assert newton[2].branch == bracket[2].branch == 2
    assert newton[2].status == bracket[2].status == "converged"
    assert max_rel_diff(newton, bracket) < REL_PATHS


@pytest.mark.parametrize("name, system, feed", CONFIGS, ids=[c[0] for c in CONFIGS])
def test_multi_start_agreement(name: str, system: SystemModel, feed: AqStream) -> None:
    org = ts.lean_organic(system)
    ref = solve_stage(feed, org, system)
    for f in np.geomspace(1e-6, 1.0, 8):
        warm = {"ligand_free": {k: f * org.ligand_total[k] for k in system.ligands}}
        out = solve_stage(feed, org, system, warm=warm, method="newton")
        assert out[2].status == "converged", f
        assert max_rel_diff(ref, out) < REL_STARTS, f


# ---------------------------------------------------------------------------------------------
# limits D = 0 and D -> inf
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("r", [0.5, 1.0, 2.0])
def test_d_limits(r: float) -> None:
    system = ts.constant_d_system({"Pr": -math.inf, "Nd": 60.0})
    feed = ts.feed_prnd()
    org = ts.lean_organic(system, flow_L_h=r)
    for method in ("newton", "bracket"):
        aq, og, diag = solve_stage(feed, org, system, method=method)
        assert diag.status == "converged"
        assert og.metals["Pr"] == 0.0 and aq.metals["Pr"] == feed.metals["Pr"]
        assert diag.d["Pr"] == 0.0 and diag.d["Nd"] == 1e60
        T = feed.metals["Nd"]
        assert aq.metals["Nd"] <= 1e-12 * T
        assert rel(og.metals["Nd"], T / r) < 1e-12
        assert og.ligand_free[ts.CONSTANT_LIGAND] == og.ligand_total[ts.CONSTANT_LIGAND]
        # p = z = 0: acid and anion pass through (to round-off of the log variables)
        assert rel(aq.h, feed.h) < 1e-14 and rel(aq.anion, feed.anion) < 1e-14
    # a metal that is in the organic inlet only, with D = 0, is stripped completely
    org2 = ts.lean_organic(system, flow_L_h=r, metals={"Pr": 0.01})
    aq, og, diag = solve_stage(replace(feed, metals={"Pr": 0.0, "Nd": 0.0}), org2, system)
    assert og.metals["Pr"] == 0.0 and rel(aq.metals["Pr"], r * 0.01) < 1e-12
    assert aq.metals["Nd"] == 0.0 and og.metals["Nd"] == 0.0


# ---------------------------------------------------------------------------------------------
# loading lowers D; cation exchange releases acid
# ---------------------------------------------------------------------------------------------

def single_metal_org(system: SystemModel, lt: float, metal: str = "Nd") -> OrgStream:
    lig = system.ligands[0]
    return OrgStream(flow_L_h=1.0, metals={metal: 0.0}, metals_by_ligand={lig: {metal: 0.0}},
                     ligand_total={lig: lt}, ligand_free={lig: lt}, acid_in_org={lig: 0.0},
                     alkali_reserve=0.0)


@pytest.mark.parametrize("mechanism", ["cation_exchange", "solvating"])
def test_loading_lowers_d(mechanism: str) -> None:
    if mechanism == "cation_exchange":
        system = ts.two_metal_cation_exchange()
        base = AqStream(flow_L_h=1.0, metals={"Nd": 0.0}, h=0.01, anion=0.31,
                        complexant_total=0.0, sodium=0.0)
    else:
        # K_H null: the corpus setting of the loading check (C1, PRE_REGISTRATION section 3);
        # with the fixture's K_H = 0.5 at 3 M HNO3 the HNO3.L species holds 82 % of the ligand
        # and D is so small that 50 mM barely loads it (checked below as monotone only)
        system = ts.two_metal_solvating(k_acid_uptake=None)
        base = AqStream(flow_L_h=1.0, metals={"Nd": 0.0}, h=3.0, anion=3.0,
                        complexant_total=0.0, sodium=0.0)
    org = single_metal_org(system, 0.1)
    logd = {}
    for mm in (1.0, 50.0):
        feed = replace(base, metals={"Nd": mm * 1e-3})
        aq, og, diag = solve_stage(feed, org, system)
        assert diag.status == "converged"
        logd[mm] = math.log10(diag.d["Nd"])
        check_closure(feed, org, (aq, og, diag))
    assert logd[50.0] < logd[1.0] - 0.05
    # the tracer value bounds both from above
    tracer = system.dmodels[system.ligands[0]].evaluate(
        "Nd", tracer_state(system, {system.ligands[0]: 0.1}, base.h, base.anion)).log_d
    assert logd[1.0] < tracer
    if mechanism == "solvating":
        with_uptake = ts.two_metal_solvating()
        org_u = single_metal_org(with_uptake, 0.1)
        d_u = {}
        for mm in (1.0, 50.0):
            aq, og, diag = solve_stage(replace(base, metals={"Nd": mm * 1e-3}), org_u, with_uptake)
            d_u[mm] = diag.d["Nd"]
            assert og.acid_in_org[ts.SOLV_LIGAND] > 0.7 * ts.SOLV_LT
        assert d_u[50.0] < d_u[1.0] < 10 ** logd[1.0]


def test_cation_exchange_releases_acid_and_solvating_does_not() -> None:
    system = ts.two_metal_cation_exchange()
    feed = ts.feed_prnd()
    for r, y_in in ((1.0, None), (0.5, {"Pr": 0.003, "Nd": 0.002}), (2.0, {"Pr": 0.0, "Nd": 0.01})):
        org = ts.lean_organic(system, flow_L_h=r, metals=y_in)
        aq, og, diag = solve_stage(feed, org, system)
        assert diag.branch == 1 and diag.status == "converged"
        dy = sum(og.metals[m] - org.metals[m] for m in feed.metals)
        assert rel(aq.h - feed.h, 3.0 * r * dy) < REL_CLOSURE
        assert og.alkali_reserve == 0.0 and aq.sodium == feed.sodium
        assert aq.anion == feed.anion                     # z = 0: no anion transport
        assert rel(proton_ledger(feed, org, system), proton_ledger(aq, og, system)) < 1e-12
    solv = ts.two_metal_solvating(k_acid_uptake=None)
    feed = ts.feed_prnd_nitrate()
    org = ts.lean_organic(solv)
    aq, og, diag = solve_stage(feed, org, solv)
    assert rel(aq.h, feed.h) < 1e-14                      # p = 0 and no uptake
    assert Flag.ACID_UPTAKE_UNMODELLED in diag.flags


def test_solvating_anion_ledger_and_acid_uptake() -> None:
    system = ts.two_metal_solvating()                  # K_H = 0.5, z = 3
    model = system.dmodels[ts.SOLV_LIGAND]
    feed = ts.feed_prnd_nitrate()
    for r in (0.5, 1.0, 2.0):
        org = ts.lean_organic(system, flow_L_h=r)
        aq, og, diag = solve_stage(feed, org, system)
        assert diag.status == "converged"
        a = og.acid_in_org[ts.SOLV_LIGAND]
        lf = og.ligand_free[ts.SOLV_LIGAND]
        assert rel(a, 0.5 * aq.h * aq.anion * lf) < 1e-12
        dy = sum(og.metals[m] - org.metals[m] for m in feed.metals)
        # residual (3): nu_out = nu_in - r (3 dy + uptake)
        assert rel(aq.anion, feed.anion - r * (3.0 * dy + a)) < 1e-11
        # residual (2), branch 1, p = 0: h_out = h_in - r uptake
        assert rel(aq.h, feed.h - r * a) < 1e-11
        # residual (1): L_f (1 + KH h nu) + q . y = L_T
        bound = sum(model.q[m] * og.metals[m] for m in feed.metals)
        assert rel(lf * (1.0 + 0.5 * aq.h * aq.anion) + bound, ts.SOLV_LT) < 1e-12
        assert rel(diag.loading_fraction[ts.SOLV_LIGAND], (bound + a) / ts.SOLV_LT) < 1e-12
        # ledgers of section 7.9 close: protons and anion
        assert rel(proton_ledger(feed, org, system), proton_ledger(aq, og, system)) < 1e-12
        anion_in = feed.flow_L_h * feed.anion
        anion_out = aq.flow_L_h * aq.anion + og.flow_L_h * (3.0 * sum(og.metals.values()) + a)
        assert rel(anion_in, anion_out) < 1e-12
    # uptake reverses at strip: an organic carrying HNO3.L into dilute acid returns it
    aq0, og0, _ = solve_stage(feed, ts.lean_organic(system), system)
    strip = AqStream(flow_L_h=1.0, metals={"Pr": 0.0, "Nd": 0.0}, h=0.01, anion=0.01,
                     complexant_total=0.0, sodium=0.0)
    aq1, og1, diag1 = solve_stage(strip, og0, system)
    assert diag1.status == "converged"
    assert og1.acid_in_org[ts.SOLV_LIGAND] < og0.acid_in_org[ts.SOLV_LIGAND]
    assert aq1.h > strip.h and aq1.anion > strip.anion
    assert rel(proton_ledger(strip, og0, system), proton_ledger(aq1, og1, system)) < 1e-12


# ---------------------------------------------------------------------------------------------
# complexant: free ligand depleted by the bound amount, balance (4) closes
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("mechanism", ["cation_exchange", "solvating"])
def test_complexant_depletes_free_ligand(mechanism: str) -> None:
    spec = ts.complexant_spec()
    cT = 0.05
    if mechanism == "cation_exchange":
        system = ts.two_metal_cation_exchange(complexant=spec)
        feed = ts.feed_prnd(complexant_total=cT)
    else:
        system = ts.two_metal_solvating(complexant=spec)
        feed = ts.feed_prnd_nitrate(complexant_total=cT)
    cm = system.complexant_model
    org = ts.lean_organic(system)
    aq, og, diag = solve_stage(feed, org, system)
    assert diag.status == "converged"
    assert aq.complexant_total == cT                 # the complexant stays aqueous
    c = free_complexant(system, aq.metals, aq.h, aq.complexant_total)
    metals = ("Pr", "Nd")
    alpha, phi, _ = cm.terms(c, metals)
    bound_to_metal = sum(aq.metals[m] * phi[i] for i, m in enumerate(metals))
    protonated = c * (cm.alpha_h(aq.h) - 1.0)
    assert 0.0 < c < cT
    assert bound_to_metal > 0.0 and protonated > 0.0
    # c_free = c_total - bound - protonated: balance (4)
    assert rel(c + bound_to_metal + protonated, cT) < REL_CLOSURE
    # the stage_state helper reports the same c and its D matches the solve
    st = stage_state(aq, og, system)
    assert rel(st.c_free, c) < 1e-12
    for i, m in enumerate(metals):
        ev = system.dmodels[system.ligands[0]].evaluate(m, st)
        assert rel(ev.d, diag.d[m]) < 1e-10
        # complexant lowers D exactly by alpha_i relative to the inner model
        inner = system.dmodels[system.ligands[0]].inner.evaluate(m, st)
        assert rel(inner.d / alpha[i], diag.d[m]) < 1e-10
    # against the same feed without complexant: Pr (the larger beta) is held back and the
    # Nd/Pr selectivity grows; Nd itself may even rise because the held-back Pr no longer
    # competes for the ligand and releases less acid (cation exchange)
    plain = (ts.two_metal_cation_exchange() if mechanism == "cation_exchange"
             else ts.two_metal_solvating())
    aq0, og0, diag0 = solve_stage(replace(feed, complexant_total=0.0), ts.lean_organic(plain),
                                  plain)
    assert og.metals["Pr"] < og0.metals["Pr"]
    assert diag.d["Pr"] < diag0.d["Pr"]
    assert (diag0.d["Nd"] / diag0.d["Pr"]) < (diag.d["Nd"] / diag.d["Pr"])
    # a complexant in the feed without a complexant model is malformed input
    with pytest.raises(ValueError):
        solve_stage(feed, ts.lean_organic(plain), plain)


# ---------------------------------------------------------------------------------------------
# alkali reserve: titration first, then branch 2
# ---------------------------------------------------------------------------------------------

def test_alkali_reserve_titrates_then_branch_two() -> None:
    system = ts.two_metal_cation_exchange()
    feed = replace(ts.feed_prnd(), sodium=0.02)
    r = 1.0
    plain = solve_stage(feed, ts.lean_organic(system), system)
    released_plain = 3.0 * sum(plain[1].metals.values())
    # (a) a reserve smaller than the released acid is titrated first (branch 1)
    b_small = 0.3 * released_plain
    org = ts.lean_organic(system, alkali_reserve=b_small)
    aq, og, diag = solve_stage(feed, org, system)
    assert diag.status == "converged" and diag.branch == 1
    assert Flag.ALKALI_EXCESS not in diag.flags
    released = 3.0 * sum(og.metals.values())
    assert rel(aq.h, feed.h + r * released - r * b_small) < REL_CLOSURE
    assert og.alkali_reserve == 0.0
    assert rel(aq.sodium, feed.sodium + r * b_small) < REL_CLOSURE
    assert aq.h < plain[0].h and og.metals["Nd"] > plain[1].metals["Nd"]   # less acid, more D
    assert rel(proton_ledger(feed, org, system), proton_ledger(aq, og, system)) < 1e-12
    sodium_in = feed.flow_L_h * feed.sodium + org.flow_L_h * org.alkali_reserve
    assert rel(sodium_in, aq.flow_L_h * aq.sodium + og.flow_L_h * og.alkali_reserve) < 1e-12
    # (b) excess base: branch 2, h = H_MIN, ALKALI_EXCESS, the reserve keeps the surplus
    b_large = 0.5
    org = ts.lean_organic(system, alkali_reserve=b_large)
    aq, og, diag = solve_stage(feed, org, system)
    assert diag.status == "converged" and diag.branch == 2
    assert Flag.ALKALI_EXCESS in diag.flags
    assert aq.h == H_MIN
    released = 3.0 * sum(og.metals.values())
    pool = feed.h + r * released                     # P (no uptake in cation exchange)
    consumed = pool - H_MIN
    assert rel(og.alkali_reserve, (r * b_large - consumed) / r) < 1e-12
    assert rel(aq.sodium, feed.sodium + consumed) < 1e-12
    assert og.alkali_reserve > 0.0
    q_in = proton_ledger(feed, org, system)
    q_out = proton_ledger(aq, og, system)
    assert abs(q_in - q_out) <= 1e-12 * max(abs(q_in), abs(q_out))
    sodium_in = feed.flow_L_h * feed.sodium + org.flow_L_h * org.alkali_reserve
    assert rel(sodium_in, aq.flow_L_h * aq.sodium + og.flow_L_h * og.alkali_reserve) < 1e-12
    check_closure(feed, org, (aq, og, diag))
    # nearly all the metal is extracted at h = H_MIN
    assert aq.metals["Nd"] < 1e-6 * feed.metals["Nd"]
    # (c) exactly the released amount plus the inlet acid leaves h at the threshold region:
    #     a reserve just below the pool stays on branch 1 with h just above H_MIN
    b_edge = (pool - 2 * H_MIN) / r
    aq, og, diag = solve_stage(feed, ts.lean_organic(system, alkali_reserve=b_edge), system)
    assert diag.status == "converged" and diag.branch == 1
    assert aq.h >= H_MIN and og.alkali_reserve == 0.0


# ---------------------------------------------------------------------------------------------
# analytic Jacobian versus finite differences
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name, system, feed", CONFIGS, ids=[c[0] for c in CONFIGS])
def test_jacobian_vs_finite_differences(name: str, system: SystemModel, feed: AqStream) -> None:
    org = ts.lean_organic(system, alkali_reserve=0.01 if name.startswith("ce") else 0.0)
    prob = _StageProblem(feed, org, system)
    w = prob.initial(None)
    w[:prob.K] -= 0.7
    w[prob.K] += 0.3
    w[prob.K + 1] -= 0.2
    if prob.has_c:
        w[prob.K + 2] -= 1.0
    for branch in (1, 2):
        ev = prob.eval_w(w, branch)
        J = prob.jacobian(ev)
        Jfd = np.zeros_like(J)
        eps = 1e-6
        for j in range(prob.n):
            wp, wm = w.copy(), w.copy()
            wp[j] += eps
            wm[j] -= eps
            Jfd[:, j] = (prob.eval_w(wp, branch).F - prob.eval_w(wm, branch).F) / (2 * eps)
        active = [i for i in range(prob.n) if not (i == prob.ih and branch == 2)
                  and not (i == prob.ic and not prob.has_c)]
        for i in active:
            row_max = float(np.abs(J[i]).max())
            for j in range(prob.n):
                a, f = J[i, j], Jfd[i, j]
                if branch == 2 and j == prob.ih:
                    continue                       # h is not an unknown on branch 2
                # rel 1e-6 on the entry; entries below 1e-8 of the row are beyond what central
                # differences resolve at round-off (a wrong entry would be of the row's size)
                assert abs(a - f) <= 1e-6 * max(abs(a), abs(f)) + 1e-8 * row_max, (
                    name, branch, i, j, a, f)


# ---------------------------------------------------------------------------------------------
# foreign DModel, warm start, malformed input, non-convergence as a status
# ---------------------------------------------------------------------------------------------

class ForeignConstant:
    """A minimal ``DModel`` that is not a ``dmodel._ModelBase`` (only the protocol)."""

    def __init__(self, log_d: dict[str, float]):
        self.ligand = "foreign"
        self.mechanism = Mechanism.SOLVATING
        self.metals = tuple(log_d)
        self.q = {m: 0.0 for m in log_d}
        self.p = dict(self.q)
        self.z = dict(self.q)
        self.domain = None
        self.provenance = Provenance(ProvStatus.ASSUMED, Source(kind="none"), None,
                                     "ASSUMED_PLACEHOLDER")
        self._log_d = dict(log_d)

    def evaluate(self, metal: str, state: StageState, activity=None) -> DEval:
        ld = self._log_d[metal]
        return DEval(ld, 10.0 ** ld, 0.0, 0.0, 0.0, 0.0, frozenset({Flag.OA_ASSUMED}), {})


def test_foreign_dmodel_and_constant_d_reproduce_the_single_stage_formula() -> None:
    log_d = {"Pr": 0.0, "Nd": math.log10(4.0)}
    foreign = SystemModel(entry=None, dmodels={"foreign": ForeignConstant(log_d)}, complexant=None,
                          activity=None, temperature_C=25.0)
    constant = ts.constant_d_system(log_d)
    feed = ts.feed_prnd()
    for system, lig in ((foreign, "foreign"), (constant, ts.CONSTANT_LIGAND)):
        for r in (0.5, 2.0):
            org = OrgStream(flow_L_h=r, metals={"Pr": 0.0, "Nd": 0.0},
                            metals_by_ligand={lig: {"Pr": 0.0, "Nd": 0.0}},
                            ligand_total={lig: 1.0}, ligand_free={lig: 1.0},
                            acid_in_org={lig: 0.0}, alkali_reserve=0.0)
            aq, og, diag = solve_stage(feed, org, system)
            assert diag.status == "converged"
            for m, ld in log_d.items():
                D = 10.0 ** ld
                # fraction extracted = D (O/A) / (1 + D (O/A)), BRIEF section 1
                assert rel(r * og.metals[m] / feed.metals[m], r * D / (1 + r * D)) < 1e-12
                assert rel(diag.d[m], D) < 1e-12
    assert Flag.OA_ASSUMED in solve_stage(feed, OrgStream(
        1.0, {"Pr": 0.0, "Nd": 0.0}, {"foreign": {"Pr": 0.0, "Nd": 0.0}}, {"foreign": 1.0},
        {"foreign": 1.0}, {"foreign": 0.0}, 0.0), foreign)[2].flags


def test_warm_start_and_diagnostics() -> None:
    system = ts.two_metal_solvating(complexant=ts.complexant_spec())
    feed = ts.feed_prnd_nitrate(complexant_total=0.05)
    org = ts.lean_organic(system)
    cold = solve_stage(feed, org, system)
    assert isinstance(cold[2], StageDiagnostics)
    for warm in (cold[2], cold, stage_state(cold[0], cold[1], system),
                 {"ligand_free": dict(cold[1].ligand_free), "h": cold[0].h,
                  "anion": cold[0].anion}):
        out = solve_stage(feed, org, system, warm=warm)
        assert out[2].status == "converged"
        assert out[2].iterations <= cold[2].iterations
        assert max_rel_diff(cold, out) < 1e-11
    with pytest.raises(ValueError):
        solve_stage(feed, org, system, warm=3.0)
    d = cold[2]
    assert set(d.d) == {"Pr", "Nd"} and set(d.d_by_ligand) == {ts.SOLV_LIGAND}
    assert d.d["Nd"] > d.d["Pr"] and 0.0 < d.loading_fraction[ts.SOLV_LIGAND] < 1.0
    assert d.residual_max < 1e-12 and d.branch == 1 and d.iterations >= 1
    assert isinstance(d.flags, frozenset) and all(isinstance(f, Flag) for f in d.flags)
    assert d.ood_distance and all(v >= 0.0 for v in d.ood_distance.values())
    # a StageState of the solved stage reproduces the diagnostics' flags through the models
    st = stage_state(cold[0], cold[1], system)
    ev = system.dmodels[ts.SOLV_LIGAND].evaluate("Nd", st)
    assert ev.flags <= d.flags


def test_malformed_input_raises_value_error() -> None:
    system = ts.two_metal_cation_exchange()
    feed = ts.feed_prnd()
    org = ts.lean_organic(system)
    with pytest.raises(ValueError):
        solve_stage(replace(feed, flow_L_h=0.0), org, system)
    with pytest.raises(ValueError):
        solve_stage(feed, replace(org, flow_L_h=-1.0), system)
    with pytest.raises(ValueError):
        solve_stage(replace(feed, metals={"Pr": 0.025, "Nd": 0.075, "Sm": 0.01}), org, system)
    with pytest.raises(ValueError):
        solve_stage(replace(feed, metals={"Pr": -0.025, "Nd": 0.075}), org, system)
    with pytest.raises(ValueError):
        solve_stage(feed, replace(org, ligand_total={}), system)
    with pytest.raises(ValueError):
        solve_stage(feed, replace(org, ligand_total={ts.CE_LIGAND: 0.0}), system)
    with pytest.raises(ValueError):
        solve_stage(replace(feed, h=-0.01), org, system)
    with pytest.raises(ValueError):
        solve_stage(feed, org, system, method="secant")
    with pytest.raises(ValueError):
        solve_stage(feed, org, system, tol=0.0)


def test_non_convergence_is_a_status_not_an_exception() -> None:
    # ConstantD with q = 3 and more metal than 3 y <= L_T allows: no root in the bracket
    system = ts.constant_d_system({"Pr": 6.0, "Nd": 6.0}, q=3.0)
    feed = ts.feed_prnd()
    org = ts.lean_organic(system, ligand_total={ts.CONSTANT_LIGAND: 0.1})
    aq, og, diag = solve_stage(feed, org, system)
    assert diag.status == "failed"
    assert {Flag.NOT_CONVERGED, Flag.LOADING_CAP_HIT} <= diag.flags
    assert diag.residual_max > 1e-9
    assert og.ligand_free[ts.CONSTANT_LIGAND] <= 1e-6 * 0.1
    check_closure(feed, org, (aq, og, diag))              # closure holds even then
    # the same model within capacity converges and reports the loading
    lean = ts.lean_organic(system, ligand_total={ts.CONSTANT_LIGAND: 1.0})
    aq, og, diag = solve_stage(feed, lean, system)
    assert diag.status == "converged"
    assert rel(diag.loading_fraction[ts.CONSTANT_LIGAND], 3.0 * sum(og.metals.values())) < 1e-12
    assert rel(og.ligand_free[ts.CONSTANT_LIGAND], 1.0 - 3.0 * sum(og.metals.values())) < 1e-12


def test_h_min_floor_and_tolerances() -> None:
    # a feed without acid on the cation-exchange fixture: the released acid sets h
    system = ts.two_metal_cation_exchange()
    feed = replace(ts.feed_prnd(), h=0.0)
    org = ts.lean_organic(system)
    aq, og, diag = solve_stage(feed, org, system)
    assert diag.status == "converged" and diag.branch == 1
    assert rel(aq.h, 3.0 * sum(og.metals.values())) < 1e-12
    assert aq.h > H_MIN
    # a coarser tolerance converges in fewer or equal iterations with a residual below it
    loose = solve_stage(ts.feed_prnd(), org, system, tol=1e-6)
    tight = solve_stage(ts.feed_prnd(), org, system, tol=1e-12)
    assert loose[2].iterations <= tight[2].iterations and loose[2].residual_max < 1e-6
    assert max_rel_diff(loose, tight) < 1e-4
