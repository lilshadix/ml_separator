"""Tests of ``gen18proc.cascade`` (DESIGN.md section 7; test rows of 12.2).

Rows covered: Kremser (``ConstantD``, ``n_scr = n_str = 0``, ``f_bleed = 1``, E in {0.5, 1, 2,
10}, N in {1, 3, 8}: ``kremser_init``, Newton and successive substitution match the oracle to
1e-10 absolute); the analytic cascade Jacobian against central finite differences on 3/2/2
cascades with recycle (rel 1e-6; both mechanisms, with and without complexant, saponified);
Newton versus successive substitution on every stream quantity (rel 1e-8; both fixtures, with
recycle, with complexant in the scrub); ``check_balances`` (rel 1e-8); monotone raffinate in
``n_ext`` (1..10) and in O/A (0.5..5) with lean organic (slack -1e-10); the origin pass (label
sums equal the totals, identity of 7.8, rel 1e-10); the pure-target scrub (``ConstantD``:
identical ``recovery_from_feed`` for scrub T = 0 and 50 mM; loading model: ``recovery_from_feed
<= 1`` and the identity); the dilute-D-versus-loading difference (> 0.01 in recovery); failure as
a status (never raises).  Extras: topology / validation (``ValueError`` on malformed specs, dry
stages), ``n_str = 0`` product, ``regime_status_of``, ``section_tracer_d``.
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
from gen18proc.cascade import (  # noqa: E402
    ORIGIN_LABELS,
    _kremser_u0,
    _Newton,
    _Problem,
    _refine_init,
    alkali_reserve_inlet,
    build_topology,
    check_balances,
    kremser_fraction_unextracted,
    kremser_init,
    regime_status_of,
    section_tracer_d,
    solve_cascade,
    solve_cascade_ss,
)
from gen18proc.types import AqStream, CascadeResult, CascadeSpec, Flag, OrgStream  # noqa: E402

REL_NEWTON_SS = 1e-8
REL_BALANCE = 1e-8
REL_ORIGIN = 1e-10
ABS_KREMSER = 1e-10


def rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def empty_aq(flow: float = 0.0) -> AqStream:
    return AqStream(flow, {}, 0.0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------------------------

CE = ts.two_metal_cation_exchange()
CS = ts.complexant_spec()
CEC = ts.two_metal_cation_exchange(complexant=CS)
SV = ts.two_metal_solvating()
SVC = ts.two_metal_solvating(k_acid_uptake=None, complexant=CS)


def spec_ce(n=(3, 2, 2), *, sap=0.0, scrub_c=0.0, scrub_t=0.0, feed_stage=None, ret=None,
            f_bleed=0.0, fresh=None, oa=1.0, feed=None, target="Nd") -> CascadeSpec:
    scrub = AqStream(0.3, {"Pr": 0.0, "Nd": scrub_t}, 0.2, 0.2, scrub_c, 0.0)
    strip = AqStream(0.5, {"Pr": 0.0, "Nd": 0.0}, 3.0, 3.0, 0.0, 0.0)
    return CascadeSpec(n[0], n[1], n[2], feed or ts.feed_prnd(), scrub, strip, oa,
                       {ts.CE_LIGAND: ts.CE_LT_DIMER}, sap, feed_stage=feed_stage,
                       scrub_return_stage=ret, f_bleed=f_bleed, fresh_organic=fresh,
                       target=target)


def spec_sv(n=(3, 2, 2), *, scrub_c=0.0, target="Nd") -> CascadeSpec:
    scrub = AqStream(0.3, {"Pr": 0.0, "Nd": 0.0}, 1.0, 1.0, scrub_c, 0.0)
    strip = AqStream(0.5, {"Pr": 0.0, "Nd": 0.0}, 0.01, 0.01, 0.0, 0.0)
    return CascadeSpec(n[0], n[1], n[2], ts.feed_prnd_nitrate(), scrub, strip, 1.0,
                       {ts.SOLV_LIGAND: ts.SOLV_LT}, 0.0, target=target)


def stream_quantities(res: CascadeResult) -> dict[str, float]:
    out: dict[str, float] = {}
    for j, (a, o) in enumerate(zip(res.stages_aq, res.stages_org)):
        for m, v in a.metals.items():
            out[f"x.{j}.{m}"] = v
        out[f"h.{j}"], out[f"nu.{j}"], out[f"na.{j}"] = a.h, a.anion, a.sodium
        out[f"cT.{j}"] = a.complexant_total
        for m, v in o.metals.items():
            out[f"y.{j}.{m}"] = v
        for k, sub in o.metals_by_ligand.items():
            for m, v in sub.items():
                out[f"yk.{j}.{k}.{m}"] = v
        for k, v in o.ligand_free.items():
            out[f"L.{j}.{k}"] = v
        for k, v in o.acid_in_org.items():
            out[f"a.{j}.{k}"] = v
        out[f"B.{j}"] = o.alkali_reserve
    return out


def max_rel_diff(a: CascadeResult, b: CascadeResult) -> float:
    qa, qb = stream_quantities(a), stream_quantities(b)
    assert set(qa) == set(qb)
    return max(rel(qa[k], qb[k]) for k in qa)


# ---------------------------------------------------------------------------------------------
# Kremser
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("E", [0.5, 1.0, 2.0, 10.0])
@pytest.mark.parametrize("n", [1, 3, 8])
def test_kremser_oracle(E: float, n: int) -> None:
    system = ts.constant_d_system({"Pr": math.log10(E), "Nd": math.log10(E)})
    feed = AqStream(1.0, {"Pr": 0.01, "Nd": 0.02}, 0.01, 0.31, 0.0, 0.0)
    spec = CascadeSpec(n, 0, 0, feed, empty_aq(), empty_aq(), 1.0, {ts.CONSTANT_LIGAND: 1.0},
                       0.0, f_bleed=1.0, target="Nd")
    oracle = kremser_fraction_unextracted(E, n)
    prob = _Problem(spec, system)
    x0 = prob.unpack(kremser_init(spec, system))[0]
    for i, m in enumerate(prob.metals):
        assert abs(x0[0, i] / feed.metals[m] - oracle) < ABS_KREMSER
    newton = solve_cascade(spec, system, method="newton")
    ss = solve_cascade(spec, system, method="ss")
    assert newton.status == "converged_newton" and ss.status == "converged_ss"
    for res in (newton, ss):
        for m in feed.metals:
            assert abs(res.raffinate.metals[m] / feed.metals[m] - oracle) < ABS_KREMSER
        assert res.balance_rel_max < REL_BALANCE
        assert abs(res.origin["recovery_from_feed"] - (1.0 - oracle)) < ABS_KREMSER
        assert res.product is res.stripped_organic         # n_str = 0: the loaded organic
    assert kremser_fraction_unextracted(1.0, 4) == 0.2


# ---------------------------------------------------------------------------------------------
# Jacobian against finite differences
# ---------------------------------------------------------------------------------------------

JAC_CASES = [
    ("ce", CE, spec_ce()),
    ("ce_sap", CE, spec_ce(sap=0.5)),
    ("ce_complexant_scrub", CEC, spec_ce(scrub_c=0.05)),
    ("solv_uptake", SV, spec_sv()),
    ("solv_complexant", SVC, spec_sv(scrub_c=0.05)),
    ("ce_return", CE, spec_ce(feed_stage=2, ret=0)),
    ("ce_bleed", CE, spec_ce(f_bleed=0.1)),
]


@pytest.mark.parametrize("name, system, spec", JAC_CASES, ids=[c[0] for c in JAC_CASES])
def test_jacobian_vs_finite_differences(name: str, system, spec: CascadeSpec) -> None:
    prob = _Problem(spec, system)
    assert prob.top.n_stages == 7 and prob.f_bleed < 1.0            # 3/2/2 with recycle
    u0 = _refine_init(prob, _kremser_u0(prob))
    nt = _Newton(prob, u0)
    rng = np.random.default_rng(18)
    u = np.maximum(u0 * (1.0 + 0.2 * rng.uniform(-1.0, 1.0, u0.size)), 1e-9)
    w = nt.to_w(u)
    ev = nt.evaluate(w)
    branch = ev.branch
    J = nt.jacobian_scaled(ev)
    n = w.size
    Jfd = np.zeros((n, n))
    F0 = ev.F.reshape(-1)
    for col in range(n):
        eps = 1e-6 * max(abs(w[col]), 1.0)
        gap = w[col] - nt.lb[col] if np.isfinite(nt.lb[col]) else np.inf
        if gap < eps:
            # at a lower bound (a pinned complexant, an empty reserve): the residual is
            # defined only above it, so use the second-order one-sided difference
            eps = 3e-7 * max(abs(w[col]), 1.0)
            w1, w2 = w.copy(), w.copy()
            w1[col] += eps
            w2[col] += 2.0 * eps
            F1 = nt.evaluate(w1, branch).F.reshape(-1)
            F2 = nt.evaluate(w2, branch).F.reshape(-1)
            Jfd[:, col] = (-3.0 * F0 + 4.0 * F1 - F2) / (2.0 * eps)
            continue
        wp, wm = w.copy(), w.copy()
        wp[col] += eps
        wm[col] -= eps
        Fp = nt.evaluate(wp, branch).F.reshape(-1)
        Fm = nt.evaluate(wm, branch).F.reshape(-1)
        Jfd[:, col] = (Fp - Fm) / (2.0 * eps)
    err = np.abs(Jfd - J)
    denom = np.maximum(np.abs(J), np.abs(Jfd))
    rowmax = np.maximum(np.abs(J).max(axis=1, keepdims=True), 1.0)
    ok = (err <= 1e-6 * denom) | (err <= 1e-9 * rowmax)
    worst = float((err / np.maximum(denom, 1e-300))[~ok].max()) if not ok.all() else 0.0
    assert ok.all(), f"{int((~ok).sum())} entries differ; worst {worst:.2e}"
    # the recycle block (stage 0 <- stage N-1) and the aqueous couplings are present
    ns = prob.ns
    assert np.abs(J[:ns, -ns:]).max() > 0.0


# ---------------------------------------------------------------------------------------------
# Newton vs successive substitution, balances
# ---------------------------------------------------------------------------------------------

AGREE_CASES = [
    ("ce_recycle", CE, spec_ce()),
    ("ce_complexant_scrub", CEC, spec_ce(scrub_c=0.05)),
    ("ce_saponified", CE, spec_ce(sap=0.3)),
    ("ce_scrub_target", CE, spec_ce(scrub_t=0.05)),
    ("solv_recycle", SV, spec_sv()),
    ("solv_complexant_scrub", SVC, spec_sv(n=(4, 2, 2), scrub_c=0.05)),
    ("ce_feed_stage", CE, spec_ce(n=(4, 2, 2), feed_stage=2, ret=3)),
    ("ce_return_stage", CE, spec_ce(n=(4, 2, 2), feed_stage=3, ret=1)),
]


@pytest.mark.parametrize("name, system, spec", AGREE_CASES, ids=[c[0] for c in AGREE_CASES])
def test_newton_vs_successive_substitution(name: str, system, spec: CascadeSpec) -> None:
    newton = solve_cascade(spec, system, method="newton")
    ss = solve_cascade_ss(spec, system)
    assert newton.status == "converged_newton", newton.status
    assert ss.status == "converged_ss", ss.status
    assert max_rel_diff(newton, ss) < REL_NEWTON_SS
    for res in (newton, ss):
        assert res.balance_rel_max < REL_BALANCE
        recomputed = check_balances(spec, res, system)
        assert set(recomputed) == set(res.balances)
        assert max(recomputed.values()) < REL_BALANCE
        assert Flag.NOT_CONVERGED not in res.flags
        assert all(d.status == "converged" for d in res.diagnostics)
    auto = solve_cascade(spec, system)
    assert auto.status == "converged_newton"
    assert max_rel_diff(auto, newton) == 0.0
    # per-metal closure of every stage from its own diagnostics
    for a, o, d in zip(newton.stages_aq, newton.stages_org, newton.diagnostics):
        for m in a.metals:
            assert rel(o.metals[m], d.d[m] * a.metals[m]) < 1e-10 or o.metals[m] < 1e-300


def test_balances_keys_and_ledgers() -> None:
    res = solve_cascade(spec_ce(n=(4, 2, 2), sap=0.2, scrub_c=0.0), CE)
    keys = set(res.balances)
    assert {"metal.Pr", "metal.Nd", f"ligand.{ts.CE_LIGAND}", "proton", "anion", "complexant",
            "sodium", "metal_stage_max", "proton_stage_max"} <= keys
    assert res.balance_rel_max == max(res.balances.values())
    assert res.balance_rel_max < REL_BALANCE
    # sodium appears because the base was consumed
    assert res.raffinate.sodium > 0.0


# ---------------------------------------------------------------------------------------------
# monotonicity
# ---------------------------------------------------------------------------------------------

def lean_spec(n_ext: int, oa: float) -> CascadeSpec:
    return CascadeSpec(n_ext, 0, 0, ts.feed_prnd(), empty_aq(), empty_aq(), oa,
                       {ts.CE_LIGAND: ts.CE_LT_DIMER}, 0.0, f_bleed=1.0, target="Nd")


def test_raffinate_monotone_in_stages_and_oa() -> None:
    prev = math.inf
    for n_ext in range(1, 11):
        res = solve_cascade(lean_spec(n_ext, 1.0), CE)
        assert res.status.startswith("converged"), (n_ext, res.status)
        x = res.raffinate.metals["Nd"]
        assert x <= prev + 1e-10, (n_ext, x, prev)
        prev = x
    prev = math.inf
    for oa in (0.5, 1.0, 2.0, 3.0, 5.0):
        res = solve_cascade(lean_spec(3, oa), CE)
        assert res.status.startswith("converged"), (oa, res.status)
        x = res.raffinate.metals["Nd"]
        assert x <= prev + 1e-10, (oa, x, prev)
        prev = x


# ---------------------------------------------------------------------------------------------
# origin pass
# ---------------------------------------------------------------------------------------------

def check_origin(res: CascadeResult, spec: CascadeSpec) -> None:
    o = res.origin
    assert o is not None
    T = spec.target
    for a in res.stages_aq:
        assert a.labels is not None
        assert rel(sum(a.labels[T].values()), a.metals[T]) < REL_ORIGIN or a.metals[T] == 0.0
    for og in res.stages_org:
        assert og.labels is not None
        assert rel(sum(og.labels[T].values()), og.metals[T]) < REL_ORIGIN or og.metals[T] == 0.0
    assert o["label_sum_rel_defect"] < REL_ORIGIN
    A_in = o["feed_target_mol_h"]
    S_in = o["scrub_target_mol_h"]
    lhs = o["recovery_total"] - o["recovery_from_feed"]
    rhs = (o["scrub_target_return"] * S_in / A_in if S_in > 0 else 0.0) \
        + o["product_fresh_mol_h"] / A_in + o["product_strip_mol_h"] / A_in
    assert abs(lhs - rhs) < REL_ORIGIN * max(1.0, abs(lhs))
    assert abs(o["net_product_mol_h"] - (o["product_total_mol_h"] - S_in)) < 1e-14
    assert 0.0 <= o["recovery_from_feed"] <= 1.0 + REL_ORIGIN
    if S_in == 0.0:
        assert math.isnan(o["scrub_target_return"])


def test_origin_identities_feed_only() -> None:
    spec = spec_ce()
    res = solve_cascade(spec, CE)
    check_origin(res, spec)
    assert rel(res.origin["recovery_total"], res.origin["recovery_from_feed"]) < REL_ORIGIN


def test_origin_identities_with_scrub_and_fresh_target() -> None:
    fresh = ts.lean_organic(CE, metals={"Pr": 0.0, "Nd": 0.002})
    spec = spec_ce(n=(4, 2, 2), scrub_t=0.05, f_bleed=0.2, fresh=fresh)
    res = solve_cascade(spec, CE)
    assert res.status == "converged_newton"
    check_origin(res, spec)
    o = res.origin
    assert o["product_scrub_mol_h"] > 0.0 and o["product_fresh_mol_h"] > 0.0
    assert o["recovery_total"] > o["recovery_from_feed"]


# ---------------------------------------------------------------------------------------------
# pure-target scrub and dilute-D-versus-loading
# ---------------------------------------------------------------------------------------------

def test_pure_target_scrub_constant_d() -> None:
    system = ts.constant_d_system({"Pr": 0.0, "Nd": 0.3})
    strip = AqStream(0.5, {}, 3.0, 3.0, 0.0, 0.0)
    recs = []
    for scrub_t in (0.0, 0.05):
        scrub = AqStream(0.3, {"Nd": scrub_t}, 0.2, 0.2, 0.0, 0.0)
        spec = CascadeSpec(4, 2, 2, ts.feed_prnd(), scrub, strip, 1.0,
                           {ts.CONSTANT_LIGAND: 1.0}, 0.0, f_bleed=1.0, target="Nd")
        res = solve_cascade(spec, system)
        assert res.status == "converged_newton"
        check_origin(res, spec)
        recs.append(res.origin["recovery_from_feed"])
    assert abs(recs[0] - recs[1]) < ABS_KREMSER


def test_pure_target_scrub_loading_model() -> None:
    spec = spec_ce(n=(4, 2, 2), scrub_t=0.05)
    res = solve_cascade(spec, CE)
    assert res.status == "converged_newton"
    assert res.origin["recovery_from_feed"] <= 1.0 + REL_ORIGIN
    assert res.origin["recovery_total"] > res.origin["recovery_from_feed"]
    check_origin(res, spec)


def test_dilute_d_differs_from_loading_model() -> None:
    spec = spec_ce(n=(4, 2, 2))
    loaded = solve_cascade(spec, CE)
    tracer = section_tracer_d(spec, CE)
    assert set(tracer) == {"extraction", "scrub", "strip"}
    # a labelled ConstantD at the extraction-section tracer D of every metal
    const = ts.constant_d_system({m: math.log10(d) for m, d in tracer["extraction"].items()})
    spec_c = replace(spec, ligand_total={ts.CONSTANT_LIGAND: 1.0})
    dilute = solve_cascade(spec_c, const)
    assert loaded.status == "converged_newton" and dilute.status.startswith("converged")
    assert abs(loaded.origin["recovery_from_feed"] - dilute.origin["recovery_from_feed"]) > 0.01


# ---------------------------------------------------------------------------------------------
# failure as a status, validation
# ---------------------------------------------------------------------------------------------

def test_failure_is_a_status() -> None:
    shifted = ts.two_metal_cation_exchange(log_k={"Pr": -2.10 + 20.0, "Nd": -1.95 + 20.0})
    spec = spec_ce(n=(200, 0, 0), oa=1e-3, f_bleed=1.0)
    res = solve_cascade(spec, shifted, max_newton=20, max_sweeps=20)
    assert res.status in {"failed", "converged_newton", "converged_ss"}
    if res.status == "failed":
        assert Flag.NOT_CONVERGED in res.flags
        assert res.regime_status == "INADMISSIBLE"
    assert len(res.stages_aq) == 200
    assert isinstance(res.balance_rel_max, float)


@pytest.mark.parametrize("bad, match", [
    (dict(n=(0, 1, 1)), "n_ext"),
    (dict(feed_stage=5), "feed_stage"),
    (dict(ret=-1), "scrub_return_stage"),
    (dict(sap=1.5), "saponification"),
    (dict(oa=-1.0), "organic flow"),
    (dict(feed=AqStream(0.0, {"Pr": 0.025, "Nd": 0.075}, 0.01, 0.31, 0.0, 0.0)), "feed flow"),
    (dict(feed=AqStream(1.0, {"Pr": 0.025, "Nd": 0.075, "La": 0.01}, 0.01, 0.31, 0.0, 0.0)),
     "metals"),
    (dict(feed=AqStream(1.0, {"Pr": -0.025, "Nd": 0.075}, 0.01, 0.31, 0.0, 0.0)),
     "non-negative"),
    (dict(target="La"), "target"),
    (dict(n=(4, 2, 2), feed_stage=1, ret=0), "no aqueous flow"),
])
def test_malformed_spec_raises(bad: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        solve_cascade(spec_ce(**bad), CE)


def test_missing_ligand_total_and_complexant_without_model() -> None:
    spec = replace(spec_ce(), ligand_total={})
    with pytest.raises(ValueError, match="ligand_total"):
        solve_cascade(spec, CE)
    with pytest.raises(ValueError, match="complexant"):
        solve_cascade(spec_ce(scrub_c=0.05), CE)      # CE has no ComplexantSpec
    with pytest.raises(ValueError, match="method"):
        solve_cascade(spec_ce(), CE, method="bogus")


# ---------------------------------------------------------------------------------------------
# topology, sections, helpers
# ---------------------------------------------------------------------------------------------

def test_topology_flows_and_links() -> None:
    top = build_topology(spec_ce(n=(4, 2, 2), feed_stage=3, ret=1))
    assert top.n_stages == 8 and top.feed_stage == 3 and top.scrub_return_stage == 1
    # scrub liquor (0.3) flows through the scrub stages and stage 3, 2 get only the feed
    assert top.a_flow.tolist() == pytest.approx([1.3, 1.3, 1.0, 1.0, 0.3, 0.3, 0.5, 0.5])
    assert top.inc[1, 4] == 1 and top.inc[3, 4] == 0 and top.inc[2, 3] == 1
    assert top.inc[6, 7] == 1 and top.inc[5, 6] == 0        # circuits are separate
    natural = build_topology(spec_ce(n=(4, 2, 2)))
    assert natural.inc[3, 4] == 1 and natural.a_flow.tolist() == pytest.approx(
        [1.3, 1.3, 1.3, 1.3, 0.3, 0.3, 0.5, 0.5])
    no_scrub = build_topology(spec_ce(n=(3, 0, 2)))
    assert no_scrub.scrub_top is None and no_scrub.a_flow.tolist() == pytest.approx(
        [1.0, 1.0, 1.0, 0.5, 0.5])


def test_no_strip_product_is_loaded_organic_and_no_recycle() -> None:
    spec = spec_ce(n=(3, 2, 0), f_bleed=0.0)
    res = solve_cascade(spec, CE)
    assert res.status == "converged_newton"
    assert isinstance(res.product, OrgStream) and res.product is res.stripped_organic
    assert res.scrub_raffinate is not None
    assert res.balance_rel_max < REL_BALANCE
    assert res.origin["recovery_total"] > 0.0
    prob = _Problem(spec, CE)
    assert prob.f_bleed == 1.0


def test_alkali_reserve_and_regime_status() -> None:
    assert alkali_reserve_inlet(spec_ce(sap=0.25), CE) == pytest.approx(0.25 * ts.CE_HA_TOTAL)
    assert alkali_reserve_inlet(spec_sv(), SV) == 0.0
    assert regime_status_of(frozenset()) == "IN_DOMAIN"
    assert regime_status_of({Flag.HIGH_LOADING}) == "IN_DOMAIN_WITH_CAVEATS"
    assert regime_status_of({Flag.OOD_ACID, Flag.HIGH_LOADING}) == "OUT_OF_DOMAIN"
    assert regime_status_of({Flag.OOD_ACID, Flag.NOT_CONVERGED}) == "INADMISSIBLE"
    res = solve_cascade(spec_ce(sap=0.9), CE)
    assert res.status.startswith("converged")
    if any(d.branch == 2 for d in res.diagnostics):
        assert Flag.ALKALI_EXCESS in res.flags and res.regime_status == "INADMISSIBLE"


def test_origin_labels_tuple() -> None:
    assert ORIGIN_LABELS == ("feed", "scrub", "fresh", "strip")
