"""Tests of ``gen18proc.dmodel`` (DESIGN.md sections 5.1-5.4, 5.6, 5.7; test rows of 12.2).

Rows covered: analytic partials versus central finite differences (rel 1e-6); two-ligand
composition ``D = D1 + D2`` with ligand 1 depleted by ``y^(1)`` only (rel 1e-12); the complexant
lowers the effective D by exactly ``alpha_i`` and a higher ``[H+]`` lowers ``alpha`` through
protonation (rel 1e-12; monotone).  Also: the chemistry flags of 5.4, the solvating and
cation-exchange caveat flags of 5.2, ``NearestConditionD`` on a tiny frame in the
``corpus_records.csv`` layout, ``EffectiveCapacity``, and the refusals, ``parameter_draw`` and
band selection of ``build_system_model``.  Every number is a synthetic fixture (``testsystems``).
"""
from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

G18 = Path(__file__).resolve().parents[1]
if str(G18) not in sys.path:
    sys.path.insert(0, str(G18))

from gen18proc import testsystems as ts  # noqa: E402
from gen18proc.dmodel import (  # noqa: E402
    LN10,
    AqueousComplexantWrapper,
    CationExchangeMassAction,
    ComplexantModel,
    ConstantD,
    EffectiveCapacity,
    NearestConditionD,
    SolvatingMassAction,
    SystemModel,
    build_system_model,
    evaluate_composite,
    ligand_chemistry_flags,
    ligand_loading_fraction,
    parse_band,
    select_band,
    tracer_state,
)
from gen18proc.equilibrium import free_complexant, solve_stage  # noqa: E402
from gen18proc.types import (  # noqa: E402
    ApplicabilityDomain,
    DistributionRecord,
    Flag,
    LigandSpec,
    Mechanism,
    ModelBuildError,
    OrgStream,
    PhaseBehaviour,
    Provenance,
    ProvStatus,
    Source,
    Sourced,
    StageState,
    Stoichiometry,
)

REL_FD = 1e-6
REL_EXACT = 1e-12


def rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------

def loaded_state(system: SystemModel, *, lt: float, frac_free: float = 0.6, h: float = 0.05,
                 nu: float = 0.31, c_free: float = 0.01, cT: float = 0.05) -> StageState:
    """A partially loaded state of the single-ligand ``system`` (``L_f = frac_free * L_T``)."""
    lig = system.ligands[0]
    st = tracer_state(system, {lig: lt}, h, nu, cT)
    return replace(st, ligand_free={lig: frac_free * lt}, c_free=c_free)


def central_difference(model, metal: str, state: StageState, var: str, eps: float = 1e-5,
                       activity=None) -> float:
    """``d log10 D / d ln(var)`` by central differences on the state."""
    lig = model.ligand

    def perturbed(scale: float) -> StageState:
        f = math.exp(scale)
        if var == "L":
            free = dict(state.ligand_free)
            free[lig] *= f
            return replace(state, ligand_free=free)
        if var == "h":
            return replace(state, h=state.h * f)
        if var == "nu":
            return replace(state, anion=state.anion * f)
        return replace(state, c_free=state.c_free * f)

    up = model.evaluate(metal, perturbed(eps), activity).log_d
    dn = model.evaluate(metal, perturbed(-eps), activity).log_d
    return (up - dn) / (2.0 * eps)


def analytic(ev, var: str) -> float:
    return {"L": ev.dlogd_dlnL, "h": ev.dlogd_dlnh, "nu": ev.dlogd_dlnanion,
            "c": ev.dlogd_dlnc}[var]


def unknown_phase(loc_metal: float | None = None, loc_acid: float | None = None) -> PhaseBehaviour:
    lit = Provenance(ProvStatus.LITERATURE, Source(kind="doi", doi="10.0/test", locator="test"))
    return PhaseBehaviour(
        loc_metal_M=(Sourced(loc_metal, "mol/L", lit) if loc_metal is not None
                     else Sourced.unknown("mol/L")),
        loc_acid_M=(Sourced(loc_acid, "mol/L", lit) if loc_acid is not None
                    else Sourced.unknown("mol/L")),
        third_phase_observed=Sourced.unknown("1"), disengagement_s=Sourced.unknown("s"),
        ligand_loss_mol_per_L_aq=Sourced.unknown("mol/L_aq"),
        max_loading_fraction_studied=Sourced.unknown("1"), regenerability_note="")


def records_frame() -> pd.DataFrame:
    """A tiny frame in the ``systems/corpus_records.csv`` layout (addendum WB1 A5): one numeric
    ``ligand_M`` column for the primary ligand named in ``ligand_name``."""
    rows = [
        # record_id, metal, log_d, acid, ligand, mM, pub, fit_eligible
        ("r01", "Nd", 1.0, 1.0, 0.1, 0.5, "pubA", True),
        ("r02", "Nd", 2.0, 3.0, 0.1, 0.5, "pubA", True),
        ("r03", "Nd", 0.0, 1.0, 0.03, 0.5, "pubB", True),
        ("r04", "Nd", 3.0, 3.0, 0.3, 5.0, "pubB", True),
        ("r05", "Nd", 9.0, 1.0, 0.1, 0.5, "pubB", False),      # ineligible: must be ignored
        ("r06", "Pr", 0.5, 1.0, 0.1, 0.5, "pubA", True),
        ("r07", "Pr", 1.5, 3.0, 0.1, 0.5, "pubA", True),
        ("r08", "Pr", -0.5, 1.0, 0.03, 0.5, "pubB", True),
        # tie for the point (log 2, log 0.1): r09 and r10 both at distance |log 2 - log 1|
        # from acid 1 and 4 M -> equal |delta lA|, resolved by the smaller record_id
        ("r09", "Ce", 0.7, 4.0, 0.1, math.nan, "pubC", True),
        ("r10", "Ce", 0.2, 1.0, 0.1, math.nan, "pubC", True),
    ]
    df = pd.DataFrame(rows, columns=["record_id", "metal", "log_d", "acid_nominal_M", "ligand_M",
                                     "metal_initial_mM", "publication_id", "fit_eligible"])
    df["d"] = 10.0 ** df["log_d"]
    df["system_id"] = "sys_test"
    df["ligand_name"] = ts.SOLV_LIGAND
    df["anion"] = "nitrate"
    df["diluent_family"] = "aliphatic_hydrocarbon"
    df["modifiers"] = ""
    df["temperature_C"] = 25.0
    return df


# ---------------------------------------------------------------------------------------------
# 12.2: analytic partials vs central finite differences (rel 1e-6)
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("mechanism", ["cation_exchange", "solvating"])
def test_partials_vs_finite_differences(mechanism: str) -> None:
    if mechanism == "cation_exchange":
        system = ts.two_metal_cation_exchange(complexant=ts.complexant_spec())
        lt = ts.CE_LT_DIMER
    else:
        # p_h != 0 so that the acid partial is exercised on the solvating side too
        system = ts.two_metal_solvating(complexant=ts.complexant_spec(),
                                        parameter_draw={f"{ts.SOLV_LIGAND}.p_h": 0.4})
        lt = ts.SOLV_LT
    model = system.dmodels[system.ligands[0]]
    assert isinstance(model, AqueousComplexantWrapper)
    state = loaded_state(system, lt=lt)
    for metal in ("Pr", "Nd"):
        ev = model.evaluate(metal, state)
        ev_inner = model.inner.evaluate(metal, state)
        for var in ("L", "h", "nu", "c"):
            for md, e in ((model, ev), (model.inner, ev_inner)):
                fd = central_difference(md, metal, state, var)
                an = analytic(e, var)
                if an == 0.0:
                    assert abs(fd) < 1e-9, (mechanism, metal, var)
                else:
                    assert rel(an, fd) < REL_FD, (mechanism, metal, var, an, fd)
    # the inner model has no complexant partial; the wrapper's is -phi / ln10
    assert model.inner.evaluate("Pr", state).dlogd_dlnc == 0.0
    assert model.evaluate("Pr", state).dlogd_dlnc < 0.0


def test_partials_anion_complexation_and_nearest() -> None:
    entry = ts.cation_exchange_entry()
    params = entry.params[ts.CE_LIGAND]["20-30C"]
    # beta_anion holds linear cumulative betas (M^-j), DESIGN 3.2 / 5.2 (d): alpha gains
    # b1 nu + b2 nu^2
    beta = {m: (ts.assumed(2.0, "L/mol"), ts.assumed(0.3, "L2/mol2")) for m in ("Pr", "Nd")}
    params = replace(params, beta_anion=beta)
    model = CationExchangeMassAction(params, entry.organic_ligands[0], entry.medium, None)
    system = SystemModel(entry=None, dmodels={ts.CE_LIGAND: model}, complexant=None,
                         activity=None, temperature_C=25.0)
    state = loaded_state(system, lt=ts.CE_LT_DIMER, c_free=0.0, cT=0.0)
    ev = model.evaluate("Nd", state)
    assert ev.dlogd_dlnanion < 0.0
    for var in ("L", "h", "nu"):
        assert rel(analytic(ev, var), central_difference(model, "Nd", state, var)) < REL_FD
    # alpha with anion complexation: 1 + b1 nu + b2 nu^2
    alpha = 1.0 + 2.0 * state.anion + 0.3 * state.anion ** 2
    plain = CationExchangeMassAction(entry.params[ts.CE_LIGAND]["20-30C"],
                                     entry.organic_ligands[0], entry.medium, None)
    assert rel(plain.evaluate("Nd", state).d / alpha, ev.d) < REL_EXACT

    nn = NearestConditionD(records_frame(), ts.SOLV_LIGAND, None, 2.5)
    nsys = SystemModel(entry=None, dmodels={ts.SOLV_LIGAND: nn}, complexant=None, activity=None,
                       temperature_C=25.0)
    state = loaded_state(nsys, lt=0.1, c_free=0.0, cT=0.0, h=1.0, nu=1.0)
    ev = nn.evaluate("Nd", state)
    assert rel(ev.dlogd_dlnL, 2.5 / LN10) < REL_EXACT
    assert rel(ev.dlogd_dlnL, central_difference(nn, "Nd", state, "L")) < REL_FD
    assert ev.dlogd_dlnh == 0.0 and ev.dlogd_dlnanion == 0.0 and ev.dlogd_dlnc == 0.0


# ---------------------------------------------------------------------------------------------
# 12.2: two-ligand composition (rel 1e-12)
# ---------------------------------------------------------------------------------------------

def two_ligand_system(second_mechanism: str = "solvating") -> tuple[SystemModel, OrgStream]:
    entry_s = ts.solvating_entry()
    ligand_a = entry_s.organic_ligands[0]
    model_a = SolvatingMassAction(entry_s.params[ts.SOLV_LIGAND]["20-30C"], ligand_a,
                                  entry_s.medium, None)
    if second_mechanism == "solvating":
        entry_b = ts.solvating_entry({"Pr": 1.0, "Nd": 1.1}, n=2.0, k_acid_uptake=None)
        ligand_b = replace(entry_b.organic_ligands[0], name="L2")
        model_b = SolvatingMassAction(entry_b.params[ts.SOLV_LIGAND]["20-30C"], ligand_b,
                                      entry_b.medium, None)
    else:
        entry_b = ts.cation_exchange_entry({"Pr": 0.5, "Nd": 0.8})
        ligand_b = replace(entry_b.organic_ligands[0], name="L2")
        model_b = CationExchangeMassAction(entry_b.params[ts.CE_LIGAND]["20-30C"], ligand_b,
                                           entry_b.medium, None)
    system = SystemModel(entry=None, dmodels={ts.SOLV_LIGAND: model_a, "L2": model_b},
                         complexant=None, activity=None, temperature_C=25.0,
                         flags=frozenset({Flag.MIXED_ORGANIC_UNMODELLED}))
    zeros = {"Pr": 0.0, "Nd": 0.0}
    org = OrgStream(flow_L_h=1.0, metals=dict(zeros),
                    metals_by_ligand={ts.SOLV_LIGAND: dict(zeros), "L2": dict(zeros)},
                    ligand_total={ts.SOLV_LIGAND: 0.1, "L2": 0.05},
                    ligand_free={ts.SOLV_LIGAND: 0.1, "L2": 0.05},
                    acid_in_org={ts.SOLV_LIGAND: 0.0, "L2": 0.0}, alkali_reserve=0.0)
    return system, org


@pytest.mark.parametrize("second", ["solvating", "cation_exchange"])
def test_two_ligand_composition(second: str) -> None:
    system, org = two_ligand_system(second)
    feed = ts.feed_prnd_nitrate(acid_M=1.0)
    feed = replace(feed, metals={"Pr": 0.01, "Nd": 0.02})
    aq, og, diag = solve_stage(feed, org, system)
    assert diag.status == "converged"
    assert Flag.MIXED_ORGANIC_UNMODELLED in diag.flags
    r = org.flow_L_h / feed.flow_L_h
    for m in ("Pr", "Nd"):
        d1 = diag.d_by_ligand[ts.SOLV_LIGAND][m]
        d2 = diag.d_by_ligand["L2"][m]
        assert rel(diag.d[m], d1 + d2) < REL_EXACT
        # y^(k) = D^(k) x and y = sum_k y^(k)
        assert rel(og.metals_by_ligand[ts.SOLV_LIGAND][m], d1 * aq.metals[m]) < REL_EXACT
        assert rel(og.metals_by_ligand["L2"][m], d2 * aq.metals[m]) < REL_EXACT
        assert rel(og.metals[m], og.metals_by_ligand[ts.SOLV_LIGAND][m]
                   + og.metals_by_ligand["L2"][m]) < REL_EXACT
        assert rel(aq.metals[m] + r * og.metals[m], feed.metals[m]) < REL_EXACT
    # ligand 1 is depleted by y^(1) only: L_f1 (1 + KH h nu) + q1 . y^(1) = L_T1
    ma = system.dmodels[ts.SOLV_LIGAND]
    bound_1 = sum(ma.q[m] * og.metals_by_ligand[ts.SOLV_LIGAND][m] for m in ("Pr", "Nd"))
    lf1 = og.ligand_free[ts.SOLV_LIGAND]
    assert rel(lf1 * (1.0 + ma.k_acid_uptake * aq.h * aq.anion) + bound_1, 0.1) < REL_EXACT
    assert rel(og.acid_in_org[ts.SOLV_LIGAND], ma.k_acid_uptake * aq.h * aq.anion * lf1) < REL_EXACT
    mb = system.dmodels["L2"]
    bound_2 = sum(mb.q[m] * og.metals_by_ligand["L2"][m] for m in ("Pr", "Nd"))
    assert rel(og.ligand_free["L2"] + bound_2, 0.05) < REL_EXACT
    assert og.acid_in_org["L2"] == 0.0
    # the same composition through evaluate_composite at the solved state
    from gen18proc.equilibrium import stage_state

    comp = evaluate_composite(system, stage_state(aq, og, system))
    for m in ("Pr", "Nd"):
        assert rel(comp[m].d, diag.d[m]) < 1e-10
    if second == "cation_exchange":
        # only the cation-exchange ligand releases protons: h_out - h_in = 3 r sum y^(2)
        released = 3.0 * sum(og.metals_by_ligand["L2"].values())
        assert rel(aq.h - feed.h + r * sum(og.acid_in_org.values()), r * released) < 1e-10


# ---------------------------------------------------------------------------------------------
# 12.2: complexant lowers D by exactly alpha_i; protonation lowers alpha with h (monotone)
# ---------------------------------------------------------------------------------------------

def test_complexant_lowers_d_by_alpha() -> None:
    spec = ts.complexant_spec()
    system = ts.two_metal_cation_exchange(complexant=spec)
    model = system.dmodels[ts.CE_LIGAND]
    assert isinstance(model, AqueousComplexantWrapper)
    for c in (0.0, 1e-4, 0.01, 0.05):
        state = loaded_state(system, lt=ts.CE_LT_DIMER, c_free=c)
        for metal, log_beta in (("Pr", 3.0), ("Nd", 2.0)):
            alpha = 1.0 + 10.0 ** log_beta * c
            d0 = model.inner.evaluate(metal, state).d
            ev = model.evaluate(metal, state)
            assert rel(ev.d, d0 / alpha) < REL_EXACT
            assert rel(ev.log_d, math.log10(d0) - math.log10(alpha)) < 1e-11
            phi = 10.0 ** log_beta * c / alpha
            if c == 0.0:
                assert ev.dlogd_dlnc == 0.0
            else:
                assert rel(ev.dlogd_dlnc, -phi / LN10) < REL_EXACT
    # the metal-specific alpha amplifies the organic selectivity Nd/Pr
    state = loaded_state(system, lt=ts.CE_LT_DIMER, c_free=0.01)
    sf0 = model.inner.evaluate("Nd", state).d / model.inner.evaluate("Pr", state).d
    sf = model.evaluate("Nd", state).d / model.evaluate("Pr", state).d
    assert sf > sf0

    # protonation: alpha_H(h) = 1 + 100 h; at fixed total the free complexant and therefore
    # alpha_i fall monotonically as h rises (tracer state and solved stage)
    cm = system.complexant_model
    assert isinstance(cm, ComplexantModel)
    alphas = []
    for h in (0.001, 0.01, 0.1, 1.0):
        assert rel(cm.alpha_h(h), 1.0 + 100.0 * h) < REL_EXACT
        st = tracer_state(system, {ts.CE_LIGAND: ts.CE_LT_DIMER}, h, 0.31, 0.05)
        assert rel(st.c_free, 0.05 / (1.0 + 100.0 * h)) < REL_EXACT
        alphas.append(cm.terms(st.c_free, ("Pr",))[0][0])
    assert all(a > b for a, b in zip(alphas, alphas[1:]))
    solved = []
    org = ts.lean_organic(system)
    for h in (0.001, 0.01, 0.1):
        feed = replace(ts.feed_prnd(complexant_total=0.05), h=h)
        aq, og, diag = solve_stage(feed, org, system)
        c = free_complexant(system, aq.metals, aq.h, aq.complexant_total)
        solved.append(cm.terms(c, ("Pr",))[0][0])
    assert all(a > b for a, b in zip(solved, solved[1:]))


def test_complexant_model_metals_and_missing_beta() -> None:
    spec = ts.complexant_spec(log_beta={"Pr": (3.0, 4.5), "Nd": (2.0,)})
    cm = ComplexantModel(spec)
    assert set(cm.metals) == {"Pr", "Nd"}
    alpha, phi, dphi = cm.terms(0.01, ("Pr", "Nd"))
    b1, b2 = 10 ** 3.0, 10 ** 4.5
    a_pr = 1 + b1 * 0.01 + b2 * 0.01 ** 2
    assert rel(alpha[0], a_pr) < REL_EXACT
    assert rel(phi[0], (b1 * 0.01 + 2 * b2 * 0.01 ** 2) / a_pr) < REL_EXACT
    assert rel(alpha[1], 1 + 100 * 0.01) < REL_EXACT
    with pytest.raises(ValueError):
        cm.beta_matrix(("Sm",))
    # a placeholder log_beta with value None excludes the metal
    unknown_nd = (ts.assumed(None, "1", (1.0, 4.0)),)
    spec2 = replace(spec, log_beta={"Pr": spec.log_beta["Pr"], "Nd": unknown_nd})
    assert ComplexantModel(spec2).metals == ("Pr",)


# ---------------------------------------------------------------------------------------------
# ConstantD, loading fraction, chemistry flags (5.4)
# ---------------------------------------------------------------------------------------------

def test_constant_d_is_a_labelled_sanity_limit() -> None:
    model = ConstantD({"Pr": -math.inf, "Nd": 60.0, "Sm": 0.3})
    assert model.provenance.status == ProvStatus.ASSUMED
    assert model.provenance.assumed_label == "ASSUMED_PLACEHOLDER"
    assert model.q == {"Pr": 0.0, "Nd": 0.0, "Sm": 0.0}
    assert model.p == model.q and model.z == model.q
    system = ts.constant_d_system({"Pr": -math.inf, "Nd": 60.0, "Sm": 0.3})
    state = tracer_state(system, {ts.CONSTANT_LIGAND: 1.0}, 0.1, 0.1)
    assert model.evaluate("Pr", state).d == 0.0
    assert model.evaluate("Nd", state).d == 1e60
    ev = model.evaluate("Sm", state)
    assert rel(ev.d, 10 ** 0.3) < REL_EXACT
    assert ev.dlogd_dlnL == ev.dlogd_dlnh == ev.dlogd_dlnanion == ev.dlogd_dlnc == 0.0
    per_metal = ConstantD({"Pr": 0.0, "Nd": 0.0}, q={"Pr": 3.0}, p=3.0)
    assert per_metal.q == {"Pr": 3.0, "Nd": 0.0} and per_metal.p == {"Pr": 3.0, "Nd": 3.0}
    with pytest.raises(ValueError):
        model.evaluate("Eu", state)


def test_loading_fraction_and_chemistry_flags() -> None:
    system = ts.two_metal_solvating()
    model = system.dmodels[ts.SOLV_LIGAND]
    lt = ts.SOLV_LT
    base = tracer_state(system, {ts.SOLV_LIGAND: lt}, 1.0, 1.0)
    assert ligand_loading_fraction(base, ts.SOLV_LIGAND, model.q) == 0.0
    assert ligand_chemistry_flags(base, ts.SOLV_LIGAND, model.q, None) == frozenset()
    # lambda = (q . y + a) / L_T with q = n = 2.7
    y = {"Pr": 0.005, "Nd": 0.01}
    st = replace(base, y=dict(y), y_by_ligand={ts.SOLV_LIGAND: dict(y)},
                 acid_in_org={ts.SOLV_LIGAND: 0.02}, ligand_free={ts.SOLV_LIGAND: 0.5 * lt})
    lam = (2.7 * 0.015 + 0.02) / lt
    assert rel(ligand_loading_fraction(st, ts.SOLV_LIGAND, model.q), lam) < REL_EXACT
    flags = ligand_chemistry_flags(st, ts.SOLV_LIGAND, model.q, unknown_phase())
    assert lam > 0.5 and Flag.HIGH_LOADING in flags
    assert Flag.PHASE_BEHAVIOUR_UNKNOWN in flags          # LOC null and lambda > 0.3
    assert Flag.LOADING_CAP_HIT not in flags
    # a sourced LOC below the organic metal + HNO3.L -> THIRD_PHASE_RISK, no PHASE_BEHAVIOUR_UNKNOWN
    flags = ligand_chemistry_flags(st, ts.SOLV_LIGAND, model.q, unknown_phase(loc_metal=0.03))
    assert Flag.THIRD_PHASE_RISK in flags and Flag.PHASE_BEHAVIOUR_UNKNOWN not in flags
    flags = ligand_chemistry_flags(st, ts.SOLV_LIGAND, model.q, unknown_phase(loc_metal=0.05))
    assert Flag.THIRD_PHASE_RISK not in flags
    flags = ligand_chemistry_flags(st, ts.SOLV_LIGAND, model.q, unknown_phase(loc_metal=0.05,
                                                                              loc_acid=0.01))
    assert Flag.THIRD_PHASE_RISK in flags                  # acid LOC exceeded by HNO3.L
    # free ligand below 1e-6 L_T -> LOADING_CAP_HIT
    st2 = replace(st, ligand_free={ts.SOLV_LIGAND: 0.5e-6 * lt})
    assert Flag.LOADING_CAP_HIT in ligand_chemistry_flags(st2, ts.SOLV_LIGAND, model.q, None)
    # moderate loading: no flag at all
    st3 = replace(base, y={"Pr": 0.001, "Nd": 0.001},
                  y_by_ligand={ts.SOLV_LIGAND: {"Pr": 0.001, "Nd": 0.001}},
                  ligand_free={ts.SOLV_LIGAND: 0.9 * lt})
    assert ligand_chemistry_flags(st3, ts.SOLV_LIGAND, model.q, unknown_phase()) == frozenset()


def test_solvating_caveat_flags() -> None:
    lt = ts.SOLV_LT
    unmodelled = ts.two_metal_solvating(k_acid_uptake=None)
    model = unmodelled.dmodels[ts.SOLV_LIGAND]
    assert model.k_acid_uptake == 0.0
    st_hi = tracer_state(unmodelled, {ts.SOLV_LIGAND: lt}, 3.0, 3.0)
    st_lo = tracer_state(unmodelled, {ts.SOLV_LIGAND: lt}, 0.5, 0.5)
    assert Flag.ACID_UPTAKE_UNMODELLED in model.evaluate("Nd", st_hi).flags
    assert Flag.ACID_UPTAKE_UNMODELLED not in model.evaluate("Nd", st_lo).flags
    modelled = ts.two_metal_solvating()
    assert Flag.ACID_UPTAKE_UNMODELLED not in modelled.dmodels[ts.SOLV_LIGAND].evaluate(
        "Nd", st_hi).flags
    # ANION_ACID_CONFOUNDED only for a corpus-fitted set applied where [anion] != [H+]
    entry = ts.solvating_entry()
    params = entry.params[ts.SOLV_LIGAND]["20-30C"]
    fitted_prov = Provenance(ProvStatus.FITTED_FROM_CORPUS, Source(kind="model"), (1.0, 2.0),
                             model_id="M1_test", fit_manifest_sha256="0" * 64)
    fitted = replace(params, log_k={m: Sourced(v.value, "1", fitted_prov)
                                    for m, v in params.log_k.items()})
    fm = SolvatingMassAction(fitted, entry.organic_ligands[0], entry.medium, None)
    assert Flag.EQUILIBRIUM_ACID_ASSUMED_NOMINAL in fm.base_flags
    fs = SystemModel(entry=None, dmodels={ts.SOLV_LIGAND: fm}, complexant=None, activity=None,
                     temperature_C=25.0)
    same = tracer_state(fs, {ts.SOLV_LIGAND: lt}, 1.0, 1.0)
    salted = tracer_state(fs, {ts.SOLV_LIGAND: lt}, 1.0, 3.0)
    assert Flag.ANION_ACID_CONFOUNDED not in fm.evaluate("Nd", same).flags
    assert Flag.ANION_ACID_CONFOUNDED in fm.evaluate("Nd", salted).flags
    plain = modelled.dmodels[ts.SOLV_LIGAND]
    assert Flag.ANION_ACID_CONFOUNDED not in plain.evaluate("Nd", salted).flags


def test_cation_exchange_medium_strength_flag() -> None:
    system = ts.two_metal_cation_exchange()      # domain log acid [-3, 1]; beta_anion None
    model = system.dmodels[ts.CE_LIGAND]
    inside = tracer_state(system, {ts.CE_LIGAND: ts.CE_LT_DIMER}, 0.1, 5.0)
    assert Flag.MEDIUM_STRENGTH_UNMODELLED not in model.evaluate("Nd", inside).flags
    far = tracer_state(system, {ts.CE_LIGAND: ts.CE_LT_DIMER}, 0.1, 25.0)   # > 2 x 10^1
    assert Flag.MEDIUM_STRENGTH_UNMODELLED in model.evaluate("Nd", far).flags


# ---------------------------------------------------------------------------------------------
# NearestConditionD on a tiny corpus_records.csv-shaped frame
# ---------------------------------------------------------------------------------------------

def test_nearest_condition_d() -> None:
    df = records_frame()
    # tracer_correction=False is the model as first written: the recorded log D is used as it
    # stands.  The corrected default is checked by test_nearest_condition_d_tracer_correction.
    nn = NearestConditionD(df, ts.SOLV_LIGAND, None, 3.0, tracer_correction=False)
    assert set(nn.metals) == {"Nd", "Pr", "Ce"}
    assert nn.n_records == 9                    # r05 (fit_eligible False) dropped
    assert nn.provenance.status == ProvStatus.MEASURED_CORPUS
    assert {Flag.EQUILIBRIUM_ACID_ASSUMED_NOMINAL, Flag.OA_ASSUMED} <= nn.base_flags
    # exact hits
    assert nn.nearest("Nd", math.log10(1.0), math.log10(0.1)) == (1.0, 0.0, "r01")
    assert nn.nearest("Nd", math.log10(3.0), math.log10(0.3)) == (3.0, 0.0, "r04")
    # nearest, not exact: (log 2.5, log 0.1) is closer to r02 (3 M) than to r01 (1 M)
    ld, dist, rid = nn.nearest("Nd", math.log10(2.5), math.log10(0.1))
    assert rid == "r02" and rel(dist, math.log10(3.0 / 2.5)) < 1e-12
    # tie: equal distance from acid 1 and 4 M at log 2; equal |delta lA| -> smaller record_id
    ld, dist, rid = nn.nearest("Ce", math.log10(2.0), math.log10(0.1))
    assert rid == "r09" and ld == 0.7 and rel(dist, math.log10(2.0)) < 1e-12
    # the ineligible r05 (log D 9 at the r01 point) never wins
    assert nn.nearest("Nd", 0.0, -1.0)[0] == 1.0

    system = SystemModel(entry=None, dmodels={ts.SOLV_LIGAND: nn}, complexant=None,
                         activity=None, temperature_C=25.0)
    zero = tracer_state(system, {ts.SOLV_LIGAND: 0.1}, 1.0, 1.0)
    ev = nn.evaluate("Nd", zero)
    assert rel(ev.log_d, 1.0) < REL_EXACT and ev.ood_distance["nn"] == 0.0
    # depletion term: log D = log D_nn + n_prior log10(L_f / L_T)
    half = replace(zero, ligand_free={ts.SOLV_LIGAND: 0.05})
    ev2 = nn.evaluate("Nd", half)
    assert rel(ev2.log_d, 1.0 + 3.0 * math.log10(0.5)) < 1e-12
    # the 1-NN uses the recorded formal concentration L_T, not the depleted L_f
    assert nn.evaluate("Nd", half).ood_distance["nn"] == 0.0
    off = tracer_state(system, {ts.SOLV_LIGAND: 0.1}, 2.5, 2.5)
    assert rel(nn.evaluate("Nd", off).ood_distance["nn"], math.log10(3.0 / 2.5)) < 1e-12
    with pytest.raises(ValueError):
        nn.evaluate("Sm", zero)
    # it drives a stage solve (single metal, the loading-check use of 10.4)
    feed = ts.feed_prnd_nitrate(acid_M=1.0)
    feed = replace(feed, metals={"Nd": 0.005})
    org = OrgStream(flow_L_h=1.0, metals={"Nd": 0.0},
                    metals_by_ligand={ts.SOLV_LIGAND: {"Nd": 0.0}},
                    ligand_total={ts.SOLV_LIGAND: 0.1}, ligand_free={ts.SOLV_LIGAND: 0.1},
                    acid_in_org={ts.SOLV_LIGAND: 0.0}, alkali_reserve=0.0)
    aq, og, diag = solve_stage(feed, org, system)
    assert diag.status == "converged"
    assert diag.d["Nd"] < 10.0                    # loading lowers it below the tracer value
    lf = og.ligand_free[ts.SOLV_LIGAND]
    assert rel(diag.d["Nd"], 10.0 * (lf / 0.1) ** 3.0) < 1e-10
    # ligand_M as a flattened per-ligand column and as a JSON mapping are accepted too
    flat = df.drop(columns=["ligand_M"]).assign(**{f"ligand_M.{ts.SOLV_LIGAND}": df["ligand_M"]})
    assert NearestConditionD(flat, ts.SOLV_LIGAND, None, 3.0).n_records == 9
    js = df.assign(ligand_M=[f'{{"{ts.SOLV_LIGAND}": {v}}}' for v in df["ligand_M"]])
    assert NearestConditionD(js, ts.SOLV_LIGAND, None, 3.0).n_records == 9
    with pytest.raises(ValueError):
        NearestConditionD(df, "OTHER", None, 3.0)     # ligand_name mismatch -> no record


# ---------------------------------------------------------------------------------------------
# EffectiveCapacity (5.6, exploratory)
# ---------------------------------------------------------------------------------------------

def test_effective_capacity() -> None:
    with pytest.raises(ValueError):
        EffectiveCapacity(0.0)
    with pytest.raises(ValueError):
        EffectiveCapacity(1.5)
    feed = ts.feed_prnd_nitrate()
    ref_sys = ts.two_metal_solvating()
    ref = solve_stage(feed, ts.lean_organic(ref_sys), ref_sys)
    one = ts.two_metal_solvating(activity=EffectiveCapacity(1.0))
    same = solve_stage(feed, ts.lean_organic(one), one)
    for m in ("Pr", "Nd"):
        assert rel(same[0].metals[m], ref[0].metals[m]) < 1e-11
    phi = 0.5
    sysm = ts.two_metal_solvating(activity=EffectiveCapacity(phi))
    org = ts.lean_organic(sysm)
    aq, og, diag = solve_stage(feed, org, sysm)
    assert diag.status == "converged"
    model = sysm.dmodels[ts.SOLV_LIGAND]
    bound = sum(model.q[m] * og.metals[m] for m in ("Pr", "Nd"))
    a = og.acid_in_org[ts.SOLV_LIGAND]
    # free ligand = phi L_T - sum q y - a  (in place of L_T - sum q y - a)
    assert rel(og.ligand_free[ts.SOLV_LIGAND], phi * ts.SOLV_LT - bound - a) < 1e-11
    assert og.ligand_total[ts.SOLV_LIGAND] == ts.SOLV_LT
    # less capacity -> less extracted
    assert og.metals["Nd"] < ref[1].metals["Nd"]
    st = tracer_state(sysm, {ts.SOLV_LIGAND: ts.SOLV_LT}, 3.0, 3.0)
    assert st.ligand_free[ts.SOLV_LIGAND] == phi * ts.SOLV_LT
    assert EffectiveCapacity(phi).aqueous_gamma(st) == {"Pr": 1.0, "Nd": 1.0}


# ---------------------------------------------------------------------------------------------
# build_system_model (5.7)
# ---------------------------------------------------------------------------------------------

def test_build_system_model_refusals() -> None:
    entry = ts.cation_exchange_entry()
    bad = build_system_model(entry, feed_anion="nitrate", temperature_C=25.0,
                             params_source="literature")
    assert isinstance(bad, ModelBuildError) and "cross-anion" in bad.reason
    missing = build_system_model(entry, feed_anion="chloride", temperature_C=25.0,
                                 params_source="literature", feed_metals=("Pr", "Nd", "Sm"))
    assert isinstance(missing, ModelBuildError) and "Sm" in missing.reason
    spec = ts.complexant_spec(log_beta={"Pr": (3.0,)})       # no Nd
    nobeta = build_system_model(entry, feed_anion="chloride", temperature_C=25.0,
                                params_source="literature", complexant=spec)
    assert isinstance(nobeta, ModelBuildError) and "Nd" in nobeta.reason
    nofit = build_system_model(entry, feed_anion="chloride", temperature_C=25.0,
                               params_source="fitted")
    assert isinstance(nofit, ModelBuildError)
    nolit = build_system_model(replace(entry, params={}), feed_anion="chloride",
                               temperature_C=25.0, params_source="literature")
    assert isinstance(nolit, ModelBuildError)
    norec = build_system_model(entry, feed_anion="chloride", temperature_C=25.0,
                               params_source="nearest")
    assert isinstance(norec, ModelBuildError) and "records" in norec.reason
    with pytest.raises(ValueError):
        build_system_model(entry, feed_anion="chloride", temperature_C=25.0,
                           params_source="guess")
    # a log_k with value None (unknown) is a missing log_k for that metal
    params = entry.params[ts.CE_LIGAND]["20-30C"]
    half = replace(params, log_k={"Pr": params.log_k["Pr"],
                                  "Nd": Sourced.unknown("1")})
    nd_missing = build_system_model(replace(entry, params={ts.CE_LIGAND: {"20-30C": half}}),
                                    feed_anion="chloride", temperature_C=25.0,
                                    params_source="literature", feed_metals=("Pr", "Nd"))
    assert isinstance(nd_missing, ModelBuildError) and "Nd" in nd_missing.reason
    good = build_system_model(entry, feed_anion="chloride", temperature_C=25.0,
                              params_source="literature")
    assert isinstance(good, SystemModel)
    assert good.params_source == "literature" and good.bands == {ts.CE_LIGAND: "20-30C"}
    assert good.assumptions and "concentrations for activities" in good.assumptions
    assert good.metals == ("Pr", "Nd") and good.ligands == (ts.CE_LIGAND,)
    # "auto" on a literature entry picks the literature block
    auto = build_system_model(entry, feed_anion="chloride", temperature_C=25.0)
    assert isinstance(auto, SystemModel) and auto.params_source == "literature"


def test_build_system_model_parameter_draw() -> None:
    entry = ts.cation_exchange_entry()
    base = ts.two_metal_cation_exchange()
    state = tracer_state(base, {ts.CE_LIGAND: ts.CE_LT_DIMER}, 0.01, 0.31)
    drawn = build_system_model(entry, feed_anion="chloride", temperature_C=25.0,
                               params_source="literature",
                               parameter_draw={f"{ts.CE_LIGAND}.log_k.Nd": -1.5,
                                               f"{ts.CE_LIGAND}.b_proton": 2.5})
    assert isinstance(drawn, SystemModel)
    d_base = base.dmodels[ts.CE_LIGAND].evaluate("Nd", state)
    d_draw = drawn.dmodels[ts.CE_LIGAND].evaluate("Nd", state)
    expected = d_base.log_d + (-1.5 - (-1.95)) - (2.5 - 3.0) * math.log10(0.01)
    assert rel(d_draw.log_d, expected) < 1e-12
    assert rel(d_draw.dlogd_dlnh, -2.5 / LN10) < 1e-12
    # Pr: its log K is untouched, only the shared b_proton draw moves it (by -0.5 * log h)
    pr_draw = drawn.dmodels[ts.CE_LIGAND].evaluate("Pr", state).log_d
    pr_base = base.dmodels[ts.CE_LIGAND].evaluate("Pr", state).log_d
    assert rel(pr_draw, pr_base - (2.5 - 3.0) * math.log10(0.01)) < 1e-12
    only_k = build_system_model(entry, feed_anion="chloride", temperature_C=25.0,
                                params_source="literature",
                                parameter_draw={f"{ts.CE_LIGAND}.log_k.Nd": -1.5})
    assert rel(only_k.dmodels[ts.CE_LIGAND].evaluate("Pr", state).log_d, pr_base) < 1e-12
    assert only_k.dmodels[ts.CE_LIGAND].params.log_k["Nd"].value == -1.5
    assert only_k.dmodels[ts.CE_LIGAND].params.log_k["Nd"].provenance.range == (-2.95, -0.95)
    for bad in ({f"{ts.CE_LIGAND}.log_k.Nd": 5.0},             # outside the declared range
                {f"{ts.CE_LIGAND}.log_k.Sm": -1.0},            # no such metal
                {f"{ts.CE_LIGAND}.n_solvation": 2.0},          # not a cation-exchange parameter
                {"OTHER.log_k.Nd": -1.0}):                     # no such ligand
        with pytest.raises(ValueError):
            build_system_model(entry, feed_anion="chloride", temperature_C=25.0,
                               params_source="literature", parameter_draw=bad)
    # complexant draws
    spec = ts.complexant_spec()
    sysc = build_system_model(entry, feed_anion="chloride", temperature_C=25.0,
                              params_source="literature", complexant=spec,
                              parameter_draw={"synthetic_complexant.log_beta.Pr": 3.5,
                                              "synthetic_complexant.protonation_logk.1": 1.5,
                                              "synthetic_complexant.regeneration_fraction": 0.7})
    assert isinstance(sysc, SystemModel)
    assert sysc.complexant.log_beta["Pr"][0].value == 3.5
    assert sysc.complexant_model.regeneration_fraction == 0.7
    assert rel(sysc.complexant_model.alpha_h(0.1), 1.0 + 10 ** 1.5 * 0.1) < 1e-12
    with pytest.raises(ValueError):
        build_system_model(entry, feed_anion="chloride", temperature_C=25.0,
                           params_source="literature", complexant=spec,
                           parameter_draw={"synthetic_complexant.log_beta.Pr": 9.0})


def test_build_system_model_temperature_band() -> None:
    entry = ts.solvating_entry()
    inside = build_system_model(entry, feed_anion="nitrate", temperature_C=25.0,
                                params_source="literature")
    outside = build_system_model(entry, feed_anion="nitrate", temperature_C=45.0,
                                 params_source="literature")
    assert isinstance(inside, SystemModel) and isinstance(outside, SystemModel)
    st = tracer_state(inside, {ts.SOLV_LIGAND: ts.SOLV_LT}, 1.0, 1.0)
    assert Flag.OOD_TEMPERATURE not in inside.dmodels[ts.SOLV_LIGAND].evaluate("Nd", st).flags
    st45 = tracer_state(outside, {ts.SOLV_LIGAND: ts.SOLV_LT}, 1.0, 1.0)
    assert Flag.OOD_TEMPERATURE in outside.dmodels[ts.SOLV_LIGAND].evaluate("Nd", st45).flags
    assert outside.bands == {ts.SOLV_LIGAND: "20-30C"} and outside.temperature_C == 45.0
    assert select_band(["<20C", "20-30C", "30-40C"], 35.0) == ("30-40C", True)
    assert select_band(["<20C", "20-30C"], 35.0) == ("20-30C", False)
    assert select_band(["20-30C", ">=50C"], 60.0) == (">=50C", True)
    assert select_band([], 25.0) == (None, False)
    assert parse_band("<20C") == (-math.inf, 20.0) and parse_band("20-30C") == (20.0, 30.0)
    assert parse_band(">=50C") == (50.0, math.inf)


def test_build_system_model_mixed_organic_and_json_block() -> None:
    entry = ts.solvating_entry()
    lig = entry.organic_ligands[0]
    second = replace(lig, name="L2", smiles="CCCC", canonical_smiles="CCCC")
    params = entry.params[ts.SOLV_LIGAND]["20-30C"]
    # the second block is given in the JSON shape of section 3.8 (model_type + Sourced objects)
    block2 = {"model_type": "solvating", "medium_anion": "nitrate", "temperature_band": "20-30C",
              "log_k": {m: v.to_json() for m, v in params.log_k.items()},
              "n_solvation": params.n_solvation.to_json(), "p_anion": params.p_anion.to_json(),
              "p_h": params.p_h.to_json(), "k_acid_uptake": {"value": None, "unit": "L2/mol2",
                                                              "status": "unknown"},
              "delta_h_kj_mol": None}
    two = replace(entry, organic_ligands=(lig, second),
                  params={ts.SOLV_LIGAND: {"20-30C": params}, "L2": {"20-30C": block2}})
    system = build_system_model(two, feed_anion="nitrate", temperature_C=25.0,
                                params_source="literature")
    assert isinstance(system, SystemModel)
    assert Flag.MIXED_ORGANIC_UNMODELLED in system.flags
    assert system.ligands == (ts.SOLV_LIGAND, "L2")
    assert system.dmodels["L2"].k_acid_uptake == 0.0
    st = tracer_state(system, {ts.SOLV_LIGAND: 0.1, "L2": 0.1}, 3.0, 3.0)
    comp = evaluate_composite(system, st)
    assert Flag.MIXED_ORGANIC_UNMODELLED in comp["Nd"].flags
    assert rel(comp["Nd"].d, 2.0 * system.dmodels["L2"].evaluate("Nd", st).d) < 1e-12
    # a cation-exchange ligand whose block is solvating is refused
    ce = replace(lig, name="L2", mechanism=Mechanism.CATION_EXCHANGE, aggregation="dimer")
    wrong = build_system_model(replace(two, organic_ligands=(lig, ce)), feed_anion="nitrate",
                               temperature_C=25.0, params_source="literature")
    assert isinstance(wrong, ModelBuildError)


def make_record(rid: str, metal: str, log_d: float, acid: float, ligand: float, mm: float,
                pub: str, eligible: bool = True) -> DistributionRecord:
    prov = Provenance(ProvStatus.MEASURED_CORPUS,
                      Source(kind="corpus", locator="synthetic", publication_id=pub,
                             safe_exp_ids=(rid,)))
    return DistributionRecord(
        record_id=rid, metal=metal, d=10 ** log_d, log_d=log_d, acid_nominal_M=acid,
        acid_eq_M=None, anion_M=acid, ligand_M={ts.SOLV_LIGAND: ligand}, complexant_M=None,
        metals_initial_mM={metal: mm}, oa_ratio=None, temperature_C=25.0, contact_time_min=None,
        diluent_name="n_dodecane", publication_id=pub, experiment_series_id=None,
        replicate_id=None, loading_series_id=None, is_tracer=False, fit_eligible=eligible,
        fit_ineligible_reason=None if eligible else "test", duplicate_flag=None,
        provenance=prov)


def test_nearest_condition_d_tracer_correction() -> None:
    """The 1-NN lifts a record to its own tracer limit before the stage depletion is applied.

    Integration 2026-09-13 (`addenda/INTEGRATION.md`): without this the deployed D source applies
    the depletion term to a measurement that already contains its own loading -- a double count
    on the 28.7 % of corpus records with a metal concentration that were measured above loading
    fraction 0.1.
    """
    df = records_frame()
    raw = NearestConditionD(df, ts.SOLV_LIGAND, None, 3.0, tracer_correction=False)
    fixed = NearestConditionD(df, ts.SOLV_LIGAND, None, 3.0)
    assert fixed.tracer_correction and not raw.tracer_correction
    assert fixed.n_tracer_corrected > 0 and raw.n_tracer_corrected == 0

    # r01: Nd, log D 1.0, 0.5 mM, 0.1 M ligand -> its own y depletes the ligand it was measured on
    y_rec = 0.5e-3 * 10.0 ** 1.0 / (1.0 + 10.0 ** 1.0)
    expected = 1.0 + 3.0 * math.log10(0.1 / (0.1 - 3.0 * y_rec))
    assert rel(fixed.nearest("Nd", math.log10(1.0), math.log10(0.1))[0], expected) < 1e-12
    assert raw.nearest("Nd", math.log10(1.0), math.log10(0.1))[0] == 1.0
    assert expected > 1.0                       # the correction always raises the tracer value

    # a record with no metal concentration cannot be corrected and is left as it stands
    assert fixed.nearest("Ce", math.log10(4.0), math.log10(0.1))[0] == 0.7

    # round trip: predicting at the record's own loading returns the recorded log D
    system = SystemModel(entry=None, dmodels={ts.SOLV_LIGAND: fixed}, complexant=None,
                         activity=None, temperature_C=25.0)
    st = tracer_state(system, {ts.SOLV_LIGAND: 0.1}, 1.0, 1.0)
    at_record_loading = replace(st, ligand_free={ts.SOLV_LIGAND: 0.1 - 3.0 * y_rec})
    assert rel(fixed.evaluate("Nd", at_record_loading).log_d, 1.0) < 1e-12


def test_build_system_model_nearest_from_records() -> None:
    entry = ts.solvating_entry()
    records = (make_record("Ca_1", "Nd", 1.0, 1.0, 0.1, 0.5, "pubA"),
               make_record("Ca_2", "Nd", 2.0, 3.0, 0.1, 0.5, "pubA"),
               make_record("Ca_3", "Nd", 0.5, 1.0, 0.03, 0.5, "pubB"),
               make_record("Ca_4", "Pr", 0.8, 1.0, 0.1, 0.5, "pubB"),
               make_record("Ca_5", "Pr", 1.6, 3.0, 0.1, 0.5, "pubB"),
               make_record("Ca_6", "Pr", 0.2, 1.0, 0.03, 0.5, "pubA"),
               make_record("Ca_7", "Pr", 7.0, 1.0, 0.1, 0.5, "pubA", eligible=False))
    corpus = replace(entry, origin="corpus", params={}, applicability={}, records=records)
    # auto on a corpus entry without a fitted block -> nearest, with a domain from the records
    system = build_system_model(corpus, feed_anion="nitrate", temperature_C=25.0)
    assert isinstance(system, SystemModel)
    assert system.params_source == "nearest"
    model = system.dmodels[ts.SOLV_LIGAND]
    assert isinstance(model, NearestConditionD)
    assert model.mechanism == Mechanism.SOLVATING and model.q == {"Nd": 3.0, "Pr": 3.0}
    assert model.z == {"Nd": 3.0, "Pr": 3.0} and model.p == {"Nd": 0.0, "Pr": 0.0}
    dom = model.domain
    assert isinstance(dom, ApplicabilityDomain)
    assert dom.log_acid == (0.0, math.log10(3.0)) and dom.n_publications == 2
    assert dom.log_ligand[ts.SOLV_LIGAND] == (math.log10(0.03), math.log10(0.1))
    assert dom.n_records == 6 and dom.anion == "nitrate"
    st = tracer_state(system, {ts.SOLV_LIGAND: 0.1}, 3.0, 3.0)
    ev = model.evaluate("Pr", st)
    # Ca_5 (log D 1.6) was itself measured at 0.5 mM Pr, so it carries its own depletion; the
    # model lifts it to its tracer limit before applying the stage's depletion (integration
    # 2026-09-13, addenda/INTEGRATION.md): log D_tracer = log D + n log10(L_T / (L_T - n y)).
    y_rec = 5e-4 * 10 ** 1.6 / (1.0 + 10 ** 1.6)
    expected = 1.6 + 3.0 * math.log10(0.1 / (0.1 - 3.0 * y_rec))
    assert rel(ev.log_d, expected) < 1e-12 and ev.ood_distance["nn"] == 0.0
    assert not (ev.flags & {Flag.OOD_ACID, Flag.OOD_LIGAND, Flag.OOD_HULL})
    assert {Flag.EQUILIBRIUM_ACID_ASSUMED_NOMINAL, Flag.OA_ASSUMED} <= ev.flags
    far = tracer_state(system, {ts.SOLV_LIGAND: 0.1}, 10.0, 10.0)
    assert Flag.OOD_ACID in model.evaluate("Pr", far).flags
    # a fitted block is used under "auto" only when the pre-registered decision adopted it
    params = entry.params[ts.SOLV_LIGAND]["20-30C"]
    fitted_prov = Provenance(ProvStatus.FITTED_FROM_CORPUS, Source(kind="model"), (1.0, 2.0),
                             model_id="M1_test", fit_manifest_sha256="0" * 64)
    fitted = replace(params, log_k={m: Sourced(v.value, "1", fitted_prov)
                                    for m, v in params.log_k.items()})
    with_fit = replace(corpus, params={ts.SOLV_LIGAND: {"20-30C": fitted}})
    adopted = build_system_model(with_fit, feed_anion="nitrate", temperature_C=25.0,
                                 decision_adopted=True)
    rejected = build_system_model(with_fit, feed_anion="nitrate", temperature_C=25.0,
                                  decision_adopted=False)
    assert isinstance(adopted, SystemModel) and adopted.params_source == "fitted"
    assert isinstance(adopted.dmodels[ts.SOLV_LIGAND], SolvatingMassAction)
    assert isinstance(rejected, SystemModel) and rejected.params_source == "nearest"
    forced = build_system_model(with_fit, feed_anion="nitrate", temperature_C=25.0,
                                params_source="fitted")
    assert isinstance(forced, SystemModel) and forced.params_source == "fitted"


def test_dimer_basis_and_ligand_spec_stoichiometry() -> None:
    """The cation-exchange model works on the dimer basis: q = 3 dimers, p = 3, z = 0; the
    formal monomer concentration of the LigandSpec is twice the code's ligand total."""
    system = ts.two_metal_cation_exchange()
    model = system.dmodels[ts.CE_LIGAND]
    assert model.q == {"Pr": 3.0, "Nd": 3.0} and model.p == {"Pr": 3.0, "Nd": 3.0}
    assert model.z == {"Pr": 0.0, "Nd": 0.0} and model.ligand_scale == 2.0
    assert ts.CE_LT_DIMER == ts.CE_HA_TOTAL / 2.0
    lt = ts.CE_LT_DIMER
    st = tracer_state(system, {ts.CE_LIGAND: lt}, 0.01, 0.31)
    ev = model.evaluate("Nd", st)
    assert rel(ev.log_d, -1.95 + 3 * math.log10(lt) - 3 * math.log10(0.01)) < 1e-12
    solv = ts.two_metal_solvating().dmodels[ts.SOLV_LIGAND]
    assert solv.q == {"Pr": 2.7, "Nd": 2.7} and solv.p == {"Pr": 0.0, "Nd": 0.0}
    assert solv.z == {"Pr": 3.0, "Nd": 3.0} and solv.ligand_scale == 1.0
    # a LigandSpec stoichiometry overrides the ideal values
    entry = ts.solvating_entry()
    lig = entry.organic_ligands[0]
    lig2 = replace(lig, stoichiometry=Stoichiometry(ts.assumed(2.0), ts.assumed(0.0),
                                                     ts.assumed(2.0)))
    m2 = SolvatingMassAction(entry.params[ts.SOLV_LIGAND]["20-30C"], lig2, entry.medium, None)
    assert m2.q == {"Pr": 2.0, "Nd": 2.0} and m2.z == {"Pr": 2.0, "Nd": 2.0}
    assert isinstance(lig2, LigandSpec)
