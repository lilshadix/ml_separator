"""Tests of ``gen18proc.metrics`` (DESIGN.md section 8; test row of 12.2).

Rows covered: purity / recovery / enrichment on a hand-built stream table (rel 1e-12);
consumption per mol of target and per kg of oxide; NaN + ``COST_INCOMPLETE`` when a price is
missing; ``LIGAND_LOSS_NOT_MEASURED`` when the ligand loss is null and ``f_bleed = 0``; NaN
metrics for a failed result.  Extras: the price table loads and every entry is a labelled
placeholder with a range; metrics of a solved cascade agree with its origin pass; ``on_spec``.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest

G18 = Path(__file__).resolve().parents[1]
if str(G18) not in sys.path:
    sys.path.insert(0, str(G18))

from gen18proc import testsystems as ts  # noqa: E402
from gen18proc.cascade import solve_cascade  # noqa: E402
from gen18proc.metrics import (  # noqa: E402
    ATOMIC_MASS_G_MOL,
    CONSUMPTION_ITEMS,
    Prices,
    compute_metrics,
    oxide_mass_per_mol_metal,
)
from gen18proc.types import (  # noqa: E402
    AqStream,
    CascadeResult,
    CascadeSpec,
    Flag,
    OrgStream,
    StageDiagnostics,
)

REL = 1e-12
PRICES_PATH = G18 / "config" / "prices.json"


def rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


# ---------------------------------------------------------------------------------------------
# a hand-built two-stage table (numbers chosen for exact arithmetic, not a physical solution)
# ---------------------------------------------------------------------------------------------

CE = ts.two_metal_cation_exchange()


def hand_spec(*, sap: float = 0.0, f_bleed: float = 0.0, n_str: int = 1) -> CascadeSpec:
    feed = AqStream(1.0, {"Pr": 0.025, "Nd": 0.075}, 0.01, 0.31, 0.0, 0.0)
    scrub = AqStream(0.5, {"Pr": 0.0, "Nd": 0.0}, 0.2, 0.3, 0.0, 0.0)      # 0.1 M added salt
    strip = AqStream(0.25, {"Pr": 0.0, "Nd": 0.0}, 2.0, 2.0, 0.0, 0.0)
    return CascadeSpec(1, 1, n_str, feed, scrub, strip, 2.0, {ts.CE_LIGAND: ts.CE_LT_DIMER},
                       sap, f_bleed=f_bleed, target="Nd")


def hand_result(spec: CascadeSpec) -> CascadeResult:
    product = AqStream(0.25, {"Pr": 0.002, "Nd": 0.2}, 2.0, 2.0, 0.0, 0.0)   # 0.0005 / 0.05 mol/h
    raff = AqStream(1.5, {"Pr": 0.016, "Nd": 0.016}, 0.1, 0.3, 0.0, 0.0)
    scrub_r = AqStream(0.5, {"Pr": 0.001, "Nd": 0.01}, 0.1, 0.3, 0.0, 0.0)
    org = OrgStream(2.0, {"Pr": 0.001, "Nd": 0.03}, {ts.CE_LIGAND: {"Pr": 0.001, "Nd": 0.03}},
                    {ts.CE_LIGAND: 0.4}, {ts.CE_LIGAND: 0.307}, {ts.CE_LIGAND: 0.0}, 0.0)
    stripped = replace(org, metals={"Pr": 0.0, "Nd": 0.0},
                       metals_by_ligand={ts.CE_LIGAND: {"Pr": 0.0, "Nd": 0.0}},
                       ligand_free={ts.CE_LIGAND: 0.4})
    diag = StageDiagnostics(d={"Pr": 0.3, "Nd": 0.6}, d_by_ligand={ts.CE_LIGAND: {"Pr": 0.3,
                                                                                    "Nd": 0.6}},
                            loading_fraction={ts.CE_LIGAND: 0.2325}, iterations=1,
                            residual_max=0.0, branch=1, flags=frozenset(), ood_distance={},
                            status="converged")
    diag2 = replace(diag, d={"Pr": 0.01, "Nd": 0.02}, loading_fraction={ts.CE_LIGAND: 0.0})
    origin = {"recovery_from_feed": 0.6, "recovery_total": 0.05 / 0.075,
              "scrub_target_return": math.nan, "net_product_mol_h": 0.05,
              "product_total_mol_h": 0.05, "feed_target_mol_h": 0.075,
              "scrub_target_mol_h": 0.0, "strip_target_mol_h": 0.0, "fresh_target_mol_h": 0.0,
              "label_sum_rel_defect": 0.0}
    return CascadeResult(
        status="converged_newton", stages_aq=(raff, scrub_r, product),
        stages_org=(org, org, stripped), diagnostics=(diag, diag, diag2), raffinate=raff,
        product=product, scrub_raffinate=scrub_r, loaded_organic=org, stripped_organic=stripped,
        origin=origin, residual_max=0.0, iterations=1, balance_rel_max=0.0, balances={},
        flags=frozenset(), ood_distance={}, regime_status="IN_DOMAIN", assumptions=())


def test_prices_table_is_all_placeholders() -> None:
    with PRICES_PATH.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    assert "PLACEHOLDER" in raw["note"]
    prices = Prices.load(PRICES_PATH)
    assert set(prices.items) == set(CONSUMPTION_ITEMS)
    for item in CONSUMPTION_ITEMS:
        entry = prices.items[item]
        assert entry["status"] == "assumed"
        assert entry["assumed_label"] == "ASSUMED_PLACEHOLDER"
        lo, hi = entry["range"]
        assert lo <= entry["value"] <= hi
        assert entry["unit"].endswith("/mol") or entry["unit"].endswith("/L")
        assert entry["currency"] == "USD" and "placeholder" in entry["note"].lower()
    rows = prices.table()
    assert [r["item"] for r in rows] == list(CONSUMPTION_ITEMS)


def test_purity_recovery_enrichment_hand_built() -> None:
    spec = hand_spec()
    res = hand_result(spec)
    prices = Prices.load(PRICES_PATH)
    m = compute_metrics(res, spec, CE, "Nd", ["Pr"], prices, spec_limits={"purity_min": 0.95,
                                                                           "recovery_min": 0.5})
    n_T, n_I = 0.25 * 0.2, 0.25 * 0.002
    assert rel(m.purity_mol, n_T / (n_T + n_I)) < REL
    mNd, mPr = ATOMIC_MASS_G_MOL["Nd"], ATOMIC_MASS_G_MOL["Pr"]
    assert rel(m.purity_mass, n_T * mNd / (n_T * mNd + n_I * mPr)) < REL
    assert rel(m.purity_oxide, n_T * (mNd + 24.0) / (n_T * (mNd + 24.0) + n_I * (mPr + 24.0))) < REL
    assert oxide_mass_per_mol_metal("Nd") == pytest.approx((2 * mNd + 48.0) / 2)
    assert m.recovery_from_feed == 0.6 and rel(m.recovery_total, 0.05 / 0.075) < REL
    assert m.scrub_target_return is None and m.net_product_mol_h == 0.05
    assert rel(m.enrichment_factor["Pr"], (n_T / n_I) / (0.075 / 0.025)) < REL
    assert m.sf_by_stage[0]["Pr"] == pytest.approx(2.0) and m.sf_by_stage[2]["Pr"] == 2.0
    assert set(m.sf_tracer) == {"extraction", "scrub", "strip"}
    assert m.n_stages_total == 3 and (m.n_ext, m.n_scr, m.n_str) == (1, 1, 1)
    assert m.oa_ext == 2.0 and m.s_over_a == 0.5 and m.w_over_a == 0.25
    assert rel(m.throughput_mol_T_per_h_per_L_org, n_T / 2.0) < REL
    kg = n_T * (mNd + 24.0) / 1000.0
    assert rel(m.throughput_kg_oxide_per_h, kg) < REL
    assert m.max_loading_fraction[ts.CE_LIGAND] == 0.2325
    assert m.phase["loc_metal_M"].value is None and m.phase["regenerability_note"] == ""
    assert m.on_spec is True and m.regime_status == "IN_DOMAIN_WITH_CAVEATS"
    c = m.consumption
    assert rel(c["acid_mol_h"], 0.5 * 0.2 + 0.25 * 2.0) < REL
    assert c["base_mol_h"] == 0.0
    assert rel(c["salting_anion_mol_h"], 0.5 * 0.1) < REL
    assert c["complexant_mol_h"] == 0.0
    assert rel(c["water_L_h"], 0.75) < REL
    assert rel(c["acid_mol_per_mol_T"], c["acid_mol_h"] / n_T) < REL
    assert rel(c["acid_mol_per_kg_oxide"], c["acid_mol_h"] / kg) < REL
    # ligand loss is null and f_bleed = 0: make-up NaN + flag; diluent NaN; cost incomplete
    assert math.isnan(c["extractant_makeup_mol_h"]) and math.isnan(c["diluent_L_h"])
    assert Flag.LIGAND_LOSS_NOT_MEASURED in m.flags and Flag.COST_INCOMPLETE in m.flags
    assert math.isnan(m.cost_proxy_per_kg_oxide)


def test_consumption_with_bleed_and_saponification_and_cost() -> None:
    spec = hand_spec(sap=0.25, f_bleed=0.1)
    res = hand_result(spec)
    prices = Prices.load(PRICES_PATH)
    m = compute_metrics(res, spec, CE, "Nd", ["Pr"], prices)
    c = m.consumption
    assert rel(c["base_mol_h"], 2.0 * 0.25 * ts.CE_HA_TOTAL) < REL
    assert rel(c["extractant_makeup_mol_h"], 0.1 * 2.0 * ts.CE_HA_TOTAL) < REL   # monomer basis
    assert rel(c["diluent_L_h"], 0.2) < REL
    assert Flag.LIGAND_LOSS_NOT_MEASURED in m.flags          # the loss term is still unmeasured
    assert Flag.COST_INCOMPLETE not in m.flags
    expected = 0.0
    for item in CONSUMPTION_ITEMS:
        unit = "L" if item in ("diluent", "water") else "mol"
        expected += prices.price(item) * c[f"{item}_{unit}_per_kg_oxide"]
    assert rel(m.cost_proxy_per_kg_oxide, expected) < REL
    # a missing price -> NaN + COST_INCOMPLETE; no price table -> the same
    m2 = compute_metrics(res, spec, CE, "Nd", ["Pr"], prices.without("water"))
    assert math.isnan(m2.cost_proxy_per_kg_oxide) and Flag.COST_INCOMPLETE in m2.flags
    m3 = compute_metrics(res, spec, CE, "Nd", ["Pr"], None)
    assert math.isnan(m3.cost_proxy_per_kg_oxide) and Flag.COST_INCOMPLETE in m3.flags
    assert m2.consumption == m.consumption
    # acidification and dilution keywords
    m4 = compute_metrics(res, spec, CE, "Nd", ["Pr"], prices, feed_acidification_mol_h=0.3,
                         feed_dilution_L_h=1.0)
    assert rel(m4.consumption["acid_mol_h"], c["acid_mol_h"] + 0.3) < REL
    assert rel(m4.consumption["water_L_h"], c["water_L_h"] + 1.0) < REL


def test_failed_result_gives_nan_metrics() -> None:
    spec = hand_spec()
    res = replace(hand_result(spec), status="failed", flags=frozenset({Flag.NOT_CONVERGED}))
    m = compute_metrics(res, spec, CE, "Nd", ["Pr"], Prices.load(PRICES_PATH))
    assert m.purity_mol is None and m.recovery_from_feed is None and m.on_spec is None
    assert m.throughput_kg_oxide_per_h is None and m.cost_proxy_per_kg_oxide is None
    assert math.isnan(m.enrichment_factor["Pr"])
    assert all(math.isnan(v) for v in m.consumption.values())
    assert m.regime_status == "INADMISSIBLE" and Flag.NOT_CONVERGED in m.flags
    assert m.n_stages_total == 3 and m.oa_ext == 2.0


def test_metrics_of_a_solved_cascade() -> None:
    scrub = AqStream(0.3, {"Pr": 0.0, "Nd": 0.0}, 0.2, 0.2, 0.0, 0.0)
    strip = AqStream(0.5, {"Pr": 0.0, "Nd": 0.0}, 3.0, 3.0, 0.0, 0.0)
    spec = CascadeSpec(4, 2, 2, ts.feed_prnd(), scrub, strip, 1.0,
                       {ts.CE_LIGAND: ts.CE_LT_DIMER}, 0.0, target="Nd")
    res = solve_cascade(spec, CE)
    assert res.status == "converged_newton"
    m = compute_metrics(res, spec, CE, "Nd", ["Pr"], Prices.load(PRICES_PATH),
                        spec_limits={"purity_min": 0.5, "recovery_min": 0.1})
    assert rel(m.recovery_from_feed, res.origin["recovery_from_feed"]) < REL
    n_T = res.product.flow_L_h * res.product.metals["Nd"]
    assert rel(m.recovery_total, n_T / 0.075) < REL
    assert 0.0 < m.purity_mol < 1.0 and m.on_spec is not None
    assert m.sf_tracer["extraction"]["Pr"] == pytest.approx(10 ** (-1.95 + 2.10), rel=1e-9)
    assert m.enrichment_factor["Pr"] > 1.0
    assert m.regime_status == res.regime_status or Flag.COST_INCOMPLETE in m.flags
    # the same metrics for a target the cascade was not solved with (origin recomputed)
    m_pr = compute_metrics(res, spec, CE, "Pr", ["Nd"], None)
    n_Pr = res.product.flow_L_h * res.product.metals["Pr"]
    assert rel(m_pr.recovery_total, n_Pr / 0.025) < REL
    assert rel(m_pr.recovery_from_feed, m_pr.recovery_total) < 1e-9     # no other Pr source
    with pytest.raises(ValueError, match="target"):
        compute_metrics(res, spec, CE, "La", ["Pr"], None)
