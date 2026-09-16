"""Tests for ``gen19ct.chemistry.support_graph`` (brief sections 16 and 27 "support-score correctness").

Toy frames only (no archive read): component counting, bridges, the section-16 support features,
fold-wise computation (a hidden cell is absent from the index built without it), and the full
multigraph's node / edge types.
"""
from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from gen19ct.chemistry import support_graph as SG

TODGA = "CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC"
TBDGA = "CCCCN(CCCC)C(=O)COCC(=O)N(CCCC)CCCC"
TDDDGA = "CCCCCCCCCCCCN(CCCCCCCCCCCC)C(=O)COCC(=O)N(CCCCCCCCCCCC)CCCCCCCCCCCC"
DHOA = "CCCCCCCC(=O)N(CCCCCC)CCCCCC"
TBP = "CCCCOP(=O)(OCCCC)OCCCC"
FAMILY = {TODGA: "diglycolamide", TBDGA: "diglycolamide", DHOA: "monoamide", TBP: "neutral_organophosphate"}


def _rows(metal_state, system, n, pub="pub_a", acid=1.0, ext=0.1, temp=25.0, element=None):
    elem = element if element is not None else metal_state.split("(")[0]
    return [{
        SG.METAL_COL: metal_state, SG.ELEMENT_COL: elem, SG.SYSTEM_COL: system, SG.PUB_COL: pub,
        SG.FAMILY_COL: FAMILY[system], SG.MECH_COL: "NEUTRAL_SOLVATING", SG.SMILES_COL: system,
        SG.ACID_ANION_COL: "nitrate", SG.DILUENT_COL: "aliphatic",
        SG.LOG_ACID_COL: math.log10(acid * (1 + 0.1 * i)), SG.LOG_EXT_COL: math.log10(ext), SG.TEMP_COL: temp,
    } for i in range(n)]


@pytest.fixture()
def toy() -> pd.DataFrame:
    rows = []
    rows += _rows("Ce(III)", TODGA, 3)
    rows += _rows("Pr(III)", TODGA, 4)
    rows += _rows("Nd(III)", TODGA, 5, pub="pub_b")
    rows += _rows("Sm(III)", TODGA, 2)
    rows += _rows("Am(III)", TODGA, 6, pub="pub_c")
    rows += _rows(None, TODGA, 2, element="Nd")              # unknown-state Nd alias rows
    rows += _rows("Nd(III)", TBDGA, 4, pub="pub_d")
    rows += _rows("Eu(III)", TBDGA, 3, pub="pub_d")
    rows += _rows("Nd(III)", DHOA, 1, pub="pub_e")
    rows += _rows("U(VI)", TBP, 7, pub="pub_f")               # an isolated island
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- #
# components and bridges
# --------------------------------------------------------------------------------------------- #

def test_component_counting_on_toy_graph(toy):
    G = SG.bipartite_graph(toy)
    assert SG.n_components(G) == 2
    ct = SG.component_table(G)
    assert list(ct["n_rows"]) == [28, 7]                     # alias rows are not graph edges by default
    assert bool(ct.loc[0, "is_giant"]) and not bool(ct.loc[1, "is_giant"])
    assert ct.loc[0, "n_systems"] == 3 and ct.loc[0, "n_metal_states"] == 6
    assert ct.loc[1, "metal_states"] == "U(VI)"
    assert ct["share_rows"].sum() == pytest.approx(1.0)

    G5 = SG.bipartite_graph(toy, min_rows=5)                 # only Nd-TODGA, Am-TODGA, U-TBP survive
    assert SG.n_components(G5) == 2
    assert sorted(SG.component_table(G5)["n_rows"]) == [7, 11]

    # member lists use "; " because multi-extractant system keys contain "|"
    mixed = pd.DataFrame(_rows("Eu(III)", TODGA, 1))
    mixed[SG.SYSTEM_COL] = TODGA + "|" + TBP
    ctm = SG.component_table(SG.bipartite_graph(pd.concat([toy, mixed], ignore_index=True)))
    assert ctm.loc[0, "n_systems"] == 4 and len(ctm.loc[0, "systems"].split(SG.MEMBER_SEP)) == 4
    assert (TODGA + "|" + TBP) in ctm.loc[0, "systems"].split(SG.MEMBER_SEP)

    Gu = SG.bipartite_graph(toy, include_unknown_state=True)
    assert "M|Nd(?)" in Gu and Gu.edges["M|Nd(?)", f"S|{TODGA}"]["weight"] == 2


def test_component_counting_pure_networkx_toy():
    G = nx.Graph()
    for m, s, w in [("A(III)", "s1", 3), ("B(III)", "s1", 1), ("B(III)", "s2", 2), ("C(II)", "s3", 4)]:
        G.add_node(f"M|{m}", node_type="metal_state")
        G.add_node(f"S|{s}", node_type="extractant_system")
        G.add_edge(f"M|{m}", f"S|{s}", weight=w)
    ct = SG.component_table(G)
    assert len(ct) == 2 and list(ct["n_rows"]) == [6, 4] and list(ct["n_systems"]) == [2, 1]


def test_bridges_and_connectivity_without_cell(toy):
    G = SG.bipartite_graph(toy)
    # Nd(III)-DHOA is a pendant edge: hiding it cuts DHOA off from Nd
    assert SG.cell_is_bridge(G, "Nd(III)", DHOA)
    assert not SG.connected_without(G, "Nd(III)", DHOA)
    # Ce(III)-TODGA is pendant too (Ce has no other system)
    assert SG.cell_is_bridge(G, "Ce(III)", TODGA)
    # Nd(III)-TBDGA: TBDGA's only other metal (Eu) has no path back to Nd -> a bridge
    assert SG.cell_is_bridge(G, "Nd(III)", TBDGA)
    extra = pd.DataFrame(_rows("Eu(III)", TODGA, 2))
    G2 = SG.bipartite_graph(pd.concat([toy, extra], ignore_index=True))
    assert SG.connected_without(G2, "Nd(III)", TODGA)       # Nd-TBDGA-Eu-TODGA closes a cycle
    assert not SG.cell_is_bridge(G2, "Nd(III)", TODGA)
    assert G2.has_edge("M|Nd(III)", f"S|{TODGA}")          # the check restores the edge
    bridges = SG.bridge_set(G2)
    assert frozenset({"M|Nd(III)", f"S|{DHOA}"}) in bridges
    assert frozenset({"M|Nd(III)", f"S|{TODGA}"}) not in bridges


# --------------------------------------------------------------------------------------------- #
# support features
# --------------------------------------------------------------------------------------------- #

def test_support_features_exact_pair_and_neighbours(toy):
    f = SG.support_features("Nd(III)", TODGA, {"acid_M": 1.0, "extractant_M": 0.1, "temperature_C": 25.0}, toy)
    assert f["exact_pair_exists"] is True and f["exact_pair_rows"] == 5 and f["exact_pair_publications"] == 1
    assert f["alias_unknown_state_rows"] == 2
    assert f["n_neighbour_metals"] == 4                       # Ce, Pr, Sm, Am
    assert f["neighbour_metals"] == "Am(III)|Ce(III)|Pr(III)|Sm(III)"
    assert f["n_neighbour_metals_same_category"] == 3          # the three lanthanides
    assert f["n_neighbour_metals_same_charge"] == 4
    assert f["n_publications_system"] == 3 and f["n_publications_metal"] == 3 and f["n_publications_pair"] == 1
    # same family: Nd rows under TODGA (5) + TBDGA (4); DHOA is another family
    assert f["system_family"] == "diglycolamide"
    assert f["n_same_family_rows_for_metal"] == 9
    assert f["n_same_family_other_system_rows_for_metal"] == 4
    assert f["n_same_family_other_systems_for_metal"] == 1
    assert f["n_systems_for_metal"] == 3
    assert f["condition_distance_system"] == pytest.approx(0.0, abs=1e-9)
    assert f["condition_dims_used"] == 3


def test_support_features_series_neighbours(toy):
    f = SG.support_features("Nd(III)", TODGA, None, toy)
    assert f["series_neighbours_present"] == "Ce(III)|Pr(III)|Sm(III)"   # Pm absent
    assert f["n_series_neighbours_pm1"] == 1                              # Pr only
    assert f["n_series_neighbours_pm2"] == 3
    assert f["series_bracketed"] is True
    g = SG.support_features("Ce(III)", TODGA, None, toy)
    assert g["series_neighbours_present"] == "Pr(III)|Nd(III)"            # La absent
    assert g["series_bracketed"] is False
    assert SG.series_neighbour_labels("Nd(III)") == {-2: "Ce(III)", -1: "Pr(III)", 1: "Pm(III)", 2: "Sm(III)"}
    assert SG.series_neighbour_labels("U(VI)")[1] == "Np(VI)"
    assert SG.series_neighbour_labels("Zr(IV)") == {}


def test_support_features_nearest_radius(toy):
    f = SG.support_features("Nd(III)", TODGA, None, toy)
    # CN8 radii (Shannon): Nd 1.109; Pr 1.126 (.017), Am 1.09 (.019), Sm 1.079 (.030), Ce 1.143 (.034)
    assert f["nearest_radius_basis"] == "CN8"
    assert f["nearest_radius_metal"] == "Pr(III)"
    assert f["nearest_radius_distance_A"] == pytest.approx(1.126 - 1.109, abs=1e-9)
    assert f["nearest_radius_same_charge"] is True
    assert f["n_radius_neighbours_within_tol"] == 4          # Ce .034, Pr .017, Sm .030, Am .019
    h = SG.support_features("Eu(III)", TBDGA, None, toy)
    assert h["nearest_radius_metal"] == "Nd(III)"
    assert h["nearest_radius_distance_A"] == pytest.approx(1.109 - 1.066, abs=1e-9)


def test_support_features_nearest_ligand(toy):
    f = SG.support_features("Nd(III)", TODGA, None, toy)
    assert f["nearest_ligand_system"] == TBDGA               # the other diglycolamide beats DHOA
    assert 0.0 < f["nearest_ligand_tanimoto"] < 1.0
    assert f["nearest_ligand_distance"] == pytest.approx(1.0 - f["nearest_ligand_tanimoto"])
    assert SG.tanimoto(TODGA, TODGA) == pytest.approx(1.0)
    assert SG.tanimoto(TODGA, TBDGA) > SG.tanimoto(TODGA, TBP)
    # pinned limitation: Morgan r=2 saturates on long linear chains -- TODGA and TDdDGA are identical
    assert SG.tanimoto(TODGA, TDDDGA) == pytest.approx(1.0)


def test_hidden_cell_features_computed_without_the_cell(toy):
    train = SG.hide_cell(toy, "Nd(III)", TODGA)
    assert not ((train[SG.METAL_COL] == "Nd(III)") & (train[SG.SYSTEM_COL] == TODGA)).any()
    assert not (train[SG.METAL_COL].isna() & (train[SG.ELEMENT_COL] == "Nd") & (train[SG.SYSTEM_COL] == TODGA)).any()
    assert len(train) == len(toy) - 7
    f = SG.support_features("Nd(III)", TODGA, {"acid_M": 1.0, "extractant_M": 0.1, "temperature_C": 25.0}, train)
    assert f["exact_pair_exists"] is False and f["exact_pair_rows"] == 0
    assert f["alias_unknown_state_rows"] == 0
    assert f["n_publications_pair"] == 0 and math.isnan(f["condition_distance_pair"])
    # the transfer evidence is still there
    assert f["n_neighbour_metals"] == 4 and f["series_neighbours_present"] == "Ce(III)|Pr(III)|Sm(III)"
    assert f["n_same_family_other_system_rows_for_metal"] == 4
    assert f["nearest_ligand_system"] == TBDGA
    # the index built on the full toy still sees it: fold-wise computation matters
    assert SG.SupportIndex(toy).features("Nd(III)", TODGA)["exact_pair_exists"] is True
    # hiding keeps the alias rows when asked to
    assert len(SG.hide_cell(toy, "Nd(III)", TODGA, include_unknown_state_alias=False)) == len(toy) - 5


def test_condition_standardisation_is_fit_on_train(toy):
    idx = SG.SupportIndex(toy)
    x = pd.to_numeric(toy[SG.TEMP_COL])
    assert idx.cond_mean[2] == pytest.approx(x.mean())
    assert idx.cond_std[2] == 1.0                             # zero spread -> unit scale, no division by 0
    z = idx.standardise({"log10_acid_M": 0.0, "temperature_C": 25.0})
    assert np.isnan(z[1]) and z[2] == pytest.approx(0.0)
    d, k = idx.condition_distance(TODGA, {"acid_M": 1.0}, same_acid_anion=False)
    assert k == 1 and d == pytest.approx(0.0, abs=1e-9)
    assert math.isnan(idx.condition_distance("not-a-system", {"acid_M": 1.0})[0])


def test_unseen_system_uses_static_labels(toy):
    systems = pd.DataFrame({"extractant_system_key": [TBP + "x"], "system_family": ["diglycolamide"],
                            "primary_extractant_smiles": [TBDGA]})
    f = SG.support_features("Nd(III)", TBP + "x", None, toy, systems=systems)
    assert f["exact_pair_exists"] is False and f["n_neighbour_metals"] == 0
    assert f["system_family"] == "diglycolamide" and f["n_same_family_other_system_rows_for_metal"] == 9
    assert f["nearest_ligand_system"] == TBDGA and f["nearest_ligand_tanimoto"] == pytest.approx(1.0)


# --------------------------------------------------------------------------------------------- #
# full multigraph
# --------------------------------------------------------------------------------------------- #

def test_support_multigraph_types_and_weights(toy):
    G = SG.build_support_graph(toy)
    summ = SG.graph_summary(G)
    assert set(summ["nodes_by_type"]) == set(SG.NODE_TYPES)
    assert summ["nodes_by_type"]["metal_state"] == 8           # 7 known + Nd(?)
    assert summ["nodes_by_type"]["extractant_system"] == 4
    assert summ["nodes_by_type"]["family"] == 3
    assert summ["nodes_by_type"]["publication"] == 6
    assert summ["nodes_by_type"]["condition_regime"] >= 1
    assert summ["edge_row_weight_by_type"]["measured"] == len(toy)
    assert summ["edge_row_weight_by_type"]["shared_family"] == len(toy)
    assert summ["edge_row_weight_by_type"]["shared_publication"] == 2 * len(toy)
    assert G.has_edge("M|Nd(III)", f"S|{TODGA}", key="measured")
    assert G.edges["M|Nd(III)", f"S|{TODGA}", "measured"]["weight"] == 5
    # the TBP island joins the rest only through the shared condition regime hub
    assert summ["n_components"] == 1
    assert SG.regime_label("nitrate", -0.2, "aliphatic") == "nitrate|[-0.5,0)|aliphatic"
    assert SG.acid_bin(float("nan")) == "NA"


def test_missing_columns_raise(toy):
    with pytest.raises(KeyError):
        SG.SupportIndex(toy.drop(columns=[SG.FAMILY_COL]))


# --------------------------------------------------------------------------------------------- #
# corrections of the Phase A/B verification (P2, P6, P10, V-NUM-01)
# --------------------------------------------------------------------------------------------- #

def test_nearest_radius_prefers_same_charge_before_basis():
    """P2: U(VI) must borrow Pu(VI) (CN6 radius only) rather than Pu(IV) (CN8 radius) -- charge first, basis second."""
    rows = _rows("U(VI)", TBP, 3) + _rows("Pu(VI)", TBP, 3) + _rows("Pu(IV)", TBP, 3)
    t = pd.DataFrame(rows)
    assert np.isnan(SG.metal_properties("Pu(VI)")["r_cn8"]) and np.isfinite(SG.metal_properties("Pu(IV)")["r_cn8"])
    f = SG.support_features("U(VI)", TBP, None, SG.hide_cell(t, "U(VI)", TBP))
    assert f["nearest_radius_metal"] == "Pu(VI)"
    assert f["nearest_radius_basis"] == "CN6"
    assert f["nearest_radius_same_charge"] is True and f["nearest_radius_same_species_charge"] is True
    # with no same-charge candidate it falls back to another charge and says so
    g = SG.support_features("U(VI)", TBP, None, t[t[SG.METAL_COL] == "Pu(IV)"])
    assert g["nearest_radius_metal"] == "Pu(IV)" and g["nearest_radius_same_charge"] is False
    assert g["radius_bracketed_same_charge"] is False


def test_radius_bracket_for_interpolation(toy):
    """P10: the two same-charge ends of a radius interpolation (B3i)."""
    train = SG.hide_cell(toy, "Nd(III)", TODGA)
    f = SG.support_features("Nd(III)", TODGA, None, train)
    # CN8: Ce 1.143, Pr 1.126 above Nd 1.109; Am 1.09, Sm 1.079 below
    assert f["radius_bracket_upper_metal"] == "Pr(III)" and f["radius_bracket_lower_metal"] == "Am(III)"
    assert f["radius_bracketed_same_charge"] is True
    h = SG.support_features("Ce(III)", TODGA, None, SG.hide_cell(toy, "Ce(III)", TODGA))
    assert h["radius_bracket_upper_metal"] is None and h["radius_bracketed_same_charge"] is False


def test_component_aware_hiding_removes_state_and_alias_rows_in_sharing_systems():
    """P6 / V-NUM-01: the registered V5 hiding removes the hidden state's rows AND the element's X(?) rows from every
    other system sharing a component, keeps the other elements there, and keeps unrelated systems."""
    mix = f"{DHOA}|{TODGA}"
    base = _rows("Nd(III)", TODGA, 4) + _rows("Pr(III)", TODGA, 3) + _rows("Nd(III)", TBDGA, 2)
    shared = _rows("Nd(III)", TODGA, 2) + _rows(None, TODGA, 3, element="Nd") + _rows("Pr(III)", TODGA, 2)
    for r in shared:
        r[SG.SYSTEM_COL] = mix
    t = pd.DataFrame(base + shared)
    cell_only = SG.hide_cell(t, "Nd(III)", TODGA)
    assert ((cell_only[SG.SYSTEM_COL] == mix) & (cell_only[SG.ELEMENT_COL] == "Nd")).sum() == 5
    aware = SG.hide_cell(t, "Nd(III)", TODGA, component_aware=True)
    assert not ((aware[SG.SYSTEM_COL] == mix) & (aware[SG.ELEMENT_COL] == "Nd")).any()   # Nd has no other state here
    assert ((aware[SG.SYSTEM_COL] == mix) & (aware[SG.ELEMENT_COL] == "Pr")).sum() == 2
    assert ((aware[SG.SYSTEM_COL] == TBDGA) & (aware[SG.ELEMENT_COL] == "Nd")).sum() == 2
    assert SG.systems_sharing_component(t[SG.SYSTEM_COL], TODGA) == {mix}
    # a component map can widen "shares a component" (parent-key sensitivity)
    assert SG.systems_sharing_component(t[SG.SYSTEM_COL], TODGA, {TBDGA: "DGA", TODGA: "DGA"}) == {mix, TBDGA}
    both = SG.hide_cells(t, [("Nd(III)", TODGA), ("Pr(III)", TODGA)], component_aware=True)
    assert set(both[SG.SYSTEM_COL]) == {TBDGA}
    # without the alias rule the X(?) rows stay in both scopes
    no_alias = SG.hide_cell(t, "Nd(III)", TODGA, include_unknown_state_alias=False, component_aware=True)
    assert ((no_alias[SG.SYSTEM_COL] == mix) & no_alias[SG.METAL_COL].isna()).sum() == 3


def test_component_aware_hiding_is_symmetric_at_the_metal_state_level():
    """Design decision (pre-seal): Pu(VI) x TODGA hidden -> in TODGA AND in TODGA|DHOA the Pu(VI) rows and the Pu(?)
    rows go; Pu(IV) stays in both systems (a different species; element-level transfer is V2)."""
    mix = f"{DHOA}|{TODGA}"
    own = (_rows("Pu(VI)", TODGA, 4) + _rows("Pu(IV)", TODGA, 3) + _rows(None, TODGA, 2, element="Pu")
           + _rows("Am(III)", TODGA, 2))
    shared = (_rows("Pu(VI)", TODGA, 2, pub="pub_m") + _rows("Pu(IV)", TODGA, 5, pub="pub_m")
              + _rows(None, TODGA, 1, pub="pub_m", element="Pu") + _rows("Am(III)", TODGA, 1, pub="pub_m"))
    for r in shared:
        r[SG.SYSTEM_COL] = mix
    unrelated = _rows("Pu(VI)", TBP, 3, pub="pub_t") + _rows(None, TBP, 1, pub="pub_t", element="Pu")
    t = pd.DataFrame(own + shared + unrelated)
    h = SG.hide_cell(t, "Pu(VI)", TODGA, component_aware=True)

    def n(system, state=None, unknown=False):
        sel = (h[SG.SYSTEM_COL] == system) & (h[SG.ELEMENT_COL].isin(["Pu", "Am"]))
        sel &= h[SG.METAL_COL].isna() & (h[SG.ELEMENT_COL] == "Pu") if unknown else (h[SG.METAL_COL] == state)
        return int(sel.sum())
    for system in (TODGA, mix):                                  # the same rule in both scopes
        assert n(system, "Pu(VI)") == 0 and n(system, unknown=True) == 0
    assert n(TODGA, "Pu(IV)") == 3 and n(mix, "Pu(IV)") == 5       # the other known state stays everywhere
    assert n(TODGA, "Am(III)") == 2 and n(mix, "Am(III)") == 1
    assert n(TBP, "Pu(VI)") == 3 and n(TBP, unknown=True) == 1     # a system sharing no component is untouched
    assert len(h) == len(t) - (4 + 2 + 2 + 1)
    # hiding Pu(IV) x TODGA|DHOA reaches TODGA through the shared component, symmetrically
    g = SG.hide_cell(t, "Pu(IV)", mix, component_aware=True)
    assert not (g[SG.SYSTEM_COL].isin([TODGA, mix]) & ((g[SG.METAL_COL] == "Pu(IV)")
                                                        | (g[SG.METAL_COL].isna() & (g[SG.ELEMENT_COL] == "Pu")))).any()
    assert int((g[SG.SYSTEM_COL].isin([TODGA, mix]) & (g[SG.METAL_COL] == "Pu(VI)")).sum()) == 6
    # the leakage guard enforces exactly this rule
    from gen19ct.data import leakage as LK
    guard = t.assign(g19_publication_id=t[SG.PUB_COL])
    te = t.index[(t[SG.METAL_COL] == "Pu(VI)") & (t[SG.SYSTEM_COL] == TODGA)]
    rep = LK.fold_isolation_check(h.index, te, guard, "V5", component_aware=True, near_dup_sig=None)
    assert rep["ok"] and rep["warnings"]["V5_other_known_state_of_hidden_element_in_component_sharing_system_rows"] == 5
    element_level = t[~(t[SG.SYSTEM_COL].isin([TODGA, mix]) & (t[SG.ELEMENT_COL] == "Pu")
                        & ~((t[SG.SYSTEM_COL] == TODGA) & (t[SG.METAL_COL] == "Pu(IV)")))]
    assert LK.fold_isolation_check(element_level.index, te, guard, "V5", component_aware=True, near_dup_sig=None)["ok"]
    cell_only = SG.hide_cell(t, "Pu(VI)", TODGA)
    rep = LK.fold_isolation_check(cell_only.index, te, guard, "V5", component_aware=True, near_dup_sig=None,
                                  raise_on_violation=False)
    assert rep["violations"]["V5_hidden_state_in_component_sharing_system"] == 3      # 2 Pu(VI) + 1 Pu(?) in the mix


def test_v6_target_rows_and_scoring_guard():
    """P1: V6_TARGET_ROWS = Pr/Nd rows (known and X(?)) of a V6 system and of systems sharing a component with it."""
    from gen19ct.folds import registered as FR
    mix = f"{DHOA}|{TODGA}"
    rows = (_rows("Pr(III)", TODGA, 5) + _rows("Nd(III)", TODGA, 5) + _rows("Ce(III)", TODGA, 2)
            + _rows("Sm(III)", TODGA, 2) + _rows(None, TODGA, 2, element="Nd") + _rows("Nd(III)", TBDGA, 3))
    shared = _rows("Pr(III)", TODGA, 2) + _rows("Eu(III)", TODGA, 2)
    for r in shared:
        r[SG.SYSTEM_COL] = mix
    t = pd.DataFrame(rows + shared)
    assert FR.v6_system_set(t) == {TODGA}
    mask = FR.v6_target_mask(t, {TODGA})
    assert int(mask.sum()) == 5 + 5 + 2 + 2              # Pr, Nd, Nd(?) under TODGA and Pr under the mixture
    assert not mask[t[SG.SYSTEM_COL] == TBDGA].any() and not mask[t[SG.ELEMENT_COL] == "Eu"].any()
    FR.assert_not_scored(t.index[~mask], mask)
    with pytest.raises(AssertionError, match="V6_TARGET_ROWS"):
        FR.assert_not_scored(t.index, mask)


def test_assign_halves_is_deterministic_and_balanced():
    """P4: the registered selection / confirmation split does not depend on dict order and balances weight."""
    from gen19ct.folds import registered as FR
    w = {f"u{i}": float(i % 7 + 1) for i in range(40)}
    a = FR.assign_halves(w)
    b = FR.assign_halves(dict(reversed(list(w.items()))))
    assert a == b and set(a.values()) == {"S", "C"}
    load = {h: sum(w[u] for u, x in a.items() if x == h) for h in ("S", "C")}
    assert abs(load["S"] - load["C"]) <= max(w.values())
    strata = {u: ("A" if int(u[1:]) < 10 else "B") for u in w}
    s = FR.assign_halves(w, strata=strata)
    assert {s[u] for u in w if strata[u] == "A"} == {"S", "C"}
