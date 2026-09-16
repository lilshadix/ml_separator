"""The registered hold-out designs run through the leakage guard on the archive (P3, P6, P7, P1, P4).

The toy-frame guard tests live in ``test_leakage.py`` / ``test_support_graph.py``.  Here the REGISTERED splits of
``preregistration_draft.md`` are built on the MODEL rows and every one must pass
``fold_isolation_check`` at its design level with ``near_dup_value_tol=0.005`` (a violation aborts a run):

* V1 -- leave-one-group-out over ``group_cross_publication_copy`` groups with >= 20 rows + one remainder fold;
* V2 -- the 23 states (>= 100 rows, >= 5 systems), element-level hiding;
* V5 -- the primary cells (k10 / p1 / m3) under the registered component-aware hiding;
* V6 -- the 13 double cells (rows5 / otherln2).

It also pins the V6 carve-out (no V5 discovery cell scores a ``V6_TARGET_ROWS`` row) and the selection /
confirmation halves.  The sweeps read ``data_audit/`` outputs and take about two minutes: marked ``slow``.
"""
from __future__ import annotations

import pandas as pd
import pytest

from gen19ct import paths
from gen19ct.chemistry import support_graph as SG
from gen19ct.data import leakage as L
from gen19ct.data import load
from gen19ct.folds import registered as FR

TOL = 0.005
CELLS = paths.DATA_AUDIT_DIR / "feasibility_v5_cells.csv"
STATES = paths.DATA_AUDIT_DIR / "feasibility_metal_states.csv"
COMPONENTS = paths.DATA_AUDIT_DIR / "leakage_publication_components.csv"
HALVES = paths.DATA_AUDIT_DIR / "feasibility_halves.csv"

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    for p in (CELLS, STATES, COMPONENTS):
        if not p.exists():
            pytest.skip(f"{p.name} not built")
    m = load.load_model_rows(copy=True)
    return m, SG.prepare_support_frame(m)


def _fail(rep: dict) -> dict:
    return {k: v for k, v in rep["violations"].items() if v}


def test_registered_v1_folds_pass_the_guard(frame) -> None:
    m, _ = frame
    comp = pd.read_csv(COMPONENTS).set_index("g19_publication_id")
    grp = m["g19_publication_id"].map(comp["group_cross_publication_copy"])
    assert grp.notna().all()
    sizes = grp.value_counts()
    folds = [grp == g for g in sorted(sizes[sizes >= 20].index)] + [grp.isin(set(sizes[sizes < 20].index))]
    bad = []
    for te in folds:
        rep = L.fold_isolation_check(m.index[~te], m.index[te], m, "V1", near_dup_value_tol=TOL, raise_on_violation=False)
        if not rep["ok"]:
            bad.append(_fail(rep))
    assert bad == []
    # every archive duplicate group lies inside one V1 group (the P3 repair)
    span = m.assign(_g=grp).dropna(subset=["duplicate_group_id"]).groupby("duplicate_group_id")["_g"].nunique()
    assert (span == 1).all()


def test_registered_v2_folds_pass_the_guard(frame) -> None:
    m, _ = frame
    ms = pd.read_csv(STATES)
    states = ms.loc[ms["v2_eligible__rows100_systems5"].astype(bool), "metal_state"].tolist()
    assert len(states) == 23
    bad = []
    for st in states:
        te = m.index[m["g19_metal_state"] == st]
        tr = m.index[m["g19_metal"] != st.split("(")[0]]
        rep = L.fold_isolation_check(tr, te, m, "V2", element_level=True, near_dup_value_tol=TOL, raise_on_violation=False)
        if not rep["ok"]:
            bad.append((st, _fail(rep)))
    assert bad == []


def test_registered_v5_primary_cells_pass_the_component_aware_guard(frame) -> None:
    m, fr = frame
    cells = pd.read_csv(CELLS)
    prim = cells[cells["eligible__g19_publication_id__k10_p1_m3"].astype(bool)]
    assert len(prim) > 0
    bad, kept_other_state = [], 0
    for state, sy in zip(prim["metal_state"], prim["extractant_system_key"]):
        train = SG.hide_cell(fr, state, sy, component_aware=True)
        te = fr.index[(fr["g19_metal_state"] == state) & (fr["extractant_system_key"] == sy)]
        assert not ((train["g19_metal_state"] == state) & (train["extractant_system_key"] == sy)).any()  # brief section 27
        rep = L.fold_isolation_check(train.index, te, m, "V5", component_aware=True, near_dup_value_tol=TOL,
                                     raise_on_violation=False)
        kept_other_state += rep["warnings"]["V5_other_known_state_of_hidden_element_in_component_sharing_system_rows"]
        if not rep["ok"]:
            bad.append((state, sy[:40], _fail(rep)))
    assert bad == []
    # the state-level rule keeps exactly the other-known-state rows the feasibility table counts
    assert kept_other_state == int(prim["other_known_state_rows_in_other_systems_sharing_a_component"].sum())
    # the element-level hiding it replaced also passes the guard (it is stricter), so the guard does not force either
    state, sy = "Pu(IV)", prim.loc[prim["metal_state"] == "Pu(IV)", "extractant_system_key"].iloc[0]
    sharing = SG.systems_sharing_component(fr["extractant_system_key"].dropna().unique(), sy)
    strict = SG.hide_cell(fr, state, sy, component_aware=True)
    strict = strict[~(strict["extractant_system_key"].isin(sharing) & (strict["g19_metal"] == "Pu"))]
    te = fr.index[(fr["g19_metal_state"] == state) & (fr["extractant_system_key"] == sy)]
    assert L.fold_isolation_check(strict.index, te, m, "V5", component_aware=True, near_dup_sig=None)["ok"]


def test_registered_v6_double_cells_pass_the_guard_and_the_carve_out_holds(frame) -> None:
    m, fr = frame
    v6 = FR.v6_system_set(fr)
    assert len(v6) == 13
    bad = []
    for sy in sorted(v6):
        train = SG.hide_cells(fr, [("Pr(III)", sy), ("Nd(III)", sy)], component_aware=True)
        te = fr.index[fr["g19_metal_state"].isin(["Pr(III)", "Nd(III)"]) & (fr["extractant_system_key"] == sy)]
        rep = L.fold_isolation_check(train.index, te, m, "V6", near_dup_value_tol=TOL, raise_on_violation=False)
        if not rep["ok"]:
            bad.append((sy[:40], _fail(rep)))
    assert bad == []
    mask = FR.v6_target_mask(fr, v6)
    cells = pd.read_csv(CELLS)
    scored = cells[cells["scored_primary_discovery"].astype(bool)]
    keys = set(zip(scored["metal_state"], scored["extractant_system_key"]))
    idx = fr.index[[(s, k) in keys for s, k in zip(fr["g19_metal_state"], fr["extractant_system_key"])]]
    FR.assert_not_scored(idx, mask, "V5 primary discovery cells")
    with pytest.raises(AssertionError):
        FR.assert_not_scored(fr.index[mask][:3], mask)


def test_registered_halves_are_complete_and_reproducible(frame) -> None:
    """P4: ``feasibility_halves.csv`` is exactly ``registered.assign_halves`` under the registered weights and strata
    (V5 systems: diglycolamide / other, weight = scored primary cells; V1 groups: no strata, weight = MODEL rows;
    V2 states: focus lanthanide / other lanthanide / actinide, weight 1)."""
    if not HALVES.exists():
        pytest.skip("feasibility_halves.csv not built")
    m, _ = frame
    h = pd.read_csv(HALVES)
    assert set(h["half"]) == {"S", "C"} and h.groupby("design")["unit"].apply(lambda u: u.is_unique).all()
    assert set(h["design"]) == {"V1", "V5_system", "V2_state"}
    cells = pd.read_csv(CELLS)
    scored = cells[cells["scored_primary_discovery"].astype(bool)]

    # V5: the units are every system listed in the cells table; weights recomputed from the scored cells
    v5 = h[h["design"] == "V5_system"]
    fam = cells.groupby("extractant_system_key")["system_family"].first()
    assert set(v5["unit"]) == set(fam.index)
    w5 = scored.groupby("extractant_system_key").size().reindex(fam.index, fill_value=0).astype(float)
    assert dict(zip(v5["unit"], v5["weight"])) == w5.to_dict()
    again = FR.assign_halves(w5.to_dict(), strata={k: ("diglycolamide" if f == "diglycolamide" else "other")
                                                    for k, f in fam.items()})
    assert again == dict(zip(v5["unit"], v5["half"]))
    assert dict(zip(cells["extractant_system_key"], cells["selection_half"])) == {k: again[k] for k in cells["extractant_system_key"]}
    per_half = scored.groupby("selection_half").size()
    assert abs(int(per_half.get("S", 0)) - int(per_half.get("C", 0))) <= 2

    # V1: groups of group_cross_publication_copy with >= 20 MODEL rows plus the remainder fold, no strata
    comp = pd.read_csv(COMPONENTS).set_index("g19_publication_id")
    sizes = m["g19_publication_id"].map(comp["group_cross_publication_copy"]).value_counts()
    w1 = {g: float(n) for g, n in sizes.items() if n >= 20}
    w1["REMAINDER"] = float(sizes[sizes < 20].sum())
    v1 = h[h["design"] == "V1"]
    assert dict(zip(v1["unit"], v1["weight"])) == w1
    assert FR.assign_halves(w1) == dict(zip(v1["unit"], v1["half"]))

    # V2: the 23 eligible states, stratified focus lanthanide / other lanthanide / actinide
    ms = pd.read_csv(STATES)
    elig = ms[ms["v2_eligible__rows100_systems5"].astype(bool)]
    focus = {"La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd"}
    strata = {st: ("ln_focus" if el in focus and cat == "lanthanide" else cat)
              for st, el, cat in zip(elig["metal_state"], elig["element"], elig["metal_category"])}
    v2 = h[h["design"] == "V2_state"]
    assert FR.assign_halves({st: 1.0 for st in elig["metal_state"]}, strata=strata) == dict(zip(v2["unit"], v2["half"]))
