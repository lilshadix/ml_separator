"""Tests of the fold builders (``gen19ct.folds``) and of the registered fold files in ``folds/`` (brief sections 11,
12 and 27; pre-registration sections 2, 3.1-3.3, 3.6, 7, 16).

Fast tests use toy frames: the Fold record and its hashes, the parquet / JSON round trip (in a temporary
directory), the seeded greedy balance, the conflict-graph colouring and the batch eligibility re-check, the fast
eligibility checker against the full rule, the scoring filter and the inner-fold functions.

Corpus tests (marker ``slow``) read the files written by ``scripts/g19_build_folds.py`` and the MODEL rows:

1. every V5 fold (exact and batched, every variant): the hidden (metal state, system) cell has ZERO training rows
   (the brief section 27 test) and, under the component-aware rule, no row of the hidden state and no X(?) row of
   its element is left in the cell's system or in a system sharing a component with it;
2. V2: the held-out element (element level) or state + X(?) (state level) is absent from training;
3. V1: the held-out group is absent from training and no training row is a near-duplicate of a test row within
   the registered 0.005 log D tolerance;
4. no fold of any design scores a ``V6_TARGET_ROWS`` row (nor an X(?) or Sr(III) row);
5. fold hashes reproduce when the designs are rebuilt;
6. batched V5 folds: no two cells of a batch share a metal state or a system, every scored cell once per seed;
7. V5-PAIR: the stored pairs pass ``pair_isolation_check`` and both rows of a pair are hidden-scored in its fold;
8. the inner-fold functions never return a row outside the outer training set;
9. heavy-arm batched V5-P / V5-PAIR (seed 104729): every unit fold once, no conflicting units in a batch, a batch hides
   and scores exactly the union of its units, INDEX batch counts and listing match the files, batch pairs isolated;
10. the nested-certificate safeguard sample in INDEX.json reproduces from the fold files;
11. the section 7 item 6 re-colouring (max 4 cells per batch) on the corpus: no batch above 4 cells, every cell
    re-checked eligible, guard passes, and the default batches of seed 104729 reproduce the stored hashes.

Fast toy tests cover the keyed colouring against the original cell colouring, ``batch_units``, the cap, the batched
V5-P re-check (including a unit that must move to a singleton), batched V5-PAIR pair regeneration and the safeguard
draw rule.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gen19ct import paths
from gen19ct.chemistry import support_graph as SG
from gen19ct.data import leakage as L
from gen19ct.folds import cell_holdout as CH
from gen19ct.folds import io as FI
from gen19ct.folds import metal_holdout as MH
from gen19ct.folds import random_split as RS
from gen19ct.folds import registered as FR
from gen19ct.folds import source_holdout as SH

INDEX = paths.FOLDS_DIR / "INDEX.json"


# --------------------------------------------------------------------------------------------- #
# toy frames
# --------------------------------------------------------------------------------------------- #

def _toy_rows(state, system, n, pub="p1", element=None, acid="HNO3", group=None):
    el = element if element is not None else str(state).split("(")[0]
    return [{SG.METAL_COL: state, SG.ELEMENT_COL: el, SG.SYSTEM_COL: system, SG.PUB_COL: pub, "acid_primary": acid,
             FI.GROUP_COL: group or f"g_{pub}"} for _ in range(n)]


def _toy(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df[FI.ROW_ID] = [f"T:{i:05d}" for i in range(len(df))]
    return df


@pytest.fixture()
def cycle() -> pd.DataFrame:
    """A 4-cycle A-S1-B-S2-A: every edge is on a cycle, but hiding one edge makes its opposite edge a bridge."""
    return _toy(_toy_rows("Nd(III)", "S1", 3, "p1") + _toy_rows("Nd(III)", "S2", 3, "p2")
                + _toy_rows("Eu(III)", "S1", 3, "p3") + _toy_rows("Eu(III)", "S2", 3, "p4"))


def test_fold_record_hashes_and_validation() -> None:
    f = FI.make_fold(design="D", variant="v", scheme="exact", fold_id="f", half="S", seed=None,
                     hidden=["b", "a", "c", "a"], scored=["c"], unit_type="cell", units=["u"])
    assert f.hidden_row_ids == ("a", "b", "c") and f.unscored_row_ids == ("a", "b")
    assert f.fold_hash == FI.fold_hash(["c", "b", "a"], ["c"])
    assert f.fold_hash != FI.fold_hash(["a", "b", "c"], ["b"])           # the role enters the hash
    g = FI.make_fold(design="D", variant="v", scheme="exact", fold_id="g", half="C", seed=1, hidden=["d"], scored=[],
                     unit_type="cell", units=["w"])
    assert FI.design_hash([f, g]) == FI.design_hash([g, f])
    with pytest.raises(ValueError):
        FI.make_fold(design="D", variant="v", scheme="e", fold_id="x", half="S", seed=None, hidden=["a"], scored=["z"],
                     unit_type="cell", units=["u"])
    with pytest.raises(ValueError):
        FI.make_fold(design="D", variant="v", scheme="e", fold_id="x", half="Q", seed=None, hidden=["a"], scored=[],
                     unit_type="cell", units=["u"])
    with pytest.raises(AssertionError):
        FI.training_ids(g, ["a", "b"])                                     # hidden id outside the universe
    assert list(FI.training_ids(f, ["a", "b", "c", "e"])) == ["e"]


def test_write_read_round_trip(tmp_path: Path) -> None:
    folds = [FI.make_fold(design="V9", variant="toy", scheme="exact", fold_id=f"f{k}", half="S" if k else "C", seed=7,
                          hidden=[f"r{k}{j}" for j in range(4)], scored=[f"r{k}0", f"r{k}2"], unit_type="cell",
                          units=[f"u,{k}"], meta={"k": k}) for k in range(3)]
    pq, js, summary = FI.write_design(folds, tmp_path)
    back = FI.read_design(js)
    assert [f.fold_hash for f in back] == [f.fold_hash for f in folds]
    assert [f.meta for f in back] == [f.meta for f in folds]
    assert summary["design_hash"] == FI.design_hash(back) and summary["n_folds"] == 3
    assert summary["by_half"]["S"]["n_folds"] == 2 and summary["n_distinct_scored_rows"] == 6
    assert FI.assignment_sha256(back) == summary["assignment_sha256"]
    body = json.loads(js.read_text(encoding="utf-8"))
    body["folds"][0]["fold_hash"] = "0" * 64
    js.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(AssertionError):
        FI.read_design(js)


def test_greedy_balance_keeps_units_whole_and_depends_on_the_seed() -> None:
    w = {f"g{i:02d}": float(v) for i, v in enumerate([90, 50, 40, 30, 30, 20, 10, 10, 5, 5, 3, 2])}
    a = FI.greedy_balance(w, 3, FI.seed_rng(104729, "V1_grouped"))
    b = FI.greedy_balance(w, 3, FI.seed_rng(104729, "V1_grouped"))
    c = FI.greedy_balance(w, 3, FI.seed_rng(130363, "V1_grouped"))
    assert a == b and set(a) == set(w) and set(a.values()) == {0, 1, 2}
    assert a != c
    load = [sum(v for u, v in w.items() if a[u] == k) for k in range(3)]
    assert max(load) - min(load) <= max(w.values())


def test_v0_split_partitions_rows_and_never_scores_excluded_rows() -> None:
    df = _toy(_toy_rows("Nd(III)", "S1", 7) + _toy_rows(None, "S1", 3, element="Nd") + _toy_rows("Sr(III)", "S1", 2)
              + _toy_rows("Pr(III)", "S2", 5))
    v6 = df[SG.METAL_COL] == "Pr(III)"
    folds = RS.v0_folds(df, 104729, v6)
    RS.check_partition(folds, df)
    scored = {r for f in folds for r in f.scored_row_ids}
    assert scored == set(df.loc[df[SG.METAL_COL] == "Nd(III)", FI.ROW_ID])
    assert [len(f.hidden_row_ids) for f in folds] == [4, 4, 3, 3, 3]
    assert [f.fold_hash for f in folds] == [f.fold_hash for f in RS.v0_folds(df, 104729, v6)]


def test_conflict_colouring_never_batches_a_shared_state_or_system() -> None:
    cells = [(m, s) for m in ("La(III)", "Nd(III)", "Eu(III)", "Am(III)") for s in ("S1", "S2", "S3")]
    for seed in FI.DISCOVERY_SEEDS:
        classes = CH.greedy_colouring(cells, FI.seed_rng(seed, "V5_batch", 1))
        assert sorted(c for cl in classes for c in cl) == sorted(cells)
        for cl in classes:
            assert len({c[0] for c in cl}) == len(cl) and len({c[1] for c in cl}) == len(cl)


def test_eligibility_space_equals_the_full_rule_and_the_recheck_moves_failures_to_singletons(cycle) -> None:
    thr = CH.Thresholds(1, 1, 1)
    el = CH.eligible_cells(cycle, thr)
    space = CH.EligibilitySpace(cycle)
    for rec in el.itertuples(index=False):
        chk = space.check((rec.metal_state, rec.extractant_system_key), thr)
        assert chk["eligible"] == bool(rec.eligible) and chk["n_rows"] == rec.n_rows
    assert el["eligible"].all()
    a, b = ("Nd(III)", "S1"), ("Eu(III)", "S2")                       # no conflict: they colour together
    variant = CH.Variant("toy", thr)
    masks = CH.cell_masks(cycle, [a, b], variant)
    # with b hidden, a's edge becomes a bridge (and vice versa)
    assert not space.check(a, thr, keep=~masks[b])["eligible"]
    assert not space.check(b, thr, keep=~masks[a])["eligible"]
    batches, st = CH.batch_cells(space, [a, b], thr, masks, FI.seed_rng(1, "V5_batch", 1))
    assert sorted(batches) == [[b], [a]] or sorted(batches) == [[a], [b]]
    assert st["n_cells_moved_to_singleton"] == 2 and st["n_colour_classes"] == 1


def test_hidden_mask_is_state_level_and_component_aware() -> None:
    df = _toy(_toy_rows("Pu(VI)", "T", 4) + _toy_rows("Pu(IV)", "D|T", 3) + _toy_rows("Pu(VI)", "D|T", 2)
              + _toy_rows(None, "D|T", 2, element="Pu") + _toy_rows(None, "T", 1, element="Pu")
              + _toy_rows("Am(III)", "D|T", 5) + _toy_rows("Pu(VI)", "D", 2))
    cell = ("Pu(VI)", "T")
    hid = CH.hidden_mask(df, [cell], CH.VARIANTS["primary"])
    assert int(hid.sum()) == 4 + 2 + 2 + 1                              # cell, sharing-system state, both X(?) sets
    kept_pu = df[~hid & (df[SG.ELEMENT_COL] == "Pu").to_numpy()]
    # the other known state stays in the sharing system; the system without a shared component is untouched
    assert set(zip(kept_pu[SG.METAL_COL], kept_pu[SG.SYSTEM_COL])) == {("Pu(IV)", "D|T"), ("Pu(VI)", "D")}
    only = CH.hidden_mask(df, [cell], CH.VARIANTS["cell_only"])
    assert int(only.sum()) == 4 + 1


def test_scorable_mask_excludes_unknown_state_sr_iii_and_v6() -> None:
    df = _toy(_toy_rows("Nd(III)", "S", 2) + _toy_rows(None, "S", 1, element="Nd") + _toy_rows("Sr(III)", "S", 1)
              + _toy_rows("Pr(III)", "S", 1))
    v6 = pd.Series([False, False, False, False, True], index=df.index)
    assert FI.scorable_mask(df, v6).tolist() == [True, True, False, False, False]


def test_inner_functions_on_a_toy_frame_stay_inside_the_training_set() -> None:
    rows = []
    for j, s in enumerate(("S1", "S2", "S3", "S4")):
        for i, m in enumerate(("La(III)", "Nd(III)", "Eu(III)", "Am(III)")):
            rows += _toy_rows(m, s, 3, pub=f"p{(i + j) % 5}")
    rows += _toy_rows(None, "S1", 2, element="Nd", pub="p0")
    train = _toy(rows)
    ids = set(train[FI.ROW_ID])
    inner5 = CH.inner_cells_V5(train, CH.Thresholds(3, 1, 2), 104729, v6_ids=[], check=False)
    assert inner5 and all(set(f.hidden_row_ids) <= ids for f in inner5)
    assert all(f.scored_row_ids for f in inner5)
    inner1 = SH.inner_folds_V1(train, 104729, v6_ids=[], check=False)
    assert sorted(r for f in inner1 for r in f.hidden_row_ids) == sorted(ids)


# --------------------------------------------------------------------------------------------- #
# batching: the keyed colouring, max_cells_per_batch, batched V5-P and V5-PAIR (section 3.1 / 7 resolutions)
# --------------------------------------------------------------------------------------------- #

def _reference_colouring(cells, rng):
    """The section 3.1 colouring as first implemented (before the keyed generalisation), kept verbatim."""
    base = sorted(set(cells))
    nb = CH.conflict_neighbours(base)
    colour = {}
    for i in rng.permutation(len(base)):
        c = base[i]
        used = {colour[n] for n in nb[c] if n in colour}
        k = 0
        while k in used:
            k += 1
        colour[c] = k
    n_col = max(colour.values()) + 1 if colour else 0
    return [sorted(c for c in base if colour[c] == k) for k in range(n_col)]


def _grid(states, systems, n=3, groups=4):
    rows = []
    for i, m in enumerate(states):
        for j, s in enumerate(systems):
            rows += _toy_rows(m, s, n, pub=f"p{(i + 2 * j) % 7}", group=f"g{(i + j) % groups}")
    return _toy(rows)


def test_keyed_colouring_reproduces_the_registered_cell_colouring() -> None:
    rng0 = np.random.default_rng(5)
    states = [f"{e}(III)" for e in ("La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Am", "Cm")]
    for trial in range(12):
        cells = sorted({(states[a], f"S{b}") for a, b in rng0.integers(0, 9, size=(40, 2))})
        for seed in FI.DISCOVERY_SEEDS[:3]:
            want = _reference_colouring(cells, FI.seed_rng(seed, "V5_batch", 1))
            assert CH.greedy_colouring(cells, FI.seed_rng(seed, "V5_batch", 1)) == want
            assert CH.greedy_colouring(cells, FI.seed_rng(seed, "V5_batch", 1), None) == want
            # a cap at least as large as the largest class changes nothing
            big = max(len(c) for c in want)
            assert CH.greedy_colouring(cells, FI.seed_rng(seed, "V5_batch", 1), big) == want


def test_batch_units_moves_failures_to_singletons_and_respects_the_cap() -> None:
    units = [f"u{i}" for i in range(9)]
    keys = {u: frozenset({("k", int(u[1:]) % 3)}) for u in units}           # three cliques of three
    bad = {"u0", "u4"}
    failing = lambda cur: [u for u in cur if u in bad and len(cur) > 1]    # noqa: E731
    out, st = CH.batch_units(units, keys, failing, np.random.default_rng(1))
    assert sorted(u for b in out for u in b) == units
    assert [b for b in out if len(b) == 1 and b[0] in bad] == [["u0"], ["u4"]] and st["n_units_moved_to_singleton"] == 2
    assert all(len({keys[u] for u in b}) == len(b) for b in out)            # no two units of a batch conflict
    capped, st2 = CH.batch_units(units, keys, lambda cur: [], np.random.default_rng(1), max_size=2)
    assert max(len(b) for b in capped) == 2 and st2["largest_batch"] == 2
    with pytest.raises(AssertionError):                                    # a singleton must pass on its own
        CH.batch_units(units, keys, lambda cur: ["u1"] if "u1" in cur else [], np.random.default_rng(1))
    with pytest.raises(ValueError):
        CH.greedy_colouring_keyed(units, keys, np.random.default_rng(1), max_size=0)


def test_max_cells_per_batch_caps_v5_batches_keeps_the_recheck_and_leaves_default_batches_unchanged() -> None:
    states = ["La(III)", "Ce(III)", "Nd(III)", "Eu(III)", "Gd(III)", "Am(III)", "Cm(III)"]
    df = _grid(states, [f"S{j}" for j in range(7)])
    thr = CH.Thresholds(3, 1, 2)
    variant = CH.Variant("toy", thr)
    cells = sorted(set(zip(df[SG.METAL_COL], df[SG.SYSTEM_COL])))
    masks = CH.cell_masks(df, cells, variant)
    space = CH.EligibilitySpace(df)
    v6 = pd.Series(False, index=df.index)
    halves = {f"S{j}": "S" for j in range(7)}
    default, st0 = CH.v5_batched_folds(df, space, variant, cells, masks, v6, halves, 104729)
    none, _ = CH.v5_batched_folds(df, space, variant, cells, masks, v6, halves, 104729, max_cells_per_batch=None)
    assert [f.fold_hash for f in none] == [f.fold_hash for f in default] and {f.scheme for f in default} == {"batched"}
    assert max(len(f.meta["cells"]) for f in default) > 4                                   # the cap will bind
    capped, st4 = CH.v5_batched_folds(df, space, variant, cells, masks, v6, halves, 104729, max_cells_per_batch=4)
    assert {f.scheme for f in capped} == {"batched_max4"} and all(f.meta["max_cells_per_batch"] == 4 for f in capped)
    assert max(len(f.meta["cells"]) for f in capped) <= 4 and st4["S"]["largest_batch"] <= 4 < st0["S"]["largest_batch"]
    assert sorted(tuple(c) for f in capped for c in f.meta["cells"]) == cells                # every cell once
    for f in capped:
        cs = [tuple(c) for c in f.meta["cells"]]
        assert len({c[0] for c in cs}) == len(cs) and len({c[1] for c in cs}) == len(cs)
        hidden = df[FI.ROW_ID].isin(set(f.hidden_row_ids)).to_numpy()
        for c in cs:                                   # eligibility re-check with the batch's other cells hidden
            others = np.logical_or.reduce([masks[o] for o in cs if o != c] or [np.zeros(len(df), dtype=bool)])
            el = CH.eligible_cells(df[~others], thr)
            assert bool(el.loc[(el["metal_state"] == c[0]) & (el["extractant_system_key"] == c[1]), "eligible"].iloc[0])
            assert not (hidden & ~np.logical_or.reduce([masks[o] for o in cs])).any()
    # the failing re-check still moves cells to singletons under a cap
    cyc = _toy(_toy_rows("Nd(III)", "S1", 3, "p1") + _toy_rows("Nd(III)", "S2", 3, "p2")
               + _toy_rows("Eu(III)", "S1", 3, "p3") + _toy_rows("Eu(III)", "S2", 3, "p4"))
    t1 = CH.Thresholds(1, 1, 1)
    a, b = ("Nd(III)", "S1"), ("Eu(III)", "S2")
    m = CH.cell_masks(cyc, [a, b], CH.Variant("toy", t1))
    batches, st = CH.batch_cells(CH.EligibilitySpace(cyc), [a, b], t1, m, FI.seed_rng(1, "V5_batch", 1), max_cells_per_batch=4)
    assert sorted(batches) == [[b], [a]] or sorted(batches) == [[a], [b]]
    assert st["n_cells_moved_to_singleton"] == 2 and st["max_cells_per_batch"] == 4


def _v5p_units(df, cells, thr_min_other=1):
    variant = CH.VARIANTS["primary"]
    masks = CH.cell_masks(df, cells, variant)
    v6 = pd.Series(False, index=df.index)
    halves = {s: "S" for s in df[SG.SYSTEM_COL].unique()}
    folds, tab, _ = CH.v5p_folds(df, df[FI.GROUP_COL], CH.EligibilitySpace(df), cells, masks, v6, halves,
                                 min_other=thr_min_other)
    return folds, v6


def test_v5p_batched_unit_failing_the_recheck_moves_to_a_singleton() -> None:
    df = _toy(_toy_rows("La(III)", "S1", 3, "pA", group="gA") + _toy_rows("Nd(III)", "S1", 3, "pB", group="gB")
              + _toy_rows("La(III)", "S2", 3, "pC", group="gC") + _toy_rows("Nd(III)", "S2", 3, "pB", group="gB")
              + _toy_rows("Nd(III)", "S3", 3, "pD", group="gD") + _toy_rows("Eu(III)", "S3", 3, "pE", group="gE"))
    a, b = ("La(III)", "S1"), ("Nd(III)", "S2")
    units, v6 = _v5p_units(df, [a, b])
    assert len(units) == 2                                         # both units pass unbatched
    # together they do not conflict, but hiding group gB takes Nd(III) out of S1: a's system keeps no other state
    assert not CH.v5p_unit_keys(a, "gA") & CH.v5p_unit_keys(b, "gB")
    folds, st = CH.v5p_batched_folds(df, df[FI.GROUP_COL], CH.EligibilitySpace(df), units, v6,
                                     thr=CH.Thresholds(1, 1, 1), min_other=1)
    assert [f.meta["n_units"] for f in folds] == [1, 1] and st["S"]["n_units_moved_to_singleton"] == 2
    assert st["S"]["recheck_failures_by_rule"]["masking_rule_failed_with_batch_hidden"] >= 1
    by_unit = {u.fold_id: u for u in units}
    for f in folds:                                                # a singleton batch IS its unbatched fold
        assert f.fold_hash == by_unit[f.meta["unit_fold_ids"][0]].fold_hash


def test_v5p_batched_folds_are_conflict_free_unions_that_pass_the_recheck() -> None:
    states = ["La(III)", "Nd(III)", "Eu(III)", "Gd(III)", "Am(III)"]
    df = _grid(states, [f"S{j}" for j in range(5)], n=4, groups=6)
    cells = sorted(set(zip(df[SG.METAL_COL], df[SG.SYSTEM_COL])))
    units, v6 = _v5p_units(df, cells)
    thr = CH.Thresholds(1, 1, 1)
    folds, st = CH.v5p_batched_folds(df, df[FI.GROUP_COL], CH.EligibilitySpace(df), units, v6, thr=thr, min_other=1)
    by_unit = {u.fold_id: u for u in units}
    assert sorted(u for f in folds for u in f.meta["unit_fold_ids"]) == sorted(by_unit)
    assert any(f.meta["n_units"] > 1 for f in folds) and {f.scheme for f in folds} == {"batched"}
    ids = df[FI.ROW_ID]
    for f in folds:
        us = [by_unit[u] for u in f.meta["unit_fold_ids"]]
        keys = [CH.v5p_unit_keys(tuple(u.meta["cells"][0]), u.meta["publication_group"]) for u in us]
        assert all(not (keys[i] & keys[j]) for i in range(len(us)) for j in range(i + 1, len(us)))
        assert set(f.hidden_row_ids) == set().union(*(u.hidden_row_ids for u in us))
        assert set(f.scored_row_ids) == set().union(*(u.scored_row_ids for u in us))
        assert f.seed == 104729 and f.fold_id.startswith("s104729_S_b")
        if len(us) < 2:
            continue
        for u in us:                                               # brute-force re-check with the full rule
            c = tuple(u.meta["cells"][0])
            others = ids.isin(set().union(*(o.hidden_row_ids for o in us if o is not u))).to_numpy()
            el = CH.eligible_cells(df[~others], thr)
            assert bool(el.loc[(el["metal_state"] == c[0]) & (el["extractant_system_key"] == c[1]), "eligible"].iloc[0])
            train = df[~ids.isin(set(f.hidden_row_ids)).to_numpy()]
            assert train.loc[train[SG.SYSTEM_COL] == c[1], SG.METAL_COL].nunique() >= 1
            assert train.loc[train[SG.METAL_COL] == c[0], SG.SYSTEM_COL].nunique() >= 1
    ones, _ = CH.v5p_batched_folds(df, df[FI.GROUP_COL], CH.EligibilitySpace(df), units, v6, thr=thr, min_other=1,
                                   max_units_per_batch=1)
    assert sorted(f.fold_hash for f in ones) == sorted(u.fold_hash for u in units)


def test_v5pair_batched_folds_regenerate_the_union_of_unit_pairs_and_stay_isolated() -> None:
    states = ["La(III)", "Nd(III)", "Eu(III)", "Gd(III)", "Am(III)"]
    rows = []
    for m in states:
        for j in range(4):
            for c in range(3):
                rows += _toy_rows(m, f"S{j}", 1, pub=f"p{j}", group=f"g{j}")
    df = _toy(rows)
    ck = pd.Series([f"c{i % 3}" for i in range(len(df))], index=df.index)
    cells = sorted(set(zip(df[SG.METAL_COL], df[SG.SYSTEM_COL])))
    variant = CH.VARIANTS["primary"]
    masks = CH.cell_masks(df, cells, variant)
    v6 = pd.Series(False, index=df.index)
    halves = {f"S{j}": ("S" if j < 2 else "C") for j in range(4)}
    space = CH.EligibilitySpace(df)
    thr = CH.Thresholds(1, 1, 1)
    units, _, upairs, counts = CH.v5pair_folds(df, df[FI.GROUP_COL], ck, space, cells, masks, v6, halves, thr=thr)
    assert counts["n_eligible_cell_pairs"] == 40
    folds, ptab, st = CH.v5pair_batched_folds(df, df[FI.GROUP_COL], ck, space, units, upairs, masks, v6, halves, thr=thr)
    by_unit = {u.fold_id: u for u in units}
    assert sorted(u for f in folds for u in f.meta["unit_fold_ids"]) == sorted(by_unit)
    assert st["S"]["n_batches"] >= 10 and st["C"]["n_batches"] >= 10          # 10 cell pairs per system all conflict
    assert any(f.meta["n_units"] == 2 for f in folds) and max(f.meta["n_units"] for f in folds) <= 2
    assert sorted(zip(ptab["row_id_a"], ptab["row_id_b"])) == sorted(zip(upairs["row_id_a"], upairs["row_id_b"]))
    universe = df[FI.ROW_ID].tolist()
    for f in folds:
        us = [by_unit[u] for u in f.meta["unit_fold_ids"]]
        states_in = [tuple(c)[0] for u in us for c in u.meta["cells"]]
        assert len(set(states_in)) == len(states_in) and len({u.meta["cells"][0][1] for u in us}) == len(us)
        assert set(f.hidden_row_ids) == set().union(*(u.hidden_row_ids for u in us))
        p = ptab[ptab["fold_id"] == f.fold_id]
        CH.check_pair_isolation(f, p, universe)
        assert set(p["unit_fold_id"]) == set(f.meta["unit_fold_ids"])
    cross = pd.DataFrame({"row_id_a": [folds[0].scored_row_ids[0]],
                          "row_id_b": [next(r for r in universe if r not in set(folds[0].hidden_row_ids))]})
    with pytest.raises(AssertionError):
        CH.check_pair_isolation(folds[0], cross, universe)
    ones, _, _ = CH.v5pair_batched_folds(df, df[FI.GROUP_COL], ck, space, units, upairs, masks, v6, halves, thr=thr,
                                         max_units_per_batch=1)
    assert sorted(f.fold_hash for f in ones) == sorted(u.fold_hash for u in units)


# --------------------------------------------------------------------------------------------- #
# corpus tests on the written fold files
# --------------------------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def corpus():
    if not INDEX.exists():
        pytest.skip("folds/INDEX.json not built (run scripts/g19_build_folds.py)")
    from gen19ct.data import load
    model = load.load_model_rows()
    frame = FI.slim_frame(model.assign(**{FI.GROUP_COL: FI.publication_groups(model)}))
    v6 = FR.v6_target_mask(model, FR.v6_system_set(model)).reindex(frame.index)
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    return {"model": model, "frame": frame, "v6": v6, "index": index}


def _folds(stem: str) -> list[FI.Fold]:
    return FI.read_design(stem, paths.FOLDS_DIR)


def _hidden_bool(frame: pd.DataFrame, fold: FI.Fold) -> np.ndarray:
    return frame[FI.ROW_ID].isin(set(fold.hidden_row_ids)).to_numpy()


def _stems(index: dict, design: str) -> list[str]:
    return sorted(k for k, v in index["designs"].items() if v["design"] == design)


@pytest.mark.slow
def test_1_hidden_v5_cells_have_zero_training_rows(corpus) -> None:
    fr = corpus["frame"]
    st, el, sy = (fr[c].to_numpy(dtype=object) for c in (SG.METAL_COL, SG.ELEMENT_COL, SG.SYSTEM_COL))
    unknown = fr[SG.METAL_COL].isna().to_numpy()
    keys = sorted(set(sy))
    pmap = FR.parent_component_map()
    n_cells = 0
    for stem in _stems(corpus["index"], "V5") + _stems(corpus["index"], "V5P") + _stems(corpus["index"], "V5PAIR"):
        for f in _folds(stem):
            train = ~_hidden_bool(fr, f)
            cmap = pmap if f.meta.get("parent_structure") else None
            for m, s in (tuple(c) for c in f.meta["cells"]):
                e = m.split("(")[0]
                # brief section 27: the hidden metal x extractant cell is absent from the training set
                assert not (train & (st == m) & (sy == s)).any(), (stem, f.fold_id, m, s[:40])
                assert not (train & unknown & (el == e) & (sy == s)).any(), (stem, f.fold_id, "X(?) in own system")
                if f.meta["component_aware"]:
                    sharing = np.isin(sy, list(SG.systems_sharing_component(keys, s, cmap)))
                    assert not (train & sharing & (st == m)).any(), (stem, f.fold_id, "state in sharing system")
                    assert not (train & sharing & unknown & (el == e)).any(), (stem, f.fold_id, "X(?) in sharing system")
                n_cells += 1
    assert n_cells > 0
    prim = corpus["index"]["designs"]["V5__primary__exact"]
    assert prim["n_folds"] == prim["n_eligible_cells"] == 224 and prim["n_carved_out_cells"] == 14
    assert prim["scored_cells_by_half"] == {"S": 105, "C": 105}


@pytest.mark.slow
def test_2_v2_element_absent_from_training(corpus) -> None:
    fr = corpus["frame"]
    for f in _folds("V2__element__exact"):
        train = fr[~_hidden_bool(fr, f)]
        assert not (train[SG.ELEMENT_COL] == f.meta["element"]).any(), f.fold_id
    for f in _folds("V2__state__exact"):
        train = fr[~_hidden_bool(fr, f)]
        assert not (train[SG.METAL_COL] == f.units[0]).any(), f.fold_id
        assert not (train[SG.METAL_COL].isna() & (train[SG.ELEMENT_COL] == f.meta["element"])).any(), f.fold_id
    assert len(_folds("V2__element__exact")) == 23


@pytest.mark.slow
def test_3_v1_group_absent_and_no_near_duplicate_in_training(corpus) -> None:
    fr = corpus["frame"]
    keys = L.near_duplicate_key(fr).to_numpy(dtype=object)
    y = pd.to_numeric(fr["log_D"], errors="coerce").to_numpy(dtype=float)
    grp = fr[FI.GROUP_COL].to_numpy(dtype=object)
    folds = _folds("V1__copy__exact")
    assert len(folds) == 104
    for f in folds:
        hid = _hidden_bool(fr, f)
        groups = set(f.meta["groups"])
        assert not np.isin(grp[~hid], list(groups)).any(), f.fold_id
        assert set(grp[hid]) == groups
        test_keys = set(keys[hid])
        cand = ~hid & np.isin(keys, list(test_keys))
        if cand.any():
            by = pd.Series(y[hid]).groupby(keys[hid]).agg(list).to_dict()
            for k, v in zip(keys[cand], y[cand]):
                assert np.min(np.abs(np.asarray(by[k]) - v)) > FI.NEAR_DUP_VALUE_TOL + 1e-12, (f.fold_id, k[:60])


@pytest.mark.slow
def test_4_no_v6_target_row_is_scored_in_any_fold(corpus) -> None:
    fr, v6 = corpus["frame"], corpus["v6"]
    v6_ids = set(fr.loc[v6.to_numpy(), FI.ROW_ID])
    assert len(v6_ids) == 580 and v6_ids == set(FI.registered_v6_ids())
    bad_state = set(fr.loc[fr[SG.METAL_COL].isna() | fr[SG.METAL_COL].isin(FI.UNSCORED_STATES), FI.ROW_ID])
    n = 0
    for pq in sorted(paths.FOLDS_DIR.glob("*__*__*.parquet")):
        if pq.name.endswith("__pairs.parquet"):
            continue
        a = pd.read_parquet(pq, columns=["row_id", "role"])
        sc = set(a.loc[a["role"] == "hidden_scored", "row_id"])
        assert not sc & v6_ids, pq.name
        assert not sc & bad_state, pq.name
        FR.assert_not_scored(fr.index[fr[FI.ROW_ID].isin(sc).to_numpy()], v6, pq.name)
        n += 1
    assert n == len(corpus["index"]["designs"])


@pytest.mark.slow
def test_5_fold_hashes_reproduce_on_rebuild(corpus) -> None:
    import sys
    sys.path.insert(0, str(paths.G19_ROOT / "scripts"))
    import g19_build_folds as B

    fr, v6, idx = corpus["frame"], corpus["v6"], corpus["index"]["designs"]

    def same(stem: str, folds: list[FI.Fold]) -> None:
        body = json.loads((paths.FOLDS_DIR / f"{stem}.json").read_text(encoding="utf-8"))
        assert [r["fold_hash"] for r in body["folds"]] == [f.fold_hash for f in folds], stem
        assert idx[stem]["design_hash"] == FI.design_hash(folds), stem

    same("V0__rows__random5", [f for s in FI.DISCOVERY_SEEDS for f in RS.v0_folds(fr, s, v6)])
    same("V1__copy__exact", SH.v1_exact_folds(fr, v6))
    same("V1__copy__grouped10", [f for s in FI.DISCOVERY_SEEDS for f in SH.v1_grouped_folds(fr, v6, s)])
    same("V2__element__exact", MH.v2_folds(fr, v6))
    same("V2__state__exact", MH.v2_folds(fr, v6, element_level=False))
    halves = FI.registered_halves("V5_system")
    cells_tab = pd.read_csv(B.CELLS_CSV)
    book = B.Book(fr, v6)
    var = CH.VARIANTS["primary"]
    cells, scored, _ = B._variant_cells(book, cells_tab, var)
    masks = CH.cell_masks(fr, cells, var)
    same("V5__primary__exact", CH.v5_exact_folds(fr, var, cells, masks, v6, halves))
    space = CH.EligibilitySpace(fr)
    same("V5__primary__batched",
         [f for s in FI.DISCOVERY_SEEDS for f in CH.v5_batched_folds(fr, space, var, scored, masks, v6, halves, s)[0]])
    _, base_scored, _ = B._variant_cells(book, cells_tab, CH.Variant("v5p_base", CH.V5P_BASE))
    base_masks = CH.cell_masks(fr, base_scored, var)
    same("V5P__base__cell_x_group", CH.v5p_folds(fr, fr[FI.GROUP_COL], space, base_scored, base_masks, v6, halves)[0])
    from gen19ct.data import normalize as N
    ck = N.condition_key(corpus["model"].loc[fr.index]).reindex(fr.index)
    pair_folds, _, ptab, _ = CH.v5pair_folds(fr, fr[FI.GROUP_COL], ck, space, cells, masks, v6, halves)
    same("V5PAIR__primary__cell_pair", pair_folds)
    stored = pd.read_parquet(paths.FOLDS_DIR / "V5PAIR__primary__pairs.parquet")
    pd.testing.assert_frame_equal(stored.reset_index(drop=True), ptab.reset_index(drop=True))
    # the heavy-arm batches of V5-P and V5-PAIR (seed 104729) rebuild from the rebuilt unit folds
    v5p_units = CH.v5p_folds(fr, fr[FI.GROUP_COL], space, base_scored, base_masks, v6, halves)[0]
    same("V5P__base__batched", CH.v5p_batched_folds(fr, fr[FI.GROUP_COL], space, v5p_units, v6)[0])
    bpair, bptab, _ = CH.v5pair_batched_folds(fr, fr[FI.GROUP_COL], ck, space, pair_folds, ptab, masks, v6, halves)
    same("V5PAIR__primary__batched", bpair)
    stored_b = pd.read_parquet(paths.FOLDS_DIR / "V5PAIR__primary__batched__pairs.parquet")
    pd.testing.assert_frame_equal(stored_b.reset_index(drop=True), bptab.reset_index(drop=True))
    # every stored design file re-derives its fold and design hashes from the parquet
    for stem in idx:
        FI.read_design(stem, paths.FOLDS_DIR)


@pytest.mark.slow
def test_6_batched_v5_folds_have_no_conflicting_cells(corpus) -> None:
    index = corpus["index"]
    for stem in (k for k in _stems(index, "V5") if k.endswith("__batched")):
        folds = _folds(stem)
        exact = _folds(stem.replace("__batched", "__exact"))
        scored_cells = sorted(tuple(f.meta["cells"][0]) for f in exact if not f.meta["carved_out"])
        for seed in FI.DISCOVERY_SEEDS:
            got = []
            for f in (x for x in folds if x.seed == seed):
                cs = [tuple(c) for c in f.meta["cells"]]
                assert len({c[0] for c in cs}) == len(cs), (stem, f.fold_id, "shared metal state")
                assert len({c[1] for c in cs}) == len(cs), (stem, f.fold_id, "shared system")
                assert f.half in ("S", "C")
                got += cs
            assert sorted(got) == scored_cells, (stem, seed)
        placeholder = index["placeholders"]["V5_batches_per_discovery_seed"][stem.split("__")[1]]
        for seed in FI.DISCOVERY_SEEDS:
            assert placeholder[str(seed)]["total"] == sum(1 for f in folds if f.seed == seed)


@pytest.mark.slow
def test_7_v5pair_pairs_are_isolated_inside_their_fold(corpus) -> None:
    fr = corpus["frame"]
    pairs = pd.read_parquet(paths.FOLDS_DIR / "V5PAIR__primary__pairs.parquet")
    folds = {f.fold_id: f for f in _folds("V5PAIR__primary__cell_pair")}
    assert set(pairs["fold_id"]) == set(folds)
    universe = fr[FI.ROW_ID].tolist()
    grp = dict(zip(fr[FI.ROW_ID], fr[FI.GROUP_COL]))
    sysk = dict(zip(fr[FI.ROW_ID], fr[SG.SYSTEM_COL]))
    for fid, p in pairs.groupby("fold_id"):
        f = folds[fid]
        CH.check_pair_isolation(f, p, universe)
        scored = set(f.scored_row_ids)
        assert p["row_id_a"].isin(scored).all() and p["row_id_b"].isin(scored).all()
        assert (p["metal_state_a"] != p["metal_state_b"]).all()
        assert all(grp[a] == grp[b] and sysk[a] == sysk[b] for a, b in zip(p["row_id_a"], p["row_id_b"]))
        assert len(p) == f.meta["n_comparable_row_pairs"] >= CH.V5PAIR_MIN_COMPARABLE_PAIRS
    # a pair that crosses the train / test boundary is caught
    f = next(iter(folds.values()))
    train_id = next(r for r in universe if r not in set(f.hidden_row_ids))
    crossing = pd.DataFrame({"row_id_a": [f.scored_row_ids[0]], "row_id_b": [train_id]})
    with pytest.raises(AssertionError):
        CH.check_pair_isolation(f, crossing, universe)


@pytest.mark.slow
def test_8_inner_fold_functions_stay_inside_the_outer_training_set(corpus) -> None:
    fr = corpus["frame"]
    outer5 = next(f for f in _folds("V5__primary__exact") if not f.meta["carved_out"])
    outer1 = _folds("V1__copy__exact")[0]
    outer2 = next(f for f in _folds("V2__element__exact") if f.units[0] == "Nd(III)")
    for outer, build in ((outer5, lambda t: CH.inner_cells_V5(t, CH.PRIMARY, FI.DISCOVERY_SEEDS[0])),
                         (outer1, lambda t: SH.inner_folds_V1(t, FI.DISCOVERY_SEEDS[0])),
                         (outer2, lambda t: MH.inner_metals_V2(t, FI.DISCOVERY_SEEDS[0]))):
        train = FI.training_frame(outer, fr)
        tids = set(train[FI.ROW_ID])
        inner = build(train)                                             # check=True runs the guard inside train
        assert inner, outer.fold_id
        for f in inner:
            assert set(f.hidden_row_ids) <= tids, (outer.fold_id, f.fold_id)
            assert not set(f.hidden_row_ids) & set(outer.hidden_row_ids)
            assert not set(f.scored_row_ids) & set(FI.registered_v6_ids())
    inner5 = CH.inner_cells_V5(FI.training_frame(outer5, fr), CH.PRIMARY, FI.DISCOVERY_SEEDS[0])
    per_inner = pd.Series([f.meta["inner_fold"] for f in inner5 for _ in f.meta["cells"]]).value_counts()
    assert (per_inner <= CH.INNER_MAX_CELLS).all()
    v6 = corpus["v6"].to_numpy(dtype=bool)
    for f in inner5:                                                     # no inner cell touches V6_TARGET_ROWS
        for c in f.meta["cells"]:
            assert not (CH.cell_rows_mask(fr, tuple(c)) & v6).any(), (f.fold_id, c[0])


@pytest.mark.slow
def test_9_heavy_arm_batched_v5p_v5pair_folds_are_unions_of_their_units(corpus) -> None:
    index, fr = corpus["index"], corpus["frame"]
    universe = fr[FI.ROW_ID].tolist()
    for stem, unit_stem in (("V5P__base__batched", "V5P__base__cell_x_group"),
                            ("V5PAIR__primary__batched", "V5PAIR__primary__cell_pair")):
        entry = index["designs"][stem]
        folds, units = _folds(stem), {u.fold_id: u for u in _folds(unit_stem)}
        assert entry["unit_design_hash"] == index["designs"][unit_stem]["design_hash"]
        assert sorted(u for f in folds for u in f.meta["unit_fold_ids"]) == sorted(units)       # every unit once
        assert {f.seed for f in folds} == {104729} and entry["guard"]["all_ok"]
        counts = entry["batches_by_half"]
        assert counts["total"] == len(folds) == sum(counts[h]["n_batches"] for h in ("S", "C"))
        for h in ("S", "C"):
            assert counts[h]["n_batches"] == sum(1 for f in folds if f.half == h) == entry["by_half"][h]["n_folds"]
            assert counts[h]["n_units"] == sum(1 for u in units.values() if u.half == h)
        assert [(r["fold_id"], r["fold_hash"]) for r in entry["batch_listing"]] == [(f.fold_id, f.fold_hash) for f in folds]
        assert index["placeholders"][stem.split("__")[0] + "_heavy_arm_batches"] == counts
        for f in folds:
            us = [units[u] for u in f.meta["unit_fold_ids"]]
            assert all(u.half == f.half for u in us)
            assert set(f.hidden_row_ids) == set().union(*(u.hidden_row_ids for u in us)), (stem, f.fold_id)
            assert set(f.scored_row_ids) == set().union(*(u.scored_row_ids for u in us)), (stem, f.fold_id)
            cells = [tuple(c) for u in us for c in u.meta["cells"]]
            assert len({c[0] for c in cells}) == len(cells), (stem, f.fold_id, "shared metal state")
            assert len({c[1] for c in {tuple(u.meta["cells"][0]) for u in us}}) == len(us), (stem, f.fold_id, "system")
            if stem.startswith("V5P__"):
                assert len({u.meta["publication_group"] for u in us}) == len(us), (stem, f.fold_id, "group")
    pairs = pd.read_parquet(paths.FOLDS_DIR / "V5PAIR__primary__batched__pairs.parquet")
    unit_pairs = pd.read_parquet(paths.FOLDS_DIR / "V5PAIR__primary__pairs.parquet")
    assert sorted(zip(pairs["unit_fold_id"], pairs["row_id_a"], pairs["row_id_b"])) == \
        sorted(zip(unit_pairs["fold_id"], unit_pairs["row_id_a"], unit_pairs["row_id_b"]))
    folds = {f.fold_id: f for f in _folds("V5PAIR__primary__batched")}
    assert set(pairs["fold_id"]) == set(folds)
    for fid, p in pairs.groupby("fold_id"):
        CH.check_pair_isolation(folds[fid], p, universe)


@pytest.mark.slow
def test_10_nested_certificate_safeguard_sample_reproduces(corpus) -> None:
    from gen19ct.folds import safeguard as SF
    stored = corpus["index"][SF.KEY]
    again = SF.registered_safeguard_samples(paths.FOLDS_DIR)
    assert json.loads(json.dumps(again)) == stored
    d = stored["designs"]
    assert set(d) == set(SF.REGISTERED_POPULATIONS)
    # task X VR-03 / VR-04: every design fitted with the V5 inner design (certificates) is sampled; V1 / V2 are vacuous
    certified = {k for k, v in d.items() if v["certificate_in_inner_splits"]}
    assert certified == {"V5__primary__exact", "V5__primary__batched", "V5__cell_only__batched",
                         "V5__parent_structure__batched", "V5P__base__cell_x_group", "V5P__base__batched",
                         "V5PAIR__primary__cell_pair", "V5PAIR__primary__batched"}
    for k in certified:
        assert d[k]["n_drawn"] == 20 and d[k]["stratified"] and not d[k]["vacuous_under_current_code"], k
        assert d[k]["filters"]["half"] == "S", k
    assert d["V5__primary__exact"]["filters"]["require_scored"] and d["V5P__base__batched"]["filters"]["seed"] == 104729
    assert {k for k, v in d.items() if v["vacuous_under_current_code"]} == {"V1__copy__grouped10", "V2__element__exact"}
    assert "InnerCellCalibration" in corpus["index"]["readings"]["nested_certificate_safeguard"]
    v5 = d["V5__primary__batched"]
    assert v5["stratified"] and v5["n_drawn"] == 20 and v5["population_size"] == \
        corpus["index"]["placeholders"]["V5_batches_per_discovery_seed"]["primary"]["104729"]["S"]
    assert d["V1__copy__grouped10"]["fold_ids"] == [f"s104729_f{k}" for k in range(10)]
    assert len(d["V2__element__exact"]["fold_ids"]) == 12 and d["V2__element__exact"]["all_taken"]


@pytest.mark.slow
def test_11_max_cells_per_batch_recolouring_on_the_corpus(corpus) -> None:
    fr, v6 = corpus["frame"], corpus["v6"]
    halves = FI.registered_halves("V5_system")
    var = CH.VARIANTS["primary"]
    scored = sorted(tuple(f.meta["cells"][0]) for f in _folds("V5__primary__exact") if not f.meta["carved_out"])
    masks = CH.cell_masks(fr, scored, var)
    space = CH.EligibilitySpace(fr)
    seed = FI.DISCOVERY_SEEDS[0]
    default, _ = CH.v5_batched_folds(fr, space, var, scored, masks, v6, halves, seed)
    stored = {f.fold_id: f.fold_hash for f in _folds("V5__primary__batched") if f.seed == seed}
    assert {f.fold_id: f.fold_hash for f in default} == stored                   # default batches unchanged
    capped, st = CH.v5_batched_folds(fr, space, var, scored, masks, v6, halves, seed, max_cells_per_batch=4)
    assert max(len(f.meta["cells"]) for f in capped) <= 4
    assert sorted(tuple(c) for f in capped for c in f.meta["cells"]) == scored
    for f in capped:
        cs = [tuple(c) for c in f.meta["cells"]]
        assert len({c[0] for c in cs}) == len(cs) and len({c[1] for c in cs}) == len(cs)
        total = np.sum([masks[c].astype(np.int32) for c in cs], axis=0)
        for c in cs:
            assert space.check(c, var.thresholds, keep=(total - masks[c]) == 0)["eligible"], (f.fold_id, c[0])
        CH.guard_v5(f, fr)


def test_safeguard_draw_rule_allocation_and_strata() -> None:
    from gen19ct.folds import safeguard as SF
    pop = [f"f{i:02d}" for i in range(30)]
    rec = SF.draw_sample(pop)
    want = sorted(pop[i] for i in np.random.default_rng(19).choice(30, size=20, replace=False))
    assert rec["fold_ids"] == want and rec["n_drawn"] == 20 and not rec["all_taken"]
    assert SF.draw_sample(pop[:12])["fold_ids"] == pop[:12] and SF.draw_sample(pop[:12])["all_taken"]
    strata = {f: ("Ln" if i < 25 else "An" if i < 35 else "other") for i, f in enumerate(f"h{i:02d}" for i in range(37))}
    rec = SF.draw_sample(strata, strata)
    assert rec["allocation"] == {"Ln": 14, "An": 5, "other": 1} and rec["strata_population"] == {"Ln": 25, "An": 10, "other": 2}
    rng = np.random.default_rng(19)
    ids = sorted(strata)
    want = [ids[:25][i] for i in rng.choice(25, 14, replace=False)] + [ids[25:35][i] for i in rng.choice(10, 5, replace=False)] \
        + [ids[35:][i] for i in rng.choice(2, 1, replace=False)]
    assert rec["fold_ids"] == sorted(want)
    assert SF.allocate({"Ln": 50, "An": 1, "other": 0}, 5) == {"Ln": 4, "An": 1, "other": 0}   # every stratum is checked
    assert SF.metal_category("Nd(III)") == "Ln" and SF.metal_category("Am(III)") == "An" and SF.metal_category("Zr(IV)") == "other"
    mk = lambda fid, states: FI.make_fold(design="V5", variant="t", scheme="batched", fold_id=fid, half="S",   # noqa: E731
                                          seed=104729, hidden=[fid], scored=[], unit_type="cell", units=[fid],
                                          meta={"cells": [[s, "S1"] for s in states]})
    folds = [mk("a", ["La(III)", "Am(III)"]), mk("b", ["Nd(III)", "Eu(III)"]), mk("c", ["Cm(III)", "Zr(IV)"])]
    # a ties Ln / An -> the category with fewer cells in the population (An 2 < Ln 3); c ties An / other -> other (1)
    assert SF.fold_strata(folds) == {"a": "An", "b": "Ln", "c": "other"}
    s = SF.safeguard_sample(folds, seed_filter=104729, half="S")
    assert s["fold_ids"] == ["a", "b", "c"] and s["stratified"] and set(s["fold_hashes"]) == {"a", "b", "c"}
    assert not SF.safeguard_sample(folds, stratify=False)["stratified"]
    # require_scored drops folds without a scored row (carved-out exact V5 folds are never fitted)
    scored = folds[:2] + [FI.make_fold(design="V5", variant="t", scheme="exact", fold_id="d", half="S", seed=104729,
                                       hidden=["d", "d2"], scored=["d"], unit_type="cell", units=["d"],
                                       meta={"cells": [["Gd(III)", "S1"]]})]
    assert SF.safeguard_sample(scored, require_scored=True)["fold_ids"] == ["d"]


def test_safeguard_covers_exactly_the_inner_splitters_that_build_certificates() -> None:
    """Task X, findings VR-03 / VR-04: only the V5 inner design builds a certificate, so the V1 and V2 samples are
    labelled vacuous and every certified population names that splitter."""
    import inspect

    from gen19ct.folds import safeguard as SF
    from gen19ct.models import interface as I

    assert "certificate=" in inspect.getsource(I.InnerCellCalibration.splits)
    for cls in (I.GroupKFoldCalibration, I.InnerMetalCalibration):
        assert "certificate" not in inspect.getsource(cls.splits), cls.__name__
    for stem, spec in SF.REGISTERED_POPULATIONS.items():
        assert spec["certificate"] == spec["inner_splitter"].startswith("InnerCellCalibration"), stem
        assert spec["certificate"] == (not stem.startswith(("V1__", "V2__"))), stem


def test_fold_fields_round_trip_and_copy_crossings(tmp_path: Path) -> None:
    folds = [FI.make_fold(design="V9", variant="toy", scheme="exact", fold_id=f"f{k}", half="S", seed=None,
                          hidden=[f"r{k}", f"q{k}"], scored=[f"r{k}"], unit_type="cell", units=[f"u{k}"]) for k in range(2)]
    tau = {"f0": {"support_tau": {"tau_in": 0.1, "tau_ext": 0.5, "tau_max": float("nan"), "n_tau_rows": 3}}}
    pq, js, summary = FI.write_design(folds, tmp_path, fold_fields=tau)
    back = FI.read_fold_fields(js, "support_tau")
    assert back["f0"] == {"tau_in": 0.1, "tau_ext": 0.5, "tau_max": None, "n_tau_rows": 3} and back["f1"] is None
    assert FI.design_hash(FI.read_design(js)) == summary["design_hash"]            # fields never enter the hash
    with pytest.raises(ValueError):
        FI.fold_record(folds[0], {"fold_hash": "x"})
    pairs = pd.DataFrame({"id_a": ["r0", "r1", "q1"], "id_b": ["x", "q1", "r0"], "kind": ["STATE_WILDCARD"] * 3,
                          "strict_copy": [True, True, False], "same_publication": [False, True, False]})
    cr = FI.copy_crossings(folds, pairs)
    # f0: r0's partners x (training) and q1 (training in f0) cross; f1: r1's partner q1 is hidden with it -> no crossing
    assert sorted(zip(cr["fold_id"], cr["row_id"], cr["partner_id"])) == [("f0", "r0", "q1"), ("f0", "r0", "x")]
    assert FI.copy_crossings(folds, pairs.iloc[0:0]).empty
