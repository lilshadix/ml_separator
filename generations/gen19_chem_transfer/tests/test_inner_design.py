"""The addendum-1 V5 inner design of the learned arms (``gen19ct/models/inner_design.py``).

``SimultaneousInnerCells``: the section 7 inner cells of ``InnerCellCalibration`` (same functions, same streams), every
cell of an inner fold hidden together in ONE split, the eligibility re-check with all of them hidden, a failing cell
hidden but not scored; the fold-mean unit-macro selection score; the cross-fit plan; the isolation guard on every split.

Fast tests run on a synthetic mini-corpus with real metal-state labels, component-sharing system keys (``A|B`` shares a
component with ``A`` and with ``B``), X(?) alias rows, an Sr(III) cell, a V6-like cell and an excluded row.  Nothing is
fitted on a registered fold and nothing is scored.  The ``slow`` test reads the outer-TRAINING rows of ONE registered
V5-primary selection-half cell fold (seed 104729) and checks the design's splits with ``fold_isolation_check``; no model
is fitted and no test row is predicted.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from gen19ct import paths
from gen19ct.chemistry import support_graph as SG
from gen19ct.folds import cell_holdout as CH
from gen19ct.folds import io as FI
from gen19ct.folds import registered as FR
from gen19ct.models import inner_design as ID
from gen19ct.models import interface as I

SEED = 104729
STATES = ("La(III)", "Ce(III)", "Pr(III)", "Nd(III)", "Sm(III)", "Eu(III)", "Gd(III)", "Am(III)")
#: ``SA|SB`` shares a component with ``SA`` and ``SB``; ``SC|SE`` with ``SC`` and ``SE``
SYSTEMS = ("SA", "SB", "SA|SB", "SC", "SD", "SE", "SC|SE")
THR = CH.Thresholds(6, 1, 2)
#: the two cells of a metal with exactly m + 1 systems, given one private majority group so they land in one fold
PAIR_STATE = "Cm(III)"
PAIR_SYSTEMS = ("SC", "SD", "SE")
PAIR_GROUP = "g_pair"


def _rec(k: int, state: str | None, system: str, group: str, *, element: str | None = None, acid: str = "HNO3",
         y: float = 0.0) -> dict:
    el = element if state is None else SG.metal_properties(state)["symbol"]
    ox = np.nan if state is None else float(state.split("(")[1].rstrip(")").count("I")) if "(I" in state else np.nan
    return {FI.ROW_ID: f"T:{k:05d}", SG.METAL_COL: state, SG.ELEMENT_COL: el, "g19_ox": ox, SG.SYSTEM_COL: system,
            SG.PUB_COL: f"pub_{group}", FI.GROUP_COL: group, I.PUB_GROUP_COL: group, "acid_primary": acid,
            SG.ACID_ANION_COL: "nitrate" if acid == "HNO3" else "chloride", SG.LOG_ACID_COL: 0.1 * (k % 7) - 0.3,
            SG.LOG_EXT_COL: -1.0 - 0.05 * (k % 5), I.TARGET_COL: y, "duplicate_group_id": None,
            "solvent_key": "dodecane:1", "acid_concentration_M": 0.5 + 0.001 * k,
            "extractant_primary_concentration_M": 0.1 + 0.0001 * k, "temperature_C": 25.0}


def mini_corpus(seed: int = 3, *, mixed_acid: bool = False, one_group: bool = False) -> pd.DataFrame:
    """8 states x 7 systems (a few cells missing, 6-12 rows each, two publication groups per cell), Cm(III) in exactly
    three systems with one private majority group, Nd(?) alias rows in ``SA`` and ``SA|SB``, an Sr(III) cell.
    ``mixed_acid``: half the rows of every even cell are HCl.  ``one_group``: every cell's majority group is ``g00``."""
    rng = np.random.default_rng(seed)
    recs, k = [], 0

    def add(state, system, n, group, **kw):
        nonlocal k
        for _ in range(n):
            recs.append(_rec(k, state, system, group, y=float(rng.normal()), **kw))
            k += 1
    for i, st in enumerate(STATES):
        for j, sy in enumerate(SYSTEMS):
            if (i + 3 * j) % 11 == 0:
                continue
            n = int(rng.integers(6, 13))
            major = "g00" if one_group else f"g{(i * 2 + j) % 9:02d}"
            minor = f"g{(i + 5 * j) % 9 + 20:02d}"
            acid_major = "HCl" if (mixed_acid and (i + j) % 2 == 0) else "HNO3"
            add(st, sy, n - n // 3, major, acid=acid_major)
            add(st, sy, n // 3, minor)
    for sy in PAIR_SYSTEMS:
        add(PAIR_STATE, sy, 8, "g00" if one_group else PAIR_GROUP)
        add(PAIR_STATE, sy, 2, "g25")
    add(None, "SA", 5, "g03", element="Nd")
    add(None, "SA|SB", 4, "g04", element="Nd")
    add("Sr(III)", "SB", 10, "g05")
    df = pd.DataFrame(recs)
    df.index = pd.Index([f"L{i}" for i in range(len(df))], dtype=object)
    return df


def v6_of(df: pd.DataFrame) -> pd.Series:
    return pd.Series((df[SG.METAL_COL] == "Pr(III)") & (df[SG.SYSTEM_COL] == "SC"), index=df.index)


def excl_of(df: pd.DataFrame) -> pd.Series:
    e = pd.Series(False, index=df.index)
    e.loc[df.index[(df[SG.METAL_COL] == "Eu(III)") & (df[SG.SYSTEM_COL] == "SD")][:1]] = True
    return e


def ctx(table: I.RowTable, df: pd.DataFrame, seed: int = SEED, **kw) -> I.FitContext:
    kw.setdefault("v6_mask", v6_of(df))
    kw.setdefault("exclude_from_scoring", excl_of(df))
    kw.setdefault("isolation_check", lambda tr, te: {"ok": True})
    return I.FitContext(table=table, seed=seed, **kw)


@pytest.fixture(scope="module")
def corpus():
    df = mini_corpus()
    table = I.RowTable(df)
    mask = np.ones(table.n, dtype=bool)
    mask[table.mask_of(df.index[(df[SG.METAL_COL] == "La(III)") & (df[SG.SYSTEM_COL] == "SB")])] = False   # an outer fold
    design = ID.SimultaneousInnerCells(THR.k, THR.p, THR.m, 3, 30)
    splits = design.splits(table, mask, ctx(table, df))
    return {"df": df, "table": table, "mask": mask, "design": design, "splits": splits}


# --------------------------------------------------------------------------------------------- #
# the splits
# --------------------------------------------------------------------------------------------- #

def test_exactly_three_splits_with_the_per_cell_designs_inner_cells(corpus) -> None:
    df, t, mask, splits, d = corpus["df"], corpus["table"], corpus["mask"], corpus["splits"], corpus["design"]
    assert len(splits) == 3 and [s.fold for s in splits] == [0, 1, 2]
    assert [s.unit for s in splits] == [("inner_fold", 0), ("inner_fold", 1), ("inner_fold", 2)]
    c = ctx(t, df)
    per_cell = I.InnerCellCalibration(THR.k, THR.p, THR.m, 3, 30)
    want = per_cell.unit_assignment(t, mask, c)
    assert [sorted(map(tuple, s.meta["cells"])) for s in splits] == [sorted(map(tuple, w)) for w in want]
    # the fold builder's functions give the same cells (one implementation per design)
    train = df[mask].copy()
    built = CH.inner_cell_assignment(CH.inner_cell_majority(train, THR, v6_of(df)), SEED, 3, 30)
    assert [sorted(map(tuple, w)) for w in want] == [sorted(map(tuple, b)) for b in built]
    assert all(1 <= len(s.meta["cells"]) <= 30 for s in splits)
    assert d.name == "V5_inner_simultaneous" and d.describe()["one_fit_per_inner_fold"]
    # the V6-like cell is never an inner cell; every scored cell has a scorable row
    assert ("Pr(III)", "SC") not in {tuple(c) for s in splits for c in s.meta["cells"]}


def test_every_cell_of_a_fold_has_no_row_in_that_splits_training(corpus) -> None:
    """The registered hiding for EVERY cell of the fold at once: own rows, the element's X(?) rows in the system and in
    the component-sharing systems, the state's rows in the component-sharing systems."""
    df, t, mask, splits = corpus["df"], corpus["table"], corpus["mask"], corpus["splits"]
    train_df = df[mask]
    for sp in splits:
        cells = [tuple(c) for c in sp.meta["cells"]]
        tr = df.loc[t.index[sp.train_mask]]
        assert not (sp.train_mask & ~mask).any()
        for st, sy in cells:
            sym = SG.metal_properties(st)["symbol"]
            scope = {sy} | SG.systems_sharing_component(SYSTEMS, sy)
            in_scope = tr[SG.SYSTEM_COL].isin(scope)
            assert not ((tr[SG.METAL_COL] == st) & in_scope).any(), (st, sy)                 # own + sharing rows
            assert not (tr[SG.METAL_COL].isna() & (tr[SG.ELEMENT_COL] == sym) & in_scope).any(), (st, sy)  # X(?)
        # exactly support_graph.hide_cells of all the fold's cells on the outer training rows
        kept = SG.hide_cells(train_df, cells, component_aware=True)
        assert set(tr.index) == set(kept.index)
        assert set(t.index[sp.hidden_positions]) == set(train_df.index) - set(kept.index)
        assert not sp.train_mask[sp.cal_positions].any() and mask[sp.cal_positions].all()
    # the alias rows really are in play: some fold hides an Nd(III) x SA or SA|SB cell and drops the Nd(?) rows with it
    nd_alias = set(df.index[df[SG.METAL_COL].isna() & (df[SG.ELEMENT_COL] == "Nd")])
    hit = [sp for sp in splits if any(c[0] == "Nd(III)" and c[1] in ("SA", "SB", "SA|SB") for c in map(tuple, sp.meta["cells"]))]
    assert hit and all(nd_alias & set(t.index[sp.hidden_positions]) for sp in hit)
    # a component-sharing cell in another fold keeps its rows in training only when no cell of THIS fold reaches it
    for sp in splits:
        cells = {tuple(c) for c in sp.meta["cells"]}
        tr_cells = set(zip(df.loc[t.index[sp.train_mask], SG.METAL_COL], df.loc[t.index[sp.train_mask], SG.SYSTEM_COL]))
        assert not (cells & tr_cells)


def _shares_component(sy_a: str, sy_b: str) -> bool:
    """The mini-corpus component rule: ``SA|SB`` shares a component with ``SA`` and ``SB``."""
    return sy_a != sy_b and bool(set(sy_a.split("|")) & set(sy_b.split("|")))


def test_dropped_cells_stay_hidden_but_are_not_scored(corpus) -> None:
    """Cm(III) has exactly m + 1 = 3 systems; its cells share one private majority group, so two or three of them
    land in one inner fold and each fails the re-check with the others hidden.  The other drops of that fold are the
    strict re-check's (task X finding V-F03): same-state cells in component-sharing systems that empty each other."""
    df, t, mask, splits, d = corpus["df"], corpus["table"], corpus["mask"], corpus["splits"], corpus["design"]
    pair = [sp for sp in splits if sum(c[0] == PAIR_STATE for c in map(tuple, sp.meta["cells"])) >= 2]
    assert len(pair) == 1
    sp = pair[0]
    cm_cells = [tuple(c) for c in sp.meta["cells"] if c[0] == PAIR_STATE]
    dropped = set(map(tuple, sp.meta["dropped_cells"]))
    assert set(cm_cells) <= dropped == set(d.last_dropped[sp.fold])
    rows = df.index[(df[SG.METAL_COL] == PAIR_STATE) & df[SG.SYSTEM_COL].isin([c[1] for c in cm_cells])]
    hidden, cal = set(t.index[sp.hidden_positions]), set(t.index[sp.cal_positions])
    assert set(rows) <= hidden and not (set(rows) & cal)                       # hidden, not scored
    assert not any(u.startswith(PAIR_STATE) for u in sp.row_units)
    checks = {(c["metal_state"], c["system"]): c for c in d.last_checks[sp.fold]}
    for c in cm_cells:
        assert not checks[c]["scored"] and checks[c]["dropped_reason"] == "failed_recheck"
        assert checks[c]["other_systems_for_metal"] < THR.m and checks[c]["eligible"] is False
        assert checks[c]["n_own_rows_hidden_by_other_cells"] == 0                # a partner-count failure, not an emptied cell
    # the remaining drops are cells emptied by a same-state, component-sharing cell of the same fold (strict re-check)
    emptied = dropped - set(cm_cells)
    cells = [tuple(c) for c in sp.meta["cells"]]
    for c in emptied:
        assert checks[c]["n_rows"] == 0 and checks[c]["n_own_rows_hidden_by_other_cells"] == checks[c]["n_training_rows"] > 0
        assert any(o[0] == c[0] and _shares_component(o[1], c[1]) for o in cells)
    # every other cell of that fold passed and is scored
    others = [c for c in cells if c not in dropped]
    assert others and all(checks[c]["scored"] for c in others)
    assert set(map(tuple, sp.meta["scored_cells"])) == set(others)
    # the per-cell design scores those Cm cells (each alone keeps m other systems): the drop is the simultaneous re-check
    per_cell = I.InnerCellCalibration(THR.k, THR.p, THR.m, 3, 30).splits(t, mask, ctx(t, df))
    assert set(cm_cells) <= {s.unit for s in per_cell}
    # every drop anywhere is a recorded re-check failure, every scored cell a recorded pass, and the split's rows agree
    for s in splits:
        chk = {(c["metal_state"], c["system"]): c for c in d.last_checks[s.fold]}
        assert set(chk) == {tuple(c) for c in s.meta["cells"]}
        assert all(not chk[tuple(c)]["eligible"] or chk[tuple(c)]["n_scorable_rows"] == 0 for c in s.meta["dropped_cells"])
        assert all(chk[tuple(c)]["eligible"] and chk[tuple(c)]["scored"] for c in s.meta["scored_cells"])
        assert len(s.meta["scored_cells"]) + len(s.meta["dropped_cells"]) == len(s.meta["cells"])
        assert set(s.row_units) == {CH.cell_label(tuple(c)) for c in s.meta["scored_cells"]}
        for c in map(tuple, s.meta["dropped_cells"]):
            rows = df.index[(df[SG.METAL_COL] == c[0]) & (df[SG.SYSTEM_COL] == c[1])]
            assert set(rows) & set(t.index[s.hidden_positions]) and not set(rows) & set(t.index[s.cal_positions])


def test_recheck_counts_the_cells_own_rows_after_the_other_cells_hiding(corpus) -> None:
    """Task X finding V-F03: the re-check is the section 3.1 batch re-check reading -- k and p are counted among the rows
    the OTHER cells of the fold leave in training, nothing restored -- so a same-state cell in a component-sharing
    system hidden in the same fold empties its partner, which stays hidden but unscored; a cell without such a partner
    keeps its own rows."""
    df, t, mask, splits, d = corpus["df"], corpus["table"], corpus["mask"], corpus["splits"], corpus["design"]
    mpos = np.flatnonzero(mask)
    n_emptied = 0
    for sp in splits:
        cells = [tuple(c) for c in sp.meta["cells"]]
        drops = {c: I.hide_cell_mask(t, mask, c[0], c[1])[1] for c in cells}
        checks = {(c["metal_state"], c["system"]): c for c in d.last_checks[sp.fold]}
        for c in cells:
            others = np.zeros(t.n, dtype=bool)
            for o in cells:
                if o != c:
                    others |= drops[o]
            own = d.cell_positions(t, c[0], c[1])
            own = own[mask[own]]
            n_after = int((~others[own]).sum())
            assert checks[c]["n_rows"] == n_after                              # nothing restored
            assert checks[c]["n_own_rows_hidden_by_other_cells"] == len(own) - n_after
            partner = any(o[0] == c[0] and _shares_component(o[1], c[1]) for o in cells)
            assert (n_after == 0) == partner                                   # emptied iff a same-state sharing cell is in the fold
            if n_after == 0:
                n_emptied += 1
                assert not checks[c]["scored"] and c in set(map(tuple, sp.meta["dropped_cells"]))
                assert set(own) <= set(sp.hidden_positions) and not set(own) & set(sp.cal_positions)
            if checks[c]["scored"]:
                assert checks[c]["n_rows"] >= THR.k and checks[c]["n_publications"] >= THR.p
    assert n_emptied > 0                                                        # the mini corpus exercises the rule
    # a cell emptied here is scorable under the per-cell design (its own rows are the only ones hidden there)
    per_cell = {s.unit for s in I.InnerCellCalibration(THR.k, THR.p, THR.m, 3, 30).splits(t, mask, ctx(t, df))}
    emptied = {tuple(c) for s in splits for c in map(tuple, s.meta["dropped_cells"])
               if next(x for x in d.last_checks[s.fold] if (x["metal_state"], x["system"]) == tuple(c))["n_rows"] == 0}
    assert emptied and emptied <= per_cell


def test_wildcard_copy_partners_of_calibration_rows_are_counted_per_split(corpus) -> None:
    """Task X finding VL-A1-02: a section 2 wildcard copy of an inner calibration row sitting in the split's training
    rows is counted per inner fold (a diagnostic; the selection score is unchanged)."""
    df, t, splits = corpus["df"], corpus["table"], corpus["splits"]
    ids = df[FI.ROW_ID].astype(str)
    sp = splits[0]
    cal_id = ids.loc[t.index[sp.cal_positions[0]]]
    tr = np.flatnonzero(sp.train_mask)
    tr_id, tr_id2 = ids.loc[t.index[tr[0]]], ids.loc[t.index[tr[1]]]
    sys_of = df.set_index(FI.ROW_ID)[SG.SYSTEM_COL].astype(str)
    pairs = pd.DataFrame({"kind": ["STRUCTURE_WILDCARD", "STATE_WILDCARD", "STRUCTURE_WILDCARD"],
                          "id_a": [cal_id, tr_id2, "T:99999"], "id_b": [tr_id, tr_id, cal_id],
                          "system_a": [sys_of[cal_id], sys_of[tr_id2], "other"],
                          "system_b": [sys_of[tr_id], sys_of[tr_id], sys_of[cal_id]]})
    out = ID.wildcard_copy_partners_in_inner_training(splits, t, ids, pairs)
    assert [o["inner_fold"] for o in out] == [s.fold for s in splits]
    first = out[0]
    assert first["n_calibration_rows"] == len(sp.cal_positions) and first["n_pairs"] == 1
    assert first["n_calibration_rows_with_partner_in_training"] == 1 and first["calibration_row_ids"] == [cal_id]
    assert first["by_kind"] == {"STRUCTURE_WILDCARD": 1}
    assert first["n_pairs_cross_system"] + first["n_pairs_same_system"] == 1
    # the training-training pair and the pair with an unknown id count nowhere; other splits see the pair only when
    # the same two rows are calibration / training there
    for o, s in zip(out[1:], splits[1:]):
        cal = set(ids.loc[t.index[s.cal_positions]])
        trs = set(ids.loc[t.index[s.train_mask]])
        expect = int((cal_id in cal and tr_id in trs) or (tr_id in cal and cal_id in trs))
        assert o["n_pairs"] == expect and o["n_calibration_rows_with_partner_in_training"] == expect
    assert ID.wildcard_copy_partners_in_inner_training(splits, t, ids, pairs.iloc[:0])[0]["n_pairs"] == 0
    with pytest.raises(KeyError):
        ID.wildcard_copy_partners_in_inner_training(splits, t, ids.iloc[:5], pairs)


def test_calibration_rows_are_the_scorable_rows_of_the_surviving_cells(corpus) -> None:
    df, t, mask, splits = corpus["df"], corpus["table"], corpus["mask"], corpus["splits"]
    v6, excl = v6_of(df), excl_of(df)
    seen = []
    for sp in splits:
        cal = t.index[sp.cal_positions]
        rows = df.loc[cal]
        assert rows[SG.METAL_COL].notna().all() and (rows[SG.METAL_COL] != I.SR_III).all()
        assert not v6.loc[cal].any() and not excl.loc[cal].any()
        scored = {tuple(c) for c in sp.meta["scored_cells"]}
        assert set(zip(rows[SG.METAL_COL], rows[SG.SYSTEM_COL])) == scored
        assert list(sp.row_units) == [CH.cell_label(c) for c in zip(rows[SG.METAL_COL], rows[SG.SYSTEM_COL])]
        # every scorable outer-training row of a scored cell is a calibration row (nothing silently left out)
        for st, sy in scored:
            own = df.index[(df[SG.METAL_COL] == st) & (df[SG.SYSTEM_COL] == sy)]
            own = own[mask[t.positions(own)] & ~excl.loc[own].to_numpy() & ~v6.loc[own].to_numpy()]
            assert set(own) <= set(cal)
        assert np.array_equal(sp.cal_positions, np.sort(sp.cal_positions))
        seen.append(set(cal))
    assert not (seen[0] & seen[1]) and not (seen[0] & seen[2]) and not (seen[1] & seen[2])
    # the excluded row is hidden with its cell but never scored
    ex = df.index[excl][0]
    holder = [sp for sp in splits if ("Eu(III)", "SD") in {tuple(c) for c in sp.meta["cells"]}]
    if holder:
        assert ex in set(t.index[holder[0].hidden_positions]) and ex not in set(t.index[holder[0].cal_positions])


def test_unit_macro_counts_a_cell_once_whatever_its_row_count() -> None:
    """Addendum 1 item 2 on a hand-made error table: a 100-row cell and a 2-row cell weigh the same inside a fold, and
    folds weigh the same in the selection score whatever their cell counts."""
    sue = pd.DataFrame([
        # fold 0: two cells, one with 100 rows
        {"config": "a", "inner_fold": 0, "unit": "big", "n_rows": 100, "sum_abs_error": 100 * 0.10},
        {"config": "a", "inner_fold": 0, "unit": "small", "n_rows": 2, "sum_abs_error": 2 * 0.50},
        # fold 1: three cells
        {"config": "a", "inner_fold": 1, "unit": "u1", "n_rows": 5, "sum_abs_error": 5 * 0.20},
        {"config": "a", "inner_fold": 1, "unit": "u2", "n_rows": 5, "sum_abs_error": 5 * 0.20},
        {"config": "a", "inner_fold": 1, "unit": "u3", "n_rows": 5, "sum_abs_error": 5 * 0.50},
        # fold 2: one cell split over two splits (pooled within the fold)
        {"config": "a", "inner_fold": 2, "unit": "v", "n_rows": 3, "sum_abs_error": 3 * 0.30, "split_id": "x"},
        {"config": "a", "inner_fold": 2, "unit": "v", "n_rows": 1, "sum_abs_error": 1 * 0.70, "split_id": "y"},
    ])
    per = ID.fold_macro_table(sue).set_index("inner_fold")["macro_mae"]
    assert per[0] == pytest.approx((0.10 + 0.50) / 2)                     # not (100*0.1 + 2*0.5) / 102
    assert per[1] == pytest.approx(0.30)
    assert per[2] == pytest.approx((3 * 0.30 + 0.70) / 4)                  # the unit pooled within its fold
    score = ID.fold_mean_scores(sue)["a"]
    assert score == pytest.approx((per[0] + per[1] + per[2]) / 3)
    pooled_over_units = np.mean([0.10, 0.50, 0.20, 0.20, 0.50, (3 * 0.30 + 0.70) / 4])
    assert score == pytest.approx(1 / 3) and pooled_over_units == pytest.approx(1.9 / 6)
    assert score != pytest.approx(pooled_over_units)                        # folds of 2, 3 and 1 units weigh the same
    assert ID.fold_mean_scores(sue, exclude_folds=(1,))["a"] == pytest.approx((per[0] + per[2]) / 2)
    with pytest.raises(KeyError):
        ID.fold_macro_table(sue.drop(columns=["unit"]))
    # per-split arrays give the same table
    e = [np.array([0.1] * 100 + [0.5] * 2), np.array([0.2] * 5 + [0.2] * 5 + [0.5] * 5), np.array([0.3] * 3), np.array([0.7])]
    u = [["big"] * 100 + ["small"] * 2, ["u1"] * 5 + ["u2"] * 5 + ["u3"] * 5, ["v"] * 3, ["v"]]
    got = ID.per_fold_unit_macro(e, u, [0, 1, 2, 2])
    assert np.allclose(got.to_numpy(), per.to_numpy())
    # configurations scored on different folds are refused
    bad = pd.concat([sue, pd.DataFrame([{"config": "b", "inner_fold": 0, "unit": "big", "n_rows": 1, "sum_abs_error": 0.0}])])
    with pytest.raises(ValueError, match="same inner folds"):
        ID.fold_mean_scores(bad)


def test_selection_mean_over_three_splits_with_tie_rule_median_count_and_cross_fit_plan() -> None:
    from gen19ct.models import boosted as BO
    grid = (BO.CatBoostConfig(4, 10.0), BO.CatBoostConfig(4, 3.0), BO.CatBoostConfig(6, 3.0))
    # per-fold unit-macro MAEs: depth6 wins two folds, depth4_l23 wins the mean; depth4_l210 ties within 0.005
    fold_mae = {"depth4_l210": [0.30, 0.40, 0.503], "depth4_l23": [0.30, 0.40, 0.50], "depth6_l23": [0.29, 0.39, 0.60]}
    recs, fits = [], []
    for cfg in grid:
        for f, v in enumerate(fold_mae[cfg.label]):
            recs.append({"config": cfg.label, "split_id": f"s{f}", "inner_fold": f, "unit": f"cell{f}", "n_rows": 10,
                         "sum_abs_error": 10 * v})
            fits.append({"config": cfg.label, "split_id": f"s{f}", "inner_fold": f, "best_iteration": 99 + f * 10,
                         "n_trees": 100 + f * 10})
    rec = {"grid": [c.record() for c in grid], "split_unit_errors": recs, "fits": fits}
    scores = ID.fold_mean_scores(pd.DataFrame(recs))
    assert scores["depth4_l23"] == pytest.approx(0.40) and scores["depth6_l23"] == pytest.approx(1.28 / 3)
    chosen, iters, sel = BO.select_excluding_folds(rec, ())
    assert chosen == BO.CatBoostConfig(4, 10.0)                              # within 0.005 -> larger l2 (smaller)
    assert iters == 110 == ID.median_count([100, 110, 120]) and sel["inner_folds_used"] == [0, 1, 2]
    # excluding fold 2 (where depth6 loses): depth6 wins the mean of folds 0 and 1 by more than 0.005 -> depth6
    chosen2, iters2, sel2 = BO.select_excluding_folds(rec, (2,))
    assert chosen2 == BO.CatBoostConfig(6, 3.0) and iters2 == ID.median_count([100, 110]) == 105
    assert sel2["excluded_inner_folds"] == [2] and sel2["inner_folds_used"] == [0, 1]
    assert ID.cross_fit_plan([0, 1, 2]) == {0: (1, 2), 1: (0, 2), 2: (0, 1)} and ID.cross_fit_plan([0]) == {}
    assert ID.median_count([3, 4]) == 4 and ID.median_count([7]) == 7
    with pytest.raises(ValueError):
        ID.median_count([])


def test_certificate_nests_the_split_and_the_nested_certificate_guard_uses_it(corpus) -> None:
    df, t, mask, splits = corpus["df"], corpus["table"], corpus["mask"], corpus["splits"]
    for sp in splits:
        ctr, cte = sp.certificate
        assert not (sp.train_mask & ~ctr).any() and np.isin(sp.cal_positions, cte).all()
        cells = [tuple(c) for c in sp.meta["cells"]]
        assert set(t.index[cte]) == set(df.index[[(a, b) in set(cells) for a, b in zip(df[SG.METAL_COL], df[SG.SYSTEM_COL])]])
        assert set(t.index[ctr]) == set(SG.hide_cells(df, cells, component_aware=True).index)   # the whole table

    class MeanArm:
        name = "mean"

        def clone(self):
            return MeanArm()

        def fit_table(self, table, m, context):
            self.mu = float(table.y[m].mean())
            return self

        def predict_positions(self, pos):
            return pd.DataFrame({"mean_logD": np.full(len(pos), self.mu)})
    calls = []

    def guard(tr, te):
        calls.append((set(tr), set(te)))
        return {"ok": True}
    c = ctx(t, df, isolation_check=guard)
    w = I.ConformalWrapper(MeanArm(), splitter=ID.SimultaneousInnerCells(THR.k, THR.p, THR.m, 3, 30),
                           guard="nested_certificate").fit_table(t, mask, c)
    assert len(calls) == 3 and len(c.guard_cache) == 3
    for (tr, te), sp in zip(calls, splits):
        assert tr == set(t.index[sp.certificate[0]]) and te == set(t.index[sp.certificate[1]])
    assert w.calibration_units == [("inner_fold", 0), ("inner_fold", 1), ("inner_fold", 2)]
    assert len(w.residuals) == sum(len(sp.cal_positions) for sp in splits)


def test_isolation_check_passes_on_every_split_of_the_mini_corpus(corpus) -> None:
    df, t, splits = corpus["df"], corpus["table"], corpus["splits"]
    slim = FI.slim_frame(df)
    reps = ID.assert_isolated(splits, t, slim)
    assert len(reps) == 3 and all(r["ok"] and r["component_aware"] for r in reps)
    assert all(r["n_test"] == len(sp.cal_positions) for r, sp in zip(reps, splits))
    cert = ID.assert_isolated(splits, t, slim, certificate=True)
    assert all(r["ok"] for r in cert)
    # a split that keeps one alias row in training is caught by the same check
    bad = splits[0]
    alias = t.positions(df.index[df[SG.METAL_COL].isna() & (df[SG.ELEMENT_COL] == "Nd")])
    if not any(("Nd(III)", s) in {tuple(c) for c in bad.meta["cells"]} for s in ("SA", "SB", "SA|SB")):
        bad = next(sp for sp in splits if any(c[0] == "Nd(III)" and c[1] in ("SA", "SB", "SA|SB")
                                              for c in map(tuple, sp.meta["cells"])))
    leaky = np.array(bad.train_mask, copy=True)
    leaky[alias] = True
    from dataclasses import replace
    rep = ID.isolation_reports([replace(bad, train_mask=leaky)], t, slim)[0]
    assert not rep["ok"] and (rep["violations"]["V5_unknown_state_alias"] > 0
                              or rep["violations"]["V5_hidden_state_in_component_sharing_system"] > 0)
    with pytest.raises(AssertionError, match="isolation"):
        ID.assert_isolated([replace(bad, train_mask=leaky)], t, slim)


def test_determinism_seed_dependence_and_mask_dependence(corpus) -> None:
    df, t, mask, splits = corpus["df"], corpus["table"], corpus["mask"], corpus["splits"]
    again = ID.SimultaneousInnerCells(THR.k, THR.p, THR.m, 3, 30).splits(t, mask, ctx(t, df))
    for a, b in zip(splits, again):
        assert a.unit == b.unit and np.array_equal(a.train_mask, b.train_mask)
        assert np.array_equal(a.cal_positions, b.cal_positions) and list(a.row_units) == list(b.row_units)
        assert a.meta["cells"] == b.meta["cells"]
    other = ID.SimultaneousInnerCells(THR.k, THR.p, THR.m, 3, 30).splits(t, mask, ctx(t, df, seed=130363))
    assert len(other) == 3
    assert [s.meta["cells"] for s in other] != [s.meta["cells"] for s in splits]     # the seed moves the assignment
    assert Counter(c for s in other for c in map(tuple, s.meta["cells"])) == \
        Counter(c for s in splits for c in map(tuple, s.meta["cells"]))                # ... not the candidate cells
    # a different outer fold (different training rows) recomputes the candidates on its own rows
    mask2 = mask.copy()
    mask2[t.mask_of(df.index[(df[SG.METAL_COL] == "Sm(III)") & (df[SG.SYSTEM_COL] == "SE")])] = False
    sm = ID.SimultaneousInnerCells(THR.k, THR.p, THR.m, 3, 30).splits(t, mask2, ctx(t, df))
    assert ("Sm(III)", "SE") not in {tuple(c) for s in sm for c in s.meta["cells"]}
    assert all(not (s.train_mask & ~mask2).any() for s in sm)


def test_require_all_folds_and_the_skip_alternative() -> None:
    df = mini_corpus(one_group=True)                      # every cell's majority group is g00 -> one inner fold only
    t = I.RowTable(df)
    mask = np.ones(t.n, dtype=bool)
    with pytest.raises(ValueError, match="inner fold"):
        ID.SimultaneousInnerCells(THR.k, THR.p, THR.m, 3, 30).splits(t, mask, ctx(t, df))
    d = ID.SimultaneousInnerCells(THR.k, THR.p, THR.m, 3, 30, require_all_folds=False)
    sp = d.splits(t, mask, ctx(t, df))
    assert len(sp) == 1 and len(sp[0].meta["cells"]) <= 30
    assert set(d.last_cells) == {0, 1, 2} and not d.last_cells[1] and not d.last_cells[2]


def test_hno3_medium_restricts_cells_and_calibration_rows() -> None:
    df = mini_corpus(mixed_acid=True)
    t = I.RowTable(df)
    mask = np.ones(t.n, dtype=bool)
    c = ctx(t, df)
    with pytest.raises(ValueError):
        ID.SimultaneousInnerCells(THR.k, THR.p, THR.m, medium="HNO3")             # needs the frame
    with pytest.raises(ValueError):
        ID.SimultaneousInnerCells(THR.k, THR.p, THR.m, medium="H2SO4", frame=df)
    hno3 = ID.SimultaneousInnerCells(THR.k, THR.p, THR.m, 3, 30, medium="HNO3", frame=df).splits(t, mask, c)
    every = ID.SimultaneousInnerCells(THR.k, THR.p, THR.m, 3, 30).splits(t, mask, c)
    med = (df["acid_primary"] == "HNO3").to_numpy()
    assert len(hno3) == 3 and all(med[s.cal_positions].all() for s in hno3)
    n_hno3 = df[med].groupby([SG.METAL_COL, SG.SYSTEM_COL]).size()
    cells_h = {tuple(c) for s in hno3 for c in s.meta["cells"]}
    cells_a = {tuple(c) for s in every for c in s.meta["cells"]}
    assert all(n_hno3.get(cell, 0) >= THR.k for cell in cells_h) and cells_h < cells_a
    # the hiding is unchanged (whole cells, whatever the medium): HCl rows of a hidden cell leave training too
    for s in hno3:
        tr = df.loc[t.index[s.train_mask]]
        for st, sy in map(tuple, s.meta["cells"]):
            assert not ((tr[SG.METAL_COL] == st) & (tr[SG.SYSTEM_COL] == sy)).any()
    v = ID.SimultaneousInnerCells.for_variant("hno3_only", frame=df)
    assert v.medium == "HNO3" and v.thresholds == CH.PRIMARY and v.component_aware
    assert not ID.SimultaneousInnerCells.for_variant("cell_only").component_aware


def test_learned_arm_inner_design_factory_and_per_cell_refusal() -> None:
    d = ID.learned_arm_inner_design("V5")
    assert isinstance(d, ID.SimultaneousInnerCells) and d.thresholds == CH.PRIMARY and d.n_folds == 3 and d.max_cells == 30
    assert ID.learned_arm_inner_design("V5", "strict").thresholds == CH.STRICT
    for other in ("V5P", "V5-PAIR", "V6"):
        x = ID.learned_arm_inner_design(other, "strict")
        assert isinstance(x, ID.SimultaneousInnerCells) and x.thresholds == CH.PRIMARY      # heavy-arm fits: primary
    assert isinstance(ID.learned_arm_inner_design("V1"), I.GroupKFoldCalibration)
    assert ID.learned_arm_inner_design("V0").unit == "publication_group"
    assert isinstance(ID.learned_arm_inner_design("V2"), I.InnerMetalCalibration)
    with pytest.raises(NotImplementedError):
        ID.learned_arm_inner_design("V3")
    assert ID.is_per_cell_design(I.InnerCellCalibration(10, 1, 3)) and not ID.is_per_cell_design(d)
    assert not ID.is_per_cell_design(I.GroupKFoldCalibration(3))
    with pytest.raises(ValueError, match="per-cell"):
        ID.assert_learned_arm_design(I.InnerCellCalibration(10, 1, 3), "B5")
    ID.assert_learned_arm_design(d)
    assert "recheck" in ID.REGISTRATION_CHOICES and ID.NAME == "V5_inner_simultaneous"


def test_splits_summary_and_meta(corpus) -> None:
    s = ID.splits_summary(corpus["splits"])
    assert list(s["fold"]) == [0, 1, 2] and (s["n_units"] == s["n_scored_cells"]).all()
    assert (s["n_calibration"] <= s["n_hidden"]).all() and (s["n_train"] + s["n_hidden"] == int(corpus["mask"].sum())).all()
    assert (s["n_dropped_cells"] == s["n_cells"] - s["n_scored_cells"]).all()


# --------------------------------------------------------------------------------------------- #
# corpus: the outer-training rows of ONE registered selection-half cell fold, seed 104729 (no fit, no prediction)
# --------------------------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def real_corpus():
    if not (paths.FOLDS_DIR / "INDEX.json").exists():
        pytest.skip("folds/INDEX.json not built (run scripts/g19_build_folds.py)")
    from gen19ct.data import load
    model = load.load_model_rows()
    fr = I.prepare_frame(model)
    fr[FI.GROUP_COL] = fr[I.PUB_GROUP_COL]
    systems, comps = I.load_descriptor_tables()
    table = I.RowTable(fr, systems=systems, components=comps)
    v6 = FR.v6_target_mask(model, FR.v6_system_set(model)).reindex(fr.index).fillna(False).astype(bool)
    return {"fr": fr, "slim": FI.slim_frame(fr), "table": table, "v6": v6}


@pytest.mark.slow
def test_real_outer_training_rows_of_one_selection_half_cell_fold_pass_the_isolation_check(real_corpus) -> None:
    """ONE V5-primary exact selection-half cell fold (Eu(III) x a DGA system), seed 104729: the design's three splits on
    its outer-TRAINING rows pass ``fold_isolation_check(level="V5", component_aware=True)`` (training rows against
    calibration rows, and the certificates), hide whole folds of the per-cell design's inner cells, and score only the
    scorable rows of the surviving cells.  Nothing is fitted; no outer test row is touched."""
    fr, slim, t, v6 = real_corpus["fr"], real_corpus["slim"], real_corpus["table"], real_corpus["v6"]
    fold = next(x for x in FI.read_design("V5__primary__exact") if x.fold_id == "Eu(III)__a9ed8f70c7")
    assert fold.half == "S"
    keep = ~fr[FI.ROW_ID].astype(str).isin(set(fold.hidden_row_ids)).to_numpy()
    c = I.FitContext(table=t, v6_mask=v6, seed=SEED, isolation_check=lambda tr, te: {"ok": True})
    design = ID.SimultaneousInnerCells.for_variant("primary")
    splits = design.splits(t, keep, c)
    assert len(splits) == 3 and [s.fold for s in splits] == [0, 1, 2]
    per_cell = I.InnerCellCalibration(10, 1, 3, 3, 30).unit_assignment(t, keep, c)
    assert [sorted(map(tuple, s.meta["cells"])) for s in splits] == [sorted(map(tuple, w)) for w in per_cell]
    assert all(1 <= len(s.meta["cells"]) <= 30 for s in splits)
    reps = ID.assert_isolated(splits, t, slim)                                  # training rows vs calibration rows
    assert all(r["ok"] for r in reps)
    certs = ID.assert_isolated(splits, t, slim, certificate=True)               # the nested-certificate sides
    assert all(r["ok"] for r in certs)
    ok = I.scorable_mask(t, c, known_state_only=True)
    for sp in splits:
        assert not (sp.train_mask & ~keep).any() and keep[sp.cal_positions].all() and ok[sp.cal_positions].all()
        assert not sp.train_mask[sp.hidden_positions].any()
        tr = fr.loc[t.index[sp.train_mask]]
        for st, sy in map(tuple, sp.meta["cells"]):
            assert not ((tr[SG.METAL_COL] == st) & (tr[SG.SYSTEM_COL] == sy)).any()
        assert set(sp.row_units) == {CH.cell_label(tuple(cc)) for cc in sp.meta["scored_cells"]}
        # no outer test row anywhere in the split
        assert not set(fold.hidden_row_ids) & set(fr.loc[t.index[sp.train_mask | np.isin(np.arange(t.n), sp.cal_positions)], FI.ROW_ID])
    dropped = {c for s in splits for c in map(tuple, s.meta["dropped_cells"])}
    n_cells = sum(len(s.meta["cells"]) for s in splits)
    assert len(dropped) < n_cells                                                # the fold is not emptied by the re-check
    emptied, additional = _print_recheck_counts("V5__primary__exact / Eu(III)__a9ed8f70c7", splits, design, t, keep)
    # task X finding V-F03: 5 cells fail on partner counts under either reading; the strict re-check drops ONE more of
    # the 90 -- Pu(IV) in a DGA system, emptied by the component-aware hiding of its same-state mixture partner in the
    # same inner fold (the partner is emptied too, but already failed its partner counts)
    assert [len(s.meta["cells"]) for s in splits] == [30, 30, 30] and len(dropped) == 6
    assert additional == [("Pu(IV)", "CCCCCCCC(=O)N(CCCCCC)CCCCCC")]
    assert set(additional) <= set(emptied) and all(c[0] == "Pu(IV)" for c in emptied) and len(emptied) == 2


def _print_recheck_counts(label: str, splits, design, table, mask) -> tuple[list, list]:
    """The re-check counts of one real fold (nothing fitted): cells, scored, dropped; the cells another cell of their
    inner fold emptied (``n_rows`` 0 after the other cells' hiding); and among the dropped cells those the own-row
    reading (k, p from the cell's own rows restored) would have scored -- the strict reading's additional drops (task X
    finding V-F03).  Returns ``(emptied, additional)``."""
    dropped = {c for s in splits for c in map(tuple, s.meta["dropped_cells"])}
    n_cells = sum(len(s.meta["cells"]) for s in splits)
    space = CH.EligibilitySpace(I.unit_frame(table, mask), design.medium)
    mpos, thr = np.flatnonzero(mask), design.thresholds
    emptied, additional = [], []
    print(f"[{label}, seed {SEED}] inner cells per fold "
          f"{[len(s.meta['cells']) for s in splits]}, scored {[len(s.meta['scored_cells']) for s in splits]}, "
          f"dropped {[len(s.meta['dropped_cells']) for s in splits]} ({len(dropped)} of {n_cells}); "
          f"calibration rows {[len(s.cal_positions) for s in splits]}, hidden rows {[len(s.hidden_positions) for s in splits]}")
    for s in splits:
        cells = [tuple(c) for c in s.meta["cells"]]
        drops = {c: I.hide_cell_mask(table, mask, c[0], c[1], component_aware=design.component_aware,
                                     component_map=design.component_map)[1] for c in cells}
        total = np.sum([drops[c][mpos].astype(np.int32) for c in cells], axis=0)
        for c in map(tuple, s.meta["dropped_cells"]):
            chk = next(x for x in design.last_checks[s.fold] if (x["metal_state"], x["system"]) == c)
            by_others = chk["n_own_rows_hidden_by_other_cells"]
            if chk["n_rows"] == 0 and by_others > 0:
                emptied.append(c)
            own = np.zeros(table.n, dtype=bool)
            own[design.cell_positions(table, c[0], c[1])] = True
            own_reading = space.check(c, thr, keep=((total - drops[c][mpos].astype(np.int32)) == 0) | own[mpos])
            if own_reading["eligible"] and chk["n_scorable_rows"] > 0:
                additional.append(c)
            print(f"  fold {s.fold} dropped {CH.cell_label(c)}: n_rows_after_others_hiding {chk['n_rows']} "
                  f"(own rows hidden by other cells {by_others}), other_systems {chk['other_systems_for_metal']}, "
                  f"other_states {chk['other_metal_states_for_system']}, connected {chk['connected_without_cell']}"
                  f"{'; scored under the own-row reading' if c in additional else ''}")
    print(f"  cells emptied by another cell of their fold: {len(emptied)}; dropped ONLY under the strict re-check "
          f"(section 3.1 reading): {len(additional)} of {n_cells}")
    return emptied, additional


@pytest.mark.slow
def test_real_outer_training_rows_of_the_benchmark_batched_fold_under_the_strict_recheck(real_corpus) -> None:
    """The V5-primary batched selection-half fold the addendum benchmark timed (``s104729_S_b000``), seed 104729,
    outer-TRAINING rows only: the strict re-check drops no cell beyond the partner-count failures there (task X finding
    V-F03: 0 of 90), so the benchmark's splits are unchanged by it.  Nothing is fitted."""
    fr, slim, t, v6 = real_corpus["fr"], real_corpus["slim"], real_corpus["table"], real_corpus["v6"]
    fold = next(x for x in FI.read_design("V5__primary__batched") if x.fold_id == "s104729_S_b000")
    assert fold.half == "S" and fold.seed == SEED
    keep = ~fr[FI.ROW_ID].astype(str).isin(set(fold.hidden_row_ids)).to_numpy()
    c = I.FitContext(table=t, v6_mask=v6, seed=SEED, isolation_check=lambda tr, te: {"ok": True})
    design = ID.SimultaneousInnerCells.for_variant("primary")
    splits = design.splits(t, keep, c)
    assert len(splits) == 3 and all(r["ok"] for r in ID.assert_isolated(splits, t, slim))
    emptied, additional = _print_recheck_counts("V5__primary__batched / s104729_S_b000", splits, design, t, keep)
    assert [len(s.meta["cells"]) for s in splits] == [30, 30, 30] and additional == []
    assert [len(s.meta["dropped_cells"]) for s in splits] == [0, 0, 4]
    assert [len(s.meta["scored_cells"]) for s in splits] == [30, 30, 26]          # the benchmark's 30/30/26 cells hidden
    for sp in splits:
        assert not (sp.train_mask & ~keep).any() and keep[sp.cal_positions].all()
        assert not set(fold.hidden_row_ids) & set(fr.loc[t.index[sp.train_mask | np.isin(np.arange(t.n), sp.cal_positions)], FI.ROW_ID])
