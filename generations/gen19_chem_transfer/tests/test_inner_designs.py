"""One implementation per section 7 inner design (pre-registration section 7; verification finding VL-01).

The fold builder (``folds.cell_holdout.inner_cells_V5``, ``folds.source_holdout.inner_folds_V1``,
``folds.metal_holdout.inner_metals_V2``) and the conformal calibration splitters that the pre-seal run uses
(``models.interface.InnerCellCalibration``, ``GroupKFoldCalibration``, ``InnerMetalCalibration``) must return the SAME
inner units -- which cells / groups / states, in which inner fold -- for one outer training set and seed, and the
calibration rows must be the rows the builder scores (known metal state, not Sr(III), not ``V6_TARGET_ROWS``; minus the
run's own ``exclude_from_scoring`` rows).

Fast tests use a synthetic frame; the corpus test (marker ``slow``) uses one registered outer fold per design and the
first discovery seed.  Nothing is fitted and nothing is scored.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import pytest

from gen19ct import paths
from gen19ct.chemistry import support_graph as SG
from gen19ct.folds import cell_holdout as CH
from gen19ct.folds import io as FI
from gen19ct.folds import metal_holdout as MH
from gen19ct.folds import registered as FR
from gen19ct.folds import source_holdout as SH
from gen19ct.models import interface as I

SEED = 104729
STATES = ("La(III)", "Ce(III)", "Pr(III)", "Nd(III)", "Sm(III)", "Eu(III)", "Gd(III)", "Am(III)", "Cm(III)")
SYSTEMS = ("S1", "S2", "S3", "S4", "S5", "S6")


def _synthetic() -> pd.DataFrame:
    """9 metal states x 6 systems, 12 publication groups, unequal cell sizes, X(?) and Sr(III) rows, and rows whose
    groups differ inside a cell (so the majority rule and its tie rule matter)."""
    rng = np.random.default_rng(3)
    recs = []
    k = 0

    def add(state, system, n, group, element=None):
        nonlocal k
        for _ in range(n):
            el = element if state is None else SG.metal_properties(state)["symbol"]
            recs.append({FI.ROW_ID: f"T:{k:05d}", SG.METAL_COL: state, SG.ELEMENT_COL: el, SG.SYSTEM_COL: system,
                         SG.PUB_COL: f"pub_{group}", FI.GROUP_COL: group, "acid_primary": "HNO3",
                         SG.ACID_ANION_COL: "nitrate", SG.LOG_ACID_COL: 0.0, SG.LOG_EXT_COL: -1.0, "log_D": 0.0})
            k += 1
    for i, st in enumerate(STATES):
        for j, sy in enumerate(SYSTEMS):
            if (i + 2 * j) % 7 == 0:
                continue                                    # a few missing cells
            n = int(rng.integers(4, 18))
            add(st, sy, n // 2, f"g{(i + j) % 12:02d}")
            add(st, sy, n - n // 2, f"g{(i * 5 + j) % 12:02d}")
    add(None, "S1", 5, "g03", element="Nd")
    add("Sr(III)", "S2", 12, "g04")
    df = pd.DataFrame(recs)
    df.index = pd.Index([f"L{i}" for i in range(len(df))])
    df[I.PUB_GROUP_COL] = df[FI.GROUP_COL]
    return df


def _ctx(table: I.RowTable, v6: pd.Series, seed: int = SEED) -> I.FitContext:
    return I.FitContext(table=table, v6_mask=v6, seed=seed, isolation_check=lambda tr, te: {"ok": True})


@pytest.mark.parametrize("seed", [SEED, 130363, 7])
def test_v5_inner_cells_identical_on_synthetic_frame(seed: int) -> None:
    df = _synthetic()
    v6 = pd.Series((df[SG.METAL_COL] == "Pr(III)") & (df[SG.SYSTEM_COL] == "S3"), index=df.index)
    thr = CH.Thresholds(8, 1, 2)
    table = I.RowTable(df)
    mask = np.ones(table.n, dtype=bool)
    mask[:40] = False                                      # an 'outer fold' removed from training
    train = df[mask]
    built = CH.inner_cells_V5(train, thr, seed, batched=False, max_cells=4, v6_ids=df.loc[v6, FI.ROW_ID], check=False)
    by_fold = defaultdict(list)
    for f in built:
        by_fold[f.meta["inner_fold"]] += [tuple(c) for c in f.meta["cells"]]
    splitter = I.InnerCellCalibration(k=thr.k, p=thr.p, m=thr.m, n_folds=3, max_cells_per_fold=4)
    splits = splitter.splits(table, mask, _ctx(table, v6, seed))
    wrapper = defaultdict(list)
    for sp in splits:
        wrapper[sp.fold].append(sp.unit)
    assert built and {k: sorted(v) for k, v in by_fold.items()} == {k: sorted(v) for k, v in wrapper.items()}
    for sp, f in zip(sorted(splits, key=lambda s: (s.fold, s.unit)),
                     sorted(built, key=lambda f: (f.meta["inner_fold"], tuple(f.meta["cells"][0])))):
        assert sorted(table.ids[sp.cal_positions]) == sorted(f.scored_row_ids)      # calibration rows = scored rows
        assert sorted(table.ids[sp.hidden_positions]) == sorted(f.hidden_row_ids)   # same hiding


@pytest.mark.parametrize("seed", [SEED, 196613])
def test_v1_inner_groups_and_calibration_rows_identical_on_synthetic_frame(seed: int) -> None:
    df = _synthetic()
    v6 = pd.Series(df[SG.METAL_COL] == "Nd(III)", index=df.index)
    table = I.RowTable(df)
    mask = (df[FI.GROUP_COL] != "g00").to_numpy()
    built = SH.inner_folds_V1(df[mask], seed, v6_ids=df.loc[v6, FI.ROW_ID], check=False)
    splits = I.GroupKFoldCalibration(3).splits(table, mask, _ctx(table, v6, seed))
    assert [tuple(sorted(f.meta["groups"])) for f in built if f.scored_row_ids] == [sp.unit for sp in splits]
    for sp, f in zip(splits, [f for f in built if f.scored_row_ids]):
        cal = set(table.ids[sp.cal_positions])
        assert cal == set(f.scored_row_ids)
        assert not any(df.loc[df[FI.ROW_ID].isin(cal), SG.METAL_COL].isna())     # no X(?) calibration row (VL-03)


@pytest.mark.parametrize("seed", [SEED, 155921])
def test_v2_inner_states_identical_on_synthetic_frame(seed: int) -> None:
    df = _synthetic()
    v6 = pd.Series(False, index=df.index)
    table = I.RowTable(df)
    mask = (df[SG.ELEMENT_COL] != "Gd").to_numpy()
    built = MH.inner_metals_V2(df[mask], seed, n_states=3, v6_ids=[], check=False)
    splitter = I.InnerMetalCalibration(n_metals=3, min_rows=40, min_systems=4)
    want = MH.inner_state_pick(df[mask], seed, 3, 40, 4)
    got = [sp.unit for sp in splitter.splits(table, mask, _ctx(table, v6, seed))]
    assert got == want and len(want) == 3
    # the builder with the registered V2 thresholds and the splitter with the same thresholds agree too
    assert [f.units[0] for f in built] == [sp.unit for sp in I.InnerMetalCalibration(3).splits(table, mask,
                                                                                                _ctx(table, v6, seed))]


def test_inner_v5_batches_respect_max_cells_per_batch_and_default_is_unchanged() -> None:
    """Section 7 compute-plan item 6: the capped re-colouring is available to the inner V5 design too ("batching
    follows the outer rule"); ``None`` leaves the registered inner batches untouched."""
    df = _synthetic()
    v6_ids = df.loc[(df[SG.METAL_COL] == "Pr(III)") & (df[SG.SYSTEM_COL] == "S3"), FI.ROW_ID]
    thr = CH.Thresholds(8, 1, 2)
    train = df.iloc[40:]
    default = CH.inner_cells_V5(train, thr, SEED, v6_ids=v6_ids, check=False)
    none = CH.inner_cells_V5(train, thr, SEED, v6_ids=v6_ids, check=False, max_cells_per_batch=None)
    assert [(f.fold_id, f.fold_hash) for f in none] == [(f.fold_id, f.fold_hash) for f in default]
    assert max(len(f.meta["cells"]) for f in default) > 2
    capped = CH.inner_cells_V5(train, thr, SEED, v6_ids=v6_ids, check=False, max_cells_per_batch=2)
    assert max(len(f.meta["cells"]) for f in capped) <= 2
    cells = lambda folds: sorted((f.meta["inner_fold"], tuple(c)) for f in folds for c in f.meta["cells"])  # noqa: E731
    assert cells(capped) == cells(default)                      # same inner cells, only the batches differ
    space = CH.EligibilitySpace(train)
    variant = CH.Variant("inner", thr)
    for f in capped:
        cs = [tuple(c) for c in f.meta["cells"]]
        assert len({c[0] for c in cs}) == len(cs) and len({c[1] for c in cs}) == len(cs)
        masks = CH.cell_masks(train, cs, variant)
        total = np.sum([masks[c].astype(np.int32) for c in cs], axis=0)
        assert all(space.check(c, thr, keep=(total - masks[c]) == 0)["eligible"] for c in cs)


def test_inner_assignment_functions_are_seeded_and_deterministic() -> None:
    maj = {("Nd(III)", f"S{i}"): f"g{i % 4}" for i in range(12)} | {("Eu(III)", f"S{i}"): f"g{i % 5}" for i in range(12)}
    a = CH.inner_cell_assignment(maj, SEED, 3, 5)
    assert a == CH.inner_cell_assignment(maj, SEED, 3, 5)
    assert all(len(x) <= 5 for x in a) and len(a) == 3
    assert {c for x in a for c in x} <= set(maj)
    g = {f"g{i}": 10 + i for i in range(9)}
    assert SH.inner_group_assignment(g, SEED) == SH.inner_group_assignment(g, SEED)
    assert set(SH.inner_group_assignment(g, SEED).values()) == {0, 1, 2}
    assert CH.majority_group(["b", "a", "b", "a", "c"]) == "a"                  # tie -> smallest label


# --------------------------------------------------------------------------------------------- #
# corpus: one registered outer fold per design
# --------------------------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def corpus():
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


def _outer(corpus, stem: str, fold_id: str):
    f = next(x for x in FI.read_design(stem) if x.fold_id == fold_id)
    keep = ~corpus["fr"][FI.ROW_ID].astype(str).isin(set(f.hidden_row_ids)).to_numpy()
    return f, keep


@pytest.mark.slow
def test_corpus_calibration_splitters_equal_the_fold_builder(corpus) -> None:
    fr, slim, t, v6 = corpus["fr"], corpus["slim"], corpus["table"], corpus["v6"]
    v6_ids = FI.registered_v6_ids()
    ctx = I.FitContext(table=t, v6_mask=v6, seed=SEED, isolation_check=lambda tr, te: {"ok": True})

    # V5: the pre-seal comparison fold of the verifier (Eu(III) x a DGA system), exact inner cells
    _, keep = _outer(corpus, "V5__primary__exact", "Eu(III)__a9ed8f70c7")
    built = CH.inner_cells_V5(slim[keep], CH.PRIMARY, SEED, batched=False, v6_ids=v6_ids, check=False)
    b = sorted((f.meta["inner_fold"], tuple(f.meta["cells"][0])) for f in built)
    splits = I.InnerCellCalibration(10, 1, 3, 3, 30).splits(t, keep, ctx)
    w = sorted((sp.fold, sp.unit) for sp in splits)
    assert b == w and len(b) > 0
    scored = {tuple(f.meta["cells"][0]): set(f.scored_row_ids) for f in built}
    for sp in splits:
        assert set(t.ids[sp.cal_positions]) == scored[sp.unit]

    # V1: the first registered fold
    fid = sorted(x.fold_id for x in FI.read_design("V1__copy__exact"))[0]
    _, keep = _outer(corpus, "V1__copy__exact", fid)
    built = SH.inner_folds_V1(slim[keep], SEED, v6_ids=v6_ids, check=False)
    splits = I.GroupKFoldCalibration(3).splits(t, keep, ctx)
    assert [tuple(sorted(f.meta["groups"])) for f in built if f.scored_row_ids] == [sp.unit for sp in splits]
    for sp, f in zip(splits, [f for f in built if f.scored_row_ids]):
        assert set(t.ids[sp.cal_positions]) == set(f.scored_row_ids)

    # V2: U(VI) held out
    _, keep = _outer(corpus, "V2__element__exact", "U(VI)")
    built = MH.inner_metals_V2(slim[keep], SEED, v6_ids=v6_ids, check=False)
    splits = I.InnerMetalCalibration(3, 100, 5).splits(t, keep, ctx)
    assert [f.units[0] for f in built] == [sp.unit for sp in splits]
    for sp, f in zip(splits, built):
        assert set(t.ids[sp.cal_positions]) == set(f.scored_row_ids)
