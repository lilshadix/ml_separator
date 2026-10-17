"""LT-25: the DMDPhPDA rule (SMILES AND cond__diluent__ch3cl == 1, never a name filter) selects 5 cells, 70
entries and 60 strict-adjacent pairs, all in C3; they are absent from every COH-NODMD population; the COH-NODMD
training sets equal the primary training sets minus the 5 cells in all 150 (design, seed, fold) units, with the
inner partition never re-dealt (IMPLEMENTATION_PLAN.md §1.1(6), §2)."""
from __future__ import annotations

import numpy as np

from gen201 import data as D
from gen201 import folds as FO


def test_rule_counts(bench, meta):
    dm = D.dmd_cells(bench)
    assert len(dm) == 5
    assert int(D.observed(bench)[dm].sum()) == 70
    assert sum(len(D.q_adj(bench, int(c))) for c in dm) == 60
    assert meta.loc[dm, "is_c3"].all()


def test_rule_is_structure_and_diluent_not_name(bench):
    f = bench.frame
    same_smiles = f["extractant"].astype(str).to_numpy() == D.DMD_SMILES
    ch3cl = f[D.DMD_COND_COLUMN].to_numpy(dtype=float) == 1
    assert same_smiles.sum() > 5                     # the curated copy shares the structure ...
    assert (same_smiles & ~ch3cl).sum() > 0          # ... under another diluent, and is kept
    assert np.array_equal(D.dmd_mask(bench), same_smiles & ch3cl)


def test_nodmd_populations_exclude_the_cells(bench, pops):
    dm = set(D.dmd_cells(bench).tolist())
    for name, rs in pops.items():
        if name.endswith("@NODMD"):
            assert not (set(rs.rows["ci"].tolist()) & dm), name
            parent = pops[name[: -len("@NODMD")]]
            assert rs.n_rows == int((~parent.rows["ci"].isin(dm)).sum())
    assert pops["COH-D-C3@NODMD"].n_rows == 2355 and pops["COH-SF-C3@NODMD"].n_rows == 1742
    assert len(pops["COH-D-C3@NODMD"].cells) == 224
    assert pops["COH-D-ALL@NODMD"].n_rows == 3289 and len(pops["COH-D-ALL@NODMD"].cells) == 516


def test_nodmd_train_sets_in_all_150_units(bench, folds):
    dm = D.dmd_cells(bench)
    dms = set(dm.tolist())
    units = 0
    for d in FO.DESIGNS:
        for fo in folds[d]:
            nd = FO.drop_cells(fo, dm)
            assert sorted(nd.train_index.tolist()) == sorted(set(fo.train_index.tolist()) - dms)
            assert np.array_equal(nd.test_index, fo.test_index)
            assert not (set(nd.inner_train_index.tolist()) | set(nd.inner_validation_index.tolist())) & dms
            prim = FO.inner_folds(bench, d, fo)
            nod = FO.inner_folds(bench, d, fo, drop=dm)
            for p, q in zip(prim, nod):                 # same dealing, cells removed afterwards
                assert q.train_pos.tolist() == sorted(set(p.train_pos.tolist()) - dms)
                assert q.val_pos.tolist() == sorted(set(p.val_pos.tolist()) - dms)
            units += 1
    assert units == 150


def test_train_rows_flag_the_cells(bench, folds):
    fo = folds["B"][0]
    t = D.train_logd_rows(bench, fo.train_index)
    assert np.array_equal(t["is_dmd"].to_numpy(), np.isin(t["ci"].to_numpy(), D.dmd_cells(bench)))
    assert len(t) == int(D.observed(bench)[fo.train_index].sum())
