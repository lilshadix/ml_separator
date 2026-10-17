"""LT-17: identical keys and y across arms (one long table, every arm scored on the same selected rows), identical
chemotype and publication lists for COH-D-C3 and COH-SF-C3 (label-free row manifests), hence one shared draw
matrix per unit; population selection fails on a missing, duplicated or mislabelled key, and a Frame refuses
seeds whose keys or y differ."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gen201 import inference as I
from gen201 import metrics as M
from gen201 import report as RP

import g201_synth as S


def test_c3_populations_share_cluster_lists_in_the_row_manifests():
    man = RP.load_row_manifests()
    d, q = man["COH-D-C3"], man["COH-SF-C3"]
    assert len(d) == 2425 and len(q) == 1802
    for col, n in (("chemotype", 38), ("publication_id", 39)):
        a, b = sorted(d[col].astype(str).unique()), sorted(q[col].astype(str).unique())
        assert a == b and len(a) == n
    # per (design, seed) the manifest rows are the same cells: the keys are seed- and design-invariant
    raw_d = pd.read_parquet(RP.ROW_MANIFESTS / "COH-D-C3.parquet", columns=["design", "split_seed", "ci", "im"])
    for (_, _), g in raw_d.groupby(["design", "split_seed"]):
        assert len(g) == 2425 and set(zip(g.ci, g.im)) == set(zip(d.ci, d.im))
    # one draw matrix object per unit for both targets
    for unit, col in (("chemotype", "chemotype"), ("publication", "publication_id")):
        A = I.draw_matrix(unit, sorted(d[col].astype(str).unique()), 50)
        B = I.draw_matrix(unit, sorted(q[col].astype(str).unique()), 50)
        assert A is B


def test_every_arm_scored_on_identical_keys_and_y():
    cells = S.make_cells(8, seed=2)
    logd, logsf = S.make_tables(cells)
    man = S.manifests_for(logd, logsf)
    sel = RP.select_population(logsf, man["COH-SF-C3"], "COH-SF-C3")
    F = I.Frame(sel, arms=["B", "C", "FLAT", "PAIRMEAN", "G14_ZERO"])
    for arm in ("B", "C", "FLAT", "PAIRMEAN", "G14_ZERO"):
        assert F.pred(arm).shape == (len(F.seeds), F.n)
    for s in F.seeds:
        part = sel[sel.split_seed == s].sort_values(["ci", "ia", "ib"])
        assert np.array_equal(part["y"].to_numpy(), F.y)
        assert np.array_equal(part[["ci", "ia", "ib"]].to_numpy(), F.rows[["ci", "ia", "ib"]].to_numpy())


def test_selection_failures():
    cells = S.make_cells(6, seed=4)
    logd, logsf = S.make_tables(cells, seeds=(1, 2))
    man = S.manifests_for(logd, logsf)
    keys = man["COH-D-C3"]
    missing = logd.drop(index=logd.index[(logd.split_seed == 2)][:1])
    with pytest.raises(M.CoverageError, match="seed 2"):
        RP.select_population(missing, keys, "COH-D-C3")
    dup = pd.concat([logd, logd.iloc[:1]], ignore_index=True)
    with pytest.raises(M.CoverageError):
        RP.select_population(dup, keys, "COH-D-C3")
    bad = keys.copy()
    bad.loc[0, "chemotype"] = "zz"
    with pytest.raises(M.CoverageError, match="identity"):
        RP.select_population(logd, bad, "COH-D-C3")


def test_frame_refuses_seed_dependent_rows():
    cells = S.make_cells(6, seed=4)
    logd, _ = S.make_tables(cells, seeds=(1, 2))
    t = logd.copy()
    t.loc[t.index[t.split_seed == 2][0], "y"] += 1.0
    with pytest.raises(M.CoverageError, match="seed-invariant"):
        I.Frame(t)
    t2 = logd[~((logd.split_seed == 2) & (logd.ci == logd.ci.min()))]
    with pytest.raises(M.CoverageError):
        I.Frame(t2)
