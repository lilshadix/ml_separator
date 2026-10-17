"""LT-07: strict adjacency is |dZ| = 1 (12 pair types, promethium absent), so Nd-Sm is never a strict pair;
gen13 adjacency (strict + Nd-Sm) is a named secondary used only by COH-SF-NDSM; pairs are lighter first."""
from __future__ import annotations

import numpy as np

from gen201 import data as D
from gen20k import data as GD

STRICT_TYPES = {"La/Ce", "Ce/Pr", "Pr/Nd", "Sm/Eu", "Eu/Gd", "Gd/Tb", "Tb/Dy", "Dy/Ho", "Ho/Er", "Er/Tm", "Tm/Yb",
                "Yb/Lu"}


def test_nd_sm_is_not_strict():
    nd, sm = D.IDX["Nd"], D.IDX["Sm"]
    assert sm == nd + 1                                  # adjacent in the 14-metal order ...
    assert not GD.is_strict_adjacent(nd, sm)             # ... but dZ = 2 (Pm absent)
    assert GD.is_gen13_adjacent(nd, sm)
    strict = {f"{D.LANTHANIDES[i]}/{D.LANTHANIDES[i + 1]}" for i in range(13) if GD.is_strict_adjacent(i, i + 1)}
    assert strict == STRICT_TYPES and len(strict) == 12


def test_q_adj_is_strict_and_lighter_first(bench):
    for ci in range(len(bench.frame)):
        obs = set(np.flatnonzero(~np.isnan(bench.Y[ci])).tolist())
        for ia, ib in D.q_adj(bench, ci):
            assert ia < ib and ia in obs and ib in obs
            assert D.Z[ib] - D.Z[ia] == 1
            assert (D.LANTHANIDES[ia], D.LANTHANIDES[ib]) != ("Nd", "Sm")


def test_strict_populations_hold_no_nd_sm(pops):
    for name in ("COH-SF-C3", "COH-SF-ANY", "COH-SF-C1", "COH-SC009-SF", "COH-NOSC009-SF", "COH-SF-C3@NODMD"):
        t = pops[name].rows
        assert t["strict"].all() and (t["dZ"] == 1).all()
        assert not (t["pair"] == "Nd/Sm").any()
        assert set(t["pair"]) <= STRICT_TYPES
        assert (t["ia"] < t["ib"]).all()


def test_ndsm_pass_is_labelled(pops):
    t = pops["COH-SF-NDSM"].rows
    nd = t["pair"] == "Nd/Sm"
    assert nd.any()
    assert (~t.loc[nd, "strict"]).all() and t.loc[nd, "gen13adj"].all() and (t.loc[nd, "dZ"] == 2).all()
    assert t.loc[~nd, "strict"].all()
