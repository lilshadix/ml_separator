"""Tests for stage-1 block B (r4.slots_b): geometry join (metal from the archive, not the safe_exp_id prefix),
support exactness, zero trees returns B1, fit poisoning, control construction.  None compares a held-out
prediction with its target.

    .venv/Scripts/python.exe -m pytest generations/gen21/round4_20261022/tests -q -p no:cacheprovider -W ignore
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r4 import refs as RF  # noqa: E402
from r4 import registry as RG  # noqa: E402
from r4 import slots_b as SB  # noqa: E402
from r4.leakage import poisoned_bench  # noqa: E402
from g21 import cache as C  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import N_LN  # noqa: E402

FOLD = ("BPG", 130363, 2)


@pytest.fixture(scope="module")
def bench():
    return D.load_bench()


@pytest.fixture(scope="module")
def rec():
    return C.load(*FOLD)


@pytest.fixture(scope="module")
def b1(bench, rec):
    return RF.fit_reference(bench, rec, bench.Y, "B1")


def test_geometry_join_uses_archive_metal(bench):
    import pandas as pd
    tab = SB.geometry_table()
    assert len(tab) > 2000
    f = bench.frame
    # every geometry entry sits on a metal the cell actually measured
    for (cid, m) in list(tab)[:500]:
        ci = int(np.flatnonzero(f.cell_id.to_numpy() == cid)[0])
        assert np.isfinite(bench.Y[ci, m])
    # prefixes like Am_SAFE exist and are not metals
    assert any(str(x).startswith("Am_SAFE") for s in f.safe_exp_ids for x in str(s).split(";"))


def test_support_exact_and_zero_trees_is_b1(bench, rec, b1):
    cells = [c for c in rec["outer"]["cells"] if c["k_max"] >= 3][:20]
    fit0 = {"model": None, "b1": b1, "mode": {"geometry": "own"}}
    for cell in cells:
        ym = np.full(N_LN, np.nan)
        for p in cell["seqs"]["DOPT_Q"][:3]:
            for m in p:
                ym[m] = 0.1 * m
        obs = SimpleNamespace(y_masked=ym)
        assert np.array_equal(SB.predict_levels(fit0, obs, cell), RF.predict_levels(b1, obs, cell["metals"]), equal_nan=True)


def test_slot_fit_uses_training_labels_only_and_keeps_supports(bench, rec, b1):
    pb, _ = poisoned_bench(bench, rec["outer"]["train_index"], seed=9)
    b1p = RF.fit_reference(pb, rec, pb.Y, "B1")
    a = SB.fit_slot(bench, rec, bench.Y, {"geometry": "own"}, b1)
    b = SB.fit_slot(pb, rec, pb.Y, {"geometry": "own"}, b1p)
    assert a["n_trees"] == b["n_trees"] and a["inner_scores"] == b["inner_scores"]
    cell = next(c for c in rec["outer"]["cells"] if c["k_max"] >= 3)
    ym = np.full(N_LN, np.nan)
    for p in cell["seqs"]["DOPT_Q"][:3]:
        for m in p:
            ym[m] = 0.05 * m
    obs = SimpleNamespace(y_masked=ym)
    la, lb = SB.predict_levels(a, obs, cell), SB.predict_levels(b, obs, cell)
    assert np.array_equal(la, lb, equal_nan=True)
    for m in range(N_LN):
        if np.isfinite(ym[m]):
            assert la[m] == ym[m]


def test_controls_construction(bench, rec):
    geo = SB.geometry_table()
    f = bench.frame
    chem_of = dict(zip(f.cell_id, f.chemotype.astype(str)))
    for seed in SB.GEO_SHUFFLE_SEEDS:
        m = SB.geometry_shuffle_map(bench, seed)
        assert all(k != v for k, v in m["pi"].items())                       # derangement
        assert set(m["src"]) == set(geo)                                     # exactly S06's coverage
        assert all(chem_of[s] != chem_of[cid] for (cid, mm), s in m["src"].items())   # always another chemotype
    tr = RF.training_cells(bench, rec)
    mm, mc = SB.metal_means(rec, tr)
    assert mm.shape == (N_LN, len(SB.GEOM_COLS)) and np.isfinite(mm[:, 0]).sum() >= 12
    cells = [c for c in rec["outer"]["cells"] if c["k_max"] >= 3]
    modes = {"own": {"geometry": "own"}, "shuffled": {"geometry": "shuffled", "map": SB.geometry_shuffle_map(bench, SB.GEO_SHUFFLE_SEEDS[0])},
             "radius_only": {"geometry": "radius_only", "metal_means": mm, "metal_comp": mc}}
    for cell in cells:
        for m_ in cell["metals"]:
            pattern = {k: bool(np.isfinite(SB.geometry_of(cell, m_, md)[0]).all()) for k, md in modes.items()}
            assert len(set(pattern.values())) == 1, (cell["cell_id"], m_, pattern)   # identical missingness
    cfg = RG.get("S09", bench)
    assert ("S06", "S09_d0") in cfg.contrasts and len([p for p in cfg.parts if p.startswith("b:S09")]) == SB.N_DRAWS
    cfg = RG.get("S22", bench)
    assert ("S06", "S22") in cfg.contrasts
