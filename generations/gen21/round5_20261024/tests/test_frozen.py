"""Tests for r5.frozen (Stage-0 inputs, CLAUDE_CODE_PROMPT.md 4.3 items 9 and 12): the all-cell fold record contains
every cell, its copy of g21.cache.build_fold reproduces a stored dev cache, a rebuild is bit-identical (record pickles,
beta_frozen, the external DOPT_Q pickle), and the external read policy behaves as sealed.  Construction checks only:
no held-out prediction is compared with a target.  The stored-object tests need scripts/s0_beta_frozen.py and
scripts/s0_external_dopt.py to have run.

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B -m pytest generations/gen21/round5_20261024/tests/test_frozen.py \
        -q -p no:cacheprovider -W ignore
"""
from __future__ import annotations

import copy
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r5 import paths  # noqa: E402,F401
from r5 import frozen as FR  # noqa: E402
from g21 import cache as GC  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import cell_view  # noqa: E402
from gen20k.splits import _redeal_inner, folds_for, inner_seed  # noqa: E402


@pytest.fixture(scope="module")
def bench():
    return D.load_bench()


@pytest.fixture(scope="module")
def stored():
    out = {}
    for d in ("BPG", "BP"):
        p = FR.record_path(d)
        if not p.exists():
            pytest.skip("run scripts/s0_beta_frozen.py first")
        out[d] = p.read_bytes()
    return out


# ------------------------------------------------------------------------------------------
# synthetic / construction-only
# ------------------------------------------------------------------------------------------
def test_declared_inner_seed():
    assert inner_seed(FR.DECLARED_SEED, FR.DECLARED_FOLD) == 104729 * 31 + 17 == 3246616


def test_read_set_and_compare_synthetic():
    seq = [(0, 2), (0, 4), (0, 6), (0, 8)]
    assert FR.read_set(seq, 2) == {0, 2}
    assert FR.read_set(seq, 3) == {0, 2, 4}
    assert FR.read_set(seq, 4) == {0, 2, 4, 6}
    for bad in (1, 6):
        with pytest.raises(ValueError):
            FR.read_set(seq, bad)
    a = {"x": np.array([1.0, np.nan]), "y": [1, (2, 3)], "z": 0.5, "s": "t"}
    b = copy.deepcopy(a)
    assert FR.compare(a, b) == ([], 0.0)
    b["z"] = 0.75
    assert FR.compare(a, b) == ([], 0.25)
    b["y"] = [1, (2, 4)]
    assert FR.compare(a, b)[0]
    c = copy.deepcopy(a)
    c["x"] = np.array([1.0, 2.0])
    assert FR.compare(a, c)[0] == ["/x: NaN pattern"]
    with pytest.raises(ValueError):                       # La, Pr: no strict-adjacent pair, so no query
        FR.external_sequence({"kmax_cap": 6, "C_all": np.eye(14), "noise_var": 0.1}, FR.design_for_metals([0, 2]))


def test_redeal_reproduces_dev_inner_split(bench):
    """_redeal_inner (BPG's dealing) reproduces the dev folds' single inner split of both designs (gen13's for BP)."""
    for design in ("BP", "BPG"):
        for f in folds_for(bench.frame, design, seeds=[104729]):
            g = _redeal_inner(bench.frame, f, f.train_index)
            assert np.array_equal(g.inner_train_index, f.inner_train_index)
            assert np.array_equal(g.inner_validation_index, f.inner_validation_index)


def test_all_cells_fold(bench):
    n = len(bench.frame)
    for design in ("BP", "BPG"):
        f = FR.all_cells_fold(bench, design)
        assert f.design == design and (f.seed, f.fold) == (FR.DECLARED_SEED, FR.DECLARED_FOLD)
        assert np.array_equal(f.train_index, np.arange(n)) and len(f.test_index) == 0
        assert set(f.inner_train_index).isdisjoint(f.inner_validation_index)
        assert set(f.inner_train_index) | set(f.inner_validation_index) == set(range(n))
    with pytest.raises(KeyError):
        FR.all_cells_fold(bench, "A")


def test_build_fold_record_reproduces_dev_cache(bench):
    """The copied build_fold, on a dev fold with test cells, equals the stored gen21 cache record."""
    f = folds_for(bench.frame, "BPG", seeds=[104729])[0]
    rec, _ = FR.build_fold_record(bench, f)
    ref = {k: v for k, v in GC.load("BPG", 104729, 0).items() if k != "seconds"}
    bad, mx = FR.compare(rec, ref)
    assert not bad, bad[:5]
    assert mx <= 1e-12, mx


# ------------------------------------------------------------------------------------------
# stored Stage-0 objects
# ------------------------------------------------------------------------------------------
def test_record_contains_every_cell(bench, stored):
    n = len(bench.frame)
    c3 = FR.c3_cells(bench)
    assert len(c3) == 229
    for d, b in stored.items():
        rec = pickle.loads(b)
        assert (rec["design"], rec["split_seed"], rec["fold"]) == (d, FR.DECLARED_SEED, FR.DECLARED_FOLD)
        assert np.array_equal(rec["outer"]["train_index"], np.arange(n))
        assert len(rec["outer"]["test_index"]) == 0 and rec["outer"]["cells"] == []
        val = np.sort(np.concatenate([np.asarray(blk["val_index"]) for blk in rec["inner"]]))
        assert np.array_equal(val, np.arange(n))                # every cell is inner-validated exactly once
        inner_ci = [c["ci"] for blk in rec["inner"] for c in blk["cells"]]
        assert len(inner_ci) == len(set(inner_ci)) and set(c3) <= set(inner_ci)
        chk = FR.record_checks(bench, rec)
        assert chk["ok"] and chk["orphan_cells"] == 0 and chk["inner_mask_leaks"] == 0


def test_rerun_bit_identical(bench, stored):
    rec, _ = FR.build_fold_record(bench, FR.all_cells_fold(bench, "BPG"))
    assert FR.dump_bytes(rec) == stored["BPG"]
    js = json.loads((FR.STAGE0 / "beta_frozen.json").read_bytes())
    fit = FR.fit_beta(bench, rec)
    assert fit["beta"] == js["beta_frozen"] and fit["lambda"] == js["lambda"]
    assert fit["inner_scores"] == js["fits"]["primary_BPG"]["inner_scores"]
    with threadpool_limits(limits=4):                       # the ambient BLAS thread count does not matter
        assert FR.fit_beta(bench, rec)["beta"] == js["beta_frozen"]
    if not FR.EXT_PATH.exists():
        pytest.skip("run scripts/s0_external_dopt.py first")
    ext = FR.build_external(bench, rec, record_sha256=FR.sha256_bytes(stored["BPG"]))
    assert FR.dump_bytes(ext) == FR.EXT_PATH.read_bytes()


def test_external_policy(bench, stored):
    if not FR.EXT_PATH.exists():
        pytest.skip("run scripts/s0_external_dopt.py first")
    ext = FR.load_external()
    rec = pickle.loads(stored["BPG"])
    assert ext["record_sha256"] == FR.sha256_bytes(stored["BPG"])
    assert ext["n_train"] == len(bench.frame) and ext["noise_var"] == 2.0 * ext["s2_all"]
    assert np.array_equal(ext["C_all"], rec["outer"]["cov"]) and ext["s2_all"] == rec["outer"]["s2"]
    assert np.allclose(ext["C_all"], ext["C_all"].T) and np.linalg.eigvalsh(ext["C_all"]).min() > 0
    c3 = FR.c3_cells(bench)
    assert len(ext["corpus_c3_sequences"]) == len(c3)
    for ci in c3:
        cv = cell_view(bench, ci)
        dz = FR.design_for_metals(cv.metals, cell_id=cv.cell_id)
        seq = FR.external_sequence(ext, dz)
        assert seq == ext["corpus_c3_sequences"][cv.cell_id]
        assert len(seq) == min(dz.k_max, FR.KMAX)
        r3, r4 = FR.read_set(seq, 3), FR.read_set(seq, 4)
        assert len(r3) == 3 and len(r4) == 4 and r3 < r4 <= set(cv.metals)
        assert not any(a in r4 and b in r4 for a, b in dz.query)
