"""Tests for stage-1 block A (r4.slots_a): support exactness, lambda = infinity returns B1, fit poisoning,
class coverage, control construction (matched capacity, non-identity permutations, class counts preserved by the
chemotype-block shuffle).  None compares a held-out prediction with its target.

    .venv/Scripts/python.exe -m pytest generations/gen21/round4_20261022/tests -q -p no:cacheprovider -W ignore
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r4 import refs as RF  # noqa: E402
from r4 import registry as RG  # noqa: E402
from r4 import slots_a as SA  # noqa: E402
from r4.leakage import poisoned_bench  # noqa: E402
from g21 import cache as C  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import N_LN  # noqa: E402

FOLD = ("BP", 130363, 2)


@pytest.fixture(scope="module")
def bench():
    return D.load_bench()


@pytest.fixture(scope="module")
def rec():
    return C.load(*FOLD)


@pytest.fixture(scope="module")
def b1(bench, rec):
    return RF.fit_reference(bench, rec, bench.Y, "B1")


def _cls(cell):
    return SA.donor_class(str(cell["extractant"]))


def test_capacity_and_shapes():
    assert len(SA.columns(SA.SPECS["S02"])) == len(SA.columns(SA.SPECS["S03"])) == 15
    assert len(SA.columns(SA.SPECS["S01"])) == 7
    sh = SA.shapes()
    P = np.vstack([sh[f"P{d}"] for d in range(2, 6)])
    assert np.allclose(P @ P.T / N_LN, np.eye(4), atol=1e-10)                 # orthonormal
    assert "e_Gd" in SA.SPECS["S03"]["dev"] and "e_Gd" in SA.SPECS["S02"]["dev"]   # same class-specific Gd term
    assert "e_Gd" not in SA.PERMUTED_SHAPES
    from r3 import interp
    x = interp.X_R
    assert np.allclose(P @ np.ones(N_LN), 0, atol=1e-9) and np.allclose(P @ (x - x.mean()), 0, atol=1e-9)
    for s in SA.PERM_SEEDS:
        assert (SA.perm(s) != np.arange(N_LN)).any()
    assert len({tuple(SA.perm(s)) for s in SA.PERM_SEEDS}) == len(SA.PERM_SEEDS)


def test_every_cohort_extractant_has_a_class(bench):
    cls = [SA.donor_class(str(s)) for s in bench.frame.extractant.unique()]
    assert set(cls) <= set(SA.CLASSES)
    import pandas as pd
    cells = pd.read_csv(Path(__file__).resolve().parents[1] / "results" / "provenance" / "cells.csv")
    c3 = cells[cells.c3].drop_duplicates("extractant_smiles")
    counts = pd.Series([SA.donor_class(s) for s in c3.extractant_smiles]).value_counts().to_dict()
    assert counts == {"DGA": 40, "OTHER": 20, "SOFT_N": 19}


def test_shuffle_preserves_chemotype_class_counts(bench):
    def per_chem(m):
        f = bench.frame
        return sorted(pd_major(f, m))

    def pd_major(f, m):
        out = []
        for ch, g in f.groupby("chemotype"):
            cls = [m[str(e)] for e in g.extractant]
            out.append(max(set(cls), key=cls.count))
        return out
    true = {str(e): SA.donor_class(str(e)) for e in bench.frame.extractant}
    for s in SA.SHUFFLE_SEEDS:
        m = SA.shuffled_class_map(bench, s)
        assert per_chem(m) == per_chem(true)
        true_major = dict(zip(sorted(bench.frame.chemotype.astype(str).unique()), pd_major(bench.frame, true)))
        shuf_major = dict(zip(sorted(bench.frame.chemotype.astype(str).unique()), pd_major(bench.frame, m)))
        assert shuf_major != true_major                              # the chemotype-level assignment really moves
        for ch, g in bench.frame.groupby("chemotype"):            # every extractant of a chemotype shares its class
            assert len({m[str(e)] for e in g.extractant}) == 1


def test_support_exact_and_infinite_lambda_is_b1(bench, rec, b1):
    for sid in ("S01", "S02", "S03"):
        spec = SA.SPECS[sid]
        rng = np.random.default_rng(3)
        fit = {"spec": spec, "coef": rng.normal(size=len(SA.columns(spec))), "b1": b1}
        zero = dict(fit, coef=np.zeros(len(SA.columns(spec))))
        for cell in rec["outer"]["cells"][:40]:
            if cell["k_max"] < 3:
                continue
            seq = cell["seqs"]["DOPT_Q"]
            ym = np.full(N_LN, np.nan)
            for p in seq[:3]:
                for m in p:
                    ym[m] = rng.normal()
            obs = SimpleNamespace(y_masked=ym)
            lv = SA.predict_levels(fit, obs, cell["metals"], _cls(cell))
            for m in range(N_LN):
                if np.isfinite(ym[m]):
                    assert lv[m] == ym[m]
            assert np.array_equal(SA.predict_levels(zero, obs, cell["metals"], _cls(cell)),
                                  RF.predict_levels(b1, obs, cell["metals"]), equal_nan=True)


def test_slot_fit_uses_training_labels_only(bench, rec, b1):
    pb, _ = poisoned_bench(bench, rec["outer"]["train_index"], seed=5)
    b1p = RF.fit_reference(pb, rec, pb.Y, "B1")
    for sid in ("S01", "S02"):
        a = SA.fit_slot(bench, rec, bench.Y, SA.SPECS[sid], b1, _cls)
        b = SA.fit_slot(pb, rec, pb.Y, SA.SPECS[sid], b1p, _cls)
        assert a["lambda"] == b["lambda"] and np.array_equal(a["coef"], b["coef"]) and a["inner_scores"] == b["inner_scores"]


def test_registry_parts_and_contrasts(bench):
    cfg = RG.get("S04", bench)
    assert cfg.parts[:3] == ("hbk", "ref:B1", "ref:B2") and len([p for p in cfg.parts if p.startswith("slot:S04")]) == 5
    assert ("S02", "S04_d0") in cfg.contrasts and cfg.slot_cols["S02"] == "slot:S02"
    cfg = RG.get("S21:S01", bench)
    assert cfg.cid == "S21-S01" and ("S01", "S21_d4") in cfg.contrasts
    for c in ("S01", "S02", "S03"):
        cfg = RG.get(c, bench)
        assert (c, "B1") in cfg.contrasts and (c, "B2") in cfg.contrasts and set(cfg.methods) >= {"B0", "B1", "B2", c}
