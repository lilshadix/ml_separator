"""Tests for r5/confirm.py, the confirmation-package statistics of the Stage-0 size gate and power worlds.  Synthetic
data and construction-only checks; no corpus score is computed (the composition loader reads the seen confirm-1 table
only to check its shape).

    D:/ml_separator_gh/.venv/Scripts/python.exe -B -m pytest generations/gen21/round5_20261024/tests/test_confirm.py -q -p no:cacheprovider -W ignore
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as SS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r5 import paths  # noqa: E402
from r5 import confirm as CF  # noqa: E402


# ------------------------------------------------------------------------------------------------ sign-flip
@pytest.mark.parametrize("G", [4, 5, 6, 7])
@pytest.mark.parametrize("alpha", [0.025, 0.05, 0.1, 0.2])
def test_signflip_exact_size_symmetric(G, alpha):
    """Under a symmetric null every sign pattern of fixed magnitudes is equally likely.  With distinct subset sums the
    p values over the 2^G patterns are exactly 1/2^G, ..., 1, so exactly floor(alpha 2^G) patterns reach p <= alpha:
    the size is floor(alpha 2^G) / 2^G <= alpha, with no Monte Carlo error."""
    a = np.random.default_rng(G).uniform(0.5, 2.0, G)
    p = CF.signflip_p(CF.sign_matrix(G) * a)
    assert np.allclose(np.sort(p), np.arange(1, 2 ** G + 1) / 2 ** G)
    assert int((p <= alpha + 1e-12).sum()) == math.floor(alpha * 2 ** G)


def test_signflip_minimum_p_and_floor():
    """G <= 5 cannot reach 0.025 (minimum p 1/32); G = 6 can (1/64), section 4.2 'G floors'."""
    for G, pmin in ((5, 1 / 32), (6, 1 / 64), (13, 1 / 8192)):
        assert CF.signflip_p(np.ones((1, G)))[0] == pytest.approx(pmin)
    assert CF.sign_matrix(3)[0].tolist() == [1, 1, 1] and CF.sign_matrix(3).shape == (8, 3)


def test_signflip_mc_size_symmetric_heavy_tailed_null():
    rng = np.random.default_rng(paths.seed_for("test_confirm|signflip_mc"))
    R, G = 20000, 6
    k = int((CF.signflip_p(rng.standard_t(3, size=(R, G))) <= CF.ALPHA + 1e-12).sum())
    lo, hi = CF.clopper_pearson(k, R, conf=0.999)
    assert lo <= 1 / 64 <= hi


def test_signflip_random_branch():
    """G > 13 uses B random flips with p = (1 + #{T* >= T}) / (1 + B)."""
    x = np.abs(np.random.default_rng(1).normal(size=(1, 15))) + 0.1
    p = CF.signflip_p(x, rng=np.random.default_rng(2))[0]
    assert 1 / (CF.N_RANDOM_FLIPS + 1) <= p <= 2 / (CF.N_RANDOM_FLIPS + 1)


# ------------------------------------------------------------------------------------------------ SIGN, t, rules
def test_sign_test_p():
    assert CF.sign_test_p(np.ones((1, 6)))[0] == pytest.approx(1 / 64)
    assert CF.sign_test_p(np.array([[1, 1, 1, 1, 1, 0, 0, -1.0]]))[0] == pytest.approx(7 / 64)   # ties dropped
    G = 8                                    # attained level at 0.025 is P(W = 8) = 1/256 (P(W >= 7) = 9/256 > 0.025)
    p = CF.sign_test_p(CF.sign_matrix(G) * np.linspace(0.5, 1.5, G))
    assert int((p <= CF.ALPHA).sum()) == 1


def test_t_interval_matches_scipy_and_clip():
    x = np.array([[-0.5, -0.2, -0.1, 0.0, 0.1, 0.3]])
    for clip in (None, CF.CLIP):
        xc = x if clip is None else np.clip(x, *clip)
        lo, hi = CF.t_interval(x, clip=clip)
        ref = SS.t.interval(0.95, 5, loc=xc.mean(), scale=SS.sem(xc[0]))
        assert (lo[0], hi[0]) == pytest.approx(ref)


def test_chord_rules():
    h = CF.chord_rule_hits(np.array([-0.30, -0.08, 0.01]), np.array([-0.15, -0.02, 0.20]), CF.chord_refs(-0.1))
    assert h["confirm"].tolist() == [True, False, False]
    assert h["kill"].tolist() == [False, True, True]
    assert h["helps"].tolist() == [True, False, False]
    assert h["harm"].tolist() == [False, False, True]
    assert h["A1"].tolist() == [True, True, False]
    assert {r for _, _, r, g in CF.CHORD_RULES if g} == {"confirm", "kill", "helps", "harm"}


def test_ect_decisions_construction():
    X = np.array([[0.5] * 7, [-0.1] * 7, [0.3, 0.2, -0.1, -0.2, -0.3, 0.1, -0.05]])
    d = CF.ect_decisions(X, X3=X)
    assert d["confirm"].tolist() == [True, False, False]
    assert d["kill"].tolist() == [False, True, True]
    assert d["inconclusive"].tolist() == [False, False, False] and not d["overlap"].any()
    assert d["seq_SH3"].tolist() == [True, False, False]


# ------------------------------------------------------------------------------------------------ CP bound
@pytest.mark.parametrize("k,n", [(0, 10), (3, 10), (10, 10), (70, 2000), (1, 4000), (200, 4000)])
def test_clopper_pearson_matches_exact(k, n):
    lo, hi = CF.clopper_pearson(k, n)
    ci = SS.binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
    assert lo == pytest.approx(ci.low, abs=1e-12) and hi == pytest.approx(ci.high, abs=1e-12)


def test_clopper_pearson_edges_vector_and_flag():
    lo, hi = CF.clopper_pearson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(1 - 0.025 ** 0.1)
    lo, hi = CF.clopper_pearson(10, 10)
    assert hi == 1.0 and lo == pytest.approx(0.025 ** 0.1)
    lo, hi = CF.clopper_pearson(np.array([0, 5, 10]), np.array([10, 10, 10]))
    assert lo.shape == hi.shape == (3,) and (lo <= hi).all()
    assert CF.clopper_pearson(200, 4000)[0] > CF.SIZE_FLAG     # observed 0.050 at 4,000 reps: flagged
    assert CF.clopper_pearson(120, 4000)[0] < CF.SIZE_FLAG     # observed 0.030: not flagged


# ------------------------------------------------------------------------------------------------ draws, inputs
def test_two_stage_construction():
    sdw, icc = 0.1, 0.5
    Z = CF.two_stage(np.random.default_rng(3), 200000, (4, 2), sdw, icc)
    assert Z.shape == (200000, 6)
    c, tot = np.cov(Z.T), sdw ** 2 / (1 - icc)
    assert c[0, 1] / tot == pytest.approx(icc, abs=0.01)          # same chemotype
    assert abs(c[0, 4]) / tot < 0.01                               # different chemotypes
    assert np.diag(c).mean() / tot == pytest.approx(1, abs=0.01)


def test_iid_and_tau_noise():
    rng = np.random.default_rng(4)
    pool = np.array([-1.0, 2.0, 3.0])
    X = CF.iid(rng, 1000, 5, "swap", pool=pool)
    assert set(np.unique(np.abs(X))) <= {1.0, 2.0, 3.0} and (X < 0).any()
    assert CF.iid(rng, 3, 4, "normal", 0.1).shape == (3, 4)
    assert CF.tau_noise(X, 0.0, rng) is X


def test_compositions_from_confirm1_file():
    comps, sdw, t = CF.load_compositions()
    assert len(comps) == 7 and sorted({sum(c) for c in comps}) == [6, 8, 13]
    assert sdw == (0.05, 0.08, 0.1) and len(t) == 63


def test_beta_frozen_parsing():
    assert CF.parse_beta_frozen({"beta_frozen": -0.1}) == -0.1
    assert CF.parse_beta_frozen({"beta_frozen": {"beta": -0.08}}) == -0.08
    assert CF.parse_beta_frozen({"value": -0.07}) == -0.07
    with pytest.raises(KeyError):
        CF.parse_beta_frozen({"beta_frozen": "x"})
    missing = paths.R5 / "results" / "stage0" / "_no_such_file.json"
    with pytest.raises(FileNotFoundError):                         # strict by default (s0-mechanics-10)
        CF.load_beta_frozen(missing)
    d = CF.load_beta_frozen(missing, allow_todo=True)
    assert d["value"] == CF.BETA_R4_FOLD_MEAN and d["todo"].startswith("TODO")


# ------------------------------------------------------------------------------------------------ finite inputs
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_unit_values_raise(bad):
    """A NaN or +-inf unit value must never reach the sealed C-ECT test (s0-statistics-1): with +inf or NaN the
    sign-flip would otherwise return p = 0 and CONFIRMED."""
    x = np.array([[-0.2, -0.1, -0.3, bad, -0.1, -0.05, -0.4]])
    for fn in (CF.signflip_p, CF.sign_test_p, CF.t_interval, CF.ect_decisions):
        with pytest.raises(ValueError, match="non-finite"):
            fn(x)
    with pytest.raises(ValueError, match="non-finite"):
        CF.ect_decisions(np.abs(np.nan_to_num(x, nan=0.1, posinf=0.1, neginf=0.1)), X3=x)
