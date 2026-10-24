"""Tests for r5/common.py on synthetic data (exact sign-flip p against brute force, the sign test, IM t, the
leave-one-publication-out jackknife on a known case, unit statistics, pooled contrasts, the per-item table) plus
construction-only checks of the round-4 rows (counts and keys; no score is computed).

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B -W ignore -m pytest generations/gen21/round5_20261024/tests -q
        -p no:cacheprovider --basetemp generations/gen21/round5_20261024/cache/_pytest   (one command line; temporary
        files stay inside the round-5 folder)
"""
from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats as SS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r5 import paths  # noqa: E402,F401
from r5 import common as C  # noqa: E402


def brute_force_p(x) -> float:
    """Explicit loop over every sign vector: share with sum(s * x) >= sum(x)."""
    x = [float(v) for v in x]
    T = sum(x)
    tol = 1e-10 * sum(abs(v) for v in x)
    hits = tot = 0
    for s in itertools.product((1, -1), repeat=len(x)):
        tot += 1
        hits += sum(si * xi for si, xi in zip(s, x)) >= T - tol
    return hits / tot


# ------------------------------------------------------------------------------------------------------------
# sign-flip
# ------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("G", [1, 2, 3, 5, 8, 11, 13])
def test_sf_pool_exact_matches_brute_force(G):
    rng = np.random.default_rng(G)
    for _ in range(3):
        x = rng.normal(0.3, 1.0, size=G)
        r = C.sf_pool(x)
        assert r["exact"] and r["n_flips"] == 2 ** G and r["seed"] is None
        assert r["p"] == pytest.approx(brute_force_p(x), abs=0)


def test_sf_pool_exact_known_cases():
    x = np.arange(1, 14, dtype=float)                       # all positive: only the identity reaches T
    assert C.sf_pool(x)["p"] == 1 / 2 ** 13
    z = np.array([2.0, 0.0, 0.0, 1.0])                      # zeros are exact ties: 2^2 of the 16 vectors reach T
    assert C.sf_pool(z)["p"] == 4 / 16
    assert C.sf_pool(-x)["p"] == 1.0                        # every vector reaches the minimum
    assert C.sf_pool([])["p"] == 1.0
    d = np.array([3.0, -1.0, 2.0, 2.0, -0.5])
    assert C.sf_pool(d)["p"] == brute_force_p(d)
    assert C.sf_pool(d)["n_ge"] == round(brute_force_p(d) * 32)


def test_sf_pool_random_flips_formula_and_seed():
    x = np.random.default_rng(1).normal(0.2, 1.0, size=20)
    with pytest.raises(ValueError):
        C.sf_pool(x)                                        # G > 13 needs a key
    r = C.sf_pool(x, "unit-test|SF_POOL")
    assert not r["exact"] and r["n_flips"] == 4999 and r["seed"] == paths.seed_for("unit-test|SF_POOL")
    F = np.random.default_rng(r["seed"]).choice((-1.0, 1.0), size=(4999, 20))
    ge = int(((F @ x) >= x.sum() - 1e-10 * np.abs(x).sum()).sum())
    assert r["n_ge"] == ge and r["p"] == (1 + ge) / (1 + 4999)
    assert C.sf_pool(x, "unit-test|SF_POOL") == r          # reproducible
    assert C.sf_pool(x, "other-key")["seed"] != r["seed"]
    assert C.sf_pool(np.abs(x) + 1, "k")["p"] == 1 / 5000   # all positive: only the observed statistic counts


# ------------------------------------------------------------------------------------------------------------
# sign test and IM t
# ------------------------------------------------------------------------------------------------------------
def test_sign_test_exact_binomial_ties_dropped():
    d = np.array([1.0] * 8 + [-1.0] * 2 + [0.0] * 3)
    r = C.sign_test(d)
    assert (r["W"], r["L"], r["T"], r["n"]) == (8, 2, 3, 10)
    exact = sum(math.comb(10, k) for k in range(8, 11)) / 2 ** 10
    assert r["p"] == pytest.approx(exact, rel=1e-12)
    assert C.sign_test(np.zeros(4))["p"] == 1.0
    assert C.sign_test(np.ones(5))["p"] == pytest.approx(1 / 32)
    with pytest.raises(TypeError):
        C.sign_test(np.array([True, False]))
    us = pd.DataFrame({"win": [True, True, False], "loss": [False, False, False], "tie": [False, False, True]})
    assert C.sign_test(us)["p"] == pytest.approx(0.25) and C.sign_test(us)["T"] == 1


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_sign_test_and_sf_pool_refuse_non_finite(bad):
    """s0-statistics-1: a NaN or +-inf unit value raises in every unit test (sign_test used to drop NaN silently)."""
    x = np.array([-0.2, -0.1, -0.3, bad, -0.1, -0.05, -0.4])
    with pytest.raises(ValueError, match="non-finite"):
        C.sign_test(x)
    with pytest.raises(ValueError, match="non-finite"):
        C.sf_pool(x)


def test_im_t_matches_scipy_one_sample_t():
    x = np.array([0.05, 0.12, -0.03, 0.08, 0.02, 0.11])
    r = C.im_t(x)
    ref = SS.ttest_1samp(x, 0.0, alternative="greater")
    assert r["t"] == pytest.approx(ref.statistic) and r["p"] == pytest.approx(ref.pvalue) and r["df"] == 5
    assert C.im_t(np.zeros(4))["p"] == 1.0                 # 0 / 0
    assert math.isnan(C.im_t([0.3])["p"])


# ------------------------------------------------------------------------------------------------------------
# jackknife, top-3, C5 split
# ------------------------------------------------------------------------------------------------------------
def test_lopo_jackknife_of_a_mean_is_the_standard_error():
    v = np.array([0.4, 0.1, 0.25, -0.05, 0.3, 0.2, 0.15])
    t = pd.DataFrame({"publication": [f"p{i}" for i in range(len(v))], "v": v})
    r = C.lopo(t, lambda d: d.v.mean())
    P = len(v)
    se = v.std(ddof=1) / np.sqrt(P)                         # the jackknife SE of a mean equals s / sqrt(n)
    q = SS.t.ppf(0.975, P - 1)
    assert r["P"] == P and r["df"] == P - 1 and r["point"] == pytest.approx(v.mean())
    assert r["se"] == pytest.approx(se, rel=1e-12)
    assert (r["lo"], r["hi"]) == (pytest.approx(v.mean() - q * se), pytest.approx(v.mean() + q * se))
    assert r["min_by"] == "p0" and r["max_by"] == "p3"     # leaving out the largest value gives the minimum


def test_lopo_groups_rows_by_publication():
    t = pd.DataFrame({"publication": ["a", "a", "b", "c", "c", "c"], "v": [1.0, 3.0, 5.0, 0.0, 0.0, 6.0]})
    r = C.lopo(t, lambda d: d.v.sum())
    assert r["values"] == {"a": 11.0, "b": 10.0, "c": 9.0}


def test_top3_share():
    assert C.top3_share([5, 3, 2, 1, -1]) == pytest.approx(1.0)
    assert C.top3_share([4, 3, 2, 1]) == pytest.approx(0.9)
    assert math.isnan(C.top3_share([-1, -2]))


def test_c5_split():
    t = pd.DataFrame({"publication": [paths.C5, paths.C5, "pub_x"], "cell_id": [paths.C5_CELL, "c2", "c3"]})
    s = C.c5_split(t)
    assert (len(s["C5_in"]), len(s["C5_out"]), len(s["C5_cell_out"])) == (3, 1, 2)


# ------------------------------------------------------------------------------------------------------------
# unit statistics and pooled contrasts on a synthetic table
# ------------------------------------------------------------------------------------------------------------
def synthetic() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for s in (1, 2):
        for u, n in (("u1", 4), ("u2", 3), ("u3", 5)):
            for i in range(n):
                y = rng.normal()
                ref = y + rng.normal(0, 0.5)
                cand = ref if u == "u3" else y + rng.normal(0, 0.3)   # u3: identical arms -> an exact tie
                rows.append({"split_seed": s, "unit": u, "y": y, "REF": ref, "CAND": cand, "publication": "p" + u,
                             "chemotype": "c" + u, "extractant": f"e{u}{i % 2}", "cell_id": f"{u}_{i}"})
    return pd.DataFrame(rows)


def test_unit_stats_seed_means_and_ties():
    t = synthetic()
    us = C.unit_stats(t, "CAND", "REF", "unit")
    for u in ("u1", "u2", "u3"):
        g = t[t.unit == u]
        er = np.mean([((h.REF - h.y) ** 2).sum() for _, h in g.groupby("split_seed")])
        ec = np.mean([((h.CAND - h.y) ** 2).sum() for _, h in g.groupby("split_seed")])
        assert us.loc[u, "SSE_ref"] == pytest.approx(er) and us.loc[u, "SSE_cand"] == pytest.approx(ec)
        assert us.loc[u, "dSSE"] == pytest.approx(er - ec) and us.loc[u, "L"] == pytest.approx(np.log(er / ec))
        assert us.loc[u, "n"] == len(g) / 2 and us.loc[u, "n_seeds"] == 2
    assert us.loc["u3", "tie"] and us.loc["u3", "dSSE"] == 0.0 and not us.loc["u3", "win"]
    res = C.unit_tests(us, "unit-test")
    assert res["G"] == 3 and res["SIGN"]["T"] == 1 and res["SF_POOL"]["exact"]


def test_pooled_contrasts_identities():
    t = synthetic()
    assert C.rel_sse(t, "REF", "REF") == 0.0
    t["PERFECT"] = t.y
    assert C.rel_sse(t, "PERFECT", "REF") == 1.0 and C.r2_seedmean(t, "PERFECT") == 1.0
    manual = np.mean([1 - ((g.CAND - g.y) ** 2).sum() / ((g.REF - g.y) ** 2).sum() for _, g in t.groupby("split_seed")])
    assert C.rel_sse(t, "CAND", "REF") == pytest.approx(manual)
    r2 = np.mean([1 - ((g.CAND - g.y) ** 2).sum() / ((g.y - g.y.mean()) ** 2).sum() for _, g in t.groupby("split_seed")])
    assert C.r2_seedmean(t, "CAND") == pytest.approx(r2)
    assert C.mse_seedmean(t, "PERFECT") == 0.0


def test_balanced_mae_is_extractant_then_chemotype_macro():
    t = synthetic()
    man = []
    for _, g in t.groupby("split_seed"):
        e = g.assign(ae=(g.CAND - g.y).abs())
        man.append(e.groupby(["chemotype", "extractant"]).ae.mean().groupby(level=0).mean().mean())
    assert C.bal_mae(t, "CAND") == pytest.approx(np.mean(man))
    assert C.dbal_mae(t, "CAND", "CAND") == 0.0


# ------------------------------------------------------------------------------------------------------------
# per-item table
# ------------------------------------------------------------------------------------------------------------
def item() -> dict:
    d = {k: "n/a" for k, _ in C.ITEM_FIELDS}
    d.update(item="T0 (synthetic)", label="TECHNICAL SMOKE", population={"cells": 3, "rows_per_seed": 6.0},
             metrics=pd.DataFrame({"arm": ["A", "B|x"], "R2": [0.5, float("nan")]}))
    return d


def test_per_item_table_requires_every_field_and_a_label(tmp_path):
    d = item()
    p = tmp_path / "TABLE.md"
    txt = C.per_item_table(d, p)
    raw = p.read_bytes()
    assert b"\r\n" not in raw and raw.decode("utf-8") == txt
    for _, name in C.ITEM_FIELDS:
        assert name in txt
    assert "B\\|x" in txt and "NaN" in txt and "## R², MSE, relative SSE vs B1" in txt
    bad = dict(d)
    bad.pop("leakage")
    with pytest.raises(KeyError):
        C.per_item_table(bad)
    with pytest.raises(ValueError):
        C.per_item_table({**d, "label": "SIGNIFICANT"})


# ------------------------------------------------------------------------------------------------------------
# construction-only checks of the round-4 rows (no score)
# ------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("design", ["BPG", "BP"])
def test_load_rows_construction(design):
    t = C.load_rows(design)
    assert len(t) == 5 * 1802 and set(t.groupby("split_seed").size()) == {1802}
    assert t.cell_id.nunique() == 229 and t.cluster_cpg.nunique() == 13
    assert t[C.KEY].equals(t[C.KEY].sort_values(C.KEY).reset_index(drop=True)) and not t.duplicated(C.KEY).any()
    assert (t.k == 3).all() and (t.k_max >= 3).all()
    pop = C.population(t)
    assert (pop["cells"], pop["units"], pop["rows_per_seed"]) == (229, 13, 1802.0)
    assert len(C.c5_split(t)["C5_in"]) - len(C.c5_split(t)["C5_out"]) == 5 * 148     # section 5.2: 148 rows per seed
