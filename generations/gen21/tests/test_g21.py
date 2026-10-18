"""gen21 validity tests.  Synthetic arrays appear only in ``test_hb_recovers_synthetic`` and are
labelled SYNTHETIC; they are never used for any reported number."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from g21 import cache as C  # noqa: E402
from g21 import evaluate as E  # noqa: E402
from g21 import methods as M  # noqa: E402
from g21 import paths  # noqa: E402
from gen20k import adapters as AD  # noqa: E402
from gen20k import cov as CV  # noqa: E402
from gen20k import data as D  # noqa: E402

FOLD = ("BP", 130363, 2)


@pytest.fixture(scope="module")
def bench():
    return D.load_bench()


@pytest.fixture(scope="module")
def rec():
    if not C.cache_path(*FOLD).exists():
        pytest.skip("cache not built")
    return C.load(*FOLD)


def test_ridge_obs_latent_equals_gen20_ridge_line(bench, rec):
    blk = rec["outer"]
    for cell in blk["cells"][:40]:
        seq = cell["seqs"]["DOPT_Q"]
        for k in range(len(seq) + 1):
            o = E.observe(bench.Y[cell["ci"]], cell, seq, k)
            c, _ = M.ridge_obs(bench.basis, o.support, blk["prior"], blk["ridge_lambda"], blk["s2"])
            r = AD.ridge_line(bench.basis, o.support, blk["prior"], blk["ridge_lambda"],
                              CV.noise_matrix(o.support.pairs, blk["s2"], "shared"))
            assert np.allclose(c, r, atol=1e-10)


def test_observed_exposes_only_read_metals(bench, rec):
    cell = rec["outer"]["cells"][0]
    seq = cell["seqs"]["DOPT_Q"]
    o = E.observe(bench.Y[cell["ci"]], cell, seq, min(3, len(seq)))
    finite = set(np.flatnonzero(np.isfinite(o.y_masked)))
    assert finite == set(o.read)
    assert not any(a in o.read and b in o.read for a, b in o.query)


def test_intervention_hidden_values_do_not_move_predictions(bench, rec):
    blk = dict(rec["outer"])
    blk["fitted"] = {**M.attempt1_fit(bench, rec), **M.attempt2_fit(bench, rec)}
    methods = {**E.gen20_baselines(bench), **M.attempt1_methods(bench, reselect=True), **M.attempt2_methods(bench)}
    rng = np.random.default_rng(1)
    for cell in blk["cells"][:25]:
        seq = cell["seqs"]["DOPT_Q"]
        for k in range(len(seq) + 1):
            o1 = E.observe(bench.Y[cell["ci"]], cell, seq, k)
            o2 = E.observe(E.scramble_hidden(bench.Y[cell["ci"]], o1.read, rng), cell, seq, k)
            for fn in methods.values():
                assert np.array_equal(np.asarray(fn(blk, cell, o1, k)), np.asarray(fn(blk, cell, o2, k)))


def test_fold_isolation_every_cached_fold(bench):
    files = sorted(paths.CACHE.glob("*.pkl"))
    if not files:
        pytest.skip("cache not built")
    for p in files:
        d, s, f = p.stem.split("_")
        E.assert_test_isolation(bench, C.load(d, int(s), int(f)))


def test_fitted_parameters_use_training_cells_only(bench, rec):
    """Poison the held-out cells' log D: the HB fit must not change."""
    from types import SimpleNamespace
    te = rec["outer"]["test_index"]
    a = M.attempt2_fit(bench, rec)
    Yp = bench.Y.copy()
    Yp[te] = np.where(np.isfinite(Yp[te]), 99.0, np.nan)
    poisoned = SimpleNamespace(Y=Yp, groups=bench.groups, n_obs=bench.n_obs, basis=bench.basis)
    b = M.attempt2_fit(poisoned, rec)
    for n in a:
        assert np.allclose(a[n].x, b[n].x)


def test_hb_recovers_synthetic():
    """SYNTHETIC data (code check only): the ML fit recovers the generating noise level."""
    rng = np.random.default_rng(0)
    bench = D.load_bench()
    B = bench.basis
    true_S = np.array([[0.2, 0.0], [0.0, 0.05]])
    Y = []
    for _ in range(400):
        th = rng.multivariate_normal([-0.1, -0.15], true_S)
        Y.append(3.0 + th @ B + rng.normal(0, 0.1, size=14))
    Y = np.array(Y)
    m = M.HBModel(B, use_gp=False).fit(Y, np.ones(len(Y)))
    assert abs(np.sqrt(m.sig2) - 0.1) < 0.01
    assert np.allclose(m.m, [-0.1, -0.15], atol=0.06)


def test_hbg_pooling_synthetic():
    """SYNTHETIC data (code check only, never a result): series drawn from the HBG model itself.
    The ML fit recovers the cell-level shape variance, and pooling staggered sibling reads must
    beat the cell's own reads when the model is true."""
    from gen20k.data import Z_VEC
    bench = D.load_bench()
    B = np.asarray(bench.basis, float)
    rng = np.random.default_rng(5)
    Sg, Sc = np.diag([0.2, 0.08]), np.diag([0.005, 0.005])
    K = 0.99 ** np.abs(Z_VEC[:, None] - Z_VEC[None, :]).astype(float)

    def series(n):
        th = rng.multivariate_normal([-0.05, -0.18], Sg)
        dg = rng.multivariate_normal(np.zeros(14), 0.5 * K)
        return np.array([rng.normal() * 2 + (th + rng.multivariate_normal([0, 0], Sc)) @ B + dg
                         + rng.multivariate_normal(np.zeros(14), 0.05 * K) + rng.normal(0, 0.09, 14) for _ in range(n)])
    Y, g = [], []
    for j in range(80):
        s = series(int(rng.integers(1, 8)))
        Y.append(s)
        g += [j] * len(s)
    mdl = M.HBGroupModel(B).fit(np.vstack(Y), np.ones(len(g)), np.array(g))
    assert np.allclose(np.diag(np.array(mdl.params()["S_cell"])), 0.005, atol=0.003)

    class Sup:
        def __init__(self, pairs, vals):
            self.pairs, self.values, self.k = pairs, vals, len(pairs)
    sets = [[0, 3, 8, 13], [1, 4, 9, 12], [2, 5, 10, 7], [0, 6, 11, 13], [1, 3, 9, 12], [2, 4, 8, 10]]
    eo, ep = [], []
    for _ in range(60):
        s = series(6)
        sups = [Sup([(st[0], m) for m in st[1:]], [s[i][st[0]] - s[i][m] for m in st[1:]]) for i, st in enumerate(sets)]
        for i in range(6):
            vo = mdl.predict(sups[i], [])
            vp = mdl.predict(sups[i], [x for j, x in enumerate(sups) if j != i])
            for a in range(13):
                y = s[i][a] - s[i][a + 1]
                eo.append((vo[a] - vo[a + 1]) - y)
                ep.append((vp[a] - vp[a + 1]) - y)
    assert np.mean(np.square(ep)) < 0.9 * np.mean(np.square(eo))


def test_series_stagger_is_label_free_and_legal(bench, rec):
    from g21 import policies as P
    blk = dict(rec["outer"])
    seqs = {c["ci"]: P.series_stagger(blk, c) for c in blk["cells"]}
    P._MEMO.clear()
    Yp = np.where(np.isfinite(bench.Y), 0.0, np.nan)      # labels changed: the policy must not move
    blk2 = dict(blk, Y=Yp)
    for c in blk["cells"]:
        assert P.series_stagger(blk2, c) == seqs[c["ci"]]
        read = {m for p in seqs[c["ci"]] for m in p}
        assert not any(a in read and b in read for a, b in c["query"])


@pytest.mark.parametrize("which", ["attempt1", "attempt3"])
def test_other_fits_use_training_cells_only(bench, rec, which):
    """Poison the held-out cells' log D: attempt 1's inner re-selection and attempt 3's HBG / HB2+GP
    maximum-likelihood fits must not change."""
    from types import SimpleNamespace
    te = rec["outer"]["test_index"]
    Yp = bench.Y.copy()
    Yp[te] = np.where(np.isfinite(Yp[te]), -42.0, np.nan)
    poisoned = SimpleNamespace(Y=Yp, groups=bench.groups, n_obs=bench.n_obs, basis=bench.basis, frame=bench.frame)
    if which == "attempt1":
        a, b = M.attempt1_fit(bench, rec), M.attempt1_fit(poisoned, rec)
        assert (a["noise_factor"], a["ridge_lambda"]) == (b["noise_factor"], b["ridge_lambda"])
        assert a["inner_mse"] == b["inner_mse"]
    else:
        a, b = M.attempt3_fit_all(bench, rec), M.attempt3_fit_all(poisoned, rec)
        for n in a:
            assert np.allclose(a[n].x, b[n].x)
