"""LT-22: MET-04/MET-07 weights sum to 1 per subset and per bootstrap draw (chemotype and publication draws and
the LOCO jackknife), and the per-draw balanced statistics equal Σ w* f with those weights.  Synthetic tables."""
from __future__ import annotations

import numpy as np
import pytest

from gen201 import inference as I
from gen201 import metrics as M

import g201_synth as S


@pytest.fixture(scope="module")
def tab():
    cells = S.make_cells(12, seed=6, max_ext=3)
    logd, _ = S.make_tables(cells, seeds=(1, 2, 3))
    return logd


def test_weights_sum_to_one_on_every_subset(tab):
    t = tab[tab.split_seed == 1]
    rng = np.random.default_rng(0)
    assert M.balanced_weights(t.extractant, t.publication_id, t.ci).sum() == pytest.approx(1.0, abs=1e-14)
    for _ in range(25):
        sub = t[rng.random(len(t)) < rng.uniform(0.1, 0.9)]
        if not len(sub):
            continue
        w = M.balanced_weights(sub.extractant, sub.publication_id, sub.ci)
        assert w.sum() == pytest.approx(1.0, abs=1e-13) and (w > 0).all()
    for by in ("chemotype", "publication_id", "metal"):
        for _, g in t.groupby(by):
            assert M.balanced_weights(g.extractant, g.publication_id, g.ci).sum() == pytest.approx(1.0, abs=1e-13)


@pytest.mark.parametrize("unit", ["chemotype", "publication"])
def test_weights_sum_to_one_per_draw(tab, unit):
    F = I.Frame(tab)
    names = F.clusters(unit)
    D = I.draw_matrix(unit, names, 300)
    f = np.abs(F.y - F.pred("A")[0])
    for r in range(60):
        w = I.balanced_draw_weights(F, unit, D[r])
        assert w.sum() == pytest.approx(1.0, abs=1e-12) and (w >= 0).all()
        # weights vanish exactly on rows of undrawn clusters
        _, K = F.cluster(unit)
        assert np.all(w[D[r][K] == 0] == 0)
        # the engine's per-draw MAE_bal is Σ w* |e|
        v = F.bal(unit, D[r:r + 1], f)[0]
        assert v == pytest.approx((w * f).sum(), abs=1e-12)
    G = len(names)
    L = np.ones((G, G)) - np.eye(G)
    for g in range(G):
        assert I.balanced_draw_weights(F, unit, L[g]).sum() == pytest.approx(1.0, abs=1e-12)


def test_point_weights_equal_metrics_weights(tab):
    F = I.Frame(tab)
    for unit in ("chemotype", "publication"):
        w = I.balanced_draw_weights(F, unit, np.ones(len(F.clusters(unit))))
        assert np.allclose(w, F.w_bal, atol=1e-15)


def test_r2_balanced_draws_use_recomputed_weights(tab):
    F = I.Frame(tab)
    names = F.clusters("chemotype")
    D = I.draw_matrix("chemotype", names, 300)
    p = F.pred("A")[0]
    e = F.y - p
    for r in range(10):
        w = I.balanced_draw_weights(F, "chemotype", D[r])
        yw = (w * F.y).sum()
        direct = 1 - (w * e * e).sum() / (w * (F.y - yw) ** 2).sum()
        v = I.seed_values(F, "chemotype", D[r:r + 1], "MET07", "A")[0, 0]
        assert v == pytest.approx(direct, abs=1e-12)
