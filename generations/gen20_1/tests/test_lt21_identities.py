"""LT-21: identities I1-I6 on synthetic arrays (IMPLEMENTATION_PLAN.md §5), checked through gen201.metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gen201 import metrics as M

RNG = np.random.default_rng(20261016)


def _cells(n_cells=40, seeds=(1, 2, 3, 4, 5)):
    rows = []
    for c in range(n_cells):
        k = int(RNG.integers(2, 9))
        ms = np.sort(RNG.choice(14, size=k, replace=False))
        lvl = RNG.normal(0, 1.5)
        for m in ms:
            rows.append({"ci": c, "cell_id": f"c{c}", "extractant": f"E{c % 9}", "chemotype": f"g{(c % 9) % 4}",
                         "publication_id": f"p{c % 7}", "im": int(m), "metal": str(m),
                         "y": lvl + 0.1 * m + RNG.normal(0, 0.3)})
    base = pd.DataFrame(rows)
    parts = []
    for s in seeds:
        t = base.copy()
        t["split_seed"] = s
        t["P"] = t["y"] + RNG.normal(0, 0.8, len(t)) + t["ci"].map({c: RNG.normal(0, 1) for c in range(n_cells)})
        t["Q"] = RNG.normal(0, 2, len(t))                      # arbitrary predictions
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


@pytest.mark.parametrize("arm", ["P", "Q"])
def test_i1_pooled_equals_level_plus_within(arm):
    t = _cells()
    for s in sorted(t.split_seed.unique()):
        m = M.seed_metrics(M.SeedRows(M._sorted_seed(t, s)), arm)
        rhs = m["s_b"] * m["R2_level"] + m["s_w"] * m["R2_within_cell"]
        assert m["R2_pooled"] == pytest.approx(rhs, abs=1e-12)


def test_i2_within_cell_equals_all_pairs_flat_skill():
    t = _cells(seeds=(1,))
    r = M.SeedRows(M._sorted_seed(t, 1))
    p = r.pred("P")
    m = M.seed_metrics(r, "P")
    num = den = 0.0
    for c in np.unique(r.cell):
        idx = np.flatnonzero(r.cell == c)
        nc = len(idx)
        for a in range(nc):
            for b in range(a + 1, nc):
                ys = r.y[idx[a]] - r.y[idx[b]]
                ps = p[idx[a]] - p[idx[b]]
                num += (ys - ps) ** 2 / nc
                den += ys ** 2 / nc
    assert m["R2_within_cell"] == pytest.approx(1 - num / den, abs=1e-12)


def test_i3_log_sf_r2_factorisation():
    y = RNG.normal(-0.05, 0.2, 300)
    p = y + RNG.normal(0, 0.15, 300)
    t = pd.DataFrame({"split_seed": 1, "ci": np.arange(300) // 3, "ia": np.arange(300) % 3,
                      "ib": np.arange(300) % 3 + 1, "cell_id": "c", "extractant": "E", "chemotype": "g",
                      "publication_id": "p", "pair": "La/Ce", "y": y, "X": p, "FLAT": 0.0})
    s = M.summarise(t, "X")
    f = M.summarise(t, "FLAT")
    assert 1 - s["R2_pooled"] == pytest.approx((1 - s["R2_0"]) * (1 - f["R2_pooled"]), abs=1e-12)
    assert f["R2_pooled"] == pytest.approx(-y.mean() ** 2 / y.var(), abs=1e-12)
    c = M.card(t)
    assert c["y_rms0"] ** 2 == pytest.approx(c["y_sd0"] ** 2 + c["y_mean"] ** 2, abs=1e-14)


def test_i4_seed_mean_r2():
    t = _cells()
    s = M.summarise(t, "P")
    per = s["R2_pooled_per_seed"]
    sse = [((g.y - g.P) ** 2).sum() for _, g in t.groupby("split_seed")]
    y = t[t.split_seed == 1]["y"].to_numpy()
    sst = ((y - y.mean()) ** 2).sum()
    assert s["R2_pooled"] == pytest.approx(1 - np.mean(sse) / sst, abs=1e-12)
    assert s["R2_pooled"] == pytest.approx(np.mean(per), abs=1e-12)


def test_i5_level_error_cancels_in_b():
    t = _cells(seeds=(1,))
    d = M._sorted_seed(t, 1)
    pairs = []
    for c, g in d.groupby("ci"):
        g = g.sort_values("im")
        for (_, a), (_, b) in zip(g.iloc[:-1].iterrows(), g.iloc[1:].iterrows()):
            pairs.append({"split_seed": 1, "ci": c, "ia": a.im, "ib": b.im, "cell_id": a.cell_id,
                          "extractant": a.extractant, "chemotype": a.chemotype, "publication_id": a.publication_id,
                          "y": a.y - b.y, "B": a.P - b.P})
    q = pd.DataFrame(pairs)
    ends = M.endpoint_residuals(d, q, "P").sort_values(["ci", "ia"])
    q = q.sort_values(["ci", "ia"])
    e_sf = (q["y"] - q["B"]).to_numpy()
    assert np.allclose(e_sf, ends["rt_a"] - ends["rt_b"], atol=1e-12)
    # adding any constant per cell to A (a pure level error) leaves B's error unchanged
    shift = d["ci"].map({c: RNG.normal(0, 5) for c in d.ci.unique()})
    d2 = d.assign(P=d["P"] + shift)
    ends2 = M.endpoint_residuals(d2, q, "P").sort_values(["ci", "ia"])
    assert np.allclose(ends2["r_a"] - ends2["r_b"], e_sf, atol=1e-12)


def test_i6_centred_loss_decomposition():
    for _ in range(20):
        n = int(RNG.integers(2, 10))
        y = RNG.normal(0, 1, n)
        c = RNG.normal(0, 1, n)                 # an arbitrary centred-target prediction (its mean is not 0)
        yt = y - y.mean()
        lhs = ((yt - c) ** 2).sum()
        pairs = sum(((y[i] - y[j]) - (c[i] - c[j])) ** 2 for i in range(n) for j in range(i + 1, n))
        assert lhs == pytest.approx(pairs / n + n * c.mean() ** 2, abs=1e-10)
