"""MET-01..MET-20 against direct hand computations on a tiny synthetic table (no real target value)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from gen201 import metrics as M

# (ci, extractant, chemotype, publication, im, y)
ROWS = [(0, "E1", "g1", "p1", 0, 1.0), (0, "E1", "g1", "p1", 1, 1.5), (0, "E1", "g1", "p1", 2, 0.5),
        (1, "E1", "g1", "p2", 0, -0.5), (1, "E1", "g1", "p2", 1, 0.25),
        (2, "E2", "g2", "p1", 1, 2.0), (2, "E2", "g2", "p1", 2, 2.5), (2, "E2", "g2", "p1", 3, 3.0),
        (3, "E2", "g2", "p1", 0, -1.0), (3, "E2", "g2", "p1", 1, 0.0),
        (4, "E3", "g2", "p3", 0, 0.3), (4, "E3", "g2", "p3", 1, -0.2)]
PRED = {11: [0.8, 1.2, 0.9, 0.0, 0.0, 1.5, 2.6, 2.4, -0.2, 0.1, 0.0, 0.0],
        22: [1.1, 1.4, 0.2, -0.4, 0.5, 2.2, 2.0, 3.3, -0.8, -0.3, 0.5, 0.1]}


def table(pred=PRED) -> pd.DataFrame:
    parts = []
    for s, p in pred.items():
        t = pd.DataFrame(ROWS, columns=["ci", "extractant", "chemotype", "publication_id", "im", "y"])
        t["split_seed"] = s
        t["cell_id"] = "c" + t["ci"].astype(str)
        t["metal"] = t["im"].map(lambda i: ["La", "Ce", "Pr", "Nd"][i])
        t["X"] = p
        t["Z0"] = 0.0
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


def hand(seed: int) -> dict:
    y = np.array([r[5] for r in ROWS])
    p = np.array(PRED[seed])
    e = y - p
    ext = np.array([r[1] for r in ROWS])
    chem = np.array([r[2] for r in ROWS])
    cell = np.array([r[0] for r in ROWS])
    out = {}
    per_l = {l: np.abs(e[ext == l]).mean() for l in sorted(set(ext))}
    out["MET01"] = np.mean(list(per_l.values()))
    out["MET02"] = np.abs(e).mean()
    g1 = per_l["E1"]
    g2 = (per_l["E2"] + per_l["E3"]) / 2
    out["MET03"] = (g1 + g2) / 2
    # MET-04 weights by hand: L = 3; E1: P = 2 (p1: c0, p2: c1); E2: P = 1, C = 2 (c2, c3); E3: P = 1, C = 1
    w = np.array([1 / (3 * 2 * 1 * 3)] * 3 + [1 / (3 * 2 * 1 * 2)] * 2 + [1 / (3 * 1 * 2 * 3)] * 3
                 + [1 / (3 * 1 * 2 * 2)] * 2 + [1 / (3 * 1 * 1 * 2)] * 2)
    out["w"] = w
    out["MET04_MAE"] = (w * np.abs(e)).sum()
    out["MET04_RMSE"] = math.sqrt((w * e * e).sum())
    out["MET04_BIAS"] = (w * (p - y)).sum()
    out["MET05"] = math.sqrt((e * e).mean())
    sst = ((y - y.mean()) ** 2).sum()
    out["SST"] = sst
    out["SSE"] = (e * e).sum()
    out["R2"] = 1 - (e * e).sum() / sst
    yw = (w * y).sum()
    out["R2_bal"] = 1 - (w * e * e).sum() / (w * (y - yw) ** 2).sum()
    ybar = np.array([y[cell == c].mean() for c in cell])
    pbar = np.array([p[cell == c].mean() for c in cell])
    yt, pt = y - ybar, p - pbar
    out["R2_bal_within"] = 1 - (w * (yt - pt) ** 2).sum() / (w * yt ** 2).sum()
    out["R2_within"] = 1 - ((yt - pt) ** 2).sum() / (yt ** 2).sum()
    out["R2_level"] = 1 - ((ybar - pbar) ** 2).sum() / ((ybar - y.mean()) ** 2).sum()
    out["s_b"] = ((ybar - y.mean()) ** 2).sum() / sst
    out["R2_0"] = 1 - (e * e).sum() / (y * y).sum()
    per_lw = {l: np.abs((yt - pt)[ext == l]).mean() for l in sorted(set(ext))}
    out["MET20"] = np.mean(list(per_lw.values()))
    out["within_2x"] = (np.abs(e) <= math.log10(2)).mean()
    out["within_2x_bal"] = (w * (np.abs(e) <= math.log10(2))).sum()
    out["bias"] = (p - y).mean()
    out["sd_ratio"] = p.std() / y.std()
    out["slope"] = ((p - p.mean()) * (y - y.mean())).sum() / ((p - p.mean()) ** 2).sum()
    out["disp"] = math.sqrt((pt ** 2).sum() / (yt ** 2).sum())
    return out


@pytest.mark.parametrize("seed", [11, 22])
def test_seed_metrics_equal_hand(seed):
    t = table()
    r = M.SeedRows(M._sorted_seed(t, seed))
    got = M.seed_metrics(r, "X")
    h = hand(seed)
    assert np.allclose(r.w_bal, h["w"]) and abs(r.w_bal.sum() - 1) < 1e-15
    pairs = {"MET01_MAE_macro_ext": "MET01", "MET02_MAE_pooled": "MET02", "MET03_MAE_macro_chem": "MET03",
             "MET04_MAE_bal": "MET04_MAE", "MET04_RMSE_bal": "MET04_RMSE", "MET04_Bias_bal": "MET04_BIAS",
             "MET05_RMSE_pooled": "MET05", "R2_pooled": "R2", "R2_balanced": "R2_bal",
             "R2_balanced_within": "R2_bal_within", "R2_within_cell": "R2_within", "R2_level": "R2_level",
             "s_b": "s_b", "R2_0": "R2_0", "MET20_MAE_macro_ext_within": "MET20", "MET15_within_2x": "within_2x",
             "MET15_within_2x_bal": "within_2x_bal", "MET16_bias": "bias", "MET16_sd_ratio": "sd_ratio",
             "MET16_ols_slope": "slope", "MET16_within_dispersion": "disp"}
    for k, hk in pairs.items():
        assert got[k] == pytest.approx(h[hk], abs=1e-13), k
    assert got["s_w"] == pytest.approx(1 - h["s_b"], abs=1e-13)
    assert got["MET12_EV_abs"] == pytest.approx(h["SST"] / len(ROWS) - h["SSE"] / len(ROWS), abs=1e-13)


def test_seed_means_and_i4():
    t = table()
    s = M.summarise(t, "X")
    h11, h22 = hand(11), hand(22)
    assert s["MET01_MAE_macro_ext"] == pytest.approx((h11["MET01"] + h22["MET01"]) / 2, abs=1e-14)
    assert s["MET01_MAE_macro_ext_per_seed"] == pytest.approx([h11["MET01"], h22["MET01"]], abs=1e-14)
    # I4: 1 - mean(SSE)/SST, equal to the mean of per-seed R² on seed-invariant rows
    i4 = 1 - (h11["SSE"] + h22["SSE"]) / 2 / h11["SST"]
    assert s["R2_pooled"] == pytest.approx(i4, abs=1e-14)
    assert s["R2_pooled"] == pytest.approx((h11["R2"] + h22["R2"]) / 2, abs=1e-14)
    assert s["R2_pooled_n_seeds_defined"] == 2 and s["R2_pooled_reason"] == ""


def test_skill_mse_and_per_extractant():
    t = table()
    sk = M.skill_mse(t, "X", "Z0")
    vals = []
    for s in (11, 22):
        tt = t[t.split_seed == s]
        vals.append(1 - ((tt.y - tt.X) ** 2).sum() / (tt.y ** 2).sum())
    assert sk["Skill_MSE"] == pytest.approx(np.mean(vals), abs=1e-14)
    # Skill vs FLAT equals R2_0 (MET-10)
    assert sk["Skill_MSE"] == pytest.approx(M.summarise(t, "X")["R2_0"], abs=1e-14)
    d = M.per_extractant_delta(t, "X", "Z0")
    assert d["delta"].mean() == pytest.approx(M.summarise(t, "Z0")["MET01_MAE_macro_ext"]
                                              - M.summarise(t, "X")["MET01_MAE_macro_ext"], abs=1e-14)
    assert list(d["chemotype"]) == ["g1", "g2", "g2"]


def test_card_met13():
    t = table()
    c = M.card(t)
    y = np.array([r[5] for r in ROWS])
    assert c["n_rows"] == 12 and c["n_cells"] == 5 and c["n_extractants"] == 3 and c["n_chemotypes"] == 2
    assert c["y_sd0"] == pytest.approx(y.std(), abs=1e-15) and c["y_var0"] == pytest.approx(y.var(), abs=1e-15)
    assert c["kish_rows"] == pytest.approx(12 ** 2 / (5 ** 2 + 7 ** 2))
    assert c["kish_ext"] == pytest.approx(9 / 5)
    assert c["largest_chemotype_share"] == pytest.approx(7 / 12)


def test_r2_within_system_groups():
    # three cells of one (publication, extractant) measured at La: one group with 3 condition sets
    rows = []
    for s in (1, 2):
        for c, (yv, pv) in enumerate([(1.0, 0.8), (2.0, 2.5), (4.0, 3.0)]):
            rows.append({"split_seed": s, "ci": c, "cell_id": f"c{c}", "extractant": "E", "chemotype": "g",
                         "publication_id": "p", "im": 0, "metal": "La", "y": yv, "X": pv})
        rows.append({"split_seed": s, "ci": 9, "cell_id": "c9", "extractant": "F", "chemotype": "h",
                     "publication_id": "q", "im": 0, "metal": "La", "y": 0.0, "X": 0.0})
    t = pd.DataFrame(rows)
    r = M.SeedRows(M._sorted_seed(t, 1))
    v, why, num, den = M.r2_within_system(r, r.pred("X"))
    y = np.array([1.0, 2.0, 4.0]); p = np.array([0.8, 2.5, 3.0])
    yc, pc = y - y.mean(), p - p.mean()
    assert v == pytest.approx(1 - ((pc - yc) ** 2).sum() / (yc ** 2).sum(), abs=1e-14) and why == ""
    # the single-cell group of F is excluded (< 3 condition sets)


def test_endpoint_residuals_and_met17():
    t = table()
    # one pair per cell of consecutive metals
    pr = []
    for s in (11, 22):
        tt = t[t.split_seed == s].set_index(["ci", "im"])
        for ci, ia, ib in [(0, 0, 1), (0, 1, 2), (1, 0, 1), (2, 1, 2), (2, 2, 3), (3, 0, 1), (4, 0, 1)]:
            a, b = tt.loc[(ci, ia)], tt.loc[(ci, ib)]
            pr.append({"split_seed": s, "ci": ci, "ia": ia, "ib": ib, "extractant": a.extractant,
                       "chemotype": a.chemotype, "publication_id": a.publication_id, "cell_id": a.cell_id,
                       "y": a.y - b.y, "B": a.X - b.X})
    q = pd.DataFrame(pr)
    ends = M.endpoint_residuals(t, q, "X")
    # I5: B's error equals r_a - r_b and equals the cell-centred difference
    e_sf = (q.sort_values(["split_seed", "ci", "ia"])["y"] - q.sort_values(["split_seed", "ci", "ia"])["B"]).to_numpy()
    ends = ends.sort_values(["split_seed", "ci", "ia"])
    assert np.allclose(e_sf, ends["r_a"] - ends["r_b"], atol=1e-14)
    assert np.allclose(e_sf, ends["rt_a"] - ends["rt_b"], atol=1e-14)
    m = M.met17(t, q, "X")
    # hand λ for seed 11
    tt = t[t.split_seed == 11]
    r = (tt.y - tt.X).to_numpy()
    rbar = tt.assign(r=r).groupby("ci")["r"].transform("mean").to_numpy()
    assert m["lambda_per_seed"][0] == pytest.approx((rbar ** 2).sum() / (r ** 2).sum(), abs=1e-14)


def test_direction_accuracy_tie_rule():
    acc, n, nz = M.direction_accuracy([0.5, -0.5, 0.2, 0.0, -0.05], [0.1, 0.0, -0.3, 1.0, -0.2], 0.0)
    # rows with |y| > 0: four; scores 1, 0.5 (zero prediction), 0, 1
    assert (acc, n, nz) == (pytest.approx((1 + 0.5 + 0 + 1) / 4), 4, 1)
    acc, n, _ = M.direction_accuracy([0.5, -0.1, 0.05], [0.1, -0.2, 0.3], 0.1)   # |y| >= 0.1: two rows
    assert n == 2 and acc == 1.0
    assert math.isnan(M.direction_accuracy([0.0], [1.0], 0.0)[0])
