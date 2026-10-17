"""LT-19 (S5 noise reads training cells only) and the pure functions of the models-side scripts, on synthetic
arrays only (no real target statistic is computed here): g201_describe (MET-13 card, s_b/s_w, COH-REPCAP),
g201_anchors (per-seed statistics, gate comparison), g201_smoke (fold choice, time projection)."""
from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pandas as pd
import pytest

from gen201 import bootstrap
from gen201 import data as D

import g201m_helpers as H


def _load(name):
    path = bootstrap.G201 / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_g201m_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def describe():
    return _load("g201_describe")


@pytest.fixture(scope="module")
def anchors():
    return _load("g201_anchors")


@pytest.fixture(scope="module")
def smoke():
    return _load("g201_smoke")


def _synthetic_frame(n_cells=12, seed=0):
    rng = np.random.default_rng(seed)
    nrep = rng.integers(1, 4, size=(n_cells, D.N_LN)).astype(float)
    sd = np.where(nrep >= 2, rng.uniform(0.0, 1.0, size=nrep.shape), np.nan)
    f = pd.DataFrame({f"nrep__{m}": nrep[:, i] for i, m in enumerate(D.LANTHANIDES)})
    for i, m in enumerate(D.LANTHANIDES):
        f[f"repsd__{m}"] = sd[:, i]
    f["cell_id"] = [f"c{i}" for i in range(n_cells)]
    obs = rng.uniform(size=(n_cells, D.N_LN)) < 0.7
    return f, obs


# ---------------- LT-19 ------------------------------------------------------------------------
def test_lt19_noise_reads_training_cells_only(describe):
    f, obs = _synthetic_frame()
    train = np.array([0, 2, 3, 7, 9])
    rec = describe.noise_record(f, train, obs)
    other = np.setdiff1d(np.arange(len(f)), train)
    g = f.copy()
    for c in [c for c in g.columns if c.startswith(("repsd__", "nrep__"))]:
        v = g[c].to_numpy(dtype=float, copy=True)
        v[other] = 99.0
        g[c] = v
    obs2 = obs.copy()
    obs2[other] = True
    assert describe.noise_record(g, train, obs2) == rec
    h = f.copy()
    h.loc[train[0], [c for c in h.columns if c.startswith("repsd__")]] = 0.123
    h.loc[train[0], [c for c in h.columns if c.startswith("nrep__")]] = 3.0
    obs3 = obs.copy()
    obs3[train[0]] = True
    assert describe.noise_record(h, train, obs3) != rec
    # the formulas on the synthetic entries
    sel = obs[train] & (f[list(D.NREP_COLS)].to_numpy(float)[train] >= 2)
    s = f[[f"repsd__{m}" for m in D.LANTHANIDES]].to_numpy(float)[train][sel]
    n = f[list(D.NREP_COLS)].to_numpy(float)[train][sel]
    assert rec["sigma2_pool"] == pytest.approx(np.sum((n - 1) * s ** 2) / np.sum(n - 1))
    cap = s <= 0.5
    assert rec["sigma2_pool_cap"] == pytest.approx(np.sum((n[cap] - 1) * s[cap] ** 2) / np.sum(n[cap] - 1))
    assert rec["median_s_e_cap"] == pytest.approx(np.median(s[cap]))
    assert rec["n_rep_entries"] == int(sel.sum())


# ---------------- describe ---------------------------------------------------------------------
def test_describe_card_and_level_shares(describe):
    rows = pd.DataFrame({"ci": [0, 0, 1, 1, 1, 2], "extractant": list("aabbbc"), "chemotype": list("xxxxxy"),
                         "publication_id": list("pppqqr")})
    y = np.array([1.0, 3.0, 0.0, 1.0, 2.0, 5.0])
    c = describe.card(rows, y, "logd")
    assert c["n_rows"] == 6 and c["n_cells"] == 3 and c["n_chemotypes"] == 2
    assert c["y_mean"] == pytest.approx(2.0)
    assert c["y_var0"] == pytest.approx(np.mean((y - 2.0) ** 2))
    assert c["s_b"] + c["s_w"] == pytest.approx(1.0) and c["pi"] == c["s_w"]
    sst = np.sum((y - 2) ** 2)
    ssb = 2 * (2 - 2) ** 2 + 3 * (1 - 2) ** 2 + 1 * (5 - 2) ** 2
    assert c["s_b"] == pytest.approx(ssb / sst)
    assert c["kish_chemotypes_rows"] == pytest.approx(36 / (25 + 1))
    s = describe.card(rows, y, "logsf")
    assert "y_rms0" in s and "s_b" not in s and s["y_rms0"] == pytest.approx(np.sqrt(np.mean(y ** 2)))


def test_describe_repcap_drops_entries_and_pairs(describe):
    f, obs = _synthetic_frame(4, seed=3)
    for m in D.LANTHANIDES:
        f[f"nrep__{m}"] = 1.0
        f[f"repsd__{m}"] = np.nan
    f.loc[1, "nrep__Ce"], f.loc[1, "repsd__Ce"] = 2.0, 0.9          # dropped
    f.loc[2, "nrep__Ce"], f.loc[2, "repsd__Ce"] = 2.0, 0.5          # kept (not > 0.5)
    obs[:] = True
    drop = describe.repcap_drop(f, obs)
    assert list(zip(drop.ci, drop.im)) == [(1, 1)]
    d_rows = pd.DataFrame({"ci": [1, 1, 2], "im": [1, 2, 1]})
    assert list(describe.apply_repcap(d_rows, drop, "logd").im) == [2, 1]
    s_rows = pd.DataFrame({"ci": [1, 1, 1, 2], "ia": [0, 1, 2, 0], "ib": [1, 2, 3, 1]})
    kept = describe.apply_repcap(s_rows, drop, "logsf")
    assert list(zip(kept.ci, kept.ia, kept.ib)) == [(1, 2, 3), (2, 0, 1)]


# ---------------- anchors ----------------------------------------------------------------------
def test_anchor_statistics_and_gate_on_synthetic_rows(anchors):
    rng = np.random.default_rng(5)
    rows = []
    for s in (1, 2):
        for i in range(30):
            rows.append({"split_seed": s, "ci": i % 7, "extractant": f"e{i % 4}", "chemotype": f"g{i % 3}",
                         "y": rng.normal(), "P": rng.normal()})
    t = pd.DataFrame(rows)
    st = anchors.seed_statistics(t[t.split_seed == 1], "P")
    g = t[t.split_seed == 1]
    e = g.y - g.P
    assert st["MAE"] == pytest.approx(np.mean(np.abs(e)))
    assert st["R2"] == pytest.approx(1 - np.sum(e ** 2) / np.sum((g.y - g.y.mean()) ** 2))
    assert st["MAE_macro_extractant"] == pytest.approx(np.abs(e).groupby(g.extractant).mean().mean())
    per = {s: anchors.seed_statistics(gg, "P") for s, gg in t.groupby("split_seed")}
    got = {"per_seed": per, "seed_mean": {k: float(np.mean([per[s][k] for s in per]))
                                          for k in ("R2", "MAE", "MAE_macro_extractant")}}
    ref = pd.Series({"R2": got["seed_mean"]["R2"], "MAE": got["seed_mean"]["MAE"],
                     "MAE_macro_extractant": got["seed_mean"]["MAE_macro_extractant"] + 5e-7,
                     "R2_per_seed": ";".join(f"{s}:{per[s]['R2']:.5f}" for s in per),
                     "MAE_per_seed": ";".join(f"{s}:{per[s]['MAE']:.5f}" for s in per)})
    anchors.EXPECTED_N_ROWS, anchors.EXPECTED_N_CELLS = 30, 7
    res = anchors.compare("X", got, ref)
    assert res["status"] == "reproduced (float32 storage or platform drift)"
    ref2 = ref.copy()
    ref2["R2"] = ref["R2"] + 1e-5
    assert anchors.compare("X", got, ref2)["status"] == "STOP"
    ref3 = ref.copy()
    ref3["MAE_macro_extractant"] = got["seed_mean"]["MAE_macro_extractant"]
    assert anchors.compare("X", got, ref3)["status"] == "reproduced"
    assert anchors.parse_per_seed("1:0.5;2:-0.25") == {1: 0.5, 2: -0.25}


def test_anchor_reference_rows_are_the_expected_csv_lines(anchors):
    ref = pd.read_csv(anchors.LEVELS_CSV)
    for name, line in anchors.CSV_ROWS.items():
        assert str(ref.iloc[line - 2]["arm"]) == f"{name}@ZERO|k0"
        assert int(ref.iloc[line - 2]["n_rows"]) == 1802 and int(ref.iloc[line - 2]["n_cells"]) == 229


# ---------------- smoke ------------------------------------------------------------------------
def test_smoke_fold_choice(bench, smoke):
    giant, big = smoke.pick_folds(H.bp_folds(bench))
    assert D.SC009 in giant.held_out_groups and len(giant.train_index) < 100
    others = [f for f in H.bp_folds(bench) if f is not giant]
    assert abs(len(big.train_index) - 490) == min(abs(len(f.train_index) - 490) for f in others)


def test_smoke_projection_is_linear_in_training_rows(smoke):
    measured = [{"n_train_rows": 1000, "families": {"A": 10.0, "C_PAIR": 4.0, "A_PRIME": 1.0}},
                {"n_train_rows": 3000, "families": {"A": 30.0, "C_PAIR": 12.0, "A_PRIME": 3.0}}]
    units = pd.DataFrame({"design": ["BP", "A"], "split_seed": [1, 1], "fold": [0, 0], "n_train_rows": [2000, 2000],
                          "n_train_rows_nodmd": [1000, 2000]})
    p = smoke.project(measured, units)
    # A: BP 20 + 10, design A 20 + 20; C_PAIR: BP primary only 8; A_PRIME: BP primary only 2
    assert p["seconds_total"] == pytest.approx(20 + 10 + 20 + 20 + 8 + 2)
