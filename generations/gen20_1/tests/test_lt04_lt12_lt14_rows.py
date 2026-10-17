"""LT-04 preprocessing sees training rows only; LT-12 A/C training rows are exactly the finite Y[c, m] of the
masked train_index; LT-14 A and C get byte-identical X, weights and random_state (IMPLEMENTATION_PLAN.md §2-§4)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gen201 import data as D
from gen201 import fitting as FI
from gen201 import learners as L

import g201m_helpers as H


@pytest.fixture(scope="module")
def fitted(bench):
    f = H.giant_fold(bench)
    X = H.context(bench).X_lean
    with H.tiny_trees():
        a = L.RowForest("raw", "K3", f.model_seed).fit(X, bench.Y, bench.groups, f.train_index)
        c = L.RowForest("centred", "K3", f.model_seed).fit(X, bench.Y, bench.groups, f.train_index)
    return f, X, a, c


def test_lt12_training_rows_are_the_finite_entries_of_the_masked_train_index(bench, fitted):
    f, X, a, c = fitted
    tr = np.asarray(f.train_index, dtype=int)
    Y = np.asarray(bench.Y, float)
    ci, mi = np.nonzero(np.isfinite(Y[tr]))
    want = sorted(zip(tr[ci].tolist(), mi.tolist()))
    for m in (a, c):
        assert sorted(zip(m.train_ci_.tolist(), m.train_mi_.tolist())) == want
        assert m.n_rows_ == len(want) == int(np.isfinite(Y[tr]).sum())
    # the COH-NODMD variant: the same minus the DMDPhPDA cells
    dm = D.dmd_cells(bench)
    tr2 = tr[~np.isin(tr, dm)]
    with H.tiny_trees(2):
        m2 = L.RowForest("raw", "K1", f.model_seed).fit(X, bench.Y, bench.groups, tr2)
    assert not np.isin(m2.train_ci_, dm).any()
    assert m2.n_rows_ == int(np.isfinite(Y[tr2]).sum())


def test_lt12_matches_the_s3_coh_train_manifest(bench, fitted):
    f, X, a, _ = fitted
    man = pd.read_parquet(D.bootstrap.MANIFESTS / "row_manifests" / "COH-TRAIN.parquet")
    sub = man[(man.design == "BP") & (man.split_seed == f.seed) & (man.fold == f.fold)]
    assert set(zip(sub.ci.astype(int), sub.im.astype(int))) == set(zip(a.train_ci_.tolist(), a.train_mi_.tolist()))


def test_lt14_a_and_c_share_x_weights_and_random_state(fitted):
    f, X, a, c = fitted
    ra, rc = a.fit_record_, c.fit_record_
    assert ra["X_sha256"] == rc["X_sha256"] and ra["w_sha256"] == rc["w_sha256"]
    assert ra["seed"] == rc["seed"] == f.model_seed
    assert ra["n_estimators"] == rc["n_estimators"] and ra["config"] == rc["config"]
    assert ra["y_sha256"] != rc["y_sha256"]                         # only the target differs
    pa, pc = a.model_[-1].get_params(), c.model_[-1].get_params()
    assert pa == pc


def test_lt14_in_a_unit_every_config_of_a_and_c_shares_inputs(bench, monkeypatch):
    f = H.giant_fold(bench)
    recs = []
    orig = L.RowForest.fit

    def spy(self, X, Y, groups, cells):
        out = orig(self, X, Y, groups, cells)
        recs.append(dict(self.fit_record_, cells=tuple(np.sort(np.asarray(cells)).tolist())))
        return out
    monkeypatch.setattr(L.RowForest, "fit", spy)
    with H.tiny_trees(2, 2):
        FI.run_unit(bench, "BP", f, variant="primary", folds_ctx=H.context(bench), include_c_pair=False, max_test_cells=5)
    lean = [r for r in recs if r["metal_cols"] and r["n_features"] == 211 and r["seed"] == f.model_seed]
    by = {}
    for r in lean:
        by.setdefault((r["cells"], r["config"]), {})[r["target"]] = r
    pairs = [v for v in by.values() if set(v) == {"raw", "centred"}]
    assert len(pairs) >= 4
    for v in pairs:
        assert v["raw"]["X_sha256"] == v["centred"]["X_sha256"] and v["raw"]["w_sha256"] == v["centred"]["w_sha256"]


def test_lt04_imputer_and_weights_are_fitted_on_training_rows_only(bench, fitted):
    f, X, a, _ = fitted
    tr = np.asarray(f.train_index, dtype=int)
    Xc = X[tr]
    lci, mi = L.row_layout(np.asarray(bench.Y, float)[tr])
    feats = np.hstack([Xc[lci], D.Z[mi][:, None], D.R[mi][:, None]])
    imp = a.model_[0]
    with np.errstate(all="ignore"):
        med = np.nanmedian(feats, axis=0)
    expect = np.where(np.isnan(med), 0.0, med)               # keep_empty_features=True fills all-NaN columns with 0
    np.testing.assert_allclose(imp.statistics_, expect, rtol=1e-12, atol=1e-12)
    # perturbing every non-training cell's features leaves the fit unchanged
    X2 = X.copy()
    other = np.setdiff1d(np.arange(len(X)), tr)
    X2[other] = 1e6
    with H.tiny_trees():
        a2 = L.RowForest("raw", "K3", f.model_seed).fit(X2, bench.Y, bench.groups, tr)
    np.testing.assert_allclose(a2.model_[0].statistics_, imp.statistics_, rtol=0, atol=0)
    assert a2.fit_record_["X_sha256"] == a.fit_record_["X_sha256"]
    # weights: chemotype-balanced over the fit's own training cells, spread over each cell's observed metals
    w = L.row_weights(bench.groups[tr], np.asarray(bench.Y, float)[tr], lci)
    assert w.mean() == pytest.approx(1.0)
    chem = np.asarray(bench.groups)[tr][lci]
    share = pd.Series(w).groupby(chem).sum()
    np.testing.assert_allclose(share.to_numpy(), share.to_numpy()[0], rtol=1e-12)
    cellw = pd.Series(w).groupby(tr[lci]).sum()
    within = pd.DataFrame({"chem": np.asarray(bench.groups)[cellw.index.to_numpy()], "w": cellw.to_numpy()})
    for _, g in within.groupby("chem"):
        np.testing.assert_allclose(g["w"].to_numpy(), g["w"].to_numpy()[0], rtol=1e-12)


def test_lt04_bd_level_is_constant_within_a_cell_and_metal_free(bench, fitted):
    f, X, _, _ = fitted
    with H.tiny_trees():
        m = L.bd_level("K2", f.model_seed).fit(X, bench.Y, bench.groups, f.train_index)
    P = m.predict(X, f.test_index)
    assert m.fit_record_["n_features"] == 209
    assert np.all(P.max(axis=1) - P.min(axis=1) == 0.0)
