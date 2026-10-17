"""LT-11 masked-label invariance (IMPLEMENTATION_PLAN.md §4): for every outer-test cell of the fold, ``bench.Y``, the
``bench.coef`` rows and the frame columns ``logD__*``, ``nrep__*``, ``repsd__*``, ``n_replicated_metals`` and
``replicate_sd_median`` are set to NaN (``cell_view`` / measured-metal mask, ``n_metals`` and ``bench.n_obs`` kept from
the unmasked bench through the label-free UnitContext); every A/B/C/BD_*/PAIRMEAN/G14_ZERO/reference prediction and
every selection record must be unchanged.  In addition ``gen20k.cov.make_ctx``, gen13 ``FitContext`` and every
``RowForest.fit`` are spied: each training index set is a subset of the masked train_index or equals a masked
inner-training set (gen20 test_leakage.py:239)."""
from __future__ import annotations

import numpy as np
import pandas as pd

import gen13sep.models as M13
import gen20k.cov as CV
from gen201 import fitting as FI
from gen201 import folds as FO
from gen201 import learners as L

import g201m_helpers as H


def _arm_columns(t: pd.DataFrame) -> list[str]:
    keys = set(FI.LOGD_KEYS) | set(FI.LOGSF_KEYS)
    return [c for c in t.columns if c not in keys]


def test_predictions_and_selection_unchanged_when_test_labels_are_masked(bench, monkeypatch):
    f = H.giant_fold(bench)
    ref = H.small_unit(bench)
    mb = H.masked_bench(bench, f.test_index)
    assert np.isnan(mb.Y[f.test_index]).all() and np.isnan(mb.coef[f.test_index]).all()
    assert np.array_equal(mb.frame["n_metals"].to_numpy(), bench.frame["n_metals"].to_numpy())

    seen = {"rf": [], "ctx": [], "fitctx": []}
    orig_fit, orig_make, orig_fc = L.RowForest.fit, CV.make_ctx, M13.FitContext

    def spy_fit(self, X, Y, groups, cells):
        seen["rf"].append(np.sort(np.asarray(cells, dtype=int)))
        return orig_fit(self, X, Y, groups, cells)

    def spy_make(b, fold, train, test, **kw):
        seen["ctx"].append((np.sort(np.asarray(train, dtype=int)), np.sort(np.asarray(test, dtype=int))))
        return orig_make(b, fold, train, test, **kw)

    def spy_fc(*a, **kw):
        ctx = orig_fc(*a, **kw)
        seen["fitctx"].append(np.sort(np.asarray(ctx.extra.get("cells"), dtype=int)))
        return ctx

    monkeypatch.setattr(L.RowForest, "fit", spy_fit)
    monkeypatch.setattr(CV, "make_ctx", spy_make)
    monkeypatch.setattr(M13, "FitContext", spy_fc)
    with H.tiny_trees():
        got = FI.run_unit(mb, "BP", f, variant="primary", folds_ctx=H.context(bench))

    # predictions: identical rows and identical arm values (y is NaN on the masked bench, by construction)
    for a, b in ((ref.logd, got.logd), (ref.logsf, got.logsf)):
        assert list(a.columns) == list(b.columns) and len(a) == len(b)
        assert np.isnan(b["y"].to_numpy(float)).all()
        for c in _arm_columns(a):
            np.testing.assert_allclose(b[c].to_numpy(float), a[c].to_numpy(float), rtol=0, atol=1e-12, err_msg=c)
    assert H.strip_volatile(got.selection) == H.strip_volatile(ref.selection)

    # spies: every training set is the masked outer train set or one masked inner-training set
    train = np.sort(np.asarray(f.train_index, dtype=int))
    inner = [np.sort(i.train_pos) for i in FO.inner_folds(bench, "BP", f) if not i.skipped]
    allowed = [train] + inner
    test = set(np.asarray(f.test_index).tolist())
    assert seen["rf"] and seen["ctx"] and seen["fitctx"]
    for cells in seen["rf"]:
        assert any(np.array_equal(cells, a) for a in allowed)
        assert not (set(cells.tolist()) & test)
    for tr, te in seen["ctx"]:
        assert np.array_equal(tr, train) and set(te.tolist()) <= test
    for cells in seen["fitctx"]:
        assert set(cells.tolist()) <= set(train.tolist())
        assert any(np.array_equal(cells, a) for a in allowed)


def test_masking_a_training_label_does_change_the_fit(bench):
    """Control: the invariance above is not vacuous; masking one training entry changes A's training rows."""
    f = H.giant_fold(bench)
    tr = np.asarray(f.train_index)
    Y = np.array(bench.Y, dtype=float)
    ci = int(tr[0])
    m = int(np.flatnonzero(np.isfinite(Y[ci]))[0])
    Y2 = Y.copy()
    Y2[ci, m] = np.nan
    with H.tiny_trees():
        a = L.RowForest("raw", "K1", f.model_seed).fit(D_X(bench), Y, bench.groups, tr)
        b = L.RowForest("raw", "K1", f.model_seed).fit(D_X(bench), Y2, bench.groups, tr)
    assert a.n_rows_ == b.n_rows_ + 1
    assert a.fit_record_["y_sha256"] != b.fit_record_["y_sha256"]


def D_X(bench):
    return H.context(bench).X_lean
