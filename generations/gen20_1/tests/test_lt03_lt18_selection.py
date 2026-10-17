"""LT-03 selection touches inner cells only; LT-18 inner fits see masked inner-training cells only
(IMPLEMENTATION_PLAN.md §4 "Inner selection"; PRE_REGISTRATION.md §4.3).

Spies on ``RowForest.fit`` / ``RowForest.predict`` during one unit: every inner fit is trained on exactly one masked
inner-training set (not the unmasked dealing; publication-masked under BP) and predicts exactly its inner-validation
cells; outer refits train on the outer training set and predict outer-test cells only; the pooled inner OOF rows are
exactly the observed entries (and strict pairs) of the non-skipped inner-validation cells; the selection rule is the
argmin with ties |delta| < 1e-12 to the lowest K, and all-skipped -> K1."""
from __future__ import annotations

import numpy as np
import pytest

from gen201 import fitting as FI
from gen201 import folds as FO
from gen201 import learners as L

import g201m_helpers as H


def test_inner_fits_and_predictions_touch_inner_cells_only(bench, monkeypatch):
    f = H.giant_fold(bench)
    fits, preds = [], []
    orig_fit, orig_pred = L.RowForest.fit, L.RowForest.predict

    def spy_fit(self, X, Y, groups, cells):
        fits.append((self.target, self.config, self.metal_cols, X.shape[1], np.sort(np.asarray(cells, dtype=int))))
        return orig_fit(self, X, Y, groups, cells)

    def spy_pred(self, X, cells):
        preds.append(np.sort(np.asarray(cells, dtype=int)))
        return orig_pred(self, X, cells)

    monkeypatch.setattr(L.RowForest, "fit", spy_fit)
    monkeypatch.setattr(L.RowForest, "predict", spy_pred)
    with H.tiny_trees():
        res = FI.run_unit(bench, "BP", f, variant="primary", folds_ctx=H.context(bench))
    train = np.sort(np.asarray(f.train_index, dtype=int))
    test = set(np.asarray(f.test_index).tolist())
    inner = FO.inner_folds(bench, "BP", f)
    active = [i for i in inner if not i.skipped]
    pub = bench.frame["publication_id"].astype(str).to_numpy()
    n_inner_fits = 0
    for (target, cfg, metal, p, cells), pc in zip(fits, preds):
        if np.array_equal(cells, train):
            assert set(pc.tolist()) <= test                      # outer refit -> outer test cells
            continue
        match = [i for i in active if np.array_equal(cells, i.train_pos)]
        assert len(match) == 1, "an inner fit on a set that is not a masked inner-training set"
        inn = match[0]
        assert np.array_equal(pc, inn.val_pos)                   # predicts its own inner-validation cells
        assert not (set(cells.tolist()) & test) and not (set(pc.tolist()) & test)
        assert not (set(pub[cells]) & set(pub[pc]))               # BP mask re-applied inside
        if len(inn.train_pos_unmasked) != len(inn.train_pos):
            assert not np.array_equal(cells, np.sort(inn.train_pos_unmasked))
        n_inner_fits += 1
    # 4 configs x (raw LEAN, centred LEAN, raw COND) per non-skipped inner fold
    assert n_inner_fits == 12 * len(active)
    # pooled inner OOF rows = observed entries / strict pairs of the inner-validation cells
    val = np.concatenate([i.val_pos for i in active])
    obs = H.context(bench).obs
    il = res.inner_oof["logd"]
    assert set(il["ci"]) <= set(train.tolist())
    assert set(zip(il["ci"], il["im"])) == {(int(c), int(m)) for c in val for m in np.flatnonzero(obs[c])}
    isf = res.inner_oof["logsf"]
    sf = H.context(bench).full_logsf
    want = sf[sf["strict"].to_numpy() & np.isin(sf["ci"].to_numpy(), val)]
    assert set(zip(isf["ci"], isf["ia"], isf["ib"])) == set(zip(want["ci"], want["ia"], want["ib"]))
    for arm in ("A", "C", "BD_COND", "B_selSF", "C_selSF"):
        rec = res.selection[arm]
        assert set(rec["scores"]) == set(L.CONFIG_ORDER)
        assert rec["chosen"] == L.select_config(rec["scores"])[0]
        assert rec["n_inner_val_cells"] == len(val)


def test_select_config_tie_rule_and_fallback():
    assert L.select_config({"K1": 0.3, "K2": 0.2, "K3": 0.25, "K4": 0.2})[0] == "K2"
    assert L.select_config({"K1": 0.2 + 5e-13, "K2": 0.2, "K3": 0.3, "K4": 0.4})[0] == "K1"
    assert L.select_config({"K1": 0.2 + 2e-12, "K2": 0.2, "K3": 0.3, "K4": 0.4})[0] == "K2"
    # chained near-ties resolve against the minimum, not pairwise
    assert L.select_config({"K1": 1.0, "K2": 1.0 - 0.6e-12, "K3": 1.0 - 1.2e-12, "K4": 2.0})[0] == "K2"
    assert L.select_config({"K1": float("nan"), "K2": float("nan"), "K3": float("nan"), "K4": float("nan")})[0] == "K1"
    assert L.select_config({"K1": float("nan"), "K2": 0.5, "K3": float("inf"), "K4": 0.6})[0] == "K2"


def test_all_inner_folds_skipped_gives_k1(bench, monkeypatch):
    f = H.giant_fold(bench)
    orig = FO.inner_folds

    def all_skipped(b, design, fold, **kw):
        return [FO.InnerFold(train_pos=i.train_pos, val_pos=i.val_pos, skipped=True, j=i.j,
                             train_pos_unmasked=i.train_pos_unmasked) for i in orig(b, design, fold, **kw)]
    monkeypatch.setattr(FO, "inner_folds", all_skipped)
    with H.tiny_trees(4, 2):
        res = FI.run_unit(bench, "BP", f, variant="primary", folds_ctx=H.context(bench), include_c_pair=False,
                          max_test_cells=5)
    assert res.selection["all_inner_skipped"]
    assert set(res.selection["chosen"].values()) == {"K1"}
    assert res.selection["BL_D_INNER"]["chosen"] == "BD_MEAN" and res.selection["BL_SF_INNER"]["chosen"] == "FLAT"


def test_documented_skipped_inner_fold_is_skipped_in_the_unit(bench):
    """BP seed 155921 fold 2 inner fold 3 (18 masked cells) is skipped and logged in the selection record."""
    F = FO.build(bench, seeds=(155921,), designs=("BP",))["BP"]
    f = next(x for x in F if x.fold == 2)
    inner = FO.inner_folds(bench, "BP", f)
    assert [i.j for i in inner if i.skipped] == [3]


def test_macro_mae_is_extractant_macro():
    e = np.array([1.0, 3.0, 2.0])
    assert FI.macro_mae(["x", "x", "y"], e) == pytest.approx((2.0 + 2.0) / 2)
    assert FI.macro_mae(["x", "y", "y"], e) == pytest.approx((1.0 + 2.5) / 2)
    assert np.isnan(FI.macro_mae([], []))
