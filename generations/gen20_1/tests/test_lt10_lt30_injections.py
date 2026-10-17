"""LT-10 injections (test cell in training, publication crossing, inner-validation cell in inner training,
target-derived feature) and LT-30 injections that must be caught (a test row in A's training rows; a feature named
n_metals, replicate_sd_median or logD__*; a dummy arm using the test-cell mean log D; a baseline fitted with test
rows; a reference reading bench.coef of a test cell; a dropped scored row) (IMPLEMENTATION_PLAN.md §4).

The "caught" mechanism for arms that read labels is masked-label invariance (LT-11): the prediction must not change
when every outer-test label is masked; the honest gen20.1 arms pass it (test_lt11), the injected ones fail it here."""
from __future__ import annotations

import copy
import warnings
from dataclasses import replace

import numpy as np
import pytest

from gen201 import baselines as BL
from gen201 import data as D
from gen201 import fitting as FI
from gen201 import folds as FO
from gen201 import learners as L

import g201m_helpers as H


def _invariant(fn, bench, fold) -> bool:
    """True iff ``fn(bench) -> array`` is unchanged when the fold's outer-test labels are masked."""
    a = np.asarray(fn(bench), dtype=float)
    b = np.asarray(fn(H.masked_bench(bench, fold.test_index)), dtype=float)
    return a.shape == b.shape and bool(np.array_equal(np.isnan(a), np.isnan(b))) and bool(
        np.allclose(np.nan_to_num(a), np.nan_to_num(b), rtol=0, atol=1e-12))


# ---------------- LT-10 -----------------------------------------------------------------------
def test_lt10_test_cell_in_training_is_refused(bench):
    f = H.giant_fold(bench)
    bad = replace(f, train_index=np.sort(np.r_[f.train_index, f.test_index[:1]]))
    with H.tiny_trees(2, 2), pytest.raises(L.LeakageError):
        FI.run_unit(bench, "BP", bad, folds_ctx=H.context(bench))


def test_lt10_publication_crossing_is_refused(bench):
    f = H.giant_fold(bench)
    pub = bench.frame["publication_id"].astype(str).to_numpy()
    chem = bench.frame["chemotype"].astype(str).to_numpy()
    held_pub, held_chem = set(pub[f.test_index]), set(chem[f.test_index])
    tr = set(np.asarray(f.train_index).tolist())
    masked_out = [i for i in range(len(pub)) if i not in tr and i not in set(f.test_index.tolist())
                  and pub[i] in held_pub and chem[i] not in held_chem]
    assert masked_out, "BP drops training cells sharing a held-out publication"
    bad = replace(f, train_index=np.sort(np.r_[f.train_index, masked_out[:1]]))
    with pytest.raises(L.LeakageError):
        FI.assert_split(bench, "BP", bad.train_index, bad.test_index, H.context(bench))
    FI.assert_split(bench, "BP", f.train_index, f.test_index, H.context(bench))      # the real fold passes
    te = np.asarray(f.test_index)
    c = next(c for c in te if (chem[te] == chem[c]).sum() >= 2)                        # a chemotype with >= 2 test cells
    with pytest.raises(L.LeakageError):                                                # chemotype crossing too
        FI.assert_split(bench, "B", np.r_[f.train_index, [c]], te[te != c], H.context(bench))


def test_lt10_inner_validation_cell_in_inner_training_is_refused(bench, monkeypatch):
    f = H.giant_fold(bench)
    orig = FO.inner_folds

    def leaky(b, design, fold, **kw):
        out = orig(b, design, fold, **kw)
        i0 = out[0]
        out[0] = FO.InnerFold(train_pos=np.sort(np.r_[i0.train_pos, i0.val_pos[:1]]), val_pos=i0.val_pos,
                              skipped=False, j=0, train_pos_unmasked=i0.train_pos_unmasked)
        return out
    monkeypatch.setattr(FO, "inner_folds", leaky)
    with H.tiny_trees(2, 2), pytest.raises(L.LeakageError):
        FI.run_unit(bench, "BP", f, folds_ctx=H.context(bench))
    ctx = H.context(bench)
    i0 = orig(bench, "BP", f)[0]
    pub = ctx.publications
    other = [c for c in np.asarray(f.train_index) if c not in set(i0.train_pos.tolist()) | set(i0.val_pos.tolist())]
    shares = [c for c in other if pub[c] in set(pub[i0.val_pos])]
    if shares:                                          # a masked-out (publication-sharing) cell put back is refused
        bad = FO.InnerFold(train_pos=np.sort(np.r_[i0.train_pos, shares[:1]]), val_pos=i0.val_pos, skipped=False, j=0)
        with pytest.raises(L.LeakageError):
            FI.assert_inner("BP", bad, f.train_index, ctx)


@pytest.mark.parametrize("name", ["n_metals", "replicate_sd_median", "logD__La", "n_replicated_metals",
                                  "recipe__x", "safe_exp_ids"])
def test_lt10_lt30_target_derived_feature_is_rejected(name):
    with pytest.raises(Exception):
        D.assert_feature_columns(["cond__diluent__ch3cl", name])


def test_lt30_target_derived_feature_in_the_bench_blocks_stops_the_context(bench):
    b = copy.copy(bench)
    b.frames = dict(bench.frames)
    cond = bench.frames["COND"].copy()
    cond["n_metals"] = bench.frame["n_metals"].to_numpy(dtype=float)
    b.frames["COND"] = cond
    with pytest.raises(Exception):
        FI.make_context(b)


# ---------------- LT-30 -----------------------------------------------------------------------
def test_lt30_test_row_in_a_training_rows_is_caught(bench):
    f = H.giant_fold(bench)
    bad = replace(f, train_index=np.sort(np.r_[f.train_index, f.test_index[-1:]]))
    with H.tiny_trees(2, 2), pytest.raises(L.LeakageError):
        FI.run_unit(bench, "BP", bad, folds_ctx=H.context(bench), variant="nodmd")
    # the honest fit is invariant to the test labels; the same learner given the test rows is not
    ctx = H.context(bench)
    with H.tiny_trees(4):
        honest = _invariant(lambda b: L.RowForest("raw", "K1", f.model_seed).fit(
            ctx.X_lean, b.Y, b.groups, f.train_index).predict(ctx.X_lean, f.test_index[:20]), bench, f)
        leaky = _invariant(lambda b: L.RowForest("raw", "K1", f.model_seed).fit(
            ctx.X_lean, b.Y, b.groups, np.r_[f.train_index, f.test_index]).predict(ctx.X_lean, f.test_index[:20]), bench, f)
    assert honest and not leaky


def test_lt30_dummy_arm_using_the_test_cell_mean_is_caught(bench):
    f = H.giant_fold(bench)
    te = np.asarray(f.test_index)

    def dummy(b):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.repeat(np.nanmean(np.asarray(b.Y, float)[te], axis=1)[:, None], D.N_LN, axis=1)
    assert not _invariant(dummy, bench, f)


def test_lt30_baseline_fitted_with_test_rows_is_caught(bench):
    f = H.giant_fold(bench)
    ok = _invariant(lambda b: BL.MeanBaselines().fit(b.Y, b.groups, f.train_index).predict_metal(3), bench, f)
    bad = _invariant(lambda b: BL.MeanBaselines().fit(b.Y, b.groups, np.r_[f.train_index, f.test_index])
                     .predict_metal(3), bench, f)
    assert ok and not bad
    ok_pm = _invariant(lambda b: BL.PairMean().fit(b.Y, b.groups, f.train_index).table_, bench, f)
    bad_pm = _invariant(lambda b: BL.PairMean().fit(b.Y, b.groups, np.r_[f.train_index, f.test_index]).table_,
                        bench, f)
    assert ok_pm and not bad_pm


def test_lt30_reference_reading_test_coef_is_caught(bench):
    f = H.giant_fold(bench)
    te = np.asarray(f.test_index)
    honest = _invariant(lambda b: BL.gen20_ref_curves(b, f, f.train_index, te[:30])["G14"], bench, f)
    peeking = _invariant(lambda b: np.asarray(b.coef, float)[te[:30]] @ np.asarray(b.basis, float), bench, f)
    assert honest and not peeking


def test_lt30_dropped_scored_row_is_caught(bench):
    res = H.small_unit(bench)
    f = H.giant_fold(bench)
    test = np.sort(np.asarray(f.test_index))
    drop_d = res.logd.drop(index=res.logd.index[len(res.logd) // 2])
    with pytest.raises(FI.CoverageError):
        FI.check_coverage(drop_d, res.logsf, "BP", f, "primary", test, H.context(bench))
    strict = res.logsf[res.logsf["strict"]]
    drop_s = res.logsf.drop(index=strict.index[0])
    with pytest.raises(FI.CoverageError):
        FI.check_coverage(res.logd, drop_s, "BP", f, "primary", test, H.context(bench))
