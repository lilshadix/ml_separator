"""Deliberate violations must be caught (IMPLEMENTATION_PLAN.md §4.4, injection tests)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from gen20k import cov as CV
from gen20k import data as D
from gen20k import design as G
from gen20k import runner as R
from gen20k import splits as S

SEED = 104729


@pytest.fixture(scope="module")
def bench():
    return D.load_bench()


def test_injected_test_cell_in_training_is_detected(bench):
    folds = S.folds_for(bench.frame, "BP", seeds=[SEED])
    bad = replace(folds[0], train_index=np.append(folds[0].train_index, folds[0].test_index[:1]))
    with pytest.raises(AssertionError):
        S.assert_design_integrity(bench.frame, [bad] + folds[1:], "BP")


def test_injected_publication_crossing_is_detected(bench):
    folds = S.folds_for(bench.frame, "B", seeds=[SEED])
    relabelled = [replace(f, design="BP") for f in folds]
    with pytest.raises(AssertionError):
        S.assert_design_integrity(bench.frame, relabelled, "BP")


def test_injected_support_in_query_is_detected(bench):
    ci = int(np.flatnonzero(bench.frame.n_metals.to_numpy() == 14)[0])
    for model in ("pair", "metal"):
        dz = G.make_design(D.cell_view(bench, ci), "adjacent", model)
        with pytest.raises(AssertionError):
            G.assert_no_closure(dz, [dz.query[0]])


def test_injected_redundant_support_is_detected(bench):
    ci = int(np.flatnonzero(bench.frame.n_metals.to_numpy() == 14)[0])
    dz = G.make_design(D.cell_view(bench, ci), "adjacent", "pair")
    la, nd, gd = D.IDX["La"], D.IDX["Nd"], D.IDX["Gd"]
    with pytest.raises(AssertionError):
        G.assert_no_closure(dz, [(la, nd), (nd, gd), (la, gd)])      # third edge closes a cycle


def test_injected_held_out_residual_in_covariance_is_detected(bench):
    """A covariance record that lists a held-out cell among its residual sources is refused."""
    f = S.folds_for(bench.frame, "BP", seeds=[SEED])[1]
    rec = CV.covariance_from_residuals(np.random.default_rng(0).normal(size=(30, 14)), basis_row=bench.basis[0],
                                       cells=list(f.train_index[:29]) + [int(f.test_index[0])])
    with pytest.raises(AssertionError):
        CV.assert_cov_provenance(rec, f.train_index)
    # and the builder itself only ever draws from the training fold
    good = CV.fold_covariance(bench, f, lambda ctx: np.zeros((len(ctx.test), 14)))
    assert not (set(good.cells) & set(int(i) for i in f.test_index))


def test_injected_validation_cell_in_inner_training_is_removed(bench, monkeypatch):
    """An inner-validation cell injected into the inner-training index never reaches a fitted
    object: the selector's mask removes it (it shares its own publication), and an unmasked overlap
    is refused outright."""
    from gen20k import cov as CVm
    f = S.folds_for(bench.frame, "BP", seeds=[SEED])[2]
    inj = int(f.inner_validation_index[0])
    bad = replace(f, inner_train_index=np.append(f.inner_train_index, inj))
    seen = []
    orig = CVm.make_ctx
    monkeypatch.setattr(R.CV, "make_ctx", lambda b, fold, train, test, **kw: (seen.append(set(int(i) for i in train)), orig(b, fold, train, test, **kw))[1])
    cfg = R.RunConfig(label="t", arms=("FLAT", "G14"), noise_factor_grid=(1.0,), lam_ridge_grid=(1.0,))
    R.select_on_inner(bench, bad, cfg)
    assert seen and all(inj not in s for s in seen)
    monkeypatch.setattr(R.CV, "inner_mask", lambda b, d, tr, va: tr)      # disable the mask: overlap must be refused
    with pytest.raises(AssertionError):
        R.select_on_inner(bench, bad, cfg)


def test_injected_publication_sharing_inner_training_is_masked(bench):
    """Under BP the inner mask removes every inner-training cell sharing a publication with an
    inner-validation cell; injecting one such cell back is undone by the mask."""
    f = S.folds_for(bench.frame, "BP", seeds=[SEED])[1]
    pub = bench.frame.publication_id.astype(str).to_numpy()
    iva = np.asarray(f.inner_validation_index)
    shared = [i for i in f.train_index if pub[i] in set(pub[iva]) and i not in set(iva)]
    assert shared, "fold has no shared-publication cell to inject"
    itr = CV.inner_mask(bench, "BP", np.append(f.inner_train_index, shared[:1]), iva)
    assert shared[0] not in set(int(i) for i in itr)


def test_target_derived_feature_is_rejected():
    from gen13sep.features import _assert_allowed
    with pytest.raises(RuntimeError):
        _assert_allowed(["cond__acid_concentration_M", "logD__La"])
