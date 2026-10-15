"""Gen20 validation tests (IMPLEMENTATION_PLAN.md §4.4): executable guarantees, not prose.

Run:  .venv/bin/python -m pytest generations/gen20_kshot/tests -q -p no:cacheprovider
"""
from __future__ import annotations

import glob
import itertools
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from gen20k import adapters as AD
from gen20k import arms as AR
from gen20k import cov as CV
from gen20k import data as D
from gen20k import design as G
from gen20k import metrics as M
from gen20k import nnc as NN
from gen20k import runner as R
from gen20k import splits as S

SEED = 104729


@pytest.fixture(scope="module")
def bench():
    return D.load_bench()


@pytest.fixture(scope="module")
def tiny_run(bench):
    """One BP fold, four cells, no neural arm: the runner path every output-level test reads."""
    f = S.folds_for(bench.frame, "BP", seeds=[SEED])[2]
    cfg = R.RunConfig(label="test", designs=("BP",), seeds=(SEED,), arms=("FLAT", "G14", "MEAN_CURVE", "NN_TANIMOTO"),
                      max_cells=4, smoke=True)
    table, meta = R.run_fold(bench, f, cfg, None)
    return f, cfg, table, meta


# ---- 1-2: hold-out integrity -----------------------------------------------------------
@pytest.mark.parametrize("design", ["A", "B", "BR", "BQ", "BP", "BPG"])
def test_no_forbidden_overlap_in_every_design(bench, design):
    folds = S.folds_for(bench.frame, design, seeds=[SEED])
    rep = S.assert_design_integrity(bench.frame, folds, design)
    assert rep["n_folds"] == 5
    if design != "A":
        assert rep["chemotype_overlap"] == 0
    if design in ("BP", "BPG"):
        assert rep["publication_overlap"] == 0
    if design == "BPG":
        assert rep["copy_group_overlap"] == 0
    held = np.concatenate([f.test_index for f in folds])
    assert sorted(held.tolist()) == list(range(len(bench.frame)))


def test_bpg_is_a_superset_mask_of_bp(bench):
    bp = S.folds_for(bench.frame, "BP", seeds=[SEED])
    bpg = S.folds_for(bench.frame, "BPG", seeds=[SEED])
    for a, b in zip(bp, bpg):
        assert np.array_equal(a.test_index, b.test_index)
        assert set(b.train_index) <= set(a.train_index)


def test_inner_mask_removes_held_out_publications(bench):
    f = S.folds_for(bench.frame, "BP", seeds=[SEED])[1]
    itr = CV.inner_mask(bench, "BP", np.asarray(f.inner_train_index), np.asarray(f.inner_validation_index))
    pub = bench.frame.publication_id.astype(str).to_numpy()
    assert not (set(pub[itr]) & set(pub[f.inner_validation_index]))
    assert set(itr) <= set(f.train_index)


# ---- 3-4: support/query isolation and closure, both measurement models ------------------
def _cell(bench, n_metals=14):
    ci = int(np.flatnonzero(bench.frame.n_metals.to_numpy() == n_metals)[0])
    return ci, D.cell_view(bench, ci)


def test_query_is_fixed_and_never_contains_a_support(bench):
    ci, cv = _cell(bench)
    for model, pols in (("pair", G.PAIR_POLICIES), ("metal", G.METAL_POLICIES)):
        dz = G.make_design(cv, "adjacent", model)
        q0 = set(dz.query)
        for how in pols:
            seq = G.support_sequence(dz, how, 10, split_seed=SEED, draw=0, cov=np.eye(14), noise_var=0.09)
            assert not (set(seq) & q0)
            assert set(G.make_design(cv, "adjacent", model).query) == q0


def test_every_policy_reaches_k_max_without_closure(bench):
    for ci in range(0, len(bench.frame), 7):
        cv = D.cell_view(bench, ci)
        for model, pols in (("pair", G.PAIR_POLICIES), ("metal", G.METAL_POLICIES)):
            dz = G.make_design(cv, "adjacent", model)
            if not dz.query:
                continue
            for how in pols:
                seq = G.support_sequence(dz, how, 10, split_seed=SEED, draw=0, cov=np.eye(14), noise_var=0.09)
                assert len(seq) == min(10, dz.k_max), (ci, model, how)
                assert not G.closed_pairs(cv.metals, seq, dz.query)
                if model == "metal":
                    read = G.read_metals(seq)
                    assert not any(a in read and b in read for a, b in dz.query)


def test_metal_model_max_independent_set_matches_brute_force(bench):
    """The path DP that bounds the metal model equals exhaustive search on cells with <= 9 metals."""
    checked = 0
    for ci in range(len(bench.frame)):
        cv = D.cell_view(bench, ci)
        if not 4 <= len(cv.metals) <= 9:
            continue
        dz = G.make_design(cv, "adjacent", "metal")
        if not dz.query:
            continue
        qs = set(dz.query)
        best = 0
        for r in range(1, len(cv.metals) + 1):
            for sub in itertools.combinations(cv.metals, r):
                if not any(a in sub and b in sub for a, b in qs):
                    best = max(best, r)
        assert G.max_independent_size(cv.metals, dz.query, frozenset()) == best, ci
        checked += 1
        if checked >= 40:
            break
    assert checked >= 20


def test_closure_validators_raise(bench):
    ci, cv = _cell(bench)
    la, ce, pr, nd, gd = (D.IDX[m] for m in ("La", "Ce", "Pr", "Nd", "Gd"))
    dz = G.make_design(cv, "adjacent", "pair")
    with pytest.raises(AssertionError):
        G.assert_no_closure(dz, [(la, pr), (ce, pr)])        # La-Pr-Ce closes La/Ce
    with pytest.raises(AssertionError):
        G.assert_no_closure(dz, [(la, nd), (nd, gd), (la, gd)])   # cycle
    dzm = G.make_design(cv, "adjacent", "metal")
    with pytest.raises(AssertionError):
        G.assert_metal_set(dzm, [la, ce])                      # adjacent metals both read
    with pytest.raises(AssertionError):
        G.assert_no_closure(dzm, G.star_pairs([la, pr, ce]))


def test_adapters_see_only_support_values(bench):
    """Replacing every non-support label by NaN leaves every adapted prediction identical."""
    ci, cv = _cell(bench)
    dz = G.make_design(cv, "adjacent", "metal")
    seq = G.support_sequence(dz, "UNIFORM", 3)
    y = bench.Y[ci]
    sup = G.reveal(y, seq)
    masked = np.full_like(y, np.nan)
    for a, b in seq:
        masked[a], masked[b] = y[a], y[b]
    assert sup == G.reveal(masked, seq)
    c0 = np.linspace(-1, 1, 14)
    R = CV.noise_matrix(sup.pairs, 0.05)
    for fn in (lambda s: AD.blup(c0, np.eye(14) * 0.1, s, R), lambda s: AD.naive_line(bench.basis, s), lambda s: AD.project(s),
               lambda s: AD.ridge_line(bench.basis, s, np.array([0.3, 0.0]), 1.0, R), lambda s: AD.rescale(c0, s)):
        assert np.allclose(fn(sup), fn(G.reveal(masked, seq)))


def test_runner_predictions_do_not_depend_on_query_labels(bench, monkeypatch, tiny_run):
    """Run one fold twice: the second time every held-out cell's log D is NaN except the metals a
    support reveals (captured from the first run).  Prediction columns must be identical."""
    f, cfg, table, _ = tiny_run
    revealed: dict[int, set] = {}
    orig = R.DS.reveal
    cells = [int(c) for c in f.test_index[: cfg.max_cells]]
    ymap = {ci: bench.Y[ci] for ci in cells}

    def spy(y_row, pairs):
        for ci, y in ymap.items():
            if np.array_equal(y, y_row, equal_nan=True):
                revealed.setdefault(ci, set()).update(m for p in pairs for m in p)
        return orig(y_row, pairs)
    monkeypatch.setattr(R.DS, "reveal", spy)
    t1, _ = R.run_fold(bench, f, cfg, None)
    Y = bench.Y.copy()
    masked = bench.Y.copy()
    for ci in cells:
        keep = revealed.get(ci, set())
        for m in range(14):
            if m not in keep:
                masked[ci, m] = np.nan
    # the design (query set, k_max) is built from cell_view(bench) = observed metals; keep that from
    # the unmasked bench, so only the *values* reaching adapters change
    monkeypatch.setattr(R.DS, "reveal", orig)
    views = {ci: D.cell_view(bench, ci) for ci in cells}
    orig_view = R.cell_view
    monkeypatch.setattr(R, "cell_view", lambda b, ci: views[int(ci)] if int(ci) in views else orig_view(b, ci))
    bench.Y = masked
    try:
        t2, _ = R.run_fold(bench, f, cfg, None)
    finally:
        bench.Y = Y
    pred_cols = [c for c in t1.columns if c not in R_KEY and c != "y"]
    assert len(t1) == len(t2)
    for c in pred_cols:
        assert np.allclose(t1[c].to_numpy(float), t2[c].to_numpy(float), equal_nan=True), c


R_KEY = {"design", "split_seed", "fold", "cell_id", "extractant", "chemotype", "publication", "n_metals", "model", "k_max",
         "policy", "draw", "k", "n_read", "A", "B", "ia", "ib", "dZ", "regime", "lab_known"}


def test_output_table_has_no_support_rows_and_only_measured_metals(bench, tiny_run):
    f, cfg, table, _ = tiny_run
    assert len(table) > 0
    for (cell, model, pol, d, k), g in table.groupby(["cell_id", "model", "policy", "draw", "k"]):
        ci = int(np.flatnonzero(bench.frame.cell_id.to_numpy() == cell)[0])
        cv = D.cell_view(bench, ci)
        dz = G.make_design(cv, "adjacent", model)
        seq = G.support_sequence(dz, pol, int(k), split_seed=f.seed, draw=int(d), cov=np.eye(14), noise_var=0.09) if pol in ("RANDOM", "WIDEST", "UNIFORM") else None
        pairs = set(zip(g.ia, g.ib))
        assert all(a in cv.metals and b in cv.metals for a, b in pairs)
        if seq is not None:
            assert not (pairs & set(seq[: int(k)]))
        q = g[g.regime == "Q"]
        assert set(zip(q.ia, q.ib)) == set(dz.query)
        if model == "metal":
            assert not q.lab_known.any()


def test_query_rows_identical_across_k_and_policies_in_a_run_table(tiny_run):
    _, _, t, _ = tiny_run
    q = t[t.regime == "Q"]
    for (seed, cell, model), g in q.groupby(["split_seed", "cell_id", "model"]):
        sets = {(pol, d, k): frozenset(zip(g_.A, g_.B)) for (pol, d, k), g_ in g.groupby(["policy", "draw", "k"])}
        assert len(set(sets.values())) == 1, (seed, cell, model)
        assert not any(k > g.k_max.iat[0] for (_, _, k) in sets)


# ---- 5-6: preprocessing, covariance and selection provenance ---------------------------
def test_selection_touches_inner_cells_only(bench, monkeypatch):
    f = S.folds_for(bench.frame, "BP", seeds=[SEED])[2]
    seen_train, seen_test = set(), set()
    orig = CV.make_ctx

    def spy(b, fold, train, test, **kw):
        seen_train.update(int(i) for i in train); seen_test.update(int(i) for i in test)
        return orig(b, fold, train, test, **kw)
    monkeypatch.setattr(CV, "make_ctx", spy)
    monkeypatch.setattr(R.CV, "make_ctx", spy)
    cfg = R.RunConfig(label="test", arms=("FLAT", "G14"), noise_factor_grid=(1.0,), lam_ridge_grid=(1.0,))
    rec = R.select_on_inner(bench, f, cfg)
    outer_test = set(int(i) for i in f.test_index)
    assert not (seen_train & outer_test) and not (seen_test & outer_test)
    assert seen_test <= set(int(i) for i in f.train_index)
    assert "G14@BLUP20" in rec["scores"] and "LINE" in rec["scores"]
    assert all(k in ("design", "split_seed", "fold", "n_inner_train", "n_inner_val", "s2_robust", "scores", "query_name",
                     "noise_factor", "s2", "ridge_lambda") for k in rec)


def test_covariance_uses_training_cells_only_and_is_psd(bench):
    f = S.folds_for(bench.frame, "BP", seeds=[SEED])[1]
    rec = CV.fold_covariance(bench, f, AR.as_fit_curves(AR.standard_arms()["G14"]))
    assert rec.cells and set(rec.cells) <= set(int(i) for i in f.train_index)
    assert not (set(rec.cells) & set(int(i) for i in f.test_index))
    assert np.linalg.eigvalsh(rec.cov).min() > 0
    cols = [c for c in bench.frame.columns if c.startswith("repsd__")]
    v = bench.frame[cols].to_numpy(float)[f.train_index]
    v = v[np.isfinite(v)]
    assert rec.noise_s2 == pytest.approx(float(np.median(v[v <= CV.REPLICATE_SD_CAP]) ** 2))
    y_adj = []
    for ci in f.train_index:
        cv = D.cell_view(bench, ci)
        y_adj += [D.pair_value(bench.Y[ci], a, b) for a, b in cv.pairs if D.is_strict_adjacent(a, b)]
    assert 2 * rec.noise_s2 < float(np.var(y_adj))        # the registered sanity bound on the noise constant


def test_nnc_preprocessing_and_training_use_training_cells_only(bench):
    f = S.folds_for(bench.frame, "BP", seeds=[SEED])[2]
    ctx = CV.make_ctx(bench, f, f.train_index, f.test_index)
    arm = NN.NNCArm(NN.NNCConfig(n_seeds=1, max_epochs=25, min_epochs=5, patience=5)).fit(ctx)
    assert set(arm.fit_cells) == set(int(i) for i in f.train_index)
    assert all(l["n_stage1_cells"] < l["n_train_cells"] for l in arm.fit_log)
    pub = bench.frame.publication_id.astype(str).to_numpy()
    # the stage-1 validation split was masked: no stage-1 training cell shares a publication with a validation cell
    tr1, va = NN._val_split(bench, np.asarray(f.train_index), "BP", int(ctx.seed) * 31 + int(ctx.fold) * 7919 + arm.val_seed_tag)
    assert not (set(pub[tr1]) & set(pub[va]))


# ---- 7: reproducibility across processes -----------------------------------------------
def test_fold_plan_supports_and_predictions_reproduce_in_a_subprocess():
    code = r"""
import sys, hashlib, numpy as np
sys.path.insert(0, %r)
from gen20k import data as D, design as G, splits as S, runner as R
b = D.load_bench()
h = hashlib.sha256()
for f in S.folds_for(b.frame, "BP", seeds=[104729]):
    h.update(np.asarray(f.train_index).tobytes()); h.update(np.asarray(f.test_index).tobytes())
for ci in range(0, len(b.frame), 25):
    for model in ("pair", "metal"):
        dz = G.make_design(D.cell_view(b, ci), "adjacent", model)
        if dz.query:
            h.update(str(G.support_sequence(dz, "RANDOM", 5, split_seed=104729, draw=2)).encode())
            h.update(str(G.support_sequence(dz, "UNIFORM", 5)).encode())
f = S.folds_for(b.frame, "BP", seeds=[104729])[2]
cfg = R.RunConfig(label="t", arms=("FLAT", "G14"), max_cells=2, smoke=True, noise_factor_grid=(1.0,), lam_ridge_grid=(1.0,))
t, _ = R.run_fold(b, f, cfg, None)
cols = [c for c in t.columns if c in ("G14@BLUP20", "LINE", "PROJ", "RIDGE2", "y")]
h.update(np.round(t[cols].to_numpy(float), 10).tobytes())
print(h.hexdigest())
""" % str(S.bootstrap.G20)
    outs = []
    for hs in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": hs}
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env)
        outs.append(r.stdout.strip().splitlines()[-1])
    assert outs[0] == outs[1]


# ---- 8-10: target arithmetic, orientation, no imputation --------------------------------
def test_orientation_and_antisymmetry(bench):
    ci, cv = _cell(bench)
    y = bench.Y[ci]
    nd, pr = D.IDX["Nd"], D.IDX["Pr"]
    v = D.pair_value(y, pr, nd)
    assert v == pytest.approx(y[pr] - y[nd])
    assert D.to_orientation(v, "heavier_first") == pytest.approx(-v)
    assert D.pair_label(pr, nd) == "Pr/Nd" and D.pair_label(pr, nd, "heavier_first") == "Nd/Pr"
    with pytest.raises(ValueError):
        D.pair_value(y, nd, pr)
    la, lu = D.IDX["La"], D.IDX["Lu"]
    assert D.pair_value(y, la, lu) == pytest.approx(D.pair_value(y, la, nd) + D.pair_value(y, nd, lu))


def test_unmeasured_metals_are_never_predicted_or_revealed(bench):
    ci = int(np.flatnonzero(bench.frame.n_metals.to_numpy() == 5)[0])
    cv = D.cell_view(bench, ci)
    assert len(cv.metals) == 5 and all(not np.isnan(bench.Y[ci, m]) for m in cv.metals)
    assert all(a in cv.metals and b in cv.metals for a, b in cv.pairs)
    missing = next(m for m in range(14) if m not in cv.metals)
    with pytest.raises(ValueError):
        G.reveal(bench.Y[ci], [(min(cv.metals[0], missing), max(cv.metals[0], missing))])


def test_strict_adjacency_excludes_nd_sm():
    nd, sm = D.IDX["Nd"], D.IDX["Sm"]
    assert not D.is_strict_adjacent(nd, sm) and D.is_gen13_adjacent(nd, sm)
    assert D.is_strict_adjacent(D.IDX["Pr"], D.IDX["Nd"])


def test_r2_undefined_cases_and_direction_tie_rule():
    assert np.isnan(M.r2([1.0], [1.0])[0]) and M.r2([1.0], [1.0])[1]
    assert np.isnan(M.r2([2.0, 2.0], [1.0, 3.0])[0])
    assert M.r2([0, 1, 2], [0, 1, 2])[0] == 1.0
    assert M.r2([0, 1, 2], [1, 1, 1])[0] == 0.0
    assert M.r2([0, 1, 2], [2, 2, 2])[0] < 0
    acc, n = M.direction_accuracy([0.5, -0.5], [0.0, 0.0])
    assert acc == 0.5 and n == 2


def test_blup_matches_gen15_with_independent_noise(bench):
    from gen15.fewshot import blup as blup15
    ci, cv = _cell(bench)
    dz = G.make_design(cv, "adjacent", "pair")
    seq = G.support_sequence(dz, "WIDEST", 3)
    sup = G.reveal(bench.Y[ci], seq)
    rng = np.random.default_rng(0)
    A = rng.normal(size=(14, 14)); cov = A @ A.T / 14 + 0.05 * np.eye(14)
    c0 = rng.normal(size=14)
    ours = AD.blup(c0, cov, sup, 0.09 * np.eye(3))
    theirs = blup15(c0, cov, sup.as_list(), noise_var=0.09)
    assert np.allclose(ours, theirs, atol=1e-8)


def test_cohort_matches_frozen_manifest(bench):
    rec = D.assert_matches_manifest(bench.frame)
    assert rec["n_cells"] == 521
    man = json.loads(open(D.MANIFEST_JSON).read())
    assert rec["manifest_sha256"] == man["sha256"]


# ---- the analysis path from the long table to the registered claims ---------------------
def test_registered_contrasts_and_mstar_decision_run_on_a_smoke_table():
    from gen20k import report as RP
    paths = sorted(glob.glob(str(S.bootstrap.PREDICTIONS / "smoke*" / "pairs_BP.parquet")))
    if not paths:
        pytest.skip("no smoke table present")
    t = pd.read_parquet(paths[-1])
    if "model" not in t.columns or not (t.model == "metal").any():
        pytest.skip("smoke table predates the two-model schema")
    metas = json.loads(open(str(S.bootstrap.PREDICTIONS / paths[-1].split("/")[-2] / "folds_BP.json")).read())
    d = RP.decide_mstar(metas)
    assert d["mstar"] in ("G14@BLUP20", "NNC@SELECTED") and d["n_selections"] == len(metas)
    rc = RP.registered_contrasts(t, d["mstar"], replicates=200)
    names = set(rc["name"])
    assert {"P1", "P2", "P3"} <= names
    reg = rc[rc.get("registered", False) == True]
    for col in ("ci95_low", "ci9875_low", "bca9875_low", "p_two_sided"):
        assert col in rc.columns
    assert (reg[reg.statistic == "MAE_macro_extractant"]["wcr_wcr_p_wcr"].notna()).all()
