"""Long-table properties of one unit (tiny forests; predictions only; y replaced by synthetic values wherever a
y-based quantity is computed):

* LT-06 orientation (lighter first, y = log D(a) - log D(b)), antisymmetry (B, C, C_PAIR), unmeasured metals never
  scored;
* LT-15 B equals the A-table difference with A's configuration (and every B_K / C / C_K / BD_*_SF likewise);
* LT-16 (coverage part) the scored rows equal the S3 row manifests and n_pred_finite == n_rows for every arm;
* LT-20 Nd/Pr <-> Pr/Nd flip invariance;
* LT-24 |R2_within_cell| <= 1e-12 for BD_MEAN and BD_LEVEL on synthetic y, and BD_LEVEL's within-cell range is 0
  exactly."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gen20k import data as GD
from gen201 import data as D
from gen201 import fitting as FI
from gen201 import learners as L

import g201m_helpers as H


def _join(logd, logsf, col_d, end):
    m = logd[["ci", "im", col_d]].rename(columns={"im": f"i{end}", col_d: f"{col_d}_{end}"})
    return logsf.merge(m, on=["ci", f"i{end}"], how="left", validate="many_to_one")


@pytest.fixture(scope="module", params=["primary", "nodmd"])
def unit(request, bench):
    return request.param, H.small_unit(bench, request.param)


def test_lt06_orientation_and_measured_metals_only(bench, unit):
    _, res = unit
    d, s = res.logd, res.logsf
    Y = np.asarray(bench.Y, float)
    assert (s["ia"] < s["ib"]).all() and (D.Z[s["ia"]] < D.Z[s["ib"]]).all()
    assert (s["pair"] == s["a"] + "/" + s["b"]).all()
    assert (s["dZ"] == (D.Z[s["ib"]] - D.Z[s["ia"]]).astype(int)).all()
    np.testing.assert_array_equal(s["y"].to_numpy(), Y[s.ci, s.ia] - Y[s.ci, s.ib])
    np.testing.assert_array_equal(d["y"].to_numpy(), Y[d.ci, d.im])
    assert np.isfinite(d["y"]).all() and np.isfinite(s["y"]).all()          # unmeasured metals never scored
    assert s["strict"].to_numpy().sum() + (~s["strict"].to_numpy()).sum() == len(s)
    assert ((s["dZ"] == 1) == s["strict"]).all()
    assert (s.loc[~s["strict"], "pair"] == "Nd/Sm").all() and s["gen13adj"].all()


def test_lt15_b_is_the_difference_of_a(unit):
    variant, res = unit
    d, s = res.logd, res.logsf
    chosen = res.selection["chosen"]
    np.testing.assert_array_equal(d["A"].to_numpy(), d[f"A_K{chosen['A'][1]}"].to_numpy())
    np.testing.assert_array_equal(d["C"].to_numpy(), d[f"C_K{chosen['C'][1]}"].to_numpy())
    np.testing.assert_array_equal(d["C_ACFG"].to_numpy(), d[f"C_K{chosen['A'][1]}"].to_numpy())
    np.testing.assert_array_equal(d["B_selSF_raw"].to_numpy(), d[f"A_K{chosen['B_selSF'][1]}"].to_numpy())
    pairs = {"B": "A", "B_K1": "A_K1", "B_K2": "A_K2", "B_K3": "A_K3", "B_K4": "A_K4", "B_selSF": "B_selSF_raw",
             "C": "C", "C_K1": "C_K1", "C_K4": "C_K4", "C_ACFG": "C_ACFG", "C_selSF": "C_selSF",
             "BD_METAL_SF": "BD_METAL", "BD_COND_SF": "BD_COND", "BD_NN_SF": "BD_NN",
             "BD_LEVEL_CURVE_SF": "BD_LEVEL_CURVE"}
    for sf_col, d_col in pairs.items():
        j = _join(d, _join(d, s, d_col, "a"), d_col, "b")
        assert len(j) == len(s) and j[[f"{d_col}_a", f"{d_col}_b"]].notna().all().all()
        np.testing.assert_array_equal(j[sf_col].to_numpy(), (j[f"{d_col}_a"] - j[f"{d_col}_b"]).to_numpy(),
                                      err_msg=sf_col)
    # BD_LEVEL is constant within a cell, so its in-cell difference is FLAT (BD_LEVEL_SF = 0 = FLAT)
    assert (s["FLAT"] == 0.0).all()


def test_lt06_c_pair_is_antisymmetric(bench):
    f = H.giant_fold(bench)
    ctx = H.context(bench)
    s = H.small_unit(bench).logsf
    with H.tiny_trees():
        cp = L.CPair(f.model_seed).fit(ctx.X_lean, bench.Y, bench.groups, f.train_index)
    ci, ia, ib = s["ci"].to_numpy(), s["ia"].to_numpy(), s["ib"].to_numpy()
    fwd, rev = cp.predict_pairs(ctx.X_lean, ci, ia, ib), cp.predict_pairs(ctx.X_lean, ci, ib, ia)
    np.testing.assert_array_equal(fwd, -rev)
    np.testing.assert_array_equal(fwd, s["C_PAIR"].to_numpy())             # same fit as in the unit (same seed)
    assert H.small_unit(bench).selection["c_pair"]["antisymmetric"]


def test_lt16_coverage_equals_the_row_manifests(bench, unit):
    variant, res = unit
    f = H.giant_fold(bench)
    suf = "@NODMD" if variant == "nodmd" else ""
    rm = D.bootstrap.MANIFESTS / "row_manifests"
    sel = lambda t: t[(t.design == "BP") & (t.split_seed == f.seed) & (t.fold == f.fold)]  # noqa: E731
    dall = sel(pd.read_parquet(rm / f"COH-D-ALL{suf}.parquet"))
    sany = sel(pd.read_parquet(rm / f"COH-SF-ANY{suf}.parquet"))
    ndsm = sel(pd.read_parquet(rm / f"COH-SF-NDSM{suf}.parquet"))
    dc3 = sel(pd.read_parquet(rm / f"COH-D-C3{suf}.parquet"))
    sc3 = sel(pd.read_parquet(rm / f"COH-SF-C3{suf}.parquet"))
    d, s = res.logd, res.logsf
    kd = set(zip(d.ci, d.im))
    ks = set(zip(s.ci, s.ia, s.ib))
    kstrict = set(zip(s.ci[s.strict], s.ia[s.strict], s.ib[s.strict]))
    assert kd == set(zip(dall.ci, dall.im)) and len(kd) == len(d)
    assert kstrict == set(zip(sany.ci, sany.ia, sany.ib))
    assert set(zip(ndsm.ci, ndsm.ia, ndsm.ib)) <= ks
    assert set(zip(dc3.ci, dc3.im)) <= kd and set(zip(sc3.ci, sc3.ia, sc3.ib)) <= kstrict
    for t, arms in ((d, res.selection["arms_logd"]), (s, res.selection["arms_logsf"])):
        for a in arms:
            v = t[a].to_numpy()
            assert v.dtype == np.float64 and int(np.isfinite(v).sum()) == len(t), a
    if variant == "nodmd":
        assert not np.isin(d.ci, D.dmd_cells(bench)).any()
        assert "A_PRIME" not in d.columns and "C_PAIR" not in s.columns
    else:
        assert "A_PRIME" in d.columns and "C_PAIR" in s.columns
    assert list(d.columns) == list(FI.LOGD_KEYS) + FI.logd_arm_columns("BP", variant)
    assert list(s.columns) == list(FI.LOGSF_KEYS) + FI.logsf_arm_columns("BP", variant)


def test_lt16_a_dropped_row_or_a_nonfinite_prediction_is_caught(bench):
    res = H.small_unit(bench)
    f = H.giant_fold(bench)
    ctx = H.context(bench)
    test = np.sort(np.asarray(f.test_index))
    with pytest.raises(FI.CoverageError):
        FI.check_coverage(res.logd.iloc[1:], res.logsf, "BP", f, "primary", test, ctx)
    with pytest.raises(FI.CoverageError):
        FI.check_coverage(res.logd, res.logsf.iloc[:-1], "BP", f, "primary", test, ctx)
    bad = res.logsf.copy()
    bad.loc[bad.index[0], "B"] = np.nan
    with pytest.raises(FI.CoverageError):
        FI.assert_finite(bad, ["B"], "logsf")


def test_lt20_flip_invariance(bench):
    s = H.small_unit(bench).logsf
    d = H.small_unit(bench).logd
    rows = s[s["pair"] == "Pr/Nd"]
    assert len(rows) > 0
    A = d.set_index(["ci", "im"])["A"]
    pr, nd = D.IDX["Pr"], D.IDX["Nd"]
    for _, r in rows.head(20).iterrows():
        heavier_first = A[(r.ci, nd)] - A[(r.ci, pr)]                         # Nd/Pr orientation
        assert GD.to_orientation(heavier_first, "heavier_first") == pytest.approx(r["B"], abs=1e-15)
        assert r["B"] == -(A[(r.ci, nd)] - A[(r.ci, pr)])
    # with synthetic y, flipping the orientation of both y and prediction leaves every absolute error unchanged
    rng = np.random.default_rng(0)
    y = rng.normal(size=len(rows))
    p = rows["B"].to_numpy()
    np.testing.assert_array_equal(np.abs(y - p), np.abs((-y) - (-p)))
    assert FI.macro_mae(rows["extractant"], np.abs(y - p)) == FI.macro_mae(rows["extractant"], np.abs(-y + p))


def test_lt24_bd_mean_and_bd_level_have_no_within_cell_skill(bench, unit):
    _, res = unit
    d = res.logd
    rng = np.random.default_rng(20261016)
    y = rng.normal(size=len(d))                                            # synthetic y, never the real target
    for col in ("BD_MEAN", "BD_LEVEL"):
        assert abs(H.r2_within_cell(d["ci"], y, d[col])) <= 1e-12, col
    rng_ = d.groupby("ci")["BD_LEVEL"].agg(lambda v: v.max() - v.min())
    assert (rng_ == 0.0).all()
    assert (d.groupby("ci")["BD_MEAN"].agg(lambda v: v.max() - v.min()) == 0.0).all()
