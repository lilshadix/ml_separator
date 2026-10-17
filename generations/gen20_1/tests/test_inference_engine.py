"""gen201.inference against direct recomputation and against gen20k.inference / gen13sep.wildcluster, on synthetic
tables only (no real target value)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gen201 import inference as I
from gen201 import metrics as M
from gen20k import inference as GI
from gen13sep import wildcluster as WC

import g201_synth as S

R_SMALL = 400
STATS = ("MET01", "MET02", "MET03", "MET04_MAE", "MET04_RMSE", "MET04_BIAS", "MET05", "MET06", "MET07", "MET07W",
         "MET08", "MET09", "MET10", "MET20", "S_B")


@pytest.fixture(scope="module")
def synth():
    cells = S.make_cells(14, seed=3)
    logd, logsf = S.make_tables(cells, skill=0.6)
    return logd, logsf[logsf["strict"]].reset_index(drop=True)


def test_points_equal_direct_metrics(synth):
    logd, logsf = synth
    for tab in (logd, logsf):
        F = I.Frame(tab, name="t")
        arm = "A" if F.target == "logd" else "B"
        d = M.summarise(tab, arm)
        for st in STATS:
            assert I.point(F, st, arm)["point"] == pytest.approx(d[I.DIRECT_KEY[st]], abs=1e-12), st
        assert I.point(F, "MET11", arm, "BD_MEAN" if F.target == "logd" else "FLAT")["point"] == pytest.approx(
            M.skill_mse(tab, arm, "BD_MEAN" if F.target == "logd" else "FLAT")["Skill_MSE"], abs=1e-12)


def _materialise(tab: pd.DataFrame, unit: str, counts: np.ndarray, names) -> pd.DataFrame:
    """The resample as a table: cluster k copied c_k times; the copies and everything nested in the unit are
    relabelled as distinct (chemotype: extractant, chemotype, cell; publication: publication, cell)."""
    col = I.UNIT_COLUMNS[unit]
    parts = []
    for k, (nm, c) in enumerate(zip(names, counts)):
        blk = tab[tab[col].astype(str) == nm]
        for j in range(int(c)):
            b = blk.copy()
            b["ci"] = b["ci"] + 100000 * (j + 1) + 1000000 * k
            b["cell_id"] = b["cell_id"] + f"#{k}.{j}"
            if unit == "chemotype":
                b["extractant"] = b["extractant"] + f"#{j}"
                b["chemotype"] = b["chemotype"] + f"#{j}"
            else:
                b["publication_id"] = b["publication_id"] + f"#{j}"
            parts.append(b)
    return pd.concat(parts, ignore_index=True)


@pytest.mark.parametrize("which", ["logd", "logsf"])
def test_chemotype_draws_equal_materialised_resample(synth, which):
    tab = synth[0] if which == "logd" else synth[1]
    arm = "A" if which == "logd" else "B"
    F = I.Frame(tab, name="t")
    names = F.clusters("chemotype")
    D = I.draw_matrix("chemotype", names, R_SMALL)
    for r in range(4):
        W = D[r:r + 1]
        mat = _materialise(tab, "chemotype", W[0], names)
        per = [M.seed_metrics(M.SeedRows(M._sorted_seed(mat, s)), arm) for s in F.seeds]
        direct = M.combine_seeds(per, F.seeds)
        for st in STATS:
            v = I.seed_mean(I.seed_values(F, "chemotype", W, st, arm))[0]
            assert v == pytest.approx(direct[I.DIRECT_KEY[st]], abs=1e-10), (st, r)


def test_publication_draws_formulas(synth):
    logd, _ = synth
    F = I.Frame(logd, name="t")
    names = F.clusters("publication")
    D = I.draw_matrix("publication", names, R_SMALL)
    for r in range(3):
        W = D[r:r + 1]
        mat = _materialise(logd, "publication", W[0], names)
        per = [M.seed_metrics(M.SeedRows(M._sorted_seed(mat, s)), "A") for s in F.seeds]
        direct = M.combine_seeds(per, F.seeds)
        for st in ("MET02", "MET05", "MET06", "MET08", "MET09", "MET10"):     # pooled statistics: plain resample
            v = I.seed_mean(I.seed_values(F, "publication", W, st, "A"))[0]
            assert v == pytest.approx(direct[I.DIRECT_KEY[st]], abs=1e-10), st
        # MET-01 by the plan §6 blocked formula: Σ_l u_l MAE_l* / Σ_l u_l
        _, K = F.cluster("publication")
        c = W[0][K]
        vals = []
        for k in range(len(F.seeds)):
            ae = np.abs(F.y - F.pred("A")[k])
            num = np.bincount(F.ext, weights=c * ae)
            n_lK = np.bincount(F.ext, weights=c)
            u = n_lK / F.n_l_vec
            mae = np.where(n_lK > 0, num / np.where(n_lK > 0, n_lK, 1), 0)
            vals.append((u * mae).sum() / u.sum())
        assert I.seed_mean(I.seed_values(F, "publication", W, "MET01", "A"))[0] == pytest.approx(np.mean(vals), abs=1e-12)
        # MET-04: within-extractant balanced mean recomputed on the materialised resample, extractant weight u_l
        vals = []
        for k, s in enumerate(F.seeds):
            m = M._sorted_seed(mat, s)
            w = M.balanced_weights(m["extractant"], m["publication_id"], m["ci"])
            L_star = m["extractant"].nunique()
            f = np.abs(m["y"] - m["A"]).to_numpy()
            B = pd.Series(L_star * w * f).groupby(m["extractant"].to_numpy()).sum()
            u = m.groupby("extractant").size() / pd.Series(F.n_l_vec, index=F.ext_names)[m.groupby("extractant").size().index]
            vals.append(float((u * B[u.index]).sum() / u.sum()))
        assert I.seed_mean(I.seed_values(F, "publication", W, "MET04_MAE", "A"))[0] == pytest.approx(np.mean(vals), abs=1e-12)


def test_draw_matrix_is_gen20k_draw_counts_and_shared(synth):
    logd, logsf = synth
    Fd, Fs = I.Frame(logd), I.Frame(logsf)
    assert Fd.clusters("chemotype") == Fs.clusters("chemotype")
    D1 = I.draw_matrix("chemotype", Fd.clusters("chemotype"), R_SMALL)
    D2 = I.draw_matrix("chemotype", Fs.clusters("chemotype"), R_SMALL)
    assert D1 is D2 and not D1.flags.writeable
    assert np.array_equal(D1, GI._draw_counts(len(Fd.clusters("chemotype")), R_SMALL, 8675309))
    P = I.draw_matrix("publication", Fd.clusters("publication"), R_SMALL)
    assert np.array_equal(P, GI._draw_counts(len(Fd.clusters("publication")), R_SMALL, 20260930))
    assert I.UNIT_SEEDS == {"chemotype": 8675309, "publication": 20260930} and I.REPLICATES == 10_000


def test_macro_contrast_equals_gen20k_blocked_unit_contrast(synth):
    logd, _ = synth
    F = I.Frame(logd)
    inf = I.infer(F, "MET01", "A", "BD_COND", contrast=True, replicates=R_SMALL, units=("chemotype",))
    dl = I.per_extractant_deltas(F, "A", "BD_COND")
    g = GI.blocked_unit_contrast(dl["delta"].to_numpy(), dl["chemotype"].to_numpy(), replicates=R_SMALL)
    assert inf["point"] == pytest.approx(g["point"], abs=1e-13)
    for mine, theirs in (("chem_pct95_low", "ci95_low"), ("chem_pct95_high", "ci95_high"),
                         ("chem_pct9875_low", "ci9875_low"), ("chem_pct9875_high", "ci9875_high"),
                         ("chem_bca95_low", "bca95_low"), ("chem_bca95_high", "bca95_high"),
                         ("chem_bca9875_low", "bca9875_low"), ("chem_p_two_sided", "p_two_sided"), ("chem_se", "se"),
                         ("loco_min", "loco_min")):
        assert inf[mine] == pytest.approx(g[theirs], abs=1e-12), mine
    assert inf["mde80_95"] == pytest.approx(g["mde_80"], abs=1e-12)
    assert inf["mde80_dec"] == pytest.approx(I.Z_DEC * g["se"], abs=1e-12) and abs(I.Z_DEC - 3.339) < 1e-3


def test_pooled_statistics_equal_gen20k(synth):
    _, logsf = synth
    F = I.Frame(logsf)
    lv = GI.pooled_levels(logsf, ["B"], replicates=R_SMALL).set_index("statistic")
    for st, key in (("MET06", "R2"), ("MET02", "MAE"), ("MET05", "RMSE")):
        inf = I.infer(F, st, "B", replicates=R_SMALL, units=("chemotype",))
        assert inf["point"] == pytest.approx(lv.loc[key, "point"], abs=1e-12)
        assert inf["chem_pct95_low"] == pytest.approx(lv.loc[key, "ci_low"], abs=1e-12)
        assert inf["chem_pct95_high"] == pytest.approx(lv.loc[key, "ci_high"], abs=1e-12)
    pc = GI.pooled_contrasts(logsf, {"c": ("FLAT", "B")}, statistics=("R2", "MAE"), replicates=R_SMALL)
    pc = pc.set_index("statistic")
    for st, key in (("MET06", "R2"), ("MET02", "MAE")):
        inf = I.infer(F, st, "B", "FLAT", contrast=True, replicates=R_SMALL, units=("chemotype",))
        for mine, theirs in (("point", "point"), ("chem_pct95_low", "ci_low"), ("chem_pct95_high", "ci_high"),
                             ("chem_bca95_low", "bca_low"), ("chem_bca95_high", "bca_high"),
                             ("chem_p_two_sided", "p_two_sided"), ("loco_min", "loco_min"), ("chem_se", "se")):
            assert inf[mine] == pytest.approx(pc.loc[key, theirs], abs=1e-12), (st, mine)
        assert inf["loco_all_positive"] == bool(pc.loc[key, "loco_all_positive"])


def test_bca_equals_gen20k_without_na():
    rng = np.random.default_rng(5)
    draws = rng.normal(0.3, 0.1, 3000)
    jack = rng.normal(0.3, 0.02, 20)
    for lv in (0.95, 0.9875):
        assert I.bca(draws, 0.31, jack, lv) == pytest.approx(GI._bca(draws, 0.31, jack, lv), abs=1e-14)


def test_loco_equals_direct_recomputation(synth):
    logd, _ = synth
    F = I.Frame(logd)
    inf = I.infer(F, "MET01", "A", "BD_NN", contrast=True, replicates=R_SMALL, units=("chemotype",))
    for chem in list(inf["loco"])[:4]:
        sub = logd[logd["chemotype"] != chem]
        direct = (M.summarise(sub, "BD_NN")["MET01_MAE_macro_ext"] - M.summarise(sub, "A")["MET01_MAE_macro_ext"])
        assert inf["loco"][chem] == pytest.approx(direct, abs=1e-12)


def test_wcr_matches_wildcluster_and_mde(synth):
    logd, _ = synth
    F = I.Frame(logd)
    dl = I.per_extractant_deltas(F, "A", "BD_METAL")
    d, cl = dl["delta"].to_numpy(), dl["chemotype"].to_numpy()
    w = I.wcr(d, cl)
    ref = WC.wild_cluster_t(d, cl, reps=9999, seed=8675309, weights="rademacher", se_kind="CR2", ci=False)
    assert w["wcr_weights"] == "rademacher" and w["p_wcr"] == ref.p_wcr and w["wcr_n_clusters"] == 14
    cr = WC.cluster_robust(d, cl)
    assert w["mde80_cr"] == pytest.approx(WC.detectable(cr.se_cr2, cr.dof_bm, alpha=0.0125), abs=1e-15)
    assert w["wcr_mean"] == pytest.approx(I.point(F, "MET01", "A", "BD_METAL", contrast=True)["point"], abs=1e-13)
    # G < 12 -> Webb weights
    keep = np.isin(cl, sorted(set(cl))[:8])
    w8 = I.wcr(d[keep], cl[keep])
    ref8 = WC.wild_cluster_t(d[keep], cl[keep], reps=9999, seed=8675309, weights="webb", se_kind="CR2", ci=False)
    assert w8["wcr_weights"] == "webb" and w8["p_wcr"] == ref8.p_wcr


def test_na_draw_handling():
    rng = np.random.default_rng(1)
    d = rng.normal(0.05, 0.02, 10_000)
    one = d.copy()
    one[:100] = np.nan                                    # exactly 1 %: kept, not degenerate
    b = I.interval_block(one, 0.05, None, "x")
    assert not b["x_degenerate"] and b["x_na_draws"] == 100
    lo_arr = np.where(np.isnan(one), -np.inf, one)
    assert b["x_pct9875_low"] == I.quantile_na(lo_arr, 0.00625) == -np.inf     # NA draws sit below every finite draw
    assert b["x_pct95_low"] == pytest.approx(np.sort(one[100:])[int((10_000 - 1) * 0.025) - 100], abs=0.01)
    two = d.copy()
    two[:101] = np.nan                                    # above 1 %: degenerate, interval NA
    b2 = I.interval_block(two, 0.05, None, "x")
    assert b2["x_degenerate"] and np.isnan(b2["x_pct95_low"]) and np.isnan(b2["x_pct95_high"])
    # NA counted unfavourably in the two-sided p
    pos = np.full(1000, 0.1)
    pos[:5] = np.nan
    assert I.p_two_sided(pos) == pytest.approx(2 * 5 / 1000)
    assert I.quantile_na(d, 0.3) == pytest.approx(np.quantile(d, 0.3), abs=0)
    assert np.isnan(I.bca(two, 0.05, rng.normal(size=10), 0.95)[0])


def test_r2_na_draws_when_sst_vanishes():
    # one chemotype with constant y; a draw that only picks it has SST = 0 -> NA, never a number
    rows = []
    for s in (1, 2):
        for ci, (chem, yv) in enumerate([("g0", 1.0), ("g0", 1.0), ("g1", 0.0), ("g1", 2.0)]):
            rows.append({"split_seed": s, "ci": ci, "cell_id": f"c{ci}", "extractant": f"E{ci}", "chemotype": chem,
                         "publication_id": f"p{ci}", "im": 0, "metal": "La", "y": yv, "X": yv + 0.1})
    F = I.Frame(pd.DataFrame(rows))
    W = np.array([[2.0, 0.0], [1.0, 1.0]])
    v = I.seed_mean(I.seed_values(F, "chemotype", W, "MET06", "X"))
    assert np.isnan(v[0]) and np.isfinite(v[1])


def test_one_cluster_population_is_na(synth):
    logd, _ = synth
    sub = logd[logd["chemotype"] == "sc009"]
    F = I.Frame(sub, name="sc009")
    inf = I.infer(F, "MET06", "A", replicates=R_SMALL)
    assert inf["chem_n_clusters"] == 1 and inf["chem_status"] == "one cluster" and np.isnan(inf["chem_pct95_low"])
    assert inf["pub_status"] in ("ok", "one cluster") and inf["chem_preliminary"]


def test_met17_infer_point_equals_direct(synth):
    cells = S.make_cells(10, seed=8)
    logd, logsf = S.make_tables(cells, skill=0.4)
    q = logsf[logsf["strict"]]
    a = M.met17(logd, q, "A")
    b = I.met17_infer(logd, q, "A", replicates=R_SMALL)
    for k in ("lambda", "kappa_adj", "rho_adj", "rho_within_adj"):
        assert b[k] == pytest.approx(a[k], abs=1e-10), k
        assert b[f"{k}_chem_pct95_low"] <= b[f"{k}_chem_pct95_high"]
