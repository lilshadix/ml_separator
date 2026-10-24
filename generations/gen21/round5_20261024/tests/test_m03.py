"""Tests for r5/m03.py (Stage-1 item 1.2, the M03 sibling-weighting factorial): synthetic data or construction-only
checks; no score is computed on real labels.

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B -W ignore -m pytest generations/gen21/round5_20261024/tests/test_m03.py -q
        -p no:cacheprovider --basetemp generations/gen21/round5_20261024/cache/_pytest

Covered: the frozen grid (every family holds the uniform limit; names; complexity order); the kernel against brute force
(weights, Kish count, shrinkage), its uniform limit = r5.ect's SIB / SH, numerical stability at large distances; the
offset-removed read distance; the acid distance and position rules; 2-nearest tie handling against brute force over
every tie-breaking order; the acid-trend arms against np.polyfit, PAIRMEAN brackets, POOLPAIR's leave-one-publication-
out slope against brute force and its poisoning invariance; the random-2 expectation; the per-publication SSE, the
nested selection (clear winner, one-SE toward uniform, order-free tie averaging, held-out labels irrelevant), the fast
R^2 contrast; packing; the intervention and its three positive controls on a synthetic pool; population counts from
the Stage-0 metadata table (counts only); the one-SE sensitivity labels; the section-9.3 sentences, unit tests and
duplicate block on synthetic tables.
"""
from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r5 import paths  # noqa: E402
from r5 import ect as E  # noqa: E402
from r5 import ect_report as REP  # noqa: E402
from r5 import m03 as M  # noqa: E402

ALL = tuple(range(14))
PAIRS = tuple((a, b) for a, b in zip(ALL[:-1], ALL[1:]) if (a, b) != (3, 4))
READS = {2: frozenset({0, 12}), 3: frozenset({0, 12, 6}), 4: frozenset({0, 12, 6, 9})}


def series(rng, n=1, scale=0.5):
    base = np.linspace(-1.0, 1.5, 14)
    return base[None, :] + rng.normal(0.0, scale, (n, 14))


def target(rng, y=None, beta=-0.1):
    y = series(rng)[0] if y is None else y
    return E.make_target("t0", ALL, PAIRS, y, READS, beta, zero=rng.normal(0.0, 0.1, len(PAIRS))), y


# ------------------------------------------------------------------------------------------------------------
# frozen grid
# ------------------------------------------------------------------------------------------------------------
def test_grid_frozen_and_uniform_in_every_family():
    assert M.LA_GRID == (math.inf, 1.0, 0.5, 0.25, 0.1) and M.LR_GRID == (math.inf, 0.4, 0.2, 0.1, 0.05)
    assert len(M.CONFIGS) == 25 and M.CONFIGS[0] == M.UNIFORM
    for fam, cfgs in M.FAMILIES.items():
        assert M.UNIFORM in cfgs
    assert len(M.FAMILIES["ACID"]) == 5 and len(M.FAMILIES["READ"]) == 5 and len(M.FAMILIES["BOTH"]) == 25
    names = [M.cfg_name(c, s) for c in M.CONFIGS for s in M.SHRINKS]
    assert len(set(names)) == 50 and M.cfg_name(M.UNIFORM, "sh") == "KW_ainf_rinf_sh"
    assert M.complexity((0, 0)) < M.complexity((4, 0)) < M.complexity((1, 1))
    assert M.complexity((1, 0)) == M.complexity((0, 1))
    assert M.parse_nest("NEST_BOTH_sh_classic") == ("BOTH", "sh", "classic")
    assert M.parse_nest(M.nest_name("ACID", "raw")) == ("ACID", "raw", M.PRIMARY_RULE)
    assert len(M.NEST_COLS) == 18 and M.B_PERM >= 200


# ------------------------------------------------------------------------------------------------------------
# kernel
# ------------------------------------------------------------------------------------------------------------
def test_kernel_bruteforce_and_uniform_limit():
    rng = np.random.default_rng(1)
    T, _ = target(rng)
    Ys = series(rng, 6)
    S = E.template_sf(T, 4, Ys)
    b1 = E.b1_sf(T, 4)
    ref = E.predict_sh(T, Ys)
    da, dr = rng.uniform(0, 1.5, 6), rng.uniform(0, 0.5, 6)
    raw, sh, neff, _ = M.kernel_predict(S[None], np.ones((1, 6), bool), b1[None], da[None], dr[None])
    for k, (a, r) in enumerate(M.CONFIGS):
        la, lr = M.LA_GRID[a], M.LR_GRID[r]
        w = np.exp(-(0 if la == math.inf else da / la) - (0 if lr == math.inf else dr / lr)) * np.ones(6)
        m = (w[:, None] * S).sum(0) / w.sum()
        ne = w.sum() ** 2 / (w ** 2).sum()
        assert np.allclose(raw[0, k], m, atol=1e-12)
        assert math.isclose(neff[0, k], ne, rel_tol=1e-12)
        assert np.allclose(sh[0, k], E.blend(m, b1, ne / (ne + M.K_SH)), atol=1e-12)
    u = M.CONFIGS.index(M.UNIFORM)
    assert np.allclose(raw[0, u], ref["SIB_r4"], atol=1e-12) and np.allclose(sh[0, u], ref["SH_r4"], atol=1e-12)
    raw2, sh2, _, dev = M.kernel_predict(S[None], np.ones((1, 6), bool), b1[None], da[None], dr[None],
                                         ref["SIB_r4"][None], ref["SH_r4"][None])
    assert np.array_equal(raw2[0, u], ref["SIB_r4"]) and np.array_equal(sh2[0, u], ref["SH_r4"]) and dev <= 1e-12


def test_kernel_padding_and_large_distances():
    rng = np.random.default_rng(2)
    S = rng.normal(size=(2, 4, 3))
    mask = np.array([[True, True, True, False], [True, True, False, False]])
    S[~mask] = 0.0
    b1 = rng.normal(size=(2, 3))
    da = np.array([[0.0, 60.0, 80.0, 0.0], [100.0, 300.0, 0.0, 0.0]])
    dr = np.zeros((2, 4))
    raw, sh, neff, _ = M.kernel_predict(S, mask, b1, da, dr)
    assert np.isfinite(raw).all() and np.isfinite(sh).all()
    k = M.CONFIGS.index((4, 0))                         # ell_acid = 0.1: all weight on the nearest sibling
    assert np.allclose(raw[0, k], S[0, 0]) and np.allclose(raw[1, k], S[1, 0]) and math.isclose(neff[1, k], 1.0)
    assert np.allclose(raw[1, 0], S[1, :2].mean(0))     # padding never enters


def test_read_distance_offset_removed():
    rng = np.random.default_rng(3)
    yt = rng.normal(size=4)
    Ys = rng.normal(size=(3, 4))
    dl, dr = M.read_distances(yt, Ys)
    dl2, dr2 = M.read_distances(yt, Ys + np.array([[0.7], [-1.3], [0.0]]))
    assert np.allclose(dr, dr2) and not np.allclose(dl[:2], dl2[:2])
    assert np.allclose(M.read_distances(yt, yt[None, :] + 0.5)[1], 0.0)


def test_acid_distance_and_position():
    d = M.acid_dist(1.0, np.array([10.0, 0.0, np.nan, 1e-6]))
    assert np.allclose(d, [1.0, 4.0, 1.0, 4.0])
    assert M.acid_dist(np.nan, np.array([np.nan]))[0] == 0.0
    assert M.acid_position(1.0, [0.5, 2.0]) == "interp" and M.acid_position(3.0, [0.5, 2.0]) == "extrap"
    assert M.acid_position(1.0, [1.0, 1.0, np.nan]) == "constant" and M.acid_position(np.nan, [1.0]) == "unknown"


def test_two_nearest_ties_against_every_order():
    rng = np.random.default_rng(4)
    for d in (np.array([0.3, 0.1, 0.3, 0.3, 0.9]), np.array([0.2, 0.2, 0.2, 0.5]), np.array([0.4, 0.1, 0.2, 0.6]),
              np.array([0.5, 0.5])):
        S = rng.normal(size=(len(d), 3))
        acc = []
        for perm in itertools.permutations(range(len(d))):
            order = sorted(range(len(d)), key=lambda j: (d[j], perm.index(j)))
            acc.append(S[order[:2]].mean(0))
        assert np.allclose(M.two_nearest(S, d), np.mean(acc, axis=0), atol=1e-12)
    assert np.isnan(M.two_nearest(np.ones((1, 3)), np.array([0.1]))).all()


def test_rand2_expectation():
    rng = np.random.default_rng(5)
    S, b1, ys = rng.normal(size=(5, 4)), rng.normal(size=4), rng.normal(size=4)
    raw, sh = M.rand2_se(S, b1, ys)
    pr = [S[[a, b]].mean(0) for a, b in itertools.combinations(range(5), 2)]
    assert np.allclose(raw, np.mean([(p - ys) ** 2 for p in pr], axis=0))
    w = E.weight(2)
    assert np.allclose(sh, np.mean([(w * p + (1 - w) * b1 - ys) ** 2 for p in pr], axis=0))


# ------------------------------------------------------------------------------------------------------------
# acid-trend arms
# ------------------------------------------------------------------------------------------------------------
def test_acid_trend_against_polyfit():
    rng = np.random.default_rng(6)
    la = np.log10([0.5, 1.0, 1.0, 3.0])
    v = rng.normal(size=(4, 5))
    for lt in (np.log10(2.0), np.log10(6.0), np.log10(0.1)):
        out = M.acid_trend(v, la, lt)
        ltc = min(max(lt, la.min()), la.max())
        for p in range(5):
            b = np.polyfit(la, v[:, p], 1)
            lin, lc, mean = np.polyval(b, lt), np.polyval(b, ltc), v[:, p].mean()
            assert math.isclose(out["LIN_r0"][p], lin, abs_tol=1e-12)
            assert math.isclose(out["LINCLIP_r0"][p], lc, abs_tol=1e-12)
            assert math.isclose(out["LINHALF_r0"][p], mean + 0.5 * (lin - mean), abs_tol=1e-12)
            assert math.isclose(out["LINHALFCLIP_r0"][p], mean + 0.5 * (lc - mean), abs_tol=1e-12)
        assert out["acid_inside"] == (la.min() <= lt <= la.max())
    inside = M.acid_trend(v, la, np.log10(2.0))
    assert np.allclose(inside["PAIRMEAN_r0"], 0.5 * (v[1:3].mean(0) + v[3]))
    assert np.isnan(M.acid_trend(v, la, np.log10(6.0))["PAIRMEAN_r0"]).all()
    flat = M.acid_trend(v[:2], np.log10([1.0, 1.0]), 0.5)
    assert np.isnan(flat["LIN_r0"]).all() and flat["sxx"] == 0.0


def _acid_frame(rng):
    rows = []
    for q, pub in enumerate(("p1", "p2", "p3")):
        for c in range(2):
            la = np.log10([0.5, 1.0, 4.0]) + 0.1 * c
            v = rng.normal(size=(3, 2))
            out = M.acid_trend(v, la, np.log10(2.0))
            for p, (A, B) in enumerate((("La", "Ce"), ("Ce", "Pr"))):
                rows.append({"cell_id": f"{pub}c{c}", "publication": pub, "A": A, "B": B, "split_seed": 1,
                             "ACOPY_r0": v[:, p].mean(), "sxy": out["sxy"][p], "sxx": out["sxx"], "xpp": out["xpp"]})
    return pd.DataFrame(rows)


def test_poolpair_lopo_slope_and_poisoning():
    rng = np.random.default_rng(7)
    d = _acid_frame(rng)
    got = M.poolpair(d)
    for k, r in d.iterrows():
        o = d[(d.publication != r.publication) & (d.A == r.A)]
        b = o.sxy.sum() / o.sxx.sum()
        assert math.isclose(got[k], r.ACOPY_r0 + b * r.xpp, abs_tol=1e-12)
    d2 = d.copy()
    own = d2.publication == "p2"
    d2.loc[own, "sxy"] = rng.normal(size=int(own.sum()))
    assert np.array_equal(M.poolpair(d2)[own.to_numpy()], got[own.to_numpy()])
    d3 = pd.concat([d, d.assign(split_seed=2)], ignore_index=True)        # repeated seeds do not reweight the slope
    assert np.allclose(M.poolpair(d3)[: len(d)], got)


# ------------------------------------------------------------------------------------------------------------
# nested selection
# ------------------------------------------------------------------------------------------------------------
def _nest_data(rng, n_pub=6, per=20, n_seed=2, eff=None, C=5):
    pc = np.repeat(np.arange(n_pub), per * n_seed)
    sc = np.tile(np.repeat(np.arange(n_seed), per), n_pub)
    y = rng.normal(size=len(pc))
    noise = rng.normal(0, 0.3, (len(pc), C))
    if eff is not None:
        noise *= np.asarray(eff)[None, :]
    return y[:, None] + noise, y, sc, pc


def test_sse_by_pub_bruteforce():
    rng = np.random.default_rng(8)
    PM, y, sc, pc = _nest_data(rng, n_pub=3, per=4)
    err = (PM - y[:, None]) ** 2
    got = M.sse_by_pub(err, sc, pc, 2, 3)
    for q in range(3):
        exp = np.mean([err[(pc == q) & (sc == s)].sum(0) for s in range(2)], axis=0)
        assert np.allclose(got[q], exp)


def test_nested_clear_winner_and_one_se_toward_uniform():
    rng = np.random.default_rng(9)
    cfgs = M.FAMILIES["ACID"]
    PM, y, sc, pc = _nest_data(rng, eff=[1.0, 1.0, 1.0, 0.2, 1.0])
    pred, recs = M.nested_select(PM, y, sc, pc, cfgs, "paired")
    assert all(r["chosen"] == M.cfg_label(cfgs[3]) for r in recs)
    assert np.allclose(pred, PM[:, 3])
    PM2, y2, sc2, pc2 = _nest_data(rng, eff=[1.0, 1.0, 1.0, 1.0, 1.0])
    PM2[:, 1:] = PM2[:, [0]]                                          # identical predictions: the simplest wins
    _, recs2 = M.nested_select(PM2, y2, sc2, pc2, cfgs, "paired")
    assert all(r["chosen"] == "ainf_rinf" for r in recs2)
    _, recs3 = M.nested_select(PM2, y2, sc2, pc2, cfgs, "argmin")      # argmin averages every exact tie
    assert all(r["n_chosen"] == 5 for r in recs3)
    # a net gain inside one paired SE: the paired and classic rules keep uniform, argmin does not
    s = np.array([1.0, -1.0, 1.0, -1.0, -1.0, 0.0])
    X = np.c_[np.full(6, 10.0), 10.0 + 0.1 * s, np.full((6, 3), 12.0)]
    train = np.arange(6) != 5
    for rule, exp in (("paired", "ainf_rinf"), ("classic", "ainf_rinf"), ("argmin", M.cfg_label(cfgs[1]))):
        ch, info = M.choose(X, train, list(cfgs), rule)
        assert info["chosen"] == exp, rule


def test_nested_ties_averaged_order_free():
    rng = np.random.default_rng(10)
    cfgs = M.FAMILIES["ACID"]
    PM, y, sc, pc = _nest_data(rng, eff=[1.0, 1.0, 0.3, 1.0, 1.0])
    PM[:, 3] = PM[:, 2]                                              # an exact tie of the loss (equal complexity rank
    _, recs = M.nested_select(PM, y, sc, pc, cfgs, "argmin")         # differs, so test with argmin: average both)
    assert all(r["n_chosen"] == 2 for r in recs)
    perm = [0, 3, 2, 1, 4]
    p1, _ = M.nested_select(PM, y, sc, pc, cfgs, "argmin")
    p2, _ = M.nested_select(PM[:, perm], y, sc, pc, [cfgs[k] for k in perm], "argmin")
    assert np.array_equal(p1, p2)
    # paired rule: two equally complex configs (acid 1 vs read 0.4) with identical, clearly better predictions
    both = M.FAMILIES["BOTH"]
    PMb, yb, scb, pcb = _nest_data(rng, C=25)
    ia, ir = both.index((1, 0)), both.index((0, 1))
    PMb[:, ia] = yb + rng.normal(0, 0.02, len(yb))
    PMb[:, ir] = PMb[:, ia]
    _, recb = M.nested_select(PMb, yb, scb, pcb, both, "paired")
    assert all(r["n_chosen"] == 2 and not r["uniform_chosen"] for r in recb)


def test_nested_held_out_labels_irrelevant():
    rng = np.random.default_rng(11)
    cfgs = M.FAMILIES["BOTH"]
    PM, y, sc, pc = _nest_data(rng, C=25)
    for rule in M.SE_RULES:
        base, recs = M.nested_select(PM, y, sc, pc, cfgs, rule)
        for q in range(int(pc.max()) + 1):
            y2, PM2 = y.copy(), PM.copy()
            r = pc == q
            y2[r], PM2[r] = rng.normal(size=int(r.sum())), rng.normal(size=(int(r.sum()), 25))
            sse = M.sse_by_pub((PM2 - y2[:, None]) ** 2, sc, pc, int(sc.max()) + 1, int(pc.max()) + 1)
            ch, info = M.choose(sse, np.arange(int(pc.max()) + 1) != q, list(cfgs), rule)
            assert ";".join(M.cfg_label(cfgs[c]) for c in ch) == recs[q]["chosen"] and info["L_min"] == recs[q]["L_min"]


def test_dr2_fast_matches_report_helpers():
    rng = np.random.default_rng(12)
    PM, y, sc, pc = _nest_data(rng, n_pub=3, per=5, C=2)
    d = pd.DataFrame({"y": y, "split_seed": sc + 100, "a": PM[:, 0], "b": PM[:, 1]})
    assert math.isclose(M.dr2_fast(y, PM[:, 0], PM[:, 1], sc), REP.r2(d, "a") - REP.r2(d, "b"), abs_tol=1e-12)


def test_pack_and_to_rows_roundtrip():
    rng = np.random.default_rng(13)
    groups = []
    pos = 0
    for k, (n, p) in enumerate(((3, 2), (2, 4))):
        groups.append({"S": rng.normal(size=(n, p)), "b1": rng.normal(size=p), "y": rng.normal(size=p),
                       "SIB_r4": rng.normal(size=p), "SH_r4": rng.normal(size=p), "da": rng.uniform(size=n),
                       "dr": rng.uniform(size=n), "js": np.arange(n), "ti": k, "rowpos": np.arange(pos, pos + p),
                       "cell": f"c{k}", "seed": 1})
        pos += p
    KD = M.pack(groups)
    assert KD.S.shape == (2, 3, 4) and KD.mask.sum() == 5
    back = M.to_rows(KD, KD.b1[:, None, :], pos)[:, 0]
    assert np.array_equal(back, np.concatenate([g["b1"] for g in groups]))


# ------------------------------------------------------------------------------------------------------------
# information contract on a synthetic pool
# ------------------------------------------------------------------------------------------------------------
def _fake(rng, n=5):
    Yv = series(rng, n + 1)
    acid_raw = np.resize(np.array([1.0, 0.5, 2.0, 2.0, 4.0, np.nan]), n + 1)
    ids = [f"s{k}" for k in range(1, n + 1)]
    pos = {"t0": 0, **{s: k + 1 for k, s in enumerate(ids)}}
    cm = SimpleNamespace(cond_dist=lambda i, j: float(M.acid_dist(acid_raw[i], np.array([acid_raw[j]]))[0]))
    corpus = SimpleNamespace(pos=pos, cm=cm, acid=np.log10(acid_raw))
    return corpus, Yv, acid_raw, ids


def test_intervention_and_positive_controls_synthetic():
    rng = np.random.default_rng(14)
    corpus, Yv, acid_raw, ids = _fake(rng)
    T = E.make_target("t0", ALL, PAIRS, Yv[0], READS, -0.1, zero=np.zeros(len(PAIRS)))
    acid_ids, near = ids[1:4], ids[:1]
    base = M.group_inputs(corpus, T, 0, ids, acid_ids, near, Yv, acid_raw)
    Yp = Yv.copy()
    un = [m for m in ALL if m not in READS[4]]
    Yp[0, un] = rng.normal(0, 3, len(un))
    Tp = E.make_target("t0", ALL, PAIRS, Yp[0], READS, -0.1, zero=np.zeros(len(PAIRS)))
    pois = M.group_inputs(corpus, Tp, 0, ids, acid_ids, near, Yp, acid_raw)
    assert all(M._same(base[k], pois[k]) for k in base if k != "js")
    a1 = M.group_inputs(corpus, T, 0, ids + ["t0"], [], near, Yv, acid_raw)
    a2 = M.group_inputs(corpus, Tp, 0, ids + ["t0"], [], near, Yp, acid_raw)
    assert not np.array_equal(a1["K_sh"], a2["K_sh"])                         # target in its own pool: caught
    p1 = M.group_inputs(corpus, T, 0, ids, [], near, Yv, acid_raw, peek=True)
    p2 = M.group_inputs(corpus, Tp, 0, ids, [], near, Yp, acid_raw, peek=True)
    assert not np.array_equal(p1["dr"], p2["dr"])                             # full-series read distance: caught
    with pytest.raises(E.LeakageError):
        E.guard_pool("t0", Yv[0], ids + ["t0"], Yv, name="M03 pool + target")
    with pytest.raises(E.LeakageError):
        E.guard_pool("t0", Yv[0], ["copy"], Yv[0][None, :], name="M03 bit")


def test_group_inputs_arms_consistent_with_ect():
    rng = np.random.default_rng(15)
    corpus, Yv, acid_raw, ids = _fake(rng)
    T = E.make_target("t0", ALL, PAIRS, Yv[0], READS, -0.1, zero=rng.normal(0, 0.1, len(PAIRS)))
    g = M.group_inputs(corpus, T, 0, ids, ids[1:4], ids[:2], Yv, acid_raw)
    ref = E.predict_sh(T, Yv[1:])
    for k in ("SIB_r4", "SH_r4", "B1_r4", "COPY", "SH0"):
        assert np.array_equal(g[k], ref[k]), k
    assert np.array_equal(g["NEAR_r4"], E.template_sf(T, 4, Yv[[1, 2]]).mean(0))
    assert np.array_equal(g["ACOPY_r0"], E.template_sf(T, 0, Yv[[2, 3, 4]]).mean(0))
    u = M.CONFIGS.index(M.UNIFORM)
    assert np.array_equal(g["K_sh"][u], ref["SH_r4"]) and np.array_equal(g["K_raw"][u], ref["SIB_r4"])


def test_large_pool_matches_target_block_layout():
    """n >= 8: SIB / SH / COPY equal Corpus.target_block's row-indexed pool means and predict_sh bit for bit."""
    rng = np.random.default_rng(16)
    corpus, Yv, acid_raw, ids = _fake(rng, n=12)
    T = E.make_target("t0", ALL, PAIRS, Yv[0], READS, -0.1, zero=rng.normal(0, 0.1, len(PAIRS)))
    g = M.group_inputs(corpus, T, 0, ids, [], ids[:1], Yv, acid_raw)
    order = list(range(12))
    S4 = E.template_sf(T, 4, Yv[1:])[order]                                     # target_block: S[r][[ui[s] ...]]
    S0 = E.template_sf(T, 0, Yv[1:])[order]
    assert np.array_equal(g["SIB_r4"], S4.mean(0)) and np.array_equal(g["COPY"], S0.mean(0))
    assert np.array_equal(g["SH_r4"], E.blend(S4.mean(0), E.b1_sf(T, 4), E.weight(12)))
    ref = E.predict_sh(T, Yv[1:])                                               # and r5.ect's external path
    for k in ("SIB_r4", "SH_r4", "B1_r4", "COPY", "SH0"):
        assert np.array_equal(g[k], ref[k]), k
    assert np.array_equal(g["SIB_r4"], E.pool_mean(S4)) and np.array_equal(g["COPY"], E.pool_mean(S0))


# ------------------------------------------------------------------------------------------------------------
# populations (Stage-0 metadata table; counts only)
# ------------------------------------------------------------------------------------------------------------
def test_population_counts_from_stage0_metadata():
    p = paths.RES / "stage0" / "ect_population_cells.csv"
    if not p.exists():
        pytest.skip("Stage-0 population table missing")
    t = pd.read_csv(p)
    pr = t[t.in_PRIMARY]
    assert len(pr) == 99 and pr.publication.nunique() == 12 and (pr.n_sib >= 2).all() and not pr.c5.any()
    assert pr[pr.publication != M.ORNL].publication.nunique() == 11
    assert len(t[t.in_WITH_C5]) == 114 and len(t[t.in_PRIMARY_NOBIT]) == 99
    assert set(M.POPULATIONS) >= {"PRIMARY", "PRIMARY_NO_ORNL", "WITH_C5", "WITH_C5_NO_C5CELL", "PRIMARY_NOBIT"}


# ------------------------------------------------------------------------------------------------------------
# labels and the section-9.3 per-item table (synthetic tables)
# ------------------------------------------------------------------------------------------------------------
def test_one_se_sensitivity_labels():
    both = f"{M.SEEN}; {M.SENS} (one-SE rule)"
    assert M.rule_label(M.SEEN, "NEST_BOTH_sh_classic") == both and M.rule_label(M.SEEN, "NEST_READ_raw_argmin") == both
    assert M.rule_label(M.SEEN, "NEST_BOTH_sh_argmin", "SH_r4") == both
    assert M.rule_label(M.SEEN, "NEST_BOTH_sh", "SH_r4") == M.SEEN and M.rule_label(M.SEEN, "LEVEL2_raw") == M.SEEN
    assert M.rule_label(M.SENS, "NEST_BOTH_sh_classic") == M.SENS


def _item_tables(both_passes: bool) -> dict:
    pops = ("PRIMARY", M.DECISION_POP, "WITH_C5", "PRIMARY_NOBIT", "PRIMARY_OFS", "PRIMARY_NO_ORNL_OFS")
    cols = {"cells": 9, "publications": 3, "units": 4, "rows_per_seed": 30.0, "cpg_clusters": 2, "kish_cpg": 1.1,
            "dR2_per_seed": "+0.0100", "SIGN_W/L/T": "3/1/0", "SIGN_p": 0.31, "p_signflip_unit": 0.25,
            "signflip_unit_exact": True, "p_signflip_chemotype_blocks": 0.5, "G_units": 4.0, "G_chemotypes": 2.0,
            "IMt_rho_p": 0.2, "IMt_log_p": 0.21, "eqw_pub_relSSE": 0.03, "lopo_lo": -0.01, "lopo_hi": 0.03,
            "top3_share": 0.9, "lopo_renested_lo": -0.02, "lopo_renested_hi": 0.04, "lopo_renested_min": -0.0042,
            "lopo_renested_min_by": "pub_x", "lopo_renested_max": 0.02, "lopo_renested_max_by": "pub_y"}
    inf = [{"stratum": s, "cand": "NEST_BOTH_sh", "ref": "SH_r4", "dR2": 0.011 + k / 1000, "label": M.SEEN, **cols}
           for k, s in enumerate(pops)]
    inf.append({"stratum": M.DECISION_POP, "cand": "NEST_BOTH_sh_classic", "ref": "SH_r4", "dR2": 0.0, "label": M.SENS})
    dep, ok = ("deploy the uniform shrunk mean (SH)",
               "nested weighting passes the in-corpus consistency check (EXPLORATORY; no claim)")
    rules = pd.DataFrame([{"rule": M.deploy_rule_name(f), "population": M.DECISION_POP, "dR2": g, "pubs_improve": 2,
                           "P": 3, "uniform_folds": 0, "folds": 3, "outcome": o, "label": M.CC}
                          for f, g, o in (("BOTH", 0.012, ok if both_passes else dep), ("ACID", -0.003, dep),
                                          ("READ", 0.02, ok))])
    nulls = pd.DataFrame([{"population": M.DECISION_POP, "null": n, "arm": "NEST_BOTH_sh", "p_perm": p}
                          for n, p in (("ACID_PERM", 0.9), ("READ_PERM", 0.004))])
    choices = pd.DataFrame([{"population": M.DECISION_POP, "family": "BOTH", "shrink": "sh", "rule": "classic",
                             "uniform_chosen": True} for _ in range(3)])
    comp = pd.DataFrame([{"stratum": "PRIMARY", "arm": a, "R2": 0.7, "MSE": 0.01, "relSSE_vs_B1_r4": 0.3,
                          "R2_per_seed": "0.7", "rows_per_seed": 30.0, "label": M.rule_label(M.SEEN, a)}
                         for a in ("SH_r4", "NEST_BOTH_sh", "NEST_BOTH_sh_classic")])
    units = pd.DataFrame([{"stratum": "PRIMARY", "contrast": "NEST_BOTH_sh - SH_r4", "publication": f"p{k}",
                           "chemotype": "sc009", "c": 0.003, "rows": 10.0, "rho": 0.1, "logratio": 0.1} for k in range(4)])
    return {"inference": pd.DataFrame(inf), "rules": rules, "nulls": nulls, "choices": choices, "comparators": comp,
            "units": units}


def test_per_item_sentences_tests_and_duplicates():
    from r5 import common as CM
    T = _item_tables(True)
    item = M.per_item(T, "BPG", seal_tag="TEST", leakage="ok", seconds=1.0)
    ch, nx = item["changed"], item["next_step"]
    assert "'nested weighting passes" in ch and "+0.0120" in ch and "2 of 3 publications" in ch
    assert "the read-match kernel carries it" in ch and "acid-permutation null p 0.900" in ch
    assert "classic one-SE rule (SENSITIVITY) uniform is chosen in 3 of 3 folds (dR2 +0.0000)" in ch
    assert "-0.0042 without pub_x" in ch
    assert nx.startswith("C-ECT stays sealed on SH@4 vs B1@4") and "read-match weighting" in nx
    assert list(item["units"].stratum) == ["PRIMARY", M.DECISION_POP] and item["units"].G_units.iloc[0] == 4
    assert {"SIGN_W/L/T", "SIGN_p", "p_signflip_chemotype_blocks", "IMt_rho_p", "IMt_log_p",
            "eqw_pub_relSSE"} <= set(item["units"].columns)
    assert list(item["duplicates"].stratum) == list(M.DUP_POPS)
    assert {"lopo_renested_min", "lopo_renested_max_by"} <= set(item["duplicates"].columns)
    txt = M.per_item_markdown(item, title="t")
    a, b, c = (txt.index(s) for s in ("## Unit W/L/T", "c_u per publication x chemotype unit", "## Duplicate"))
    assert a < b < c and txt.count("| p3 | sc009 |") == 1
    ins = (f"\nc_u per publication x chemotype unit, NEST_BOTH_sh - SH_r4 on PRIMARY (units_BPG.csv):\n\n"
           f"{CM.md_table(item['unit_table'])}\n")
    assert txt.replace(ins, "", 1) == CM.per_item_table(item, None, title="t")      # only the unit table is added
    nx2 = M.per_item(_item_tables(False), "BP", seal_tag="TEST", leakage="ok", seconds=1.0)["next_step"]
    assert nx2.startswith("C-ECT stays sealed") and "no sibling weighting is carried forward" in nx2


def test_fit_poisoning_has_a_positive_control():
    """s0-mechanics-16: the leaky selection (the outer publication's own losses included) must change for every
    kernel family, so a choose() that ignored its loss argument could not pass."""
    rng = np.random.default_rng(21)
    PM, y, sc, pc = _nest_data(rng, C=25)
    d = pd.DataFrame({"split_seed": sc, "publication": [f"p{q}" for q in pc], "y": y, "in_ACIDTREND": False})
    for shrink in M.SHRINKS:
        for k, c in enumerate(M.CONFIGS):
            d[M.cfg_name(c, shrink)] = PM[:, k]
    rep = M.fit_poisoning({"SYN": d}, "TEST", log=lambda *a: None)
    assert rep["identical"] == rep["fold_fits"] > 0 and rep["pc_runs"] == rep["fold_fits"]
    assert set(rep["pc_by_family"]) == set(M.FAMILIES) and all(v["caught"] > 0 for v in rep["pc_by_family"].values())
    assert rep["passed"]

