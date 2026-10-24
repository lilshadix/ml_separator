"""Tests for r5/ect.py and r5/ect_report.py (Stage-1 item 1.1, E-CT): synthetic data or construction-only checks; no
score is computed on real labels.

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B -W ignore -m pytest generations/gen21/round5_20261024/tests/test_ect.py -q
        -p no:cacheprovider --basetemp generations/gen21/round5_20261024/cache/_pytest

Covered: B1 against r4.refs.predict_levels; exact supports and linearity of the read-corrected template; target masking
and support validation; SH weights; the SH = OFF identity on aligned reads; bit-for-bit agreement of the external
(predict_sh) and in-corpus (Corpus.target_block) sibling averages on a synthetic pool of 10; OFF's empty-pool
policies; expectation arms against brute force; seeded subsets; the pool guard and both leakage positive controls (the
target itself, a
bit-identical copy); the prediction intervention on synthetic pools; legal ladder / HULL read sets; the external read
policy (construction only); the report helpers (R^2 per seed, unit contributions summing to the pooled contrast, exact
sign-flips against brute force, the report-rule labels and the mechanism row, the per-item table's unit-test field and
its p formatting); population and variant construction on the real metadata (counts only).

``test_p1_reproduction_requires_seal`` reproduces P1's PRIMARY B1 0.5309 / SH 0.7082 on the CLEAN population without
the bit-identity rule.  It REQUIRES THE ROUND-5 SEAL and runs only when R5_RUN_SEALED=1 is set after the item seal.
"""
from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r5 import paths  # noqa: E402,F401
from r5 import ect as E  # noqa: E402
from r5 import ect_report as REP  # noqa: E402
from r3 import interp  # noqa: E402
from r4 import refs as RF  # noqa: E402

ALL = tuple(range(14))
PAIRS = tuple((a, b) for a, b in zip(ALL[:-1], ALL[1:]) if (a, b) != (3, 4))       # Nd/Sm is not adjacent
READS = {2: frozenset({0, 12}), 3: frozenset({0, 12, 6}), 4: frozenset({0, 12, 6, 9})}


def series(rng, n=1, scale=0.5):
    base = np.linspace(-1.0, 1.5, 14)
    return base[None, :] + rng.normal(0.0, scale, (n, 14))


def target(rng, beta=-0.1, y=None, zero=True):
    y = series(rng)[0] if y is None else y
    z = rng.normal(0.0, 0.1, len(PAIRS)) if zero else None
    return E.make_target("t0", ALL, PAIRS, y, READS, beta, zero=z), y


# ------------------------------------------------------------------------------------------------------------
# predictor core
# ------------------------------------------------------------------------------------------------------------
def test_b1_levels_matches_r4_predict_levels_and_is_exact_at_reads():
    rng = np.random.default_rng(1)
    y = series(rng)[0]
    for rs in READS.values():
        ym = E.mask(y, rs)
        lev = E.b1_levels(ym, ALL, -0.11)
        ref = RF.predict_levels({"spec": RF.SPECS["B1"], "kind": "R", "coef": np.array([-0.11])},
                                SimpleNamespace(y_masked=ym), ALL)
        assert np.max(np.abs(lev - ref)) < 1e-12
        assert np.allclose(lev[sorted(rs)], y[sorted(rs)], atol=1e-14, rtol=0)
        assert np.max(np.abs(E.b0_levels(ym, ALL) - interp.pwl_levels(ym, ALL, "R"))) == 0.0


def test_corrected_templates_exact_at_reads_linear_and_equal_to_p1_formula():
    rng = np.random.default_rng(2)
    y = series(rng)[0]
    T = series(rng, 6)
    ym = E.mask(y, READS[4])
    X = E.corrected_templates(ym, ALL, T)
    rd = sorted(READS[4])
    assert np.allclose(X[:, rd], y[rd][None, :], atol=1e-14, rtol=0)
    assert np.max(np.abs(X.mean(0) - E.corrected_templates(ym, ALL, T.mean(0))[0])) < 1e-12
    for j in range(len(T)):                                     # P1_ct_build: Y_j + pwl_levels(y_S - Y_j,S)
        res = np.full(14, np.nan)
        res[rd] = y[rd] - T[j, rd]
        p1 = T[j] + interp.pwl_levels(res, ALL, "R")
        assert np.max(np.abs(X[j] - p1)) < 1e-14
    raw = E.corrected_templates(np.full(14, np.nan), ALL, T)
    assert np.array_equal(raw, T)


def test_make_target_masks_unread_values_and_validates_supports():
    rng = np.random.default_rng(3)
    t, y = target(rng)
    for r, rs in READS.items():
        assert set(np.where(np.isfinite(t.reads[r]))[0]) == set(rs)
    with pytest.raises(ValueError):                              # not nested
        E.make_target("t", ALL, PAIRS, y, {2: {0, 12}, 3: {1, 5, 9}}, 0.0)
    with pytest.raises(ValueError):                              # both ends of a scored pair read
        E.make_target("t", ALL, PAIRS, y, {2: {0, 1}}, 0.0)
    with pytest.raises(ValueError):                              # wrong size
        E.make_target("t", ALL, PAIRS, y, {3: {0, 12}}, 0.0)


def test_predict_sh_weights_and_same_read_shrinkage():
    rng = np.random.default_rng(4)
    t, _ = target(rng)
    Ys = series(rng, 3)
    out = E.predict_sh(t, Ys)
    w = 3 / (3 + 1.78)
    for r in (2, 3, 4):
        assert np.allclose(out[f"SH_r{r}"], w * out[f"SIB_r{r}"] + (1 - w) * out[f"B1_r{r}"], atol=1e-14)
    assert np.allclose(out["SHK2_r4"], 0.6 * out["SIB_r4"] + 0.4 * out["B1_r4"], atol=1e-14)
    assert np.allclose(out["SH0"], w * out["COPY"] + (1 - w) * t.zero, atol=1e-14)
    assert np.allclose(out["COPY"], E.pair_sf(Ys.mean(0), PAIRS), atol=1e-14)
    empty = E.predict_sh(t, np.zeros((0, 14)))
    assert np.isnan(empty["SH_r4"]).all() and np.isfinite(empty["B1_r4"]).all()
    assert E.K_SH == 1.78 and abs(E.weight(1) - 1 / 2.78) < 1e-15


def test_sh_equals_off_when_every_sibling_is_read_at_the_targets_metals():
    rng = np.random.default_rng(5)
    t, _ = target(rng, beta=-0.09)
    Ys = series(rng, 4)
    sh = E.predict_sh(t, Ys)["SH_r4"]
    resid = np.array([E.b1_residual_sf(Ys[j], ALL, READS[4], -0.09, PAIRS) for j in range(4)])
    off, n_p = E.off_sf(E.b1_sf(t, 4), resid)
    assert (n_p == 4).all()
    assert np.max(np.abs(sh - off)) < 1e-12


def _synthetic_block_inputs(rng, n):
    """A stand-in Corpus (only the attributes target_block reads), its design context and cache cell for one target
    't0' with n siblings in every point-arm pool; every other pool empty, no floor metals, no ladder."""
    Y = series(rng, n + 1)
    ids = ["t0"] + [f"s{k}" for k in range(1, n + 1)]
    var = {k: [] for k, _ in E.GUARD_POOLS}
    for k in ("bit", "NB", "V1", "INWIN") + E.VARIANTS_R4:
        var[k] = ids[1:]
    corpus = SimpleNamespace(pos={c: q for q, c in enumerate(ids)}, pools={"t0": {}}, var={"t0": var}, Y=Y, floor={},
                             _guarded=set(), acid=np.zeros(n + 1), ladder=lambda t, r: (None, "none"))
    seed, fold, beta = 1, 0, -0.1
    zero = rng.normal(0.0, 0.1, len(PAIRS))
    rows = pd.DataFrame({"fold": fold, "ia": [a for a, _ in PAIRS], "ib": [b for _, b in PAIRS], "G14@ZERO": zero})
    ctx = {"rows": rows, "groups": {(seed, "t0"): rows.index.to_numpy()}, "betas": {(seed, fold): beta}, "res": {}}
    cell = {"cell_id": "t0", "metals": ALL, "seqs": {"DOPT_Q": [(0, 12), (12, 6), (6, 9)]}}     # nested = READS
    return corpus, ctx, cell, seed, fold, E.make_target("t0", ALL, PAIRS, Y[0], READS, beta, zero=zero), Y


def test_external_and_in_corpus_paths_agree_bit_for_bit_on_a_pool_of_10():
    """predict_sh (C-ECT) and Corpus.target_block (in-corpus) average the siblings in the same array order: SH / SIB /
    COPY / SH0 (and SHK2) identical at n = 10, where an F- and a C-ordered mean over the pool differ by 1 ulp."""
    rng = np.random.default_rng(17)
    n = 10
    corpus, ctx, cell, seed, fold, T, Y = _synthetic_block_inputs(rng, n)
    blk = E.Corpus.target_block(corpus, ctx, cell, seed, fold, y_score=np.zeros(len(PAIRS)))
    ext = E.predict_sh(T, Y[1:])
    cols = ["COPY", "SH0", "SHK2_r4", "B1_r4"] + [f"{a}_r{r}" for a in ("SIB", "SH") for r in READS]
    for c in cols:
        assert np.array_equal(blk[c], ext[c]), c
    for pre in ("NB_", "V1_"):                                       # variant pools, every read budget
        for c in ("COPY", "SH0", "SHK2_r4") + tuple(f"{a}_r{r}" for a in ("SIB", "SH") for r in READS):
            assert np.array_equal(blk[pre + c], ext[c]), pre + c
    for pre in tuple(k + "_" for k in E.VARIANTS_R4) + ("INWIN_",):  # 4-read variant pools
        for c in ("COPY", "SH0", "SIB_r4", "SH_r4"):
            assert np.array_equal(blk[pre + c], ext[c]), pre + c
    # the identity is not vacuous: on this pool the two layouts of the same per-series matrix disagree in the last bit
    S = {r: E.template_sf(T, r, Y[1:]) for r in (0,) + tuple(READS)}
    assert all(s.flags["C_CONTIGUOUS"] for s in S.values())
    assert any(not np.array_equal(np.asfortranarray(s).mean(0), np.ascontiguousarray(s).mean(0)) for s in S.values())
    for r, s in S.items():
        col = "COPY" if r == 0 else f"SIB_r{r}"
        assert np.array_equal(E.pool_mean(np.asfortranarray(s)), ext[col]), col     # either layout in, one order out
        assert np.array_equal(E.pool_mean(s[list(range(n))]), blk[col]), col         # target_block's row indexing


def test_off_empty_policies_and_partial_donors():
    base = np.array([0.1, -0.2, 0.3])
    v0, n0 = E.off_sf(base, np.zeros((0, 3)))
    assert np.array_equal(v0, base) and (n0 == 0).all()
    vn, _ = E.off_sf(base, np.zeros((0, 3)), empty="nan")
    assert np.isnan(vn).all()
    R = np.array([[0.2, np.nan, 0.1], [0.4, np.nan, np.nan]])
    v, n = E.off_sf(base, R, empty="nan")
    assert list(n) == [2, 0, 1]
    assert abs(v[0] - (0.1 + 0.6 / 3.78)) < 1e-15 and np.isnan(v[1]) and abs(v[2] - (0.3 + 0.1 / 2.78)) < 1e-15


def test_expectation_arms_and_seeded_subsets():
    rng = np.random.default_rng(6)
    S = rng.normal(size=(6, 5))
    y = rng.normal(size=5)
    e1 = E.expected_se(S, y)
    assert np.allclose(e1, np.mean([(S[j] - y) ** 2 for j in range(6)], axis=0))
    idx = E.subset_index(6, 2, "k")
    assert len(idx) == 15 and {tuple(r) for r in idx} == set(itertools.combinations(range(6), 2))
    e2 = E.expected_se(E.draw_means(S, idx), y)
    assert np.allclose(e2, np.mean([((S[a] + S[b]) / 2 - y) ** 2 for a, b in itertools.combinations(range(6), 2)], axis=0))
    big1, big2 = E.subset_index(18, 5, "ECT|SUB|x|m5"), E.subset_index(18, 5, "ECT|SUB|x|m5")
    assert big1.shape == (5000, 5) and np.array_equal(big1, big2)
    assert not np.array_equal(big1, E.subset_index(18, 5, "ECT|SUB|y|m5"))
    assert (np.diff(big1, axis=1) > 0).all()
    assert E.subset_index(3, 5, "k").shape[0] == 0


# ------------------------------------------------------------------------------------------------------------
# leakage: guard, positive controls, intervention
# ------------------------------------------------------------------------------------------------------------
def test_guard_catches_the_target_itself_and_a_bit_identical_copy():
    rng = np.random.default_rng(7)
    y = series(rng)[0]
    other = series(rng)[0]
    E.guard_pool("t", y, ["a"], other[None, :])                                  # unrelated series passes
    with pytest.raises(E.LeakageError):
        E.guard_pool("t", y, ["a", "t"], np.vstack([other, y]))                 # positive control 1: itself
    with pytest.raises(E.LeakageError):
        E.guard_pool("t", y, ["copy"], y[None, :])                              # positive control 2: a copy
    with pytest.raises(E.LeakageError):
        E.guard_pool("t", y, ["t"], y[None, :], allow_bit=True)                 # itself is never allowed
    E.guard_pool("t", y, ["copy"], y[None, :], allow_bit=True)                  # NB / V1 sensitivity pools
    two = other.copy()
    two[:2] = y[:2]
    E.guard_pool("t", y, ["a"], two[None, :])                                   # 2 identical metals: not flagged
    six = other.copy()
    six[:6] = y[:6]
    E.guard_pool("t", y, ["a"], six[None, :])                                   # 6 of 14 (< 50 %): not flagged
    sev = other.copy()
    sev[:7] = y[:7]
    with pytest.raises(E.LeakageError):
        E.guard_pool("t", y, ["a"], sev[None, :])                               # 7 of 14 (>= 50 %, >= 3): flagged
    assert E.bit_identical(y, y) and not E.bit_identical(y, other)


def test_intervention_on_unread_labels_and_its_positive_control():
    rng = np.random.default_rng(8)
    y = series(rng)[0]
    Ys = series(rng, 3)
    t, _ = target(rng, y=y)
    base = E.predict_sh(t, Ys)
    yp = y.copy()
    unread = [m for m in ALL if m not in READS[4]]
    yp[unread] = rng.normal(0.0, 3.0, len(unread))
    tp = E.make_target("t0", ALL, PAIRS, yp, READS, t.beta, zero=t.zero)
    pois = E.predict_sh(tp, Ys)
    assert all(np.array_equal(base[k], pois[k], equal_nan=True) for k in base)
    leak_a = E.predict_sh(t, np.vstack([Ys, y]))                    # positive control: target in its own pool
    leak_b = E.predict_sh(tp, np.vstack([Ys, yp]))
    assert not np.array_equal(leak_a["SH_r4"], leak_b["SH_r4"])    # caught by the intervention


# ------------------------------------------------------------------------------------------------------------
# read policies
# ------------------------------------------------------------------------------------------------------------
def test_ladder_and_hull_read_sets_are_legal():
    from r5 import frozen as FR
    from gen20k import design as DS
    dz = FR.design_for_metals(ALL)
    for r, names in E.LADDER.items():
        rs, src = E.ladder_reads(dz, r)
        assert src == "ladder" and rs == frozenset(E.LN.index(m) for m in names)
    no_ho = tuple(m for m in ALL if E.LN[m] != "Ho")
    dz2 = FR.design_for_metals(no_ho)
    for r in (2, 3):
        rs, src = E.ladder_reads(dz2, r)
        assert src == "hull" and len(rs) == r
        assert DS.max_independent_size(dz2.cell.metals, dz2.query, rs) >= 0       # no query pair fully read
        assert min(no_ho) in rs and max(no_ho) in rs
    short = (0, 1, 2)
    rs, src = E.ladder_reads(FR.design_for_metals(short), 3)
    assert rs is None and src == "none"


def test_external_read_sets_are_nested_and_values_only_at_reads():
    if not (paths.CACHE / "ext" / "ext_dopt.pkl").exists():
        pytest.skip("run scripts/s0_external_dopt.py first")
    from r5 import frozen as FR
    ext = FR.load_external()
    rs = E.external_read_sets(ext, ALL)
    assert [len(rs[r]) for r in (2, 3, 4)] == [2, 3, 4] and rs[2] < rs[3] < rs[4]
    vals = {m: 0.1 * m for m in rs[4]}
    t = E.external_target("new", ALL, PAIRS, vals, ext, beta=-0.1)
    assert set(np.where(np.isfinite(t.reads[4]))[0]) == set(rs[4])
    with pytest.raises(ValueError):
        E.external_target("new", ALL, PAIRS, {m: 0.0 for m in list(rs[4])[:2]}, ext, beta=-0.1)


# ------------------------------------------------------------------------------------------------------------
# report helpers (synthetic rows)
# ------------------------------------------------------------------------------------------------------------
def synthetic_rows(seed=9, n_pub=5, cells_per=4, seeds=(1, 2, 3)):
    rng = np.random.default_rng(seed)
    rows = []
    for s in seeds:
        for p in range(n_pub):
            for c in range(cells_per):
                for q in range(6):
                    y = rng.normal(0, 0.3)
                    rows.append({"split_seed": s, "publication": f"pub{p}", "chemotype": f"ch{p % 3}", "cell_id": f"c{p}_{c}",
                                 "cluster_cpg": f"g{p % 2}", "y": y, "A": "La", "B": "Ce",
                                 "ref": y + rng.normal(0, 0.2), "cand": y + rng.normal(0, 0.15)})
    d = pd.DataFrame(rows)
    d["pubchem"] = d.publication + "|" + d.chemotype
    d["E_x"] = (d.cand - d.y) ** 2
    return d


def test_r2_per_seed_and_unit_contributions_sum_to_the_pooled_contrast():
    d = synthetic_rows()
    for s, g in d.groupby("split_seed"):
        r = 1 - ((g.y - g.cand) ** 2).sum() / ((g.y - g.y.mean()) ** 2).sum()
        assert abs(REP.r2_seeds(d, "cand")[sorted(d.split_seed.unique()).index(s)] - r) < 1e-12
    assert abs(REP.r2(d, "E_x") - REP.r2(d, "cand")) < 1e-12               # expectation arms enter as SSE
    u = REP.unit_contrib(d, "cand", "ref")
    assert abs(u.c.sum() - (REP.r2(d, "cand") - REP.r2(d, "ref"))) < 1e-12
    row, units = REP.infer(d, "cand", "ref", stratum_name="S", label=REP.SEEN)
    assert abs(row["dR2"] - row["contribution_total"]) < 1e-12 and len(units) == row["G_units"]
    assert row["label"] == REP.SEEN and np.isfinite(row["lopo_lo"])


@pytest.mark.parametrize("G", [1, 3, 6, 9])
def test_exact_signflip_matches_brute_force(G):
    rng = np.random.default_rng(G)
    x = rng.normal(0.1, 1.0, G)
    T = x.sum()
    hits = sum(np.dot(s, x) >= T - 1e-10 * np.abs(x).sum() for s in itertools.product((1, -1), repeat=G))
    assert abs(REP.signflip_p(x)["p"] - hits / 2 ** G) < 1e-15
    blocks = np.arange(G) % 2
    xb = np.array([x[blocks == b].sum() for b in sorted(set(blocks))])
    hb = sum(np.dot(s, xb) >= xb.sum() - 1e-10 * np.abs(xb).sum() for s in itertools.product((1, -1), repeat=len(xb)))
    assert abs(REP.signflip_p(x, blocks)["p"] - hb / 2 ** len(xb)) < 1e-15


def test_rules_labels_and_the_mechanism_row():
    inf = [{"stratum": "PRIMARY", "cand": "SH_r4", "ref": "B1_r4", "dR2": 0.17, "lopo_lo": 0.05, "lopo_hi": 0.28,
            "units_pos": 14, "G_units": 14, "R2_ref": 0.53}]
    dup = pd.DataFrame({"stratum": ["PRIMARY (x)", "PRIMARY_NOBIT (x)", "PRIMARY_I1 (x)"], "dR2": [0.17, 0.18, 0.17],
                        "lopo_lo": 0.05, "lopo_hi": 0.28, "units_pos": [14, 13, 14], "units": 14})
    excl = pd.DataFrame({"stratum": ["PRIMARY"], "rows": ["all pairs"], "Ce_share_of_SH-B1_gain": [0.33],
                         "Gd_share_of_SH-B1_gain": [0.25]})
    mech = pd.DataFrame({"stratum": "PRIMARY", "pool_variant": "primary (bit rule)",
                         "comparator": ["OFF_XEXT", "OFF_XEXT", "PB_PUB", "PB_PUB", "OFF_XEXT"],
                         "rows": ["rows where OFF_XEXT exists"] * 2 + ["rows where PB_PUB exists"] * 2
                         + ["rows where both OFF_XEXT and PB_PUB exist"],
                         "publication": ["POOLED", "pub_a", "POOLED", "pub_a", "POOLED"],
                         "dR2_SH-cmp": [0.044, np.nan, 0.045, np.nan, 0.9], "recovered_share": [0.467, 1.145, 0.458, 1.001, 0.1],
                         "families_with_other_extractant_pool": [6, np.nan, 6, np.nan, 6],
                         "publications_with_other_extractant_pool": [3, np.nan, 3, np.nan, 3]})
    out = REP.t_rules(None, inf, excl, mech, dup)
    assert list(out.label[:3]) == [REP.SEEN, REP.SENS, REP.SENS] and set(out.label[3:]) == {REP.CC}
    assert list(out.units_up[:3]) == ["14 of 14", "13 of 14", "14 of 14"]
    m = out[out.rule.str.startswith("mechanism wording")].iloc[0]
    assert m.value == 0.044 and m["SH-PB_PUB"] == 0.045 and m.families_with_other_extractant_pool == 6
    assert m.recovered_share_OFF_XEXT == "pooled 0.467; pub_a 1.145" and m.recovered_share_PB_PUB == "pooled 0.458; pub_a 1.001"
    assert m.families == "6 families (publication x extractant) in 3 publications"


def test_per_item_table_prints_the_unit_tests_beside_the_unit_table(tmp_path):
    d = synthetic_rows()
    row, units = REP.infer(d, "cand", "ref", stratum_name="PRIMARY", label=REP.SEEN)
    item = {k: "n/a" for k, _ in REP.CM.ITEM_FIELDS}
    item.update({"label": REP.SEEN, "units": units[["publication", "chemotype", "c"]], "units_tests": REP.unit_tests_txt(row)})
    txt = REP.per_item_table(item, tmp_path / "t.md", title="t")
    field = next(x for x in txt.splitlines() if x.startswith("| Unit W/L/T and the descriptive tests |"))
    assert f"SIGN W/L/T {row['SIGN_W/L/T']}, SIGN p {REP.fmt_p(row['SIGN_p'])}" in field and "IM-t" in field
    assert "equal-weight per-publication relative SSE" in field and field.endswith("below |")
    assert "## Unit W/L/T and the descriptive tests" in txt and (tmp_path / "t.md").read_bytes().decode("utf-8") == txt
    assert REP.fmt_p(6.1035e-05) == "6.10e-05" and REP.fmt_p(0.015625) == "0.0156" and REP.fmt_p(np.nan) == "NaN"


def test_surrogate_keeps_nan_pattern_and_replaces_labels():
    d = synthetic_rows()
    d.loc[::7, "cand"] = np.nan
    S = REP.surrogate(d, "BPG")
    assert np.array_equal(S.cand.isna(), d.cand.isna())
    assert not np.allclose(S.y, d.y) and (S.E_x >= 0).all()


# ------------------------------------------------------------------------------------------------------------
# construction-only on the real metadata (counts; no label of a target, no score)
# ------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def corpus():
    return E.Corpus(log=lambda *a, **k: None)


def test_population_and_variant_pools_construction(corpus):
    C = corpus
    assert len(C.c3) == 229 and len(C.bit_links) == 6
    for t, v in C.var.items():
        bit = set(v["bit"])
        for k in ("bit", "NB", "V1", "I1", "SD", "OFS", "QNT", "V25", "INWIN", "OUTWIN", "near", "acid", "off", "off_all",
                  "off_pub", "off_xext", "NB_off", "V1_off") + tuple("PB_" + p for p in E.PB_POOLS):
            assert t not in v[k], (t, k)
        for k in ("SD", "OFS", "QNT", "INWIN", "OUTWIN", "near", "acid"):
            assert set(v[k]) <= bit, (t, k)
        assert set(v["I1"]) == {s for s in C.pools[t]["clean"] if frozenset((t, s)) != E.ITEM1}
        assert all(frozenset((t, s)) not in C.bits for s in v["bit"])
        assert set(v["INWIN"]) | set(v["OUTWIN"]) == bit and not set(v["INWIN"]) & set(v["OUTWIN"])
    tt = C.target_table()
    assert int(tt.in_PRIMARY.sum()) == 99 and int(tt.in_PRIMARY_NOBIT.sum()) == 99 and int(tt.in_EQ1.sum()) == 16
    item1 = [c for c in E.ITEM1 if c in C.var]
    assert item1 and all(len(C.var[c]["I1"]) == len(C.pools[c]["clean"]) - 1 for c in item1)


def test_the_target_itself_in_its_pool_is_caught_through_the_builder(corpus):
    """Positive control through r5.ect.Corpus.target_block (cache cells and REF rows; predictions only)."""
    from g21 import cache as GC
    ctx = corpus.design_ctx("BPG")
    seed, fold = paths.DEV_SEEDS[0], 0
    rec = GC.load("BPG", seed, fold)
    cell = next(c for c in rec["outer"]["cells"] if (seed, c["cell_id"]) in ctx["groups"] and corpus.var[c["cell_id"]]["bit"])
    with pytest.raises(E.LeakageError):
        corpus.target_block(ctx, cell, seed, fold, y_score=np.zeros(len(ctx["groups"][(seed, cell["cell_id"])])),
                            inject={"bit": [cell["cell_id"]]})


# ------------------------------------------------------------------------------------------------------------
# reproduction (REQUIRES THE SEAL; never run before it)
# ------------------------------------------------------------------------------------------------------------
@pytest.mark.skipif(os.environ.get("R5_RUN_SEALED") != "1",
                    reason="requires the round-5 seal: set R5_RUN_SEALED=1 after the Stage-0 and item seals")
def test_p1_reproduction_requires_seal(corpus):
    """In-corpus r5/ect.py on P1's CLEAN population without the bit-identity rule reproduces PRIMARY B1 0.5309 and
    SH 0.7082 (PC P1), BPG.  TECHNICAL SMOKE; requires the seal."""
    from r5 import sealcheck
    sealcheck.require_seal()
    R, checks = corpus.build_rows("BPG")
    rep = REP.reproduction(R, checks, "BPG")
    t = rep["targets"]["P1_PRIMARY_NOBIT"]
    assert t["counts_ok"]
    assert abs(t["values_found"]["B1_r4"] - 0.5309) <= REP.REPRO_TOL
    assert abs(t["values_found"]["NB_SH_r4"] - 0.7082) <= REP.REPRO_TOL
    assert rep["b1_ok"]
