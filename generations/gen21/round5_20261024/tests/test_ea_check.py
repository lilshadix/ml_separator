"""Tests for r5/ea_check.py and scripts/s1_ea_check.py (item 1.0 EA_CHECK).  Synthetic data for every statistic (the
M35 gate legs, unit / big-fold / breadth / six-design tables, the M08 moments against brute force) and
construction-only checks on real data (isolation, intervention with its peek positive control, the one-fold
reproduction identity, the cache manifest, the seal gate of the script).  No round-5 score is computed.

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B -W ignore -m pytest generations/gen21/round5_20261024/tests/test_ea_check.py
        -q -p no:cacheprovider --basetemp generations/gen21/round5_20261024/cache/_pytest   (one command line)
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r5 import paths  # noqa: E402
from r5 import common as CM  # noqa: E402
from r5 import ea_check as EA  # noqa: E402

PAIRS = [tuple(p.split("/")) for p in EA.PAIRS]


# ------------------------------------------------------------------------------------------------------------
# synthetic rows
# ------------------------------------------------------------------------------------------------------------
SIZES = [12, 6, 5, 4, 3, 3, 2, 2, 1, 1, 1, 1, 1]          # cells per cluster: 13 clusters
ORNL_ONLY = {5, 6}                                        # clusters whose every cell is an ORNL import
ORNL_PART = {7}


def synth(seed: int = 0, n_seeds: int = 5, noise=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(rows, provenance) with 13 clusters, 4 pair types per cell, C5-BTBP = one publication of cluster 1."""
    rng = np.random.default_rng(seed)
    noise = noise or {"B0": 0.30, "B1": 0.20, "HIST_D7": 0.22}
    cells, prov = [], []
    k = 0
    for g, n in enumerate(SIZES):
        for j in range(n):
            pub = paths.C5 if (g == 1 and j < 3) else f"pub_{g}_{j % 3 if g == 0 else j % 2}"
            cid = paths.C5_CELL if (g == 1 and j == 0) else f"cell{k:04d}"
            cells.append({"cell_id": cid, "cluster_cpg": f"g{g:02d}", "publication": pub, "chemotype": f"sc{g:03d}",
                          "extractant": f"ext{g}_{j // 2}", "fold": 0 if g == 0 else (1 if g == 1 else 2 + g % 3),
                          "pairs": [PAIRS[(k + i) % 12] for i in range(4)]})
            ornl = g in ORNL_ONLY or (g in ORNL_PART and j == 0)
            prov.append({"cell_id": cid, "loc_class": ["FIG", "TAB", "NONE"][k % 3], "ornl_import": ornl,
                         "curator": "x"})
            k += 1
    rows = []
    for s in range(n_seeds):
        for c in cells:
            for a, b in c["pairs"]:
                y = rng.normal(0, 0.5)
                r = {"split_seed": 1000 + s, "fold": c["fold"], "cell_id": c["cell_id"], "A": a, "B": b, "y": y,
                     "publication": c["publication"], "chemotype": c["chemotype"], "extractant": c["extractant"],
                     "cluster_cpg": c["cluster_cpg"], "k": 3, "k_max": 3, "read_set": "La,Nd,Gd,Lu",
                     "a_read": False, "b_read": False}
                for arm, sd in noise.items():
                    r[arm] = y + rng.normal(0, sd)
                rows.append(r)
    t = pd.DataFrame(rows).sort_values(CM.KEY).reset_index(drop=True)
    t["pubchem"] = t.publication + "|" + t.chemotype
    return t, pd.DataFrame(prov).set_index("cell_id")


@pytest.fixture(scope="module")
def syn():
    return synth()


# ------------------------------------------------------------------------------------------------------------
# provenance, labels, small helpers
# ------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("s, want", [("Figure 2", "FIG"), ("Table SI1", "TAB"), ("Figure 3;Table 4", "MIXED"),
                                     ("Graph 4;Graph 5", "FIG"), ("", "NONE"), (None, "NONE"), (float("nan"), "NONE"),
                                     ("comment", "OTHER"), ("  ", "NONE")])
def test_loc_class(s, want):
    assert EA.loc_class(s) == want


def test_labels():
    assert EA.label("BPG") == "EXPLORATORY (seen); DIAGNOSTIC"
    assert "SENSITIVITY" in EA.label("BP") and "SENSITIVITY" in EA.label("BPG", "C5_out")
    assert EA.label("A", "C5_in", "x").endswith("; x")


def test_outputs_stay_inside_round5():
    for p in (EA.OUT, EA.LEAK, EA.MANIFEST, EA.POISON_REPORT, EA.INTERVENTION_REPORT, EA.cache_path("A", 104729, 0)):
        assert paths.R5 in Path(p).resolve().parents
    assert EA.cache_path("BQ", 1, 2).name == "BQ_1_2.pkl" and EA.cache_path("BQ", 1, 2).parent == paths.CACHE


def test_same_partition():
    assert EA.same_partition(["a", "a", "b"], ["x", "x", "y"])
    assert not EA.same_partition(["a", "a", "b"], ["x", "y", "y"])
    assert not EA.same_partition(["a", "b"], ["x", "x"])


def test_diff_paths():
    a = {"x": np.array([1.0, np.nan]), "y": [(1, 2), (3, 4)], "z": {"w": 0.5, "s": "t"}, "i": np.array([1, 2])}
    b = {"x": np.array([1.0, np.nan]), "y": [(1, 2), (3, 4)], "z": {"w": 0.5, "s": "t"}, "i": np.array([1, 2])}
    assert EA.diff_paths(a, b) == []
    b2 = dict(b, x=np.array([1.0, 2.0]))
    assert EA.diff_paths(a, b2) == ["/x: array differs"]
    assert EA.diff_paths(a, dict(b, z={"w": 0.5 + 2 ** -52, "s": "t"})) == ["/z/w: 0.5 != 0.5000000000000002"]
    assert EA.diff_paths(a, dict(b, i=np.array([1.0, 2.0])))            # dtype change counts
    assert EA.diff_paths(a, {k: v for k, v in b.items() if k != "z"}) == [": keys differ"]
    assert EA.diff_paths(a, dict(b, y=[(1, 2)])) == ["/y: length differs"]


def test_registry_status(tmp_path):
    p = tmp_path / "REGISTRY.csv"
    assert EA.registry_status("M08", p)["status"] == "MISSING"
    pd.DataFrame({"id": ["1.0", "1.0-app"], "item": ["EA_CHECK", "M08 moment table"],
                  "status": ["PLANNED", "planned"]}).to_csv(p, index=False)
    assert EA.registry_status("M08", p)["status"] == "PLANNED"
    pd.DataFrame({"id": ["1.0-app"], "item": ["M08 appendix"], "Status": ["EXCLUDED"]}).to_csv(p, index=False)
    assert EA.registry_status("M08", p)["status"] == "EXCLUDED"
    pd.DataFrame({"id": ["1.1"], "item": ["ECT"], "status": ["PLANNED"]}).to_csv(p, index=False)
    assert EA.registry_status("M08", p)["status"] == "MISSING"


def test_rng_seeds_are_registry_scheme():
    assert EA.rng_seed("poison", "A", 104729, 0) == paths.seed_for("EA_CHECK|poison|A|104729|0")
    assert EA.rng_seed("poison", "A", 104729, 0) != EA.rng_seed("poison", "A", 104729, 1)


# ------------------------------------------------------------------------------------------------------------
# the M35 gate legs (synthetic)
# ------------------------------------------------------------------------------------------------------------
def test_gate_passes_a_uniform_strong_gain(syn):
    t, _ = syn
    t = t.copy()
    t["B1"] = t.y + 0.5 * (t.B0 - t.y)                     # every row's error halved: every unit wins
    row, us = EA.contrast_stats(t, "B1", "B0", "test")
    assert row["G"] == 13 and row["SF_POOL_exact"] and row["SF_POOL_n_flips"] == 2 ** 13
    assert row["W/L/T"] == "13/0/0" and row["SF_POOL_p"] == pytest.approx(1 / 8192, abs=0)
    assert row["relSSE"] == pytest.approx(0.75, abs=1e-12)
    assert row["gate_pass"] and row["dMAE_bal"] > 0 and row["seeds_relSSE_positive"] == 5
    assert len(us) == 13 and np.allclose(us.dSSE, us.SSE_ref - us.SSE_cand)


def test_gate_legs_are_independent(syn):
    t, _ = syn
    t = t.copy()
    t["W"] = t.y + 1.2 * (t.B0 - t.y)                      # uniformly worse
    row, _ = EA.contrast_stats(t, "W", "B0", "test")
    assert not row["gate_pass"] and not row["gate_relSSE_ge_0.02"] and row["W/L/T"] == "0/13/0"
    # one huge error per seed removed (relative SSE large) while every other row gets worse: guard and SF_POOL fail
    u = t.copy()
    u["R"] = u.y + 0.01
    u["C"] = u.y + 0.05
    big = u[u.cluster_cpg == "g00"].groupby("split_seed").head(1).index
    u.loc[big, "R"] = u.loc[big, "y"] + 3.0
    u.loc[big, "C"] = u.loc[big, "y"]
    row, _ = EA.contrast_stats(u, "C", "R", "test")
    assert row["gate_relSSE_ge_0.02"] and not row["guard_dMAE_bal_ge_-0.001"] and not row["gate_SF_POOL_p_le_0.025"]
    assert not row["gate_pass"]
    for k in ("relSSE", "SF_POOL_p", "dMAE_bal"):
        assert np.isfinite(row[k])


def test_gate_rule_is_the_conjunction(syn):
    t, _ = syn
    rng = np.random.default_rng(3)
    for i in range(6):
        u = t.assign(Z=t.y + rng.uniform(0.6, 1.1) * (t.B0 - t.y) + rng.normal(0, 0.05, len(t)))
        row, _ = EA.contrast_stats(u, "Z", "B0", f"t{i}")
        want = row["relSSE"] >= 0.02 and row["SF_POOL_p"] <= 0.025 and row["dMAE_bal"] >= -0.001
        assert row["gate_pass"] == want


# ------------------------------------------------------------------------------------------------------------
# mandatory tables (synthetic)
# ------------------------------------------------------------------------------------------------------------
def test_unit_table_provenance(syn):
    t, prov = syn
    row, us = EA.contrast_stats(t, "B1", "B0", "u")
    pu = EA.unit_provenance(t, prov)
    tab = EA.unit_table(us, pu, "L")
    assert len(tab) == 13 and tab.share_of_dSSE.sum() == pytest.approx(1.0)
    assert set(tab.index[tab.ornl_only]) == {f"g{g:02d}" for g in ORNL_ONLY}
    assert tab.loc["g07", "n_ornl_import_cells"] == 1 and not tab.loc["g07", "ornl_only"]
    assert tab.loc["g00", "n_cells"] == 12 and tab.loc["g00", "rows_per_seed"] == 48
    assert tab.rows_per_seed.is_monotonic_decreasing and (tab.label == "L").all()
    assert all(isinstance(s, str) and s for s in tab.data_location_classes)


def test_big_fold(syn, tmp_path, monkeypatch):
    t, _ = syn
    b = EA.add_big_fold(t)
    assert set(b[b.big_fold].fold) == {0} and b.groupby("split_seed").big_fold.sum().eq(48).all()
    monkeypatch.setattr(EA, "FOLD_TRAINING", tmp_path / "fold_training_{design}.csv")
    with pytest.raises(FileNotFoundError):            # s0-mechanics-9: a missing fold-training table is never skipped
        EA.bigfold_table(b, "B1", "B0", "SYN", "k")
    pd.DataFrame({"seed": sorted(b.split_seed.unique()) * 5,
                  "fold": [f for f in range(5) for _ in sorted(b.split_seed.unique())],
                  "train_c3_cells": 7}).to_csv(tmp_path / "fold_training_SYN.csv", index=False)
    tab = EA.bigfold_table(b, "B1", "B0", "SYN", "k")
    assert list(tab.part) == ["big fold", "other folds"]
    assert tab.share_of_rows.sum() == pytest.approx(1.0) and tab.share_of_dSSE.sum() == pytest.approx(1.0)
    assert tab.folds.iloc[0].count(":0") == 5 and set(tab.train_c3_cells.iloc[0].split(";")) == {"7"}


def test_breadth_variants(syn):
    t, prov = syn
    pu = EA.unit_provenance(t, prov)
    sp = EA.breadth_units(t, pu, "SMALL_POOLED")
    kept = set(sp.unique()) - {"small sources"}
    assert kept == {"g00", "g01", "g02"}                    # g03 has 4 cells but 16 rows per seed (< 20)
    oc = EA.breadth_units(t, pu, "ORNL_COLLAPSED")
    assert "ORNL-only" in set(oc) and not ({"g05", "g06"} & set(oc)) and "g07" in set(oc)
    assert (EA.breadth_units(t, pu, "RAW") == t.cluster_cpg).all()
    s, u, gp = EA.breadth(t, "B1", "B0", pu, "BPG", "C5_in", "k")
    assert list(s.variant) == ["SMALL_POOLED", "ORNL_COLLAPSED", "RAW"] and list(s.G) == [4, 12, 13]
    raw = u[u.variant == "RAW"]
    assert s.mean_L.iloc[2] == pytest.approx(float(raw.L.mean()))
    assert set(gp.giant_cluster) == {"g00"} and set(gp.publication) == {"pub_0_0", "pub_0_1", "pub_0_2"}
    assert gp.share_of_giant_dSSE.sum() == pytest.approx(1.0)


def test_lopo_table(syn):
    t, _ = syn
    j, v = EA.lopo_table(t, "B1", "B0")
    assert j["P"] == t.publication.nunique() == len(v)
    assert j["point"] == pytest.approx(CM.rel_sse(t, "B1", "B0"))


def _m08_brute(g: pd.DataFrame, resid="B1") -> dict:
    e = {(c, f"{a}/{b}"): y - p for c, a, b, y, p in zip(g.cell_id, g.A, g.B, g.y, g[resid])}
    meta = g.drop_duplicates("cell_id").set_index("cell_id")
    cells = list(meta.index)
    acc = {}
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            ci, cj = cells[i], cells[j]
            se = meta.extractant[ci] == meta.extractant[cj]
            sp = meta.publication[ci] == meta.publication[cj]
            sc = meta.chemotype[ci] == meta.chemotype[cj]
            cls = ("SE_SP" if sp else "SE_OP") if se else (("OE_SP_SC" if sc else "OE_SP_OC") if sp else
                                                          ("OE_OP_SC" if sc else "OE_OP_OC"))
            for p in EA.PAIRS:
                if (ci, p) in e and (cj, p) in e:
                    n, di, dj = acc.get(cls, (0.0, 0.0, 0.0))
                    acc[cls] = (n + e[(ci, p)] * e[(cj, p)], di + e[(ci, p)] ** 2, dj + e[(cj, p)] ** 2)
    return {c: n / math.sqrt(di * dj) for c, (n, di, dj) in acc.items()}


def test_m08_matches_brute_force(syn):
    t, _ = syn
    g = t[t.split_seed == t.split_seed.min()]
    out, prods, loco = EA._m08_seed(g, "B1")
    want = _m08_brute(g)
    for c, r in want.items():
        assert out[c]["rho"] == pytest.approx(r, rel=1e-10, abs=1e-12)
    for c in out:
        if c not in want:
            assert math.isnan(out[c]["rho"]) and out[c]["cell_pairs"] == 0
    assert set(prods["class"]) <= {c for c, _ in EA.M08_CLASSES}
    assert set(loco.cluster_left_out) == set(g.cluster_cpg)


def test_m08_shared_residual_is_found():
    t, _ = synth(seed=5)
    rng = np.random.default_rng(9)
    off = {k: rng.normal(0, 0.4) for k in set(zip(t.extractant, t.publication, t.A, t.B, t.split_seed))}
    t["B1"] = t.y - np.array([off[k] for k in zip(t.extractant, t.publication, t.A, t.B, t.split_seed)])
    m = EA.m08_moments(t, "BPG", "C5_in")
    r = m.set_index("class").rho
    assert r["SE_SP"] == pytest.approx(1.0, abs=1e-9) and abs(r["OE_OP_OC"]) < 0.3
    assert set(m.identifiability.str.split(" ").str[0]) <= {"identified", "not"}
    assert (m.label.str.contains("M08 appendix")).all()


def test_score_design_and_consistency(syn):
    t, prov = syn
    res = EA.score_design(t, prov, "BPG", m08=True)
    assert {"gate", "c5", "units", "perseed", "bigfold", "lopo", "lopo_values", "breadth", "breadth_units",
            "giant_publications", "arms", "m08_moments"} <= set(res)
    g = res["gate"]
    assert len(g) == len(EA.CONTRASTS) * len(EA.STRATA)
    assert set(g.stratum) == set(EA.STRATA) and set(g.contrast) == {"B1 - B0", "HIST_D7 - B1"}
    assert g[g.stratum == "C5_in"].rows_per_seed.iloc[0] > g[g.stratum == "C5_out"].rows_per_seed.iloc[0]
    for name, df in res.items():
        assert "label" in df.columns and df.label.astype(str).str.contains("EXPLORATORY \\(seen\\)").all(), name
    assert set(res["perseed"].split_seed) == set(t.split_seed)
    c = EA.consistency(g)
    assert set(c) == {"B1 - B0 BPG"} and c["B1 - B0 BPG"]["label"] == EA.L_CC
    fake = g.copy()
    fake.loc[(fake.contrast == "B1 - B0") & (fake.stratum == "C5_in"), ["relSSE", "SF_POOL_p"]] = [0.05851, 26 / 8192]
    c = EA.consistency(fake)["B1 - B0 BPG"]
    assert c["relSSE_pct"]["match"] and c["SF_POOL_p"]["match"]


def test_consistency_carries_the_pooled_over_seeds_value(syn):
    t, prov = syn
    g = EA.score_design(t, prov, "BPG")["gate"]
    c = EA.consistency(g, {"BPG": t})["B1 - B0 BPG"]["relSSE_pct"]
    pooled = 1 - ((t.B1 - t.y) ** 2).sum() / ((t.B0 - t.y) ** 2).sum()
    assert c["now_pooled_over_seeds"] == pytest.approx(pooled, rel=1e-12)
    assert c["now_seed_mean"] == pytest.approx(CM.rel_sse(t, "B1", "B0"), rel=1e-12)
    assert c["now"] == round(100 * CM.rel_sse(t, "B1", "B0"), 2) and "pooled-over-seeds" in c["note"]
    assert c["match"] == (c["now"] == c["seen"])                 # judged on the sealed seed mean, as registered
    assert EA.consistency(g) == {k: {**v, "relSSE_pct": {x: v["relSSE_pct"][x] for x in ("seen", "now", "match")}}
                                 for k, v in EA.consistency(g, {"BPG": t}).items()}


def test_duplicate_table_reports_a_one_unit_stratum_descriptively(syn, monkeypatch):
    from r5 import population as PO
    t, _ = syn
    g00 = set(t[t.cluster_cpg == "g00"].cell_id.astype(str))          # one unit: G = 1, like the real V2-3 stratum
    monkeypatch.setattr(PO, "duplicate_cell_sets", lambda: {"noV2": set(sorted(g00)[:2]), "noBITDUP": {"cell0013"},
                                                            "V2-3": g00})
    d = EA.duplicate_table(t, "BPG")
    assert len(d) == len(EA.CONTRASTS) * len(EA.DUP_VARIANTS)
    v = d[(d.variant == "V2-3") & (d.contrast == "B1 - B0")].iloc[0]
    sub = t[t.cluster_cpg == "g00"]
    assert v.G == 1 and v.cells_kept == 12 and v.relSSE == pytest.approx(CM.rel_sse(sub, "B1", "B0"), rel=1e-12)
    assert v.dR2 == pytest.approx(v.R2_cand - v.R2_ref, rel=1e-12) and np.isfinite(v.dMAE_bal)
    assert np.isnan(v.SF_POOL_p) and v["W/L/T"] == v.gate_pass == "not computable (G = 1)"
    assert "descriptive" in v.label and "SENSITIVITY" in v.label and "not computable" in v.note
    w = d[(d.variant == "without_V2-3") & (d.contrast == "B1 - B0")].iloc[0]
    assert w.G == 12 and isinstance(w.gate_pass, (bool, np.bool_)) and np.isfinite(w.SF_POOL_p)


def _per_item_inputs(syn):
    t, prov = syn
    t = t.copy()
    t["B1"] = t.y + 0.5 * (t.B0 - t.y)                     # the positive control passes on every design
    mod = _script()
    res = {d: dict(EA.score_design(t, prov, d), rows=t) for d in paths.DESIGNS}
    return mod, res


def test_per_item_tables_per_design_and_both(syn):
    mod, res = _per_item_inputs(syn)
    gates = pd.concat([res[d]["gate"] for d in paths.DESIGNS], ignore_index=True)
    leak = {"text": "leakage ok"}
    for ds in (("BPG",), ("BP",), ("BPG", "BP")):
        dec = mod.make_decision(gates[gates.design.isin(ds)])
        item = mod.item_table(res, ds, leak, "1 s", dec, "TAG")
        txt = CM.per_item_table(item, None)
        assert all(k in item for k, _ in CM.ITEM_FIELDS) and "seal TAG" in item["label"]
        assert set(item["c5"].design) == set(ds) and set(item["duplicates"].design) == set(ds)
        assert item["population"] == CM.population(res["BPG"]["rows"]) and "four reads" in item["read_budget"]
        assert "## Duplicate sensitivities" in txt and "item 1.1" in item["next_step"]
        if "BPG" in ds:
            assert "passes its own positive control" in item["changed"] and "does not fire" in item["changed"]
            assert "HIST_D7 - B1" in item["changed"] and "stays" in item["next_step"]
        else:
            assert "not evaluated in this run (BPG not scored)" in item["changed"] and "needs the BPG run" in item["next_step"]
            assert "BP sensitivity" in item["changed"]
    # the sentences follow the computed outcome, not fixed text: a failing positive control fires the kill
    bad = gates.copy()
    bad.loc[(bad.design == "BPG") & (bad.contrast == "B1 - B0"), "gate_pass"] = False
    dec = mod.make_decision(bad)
    ch, nx = EA.outcome_sentences(bad, dec)
    assert dec["M35_kill_fires"] and "fails its own positive control" in ch and "abandoned" in ch and "abandoned" in nx


def test_run_detail_never_reports_an_unevaluated_control_as_fail(syn):
    mod, res = _per_item_inputs(syn)
    gates = pd.concat([res[d]["gate"] for d in paths.DESIGNS], ignore_index=True)
    bp_only = mod.run_detail(mod.make_decision(gates[gates.design == "BP"]))
    assert bp_only.startswith("positive control not evaluated in this run (BPG not scored)") and "FAIL" not in bp_only
    assert "BP sensitivity B1 - B0 PASS" in bp_only
    assert mod.run_detail(mod.make_decision(gates[gates.design == "BPG"])) == "positive control PASS on BPG"
    bad = gates.copy()
    bad.loc[bad.design == "BPG", "gate_pass"] = False
    assert mod.run_detail(mod.make_decision(bad)).startswith("positive control FAIL on BPG")


def test_six_design_tables_and_fixed_denominators():
    base, prov = synth(seed=11)
    rows = {}
    rng = np.random.default_rng(1)
    for i, d in enumerate(EA.SIX):
        t = base.copy()
        t["B0"] = t.y + rng.normal(0, 0.3, len(t))
        t["HIST_D7"] = t.B0 + rng.normal(0, 0.05, len(t))
        if d == "A":
            t["read_set"] = np.where(t.cell_id.str.endswith("1"), "La,Ce,Gd,Lu", t.read_set)
            t["fold"] = (t.fold + 1) % 5
        rows[d] = t
    same = EA.assert_same_queries(rows)
    assert all(v["same_queries"] and v["same_y"] for v in same.values())
    out = EA.six_design(rows, prov)
    assert len(out["sixdesign"]) == 12 and set(out["sixdesign"].design) == set(EA.SIX)
    assert len(out["sixdesign_levels"]) == 24 and len(out["sixdesign_readsets"]) == 6
    rs = out["sixdesign_readsets"].set_index("design").share_cells_readset_differs_from_BPG
    assert rs["BPG"] == 0 and rs["B"] == 0 and 0 < rs["A"] < 1
    bad = dict(rows, B=rows["B"].assign(y=rows["B"].y + 1e-9))
    with pytest.raises(AssertionError):
        EA.assert_same_queries(bad)
    bad = dict(rows, BR=rows["BR"].iloc[1:])
    with pytest.raises(AssertionError):
        EA.assert_same_queries(bad)


# ------------------------------------------------------------------------------------------------------------
# construction-only checks on real data (labels read; no score computed)
# ------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def bench():
    from gen20k import data as D
    return D.load_bench()


def _gen21_cache_present(design="BP", seed=104729, fold=1) -> bool:
    return (paths.CACHE_R4 / f"{design}_{seed}_{fold}.pkl").exists()


@pytest.mark.skipif(not _gen21_cache_present(), reason="gen21 cache missing")
def test_isolation_intervention_peek_on_a_stored_fold(bench):
    rec = EA.load_record("BP", 104729, 1)
    iso = EA.isolation(bench, rec)
    assert iso["passed"] and iso["required_zero"] == list(EA.REQUIRED_ZERO["BP"])
    iv = EA.intervention(bench, rec)
    assert iv["passed"] and iv["max_abs_change"] == 0.0 and iv["peek_caught"] and iv["support_ok"]
    assert iv["n_predictions"] == 2 * len(EA.fold_rows(bench, rec))
    # a design whose guarantees are weaker than BP's must not be judged by BP's rule
    rec_a = dict(rec, design="A")
    assert EA.isolation(bench, rec_a)["required_zero"] == ["cell", "extractant"]


@pytest.mark.skipif(not _gen21_cache_present("BPG", 104729, 1), reason="gen21 cache missing")
def test_reproduction_identity_one_fold(bench):
    rp = EA.repro(bench, designs=("BPG",), fold_filter={(104729, 1)})
    d = rp["designs"]["BPG"]
    assert rp["label"] == EA.L_SMOKE and d["same_keys"] and d["same_y"] and d["same_read_set"] and d["passed"]
    assert d["rows"] > 0 and not rp["full"]


@pytest.mark.skipif(not EA.MANIFEST.exists(), reason="round-5 caches not built yet (--build-cache)")
def test_cache_manifest_complete(bench):
    man = EA.load_manifest()
    recs = [v for v in man["records"].values() if v["design"] in EA.NEW_DESIGNS]
    assert len(recs) == 100
    assert all(v["poison"]["identical"] and v["isolation"]["passed"] for v in recs)
    assert sum(v["poison"]["poisoned_design_excluded_cells"] > 0 for v in recs if v["design"] in ("BR", "BQ")) > 0
    assert all(v["poison"]["poisoned_design_excluded_cells"] == 0 for v in recs if v["design"] in ("A", "B"))
    assert EA.cache_problems(bench, man=man) == []
    rec = EA.load_record("A", 104729, 0, man)
    assert rec["design"] == "A" and rec["version"] == 2 and len(rec["inner"]) >= 1


def _script():
    spec = importlib.util.spec_from_file_location("s1_ea_check", paths.R5 / "scripts" / "s1_ea_check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_refuses_scoring_without_the_seal(monkeypatch):
    mod = _script()
    calls = []

    def refuse(*a, **k):
        calls.append(1)
        raise SystemExit("round-5 seal does not verify: scoring refused")
    monkeypatch.setattr(mod.sealcheck, "require_seal", refuse)
    monkeypatch.setattr(mod.D, "load_bench", lambda: pytest.fail("bench loaded before the seal check"))
    for argv in ([], ["--design", "BPG"], ["--design", "BP"], ["--repro"]):
        with pytest.raises(SystemExit, match="scoring refused"):
            mod.main(argv)
    assert len(calls) == 4
