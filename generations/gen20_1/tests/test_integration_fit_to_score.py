"""Integration of the two wave-2 halves (models -> scoring), CONTRACT.md "Wave-2 interfaces".

* One BP unit (giant fold, both variants, tiny forests; ``g201m_helpers.small_unit``) is written by
  ``fitting.write_unit`` and re-read by ``fitting.load_tables`` / ``load_selections``.  Its ``y`` column is dropped
  the moment the tables are loaded and replaced by seeded synthetic values (seed-invariant, log SF = difference of
  the synthetic log D endpoints); the unit is replicated over the five split seeds (predictions perturbed by seeded
  noise) and copied as design B.  ``report.Scorer`` then runs every registered claim, NC1, the thresholds, the
  83-test exploratory family and every table on it: every arm the scoring side names must exist in the tables the
  models side writes, the selection records must yield the BL_D_INNER / BL_SF_INNER references, and the row-manifest
  joins must cover every key.  No statistic of the real y is computed.
* report's default readers (status from DONE markers, selection records, ``fitting.load_tables``) find exactly the
  files ``fitting.write_unit`` writes.
* COH-REPCAP manifests in g201_describe's naming (``COH-REPCAP@<parent>``) map to COH-REPCAP-D / COH-REPCAP-SF.
* TAB-02 copies the S5 noise records from ``manifests/describe_g201.json`` (the file g201_describe writes).
* g201_confirm reads the S8 smoke flags with the discovery script's rule.
"""
from __future__ import annotations

import importlib.util
import json

import numpy as np
import pandas as pd
import pytest

from gen201 import bootstrap
from gen201 import fitting as FI
from gen201 import folds as FO
from gen201 import metrics as M
from gen201 import report as RP
from gen201 import rules as R

import g201m_helpers as H

LABEL = "integ"


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_integ", bootstrap.G201 / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _synthetic_y(tables: dict) -> dict:
    """Seeded synthetic log D per (ci, im) over every key of every variant (seed-invariant)."""
    rng = np.random.default_rng(20261016)
    keys = sorted({(int(c), int(m)) for d, _ in tables.values() for c, m in zip(d["ci"], d["im"])})
    cells = sorted({c for c, _ in keys})
    level = dict(zip(cells, rng.normal(0.0, 1.0, len(cells))))
    curve = np.cumsum(rng.normal(0.0, 0.3, 14))
    return {k: level[k[0]] + curve[k[1]] + rng.normal(0.0, 0.15) for k in keys}


@pytest.fixture(scope="module")
def world(bench, tmp_path_factory):
    root = tmp_path_factory.mktemp("g201_integ")
    f = H.giant_fold(bench)
    code = FI.code_digest()
    raw, sel = {}, {}
    for v in FI.VARIANTS:
        res = H.small_unit(bench, v)
        FI.write_unit(res, FI.unit_paths(root, "smoke_integ", "BP", v, f.seed, f.fold), code, label="smoke_integ")
        d, s = FI.load_tables("smoke_integ", "BP", v, root=root, expect_units=1)
        raw[v] = (d.drop(columns=["y"]), s.drop(columns=["y"]))          # the real y never reaches the scorer
        sel[v] = FI.load_selections("smoke_integ", "BP", v, root=root)
        assert len(sel[v]) == 1
    ys = _synthetic_y(raw)
    out, sels = {}, {}
    for design in ("BP", "B"):
        for v in FI.VARIANTS:
            d0, s0 = raw[v]
            d0 = d0.assign(y=[ys[(c, m)] for c, m in zip(d0["ci"], d0["im"])])
            s0 = s0.assign(y=[ys[(c, a)] - ys[(c, b)] for c, a, b in zip(s0["ci"], s0["ia"], s0["ib"])])
            parts_d, parts_s, recs = [], [], {}
            for k, seed in enumerate(FO.SEEDS):
                d, s = d0.copy(), s0.copy()
                d["split_seed"] = s["split_seed"] = seed
                d["design"] = s["design"] = design
                if design != "BP":
                    d = d.drop(columns=["A_PRIME"], errors="ignore")
                if k or design != "BP":
                    pr = np.random.default_rng(100 + k + (50 if design == "B" else 0))
                    for t in (d, s):
                        for c in M.arm_columns(t):
                            if c != "FLAT":
                                t[c] = t[c].to_numpy(dtype=np.float64) + pr.normal(0.0, 0.05, len(t))
                parts_d.append(d)
                parts_s.append(s)
                recs[(seed, int(f.fold))] = {**sel[v][0], "split_seed": seed, "design": design}
            out[(design, v)] = (pd.concat(parts_d, ignore_index=True), pd.concat(parts_s, ignore_index=True))
            sels[(design, v)] = recs
    # row manifests: the real (label-free) keys restricted to the scored cells, plus a synthetic COH-REPCAP
    cells = {v: set(raw[v][0]["ci"].astype(int)) for v in FI.VARIANTS}
    man = {}
    for name, keys in RP.load_row_manifests().items():
        k = keys[keys["ci"].isin(cells["nodmd" if name.endswith("@NODMD") else "primary"])].reset_index(drop=True)
        if len(k):
            man[name] = k
    kd = man["COH-D-C3"]
    drop = kd.iloc[np.sort(np.random.default_rng(7).choice(len(kd), size=len(kd) // 30, replace=False))]
    bad = set(zip(drop["ci"], drop["im"]))
    man["COH-REPCAP-D"] = kd[[(c, m) not in bad for c, m in zip(kd["ci"], kd["im"])]].reset_index(drop=True)
    ks = man["COH-SF-C3"]
    man["COH-REPCAP-SF"] = ks[[(c, a) not in bad and (c, b) not in bad
                               for c, a, b in zip(ks["ci"], ks["ia"], ks["ib"])]].reset_index(drop=True)
    sc = RP.Scorer(LABEL, loader=lambda lab, d, v: out[(d, v)],
                   status=lambda lab, d, v: {"status": "complete" if (d, v) in out else "not run"},
                   manifests=man, selections=lambda lab, d, v: sels.get((d, v), {}), replicates=200,
                   log=lambda *a: None)
    return {"scorer": sc, "state": sc.score(), "tables": out, "manifests": man, "root": root, "fold": f}


def test_report_default_io_reads_the_fitting_layout(world, monkeypatch):
    """report.default_status / load_selections / default_loader find exactly what fitting.write_unit wrote
    (predictions/<label>/<design>/<variant>/s<seed>_f<fold>_*); only the layout is checked, never y."""
    f = world["fold"]
    monkeypatch.setattr(bootstrap, "PREDICTIONS", world["root"])
    for v in FI.VARIANTS:
        assert RP.default_status("smoke_integ", "BP", v, n_units=1)["status"] == "complete"
        assert RP.default_status("smoke_integ", "BP", v)["status"] == "incomplete"          # 1 of 25 units
        sels = RP.load_selections("smoke_integ", "BP", v)
        assert list(sels) == [(int(f.seed), int(f.fold))] and sels[(int(f.seed), int(f.fold))]["variant"] == v
        logd, logsf = RP.default_loader("smoke_integ", "BP", v)
        assert list(logd.columns[:len(FI.LOGD_KEYS)]) == list(FI.LOGD_KEYS)
        assert list(logsf.columns[:len(FI.LOGSF_KEYS)]) == list(FI.LOGSF_KEYS)
        assert set(M.arm_columns(logd)) == set(FI.logd_arm_columns("BP", v))
        assert set(M.arm_columns(logsf)) == set(FI.logsf_arm_columns("BP", v))
    assert RP.default_status("smoke_integ", "B", "primary")["status"] == "not run"


def test_every_scored_arm_exists_in_the_fitted_tables(world):
    st = world["state"]
    for h in ("H1", "H2", "H3", "H4"):
        assert st["registered"][h]["evidence"], h                      # candidate and comparators found
        assert st["registered_nodmd"][h]["evidence"], h
    for ev in st["registered"]["H1"]["evidence"]:
        assert ev["design_points"]["B"] is not None and ev["design_points"]["A"] is None
    assert st["nc1"]["status"] == "run"                                # A_PRIME (BP primary)
    ex = st["exploratory"]
    assert len(ex) == R.M_EXPLORATORY
    assert [r["id"] for r in ex if r["status"] != "run"] == []         # C, C_PAIR, BL_*_INNER, REPCAP, NODMD, ...
    tabs = st["tables"]
    assert any(r["arm"] == "C_PAIR" and r["status"] == "run" for r in tabs["tab04_logsf"])
    assert {r["arm"] for r in tabs["tab13_not_selectable"]} == set(RP.NOT_SELECTABLE["logd"]) | set(RP.NOT_SELECTABLE["logsf"])
    assert any(r.get("population") == "COH-REPCAP-SF" and r.get("status") == "run" for r in tabs["tab09_secondary"])


def test_selection_records_feed_the_scorer(world):
    tabs = world["state"]["tables"]
    rows = tabs["tab11_folds"]
    assert len(rows) == 2 * 2 * len(FO.SEEDS)
    for r in rows:
        for arm in RP.SELECTED_ARMS:
            assert r[f"{arm}_chosen"] in ("K1", "K2", "K3", "K4") and json.loads(r[f"{arm}_inner_scores"])
        assert r["BL_D_INNER_chosen"] in R.H1_COMPARATORS and r["BL_SF_INNER_chosen"] in ("FLAT", "PAIRMEAN")
        assert r["rec_n_logd_rows"] > 0 and "fallback_BD_NN_rows_any" in r
    assert {r["fallback"] for r in tabs["tab03_fallbacks"]} == {"BD_METAL_rows", "BD_NN_rows_any", "BD_NN_rows_all",
                                                               "PAIRMEAN_rows"}
    logd, logsf = world["scorer"].tables("BP", "primary")
    assert "BL_D_INNER" in logd.columns and "BL_SF_INNER" in logsf.columns


def test_results_and_markdown_render(world, tmp_path):
    paths = RP.write_results(world["state"], tmp_path / "res")
    assert {"tab05_verdicts.csv", "tab06_thresholds.csv", "tab12_exploratory.csv", "report_state.json"} <= {p.name for p in paths}
    md = RP.render(json.loads((tmp_path / "res" / "report_state.json").read_text(encoding="utf-8")))
    assert "## Registered verdicts" in md and "## Threshold attainment" in md and LABEL in md


def test_smoke_labels_refused_before_scoring():
    with pytest.raises(RP.ReportRefused):
        RP.build("smoke_integ")
    with pytest.raises(RP.ReportRefused):
        RP.Scorer("smoke_integ", manifests={}, status=lambda *a: {"status": "not run"})


def _keys_d(cells, metals=(2, 3, 4)):
    return pd.DataFrame([{"design": "BP", "split_seed": 1, "fold": 0, "ci": c, "cell_id": f"c{c}", "extractant": "e",
                          "chemotype": "g", "publication_id": "p", "n_metals": 3, "metal": "x", "im": m}
                         for c in cells for m in metals])


def _keys_sf(cells):
    return pd.DataFrame([{"design": "BP", "split_seed": 1, "fold": 0, "ci": c, "cell_id": f"c{c}", "extractant": "e",
                          "chemotype": "g", "publication_id": "p", "n_metals": 3, "a": "x", "b": "y", "ia": a,
                          "ib": a + 1, "dZ": 1, "pair": "x/y", "strict": True, "gen13adj": True}
                         for c in cells for a in (2, 3)])


def test_repcap_manifests_in_describe_naming(tmp_path):
    d = tmp_path / "rm"
    d.mkdir()
    _keys_d(range(6)).to_parquet(d / "COH-D-C3.parquet", index=False)
    _keys_sf(range(6)).to_parquet(d / "COH-SF-C3.parquet", index=False)
    _keys_d(range(6)).to_parquet(d / "COH-SC009-D.parquet", index=False)
    pd.DataFrame({"ci": [0], "cell_id": ["c0"], "im": [3], "metal": ["x"], "nrep": [2]}).to_parquet(
        d / "COH-REPCAP.parquet", index=False)                          # the drop list: not a population
    _keys_d(range(1, 6)).to_parquet(d / "COH-REPCAP@COH-D-C3.parquet", index=False)
    _keys_sf(range(1, 6)).to_parquet(d / "COH-REPCAP@COH-SF-C3.parquet", index=False)
    _keys_d(range(3, 6)).to_parquet(d / "COH-REPCAP@COH-SC009-D.parquet", index=False)        # decoys: other parents
    _keys_sf(range(4, 6)).to_parquet(d / "COH-REPCAP@COH-SF-NDSM.parquet", index=False)
    _keys_d(range(2, 6)).to_parquet(d / "COH-REPCAP@COH-D-C3@NODMD.parquet", index=False)
    man = RP.load_row_manifests(d)
    assert set(man) == {"COH-D-C3", "COH-SF-C3", "COH-SC009-D", "COH-REPCAP-D", "COH-REPCAP-SF"}
    assert sorted(man["COH-REPCAP-D"]["ci"].unique()) == [1, 2, 3, 4, 5] and len(man["COH-REPCAP-D"]) == 15
    assert sorted(man["COH-REPCAP-SF"]["ci"].unique()) == [1, 2, 3, 4, 5] and len(man["COH-REPCAP-SF"]) == 10
    _keys_sf(range(2)).to_parquet(d / "COH-REPCAP@COH-D-C3.parquet", index=False)
    with pytest.raises(M.CoverageError):
        RP.load_row_manifests(d)


def test_tab02_reads_the_s5_describe_json(tmp_path, monkeypatch):
    summ = {k: {"n": 25, "median": 0.01, "min": 0.0, "max": 0.02}
            for k in ("sigma2_pool", "sigma2_pool_cap", "median_s_e_cap", "n_rep_entries", "n_rep_entries_cap",
                      "sqrt2_sigma_bound_logsf")}
    rec = {"noise_logd_summary_BP": summ,
           "noise_logd": {"BP": [{"split_seed": 1, "fold": f, "sigma2_pool": 0.01} for f in range(5)]},
           "noise_logsf": {"status": "not estimable", "reason": "r", "bound": "b"},
           "repcap": {"n_entries_dropped": 3, "n_cells_touched": 2}}
    (tmp_path / RP.S5_DESCRIBE_JSON).write_text(json.dumps(rec), encoding="utf-8")
    monkeypatch.setattr(bootstrap, "MANIFESTS", tmp_path)
    sc = RP.Scorer(LABEL, manifests={}, status=lambda *a: {"status": "not run"}, loader=lambda *a: None,
                   selections=lambda *a: {}, log=lambda *a: None)
    rows = sc.target_rows()
    stats = {r["statistic"]: r for r in rows if "statistic" in r}
    assert set(stats) == set(summ) and stats["sigma2_pool"]["median"] == 0.01
    assert sum(1 for r in rows if r["population"].startswith("MET-19 log D noise per BP")) == 5
    assert any(r.get("status") == "not estimable" for r in rows)
    assert any(r.get("n_entries_dropped") == 3 for r in rows)


def test_confirmation_reads_the_smoke_flags(tmp_path):
    C = _script("g201_confirm")
    p = tmp_path / "g201_smoke__smoke_S8.json"
    p.write_text(json.dumps({"name": "g201_smoke", "status": "ok", "label": "smoke_S8", "args": {"tiny": False},
                             "records": {"pass": {"coverage": True, "finite": True, "C": True, "C_PAIR": False,
                                                  "frozen_unchanged": True},
                                         "run_design": {"root": str(bootstrap.PREDICTIONS),
                                                        "units": [{"status": "done"}] * 4}}}), encoding="utf-8")
    flags = C.smoke_flags(p)
    assert flags["C"] is True and flags["C_PAIR"] is False
    with pytest.raises(RuntimeError):
        C.smoke_flags(tmp_path / "missing.json")
