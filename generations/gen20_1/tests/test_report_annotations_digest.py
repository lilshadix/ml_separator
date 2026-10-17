"""Scoring-side review fixes on synthetic tables (no real target value is read):

* TAB-06 rows and every threshold sentence carry the §5 context and the §7 mandatory annotations (I1 with T1,
  R2_0 / R²_FLAT / I3 with T2 and T3), also in the confirmation re-render;
* TAB-09 carries the MET-01 chemotype / publication intervals (COH-SC009: chemotype "one cluster");
* TAB-11 / MET-22 print the inner sizes and the skipped inner folds;
* TAB-08, TAB-13 and the MET-24 card carry R2_reason and n_seeds_defined;
* one prediction code digest over every complete (design, variant); selection records verified against their
  DONE markers; S11 refuses a code digest that differs from discovery's, also on --resume.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json

import numpy as np
import pytest

from gen201 import bootstrap
from gen201 import metrics as M
from gen201 import report as RP

from test_report_synthetic import build_world, scorer

DESIGNS = ("BP", "B")


def _inner_sels(world):
    """Selection records shaped like fitting.run_unit's: per selected arm a dict with inner counts and skipped
    inner folds, plus the unit-level 'inner' list (fold 0 of the first seed has inner fold 2 skipped)."""
    out = {}
    for (d, v), (logd, _) in world.items():
        recs = {}
        units = sorted((int(s), int(f)) for s, f in logd[["split_seed", "fold"]].drop_duplicates().itertuples(index=False))
        for s, f in units:
            skipped = [2] if (s, f) == units[0] else []
            arm = {"chosen": "K2", "scores": {"K1": 0.3, "K2": 0.2, "K3": 0.25, "K4": 0.26}, "n_inner_val_cells": 40,
                   "n_inner_rows": 300, "skipped_inner_folds": skipped}
            recs[(s, f)] = {**{a: dict(arm) for a in RP.SELECTED_ARMS},
                            "BL_D_INNER": {"chosen": "BD_METAL"}, "BL_SF_INNER": {"chosen": "FLAT"},
                            "n_train_cells": 100, "n_logd_rows": int(len(logd)), "n_logsf_rows": 10,
                            "dmd_removed": {"train": 3, "test": 0},
                            "inner": [{"j": j, "n_train": 70 + j, "n_train_unmasked": 75 + j, "n_val": 20 + j,
                                       "skipped": j in skipped} for j in range(4)],
                            "skipped_inner_folds": skipped, "all_inner_skipped": False}
        out[(d, v)] = recs
    return out


@pytest.fixture(scope="module")
def scored():
    world, man, _ = build_world(skill=0.8, designs=DESIGNS, n_chem=12, seed=3)
    sels = _inner_sels(world)
    sc = RP.Scorer("synthetic", loader=lambda lab, d, v: world[(d, v)],
                   status=lambda lab, d, v: {"status": "complete" if (d, v) in world else "not run"},
                   manifests=man, selections=lambda lab, d, v: sels.get((d, v), {}),
                   digests=lambda lab, d, v: {"c0de"}, designs=DESIGNS, replicates=200, log=lambda *a: None)
    return sc, sc.score()


# ---------------------------------------------------------------------------- finding 1: TAB-06 and sentences
def test_tab06_rows_carry_context_and_mandatory_annotations(scored):
    _, st = scored
    md = RP.render(RP._clean(st))
    rows = [ln for ln in md.splitlines() if ln.startswith(("| T1 |", "| T1w |", "| T2 |", "| T3 |"))]
    assert len(rows) == 4 * 2
    for ln in rows:
        for needle in ("y_sd0", "y_var0", "MET-01", "MET-02", "MET-05", "simple baseline", "MAE", "RMSE",
                       "98.75 % lower bounds: chemotype", "publication", "LOCO min", "COH-SC009 point",
                       "COH-NOSC009 point", "COH-NODMD point", "τ + 1.96·se_boot", "R2_reason", "n_seeds_defined"):
            assert needle in ln, (needle, ln[:80])
    for ln in rows:
        tid = ln.split("|")[1].strip()
        if tid in ("T1", "T1w"):
            assert "simple baseline BD_METAL" in ln and "I1: R²_pooled" in ln and "s_b" in ln and "s_w" in ln
        else:
            assert "simple baseline FLAT" in ln and "R2_0" in ln and "R²_FLAT" in ln and "I3:" in ln
    # the printed context numbers are the scorer's own (T1 at τ = 0.4)
    ctx = st["thresholds"]["T1"]["annotations"]["context"]
    t1 = [ln for ln in rows if ln.startswith("| T1 |")][0]
    assert f"y_var0 {RP._f(ctx['y_var0'])}" in t1 and f"RMSE {RP._f(ctx['simple_RMSE'])}" in t1


def test_threshold_sentences_carry_context(scored):
    _, st = scored
    assert all("context" in s and "I1: R²_pooled" in s["context"] and "R2_0" in s["context"]
               for s in st["sentences"]["thresholds"])
    md = RP.render(RP._clean(st))
    for s in st["sentences"]["thresholds"]:
        line = [ln for ln in md.splitlines() if ln.startswith(f"- **{s['id']}.** {s['text']}")]
        assert line, s["id"]
        # the T1 decomposition is printed on the same line as the sentence (§7 "in the same sentence")
        assert "Context (§5, §7" in line[0] and "I1: R²_pooled" in line[0] and "y_var0" in line[0]


def test_confirmation_rerender_prints_confirmation_classes_with_context(scored):
    _, st = scored
    clean = RP._clean(st)
    md = RP.render(clean, confirmation=clean)
    assert "Threshold classes on the confirmation seeds" in md
    conf_part = md.split("Threshold classes on the confirmation seeds")[1]
    rows = [ln for ln in conf_part.splitlines() if ln.startswith(("| T1 |", "| T2 |"))]
    assert rows and all("y_var0" in ln and "COH-NODMD point" in ln for ln in rows)
    assert "identical" in md.split("## Registered verdicts")[0]


# ---------------------------------------------------------------------------- finding 2: TAB-09 intervals
def test_tab09_carries_met01_intervals_and_one_cluster_status(scored):
    _, st = scored
    tab09 = [r for r in st["tables"]["tab09_secondary"] if r.get("status") == "run"]
    assert tab09
    for r in tab09:
        for k in ("MET01_chem_pct95_low", "MET01_chem_pct9875_high", "MET01_pub_pct95_low", "MET01_pub_pct9875_high",
                  "MET01_chem_status", "MET01_pub_status", "MET01_chem_preliminary", "MET01_pub_preliminary"):
            assert k in r, (r["population"], k)
    sc009 = [r for r in tab09 if r["population"] in ("COH-SC009-D", "COH-SC009-SF")]
    assert sc009
    for r in sc009:
        assert r["MET01_chem_status"] == "one cluster" and np.isnan(r["MET01_chem_pct95_low"])
        if r["MET01_pub_n_clusters"] >= 2:
            assert np.isfinite(r["MET01_pub_pct95_low"]) and r["MET01_pub_status"] != "one cluster"


# ---------------------------------------------------------------------------- finding 3: TAB-11 / MET-22
def test_tab11_inner_sizes_and_skipped_folds(scored):
    _, st = scored
    rows = st["tables"]["tab11_folds"]
    assert rows
    for r in rows:
        for j in range(4):
            assert r[f"inner{j}_n_train"] == 70 + j and r[f"inner{j}_n_val"] == 20 + j
            assert r[f"inner{j}_n_train_unmasked"] == 75 + j and isinstance(r[f"inner{j}_skipped"], bool)
        assert r["A_n_inner_val_cells"] == 40 and r["A_n_inner_rows"] == 300
        assert r["rec_dmd_removed_train"] == 3
    skipped = [r for r in rows if r["skipped_inner_folds"]]
    assert skipped and all(r["skipped_inner_folds"] == "2" and r["inner2_skipped"] for r in skipped)
    summary = [r for r in st["tables"]["tab11_MET22"] if r.get("config") == "skipped inner folds"]
    assert len(summary) == 1
    n_bp = sum(1 for r in rows if r["design"] == "BP" and r["variant"] == "primary" and r["skipped_inner_folds"])
    assert summary[0]["BP_units_with_skipped_inner_fold"] == n_bp >= 1
    assert summary[0]["skipped_seed_fold_j"].count("(") == n_bp


# ---------------------------------------------------------------------------- finding 4: R2_reason everywhere
def test_r2_reason_and_seed_counts_in_tab08_tab13_met24(scored):
    _, st = scored
    tabs = st["tables"]
    arm_rows = [r for r in tabs["tab08_designs"] if "arm" in r and "MET06" in r]
    assert arm_rows
    for r in arm_rows:
        assert {"MET06_reason", "MET06_n_seeds_defined", "simple_R2_reason", "simple_R2_n_seeds_defined"} <= set(r)
        assert ("MET08_reason" in r and "MET08_n_seeds_defined" in r) if r["population"] == "COH-D-C3" else \
            ("MET10_reason" in r and "MET10_n_seeds_defined" in r)
    for r in tabs["tab13_not_selectable"]:
        assert {"MET06_reason", "MET06_n_seeds_defined"} <= set(r)
        st2 = "MET08" if r["population"] == "COH-D-C3" else "MET10"
        assert {st2, f"{st2}_reason", f"{st2}_n_seeds_defined"} <= set(r)
    prnd = tabs["tab10_prnd_MET24"]
    assert prnd and all("R2_n_seeds_defined" in r for r in prnd if r.get("n_rows"))


def test_met24_n_seeds_defined_on_synthetic_rows():
    world, _, _ = build_world(skill=0.5, designs=("BP",), n_chem=10, seed=4)
    logsf = world[("BP", "primary")][1]
    c = M.met24(logsf, "B")
    assert c["n_rows"] > 0
    assert c["R2_n_seeds_defined"] == logsf["split_seed"].nunique() and "R2_reason" in c


# ---------------------------------------------------------------------------- finding 5: code digests
def test_scorer_refuses_mixed_code_digests():
    world, man, sels = build_world(skill=0.5, designs=DESIGNS, n_chem=10, seed=6)
    kw = dict(loader=lambda lab, d, v: world[(d, v)],
              status=lambda lab, d, v: {"status": "complete" if (d, v) in world else "not run"},
              manifests=man, selections=lambda lab, d, v: sels.get((d, v), {}), designs=DESIGNS, replicates=50,
              log=lambda *a: None)
    with pytest.raises(M.CoverageError, match="code versions"):
        RP.Scorer("synthetic", digests=lambda lab, d, v: {"old" if d == "B" else "new"}, **kw)
    with pytest.raises(M.CoverageError):
        RP.Scorer("synthetic", digests=lambda lab, d, v: {"missing"}, **kw)
    sc = RP.Scorer("synthetic", digests=lambda lab, d, v: {"same"}, **kw)
    assert sc.code_digest == "same" and set(sc.code_digest_by_design) == {f"{d}/{v}" for d in DESIGNS
                                                                         for v in RP.VARIANTS}
    st = sc.score(full=False)
    assert st["code_digest"] == "same"
    assert "`same`" in RP.render(RP._clean(st))
    # an injected loader without ``digests`` skips the check (synthetic tables have no DONE markers)
    assert RP.Scorer("synthetic", **kw).code_digest is None


def _write_unit_markers(d, seed, fold, code, *, sel_ok=True):
    d.mkdir(parents=True, exist_ok=True)
    sel = d / f"s{seed}_f{fold}_selection.json"
    sel.write_text(json.dumps({"split_seed": seed, "fold": fold}), encoding="utf-8")
    sha = hashlib.sha256(sel.read_bytes()).hexdigest()
    (d / f"s{seed}_f{fold}_DONE.json").write_text(json.dumps({"split_seed": seed, "fold": fold, "code_digest": code,
                                                              "selection_sha256": sha if sel_ok else "0" * 64}),
                                                  encoding="utf-8")


def test_unit_digests_and_selection_verification(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, "PREDICTIONS", tmp_path)
    d = tmp_path / "lab" / "BP" / "primary"
    _write_unit_markers(d, 1, 0, "aaa")
    _write_unit_markers(d, 1, 1, {"sha256": "aaa", "files": {}})
    assert RP.unit_digests("lab", "BP", "primary") == {"aaa"}
    assert RP.unit_digests("lab", "B", "primary") == set()
    # a selection without its DONE marker belongs to an interrupted unit: not read
    (d / "s2_f0_selection.json").write_text("{}", encoding="utf-8")
    assert set(RP.load_selections("lab", "BP", "primary")) == {(1, 0), (1, 1)}
    _write_unit_markers(d, 3, 0, "bbb", sel_ok=False)
    assert RP.unit_digests("lab", "BP", "primary") == {"aaa", "bbb"}
    with pytest.raises(M.CoverageError, match="selection_sha256"):
        RP.load_selections("lab", "BP", "primary")
    # the default scorer (no injected loader) reads the DONE markers and refuses the two digests
    with pytest.raises(M.CoverageError, match="code versions"):
        RP.Scorer("lab", manifests={}, designs=("BP",), variants=("primary",),
                  status=lambda lab, dd, v: {"status": "complete"})


def test_build_refuses_a_confirmation_with_another_code_digest(tmp_path):
    world, man, sels = build_world(skill=0.5, designs=("BP",), n_chem=10, seed=7)
    kw = dict(loader=lambda lab, d, v: world[(d, v)],
              status=lambda lab, d, v: {"status": "complete" if (d, v) in world else "not run"},
              manifests=man, selections=lambda lab, d, v: sels.get((d, v), {}), designs=("BP",), replicates=50,
              digests=lambda lab, d, v: {"disc"})
    with pytest.raises(RP.ReportRefused, match="identical procedure"):
        RP.build("discovery", out_dir=tmp_path / "res", report_md=tmp_path / "R.md",
                 confirmation={"code_digest": "other"}, scorer_kwargs=kw, log=lambda *a: None)
    assert not (tmp_path / "R.md").exists() and not (tmp_path / "res").exists()


@pytest.fixture(scope="module")
def C():
    spec = importlib.util.spec_from_file_location("g201_confirm_digest", bootstrap.G201 / "scripts" / "g201_confirm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_confirm_resume_compares_the_code_digest(C, tmp_path):
    real_before = C.RUN_ONCE.read_bytes() if C.RUN_ONCE.exists() else None
    p = tmp_path / "confirmation" / "RUN_ONCE.json"
    rec = {"seeds": [1, 2], "commitment_sha256": "x", "fold_digest_BP": "y", "code_digest": "c1"}
    C.run_once_open(p, record=rec)
    with pytest.raises(C.ConfirmationRefused, match="code_digest"):
        C.run_once_open(p, resume=True, record={**rec, "code_digest": "c2"})
    assert len(C.run_once_open(p, resume=True, record=rec)["resumes"]) == 1
    # a record written with the digest dict (older layout) resumes under the same sha256 only
    q = tmp_path / "confirmation2" / "RUN_ONCE.json"
    C.run_once_open(q, record={**rec, "code_digest": {"sha256": "c1", "files": {}}})
    assert len(C.run_once_open(q, resume=True, record=rec)["resumes"]) == 1
    with pytest.raises(C.ConfirmationRefused):
        C.run_once_open(q, resume=True, record={**rec, "code_digest": "c2"})
    # the real RUN_ONCE.json (present after S11) must be untouched by the test
    assert "code_digest" in C.RESUME_KEYS
    assert (C.RUN_ONCE.read_bytes() if C.RUN_ONCE.exists() else None) == real_before


def test_confirm_requires_discovery_code_digest(C, tmp_path):
    assert C.check_code_digest("abc", "abc") == "abc"
    with pytest.raises(C.ConfirmationRefused, match="identical procedure"):
        C.check_code_digest("abc", "abd")
    empty = tmp_path / "empty"
    assert C.discovery_code_digest({"code_digest": "abc"}, root=empty) == "abc"
    with pytest.raises(C.ConfirmationRefused, match="cannot be established"):
        C.discovery_code_digest({}, root=empty)
    root = tmp_path / "pred"
    _write_unit_markers(root / "discovery" / "BP" / "primary", 1, 0, "abc")
    assert C.discovery_code_digest({}, root=root) == "abc"
    assert C.discovery_code_digest({"code_digest": "abc"}, root=root) == "abc"
    with pytest.raises(C.ConfirmationRefused, match="differs from its DONE markers"):
        C.discovery_code_digest({"code_digest": "zzz"}, root=root)
    _write_unit_markers(root / "discovery" / "BP" / "nodmd", 1, 0, "def")
    with pytest.raises(C.ConfirmationRefused, match="several"):
        C.discovery_code_digest({"code_digest": "abc"}, root=root)
