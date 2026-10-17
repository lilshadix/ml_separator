"""End-to-end wiring of the models-side scripts on temporary directories, with every target replaced by synthetic
values before any statistic is computed (``g201m_helpers.synthetic_bench``) and tiny forests: g201_describe (S5:
outputs, byte-identical re-run, STOP on a changed file, staging inside the output directory), g201_anchors (S7:
row/cell/Kish checks on the real row structure; the synthetic targets must STOP the value gate), g201_smoke (S8: pass
flags, timings, projection; a failing centred learner recorded "not run" instead of crashing) and the discovery
script's smoke-record and anchor gates.  Nothing is written under generations/gen20_1/{manifests,predictions,results}."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gen201 import bootstrap
from gen201 import data as D
from gen201 import learners as L

import g201m_helpers as H


def _load(name):
    path = bootstrap.G201 / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_g201m_e2e_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def synthetic(bench, monkeypatch):
    sb = H.synthetic_bench(bench)
    monkeypatch.setattr(D, "load", lambda check=True: sb)
    return sb


def _untouched():
    """The real output locations hold nothing written by these tests."""
    return {p.name for p in bootstrap.MANIFESTS.glob("g201_*__T*.json")} | (
        {p.name for p in (bootstrap.RESULTS / "anchors").glob("*")} if (bootstrap.RESULTS / "anchors").exists() else set())


def _system_temp_staging():
    return sorted(p.name for p in Path(tempfile.gettempdir()).glob("g201_describe_*"))


def test_describe_main_writes_compares_and_stops(synthetic, tmp_path):
    mod = _load("g201_describe")
    before = _untouched()
    temp_before = _system_temp_staging()
    out = tmp_path / "describe"
    assert mod.main(["--label", "T1", "--out", str(out)]) == 0
    assert not list(out.glob(mod.STAGING_PREFIX + "*"))                # staging removed after success
    js = json.loads((out / "describe_g201.json").read_text(encoding="utf-8"))
    assert {"COH-D-C3", "COH-SF-C3", "COH-REPCAP@COH-D-C3", "COH-D-C3@NODMD"} <= set(js["cards"])
    assert js["cards"]["COH-D-C3"]["n_rows"] == 2425 and js["cards"]["COH-SF-C3"]["n_rows"] == 1802
    assert js["cards"]["COH-SF-C3"]["n_cells"] == 229 and "y_rms0" in js["cards"]["COH-SF-C3"]
    assert len(js["noise_logd"]["BP"]) == 25 and js["noise_logsf"]["status"] == "not estimable"
    assert (out / "row_manifests" / "COH-REPCAP.parquet").exists()
    rc = pd.read_parquet(out / "row_manifests" / "COH-REPCAP@COH-SF-C3.parquet")
    assert set(rc["design"]) == {"BP", "B", "BR", "BQ", "BPG", "A"}
    run = json.loads((out / "g201_describe__T1.json").read_text(encoding="utf-8"))
    assert run["status"] == "ok" and run["records"]["frozen_trees_unchanged"]
    # S6 re-run: byte-identical, nothing overwritten
    assert mod.main(["--label", "T2", "--out", str(out)]) == 0
    assert all(json.loads((out / "g201_describe__T2.json").read_text(encoding="utf-8"))["records"]
               ["rerun_byte_comparison"].values())
    assert mod.main(["--label", "T2", "--out", str(out)]) == 1                 # an existing run manifest: refused
    p = out / "row_manifests" / "COH-REPCAP.parquet"
    p.write_bytes(p.read_bytes() + b"\0")
    with pytest.raises(AssertionError, match="STOP"):
        mod.main(["--label", "T3", "--out", str(out)])
    kept = list(out.glob(mod.STAGING_PREFIX + "*"))                    # kept for diagnosis, inside the output dir
    assert len(kept) == 1 and (kept[0] / mod.OUT_JSON).exists()
    assert _system_temp_staging() == temp_before                       # nothing staged in the system temp dir
    assert _untouched() == before


def test_anchors_main_checks_rows_and_stops_on_synthetic_targets(synthetic, tmp_path):
    mod = _load("g201_anchors")
    out, man = tmp_path / "anchors", tmp_path / "man"
    rc = mod.main(["--label", "T1", "--no-run-fold", "--no-subprocess", "--out", str(out), "--manifest-dir", str(man)])
    res = json.loads((out / "anchors_g201.json").read_text(encoding="utf-8"))
    assert rc == 1 and res["status"] == "STOP"                    # synthetic targets cannot reproduce gen20's values
    assert res["kish_pass"] and len(res["kish_pairs_per_seed"]) == 5
    for name in ("FLAT", "G14", "MEAN_CURVE", "NN_TANIMOTO"):
        a = res["anchors"][name]
        assert all(v["n_rows"] == 1802 and v["n_cells"] == 229 for v in a["per_seed"].values())
    assert (man / "g201_anchors__T1.json").exists()


def test_smoke_main_and_the_discovery_smoke_gate(bench, tmp_path):
    mod = _load("g201_smoke")
    root, man = tmp_path / "pred", tmp_path / "man"
    with H.tiny_trees(4, 2):
        rc = mod.main(["--label", "smoke_t", "--root", str(root), "--manifest-dir", str(man), "--tiny"])
    assert rc == 0
    rec = json.loads((man / "g201_smoke__smoke_t.json").read_text(encoding="utf-8"))
    assert rec["status"] == "ok"
    assert rec["records"]["pass"] == {"coverage": True, "finite": True, "C": True, "C_PAIR": True,
                                      "frozen_unchanged": True}
    proj = rec["records"]["projection"]
    assert proj["hours_total"] > 0 and proj["gate_hours"] == 24.0
    assert {f["family"] for f in rec["records"]["fits"]} >= {"A", "C", "BD_COND", "BD_LEVEL", "A_PRIME", "C_PAIR"}
    assert all(f["peak_rss"] for f in rec["records"]["fits"])
    assert len(list((root / "smoke_t" / "BP" / "primary").glob("*_DONE.json"))) == 2
    assert rec["records"]["arm_failures"] == {} and rec["records"]["include"] == {"C": True, "C_PAIR": True}
    assert mod.main(["--label", "notsmoke", "--root", str(root), "--manifest-dir", str(man)]) == 1
    disc = _load("g201_discovery")
    with pytest.raises(RuntimeError, match="tiny"):                     # a 4/2-tree dry run never gates discovery
        disc.smoke_flags(man / "g201_smoke__smoke_t.json")
    flags = disc.smoke_flags(_as_full_size(rec, tmp_path / "full.json"))
    assert flags["C"] and flags["C_PAIR"] and flags["arm_failures"] == {}
    with pytest.raises(RuntimeError):
        disc.smoke_flags(tmp_path / "missing.json")
    assert disc.main(["--label", "smoke_x"]) == 1


def _as_full_size(rec: dict, path, **edit):
    """A copy of a tiny smoke record as the full-size run would write it (not --tiny, units under predictions/);
    ``edit`` replaces top-level keys.  Only the record's provenance fields are changed."""
    r = copy.deepcopy(rec)
    r["args"]["tiny"] = False
    r["records"]["run_design"]["root"] = str(bootstrap.PREDICTIONS.resolve())
    r.update(edit)
    path.write_text(json.dumps(r), encoding="utf-8")
    return path


def _smoke_record(**over):
    rec = {"name": "g201_smoke", "label": "smoke_S8", "status": "ok", "args": {"tiny": False},
           "records": {"pass": {"coverage": True, "finite": True, "C": True, "C_PAIR": True, "frozen_unchanged": True},
                       "run_design": {"root": str(bootstrap.PREDICTIONS), "units": [{"status": "done"}] * 4},
                       "projection": {"hours_total": 1.0}}}
    rec.update(over)
    return rec


@pytest.mark.parametrize("change, match", [
    ({"name": "g201_discovery"}, "not a g201_smoke"),
    ({"label": "S8"}, "smoke"),
    ({"status": "error: boom"}, "did not finish"),
    ({"args": {"tiny": True}}, "tiny"),
    ({"records": {"pass": {}, "run_design": {"root": "x", "units": [{"status": "done"}]}}}, "wrote to"),
])
def test_discovery_smoke_gate_refuses_other_records(tmp_path, change, match):
    disc = _load("g201_discovery")
    p = tmp_path / "s.json"
    p.write_text(json.dumps(_smoke_record(**change)), encoding="utf-8")
    with pytest.raises(RuntimeError, match=match):
        disc.smoke_flags(p)
    rec = _smoke_record()
    rec["records"]["run_design"]["units"] = [{"status": "done"}, {"status": "skipped (DONE)"}]
    p.write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(RuntimeError, match="every unit"):
        disc.smoke_flags(p)
    p.write_text(json.dumps(_smoke_record()), encoding="utf-8")
    assert disc.smoke_flags(p)["C"] is True


def _anchor_record(**over):
    rec = {"status": "PASS", "kish_pass": True,
           "anchors": {a: {"status": "reproduced"} for a in ("FLAT", "G14", "MEAN_CURVE", "NN_TANIMOTO")},
           "run_fold_crosscheck": {"pass": True, "max_abs_delta": 0.0, "rows_compared": 9010}}
    rec.update(over)
    return rec


def test_discovery_anchor_gate(tmp_path):
    disc = _load("g201_discovery")
    p = tmp_path / "anchors_g201.json"
    with pytest.raises(RuntimeError, match="no S7 anchor record"):
        disc.anchors_gate(p)
    p.write_text(json.dumps(_anchor_record()), encoding="utf-8")
    g = disc.anchors_gate(p)
    assert g["status"] == "PASS" and g["kish_pass"] and len(g["sha256"]) == 64
    drift = _anchor_record()
    drift["anchors"]["G14"]["status"] = "reproduced (float32 storage or platform drift)"
    p.write_text(json.dumps(drift), encoding="utf-8")
    assert disc.anchors_gate(p)["anchors"]["G14"].startswith("reproduced")
    bad_cases = [_anchor_record(status="STOP"), _anchor_record(kish_pass=False),
                 _anchor_record(run_fold_crosscheck={"pass": False}), _anchor_record(anchors={})]
    no_fold = _anchor_record()
    del no_fold["run_fold_crosscheck"]                                   # --no-run-fold never opens the gate
    stopped = _anchor_record()
    stopped["anchors"]["FLAT"]["status"] = "STOP"
    for rec in bad_cases + [no_fold, stopped]:
        p.write_text(json.dumps(rec), encoding="utf-8")
        with pytest.raises(RuntimeError, match="STOP: S7 anchor gate"):
            disc.anchors_gate(p)


def test_smoke_records_a_failing_centred_learner_as_not_run(bench, tmp_path, monkeypatch):
    """PRE_REGISTRATION.md §4.2 / O11: C failing S8 is "not run" (recorded), not a crash of the smoke run."""
    mod = _load("g201_smoke")
    root, man = tmp_path / "pred", tmp_path / "man"
    orig = L.RowForest.predict

    def nan_centred(self, X, cells):
        P = orig(self, X, cells)
        return np.full_like(P, np.nan) if self.target == "centred" else P

    monkeypatch.setattr(L.RowForest, "predict", nan_centred)
    with H.tiny_trees(4, 2):
        rc = mod.main(["--label", "smoke_c", "--root", str(root), "--manifest-dir", str(man), "--tiny"])
    assert rc == 0
    rec = json.loads((man / "g201_smoke__smoke_c.json").read_text(encoding="utf-8"))
    assert rec["status"] == "ok"
    assert rec["records"]["pass"] == {"coverage": True, "finite": True, "C": False, "C_PAIR": True,
                                      "frozen_unchanged": True}
    assert rec["records"]["include"] == {"C": False, "C_PAIR": True}
    assert rec["records"]["arm_failures"]["C"]["arm"] == "C"
    assert all(u["status"] == "done" for u in rec["records"]["run_design"]["units"])
    assert rec["records"]["run_design"]["include_c"] is False
    for v in ("primary", "nodmd"):
        for p in (root / "smoke_c" / "BP" / v).glob("*_DONE.json"):
            assert "C" not in json.loads(p.read_text(encoding="utf-8"))["arms_logd"]
    flags = _load("g201_discovery").smoke_flags(_as_full_size(rec, tmp_path / "full.json"))
    assert flags["C"] is False and flags["C_PAIR"] is True and "C" in flags["arm_failures"]
