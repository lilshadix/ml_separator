"""Cross-side gates added at the final integration (no model is fitted, no target value is read):

* ``report.default_status`` counts only DONE markers whose JSON parses (an unreadable marker makes the design
  "incomplete"); ``report.unit_digests`` reports an unreadable marker as "unreadable" and the scorer refuses it;
  ``report.load_selections`` skips a unit whose marker does not parse;
* S11 applies the S7 anchor gate by the discovery script's own rule (``g201_confirm.anchors_gate``) before its run
  manifest is opened, and records the code digest of the imported code (``fitting.CODE_AT_IMPORT``).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json

import pytest

from gen201 import bootstrap
from gen201 import metrics as M
from gen201 import report as RP


def _unit(d, seed, fold, code="aaa", *, torn=False):
    d.mkdir(parents=True, exist_ok=True)
    sel = d / f"s{seed}_f{fold}_selection.json"
    sel.write_text(json.dumps({"split_seed": seed, "fold": fold}), encoding="utf-8")
    done = d / f"s{seed}_f{fold}_DONE.json"
    if torn:
        done.write_text('{"split_seed": 1, "code_dig', encoding="utf-8")
    else:
        done.write_text(json.dumps({"split_seed": seed, "fold": fold, "code_digest": code,
                                    "selection_sha256": hashlib.sha256(sel.read_bytes()).hexdigest()}),
                        encoding="utf-8")


def test_default_status_counts_only_parseable_markers(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, "PREDICTIONS", tmp_path)
    d = tmp_path / "lab" / "BP" / "primary"
    _unit(d, 1, 0)
    _unit(d, 1, 1)
    (d / "s1_f2_DONE.json.tmp").write_text("{}", encoding="utf-8")         # an interrupted write: never counted
    st = RP.default_status("lab", "BP", "primary", n_units=2)
    assert st == {"status": "complete", "n_done": 2, "n_units": 2, "n_unreadable": 0}
    assert RP.default_status("lab", "BP", "primary", n_units=2, root=tmp_path)["status"] == "complete"
    assert RP.default_status("lab", "BP", "primary")["status"] == "incomplete"            # 2 of 25
    assert RP.default_status("lab", "B", "primary")["status"] == "not run"
    _unit(d, 1, 2, torn=True)
    st = RP.default_status("lab", "BP", "primary", n_units=2)
    assert st["status"] == "incomplete" and st["n_done"] == 2 and st["n_unreadable"] == 1
    (d / "s1_f2_DONE.json").write_text("[1, 2]", encoding="utf-8")         # parses, but is not a marker record
    assert RP.default_status("lab", "BP", "primary", n_units=2)["status"] == "incomplete"


def test_unreadable_marker_is_refused_by_digests_and_skipped_by_selections(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, "PREDICTIONS", tmp_path)
    d = tmp_path / "lab" / "BP" / "primary"
    _unit(d, 1, 0, "aaa")
    _unit(d, 1, 1, torn=True)
    assert RP.unit_digests("lab", "BP", "primary") == {"aaa", "unreadable"}
    assert set(RP.load_selections("lab", "BP", "primary")) == {(1, 0)}
    with pytest.raises(M.CoverageError, match="code versions"):
        RP.Scorer("lab", manifests={}, designs=("BP",), variants=("primary",),
                  status=lambda lab, dd, v: {"status": "complete"})
    # only unreadable markers: still refused (one "digest" that names no code)
    d2 = tmp_path / "lab2" / "BP" / "primary"
    _unit(d2, 1, 0, torn=True)
    with pytest.raises(M.CoverageError, match="code versions"):
        RP.Scorer("lab2", manifests={}, designs=("BP",), variants=("primary",),
                  status=lambda lab, dd, v: {"status": "complete"})
    # and the default status never calls such a design complete
    assert RP.default_status("lab2", "BP", "primary", n_units=1)["status"] == "incomplete"


@pytest.fixture(scope="module")
def C():
    spec = importlib.util.spec_from_file_location("g201_confirm_gates", bootstrap.G201 / "scripts" / "g201_confirm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _anchor_record(**over):
    rec = {"status": "PASS", "kish_pass": True,
           "anchors": {a: {"status": "reproduced"} for a in ("FLAT", "G14", "MEAN_CURVE", "NN_TANIMOTO")},
           "run_fold_crosscheck": {"pass": True, "max_abs_delta": 0.0, "rows_compared": 9010}}
    rec.update(over)
    return rec


def test_confirm_applies_the_discovery_anchor_gate(C, tmp_path):
    mod = C.discovery_module()
    assert mod is C.discovery_module()                                    # loaded once
    assert C.anchors_gate.__doc__ and "discovery script's rule" in C.anchors_gate.__doc__
    p = tmp_path / "anchors_g201.json"
    with pytest.raises(RuntimeError, match="no S7 anchor record"):
        C.anchors_gate(p)
    p.write_text(json.dumps(_anchor_record()), encoding="utf-8")
    assert C.anchors_gate(p) == mod.anchors_gate(p)
    no_fold = _anchor_record()
    del no_fold["run_fold_crosscheck"]
    for rec in (_anchor_record(status="STOP"), _anchor_record(kish_pass=False), no_fold):
        p.write_text(json.dumps(rec), encoding="utf-8")
        with pytest.raises(RuntimeError, match="STOP: S7 anchor gate"):
            C.anchors_gate(p)


class _FakeRun:
    """Stands in for gen201.run.Run: records in memory, writes no manifest."""
    last = None

    def __init__(self, *a, **k):
        self.records = {}
        _FakeRun.last = self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def record(self, k, v):
        self.records[k] = v

    def log(self, *a):
        pass


class _StopBeforeLoad(Exception):
    pass


def _main_world(C, tmp_path, monkeypatch, *, disc_digest):
    state = tmp_path / "report_state.json"
    state.write_text(json.dumps({"code_digest": disc_digest}), encoding="utf-8")
    monkeypatch.setattr(C, "DISCOVERY_STATE", state)
    monkeypatch.setattr(C, "RUN_ONCE", tmp_path / "confirmation" / "RUN_ONCE.json")
    monkeypatch.setattr(C, "smoke_flags", lambda path=None: {"C": True, "C_PAIR": True, "label": "smoke_S8"})
    monkeypatch.setattr(C, "Run", _FakeRun)
    monkeypatch.setattr(C, "discovery_code_digest", lambda st, root=None: st["code_digest"])
    monkeypatch.setattr(C.guard, "snapshot", lambda: {})
    _FakeRun.last = None


def test_confirm_main_stops_at_the_anchor_gate_before_its_run(C, tmp_path, monkeypatch):
    _main_world(C, tmp_path, monkeypatch, disc_digest="x")
    monkeypatch.setattr(C.discovery_module(), "DEFAULT_ANCHORS", tmp_path / "missing_anchors.json")
    with pytest.raises(RuntimeError, match="no S7 anchor record"):
        C.main([])
    assert _FakeRun.last is None and not (tmp_path / "confirmation").exists()


def test_confirm_main_records_the_imported_code_digest(C, tmp_path, monkeypatch):
    from gen201 import data as D
    from gen201 import fitting as FI
    p = tmp_path / "anchors_g201.json"
    p.write_text(json.dumps(_anchor_record()), encoding="utf-8")
    monkeypatch.setattr(C.discovery_module(), "DEFAULT_ANCHORS", p)
    # another discovery digest: refused before the bench is loaded, the anchor gate already recorded
    _main_world(C, tmp_path, monkeypatch, disc_digest="0" * 64)
    with pytest.raises(C.ConfirmationRefused, match="identical procedure"):
        C.main([])
    assert _FakeRun.last.records["anchors_gate"]["status"] == "PASS" and "code_digest" not in _FakeRun.last.records
    # the discovery digest of the imported code: recorded from CODE_AT_IMPORT, then the bench would load

    def stop(*a, **k):
        raise _StopBeforeLoad()
    monkeypatch.setattr(D, "load", stop)
    _main_world(C, tmp_path, monkeypatch, disc_digest=FI.CODE_AT_IMPORT["sha256"])
    with pytest.raises(_StopBeforeLoad):
        C.main([])
    rec = _FakeRun.last.records
    assert rec["code_digest"] == {"sha256": FI.CODE_AT_IMPORT["sha256"], "source": "fitting.CODE_AT_IMPORT",
                                  "equals_discovery": True}
    assert set(rec) == {"smoke", "anchors_gate", "code_digest"}
    assert not (tmp_path / "confirmation").exists()                       # RUN_ONCE is opened after the fold gates
