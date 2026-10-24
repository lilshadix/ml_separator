"""Gate tests for every Stage-1 entry point (pre-seal review s0-mechanics-3, -4, -17, -21): with the heavy data paths
replaced by stubs, each sealed scoring run (1) refuses without a verified seal, before any work; (2) runs its
reproduction target first and stops on a mismatch with SystemExit, a STOP line in the run record and no rows parquet
written.  Nothing here reads a label, computes a score or writes outside the pytest temporary folder.

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B -W ignore -m pytest generations/gen21/round5_20261024/tests/test_s1_gates.py
        -q -p no:cacheprovider --basetemp generations/gen21/round5_20261024/cache/_pytest
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

R5 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R5))
from r5 import common as CM  # noqa: E402
from r5 import sealcheck  # noqa: E402

SCRIPTS = ("s1_ea_check", "s1_ect", "s1_m03", "s1_elab", "s1_window", "s1_edec", "s1_ed", "s1_ew", "s1_ekx")


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"gate_{name}", R5 / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("name", SCRIPTS + ("s1_ed_poison",))
def test_scoring_refuses_without_the_seal(name, monkeypatch):
    mod = load(name)

    def refuse(*a, **k):
        raise SystemExit("REFUSED-TEST")
    monkeypatch.setattr(sealcheck, "require_seal", refuse)
    with pytest.raises(SystemExit, match="REFUSED-TEST"):
        mod.main([])


@pytest.mark.parametrize("name", ("s1_ect", "s1_m03", "s1_elab", "s1_edec", "s1_ed", "s1_ew"))
def test_reduced_coverage_flags_are_smoke_only(name):
    mod = load(name)
    flag = {"s1_ect": ["--intervention-seeds", "1"], "s1_m03": ["--intervention-seeds", "1"],
            "s1_elab": ["--iv-seeds", "1"], "s1_edec": ["--poison-seeds", "1"], "s1_ed": ["--no-intervention"],
            "s1_ew": ["--no-lopo"]}[name]
    with pytest.raises(SystemExit, match="reduced leakage coverage"):
        mod.main(flag)


class _Corpus:
    def target_table(self):
        return pd.DataFrame({"cell_id": ["t"]})

    def links_table(self):
        return pd.DataFrame({"target": ["t"], "sibling": ["s"]})

    def build_rows(self, design, y_mode="labels", **k):
        return pd.DataFrame({"x": [1.0]}), {"b1_maxabs": 0.0}


def _mismatch(*a, **k):
    return {"passed": False, "ok": False, "identity_ok": False, "targets": {}, "checks": {"rows_compared": 0},
            "rows": 0, "designs": {}}


def _stubs(name, mod, monkeypatch, tmp):
    """Replace every heavy path of ``name`` so that its run reaches the reproduction gate on stub data."""
    from gen20k import data as GD
    bench = SimpleNamespace(frame=pd.DataFrame({"cell_id": ["c1"]}), Y=np.zeros((1, 14)), basis=None)
    monkeypatch.setattr(GD, "load_bench", lambda *a, **k: bench)
    monkeypatch.setattr(mod, "OUT", tmp)
    if name == "s1_ea_check":
        monkeypatch.setattr(mod.EA, "repro", _mismatch)
        return []
    if name == "s1_ect":
        monkeypatch.setattr(mod.ECT, "Corpus", _Corpus)
        monkeypatch.setattr(mod.REP, "reproduction", _mismatch)
        return ["--design", "BPG"]
    if name == "s1_m03":
        monkeypatch.setattr(mod.ECT, "Corpus", _Corpus)
        chk = {"b1_maxabs": 0.0, "y_maxabs": 0.0, "readset_mismatch": 0, "ect_identity_maxabs": {},
               "ect_identity_groups": 1}
        monkeypatch.setattr(mod.M3, "build", lambda *a, **k: (pd.DataFrame({"x": [1.0]}), None, dict(chk)))
        monkeypatch.setattr(mod.M3, "add_config_columns", lambda *a, **k: 0.0)
        monkeypatch.setattr(mod.M3, "reproduction", _mismatch)
        return ["--design", "BPG"]
    if name == "s1_elab":
        class Camp:
            def build(self, design, **k):
                return pd.DataFrame({"x": [1.0]}), {"b1_parquet_maxabs": 0.0}, None
        monkeypatch.setattr(mod.EL, "Campaign", Camp)
        monkeypatch.setattr(mod.EL, "reproduction", _mismatch)
        return ["--design", "BPG"]
    if name == "s1_ekx":
        monkeypatch.setattr(mod.EKX, "Donors", lambda *a, **k: SimpleNamespace())
        monkeypatch.setattr(mod.EKX, "build_rows", lambda D, d: (pd.DataFrame({"x": [1.0]}), {}))
        monkeypatch.setattr(mod.EKX, "reproduction", lambda D, built: {
            "passed": False, "rows_total": 0, "consistency": {**{d: {} for d in built}, "frozen_value_source": {}},
            "targets": {d: {"passed": False, "b1_predict_levels_maxabs": 1.0, "checks": {"rows": 0}} for d in built}})
        return ["--design", "BPG"]
    if name == "s1_edec":
        monkeypatch.setattr(mod, "load_ect", lambda *a, **k: (None, None, "stub"))
        monkeypatch.setattr(mod.CM, "load_rows", lambda *a, **k: pd.DataFrame({"x": [1.0]}))
        monkeypatch.setattr(mod.ED, "prepare", lambda d: d)
        monkeypatch.setattr(mod.ED, "read_values", lambda d, b: (None, None))
        monkeypatch.setattr(mod, "repro_targets", lambda d, c: {"stub": {"ok": False}})
        return ["--design", "BPG"]
    if name == "s1_window":
        R = pd.DataFrame({"cell_id": ["c1"], "split_seed": [1], "s": [0.0]})
        monkeypatch.setattr(mod.WIN, "assemble", lambda *a, **k: {"rows": R, "ops": pd.DataFrame(),
                                                                   "checks": {"folds": 1, "seconds": 0.0}})
        monkeypatch.setattr(mod.WIN, "checks_failed", lambda c: [])
        monkeypatch.setattr(mod.WIN, "repro_checks", _mismatch)
        return ["--design", "BPG"]
    if name == "s1_ed":
        monkeypatch.setattr(mod, "cluster_map", lambda: {"c1": "g1"})
        monkeypatch.setattr(mod.ED, "LEAK", tmp)
        monkeypatch.setattr(mod.ED, "static_tables", lambda *a, **k: {0: {"sets": {}, "cx": False}})
        monkeypatch.setattr(mod.ED, "loco_static_selection", lambda *a, **k: pd.DataFrame({"reads_req": [2]}))
        monkeypatch.setattr(mod.ED, "poison_loco", lambda *a, **k: {"passed": True, "clusters": 1, "changed": [],
                                                                    "pc_caught": 1, "pc_runs": 1})
        ck = {"folds": 1, "intervention_checked": 1, "read_sets_checked": 1, "peek_control_runs": 1,
              "peek_control_caught": 1}
        monkeypatch.setattr(mod.ED, "compute_design", lambda *a, **k: {
            "rows": pd.DataFrame({"family": ["LAD"], "se_B1": [0.0]}), "cand": pd.DataFrame(), "checks": dict(ck),
            "fits": pd.DataFrame({"k": []}), "seconds": 0.0})
        monkeypatch.setattr(mod.ED, "repro_checks", _mismatch)
        return ["--design", "BPG"]
    if name == "s1_ew":
        monkeypatch.setattr(mod.EW, "cluster_map", lambda: {"c1": "g1"})
        monkeypatch.setattr(mod.EW, "compute_design", lambda *a, **k: {"rows": pd.DataFrame({"x": [1.0]}),
                                                                       "checks": {}, "seconds": 0.0})
        monkeypatch.setattr(mod, "_counts", lambda R: "")
        monkeypatch.setattr(mod.EW, "bad_checks", lambda c, **k: [])
        monkeypatch.setattr(mod.EW, "repro_checks", lambda R, d: {**_mismatch(), "rows_matched": 0, "rows_REF": 0,
                                                                  "max_abs_diff": {}})
        return ["--design", "BPG"]
    raise KeyError(name)


@pytest.mark.parametrize("name", SCRIPTS)
def test_reproduction_mismatch_stops_before_any_scored_write(name, monkeypatch, tmp_path):
    mod = load(name)
    out = tmp_path / "out"
    out.mkdir()
    rr = tmp_path / "RUN_RECORD.md"
    monkeypatch.setattr(CM, "RUN_RECORD", rr)
    monkeypatch.setattr(sealcheck, "require_seal", lambda *a, **k: "TESTTAG")
    ends = []
    monkeypatch.setattr(sealcheck, "post_run", lambda tag, item, note="", **k: ends.append(note) or True)
    argv = _stubs(name, mod, monkeypatch, out)
    with pytest.raises(SystemExit, match="mismatch"):
        mod.main(argv)
    assert not list(out.rglob("rows_*.parquet")) and not list(out.rglob("summary*.json"))
    text = rr.read_text(encoding="utf-8")
    assert " START " in text and " STOP " in text and "seal=TESTTAG" in text
    assert ends and "STOP" in ends[-1]                    # the post-run check ran and its END line names the stop
