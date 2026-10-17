"""Robustness of the models side (review fixes, 2026-10-16), on synthetic arrays, tiny forests on the real bench
(predictions only, never a metric) and pytest's tmp_path:

* S8 recordability: a C / C_PAIR failure inside ``run_unit`` is an ``ArmFailure`` naming the learner (non-finite
  column or an exception in its fit), memory exhaustion is not; ``g201_smoke.run_with_fallback`` switches exactly
  that learner off and repeats, any other failure raises;
* resume keys: ``done_matches`` honours the arm set / tree counts / library versions, ``load_tables`` refuses mixed
  units and torn DONE markers;
* code digest at import: ``run_design`` / ``write_unit`` refuse once the code on disk differs (``CodeChanged``);
* writes: no ``.tmp`` leftovers, PermissionError retried, one writer per (label, design) (live lock refused, stale
  lock taken over, reused pid told apart on Windows);
* discovery: a failed design is recorded and the next starts (STOP errors still stop), children get resolved paths
  and the parent's code digest and are terminated when the parent fails;
* warning filters: an emptied ``warnings.filters`` (the sklearn worker-thread race) is restored around every forest
  call, so no "`sklearn.utils.parallel.delayed` should be used" flood.
"""
from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from gen201 import bootstrap
from gen201 import fitting as FI
from gen201 import folds as FO
from gen201 import learners as L

import g201m_helpers as H


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"_g201_rob_{name}", bootstrap.G201 / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def smoke():
    return _script("g201_smoke")


@pytest.fixture(scope="module")
def disc():
    return _script("g201_discovery")


@pytest.fixture(scope="module")
def unit_root(bench, tmp_path_factory):
    """One BP giant-fold unit (both variants, 12 test cells, tiny forests) written by run_design."""
    root = tmp_path_factory.mktemp("g201_rob")
    f = H.giant_fold(bench)
    with H.tiny_trees(4, 2):
        FI.run_design(bench, "BP", seeds=(f.seed,), label="smoke_rob", variants=("primary", "nodmd"), smoke=True,
                      fold_ids=[(f.seed, f.fold)], max_test_cells=12, out_root=root, folds_ctx=H.context(bench),
                      folds_list=H.bp_folds(bench), check_plan=False, log=lambda m: None)
    return root, f


# ---------------- S8: C / C_PAIR failures are recordable ---------------------------------------
def test_arm_family_and_assert_finite_names_the_arm():
    assert {FI.arm_family(a) for a in FI.C_FAMILY_ARMS} == {"C"}
    assert FI.arm_family("C_PAIR") == "C_PAIR" and FI.arm_family("A") == "A"
    import pandas as pd
    t = pd.DataFrame({"A": [1.0, 2.0], "C_K3": [0.0, np.nan], "C_PAIR": [np.inf, 0.0]})
    with pytest.raises(FI.ArmFailure) as e:
        FI.assert_finite(t, ["A", "C_K3", "C_PAIR"], "logsf")
    assert e.value.arm == "C_K3" and e.value.family == "C" and isinstance(e.value, FI.CoverageError)
    with pytest.raises(FI.ArmFailure) as e:
        FI.assert_finite(t, ["A", "C_PAIR"], "logsf")
    assert e.value.family == "C_PAIR"


def test_run_unit_raises_arm_failure_for_a_failing_centred_fit(bench, monkeypatch):
    orig = L.RowForest.fit

    def fit(self, *a, **k):
        if self.target == "centred":
            raise ValueError("synthetic failure of the centred learner")
        return orig(self, *a, **k)

    monkeypatch.setattr(L.RowForest, "fit", fit)
    with H.tiny_trees(4, 2), pytest.raises(FI.ArmFailure) as e:
        FI.run_unit(bench, "BP", H.giant_fold(bench), folds_ctx=H.context(bench), max_test_cells=5)
    assert e.value.family == "C" and isinstance(e.value.__cause__, ValueError)


def test_memory_errors_are_never_turned_into_a_not_run_learner(bench, monkeypatch):
    def fit(self, *a, **k):
        raise MemoryError("synthetic")

    monkeypatch.setattr(L.RowForest, "fit", fit)
    with H.tiny_trees(4, 2), pytest.raises(MemoryError):
        FI.run_unit(bench, "BP", H.giant_fold(bench), folds_ctx=H.context(bench), max_test_cells=5)


def test_run_unit_raises_arm_failure_for_a_failing_c_pair(bench, monkeypatch):
    def fit(self, *a, **k):
        raise RuntimeError("synthetic C_PAIR failure")

    monkeypatch.setattr(L.CPair, "fit", fit)
    with H.tiny_trees(4, 2), pytest.raises(FI.ArmFailure) as e:
        FI.run_unit(bench, "BP", H.giant_fold(bench), folds_ctx=H.context(bench), max_test_cells=5)
    assert e.value.arm == "C_PAIR" and e.value.family == "C_PAIR" and isinstance(e.value.__cause__, RuntimeError)
    with H.tiny_trees(4, 2):                                  # switched off: the unit runs, the column is absent
        r = FI.run_unit(bench, "BP", H.giant_fold(bench), folds_ctx=H.context(bench), max_test_cells=5,
                        include_c_pair=False)
    assert "C_PAIR" not in r.logsf.columns and r.selection["c_pair"] is None


def test_smoke_fallback_switches_off_only_the_failing_learner(smoke):
    calls = []

    def fit(ic, icp):
        calls.append((ic, icp))
        if ic:
            raise FI.ArmFailure("C_K2", "logd: arm C_K2 has 3 non-finite predictions of 40")
        if icp:
            raise FI.ArmFailure("C_PAIR", "C_PAIR: fit/prediction failed")
        return {"units": []}

    summ, inc, fail = smoke.run_with_fallback(fit, log=lambda m: None)
    assert calls == [(True, True), (False, True), (False, False)]
    assert inc == {"C": False, "C_PAIR": False} and fail["C"]["arm"] == "C_K2" and "C_PAIR" in fail
    for exc in (FI.CoverageError("logd rows differ"), FI.ArmFailure("A", "A non-finite"), ValueError("x")):
        def bad(ic, icp, exc=exc):
            raise exc
        with pytest.raises(type(exc)):
            smoke.run_with_fallback(bad, log=lambda m: None)

    def again(ic, icp):                                      # a failure of a learner already off is not absorbed
        raise FI.ArmFailure("C", "C again")
    with pytest.raises(FI.ArmFailure):
        smoke.run_with_fallback(again, log=lambda m: None)


# ---------------- resume keys and load_tables ---------------------------------------------------
def test_done_matches_honours_arm_set_tree_counts_and_versions(unit_root):
    root, f = unit_root
    p = FI.unit_paths(root, "smoke_rob", "BP", "primary", f.seed, f.fold)
    code = FI.CODE_AT_IMPORT["sha256"]
    with H.tiny_trees(4, 2):
        assert FI.done_matches(p, code, expect=FI.unit_expectation("BP", "primary"))
        assert not FI.done_matches(p, code, expect=FI.unit_expectation("BP", "primary", include_c=False))
        assert not FI.done_matches(p, code, expect=FI.unit_expectation("BP", "primary", include_c_pair=False))
        assert not FI.done_matches(p, code, expect={**FI.unit_expectation("BP", "primary"),
                                                    "runtime": {**FI.runtime_versions(), "sklearn": "0.0"}})
    assert not FI.done_matches(p, code, expect=FI.unit_expectation("BP", "primary"))      # 400/200 trees now
    rec = json.loads(p["done"].read_text(encoding="utf-8"))
    assert rec["c_pair_n_estimators"] == 2 and rec["runtime"] == FI.runtime_versions()


def test_resume_refits_units_produced_with_another_arm_set(bench, unit_root, tmp_path):
    root, f = unit_root
    work = tmp_path / "pred"
    shutil.copytree(root, work)
    kw = dict(seeds=(f.seed,), label="smoke_rob", variants=("primary",), smoke=True, fold_ids=[(f.seed, f.fold)],
              max_test_cells=12, out_root=work, folds_ctx=H.context(bench), folds_list=H.bp_folds(bench),
              check_plan=False, log=lambda m: None)
    with H.tiny_trees(4, 2):
        same = FI.run_design(bench, "BP", **kw)
        assert same["units"][0]["status"].startswith("skipped")
        changed = FI.run_design(bench, "BP", include_c=False, **kw)      # the S8 flags changed: refit, not reuse
    assert changed["units"][0]["status"] == "done"
    logd, logsf = FI.load_tables("smoke_rob", "BP", "primary", root=work)
    assert "C" not in logd.columns and "C" not in logsf.columns


def test_load_tables_refuses_mixed_units_and_torn_markers(unit_root, tmp_path):
    root, f = unit_root
    src = FI.unit_paths(root, "smoke_rob", "BP", "primary", f.seed, f.fold)
    work = tmp_path / "pred"
    for seed in (1, 2):
        dst = FI.unit_paths(work, "smoke_rob", "BP", "primary", seed, f.fold)
        dst["dir"].mkdir(parents=True, exist_ok=True)
        for k in ("logd", "logsf", "selection", "inner_logd", "inner_logsf"):
            shutil.copyfile(src[k], dst[k])
        rec = json.loads(src["done"].read_text(encoding="utf-8"))
        rec["split_seed"] = seed
        if seed == 2:
            rec["arms_logd"] = [a for a in rec["arms_logd"] if FI.arm_family(a) != "C"]
        dst["done"].write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(FI.CoverageError, match="arms_logd"):
        FI.load_tables("smoke_rob", "BP", "primary", root=work)
    torn = FI.unit_paths(work, "smoke_rob", "BP", "primary", 2, f.fold)["done"]
    torn.write_text('{"label": "smoke_rob", "spl', encoding="utf-8")
    with pytest.raises(FI.CoverageError, match="unreadable DONE"):
        FI.load_tables("smoke_rob", "BP", "primary", root=work)


def test_unit_writes_leave_no_temporary_files(unit_root):
    root, _ = unit_root
    assert not list(root.rglob("*.tmp")) and not list(root.rglob(FI.LOCK_NAME))


def test_retry_on_permission_error(monkeypatch):
    monkeypatch.setattr(FI.time, "sleep", lambda s: None)
    n = {"k": 0}

    def flaky():
        n["k"] += 1
        if n["k"] < 3:
            raise PermissionError("held by the indexer")
        return "ok"

    assert FI._retry(flaky) == "ok" and n["k"] == 3

    def never():
        raise PermissionError("held")
    with pytest.raises(PermissionError):
        FI._retry(never)


# ---------------- code digest at import ----------------------------------------------------------
def test_changed_code_stops_run_design_and_write_unit(bench, unit_root, tmp_path, monkeypatch):
    assert FI.assert_code_unchanged()["sha256"] == FI.CODE_AT_IMPORT["sha256"]
    root, f = unit_root
    old = dict(FI.CODE_AT_IMPORT)
    monkeypatch.setattr(FI, "CODE_AT_IMPORT", {"sha256": "0" * 64, "files": {k: None for k in old["files"]}})
    with pytest.raises(FI.CodeChanged):
        FI.assert_code_unchanged()
    with H.tiny_trees(4, 2), pytest.raises(FI.CodeChanged):
        FI.run_design(bench, "BP", seeds=(f.seed,), label="smoke_rob", variants=("primary",), smoke=True,
                      fold_ids=[(f.seed, f.fold)], max_test_cells=12, out_root=tmp_path, folds_ctx=H.context(bench),
                      folds_list=H.bp_folds(bench), check_plan=False, log=lambda m: None)
    res = H.small_unit(bench)
    with pytest.raises(FI.CodeChanged):
        FI.write_unit(res, FI.unit_paths(tmp_path, "smoke_rob", "BP", "primary", 1, 0), old, label="smoke_rob")
    assert not (tmp_path / "smoke_rob").exists()


def test_child_refuses_a_different_code_digest(disc, capsys):
    assert disc.main(["--label", "x", "--child", "--expect-code-digest", "0" * 64]) == 1
    assert "STOP" in capsys.readouterr().err


# ---------------- one writer per (label, design) -------------------------------------------------
def _dead_pid() -> int:
    r = subprocess.run([sys.executable, "-B", "-c", "import os; print(os.getpid())"], capture_output=True, text=True)
    return int(r.stdout.strip())


def test_design_lock_live_stale_and_reused_pid(bench, tmp_path):
    d = tmp_path / "lab" / "BP"
    with FI.design_lock(d, log=lambda m: None) as me:
        assert me["pid"] == os.getpid() and (d / FI.LOCK_NAME).exists()
        with pytest.raises(FI.DesignLocked):
            with FI.design_lock(d, log=lambda m: None):
                pass
    assert not (d / FI.LOCK_NAME).exists()
    (d / FI.LOCK_NAME).write_text(json.dumps({"pid": _dead_pid(), "process_start": None, "host": platform.node()}),
                                  encoding="utf-8")
    msgs = []
    with FI.design_lock(d, log=msgs.append):
        pass
    assert msgs and "stale" in msgs[0] and not (d / FI.LOCK_NAME).exists()
    if os.name == "nt":                                       # this pid, but another process start time: reused pid
        assert FI.process_state(os.getpid())[1] is not None
        (d / FI.LOCK_NAME).write_text(json.dumps({"pid": os.getpid(), "process_start": 1, "host": platform.node()}),
                                      encoding="utf-8")
        with FI.design_lock(d, log=lambda m: None):
            pass
    # run_design refuses while a live process holds the design
    f = H.giant_fold(bench)
    with FI.design_lock(tmp_path / "smoke_lock" / "BP", log=lambda m: None):
        with H.tiny_trees(4, 2), pytest.raises(FI.DesignLocked):
            FI.run_design(bench, "BP", seeds=(f.seed,), label="smoke_lock", variants=("primary",), smoke=True,
                          fold_ids=[(f.seed, f.fold)], max_test_cells=5, out_root=tmp_path,
                          folds_ctx=H.context(bench), folds_list=H.bp_folds(bench), check_plan=False,
                          log=lambda m: None)


# ---------------- discovery: per-design isolation and children --------------------------------
class _Run:
    def __init__(self):
        self.records = {}

    def record(self, k, v):
        self.records[k] = v


def test_discovery_records_a_failed_design_and_continues(disc):
    run = _Run()

    def one(d):
        if d == "B":
            raise MemoryError("synthetic")
        return {"design": d, "units": []}

    assert disc.run_designs(["BP", "B", "BR"], one, run, lambda m: None) == ["B"]
    assert run.records["design_BR"]["design"] == "BR" and run.records["design_B"]["status"] == "failed"
    assert "MemoryError" in run.records["design_B"]["traceback"]
    for stop in (FO.FoldGateError("STOP INCOMPARABLE_FOLDS"), L.LeakageError("x"), FI.CodeChanged("x"),
                 FI.DesignLocked("x")):
        def bad(d, stop=stop):
            raise stop
        with pytest.raises(type(stop)):
            disc.run_designs(["BP", "B"], bad, _Run(), lambda m: None)


def test_children_get_resolved_paths_and_the_code_digest(disc, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    args = SimpleNamespace(label="discovery", variants="primary,nodmd", smoke_record="rel/smoke.json",
                           anchors_record="rel/anchors.json", manifest_dir="rel/man", no_resume=False, tiny=False,
                           no_subprocess=False)
    root = bootstrap.PREDICTIONS.resolve()
    cmd = disc.child_command("B", args, root=root, code_sha="ab" * 32)
    val = {k: cmd[cmd.index(k) + 1] for k in ("--smoke-record", "--anchors-record", "--root", "--manifest-dir",
                                               "--expect-code-digest", "--designs")}
    for k in ("--smoke-record", "--anchors-record", "--root", "--manifest-dir"):
        assert os.path.isabs(val[k]), k
    assert val["--smoke-record"] == str((tmp_path / "rel" / "smoke.json").resolve())
    assert val["--expect-code-digest"] == "ab" * 32 and val["--designs"] == "B" and "--child" in cmd


def test_children_are_terminated_when_the_parent_fails(disc, monkeypatch, tmp_path):
    monkeypatch.setattr(disc, "child_command",
                        lambda d, args, root, code_sha: [sys.executable, "-B", "-c", "import time; time.sleep(120)"])
    pids = []

    class ParentDies(RuntimeError):
        pass

    def log(m):
        if m.startswith("started child"):
            pids.append(int(m.split("pid ")[1].rstrip(")")))
            raise ParentDies(m)

    with pytest.raises(ParentDies):
        disc.run_children(["BP"], SimpleNamespace(workers=1), log, root=tmp_path, code_sha="x", poll=0.1)
    assert pids and not FI.process_state(pids[0])[0]


# ---------------- warning filters around threaded forest calls -------------------------------
def _delayed_warnings(log) -> int:
    return sum("sklearn.utils.parallel.delayed" in str(w.message) for w in log)


def test_emptied_warning_filters_are_restored_around_forest_calls():
    import gen13sep.models as M13
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 6))
    Y = np.where(rng.uniform(size=(40, 14)) < 0.6, rng.normal(size=(40, 14)), np.nan)
    Y[:, 0] = rng.normal(size=40)
    groups = np.array([f"g{i % 5}" for i in range(40)])
    with warnings.catch_warnings(record=True) as log:
        warnings.filters[:] = []                             # the state the worker-thread race leaves behind
        M13.tree_pipeline(0, n_estimators=4).fit(X, rng.normal(size=40))
        assert _delayed_warnings(log) > 0                    # control: the flood mechanism is real (sklearn 1.9)
    with warnings.catch_warnings(record=True) as log:
        warnings.filters[:] = []
        n0 = L.FILTER_RESTORES
        rf = L.RowForest("raw", "K1", 0, n_estimators=4).fit(X, Y, groups, np.arange(30))
        warnings.filters[:] = []
        P = rf.predict(X, np.arange(30, 40))
        assert np.isfinite(P).all() and _delayed_warnings(log) == 0
        assert L.FILTER_RESTORES >= n0 + 2 and warnings.filters
    saved = list(warnings.filters)
    with L.guarded_filters():
        warnings.filters[:] = []                             # a call that empties the list is undone after it
    assert warnings.filters == saved


def test_unit_selection_records_the_filter_restores(unit_root):
    root, f = unit_root
    sel = json.loads(FI.unit_paths(root, "smoke_rob", "BP", "primary", f.seed, f.fold)["selection"]
                     .read_text(encoding="utf-8"))
    assert isinstance(sel["timing"]["warning_filter_restores"], int)
