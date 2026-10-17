"""run_design / load_tables / code_digest (contract "Wave-2 interfaces"): unit files, DONE markers, resume, the
smoke-label rules, the registered-tree-count guard, and LT-05's prediction part (a unit reproduces bit for bit in
fresh interpreters under PYTHONHASHSEED 1 and 2).  Written only under pytest's tmp_path; tiny forests."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from gen201 import bootstrap
from gen201 import fitting as FI
from gen201 import learners as L

import g201m_helpers as H


@pytest.fixture(scope="module")
def written(bench, tmp_path_factory):
    root = tmp_path_factory.mktemp("g201_pred")
    f = H.giant_fold(bench)
    with H.tiny_trees(4, 2):
        summ = FI.run_design(bench, "BP", seeds=(f.seed,), label="smoke_test", variants=("primary", "nodmd"),
                             smoke=True, fold_ids=[(f.seed, f.fold)], max_test_cells=12, out_root=root,
                             folds_ctx=H.context(bench), folds_list=H.bp_folds(bench), check_plan=False, log=lambda m: None)
    return root, f, summ


def test_unit_files_and_done_markers(written):
    root, f, summ = written
    assert [u["status"] for u in summ["units"]] == ["done", "done"]
    for v in ("primary", "nodmd"):
        p = FI.unit_paths(root, "smoke_test", "BP", v, f.seed, f.fold)
        for k in ("logd", "logsf", "selection", "inner_logd", "inner_logsf", "done"):
            assert p[k].exists(), (v, k)
        done = json.loads(p["done"].read_text(encoding="utf-8"))
        assert done["code_digest"] == FI.code_digest()["sha256"]
        assert done["logd_sha256"] == FI._sha(p["logd"]) and done["n_estimators"] == 4
        assert not done["registered_tree_counts"]
        sel = json.loads(p["selection"].read_text(encoding="utf-8"))
        assert sel["n_test_cells_scored"] == 12 and set(sel["chosen"]) == {"A", "C", "BD_COND", "B_selSF", "C_selSF"}
        for arm in ("A", "C", "BD_COND", "B_selSF", "C_selSF"):
            assert set(sel[arm]["scores"]) == set(L.CONFIG_ORDER)
        assert sel["BL_D_INNER"]["chosen"] in FI.H1_MEMBERS and sel["BL_SF_INNER"]["chosen"] in ("FLAT", "PAIRMEAN")


def test_load_tables_concatenates_and_verifies(written):
    root, f, _ = written
    logd, logsf = FI.load_tables("smoke_test", "BP", "primary", root=root, expect_units=1)
    assert logd["ci"].nunique() == 12 and len(logsf) > 0
    assert logd["y"].dtype == np.float64 and all(logd[a].dtype == np.float64 for a in FI.logd_arm_columns("BP", "primary"))
    with pytest.raises(FI.CoverageError):
        FI.load_tables("smoke_test", "BP", "primary", root=root, expect_units=25)


def test_resume_skips_matching_units_and_tampering_is_detected(bench, written):
    root, f, _ = written
    with H.tiny_trees(4, 2):
        again = FI.run_design(bench, "BP", seeds=(f.seed,), label="smoke_test", variants=("primary",), smoke=True,
                              fold_ids=[(f.seed, f.fold)], max_test_cells=12, out_root=root, folds_ctx=H.context(bench),
                              folds_list=H.bp_folds(bench), check_plan=False, log=lambda m: None)
    assert again["units"][0]["status"].startswith("skipped")
    p = FI.unit_paths(root, "smoke_test", "BP", "nodmd", f.seed, f.fold)
    assert FI.done_matches(p, FI.code_digest()["sha256"])
    assert not FI.done_matches(p, "0" * 64)
    raw = p["logsf"].read_bytes()
    p["logsf"].write_bytes(raw + b"\0")
    try:
        assert not FI.done_matches(p, FI.code_digest()["sha256"])
        with pytest.raises(FI.CoverageError):
            FI.load_tables("smoke_test", "BP", "nodmd", root=root)
    finally:
        p["logsf"].write_bytes(raw)


def test_label_rules_and_tree_count_guard(bench, tmp_path):
    with pytest.raises(ValueError):
        FI.run_design(bench, "BP", seeds=(104729,), label="smoke_x", out_root=tmp_path)
    with pytest.raises(ValueError):
        FI.run_design(bench, "BP", seeds=(104729,), label="discovery", smoke=True, out_root=tmp_path)
    with pytest.raises(ValueError):
        FI.run_design(bench, "BP", seeds=(104729,), label="discovery", fold_ids=[(104729, 0)], out_root=tmp_path)
    with H.tiny_trees(), pytest.raises(RuntimeError):
        FI.run_design(bench, "BP", seeds=(104729,), label="discovery", out_root=bootstrap.PREDICTIONS)
    with pytest.raises(PermissionError):
        FI.run_design(bench, "BP", seeds=(104729,), label="x", out_root=bootstrap.G20 / "predictions")


def test_code_digest_covers_the_fitting_code():
    cd = FI.code_digest()
    assert len(cd["sha256"]) == 64
    for rel in ("gen201/learners.py", "gen201/baselines.py", "gen201/fitting.py", "gen201/folds.py", "gen201/data.py"):
        assert cd["files"][rel] is not None


_LT05 = r"""
import sys, hashlib, json
sys.dont_write_bytecode = True
sys.path.insert(0, %r)
from gen201 import data as D, folds as FO, fitting as FI, learners as L, guard
guard.install()
L.N_ESTIMATORS, L.C_PAIR_N_ESTIMATORS = 4, 2
b = D.load(check=False)
f = next(x for x in FO.build(b, seeds=(104729,), designs=("BP",))["BP"] if D.SC009 in x.held_out_groups)
r = FI.run_unit(b, "BP", f, variant="primary", max_test_cells=15)
h = hashlib.sha256()
for t in (r.logd, r.logsf):
    for c in t.columns:
        h.update(c.encode()); h.update(t[c].astype(str).str.cat(sep="|").encode())
print(json.dumps({"digest": h.hexdigest(), "chosen": r.selection["chosen"]}))
"""


def test_lt05_unit_predictions_reproduce_in_subprocesses():
    outs = []
    for hs in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": hs, "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
        r = subprocess.run([sys.executable, "-B", "-c", _LT05 % str(bootstrap.G201)], capture_output=True, text=True,
                           env=env, cwd=str(bootstrap.ROOT))
        assert r.returncode == 0, r.stderr[-3000:]
        outs.append(json.loads(r.stdout.strip().splitlines()[-1]))
    assert outs[0] == outs[1]
