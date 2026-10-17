"""LT-28: the fold-identity gates (a)-(e) of IMPLEMENTATION_PLAN.md §4: (a) the gen16 test-index and train+test
plan digests, (b) the per-design digests (re-implemented, never imported from gen16), (c) the 25 BP fold sizes of
gen20's folds_BP.json, (d) B/BR/BQ/BPG test sets equal BP's, (e) the new outer+inner digest (deterministic, and
equal to manifests/fold_plan_g201.json once S3 has written it).  Perturbed plans must fail."""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

from gen201 import bootstrap
from gen201 import folds as FO


def test_reference_constants_equal_anchors_json():
    ref = FO.anchors_reference()
    assert ref["sha256"] == FO.EXPECTED_TEST_PLAN_SHA256
    assert ref["sha256_train_test"] == FO.EXPECTED_TRAIN_TEST_SHA256
    assert ref["per_design"] == FO.EXPECTED_PER_DESIGN_SHA256
    assert FO.EXPECTED_TEST_PLAN_SHA256.startswith("7046c640")
    assert FO.EXPECTED_TRAIN_TEST_SHA256.startswith("9a24b6e9")
    assert FO.EXPECTED_PER_DESIGN_SHA256["BP"].startswith("b50c9fd6")
    assert FO.EXPECTED_PER_DESIGN_SHA256["A"].startswith("afda7475")
    from gen15.valuebench import DESIGNS
    assert tuple(DESIGNS) == FO.GEN16_DESIGN_ORDER
    import sys
    assert not any(m == "gen16" or m.startswith("gen16.") for m in sys.modules)   # never imported


def test_gates_a_b(folds):
    dig = FO.gen16_digest(folds)
    assert FO.gate_a(dig)["pass"]
    assert FO.gate_b(dig)["pass"], FO.gate_b(dig)["mismatched"]
    assert dig["n_folds"] == {d: 25 for d in FO.GEN16_DESIGN_ORDER}


def test_gate_c(folds):
    g = FO.gate_c(folds["BP"])
    assert g["pass"] and g["n_match"] == 25 and not g["mismatched"]


def test_gate_d(folds):
    assert FO.gate_d(folds)["pass"]


def test_gate_e_recorded_or_deterministic(bench, folds, plan):
    rec = FO.recorded_plan()
    ge = FO.gate_e(plan, recorded=rec)
    assert ge["pass"], ge
    assert len(plan["sha256"]) == 64 and set(plan["per_design"]) == set(FO.DESIGNS)
    if rec is not None:
        assert ge["equals_recorded"]
        path = bootstrap.MANIFESTS / "fold_plan_g201.json"
        fp = json.loads(path.read_text(encoding="utf-8"))
        assert fp["gate_a"]["pass"] and fp["gate_b"]["pass"] and fp["gate_c"]["pass"] and fp["gate_d"]["pass"]


def _swap_test_cell(F):
    """Move one held-out cell of fold 0 into fold 1's test set (seed 0): same cells, different plan."""
    f0, f1 = F[0], F[1]
    c = f0.test_index[0]
    g0 = replace(f0, test_index=f0.test_index[1:], train_index=np.sort(np.r_[f0.train_index, c]))
    g1 = replace(f1, test_index=np.sort(np.r_[f1.test_index, c]),
                 train_index=f1.train_index[f1.train_index != c])
    return [g0, g1] + list(F[2:])


def test_perturbed_plans_fail_gates(bench, folds, plan):
    bad = dict(folds)
    bad["BP"] = _swap_test_cell(folds["BP"])
    dig = FO.gen16_digest(bad)
    assert not FO.gate_a(dig)["pass"]
    assert not FO.gate_b(dig)["pass"] and "BP" in FO.gate_b(dig)["mismatched"]
    assert not FO.gate_c(bad["BP"])["pass"]
    assert not FO.gate_d(bad)["pass"]
    pbad = FO.plan_digest(bench, bad)
    assert pbad["sha256"] != plan["sha256"]
    assert not FO.gate_e(pbad, recorded={"sha256": plan["sha256"], "per_design": plan["per_design"]})["pass"]
    # a train-only change moves the train+test digest but not the test-index digest
    tr = dict(folds)
    f = folds["B"][3]
    tr["B"] = list(folds["B"][:3]) + [replace(f, train_index=f.train_index[1:])] + list(folds["B"][4:])
    d2 = FO.gen16_digest(tr)
    assert d2["sha256"] == FO.EXPECTED_TEST_PLAN_SHA256
    assert d2["sha256_train_test"] != FO.EXPECTED_TRAIN_TEST_SHA256


def test_inner_change_moves_only_the_gate_e_digest(bench, folds, plan, monkeypatch):
    monkeypatch.setattr(FO, "MIN_INNER_TRAIN", 18)          # un-skips the one 18-cell BP inner fold
    p2 = FO.plan_digest(bench, folds)
    assert p2["per_design"]["BP"] != plan["per_design"]["BP"]
    assert p2["per_design"]["B"] == plan["per_design"]["B"]
