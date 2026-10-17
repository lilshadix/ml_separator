"""LT-05 (fold part): the fold plan reproduces in fresh interpreters under PYTHONHASHSEED 1 and 2: the gen16
digests of gates (a)/(b) and the gate-(e) outer+inner digest are identical to the in-process values.
(The prediction part of LT-05 is added with the learners.)"""
from __future__ import annotations

from gen201 import folds as FO


def test_fold_digests_reproduce_in_subprocesses(folds, plan):
    dig = FO.gen16_digest(folds)
    res = FO.gate_b_subprocess(("1", "2"), plan_sha256=plan["sha256"])
    assert res["pass"], res
    assert res["identical_across_hash_seeds"]
    assert res["plan_digest_equals_in_process"]
    for run in res["runs"].values():
        assert run["sha256"] == dig["sha256"]
        assert run["sha256_train_test"] == dig["sha256_train_test"]
        assert run["per_design"] == dig["per_design"]
        assert run["plan_per_design"] == plan["per_design"]
    ge = FO.gate_e(plan, subprocess_runs=res["runs"])
    assert ge["pass"] and ge["subprocess_identical"]


def test_plan_digest_is_deterministic_in_process(bench, folds, plan):
    again = FO.plan_digest(bench, FO.build(bench))
    assert again["sha256"] == plan["sha256"]
    assert again["per_design"] == plan["per_design"]
