"""The single registered confirmation run (pre-registration section 15, POST-HOC addendum 5 item 1).

Nothing here touches the real confirmation half, V6 or the withheld seeds: the corpus is a synthetic mini-corpus in
``tmp_path``, the seed store is five PLACEHOLDER seeds written in ``tmp_path`` with their own commitment, and every gate
runs against a registry file of the test's own.  What is asserted: the five gates refuse for the five registered
reasons, a withheld seed reaches no written file or log line, the fold colouring is seed-dependent, R19 item 4 needs
5 of 5, S2(a) counts signs conservatively, S1(c) combines the seeds as its confirmation rule says, the section 11 V6
deltas and the section 8 power check are recorded NOT_RUN with cost and consequence, and the confirmation half and the
V6 carve-out are both enforced in both directions.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gen19ct import paths
from gen19ct.chemistry import support_graph as SG
from gen19ct.evaluation import confirmation as CF
from gen19ct.evaluation import discovery as D
from gen19ct.evaluation import registry as REG
from gen19ct.evaluation import transfer as ET
from gen19ct.folds import cell_holdout as CH
from gen19ct.folds import io as FI
from gen19ct.models import interface as I

SCRIPTS = paths.G19_ROOT / "scripts"


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


RC = _load("g19_run_confirmation")
SEAL = _load("g19_seal_prereg")

#: five PLACEHOLDER seeds -- in the registered range, not discovery seeds, and not the withheld ones (nobody knows those)
FAKE_SEEDS = [123457, 234561, 345612, 456123, 561234]


# --------------------------------------------------------------------------------------------- #
# fixtures: a fake seed store, a fake plan, a registry of the test's own
# --------------------------------------------------------------------------------------------- #

@pytest.fixture()
def root(tmp_path: Path) -> Path:
    (tmp_path / "manifests").mkdir(parents=True, exist_ok=True)
    (tmp_path / "decisions").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _write_store(root: Path, store_path: Path, seeds=None) -> Path:
    payload = {"n": CF.N_SEEDS, "salt": "ab" * 32, "schema": SEAL.SEED_SCHEMA,
               "seeds": sorted(FAKE_SEEDS if seeds is None else seeds)}
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    (root / "manifests" / "confirmation_seeds_sha256.txt").write_text(SEAL.seed_digest(payload) + "\n",
                                                                     encoding="utf-8", newline="\n")
    return store_path


PLAN = """# CONFIRMATION_PLAN (synthetic)

## 2. The frozen claims

| # | claim | family | arm (candidate) | comparator | margin | design(s) | discovery | verdict |
|---|---|---|---|---|---|---|---|---|
| **C1** | M2 vs B3i @ V5 | primary (H1, S1(a)) | **M2** | **B3i** | **d5 = 0.10568948790529951** | V5-primary | **+0.262976** | UNDECIDED |
| **C4** | B6:WITH vs B6:ACT_PERMUTED @ V2 | H3 | **B6:WITH** | **B6:ACT_PERMUTED** | **0.05** | V2 Ln(III) folds | **+0.081519** | UNDECIDED |
"""


@pytest.fixture()
def plan_file(root: Path) -> Path:
    p = root / "decisions" / "CONFIRMATION_PLAN.md"
    p.write_text(PLAN, encoding="utf-8", newline="\n")
    return p


@pytest.fixture()
def registry(root: Path, monkeypatch) -> Path:
    """A registry file of the test's own with a ``confirmation`` entry carrying the LIVE code digest."""
    p = root / "manifests" / "digest_registry.json"
    monkeypatch.setattr(REG, "registry_path", lambda r=None: p)
    REG.register_stage(CF.STAGE, below_footer_sha256="a" * 64, code_digest=RC.code_digest()["combined"],
                       git_head=None, addenda_count=5, note="test", path=p)
    return p


def _gate_kwargs(root: Path):
    return {"check": lambda: 0,
            "digests": lambda: {"footer": D.REGISTERED_PREREG_SHA256, "recomputed": D.REGISTERED_PREREG_SHA256,
                                "digest_file": D.REGISTERED_PREREG_SHA256, "n_addenda": 5, "addenda_sha256": "a" * 64}}


def _prereg_paths(root: Path):
    return SEAL.PreregPaths(root=root, repo_root=root.parent)


def _gate(root: Path, store_path: Path | None, *, resume: bool = False, plan=None, registry_path: Path | None = None):
    store = None
    if store_path is not None:
        store = CF.load_seed_store(store_path, root=root, seal=SEAL, prereg_paths=_prereg_paths(root))
    return CF.gate(root, seed_store=store, code_digest=RC.code_digest()["combined"], resume=resume,
                   plan=plan, registry_path=registry_path, **_gate_kwargs(root))


# --------------------------------------------------------------------------------------------- #
# (a)-(e) the five gates
# --------------------------------------------------------------------------------------------- #

def test_gate_passes_with_a_verified_store(root, plan_file, registry):
    store_path = _write_store(root, root.parent / "outside_store.json")
    g = _gate(root, store_path, registry_path=registry)
    assert g["claims"] == ["C1", "C4"] and g["seed_store"]["verified_against_commitment"] is True
    assert g["seed_store"]["values"] == "withheld" and g["lock"]["mode"] == "first_run"
    # the gate's own output names no seed
    assert not any(str(s) in json.dumps(g, default=str) for s in FAKE_SEEDS)


def test_gate_refuses_without_a_store(root, plan_file, registry):
    with pytest.raises(SystemExit, match="needs --seed-store"):
        _gate(root, None, registry_path=registry)


def test_gate_refuses_a_store_that_does_not_verify(root, plan_file, registry):
    """A wrong digest: the store is edited after the commitment was written."""
    store_path = _write_store(root, root.parent / "outside_store.json")
    bad = json.loads(store_path.read_text(encoding="utf-8"))
    bad["seeds"] = sorted([*bad["seeds"][:-1], 999991])
    store_path.write_text(json.dumps(bad, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(SystemExit, match="does not verify against the section 15 commitment"):
        _gate(root, store_path, registry_path=registry)


def test_gate_refuses_a_missing_store_file(root, plan_file, registry):
    _write_store(root, root.parent / "outside_store.json")
    with pytest.raises(SystemExit, match="does not exist"):
        _gate(root, root.parent / "absent_store.json", registry_path=registry)


def test_gate_refuses_a_second_run_without_resume(root, plan_file, registry):
    store_path = _write_store(root, root.parent / "outside_store.json")
    CF.write_decisions(root, {"code_digest": RC.code_digest()["combined"], "claim_ids": ["C1", "C4"]})
    with pytest.raises(SystemExit, match="nothing is re-run"):
        _gate(root, store_path, registry_path=registry)
    g = _gate(root, store_path, resume=True, registry_path=registry)
    assert g["lock"]["mode"] == "resume" and g["lock"]["may_only"] == "complete unfitted folds"


def test_resume_refuses_new_code_or_a_wider_claim_list(root, plan_file, registry):
    CF.write_decisions(root, {"code_digest": "other-code", "claim_ids": ["C1"]})
    v = CF.lock_verdict(root, resume=True, code_digest="live-code", claim_ids=["C1"])
    assert v["ok"] is False and v["mode"] == "resume_refused_code_changed"
    CF.write_decisions(root, {"code_digest": "live-code", "claim_ids": ["C1"]})
    v2 = CF.lock_verdict(root, resume=True, code_digest="live-code", claim_ids=["C1", "C9"])
    assert v2["ok"] is False and v2["new_claims"] == ["C9"]


def test_a_claim_not_in_the_plan_is_refused(root, plan_file):
    plan = CF.read_plan(plan_file)
    assert CF.claim_of(plan, "C4").design == "V2"
    with pytest.raises(SystemExit, match="is not a frozen claim of the plan"):
        CF.claim_of(plan, "C3")


def test_plan_refuses_more_than_five_claims(root):
    rows = "\n".join(f"| **C{k}** | x | f | **M2** | **B0** | **0.05** | V5-primary | **+0.1** | UNDECIDED |"
                     for k in range(1, 7))
    p = root / "decisions" / "plan6.md"
    p.write_text("| # |\n|---|\n" + rows + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(SystemExit, match="at most 5"):
        CF.read_plan(p)


def test_gate_refuses_an_unregistered_stage(root, plan_file, monkeypatch):
    p = root / "manifests" / "empty_registry.json"
    REG.log_code_change([], None, "make an empty registry", path=p)
    monkeypatch.setattr(REG, "registry_path", lambda r=None: p)
    store_path = _write_store(root, root.parent / "outside_store.json")
    with pytest.raises(SystemExit):
        _gate(root, store_path, registry_path=p)


def test_check_only_runs_the_gates_and_writes_nothing(root, plan_file, registry, capsys, monkeypatch):
    store_path = _write_store(root, root.parent / "outside_store.json")
    monkeypatch.setattr(RC, "seal_module", lambda: SEAL)
    rc = RC.main(["--out-root", str(root), "--seed-store", str(store_path), "--check-only"],
                 registry_path=registry, seal=SEAL, **_gate_kwargs(root))
    out = capsys.readouterr().out
    assert rc == 0 and '"mode": "check_only"' in out
    assert not CF.decisions_path(root).exists() and not CF.conf_root(root).exists()
    assert not any(str(s) in out for s in FAKE_SEEDS)


# --------------------------------------------------------------------------------------------- #
# the withheld seeds never appear
# --------------------------------------------------------------------------------------------- #

def test_seed_store_repr_and_public_are_redacted(root):
    store_path = _write_store(root, root.parent / "outside_store.json")
    store = CF.load_seed_store(store_path, root=root, seal=SEAL, prereg_paths=_prereg_paths(root))
    for text in (repr(store), str(store), json.dumps(store.public()), f"{store}"):
        assert not any(str(s) in text for s in FAKE_SEEDS)
    assert store.indices() == (1, 2, 3, 4, 5)
    assert store.seed(1) == min(FAKE_SEEDS) and store.index_of(min(FAKE_SEEDS)) == 1


def test_scrub_replaces_every_seed_and_the_scan_proves_it(root, tmp_path):
    store_path = _write_store(root, root.parent / "outside_store.json")
    store = CF.load_seed_store(store_path, root=root, seal=SEAL, prereg_paths=_prereg_paths(root))
    s = store.seed(3)
    rec = {"job": {"seed": s, "key": f"M2@V5/s{s}"}, "folds": [f"b{s}_7", 12], "nested": {"x": (s, "ok")}}
    clean = CF.scrub(rec, store)
    assert clean == {"job": {"seed": "i3", "key": "M2@V5/si3"}, "folds": ["bi3_7", 12], "nested": {"x": ("i3", "ok")}}
    out = root / "records" / "r.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(clean), encoding="utf-8")
    assert CF.scan_for_seed_leak(store, [out])["ok"] is True
    leaky = root / "records" / "leak.json"
    leaky.write_text(json.dumps({"seed": s}), encoding="utf-8")
    scan = CF.scan_for_seed_leak(store, [root / "records"], extra_text=[f"fitting with {s}"])
    assert scan["ok"] is False and str(leaky) in scan["files_containing_a_withheld_seed"]
    assert "<log text>" in scan["files_containing_a_withheld_seed"]
    assert not any(str(x) in json.dumps(scan) for x in FAKE_SEEDS)      # the scan itself names no seed


def test_a_wrong_seed_count_is_refused(root):
    store_path = _write_store(root, root.parent / "outside_store.json", seeds=[111111, 222222, 333333, 444444])
    with pytest.raises(SystemExit):
        CF.load_seed_store(store_path, root=root, seal=SEAL, prereg_paths=_prereg_paths(root))


# --------------------------------------------------------------------------------------------- #
# the fold builder is seed-dependent
# --------------------------------------------------------------------------------------------- #

def test_the_colouring_is_seed_dependent_and_reproducible():
    """The registered batching (``cell_holdout.greedy_colouring``, the rule the withheld-seed files are built with) must
    depend on the seed -- otherwise the 5 withheld seeds would give 5 identical fold sets and R19 item 4 would be
    vacuous -- and must reproduce for one seed."""
    cells = [(f"M{i}(III)", f"S{j}") for i in range(6) for j in range(4)]
    a = CH.greedy_colouring(cells, np.random.default_rng(FAKE_SEEDS[0]), max_cells_per_batch=4)
    b = CH.greedy_colouring(cells, np.random.default_rng(FAKE_SEEDS[1]), max_cells_per_batch=4)
    a2 = CH.greedy_colouring(cells, np.random.default_rng(FAKE_SEEDS[0]), max_cells_per_batch=4)
    assert [sorted(x) for x in a] == [sorted(x) for x in a2]
    assert [sorted(x) for x in a] != [sorted(x) for x in b]
    assert all(len(batch) <= 4 for batch in a + b)


def test_the_fold_plan_is_one_file_per_seed_index_and_names_no_seed(root):
    plans = RC.build_confirmation_folds(None, root, dry_run=True)
    assert {p.seed_index for p in plans} == {1, 2, 3, 4, 5}
    assert {p.stem for p in plans} == set(RC.BATCHED_STEMS)
    assert not CF.folds_dir(root).exists()      # a dry run builds nothing
    rec = json.dumps([p.record() for p in plans])
    assert "104729" not in rec and not any(str(s) in rec for s in FAKE_SEEDS)


def test_building_the_withheld_seed_folds_refuses_while_unimplemented(root):
    store_path = _write_store(root, root.parent / "outside_store.json")
    store = CF.load_seed_store(store_path, root=root, seal=SEAL, prereg_paths=_prereg_paths(root))
    with pytest.raises(SystemExit, match="NOT IMPLEMENTED"):
        RC.build_confirmation_folds(store, root)


# --------------------------------------------------------------------------------------------- #
# R19 item 4, the seed combination, S1(c), S2
# --------------------------------------------------------------------------------------------- #

def test_item4_needs_five_of_five():
    assert CF.item4({1: 0.2, 2: 0.1, 3: 0.3, 4: 0.05, 5: 0.4})["status"] == "PASS"
    four_positive = CF.item4({1: 0.2, 2: 0.1, 3: -0.3, 4: 0.05, 5: 0.4})
    assert four_positive["status"] == "FAIL" and four_positive["n_positive"] == 4
    missing = CF.item4({1: 0.2, 2: 0.1, 3: 0.3, 4: 0.05})
    assert missing["status"] == "FAIL" and missing["n_seeds_scored"] == 4
    assert CF.item4({1: 0.2, 2: 0.1, 3: 0.3, 4: 0.0, 5: 0.4})["status"] == "FAIL"     # zero is not > 0
    # and the registered rule the value comes from
    assert CF.N_SEEDS_REQUIRED == ET.MIN_SEEDS_POSITIVE_CONFIRMATION == 5


def test_r19_at_confirmation_reads_item4_as_five_of_five(root):
    """``transfer.r19(stage='confirmation')`` and :func:`confirmation.item4` must agree, or ``score_claim`` refuses."""
    claim = CF.Claim(claim_id="C2", label="M2 vs B0 @ V5", family="S1(b)", candidate="M2", comparator="B0",
                     design="V5", margin=0.05)
    units = pd.Index([f"c{i}" for i in range(12)])
    clusters = {"system": pd.Series([f"s{i % 4}" for i in range(12)], index=units),
                "publication_group": pd.Series([f"g{i % 3}" for i in range(12)], index=units)}
    cand = pd.Series(np.linspace(0.2, 0.4, 12), index=units)
    comp = cand + 0.3
    pu = D.PairedUnits(design="V5", candidate="M2", comparator="B0", cand_mae=cand, comp_mae=comp, clusters=clusters,
                       n_rows=48)
    boots = D.bootstraps(pu, name=claim.key)
    res = CF.score_claim(claim, point=pu.delta, bootstraps=boots,
                         seed_deltas={1: 0.3, 2: 0.31, 3: 0.29, 4: 0.3, 5: 0.28},
                         sensitivity_deltas={n: 0.3 for n in ET.REGISTERED_SENSITIVITIES["V5"]})
    assert res["item4"]["status"] == "PASS" and res["r19_verdict"] == "PASS" and res["confirmed"] is True
    res2 = CF.score_claim(claim, point=pu.delta, bootstraps=boots,
                          seed_deltas={1: 0.3, 2: -0.31, 3: 0.29, 4: 0.3, 5: 0.28},
                          sensitivity_deltas={n: 0.3 for n in ET.REGISTERED_SENSITIVITIES["V5"]})
    assert res2["item4"]["status"] == "FAIL" and res2["r19_verdict"] == "FAIL" and res2["confirmed"] is False


def test_the_seed_mean_is_the_macro_of_the_seed_mean(root):
    """The pooling of items 1, 2, 3 and 5 (:data:`g19_run_confirmation.SEED_POOLING`): the mean over seeds of the
    per-seed macro Delta equals the macro of the seed-mean per-unit MAE, so the two readings cannot disagree."""
    units = pd.Index([f"c{i}" for i in range(9)])
    clusters = {"system": pd.Series([f"s{i % 3}" for i in range(9)], index=units),
                "publication_group": pd.Series([f"g{i % 3}" for i in range(9)], index=units)}
    rng = np.random.default_rng(19)
    per_seed = {}
    for i in range(1, 6):
        cand = pd.Series(rng.uniform(0.2, 0.6, 9), index=units)
        per_seed[i] = D.PairedUnits(design="V5", candidate="M2", comparator="B3i", cand_mae=cand,
                                    comp_mae=cand + 0.2 + 0.01 * i, clusters=clusters, n_rows=36)
    pooled = RC.seed_mean_paired_units(per_seed, design="V5", candidate="M2", comparator="B3i")
    assert pooled.delta == pytest.approx(float(np.mean([per_seed[i].delta for i in per_seed])))


def test_s1c_seed_mean_bootstrap_and_verdict():
    by_system = {i: {f"S{k}": 0.06 + 0.001 * i + 0.002 * k for k in range(5)} for i in range(1, 6)}
    st = CF.seed_mean_system_bootstrap(by_system, n_resamples=200)
    assert st["n_seeds"] == 5 and st["n_systems"] == 5 and st["excludes_zero"] is True
    assert st["point"] == pytest.approx(float(np.mean([np.mean(list(v.values())) for v in by_system.values()])))
    res = CF.s1c_confirmation({"HEAVIER": by_system, "B3i-derived": by_system}, logsf_gain={"FLAT": 0.5, "B3i": 0.07},
                              selection_min_delta=-0.0168)
    assert res["verdict"] == "PASS" and res["min_delta"] > CF.GAMMA5 and res["counterweight_pass"] is True
    # the binding yardstick below gamma5 fails the direction half, not the magnitude half
    low = {i: {k: 0.03 for k in by_system[i]} for i in by_system}
    res2 = CF.s1c_confirmation({"HEAVIER": by_system, "B3i-derived": low}, logsf_gain={"FLAT": 0.5, "B3i": 0.07},
                               selection_min_delta=-0.0168)
    assert res2["verdict"] == "FAIL" and res2["binding_yardstick"] == "B3i-derived" and res2["logsf_pass"] is True
    # a seed with the wrong sign fails item 4 inside S1(c)
    mixed = {**by_system, 3: {k: -0.1 for k in by_system[3]}}
    res3 = CF.s1c_confirmation({"HEAVIER": mixed}, logsf_gain={"FLAT": 0.5, "B3i": 0.07}, selection_min_delta=-0.0168)
    assert res3["per_yardstick"]["HEAVIER"]["item4"]["status"] == "FAIL" and res3["verdict"] == "FAIL"


def test_s2a_sign_counting_is_conservative():
    obs = {f"S{k}": (1.0 if k % 2 else -1.0) for k in range(13)}
    pred = dict(obs)
    assert CF.s2a_sign_count(obs, pred)["status"] == "PASS"
    two_wrong = {**pred, "S0": 1.0, "S1": -1.0}
    r = CF.s2a_sign_count(obs, two_wrong)
    assert r["n_sign_agree"] == 11 and r["status"] == "PASS"
    three_wrong = {**two_wrong, "S2": 1.0}
    assert CF.s2a_sign_count(obs, three_wrong)["status"] == "FAIL"
    # a zero or missing predicted median does NOT agree, and a missing system breaks the 13-system count
    zero = {**pred, "S0": 0.0}
    assert CF.s2a_sign_count(obs, zero)["n_sign_agree"] == 12
    short = {k: v for k, v in pred.items() if k != "S12"}
    r2 = CF.s2a_sign_count(obs, short)                      # a system with no prediction does not agree
    assert r2["n_systems_scored"] == 13 and r2["n_sign_agree"] == 12 and r2["status"] == "PASS"
    obs12 = {k: v for k, v in obs.items() if k != "S12"}    # a system missing from BOTH breaks the 13-system count
    assert CF.s2a_sign_count(obs12, short)["status"] == "FAIL"


def test_s2b_and_the_coverage_bands():
    ok = CF.s2b_magnitude(logsf_mae=0.70, flat_logsf_mae=0.80, lookup_logsf_mae=0.75,
                          gain_interval_excludes_zero=True, logd_macro_mae=0.70, v5_lookup_comparator_mae=0.7670)
    assert ok["status"] == "PASS" and ok["gain_over_flat"] == pytest.approx(0.10)
    thin = CF.s2b_magnitude(logsf_mae=0.79, flat_logsf_mae=0.80, lookup_logsf_mae=0.75,
                            gain_interval_excludes_zero=True, logd_macro_mae=0.70, v5_lookup_comparator_mae=0.7670)
    assert thin["status"] == "FAIL"        # 0.01 gain is below the registered 0.02
    undec = CF.s2b_magnitude(logsf_mae=0.70, flat_logsf_mae=0.80, lookup_logsf_mae=0.75,
                             gain_interval_excludes_zero=None, logd_macro_mae=0.70, v5_lookup_comparator_mae=0.7670)
    assert undec["status"] == "UNDECIDED"
    assert CF.coverage_bands({"80": 0.81, "95": 0.93}, CF.S2C_BANDS, min_95=CF.S2C_MIN_95)["status"] == "PASS"
    assert CF.coverage_bands({"80": 0.62, "95": 0.93}, CF.S2C_BANDS, min_95=CF.S2C_MIN_95)["status"] == "FAIL"
    s1d = CF.s1d_calibration({"50": 0.5, "80": 0.8, "95": 0.95},
                             by_category={"IN_DOMAIN": {"80": 0.80, "n_cells": 40},
                                          "UNSUPPORTED": {"80": 0.30, "n_cells": 5}})
    assert s1d["status"] == "PASS"      # the 5-cell category is printed and decides nothing
    assert [c["decides"] for c in s1d["by_category"]] == [True, False]


def test_bh_is_printed_and_decides_nothing():
    adj = CF.bh_adjust({"a": 0.01, "b": 0.04, "c": 0.20})
    assert adj["a"] <= adj["b"] <= adj["c"] and adj["c"] == pytest.approx(0.2)


# --------------------------------------------------------------------------------------------- #
# what was NOT run
# --------------------------------------------------------------------------------------------- #

def test_v6_actinide_deltas_and_the_power_check_are_recorded_not_run(root):
    for block in (CF.V6_ACTINIDE_DELTAS_NOT_RUN, CF.POWER_CHECK_NOT_RUN):
        assert block["status"] == CF.NOT_RUN
        assert "addendum 5" in block["not_run_by"]
        assert block["consequence"]
    assert CF.V6_ACTINIDE_DELTAS_NOT_RUN["cost_hours_serial"] > 0
    assert "UNDECIDED" in CF.POWER_CHECK_NOT_RUN["consequence"] and "never" in CF.POWER_CHECK_NOT_RUN["consequence"]
    body = {"claims": [], "not_run": {"v6_actinide_deltas": CF.V6_ACTINIDE_DELTAS_NOT_RUN,
                                      "power_check": CF.POWER_CHECK_NOT_RUN},
            "seed_store": {"commitment_sha256": "x" * 64, "n_seeds": 5, "verified_against_commitment": True,
                           "disclosure": CF.SEED_DISCLOSURE}}
    p = CF.write_decisions(root, body)
    got = json.loads(p.read_text(encoding="utf-8"))
    assert got["not_run"]["v6_actinide_deltas"]["status"] == CF.NOT_RUN
    md = CF.confirmation_report(got)
    assert "NOT_RUN" in md and "VERIFIED" in md and "VALUES are not printed" in md
    tabs = CF.confirmation_tables(got)
    assert set(tabs) == {"confirmation_claims", "confirmation_r19_items", "confirmation_not_run"}
    assert len(tabs["confirmation_not_run"]) == 2


def test_the_cost_estimate_excludes_what_was_not_run(root, plan_file):
    plan = CF.read_plan(plan_file)
    cost = CF.cost_estimate(CF.enumerate_jobs(plan))
    assert cost["unknown_unit_cost"] == [] and cost["serial_hours"] > 0
    assert cost["wall_hours_2_workers"] == pytest.approx(cost["serial_hours"] / CF.PARALLEL_EFFICIENCY_2_WORKERS)
    assert "v6_actinide_deltas" in cost["not_in_this_estimate"]
    # the item-6 refits are on ONE seed (addendum 1 item 4 / addendum 5 item 1), so seed_index 0 marks them
    assert {j.seed_index for j in CF.enumerate_jobs(plan) if "item 6" in j.purpose} == {0}


# --------------------------------------------------------------------------------------------- #
# the confirmation half and the V6 carve-out, both directions
# --------------------------------------------------------------------------------------------- #

def test_assert_confirmation_rows():
    half = {"r1": "C", "r2": "C", "r3": "S"}
    CF.assert_confirmation_rows(["r1", "r2"], half, "test")
    with pytest.raises(AssertionError, match="outside the confirmation half"):
        CF.assert_confirmation_rows(["r1", "r3"], half, "test")
    with pytest.raises(AssertionError):
        CF.assert_confirmation_rows(["r1", "unknown"], half, "test")


def test_confirmation_scored_ids_mirrors_the_discovery_selector():
    fold = FI.make_fold(design="V5", variant="primary", scheme="exact", fold_id="f1", half="C", seed=None,
                        hidden=["a", "b", "c"], scored=["a", "b", "c"], unit_type="cell", units=["u"],
                        row_unit={r: "u" for r in "abc"}, row_half={"a": "C", "b": "S", "c": "C"})
    assert CF.confirmation_scored_ids(fold) == ("a", "c")
    assert D.selection_scored_ids(fold) == ("b",)


def test_v6_rows_are_scored_here_and_nowhere_else():
    labels = pd.Index(["L0", "L1", "L2", "L3"])
    v6 = pd.Series([True, True, False, False], index=labels)
    CF.assert_v6_scope(labels[:2], v6, "V6", "the single V6 run")        # a V6 job scores V6 rows -- the point of it
    CF.assert_v6_scope(labels[2:], v6, "V5", "a V5 job")
    with pytest.raises(AssertionError, match="carves them out"):
        CF.assert_v6_scope(labels[:3], v6, "V5", "a V5 job")             # nowhere else
    with pytest.raises(AssertionError, match="outside V6_TARGET_ROWS"):
        CF.assert_v6_scope(labels, v6, "V6", "the single V6 run")
    with pytest.raises(AssertionError, match="not in the V6 mask"):
        CF.assert_v6_scope(pd.Index(["LX"]), v6, "V6", "the single V6 run")


def test_prepare_fold_refuses_a_selection_half_fold(root):
    fold = FI.make_fold(design="V5", variant="primary", scheme="exact", fold_id="f1", half="S", seed=None,
                        hidden=["a"], scored=["a"], unit_type="cell", units=["u"], row_unit={"a": "u"},
                        row_half={"a": "S"})
    job = D.JobSpec(kind="fit", arm="M2", design="V5", variant="primary", scheme="exact", seed=1)
    with pytest.raises(AssertionError, match="selection-half fold is never fitted in the confirmation run"):
        RC.prepare_fold(job, fold, 0, object(), root, D.PlanState())


def test_workers_are_capped(root, plan_file):
    with pytest.raises(SystemExit, match="exceeds 2"):
        RC.main(["--out-root", str(root), "--workers", "3", "--dry-run"])


def test_dry_run_reads_no_seed_and_writes_nothing(root, plan_file, capsys):
    rc = RC.main(["--out-root", str(root), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0 and '"mode": "dry_run"' in out
    body = json.loads(out[out.index("{"):])
    assert body["claims"] == ["C1", "C4"] and body["n_folds"] > 0 and body["serial_hours"] > 0
    assert body["not_run"] == {"v6_actinide_deltas": CF.NOT_RUN, "power_check": CF.NOT_RUN}
    assert not CF.decisions_path(root).exists()
    assert not any(str(s) in out for s in FAKE_SEEDS)
    assert [s["stage"] for s in body["run_stages"] if not s["implemented"]]       # the refusal inventory is printed


def test_registry_holds_the_confirmation_rule():
    """The stage exists in the registry's own STAGES and has a code-digest rule (task 1)."""
    assert CF.STAGE in REG.STAGES and CF.STAGE in REG.CODE_DIGEST_SOURCE
    assert REG.stage_code_digest(CF.STAGE) == RC.code_digest()["combined"]
