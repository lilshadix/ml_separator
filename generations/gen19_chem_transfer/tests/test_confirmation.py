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
from gen19ct.evaluation import metrics as EM
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
    assert {p.stem for p in plans} == set(RC.BATCHED_STEMS) | {RC.V6_STEM}
    assert not CF.folds_dir(root).exists()      # a dry run builds nothing
    rec = json.dumps([p.record() for p in plans])
    assert "104729" not in rec and not any(str(s) in rec for s in FAKE_SEEDS)


# --------------------------------------------------------------------------------------------- #
# the withheld-seed fold files, built inside the run (POST-HOC addendum 6 item 2)
# --------------------------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def fold_corpus():
    """The registered builder's corpus, loaded once (it is the real corpus; the builders touch no target value)."""
    return RC.load_fold_corpus()


def test_the_v6_design_hides_exactly_what_section_3_4_says(fold_corpus):
    """Section 3.4 / addendum 6 item 2: 13 systems; per system Pr(III) x S and Nd(III) x S hidden TOGETHER under the
    component-aware state-level rule, so the hidden rows are the Pr and Nd rows -- known state and X(?) -- of S and of
    every component-sharing system, while only the known-state Pr / Nd rows of S itself are SCORED."""
    from gen19ct.chemistry import support_graph as SGx

    folds, stats = RC.build_v6_folds(fold_corpus)
    assert len(folds) == CF.S2A_N_SYSTEMS == 13 and stats["n_systems"] == 13
    fr, v6 = fold_corpus.frame, fold_corpus.v6.to_numpy(dtype=bool)
    ids = fr[FI.ROW_ID].astype(str).to_numpy(dtype=object)
    v6_ids = set(ids[v6])
    for f in folds:
        cells = [tuple(c) for c in f.meta["cells"]]
        assert [c[0] for c in cells] == list(CF.V6_METALS) and cells[0][1] == cells[1][1]
        # the hidden set IS support_graph.hide_cells on the double cell, component-aware
        kept = SGx.hide_cells(fr, cells, component_aware=True)
        assert set(f.hidden_row_ids) == set(ids[~fr.index.isin(kept.index)])
        # every scored row is a V6 target row of THIS system, known state, one of the two metals
        sc = fr[fr[FI.ROW_ID].astype(str).isin(set(f.scored_row_ids))]
        assert set(f.scored_row_ids) <= v6_ids
        assert set(sc[SG.SYSTEM_COL]) == {cells[0][1]}
        assert set(sc[SG.METAL_COL]) <= set(CF.V6_METALS) and not sc[SG.METAL_COL].isna().any()
        # an X(?) row of Pr or Nd under this system is HIDDEN and never scored (section 2)
        unk = fr[fr[SG.METAL_COL].isna() & fr[SG.ELEMENT_COL].isin(["Pr", "Nd"])
                 & (fr[SG.SYSTEM_COL] == cells[0][1])]
        unk_ids = set(unk[FI.ROW_ID].astype(str))
        assert unk_ids <= set(f.hidden_row_ids) and not (unk_ids & set(f.scored_row_ids))
        assert f.half == "NA" and f.seed is None and f.meta["component_aware"] is True
    # the fold ids are file-name safe and the design covers each system once
    assert len({f.fold_id for f in folds}) == 13
    assert all(f.fold_id == D.safe_fold_name(f.fold_id) for f in folds)
    # 209 comparable Pr/Nd row pairs (section 3.4) come from 209 Pr + 209 Nd scored rows
    assert stats["n_scored_rows"] == 2 * 209


def test_the_withheld_seed_colourings_are_seed_dependent_and_derive_their_counts(root, fold_corpus):
    """The per-seed files are built by the registered rule, differ between seeds, and their counts are DERIVED."""
    store_path = _write_store(root, root.parent / "outside_store.json")
    store = CF.load_seed_store(store_path, root=root, seal=SEAL, prereg_paths=_prereg_paths(root))
    plans = RC.build_confirmation_folds(store, root, stems=("V5__primary__batched_max4",), corpus=fold_corpus,
                                       with_v6=True)
    by_stem = {}
    for p in plans:
        by_stem.setdefault(p.stem, {})[p.seed_index] = p
    v5 = by_stem["V5__primary__batched_max4"]
    assert set(v5) == {1, 2, 3, 4, 5}
    # seed dependent: the design hashes differ between seeds (R19 item 4 would be vacuous otherwise)
    assert len({p.design_hash for p in v5.values()}) == 5
    # V6 is seed-INDEPENDENT as a design: the same 13 folds under every seed index
    assert len({p.design_hash for p in by_stem[RC.V6_STEM].values()}) == 1
    # counts are derived, and the plan's figure is only compared
    assert all(p.n_folds > 0 and p.n_scored_units > 0 for p in v5.values())
    assert all(abs(p.n_folds_confirmation_half - 27) <= 2 for p in v5.values())
    assert by_stem[RC.V6_STEM][1].n_folds == 13
    # no file, directory or recorded field names a seed
    scan = CF.scan_for_seed_leak(store, [CF.folds_dir(root)])
    assert scan["ok"], scan
    assert json.dumps([p.record() for p in plans]).count("104729") == 0
    # every written design reads back with its recorded hashes
    for p in plans:
        folds = FI.read_design(Path(p.path))
        assert FI.design_hash(folds) == p.design_hash and len(folds) == p.n_folds
    # and the V6 carve-out holds in both directions on the FILES
    v6_ids = set(fold_corpus.frame.loc[fold_corpus.v6.to_numpy(dtype=bool), FI.ROW_ID].astype(str))
    for p in plans:
        for f in FI.read_design(Path(p.path)):
            inside = set(f.scored_row_ids) & v6_ids
            assert (inside == set(f.scored_row_ids)) if CF.is_v6(f.design) else (not inside)


def test_a_fake_store_is_refused(root):
    """The gate's (d): a store whose seeds do not hash to the section 15 commitment never reaches a fit."""
    bad = root.parent / "not_the_committed_store.json"
    bad.write_text(json.dumps({"schema": "gen19.confirmation_seeds.v1", "n": 5, "salt": "00" * 32,
                               "seeds": [111111, 222222, 333333, 444444, 555555]}), encoding="utf-8")
    with pytest.raises(SystemExit, match="does not verify"):
        CF.load_seed_store(bad, root=root, seal=SEAL, prereg_paths=_prereg_paths(root))


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
    # the dry run NAMES the stage that is not written, so the plan is never read as a plan that could start
    assert [s["stage"] for s in body["run_stages"] if not s["implemented"]] == ["v6_arm_fitting"]
    assert {s["stage"] for s in body["run_stages"]} >= {"withheld_seed_fold_files", "fit_loop", "writers"}


def test_the_writers_reveal_the_seeds_only_in_CONFIRMATION_md_and_only_at_the_end(root, tmp_path):
    """POST-HOC addendum 6 item 4: nothing written before or during the run names a seed; ``decisions/CONFIRMATION.md``
    is written LAST and is the one artefact that names all five, beside the --verify-seeds verdict."""
    store_path = _write_store(root, root.parent / "outside_store.json")
    store = CF.load_seed_store(store_path, root=root, seal=SEAL, prereg_paths=_prereg_paths(root))
    body = {"claims": [], "not_run": {}, "seed_store": store.public(), "claim_ids": []}
    # before: the report written without the store names no seed, and the scan is clean
    pre_md = CF.confirmation_report(body)
    assert not any(str(s) in pre_md for s in FAKE_SEEDS)
    written = RC.write_outputs(root, body, store=store, seal=SEAL)
    rp = CF.report_path(root)
    assert rp in written and written[-1] == rp                      # LAST
    text = rp.read_text(encoding="utf-8")
    assert all(str(store.seed(i)) in text for i in store.indices())  # all five revealed
    assert "VERIFIED" in text
    # every other artefact still names none, and the scan says exactly that
    others = [p for p in written if p != rp]
    assert CF.scan_for_seed_leak(store, others)["ok"]
    post = CF.scan_for_seed_leak(store, written, allow=[rp], require=[rp])
    assert post["ok"] and post["revelation_incomplete"] == []
    # and a run that never wrote the report fails the revelation check
    missing = CF.scan_for_seed_leak(store, others, allow=[rp], require=[rp])
    assert missing["ok"] is False and missing["revelation_incomplete"] == [str(rp)]


def test_the_lock_allows_only_resume_and_never_a_second_scoring(root, plan_file, registry):
    """Section 15's "nothing is re-run" as the lock, with --resume the only second invocation."""
    CF.write_decisions(root, {"code_digest": "x" * 64, "claim_ids": ["C1", "C4"]})
    first = CF.lock_verdict(root, resume=False, code_digest="x" * 64, claim_ids=["C1", "C4"])
    assert first["ok"] is False and first["mode"] == "already_run"
    ok = CF.lock_verdict(root, resume=True, code_digest="x" * 64, claim_ids=["C1", "C4"])
    assert ok["ok"] is True and ok["mode"] == "resume" and ok["may_only"] == "complete unfitted folds"
    grew = CF.lock_verdict(root, resume=True, code_digest="x" * 64, claim_ids=["C1", "C4", "C9"])
    assert grew["ok"] is False and grew["new_claims"] == ["C9"]
    newcode = CF.lock_verdict(root, resume=True, code_digest="y" * 64, claim_ids=["C1"])
    assert newcode["ok"] is False and newcode["mode"] == "resume_refused_code_changed"


def test_the_s1c_yardstick_artefact_is_diagnosed_not_refitted():
    """The pre-existing artefact verifies against the SUPERSEDED two-addenda 'scorer' entry (its own stage); it fails
    only when a reader routes it through registry.job_stage, which is written for runner fold records."""
    out = RC.s1c_artefact_diagnosis()
    assert out["action"] == "diagnosed, NOT refitted"
    if out["artefacts"]:
        for r in out["artefacts"]:
            assert r["registry_stage_named_by_the_record"] == "scorer"
            assert r["stage_registry_job_stage_would_use"] == "discovery"
            assert r["verifies_against_its_own_stage"] is True and r["matched_entry"] == "superseded"
            assert r["verifies_when_routed_through_job_stage"] is False
            assert r["mismatches_when_routed"] == ["prereg_addenda_sha256", "prereg_n_addenda"]
        assert out["status"] == "EXPLAINED"


def test_the_seed_mean_statistics_on_hand_computed_values():
    """Addendum 6 item 1 by hand: the statistic is the mean over the 5 seeds of the per-seed macro value."""
    per_seed = {1: {"A": 0.10, "B": 0.20}, 2: {"A": 0.20, "B": 0.30}, 3: {"A": 0.00, "B": 0.10},
                4: {"A": 0.30, "B": 0.40}, 5: {"A": 0.40, "B": 0.50}}
    st = CF.seed_mean_system_bootstrap(per_seed)
    assert st["per_seed"] == pytest.approx({1: 0.15, 2: 0.25, 3: 0.05, 4: 0.35, 5: 0.45})
    assert abs(st["point"] - 0.25) < 1e-12                    # mean of the five per-seed macros
    assert st["n_systems"] == 2 and st["n_seeds"] == 5 and st["n_resamples"] == 10000
    assert st["bootstrap_seed"] == 19                          # section 8
    assert CF.item4(st["per_seed"])["status"] == "PASS"
    # one seed at zero is not > 0, so item 4 is 4 of 5 and FAILS
    flip = {**per_seed, 3: {"A": -0.10, "B": 0.10}}
    st2 = CF.seed_mean_system_bootstrap(flip)
    assert st2["per_seed"][3] == 0.0 and CF.item4(st2["per_seed"])["status"] == "FAIL"


def test_s2a_counts_signs_over_the_thirteen_systems():
    obs = {f"S{i}": (1.0 if i % 2 else -1.0) for i in range(13)}
    pred = dict(obs)
    assert CF.s2a_sign_count(obs, pred)["status"] == "PASS"
    two_wrong = {**pred, "S0": 1.0, "S1": -1.0}
    r = CF.s2a_sign_count(obs, two_wrong)
    assert r["n_sign_agree"] == 11 and r["status"] == "PASS"       # exactly 11 of 13 passes
    three_wrong = {**two_wrong, "S2": 1.0}
    assert CF.s2a_sign_count(obs, three_wrong)["n_sign_agree"] == 10
    assert CF.s2a_sign_count(obs, three_wrong)["status"] == "FAIL"
    # a zero median does NOT agree (the conservative side of a sign count)
    assert CF.s2a_sign_count(obs, {**pred, "S0": 0.0})["n_sign_agree"] == 12
    # an observed median that is absent gives NaN on that system, which does not agree either; the system set is the
    # UNION of the two sides, so a system missing from one side is still counted and still scored as 13
    part = CF.s2a_sign_count({k: v for k, v in list(obs.items())[:12]}, pred)
    assert part["n_systems_scored"] == 13 and part["n_sign_agree"] == 12
    # fewer systems on BOTH sides is not the registered 13 and fails whatever the signs say
    few = {k: v for k, v in list(obs.items())[:12]}
    assert CF.s2a_sign_count(few, few)["status"] == "FAIL"


def test_registry_holds_the_confirmation_rule():
    """The stage exists in the registry's own STAGES and has a code-digest rule (task 1)."""
    assert CF.STAGE in REG.STAGES and CF.STAGE in REG.CODE_DIGEST_SOURCE
    assert REG.stage_code_digest(CF.STAGE) == RC.code_digest()["combined"]


# --------------------------------------------------------------------------------------------- #
# TASK X: the confirmation-half fold selector, the V6 leg, the item 6 reduced set, S2
# --------------------------------------------------------------------------------------------- #

def _fold(fold_id: str, half: str, rows: dict[str, str], design: str = "V5", variant: str = "primary",
          scheme: str = "exact", seed=None):
    return FI.make_fold(design=design, variant=variant, scheme=scheme, fold_id=fold_id, half=half, seed=seed,
                        hidden=list(rows), scored=list(rows), unit_type="cell", units=["u"],
                        row_unit={r: "u" for r in rows}, row_half=dict(rows))


def test_the_confirmation_selector_is_the_mirror_of_discoverys():
    """TASK X E1. ``discovery.fittable_folds`` drops every fold whose ``half`` is the confirmation half and keeps only
    folds with a SELECTION-half scored row, so on this half it returns either nothing or exactly the folds this run may
    never fit; the confirmation selector returns the confirmation-half folds and no other."""
    c = _fold("f_c", "C", {"a": "C", "b": "C"})
    s = _fold("f_s", "S", {"c": "S"})
    mixed = _fold("f_m", "C", {"d": "C", "e": "S"})
    job = D.JobSpec(kind="fit", arm="M2", design="V5", variant="primary", scheme="exact", seed=7)
    folds = [c, s, mixed]
    assert [f.fold_id for f, _ in D.fittable_folds(job, folds, [])] == ["f_s"]
    got = CF.confirmation_fittable_folds(job, folds, [])
    assert [f.fold_id for f, _ in got] == ["f_c", "f_m"]
    assert CF.fit_scored_ids(mixed) == ("d",)                       # only the confirmation-half row
    # a fold whose confirmation-half rows are all acidic co-extractant rows is not fittable (section 2)
    assert CF.confirmation_fittable_folds(job, [c], ["a", "b"]) == []
    # the ordinals are discovery's: a property of the design file and the run seed, not of the half
    ords = D.fold_ordinals(folds, 7)
    assert {f.fold_id: k for f, k in got} == {"f_c": ords["f_c"], "f_m": ords["f_m"]}


def test_a_v6_fold_is_selectable_and_carries_no_half():
    """TASK X E2. A V6 fold carries ``half='NA'`` and ``row_half`` all ``NA`` (READINGS['v6_half']), so reading the
    confirmation half would return nothing and the whole V6 leg would be unreachable."""
    f = FI.make_fold(design="V6", variant="prnd", scheme="exact", fold_id="v6_prnd__x", half="NA", seed=None,
                     hidden=["a", "b", "c"], scored=["a", "b"], unit_type="system", units=["S"],
                     row_unit={"a": "c1", "b": "c2", "c": "h"}, row_half={"a": "NA", "b": "NA", "c": "NA"})
    assert CF.confirmation_scored_ids(f) == ()
    assert CF.fit_scored_ids(f) == ("a", "b")
    job = D.JobSpec(kind="fit", arm="M2", design="V6", variant="prnd", scheme="exact", seed=7)
    assert D.fittable_folds(job, [f], []) == []
    assert len(CF.confirmation_fittable_folds(job, [f], [])) == 1
    assert "V6" not in D.HALF_TABLE                                 # so half_by_id['V6'] would be a KeyError


def test_the_item6_pass_pins_the_public_seed_on_a_multi_seed_file(tmp_path):
    """TASK X E3. The registered ``batched_max4`` files hold all five discovery seeds' colourings; the item 6 refit pass
    reads them and must name the seed, which is derived from the FILE, not assumed."""
    class _C:
        def __init__(self, folds):
            self._f = folds
        def folds(self, stem):
            return self._f

    multi = [_fold("s104729_C_b1", "C", {"a": "C"}, scheme="batched_max4", variant="strict", seed=104729),
             _fold("s130363_C_b1", "C", {"a": "C"}, scheme="batched_max4", variant="strict", seed=130363)]
    job = CF.ConfJob("R19 item 6 refits", "M2", "V5", "V5__strict__batched_max4", 0, 8)
    spec = RC.fold_seeded_spec(job, CF.ITEM6_REFIT_SEED, _C(multi))
    assert spec.fold_seed == CF.ITEM6_REFIT_SEED == 104729
    assert [f.fold_id for f, _ in CF.confirmation_fittable_folds(spec, multi, [])] == ["s104729_C_b1"]
    # without it the selector refuses rather than guessing -- which is what the unfixed runner hit
    bare = RC.job_spec_of(job, CF.ITEM6_REFIT_SEED)
    with pytest.raises(ValueError, match="give fold_seed"):
        CF.confirmation_fittable_folds(bare, multi, [])
    # a seed-free file (a withheld-seed colouring, or a registered exact design) needs no fold_seed
    free = [_fold("f1", "C", {"a": "C"})]
    jb = CF.ConfJob("comparators", "B3i", "V5", "V5__primary__exact", 3, 111)
    assert RC.fold_seeded_spec(jb, 999983, _C(free)).fold_seed is None


def test_a_closed_form_comparator_is_a_comparator_intervals_job():
    """TASK X E1/E3. ``runner_for`` looks a closed-form arm up under ``comparator:<arm>``; a ``fit`` job for B3i raises
    ``KeyError('no runner for B3i')`` -- uncaught by the fit loop, which catches only guard failures."""
    rd = RC.runner_module()
    runners = rd.default_runners()
    for arm in ("B3i", "B0", "B3x"):
        spec = RC.job_spec_of(CF.ConfJob("p", arm, "V5", "V5__primary__exact", 1, 111), 999983)
        assert spec.kind == "comparator_intervals"
        assert rd.runner_for(spec, runners) is not None
    for arm in ("M1", "M2", "B6", "B8"):
        spec = RC.job_spec_of(CF.ConfJob("p", arm, "V5", "V5__primary__exact", 1, 111), 999983)
        assert spec.kind == "fit"
        assert rd.runner_for(spec, runners) is not None


def test_item6_is_decided_on_the_reduced_set_and_never_fails_on_a_refit_that_was_not_run():
    """TASK X E4. ``ET.r19``'s own item-6 loop reads every registered name and treats any string other than
    ``UNTESTABLE`` as a FAILURE, so a sensitivity addendum 1 item 4 does not run would FAIL every claim before any data
    was seen.  ``score_claim`` rebuilds item 6 with ``discovery.reduced_item6``, exactly as discovery and H3 do."""
    units = [f"u{i}" for i in range(12)]
    cand = pd.Series(np.linspace(0.2, 0.4, 12), index=units)
    comp = pd.Series(cand.to_numpy() + 0.5, index=units)
    clusters = {u: pd.Series([f"cl{i % 4}" for i in range(12)], index=units)
                for u in ET.REGISTERED_CLUSTER_UNITS["V5"]}
    pu = D.PairedUnits(design="V5", candidate="M2", comparator="B3i", cand_mae=cand, comp_mae=comp,
                       clusters=clusters, n_rows=120)
    claim = CF.Claim(claim_id="C1", label="M2 vs B3i @ V5", family="primary", candidate="M2", comparator="B3i",
                     design="V5", margin=0.1057, discovery_point=0.26)
    boots = D.bootstraps(pu, name="C1")
    filters = [n for n in D.SCORING_FILTER_SENSITIVITIES if n in ET.REGISTERED_SENSITIVITIES["V5"]]
    reduced = filters + ["strict_setting", "HNO3_only_cells"]
    sens = {n: 0.4 for n in reduced}
    for n in D.LEARNED_REFITS_NOT_RUN["V5"]:
        sens[n] = ET.UNTESTABLE
    kw = dict(point=pu.delta, bootstraps=boots, seed_deltas={i: 0.4 for i in range(1, 6)})
    # without the reduced set the registered loop FAILS on the refits addendum 1 item 4 does not run
    bad = {**sens, **{n: D.ITEM6_NOT_EVALUATED for n in D.LEARNED_REFITS_NOT_RUN["V5"]}}
    assert CF.score_claim(claim, sensitivity_deltas=bad, **kw)["r19_verdict"] == "FAIL"
    ok = CF.score_claim(claim, sensitivity_deltas=sens, reduced_sensitivities=reduced, **kw)
    assert next(i["status"] for i in ok["items"] if i["item"] == 6) == "PASS"
    assert ok["r19_verdict"] == "PASS" and ok["sensitivity_set"] == D.ADDENDUM_LABEL
    # a member of addendum 1 item 4's OWN reduced set that could not be evaluated is NOT_EVALUATED, never PASS
    part = CF.score_claim(claim, sensitivity_deltas={**sens, "strict_setting": ET.UNTESTABLE},
                          reduced_sensitivities=[n for n in reduced if n != "strict_setting"],
                          sensitivities_not_run_extra={"strict_setting": "UNTESTABLE: no seed-104729 record set"}, **kw)
    assert next(i["status"] for i in part["items"] if i["item"] == 6) == D.ITEM6_NOT_EVALUATED
    assert part["r19_verdict"] == "UNDECIDED" and not part["confirmed"]
    # and a genuinely negative sensitivity still FAILS
    neg = CF.score_claim(claim, sensitivity_deltas={**sens, filters[0]: -0.1}, reduced_sensitivities=reduced, **kw)
    assert next(i["status"] for i in neg["items"] if i["item"] == 6) == "FAIL" and neg["r19_verdict"] == "FAIL"


def test_the_scoring_filter_names_are_apply_scoring_filters_own_vocabulary():
    """TASK X E5. ``transfer.scoring_filters`` returns filter LABELS (``acid_grid_rows_excluded_scoring``) and omits
    ``non_DGA_stratum``; ``discovery.apply_scoring_filter`` speaks SENSITIVITY names."""
    fr = pd.DataFrame({"acid_grid_flag": [True, False], "dga_stratum": ["non_DGA", "diglycolamide"],
                       "censoring_candidate": [False, True], "wildcard_copy_partner_in_training": [False, False]})
    for name in D.SCORING_FILTER_SENSITIVITIES:
        assert len(D.apply_scoring_filter(fr, name)) >= 1
    for label in ET.scoring_filters("V5"):
        if label not in D.SCORING_FILTER_SENSITIVITIES:
            with pytest.raises(ValueError, match="unknown scoring filter"):
                D.apply_scoring_filter(fr, label)
    assert "non_DGA_stratum" in ET.REGISTERED_SENSITIVITIES["V5"]
    assert "non_DGA_stratum" not in ET.scoring_filters("V5")
    assert "non_DGA_stratum" in D.SCORING_FILTER_SENSITIVITIES


def test_the_refit_sensitivity_takes_both_arms_from_the_settings_own_design():
    """TASK X E4. A setting's Delta needs the candidate AND the comparator on that setting (section 5
    ``comparator_folds``: the closed form on the setting's EXACT folds), and the inventory must fit both."""
    assert RC.REFIT_STEMS["V5"] == {"strict_setting": ("V5__strict__batched_max4", "V5__strict__exact"),
                                    "HNO3_only_cells": ("V5__hno3_only__batched_max4", "V5__hno3_only__exact")}
    plan = CF.read_plan(SCRIPTS.parent / "decisions" / "CONFIRMATION_PLAN.md")
    jobs = CF.enumerate_jobs(plan)
    at0 = {(j.arm, j.stem) for j in jobs if j.seed_index == 0}
    for _, comp_stem in RC.REFIT_STEMS["V5"].values():
        assert ("B3i", comp_stem) in at0 and ("B0", comp_stem) in at0 and ("B6", comp_stem) in at0
    for stem in ("V5__strict__batched_max4", "V5__hno3_only__batched_max4"):
        assert ("M1", stem) in at0 and ("M2", stem) in at0
    assert not any(not np.isfinite(j.unit_seconds) for j in jobs)


def test_s2a_needs_every_yardstick_in_both_strata():
    """TASK X E6. S2(a)'s second half is ``min_Y Delta_Y >= 0.05`` over {HEAVIER, B3x-derived, B3i-derived, B8},
    pooled AND in the HNO3 pairs; a yardstick that was not scored leaves it UNDECIDED, never passed."""
    good = {y: {i: {f"S{k}": 0.2 for k in range(13)} for i in range(1, 6)} for y in CF.S2A_YARDSTICKS}
    r = CF.s2a_direction(good)
    assert r["status"] == "PASS" and r["min_delta"] == pytest.approx(0.2)
    assert CF.s2a_direction({k: v for k, v in good.items() if k != "B8"})["status"] == CF.UNDECIDED
    low = {**good, "B3i_derived": {i: {f"S{k}": 0.01 for k in range(13)} for i in range(1, 6)}}
    r2 = CF.s2a_direction(low)
    assert r2["status"] == "FAIL" and r2["binding_yardstick"] == "B3i_derived"
    # one seed negative breaks the 5-of-5 rule even when the seed mean clears the margin
    one_bad = {**good, "HEAVIER": {i: {f"S{k}": (0.6 if i > 1 else -0.2) for k in range(13)} for i in range(1, 6)}}
    assert CF.s2a_direction(one_bad)["status"] == "FAIL"
    assert CF.S2A_STRATA == ("pooled", "HNO3")


def test_s2c_reports_both_interval_readings_and_says_when_they_disagree():
    """TASK X E6. Section 9 S2(c) fixes the bands but not the construction of a predicted-logSF interval; both are
    computed, the primary decides, and the record says whether the choice mattered."""
    r = CF.s2c_coverage({"interval_arithmetic": {"80": 0.80, "95": 0.92},
                         "quadrature": {"80": 0.62, "95": 0.80}})
    assert r["status"] == "PASS" and r["readings_disagree"]
    assert r["statuses_by_reading"] == {"interval_arithmetic": "PASS", "quadrature": "FAIL"}
    assert CF.s2c_coverage({"quadrature": {"80": 0.8, "95": 0.9}})["status"] == CF.NOT_EVALUATED
    assert CF.S2C_PRIMARY_READING == "interval_arithmetic"


def test_a_v6_pair_interval_is_the_interval_of_the_difference():
    """TASK X E6. ``interval_arithmetic`` is ``[lo_a - hi_b, hi_a - lo_b]``; ``quadrature`` is the root-sum-square of the
    half-widths about the predicted difference, which is never wider."""
    frame = pd.DataFrame({"lo80": [0.0, 1.0], "hi80": [2.0, 3.0], EM.PRED_COL: [1.0, 2.0]}, index=["A", "B"])
    pairs = pd.DataFrame({"idx_a": ["A"], "idx_b": ["B"]})
    iv = RC.v6_logsf_intervals(frame, pairs, "80")
    assert iv["interval_arithmetic"]["lo"].iloc[0] == pytest.approx(-3.0)
    assert iv["interval_arithmetic"]["hi"].iloc[0] == pytest.approx(1.0)
    assert iv["quadrature"]["lo"].iloc[0] == pytest.approx(-1.0 - np.sqrt(2.0))
    assert iv["quadrature"]["hi"].iloc[0] == pytest.approx(-1.0 + np.sqrt(2.0))
    assert not RC.v6_logsf_intervals(frame.drop(columns=["lo80"]), pairs, "80")


def test_the_run_names_the_one_stage_it_does_not_implement():
    """TASK X E2/E6. V6 runs ONCE, so its fitting path cannot be rehearsed; while the arms have no V6 branch the runner
    refuses BEFORE the lock, naming what is missing, rather than producing a number from code that does not exist."""
    todo = [n for n, ok, _ in RC.RUN_STAGES if not ok]
    assert todo == ["v6_arm_fitting"]
    rd = RC.runner_module()
    job = D.JobSpec(kind="fit", arm="M2", design="V6", variant="prnd", scheme="exact", seed=7)
    assert not rd.v5_family(job)                       # the measured cause, not an assumption
    assert rd.inner_design_signature(job) is None
    text = RC.refusal("v6_arm_fitting")
    assert "v5_family" in text and "runs ONCE" in text and "has NOT been spent" in text


def test_a_non_guard_error_in_the_fit_loop_is_redacted(root, monkeypatch):
    """TASK X E2. A guard failure is INCOMPLETE_GUARD_FAILURE; anything else is a defect and stops the run -- re-raised
    with the message SCRUBBED and ``from None``, because ``job.key`` carries the run seed and a traceback must never
    print a withheld seed (addendum 6 item 4)."""
    store_path = _write_store(root, root.parent / "outside_store.json")
    store = CF.load_seed_store(store_path, root=root, seal=SEAL, prereg_paths=_prereg_paths(root))

    class _C:
        coext_ids: list = []
        def __init__(self):
            self._folds: dict = {}
            self.guard_cache: dict = {}
        def folds(self, stem):
            return [_fold("f_c", "C", {"a": "C"})]

    job = CF.ConfJob("p", "M2", "V5", "V5__primary__batched_max4", 1, 1)
    seed = store.seed(1)

    def boom(*a, **k):
        raise ValueError(f"no confirmation-half scored row in fit:M2:V5__primary_batched_max4:s{seed}")

    monkeypatch.setattr(RC, "run_fold", boom)
    with pytest.raises(CF.RedactedRunError) as exc:
        RC.fit_loop([job], store, root, corpus=_C(), state=D.PlanState(), runners={}, code="c", prereg={})
    msg = str(exc.value)
    assert str(seed) not in msg and "i1" in msg and exc.value.__cause__ is None
    assert not any(str(s) in msg for s in FAKE_SEEDS)

    def guard(*a, **k):
        raise AssertionError(f"a scored row is not hidden (s{seed})")

    monkeypatch.setattr(RC, "run_fold", guard)
    led = RC.fit_loop([job], store, root, corpus=_C(), state=D.PlanState(), runners={}, code="c", prereg={})
    assert led.errors and led.errors[0]["status"] == CF.INCOMPLETE_GUARD_FAILURE
    assert str(seed) not in json.dumps(led.record()) + json.dumps(led.errors)


def _v6_pred_and_attrs(n_systems: int = 13):
    """A synthetic V6 prediction set and its attributes: 13 systems, 3 condition groups, Pr(III) and Nd(III)."""
    rows, preds = [], []
    rng = np.random.default_rng(3)
    for s in range(n_systems):
        system, fold = f"SYS{s}", f"v6_prnd__{s:03d}"
        acid = "HNO3" if s % 2 == 0 else "HCl"
        for g in range(3):
            for metal, off in (("Pr(III)", 0.0), ("Nd(III)", 0.6)):
                rid, y = f"r_{s}_{g}_{metal[:2]}", 1.0 + off + 0.1 * g
                rows.append({"row_id": rid, EM.SYSTEM_COL: system, EM.PUB_GROUP_COL: f"P{s}",
                             EM.CONDITION_KEY_COL: f"ck{g}", EM.METAL_STATE_COL: metal, EM.Y_COL: y,
                             "acid_stratum": acid})
                preds.append({"row_id": rid, "fold_id": fold, "mean_logD": y + rng.normal(0, 0.05),
                              "lo80": y - 0.5, "hi80": y + 0.5, "lo95": y - 0.9, "hi95": y + 0.9})
    attrs = pd.DataFrame(rows).set_index("row_id", drop=False)
    return pd.DataFrame(preds), attrs, pd.Series(True, index=attrs.index)


def test_the_registered_pair_guard_refuses_every_v6_pair_so_the_v6_run_has_its_own():
    """TASK X E6. ``transfer.guard_scored_pairs`` ends in ``registered.assert_not_scored``, which refuses ANY
    V6_TARGET_ROWS member -- correct everywhere else, fatal to the one run section 3.4 registers.  The V6 guard makes
    every one of its checks and replaces the carve-out check with the STRONGER two-sided one."""
    pred, attrs, v6 = _v6_pred_and_attrs()
    cin = RC.v6_pair_inputs(pred, attrs, None, v6=v6, what="test")
    pairs, folds, mask = cin["pairs"], cin["folds"], cin["v6_mask"]
    assert len(pairs) == 39 and pairs[EM.SYSTEM_COL].nunique() == 13
    with pytest.raises(AssertionError, match="V6_TARGET_ROWS row"):
        ET.guard_scored_pairs(pairs, folds=folds, v6_mask=mask, design=CF.V6_DESIGN, what="the registered guard")
    assert len(RC.v6_guard_scored_pairs(pairs, folds=folds, v6_mask=mask, what="test")) == 78
    # two-sided: a member that is NOT a V6 row is refused too
    off = mask.copy()
    off.iloc[0] = False
    with pytest.raises(AssertionError, match="outside V6_TARGET_ROWS"):
        RC.v6_guard_scored_pairs(pairs, folds=folds, v6_mask=off, what="test")
    # a member predicted in another fold than the pair's is refused
    bad = pairs.copy()
    bad.loc[bad.index[0], "fold"] = "v6_prnd__999"
    with pytest.raises(AssertionError, match="predicted in another fold"):
        RC.v6_guard_scored_pairs(bad, folds=folds, v6_mask=mask, what="test")


def test_the_confirmation_scoring_frame_sets_the_metric_column():
    """TASK X. Discovery's ``scoring_frame`` sets ``metrics.PRED_COL``; the confirmation one did not, so
    ``metrics.design_per_unit_table`` and every pair reader raised ``KeyError('pred')`` -- S1(d), S1(e) and all of S2."""
    pred, attrs, v6 = _v6_pred_and_attrs()
    fr = CF.confirmation_scoring_frame(pred, attrs, design=CF.V6_DESIGN, v6_mask=v6, what="test")
    assert EM.PRED_COL in fr.columns
    assert np.allclose(fr[EM.PRED_COL].to_numpy(dtype=float), fr["mean_logD"].to_numpy(dtype=float))
    unit = EM.design_per_unit_table(fr, CF.V6_DESIGN, v6_mask=None)      # the registered V6 escape
    assert len(unit) == 13 and np.isfinite(unit["mae"]).all()


def test_s2_assembles_every_part_of_section_9_s2(monkeypatch):
    """TASK X E6. The whole S2 layer on a synthetic V6 record set: S2(a)'s sign count AND its paired yardstick deltas in
    both strata, S2(b)'s lookup and log D terms and the 13-system gain bootstrap, S2(c) under both readings."""
    from gen19ct.evaluation import pairs as EP

    pred, attrs, v6 = _v6_pred_and_attrs()
    cin = RC.v6_pair_inputs(pred, attrs, None, v6=v6, what="test")
    pairs, strata = cin["pairs"], RC.v6_pair_strata(cin["pairs"])
    assert strata["pooled"].sum() == 39 and strata[RC.HNO3].sum() == 21
    logd = EM.design_per_unit_table(cin["frame"].assign(**{EM.PRED_COL: cin["logd"].to_numpy()}),
                                    CF.V6_DESIGN, v6_mask=None)
    systems = sorted(set(pairs[EM.SYSTEM_COL].astype(str)))
    block = {
        "n_rows": len(cin["frame"]), "n_pairs": len(pairs),
        "n_pairs_by_stratum": {k: int(np.asarray(m).sum()) for k, m in strata.items()},
        "observed_median_by_system": {str(k): float(v) for k, v in
                                      pairs.groupby(pairs[EM.SYSTEM_COL].astype(str))["logsf_obs"].median().items()},
        "predicted_median_by_system": {str(k): float(v) for k, v in
                                       cin["logsf"].groupby(pairs[EM.SYSTEM_COL].astype(str).to_numpy()).median().items()},
        "direction_delta_by_system": {st: {y: {s: 0.2 for s in systems} for y in CF.S2A_YARDSTICKS}
                                      for st in CF.S2A_STRATA},
        "direction_detail": {},
        "logsf_mae_by_system": {st: {str(k): float(x) for k, x in
                                     RC.v6_logsf_mae_by_system(pairs, cin["logsf"], keep=m).items()}
                                for st, m in strata.items()},
        "flat_logsf_mae_by_system": {st: {str(k): float(x) for k, x in
                                          RC.v6_logsf_mae_by_system(pairs, EP.flat_logsf(pairs), keep=m).items()}
                                     for st, m in strata.items()},
        "lookup_logsf_mae_by_system": {st: {str(k): float(x) + 0.3 for k, x in
                                            RC.v6_logsf_mae_by_system(pairs, cin["logsf"], keep=m).items()}
                                       for st, m in strata.items()},
        "logd_mae_by_system": {str(k): float(x) for k, x in logd["mae"].items()},
        "lookup_logd_mae_by_system": {str(k): float(x) + 0.4 for k, x in logd["mae"].items()},
        "logsf_interval_coverage": {r: {"80": 0.80, "95": 0.95} for r in CF.S2C_INTERVAL_READINGS},
        "yardsticks_scored": list(CF.S2A_YARDSTICKS), "yardsticks_not_scored": [],
    }

    class _Store:
        digest = "0" * 64
        def indices(self):
            return (1, 2, 3, 4, 5)

    class _Corpus:
        def __init__(self):
            self.v6 = v6
            self.frame = attrs.assign(**{FI.ROW_ID: attrs.index})

    monkeypatch.setattr(RC, "s2_seed_block", lambda *a, **k: (block, []))
    out = RC.s2_assembly(None, attrs, _Corpus(), store=_Store())
    assert out["status"] == "COMPLETE" and out["verdict"] == "PASS"
    assert out["s2a"]["sign"]["n_sign_agree"] == 13
    assert set(out["s2a"]["direction"]) == set(CF.S2A_STRATA)
    assert all(d["status"] == "PASS" for d in out["s2a"]["direction"].values())
    assert out["s2b"]["gain_over_flat"] > CF.S2B_MARGIN and out["s2b"]["gain_interval_excludes_zero"]
    assert out["s2b"]["at_most_lookup"] and out["s2b"]["logd_at_most_lookup"]
    assert out["s2b"]["averaging_unit"] == list(EM.REGISTERED_UNIT_COLS[CF.V6_DESIGN]) == ["extractant_system_key"]
    assert RC.HNO3 in out["s2b"]["by_stratum"] and "pooled" in out["s2b"]["by_stratum"]
    assert out["s2c"]["status"] == "PASS" and not out["s2c"]["readings_disagree"]
    json.dumps(out, default=str)                                   # the decisions writer must be able to serialise it
    # one missing yardstick: S2 is UNDECIDED, never passed
    monkeypatch.setattr(RC, "s2_seed_block", lambda *a, **k: (
        {**block, "direction_delta_by_system": {st: {y: v for y, v in block["direction_delta_by_system"][st].items()
                                                     if y != "B8"} for st in CF.S2A_STRATA}}, ["B8"]))
    out2 = RC.s2_assembly(None, attrs, _Corpus(), store=_Store())
    assert out2["s2a"]["status"] == CF.UNDECIDED and out2["verdict"] == CF.UNDECIDED
    # a missing seed: the contrast record set is the completeness unit and S2 carries no verdict
    monkeypatch.setattr(RC, "s2_seed_block", lambda *a, **k: (None, []))
    out3 = RC.s2_assembly(None, attrs, _Corpus(), store=_Store())
    assert out3["status"] == "INCOMPLETE" and out3["verdict"] == CF.UNDECIDED
    assert out3["completeness_unit"] == CF.COMPLETENESS_UNIT
