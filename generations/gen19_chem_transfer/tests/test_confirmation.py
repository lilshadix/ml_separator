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
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from gen19ct import paths
from gen19ct.chemistry import support_graph as SG
from gen19ct.evaluation import confirmation as CF
from gen19ct.evaluation import discovery as D
from gen19ct.evaluation import h3 as H3
from gen19ct.evaluation import metrics as EM
from gen19ct.evaluation import pairs as EP
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


def test_gate_refuses_a_second_run_with_or_without_resume(root, plan_file, registry):
    """A FINISHED run (decisions/confirmation.json) is spent: --resume completes an UNFINISHED one, never re-opens this
    one (POST-HOC addendum 8 item 2)."""
    store_path = _write_store(root, root.parent / "outside_store.json")
    CF.write_decisions(root, {"code_digest": RC.code_digest()["combined"], "claim_ids": ["C1", "C4"]})
    with pytest.raises(SystemExit, match="nothing is re-run"):
        _gate(root, store_path, registry_path=registry)
    with pytest.raises(SystemExit, match="spent"):
        _gate(root, store_path, resume=True, registry_path=registry)
    # ... while the STARTED marker of a run that did not finish is what --resume may complete
    CF.decisions_path(root).unlink()
    from gen19ct.manifest import write_json as _wj
    _wj(CF.started_path(root), {"code_digest": RC.code_digest()["combined"], "claim_ids": ["C1", "C4"]})
    g = _gate(root, store_path, resume=True, registry_path=registry)
    assert g["lock"]["mode"] == "resume_unfinished"
    assert g["lock"]["may_only"] == "complete unfitted folds of the started run"


def test_resume_refuses_new_code_or_a_wider_claim_list(root, plan_file, registry):
    """On the run --resume may continue: the one that STARTED and never wrote decisions/confirmation.json."""
    from gen19ct.manifest import write_json as _wj

    _wj(CF.started_path(root), {"code_digest": "other-code", "claim_ids": ["C1"]})
    v = CF.lock_verdict(root, resume=True, code_digest="live-code", claim_ids=["C1"])
    assert v["ok"] is False and v["mode"] == "resume_refused_code_changed"
    _wj(CF.started_path(root), {"code_digest": "live-code", "claim_ids": ["C1"]})
    v2 = CF.lock_verdict(root, resume=True, code_digest="live-code", claim_ids=["C1", "C9"])
    assert v2["ok"] is False and v2["new_claims"] == ["C9"]
    # and a FINISHED run is refused whatever its code and claims are (addendum 8 item 2)
    CF.write_decisions(root, {"code_digest": "live-code", "claim_ids": ["C1"]})
    v3 = CF.lock_verdict(root, resume=True, code_digest="live-code", claim_ids=["C1"])
    assert v3["ok"] is False and v3["mode"] == "already_run_resume_refused"


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


def _v6_analogue():
    """A synthetic analogue of section 3.4 for the REAL ``RC.build_v6_folds``: 13 V6 systems C0..C12, each with a
    component-sharing system ``Ck|Q`` (shares component Ck), two unrelated systems, five states, X(?) Pr / Nd / La rows.
    Built here so that nothing before the confirmation run touches the real V6 folds, even in memory (POST-HOC addendum 6
    item 2, task X finding V-F6); the real-corpus facts the old test asserted (13 systems, 2 x 209 scored rows) are in
    section 3.4 and are re-derived by the run itself (``fold_files[*].n_scored_rows``)."""
    from types import SimpleNamespace

    from gen19ct.models import interface as I

    v6_sys = [f"C{k}" for k in range(13)]
    share = {s: f"{s}|Q" for s in v6_sys}
    other = ["U", "Z"]
    states = ["Pr(III)", "Nd(III)", "La(III)", "Sm(III)", "Am(III)"]
    recs, rid = [], [0]
    rng = np.random.default_rng(7)

    def add(sy, st, el, grp, n):
        for k in range(n):
            recs.append({FI.ROW_ID: f"R:{rid[0]}", SG.METAL_COL: st, SG.ELEMENT_COL: el, SG.SYSTEM_COL: sy,
                         SG.PUB_COL: f"pub_{grp}", FI.GROUP_COL: grp, I.PUB_GROUP_COL: grp, "acid_primary": "HNO3",
                         SG.LOG_ACID_COL: float(k) / 6.0, SG.LOG_EXT_COL: -1.0 + 0.01 * len(sy), SG.TEMP_COL: 25.0,
                         "condition_key": f"ck{k}", "duplicate_group_id": f"dg_{rid[0]}", "g19_ox": (3 if st else np.nan),
                         "solvent_key": "kerosene", "acid_concentration_M": 1.0 + k, "extractant_primary_concentration_M": 0.1,
                         "components": None, "acid_signature": "HNO3", "acid_anion": "nitrate",
                         "acid_concentration_organic_M": np.nan, "nitrate_concentration_M": 1.0 + k, "n_organic_extractants": 1,
                         "metal_concentration_M": 1e-3, "phase_ratio_org_aq": 1.0, "solvent_primary": "kerosene",
                         "solvent_components": None, "modifier_name": None, "modifier_concentration_M": np.nan,
                         "contact_time_min": 30.0, "shaking_time_min": 30.0, I.TARGET_COL: float(rng.normal())})
            rid[0] += 1

    for s in v6_sys:
        for st in states:
            add(s, st, SG.metal_properties(st)["symbol"], f"g_{s}", 6)
        add(s, None, "Pr", f"g_{s}", 2)               # X(?) Pr rows of S: hidden, never scored (section 2)
        add(s, None, "Nd", f"g_{s}", 2)
        add(s, None, "La", f"g_{s}", 1)               # X(?) La row of S: must stay in training
        for st in ("Pr(III)", "Nd(III)", "La(III)"):
            add(share[s], st, SG.metal_properties(st)["symbol"], f"g_{share[s]}", 4)
        add(share[s], None, "Nd", f"g_{share[s]}", 1)  # X(?) Nd of the sharing system: hidden unscored
    for o in other:
        for st in ("Pr(III)", "Nd(III)", "La(III)"):
            add(o, st, SG.metal_properties(st)["symbol"], f"g_{o}", 5)
    df = pd.DataFrame(recs)
    df.index = pd.Index([f"L{i}" for i in range(len(df))])
    df[SG.METAL_COL] = df[SG.METAL_COL].astype(object)
    v6 = pd.Series(df[SG.SYSTEM_COL].isin(v6_sys).to_numpy() & df[SG.METAL_COL].isin(list(CF.V6_METALS)).to_numpy(),
                   index=df.index)
    fc = SimpleNamespace(frame=df, v6=v6, v6_systems=tuple(v6_sys), universe=set(df[FI.ROW_ID].astype(str)))
    return fc, share, other


def test_the_v6_design_hides_exactly_what_section_3_4_says():
    """Section 3.4 / addendum 6 item 2, on the synthetic 13-system analogue with the REAL builder: per system Pr(III) x S
    and Nd(III) x S hidden TOGETHER under the component-aware state-level rule -- the hidden rows are the Pr and Nd rows,
    known state and X(?), of S and of its component-sharing system; only the known-state Pr / Nd rows of S itself are
    SCORED; every other row (other states of S, X(?) La of S, La(III) of the sharing system, unrelated and other V6
    systems) stays in training.  The design is seed-independent and refuses a corpus of 12 systems."""
    from types import SimpleNamespace

    fc, share, other = _v6_analogue()
    df, v6 = fc.frame, fc.v6
    folds, stats = RC.build_v6_folds(fc)
    assert len(folds) == CF.S2A_N_SYSTEMS == 13 and stats["n_systems"] == 13 and stats["guard_all_ok"]
    ids = df[FI.ROW_ID].astype(str)
    prnd = df[SG.METAL_COL].isin(list(CF.V6_METALS)) | (df[SG.METAL_COL].isna() & df[SG.ELEMENT_COL].isin(["Pr", "Nd"]))
    for f, ps in zip(folds, stats["per_system"]):
        s = f.meta["system"]
        cells = [tuple(c) for c in f.meta["cells"]]
        assert [c[0] for c in cells] == list(CF.V6_METALS) and cells[0][1] == cells[1][1] == s
        hid, sc = set(f.hidden_row_ids), set(f.scored_row_ids)
        scope = (df[SG.SYSTEM_COL] == s) | (df[SG.SYSTEM_COL] == share[s])
        assert hid == set(ids[df[scope & prnd].index])                                   # exactly Pr/Nd of S + sharer
        assert sc == set(ids[df[(df[SG.SYSTEM_COL] == s) & df[SG.METAL_COL].isin(list(CF.V6_METALS))].index])
        train = set(ids) - hid
        for cond in ((df[SG.SYSTEM_COL] == s) & df[SG.METAL_COL].isna() & (df[SG.ELEMENT_COL] == "La"),
                     df[SG.SYSTEM_COL].isin(other),
                     (df[SG.SYSTEM_COL] == s) & df[SG.METAL_COL].isin(["La(III)", "Sm(III)", "Am(III)"]),
                     (df[SG.SYSTEM_COL] == share[s]) & (df[SG.METAL_COL] == "La(III)"),
                     df[SG.SYSTEM_COL].isin([x for x in fc.v6_systems if x != s])):
            assert set(ids[df[cond].index]) <= train
        assert f.half == "NA" and f.seed is None and f.meta["component_aware"] is True
        assert f.meta["n_scored_pr"] == f.meta["n_scored_nd"] == 6
        assert all(not f.row_unit[r].startswith("hidden_unscored") for r in sc)
        assert all(f.row_unit[r].startswith("hidden_unscored") for r in hid - sc)
        assert ps["n_own_known_state_rows"] == 12 and ps["n_own_hidden_not_v6"] == 0    # the V-F7 stats, recorded
    assert len({f.fold_id for f in folds}) == 13 and all(f.fold_id == D.safe_fold_name(f.fold_id) for f in folds)
    assert stats["n_scored_rows"] == 13 * 12 and set(stats) >= {"rule", "sensitivity_7_systems"}
    scope = RC.assert_fold_v6_scope([], {(CF.V6_STEM, 1): folds}, set(ids[v6.to_numpy()]))
    assert scope["carve_out_holds_for_every_non_v6_design"] and scope["v6_rows_scored_only_in_the_v6_job"]
    # seed-INDEPENDENT as a design: the same folds whatever the seed index, so one hash under every index
    assert FI.design_hash(RC.build_v6_folds(fc)[0]) == FI.design_hash(folds)
    with pytest.raises(AssertionError, match="registers 13 V6 systems"):
        RC.build_v6_folds(SimpleNamespace(frame=df, v6=v6, v6_systems=tuple(fc.v6_systems[:12]), universe=fc.universe))


def test_the_v6_builder_refuses_a_mask_that_would_hide_a_known_state_prnd_row_unscored():
    """Task X finding V-F7: the builder asserted 'a scored row is not a V6_TARGET_ROW' but not the converse, so a
    known-state Pr / Nd row of a V6 system ABSENT from the V6 mask was hidden and silently left unscored (measured: a mask
    missing one Pr row was NOT refused).  Now it is, per system, with the count; the real corpus mask has 0 such rows in
    all 13 systems (checked without building a fold), so the assertion cannot fire on the real run."""
    from types import SimpleNamespace

    fc, _share, _other = _v6_analogue()
    df = fc.frame
    folds, _ = RC.build_v6_folds(fc)
    bad = fc.v6.copy()
    first_scored = df.index[df[FI.ROW_ID].astype(str) == folds[0].scored_row_ids[0]][0]
    bad.loc[first_scored] = False
    with pytest.raises(AssertionError, match="1 known-state Pr/Nd row.s. of the system are hidden but not V6_TARGET_ROWS"):
        RC.build_v6_folds(SimpleNamespace(frame=df, v6=bad, v6_systems=fc.v6_systems, universe=fc.universe))
    # ... and the existing direction still holds: a mask row that is not a known-state Pr/Nd row of the system is never scored
    wide = fc.v6.copy()
    la = df.index[(df[SG.SYSTEM_COL] == "C0") & (df[SG.METAL_COL] == "La(III)")][0]
    wide.loc[la] = True
    folds2, _ = RC.build_v6_folds(SimpleNamespace(frame=df, v6=wide, v6_systems=fc.v6_systems, universe=fc.universe))
    assert str(df.loc[la, FI.ROW_ID]) not in set(folds2[0].scored_row_ids)


def test_the_withheld_seed_colourings_are_seed_dependent_and_derive_their_counts(root, fold_corpus):
    """The per-seed files are built by the registered rule, differ between seeds, and their counts are DERIVED."""
    store_path = _write_store(root, root.parent / "outside_store.json")
    store = CF.load_seed_store(store_path, root=root, seal=SEAL, prereg_paths=_prereg_paths(root))
    # with_v6=False: nothing before the confirmation run builds the real V6 folds, even in memory (addendum 6 item 2,
    # task X finding V-F6); the V6 design is exercised on the synthetic analogue above with the real builder
    plans = RC.build_confirmation_folds(store, root, stems=("V5__primary__batched_max4",), corpus=fold_corpus,
                                       with_v6=False)
    by_stem = {}
    for p in plans:
        by_stem.setdefault(p.stem, {})[p.seed_index] = p
    v5 = by_stem["V5__primary__batched_max4"]
    assert set(v5) == {1, 2, 3, 4, 5}
    # seed dependent: the design hashes differ between seeds (R19 item 4 would be vacuous otherwise)
    assert len({p.design_hash for p in v5.values()}) == 5
    assert RC.V6_STEM not in by_stem                     # the real V6 design is built by the run alone (V-F6)
    # counts are derived, and the plan's figure is only compared
    assert all(p.n_folds > 0 and p.n_scored_units > 0 for p in v5.values())
    assert all(abs(p.n_folds_confirmation_half - 27) <= 2 for p in v5.values())
    # no file, directory or recorded field names a seed
    scan = CF.scan_for_seed_leak(store, [CF.folds_dir(root)])
    assert scan["ok"], scan
    assert json.dumps([p.record() for p in plans]).count("104729") == 0
    # every written design reads back with its recorded hashes
    for p in plans:
        folds = FI.read_design(Path(p.path))
        assert FI.design_hash(folds) == p.design_hash and len(folds) == p.n_folds
    # and the V6 carve-out holds on the FILES: no V5 colouring scores a V6_TARGET_ROW
    v6_ids = set(fold_corpus.frame.loc[fold_corpus.v6.to_numpy(dtype=bool), FI.ROW_ID].astype(str))
    for p in plans:
        for f in FI.read_design(Path(p.path)):
            assert not CF.is_v6(f.design) and not (set(f.scored_row_ids) & v6_ids)


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
    # every stage of the single run is implemented (the two task X left open -- the V6 configuration reading and C4's
    # ACT_PERMUTED transform -- are now written, tested and rehearsed end to end in tests/test_confirmation_rehearsal.py)
    assert [s["stage"] for s in body["run_stages"] if not s["implemented"]] == []
    assert {s["stage"] for s in body["run_stages"]} >= {"v6_frozen_configurations", "c4_act_permuted_training_transform"}
    assert {s["stage"] for s in body["run_stages"]} >= {"withheld_seed_fold_files", "fit_loop", "writers",
                                                        "v6_arm_fitting", "s2c_pair_conformal"}


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
    from gen19ct.manifest import write_json as _wj

    CF.write_decisions(root, {"code_digest": "x" * 64, "claim_ids": ["C1", "C4"]})
    first = CF.lock_verdict(root, resume=False, code_digest="x" * 64, claim_ids=["C1", "C4"])
    assert first["ok"] is False and first["mode"] == "already_run"
    spent = CF.lock_verdict(root, resume=True, code_digest="x" * 64, claim_ids=["C1", "C4"])
    assert spent["ok"] is False and spent["mode"] == "already_run_resume_refused"   # addendum 8 item 2
    # the second invocation --resume may serve is the one that STARTED and did not finish
    CF.decisions_path(root).unlink()
    _wj(CF.started_path(root), {"code_digest": "x" * 64, "claim_ids": ["C1", "C4"]})
    ok = CF.lock_verdict(root, resume=True, code_digest="x" * 64, claim_ids=["C1", "C4"])
    assert ok["ok"] is True and ok["mode"] == "resume_unfinished"
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
    """POST-HOC addendum 7 item 1 registers the construction: the pair-conformal reading DECIDES, the quadrature one is
    printed beside it, the record says whether the choice mattered, and the fallback folds are flagged and counted."""
    r = CF.s2c_coverage({"pair_conformal": {"80": 0.80, "95": 0.92},
                         "quadrature": {"80": 0.62, "95": 0.80}})
    assert r["status"] == "PASS" and r["readings_disagree"]
    assert r["statuses_by_reading"] == {"pair_conformal": "PASS", "quadrature": "FAIL"}
    assert r["quadrature_fallback"] == {"n_folds": 0, "n_pairs": 0,
                                        "min_calibration_pairs": CF.S2C_MIN_CALIBRATION_PAIRS}
    assert "no fold used the fallback" in r["detail"]
    # the registered value is missing -> no verdict, and the fallback flag is still carried
    nv = CF.s2c_coverage({"quadrature": {"80": 0.8, "95": 0.9}})
    assert nv["status"] == CF.NOT_EVALUATED and nv["quadrature_fallback"]["n_folds"] == 0
    # a fallback is COUNTED in the verdict block, never hidden inside the number
    fb = CF.s2c_coverage({"pair_conformal": {"80": 0.80, "95": 0.92}, "quadrature": {"80": 0.80, "95": 0.92}},
                         fallback_folds={"n_folds": 2, "n_pairs": 9})
    assert fb["status"] == "PASS" and not fb["readings_disagree"]
    assert fb["quadrature_fallback"]["n_folds"] == 2 and "fell back to quadrature" in fb["detail"]
    assert CF.S2C_PRIMARY_READING == "pair_conformal" and CF.S2C_MIN_CALIBRATION_PAIRS == 20
    assert CF.S2C_INTERVAL_READINGS == ("pair_conformal", "quadrature")


def test_a_v6_pair_interval_is_the_pair_conformal_one_with_the_quadrature_fallback():
    """POST-HOC addendum 7 item 1. The registered ``pair_conformal`` interval is the predicted difference +- the PAIR
    conformal quantile of the fold; ``quadrature`` is the root-sum-square of the two ROW half-widths about the same
    centre and is always printed; a pair whose fold carries no pair quantile takes the quadrature interval and is
    flagged.  The row interval columns are ``metrics.interval_columns``, not the ``lo80`` / ``hi80`` the earlier reading
    looked for and would never have found on a real frame."""
    lo, hi = EM.interval_columns(0.80)
    assert (lo, hi) == ("lower_80", "upper_80")
    frame = pd.DataFrame({lo: [0.0, 1.0], hi: [2.0, 3.0], EM.PRED_COL: [1.0, 2.0],
                          RC.pair_quantile_column("80"): [0.4, 0.4]}, index=["A", "B"])
    pairs = pd.DataFrame({"idx_a": ["A"], "idx_b": ["B"]})
    iv = RC.v6_logsf_intervals(frame, pairs, "80")
    assert set(iv) == {"pair_conformal", "quadrature"} == set(CF.S2C_INTERVAL_READINGS)
    assert iv["pair_conformal"]["lo"].iloc[0] == pytest.approx(-1.4)      # centre -1.0, +- the pair quantile 0.4
    assert iv["pair_conformal"]["hi"].iloc[0] == pytest.approx(-0.6)
    assert not bool(iv["pair_conformal"]["fallback"].iloc[0])
    assert iv["quadrature"]["lo"].iloc[0] == pytest.approx(-1.0 - np.sqrt(2.0))
    assert iv["quadrature"]["hi"].iloc[0] == pytest.approx(-1.0 + np.sqrt(2.0))
    # a fold that fell back carries NaN as its pair quantile: the pair takes quadrature and says so
    ivf = RC.v6_logsf_intervals(frame.assign(**{RC.pair_quantile_column("80"): [np.nan, np.nan]}), pairs, "80")
    assert bool(ivf["pair_conformal"]["fallback"].iloc[0])
    assert ivf["pair_conformal"]["lo"].iloc[0] == pytest.approx(ivf["quadrature"]["lo"].iloc[0])
    # without the pair quantile column at all: only the printed construction, never a pair_conformal verdict
    assert set(RC.v6_logsf_intervals(frame.drop(columns=[RC.pair_quantile_column("80")]), pairs, "80")) == {"quadrature"}
    assert not RC.v6_logsf_intervals(frame.drop(columns=[lo]), pairs, "80")
    # two members with DIFFERENT pair quantiles would mean two folds, which section 3.4 forbids
    with pytest.raises(AssertionError, match="different PAIR conformal quantiles"):
        RC.v6_logsf_intervals(frame.assign(**{RC.pair_quantile_column("80"): [0.4, 0.9]}), pairs, "80")


def test_the_v6_dispatch_is_the_v5_inner_design_and_the_refusal_machinery_still_works():
    """POST-HOC addendum 7 item 2. A V6 job now takes the V5 inner design of section 7 as amended by addendum 1 item 1:
    the simultaneous-hiding inner cells, their calibration and their recorded signature.  MEASURED on the live runner,
    not assumed.  ``RC.refusal`` still names every unimplemented stage (task X found two, below)."""
    import inspect

    from gen19ct.models import inner_design as ID

    assert [n for n, ok, _ in RC.RUN_STAGES if not ok] == []          # every stage of the single run is implemented
    rd = RC.runner_module()
    job = D.JobSpec(kind="fit", arm="M2", design="V6", variant="prnd", scheme="exact", seed=7)
    assert rd.v5_family(job) and rd.inner_variant(job) == "primary"
    sig = rd.inner_design_signature(job)
    assert sig is not None and sig["inner_design"] == D.INNER_DESIGN_NAME and sig["module"] == D.INNER_DESIGN_MODULE
    assert sig == rd.inner_design_signature(D.JobSpec(kind="fit", arm="M2", design="V5", variant="primary",
                                                      scheme="exact", seed=7))
    assert type(rd.boosted_design(job)).__name__ == "V5SimultaneousTuning"
    desc = ID.learned_arm_inner_design("V6", "primary", frame=pd.DataFrame({"acid_primary": ["HNO3"]},
                                                                          index=["r0"])).describe()
    assert all(desc[k] == sig[k] for k in sig if k != "module")
    # the comparator splitter of a V6 job: the V5 CELL design, not the V2 metal-holdout one
    src = inspect.getsource(rd.ComparatorIntervalsRunner.point)
    assert 'job.design in ("V5", "V6")' in src and "InnerCellCalibration" in src
    text = RC.refusal("v6_arm_fitting")
    assert "has NOT been spent" in text and "NOT IMPLEMENTED" in text


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


# --------------------------------------------------------------------------------------------- #
# POST-HOC addendum 7 items 1-2: the V6 path end to end on a SYNTHETIC mini-corpus
# --------------------------------------------------------------------------------------------- #

#: five systems and five states, because the registered V5 PRIMARY eligibility (``k10_p1_m3``) needs >= 4 other systems
#: for the metal and >= 4 other states for the system before ONE cell is eligible at all
V6_SMOKE_SYSTEMS = ("S1", "S2", "S3", "S4", "S5")
V6_SMOKE_STATES = ("Pr(III)", "Nd(III)", "La(III)", "Ce(III)", "Sm(III)")
#: the V6 system of the mini-corpus: its Pr(III) and Nd(III) rows are the V6_TARGET_ROWS
V6_SMOKE_SYSTEM = "S1"
V6_SMOKE_GROUPS = ("g1", "g2", "g3", "g4")


def _v6_frame() -> pd.DataFrame:
    """A mini-corpus the V5 PRIMARY inner design is satisfiable on: 25 cells of 12 rows each over six condition keys,
    every cell inside ONE publication group (so its majority group -- the unit the inner folds are assigned by -- is
    unambiguous) and the four groups spread over the cells, so the three inner folds all hold cells.  The V6 system's
    Pr(III) and Nd(III) rows share a group, so they form section 2 comparable pairs at each condition key."""
    rng = np.random.default_rng(1907)
    recs = []
    for si, sy in enumerate(V6_SMOKE_SYSTEMS):
        for mi, st in enumerate(V6_SMOKE_STATES):
            grp = (V6_SMOKE_GROUPS[0] if (sy == V6_SMOKE_SYSTEM and st in ("Pr(III)", "Nd(III)"))
                   else V6_SMOKE_GROUPS[(si + mi) % len(V6_SMOKE_GROUPS)])
            for k in range(12):
                recs.append({FI.ROW_ID: f"V:{si}{mi}{k:02d}", SG.METAL_COL: st,
                             SG.ELEMENT_COL: SG.metal_properties(st)["symbol"], SG.SYSTEM_COL: sy,
                             SG.PUB_COL: f"pub_{grp}", FI.GROUP_COL: grp, I.PUB_GROUP_COL: grp,
                             "acid_primary": "HNO3", SG.ACID_ANION_COL: "nitrate",
                             SG.LOG_ACID_COL: float(k % 6) / 6.0, SG.LOG_EXT_COL: -1.0 + 0.1 * si,
                             SG.TEMP_COL: 25.0, "condition_key": f"ck{k % 6}",
                             # the columns folds.io.FRAME_COLUMNS / leakage.KEY_* need for the V6 guards
                             "duplicate_group_id": f"dg_{si}{mi}{k:02d}", "g19_ox": 3,
                             "solvent_key": "kerosene", "acid_concentration_M": 1.0 + (k % 6),
                             "extractant_primary_concentration_M": 0.1 + 0.01 * si,
                             # the archive columns normalize.condition_vector needs (the arms' condition block)
                             "components": None, "acid_signature": "HNO3", "acid_anion": "nitrate",
                             "acid_concentration_organic_M": np.nan, "nitrate_concentration_M": 1.0 + (k % 6),
                             "n_organic_extractants": 1, "metal_concentration_M": 1e-3,
                             "phase_ratio_org_aq": 1.0, "solvent_primary": "kerosene", "solvent_components": None,
                             "modifier_name": None, "modifier_concentration_M": np.nan,
                             "contact_time_min": 30.0, "shaking_time_min": 30.0,
                             I.TARGET_COL: float(0.4 * mi + 0.3 * si + 0.2 * (k % 6) + 0.05 * rng.normal())})
    df = pd.DataFrame(recs)
    df.index = pd.Index([f"V{i}" for i in range(len(df))])
    return df


def _v6_corpus(tmp_path: Path) -> tuple[Any, FI.Fold, pd.DataFrame]:
    """The mini-corpus, its ONE V6 fold (Pr(III) x S1 and Nd(III) x S1 hidden together, section 3.4) and the pair-key
    attribute frame the S2 assembly and the S2(c) pair calibration read."""
    df = _v6_frame()
    rd = RC.runner_module()
    is_v6 = (df[SG.SYSTEM_COL] == V6_SMOKE_SYSTEM) & df[SG.METAL_COL].isin(["Pr(III)", "Nd(III)"])
    v6 = pd.Series(is_v6.to_numpy(dtype=bool), index=df.index)
    coext = pd.Series(False, index=df.index)
    halves = {d: pd.Series("C", index=df.index) for d in ("V5", "V5P", "V5PAIR", "V1", "V2")}
    folds_dir = tmp_path / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)
    hid = df.loc[is_v6, FI.ROW_ID].astype(str).tolist()
    fold = FI.make_fold(design=CF.V6_DESIGN, variant=CF.V6_VARIANT_NAME, scheme=CF.V6_SCHEME,
                        fold_id=f"v6_prnd__{V6_SMOKE_SYSTEM}", half="NA", seed=None, hidden=hid, scored=hid,
                        unit_type="system", units=[V6_SMOKE_SYSTEM],
                        row_unit={r: V6_SMOKE_SYSTEM for r in hid}, row_half={r: "NA" for r in hid},
                        meta={"cells": [["Pr(III)", V6_SMOKE_SYSTEM], ["Nd(III)", V6_SMOKE_SYSTEM]],
                              "component_aware": True})
    FI.write_design([fold], folds_dir)
    corpus = rd.Corpus(frame=df, slim=df, table=I.RowTable(df), v6=v6, coext=coext, systems=None, comps=None, cv=None,
                       pmap=None, row_half=halves, folds_dir=folds_dir)
    attrs = pd.DataFrame({"row_id": df[FI.ROW_ID].astype(str).to_numpy(),
                          EM.PUB_GROUP_COL: df[FI.GROUP_COL].astype(str).to_numpy(),
                          EM.SYSTEM_COL: df[SG.SYSTEM_COL].astype(str).to_numpy(),
                          EM.CONDITION_KEY_COL: df["condition_key"].astype(str).to_numpy(),
                          EM.METAL_STATE_COL: df[SG.METAL_COL].astype(object).to_numpy(),
                          EM.Y_COL: df[I.TARGET_COL].to_numpy(dtype=float),
                          RC.ACID_STRATUM_COL: "HNO3"}).set_index("row_id", drop=False)
    return corpus, fold, attrs


@pytest.mark.slow
def test_the_v6_path_end_to_end_on_a_synthetic_mini_corpus(root, tmp_path, monkeypatch, capsys):
    """POST-HOC addendum 7 items 1-2, exercised END TO END on a SYNTHETIC mini-corpus with a fake seed store: one V6
    fold, one arm (M1, in the V6 inventory), prepare_fold -> tune -> refit -> predict -> intervals -> record -> score.

    The real V6 hold-out runs ONCE (section 3.4) and is untouched here: the corpus, the fold, the attributes and the
    five seeds are all the test's own, in ``tmp_path``.  What this proves about the corrected dispatch:

    * the V6 job is tuned and calibrated on the addendum-1 SIMULTANEOUS V5 inner design, and its record says so;
    * the V6 guards pass (``v6_guard``, ``v6_inner_check``) and the carve-out holds in both directions;
    * the interval step runs and writes the row intervals AND the addendum-7 PAIR conformal quantiles;
    * the pair guard (``v6_guard_scored_pairs``) accepts the scored pairs, and S2(a), S2(b) and S2(c) assemble from them;
    * no withheld-seed-shaped label appears in any record, parquet value or log line.
    """
    monkeypatch.setattr(REG, "registry_path", lambda r=None: root / "manifests" / "digest_registry.json")
    store_path = _write_store(root, root.parent / "v6_smoke_store.json")
    store = CF.load_seed_store(store_path, root=root, seal=SEAL, prereg_paths=_prereg_paths(root))
    code = RC.code_digest()["combined"]
    REG.register_stage(CF.STAGE, below_footer_sha256="a" * 64, code_digest=code, git_head=None, addenda_count=7,
                       note="v6 smoke test", path=root / "manifests" / "digest_registry.json")
    corpus, fold, attrs = _v6_corpus(tmp_path)
    rd = RC.runner_module()
    seed_index = 1
    seed = store.seed(seed_index)
    job = D.JobSpec(kind="fit", arm="M1", design=CF.V6_DESIGN, variant=CF.V6_VARIANT_NAME, scheme=CF.V6_SCHEME,
                    seed=seed, writes=("M1",))
    assert rd.v5_family(job) and rd.inner_design_signature(job) is not None      # the corrected dispatch, measured
    out_root = tmp_path / "out"
    r = RC.run_fold(job, fold, 0, corpus, out_root, store=store, seed_index=seed_index,
                    runners=rd.default_runners(), code=code, state=D.PlanState(guard_mode={}),
                    prereg={"addenda_sha256": "a" * 64, "n_addenda": 7}, pair_attrs=attrs)
    assert r["status"] == "fitted"

    # ---- the record: the V5 inner design, the V6 half reading, the guards, and both interval constructions
    js = Path(r["records"]["M1"])
    rec = json.loads(js.read_text(encoding="utf-8"))
    assert rec["registry_stage"] == CF.STAGE and rec["half"] == "NA" and rec["seed_index"] == seed_index
    idr = rec["arm_record"]["inner_design"]
    assert idr["design"] == D.INNER_DESIGN_NAME and "V1 / V2" not in json.dumps(idr)   # NOT the V1 / V2 label
    assert idr["n_inner_folds"] >= 1 and idr["reading"] == D.READINGS["addendum1_inner_design"]
    assert rec["reading"] == CF.READINGS["v6_half"]
    assert all(g["ok"] for g in rec["outer_guard"])
    istep = rec["steps"][RC.INTERVAL_STEP]
    assert istep["status"] == "calibrated" and istep["n_calibration"] > 0
    pc = istep["pair_conformal"]
    assert pc["min_calibration_pairs"] == CF.S2C_MIN_CALIBRATION_PAIRS
    assert pc["method"] in (CF.S2C_PAIR_CONFORMAL, CF.S2C_QUADRATURE_FALLBACK)
    assert pc["fallback"] == (pc["n_calibration_pairs"] < CF.S2C_MIN_CALIBRATION_PAIRS)
    # the REGISTERED path is what this corpus exercises, so a later change cannot leave only the fallback tested
    assert pc["method"] == CF.S2C_PAIR_CONFORMAL and not pc["fallback"]
    assert pc["n_calibration_pairs"] >= CF.S2C_MIN_CALIBRATION_PAIRS and pc["n_calibration_rows"] > 0
    assert sum(s["n_pairs"] for s in pc["per_inner_split"]) == pc["n_calibration_pairs"]
    # E6, WIRED: the recorded consequence of the registered calibration population reaches the record, not only the
    # library function -- the composition of the calibration pairs and the exploratory Pr/Nd-only quantiles beside it
    pop = pc["population"]
    assert 0.0 <= pop["prnd_share"] <= 1.0 and pop["n_pairs"] == pc["n_calibration_pairs"]
    assert 0 <= pop["n_prnd_pairs"] <= pop["n_pairs"] and sum(pop["by_state_pair"].values()) <= pop["n_pairs"]
    assert pop["prnd_state_pair"] == " | ".join(sorted(CF.V6_METALS)) and pop["n_distinct_state_pairs"] >= 1
    assert pop["median_abs_residual"] is not None
    po = pc["prnd_only"]                                  # the EXPLORATORY Pr/Nd-only diagnostic, beside it
    assert po["n_calibration_pairs"] == pop["n_prnd_pairs"] and "EXPLORATORY" in po["label"]
    assert po["reading"] == CF.READINGS["s2c_calibration_population"]
    assert set(pc["quantiles"]) == {"50", "80", "95"} and all(v >= 0 for v in pc["quantiles"].values())
    assert pc["quantiles"]["50"] <= pc["quantiles"]["80"] <= pc["quantiles"]["95"]
    # and the fallback itself: below the registered minimum no pair quantile is produced at all
    few = CF.pair_conformal_quantiles([0.1] * (CF.S2C_MIN_CALIBRATION_PAIRS - 1))
    assert few["fallback"] and few["method"] == CF.S2C_QUADRATURE_FALLBACK and few["quantiles"] == {}
    assert REG.verify_record(rec, CF.STAGE, root / "manifests" / "digest_registry.json")["ok"] is True

    # ---- the prediction frame: the row intervals and the pair quantile columns
    pred = pd.read_parquet(js.with_suffix(".parquet"))
    lo80, hi80 = EM.interval_columns(0.80)
    assert {lo80, hi80, RC.pair_quantile_column("80"), RC.PAIR_METHOD_COL, RC.PAIR_N_CAL_COL} <= set(pred.columns)
    assert (pred[hi80] > pred[lo80]).all() and pred["intervals_status"].eq("split_conformal_inner").all()
    assert set(pred["row_id"]) == set(fold.scored_row_ids)

    # ---- the pair guard and S2(a)-(c) on those predictions
    v6 = pd.Series(corpus.v6.to_numpy(dtype=bool), index=corpus.frame[FI.ROW_ID].astype(str).to_numpy())
    cin = RC.v6_pair_inputs(pred, attrs, corpus, v6=v6, what="v6 smoke")
    pairs = cin["pairs"]
    assert len(pairs) > 0 and set(pairs["fold"].astype(str)) == {fold.fold_id}
    obs = pairs["logsf_obs"].to_numpy(dtype=float)
    cov, fb = {}, {}
    for lvl in ("80", "95"):
        iv = RC.v6_logsf_intervals(cin["frame"], pairs, lvl)
        assert set(iv) == set(CF.S2C_INTERVAL_READINGS)
        for reading, f in iv.items():
            inside = (obs >= f["lo"].to_numpy(dtype=float)) & (obs <= f["hi"].to_numpy(dtype=float))
            cov.setdefault(reading, {})[lvl] = float(inside.mean())
            if reading == CF.S2C_PRIMARY_READING:
                fb = {"n_pairs": int(f["fallback"].to_numpy(dtype=bool).sum())}
    s2c = CF.s2c_coverage(cov, fallback_folds=fb)
    assert s2c["status"] in ("PASS", "FAIL") and s2c["primary_reading"] == CF.S2C_PRIMARY_READING
    assert set(s2c["per_reading"]) == set(CF.S2C_INTERVAL_READINGS)
    # S2(a): the sign count and one paired direction contrast against HEAVIER, which needs no fitted yardstick
    sysv = pairs[EM.SYSTEM_COL].astype(str).to_numpy()
    sign = CF.s2a_sign_count({str(k): float(v) for k, v in pairs.groupby(sysv)["logsf_obs"].median().items()},
                             {str(k): float(v) for k, v in cin["logsf"].groupby(sysv).median().items()})
    assert sign["status"] in ("PASS", "FAIL") and sign["n_systems_scored"] == 1
    d = RC.v6_direction_by_system(pairs, cin["logsf"], EP.heavier_direction(pairs), direction_only=True)
    assert d["threshold"] == EP.V6_DIRECTION_THRESHOLD and set(d["delta_by_system"]) <= {V6_SMOKE_SYSTEM}
    # S2(b): the per-system logSF MAE against FLAT, on the same pairs
    mae = RC.v6_logsf_mae_by_system(pairs, cin["logsf"])
    flat = RC.v6_logsf_mae_by_system(pairs, EP.flat_logsf(pairs))
    s2b = CF.s2b_magnitude(logsf_mae=float(mae.mean()), flat_logsf_mae=float(flat.mean()),
                           lookup_logsf_mae=float(flat.mean()), gain_interval_excludes_zero=True,
                           logd_macro_mae=0.1, v5_lookup_comparator_mae=0.7670)
    assert s2b["status"] in ("PASS", "FAIL") and np.isfinite(s2b["gain_over_flat"])

    # ---- no withheld-seed-shaped label anywhere: the record, the parquet, the paths, the logs
    blobs = [js.read_text(encoding="utf-8"), json.dumps(rec, default=str), str(js), str(js.with_suffix(".parquet")),
             capsys.readouterr().out, json.dumps(s2c, default=str), json.dumps(sign, default=str),
             json.dumps(d, default=str)]
    for col in pred.columns:
        if pred[col].dtype == object:
            blobs.append(" ".join(map(str, pred[col].tolist())))
    for s in FAKE_SEEDS + [seed]:
        for b in blobs:
            assert str(s) not in b, (s, b[:200])
    assert CF.seed_token(seed_index) in str(js)                  # the opaque index is what labels the record instead


# --------------------------------------------------------------------------------------------- #
# task X: the six errors the verification pass found, each fixed and each measured here
# --------------------------------------------------------------------------------------------- #

@pytest.mark.slow
def test_m2_reads_m1s_confirmation_record_and_refuses_a_missing_or_stale_one(root, tmp_path, monkeypatch):
    """E1, the blocking one: M2 at confirmation reads M1's CONFIRMATION record of the same fold.

    ``NeuralRunner.m1_record`` resolved ``evaluation/discovery/M1/<design>/s<withheld seed>/<fold>.json``, a path this
    run never writes and whose directory names the withheld seed, so EVERY M2 fold raised "M2 needs M1's per-fold
    values" -- M2 on V5-primary (claims C1-C3), on V5-PAIR (S1(c)) and on V6 (S2). Measured here on the synthetic V6
    fold: M2 refuses before M1 exists, succeeds after it, records WHICH M1 record it used, and refuses a record written
    under different code.
    """
    monkeypatch.setattr(REG, "registry_path", lambda r=None: root / "manifests" / "digest_registry.json")
    store_path = _write_store(root, root.parent / "m2_store.json")
    store = CF.load_seed_store(store_path, root=root, seal=SEAL, prereg_paths=_prereg_paths(root))
    code = RC.code_digest()["combined"]
    REG.register_stage(CF.STAGE, below_footer_sha256="a" * 64, code_digest=code, git_head=None, addenda_count=7,
                       note="m2 path", path=root / "manifests" / "digest_registry.json")
    corpus, fold, attrs = _v6_corpus(tmp_path)
    rd = RC.runner_module()
    out_root, i = tmp_path / "out", 1
    seed = store.seed(i)
    kw = dict(store=store, seed_index=i, runners=rd.default_runners(), code=code,
              state=D.PlanState(guard_mode={}), prereg={"addenda_sha256": "a" * 64, "n_addenda": 7}, pair_attrs=attrs)

    def spec(arm):
        return D.JobSpec(kind="fit", arm=arm, design=CF.V6_DESIGN, variant=CF.V6_VARIANT_NAME, scheme=CF.V6_SCHEME,
                         seed=seed, writes=(arm,))

    # ---- before M1 exists: a NAMED refusal that carries no withheld seed and points at the confirmation layout
    with pytest.raises(RuntimeError) as ei:
        RC.run_fold(spec("M2"), fold, 0, corpus, out_root, **kw)
    msg = str(ei.value)
    assert "needs M1's per-fold values" in msg and "records/M1/" in msg and CF.seed_token(i) in msg
    assert not any(str(s) in msg for s in FAKE_SEEDS + [seed])
    assert "evaluation/discovery" not in msg.replace("\\", "/")

    # ---- M1, then M2: the confirmation path, end to end
    r1 = RC.run_fold(spec("M1"), fold, 0, corpus, out_root, **kw)
    assert r1["status"] == "fitted"
    r2 = RC.run_fold(spec("M2"), fold, 0, corpus, out_root, **kw)
    assert r2["status"] == "fitted"
    m1rec = json.loads(Path(r1["records"]["M1"]).read_text(encoding="utf-8"))
    m2rec = json.loads(Path(r2["records"]["M2"]).read_text(encoding="utf-8"))
    ar = m2rec["arm_record"]
    assert ar["m1_config_used"] is not None
    assert ar["m1_config_used"]["emb_dim"] == m1rec["arm_record"]["selected"]["emb_dim"]
    # the link is RECORDED: M2 names the exact M1 record it took its hyperparameters from (no discovery fold digest here)
    assert ar["m1_record_digest"] == CF.confirmation_record_digest(m1rec)
    assert not any(str(s) in json.dumps(m2rec, default=str) for s in FAKE_SEEDS + [seed])

    # ---- a stale M1: written under different code -> StaleRecordError, nothing re-scored
    js1 = Path(r1["records"]["M1"])
    js1.write_text(json.dumps({**m1rec, "code_digest": "f" * 64}), encoding="utf-8")
    CF.fold_paths(out_root, "M2", spec("M2").design_dir, i, CF.scrub(fold.fold_id, store))[1].unlink()
    with pytest.raises(D.StaleRecordError) as es:
        RC.run_fold(spec("M2"), fold, 0, corpus, out_root, **kw)
    assert "code_digest" in str(es.value) and not any(str(s) in str(es.value) for s in FAKE_SEEDS + [seed])

    # ---- and discovery's own path is untouched: with no locator, m1_record is discovery's
    fc, _ = RC.prepare_fold(spec("M2"), fold, 0, corpus, out_root, D.PlanState(guard_mode={}), code=code, what="disc")
    assert fc.sibling_record is None
    with pytest.raises(RuntimeError, match="M1's per-fold values"):
        rd.NeuralRunner("M2").m1_record(fc)


def test_fit_loop_dispatches_to_a_pool_and_the_worker_returns_only_scrubbed_values(root, tmp_path, monkeypatch):
    """E2: ``--workers`` was accepted, bounds-checked, logged and then ignored -- ``fit_loop`` was a sequential loop with
    no ``ProcessPoolExecutor`` anywhere, while the plan, the dry run and the stage table all advertised the 2-worker wall
    clock. The pool's dispatch loop and the worker body are exercised here in process (the real spawn is the next test),
    on the synthetic corpus: the ledger absorbing each completion, resumability through the pool, and the worker's own
    classification of a guard failure and of a defect -- both already SCRUBBED when they leave the worker.
    """
    monkeypatch.setattr(REG, "registry_path", lambda r=None: root / "manifests" / "digest_registry.json")
    store_path = _write_store(root, root.parent / "pool_store.json")
    store = CF.load_seed_store(store_path, root=root, seal=SEAL, prereg_paths=_prereg_paths(root))
    code = RC.code_digest()["combined"]
    REG.register_stage(CF.STAGE, below_footer_sha256="a" * 64, code_digest=code, git_head=None, addenda_count=7,
                       note="pool", path=root / "manifests" / "digest_registry.json")
    corpus, fold, attrs = _v6_corpus(tmp_path)
    rd = RC.runner_module()
    out_root = tmp_path / "out"
    seen = []

    class InlinePool:
        """The pool's interface, run in process on the synthetic corpus: ``_conf_worker_fold`` is called for real."""

        def __init__(self, **kw):
            RC.CW.corpus, RC.CW.runners, RC.CW.store = corpus, rd.default_runners(), store
            RC.CW.attached, RC.CW.pair_attrs = None, attrs

        def submit(self, fn, *a):
            from concurrent.futures import Future
            f = Future()
            f.set_running_or_notify_cancel()
            try:
                f.set_result(fn(*a))
            except BaseException as exc:                  # noqa: BLE001 -- the pool's own contract
                f.set_exception(exc)
            seen.append(1)
            return f

        def shutdown(self, wait=True):
            pass

    monkeypatch.setattr(RC, "ProcessPoolExecutor", InlinePool)
    monkeypatch.setattr(RC, "attach_seed_folds", lambda *a, **k: {})
    pids = iter(range(1, 1000))
    monkeypatch.setattr(RC, "_conf_worker_state",                      # one distinct pid per probe (every worker served)
                        lambda *a: {"pid": next(pids), "n_rows": int(corpus.table.n), "n_runners": 1, "store_verified": True,
                                    "seed_commitment_sha256": store.digest, "n_seed_indices": CF.N_SEEDS})
    job = CF.ConfJob("pool", "M1", "V6", CF.V6_STEM, 1, 1, tuned=False)
    kw = dict(corpus=corpus, state=D.PlanState(guard_mode={}), runners=rd.default_runners(), code=code,
              prereg={"addenda_sha256": "a" * 64, "n_addenda": 7}, workers=2, seed_store_path=str(store_path),
              pair_attrs=attrs)
    led = RC.fit_loop([job], store, out_root, **kw)
    rec = led.record()
    assert rec["workers_used"] == 2 and rec["n_folds_fitted"] == 1 and rec["n_errors"] == 0
    assert rec["worker_preflight"] and rec["worker_preflight"][0]["store_verified"] is True
    assert seen                                        # the pool really was submitted to
    assert CF.fold_paths(out_root, "M1", "V6__prnd_exact", 1, fold.fold_id)[1].exists()
    # the second invocation skips the done fold rather than refitting it (resumability through the pool)
    led2 = RC.fit_loop([job], store, out_root, **kw)
    assert led2.record()["n_folds_skipped_already_done"] == 1 and led2.record()["n_folds_fitted"] == 0

    # ---- the worker's own error classification, and that nothing unscrubbed leaves it
    spec = D.JobSpec(kind="fit", arm="M1", design=CF.V6_DESIGN, variant=CF.V6_VARIANT_NAME, scheme=CF.V6_SCHEME,
                     seed=store.seed(1), writes=("M1",))
    state_rec = D.PlanState(guard_mode={}).record()

    def _raise(exc):
        def f(*a, **k):
            raise exc
        return f

    monkeypatch.setattr(RC, "run_fold", _raise(AssertionError("guard at " + str(store.seed(1)))))
    g = RC._conf_worker_fold(spec, fold.fold_id, 0, str(out_root), 1, code, state_rec, None, "k")
    assert g["status"] == "guard_failure" and not any(str(s) in json.dumps(g) for s in FAKE_SEEDS)
    monkeypatch.setattr(RC, "run_fold", _raise(KeyError("boom " + str(store.seed(1)))))
    e = RC._conf_worker_fold(spec, fold.fold_id, 0, str(out_root), 1, code, state_rec, None, "k")
    assert e["status"] == "error" and "traceback_suppressed" in e
    assert not any(str(s) in json.dumps(e) for s in FAKE_SEEDS)
    # and the parent turns a worker error into the redacted run error that stops the run
    with pytest.raises(CF.RedactedRunError):
        RC.fit_loop([job], store, tmp_path / "out2", **kw)


def test_a_real_two_worker_pool_spawns_verifies_its_store_and_loads_the_corpus(root, monkeypatch):
    """E2, the part only a real process can prove: a spawned worker imports this script, loads the REAL corpus and
    VERIFIES the seed store itself.

    It fits nothing: the only thing submitted is the pre-flight self-check :func:`RC._conf_worker_state`, which reports
    what the worker holds. That is deliberate -- fitting a real fold here would be a confirmation-half or V6 score, and
    section 3.4 gives V6 exactly one run. The store is the test's PLACEHOLDER one and ``out_root`` is ``tmp_path``, whose
    ``manifests/`` holds its matching commitment, so the worker's verification is real against a fake commitment.
    """
    from concurrent.futures import ProcessPoolExecutor

    store_path = _write_store(root, root.parent / "spawn_store.json")
    store = CF.load_seed_store(store_path, root=root, seal=SEAL, prereg_paths=_prereg_paths(root))
    pool = ProcessPoolExecutor(max_workers=2, initializer=RC._init_conf_worker,
                               initargs=(str(store_path), str(root)))
    try:
        probes = [f.result() for f in [pool.submit(RC._conf_worker_state) for _ in range(4)]]
    finally:
        pool.shutdown(wait=True)
    pids = {p["pid"] for p in probes}
    # a SPAWNED process really ran (the pid is not this one) and never more than the pool's size.  The pool spawns
    # lazily, so four fast probes can all land on one worker; what this proves is that a child can import this script,
    # run the initializer, load the corpus and verify the store -- the four things --workers 2 depends on
    import os
    assert 1 <= len(pids) <= 2 and os.getpid() not in pids
    assert all(p["store_verified"] for p in probes)
    assert all(p["seed_commitment_sha256"] == store.digest for p in probes)
    assert all(p["n_rows"] > 1000 and p["n_runners"] > 0 for p in probes)
    assert all(p["n_seed_indices"] == CF.N_SEEDS for p in probes)
    assert not any(str(s) in json.dumps(probes) for s in FAKE_SEEDS)  # the probe names no seed


def test_the_closed_form_comparators_are_not_priced_at_zero():
    """E3: 1,702 of the 3,418 folds -- every B0 / B3i / B3x job -- were priced 0.0 s on a zero nothing had measured.

    Each one pays ``prepare_fold`` plus a full ``ConformalWrapper`` fit over the design's inner calibration splits.  The
    measured means are in ``CF.UNIT_SECONDS``; what this asserts is that no enumerated fold is free and that the plan's
    own arithmetic now includes them.
    """
    plan = CF.read_plan(CF.plan_path(paths.G19_ROOT))
    jobs = CF.enumerate_jobs(plan)
    free = [j.key for j in jobs if j.unit_seconds == 0.0]
    assert not free, free
    comparator_folds = sum(j.n_folds for j in jobs if j.arm in D.COMPARATOR_INTERVAL_ARMS)
    assert comparator_folds == 1702, comparator_folds
    cost = CF.cost_estimate(jobs)
    assert not cost["unknown_unit_cost"]
    added = sum(j.unit_seconds * j.n_folds for j in jobs if j.arm in D.COMPARATOR_INTERVAL_ARMS) / 3600.0
    assert 1.0 < added < 6.0, added          # the band the verification pass predicted; measured at about 2.2 h
    assert cost["serial_hours"] > 108.05     # strictly more than the figure that priced them at nothing


def test_the_fold_build_phase_cannot_print_a_withheld_seed(root, tmp_path, monkeypatch):
    """E4: until ``write_seed_design`` scrubs them, a batched fold carries the raw id ``s<withheld seed>_C_b000``.

    ``build_v5_batched`` / ``build_v5pair_batched`` raise ``AssertionError(f"{f.fold_id}: ...")``, and ``folds.io.guard``
    and ``cell_holdout.check_pair_isolation`` do the same, so every assertion of this phase names the seed.  ``fit_loop``
    scrubs both of its except arms; this phase, which runs BEFORE it, had no wrapper at all, and ``Run.__exit__`` returns
    early on an exception, so the leak was to the terminal (addendum 6 item 4 forbids it).
    """
    import traceback as _tb

    store_path = _write_store(root, root.parent / "build_store.json")
    store = CF.load_seed_store(store_path, root=root, seal=SEAL, prereg_paths=_prereg_paths(root))
    seeds = [str(s) for s in FAKE_SEEDS]

    def boom(fc, variant_name, seed):
        raise AssertionError(f"s{seed}_C_b000: two cells of one batch share a metal state or a system")

    stub = SimpleNamespace(frame=pd.DataFrame({FI.ROW_ID: ["a"]}), v6=pd.Series([False]))
    monkeypatch.setattr(RC, "load_fold_corpus", lambda: stub)
    monkeypatch.setattr(RC, "build_v6_folds", lambda fc: ([], {}))
    monkeypatch.setattr(RC, "build_v5_batched", boom)
    with pytest.raises(AssertionError) as ei:
        RC.build_confirmation_folds(store, tmp_path, stems=("V5__primary__batched_max4",), with_v6=False)
    txt = str(ei.value)
    assert "two cells of one batch" in txt and "si1_C_b000" in txt        # the id survives, the SEED does not
    assert not any(s in txt for s in seeds)
    assert ei.value.__context__ is None or ei.value.__suppress_context__
    assert not any(s in "".join(_tb.format_exception(ei.value)) for s in seeds)

    # a non-assertion defect becomes the redacted run error, as in the fit loop
    def bang(fc, variant_name, seed):
        raise KeyError(f"s{seed}_C_b000 missing")

    monkeypatch.setattr(RC, "build_v5_batched", bang)
    with pytest.raises(CF.RedactedRunError) as e2:
        RC.build_confirmation_folds(store, tmp_path, stems=("V5__primary__batched_max4",), with_v6=False)
    assert not any(s in str(e2.value) for s in seeds)
    # ... and with no store (the synthetic path) the original exception is left exactly as it was
    monkeypatch.setattr(RC, "build_v5_batched", boom)
    with pytest.raises(AssertionError, match="s1_C_b000"):
        RC.build_confirmation_folds(None, tmp_path, stems=("V5__primary__batched_max4",), with_v6=False)


def test_the_lock_is_spent_at_the_start_and_a_mixed_code_record_set_cannot_score(root, tmp_path, monkeypatch, plan_file,
                                                                                registry):
    """E5: the lock was spent only by ``decisions/confirmation.json``, which is written LAST.

    A run that wrote records and died left ``lock_verdict`` reading ``first_run``, so a second invocation -- under
    changed code, after the addendum 7 item 3 re-registration moved the registry entry -- could have concatenated two
    record sets written under two code digests into one score, because not one of ``read_seed_predictions``' call sites
    passed ``code=``.  Both halves are fixed and both are measured here.
    """
    from gen19ct.manifest import write_json as _wj

    code = RC.code_digest()["combined"]
    # (i) the START marker spends the lock, and --resume is the only thing that may continue
    assert CF.lock_verdict(root, resume=False, code_digest=code, claim_ids=["C1"])["mode"] == "first_run"
    _wj(CF.started_path(root), {"code_digest": code, "claim_ids": ["C1", "C4"],
                                "started_utc": "2026-09-24T00:00:00+00:00"})
    v = CF.lock_verdict(root, resume=False, code_digest=code, claim_ids=["C1"])
    assert v["ok"] is False and v["mode"] == "already_started" and "already STARTED" in v["reason"]
    assert CF.lock_verdict(root, resume=True, code_digest=code, claim_ids=["C1"])["mode"] == "resume_unfinished"
    assert CF.lock_verdict(root, resume=True, code_digest="f" * 64,
                           claim_ids=["C1"])["mode"] == "resume_refused_code_changed"
    assert CF.lock_verdict(root, resume=True, code_digest=code,
                           claim_ids=["C1", "C9"])["mode"] == "resume_refused_claims_grew"
    # the gate refuses on it too, so a second invocation cannot reach the fit loop
    with pytest.raises(SystemExit, match="already STARTED"):
        _gate(root, _write_store(root, root.parent / "lock_store.json"), registry_path=registry)

    # (ii) the exact scenario the addendum 7 item 3 re-registration opens: a record written under the code the registry
    #      held THEN, re-registered since.  registry.verify_record ACCEPTS it -- that is what the superseded fallback is
    #      for -- so the only thing that can refuse it at scoring time is the live-code check, and not one call site
    #      passed one.
    corpus, fold, attrs = _v6_corpus(tmp_path)
    monkeypatch.setattr(REG, "registry_path", lambda r=None: registry)
    old_code = "a1" * 32
    REG.register_stage(CF.STAGE, below_footer_sha256="a" * 64, code_digest=old_code, git_head=None, addenda_count=5,
                       note="the code the run started under", path=registry, force=True)
    store = CF.load_seed_store(_write_store(root, root.parent / "lock_store.json"), root=root, seal=SEAL,
                               prereg_paths=_prereg_paths(root))
    out_root = tmp_path / "out"
    r = RC.run_fold(D.JobSpec(kind="fit", arm="M1", design=CF.V6_DESIGN, variant=CF.V6_VARIANT_NAME,
                              scheme=CF.V6_SCHEME, seed=store.seed(1), writes=("M1",)), fold, 0, corpus, out_root,
                    store=store, seed_index=1, runners=RC.runner_module().default_runners(), code=old_code,
                    state=D.PlanState(guard_mode={}),
                    prereg={"addenda_sha256": "a" * 64, "n_addenda": 5},   # the `registry` fixture's own count
                    pair_attrs=attrs)
    rec = json.loads(Path(r["records"]["M1"]).read_text(encoding="utf-8"))
    assert rec["code_digest"] == old_code
    assert RC.read_seed_predictions(out_root, "M1", "V6__prnd_exact", 1, code=old_code) is not None
    REG.register_stage(CF.STAGE, below_footer_sha256="a" * 64, code_digest=code, git_head=None, addenda_count=5,
                       note="re-registered after an addendum (item 3)", path=registry, force=True)
    assert REG.verify_record(rec, CF.STAGE, registry)["matched_entry"] == "superseded"     # the registry is satisfied
    with pytest.raises(D.StaleRecordError, match="written under code"):
        RC.read_seed_predictions(out_root, "M1", "V6__prnd_exact", 1)                      # default = the LIVE digest
    assert RC.read_seed_predictions(out_root, "M1", "V6__prnd_exact", 1, code=None) is not None   # opt out, explicitly


def test_s2c_records_the_consequence_of_its_registered_calibration_population():
    """E6: the S2(c) calibration population is not the scored population, and that is REGISTERED.

    POST-HOC addendum 7 item 1 calibrates on "the comparable pairs" of the inner calibration set; section 2's pair key
    is (publication group, system, condition key), so ANY two distinct metal states of a condition group count -- Am/Eu
    exactly as much as Nd/Pr -- while S2(c) SCORES Nd/Pr pairs only, whose |logSF| is small.  The interval is therefore
    biased WIDE and the [0.70, 0.90] band at 80 % can FAIL on the HIGH side as an artefact.  The code is as registered
    and is NOT changed; what is asserted is that the consequence is recorded as numbers.
    """
    rng = np.random.default_rng(11)
    n = 60
    states = (["Nd(III)", "Pr(III)"] * 10) + (["Am(III)", "Eu(III)"] * 20)
    rows = pd.DataFrame({"row_id": [f"r{i}" for i in range(n)], EM.PUB_GROUP_COL: "g0", EM.SYSTEM_COL: "S",
                         EM.CONDITION_KEY_COL: [f"ck{i // 2}" for i in range(n)],
                         EM.METAL_STATE_COL: states, EM.Y_COL: rng.normal(size=n)}).set_index("row_id", drop=False)
    # the Am/Eu pairs are given far larger residuals, which is the whole mechanism
    signed = np.where(np.arange(n) < 20, rng.normal(scale=0.05, size=n), rng.normal(scale=1.0, size=n))
    res = CF.pair_calibration_residuals([{"unit": "c0", "fold": 0, "row_ids": list(rows.index),
                                          "signed": signed.tolist()}], rows, what="pop")
    pop = res["population"]
    assert pop["n_pairs"] == 30 and pop["n_prnd_pairs"] == 10 and abs(pop["prnd_share"] - 10 / 30) < 1e-12
    assert pop["prnd_state_pair"] == " | ".join(sorted(CF.V6_METALS))
    assert set(pop["by_state_pair"]) == {" | ".join(sorted(CF.V6_METALS)), "Am(III) | Eu(III)"}
    # the bias has the direction the reading states: the Pr/Nd residuals are the SMALLER ones
    assert pop["median_abs_residual_prnd"] < pop["median_abs_residual"]
    reg = CF.pair_conformal_quantiles(res["abs_residuals"])
    expl = CF.prnd_only_quantiles(res["prnd_only_abs_residuals"])
    assert not reg["fallback"] and reg["method"] == CF.S2C_PAIR_CONFORMAL
    assert expl["quantiles"] == {}                                   # 10 Pr/Nd pairs is below the registered minimum
    assert "EXPLORATORY" in expl["label"]
    big = CF.pair_calibration_residuals([{"unit": "c0", "fold": 0, "row_ids": list(rows.index),
                                          "signed": signed.tolist()},
                                         {"unit": "c1", "fold": 1, "row_ids": list(rows.index),
                                          "signed": (signed * 1.01).tolist()},
                                         {"unit": "c2", "fold": 2, "row_ids": list(rows.index),
                                          "signed": (signed * 0.99).tolist()}], rows, what="pop3")
    e2 = CF.prnd_only_quantiles(big["prnd_only_abs_residuals"])
    r2 = CF.pair_conformal_quantiles(big["abs_residuals"])
    assert e2["quantiles"] and e2["quantiles"]["80"] < r2["quantiles"]["80"]      # the artefact, as a number
    # ---- and the verdict block carries it, with the high-side flag, changing no status
    s = CF.s2c_coverage({"pair_conformal": {"80": 0.958, "95": 0.99}, "quadrature": {"80": 0.55, "95": 0.70}},
                        population={"prnd_share": 10 / 30, "exploratory_prnd_only_coverage": {1: {"80": 0.80}}})
    assert s["status"] == "FAIL" and s["high_side_80"] is True
    assert "biased WIDE" in s["detail"] and "33.3%" in s["detail"]
    assert s["calibration_population"]["prnd_share"] == 10 / 30
    assert "not registered" in s["calibration_population"]["reading"]
    lo = CF.s2c_coverage({"pair_conformal": {"80": 0.40, "95": 0.99}}, population={"prnd_share": 0.5})
    assert lo["status"] == "FAIL" and lo["high_side_80"] is False     # a LOW-side miss is not from the population
    ok = CF.s2c_coverage({"pair_conformal": {"80": 0.80, "95": 0.95}}, population={"prnd_share": 0.5})
    assert ok["status"] == "PASS" and ok["high_side_80"] is None


@pytest.mark.slow
def test_e7_the_v6_configuration_reading_and_c4s_control_transform_are_implemented(root, tmp_path, monkeypatch):
    """E7, CLOSED.  The seventh defect of the verification pass was two registered stages the fit loop did not do:

    * **section 3.4** -- *"It runs once, with the frozen configurations and the withheld seeds"*.  The reading the runner
      implements (``CF.READINGS['v6_frozen_configurations']``, ``RC.V6_FROZEN_CONFIGURATIONS``) is the sealed text's own:
      section 7 registers an inner design for V6, addendum 7 item 2 registers that a V6 job tunes on the V5 inner design,
      addendum 8 item 1 registers that the plan fits M1 on each V6 fold and that M2 takes M1's same-fold record.  So a
      V6 M1 fold IS tuned (measured here: ``tune_m1`` once, a grid of configurations over the inner folds), and its record
      carries the reading.  The V6 learned arms are priced at their tuned cost.
    * **section 11 / CONFIRMATION_PLAN** -- claim C4's control leg.  ``run_fold`` now applies ``h3.transformed_frame``,
      refits ``h3.FrozenB6`` off the WITH record of the same fold and binds the guard to the recorded corpus; the two
      legs have distinct record paths.  Exercised on a synthetic V2 fold in ``tests/test_confirmation_rehearsal.py``.
    """
    from gen19ct.models import neural as NN

    # ---- no stage of the single run is unimplemented, and the two readings are recorded where every reader looks
    assert [n for n, ok, _ in RC.RUN_STAGES if not ok] == []
    stages = {n: why for n, ok, why in RC.RUN_STAGES}
    assert stages["v6_frozen_configurations"] is RC.V6_FROZEN_CONFIGURATIONS
    assert stages["c4_act_permuted_training_transform"] is RC.C4_TRANSFORM
    for key in ("v6_frozen_configurations", "c4_transform", "c4_ln_test_set", "max_folds_pause"):
        assert key in CF.READINGS
    assert "addendum 7 item 2" in RC.V6_FROZEN_CONFIGURATIONS and "addendum 8 item 1" in RC.V6_FROZEN_CONFIGURATIONS
    assert "h3.FrozenB6" in RC.C4_TRANSFORM and "addendum 5 item 2" in RC.C4_TRANSFORM
    # the refusal machinery still works when asked, and says nothing is left to write
    assert "has NOT been spent" in RC.refusal("gates") and "Still to write: ." in RC.refusal("gates")

    # ---- C4's two legs: distinct paths; the control leg's job carries the transform the frozen branch reads
    plan = CF.read_plan(CF.plan_path(paths.G19_ROOT))
    c4 = [j for j in CF.enumerate_jobs(plan) if j.purpose.startswith("claim C4")][:2]
    assert [j.transform for j in c4] == ["WITH", "ACT_PERMUTED"] and [j.tuned for j in c4] == [True, False]
    specs = [RC.job_spec_of(j, 999983) for j in c4]
    assert [s_.condition for s_ in specs] == ["WITH", "ACT_PERMUTED"]
    assert specs[1].condition in H3.VALUE_PERMUTING_TRANSFORMS       # the branch run_fold takes for the control leg
    paths_ = [CF.fold_paths(Path("X"), s_.arm, s_.design_dir, 1, "f", s_.condition or "")[1] for s_ in specs]
    assert paths_[0] != paths_[1]
    c4claim = CF.claim_of(plan, "C4")
    assert c4claim.candidate == c4claim.comparator == "B6"
    assert (c4claim.transform, c4claim.comparator_transform) == ("WITH", "ACT_PERMUTED")
    # the V6 jobs: M1 and M2 tuned per the reading, B8 / B3x / B3i without tuning
    v6 = {j.arm: j.tuned for j in CF.enumerate_jobs(plan) if j.design == "V6" and j.seed_index == 1}
    assert v6 == {"M1": True, "M2": True, "B8": False, "B3x": False, "B3i": False}
    # ... and the cost basis prices them at the tuned cost and says so
    cost = CF.cost_estimate(CF.enumerate_jobs(plan))
    assert "132 h" in cost["basis"] and "TUNED" in cost["basis"]
    assert CF.UNIT_SECONDS["M1@V6__prnd__exact"] == CF.UNIT_SECONDS["M1@V5__primary__batched_max4"]
    assert 125 < cost["serial_hours"] < 140

    # ---- a V6 M1 fold on the synthetic corpus: tuned per section 7 (the reading), the record says which reading
    monkeypatch.setattr(REG, "registry_path", lambda r=None: root / "manifests" / "digest_registry.json")
    store = CF.load_seed_store(_write_store(root, root.parent / "e7_store.json"), root=root, seal=SEAL,
                               prereg_paths=_prereg_paths(root))
    code = RC.code_digest()["combined"]
    REG.register_stage(CF.STAGE, below_footer_sha256="a" * 64, code_digest=code, git_head=None, addenda_count=8,
                       note="e7", path=root / "manifests" / "digest_registry.json")
    corpus, fold, attrs = _v6_corpus(tmp_path)
    calls = {"tune_m1": 0, "frozen_runner": 0}

    def counted(name, fn):
        def g(*a, **k):
            calls[name] += 1
            return fn(*a, **k)
        return g

    monkeypatch.setattr(NN, "tune_m1", counted("tune_m1", NN.tune_m1))
    monkeypatch.setattr(H3, "frozen_runner", counted("frozen_runner", H3.frozen_runner))
    rd = RC.runner_module()
    r = RC.run_fold(D.JobSpec(kind="fit", arm="M1", design=CF.V6_DESIGN, variant=CF.V6_VARIANT_NAME,
                              scheme=CF.V6_SCHEME, seed=store.seed(1), writes=("M1",)), fold, 0, corpus,
                    tmp_path / "out", store=store, seed_index=1, runners=rd.default_runners(), code=code,
                    state=D.PlanState(guard_mode={}), prereg={"addenda_sha256": "a" * 64, "n_addenda": 8},
                    pair_attrs=attrs)
    rec = json.loads(Path(r["records"]["M1"]).read_text(encoding="utf-8"))
    assert r["status"] == "fitted"
    assert calls["tune_m1"] == 1 and calls["frozen_runner"] == 0     # section 7 tuning on the V6 fold, no frozen refit
    assert len({f.get("config") for f in rec["arm_record"]["fits"]}) > 1
    assert rec["v6_configuration_reading"] == CF.READINGS["v6_frozen_configurations"]
    assert rec["arm_record"]["inner_design"]["design"] == D.INNER_DESIGN_NAME       # the V5 inner design (addendum 7)
    assert rec["design_hash"] == corpus.design_hash(CF.V6_STEM)                    # the design hash is now recorded


# --------------------------------------------------------------------------------------------- #
# the gaps the third verification round left open: an M2 V5 CONFIRMATION-HALF fold end to end, a REAL
# two-process pool with overlapping per-fold clocks and a killed worker, and every raise of the build phase
# --------------------------------------------------------------------------------------------- #

#: the V5 cell the synthetic V5 fold hides: NOT the V6 system's Pr/Nd, so ``assert_v6_scope`` holds in both directions
V5_SMOKE_CELL = ("La(III)", "S2")


def _v5_corpus(tmp_path: Path, *, n_folds: int = 1) -> tuple[Any, list[FI.Fold]]:
    """The same mini-corpus as :func:`_v6_corpus` with ``n_folds`` **V5 primary batched_max4** folds whose scored rows
    are in the CONFIRMATION half -- the design claims C1-C3 are scored on.

    Each fold hides one cell (state x system) that is not the V6 system's Pr/Nd, so the V6 carve-out holds in both
    directions and the fold is one the confirmation run may fit.  The folds are written as a real design file, so
    ``Corpus.folds`` / ``Corpus.design_hash`` read them exactly as the run does.
    """
    df = _v6_frame()
    rd = RC.runner_module()
    is_v6 = (df[SG.SYSTEM_COL] == V6_SMOKE_SYSTEM) & df[SG.METAL_COL].isin(["Pr(III)", "Nd(III)"])
    v6 = pd.Series(is_v6.to_numpy(dtype=bool), index=df.index)
    coext = pd.Series(False, index=df.index)
    halves = {d: pd.Series("C", index=df.index) for d in ("V5", "V5P", "V5PAIR", "V1", "V2")}
    folds_dir = tmp_path / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)
    cells = [(st, sy) for sy in V6_SMOKE_SYSTEMS[1:] for st in V6_SMOKE_STATES][:n_folds]
    assert len(cells) == n_folds and all(not (sy == V6_SMOKE_SYSTEM) for _, sy in cells)
    folds = []
    for k, (state, system) in enumerate(cells):
        m = (df[SG.METAL_COL] == state) & (df[SG.SYSTEM_COL] == system)
        hid = df.loc[m, FI.ROW_ID].astype(str).tolist()
        # a withheld-seed colouring written by the run carries seed=None on every fold and an already-scrubbed id
        folds.append(FI.make_fold(design="V5", variant="primary", scheme="batched_max4",
                                  fold_id=f"si1_C_b{k:03d}", half="C", seed=None, hidden=hid, scored=hid,
                                  unit_type="cell", units=[f"{state} x {system}"], batch_id=f"b{k:03d}",
                                  row_unit={r: f"{state} x {system}" for r in hid},
                                  row_half={r: "C" for r in hid},
                                  meta={"cells": [[state, system]], "component_aware": True, "carved_out": False,
                                        "max_cells_per_batch": 4, "parent_structure": False}))
    FI.write_design(folds, folds_dir)
    corpus = rd.Corpus(frame=df, slim=df, table=I.RowTable(df), v6=v6, coext=coext, systems=None, comps=None, cv=None,
                       pmap=None, row_half=halves, folds_dir=folds_dir)
    return corpus, folds


@pytest.mark.slow
def test_m2_reads_m1s_record_on_a_v5_confirmation_half_fold_too(root, tmp_path, monkeypatch):
    """E1 on the design the frozen claims are scored on: an **M2 V5 confirmation-half** fold, end to end.

    The V6 leg is covered by ``test_m2_reads_m1s_confirmation_record_and_refuses_a_missing_or_stale_one``; claims C1-C3
    and S1(c) are M2 on V5 / V5-PAIR, which is 63 of the plan's 97 M2 hours, and no test exercised that path at all.
    POST-HOC addendum 8 item 1 is asserted where it bites: M2 takes the M1 record of the SAME confirmation fold, the same
    design file and the same withheld-seed INDEX -- never another index -- and stores that record's digest.
    """
    monkeypatch.setattr(REG, "registry_path", lambda r=None: root / "manifests" / "digest_registry.json")
    store_path = _write_store(root, root.parent / "v5_store.json")
    store = CF.load_seed_store(store_path, root=root, seal=SEAL, prereg_paths=_prereg_paths(root))
    code = RC.code_digest()["combined"]
    REG.register_stage(CF.STAGE, below_footer_sha256="a" * 64, code_digest=code, git_head=None, addenda_count=8,
                       note="v5 m2", path=root / "manifests" / "digest_registry.json")
    corpus, folds = _v5_corpus(tmp_path)
    fold = folds[0]
    rd = RC.runner_module()
    out_root, i = tmp_path / "out", 2
    kw = dict(store=store, seed_index=i, runners=rd.default_runners(), code=code, state=D.PlanState(guard_mode={}),
              prereg={"addenda_sha256": "a" * 64, "n_addenda": 8})

    def spec(arm):
        return D.JobSpec(kind="fit", arm=arm, design="V5", variant="primary", scheme="batched_max4",
                         seed=store.seed(i), writes=(arm,))

    # ---- before M1 exists: the refusal points at the CONFIRMATION layout and names no seed
    with pytest.raises(RuntimeError) as ei:
        RC.run_fold(spec("M2"), fold, 0, corpus, out_root, **kw)
    msg = str(ei.value)
    assert "needs M1's per-fold values" in msg and "records/M1/" in msg and CF.seed_token(i) in msg
    assert "evaluation/discovery" not in msg.replace("\\", "/")
    assert not any(str(s) in msg for s in FAKE_SEEDS)

    # ---- M1 then M2 on the confirmation half, end to end
    r1 = RC.run_fold(spec("M1"), fold, 0, corpus, out_root, **kw)
    r2 = RC.run_fold(spec("M2"), fold, 0, corpus, out_root, **kw)
    assert r1["status"] == r2["status"] == "fitted"
    m1rec = json.loads(Path(r1["records"]["M1"]).read_text(encoding="utf-8"))
    m2rec = json.loads(Path(r2["records"]["M2"]).read_text(encoding="utf-8"))
    assert m1rec["half"] == m2rec["half"] == D.CONFIRMATION and m2rec["seed_index"] == i
    ar = m2rec["arm_record"]
    assert ar["m1_config_used"]["emb_dim"] == m1rec["arm_record"]["selected"]["emb_dim"]
    assert ar["m1_config_used"]["weight_decay"] == m1rec["arm_record"]["selected"]["weight_decay"]
    assert ar["m1_record_digest"] == CF.confirmation_record_digest(m1rec)      # WHICH M1 fit, recorded
    assert m2rec["model_seed"] == m1rec["model_seed"] == rd.model_seed_of(0)   # the same fold's model seed (section 15)
    # the pairing block of addendum 8 item 1, on a REAL record: the digest, the configuration taken, the model seed
    # asserted equal to M1's, and the two stopping counts printed side by side because M2's is its own (section 7)
    pair = m2rec["m2_pairing"]
    assert pair["m1_record_digest"] == ar["m1_record_digest"] and pair["m1_model_seed"] == pair["m2_model_seed"]
    assert pair["m1_selected"] == m1rec["arm_record"]["selected"]
    assert pair["m1_n_epochs"] == m1rec["arm_record"]["n_epochs"]
    assert pair["m2_n_epochs"] == m2rec["arm_record"]["n_epochs"]
    assert pair["configuration_source"] == "the M1 record of this fold" and "NOT M1's" in pair["stopping_count_source"]
    assert "m2_pairing" not in m1rec                                          # M1 takes nothing from anybody
    # every scored prediction is a confirmation-half row and none is a V6 row
    pred = pd.read_parquet(Path(r2["records"]["M2"]).with_suffix(".parquet"))
    assert set(pred["half"]) == {D.CONFIRMATION}
    assert not corpus.v6.reindex(corpus.labels_of(list(pred["row_id"]))).any()
    assert not any(str(s) in json.dumps(m2rec, default=str) for s in FAKE_SEEDS)

    # ---- addendum 8 item 1: ANOTHER seed index is not a source.  The i2 M1 record exists; an i3 M2 fold refuses
    with pytest.raises(RuntimeError, match="needs M1's per-fold values"):
        RC.run_fold(spec("M2"), fold, 0, corpus, out_root, **{**kw, "seed_index": 3})
    # ---- and a DISCOVERY record of the same fold is not a source either: the locator reads records/M1 only
    dpq, djs = D.fold_paths(out_root, spec("M1"), "M1", fold.fold_id)
    djs.parent.mkdir(parents=True, exist_ok=True)
    djs.write_text(json.dumps(m1rec), encoding="utf-8")
    dpq.write_bytes(Path(r1["records"]["M1"]).with_suffix(".parquet").read_bytes())
    CF.fold_paths(out_root, "M2", spec("M2").design_dir, i, CF.scrub(fold.fold_id, store))[1].unlink()
    CF.fold_paths(out_root, "M1", spec("M1").design_dir, i, CF.scrub(fold.fold_id, store))[1].unlink()
    with pytest.raises(RuntimeError, match="needs M1's per-fold values"):
        RC.run_fold(spec("M2"), fold, 0, corpus, out_root, **kw)


#: the worker-side probe of the REAL pool test.  It lives in a file written to ``tmp_path`` because a spawned worker must
#: be able to import the function the parent submitted, and ``sys.path`` (which ``tmp_path`` is added to) is what
#: ``multiprocessing.spawn`` carries into the child.  It writes a marker per fold instead of fitting one: fitting a real
#: fold in a spawned worker would need the real corpus, and a real V6 or confirmation-half fit is the one thing this run
#: gets once.
POOL_PROBE = '''
"""Worker-side probe for the two-worker dispatch test: real processes, real clocks, no fit."""
import importlib.util, json, os, sys, time
from pathlib import Path

G19 = Path(os.environ["G19_ROOT"])


def _rc():
    if "g19_run_confirmation" in sys.modules:
        return sys.modules["g19_run_confirmation"]
    if str(G19) not in sys.path:
        sys.path.insert(0, str(G19))
    spec = importlib.util.spec_from_file_location("g19_run_confirmation", G19 / "scripts" / "g19_run_confirmation.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["g19_run_confirmation"] = m
    spec.loader.exec_module(m)
    return m


def init(store_path, out_root):
    """The initializer's contract: load and VERIFY the store in the worker, and leave the pre-flight self-check passing."""
    rc = _rc()
    sealm = rc.seal_module()
    cf = sys.modules["gen19ct.evaluation.confirmation"]
    store = cf.load_seed_store(store_path, root=Path(out_root), seal=sealm,
                               prereg_paths=sealm.PreregPaths(root=Path(out_root), repo_root=Path(out_root).parent))
    if not store.verified:
        raise SystemExit("refused: a worker's seed store does not verify")
    rc.CW.store = store
    rc.CW.corpus = type("C", (), {"table": type("T", (), {"n": 4242})()})()
    rc.CW.runners = {"probe": 1}
    rc.CW.attached, rc.CW.pair_attrs = None, None


def fold(spec, fold_id, ordinal, out_root, seed_index, code, state_rec, prereg, job_key):
    """``_conf_worker_fold``'s signature. Writes a marker, sleeps, and returns its own pid and clock."""
    marker = Path(out_root) / "markers" / f"{fold_id}.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        return {"status": "skipped_done", "job": job_key, "fold_id": fold_id, "pid": os.getpid()}
    if os.environ.get("G19_KILL_ON") == fold_id:
        os._exit(9)                      # the worker dies mid-fold, exactly as an OOM kill would
    t0 = time.time()
    time.sleep(0.8)
    t1 = time.time()
    marker.write_text(json.dumps({"fold_id": fold_id, "pid": os.getpid(), "t0": t0, "t1": t1}), encoding="utf-8")
    return {"status": "fitted", "job": job_key, "fold_id": fold_id, "pid": os.getpid(), "t0": t0, "t1": t1}
'''


@pytest.mark.slow
def test_two_real_workers_overlap_and_a_killed_worker_leaves_a_resumable_run(root, tmp_path, monkeypatch):
    """E2, the part the in-process pool cannot show: ``fit_loop`` at ``--workers 2`` really runs two folds at once, and a
    worker that dies leaves the finished folds on disk for ``--resume`` (POST-HOC addendum 8 item 2).

    ``test_fit_loop_dispatches_to_a_pool...`` drives the dispatch loop through an in-process stand-in and
    ``test_a_real_two_worker_pool...`` proves a spawned worker can verify the store and load the corpus; neither shows two
    folds in flight at the same wall-clock instant, which is the whole claim behind the 56 h wall figure the plan
    advertises.  Here the loop dispatches four folds to a REAL ``ProcessPoolExecutor`` of 2 and the per-fold clocks are
    compared: at least one pair overlaps, and at least two distinct pids did the work.
    """
    import os as _os
    from concurrent.futures.process import BrokenProcessPool

    (tmp_path / "poolprobe.py").write_text(POOL_PROBE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("G19_ROOT", str(paths.G19_ROOT))
    monkeypatch.delenv("G19_KILL_ON", raising=False)
    import poolprobe                                                    # noqa: PLC0415 -- written just above

    store_path = _write_store(root, root.parent / "realpool_store.json")
    store = CF.load_seed_store(store_path, root=root, seal=SEAL, prereg_paths=_prereg_paths(root))
    corpus, folds = _v5_corpus(tmp_path, n_folds=4)
    code = RC.code_digest()["combined"]
    monkeypatch.setattr(RC, "_conf_worker_fold", poolprobe.fold)
    monkeypatch.setattr(RC, "_init_conf_worker", poolprobe.init)
    monkeypatch.setattr(RC, "attach_seed_folds", lambda *a, **k: {})
    out_root = root                      # the worker verifies the store against root/manifests
    job = CF.ConfJob("real pool", "M1", "V5", "V5__primary__batched_max4", 1, len(folds))
    kw = dict(corpus=corpus, state=D.PlanState(guard_mode={}), runners={}, code=code,
              prereg={"addenda_sha256": "a" * 64, "n_addenda": 8}, workers=2, seed_store_path=str(store_path))

    # ---- four folds, two real workers
    led = RC.fit_loop([job], store, out_root, **kw)
    rec = led.record()
    rec = led.record()
    assert rec["workers_used"] == 2 and rec["n_folds_fitted"] == 4 and rec["n_errors"] == 0
    pre = rec["worker_preflight"]
    assert pre and all(p["store_verified"] and p["n_rows"] == 4242 for p in pre)
    assert _os.getpid() not in {p["pid"] for p in pre}                   # the pre-flight ran in the CHILDREN
    marks = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((out_root / "markers").glob("*.json"))]
    assert len(marks) == 4
    assert len({m["pid"] for m in marks}) >= 2, marks                    # two worker processes did the work
    overlaps = [(a["fold_id"], b["fold_id"]) for a in marks for b in marks
                if a["fold_id"] < b["fold_id"] and a["t0"] < b["t1"] and b["t0"] < a["t1"]]
    assert overlaps, marks                                               # ... AT THE SAME TIME
    t0 = min(m["t0"] for m in marks)
    print("per-fold clocks (pid, start, end, seconds, relative to the first start):")
    for m in sorted(marks, key=lambda m: m["t0"]):
        print(f'  {m["fold_id"]:14s} pid={m["pid"]:<7d} {m["t0"] - t0:6.3f} -> {m["t1"] - t0:6.3f}')
    print(f"  overlapping pairs: {overlaps}")
    assert not any(str(s) in json.dumps(marks) for s in FAKE_SEEDS)

    # ---- resume: the done folds are skipped, nothing is refitted
    led2 = RC.fit_loop([job], store, out_root, **kw)
    assert led2.record()["n_folds_skipped_already_done"] == 4 and led2.record()["n_folds_fitted"] == 0

    # ---- a worker KILLED mid-fold: the run stops, the completed folds survive, and a resume finishes the rest
    out2 = tmp_path / "killed"
    (out2 / "manifests").mkdir(parents=True, exist_ok=True)
    (out2 / "manifests" / "confirmation_seeds_sha256.txt").write_text(
        (root / "manifests" / "confirmation_seeds_sha256.txt").read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setenv("G19_KILL_ON", folds[3].fold_id)
    with pytest.raises(BrokenProcessPool):
        RC.fit_loop([job], store, out2, **kw)
    done = sorted(p.stem for p in (out2 / "markers").glob("*.json"))
    assert done and folds[3].fold_id not in done                         # the killed fold wrote nothing
    monkeypatch.delenv("G19_KILL_ON")
    led3 = RC.fit_loop([job], store, out2, **kw)
    r3 = led3.record()
    assert r3["n_folds_skipped_already_done"] == len(done) and r3["n_folds_fitted"] == len(folds) - len(done)
    assert len(list((out2 / "markers").glob("*.json"))) == len(folds)


#: the seed-shaped fold id every build-phase raise is driven with: exactly what a batched withheld-seed colouring carries
#: before ``write_seed_design`` scrubs it
#: the per-half statistics ``cell_holdout.v5_batched_folds`` / ``v5pair_batched_folds`` return, as the build phase
#: reads them (``n_batches`` / ``largest_batch`` / ``n_cells_moved_to_singleton`` per half)
_BATCH_STATS = {h: {"n_batches": 1, "largest_batch": 1, "n_cells_moved_to_singleton": 0} for h in ("S", "C")}


def _seedy(seed: int, k: int = 0) -> str:
    return f"s{seed}_C_b{k:03d}"


def _throw(exc):
    def f(*a, **k):
        raise exc
    return f


def _stub_fold_corpus(seed: int) -> Any:
    """A :class:`RC.FoldCorpus`-shaped stand-in whose builders are stubs, so each REAL raise of the build phase can be
    driven with a seed-shaped fold id without the 12 k-row corpus."""
    df = pd.DataFrame({FI.ROW_ID: ["a", "b", "c", "d"], SG.METAL_COL: ["La(III)", "Ce(III)", "Pr(III)", "Nd(III)"],
                       SG.SYSTEM_COL: ["S1", "S1", "S2", "S2"],
                       # build_v5pair_batched reads the publication group column off the frame before any stub is reached
                       FI.GROUP_COL: ["g1", "g1", "g2", "g2"]})
    return SimpleNamespace(frame=df, v6=pd.Series([False, False, True, True]),
                           halves=pd.Series("C", index=df.index), cells_tab=pd.DataFrame(),
                           condition_key=pd.Series("ck0", index=df.index),
                           book=SimpleNamespace(validate=lambda folds: None), builder=SimpleNamespace(),
                           pmap=None, v6_systems=["S1"], universe={"a", "b", "c", "d"})


def _batch_fold(seed: int, k: int, cells, *, carved: bool = False, cap: int = RC.MAX_CELLS_PER_BATCH) -> FI.Fold:
    return FI.make_fold(design="V5", variant="primary", scheme="batched_max4", fold_id=_seedy(seed, k), half="C",
                        seed=seed, hidden=["a"], scored=["a"], unit_type="cell", units=["u"], batch_id=f"b{k:03d}",
                        row_unit={"a": "u"}, row_half={"a": "C"},
                        meta={"cells": [list(c) for c in cells], "carved_out": carved, "max_cells_per_batch": cap,
                              "component_aware": True, "parent_structure": False})


@pytest.mark.slow
def test_every_raise_of_the_build_phase_is_scrubbed_and_writes_no_seed(root, tmp_path, monkeypatch):
    """E4 in full: **every** raise of the fold-build phase driven with a seed-shaped value.

    ``test_the_fold_build_phase_cannot_print_a_withheld_seed`` drives ONE stubbed raise. Addendum 6 item 4 is a property
    of the phase, not of one call site, so each raise is executed here -- the four per-batch assertions of
    ``build_v5_batched``, the book guard, a failure inside ``write_seed_design``, the three of ``build_v5pair_batched``
    including ``cell_holdout.check_pair_isolation``, ``build_v6_folds``' guard, the unknown-stem ``ValueError`` and
    ``assert_fold_v6_scope`` -- and after each one the message, the whole traceback and every file written under
    ``out_root`` are checked for all five placeholder seeds. The two helpers whose real messages carry the fold id
    (``folds.io.guard`` and ``check_pair_isolation``) are also called directly, to show that the raw text really does
    name the seed and that only the wrapper keeps it off the terminal.
    """
    import traceback as _tb

    store_path = _write_store(root, root.parent / "raise_store.json")
    store = CF.load_seed_store(store_path, root=root, seal=SEAL, prereg_paths=_prereg_paths(root))
    seed = store.seed(1)
    seeds = [str(s) for s in FAKE_SEEDS]
    assert str(seed) in seeds
    V5STEM, PAIRSTEM = "V5__primary__batched_max4", "V5PAIR__primary__batched"

    # ---- the REAL raises of the two helpers folds.io and cell_holdout own, with a seed-shaped fold id
    seedy_fold = _batch_fold(seed, 0, [("La(III)", "S1")])
    with pytest.raises(AssertionError, match="hidden ids missing from the frame") as g1:
        FI.guard(seedy_fold, pd.DataFrame({FI.ROW_ID: ["z"]}), pd.Index([0]), ["z"], "V5")
    assert str(seed) in str(g1.value)                    # the raw message DOES name the seed -- that is the hazard
    pair_fold = FI.make_fold(design="V5PAIR", variant="primary", scheme="batched", fold_id=_seedy(seed, 1), half="C",
                             seed=seed, hidden=["a"], scored=["a"], unit_type="cell_pair", units=["u"],
                             row_unit={"a": "u"}, row_half={"a": "C"}, meta={})
    with pytest.raises(AssertionError, match="not hidden-scored") as g2:
        # both members are TRAINING rows, so the fold-crossing level passes and the hidden-scored level -- the one whose
        # message is built from the fold id -- is the one that raises
        CH.check_pair_isolation(pair_fold, pd.DataFrame({"row_id_a": ["c"], "row_id_b": ["d"], "fold_id": ["x"]}),
                                ["a", "c", "d"])
    assert str(seed) in str(g2.value)

    # ---- each raise, driven through build_confirmation_folds, which is the only thing between it and the terminal
    cases: list[tuple] = []

    def case(name, setup, needle, kind=AssertionError, stem=V5STEM, with_v6=False):
        cases.append((name, setup, needle, kind, stem, with_v6))

    def v5_returns(folds, *, scored_cells=(("La(III)", "S1"),)):
        def setup(mp, fc):
            mp.setattr(fc.builder, "_variant_cells", lambda *a: (list(scored_cells), list(scored_cells), None),
                       raising=False)
            mp.setattr(CH, "cell_masks", lambda *a, **k: {})
            mp.setattr(CH, "EligibilitySpace", lambda *a, **k: None)
            mp.setattr(CH, "v5_batched_folds", lambda *a, **k: (folds, _BATCH_STATS))
            mp.setattr(CH, "guard_v5", lambda *a, **k: {"ok": True, "level": "V5"})
        return setup

    case("v5 cover", v5_returns([_batch_fold(seed, 0, [("Ce(III)", "S9")])]),
         "batches do not cover every scored cell exactly once")
    case("v5 two cells one state", v5_returns([_batch_fold(seed, 0, [("La(III)", "S1"), ("La(III)", "S2")])],
                                              scored_cells=(("La(III)", "S1"), ("La(III)", "S2"))),
         "two cells of one batch share a metal state or a system")
    case("v5 cap", v5_returns([_batch_fold(seed, 0, [("La(III)", "S1")], cap=99)]), "the section 7 item 6 cap")
    case("v5 carved out", v5_returns([_batch_fold(seed, 0, [("La(III)", "S1")], carved=True)]),
         "a carved-out cell was batched")

    def v5_guard_fails(mp, fc):
        v5_returns([_batch_fold(seed, 0, [("La(III)", "S1")])])(mp, fc)
        mp.setattr(CH, "guard_v5", lambda *a, **k: {"ok": False, "level": "V5"})
    case("v5 guard", v5_guard_fails, "a withheld-seed batch fails the V5 guard")

    def v5_book_rejects(mp, fc):
        v5_returns([_batch_fold(seed, 0, [("La(III)", "S1")])])(mp, fc)
        mp.setattr(fc.book, "validate", _throw(AssertionError(f"{_seedy(seed)}: the book rejects the batch")),
                   raising=False)
    case("v5 book", v5_book_rejects, "the book rejects the batch")

    def v5_write_fails(mp, fc):
        v5_returns([_batch_fold(seed, 0, [("La(III)", "S1")])])(mp, fc)
        mp.setattr(FI, "write_design", _throw(AssertionError(
            f"fold isolation violated in V5/primary/batched_max4/{_seedy(seed)}: dict")))
    case("v5 write_design", v5_write_fails, "fold isolation violated in")

    def pair_setup(*, units_ok=True, pairs_ok=True, isolation=True, guard=True):
        unit = FI.make_fold(design="V5PAIR", variant="primary", scheme="cell_pair", fold_id="u0", half="C", seed=None,
                            hidden=["a"], scored=["a"], unit_type="cell_pair", units=["u"],
                            row_unit={"a": "u"}, row_half={"a": "C"}, meta={"cells": [["La(III)", "S1"]]})
        batch = FI.make_fold(design="V5PAIR", variant="primary", scheme="batched", fold_id=_seedy(seed, 2), half="C",
                             seed=seed, hidden=["a"], scored=["a"], unit_type="cell_pair", units=["u"],
                             row_unit={"a": "u"}, row_half={"a": "C"},
                             meta={"cells": [["La(III)", "S1"]], "unit_fold_ids": ["u0" if units_ok else "u9"]})
        ptab = pd.DataFrame({"fold_id": [batch.fold_id], "unit_fold_id": ["u0" if pairs_ok else "u9"],
                             "row_id_a": ["a"], "row_id_b": ["b"]})
        upairs = pd.DataFrame({"fold_id": ["u0"], "row_id_a": ["a"], "row_id_b": ["b"]})

        def setup(mp, fc):
            mp.setattr(FI, "read_design", lambda *a, **k: [unit])
            mp.setattr(RC.pd, "read_parquet", lambda *a, **k: upairs)
            mp.setattr(CH, "cell_masks", lambda *a, **k: {})
            mp.setattr(CH, "EligibilitySpace", lambda *a, **k: None)
            mp.setattr(CH, "v5pair_batched_folds",
                       lambda *a, **k: ([batch], ptab, {**_BATCH_STATS, "n_test_test_row_pairs": 1}))
            mp.setattr(FI, "design_hash", lambda *a, **k: "0" * 64)
            mp.setattr(CH, "check_pair_isolation", (lambda *a, **k: None) if isolation else
                       _throw(AssertionError(f"{_seedy(seed, 2)}: 1 pair(s) with a member that is not hidden-scored")))
            mp.setattr(CH, "guard_v5", lambda *a, **k: {"ok": bool(guard), "level": "V5"})
        return setup

    case("pair units", pair_setup(units_ok=False), "batches do not cover every unit fold exactly once", stem=PAIRSTEM)
    case("pair list", pair_setup(pairs_ok=False), "the batch pair list is not the union", stem=PAIRSTEM)
    case("pair isolation", pair_setup(isolation=False), "not hidden-scored", stem=PAIRSTEM)
    case("pair guard", pair_setup(guard=False), "fails the V5 guard", stem=PAIRSTEM)
    case("unknown stem", lambda mp, fc: None, "no registered builder for stem", CF.RedactedRunError, "V9__nope__exact")

    def v6_guard_fails(mp, fc):
        v5_returns([_batch_fold(seed, 0, [("La(III)", "S1")])])(mp, fc)
        mp.setattr(RC, "build_v6_folds", _throw(AssertionError(f"V6: a fold fails the V5 guard ({_seedy(seed)})")))
    case("v6 guard", v6_guard_fails, "V6: a fold fails the V5 guard", AssertionError, V5STEM, True)

    def scope_fails(mp, fc):
        v5_returns([_batch_fold(seed, 0, [("La(III)", "S1")])])(mp, fc)
        # assert_fold_v6_scope runs AFTER the design file is written, so this case reaches write_seed_design; the writer
        # is stubbed (the stub corpus carries none of folds.io's FRAME_COLUMNS) and the ONE thing under test here is that
        # the scope assertion's seed-shaped message is scrubbed
        def _write(sc, d, extra=None):
            Path(d).mkdir(parents=True, exist_ok=True)
            (Path(d) / "design.json").write_text(json.dumps({"extra": extra}, default=str), encoding="utf-8")
            return (Path(d) / "design.parquet", Path(d) / "design.json",
                    {"design_hash": "0" * 64, "assignment_sha256": "1" * 64, "n_scored_units": 1,
                     "n_distinct_scored_rows": 1, "by_half": {"C": {"n_folds": 1}}})
        mp.setattr(FI, "write_design", _write)
        mp.setattr(RC, "assert_fold_v6_scope",
                   _throw(AssertionError(f"{_seedy(seed)}: a V6 row is scored outside the V6 job")))
    case("v6 scope", scope_fails, "a V6 row is scored outside the V6 job")

    for k, (name, setup, needle, kind, stem, with_v6) in enumerate(cases):
        out = tmp_path / f"case{k}"
        out.mkdir(parents=True, exist_ok=True)
        with monkeypatch.context() as mp:
            fc = _stub_fold_corpus(seed)
            mp.setattr(RC, "load_fold_corpus", lambda fc=fc: fc)
            mp.setattr(RC, "build_v6_folds", lambda c: ([], {}))
            setup(mp, fc)
            with pytest.raises(kind) as ei:
                RC.build_confirmation_folds(store, out, stems=(stem,), with_v6=with_v6)
        txt = str(ei.value)
        tb = "".join(_tb.format_exception(ei.value))
        assert needle in txt, (name, txt)
        assert "building the withheld-seed fold files" in txt, (name, txt)
        assert not any(s in txt for s in seeds), (name, txt)                  # the MESSAGE is scrubbed
        assert not any(s in tb for s in seeds), (name, tb)                    # ... and so is the whole traceback
        assert ei.value.__context__ is None or ei.value.__suppress_context__  # no chained frame carries it either
        if "_C_b" in txt:
            assert CF.seed_token(1) + "_C_b" in txt, (name, txt)              # the opaque index replaced the seed
        leaked = []
        for p in out.rglob("*"):
            if not p.is_file():
                continue
            body = p.read_bytes().decode("utf-8", "replace")
            if any(s in body or s in str(p) for s in seeds):
                leaked.append(str(p))
        assert not leaked, (name, leaked)                                     # no WRITTEN file carries one either
    assert len(cases) == 14


@pytest.mark.slow
def test_a_resume_refuses_a_fold_whose_record_carries_another_code_digest(root, tmp_path, monkeypatch):
    """The remaining half of POST-HOC addendum 8 item 2, at the level the addendum states it.

    The addendum: ``--resume`` "may only fit folds that are missing and must refuse any fold whose stored record was
    written under a different code digest than the one now registered for the ``confirmation`` stage". Two guards existed
    -- the run-level lock (the START marker's digest against the live one) and ``read_seed_predictions`` at scoring time
    -- and neither covers the case the addendum names: a fold record under digest B while the marker records A == live.
    ``run_fold`` returned ``skipped_done`` on the mere existence of the JSON, before ``refuse_unless_writable`` even ran,
    so such a fold was silently kept and the run died 56 h later at the scoring gate instead of at dispatch.
    """
    monkeypatch.setattr(REG, "registry_path", lambda r=None: root / "manifests" / "digest_registry.json")
    store = CF.load_seed_store(_write_store(root, root.parent / "resume_store.json"), root=root, seal=SEAL,
                               prereg_paths=_prereg_paths(root))
    code = RC.code_digest()["combined"]
    REG.register_stage(CF.STAGE, below_footer_sha256="a" * 64, code_digest=code, git_head=None, addenda_count=8,
                       note="resume", path=root / "manifests" / "digest_registry.json")
    corpus, folds = _v5_corpus(tmp_path)
    fold, out_root, i = folds[0], tmp_path / "out", 1
    spec = D.JobSpec(kind="fit", arm="M1", design="V5", variant="primary", scheme="batched_max4",
                     seed=store.seed(i), writes=("M1",))
    kw = dict(store=store, seed_index=i, runners=RC.runner_module().default_runners(), code=code,
              state=D.PlanState(guard_mode={}), prereg={"addenda_sha256": "a" * 64, "n_addenda": 8})

    # the fold is fitted once, then a second invocation SKIPS it: the resume contract, unchanged
    assert RC.run_fold(spec, fold, 0, corpus, out_root, **kw)["status"] == "fitted"
    assert RC.run_fold(spec, fold, 0, corpus, out_root, **kw)["status"] == "skipped_done"

    # ... but a record written under ANOTHER code digest is refused rather than skipped, and named
    js = CF.fold_paths(out_root, "M1", spec.design_dir, i, CF.scrub(fold.fold_id, store))[1]
    rec = json.loads(js.read_text(encoding="utf-8"))
    js.write_text(json.dumps({**rec, "code_digest": "b7" * 32}), encoding="utf-8")
    with pytest.raises(CF.RedactedRunError) as ei:
        RC.run_fold(spec, fold, 0, corpus, out_root, **kw)
    msg = str(ei.value)
    assert "addendum 8 item 2" in msg and "b7b7b7b7b7b7" in msg and code[:12] in msg
    assert "complete folds MISSING" in msg
    assert not any(str(s) in msg for s in FAKE_SEEDS)              # the label is the scrubbed one
    # a record with no code digest at all is refused the same way, and the run is not allowed to guess
    js.write_text(json.dumps({k: v for k, v in rec.items() if k != "code_digest"}), encoding="utf-8")
    with pytest.raises(CF.RedactedRunError, match="no code digest"):
        RC.run_fold(spec, fold, 0, corpus, out_root, **kw)
    # through the fit loop it REFUSES -- it is not an AssertionError, so it is not filed as an incompletion and the run
    # stops, which is what "must refuse" means (addendum 8 item 2)
    js.write_text(json.dumps({**rec, "code_digest": "b7" * 32}), encoding="utf-8")
    job = CF.ConfJob("resume", "M1", "V5", "V5__primary__batched_max4", i, 1)
    monkeypatch.setattr(RC, "attach_seed_folds", lambda *a, **k: {})
    loop = dict(corpus=corpus, state=D.PlanState(guard_mode={}), runners=RC.runner_module().default_runners(),
                code=code, prereg={"addenda_sha256": "a" * 64, "n_addenda": 8}, workers=1)
    with pytest.raises(CF.RedactedRunError) as e2:
        RC.fit_loop([job], store, out_root, **loop)
    assert "addendum 8 item 2" in str(e2.value) and not any(str(s) in str(e2.value) for s in FAKE_SEEDS)

    # ---- and the OTHER half of the same vocabulary bug: a stale M1 record is INCOMPLETE_STALE_RECORD, never the guard's
    js.write_text(json.dumps(rec), encoding="utf-8")                       # M1 restored under the live digest
    m2 = D.JobSpec(kind="fit", arm="M2", design="V5", variant="primary", scheme="batched_max4",
                   seed=store.seed(i), writes=("M2",))
    assert RC.run_fold(m2, fold, 0, corpus, out_root, **kw)["status"] == "fitted"
    CF.fold_paths(out_root, "M2", m2.design_dir, i, CF.scrub(fold.fold_id, store))[1].unlink()
    js.write_text(json.dumps({**rec, "fold_hash": "0" * 40}), encoding="utf-8")   # M1's record no longer verifies
    with pytest.raises(D.StaleRecordError):
        RC.run_fold(m2, fold, 0, corpus, out_root, **kw)
    led = RC.fit_loop([CF.ConfJob("resume", "M2", "V5", "V5__primary__batched_max4", i, 1)], store, out_root, **loop)
    errs = led.record()["errors"]
    assert [e["status"] for e in errs] == [CF.INCOMPLETE_STALE_RECORD]           # NOT INCOMPLETE_GUARD_FAILURE
    assert "fold_hash" in errs[0]["detail"] and not any(str(s) in json.dumps(errs) for s in FAKE_SEEDS)
    assert CF.INCOMPLETE_STALE_RECORD in led.record()["guard_failure_vocabulary"]
    # and the record that caused each refusal is still on disk exactly as it was: nothing is deleted or refitted
    assert json.loads(js.read_text(encoding="utf-8"))["fold_hash"] == "0" * 40
    assert not CF.fold_paths(out_root, "M2", m2.design_dir, i, CF.scrub(fold.fold_id, store))[1].exists()


# --------------------------------------------------------------------------------------------- #
# TASK X, round four: what the fourth verification round found by execution -- a missing M1 record ended
# the RUN instead of the M2 FOLD, a finished run was resumable, C4's two legs shared one record path, and
# the audited M1 pairing did not pin the values M2 consumes
# --------------------------------------------------------------------------------------------- #

@pytest.mark.slow
def test_a_missing_m1_record_ends_the_m2_fold_not_the_run(root, tmp_path, monkeypatch):
    """POST-HOC addendum 8 item 1: "If that M1 record is absent ... the **M2 fold** is INCOMPLETE and reported as such".

    Measured before the fix: ``m1_record_locator.locate`` raised a bare ``RuntimeError``, which the fit loop's defect
    arm re-raised as ``RedactedRunError`` -- the ledger never recorded the fold and the run STOPPED.  Since
    ``enumerate_jobs`` puts M1 before M2 at each seed index, one M1 fold lost to a guard failure (14 such folds happened
    in H3) killed the whole 56 h run at its M2 sibling, and ``--resume`` died at the same fold for ever, because a code
    fix changes the confirmation digest and addendum 8 item 2 then refuses every record already written.
    """
    monkeypatch.setattr(REG, "registry_path", lambda r=None: root / "manifests" / "digest_registry.json")
    store = CF.load_seed_store(_write_store(root, root.parent / "m1_missing_store.json"), root=root, seal=SEAL,
                               prereg_paths=_prereg_paths(root))
    code = RC.code_digest()["combined"]
    REG.register_stage(CF.STAGE, below_footer_sha256="a" * 64, code_digest=code, git_head=None, addenda_count=8,
                       note="m1 missing", path=root / "manifests" / "digest_registry.json")
    corpus, folds = _v5_corpus(tmp_path)
    fold, out_root, i = folds[0], tmp_path / "out", 1
    monkeypatch.setattr(RC, "attach_seed_folds", lambda *a, **k: {})
    loop = dict(corpus=corpus, state=D.PlanState(guard_mode={}), runners=RC.runner_module().default_runners(),
                code=code, prereg={"addenda_sha256": "a" * 64, "n_addenda": 8}, workers=1)
    m2 = CF.ConfJob("claims C1-C3 (core)", "M2", "V5", "V5__primary__batched_max4", i, 1)
    m1 = CF.ConfJob("claims C1-C3 (core)", "M1", "V5", "V5__primary__batched_max4", i, 1)

    # the M2 job whose M1 record is missing, and a LATER job in the same loop: the fold is incomplete, the run goes on
    led = RC.fit_loop([m2, m1], store, out_root, **loop)
    errs = led.record()["errors"]
    assert [e["status"] for e in errs] == [CF.INCOMPLETE_M1_RECORD_MISSING]
    assert errs[0]["fold_id"] == CF.scrub(fold.fold_id, store) and "records/M1/" in errs[0]["detail"]
    assert led.record()["n_folds_fitted"] == 1                           # the M1 job after it still ran
    assert CF.fold_paths(out_root, "M1", "V5__primary_batched_max4", i, CF.scrub(fold.fold_id, store))[1].exists()
    assert not any(str(s) in json.dumps(led.record()) for s in FAKE_SEEDS)

    # ... and once M1's record exists, the same M2 fold fits: the incompletion is of the fold, and it is recoverable
    led2 = RC.fit_loop([m2], store, out_root, **loop)
    assert led2.record()["errors"] == [] and led2.record()["n_folds_fitted"] == 1

    # the WORKER path classifies it identically (the parent maps the status through ``absorb``)
    monkeypatch.setattr(RC.CW, "store", store, raising=False)
    monkeypatch.setattr(RC.CW, "corpus", corpus, raising=False)
    monkeypatch.setattr(RC.CW, "attached", i, raising=False)
    monkeypatch.setattr(RC.CW, "runners", {}, raising=False)

    def absent(*a, **k):
        raise CF.MissingSiblingRecordError(f"M2@V5/i{i}/f: M2 needs M1's per-fold values of this fold")

    monkeypatch.setattr(RC, "run_fold", absent)
    spec = D.JobSpec(kind="fit", arm="M2", design="V5", variant="primary", scheme="batched_max4",
                     seed=store.seed(i), writes=("M2",))
    r = RC._conf_worker_fold(spec, fold.fold_id, 0, str(out_root), i, code, D.PlanState(guard_mode={}).record(),
                             None, "M2@V5__primary__batched_max4/i1")
    assert r["status"] == "m1_record_missing" and not any(str(s) in json.dumps(r) for s in FAKE_SEEDS)
    # ... and a MissingSiblingRecordError is a RuntimeError, so no caller of the locator sees a different type
    assert issubclass(CF.MissingSiblingRecordError, RuntimeError)


def test_a_finished_run_is_never_resumable(root, plan_file, registry):
    """POST-HOC addendum 8 item 2: "Once ``confirmation.json`` exists the run is spent and the runner refuses."

    ``lock_verdict`` returned ``{"ok": True, "mode": "resume"}`` for a FINISHED run whenever the code digest and the
    claim list matched, so a post-completion ``--resume`` passed the gate, re-ran assembly and overwrote
    ``confirmation.json``, the tables and ``CONFIRMATION.md`` -- a second scoring of the single registered run.
    Resumption belongs to the STARTED marker, which is the run that did not finish.
    """
    code = RC.code_digest()["combined"]
    CF.write_decisions(root, {"code_digest": code, "claim_ids": ["C1", "C4"]})
    for resume in (False, True):
        v = CF.lock_verdict(root, resume=resume, code_digest=code, claim_ids=["C1", "C4"])
        assert v["ok"] is False and v["mode"] == ("already_run_resume_refused" if resume else "already_run")
        assert "spent" in v["reason"] and "addendum 8 item 2" in v["reason"]
    store_path = _write_store(root, root.parent / "spent_store.json")
    with pytest.raises(SystemExit, match="spent"):
        _gate(root, store_path, resume=True, registry_path=registry)
    # the file is not rewritten by the refusal
    assert json.loads(CF.decisions_path(root).read_text(encoding="utf-8"))["claim_ids"] == ["C1", "C4"]

    # ---- the UNFINISHED run is the one --resume may complete, and its two refusals still bite there
    CF.decisions_path(root).unlink()
    from gen19ct.manifest import write_json as _wj
    marker = CF.started_path(root)
    marker.parent.mkdir(parents=True, exist_ok=True)
    _wj(marker, {"code_digest": code, "claim_ids": ["C1", "C4"], "started_utc": "2026-09-24T00:00:00+00:00"})
    assert CF.lock_verdict(root, resume=False, code_digest=code, claim_ids=["C1"])["mode"] == "already_started"
    ok = CF.lock_verdict(root, resume=True, code_digest=code, claim_ids=["C1", "C4"])
    assert ok["ok"] is True and ok["mode"] == "resume_unfinished"
    assert CF.lock_verdict(root, resume=True, code_digest="y" * 64,
                           claim_ids=["C1"])["mode"] == "resume_refused_code_changed"
    assert CF.lock_verdict(root, resume=True, code_digest=code,
                           claim_ids=["C1", "C9"])["mode"] == "resume_refused_claims_grew"


def test_the_m1_pairing_digest_pins_the_values_m2_consumes(root, tmp_path, monkeypatch):
    """POST-HOC addendum 8 item 1: "M2's record stores the M1 record's digest so the pairing is auditable."

    ``RECORD_IDENTITY_FIELDS`` covered the top-level ``selected_config`` LABEL, while ``NeuralRunner.point`` consumes
    ``arm_record.selected``.  Measured: tampering M1's ``arm_record.selected`` from emb_dim 8 / wd 0.01 to 16 / 0.0001
    left ``m1_record_digest`` identical, and M2 adopted the tampered values -- the audit named a pairing that was not
    the one used.
    """
    rec = {"schema": CF.SCHEMA, "registry_stage": CF.STAGE, "arm": "M1", "stem": "V5__primary__batched_max4",
           "fold_id": "si1_C_b000", "fold_hash": "h" * 40, "seed_index": 1, "code_digest": "c" * 64,
           "prereg_sha256": "p" * 64, "prereg_addenda_sha256": "a" * 64, "model_seed": 10000033,
           "selected_config": RC.configuration_label({"emb_dim": 8, "weight_decay": 0.01, "rank": 0}),
           "arm_record": {"selected": {"emb_dim": 8, "weight_decay": 0.01, "rank": 0}, "n_epochs": 166,
                          "model_seed": 10000033, "inner_folds_used": [0, 1, 2]}}
    base = CF.confirmation_record_digest(rec)
    for tamper in ({"emb_dim": 16, "weight_decay": 0.01, "rank": 0},
                   {"emb_dim": 8, "weight_decay": 0.0001, "rank": 0}):
        moved = {**rec, "arm_record": {**rec["arm_record"], "selected": tamper}}
        assert CF.confirmation_record_digest(moved) != base
    for key, value in (("n_epochs", 999), ("model_seed", 42), ("inner_folds_used", [0, 1])):
        moved = {**rec, "arm_record": {**rec["arm_record"], key: value}}
        assert CF.confirmation_record_digest(moved) != base, key

    # ... and the locator refuses a record whose label and whose nested configuration disagree, rather than adopting one
    monkeypatch.setattr(REG, "registry_path", lambda r=None: root / "manifests" / "digest_registry.json")
    store = CF.load_seed_store(_write_store(root, root.parent / "pin_store.json"), root=root, seal=SEAL,
                               prereg_paths=_prereg_paths(root))
    REG.register_stage(CF.STAGE, below_footer_sha256="a" * 64, code_digest="c" * 64, git_head=None, addenda_count=8,
                       note="pin", path=root / "manifests" / "digest_registry.json")
    fold = FI.make_fold(design="V5", variant="primary", scheme="batched_max4", fold_id="si1_C_b000", half="C",
                        seed=None, hidden=["a"], scored=["a"], unit_type="cell", units=["u"], batch_id="b000",
                        row_unit={"a": "u"}, row_half={"a": "C"}, meta={})
    rec["fold_hash"] = fold.fold_hash
    out_root = tmp_path / "pin"
    pq, js = CF.fold_paths(out_root, "M1", "V5__primary_batched_max4", 1, "si1_C_b000")
    js.parent.mkdir(parents=True, exist_ok=True)
    mjob = D.JobSpec(kind="fit", arm="M1", design="V5", variant="primary", scheme="batched_max4",
                     seed=store.seed(1), writes=("M1",))
    loc = RC.m1_record_locator(out_root, seed_index=1, code="c" * 64, store=store, fold=fold, what="M2@V5/i1/f")
    good = {**rec, "steps": {RC.POINT_STEP: {"seconds": 1.0}}}
    js.write_text(json.dumps(good), encoding="utf-8")
    pq.write_bytes(b"not a parquet, but it exists")
    assert loc(mjob, "M1")["digest"] == CF.confirmation_record_digest(good)
    js.write_text(json.dumps({**good, "arm_record": {**good["arm_record"],
                                                     "selected": {"emb_dim": 16, "weight_decay": 0.0001, "rank": 0}}}),
                  encoding="utf-8")
    with pytest.raises(D.StaleRecordError, match="internally inconsistent"):
        loc(mjob, "M1")
    js.write_text(json.dumps({**good, "arm_record": {"n_epochs": 1}}), encoding="utf-8")
    with pytest.raises(D.StaleRecordError, match="arm_record.selected"):
        loc(mjob, "M1")

    # ---- and the pairing block: the model seed is ASSERTED to be M1's, the stopping count is recorded as M2's own
    m1 = {**good, "digest": "d" * 64}
    pairing = RC.m2_pairing(m1, {"model_seed": 10000033, "arm_record": {"n_epochs": 140}}, what="M2@V5/i1/f")
    assert pairing["m1_record_digest"] == "d" * 64 and pairing["m1_n_epochs"] == 166 and pairing["m2_n_epochs"] == 140
    assert "NOT M1's" in pairing["stopping_count_source"] and "section 7" in pairing["reading"]
    with pytest.raises(D.StaleRecordError, match="model seed"):
        RC.m2_pairing(m1, {"model_seed": 99, "arm_record": {"n_epochs": 140}}, what="M2@V5/i1/f")


def test_c4s_two_legs_can_never_share_one_record(root):
    """Measured: ``job_spec_of`` puts the transform in ``JobSpec.condition``, and ``design_dir`` / ``fold_paths``
    ignored it, so ``B6:WITH`` and ``B6:ACT_PERMUTED`` both resolved to
    ``records/B6/V2__element_exact/i1/f000.json`` -- the second leg returned ``skipped_done`` and C4's delta was
    exactly 0.0 on all five seeds, an artefact of the path reported as a failed R19 item 4."""
    legs = [CF.ConfJob("C4", "B6", "V2", "V2__element__exact", 1, 1, transform=t)
            for t in ("WITH", "ACT_PERMUTED")]
    specs = [RC.job_spec_of(j, 111111) for j in legs]
    ps = [CF.fold_paths(Path("X"), s.arm, s.design_dir, 1, "f000", s.condition or "")[1] for s in specs]
    assert ps[0] != ps[1] and [p.parent.parent.parent.name for p in ps] == ["WITH", "ACT_PERMUTED"]
    assert [p.parent.parent.parent.parent.name for p in ps] == ["B6", "B6"]        # <arm>/<transform>/<design>/i<k>
    # an arm with no transform keeps the path it had: no other record moves
    plain = CF.fold_paths(Path("X"), "M1", "V5__primary_batched_max4", 1, "f000")
    assert plain[1].parent.parent.name == "V5__primary_batched_max4" and plain[1].parent.parent.parent.name == "M1"
    # the inventory is checked BEFORE the gates: a colliding one refuses, naming both jobs
    with pytest.raises(AssertionError, match="more than one job"):
        CF.assert_distinct_record_dirs([("B6", "V2__element_exact", 1, ""), ("B6", "V2__element_exact", 1, "")])
    assert RC.assert_inventory_paths(legs) == ["B6/WITH/V2__element_exact/i1", "B6/ACT_PERMUTED/V2__element_exact/i1"]
    # ... and the REAL inventory of the frozen plan has one directory per job
    plan = CF.read_plan(CF.plan_path(paths.G19_ROOT))
    jobs = CF.enumerate_jobs(plan)
    keys = RC.assert_inventory_paths(jobs)
    assert len(keys) == len(set(keys)) == len(jobs)


def test_a_refusal_says_why_the_record_does_not_verify(root):
    """``registry.verify_record`` sets ``reason`` only for the stale-after-supersession case, so every message built as
    ``f"{path}: {v.get('reason')}"`` printed ``<path>: None`` -- the mixed-code refusal of a 56 h run naming nothing."""
    v = {"ok": False, "stage": CF.STAGE, "mismatches": {"code_digest": ("aa" * 32, "bb" * 32)},
         "registry_entry": {"code_digest": "bb" * 32}}
    msg = RC.verification_reason(v)
    assert "code_digest" in msg and "aaaaaaaaaaaa" in msg and "bbbbbbbbbbbb" in msg and "None" not in msg
    assert RC.verification_reason({"ok": None, "stage": "confirmation", "mismatches": {}}) == \
        "no registry entry for stage 'confirmation'"
    assert RC.verification_reason({"ok": False, "reason": "written after supersession"}) == "written after supersession"


def test_the_dispatch_preamble_of_the_fit_loop_cannot_print_a_withheld_seed(root, monkeypatch):
    """The E4 class of defect one phase later: the fit loop's preamble holds a SEEDED ``JobSpec`` and had no handler.

    ``attach_seed_folds``, ``fold_seeded_spec`` and ``confirmation_fittable_folds`` are called between the pool's
    ``try`` and its ``finally``; ``confirmation_job_folds`` raises ``f"{job.key}: fold file ... holds seeds ..."`` and
    ``job.key`` of a confirmation ``JobSpec`` carries the withheld seed, so such an exception reached ``main`` raw --
    message and frames -- which POST-HOC addendum 6 item 4 forbids.  Every one of the three is now scrubbed.
    """
    store_path = _write_store(root, root.parent / "preamble_store.json")
    store = CF.load_seed_store(store_path, root=root, seal=SEAL, prereg_paths=_prereg_paths(root))
    seed = store.seed(1)

    class _C:
        coext_ids: list = []

        def __init__(self):
            self.guard_cache: dict = {}
            self._folds: dict = {}

        def folds(self, stem):
            return [_fold("f_c", "C", {"a": "C"})]

    job = CF.ConfJob("p", "M2", "V5", "V5__primary__batched_max4", 1, 1)
    loop = dict(corpus=_C(), state=D.PlanState(), runners={}, code="c", prereg={})

    # (a) the REAL raise of the selector, on the real message it builds from a seeded job.key
    def seedy_selector(spec, folds, excluded):
        raise ValueError(f"{spec.key}: fold file {spec.stem} holds seeds ['{seed}', 'None']; give fold_seed")

    monkeypatch.setattr(RC, "attach_seed_folds", lambda *a, **k: {})
    monkeypatch.setattr(CF, "confirmation_fittable_folds", seedy_selector)
    with pytest.raises(CF.RedactedRunError) as ei:
        RC.fit_loop([job], store, root, **loop)
    msg = str(ei.value)
    assert str(seed) not in msg and "i1" in msg and ei.value.__cause__ is None and ei.value.__suppress_context__
    assert not any(str(s) in msg for s in FAKE_SEEDS)
    assert "ValueError" in msg                                    # the class is named, the seed is not

    # (b) the spec builder and (c) the fold attachment, on the same contract
    monkeypatch.setattr(CF, "confirmation_fittable_folds", lambda *a, **k: [])
    monkeypatch.setattr(RC, "fold_seeded_spec", lambda *a, **k: (_ for _ in ()).throw(KeyError(f"s{seed}")))
    with pytest.raises(CF.RedactedRunError) as e2:
        RC.fit_loop([job], store, root, **loop)
    assert not any(str(s) in str(e2.value) for s in FAKE_SEEDS) and "i1" in str(e2.value)
    monkeypatch.undo()

    monkeypatch.setattr(RC, "attach_seed_folds",
                        lambda *a, **k: (_ for _ in ()).throw(OSError(f"cannot read folds/i1/s{seed}.json")))
    with pytest.raises(CF.RedactedRunError) as e3:
        RC.fit_loop([job], store, root, **loop)
    assert not any(str(s) in str(e3.value) for s in FAKE_SEEDS) and "i1" in str(e3.value)


def test_a_record_set_the_loop_left_incomplete_is_never_scored_on_the_folds_it_did_write(root, tmp_path, monkeypatch):
    """The consequence of making a missing M1 record incomplete instead of fatal, closed at the reader.

    While an incomplete M2 fold ENDED the run, no partial record set could reach the assembly.  Now the run continues,
    so ``read_seed_predictions`` would read the directory, find the folds that WERE written, and
    ``claim_paired_units`` would intersect the two arms and score the claim on fewer units -- the "verdict from part of
    a design" POST-HOC addendum 6 item 3(a) forbids.  The loop names what it failed to write and the reader refuses it.
    """
    monkeypatch.setattr(REG, "registry_path", lambda r=None: root / "manifests" / "digest_registry.json")
    store = CF.load_seed_store(_write_store(root, root.parent / "partial_store.json"), root=root, seal=SEAL,
                               prereg_paths=_prereg_paths(root))
    code = RC.code_digest()["combined"]
    REG.register_stage(CF.STAGE, below_footer_sha256="a" * 64, code_digest=code, git_head=None, addenda_count=8,
                       note="partial", path=root / "manifests" / "digest_registry.json")
    corpus, folds = _v5_corpus(tmp_path, n_folds=2)
    out_root, i = tmp_path / "out", 1
    monkeypatch.setattr(RC, "attach_seed_folds", lambda *a, **k: {})
    monkeypatch.setattr(RC, "INCOMPLETE_RECORD_SETS", set())
    loop = dict(corpus=corpus, state=D.PlanState(guard_mode={}), runners=RC.runner_module().default_runners(),
                code=code, prereg={"addenda_sha256": "a" * 64, "n_addenda": 8}, workers=1)
    m1 = CF.ConfJob("claims C1-C3 (core)", "M1", "V5", "V5__primary__batched_max4", i, 2)

    # one of the two folds fails its guard; the other is written
    real = RC.run_fold

    def one_bad(job, fold, ordinal, *a, **k):
        if fold.fold_id.endswith("b001"):
            raise AssertionError(f"{job.arm}: the guard of this fold failed")
        return real(job, fold, ordinal, *a, **k)

    monkeypatch.setattr(RC, "run_fold", one_bad)
    led = RC.fit_loop([m1], store, out_root, **loop)
    assert [e["status"] for e in led.errors] == [CF.INCOMPLETE_GUARD_FAILURE]
    assert led.record()["n_folds_fitted"] == 1
    monkeypatch.setattr(RC, "run_fold", real)

    # the directory holds ONE of the two records, so the unguarded reader would return a frame
    dd = RC.job_spec_of(m1, store.seed(i)).design_dir
    assert RC.read_seed_predictions(out_root, "M1", dd, i, code=code) is not None
    # ... and once the loop's own incompletions are named, the same read returns None and the claim carries no verdict
    marked = RC.mark_incomplete_record_sets(led, [m1])
    assert marked and marked[0]["arm"] == "M1" and marked[0]["design_dir"] == dd and marked[0]["seed_index"] == i
    assert marked[0]["transform"] == ""                     # the set is keyed by the transform too (C4's two legs)
    assert ("M1", dd, i, "") in RC.INCOMPLETE_RECORD_SETS
    assert RC.read_seed_predictions(out_root, "M1", dd, i, code=code) is None
    # a DIFFERENT seed index of the same arm and design is untouched: the unit is the record set, not the design
    assert ("M1", dd, i + 1, "") not in RC.INCOMPLETE_RECORD_SETS
    assert not any(str(s) in json.dumps(marked) for s in FAKE_SEEDS)
