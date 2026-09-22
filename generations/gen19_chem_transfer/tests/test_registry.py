"""The digest registry of POST-HOC addendum 2 item 5 (``gen19ct.evaluation.registry``) and the registry-first reading of
the post-discovery runners.  Every registry file lives under ``tmp_path``; the real ``manifests/digest_registry.json`` is
never created (its absence is what every fallback test relies on).  No discovery record of the running plan is read:
the record-field reader runs on synthetic JSON records written here.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gen19ct import paths
from gen19ct.evaluation import discovery as D
from gen19ct.evaluation import h3 as H3
from gen19ct.evaluation import registry as REG
from gen19ct.evaluation import report as R

REAL_REGISTRY = paths.G19_ROOT / "manifests" / "digest_registry.json"
A1 = D.REGISTERED_ADDENDA_SHA256
A2 = "a2" * 32
CODE = "c0" * 32


def _seal_module():
    name = "g19_seal_prereg"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, paths.G19_ROOT / "scripts" / "g19_seal_prereg.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_the_real_registry_is_absent_before_the_run_ends_and_holds_the_addendum_1_discovery_entry_after():
    """Addendum 2: the registry is created after the discovery run (never by a test or a runner).  Before that the file is
    absent and the constants govern; once created, its discovery entry holds the addendum-1 digest the records carry."""
    assert REG.registry_path() == REAL_REGISTRY and REG.REGISTRY_REL == "manifests/digest_registry.json"
    if not REAL_REGISTRY.exists():
        assert REG.read_registry() is None and REG.registered("discovery") is None
        assert REG.gate_expectations("discovery")["source"] == "constants"
        return
    disc = REG.registered("discovery")
    assert disc is not None and disc["below_footer_sha256"] == D.REGISTERED_ADDENDA_SHA256 and disc["addenda_count"] == 1
    assert disc.get("source") == "record fields"


def _real_text_matches(real: dict) -> None:
    """The real sealed text against the constants (before the registry) or the registry's discovery entry (after)."""
    disc = REG.registered("discovery")
    if disc is None:
        assert real["below_footer_sha256"] == D.REGISTERED_ADDENDA_SHA256 and real["n_addenda"] == D.N_ADDENDA_EXPECTED == 1
    else:
        assert disc["below_footer_sha256"] == D.REGISTERED_ADDENDA_SHA256 and disc["addenda_count"] == 1
        assert real["n_addenda"] >= disc["addenda_count"]


# --------------------------------------------------------------------------------------------- #
# register / registered / verify / log
# --------------------------------------------------------------------------------------------- #

def test_register_stage_writes_lf_sorted_json_and_is_immutable_per_stage(tmp_path):
    p = tmp_path / "manifests" / "digest_registry.json"
    e = REG.register_stage("discovery", below_footer_sha256=A1, code_digest=CODE, git_head="deadbeef", addenda_count=1,
                           note="addendum-1 digest and the discovery code digest, read from the records", path=p)
    assert e["stage"] == "discovery" and e["below_footer_sha256"] == A1 and e["code_digest"] == CODE and e["addenda_count"] == 1
    raw = p.read_bytes()
    assert b"\r\n" not in raw and raw.endswith(b"\n")
    body = json.loads(raw.decode("utf-8"))
    assert body["schema"] == REG.SCHEMA and list(body) == sorted(body) and list(body["stages"]["discovery"]) == sorted(body["stages"]["discovery"])
    assert REG.registered("discovery", p)["git_head"] == "deadbeef" and REG.registered("ladder", p) is None
    # identical re-registration: the digests and the registration time are untouched, but the DESCRIPTIVE fields may be
    # corrected (task X finding V-REG-08: a misleading 'source' note on an entry whose digests are right)
    same = REG.register_stage("discovery", below_footer_sha256=A1, code_digest=CODE, git_head="deadbeef",
                              addenda_count=1, note="again", path=p, extra={"code_digest_source": "the records"})
    assert same["registered_utc"] == e["registered_utc"] and same["git_head"] == "deadbeef"
    assert same["note"] == "again" and same["code_digest_source"] == "the records" and "amended_utc" in same
    assert same["below_footer_sha256"] == A1 and same["code_digest"] == CODE and same["addenda_count"] == 1
    noop = REG.register_stage("discovery", below_footer_sha256=A1, code_digest=CODE, git_head="deadbeef",
                              addenda_count=1, note="again", path=p, extra={"code_digest_source": "the records"})
    assert noop["amended_utc"] == same["amended_utc"]                      # nothing to correct: a no-op
    # a different digest for a registered stage is refused: the records were written once
    with pytest.raises(ValueError, match="already registered with other digests"):
        REG.register_stage("discovery", below_footer_sha256=A2, code_digest=CODE, git_head="x", addenda_count=2, note="n", path=p)
    assert REG.registered("discovery", p)["below_footer_sha256"] == A1
    # force keeps the superseded entry
    REG.register_stage("discovery", below_footer_sha256=A2, code_digest=CODE, git_head="x", addenda_count=2, note="forced",
                       path=p, force=True)
    body = json.loads(p.read_text(encoding="utf-8"))
    assert body["stages"]["discovery"]["below_footer_sha256"] == A2 and body["superseded"][0]["below_footer_sha256"] == A1
    assert body["superseded"][0]["superseded_by_note"] == "forced"
    # a second stage beside the first; unknown stages and malformed digests are refused
    REG.register_stage("ladder", below_footer_sha256=A2, code_digest=CODE, git_head=None, addenda_count=2, note="l", path=p)
    assert sorted(json.loads(p.read_text(encoding="utf-8"))["stages"]) == ["discovery", "ladder"]
    with pytest.raises(ValueError, match="unknown stage"):
        REG.register_stage("nope", below_footer_sha256=A2, code_digest=CODE, git_head=None, addenda_count=2, note="n", path=p)
    with pytest.raises(ValueError, match="not a SHA-256"):
        REG.register_stage("h3", below_footer_sha256="abc", code_digest=CODE, git_head=None, addenda_count=2, note="n", path=p)


def test_verify_record_compares_to_the_stage_entry_never_to_live_text_or_code(tmp_path, monkeypatch):
    p = tmp_path / "reg.json"
    REG.register_stage("discovery", below_footer_sha256=A1, code_digest=CODE, git_head=None, addenda_count=1, note="d", path=p)
    rec = {"prereg_addenda_sha256": A1, "code_digest": CODE, "prereg_n_addenda": 1, "digest": "x"}
    ok = REG.verify_record(rec, "discovery", p)
    assert ok["ok"] is True and ok["mismatches"] == {} and ok["registry_entry"]["code_digest"] == CODE
    # the live text and code are irrelevant: a changed constant or a changed live digest does not touch the verdict
    monkeypatch.setattr(D, "REGISTERED_ADDENDA_SHA256", "f" * 64)
    monkeypatch.setattr(REG, "live_discovery_code_digest", lambda: "e" * 64)
    assert REG.verify_record(rec, "discovery", p)["ok"] is True
    bad = REG.verify_record({**rec, "code_digest": "d" * 64}, "discovery", p)
    assert bad["ok"] is False and set(bad["mismatches"]) == {"code_digest"} and bad["mismatches"]["code_digest"]["registry"] == CODE
    bad2 = REG.verify_record({**rec, "prereg_addenda_sha256": A2, "prereg_n_addenda": 2}, "discovery", p)
    assert set(bad2["mismatches"]) == {"prereg_addenda_sha256", "prereg_n_addenda"}
    with pytest.raises(D.StaleRecordError, match="differs from its registry entry"):
        REG.verify_record({**rec, "code_digest": "d" * 64}, "discovery", p, raise_on_mismatch=True)
    none = REG.verify_record(rec, "ladder", p)
    assert none["ok"] is None and "no registry entry" in none["reason"]
    with pytest.raises(D.StaleRecordError, match="no registry entry"):
        REG.verify_record(rec, "ladder", p, raise_on_mismatch=True)
    # a record without the count field is compared on the two digests only
    assert REG.verify_record({"prereg_addenda_sha256": A1, "code_digest": CODE}, "discovery", p)["ok"] is True


def test_a_record_written_under_a_superseded_entry_of_its_stage_still_verifies(tmp_path):
    """Addendum 2 item 5: a later change "can neither validate nor invalidate a record written earlier".  Appending a
    POST-HOC addendum changes the below-footer digest of every stage that has not written its records yet, so those
    stages are re-registered (force) -- and any record already written under the earlier entry must keep verifying
    against THAT entry, which ``force`` kept under ``superseded``."""
    p = tmp_path / "reg.json"
    A3 = "a3" * 32
    REG.register_stage("scorer", below_footer_sha256=A2, code_digest=CODE, git_head=None, addenda_count=2,
                       note="two addenda", path=p)
    old_rec = {"prereg_addenda_sha256": A2, "code_digest": CODE, "prereg_n_addenda": 2, "registry_stage": "scorer"}
    assert REG.verify_record(old_rec, None, p)["matched_entry"] == "current"
    # a third addendum is appended and the stage is re-registered under the new below-footer digest
    REG.register_stage("scorer", below_footer_sha256=A3, code_digest=CODE, git_head=None, addenda_count=3,
                       note="three addenda (addendum 3)", path=p, force=True)
    assert REG.registered("scorer", p)["addenda_count"] == 3
    new_rec = {**old_rec, "prereg_addenda_sha256": A3, "prereg_n_addenda": 3}
    assert REG.verify_record(new_rec, None, p)["matched_entry"] == "current"
    # the earlier record is NOT invalidated: it verifies against the superseded entry, which is named
    got = REG.verify_record(old_rec, None, p)
    assert got["ok"] is True and got["matched_entry"] == "superseded"
    assert got["superseded_match"]["below_footer_sha256"] == A2 and got["superseded_match"]["addenda_count"] == 2
    assert set(got["mismatches_against_current"]) == {"prereg_addenda_sha256", "prereg_n_addenda"}
    assert "invalidates no record written earlier" in got["reason"]
    assert REG.verify_record(old_rec, "scorer", p, raise_on_mismatch=True)["ok"] is True
    assert [e["below_footer_sha256"] for e in REG.superseded_entries("scorer", p)] == [A2]
    assert REG.superseded_entries("h3", p) == []
    # a record matching NO entry of its stage, current or superseded, still does not verify
    foreign = {**old_rec, "code_digest": "f" * 64}
    assert REG.verify_record(foreign, None, p)["ok"] is False
    with pytest.raises(D.StaleRecordError, match="superseded"):
        REG.verify_record(foreign, "scorer", p, raise_on_mismatch=True)
    # the real registry: the scorer's S1(c) yardstick records were written under addendum 2 and must still verify
    if REAL_REGISTRY.exists():
        yard = paths.G19_ROOT / "evaluation" / "discovery" / "_s1c_yardsticks"
        for js in sorted(yard.rglob("yardsticks.json")):
            rec = json.loads(js.read_text(encoding="utf-8"))
            out = REG.verify_record(rec, None)
            assert out["ok"] is True, f"{js}: {out}"


def test_log_code_change_appends_and_touches_no_stage_entry(tmp_path):
    p = tmp_path / "reg.json"
    REG.register_stage("discovery", below_footer_sha256=A1, code_digest=CODE, git_head=None, addenda_count=1, note="d", path=p)
    f = tmp_path / "mod.py"
    f.write_bytes(b"x = 1\r\ny = 2\r\n")
    ch = REG.log_code_change([f, tmp_path / "absent.py"], "abc123", "addendum 2 item 4: shared-only embedding", path=p,
                             root=tmp_path)
    assert ch["commit"] == "abc123" and ch["files"][0]["path"] == "mod.py"
    assert ch["files"][0]["sha256_lf"] == hashlib.sha256(b"x = 1\ny = 2\n").hexdigest() and ch["files"][0]["exists"]
    assert ch["files"][1]["sha256_lf"] is None and not ch["files"][1]["exists"]
    assert "validates or invalidates no record" in ch["effect_on_records"]
    body = json.loads(p.read_text(encoding="utf-8"))
    assert len(body["code_changes"]) == 1 and body["stages"]["discovery"]["code_digest"] == CODE
    REG.log_code_change([f], None, "second", path=p, root=tmp_path)
    assert len(json.loads(p.read_text(encoding="utf-8"))["code_changes"]) == 2
    # logging into a not-yet-existing registry creates it with no stage
    q = tmp_path / "new.json"
    REG.log_code_change([f], "c", "r", path=q, root=tmp_path)
    assert json.loads(q.read_text(encoding="utf-8"))["stages"] == {}


# --------------------------------------------------------------------------------------------- #
# the digests: below-footer rule and record fields
# --------------------------------------------------------------------------------------------- #

def test_below_footer_digest_uses_the_seal_scripts_rule_and_matches_the_runner(tmp_path):
    sp = _seal_module()
    body = "# draft\n\n## 15. Seeds\ncommitment: abc\n"
    sealed, digest = sp.build_sealed_text(body)
    addendum = "\n## POST-HOC addendum 1 (2026-09-15, orchestrator; results seen: no)\n\nText of the addendum.\n"
    (tmp_path / "manifests").mkdir()
    sp.write_lf(tmp_path / "preregistration.md", sealed + addendum)
    sp.write_lf(tmp_path / "manifests" / "prereg_sha256.txt", digest + "\n")
    dg = REG.below_footer_digest(tmp_path / "preregistration.md", tmp_path / "manifests" / "prereg_sha256.txt")
    assert dg["footer"] == dg["recomputed"] == dg["digest_file"] == digest and dg["n_addenda"] == 1
    _, _, below = sp.split_footer(sp.read_normalised(tmp_path / "preregistration.md"))
    assert dg["below_footer_sha256"] == D.sha256_text(below) == hashlib.sha256(addendum.encode("utf-8")).hexdigest()
    # CRLF on disk gives the same digests (LF normalisation)
    (tmp_path / "crlf.md").write_bytes((sealed + addendum).replace("\n", "\r\n").encode("utf-8"))
    dg2 = REG.below_footer_digest(tmp_path / "crlf.md", tmp_path / "manifests" / "prereg_sha256.txt")
    assert dg2["below_footer_sha256"] == dg["below_footer_sha256"] and dg2["recomputed"] == digest
    # appending addendum 2 changes the below-footer digest and the count, never the footer digest
    sp.write_lf(tmp_path / "preregistration.md", sealed + addendum +
                "\n## POST-HOC addendum 2 (2026-09-19, orchestrator; results seen: yes)\n\nMore.\n")
    dg3 = REG.below_footer_digest(tmp_path / "preregistration.md", tmp_path / "manifests" / "prereg_sha256.txt")
    assert dg3["n_addenda"] == 2 and dg3["below_footer_sha256"] != dg["below_footer_sha256"] and dg3["footer"] == digest
    # the real sealed text: the same function reproduces the constants the discovery run was gated on
    real = REG.below_footer_digest()
    assert real["footer"] == real["recomputed"] == real["digest_file"] == D.REGISTERED_PREREG_SHA256
    _real_text_matches(real)


def _record(out: Path, arm: str, name: str, **fields) -> Path:
    d = D.discovery_root(out) / arm / "V5__primary_exact" / "s104729"
    d.mkdir(parents=True, exist_ok=True)
    js = d / f"{name}.json"
    js.write_text(json.dumps({"digest": "d", "fold_id": name, "code_digest": CODE, "prereg_addenda_sha256": A1,
                              "prereg_n_addenda": 1, "mean_logD_must_not_be_read": 1.0, **fields}), encoding="utf-8")
    return js


def test_discovery_stage_is_registered_from_the_record_fields_not_recomputed(tmp_path, monkeypatch):
    out = tmp_path / "out"
    with pytest.raises(FileNotFoundError):
        REG.digests_from_discovery_records(out)
    _record(out, "B5", "f0")
    _record(out, "M2", "f1")
    (D.discovery_root(out) / "B6" / "V5__primary_exact" / "s104729").mkdir(parents=True)
    (D.discovery_root(out) / "B6" / "V5__primary_exact" / "s104729" / "broken.json").write_text("{not json", encoding="utf-8")
    dg = REG.digests_from_discovery_records(out)
    assert {k: dg[k] for k in ("below_footer_sha256", "code_digest", "addenda_count", "n_records", "source")} == {
        "below_footer_sha256": A1, "code_digest": CODE, "addenda_count": 1, "n_records": 2, "source": "record fields"}
    assert dg["stage"] == "discovery" and dg["restricted_to_plan"] is False and dg["n_ignored"] == 0   # no plan state on disk
    # recomputation is never involved: a changed live code digest or constant does not change what is registered
    monkeypatch.setattr(REG, "live_discovery_code_digest", lambda: (_ for _ in ()).throw(AssertionError("must not recompute")))
    monkeypatch.setattr(D, "REGISTERED_ADDENDA_SHA256", "f" * 64)
    p = tmp_path / "reg.json"
    e = REG.register_discovery_from_records(out, note="from records", path=p, git_head="h")
    assert e["below_footer_sha256"] == A1 and e["code_digest"] == CODE and e["addenda_count"] == 1
    assert e["source"] == "record fields" and e["n_records"] == 2 and e["git_head"] == "h"
    # records that disagree are refused (never averaged, never the newest), and the offending paths are named per value
    _record(out, "M1", "f2", code_digest="e" * 64)
    with pytest.raises(ValueError, match="do not agree") as ei:
        REG.digests_from_discovery_records(out)
    assert "M1/V5__primary_exact/s104729/f2.json" in str(ei.value) and "B5/V5__primary_exact/s104729/f0.json" in str(ei.value)


# --------------------------------------------------------------------------------------------- #
# registry first, constants otherwise
# --------------------------------------------------------------------------------------------- #

def test_gate_expectations_fall_back_to_the_constants_until_the_registry_exists(tmp_path):
    p = tmp_path / "reg.json"
    for stage in ("discovery", "ladder", "h3", "power", "process", "figures", "report"):
        exp = REG.gate_expectations(stage, p)
        assert exp == {"stage": stage, "n_addenda": D.N_ADDENDA_EXPECTED, "below_footer_sha256": D.REGISTERED_ADDENDA_SHA256,
                       "source": "constants", "registry_present": False}
        assert REG.below_footer_sha256(stage, p) == D.REGISTERED_ADDENDA_SHA256 and REG.addenda_count(stage, p) == 1
    REG.register_stage("discovery", below_footer_sha256=A1, code_digest=CODE, git_head=None, addenda_count=1, note="d", path=p)
    REG.register_stage("ladder", below_footer_sha256=A2, code_digest="c1" * 32, git_head=None, addenda_count=2, note="l", path=p)
    lad = REG.gate_expectations("ladder", p)
    assert lad["source"] == "registry" and lad["n_addenda"] == 2 and lad["below_footer_sha256"] == A2 and lad["code_digest"] == "c1" * 32
    assert REG.below_footer_sha256("ladder", p) == A2 and REG.addenda_count("ladder", p) == 2
    # a registry without the stage: the constants (the runner then refuses on a 2-addenda text, as it must)
    h3 = REG.gate_expectations("h3", p)
    assert h3["source"] == "constants" and h3["registry_present"] is True and h3["below_footer_sha256"] == A1
    assert REG.discovery_code_digest(p) == CODE


def test_discovery_code_digest_prefers_the_registry_and_falls_back_to_the_live_digest(tmp_path, monkeypatch):
    monkeypatch.setattr(REG, "live_discovery_code_digest", lambda: "live" + "0" * 60)
    assert REG.discovery_code_digest(tmp_path / "none.json") == "live" + "0" * 60
    p = tmp_path / "reg.json"
    REG.register_stage("ladder", below_footer_sha256=A2, code_digest=CODE, git_head=None, addenda_count=2, note="l", path=p)
    assert REG.discovery_code_digest(p) == "live" + "0" * 60                # no discovery entry yet
    REG.register_stage("discovery", below_footer_sha256=A1, code_digest="ab" * 32, git_head=None, addenda_count=1, note="d", path=p)
    assert REG.discovery_code_digest(p) == "ab" * 32


def _digests(n: int, sha: str):
    return lambda: {"footer": D.REGISTERED_PREREG_SHA256, "recomputed": D.REGISTERED_PREREG_SHA256,
                    "digest_file": D.REGISTERED_PREREG_SHA256, "addenda_sha256": sha, "n_addenda": n}


def test_refuse_unless_sealed_delegates_to_the_constants_without_a_registry_and_reads_the_stage_entry_with_one(tmp_path, monkeypatch):
    calls = []

    def stub_gate(check, digests, expect_addenda=None):
        calls.append((check, digests, expect_addenda))
        return {"prereg_sha256": D.REGISTERED_PREREG_SHA256, "n_addenda": 1, "addenda_sha256": A1}

    monkeypatch.setattr(REG, "_runner_module", lambda: SimpleNamespace(refuse_unless_sealed=stub_gate))
    p = tmp_path / "reg.json"
    rec = REG.refuse_unless_sealed("ladder", lambda: 0, _digests(1, A1), expect_addenda=None, path=p)
    assert rec["gate_source"] == "constants" and rec["stage"] == "ladder" and len(calls) == 1      # unchanged behaviour
    # with the stage registered under addendum 2's text, the registry decides -- the runner's function is not called
    REG.register_stage("ladder", below_footer_sha256=A2, code_digest=CODE, git_head=None, addenda_count=2, note="l", path=p)
    ok = REG.refuse_unless_sealed("ladder", lambda: 0, _digests(2, A2), path=p)
    assert ok["gate_source"] == "registry" and ok["n_addenda"] == ok["n_addenda_expected"] == 2 and ok["addenda_sha256"] == A2
    assert ok["addendum_implemented"] == 2 and ok["prereg_sha256"] == D.REGISTERED_PREREG_SHA256 and len(calls) == 1
    with pytest.raises(SystemExit, match="1 POST-HOC addendum/addenda below the footer, 2 expected"):
        REG.refuse_unless_sealed("ladder", lambda: 0, _digests(1, A1), path=p)
    with pytest.raises(SystemExit, match="not the registry entry of stage 'ladder'"):
        REG.refuse_unless_sealed("ladder", lambda: 0, _digests(2, "b" * 64), path=p)
    with pytest.raises(SystemExit, match="never passes"):                # --expect-addenda overrides the count, never the digest
        REG.refuse_unless_sealed("ladder", lambda: 0, _digests(3, "b" * 64), expect_addenda=3, path=p)
    with pytest.raises(SystemExit, match="exited 1"):
        REG.refuse_unless_sealed("ladder", lambda: 1, _digests(2, A2), path=p)
    bad_seal = _digests(2, A2)()
    bad_seal["footer"] = "0" * 64
    with pytest.raises(SystemExit, match="not the registered text"):
        REG.refuse_unless_sealed("ladder", lambda: 0, lambda: bad_seal, path=p)
    # an unregistered stage of an EXISTING registry: the constants' expectations, checked here (not delegated to the
    # runner, whose default stage would name 'discovery' in the refusal) and the refusal names THAT stage
    rec2 = REG.refuse_unless_sealed("h3", lambda: 0, _digests(1, A1), path=p)
    assert rec2["gate_source"] == "constants" and rec2["stage"] == "h3" and len(calls) == 1
    with pytest.raises(SystemExit, match="stage 'h3' is not registered") as ei:
        REG.refuse_unless_sealed("h3", lambda: 0, _digests(2, A2), path=p)
    assert "discovery" not in str(ei.value)


def test_h3_fold_digest_and_records_read_the_registry_entry_of_stage_h3(tmp_path, monkeypatch):
    fold = SimpleNamespace(fold_id="f", fold_hash="h")
    job = D.JobSpec(kind="fit", arm="M2:WITHOUT", design="V5", variant="primary", scheme="exact", seed=D.PRIMARY_SEED,
                    writes=("M2:WITHOUT",), stage=H3.STAGE)
    kw = dict(transform="WITHOUT", with_digest="w", guard_mode="every_split", design_hash="d", ordinal=0, model_seed=1)
    before = H3.fold_digest(job, fold, "code", **kw)
    # the below-footer digest in the digest is the one stage 'h3' is registered under -- its entry when the registry
    # exists, the addendum-1 constant while it does not
    assert REG.below_footer_sha256("h3") == (REG.registered("h3") or {}).get("below_footer_sha256", A1)
    assert before == D.fold_digest(job, fold, "code", {"schema": H3.SCHEMA, "transform": "WITHOUT", "with_record_digest": "w",
                                                       "guard_mode": "every_split", "design_hash": "d", "model_fold_number": 0,
                                                       "model_seed": 1, "prereg_sha256": D.REGISTERED_PREREG_SHA256,
                                                       "prereg_addenda_sha256": REG.below_footer_sha256("h3")})
    p = tmp_path / "reg.json"
    REG.register_stage("h3", below_footer_sha256=A2, code_digest=CODE, git_head=None, addenda_count=2, note="h3", path=p)
    monkeypatch.setattr(REG, "registry_path", lambda root=None: p)
    assert REG.below_footer_sha256("h3") == A2
    after = H3.fold_digest(job, fold, "code", **kw)
    assert after != before                                                 # the registry entry now enters the digest
    assert "addendum 2 item 5" in H3.fold_digest.__doc__


def test_stage_code_digest_rules_and_register_from_tree(tmp_path, monkeypatch):
    fake = {"g19_run_ladder": SimpleNamespace(ladder_code_digest=lambda: {"own": "o", "discovery": "d" * 64, "combined": "c"}),
            "g19_run_h3": SimpleNamespace(code_digest=lambda: {"combined": "h" * 64, "parts": {}}),
            "g19_run_process": SimpleNamespace(code_digests=lambda: {"own": "p" * 64})}
    monkeypatch.setattr(REG, "_script", lambda name: fake[name])
    monkeypatch.setattr(REG, "live_discovery_code_digest", lambda: "l" * 64)
    assert REG.stage_code_digest("ladder") == "d" * 64 and REG.stage_code_digest("h3") == "h" * 64
    assert REG.stage_code_digest("process") == "p" * 64 and REG.stage_code_digest("discovery") == "l" * 64
    with pytest.raises(ValueError):
        REG.stage_code_digest("confirmation")
    sp = _seal_module()
    sealed, digest = sp.build_sealed_text("# t\n## 15. Seeds\nc\n")
    add = "\n## POST-HOC addendum 1 (2026-09-15, o; results seen: no)\n\nA.\n\n## POST-HOC addendum 2 (2026-09-19, o; results seen: yes)\n\nB.\n"
    (tmp_path / "m").mkdir()
    sp.write_lf(tmp_path / "p.md", sealed + add)
    sp.write_lf(tmp_path / "m" / "sha.txt", digest + "\n")
    p = tmp_path / "reg.json"
    e = REG.register_stage_from_tree("ladder", note="under addendum 2", path=p, sealed=tmp_path / "p.md", sha_file=tmp_path / "m" / "sha.txt")
    assert e["addenda_count"] == 2 and e["code_digest"] == "d" * 64 and e["source"].startswith("current tree")
    # task X finding V-REG-08: 'source' describes the SEALED TEXT; where the code digest came from is its own field,
    # because the scorer's and the ladder's digest is the registry's discovery entry, not a digest of their own code
    assert e["code_digest_source"] == REG.CODE_DIGEST_SOURCE["ladder"] and "discovery" in e["code_digest_source"]
    assert "records" in REG.CODE_DIGEST_SOURCE["scorer"] and "tree" in REG.CODE_DIGEST_SOURCE["h3"]
    assert set(REG.CODE_DIGEST_SOURCE) == set(REG.STAGES)
    given = REG.register_stage_from_tree("h3", note="n", path=p, code_digest="a" * 64, sealed=tmp_path / "p.md",
                                         sha_file=tmp_path / "m" / "sha.txt")
    assert given["code_digest_source"].startswith("passed explicitly")
    assert e["below_footer_sha256"] == hashlib.sha256(add.encode("utf-8")).hexdigest()
    with pytest.raises(ValueError, match="registered from its records"):
        REG.register_stage_from_tree("discovery", note="x", path=p)


def test_cli_show_verify_and_log_change(tmp_path, capsys):
    p = tmp_path / "reg.json"
    REG.register_stage("discovery", below_footer_sha256=A1, code_digest=CODE, git_head=None, addenda_count=1, note="d", path=p)
    rec = tmp_path / "rec.json"
    rec.write_text(json.dumps({"prereg_addenda_sha256": A1, "code_digest": CODE, "prereg_n_addenda": 1}), encoding="utf-8")
    assert REG.main(["verify", "--stage", "discovery", "--record", str(rec), "--registry", str(p)]) == 0
    rec.write_text(json.dumps({"prereg_addenda_sha256": A2, "code_digest": CODE}), encoding="utf-8")
    assert REG.main(["verify", "--stage", "discovery", "--record", str(rec), "--registry", str(p)]) == 1
    f = tmp_path / "x.py"
    f.write_text("a\n", encoding="utf-8")
    assert REG.main(["log-change", "--files", str(f), "--commit", "abc", "--reason", "why", "--registry", str(p)]) == 0
    assert REG.main(["show", "--registry", str(p)]) == 0
    out = capsys.readouterr().out
    assert '"code_changes"' in out and "abc" in out
    assert REG.main(["current"]) == 0
    cur = json.loads(capsys.readouterr().out)
    assert cur["below_footer_sha256"] == REG.below_footer_digest()["below_footer_sha256"] and "discovery_code_digest" not in cur
    _real_text_matches(cur)


def test_report_prints_every_registry_entry_and_the_numbers_re_resolve(tmp_path):
    rel = "manifests/digest_registry.json"
    p = tmp_path / rel
    REG.register_stage("discovery", below_footer_sha256=A1, code_digest=CODE, git_head="g1", addenda_count=1, note="records", path=p)
    REG.register_stage("ladder", below_footer_sha256=A2, code_digest="c1" * 32, git_head="g2", addenda_count=2, note="tree", path=p)
    REG.log_code_change([tmp_path / "gen19ct" / "x.py"], "abc", "addendum 2 item 4", path=p, root=tmp_path)
    res = R.build_report(tmp_path, extra={"git_head": "x", "addenda_sha256": A2, "n_addenda": 2, "recomputed": "r" * 64})
    rep = res["report"]
    assert f"digest registry (`{rel}`, addendum 2 item 5): 2 stage entries, 1 logged code change(s)" in rep
    assert f"stage `discovery`: below-footer SHA-256 `{A1}`, code digest `{CODE}`, 1 addenda, git head `g1`" in rep
    assert f"stage `ladder`: below-footer SHA-256 `{A2}`" in rep and "code change 1 (commit `abc`" in rep
    # the live addenda digest is compared with the registry, not with the code constant
    assert f"`{A2}` (= the below-footer digest of registry stage(s) `ladder` in `{rel}`)" in rep
    assert "REGISTERED_ADDENDA_SHA256" not in rep.split("SHA-256 of the POST-HOC addenda text")[1].split("\n")[0]
    nums = res["numbers"]
    reg_rows = nums[nums["source_path"] == rel]
    assert len(reg_rows) >= 6 and set(reg_rows["kind"]) == {"count", "text", "value"}
    ver = R.verify_numbers(tmp_path, reg_rows)                  # every registry token re-resolves from the file
    assert ver["ok"].all(), ver[~ver["ok"]]
    # a live text matching no stage is said so, without printing the token
    res2 = R.build_report(tmp_path, extra={"git_head": "x", "addenda_sha256": "b" * 64, "n_addenda": 3, "recomputed": "r" * 64})
    assert "matches NO stage entry" in res2["report"] and "b" * 64 not in res2["report"]
    # without a registry the constants path of before is printed (the report cites the literal in discovery.py)
    (tmp_path / "empty" / "gen19ct" / "evaluation").mkdir(parents=True)
    (tmp_path / "empty" / "gen19ct" / "evaluation" / "discovery.py").write_text(
        f'REGISTERED_ADDENDA_SHA256 = "{A1}"\n', encoding="utf-8")
    res3 = R.build_report(tmp_path / "empty", extra={"git_head": "x", "addenda_sha256": A1, "n_addenda": 1, "recomputed": "r" * 64})
    assert "digest registry (`manifests/digest_registry.json`): absent -- the gate constants" in res3["report"]
    assert "REGISTERED_ADDENDA_SHA256" in res3["report"]


# --------------------------------------------------------------------------------------------- #
# the two stages of the discovery runner (task X verifier finding 2: the freezing-candidate pass runs after the scorer)
# --------------------------------------------------------------------------------------------- #

def _test_module(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_job_stage_code_digest_for_record_dir_code_and_refuse_unless_writable(tmp_path, monkeypatch):
    assert REG.CANDIDATES in REG.STAGES and REG.RUNNER_STAGES == ("discovery", "discovery_candidates")
    disc = D.JobSpec(kind="fit", arm="M2", design="V5", variant="primary", scheme="batched", seed=D.PRIMARY_SEED,
                     fold_seed=D.PRIMARY_SEED, stage=D.STAGES["p_m2"])
    cand = D.JobSpec(kind="fit", arm="B8", design="V5PAIR", variant="primary", scheme="batched", seed=D.PRIMARY_SEED,
                     fold_seed=D.PRIMARY_SEED, stage=D.STAGES["candidates"])
    assert REG.job_stage(disc) == "discovery" and REG.job_stage(cand) == "discovery_candidates"
    assert REG.job_stage({"job": disc.record()}) == "discovery" and REG.job_stage({"job": cand.record()}) == "discovery_candidates"
    assert REG.job_stage(cand.record()) == "discovery_candidates" and REG.job_stage({}) == "discovery"
    assert REG.job_stage({"registry_stage": "discovery_candidates", "job": disc.record()}) == "discovery_candidates"
    # the stage is a label outside the resume digest, so it never changes what a job predicts or how it resumes
    assert "stage" in D.LABEL_FIELDS
    monkeypatch.setattr(REG, "live_discovery_code_digest", lambda: "l" * 64)
    none = tmp_path / "none.json"
    assert REG.code_digest_for("discovery", path=none) == "l" * 64
    assert REG.code_digest_for(REG.CANDIDATES, live="v" * 64, path=none) == "v" * 64
    p = tmp_path / "reg.json"
    REG.register_stage("discovery", below_footer_sha256=A1, code_digest=CODE, git_head=None, addenda_count=1, note="d", path=p)
    REG.register_stage(REG.CANDIDATES, below_footer_sha256=A2, code_digest="c2" * 32, git_head=None, addenda_count=2, note="c", path=p)
    assert REG.code_digest_for("discovery", live="v" * 64, path=p) == CODE and REG.code_digest_for(REG.CANDIDATES, path=p) == "c2" * 32
    assert REG.code_digest_for("scorer", live="v" * 64, path=p) == "v" * 64          # unregistered: the live fallback
    out = tmp_path / "out"
    root = D.discovery_root(out)
    for job, arm in ((disc, "M2"), (cand, "B8")):
        d = root / arm / job.design_dir / f"s{job.seed}"
        d.mkdir(parents=True)
        (d / "f.json").write_text(json.dumps({"job": job.record(), "digest": "x"}), encoding="utf-8")
    assert REG.record_dir_stage(root / "M2" / disc.design_dir / f"s{disc.seed}") == "discovery"
    assert REG.record_dir_stage(root / "B8" / cand.design_dir / f"s{cand.seed}") == "discovery_candidates"
    assert REG.record_dir_stage(root / "nowhere") is None
    assert REG.record_dir_code(out, "M2", disc.design_dir, disc.seed, default="v" * 64, path=p) == CODE
    assert REG.record_dir_code(out, "B8", cand.design_dir, cand.seed, default="v" * 64, path=p) == "c2" * 32
    assert REG.record_dir_code(out, "B8", cand.design_dir, cand.seed, default="v" * 64, path=none) == "v" * 64
    assert REG.record_dir_code(out, "B5", "V5__primary_batched", 104729, default="v" * 64, path=p) == "v" * 64   # no record
    # the runner's rule before it FITS: without a registry nothing changes; with one, only the registered code writes
    assert REG.refuse_unless_writable("discovery", "l" * 64, path=none)["source"] == "constants"
    assert REG.refuse_unless_writable("discovery", CODE, path=p)["source"] == "registry"
    with pytest.raises(SystemExit, match="written only under the code the stage was registered with"):
        REG.refuse_unless_writable("discovery", "l" * 64, path=p)                # the closure was patched: no refit, ever
    assert REG.refuse_unless_writable(REG.CANDIDATES, "c2" * 32, path=p)["ok"]
    with pytest.raises(SystemExit, match="stage 'scorer' is not registered"):
        REG.refuse_unless_writable("scorer", "l" * 64, path=p)
    # verify_record takes the stage the record carries when none is named
    rec = {"prereg_addenda_sha256": A2, "code_digest": "c2" * 32, "prereg_n_addenda": 2, "job": cand.record()}
    assert REG.verify_record(rec, None, path=p)["ok"] and REG.verify_record(rec, None, path=p)["stage"] == REG.CANDIDATES
    assert not REG.verify_record(rec, "discovery", path=p)["ok"]
    # the candidates stage is registered from the tree (its records do not exist before the pass), under the live code
    assert REG.stage_code_digest(REG.CANDIDATES) == "l" * 64
    reading = REG.READINGS["candidates"].lower()
    assert "'discovery_candidates'" in reading and "after the scorer" in reading and "never a refit" in reading


def test_digests_from_discovery_records_are_stage_aware_restricted_to_the_plan_and_name_paths(tmp_path):
    out = tmp_path / "out"
    state = D.PlanState(v5_batched_check="passed", v1_tenfold_check="passed")
    dec = D.discovery_root(out) / "decisions"
    dec.mkdir(parents=True)
    (dec / "plan_state.json").write_text(json.dumps(state.record()), encoding="utf-8")
    plan = [j for j in D.enumerate_plan(state) if j.kind == "fit"]
    assert all(REG.job_stage(j) == "discovery" for j in plan)             # no freezing candidate yet: one runner stage
    root = D.discovery_root(out)
    dirs = REG.plan_record_dirs(out)
    b6 = next(j for j in plan if j.arm == "B6" and j.design == "V5" and j.variant == "primary" and j.scheme == "exact")
    b5 = next(j for j in plan if j.arm == "B5" and j.design == "V5" and j.variant == "primary")
    assert root / "B6" / b6.design_dir / f"s{b6.seed}" in dirs and root / "B6r0" / b6.design_dir / f"s{b6.seed}" in dirs
    assert root / "B5" / b5.design_dir / f"s{b5.seed}" in dirs and b5.design_dir == "V5__primary_batched"
    assert REG.plan_record_dirs(out, stage=REG.CANDIDATES) == [] and REG.plan_record_dirs(tmp_path / "empty") is None

    def rec(job, arm, name, **fields):
        d = root / arm / job.design_dir / f"s{job.seed}"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.json").write_text(json.dumps({"digest": "d", "fold_id": name, "code_digest": CODE,
                                                    "prereg_addenda_sha256": A1, "prereg_n_addenda": 1,
                                                    "job": job.record(), **fields}), encoding="utf-8")

    rec(b6, "B6", "f0")
    rec(b6, "B6r0", "f0")
    rec(b5, "B5", "f1")
    # a record of an EARLIER plan (a directory no current job writes; another code digest) is ignored, never a blocker
    old = D.JobSpec(kind="fit", arm="B5", design="V5", variant="primary", scheme="exact", seed=D.PRIMARY_SEED,
                    stage=D.STAGES["p_b5"], writes=("B5",))
    rec(old, "B5", "f9", code_digest="e" * 64)
    # a freezing-candidate record is another registry stage, whatever its digests
    cand = D.JobSpec(kind="fit", arm="B8", design="V5", variant="strict", scheme="batched", seed=D.PRIMARY_SEED,
                     fold_seed=D.PRIMARY_SEED, stage=D.STAGES["candidates"], writes=("B8",))
    rec(cand, "B8", "f2", code_digest="a" * 64, prereg_addenda_sha256=A2, prereg_n_addenda=2)
    dg = REG.digests_from_discovery_records(out)
    assert dg["n_records"] == 3 and dg["code_digest"] == CODE and dg["below_footer_sha256"] == A1 and dg["addenda_count"] == 1
    assert dg["restricted_to_plan"] and dg["n_plan_dirs"] == len(dirs) and dg["n_ignored"] == 1 and dg["n_other_stage"] == 1
    assert dg["ignored_paths"] == ["B5/V5__primary_exact/s104729/f9.json"]
    e = REG.register_discovery_from_records(out, note="n", path=tmp_path / "reg.json", git_head="h")
    assert e["code_digest"] == CODE and e["n_records"] == 3 and e["restricted_to_plan"] is True and e["n_ignored_records"] == 1
    cd = REG.digests_from_discovery_records(out, stage=REG.CANDIDATES, restrict_to_plan=False)
    assert cd["n_records"] == 1 and cd["code_digest"] == "a" * 64 and cd["addenda_count"] == 2 and cd["below_footer_sha256"] == A2
    with pytest.raises(FileNotFoundError, match="of stage 'discovery_candidates'"):
        REG.digests_from_discovery_records(out, stage=REG.CANDIDATES)       # the plan holds no candidate job: no directory
    # unrestricted, the old record blocks -- and is named, with the plan record it disagrees with
    with pytest.raises(ValueError, match="do not agree") as ei:
        REG.digests_from_discovery_records(out, restrict_to_plan=False)
    assert "B5/V5__primary_exact/s104729/f9.json" in str(ei.value) and "B5/V5__primary_batched/s104729/f1.json" in str(ei.value)


def test_discovery_complete_verifies_each_record_set_against_its_own_stage(tmp_path, monkeypatch):
    """H3's completeness check (and every reader) resolves the code digest per record set from the stage its records
    carry: a freezing-candidate set written under the post-patch code verifies against the discovery_candidates entry,
    the run's own sets against the discovery entry; without a registry one code covers everything, as before."""
    TH = _test_module("test_h3")
    monkeypatch.setattr(REG, "registry_path", lambda root=None: tmp_path / "absent.json")
    corpus = TH._corpus(tmp_path)
    TH._write([TH._cell_fold(corpus.frame, "Nd(III)", "S1")], corpus.folds_dir)
    state = D.PlanState()
    job = D.JobSpec(kind="fit", arm="B5", design="V5", variant="primary", scheme="exact", seed=D.PRIMARY_SEED,
                    stage=D.STAGES["p_b5"], writes=("B5",))
    cand = D.JobSpec(kind="fit", arm="FLAT_CAT", design="V5", variant="primary", scheme="exact", seed=D.PRIMARY_SEED,
                     stage=D.STAGES["candidates"], writes=("FLAT_CAT",))
    out = tmp_path / "out"
    TH._complete_discovery(out, corpus, job, code="c1" * 32, state=state)        # written before the registry exists
    kw = dict(state=state, excluded_ids=[], folds_dir=corpus.folds_dir, jobs=[job, cand])
    # the production order (registry.READINGS['candidates']): the discovery entry is registered from the completed run's
    # records, the candidates stage from the tree, and only THEN is a freezing-candidate job fitted -- so its record
    # carries the candidates entry's below-footer digest and code digest
    p = tmp_path / "reg.json"
    monkeypatch.setattr(REG, "registry_path", lambda root=None: p)
    REG.register_stage("discovery", below_footer_sha256=A1, code_digest="c1" * 32, git_head=None, addenda_count=1, note="d")
    REG.register_stage(REG.CANDIDATES, below_footer_sha256=A2, code_digest="c2" * 32, git_head=None, addenda_count=2, note="c")
    TH._complete_discovery(out, corpus, cand, code="c2" * 32, state=state)
    TH.RD.record_wall_clock(out, "2026-09-18T00:00:00+00:00", 10.0, [job.stage, cand.stage, D.STAGES["not_implemented"]])
    done = H3.discovery_complete(out, code="z" * 64, **kw)          # the live code is irrelevant once both are registered
    assert done["complete"] and done["n_complete"] == 2 and done["n_incomplete"] == 0
    # without a registry one code (and one addendum text) covers everything, so the candidates set is stale -- as before
    monkeypatch.setattr(REG, "registry_path", lambda root=None: tmp_path / "absent.json")
    with pytest.raises(D.StaleRecordError):
        H3.discovery_complete(out, code="c1" * 32, **kw)
    # with the discovery entry alone the candidates set is verified with the live code and refused
    only_disc = tmp_path / "reg_discovery_only.json"
    REG.register_stage("discovery", below_footer_sha256=A1, code_digest="c1" * 32, git_head=None, addenda_count=1,
                       note="d", path=only_disc)
    monkeypatch.setattr(REG, "registry_path", lambda root=None: only_disc)
    with pytest.raises(D.StaleRecordError):
        H3.discovery_complete(out, code="z" * 64, **kw)
    assert "record_dir_code" in H3.discovery_complete.__doc__


def test_the_superseded_fallback_is_ordered_by_the_records_own_timestamp(tmp_path):
    """Task X finding V-P06: the fallback of the test above protects a record written EARLIER.  Without an ordering
    check a record written AFTER a re-registration, by a runner still holding the stale code, verifies exactly like one
    written before it -- so the check is made explicit, on the record's own timestamp."""
    p = tmp_path / "reg.json"
    A3, OLD, NEW = "a3" * 32, "0a" * 32, "0b" * 32
    REG.register_stage("h3", below_footer_sha256=A2, code_digest=OLD, git_head=None, addenda_count=2,
                       note="before the edit", path=p)
    base = {"prereg_addenda_sha256": A2, "code_digest": OLD, "prereg_n_addenda": 2, "registry_stage": "h3"}
    REG.register_stage("h3", below_footer_sha256=A3, code_digest=NEW, git_head=None, addenda_count=3,
                       note="after the edit (addendum 3 + a code change)", path=p, force=True)
    sup = REG.superseded_entries("h3", p)[0]["superseded_utc"]
    assert sup

    def shift(stamp: str, minutes: int) -> str:
        return (_dt.datetime.fromisoformat(stamp) + _dt.timedelta(minutes=minutes)).isoformat(timespec="seconds")

    earlier, later = shift(sup, -30), shift(sup, +30)
    # written EARLIER: verifies against the superseded entry, and the ordering WAS checked
    ok = REG.verify_record({**base, "written_utc": earlier}, None, p)
    assert ok["ok"] is True and ok["matched_entry"] == "superseded" and ok["ordering_checked"] is True
    # written LATER under the same stale code: does NOT verify
    bad = REG.verify_record({**base, "written_utc": later}, None, p)
    assert bad["ok"] is False and bad["matched_entry"] is None
    assert bad["stale_after_supersession"][0]["code_digest"] == OLD
    assert "after that entry was superseded" in bad["reason"] and "refit it under the registered code" in bad["reason"]
    with pytest.raises(D.StaleRecordError, match="after that entry was superseded"):
        REG.verify_record({**base, "written_utc": later}, "h3", p, raise_on_mismatch=True)
    # the timestamp may also come from the record's own run steps, or be passed in
    assert REG.record_run_utc({"steps": {"point": {"date_utc": earlier}, "intervals": {"date_utc": later}}}) == later
    assert REG.record_run_utc(base) is None
    assert REG.verify_record({**base, "steps": {"point": {"date_utc": later}}}, None, p)["ok"] is False
    assert REG.verify_record(base, None, p, written_utc=later)["ok"] is False
    # a record with NO timestamp at all cannot be ordered: it still verifies, and the result says the check did not run
    none = REG.verify_record(base, None, p)
    assert none["ok"] is True and none["matched_entry"] == "superseded" and none["ordering_checked"] is False
    assert "carries no timestamp" in none["reason"]


def test_registered_code_digests_lists_the_current_entry_then_the_superseded_ones(tmp_path):
    """The resolution a reader uses for a record SET written before a re-registration (``h3.record_set_code``)."""
    p = tmp_path / "reg.json"
    REG.register_stage("h3", below_footer_sha256=A1, code_digest="0a" * 32, git_head=None, addenda_count=1,
                       note="first", path=p)
    REG.register_stage("h3", below_footer_sha256=A1, code_digest="0b" * 32, git_head=None, addenda_count=1,
                       note="second", path=p, force=True)
    REG.register_stage("h3", below_footer_sha256=A1, code_digest="0c" * 32, git_head=None, addenda_count=1,
                       note="third", path=p, force=True)
    got = REG.registered_code_digests("h3", p)
    assert [g["code_digest"] for g in got] == ["0c" * 32, "0b" * 32, "0a" * 32]
    assert [g["matched_entry"] for g in got] == ["current", "superseded", "superseded"]
    assert got[0]["superseded_utc"] is None and got[1]["superseded_utc"]
    assert REG.registered_code_digests("power", p) == []
