"""Unit tests of gen201.seal (IMPLEMENTATION_PLAN.md §8 "Seal mechanics", O19): LF-normalised digests, refusal
while a draft marker remains, the last-line footer, require_sealed, and the confirmation-seed rule of
PRE_REGISTRATION.md §0 (scripts/g201_seal.py, computed in memory only)."""
from __future__ import annotations

import hashlib
import importlib.util
import json

import pytest

from gen201 import bootstrap
from gen201 import seal as SL

BODY = "# title\n\nline one\nline two with SEALED-SHA256: in the middle\n"


def _w(p, text: str, crlf: bool = False):
    data = text.replace("\n", "\r\n") if crlf else text
    p.write_bytes(data.encode("utf-8"))
    return p


def test_lf_text_and_digest_ignore_line_endings(tmp_path):
    a, b = _w(tmp_path / "a.md", BODY), _w(tmp_path / "b.md", BODY, crlf=True)
    assert SL.lf_text(a) == SL.lf_text(b) == BODY
    assert SL.lf_digest(a) == SL.lf_digest(b) == hashlib.sha256(BODY.encode()).hexdigest()
    assert hashlib.sha256(b.read_bytes()).hexdigest() != SL.lf_digest(b)


def test_state_of_an_unsealed_file(tmp_path):
    p = _w(tmp_path / "p.md", BODY + "TO BE FINALISED\n- [ ] item\n")
    st = SL.preregistration_state(p)
    assert st["exists"] and not st["sealed"] and not st["valid"] and st["recorded"] is None
    assert st["markers_remaining"] == {"TO BE FINALISED": 1, "- [ ]": 1}
    assert st["digest"] == SL.lf_digest(p)
    assert SL.preregistration_state(tmp_path / "missing.md")["exists"] is False


@pytest.mark.parametrize("marker", SL.DRAFT_MARKERS)
def test_seal_refuses_while_a_marker_remains(tmp_path, marker):
    p = _w(tmp_path / "p.md", BODY + f"status: {marker}\n")
    raw = p.read_bytes()
    with pytest.raises(RuntimeError, match="draft markers"):
        SL.seal(p)
    assert p.read_bytes() == raw


def test_seal_appends_a_valid_footer(tmp_path):
    p = _w(tmp_path / "p.md", BODY + "\n\n")
    digest = SL.seal(p)
    lines = SL.lf_text(p).rstrip("\n").split("\n")
    assert lines[-1] == f"SEALED-SHA256: {digest}"
    body = "\n".join(lines[:-1]) + "\n"
    assert digest == hashlib.sha256(body.encode("utf-8")).hexdigest()
    assert body == BODY                                   # trailing blank lines are not part of the body
    st = SL.require_sealed(p)
    assert st["sealed"] and st["valid"] and st["recorded"] == digest and st["digest"] == digest
    with pytest.raises(RuntimeError, match="already"):
        SL.seal(p)


def test_seal_keeps_crlf_and_digest_is_line_ending_free(tmp_path):
    a = _w(tmp_path / "a.md", BODY, crlf=True)
    b = _w(tmp_path / "b.md", BODY)
    da, db = SL.seal(a), SL.seal(b)
    assert da == db
    assert b"\r\n" in a.read_bytes() and a.read_bytes().endswith(b"\r\n")
    assert b"\r\n" not in b.read_bytes()
    SL.require_sealed(a)
    SL.require_sealed(b)


def test_require_sealed_rejects_tampering(tmp_path):
    p = _w(tmp_path / "p.md", BODY)
    with pytest.raises(RuntimeError, match="not sealed"):
        SL.require_sealed(p)
    SL.seal(p)
    text = SL.lf_text(p)
    _w(p, text.replace("line one", "line 1"))
    with pytest.raises(RuntimeError, match="does not match"):
        SL.require_sealed(p)
    _w(p, text[:-2] + ("0" if text[-2] != "0" else "1") + "\n")
    with pytest.raises(RuntimeError, match="does not match"):
        SL.require_sealed(p)
    with pytest.raises(RuntimeError, match="does not exist"):
        SL.require_sealed(tmp_path / "missing.md")


def test_require_sealed_rejects_a_valid_footer_over_draft_markers(tmp_path):
    """A footer appended by hand (or by another tool) over an unfinished pre-registration has a matching digest
    but is refused, as seal() would refuse it."""
    body = BODY + "status: TO BE FINALISED\n- [ ] item\n"
    p = _w(tmp_path / "p.md", body + f"SEALED-SHA256: {hashlib.sha256(body.encode()).hexdigest()}\n")
    st = SL.preregistration_state(p)
    assert st["sealed"] and st["valid"] and st["markers_remaining"]
    with pytest.raises(RuntimeError, match="draft markers remain"):
        SL.require_sealed(p)


def test_real_documents_are_only_read(tmp_path):
    for doc in (bootstrap.PREREG, bootstrap.PLAN):
        raw = doc.read_bytes()
        st = SL.preregistration_state(doc)
        assert st["exists"] and len(st["digest"]) == 64
        assert doc.read_bytes() == raw


def _seal_script():
    spec = importlib.util.spec_from_file_location("g201_seal_script", bootstrap.G201 / "scripts" / "g201_seal.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_confirmation_seed_rule():
    S = _seal_script()
    gen20 = S.gen20_confirmation_seeds()
    assert tuple(gen20) == S.GEN20_CONFIRMATION_SEEDS_DOCUMENTED == (148569, 228670, 253534, 561551, 978178)
    run_once = json.loads(S.GEN20_RUN_ONCE.read_text(encoding="utf-8"))
    assert S.commitment(gen20) == run_once["seeds_commitment"]
    seeds, trace = S.confirmation_seeds()
    assert len(seeds) == 5 and seeds == sorted(seeds) and len(set(seeds)) == 5
    excluded = set(S.DISCOVERY_SEEDS) | set(S.GEN16_CONFIRMATION_SEEDS) | set(gen20)
    assert not set(seeds) & excluded
    rule = lambda i: int(hashlib.sha256(f"gen20_1-confirmation-seed-{i}".encode()).hexdigest()[:8], 16) % 900000 + 100000  # noqa: E731
    accepted = [t["seed"] for t in trace if t["skipped"] is None]
    assert sorted(accepted) == seeds
    assert all(t["seed"] == rule(t["i"]) for t in trace)
    assert all(100000 <= s < 1000000 for s in seeds)
    assert S.commitment(seeds) == hashlib.sha256(json.dumps(sorted(seeds)).encode()).hexdigest()


def test_seal_script_refuses_without_seeds(tmp_path, capsys):
    """'seal' never touches PRE_REGISTRATION.md before the confirmation seeds are committed."""
    S = _seal_script()
    prereg = _w(tmp_path / "PRE.md", BODY)
    plan = _w(tmp_path / "PLAN.md", BODY)
    raw = prereg.read_bytes()
    assert S.cmd_seal(tmp_path, prereg, plan) == 1
    assert prereg.read_bytes() == raw
    assert "seeds" in capsys.readouterr().out


SEEDS = [215593, 276871, 348239, 385670, 915723]
GATES_OK = {"a": True, "b": True, "b_subprocess": True, "c": True, "d": True, "e": True}


def _s3_outputs(d, S, *, counts: dict | None = None, seeds=SEEDS):
    """Fake S3 reference files and committed seeds in ``d`` (the seal script only reads and hashes them)."""
    (d / "confirmation_seeds.json").write_text(json.dumps({"seeds": seeds}), encoding="utf-8")
    (d / "confirmation_seeds_sha256.txt").write_text(S.commitment(seeds) + "\n", encoding="utf-8")
    c = {"s3_status": "PASS", "n_mismatches": 0, "gates": dict(GATES_OK), "frozen_trees_unchanged": True}
    c.update(counts or {})
    (d / "counts_g201.json").write_text(json.dumps(c), encoding="utf-8")
    (d / "fold_plan_g201.json").write_text("{}", encoding="utf-8")
    (d / "fold_records_g201.parquet").write_bytes(b"PAR1 fake")
    (d / "row_manifests").mkdir(exist_ok=True)
    (d / "row_manifests" / "DIGESTS.json").write_text("{}", encoding="utf-8")


def _prereg_with_seeds(p, seeds=SEEDS, crlf=False):
    return _w(p, BODY + "confirmation seeds: " + ", ".join(map(str, seeds)) + "\n", crlf=crlf)


def test_seal_script_flow_on_temporary_copies(tmp_path, capsys):
    """'seal' on temporary files: checks the S4 preconditions, seals the pre-registration, records the LF digest
    of the plan, the seed commitment and the sha256 of the S3 reference files in seal_g201.json (the real
    documents are never touched here)."""
    S = _seal_script()
    _s3_outputs(tmp_path, S)
    prereg = _prereg_with_seeds(tmp_path / "PRE.md", crlf=True)
    plan = _w(tmp_path / "PLAN.md", BODY + "more\n", crlf=True)
    assert S.cmd_seal(tmp_path, prereg, plan) == 0
    st = SL.require_sealed(prereg)
    rec = json.loads((tmp_path / "seal_g201.json").read_text(encoding="utf-8"))
    assert rec["preregistration"]["sha256_lf_body"] == st["digest"]
    assert rec["implementation_plan"]["sha256_lf_whole_file"] == SL.lf_digest(plan)
    assert rec["confirmation_seeds"]["commitment_sha256"] == S.commitment(SEEDS)
    ref = rec["s3_reference_sha256"]
    assert set(ref) == set(S.S3_REFERENCE_FILES)
    for n, h in ref.items():
        assert h == hashlib.sha256((tmp_path / n).read_bytes()).hexdigest()
    assert rec["s4_preconditions"]["seeds_written_in_preregistration"] is True
    capsys.readouterr()


@pytest.mark.parametrize("case", ["no_counts", "stop", "mismatch", "gate_e", "no_subprocess_gate", "frozen",
                                  "seeds_absent", "reference_missing"])
def test_seal_script_refuses_unless_s4_preconditions_hold(tmp_path, capsys, case):
    """'seal' never touches PRE_REGISTRATION.md unless S3 passed (status PASS, 0 mismatches, every gate including
    the subprocess gate, frozen trees unchanged), every committed seed is written in its body, and the S3
    reference files are present."""
    S = _seal_script()
    counts = {"stop": {"s3_status": "STOP"}, "mismatch": {"n_mismatches": 2},
              "gate_e": {"gates": {**GATES_OK, "e": False}},
              "no_subprocess_gate": {"s3_status": "PASS", "gates": {k: v for k, v in GATES_OK.items() if k != "b_subprocess"}},
              "frozen": {"frozen_trees_unchanged": False}}.get(case)
    _s3_outputs(tmp_path, S, counts=counts)
    if case == "no_counts":
        (tmp_path / "counts_g201.json").unlink()
    if case == "reference_missing":
        (tmp_path / "row_manifests" / "DIGESTS.json").unlink()
    prereg = _w(tmp_path / "PRE.md", BODY) if case == "seeds_absent" else _prereg_with_seeds(tmp_path / "PRE.md")
    plan = _w(tmp_path / "PLAN.md", BODY)
    raw = prereg.read_bytes()
    assert S.cmd_seal(tmp_path, prereg, plan) == 1
    assert prereg.read_bytes() == raw
    assert not (tmp_path / "seal_g201.json").exists()
    assert "S4 preconditions fail" in capsys.readouterr().out
