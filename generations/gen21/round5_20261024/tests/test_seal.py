"""Tests for scripts/r5_seal.py and r5/sealcheck.py on temporary copies (light manifests: the round-5 folder's own
documents, code, inputs and copy digests): the append-only marker, refusal on a changed file, the sealed-input set
(every results/*/ITEM_SPEC.md, manifests/ea_check_caches.json, the 100 cache/{A,B,BR,BQ}_<seed>_<fold>.pkl,
cache/frozen/, results/stage0/ in full) and refusal on a missing required input, reseal rules, the copy rule, and the
read-only round-3 LF check; the real folder's input set is digested (construction only).  Added at the pre-seal review:
whole-line markers only (s0-mechanics-1), prefix checks of every append-only text (s0-mechanics-2), COPIED_FROM
declarations and the real-tree copy records (s0-mechanics-7), the extended required-input set (s0-mechanics-10), no
scored output before the first seal and named scored outputs on a reseal (s0-mechanics-24), and the sealcheck helpers
(post_run, refuse_if_sealed, reduced coverage, run_sealed).  Nothing here seals the real round-5 folder or runs the
round-4 verifier.

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B -W ignore -m pytest generations/gen21/round5_20261024/tests -q
        -p no:cacheprovider --basetemp generations/gen21/round5_20261024/cache/_pytest   (one command line; temporary
        files stay inside the round-5 folder)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r5 import paths  # noqa: E402
from r5 import sealcheck  # noqa: E402

S = sealcheck.seal_module()
MARK = "<!-- APPEND-ONLY BELOW -->"


def make_base(tmp: Path) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    for d in S.DOCS:
        body = f"# {d}\n\nsealed text of {d}\n"
        if d == "DECISION_LOG.md":
            body += f"\n{MARK}\n\n- request U1: open\n"
        (tmp / d).write_bytes(body.encode())
    (tmp / S.LEDGER).write_bytes(b"lab_id,n\nL1,1\n")
    (tmp / "AMENDMENTS.md").write_bytes(b"# Round-5 amendments (append-only)\n")
    (tmp / "RUN_RECORD.md").write_bytes(b"# Run record\n")
    files = [("r5/x.py", b"X = 1\n"), ("scripts/y.py", b"print('y')\n"), ("tests/z.py", b"Z = 0\n"),
             ("results/stage0/a.json", b'{"a": 1}\n'), ("cache/ext/b.bin", b"\x00\x01\x02"),
             ("cache/frozen/fold_all_BPG_1_0.pkl", b"\x80\x04frozen")]
    files += [(rel, f"stand-in for {rel}\n".encode() if not rel.endswith(".pkl") else b"\x80\x04" + rel.encode())
              for rel in S.required_inputs()]                    # item specs, EA manifest, 100 EA caches, D3 size check
    for rel, data in files:
        p = tmp / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return tmp


def test_append_only_marker_digests_only_the_text_before_it(tmp_path):
    p = tmp_path / "DOC.md"
    p.write_bytes(f"head\n{MARK}\nrow 1\n".encode())
    h1, m1 = S.doc_digest(p)
    assert m1
    p.write_bytes(f"head\n{MARK}\nrow 1\nrow 2\n".encode())
    assert S.doc_digest(p) == (h1, True)
    p.write_bytes(f"head\r\n{MARK}\r\nrow 1\r\n".encode())          # CRLF checkout: same digest
    assert S.doc_digest(p) == (h1, True)
    p.write_bytes(f"head changed\n{MARK}\nrow 1\n".encode())
    assert S.doc_digest(p)[0] != h1
    p.write_bytes(b"head\nrow 1\n")                                    # marker removed: the whole text, flag off
    h2, m2 = S.doc_digest(p)
    assert not m2 and h2 != h1


def test_light_seal_verifies_and_refuses_a_changed_file(tmp_path):
    base = make_base(tmp_path)
    s = S.seal(base, "test seal", full=False)
    assert S.verify(base, full=False) == (True, "ok", s)
    assert sealcheck.require_seal(base, full=False, quiet=True) == s[:12]
    assert not S.verify(base, full=True)[0]                          # a light manifest never passes a full verify
    log = base / "DECISION_LOG.md"
    log.write_bytes(log.read_bytes() + b"- request U2: answered\n")   # below the marker: still sealed
    assert S.verify(base, full=False)[0]
    changes = [("r5/x.py", b"X = 2\n", "code_local"), ("PROTOCOL.md", b"# edited\n", "docs"),
               ("results/stage0/a.json", b'{"a": 2}\n', "inputs_local"), (S.LEDGER, b"lab_id,n\nL1,2\n", "inputs_local"),
               ("scripts/y.py", b"print('z')\n", "code_local")]
    for rel, new, key in changes:
        p = base / rel
        old = p.read_bytes()
        p.write_bytes(new)
        ok, msg, _ = S.verify(base, full=False)
        assert not ok and msg.startswith(key) and rel in msg, (rel, msg)
        with pytest.raises(SystemExit):
            sealcheck.require_seal(base, full=False, quiet=True)
        p.write_bytes(old)
        assert S.verify(base, full=False)[0], rel
    extra = base / "cache" / "ext" / "new.bin"                         # a new input file
    extra.write_bytes(b"x")
    assert not S.verify(base, full=False)[0]
    extra.unlink()
    man = base / "manifests" / "stage0_manifest.json"
    man.write_bytes(man.read_bytes() + b" ")
    assert S.verify(base, full=False)[1] == "manifest digest != seal"


def _ea(fam: str = "BQ", k: int = 4) -> str:
    return f"cache/{fam}_{paths.DEV_SEEDS[-1]}_{k}.pkl"


def test_sealed_inputs_cover_item_specs_ea_caches_frozen_and_stage0(tmp_path):
    base = make_base(tmp_path)
    inp = S.local_inputs(base)
    req = S.required_inputs()
    assert len(req) == len(set(req)) == 9 + 1 + 100 + 1 + len(S.STAGE0_FILES) + len(S.FROZEN_OBJECTS) == 135
    assert set(req) <= set(inp) and "MISSING" not in inp.values()
    assert sum(1 for k in inp if S.EA_CACHE.fullmatch(k.split("/")[-1]) and k.count("/") == 1) == 100
    assert {"results/stage0/a.json", "cache/frozen/fold_all_BPG_1_0.pkl", "cache/ext/b.bin", S.EA_MANIFEST,
            S.ECT_SIZE_CHECK} <= set(inp)
    assert {f"results/{it}/ITEM_SPEC.md" for it in S.ITEMS} == {k for k in inp if k.endswith("/ITEM_SPEC.md")}
    s = S.seal(base, "test seal", full=False)
    changes = [("results/ECT/ITEM_SPEC.md", b"# edited spec\n"), (S.EA_MANIFEST, b'{"records": {}}\n'),
               (_ea(), b"\x80\x04changed"), (_ea("A", 0), b"\x80\x04changed"),
               ("cache/frozen/fold_all_BPG_1_0.pkl", b"\x80\x04changed"), (S.ECT_SIZE_CHECK, b'{"size": 1}\n')]
    for rel, new in changes:
        p = base / rel
        old = p.read_bytes()
        p.write_bytes(new)
        ok, msg, _ = S.verify(base, full=False)
        assert not ok and msg.startswith("inputs_local") and rel in msg, (rel, msg)
        p.write_bytes(old)
        assert S.verify(base, full=False) == (True, "ok", s), rel
    spec = base / "results" / "EW" / "ITEM_SPEC.md"                  # a CRLF checkout of a spec: same LF digest
    spec.write_bytes(spec.read_bytes().replace(b"\n", b"\r\n"))
    assert S.verify(base, full=False)[0]
    for rel in ("results/NEWITEM/ITEM_SPEC.md", "cache/A_999_0.pkl", "results/stage0/sub/new.json"):
        p = base / rel                                                 # a new spec, EA cache or Stage-0 file: refused
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"new\n")
        ok, msg, _ = S.verify(base, full=False)
        assert not ok and msg.startswith("inputs_local") and rel in msg, (rel, msg)
        p.unlink()
        assert S.verify(base, full=False)[0], rel
    for rel in ("cache/_inner/x.pkl", "cache/notes.pkl", "results/leakage/poison.json"):    # not sealed inputs
        p = base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"scratch\n")
    assert S.verify(base, full=False)[0]


@pytest.mark.parametrize("rel", ["results/ECT_M03/ITEM_SPEC.md", "manifests/ea_check_caches.json", "cache/BR_{s}_2.pkl",
                                 "results/stage0/size_gate_ect_chemotype.json", "results/stage0/beta_frozen.json",
                                 "results/stage0/supply_audit.json", "cache/ext/ext_dopt.pkl",
                                 "cache/frozen/fold_all_BP_104729_0.pkl"])
def test_seal_refuses_a_missing_required_input(tmp_path, rel):
    rel = rel.format(s=paths.DEV_SEEDS[0])
    base = make_base(tmp_path)
    (base / rel).unlink()
    assert S.local_inputs(base)[rel] == "MISSING"
    with pytest.raises(SystemExit, match=f"missing input {rel}"):
        S.seal(base, "x", full=False)
    assert not (base / "manifests" / "stage0_seal.txt").exists()


def test_real_folder_inputs_construction_only():
    """The real round-5 folder's sealed-input set (digests only; nothing is sealed): every item spec, the EA-CHECK
    manifest and its 100 caches (byte digests equal to the manifest's records), every Stage-0 and cache/frozen file."""
    import json
    inp = S.local_inputs(paths.R5)
    for rel in S.required_inputs():
        assert inp[rel] != "MISSING", rel
    rec = json.loads((paths.R5 / S.EA_MANIFEST).read_bytes())["records"]
    ea = {k: v for k, v in inp.items() if k.startswith("cache/") and S.EA_CACHE.fullmatch(k[len("cache/"):])}
    assert len(ea) == len(rec) == 100
    assert all(ea[v["path"]] == v["sha256"] for v in rec.values())
    for sub in ("results/stage0", "cache/frozen", "cache/ext"):
        files = {S._rel(p, paths.R5) for p in (paths.R5 / sub).rglob("*")
                 if p.is_file() and "__pycache__" not in p.parts}
        assert files and files <= set(inp), sub
    specs = {S._rel(p, paths.R5) for p in paths.R5.glob(S.ITEM_SPEC_GLOB)}
    assert specs == {k for k in inp if k.endswith("ITEM_SPEC.md")} and len(specs) >= len(S.ITEMS)
    assert not any(k.startswith(("cache/_pytest", "cache/_inner", "manifests/stage0")) for k in inp)


def test_reseal_needs_an_amendment_naming_the_old_seal(tmp_path):
    base = make_base(tmp_path)
    s1 = S.seal(base, "first", full=False)
    with pytest.raises(SystemExit):
        S.seal(base, "second", full=False)
    with pytest.raises(SystemExit):
        S.seal(base, "", full=False)
    am = base / "AMENDMENTS.md"
    am.write_bytes(am.read_bytes() + f"\n## A1\nsupersedes {s1[:12]}; results seen: none\n".encode())
    s2 = S.seal(base, "A1", full=False)
    assert s2 != s1
    arch = base / "manifests" / f"stage0_manifest_{s1[:12]}.json"
    assert arch.exists() and (base / "manifests" / f"stage0_seal_{s1[:12]}.txt").exists()
    import json
    assert json.loads((base / "manifests" / "stage0_manifest.json").read_bytes())["supersedes"] == s1
    assert (base / "RUN_RECORD.md").read_bytes().count(b" SEAL ") == 2
    assert b"\r\n" not in (base / "manifests" / "stage0_manifest.json").read_bytes()


def test_seal_refuses_a_missing_document_or_ledger(tmp_path):
    base = make_base(tmp_path)
    (base / "PROTOCOL.md").unlink()
    with pytest.raises(SystemExit, match="missing document PROTOCOL.md"):
        S.seal(base, "x", full=False)
    (base / "PROTOCOL.md").write_bytes(b"# p\n")
    (base / S.LEDGER).unlink()
    with pytest.raises(SystemExit, match="missing SUPPLY_LEDGER.csv"):
        S.seal(base, "x", full=False)
    assert not (base / "manifests" / "stage0_seal.txt").exists()


def test_copy_rule_digest_is_checked(tmp_path):
    src = paths.DESIGN / "probes" / "scripts" / "P1_ct_build.py"
    h = paths.sha256_text_lf(src)
    base = make_base(tmp_path / "good")
    (base / "scripts" / "c.py").write_bytes(
        f"# copied from round5_design_20261023/probes/scripts/P1_ct_build.py (LF SHA-256 {h})\nA = 1\n".encode())
    (base / "r5" / "d.py").write_bytes(f'"""Adapted from probes/scripts/P1_ct_build.py\n(sha256_lf={h})."""\n'.encode())
    rec = S.copied_sources(base)["scripts/c.py"]
    assert len(rec) == 1 and rec[0]["match"] and rec[0]["resolved"].endswith("probes/scripts/P1_ct_build.py")
    assert S.copied_sources(base)["r5/d.py"][0]["match"]
    S.seal(base, "ok", full=False)
    bad = make_base(tmp_path / "bad")
    (bad / "scripts" / "c.py").write_bytes(
        f"# copied from round5_design_20261023/probes/scripts/P1_ct_build.py (LF SHA-256 {'0' * 64})\n".encode())
    with pytest.raises(SystemExit, match="copied source changed"):
        S.seal(bad, "x", full=False)
    other = make_base(tmp_path / "other")                              # a hex without 'SHA-256' is not a copy record
    (other / "scripts" / "c.py").write_bytes(f"# runs scripts/r4_seal.py; seal {'a' * 64}\n".encode())
    assert S.copied_sources(other) == {}


def test_relative_imports_resolve_to_the_r5_package():
    names = S._ast_imports([paths.R5 / "r5" / "common.py"])
    assert {"r5.paths", "r4.metrics"} <= names


def test_round3_lf_digests_still_match():
    r = S.round3_check()
    assert r["ok"] and r["code_files"] == 78 and r["seal"] == S.R3_SEAL


# ------------------------------------------------------------------------------------------------------------
# pre-seal review additions
# ------------------------------------------------------------------------------------------------------------
def test_inline_marker_mention_never_truncates(tmp_path):
    """s0-mechanics-1: only a whole-line marker truncates the digest; PROTOCOL's inline mention does not."""
    p = tmp_path / "DOC.md"
    p.write_bytes(f"head\nstatus goes below the line `{MARK}` of X\nmore sealed text\n".encode())
    h, m = S.doc_digest(p)
    assert not m and h == S._sha(p.read_bytes())
    p.write_bytes(f"head\nstatus goes below the line `{MARK}` of X\nmore sealed text, edited\n".encode())
    assert S.doc_digest(p)[0] != h
    p.write_bytes(f"head\n{MARK}  \nrow\n".encode())                 # trailing blanks: still a whole line
    assert S.doc_digest(p) == (S._sha(b"head\n"), True)


def test_real_protocol_is_digested_in_full():
    p = paths.R5 / "PROTOCOL.md"
    h, m = S.doc_digest(p)
    assert not m and h == S._sha(p.read_bytes().replace(b"\r\n", b"\n"))


@pytest.mark.parametrize("doc,lines,ok", [("PROTOCOL.md", 1, False), ("DECISION_LOG.md", 2, False),
                                          ("README.md", 1, True)])
def test_marker_lines_where_allowed(tmp_path, doc, lines, ok):
    base = make_base(tmp_path)
    body = f"# {doc}\n\nsealed\n" + "".join(f"\n{MARK}\n\n- row {i}\n" for i in range(lines))
    (base / doc).write_bytes(body.encode())
    if ok:
        S.seal(base, "x", full=False)
    else:
        with pytest.raises(SystemExit, match="marker line"):
            S.seal(base, "x", full=False)


def test_append_only_texts_are_prefix_checked(tmp_path):
    """s0-mechanics-2: appending to AMENDMENTS / RUN_RECORD / a marker tail verifies; editing an earlier line does not."""
    base = make_base(tmp_path)
    s = S.seal(base, "test seal", full=False)
    assert S.verify(base, full=False) == (True, "ok", s)
    for rel in ("AMENDMENTS.md", "RUN_RECORD.md", "DECISION_LOG.md"):
        f = base / rel
        old = f.read_bytes()
        f.write_bytes(old + b"- appended line\n")
        assert S.verify(base, full=False)[0], rel
        f.write_bytes(old)
    for rel, a, b in (("AMENDMENTS.md", b"# Round-5 amendments", b"# Round-5 AMENDMENTS"),
                      ("RUN_RECORD.md", b"# Run record", b"# Run log"),
                      ("DECISION_LOG.md", b"- request U1: open", b"- request U1: granted")):
        f = base / rel
        old = f.read_bytes()
        assert a in old
        f.write_bytes(old.replace(a, b))
        ok, msg, _ = S.verify(base, full=False)
        assert not ok and msg.startswith("append_only") and rel in msg, (rel, msg)
        f.write_bytes(old)
        assert S.verify(base, full=False)[0], rel


def test_reseal_rules_append_only_and_scored_outputs(tmp_path):
    """s0-mechanics-24 and -2: no scored output before the first seal; a reseal needs an AMENDMENTS entry appended
    since the previous seal naming its prefix and every scored output, and intact append-only prefixes."""
    base = make_base(tmp_path)
    out = base / "results" / "ECT" / "rows_BPG.parquet"
    out.write_bytes(b"PAR1")
    with pytest.raises(SystemExit, match="scored outputs exist before the first seal"):
        S.seal(base, "x", full=False)
    out.unlink()
    (base / "results" / "ECT" / "SMOKE_COUNTS_BPG.json").write_bytes(b"{}\n")      # smoke counts are not scored
    s1 = S.seal(base, "first", full=False)
    out.write_bytes(b"PAR1")
    am = base / "AMENDMENTS.md"
    am.write_bytes(am.read_bytes() + f"\n## A1\nsupersedes {s1[:12]}; results seen: none\n".encode())
    with pytest.raises(SystemExit, match="every scored output"):
        S.seal(base, "A1", full=False)
    am.write_bytes(am.read_bytes() + b"results seen: results/ECT/rows_BPG.parquet\n")
    s2 = S.seal(base, "A1", full=False)
    am.write_bytes(am.read_bytes().replace(b"supersedes", b"replaces"))       # an earlier entry edited
    with pytest.raises(SystemExit, match="edited, not appended"):
        S.seal(base, "A2", full=False)
    assert s2 != s1


def test_reseal_needs_the_old_prefix_in_the_appended_part(tmp_path):
    base = make_base(tmp_path)
    s1 = S.seal(base, "first", full=False)
    am = base / "AMENDMENTS.md"
    am.write_bytes(am.read_bytes() + b"\n## A1\nsupersedes nothing in particular; results seen: none\n")
    with pytest.raises(SystemExit, match="appended since that seal"):
        S.seal(base, "A1", full=False)
    am.write_bytes(am.read_bytes() + f"supersedes {s1[:12]}\n".encode())
    assert S.seal(base, "A1", full=False) != s1


def test_copied_from_declaration_is_checked(tmp_path):
    src = "generations/gen21/round5_design_20261023/probes/scripts/P1_ct_build.py"
    h = paths.sha256_text_lf(paths.ROOT / src)
    base = make_base(tmp_path / "good")
    (base / "r5" / "c.py").write_bytes(f'COPIED_FROM = {{"{src}": "{h}"}}\nX = 1\n'.encode())
    rec = S.copied_sources(base)["r5/c.py"]
    assert len(rec) == 1 and rec[0]["match"] and rec[0]["declared"]
    S.seal(base, "ok", full=False)
    bad = make_base(tmp_path / "bad")
    (bad / "r5" / "c.py").write_bytes(f'COPIED_FROM = {{"{src}": "{"0" * 64}"}}\n'.encode())
    with pytest.raises(SystemExit, match="copied source changed"):
        S.seal(bad, "x", full=False)


def test_real_tree_copy_records():
    """s0-mechanics-7: every copying module declares COPIED_FROM (39 repository .py sources and 3 design texts); every
    record resolves and matches; no record names the session-scratchpad eb_checks.py."""
    cs = S.copied_sources(paths.R5)
    decl = {f: S.declared_copies(paths.R5 / f) for f in cs}
    assert all(d is not None for d in decl.values())
    recs = [r for v in cs.values() for r in v]
    assert len(recs) == sum(len(d) for d in decl.values()) == 42
    assert sum(r["source"].endswith(".py") for r in recs) == 39
    assert all(r["match"] and r["resolved"] for r in recs)
    assert not any("eb_checks" in r["source"] for r in recs)


def test_design_digests_cover_binaries():
    """s0-mechanics-8: the E-LAB reproduction reference lab_rows2.parquet is digested (byte digest)."""
    des, _ = S.design_digests()
    key = "generations/gen21/round5_design_20261023/ideas2/lab_out/lab_rows2.parquet"
    assert key in des and des[key] == paths.sha256_file(paths.ROOT / key)


# ------------------------------------------------------------------------------------------------------------
# r5.sealcheck helpers
# ------------------------------------------------------------------------------------------------------------
def test_post_run_records_and_never_raises(tmp_path):
    base = make_base(tmp_path)
    s = S.seal(base, "x", full=False)
    assert sealcheck.post_run(s[:12], "TESTITEM", "note", base=base, full=False)
    assert b"END TESTITEM seal=" + s[:12].encode() + b" post-run check ok" in (base / "RUN_RECORD.md").read_bytes()
    assert (base / "results" / "leakage" / "postrun_TESTITEM.json").exists()
    (base / "PROTOCOL.md").write_bytes(b"# edited\n")
    assert not sealcheck.post_run(s[:12], "TESTITEM", base=base, full=False)
    assert b"post-run check FAILED" in (base / "RUN_RECORD.md").read_bytes()
    assert sealcheck.post_run(s[:12], "TESTITEM", base=base, full=False) is False   # no exception
    assert S.scored_outputs(base) == []                                             # postrun_* is not a score


def test_refuse_if_sealed(tmp_path):
    sf = tmp_path / "stage0_seal.txt"
    sealcheck.refuse_if_sealed("x", seal_file=sf)
    sf.write_bytes(b"abc  stage0_manifest.json\n")
    with pytest.raises(SystemExit, match="sealed"):
        sealcheck.refuse_if_sealed("x", seal_file=sf)
    sealcheck.refuse_if_sealed("x", True, seal_file=sf)


def test_reduced_coverage_is_refused_outside_smoke():
    n = len(paths.DEV_SEEDS)
    assert sealcheck.reduced_coverage(a=None, b=False, c=n) == []
    assert sealcheck.reduced_coverage(a=0, b=True, c=1) == ["a=0", "b=True", "c=1"]
    sealcheck.forbid_reduced_coverage(True, a=0)
    with pytest.raises(SystemExit, match="reduced leakage coverage"):
        sealcheck.forbid_reduced_coverage(False, intervention_seeds=0)
    sealcheck.forbid_reduced_coverage(False, intervention_seeds=n, no_lopo=False)


def test_run_sealed_records_start_end_and_stop(tmp_path, monkeypatch):
    from r5 import common as CM
    rr = tmp_path / "RUN_RECORD.md"
    monkeypatch.setattr(CM, "RUN_RECORD", rr)
    seen = []
    monkeypatch.setattr(sealcheck, "post_run", lambda tag, item, note="", **k: seen.append(note) or True)
    assert sealcheck.run_sealed("IT", "run", "TAG", lambda: (0, "detail")) == 0
    assert seen[-1] == "run status=0; detail" and b"START IT run seal=TAG" in rr.read_bytes()
    with pytest.raises(SystemExit, match="boom"):
        sealcheck.run_sealed("IT", "run", "TAG", lambda: sealcheck.stop("IT", "run", "TAG", "boom"))
    assert seen[-1].startswith("run STOP (boom)") and b"STOP IT run seal=TAG boom" in rr.read_bytes()
    monkeypatch.setattr(sealcheck, "post_run", lambda *a, **k: False)
    assert sealcheck.run_sealed("IT", "run", "TAG", lambda: 2) == 6

