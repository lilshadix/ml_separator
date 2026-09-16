"""Tests for ``scripts/g19_seal_prereg.py`` (brief sections 21 and 24).

Every test works on copies under ``tmp_path``; the real draft is only ever *read* (and copied) and the
real ``preregistration.md`` / ``manifests/`` are never touched.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

G19 = Path(__file__).resolve().parents[1]
if str(G19) not in sys.path:
    sys.path.insert(0, str(G19))

_SPEC = importlib.util.spec_from_file_location("g19_seal_prereg", G19 / "scripts" / "g19_seal_prereg.py")
sp = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules["g19_seal_prereg"] = sp
_SPEC.loader.exec_module(sp)

REAL_DRAFT = G19 / "preregistration_draft.md"

SECTIONS = """
## 1. Questions and the primary hypothesis
H1 text.
## 3. Hold-out designs
V5.
## 4. Metrics
MAE.
## 5. Baselines
B0.
## 6. Allowed model families
M0.
## 7. Tuning procedure
inner.
## 9. Success thresholds
delta.
## 10. Failure conditions
F1.
## 12. Uncertainty evaluation
cov.
## 14. Process evaluation
MC.
## 15. Seeds and confirmation
commitment: {commitment}
## 17. Deviations from the brief
none.
"""


def _layout(tmp_path: Path, *, audit: bool = True, feasibility: bool = True) -> "sp.PreregPaths":
    root = tmp_path / "repo" / "gen19"
    (root / "manifests").mkdir(parents=True)
    if audit:
        (root / "DATA_AUDIT.md").write_bytes(b"# audit\n")
    if feasibility:
        (root / "FEASIBILITY.md").write_bytes(b"# feasibility\n")
    return sp.PreregPaths(root=root, repo_root=tmp_path / "repo")


def _commit(p: "sp.PreregPaths", tmp_path: Path) -> str:
    store = tmp_path / "outside" / "seeds.json"
    code, msg = sp.commit_seeds(p, store)
    assert code == 0, msg
    digest = sp.read_digest_file(p.seed_commitment)
    assert digest is not None
    return digest


def _ready_draft(p: "sp.PreregPaths", commitment: str, *, crlf: bool = False) -> str:
    text = "# Gen19 pre-registration\n\nStatus: final.\n" + SECTIONS.format(commitment=commitment)
    data = text.replace("\n", "\r\n") if crlf else text
    p.draft.write_bytes(data.encode("utf-8"))
    return text


# ------------------------------------------------------------------------------------------------ #
# digest stability
# ------------------------------------------------------------------------------------------------ #

def test_digest_identical_for_crlf_and_lf(tmp_path: Path) -> None:
    body = "line one\nline two with `code`\n\n| a | b |\n"
    lf, crlf = tmp_path / "lf.md", tmp_path / "crlf.md"
    lf.write_bytes(body.encode("utf-8"))
    crlf.write_bytes(body.replace("\n", "\r\n").encode("utf-8"))
    assert lf.read_bytes() != crlf.read_bytes()
    t_lf, t_crlf = sp.read_normalised(lf), sp.read_normalised(crlf)
    assert t_lf == t_crlf
    s1, d1 = sp.build_sealed_text(t_lf)
    s2, d2 = sp.build_sealed_text(t_crlf)
    assert d1 == d2 and s1 == s2


def test_digest_ignores_bom_but_not_content(tmp_path: Path) -> None:
    a = sp.build_sealed_text("text\n")[1]
    assert sp.build_sealed_text(chr(0xFEFF) + "text\r\n")[1] == a
    assert sp.build_sealed_text("text.\n")[1] != a


def test_sealed_crlf_draft_writes_lf_and_checks_in_both_line_endings(tmp_path: Path) -> None:
    p = _layout(tmp_path)
    _ready_draft(p, _commit(p, tmp_path), crlf=True)
    code, msg = sp.seal(p)
    assert code == 0, msg
    for f in (p.sealed, p.sha_file, p.seed_commitment):
        raw = f.read_bytes()
        assert b"\r" not in raw, f"{f.name} was written with CR (gen18 portability defect)"
        assert raw.endswith(b"\n")
    ok, msgs = sp.check(p)
    assert ok, msgs
    # a CRLF checkout of the sealed file and the digest file still verifies
    p.sealed.write_bytes(p.sealed.read_bytes().replace(b"\n", b"\r\n"))
    p.sha_file.write_bytes(p.sha_file.read_bytes().replace(b"\n", b"\r\n"))
    ok, msgs = sp.check(p)
    assert ok, msgs


def test_sha_file_holds_exactly_the_footer_digest(tmp_path: Path) -> None:
    p = _layout(tmp_path)
    _ready_draft(p, _commit(p, tmp_path))
    assert sp.seal(p)[0] == 0
    above, footer, below = sp.split_footer(sp.read_normalised(p.sealed))
    digest = sp.footer_digest(footer)
    assert p.sha_file.read_bytes() == (digest + "\n").encode("ascii")
    assert digest == sp.digest_of_above(above)
    assert below == ""


# ------------------------------------------------------------------------------------------------ #
# refusals
# ------------------------------------------------------------------------------------------------ #

def test_refuses_while_placeholders_remain(tmp_path: Path) -> None:
    p = _layout(tmp_path)
    text = _ready_draft(p, _commit(p, tmp_path))
    p.draft.write_bytes((text + f"\nmargin = [{sp.PLACEHOLDER} after Phase C]\n").encode("utf-8"))
    code, msg = sp.seal(p)
    assert code == 2 and sp.PLACEHOLDER in msg
    assert not p.sealed.exists() and not p.sha_file.exists()


def test_refuses_while_a_checklist_box_is_unticked(tmp_path: Path) -> None:
    p = _layout(tmp_path)
    text = _ready_draft(p, _commit(p, tmp_path))
    p.draft.write_bytes((text + "\n## 20. Pre-seal checklist\n\n- [x] done.\n- [ ] the pre-seal run was checked.\n")
                        .encode("utf-8"))
    code, msg = sp.seal(p)
    assert code == 2 and "unticked" in msg
    assert not p.sealed.exists()
    ok, msgs = sp.check(p)
    assert not ok and any("unticked" in m for m in msgs)
    p.draft.write_bytes((text + "\n## 20. Pre-seal checklist\n\n- [x] done.\n- [x] checked.\n").encode("utf-8"))
    assert sp.unchecked_box_lines(sp.read_normalised(p.draft)) == []
    assert sp.seal(p)[0] == 0


def test_refuses_while_draft_banner_present(tmp_path: Path) -> None:
    p = _layout(tmp_path)
    text = _ready_draft(p, _commit(p, tmp_path))
    p.draft.write_bytes((f"> **{sp.DRAFT_BANNER}.**\n" + text).encode("utf-8"))
    code, msg = sp.seal(p)
    assert code == 2 and "banner" in msg
    assert not p.sealed.exists()


@pytest.mark.parametrize("missing", ["audit", "feasibility"])
def test_refuses_without_audit_or_feasibility(tmp_path: Path, missing: str) -> None:
    p = _layout(tmp_path, audit=missing != "audit", feasibility=missing != "feasibility")
    _ready_draft(p, _commit(p, tmp_path))
    code, msg = sp.seal(p)
    assert code == 2
    assert ("DATA_AUDIT.md" if missing == "audit" else "FEASIBILITY.md") in msg
    assert not p.sealed.exists()


def test_refuses_without_seed_commitment_or_when_not_quoted(tmp_path: Path) -> None:
    p = _layout(tmp_path)
    _ready_draft(p, "0" * 64)
    code, msg = sp.seal(p)
    assert code == 2 and "--commit-seeds" in msg
    _commit(p, tmp_path)
    code, msg = sp.seal(p)                       # commitment exists but the draft quotes zeros
    assert code == 2 and "not quoted" in msg
    assert not p.sealed.exists()


def test_refuses_missing_required_section_and_embedded_footer(tmp_path: Path) -> None:
    p = _layout(tmp_path)
    text = _ready_draft(p, _commit(p, tmp_path))
    p.draft.write_bytes(text.replace("## 10. Failure conditions", "## 10. Other").encode("utf-8"))
    code, msg = sp.seal(p)
    assert code == 2 and "failure conditions" in msg
    p.draft.write_bytes((text + f"\n{sp.FOOTER_PREFIX} `{'a' * 64}`\n").encode("utf-8"))
    code, msg = sp.seal(p)
    assert code == 2 and "footer" in msg


def test_resealing_same_text_is_idempotent_but_changed_draft_is_refused(tmp_path: Path) -> None:
    p = _layout(tmp_path)
    text = _ready_draft(p, _commit(p, tmp_path))
    assert sp.seal(p)[0] == 0
    sealed_bytes = p.sealed.read_bytes()
    code, msg = sp.seal(p)
    assert code == 0 and msg.startswith("already sealed")
    assert p.sealed.read_bytes() == sealed_bytes
    p.draft.write_bytes((text + "\nA late change to the analysis.\n").encode("utf-8"))
    code, msg = sp.seal(p)
    assert code == 2 and "POST-HOC" in msg
    assert p.sealed.read_bytes() == sealed_bytes


# ------------------------------------------------------------------------------------------------ #
# check / tampering / POST-HOC addenda
# ------------------------------------------------------------------------------------------------ #

def _sealed(tmp_path: Path) -> "sp.PreregPaths":
    p = _layout(tmp_path)
    _ready_draft(p, _commit(p, tmp_path))
    code, msg = sp.seal(p)
    assert code == 0, msg
    return p


def test_check_detects_edit_above_footer(tmp_path: Path) -> None:
    p = _sealed(tmp_path)
    assert sp.check(p)[0]
    text = sp.read_normalised(p.sealed)
    p.sealed.write_bytes(text.replace("H1 text.", "H1 text, quietly changed.").encode("utf-8"))
    ok, msgs = sp.check(p)
    assert not ok and any("mismatch" in m for m in msgs)


def test_check_detects_tampered_or_missing_sha_file(tmp_path: Path) -> None:
    p = _sealed(tmp_path)
    p.sha_file.write_bytes(b"f" * 64 + b"\n")
    ok, msgs = sp.check(p)
    assert not ok and any("prereg_sha256.txt" in m for m in msgs)
    p.sha_file.unlink()
    ok, msgs = sp.check(p)
    assert not ok and any("missing" in m for m in msgs)


def test_check_detects_forged_footer_and_duplicate_footer(tmp_path: Path) -> None:
    p = _sealed(tmp_path)
    above, footer, below = sp.split_footer(sp.read_normalised(p.sealed))
    edited = above.replace("V5.", "V6.")
    forged = sp.digest_of_above(edited)                 # attacker recomputes the footer only
    p.sealed.write_bytes(f"{edited}\n{sp.FOOTER_PREFIX} `{forged}`\n".encode("utf-8"))
    ok, msgs = sp.check(p)
    assert not ok and any("prereg_sha256.txt" in m for m in msgs)
    p2 = _sealed(tmp_path / "second")
    txt = sp.read_normalised(p2.sealed)
    p2.sealed.write_bytes((txt + txt.splitlines()[-1] + "\n").encode("utf-8"))
    ok, msgs = sp.check(p2)
    assert not ok and any("footer lines" in m for m in msgs)


def test_check_accepts_dated_posthoc_addenda_and_rejects_others(tmp_path: Path) -> None:
    p = _sealed(tmp_path)
    base = sp.read_normalised(p.sealed)
    good = base + ("\n## POST-HOC addendum 1 (2026-10-01, orchestrator; results seen: no)\n\nChanged X because Y.\n"
                   "\n## POST-HOC addendum 2 (2026-10-03, orchestrator; results seen: yes)\n\nChanged Z.\n")
    p.sealed.write_bytes(good.encode("utf-8"))
    ok, msgs = sp.check(p)
    assert ok, msgs
    assert any("addenda below the footer: 2" in m for m in msgs)
    for bad in ("\nAn undated note.\n",
                "\n## Addendum (2026-10-01)\n\nno POST-HOC label.\n",
                "\n## POST-HOC addendum 2 (2026-10-01, x; results seen: no)\n\nskips number 1.\n",
                "\n## POST-HOC addendum 1 (2026-10-05, x; results seen: no)\n\na.\n"
                "\n## POST-HOC addendum 2 (2026-10-01, x; results seen: no)\n\ngoes back in time.\n"):
        p.sealed.write_bytes((base + bad).encode("utf-8"))
        ok, msgs = sp.check(p)
        assert not ok, bad


def test_check_unsealed_reports_blockers(tmp_path: Path) -> None:
    p = _layout(tmp_path, feasibility=False)
    p.draft.write_bytes(f"# draft\n> {sp.DRAFT_BANNER}\nx = [{sp.PLACEHOLDER}]\n".encode("utf-8"))
    ok, msgs = sp.check(p)
    text = "\n".join(msgs)
    assert not ok
    assert "FEASIBILITY.md" in text and sp.PLACEHOLDER in text and "banner" in text
    assert not p.sealed.exists()


def test_cli_check_exit_codes_on_tmp_root(tmp_path: Path, capsys) -> None:
    p = _layout(tmp_path)
    assert sp.main(["--check", "--root", str(p.root)]) == 1
    _ready_draft(p, _commit(p, tmp_path))
    assert sp.seal(p)[0] == 0
    assert sp.main(["--check", "--root", str(p.root)]) == 0
    assert "sealed:" in capsys.readouterr().out


# ------------------------------------------------------------------------------------------------ #
# confirmation seeds
# ------------------------------------------------------------------------------------------------ #

def test_commit_seeds_writes_digest_only_and_verifies(tmp_path: Path, capsys) -> None:
    p = _layout(tmp_path)
    store = tmp_path / "outside" / "seeds.json"
    code, msg = sp.commit_seeds(p, store)
    assert code == 0
    payload = json.loads(store.read_text(encoding="utf-8"))
    seeds = payload["seeds"]
    assert len(seeds) == sp.N_CONFIRMATION == len(set(seeds)) and seeds == sorted(seeds)
    assert all(sp.SEED_LOW <= s <= sp.SEED_HIGH and s not in sp.DISCOVERY_SEEDS for s in seeds)
    commitment = p.seed_commitment.read_bytes()
    assert commitment == (sp.seed_digest(payload) + "\n").encode("ascii")   # the digest and nothing else
    printed = [str(s) for s in seeds]
    # the message quotes only the 64-hex digest; no decimal seed token may appear outside it
    msg_wo_digest = msg.replace(sp.seed_digest(payload), "")
    assert all(tok not in msg_wo_digest for tok in printed)
    assert sp.load_committed_seeds(p, store) == seeds
    ok, vmsg = sp.verify_seed_store(p, store)
    assert ok and all(tok not in vmsg for tok in printed)
    # CRLF / re-indented store still verifies (canonical JSON)
    store.write_bytes(json.dumps(payload, indent=4).replace("\n", "\r\n").encode("utf-8"))
    assert sp.verify_seed_store(p, store)[0]


def test_seed_store_tampering_detected(tmp_path: Path) -> None:
    p = _layout(tmp_path)
    store = tmp_path / "outside" / "seeds.json"
    assert sp.commit_seeds(p, store)[0] == 0
    payload = json.loads(store.read_text(encoding="utf-8"))
    payload["seeds"] = sorted(payload["seeds"][:-1] + [payload["seeds"][-1] % 899_999 + 100_001])
    store.write_text(json.dumps(payload), encoding="utf-8")
    ok, msg = sp.verify_seed_store(p, store)
    assert not ok and "MISMATCH" in msg
    with pytest.raises(ValueError):
        sp.load_committed_seeds(p, store)


def test_commit_seeds_refuses_inside_repo_and_second_draw(tmp_path: Path) -> None:
    p = _layout(tmp_path)
    inside = p.repo_root / "seeds.json"
    code, msg = sp.commit_seeds(p, inside)
    assert code == 2 and "outside" in msg and not inside.exists() and not p.seed_commitment.exists()
    store = tmp_path / "outside" / "seeds.json"
    assert sp.commit_seeds(p, store)[0] == 0
    first = p.seed_commitment.read_bytes()
    code, msg = sp.commit_seeds(p, tmp_path / "outside" / "again.json")
    assert code == 2 and "already exists" in msg
    assert p.seed_commitment.read_bytes() == first


def test_seed_draw_is_deterministic_under_injected_randomness() -> None:
    seq = iter([0, 4729, 4729, 899_999, 1, 2, 3])        # 104729 is a discovery seed; a duplicate follows
    seeds = sp.draw_confirmation_seeds(5, randbelow=lambda n: next(seq))
    assert seeds == [100_000, 100_001, 100_002, 100_003, 999_999]


# ------------------------------------------------------------------------------------------------ #
# the real draft (read-only copy)
# ------------------------------------------------------------------------------------------------ #

@pytest.mark.skipif(not REAL_DRAFT.exists(), reason="preregistration_draft.md not present")
def test_real_draft_has_every_required_section_and_one_banner(tmp_path: Path) -> None:
    copy = tmp_path / "draft_copy.md"
    shutil.copyfile(REAL_DRAFT, copy)
    text = sp.read_normalised(copy)
    assert sp.missing_sections(text) == []
    assert sp.footer_line_indices(text) == []
    assert text.count(sp.DRAFT_BANNER) <= 1
    if sp.DRAFT_BANNER in text:                         # still a draft: it must also be unsealable
        assert sp.placeholder_lines(text), "a draft banner without any placeholder cannot be sealed safely"
    # the verifier's seal-gate case (M1): with the marked placeholders and the banner gone, an unticked pre-seal
    # checklist box must still block sealing
    stripped = "\n".join(ln for ln in text.split("\n") if sp.PLACEHOLDER not in ln and sp.DRAFT_BANNER not in ln)
    if sp.unchecked_box_lines(text):
        assert sp.unchecked_box_lines(stripped)
