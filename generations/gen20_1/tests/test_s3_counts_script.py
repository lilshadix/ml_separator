"""scripts/g201_counts.py without running S3: the re-run byte comparison (plan §9 S6), publication of a first
run, the refusals that protect the S3 reference (--no-subprocess, an existing run manifest), and the coverage audit
of the tagged document blocks (whole blocks, no blanket list of small integers, a STOP on any unmapped number)."""
from __future__ import annotations

import hashlib
import importlib.util
import json

import pytest

from gen201 import bootstrap


@pytest.fixture(scope="module")
def S():
    spec = importlib.util.spec_from_file_location("g201_counts_script", bootstrap.G201 / "scripts" / "g201_counts.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ref(d, *, extra: bytes = b""):
    (d / "row_manifests").mkdir(parents=True, exist_ok=True)
    (d / "fold_plan_g201.json").write_text('{"gate_e": {}}', encoding="utf-8")
    (d / "counts_g201.json").write_text("{}", encoding="utf-8")
    (d / "fold_records_g201.parquet").write_bytes(b"PAR1 records")
    (d / "row_manifests" / "DIGESTS.json").write_text("{}", encoding="utf-8")
    (d / "row_manifests" / "COH-D-C3.parquet").write_bytes(b"PAR1 d" + extra)
    (d / "row_manifests" / "COH-SF-C3.parquet").write_bytes(b"PAR1 sf")
    (d / "g201_counts__S3.json").write_text("{}", encoding="utf-8")          # a run manifest: not a reference file


def test_reference_files_and_identical_rerun(tmp_path, S):
    a, b = tmp_path / "ref", tmp_path / "rerun"
    _ref(a)
    _ref(b)
    names = S.reference_files(a)
    assert names == sorted(["fold_plan_g201.json", "counts_g201.json", "fold_records_g201.parquet",
                            "row_manifests/DIGESTS.json", "row_manifests/COH-D-C3.parquet",
                            "row_manifests/COH-SF-C3.parquet"])
    cmp = S.compare_reference(a, b)
    assert cmp["identical"] and cmp["n_files"] == 6 and not cmp["differing"]
    assert all(r["reference_sha256"] == hashlib.sha256((a / r["file"]).read_bytes()).hexdigest() for r in cmp["files"])


def test_rerun_difference_missing_and_extra_files_are_stops(tmp_path, S):
    a, b = tmp_path / "ref", tmp_path / "rerun"
    _ref(a)
    _ref(b, extra=b"x")
    assert S.compare_reference(a, b)["differing"] == ["row_manifests/COH-D-C3.parquet"]
    (b / "row_manifests" / "COH-NEW.parquet").write_bytes(b"PAR1 new")
    (b / "counts_g201.json").unlink()
    cmp = S.compare_reference(a, b)
    assert not cmp["identical"]
    assert set(cmp["differing"]) == {"row_manifests/COH-D-C3.parquet", "row_manifests/COH-NEW.parquet", "counts_g201.json"}
    assert S.compare_reference(tmp_path / "empty1", tmp_path / "empty2")["identical"] is False


def test_publish_copies_the_reference_files_only(tmp_path, S):
    work, out = tmp_path / "work", tmp_path / "out"
    _ref(work)
    out.mkdir()
    written = S.publish(work, out)
    assert sorted(p.relative_to(out).as_posix() for p in written) == S.reference_files(work)
    assert not (out / "g201_counts__S3.json").exists()
    assert S.compare_reference(work, out)["identical"]


def test_no_subprocess_is_refused_for_the_default_manifests_dir(S):
    with pytest.raises(SystemExit, match="no-subprocess"):
        S.main(["--no-subprocess"])
    with pytest.raises(SystemExit, match="no-subprocess"):
        S.main(["--no-subprocess", "--out", str(bootstrap.MANIFESTS)])


def test_an_existing_run_manifest_is_never_overwritten(tmp_path, S):
    (tmp_path / "g201_counts__S6.json").write_text("{}", encoding="utf-8")
    raw = (tmp_path / "g201_counts__S6.json").read_bytes()
    with pytest.raises(SystemExit, match="never overwritten"):
        S.main(["--out", str(tmp_path), "--label", "S6"])
    assert (tmp_path / "g201_counts__S6.json").read_bytes() == raw
    with pytest.raises(SystemExit, match="--label"):
        S.main(["--out", str(tmp_path), "--label", "../x"])


def test_num_variants_do_not_round_to_integers(S):
    v = S._num_variants([12.97, 0.489, 3.430, 100, 3.8511214634053776])
    assert "12.97" in v and "13" not in v and "13.0" not in v
    assert {"0.489", "49"} <= v
    assert {"3.43", "3.430"} <= v
    assert "100" in v and "1" not in v
    assert "3.85" in v and "3.8511214634053776" in v


def test_tagged_blocks_are_whole_paragraphs_items_and_rows(S):
    lines = ["intro line", "", "- item one [CNT] unless stated:", "  continues 111 cells", "- other item 222",
             "| row | 333 [CNT] |", "| row | 444 |", "para 555", "with tag [DMD] here", "", "# head [INNER]"]
    assert S.tagged_blocks(lines) == [(3, 4), (6, 6), (8, 9), (11, 11)]


def test_coverage_audit_flags_values_after_the_tag_and_small_counts(tmp_path, S, monkeypatch):
    """A number after its tag is audited; small integers are not excused wholesale (5, 14, 100 must be expected
    values or NOT_A_COUNT entries in context)."""
    doc = tmp_path / "doc.md"
    doc.write_text("Counts [CNT] unless stated:\n229 cells, 5 DMD cells, 14 metals, 100 inner folds and 2026-10-15.\n"
                   "\n| x | 3.85 [CNT] (levels csv:6,18) |\n", encoding="utf-8")
    monkeypatch.setattr(S, "DOCS", {"DOC": doc})
    T = S.Table()
    T.check("x", "s", "[CNT]", "w", 229, 229)
    rep = S.coverage_audit(T)
    assert [r["lines"] for r in rep] == [[1, 2], [4, 4]]
    assert rep[0]["unmapped"] == ["100", "14", "5"]            # the date is a NOT_A_COUNT match, 229 is expected
    assert rep[1]["unmapped"] == ["3.85"]                      # csv:6,18 is a citation
    T.check("y", "s", "[CNT]", "w", [5, 14, 100, 3.8511214634053776], [5, 14, 100, 3.8511214634053776])
    assert not any(r["unmapped"] for r in S.coverage_audit(T))


def test_real_documents_have_no_unmapped_number(S):
    """Every number in a tagged block of the three documents is an expected value of the S3 comparison table or a
    NOT_A_COUNT entry in context (checked against the recorded S3 table, when S3 has run)."""
    p = bootstrap.MANIFESTS / "counts_g201.json"
    if not p.exists():
        pytest.skip("S3 has not written manifests/counts_g201.json yet")
    T = S.Table()
    T.rows = json.loads(p.read_text(encoding="utf-8"))["comparison"]
    rep = S.coverage_audit(T)
    assert len(rep) > 40
    assert not [(r["doc"], r["lines"], r["unmapped"]) for r in rep if r["unmapped"]]
