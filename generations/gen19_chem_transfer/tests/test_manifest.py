"""Brief section 27 "manifest verification" (P7) and the provenance guard list.

* ``gen19ct.manifest.Run`` writes a deterministic manifest: a byte-identical rerun gives a byte-identical
  ``manifests/<name>.json`` while ``run_info`` stays volatile;
* ``scripts/g19_run_phaseAB.verify_manifests`` accepts an untouched output and a CRLF copy of it (LF digest),
  and rejects a tampered or missing output;
* the real step manifests verify against the files on disk (slow);
* every schema.md "Provenance" column and every target-derived column is on ``load.PROVENANCE_COLUMNS``
  (the shared loader was extended; the former strict ``xfail`` marker is gone).
* ``write_json`` writes strict JSON: NaN / inf become ``null``.

Everything that writes works under ``tmp_path`` with ``paths.REPO_ROOT`` / ``paths.MANIFESTS_DIR`` patched.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from gen19ct import paths
from gen19ct.data import load
from gen19ct.manifest import Run, write_csv, write_json

RUNNER = paths.G19_ROOT / "scripts" / "g19_run_phaseAB.py"


def _runner():
    spec = importlib.util.spec_from_file_location("g19_run_phaseAB_for_tests", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(paths, "MANIFESTS_DIR", tmp_path / "manifests")
    return tmp_path


def _toy_run(root: Path, name: str = "toy_step") -> Path:
    inp = root / "in.csv"
    if not inp.exists():
        write_csv(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}), inp)
    with Run(name, args={"k": 3}, seed=7) as run:
        run.inputs(inp)
        out = write_csv(pd.DataFrame({"a": [1.5, 2.25], "c": ["p", "q"]}), root / "out" / "table.csv")
        js = write_json(root / "out" / "summary.json", {"b": 2, "a": [1, 2]})
        run.outputs(out, js)
    return paths.MANIFESTS_DIR / f"{name}.json"


def test_run_manifest_is_byte_identical_on_rerun(sandbox: Path) -> None:
    m1 = _toy_run(sandbox).read_bytes()
    info1 = (paths.MANIFESTS_DIR / "run_info" / "toy_step.json").read_text(encoding="utf-8")
    m2 = _toy_run(sandbox).read_bytes()
    assert m1 == m2
    body = json.loads(m1)
    assert body["seed"] == 7 and body["arguments"] == {"k": 3}
    assert [o["path"] for o in body["outputs"]] == ["out/summary.json", "out/table.csv"]
    assert all(o["sha256_lf"] for o in body["outputs"])
    assert b"\r\n" not in m1
    assert "runtime_s" in json.loads(info1)          # volatile fields live only in run_info


def test_verify_manifests_detects_tampering_but_accepts_crlf(sandbox: Path) -> None:
    runner = _runner()
    _toy_run(sandbox)
    report, claimed = runner.verify_manifests(("toy_step",))
    assert report["toy_step"]["outputs_digest_mismatch"] == [] and report["toy_step"]["outputs_missing"] == []
    assert claimed == {"out/table.csv", "out/summary.json"}
    table = sandbox / "out" / "table.csv"
    table.write_bytes(table.read_bytes().replace(b"\n", b"\r\n"))      # a CRLF checkout is not a failure
    assert runner.verify_manifests(("toy_step",))[0]["toy_step"]["outputs_digest_mismatch"] == []
    table.write_text("a,c\n1.5,p\n9.99,q\n", encoding="utf-8")        # a changed value is
    assert runner.verify_manifests(("toy_step",))[0]["toy_step"]["outputs_digest_mismatch"] == ["out/table.csv"]
    (sandbox / "out" / "summary.json").unlink()
    assert runner.verify_manifests(("toy_step",))[0]["toy_step"]["outputs_missing"] == ["out/summary.json"]
    assert runner.verify_manifests(("no_such_step",))[0]["no_such_step"] == {"manifest_missing": True}


@pytest.mark.slow
def test_real_step_manifests_verify_against_disk() -> None:
    runner = _runner()
    report, _ = runner.verify_manifests(runner.STEPS)
    bad = {k: v for k, v in report.items()
           if v.get("manifest_missing") or v["outputs_digest_mismatch"] or v["outputs_missing"]}
    assert not bad, bad


def _outputs(name: str) -> dict[str, str]:
    body = json.loads((paths.MANIFESTS_DIR / f"{name}.json").read_text(encoding="utf-8"))
    return {o["path"]: o["sha256"] for o in body["outputs"]}


def _on_disk(path: str, sha: str) -> bool:
    p = paths.REPO_ROOT / path
    return p.exists() and paths.matches(p, sha)


@pytest.mark.slow
def test_preseal_chain_manifests_match_disk_or_are_superseded() -> None:
    """Task X, finding VR-01: ``g19_build_folds.json`` keeps the full build's digests of ``INDEX.json`` and
    ``wildcard_copy_crossings.csv``; ``g19_build_folds_incremental.json`` supersedes exactly those
    (``merge.full_build_link``).  Every other output of the pre-seal chain matches disk, and ``g19_run_preseal.json``
    records every stored prediction / support file and every output of the support update."""
    if not (paths.MANIFESTS_DIR / "g19_run_preseal.json").exists():
        pytest.skip("pre-seal manifests absent")
    inc = json.loads((paths.MANIFESTS_DIR / "g19_build_folds_incremental.json").read_text(encoding="utf-8"))
    link = inc["merge"]["full_build_link"]
    assert link["base_manifest_sha256"] == paths.digests(paths.MANIFESTS_DIR / "g19_build_folds.json")["sha256"]
    inc_out = _outputs("g19_build_folds_incremental")
    for path, sha in _outputs("g19_build_folds").items():
        if not _on_disk(path, sha):
            assert link["superseded_outputs"].get(path) == sha, path
            assert _on_disk(path, inc_out[path]), path
    for name in ("g19_build_folds_incremental", "g19_run_preseal", "g19_update_support_preseal"):
        bad = [path for path, sha in _outputs(name).items() if not _on_disk(path, sha)]
        assert not bad, (name, bad)
    rp = _outputs("g19_run_preseal")
    ev = paths.G19_ROOT / "evaluation" / "preseal"
    stored = {paths.rel(p) for d in ("predictions", "support") for p in (ev / d).glob("*.parquet")}
    assert stored <= set(rp) and set(_outputs("g19_update_support_preseal")) <= set(rp)


def test_difficulty_json_is_the_digest_section9_quotes() -> None:
    """``evaluation/preseal/difficulty.json`` is the as-run record whose SHA-256 pre-registration section 9 quotes;
    ``g19_run_preseal.py`` refuses a full-data rerun that would change it (task X, finding VR-01)."""
    diff = paths.G19_ROOT / "evaluation" / "preseal" / "difficulty.json"
    if not diff.exists():
        pytest.skip("pre-seal outputs absent")
    text = (paths.G19_ROOT / "preregistration_draft.md").read_text(encoding="utf-8")
    quoted = re.search(r"difficulty\.json`, SHA-256\s+`([0-9a-f]{64})`", text)
    assert quoted is not None
    assert paths.digests(diff)["sha256"] == quoted.group(1)
    spec = importlib.util.spec_from_file_location("g19_run_preseal_for_tests", paths.G19_ROOT / "scripts" / "g19_run_preseal.py")
    src = spec.origin and Path(spec.origin).read_text(encoding="utf-8")
    assert f'REGISTERED_DIFFICULTY_SHA256 = "{quoted.group(1)}"' in src


#: target-derived columns that must never be features (P7): the raw target string and the value-conflict bookkeeping
TARGET_DERIVED_COLUMNS = ("D_raw", "in_value_conflict", "duplicate_class", "duplicate_class_reason")


def _schema_provenance_columns() -> list[str]:
    text = (paths.ARCHIVE_DIR / "reports" / "schema.md").read_text(encoding="utf-8")
    sec = re.split(r"^## ", text, flags=re.MULTILINE)
    prov = next(s for s in sec if s.startswith("Provenance"))
    return re.findall(r"^\| `([^`]+)` \|", prov, flags=re.MULTILINE)


def test_schema_provenance_section_is_parsed() -> None:
    cols = _schema_provenance_columns()
    assert "canonical_measurement_id" in cols and "D_raw" in cols and len(cols) > 40


def test_provenance_guard_list_covers_schema_provenance_and_target_derived_columns() -> None:
    missing = sorted((set(_schema_provenance_columns()) | set(TARGET_DERIVED_COLUMNS)) - set(load.PROVENANCE_COLUMNS))
    assert missing == []


def test_write_json_is_strict_json(tmp_path: Path) -> None:
    p = write_json(tmp_path / "x.json", {"a": float("nan"), "b": [1.0, float("inf")], "c": 2})
    text = p.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text, parse_constant=lambda tok: pytest.fail(f"non-strict token {tok}")) == {
        "a": None, "b": [1.0, None], "c": 2}


@pytest.mark.slow
def test_written_json_outputs_are_strict() -> None:
    for p in sorted(paths.DATA_AUDIT_DIR.glob("*.json")) + sorted(paths.MANIFESTS_DIR.glob("*.json")):
        json.loads(p.read_text(encoding="utf-8"), parse_constant=lambda tok, p=p: pytest.fail(f"{p.name}: {tok}"))
