"""LT-26 (the part that exists at S3): forbidden writers of frozen generations raise; gen20.1 write targets are
confined to generations/gen20_1; a representative S3 workload leaves every frozen tree and read-only input
byte-for-byte untouched (no file, no __pycache__).  The session-wide snapshot check lives in conftest.py."""
from __future__ import annotations

import importlib
import os
import sys
import textwrap

import pytest

from gen201 import bootstrap
from gen201 import guard
from gen201.run import Run


def test_forbidden_writers_raise():
    inst = guard.install()
    assert inst["gen13sep.cohort"] == ["write_cohort"]
    assert set(inst["gen20k.manifest"]) == {"Run", "preregistration_state", "require_sealed_preregistration",
                                            "seal_preregistration", "environment"}
    assert set(inst["gen20k.report"]) == {"load_tables", "build_design", "figures", "setting_b_tables"}
    assert inst["gen13sep.runner"] == ["run_ladder"]
    assert inst["gen14.dirbench"] == ["load"]
    import gen13sep.cohort as C
    import gen13sep.runner as GR
    import gen20k.manifest as MF
    import gen20k.report as RP
    for call in (lambda: C.write_cohort(None), lambda: MF.Run("x"), lambda: MF.seal_preregistration(),
                 lambda: MF.require_sealed_preregistration(), lambda: MF.preregistration_state(),
                 lambda: MF.environment(), lambda: RP.load_tables("discovery"), lambda: RP.figures("x", None),
                 lambda: RP.build_design(None, "BP", None), lambda: RP.setting_b_tables("x", None),
                 lambda: GR.run_ladder(None, None, [], None)):
        with pytest.raises(guard.ForbiddenWriterError):
            call()


def test_gen14_rebuild_is_refused():
    guard.install()
    import gen14.dirbench as DB
    with pytest.raises(guard.ForbiddenWriterError):
        DB.load(rebuild=True)


def test_torch_modules_are_not_imported():
    """gen20k.runner and gen20k.nnc import torch; S3 never imports them and the guard does not either."""
    guard.install()
    assert "gen20k.runner" not in sys.modules and "gen20k.nnc" not in sys.modules
    assert "torch" not in sys.modules


def test_lazy_modules_are_patched_on_import(tmp_path, monkeypatch):
    """The post-import hook patches a guarded module the moment it is imported (exercised on a stand-in
    module so that torch is never imported)."""
    pkg = tmp_path / "g201fakepkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "lazy.py").write_text(textwrap.dedent("""
        def run(*a, **k):
            return "wrote into a frozen tree"
    """), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(guard, "_LAZY", guard._LAZY + ("g201fakepkg.lazy",))
    monkeypatch.setitem(guard.FORBIDDEN, "g201fakepkg.lazy", ("run",))
    guard.install()
    try:
        mod = importlib.import_module("g201fakepkg.lazy")
        with pytest.raises(guard.ForbiddenWriterError):
            mod.run()
    finally:
        for m in ("g201fakepkg.lazy", "g201fakepkg"):
            sys.modules.pop(m, None)


def test_write_targets_are_confined(tmp_path):
    assert bootstrap.assert_under_g201(bootstrap.MANIFESTS / "x.json") == (bootstrap.MANIFESTS / "x.json").resolve()
    assert bootstrap.assert_under_g201(tmp_path / "scratch") == (tmp_path / "scratch").resolve()
    for frozen in bootstrap.FROZEN_TREES:
        with pytest.raises(PermissionError):
            bootstrap.assert_under_g201(frozen / "manifests" / "x.json")
    with pytest.raises(PermissionError):
        bootstrap.assert_under_g201(bootstrap.ROOT / "docs" / "x.md")
    with pytest.raises(PermissionError):
        Run("x", manifest_dir=bootstrap.G20 / "manifests")


def test_outside_the_repository_only_the_temporary_directory_is_scratch(tmp_path):
    """A target outside the repository is allowed only under tempfile.gettempdir() (pytest's tmp_path is
    there); anything else outside the repository is refused."""
    import tempfile
    from pathlib import Path
    assert any(bootstrap._within(tmp_path.resolve(), s) for s in bootstrap.scratch_roots())
    elsewhere = Path(bootstrap.ROOT.resolve().anchor) / "g201_not_a_scratch_dir" / "x.json"
    if not any(bootstrap._within(elsewhere, s) for s in bootstrap.scratch_roots()):
        with pytest.raises(PermissionError, match="outside generations/gen20_1 and outside the temporary"):
            bootstrap.assert_under_g201(elsewhere)
    assert bootstrap.assert_under_g201(Path(tempfile.gettempdir()) / "g201_scratch" / "x.json")


@pytest.mark.skipif(os.name != "nt", reason="the Git-Bash drive-path hazard exists on Windows only")
def test_git_bash_drive_paths_are_refused():
    """'/d/ml_separator_gh/...' would resolve to '<drive>:\\d\\ml_separator_gh\\...' under Windows Python
    (IMPLEMENTATION_PLAN.md §8, runner.py:324); it is refused before resolving, in str and Path form."""
    from pathlib import Path
    for p in ("/d/ml_separator_gh/generations/gen20_1/manifests", "/c/Users/x/AppData/Local/Temp/x", "/d",
              Path("/d/ml_separator_gh/generations/gen20_1/manifests")):
        with pytest.raises(PermissionError, match="Git-Bash"):
            bootstrap.assert_under_g201(p)
    # a relative path, a drive path and a UNC-looking path are not mistaken for one
    assert bootstrap.assert_under_g201("generations/gen20_1/manifests") == bootstrap.MANIFESTS.resolve()
    assert bootstrap.assert_under_g201(str(bootstrap.MANIFESTS)) == bootstrap.MANIFESTS.resolve()


def test_git_state_never_takes_optional_locks(monkeypatch):
    """git_state() must not refresh .git/index: every git call carries --no-optional-locks and
    GIT_OPTIONAL_LOCKS=0."""
    from gen201 import env
    calls = []

    class _Done:
        stdout = ""

    def fake_run(cmd, **kw):
        calls.append((list(cmd), dict(kw.get("env") or {})))
        return _Done()
    monkeypatch.setattr(env.subprocess, "run", fake_run)
    env.git_state()
    assert calls and all(c[0][:2] == ["git", "--no-optional-locks"] for c in calls)
    assert all(c[1].get("GIT_OPTIONAL_LOCKS") == "0" for c in calls)
    assert any("status" in c[0] for c in calls) and any("describe" in c[0] for c in calls)


def test_s3_workload_leaves_frozen_trees_untouched(bench, folds, frozen_snapshot):
    from gen201 import data as D
    from gen201 import folds as FO
    before = guard.snapshot()
    assert before == frozen_snapshot                       # nothing changed since the session started
    D.cell_meta(bench)
    D.populations(bench, nodmd=False)
    F = FO.build(bench, seeds=(FO.SEEDS[0],), designs=("BP", "BPG"))
    FO.plan_digest(bench, F, designs=("BP", "BPG"))
    FO.gate_c(folds["BP"])
    FO.anchors_reference()
    D.reading_index(bench)
    guard.assert_unchanged(before)
    assert not [k for k in guard.snapshot() if "__pycache__" in k and k not in frozen_snapshot]


def test_snapshot_diff_detects_changes():
    a = {"generations/gen13_separation/x.py": (1, 1)}
    assert guard.diff(a, {"generations/gen13_separation/x.py": (1, 2)})["changed"] == ["generations/gen13_separation/x.py"]
    assert guard.diff(a, {})["removed"] == ["generations/gen13_separation/x.py"]
    assert guard.diff({}, a)["added"] == ["generations/gen13_separation/x.py"]
    with pytest.raises(AssertionError):
        guard.assert_unchanged(a, {"generations/gen13_separation/x.py": (2, 1)})
    guard.assert_unchanged(a, dict(a))


def test_protected_trees_cover_the_frozen_generations():
    names = {p.name for p in guard.PROTECTED}
    for g in ("gen12_2_eu_pred", "gen12_eu_pred", "gen12_eu_pred_2", "gen13_separation", "gen14_direction",
              "gen15_curve", "gen16_anchor", "gen16_leads", "gen16_protocol", "gen19_chem_transfer", "gen20_kshot",
              "tests", "src"):
        assert g in names
    assert bootstrap.G201 not in guard.PROTECTED
    # every directory under generations/ except gen20_1 is frozen
    on_disk = {p.name for p in bootstrap.GENERATIONS.iterdir() if p.is_dir()}
    assert {p.name for p in bootstrap.FROZEN_TREES} == on_disk - {bootstrap.G201.name}


def test_default_snapshot_covers_the_whole_repository_outside_gen20_1(frozen_snapshot):
    """The LT-26 snapshot walks the repository (skipping only .git, .venv, .claude and generations/gen20_1) and
    adds the three gen20.1 documents; the seal step alone may exempt PRE_REGISTRATION.md."""
    snap = frozen_snapshot
    tops = {k.split("/")[0] for k in snap} | {"/".join(k.split("/")[:2]) for k in snap if k.startswith("generations/")}
    for t in ("docs", "runs", "scripts", "tests", "src", "dataset_all_metals", "dataset with 3D structures",
              "README.md", "generations/gen12_eu_pred", "generations/gen12_eu_pred_2", "generations/gen12_2_eu_pred",
              "generations/tests", "generations/README.md", "generations/gen13_separation", "generations/gen20_kshot"):
        assert t in tops, t
    docs = {"generations/gen20_1/PRE_REGISTRATION.md", "generations/gen20_1/IMPLEMENTATION_PLAN.md",
            "generations/gen20_1/README.md"}
    assert docs <= set(snap)
    assert not [k for k in snap if k.startswith("generations/gen20_1/") and k not in docs]
    assert not [k for k in snap if k.split("/")[0] in (".git", ".venv", ".claude")]
    ex = guard.snapshot(exclude=(bootstrap.PREREG,))
    assert "generations/gen20_1/PRE_REGISTRATION.md" not in ex and set(snap) - set(ex) == {
        "generations/gen20_1/PRE_REGISTRATION.md"}
