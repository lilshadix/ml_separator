"""generations/migrate_local_artifacts.py moves what it should and nothing else.

Each test builds a throwaway git repository in ``tmp_path`` that looks like a clone which
pulled the move into generations/ but still holds local files at the old generation paths,
copies the script into its ``generations/`` directory (the script finds the repository root
from its own location) and runs it in a subprocess.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "migrate_local_artifacts.py"

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs the git command line")

GITIGNORE = """\
__pycache__/
generations/gen13_separation/models/*.joblib
generations/gen14_direction/cache/
generations/gen15_curve/cache/
gen13_separation/models/*.joblib
gen14_direction/cache/
gen15_curve/cache/
"""

# relative path -> bytes, written before the script runs
TRACKED = {
    ".gitignore": GITIGNORE.encode(),
    "generations/gen13_separation/README.md": b"tracked readme\n",
    "gen16_leads/stray.md": b"a file still tracked at its old path\n",
}
LOCAL = {
    # ignored, destination free: moved
    "gen13_separation/models/deploy.joblib": b"fitted model\n",
    # untracked and not ignored: kept
    "gen13_separation/notes/my_note.md": b"my own notes\n",
    # ignored, destination holds different bytes: conflict
    "gen14_direction/cache/bench.pkl": b"old bench\n",
    "generations/gen14_direction/cache/bench.pkl": b"new bench\n",
    # ignored, destination holds identical bytes: old copy deleted
    "gen14_direction/cache/same.pkl": b"same bytes\n",
    "generations/gen14_direction/cache/same.pkl": b"same bytes\n",
    # ignored bytecode whose destination differs: deleted, never a conflict
    "gen13_separation/gen13sep/__pycache__/paths.cpython-313.pyc": b"old bytecode\n",
    "generations/gen13_separation/gen13sep/__pycache__/paths.cpython-313.pyc": b"new bytecode\n",
    # the only content of an old generation directory: moved, directory removed
    "gen15_curve/cache/deep/table.parquet": b"cached table\n",
}


def _env(tmp_path: Path) -> dict:
    env = dict(os.environ)
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env.update(HOME=str(home), GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=str(home / ".gitconfig"),
               PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    return env


def _git(repo: Path, env: dict, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid",
                    "-c", "commit.gpgsign=false", *args], cwd=repo, env=env, check=True,
                   capture_output=True)


def _write(repo: Path, files: dict) -> None:
    for rel, data in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


@pytest.fixture
def clone(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    env = _env(tmp_path)
    _git(repo, env, "init", "-q")
    _write(repo, TRACKED)
    _git(repo, env, "add", "--", *TRACKED)
    _git(repo, env, "commit", "-q", "-m", "fixture")
    _write(repo, LOCAL)
    shutil.copy2(SCRIPT, repo / "generations" / SCRIPT.name)
    return repo, env


def _run(repo: Path, env: dict, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(repo / "generations" / SCRIPT.name), *args],
                          cwd=repo, env=env, capture_output=True, text=True)


def _snapshot(repo: Path) -> dict:
    tree = {}
    for dirpath, dirnames, filenames in os.walk(repo):
        dirnames[:] = [d for d in dirnames if not (dirpath == str(repo) and d == ".git")]
        rel_dir = Path(dirpath).relative_to(repo).as_posix()
        tree[rel_dir + "/"] = None
        for name in filenames:
            path = Path(dirpath) / name
            tree[path.relative_to(repo).as_posix()] = path.read_bytes()
    return tree


def test_dry_run_changes_nothing(clone):
    repo, env = clone
    before = _snapshot(repo)
    result = _run(repo, env)
    assert result.returncode == 1, result.stdout + result.stderr
    assert _snapshot(repo) == before
    out = result.stdout
    assert "DRY RUN" in out
    assert "MOVE       gen13_separation/models/deploy.joblib" in out
    assert "CONFLICT   gen14_direction/cache/bench.pkl" in out
    assert "UNTRACKED  gen13_separation/notes/my_note.md" in out
    assert "TRACKED    gen16_leads/stray.md" in out


def test_apply_moves_ignored_keeps_untracked_and_reports_conflict(clone):
    repo, env = clone
    result = _run(repo, env, "--apply")
    assert result.returncode == 1, result.stdout + result.stderr
    assert re.search(r"conflicts \(left untouched\):\s+1\n", result.stdout), result.stdout

    # the ignored file with a free destination moved, bytes intact
    assert not (repo / "gen13_separation/models/deploy.joblib").exists()
    assert (repo / "generations/gen13_separation/models/deploy.joblib").read_bytes() == b"fitted model\n"
    # the untracked note stays where it was and is not copied
    assert (repo / "gen13_separation/notes/my_note.md").read_bytes() == b"my own notes\n"
    assert not (repo / "generations/gen13_separation/notes").exists()
    # the conflict leaves both copies untouched
    assert (repo / "gen14_direction/cache/bench.pkl").read_bytes() == b"old bench\n"
    assert (repo / "generations/gen14_direction/cache/bench.pkl").read_bytes() == b"new bench\n"
    # an identical duplicate loses its old copy only
    assert not (repo / "gen14_direction/cache/same.pkl").exists()
    assert (repo / "generations/gen14_direction/cache/same.pkl").read_bytes() == b"same bytes\n"
    # regenerated bytecode is deleted, not moved over the new copy
    assert not (repo / "gen13_separation/gen13sep").exists()
    assert ((repo / "generations/gen13_separation/gen13sep/__pycache__/paths.cpython-313.pyc").read_bytes()
            == b"new bytecode\n")
    # tracked files are left alone
    assert (repo / "gen16_leads/stray.md").read_bytes() == b"a file still tracked at its old path\n"
    assert (repo / "generations/gen13_separation/README.md").read_bytes() == b"tracked readme\n"
    assert (repo / "generations/gen15_curve/cache/deep/table.parquet").read_bytes() == b"cached table\n"

    # directories left empty are removed; directories that still hold files are kept
    assert not (repo / "gen13_separation/models").exists()
    assert not (repo / "gen15_curve").exists()
    assert (repo / "gen13_separation/notes").is_dir()
    assert (repo / "gen14_direction/cache").is_dir()


def test_include_untracked_moves_the_note(clone):
    repo, env = clone
    assert _run(repo, env, "--apply").returncode == 1
    (repo / "gen14_direction/cache/bench.pkl").unlink()  # the user resolves the conflict
    result = _run(repo, env, "--apply", "--include-untracked")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (repo / "generations/gen13_separation/notes/my_note.md").read_bytes() == b"my own notes\n"
    assert not (repo / "gen13_separation").exists()
    assert not (repo / "gen14_direction").exists()
    assert (repo / "gen16_leads/stray.md").exists()
