"""The gen12-family relocation into generations/ holds.

* ``generations/verify_relocation.py`` passes: the Gen12 and Gen12.2 manifests still
  verify, with only the substitutions recorded in ``generations/RELOCATION.json``.
* It fails on a copy of the verified files that has been tampered with: a re-stamped
  manifest, a byte flipped in a file the manifest records by size only, a relocation
  record widened to cover a result table, or a deleted listed file whose git lookup fails.
* Each of the three gen12-family ``paths`` modules resolves ``REPO_ROOT`` to the
  repository root (and ``GEN12_ROOT``, where defined, to generations/gen12_eu_pred).

Every check runs in a subprocess so that the ``sys.path`` each module sees is the one its
own scripts set up, and nothing leaks into the pytest process.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATIONS = REPO_ROOT / "generations"

# Import a paths module the way the generation's scripts do (generation directory first on
# sys.path).  Path.mkdir is disabled so the probe creates no output directories.
PROBE = (
    "import importlib, json, pathlib, sys\n"
    "pathlib.Path.mkdir = lambda self, *args, **kwargs: None\n"
    "sys.path.insert(0, sys.argv[1])\n"
    "module = importlib.import_module(sys.argv[2])\n"
    "print(json.dumps({'REPO_ROOT': str(module.REPO_ROOT),\n"
    "                  'GEN12_ROOT': str(getattr(module, 'GEN12_ROOT', ''))}))\n"
)


def _env() -> dict:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def test_verify_relocation_passes():
    result = subprocess.run(
        [sys.executable, str(GENERATIONS / "verify_relocation.py"), "--root", str(REPO_ROOT)],
        capture_output=True, text=True, cwd=REPO_ROOT, env=_env())
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RESULT: PASS" in result.stdout


# Run verify() on a copied tree, with its git lookups pointed at the real repository.  Any
# further arguments are pre-move paths whose `git show` is made to fail.
TAMPER_DRIVER = (
    "import importlib.util, pathlib, sys\n"
    "tree, repo = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])\n"
    "failing = set(sys.argv[3:])\n"
    "spec = importlib.util.spec_from_file_location(\n"
    "    'verify_relocation', repo / 'generations' / 'verify_relocation.py')\n"
    "module = importlib.util.module_from_spec(spec)\n"
    "spec.loader.exec_module(module)\n"
    "blob, has_commit = module._git_blob, module._git_has_commit\n"
    "module._git_blob = lambda root, commit, path: None if path in failing else blob(repo, commit, path)\n"
    "module._git_has_commit = lambda root, commit: has_commit(repo, commit)\n"
    "sys.exit(module.verify(tree))\n"
)
MANIFESTS = ("generations/gen12_eu_pred/manifests/manifest.json",
             "generations/gen12_2_eu_pred/manifests/manifest.json")


def _copy_verified_files(tree: Path) -> None:
    """Copy RELOCATION.json, both manifests and every file they list into ``tree``."""
    files = ["generations/RELOCATION.json", *MANIFESTS]
    for manifest in MANIFESTS:
        generation = Path(manifest).parents[1].as_posix()
        artefacts = json.loads((REPO_ROOT / manifest).read_text(encoding="utf-8"))["artefacts"]
        files.extend(f"{generation}/{key}" for key in artefacts)
    for rel in files:
        if (REPO_ROOT / rel).is_file():
            (tree / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO_ROOT / rel, tree / rel)


def _restamp(generation: Path, key: str) -> None:
    """Write the file's current hash into the manifest, as gen12_manifest.py does."""
    manifest = generation / "manifests" / "manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    content = (generation / key).read_bytes()
    data["artefacts"][key] = {"bytes": len(content),
                              "blake2b_128": hashlib.blake2b(content, digest_size=16).hexdigest()}
    manifest.write_bytes(json.dumps(data, indent=1, default=str).encode("utf-8"))


@pytest.mark.parametrize("tamper, expected", [
    ("none", "RESULT: PASS"),
    ("restamp_unrecorded_edit", "differs from its fdb1e15 blob"),
    ("restamp_relocated_file", "differs from its fdb1e15 blob"),
    ("size_only_byte_flip", "size-only entry and content differs from fdb1e15"),
    ("widened_relocation_record", "is not the RELOCATION_SHA256 pinned"),
    ("deleted_file_git_fails", "new drift without a relocation entry (missing)"),
])
def test_verify_relocation_catches_tampering(tmp_path, tamper, expected):
    tree = tmp_path.resolve()
    _copy_verified_files(tree)
    gen12 = tree / "generations" / "gen12_eu_pred"
    failing = []
    if tamper == "widened_relocation_record":
        # a value change in a frozen result table, recorded as if it were a relocation edit
        key = "headline_tables/t1_zero_shot_leaderboard.csv"
        table = gen12 / key
        table.write_bytes(table.read_bytes().replace(b"B,T1_EXTRATREES,1.0916,", b"B,T2_EXTRATREES,1.0916,"))
        record_path = tree / "generations" / "RELOCATION.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["entries"].append({
            "path": f"generations/gen12_eu_pred/{key}", "path_before": f"gen12_eu_pred/{key}",
            "manifest": MANIFESTS[0], "manifest_key": key,
            "substitutions": [{"after": "B,T2_EXTRATREES,1.0916,", "before": "B,T1_EXTRATREES,1.0916,"}]})
        record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    elif tamper == "deleted_file_git_fails":
        (gen12 / "gen12eu" / "cohort.py").unlink()
        failing = ["gen12_eu_pred/gen12eu/cohort.py"]
    elif tamper == "restamp_unrecorded_edit":
        cohort = gen12 / "gen12eu" / "cohort.py"
        cohort.write_bytes(cohort.read_bytes() + b"UNRECORDED_CHANGE = 1\n")
        _restamp(gen12, "gen12eu/cohort.py")
    elif tamper == "restamp_relocated_file":
        _restamp(gen12, "gen12eu/paths.py")
    elif tamper == "size_only_byte_flip":
        figure = gen12 / "figures" / "fig1_leaderboard_B.png"
        data = bytearray(figure.read_bytes())
        data[-1] ^= 0xFF
        figure.write_bytes(bytes(data))
    result = subprocess.run(
        [sys.executable, "-c", TAMPER_DRIVER, str(tree), str(REPO_ROOT), *failing],
        capture_output=True, text=True, cwd=REPO_ROOT, env=_env())
    assert expected in result.stdout, result.stdout + result.stderr
    assert result.returncode == (0 if tamper == "none" else 1), result.stdout + result.stderr


@pytest.mark.parametrize("generation, module", [
    ("gen12_eu_pred", "gen12eu.paths"),
    ("gen12_2_eu_pred", "gen122.paths"),
    ("gen12_eu_pred_2", "gen12eu2.paths"),
])
def test_paths_module_resolves_the_repository_root(generation, module):
    result = subprocess.run(
        [sys.executable, "-c", PROBE, str(GENERATIONS / generation), module],
        capture_output=True, text=True, cwd=REPO_ROOT, env=_env())
    assert result.returncode == 0, result.stderr
    resolved = json.loads(result.stdout.strip().splitlines()[-1])
    assert Path(resolved["REPO_ROOT"]) == REPO_ROOT
    if resolved["GEN12_ROOT"]:
        assert Path(resolved["GEN12_ROOT"]) == GENERATIONS / "gen12_eu_pred"
