"""``scripts/g19_manifest_discovery.py`` -- ``evaluation/<run>/MANIFEST.sha256`` (``--root``).

Two runs are manifested by this one script: ``--root discovery`` (the default, the completed discovery
run) and ``--root confirmation`` (the once-only confirmation run's raw fold records).  Only the
excluded set differs, and :data:`TARGETS` states it per run; everything else -- the digests, the
gitignore cross-check, ``--check`` -- is shared.

The raw fold records of the completed discovery run (3,900 files, 224.0 MB) and the runner's stdout
log (16.3 MB) are excluded from version control by ``generations/gen19_chem_transfer/.gitignore``,
the same convention as ``generations/gen18_process/.gitignore`` +
``generations/gen18_process/results/MANIFEST.sha256``.  This script records a SHA-256 and a byte
count for every excluded file, one ``sha256  path  bytes`` line per file, sorted by path, LF, so a
regenerated run can be compared file by file with the one the decision files were written from.

Digests come from :func:`gen19ct.paths.digests` and verification from :func:`gen19ct.paths.matches`,
so the recorded value is the binary SHA-256 while a text record that a checkout turned into CRLF
still verifies (brief section 24, the gen18 manifest portability problem).

``--check`` recomputes the digest of every listed file and exits 1 on a mismatch, on a missing file,
or when a listed file is *not* in fact excluded by ``.gitignore`` (the manifest would then be
claiming cover for a file git is about to commit).  An excluded file that is not yet listed is
reported and is not a failure -- run without ``--check`` to re-record it.

This script **verifies no discovery record**: it hashes bytes on disk and reads no record field.
Record verification is a different question and is answered only against
``manifests/digest_registry.json`` (POST-HOC addendum 2 item 5), never against live code.

Run from the repository root:
    PYTHONIOENCODING=utf-8 PYTHONPATH=generations/gen19_chem_transfer .venv/Scripts/python.exe \
        generations/gen19_chem_transfer/scripts/g19_manifest_discovery.py [--check] [--root confirmation]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen19ct import paths  # noqa: E402

MANIFEST_NAME = "MANIFEST.sha256"
#: never walked: interpreter caches and the manifest itself
SKIP_NAMES = frozenset({MANIFEST_NAME})
SKIP_PARTS = frozenset({"__pycache__", ".pytest_cache"})


def _discovery_excluded(rel: str) -> bool:
    """``evaluation/discovery/*/*/s*/`` and ``evaluation/discovery/logs/``."""
    parts = rel.split("/")
    if parts[0] == "logs":
        return True
    return len(parts) >= 4 and parts[2].startswith("s")


def _confirmation_excluded(rel: str) -> bool:
    """``evaluation/confirmation/records/`` only.

    The confirmation run's fold designs (``folds/``, 4.7 MB) and its stdout logs (26 KB) ARE committed:
    they are small, and they are the evidence of which folds the withheld seeds produced.  Neither leaks a
    seed -- a design is a fold assignment and its hash, and recovering a seed from one would mean
    rebuilding the design for every candidate seed (10-25 s each) over the whole seed space.
    """
    return rel.split("/")[0] == "records"


@dataclass(frozen=True)
class Target:
    """One manifested directory: where it is, and which of its files ``.gitignore`` excludes."""

    name: str
    dir: Path
    is_excluded: Callable[[str], bool]
    patterns: tuple[str, ...]

    @property
    def manifest(self) -> Path:
        return self.dir / MANIFEST_NAME


TARGETS: dict[str, Target] = {
    "discovery": Target("discovery", paths.G19_ROOT / "evaluation" / "discovery", _discovery_excluded,
                        ("<arm>/<design>/s*/**", "logs/**")),
    "confirmation": Target("confirmation", paths.G19_ROOT / "evaluation" / "confirmation", _confirmation_excluded,
                           ("records/**",)),
}
#: the directory this invocation manifests; ``--root`` selects it, and every function below reads it
TARGET: Target = TARGETS["discovery"]


def walk(directory: Path) -> list[Path]:
    out = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name in SKIP_NAMES or SKIP_PARTS & set(path.parts):
            continue
        out.append(path)
    return out


def rel_of(path: Path) -> str:
    return path.resolve().relative_to(TARGET.dir.resolve()).as_posix()


def excluded_files() -> list[Path]:
    """Every file under the target directory the ``.gitignore`` patterns exclude."""
    return [p for p in walk(TARGET.dir) if TARGET.is_excluded(rel_of(p))]


def git_ignored() -> set[str] | None:
    """What git itself excludes under the target directory (target-relative paths), or
    ``None`` when git cannot be asked.  Used only to cross-check :func:`is_excluded`."""
    try:
        proc = subprocess.run(["git", "ls-files", "--others", "--ignored", "--exclude-standard", "--",
                               paths.rel(TARGET.dir)], cwd=paths.REPO_ROOT, capture_output=True, text=True,
                              timeout=180)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    prefix = paths.rel(TARGET.dir) + "/"
    out = set()
    for line in proc.stdout.splitlines():
        line = line.strip().strip('"')
        if line.startswith(prefix):
            rel = line[len(prefix):]
            if rel not in SKIP_NAMES and not (SKIP_PARTS & set(rel.split("/"))):
                out.add(rel)
    return out


def cross_check(rels: set[str]) -> list[str]:
    """Report every disagreement between :func:`is_excluded` and git; empty when git agrees or is
    unavailable."""
    ignored = git_ignored()
    if ignored is None:
        print("note: git could not be asked which files it ignores; the patterns of this module govern")
        return []
    problems = [f"LISTED BUT NOT IGNORED BY GIT  {r}" for r in sorted(rels - ignored)]
    problems += [f"IGNORED BY GIT BUT NOT LISTED  {r}" for r in sorted(ignored - rels)]
    return problems


def write_manifest() -> int:
    files = excluded_files()
    lines, total = [], 0
    for p in files:
        d = paths.digests(p)
        total += int(d["bytes"])
        lines.append(f"{d['sha256']}  {rel_of(p)}  {d['bytes']}")
    TARGET.manifest.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    problems = cross_check({rel_of(p) for p in files})
    for line in problems:
        print(line)
    print(f"wrote {paths.rel(TARGET.manifest)} ({len(files)} files, {total / 1e6:.1f} MB)")
    return 1 if problems else 0


def read_manifest() -> dict[str, tuple[str, int]]:
    recorded: dict[str, tuple[str, int]] = {}
    for line in TARGET.manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sha, rel, size = line.split("  ", 2)
        recorded[rel] = (sha, int(size))
    return recorded


def check_manifest() -> int:
    if not TARGET.manifest.exists():
        print(f"missing: {paths.rel(TARGET.manifest)}; run without --check to write it")
        return 1
    recorded = read_manifest()
    missing, changed, resized = [], [], []
    for rel, (sha, size) in recorded.items():
        path = TARGET.dir / rel
        if not path.exists():
            missing.append(rel)
            continue
        if not paths.matches(path, sha):
            changed.append(rel)
        elif path.stat().st_size != size:
            resized.append(rel)
    present = {rel_of(p) for p in excluded_files()}
    extra = sorted(present - set(recorded))
    for rel in missing:
        print(f"MISSING  {rel}")
    for rel in changed:
        print(f"CHANGED  {rel}")
    for rel in resized:
        print(f"SIZE     {rel} (digest matches; recorded {recorded[rel][1]} bytes)")
    for rel in extra:
        print(f"new (not listed, not a failure)  {rel}")
    problems = cross_check(set(recorded))
    for line in problems:
        print(line)
    total = sum(size for _, size in recorded.values())
    ok = not (missing or changed or resized or problems)
    print(f"{'OK' if ok else 'FAILED'}: {len(recorded)} listed ({total / 1e6:.1f} MB), {len(changed)} changed, "
          f"{len(missing)} missing, {len(resized)} resized, {len(extra)} new, {len(problems)} gitignore disagreement(s)")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="verify; exit 1 on a mismatch")
    ap.add_argument("--root", choices=sorted(TARGETS), default="discovery",
                    help="which run's excluded files to manifest (default: discovery)")
    args = ap.parse_args(argv)
    global TARGET                                          # noqa: PLW0603 -- the CLI selects the target
    TARGET = TARGETS[args.root]
    if not TARGET.dir.exists():
        print(f"missing: {paths.rel(TARGET.dir)}")
        return 1
    return check_manifest() if args.check else write_manifest()


if __name__ == "__main__":
    raise SystemExit(main())
