"""``scripts/g19_manifest_discovery.py`` -- ``evaluation/discovery/MANIFEST.sha256``.

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
        generations/gen19_chem_transfer/scripts/g19_manifest_discovery.py [--check]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen19ct import paths  # noqa: E402

DISCOVERY_DIR = paths.G19_ROOT / "evaluation" / "discovery"
MANIFEST = DISCOVERY_DIR / "MANIFEST.sha256"
#: never walked: interpreter caches and the manifest itself
SKIP_NAMES = frozenset({MANIFEST.name})
SKIP_PARTS = frozenset({"__pycache__", ".pytest_cache"})
#: the ``.gitignore`` patterns this manifest covers, as a predicate on the discovery-relative path;
#: ``evaluation/discovery/*/*/s*/`` and ``evaluation/discovery/logs/``
EXCLUDED_PATTERNS: tuple[str, ...] = ("<arm>/<design>/s*/**", "logs/**")


def is_excluded(rel: str) -> bool:
    """True when ``rel`` (discovery-relative, POSIX) is matched by :data:`EXCLUDED_PATTERNS`."""
    parts = rel.split("/")
    if parts[0] == "logs":
        return True
    return len(parts) >= 4 and parts[2].startswith("s")


def walk(directory: Path) -> list[Path]:
    out = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name in SKIP_NAMES or SKIP_PARTS & set(path.parts):
            continue
        out.append(path)
    return out


def rel_of(path: Path) -> str:
    return path.resolve().relative_to(DISCOVERY_DIR.resolve()).as_posix()


def excluded_files() -> list[Path]:
    """Every file under ``evaluation/discovery`` the ``.gitignore`` patterns exclude."""
    return [p for p in walk(DISCOVERY_DIR) if is_excluded(rel_of(p))]


def git_ignored() -> set[str] | None:
    """What git itself excludes under ``evaluation/discovery`` (discovery-relative paths), or
    ``None`` when git cannot be asked.  Used only to cross-check :func:`is_excluded`."""
    try:
        proc = subprocess.run(["git", "ls-files", "--others", "--ignored", "--exclude-standard", "--",
                               paths.rel(DISCOVERY_DIR)], cwd=paths.REPO_ROOT, capture_output=True, text=True,
                              timeout=180)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    prefix = paths.rel(DISCOVERY_DIR) + "/"
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
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    problems = cross_check({rel_of(p) for p in files})
    for line in problems:
        print(line)
    print(f"wrote {paths.rel(MANIFEST)} ({len(files)} files, {total / 1e6:.1f} MB)")
    return 1 if problems else 0


def read_manifest() -> dict[str, tuple[str, int]]:
    recorded: dict[str, tuple[str, int]] = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sha, rel, size = line.split("  ", 2)
        recorded[rel] = (sha, int(size))
    return recorded


def check_manifest() -> int:
    if not MANIFEST.exists():
        print(f"missing: {paths.rel(MANIFEST)}; run without --check to write it")
        return 1
    recorded = read_manifest()
    missing, changed, resized = [], [], []
    for rel, (sha, size) in recorded.items():
        path = DISCOVERY_DIR / rel
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
    args = ap.parse_args(argv)
    if not DISCOVERY_DIR.exists():
        print(f"missing: {paths.rel(DISCOVERY_DIR)}")
        return 1
    return check_manifest() if args.check else write_manifest()


if __name__ == "__main__":
    raise SystemExit(main())
