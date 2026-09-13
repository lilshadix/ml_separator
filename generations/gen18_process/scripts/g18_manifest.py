"""Script 13 of DESIGN.md section 11: the results manifest.

DESIGN.md section 11 requires ``results/MANIFEST.sha256`` to list every output, but each script
only writes its own ``manifest.json``; this script aggregates them (integration, 2026-09-13, see
``addenda/INTEGRATION.md``).  It walks ``results/`` and records every file with its size and
SHA-256, sorted by path so the file is byte-identical across runs on the same outputs.

``--check`` recomputes the digests of the listed files that are still present and exits 1 on any
mismatch or on a missing file, so a regenerated artefact cannot silently diverge from what
``GEN18_REPORT.md`` was written from.  New files that are not yet listed are reported but do not
fail the check (run without ``--check`` to re-record them).

Run from the repository root:
    .venv/Scripts/python.exe generations/gen18_process/scripts/g18_manifest.py [--check]

No wall-clock value is written (DESIGN.md section 1.5); ``results/bench/timing.json`` is listed
by digest like any other file.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen18proc import paths  # noqa: E402

MANIFEST = paths.RESULTS_DIR / "MANIFEST.sha256"
#: never listed: interpreter caches and the manifest itself
SKIP_NAMES = {MANIFEST.name}
SKIP_PARTS = {"__pycache__", ".pytest_cache"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def listed_files() -> list[Path]:
    out = []
    for path in sorted(paths.RESULTS_DIR.rglob("*")):
        if not path.is_file() or path.name in SKIP_NAMES:
            continue
        if SKIP_PARTS & set(path.parts):
            continue
        out.append(path)
    return out


def write_manifest() -> int:
    files = listed_files()
    lines = [f"{digest(p)}  {p.relative_to(paths.RESULTS_DIR).as_posix()}  {p.stat().st_size}"
             for p in files]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    total = sum(p.stat().st_size for p in files)
    print(f"wrote {MANIFEST} ({len(files)} files, {total / 1e6:.1f} MB)")
    return 0


def check_manifest() -> int:
    if not MANIFEST.exists():
        print(f"missing: {MANIFEST}; run without --check to write it")
        return 1
    recorded: dict[str, str] = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sha, rel, _size = line.split("  ", 2)
        recorded[rel] = sha
    bad, missing = [], []
    for rel, sha in recorded.items():
        path = paths.RESULTS_DIR / rel
        if not path.exists():
            missing.append(rel)
        elif digest(path) != sha:
            bad.append(rel)
    present = {p.relative_to(paths.RESULTS_DIR).as_posix() for p in listed_files()}
    extra = sorted(present - set(recorded))
    for rel in missing:
        print(f"MISSING  {rel}")
    for rel in bad:
        print(f"CHANGED  {rel}")
    for rel in extra:
        print(f"new (not listed, not a failure)  {rel}")
    ok = not bad and not missing
    print(f"{'OK' if ok else 'FAILED'}: {len(recorded)} listed, {len(bad)} changed, "
          f"{len(missing)} missing, {len(extra)} new")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="verify; exit 1 on a mismatch")
    args = ap.parse_args()
    return check_manifest() if args.check else write_manifest()


if __name__ == "__main__":
    raise SystemExit(main())
