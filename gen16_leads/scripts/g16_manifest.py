"""Manifest the large artefacts that are deliberately not committed.

`START_HERE.md` §8: "prediction dumps and large parquet under `runs/` are excluded by design and
their SHA-256 goes in a manifest instead.  Follow that convention for anything large you
generate."  This script applies that convention to `gen16_leads/results/`: every file at or above
``LIMIT`` bytes, plus scratch pickles and the warning-spam run logs, is recorded in
``results/MANIFEST.sha256`` with its size and digest and excluded by ``gen16_leads/.gitignore``.

The files stay on disk; only their bytes leave the repository.  Anything a report cites by number
lives in a small CSV or JSON that is committed.

Run:  .venv/Scripts/python.exe gen16_leads/scripts/g16_manifest.py [--check]
``--check`` recomputes the digests of the listed files that are still present and exits 1 on any
mismatch, so a regenerated artefact cannot silently diverge from what the report was written from.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
RESULTS = HERE / "results"
MANIFEST = RESULTS / "MANIFEST.sha256"
LIMIT = 1_000_000
#: always excluded regardless of size: interpreter scratch and logs that are mostly warning spam
SCRATCH_SUFFIX = (".pkl",)
SCRATCH_PREFIX = ("_",)
LOG_LIMIT = 200_000


def digest(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


SKIP_TREES = ("L1/reference_species/out",)


def excluded(p: Path) -> bool:
    if p.name == MANIFEST.name:
        return False
    rel = p.relative_to(RESULTS).as_posix()
    if any(rel.startswith(t) for t in SKIP_TREES):
        return False        # git-ignored wholesale; digested in sha256_out.txt instead
    size = p.stat().st_size
    if size >= LIMIT:
        return True
    if p.suffix in SCRATCH_SUFFIX or p.name.startswith(SCRATCH_PREFIX):
        return True
    if p.suffix == ".log" and size >= LOG_LIMIT:
        return True
    return False


def scan() -> list[tuple[str, int, str]]:
    out = []
    for p in sorted(RESULTS.rglob("*")):
        if p.is_file() and excluded(p):
            out.append((p.relative_to(HERE).as_posix(), p.stat().st_size, digest(p)))
    return out


def main() -> int:
    rows = scan()
    if "--check" in sys.argv:
        if not MANIFEST.exists():
            print("no manifest to check")
            return 1
        want = {}
        for line in MANIFEST.read_text(encoding="utf-8").splitlines():
            if line.startswith("#") or not line.strip():
                continue
            d, size, path = line.split("  ", 2)
            want[path] = (d, int(size))
        bad = 0
        for path, (d, _size) in want.items():
            p = HERE / path
            if not p.exists():
                continue
            got = digest(p)
            if got != d:
                print(f"MISMATCH {path}\n  manifest {d}\n  on disk  {got}")
                bad += 1
        print(f"manifest check: {len(want)} entries, {bad} mismatches "
              f"({sum(1 for k in want if not (HERE / k).exists())} absent)")
        return 1 if bad else 0
    lines = ["# gen16 large artefacts: not committed, digests recorded (START_HERE.md section 8).",
             "# Regenerate with the script named in the owning lead's report; verify with",
             "#   .venv/Scripts/python.exe gen16_leads/scripts/g16_manifest.py --check",
             f"# {len(rows)} files, {sum(r[1] for r in rows) / 1e6:.1f} MB total.",
             "# sha256  bytes  path"]
    lines += [f"{d}  {size}  {path}" for path, size, d in rows]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {MANIFEST.relative_to(HERE)}: {len(rows)} files, "
          f"{sum(r[1] for r in rows) / 1e6:.1f} MB excluded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
