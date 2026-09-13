"""Script 3 of DESIGN.md section 11: seal (or check) ``PRE_REGISTRATION.md``.

Sealing writes the SHA-256 of everything above the footer line
``Sealed SHA-256 of everything above this line: `<hex>``` into that footer and into
``results/eval/prereg_sha256.txt``.  It refuses when ``DATA_AUDIT.md`` does not exist (the
audit freezes the cohort first) and when the file is already sealed with a different digest
(after sealing, a change is a dated addendum appended *below* the footer, which the digest
does not cover).  ``--check`` recomputes the digest and exits non-zero on mismatch or when the
file is unsealed; the fit scripts call it.

Usage (from the repository root):
    .venv/Scripts/python.exe generations/gen18_process/scripts/g18_seal_prereg.py [--check]

Newlines are normalised to LF before hashing so the digest is the same on Windows and Linux.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen18proc import paths  # noqa: E402

FOOTER_PREFIX = "Sealed SHA-256 of everything above this line:"
PLACEHOLDER = "<written by scripts/g18_seal_prereg.py after DATA_AUDIT.md>"
SHA_FILE = paths.RESULTS_EVAL_DIR / "prereg_sha256.txt"
_HEX = re.compile(r"`([0-9a-f]{64})`")


def split_footer(text: str) -> tuple[str, str, str]:
    """``(above, footer_line, below)``; raises ``ValueError`` when no footer line exists."""
    lines = text.split("\n")
    idx = [i for i, ln in enumerate(lines) if ln.startswith(FOOTER_PREFIX)]
    if not idx:
        raise ValueError(f"no footer line starting with {FOOTER_PREFIX!r}")
    i = idx[-1]
    return "\n".join(lines[:i]), lines[i], "\n".join(lines[i + 1:])


def digest_of(above: str) -> str:
    return hashlib.sha256((above + "\n").encode("utf-8")).hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def footer_digest(footer: str) -> str | None:
    m = _HEX.search(footer)
    return m.group(1) if m else None


def check(path: Path = paths.PRE_REGISTRATION_MD) -> tuple[bool, str]:
    text = read_text(path)
    above, footer, _ = split_footer(text)
    recorded = footer_digest(footer)
    if recorded is None:
        return False, "unsealed: the footer still holds the placeholder"
    actual = digest_of(above)
    if recorded != actual:
        return False, f"mismatch: footer {recorded[:12]}... != recomputed {actual[:12]}..."
    if not SHA_FILE.exists():
        return False, f"mismatch: {SHA_FILE} missing"
    on_disk = SHA_FILE.read_text(encoding="utf-8").split()[0]
    if on_disk != actual:
        return False, f"mismatch: {SHA_FILE.name} {on_disk[:12]}... != recomputed {actual[:12]}..."
    return True, f"sealed: {actual}"


def seal(path: Path = paths.PRE_REGISTRATION_MD) -> tuple[int, str]:
    if not paths.DATA_AUDIT_MD.exists():
        return 2, f"refused: {paths.DATA_AUDIT_MD} does not exist; run g18_audit.py first"
    text = read_text(path)
    above, footer, below = split_footer(text)
    actual = digest_of(above)
    recorded = footer_digest(footer)
    if recorded is not None:
        if recorded == actual:
            SHA_FILE.write_text(actual + "\n", encoding="utf-8")
            return 0, f"already sealed: {actual}"
        return 2, ("refused: the file is sealed with a different digest; the text above the "
                   "footer was edited after sealing. Restore it and append a dated addendum "
                   "below the footer instead.")
    new_footer = f"{FOOTER_PREFIX} `{actual}`"
    new_text = above + "\n" + new_footer + ("\n" + below if below else "\n")
    if not new_text.endswith("\n"):
        new_text += "\n"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(new_text)
    SHA_FILE.parent.mkdir(parents=True, exist_ok=True)
    SHA_FILE.write_text(actual + "\n", encoding="utf-8")
    return 0, f"sealed: {actual} (written to the footer and {SHA_FILE})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="verify the seal; exit 1 on mismatch")
    args = ap.parse_args()
    if args.check:
        ok, msg = check()
        print(msg)
        return 0 if ok else 1
    code, msg = seal()
    print(msg)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
