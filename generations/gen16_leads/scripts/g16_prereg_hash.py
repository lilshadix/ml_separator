"""Seal or verify PRE_REGISTRATION.md.

The file ends with a footer of exactly two lines:

    ---
    SHA-256 of every byte above the preceding '---' line: <hex>

``--seal`` computes the digest of the current content (which must not already carry a footer)
and appends the footer.  Without a flag the script verifies the footer and exits 1 on mismatch.
Every addendum added later goes ABOVE the footer and the file is re-sealed with ``--reseal``,
which records the previous digest in the addendum so the history of seals is visible.

Run:  .venv/Scripts/python.exe generations/gen16_leads/scripts/g16_prereg_hash.py [--seal | --reseal]
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "PRE_REGISTRATION.md"
MARK = "SHA-256 of every byte above the preceding '---' line: "


def split(text: str) -> tuple[str, str | None]:
    lines = text.split("\n")
    # strip a trailing empty line produced by the final newline
    tail = lines[-3:] if lines and lines[-1] == "" else lines[-2:]
    if len(tail) >= 2 and tail[0] == "---" and tail[1].startswith(MARK):
        n_footer = 3 if lines[-1] == "" else 2
        body = "\n".join(lines[:-n_footer]) + "\n"
        return body, tail[1][len(MARK):].strip()
    return text, None


def digest(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def main() -> int:
    text = PATH.read_text(encoding="utf-8")
    body, recorded = split(text)
    if "--seal" in sys.argv or "--reseal" in sys.argv:
        if recorded is not None and "--reseal" not in sys.argv:
            print("already sealed; use --reseal after adding an addendum above the footer")
            return 1
        d = digest(body)
        PATH.write_text(body + "---\n" + MARK + d + "\n", encoding="utf-8")
        print("sealed:", d)
        return 0
    if recorded is None:
        print("PRE_REGISTRATION.md carries no seal footer")
        return 1
    d = digest(body)
    ok = d == recorded
    print(("OK " if ok else "MISMATCH ") + f"recorded {recorded} computed {d}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
