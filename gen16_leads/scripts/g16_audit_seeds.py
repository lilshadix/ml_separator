"""Seed-withholding audit (PRE_REGISTRATION.md section 0).

Discovery runs use the five frozen discovery seeds that ``gen13sep.splits.SPLIT_SEEDS`` supplies by
default.  The five confirmation seeds are withheld by the orchestrator and may be consumed by
exactly one script, ``scripts/g16_confirm.py``.  This audit greps every Python file under
``gen16_leads/`` for a ``seeds=`` / ``seeds =`` keyword handed to the bench (``all_folds``,
``build_folds``, ``score``, ``evaluate``, ``run_arms``, ``run``) or for the literal confirmation
seed values, and fails on any file other than the confirmation script.

Run:  .venv/Scripts/python.exe gen16_leads/scripts/g16_audit_seeds.py
Exit status 0 = clean, 1 = a violation was found (listed on stdout).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ALLOWED = {"scripts/g16_confirm.py", "scripts/g16_audit_seeds.py"}
DISCOVERY = {104729, 130363, 155921, 196613, 262147}
KW = re.compile(r"\bseeds\s*=\s*(?!None\b)(?!SPLIT_SEEDS\b)[\w\[\(]")   # a real argument, not prose
CALLS = ("all_folds", "build_folds", "score(", "evaluate(", "run_arms(", "run(", "seeds=")


def main() -> int:
    bad: list[str] = []
    for p in sorted(HERE.rglob("*.py")):
        rel = p.relative_to(HERE).as_posix()
        if rel in ALLOWED or "__pycache__" in rel:
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), 1):
            s = line.strip()
            if s.startswith("#"):
                continue
            if KW.search(line) and any(c in line for c in CALLS):
                bad.append(f"{rel}:{i}: {s}")
            for tok in re.findall(r"\b\d{6}\b", line):
                v = int(tok)
                if v not in DISCOVERY and 100_000 <= v < 1_000_000 and "seed" in line.lower():
                    bad.append(f"{rel}:{i}: six-digit seed literal {v}: {s}")
    if bad:
        print("SEED AUDIT: violations")
        print("\n".join(bad))
        return 1
    print("SEED AUDIT: clean (no discovery script passes its own seeds to the bench)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
