"""Seed-withholding audit (PRE_REGISTRATION.md section 0).

Discovery runs use the five frozen discovery seeds that ``gen13sep.splits.SPLIT_SEEDS`` supplies
by default.  The five confirmation seeds are withheld and may be consumed by exactly one script,
``scripts/g16_confirm.py``.  This audit fails if any other file under ``gen16_leads/``

* passes a ``seeds=`` keyword to one of the frozen fold/bench entry points, or
* contains a confirmation-seed literal.

The check is an AST walk, not a text match, so a keyword split across lines is caught and a
dataclass field or a local variable that merely happens to be called ``seeds`` is not.  The
confirmation seeds are recomputed here from the public rule recorded in the pre-registration;
they are used only for comparison and are never printed.

Run:  .venv/Scripts/python.exe gen16_leads/scripts/g16_audit_seeds.py
Exit 0 = clean, 1 = a violation (listed on stdout).
"""
from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ALLOWED = {"scripts/g16_confirm.py", "scripts/g16_audit_seeds.py"}
DISCOVERY = (104729, 130363, 155921, 196613, 262147)
#: the frozen entry points whose ``seeds`` keyword selects the split seeds
BENCH_CALLS = {"all_folds", "build_folds", "score", "evaluate", "run_arms", "run",
               "run_folds", "few_shot", "fewshot"}


def confirmation_seeds() -> set[int]:
    """The registered rule, recomputed.  Never printed."""
    out: list[int] = []
    i = 1
    while len(out) < 5:
        h = hashlib.sha256(f"gen16-confirmation-seed-{i}".encode()).hexdigest()
        s = int(h[:8], 16) % 900_000 + 100_000
        if s not in DISCOVERY and s not in out:
            out.append(s)
        i += 1
    return set(out)


def _func_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def audit_file(path: Path, rel: str, forbidden: set[int]) -> list[str]:
    src = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:                       # a half-written file from a live agent
        return [f"{rel}: does not parse ({e.msg} at line {e.lineno}) — audit inconclusive"]
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _func_name(node.func) in BENCH_CALLS:
            for kw in node.keywords:
                if kw.arg != "seeds":
                    continue
                v = kw.value
                literal_none = isinstance(v, ast.Constant) and v.value is None
                frozen = isinstance(v, ast.Name) and v.id == "SPLIT_SEEDS"
                if not (literal_none or frozen):
                    bad.append(f"{rel}:{node.lineno}: {_func_name(node.func)}(..., seeds=...) "
                               f"— a discovery script may not choose its own seeds")
        if isinstance(node, ast.Constant) and isinstance(node.value, int) \
                and node.value in forbidden:
            bad.append(f"{rel}:{node.lineno}: a confirmation-seed literal appears in a "
                       f"discovery file")
    return bad


def main() -> int:
    forbidden = confirmation_seeds()
    bad: list[str] = []
    n = 0
    for p in sorted(HERE.rglob("*.py")):
        rel = p.relative_to(HERE).as_posix()
        if rel in ALLOWED or "__pycache__" in rel:
            continue
        n += 1
        bad.extend(audit_file(p, rel, forbidden))
    if bad:
        print(f"SEED AUDIT: {len(bad)} violation(s) over {n} files")
        print("\n".join(bad))
        return 1
    print(f"SEED AUDIT: clean ({n} files; no discovery script selects its own seeds)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
