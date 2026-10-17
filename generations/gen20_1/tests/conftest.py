"""gen20.1 test configuration (IMPLEMENTATION_PLAN.md §4 "Tests").

Run from the repository root:
    PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -B -m pytest generations/gen20_1/tests -q -p no:cacheprovider

* inserts generations/gen20_1 and generations/gen20_kshot on sys.path and imports ``gen201`` (its bootstrap
  sets ``sys.dont_write_bytecode`` before any frozen import);
* installs the forbidden-writer guard for the whole session (LT-26);
* snapshots the whole repository outside generations/gen20_1 (skipping only .git, .venv and .claude) plus the
  three gen20.1 documents at session start, and fails the session if any of those files was added, removed or
  modified at the end (LT-26; this also catches ``__pycache__``).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
G201 = Path(__file__).resolve().parents[1]
G20 = G201.parent / "gen20_kshot"
for _p in (G20, G201):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pytest  # noqa: E402

from gen201 import bootstrap  # noqa: E402,F401
from gen201 import guard  # noqa: E402

_SNAPSHOT: dict = {}


def pytest_sessionstart(session):
    _SNAPSHOT.clear()
    _SNAPSHOT.update(guard.snapshot())
    guard.install()


def pytest_sessionfinish(session, exitstatus):
    d = guard.diff(_SNAPSHOT, guard.snapshot())
    if any(d.values()):
        print(f"\nLT-26 FAILED: frozen trees changed during the test session: "
              f"{ {k: v[:20] for k, v in d.items() if v} }", file=sys.stderr)
        session.exitstatus = 1


@pytest.fixture(scope="session")
def frozen_snapshot():
    return dict(_SNAPSHOT)


@pytest.fixture(scope="session")
def bench():
    from gen201 import data as D
    return D.load(check=True)


@pytest.fixture(scope="session")
def meta(bench):
    from gen201 import data as D
    return D.cell_meta(bench)


@pytest.fixture(scope="session")
def pops(bench):
    from gen201 import data as D
    return D.populations(bench)


@pytest.fixture(scope="session")
def folds(bench):
    """{design: [Fold]} of the six designs, one gen20k.splits.folds_for call per design."""
    from gen201 import folds as FO
    return FO.build(bench)


@pytest.fixture(scope="session")
def plan(bench, folds):
    from gen201 import folds as FO
    return FO.plan_digest(bench, folds)
