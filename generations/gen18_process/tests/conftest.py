"""pytest configuration for generations/gen18_process/tests.

Puts ``generations/gen18_process`` on ``sys.path`` so every test can ``import gen18proc`` (the
pattern of DESIGN.md section 2), and registers the ``slow`` and ``validation`` markers of
section 12.  Fast run from the repository root:

    .venv/Scripts/python.exe -m pytest generations/gen18_process/tests -q \
        -m "not slow and not validation"
"""
from __future__ import annotations

import sys
from pathlib import Path

G18 = Path(__file__).resolve().parents[1]
if str(G18) not in sys.path:
    sys.path.insert(0, str(G18))


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers", "slow: needs built database files or a long-running solve on disk")
    config.addinivalue_line(
        "markers",
        "validation: a failure is a reported result, not a suite blocker (DESIGN.md section 12.3)")
