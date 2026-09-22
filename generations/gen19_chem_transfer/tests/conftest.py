"""pytest configuration for generations/gen19_chem_transfer/tests.

Puts ``generations/gen19_chem_transfer`` on ``sys.path`` so tests can ``import gen19ct``.
Fast run from the repository root:

    .venv/Scripts/python.exe -m pytest generations/gen19_chem_transfer/tests -q -m "not slow"
"""
from __future__ import annotations

import sys
from pathlib import Path

G19 = Path(__file__).resolve().parents[1]
if str(G19) not in sys.path:
    sys.path.insert(0, str(G19))


def pytest_configure(config) -> None:
    config.addinivalue_line("markers", "slow: long-running fit or full-corpus sweep")
    config.addinivalue_line("markers", "real_registry: reads the generation's manifests/digest_registry.json (POST-HOC "
                                       "addendum 2 item 5) instead of the isolated one every other test gets")
