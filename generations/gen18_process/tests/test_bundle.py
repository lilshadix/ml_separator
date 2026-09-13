"""The frozen bundle has not changed (DESIGN.md section 12.2, row 1).

Run:  .venv/Scripts/python.exe -m pytest generations/gen18_process/tests/test_bundle.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

G18 = Path(__file__).resolve().parents[1]
if str(G18) not in sys.path:
    sys.path.insert(0, str(G18))
from gen18proc import paths  # noqa: E402

FROZEN_SHA256 = "fefbefc6fe993aa9ce9db1a0c338adb9e5f58a8b75bab084cc1df4e024faf5dd"


def test_bundle_sha256_is_frozen():
    """``assert_bundle_unchanged`` returns the frozen digest of BRIEF.md section 2 (exact)."""
    assert paths.BUNDLE_SHA256 == FROZEN_SHA256
    assert paths.assert_bundle_unchanged() == FROZEN_SHA256


def test_frozen_inputs_exist_and_are_read_only_paths():
    assert paths.BUNDLE_PARQUET.exists()
    assert paths.GEN6_PROVENANCE_PARQUET.exists()
    assert not str(paths.BUNDLE_PARQUET).startswith(str(paths.G18_ROOT))
