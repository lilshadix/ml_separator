"""Print the SHA-256 fingerprint of the five-design fold plan as one JSON line.

Run it twice as two separate processes and compare: ``hash()`` is salted per process, so an
in-session comparison proves nothing.  Used by ``g16_anchors.py`` and ``tests/test_anchors.py``.

Usage:  .venv/Scripts/python.exe gen16_leads/scripts/g16_fold_hash.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen16 import bootstrap  # noqa: E402,F401
from gen16.foldhash import fold_plan_digest  # noqa: E402
from gen14.dirbench import load  # noqa: E402

if __name__ == "__main__":
    print(json.dumps(fold_plan_digest(load().frame)), flush=True)
