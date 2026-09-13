"""Path and thread bootstrap for gen16 code.  Import it before any frozen module.

It does two things and nothing else:

* puts ``gen15_curve``, ``gen14_direction`` and ``gen13_separation`` on ``sys.path`` so the
  frozen machinery imports exactly the way its own scripts import it (never re-implemented);
* caps joblib's ``n_jobs=-1`` at two threads through ``LOKY_MAX_CPU_COUNT`` unless the caller
  already set it, because several agents share one 8 GB / 12-thread machine.  Extra-trees with a
  fixed ``random_state`` is invariant to the thread count, so this moves no anchor (verified:
  ``results/anchors/anchors.json``).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

ROOT = Path(__file__).resolve().parents[2]
G16 = ROOT / "gen16_leads"
RESULTS = G16 / "results"
for _p in (ROOT / "gen15_curve", ROOT / "gen14_direction", ROOT / "gen13_separation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
