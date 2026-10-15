"""Path and thread bootstrap for gen20.  Import it before any frozen module.

Puts ``gen15_curve``, ``gen14_direction`` and ``gen13_separation`` on ``sys.path`` so the frozen
machinery is imported exactly as its own scripts import it (never re-implemented), and caps
joblib/torch threads so a run fits the compute contract (2 workers, 4 threads, no nested
parallelism).  Nothing here touches a frozen directory.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("GEN20_THREADS", "4"))

ROOT = Path(__file__).resolve().parents[3]
G20 = ROOT / "generations" / "gen20_kshot"
RESULTS = G20 / "results"
PREDICTIONS = G20 / "predictions"
MANIFESTS = G20 / "manifests"
for _p in (ROOT / "generations" / "gen15_curve", ROOT / "generations" / "gen14_direction",
           ROOT / "generations" / "gen13_separation"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def torch_threads(n: int | None = None) -> None:
    """Cap torch's intra-op threads (``GEN20_THREADS`` overrides; default 4)."""
    try:
        import torch
        torch.set_num_threads(int(os.environ.get("GEN20_THREADS", n or 4)))
    except ImportError:  # pragma: no cover
        pass
