"""Paths for gen21.  gen20's package is imported read-only (its bootstrap puts gen13-gen15 on
``sys.path``); nothing under any frozen generation is written."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

G21 = Path(__file__).resolve().parents[1]
ROOT = G21.parents[1]
G20 = ROOT / "generations" / "gen20_kshot"
if str(G20) not in sys.path:
    sys.path.insert(0, str(G20))

from gen20k import bootstrap  # noqa: E402,F401  (path + thread bootstrap of the frozen stack)

CACHE = G21 / "cache"
RESULTS = G21 / "results"
PREDICTIONS = G21 / "predictions"
MANIFESTS = G21 / "manifests"

DEV_SEEDS: tuple[int, ...] = (104729, 130363, 155921, 196613, 262147)   # gen20's discovery seeds


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()
