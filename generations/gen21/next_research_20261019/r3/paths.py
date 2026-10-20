"""Paths for round 3.  Imports gen21's verified package (g21) and through it gen20k read-only;
nothing outside this directory is written."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

R3 = Path(__file__).resolve().parents[1]
G21 = R3.parent
ROOT = G21.parents[1]
if str(G21) not in sys.path:
    sys.path.insert(0, str(G21))

from g21 import paths as G21P  # noqa: E402,F401  (puts gen20k and gen13-15 on sys.path)

CACHE = G21 / "cache"                    # round-1 fold cache (read-only reuse)
PRED = R3 / "predictions"
RES = R3 / "results"
MAN = R3 / "manifests"
DEV_SEEDS = G21P.DEV_SEEDS


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
