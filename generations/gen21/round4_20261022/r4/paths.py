"""Paths for round 4.  Imports round 3's package (r3) and through it g21 / gen20k read-only; the sealed
round-3 folder is never written.  Everything round 4 writes lives under this directory."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

R4 = Path(__file__).resolve().parents[1]
G21 = R4.parent
R3DIR = G21 / "next_research_20261019"
ROOT = G21.parents[1]
if str(R3DIR) not in sys.path:
    sys.path.insert(0, str(R3DIR))

from r3 import paths as R3P  # noqa: E402  (puts g21, gen20k and gen13-15 on sys.path)

CACHE = R3P.CACHE                       # dev fold cache of gen21 round 1 (read-only reuse)
PRED = R4 / "predictions"
RES = R4 / "results"
MAN = R4 / "manifests"
DEV_SEEDS = R3P.DEV_SEEDS


def sha256_file(p) -> str:
    return R3P.sha256_file(p)


def sha256_text_lf(p) -> str:
    """Digest of a text file with CRLF converted to LF (stable across checkouts)."""
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
