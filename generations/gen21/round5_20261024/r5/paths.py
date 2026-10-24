"""Paths for round 5.  Imports round 4 (r4) and through it round 3 (r3), g21, gen20k and gen13 read-only; the sealed
round-3 and round-4 folders and the design folder are never written.  Everything round 5 writes lives under this
directory (CLAUDE_CODE_PROMPT.md section 1.2, 2.4).

Importing this module also redirects r3's inner cache to <round5>/cache/_inner, so no runner path writes into the
round-3 or round-4 tree.  Call redirect_inner_cache() again after any import of r4.registry (which re-points it)."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

R5 = Path(__file__).resolve().parents[1]
G21 = R5.parent
ROOT = G21.parents[1]
R4DIR = G21 / "round4_20261022"
R3DIR = G21 / "next_research_20261019"
DESIGN = G21 / "round5_design_20261023"
if str(R4DIR) not in sys.path:
    sys.path.insert(0, str(R4DIR))

from r4 import paths as R4P  # noqa: E402  (puts r3, g21, gen20k and gen13-15 on sys.path)

DEV_SEEDS = R4P.DEV_SEEDS
DESIGNS = ("BPG", "BP")
R4PRED = R4DIR / "predictions"
R4RES = R4DIR / "results"
CACHE_R4 = R4P.CACHE                    # gen21 dev fold caches (BP, BPG); read-only
CACHE = R5 / "cache"                    # round-5 caches (A, B, BR, BQ fold records; external objects)
RES = R5 / "results"
MAN = R5 / "manifests"
C5 = "pub_6a0ef8528c"                   # C5-BTBP publication (section 5.2)
C5_CELL = "c950dc2982e96b83"


def redirect_inner_cache() -> None:
    from r3 import configs as CF
    CF.INNER_CACHE_DIR = CACHE / "_inner"


redirect_inner_cache()


def sha256_file(p) -> str:
    return R4P.sha256_file(p)


def sha256_text_lf(p) -> str:
    """Digest of a text file with CRLF converted to LF (stable across checkouts)."""
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def write_text_lf(p, text: str) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_bytes(text.replace("\r\n", "\n").encode("utf-8"))


def seed_for(key: str) -> int:
    """RNG seed scheme of REGISTRY.csv: 20261023 + crc32(key) (P2's scheme)."""
    import zlib
    return 20261023 + zlib.crc32(key.encode("utf-8"))
