"""``paths.py`` -- every location gen19 reads or writes, and the two digests it records.

Importing this module has no side effects (unlike ``gen18proc.paths``): output directories are
created by the scripts that write into them, via :func:`ensure_dir`.

Brief section 24 (the gen18 manifest portability problem): a file checked out with CRLF on Windows
must not look like a scientific reproducibility failure.  :func:`digests` therefore records the
binary SHA-256 *and*, for text files, the SHA-256 of the content with CRLF normalised to LF.
Verification (:func:`matches`) accepts a text file when either digest matches.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

G19_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = G19_ROOT.parents[1]
GEN18_ROOT = REPO_ROOT / "generations" / "gen18_process"
GEN13_ROOT = REPO_ROOT / "generations" / "gen13_separation"

ARCHIVE_DIR = REPO_ROOT / "dataset_all_metals"
ARCHIVE_MASTER = ARCHIVE_DIR / "clean" / "master_clean.parquet"
ARCHIVE_MANIFEST = ARCHIVE_DIR / "reports" / "manifest.json"
BUNDLE_DIR = REPO_ROOT / "dataset with 3D structures"
BUNDLE_DATASET = BUNDLE_DIR / "dataset.parquet"
GEN6_PROVENANCE = REPO_ROOT / "runs" / "gen6_provenance" / "provenance_table.parquet"
GEN7_CHEMISTRY_MAP = REPO_ROOT / "runs" / "gen7_architecture" / "cache" / "chemistry_map.parquet"

#: Pinned by ``dataset_all_metals/reports/manifest.json`` (clean/master_clean.parquet).
ARCHIVE_MASTER_SHA256 = "7b32979383c5246f22d36ade6246231a347ea22490f4589a76e189af07b09b01"
#: Pinned by ``gen3_protocol.json`` and ``gen13sep/paths.py``.
BUNDLE_DATASET_SHA256_PREFIX = "fefbefc6"

DATA_AUDIT_DIR = G19_ROOT / "data_audit"
DESCRIPTORS_DIR = G19_ROOT / "descriptors"
FOLDS_DIR = G19_ROOT / "folds"
TABLES_DIR = G19_ROOT / "tables"
FIGURES_DIR = G19_ROOT / "figures"
MANIFESTS_DIR = G19_ROOT / "manifests"
DECISIONS_DIR = G19_ROOT / "decisions"

TEXT_SUFFIXES = frozenset({".csv", ".json", ".md", ".txt", ".py", ".tsv", ".yaml", ".yml", ".sha256"})


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digests(path: Path) -> dict[str, str | int | None]:
    """``{"sha256": binary, "sha256_lf": LF-normalised or None, "bytes": size}``."""
    data = Path(path).read_bytes()
    lf = None
    if Path(path).suffix.lower() in TEXT_SUFFIXES:
        lf = sha256_bytes(data.replace(b"\r\n", b"\n"))
    return {"sha256": sha256_bytes(data), "sha256_lf": lf, "bytes": len(data)}


def matches(path: Path, expected: str) -> bool:
    """True when ``expected`` equals the binary digest or (text files) the LF-normalised one."""
    d = digests(path)
    return expected in {d["sha256"], d["sha256_lf"]}


def rel(path: Path) -> str:
    """Repository-relative POSIX path (never a backslash, never absolute)."""
    return Path(path).resolve().relative_to(REPO_ROOT).as_posix()


def add_gen18_to_path() -> None:
    """Make ``gen18proc`` (and through it ``gen13sep``) importable, read-only use."""
    for p in (GEN18_ROOT, GEN13_ROOT, REPO_ROOT / "src"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
