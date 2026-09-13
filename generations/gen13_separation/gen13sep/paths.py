"""Every path Gen13 reads or writes, in one place.

Gen13 reads the frozen bundle, the frozen gen6 chemistry map, the gen6 provenance
table, the bundle's 2D ligand descriptors and Gen12.2's frozen coordination block
**read-only**.  It writes only under ``gen13_separation/``.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

GEN13_ROOT = Path(__file__).resolve().parents[1]          # gen13_separation/
REPO_ROOT = GEN13_ROOT.parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# ---- frozen inputs (read-only) --------------------------------------------
BUNDLE_DIR = REPO_ROOT / "dataset with 3D structures"
BUNDLE_PARQUET = BUNDLE_DIR / "dataset.parquet"
BUNDLE_SHA256 = "fefbefc6fe993aa9ce9db1a0c338adb9e5f58a8b75bab084cc1df4e024faf5dd"
LIG2D_PARQUET = BUNDLE_DIR / "ligand_2d_descriptors.parquet"
CHEMISTRY_MAP_PARQUET = REPO_ROOT / "runs" / "gen7_architecture" / "cache" / "chemistry_map.parquet"
GEN6_PROVENANCE_PARQUET = REPO_ROOT / "runs" / "gen6_provenance" / "provenance_table.parquet"
GEN122_COORDINATION_PARQUET = (REPO_ROOT / "generations" / "gen12_2_eu_pred" / "features"
                               / "coordination_descriptors.parquet")
GEN122_COORDINATION_POSTHOC_PARQUET = (REPO_ROOT / "generations" / "gen12_2_eu_pred" / "features"
                                       / "coordination_descriptors_posthoc.parquet")

# ---- Gen13 outputs ---------------------------------------------------------
CONFIG_DIR = GEN13_ROOT / "config"
MANIFEST_DIR = GEN13_ROOT / "manifests"
PREDICTION_DIR = GEN13_ROOT / "predictions"
METRIC_DIR = GEN13_ROOT / "metrics"
BOOTSTRAP_DIR = GEN13_ROOT / "bootstrap"
HEADLINE_DIR = GEN13_ROOT / "headline_tables"
ANALYSIS_DIR = GEN13_ROOT / "analysis"
FIGURE_DIR = GEN13_ROOT / "figures"

for _d in (CONFIG_DIR, MANIFEST_DIR, PREDICTION_DIR, METRIC_DIR, BOOTSTRAP_DIR,
           HEADLINE_DIR, ANALYSIS_DIR, FIGURE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_bundle_unchanged() -> str:
    digest = sha256_of(BUNDLE_PARQUET)
    if digest != BUNDLE_SHA256:
        raise RuntimeError(f"bundle sha256 {digest} != frozen {BUNDLE_SHA256}")
    return digest
