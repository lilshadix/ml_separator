"""Every path Gen12 reads or writes, in one place."""
from __future__ import annotations

import sys
from pathlib import Path

GEN12_ROOT = Path(__file__).resolve().parents[1]          # gen12_eu_pred/
REPO_ROOT = GEN12_ROOT.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# ---- frozen inputs (read-only) --------------------------------------------
BUNDLE_DIR = REPO_ROOT / "dataset with 3D structures"
BUNDLE_PARQUET = BUNDLE_DIR / "dataset.parquet"
BUNDLE_SHA256 = "fefbefc6fe993aa9ce9db1a0c338adb9e5f58a8b75bab084cc1df4e024faf5dd"
LIG2D_PARQUET = BUNDLE_DIR / "ligand_2d_descriptors.parquet"
CHEMISTRY_MAP_PARQUET = REPO_ROOT / "runs" / "gen7_architecture" / "cache" / "chemistry_map.parquet"
GEN7_COHORT_PARQUET = REPO_ROOT / "runs" / "gen7_architecture" / "cache" / "cohort.parquet"
ARCHIVE_MASTER_PARQUET = REPO_ROOT / "dataset_all_metals" / "clean" / "master_clean.parquet"
ARCHIVE_MANIFEST = REPO_ROOT / "dataset_all_metals" / "reports" / "manifest.json"
GEN6_PROVENANCE_PARQUET = REPO_ROOT / "runs" / "gen6_provenance" / "provenance_table.parquet"

# ---- Gen12 outputs ---------------------------------------------------------
CONFIG_DIR = GEN12_ROOT / "config"
MANIFEST_DIR = GEN12_ROOT / "manifests"
SPLIT_DIR = GEN12_ROOT / "splits"
MODEL_DIR = GEN12_ROOT / "models"
PREDICTION_DIR = GEN12_ROOT / "predictions"
METRIC_DIR = GEN12_ROOT / "metrics"
BOOTSTRAP_DIR = GEN12_ROOT / "bootstrap"
FIGURE_DIR = GEN12_ROOT / "figures"
HEADLINE_DIR = GEN12_ROOT / "headline_tables"

for _d in (CONFIG_DIR, MANIFEST_DIR, SPLIT_DIR, MODEL_DIR, PREDICTION_DIR, METRIC_DIR,
           BOOTSTRAP_DIR, FIGURE_DIR, HEADLINE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def sha256_of(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
