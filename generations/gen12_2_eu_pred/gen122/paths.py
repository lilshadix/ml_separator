"""Every path Gen12.2 reads or writes.

Gen12 is read-only from here.  ``GEN12_ROOT`` is on ``sys.path`` so that the
frozen cohort builder, splitter, metrics, bootstrap and few-shot code are *the
same objects* Gen12 used — importing them rather than copying them is what makes
"the same frozen design-B folds" a fact instead of an intention.
"""
from __future__ import annotations

import sys
from pathlib import Path

GEN122_ROOT = Path(__file__).resolve().parents[1]          # gen12_2_eu_pred/
REPO_ROOT = GEN122_ROOT.parent.parent
GEN12_ROOT = REPO_ROOT / "generations" / "gen12_eu_pred"
if str(GEN12_ROOT) not in sys.path:
    sys.path.insert(0, str(GEN12_ROOT))

# ---- frozen Gen12 artefacts (read-only) ------------------------------------
GEN12_PREDICTIONS = GEN12_ROOT / "predictions"
GEN12_METRICS = GEN12_ROOT / "metrics"
GEN12_BOOTSTRAP = GEN12_ROOT / "bootstrap"
GEN12_MANIFESTS = GEN12_ROOT / "manifests"
GEN12_COHORT_PARQUET = GEN12_MANIFESTS / "cohort.parquet"
GEN12_FOLD_PLAN_B = GEN12_PREDICTIONS / "B" / "fold_plan.json"

# ---- Gen12.2 outputs -------------------------------------------------------
CONFIG_DIR = GEN122_ROOT / "config"
MANIFEST_DIR = GEN122_ROOT / "manifests"
SPLIT_DIR = GEN122_ROOT / "splits"
FEATURE_DIR = GEN122_ROOT / "features"
PREDICTION_DIR = GEN122_ROOT / "predictions"
METRIC_DIR = GEN122_ROOT / "metrics"
BOOTSTRAP_DIR = GEN122_ROOT / "bootstrap"
FIGURE_DIR = GEN122_ROOT / "figures"
HEADLINE_DIR = GEN122_ROOT / "headline_tables"
ANALYSIS_DIR = GEN122_ROOT / "analysis"

COORDINATION_SMARTS_JSON = CONFIG_DIR / "coordination_smarts.json"

for _d in (CONFIG_DIR, MANIFEST_DIR, SPLIT_DIR, FEATURE_DIR, PREDICTION_DIR, METRIC_DIR,
           BOOTSTRAP_DIR, FIGURE_DIR, HEADLINE_DIR, ANALYSIS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def sha256_of(path) -> str:
    import hashlib
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def blake2b_of_frame(frame, digest_size: int = 8) -> str:
    """Content hash of a feature matrix, for the manifest."""
    import hashlib
    import pandas as pd
    payload = pd.util.hash_pandas_object(
        frame[sorted(frame.columns)], index=True).to_numpy().tobytes()
    return hashlib.blake2b(payload, digest_size=digest_size).hexdigest()
