"""Every path Gen12.2 reads or writes.  Gen12 is read-only from here."""
from __future__ import annotations

import sys
from pathlib import Path

GEN122_ROOT = Path(__file__).resolve().parents[1]          # gen12_eu_pred_2/
REPO_ROOT = GEN122_ROOT.parent
GEN12_ROOT = REPO_ROOT / "gen12_eu_pred"                    # frozen, read-only
SRC_ROOT = REPO_ROOT / "src"
for _p in (SRC_ROOT, GEN12_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ---- frozen Gen12 inputs (read-only) ---------------------------------------
GEN12_PREDICTIONS = GEN12_ROOT / "predictions"
GEN12_METRICS = GEN12_ROOT / "metrics"
GEN12_MANIFESTS = GEN12_ROOT / "manifests"
GEN12_FOLD_PLAN_B = GEN12_PREDICTIONS / "B" / "fold_plan.json"
GEN12_COHORT_FINGERPRINT = "2a364bb5264e9935"

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

for _d in (CONFIG_DIR, MANIFEST_DIR, SPLIT_DIR, FEATURE_DIR, PREDICTION_DIR, METRIC_DIR,
           BOOTSTRAP_DIR, FIGURE_DIR, HEADLINE_DIR, ANALYSIS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def sha256_of(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def blake2b_of_frame(frame) -> str:
    """Order-independent content hash of a DataFrame (sorted columns, index dropped)."""
    import hashlib
    import pandas as pd
    payload = pd.util.hash_pandas_object(frame[sorted(frame.columns)], index=False).to_numpy().tobytes()
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


def assert_gen12_untouched() -> None:
    """Gen12 artefacts must not change while Gen12.2 runs.  Checked against Gen12's own manifest."""
    import json
    manifest = json.loads((GEN12_MANIFESTS / "manifest.json").read_text())
    import hashlib
    drift = []
    for rel, meta in manifest["artefacts"].items():
        path = GEN12_ROOT / rel
        if not path.exists():
            drift.append((rel, "missing"))
            continue
        digest = hashlib.blake2b(path.read_bytes(), digest_size=16).hexdigest()
        if digest != meta["blake2b_128"]:
            drift.append((rel, "changed"))
    # The manifest hashes itself last, and the report/README may legitimately have been
    # touched by Gen12's own finalisation after the manifest was written; anything else is fatal.
    tolerated = {"manifests/manifest.json", "DECISION_REPORT.md", "README.md"}
    fatal = [d for d in drift if d[0] not in tolerated]
    if fatal:
        raise RuntimeError(f"Gen12 artefacts drifted: {fatal[:5]}")
