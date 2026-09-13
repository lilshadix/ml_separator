"""Every path gen18 reads or writes, in one place (DESIGN.md section 2).

Frozen inputs are taken **read-only** from ``gen13sep.paths`` (bundle parquet with its frozen
SHA-256, the gen6 provenance table, the gen7 chemistry map); ``generations/gen13_separation`` is
inserted on ``sys.path`` for that import.  gen18 writes only under ``generations/gen18_process/``.
Output directories are created on import, like ``gen13sep.paths`` does.
"""
from __future__ import annotations

import sys
from pathlib import Path

G18_ROOT = Path(__file__).resolve().parents[1]          # generations/gen18_process/
REPO_ROOT = G18_ROOT.parent.parent
GEN13_ROOT = REPO_ROOT / "generations" / "gen13_separation"
if str(GEN13_ROOT) not in sys.path:
    sys.path.insert(0, str(GEN13_ROOT))

from gen13sep.paths import (  # noqa: E402  (sys.path insert above)
    BUNDLE_DIR,
    BUNDLE_PARQUET,
    BUNDLE_SHA256,
    CHEMISTRY_MAP_PARQUET,
    GEN6_PROVENANCE_PARQUET,
    assert_bundle_unchanged,
    sha256_of,
)

__all__ = [
    "BUNDLE_DIR", "BUNDLE_PARQUET", "BUNDLE_SHA256", "CHEMISTRY_MAP_PARQUET",
    "GEN6_PROVENANCE_PARQUET", "assert_bundle_unchanged", "sha256_of",
    "G18_ROOT", "REPO_ROOT", "GEN13_ROOT", "GEN15_DEPLOY", "GEN15_PREDICT_SCRIPT",
    "SYSTEMS_DIR", "RESULTS_DIR", "CASES_DIR", "CONFIG_DIR", "SCRIPTS_DIR", "TESTS_DIR",
    "ADDENDA_DIR", "RESULTS_AUDIT_DIR", "RESULTS_DMODELS_DIR", "RESULTS_EVAL_DIR",
    "RESULTS_LOADING_DIR", "RESULTS_BENCH_DIR", "RESULTS_CASE_PRND_DIR", "RESULTS_RECIPES_DIR",
    "RESULTS_OPTIMIZE_DIR", "RESULTS_SCREEN_DIR", "RESULTS_MANIFEST", "PRE_REGISTRATION_MD",
    "DATA_AUDIT_MD", "GEN18_REPORT_MD",
]

# ---- optional gen15 pre-screen prior (read-only; may be absent, gitignored) ----------------
GEN15_DEPLOY = REPO_ROOT / "generations" / "gen15_curve" / "models" / "deploy_g15.joblib"
GEN15_PREDICT_SCRIPT = REPO_ROOT / "generations" / "gen15_curve" / "scripts" / "g15_predict.py"

# ---- gen18 tree ------------------------------------------------------------------------------
SYSTEMS_DIR = G18_ROOT / "systems"          # <system_id>.json, corpus_records.csv, registry.json
RESULTS_DIR = G18_ROOT / "results"
CASES_DIR = G18_ROOT / "cases"              # prnd_feed.json, prnd_spec.json, todga_prnd_feed.json
CONFIG_DIR = G18_ROOT / "config"            # prices.json, design_spaces.json
SCRIPTS_DIR = G18_ROOT / "scripts"
TESTS_DIR = G18_ROOT / "tests"
ADDENDA_DIR = G18_ROOT / "addenda"

RESULTS_AUDIT_DIR = RESULTS_DIR / "audit"
RESULTS_DMODELS_DIR = RESULTS_DIR / "dmodels"
RESULTS_EVAL_DIR = RESULTS_DIR / "eval"
RESULTS_LOADING_DIR = RESULTS_DIR / "loading"
RESULTS_BENCH_DIR = RESULTS_DIR / "bench"   # the only place wall-clock values may be written
RESULTS_CASE_PRND_DIR = RESULTS_DIR / "case_prnd"
RESULTS_RECIPES_DIR = RESULTS_DIR / "recipes"
RESULTS_OPTIMIZE_DIR = RESULTS_DIR / "optimize"
RESULTS_SCREEN_DIR = RESULTS_DIR / "screen"
RESULTS_MANIFEST = RESULTS_DIR / "MANIFEST.sha256"

PRE_REGISTRATION_MD = G18_ROOT / "PRE_REGISTRATION.md"
DATA_AUDIT_MD = G18_ROOT / "DATA_AUDIT.md"
GEN18_REPORT_MD = G18_ROOT / "GEN18_REPORT.md"

for _d in (SYSTEMS_DIR, RESULTS_DIR, CASES_DIR, CONFIG_DIR, ADDENDA_DIR,
           RESULTS_AUDIT_DIR, RESULTS_DMODELS_DIR, RESULTS_EVAL_DIR, RESULTS_LOADING_DIR,
           RESULTS_BENCH_DIR, RESULTS_CASE_PRND_DIR, RESULTS_RECIPES_DIR, RESULTS_OPTIMIZE_DIR,
           RESULTS_SCREEN_DIR):
    _d.mkdir(parents=True, exist_ok=True)
