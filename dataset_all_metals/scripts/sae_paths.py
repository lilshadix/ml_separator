"""Shared paths for the multi-metal SAE dataset workspace."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "raw"
INTERMEDIATE_DIR = ROOT / "intermediate"
CLEAN_DIR = ROOT / "clean"
AUDIT_DIR = ROOT / "audit"
REPORTS_DIR = ROOT / "reports"
SCRIPTS_DIR = ROOT / "scripts"

# Mirror of the deliverables directory required by the brief.  The canonical
# artefacts are written here as well so the run is self-describing.
RUN_DIR = ROOT.parent / "runs" / "sae_dataset_audit"


def ensure_dirs() -> None:
    for path in (INTERMEDIATE_DIR, CLEAN_DIR, AUDIT_DIR, REPORTS_DIR, RUN_DIR):
        path.mkdir(parents=True, exist_ok=True)
