#!/usr/bin/env bash
# Re-derive every table, figure, manifest and verification from the predictions on disk.
# Safe to re-run: it recomputes analysis, never models.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=".venv/bin/python"
export PYTHONPATH=gen12_eu_pred

$PY gen12_eu_pred/scripts/gen12_analysis.py --design B
$PY gen12_eu_pred/scripts/gen12_analysis.py --design A
$PY gen12_eu_pred/scripts/gen12_hypotheses.py
$PY gen12_eu_pred/scripts/gen12_frontier.py
$PY gen12_eu_pred/scripts/gen12_diagnostics.py
$PY gen12_eu_pred/scripts/gen12_figures.py --design B
$PY gen12_eu_pred/scripts/gen12_headline_tables.py
$PY gen12_eu_pred/scripts/gen12_manifest.py
$PY gen12_eu_pred/scripts/gen12_self_audit.py
$PY gen12_eu_pred/scripts/gen12_verify_report.py
( cd gen12_eu_pred && ../.venv/bin/python -m pytest tests/ -q )
