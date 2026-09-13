#!/usr/bin/env bash
# Re-derive every table, figure, manifest and verification from the predictions on disk.
# Safe to re-run: it recomputes analysis, never models.
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=".venv/bin/python"
export PYTHONPATH=generations/gen12_eu_pred

$PY generations/gen12_eu_pred/scripts/gen12_analysis.py --design B
$PY generations/gen12_eu_pred/scripts/gen12_analysis.py --design A
$PY generations/gen12_eu_pred/scripts/gen12_hypotheses.py
$PY generations/gen12_eu_pred/scripts/gen12_frontier.py
$PY generations/gen12_eu_pred/scripts/gen12_diagnostics.py
$PY generations/gen12_eu_pred/scripts/gen12_figures.py --design B
$PY generations/gen12_eu_pred/scripts/gen12_headline_tables.py
$PY generations/gen12_eu_pred/scripts/gen12_manifest.py
$PY generations/gen12_eu_pred/scripts/gen12_self_audit.py
$PY generations/gen12_eu_pred/scripts/gen12_verify_report.py
( cd generations/gen12_eu_pred && ../../.venv/bin/python -m pytest tests/ -q )
