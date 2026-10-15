#!/usr/bin/env bash
# The single command that reproduces gen20 end to end (from the repository root).
#   bash generations/gen20_kshot/scripts/g20_reproduce.sh            # full run
#   SMOKE=1 bash generations/gen20_kshot/scripts/g20_reproduce.sh    # labelled smoke run only
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=.venv/bin/python
export LOKY_MAX_CPU_COUNT=2 OMP_NUM_THREADS=4
$PY -m pytest generations/gen16_leads/tests/test_anchors.py -q -m "not slow" -p no:cacheprovider   # anchors first
$PY -m pytest generations/gen20_kshot/tests -q -p no:cacheprovider                                   # validity tests
if [ "${SMOKE:-0}" = "1" ]; then
  $PY generations/gen20_kshot/scripts/g20_discovery.py --label smoke --designs BP --seeds 104729 --smoke --setting-b
  $PY generations/gen20_kshot/scripts/g20_report.py --label smoke --replicates 500
  exit 0
fi
$PY generations/gen20_kshot/scripts/g20_seeds.py
$PY generations/gen20_kshot/scripts/g20_replicate_gen15.py
$PY generations/gen20_kshot/scripts/g20_discovery.py --label discovery --setting-b
$PY generations/gen20_kshot/scripts/g20_report.py --label discovery
$PY generations/gen20_kshot/scripts/g20_confirm.py --discovery-label discovery
$PY generations/gen20_kshot/scripts/g20_discovery.py --label discovery_nd_sm --designs BP --query-name adjacent_gen13 --models metal
