#!/usr/bin/env bash
#SBATCH --job-name=ln-gen3-agg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -Eeuo pipefail
umask 027

PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-}}
RUN_ROOT=${RUN_ROOT:-}
PROTOCOL_PATH=${PROTOCOL_PATH:-$PROJECT_DIR/gen3_protocol.json}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}

if [[ -z "$PROJECT_DIR" || -z "$RUN_ROOT" ]]; then
  echo "PROJECT_DIR and RUN_ROOT are required." >&2
  exit 2
fi
PROJECT_DIR=$(cd "$PROJECT_DIR" && pwd -P)
[[ -d "$RUN_ROOT" ]] || { echo "RUN_ROOT is missing: $RUN_ROOT" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "Python is not executable: $PYTHON_BIN" >&2; exit 2; }

"$PYTHON_BIN" "$PROJECT_DIR/scripts/aggregate_gen3_runs.py" \
  --protocol "$PROTOCOL_PATH" \
  --run-root "$RUN_ROOT"

