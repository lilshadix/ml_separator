#!/usr/bin/env bash
# Submit the frozen five-split generation-3 protocol. DRY_RUN=1 is the default
# and performs no write and no scheduler action.

set -Eeuo pipefail
umask 027

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_DIR=${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd -P)}
PROJECT_DIR=$(cd "$PROJECT_DIR" && pwd -P)
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}
PROTOCOL_PATH=${PROTOCOL_PATH:-$PROJECT_DIR/gen3_protocol.json}
DRY_RUN=${DRY_RUN:-1}
PARTITION=${PARTITION:-campus}
ACCOUNT=${ACCOUNT:-acf-utk0011}
MAX_CONCURRENT=${MAX_CONCURRENT:-2}
CPUS=${CPUS:-16}
MEMORY=${MEMORY:-96G}
WALLTIME=${WALLTIME:-48:00:00}
RUN_TAG=${RUN_TAG:-gen3_$(date -u +%Y%m%dT%H%M%SZ)}
RUN_ROOT=${RUN_ROOT:-$PROJECT_DIR/runs/$RUN_TAG}

if [[ "$DRY_RUN" != 0 && "$DRY_RUN" != 1 ]]; then
  echo "DRY_RUN must be 0 or 1." >&2
  exit 2
fi
[[ -x "$PYTHON_BIN" ]] || { echo "Python is not executable: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$PROTOCOL_PATH" ]] || { echo "Protocol is missing: $PROTOCOL_PATH" >&2; exit 2; }
[[ "$RUN_ROOT" = /* ]] || { echo "RUN_ROOT must be absolute." >&2; exit 2; }
if [[ -e "$RUN_ROOT" ]]; then
  echo "RUN_ROOT already exists; refusing to overwrite: $RUN_ROOT" >&2
  exit 2
fi

plan=$(
  "$PYTHON_BIN" - "$PROTOCOL_PATH" <<'PY'
import json, pathlib, sys
p = json.loads(pathlib.Path(sys.argv[1]).read_text())
model = int(p["splits"]["primary_model_seed"])
print("task_index\tsplit_seed\tmodel_seed\trun_role")
for index, split in enumerate(p["splits"]["split_seeds"]):
    print(f"{index}\t{int(split)}\t{model}\tprimary_split_seed")
PY
)

protocol_hash=$(
  "$PYTHON_BIN" - "$PROTOCOL_PATH" <<'PY'
import hashlib, pathlib, sys
print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)

echo "Generation-3 frozen submission plan"
echo "RUN_ROOT=$RUN_ROOT"
echo "PROTOCOL_SHA256=$protocol_hash"
echo "PARTITION=$PARTITION ACCOUNT=$ACCOUNT"
echo "$plan"
echo "Worker: array=0-4%$MAX_CONCURRENT cpus=$CPUS mem=$MEMORY time=$WALLTIME"
echo "Aggregator: dependency=afterany:<worker_job_id> (fail-closed validation)"

if [[ "$DRY_RUN" == 1 ]]; then
  echo "DRY_RUN=1: no directories or SLURM jobs were created."
  exit 0
fi

command -v sbatch >/dev/null || { echo "sbatch is unavailable." >&2; exit 2; }
mkdir -p "$RUN_ROOT/logs"
printf '%s\n' "$plan" > "$RUN_ROOT/run_plan.tsv"
cp "$PROTOCOL_PATH" "$RUN_ROOT/gen3_protocol.json"

worker_job=$(
  sbatch --parsable \
    --partition="$PARTITION" \
    --account="$ACCOUNT" \
    --array="0-4%$MAX_CONCURRENT" \
    --cpus-per-task="$CPUS" \
    --mem="$MEMORY" \
    --time="$WALLTIME" \
    --output="$RUN_ROOT/logs/gen3-%A_%a.out" \
    --error="$RUN_ROOT/logs/gen3-%A_%a.err" \
    --export="ALL,PROJECT_DIR=$PROJECT_DIR,RUN_ROOT=$RUN_ROOT,PROTOCOL_PATH=$RUN_ROOT/gen3_protocol.json,PYTHON_BIN=$PYTHON_BIN,RUN_PLAN=$RUN_ROOT/run_plan.tsv" \
    "$PROJECT_DIR/slurm/gen3_multisplit.slurm"
)
aggregate_job=$(
  sbatch --parsable \
    --partition="$PARTITION" \
    --account="$ACCOUNT" \
    --dependency="afterany:$worker_job" \
    --output="$RUN_ROOT/logs/gen3-aggregate-%j.out" \
    --error="$RUN_ROOT/logs/gen3-aggregate-%j.err" \
    --export="ALL,PROJECT_DIR=$PROJECT_DIR,RUN_ROOT=$RUN_ROOT,PROTOCOL_PATH=$RUN_ROOT/gen3_protocol.json,PYTHON_BIN=$PYTHON_BIN" \
    "$PROJECT_DIR/slurm/aggregate_gen3.slurm"
)
printf '%s\n' "$worker_job" > "$RUN_ROOT/worker_job.id"
printf '%s\n' "$aggregate_job" > "$RUN_ROOT/aggregate_job.id"
echo "Submitted worker job $worker_job and fail-closed aggregate job $aggregate_job."

