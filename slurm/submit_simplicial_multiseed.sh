#!/usr/bin/env bash
set -Eeuo pipefail
umask 027

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_DIR=${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd -P)}
PROJECT_DIR=$(cd "$PROJECT_DIR" && pwd -P)
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}
MODEL_SEEDS=${MODEL_SEEDS:-"42 7 137 2027 9001"}
SPLIT_SEEDS=${SPLIT_SEEDS:-"104729 130363 169087 214087 275015"}
PAIR_SCOPE=${PAIR_SCOPE:-all}
MAX_CONCURRENT=${MAX_CONCURRENT:-1}
CPUS=${CPUS:-8}
MEMORY=${MEMORY:-32G}
WALLTIME=${WALLTIME:-24:00:00}
AGGREGATE_MEMORY=${AGGREGATE_MEMORY:-8G}
AGGREGATE_WALLTIME=${AGGREGATE_WALLTIME:-01:00:00}
ACCELERATOR=${ACCELERATOR:-cpu}
DETERMINISM=${DETERMINISM:-strict}
STAGE_VR_TO_TMP=${STAGE_VR_TO_TMP:-0}
RESUME_RUN_ROOT=${RESUME_RUN_ROOT:-0}
RESUME_COMPLETED=${RESUME_COMPLETED:-1}
RESUME_AGGREGATE=${RESUME_AGGREGATE:-1}
DRY_RUN=${DRY_RUN:-1}
RUN_ROOT=${RUN_ROOT:-$PROJECT_DIR/runs/simplicial_${PAIR_SCOPE}_$(date -u +%Y%m%dT%H%M%SZ)}

RUN_ROOT_CHECK=$RUN_ROOT/
if [[ "$RUN_ROOT" != /* || "$RUN_ROOT_CHECK" == *"/../"* || "$RUN_ROOT_CHECK" == *"/./"* ]]; then
  echo "RUN_ROOT must be an absolute normalized path without . or .. segments." >&2
  exit 2
fi
for value_name in DRY_RUN STAGE_VR_TO_TMP RESUME_RUN_ROOT RESUME_COMPLETED RESUME_AGGREGATE; do
  value=${!value_name}
  if [[ "$value" != 0 && "$value" != 1 ]]; then
    echo "$value_name must be 0 or 1." >&2
    exit 2
  fi
done
for value_name in MAX_CONCURRENT CPUS; do
  value=${!value_name}
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$value_name must be a positive integer." >&2
    exit 2
  fi
done
for value_name in MEMORY AGGREGATE_MEMORY; do
  value=${!value_name}
  if [[ ! "$value" =~ ^[1-9][0-9]*([KMGTP]([iI]?[bB])?)?$ ]]; then
    echo "$value_name is not a supported SLURM memory value: $value" >&2
    exit 2
  fi
done
for value_name in WALLTIME AGGREGATE_WALLTIME; do
  value=${!value_name}
  if [[ ! "$value" =~ ^([0-9]+-)?[0-9]+:[0-5][0-9]:[0-5][0-9]$ ]]; then
    echo "$value_name must use [days-]hours:minutes:seconds." >&2
    exit 2
  fi
done
if [[ "$ACCELERATOR" != cpu && "$ACCELERATOR" != cuda ]]; then
  echo "ACCELERATOR must be cpu or cuda." >&2
  exit 2
fi
if [[ "$PAIR_SCOPE" != all && "$PAIR_SCOPE" != adjacent ]]; then
  echo "PAIR_SCOPE must be all or adjacent." >&2
  exit 2
fi
if [[ "$DETERMINISM" != strict && "$DETERMINISM" != warn ]]; then
  echo "DETERMINISM must be strict or warn." >&2
  exit 2
fi
if [[ "$ACCELERATOR" == cpu && "$DETERMINISM" != strict ]]; then
  echo "The primary CPU profile requires DETERMINISM=strict." >&2
  exit 2
fi
if [[ "${PARTITION:-}" == "your_partition" || \
      "${AGGREGATE_PARTITION:-}" == "your_partition" ]]; then
  echo "Replace the literal placeholder 'your_partition' with a real SLURM partition, or omit PARTITION to use the cluster default." >&2
  exit 2
fi
if [[ "${ACCOUNT:-}" == "your_account" ]]; then
  echo "Replace the literal placeholder 'your_account' with a real SLURM account, or omit ACCOUNT if the cluster supplies a default." >&2
  exit 2
fi

read -r -a model_seed_array <<< "$MODEL_SEEDS"
read -r -a split_seed_array <<< "$SPLIT_SEEDS"
if [[ ${#model_seed_array[@]} -ne ${#split_seed_array[@]} ]]; then
  echo "MODEL_SEEDS and SPLIT_SEEDS must have equal length." >&2
  exit 2
fi
if (( ${#model_seed_array[@]} < 2 )); then
  echo "At least two prespecified seed pairs are required." >&2
  exit 2
fi
contains_seed() {
  local candidate=$1
  shift
  local seen
  for seen in "$@"; do
    [[ "$seen" == "$candidate" ]] && return 0
  done
  return 1
}
# Bash 3.2 treats expansion of a truly empty array as an unbound variable under
# `set -u`, so keep a sentinel that cannot collide with a numeric seed.
seen_model=("__empty__")
seen_split=("__empty__")
for seed in "${model_seed_array[@]}"; do
  [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Invalid model seed: $seed" >&2; exit 2; }
  (( 10#$seed <= 4000000000 )) || { echo "Model seed exceeds 4000000000: $seed" >&2; exit 2; }
  ! contains_seed "$seed" "${seen_model[@]}" || { echo "Duplicate model seed: $seed" >&2; exit 2; }
  seen_model+=("$seed")
done
for seed in "${split_seed_array[@]}"; do
  [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Invalid split seed: $seed" >&2; exit 2; }
  (( 10#$seed <= 4000000000 )) || { echo "Split seed exceeds 4000000000: $seed" >&2; exit 2; }
  ! contains_seed "$seed" "${seen_split[@]}" || { echo "Duplicate split seed: $seed" >&2; exit 2; }
  seen_split+=("$seed")
done

export PROJECT_DIR PYTHON_BIN MODEL_SEEDS SPLIT_SEEDS RUN_ROOT PAIR_SCOPE ACCELERATOR DETERMINISM
export STAGE_VR_TO_TMP RESUME_COMPLETED RESUME_AGGREGATE
export EXPECTED_RUNS=${#model_seed_array[@]}
array_spec="0-$((${#model_seed_array[@]} - 1))%$MAX_CONCURRENT"
worker_args=(
  --parsable
  --array="$array_spec"
  --cpus-per-task="$CPUS"
  --mem="$MEMORY"
  --time="$WALLTIME"
  --export=ALL
  --output="$RUN_ROOT/logs/%x-%A_%a.out"
  --error="$RUN_ROOT/logs/%x-%A_%a.err"
)
aggregate_args=(
  --parsable
  --cpus-per-task=1
  --mem="$AGGREGATE_MEMORY"
  --time="$AGGREGATE_WALLTIME"
  --export=ALL
  --output="$RUN_ROOT/logs/aggregate_%j.out"
  --error="$RUN_ROOT/logs/aggregate_%j.err"
)
if [[ -n "${PARTITION:-}" ]]; then
  worker_args+=(--partition="$PARTITION")
  if [[ "$ACCELERATOR" == cpu && -z "${AGGREGATE_PARTITION:-}" ]]; then
    aggregate_args+=(--partition="$PARTITION")
  fi
fi
if [[ -n "${AGGREGATE_PARTITION:-}" ]]; then
  aggregate_args+=(--partition="$AGGREGATE_PARTITION")
fi
if [[ -n "${ACCOUNT:-}" ]]; then
  worker_args+=(--account="$ACCOUNT")
  aggregate_args+=(--account="$ACCOUNT")
fi
if [[ -n "${QOS:-}" ]]; then
  worker_args+=(--qos="$QOS")
  if [[ "$ACCELERATOR" == cpu && -z "${AGGREGATE_QOS:-}" ]]; then
    aggregate_args+=(--qos="$QOS")
  fi
fi
if [[ -n "${AGGREGATE_QOS:-}" ]]; then
  aggregate_args+=(--qos="$AGGREGATE_QOS")
fi
if [[ -n "${CONSTRAINT:-}" ]]; then
  worker_args+=(--constraint="$CONSTRAINT")
fi
if [[ "$ACCELERATOR" == cuda ]]; then
  worker_args+=(--gres="${GPU_GRES:-gpu:1}")
fi

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

echo "Run root: $RUN_ROOT"
echo "Array: $array_spec"
echo "Model seeds: $MODEL_SEEDS"
echo "Split seeds: $SPLIT_SEEDS"
echo "Pair scope: $PAIR_SCOPE"
echo "Accelerator/determinism: $ACCELERATOR/$DETERMINISM"
if [[ "$DRY_RUN" == 1 ]]; then
  echo "DRY_RUN=1; nothing will be submitted and no directories will be created."
  echo "The sbatch lines below are an inspection plan; submit through the wrapper command."
  print_command sbatch "${worker_args[@]}" "$PROJECT_DIR/slurm/simplicial_multiseed.slurm"
  echo "Aggregate job will use afterany:<array_job_id>:"
  print_command sbatch "${aggregate_args[@]}" \
    "--dependency=afterany:<array_job_id>" "$PROJECT_DIR/slurm/aggregate_simplicial.slurm"
  submit_command=(
    env
    "PROJECT_DIR=$PROJECT_DIR"
    "PYTHON_BIN=$PYTHON_BIN"
    "MODEL_SEEDS=$MODEL_SEEDS"
    "SPLIT_SEEDS=$SPLIT_SEEDS"
    "RUN_ROOT=$RUN_ROOT"
    "PAIR_SCOPE=$PAIR_SCOPE"
    "MAX_CONCURRENT=$MAX_CONCURRENT"
    "CPUS=$CPUS"
    "MEMORY=$MEMORY"
    "WALLTIME=$WALLTIME"
    "AGGREGATE_MEMORY=$AGGREGATE_MEMORY"
    "AGGREGATE_WALLTIME=$AGGREGATE_WALLTIME"
    "ACCELERATOR=$ACCELERATOR"
    "DETERMINISM=$DETERMINISM"
    "STAGE_VR_TO_TMP=$STAGE_VR_TO_TMP"
    "RESUME_RUN_ROOT=$RESUME_RUN_ROOT"
    "RESUME_COMPLETED=$RESUME_COMPLETED"
    "RESUME_AGGREGATE=$RESUME_AGGREGATE"
    DRY_RUN=0
  )
  for optional_name in \
    PARTITION AGGREGATE_PARTITION ACCOUNT QOS AGGREGATE_QOS CONSTRAINT GPU_GRES \
    DATASET_PATH VR_ASSET_PATH ROW_MAP_PATH EXPECTED_DATASET_SHA256 EXPECTED_VR_SHA256 \
    TREES BOOTSTRAP OUTER_FOLDS INNER_FOLDS EPOCHS INITIALIZATIONS HIDDEN_DIM LAYERS \
    MAX_PAIRS_PER_BATCH MAX_SIMPLICES_PER_BATCH ENSEMBLE_BOOTSTRAP \
    ENSEMBLE_BOOTSTRAP_SEED; do
    if [[ -n "${!optional_name:-}" ]]; then
      submit_command+=("$optional_name=${!optional_name}")
    fi
  done
  submit_command+=(bash "$PROJECT_DIR/slurm/submit_simplicial_multiseed.sh")
  echo "Self-contained wrapper command for submission after review:"
  print_command "${submit_command[@]}"
  exit 0
fi

command -v sbatch >/dev/null || { echo "sbatch is unavailable." >&2; exit 2; }
command -v flock >/dev/null || { echo "flock is required for submission locking." >&2; exit 2; }
if [[ -e "$RUN_ROOT" ]]; then
  if [[ "$RESUME_RUN_ROOT" != 1 || ! -d "$RUN_ROOT" ]]; then
    echo "RUN_ROOT already exists; set RESUME_RUN_ROOT=1 for an explicit retry." >&2
    exit 2
  fi
else
  mkdir -p "$RUN_ROOT"
fi
RUN_ROOT=$(cd "$RUN_ROOT" && pwd -P)
export RUN_ROOT
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/status" "$RUN_ROOT/submissions"
exec 8>"$RUN_ROOT/.submission.lock"
if ! flock -n 8; then
  echo "Another submission wrapper is active for RUN_ROOT: $RUN_ROOT" >&2
  exit 2
fi

submission_tag=$(date -u +%Y%m%dT%H%M%SZ)_pid_$$
submission_env=$RUN_ROOT/submissions/${submission_tag}.env
submission_ids=$RUN_ROOT/submissions/${submission_tag}.ids
seed_plan=$RUN_ROOT/seed_plan.tsv
seed_plan_candidate=$RUN_ROOT/submissions/${submission_tag}.seed_plan.tsv
{
  printf 'task_index\tmodel_seed\tsplit_seed\n'
  for index in "${!model_seed_array[@]}"; do
    printf '%s\t%s\t%s\n' \
      "$index" "${model_seed_array[$index]}" "${split_seed_array[$index]}"
  done
} > "$seed_plan_candidate"
if [[ -f "$seed_plan" ]]; then
  if ! cmp -s "$seed_plan" "$seed_plan_candidate"; then
    echo "Requested seeds do not match immutable seed plan: $seed_plan" >&2
    exit 2
  fi
else
  cp "$seed_plan_candidate" "$seed_plan"
  chmod 0444 "$seed_plan"
fi
{
  printf 'PROJECT_DIR=%q\n' "$PROJECT_DIR"
  printf 'PYTHON_BIN=%q\n' "$PYTHON_BIN"
  printf 'MODEL_SEEDS=%q\n' "$MODEL_SEEDS"
  printf 'SPLIT_SEEDS=%q\n' "$SPLIT_SEEDS"
  printf 'PAIR_SCOPE=%q\n' "$PAIR_SCOPE"
  printf 'ACCELERATOR=%q\n' "$ACCELERATOR"
  printf 'DETERMINISM=%q\n' "$DETERMINISM"
  printf 'ARRAY_SPEC=%q\n' "$array_spec"
  printf 'RESUME_RUN_ROOT=%q\n' "$RESUME_RUN_ROOT"
  printf 'RESUME_COMPLETED=%q\n' "$RESUME_COMPLETED"
} > "$submission_env"

array_raw=$(sbatch "${worker_args[@]}" "$PROJECT_DIR/slurm/simplicial_multiseed.slurm")
array_job_id=${array_raw%%;*}
if [[ ! "$array_job_id" =~ ^[0-9]+$ ]]; then
  echo "Unexpected array sbatch response: $array_raw" >&2
  exit 2
fi
printf 'array_job_id=%s\n' "$array_job_id" > "$submission_ids"
echo "Submitted array job: $array_job_id"

aggregate_args+=(--dependency="afterany:$array_job_id")
aggregate_raw=$(sbatch "${aggregate_args[@]}" "$PROJECT_DIR/slurm/aggregate_simplicial.slurm")
aggregate_job_id=${aggregate_raw%%;*}
if [[ ! "$aggregate_job_id" =~ ^[0-9]+$ ]]; then
  echo "Unexpected aggregate sbatch response: $aggregate_raw" >&2
  exit 2
fi
printf 'aggregate_job_id=%s\n' "$aggregate_job_id" >> "$submission_ids"
echo "Submitted fail-closed aggregate job: $aggregate_job_id"
echo "Submission record: $submission_ids"
