#!/usr/bin/env bash
# Submission wrapper for the metal-site descriptor evaluation.
#
# Defaults to DRY_RUN=1: it prints the exact sbatch lines and a self-contained
# command to run after review, and creates nothing.

set -Eeuo pipefail
umask 027

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_DIR=${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd -P)}
PROJECT_DIR=$(cd "$PROJECT_DIR" && pwd -P)
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}
MODEL_SEEDS=${MODEL_SEEDS:-"42 7 137 2027 9001"}
SPLIT_SEEDS=${SPLIT_SEEDS:-"104729 130363 169087 214087 275015"}
DESCRIPTOR_BLOCKS=${DESCRIPTOR_BLOCKS:-"ligand_field,enclosure"}
DESCRIPTOR_PROFILE=${DESCRIPTOR_PROFILE:-core}
NULL_PERMUTATION=${NULL_PERMUTATION:-metal-preserving}
NULL_PERMUTATION_BASE_SEED=${NULL_PERMUTATION_BASE_SEED:-8100011}
MAX_CONCURRENT=${MAX_CONCURRENT:-2}
CPUS=${CPUS:-8}
MEMORY=${MEMORY:-24G}
WALLTIME=${WALLTIME:-08:00:00}
AGGREGATE_MEMORY=${AGGREGATE_MEMORY:-8G}
AGGREGATE_WALLTIME=${AGGREGATE_WALLTIME:-01:00:00}
TREES=${TREES:-200}
BOOTSTRAP=${BOOTSTRAP:-2000}
RESUME_RUN_ROOT=${RESUME_RUN_ROOT:-0}
RESUME_COMPLETED=${RESUME_COMPLETED:-1}
RESUME_AGGREGATE=${RESUME_AGGREGATE:-1}
DRY_RUN=${DRY_RUN:-1}
RUN_ROOT=${RUN_ROOT:-$PROJECT_DIR/runs/descriptor_$(date -u +%Y%m%dT%H%M%SZ)}

RUN_ROOT_CHECK=$RUN_ROOT/
if [[ "$RUN_ROOT" != /* || "$RUN_ROOT_CHECK" == *"/../"* || "$RUN_ROOT_CHECK" == *"/./"* ]]; then
  echo "RUN_ROOT must be an absolute normalized path without . or .. segments." >&2
  exit 2
fi
for value_name in DRY_RUN RESUME_RUN_ROOT RESUME_COMPLETED RESUME_AGGREGATE; do
  value=${!value_name}
  if [[ "$value" != 0 && "$value" != 1 ]]; then
    echo "$value_name must be 0 or 1." >&2
    exit 2
  fi
done
for value_name in MAX_CONCURRENT CPUS TREES BOOTSTRAP; do
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
if [[ "$DESCRIPTOR_PROFILE" != core && "$DESCRIPTOR_PROFILE" != full ]]; then
  echo "DESCRIPTOR_PROFILE must be core or full." >&2
  exit 2
fi
if [[ "$NULL_PERMUTATION" != free && "$NULL_PERMUTATION" != metal-preserving ]]; then
  echo "NULL_PERMUTATION must be free or metal-preserving." >&2
  exit 2
fi
if [[ ! "$DESCRIPTOR_BLOCKS" =~ ^(ligand_field|enclosure)(,(ligand_field|enclosure))*$ ]]; then
  echo "DESCRIPTOR_BLOCKS must be a comma-separated subset of ligand_field,enclosure." >&2
  exit 2
fi
if [[ "${PARTITION:-}" == "your_partition" || "${AGGREGATE_PARTITION:-}" == "your_partition" ]]; then
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
declare -A seen_model=()
declare -A seen_split=()
for seed in "${model_seed_array[@]}"; do
  [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Invalid model seed: $seed" >&2; exit 2; }
  (( 10#$seed <= 4000000000 )) || { echo "Model seed exceeds 4000000000: $seed" >&2; exit 2; }
  [[ -z "${seen_model[$seed]:-}" ]] || { echo "Duplicate model seed: $seed" >&2; exit 2; }
  seen_model[$seed]=1
done
for seed in "${split_seed_array[@]}"; do
  [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Invalid split seed: $seed" >&2; exit 2; }
  (( 10#$seed <= 4000000000 )) || { echo "Split seed exceeds 4000000000: $seed" >&2; exit 2; }
  [[ -z "${seen_split[$seed]:-}" ]] || { echo "Duplicate split seed: $seed" >&2; exit 2; }
  seen_split[$seed]=1
done

seed_count=${#model_seed_array[@]}
export PROJECT_DIR PYTHON_BIN MODEL_SEEDS SPLIT_SEEDS RUN_ROOT
export DESCRIPTOR_BLOCKS DESCRIPTOR_PROFILE NULL_PERMUTATION NULL_PERMUTATION_BASE_SEED
export TREES BOOTSTRAP RESUME_COMPLETED RESUME_AGGREGATE
export EXPECTED_RUNS=$seed_count
array_spec="0-$((2 * seed_count - 1))%$MAX_CONCURRENT"
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
  if [[ -z "${AGGREGATE_PARTITION:-}" ]]; then
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
  aggregate_args+=(--qos="$QOS")
fi
if [[ -n "${CONSTRAINT:-}" ]]; then
  worker_args+=(--constraint="$CONSTRAINT")
fi

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

echo "Run root: $RUN_ROOT"
echo "Array: $array_spec  (tasks 0-$((seed_count - 1)) = real arm, $seed_count-$((2 * seed_count - 1)) = permuted null arm)"
echo "Model seeds: $MODEL_SEEDS"
echo "Split seeds: $SPLIT_SEEDS"
echo "Descriptor blocks: $DESCRIPTOR_BLOCKS  profile: $DESCRIPTOR_PROFILE"
echo "Null permutation: $NULL_PERMUTATION  base seed: $NULL_PERMUTATION_BASE_SEED"
if [[ "$DRY_RUN" == 1 ]]; then
  echo "DRY_RUN=1; nothing will be submitted and no directories will be created."
  echo "The sbatch lines below are an inspection plan; submit through the wrapper command."
  print_command sbatch "${worker_args[@]}" "$PROJECT_DIR/slurm/descriptor_multiseed.slurm"
  echo "Aggregate job will use afterany:<array_job_id>:"
  print_command sbatch "${aggregate_args[@]}" \
    "--dependency=afterany:<array_job_id>" "$PROJECT_DIR/slurm/aggregate_descriptor.slurm"
  submit_command=(
    env
    "PROJECT_DIR=$PROJECT_DIR"
    "PYTHON_BIN=$PYTHON_BIN"
    "MODEL_SEEDS=$MODEL_SEEDS"
    "SPLIT_SEEDS=$SPLIT_SEEDS"
    "RUN_ROOT=$RUN_ROOT"
    "DESCRIPTOR_BLOCKS=$DESCRIPTOR_BLOCKS"
    "DESCRIPTOR_PROFILE=$DESCRIPTOR_PROFILE"
    "NULL_PERMUTATION=$NULL_PERMUTATION"
    "NULL_PERMUTATION_BASE_SEED=$NULL_PERMUTATION_BASE_SEED"
    "MAX_CONCURRENT=$MAX_CONCURRENT"
    "CPUS=$CPUS"
    "MEMORY=$MEMORY"
    "WALLTIME=$WALLTIME"
    "AGGREGATE_MEMORY=$AGGREGATE_MEMORY"
    "AGGREGATE_WALLTIME=$AGGREGATE_WALLTIME"
    "TREES=$TREES"
    "BOOTSTRAP=$BOOTSTRAP"
    "RESUME_RUN_ROOT=$RESUME_RUN_ROOT"
    "RESUME_COMPLETED=$RESUME_COMPLETED"
    "RESUME_AGGREGATE=$RESUME_AGGREGATE"
    DRY_RUN=0
  )
  for optional_name in \
    PARTITION AGGREGATE_PARTITION ACCOUNT QOS CONSTRAINT \
    DATASET_PATH VR_ASSET_PATH EXPECTED_DATASET_SHA256 EXPECTED_VR_SHA256 \
    OUTER_FOLDS INNER_FOLDS; do
    if [[ -n "${!optional_name:-}" ]]; then
      submit_command+=("$optional_name=${!optional_name}")
    fi
  done
  submit_command+=(bash "$PROJECT_DIR/slurm/submit_descriptor_multiseed.sh")
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
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/status" "$RUN_ROOT/submissions" "$RUN_ROOT/attempts"
exec 8>"$RUN_ROOT/.submission.lock"
if ! flock -n 8; then
  echo "Another submission wrapper is active for RUN_ROOT: $RUN_ROOT" >&2
  exit 2
fi

submission_tag=$(date -u +%Y%m%dT%H%M%SZ)_pid_$$
submission_ids=$RUN_ROOT/submissions/${submission_tag}.ids
seed_plan=$RUN_ROOT/seed_plan.tsv
seed_plan_candidate=$RUN_ROOT/submissions/${submission_tag}.seed_plan.tsv
{
  printf 'arm\ttask_index\tmodel_seed\tsplit_seed\tdescriptor_permutation\tpermutation_seed\n'
  for index in "${!model_seed_array[@]}"; do
    printf 'real\t%s\t%s\t%s\tnone\t0\n' \
      "$index" "${model_seed_array[$index]}" "${split_seed_array[$index]}"
  done
  for index in "${!model_seed_array[@]}"; do
    printf 'null\t%s\t%s\t%s\t%s\t%s\n' \
      "$((index + seed_count))" "${model_seed_array[$index]}" "${split_seed_array[$index]}" \
      "$NULL_PERMUTATION" "$((NULL_PERMUTATION_BASE_SEED + index))"
  done
} > "$seed_plan_candidate"
if [[ -f "$seed_plan" ]]; then
  if ! cmp -s "$seed_plan" "$seed_plan_candidate"; then
    echo "Requested seed plan does not match the immutable plan: $seed_plan" >&2
    exit 2
  fi
else
  cp "$seed_plan_candidate" "$seed_plan"
  chmod 0444 "$seed_plan"
fi

array_raw=$(sbatch "${worker_args[@]}" "$PROJECT_DIR/slurm/descriptor_multiseed.slurm")
array_job_id=${array_raw%%;*}
if [[ ! "$array_job_id" =~ ^[0-9]+$ ]]; then
  echo "Unexpected array sbatch response: $array_raw" >&2
  exit 2
fi
printf 'array_job_id=%s\n' "$array_job_id" > "$submission_ids"
echo "Submitted array job: $array_job_id"

aggregate_args+=(--dependency="afterany:$array_job_id")
aggregate_raw=$(sbatch "${aggregate_args[@]}" "$PROJECT_DIR/slurm/aggregate_descriptor.slurm")
aggregate_job_id=${aggregate_raw%%;*}
if [[ ! "$aggregate_job_id" =~ ^[0-9]+$ ]]; then
  echo "Unexpected aggregate sbatch response: $aggregate_raw" >&2
  exit 2
fi
printf 'aggregate_job_id=%s\n' "$aggregate_job_id" >> "$submission_ids"
echo "Submitted fail-closed aggregate job: $aggregate_job_id"
echo "Submission record: $submission_ids"
