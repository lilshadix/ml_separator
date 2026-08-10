#!/usr/bin/env bash
# Submission wrapper for the leakage-safe all-pairs A0--A6 benchmark.
#
# DRY_RUN=1 is the safe default.  A dry run prints the exact plan and creates
# no directories, seed files, locks, or SLURM jobs.

set -Eeuo pipefail
umask 027

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_DIR=${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd -P)}
PROJECT_DIR=$(cd "$PROJECT_DIR" && pwd -P)
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}
DATASET_PATH=${DATASET_PATH:-$PROJECT_DIR/dataset with 3D structures/dataset.parquet}
VR_ASSET_PATH=${VR_ASSET_PATH:-$PROJECT_DIR/dataset with 3D structures/features/vietoris_rips_inputs.npz}
MODEL_SEEDS=${MODEL_SEEDS:-"42 7 137 2027 9001"}
SPLIT_SEED=${SPLIT_SEED:-104729}
PAIR_SCOPE=${PAIR_SCOPE:-all}
GROUP_MODE=${GROUP_MODE:-extractant}
DESCRIPTOR_BLOCKS=${DESCRIPTOR_BLOCKS:-global_shape,coordination_shape}
DESCRIPTOR_PROFILE=${DESCRIPTOR_PROFILE:-core}
SHUFFLE_SEEDS=${SHUFFLE_SEEDS:-1009,2017,3019}
INCLUDE_BLOCK_ABLATIONS=${INCLUDE_BLOCK_ABLATIONS:-1}
# Generation-2 switches. All default to off, so an unset environment submits the
# frozen A0--A6 protocol exactly as before.
INCLUDE_PAIR_RESPONSE_3D=${INCLUDE_PAIR_RESPONSE_3D:-0}
INCLUDE_ELECTRONIC=${INCLUDE_ELECTRONIC:-0}
INCLUDE_EXTENSION_ARMS=${INCLUDE_EXTENSION_ARMS:-0}
INCLUDE_SYMMETRIC_3D=${INCLUDE_SYMMETRIC_3D:-0}
INCLUDE_REFERENCE_BASELINES=${INCLUDE_REFERENCE_BASELINES:-0}
INCLUDE_2D_SENSITIVITY=${INCLUDE_2D_SENSITIVITY:-0}
EXTENSION_SHUFFLE_SEEDS=${EXTENSION_SHUFFLE_SEEDS:-}
PRESPECIFIED_ARMS=${PRESPECIFIED_ARMS:-}
DELTA3D_FEATURE_SET=${DELTA3D_FEATURE_SET:-compact-invariant}
REPLICATE_POLICY=${REPLICATE_POLICY:-unique}
INCLUDE_DONOR_COMPOSITION_COUNTS=${INCLUDE_DONOR_COMPOSITION_COUNTS:-1}
OUTER_FOLDS=${OUTER_FOLDS:-5}
INNER_FOLDS=${INNER_FOLDS:-3}
TREES=${TREES:-200}
BOOTSTRAP=${BOOTSTRAP:-1000}
AGGREGATE_BOOTSTRAP=${AGGREGATE_BOOTSTRAP:-2000}
AGGREGATE_BOOTSTRAP_SEED=${AGGREGATE_BOOTSTRAP_SEED:-8675309}
MAX_CONCURRENT=${MAX_CONCURRENT:-2}
CPUS=${CPUS:-8}
MEMORY=${MEMORY:-48G}
WALLTIME=${WALLTIME:-24:00:00}
AGGREGATE_MEMORY=${AGGREGATE_MEMORY:-16G}
AGGREGATE_WALLTIME=${AGGREGATE_WALLTIME:-02:00:00}
# Frozen ISAAC submission identity for this project. Both remain overrideable
# for an intentional run on another valid account/partition.
PARTITION=${PARTITION:-campus}
ACCOUNT=${ACCOUNT:-acf-utk0011}
QUICK=${QUICK:-0}
RESUME_RUN_ROOT=${RESUME_RUN_ROOT:-0}
RESUME_COMPLETED=${RESUME_COMPLETED:-1}
RESUME_AGGREGATE=${RESUME_AGGREGATE:-1}
DRY_RUN=${DRY_RUN:-1}
RUN_ROOT=${RUN_ROOT:-$PROJECT_DIR/runs/ablation_${PAIR_SCOPE}_$(date -u +%Y%m%dT%H%M%SZ)}

RUN_ROOT_CHECK=$RUN_ROOT/
if [[ "$RUN_ROOT" != /* || "$RUN_ROOT_CHECK" == *"/../"* || "$RUN_ROOT_CHECK" == *"/./"* ]]; then
  echo "RUN_ROOT must be an absolute normalized path without . or .. segments." >&2
  exit 2
fi
for value_name in \
  DRY_RUN QUICK INCLUDE_BLOCK_ABLATIONS INCLUDE_DONOR_COMPOSITION_COUNTS \
  RESUME_RUN_ROOT RESUME_COMPLETED RESUME_AGGREGATE \
  INCLUDE_PAIR_RESPONSE_3D INCLUDE_ELECTRONIC INCLUDE_EXTENSION_ARMS \
  INCLUDE_SYMMETRIC_3D INCLUDE_REFERENCE_BASELINES INCLUDE_2D_SENSITIVITY; do
  value=${!value_name}
  if [[ "$value" != 0 && "$value" != 1 ]]; then
    echo "$value_name must be 0 or 1." >&2
    exit 2
  fi
done
for value_name in MAX_CONCURRENT CPUS OUTER_FOLDS INNER_FOLDS TREES BOOTSTRAP AGGREGATE_BOOTSTRAP; do
  value=${!value_name}
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$value_name must be a positive integer." >&2
    exit 2
  fi
done
if (( OUTER_FOLDS < 2 || INNER_FOLDS < 2 )); then
  echo "OUTER_FOLDS and INNER_FOLDS must both be at least 2." >&2
  exit 2
fi
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
if [[ "$PAIR_SCOPE" != all && "$PAIR_SCOPE" != adjacent ]]; then
  echo "PAIR_SCOPE must be all or adjacent." >&2
  exit 2
fi
if [[ "$GROUP_MODE" != extractant ]]; then
  echo "The primary multi-seed protocol requires GROUP_MODE=extractant." >&2
  exit 2
fi
if [[ "$DESCRIPTOR_PROFILE" != core && "$DESCRIPTOR_PROFILE" != full ]]; then
  echo "DESCRIPTOR_PROFILE must be core or full." >&2
  exit 2
fi
if [[ ! "$DESCRIPTOR_BLOCKS" =~ ^(global_shape|coordination_shape|ligand_field|enclosure)(,(global_shape|coordination_shape|ligand_field|enclosure))*$ ]]; then
  echo "DESCRIPTOR_BLOCKS must be a comma-separated subset of ligand_field,enclosure,global_shape,coordination_shape." >&2
  exit 2
fi
CANONICAL_DESCRIPTOR_BLOCKS=
for block in ligand_field enclosure global_shape coordination_shape; do
  case ",$DESCRIPTOR_BLOCKS," in
    *",$block,"*)
      if [[ -n "$CANONICAL_DESCRIPTOR_BLOCKS" ]]; then
        CANONICAL_DESCRIPTOR_BLOCKS=$CANONICAL_DESCRIPTOR_BLOCKS,$block
      else
        CANONICAL_DESCRIPTOR_BLOCKS=$block
      fi
      ;;
  esac
done
if [[ "$CANONICAL_DESCRIPTOR_BLOCKS" != "$DESCRIPTOR_BLOCKS" ]]; then
  echo "DESCRIPTOR_BLOCKS must be unique and in canonical order: ligand_field,enclosure,global_shape,coordination_shape." >&2
  exit 2
fi
if [[ ! "$SHUFFLE_SEEDS" =~ ^([0-9]+)(,[0-9]+)*$ && "$SHUFFLE_SEEDS" != none && "$SHUFFLE_SEEDS" != off ]]; then
  echo "SHUFFLE_SEEDS must be 'none', 'off', or comma-separated nonnegative integers." >&2
  exit 2
fi
if [[ "$DELTA3D_FEATURE_SET" != compact-invariant && "$DELTA3D_FEATURE_SET" != all-ranked ]]; then
  echo "DELTA3D_FEATURE_SET must be compact-invariant or all-ranked." >&2
  exit 2
fi
if [[ "$REPLICATE_POLICY" != unique && "$REPLICATE_POLICY" != median ]]; then
  echo "REPLICATE_POLICY must be unique or median." >&2
  exit 2
fi
if [[ "$INCLUDE_DONOR_COMPOSITION_COUNTS" != 1 ]]; then
  echo "The ablation runner currently requires INCLUDE_DONOR_COMPOSITION_COUNTS=1." >&2
  exit 2
fi
if [[ "$INCLUDE_EXTENSION_ARMS" == 1 && "$INCLUDE_PAIR_RESPONSE_3D" == 0 && "$INCLUDE_ELECTRONIC" == 0 ]]; then
  echo "INCLUDE_EXTENSION_ARMS=1 needs INCLUDE_PAIR_RESPONSE_3D=1 and/or INCLUDE_ELECTRONIC=1." >&2
  exit 2
fi
if [[ -n "$EXTENSION_SHUFFLE_SEEDS" ]]; then
  if [[ "$INCLUDE_EXTENSION_ARMS" != 1 ]]; then
    echo "EXTENSION_SHUFFLE_SEEDS requires INCLUDE_EXTENSION_ARMS=1." >&2
    exit 2
  fi
  if [[ ! "$EXTENSION_SHUFFLE_SEEDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "EXTENSION_SHUFFLE_SEEDS must be comma-separated nonnegative integers." >&2
    exit 2
  fi
fi
if [[ -n "$PRESPECIFIED_ARMS" ]]; then
  if [[ ! "$PRESPECIFIED_ARMS" =~ ^A[0-6](,A[0-6])*$ ]]; then
    echo "PRESPECIFIED_ARMS must be a comma-separated subset of A0..A6." >&2
    exit 2
  fi
  if [[ ",$PRESPECIFIED_ARMS," != *",A2,"* ]]; then
    echo "PRESPECIFIED_ARMS must include A2, the reference arm of every comparison." >&2
    exit 2
  fi
fi
for digest_name in EXPECTED_DATASET_SHA256 EXPECTED_VR_SHA256; do
  digest_value=${!digest_name:-}
  if [[ -n "$digest_value" && ! "$digest_value" =~ ^[0-9a-f]{64}$ ]]; then
    echo "$digest_name must be a lowercase 64-character SHA-256 digest." >&2
    exit 2
  fi
done
if [[ ! "$SPLIT_SEED" =~ ^[0-9]+$ ]] || (( 10#$SPLIT_SEED > 4000000000 )); then
  echo "SPLIT_SEED must be an integer in [0, 4000000000]." >&2
  exit 2
fi
if [[ ! "$AGGREGATE_BOOTSTRAP_SEED" =~ ^[0-9]+$ ]] || (( 10#$AGGREGATE_BOOTSTRAP_SEED > 4000000000 )); then
  echo "AGGREGATE_BOOTSTRAP_SEED must be an integer in [0, 4000000000]." >&2
  exit 2
fi
if [[ "$PARTITION" == your_partition || "$PARTITION" == actual_partition || \
      "${AGGREGATE_PARTITION:-}" == your_partition || \
      "${AGGREGATE_PARTITION:-}" == actual_partition ]]; then
  echo "Placeholder partition is forbidden; use the default 'campus' or an explicitly verified partition." >&2
  exit 2
fi
if [[ "$ACCOUNT" == your_account || "$ACCOUNT" == actual_account ]]; then
  echo "Placeholder account is forbidden; use the default 'acf-utk0011' or an explicitly verified account." >&2
  exit 2
fi

read -r -a MODEL_SEED_ARRAY <<< "$MODEL_SEEDS"
if [[ ${#MODEL_SEED_ARRAY[@]} -ne 5 ]]; then
  echo "Exactly five model seeds are required; observed ${#MODEL_SEED_ARRAY[@]}." >&2
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
SEEN_MODEL_SEEDS=("__empty__")
for seed in "${MODEL_SEED_ARRAY[@]}"; do
  if [[ ! "$seed" =~ ^[0-9]+$ ]] || (( 10#$seed > 4000000000 )); then
    echo "Model seeds must be integers in [0, 4000000000]: $seed" >&2
    exit 2
  fi
  if contains_seed "$seed" "${SEEN_MODEL_SEEDS[@]}"; then
    echo "Duplicate model seed: $seed" >&2
    exit 2
  fi
  SEEN_MODEL_SEEDS+=("$seed")
done
if [[ "$SHUFFLE_SEEDS" != none && "$SHUFFLE_SEEDS" != off ]]; then
  IFS=, read -r -a SHUFFLE_SEED_ARRAY <<< "$SHUFFLE_SEEDS"
  SEEN_SHUFFLE_SEEDS=("__empty__")
  for seed in "${SHUFFLE_SEED_ARRAY[@]}"; do
    if (( 10#$seed > 4000000000 )); then
      echo "Shuffle seed exceeds 4000000000: $seed" >&2
      exit 2
    fi
    if contains_seed "$seed" "${SEEN_SHUFFLE_SEEDS[@]}"; then
      echo "Duplicate shuffle seed: $seed" >&2
      exit 2
    fi
    SEEN_SHUFFLE_SEEDS+=("$seed")
  done
fi

EXPECTED_RUNS=5
export PROJECT_DIR PYTHON_BIN MODEL_SEEDS SPLIT_SEED RUN_ROOT PAIR_SCOPE GROUP_MODE
export DESCRIPTOR_BLOCKS DESCRIPTOR_PROFILE SHUFFLE_SEEDS INCLUDE_BLOCK_ABLATIONS
export DELTA3D_FEATURE_SET REPLICATE_POLICY INCLUDE_DONOR_COMPOSITION_COUNTS
export OUTER_FOLDS INNER_FOLDS TREES BOOTSTRAP QUICK RESUME_COMPLETED
export INCLUDE_PAIR_RESPONSE_3D INCLUDE_ELECTRONIC INCLUDE_EXTENSION_ARMS
export INCLUDE_SYMMETRIC_3D INCLUDE_REFERENCE_BASELINES INCLUDE_2D_SENSITIVITY
export EXTENSION_SHUFFLE_SEEDS PRESPECIFIED_ARMS
export EXPECTED_RUNS AGGREGATE_BOOTSTRAP AGGREGATE_BOOTSTRAP_SEED RESUME_AGGREGATE
if [[ -n "${EXPECTED_DATASET_SHA256:-}" ]]; then export EXPECTED_DATASET_SHA256; fi
if [[ -n "${EXPECTED_VR_SHA256:-}" ]]; then export EXPECTED_VR_SHA256; fi
export DATASET_PATH VR_ASSET_PATH

ARRAY_SPEC="0-4%$MAX_CONCURRENT"
WORKER_ARGS=(
  --parsable
  --array="$ARRAY_SPEC"
  --cpus-per-task="$CPUS"
  --mem="$MEMORY"
  --time="$WALLTIME"
  --export=ALL
  --output="$RUN_ROOT/logs/%x-%A_%a.out"
  --error="$RUN_ROOT/logs/%x-%A_%a.err"
)
AGGREGATE_ARGS=(
  --parsable
  --cpus-per-task=1
  --mem="$AGGREGATE_MEMORY"
  --time="$AGGREGATE_WALLTIME"
  --export=ALL
  --output="$RUN_ROOT/logs/aggregate_%j.out"
  --error="$RUN_ROOT/logs/aggregate_%j.err"
)
if [[ -n "${PARTITION:-}" ]]; then
  WORKER_ARGS+=(--partition="$PARTITION")
  if [[ -z "${AGGREGATE_PARTITION:-}" ]]; then
    AGGREGATE_ARGS+=(--partition="$PARTITION")
  fi
fi
if [[ -n "${AGGREGATE_PARTITION:-}" ]]; then
  AGGREGATE_ARGS+=(--partition="$AGGREGATE_PARTITION")
fi
if [[ -n "${ACCOUNT:-}" ]]; then
  WORKER_ARGS+=(--account="$ACCOUNT")
  AGGREGATE_ARGS+=(--account="$ACCOUNT")
fi
if [[ -n "${QOS:-}" ]]; then
  WORKER_ARGS+=(--qos="$QOS")
  AGGREGATE_ARGS+=(--qos="$QOS")
fi
if [[ -n "${CONSTRAINT:-}" ]]; then
  WORKER_ARGS+=(--constraint="$CONSTRAINT")
fi

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

echo "Run root: $RUN_ROOT"
echo "Array: $ARRAY_SPEC  (five estimator seeds, one frozen split)"
echo "Model seeds: $MODEL_SEEDS"
echo "Split seed: $SPLIT_SEED"
echo "Pair scope: $PAIR_SCOPE"
echo "Group mode: $GROUP_MODE"
echo "Descriptor blocks: $DESCRIPTOR_BLOCKS  profile: $DESCRIPTOR_PROFILE"
echo "Training-only 3D shuffle seeds: $SHUFFLE_SEEDS"
echo "A2+D1...D5 exploratory arms enabled: $INCLUDE_BLOCK_ABLATIONS"
echo "Generation-2 blocks: pair_response=$INCLUDE_PAIR_RESPONSE_3D electronic=$INCLUDE_ELECTRONIC arms=$INCLUDE_EXTENSION_ARMS symmetric=$INCLUDE_SYMMETRIC_3D references=$INCLUDE_REFERENCE_BASELINES two_d_sensitivity=$INCLUDE_2D_SENSITIVITY"
echo "Extension permutation-control seeds: ${EXTENSION_SHUFFLE_SEEDS:-none}"
echo "Pre-specified arms recomputed: ${PRESPECIFIED_ARMS:-A0..A6}"
echo "Delta3D feature set: $DELTA3D_FEATURE_SET  replicate policy: $REPLICATE_POLICY"
echo "Donor-composition counts enabled: $INCLUDE_DONOR_COMPOSITION_COUNTS"
echo "Dataset path: $DATASET_PATH"
echo "VR asset path: $VR_ASSET_PATH"
if [[ "$DRY_RUN" == 1 ]]; then
  echo "DRY_RUN=1; nothing will be submitted and no directories will be created."
  echo "The sbatch lines below are an inspection plan; submit through the wrapper command."
  print_command sbatch "${WORKER_ARGS[@]}" "$PROJECT_DIR/slurm/ablation_multiseed.slurm"
  echo "Aggregate job will use afterany:<array_job_id> and fail closed on missing runs:"
  print_command sbatch "${AGGREGATE_ARGS[@]}" \
    "--dependency=afterany:<array_job_id>" "$PROJECT_DIR/slurm/aggregate_ablation.slurm"
  SUBMIT_COMMAND=(
    env
    "PROJECT_DIR=$PROJECT_DIR"
    "PYTHON_BIN=$PYTHON_BIN"
    "DATASET_PATH=$DATASET_PATH"
    "VR_ASSET_PATH=$VR_ASSET_PATH"
    "MODEL_SEEDS=$MODEL_SEEDS"
    "SPLIT_SEED=$SPLIT_SEED"
    "RUN_ROOT=$RUN_ROOT"
    "PAIR_SCOPE=$PAIR_SCOPE"
    "GROUP_MODE=$GROUP_MODE"
    "DESCRIPTOR_BLOCKS=$DESCRIPTOR_BLOCKS"
    "DESCRIPTOR_PROFILE=$DESCRIPTOR_PROFILE"
    "SHUFFLE_SEEDS=$SHUFFLE_SEEDS"
    "INCLUDE_BLOCK_ABLATIONS=$INCLUDE_BLOCK_ABLATIONS"
    "INCLUDE_PAIR_RESPONSE_3D=$INCLUDE_PAIR_RESPONSE_3D"
    "INCLUDE_ELECTRONIC=$INCLUDE_ELECTRONIC"
    "INCLUDE_EXTENSION_ARMS=$INCLUDE_EXTENSION_ARMS"
    "INCLUDE_SYMMETRIC_3D=$INCLUDE_SYMMETRIC_3D"
    "INCLUDE_2D_SENSITIVITY=$INCLUDE_2D_SENSITIVITY"
    "INCLUDE_REFERENCE_BASELINES=$INCLUDE_REFERENCE_BASELINES"
    "EXTENSION_SHUFFLE_SEEDS=$EXTENSION_SHUFFLE_SEEDS"
    "PRESPECIFIED_ARMS=$PRESPECIFIED_ARMS"
    "DELTA3D_FEATURE_SET=$DELTA3D_FEATURE_SET"
    "REPLICATE_POLICY=$REPLICATE_POLICY"
    "INCLUDE_DONOR_COMPOSITION_COUNTS=$INCLUDE_DONOR_COMPOSITION_COUNTS"
    "OUTER_FOLDS=$OUTER_FOLDS"
    "INNER_FOLDS=$INNER_FOLDS"
    "TREES=$TREES"
    "BOOTSTRAP=$BOOTSTRAP"
    "AGGREGATE_BOOTSTRAP=$AGGREGATE_BOOTSTRAP"
    "AGGREGATE_BOOTSTRAP_SEED=$AGGREGATE_BOOTSTRAP_SEED"
    "MAX_CONCURRENT=$MAX_CONCURRENT"
    "CPUS=$CPUS"
    "MEMORY=$MEMORY"
    "WALLTIME=$WALLTIME"
    "AGGREGATE_MEMORY=$AGGREGATE_MEMORY"
    "AGGREGATE_WALLTIME=$AGGREGATE_WALLTIME"
    "QUICK=$QUICK"
    "RESUME_RUN_ROOT=$RESUME_RUN_ROOT"
    "RESUME_COMPLETED=$RESUME_COMPLETED"
    "RESUME_AGGREGATE=$RESUME_AGGREGATE"
    DRY_RUN=0
  )
  for optional_name in \
    PARTITION AGGREGATE_PARTITION ACCOUNT QOS CONSTRAINT \
    EXPECTED_DATASET_SHA256 EXPECTED_VR_SHA256; do
    if [[ -n "${!optional_name:-}" ]]; then
      SUBMIT_COMMAND+=("$optional_name=${!optional_name}")
    fi
  done
  SUBMIT_COMMAND+=(bash "$PROJECT_DIR/slurm/submit_ablation_multiseed.sh")
  echo "Self-contained wrapper command for submission after review:"
  print_command "${SUBMIT_COMMAND[@]}"
  exit 0
fi

command -v sbatch >/dev/null || { echo "sbatch is unavailable." >&2; exit 2; }
command -v flock >/dev/null || { echo "flock is required for submission locking." >&2; exit 2; }
if [[ "$PYTHON_BIN" == */* ]]; then
  [[ -x "$PYTHON_BIN" ]] || { echo "Python is not executable: $PYTHON_BIN" >&2; exit 2; }
else
  command -v "$PYTHON_BIN" >/dev/null || { echo "Python not found: $PYTHON_BIN" >&2; exit 2; }
fi
[[ -f "$PROJECT_DIR/slurm/ablation_multiseed.slurm" ]] || { echo "Worker script is missing." >&2; exit 2; }
[[ -f "$PROJECT_DIR/slurm/aggregate_ablation.slurm" ]] || { echo "Aggregate script is missing." >&2; exit 2; }
DATASET_PATH=$("$PYTHON_BIN" -c '
import pathlib, sys
path = pathlib.Path(sys.argv[1]).expanduser().resolve(strict=True)
if not path.is_file():
    raise SystemExit("Dataset is not a regular file: " + str(path))
print(path)
' "$DATASET_PATH")
VR_ASSET_PATH=$("$PYTHON_BIN" -c '
import pathlib, sys
path = pathlib.Path(sys.argv[1]).expanduser().resolve(strict=True)
if not path.is_file():
    raise SystemExit("VR asset is not a regular file: " + str(path))
print(path)
' "$VR_ASSET_PATH")
for path_name in DATASET_PATH VR_ASSET_PATH; do
  path_value=${!path_name}
  case "$path_value" in
    *$'\t'*|*$'\n'*)
      echo "$path_name cannot contain tab or newline characters." >&2
      exit 2
      ;;
  esac
done
DATASET_SHA256=$("$PYTHON_BIN" -c '
import hashlib, pathlib, sys
digest = hashlib.sha256()
with pathlib.Path(sys.argv[1]).open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
' "$DATASET_PATH")
VR_ASSET_SHA256=$("$PYTHON_BIN" -c '
import hashlib, pathlib, sys
digest = hashlib.sha256()
with pathlib.Path(sys.argv[1]).open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
' "$VR_ASSET_PATH")
if [[ -n "${EXPECTED_DATASET_SHA256:-}" && "$EXPECTED_DATASET_SHA256" != "$DATASET_SHA256" ]]; then
  echo "Dataset SHA-256 mismatch: expected $EXPECTED_DATASET_SHA256, observed $DATASET_SHA256." >&2
  exit 2
fi
if [[ -n "${EXPECTED_VR_SHA256:-}" && "$EXPECTED_VR_SHA256" != "$VR_ASSET_SHA256" ]]; then
  echo "VR asset SHA-256 mismatch: expected $EXPECTED_VR_SHA256, observed $VR_ASSET_SHA256." >&2
  exit 2
fi
EXPECTED_DATASET_SHA256=$DATASET_SHA256
EXPECTED_VR_SHA256=$VR_ASSET_SHA256
export DATASET_PATH VR_ASSET_PATH EXPECTED_DATASET_SHA256 EXPECTED_VR_SHA256
echo "Frozen dataset SHA-256: $DATASET_SHA256"
echo "Frozen VR asset SHA-256: $VR_ASSET_SHA256"
RUN_ROOT_PREEXISTED=0
if [[ -e "$RUN_ROOT" ]]; then
  RUN_ROOT_PREEXISTED=1
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

SUBMISSION_TAG=$(date -u +%Y%m%dT%H%M%SZ)_pid_$$
SUBMISSION_IDS=$RUN_ROOT/submissions/${SUBMISSION_TAG}.ids
SEED_PLAN=$RUN_ROOT/seed_plan.tsv
SEED_PLAN_CANDIDATE=$RUN_ROOT/submissions/${SUBMISSION_TAG}.seed_plan.tsv
EXPERIMENT_CONTRACT=$RUN_ROOT/experiment_contract.tsv
EXPERIMENT_CONTRACT_CANDIDATE=$RUN_ROOT/submissions/${SUBMISSION_TAG}.experiment_contract.tsv
{
  printf 'task_index\tmodel_seed\tsplit_seed\n'
  for index in "${!MODEL_SEED_ARRAY[@]}"; do
    printf '%s\t%s\t%s\n' "$index" "${MODEL_SEED_ARRAY[$index]}" "$SPLIT_SEED"
  done
} > "$SEED_PLAN_CANDIDATE"
if [[ -f "$SEED_PLAN" ]]; then
  if ! cmp -s "$SEED_PLAN" "$SEED_PLAN_CANDIDATE"; then
    echo "Requested seed plan does not match the immutable plan: $SEED_PLAN" >&2
    exit 2
  fi
elif [[ "$RUN_ROOT_PREEXISTED" == 1 ]]; then
  echo "Refusing to resume a run root without its immutable seed plan: $SEED_PLAN" >&2
  exit 2
else
  chmod 0444 "$SEED_PLAN_CANDIDATE"
  cp -p "$SEED_PLAN_CANDIDATE" "$RUN_ROOT/.seed_plan.tsv.tmp.$$.new"
  mv "$RUN_ROOT/.seed_plan.tsv.tmp.$$.new" "$SEED_PLAN"
fi

{
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    schema_version pair_scope group_mode split_seed outer_folds inner_folds trees \
    bootstrap descriptor_blocks descriptor_profile shuffle_seeds \
    include_block_ablations quick_mode delta3d_feature_set replicate_policy \
    include_donor_composition_counts aggregate_bootstrap_replicates \
    aggregate_bootstrap_seed dataset_path dataset_sha256 vr_asset_path \
    vr_asset_sha256 include_pair_response_3d include_electronic \
    include_extension_arms include_symmetric_3d include_reference_baselines \
    include_2d_sensitivity \
    extension_shuffle_seeds prespecified_arms
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    2.0 "$PAIR_SCOPE" "$GROUP_MODE" "$SPLIT_SEED" "$OUTER_FOLDS" "$INNER_FOLDS" \
    "$TREES" "$BOOTSTRAP" "$DESCRIPTOR_BLOCKS" "$DESCRIPTOR_PROFILE" \
    "$SHUFFLE_SEEDS" "$INCLUDE_BLOCK_ABLATIONS" "$QUICK" \
    "$DELTA3D_FEATURE_SET" "$REPLICATE_POLICY" "$INCLUDE_DONOR_COMPOSITION_COUNTS" \
    "$AGGREGATE_BOOTSTRAP" "$AGGREGATE_BOOTSTRAP_SEED" "$DATASET_PATH" \
    "$DATASET_SHA256" "$VR_ASSET_PATH" "$VR_ASSET_SHA256" \
    "$INCLUDE_PAIR_RESPONSE_3D" "$INCLUDE_ELECTRONIC" "$INCLUDE_EXTENSION_ARMS" \
    "$INCLUDE_SYMMETRIC_3D" "$INCLUDE_REFERENCE_BASELINES" \
    "$INCLUDE_2D_SENSITIVITY" \
    "${EXTENSION_SHUFFLE_SEEDS:-none}" "${PRESPECIFIED_ARMS:-A0-A6}"
} > "$EXPERIMENT_CONTRACT_CANDIDATE"
if [[ -f "$EXPERIMENT_CONTRACT" ]]; then
  if ! cmp -s "$EXPERIMENT_CONTRACT" "$EXPERIMENT_CONTRACT_CANDIDATE"; then
    echo "Requested scientific protocol does not match the immutable experiment contract: $EXPERIMENT_CONTRACT" >&2
    exit 2
  fi
elif [[ "$RUN_ROOT_PREEXISTED" == 1 ]]; then
  echo "Refusing to resume a run root without its immutable experiment contract: $EXPERIMENT_CONTRACT" >&2
  exit 2
else
  chmod 0444 "$EXPERIMENT_CONTRACT_CANDIDATE"
  cp -p "$EXPERIMENT_CONTRACT_CANDIDATE" "$RUN_ROOT/.experiment_contract.tsv.tmp.$$.new"
  mv "$RUN_ROOT/.experiment_contract.tsv.tmp.$$.new" "$EXPERIMENT_CONTRACT"
fi

ARRAY_RAW=$(sbatch "${WORKER_ARGS[@]}" "$PROJECT_DIR/slurm/ablation_multiseed.slurm")
ARRAY_JOB_ID=${ARRAY_RAW%%;*}
if [[ ! "$ARRAY_JOB_ID" =~ ^[0-9]+$ ]]; then
  echo "Unexpected array sbatch response: $ARRAY_RAW" >&2
  exit 2
fi
printf 'array_job_id=%s\n' "$ARRAY_JOB_ID" > "$SUBMISSION_IDS"
echo "Submitted five-seed array job: $ARRAY_JOB_ID"

AGGREGATE_ARGS+=(--dependency="afterany:$ARRAY_JOB_ID")
AGGREGATE_RAW=$(sbatch "${AGGREGATE_ARGS[@]}" "$PROJECT_DIR/slurm/aggregate_ablation.slurm")
AGGREGATE_JOB_ID=${AGGREGATE_RAW%%;*}
if [[ ! "$AGGREGATE_JOB_ID" =~ ^[0-9]+$ ]]; then
  echo "Unexpected aggregate sbatch response: $AGGREGATE_RAW" >&2
  exit 2
fi
printf 'aggregate_job_id=%s\n' "$AGGREGATE_JOB_ID" >> "$SUBMISSION_IDS"
echo "Submitted fail-closed afterany aggregate job: $AGGREGATE_JOB_ID"
echo "Submission record: $SUBMISSION_IDS"
