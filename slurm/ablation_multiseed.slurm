#!/usr/bin/env bash
#SBATCH --job-name=ln-ablation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-4%2
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# One model-seed replicate of the leakage-safe A0--A6 benchmark.
#
# Every array task uses the same SPLIT_SEED and therefore the same outer and
# inner grouped folds.  Only estimator/bootstrap randomness changes across
# tasks.  The immutable seed_plan.tsv and experiment_contract.tsv written by
# the submission wrapper are the authorities for task mapping and protocol.

set -Eeuo pipefail
umask 027

PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$PROJECT_DIR" ]]; then
  echo "PROJECT_DIR is required." >&2
  exit 2
fi
PROJECT_DIR=$(cd "$PROJECT_DIR" && pwd -P)

PYTHON_BIN=${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}
DATASET_PATH=${DATASET_PATH:-$PROJECT_DIR/dataset with 3D structures/dataset.parquet}
VR_ASSET_PATH=${VR_ASSET_PATH:-$PROJECT_DIR/dataset with 3D structures/features/vietoris_rips_inputs.npz}
RUN_ROOT=${RUN_ROOT:-$PROJECT_DIR/runs/ablation_${SLURM_ARRAY_JOB_ID:-manual}}
MODEL_SEEDS=${MODEL_SEEDS:-"42 7 137 2027 9001"}
SPLIT_SEED=${SPLIT_SEED:-104729}
PAIR_SCOPE=${PAIR_SCOPE:-all}
GROUP_MODE=${GROUP_MODE:-extractant}
DESCRIPTOR_BLOCKS=${DESCRIPTOR_BLOCKS:-global_shape,coordination_shape}
DESCRIPTOR_PROFILE=${DESCRIPTOR_PROFILE:-core}
SHUFFLE_SEEDS=${SHUFFLE_SEEDS:-1009,2017,3019}
INCLUDE_BLOCK_ABLATIONS=${INCLUDE_BLOCK_ABLATIONS:-1}
DELTA3D_FEATURE_SET=${DELTA3D_FEATURE_SET:-compact-invariant}
REPLICATE_POLICY=${REPLICATE_POLICY:-unique}
INCLUDE_DONOR_COMPOSITION_COUNTS=${INCLUDE_DONOR_COMPOSITION_COUNTS:-1}
TREES=${TREES:-200}
BOOTSTRAP=${BOOTSTRAP:-1000}
AGGREGATE_BOOTSTRAP=${AGGREGATE_BOOTSTRAP:-2000}
AGGREGATE_BOOTSTRAP_SEED=${AGGREGATE_BOOTSTRAP_SEED:-8675309}
OUTER_FOLDS=${OUTER_FOLDS:-5}
INNER_FOLDS=${INNER_FOLDS:-3}
QUICK=${QUICK:-0}
RESUME_COMPLETED=${RESUME_COMPLETED:-1}

if [[ "$RUN_ROOT" != /* ]]; then
  echo "RUN_ROOT must be an absolute path." >&2
  exit 2
fi
RUN_ROOT_CHECK=$RUN_ROOT/
if [[ "$RUN_ROOT_CHECK" == *"/../"* || "$RUN_ROOT_CHECK" == *"/./"* ]]; then
  echo "RUN_ROOT must be normalized without . or .. segments." >&2
  exit 2
fi
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
for value_name in INCLUDE_BLOCK_ABLATIONS INCLUDE_DONOR_COMPOSITION_COUNTS QUICK RESUME_COMPLETED; do
  value=${!value_name}
  if [[ "$value" != 0 && "$value" != 1 ]]; then
    echo "$value_name must be 0 or 1." >&2
    exit 2
  fi
done
for value_name in TREES BOOTSTRAP OUTER_FOLDS INNER_FOLDS; do
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
if [[ ! "$SPLIT_SEED" =~ ^[0-9]+$ ]] || (( 10#$SPLIT_SEED > 4000000000 )); then
  echo "SPLIT_SEED must be an integer in [0, 4000000000]." >&2
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

TASK_INDEX=${SLURM_ARRAY_TASK_ID:-0}
if [[ ! "$TASK_INDEX" =~ ^[0-9]+$ ]] || (( TASK_INDEX >= ${#MODEL_SEED_ARRAY[@]} )); then
  echo "Array task index $TASK_INDEX is outside the five-seed plan." >&2
  exit 2
fi
MODEL_SEED=${MODEL_SEED_ARRAY[$TASK_INDEX]}
RUN_BASENAME=run_${TASK_INDEX}_model_${MODEL_SEED}_split_${SPLIT_SEED}
RUN_DIR=$RUN_ROOT/$RUN_BASENAME
SEED_PLAN=$RUN_ROOT/seed_plan.tsv
EXPERIMENT_CONTRACT=$RUN_ROOT/experiment_contract.tsv

if [[ ! -f "$DATASET_PATH" || ! -f "$VR_ASSET_PATH" ]]; then
  echo "Dataset or VR coordinate asset is missing." >&2
  exit 2
fi
if [[ "$PYTHON_BIN" == */* ]]; then
  [[ -x "$PYTHON_BIN" ]] || { echo "Python is not executable: $PYTHON_BIN" >&2; exit 2; }
else
  command -v "$PYTHON_BIN" >/dev/null || { echo "Python not found: $PYTHON_BIN" >&2; exit 2; }
fi
[[ -f "$PROJECT_DIR/scripts/run_ablation_benchmark.py" ]] || {
  echo "Ablation runner is missing." >&2
  exit 2
}
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
  echo "Dataset SHA-256 mismatch before benchmark execution." >&2
  exit 2
fi
if [[ -n "${EXPECTED_VR_SHA256:-}" && "$EXPECTED_VR_SHA256" != "$VR_ASSET_SHA256" ]]; then
  echo "VR asset SHA-256 mismatch before benchmark execution." >&2
  exit 2
fi
EXPECTED_DATASET_SHA256=$DATASET_SHA256
EXPECTED_VR_SHA256=$VR_ASSET_SHA256
command -v flock >/dev/null || { echo "flock is required for atomic publication." >&2; exit 2; }
[[ -d "$RUN_ROOT" ]] || { echo "RUN_ROOT does not exist: $RUN_ROOT" >&2; exit 2; }
RUN_ROOT=$(cd "$RUN_ROOT" && pwd -P)
RUN_DIR=$RUN_ROOT/$RUN_BASENAME
SEED_PLAN=$RUN_ROOT/seed_plan.tsv
EXPERIMENT_CONTRACT=$RUN_ROOT/experiment_contract.tsv

# Refuse an array whose environment has drifted from the immutable task map.
"$PYTHON_BIN" -c '
import csv, pathlib, sys
path = pathlib.Path(sys.argv[1])
task_index, model_seed, split_seed = map(int, sys.argv[2:5])
if not path.is_file():
    raise SystemExit("Missing immutable seed plan: " + str(path))
with path.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    if reader.fieldnames != ["task_index", "model_seed", "split_seed"]:
        raise SystemExit("Unexpected seed-plan schema")
    rows = list(reader)
if len(rows) != 5:
    raise SystemExit("Seed plan must contain exactly five rows")
indices = [int(row["task_index"]) for row in rows]
models = [int(row["model_seed"]) for row in rows]
splits = [int(row["split_seed"]) for row in rows]
if indices != list(range(5)) or len(set(models)) != 5 or set(splits) != {split_seed}:
    raise SystemExit("Seed plan is not a five-model/one-split plan")
row = rows[task_index]
if int(row["model_seed"]) != model_seed or int(row["split_seed"]) != split_seed:
    raise SystemExit("Array environment disagrees with immutable seed plan")
' "$SEED_PLAN" "$TASK_INDEX" "$MODEL_SEED" "$SPLIT_SEED"

# Refuse task environments that differ in any scientific field from the
# immutable contract created before the first sbatch call.
"$PYTHON_BIN" -c '
import csv, pathlib, sys
path = pathlib.Path(sys.argv[1])
fieldnames = [
    "schema_version", "pair_scope", "group_mode", "split_seed",
    "outer_folds", "inner_folds", "trees", "bootstrap",
    "descriptor_blocks", "descriptor_profile", "shuffle_seeds",
    "include_block_ablations", "quick_mode", "delta3d_feature_set",
    "replicate_policy", "include_donor_composition_counts",
    "aggregate_bootstrap_replicates", "aggregate_bootstrap_seed",
    "dataset_path", "dataset_sha256", "vr_asset_path", "vr_asset_sha256",
]
expected = dict(zip(fieldnames, sys.argv[2:]))
if len(expected) != len(fieldnames) or len(sys.argv[2:]) != len(fieldnames):
    raise SystemExit("Internal experiment-contract argument mismatch")
if not path.is_file():
    raise SystemExit("Missing immutable experiment contract: " + str(path))
with path.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    if reader.fieldnames != fieldnames:
        raise SystemExit("Unexpected experiment-contract schema")
    rows = list(reader)
if len(rows) != 1 or rows[0] != expected:
    raise SystemExit("Array environment disagrees with immutable experiment contract")
' "$EXPERIMENT_CONTRACT" 1.0 "$PAIR_SCOPE" "$GROUP_MODE" "$SPLIT_SEED" \
  "$OUTER_FOLDS" "$INNER_FOLDS" "$TREES" "$BOOTSTRAP" \
  "$DESCRIPTOR_BLOCKS" "$DESCRIPTOR_PROFILE" "$SHUFFLE_SEEDS" \
  "$INCLUDE_BLOCK_ABLATIONS" "$QUICK" "$DELTA3D_FEATURE_SET" \
  "$REPLICATE_POLICY" "$INCLUDE_DONOR_COMPOSITION_COUNTS" \
  "$AGGREGATE_BOOTSTRAP" "$AGGREGATE_BOOTSTRAP_SEED" "$DATASET_PATH" \
  "$DATASET_SHA256" "$VR_ASSET_PATH" "$VR_ASSET_SHA256"

validate_completion() {
  local directory=$1
  local expected_pair_scope=$2
  local expected_group_mode=$3
  local expected_model_seed=$4
  local expected_split_seed=$5
  local experiment_contract=$6
  "$PYTHON_BIN" -c '
import csv, hashlib, json, pathlib, sys
directory = pathlib.Path(sys.argv[1])
expected_scope, expected_group = sys.argv[2:4]
expected_model, expected_split = map(int, sys.argv[4:6])
contract_path = pathlib.Path(sys.argv[6])
summary_path = directory / "summary.json"
success_path = directory / "_SUCCESS.json"
manifest_path = directory / "artifact_hashes.json"
validation_path = directory / "validation.json"
if not all(path.is_file() for path in (summary_path, success_path, manifest_path, validation_path)):
    raise SystemExit(1)
if (directory / "_INCOMPLETE").exists():
    raise SystemExit(1)
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
success = json.loads(success_path.read_text(encoding="utf-8"))
summary = json.loads(summary_path.read_text(encoding="utf-8"))
validation = json.loads(validation_path.read_text(encoding="utf-8"))
manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
declared = manifest_payload.get("sha256")
if success.get("status") != "complete" or summary.get("status") != "complete" or validation.get("passed") is not True:
    raise SystemExit(1)
if success.get("summary_sha256") != digest(summary_path):
    raise SystemExit(1)
if success.get("validation_sha256") != digest(validation_path):
    raise SystemExit(1)
if success.get("artifact_hashes_sha256") != digest(manifest_path):
    raise SystemExit(1)
if not isinstance(declared, dict) or not declared:
    raise SystemExit(1)
for name, expected in declared.items():
    relative = pathlib.Path(name)
    if relative.is_absolute() or ".." in relative.parts or name in {"_SUCCESS.json", "artifact_hashes.json", "_INCOMPLETE"}:
        raise SystemExit(1)
    artifact = directory / relative
    if not artifact.is_file() or digest(artifact) != expected:
        raise SystemExit(1)
observed = {
    path.relative_to(directory).as_posix()
    for path in directory.rglob("*")
    if path.is_file() and path.name not in {"_SUCCESS.json", "artifact_hashes.json"}
}
if observed != set(declared):
    raise SystemExit(1)
if summary.get("pair_scope") != expected_scope or summary.get("group_mode") != expected_group:
    raise SystemExit(1)
if int(summary.get("model_seed", -1)) != expected_model or int(summary.get("split_seed", -1)) != expected_split:
    raise SystemExit(1)
with contract_path.open(newline="", encoding="utf-8") as handle:
    contract_rows = list(csv.DictReader(handle, delimiter="\t"))
if len(contract_rows) != 1:
    raise SystemExit(1)
contract = contract_rows[0]
if summary.get("dataset_path") != contract["dataset_path"]:
    raise SystemExit(1)
if summary.get("dataset_sha256") != contract["dataset_sha256"]:
    raise SystemExit(1)
if summary.get("vr_asset_path") != contract["vr_asset_path"]:
    raise SystemExit(1)
if summary.get("vr_asset_sha256") != contract["vr_asset_sha256"]:
    raise SystemExit(1)
quick = contract["quick_mode"] == "1"
shuffle = [] if contract["shuffle_seeds"] in {"none", "off"} else [
    int(value) for value in contract["shuffle_seeds"].split(",")
]
if quick:
    shuffle = shuffle[:1]
protocol = summary.get("scientific_protocol")
expected_protocol = {
    "pair_scope": contract["pair_scope"],
    "group_mode": contract["group_mode"],
    "split_seed": int(contract["split_seed"]),
    "outer_folds": int(contract["outer_folds"]),
    "inner_folds": int(contract["inner_folds"]),
    "trees": min(int(contract["trees"]), 48) if quick else int(contract["trees"]),
    "bootstrap_replicates": min(int(contract["bootstrap"]), 100) if quick else int(contract["bootstrap"]),
    "geometry_descriptor_blocks": contract["descriptor_blocks"].split(","),
    "descriptor_profile": contract["descriptor_profile"],
    "shuffle_seeds": shuffle,
    "include_block_ablations": contract["include_block_ablations"] == "1",
    "quick_mode": quick,
    "delta3d_feature_set": contract["delta3d_feature_set"],
    "replicate_policy": contract["replicate_policy"],
    "include_donor_composition_counts": contract["include_donor_composition_counts"] == "1",
}
if not isinstance(protocol, dict) or any(protocol.get(key) != value for key, value in expected_protocol.items()):
    raise SystemExit(1)
' "$directory" "$expected_pair_scope" "$expected_group_mode" "$expected_model_seed" "$expected_split_seed" "$experiment_contract"
}

mkdir -p "$RUN_ROOT/status" "$RUN_ROOT/attempts"
if [[ -e "$RUN_DIR" ]]; then
  if validate_completion "$RUN_DIR" "$PAIR_SCOPE" "$GROUP_MODE" "$MODEL_SEED" "$SPLIT_SEED" "$EXPERIMENT_CONTRACT"; then
    if [[ "$RESUME_COMPLETED" == 1 ]]; then
      echo "Validated completed run; skipping: $RUN_DIR"
      exit 0
    fi
    echo "Validated completed run already exists: $RUN_DIR" >&2
  else
    echo "Refusing to overwrite corrupt or incomplete final run: $RUN_DIR" >&2
  fi
  exit 2
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  ATTEMPT_TOKEN=array_${SLURM_ARRAY_JOB_ID:-none}_job_${SLURM_JOB_ID}_task_${TASK_INDEX}_restart_${SLURM_RESTART_COUNT:-0}
else
  ATTEMPT_TOKEN=manual_task_${TASK_INDEX}_pid_$$_$(date -u +%Y%m%dT%H%M%SZ)
fi
ATTEMPT_DIR=$RUN_ROOT/attempts/${RUN_BASENAME}__${ATTEMPT_TOKEN}
if [[ -e "$ATTEMPT_DIR" ]]; then
  echo "Attempt directory already exists: $ATTEMPT_DIR" >&2
  exit 2
fi

FAILURE_MARKER=$RUN_ROOT/status/${RUN_BASENAME}.${ATTEMPT_TOKEN}.failed
on_exit() {
  status=$?
  if (( status != 0 )); then
    printf 'exit_code=%s\njob_id=%s\ntask_id=%s\nmodel_seed=%s\nsplit_seed=%s\nattempt_dir=%s\n' \
      "$status" "${SLURM_JOB_ID:-manual}" "$TASK_INDEX" "$MODEL_SEED" "$SPLIT_SEED" "$ATTEMPT_DIR" \
      > "$FAILURE_MARKER"
  fi
}
trap on_exit EXIT

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=$MODEL_SEED
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export JOBLIB_TEMP_FOLDER=${SLURM_TMPDIR:-$RUN_ROOT/tmp}/joblib_${SLURM_JOB_ID:-manual}_${TASK_INDEX}
mkdir -p "$JOBLIB_TEMP_FOLDER"

COMMAND=(
  "$PYTHON_BIN" -u "$PROJECT_DIR/scripts/run_ablation_benchmark.py"
  --dataset "$DATASET_PATH"
  --vr-assets "$VR_ASSET_PATH"
  --output-dir "$ATTEMPT_DIR"
  --pair-scope "$PAIR_SCOPE"
  --group-mode "$GROUP_MODE"
  --model-seed "$MODEL_SEED"
  --split-seed "$SPLIT_SEED"
  --outer-folds "$OUTER_FOLDS"
  --inner-folds "$INNER_FOLDS"
  --trees "$TREES"
  --bootstrap "$BOOTSTRAP"
  --n-jobs "$SLURM_CPUS_PER_TASK"
  --shuffle-seeds "$SHUFFLE_SEEDS"
  --geometry-descriptor-blocks "$DESCRIPTOR_BLOCKS"
  --descriptor-profile "$DESCRIPTOR_PROFILE"
  --delta3d-feature-set "$DELTA3D_FEATURE_SET"
  --replicate-policy "$REPLICATE_POLICY"
)
if [[ "$INCLUDE_BLOCK_ABLATIONS" == 0 ]]; then
  COMMAND+=(--no-block-ablations)
fi
if [[ "$QUICK" == 1 ]]; then
  COMMAND+=(--quick)
fi
if [[ -n "${EXPECTED_DATASET_SHA256:-}" ]]; then
  COMMAND+=(--expected-dataset-sha256 "$EXPECTED_DATASET_SHA256")
fi
if [[ -n "${EXPECTED_VR_SHA256:-}" ]]; then
  COMMAND+=(--expected-vr-sha256 "$EXPECTED_VR_SHA256")
fi

echo "Starting pair_scope=$PAIR_SCOPE group_mode=$GROUP_MODE model_seed=$MODEL_SEED split_seed=$SPLIT_SEED"
"${COMMAND[@]}"
if ! validate_completion "$ATTEMPT_DIR" "$PAIR_SCOPE" "$GROUP_MODE" "$MODEL_SEED" "$SPLIT_SEED" "$EXPERIMENT_CONTRACT"; then
  echo "Runner returned without a valid completion contract: $ATTEMPT_DIR" >&2
  exit 2
fi

exec 9>"$RUN_ROOT/.publish.lock"
flock 9
if [[ -e "$RUN_DIR" ]]; then
  if ! validate_completion "$RUN_DIR" "$PAIR_SCOPE" "$GROUP_MODE" "$MODEL_SEED" "$SPLIT_SEED" "$EXPERIMENT_CONTRACT"; then
    echo "A competing task published an invalid final run: $RUN_DIR" >&2
    exit 2
  fi
  echo "Another retry already published the validated final; preserving this attempt: $ATTEMPT_DIR"
else
  mv -- "$ATTEMPT_DIR" "$RUN_DIR"
fi
flock -u 9

if ! validate_completion "$RUN_DIR" "$PAIR_SCOPE" "$GROUP_MODE" "$MODEL_SEED" "$SPLIT_SEED" "$EXPERIMENT_CONTRACT"; then
  echo "Final run failed post-publication validation: $RUN_DIR" >&2
  exit 2
fi
printf 'complete\nfinal_dir=%s\nmodel_seed=%s\nsplit_seed=%s\nattempt_token=%s\n' \
  "$RUN_DIR" "$MODEL_SEED" "$SPLIT_SEED" "$ATTEMPT_TOKEN" \
  > "$RUN_ROOT/status/${RUN_BASENAME}.${ATTEMPT_TOKEN}.success"
echo "Published validated run: $RUN_DIR"
