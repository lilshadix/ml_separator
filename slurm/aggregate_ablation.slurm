#!/usr/bin/env bash
#SBATCH --job-name=ln-ablation-agg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Keep the OpenMP/cpus-per-task contract next to the scheduler directives so
# cluster submission filters can validate it before accepting the job.
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Fail-closed cross-seed aggregation.  This job is submitted with afterany so
# it records a useful failure when any worker is missing; the Python aggregator
# will only publish a scientific aggregate when all five hashed runs align.

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
MODEL_SEEDS=${MODEL_SEEDS:-"42 7 137 2027 9001"}
SPLIT_SEED=${SPLIT_SEED:-104729}
PAIR_SCOPE=${PAIR_SCOPE:-all}
GROUP_MODE=${GROUP_MODE:-extractant}
DESCRIPTOR_BLOCKS=${DESCRIPTOR_BLOCKS:-global_shape,coordination_shape}
DESCRIPTOR_PROFILE=${DESCRIPTOR_PROFILE:-core}
SHUFFLE_SEEDS=${SHUFFLE_SEEDS:-1009,2017,3019}
INCLUDE_BLOCK_ABLATIONS=${INCLUDE_BLOCK_ABLATIONS:-1}
# Generation-2 switches, mirrored from the worker so the aggregate validates the
# same immutable contract. All default to off.
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
QUICK=${QUICK:-0}
EXPECTED_RUNS=${EXPECTED_RUNS:-5}
AGGREGATE_BOOTSTRAP=${AGGREGATE_BOOTSTRAP:-2000}
AGGREGATE_BOOTSTRAP_SEED=${AGGREGATE_BOOTSTRAP_SEED:-8675309}
RESUME_AGGREGATE=${RESUME_AGGREGATE:-1}

if [[ -z "${RUN_ROOT:-}" || "$RUN_ROOT" != /* ]]; then
  echo "RUN_ROOT must be an absolute path." >&2
  exit 2
fi
[[ -d "$RUN_ROOT" ]] || { echo "RUN_ROOT does not exist: $RUN_ROOT" >&2; exit 2; }
RUN_ROOT=$(cd "$RUN_ROOT" && pwd -P)
if [[ "$PAIR_SCOPE" != all && "$PAIR_SCOPE" != adjacent ]]; then
  echo "PAIR_SCOPE must be all or adjacent." >&2
  exit 2
fi
if [[ "$GROUP_MODE" != extractant ]]; then
  echo "The primary aggregate requires GROUP_MODE=extractant." >&2
  exit 2
fi
if [[ "$EXPECTED_RUNS" != 5 ]]; then
  echo "EXPECTED_RUNS must be 5 for the frozen multi-seed plan." >&2
  exit 2
fi
for value_name in AGGREGATE_BOOTSTRAP AGGREGATE_BOOTSTRAP_SEED; do
  value=${!value_name}
  if [[ ! "$value" =~ ^[0-9]+$ ]] || (( 10#$value > 4000000000 )); then
    echo "$value_name must be an integer in [0, 4000000000]." >&2
    exit 2
  fi
done
if (( AGGREGATE_BOOTSTRAP < 1 )); then
  echo "AGGREGATE_BOOTSTRAP must be positive." >&2
  exit 2
fi
for value_name in \
  INCLUDE_BLOCK_ABLATIONS INCLUDE_DONOR_COMPOSITION_COUNTS QUICK RESUME_AGGREGATE; do
  value=${!value_name}
  if [[ "$value" != 0 && "$value" != 1 ]]; then
    echo "$value_name must be 0 or 1." >&2
    exit 2
  fi
done
if [[ "$PYTHON_BIN" == */* ]]; then
  [[ -x "$PYTHON_BIN" ]] || { echo "Python is not executable: $PYTHON_BIN" >&2; exit 2; }
else
  command -v "$PYTHON_BIN" >/dev/null || { echo "Python not found: $PYTHON_BIN" >&2; exit 2; }
fi
[[ -f "$PROJECT_DIR/scripts/aggregate_ablation_runs.py" ]] || {
  echo "Ablation aggregator is missing." >&2
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
  echo "Dataset SHA-256 mismatch before aggregate validation." >&2
  exit 2
fi
if [[ -n "${EXPECTED_VR_SHA256:-}" && "$EXPECTED_VR_SHA256" != "$VR_ASSET_SHA256" ]]; then
  echo "VR asset SHA-256 mismatch before aggregate validation." >&2
  exit 2
fi
command -v flock >/dev/null || { echo "flock is required for atomic publication." >&2; exit 2; }

read -r -a MODEL_SEED_ARRAY <<< "$MODEL_SEEDS"
if [[ ${#MODEL_SEED_ARRAY[@]} -ne 5 ]]; then
  echo "Exactly five model seeds are required." >&2
  exit 2
fi

SEED_PLAN=$RUN_ROOT/seed_plan.tsv
EXPERIMENT_CONTRACT=$RUN_ROOT/experiment_contract.tsv
"$PYTHON_BIN" -c '
import csv, pathlib, sys
path = pathlib.Path(sys.argv[1])
expected_models = [int(value) for value in sys.argv[2].split()]
expected_split = int(sys.argv[3])
if not path.is_file():
    raise SystemExit("Missing immutable seed plan: " + str(path))
with path.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    if reader.fieldnames != ["task_index", "model_seed", "split_seed"]:
        raise SystemExit("Unexpected seed-plan schema")
    rows = list(reader)
observed_indices = [int(row["task_index"]) for row in rows]
observed_models = [int(row["model_seed"]) for row in rows]
observed_splits = [int(row["split_seed"]) for row in rows]
if observed_indices != list(range(5)) or observed_models != expected_models:
    raise SystemExit("Seed plan disagrees with expected model-seed order")
if len(set(observed_models)) != 5 or set(observed_splits) != {expected_split}:
    raise SystemExit("Seed plan is not a five-model/one-split plan")
' "$SEED_PLAN" "$MODEL_SEEDS" "$SPLIT_SEED"

# The aggregate environment must reproduce the exact scientific plan frozen
# before the worker array was submitted.
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
    "include_pair_response_3d", "include_electronic", "include_extension_arms",
    "include_symmetric_3d", "include_reference_baselines",
    "include_2d_sensitivity",
    "extension_shuffle_seeds", "prespecified_arms",
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
    raise SystemExit("Aggregate environment disagrees with immutable experiment contract")
' "$EXPERIMENT_CONTRACT" 2.0 "$PAIR_SCOPE" "$GROUP_MODE" "$SPLIT_SEED" \
  "$OUTER_FOLDS" "$INNER_FOLDS" "$TREES" "$BOOTSTRAP" \
  "$DESCRIPTOR_BLOCKS" "$DESCRIPTOR_PROFILE" "$SHUFFLE_SEEDS" \
  "$INCLUDE_BLOCK_ABLATIONS" "$QUICK" "$DELTA3D_FEATURE_SET" \
  "$REPLICATE_POLICY" "$INCLUDE_DONOR_COMPOSITION_COUNTS" \
  "$AGGREGATE_BOOTSTRAP" "$AGGREGATE_BOOTSTRAP_SEED" "$DATASET_PATH" \
  "$DATASET_SHA256" "$VR_ASSET_PATH" "$VR_ASSET_SHA256" \
  "$INCLUDE_PAIR_RESPONSE_3D" "$INCLUDE_ELECTRONIC" "$INCLUDE_EXTENSION_ARMS" \
  "$INCLUDE_SYMMETRIC_3D" "$INCLUDE_REFERENCE_BASELINES" \
  "$INCLUDE_2D_SENSITIVITY" \
  "${EXTENSION_SHUFFLE_SEEDS:-none}" "${PRESPECIFIED_ARMS:-A0-A6}"

# Do not rely only on cross-run equality: all five summaries must also match
# the immutable root contract.  This catches a coherent but wrong set copied
# into the run root before either a fresh aggregation or aggregate resume.
"$PYTHON_BIN" -c '
import csv, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
contract_path = pathlib.Path(sys.argv[2])
expected_models = sorted(int(value) for value in sys.argv[3].split())
with contract_path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
if len(rows) != 1:
    raise SystemExit("Experiment contract must contain exactly one row")
contract = rows[0]
quick = contract["quick_mode"] == "1"
shuffle = [] if contract["shuffle_seeds"] in {"none", "off"} else [
    int(value) for value in contract["shuffle_seeds"].split(",")
]
if quick:
    shuffle = shuffle[:1]
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
run_dirs = sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("run_"))
if len(run_dirs) != len(expected_models):
    raise SystemExit(f"Expected {len(expected_models)} published run directories; observed {len(run_dirs)}")
observed_models = []
for run_dir in run_dirs:
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        raise SystemExit("Missing run summary: " + str(summary_path))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    protocol = summary.get("scientific_protocol")
    if not isinstance(protocol, dict) or any(protocol.get(key) != value for key, value in expected_protocol.items()):
        raise SystemExit("Run scientific protocol disagrees with immutable experiment contract: " + run_dir.name)
    if summary.get("pair_scope") != contract["pair_scope"] or summary.get("group_mode") != contract["group_mode"]:
        raise SystemExit("Run scope/group disagrees with immutable experiment contract: " + run_dir.name)
    if int(summary.get("split_seed", -1)) != int(contract["split_seed"]):
        raise SystemExit("Run split seed disagrees with immutable experiment contract: " + run_dir.name)
    if summary.get("dataset_path") != contract["dataset_path"]:
        raise SystemExit("Run dataset path disagrees with immutable experiment contract: " + run_dir.name)
    if summary.get("dataset_sha256") != contract["dataset_sha256"]:
        raise SystemExit("Run dataset hash disagrees with immutable experiment contract: " + run_dir.name)
    if summary.get("vr_asset_path") != contract["vr_asset_path"]:
        raise SystemExit("Run VR asset path disagrees with immutable experiment contract: " + run_dir.name)
    if summary.get("vr_asset_sha256") != contract["vr_asset_sha256"]:
        raise SystemExit("Run VR asset hash disagrees with immutable experiment contract: " + run_dir.name)
    observed_models.append(int(summary.get("model_seed", -1)))
if sorted(observed_models) != expected_models or len(set(observed_models)) != len(expected_models):
    raise SystemExit("Published runs disagree with immutable model-seed plan")
' "$RUN_ROOT" "$EXPERIMENT_CONTRACT" "$MODEL_SEEDS"

validate_aggregate() {
  local directory=$1
  local expected_pair_scope=$2
  local expected_runs=$3
  local expected_split=$4
  local expected_models=$5
  local experiment_contract=$6
  local aggregate_implementation=$7
  "$PYTHON_BIN" -c '
import csv, hashlib, json, pathlib, sys
directory = pathlib.Path(sys.argv[1])
expected_scope = sys.argv[2]
expected_runs, expected_split = map(int, sys.argv[3:5])
expected_models = sorted(int(value) for value in sys.argv[5].split())
contract_path = pathlib.Path(sys.argv[6])
implementation_path = pathlib.Path(sys.argv[7])
success_path = directory / "_SUCCESS.json"
summary_path = directory / "aggregate_summary.json"
validation_path = directory / "validation.json"
if not all(path.is_file() for path in (success_path, summary_path, validation_path)):
    raise SystemExit(1)
if (directory / "_INCOMPLETE").exists():
    raise SystemExit(1)
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
success = json.loads(success_path.read_text(encoding="utf-8"))
summary = json.loads(summary_path.read_text(encoding="utf-8"))
validation = json.loads(validation_path.read_text(encoding="utf-8"))
if success.get("status") != "complete" or validation.get("passed") is not True:
    raise SystemExit(1)
if success.get("pair_scope") != expected_scope or summary.get("pair_scope") != expected_scope:
    raise SystemExit(1)
if success.get("aggregate_summary_sha256") != digest(summary_path):
    raise SystemExit(1)
declared = success.get("artifact_sha256")
if not isinstance(declared, dict) or not declared:
    raise SystemExit(1)
for name, expected in declared.items():
    relative = pathlib.Path(name)
    if relative.is_absolute() or ".." in relative.parts or name == "_SUCCESS.json":
        raise SystemExit(1)
    artifact = directory / relative
    if not artifact.is_file() or digest(artifact) != expected:
        raise SystemExit(1)
observed = {
    path.relative_to(directory).as_posix()
    for path in directory.rglob("*")
    if path.is_file() and path.name != "_SUCCESS.json"
}
if observed != set(declared):
    raise SystemExit(1)
if int(summary.get("run_count", -1)) != expected_runs:
    raise SystemExit(1)
if int(summary.get("split_seed", -1)) != expected_split:
    raise SystemExit(1)
if sorted(int(value) for value in summary.get("model_seeds", [])) != expected_models:
    raise SystemExit(1)
fingerprints = validation.get("fingerprints", {})
if fingerprints.get("group_mode") != ["extractant"]:
    raise SystemExit(1)
with contract_path.open(newline="", encoding="utf-8") as handle:
    contract_rows = list(csv.DictReader(handle, delimiter="\t"))
if len(contract_rows) != 1:
    raise SystemExit(1)
contract = contract_rows[0]
if fingerprints.get("dataset_sha256") != [contract["dataset_sha256"]]:
    raise SystemExit(1)
if fingerprints.get("vr_asset_sha256") != [contract["vr_asset_sha256"]]:
    raise SystemExit(1)
bootstrap_replicates = int(contract["aggregate_bootstrap_replicates"])
bootstrap_seed = int(contract["aggregate_bootstrap_seed"])
implementation_sha256 = digest(implementation_path)
for payload in (success, validation):
    if payload.get("aggregate_implementation_sha256") != implementation_sha256:
        raise SystemExit(1)
    if int(payload.get("bootstrap_replicates", -1)) != bootstrap_replicates:
        raise SystemExit(1)
    if int(payload.get("bootstrap_seed", -1)) != bootstrap_seed:
        raise SystemExit(1)
primary_comparisons = ("A2_vs_A5", "A2_vs_A6", "A3_vs_A2")
paired = summary.get("paired_comparisons")
if not isinstance(paired, dict):
    raise SystemExit(1)
for comparison_index, comparison in enumerate(primary_comparisons):
    item = paired.get(comparison)
    bootstrap = item.get("paired_extractant_bootstrap") if isinstance(item, dict) else None
    if not isinstance(bootstrap, dict):
        raise SystemExit(1)
    if int(bootstrap.get("replicates", -1)) != bootstrap_replicates:
        raise SystemExit(1)
    if int(bootstrap.get("seed", -1)) != bootstrap_seed + comparison_index * 1009:
        raise SystemExit(1)
' "$directory" "$expected_pair_scope" "$expected_runs" "$expected_split" "$expected_models" \
  "$experiment_contract" "$aggregate_implementation"
}

AGGREGATE_DIR=$RUN_ROOT/aggregate
if [[ -e "$AGGREGATE_DIR" ]]; then
  if validate_aggregate "$AGGREGATE_DIR" "$PAIR_SCOPE" "$EXPECTED_RUNS" "$SPLIT_SEED" "$MODEL_SEEDS" "$EXPERIMENT_CONTRACT" "$PROJECT_DIR/scripts/aggregate_ablation_runs.py"; then
    if [[ "$RESUME_AGGREGATE" == 1 ]]; then
      echo "Validated aggregate already exists; skipping: $AGGREGATE_DIR"
      exit 0
    fi
    echo "Validated aggregate already exists and RESUME_AGGREGATE=0: $AGGREGATE_DIR" >&2
  else
    echo "Refusing to overwrite corrupt or incomplete aggregate: $AGGREGATE_DIR" >&2
  fi
  exit 2
fi

mkdir -p "$RUN_ROOT/aggregate_attempts" "$RUN_ROOT/status"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  ATTEMPT_TOKEN=job_${SLURM_JOB_ID}_restart_${SLURM_RESTART_COUNT:-0}
else
  ATTEMPT_TOKEN=manual_pid_$$_$(date -u +%Y%m%dT%H%M%SZ)
fi
ATTEMPT_DIR=$RUN_ROOT/aggregate_attempts/aggregate__${ATTEMPT_TOKEN}
if [[ -e "$ATTEMPT_DIR" ]]; then
  echo "Aggregate attempt already exists: $ATTEMPT_DIR" >&2
  exit 2
fi

FAILURE_MARKER=$RUN_ROOT/status/aggregate.${ATTEMPT_TOKEN}.failed
on_exit() {
  status=$?
  if (( status != 0 )); then
    printf 'exit_code=%s\njob_id=%s\nattempt_dir=%s\n' \
      "$status" "${SLURM_JOB_ID:-manual}" "$ATTEMPT_DIR" > "$FAILURE_MARKER"
  fi
}
trap on_exit EXIT

"$PYTHON_BIN" -u "$PROJECT_DIR/scripts/aggregate_ablation_runs.py" \
  --run-root "$RUN_ROOT" \
  --output-dir "$ATTEMPT_DIR" \
  --expected-runs "$EXPECTED_RUNS" \
  --expected-pair-scope "$PAIR_SCOPE" \
  --bootstrap-replicates "$AGGREGATE_BOOTSTRAP" \
  --bootstrap-seed "$AGGREGATE_BOOTSTRAP_SEED"

if ! validate_aggregate "$ATTEMPT_DIR" "$PAIR_SCOPE" "$EXPECTED_RUNS" "$SPLIT_SEED" "$MODEL_SEEDS" "$EXPERIMENT_CONTRACT" "$PROJECT_DIR/scripts/aggregate_ablation_runs.py"; then
  echo "Aggregator returned without a valid completion contract: $ATTEMPT_DIR" >&2
  exit 2
fi

exec 9>"$RUN_ROOT/.aggregate_publish.lock"
flock 9
if [[ -e "$AGGREGATE_DIR" ]]; then
  if ! validate_aggregate "$AGGREGATE_DIR" "$PAIR_SCOPE" "$EXPECTED_RUNS" "$SPLIT_SEED" "$MODEL_SEEDS" "$EXPERIMENT_CONTRACT" "$PROJECT_DIR/scripts/aggregate_ablation_runs.py"; then
    echo "A competing aggregate publication is invalid: $AGGREGATE_DIR" >&2
    exit 2
  fi
  echo "Another retry already published the validated aggregate; preserving $ATTEMPT_DIR"
else
  mv -- "$ATTEMPT_DIR" "$AGGREGATE_DIR"
fi
flock -u 9

if ! validate_aggregate "$AGGREGATE_DIR" "$PAIR_SCOPE" "$EXPECTED_RUNS" "$SPLIT_SEED" "$MODEL_SEEDS" "$EXPERIMENT_CONTRACT" "$PROJECT_DIR/scripts/aggregate_ablation_runs.py"; then
  echo "Published aggregate failed validation: $AGGREGATE_DIR" >&2
  exit 2
fi
printf 'complete\naggregate_dir=%s\nattempt_token=%s\n' "$AGGREGATE_DIR" "$ATTEMPT_TOKEN" \
  > "$RUN_ROOT/status/aggregate.${ATTEMPT_TOKEN}.success"
echo "Published validated aggregate: $AGGREGATE_DIR"
