#!/usr/bin/env bash
#SBATCH --job-name=ln-descriptor
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --array=0-9%2
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

# Metal-site descriptor evaluation.
#
# The array runs two arms over the same prespecified seed plan:
#
#   real  descriptors computed from the geometry they belong to
#   null  descriptors detached from their geometry by a seeded permutation
#
# The null arm is not optional. Adding continuous columns of any kind to a
# mostly-binary fingerprint matrix changes how ExtraTrees samples split
# candidates, so a real-arm gain only counts as chemistry if it exceeds the
# null arm. Task index maps to arm-major order: tasks [0, N) are the real arm
# and tasks [N, 2N) are the null arm.

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
RUN_ROOT=${RUN_ROOT:-$PROJECT_DIR/runs/descriptor_${SLURM_ARRAY_JOB_ID:-manual}}
MODEL_SEEDS=${MODEL_SEEDS:-"42 7 137 2027 9001"}
SPLIT_SEEDS=${SPLIT_SEEDS:-"104729 130363 169087 214087 275015"}
DESCRIPTOR_BLOCKS=${DESCRIPTOR_BLOCKS:-"ligand_field,enclosure"}
DESCRIPTOR_PROFILE=${DESCRIPTOR_PROFILE:-core}
NULL_PERMUTATION=${NULL_PERMUTATION:-metal-preserving}
NULL_PERMUTATION_BASE_SEED=${NULL_PERMUTATION_BASE_SEED:-8100011}
TREES=${TREES:-200}
BOOTSTRAP=${BOOTSTRAP:-2000}
OUTER_FOLDS=${OUTER_FOLDS:-5}
INNER_FOLDS=${INNER_FOLDS:-3}
RESUME_COMPLETED=${RESUME_COMPLETED:-0}

if [[ "$RUN_ROOT" != /* ]]; then
  echo "RUN_ROOT must be an absolute path." >&2
  exit 2
fi
mkdir -p "$RUN_ROOT"
RUN_ROOT=$(cd "$RUN_ROOT" && pwd -P)
if [[ "$RESUME_COMPLETED" != 0 && "$RESUME_COMPLETED" != 1 ]]; then
  echo "RESUME_COMPLETED must be 0 or 1." >&2
  exit 2
fi
if [[ "$DESCRIPTOR_PROFILE" != core && "$DESCRIPTOR_PROFILE" != full ]]; then
  echo "DESCRIPTOR_PROFILE must be core or full." >&2
  exit 2
fi
if [[ "$NULL_PERMUTATION" != free && "$NULL_PERMUTATION" != metal-preserving ]]; then
  echo "NULL_PERMUTATION must be free or metal-preserving." >&2
  exit 2
fi
if [[ ! "$NULL_PERMUTATION_BASE_SEED" =~ ^[0-9]+$ ]]; then
  echo "NULL_PERMUTATION_BASE_SEED must be a nonnegative integer." >&2
  exit 2
fi

read -r -a MODEL_SEED_ARRAY <<< "$MODEL_SEEDS"
read -r -a SPLIT_SEED_ARRAY <<< "$SPLIT_SEEDS"
if [[ ${#MODEL_SEED_ARRAY[@]} -ne ${#SPLIT_SEED_ARRAY[@]} ]]; then
  echo "MODEL_SEEDS and SPLIT_SEEDS must have equal length." >&2
  exit 2
fi
for seed in "${MODEL_SEED_ARRAY[@]}" "${SPLIT_SEED_ARRAY[@]}"; do
  if [[ ! "$seed" =~ ^[0-9]+$ ]] || (( 10#$seed > 4000000000 )); then
    echo "Seeds must be integers in [0, 4000000000]: $seed" >&2
    exit 2
  fi
done

SEED_COUNT=${#MODEL_SEED_ARRAY[@]}
TASK_INDEX=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_INDEX < 0 || TASK_INDEX >= 2 * SEED_COUNT )); then
  echo "Array task index $TASK_INDEX is outside the arm/seed map." >&2
  exit 2
fi
if (( TASK_INDEX < SEED_COUNT )); then
  ARM=real
  SEED_INDEX=$TASK_INDEX
  PERMUTATION=none
  PERMUTATION_SEED=0
else
  ARM=null
  SEED_INDEX=$((TASK_INDEX - SEED_COUNT))
  PERMUTATION=$NULL_PERMUTATION
  PERMUTATION_SEED=$((NULL_PERMUTATION_BASE_SEED + SEED_INDEX))
fi
MODEL_SEED=${MODEL_SEED_ARRAY[$SEED_INDEX]}
SPLIT_SEED=${SPLIT_SEED_ARRAY[$SEED_INDEX]}
ARM_ROOT=$RUN_ROOT/arm_$ARM
RUN_BASENAME=run_${SEED_INDEX}_model_${MODEL_SEED}_split_${SPLIT_SEED}
RUN_DIR=$ARM_ROOT/$RUN_BASENAME

if [[ ! -f "$DATASET_PATH" || ! -f "$VR_ASSET_PATH" ]]; then
  echo "Dataset or coordinate asset is missing." >&2
  exit 2
fi
if [[ "$PYTHON_BIN" == */* ]]; then
  [[ -x "$PYTHON_BIN" ]] || { echo "Python is not executable: $PYTHON_BIN" >&2; exit 2; }
else
  command -v "$PYTHON_BIN" >/dev/null || { echo "Python not found: $PYTHON_BIN" >&2; exit 2; }
fi
command -v flock >/dev/null || { echo "flock is required for atomic publication." >&2; exit 2; }

validate_completion() {
  local directory=$1
  "$PYTHON_BIN" -c '
import hashlib, json, pathlib, sys
directory = pathlib.Path(sys.argv[1])
summary = directory / "summary.json"
marker = directory / "_SUCCESS.json"
if not summary.is_file() or not marker.is_file() or (directory / "_INCOMPLETE").exists():
    raise SystemExit(1)
payload = json.loads(marker.read_text(encoding="utf-8"))
digest = hashlib.sha256(summary.read_bytes()).hexdigest()
summary_payload = json.loads(summary.read_text(encoding="utf-8"))
artifact_hashes = summary_payload.get("artifacts", {}).get("sha256", {})
if not isinstance(artifact_hashes, dict) or not artifact_hashes:
    raise SystemExit(1)
for name, expected in artifact_hashes.items():
    relative = pathlib.Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise SystemExit(1)
    artifact = directory / relative
    if not artifact.is_file() or hashlib.sha256(artifact.read_bytes()).hexdigest() != expected:
        raise SystemExit(1)
raise SystemExit(0 if payload.get("status") == "complete" and payload.get("summary_sha256") == digest else 1)
' "$directory"
}

mkdir -p "$ARM_ROOT" "$RUN_ROOT/status" "$RUN_ROOT/attempts"
if [[ -e "$RUN_DIR" ]]; then
  if validate_completion "$RUN_DIR"; then
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
# Attempts live beside the arm directories, not inside them: the aggregator
# discovers runs by walking an arm root, and a preserved attempt carrying a valid
# summary.json would otherwise be pooled as if it were a published run. Both
# paths are under RUN_ROOT, so the publishing rename stays atomic.
ATTEMPT_DIR=$RUN_ROOT/attempts/${ARM}_${RUN_BASENAME}__${ATTEMPT_TOKEN}
if [[ -e "$ATTEMPT_DIR" ]]; then
  echo "Attempt directory already exists: $ATTEMPT_DIR" >&2
  exit 2
fi

failure_marker=$RUN_ROOT/status/${ARM}_${RUN_BASENAME}.${ATTEMPT_TOKEN}.failed
on_exit() {
  status=$?
  if (( status != 0 )); then
    printf 'exit_code=%s\njob_id=%s\ntask_id=%s\narm=%s\nattempt_dir=%s\n' \
      "$status" "${SLURM_JOB_ID:-manual}" "$TASK_INDEX" "$ARM" "$ATTEMPT_DIR" > "$failure_marker"
  fi
}
trap on_exit EXIT

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=$MODEL_SEED
# Some clusters reject --cpus-per-task unless this exact OpenMP contract is
# present in the batch file. SLURM always defines it for submitted jobs; the
# fallback keeps deliberate direct/manual execution well-defined.
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export JOBLIB_TEMP_FOLDER=${SLURM_TMPDIR:-$RUN_ROOT/tmp}/joblib_${SLURM_JOB_ID:-manual}_${TASK_INDEX}
mkdir -p "$JOBLIB_TEMP_FOLDER"

command=(
  "$PYTHON_BIN" -u "$PROJECT_DIR/scripts/run_delta3d_benchmark.py"
  --dataset "$DATASET_PATH"
  --vr-assets "$VR_ASSET_PATH"
  --output-dir "$ATTEMPT_DIR"
  --seed "$MODEL_SEED"
  --split-seed "$SPLIT_SEED"
  --outer-folds "$OUTER_FOLDS"
  --inner-folds "$INNER_FOLDS"
  --trees "$TREES"
  --bootstrap "$BOOTSTRAP"
  --n-jobs "$SLURM_CPUS_PER_TASK"
  --geometry-descriptor-blocks "$DESCRIPTOR_BLOCKS"
  --descriptor-profile "$DESCRIPTOR_PROFILE"
  --descriptor-permutation "$PERMUTATION"
  --descriptor-permutation-seed "$PERMUTATION_SEED"
  --delta3d-feature-set compact-invariant
  --group-mode ecfp-exact-cluster
  --replicate-policy unique
  --evaluation-only
)
if [[ -n "${EXPECTED_DATASET_SHA256:-}" ]]; then
  command+=(--expected-dataset-sha256 "$EXPECTED_DATASET_SHA256")
fi
if [[ -n "${EXPECTED_VR_SHA256:-}" ]]; then
  command+=(--expected-vr-sha256 "$EXPECTED_VR_SHA256")
fi

echo "Starting arm=$ARM model_seed=$MODEL_SEED split_seed=$SPLIT_SEED permutation=$PERMUTATION"
"${command[@]}"
if ! validate_completion "$ATTEMPT_DIR"; then
  echo "Runner returned without a valid completion contract: $ATTEMPT_DIR" >&2
  exit 2
fi

# Attempts and finals share ARM_ROOT, so rename is atomic. The lock closes the
# retry race; mv -T prevents accidentally nesting an attempt under a final.
exec 9>"$RUN_ROOT/.publish.lock"
flock 9
if [[ -e "$RUN_DIR" ]]; then
  if ! validate_completion "$RUN_DIR"; then
    echo "A competing task published an invalid final run: $RUN_DIR" >&2
    exit 2
  fi
  echo "Another retry already published the validated final; preserving this attempt."
else
  mv -T -- "$ATTEMPT_DIR" "$RUN_DIR"
fi
flock -u 9

if ! validate_completion "$RUN_DIR"; then
  echo "Final run failed post-publication validation: $RUN_DIR" >&2
  exit 2
fi
printf 'complete\narm=%s\nfinal_dir=%s\nattempt_token=%s\n' "$ARM" "$RUN_DIR" "$ATTEMPT_TOKEN" \
  > "$RUN_ROOT/status/${ARM}_${RUN_BASENAME}.${ATTEMPT_TOKEN}.success"
echo "Published validated run: $RUN_DIR"
