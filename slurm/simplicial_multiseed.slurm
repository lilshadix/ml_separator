#!/usr/bin/env bash
#SBATCH --job-name=ln-simplicial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-4%1
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

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
ROW_MAP_PATH=${ROW_MAP_PATH:-$PROJECT_DIR/dataset with 3D structures/row_geometry_map.csv}
RUN_ROOT=${RUN_ROOT:-$PROJECT_DIR/runs/simplicial_${SLURM_ARRAY_JOB_ID:-manual}}
MODEL_SEEDS=${MODEL_SEEDS:-"42 7 137 2027 9001"}
SPLIT_SEEDS=${SPLIT_SEEDS:-"104729 130363 169087 214087 275015"}
PAIR_SCOPE=${PAIR_SCOPE:-all}
TREES=${TREES:-200}
BOOTSTRAP=${BOOTSTRAP:-2000}
OUTER_FOLDS=${OUTER_FOLDS:-5}
INNER_FOLDS=${INNER_FOLDS:-3}
EPOCHS=${EPOCHS:-160}
INITIALIZATIONS=${INITIALIZATIONS:-3}
HIDDEN_DIM=${HIDDEN_DIM:-64}
LAYERS=${LAYERS:-2}
MAX_PAIRS_PER_BATCH=${MAX_PAIRS_PER_BATCH:-64}
MAX_SIMPLICES_PER_BATCH=${MAX_SIMPLICES_PER_BATCH:-60000}
DEVICE=${ACCELERATOR:-cpu}
DETERMINISM=${DETERMINISM:-strict}
RESUME_COMPLETED=${RESUME_COMPLETED:-0}
STAGE_VR_TO_TMP=${STAGE_VR_TO_TMP:-0}

if [[ "$RUN_ROOT" != /* ]]; then
  echo "RUN_ROOT must be an absolute path." >&2
  exit 2
fi
mkdir -p "$RUN_ROOT"
RUN_ROOT=$(cd "$RUN_ROOT" && pwd -P)
for value_name in RESUME_COMPLETED STAGE_VR_TO_TMP; do
  value=${!value_name}
  if [[ "$value" != 0 && "$value" != 1 ]]; then
    echo "$value_name must be 0 or 1." >&2
    exit 2
  fi
done
if [[ "$PAIR_SCOPE" != all && "$PAIR_SCOPE" != adjacent ]]; then
  echo "PAIR_SCOPE must be all or adjacent." >&2
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
TASK_INDEX=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_INDEX < 0 || TASK_INDEX >= ${#MODEL_SEED_ARRAY[@]} )); then
  echo "Array task index $TASK_INDEX is outside the seed map." >&2
  exit 2
fi
MODEL_SEED=${MODEL_SEED_ARRAY[$TASK_INDEX]}
SPLIT_SEED=${SPLIT_SEED_ARRAY[$TASK_INDEX]}
RUN_BASENAME=run_${TASK_INDEX}_model_${MODEL_SEED}_split_${SPLIT_SEED}
RUN_DIR=$RUN_ROOT/$RUN_BASENAME

if [[ ! -f "$DATASET_PATH" || ! -f "$VR_ASSET_PATH" || ! -f "$ROW_MAP_PATH" ]]; then
  echo "Dataset, VR asset, or row map is missing." >&2
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
  local expected_pair_scope=$2
  "$PYTHON_BIN" -c '
import hashlib, json, pathlib, sys
directory = pathlib.Path(sys.argv[1])
expected_pair_scope = sys.argv[2]
summary = directory / "summary.json"
marker = directory / "_SUCCESS.json"
if not summary.is_file() or not marker.is_file() or (directory / "_INCOMPLETE").exists():
    raise SystemExit(1)
payload = json.loads(marker.read_text(encoding="utf-8"))
digest = hashlib.sha256(summary.read_bytes()).hexdigest()
summary_payload = json.loads(summary.read_text(encoding="utf-8"))
if summary_payload.get("arguments", {}).get("pair_scope") != expected_pair_scope:
    raise SystemExit(1)
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
' "$directory" "$expected_pair_scope"
}

mkdir -p "$RUN_ROOT/status" "$RUN_ROOT/attempts"
if [[ -e "$RUN_DIR" ]]; then
  if validate_completion "$RUN_DIR" "$PAIR_SCOPE"; then
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

failure_marker=$RUN_ROOT/status/${RUN_BASENAME}.${ATTEMPT_TOKEN}.failed
on_exit() {
  status=$?
  if (( status != 0 )); then
    printf 'exit_code=%s\njob_id=%s\ntask_id=%s\nattempt_dir=%s\n' \
      "$status" "${SLURM_JOB_ID:-manual}" "$TASK_INDEX" "$ATTEMPT_DIR" > "$failure_marker"
  fi
}
trap on_exit EXIT

if [[ "$STAGE_VR_TO_TMP" == 1 ]]; then
  if [[ -z "${SLURM_TMPDIR:-}" ]]; then
    echo "STAGE_VR_TO_TMP=1 requires SLURM_TMPDIR." >&2
    exit 2
  fi
  staged_vr=$SLURM_TMPDIR/vietoris_rips_inputs_${SLURM_JOB_ID:-manual}_${TASK_INDEX}.npz
  cp "$VR_ASSET_PATH" "$staged_vr"
  VR_ASSET_PATH=$staged_vr
fi

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
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export JOBLIB_TEMP_FOLDER=${SLURM_TMPDIR:-$RUN_ROOT/tmp}/joblib_${SLURM_JOB_ID:-manual}_${TASK_INDEX}
mkdir -p "$JOBLIB_TEMP_FOLDER"

command=(
  "$PYTHON_BIN" -u "$PROJECT_DIR/scripts/run_simplicial_benchmark.py"
  --dataset "$DATASET_PATH"
  --vr-assets "$VR_ASSET_PATH"
  --row-geometry-map "$ROW_MAP_PATH"
  --output-dir "$ATTEMPT_DIR"
  --logical-output-dir "$RUN_DIR"
  --pair-scope "$PAIR_SCOPE"
  --seed "$MODEL_SEED"
  --split-seed "$SPLIT_SEED"
  --outer-folds "$OUTER_FOLDS"
  --inner-folds "$INNER_FOLDS"
  --trees "$TREES"
  --bootstrap "$BOOTSTRAP"
  --n-jobs "$SLURM_CPUS_PER_TASK"
  --torch-threads "$SLURM_CPUS_PER_TASK"
  --device "$DEVICE"
  --determinism "$DETERMINISM"
  --epochs "$EPOCHS"
  --initializations "$INITIALIZATIONS"
  --hidden-dim "$HIDDEN_DIM"
  --layers "$LAYERS"
  --max-pairs-per-batch "$MAX_PAIRS_PER_BATCH"
  --max-simplices-per-batch "$MAX_SIMPLICES_PER_BATCH"
  --shell-mode coordination
  --radius-angstrom 3.10
  --max-filtration-angstrom 4.00
  --max-edges-per-complex 512
  --max-triangles-per-complex 512
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

echo "Starting simplicial run pair_scope=$PAIR_SCOPE model_seed=$MODEL_SEED split_seed=$SPLIT_SEED"
"${command[@]}"
if ! validate_completion "$ATTEMPT_DIR" "$PAIR_SCOPE"; then
  echo "Runner returned without a valid completion contract: $ATTEMPT_DIR" >&2
  exit 2
fi

# Attempts and finals share RUN_ROOT, so rename is atomic. The lock closes the
# retry race; mv -T prevents accidentally nesting an attempt under a final.
exec 9>"$RUN_ROOT/.publish.lock"
flock 9
if [[ -e "$RUN_DIR" ]]; then
  if ! validate_completion "$RUN_DIR" "$PAIR_SCOPE"; then
    echo "A competing task published an invalid final run: $RUN_DIR" >&2
    exit 2
  fi
  echo "Another retry already published the validated final; preserving this attempt."
else
  mv -T -- "$ATTEMPT_DIR" "$RUN_DIR"
fi
flock -u 9

if ! validate_completion "$RUN_DIR" "$PAIR_SCOPE"; then
  echo "Final run failed post-publication validation: $RUN_DIR" >&2
  exit 2
fi
printf 'complete\nfinal_dir=%s\nattempt_token=%s\n' "$RUN_DIR" "$ATTEMPT_TOKEN" \
  > "$RUN_ROOT/status/${RUN_BASENAME}.${ATTEMPT_TOKEN}.success"
echo "Published validated run: $RUN_DIR"
