#!/usr/bin/env bash
#SBATCH --job-name=ln-simp-aggregate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -Eeuo pipefail
umask 027

PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$PROJECT_DIR" ]]; then
  echo "PROJECT_DIR is required." >&2
  exit 2
fi
PROJECT_DIR=$(cd "$PROJECT_DIR" && pwd -P)
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT is required}
EXPECTED_RUNS=${EXPECTED_RUNS:?EXPECTED_RUNS is required}
RESUME_AGGREGATE=${RESUME_AGGREGATE:-1}

# Required by clusters that enforce an explicit OpenMP/cpus-per-task contract.
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

if [[ "$RUN_ROOT" != /* ]]; then
  echo "RUN_ROOT must be absolute." >&2
  exit 2
fi
[[ -d "$RUN_ROOT" ]] || { echo "Run root not found: $RUN_ROOT" >&2; exit 2; }
RUN_ROOT=$(cd "$RUN_ROOT" && pwd -P)
FINAL_DIR=$RUN_ROOT/aggregate
if [[ "$PYTHON_BIN" == */* ]]; then
  [[ -x "$PYTHON_BIN" ]] || { echo "Python is not executable: $PYTHON_BIN" >&2; exit 2; }
else
  command -v "$PYTHON_BIN" >/dev/null || { echo "Python not found: $PYTHON_BIN" >&2; exit 2; }
fi
command -v flock >/dev/null || { echo "flock is required for atomic publication." >&2; exit 2; }
if [[ "$RESUME_AGGREGATE" != 0 && "$RESUME_AGGREGATE" != 1 ]]; then
  echo "RESUME_AGGREGATE must be 0 or 1." >&2
  exit 2
fi

validate_aggregate() {
  local directory=$1
  "$PYTHON_BIN" -c '
import hashlib, json, pathlib, sys
directory = pathlib.Path(sys.argv[1])
summary = directory / "aggregate_summary.json"
marker = directory / "_SUCCESS.json"
if not summary.is_file() or not marker.is_file() or (directory / "_INCOMPLETE").exists():
    raise SystemExit(1)
payload = json.loads(marker.read_text(encoding="utf-8"))
digest = hashlib.sha256(summary.read_bytes()).hexdigest()
artifact_hashes = payload.get("artifact_sha256", {})
if not isinstance(artifact_hashes, dict) or not artifact_hashes:
    raise SystemExit(1)
for name, expected in artifact_hashes.items():
    relative = pathlib.Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise SystemExit(1)
    artifact = directory / relative
    if not artifact.is_file() or hashlib.sha256(artifact.read_bytes()).hexdigest() != expected:
        raise SystemExit(1)
raise SystemExit(0 if payload.get("status") == "complete" and payload.get("aggregate_summary_sha256") == digest else 1)
' "$directory"
}

if [[ -e "$FINAL_DIR" ]]; then
  if validate_aggregate "$FINAL_DIR"; then
    if [[ "$RESUME_AGGREGATE" == 1 ]]; then
      echo "Validated aggregate already exists; skipping: $FINAL_DIR"
      exit 0
    fi
    echo "Validated aggregate already exists: $FINAL_DIR" >&2
  else
    echo "Refusing to overwrite corrupt or incomplete aggregate: $FINAL_DIR" >&2
  fi
  exit 2
fi

mkdir -p "$RUN_ROOT/aggregate_attempts"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  ATTEMPT_TOKEN=job_${SLURM_JOB_ID}_restart_${SLURM_RESTART_COUNT:-0}
else
  ATTEMPT_TOKEN=manual_pid_$$_$(date -u +%Y%m%dT%H%M%SZ)
fi
ATTEMPT_DIR=$RUN_ROOT/aggregate_attempts/aggregate_${ATTEMPT_TOKEN}
if [[ -e "$ATTEMPT_DIR" ]]; then
  echo "Aggregate attempt already exists: $ATTEMPT_DIR" >&2
  exit 2
fi

"$PYTHON_BIN" -u "$PROJECT_DIR/scripts/aggregate_simplicial_runs.py" \
  --run-root "$RUN_ROOT" \
  --output-dir "$ATTEMPT_DIR" \
  --expected-runs "$EXPECTED_RUNS" \
  --ensemble-bootstrap "${ENSEMBLE_BOOTSTRAP:-5000}" \
  --ensemble-bootstrap-seed "${ENSEMBLE_BOOTSTRAP_SEED:-8675309}"

if ! validate_aggregate "$ATTEMPT_DIR"; then
  echo "Aggregator returned without a valid completion contract: $ATTEMPT_DIR" >&2
  exit 2
fi

exec 9>"$RUN_ROOT/.publish.lock"
flock 9
if [[ -e "$FINAL_DIR" ]]; then
  if ! validate_aggregate "$FINAL_DIR"; then
    echo "A competing job published an invalid aggregate: $FINAL_DIR" >&2
    exit 2
  fi
  echo "Another job already published the validated aggregate; preserving this attempt."
else
  mv -T -- "$ATTEMPT_DIR" "$FINAL_DIR"
fi
flock -u 9

if ! validate_aggregate "$FINAL_DIR"; then
  echo "Final aggregate failed post-publication validation: $FINAL_DIR" >&2
  exit 2
fi
echo "Published validated aggregate: $FINAL_DIR"
