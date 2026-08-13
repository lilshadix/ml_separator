#!/usr/bin/env bash
#SBATCH --job-name=ln-gen3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --array=0-4%2
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -Eeuo pipefail
umask 027

export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-16}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-}}
RUN_ROOT=${RUN_ROOT:-}
PROTOCOL_PATH=${PROTOCOL_PATH:-$PROJECT_DIR/gen3_protocol.json}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}
RUN_PLAN=${RUN_PLAN:-$RUN_ROOT/run_plan.tsv}
TASK_INDEX=${SLURM_ARRAY_TASK_ID:-}

if [[ -z "$PROJECT_DIR" || -z "$RUN_ROOT" || -z "$TASK_INDEX" ]]; then
  echo "PROJECT_DIR, RUN_ROOT, and SLURM_ARRAY_TASK_ID are required." >&2
  exit 2
fi
PROJECT_DIR=$(cd "$PROJECT_DIR" && pwd -P)
[[ "$RUN_ROOT" = /* ]] || { echo "RUN_ROOT must be absolute." >&2; exit 2; }
[[ -f "$RUN_PLAN" && -f "$PROTOCOL_PATH" ]] || {
  echo "Frozen run plan or protocol is missing." >&2
  exit 2
}
[[ -x "$PYTHON_BIN" ]] || { echo "Python is not executable: $PYTHON_BIN" >&2; exit 2; }

plan_row=$(awk -F '\t' -v task="$TASK_INDEX" 'NR > 1 && $1 == task {print; found=1} END {if (!found) exit 3}' "$RUN_PLAN")
IFS=$'\t' read -r planned_task split_seed model_seed run_role <<< "$plan_row"
if [[ "$run_role" != primary_split_seed ]]; then
  echo "The primary array accepts only primary_split_seed rows." >&2
  exit 2
fi
run_dir=$RUN_ROOT/run_${planned_task}_split_${split_seed}_model_${model_seed}
if [[ -e "$run_dir" ]]; then
  echo "Refusing to overwrite existing task output: $run_dir" >&2
  exit 2
fi

"$PYTHON_BIN" "$PROJECT_DIR/scripts/run_gen3_benchmark.py" \
  --protocol "$PROTOCOL_PATH" \
  --output-dir "$run_dir" \
  --split-seed "$split_seed" \
  --model-seed "$model_seed" \
  --n-jobs "$SLURM_CPUS_PER_TASK"

