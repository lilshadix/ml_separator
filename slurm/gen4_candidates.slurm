#!/usr/bin/env bash
#SBATCH --job-name=ln-gen4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# One job evaluates every requested gen4 candidate arm on all five frozen gen3
# outer-fold seeds and writes the leaderboard/decision inputs to RUN_ROOT.
# The harness needs the completed gen3 run (folds + A2 params) at GEN3_RUN_DIR
# and the committed ligand-descriptor parquet; RDKit is NOT required.

set -Eeuo pipefail
umask 027

export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-16}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-}}
RUN_ROOT=${RUN_ROOT:-}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}
GEN3_RUN_DIR=${GEN3_RUN_DIR:-$PROJECT_DIR/runs/gen3_primary_20260815T181555Z}
ARMS=${ARMS:-A2_refit PRIOR PRIOR_TRENDONLY A2_lig2d A2_lig2d_SHUF SCALE_s50 HIER A2_ens}
SPLIT_SEEDS=${SPLIT_SEEDS:-104729 130363 155921 196613 262147}
N_ESTIMATORS=${N_ESTIMATORS:-200}
BOOTSTRAP_REPLICATES=${BOOTSTRAP_REPLICATES:-10000}

if [[ -z "$PROJECT_DIR" || -z "$RUN_ROOT" ]]; then
  echo "PROJECT_DIR and RUN_ROOT are required." >&2
  exit 2
fi
PROJECT_DIR=$(cd "$PROJECT_DIR" && pwd -P)
[[ "$RUN_ROOT" = /* ]] || { echo "RUN_ROOT must be absolute." >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "Python is not executable: $PYTHON_BIN" >&2; exit 2; }
[[ -d "$GEN3_RUN_DIR/aggregate" ]] || {
  echo "Completed gen3 run (with aggregate/) is missing: $GEN3_RUN_DIR" >&2
  exit 2
}
[[ -f "$PROJECT_DIR/dataset with 3D structures/ligand_2d_descriptors.parquet" ]] || {
  echo "Ligand descriptor parquet is missing; run scripts/build_ligand_descriptors.py (needs RDKit) or drop *_lig2d arms." >&2
  exit 2
}
mkdir -p "$RUN_ROOT"

# shellcheck disable=SC2086
"$PYTHON_BIN" "$PROJECT_DIR/scripts/run_gen4_candidates.py" \
  --run-dir "$GEN3_RUN_DIR" \
  --split-seeds $SPLIT_SEEDS \
  --arms $ARMS \
  --n-estimators "$N_ESTIMATORS" \
  --n-jobs "$SLURM_CPUS_PER_TASK" \
  --bootstrap-replicates "$BOOTSTRAP_REPLICATES" \
  --output-dir "$RUN_ROOT"

"$PYTHON_BIN" "$PROJECT_DIR/scripts/gen4_decision.py" "$RUN_ROOT/oof_predictions.csv" \
  --extra A2_refit_TP PRIOR_TP A2_lig2d_TP A2_run A2_run_TP PAIRMEAN_baseline \
  --replicates "$BOOTSTRAP_REPLICATES" | tee "$RUN_ROOT/decision_report.txt"
