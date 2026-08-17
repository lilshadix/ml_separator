#!/usr/bin/env bash
#SBATCH --job-name=ln-gen5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# gen5: log D level model.  One job evaluates every arm under both regimes
# (unseen ligand / unseen conditions) on all split seeds, then runs the k-shot
# calibration on the unseen-ligand OOF and writes the leaderboard to RUN_ROOT.
# Needs only the committed dataset parquet; no gen3 run, no RDKit.

set -Eeuo pipefail
umask 027

export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-}}
RUN_ROOT=${RUN_ROOT:-}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}
ARMS=${ARMS:-A_metal A_cond A_physchem A_ecfp A_lig2d_ext A_donors A_complex_phys A_polyhedron MC MC_physchem MC_ecfp MC_lig2d_ext MC_donors MC_complex_phys MC_polyhedron MC_all2d MC_all3d MC_everything}
REGIMES=${REGIMES:-unseen_ligand unseen_conditions}
SPLIT_SEEDS=${SPLIT_SEEDS:-104729 130363 155921 196613 262147}
FOLDS=${FOLDS:-5}
N_ESTIMATORS=${N_ESTIMATORS:-400}
MIN_ROWS=${MIN_ROWS:-10}
KSHOT=${KSHOT:-1 2 3 5}
KSHOT_DRAWS=${KSHOT_DRAWS:-20}
BOOTSTRAP_REPLICATES=${BOOTSTRAP_REPLICATES:-5000}
LEARNERS=${LEARNERS:-extratrees hgb ridge}

if [[ -z "$PROJECT_DIR" || -z "$RUN_ROOT" ]]; then
  echo "PROJECT_DIR and RUN_ROOT are required." >&2
  exit 2
fi
PROJECT_DIR=$(cd "$PROJECT_DIR" && pwd -P)
[[ "$RUN_ROOT" = /* ]] || { echo "RUN_ROOT must be absolute." >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "Python is not executable: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$PROJECT_DIR/dataset with 3D structures/dataset.parquet" ]] || {
  echo "Dataset parquet is missing under $PROJECT_DIR." >&2
  exit 2
}
DESCRIPTORS=${DESCRIPTORS:-$PROJECT_DIR/dataset with 3D structures/ligand_2d_descriptors.parquet}
[[ -f "$DESCRIPTORS" ]] || echo "NOTE: descriptor parquet not found at $DESCRIPTORS; LIG2D_EXT arms will be skipped." >&2
mkdir -p "$RUN_ROOT"

# shellcheck disable=SC2086
"$PYTHON_BIN" "$PROJECT_DIR/scripts/run_gen5_levels.py" \
  --arms $ARMS \
  --regimes $REGIMES \
  --split-seeds $SPLIT_SEEDS \
  --folds "$FOLDS" \
  --n-estimators "$N_ESTIMATORS" \
  --n-jobs "$SLURM_CPUS_PER_TASK" \
  --min-rows-per-extractant "$MIN_ROWS" \
  --learners $LEARNERS \
  --descriptors "$DESCRIPTORS" \
  --kshot $KSHOT \
  --kshot-draws "$KSHOT_DRAWS" \
  --replicates "$BOOTSTRAP_REPLICATES" \
  --output-dir "$RUN_ROOT"
