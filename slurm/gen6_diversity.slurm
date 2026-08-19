#!/usr/bin/env bash
#SBATCH --job-name=ln-gen6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# gen6 Phase 1: the two experiments that decide whether chemical coverage, rather
# than model capacity, is the binding constraint on new-ligand prediction.
#
#   EXPERIMENT=A   BASE91 vs EXPANDED152 on byte-identical test rows
#   EXPERIMENT=B   diversity vs depth at equal training-row budget
#   EXPERIMENT=AB  both, in sequence, into RUN_ROOT/{experiment_a,experiment_b}
#
# Needs only the committed dataset parquet and descriptor parquet; no RDKit, no
# gen5 run.  Provenance reconstruction is NOT part of this job: it needs the raw
# SAFE exports, which live outside the repo, so run scripts/reconstruct_provenance.py
# on a machine that has them and pass the resulting state with PROVENANCE_STATE_JSON.
#
# Note before submitting anything: on a 2026 laptop the whole of Phase 1 takes
# minutes, not hours (one 400-tree fit on the 5,248-row cohort is ~1.2 s).  This
# wrapper exists for larger sweeps — more seeds, more feature sets, more budgets —
# not because the default configuration needs a cluster.

set -Eeuo pipefail
umask 027

export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-}}
RUN_ROOT=${RUN_ROOT:-}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}
EXPERIMENT=${EXPERIMENT:-AB}
FEATURE_SETS=${FEATURE_SETS:-MC_lig2d_ext_massaction MC_donors}
ARMS=${ARMS:-BASE EXPANDED EXPANDED_ROWMATCHED EXPANDED_SHUFFLED}
SPLIT_SEEDS=${SPLIT_SEEDS:-104729 130363 155921 196613 262147}
FOLDS=${FOLDS:-5}
N_ESTIMATORS=${N_ESTIMATORS:-400}
EVAL_MIN_CELLS=${EVAL_MIN_CELLS:-3}
BASE_MIN_CELLS=${BASE_MIN_CELLS:-10}
WEIGHTING=${WEIGHTING:-cluster}
BOOTSTRAP_REPLICATES=${BOOTSTRAP_REPLICATES:-5000}
POLICIES=${POLICIES:-depth diversity random maxmin}
BUDGETS=${BUDGETS:-250 500 1000 2000 3000 all}
DRAWS=${DRAWS:-3}
PROVENANCE_STATE_JSON=${PROVENANCE_STATE_JSON:-}

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
case "$EXPERIMENT" in A|B|AB) ;; *) echo "EXPERIMENT must be A, B or AB." >&2; exit 2 ;; esac

DESCRIPTORS=${DESCRIPTORS:-$PROJECT_DIR/dataset with 3D structures/ligand_2d_descriptors.parquet}
[[ -f "$DESCRIPTORS" ]] || {
  echo "Descriptor parquet not found at $DESCRIPTORS; LIG2D_EXT feature sets cannot run." >&2
  exit 2
}
mkdir -p "$RUN_ROOT"

# One frozen chemistry map for both experiments, so "super-cluster" means the same
# object in every artifact the run produces.
CHEMISTRY_DIR="$RUN_ROOT/chemistry"
if [[ ! -f "$CHEMISTRY_DIR/chemistry_map.parquet" ]]; then
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/build_chemistry_map.py" \
    --dataset "$PROJECT_DIR/dataset with 3D structures/dataset.parquet" \
    --descriptors "$DESCRIPTORS" \
    --output-dir "$CHEMISTRY_DIR"
fi
CHEMISTRY_MAP="$CHEMISTRY_DIR/chemistry_map.parquet"

prov_args=()
[[ -n "$PROVENANCE_STATE_JSON" ]] && prov_args=(--provenance-state-json "$PROVENANCE_STATE_JSON")

if [[ "$EXPERIMENT" == "A" || "$EXPERIMENT" == "AB" ]]; then
  # shellcheck disable=SC2086
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/run_diversity_causal.py" \
    --dataset "$PROJECT_DIR/dataset with 3D structures/dataset.parquet" \
    --descriptors "$DESCRIPTORS" \
    --chemistry-map "$CHEMISTRY_MAP" \
    --feature-sets $FEATURE_SETS \
    --arms $ARMS \
    --split-seeds $SPLIT_SEEDS \
    --folds "$FOLDS" \
    --n-estimators "$N_ESTIMATORS" \
    --n-jobs "$SLURM_CPUS_PER_TASK" \
    --eval-min-cells "$EVAL_MIN_CELLS" \
    --base-min-cells "$BASE_MIN_CELLS" \
    --weighting "$WEIGHTING" \
    --replicates "$BOOTSTRAP_REPLICATES" \
    "${prov_args[@]}" \
    --output-dir "$RUN_ROOT/experiment_a"
fi

if [[ "$EXPERIMENT" == "B" || "$EXPERIMENT" == "AB" ]]; then
  # shellcheck disable=SC2086
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/run_diversity_learning_curve.py" \
    --dataset "$PROJECT_DIR/dataset with 3D structures/dataset.parquet" \
    --descriptors "$DESCRIPTORS" \
    --chemistry-map "$CHEMISTRY_MAP" \
    --policies $POLICIES \
    --budgets $BUDGETS \
    --draws "$DRAWS" \
    --split-seeds $SPLIT_SEEDS \
    --folds "$FOLDS" \
    --n-estimators "$N_ESTIMATORS" \
    --n-jobs "$SLURM_CPUS_PER_TASK" \
    --replicates "$BOOTSTRAP_REPLICATES" \
    --output-dir "$RUN_ROOT/experiment_b"
fi

# Fail-closed aggregation: refuses to combine runs that disagree on the dataset,
# chemistry or feature-registry hashes.
"$PYTHON_BIN" "$PROJECT_DIR/scripts/aggregate_gen6.py" \
  --runs "$RUN_ROOT"/experiment_* \
  --output-dir "$RUN_ROOT/aggregate"
