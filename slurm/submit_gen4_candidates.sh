#!/usr/bin/env bash
# Submit the gen4 candidate evaluation as one SLURM job. DRY_RUN=1 is the
# default and performs no write and no scheduler action.
#
#   DRY_RUN=0 PYTHON_BIN="$PWD/.venv-gen3/bin/python" slurm/submit_gen4_candidates.sh
#
# Override ARMS / SPLIT_SEEDS / CPUS / MEMORY / WALLTIME / RUN_TAG as needed.

set -Eeuo pipefail
umask 027

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_DIR=${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd -P)}
PROJECT_DIR=$(cd "$PROJECT_DIR" && pwd -P)
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}
GEN3_RUN_DIR=${GEN3_RUN_DIR:-$PROJECT_DIR/runs/gen3_primary_20260815T181555Z}
DRY_RUN=${DRY_RUN:-1}
PARTITION=${PARTITION:-campus}
ACCOUNT=${ACCOUNT:-acf-utk0011}
CPUS=${CPUS:-8}
MEMORY=${MEMORY:-12G}
WALLTIME=${WALLTIME:-03:00:00}
ARMS=${ARMS:-A2_refit PRIOR PRIOR_TRENDONLY A2_lig2d A2_lig2d_SHUF SCALE_s50 HIER A2_ens}
SPLIT_SEEDS=${SPLIT_SEEDS:-104729 130363 155921 196613 262147}
RUN_TAG=${RUN_TAG:-gen4_candidates_$(date -u +%Y%m%dT%H%M%SZ)}
RUN_ROOT=${RUN_ROOT:-$PROJECT_DIR/runs/$RUN_TAG}

if [[ "$DRY_RUN" != 0 && "$DRY_RUN" != 1 ]]; then
  echo "DRY_RUN must be 0 or 1." >&2
  exit 2
fi
[[ -x "$PYTHON_BIN" ]] || { echo "Python is not executable: $PYTHON_BIN" >&2; exit 2; }
[[ -d "$GEN3_RUN_DIR/aggregate" ]] || { echo "Completed gen3 run is missing: $GEN3_RUN_DIR" >&2; exit 2; }
[[ -f "$PROJECT_DIR/dataset with 3D structures/ligand_2d_descriptors.parquet" ]] || {
  echo "Ligand descriptor parquet is missing (build with scripts/build_ligand_descriptors.py or drop *_lig2d arms)." >&2
  exit 2
}
[[ "$RUN_ROOT" = /* ]] || { echo "RUN_ROOT must be absolute." >&2; exit 2; }
if [[ -e "$RUN_ROOT" ]]; then
  echo "RUN_ROOT already exists; refusing to overwrite: $RUN_ROOT" >&2
  exit 2
fi

echo "Gen4 candidate submission plan"
echo "RUN_ROOT=$RUN_ROOT"
echo "GEN3_RUN_DIR=$GEN3_RUN_DIR"
echo "PARTITION=$PARTITION ACCOUNT=$ACCOUNT cpus=$CPUS mem=$MEMORY time=$WALLTIME"
echo "ARMS=$ARMS"
echo "SPLIT_SEEDS=$SPLIT_SEEDS"

if [[ "$DRY_RUN" == 1 ]]; then
  echo "DRY_RUN=1: no directories or SLURM jobs were created."
  exit 0
fi

command -v sbatch >/dev/null || { echo "sbatch is unavailable." >&2; exit 2; }
mkdir -p "$RUN_ROOT/logs"

job=$(
  sbatch --parsable \
    --partition="$PARTITION" \
    --account="$ACCOUNT" \
    --cpus-per-task="$CPUS" \
    --mem="$MEMORY" \
    --time="$WALLTIME" \
    --output="$RUN_ROOT/logs/gen4-%j.out" \
    --error="$RUN_ROOT/logs/gen4-%j.err" \
    --export="ALL,PROJECT_DIR=$PROJECT_DIR,RUN_ROOT=$RUN_ROOT,PYTHON_BIN=$PYTHON_BIN,GEN3_RUN_DIR=$GEN3_RUN_DIR,ARMS=$ARMS,SPLIT_SEEDS=$SPLIT_SEEDS" \
    "$PROJECT_DIR/slurm/gen4_candidates.slurm"
)
printf '%s\n' "$job" > "$RUN_ROOT/job.id"
echo "Submitted gen4 candidate job $job."
