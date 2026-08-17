#!/usr/bin/env bash
# Submit the gen5 log D level study as one SLURM job. DRY_RUN=1 is the default
# and performs no write and no scheduler action.
#
#   DRY_RUN=0 slurm/submit_gen5_levels.sh
#
# Override ARMS / SPLIT_SEEDS / CPUS / MEMORY / WALLTIME / RUN_TAG as needed.

set -Eeuo pipefail
umask 027

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_DIR=${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd -P)}
PROJECT_DIR=$(cd "$PROJECT_DIR" && pwd -P)
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}
DRY_RUN=${DRY_RUN:-1}
PARTITION=${PARTITION:-campus}
ACCOUNT=${ACCOUNT:-acf-utk0011}
CPUS=${CPUS:-8}
MEMORY=${MEMORY:-24G}
WALLTIME=${WALLTIME:-06:00:00}
ARMS=${ARMS:-A_metal A_cond A_physchem A_ecfp A_lig2d_ext A_donors A_complex_phys A_polyhedron MC MC_physchem MC_ecfp MC_lig2d_ext MC_donors MC_complex_phys MC_polyhedron MC_all2d MC_all3d MC_everything}
REGIMES=${REGIMES:-unseen_ligand unseen_conditions}
SPLIT_SEEDS=${SPLIT_SEEDS:-104729 130363 155921 196613 262147}
RUN_TAG=${RUN_TAG:-gen5_levels_$(date -u +%Y%m%dT%H%M%SZ)}
RUN_ROOT=${RUN_ROOT:-$PROJECT_DIR/runs/$RUN_TAG}

if [[ "$DRY_RUN" != 0 && "$DRY_RUN" != 1 ]]; then
  echo "DRY_RUN must be 0 or 1." >&2
  exit 2
fi
[[ -x "$PYTHON_BIN" ]] || { echo "Python is not executable: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$PROJECT_DIR/dataset with 3D structures/dataset.parquet" ]] || {
  echo "Dataset parquet is missing under $PROJECT_DIR." >&2
  exit 2
}
[[ "$RUN_ROOT" = /* ]] || { echo "RUN_ROOT must be absolute." >&2; exit 2; }
if [[ -e "$RUN_ROOT" ]]; then
  echo "RUN_ROOT already exists; refusing to overwrite: $RUN_ROOT" >&2
  exit 2
fi

echo "Gen5 level (log D) submission plan"
echo "RUN_ROOT=$RUN_ROOT"
echo "PARTITION=$PARTITION ACCOUNT=$ACCOUNT cpus=$CPUS mem=$MEMORY time=$WALLTIME"
echo "ARMS=$ARMS"
echo "REGIMES=$REGIMES"
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
    --output="$RUN_ROOT/logs/gen5-%j.out" \
    --error="$RUN_ROOT/logs/gen5-%j.err" \
    --export="ALL,PROJECT_DIR=$PROJECT_DIR,RUN_ROOT=$RUN_ROOT,PYTHON_BIN=$PYTHON_BIN,ARMS=$ARMS,REGIMES=$REGIMES,SPLIT_SEEDS=$SPLIT_SEEDS" \
    "$PROJECT_DIR/slurm/gen5_levels.slurm"
)
printf '%s\n' "$job" > "$RUN_ROOT/job.id"
echo "Submitted gen5 level job $job."
