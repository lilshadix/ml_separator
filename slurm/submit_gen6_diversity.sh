#!/usr/bin/env bash
# Submit gen6 Phase 1 (Experiments A and B) as one SLURM job. DRY_RUN=1 is the
# default and performs no write and no scheduler action.
#
#   DRY_RUN=0 slurm/submit_gen6_diversity.sh
#
# Override EXPERIMENT / FEATURE_SETS / SPLIT_SEEDS / POLICIES / BUDGETS / DRAWS /
# CPUS / MEMORY / WALLTIME / RUN_TAG as needed.
#
# Honest sizing note: the default configuration finishes in minutes on a laptop
# (one 400-tree fit on the 5,248-row cohort is ~1.2 s; the whole of Experiment A
# with 2 feature sets x 5 seeds x 5 folds x 4 arms took ~2 minutes locally).
# Submit this only for a sweep genuinely larger than that — more seeds, more
# feature sets, finer budgets — otherwise just run the two scripts directly.

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
MEMORY=${MEMORY:-16G}
WALLTIME=${WALLTIME:-04:00:00}
EXPERIMENT=${EXPERIMENT:-AB}
FEATURE_SETS=${FEATURE_SETS:-MC_lig2d_ext_massaction MC_donors}
ARMS=${ARMS:-BASE EXPANDED EXPANDED_ROWMATCHED EXPANDED_SHUFFLED}
SPLIT_SEEDS=${SPLIT_SEEDS:-104729 130363 155921 196613 262147}
POLICIES=${POLICIES:-depth diversity random maxmin}
BUDGETS=${BUDGETS:-250 500 1000 2000 3000 all}
DRAWS=${DRAWS:-3}
N_ESTIMATORS=${N_ESTIMATORS:-400}
WEIGHTING=${WEIGHTING:-cluster}
PROVENANCE_STATE_JSON=${PROVENANCE_STATE_JSON:-}
RUN_TAG=${RUN_TAG:-gen6_diversity_$(date -u +%Y%m%dT%H%M%SZ)}
RUN_ROOT=${RUN_ROOT:-$PROJECT_DIR/runs/$RUN_TAG}

if [[ "$DRY_RUN" != 0 && "$DRY_RUN" != 1 ]]; then
  echo "DRY_RUN must be 0 or 1." >&2
  exit 2
fi
case "$EXPERIMENT" in A|B|AB) ;; *) echo "EXPERIMENT must be A, B or AB." >&2; exit 2 ;; esac
[[ -x "$PYTHON_BIN" ]] || { echo "Python is not executable: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$PROJECT_DIR/dataset with 3D structures/dataset.parquet" ]] || {
  echo "Dataset parquet is missing under $PROJECT_DIR." >&2
  exit 2
}
[[ -f "$PROJECT_DIR/dataset with 3D structures/ligand_2d_descriptors.parquet" ]] || {
  echo "Descriptor parquet is missing; LIG2D_EXT feature sets cannot run." >&2
  exit 2
}
[[ "$RUN_ROOT" = /* ]] || { echo "RUN_ROOT must be absolute." >&2; exit 2; }
if [[ -e "$RUN_ROOT" ]]; then
  echo "RUN_ROOT already exists; refusing to overwrite: $RUN_ROOT" >&2
  exit 2
fi
if [[ -n "$PROVENANCE_STATE_JSON" && ! -f "$PROVENANCE_STATE_JSON" ]]; then
  echo "PROVENANCE_STATE_JSON does not exist: $PROVENANCE_STATE_JSON" >&2
  exit 2
fi

echo "gen6 Phase 1 submission plan"
echo "RUN_ROOT=$RUN_ROOT"
echo "PARTITION=$PARTITION ACCOUNT=$ACCOUNT cpus=$CPUS mem=$MEMORY time=$WALLTIME"
echo "EXPERIMENT=$EXPERIMENT"
echo "FEATURE_SETS=$FEATURE_SETS"
echo "ARMS=$ARMS"
echo "SPLIT_SEEDS=$SPLIT_SEEDS  N_ESTIMATORS=$N_ESTIMATORS  WEIGHTING=$WEIGHTING"
echo "POLICIES=$POLICIES"
echo "BUDGETS=$BUDGETS  DRAWS=$DRAWS"
echo "PROVENANCE_STATE_JSON=${PROVENANCE_STATE_JSON:-<none; provenance recorded as not_audited>}"
echo
echo "Provenance note: publication identity needs the raw *_SAFE.csv exports, which live outside"
echo "this repository. Run scripts/reconstruct_provenance.py where they are reachable and pass the"
echo "resulting summary.json via PROVENANCE_STATE_JSON, or the run records provenance as unaudited."

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
    --output="$RUN_ROOT/logs/gen6-%j.out" \
    --error="$RUN_ROOT/logs/gen6-%j.err" \
    --export="ALL,PROJECT_DIR=$PROJECT_DIR,RUN_ROOT=$RUN_ROOT,PYTHON_BIN=$PYTHON_BIN,EXPERIMENT=$EXPERIMENT,FEATURE_SETS=$FEATURE_SETS,ARMS=$ARMS,SPLIT_SEEDS=$SPLIT_SEEDS,POLICIES=$POLICIES,BUDGETS=$BUDGETS,DRAWS=$DRAWS,N_ESTIMATORS=$N_ESTIMATORS,WEIGHTING=$WEIGHTING,PROVENANCE_STATE_JSON=$PROVENANCE_STATE_JSON" \
    "$PROJECT_DIR/slurm/gen6_diversity.slurm"
)
printf '%s\n' "$job" > "$RUN_ROOT/job.id"
echo "Submitted gen6 Phase 1 job $job."
