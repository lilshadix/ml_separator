#!/usr/bin/env bash
#SBATCH --job-name=ln-descriptor-agg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Fail-closed aggregation of both descriptor arms, followed by the head-to-head
# real-versus-permuted comparison that carries the actual scientific claim.

set -Eeuo pipefail
umask 027

PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$PROJECT_DIR" ]]; then
  echo "PROJECT_DIR is required." >&2
  exit 2
fi
PROJECT_DIR=$(cd "$PROJECT_DIR" && pwd -P)
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}
PAIR_SCOPE=${PAIR_SCOPE:-all}

if [[ -z "${RUN_ROOT:-}" || "$RUN_ROOT" != /* ]]; then
  echo "RUN_ROOT must be an absolute path." >&2
  exit 2
fi
[[ -d "$RUN_ROOT" ]] || { echo "RUN_ROOT does not exist: $RUN_ROOT" >&2; exit 2; }
RUN_ROOT=$(cd "$RUN_ROOT" && pwd -P)

EXPECTED_RUNS=${EXPECTED_RUNS:-5}
if [[ ! "$EXPECTED_RUNS" =~ ^[2-9][0-9]*$ ]]; then
  echo "EXPECTED_RUNS must be an integer of at least 2." >&2
  exit 2
fi
if [[ "$PAIR_SCOPE" != all && "$PAIR_SCOPE" != adjacent ]]; then
  echo "PAIR_SCOPE must be all or adjacent." >&2
  exit 2
fi
RESUME_AGGREGATE=${RESUME_AGGREGATE:-1}
if [[ "$RESUME_AGGREGATE" != 0 && "$RESUME_AGGREGATE" != 1 ]]; then
  echo "RESUME_AGGREGATE must be 0 or 1." >&2
  exit 2
fi

overall=0
for arm in real null; do
  arm_root=$RUN_ROOT/arm_$arm
  if [[ ! -d "$arm_root" ]]; then
    echo "Missing arm directory: $arm_root" >&2
    overall=2
    continue
  fi
  aggregate_dir=$arm_root/aggregate
  if [[ -e "$aggregate_dir" ]]; then
    if [[ "$RESUME_AGGREGATE" == 1 ]]; then
      echo "Replacing previous aggregate: $aggregate_dir"
      rm -rf -- "$aggregate_dir"
    else
      echo "Aggregate already exists and RESUME_AGGREGATE=0: $aggregate_dir" >&2
      overall=2
      continue
    fi
  fi
  echo "Aggregating arm=$arm"
  if ! "$PYTHON_BIN" -u "$PROJECT_DIR/scripts/aggregate_delta3d_runs.py" \
    --run-root "$arm_root" \
    --expected-pair-scope "$PAIR_SCOPE" \
    --comparison extended_vs_delta3d \
    --expected-runs "$EXPECTED_RUNS"; then
    echo "Aggregation failed for arm=$arm" >&2
    overall=2
  fi
done

if (( overall != 0 )); then
  echo "At least one arm failed to aggregate; skipping the head-to-head comparison." >&2
  exit "$overall"
fi

"$PYTHON_BIN" -u "$PROJECT_DIR/scripts/compare_descriptor_arms.py" \
  --real-aggregate "$RUN_ROOT/arm_real/aggregate/aggregate_summary.json" \
  --null-aggregate "$RUN_ROOT/arm_null/aggregate/aggregate_summary.json" \
  --output-dir "$RUN_ROOT/comparison"

echo "Aggregates: $RUN_ROOT/arm_real/aggregate, $RUN_ROOT/arm_null/aggregate"
echo "Verdict: $RUN_ROOT/comparison/arm_comparison.md"
