#!/usr/bin/env bash
# Local sequential driver for the generation-2 multi-seed benchmark.
#
# Mirrors the argument contract of slurm/ablation_multiseed.slurm exactly; it
# exists only because this workstation has no SLURM. Frozen outer folds
# (--split-seed) are shared by every model seed, as the protocol requires.

set -Eeuo pipefail
umask 027

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_DIR=$(cd "$SCRIPT_DIR/.." && pwd -P)
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}
DATASET_PATH="$PROJECT_DIR/dataset with 3D structures/dataset.parquet"
VR_ASSET_PATH="$PROJECT_DIR/dataset with 3D structures/features/vietoris_rips_inputs.npz"

MODEL_SEEDS=${MODEL_SEEDS:-"42 7 137 2027 9001"}
SPLIT_SEED=${SPLIT_SEED:-104729}
PAIR_SCOPE=${PAIR_SCOPE:-all}
GROUP_MODE=extractant
N_JOBS=${N_JOBS:-8}
RUN_ROOT=${RUN_ROOT:-$PROJECT_DIR/runs/gen2_$(date -u +%Y%m%dT%H%M%SZ)}

mkdir -p "$RUN_ROOT/logs"
printf 'task_index\tmodel_seed\tsplit_seed\n' > "$RUN_ROOT/seed_plan.tsv"

index=0
for MODEL_SEED in $MODEL_SEEDS; do
  printf '%s\t%s\t%s\n' "$index" "$MODEL_SEED" "$SPLIT_SEED" >> "$RUN_ROOT/seed_plan.tsv"
  RUN_DIR=$RUN_ROOT/run_${index}_model_${MODEL_SEED}_split_${SPLIT_SEED}
  if [[ -f "$RUN_DIR/_SUCCESS.json" ]]; then
    echo "skip completed $RUN_DIR"
    index=$((index + 1))
    continue
  fi
  rm -rf -- "$RUN_DIR"
  echo "=== model_seed=$MODEL_SEED -> $RUN_DIR"
  "$PYTHON_BIN" -u "$PROJECT_DIR/scripts/run_ablation_benchmark.py" \
    --dataset "$DATASET_PATH" \
    --vr-assets "$VR_ASSET_PATH" \
    --output-dir "$RUN_DIR" \
    --pair-scope "$PAIR_SCOPE" \
    --group-mode "$GROUP_MODE" \
    --model-seed "$MODEL_SEED" \
    --split-seed "$SPLIT_SEED" \
    --outer-folds 5 \
    --inner-folds 3 \
    --trees 200 \
    --bootstrap 1000 \
    --n-jobs "$N_JOBS" \
    --shuffle-seeds 1009,2017,3019 \
    --geometry-descriptor-blocks global_shape,coordination_shape \
    --descriptor-profile core \
    --delta3d-feature-set compact-invariant \
    --replicate-policy unique \
    --pair-response-3d \
    --electronic \
    --extension-arms \
    --two-d-sensitivity \
    --extension-shuffle-seeds 1009,2017,3019 \
    --symmetric-3d \
    --reference-baselines \
    2>&1 | tee "$RUN_ROOT/logs/model_${MODEL_SEED}.log"
  index=$((index + 1))
done

echo "=== aggregating"
"$PYTHON_BIN" -u "$PROJECT_DIR/scripts/aggregate_ablation_runs.py" \
  --run-root "$RUN_ROOT" \
  --output-dir "$RUN_ROOT/aggregate" \
  --expected-runs "$index" \
  --expected-pair-scope "$PAIR_SCOPE" \
  --bootstrap-replicates 2000 \
  --bootstrap-seed 8675309 \
  2>&1 | tee "$RUN_ROOT/logs/aggregate.log"
echo "=== done: $RUN_ROOT"
