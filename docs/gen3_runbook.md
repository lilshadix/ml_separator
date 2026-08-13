# Generation 3 runbook

Generation 3 is additive. Do not place outputs under the frozen generation-2 run or edit
`gen3_protocol.json` after any scientific task starts. Every scientific task verifies the
dataset, cohort, feature registry, exact package versions, Linux x86_64 platform, source-code
hashes, fold invariants, and artifact hashes. A quick run is permanently marked ineligible for
aggregation.

## 1. Exact environment

Use Python 3.11.11 on Linux x86_64. Replace `/path/to/python3.11.11` with that exact interpreter:

```bash
cd /path/to/ml_separator
/path/to/python3.11.11 -m venv .venv-gen3
.venv-gen3/bin/python -m pip install --upgrade pip
.venv-gen3/bin/python -m pip install -r requirements-gen3.txt
.venv-gen3/bin/python -m pip install -e .
.venv-gen3/bin/python -m pytest -q
```

The runner refuses a non-quick scientific task if the frozen software/platform contract does
not match. The aggregator repeats that check and also requires identical implementation hashes
across runs.

## 2. Local non-scientific smoke

This exercises every H1/H2/H3 branch with a deliberately tiny budget and 100 bootstrap draws:

```bash
SMOKE_DIR=$(mktemp -d /tmp/ln-gen3-smoke.XXXXXX)
.venv/bin/python scripts/run_gen3_benchmark.py \
  --protocol gen3_protocol.json \
  --output-dir "$SMOKE_DIR/run" \
  --split-seed 104729 \
  --model-seed 42 \
  --n-jobs 4 \
  --quick
jq . "$SMOKE_DIR/run/validation.json"
```

`quick=true` and `scientific_result_eligible=false` are intentional.

## 3. Required dry run

The submission wrapper defaults to dry-run and creates neither directories nor jobs:

```bash
cd /path/to/ml_separator
DRY_RUN=1 \
PYTHON_BIN="$PWD/.venv-gen3/bin/python" \
RUN_TAG=gen3_primary_$(date -u +%Y%m%dT%H%M%SZ) \
bash slurm/submit_gen3_multisplit.sh
```

Verify that the printed plan contains exactly model seed 42 paired with split seeds 104729,
130363, 155921, 196613, and 262147, plus array `0-4`. The protocol hash printed by the wrapper
must be preserved with the run.

## 4. Scientific submission (not run during implementation)

Only after the dry-run plan and environment are accepted, submit the same command with
`DRY_RUN=0`:

```bash
cd /path/to/ml_separator
DRY_RUN=0 \
PYTHON_BIN="$PWD/.venv-gen3/bin/python" \
RUN_TAG=gen3_primary_$(date -u +%Y%m%dT%H%M%SZ) \
MAX_CONCURRENT=2 \
bash slurm/submit_gen3_multisplit.sh
```

The wrapper creates a five-task array and an `afterany` fail-closed aggregate job. A failed,
missing, quick, version-mismatched, hash-mismatched, or leakage-invalid task makes aggregation
fail; it is never counted as completed science.

## 5. Secondary model-seed confirmations

Model seeds 7 and 137 are frozen confirmations only on split 104729. Run them into separate
directories under the same run root after the primary tasks if desired:

```bash
RUN_ROOT=/absolute/path/to/runs/gen3_primary_TIMESTAMP
for MODEL_SEED in 7 137; do
  "$PWD/.venv-gen3/bin/python" scripts/run_gen3_benchmark.py \
    --protocol "$RUN_ROOT/gen3_protocol.json" \
    --output-dir "$RUN_ROOT/run_confirm_split_104729_model_${MODEL_SEED}" \
    --split-seed 104729 \
    --model-seed "$MODEL_SEED" \
    --n-jobs 16
done
```

These confirmation runs never enter the five-split champion score and never give split 104729
extra weight.

## 6. Completion evidence

Trust the aggregate only when all of these exist and validate:

- five primary task `_SUCCESS.json` files;
- `aggregate/_SUCCESS.json` and `aggregate/validation.json` with `passed=true`;
- `aggregate/artifact_hashes.json` matching every declared aggregate artifact;
- `aggregate/report.md`, beginning with the primary macro-MAE leaderboard;
- `aggregate/aggregate_summary.json`, containing the frozen-rule champion or the valid negative
  result retaining `A2_current_champion`.

The scientific answers to Q1, Q2, and Q3 come only from that validated five-split aggregate,
not from smoke output or an individual split.
