# Generated runs

The tabular and simplicial runners write reproducible benchmark artifacts here.
Generated results are ignored by Git. A directory is complete only when both
`summary.json` and its matching `_SUCCESS.json` exist; `_INCOMPLETE` is a hard
failure/preemption marker, not a resumable scientific result.

SLURM workers write unique directories under `attempts/` and atomically publish
only hash-validated results as top-level `run_*` directories. Aggregation ignores
attempts and follows the same attempt-then-publish contract under
`aggregate_attempts/`.
