# Generated runs

The tabular and simplicial runners write reproducible benchmark artifacts here.
A directory is complete only when both `summary.json` and its matching
`_SUCCESS.json` exist; `_INCOMPLETE` is a hard failure/preemption marker, not a
resumable scientific result. `run_config.json` is a startup snapshot and its
`status` field stays `incomplete` for the life of the run, so never use it as a
completion signal.

SLURM workers write unique directories under `attempts/` and atomically publish
only hash-validated results as `run_*` directories. Aggregation ignores attempts
and follows the same attempt-then-publish contract.

## Retention

Published artifacts of the current protocols (`simplicial_*`, `descriptor_*`) are
tracked by Git so that a reviewer with a clean clone can verify every hash a
`_SUCCESS.json` declares. Ignored inside those roots: `logs/`, `tmp/`,
`attempts/`, `submissions/`, `status/`, lock files, `_INCOMPLETE`, and
`*.joblib` fitted forests, which are hundreds of megabytes and are reproducible
from the recorded seeds, contracts and asset hashes.

Legacy directories from earlier protocols remain ignored. They use the older
split protocol and an older compact feature contract, they lack the current
completion markers, and they must not be pooled into a current aggregate.

## Layout of a descriptor run root

    runs/descriptor_<stamp>/
      seed_plan.tsv              immutable arm/seed plan, written once
      arm_real/run_*/            descriptors from their own geometry
      arm_real/aggregate/        fail-closed multi-seed aggregate
      arm_null/run_*/            descriptors detached by a seeded permutation
      arm_null/aggregate/
      comparison/arm_comparison.md   real-versus-null verdict

The verdict in `comparison/` is the only output that carries a chemical claim.
