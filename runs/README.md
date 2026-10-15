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

## Index of run roots

One directory per published run, named after its generation. The protocol references and the
code paths below are fixed by hashes or by imports, so these names must not change.

| gen | run roots | note |
|---|---|---|
| 2 | `ablation_all_20260810T181058Z` | the gen2 primary; pinned by `gen3_protocol.json` and the gen3 run manifests, so it stays at this path |
| 2 | `gen2/` | every other gen2 run: the 2026-08-07 delta3d / compact / strict / shrinkage runs, `ablation_all_20260809T080927Z`, `ablation_prespecified_extractant_s42`, `descriptor_20260808T182353Z`, `simplicial_20260807T200005Z`, `condition_regime_20260815T171535Z`, `gen2_2026081*` (moved here 2026-10-15; the `output_dir` fields inside their `summary.json` record where they were written and were left as they were) |
| 3 | `gen3_primary_20260815T181555Z` | hash-validated split × model-seed tasks plus `aggregate/` |
| 4 | `gen4_candidates_20260817T003449Z`, `gen4_candidates_confirm_5seeds`, `gen4_candidates_controls_5seeds`, `gen4_candidates_seed104729_v{1,2}`, `kshot_calibration_20260817T181016Z` | candidates, confirmation, controls and the first k-shot study |
| 5 | `gen5_levels_20260817T195411Z`, `gen5_levels_20260818T180930Z` (crashed, kept for the record), `gen5_levels_20260818T211105Z`, `gen5_massaction_local_20260818` | the `log D` level target |
| 6 | `gen6_phase0`, `gen6_chemistry`, `gen6_provenance`, `gen6_exp{A,B,C,F}_*`, `gen6_aggregate` | `gen6_provenance/provenance_table.parquet` is read by gen13 onwards |
| 7–11 | `gen7_architecture`, `gen8_architecture`, `gen9_shape`, `gen10_final`, `gen11_transfer` | `gen7_architecture/cache/` holds the frozen cohort every later generation uses |
| other | `sae_dataset_audit` | audit of the all-metals archive build (`dataset_all_metals/`) |

New cluster runs still land at `runs/<protocol>_<stamp>/`, as the `slurm/` jobs write them.

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
