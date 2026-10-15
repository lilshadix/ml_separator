# slurm/ — cluster jobs for gen2 to gen6

Submission is the user's; every `submit_*.sh` defaults to a dry run. Each job calls a script under
`scripts/` by its path, which is why `scripts/` stays flat. Jobs in generation order:

| gen | job | runs | aggregate |
|---|---|---|---|
| 2 | `ablation_multiseed.slurm` / `submit_ablation_multiseed.sh` | `scripts/run_ablation_benchmark.py` | `aggregate_ablation.slurm` → `scripts/aggregate_ablation_runs.py` |
| 2 | `descriptor_multiseed.slurm` / `submit_descriptor_multiseed.sh` | `scripts/run_delta3d_benchmark.py`, `scripts/build_ligand_descriptors.py` | `aggregate_descriptor.slurm` → `scripts/aggregate_delta3d_runs.py`, `scripts/compare_descriptor_arms.py` |
| 2 | `simplicial_multiseed.slurm` / `submit_simplicial_multiseed.sh` | `scripts/run_simplicial_benchmark.py` | `aggregate_simplicial.slurm` → `scripts/aggregate_simplicial_runs.py` |
| 3 | `gen3_multisplit.slurm` / `submit_gen3_multisplit.sh` | `scripts/run_gen3_benchmark.py` | `aggregate_gen3.slurm` → `scripts/aggregate_gen3_runs.py` |
| 4 | `gen4_candidates.slurm` / `submit_gen4_candidates.sh` | `scripts/run_gen4_candidates.py`, `scripts/gen4_decision.py` | — |
| 5 | `gen5_levels.slurm` / `submit_gen5_levels.sh` | `scripts/run_gen5_levels.py` | — |
| 6 | `gen6_diversity.slurm` / `submit_gen6_diversity.sh` | `scripts/reconstruct_provenance.py`, `scripts/build_chemistry_map.py`, `scripts/run_diversity_causal.py`, `scripts/run_diversity_learning_curve.py` | `scripts/aggregate_gen6.py` |

gen7 onwards ran locally (gen7–gen11 through `scripts/`, gen12 onwards through
`generations/<gen>/scripts/`).
