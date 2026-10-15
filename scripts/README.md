# scripts/ — gen2 to gen11 runners and analysis scripts

Run every script from the repository root with `.venv/bin/python scripts/<name>.py`. The layout is
flat and must stay flat: `slurm/*.slurm` and the run manifests reference these paths. Each script's
module docstring states its purpose; this index orders them by generation and by the stage of the
pipeline they belong to (data → run → aggregate → analyse → report). gen12 onwards live under
`generations/<gen>/scripts/`.

## Shared data preparation

| stage | script | purpose |
|---|---|---|
| descriptors | `build_ligand_descriptors.py` | extended 2D ligand-descriptor table for every unique extractant |
| descriptors | `build_chemistry_map.py` | the frozen all-190 chemistry map (chemotypes, ECFP clusters) used from gen6 on |
| descriptors | `build_gen7_ligand_embeddings.py` | frozen pretrained molecular embeddings (gen7) |
| provenance | `reconstruct_provenance.py` | reconstruct publication / series provenance (gen6) |
| utility | `compress_run_artifacts.py` | losslessly compress oversized run artefacts for transport |
| utility | `reproduce_headline_metrics.py` | recompute the headline pair-model metrics from a committed OOF file |

## gen2 — does 3D coordination geometry beat 2D? (`src/lanthanide_separation/ablation.py`, `evaluation.py`, `simplicial.py` in the same package)

| stage | script |
|---|---|
| run | `run_ablation_benchmark.py` (A0–A6 feature families), `run_delta3d_benchmark.py`, `run_simplicial_benchmark.py` |
| aggregate | `aggregate_ablation_runs.py`, `aggregate_delta3d_runs.py`, `aggregate_simplicial_runs.py` |
| analyse | `compare_descriptor_arms.py` (real vs permuted null), `analyze_oof_r2.py`, `analyze_screening_utility.py`, `evaluate_condition_regime.py` |
| audit | `audit/gen13_sf_data_audit_part{1,2}.py` (the gen13 separation-factor data audit; outputs beside them) |

## gen3 — learner, residual and architecture changes (`src/lanthanide_separation/gen3_*.py`, `gen3_protocol.json`)

| stage | script |
|---|---|
| run | `run_gen3_benchmark.py` (one hash-validated split/model-seed task) |
| aggregate | `aggregate_gen3_runs.py` |

## gen4 — candidate families and the first k-shot study (`src/lanthanide_separation/gen4_candidates.py` and `kshot_calibration.py`)

| stage | script |
|---|---|
| run | `run_gen4_candidates.py`, `run_kshot_calibration.py` |
| decide | `gen4_decision.py`, `foldseed_paired_intervals.py` |

## gen5 — the `log D` level target (`src/lanthanide_separation/levels.py`)

| stage | script |
|---|---|
| run | `run_gen5_levels.py` |

## gen6 — coverage or capacity? (`src/lanthanide_separation/gen6/`)

| stage | script |
|---|---|
| gate | `gen6_phase0.py` (the analytical package that must pass before any run) |
| run | `run_diversity_causal.py` (A), `run_diversity_learning_curve.py` (B), `run_hierarchical_levels.py` (C), `run_ligand_acquisition_sim.py` (F) |
| aggregate | `aggregate_gen6.py` |

## gen7 — where is the ceiling? (`src/lanthanide_separation/gen7/`)

| stage | script |
|---|---|
| run | `run_gen7_experiment.py`, `gen7_level_benchmark.py`, `gen7_kshot.py`, `gen7_ablation.py`, `gen7_ensemble.py` |
| analyse | `gen7_collect.py`, `gen7_ceiling_analysis.py`, `gen7_error_archaeology.py` |
| report | `gen7_report_tables.py`, `gen7_figures.py`, `gen7_manifest.py` |

## gen8 — few-shot calibration and acquisition (`src/lanthanide_separation/gen8/`)

| stage | script |
|---|---|
| audit | `gen8_series_audit.py` |
| run | `gen8_run.py` (policies × adapters × k), `gen8_kshot.py`, `gen8_curve_baselines.py`, `gen8_cnp.py`, `gen8_physics_latent.py`, `gen8_slope_restore.py`, `gen8_uncertainty.py` |
| verify | `gen8_cnp_verify.py`, `gen8_cnp_probe_lambda.py`, `gen8_physics_latent_ablation.py`, `gen8_ablation.py` |
| analyse | `gen8_analysis.py`, `gen8_calibration_geography.py`, `gen8_slopes.py`, `gen8_lanthanide.py`, `gen8_mechanism_similarity.py`, `gen8_error_archaeology.py`, `gen8_uncertainty_analysis.py`, `gen8_recommend_experiment.py` |
| report | `gen8_curve_baselines_report.py`, `gen8_cnp_report.py`, `gen8_cnp_verify_report.py`, `gen8_physics_latent_report.py`, `gen8_uncertainty_report.py`, `gen8_figures.py`, `gen8_manifest.py` |

## gen9 — curve shape (`src/lanthanide_separation/gen9/`)

| stage | script |
|---|---|
| gate | `gen9_reproduce_gen8.py` (Phase 0), `gen9_data_audit.py` |
| run | `gen9_curves.py`, `gen9_tune.py`, `gen9_shape_train.py`, `gen9_kshot.py`, `gen9_acquisition.py` |
| analyse | `gen9_shape_analysis.py`, `gen9_frontier.py`, `gen9_sensitivity.py` |
| report | `gen9_figures.py`, `gen9_manifest.py` |

## gen10 — finalise and freeze the `log D` pipeline (`src/lanthanide_separation/gen10/`; phases in order)

`gen10_phase0.py` (freeze and reproduce) → `gen10_query_consistency.py` (1) → `gen10_train.py`
(2, 4) → `gen10_adaptation.py` (5) → `gen10_data_ceiling.py` (7) → `gen10_publication_sensitivity.py`
(8) → `gen10_budget_simulation.py` (9) → `gen10_error_decomposition.py` (10) →
`gen10_final_locked.py` (11) → `gen10_self_audit.py` (12) → `gen10_report_tables.py`, `gen10_manifest.py`.

## gen11 — multi-metal actinide transfer (`src/lanthanide_separation/gen11/`)

`gen11_run.py` → `gen11_metrics.py` → `gen11_negative_transfer.py`, `gen11_query_consistency.py`,
`gen11_matched_analysis.py` → `gen11_self_audit.py` → `gen11_report_tables.py`, `gen11_manifest.py`.
