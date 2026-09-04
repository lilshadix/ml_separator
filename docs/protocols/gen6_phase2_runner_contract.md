# gen6 Phase 2 runner contract (Experiments C and F) — implementation notes

*Engineering companion to [`gen6_phase2_protocol_20260819.md`](gen6_phase2_protocol_20260819.md).
The protocol is authoritative on science; this file is authoritative on how the runners call the
tested modules. Kept in `docs/` so it survives and so a reader can see what the scripts were built to.*

Repo `/Users/lilshadix/PycharmProjects/ml_separator`, python `.venv/bin/python` (numpy, pandas 3,
scipy, sklearn, pyarrow, pytest; NO rdkit). Read, in order:
1. `docs/gen6_phase2_protocol_20260819.md` — pre-registration (hypotheses, models, policies, endpoints).
2. `src/lanthanide_separation/gen6/hierarchical.py` (C) / `acquisition.py` (F) — the tested core
   modules to CALL; do not reimplement the models or policies.
3. `src/lanthanide_separation/gen6/{cohorts,metrics,manifest,chemistry}.py` — reuse
   `diversity_splits`, `assert_split_integrity`, `seeded_group_kfold`, `gen6_metric_table`,
   `decompose_level_shape`, `hard_chemistry_endpoints`, `per_unit_statistics`,
   `paired_unit_bootstrap` (returns percentile + BCa + cluster-robust + block-macro columns),
   `RunManifest`, `validate_run`, `write_success`, `build_chemistry_map` / `ChemistryMap`.
4. `scripts/run_diversity_causal.py` — the runner convention to MATCH: argparse `parse_args(argv=None)`,
   `main(argv=None) -> int`, UTC-stamped `runs/` dir, `log()` helper, manifest/validation/artifact
   hashes/`_SUCCESS.json`, `decision_report.md` with every number carrying its grouping and n, a
   verdict section scored PASS/FAIL/INCONCLUSIVE against the pre-registered condition with a
   "what would falsify this" line, and a "How to break this result" section. `--pilot`.

HARD RULES: modify only your assigned files; never edit a gen6 module or another script; report a
module bug with evidence instead of patching it; never fabricate a number; verify by running
`--pilot` on the real data and paste the real output.

Shared facts: cohort `build_level_dataset(min_rows_per_extractant=3, ligand_descriptors=desc)`
= 5,248 rows / 152 ext / 131 ECFP / 79 chemotypes. Chemotype folds = `diversity_splits(frame,
n_splits=5, seed=<split seed>)`, training side `train_index_by_arm["EXPANDED"]`, test
`test_index`. Fold seed: `random_state = 42 + fold*1009 + 9_999_991`. Group weights:
`LevelRegressor.fit(..., groups=train["ecfp_cluster"])`. One ExtraTrees(400) fit ≈ 1.2 s.
Chemistry map: `build_chemistry_map(source, ligand_descriptors=desc)`.

## scripts/run_hierarchical_levels.py (Experiment C)

Per regime in `--regimes` (default `unseen_chemotype unseen_ligand unseen_series`), per split seed
(default the 5 gen5 seeds 104729 130363 155921 196613 262147), per fold:
* `unseen_chemotype` folds from `diversity_splits` (so MONO_ET reproduces Experiment A's EXPANDED
  arm — ASSERT: load `runs/gen6_expA_5seed/oof_predictions.parquet`, feature_set ==
  `MC_lig2d_ext_massaction`, compare `prediction_EXPANDED` to MONO_ET per row_id per seed; max abs
  diff < 1e-9 on the same machine in a FULL run (400 trees); in `--pilot` (120 trees) SKIP with a
  reason). Other regimes: `seeded_group_kfold` on `ecfp_cluster` / `series_id`.
* MODEL_NAMES has EIGHT entries — MONO_ET, MONO_RIDGE, C1_HIER_RIDGE, C2_TWO_STAGE,
  C2_TRUECENTRE (protocol amendment: Stage B on the true within-cell departure;
  `predict_two_stage` returns `prediction_C2_TRUECENTRE`), C3_SHARED_RIDGE, ORACLE_LEVEL,
  ORACLE_METAL. Score all eight.
* Fit: MONO_ET = `LevelRegressor(data.block_columns(FOREST_BLOCKS), params)`; MONO_RIDGE and
  C1_HIER_RIDGE via `tune_ridge(train, data, group_column=<regime group column>,
  hierarchical=False/True, seed=split_seed)` then `HierarchicalRidge(design, λ_fixed, λ_ligand).fit(
  x, y, weights=group_balanced_weights(train ecfp_cluster), ligands=train extractant)`;
  C3_SHARED_RIDGE = HierarchicalRidge with C1's chosen λs plus `extra_x/extra_y/extra_w` from
  `pair_difference_rows(train, x_train, weights)`; C2 via `fit_two_stage(train, data,
  params=params, group_column=<regime group column>, seed=split_seed)` + `predict_two_stage(fit,
  test, data)` (gives C2_TWO_STAGE, C2_TRUECENTRE, ORACLE_LEVEL, ORACLE_METAL, stage_a, stage_b,
  stage_b_truecentre, cell_true_mean, cell_n_metals). Record chosen λs per (regime, seed, fold)
  and C1's `predict_components` aggregated per block (mean |contribution|) as attribution.
* OOF frame: one row per cohort row per regime per seed: identity columns, outer_fold,
  `prediction_<MODEL>` ×8, `cell_n_metals`, `nn_train_tanimoto` (max Tanimoto of the test ligand
  to that fold's training ligands, via `ChemistryMap.nearest_neighbour`).
* Metrics: `gen6_metric_table` (similarity_column `nn_train_tanimoto`); `per_unit_statistics` +
  `paired_unit_bootstrap` (unit ecfp_cluster, block tanimoto_cluster) for: C1_HIER_RIDGE vs
  MONO_RIDGE; C2_TWO_STAGE vs MONO_ET; C2_TRUECENTRE vs MONO_ET (exploratory label);
  C3_SHARED_RIDGE vs C1_HIER_RIDGE; ORACLE contrasts on multi-metal test cells (cell_n_metals ≥ 2)
  AND on all rows: comparisons {"level_oracle_gain": ("C2_TWO_STAGE","ORACLE_LEVEL"),
  "metal_oracle_gain": ("C2_TWO_STAGE","ORACLE_METAL"), "level_minus_metal_gain":
  ("ORACLE_METAL","ORACLE_LEVEL")} — the last one's delta is exactly the difference of the two
  gains and its BCa low > 0 is C1's pass condition. Statistics mae, offset_mae, shape_mae.
  Pool over seeds like Experiment A (contrasts.csv per seed + contrast_summary.csv pooled).
* Pairs: per test fold, `derived_pairs(test_with_predictions, prediction_columns)`,
  `pair_label_mean_null(train, pairs)` → `pair_NULL`, `pair_metrics` + `pair_consistency` per
  model; write `pair_predictions.parquet`, `pair_metrics.csv`; C4 = max antisymmetry and
  transitivity residual < 1e-9 over all folds.
* Verdicts C1–C4 exactly as pre-registered (C1 on multi-metal cells; all-rows beside it; C3 for
  both C2 variants with the amendment label). Report the λ_ligand frequency table per regime.
* Artifacts: oof_predictions.parquet, per_ligand_metrics.csv, arm_metrics.csv, hard_chemistry_metrics.csv,
  contrasts.csv, contrast_summary.csv, component_attribution.csv, penalties.csv, pair_metrics.csv,
  pair_predictions.parquet, decision_report.md, summary.json, manifest/validation/artifact_hashes/
  _SUCCESS. `--pilot`: one seed, `unseen_chemotype`, 120 trees.

## scripts/run_ligand_acquisition_sim.py (Experiment F)

Per split seed (default `104729 130363 155921`), per chemotype fold (EXPANDED training index,
fixed test_index): `start = start_cohort(frame, train_index, n_ligands=10)`; per policy in
`POLICIES`, per replicate r in range(--replicates, default 2): `run_acquisition(frame, data,
start=start, train_index=..., test_index=..., policy=..., feature_columns=
data.block_columns(FOREST_BLOCKS), params=LevelForestParameters(n_estimators=200, random_state=
42 + fold*1009 + 9_999_991 + 101*r, n_jobs=-1), chemistry=chemistry, budget_per_ligand=3,
n_steps=30, checkpoints=DEFAULT_CHECKPOINTS, seed=<deterministic from seed,fold,r>,
evaluate=<closure>)`.
* `evaluate(pred, train_rows, acquired)` on the fixed test rows: macro MAE over ecfp_cluster;
  hard-chemotype MAE on test rows whose max Tanimoto to the **start cohort's** extractants is < 0.4
  (and < 0.6) — computed ONCE per fold, never moving with policy or step; offset_mae / shape_mae via
  `decompose_level_shape`; worst-quartile ligand MAE; coverage: n chemotypes among training
  ligands, mean nn Tanimoto of test ligands to the current training ligands.
* Assert and record that no revealed row is a test row (the closure receives train_rows).
* Sensitivity `--budget-all-policies maxmin random` (budget_per_ligand=None, fewer reps ok).
* Aggregate: `acquisition_curves.csv` (seed/fold/policy/replicate/checkpoint + metrics),
  `acquisition_steps.csv` (every pick with nn_to_acquired_before, chemotype), `curve_summary.csv`
  (policy × checkpoint mean/sd), and paired bootstraps per checkpoint: per (seed, fold, checkpoint)
  build the fold's test rows with `prediction_<policy>` = mean over replicates of
  `trace.predictions[checkpoint]`; concatenate over folds/seeds; `per_unit_statistics` (unit
  ecfp_cluster); `paired_unit_bootstrap` (block tanimoto_cluster); comparisons random vs every
  other policy, statistics mae/offset_mae/shape_mae; AND the hard endpoint (restrict to the fixed
  hard mask before per_unit_statistics). F1 = hard endpoint maxmin vs random; F2 = offset_mae
  offset_uncertainty vs random; F3 = hard endpoint same_chemotype_first vs random, sign reversed.
* F4: implement `depth_on_start` only under `--f4-cap-start-rows N` (cap each start ligand to N
  rows so there are unrevealed start rows to "buy"; the cap then applies to the whole run so all
  policies share the same start); default: F4 SKIPPED with reason "needs --f4-cap-start-rows;
  Experiment B already measured depth vs breadth".
* Print a time estimate from the first fold before the main loop.
* Artifacts: acquisition_curves.csv, acquisition_steps.csv, curve_summary.csv, contrasts.csv,
  decision_report.md, summary.json, manifest/validation/_SUCCESS. `--pilot`: one seed, folds 0
  and 2, policies random maxmin offset_uncertainty, n_steps 8, checkpoints (1,2,3,5,8),
  1 replicate, 80 trees.

Tests (`tests/test_gen6_hierarchical_runner.py` / `tests/test_gen6_acquisition_runner.py`): load
the script via importlib, inserting into `sys.modules` BEFORE `exec_module` (see
`tests/test_gen6_diversity_causal.py`); test helpers on synthetic frames: the hard mask is fixed
across steps; verdict logic; the MONO_ET-equals-Experiment-A assertion; the OOF layout.
