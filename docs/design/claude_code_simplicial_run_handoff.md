# Claude Code Handoff: Simplicial Multi-Seed Evaluation

Date of local audit: 2026-08-08  
Repository: `ml_separator`  
Run bundle: `runs/simplicial_20260807T200005Z`

## Executive decision

The current-protocol five-seed run is complete, internally consistent, and
scientifically negative for the SNN hypothesis.

- Do **not** claim that the guarded simplicial neural network improves the
  adaptive tabular Delta3D model.
- Do **not** train or publish a deployment SNN from this evaluation-only run.
- Keep adaptive tabular Delta3D as the current preferred 3D model.
- Treat the guarded SNN result as a successful negative experiment: the guard
  prevented the much weaker raw neural branch from causing a large regression,
  but it did not establish a transferable gain.

The small row-weighted R2 point gain is not enough to reverse this decision.
The group-balanced R2 point estimate is negative, both MAE endpoints regress,
and every cross-seed ensemble confidence interval contains zero.

## Artifact status and integrity

The local bundle currently contains 94 files and occupies approximately 11 MB.
Each of the five run directories has 13 published artifacts, and the aggregate
directory has six.

Validated run directories:

1. `run_0_model_42_split_104729`
2. `run_1_model_7_split_130363`
3. `run_2_model_137_split_169087`
4. `run_3_model_2027_split_214087`
5. `run_4_model_9001_split_275015`

Aggregate artifacts:

- `aggregate/_SUCCESS.json`
- `aggregate/aggregate_summary.json`
- `aggregate/cross_seed_oof_predictions.csv`
- `aggregate/per_run_metrics.csv`
- `aggregate/report.md`
- `aggregate/validation.json`

Independent local verification found:

- all five run `_SUCCESS.json` summary hashes match the corresponding
  `summary.json` files;
- all 55 per-run artifact hashes declared by the five summaries match the
  physical files;
- the aggregate summary hash and all four aggregate artifact hashes match;
- `aggregate/validation.json` reports `passed: true`, five expected runs, five
  discovered summaries, five validated runs, and an empty error list;
- all runs have identical dataset, cohort, asset, feature-contract, protocol,
  software, pair-audit, and implementation fingerprints;
- all five training-history files are present, with 9,600 rows each and no
  non-finite `train_loss` values in 48,000 total rows;
- the locally checked implementation files still match the implementation
  hashes recorded by the run.

Important status nuance: every `run_config.json` is a startup snapshot and its
`status` remains `incomplete`. Do not use that field as the completion signal.
For this workflow, completion is established by each run's `_SUCCESS.json` plus
the passing aggregate validation.

Authoritative starting points:

- machine-readable verdict: `runs/simplicial_20260807T200005Z/aggregate/validation.json`
- aggregate statistics: `runs/simplicial_20260807T200005Z/aggregate/aggregate_summary.json`
- concise generated report: `runs/simplicial_20260807T200005Z/aggregate/report.md`
- per-pair ensemble predictions: `runs/simplicial_20260807T200005Z/aggregate/cross_seed_oof_predictions.csv`

## Primary result

The primary comparison is:

```text
guarded SNN hybrid - adaptive tabular Delta3D
```

For reduction metrics, a positive number means lower error for the SNN hybrid;
a negative number means regression.

| Model seed | Split seed | R2 gain | Group-balanced R2 gain | MAE reduction | Macro-group MAE reduction |
|---:|---:|---:|---:|---:|---:|
| 42 | 104729 | +0.001204 | -0.002428 | -0.000181 | -0.000445 |
| 7 | 130363 | +0.001334 | +0.004959 | -0.000077 | -0.000224 |
| 137 | 169087 | +0.003129 | -0.020167 | -0.000720 | -0.003778 |
| 2027 | 214087 | +0.001035 | +0.004013 | -0.000357 | -0.001066 |
| 9001 | 275015 | -0.000036 | -0.005584 | -0.000447 | -0.001391 |

Across the five runs:

| Metric | Mean | Standard deviation | Positive runs |
|---|---:|---:|---:|
| R2 gain | +0.001333 | 0.001141 | 4/5 |
| Group-balanced R2 gain | -0.003841 | 0.010133 | 2/5 |
| MAE reduction | -0.000356 | 0.000249 | 0/5 |
| Macro-group MAE reduction | -0.001381 | 0.001419 | 0/5 |
| Sign-accuracy gain | approximately 0 | 0.002358 | 3/5 |

Cross-seed OOF ensemble, aligned by `pair_id`:

| Metric | Point estimate | 95% paired group-bootstrap CI |
|---|---:|---:|
| R2 gain | +0.001917 | [-0.006264, +0.010758] |
| Group-balanced R2 gain | -0.002466 | [-0.014000, +0.009536] |
| MAE reduction | -0.000386 | [-0.001793, +0.000241] |
| Macro-group MAE reduction | -0.001418 | [-0.003225, +0.000104] |
| Sign-accuracy gain | -0.001850 | [-0.010929, +0.005588] |

No interval excludes zero in the favorable direction. In particular, the
group-balanced and macro-group point estimates are unfavorable.

## Raw SNN diagnostic

The unshrunk SNN branch is substantially worse than adaptive Delta3D in every
seed and every reported endpoint.

Mean unshrunk SNN minus Delta3D effects across the five runs:

| Metric | Mean effect | Positive runs |
|---|---:|---:|
| R2 gain | -0.147360 | 0/5 |
| Group-balanced R2 gain | -0.182142 | 0/5 |
| MAE reduction | -0.018755 | 0/5 |
| Macro-group MAE reduction | -0.025097 | 0/5 |
| Sign-accuracy gain | -0.116374 | 0/5 |

The inner-CV weight search proposed a nonzero raw SNN weight in 10 of 25 outer
folds. The uncertainty guard rejected three of those. The final effective SNN
weight was nonzero in only 7 of 25 folds, and every retained value was 0.25:

| Model seed | Effective weights across the five outer folds |
|---:|---|
| 42 | 0, 0.25, 0, 0, 0 |
| 7 | 0, 0.25, 0, 0, 0 |
| 137 | 0, 0.25, 0, 0.25, 0.25 |
| 2027 | 0, 0, 0, 0, 0.25 |
| 9001 | 0, 0.25, 0, 0, 0 |

Interpretation: the guarded model is mostly an exact Delta3D fallback with a
small neural admixture. The guard is working as a safety mechanism; the neural
representation itself is not yet competitive.

## Cohort and split risk

The frozen cohort contains:

- 5,992 source rows;
- 129 quarantined rows for
  `todga_structure_assigned_to_different_extractant`;
- 3,012 rows after quarantine, target, and complete-condition filtering;
- 1,306 true adjacent-metal pairs before pair-specific exclusions;
- 101 pairs excluded for missing geometry;
- 122 pairs excluded by the unique-replicate policy;
- two pairs excluded for missing selected 3D features;
- 1,081 final pairs;
- 32 extractants;
- 26 exact ECFP clusters;
- 12 adjacent lanthanide pair types.

The grouping is scientifically necessary but strongly imbalanced. The largest
exact-ECFP cluster contains 525 of 1,081 pairs (48.6%) and, for model seed 42,
occupies one entire 525-row outer test fold. The remaining four test folds have
135-142 rows each. This is why row-weighted R2 must not be the only decision
metric; group-balanced R2 and macro-group MAE are primary safeguards against a
large chemistry cluster dominating the conclusion.

The recorded leakage audit passes. Across every outer fold there is zero
train/test overlap for the grouping key and the audited identity/provenance
fields, including exact ECFP cluster, extractant, source ID, geometry key,
geometry feature build ID, and VR graph index.

## Frozen protocol

- Five outer folds and three inner folds.
- Seeded, shuffled `StratifiedGroupKFold`, stratified by adjacent pair label.
- Grouping key: `ecfp_exact_cluster`.
- Shared fold assignments for 2D, tabular Delta3D, and SNN comparisons.
- Complete conditions only; unique-replicate policy; geometry and selected 3D
  features required.
- Evaluation-only: no final deployment model was fitted or saved.
- Inverse group-frequency training weights.
- Exact antisymmetric inference:

  ```text
  (head(z_A - z_B, c) - head(z_B - z_A, c)) / 2
  ```

- Guard candidate weights: 0, 0.25, 0.5, 0.75, and 1.0.
- Guard condition: inner-CV mean group-MAE reduction minus 0.5 standard errors
  must be positive for a nonzero selected weight to survive.
- Paired group bootstrap: 2,000 replicates per run; the aggregate ensemble
  interval was recomputed with 5,000 replicates from aligned OOF predictions.
- Strict deterministic CPU execution with nine threads.

## Features and simplicial assets

- Baseline feature count: 2,130.
- Tabular Delta3D feature count: 2,159.
- Compact invariant 3D block: 29 usable columns, no all-null columns.
- SNN context feature count: 76.
- Required unique VR graphs: 384.
- Graph nodes: 9-10, median 10.
- Graph edges: 23-34, median 29.
- Graph triangles: 15-25, median 21.
- Configured hard caps: 512 edges and 512 triangles per complex.
- Maximum observed edge filtration and triangle side: 3.999475 angstrom.
- Coordination-shell radius: 3.1 angstrom.
- RBF/max-filtration scale: 4.0 angstrom.
- Partial charges disabled.

The assets are far below the configured caps, so this result is not explained
by silent graph truncation.

SNN configuration:

- hidden dimension 64;
- two layers;
- dropout 0.1;
- 16 RBF channels;
- learning rate 0.001;
- weight decay 0.003;
- 160 fixed epochs;
- three neural initializations per fit;
- maximum 64 pairs and 60,000 simplices per batch.

## Frozen fingerprints

All five runs agree on these fingerprints:

| Object | SHA-256 |
|---|---|
| Dataset | `fefbefc6fe993aa9ce9db1a0c338adb9e5f58a8b75bab084cc1df4e024faf5dd` |
| Cohort | `e557b9f5975886ef519b9ac74df2d16be6db1418b38fa236c6a6569cf212d17a` |
| VR asset | `eb7279b5ea78ecc8994e942c1cc15c7b398c904b60d21c591b8f1b93f761ce37` |
| Row-to-geometry map | `70e8fc3151444a1c323b2ce60ed0937dd1cc11dff0d1b89d3776c811d61764fc` |
| Feature contract | `047edcab98fe2eb311f0a9010a4bce8a0ccaea2823ca0c59ea7844b4f69eb362` |
| Protocol | `353eb1723819a24d9d2db969c8eb9735d951c031f2528092d5fc6368e4b98a23` |
| Implementation bundle | `71aa6e8928d06601d474f7c619cfd55925a7a09f0eb448110c13d6304a920e61` |
| Seed plan | `90a9affc70838e3b7e362244d338951f8b625e75b44f45bfe9c7c3dff314455b` |
| Aggregate combined result | `432dd5cd3dc9e0f79021237d88363e783799e897dc1c76ec54d6d112ee8a6f1a` |

Current implementation files were re-hashed locally and match the run snapshot:

| File | SHA-256 |
|---|---|
| `scripts/run_simplicial_benchmark.py` | `ebc4cdd3a4b363cfc345d71e144a9cf477de65b546385e9567cf7aef76421dc5` |
| `src/lanthanide_separation/pairs.py` | `14ec7795cb048672ae3f53a14a01dabdc77d52446e1d881694d270dd113d4fd2` |
| `src/lanthanide_separation/evaluation.py` | `b0edb6c89b678eb4ceff5970445a66417a9ac231e8231610e1b6fb3ac389d1a6` |
| `src/lanthanide_separation/simplicial.py` | `da52460bbb11031f5270b07daa9ce7046c7e77ecc5d6d54d5575bbeb8147b2f2` |
| `src/lanthanide_separation/deep_evaluation.py` | `ce7ce9e74e8f31a67d49b5ddf16cad21a377cfdfa73f48e1a4497a70a7861458` |

## Repository handoff problem

The current `.gitignore` contains a blanket `runs/*` rule. As a result, Git can
carry `_SUCCESS.json` and `run_config.json` only when they are force-added, while
the OOF predictions, summaries, contracts, audits, histories, and aggregate
outputs remain absent after a normal `git pull`.

This caused a real audit failure: the repository initially appeared to contain
completion markers whose declared hashes could not be checked because the
hashed payload was still only on the cluster. The payload is local now because
it was transferred separately, not because Git delivered it.

Claude Code should not assume that ignored training artifacts are disposable.
Before the next handoff, implement one explicit retention strategy:

1. Preferred for this 11 MB bundle: track the immutable published run and
   aggregate artifacts while continuing to ignore attempts, temporary staging
   directories, caches, and routine logs.
2. If run payloads must remain external: publish a versioned artifact archive
   plus checksum manifest and retrieval instructions. Do not commit success
   markers that reference locally unavailable files.

Do not delete or rewrite the current validated bundle while fixing retention.

## Legacy runs are not comparable

Do not pool this result with the older run directories in `runs/`. Those are
legacy tabular Delta3D experiments, not current-protocol SNN runs. They lack the
current completion markers/configuration contract, use the older split protocol,
and use an older compact feature contract. They can be retained for historical
reference but must not enter the current five-seed aggregate.

## Recommended next work for Claude Code

Proceed diagnostics-first. More seeds or more epochs alone are unlikely to fix
a raw branch that loses roughly 0.15 R2 and 0.025 macro-group MAE on average.

1. Preserve this run as an immutable negative baseline.
2. Fix artifact retention/handoff so every success marker is accompanied by
   locally verifiable hashed artifacts.
3. Add a compact audit command that verifies success markers, all declared
   hashes, seed-plan membership, shared fingerprints, expected run count, and
   finite training history in one pass.
4. Export per-fold neural diagnostics: raw and guarded predictions, selected and
   rejected gate weights, loss curves by initialization, prediction variance
   across initializations, target/prediction scale, and error by ECFP cluster.
5. Diagnose representation or objective mismatch before tuning capacity:
   inspect residual scale, branch calibration, pooling collapse, and whether the
   graph embedding adds signal beyond the 29 invariant tabular features.
6. Run single-factor ablations on the same frozen cohort and seed plan. Useful
   first ablations are direct triangle-pooling removal, context-only neural
   control, graph-only control, and a capacity-matched MLP over the 29 compact
   invariants.
7. Keep the matched adaptive Delta3D and matched 2D ensemble controls, exact
   antisymmetry, shared folds, group weighting, and leakage audit unchanged.
8. Do not select a model by row-weighted R2 alone. Report group-balanced R2,
   macro-group MAE, sign accuracy, and the dominant-cluster sensitivity every
   time.

## Acceptance criteria for a successor experiment

Pre-register the decision rule before launching. At minimum, a claimed SNN
improvement should satisfy all of the following on the held-out exact-ECFP-group
protocol:

- positive mean group-balanced R2 gain versus adaptive Delta3D;
- positive mean macro-group MAE reduction;
- no systematic row-level MAE regression across seeds;
- favorable effects across multiple independently shuffled split seeds, not one
  selected seed;
- a cross-seed paired-group-bootstrap interval that excludes zero for at least
  one pre-declared primary group-level endpoint;
- zero audited train/test overlap and identical evaluation pairs for every
  comparator;
- complete `_SUCCESS.json`, contracts, OOF predictions, histories, summaries,
  and aggregate validation available to the reviewer.

Until those conditions are met, the correct conclusion is:

> The compact tabular Delta3D representation remains the preferred model. The
> current simplicial branch does not show robust out-of-group predictive value.
