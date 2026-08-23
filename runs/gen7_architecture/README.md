# gen7 — can architecture break the gen6 zero-shot ceiling?

gen6 settled that chemical coverage helps. gen7 asks the complementary question: with
the rows we already own, how much of the remaining held-out-chemotype error can a
better representation, objective, decomposition, learner or inductive bias remove?

## The evaluation contract

Every contender in this directory is scored on **byte-identical test rows**:

* one shared cohort, built once through the gen5 builder at `min_cells = 3` —
  5,248 rows / 152 extractants / 131 ECFP clusters / 79 Tanimoto-0.7 chemotypes;
* folds hold out whole Tanimoto-0.7 chemotypes, using gen5's `seeded_group_kfold`
  reused byte-for-byte;
* metrics come from `gen6.metrics`, unchanged — macro MAE is still one ECFP cluster,
  one vote, and offset/shape still satisfy the SSE identity;
* the learner's `random_state` is `42 + fold*1009 + 9_999_991`, derived from a fixed
  model seed and *not* from the split seed.

**Reproduction check.** `TREE_MC_lig2d_ext_massaction` returns macro 1.067993 /
offset 0.936996 / shape 0.502044 on split seed 104729 — matching
`runs/gen6_expA_5seed/arm_metrics.csv` to every printed digit. The harness is a
faithful superset of gen6's, not an approximation of it.

The cohort fingerprint hashes only `(row_id, log_D)`, never the column count, so
adding a feature block cannot change it. The contract is *identical rows and targets*,
not *identical columns* — the latter would forbid the comparison gen7 exists to make.

## Layout

```text
README.md                      this file
environment.json               interpreter, packages, the macOS OpenMP fix
data_audit.json                cohort composition, blocks, fold plan, fingerprint
model_registry.json            every contender, generated from the code that ran
experiment_manifest.json       every run directory with per-file hashes
literature_review.md           §19 — methods surveyed, with test / don't-test calls
recovered_variables_raw.md     §2 — the upstream column-by-column audit
ambiguity_audit.md             the 17 multi-name canonical SMILES, resolved
geometry_audit.md              §6 — what the 1,155 Architector structures actually are
leaderboard_all.csv            every arm, every suite, one table
all_scores_by_seed.csv         the same, per split seed
oracles/                       how much is left to win (arms given what no model knows)
indicators/                    what the missing-value flags are actually worth
metal/                         §4 — one-hot vs continuous vs structured lanthanide
ligphys/                       §8 — physically motivated level features
learners/                      §13 — the learner benchmark on frozen features
representations/ embeddings/   §5 — descriptors, fingerprints, pretrained encoders
kernels/                       §9 — kernel ridge and an exact Tanimoto GP
hnn/                           §3/§7 — the offset/shape network and its knobs
level_benchmark/               the level sub-problem, isolated to ~120 samples
finalists/                     the survivors, five seeds, full metrics
ablations/                     §16 — leave-one-component-out with paired bootstrap
ensemble/                      §17 — residual correlations and honest stacking
kshot/                         §12 — few-shot calibration against a no-model null
error_analysis/                §18 — the worst ligands and their failure mechanism
ceiling/                       the batch-offset variance components and the floor
decision_report.md             what we concluded and why
```

## Reproducing

```bash
.venv/bin/python scripts/run_gen7_experiment.py --suite oracles --seeds 3
.venv/bin/python scripts/run_gen7_experiment.py --suite learners --seeds 3
.venv/bin/python scripts/gen7_level_benchmark.py --seeds 3 --split chemotype
.venv/bin/python scripts/gen7_ablation.py --seeds 3
.venv/bin/python scripts/gen7_collect.py
```

Two environment notes that are not optional on macOS: LightGBM and XGBoost each ship
their own `libomp`, and a process that has already initialised torch's deadlocks at
0 % CPU when the second of them starts a thread pool. `environment.json` records the
rpath fix. `OMP_NUM_THREADS=4` is set by the runner before either is imported.
