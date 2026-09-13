# gen12_eu_pred_2 — unfinished early draft of the Gen12.2 code (superseded, no results)

This directory is an early code draft for Gen12.2, not a generation of its own. It holds a
single package, `gen12eu2`, with two jobs. First, it rebuilds Gen12's frozen europium cohort and
design-B chemotype hold-out folds and checks them against Gen12's saved fold plan. Second, it
defines the extractant level `alpha_i` in `y_ij = alpha_i + g(c_ij) + eps_ij` in five candidate
ways, each computed inside a fold from that fold's training rows. Both jobs serve Gen12.2's
question: can structure-based descriptors predict the intrinsic Eu extraction level of extractants
the model has never seen? The directory neither asks nor answers it. There are no scripts,
pre-registration or reports, and the draft produced no numbers.
**Cite every Gen12.2 result from [`gen12_2_eu_pred/`](../gen12_2_eu_pred/README.md).**

## Status

- **Superseded and unreviewed.** It was added in commit b7d0c7f (2026-09-07) in the same commit as
  `gen12_eu_pred/` and `gen12_2_eu_pred/`, with no report, pre-registration, test or review record.
  Git cannot tell whether this draft or the finished study came first. Commit 155dc6c (2026-09-13)
  moved it into `generations/` without changing content.
- Calling it a draft of `gen12_2_eu_pred` is inferred from the code. The package docstring has Gen12.2's
  title, and `paths.py` names its root `GEN122_ROOT`. The successor's `gen122/levels.py` lists the same five
  definitions in the same order, renamed `LVL_MEAN`, `LVL_MEDIAN`, `LVL_FE_INTERCEPT`, `LVL_SHRUNK` and `LVL_COND_RESIDUAL`.

## Read first

1. `gen12eu2/__init__.py`, `gen12eu2/gen12_bridge.py`, `gen12eu2/level_targets.py` (their docstrings state the intent).
2. The successor: [`README.md`](../gen12_2_eu_pred/README.md), then the docstring of
   [`gen122/levels.py`](../gen12_2_eu_pred/gen122/levels.py), then [`DATA_AUDIT.md`](../gen12_2_eu_pred/DATA_AUDIT.md) §7
   and [`PRE_REGISTRATION.md`](../gen12_2_eu_pred/PRE_REGISTRATION.md) Addendum 1.

## Key files

- `gen12eu2/gen12_bridge.py`: `load_design_b()` rebuilds the cohort and design-B folds through `gen12eu`.
  It refuses to continue unless the cohort fingerprint is `2a364bb5264e9935` and the seed, fold order, test row ids
  and held-out chemotypes all match `gen12_eu_pred/predictions/B/fold_plan.json`. It also checks
  `predictions/B/similarity.parquet` against a rebuilt table. `gen12_predictions(arm)` loads a frozen Gen12
  prediction parquet from `B/`, `B_ablation/` or `B_multiln/`.
- `gen12eu2/level_targets.py`: `LevelDefinitions` computes `raw_mean`, `median`, `fe_intercept` (a
  within-extractant-demeaned ridge, `RIDGE_LAMBDA = 1.0`, on the COND and MASSACT blocks), `shrunk_mean`
  (James-Stein shrinkage towards the training grand mean) and `resid_condonly` (the mean residual after a
  global condition-only ExtraTrees, 300 trees, fitted in-sample).
- `gen12eu2/paths.py`: puts `src/` and `generations/gen12_eu_pred` on `sys.path`. **Importing it creates** the
  ten empty, untracked output directories here (`analysis/` through `splits/`). `assert_gen12_untouched()`
  is meant to re-hash Gen12's manifest artefacts through `generations/verify_relocation.py`, but raises KeyError (see caveats).

## How to run (from the repository root)

There are no entry scripts, and none of these commands was run for this README.

```bash
# the only test touching this unit (post-move path resolution)
PYTHONPATH=src .venv/bin/python -m pytest generations/tests/test_generations_relocation.py -q

# example import of the bridge (needs the post-move paths.py fix; creates the output dirs if missing)
PYTHONPATH=generations/gen12_eu_pred_2 .venv/bin/python -c \
  "from gen12eu2 import gen12_bridge; fd = gen12_bridge.load_design_b(); print(len(fd.folds))"

# the finished level-definition study that replaces level_targets.py. It OVERWRITES committed files:
# gen12_2_eu_pred/analysis/level_definition_study.md and manifests/level_definition_choice.json
PYTHONPATH=generations/gen12_2_eu_pred .venv/bin/python generations/gen12_2_eu_pred/scripts/g122_level_definition_study.py
```

## Tests

None in this directory. `generations/tests/test_generations_relocation.py` only checks, in a subprocess
with `mkdir` disabled, that `gen12eu2.paths` resolves `REPO_ROOT` to the repository root and `GEN12_ROOT`
to `generations/gen12_eu_pred`. No test covers the fold checks or the level definitions.

## Dependencies

- `generations/gen12_eu_pred`: `gen12eu.splits`, `.chemistry`, `.cohort` and `.preprocess`; `predictions/B/fold_plan.json`,
  `predictions/B/similarity.parquet`, `predictions/{B,B_ablation,B_multiln}/<arm>.parquet`, `manifests/manifest.json`.
- `src/`, and the data Gen12's cohort builder reads relative to the repository root
  (`dataset with 3D structures/`, `runs/`, `dataset_all_metals/`; see `generations/RELOCATION.json`).
- `generations/verify_relocation.py`.

## Caveats and later corrections

- **Commit 155dc6c alone breaks `paths.py`.** There `REPO_ROOT = GEN122_ROOT.parent` became `generations/`, so
  `SRC_ROOT` points to a non-existent `generations/src`. The follow-up relocation commit adds one `.parent` to
  `REPO_ROOT` and `generations/` to `GEN12_ROOT`.
- **`assert_gen12_untouched()` cannot complete.** It indexes `meta["blake2b_128"]` for every artefact, and Gen12's
  manifest records the four `figures/fig*_B.png` entries by size only, so it raises KeyError 'blake2b_128' (a defect
  already present at fdb1e15). Nothing calls it. Use `.venv/bin/python generations/verify_relocation.py` to check Gen12 instead.
- **Import order (from reading the code, not running it).** Neither `level_targets.py` nor `__init__.py` imports `paths`
  before `gen12eu.preprocess` is needed. Importing `gen12eu2.level_targets` alone therefore fails unless `gen12eu2.paths`
  or `gen12_bridge` has already been imported, or `generations/gen12_eu_pred` is on `PYTHONPATH`.
- **`resid_condonly` falls into the two-stage residual trap.** Its condition model is fitted in-sample with no
  cross-fit. The successor's `levels.py` cross-fits over training chemotypes to avoid exactly this, and the
  draft's own docstring already expected this definition to be distorted.
- **The draft and the successor would give different numbers.** Draft `fe_intercept` is a demeaned, extractant-balanced
  ridge; the successor's is a pooled ridge with one dummy per extractant and penalised slopes only. Draft
  `tau2` is `var(means) - sigma2*mean(1/n)` with a 1e-3 floor. Draft `per_extractant()` also shrinks held-out rows,
  which the successor never does to the evaluation target. No `gen12_2_eu_pred` level number describes this code.
- **Correction from the successor.** DATA_AUDIT §7 and PRE_REGISTRATION Addendum 1 (2026-09-05) cover the
  *cross-fitted* condition-residual level. Every number below comes from training rows only of one development fold
  (design B, split seed 104729, fold 0; 1148 rows, 153 extractants). The condition-only model explains 8.9 % of
  pooled row variance. Yet its per-extractant mean prediction carries 43.9 % of the between-extractant variance and
  correlates with the level at Spearman 0.55, so subtracting it removes real level. Its split-half reliability
  is Spearman 0.841, against 0.920 for the raw mean, over the 53 extractants with at least four cells. `LVL_MEAN` therefore
  became primary and `LVL_COND_RESIDUAL` secondary, before any level model was fitted. The draft's in-sample version was never evaluated.
- Later Gen12.2 corrections, such as the four post-hoc descriptor defects in `gen12_2_eu_pred/DECISION_REPORT.md`
  §13 item 14, apply to `gen12_2_eu_pred`, not to this code. No later generation and no file under `docs/` cites this directory.
