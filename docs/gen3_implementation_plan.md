# Generation 3 experiment layer: implementation plan

Generation 3 is an additive experiment layer. It does not change the generation-2 pair
cohort, feature-family definitions, frozen split result, model artifacts, or aggregate
reports. The authoritative generation-2 reference is the completed all-pairs run rooted at
`runs/ablation_all_20260810T181058Z/`.

## Reuse from the existing pipeline

- `pairs.py`: build the unchanged 6,699-row all-pairs cohort and emit the existing A2 and
  electronic contracts.
- `feature_registry.py`: obtain A2, `ELEC_PAIR`, and `ELEC_COMPLEX` columns without adding a
  geometry family.
- `evaluation.py` / `ablation.py`: reuse target-independent `StratifiedGroupKFold` plans,
  pair reversal, the current antisymmetric ExtraTrees A2 champion, fold-local preprocessing,
  training-only block shuffling, and provenance-overlap assertions.
- `run_ablation_benchmark.py` / `aggregate_ablation_runs.py`: reuse atomic artifact writes,
  SHA-256 manifests, fail-closed validation, long OOF tables, and extractant-multiplicity-safe
  paired bootstrap conventions.

## New isolated components

1. A gen3 protocol loader validates that the committed protocol is internally consistent and
   that the dataset and generation-2 feature registry match their frozen hashes.
2. A gen3 adapter exposes A2, raw E3, compact electronic response, extractant-family labels,
   and shoulder-specific A/B inputs. Compact features use only existing electronic means and
   differences; no target-derived feature is allowed.
3. H1 evaluates antisymmetric CatBoost on the exact A2 columns with fold-local `none`,
   `group_sqrt`, and `group_equal` weights. Its fixed two-stage budget screens all nine
   weight/loss arms with two anchor configurations, then evaluates two expanded endpoint
   configurations only for the best three arms inside each outer-training set. This is 24
   candidate evaluations (72 three-fold CatBoost fits) per outer fold, with no adaptive
   changes between split seeds.
4. H2 cross-fits Stage 1 before forming every residual, tunes the correction head and lambda
   inside the outer training set, and builds real and training-only shuffled controls from the
   identical procedure. Stability selection is nested and is reported as a separate follow-up
   arm after the full-E3 arms.
5. H3 uses one shared scalar scoring network `g(context, Ln[, electronic])`; prediction is only
   `g(A) - g(B)`. No unconstrained pair head is permitted.
6. Per-run output stores folds, cross-fitted predictions, tuning, weights, shuffles, selected
   features, leakage assertions, metrics, validation, hashes, and a report. Aggregation accepts
   only hash-valid runs matching the frozen protocol and computes the predeclared multi-split
   leaderboard and paired extractant bootstrap.

## Verification before a scientific run

- Install `requirements-gen3.txt` under Python 3.11.11 on Linux x86_64. The exact
  NumPy/pandas/SciPy/scikit-learn versions match the validated generation-2 cluster
  environment; the aggregator rejects runs whose implementation or software contracts differ.
- Unit tests cover weight normalization, group separation, residual cross-fitting provenance,
  training-only shuffling, bootstrap multiplicity, compact-feature equations, and exact H3
  antisymmetry/transitivity.
- A quick synthetic/in-repository smoke run must finish with `validation.json: passed=true`.
- SLURM submission is dry-run first. No real cluster job is submitted as part of implementation.
