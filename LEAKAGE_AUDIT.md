# LEAKAGE_AUDIT.md

Audit of the existing `ml_separator` pipeline against the 17-item leakage checklist,
performed **before** any modelling change and **before** any performance number was
computed. Verdicts are evidence-based; every claim below is reproducible from the
commands recorded in `docs/audit_commands.md`.

Cohort under audit: `pair_scope=all`, `require_geometry=True`, `replicate_policy=unique`,
`geometry_descriptor_blocks=global_shape,coordination_shape`,
`include_donor_composition_counts=True` — the configuration the ablation runner uses by
default.

| Quantity | Value |
|---|---|
| Source rows | 5 992 |
| Rows surviving quarantine + finite target + complete conditions | 3 012 |
| Pair rows | 6 699 |
| **Distinct extractants (evaluation units)** | **34** |
| Distinct exact-ECFP clusters | 28 |
| Pair types (strata) | 91 |
| Model features | 2 180 (CONDITIONS 64, LN 8, 2D 2 058, 3D_GLOBAL 12, 3D_LOCAL 38) |

---

## Summary of verdicts

| # | Item | Verdict |
|---|---|---|
| 1 | random `train_test_split` | SAFE |
| 2 | preprocessing before group splitting | SAFE |
| 3 | `StandardScaler` fit on all data | SAFE |
| 4 | imputation fit globally | SAFE |
| 5 | PCA fit globally | SAFE |
| 6 | feature selection before CV | LOW |
| 7 | correlation filtering on all rows | SAFE |
| 8 | target encoding | SAFE |
| 9 | target-derived features | SAFE |
| 10 | **duplicated extractants across folds** | **CRITICAL** |
| 11 | duplicated complexes across folds | SAFE |
| 12 | HPO against outer-test | SAFE |
| 13 | early stopping on outer-test | SAFE (N/A) |
| 14 | model selection on final test performance | MEDIUM |
| 15 | geometry QC thresholds tuned on test performance | LOW |
| 16 | descriptor generation using target information | SAFE |
| 17 | aggregate descriptors using other rows | SAFE |

Additional findings outside the checklist that bear on whether the hypothesis is testable
at all: **A (CRITICAL)**, **B (CRITICAL)**, **C (HIGH)**, **D (LOW)**, **E (HIGH)**.

---

## Item-by-item

### 1. Random `train_test_split` — SAFE
No `train_test_split`, `ShuffleSplit`, or plain `KFold` anywhere in `src/` or `scripts/`.
The only splitter is `StratifiedGroupKFold` inside
[`_group_folds`](src/lanthanide_separation/evaluation.py:224), always called with
`groups=` and `shuffle=True` plus an explicit `random_state`.

### 2. Preprocessing before group splitting — SAFE
All learned preprocessing lives in a `Pipeline` constructed *inside*
[`AntisymmetricExtraTreesRegressor.fit`](src/lanthanide_separation/evaluation.py:113).
It is instantiated fresh per fit (`_new_pipeline`), so no state survives across folds.

### 3 / 5. Global `StandardScaler` / PCA — SAFE
Neither exists. The model family is tree-based and needs no scaling; there is no
dimensionality reduction step to leak.

### 4. Global imputation — SAFE
`SimpleImputer(strategy="median", add_indicator=True)` is the first pipeline step and is
therefore fitted on the training fold only. The runner additionally persists
`imputer_statistics` and their SHA-256 per arm per fold
([`ablation.py:678`](src/lanthanide_separation/ablation.py:678)), so the claim is auditable
after the fact rather than merely asserted.

### 6. Feature selection before CV — LOW
Feature sets are **pre-declared**, not data-driven: `FEATURE_FAMILIES` → `ABLATION_FAMILIES`
A0–A6 in [`feature_registry.py`](src/lanthanide_separation/feature_registry.py:57), with
fail-closed classification (an unrecognised column raises rather than being silently
absorbed). No supervised selection occurs at any point.

Two whole-cohort, **target-independent** decisions remain:
- `usable_3d_cols` drops columns that are all-NaN across the entire dataset
  ([`pairs.py:581`](src/lanthanide_separation/pairs.py:581));
- `require_complete_3d` drops pairs with any missing selected 3D contrast (6 pairs here).

Both are computed without the target and applied identically to every arm, so they cannot
bias A2-vs-A5. They do define the cohort, which is a generalisation-scope caveat
(§ Scope) rather than leakage. **LOW.**

### 7 / 8. Correlation filtering, target encoding — SAFE
Neither appears. No `.corr()` call, no `TargetEncoder`.

### 9. Target-derived features — SAFE
`build_lanthanide_pair_dataset` enforces a forbidden-fragment check over the *entire*
model contract and raises `AssertionError` if `log_D`, `log_SF`, `safe_exp_id`, `build_id`,
`geometry_key`, `sample_weight`, `asset_index` or `xyz_path` ever appears
([`pairs.py:780`](src/lanthanide_separation/pairs.py:780)). `feature_registry._classify_column`
repeats the check independently. The one feature that mixes a physical constant into a
geometry value — `derived_invariant__shell_clearance_mean = ln_donor_distance_mean −
Ionic Radius_metal` — uses a fixed tabulated ionic radius, not a fitted quantity.

### 10. Duplicated extractants across folds — **CRITICAL**

Literal duplication is correctly prevented: `ablation.py` asserts zero overlap on
`extractant`, `source_id`, `geometry_key`, `geometry_feature_build_id` and `vr_graph_index`
and raises if any is non-zero ([`ablation.py:471`](src/lanthanide_separation/ablation.py:471)).
Measured overlap is `[0, 0, 0, 0, 0]`.

**But chemical duplication is not prevented under the default protocol.** The runner
defaults to `--group-mode extractant`, i.e. groups are literal `canonical_smiles`.
In the full dataset, 18 exact-ECFP clusters contain more than one distinct SMILES —
homolog series that differ only in alkyl chain length, which Morgan fingerprints of this
radius cannot see. Examples:

```
7 SMILES in one cluster:  TODGA (C8) + C6 / C10 / C12 / mixed-chain diglycolamides
4 SMILES in one cluster:  methyl-substituted diglycolamide stereo/chain variants
```

In the modelling cohort this produces, at `split_seed=42`:

```
ECFP cluster 0982ebf465b7 : 5 extractants spread across outer folds 0, 1, 3, 4
                            rows affected: 3 334 / 6 699  = 49.8 % of the cohort
```

For half the cohort, the held-out "unseen extractant" has a training-set partner with a
**bit-identical 2 048-bit ECFP fingerprint**. Every ECFP feature — 2 058 of the 2 180 model
columns — is therefore literally copied between train and test for those rows. This does
not violate the letter of leave-extractants-out, but it directly violates the hypothesis's
requirement of *"completely unseen extractants"*, and it inflates the A2 (2D) baseline
specifically, because ECFP is where the copying happens.

The stricter grouping already exists (`--group-mode ecfp-exact-cluster`) and is provably
coarser: each `canonical_smiles` maps to exactly one ECFP (verified — 0 SMILES with >1
fingerprint), so ECFP clusters are unions of extractant groups. **It must be the primary
protocol, not a sensitivity.** The README's claim that ECFP-cluster holdout is "a stricter
variant of leave-extractants-out" is correct; the defect is that the ablation runner does
not default to it.

### 11. Duplicated complexes across folds — SAFE
Asserted zero for `geometry_key_{A,B}`, `geometry_feature_build_id_{A,B}` and
`vr_graph_index_{A,B}`, and the pair→asset chain is independently verified in
`attach_geometry_descriptors`, which cross-checks `vr_graph_index` against the descriptor
asset and raises on any disagreement.

### 12. Hyperparameter optimisation against outer-test — SAFE
Selection uses `inner_macro_group_mae` computed from inner-CV out-of-fold predictions on
the outer-training rows only ([`ablation.py:623`](src/lanthanide_separation/ablation.py:623)).
Outer-test rows are touched exactly once, by `final_model.predict(outer_test)`, after the
arm is fully specified. Tie-breaking is deterministic (`candidate_index`), so no hidden
test-driven choice enters.

### 13. Early stopping on outer-test — SAFE (not applicable)
ExtraTrees has no iterative stopping; `n_estimators` is fixed a priori. No
`early_stopping` / `n_iter_no_change` / `validation_fraction` anywhere.

### 14. Model selection based on final test performance — MEDIUM
Within a single run the protocol is clean. The risk is at project level:

- `runs/` contains 20 completed runs of the **legacy** Delta3D/simplicial paths under many
  different configurations (`compact` / `strict` / `full` / `shrinkage` / `sensitivity_unique`
  / `sensitivity_ecfp_cluster`, seeds 7 / 137 / 104729).
- The pre-specified A0–A6 ablation — the experiment that actually addresses the hypothesis —
  **has never been executed**: there is no ablation run directory at all.

So the analysis space has been explored repeatedly under the older framing while the
pre-registered comparison remains un-run. That is not leakage in the executed code, but it
is exactly the condition under which post-hoc protocol selection happens. Mitigation:
freeze the protocol (done — see `validation.json` `scientific_protocol_sha256`) and report
the first executed A0–A6 result whatever it shows. **MEDIUM.**

### 15. Geometry QC thresholds tuned on prediction performance — LOW
QC is structural and upstream. `accepted_geometries.csv` records `qc_class=OK` for all
1 155 accepted geometries with `qc_note=nearest_coreCN_qc_ok`; `excluded_geometries.csv`
records 101 rejects with explicit structural reasons:

```
BORDERLINE_AMBIGUOUS_SHELL 40 | FAIL_LONG_BOND 26 | UNAVAILABLE 23 | BORDERLINE_LONGISH 12
```

The criteria are coordination-shell geometry (longest Ln–donor bond, gap to the next
donor), not model error, and the rejection happened when the bundle was built — before any
model in this repo. No code path in `src/` or `scripts/` reads a metric and adjusts a QC
threshold. Residual risk is that the upstream generation pipeline is not in this repository
and cannot be re-executed here, so the *absence* of performance-driven tuning is inferred
from the recorded reasons rather than proven. **LOW.**

### 16. Descriptor generation using target information — SAFE
`MetalSiteDescriptorBuilder` takes exactly one input: the Vietoris–Rips coordinate asset
(`vietoris_rips_inputs.npz`). Grep for `log_D` / `log_SF` / `target` in
`geometry_descriptors.py` returns nothing. Descriptors are functions of coordinates alone.

### 17. Aggregate descriptors computed from other rows — SAFE
Every derived feature in `_add_derived_invariant_3d_features` is a **row-wise** reduction
(`axis=1`) over that row's own ranked donor arrays — quantiles, Legendre moments, spans.
No `groupby().transform()`, no dataset-level mean, no neighbour pooling.

The one aggregation that exists is the replicate median inside
`(canonical_smiles, conditions, metal)` cells. That cell is nested strictly inside one
extractant, so it can never combine a training extractant with a test extractant.

---

## Findings outside the checklist

### A. 3D enters the model **only as an A−B difference** — CRITICAL (scope, not leakage)

In `build_lanthanide_pair_dataset`:

- 2D and conditions enter as **absolute** values: `base__X = X_A` ([`pairs.py:724`](src/lanthanide_separation/pairs.py:724))
- 3D enters **only** as a contrast: `delta3d__X = X_A − X_B` ([`pairs.py:728`](src/lanthanide_separation/pairs.py:728))

Because A and B are the *same ligand with a different lanthanide*, the difference removes
almost everything about the coordination environment itself and retains only how that
environment responds to swapping the metal. Measured consequences in the D3
(coordination number / donor composition) block:

| feature | sd of the Δ | % exactly zero |
|---|---|---|
| `donor_count_P` | 0 | 100 % |
| `donor_count_S` | 0 | 100 % |
| `donor_count_other` | 0 | 100 % |
| `donor_count_N` | 0.357 | 85.7 % |
| `coordination_number` | 0.498 | 54.7 % |
| `donor_count_O` | 0.718 | 54.6 % |

Three of the seven "donor composition" features are **identically zero for every row** —
they are not weak features, they carry no information whatsoever. Coordination number is
zero for the majority of pairs.

The stated hypothesis is about *"metal-centered 3D coordination information"*. As encoded,
the pipeline can only test the far narrower claim *"the metal-induced **change** in
coordination geometry"*. Absolute coordination chemistry — CN 8 vs 9, O-donor vs N-donor
ligand, polyhedron size — is structurally invisible to A5 and A6.

This is fixable within the existing antisymmetry contract and **without weakening it**.
`reverse_pair_features` negates `delta3d__*`, swaps `pair__{Z,ionic_radius}_{A,B}`, and
leaves every other column untouched — which is precisely correct for a swap-**symmetric**
feature. The LN family already exploits this with `pair__Z_mean` and
`pair__ionic_radius_mean`. The symmetric 3D analogue `sym3d__X = (X_A + X_B)/2` is
invariant under A↔B, so exact antisymmetry of the prediction is preserved.

I am recording this **before running any model**, so the extension arm is motivated
structurally, not by looking at results. It is added as a *declared additional arm*
(A5s / A6s), never as a replacement for the pre-specified A5 / A6, which are reported as-is.

### B. One extractant is 44 % of the cohort and forms an entire outer fold — CRITICAL

```
pairs per extractant:  min 1   median 91   max 2 921
top 5 extractants  =  63 % of all pair rows
outer fold sizes (seed 42, group=extractant):
  fold0  956 rows /  9 groups
  fold1  916 rows / 10 groups
  fold2  915 rows /  7 groups
  fold3 2921 rows /  1 group    <-- 43.6 % of the cohort, a single ligand
  fold4  991 rows /  7 groups
```

Fold 3 is one extractant. Its fold-level metrics describe that one molecule; the global OOF
metric is 44 %-determined by it. Checked across `split_seed ∈ {0, 7, 42, 137, 1009, 104729}`:
the partitions genuinely differ, but **the giant extractant is alone in its own fold in every
seed**. Seed variation reshuffles the other 33 extractants and leaves the dominant structure
untouched, so multi-seed averaging cannot average this away.

### C. Effective sample size is ≈ 4–5 groups, not 6 699 rows — HIGH

Kish effective group count `1/Σpᵢ²` over row shares:

```
group = extractant           n = 34   n_eff = 4.8
group = ecfp_exact_cluster   n = 28   n_eff = 3.8
```

Any confidence interval, bootstrap or paired test must treat the extractant as the sampling
unit — which `_bootstrap_comparisons` correctly does via `delta_macro_group_mae`, giving each
sampled group one vote. With n_eff ≈ 4, that interval will be wide, and it should be: the
experiment has little power to resolve a small 3D effect. This is a property of the data, not
a bug, but reporting a row-level CI here would badly overstate confidence.

### D. Three constant features are carried as if informative — LOW
`donor_count_{P,S,other}` have zero variance (§A). They are harmless to a tree but they
inflate the reported D3 block size from 4 informative columns to 7, and would distort any
per-block importance normalisation. They should be marked *unavailable* rather than passed
through — matching the brief's rule that an uncomputable descriptor is declared, not silently
emitted.

### E. Required baselines are absent — HIGH (specification gap)
`DummyRegressor`, a regularized linear baseline, `permutation_importance` and SHAP do not
appear anywhere in `src/` or `scripts/`. The brief requires the trivial and linear baselines
in every comparison; without them, "3D helps" cannot be separated from "this whole feature
set barely beats predicting the mean". Being added as part of this work.

---

## Scope caveats (not leakage, but they bound the claim)

1. **48 % of rows are dropped for incomplete conditions.** `incomplete_condition_rows_seen =
   2 851` of 5 863. Requiring all 64 `cond__` values to be observed is defensible for an
   exact-match difference target, but it is what collapses 190 extractants to 34. The
   conclusion generalises to well-annotated extraction systems, not to the literature at large.
2. **The cohort is conditioned on 3D availability** (`require_geometry=True`). Correct for a
   paired 2D-vs-3D comparison — both arms see identical rows — but it means A2 here is not the
   2D model's performance on the full 2D-available data.
3. **Target provenance.** As the README already states, no `publication_id` /
   `experiment_series_id` exists, so `log_SF = log_D_A − log_D_B` is a condition-matched proxy;
   two `log_D` values matched on all observed conditions may still come from different
   experimental series.

---

## Required protocol changes before the result can be believed

1. **Primary grouping must be `ecfp_exact_cluster`, not `extractant`** (Finding 10). Report
   literal-extractant grouping as the *weaker* sensitivity, not the headline.
2. **Report the macro-per-group statistic as primary** (Finding B/C); never the row-level R²
   as the headline, and never a row-level CI.
3. **Add DummyRegressor + regularized linear baselines** to every arm (Finding E).
4. **Add the declared symmetric-3D arms A5s/A6s** alongside — not instead of — A5/A6
   (Finding A), and report both.
5. **Mark zero-variance descriptors unavailable** (Finding D).

Items 1–5 are implemented in the accompanying work; the pre-specified A0–A6 result is
reported unchanged next to them.

---

## Isolation of the generation-2 negative controls

Every permutation control is a *training-time* transformation. Two independent mechanisms,
audited separately, keep them from becoming leakage in their own right.

**Tabular blocks** (`G2`, `G4`, `E2`, `E3`, `E4`, `S4`, `S5`, and the legacy `A5`). The
shuffler receives only the current inner- or outer-training row index set. It moves whole
complex-pair feature vectors between recipients within one `pair_label`, so a geometry reused
by several rows stays internally consistent, and dimensionality, marginal distributions and
model capacity are unchanged. `shuffle_audit.csv` records `test_rows_touched` per application;
the assertion is `0`, not "expected to be 0".

**Learned geometry** (`simplicial_shuffled_s*`). Same principle one level lower: rather than
permuting a feature matrix, `permuted_geometry_view` permutes the `(complex_A, complex_B)`
*assignment* over training rows within one `pair_label`, then re-checks every row outside the
training index set and raises if any changed. Held-out rows keep their true geometry, so the
control scores a permutation-trained model on the untouched cohort — the same held-out rows,
the same architecture, the same epoch count and the same number of initialisations as the real
arm. `summary.geometry_null_audit` carries the per-fold record.

Both are read the same way: an arm that does not beat its own permuted twin is **unsupported**,
regardless of how it compares to `A2`.
