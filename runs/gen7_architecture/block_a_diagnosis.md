# gen7 Block A — forensic diagnosis of the current error

**Purpose.** Establish, before any gen7 architecture change, exactly what the task is, how it is
scored, what the numbers to beat are, where the error actually sits, and what would invalidate a
gen7 result. Every number below is traced to a file. Where a number is *derived*, the arithmetic is
shown. Where a claim is unverified, it is labelled.

**Status of the evidence.** gen5 and gen6 numbers are final (5 seeds). The gen7 numbers in
`runs/gen7_architecture/` are **in-flight**: `learners`, `oracles`, `indicators` and `recovered` ran
at **3 seeds** (104729, 130363, 155921); `representations` and `embeddings` ran at **1 seed**;
`metal`, `level_benchmark`, `screen_kernels` and `screen_hnn` are **empty**. No arm *ordering* in
gen7 is yet believable at the 0.01–0.04 resolution that matters. Treat the gen7 leaderboards as
*direction-finding*, not as results.

---

## 1. The task

**Target.** `y = log_D = log₁₀(D)`, the distribution coefficient (metal in organic phase / metal in
aqueous phase), one value per **(extractant, condition-vector, metal)** cell.
`LEVEL_TARGET_COLUMN = "log_D"` — `src/lanthanide_separation/levels.py:52`.

This is the **level** — how much metal is pulled into the organic phase — as opposed to gen2–gen4's
**pair** target `log_SF_A_over_B = log D_A − log D_B`, which cancels the extractant's absolute level
by construction. All errors are in log₁₀ units. Target sd on the gen7 cohort = **1.6574**
(`runs/gen6_expA_5seed/summary.json` → `cohort_audit`; gen5's 91-extractant cohort had 1.6426).

**Prediction unit.** A row is one replicate-averaged cell. There is no ranking or classification
objective; the deliverable is an absolute number a chemist can act on, plus (not currently produced)
an honest statement of when not to trust it.

---

## 2. The evaluation contract, exactly

### 2.1 Cohort

Built once by `levels.build_level_dataset(min_rows_per_extractant=3, replicate_policy="mean",
drop_below_log_d=-6.0)` — `src/lanthanide_separation/gen7/harness.py`, `GEN7_MIN_CELLS = 3`.

| quantity | value | source |
|---|---|---|
| rows | **5,248** | `runs/gen7_architecture/oracles/summary.json` |
| cohort fingerprint | `bed178ec1a7a82b0` | same |
| extractants | 152 | `runs/gen6_expA_5seed/cohort_comparison.json` |
| ECFP clusters (scoring units) | **131** | same |
| Tanimoto-0.7 chemotypes (fold + bootstrap blocks) | **79** | same |
| conditions / series / metals | 2,055 / 303 / 14 | `levels.build_level_dataset` audit |
| largest extractant share of rows | 28.35 % | `runs/gen6_expC_5seed/summary.json` |
| largest ECFP cluster share | 37.96 % | same |
| **largest chemotype share** | **63.64 %** | same |

`min_rows_per_extractant` is a minimum number of **cells**, not rows (`levels.py:431` counts
`drop_duplicates(["extractant","condition_id","metal_symbol"])`). The name and the CLI flag both say
"rows". At `min_cells=10` this cohort is BASE: 4,881 rows / 91 extractants / 74 clusters / 40
chemotypes.

### 2.2 Folds

`gen6.cohorts.seeded_group_kfold(groups=tanimoto_cluster, n_splits=5, seed=<split seed>)` —
shuffle unique chemotype labels with `np.random.default_rng(seed)`, deal round-robin,
`fold_of[g] = i % 5`. Deliberately **not** `sklearn.GroupKFold`. Five split seeds
`(104729, 130363, 155921, 196613, 262147)`. Forest seed per fold:
`random_state = 42 + fold*1009 + 9_999_991`.

**The folds are wildly unbalanced by construction.** At seed 104729 test sizes are
382 / 3,753 / 286 / 501 / 326 — one fold is 71.5 % of the cohort and trains on 1,495 rows, because
the 63.6 %-of-rows chemotype falls entirely in it. Every per-fold number and the OOF pool itself are
dominated by that fold.

### 2.3 Metrics

- **Primary: `macro_mae`** = mean over the **131 ECFP clusters** of each cluster's row-mean absolute
  error (`gen6/metrics.py:656`). One ligand ≈ one vote (89.3 % of clusters hold exactly one ligand).
- **`offset_mae` / `shape_mae`** (`gen6/metrics.py:198–200`), per held-out ligand *l* with
  `e_i = ŷ_i − y_i`:
  - `b_l = mean(e)`, `offset_mae = mean_l |b_l|` — **equal weight per LIGAND (152 units)**
  - `yc = y − ȳ_l`, `pc = p − p̄_l`, `shape_mae = mean_{n_l≥2} mean_i |pc − yc|`
  - `shape_r2 = 1 − Σ(pc−yc)² / Σ yc²`
  - Identity: `sse_total = sse_centred + Σ_l n_l·b_l²`, unit-tested to 3.6e-15.
- **Novelty endpoints**: `nn_train_tanimoto < 0.4` and `< 0.6`, strict `<`, **nested not disjoint**,
  computed with `exclude_self=False` against the fold's whole training set.

### 2.4 Uncertainty

`paired_unit_bootstrap` (`gen6/metrics.py:465`): 5,000 replicates, seed 8675309, one shared index
matrix; **scoring unit = ECFP cluster (131), resampling block = chemotype (79)**. `_robust_intervals`
adds a cluster-robust *t* interval and BCa with jackknifed acceleration over blocks.

---

## 3. Reference numbers to beat, and where they come from

### 3.1 The incumbent

| endpoint | value | source |
|---|---|---|
| **unseen_chemotype macro MAE** | **1.0468** (offset 0.8744, shape 0.5069) | `runs/gen6_expC_5seed/arm_metrics.csv`, `MONO_ET` = `MC_lig2d_ext_massaction`, EXPANDED, 5 seeds |
| nn<0.4 | 1.0429 (offset 0.9437, shape 0.3782) | `runs/gen6_expA_5seed/hard_chemistry_metrics.csv` |
| nn<0.6 | 1.1601 | same |
| unseen_ligand | 0.8526 | `runs/gen6_expC_5seed/arm_metrics.csv` |
| unseen_series | 0.8080 | same |
| beaten baselines | BASE 1.2100, SHUFFLED 1.2193, MONO_RIDGE 1.1476, C2_TWO_STAGE 1.1910 | `runs/gen6_expA_5seed/arm_metrics.csv`, `runs/gen6_expC_5seed/arm_metrics.csv` |
| best structured alternative | C2_TRUECENTRE 1.0433 (Δ +0.0035, not distinguishable) | `runs/gen6_expC_5seed/contrast_summary.csv` |

### 3.2 What gen7 has measured so far (3 seeds unless noted)

`runs/gen7_architecture/learners/leaderboard.csv`, `oracles/leaderboard.csv`:

| arm | macro | offset | shape | nn<0.4 | pooled R² |
|---|---|---|---|---|---|
| `ORACLE_cell` (sanity) | 0.0000 | 0 | 0 | 0 | 1.000 |
| **`ORACLE_batch`** | **0.3502** | ~0 | **0.3634** | 0.3076 | 0.771 |
| **`ORACLE_level`** | **0.4974** | ~0 | **0.5086** | 0.3761 | 0.609 |
| **`TREE_MC_ecfp_massaction`** (best real) | **0.9976** | 0.8468 | 0.5045 | 0.8409 | 0.258 |
| `TREE_MC_donors` | 1.0069 | 0.8553 | 0.4998 | 0.8515 | — |
| `TREE_MC_all2d_massaction` | 1.0074 | 0.8570 | 0.5065 | 0.8747 | — |
| `TREE_MC_lig2d_ext_massaction` (gen6 champion) | 1.0475 | 0.8704 | 0.5095 | 0.9824 | — |
| `LRN_catboost_rmse` | 1.0771 | 0.9423 | 0.4999 | 1.0948 | — |
| `TREE_MC` (metal + conditions only) | 1.0852 | 0.9107 | 0.5499 | 0.8676 | — |
| **`NULL_metal_cond`** (no ligand information) | **1.0882** | 0.8905 | 0.5614 | 0.9289 | 0.051 |
| `LRN_kernelridge_rbf` | 1.2684 | 1.0809 | 0.5980 | 1.3665 | — |
| `LRN_mlp` | 1.3599 | 1.2083 | 0.5439 | 1.4191 | — |
| `LRN_ridge` | 1.4563 | 1.2591 | 0.6100 | 1.4185 | — |
| `NULL_global_mean` | 1.5525 | 1.3898 | 0.6028 | 1.7893 | — |

`TREE_MC_lig2d_ext_massaction` at 1.0475 (3 seeds) against gen6's 1.0468 (5 seeds) is a satisfactory
harness reproduction — but it is **not** a bit-exact check, because the seed sets differ. A proper
reproduction gate (5 seeds, byte-identical folds) has not been run in gen7.

### 3.3 Discrimination floor

`runs/gen6_phase0/summary.json` → `steps.reproduction`: refitting the same arm on a different machine
gives mean 0.00286, **max 0.01013** macro MAE difference. **An effect below ≈0.01 macro MAE is
indistinguishable from a change of machine.** In gen6 Experiment F the replicate-to-replicate spread
was mean |Δ| 0.038, sd 0.049 — so below ≈0.04 an effect is inside replicate noise at that scale.

---

## 4. The error budget: where the 1.0 actually sits

The nested oracle chain in `runs/gen7_architecture/oracles/leaderboard.csv` was run on **identical
folds and identical rows**, so successive differences are interpretable as "what does knowing this
buy?". These are differences in macro MAE between nested arms, **not** a variance decomposition;
they telescope only because the chain is nested.

```
NULL_global_mean         1.5525
   ↓  −0.4643   metal + conditions (29.9 %)
NULL_metal_cond          1.0882
   ↓  −0.0906   ALL ligand chemistry as currently modelled ( 5.8 %)
REAL_best_tree           0.9976
   ↓  −0.5002   the true per-ligand level, handed over (32.2 %)
ORACLE_level             0.4974
   ↓  −0.1472   the true per-(ligand, batch) calibration ( 9.5 %)
ORACLE_batch             0.3502
   ↓  −0.3502   everything else: metal × condition response  (22.6 %)
ORACLE_cell              0.0000
```

### 4.1 Finding 1 — chemistry is worth 0.0906, and that is the whole prize we have collected

`NULL_metal_cond` (an ExtraTrees on METAL + COND only, **no ligand information at all**,
`gen7/contenders.py:45`) scores **1.0882**. The best real model scores **0.9976**. Every fingerprint,
descriptor, donor census, mass-action term and 3D block in the repository, together, is worth
**0.0906 macro MAE — 8.3 % of the no-chemistry baseline, and 15.3 % of the 0.5908 gap between that
baseline and a perfect level model.**

This is the Chuang & Keiser control run on our data, and it is the single most important number in
Block A. It reframes everything: gen2→gen6 has been optimising a 0.09-wide band.

Corroborating: `IND_noligand_all` 1.0903 vs `IND_all` 1.0475
(`runs/gen7_architecture/indicators/leaderboard.csv`) — removing the entire LIG2D_EXT block from the
champion costs only **0.0428**.

### 4.2 Finding 2 — the ligand level is 32 % of the error and remains almost entirely unlearned

`ORACLE_level` is handed each held-out ligand's own true mean `log_D` and re-learns the response
around it (`gen7/decompositions.py:Oracle`, kind `"level"`). Its `offset_mae` is 0 by construction
(6.3e-16). It scores **0.4974**, so the level we fail to predict is worth **0.5002** — 5.5× larger
than everything chemistry currently contributes. Confirmed independently by gen6 Phase 0
(offset share of SSE 0.399 for `MC_lig2d_ext` on unseen_chemotype) and externally by Zahariev et al.
2024, whose six held-out-ligand log K RMSE falls 0.53 → 0.31 when a single per-ligand constant shift
is removed.

### 4.3 Finding 3 (NEW) — ~29 % of the "immovable" shape error is study-to-study calibration

Shape error has been reported as immovable across gen5 and gen6: 0.46–0.78 across every regime, arm
and novelty bin, unchanged by any feature block. `ORACLE_batch` breaks it. Given the true
(ligand, batch) mean — where batch = (DOI, table/figure), 634 distinct batches per
`gen7/recovered.py` — shape falls from **0.5086 → 0.3634, −0.1452, −28.5 %**, and macro from
0.4974 → 0.3502.

Because `shape_mae` centres within **ligand** (not within ligand-batch), this says directly: *more
than a quarter of the within-ligand residual scatter is a constant offset between the publications /
figures a ligand was measured in.* It is not chemistry, not conditions, and not irreducible
measurement noise — it is inter-study calibration, and it is the largest structured signal in the
data that no gen1–gen6 model has ever addressed.

Independent support: `gen7/decompositions.py` (module docstring) records publication identity
explaining 42 % of `log_D` variance and the (DOI, table/figure) batch 64 %; the SAFE-recovery audit
puts `DOI × Data Location` group-mean R² at **0.5001** over 197 series and `addition_date` at 0.6474.
gen6 Phase 0 found the 26 replicated cells with **no** recoverable physical difference scatter more
(median range 0.875) than the 79 with one (0.342) — consistent with a per-study offset rather than a
hidden physical variable.

**Caveat that must travel with this number.** `ORACLE_batch` is an oracle: a new ligand has no DOI,
so batch identity is unavailable at inference. The 0.1472 is not a gain we can collect directly. It
is a bound on what *deconfounding* (removing training-side batch offsets so the model learns
chemistry from a cleaner target) could recover, plus a warning that any model that can *infer* batch
from features is being rewarded for the wrong thing (see §6.2).

### 4.4 Finding 4 — missing-value indicators are worth more than the whole ligand descriptor block

Verified bit-identical at full precision:

| arm | macro MAE | note |
|---|---|---|
| `IND_none` ≡ `LRN_extratrees` | 1.108948770209826 | median impute, no indicators |
| `IND_all` ≡ `TREE_MC_lig2d_ext_massaction` | 1.047536591460264 | median impute **+ missing indicators** |

**Δ = 0.061412 macro MAE, 0.074127 offset MAE**, from a single boolean. That is 1.4× the entire
LIG2D_EXT contribution (0.0428) and 68 % of everything all ligand chemistry contributes (0.0906).

Two consequences. (a) It is free and should be on everywhere. (b) **It is also a red flag**: the NaN
columns driving it are `cond__contact_time_min` (39.95 % NaN), `cond__metal_concentration_mM`
(22.52 %) and `cond__temperature_C` (1.62 %), and missingness is curator-batch-structured — the
structured `comments_description` schema covers exactly the 3,388 rows (56.54 %) entered by one
curator. A missing indicator is partly a **study fingerprint**, so part of this 0.061 may be the same
batch signal `ORACLE_batch` quantifies, arriving through the back door. This must be tested, not
assumed (see §10, World D).

### 4.5 Finding 5 — the shape floor, cleanly bounded

`ORACLE_batch` still leaves 0.3502 macro / 0.3634 shape. With the level *and* the study calibration
both handed over, a third of a log unit of metal×condition response error remains. That is the true
target for any conditions-side or metal-side modelling, and it is **larger than the entire ligand
chemistry contribution**.

### 4.6 The hard-chemistry endpoint is not harder than the medium one

`REAL_best_tree`: nn<0.4 = 0.8409 (409 rows) but nn<0.6 = 1.1007 (1,494 rows), and 'all' = 0.9976.
gen6 saw the same inversion (nn<0.4 1.0429, nn<0.6 1.1601). The nn<0.4 bin holds ~36 ligands whose
`n_ligands == n_ecfp_clusters`, so "macro over clusters" degenerates to "macro over ligands" and the
bootstrap block count collapses. **Do not read nn<0.4 as the hardest endpoint** — it is a small,
idiosyncratic subset. Also note gen7's `nn_train_tanimoto` is computed against the *whole* training
set (152 ligands, `exclude_self=False`) while gen6 Exp A cut on `nn_reference_tanimoto` against the
91-ligand BASE set: 409 rows vs 544. **The two novelty endpoints are not comparable.**

---

## 5. Harness traps gen7 must not repeat

Each was actually committed in gen5 or gen6 and found by audit.

**T1 — An arm list that diverges from the module default.** `slurm/gen5_levels.slurm:25` hard-coded
18 arms, omitting every `*_massaction` arm that `levels.LEVEL_DEFAULT_ARMS` contained. Result: 16 NaN
rows in `property_families.csv`, and every published massaction number came from a *separate local
run* whose base arms differ by up to 0.004 macro MAE. The champion has never been fitted alongside
the others in one run.

**T2 — Bootstrap resampling a unit the folds did not hold out.** `run_gen5_levels.py:502-505` falls
back to `ecfp_cluster` blocks for `unseen_series` and `unseen_conditions`, where no ECFP cluster is
ever held out. gen6 hard-codes the same mapping for every regime. Under a chemotype hold-out the
mapping is right; under anything else it is not.

**T3 — Metric names that grant a free oracle.** `shape_mae` / `shape_r2` centre **both** prediction
and truth on the test rows' own per-ligand means (`gen6/metrics.py:164,174,200`) — the model gets a
per-ligand offset it never has at deployment. gen5 called this `within_ligand_r2_shape` and forbade
calling it "within-ligand R²"; gen6 dropped the qualifier. A constant-per-ligand predictor scores
0.0000 under the shape metric and −1.1131 under the deployable one. **gen7 must report both, and
must never call `shape_r2` a deployable R².**

**T4 — Mixed denominators.** `macro_mae` weights 131 ECFP clusters; `offset_mae` and `shape_mae`
weight 152 ligands. They appear in the same CSV row and **do not decompose `macro_mae`**. Measured
discrepancy between unit-averaged and leaderboard values: 0.019–0.021 — larger than the 0.004–0.014
claimed in the `UNIT_STATISTICS` docstring, and comparable to whole effect sizes.

**T5 — Quoting the percentile CI alone.** One chemotype holds 28 of 131 scoring units (21 % of the
vote in 1.3 % of the evidence). Double-bootstrap coverage puts the one-sided Type-I rate at **12.7 %**
at the `all` endpoint. On the headline gen6 contrast the percentile CI [0.034, 0.313] excludes zero
while BCa [−0.003, 0.277] does not. Report BCa; never claim significance from the percentile alone.

**T6 — `block_macro_delta` has no interval.** It is the estimand matching the fold design (one vote
per held-out chemotype) and it is the only statistic without a CI. On gen6 Exp A seed 104729 it is
0.271 against a unit point estimate of 0.150 — quoting it beside the percentile CI mixes two
estimands.

**T7 — Counting seeds as replication.** The five split seeds re-partition the *same* fixed cohort;
residual correlation across seeds is r = 0.97–0.99. "5/5 seeds" is close to one observation counted
five times. **The cluster/chemotype-block bootstrap is the load-bearing statistic.**

**T8 — The two-stage residual trap.** Stage B trained on a cross-fitted residual learns Stage A's
level error on chemistry the inner model never saw and mis-applies it: 1.2588 vs 1.0680 for the
monolith at seed 104729; mean |B̂| 0.5291 cross-fitted vs 0.1477 true-centred. Centring on the
**observed** cell mean recovers parity (1.0433) and nothing more. Any gen7 residual or delta model
must anchor on an observed label, never on a predicted one. `gen7/decompositions.py:OffsetShapeTree`
already implements the working variant ("the OBSERVED centred target, never Stage A's error").

**T9 — Constant columns are never dropped.** `levels.py` drops all-null columns
(`levels.py:487-493`, exactly 5) but not constant ones. **1,657 of 2,482 features are constant across
the gen5 cohort** — including 1,639 of the 2,048 ECFP bits. Every block-size figure in every report
overstates available information by ~67 %.

**T10 — `n_eff_pooled_rows` is a misnomer.** It is a Kish effective number of **clusters** (6.256),
printed in the same row as `n_rows = 5248`.

**T11 — `sign_accuracy` is row-order dependent** (0.6261 vs 0.6240 on reshuffled identical data),
contradicting its own docstring, and is n_pairs-weighted so one 1,488-row ligand dominates.

**T12 — Endpoint populations differ.** NaN-similarity rows are kept in `all` and dropped from every
threshold subset; `ood_calibration_table` drops them everywhere. The two tables disagree on which
rows exist. Endpoints are nested; no multiplicity adjustment is applied.

**T13 — Un-timestamped pre-registration.** `git log` shows the gen6 protocol and its results were
committed together (`6df9d55`, `fc39fac`), contradicting the Phase 1 doc's claim that the protocol
was committed before any model was fitted. Only the Phase 2 doc discloses this.

**T14 — MASSACTION contains three columns with no thermodynamic meaning.** The block's docstring
derives it from `log D = log K_ex + n log[L] + 3 log[NO₃⁻]`, but the code emits log₁₀ of contact
time, metal concentration and **temperature in Celsius**. `log₁₀(T/°C)` has an arbitrary zero point;
van 't Hoff needs 1/T in kelvin.

**T15 — Latent crash.** `LevelRegressor.feature_importance_frame` (`levels.py:677-686`) raises
whenever `DropAllNaNColumns` removes a column in-fold — exactly the unseen_chemotype case it was
added for. Currently uncalled.

**T16 (gen7-specific, live) — the learner/representation screens are not like-for-like.**
`Tabular.add_indicator` defaults to `False` (`gen7/contenders.py:145`), so every `LRN_*` and `REP_*`
arm ran **without** missing indicators while every `TREE_*` arm ran **with** them. The handicap is
0.0614 macro MAE — larger than most differences those screens report. Any conclusion of the form
"the learner does not matter" or "representation X beats Y" from
`runs/gen7_architecture/learners/` or `screen_representations/` is currently unsafe.

**T17 (gen7-specific, live) — seed counts.** `representations` and `embeddings` ran at 1 seed;
per-arm seed sd in the 3-seed runs is 0.006–0.039 macro MAE. Single-seed orderings within 0.05 are
noise.

---

## 6. Leakage risks

### 6.1 Target-derived columns that survive into the frame

- `levels.py` has **no forbidden-name audit**. The raw distribution ratio `D` — the un-logged target
  — survives to the final `frame[keep_columns]` selection at `levels.py:496-501` and is excluded only
  because it matches no block prefix. Same for `sample_weight_inv_metal_freq`, `safe_exp_id`,
  `build_id`, `geometry_key`, `xyz_path`, `split`. `feature_registry.py` has an explicit
  `_FORBIDDEN_FEATURE_FRAGMENTS` check (`:292`, enforced at `:844-851`); the level pipeline has
  nothing equivalent. Adding a `cond__D` column would admit the target silently.
- Upstream, `obsDvaluesValue == D` on **5,992/5,992 rows to 1e-6**. Any wholesale join of the SAFE
  tables imports the label. `scripts/reconstruct_provenance.py` guards this via `CORE_COLUMNS`;
  nothing enforces it elsewhere.
- `f_Metal_Concentration_mM` is the **organic-phase metal concentration — the numerator of D**. It is
  0 % populated on these rows so it looks harmless, but it must be permanently blacklisted before any
  wider upstream join.

### 6.2 Batch / provenance proxies

- `addition_date` explains **83.10 %** of within-repeated-cell SS and gives group-mean R² **0.6474**
  — pure bookkeeping (consecutive seconds of typing = consecutive points of one digitised figure).
  The worst available leak. Only 4 distinct calendar days exist, so day-level truncation drops it to
  R² 0.1258 — the leak lives entirely in the seconds. Not in the bundle; never join it.
- `DOI × Data Location` → 197 series, group-mean R² **0.5001**. This is the true experiment-series id
  gen5's `unseen_series` folds approximated. Legitimate as a **fold group** and as `ORACLE_batch`;
  never as a feature. `gen7/recovered.py:427-428` correctly excludes `nuisance__*` from
  `recovered_feature_columns()` — verified.
- `volType` / `volValue` and the `Metal_Oxidation_state` missingness flag are **curator-batch
  markers**: `volType` is present on exactly the 3,388 Thomas Summers rows and nowhere else.
- `Publication_Year` has group-mean R² 0.1398 purely because extractant families are fashionable by
  era. It will look predictive on a random split and collapse on a chemotype hold-out.
- **The live one:** missing-value indicators (§4.4) are partly curator-batch proxies. Under a
  chemotype hold-out a publication can span several chemotypes, so a missingness pattern can let the
  model infer "this row is from a study whose other rows are in training" and pick up that study's
  calibration offset. This is a legitimate-looking +0.0614 that may not survive publication-clean
  folds.

### 6.3 Fold-level contamination

- gen5's `unseen_series` group spans more than one publication in **26.32 %** of rows (1,577/5,992);
  the extractant group in **61.92 %**. Only the averaging cell is clean (0.10 %, 6 rows).
  gen5 called `unseen_series` "the honest known-ligand regime"; its numbers are optimistic by an
  unmeasured amount. gen6 Experiment C scored hypothesis C2 in that regime.
- No gen1–gen7 run has ever used `publication_id` as a CV group, despite the audit existing.

### 6.4 Target-dependent preprocessing

`build_level_dataset` drops rows with `log_D <= -6.0` **before** counting cells for eligibility
(`levels.py:396-401`, 3 rows). Permuting `log_D` and rebuilding gives 5,249 rows with 5 row ids
differing. The chemistry map is target-invariant; the cohort it is applied to is not. Small, but it
is target-dependent preprocessing applied to the evaluation population and must be stated.

### 6.5 3D-specific leakage

- `geometry_key = "<Z>|<canonical_smiles>|<anion>"` — a hash of metal + ligand + medium. **801 of
  1,155 geometries serve exactly one row**, so per-geometry descriptors are near-unique row labels
  for those rows. One geometry serves 377 rows (mean 4.74), so a random-row split puts an identical
  3D vector on both sides of a fold.
- Geometry variant choice re-encodes an existing flag: **all 3,957 nitrate-geometry rows have
  `cond__acid__hno3 = 1`**. Any 3D block smuggles in the nitric-acid indicator.
- `coordination_number` is the declared build spec, not a measurement — it disagrees with the
  measured donor count within 3.10 Å in **22.4 %** of structures, and is perfectly collinear with the
  Vietoris-Rips `is_coord_donor` flag count.
- Nullness of `ln_donor_distance_09` is a perfect proxy for `coreCN == 8` (43.5 % of rows), so an
  imputation indicator on the polyhedron block leaks the CN label.

### 6.6 Two live data-integrity defects the level pipeline does not fix

- **TODGA.** `pairs.py::apply_default_quarantine` drops rows flagged
  `todga_structure_assigned_to_different_extractant`, but `run_gen5_levels.py`, `gen6_phase0.py` and
  the gen7 harness **never call it**. 111 dataset rows over 19 distinct ligands have TODGA's SMILES
  pasted in; 129 rows enter **69 level cells attributed to TODGA = 4.64 %** of TODGA's 1,488 level
  cells. They do not average into real TODGA cells (0 mixed) — they arrive as extra, wrong cells.
- **DMDPhPDA decimal corruption.** The same molecule from the same paper (DOI
  10.1081/SEI-120030392) was entered twice by two curators; the pairwise Δ log_D at matched
  (metal, HNO₃) is an **exact integer in all 70 cells** (0×11, 1×17, 2×14, 3×14, 4×14) — one copy
  kept only the mantissa. 70 rows are wrong by 1–4 orders of magnitude, and because the two entries
  wrote `CH3Cl` vs `Chloroform`, **83 % of `cond__diluent__ch3cl` (70 of 84 rows) is this one
  corruption.** Any tree splitting on that column is learning "this log_D is 10ⁿ too high" and
  reporting it as a diluent effect.

Both are cheap to fix and both are currently in every gen7 number.

---

## 7. Inventory of available features and structures

### 7.1 Feature blocks in `levels.py` (column counts measured on the built cohort)

| block | source prefix | cols | notes |
|---|---|---|---|
| METAL | explicit list | 3 | atomic number, lanthanide index, ionic radius |
| PHYSCHEM | explicit list | 10 | MolWt, TPSA, HBD/HBA, rotatable bonds, rings, FractionCSP3, MolLogP |
| DONORS | `donor__*` + 4 scalars | 13 | **4 of 13 vary with the metal** (DENTATE, coreCN, n_ligs, n_fill) — not a pure 2D ligand block |
| COND | `cond__` | 64 | 9 acid one-hots, 9 additive, 41 diluent, 5 continuous |
| ECFP | `ecfp_0…2047` | 2,048 | **1,639 constant**; 190 SMILES → only **164 distinct fingerprints** (18 collision groups, 44 ligands with a Tanimoto-1.0 partner) |
| LIG2D_EXT | `lig2d__` | 206 | 168 RDKit + 38 hand-crafted; **schema is cohort-dependent** (constant columns dropped at build time) |
| COMPLEX_PHYS | `feat3d__complex_physical__` | 26 | 5 all-null dropped; **ligand×metal, not ligand-only** |
| POLYHEDRON | `feat3d__polyhedron` | 58 | prefix has no trailing `__`, absorbs `polyhedron_scalars__` too; all 58 vary within an extractant |
| GEOM_COND | `geom_cond__` | 46 | one-hots; `pH_bin=missing` and `phase_ratio_bin=missing` are always 1 |
| MASSACTION | `massact__` | 8 | 5 log₁₀ continuous conditions + `logL×DENTATE`, `logL×coreCN`, `logL×logH`; see T14 |

Sum = 2,482, disjoint. Arms: `MC` 67, `MC_ecfp` 2,115, `MC_lig2d_ext` 273, `MC_all2d` 2,344,
`MC_everything_massaction` 2,482.

### 7.2 Blocks gen7 adds (`gen7/harness.py:118-200`)

`RECOVERED` (upstream-recovered variables, §8), `METALPHYS` (physical lanthanide properties),
`METAL_ONEHOT`, `LIGPHYS` (physically motivated level features + ligand-diluent coupling), and five
frozen embedding blocks: `EMB_CHEMBERTA`, `EMB_CHEMBERTA_MLM`, `EMB_MOLFORMER` (mean-pooled) and
`EMB_CHEMBERTA_CLS`, `EMB_MOLFORMER_CLS`, cached in
`dataset with 3D structures/ligand_pretrained_embeddings.parquet` for all 190 extractants.

Verified encoder properties: ChemBERTa's tokenizer **silently drops stereochemistry** (190 SMILES →
188 distinct vectors; median max cosine 0.983); MoLFormer keeps `[C@H]`/`[C@@H]` (190 distinct;
median max cosine 0.957). All 16 embedding arms are worse than the incumbent at 1 seed.

### 7.3 3D structures

`dataset with 3D structures/geometries/` — **1,155 XYZ files**, all `qc_class = OK`, checksums all
match. Each is a **single-Ln cationic complex** (exactly one lanthanide in 1,155/1,155; net charge
+3/+2/+1 = 901/181/63, **never neutral**; 911 contain no nitrate at all). Coverage: 177 of 190
ligands (93.2 %), **5,479 of 5,992 rows (91.44 %)**.

**There is no conformational ensemble.** One file per `geometry_key` and per full chemical
environment; the 124 (ligand, metal) cells with two files differ in `inner_sphere_anion`, not
conformation. Nothing to Boltzmann-average.

The 8.04 % exclusion (101 of 1,256) is **non-random**: it removes the largest, most flexible
scaffolds (TWE-7, TWE-10, DO-PyranDGA, tris-DGA benzenes, a hexameric calixarene). A 3D-vs-2D
head-to-head must be restricted to the 5,479 shared rows.

Quality flags inside the accepted set: **39 unrelaxed structures** (no energy on the comment line;
10 with no charge column) still carry `qc_class = OK`; **14 files** have a metal–ligand contact
< 2.0 Å (min 1.71 Å); **9 files** have an interatomic pair < 0.9 Å (worst 0.581 Å).
`geometry_descriptors.py`'s van der Waals table has **no entry for any lanthanide** — the metal is
enclosed with the 1.70 Å carbon fallback.

Electronic block: 7 precomputed GFN2-xTB scalars, non-null on 5,365/5,992 rows and 163/190 ligands;
`complex_free_energy_eV` is **bit-identical** to `complex_total_energy_eV`; five requested columns
(`homo_eV`, `lumo_eV`, `homo_lumo_gap_eV`, `binding_energy_eV`, `strain_energy_eV`) are 0 % populated.
**None of it is usable prospectively** — a new extractant has no bundled complex geometry and nothing
in this repo generates one.

gen7 measurement: `REP_3d_complexphys` = **1.1732**, the worst arm in the representation screen.
Sixth consecutive study in which 3D loses to 2D.

---

## 8. The recovered-variable opportunity

The bundle joins **5,992/5,992** to the 31 upstream `*_SAFE.csv` exports on `safe_exp_id`, with four
independent cross-checks agreeing on every row. What was recovered, and what it is worth:

| variable | coverage | evidence of value |
|---|---|---|
| **Aqueous complexant** (name / conc. / SMILES, parsed out of `comments_description`) | **557 rows (9.30 %)**: DTPA 165, TEDGA 88, DOODA(C2) 88, NaNO₃ 70, Amicacid 69, malonamide 56, HEDTA 10, BTP-4Me 10, CDTA 1 | **explains 56.67 % of within-repeated-cell log_D SS**; within-cell shift mean −0.918 / median −1.071 over 40 mixed cells; η² adj 0.297, p < 0.003 inside replicated full-condition cells |
| Diluent physics (parsed `Solvent_Name` → volume-weighted dielectric, dipole, logP, molar volume, Hansen, aromatic/halogen/hydroxyl content) | 83 upstream solvents vs 34 downstream one-hots; **305 rows / 43 solvents dumped into `cond__diluent__other`**; a modifier volume fraction is recoverable on 1,357 rows (22.65 %) | explains 4.95 % of within-cell SS; **interpolates to unseen diluents**, which a one-hot cannot |
| `Shaking_Time_min` | 1,067 rows (17.81 %), **perfectly disjoint** from `Contact_Time_min` — 0 rows have both | coalescing raises equilibration-time coverage **63.02 % → 80.83 %**; explains only 0.27 % of within-cell SS |
| `Phase_Modifier_Concentration_M` | 265 rows, 15 values 0.1–2.537 M | identity survived downstream, amount did not |
| `No. of Metals` / competitive loading | 3,061 single-metal vs 327 mixtures (2–33 metals) | mean log_D 0.618 vs 0.778 |
| `nuisance__doi`, `nuisance__batch` (DOI × Data Location) | 115 DOIs / 634 batches (`gen7/recovered.py`); the gen6 provenance audit reports **105 publications, 0 % ambiguous** | **nuisance only** — the `ORACLE_batch` signal (§4.3). Never a feature. |

**Two things the dedicated columns get wrong.** `Holdback_Agent_Name` is **0 % populated across all
48,138 upstream rows** — reading it alone gives exactly the wrong answer, because 557 rows *do* have
one, typed into a comment field. And `geom_cond__nitrate_activity` is derived from the acid alone, so
the 235 rows with 1–5 M added NaNO₃ are labelled trace/very_low/low while the true nitrate activity
is molar — a physically **wrong** feature, not merely a missing one.

**Measured so far** (`runs/gen7_architecture/screen_recovered/scores_by_seed.csv`, 3 seeds):

| arm | macro | offset | shape |
|---|---|---|---|
| `REC_base` (METAL, COND, DONORS, MASSACTION) | 1.0359 | 0.8874 | 0.4956 |
| `REC_plus_recovered` | **1.0049** | 0.8363 | 0.5075 |
| `REC_champion` (= `IND_none`) | 1.1089 | 0.9446 | 0.5158 |
| `REC_champion_plus_recovered` | **1.0590** | 0.9048 | 0.5035 |

**+0.0310 and +0.0499 macro MAE, consistently, and entirely in the offset** (0.0511 and 0.0398). Both
exceed the 0.01 machine-noise floor; neither yet has a paired bootstrap CI. Note the coverage caveat:
the parsed `comments_description` schema covers 56.54 % of rows *by curator batch*, not by chemistry,
so any feature built from it is missing-not-at-random along a boundary that itself correlates with
log_D.

---

## 9. Summary of the diagnosis in one paragraph

Under a Tanimoto-0.7 chemotype hold-out on 5,248 rows, a metal-plus-conditions model with **no ligand
information at all** scores 1.0882 macro MAE; the best model in the repository scores 0.9976. Six
generations of ligand representation work have therefore purchased **0.0906 log units**. Handing the
model each held-out ligand's true level buys **0.5002**, and handing it the true per-(DOI, figure)
calibration on top buys a further **0.1472** — while cutting the "immovable" within-ligand shape error
by 28.5 %. Meanwhile a single boolean (missing-value indicators) is worth 0.0614, more than the entire
206-column 2D descriptor block. The error is not in the learner (18 learners span 0.998–1.55 with
every tree clustered inside 0.10), not in the fingerprint, and not in 3D geometry (worst arm, 1.1732).
It is in an unlearned per-ligand level and an unmodelled per-study offset, on a cohort whose
chemotype coverage gen6 already measured to be the binding constraint.

---

## 10. Ranked architecture hypotheses, with falsifiable predictions

Framing: five worlds for where the residual error lives. Each carries a **prediction that could come
out false**, and the cheapest experiment that would decide it. Ranked by expected information per
CPU-hour.

---

### World D — *Nuisance world*: a large part of the residual is study-to-study calibration, not chemistry. **Rank 1.**

**Claim.** log_D as recorded is `chemistry(ligand, metal, conditions) + α_batch + ε`, where α_batch is
a per-(publication, figure) offset of order 0.3–0.5 log units arising from digitisation, unrecorded
protocol, and lab calibration. Models are currently fitting α_batch into their chemistry terms and
paying for it on new chemotypes.

**Direct evidence.** `ORACLE_batch` 0.3502 vs `ORACLE_level` 0.4974 (Δ 0.1472); shape 0.3634 vs
0.5086 (−28.5 %); pooled R² 0.771 vs 0.609. `DOI × Data Location` group-mean R² 0.5001. The 26
replicated cells with no recoverable physical difference scatter *more* (0.875) than the 79 with one
(0.342). Publication mixing already contaminates gen5's `unseen_series` at 26.32 % of rows.

**Falsifiable predictions.**
- **D1.** Subtracting shrunken training-side (ligand, batch) intercepts from the target before
  fitting (`gen7/decompositions.py:BatchDeconfounded`, already implemented) improves macro MAE by
  **≥ 0.03** with a BCa CI excluding zero over 5 seeds. *If the CI spans zero, the batch signal is
  not separable from chemistry with 634 batches and 152 ligands, and World D is a measurement
  artefact of the oracle.*
- **D2.** Re-running the champion with **publication-clean folds** (no publication spanning train and
  test) degrades macro MAE by **≥ 0.03**. *If it does not move, cross-publication leakage is not
  inflating our numbers and D loses most of its force.*
- **D3.** The missing-indicator gain (0.0614) shrinks by **≥ 40 %** under publication-clean folds.
  *If it survives intact, the indicators are encoding genuine experimental design, not study
  identity, and should simply be kept.*

**Cost.** D1: hours (implemented). D2: 1 day (needs a publication-grouped splitter). D3: free once D2
exists. **Highest expected value on the list** — it is the only hypothesis backed by a new oracle
measurement, it is cheap, and D2/D3 are protocol audits we owe regardless of the answer.

---

### World E — *Level world*: the error is a single scalar per ligand, and the architecture should predict that scalar directly. **Rank 2.**

**Claim.** The deployable quantity is `level(ligand, series) + response(metal, conditions | ligand)`,
the response is already well predicted, and every gen2–gen6 model has been fitting a joint function
that spends its capacity on the wrong term.

**Direct evidence.** `ORACLE_level` 0.4974 vs 0.9976 — the level is 50 % of the error and 5.5× the
total ligand-chemistry contribution. Offset share of SSE 0.30–0.50 in every gen6 regime. gen6 Phase 2:
level oracle 0.60 vs metal oracle 0.21 on new chemotypes. Externally: Zahariev et al. 2024, six
held-out ligands, log K RMSE 0.53 → 0.31 after removing one constant shift per ligand.

**Falsifiable predictions.**
- **E1.** A two-head model — a level head fitted on **one row per training ligand** (so a 1,500-row
  ligand does not outvote a 3-row one) plus a response head on the observed within-ligand centred
  target (`gen7/decompositions.py:OffsetShapeTree`, implemented) — beats the monolith by **≥ 0.02**
  macro MAE, and the gain is **entirely in `offset_mae`**. *If offset does not move, decomposition
  buys nothing and only the level *predictor* matters, not the architecture.*
- **E2.** Anchored pairwise-difference regression (PADRE) — predicting `Δlog_D` against **observed**
  training labels and averaging `anchor + Δ̂` — beats the monolith by **≥ 0.03**. *If it does not, the
  offset is not learnable from chemistry at Tanimoto < 0.4 and World B is the truth.*
- **E3 (the trap check).** A Stage-B model on a **cross-fitted residual** must be *worse* than the
  monolith, reproducing gen6's 1.25 vs 1.08. *If it is not worse, our understanding of the gen6 trap
  is wrong and the whole two-stage literature reopens.*

**Cost.** E1: hours (implemented, unrun). E2: 2–4 days. E3: hours, and it is a required control.

---

### World B — *Coverage world*: the model is a near-neighbour lookup and nothing but more chemistry will move it. **Rank 3 (largely established; needs bounding, not testing).**

**Claim.** Predictive skill is a monotone function of nearest-training Tanimoto, and below ~0.4 the
model carries no edge over a constant. The binding constraint is the number of distinct chemotypes,
not the model.

**Direct evidence.** gen5: macro MAE edge over `NULL_global_mean` by nn-Tanimoto bin = **−0.041**
(<0.4), +0.179 (0.4–0.6), +0.602 (0.6–0.8), **+0.656** (≥0.8); 58.87 % of unseen_ligand test rows have
a training ligand at Tanimoto ≥ 0.8. gen6 Exp A: expanding 91 → 152 extractants for **+7 % rows**
bought **+0.1633** macro MAE overall and **+0.4630** at nn<0.4, 5/5 seeds, and the gain lands almost
entirely on the 57 clusters made of added ligands (+0.3670) rather than the 72 BASE-eligible ones
(+0.0081). gen6 Exp B: B3 **FAILED** — the depth curve was still improving at the largest budget
(+0.3508 [+0.1996, +0.5667]), so no plateau is located in either direction.

**Falsifiable predictions.**
- **B1.** Adding the SPT 2026 3,343-point REE database (after deduplication) improves macro MAE by
  **≥ 0.05** on the *existing* 79 chemotypes' held-out rows — i.e. on rows the new data does not
  itself contain. *If it does not, coverage has saturated and World B is spent.*
- **B2.** With the current cohort, no architecture beats the incumbent by more than **0.05** macro
  MAE. *This is the strong form of World B; a single arm exceeding it falsifies it.* (Note gen7
  already has `TREE_MC_ecfp_massaction` at 0.9976, +0.0499 over gen6's 1.0468 — right at the boundary,
  and at 3 seeds.)
- **B3.** A Lo-Hi-style strict cut (no cross-fold ligand pair above Tanimoto 0.7) *degrades* the
  champion by **≥ 0.05**. *If it does not, our chemotype folds are already genuinely disjoint and the
  headline is honest as stated.*

**Cost.** B1: 2–4 days of curation. B2: free (it is the criterion for every other arm). B3: half a day.

---

### World C — *Missing-variable world*: much of the "irreducible" error is variables thrown away on the way into the bundle. **Rank 4 (partly refuted, partly confirmed).**

**Claim.** The replicate noise floor is not measurement noise; it is unrecorded chemistry — aqueous
complexants, diluent identity beyond a one-hot, equilibration time, modifier concentration.

**Direct evidence for.** The aqueous complexant explains **56.67 %** of within-repeated-cell log_D SS
and is invisible downstream (`Holdback_Agent_Name` is 0 % populated in 48,138 rows); its within-cell
effect is −0.918 mean / −1.071 median. `REC_plus_recovered` beats `REC_base` by **+0.0310** (3 seeds).
gen6 Phase 0 found the published noise floor "contaminated, true value unknown".

**Direct evidence against.** The related "the 17/190 ambiguous SMILES are a missing-variable ceiling"
version of World C is **refuted**: once conditioned on the full `cond__` vector, extractant name
explains nothing (adjusted η² −0.082, p = 0.53), and after removing three known curation defects
aliased rows agree *better* (RMS |Δ| 0.694, n=37) than genuine same-name replicates (1.325, n=4,226).
`Shaking_Time_min` explains 0.27 % of within-cell SS; `Solvent_Name` 4.95 %.

**Falsifiable predictions.**
- **C1.** The RECOVERED block's gain survives a 5-seed paired chemotype-block bootstrap with BCa
  excluding zero. *If not, the +0.031 is 3-seed noise.*
- **C2.** Restricting the RECOVERED block to the **complexant columns alone** captures ≥ 60 % of its
  gain. *If the gain is instead carried by the sparse scalars (4–18 % coverage) or by
  `rec__solvent_parsed`, it is a missingness proxy, not chemistry.*
- **C3.** A recomputed aleatoric floor from **cross-publication duplicate cells** (SC3's method) lands
  **above 0.35** log units. *If it lands near 0.19 as published, then 0.9976 is 5× the floor and there
  is genuine modelling headroom; if it lands near 0.6, we are within 1.7× and the project is close to
  done.* **This single number changes what work is worth doing and has never been computed.**

**Cost.** C1: 1 day CPU. C2: hours. C3: 1 day.

---

### World A — *Capacity/representation world*: a better learner or a better molecular representation is the answer. **Rank 5 — largely refuted; run only the cheap residual controls.**

**Claim.** A stronger function approximator, a pretrained encoder, or 3D geometry would close the gap.

**Evidence against, from our own artefacts.** Eighteen learners on identical features span
0.9976–1.5525, and **every tree is inside a 0.11 band** (0.998–1.109) while every non-tree is far
worse (kernel ridge 1.2684, MLP 1.3599, ridge 1.4563). All 16 pretrained-embedding arms are worse
than the incumbent. `REP_3d_complexphys` is the worst representation arm (1.1732). gen6 Phase 0
concluded coverage, not capacity, is the bottleneck (+0.163 macro MAE on identical test rows from
breadth alone). Externally: SC3 has LightGBM+RDKit beating every GNN and foundation model ID *and*
OOD; Praski et al. find nearly all pretrained embeddings indistinguishable from ECFP under
hierarchical Bayesian testing; BOOM finds MLM pretraining sometimes *degrades* OOD.

**Evidence for, and what keeps it alive at all.** Two independent groups (Zahariev 2024, Karunaratne
2025) find D-MPNN beating tree-on-fingerprints **for metal–ligand binding specifically**, on
held-out-ligand splits, and beating fine-tuned ChemBERTa and MolBERT on the same splits. And within
representations, the donor-centric block is winning: `REP_donors` 1.0535 vs `REP_ecfp` 1.1159 vs
`REP_lig2d` 1.1403 — the coordinating environment is the only ligand channel with signal.

**Falsifiable predictions.**
- **A1.** No learner swap on fixed features beats the best tree by **> 0.03** at 5 seeds with the
  indicator confound (T16) removed. *A CatBoost or LightGBM arm exceeding it falsifies A's refutation
  and means our learner comparison was never fair.*
- **A2.** Donor-anchored RAC-style autocorrelations (metal-site-centred, ~40–60 columns) beat
  `REP_donors` by **≥ 0.03**. *If they do not, "better ligand descriptors" is closed for good, and the
  only remaining representation question is the coordination environment we cannot see in 2D.*
- **A3.** A Chemprop D-MPNN with the metal as a disconnected node, pretrained on the 32,459-point
  IUPAC log K corpus and fine-tuned on our 5,248 rows, beats the tree by **≥ 0.05**. *This is the one
  architecture claim from the domain literature we have never tested; if it fails, World A is closed
  and no further capacity work is justified.*

**Cost.** A1: 1 day (mostly re-running with T16 fixed). A2: 2–3 days. A3: 1–3 h/fold plus an overnight
pretrain — the most expensive item on the list, and it should be run last.

---

### Cross-world ordering

1. **World D** — new, cheap, backed by a fresh oracle measurement, and its D2/D3 legs are audits we
   owe regardless.
2. **World E** — largest measured headroom (0.5002), two implementations already written and unrun.
3. **World B** — mostly established; needs bounding (B1) and one honest split audit (B3).
4. **World C** — half-refuted, half-confirmed; C3 (the aleatoric floor) is the single highest-value
   unmeasured number in the project.
5. **World A** — refuted for learners, fingerprints and 3D; keep only A2 and the domain-specific A3.

**Before any of it:** fix T16 (the indicator confound), re-run every gen7 screen at 5 seeds with the
paired chemotype-block bootstrap, and apply the TODGA and DMDPhPDA quarantines. Until those three are
done, no gen7 leaderboard difference under 0.06 macro MAE means anything.
