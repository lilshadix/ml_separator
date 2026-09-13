# Gen12.2 — pre-registration

*Frozen 2026-09-05, after `DATA_AUDIT.md` and `COORDINATION_DESCRIPTOR_SPEC.md` and before any
Gen12.2 level model was fitted. When this document was written the only quantities that existed
were Gen12's frozen artefacts, the target-free coordination feature matrix, and the training-only
level-definition study of §3.*

**Amendment rule.** This file is not edited once an outcome-bearing run starts. A change is a
dated addendum at the end, stating what changed, why, and whether any result had been seen.

---

## 1. Question

**Primary.** Do chemically informed coordination-topology descriptors improve zero-shot prediction
of the extractant-specific intrinsic europium extraction level, `alpha_i`, for chemically unseen
extractants, over the same learner given Gen12's generic molecular representation?

**Secondary.** If level prediction improves, does the improvement carry into full zero-shot
`log_D` prediction on unseen chemotypes?

**Tertiary.** Does explicit decomposition into a structure-only level plus a condition-dependent
shape outperform direct `log_D` regression?

The three are kept apart by construction. A level number is never quoted as a `log_D` number, and
the decomposed predictor is built only after the level models are frozen.

## 2. What is inherited unchanged

Gen12's cohort (fingerprint `2a364bb5264e9935`, 1,329 cells, 183 extractants, 97 chemotypes), its
design-B chemotype hold-out with the same five split seeds and the same model-seed rule, its
near/mid/far bands, its extractant-macro metric, its chemotype-blocked BCa bootstrap with seed
8675309 and 10,000 replicates, its 0.01 reproducibility floor and 0.02 equivalence margin, and its
few-shot draw function. These are **imported from `gen12_eu_pred/`, not copied**, and the rebuilt
fold plan was compared row-id by row-id against Gen12's and matches in all 25 folds.

Design A is not run. Gen12 measured its cost and this generation has no use for it.

## 3. The level target, chosen before any level model

`alpha_i` is not given by the data and had to be defined. Five candidates were compared on the
**training rows of one designated development fold** (design B, split seed 104729, fold 0) under a
decision rule written to `config/level_definition_rules.json` and digested *before* the study
script that reads it was run: raw per-extractant mean, robust median, condition-adjusted
fixed-effect intercept, hierarchically shrunken intercept, and residual level after a global
condition-only model.

**Primary: `LVL_MEAN`** — the mean of `log_D` over the cells at which the extractant was measured.
It is the only candidate that can be stated without naming a fitted model, it has the best
split-half reliability of the five (Spearman 0.920 over extractants with at least four cells), and
it is exactly the quantity Gen12's level component already measures, so Gen12.2's level numbers
are directly comparable with Gen12's.

**Secondary: `LVL_COND_RESIDUAL`** — the mean of `log_D` minus a cross-fitted condition-only
model. Every headline level contrast is repeated under it.

The frozen rule's second override fired and selected the condition-adjusted definition; it was
mis-specified, and the amendment, its training-only evidence and what the original rule would have
chosen are recorded in `DATA_AUDIT.md` §7 and in the rules file. **No model had been fitted and no
held-out extractant had been scored when the amendment was written.**

**Evaluation.** A held-out extractant's `alpha_i` is computed from its own held-out cells. That is
evaluation truth, in the same sense that `log_D` is the truth for a query row, and it is never an
input to anything. Under design B every extractant is held out once per seed with all of its
cells, so the level task has exactly 183 evaluation units in 97 chemotypes and the truth does not
vary by seed.

## 4. Cohorts

| cohort | rule | extractants | chemotypes | n_eff |
|---|---|---|---|---|
| **full — headline** | every eligible extractant | 183 | 97 | 13.70 |
| **level-reliable** | at least 5 measured cells | 71 | 41 | 11.75 |

The threshold of 5 follows the rule frozen before it was computed: the smallest cell count at which
the standard error of a per-extractant mean falls to a quarter of the between-extractant spread.
One-cell extractants are retained in the headline cohort. **Both cohorts are reported for every
primary contrast**; a result quoted on one is never described as holding on the other.

## 5. Feature blocks

| block | columns | contents |
|---|---|---|
| `ECFP` | 2,048 (662 informative) | the bundle's Morgan radius-2 achiral fingerprint, Gen12's |
| `GENERIC` | 231 | Gen12's `PHYSCHEM` (10) + `DONORS` (15) + `LIG2D` (206) |
| `COORD` | 114 (109 informative) | the coordination-topology block, spec 1.1.0 |
| `COND`, `MASSACT` | 64 + 8 | conditions; **excluded from every level model by definition** |

`GENERIC` is exactly the molecular content of Gen12's best-scoring arm `ABL_D`. That makes the
comparator harder than the brief requires: **the coordination block must beat a baseline that
already contains a donor census and 206 hand-crafted 2D descriptors**, not a bare fingerprint.

## 6. The level ladder

One learner, extremely randomised trees, because Gen12 established it as the strongest inductive
bias on this cohort and **the question is representation, not algorithm selection**. Identical
folds, identical held-out extractants, identical preprocessing.

| arm | ablation letter | features |
|---|---|---|
| `L0_GLOBAL_MEAN` | — | the training grand mean of `alpha` |
| `L0_NN_TANIMOTO` | — | the level of the nearest training extractant by Tanimoto |
| `L1_ECFP` | A | ECFP |
| `L2_GENERIC` | B | GENERIC |
| `L3_ECFP_GENERIC` | **D** | ECFP + GENERIC |
| `L4_COORD` | C | COORD |
| `L5_ECFP_COORD` | E | ECFP + COORD |
| `L6_GENERIC_COORD` | F | GENERIC + COORD |
| `L7_ALL` | **G** | ECFP + GENERIC + COORD |

`L0_NN_TANIMOTO` is mandatory rather than decorative: gen7 measured that on a level target no model
beat a 1-nearest-neighbour Tanimoto lookup, and a level claim that does not beat it is not a claim.

Hyperparameters come from Gen12's pre-declared grid — `max_features` in {0.30, sqrt},
`min_samples_leaf` in {1, 2} — chosen per outer fold on that fold's **inner validation block only**,
carved from the training chemotypes by Gen12's nested splitter. Level predictions are clipped to
the training level range plus one decade each side, Gen12's rule. Training extractants are
unweighted, one row per extractant; a reliability-weighted variant is a declared sensitivity.

Secondary learner checks, on the primary contrast only and never as a headline: XGBoost and
CatBoost. No graph network, no transformer, no architecture sweep.

## 7. The full-prediction architecture

`y_hat_ij = alpha_hat_i + delta_hat_ij`, built only after the level ladder is frozen. Per outer
fold, in this order:

1. level targets from training rows only;
2. level model fitted on the training extractants;
3. training residuals formed as `y_ij - alpha_i^train`, against the **observed training level**,
   never against a cross-fitted prediction — the repository measured that centring on a
   cross-fitted residual makes a two-stage model worse under a chemotype hold-out (1.25 against
   1.08) and that centring on the true cell mean is what works (1.04);
4. shape model fitted on those residuals;
5. the held-out extractant's level predicted from **structure only**;
6. its query residual predicted from conditions and structure;
7. the two summed.

Shape arms: `S0` conditions only; `S1` conditions + ECFP; `S2` conditions + COORD; `S3` conditions
+ ECFP + COORD.

**Direct comparators**, on identical query rows: Gen12's frozen `ABL_D_PLUS_LIG2D` and a new
`DIRECT_ALL` arm that is `ABL_D` plus the coordination block, so the coordination block is tested
in the direct formulation as well as the decomposed one.

## 8. Endpoints

**Primary.** The paired difference in **extractant-level MAE**, zero-shot, design B, full cohort,
between `L3_ECFP_GENERIC` (ablation D) and `L7_ALL` (ablation G): the incremental value of the
coordination block over the strongest matched generic representation. `E − A`
(`L5_ECFP_COORD` − `L1_ECFP`) is reported beside it as the isolation the brief also names.

**Secondary.** The paired difference in full-query **extractant-macro `log_D` MAE** between the
decomposed predictor built on `L3` and the one built on `L7`.

**Tertiary.** The change in **mean signed level error on the MULTI_ARM subgroup**. Gen12
under-predicts that family; the coordination block addresses that failure only if it reduces the
systematic negative bias, not merely the absolute error.

Every quantity carries its regime: {level | full `log_D`} x {full | level-reliable cohort} x
{zero-shot | one-shot} x {extractant-macro | chemotype-macro} x {overall | far | mid | near} x
{selected model | best observed test arm}.

## 9. Hypotheses

| id | statement | decision rule |
|---|---|---|
| **H1** | a large fraction of Gen12's zero-shot error is extractant-level bias rather than within-extractant shape error | quantified, not tested: the level and shape components of the frozen Gen12 predictions, the level-oracle score, and the between/within variance split |
| **H2** | coordination-aware descriptors improve zero-shot level prediction over the matched generic representation | paired level-MAE delta `L3 − L7` on the full cohort, BCa excluding zero and point delta above 0.01 |
| **H3** | improved level prediction improves full zero-shot `log_D` | paired extractant-macro delta between the two decomposed predictors, BCa excluding zero |
| **H4** | the coordination representation reduces the systematic under-prediction of multi-armed ligands | mean signed level error on MULTI_ARM moves towards zero, BCa on the paired change excluding zero |
| **H5** | any improvement is not confined to near chemistry | the H2 delta is reported separately on far, mid and near; a gain present only in the near band is reported as such and not as extrapolation |
| **H6** | better structural level prediction leaves less for one measurement to correct | `MAE(k=0) − MAE(k=1)` is smaller for the new model than for Gen12's, on byte-identical support and query draws. **H6 is not required for the generation to succeed** |

A hypothesis that fails is reported as failed. Non-significance is never reported as equivalence:
the margin is 0.02 and the MDE is printed before any contrast is interpreted.

## 10. The MULTI_ARM subgroup, defined before any model ran

`MULTI_ARM` is `coord__arm__local_donor_cluster_count >= 2` — two or more spatially separate
chelating pockets on the molecular graph. The threshold of 2 is the mono-topic / poly-topic
boundary and comes from structural semantics, not from any error table. Applied identically to all
183 structures it gives **43 multi-arm extractants in 21 chemotypes against 140 in 81**, with a
Kish effective sample size of **3.96 independent chemical units**. The counts are published in
`manifests/multi_arm_subgroup.csv` and in `DATA_AUDIT.md` §5, before any Gen12.2 model was fitted.

**That effective sample size is the binding constraint on the tertiary endpoint and is stated here
rather than discovered afterwards.**

## 11. Statistics

Gen12's implementation, imported unchanged: paired block bootstrap, resampling unit **chemotype**,
10,000 replicates, RNG seed 8675309, one index matrix shared by every comparison, BCa governing and
percentile reported beside it. Per-extractant values are averaged over split seeds before
resampling. Reported with every comparison: point delta, 95 % BCa, bootstrap SE, units improved of
units total, per-seed sign as consistency and never as significance, the minimum detectable effect
at 80 % power as 2.80 × SE, and leave-one-chemotype-out influence.

**Selected model and best observed test arm are separate concepts and are never conflated.** Gen12
caught exactly this: its pre-registered rule chose 1.092 while 1.048 was the best test score, and
both were reported. Selection here uses inner validation only.

## 12. Invariants

The run fails rather than proceeds if any of these is violated. They are executable in `tests/`
and re-checked against the artefacts on disk by `scripts/g122_self_audit.py`.

1. The design-B fold plan is byte-identical to Gen12's, fold by fold and row-id by row-id.
2. No canonical SMILES and no chemotype crosses a train/test boundary.
3. No coordination descriptor is a function of the target: rebuilding the matrix with the target
   replaced by noise is bit-identical.
4. The SMARTS specification digest at scoring time equals the digest frozen before the ladder ran.
5. Preprocessing statistics are fitted on training rows only.
6. No held-out extractant's own target participates in estimating its zero-shot level.
7. Level residualisation for the shape model uses training levels only.
8. Shape centring is per `(split_seed, extractant)`, never seed-pooled.
9. Every compared arm is scored on identical query rows and identical evaluation units.
10. One-shot comparisons use byte-identical support and query draws, reproducible across processes.
11. `MULTI_ARM` membership is a function of structure alone and does not depend on any model error.
12. Every transformed numerical input is finite, and the largest standardised magnitude per fold is
    recorded; clipping is reported per fold and may never conceal divergence.
13. Model selection reads inner validation only; the locked test is scored once per arm.
14. All headline pairwise tests operate on matched units.

## 13. Execution order

Phase 0 Gen12 audit and reproduction (done) → Phase 1 this freeze → Phase 2 coordination
descriptors (done, target-free) → Phase 3 descriptor validation and invariants → Phase 4 level
baselines → Phase 5 coordination level models → Phase 6 selection on inner validation → Phase 7
locked level test → Phase 8 decomposed predictor → Phase 9 direct versus decomposed → Phase 10
frontier → Phase 11 multi-arm family → Phase 12 one-shot mechanism → Phase 13 ablations and
grouped importance → Phase 14 bootstrap, power, influence → Phase 15 `DECISION_REPORT.md`.

## 14. Stopping and decision rules

- An effect below **0.01** is reported as null regardless of its interval.
- A representation is declared better only if the BCa interval on the paired chemotype-blocked
  delta excludes zero, the point delta exceeds 0.01, and no single chemotype flips its sign.
- If the primary contrast cannot resolve a plausible effect, that is stated **before** the result
  is interpreted and the null is reported as underpowered rather than as equivalence.
- **A clean negative result is a complete outcome.** If 2D coordination topology does not improve
  level prediction, that is the finding, and it says the next step is more independent chemistry,
  external pretraining or genuinely physical descriptors rather than another handcrafted block.

## 15. What this generation will not do

No generic model zoo. No descriptor selection on test. No manual tag for the ligands already known
to fail. No feature that exists only for molecules known to be mispredicted. No training on
same-chemotype chemistry under another metal. No deletion of one-condition extractants. No
averaging of seeds before level/shape decomposition. No test-best arm reported as the selected
model. No non-significance read as equivalence. No divergence hidden behind a target clip. No
claim of a representation improvement without a matched learner.

---

## Addendum 1 — 2026-09-05, before any level model was fitted

**What changed.** The level-definition decision rule's second override was amended to require that
the condition adjustment not itself be removing level. See `DATA_AUDIT.md` §7 and
`config/level_definition_rules.json` → `addendum_1`.

**Had results been seen?** No model had been fitted and no held-out extractant had been scored. The
evidence for the amendment is entirely training-only and model-free.

**What was not changed.** The development fold, the four criteria, the default, override 1, the
secondary-definition rule and the level-reliable-cohort rule are unchanged.

## Addendum 2 — 2026-09-05, before the level ladder was run

**What changed.** `DENTATE`, `coreCN`, `n_ligs` and `n_fill` — four of the fifteen columns of §5's
`DONORS` block — are built per experiment by the Architector complex recipe and vary between rows
of the same extractant for 10 of the 183. For the structure-only level model they are collapsed to
the per-extractant modal value, deterministically, with the affected count recorded. Any other
structural column that varies within an extractant is a fatal error.

**Why.** A column that varies by row is not a structural property, and admitting one into a
structure-only level model would let the experiment in through the back door.

**Had results been seen?** No. The condition was found by the assertion that guards
`structure_frame`, on the first attempt to build the level dataset.

## Addendum 3 — 2026-09-05, after the locked evaluation

**What changed in reporting, and nothing in the results.** An adversarial review of the code and
artefacts, run after the locked evaluation, found thirteen defects. Nine are reporting or tooling
defects and are fixed with the results unchanged: the percentile interval and the two-sided
bootstrap p-value now appear beside every BCa interval; `units_improved` and the per-seed sign are
oriented correctly for signed-error contrasts; band macro values are computed under the same
weighting as their delta; the band-overlap fact is published; level R² and dispersion average the
truth over seeds; the multiplicity count no longer counts its own output; and three invariant tests
that were weaker than their names now assert the property itself.

**Four are descriptor-definition defects** (`DECISION_REPORT.md` §13.13). They are **not** corrected
in the frozen specification. `config/coordination_smarts.json` version 1.1.0 remains what every
locked result was computed against and still reproduces its matrix bit-identically at hash
`da609be8fcd5f260`. A corrected specification is provided separately as
`config/coordination_smarts_posthoc.json` and its effect is measured as a labelled post-hoc
sensitivity in `analysis/descriptor_defect_sensitivity.md`.

**Had results been seen?** Yes, all of them. This addendum is written after the fact and is
labelled as such throughout. **No hypothesis verdict, no endpoint and no primary number changed**:
the pre-registered primary contrast is +0.034 before and after, and what changed is that its
percentile interval and p-value are now reported beside its BCa interval.
