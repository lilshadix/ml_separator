# Gen12Eu_pred — pre-registration

*Frozen 2026-09-05, after `DATA_AUDIT.md` and before any Gen12 model was fitted. No outcome-bearing
run had been executed when this document was written; the only quantities that existed were the
target-free cohort, split and similarity statistics in the audit.*

**Amendment rule.** This file is not edited once an outcome-bearing run starts. A change is a dated
addendum at the end, stating what changed, why, and whether any result had been seen. A statistic
substituted after seeing a result is reported as a correction with its mechanism, never as a bare
replacement.

---

## 1. Question

**Primary.** Can extractant molecular structure plus experimental conditions predict europium
extraction, `log_D(Eu)`, for extractants absent from training?

**Secondary.** How fast does prediction for a novel extractant improve as 1, 2, 3 and 5 Eu
measurements of that extractant become available?

**Tertiary, tested only after the Eu-only benchmark is locked.** Does training on other lanthanides
improve Eu prediction on strictly held-out extractants?

The three are kept apart by construction: no few-shot number may be quoted as a zero-shot number,
and the cross-lanthanide arm runs only after the zero-shot leaderboard is frozen.

## 2. Dataset fingerprint

| | |
|---|---|
| source | `dataset with 3D structures/dataset.parquet` |
| source SHA-256 | `fefbefc6fe993aa9ce9db1a0c338adb9e5f58a8b75bab084cc1df4e024faf5dd` |
| cohort fingerprint | **`2a364bb5264e9935`** |
| cohort | 1,329 rows, 183 extractants, 162 ECFP clusters, 97 chemotypes, 305 series |
| builder | `gen12eu.cohort.build_cohort` |

The builder refuses to run against a different bundle digest. Every artefact records the cohort
fingerprint; a table whose fingerprint does not match is not comparable and is rejected rather than
merged.

## 3. Cohort definition

One row per (extractant, condition) cell of the Eu subset, after the repository's TODGA
name/structure quarantine and after averaging replicate measurements within a cell. **No
minimum-rows-per-extractant filter.** Single-row extractants are retained: they carry a level error
and are exactly the chemically unusual ligands a coverage filter would delete.

## 4. Target

`log_D`, base-10 logarithm of the distribution ratio, dimensionless, finite on all 1,329 rows.
Untransformed. No winsorisation, no floor, no clipping of the truth. **This is fixed and will not
be replaced after results are seen.**

A level/shape decomposition is a declared *secondary* view, never a substitute endpoint. It is
computed strictly per `(split_seed, extractant)`: gen11 measured that centring residuals on a mean
pooled across split seeds scores between-seed level wobble as shape and that this reached a
published headline number. Extractants with fewer than two rows contribute a level error and no
shape statistic, and are counted separately.

## 5. Feature groups

| block | columns | contents |
|---|---|---|
| `COND` | 64 | acid, diluent and additive one-hots plus 5 continuous conditions |
| `MASSACT` | 8 | log10 of the continuous conditions and the `n·log[L]` products of the mass-action law |
| `ECFP` | 2,048 (662 informative) | the bundle's Morgan radius-2, 2,048-bit achiral fingerprint |
| `PHYSCHEM` | 10 | generic RDKit scalars |
| `DONORS` | 15 | donor-atom census, denticity, coordination number, stoichiometry |
| `LIG2D` | 206 | extended RDKit and hand-crafted 2D descriptors |

**Excluded on purpose, with reasons.** Metal-identity columns are constant at Eu and are dropped.
The 3D geometry blocks are not used: they cover only 1,370 of the raw Eu rows, and the repository's
own verdict across gen2, gen3 and gen5 is that they do not add transferable signal over 2D.
Pretrained ligand embeddings are not used in the primary ladder. Provenance columns — DOI,
archive ids, review flags, `safe_exp_id` — are audit metadata and a forbidden-token assertion
raises if any reaches a feature matrix.

`LIG2D` is **not** in the default arm. gen7 measured that deleting this exact 206-column block
significantly improved the champion, the only block whose removal was a significant gain. It enters
only as ablation arm D.

## 6. Splits

Both designs: `seeded_group_kfold`, 5 folds x 5 split seeds `{104729, 130363, 155921, 196613,
262147}`. Folds are new — the cohort and task differ from gen11's — but the seeds are reused so
split-to-split variation is comparable across generations.

- **Design B, chemotype hold-out — primary.** Groups are the frozen gen6 Tanimoto-0.7 single-linkage
  super-clusters over all 190 bundle structures. Measured: 0 extractant, 0 ECFP-cluster and 0
  chemotype overlap; max train similarity 0.698.
- **Design A, exact-extractant hold-out — sensitivity only.** Measured: 139 ECFP-cluster and 234
  chemotype overlaps across 25 folds, max train similarity 1.000. **May not carry a zero-shot
  headline.**

**Inner validation is nested.** Within each outer training set the same grouped splitter carves a
validation block from the training groups only (4-way, first block, inner seed
`seed*31 + fold*7919 + 17`). Hyperparameters, early stopping and every model choice are decided on
that block, per outer fold. No outer test row participates in any fitting or any selection.

Model seed is fixed: `42 + fold*1009 + 9,999,991`, independent of the split seed, matching gen5
through gen11.

## 7. Similarity bands

`max_train_tanimoto` for a held-out extractant is its maximum Tanimoto to the **training**
extractants of that fold — never to the rest of the test set, never to the whole cohort, never
using the target.

| band | rule |
|---|---|
| far | <= 0.40 |
| mid | 0.40 to 0.60 |
| near | > 0.60 |

gen10/gen11 cut points, reused unchanged and **frozen now**. Population under design B, measured
before any model ran: far 27 extractants / 25 chemotypes / ~17 per seed; mid 103 / 59 / ~71;
near 105 / 58 / ~94. Under design B similarity is capped at 0.698, so "near" is the slice
0.60 to 0.698 and is described that way wherever it is quoted.

## 8. Metrics

**Primary: `macro_mae_extractant`** — the MAE within each held-out extractant, averaged with equal
weight over extractants, computed per split seed and then averaged over seeds.

Secondary, all reported for overall / far / mid / near: chemotype-macro MAE, pooled row MAE, RMSE,
median absolute error, pooled R², Spearman, calibration slope and intercept, per-extractant bias,
level/shape decomposition, fraction within 0.5 and 1.0 decades, and error against
`max_train_tanimoto`.

A row mean is never called MAE without the word "pooled". No number is quoted without its regime:
{zero-shot | k-shot}, {design B | design A}, {extractant-macro | chemotype-macro | row}, {overall |
far | mid | near}.

## 9. Statistical evaluation

Paired block bootstrap, resampling unit **chemotype**, 10,000 replicates, RNG seed 8675309. One
index matrix is drawn and shared by every comparison and statistic. Per-extractant values are
averaged over seeds before resampling: the five seeds re-partition the same 183 extractants and are
not five experiments. Percentile and BCa intervals are both reported; BCa governs.

Reported alongside every comparison: point delta, 95 % interval, bootstrap SE, units improved of
units total, per-seed direction (as consistency, never as significance), and the minimum detectable
effect at 80 % power, `MDE = 2.80 x SE`, computed from the realised SE.

Leave-one-chemotype-out influence is mandatory: the largest chemotype holds half the rows, so any
headline delta is recomputed with each chemotype removed and the largest influence is reported.

Robustness variant: the same bootstrap blocked on DOI instead of chemotype, since 24 of 80
publications span more than one chemotype.

**A non-significant difference is not evidence of equivalence.** Where the question is "does X add
nothing", the equivalence margin is declared now: **0.02 macro MAE**, twice the repository's
measured cross-machine reproducibility floor of ~0.01.

## 10. Models

Every arm sees identical folds, identical test rows and identical fold-local preprocessing
(median imputation with a missingness indicator, fitted on the training rows of that fold; scaling
only where the learner needs it).

**Tier 0 — sanity baselines**
- `B0_GLOBAL_MEAN` — the training-set mean.
- `B1_COND_ONLY` — conditions and MASSACT only, no molecular information of any kind. Gradient
  boosting. This is how much is predictable without chemistry.
- `B2_NN_CHEMICAL` — for a test extractant, take the most similar training extractants by Tanimoto
  and predict from their behaviour under the nearest available conditions, with a leakage-safe,
  target-free selection rule. A sophisticated model must beat neighbour interpolation.

**Tier 1 — fingerprint and tabular**
- `T1_CATBOOST`, `T1_XGBOOST`, `T1_RF` on ECFP + PHYSCHEM + DONORS + COND + MASSACT.
Hyperparameters from a pre-declared grid of at most 6 configurations per family, chosen per outer
fold on the inner validation block. **The best Tier-1 arm is the principal baseline every neural
model must beat.**

**Tier 2 — descriptor MLP.** Standardised inputs, 2 hidden layers, dropout, weight decay, early
stopping on the inner validation block, 3 model seeds averaged. Deliberately modest: there are 183
independent extractants, not 1,329 independent rows.

**Tier 3 — Chemprop v2 D-MPNN.** Molecular graph from SMILES; conditions enter *after* message
passing as an extra datapoint-descriptor vector concatenated to the learned molecular embedding.
Variants: graph only; graph + conditions; graph + conditions + RDKit descriptors. Same folds, same
test rows, descriptor scaling fitted on training rows only.

**Two rules that apply identically to every arm**, fixed here before any model ran:

- **Training rows are weighted so that each training extractant contributes equal total mass.** The
  primary metric is extractant-macro MAE, and one extractant is 23 % of the cohort rows; an
  unweighted fit optimises a different quantity from the one being reported. This is the
  repository's own practice ("sample weights make the model optimise exactly this").
- **Predictions are clipped to the training target range extended by one decade at each end.** A
  neural arm that extrapolates to an absurd value would otherwise be penalised on an axis this
  study is not about. Trees are unaffected; the rule is applied to every arm so it cannot favour one.

**Not primary, by decision.** No CNN: a convolution has no motivated structure over a folded
fingerprint or a tabular condition vector. A Siamese / pairwise-delta model is an optional
*ablation* after the ladder is complete, with `lambda_pair` tuned on validation only, and it does
not replace direct regression.

## 11. Zero-shot protocol

k = 0 means no Eu measurement of a held-out extractant enters training, preprocessing, calibration,
feature construction or model selection. Predictions are produced for every eligible condition of
every held-out extractant. Headline: **zero-shot extractant-macro MAE under design B**, plus its
far / mid / near values.

## 12. Few-shot protocol

Run only after the zero-shot models are frozen. k in {1, 2, 3, 5}. For each held-out extractant,
k support rows are chosen and every remaining row is a query; support rows are never scored. Draws
are deterministic in `(seed, repeat, extractant, n_rows)` through BLAKE2b — Python's `hash` is
salted per process and silently broke cross-run pairing in gen8. Every competing model sees
**byte-identical support and query sets**.

- **Adapter A, residual level calibration.** `b̂` is a shrinkage-corrected mean of the support
  residuals; `f_k(x) = f(x) + b̂`. The shrinkage constant is chosen on training-fold out-of-fold
  residuals only.
- **Adapter B, limited head adaptation.** Neural encoder frozen, small head or affine adapter
  updated on the k support points, strongly regularised. Compared against Adapter A; if it does not
  beat the trivial calibration this is reported as such.

Required nulls, taken from the repository's own rules: `PAIRMEAN + the same k measurements`, and a
no-model fit on the same k points. A k-shot claim that does not beat both is not a claim.

Learning curves are drawn on the **common cohort** — 42 extractants in 21 chemotypes with at least
7 rows — so the curve is not confounded with a shrinking cohort. Level and shape improvements are
reported separately, centred per `(split_seed, extractant)`.

## 13. Cross-lanthanide protocol

Two arms, same Eu test extractants, same folds, same metrics.

- `EU_ONLY` — the frozen best zero-shot arm, trained on Eu rows only.
- `MULTI_LN_STRICT` — the same architecture with metal identity as an explicit input, trained on
  all 14 lanthanides, **after deleting every row of every held-out chemotype under every metal**.
  Evaluated only on Eu rows.

The audit measured 4,148 non-Eu rows on Eu cohort extractants and only 7 non-Eu extractants that
are new chemistry, so without this deletion the arm would have seen its own test chemistry. A
per-fold auxiliary-row count and a zero-overlap assertion are published with the result.

**Matched controls, required before any transfer claim is believed.**
1. Row-count-matched Eu-only control: Eu rows up-weighted to the multi-metal arm's effective
   training mass.
2. Metal-label permutation: identical rows, metal identity shuffled within the auxiliary block.
3. Target-shuffled auxiliary rows.

gen11 showed how badly auxiliary block size, diversity and weighting confound the chemistry
reading. "More rows wins" is not transfer.

## 14. Hypotheses

| id | statement | decision rule |
|---|---|---|
| **H1** | structure + conditions beats condition-only and trivial baselines on unseen extractants | zero-shot design-B extractant-macro MAE improvement over the better of `B0`/`B1`, BCa lower bound > 0 |
| **H2** | a graph or fingerprint model generalises beyond near-neighbour chemistry | non-trivial far-band performance: the best arm beats `B1` on the far band with BCa excluding zero |
| **H3** | support measurements monotonically improve prediction | macro MAE decreasing in k on the common cohort at k = 1, 2, 3, 5, each beating the k-shot nulls |
| **H4** | multi-lanthanide training improves Eu prediction on strictly held-out extractants | `MULTI_LN_STRICT` beats `EU_ONLY` by >= 0.02 macro MAE with BCa excluding zero |
| **H5** | any multi-Ln gain survives the controls | H4's effect persists against all three §13 controls |

Hypotheses are not rewritten after results. A hypothesis that fails is reported as failed.

## 15. Stopping and decision rules

- An effect below **0.01 macro MAE** is indistinguishable from a change of machine and is reported
  as null regardless of its interval.
- A model is declared better than the principal Tier-1 baseline only if the BCa interval on the
  paired chemotype-blocked delta excludes zero **and** the point delta exceeds 0.01 **and** the
  leave-one-chemotype-out influence does not flip its sign.
- Model selection for the locked test evaluation uses **inner validation only**. The outer test
  rows are scored once per arm.
- If the far band cannot resolve a plausible effect, that is stated before the band is interpreted,
  and the null is reported as underpowered rather than as equivalence.
- A valid negative result — nothing beats the baseline — is a complete outcome and is reported as
  the finding.

## 16. Invariants

The run fails rather than proceeds if any of these is violated; they are enforced by
`tests/` and re-checked by `scripts/gen12_self_audit.py`.

1. Zero exact-extractant overlap between train and test in every fold of every design.
2. Zero canonical-SMILES and zero chemotype overlap under design B.
3. Eu test extractants do not appear under any other metal in strict multi-Ln training.
4. No test row enters any preprocessing fit.
5. No test target enters feature construction; a forbidden-token assertion covers every block.
6. No support point is scored as a few-shot query.
7. Identical test queries across compared models.
8. Identical few-shot support draws across compared models, reproducible across processes.
9. Model selection uses inner validation only.
10. Target values are unchanged by any model-specific pipeline.
11. Band labels depend only on molecular similarity to training, never on the target.
12. Every bootstrap comparison operates on matched evaluation units.

## 17. Execution order

Phase 0 audit (done) -> Phase 1 freeze (this document) -> Phase 2 sanity baselines -> Phase 3
boosted trees -> Phase 4 MLP -> Phase 5 D-MPNN -> Phase 6 selection on validation -> Phase 7 locked
zero-shot test -> Phase 8 similarity frontier -> Phase 9 few-shot -> Phase 10 multi-lanthanide ->
Phase 11 controls and ablations -> Phase 12 power, bootstrap, influence -> Phase 13
`DECISION_REPORT.md`.

## 18. Feature ablations

On the winning architecture only, six arms: A conditions only; B molecule only; C molecule +
conditions; D C + RDKit `LIG2D`; E ECFP + conditions; F graph + conditions. The purpose is
mechanistic attribution — how much is conditions, how much is ligand structure, does graph learning
add anything over fingerprints — not a combinatorial sweep.

## 19. Required artefacts

Cohort manifest and fingerprint; dataset SHA-256; exact split memberships; model configs;
dependency versions; seeds; per-row predictions for every arm; per-extractant and per-chemotype
metrics; near/mid/far tables; bootstrap inputs and distributions; few-shot support/query
membership; leaderboard; pairwise comparison table; power analysis; figures; `DECISION_REPORT.md`.


---

## Addendum 1 — 2026-09-05, after results were seen

**What changed.** Standardised feature preprocessing now clamps a held-out value to the training
fold's observed range before standardising (`FoldPreprocessor(clip_to_train_range=True)`). It is
enabled for every model that *multiplies* its inputs — the nearest-chemistry baseline's condition
kernel, the descriptor MLP and all three D-MPNN variants — and for no tree arm, which only compares
its inputs and is unaffected.

**Why.** A numerical defect, not a modelling preference. `lig2d__rd__Ipc` spans 1.1e8 to 3.8e29
across these 183 ligands. Under a chemotype hold-out the extreme molecule is frequently absent from
the training fold, whose standard deviation is then 5.3e12, so the held-out value standardises to
**7.3e16**. Measured across all 25 design-B folds the largest standardised value handed to a model
was **7.255e16 before the guard and 33.93 after**. The consequence was visible: `T3_DMPNN_COND_DESC`
diverged in **8 of 25 folds**, with inner-validation MAE up to 3.5e11, and the pre-registered
prediction clip partially hid it — the arm still produced plausible per-fold MAEs because every
prediction was clamped to the training target range.

**Had results been seen?** Yes. The defect was found by reading the per-fold selection log of an arm
that had already run, because its mean inner-validation MAE was 1.4e10. This addendum is therefore
written after the fact and is labelled as such.

**How it is reported.** The five affected arms were re-run. Their pre-guard predictions are
preserved under `predictions/B_pre_clip_guard/` and both sets of numbers appear in
`DECISION_REPORT.md`. No tree arm, no ablation arm, no cross-lanthanide arm and no few-shot number
changes, because none of them uses standardised inputs. The headline zero-shot arms are trees and
are untouched, so **no hypothesis verdict depends on this change**.

**What was not changed.** The clamp is not applied to tree preprocessing, the prediction clip is
unchanged, and no hyperparameter, split, seed, metric or hypothesis was altered.
