# Gen12.2 — data audit

*Written before any Gen12.2 model was fitted. Gen12.2 inherits Gen12's cohort, splits, bands and
statistics unchanged and adds one thing: a coordination-topology descriptor block and a level
target. This audit covers what is new and what Gen12.2 had to verify about what it inherited; it
does not restate `gen12_eu_pred/DATA_AUDIT.md`, which remains the audit of the cohort itself.*

**Directory naming.** The brief calls this generation "Gen12.2" in its title and
`Gen1Eu_pred-2/` in its artefact list. The repository's convention is `gen12_eu_pred/`, so the
directory is `gen12_2_eu_pred/`. It is the directory the brief asks for under the repository's
own naming.

---

## 1. What is inherited, and the proof that it is the same object

Gen12.2 does not copy Gen12's cohort builder, splitter, metrics, bootstrap or few-shot code. It
**imports them**, by putting `gen12_eu_pred/` on `sys.path`. "The same frozen design-B folds" is
therefore a fact about the code path rather than an intention.

Verified before any Gen12.2 model ran:

| check | result |
|---|---|
| bundle SHA-256 | `fefbefc6…4faf5dd`, unchanged |
| cohort fingerprint rebuilt from the bundle | **`2a364bb5264e9935`**, identical to Gen12's |
| design-B fold plan rebuilt and compared row-id by row-id against `gen12_eu_pred/predictions/B/fold_plan.json` | **0 mismatches in 25 folds**, and identical held-out chemotype lists |
| Gen12 invariant suite | **23 of 23 pass** |
| Gen12 self-audit against the artefacts on disk | **14 of 14 pass** |
| Gen12 report verifier | **67 of 67 quoted numbers recomputed** |
| Gen12 zero-shot leaderboard recomputed from the saved per-row predictions | every arm reproduces to 4 decimals |

Three Gen12 statements this generation depends on were recomputed rather than quoted:

| Gen12 statement | recomputed here |
|---|---|
| 63 % of the target variance is between extractants | **0.6324** between, 0.3676 within |
| a perfect per-extractant level with no shape model scores 0.334 | **0.3345** extractant-macro MAE |
| the champion's error is mostly level: offset 0.917, shape 0.578 | 0.9170 / 0.5783, identical |

**Nothing in `gen12_eu_pred/` is modified by anything in this generation.** Gen12 is opened
read-only; every Gen12.2 artefact is written under `gen12_2_eu_pred/`.

## 2. The level target

The unit of the level task is the **extractant**, not the row. Under design B every extractant is
held out exactly once per split seed with *all* of its cells, because the split groups chemotypes
and an extractant's rows all share its chemotype. The evaluation level of an extractant is
therefore the same number in all five seeds, and the level task has exactly **183 evaluation
units in 97 chemotypes**.

Distribution of the primary level target, the per-extractant mean of `log_D` (§ `PRE_REGISTRATION.md` §3):

| statistic | value |
|---|---|
| n extractants | 183 |
| mean | −0.390 |
| **between-extractant sd** | **1.7465** |
| median | −0.208 |
| min / max | −3.599 / 3.021 |
| interquartile range | −1.960 to 0.955 |

Within-extractant residual sd, measured on the development fold's training rows: **0.9714**. The
ratio of the two is what sets the level-reliable cohort threshold (§4).

## 3. The coordination descriptor block

Built from the canonical SMILES of the 183 cohort structures and from nothing else.

| | |
|---|---|
| specification | `config/coordination_smarts.json`, version **1.1.0**, SHA-256 `4b073660…` |
| structures | 183, all parsed by RDKit 2026.03.5 without error |
| columns defined | **114** |
| columns informative (non-constant on this cohort) | **109** |
| constant on this cohort | 5 — formal charge, the two phosphine-P columns, ester carbonyl O, and the aryl-ether fraction |
| feature matrix content hash | `da609be8fcd5f260` |
| families | donor 23, motif 23, arm 17, dist 22, arch 24 |

Two properties were asserted before the matrix was used:

- **Determinism.** Rebuilding the table from a shuffled input list reproduces it bit-identically,
  so no descriptor depends on row order.
- **No heavy tail.** Gen12's neural arms diverged in 8 of 25 folds because `lig2d__rd__Ipc` spans
  1.1e8 to 3.8e29 and standardised to 7.3e16 under a chemotype hold-out. The largest ratio of
  maximum value to standard deviation over the 109 informative coordination columns is **20.58**.
  The block is counts and short graph distances and cannot produce that failure, but the check is
  run and recorded rather than assumed.

## 4. Cohorts

| cohort | rule | extractants | chemotypes | cells | Kish n_eff by chemotype |
|---|---|---|---|---|---|
| **full** | every eligible extractant | 183 | 97 | 1,329 | **13.70** |
| **level-reliable** | at least 5 measured cells | 71 | 41 | 1,179 | **11.75** |

The threshold of 5 comes from the rule frozen in `config/level_definition_rules.json` before it was
computed: the smallest cell count at which the standard error of a per-extractant mean,
0.9714/√n, falls to a quarter of the between-extractant spread of 1.7465. At n = 5 that is 0.434
against a target of 0.437.

**One-cell extractants are not deleted.** 85 of 183 have exactly one cell and they are exactly the
chemically unusual ligands a coverage filter would remove; gen5 measured that its `min_rows = 10`
filter discarded 37 of its 47 chemically novel extractants. Both cohorts are reported for every
primary contrast and the full cohort is the headline.

## 5. The pre-registered MULTI_ARM subgroup

`MULTI_ARM` is `coord__arm__local_donor_cluster_count >= 2`: the molecule presents two or more
spatially separate chelating pockets on its molecular graph. The rule is structural, applied
identically to all 183 structures, and was computed before any Gen12.2 model was fitted.

| | MULTI_ARM | single-pocket |
|---|---|---|
| extractants | **43** | 140 |
| chemotypes | 21 | 81 |
| cells | 250 | 1,079 |
| **Kish n_eff by chemotype** | **3.96** | 21.92 |

By similarity band: far 4 / 11, mid 28 / 35, near 11 / 94. By scaffold family the multi-arm set is
28 diglycolamides, 6 other amides, 3 malonamides, 3 N-heterocyclic polydentates, and one each of
phosphoryl, sulfur-donor and hydroxy-acid.

**The binding constraint on the tertiary endpoint is stated here rather than discovered later:
the multi-arm subgroup rests on an effective sample size of about four independent chemical
units.** Its minimum detectable effect is computed and printed before the subgroup result is
interpreted.

## 6. Leakage hazards specific to this generation

| hazard | status | mechanism |
|---|---|---|
| a coordination descriptor computed from the target | **impossible by construction** | `coordination.descriptors_for` takes one SMILES string and has no other argument; a test rebuilds the matrix with the target replaced by noise and asserts the bytes are identical |
| SMARTS tuned to the molecules that are known to fail | **guarded** | the specification is a versioned file with a SHA-256 recorded beside every feature matrix; a test asserts the digest at scoring time matches the digest frozen before the level ladder ran |
| the level definition chosen to flatter a result | **guarded** | the decision rule was written to `config/level_definition_rules.json` and its digest recorded *before* the study script that reads it was run; the amendment to it is dated and its evidence is training-only (§7) |
| a held-out extractant's own targets used to predict its level | **forbidden** | the level model is fitted on the training extractants of the fold; the held-out extractant contributes only structural features |
| the condition model that defines the secondary level seeing its own extractant | **cross-fitted** | the condition-only model is cross-fitted over training chemotypes, so a training row's condition prediction comes from a model that never saw its chemistry |
| shape residuals centred on a test-derived mean | **forbidden** | training residuals are formed against the training level; a held-out extractant's level is never computed before its prediction |
| a subgroup defined from model error | **forbidden** | `MULTI_ARM` is a structural rule with its counts published above, before any model ran |

## 7. One correction made during Phase 1, recorded rather than absorbed

The frozen level-definition rule fired an override that selected the condition-adjusted level, and
that override was mis-specified. It assumed a large gap between the raw per-extractant mean and
the condition-adjusted level means the raw mean is distorted by heterogeneous condition sampling.
Three training-only measurements say it means something else:

- the cross-fitted condition-only model explains **8.9 %** of pooled row variance, but its
  per-extractant mean prediction has a variance equal to **43.9 %** of the between-extractant
  variance of the target and correlates with the true level at **Spearman 0.55**. Subtracting it
  removes a large part of the genuine level, because each extractant was measured at its own
  acidities, diluents and series. gen7 measured the same mechanism from the other side: its
  raw-level "capture" was really learning each ligand's conditions;
- the shift does not shrink where the sampling is thinnest, which is what a sampling artefact
  would do. Median absolute shift by cell count: 0.63 at one cell, **2.03 at two to three**, 1.67
  at four to six, 1.06 at seven to twenty, 0.33 above twenty. Correlation with cell count is 0.088;
- the condition-adjusted level has the **worst split-half reliability** of the five candidates,
  Spearman 0.841 against 0.920 for the raw mean.

The override was amended to require that the adjustment not itself be removing level. Its second
and third conditions fail, so the default stands. This happened after the training-only study and
**before any level model was fitted or any held-out extractant scored**, which is the phase in
which the definition is supposed to be settled. Both the original rule and the amendment are in
`config/level_definition_rules.json`; the study output records what the original rule would have
chosen.

## 8. Environment

Identical to Gen12's, and checked rather than assumed: Python 3.13.0, numpy 2.5.1, pandas 3.0.5,
scipy 1.18.0, scikit-learn 1.9.0, RDKit 2026.03.5, CatBoost 1.2.10, XGBoost 3.4.1, PyTorch 2.13.0.
**No package is installed for this generation.** The repository has been moved once by an unrelated
`pip install` that silently downgraded scikit-learn and pandas and shifted every result by about
0.0014 against effects of 0.004 to 0.02; RDKit is already present and is the only new import.
