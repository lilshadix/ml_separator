# Gen12.2 — decision report

*Coordination-aware prediction of the intrinsic europium extraction level on chemically unseen
extractants. Cohort fingerprint `2a364bb5264e9935`, inherited from Gen12 and rebuilt fold-by-fold
against its frozen plan. Every number is produced by the scripts in `scripts/` and stored under
`headline_tables/`, `metrics/`, `bootstrap/` and `analysis/`. Each quantity carries its regime:
{level | full log_D} x {full | level-reliable cohort} x {LVL_MEAN | LVL_COND_RESIDUAL} x
{zero-shot | one-shot} x {extractant-macro | chemotype-macro} x {overall | far | mid | near} x
{selected model | best observed test arm}. A number without its regime is not a number in this
study.*

---

## Headline

**Gen12 was right that the level is the bottleneck. Coordination topology carries real information
the fingerprint lacks — but not enough, on top of what Gen12 already had, for this cohort to
resolve, and none of it survives one experimental measurement.**

Three findings, in decreasing order of how well they are supported.

**1. The level is where the error is, decisively.** Replacing Gen12's champion's predicted level
with the true level, keeping its own within-extractant shape, takes extractant-macro MAE from
1.048 to **0.310**. Doing the reverse takes it to 0.917. **The level is worth 0.738 and the shape
0.131.** 56 % of the squared error sits in the level, and the correlation between an extractant's
level error and its overall error is 0.883.

**2. Coordination descriptors add information over a fingerprint, and that is the one robust
positive.** Adding them to ECFP improves zero-shot level MAE from 1.080 to **1.021**, a paired gain
of **+0.060**, percentile 95 % [+0.016, +0.092], BCa [+0.020, +0.095], **two-sided bootstrap
p = 0.007**, realised power 0.86. Inside the diglycolamide family, *replacing* the fingerprint with them
outright is worth **+0.368 [+0.072, +0.454], p = 0.012**.

**3. The pre-registered primary endpoint is a borderline positive that should not be quoted
alone.** Adding the block on top of Gen12's *full* molecular representation improves level MAE from
1.062 to 1.028, **+0.034**. Under the pre-registered rule that BCa governs, its interval
[+0.004, +0.075] excludes zero. **The percentile interval from the identical 10,000 draws is
[−0.003, +0.063] and includes zero, and the two-sided bootstrap p is 0.092.** Realised power is
**0.46**, so a significant estimate here is inflated by about **1.5x**. The gain also fails to
replicate under the declared secondary level definition (+0.007) and reverses under XGBoost
(−0.010). **H2 passes on its own pre-registered rule and on nothing else, and the honest summary is
that it is unresolved rather than established.**

**A post-hoc sensitivity says part of that fragility was our own descriptors.** An adversarial
review found four rule-level defects in the block after the locked evaluation — aromatic donors
invisible, a mis-defined lactam test, a ring class that contradicted its own written definition, and
a double-counted motif sum. With them corrected the primary endpoint becomes **+0.046 with both
intervals excluding zero and p = 0.020**. It is a sensitivity, not a replacement: the corrections
were chosen knowing which columns were wrong, and the frozen block remains the basis of every
headline number here.

**Why the two contrasts differ is the mechanism.** The coordination block's marginal value shrinks as the
baseline gets richer: **+0.060 over a bare fingerprint, +0.042 over the generic descriptors,
+0.034 over both.** 30 of its 114 columns are rank-identical to a column Gen12 already had. Grouped
permutation importance says the working part is the **donor census** (0.037) and that
**binding-arm multiplicity is the block's weakest family** (0.010) — below Gen12's own 15-column
donor census. **The generation's own central construct is the part that contributes least.**

**H3, H4, H5 and H6 all fail.** The level gain does not carry into full `log_D` (+0.028, p = 0.32).
Explicit level-plus-shape decomposition is **worse** than direct regression, significantly so for
the generic level model (−0.089, p = 0.002). Multi-armed ligands stay under-predicted by 1.13
decades and the block moves that by 0.04. The gain is not established on far or mid chemistry. And
**one measurement of the new extractant erases the representation difference entirely**: 1.219
against 1.243 zero-shot becomes 0.9647 against 0.9655 after a single point.

**One limitation applies to everything above and cannot be fixed inside this generation.** The
coordination block exists *because* Gen12 reported that bridged and multi-armed diglycolamides were
under-predicted — an observation made on the held-out predictions of **the same 183 extractants**
this generation evaluates on. No descriptor was fitted to a test target, the specification was
frozen and digested before any model ran, and the folds are Gen12's own; but the *hypothesis* was
generated from the same data the primary endpoint tests it on. **A pre-registration protects
against fitting to the test set and not against forming the question from it**, and no p-value here
accounts for that. The +0.034, and more so the +0.076 inside the diglycolamide family, should be
read as the size of an effect worth testing on new chemistry, not as a confirmed one.

---

## 1. What was run

| phase | what | where |
|---|---|---|
| 0 | Gen12 audit and reproduction | `DATA_AUDIT.md` §1 — 23 Gen12 tests, 14 self-audit checks, 67 report numbers reproduced |
| 1 | level definition frozen on training rows only | `analysis/level_definition_study.md`, `config/level_definition_rules.json` |
| 2 | coordination descriptors built and audited | `COORDINATION_DESCRIPTOR_SPEC.md`, `features/` |
| 3 | exploratory representation analysis, training rows only | `analysis/exploratory_level_analysis.md` |
| 4–7 | level ladder, selection on inner validation, locked level test | `headline_tables/t1`, `t2` |
| 8–9 | decomposed predictor, direct versus decomposed | `headline_tables/t6`, `t7` |
| 10 | frontier, bands and continuous similarity | `headline_tables/t3`, `metrics/frontier_sextiles.csv` |
| 11 | Gen12's failure family, reproduced | `analysis/gen12_failure_family.md` |
| 12 | one-shot mechanism | `headline_tables/t8` |
| 13 | ablations and grouped importance | `headline_tables/t9` |
| 14 | bootstrap, power, influence, statistical caveats | `headline_tables/t4`, `t10`, `t11` |
| — | post-hoc: do four descriptor defects change the answer | `analysis/descriptor_defect_sensitivity.md` |
| 15 | this report | — |

22 Gen12.2 invariant tests pass. 17 of 17 self-audit checks pass **against the artefacts on disk**,
including a re-hash of every Gen12 artefact against Gen12's own manifest: **nothing in
`gen12_eu_pred/` was modified**, and the frozen coordination matrix still reproduces bit-identically
at hash `da609be8fcd5f260`.

## 2. Was Gen12 right that unseen-extractant level is the principal bottleneck?

**Yes, and by a wider margin than Gen12 stated.** 63.2 % of the target variance sits between
extractants; the between-extractant sd of the level is 1.747 against a target sd of 1.898.

Zero-shot, design B, extractant-macro MAE, on Gen12's frozen predictions:

| Gen12 arm | as measured | true level, own shape | true shape, own level | true level, no shape model |
|---|---|---|---|---|
| `ABL_D` (best observed test) | 1.048 | **0.310** | 0.917 | 0.334 |
| `ABL_C` (pre-registered selection) | 1.092 | 0.305 | 0.977 | 0.334 |
| `ABL_E` (fingerprint + conditions) | 1.107 | 0.301 | 0.994 | 0.334 |
| `ABL_A` (no molecular information) | 1.134 | 0.328 | 1.029 | 0.334 |

**Fixing the level recovers 0.738; fixing the shape recovers 0.131.** The Spearman correlation
between an extractant's level error and its overall error is **0.883**, and 56 % of the squared
error sits in the level component. **H1 is supported.**

The corollary matters as much: **the shape is close to solved.** No arm in this generation moved
the shape component by more than 0.06, and the best shape model in the whole study uses no
molecular information at all (§7).

## 3. Can molecular structure predict the extractant-specific Eu level?

**Weakly, and barely better than looking up the nearest molecule you have already measured.**
Zero-shot, design B, full cohort of 183 extractants in 97 chemotypes, primary definition `LVL_MEAN`,
extremely randomised trees:

| arm | letter | level MAE | chemotype-macro | signed bias | between-extractant R² | Spearman | inner validation |
|---|---|---|---|---|---|---|---|
| `L4_COORD` | C | **0.962** | 0.881 | −0.260 | **0.486** | 0.718 | 0.997 |
| `L5_ECFP_COORD` | E | 1.021 | 0.861 | −0.433 | 0.367 | 0.648 | 0.967 |
| **`L7_ALL`** | **G** | 1.028 | 0.862 | −0.410 | 0.373 | 0.653 | 0.967 |
| `L6_GENERIC_COORD` | F | 1.032 | 0.862 | −0.377 | 0.364 | 0.649 | **0.965** |
| **`L3_ECFP_GENERIC`** | **D** | 1.062 | 0.868 | −0.459 | 0.332 | 0.637 | 0.973 |
| `L2_GENERIC` | B | 1.074 | 0.870 | −0.404 | 0.307 | 0.604 | 0.972 |
| `L1_ECFP` | A | 1.080 | 0.907 | −0.484 | 0.309 | 0.609 | 0.993 |
| `L0_NN_TANIMOTO` | — | 1.100 | 0.929 | −0.240 | 0.293 | 0.628 | 1.155 |
| `L0_GLOBAL_MEAN` | — | 1.596 | 1.532 | −0.143 | −0.145 | −0.400 | 1.505 |

Every arm beats predicting the training mean by 0.5 or more (p < 0.001). Against the
**1-nearest-neighbour Tanimoto level lookup**, on the full cohort, only coordination-only clears
zero and only just: +0.138, percentile [−0.003, +0.265], p = 0.053. The generic representation is
+0.037 with p = 0.66. **This reproduces gen7's finding on a new target and a new metal: on a level
target no model convincingly beats a 1-NN lookup.** On the level-reliable cohort of 71 extractants
the picture is better and robust — `L7_ALL` beats the lookup by +0.217 (p = 0.007) and `L4_COORD`
by +0.271 (p = 0.002) — so the lookup's competitiveness on the full cohort is partly the
one-cell extractants, where the truth itself is a single measurement.

Between-extractant variance captured zero-shot is 33 % for the generic representation, 37 % with
the coordination block and 49 % for coordination alone. Against the 63.2 % of total variance that
is between extractants, that is **21 % to 31 % of the total target variance recovered from
structure alone**. Every arm is shrunk: dispersion ratio 0.55 to 0.63 against a truth spread of
1.75.

## 4. Do coordination-topology descriptors improve level prediction beyond ECFP and generic descriptors?

**The answer depends on what they are added to, and saying only "yes" would be misleading.**

| contrast | what it adds coordination to | delta | percentile 95 % | BCa 95 % | p | power | Type-M |
|---|---|---|---|---|---|---|---|
| **E − A** | a bare fingerprint | **+0.060** | **[+0.016, +0.092]** | [+0.020, +0.095] | **0.007** | 0.86 | 1.08 |
| F − B | the generic descriptors | +0.042 | [−0.015, +0.086] | [−0.003, +0.102] | 0.218 | 0.33 | 1.72 |
| **G − D (primary)** | both | **+0.034** | **[−0.003, +0.063]** | [+0.004, +0.075] | **0.092** | 0.46 | 1.47 |

**The marginal value falls as the baseline gets richer**, which is the signature of information the
existing blocks already partly carry. Measured directly: **30 of the 114 coordination columns are
rank-identical to a column already in Gen12's `PHYSCHEM`, `DONORS` or `LIG2D` blocks**, and the
block holds only 100 distinct rank classes.

On the level-reliable cohort the primary is +0.036, percentile [−0.006, +0.067], BCa
[+0.004, +0.078], p = 0.105 — the same borderline picture on better-measured extractants.

**Leave-one-chemotype-out.** The primary delta stays positive with any single chemotype removed,
+0.011 to +0.038, and no chemotype flips its sign. The largest influence is the dominant
diglycolamide chemotype `sc009` (46 extractants): removing it *shrinks* the effect to +0.011, so
the gain is **concentrated in the diglycolamides rather than diluted by them**.

**Two robustness axes, neither confirmatory.**

| axis | primary contrast | percentile 95 % | p | verdict |
|---|---|---|---|---|
| `LVL_MEAN`, ExtraTrees (pre-registered) | +0.034 | [−0.003, +0.063] | 0.092 | borderline |
| `LVL_COND_RESIDUAL`, ExtraTrees | +0.007 | [−0.004, +0.022] | 0.249 | **below the 0.01 floor** |
| CatBoost, `LVL_MEAN` | +0.040 | [−0.019, +0.087] | 0.335 | same sign, 5/5 seeds, unresolved |
| XGBoost, `LVL_MEAN` | −0.010 | [−0.030, +0.005] | 0.198 | **sign reverses**, 2/5 seeds |

The secondary definition subtracts a cross-fitted condition-only model before averaging. That model
explains 8.9 % of pooled row variance but its per-extractant mean carries 43.9 % of the
between-extractant variance, so subtracting it removes much of what the level models compete over,
and every arm converges to 0.96–1.00. It also makes the *evaluation target itself* fold-local:
under that definition every one of the 183 extractants has a different level in different split
seeds, median spread 0.40 decades against a between-extractant sd of 1.32. **The secondary-definition
null is therefore partly a noisier yardstick, not only a different one**, and `PRE_REGISTRATION.md`
§3's statement that the truth does not vary by seed is true of `LVL_MEAN` and false of the
secondary.

### Post-hoc: four descriptor defects were suppressing the effect

An adversarial review of the descriptor block, run **after** the locked evaluation, found four
rule-level defects (§13.14). Rebuilding the block with them corrected and re-running the same arms
on the same frozen folds:

| contrast | frozen v1.1.0 | | post-hoc corrected | |
|---|---|---|---|---|
| | delta | p | delta | p |
| **G − D (primary)** | +0.034 | 0.092 | **+0.046** | **0.020** |
| E − A | +0.060 | 0.007 | +0.069 | 0.035 |
| C − D | +0.100 | 0.464 | +0.088 | 0.587 |

Under the corrected block the primary endpoint's **percentile interval also excludes zero**,
[+0.003, +0.082], where the frozen one did not. `L7_ALL` improves from 1.028 to 1.017 and
`L3_ECFP_GENERIC` is unchanged at 1.062 by construction, since only coordination columns move.

**This is a sensitivity and not a result.** It was computed after the locked evaluation, the
corrections were chosen with knowledge of which columns were wrong, and a corrected block would
need its own pre-registered evaluation before the number could be claimed. What it does establish
is that **the borderline character of the frozen primary endpoint is not purely a power problem**:
part of it was 52 of 114 descriptor columns carrying values their own specification did not
describe. The frozen v1.1.0 matrix still reproduces bit-identically at hash `da609be8fcd5f260` and
remains the basis of every headline number in this report.

### Where the gain comes from — family-stratified

Level MAE by scaffold family, full cohort:

| family | n | `L1_ECFP` | `L3_ECFP_GENERIC` | `L7_ALL` | `L4_COORD` |
|---|---|---|---|---|---|
| **diglycolamide** | 74 | 1.481 | 1.463 | 1.386 | **1.114** |
| N-heterocyclic polydentate | 41 | 0.609 | **0.593** | 0.599 | 0.820 |
| other amide | 28 | **0.752** | 0.776 | 0.789 | 0.816 |
| malonamide | 11 | 0.911 | 0.925 | 0.840 | **0.697** |
| phosphoryl | 11 | **1.013** | 1.026 | 1.072 | 1.083 |
| sulfur donor | 8 | 1.322 | 1.253 | 1.211 | **1.137** |

Contrasts computed inside a family:

| family | extractants | chemotypes | contrast | delta | percentile 95 % | p |
|---|---|---|---|---|---|---|
| **diglycolamide** | 74 | 22 | **C − A** (coordination replaces the fingerprint) | **+0.368** | **[+0.072, +0.454]** | **0.012** |
| diglycolamide | 74 | 22 | G − D | +0.076 | [−0.005, +0.101] | 0.080 |
| diglycolamide | 74 | 22 | C − D | +0.349 | [−0.016, +0.464] | 0.070 |
| **N-heterocyclic** | 41 | 27 | **C − D** | **−0.228** | **[−0.419, −0.056]** | **0.001** |
| N-heterocyclic | 41 | 27 | C − A | −0.211 | [−0.398, −0.032] | 0.014 |
| other amide | 28 | 19 | G − D | −0.013 | [−0.049, +0.017] | 0.422 |

**This is the clearest chemistry in the generation.** Coordination topology is a good representation
of amide/ether donor chemistry, where extraction strength tracks how many chelating units a
molecule carries, and a **significantly worse** representation of N-donor heterocycles than a
fingerprint is. The aggregate +0.034 is the average of a real gain on 40 % of the cohort and a real
loss on 22 % of it.

### Grouped permutation importance

`L7_ALL`, level task, permuting each family among the held-out extractants of each fold,
10 repeats, 25 folds. **Interpretation only: no feature set was selected from this table.** The
metric here weights every fold equally rather than every extractant, so its unpermuted base is
0.932 and not the 1.028 of §3; only the ordering and the relative sizes are read from it.

| group | columns | increase in level MAE when permuted | folds positive |
|---|---|---|---|
| ECFP | 2,048 | **0.256** | 25/25 |
| generic RDKit `LIG2D` | 206 | **0.239** | 25/25 |
| **coordination donor census** | 27 | **0.037** | 24/25 |
| coordination motif counts | 23 | 0.021 | 21/25 |
| coordination donor topology | 22 | 0.019 | 23/25 |
| coordination architecture | 25 | 0.017 | 20/25 |
| Gen12 donor census | 15 | 0.012 | 24/25 |
| **coordination binding-arm multiplicity** | 17 | **0.010** | 21/25 |
| generic physico-chemical | 10 | 0.007 | 22/25 |

**The generation's central construct is its weakest family.** The coordination block's useful part
is a better donor census; arm multiplicity contributes less than Gen12's own donor census did. This
agrees with the training-only exploratory analysis, where pocket count correlated with the level at
Spearman 0.215 and only 0.055 after controlling for molecular size, while the donor and motif
counts reached 0.53 to 0.65. **Read together, the two say the block works as chemistry
identification and not as topology.**

Two caveats on this table, both from the adversarial review: 30 coordination columns have a
rank-identical twin in a generic group, so permuting one group leaves its twin intact and both are
understated; and the groups are not independent.

## 5. Does the improvement survive chemically distant hold-out?

**No. Only the near band clears zero, and only on one contrast.**

| band | extractants | chemotypes | `L3` | `L7` | G − D | percentile | p | E − A | p |
|---|---|---|---|---|---|---|---|---|---|
| far (≤ 0.40) | 27 | 25 | 1.204 | 1.186 | +0.033 | [−0.022, +0.089] | 0.240 | +0.037 | 0.522 |
| mid (0.40–0.60) | 103 | 59 | 1.355 | 1.306 | +0.037 | [−0.024, +0.076] | 0.334 | +0.031 | 0.499 |
| **near (> 0.60)** | 105 | 58 | 0.818 | 0.791 | +0.023 | [−0.006, +0.043] | 0.113 | **+0.052** | **0.023** |

The point estimate is positive and of similar size in all three bands, which does differ from
Gen12, whose molecular advantage *reversed* on far chemistry. But **H5 is not met**: no band
resolves the primary contrast, and the only band-level result that survives both interval methods
is the fingerprint-versus-fingerprint-plus-coordination contrast in the near band.

**The bands are not a partition and the band contrasts share units.** `max_train_tanimoto` is a
property of a fold's training set, so an extractant can be far under one split seed and near under
another: **52 of 183 extractants occupy more than one band**, and the per-band unique counts sum to
235, not 183. Assigning each extractant its modal band instead gives 15 far, 63 mid and 105 near.
This is inherited from Gen12's band definition and is published here rather than left implicit.

In continuous similarity the frontier is a step, not a gradient. Level MAE for `L7_ALL` by
train-similarity sextile: 1.44, 1.43, 0.88, 0.72, 0.86, 0.84. Everything below about 0.53 is equally
hard. Spearman between similarity and per-extractant level error is −0.25 for `L7_ALL`, −0.24 for
`L3` and −0.13 for the nearest-neighbour lookup.

## 6. Does improved level prediction improve full zero-shot `log_D`?

**No, not measurably.** Secondary endpoint, extractant-macro `log_D` MAE, identical query rows:

| arm | macro MAE | chemotype-macro | pooled R² | level | shape |
|---|---|---|---|---|---|
| `GEN12_ABL_D` (Gen12's best observed test arm) | **1.048** | 0.853 | 0.284 | 0.917 | 0.578 |
| `DIRECT_ALL` (Gen12's arm + the coordination block) | 1.049 | 0.857 | 0.294 | 0.918 | 0.575 |
| `DEC_L7_S0` (coordination level + conditions-only shape) | 1.060 | 0.983 | 0.323 | 0.946 | 0.569 |
| `DEC_L4_S1` (coordination-only level) | 1.067 | 1.018 | **0.421** | 0.955 | 0.556 |
| `GEN12_ABL_C` (Gen12's pre-registered selection) | 1.092 | 0.887 | 0.240 | 0.977 | 0.569 |
| `DEC_L7_S1` | 1.109 | 0.971 | 0.324 | 1.000 | 0.551 |
| `GEN12_ABL_A` (no molecular information) | 1.134 | 1.025 | 0.162 | 1.029 | 0.612 |
| `DEC_L3_S1` | 1.137 | 0.972 | 0.262 | 1.032 | **0.550** |

- **secondary endpoint, `DEC_L7_S1` − `DEC_L3_S1`: +0.028, percentile [−0.016, +0.061], p = 0.319.**
  The level improvement of +0.034 does appear, at +0.032 on the level component, and it does not
  reach significance on the full query. **H3 fails.**
- **`DIRECT_ALL` − `GEN12_ABL_D`: −0.001, p = 0.857.** Adding the coordination block to Gen12's own
  direct model changes nothing at all. The 0.0159 minimum detectable effect there is the tightest
  in the study, so this one *is* a well-powered null.

## 7. Does explicit level + shape decomposition outperform direct regression?

**No. It is worse, and significantly so in the matched case.**

| contrast | delta | percentile 95 % | p |
|---|---|---|---|
| `DEC_L3_S1` versus `GEN12_ABL_D` | **−0.089** | **[−0.172, −0.030]** | **0.002** |
| `DEC_L7_S1` versus `GEN12_ABL_D` | −0.061 | [−0.172, +0.018] | 0.148 |
| `DEC_L7_S1` versus `DIRECT_ALL` | −0.060 | [−0.168, +0.019] | 0.157 |

**The decomposition loses the level and wins the shape**, exactly as the repository measured two
generations earlier with a different two-stage model. `DEC_L7_S1`'s level component is 1.000 against
the direct model's 0.917 while its shape component is 0.551 against 0.578, and against the
no-chemistry control the decomposed shape is significantly better (+0.061, BCa [+0.023, +0.110]).
The structure-only level model is simply a weaker level estimator than a joint model that can also
use the conditions — which, given §4's finding that conditions partly identify the extractant, is
what should have been expected.

**The shape model gains nothing from molecular structure.** On identical folds:

| shape contract | macro MAE | shape component |
|---|---|---|
| `S0` conditions only | **1.060** | 0.569 |
| `S2` conditions + coordination | 1.101 | 0.552 |
| `S3` conditions + fingerprint + coordination | 1.108 | 0.556 |
| `S1` conditions + fingerprint | 1.109 | 0.551 |

`S3` versus `S1` is +0.001 (p = 0.86) and `S2` versus `S1` is +0.008 (p = 0.84). The conditions-only
shape is the best of the four on the full metric. **Whatever the within-extractant response to
acidity and ligand concentration is, it is close to shared across this chemistry**, and adding
molecular features to the shape stage only adds variance.

## 8. Are bridged, tripodal and multi-arm ligand errors reduced?

**No. H4 fails.**

The failure family is reproduced first, from Gen12's frozen predictions and a structural rule fixed
before any Gen12.2 model ran. `MULTI_ARM` is `local_donor_cluster_count >= 2`: 43 extractants in 21
chemotypes.

| on Gen12's champion `ABL_D` | MULTI_ARM (43) | single-pocket (140) |
|---|---|---|
| mean signed level error | **−1.133** | −0.083 |
| median signed level error | −1.449 | +0.022 |
| mean row MAE | 1.841 | 0.804 |
| fraction under-predicted | 67 % | 49 % |

12 of Gen12's 15 worst extractants are `MULTI_ARM` and 14 of 15 are under-predicted. Level bias
against pocket count: **−0.08 at one pocket, −1.03 at two, −1.95 at three.** The prospectively
defined structural rule recovers Gen12's hand-read failure family without being told about it.

Mean signed level error on the Gen12.2 arms (negative is under-prediction):

| arm | MULTI_ARM | single-pocket |
|---|---|---|
| `L1_ECFP` | −1.190 | −0.267 |
| `L3_ECFP_GENERIC` | −1.164 | −0.242 |
| `L5_ECFP_COORD` | −1.190 | −0.200 |
| `L7_ALL` | −1.125 | −0.190 |
| **`L4_COORD`** | **−0.859** | −0.076 |
| `L0_NN_TANIMOTO` | −0.744 | −0.085 |

On the pre-registered subgroup, `L7` against `L3` changes the signed error by 0.040 with p = 0.71,
and `L4` against `L3` by 0.306 with p = 0.61. **On the complement — the 140 single-pocket
extractants — the same contrasts are significant** (0.052, p < 0.001 and 0.166, p = 0.013), which
is the endpoint the pre-registration did not name and is reported as such.

**The mechanism is visible even where the significance is not.** Every arm containing the
fingerprint sits at −1.12 to −1.19; the two arms without it sit at −0.86 and −0.74. **Adding ECFP
to the coordination block restores the under-prediction the block had removed.** Seeing
diglycolamide-like bits, the fingerprint predicts diglycolamide-like behaviour, and it wins the
argument whenever it is present. That is a representation-*mixing* failure and no further
descriptor block will fix it.

**Two limits on this subgroup, both material.** Its Kish effective sample size is **4.0 chemotypes**,
stated in the pre-registration before the result was read; nothing of this size can be resolved
there. And the adversarial review found that the rule is permissive: **18 of the 43 have at most one
pocket holding two or more donors**, so a pendant solubilising ether counts as an arm. A stricter
post-hoc rule — two or more pockets each with two or more donors — gives 25 extractants in **5**
chemotypes, which is less resolvable still. The pre-registered rule is what the result is reported
on; the alternative is recorded, not substituted.

## 9. Does one experimental measurement still produce the same large gain?

**Yes, and it erases the representation difference entirely.** Common cohort of 42 extractants in
21 chemotypes, byte-identical support and query draws across every arm, shrunk offset adapter.

| arm | zero-shot | one-shot | gain | five-shot | level k=0 → k=1 | shape k=0 → k=1 |
|---|---|---|---|---|---|---|
| `DIRECT_ALL` | 1.220 | 1.014 | 0.206 | 0.882 | 0.874 → 0.662 | 0.781 → 0.779 |
| `GEN12_ABL_D` | 1.232 | 1.018 | 0.215 | 0.883 | 0.884 → 0.668 | 0.782 → 0.780 |
| **`DEC_L7_S1`** | 1.219 | **0.965** | 0.254 | 0.835 | 0.919 → 0.634 | 0.739 → 0.736 |
| `GEN12_ABL_C` | 1.271 | 1.007 | 0.265 | 0.868 | 0.990 → 0.674 | 0.767 → 0.765 |
| `DEC_L3_S1` | 1.243 | 0.966 | 0.278 | 0.831 | 0.954 → 0.639 | 0.735 → 0.732 |

Every gain is significant. **All of it is level and none of it is shape**, exactly as Gen12 found:
the level component falls by 0.21 to 0.32 while the shape component moves by 0.003.

**H6 is directionally supported in the matched comparison and fails against Gen12.** Within the
decomposed family the coordination-aware level does leave less for one measurement to correct
(0.254 against 0.278), which is the mechanism H6 predicts. Against Gen12's champion the gain is
*larger* (0.254 against 0.215), because the decomposed arm starts from a worse zero-shot number.

**Two properties of the adapter, stated rather than assumed.** The shrinkage constant is Gen12's
`shrinkage_from_training`, whose name overstates it: it is estimated from the out-of-fold residuals
of all 183 held-out extractants, including the one being adapted, so each extractant contributes
about 1/915 of its own shrinkage weight. Measured by leave-one-extractant-out, the ratio moves by
at most 0.015 and the k = 1 weight by at most 0.007 — the same constant is used for every arm, so
the comparison is unaffected. And that constant is taken from one arm (Gen12's champion, 0.470)
rather than refitted per arm (0.426 for `ABL_C`, 0.505 for `ABL_A`), deliberately, so that arms
differ only in their predictions and not in their adapter.

**The result that matters is neither.** Zero-shot, `DEC_L7_S1` beats `DEC_L3_S1` by 0.024 on this
cohort. After one measurement the gap is **0.0007**. One experimental point does what the entire
coordination block does, several times over, and makes the choice of representation irrelevant.

## 10. What is statistically supported, and what is merely suggestive

Supported here means: BCa **and** percentile intervals both exclude zero, the two-sided bootstrap
p is below 0.05, the effect exceeds the 0.01 reproducibility floor, and no single chemotype flips
the sign.

**Supported**

- the level is the bottleneck: fixing it recovers 0.738 against 0.131 for the shape, and 56 % of
  the squared error is level;
- coordination descriptors added to a bare fingerprint improve zero-shot level MAE by +0.060
  (p = 0.007), and by +0.052 in the near band (p = 0.023);
- inside the diglycolamide family, coordination descriptors *replacing* the fingerprint are worth
  +0.368 (p = 0.012);
- inside the N-heterocyclic polydentate family, coordination descriptors alone are significantly
  **worse** than the generic representation, −0.228 (p = 0.001);
- on the level-reliable cohort both `L7_ALL` and `L4_COORD` beat the nearest-chemistry level lookup
  (p = 0.007 and p = 0.002);
- every structural representation beats predicting the training mean level (p < 0.001);
- explicit level-plus-shape decomposition with a generic level model is **worse** than direct
  regression, −0.089 (p = 0.002);
- one measurement improves every arm, and all of the improvement is level;
- the coordination block adds nothing to Gen12's direct model, −0.001 (p = 0.86), on the
  best-powered contrast in the study.

**Suggestive but not established** — point estimate in the expected direction, at least one
interval or the p-value against it:

- the pre-registered primary endpoint, +0.034 (BCa excludes zero, percentile does not, p = 0.092);
- the same contrast on the level-reliable cohort, +0.036 (p = 0.105);
- coordination descriptors alone beat the nearest-chemistry lookup on the full cohort, +0.138
  (p = 0.053);
- coordination descriptors reduce the multi-arm under-prediction when used alone, by 0.31;
- the level gain carries into full `log_D`, +0.028 (p = 0.319).

**Not supported**: H2 in any robust sense, H3, H4, H5. H6 is supported in the matched comparison
only.

**No equivalence claim is made anywhere.** The pre-registered margin is 0.02 and nothing meets it.

## 11. What remains underpowered

| contrast | independent units | MDE at 80 % | observed | realised power |
|---|---|---|---|---|
| primary, full cohort | 97 chemotypes, n_eff 13.7 | 0.052 | +0.034 | **0.46** |
| primary, far band | 25 chemotypes | 0.086 | +0.033 | 0.16 |
| secondary endpoint, full `log_D` | 97 chemotypes | 0.059 | +0.028 | 0.33 |
| coordination-only versus generic | 97 chemotypes | 0.268 | +0.100 | 0.18 |
| **multi-arm signed bias** | **21 chemotypes, n_eff 4.0** | **0.32** | 0.04 / 0.31 | < 0.1 |

At 46 % power a significant estimate is inflated by about **1.5x** (Gelman–Carlin Type-M), so the
primary effect's likely true size is nearer 0.023 than 0.034.

**Multiplicity.** 348 bootstrap rows are emitted across `bootstrap/`, 76 of them flagged as excluding zero, and **18 of those 76 have a percentile interval that includes zero**. One primary
endpoint was pre-registered and is protected by that; **every other contrast in this directory is
unadjusted and exploratory**, and the count is published rather than left implicit.

**The multi-arm subgroup is the endpoint the generation most wanted to resolve and the one it
cannot.** Resolving a 0.3-decade change in its signed bias needs roughly four times the independent
multi-armed chemistry — new bridged and tripodal ligands, not more measurements of the ones already
present.

## 12. Chemical failure modes that remain

1. **The fingerprint overrides the coordination block on multi-armed ligands.** Every arm containing
   ECFP predicts a multi-arm level about 1.15 decades too low; the two arms without it sit at −0.86
   and −0.74. This is a representation-mixing failure: it needs a model that cannot ignore the
   coordination features, or a molecular representation that does not encode the local fragment so
   dominantly — not another descriptor block.
2. **Coordination topology is the wrong representation for N-donor heterocycles**, where it is
   significantly worse than a fingerprint. Counting pyridine-type nitrogens and their graph
   distances does not capture what makes a BTBP or a phenanthroline diamide extract, and the
   donor-cluster rule gives most of them one large undifferentiated pocket.
3. **Binding-arm multiplicity, the construct this generation was built around, is its least useful
   family** — 0.010 in grouped importance, below Gen12's own donor census, and a partial correlation
   with the level of 0.055 once molecular size is controlled.
4. **The phosphorothioate TWE-24 is again an extreme outlier**, under-predicted by 4.06 decades by
   Gen12 and reached by no representation here. The repository's review queue disputes its identity;
   Gen12.2 records that the same structure is an extreme outlier under a fourth representation and
   does not resolve it.
5. **Single-cell extractants dominate the worst-error list.** Eight of Gen12's fifteen worst have one
   measured condition, so their "true level" is one measurement. The level-reliable cohort bounds
   that: the primary effect is the same size there, which is evidence it is not an artefact of thin
   extractants.
6. **The near band is not near.** Design B caps train similarity at 0.698, so nothing here speaks to
   the case a practitioner most often has — a close homologue of something already measured.

## 13. Defects found during this work

Four were found before any locked evaluation and corrected; five were found by an adversarial
review afterwards and are reported rather than absorbed.

**Found before the locked evaluation**

1. **The first chelate-link rule was chemically wrong.** At three bonds every malonamide and every
   P(=S)–CH₂–P(=S) dithiophosphinate was split into two pockets, though both are single-pocket
   bidentates closing a six-membered ring. Found by inventorying the descriptor over all 183
   structures — **no target value and no model output was consulted** — and corrected to four bonds
   in spec 1.1.0. Reference-structure misassignments fell from 4 to 1.
2. **The pre-registered level-definition override was mis-specified and fired**, selecting the
   condition-adjusted level. Three training-only measurements showed the shift it keys on is the
   condition model absorbing genuine level, not sampling distortion. Amended, dated, with the
   original rule and what it would have chosen preserved.
3. **Four columns of Gen12's donor block are not structural.** `DENTATE`, `coreCN`, `n_ligs` and
   `n_fill` are built per experiment by the Architector recipe and vary between rows of the same
   molecule for 10 of 183 extractants. Collapsed to the per-extractant mode for the structure-only
   level model, deterministically, with the count recorded. **A consequence for Gen12: its
   "molecule only" ablation was not purely molecular.**
4. **A shape contract using the coordination block crashed** because the block was joined to the
   per-extractant frame and not the per-row frame. Fixed with an assertion that the merge preserves
   row order; the arms that had already run use contracts without the block, and the identity was
   *checked* rather than assumed — re-running `DEC_L7_S1` under the fixed code path reproduces the
   stored predictions to 2.2e-15.

**Found by adversarial review after the locked evaluation**

5. **The primary endpoint's significance rests on the BCa endpoint shift alone.** The percentile
   interval from the identical draws includes zero and p = 0.092. The implementation is correct —
   three independent verifiers confirmed the arithmetic — but reporting only the BCa interval would
   have overstated the result. Both intervals and the p-value now appear in every headline table.
6. **`units_improved` and the per-seed sign were inverted for every signed-error contrast.** Gen12's
   bootstrap fixes "positive delta means the candidate is better", which is right for an error
   magnitude and backwards for a signed error on a negatively-biased population. The frozen function
   is unmodified; the orientation is corrected in Gen12.2's own analysis output, which now also
   prints the direction convention on every row.
7. **The band macro numbers printed beside each band delta used a different weighting from the
   delta.** They are now computed from the same unit table, with an assertion that the delta
   reproduces by subtraction.
8. **The decomposed arms do not carry the ladder's level models verbatim.** Their level stage
   chooses its hyperparameter on a grouped split of the fold's training *extractants*, while the
   ladder splits the fold's training *rows*. Both are leakage-safe and they agree in accuracy —
   level MAE 1.0268 against the ladder's 1.0281 for `L7`, 1.0620 against 1.0624 for `L3` — but the
   per-extractant predictions differ (mean absolute difference 0.167, correlation 0.967). The
   secondary endpoint compares two decomposed arms built the same way, so the contrast is matched;
   the wording "the decomposed arm uses the ladder's level model" would not have been.
9. **Level R², Spearman and dispersion took the truth from one split seed and the prediction from
   five.** Harmless under `LVL_MEAN`, where the truth is seed-invariant, and wrong under the
   secondary definition, where it is not. Corrected to average both.
10. **The multiplicity count counted its own output**, so it grew on every re-run. Fixed to exclude
    the two files the script itself writes.
11. **Three invariant tests were weaker than their names.** The cross-fit test would have passed
    with every chemotype leaking across the inner folds; the level-definition test asserted only
    that a digest is 64 characters long; and invariant 14 had no test at all. All three now assert
    the property itself.
12. **The bands are not a partition**: 52 of 183 extractants appear in more than one band, so the
   per-band counts sum to 235 and the band contrasts share units. Published in
   `metrics/level/LVL_MEAN/band_overlap.json` with a modal-band partition beside it.
13. **The few-shot shrinkage constant is not estimated the way its docstring says.** Gen12's
    `shrinkage_from_training` pools the out-of-fold residuals of every held-out extractant,
    including the one being adapted, while its docstring says "extractants *other than* the one
    being adapted". Measured here: leaving any single extractant out moves the ratio by at most
    0.015 and the k = 1 shrinkage weight by at most 0.007, and the same constant is applied to
    every arm, so no comparison in §9 is affected. The frozen Gen12 function is not modified.
14. **Four descriptor-definition defects**, all rule-level: aromatic ring oxygens and
   aromatic-carbon carbonyls were invisible to the donor classes (5 structures affected); the lactam
   test did not require the carbonyl carbon in the ring (12 of 13 non-zero values were false
   positives); the pyridine-like ring class counted nitrogen by element while the frozen prose says
   `[nX2]`; and the named-chelating-unit sum double-counted malonamides while omitting the
   P(=S)–CH₂–P(=S) motif. A corrected specification is provided as
   `config/coordination_smarts_posthoc.json` and its effect is measured in
   `analysis/descriptor_defect_sensitivity.md`: correcting them moves the primary endpoint from
   +0.034 (p = 0.092) to **+0.046 (p = 0.020)** and changes 52 of the 114 columns. **The frozen
   v1.1.0 matrix still reproduces bit-identically and remains the basis of every number above.**

## 14. Guardrails — what a reader must not do with these numbers

1. **Do not quote the primary +0.034 without its percentile interval and p-value.** BCa alone
   overstates it.
2. **Do not quote it without its power.** 46 % realised power means a significant estimate is
   inflated by about 1.5x.
3. **Do not quote it without the robustness axes**: +0.007 under the secondary level definition,
   −0.010 under XGBoost.
4. **Do not read the aggregate as chemistry-wide.** +0.076 inside the diglycolamides, −0.007 inside
   the N-heterocyclic polydentates.
5. **Do not quote a level MAE as a `log_D` MAE.** 183 evaluation units against 6,645 rows.
6. **Do not treat any non-significant contrast as equivalence**, least of all the multi-arm
   subgroup, whose effective sample size is four chemotypes.
7. **Do not quote `L4_COORD`'s 0.962 as the model's score.** It is the best observed test arm, the
   *worst* tree arm on inner validation, and its lead over the generic representation carries
   p = 0.46.
8. **Do not read "near" as "a close analogue was in training".** Design B caps train similarity at
   0.698, and the bands are not a partition.
9. **Do not mix the cohorts**: 183 on the full cohort, 71 on the level-reliable one, 42 on the
   few-shot one.
10. **Do not describe any descriptor here as geometry.** They are counts and graph distances.
11. **Do not treat the primary endpoint as an independent confirmation.** The hypothesis it tests
    was generated from Gen12's held-out errors on the same 183 extractants.
12. **An effect below 0.01 is not an effect** — it is within this pipeline's measured cross-machine
    reproducibility.

## 15. Recommendation

**Do not adopt the coordination block as a general improvement.** On Gen12's own direct model it is
worth −0.001 with the tightest interval in the study. On the level task its aggregate gain is
borderline, learner-dependent, definition-dependent, and smaller than the reproducibility of the
conclusion demands.

**Do adopt it, or something like it, for diglycolamide-like chemistry specifically**, where
replacing a fingerprint with 114 interpretable coordination descriptors is worth +0.368 in level
MAE with p = 0.012 — and do not use it on N-donor heterocycles, where the same swap costs 0.211.
A representation that is chosen per chemical family is a different deployment from a single global
model, and this cohort supports the first and not the second.

**Deploy the measure-once protocol, as Gen12 recommended.** One measurement is worth 0.21 to 0.28
macro MAE, and after it the coordination-aware and generic representations are indistinguishable to
four decimal places. **Nothing in this generation changes the practical conclusion that measuring
one point beats every representational improvement available.**

**Re-run the block once its definitions are right, and pre-register that run.** The post-hoc
sensitivity is the single most actionable finding here: four descriptor defects, none of which
required new chemistry to fix, moved the primary endpoint from unresolved to resolved. That is
cheap to redo properly and is a better next experiment than any new architecture.

**What would settle the open question.** The primary contrast needs about four times the independent
chemistry to resolve a 0.034 effect — roughly 100 additional chemotypes, not 100 additional rows.
The multi-arm question needs new bridged and tripodal ligands specifically: 21 chemotypes with an
effective sample size of four cannot answer it however carefully it is analysed. And the finding
that binding-arm multiplicity is the *least* useful family of the block says that if the missing
information is coordination chemistry, **it is not recoverable from a 2D molecular graph** — the
next step is genuinely physical descriptors, external pretraining, or more independent chemistry,
not another handcrafted block.
