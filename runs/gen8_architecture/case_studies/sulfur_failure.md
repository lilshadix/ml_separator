# The sulfur-donor failure, dissected

*gen8 brief, section 27. Model: `REC_ecfp_plus_recovered`, out-of-fold over 5 split seeds x 5
folds, held-out chemotypes. Everything below is reproduced by
`runs/gen8_architecture/case_studies/sulfur_failure.py`; the per-ligand numbers are in
`per_ligand_kshot.csv`, `curve_slopes.csv` and `sulfur_compact.csv` in this directory.*

*Revised after adversarial verification. Every number in the first draft reproduced to the digit
from the frozen artefacts; four claims about those numbers did not survive, and each is marked
where it occurs — the **solvent verdict** in section 7 (reversed from NO to the leading
explanation), the **mechanism** given for the acquisition counter-example in section 5 (it
described the oracle, not the deployable rule, and the rule's real cost here is 0.03 rather than
nothing), the **contagion figure** in section 7 (3.9 log units is a 1NN surrogate's number, not the
model's), and the **subgroup verdicts** in section 9 (point estimates reported as findings, whose
intervals all span zero).*

*Selection warning, since it governs how everything below should be read: **the subject of this
case study was chosen by its outcome.** TWE-24 is here because it has the largest zero-shot error
in the cohort. Nothing about it is a random draw, no p-value in this document is protected against
that choice, and the subgroup comparisons in section 9 were drawn after seeing which ligands
failed. Section 8 (all eleven S-donors, pre-specified by donor atom) is the part of the document
that is not outcome-selected.*

---

## 0. Correction to the brief's premise, stated up front

The brief asks for the archetypal case: *a bis(dithiophosphonate) whose ECFP neighbours are hard
phosphorus extractants but whose donor set is soft sulfur.* Two parts of that premise do not
survive contact with the cohort, and the case study is more interesting because of it.

**There is no bis(dithiophosphonate) — or any other dithio acid — in this cohort.** Of the 152
extractants, 11 carry at least one S donor. Every one of them is *neutral*:
`mech__n_acidic_H == 0` for all 11, and a direct RDKit scan finds no P-SH, no P-S(-), and no S-H
of any kind anywhere in the 152 SMILES. Cyanex 301/302-type dithiophosphinic acids, the reagents
that actually make soft-donor Ln/An separation work, are simply absent. The S chemistry present
is three neutral classes: thiophosphoryl P=S (6 ligands), thioether/thiol C-S-C (4), thiocarbonyl
C=S (2, one of which also has a thioether S).

**The study therefore runs on the worst S-donor extractant that *is* present: `TWE-24`,**
canonical SMILES `CCCCOP(=S)(CP(=S)(OCCCC)OCCCC)OCCCC` — *O,O,O',O'-tetrabutyl
methylenebis(phosphonothioate)*, the P=S analogue of tetrabutyl methylenediphosphonate. Two
thiophosphoryl sulfurs bridged by a single CH2, four butoxy oxygens, neutral, no exchangeable
proton, `mech__softness_mean` 0.400. It is not merely the worst S-donor: **it is the worst
zero-shot ligand in the entire cohort**, macro MAE 4.098 against a cohort macro of 0.995 —
rank 1 of 152, and 0.60 log units worse than rank 2.

**And ECFP does not confuse it with hard phosphorus chemistry.** Four of its five nearest ECFP
neighbours already carry P=S (section 1). The fingerprint gets the chemistry right. The failure
is somewhere else entirely, and finding out where is the point of this document.

---

## 1. Nearest neighbours under ECFP Tanimoto

Tanimoto over the frozen 2048-bit `ecfp_*` block of `cohort.parquet`.

| rank | neighbour | Tanimoto | donor classes | softness | its own zero-shot MAE |
|---|---|---|---|---|---|
| 1 | TWE-29 `CCCCOP(=S)(COCP(=S)(OCCCC)OCCCC)OCCCC` | 0.667 | 5 ether O, 2 thiophosphoryl S | 0.357 | 2.034 |
| 2 | TWE-23 `CCCCP(=S)(CCCC)CP(=S)(CCCC)CCCC` | 0.400 | 2 thiophosphoryl S | 1.000 | 1.318 |
| 3 | dihexyl hydrogen phosphate `CCCCCCOP(=O)(O)OCCCCCC` | 0.379 | 2 ether O, 1 hydroxyl O, 1 phosphoryl O | 0.100 | 0.843 |
| 4 | TWE-27 `CCCCP(=S)(CCCC)COCP(=S)(CCCC)CCCC` | 0.357 | 1 ether O, 2 thiophosphoryl S | 0.700 | 0.799 |
| 5 | TWE-28 `CCOP(=S)(COCP(=S)(OCC)OCC)OCC` | 0.333 | 5 ether O, 2 thiophosphoryl S | 0.357 | 0.695 |

TWE-24 and TWE-29 are **mutual** nearest neighbours at 0.667 and differ only in the bridge that
separates the two P=S groups: `-CH2-` versus `-CH2-O-CH2-`. Mean absolute softness gap over the top 5 is 0.257, and that
is entirely carried by two entries — the all-S TWE-23 (1.000) and the one genuinely hard neighbour,
dihexyl hydrogen phosphate (0.100). The nearest neighbour's softness gap is 0.043.

## 2. Nearest neighbours under the mechanistic distance

Euclidean distance on the 36 z-scored `MECHANISM_COLUMNS` of `gen8/mechanism.py`, standardised
exactly as `scripts/gen8_mechanism_similarity.py` does it.

| rank | neighbour | mech distance | its Tanimoto to TWE-24 | donor classes | softness |
|---|---|---|---|---|---|
| 1 | TWE-29 | 1.667 | 0.667 | 5 ether O, 2 thiophosphoryl S | 0.357 |
| 2 | TWE-28 | 1.925 | 0.333 | 5 ether O, 2 thiophosphoryl S | 0.357 |
| 3 | TWE-27 | 6.278 | 0.357 | 1 ether O, 2 thiophosphoryl S | 0.700 |
| 4 | TWE-30 `S=P(COCP(=S)(Ph)Ph)(Ph)Ph` | 6.873 | 0.147 | 1 ether O, 2 thiophosphoryl S | 0.700 |
| 5 | TWE-33 `CCCCCCCCN(CCCCCCCC)C(=O)COCP(=O)(OCCCC)OCCCC` | 8.334 | 0.293 | 1 amide O, 3 ether O, 1 phosphoryl O | 0.130 |

## 3. Donor classes compared

| | TWE-24 | ECFP top-5 | mechanistic top-5 |
|---|---|---|---|
| carries P=S | yes (2) | 4 of 5 | 4 of 5 |
| carries any S donor | yes (2) | 4 of 5 | 4 of 5 |
| mean abs. softness gap to TWE-24 | — | 0.257 | 0.191 |
| the one hard-donor intruder | — | dihexyl hydrogen phosphate (P=O, softness 0.100, rank 3) | TWE-33 (P=O amide-phosphonate, softness 0.130, rank 5) |

**This is the load-bearing negative result of the case study.** The stated hypothesis — that a
generic fingerprint reads a soft P=S extractant as a hard P=O one, and so the model inherits the
wrong extraction mechanism — is *false here*. Both similarity measures put the four other
thiophosphoryl compounds at the top. The mechanistic distance is marginally tidier (it demotes the
hard phosphate out of the top 5, and its softness gap is 0.19 versus 0.26), which is consistent
with the cohort-level result that mechanism-aware similarity *does* separate donor substitutions —
but it buys nothing here, because ECFP was not making that mistake for this molecule. It also
agrees with the established cohort verdict that 1NN Tanimoto beats 1NN mechanistic distance on the
level by 0.085.

## 4. Predicted versus observed curves

TWE-24 contributes one series (`484290a993a179f8`), one curve, on the acid axis: 6 points, Eu(III)
only, 0.09 M extractant, HNO3 0.0113 - 4.0133 M, 22 C, 60 min, and a diluent the model sees as
`other` — upstream it is TPH with 12.5 vol% 1-octanol, which section 7 argues is the leading
explanation for the level error.
Predictions below are the mean over the 5 split seeds (seed-to-seed sd 0.18 - 0.25).

| [HNO3] / M | log10[HNO3] | observed log D | predicted log D | residual |
|---|---|---|---|---|
| 0.0113 | -1.947 | +2.706 | -1.371 | +4.077 |
| 0.1097 | -0.960 | +2.104 | -1.471 | +3.575 |
| 1.0090 | +0.004 | +2.462 | -2.274 | +4.736 |
| 1.9855 | +0.298 | +2.086 | -2.624 | +4.710 |
| 3.0417 | +0.483 | +0.748 | -2.768 | +3.517 |
| 4.0133 | +0.604 | +1.233 | -2.737 | +3.970 |

**Slope, the only axis this ligand varies:**

| axis | true d(log D)/d(log10[HNO3]) | predicted | error |
|---|---|---|---|
| acid | **-0.535** | **-0.608** | 0.074 |

For scale: over all 496 acid curves in the corpus the median true slope is +1.573 and the median
predicted slope is +0.404 — the model's well-documented flattening pathology. On TWE-24 there is
no flattening at all. The predicted curve is *parallel* to the measured one to within 0.07 log
units per decade of acid, it has the correct (negative) sign, and the within-ligand Spearman
between prediction and truth is 0.943. The residual scatter about its own mean is 0.499, *below*
the cohort mean of 0.596 (0.600 over the 143-ligand k-shot population).

The model reproduces the shape of this curve better than it reproduces an average curve. It places
it four decades too low.

**The contrast with its own nearest neighbour is instructive.** TWE-29, one bridging oxygen away,
same acid grid, same campaign:

| [HNO3] / M | observed log D | predicted log D | residual |
|---|---|---|---|
| 0.0113 | -0.721 | -0.352 | -0.369 |
| 0.1097 | -1.201 | -0.392 | -0.808 |
| 1.0090 | -2.222 | -0.469 | -1.753 |
| 1.9855 | -3.854 | -0.526 | -3.328 |
| 3.0417 | -3.620 | -0.642 | -2.978 |
| 4.0133 | -3.538 | -0.637 | -2.900 |

TWE-29 is the *opposite* failure: true slope -1.260, predicted -0.111 — flattened by a factor of
11, the textbook shape failure, and its level is 2.0 too high. The two mutual nearest neighbours
fail in mirror-image ways, and section 7 shows why.

## 5. One-shot offset calibration, every candidate row

Measure one row, add its residual to every other prediction, score the remaining 5 rows. All six
candidates enumerated; MAE averaged over the 5 seeds.

| measured row ([HNO3] / M) | its acid tercile | 1-shot MAE on the other 5 rows |
|---|---|---|
| 0.0113 | low | **0.508** (best) |
| 4.0133 | high | 0.516 |
| 0.1097 | low | 0.683 |
| 3.0417 | high | 0.722 |
| 1.9855 | mid | 0.751 |
| 1.0090 | mid | **0.771** (worst) |

| | TWE-24 | cohort (143 ligands, >=4 rows) |
|---|---|---|
| zero-shot | 4.098 | 0.995 |
| random 1-shot (mean over candidates) | 0.658 | 0.709 |
| best 1-shot (oracle choice) | 0.508 | 0.522 |
| worst 1-shot | 0.835 | 1.325 |
| oracle level (best constant, all 6 rows) | 0.423 | 0.480 |

*Convention, because the two tables above disagree and the difference is not noise:* the
per-candidate table averages the five seeds first and then takes the extreme over candidates
(worst **0.771**); this table takes the extreme over candidates within each seed and then averages
(worst **0.835**), which is how the cohort column is computed. The best-candidate cell is 0.508
either way, because the lowest-acid row is the best candidate in all five seeds; the worst
candidate is not the same row in every seed, so the two orderings separate.

One measurement takes the corpus's worst ligand from 4.098 to 0.658 — a **84% reduction**, and it
lands *below* the cohort-average one-shot number. After one measurement TWE-24 is a
better-than-typical ligand.

**The cohort-wide acquisition rule does not transfer to this ligand.** The established gen8 result
is that measuring at mid-range acidity beats the low-acid extreme by 0.244 (CI [0.175, 0.322],
5/5 seeds). Here the ranking is inverted: the two *extremes* are the two best candidates (0.508 and
0.516) and the two mid-acid rows are the two worst (0.751, 0.771). The inversion is stable — the
two extremes beat the two mid-acid rows in 5/5 split seeds, by 0.09 to 0.31 — so it is a real
counter-example to the rule, not sampling noise.

**What the deployable rule actually does here, measured rather than assumed.** `CENTRAL` selects
the candidate nearest the *mean of the standardised condition axes*; `MEDOID` and `MID_ACID`
select on the same axes. None of them reads a residual — they cannot, the residual is the
held-out target. On this ligand's full six-row pool all three land on the 1.009 M row, which is
the single *worst* candidate (0.771). In the acquisition harness's own paired evaluation the cost
is smaller because the pool is subsampled: k=1 with offset adaptation gives `CENTRAL` = `MEDOID` =
`MID_ACID` **0.682** against `RANDOM` **0.656** and `FARTHEST_FROM_EXISTING` **0.571** (oracle
0.467). So following the cohort rule on this ligand costs about **0.03**, not nothing — small
against a 4.098 starting error, but the sign is against the rule.

The *oracle* choice is the one that keys off the residual, and that is why the extremes win: this
ligand's residuals are non-monotone in acid (+4.08, +3.58, +4.74, +4.71, +3.52, +3.97, median
+4.02), so the two rows nearest the median residual happen to be the lowest-acid and highest-acid
points. That mechanism explains the oracle ranking; it is not available at deployment time.

## 6. What is left after 1 and 2 measurements

| | TWE-24 | cohort (143) |
|---|---|---|
| zero-shot | 4.098 | 0.995 |
| k=1 random | 0.658 | 0.709 |
| k=1 best / worst | 0.508 / 0.835 | 0.522 / 1.325 |
| k=2 random | 0.579 | 0.625 |
| k=2 best / worst | 0.344 / 0.950 | 0.451 / 1.255 |

(k=1 is scored on the 5 unmeasured rows, k=2 on the 4 unmeasured rows, the oracle level on all 6;
the evaluation sets differ, which is why best-k=2 (0.344) can sit below the oracle constant
(0.423) without contradiction.)

The second measurement buys 0.079 on top of the first — consistent with the corpus-wide pattern
that the first measurement is worth an order of magnitude more than the second. Paired chemotype
bootstrap of zero-shot minus one-shot: **all 143 ligands +0.286, CI [0.176, 0.415], BCa
[0.187, 0.438], 83/143 units improved, 5/5 seeds**; **the 11 S-donors +0.528, CI [0.107, 1.171],
BCa [0.169, 1.530], 8/11 improved, 5/5 seeds**. The S-donor interval is wide — 11 ligands, one per Tanimoto chemotype — but it excludes zero and its point estimate is nearly twice the cohort's, because the
S-donor population contains the one ligand with a four-decade offset.

## 7. Classification of the failure

| category | verdict | evidence |
|---|---|---|
| **pure offset** | **YES — this is the whole shape of the error; the cause is the row below** | The best constant removes 89.7% of the error (4.098 -> 0.423). One measurement removes 84%. Residual sd 0.499 < cohort 0.596. Spearman(pred, truth) = 0.943. The residual never changes sign across the six rows: +3.52 to +4.74. |
| **mechanism-shape** | **NO for TWE-24; YES for its family** | TWE-24's acid slope is matched to 0.074 (-0.535 true vs -0.608 predicted), against a cohort median of +1.573 true vs +0.404 predicted. But TWE-29 (-1.260 -> -0.111) and TWE-23 (-1.136 -> -0.074) are flattened by 11x and 15x. The shape pathology is real in this chemistry — it just is not what makes TWE-24 the worst ligand. |
| **solvent mismatch** | **YES, and it is the leading candidate — this verdict was reversed on review** | TWE-24 is the only member of its family in diluent `other` (236 rows, 32 ligands), at 0.09 M rather than 0.10 M. Read at the level of the bundle's flattened one-hot that looks harmless: the `other` bucket's marginal bias is only -0.21 (mean truth -0.049, mean prediction -0.263), and the rest of the corpus is biased *further* the same way (-0.40), so nothing in the bucket explains 4.1. But the bucket is the wrong object to test. The primary export records TWE-24's solvent as **`tph 0.875, 1-octanol 0.125`** — TPH with 12.5 vol% 1-octanol — against neat TPH for TWE-23/27/28/29 and neat 1-octanol for TWE-30. See below. |
| **donor-hardness representation** | **NO** | Section 3. 4 of the 5 ECFP neighbours and 4 of the 5 mechanistic neighbours already carry P=S; nearest-neighbour softness gap 0.043. Neither representation mistakes this molecule for a hard P=O extractant. |
| **missing chemistry** | **NO in the "unseen donor class" sense; YES in the "under-determined family" sense** | The six P=S compounds sit in six different Tanimoto chemotypes, so 2-5 of the 5 siblings (mean 3.8) were in the training set in every fold. The model had the chemistry and used it: it predicted TWE-24's level as **-2.21**, against a sibling-family mean level of **-2.39** — right, to 0.18 log units. What it could not know — because the feature set does not carry it — is that TWE-24 was run with a phase modifier and its siblings were not, and that its recorded level is **+1.89**, **4.27 log units above the five-sibling mean and 3.89 above the highest of them**. The chemistry was present; the *conditions* were not. |

### The sixth bucket: a recorded experimental variable the feature set destroys

The five P=S compounds measured on the identical acid grid, in the same campaign, from the same
single upstream source (CORDIS project 211267 / `10.1021/jacs.5c19738`, joined 1:1 on
`safe_exp_id` to the dataset builder's raw SAFE exports), span raw D = 1.0e-4 to 2.2e-1.
TWE-24's six raw D values are **508, 127, 290, 122, 5.6, 17.1**.

| ligand | bridge | true ligand level (mean log D) | model's level | residual |
|---|---|---|---|---|
| **TWE-24** | -CH2- | **+1.890** | -2.208 | **+4.098** |
| TWE-27 | -CH2-O-CH2- | -2.000 | -2.322 | +0.322 |
| TWE-28 | -CH2-O-CH2- | -2.000 | -1.721 | -0.279 |
| TWE-23 | -CH2- | -2.335 | -1.339 | -0.996 |
| TWE-29 | -CH2-O-CH2- | -2.526 | -0.503 | -2.023 |
| TWE-30 | -CH2-O-CH2- | -3.064 | -2.265 | -0.799 |

Note that TWE-23 shares TWE-24's single-CH2 bridge and sits at -2.335, so the bridge length does
not explain the gap.

**The upstream record explains most of it, and the explanation is a solvent.** Joining
`safe_exp_id` back to the raw SAFE exports recovers a `Solvent_Name` column that the modelling
bundle flattens away:

| ligand | recorded solvent | what the model is told |
|---|---|---|
| **TWE-24** | **tph 0.875, 1-octanol 0.125** | `diluent__other`, `diluent_family=aliphatic_hydrocarbon`, **`modifier_class=none`** |
| TWE-23 / 27 / 28 / 29 | TPH | `diluent__tph`, `modifier_class=none` |
| TWE-30 | 1-octanol | `diluent__1_octanol`, `modifier_class=none` |

TWE-24 is run with a **12.5 vol% 1-octanol phase modifier** and its siblings are not. The bundle
has no one-hot for that mixture (it has one for the commoner `tph 0.95, 1-octanol 0.05`), so the
rarer ratio falls into `diluent__other` — and the modifier flag does not merely go missing, it is
set to `modifier_class=none`. The model is told there is no modifier.

Inside this one campaign, ligand level by solvent class (one extractant one vote; neat 1-octanol
as a *diluent* is a different thing from octanol as a *modifier* in a hydrocarbon diluent, and
they are kept apart):

| solvent class | extractants | mean ligand level |
|---|---|---|
| hydrocarbon diluent + octanol modifier (TWE-24's case) | 49 | **+0.92** |
| no octanol | 48 | -2.03 |
| neat 1-octanol | 33 | -2.34 |

The modifier class sits **+2.96 decades** above the unmodified one, bootstrap CI95 over extractants
**[+2.53, +3.38]**. **This is a between-ligand contrast and it is confounded**: no extractant in
this campaign was measured both ways (0 of 116), and a modifier is added precisely to the systems
that extract strongly enough to risk a third phase, so cause and selection are not separable here.
What it does establish is scale — the variable the feature set throws away moves this campaign's
levels by about three decades, which is the same order as TWE-24's error. Against that reference
TWE-24's +1.89 is the **80th percentile of the 49 modifier ligands** — high, but ordinary — while
it is above **all 48** unmodified ones. Its +4.27 gap to its neat-TPH siblings is of the same size
as the modifier class difference.

**This replaces the reading in the first draft of this document, which offered "its D column is
off by roughly three decades" (by analogy with the DMDPhPDA exponent corruption) as a live
possibility and marked the level as an unresolved anomaly.** No corruption need be postulated: the
bundle faithfully carries the upstream values (`D == obsDvaluesValue` to 1e-6 on all six rows), and
the level is consistent with the octanol-modified subpopulation. `ST65.json`, the primary table
cited in the export's comments, is not on this machine, so the underlying measurement is still
unverified against the paper — but it is no longer the leading hypothesis. The failure
classification above does not depend on this: under either reading the diagnosis is "pure offset".

**Does the outlying level propagate?** TWE-24 is TWE-29's nearest ECFP neighbour and
sits in a different fold from TWE-29 under all five split seeds, so it is always in TWE-29's
training set. A 1-nearest-neighbour ligand-level lookup, run leave-one-out over the cohort:

| | 1NN level MAE |
|---|---|
| with TWE-24 in the neighbour pool | 0.8807 |
| with TWE-24 removed | 0.8549 |

Exactly one ligand's nearest neighbour changes, and it is TWE-29: its level error goes from
**4.416 to 0.526**, and the cohort figure moves 0.026.

**Read that as a property of the 1NN surrogate, not of the deployed model.** The probe is a
condition-blind lookup — it copies a neighbour's mean log D wholesale, so it necessarily copies
the neighbour's solvent too, which is precisely the variable that separates these two ligands. The
gradient-boosted model does not behave that way: its own level for TWE-29 is **-0.50** against a
1NN-with-TWE-24 value of **+1.89** and a true level of -2.53, i.e. it is nowhere near copying its
nearest neighbour, and its actual error on TWE-29 is 2.03, not 3.9. No model was retrained with
TWE-24 removed, so **the contamination cost to the deployed model is not measured here** — 3.9 is
the surrogate's number and is an upper bound on nothing in particular. What the probe does show is
the mirror-image failure of section 4 and the reason for it.

---

## 8. The same seven steps for all 11 S-donor extractants

`softness` = `mech__softness_mean`; `nn_ecfp` is the nearest ECFP neighbour with its Tanimoto;
`nn_mech` the nearest mechanistic neighbour; slopes are means over that ligand's usable curves on
the named axis; `level_share` = 1 - oracle_level/zero_shot, the fraction of the error a single
best constant would remove; `k1_*` and `k2_*` are exhaustive offset calibration.

| name | donors | softness | n_rows | nn_ecfp | nn_mech | axis | true_slope | pred_slope | zero_shot | oracle_level | level_share | k1_rand | k1_best | k1_worst | k2_rand |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TWE-24 | 4 ether O, 2 thiophosphoryl S | 0.400 | 6 | TWE-29 (0.67) | TWE-29 | acid | -0.535 | -0.608 | 4.098 | 0.423 | 0.897 | 0.658 | 0.508 | 0.835 | 0.579 |
| TWE-29 | 5 ether O, 2 thiophosphoryl S | 0.357 | 6 | TWE-24 (0.67) | TWE-28 | acid | -1.260 | -0.111 | 2.034 | 1.046 | 0.486 | 1.497 | 1.255 | 1.984 | 1.314 |
| T8-PHEN-TAM | 2 aromatic N, 2 thiocarbonyl S | 0.775 | 12 | tetraoctyl-phenanthroline-2,9-dicarboxamide (0.68) | T8-THP-TAM | acid | -0.341 | 0.107 | 1.513 | 0.534 | 0.647 | 0.912 | 0.582 | 2.982 | 0.826 |
| TWE-23 | 2 thiophosphoryl S | 1.000 | 6 | TWE-27 (0.59) | TWE-27 | acid | -1.136 | -0.074 | 1.318 | 0.858 | 0.349 | 1.457 | 1.029 | 1.973 | 1.294 |
| TWE-27 | 1 ether O, 2 thiophosphoryl S | 0.700 | 6 | TWE-23 (0.59) | TWE-30 | acid | 0.000 | -0.799 | 0.799 | 0.651 | 0.185 | 0.973 | 0.781 | 1.317 | 0.890 |
| TWE-30 | 1 ether O, 2 thiophosphoryl S | 0.700 | 6 | TWE-28 (0.28) | TWE-27 | acid | -0.046 | 0.030 | 0.799 | 0.231 | 0.711 | 0.362 | 0.277 | 0.530 | 0.309 |
| T8-THP-CAM | 2 amide O, 1 thioether S | 0.433 | 4 | T8-THP-TAM (0.65) | cPSNDGA | no usable curve | | | 0.771 | 0.226 | 0.707 | 0.406 | 0.301 | 0.546 | 0.338 |
| cPSNDGA | 2 amide O, 1 ether O, 2 thioether S | 0.460 | 14 | DOpyrDGA (0.38) | T8-THP-CAM | metal_series | 0.141 | 0.117 | 0.739 | 0.119 | 0.839 | 0.168 | 0.128 | 0.285 | 0.146 |
| TWE-28 | 5 ether O, 2 thiophosphoryl S | 0.357 | 6 | TWE-29 (0.58) | TWE-29 | acid | 0.000 | -0.375 | 0.695 | 0.296 | 0.575 | 0.450 | 0.355 | 0.671 | 0.409 |
| Cy5-S-Me4-BTBP | 8 aromatic N, 2 thioether S | 0.660 | 11 | CyMe4-BTTP (0.61) | CyMe4-BTTP | acid | 0.445 | 0.265 | 0.240 | 0.224 | 0.068 | 0.342 | 0.247 | 0.604 | 0.292 |
| T8-THP-TAM | 1 thioether S, 2 thiocarbonyl S | 0.933 | 4 | T8-THP-CAM (0.65) | T8-PHEN-TAM | no usable curve | | | 0.172 | 0.077 | 0.551 | 0.146 | 0.103 | 0.232 | 0.129 |

Group aggregates, all macro (one ligand one vote):

| population | n | zero-shot | oracle level | level share | k=1 random | k=1 best | k=2 random |
|---|---|---|---|---|---|---|---|
| cohort (>=4 rows) | 143 | 0.995 | 0.480 | 0.518 | 0.709 | 0.522 | 0.625 |
| all S-donors | 11 | 1.198 | 0.426 | 0.644 | 0.670 | 0.506 | 0.593 |
| S-donors excluding TWE-24 | 10 | **0.908** | 0.426 | 0.531 | 0.671 | 0.506 | 0.595 |
| P=S subfamily | 6 | 1.624 | 0.584 | 0.640 | 0.900 | 0.701 | 0.799 |
| P=S excluding TWE-24 | 5 | 1.129 | 0.616 | 0.454 | 0.948 | 0.740 | 0.843 |

The `cohort` row **contains** the S-donor rows below it, so the vertical comparison is partly a
population against itself and none of these differences carries an interval. Section 9 repeats
them against the disjoint 132 non-S ligands, with chemotype-block intervals; every one of those
intervals spans zero.

Zero-shot ranks within the cohort (1 = worst of 152): TWE-24 **1**, TWE-29 12, T8-PHEN-TAM 27,
TWE-23 41, TWE-27 86, TWE-30 87, T8-THP-CAM 89, cPSNDGA 93, TWE-28 96, Cy5-S-Me4-BTBP 139,
T8-THP-TAM 150.

---

## 9. Is this ligand an outlier or typical?

**An outlier; and there is no evidence that the S-donor class as a whole is a weakness.**

Every contrast in this section is **unpaired** — disjoint sets of ligands, no common rows — and
both subgroups were drawn after seeing which ligands failed, so all of it is exploratory. The
intervals below resample Tanimoto chemotypes and compare against the **132 non-S ligands with >=4
rows**, not against the 143-ligand cohort mean (which contains the S-donors and would be partly
comparing them with themselves).

| contrast (zero-shot unless noted) | group | non-S | difference | CI95 |
|---|---|---|---|---|
| 10 S-donors, TWE-24 excluded | 0.908 | 0.978 | **-0.070** | [-0.398, +0.325] |
| 11 S-donors, TWE-24 included | 1.198 | 0.978 | **+0.220** | [-0.307, +0.915] |
| 5 P=S, TWE-24 excluded | 1.129 | 0.978 | **+0.151** | [-0.246, +0.657] |
| 5 P=S, TWE-24 excluded, k=1 random | 0.948 | 0.712 | **+0.235** | [-0.183, +0.677] |
| 6 P=S, TWE-24 included, k=1 random | 0.900 | 0.712 | **+0.187** | [-0.171, +0.588] |

* Every one of those intervals spans zero. With 5 to 11 ligands, one per chemotype, this design
  cannot resolve a subgroup effect of the size in question. The honest statements are: **removing
  TWE-24 leaves no detectable S-donor penalty** (point estimate slightly in their favour, interval
  ±0.4), and **the P=S subfamily is not shown to be harder** — the point estimates lean that way
  and the mechanism is plausible, but nothing here establishes it. The first draft of this document
  called the P=S subfamily "genuinely harder" and said sulfur donors "are not harder than average";
  both were point estimates reported as verdicts.
* Two S-donors (Cy5-S-Me4-BTBP 0.240, T8-THP-TAM 0.172) are among the easiest ligands in the corpus.
* What *is* directly observable rather than inferential: the P=S family has the worst slope
  reproduction in the group (TWE-29 and TWE-23 flattened 11-15x), and half of its curves carry no
  usable slope information at all — TWE-27 and TWE-28 are flat lines at log D = -2.000 and TWE-23
  touches the -4.000 floor. Six molecules from one campaign, several of them censored, is not much
  chemistry to learn a family level from.
* What generalises from TWE-24 is not "soft donors break the model" but the standing gen8 result in
  its sharpest form: **the level is the failure and one measurement is the fix.** The worst ligand
  in the corpus, off by 4.1 log units, has a correctly shaped curve and drops to 0.658 after a
  single measurement — better than the average ligand's one-shot score.
* The one genuinely new lever this case exposes is upstream, and it is a **feature-set** lever
  rather than a data-quality one. The worst ligand in the corpus is worst because the bundle
  discards a recorded solvent composition and then asserts the opposite of it
  (`modifier_class=none` for a system with 12.5 vol% 1-octanol). The cheap fix is not a
  quarantine rule but a build-time one: keep the phase modifier as a fraction rather than an
  identity one-hot, so that a mixture the corpus has seen only once still enters the model as
  "TPH plus 12.5% alcohol" rather than as `other`.
* A level-outlier pre-flight check — flag any ligand whose level sits far from its nearest
  chemotype neighbours measured in the same campaign — would still have flagged TWE-24 (4.27 log
  units from its five-sibling mean, on a family whose own spread is 1.06). Worth having, but this
  case is a reminder that such a flag is a prompt to look for a missing covariate first and a
  transcription error second. The corpus's other known level anomaly, the DMDPhPDA exponent
  corruption, has the same signature and a different cause.
