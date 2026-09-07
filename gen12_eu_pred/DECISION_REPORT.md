# Gen12Eu_pred — decision report

*Europium extraction on chemically unseen extractants. Cohort fingerprint `2a364bb5264e9935`;
bundle SHA-256 `fefbefc6…4faf5dd`. Every number here is produced by the scripts in `scripts/` and
stored under `headline_tables/`, `metrics/` and `bootstrap/`. A regime is stated on every quantity:
{zero-shot | k-shot} x {design B | design A} x {extractant-macro | chemotype-macro | row} x
{overall | far | mid | near}. A number without its regime is not a number in this study.*

---

## Headline

**Europium extraction on a chemically unseen extractant is predictable in a weak sense and not in
the strong one.** The best zero-shot model reaches extractant-macro MAE **1.048** on design B
against a target spread of 1.90 and a predict-the-mean baseline of 1.731. **That trivial baseline
is the only one it significantly beats.** Measured against the control that matters — the *same learner* given experimental
conditions and no molecular information at all, at **1.134** — molecular structure is worth
**+0.086, 95 % BCa [−0.025, +0.189]**, an interval that includes zero, while the minimum detectable
effect at 80 % power is **0.163**. **H1 is not supported, and the cause is power rather than a
demonstrated absence**: this cohort resolves about twice the effect that appears to exist, and the
pre-registered equivalence margin of 0.02 is not met either. Error does rise with chemical distance
(Spearman −0.20 between train-similarity and per-extractant error), but not as a gradient: once each
band is normalised by its own target spread, far and mid are equally hard (0.65 and 0.68) and only
the near band is easier (0.46). The molecular advantage lives in the near band and **reverses on far
chemistry**, so **H2 fails**. Extremely randomised trees win: the descriptor MLP (1.241) and all three
directed message-passing variants (1.197 to 1.350) lose not only to the trees but to knowing
nothing about the molecule at all. **H3 is the one clean positive**: measurements of the new
extractant help monotonically and significantly, 1.271 → 1.006 → 0.926 → 0.893 → 0.867 at
k = 0, 1, 2, 3, 5, and **all of it is level and none of it is shape**. **H4 fails**: multi-lanthanide
training is worth **−0.003 [−0.026, +0.016]** once training mass is matched, while the same
experiment run *without* the leakage filter gains **+0.327** — the leak is four times larger than
everything molecular structure contributes, and is the single most useful number in this report.
**H5 is not reached**, there being no gain to control.

---

## 1. What was run

| phase | what | where |
|---|---|---|
| 0 | repository and Eu data audit | `DATA_AUDIT.md` — 0 fatal hazards, 3 recorded warnings |
| 1 | cohort, splits, metrics, hypotheses, seeds frozen | `PRE_REGISTRATION.md` |
| 2–5 | Tier 0 baselines, Tier 1 trees, Tier 2 MLP, Tier 3 D-MPNN | 15 arms on design B, 9 on design A |
| 6 | selection on inner validation only | reference arm fixed before any test score was read |
| 7 | locked zero-shot test evaluation | `headline_tables/t1_zero_shot_leaderboard.*` |
| 8 | similarity frontier | `bootstrap/B_frontier_by_band.csv`, `metrics/B/frontier_deciles.csv` |
| 9 | few-shot k = 1, 2, 3, 5 | `headline_tables/t5_fewshot_curve.*` |
| 10 | multi-lanthanide transfer, strict | `headline_tables/t6_multi_lanthanide.*` |
| 11 | matched controls and six feature ablations | `headline_tables/t7_ablation.*` |
| 12 | power, paired bootstrap, influence | `headline_tables/t4_power.*`, `t9_hypotheses.md` |
| 13 | this report | — |

19 invariant tests pass. 14 of 14 self-audit checks pass **against the artefacts on disk**, not
merely against freshly built objects.

## 2. The cohort, in one paragraph

1,329 cells — one row per (extractant, condition) — over **183 extractants, 162 bit-identical
fingerprint clusters and 97 Tanimoto-0.7 chemotypes**, built from the 1,566 Eu rows of the frozen
bundle by removing 125 rows whose structure is TODGA but whose recorded name is a different ligand,
then averaging replicates within a cell. The target is `log_D`, finite on every row, mean −0.060,
sd 1.899. No minimum-rows filter is applied: **85 extractants have exactly one measured condition
and 132 have a single series**, and those are exactly the unusual chemotypes a coverage filter would
delete. 63 % of the target variance is between extractants and 37 % within.

The binding constraint is stated before any result: Kish effective sample size is **13.7
extractants once weighted by chemotype**, not 183, and the largest chemotype holds 46 extractants
and half the rows.

## 3. Can Eu be predicted zero-shot on unseen extractants?

Zero-shot, design B (chemotype hold-out), extractant-macro MAE, 5 split seeds:

| arm | macro MAE | chemotype-macro | pooled MAE | pooled R² | level | shape |
|---|---|---|---|---|---|---|
| **ABL_D — trees, molecule + conditions + 2D descriptors** | **1.048** | 0.853 | — | — | 0.917 | 0.578 |
| ABL_C = T1_EXTRATREES — trees, molecule + conditions | 1.092 | 0.887 | 1.266 | 0.240 | 0.977 | 0.569 |
| ABL_E — trees, fingerprint + conditions | 1.107 | 0.916 | — | — | 0.994 | 0.562 |
| T1_RANDOMFOREST | 1.134 | 0.900 | 1.300 | 0.213 | 1.016 | 0.586 |
| **ABL_A — trees, conditions only (the matched control)** | **1.134** | 1.025 | — | — | 1.029 | 0.612 |
| T1_XGBOOST | 1.140 | 0.918 | 1.305 | 0.208 | 1.017 | 0.591 |
| T1_CATBOOST | 1.149 | 0.953 | 1.315 | 0.205 | 1.013 | 0.630 |
| B1_COND_ONLY — boosting, conditions only | 1.167 | 1.089 | 1.441 | 0.059 | 1.053 | 0.649 |
| B2_NN_CHEMICAL — nearest-chemistry interpolation | 1.170 | 1.111 | 1.388 | 0.113 | 1.060 | 0.623 |
| ABL_B — trees, molecule only | 1.215 | 1.031 | — | — | 1.072 | 0.624 |
| B3_NN_LEVEL_ONLY | 1.222 | 1.037 | 1.431 | 0.031 | 1.100 | 0.625 |
| T2_MLP | 1.237 | 1.095 | 1.365 | 0.179 | 1.133 | 0.595 |
| B0_GLOBAL_MEAN | 1.731 | 1.757 | 1.818 | −0.269 | 1.641 | 0.625 |

**The answer to "can it be predicted" depends entirely on what you compare against.**

| the champion is compared against | delta | BCa | verdict |
|---|---|---|---|
| predicting the training mean | +0.683 | [+0.519, +0.887] | **beaten decisively** |
| nearest-chemistry level lookup (`B3`) | +0.174 | [−0.011, +0.341] | not established |
| nearest-chemistry interpolation (`B2`) | +0.123 | [−0.180, +0.335] | not established |
| the same learner, conditions only (`ABL_A_CONDITIONS`) | +0.086 | [−0.025, +0.189] | **not established — this is H1** |

**Only the trivial mean baseline is significantly beaten.** Every baseline that knows *something* —
the nearest training chemistry, or the experimental conditions — survives the comparison. The point
estimates all favour the model and 5 of 5 split seeds do too, but no interval clears zero, and the
minimum detectable effect at 80 % power is 0.163 for H1 and 0.28 against nearest-chemistry
interpolation.

**A methodological correction, applied and reported rather than absorbed.** The Tier-0
condition-only baseline uses histogram gradient boosting while the champion uses extremely
randomised trees, so a contrast between them confounds *information* with *learner*. Against that
unmatched control the apparent chemistry gain is +0.119; against the matched control it is +0.086.
**A third of the apparent effect was the learner.** Every H1 and H2 number in this report uses the
matched control, `ABL_A_CONDITIONS`.

Two further properties of the champion. Predictions are **shrunk**: sd 1.13 against a truth sd of
1.90, a dispersion ratio of 0.60, with calibration slope 0.98 and pooled Spearman 0.53. And **the
error is mostly level**: offset 0.917 against shape 0.578, so 61 % of the two components. A perfect
per-extractant level with no shape model at all would score **0.334**; the champion scores 1.048.
Almost the entire gap is the level of a molecule the model has never seen.

## 4. How much does accuracy deteriorate with chemical distance?

Under design B, holding out a single-linkage Tanimoto-0.7 cluster **caps train similarity at 0.698
by construction**. "Near" therefore means 0.60 to 0.698 — just below the clustering threshold — and
not "a close homologue was in training". The same cap applies to gen10 and gen11.

| band | rule | extractants | chemotypes | rows/seed | champion macro MAE | matched control | target sd | champion / band sd |
|---|---|---|---|---|---|---|---|---|
| far | ≤ 0.40 | 27 | 25 | 76 | 1.062 | 1.140 | 1.624 | 0.654 |
| mid | 0.40–0.60 | 103 | 59 | 395 | 1.328 | 1.437 | 1.954 | 0.680 |
| near | > 0.60 | 105 | 58 | 858 | 0.834 | 0.906 | 1.811 | 0.461 |

**The frontier is a step, not a gradient.** Raw macro MAE is worst in the *mid* band, which looks
paradoxical until each band is normalised by its own target spread: far and mid are then equally
hard (0.654 and 0.680) and only the near band is easier (0.461). The mid band is not more distant
chemistry, it is chemistry with a wider `log_D` range. Reading the raw band table as a pure
distance effect would be wrong, and this is why the normalised column is printed beside it.

Inside each band, against the matched no-chemistry control, **no molecular arm's interval excludes
zero**:

| band | champion − control | BCa | MDE at 80 % | blocks |
|---|---|---|---|---|
| far | +0.065 | [−0.134, +0.288] | 0.302 | 25 |
| mid | +0.109 | [−0.082, +0.287] | 0.290 | 59 |
| near | +0.066 | [−0.024, +0.146] | 0.119 | 58 |

For `ABL_C_MOL_COND` (no 2D descriptors) the far-band contrast is **−0.031**: the model is *worse* than
knowing nothing about the molecule. **H2 fails.** Across extractants the frontier is real but
modest — Spearman between train-similarity and per-extractant error is −0.20 for the champion and
−0.17 for the no-chemistry control, so a third of the apparent distance effect is present without
any chemistry at all.

## 5. Which model actually wins?

**Extremely randomised trees, on fingerprints plus conditions plus donor census plus 2D
descriptors.** The ordering is stable between inner validation and the locked test, which is the
check that the selection was honest: ExtraTrees is first on validation (0.920) and first among
Tier-1 arms on test.

Neural models lose. The descriptor MLP scores 1.241 and the best graph variant 1.197, both worse
than conditions alone at 1.134. §9 gives the graph result in full.

**One inversion is worth reporting because it cuts against the pre-registration.** The `LIG2D`
block of 206 extended 2D descriptors was excluded from the primary contract because gen7 measured
that deleting exactly that block was a significant improvement — the only block whose removal was a
significant gain. On this Eu-only task **adding it is a significant improvement: +0.044, BCa
[+0.016, +0.072], 118 of 183 extractants improved.** But it is **worse on inner validation**
(0.925 against 0.920), so the pre-registered selection rule does not choose it and it cannot be
promoted to the headline. The honest statement is: the pre-registered primary model is `ABL_C_MOL_COND` at
1.092; the pre-registered ablation D beats it on test by 0.044 while losing on validation by 0.005;
and a study that had selected on test would have reported the better number. This is exactly the
failure mode the selection rule exists to prevent, and it fired.

The other ablations, all against `ABL_C_MOL_COND`:

| arm | contents | macro MAE | vs C | BCa |
|---|---|---|---|---|
| A | conditions only | 1.134 | −0.043 | [−0.155, +0.091] |
| B | molecule only | 1.215 | −0.123 | [−0.188, −0.065] |
| C | molecule + conditions | 1.092 | — | — |
| D | C + 2D descriptors | 1.048 | **+0.044** | [+0.016, +0.072] |
| E | fingerprint + conditions | 1.107 | −0.015 | [−0.033, −0.002] |
| F | learned graph + conditions | 1.275 | **−0.184** | [−0.279, −0.100] |

Reading them mechanistically: conditions alone are worth far more than molecule alone (1.134 vs
1.215), the two together beat either, the donor census and physico-chemical scalars contribute a
small but resolvable amount (C beats E), hand-crafted 2D descriptors contribute more than the
fingerprint's marginal content, and **replacing the fingerprint with a learned graph representation
is the single most damaging change in the set** (F, −0.184, 48 of 183 extractants improved).

**Influence.** Removing the dominant chemotype — 46 extractants, half the rows, the diglycolamide
family — *increases* the chemistry advantage from +0.086 to +0.147, and no single chemotype flips
the sign of any headline contrast. The effect is diluted by the dominant family, not manufactured
by it.

## 6. How much do 1, 2, 3 and 5 Eu observations help?

Learning curve on the **common cohort** — the 42 extractants in 21 chemotypes with at least seven
cells, so the cohort does not shrink with k:

| k | model + shrunk offset | model + plain offset | no model: mean of measured |
|---|---|---|---|
| 0 | 1.271 | 1.271 | — |
| 1 | **1.006** | 1.068 | 1.238 |
| 2 | **0.926** | 0.939 | 1.095 |
| 3 | **0.893** | 0.896 | 1.048 |
| 5 | **0.867** | 0.865 | 1.015 |

Monotone at every step, and the model plus offset beats the no-model null at every k with BCa
excluding zero. **H3 is supported.** This is the one hypothesis that passes cleanly.

**Every bit of it is level.** Over the same range the level component falls 0.990 → 0.437 while the
shape component barely moves, 0.767 → 0.728. Offset calibration cannot change shape by
construction, and nothing else in the k-shot chain does either. The practical reading is that a
measurement tells you *where* an extractant sits, and nothing about *how it responds*.

**Shrinkage matters only at k = 1** (−0.061, BCa [−0.108, −0.007] against the plain offset) and is
indistinguishable from it at k ≥ 2. This reproduces the repository's "shrinkage, not degrees of
freedom" correction on a new target.

**Does one measurement beat the whole model?** Scored on identical query rows across the 80
extractants with at least three cells: zero-shot model **1.154**, one measurement with no model at
all **1.030**, model plus one measurement **0.922**.

- model + 1 vs zero-shot: **+0.233, BCa [+0.074, +0.418]** — significant.
- model + 1 vs one measurement alone: **+0.109, BCa [+0.029, +0.213]** — significant.
- one measurement alone vs zero-shot model: +0.124, BCa [−0.092, +0.341] — **suggestive, not
  established.**

So on this task the model still earns its place after the first measurement, which is a genuine
difference from the pair-target result five generations earlier where a no-model fit caught up by
k ≈ 2.

**The crossover is chemical, and it is deployable.** On the common cohort, by band: in the mid band
one measurement alone (1.092) beats the zero-shot model (1.764); in the near band the zero-shot
model (0.949) beats one measurement alone (1.330). **Trust the model on chemistry close to what it
has seen; measure once on chemistry that is not.**

**The most important limitation in this section.** The extractants that most need few-shot are the
ones on which it cannot be evaluated: **only 4 of the 15 far-band extractants have three or more
cells**, and only 2 have seven. The far column of the learning curve rests on 3 extractants and is
not reported as a result. Few-shot on genuinely distant chemistry is untested here, not tested and
found wanting.

## 7. Do other lanthanides improve Eu prediction?

The audit fixed the shape of this question before the arm ran: **4,148 of the 4,426 non-Eu rows sit
on extractants already in the Eu cohort**, and exactly **one** auxiliary chemotype is absent from
it. The archive adds metals and curves, not chemistry.

All arms share folds, test rows and learner; only the training pool differs.

| arm | macro MAE | vs matched control | BCa | deployable |
|---|---|---|---|---|
| `LEAKY_NO_FILTER` — the naive multi-metal analysis | **0.741** | **+0.327** | **[+0.204, +0.465]** | **no** |
| `MATCHED_EU_ONLY` — Eu rows, training mass matched | 1.068 | — | — | yes |
| `STRICT` — all lanthanides, held-out chemotypes deleted under every metal | 1.071 | −0.003 | [−0.026, +0.016] | yes |
| `EU_ONLY_ANCHOR` — the frozen champion | 1.092 | −0.023 | [−0.041, −0.004] | yes |
| `PERMUTED_METAL` — auxiliary metal identity shuffled | 1.101 | −0.033 | [−0.069, −0.013] | control |
| `SHUFFLED_TARGET` — auxiliary targets shuffled | 1.114 | −0.046 | [−0.096, +0.029] | control |

**H4 fails.** Strict multi-lanthanide training is worth −0.003 with an interval well inside the
0.01 reproducibility floor. The strict filter admits 3,144 auxiliary rows over 76 extractants per
fold, with a per-fold assertion of zero test-extractant overlap, so the null is not a filtering
artefact — there is simply nothing to transfer once the test chemistry is properly removed. H5 is
not reached.

**The number worth carrying away is the leaky one.** Running the identical experiment without the
chemotype filter — the analysis a naive multi-metal study would report — gains **0.327 macro MAE**
with an interval far from zero, because 17.6 test extractants per fold survive in the training pool
under another metal. **That leak is four times larger than everything molecular structure
contributes in this study.** Any multi-metal extraction paper that does not delete held-out
chemistry under *every* metal is reporting this artefact.

Two smaller results. Permuting metal identity is significantly *worse* than the matched control, so
the metal descriptors are doing real work as features even though the extra rows are not. And the
matched control significantly beats the unmatched anchor (+0.023), so **part of what looks like
transfer in an unmatched design is just training mass**.

## 8. The cost of the wrong split, measured

The exact-extractant hold-out — the design most readers would build — leaks **139 bit-identical
fingerprint clusters and 234 chemotypes across 25 folds** and reaches Tanimoto 1.0.

| arm | design A (exact extractant) | design B (chemotype) | inflation |
|---|---|---|---|
| T1_EXTRATREES | 0.772 | 1.092 | −0.320 |
| T1_XGBOOST | 0.800 | 1.140 | −0.341 |
| B1_COND_ONLY | 0.883 | 1.167 | −0.284 |
| B0_GLOBAL_MEAN | 1.602 | 1.731 | −0.129 |

The naive split makes the champion look **0.320 better**, and **88.7 % of that advantage is
available to a model with no molecular information at all**. It is not a chemistry result; it is a
condition-matching result. Design A numbers appear in this report only in this table.

## 9. The graph model

The pre-registered primary neural candidate is a directed message-passing network over the
extractant graph, with the experimental conditions entering **after** message passing as a
concatenated descriptor vector. It is reproduced rather than imported: `chemprop==2.3.1` installs
without moving any of numpy, pandas, scipy, scikit-learn, PyTorch, RDKit, CatBoost, XGBoost or
pyarrow — checked by dry run before installing — but cannot be *imported* here, because Lightning
pulls `torchmetrics`, which eagerly imports `transformers`, whose installed version needs a newer
`huggingface_hub` than the one gen7's ChemBERTa path is pinned against. Repairing that means
mutating a shared environment five generations' numbers are defined against, and this repository
has already been moved once by a `pip install`. The implementation in `gen12eu/dmpnn.py` is the
Yang et al. (2019) architecture Chemprop v2 implements: bond-centred messages, the
`sum over N(u)\{v}` update written as `(everything into u) − h_uv` so the pass is O(E), a residual
on the initial message, and a mean over atom states.

Zero-shot, design B, extractant-macro MAE, with the matched no-chemistry control for reference:

| arm | macro MAE | vs `ABL_A_CONDITIONS` | BCa | seeds favouring the graph |
|---|---|---|---|---|
| `ABL_A_CONDITIONS` (no molecule at all) | 1.134 | — | — | — |
| `T3_DMPNN_COND_DESC` | 1.197 | −0.063 | [−0.140, +0.026] | 2 of 5 |
| `T3_DMPNN_COND` | 1.275 | **−0.141** | [−0.213, −0.058] | 0 of 5 |
| `T3_DMPNN_GRAPH_ONLY` | 1.350 | **−0.216** | [−0.343, −0.119] | 0 of 5 |

**Every graph variant loses** — to the trees, to the descriptor MLP, and to knowing nothing about
the molecule at all. Two of the three are *significantly* worse than the no-chemistry control, and
the third is worse by 0.063 with an interval that reaches only just past zero. The ordering within the family is informative — graph only 1.350, plus
conditions 1.275, plus RDKit descriptors 1.197 — so conditions and hand-crafted descriptors each
help the graph model, and it still never reaches the fingerprint trees at 1.048.

**Ablation F, the question the ablation set exists to answer, is answered negatively: graph learning
adds nothing over a fingerprint on this cohort.** At matched information, `ABL_E_ECFP_COND` (fingerprint +
conditions, trees) scores 1.107 against `ABL_F_GRAPH_COND` (the same information as a learned graph
representation) at 1.275 — the widest gap in the ablation set. With 183
independent molecules and 97 chemotypes there is not enough chemistry to fit a representation from
scratch, and a 2,048-bit fingerprint that encodes the same substructures by hand is the better
inductive bias.

**A defect in this arm, found after it ran, fixed, and re-run.** `T3_DMPNN_COND_DESC` originally
diverged in **8 of 25 folds**, with inner-validation MAE up to 3.5e11. The cause was not the
architecture: `lig2d__rd__Ipc` spans 1.1e8 to 3.8e29 across these ligands, and under a chemotype
hold-out the extreme molecule is usually absent from the training fold, whose standard deviation is
then 5.3e12, so the held-out value standardised to **7.3e16**. Trees only compare their inputs and
were immune; the network multiplied it. **The pre-registered prediction clip partly hid the
failure**, because every diverged output was clamped back into the training target range and the
per-fold MAEs still looked plausible. The repair clamps a held-out value to the training fold's
observed range before standardising; across all 25 folds the largest standardised input a model
receives falls from **7.255e16 to 33.93**, and divergence goes from 8 folds to 0. The change is
recorded as a dated addendum in `PRE_REGISTRATION.md`, results having already been seen, and both
sets of predictions are kept — the pre-guard ones under `predictions/B_pre_clip_guard/`.

| arm | before the guard | after | change |
|---|---|---|---|
| `T3_DMPNN_COND_DESC` | 1.237 | **1.197** | −0.040 |
| `T3_DMPNN_GRAPH_ONLY` | 1.322 | 1.350 | +0.029 |
| `T3_DMPNN_COND` | 1.271 | 1.275 | +0.004 |
| `T2_MLP` | 1.237 | 1.241 | +0.003 |
| `B2_NN_CHEMICAL` | 1.170 | 1.170 | 0.000 |

**No conclusion in this report turns on the repair.** Every headline arm is a tree, trees do not
standardise, and the graph models lose by a wide margin in both the diverged and the repaired form.

## 10. What is statistically supported, and what is merely suggestive

**Supported** (BCa excludes zero, effect above the 0.01 reproducibility floor, sign stable under
leave-one-chemotype-out):

- every model beats predicting the training mean;
- adding 2D descriptors improves test macro MAE by 0.044;
- removing conditions, or removing the molecule, each makes things significantly worse;
- k-shot improves monotonically and beats every null at k = 1, 2, 3, 5;
- model plus one measurement beats both the zero-shot model and one measurement alone;
- shrinkage beats a plain offset at k = 1 and not beyond;
- the naive multi-metal analysis gains 0.327 purely from leakage;
- permuting metal identity hurts;
- two of the three graph variants are significantly *worse* than knowing nothing about the molecule
  (−0.141 and −0.216, both BCa excluding zero);
- replacing the fingerprint with a learned graph representation is the most damaging single change
  in the ablation set (−0.184, BCa [−0.279, −0.100]).

**Suggestive but not established** — point estimate in the expected direction, interval containing
zero:

- molecular structure beats conditions alone by 0.086 (5/5 seeds);
- the champion beats nearest-chemistry interpolation by 0.123 and the nearest-chemistry level
  lookup by 0.174;
- one measurement alone beats the zero-shot model by 0.124;
- the champion beats the no-chemistry control on the far band by 0.065.

**Not supported**: H1, H2, H4, H5.

**Nothing here supports an equivalence claim.** The pre-registered margin was 0.02 macro MAE and no
comparison meets it. The MDE at 80 % power is 0.163 for the primary contrast, 0.30 on the far band
and 0.19 for the multi-lanthanide arm. **This cohort is powered to detect roughly twice the effect
that appears to exist**, and that was known and written down before the comparisons were run.

## 11. Negative results and defects found during the work

1. **H1, H2, H4 and H5 all fail.** Only H3 passes. The generation's headline is a negative result
   with a measured power bound, not a modelling win.
2. **A learner/information confound in our own baseline.** The Tier-0 condition-only arm used a
   different learner from the champion; a third of the apparent chemistry effect was the learner.
   Found by adding the matched ablation arm, corrected everywhere, and both numbers reported.
3. **The two pre-registered k-shot nulls are the same estimator.** "Corpus level shifted by the k
   measurements" is algebraically `mean(y_support)` — the corpus level cancels exactly. They were
   run, produced identical columns, and the identity is now an executable test. The second null's
   role is filled instead by the nearest-chemistry level arm with the same k points, which turns out
   to be *also* identical to the no-model null once a plain offset is applied to a
   constant-per-extractant prediction. **Two of the three pre-registered nulls collapsed into one.**
4. **A bug in our own invariant checker.** The "all arms scored on identical rows" check hashed
   `DataFrame.to_numpy().tobytes()` on a mixed-dtype frame, which hashes object *pointers*. It
   reported every arm as mismatched. Fixed to hash text; the check now passes on 15 arms.
5. **Chemprop could not be imported**, for a reason unrelated to chemprop: Lightning pulls
   `torchmetrics`, which eagerly imports `transformers`, whose installed version needs a newer
   `huggingface_hub` than the one gen7's ChemBERTa path is pinned against. Repairing it would mutate
   an environment five generations' numbers are defined against, and this repository has already
   been moved once by a `pip install`. The D-MPNN was reproduced instead, as the pre-registration
   permits, and the install was dry-run first and confirmed to move none of numpy, pandas, scipy,
   scikit-learn, PyTorch, RDKit, CatBoost, XGBoost or pyarrow.
6. **A silent all-NaN column inside a training fold.** A condition column observed in the cohort can
   be empty in one fold. This crashed a gen5 run outright; here it produced a warning and a column
   the preprocessor silently dropped. Now handled explicitly with a test, and the fix is proven to
   be numerically a no-op against the previous path.
7. **The review-queue stratum runs the wrong way here.** Rows flagged by the curated data-quality
   queue are *easier* for the model (macro MAE 0.784 flagged against 1.130 clean), the opposite of
   the repository's own triage finding. The explanation is confounding, not contradiction: 225 of
   the 343 flagged rows are `diluent_name_ambiguous`, concentrated in heavily-measured
   diglycolamide systems. The stratum is reported, not used.
8. **The far band cannot carry a few-shot result.** 4 of 15 far extractants have three or more
   cells. This was discovered after the k-shot design was frozen and is reported as a gap.
9. **A heavy-tailed descriptor silently poisoned the neural arms, and the safety clip hid it.**
   `lig2d__rd__Ipc` spans 1.1e8 to 3.8e29 over these 183 ligands. Fold-local standardisation is not
   a safe transform for such a column under a chemical hold-out: when the extreme molecule is held
   out the training sd is 5.3e12, and the held-out value standardises to **7.3e16**. Trees only
   compare and were immune; the D-MPNN multiplied it and diverged in 8 of 25 folds. **The
   pre-registered prediction clip then masked the divergence**, clamping every diverged output back
   into the training target range so the per-fold MAEs looked ordinary — only the inner-validation
   log gave it away. Found by reading a selection log, repaired with a training-range clamp,
   re-run, and recorded as a dated addendum. Largest standardised input across 25 folds: 7.255e16
   before, **33.93** after; divergence 8 folds before, 0 after. **A safety device that makes a
   failure invisible is worse than no safety device**, and the clip is now reported beside every
   neural arm rather than left silent.
10. **The champion's worst errors are a coherent chemical family, not noise.** Ten of the fifteen
   worst extractants are bridged, tripodal or multi-armed diglycolamides, and the model
   **under-predicts every one of them by 2.6 to 4.2 decades**. Their true `log_D` averages +2.5 to
   +3.0 against a cohort mean of −0.06. Seeing a diglycolamide-like fingerprint, the model predicts
   diglycolamide-like behaviour and misses that multiplying the binding arms multiplies the
   extraction. This is the single clearest chemical failure mode in the study and it is a
   representation failure, not a data failure.
11. **The phosphorothioate the repository's review queue disputes appears again**, as the second
    worst extractant at MAE 4.06 with 4.06 of under-prediction. In this cohort it carries the name
    TWE-24, six cells, `log_D` from 0.75 to 2.71, and no DOI. Gen12 does not resolve the identity
    question; it records that the same structure is again an extreme outlier.

## 12. Guardrails — what a reader must not do with these numbers

1. **Do not quote a macro MAE without its split design.** The same frozen champion scores 1.092
   under a chemotype hold-out and 0.772 under an exact-extractant hold-out. Only the first is a
   statement about unseen chemistry.
2. **Do not read "near" as "a close analogue was in training".** Design B caps train similarity at
   0.698 by construction, so near means 0.60–0.698. There is no genuinely-near band in this study.
3. **Do not read the raw band table as a distance gradient.** Mid is the worst band in raw macro MAE
   only because it has the widest target spread. Normalise by the band's own sd before comparing.
4. **Do not treat any non-significant contrast here as equivalence.** The pre-registered margin is
   0.02 and nothing meets it. The MDE is 0.163 overall and 0.30 on the far band.
5. **Do not mix the few-shot cohort with the zero-shot one.** k-shot numbers are on 42 extractants
   in 21 chemotypes; zero-shot numbers are on 183 in 97. The zero-shot value on the few-shot cohort
   is 1.271, not 1.048.
6. **Do not quote the ablation-D number as the model's score.** 1.048 is the best test score; 1.092
   is what the pre-registered selection rule chose. A study that reports the first without the
   second has selected on test.
7. **Do not compare a Tier-0 baseline with a Tier-1 model to size "the value of chemistry".** They
   use different learners; that contrast is a third larger than the matched one.
8. **Do not carry the multi-lanthanide leaky number anywhere except as a warning.** 0.741 is what a
   naive analysis reports and it is an artefact.
9. **Do not average across split seeds before centring within an extractant.** gen11 measured that
   this scores between-seed level wobble as shape and it reached a published headline. Every
   decomposition here centres per (split_seed, extractant).
10. **An effect below 0.01 macro MAE is not an effect** — it is within this pipeline's measured
    cross-machine reproducibility.

## 13. Recommendation

**Do not deploy a zero-shot Eu model for chemistry outside the training chemotypes.** On far
chemistry it does not beat knowing the experimental conditions alone, and the interval is wide
enough that it could be worse.

**Deploy the measure-once protocol instead.** One measurement of a new extractant, combined with the
model through a shrunk offset, reaches 0.92 macro MAE against the zero-shot 1.15 on the same query
rows, and the gain is statistically supported. Trust the model unaided only inside the near band.

**What would settle the open question.** The primary contrast needs about four times the
independent chemistry to resolve a 0.09 effect: roughly 100 additional chemotypes, not 100
additional rows. Depth on existing families will not help — the dominant family already dilutes the
effect. The cheapest informative next experiment is not another architecture; it is measuring two
or three conditions on each of the 85 extractants that currently have one, which would move most of
the cohort into few-shot range and make the far band testable for the first time.
