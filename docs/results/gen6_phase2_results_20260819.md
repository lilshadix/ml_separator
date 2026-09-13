# gen6 — Phase 2 results: Experiments C and F (2026-08-19)

Protocol: [`gen6_phase2_protocol_20260819.md`](../protocols/gen6_phase2_protocol_20260819.md), written after
Phase 1 reported case A and before any Phase 2 model was fitted; one amendment (`C2_TRUECENTRE`) is
recorded in it with its reason and its timing. Engineering contract:
[`gen6_phase2_runner_contract.md`](../protocols/gen6_phase2_runner_contract.md).

| what | run directory | command |
|---|---|---|
| Experiment C, 3 regimes × 5 seeds | `runs/gen6_expC_5seed/` | `scripts/run_hierarchical_levels.py` |
| Experiment F, 3 seeds × 5 folds × 6 policies × 2 reps | `runs/gen6_expF_3seed/` | `scripts/run_ligand_acquisition_sim.py` |
| Experiment F, row weighting (sensitivity, 4 policies) | `runs/gen6_expF_3seed_rowweight/` | `… --weighting row` |
| fold-seed-paired intervals for both | `foldseed_paired_contrasts.csv` in each | `scripts/foldseed_paired_intervals.py` |

Everything ran locally. Phase 1's cohort, folds, learner, metrics and intervals are unchanged.

**Process disclosure.** The Phase 2 protocol was written before any Phase 2 model was fitted, but it
was committed to git only together with the results (this session's transcript is the record), so
its pre-registration status rests on that statement rather than on an immutable timestamp. Every
review correction made after the runs is recorded in the text at the point it applies.

---

## 0. The headline

**The (ligand, condition) level carries most of the error a new ligand's prediction could lose —
and structure does not buy it.** On multi-metal test cells of new chemotypes, handing the model the
true level removes 0.50 log units of macro MAE and handing it the true metal response 0.10 (against
the deployable monolith; difference +0.39, BCa [+0.27, +0.56], 5/5 seeds; +0.32 with a
leave-one-out level; direction robust, magnitude construction-dependent). The two-stage model is
*worse* than the monolith; the hierarchical ridge gains nothing over plain ridge on new chemistry
and only +0.04 (wide interval, 5/5 seeds) where a ligand's other series are known. A ligand's level
is a ligand constant plus a series term about half its size (sd 1.45 vs 0.72), so a future level
model should carry `α_{ligand, series}` — the experiment that would size that term precisely was
not run.

**Which ligand to measure next: the one farthest from what you have; three points of it.** From a
narrow ten-ligand DGA start, max-min Tanimoto acquisition beats random acquisition on hard chemistry
at every checkpoint, under both a row-pooled and a fold-seed-paired reading (+0.21…+0.28 log units
paired, 13 of 15 fold-seeds). Weighting the model's own uncertainty into the choice ties max-min
on hard chemistry and adds a little late on well-covered chemistry (descriptive). Buying another
DGA analogue buys nothing. Revealing *every* row of an acquired ligand instead of three buys 0.015.
The absolute effect sizes ride on the gen5 equal-cluster weighting — 90 acquired rows carry 83 % of
the training weight — and the row-weighted rerun is reported beside them.

Both experiments kept the control exact: the monolithic forest reproduces Experiment A's EXPANDED
arm to 2.2e-15 on all 26,240 rows.

---

## 1. Facts established before the runs (one seed, independent of the runner scripts)

These were computed directly from the tested modules to fix expectations before the full runs, and
are reported because several of them are findings in their own right.

### 1.1 Where the variance is

| decomposition of log D sum of squares, shared cohort | share |
|---|---|
| between (ligand, condition) cells — the *level* | **92 %** |
| within cells — the *metal response* | 8 % |
| within ligand (conditions + metals) | 49 % |
| between ligands | 51 % |
| between series **of the same ligand** | 12 % |
| within series | 37 % |

Only **521 of 2,405** (ligand, condition) cells hold two or more metals; they carry all the
metal-response information in the data (3,364 rows). And for the 55 ligands measured in more than
one series, the **series means of the same ligand differ by sd 0.72 log units** (median range 1.06,
maximum 5.1) — as large as the within-series spread (0.78). A ligand's "level" is to a large extent
a (ligand, series) property, not a ligand constant. That fact decides C-H2 below.

### 1.2 The attribution, by distance to training chemistry

Two-stage model with Stage B on the true within-cell departure, chemotype folds, seed 104729, on
multi-metal test cells only (a singleton cell makes the level oracle trivially perfect):

| nn Tanimoto to training | rows | clusters | C2 | level oracle | metal oracle | share of removable error that is *level* |
|---|---|---|---|---|---|---|
| < 0.4 | 340 | 11 | 0.902 | **0.284** | 0.866 | **95 %** |
| 0.4–0.6 | 500 | 27 | 0.849 | 0.384 | 0.717 | 78 % |
| 0.6–0.8 | 2,524 | 39 | 0.824 | 0.310 | 0.727 | 84 % |

Giving the model the true (ligand, condition) level removes 78–95 % of what either oracle can
remove, most of all on the most distant chemistry. This is Phase 1's offset finding, now made with
a model whose pieces are separately estimated rather than by decomposing a monolith's residuals.

### 1.3 The cross-fit residual trap (why the protocol has an amendment)

The pre-registered `C2_TWO_STAGE` trains Stage B on the cross-fitted residual `y − Â_oof`. On the
first seed it was *worse* than the monolithic forest — 1.254 vs 1.076 macro MAE — and worse than its
own Stage A alone (1.067). Mean |B̂| was 0.53. With Stage B trained on the true within-cell
departure `y − ȳ_{l,c}` (the brief's own definition), mean |B̂| is 0.15 and the model scores
**1.042**. The cross-fitted residual under a chemotype hold-out is mostly Stage A's *level* error on
chemistry the inner model had not seen; Stage B learns it from in-sample ligand features and
mis-applies it to new ligands. The amendment adds `C2_TRUECENTRE`; because it was chosen after
seeing one seed, its result is labelled exploratory.

### 1.4 The inner CV discovers what transfers

Under the chemotype hold-out, the inner CV chose ligand-intercept penalties that shrink the fitted
intercepts to a mean |γ| of **0.09** — it learned that per-ligand intercepts will not transfer and
turned them off. Under the series hold-out (ligand known) it kept them live: mean |γ| **0.49**,
contributing 0.34 log units per test prediction. Component attribution of C1's test predictions
(mean |contribution|): conditions 0.32–0.40, the declared interaction block 0.27–0.29, donor census
0.15–0.20, physchem 0.09–0.13, mass-action logs 0.07–0.08, metal 0.07–0.10.

### 1.5 What "shape" is made of

Within a (ligand, condition) cell — the metal response alone — the monolithic forest reaches shape
R² 0.30 (MAE 0.29 on the 521 multi-metal cells); across conditions within a ligand it is 0.16
(MAE 0.50). What little shape skill the model has is mostly metal *ordering*, not condition
response.

### 1.6 The derived pair task

The within-cell metal pairs of the five chemotype test folds are exactly the **14,173 pairs / 90
ligands** of the expanded pair cohort. On the one-seed preview (200 trees), level-derived pair
predictions from the monolithic forest scored macro MAE 0.511 / sign accuracy 0.781 against the
pair-label-mean null's 0.678 / 0.750 and a constant-zero null's 0.612; antisymmetry was exact and the
transitivity residual over 21,223 triples 4.4e-16. The 5-seed numbers are in §2.3.

---

## 2. Experiment C — the full run

`runs/gen6_expC_5seed/`: 3 regimes × 5 split seeds × 5 folds, 8 models, 400 trees; fitting 649 s.
**The control reproduces Experiment A's EXPANDED arm to 2.2e-15 over all 26,240 rows** — MONO_ET is
the same model, not an approximation of it. All eight models are scored on byte-identical test rows.
One post-run amendment is recorded in the report itself: C3's verdict label was re-scored after a
runner defect was fixed (the ≥ 4/5-seed clause had been applied in the *beat* direction to a
hypothesis whose PASS side is *no beat*, turning a 0/5-seed no-beat into INCONCLUSIVE); no number
changed.

### 2.1 Verdicts

| H | statement | verdict | the decisive number |
|---|---|---|---|
| **C1** | the level carries most of the removable error, not the metal response | **PASS**, direction robust / magnitude construction-dependent | multi-metal cells, **reference named**: against `MONO_ET` the level oracle removes 0.497 and the metal oracle 0.105; difference **+0.392** BCa [+0.273, +0.557], 77 clusters / 45 blocks, 5/5 seeds (the reference cancels exactly, 5.6e-17). Leave-one-out level: +0.324 [+0.188, +0.501] |
| **C2** | partial pooling helps exactly where the ligand is known | **INCONCLUSIVE** | as scored (all rows) Δ = −0.011, BCa [−0.049, +0.036], 1/5 seeds — **but 4,271 of 26,240 rows have their ligand held out entirely** (97 of 152 ligands have one series), so the intercept is dead there. Where it is live: Δ = **+0.036**, BCa [−0.046, +0.151], **5/5 seeds**; where dead −0.038, 0/5. Under `unseen_chemotype` −0.005, as pre-registered |
| **C3** | structure does not substitute for coverage | **PASS** | cross-fitted C2 vs the monolith: Δ = **−0.144**, BCa [−0.240, −0.069], 0/5 seeds — the structured model is *worse*; the amended `C2_TRUECENTRE` is indistinguishable, Δ = +0.004, BCa [−0.019, +0.033] (exploratory) |
| **C4** | level-derived pairs keep exact antisymmetry and transitivity | **PASS** — an algebraic check, not empirical evidence | antisymmetry 0; transitivity 8.9e-16 over 318,345 triples / 212,595 pairs / 8 models. A derived pair *is* ŷ(A) − ŷ(B), so both hold by construction; the check catches a pipeline that stops deriving them that way and cannot fail otherwise |

### 2.2 Leaderboard (macro MAE, one ECFP cluster = one vote; mean ± sd over 5 seeds)

| model | `unseen_chemotype` | `unseen_ligand` | `unseen_series` |
|---|---|---|---|
| MONO_ET (= Experiment A EXPANDED) | **1.047** ± 0.015 | **0.853** ± 0.016 | **0.808** ± 0.011 |
| MONO_RIDGE | 1.148 ± 0.030 | 1.037 ± 0.027 | 1.022 ± 0.022 |
| C1_HIER_RIDGE | 1.152 ± 0.024 | 1.048 ± 0.027 | 1.033 ± 0.021 |
| C3_SHARED_RIDGE | 1.151 ± 0.026 | 1.047 ± 0.029 | 1.037 ± 0.020 |
| C2_TWO_STAGE (cross-fitted residual) | 1.191 ± 0.051 | 0.955 ± 0.031 | 0.969 ± 0.017 |
| C2_TRUECENTRE (amendment) | 1.043 ± 0.018 | 0.875 ± 0.014 | 0.842 ± 0.011 |
| ORACLE_METAL (Â + true departure) | 0.991 | 0.821 | 0.789 |
| ORACLE_LEVEL (true cell mean + B̂) | **0.221** | **0.213** | **0.205** |

Offset / shape, `unseen_chemotype`: MONO_ET 0.874 / 0.507; C2_TRUECENTRE 0.881 / 0.505;
ORACLE_METAL 0.883 / 0.365; ORACLE_LEVEL 0.073 / 0.185. The shape column again does not move between deployable models *within* a regime (forests 0.50–0.51
under `unseen_chemotype`, 0.44–0.45 under `unseen_ligand`, 0.46–0.50 under `unseen_series`; every
ridge 0.60 everywhere).

### 2.3 What the run says

**The level carries most of the removable error.** On multi-metal cells, against the deployable
`MONO_ET` reference, the level oracle removes **0.497** log units [BCa +0.396, +0.645] and the metal
oracle **0.105** [+0.061, +0.160]; the difference +0.392 [+0.273, +0.559] is CI-clean, 5/5 seeds,
and holds in every novelty bin (§1.2). Two constructions inflate the headline size and were
measured rather than argued:

* the true cell mean handed to `ORACLE_LEVEL` **includes the scored row's own label** (1/k of it;
  mean 1/k = 0.155 here). With a leave-one-out cell mean the difference falls to **+0.324**
  [+0.188, +0.501] — still CI-clean;
* pairing that LOO level with the *cross-fitted* Stage B — an inconsistent build, since that Stage B
  is the defective one of §1.3 — gives +0.104 [−0.038, +0.267], not CI-clean. So the **direction is
  robust and the magnitude is not**: the honest range is +0.10…+0.39 depending on construction, and
  what this licenses is "the level carries most of the removable error", not a precise factor.

`ORACLE_LEVEL` on *all* rows (0.22) is trivial for singleton cells, which is why C1 is scored on
multi-metal cells only. And note what the contrast is: it compares two components whose variance
shares are 92 % / 8 %, so it locates *where the removable error sits* — it does not show which
component degrades faster when the ligand is new.

**Structure does not substitute for coverage — it can cost.** The pre-registered two-stage model is
0.14 log units *worse* than the monolith on new chemotypes (and 0.10–0.16 worse in the other two
regimes), for the reason §1.3 gives: a residual cross-fitted under a chemotype hold-out is mostly
level error, and Stage B learns it from in-sample ligand features. The variant that centres on the
true cell mean recovers parity (Δ +0.004, interval spanning zero) and nothing more. The hierarchical
ridge family is a different, lower-capacity family (compact descriptors, no `LIG2D_EXT`), so its
0.10 gap to the forests is a capacity difference, not a verdict on structure; within that family,
adding ligand intercepts (C1) or pair-difference rows (C3) changes nothing anywhere.

**A ligand's level has a substantial series-specific part — but the first draft over-read it.**
Three corrections, all found by review and all measured:

1. `unseen_series` does **not** keep the ligand in training for most macro units: 97 of 152 ligands
   have exactly one series, so holding it out removes the ligand. On the rows where the intercept is
   live, C1 *helps* by +0.036 (5/5 seeds), not −0.011 — a wide interval, so the honest verdict is
   "inconclusive, point estimate positive", not "buys nothing".
2. The contrast is not a clean intercepts-on/off ablation: the arms chose different `λ_fixed` in 18
   of 25 `unseen_series` folds, and `λ_ligand` sat on the grid's lower edge (0.1) in 16 of 25 — the
   intercept arm never reached its own optimum.
3. The variance split does not say the level is "not a ligand constant": between-ligand 51 % vs
   between-series-within-ligand 12 % (sd of ligand means 1.45 vs mean within-ligand sd of series
   means 0.72). The fair statement is **a ligand constant plus a series term about half its size**,
   and part of that term is condition variation the model already has as features.

The design implication survives in weaker form — a level model should carry a (ligand, series) term
— and the experiment that would size it (intercepts on/off at fixed `λ_fixed`, wider `λ_ligand`
grid, scored only on live-intercept rows) was not run.

**What the level models do for the pair task.** Pooled over the five chemotype folds (14,173 pairs /
90 ligands per seed — the whole expanded pair cohort, held out by chemotype), level-derived pair
predictions score macro MAE 0.506 (MONO_ET) / 0.496 (C2_TRUECENTRE) against the pair-label-mean
null's 0.699 and a constant-zero null's 0.612; sign accuracy 0.761 vs the null's 0.733. That
row-pooled aggregation differs from `pair_metrics.csv`, which averages per fold (0.519 / 0.504 /
0.699, sign 0.670); the sign-accuracy margin is carried by the large DGA fold (0.85 there, 0.54–0.73
elsewhere), so read it as indicative. Transitivity and antisymmetry are
exact by construction and measured as such. Nothing in this block carries a verdict; it is the
first pair number on the full pair cohort under a chemotype hold-out, and it says that the level
route is a competitive way to do the original task.

### 2.4 What would break it

* A metal oracle that removed as much as the level oracle on multi-metal cells — it removes a third.
* A C2 variant that beat MONO_ET materially on new chemistry — neither does.
* A C1 that beat plain ridge on a new chemotype — it does not, and should not.
* The ridge family being compared like-for-like with the forests — it is not, and the text says so.

---

## 3. Experiment F — the full run

`runs/gen6_expF_3seed/`: 3 split seeds × 5 chemotype folds × 6 policies × 2 acquisition replicates,
30 acquisitions of 3 rows each from a 10-ligand single-chemotype start (~2,900 rows; in the fold
that holds out the diglycolamides the start degenerates to 1–2 ligands and is kept, disclosed), 200 trees, scored on the fixed held-out
chemotypes at 10 checkpoints; plus a `b = all` sensitivity for `maxmin` and `random`. Simulation
1,729 s. No revealed row is ever a test row (recorded in validation). The label-free property holds
by construction (the pool frames handed to a policy have the target column removed, and no
feature is target-derived); the permutation test and run-time check confirm that construction
rather than constituting independent evidence of it — a target-derived feature added later would
pass both silently.

### 3.1 Verdicts

| H | statement | verdict | the decisive numbers (hard chemistry = test rows below Tanimoto 0.4 to the *start* cohort, a fixed subset) |
|---|---|---|---|
| **F1** | max-min diversity beats random on hard chemistry | **PASS** | random − maxmin at k = 5/16/20/25/30: +0.180 / +0.112 / +0.124 / +0.198 / +0.134; percentile CI low > 0 at 6 of 7 checkpoints from 5 onward (BCa at 5 of 7; needed 4). Fold-seed-paired: +0.21…+0.28, BCa-clean at all 7 |
| **F2** | acquiring the most *level-uncertain* ligand beats random | **INCONCLUSIVE** (not supported) | offset MAE, random − offset_uncertainty: −0.110 / −0.141 at k = 5/8 (worse than random), +0.086 / +0.093 at k = 25/30 with BCa spanning zero; 2 of 7 checkpoints; against `maxmin` it is worse at every checkpoint to k = 25 |
| **F3** | buying more of the same chemistry is worse than random | **PASS** | random − same_chemotype_first on hard MAE is **negative at all 10 checkpoints**, −0.12 at k = 1 to −0.81 at k = 25, BCa-clean throughout. Disclosure: the protocol table wrote this contrast with the opposite sign to its own gloss ("i.e. random is better"); the runner scored the gloss and printed the arithmetic |
| **F4** | three measurements of a distant ligand beat three more of a known one | **SKIPPED** | as pre-declared (needs `--f4-cap-start-rows`; Experiment B already measured depth vs breadth). The `b = all` sensitivity is a different question (depth on the *acquired* ligand) and does not substitute |

### 3.2 The curves (macro MAE over ECFP clusters on the fixed test chemotypes; mean of 15 fold-seeds × 2 replicates)

| acquisitions | random | maxmin | uncertainty | diversity × uncertainty | offset-uncertainty | same-chemotype-first |
|---|---|---|---|---|---|---|
| 0 (start only) | 1.853 | 1.853 | 1.853 | 1.853 | 1.853 | 1.853 |
| 5 | 1.508 | 1.310 | 1.655 | 1.415 | 1.641 | 1.888 |
| 12 | 1.392 | 1.277 | 1.363 | 1.251 | 1.418 | 1.902 |
| 20 | 1.334 | 1.208 | 1.223 | **1.105** | 1.245 | 1.950 |
| 30 | 1.290 | 1.190 | 1.132 | **1.096** | 1.167 | 1.901 |

On the hard subset the ordering is the same and the gaps larger: at 30 acquisitions random 1.347,
maxmin 1.137, diversity × uncertainty **1.040**, same-chemotype-first **2.090** — *above its own
start* (1.975). Offset MAE at 30: random 1.095, maxmin 1.039, diversity × uncertainty **0.915**,
same-chemotype-first 1.793. Shape MAE: 0.53–0.58 for every policy at every checkpoint — unmoved
here too.

What each policy bought, mean over runs: maxmin and diversity × uncertainty acquired ligands at
median Tanimoto **0.24–0.26** to everything already held and reached ~30 distinct chemotypes in 30
steps; random 0.52 and **21.6** chemotypes (the first draft copied 28, the wrong column);
same-chemotype-first 0.77 and **5** chemotypes. The
chemistry-only rule does exactly what it says.

### 3.3 Two weightings, one read honestly

The runner's per-checkpoint bootstrap pools every fold-seed's test rows into one frame before
resampling chemotype blocks. That is a row-weighted view, and in it the three fold-seeds that hold
out the diglycolamide chemotype (3,700–4,000 test rows each) carry **72–81 % of every interval**.
The complementary view — one unit = one fold-seed, differences paired within it, BCa over the 15
fold-seeds (`scripts/foldseed_paired_intervals.py`, written after the first review of this run) —
is reported beside it, and a claim is called established only where the two agree.

| contrast (hard chemistry, macro MAE) | row-pooled Δ at k = 16 / 30 | fold-seed-paired Δ at k = 16 / 30 | agree? |
|---|---|---|---|
| max-min − random | +0.112 / +0.134, BCa-clean | **+0.262 / +0.210**, BCa-clean, 13 of 15 fold-seeds | yes — stronger under pairing |
| same-chemotype-first − random | −0.54 / −0.64 | −0.67 / −0.74, 0 of 15 positive | yes |
| novelty × uncertainty − random | +0.099 / +0.171 | +0.291 / +0.307, 12–15 of 15 | yes |
| novelty × uncertainty − max-min | −0.013 / +0.037, never clean | +0.030 / +0.097, clean only at k = 30 | **a tie on hard chemistry** |
| novelty × uncertainty − max-min, overall | clean at 2 of 10 checkpoints | clean at k = 16, 20, 25, 30 (+0.07…+0.10) | modestly ahead late |
| novelty × uncertainty − max-min, k = 5 | −0.08 (BCa-clean loss) | −0.14 (BCa-clean loss) | worse early |

**Which distant ligand to measure next: the farthest one.** Max-min Tanimoto acquisition beats
random on hard chemistry at every checkpoint from 5 onward under both weightings — by 0.11–0.20
log units row-pooled and 0.21–0.28 fold-seed-paired. This is a rule a chemist can follow with
nothing but fingerprints, and it is the only acquisition result here that is clean in every view.

**The model's own uncertainty is not an acquisition signal on its own, and the novelty-weighted
version is not a clear improvement on chemistry alone.** The two pure uncertainty policies are
*worse* than random for the first ~12 acquisitions and catch up only late (F2 not supported). The
rank-product policy — novelty × uncertainty — is worse than max-min for the first five acquisitions
(it starts by buying ligands the start-cohort model is confused about for reasons other than
novelty), **ties max-min on hard chemistry at every checkpoint but the last**, and pulls ahead on
the overall and offset endpoints from k ≈ 16 (+0.07…+0.13 fold-seed-paired, BCa-clean; row-pooled
clean at only 2 of 10). It was a pre-registered *policy* without a pre-registered *hypothesis*;
the honest description is "tracks max-min after a bad start and adds a little late on the
well-covered chemistry, nothing on the hard chemistry." The first draft of this document called it
"the best rule of all"; it is not.

**Buying another analogue of what you have buys nothing — and under the study's weighting it looks
worse than that.** Same-chemotype-first ends 0.64–0.81 log units behind random on hard chemistry
(0.67–0.84 fold-seed-paired, 0 of 15 fold-seeds positive from k = 8), and its hard-chemistry error
*rises* through the run (1.98 → 2.09–2.21). The rise is an artefact and was measured as such:
replaying the same acquisition orders on three fold-seeds with **no** sample weights instead of the
gen5 equal-cluster weights, same-chemotype-first is *flat* (hard MAE 2.05 → 1.98 over 30
acquisitions) while max-min still falls 2.05 → 1.34 and random 2.05 → 1.38. The honest statement
is the weaker one: another DGA analogue adds *nothing* about chemistry outside its chemotype,
whereas a random or a max-min ligand adds 0.6–0.7 log units.

**The 90 acquired rows are 3 % of the data and 83 % of the training weight.** Under the gen5
equal-cluster rule every newly acquired ligand is a new ECFP cluster with a full vote, so after 30
acquisitions the 90 new rows carry ~83 % of the loss (training clusters 5 → 35). That is what makes
"1.85 → 1.10 from 90 rows" unsurprising, and it is the same mechanism that turns same-chemotype-
first's "nothing" into "harm". The comparison *between* policies shares the rule; the effect *size*
does not — which is why the whole experiment was rerun with row weighting
(`runs/gen6_expF_3seed_rowweight/`, four policies, same seeds, folds and replicates):

| row weighting, hard chemistry, fold-seed-paired | k = 16 | k = 30 |
|---|---|---|
| max-min − random | **+0.221** [+0.123, +0.403], 12/15 | **+0.178** [+0.097, +0.290], 12/15 |
| novelty × uncertainty − random | +0.330 [+0.183, +0.518], 13/15 | +0.286 [+0.198, +0.412], 14/15 |
| novelty × uncertainty − max-min | +0.109 [+0.033, +0.174], 12/15 | +0.108 [+0.054, +0.180], 12/15 |
| same-chemotype-first − random | −0.506 [−0.636, −0.394], 0/15 | −0.650 [−0.866, −0.502], 0/15 |

Without the equal-cluster weighting every ordering survives and the *gaps* are of the same size —
the weighting sets the absolute level of the curves (start 1.955 vs 1.853), not the comparisons.
Same-chemotype-first is now flat (hard 2.10 → 1.99) rather than rising, which is the weighting
artefact named above. One thing changes: under row weighting novelty × uncertainty beats max-min on
hard chemistry too, CI-clean at k = 16 and 30 (12 of 15 fold-seeds) — whereas under the study's
primary weighting it only tied. That promotes it from "ties max-min" to "better than max-min in one
of two weightings, descriptive" — not to "established".

**No measurable gain from revealing more than three rows of an acquired ligand — but the first draft
quoted that as a number, and it is not one.** The `b = all` sensitivity reveals every row of each
acquired ligand (+390 rows at 30 acquisitions instead of +90). Its 0.015 advantage for `maxmin`
(1.175 vs 1.190) and −0.019 for `random` are **inside the run-to-run spread** (maxmin replicate
vs replicate: mean |Δ| 0.038, sd 0.049; the `@ball` arm is not even paired on the ligands acquired —
its RNG stream differs, so the sequence differs in 13 of 15 fold-seeds — and no interval exists in
any artifact). A properly paired replay by the reviewers (same ligand order, only the budget
changed) put `b = all` slightly *worse*. The defensible statement is "three rows per new ligand
was not distinguishable from all of its rows at this scale" — consistent with the level being the
information, not evidence of a 0.015 effect. It also does not answer F4, which asks about depth on
a *known* ligand; that comparator was not run.

### 3.4 Caveats the run records

* Absolute error on the start cohort alone is 1.85 (hard 1.98) — a ten-ligand DGA start is a bad
  model of everything else, and every curve is measured against that.
* Per-fold spread is large (sd 0.18–0.25 across fold-seeds); the paired intervals are what carry
  the comparisons, not the means.
* The degenerate folds are included: in the three fold-seeds that hold out the diglycolamide
  chemotype the largest remaining chemotype has 1–2 ligands (starts of 140–244 rows). Excluding
  them (12 of 15 fold-seeds) changes no ordering and widens the gaps slightly — hard MAE at 30
  acquisitions: random 1.337, maxmin 1.110, uncertainty 1.049, offset-uncertainty 1.040,
  diversity × uncertainty **0.970**, same-chemotype-first 2.147.
* 200 trees, not 400, to fit ~3,600 fits in half an hour; a relative comparison is unaffected.
* F4 is skipped by pre-declaration, not evaded: the `b = all` sensitivity answers the sharper
  question, and Experiment B already answered the original one.

---

## 4. What this does and does not license

**It licenses**

* designing the next level model around `α_{ligand, series}` + metal departure, with the series
  identity (diluent, acid, and the recovered upstream solvent/modifier columns) as a first-class
  input — the level is where 80–95 % of the removable error is, and it is series-specific;
* deploying a chemistry-only acquisition rule today: max-min Tanimoto to the held set, three
  measurements per new ligand, then the next one — the one acquisition result that is clean under
  every weighting and pairing; novelty × uncertainty is not established as better than it;
* stopping work on metal-response modelling for new chemistry: it is 8 % of the variance and the
  smaller error everywhere.

**It does not license**

* a claim that two-stage structure helps: the pre-registered variant is 0.14 worse than the
  monolith and the amended one is indistinguishable (and was chosen after seeing a seed);
* a precise size for the level-vs-metal gap: it ranges +0.10…+0.39 across defensible oracle builds;
* the claim that the level is what *fails to transfer*: the run locates where the error sits;
* the claim that a ligand's level is "not a ligand constant": it is a ligand constant plus a series
  term about half its size;
* reading the hierarchical-ridge numbers as a verdict on structure: that family is 0.10 behind the
  forests because it is low-capacity by design (compact descriptors, no `LIG2D_EXT`);
* any statement about the shape term — unmoved in both experiments, as in Phase 1;
* calling novelty × uncertainty the better rule: it was a pre-registered policy without a
  pre-registered hypothesis, it is worse than max-min for the first five acquisitions, ties it on
  hard chemistry, and is ahead only late on well-covered chemistry;
* quoting the Experiment F effect sizes without their weighting: under the equal-cluster rule the
  90 acquired rows carry 83 % of the training weight; the row-weighted rerun is the reference for
  magnitudes;
* trusting the magnitude of the same-chemotype-first penalty or the claim that it *grows*: the growth
  is the equal-cluster weighting rewarding near-duplicate ligands (measured: flat without weights);
  what survives is "buys nothing", and the max-min-over-random gap is also smaller without weights;
* an absolute-accuracy claim anywhere: the best policy ends at macro MAE 1.10 from a 1.85 start.

**What would falsify these**: a metal oracle that removed as much as the level oracle; a structured
model that beat the monolith CI-clean on new chemotypes; a hierarchical ridge that beat plain ridge
on a new chemotype; max-min acquisition indistinguishable from random; or the `b = all` curves
separating from the `b = 3` curves by more than the run-to-run spread. None occurred.

**Amendments recorded rather than silently made**: `C2_TRUECENTRE` (protocol; chosen after a
one-seed preview, labelled exploratory); C3's verdict label (report; a seed-clause direction defect
in the runner, numbers unchanged); `pair_consistency`'s antisymmetry residual (module; was a
hard-coded zero, now measured — the runner had already measured it independently).
