# Gen16 — working brief

**Read this file completely before running anything.  It is the task, the rules and the state of
the programme.  Budget: 12–16 hours of autonomous work with a fleet of parallel agents.**

You are continuing a four-generation research programme on predicting lanthanide separation
factors.  Gen13, gen14 and gen15 are finished and locked; their reports are in
`gen13_separation/DECISION_REPORT.md`, `gen14_direction/GEN14_REPORT.md` and
`gen15_curve/GEN15_REPORT.md`.  Gen16 is you.

**Read gen15 §1a, §7 and §8 before anything else.**  They landed after the first draft of this brief
and they changed it: §1a corrects every own-cell ceiling downward, §7 localises the one live
estimator lead, and §8 closes eight more axes (chemical language-model embeddings, 85 similarity
kernels, modern tabular learners, aqueous logK transfer, within-publication conditions, label
de-noising) and reports that the model cannot rank candidate ligands at all.  Their code is under
`gen15_curve/exp/<slug>/`.  This brief has been revised against them; if you find a further
contradiction, the locked report wins and you say so in your own report.

The user commissioning this brief explicitly asked for multi-agent parallel execution.  Treat this
file as that opt-in: you may use the Workflow tool and large fan-outs of subagents.

---

## 0. The one thing that matters most

This programme's value so far is that **it does not report results it cannot defend.**  Three times
a strong-looking result was killed by its own control:

| what looked true | what killed it |
|---|---|
| "experimental conditions beat molecular structure" | the 64 condition columns identify the publication with 94 % 1-NN accuracy |
| "the model gains +0.122 over the corpus mean curve" | the mean curve is *worse* than predicting no separation at all |
| "xTB energies correlate with selectivity at ρ = +0.644" | only 1 of 41 metal series has constant inner-sphere composition |

You are being asked to find a **positive, valid, significant** result.  Those three words are in
tension, and the tension is the task.  A fleet of agents searching one small cohort will manufacture
false positives unless the search is disciplined.  **The discipline is not optional overhead; it is
the deliverable.**  A well-supported null, reported honestly with its mechanism, is a success.  A
significant-looking positive that dies under refutation is a failure even if you found it first.

Never weaken a control, drop a design, change a baseline, or re-scope an endpoint *after* seeing a
number.  If you are tempted to, that is the finding: write down what you were tempted to do and why.

---

## 1. The validity contract (non-negotiable)

Every one of these is mechanically checkable.  A result that violates any of them does not go in the
report.

1. **Pre-register before you look.**  Before any lead is run, write `PRE_REGISTRATION.md` in this
   directory: the question, the arms, the primary endpoint, the decision rule, the stopping rule,
   and the exact list of contrasts you will report.  Commit it.  Record its SHA-256 in the file
   itself and in the final report.  Anything not in it is **exploratory** and must be labelled so in
   every table.
2. **Five designs, always, and the full table.**  Every candidate is scored under all five hold-out
   designs — A (exact extractant), B (chemotype hold-out), BR (random-cell control), BQ
   (random-publication control), BP (publication-masked).  Report all five.  Never report a subset.
   A candidate is supported only if the sign is consistent across all five.  This rule exists
   because gen13's amplitude-only arm scored +0.013 under BP and died under the other four; which
   design would kill it was not predictable in advance.
3. **BP selects the deployed model.**  Design B must never select anything.  Any number quoted for
   "new chemistry from a new group" is the BP number.
4. **`FLAT` is the floor.**  `FLAT` means predict no separation at all.  It scores **0.589** under
   BP.  The corpus mean curve scores 0.622 and is *worse than doing nothing*; it is not a baseline
   and must not be used as one.  Every headline gain is quoted against `FLAT`.
5. **Also quote the increment over the cheapest sensible competitor**, not only over a constant.
   For direction, that is the 13-column gen6 donor census.  For the measured mode, that is
   `NAIVE_LINE` (a straight line through the measured pairs, no corpus).  For any new
   representation, it is the existing lean block set.  Both numbers are true; only the second says
   whether the new thing earned its place.
6. **Matched null for every positive.**  Any block or descriptor that helps must be re-run as a
   width-matched shuffled version.  Report `real − shuffled`, not only `real − nothing`.
7. **Chemotype-blocked paired bootstrap, 5 seeds, LOCO stability.**  Use gen13's frozen bootstrap.
   The full decision rule (call it P1) is: point estimate above the margin, 95 % CI excludes zero,
   5/5 seeds same sign, and no single held-out chemotype flips the sign.
8. **Count your comparisons.**  Record the total number of arms and contrasts your fleet evaluated,
   in the report, as a number.  With a large fan-out this will be in the hundreds.  Apply
   Benjamini–Hochberg across the exploratory family and report both raw and adjusted p-values.  A
   nominal p = 0.03 out of 200 arms is noise and you will say so.
9. **Confirmation on fresh seeds, run exactly once.**  See §6.  This is the strongest protection you
   have and it only works if you do not peek.
10. **Reproduce the anchors before you trust anything.**  See §3.  If an anchor does not reproduce,
    stop and diagnose; do not proceed with a bench that has drifted.
11. **Every oracle you compute must be leave-pair-out.**  An oracle fitted to the rows it is scored
    on is not a ceiling; gen15 §1a measured that this inflated the two-parameter ceiling from 0.274
    to 0.181.  If you report a headroom, report it honestly or do not report it.

---

## 2. What the programme has established

Frozen cohort fingerprint `4c3c6628ea0be949`.

| | |
|---|---|
| cells (extractant × publication × exact 64-condition key, ≥ 2 lanthanides) | 521 |
| extractants | 90 |
| chemotypes | 45 |
| **Kish effective sample size** | **11.7 chemotypes** |
| pairwise `log SF` observations per seed | 14 150 |
| diglycolamide chemotype | 375 cells, 23 extractants |
| replicate disagreement, median per (cell, metal) | 0.30 log D |
| half-vs-half replicate split, pairwise MAE | 0.60 |

**You are working with roughly twelve independent units.**  Every design decision follows from
that.  It is why a linear model beats a forest here, why seven different priors for one scalar all
land within 0.007 of a constant, and why any method with many free parameters will look good under
design B and collapse under BP.

The zero-shot ladder under BP (from `gen15_curve/GEN15_REPORT.md` §1), extractant-macro MAE of
predicted `log SF`:

| arm | BP | what it is |
|---|---|---|
| `O_BOTH` | 0.181 in sample, **0.274 leave-pair-out** | the cell's own two coefficients; see the correction below |
| `O_AMP` | 0.322 | the cell's own radius coefficient |
| `O_CURV` | 0.427 | gen14's amplitude with the cell's own curvature |
| `O_SIGN` | 0.486 | a perfect direction call, constant magnitude |
| `G14` | **0.500** | the deployed model |
| `G13_FULL` | 0.552 | gen13's 209-column regression |
| `FLAT` | **0.589** | predict no separation at all |
| `MEAN_CURVE` | 0.622 | not a baseline; worse than nothing |

Honest four-generation zero-shot gain: `G14 − FLAT` = **+0.0884**, CI [+0.0010, +0.1473], p = 0.047.
Real, significant, design-invariant, and small.

**Ceiling correction (gen15 §1a), load-bearing.**  Every own-cell oracle above is fitted to the same
log D values it is then scored against, so part of its advantage is the coefficient absorbing the
noise of the very pair being scored.  Refitting each cell's coefficients *without* the two metals of
the scored pair gives the honest numbers:

| arm | in sample | leave-pair-out | was noise |
|---|---|---|---|
| own `a` and `b` | 0.1811 | **0.2736** | 0.093 |
| own `a`, constant `b` | 0.3191 | 0.3370 | 0.018 |
| true sign, own `b` | 0.3771 | 0.4245 | 0.047 |
| true sign, constant `b` | 0.4724 | 0.4724 | — |

**The two-parameter representation ceiling is ≈ 0.27, not 0.18.**  Read every own-cell oracle gap in
the locked reports as roughly half again too large, and the correction is largest exactly where the
fitted quantity is least determined, which is the curvature.  The labels themselves are clean
(reliability 0.978 for `a`, 0.936 for `b`), so this is not a data problem: an in-sample oracle is
simply not a ceiling.  Source: `gen15_curve/exp/labelerr/results/s4b_oracle_honesty.csv`.

The measured mode under BP, common scoring set, all five designs (gen15 §5):

| arm | B | BP |
|---|---|---|
| `FLAT` | 0.4992 | 0.5025 |
| `G14`, k = 0 | 0.4147 | 0.4242 |
| widest dZ, k = 1 | 0.2172 | 0.2261 |
| **D-optimal, k = 3** | **0.1633** | **0.1700** |
| `O_BOTH`, D-optimal k = 3 | 0.1428 | 0.1471 |

Strong-pair sign accuracy at D-optimal k = 3: 0.954 under BP, 0.958–0.961 elsewhere.
`G14 − NAIVE_LINE @ dopt k3` = +0.062 to +0.065, p < 1e-4, passes in all five designs.
`widest − random @ k1` = +0.099 to +0.103: **which pair you measure matters more than any estimator
choice in the programme.**

Established negatives, do not re-litigate: the 45-column 3D "response" block; the external aqueous
logK prior, in both directions; ECFP and the 206-column RDKit block (both *dilute* the signal); gain
calibration; tetrad basis terms; target reparametrisations; 30 direction classifiers; **twelve**
magnitude priors, all within ~0.007 of a constant; curvature predicted from 2D by any of {sign
classifier, direction-conditional constants, 3/4/6-prototype alphabet}; **pretrained chemical
language-model embeddings** (ChemBERTa-77M, MoLFormer-XL: null for magnitude and curvature, and
*actively harmful* for direction, 0.569 macro accuracy against TOPO39's 0.821 and an always-heavy
floor of 0.559); **85 similarity-kernel arms** (see the structural trap in §4); **modern tabular
learners** (TabPFN v2, CatBoost, monotone XGBoost, isotonic bite fit, spline GAM, a 4602-term
symbolic search); **label de-noising** (labels are clean, reliability 0.978 for `a`, every
correction null or harmful); and **within-publication conditions** via the Frisch–Waugh
transformation (the acid association is real, within-publication Spearman +0.356 with |a|, and worth
+0.009 on a mean |a| of 0.295, with inner LOPO cross-validation shrinking the coefficient to exactly
zero in 90 % of folds).

A perfect direction call would now buy only +0.014.  **The direction bit is spent, and the estimator
axis is closed.**  What is not closed: the physics descriptor (L1), the covariance (L5), and the
size of the corpus (L4, L6).

---

## 3. Anchors you must reproduce first

Before any new science, one agent reproduces these exactly and reports the digits:

| anchor | value | source |
|---|---|---|
| gen13 locked stage-3 headline, `G13_ET_TOPO39` under BP | `0.7683085207475452` | gen14 reproduced it to the last digit |
| gen14 deployed model under BP | `0.5001` | gen15 reproduced it to four decimals |
| `FLAT` under BP | `0.589` | gen15 §1 |

Import the frozen machinery, never reimplement it: `gen13_separation/gen13sep/splits.py`,
`gen14_direction/gen14/dirbench.py`, `gen15_curve/gen15/valuebench.py`,
`gen15_curve/gen15/fewshot.py`.  Any new arm is a function `arm(ctx) -> (n_test, 2)` scored by
`valuebench.py`.  If you find yourself writing your own metric, splitter or bootstrap, stop: you are
about to break comparability with three generations of results.

---

## 4. Known traps in this repository

These have each cost a previous generation real time.  Check them; do not rediscover them.

- **`hash()` is salted per process.**  It silently broke cross-run pairing in gen8.  Any split
  reproducibility must be tested in a *subprocess*, not just in-session.
- **Do not install TabPFN.**  It pins `sklearn<1.7` and `pandas<3`, shifts every result by ~0.0014
  and breaks comparability with every earlier suite.  If any dependency you add moves an anchor,
  revert it.
- **Macro vs pooled.**  One extractant dominates the pair corpus; pooled and extractant-macro
  metrics have ranked arms in opposite orders.  Extractant-macro is the metric.
- **Seed-pooled centring contaminated gen11's published level/shape numbers.**  Centre within fold
  and seed, never across.
- **The `min_rows` cohort filter is the real ceiling.**  In gen5 it discarded 37 of 47 chemically
  novel extractants.  Audit before you accept a cohort as given.
- **HistGradientBoosting crashes on all-NaN columns** under chemotype hold-out; the
  `DropAllNaNColumns` transformer already exists in `src/lanthanide_separation/levels.py`.  Reuse
  it; do not write a second one.
- **Uppercase `O`/`C` in SMARTS is aliphatic-only.**  Four descriptor bugs in gen12.2 came from
  this; one of them moved a primary endpoint from p = 0.092 to p = 0.020.
- **Kernel and similarity methods are structurally doomed here.**  The chemotype hold-out is
  single-linkage Tanimoto 0.7 on the very ECFP bits a similarity kernel uses, so the kernel
  extrapolates on 100 % of held-out cells *by construction*.  Gen15 ran 85 kernel arms to establish
  this.  Any new similarity-based method inherits the defect unless its representation is not the
  one the fold plan is defined on.
- **Inner-sphere composition changes along the metal series** (see L1).  Any quantity built from a
  total energy across metals is a step function unless you handle this.

---

## 5. The leads

Six leads, ranked by (expected value × feasibility) **as revised after gen15 §7 and §8 landed**.
L1 and L4 are now the substance.  L2 is gated on a headroom recomputation that may close it within
an hour.  L3 has shrunk to three narrow questions, because its obvious form has already been run and
the answer is negative.  L5 is reframed around covariance and calibration rather than point
accuracy.  L6 is support.  Run them in parallel (§7).  Each lead gets its own pre-registered
section, its own five-design table, and its own refuters.

### L1 — Rebuild the xTB descriptor with a proper thermodynamic cycle

**This is the only live hypothesis about new physics in the programme.**  It is also the one that
died most spectacularly, and it died from a *bookkeeping* error, not from chemistry.

Facts.  `dataset with 3D structures/accepted_geometries.csv` holds 1155 QC-accepted GFN2-xTB
lanthanide-complex geometries.  Gen15 built a two-way (ligand × metal) additive fit of
`complex_total_energy_eV` and got the strongest correlation with observed amplitude in the
programme's history:

| construction | n extractants | Spearman with amplitude | p |
|---|---|---|---|
| naive, complete 14-metal series | 39 | **+0.644** | < 1e-4 |
| contrasts inside constant-composition blocks only | 70 | +0.195 | 0.11 |
| whole panel, element counts as covariates | 71 | +0.216 | 0.07 |

Why it died: the builder chose a different inner-sphere composition for different metals of the same
ligand, so the "slope in ionic radius" is partly a step function in where the recipe changed.  The
composition columns are `coreCN`, `n_ligs`, `inner_sphere_anion`, `fill_ligand`, `n_fill`.

An audit run for this brief (reproduce it, do not take it on trust):

| | |
|---|---|
| extractants with accepted geometries | 177 |
| of those, constant composition across their metals | 68 |
| extractants with all 14 metals | 42 |
| **extractants retaining ≥ 8 metals if you keep only their single most common composition** | **6** |

So **filtering to constant composition is not available** — it leaves six compounds.  And it is not
even correct: coordination number genuinely falls across the lanthanide series, and forcing it
constant would discard real chemistry.

**The hypothesis.**  Compute a proper formation energy against explicit reference species instead of
a raw total energy:

```
dE(ligand, metal) = E(complex)
                  - E(metal ion)
                  - n_ligs  * E(free ligand)
                  - n_fill  * E(fill ligand)
                  - E(inner_sphere_anion)
```

with every reference term at the same level of theory.  Composition jumps then cancel **exactly**,
rather than approximately through a regression on element counts (which is what the +0.216 row
above already is, and it is nearly significant).  Then refit the two-way model and recompute the
correlation with amplitude and with curvature.

**Cost is low.**  This does not need the 1155 complexes recomputed.  It needs single-point energies
for a few dozen reference species: the free ligands, the fill ligands, the anions, and the bare
metal ions.  **`xtb` was verified as NOT on PATH when this brief was written**, so plan for the
cluster route from the start: build the reference-species input set locally, verify it is complete
against every distinct `(fill_ligand, n_fill, inner_sphere_anion, n_ligs)` combination in
`accepted_geometries.csv`, and hand the user a ready submit command with expected runtime and
memory.  Cluster submission is the user's, not yours.  Re-check `which xtb` first in case the
environment has changed.

While that job is queued, do not idle: the *bookkeeping half* of L1 is testable immediately.  Fit
the two-way model with explicit per-species indicator terms for composition (one coefficient per
distinct fill ligand and anion, entering with its actual count) on the energies you already have.
That is strictly stronger than gen15's element-count covariates, which already reached +0.216 at
p = 0.07, and it tells you before any new computation whether the cycle is likely to rescue the
correlation.

Scaffolding already exists in `gen15_curve/exp/phys3d/`: `build_block.py`, `geometry_long.parquet`,
`extractant_targets.parquet`, `trap_check.py`, `robust_stats.py`.  Extend it; do not start over.

Gen15 §8's `phys3d` arm independently confirmed that the 28-column xTB *geometry* block is null
(best honest fold-refit 0.5013, worse than gen14) and then stated, in its own words, the caveat that
makes this lead live: **the reference jobs were never run, so binding, strain and frontier-orbital
energies are all NULL.**  The published verdict is "these xTB scalars are useless here", not "xTB is
useless here".  L1 is precisely the experiment that closes that gap.

**Decision rule, pre-register it.**  The honest 2D benchmark is `frac_donor_pairs_within_3` at
ρ = −0.53.  L1 is a *positive result* only if the cycle-corrected slope reaches |ρ| ≥ 0.40 with a
LOCO-stable sign **and** survives the confound control for the number of metals the experiment
measured (that confound is ρ = +0.49 with |amplitude|; gen15's partial correlation handles it).
It is a *strong* result if the corrected descriptor also improves BP macro MAE over the lean block
set under the full decision rule.  If it lands at |ρ| ≈ 0.2, say so plainly and close the lead
permanently — that is a valuable, publishable negative with a named mechanism.

Guardrails: report `n` for every correlation; a correlation computed on 39 of 82 extractants is not
the same claim as one on all of them.  Check that the reference energies are converged and that no
species is missing (a missing reference silently reintroduces the step function).

### L2 — Reparametrise the curve as a size-match, not a quadratic (gated)

**Read the ceiling correction in §2 first: this lead's premise shrank.**  In sample, giving each
cell its own curvature scores 0.427 against 0.500, which read as "curvature is the unclaimed half of
the error".  Under the leave-pair-out correction the comparable own-curvature oracle moves from
0.3771 to 0.4245, so about 0.047 of that apparent gap was the coefficient absorbing the noise of the
scored pair.  Gen15 states that the correction is largest exactly here, because curvature is the
least determined quantity in the model.

What survives the correction: curvature is uncorrelated with amplitude (Pearson 0.03), has chemotype
ICC 0.67, and 27 % of well-determined cells have a genuine interior extremum inside their measured
range.  The structure is real; the prize is smaller than the locked §2 suggests.

**Gate, and do this before anything else in L2.**  Recompute the curvature headroom honestly with
the leave-pair-out oracle (`exp/labelerr/results/s4b_oracle_honesty.csv` has the machinery).  If the
honest `O_CURV − G14` gap is under the pre-registered 0.02 margin, **close this lead, write one
paragraph saying so, and move those agents to L1 or L4.**  Only continue if the honest gap is worth
chasing.  This gate should take under an hour.

**The untried idea, if the gate opens.**  A ligand cavity with a preferred ionic radius produces
exactly a peaked curve.  Reparametrise the same quadratic from (slope, curvature) to (**preferred
radius** `r0`, **stiffness** `k`), where `r0 = −a / (2b)`.  The hypothesis is that `r0` is a
physically meaningful, chemotype-level quantity, far more predictable from bite-size descriptors
than `b` is; gen13 already found amplitude correlates −0.53 with the fraction of donor pairs within
three bonds, which is a bite-size proxy.

**Be careful, this is a ratio estimator.**  `r0` explodes as `b → 0`, and for 73 % of cells the
vertex lies outside the measured range where it is not identified at all.  Therefore:

- restrict to well-determined cells with a bounded, in-range vertex, and *state that subset's size*;
- report `r0`'s chemotype ICC against `b`'s 0.67; if it is not clearly higher, the lead is dead;
- prefer a bounded reparametrisation (fit `r0` directly by nonlinear least squares with a prior
  keeping it near the measured range) over dividing two fitted coefficients;
- the endpoint is still BP macro MAE through `valuebench.py`, not the ICC.  A better-behaved
  parameter that does not move the MAE is a null.

Predicting `sign(b)` conditioned on direction was tried and failed; predicting `r0` as a regression
was not.  Cross-check `r0` against L1's descriptor: a size-match parameter and a computed
interaction slope should agree if both are real, and that mutual check is worth more than either
alone.

### L3 — Screening: the obvious question is answered, and the answer is no

**Do not re-run the obvious version.**  Gen15 §8's `decision` arm asked exactly it, with a
permutation null behind every number, and the result is negative and mechanistic:

| question, design BP | model | baseline |
|---|---|---|
| which lanthanide of a pair enters the organic phase | **0.812** | 0.630 (always heavier) |
| which of these extractants best separates this pair, top-1 | 0.023 | 0.017 (random) |
| rank candidates with the laboratory held fixed | not better than random | |

The mechanism is localised: gen14 emits one bit per extractant, and that bit is **identical across
the candidate set in 70–85 % of real comparisons**, so the model has nothing to rank with.
Answering "which ligand" needs the magnitude, which is what four generations failed to predict.

The honest one-sentence state of the programme is therefore: **zero-shot, this model answers "which
way" and cannot answer "which ligand".**  What remains open is narrow, and worth about an hour each:

- **Decision value of the direction call alone.**  It is deployable at 0.812 against 0.630.  Given a
  target pair and a candidate set, how much measurement is avoided by discarding the wrong-direction
  half?  Report it as measurements saved, with a permutation null and a chemotype-blocked interval.
- **Ranking *within* a chemotype**, where the bit is constant and only the magnitude varies.  Gen15
  did not separate this case out, and it is where a magnitude signal would first appear if L1 finds
  one.  Pre-register it as conditional on L1.
- **Ranking at k = 1.**  Once one pair is measured the magnitude is largely known.  Ranking
  candidates that each have one measurement is a different and realistic procurement question, and
  it has never been scored.

Anything beyond these three is re-running a closed experiment.  If L1 fails, the second bullet is
moot and should be dropped rather than run.

### L4 — Corpus-level experimental design: which *ligands* to measure next

The D-optimal machinery in `gen15_curve/gen15/fewshot.py` chooses which metal pair to measure for a
given ligand.  Nobody has asked the outer question: **which ligands should be measured at all to
most improve the model?**

The binding constraint on this programme is 11.7 effective chemotypes.  No estimator fixes that.  A
principled answer to "measure these ten compounds next, and these three pairs of each" attacks the
actual bottleneck and is a deliverable a laboratory can act on.

Method sketch: use the fitted residual covariance and the chemotype structure to compute the
expected reduction in posterior variance over the ligand space from adding a candidate compound;
greedily select a batch; validate **retrospectively** by simulating corpus growth — hold out
chemotypes, add them back in model-chosen versus random versus maxmin order, and measure how fast BP
macro MAE falls.  Gen6 already found that maxmin and level-uncertainty beat random for a related
question, so there is a prior that this works.

The deliverable is a ranked, chemically annotated list plus the retrospective learning curves.  Be
explicit about what the simulation cannot tell you: it can only re-add chemotypes the corpus already
contains, so it measures ordering, not the value of genuinely new chemistry.

### L5 — Covariance shrinkage and honest intervals, not a better point estimate

**The estimator axis for point predictions is closed.**  Gen15 §8's `tabpfn` arm ran TabPFN v2,
CatBoost, monotone XGBoost, an isotonic bite fit, a spline GAM and a 4602-term exhaustive symbolic
search; twelve magnitude priors now sit within about 0.007 of a constant.  Do not add a thirteenth.

What is *not* closed is the covariance, and gen15 §7 localised the mechanism.  A measurement-
conditioned prototype mixture fails, but not because the shapes are wrong: the pre-check passes, one
measurement separates the components by a median 2.8 nats.  It fails because a component holding
30–120 cells cannot support 105 covariance parameters.  Giving every component the prototype **mean**
but the **pooled covariance** turns the loss into a gain:

| arm, design BP | k = 1 | k = 2 | k = 3 |
|---|---|---|---|
| `POOLED` (training-fold covariance) | 0.2381 | 0.2183 | 0.1922 |
| `MIX6meanPC` (component means, pooled covariance) | **0.2335** | **0.2136** | **0.1865** |

Gain +0.0046 / +0.0047 / +0.0070, p = 0.049 / 0.023 / < 1e-4, 67 of 84 extractants and 5/5 seeds at
k = 3, LOCO-sign-stable.  It is under the pre-registered 0.02 margin, and it is measured against the
training-fold route (`POOLED@k3` = 0.1922) rather than the deployed leave-chemotype-out one
(0.1700), so it does **not** pass P1 as it stands.  It is nonetheless the only place in this
programme where a structured estimator beat its pooled control at all.

**The lead.**  Build the hierarchical model with chemotype random effects on both coefficients and
properly shrunk covariances (Ledoit–Wolf, an inverse-Wishart prior, or low-rank plus diagonal), so
zero-shot and k-shot become one model at k = 0 and k > 0.  Then re-test the §7 gain against the
**deployed** baseline, which is the comparison that matters and the one gen15 did not run.

**Judge it on calibration as well as MAE**, and pre-register that endpoint: interval coverage and
sharpness in the zero-shot regime, where the programme currently has no honest intervals at all.
Given §8's finding that the only defensible zero-shot claim is a direction call, a calibrated
probability attached to that call is itself a deliverable, even if the MAE does not move.

### L6 — Cohort audit: are chemotypes being thrown away?

Gen13's cohort is 521 cells, 90 extractants, 45 chemotypes.  In gen5 a `min_rows` filter discarded
37 of 47 chemically novel extractants and was later identified as the programme's real ceiling.
Audit gen13's cohort construction the same way:

- how many extractants have ≥ 2 lanthanides measured but are excluded, and by which rule;
- what chemotypes those excluded compounds represent, and their nearest-neighbour Tanimoto to the
  kept cohort;
- what the Kish effective sample size would be if each filter were relaxed one at a time.

If relaxing a filter adds even five independent chemotypes, that is worth more than any model change
in this brief.  **Do not silently change the cohort** — the fingerprint `4c3c6628ea0be949` is what
makes gen13–gen15 comparable.  Any expanded cohort is a *separate, additional* set of runs, reported
alongside the frozen one, never replacing it.

---

## 6. Multiple comparisons and the confirmation protocol

You will run hundreds of arms.  Read this twice.

**Discovery phase** uses the five frozen seeds that gen13–gen15 used.  Everything in §5 happens
here.  Discovery results are *hypothesis-generating*, and every table from this phase is labelled
`DISCOVERY`.

**Confirmation phase** uses **five fresh seeds that no discovery agent has ever evaluated.**
Generate them at the start, from a recorded fixed rule, and write them into `PRE_REGISTRATION.md`.
Then have the orchestrator withhold them.  No lead agent may see or run them.

At the end of discovery, you freeze a list of **at most five** candidate claims.  Only those five go
to confirmation.  The confirmation run executes **once**.  Whatever it returns is the result.  If a
claim shrinks or reverses in confirmation, that is reported as the finding, and you do not go back
and try a sixth candidate.

Report, for every claim, in one table: discovery point estimate, confirmation point estimate, both
CIs, raw p, BH-adjusted p over the full exploratory family, number of arms in that family, seeds
agreeing, and LOCO stability.

**Adversarial refutation.**  Every claim that survives discovery gets **two independent refuter
agents**, each spawned without the other's output and each given a single instruction: *find the
control, confound or bookkeeping error that would make this result disappear.*  Gen13 ran this
pattern with three lenses and 41 findings before its locked run.  A claim that no refuter can dent,
on fresh seeds, under five designs, against the cheapest competitor, is a real result.  Nothing else
in this brief is.

Refuters should specifically check, for every claim: is the effect carried by the diglycolamides
alone?  Does it survive removing them?  Is it an artefact of `n` metals measured per extractant
(ρ = +0.49 with |amplitude|)?  Does it survive the publication mask?  Is the comparison arm matched
on training-set size?  Does the shuffled version of the block score the same?

---

## 7. Fleet plan and time budget

Suggested phasing.  Adjust the internals, keep the gates.

| phase | hours | what runs |
|---|---|---|
| 0 | 0–1 | one agent reproduces the three anchors; one agent runs the L6 cohort audit; one agent verifies the environment (no dependency has moved an anchor) |
| 1 | 1–2 | write and commit `PRE_REGISTRATION.md`, including the withheld confirmation seeds |
| 2 | 2–3 | **two gates run first and can retire work**: L2's leave-pair-out headroom recomputation, and L1's composition-indicator refit on the energies already in hand.  Either may close its lead before a single cluster job is queued. |
| 3 | 2–8 | L1, L4, L5 in parallel, each a lead agent with its own sub-fleet; L2 only if its gate opened; L3's three narrow questions as one small agent |
| 4 | 8–11 | two refuter agents per surviving claim, plus matched-null agents; refuters run blind to each other |
| 5 | 11–13 | confirmation run on the withheld seeds, executed once |
| 6 | 13–15 | synthesis: `DECISION_REPORT.md` in gen13's format; then a separate audit agent whose only job is to attack the report |
| 7 | 15–16 | tests, commit, handover: submit commands for anything needing the cluster |

Parallelism guidance: fan out **within** a lead (many arms, many seeds, many refuters) rather than
spawning many agents on the same question, which just multiplies the false-positive count without
adding information.  Independent leads run concurrently.  Refuters must not share context.

Checkpoint discipline: at the end of each phase, write a short status file into this directory with
what is confirmed, what is pending and what has been abandoned, so the work survives a context
reset.  Commit as you go; do not hold 14 hours of work in one uncommitted tree.

---

## 8. Deliverables

In `gen16_leads/`:

- `PRE_REGISTRATION.md` — locked, hashed, committed before any lead runs
- `DECISION_REPORT.md` — gen13's format: headline, what was run, per-lead sections with full
  five-design tables, what is statistically supported and what is not, defects found, guardrails for
  readers, recommendation
- `CONFIRMATION.md` — the single confirmation run, its five claims, discovery-vs-confirmation table
- `REFUTATION_LOG.md` — every refuter's attempt and outcome, including the ones that failed to break
  a claim
- `headline_tables/`, `metrics/`, `bootstrap/`, `predictions/`, `scripts/`, `tests/` — following the
  gen13/gen15 layout
- regression tests for anything new, and a test that the three anchors still reproduce

Respect `.gitignore`: prediction dumps and large parquet under `runs/` are excluded by design and
their SHA-256 goes in a manifest instead.  Follow that convention for anything large you generate.

---

## 9. What counts as success

In descending order of value:

1. **A new representation that beats the lean block set under BP**, on confirmation seeds, in all
   five designs, against the cheapest competitor, surviving both refuters.  L1 is the most likely
   source.  This would be the first zero-shot representational gain since gen2.
2. **A defensible corpus-expansion plan** (L4) with retrospective learning curves showing that
   model-chosen ordering beats random and maxmin.  This attacks the actual bottleneck, 11.7
   effective chemotypes, which no estimator can fix.
3. **Decision value made concrete** (L3's narrow forms, or L5): measurements saved by the direction
   call with a permutation null, or the first calibrated zero-shot intervals in the programme, or
   the §7 covariance gain re-tested against the deployed baseline and shown to hold.  All three are
   modest and all three are defensible.
4. **A clean, mechanistic null.**  "The cycle-corrected xTB slope is ρ = 0.2, here is why the 0.644
   was an artefact, here is the exact bookkeeping that produces it" closes a lead permanently and is
   publishable.
5. **A methodological finding** about how this class of data must be validated.

What is *not* success: a nominally significant result from an unregistered arm; a gain quoted
against `MEAN_CURVE`; a number from design B; a positive that only appears in one of five designs;
anything that a refuter dented and was then quietly re-scoped.

If at hour 14 nothing has cleared the bar, the correct output is a report that says so, names what
was measured and at what power, and states what would have to change.  **Write that report.  Do not
lower the bar to produce a headline.**

---

## 10. Practical notes

- Cluster and SLURM submissions belong to the user.  Do local analysis and hand over ready submit
  commands with expected runtime and memory.
- The deployable predictor is `gen15_curve/scripts/g15_predict.py` (SMILES + measurements → the
  14-metal curve, 91 pairwise `log SF` with 90 % intervals, and the next pair to measure).  If any
  lead succeeds, update it and keep its interface.
- Read `gen15_curve/GEN15_REPORT.md` §5 before touching the measured mode.  Two protocol defects
  were fixed there — a non-like-for-like k-ladder and a covariance that could see the test cell's
  publication — and both had flattered earlier numbers.  Do not reintroduce them.
- Report every number with its regime.  "0.50" is meaningless without "extractant-macro MAE of
  `log SF`, design BP, 5 seeds".
