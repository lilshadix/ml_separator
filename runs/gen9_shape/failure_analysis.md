# gen9 failure analysis — every arm that did not work, and why

*The brief (§27) asks for this as a first-class artefact rather than an appendix.
Losing arms are not omitted, and each is diagnosed against the same questions: did it
fail to move shape at all; did it move shape but hurt level; did it overfit an axis;
did it help one axis and hurt another; did it fail only on hard chemotypes; was the
hypothesis wrong or only the implementation?*

*Numbers marked **development** come from the screening seeds or single-seed probes
and are never quoted as confirmed effects.*

---

## 1. Sampler screen — **locked, five seeds, post-determinism-fix**

All arms share one learner, one hyperparameter set, one design matrix and one set of
model seeds; the only difference is which pairs of rows on a curve the loss sees. The
three-seed screen originally run for this section was produced *before* the
reproducibility defect (decision report §0.3, issue 6) was found, and is superseded —
none of its numbers are quoted. Source: `runs/gen9_shape/finalists/`,
`runs/gen9_shape/shape/`.

### 1.1 The one thing every shape arm does

**Dynamic-range compression falls, significantly, on the extractant axis, for every
sampler, in 5/5 seeds.** Against the `A0_ROW_ONLY` control (positive = better; the
metric is distance of span recovery from 1.0):

| sampler | span-recovery gain | 95 % CI | ligands improved | seeds |
|---|---|---|---|---|
| `A1_ROW_ADJACENT` | +0.067 | [0.039, 0.117] | **25/25** | 5/5 |
| `A2_ROW_ENDPOINT` | +0.107 | [0.037, 0.191] | 23/25 | 5/5 |
| `A3_ROW_RANDOM_PAIR` | +0.088 | [0.060, 0.144] | **25/25** | 5/5 |
| `A4_ROW_MULTISCALE` | +0.033 | [0.006, 0.090] | 21/25 | 4/5 |

Median extractant span recovery rises from 0.046 (control) to 0.068 (A1), 0.074 (A3),
0.097 (A2). That is the mechanism working — and it is the *only* pre-registered
endpoint that moves.

### 1.2 Everything else gets worse

| requirement (protocol §3, extractant axis) | result |
|---|---|
| slope MAE improves, CI excludes zero | **fails for every sampler.** A1 −0.091, A4 −0.249, A2 −0.251, A3 −0.344 — all *worse* than the control |
| median predicted/true slope ratio materially above ≈ 0.05 | **fails.** Median predicted slope: control 0.108, A1 0.105, A2 0.095, A3 0.084, A4 0.079 — against a measured 2.574. Not one sampler raises it |
| shape MAE improves, CI excludes zero | **fails for every sampler.** A1 −0.069, A4 −0.082, A3 −0.118, A2 −0.119 |
| no catastrophic macro-MAE loss (≤ +0.10) | holds for A1 (+0.007), A4 (+0.006), A3 (+0.019); **fails for A2** (+0.081) |

Within-curve Spearman falls in every case: 0.692 (control) → 0.444 (A1), 0.378 (A4),
0.336 (A3), 0.247 (A2).

**The pre-registered experiment therefore fails on three of its four conditions.** It
buys range and pays for it with ordering, and the range it buys is a twentieth of what
is missing.

### 1.3 A2_ROW_ENDPOINT — the informative failure

Endpoint-only supervision isolates dynamic range and behaves exactly as the isolation
predicts: it wins the metric it targets and destroys everything else.

| extractant axis | A0 control | A2 endpoint |
|---|---|---|
| span recovery (median) | 0.046 | **0.097** |
| within-curve Spearman | 0.692 | **0.247** |
| within-curve sign accuracy | 0.823 | **0.615** |
| shape MAE | 0.666 | 0.692 |
| macro MAE | 0.989 | **1.071** |

Sign accuracy 0.615 is close to the 0.5 a constant predictor scores. A loss that sees
only a curve's two ends is satisfied by any function with the right range, including
one that runs backwards through the middle. **On span alone A2 looks like the best arm
in the study**, which is precisely why the protocol required the shape metrics to move
coherently before a flattening claim could be made.

### 1.4 A4_ROW_MULTISCALE — the pre-declared candidate came last

`A4_ROW_MULTISCALE` was named the primary candidate in the protocol before any arm
ran. At five seeds it has the *smallest* span gain of the four (+0.033, the only one
not clearing 5/5 seeds), the second-worst slope error, and a lower predicted slope than
the control. Mixing scales lets the endpoint component pull the range while the
adjacent component is too weak to hold the ordering, and neither wins.

### 1.5 The high-weight arms

`X_MULTI_d2` (λ=2) and `X_MULTI_d8c` (λ=8, cluster-balanced) were added EXPLORATORY
after the dose–response probe. At five seeds they confirm the trade-off and nothing
else: `d8c` reaches extractant span recovery 0.335 and lanthanide span recovery
**1.257** — overshooting the measured range — while its extractant slope stays at
0.110, its Spearman collapses to 0.090, and its macro MAE is 1.273. `d2` is macro
1.195 with a Spearman of 0.179. The single-seed probe had made `d8c` look like the best
trade-off in the study; five seeds says it is the worst arm in it. **This is the
clearest single argument in gen9 for not reading a single-seed probe as a result.**

### 1.6 What actually worked, and it was not a loss

Two EXPLORATORY arms, added after this screen was read, against the same control:

| extractant axis, vs `A0_ROW_ONLY` | `GEN9_REL_MONOLITH` | `GEN9_SHAPE_RECOMPOSED` |
|---|---|---|
| slope MAE | **+0.504** [0.169, 0.631], 23/25, 5/5 | **+0.883** [0.435, 1.069], 21/25, 5/5 |
| shape MAE | **+0.091** [0.034, 0.104], 23/25, 5/5 | **+0.172** [0.084, 0.203], 21/25, 5/5 |
| span-recovery distance | **+0.242** [0.105, 0.303], 25/25, 5/5 | **+0.365** [0.253, 0.425], 23/25, 5/5 |
| median predicted slope | 0.521 | **1.024** |
| macro MAE | 0.986 | **0.969** (control 0.989, frozen 0.981) |

Neither is a loss function. Both are the same five columns telling the model where on
its own curve a row sits. The full account is in the decision report §1 and §2.

### 1.7 Was the hypothesis wrong, or the implementation?

**The hypothesis was wrong, and the way it was wrong is the finding.** The premise was
that a row-wise objective rewards conditional averaging and destroys derivatives, so
supervising derivatives would restore them. The derivatives were never destroyed by the
objective: they were never expressible, because the model had no coordinate in which
the within-curve response is a function of the inputs. A model trained *only* on the
curve-centred target — no level to average toward, nothing to trade against — still
recovers 2.5 % of the extractant range. No weighting of a loss over that function class
could have worked.

## 2. Loss-weight and capacity probe — **development**, one split seed, all five folds

The phase-1 screen fixed `lambda_delta = 0.5`. This probe asks what happens above and
below it, and tests the two implementation hypotheses declared in §1.7. Source:
`scratchpad probe_lambda`; single seed (104729), all five folds, so the held-out
chemotype coverage is complete but the seed spread is not measured.

### 2.1 The dose–response

| arm | macro MAE | extractant slope | extractant span | extractant Spearman | acid slope | acid span |
|---|---|---|---|---|---|---|
| control (3-seed) | 0.982 | 0.115 | 0.053 | 0.650 | 0.392 | 0.231 |
| `lambda_delta = 0.5`, curve | 1.011 | 0.172 | 0.075 | 0.398 | 0.460 | 0.391 |
| `lambda_delta = 2`, curve | 1.210 | 0.279 | 0.310 | 0.197 | 0.437 | 0.686 |
| `lambda_delta = 8`, curve | **2.155** | 0.486 | **0.756** | 0.186 | **−0.298** | 1.455 |
| `lambda_delta = 2`, cluster | 1.032 | 0.110 | 0.073 | 0.357 | 0.402 | 0.244 |
| `lambda_delta = 8`, cluster | 1.312 | 0.385 | 0.334 | 0.306 | 0.210 | 0.687 |

**Hypothesis 1 (the weights were too small) is confirmed, and the confirmation is a
failure.** The range *is* recoverable: `lambda_delta = 8` reaches span recovery 0.756,
past the pre-registered "strong result" bar of 0.5. It costs +1.17 macro MAE against
the control — the pre-registered catastrophe bar is +0.10 — collapses the within-curve
ordering to 0.186, and **inverts the acid slope** from +0.39 to −0.30 while
overshooting the acid range to 1.455 of truth. There is no setting at which the range
comes back and the model still works.

### 2.2 The cluster-balanced pair weighting — a clean negative

The weighting hypothesis was that the row loss's cluster-balanced weights (up to ~7
for rows in rare ECFP clusters) outgun a delta term whose Huber gradient saturates at
0.5, so matching the two weightings should let the shape term bite. **It does the
opposite.** At `lambda_delta = 2`, cluster balancing takes span recovery from 0.310
back down to 0.073 — essentially the control.

The reason is that the arithmetic runs the other way from the guess. Most extractant
titrations belong to *heavily measured* ligands, which sit in large ECFP clusters and
therefore carry **small** row weights. Under `curve` balance the shape term already
dominates on exactly those rows; cluster balancing down-weights their curves to match
their rows and removes the signal. The default was already right.

The one place cluster balancing helps is the trade-off at high `lambda_delta`: at
`lambda_delta = 8` it reaches span recovery 0.334 for macro 1.312, against curve
balance's 0.310 for macro 1.210 — and it is the **only configuration in the study to
improve extractant shape MAE** (0.637 against the control's 0.663). That is why it is
carried into the five-seed finalists as `X_MULTI_d8c`, labelled EXPLORATORY.

### 2.3 Hypothesis 2 (feature subsampling) — not tested to completion

The `max_features = 1.0` arms were started and stopped: at 2,160 columns they cost
roughly four times the runtime of the rest of the study, and the dose–response above
had already answered the question they were asked to answer. The model *can* express
steep responses when the loss demands it (`lambda_delta = 8` → median slope 0.486), so
the binding constraint is not the split-proposal rate on condition columns; it is the
trade-off against row accuracy. A full `max_features` sweep is a gen10 item, and it is
recorded here as **not run** rather than as a null.

## 3. Axis and weighting variants

*(the `AX_*` stage was not run: the phase-1 and dose–response results made the axis
inventory a second-order question next to the trade-off, and the compute went to the
five-seed finalists and the k-shot frontier instead. Recorded as not run.)*

## 4. Acquisition learners — **fails its target**, three ways

gen9-B was to predict `q_i = |r_i − median(r)|` from observables and thereby beat
geometric centrality. It beats **random** (+0.070 [0.028, 0.099], 99/143, 5/5) and does
not beat **centrality** (−0.005 [−0.018, +0.014] against `MEDOID`, 0/5 seeds). The
pre-registered target — recover ≥ 30 % of the remaining MEDOID→oracle gap — is missed
entirely: the closed fraction is zero.

### 4.1 The target is weaker than gen8's proof implies

gen8 verified `argmin |r_i − median(r)|` attains the realised oracle to 6.7e-16 — in
its **exhaustive** protocol, where the candidates and the scored rows are the same
points. gen9 selects from a pool and scores on a disjoint evaluation set. Measured over
**8,580 blocks** under that protocol:

* exactly optimal in **60.1 %** of blocks (not 100 %);
* median regret 0.000, mean 0.019, 95th percentile 0.112;
* median Spearman against the realised score 0.948;
* and, decisively, `SURROGATE_ORACLE` — following the rule *with the residuals known* —
  scores 0.501 against the true oracle's 0.481, i.e. it recovers only **53 %** of the
  oracle gap.

A learner trained on this target inherits a 47 % ceiling before it makes an error of its
own. That is a property of the *target*, not of the learner, and it was not visible
until the identity was re-derived under the deployment protocol.

### 4.2 More information makes it worse

| feature set | one-shot macro MAE (best learner) |
|---|---|
| `geometry` | **0.651** |
| `full` (geometry + predictions + response-surface context) | 0.653 |
| `geometry+prediction` | 0.653 |
| `full_shuffled` (the §15 control) | 0.655 |

Adding the model's own predictions and the response-surface context makes the policy
*worse*, and the shuffled control says why there was nothing there: destroying the
correspondence between a candidate and its curve costs **0.002**. This reproduces gen8's
finding that molecular and model-derived information does not help acquisition — now
with a response-surface representation that demonstrably works for *prediction*
(§1.6), so the null is not about the features being bad.

### 4.3 The three learners, ranked

`LEARNED_BLEND` (0.651) > `LEARNED_SCALAR` (0.653) > `LEARNED_RANK` (0.664). The blend
is best because it can fall back — and it only *could* fall back after the defect in
its `alpha` selection was fixed (decision report §0.3, issue 8). Before the fix it chose
`alpha = 0` in **25 folds out of 25** and scored 0.740, worse than the baseline it was
built to nest. The pairwise ranker being the worst of the three is itself a small result:
optimising the ordering directly does not beat regressing the value, on this target.

### 4.4 Diagnosis

* Did it fail to predict anything? No — it beats random by 0.070 with every seed agreeing.
* Did it fail on hard chemotypes only? No — the deficit against `MEDOID` is flat across
  the ligand distribution (53/143 improved, i.e. a coin flip).
* Was the hypothesis wrong or the implementation? **The hypothesis was half-right and the
  target was oversold.** Centrality already captures what is predictable about
  `argmin |r_i − median(r)|`; what remains is the part that depends on the residual
  field itself, which is exactly what cannot be known before measuring.

## 5. Series-local adapter — **fails**, and the reason is the prior

gen9-C replaced gen8's hand-set ridge penalty of 4.0 with penalties estimated
fold-locally by empirical Bayes: `sigma^2 / tau^2`, where `tau` is how much a
coefficient family actually varies between the fold's own training ligands and `sigma`
is the scatter left inside a ligand once it is fitted. The argument for it was that a
measurement beats a knob. The measurement lost.

`GEN9_REL_MONOLITH` under `CENTRAL_THEN_SPREAD`, macro MAE on the common cohort:

| adapter | k=1 | k=2 | k=3 | k=5 |
|---|---|---|---|---|
| `OFFSET_K1` (gen8, level only) | 0.630 | 0.603 | 0.590 | 0.609 |
| `OFFSET_K3` (gen8, fixed ridge 4.0) | 0.630 | **0.541** | **0.489** | **0.478** |
| `SERIES_MAP` (gen9 hierarchy) | 0.630 | 0.596 | 0.573 | 0.574 |
| `SERIES_MAP_NOSERIES` | 0.630 | 0.602 | 0.587 | 0.605 |
| `SERIES_MAP_LEVELONLY` | 0.630 | 0.597 | 0.575 | 0.577 |
| `SLOPE_L_s1_K3` (gen8 repair + fixed ridge) | 0.620 | **0.531** | **0.480** | **0.468** |

Three things to read out of that table.

**The k = 1 column is identical for every adapter, and that is a theorem, not a
coincidence.** A ridge with an unpenalised intercept fitted on one observation has the
exact solution `beta = [r, 0, ..., 0]`: substituting it into
`(dd' + diag(0, lambda...)) beta = d r` gives `d(1*r + z*0) = d r`. Every penalised
coefficient is exactly zero, so K1, K2, K3 and the hierarchy are algebraically the same
model. Verified to 6.7e-16 on 286 held-out units. gen9-C is only testable at k >= 2.

**The hierarchy's series component works, slightly.** `SERIES_MAP` beats
`SERIES_MAP_NOSERIES` at every k (0.596 vs 0.602 at k=2, 0.573 vs 0.587 at k=3), which
is gen8's "calibration is series-local" finding reproduced as a modelling gain. It is
small.

**The hierarchy's estimated penalties are much worse than gen8's fixed one.**
`SERIES_MAP` loses to `OFFSET_K3` by 0.055 at k=2, 0.084 at k=3 and 0.096 at k=5 —
larger gaps than anything else in the table. The diagnosis is over-shrinkage, and it is
mechanical: `estimate_prior` fits each training ligand's out-of-fold residuals with a
ridge of its own, so the *fitted* coefficients are already shrunk toward zero; taking
their spread as `tau` therefore underestimates how much the coefficients really vary,
which inflates `sigma^2/tau^2` and shrinks the response coefficients far harder than
gen8's 4.0 does. **The empirical prior measures a shrunk quantity and treats it as the
truth.**

Diagnosis: the *hypothesis* — that calibration is series-local and a hierarchy should
express that — is supported by the `NOSERIES` ablation. The *implementation* of the
prior is wrong, and an unbiased estimate of `tau` (deconvolving the inner ridge, or a
proper marginal-likelihood fit) is the obvious repair. Recorded as a gen10 item, not
attempted here.

## 6. Cross-cutting diagnoses

**Every arm designed *before* seeing data failed; the two that worked were designed
after.** The pre-registered loss, its weight sweep, the pairwise ranker, the empirical-
Bayes hierarchy — all four were reasoned from gen8's findings and all four lost. The
relative-position representation was written in twenty minutes after a control
experiment made the mechanism obvious. That is not an argument against pre-registration:
the pre-registered arms are what *produced* the control experiment, by failing in a way
that ruled out the obvious explanation. It is an argument for running the cheap
diagnostic before the expensive architecture.

**Three of the eight defects were found by a scan failing to rediscover something
already known.** The duplicate-cell scan reporting zero exact-decade pairs, the level
scan reporting zero outliers, the seven geometric policies reporting one number — in
each case the *absence* of an expected finding was the signal. A scan that finds nothing
should be tested against a known positive before it is believed.

**Single-seed probes were actively misleading twice.** `X_MULTI_d8c` looked like the best
trade-off in the study at one seed and is the worst arm in it at five. The `lambda`
dose–response was directionally right and quantitatively wrong. Every probe number in
this document is labelled development for that reason.

**The one thing that survived every attack is also the smallest claim.** The
relative-position representation moves six shape metrics on three axes with 5/5 seeds,
survives the window-width, ligand-concentration, publication-concentration and
regression-to-the-mean checks — and buys 0.025 macro MAE at k = 0, falling to 0.007 by
k = 5. It is a real mechanism with a modest payoff, and the payoff shrinks exactly where
theory says it should: a measurement supplies what the model was missing.
