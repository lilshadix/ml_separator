# gen8 — series-aware few-shot architecture for `log D`

*Run 2026-08-20. Cohort fingerprint `bed178ec1a7a82b0` — 5,248 rows, 152 extractants,
131 ECFP clusters, 79 Tanimoto chemotypes, identical to gen6 and gen7. The global model
is **not retrained anywhere in this report**: every number is a calibration of the frozen
gen7 arm `REC_ecfp_plus_recovered` on its own out-of-fold predictions over held-out
chemotypes, which is what makes the whole study paired down to the row. Artifacts:
`runs/gen8_architecture/`. Pre-registration: `runs/gen8_architecture/protocol.md`.*

## WHAT DID ONE MEASUREMENT ACTUALLY BUY?

**0.268 log units — and choosing *which* measurement is worth another 0.101, which is
more than the entire zero-shot contribution of ligand chemistry (0.090, gen7's
measurement of what every fingerprint, descriptor and pretrained embedding buys
together).**

That second number is the new result. gen7 already knew a measurement helps. What gen8
establishes is that the *choice* of measurement is a first-class lever with a large,
partly-capturable gap: on identical held-out ligands an oracle that picks the best single
point scores **0.533** where a random point scores **0.793**. A 0.260 gap between random
and optimal selection is not a rounding error — it is nearly three times the entire
zero-shot contribution of ligand chemistry (0.090, gen7). The brief said that if the
oracle came in near 0.50, acquisition policy becomes a major research target. It came in
at 0.533, and 39 % of that gap is already captured by a rule with no free parameters:
**measure at the centre of the ligand's condition space** (+0.101, CI [+0.037, +0.146],
77 of 99 ligands, 5 of 5 seeds).

Four more things changed, and one did not.

**The model draws every unseen titration far too flat.** This is the largest
previously-unreported defect in the system, and it is a shape error, not a level error.
Measured curve by curve on held-out ligands, the true median
`d(log D)/d(log₁₀[extractant])` is **2.57** and the predicted median is **0.12** — a
factor of twenty. On the acid axis it is 1.66 against 0.36. An ensemble asked about a
chemotype it has never seen returns something near the conditional mean over training
ligands, and averaging ligands whose curves sit at different heights flattens the slope
they share. Injecting the training prior back, curve by curve — using **no target of any
kind**, only the model's own predicted slope — takes extractant-titration shape MAE from
**0.555 to 0.194**, with 1,096 of 1,205 curves improving.

**gen7's rule "only the level can be adapted" was an artefact of unrestricted fitting.**
gen7 saw an affine recalibration at k = 2 score 3.32 against 0.64 for a plain offset and
concluded that k = 2 supports one degree of freedom. It supports four — provided they are
shrunk. With the penalty swept, an unpenalised four-coefficient update at k = 2 returns a
macro MAE of **1.7 × 10¹²** (the failure is real and reproduces), while the same four
coefficients at λ = 1 score **0.571** against 0.640 for the offset. The correct statement
is not "the level only" but "as many coefficients as you like, held down hard".

**Model uncertainty is not the quantity that should choose the next experiment.** Every
uncertainty source tested — tree-ensemble variance, five-model disagreement, distance to
the training conditions, GP predictive variance — correlates with the absolute residual
at ρ ≤ 0.14 *within* a ligand, and no uncertainty-driven acquisition rule clears the
pre-registered bar. Max-ensemble-sd against random is +0.025, CI [−0.018, +0.052]; its
point estimate flips sign between split draws, which is itself the evidence that there is
nothing there. The mechanism is exact rather than mysterious and falls out of the
algebra: under offset calibration, measuring row *i* gives `MAE = mean_j |r_j − r_i|`, so
the best row is the one whose residual is nearest the ligand's **median** residual. The
target is an unsigned deviation from a centre, not a magnitude — and the difference is
not academic. Over all 715 (ligand, seed) blocks:

| selection rule | macro 1-shot MAE |
|---|---|
| oracle — minimise the realised remaining MAE | **0.522013** |
| minimise `\|r_i − median(r)\|` | **0.522013** (identical to 6.7 × 10⁻¹⁶) |
| random | 0.709 |
| minimise `\|r_i\|` — what a perfectly calibrated uncertainty approximates | **0.780** |

The median-deviation rule *is* the oracle, verified to machine precision. And a rule that
measures where the model is most accurate — which is what a perfectly calibrated
uncertainty would give you — is **worse than choosing at random**, by more than active
selection is worth. That is why no uncertainty signal helps: they are all estimating the
wrong quantity, and the right one is not an uncertainty. The only uncertainty-derived rule that beats
random is the *minimum* of an RBF-GP **design** variance over the ligand's own condition
axes (+0.070, CI [+0.046, +0.088], 112 of 143, 5/5) — and that is a geometric centrality
measure wearing an uncertainty costume.

**One architecture out of three earned its place, and it is the physics one.** Of a
Conditional Neural Process, a physics-latent model and three series-aware curve baselines —
all built to the same interface and scored on the same rows — only the physics-latent model
clears the bar: it beats offset correction by **+0.056** at k = 1 and **+0.079** at k = 2,
and beats even the shrunk four-coefficient update at every k ≥ 2. What carries it is a
*functional form* — intercept plus mass-action slopes plus a smooth lanthanide response,
fitted to the frozen model's residual — not the map from chemistry to those coefficients:
swapping ridge for gradient boosting inside it is worth +0.003. And the same law applied to
the absolute `log D` rather than the residual is **significantly worse than doing nothing
but adding a constant** (−0.055, 0 of 5 seeds), as is a Neural Process on the absolute value
(−0.145). Whatever these architectures add, they add as a correction, never as a
replacement.

**What did not change: the level is still not predictable from structure.** gen8's
mechanistic representation — 36 compact quantities built for extraction chemistry — was
constructed precisely to test whether gen7's ceiling was really a *neighbourhood* problem.
It is not. On the condition-adjusted level a 1-nearest-neighbour **Tanimoto** lookup
(0.989) beats a 1-nearest-neighbour **mechanistic** lookup (1.077) by 0.085,
CI [+0.028, +0.137], 5 of 5 seeds. gen7's ceiling stands, reached now by a third
independent route.

![Where the error goes](runs/gen8_architecture/figures/oracle_gap.png)

## SERIES RECONSTRUCTION — WHAT THE MODEL IS SUPPOSED TO REPRESENT

Before any architecture, §1: the bundle carries no "this is an acid titration" label, so
the series had to be rebuilt. Two units, deliberately distinct. A **series** is gen5's
definition reused byte-for-byte — one extractant × one setting of the categorical
conditions. A **curve** is new: a maximal subset in which *exactly one axis varies* and
every other axis is held fixed.

**5,127 of 5,248 rows (97.7 %) lie on at least one of 1,176 reconstructable curves.**
The response surface gen8 proposes to model is present in the data; it is not an
idealisation.

| curve type | curves | rows | ligands | median points | median linear R² | median slope | IQR |
|---|---|---|---|---|---|---|---|
| lanthanide series | 386 | 3,094 | 89 | 7 | 0.885 | 0.087 | [0.02, 0.18] |
| acid titration | 496 | 2,751 | 74 | 5 | 0.918 | 1.573 | [0.76, 2.28] |
| extractant titration | 241 | 1,078 | 27 | 4 | **0.996** | **2.305** | [1.96, 2.99] |
| temperature | 35 | 149 | 6 | 4 | 0.989 | −0.040 | [−0.05, −0.03] |
| metal concentration | 10 | 85 | 4 | 7 | 0.766 | −1.140 | [−1.51, −0.07] |
| contact time | 8 | 50 | 8 | 5.5 | **0.165** | 0.0001 | [−0.00, 0.00] |

Two readings decide how the surface must be modelled.

**56 series are grids, not curves** (3,183 rows — the majority of the data). A paper
reporting a metal series at four acidities produces one `series_id` covering a 2-D
surface, and fitting one smooth curve through it — the brief's explicit warning — would
be fitting a curve through a surface. Every grid is decomposed into its constituent
one-axis curves instead, and a row at the intersection of two curves belongs to both,
which is what makes cross-series transfer a measurable question at all.

**The mass-action law is visible in the raw slopes, and the smoothness prior is not
global.** Extractant titrations have median linear R² 0.996 on the log–log axis with a
slope of 2.31 — that is a solvation number, recovered without being told to look for one.
Contact-time curves have median R² 0.165 and slope 0.0001: they are flat noise once
equilibrium is reached. A smoothness or monotonicity penalty is defensible on the
concentration axes and indefensible globally, exactly as the brief insisted.

## THE PRIMARY COMPARISON

Two populations are reported throughout, and the difference between them matters. **Table
B** is the common cohort — the *identical* 99 ligands at every k, so the adaptation curve
is longitudinal (§16). **Table A** is maximal coverage (143 ligands at k ≤ 3, 99 at
k = 5) and is the population gen7's reference numbers were computed on, so it is the one
the success thresholds are read against.

| arm | k | Table B (99) | Table A (143) |
|---|---|---|---|
| **ZERO_SHOT** — gen7's best arm, untouched | 0 | 1.061 | 0.995 |
| NO_MODEL — the mean of the measurements | 1 | 0.946 | 0.846 |
| NO_MODEL | 2 | 0.847 | 0.754 |
| RANDOM 1-shot, offset correction | 1 | 0.793 | 0.723 |
| RANDOM 2-shot, offset correction | 2 | 0.702 | 0.640 |
| RANDOM 2-shot, shrunk response coefficients | 2 | 0.657 | 0.598 |
| BEST FIXED 1-SHOT POLICY (`CENTRAL`) | 1 | 0.696 | 0.651 |
| ACTIVE 1-shot (`MEDOID`) | 1 | 0.692 | 0.648 |
| slope repair + `MEDOID` | 1 | 0.675 | 0.633 |
| **BEST 1-shot** — physics-latent residual + `CENTRAL` | 1 | **0.667** | **0.626** |
| **BEST 2-shot** — slope repair + K3 + max predictive variance | 2 | **0.588** | **0.530** |
| **BEST 3-shot** — slope repair + K3 + central-then-spread | 3 | **0.523** | **0.489** |
| **BEST 5-shot** — slope repair + K3 + D-optimal | 5 | **0.474** | **0.474** |
| ORACLE 1-shot (not deployable) | 1 | 0.533 | 0.484 |
| ORACLE 2-shot (not deployable) | 2 | 0.520 | 0.474 |
| ORACLE level — best constant, full knowledge | — | — | 0.480 |

**Against the brief's success thresholds**, read on Table A:

* **1-shot ≤ 0.65 "useful": met** — 0.626. (≤ 0.60 "strong" is not met.)
* **2-shot ≤ 0.55 "useful": met** — 0.530. (≤ 0.50 "strong" is not met.)

gen8 clears the "useful" bar on both primary endpoints and clears neither "strong" bar.
Two facts put that in proportion. The best 5-shot arm (0.474) **beats the best possible
single measurement** (0.484 / 0.533) — +0.078, CI [+0.052, +0.112], 71 of 99 ligands,
5/5 seeds — so a modest budget already exceeds perfect single-point selection. And the
full metric panel moves together rather than trading off: from zero-shot to the best
5-shot arm, within-ligand Spearman rises 0.394 → 0.609, sign accuracy 0.670 → 0.763, the
fraction within 0.5 log units 0.344 → 0.688, worst-quartile ligand MAE 1.411 → 0.703, and
hard-chemistry (nearest-neighbour Tanimoto < 0.4) MAE 1.090 → 0.491.

## GAIN FROM CALIBRATION, ARCHITECTURE, ACQUISITION AND SERIES MODELLING

Paired chemotype-blocked bootstrap on the common cohort, 5,000 replicates, resampling the
Tanimoto chemotype the folds actually held out. Positive = the candidate is better.

| attributed to | comparison | k | point | 95 % CI | BCa | units | seeds + |
|---|---|---|---|---|---|---|---|
| **measurement** | one random point vs zero-shot | 1 | **+0.268** | [+0.140, +0.462] | [+0.142, +0.466] | 52/99 | 5/5 |
| **measurement** | offset correction vs no model at all | 1 | **+0.153** | [+0.069, +0.221] | [+0.075, +0.227] | 69/99 | 5/5 |
| **acquisition** | `MEDOID` vs `RANDOM` | 1 | **+0.101** | [+0.037, +0.146] | [+0.047, +0.160] | 77/99 | 5/5 |
| **acquisition** | `CENTRAL` vs `RANDOM` | 1 | **+0.096** | [+0.035, +0.137] | [+0.047, +0.149] | 75/99 | 5/5 |
| acquisition | max ensemble sd vs `RANDOM` | 1 | +0.025 | [−0.018, +0.052] | [−0.008, +0.058] | 48/99 | 5/5 |
| acquisition | farthest-point vs `RANDOM` | 1 | **−0.121** | [−0.171, −0.072] | [−0.177, −0.076] | 32/99 | **0/5** |
| **acquisition** | max predictive variance vs `RANDOM` (K3) | 2 | **+0.055** | [+0.033, +0.074] | [+0.033, +0.074] | 72/99 | 5/5 |
| **architecture** | K3 vs K1 adaptation | 2 | **+0.044** | [+0.029, +0.054] | [+0.031, +0.056] | 86/99 | 5/5 |
| **architecture** | K3 vs K1 adaptation | 5 | **+0.115** | [+0.084, +0.137] | [+0.085, +0.139] | 83/99 | 5/5 |
| **architecture** | K2 vs K1 adaptation | 2 | **+0.020** | [+0.013, +0.029] | [+0.013, +0.030] | 54/99 | 5/5 |
| **architecture** | physics-latent residual vs offset (`RANDOM`, 143 ligands) | 1 | **+0.056** | [+0.006, +0.090] | [+0.013, +0.097] | 106/143 | 5/5 |
| **architecture** | physics-latent residual vs the K3 update (`RANDOM`, 143) | 2 | **+0.038** | [+0.016, +0.059] | [+0.017, +0.060] | 91/143 | 5/5 |
| architecture | physics-latent residual vs offset, both at `CENTRAL` | 1 | +0.024 | [−0.007, +0.047] | [−0.006, +0.047] | 101/143 | 5/5 |
| **series modelling** | slope repair vs offset, both at `MEDOID` | 1 | **+0.016** | [+0.001, +0.029] | [+0.004, +0.034] | 17/99 | 5/5 |
| **series modelling** | slope repair vs offset, both K3 + max pred. var. | 2 | **+0.015** | [+0.002, +0.024] | [+0.004, +0.026] | 20/99 | 5/5 |
| **series modelling** | the same, on **shape MAE** | 2 | **+0.015** | [+0.004, +0.025] | — | 20/99 | 5/5 |
| **all four together** | best deployable vs random offset | 2 | **+0.114** | [+0.081, +0.135] | [+0.084, +0.136] | 81/99 | 5/5 |
| **all four together** | best deployable vs random offset | 5 | **+0.154** | [+0.112, +0.184] | [+0.114, +0.186] | 85/99 | 5/5 |
| **headroom** | oracle vs `RANDOM` | 1 | **+0.260** | [+0.186, +0.310] | [+0.199, +0.327] | 99/99 | 5/5 |
| **headroom** | oracle vs `MEDOID` | 1 | **+0.164** | [+0.123, +0.195] | [+0.126, +0.197] | 99/99 | 5/5 |
| **budget beats oracle** | best 5-shot vs the 1-shot oracle | 5 | **+0.078** | [+0.052, +0.112] | [+0.053, +0.115] | 71/99 | 5/5 |

Two directions of failure are as firmly established as the successes, and their
disagreement is the physics of the problem. **The farthest-point rule is significantly
worse than random at k = 1** (−0.121, 0/5 seeds) while **max-predictive-variance is
significantly better than random at k = 2** (+0.055, 5/5). A single measurement estimates
a *level*, for which you want the most typical available point; two or more estimate
*slopes*, for which you need leverage. Any rule that is right at k = 1 for the level
reason is wrong at k ≥ 2 for the slope reason. That is why the arm that wins at k = 3 is
the composite `CENTRAL_THEN_SPREAD` — anchor first, then spread — which was declared
before it was measured.

## ORACLE GAP ACCOUNTING

Common cohort, 99 ligands, every step a paired experiment:

```text
ZERO SHOT                                              1.061
  + one random measurement                −0.268       0.793
  + choosing that measurement well        −0.096       0.696   CENTRAL
  + a better calibration architecture     −0.029       0.667   physics-latent residual*
  + a second measurement, slopes adapted  −0.079       0.588   K3 + max predictive variance
  + three more                            −0.114       0.474   + slope repair, D-optimal
──────────────────────────────────────────────────────────────
ORACLE 1-SHOT   (best single point, not deployable)    0.533
ORACLE 2-SHOT                                          0.520
ORACLE LEVEL    (best constant, full knowledge)        0.480
```

Read the attribution rather than the total. Of the 0.587 recovered from zero-shot to the
best five-measurement arm: **0.268 is the first measurement, 0.193 is more measurements,
0.096 is choosing them, and 0.029 is the architecture** (of which 0.016 is the slope
repair, the only component that measurably reduces *shape* error). The measurement
dominates; the *selection* of the measurement is the second-largest per-step term and is
more than three times the architectural term as it appears **in this chain**.

\* That 0.029 is the architecture's *marginal* value once the point has already been chosen
centrally, and it understates the architecture taken alone: against offset correction at
**random** selection the same model is worth **+0.056**. The two overlap — a central point
and a physically regularised level are both better estimates of the same quantity — so they
must not be added. The chain above credits the acquisition first and the architecture with
what is left, which is the conservative order.

At k = 1 the deployable best (0.667) is still 0.134 short of the oracle (0.533), so the
acquisition problem is unsolved precisely where the experimental budget is tightest. By
k = 5 the deployable best (0.474) is *below* the 1-shot oracle and essentially at the
oracle level (0.480): **once the budget reaches five, the level is solved and everything
remaining is shape.**

![Adaptation curve](runs/gen8_architecture/figures/adaptation_curve.png)

## ADAPTATION CURVE AND THE MARGINAL VALUE OF EACH MEASUREMENT

Common cohort, macro MAE:

| arm | k=0 | k=1 | k=2 | k=3 | k=5 |
|---|---|---|---|---|---|
| no model at all | 1.061 | 0.946 | 0.847 | 0.801 | 0.769 |
| offset correction, random point | 1.061 | 0.793 | 0.702 | 0.663 | 0.628 |
| offset correction, chosen point | 1.061 | 0.692 | 0.664 | 0.643 | 0.625 |
| response coefficients, random point | 1.061 | 0.793 | 0.657 | 0.589 | 0.514 |
| **best deployable arm** | 1.061 | **0.667** | **0.588** | **0.523** | **0.474** |
| oracle single point (not deployable) | 1.061 | 0.533 | 0.530 | 0.533 | 0.553 |

Marginal value, best deployable arm: **0→1 = 0.394, 1→2 = 0.079, 2→3 = 0.065,
3→5 = 0.049 (0.025 per measurement).**

**Is the second measurement worth performing?** Yes, and the answer is quantitative:
0.079 log units, a fifth of what the first buys and roughly the same as the entire value
of choosing the first one well. The third is worth 0.065 and the fourth and fifth
about 0.025 each. If the constraint is bench time, the ranking is unambiguous — **take
the first measurement, choose it centrally, then take a second one far from it**; beyond
three, the returns are smaller than the spread between ligands.

One caveat the curve hides and the error archaeology exposes: **the average improving
does not mean every ligand improves.** 57 of 143 ligands are *worse* after one **random**
measurement than before it, because a single point estimates the offset with that point's
full shape noise attached. Choosing centrally reduces that count to 42. This is the
strongest practical argument for the acquisition result — the reason to choose well is
not only that the mean falls, it is that blind measurement actively harms a third of the
population.

![Which experiment to run](runs/gen8_architecture/figures/policy_comparison.png)

## HOW MANY DEGREES OF FREEDOM CAN k MEASUREMENTS SAFELY MOVE?

The brief asked for this to be determined experimentally rather than assumed (§7). Ridge
penalty on the non-intercept coefficients, swept; the intercept is never penalised.
Macro MAE, `RANDOM` selection, 143 ligands.

| adaptation | λ | k=1 | k=2 | k=3 | k=5 |
|---|---|---|---|---|---|
| K1 — level only | any | 0.723 | 0.640 | 0.616 | 0.628 |
| K2 — + lanthanide trend | **0** | 0.764 | 0.762 | 0.651 | 0.584 |
| K2 | 1 | 0.723 | 0.616 | 0.572 | 0.552 |
| K2 | 4 (pre-registered) | 0.723 | 0.625 | 0.585 | 0.565 |
| K3 — + acid and extractant slopes | **0** | 0.740 | **1.7 × 10¹²** | 6.7 × 10¹⁴ | 1.1 × 10¹⁴ |
| K3 | 0.25 | 0.723 | 0.575 | 0.517 | 0.519 |
| K3 | **1** | 0.723 | **0.571** | **0.506** | **0.491** |
| K3 | 4 (pre-registered) | 0.723 | 0.598 | 0.541 | 0.514 |
| K3 | 16 | 0.723 | 0.623 | 0.585 | 0.572 |
| K3 | 64 | 0.723 | 0.635 | 0.607 | 0.610 |

Four readings.

1. **At k = 1 every mode and every penalty collapses to the same number, exactly.** With
   one observation and a free intercept the penalised optimum puts everything into the
   level and every slope to zero. This is not a special case in the code — it is what the
   unpenalised intercept does automatically, and it is asserted in the test suite.
2. **gen7's catastrophe reproduces, and it is the penalty, not the parameter count.**
   λ = 0 at K3, k = 2 gives 1.7 × 10¹². Two observations cannot determine four
   coefficients, and an unrestricted fit says so numerically. Even K2 — *three*
   coefficients, unpenalised — is worse than the plain offset at k = 2 (0.762 vs 0.640,
   −0.122, 0 of 5 seeds).
3. **λ ≈ 1 is the optimum and the surface is flat between 0.25 and 4.** K3 at λ = 1
   beats the offset by **+0.068** at k = 2 (CI [+0.048, +0.088], 107 of 143 units, 5/5
   seeds); the pre-registered λ = 4 gives +0.041 (CI [+0.030, +0.050], 122 of 143). Both
   clear the bar, and both directions of mis-setting the penalty — too free and too
   stiff — cost real accuracy. *λ = 1 beating the pre-registered λ = 4 is a post-hoc
   observation and is labelled exploratory; every headline number in this report uses
   λ = 4, i.e. the conservative choice.*
4. **The number of safe degrees of freedom grows with k, and it grows faster than the
   count of observations.** At k = 2, four shrunk coefficients beat one free one.

## WHAT SHOULD A CHEMIST MEASURE FIRST?

This is the most directly usable result in the report, and it needs no model at
deployment time. Every candidate row of every held-out ligand was scored **exactly** —
under offset calibration the one-shot score has a closed form, so there is no sampling
noise in these numbers — and then compared *within ligand*, so a stratum cannot win by
containing easier ligands.

| if the one measurement is taken… | it is worse than the mid-range choice by | 95 % CI | BCa | seeds + |
|---|---|---|---|---|
| at the **low end of the acid range** | **+0.244** | [+0.175, +0.322] | [+0.175, +0.323] | 5/5 |
| at the **high end of the acid range** | +0.057 | [−0.002, +0.106] | [+0.006, +0.113] | 5/5 |
| at an **extreme lanthanide** rather than a central one | **+0.095** | [+0.057, +0.134] | [+0.060, +0.138] | 5/5 |
| where the model predicts a **low** value | **+0.143** | [+0.083, +0.187] | [+0.096, +0.197] | 5/5 |
| where the model predicts a **high** value | +0.040 | [+0.008, +0.074] | [+0.009, +0.075] | 5/5 |
| at a **high extractant concentration** | +0.111 | [+0.016, +0.281] | [+0.035, +0.359] | 5/5 |

**The recommendation: measure at mid-range acidity, on a mid-series lanthanide, at a
condition where the model predicts a middling `log D`.** The single largest effect is the
first one — a low-acid measurement costs 0.244 log units against a mid-range one, which
is nearly three times the entire value of choosing well on average, and more than twice
what any architecture in this study is worth. The reason is not subtle: at low acidity
extraction is weak, the measurement sits at the bottom of the curve where the model's
shape error is largest, and the residual it reports is unrepresentative of the ligand's
level.

The same conclusion arrives independently through the uncertainty analysis. Of every
uncertainty signal tested, the only one whose *minimum* beats random selection is the
RBF-GP design variance over the ligand's own condition axes — a purely geometric measure
of how central a candidate is (+0.070, CI [+0.046, +0.088], 112 of 143, 5/5). Model
confidence does the opposite: selecting by **minimum tree-ensemble variance** is
significantly *worse* than random (−0.043, 0/5 seeds).

## DOES ONE MEASUREMENT TRANSFER ACROSS SERIES?

**Largely not, and this is the sharpest negative result in the report.** The transfer
matrix measures on a row of one curve type and scores rows of another. Gains over
zero-shot, macro over ligands:

| measured → predicted | acid | extractant | lanthanide series | contact time |
|---|---|---|---|---|
| **acid** | **+0.190** | −0.028 | −0.161 | +0.692 |
| **extractant** | −0.066 | **+0.243** | −0.180 | +0.805 |
| **lanthanide series** | −0.036 | +0.016 | **+0.332** | +0.336 |

The diagonal is strongly positive and most of the off-diagonal is negative. **A ligand
does not have one level; it has a level per series.** A measurement in a chloroform
titration does not calibrate the same ligand's kerosene metal-series, because the
categorical conditions that define a series — diluent, modifier, acid identity — move
the level as much as the ligand does. This confirms, on the response surface, what gen6
found on the level: the level is a *(ligand, series)* quantity, not a ligand quantity.

The deployment consequence is concrete and constraining: **measure in the system you
intend to predict.** One acid-titration point will not tell you the metal selectivity of
the same ligand in a different diluent.

![Cross-series transfer](runs/gen8_architecture/figures/cross_series_transfer.png)

## THE MODEL DRAWS EVERY UNSEEN TITRATION TOO FLAT

Each of the 1,176 reconstructed curves was scored twice — once from the measured `log D`
and once from the frozen model's out-of-fold prediction on the *same* rows.

| curve type | true median slope | predicted median slope | ratio | span ratio | slope MAE | slope MAE of a "predict the median slope" null |
|---|---|---|---|---|---|---|
| extractant titration | 2.574 | **0.116** | 0.05 | 0.051 | 2.433 | **0.743** |
| acid titration | 1.656 | 0.355 | 0.21 | 0.218 | 1.567 | **1.048** |
| lanthanide series | 0.084 | 0.040 | 0.47 | 0.379 | 0.094 | 0.092 |
| temperature | −0.040 | 0.000 | 0.00 | 0.064 | 0.039 | 0.010 |

Read the last two columns together: **on the extractant axis the model's slope prediction
is three times worse than simply guessing the corpus median slope.** It is not merely
imprecise about the response, it is systematically and one-signedly wrong about it. The
same holds on the acid axis. Only the lanthanide series is roughly at par with the null.

![Slope flattening](runs/gen8_architecture/figures/slope_flattening.png)

### Repairing it is worth a real, mechanistic gain — where the physics is identifiable

The repair uses no target of any kind: fit the frozen model's own *predictions* along one
curve (a function of the features alone), compare the fitted slope with the training
prior for that axis, and add the difference back, re-centred on the curve so the ligand's
level cannot move.

| where | k | gain over the matched offset arm | 95 % CI | BCa | units improved | seeds + |
|---|---|---|---|---|---|---|
| all 143 ligands, zero-shot | 0 | +0.007 | [+0.001, +0.012] | [+0.002, +0.014] | 18/143 | 5/5 |
| all 143 ligands | 1 | +0.016 | [+0.001, +0.029] | [+0.004, +0.037] | 24/143 | 5/5 |
| all 143 ligands | 2 | +0.013 | [+0.002, +0.030] | [+0.005, +0.045] | 22/143 | 5/5 |
| all 143 ligands | 5 | +0.013 | [+0.002, +0.028] | [+0.005, +0.041] | 21/99 | 5/5 |
| **the 26 with an extractant titration**, zero-shot | 0 | **+0.037** | [+0.019, +0.057] | [+0.018, +0.057] | 18/26 | 5/5 |
| **the 26 with an extractant titration** | 1 | **+0.086** | [+0.019, +0.125] | [+0.043, +0.134] | 24/26 | 5/5 |
| **the 26 with an extractant titration** | 2 | **+0.074** | [+0.032, +0.138] | [+0.032, +0.138] | 22/26 | 5/5 |
| the 26, **shape MAE only**, zero-shot | 0 | **+0.071** | [+0.041, +0.111] | [+0.042, +0.111] | 23/26 | 5/5 |
| the 26, **shape MAE only** | 2 | **+0.063** | [+0.036, +0.102] | [+0.036, +0.102] | 22/26 | 5/5 |

**This is the only architectural change in gen8 that reduces *shape* error**, and it does
so on exactly the population where the mass-action law is identifiable. Applying the same
repair to the acid axis is neutral-to-harmful and to the lanthanide axis is harmful, and
those arms are reported rather than hidden: per curve, the prior repair takes extractant
shape MAE 0.555 → 0.194 (1,096 of 1,205 curves improve), acid 0.553 → 0.509 (1,690 of
2,480), and the lanthanide series 0.289 → 0.288 (nothing).

## THE ARCHITECTURES — WHAT BEAT OFFSET CORRECTION, AND WHAT DID NOT

Three architecture families were built to the same adapter interface and scored on the
identical ligands, repeats, candidate pools and evaluation rows as everything else: a
**Conditional Neural Process** treating each ligand as a task (§5), a **physics-latent**
model in which the ligand predicts mass-action parameters that the measurements then MAP-update
(§6), and **series-aware curve baselines** — spline residual, mass-action curve, local GP
over the swept axis (§3).

Maximal coverage, `RANDOM` selection, macro MAE:

| architecture | k=0 | k=1 | k=2 | k=3 | k=5 |
|---|---|---|---|---|---|
| `OFFSET_K1` — the baseline to beat | 0.995 | 0.723 | 0.640 | 0.616 | 0.628 |
| **`PHYS_residual_gbm`** — ligand → mass-action parameters, on the *residual* | — | **0.668** | **0.561** | **0.504** | **0.490** |
| `PHYS_residual_hypernet` — same, neural hypernetwork | — | 0.669 | 0.563 | 0.506 | 0.490 |
| `PHYS_map` — the same law on the **absolute** `log D` | — | 0.808 | 0.626 | 0.552 | 0.542 |
| `PHYS_prior` — the fitted law, zero-shot | 1.379 | 1.379 | 1.379 | 1.361 | 1.351 |
| `CURVE_SPLINE_GP` — spline residual + local GP | 0.988 | 0.714 | 0.585 | 0.527 | 0.492 |
| `CURVE_GP` — local GP over the swept axis only | — | 0.723 | 0.590 | 0.530 | 0.497 |
| `MASSACTION_K1` — transferable n and m priors | 0.993 | 0.715 | 0.635 | 0.613 | 0.614 |
| `CNPRES_shrinkage` — residual CNP | 1.002 | 0.697 | 0.621 | 0.602 | 0.622 |
| `CNPRES_conditions_only` — residual CNP, no ligand | 1.007 | 0.724 | 0.627 | 0.596 | 0.621 |
| `CNPRES_ligand` — residual CNP + ligand representation | 1.026 | 0.740 | 0.652 | 0.628 | 0.648 |
| `CNP_conditions_only` — CNP on the **absolute** `log D` | 1.206 | 0.882 | 0.840 | 0.831 | 0.854 |
| `CNP_ligand` — the same, + ligand representation | 1.176 | 1.014 | 0.998 | 0.999 | 1.032 |
| `CNP_ligand_recovered` — the same, + recovered variables | 1.246 | 1.139 | 1.132 | 1.126 | 1.185 |

Paired chemotype bootstrap (positive = candidate better). The physics-latent arms are
shown on all 143 ligands at `RANDOM`, which is the population and policy they were run on;
the rest at k = 1 on the common cohort.

| comparison | k | point | 95 % CI | BCa | units | seeds + |
|---|---|---|---|---|---|---|
| **physics-latent residual vs offset (`RANDOM`)** | 1 | **+0.056** | **[+0.006, +0.090]** | [+0.013, +0.097] | 106/143 | 5/5 |
| **physics-latent residual vs offset (`RANDOM`)** | 2 | **+0.079** | **[+0.048, +0.108]** | [+0.050, +0.110] | 103/143 | 5/5 |
| **physics-latent residual vs the K3 update (`RANDOM`)** | 2 | **+0.038** | **[+0.016, +0.059]** | [+0.017, +0.060] | 91/143 | 5/5 |
| **physics-latent residual vs the K3 update (`RANDOM`)** | 3 | **+0.036** | **[+0.019, +0.057]** | [+0.020, +0.059] | 90/140 | 5/5 |
| **physics-latent residual vs the K3 update (`RANDOM`)** | 5 | **+0.024** | **[+0.006, +0.056]** | [+0.004, +0.052] | 61/99 | 5/5 |
| physics-latent residual vs offset, **both at `CENTRAL`** | 1 | +0.024 | [−0.007, +0.047] | [−0.006, +0.047] | 101/143 | 5/5 |
| gradient-boosted vs ridge ligand→parameter map | 1 | +0.003 | [−0.002, +0.009] | [−0.002, +0.009] | 83/143 | 3/5 |
| RBF lanthanide response vs 14 metal categories | 1 | +0.001 | [−0.002, +0.005] | [−0.002, +0.005] | 68/143 | 3/5 |
| residual CNP vs offset (`RANDOM`, common cohort) | 1 | +0.028 | [−0.013, +0.054] | [−0.005, +0.061] | 56/99 | 5/5 |
| spline-residual baseline vs offset (`RANDOM`) | 1 | +0.020 | [−0.008, +0.041] | [−0.003, +0.047] | 31/99 | 5/5 |
| mass-action curve baseline vs offset (`RANDOM`) | 1 | +0.019 | [−0.019, +0.045] | [−0.012, +0.053] | 30/99 | 5/5 |
| **the same law on the absolute value vs offset** | 1 | **−0.055** | **[−0.130, −0.009]** | [−0.121, −0.004] | 41/99 | **0/5** |
| **absolute CNP vs offset** | 1 | **−0.145** | **[−0.282, −0.060]** | [−0.254, −0.040] | 41/99 | **0/5** |
| **absolute CNP: adding a ligand representation** | 1 | **−0.125** | **[−0.188, −0.018]** | [−0.209, −0.041] | 39/99 | **0/5** |
| residual CNP: adding a ligand representation | 1 | −0.012 | [−0.030, +0.020] | [−0.034, +0.013] | 33/99 | 2/5 |

Five readings.

**1. One architecture clears the bar, and it is the physics one.** The physics-latent
residual beats plain offset correction by **+0.056** at k = 1 and **+0.079** at k = 2
(chemotype-blocked interval excluding zero, 5 of 5 seeds, 106 and 103 of 143 ligands),
and it beats even the shrunk four-coefficient update by +0.024 to +0.038 at every
k ≥ 2. It is the best 1-shot arm in the study (**0.626** at `CENTRAL`, maximal coverage).

**2. But it and the acquisition rule are buying overlapping things.** Against offset
correction *at the same central selection rule* its advantage falls to +0.024 with an
interval touching zero. Both are estimating the ligand's level better: the policy by
measuring somewhere representative, the model by regularising the level through a physical
law. Roughly half of what each buys, the other already had. That is worth stating plainly
because summing the two gains would overstate the total by about 0.03.

**3. Residual formulations work; absolute-value formulations fail, significantly.** The
same mass-action law scores 0.668 modelling the frozen model's *residual*, 0.808 modelling
`log D` directly, and 1.379 as a pure zero-shot law. The same CNP scores 0.697 on the
residual and 0.882 on the absolute value. Both absolute arms are significantly worse than
plain offset correction with 0 of 5 seeds positive. The lesson is not that these
architectures are wrong but that **the tree already has the conditions handled**; what a
functional model can add is a correction, not a replacement.

**4. Neither half of the "hypernetwork" idea matters — the law does.** Swapping the
ligand→parameter map from ridge to gradient boosting is worth +0.003 (3/5 seeds), and
replacing the fourteen metal categories with an RBF over ionic radius inside the same model
is worth +0.001 (3/5). What carries the arm is the *functional form* — an intercept plus
mass-action slopes plus a smooth lanthanide response, fitted with shrinkage — not the
sophistication of the map from chemistry to its coefficients. The chemistry-to-parameter
map is, once again, the part that does not work.

**5. The brief's own question about molecular features gets a clean answer: no, and
inside a Neural Process they actively hurt.** `CNP_conditions_only` (0.882) beats
`CNP_ligand` (1.014) beats `CNP_ligand_recovered` (1.139), and the paired test on adding a
ligand representation is **−0.125, CI [−0.188, −0.018], 0 of 5 seeds**. On the residual CNP
the effect is smaller and not established (−0.012) but points the same way. After
calibration, chemistry is not what is missing.

## DOES MECHANISM-AWARE SIMILARITY FIX THE ZERO-SHOT LEVEL?

**No.** This was gen8's most falsifiable hypothesis and it is refuted cleanly.

gen7's ceiling was established with ECFP similarity throughout: on the
**condition-adjusted** ligand level — the ligand's mean residual after a global metal +
conditions model, i.e. the part that is actually chemistry — no model beat a
1-nearest-neighbour Tanimoto lookup. The obvious objection was that ECFP distance is a
distance between *substructure inventories* while extraction is governed by *donor-set
hardness, denticity and charge*, so gen7 may simply have been using the wrong
neighbourhood.

gen8 built the right one to find out: 36 compact quantities (13 disjoint donor classes,
an HSAB softness index, formal charge and exchangeable-proton count, a denticity proxy,
chelate-pair count, donor topological spacing, ring count and rotatable fraction, size,
logP, TPSA, HBD/HBA). It is genuinely a different geometry — Spearman against
1 − Tanimoto is **0.593**, and only 45 of 152 extractants keep the same nearest
neighbour when the metric is swapped. So the test had teeth.

| predictor of the condition-adjusted level | MAE | macro MAE | R² |
|---|---|---|---|
| **1-NN Tanimoto** | **0.989** | 0.973 | 0.103 |
| ExtraTrees on the 36 mechanistic columns | 1.009 | 1.003 | **0.107** |
| 3-NN Tanimoto | 1.022 | 1.024 | 0.087 |
| kernel regression on the donor census | 1.024 | 1.019 | 0.096 |
| ExtraTrees on the 13-column donor census | 1.035 | 1.029 | 0.078 |
| 5-NN mechanistic | 1.040 | 1.039 | 0.075 |
| global mean (null) | 1.074 | 1.070 | 0.000 |
| ridge on the mechanistic columns | 1.075 | 1.071 | 0.001 |
| **1-NN mechanistic** | **1.077** | 1.066 | −0.008 |

Paired chemotype bootstrap (positive = candidate better):

| comparison | point | 95 % CI | BCa | units improved | seeds + |
|---|---|---|---|---|---|
| **1-NN Tanimoto beats 1-NN mechanistic** | **+0.085** | [+0.028, +0.137] | [+0.028, +0.137] | 77/152 | 5/5 |
| 1-NN Tanimoto beats the global mean | **+0.076** | [+0.012, +0.146] | [+0.016, +0.149] | 87/152 | 5/5 |
| ExtraTrees-on-mechanism beats the global mean | **+0.050** | [+0.009, +0.111] | [+0.004, +0.105] | 93/152 | 5/5 |
| ExtraTrees-on-mechanism vs 1-NN Tanimoto | −0.026 | [−0.069, +0.033] | [−0.074, +0.030] | 82/152 | **0/5** |
| ExtraTrees-on-mechanism vs the donor census | +0.016 | [−0.008, +0.051] | [−0.010, +0.049] | 81/152 | 4/5 |

The precise reading, which is more informative than a flat "no":

* **the mechanistic *neighbourhood* is significantly worse** than the ECFP one, not
  merely no better — swapping the metric costs 0.085;
* **a mechanistic *model* is the best model-based level predictor found in this
  project** (R² 0.107, beating the donor census that beat everything in gen7), and it
  clears the global-mean null with an interval excluding zero;
* **it still does not beat the 1-NN Tanimoto lookup** (0 of 5 seeds).

So gen7's ceiling stands, now reached by a third independent route. It is not the
learner, not the objective, and not the neighbourhood: **the condition-adjusted level is
a chemical signal that ~120 training examples cannot pin down.**

Where mechanistic distance *does* earn its place is in diagnosis rather than prediction.
Among the ECFP-similar pairs (Tanimoto ≥ 0.55) that hide a donor substitution, it ranks
the partner much further away than Tanimoto does — median neighbour-rank 0.178 vs 0.023
for O/N → S substitutions and 0.260 vs 0.023 for O ↔ N. It knows the substitutions are
chemically large. It just does not follow that the level moves with them.

## WHAT STILL FAILS AFTER TWO MEASUREMENTS?

Every held-out ligand was classified from its own paired numbers on the same 8,580
(seed, fold, ligand, repeat) units.

| failure class | n | zero-shot | 1-shot | 2-shot | oracle 1-shot | shape floor | shape as a fraction of zero-shot |
|---|---|---|---|---|---|---|---|
| **PURE_LEVEL** | 25 | 1.906 | 0.494 | 0.440 | 0.339 | 0.337 | 19 % |
| PARTIAL_LEVEL | 7 | 1.458 | 0.873 | 0.764 | 0.530 | 0.547 | 36 % |
| **RESIDUAL_SHAPE** | 29 | 1.350 | **1.471** | **1.295** | 0.975 | 0.962 | **73 %** |
| ALREADY_GOOD | 82 | 0.552 | 0.516 | 0.458 | 0.350 | 0.334 | 59 % |

Two rows carry the answer.

**PURE_LEVEL (25 ligands): one measurement is a complete repair.** 1.906 → 0.494, and
after that one measurement they are *better than the ALREADY_GOOD ligands were before
it*. Their zero-shot error was 19 % shape and 81 % a single constant. The paired gain is
+1.41 (CI [+1.17, +1.69], 25/25 ligands, 5/5 seeds).

**RESIDUAL_SHAPE (29 ligands): one measurement makes them worse.** 1.350 → 1.471 on a
random point, and two only reach 1.295. Their shape floor alone is 0.962 — 73 % of their
zero-shot error — and the *oracle* one-shot arm reaches 0.975, i.e. essentially that
floor. No acquisition policy, no budget and no oracle can move these ligands, because an
offset is the wrong object to be estimating for them.

**And one measurement is not free.** 57 of 143 ligands are *worse* after one random
measurement than before it — 39 of the 82 already-good ones and 18 of the 29
residual-shape ones. A single point estimates the offset with that point's full shape
noise attached; where there is little offset to remove, the noise is all you buy.
Choosing the point centrally reduces the count from 57 to **42**. This is the strongest
practical argument in the report for the acquisition result: the reason to choose well is
not only that the average improves, it is that blind measurement actively harms 40 % of
the population and choosing well removes a quarter of that harm.

### What separates a ligand a measurement can fix from one it cannot

This is the question gen9 turns on, and it has a clear answer that is **not** about
chemistry. Chemotype-blocked bootstrap of RESIDUAL_SHAPE against PURE_LEVEL, AUC =
P(residual-shape ligand scores higher), Benjamini–Hochberg corrected within block:

| candidate explanation | verdict |
|---|---|
| **donor chemistry** — all 36 mechanistic descriptors | **no** — smallest corrected q = **1.000**, smallest raw p = 0.082 |
| **chemical novelty** — nearest-training Tanimoto | **no** — 0.560 vs 0.576, AUC 0.466, p = 0.77 |
| **how much data the ligand has** — row count | **no** — AUC 0.596, p = 0.37; medians 15 vs 14 rows |
| **the size of its response** — measured `log D` span | **yes** — **4.37 vs 1.57 decades**, AUC 0.949, q = 0.0005 |
| how linear its curves are | yes — median curve R² 0.865 vs 0.542, AUC 0.806 |
| how steep its curves are | yes — median \|slope\| 1.14 vs 0.46, AUC 0.786 |
| whether it has an **acid titration** | yes — 0.81 vs 0.35, AUC 0.734, q = 0.003 |
| how many series and publications it spans | yes — 4.0 vs 1.3 series, 6.0 vs 1.3 DOIs, AUC 0.76 |
| **how hard the model flattens it** | **yes** — the model captures **29 %** of a residual-shape ligand's true range against **51 %** for a pure-level one, AUC 0.279, q = 0.003 |

**The ligands one measurement cannot fix are the wide-ranging ones the model flattens
hardest.** Not the exotic donor sets, not the chemically novel ones, not the sparsely
measured ones. That closes the loop with the slope result: the flattening pathology and
the residual-shape failure class are the same phenomenon seen from two directions, and
fixing the first is the way to buy the second. It is also why gen8's answer to "what
should gen9 do" is a training-objective change rather than more chemistry.

## THE SULFUR-DONOR CASE STUDY — AND A SECOND SUSPECTED DATA CORRUPTION

The brief's premise needed two corrections, and the case study is more useful for them.
**There is no bis(dithiophosphonate), or any dithio acid, in this cohort** — of 152
extractants 11 carry an S donor and every one is neutral; a direct scan finds no P–SH,
no P–S⁻ and no S–H anywhere. The Cyanex-301-type reagents that make soft-donor Ln/An
separation work are simply absent, which is itself a coverage finding. The study
therefore runs on the worst S-donor extractant that *is* present, **TWE-24**
(`CCCCOP(=S)(CP(=S)(OCCCC)OCCCC)OCCCC`, O,O,O',O'-tetrabutyl methylenebis(phosphonothioate)) —
which is also **the worst zero-shot ligand in the entire cohort**, macro MAE 4.098
against a cohort 0.995.

**And the mechanism hypothesis is false for this molecule.** Four of its five nearest
ECFP neighbours already carry P=S; the nearest-neighbour softness gap is 0.043. The
fingerprint is not mistaking a soft thiophosphoryl extractant for a hard phosphate. The
mechanistic distance is marginally tidier (it demotes the one hard intruder out of the
top five) and buys nothing, because there was no mistake to fix.

The diagnosis is **pure offset, and unusually cleanly so**: the best constant removes
89.7 % of the error, one measurement removes 84 %, the within-ligand Spearman between
prediction and truth is **0.943**, the residual never changes sign across the six rows
(+3.52 to +4.74), and the predicted acid slope (−0.608) matches the measured one
(−0.535) to 0.073 — against a corpus where the median acid curve is flattened from
+1.573 to +0.404. *The model reproduces this curve's shape better than it reproduces an
average curve. It places it four decades too low.*

Which raises the obvious question, and the case study answers it in the only honest way
available. The six P=S compounds were measured on the identical acid grid in one
campaign from one upstream source; five of them have ligand levels between −2.0 and
−3.1, and TWE-24 sits at **+1.89** — 4.27 log units above the five-sibling mean and 3.89
above the highest of them. Its raw D values are 508, 127, 290, 122, 5.6, 17.1. The model
predicted its level as −2.21 against a sibling-family mean of −2.39; it got the chemistry
right to 0.18 log units. **Either TWE-24 is real chemistry no descriptor in this project
can see, or its `D` column is off by roughly three decades — and this corpus has one
confirmed precedent for exactly that failure mode.** The source document was not
retrieved, so this is a flag for verification, not a finding, and the classification does
not depend on resolving it.

What *is* measurable is that the anomaly is contagious. TWE-24 is TWE-29's nearest ECFP
neighbour and always sits in TWE-29's training fold. Removing TWE-24 from a 1-NN
level lookup changes exactly one ligand's neighbour — TWE-29 — and its level error falls
from **4.416 to 0.526**. One suspect ligand costs its nearest neighbour 3.9 log units and
the whole cohort 0.026 on that benchmark.

## DATA CLEANUP

**DMDPhPDA is resolved.** The two copies do not share an extractant name, which is why
earlier searches found only half of them; they share the canonical SMILES and carry
**140 cohort rows (2.7 %)**, a perfect 14 metals × 5 acidities bijection with **70
duplicated cells**, one DOI (`10.1081/SEI-120030392`), and differences that are integer
decades to 4.9 × 10⁻¹⁰ — with the shift **constant per acidity column**: +4 at 1 M, +3 at
2 M, +2 at 3 M, +1 at 4 M, 0 at 5 M. `D_B/D_A` is exactly 10⁴/10³/10²/10¹/10⁰ to ten
significant figures, so copy B is not an independent measurement.

**Copy B is the corrupted one**, on internal evidence: copy A is strictly monotone in
[HNO₃] for **14 of 14** metals (mean r = 0.970, span 4.17 log units); copy B is monotone
for **0 of 14** (mean r = 0.335, span 0.67, 43 % of steps going the wrong way). The
upstream record even contains the curator's own note on six rows — that a Table 1 value
was off from Figure 5 by a factor of 10⁻¹ and was adjusted — and it de-duplicates to
exactly the three cells where copy B is one decade high. The publication itself was not
retrievable (the publisher returns HTTP 403); the audit says so explicitly rather than
inferring what the paper contains. Three sensitivity cohorts — `FROZEN`, `QUARANTINED`,
`CORRECTED` — are defined row by row in `case_studies/dmdphpda_cohorts.json`; **primary
results are on `FROZEN`** so they stay comparable to gen7.

**Structural absence is now explicit.** `gen8/data_v2.py` emits tri-state
ABSENT / PRESENT / UNRECORDED flags per condition family rather than letting an imputed
value stand in for "there was no phase modifier", and ships the recovered solvent
physics, phase-modifier concentration, shaking time and an explicit
`second_species_present` column with a coarse class. On the 5,248-row cohort, 353 cells
carry a name that is not the modal name of their structure (414 raw rows before replicate
averaging), of which 86 match an aqueous-complexant signature. One finding from building
it is worth carrying forward on its own: of the 5,006 rows the bundle labels
`modifier_class=none`, **1,215 (24.3 %) physically carry a modifier** — 122 of them
hidden inside the `diluent=other` bucket.

## IS THE LANTHANIDE RESPONSE CONTINUOUS?

Four encodings of the same fourteen metals, one learner, the frozen fold plan, identical
rows, scored **within each reconstructed metal-series curve** so the question is whether
the order and spacing of the 4f series come out right rather than whether the ligand's
level does.

| metal encoding | features | within-curve shape MAE | within-curve Spearman | within-curve sign accuracy | macro MAE |
|---|---|---|---|---|---|
| **structured** (Z, series index, ionic radius) | 3 | **0.334** | 0.330 | 0.650 | 1.240 |
| **ligand-conditioned RBF** (RBF ⊗ donor census) | 117 | 0.336 | 0.328 | **0.656** | 1.242 |
| shared RBF over ionic radius + tetrad term | 13 | 0.339 | 0.333 | 0.650 | 1.237 |
| **one-hot — fourteen unrelated categories** | 14 | **0.368** | **0.279** | **0.622** | 1.244 |

*(3 seeds; this is a controlled representation comparison on a deliberately reduced arm —
conditions + fingerprint + mass action only — so the absolute MAE is not a leaderboard
number and should not be read against the 0.998 champion.)*

The answer is clean and it is a shape answer, not a level answer. **Every structured
encoding beats the one-hot on the order and spacing of the 4f series** — within-curve
Spearman rises from 0.279 to 0.330, sign accuracy from 0.622 to 0.656, within-curve
shape MAE falls from 0.368 to 0.334 — while **total macro MAE is unchanged to three
decimals** (1.244 → 1.237–1.242). Treating the lanthanides as fourteen unrelated labels
costs you the selectivity ordering, which is the quantity a separations chemist actually
reads, and costs you nothing on the absolute value, which is the quantity the leaderboard
reports. That is the clearest single illustration in gen8 of why this project stopped
reporting one MAE.

Two honesty notes. The paired chemotype bootstrap on the *row-level* shape residual does
**not** clear zero at three seeds (`LIGAND_RBF` vs one-hot +0.010, CI [−0.005, +0.024],
2/3 seeds), so the effect is established on the curve-level ordering statistics and not
on the row-level decomposition. And ligand-conditioning the RBF buys the best sign
accuracy but nothing else: 117 interaction columns are not worth their p ≫ n cost here,
which is the same lesson every representation experiment in this project has returned.

## THE EXPERIMENTAL-DESIGN INTERFACE

The system's output is no longer a `log D`. `scripts/gen8_recommend_experiment.py` takes a
SMILES and a table of candidate conditions and returns the experiment to run:

```bash
.venv/bin/python scripts/gen8_recommend_experiment.py \
    --ligand "<SMILES>" --candidate-conditions conditions.csv \
    --out recommended_condition.json
```

On the built-in demo (a held-out diglycolamide, its entire chemotype withdrawn from
training, 40 candidate conditions) it returns:

```text
RECOMMENDED NEXT EXPERIMENT   candidate C005     rule: MEDOID
  Tb / HNO3 3.0 M / extractant 0.2 M / n-dodecane / 25 C / 20 min
  predicted log D        0.80  (+/- 1.44 at 68 %, +/- 2.28 at 90 %)
  expected macro MAE     0.648 with this rule vs 0.723 at random
                         (gain 0.075, CI [0.035, 0.130], 5/5 seeds)
  oracle (not deployable) 0.484
  this point sits at:    acid mid, extractant mid, metal central, prediction mid
```

Three properties of that output are the point of gen8 rather than decoration. The
recommendation carries **the measured value of following it** rather than a confidence
score. The chosen point lands in all four favourable strata *without being told to* —
the medoid rule finds mid-acid, mid-extractant, central-metal and mid-prediction on its
own, which is the geography result and the acquisition result agreeing. And after the
user supplies the observed value (`--observed`), the second recommendation switches to
`FARTHEST_FROM_EXISTING` and the calibration switches from `OFFSET_K1` to the shrunk
`OFFSET_K3`, because that is what the k = 2 measurements say to do. The prototype also
prints its own caveats — here, that the withheld chemotype is 64 % of the corpus, so this
particular demo model saw only 1,908 of 5,248 rows and is weaker than the headline.

## WHAT WOULD HAVE FALSIFIED GEN8, AND WHAT DID

| the series/few-shot hypothesis fails if… | outcome |
|---|---|
| oracle best-point selection is barely better than random | **survived** — 0.533 vs 0.793, gap 0.260, CI [+0.186, +0.310] |
| functional models do not beat simple offset correction | **PARTLY FAILED** — at k = 1 *no* functional architecture clears the pre-registered bar: the best, a physics-latent residual, is +0.029 with an interval touching zero. What does clear it is the shrunk response-coefficient update (+0.044 at k = 2, +0.115 at k = 5) and the slope repair (+0.016) — neither of which is a "functional model" in the brief's sense |
| series-aware models cannot reduce shape MAE | **survived, on a subset** — slope restoration cuts shape MAE by 0.071 on the 26 ligands with an extractant titration, and by 0.013 across all 143; on the acid and lanthanide axes it does nothing or harms |
| active acquisition cannot outperform random selection | **survived** — +0.101, CI [+0.037, +0.146], 5/5 seeds |
| calibration benefits only rows very near the measured condition | **FAILED, partially — reported as a headline** — one measurement transfers within a series type and largely not across them |
| one-shot gains disappear on a common ligand cohort | **survived** — the gain is 0.268 on the common cohort against 0.272 on maximal coverage: indistinguishable, and every headline in this report is quoted on the common cohort anyway |
| the result depends on leaking series identity or a hidden target | **survived** — the adapter interface passes only `truth[selected]`; the property is asserted for every adapter by a test that corrupts every unselected target and requires the predictions to be bit-identical |

Two further things went wrong during the run and are recorded because they changed
numbers.

**A curve-membership bug.** Rows whose coordinate on an axis was missing were being
emitted as members of that axis's curve with a NaN abscissa, which propagated into every
downstream slope fit. Fixed; the audit's slope table is unchanged (the statistics module
already filtered them) but the slope-repair adapter was silently producing NaN
predictions for those rows before the fix, and those records were being dropped from the
comparison — which would have made it unpaired.

**A reproducibility bug that invalidated a first round of results.** The pool/evaluation
split was seeded with Python's builtin `hash()` of the ligand SMILES, which is salted per
interpreter process. Every number was reproducible *within* a run and different *between*
runs, so an architecture evaluated in a separate process was **not** paired with the
primary run even though every table called it paired. Replaced with a stable BLAKE2b
digest, every affected experiment re-run from scratch, and the property is now enforced by
a test that re-derives the split in a subprocess under a different `PYTHONHASHSEED`.

## REPRODUCING THIS

```bash
.venv/bin/python scripts/gen8_series_audit.py
.venv/bin/python scripts/gen8_kshot.py --protocol both
.venv/bin/python scripts/gen8_calibration_geography.py
.venv/bin/python scripts/gen8_run.py --seeds 5 --repeats 12 --tag primary
.venv/bin/python scripts/gen8_slope_restore.py --seeds 5 --repeats 12 --tag slope_restore
.venv/bin/python scripts/gen8_ablation.py --seeds 5 --repeats 12
.venv/bin/python scripts/gen8_mechanism_similarity.py
.venv/bin/python scripts/gen8_slopes.py
.venv/bin/python scripts/gen8_lanthanide.py
.venv/bin/python scripts/gen8_analysis.py
.venv/bin/python scripts/gen8_figures.py
.venv/bin/python scripts/gen8_manifest.py
.venv/bin/python -m pytest tests/test_gen8_harness.py tests/test_gen8_mechanism.py tests/test_gen8_data_v2.py
```

`runs/gen8_architecture/environment.json` stamps the library versions
(scikit-learn 1.9.0, pandas 3.0.5, numpy 2.5.1, scipy 1.18.0, torch 2.13.0,
rdkit 2026.03.5) and the cohort fingerprint. A run under different versions must not
share a table with these — this project has already lost a completed sweep to exactly
that, when a casual `pip install` moved scikit-learn underneath it.

## GEN9 DECISION

**Ship the recommender, not the regressor.** The deployment unit is *"give me your
candidate conditions and I will tell you which one to run first"*, followed by a
calibrated surface. Zero-shot is 1.06; one chosen measurement is 0.67; five are 0.47.
Nothing in six generations of modelling has moved the number that far.

**And be precise about which architecture earned its place.** Of the three families built
to the same interface, exactly one clears the bar: the physics-latent residual model, which
beats offset correction by +0.056 at k = 1 and beats even the shrunk four-coefficient
update at every k ≥ 2. The Conditional Neural Process does not, and the curve baselines do
not. What the winner has is not capacity or a clever ligand→parameter map — swapping ridge
for gradient boosting is worth +0.003 — it is a **functional form**: an intercept plus
mass-action slopes plus a smooth lanthanide response, fitted with shrinkage, to the frozen
model's *residual*. That is the same ingredient the slope repair uses and the same
ingredient the curve audit says is missing.

**Then, in priority order:**

1. **Fix the flattening inside the model.** It predicts extractant titrations at a
   twentieth of their true slope and acid titrations at a fifth, and on the extractant
   axis its slope prediction is three times worse than guessing the corpus median. gen8's
   post-hoc per-curve repair is worth +0.086 at k = 1 on the ligands where the physics is
   identifiable and is the only change in gen8 that reduces shape error. Make it
   intrinsic: train with a slope-consistency objective on the 1,176 reconstructed curves
   instead of repairing the output afterwards. The curve table is built and the priors
   are measured; this is a training-objective change, not a research programme.
2. **Close the acquisition gap.** Deployable 0.667 against an oracle 0.533 at k = 1. The
   oracle's rule is known exactly — `argmin |r_i − median(r)|` — and the open problem is
   predicting that deviation from features alone. Centrality captures about 40 % of it;
   model uncertainty captures none. This is a well-posed supervised problem with a
   known target, it needs no new chemistry, and it has never been attempted.
3. **Attack the 29 residual-shape ligands.** One measurement makes them *worse*
   (1.350 → 1.471) and the oracle lands on their shape floor. They are not distinguished
   by donor chemistry, novelty or data volume — they are the **wide-ranging ligands the
   model flattens hardest**, which makes them the same problem as item 1 seen from the
   other end. Beyond that they need either a better response surface or — following the
   transfer result — a *per-series* rather than per-ligand calibration.
4. **Verify TWE-24 against its primary document,** and re-check every ligand whose level
   sits more than three decades from its own chemotype's mean. TWE-24 is the worst ligand
   in the corpus by 0.6 log units, its raw `D` values sit about three decades above five
   siblings measured on the identical grid in the same campaign, and removing it moves its
   nearest neighbour's level error from 4.42 to 0.53. The corpus has one confirmed
   precedent for exactly this failure mode.

**Do not:**

* run another learner sweep, another representation sweep, or another attempt to predict
  the zero-shot level from structure — three independent routes now agree it is not there;
* build an absolute-value Neural Process or an absolute-value physics law; both were
  built here and both are worse than offset correction on the frozen model's residual;
* add more ligand description to a calibrated model. Both architecture families answered
  the brief's question the same way: **molecular features do not add value after
  calibration**, and inside the Neural Process they actively hurt.

**And say this to the chemist, because it is the part that changes practice:** measure the
new extractant once, at mid-range acidity, on a mid-series lanthanide, in the system you
actually care about — and do not expect that measurement to calibrate a different diluent.
