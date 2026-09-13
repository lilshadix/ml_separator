# Gen13 stage 3 — what the donor set decides, and nine things that do not move the error

Stage 2 measured that 82–87 % of the pairwise error of every Gen13 arm is error in one number per
cell — the coefficient of the standardised Shannon radius — and that only the compact chemistry
blocks transfer across laboratories.  Stage 3 attacks that number directly.

Everything here uses a new bench, `gen13sep/amplitude_bench.py`, which freezes the cohort, the fold
plans, the physics basis, the per-cell ridge and the extractant-macro scoring, and exposes a single
pluggable function so that a candidate is a few lines rather than an arm class.  It reproduces the
locked `M_PHYSICS_radius+radius_sq` arm to four decimal places on every metric (0.4979 / 0.5425 /
0.6572 / 0.7440 / 0.4912) and evaluates a new candidate over 25 folds in 40 seconds instead of
minutes, which is what made a search of this size possible.

**Every candidate is run under at least three hold-out designs.**  That rule was adopted after the
first candidate of this stage looked like a winner under one design and lost under four; at the
effect sizes available here a single-design result is not evidence.

---

## 1. The result: donor topology decides which end of the series an extractant prefers

The target is the *direction* of selectivity — whether a cell's radius coefficient is negative
(heavy-lanthanide selective) or positive.  It is scored out of fold on cells whose curve is well
determined (≥ 5 measured metals): 289 cells, 82 extractants, 40 chemotypes, 70 % of cells and 55 %
of extractants heavy-selective.  The unit of scoring is the extractant, the unit of resampling is
the frozen chemotype, and the bootstrap uses one shared set of chemotype resamples for every model
so the differences are paired.

| design | donor topology (39 columns) | full compact set (209 columns) | gain over always-heavy | p | extractants better / worse |
|---|---|---|---|---|---|
| B | 0.822 [0.710, 0.897] | 0.767 | **+0.263** [+0.105, +0.503] | 0.0004 | 28 / 6 |
| BR | 0.816 [0.701, 0.892] | 0.766 | **+0.257** [+0.099, +0.497] | 0.0006 | 28 / 7 |
| BQ | 0.814 [0.695, 0.891] | 0.761 | **+0.256** [+0.100, +0.498] | 0.0008 | 27 / 8 |
| A | 0.798 [0.678, 0.872] | 0.804 | **+0.240** [+0.088, +0.477] | 0.0020 | 28 / 8 |
| BP | 0.768 [0.627, 0.848] | 0.721 | **+0.210** [+0.067, +0.425] | 0.0022 | 26 / 7 |

The baseline is always predicting heavy-selective, which scores 0.559 [0.315, 0.715] macro accuracy
in every design (55 % of extractants are heavy-selective when each counts once, although 70 % of
cells are).  Intervals are chemotype-blocked with one shared set of resamples per design, so the
gains are paired.

So for an extractant whose entire chemical family is absent from training, and with every
publication that studied it removed as well, the direction of its lanthanide selectivity is called
correctly about three times in four from the topology of its donor set alone.  The full compact set
does not do better; the difference is inside its own noise, which the audit below states precisely.

The descriptors are counts of bonds on the 2D molecular graph, not geometry
(`gen12_2_eu_pred/COORDINATION_DESCRIPTOR_SPEC.md`), so this is computable for any candidate ligand
from its structure alone, with no conformer, no DFT and no measurement.

### What an independent audit changed

The claim was audited by a second agent that re-derived it from scratch before reading any of the
code above (`analysis/stage3/verify_direction/VERDICT.md`).  The primary result survived; four
things had to be restated, and they are restated here rather than in a footnote.

* **The comparison set is 209 columns, not 137.**  `LEAN_BLOCKS` is conditions + mass action +
  physchem + donors + coordination.  The 137-column set is physchem + donors + coordination alone.
* **Topology is *no worse than* the full set, not better.**  Against the 209-column set the
  difference is +0.047 [+0.0005, +0.096]; against the true 137-column set it is +0.035
  [−0.015, +0.070], p = 0.20.  The honest statement is that 39 topological columns lose nothing.
* **The honest increment of topology is +0.075, not +0.21.**  The 13-column gen6 donor census on its
  own already reaches 0.694, and coordination-minus-topology reaches 0.691.  Topology's increment
  over the donor census is **+0.075 [+0.022, +0.151], p = 0.006** — still real, and about a quarter
  of the gain over the constant baseline.
* **The advantage lives where the selectivity is weak.**  Restricting to cells whose whole-series
  contrast exceeds about 0.6 log units, the gain falls to +0.061 [−0.032, +0.187], p = 0.18.  The
  model is near chance where the label is weak (macro 0.55 below |amplitude| 0.05) and 0.90–0.92
  where it is strong; what collapses in the restricted set is the *baseline*, which becomes hard to
  beat once only strongly directed extractants remain.

Two facts to disclose with the result.  Sixty-one of the 82 extractants contribute a single cell.
And the 39 topological columns take only 22 distinct values across the 82 extractants, so 84 % of
held-out cells have a bit-identical training row: the chemotype label is held out, the feature
vector often is not.  Cells whose topology vector is genuinely unseen still gain +0.396, and 40
label permutations give a null of −0.067 ± 0.062 with a maximum of +0.056, so this is not leakage —
but "the whole chemical family was held out" is a statement about the chemotype label, not about the
representation.

The audit also confirmed what does not break it: the result is unchanged if the label comes from a
plain least-squares fit instead of the ridge (2 labels of 289 move), holds at every metal-count
threshold from 4 to 8, *strengthens* when the dominant diglycolamide chemotype is removed entirely
(0.729 against 0.444, +0.285, p = 0.004), and survives every leakage check — zero extractant, ECFP
cluster, chemotype or publication overlap on the masked folds.  Across four estimators the headline
sits between 0.70 and 0.82.

### The chemistry it recovers

At the extractant level the single strongest correlates of the selectivity amplitude are all
donor-separation descriptors, and their direction is the cavity-size argument:

| descriptor | Spearman with amplitude | chemotype-blocked 95 % CI | p | leave-one-chemotype-out | excluding diglycolamides | controlling for acid |
|---|---|---|---|---|---|---|
| fraction of donor pairs within 3 bonds | **−0.530** | [−0.723, −0.155] | 0.007 | [−0.593, −0.401] | −0.401 | −0.489 |
| shortest donor-to-donor path | **+0.480** | [+0.236, +0.664] | < 0.001 | [+0.434, +0.521] | +0.521 | +0.377 |
| minimum donor eccentricity | +0.509 | [+0.081, +0.687] | 0.016 | [+0.329, +0.570] | +0.329 | +0.484 |
| median donor-to-donor path | +0.500 | [+0.052, +0.679] | 0.026 | [+0.305, +0.561] | +0.305 | +0.480 |

A donor pair *d* bonds apart closes a *(d+2)*-membered chelate ring with the metal, so a short
donor path is a small chelate ring.  Negative amplitude is heavy-selective.  The table therefore
says: **the tighter the chelate bite an extractant can close, the more it prefers the heavy, smaller
lanthanides** — and the relation survives removing the dominant diglycolamide family, aggregation to
chemotype means, and controlling for the acid the measurements were made in.

**No single descriptor is enough.**  A threshold rule on the shortest donor path alone scores 0.451
under BP, *worse* than the majority rule.  The signal lives in the joint topology of the donor set,
not in one number, which is why the 39-column family is the right object and a hand-picked feature
is not.

### What that one bit is worth on the programme's own metric

Take the direction called from donor topology, multiply it by the training fold's mean amplitude
magnitude, and use the corpus curvature.  That model has two degrees of freedom and no knowledge of
the extractant beyond one bit.  Under the publication-masked design, on byte-identical pairs:

| predictor of the held-out curve | extractant-macro MAE | sign accuracy on strong pairs |
|---|---|---|
| true direction × mean magnitude (oracle) | **0.483** | 0.854 |
| trees on the full 209-column compact set | 0.552 | 0.689 |
| **direction called from topology × mean magnitude** | **0.561** | 0.700 |
| corpus mean curve | 0.622 | 0.512 |
| training-majority direction × mean magnitude | 0.715 | 0.389 |

| contrast | point | 95 % interval | p | passes P1 |
|---|---|---|---|---|
| topology direction vs majority direction | +0.153 | [0.054, 0.222] | 0.0002 | yes |
| topology direction vs corpus mean curve | **+0.060** | [0.008, 0.109] | 0.027 | **yes** |
| full 209-column model vs topology direction | +0.009 | [−0.048, 0.048] | 0.76 | no |
| oracle direction vs the full model | +0.070 | [0.024, 0.122] | 0.006 | yes |

Two statements follow, and they are the point of this stage.

**A two-parameter model that knows only which way an extractant separates is statistically
indistinguishable from the full compact regression** (+0.009, p = 0.76).  Everything the
209-column model has learned beyond that one bit is inside its own noise.

**A perfect direction call would beat the full model by 0.070** with the interval excluding zero.
So the direction is not a curiosity on the side of the regression: it is the part of the problem
that is actually learnable from structure, and the regression is currently capturing about half of
it.  The gap between the called direction and the oracle, 0.079, is the concrete target for the next
generation, and it is a classification problem with 82 labelled units rather than a regression
problem with a dozen effective ones.


### It is exactly one bit, and it is distributed across the family

Two checks bound the claim from above.

**Topology carries the direction but not the magnitude.**  Against a three-way magnitude class
(strongly heavy-selective, weak, light-selective, cut at the terciles of the amplitude) the same
donor-topology model scores 0.482 [0.384, 0.643] macro accuracy, the full compact set 0.477, and the
majority class 0.462.  All three overlap.  Whatever structure decides *how strongly* an extractant
separates is not in these descriptors, and probably not in this corpus.

**No single descriptor carries the direction.**  Permuting each of the 39 columns in turn on the
held-out folds, only `coord__dist__donor_pair_min` has an importance whose interval excludes zero
(0.014 [0.003, 0.036], p = 0.008); the largest single importance is 0.022 and every other column is
inside its own noise.  A family worth +0.21 whose best member is worth 0.02 is a redundant,
distributed representation — which is also why the in-fold univariate screens of §2 destroyed the
signal and why a hand-picked threshold rule scores below the majority baseline.


---

## 2. Nine things that do not move the pairwise error

Each was run on the frozen folds under at least two designs and scored with a chemotype-blocked
paired bootstrap.  None passes.  They are listed because each closes a direction that looked
reasonable before it was measured.

| candidate | best result | verdict |
|---|---|---|
| kernel ridge (RBF and linear) on the compact set | 0.583 / 0.677 against 0.552 for trees, BP | much worse |
| Gaussian-process-style linear models, PLS, ridge | 0.651–0.677 | much worse |
| k-nearest-neighbour in the coordination space | 0.581 | worse |
| gradient boosting with an absolute-error loss | 0.628 | worse |
| in-fold univariate screening to 6, 12 or 24 features | 0.587–0.604 | worse than using all of them |
| predicting the amplitude only, curvature from the fold | +0.013 BP but −0.012 B, −0.007 BR, −0.012 BQ, −0.024 A | falsified across designs |
| de-shrinking the ridge-attenuated training target | −0.011 BP, −0.003 B | worse |
| precision weighting by target determinacy | −0.005 BP (p = 0.012), −0.003 B | worse |
| training only on well-determined cells | −0.013 BP, +0.004 B | inconsistent |
| averaging predictions over the cells of one extractant | +0.0005 BP, +0.0002 B | nothing to remove |

Two of these deserve a sentence.

**The estimator is not the problem.**  Extremely randomised trees beat every kernel, linear and
boosting alternative by 0.03–0.13.  The hypothesis that 400 trees on 209 columns are a poor match to
a dozen effective chemotypes is wrong: they are the best match available.

**The training target is attenuated but correcting it hurts.**  A cell's coefficients come from a
ridge fit on the metals it measured, so the estimate is pulled toward zero by
`(B B' + λI)^-1 (B B')`; the median attenuation is 0.873 and a third of the corpus is below 0.8.
De-shrinking widens the target's spread from 0.330 to 0.544, in the direction of the 1.8–2.4×
too-narrow predictions the arms make — and makes the error worse in both designs.  The bias is real
and the noise it hides behind is larger.


---

## 3. What this stage changes

The programme has been treating zero-shot separation as a regression problem on a scalar with about
a dozen effective training units, and it has been stuck at 0.48–0.55 extractant-macro MAE for three
generations of architecture work.  Stage 3 says that framing is the wrong one for this corpus.

* The transferable content of the ligand structure, measured against the honest design, is **one
  bit per extractant**: which end of the series it prefers.  A two-parameter model carrying only
  that bit matches a 209-column regression (p = 0.76) and beats the corpus mean curve with the
  pre-registered rule satisfied (+0.060, p = 0.027).
* That bit is **predictable at 0.77–0.82 from the donor set alone**, replicated across five hold-out
  designs, and the chemistry it recovers is the cavity-size argument: a tighter chelate bite prefers
  the smaller, heavier lanthanides.  Most of the callable signal is already in the 13-column donor
  census (0.694); the topological family adds a further +0.075 [+0.022, +0.151], p = 0.006.  The
  "one bit" statement is about the bit, not about which descriptors produce it — the magnitude
  result below is unchanged whichever of the two calls the direction.
* The magnitude of the selectivity is **not** predictable from the same descriptors, and nine
  distinct attempts to improve the regression side all failed.

The next generation should therefore be built as a classifier with an amplitude prior, not as a
regression: predict the direction from the donor set, take the magnitude from the chemotype-level
prior, and spend any new measurement on the widest pair of the candidate, which stage 2 measured at
0.54 → 0.27.  The concrete target is the 0.079 gap between the called direction and the oracle,
which is a classification problem with 82 labelled units rather than a regression problem with
twelve effective ones.

Two limits to carry with it.  The advantage over the constant baseline concentrates in weakly
directed extractants; among strongly directed ones the constant baseline is already hard to beat
(+0.061, p = 0.18).  And the 39 topological columns take only 22 distinct values across the 82
extractants, so a genuinely novel binding motif is outside anything this corpus has seen — the
representation, not the chemotype label, is what limits how far the classifier can be trusted.
