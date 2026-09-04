# Results text for each main figure

Draft prose, in the order the figures should appear. Each block follows
OBSERVATION → QUANTITATIVE EVIDENCE → INTERPRETATION → LIMITATION. Numbers are the verified
values in `METRIC_AUDIT.md`; the limitation sentences are deliberately explicit about what
the result does *not* establish.

---

## Figure 2 — the value of a measurement

**Zero-shot prediction of log₁₀ D for an extractant with no analogue in the training set is
poor, and a single measurement on that extractant changes the picture more than any model
we built.** On 99 held-out extractants, held out together with every extractant within
Tanimoto 0.7 of them, the frozen pipeline reaches a macro MAE of 1.036 log units with no
measurements at all. One measurement, chosen by a target-blind centrality rule, takes it to
0.654; two to 0.559; three to 0.493; five to 0.441. The first measurement alone removes
0.382 log units (chemotype-block bootstrap BCa 95 % CI [0.264, 0.565], 5/5 split seeds,
67 % of extractants improved), the second 0.095 [0.074, 0.127], the third 0.066
[0.047, 0.085] and the fourth and fifth 0.026 [0.019, 0.034] each.

**For comparison, the entire modelling ladder in this study is worth an order of magnitude
less.** Replacing the previous champion global model with the shape-recomposed one is worth
0.025 [0.016, 0.038] at *k* = 0, and replacing the calibration rule with a
marginal-likelihood series-local prior is worth 0.031 [0.009, 0.048] at *k* = 5; the two
act at opposite ends of the *k* axis (Fig. 2C), and neither approaches the 0.382 that the
first measurement supplies. A model-free baseline that fits only the *k* measurements
plateaus at 0.76, so the model and the measurements are complementary rather than
substitutes — but the marginal product of a measurement is far larger than the marginal
product of a better model. The right unit of deployment is therefore not a prediction; it
is a prediction plus a chosen experiment.

**This does not show that the model is unnecessary, and it does not show that one
measurement is safe.** The measurement-only baseline is 0.76 against the pipeline's 0.654
at *k* = 1, so the model is still carrying most of the burden at small *k*. And the average
hides real heterogeneity: 33 of 99 extractants are *worse* after one measurement than
before it (Fig. 2D), because a single point on an unrepresentative condition drags the
whole calibrated curve. The gap between the deployable rule and an oracle that picks the
support point by reading held-out targets is 0.654 versus 0.504, which is why the choice of
measurement is a separate question (Fig. 6). Finally, *k* here counts measurements on the
*same* extractant in the *same* series; gen8's transfer matrix shows that a point measured
in one solvent system does not calibrate the same extractant in another.

---

## Figure 3 — the flattening, and its repair

**A model can have an unremarkable average error and still get the internal structure of
every held-out titration wrong.** Across 155 extractant-concentration curves from 25
held-out extractants (775 curve × seed evaluations), the measured median slope of log₁₀ *D*
against log₁₀ [extractant] is 2.57 decades per decade; the baseline model predicts 0.12. It
recovers 5.1 % of each curve's measured range, and its within-curve rank correlation with
the measurements is 0.58. In other words the model draws a nearly horizontal line through
the curve's mean — which is, given what it was asked, the mean-absolute-error-optimal
answer.

**The cause is a coordinate-system error, not a capacity limit, and a five-column
representation change fixes most of it.** A control trained *only* on the curve-centred
target — with no level to compete with — recovers 2.5 % of the range, worse than the
monolith, which rules out competition between level and shape. What the model was never
told is where a row sits inside its own measurement window: log₁₀ *D* at an absolute
extractant concentration is dominated by the extractant's level, and different titrations
span different windows, so the conditional mean of the centred response given absolute
conditions alone is close to zero everywhere. Adding the row's relative position, window
width, point count and endpoint flag moves the median predicted slope to 0.52 and span
recovery to 0.21; recombining the resulting shape with the monolith's per-curve mean — a
mean-preserving recomposition — moves them to 1.02 and 0.42 [0.33, 0.54], with within-curve
Spearman 0.89 [0.67, 0.91] and shape MAE 0.665 → 0.469. All five split seeds move in the
same direction, and the gain survives publication-blocked and two-factor-blocked
resampling (+0.196 shape MAE, 95 % CI [0.140, 0.257]).

**The repair is partial, it is bought at a price, and it is exploratory.** Even after
recomposition the median predicted slope is 1.02 against a measured 2.57, and the per-curve
level error is untouched by construction — the median example in Fig. 3A is still 1.95 log
units below its measurements. The mechanism reads the candidate condition list, so the same
fitted model gives different answers when the user's candidate design changes: adding two
candidate points two to three decades outside the intended window shifts predictions by a
median 0.117 log units, more than the 0.025 macro MAE the mechanism buys (Fig. S2). The
representation was designed after the pre-registered sweep was read and is reported as a
mechanism, not as a confirmed pre-registered effect.

---

## Figure 4 — where the error is

**Decomposing the remaining error shows that it is overwhelmingly a per-extractant level
problem, and that the level is exactly the quantity a single experiment supplies.**
Substituting the true per-extractant level into the frozen model's predictions takes macro
MAE from 1.036 to 0.507 on the common cohort; substituting the true per-series level takes
it to 0.430, the true per-curve level to 0.384, and the true per-curve level *and* slope to
0.181. On the full cohort the same accounting attributes 0.520 of 0.970 to a per-extractant
constant, 0.192 to within-curve shape, 0.057 to series-local level, 0.060 to distant
chemistry, 0.037 to identified data-quality problems and 0.103 to nothing we could name.

**The two levers act on different components, and they are nearly orthogonal.** In
(level error, shape error) coordinates the model changes move almost vertically and the
measurements almost horizontally (Fig. 4C): the recomposition changes shape error from
0.559 to 0.527 while leaving level error at 0.8468 to twelve decimal places — a bit-exact
consequence of the mean-preserving recombination — whereas five measurements take level
error from 0.861 to 0.231 and shape error only from 0.514 to 0.395. The most striking entry
in Fig. 4A is that one *optimally chosen* measurement (0.504) matches knowing the true
per-extractant level exactly (0.507): under an offset calibration the best single point is
the one whose residual equals the extractant's median residual, and choosing it well is
worth as much as an oracle for the level.

**These are bounds, not achievements, and the accounting overlaps.** Every oracle in
Fig. 4A reads a held-out target and none is deployable; the components in Fig. 4B are not
disjoint (a mismatch row is also a distant-chemistry row), so their sum is a floor on the
removable fraction rather than an exact partition, and the 0.103 "unexplained" is defined
as what the identified components fail to cover. The comparison against a model with no
ligand information (1.115 vs 1.032 on this cohort; 1.099 vs 0.969 on the full cohort) says
that *this* representation of chemistry, on *this* corpus, is worth about a tenth of a log
unit — not that molecular structure is uninformative in principle.

---

## Figure 5 — generalisation and coverage

**Zero-shot error appears to grow with distance from the training chemistry, and the value
of a measurement with it — but the observational comparison alone is underpowered.**
Splitting the 99 held-out extractants into terciles of nearest-training-neighbour Tanimoto
similarity, zero-shot macro MAE is 0.900 for the nearest third, 0.962 for the middle and
1.255 for the furthest, and the first measurement removes 0.235, 0.448 and 0.495
respectively. An unpaired chemotype-block bootstrap of the far − near difference at *k* = 0
gives +0.35 with a 95 % interval of [−0.04, +0.66]: with 17 chemotypes per tercile the
trend is not separated from zero, and we do not claim it on this evidence alone.

**A controlled experiment supplies the evidence the observational split cannot.** Holding the held-out rows byte-identical, the learner,
the folds and the seeds fixed, and changing only which extractants may enter training,
lowering the eligibility threshold from ten condition cells to three (91 → 152 extractants,
+7 % of rows) improves macro MAE by 0.163 (BCa 95 % CI [0.012, 0.295], 79 of 131 scoring
units, 5/5 seeds). The improvement is +0.463 [0.251, 0.685] on extractants whose nearest
training neighbour is below Tanimoto 0.4. A row-count-matched arm reproduces the effect
(+0.161), so it is not the extra rows; an arm with the added targets permuted among
themselves recovers none of it (+0.173 against that control), so it is not regularisation
or reweighting; and the per-extractant gain tracks how much closer the expansion actually
brought the nearest training neighbour (Spearman ρ = +0.290, p = 2.9 × 10⁻⁴; +0.038 for
extractants brought no closer, +0.491 for those brought more than 0.15 closer, Fig. S9).
Almost all of the gain lands in the level, not the shape (offset +0.177 vs shape +0.024).

**The magnitude is a property of this cohort's composition, and the panels are not on a
common model.** 57 of the 131 scoring units are ECFP clusters made entirely of extractants
the expansion added, which the restricted arm structurally cannot serve; on the 91
extractants the restricted arm could already cover, the effect is +0.019 — a wash. The
row-weighted contrast is +0.068. Panel A uses the frozen pipeline and panel B a different
learner (the coverage experiment predates the shape work), so the two are deliberately not
plotted on one axis. Finally, "distance" here is fingerprint distance, which is a proxy: it
cannot separate "the model has not seen this scaffold" from "this scaffold's extraction
chemistry is intrinsically harder".

---

## Figure 6 — which measurement

**When only one experiment can be run, which one it is matters as much as which model made
the prediction.** Over 143 held-out extractants with identical candidate pools and
identical evaluation rows, choosing the geometric medoid of the candidate conditions gives
macro MAE 0.629 against 0.692 for a random choice — a paired improvement of 0.063
[0.028, 0.105], 98 of 143 extractants, 5/5 split seeds. That is more than twice the macro
gain of the best model change in this study.

**The signal that works is centrality, and the signal that does not work is the model's own
uncertainty.** No uncertainty-driven rule separates from random choice (maximum ensemble
standard deviation +0.004 [−0.023, +0.036]); choosing the candidate farthest from the
existing design is significantly *worse* than random (−0.062 [−0.112, −0.021]); and rankers
trained directly on realised regret do not beat plain centrality, choosing pure centrality
in 24 of 25 inner folds when allowed to blend. The reason is structural rather than a
calibration failure: under an offset calibration the ideal support point is the one whose
residual is closest to the extractant's *median* residual, not the one whose residual is
largest, so a well-calibrated uncertainty is optimising the wrong quantity.

**Even the best deployable policy harms a substantial minority.** 28 % of extractants are
worse after one medoid-chosen measurement than they were zero-shot (34 % under random
choice, 40 % under the farthest-point rule). The oracle bound is 0.463, so 0.166 log units
of the *k* = 1 error is attributable to not knowing which point to pick, and none of the
deployable policies recovers any of that gap. This is an argument for measuring two points
rather than one where the budget allows, not for a better acquisition score.
