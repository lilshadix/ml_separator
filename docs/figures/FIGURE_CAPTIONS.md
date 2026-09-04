# Figure captions

Publication-ready. Every number below is one of the 67 verified entries in
`METRIC_AUDIT.md` or is read from the corresponding `figures/derived/fig*_values.json`.
Abbreviations used throughout: *macro MAE* = mean absolute error in log₁₀ D averaged with
one vote per experimental unit (stated per figure); *T* = Tanimoto similarity on ECFP
fingerprints; *k* = the number of measurements taken on the held-out extractant at test
time.

---

## Figure 1

**Figure 1 | Task, held-out protocol and model.** The prediction target is log₁₀ *D*, the
distribution ratio of a single lanthanide between an aqueous and an organic phase, for a
given (extractant, lanthanide, conditions) row. **(A)** The corpus holds 5,248 such
measurements on 152 extractants, 14 lanthanides (promethium has no stable isotope) and
2,055 distinct condition cells. Extractants are grouped into 79 Tanimoto-0.7 chemotypes and
whole chemotypes are held out, so a held-out extractant has no analogue closer than
*T* = 0.7 anywhere in training; five outer folds × five split seeds, with the model seed
fixed at 42 and independent of the split seed. **(B)** The global model is two
`ExtraTrees` forests fitted on training chemotypes only. Forest 1 predicts log₁₀ *D* from
molecular fingerprints and descriptors, the lanthanide, the experimental conditions and
log-concentration terms derived from the extraction mass-action law. Forest 2 predicts the
*curve-centred* response — log₁₀ *D* minus the mean over the row's own titration curve —
from the same design plus five columns describing where the row sits inside its own
measurement window. The two are recombined per curve as (mean of forest 1 over the curve)
+ (shape of forest 2, itself centred), which is mean-preserving: the per-extractant level
is bit-identical to forest 1 alone (offset MAE 0.821172 for both, all five seeds), so only
the within-curve shape changes. **(C)** At deployment the user supplies a list of candidate
conditions for a new extractant; the model scores all of them zero-shot; a target-blind
acquisition policy nominates *k* of them; those *k* are measured; and a ridge-shrunk
calibrator with series-local terms adjusts the frozen model using only those *k* targets.
Colour encodes the information class throughout: training information, target-free
conditions of the held-out extractant, the *k* measured support targets, and the held-out
query targets that are scored and never read. The candidate pool and the evaluation set are
disjoint by construction and are redrawn 12 times per split seed, so a support row is never
also scored as a query row. The model is never refitted on the held-out extractant: the *k*
measurements buy calibration, not training. Evaluation cohort for the *k*-shot results:
99 extractants with at least 5 pool rows and 2 query rows in every arm at every *k*.

---

## Figure 2

**Figure 2 | A single measurement on an unseen extractant is worth more than every
modelling gain in this study.** All panels: 99 held-out extractants (41 chemotypes), five
split seeds × twelve pool/query draws, all methods evaluated on identical held-out
extractants and identical support/query draws, so every comparison is paired.
**(A)** Macro MAE (one vote per extractant) against the number *k* of measured support
points, for four pre-specified deployment rules — measurements only with no model; the
baseline global model with gen8's slope-repair calibration; the same calibration on the
shape-recomposed global model; and the final pipeline, which adds marginal-likelihood
series-local adaptation at *k* ≥ 2. Vertical ticks give the range over the five split
seeds (min–max, not a confidence interval; e.g. 1.000–1.078 at *k* = 0 for the final
pipeline). The green dashed line is an oracle acquisition policy that chooses each support
point by reading held-out targets so as to minimise the error of a level-only calibration;
it is not deployable. It is a genuine upper bound at *k* = 1 (0.504 against the deployable
0.654), but not beyond: from *k* = 2 the calibrator is estimating slopes rather than a
level, the two objectives want different points, and by *k* = 5 the deployable
central-then-spread rule overtakes it (0.441 against 0.453). The gap at *k* = 1 is the cost
of not knowing which experiment to run, and Figure 6 measures how much of it any deployable
policy recovers. The final
pipeline reaches 1.036 / 0.654 / 0.559 / 0.493 / 0.441 at *k* = 0 / 1 / 2 / 3 / 5.
**(B)** Macro MAE removed per additional measurement, paired over extractants with a
chemotype-block bootstrap BCa 95 % interval; percentages beneath the axis are the fraction
of the 99 extractants that measurement improves. The first measurement removes 0.382
[0.264, 0.565] and improves 67 % of extractants; the second removes 0.095 [0.074, 0.127]
and improves 81 %. **(C)** The two model-side contributions, each paired against the stage
below it: shape recomposition (blue) is worth 0.0247 [0.0157, 0.0375] at *k* = 0 and fades
as *k* grows, while series-local adaptation (vermillion) is the identity at *k* ≤ 1 by
construction and grows to 0.0309 [0.0093, 0.0476] at *k* = 5. **(D)** Per-extractant error
before and after one measurement, with the line of equality: 66 of 99 extractants improve,
median improvement +0.17 log units, and 33 get worse — one measurement is not
unconditionally safe.

---

## Figure 3

**Figure 3 | The model draws unseen titrations flat, and telling it where a point sits
inside its own titration window fixes most of that.** The relative-position representation
was designed after the pre-registered gen9 sweep had been read and is reported as an
exploratory mechanism, not as a confirmed pre-registered effect. Extractant-concentration titrations
of held-out extractants: 155 curves with at least four points, from 25 extractants,
evaluated under five split seeds (775 curve × seed evaluations). **(A–C)** Three measured
titrations with the three model stages overlaid, plotted as the curve-centred response
(log₁₀ *D* minus the mean over the curve) because that is the quantity the shape metrics
and the mean-preserving recomposition act on; each panel prints the per-curve level error,
which the recomposition leaves unchanged by construction. The three curves were selected by
a rule fixed before any curve was inspected (`FIGURE_PLAN.md` §9; the ranked table and the
runners-up are in `figures/derived/fig3_selected_curves.csv`, and the same six curves are
plotted on the raw log₁₀ *D* axis in Figure S7): the curve closest to the joint median of
baseline shape error and repair size (measured span 2.02 decades, level error −1.95); a
curve at the median of the top decile of repair (span 3.50, level error +0.005); and a
curve at the median of the *bottom* decile, where the repair does not help (span 2.93,
level error −0.90). **(D)** Distribution of the fitted slope of log₁₀ *D* against
log₁₀ [extractant] over all 775 curve × seed evaluations, measured and predicted; boxes are
quartiles, whiskers 1.5 × IQR, fliers suppressed. The 775 evaluations are 155 distinct
curves seen under five split seeds, so the plotted distributions contain five correlated
evaluations per curve; this affects the visual density but not the intervals in E and F,
which resample chemotypes. The measured median slope is 2.57; the
baseline model predicts 0.12, the relative-position representation 0.52, and the
recomposition 1.02. **(E)** Span recovery — the predicted log₁₀ *D* range of a curve
divided by the measured range — shown as the full per-curve distribution with the median
(tick) and its chemotype-block bootstrap 95 % interval (bar): 0.051 [0.045, 0.086] →
0.210 [0.116, 0.289] → 0.423 [0.327, 0.544]. **(F)** Within-curve Spearman correlation
between predicted and measured log₁₀ *D*, mean over curves with its chemotype-block
interval: 0.582 [0.264, 0.708] → 0.840 [0.444, 0.880] → 0.886 [0.674, 0.914]. Shape MAE
over the same curves falls 0.665 → 0.567 → 0.469. The wide intervals reflect that only 8
Tanimoto-0.7 chemotypes contribute extractant titrations at all; the paired,
publication-blocked interval for the shape gain is +0.196 [0.140, 0.257]
(`runs/gen10_final/publication_sensitivity/`).

---

## Figure 4

**Figure 4 | Half the remaining error is a per-extractant constant that one experiment
supplies; the rest is bounded.** **(A)** Macro MAE (one vote per extractant) on the
99-extractant common cohort, five split seeds, for four deployable stages of the frozen
pipeline (solid) and five oracle substitutions (hatched, green, not deployable). The
oracles replace, in turn, the true per-extractant level, the true per-series level, the
true per-curve level, and additionally the true per-curve slope; the oracle acquisition
arm chooses the single support point by reading held-out targets. Knowing the true
per-extractant level exactly is worth 1.036 → 0.507; one *optimally chosen* measurement
achieves 0.504, i.e. essentially the same thing, while the deployable central choice
achieves 0.654 — the gap between them is the value of choosing well (Figure 6). With a
perfect per-curve level *and* slope the floor is 0.181. **(B)** The seven-component budget
of the 0.970 macro MAE measured on the full 152-extractant cohort with one vote per ECFP
cluster (`runs/gen10_final/error_decomposition/`): grey is each component's current
contribution, vermillion the part a deployable action in this study actually removed. The
per-extractant level contributes 0.520, of which one central measurement removes 0.349;
within-curve shape contributes 0.192, of which the deployable slope repair removes 0.008;
0.103 remains unexplained by any identified component. **(C)** Trajectory in
(level error, shape error) space on the 99-extractant cohort. A better zero-shot model
(blue) moves almost vertically — the recomposition changes shape error 0.559 → 0.527 while
leaving the level error at 0.8468 to twelve decimals — whereas measurements (vermillion)
move almost horizontally, taking level error 0.861 → 0.231 between *k* = 0 and *k* = 5
while shape error falls only 0.514 → 0.395. Removing ligand information entirely costs
1.115 versus 1.032, so on this cohort every fingerprint, descriptor, 3-D block and
pretrained embedding together is worth 0.083 log units; on the 152-extractant cohort with
cluster-level voting the same contrast is 0.130.

---

## Figure 5

**Figure 5 | Zero-shot error grows with distance from the training chemistry, and adding
training chemistry causally repairs it.** **(A)** Final pipeline, 99 held-out extractants
split into terciles of nearest-training-neighbour Tanimoto similarity *T* (near
0.66–0.69, *n* = 38; mid 0.59–0.66, *n* = 28; far 0.24–0.59, *n* = 33), macro MAE against
*k*. Ticks are marginal chemotype-block bootstrap 95 % intervals of the mean and are wide
because each tercile contains only about 17 chemotypes; the *k*-dependence within a tercile
is paired and much better determined. The zero-shot point estimates rise with distance
(0.900 / 0.962 / 1.255) and the first measurement removes 0.235 / 0.448 / 0.495
respectively, but an unpaired chemotype-block interval on the far − near difference at
*k* = 0 is +0.35 [−0.04, +0.66] and does **not** exclude zero: this observational cohort,
with 17 chemotypes per tercile, cannot separate the distance effect on its own. Panels B
and C carry the interventional evidence. After the first measurement the ordering by
distance no longer holds: the near tercile is dominated by one very large chemotype whose
members are hard for other reasons. **(B, C)** A controlled coverage experiment
(gen6 Experiment A): the same learner, the same folds and byte-identical held-out rows;
only the *training* mask changes. `restricted` trains on extractants with at least 10
condition cells, `full` on those with at least 3; the shuffled-target control keeps the
added rows and permutes their targets among themselves. Macro MAE here is one vote per
ECFP cluster over 131 clusters, five split seeds; error bars are the seed range.
**(B)** Full coverage beats restricted coverage at every distance endpoint, and the gap
widens as the held-out extractant gets further from the restricted training set
(1.210 → 1.047 overall; 1.537 → 1.043 for *T* < 0.4), while the shuffled control does not
recover any of it. **(C)** Paired improvements with chemotype-block BCa 95 % intervals:
+0.163 [0.012, 0.295] overall (79/131 units improved, 5/5 seeds), +0.463 [0.251, 0.685] on
*T* < 0.4 (36/42 units), +0.161 [0.010, 0.293] for the row-count-matched control — so the
effect is the added chemistry, not the extra 7 % of rows — and +0.173 [0.028, 0.309]
against the shuffled-target control. **Qualification:** 57 of the 131 scoring units are
ECFP clusters composed entirely of extractants the expansion added, which the restricted
arm structurally cannot serve; on the 91 extractants the restricted arm could already cover
the effect is +0.019. Figure S9 shows the dose–response that separates chemical transfer
from a class prior.

---

## Figure 6

**Figure 6 | Which single experiment to run matters as much as the model, and model
uncertainty is the wrong signal for choosing it.** Seventeen acquisition policies scored
on identical candidate pools and identical evaluation rows of the same held-out extractant:
143 extractants, five split seeds × eight pool draws, frozen global model, one measurement
(*k* = 1). **(A)** Macro MAE after the measurement, one vote per extractant, coloured by
policy family; horizontal bars are the chemotype-block bootstrap BCa 95 % interval of the
paired difference against `RANDOM`, drawn about each policy's own point estimate; the
dashed line is the zero-shot error (0.980). Geometric centrality wins: `MEDOID` is +0.063
[+0.028, +0.105] better than random choice, improving 98 of 143 extractants on 5/5 seeds.
No uncertainty-driven rule separates from random (`MAX_ENSEMBLE_SD` +0.004
[−0.023, +0.036]), and choosing the candidate farthest from the existing design is
significantly *worse* than random (−0.062 [−0.112, −0.021]). Learned rankers trained on
realised regret do not beat plain centrality. `ORACLE` (0.463) and `SURROGATE_ORACLE`
read held-out targets and are upper bounds, not methods. Three further learned variants
that additionally read the model's own prediction are omitted from the panel because they
belong to a second feature set; all rank below the geometry-only versions and are listed
in `runs/gen10_final/acquisition/realised_summary.csv` and
`figures/derived/fig6_values.json`. **(B)** The fraction of
extractants that the single measurement makes *worse* than the zero-shot prediction: 28 %
even under the best deployable policy, 34 % under random choice, and 40 % under the
farthest-point rule. A deployment that measures at an arbitrary condition is not safe by
default. The centrality result is specific to the *first* measurement: from *k* = 2 onward
the requirement changes from anchoring a level to estimating slopes, and by *k* = 5 a
D-optimal spread policy is marginally ahead of central-then-spread (0.4403 vs 0.4405 macro
MAE on the 99-extractant cohort), which is why the shipped rule measures centrally first
and then spreads.

---

# Supplementary figure captions

**Figure S1 | Composition of the corpus.** One split seed (the cohort is identical in
every seed). **(A)** Measurements per lanthanide; europium alone carries 1,343 of 5,248
rows because it is the workhorse of solvent-extraction studies, and promethium is absent.
**(B)** Measurements per extractant, ranked, on a log axis: the median extractant has 13
measurements and the largest has 1,488, i.e. 28 % of the corpus. The dashed line marks the
10-row threshold used by the restricted-coverage arm of Figure 5B. **(C)** Cumulative share
of rows by Tanimoto-0.7 chemotype: the largest chemotype (the diglycolamides) holds 64 % of
all rows, which is why every interval in this paper resamples chemotypes rather than rows.
**(D)** Titration and series curves by axis, with the median number of points per curve.
The main-text shape analysis uses the 155 extractant-concentration curves that have at
least four points, of the 241 present.

**Figure S2 | The relative-position mechanism is query-set dependent — the paper's main
negative result.** The same fitted model is asked for the same absolute conditions while
the *candidate design* around them changes; a model that reads only absolute conditions
must answer identically. Five split seeds; 66 extractants with synthesisable concentration
axes for the decoy/extension/density variants, 130 for the subset variants; between 1,685
and 8,425 comparisons per cell. **(A)** Median absolute prediction shift. The baseline
model shifts by exactly zero under every perturbation. Adding two candidate points two to
three decades outside the intended window shifts the recomposed model's predictions by a
median 0.117 log units — larger than the 0.025 macro MAE the mechanism buys (dashed line).
**(B)** The tail: the 95th percentile across extractants of each extractant's own 95th
percentile shift, reaching 0.68 for the decoy perturbation. Permuting the order of the
candidate list and asking per-series rather than per-ligand change nothing, confirming the
effect is the window and not an implementation artefact.

**Figure S3 | Every calibration adapter at every k.** Frozen global model, 99-extractant
common cohort, central-then-spread acquisition, five split seeds × twelve draws.
**(A)** `NO_MODEL` (fit only the *k* measurements) plateaus at 0.76; a level-only update
(`OFFSET_K1`) plateaus at 0.59; allowing shrunk response coefficients (`OFFSET_K3`) reaches
0.479 at *k* = 5; and the marginal-likelihood series-local prior (`SERIES_ML`) reaches
0.441. `SERIES_ML` also beats its own no-series ablation (0.467 at *k* = 5), so the
series term earns its place. **(B)** Paired chemotype-block BCa intervals against
`OFFSET_K3` from `runs/gen10_final/adaptation/bootstrap_vs_offset_k3.csv`.

**Figure S4 | Zero-shot architecture and representation ablation.** Twenty-four arms, all
fitted on the same folds with the same learner family and scored on the same held-out rows;
152 extractants, one vote per ECFP cluster, five split seeds. **(A)** Macro MAE, mean ± sd
over seeds, drawn as points because the axis is truncated. No explicit level + shape head,
residual correction, two-branch model or learned set encoder beats the recomposition, and
no change to feature access or capacity reproduces it. **(B)** The same arms in
(level error, shape error) space: every recomposed arm sits at exactly the baseline's level
error, because the recomposition is mean-preserving by construction, and differs only in
shape.

**Figure S5 | The compression is worst on the extractant axis but is not confined to it.**
Same three model stages as Figure 3, all six condition axes, five split seeds.
Span recovery improves on the extractant axis (0.05 → 0.42), the acid axis (0.22 → 0.39)
and the lanthanide series (0.38 → 0.72), and barely moves on temperature, contact time and
metal concentration, where there are 8–35 curves in total and the measured spans are small.
Within-curve Spearman improves markedly only on the extractant axis. Curve counts per axis
are in Figure S1D.

**Figure S6 | Per-split-seed stability.** **(A)** The *k*-shot frontier of the baseline
model and of the final pipeline, one line per split seed; the pipeline is below the
baseline at every *k* in 5 of 5 seeds from *k* = 2 onward. **(B)** Median predicted
extractant slope by seed for the three model stages against the measured median of 2.57;
the ordering baseline < relative position < recomposition holds in every seed.

**Figure S7 | The Figure 3 example curves on the raw axis, with their runners-up.** The
three curves selected by the pre-declared rule (**A–C**) and the next-ranked curve in each
role (**D–F**), plotted as raw log₁₀ *D* rather than curve-centred. The vertical offset
visible in A, C and F is the per-curve level error, which the mean-preserving recomposition
does not touch and which Figure 4 addresses; the change in slope between the orange and
blue traces is the effect Figure 3 is about. Selection statistics for all six curves are in
`figures/derived/figS7_values.json`.

**Figure S8 | Prediction diagnostics and data-quality strata.** Frozen global model,
26,240 held-out row × seed predictions. **(A)** Predicted against measured log₁₀ *D* with
the line of equality: the predictions have a standard deviation of 0.85 against 1.66 for
the measurements — the model compresses the whole target range, not only the within-curve
range. **(B)** Residual (prediction − measurement) against prediction, with a binned
mean: the mean residual is −0.39, i.e. the model systematically under-predicts on held-out
chemistry, and does so increasingly at high predicted *D*. **(C)** Macro MAE by the
source-audit strata defined in gen10 Phase 7: consistent rows (0.95, 145 extractants),
rows whose recorded name and structure disagree (1.00, 15 extractants), decade-shifted
duplicate cells (1.30, 3 extractants), the single suspect extractant TWE-24 (4.10, 30 rows)
and rows flagged uncertain (1.41, 3 extractants). Nothing was corrected or deleted; the
strata are reported so that the reader can see how much of the headline is data quality
(0.037 of 0.970 macro MAE in total).

**Figure S9 | The coverage effect is chemical transfer, not a class prior.** For every
held-out extractant, how much closer the coverage expansion brought its nearest training
neighbour in Tanimoto similarity, against how much macro MAE the expansion removed for it;
152 extractants, five split seeds, recomputed from the stored out-of-fold predictions.
**(A)** Binned means: extractants the expansion brought no closer gain +0.038, those it
brought more than 0.15 closer gain +0.491. **(B)** The same, per extractant;
Spearman ρ = +0.290, p = 2.9 × 10⁻⁴. A class-level prior would help every added-class
extractant equally regardless of proximity; it does not.

**Figure S10 | How to spend a fixed measurement budget.** Each strategy is scored by
composing the frozen per-extractant adaptation curves over the 99 held-out extractants, so
no model is refitted; the budget axis is logarithmic. Below about one measurement per
extractant, breadth beats depth by a wide margin, and covering new chemotypes first beats
covering extractants in arbitrary order (0.979 against 0.999 at a budget of 9, and 0.782
against 0.847 at a budget of 49). Once every extractant has one measurement the
breadth-only strategies saturate at 0.654 and depth takes over, so the strategy that wins
at every budget is central-then-spread with new chemotypes first. Dashed lines mark the two one-measurement-each orderings; "1 each, new
chemotypes first" coincides exactly with central-then-spread until every extractant has
been measured once, which is why it is drawn dashed over it. Two further strategies
(`A_DEPTH_3`, `E_CENTRAL_SPREAD`) are omitted from the panel because they lie on top of the
plotted ones; all eight are in `figures/derived/figS10_values.json`. The source table's
`gain_per_measurement` column is *not* plotted: it divides by the measurements actually
spent, so a strategy that cannot spend the budget scores well on it while achieving a worse
macro MAE.
