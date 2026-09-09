# Gen14 — the separation curve as one bit and one scalar

Gen13 stage 3 ended with a recommendation: *build the next generation as a direction classifier plus
an amplitude prior, because essentially all the ligand chemistry that transfers across laboratories
in this corpus is one bit per extractant.*  Gen14 is that model.  It predicts a cell's centred
lanthanide curve from exactly two numbers,

```
log D(Ln) − mean_Ln log D  =  a · radius(Ln) + b · radius(Ln)²,     a = s · m
```

where `s = ±1` is the **direction** of selectivity called from the ligand's donor topology, `m` is a
single **magnitude** taken from the training fold, and `b` is the training fold's mean curvature.
Nothing else is fitted; the only input is one SMILES string, because the 39 topology descriptors are
bond counts on the 2D molecular graph.

Everything the gen13 ladder freezes is imported, not re-implemented — cohort, physics basis, per-cell
ridge, the five fold plans, the chemotype-balanced weights, the pairwise metrics and the
chemotype-blocked paired bootstrap.  The bench reproduces gen13's locked stage-3 headline to the last
digit (`G13_ET_TOPO39` under BP = `0.7683085207475452`, published `0.7683085207475452`).

Protocol, from `feedback-evaluation-protocol` and Addendum 3 of gen13's pre-registration: **every
candidate is scored under all five hold-out designs and the full table is reported**, and **every
increment is stated against the cheapest sensible alternative** (the 13-column gen6 donor census),
not only against the constant baseline.  Designs: A exact-extractant, B chemotype hold-out,
BR random-cell control, BQ random-publication control, BP publication-masked (the deployment-relevant
one under gen13's Addendum 2).

---

## 1. The direction: a linear model on the same 39 columns beats gen13's trees

| model (direction) | B | BR | BQ | A | BP |
|---|---|---|---|---|---|
| **`LOGIT_TOPO39`** L2 logistic, 39 topology columns | **0.840** | **0.835** | **0.840** | 0.835 | **0.821** |
| `G13_ET_TOPO39` gen13's locked 400 extra trees, same columns | 0.822 | 0.816 | 0.814 | 0.798 | 0.768 |
| `CHEM137_ET` physchem + donors + coordination | 0.749 | 0.729 | 0.742 | 0.791 | 0.734 |
| `LEAN209_ET` the full compact set | 0.767 | 0.766 | 0.761 | 0.804 | 0.721 |
| `DONORS13_ET` the gen6 donor census — cheapest alternative | 0.730 | 0.690 | 0.713 | 0.743 | 0.694 |
| `ALWAYS_HEAVY` the constant rule | 0.559 | 0.559 | 0.559 | 0.559 | 0.559 |

Macro accuracy over 82 extractants, cells with ≥ 5 measured metals, chemotype-blocked resampling.

Paired, on the same units:

| contrast | B | BR | BQ | A | BP |
|---|---|---|---|---|---|
| logistic − always-heavy | +0.282 | +0.276 | +0.282 | +0.276 | +0.262 |
| | p < 0.001 | p < 0.001 | p < 0.001 | p < 0.001 | p = 0.0008 |
| logistic − donor census | +0.110 | +0.144 | +0.127 | +0.091 | +0.127 |
| | p < 0.001 | p < 0.001 | p < 0.001 | p < 0.001 | p < 0.001 |
| logistic − gen13's trees | +0.018 | +0.019 | +0.026 | +0.037 | +0.052 |
| | p = 0.26 | p = 0.23 | p = 0.13 | p = 0.003 | p = 0.006 |

The gain over gen13's estimator is positive in all five designs and leave-one-chemotype-out stable in
all five, significant in the two that break the most correlation (A and BP), and largest exactly
where gen13's trees lost the most — under publication masking, where the trees fall 0.822 → 0.768
and the logistic falls only 0.840 → 0.821.  A linear model on 39 columns extrapolates to a topology
vector it has not seen; a forest can only interpolate between the 22 distinct vectors the corpus
contains, which is why it degrades when the fold plan takes more of them away.

The honest increment — over the plain 13-column donor census, which already reaches 0.69 — is
**+0.09 to +0.14 depending on design**, against gen13's +0.05 to +0.13 for the same comparison.

---

## 2. What it is worth on the programme's own metric

Extractant-macro MAE of predicted log SF over all metal pairs of every held-out cell:

| arm | B | BR | BQ | A | BP |
|---|---|---|---|---|---|
| `DIR_ORACLE` true direction × constant magnitude | 0.478 | 0.478 | 0.478 | 0.478 | 0.486 |
| **`G14_DIR_HARD` gen14: logistic direction × constant magnitude** | 0.493 | **0.491** | **0.491** | 0.492 | **0.500** |
| `G13_DIR_HARD` gen13's tree direction, same conventions | 0.493 | 0.494 | 0.494 | 0.501 | 0.523 |
| `G13_FULL_MODEL` the 209-column regression on both coefficients | **0.482** | 0.485 | 0.480 | **0.437** | 0.552 |
| `MEAN_CURVE` the corpus mean curve | 0.597 | 0.597 | 0.597 | 0.583 | 0.622 |

Two things in this table matter more than the ranking.

**Gen14 is the first arm in the programme whose score does not depend on the hold-out design.**  Its
five numbers span 0.491–0.500, a range of 0.010.  The 209-column regression spans 0.437–0.552, a
range of 0.115 — it looks best where the design leaks (exact-extractant A) and worst where the design
removes the laboratory (BP).  A model with two degrees of freedom has almost nothing to fit to the
fold plan, and that is visible as design-invariance, which is the property a deployed model needs.

**Under BP — the design gen13 fixed as the one deployment is chosen under — gen14 wins.**
0.500 against 0.552 for the full regression (+0.052, 59/90 extractants improved, 5/5 seeds,
LOCO-stable, p = 0.058, interval [−0.002, +0.098]), against 0.536 for gen13's deployed lean bag
`V2_BAG4@lean`, and against 0.622 for the corpus mean curve (+0.122, p = 0.002, **passes P1** in
every design).  Under B, BR and BQ the two models tie inside noise (−0.011 to −0.006, p ≥ 0.53) and
under A the regression is better (−0.055, p = 0.049), which is the leak being priced, exactly as
gen11–gen13 priced it.

For comparison, gen13's published stage-3 arm scored 0.561 under BP; 0.038 of the improvement to
0.500 comes from two conventions (training the classifier only on well-determined cells, and taking
the magnitude from those cells), and 0.023 from the estimator.

---

## 3. The direction is finished; the magnitude is where the error now lives

| contrast (BP) | point | 95 % CI | p |
|---|---|---|---|
| oracle direction − gen14 direction | +0.014 | [−0.012, +0.057] | 0.36 |
| oracle magnitude − loss-fitted magnitude (predicted direction) | **+0.149** | [+0.053, +0.222] | < 0.001 |
| oracle both | 0.322 vs 0.500 | | |

`ORACLE_vs_G14hard` is +0.012 to +0.015 in **all five designs** — a perfect direction call would now
buy about a hundredth of a log unit.  The bit is spent.  The magnitude, given the predicted direction,
is worth ten times as much, and every prior tried for it lands inside 0.007 of a constant:

| amplitude prior (BP) | macro MAE |
|---|---|
| oracle magnitude | 0.353 |
| loss-fitted against the *oracle* training direction | 0.495 |
| direction-conditional (separate `m` for heavy and light) | 0.499 |
| training median \|amplitude\| | 0.499 |
| **training mean \|amplitude\| (gen13's prior, deployed)** | **0.500** |
| loss-fitted against inner-CV training directions | 0.502 |
| extra trees on log \|amplitude\| | 0.505 |
| ridge on log \|amplitude\| | 1.09 (unstable) |

Fitting the scalar to the exact loss it is judged by, on the training fold, with the classifier's own
error rate folded in, buys **nothing** (−0.002, p = 0.85).  This is a strong negative: the magnitude
is not a statistic that is being estimated badly, it is a quantity the ligand structure does not
carry.  That agrees with stage 2 (amplitude ICC 0.72 at chemotype level, Kish n_eff ≈ 12) and stage 3
(magnitude class not predictable, 0.482 vs 0.462 majority) and now also holds on the MAE metric.

---

## 4. How much of the remaining direction error is removable at all

| ceiling (BP) | macro accuracy |
|---|---|
| gen14 model | 0.821 |
| any model constant within an extractant (10 of 82 extractants contain cells of both signs) | 0.959 |
| a perfect model against a label estimated from the cell's own residual σ | 0.929 |
| a perfect model against a label at the corpus replicate noise floor (σ = 0.237) | 0.891 |

Banded by how well the label is determined:

| \|amplitude\| | extractants | model | ceiling (replicate σ) |
|---|---|---|---|
| < 0.05 | 18 | 0.588 | 0.599 |
| 0.05 – 0.1 | 25 | 0.673 | 0.776 |
| 0.1 – 0.2 | 15 | 0.805 | 0.921 |
| 0.2 – 0.5 | 29 | 0.917 | 0.996 |
| > 0.5 | 24 | 0.933 | 1.000 |

**In the weakest band the model is already at the noise floor** — where a cell's whole La→Lu contrast
is under 0.16 log units, its observed direction is a coin flip and no model can agree with it.  The
recoverable headroom is +0.07 and it sits entirely in the strongly directed extractants, where the
model reaches 0.92 and the label is exact.  Abstention does not find it: ordering by the model's own
confidence and keeping the top quarter moves macro accuracy only 0.821 → 0.850, so the classifier's
probability does not rank its own errors.

---

## 5. Measured dead ends

Thirty direction candidates and eight amplitude priors / decision rules, each under all five designs.
Nothing beat `LOGIT_TOPO39` by more than 0.005.

*Estimator* — penalty chosen honestly by chemotype-grouped inner CV (−0.086 under BP: the inner split
does not predict the outer one), C = 0.3 (−0.042 BP), L1 (−0.021), rank transform (−0.105 BP),
PCA-5/PCA-8 (−0.008), linear SVM (−0.008 worst design), a blend with gen13's trees (−0.010).
C = 3 and C = 10 are +0.000 to +0.007, inside noise, and C is a free parameter — C = 1 stays.

*Label and training set* — down-weighting cells whose direction is barely determined by |amplitude|
(logistic ±0.003 in every design; it helps gen13's trees by +0.012 under BP but leaves them 0.040
below the logistic); training on all 521 cells instead of the 289 well-determined ones (+0.006 under B, BR,
BQ but **−0.046 under BP**); replacing a cell's label with its extractant's mean direction (−0.002 to
+0.010); collapsing the training fold to one row per extractant (−0.007 BP).

*Target* — a ridge on the signed amplitude squashed into a probability, which uses the magnitude a
classifier discards: −0.045 to −0.062 in all five designs.

*Representation* — topology plus donor elements (−0.057 BP), plus the gen6 census (−0.060), plus
architecture (−0.100), plus mass action (−0.100), the whole 114-column coordination block (−0.117),
six hand-picked bite descriptors (0.783 under B but 0.625 under BP).  **Mass action is the
informative failure**: the eight physical condition columns are the only route by which conditions
could explain the ten extractants whose own cells disagree about the direction, and they cost
0.07–0.10 under every design.  The direction is a property of the ligand, not of the chemistry it is
run in.

*Decision rule* — the posterior mean `(1−2p)·m`, which shrinks toward the flat curve when the
classifier is unsure, costs 0.021–0.024 MAE in every design; the square-root rule costs 0.006–0.012.
Under an absolute-error metric the hard call is right.

### The one post-hoc idea, and its falsification

The error analysis under BP found the model right on 40 of 43 strongly directed extractants, and that
two of the three failures are the same motif — a diglycolamide with alkyl substituents on the ether
backbone, light-selective at +0.30 and +0.36, whose 39 topology columns are bit-identical to an
ordinary, strongly heavy-selective DGA.  Bond counts between donors cannot see a substituent hanging
off the chelate ring, so a seven-column **steric** block (carbon per donor, heavy atoms per donor,
mean heavy degree, branch counts, sp³ fraction, symmetry) was added and run under all five designs.

It fails, and it fails hardest in the band the hypothesis was about: macro accuracy on strongly
directed extractants drops 0.920 → 0.720 under BP, and overall 0.821 → 0.720.  Steric columns alone
score 0.39–0.50, below the constant rule.  The α-alkylated DGA reversal is real chemistry that this
descriptor set does not contain, and diluting the topology block does not recover it.

---

## 6. What is deployed, and what to do next

`gen14_direction/scripts/g14_predict.py` fits the logistic on all 289 well-determined cells and
predicts from a SMILES string alone: direction, calibrated-ish probability, amplitude, the 14-metal
curve, every pairwise log SF, the widest pair to measure first, and a flag for a topology vector no
training ligand has.  Smoke test: TODGA → heavy-selective (observed −0.33); the corresponding
malonamide, which loses the ether oxygen and the five-membered bite → light-selective (observed
+0.36); a bis-phosphonate → heavy-selective with the novel-topology flag set.

**Next.**  The direction bit is exhausted and the magnitude is not in the 2D structure.  Three moves
are left, in order of expected value:

1. **More chemotypes, not more model.**  Forty chemotypes and ~12 effective units set every interval
   in this report; the model is not the binding constraint.
2. **One measurement.**  Stage 2's BLUP on the widest available pair takes the MAE from 0.541 to
   0.266 under BP.  Gen14 improves only the zero-measurement mode; the measured mode remains the
   largest single lever in the programme.
3. **A descriptor that sees substitution near the donor set.**  The one identified failure mode is
   backbone substitution reversing selectivity while the donor graph is unchanged.  The seven-column
   steric proxy tried here is not it; this needs a descriptor designed for the question, and it
   should be validated on a cohort that contains more than two examples of the effect.

---

## Files

| | |
|---|---|
| `gen14/dirbench.py` | frozen bench: labels, five designs, macro accuracy, chemotype-blocked bootstrap, decision rules |
| `gen14/models.py` | direction models, amplitude priors, weights, composition |
| `scripts/g14_baseline.py` | reproduction of gen13's locked number + the five-design baseline table |
| `scripts/g14_sweep.py`, `scripts/g14_sweep2.py` | the 22 candidates |
| `scripts/g14_value.py` | the pairwise-MAE endpoint, five designs |
| `scripts/g14_magnitude.py` | amplitude priors including the loss-fitted ones |
| `scripts/g14_ceiling.py` | label-noise, extractant-constancy and abstention ceilings |
| `scripts/g14_errors.py` | which strongly directed extractants still fail |
| `scripts/g14_steric.py` | the post-hoc hypothesis and its falsification |
| `scripts/g14_predict.py` | the deployable predictor |
| `results/TABLES.md` | every table above, regenerated from the CSVs |
