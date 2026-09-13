# Gen15 — the honest floor, the curvature, and the measured mode

Gen14 closed the direction question: an L2 logistic on 39 donor-topology columns calls the direction
of lanthanide selectivity for a held-out chemotype in a held-out publication at 0.821 macro accuracy,
a perfect call would now buy only about 0.014 of pairwise MAE, and seven priors for the magnitude all
land inside 0.007 of a constant.  It ended by naming three moves: more chemotypes, one measurement,
and a descriptor that sees substitution near the donor set.

Gen15 does three things.  It prints the number the programme had never printed — what the model is
worth against *predicting no separation at all*.  It measures where the remaining error actually sits,
and finds a second coefficient nobody has ever modelled.  And it rebuilds the measurement-assisted
mode so that its ladder is a like-for-like comparison, and audits it for a leak the locked protocol
left open.

Everything is scored on the frozen gen13 cohort with the frozen fold plans, the frozen pairwise
metrics and gen13's chemotype-blocked paired bootstrap; the bench reproduces gen14's deployed number
to four decimals (0.5001 under BP).

---

## 1. The floor: what the programme is worth against nothing

Extractant-macro MAE of predicted log SF, design BP (chemotype held out **and** every training cell
from a held-out cell's publication dropped — the design gen13 fixed as the one deployment is chosen
under):

| arm | BP | what it is |
|---|---|---|
| `O_BOTH` | **0.181** | the cell's own two coefficients — **in sample; see §1a, the honest value is 0.274** |
| `O_AMP` | 0.322 | the cell's own radius coefficient |
| `O_CURV` | **0.427** | gen14's amplitude with the cell's own **curvature** |
| `O_PUBMAG` | 0.448 | magnitude = the cell's own publication's mean \|a\| |
| `O_LEVEL_MAX` | 0.467 | magnitude as a monotone function of the cell's own level of log D |
| `O_SIGN` | 0.486 | a perfect direction call, constant magnitude |
| `O_EXTMAG` | 0.488 | magnitude = the same extractant's other cells' mean \|a\| |
| **`G14`** | **0.500** | the deployed model |
| `QUERY_SPAN` | 0.510 | magnitude from the radius span of the metals asked about |
| `G13_FULL` | 0.552 | gen13's 209-column regression |
| **`FLAT`** | **0.589** | **predict no separation at all** |
| `MEAN_CURVE` | 0.622 | the corpus mean curve |

Two things follow immediately.

**The corpus mean curve is not the right baseline.**  It scores 0.622 — *worse* than predicting a flat
line.  Every "+0.122 over the mean curve" in the gen14 report is therefore an overstatement of what
the model is worth against doing nothing.

**The honest zero-shot gain is +0.088** (`G14_vs_FLAT`: +0.0884, 95 % CI [+0.0010, +0.1473],
p = 0.047, 59/90 extractants improved, 5/5 seeds, LOCO-stable, passes P1).  Four generations of
ligand modelling move the pairwise error from 0.589 to 0.500.  That is a real, significant,
design-invariant gain, and it is small.

### 1a. Correction: every own-cell oracle in this report is optimistic, and by how much

An earlier draft called `O_BOTH` = 0.181 "the representation ceiling".  That is wrong and the
correction is load-bearing enough to state before anything else in this report is read.

A cell's coefficients are fitted to the same log D values the arm is then scored against, so part of
every own-cell oracle's advantage is the coefficient absorbing the noise of the very pair being
scored.  The clean correction needs no assumption about the noise at all: **refit the cell's
coefficients without the two metals of the pair being scored**, and score that.  On identical rows
(`exp/labelerr/results/s4b_oracle_honesty.csv`; the in-sample arm reproduces this report's `O_BOTH`
to 0.18111057, so it is the same pipeline):

| arm | in sample | leave-pair-out | what the oracle was fitting |
|---|---|---|---|
| own a and b | 0.1811 | **0.2736** | 0.093 of it was noise |
| own a, constant b | 0.3191 | 0.3370 | 0.018 |
| true sign, own b | 0.3771 | 0.4245 | 0.047 |
| true sign, constant b | 0.4724 | 0.4724 | — (nothing own-cell to fit) |

So the two-parameter representation ceiling is **≈ 0.27**, not 0.18, and a σ-based simulation puts
the honest two-term ceiling at 0.234–0.299 across the defensible noise range with a noise-only floor
of 0.13–0.22.  The labels themselves are clean — reliability 0.978 for `a` and 0.936 for `b`, and
every attempt to de-noise or precision-weight them is null or harmful — so this is not a data
problem.  It is that an in-sample oracle is not a ceiling.

Everything that follows quotes the in-sample oracles because they are what the five-design bench
computes, but **every own-cell oracle gap in this report should be read as roughly half again too
large**, and the correction is largest exactly where the fitted quantity is least determined: the
curvature.

---

## 2. The curvature is the unclaimed half of the error

`O_CURV` — gen14's amplitude, with each cell given its own second coefficient instead of the training
fold's mean — scores 0.427 against 0.500.

| contrast (BP) | point | 95 % CI | p | P1 |
|---|---|---|---|---|
| `O_CURV` − `G14` | **+0.0729** | [+0.0304, +0.1334] | 0.0006 | **passes** |
| `O_AMP` − `G14` | +0.1779 | [+0.0952, +0.2390] | < 0.0001 | passes |
| `O_PUBMAG` − `G14` | +0.0524 | [−0.0383, +0.1222] | 0.39 | no |
| `O_LEVEL_MAX` − `G14` | +0.0332 | [−0.0017, +0.0569] | 0.067 | no |

**Read that table with §1a.**  On matched rows the curvature oracle's in-sample advantage is +0.095
and its leave-pair-out advantage is +0.048, so **about half of the +0.073 above is the oracle fitting
the noise of the pair it is scored on**, and the honest curvature headroom is nearer +0.036.  The
curvature is still the second-largest identified lever and it is still unclaimed, but it is a
smaller prize than the in-sample number says, and any future arm must be compared with the
leave-pair-out oracle rather than this one.

The curvature is not a redundant re-parameterisation of the amplitude: the two are uncorrelated
(Pearson 0.03, Spearman −0.10).  It has chemotype ICC 0.67 — the same grouped structure the amplitude
has.  Its mean depends strongly on the direction (heavy-selective cells sit at −0.009,
light-selective at −0.136).  And it carries the one thing a monotone ramp cannot express: **27 % of
well-determined cells have an interior extremum inside their measured range**, a genuine maximum or
minimum in the middle of the series.  `O_CURV` also lifts strong-pair sign accuracy above the
perfect-direction oracle (0.866 vs 0.852), which is exactly what a peak does — it makes the sign of a
pair depend on where in the series the pair sits.

### …and it is not in the 2D molecular graph either

Every route tried fails, under BP:

| arm | BP | vs G14 |
|---|---|---|
| `B_MEAN_RICH` (constant b, well-determined cells only) | 0.5005 | −0.000 |
| `B_DIRCOND_MEAN` (one b for heavy, one for light) | — | −0.0022, p = 0.10 |
| `B_MEDIAN` (weighted median instead of mean) | 0.5046 | −0.0045, p = 0.10 |
| `B_DIRCOND_MED` | — | −0.0062, p = 0.006 (worse) |
| `B_LOGIT` (logistic on sign(b), 39 topology columns) | 0.5496 | −0.0495 |
| `B_LOGIT_DC` | 0.5536 | −0.0536 |
| `SHAPE6` (6 learned curve prototypes, multiclass logistic) | 0.5945 | −0.0944 |
| `SHAPE4` | 0.6103 | −0.1102 |
| `SHAPE3` | 0.6258 | −0.1257 |
| *oracle assignments, for reference* | | |
| `B_LOGIT_ORACLE` | 0.4783 | +0.022 |
| `SHAPE4_ORACLE` | 0.4438 | +0.0562, p = 0.003, **passes P1** |
| `SHAPE6_ORACLE` | 0.4192 | +0.081 |

The oracles confirm the headroom; the classifiers cannot reach it.  This is the same shape of result
gen14 got for the magnitude, from a different direction: **beyond one bit, the 2D donor topology does
not carry the curve.**

The prototype experiment is worth stating plainly because it tests the programme's own theory of
itself.  Stage 2 measured about a dozen effective training units for a chemotype-level quantity, so
the natural hypothesis is that the corpus can support a small *alphabet* of canonical lanthanide
curves.  It can — the oracle alphabet is worth +0.056 at k = 4 — but the alphabet cannot be predicted
from structure any better than the single bit can.

---

## 3. A strong 3D lead, and the control that killed it

The repository carries 1155 QC-accepted GFN2-xTB lanthanide-complex geometries covering all 82
well-determined extractants (81 with ≥ 4 metals, 65 with ≥ 8), with per-metal energies, Ln–donor
distances, partial charges and coordination numbers.  Gen13 fed a 45-column "response" block from
these into a 2500-column tree model and measured +0.000.  Nobody had used the **energies**.

The amplitude is a differential quantity, so the natural construction is a two-way (ligand × metal)
additive fit of `complex_total_energy_eV`, which removes the free-ion energy of the metal and the
internal energy of the ligand, leaving an interaction whose slope in the ionic radius is a *computed*
selectivity.  On the 39 extractants with a complete 14-metal series that slope reaches:

- Spearman **+0.644** with the observed amplitude (p < 1e-4), −0.636 for its curvature term;
- leave-one-chemotype-out range [+0.598, +0.745];
- partial Spearman +0.645 after controlling for the number of metals the experiment measured, which
  is the corpus's nastiest confound (ρ = +0.49 with |a|).

That is stronger than the best 2D descriptor the programme has (`frac_donor_pairs_within_3`, −0.53),
and unlike it, it also tracks the magnitude and the curvature.

**It does not survive its own control.**  Only 1 of those 41 complete series has a constant
inner-sphere composition; 85 of 219 series change composition somewhere along the lanthanide series,
by up to 109 atoms.  A total energy that jumps by hundreds of eV when the builder adds a ligand copy
makes the "slope" a step function in where the recipe changed.  Recomputed honestly:

| construction | n extractants | Spearman with a | p |
|---|---|---|---|
| naive, complete 14-metal series | 39 | **+0.644** | < 1e-4 |
| contrasts inside constant-composition blocks only | 70 | +0.195 | 0.11 |
| whole panel, element counts as covariates | 71 | +0.216 | 0.07 |

The geometry descriptors fall the same way (`ln_donor_distance_std__mid`, the donor-distance spread
and a natural preorganisation proxy, goes from −0.567 to −0.226).  The honest residual signal is
around |ρ| ≈ 0.2, comparable to a weak 2D column.

---

## 4. The measured mode, rebuilt

Stage 2 established the largest lever in the programme: one measured separation factor plus the BLUP
with a leave-chemotype-out residual covariance takes the MAE from 0.541 to 0.266 under BP.  Two
things about that number needed fixing before it could be believed or built on.

**The ladder was not like for like.**  Excluding the support pair from scoring makes each rung a
slightly easier question than the last.  Gen15 excludes the support pairs of the *largest* budget at
*every* budget, so k = 0 is asked exactly the question k = 3 is asked.

**The covariance could still see the test cell's laboratory.**  Design BP removes a held-out cell's
publication from training, but the covariance was estimated from other chemotypes' held-out
residuals — which may come from the very publication BP masked.  Gen15 adds `mask_publication`, which
also excludes residual curves from the test cell's own publication.  Every number below uses it.

Extractant-macro MAE, design BP, identical held-out pairs at every budget, support = widest available
dZ:

| arm | k = 0 | k = 1 | k = 2 | k = 3 |
|---|---|---|---|---|
| `G14` prior + BLUP | 0.439 | **0.231** | 0.223 | **0.205** |
| `FLAT` prior + BLUP | 0.521 | 0.233 | 0.224 | 0.207 |
| `MEAN_CURVE` prior + BLUP | 0.545 | 0.238 | 0.229 | 0.207 |
| `NAIVE_LINE` (a ramp through the measured pairs, no corpus) | — | 0.240 | 0.237 | 0.236 |
| `LINE_PLUS_CURV` (the line plus a direction-conditional b) | — | 0.243 | 0.243 | 0.239 |
| `O_BOTH` (the cell's own coefficients) | 0.182 | — | — | — |

Sign accuracy on strong pairs goes 0.814 → 0.911 → 0.913 → 0.929; pair Spearman 0.474 → 0.561 →
0.580 → 0.640.

| contrast (BP) | point | 95 % CI | p | P1 |
|---|---|---|---|---|
| `G14@k1` − `G14@k0` | **+0.2073** | [+0.1286, +0.2637] | < 1e-4 | **passes** |
| `G14@k3` − `G14@k1` | +0.0256 | [+0.0176, +0.0324] | < 1e-4 | **passes** |
| `O_BOTH@k1` − `G14@k1` | +0.0493 | [+0.0361, +0.0668] | < 1e-4 | passes |
| `G14@k1` − `NAIVE_LINE@k1` | +0.0081 | [−0.0105, +0.0222] | 0.35 | no |
| `FLAT@k1` − `NAIVE_LINE@k1` | +0.0059 | [−0.0032, +0.0143] | 0.18 | no |
| `G14@k1` − `FLAT@k1` | +0.0022 | [−0.0119, +0.0126] | 0.72 | no |

Read that honestly.  **One measurement is worth +0.207 and it passes P1.**  Of that, the *ligand
chemistry* contributes +0.008 (not significant) and the *learned residual covariance* another +0.006
(not significant): once a separation factor exists, a straight line through it does nearly everything.
`LINE_PLUS_CURV` — giving the line the training fold's direction-conditional curvature, the cheapest
thing a corpus can add on top of a measurement — makes it worse, which is one more null for the
curvature.

---

## 5. Which pair to measure — and where the model finally earns its place

The locked protocol measures the widest available dZ.  The estimator-aware alternative is greedy
**D-optimal** selection under the residual covariance: pick the measurement that removes the most
posterior variance from the *whole* curve, not the one with the widest contrast.

This comparison is easy to get wrong, and the first attempt did.  Each strategy removes its own
support pairs from scoring, and taking out the widest-dZ pair removes the hardest pair from the test
set: on k = 0, where no measurement is used at all, that alone moved the score by 0.02.  The
admissible comparison runs every strategy in one pass and excludes the **union** of their support
pairs from scoring for all of them, which is what `scripts/g15_support.py` does — visible below in
the k = 0 column, identical at 0.4242 for all three strategies.

Extractant-macro MAE, common scoring set, **all five designs**:

| arm | B | BR | BQ | A | BP |
|---|---|---|---|---|---|
| `FLAT` — predict no separation | 0.4992 | 0.4989 | 0.5000 | 0.4985 | 0.5025 |
| `G14`, k = 0 | 0.4147 | 0.4126 | 0.4146 | 0.4112 | 0.4242 |
| widest dZ, k = 1 | **0.2172** | **0.2179** | **0.2192** | **0.2194** | **0.2261** |
| widest dZ, k = 3 | 0.1967 | 0.1971 | 0.1983 | 0.1986 | 0.2026 |
| D-optimal, k = 1 | 0.2196 | 0.2223 | 0.2215 | 0.2228 | 0.2313 |
| D-optimal, k = 2 | 0.1938 | 0.1951 | 0.1954 | 0.1948 | 0.2052 |
| **D-optimal, k = 3** | **0.1633** | **0.1639** | **0.1638** | **0.1649** | **0.1700** |
| random pair, k = 3 | 0.2143 | 0.2142 | 0.2153 | 0.2163 | 0.2230 |
| `NAIVE_LINE`, D-optimal k = 3 | 0.2254 | 0.2254 | 0.2269 | 0.2263 | 0.2347 |
| `O_BOTH`, D-optimal k = 3 | 0.1428 | 0.1436 | 0.1445 | 0.1438 | 0.1471 |

Strong-pair sign accuracy, D-optimal: 0.808 → 0.903 → 0.924 → **0.954** (BP; 0.958–0.961 in the
other four).  Pair Spearman: 0.474 → 0.561 → 0.610 → **0.722**.

**`G14@doptk3` spans 0.1633 to 0.1700 across the five designs — a range of 0.0068.**  That is the
property gen14 identified as the one a deployed model needs, and this arm has it more strongly than
gen14's own zero-shot arm (0.010).

Paired contrasts, all five designs (point estimate per design; every row below is LOCO-sign-stable
and 5/5 seeds positive unless stated):

| contrast | B | BR | BQ | A | BP | P1 |
|---|---|---|---|---|---|---|
| **D-optimal − widest @ k = 3** | **+0.0350** | **+0.0346** | **+0.0351** | **+0.0342** | **+0.0326** | **passes in all five** |
| D-optimal − widest @ k = 2 | +0.0145 | +0.0129 | +0.0154 | +0.0161 | +0.0132 | no (under the 0.02 margin) |
| D-optimal − widest @ k = 1 | −0.0033 | −0.0062 | −0.0030 | −0.0037 | −0.0053 | no — widest wins at k = 1 |
| widest − random @ k = 1 | +0.0991 | +0.0964 | +0.0983 | +0.1007 | +0.1026 | **passes in all five** |
| D-optimal k = 1 − k = 0 | +0.1930 | +0.1874 | +0.1918 | +0.1875 | +0.1929 | **passes in all five** |
| D-optimal k = 3 − k = 1 | +0.0578 | +0.0605 | +0.0586 | +0.0584 | +0.0613 | **passes in all five** |
| **`G14` − `NAIVE_LINE` @ dopt k = 3** | **+0.0641** | **+0.0636** | **+0.0640** | **+0.0620** | **+0.0646** | **passes in all five** |
| `G14` − `NAIVE_LINE` @ dopt k = 1 | +0.0155 | +0.0181 | +0.0158 | +0.0148 | +0.0108 | no (p 0.04–0.18) |
| `G14` − `FLAT` prior @ dopt k = 1 | +0.0053 | +0.0049 | +0.0047 | +0.0053 | +0.0025 | no (p ≥ 0.39) |
| `O_BOTH` − `G14` @ dopt k = 3 | +0.0203 | +0.0200 | +0.0192 | +0.0211 | +0.0229 | remaining headroom |

Every one of these is essentially design-invariant: the D-optimal gain spans 0.0326–0.0351 and the
model-over-line gain spans 0.0620–0.0646.  A result that does not move when the fold plan removes the
laboratory, the chemotype or the publication is not being carried by a leak.

Three things worth stating separately.

**Which pair you measure matters more than any estimator choice in this report.**  A random pair
costs +0.10 against the widest one at k = 1 — larger than the entire zero-shot gain of the last four
generations.

**At k = 1 the widest pair is right; from k = 2 the D-optimal sequence is.**  That is the expected
physics: one measurement should pin the linear ramp, which the widest contrast does best, and the
second and third should probe the curvature, which the widest-remaining pairs do not.  The deployed
predictor reproduces this unprompted — given TODGA it asks for La/Lu first and, once that is
supplied, for **Nd/Er**, a mid-series pair.

**This is where the model stops being decoration.**  At k = 1 the corpus adds +0.011 over a naive
line (not significant).  At D-optimal k = 3 it adds **+0.065, p < 1e-4, passing P1** — because the
line saturates (0.242 → 0.237 → 0.235; it has one amplitude to fit no matter how many measurements
it gets) while the BLUP keeps spending them against the full 14-metal covariance.  The gain comes
from the learned covariance rather than from the ligand descriptors — `G14` and `FLAT` priors are
within 0.003 of each other — but it is a corpus-learned object, and without it three measurements are
worth almost no more than one.

At D-optimal k = 3 the model reaches **0.163–0.170 extractant-macro MAE and 0.954–0.961 strong-pair
sign accuracy in every design**, against 0.499–0.503 for predicting no separation on the same pairs,
and sits 0.020–0.023 above what the cell's own two coefficients fitted to its complete 14-metal data
would give.  Three measurements chosen by the model, on a ligand whose chemical family and whose
laboratory are both absent from training, recover **67 % of the pairwise separation error**.

---

## 6. What the deployed predictor does

`scripts/g15_predict.py` takes a SMILES string and any measured separation factors and returns the
14-metal curve, all 91 pairwise log SF with 90 % intervals from the BLUP posterior, and the pair
worth measuring next.  The intervals behave as they should: for TODGA zero-shot the La/Lu prediction
is −1.05 with sd 1.12, and one measured La/Lu collapses that to sd 0.42 across the whole series.

The wide zero-shot interval is the honest one.  Zero-shot this programme knows the direction of
selectivity and very little else, and the interval says so.

---

## 7. Letting the measurement pick the curve shape — a null with a mechanism

Two results in this report point at each other.  §2 measured that a small alphabet of canonical
lanthanide curves is real and valuable — assigning a held-out cell to the nearest of four prototypes
scores 0.4438 against 0.5001, +0.056, passing P1 — and that no model can pick the letter from
structure.  §5 measured that once a separation factor is measured, the measurement and not the
chemistry is what identifies the cell.  So: use the measurement to pick the letter.

The prior over a held-out cell's curve becomes a mixture of Gaussians, one per prototype, each with
its own residual covariance; a measured log SF updates the weights by their likelihoods and the
prediction is the posterior mixture of the per-component BLUPs.  The pooled BLUP of §5 is the
K = 1 case.  Components are shifted so the prior mixture mean equals the gen14 curve, which makes
k = 0 *identical* to the gen15 arm (0.4443 for both) and every later difference attributable to the
measurement selecting a shape.

**The pre-check passes and the arm still fails.**  One measurement separates the components by a
median 2.8 nats at K = 4 and 3.2 at K = 6, and the posterior collapses onto a single component in
only 2 % of held-out cells — so the measurement *can* tell the prototypes apart.  It does not help:

| design BP | k = 0 | k = 1 | k = 2 | k = 3 |
|---|---|---|---|---|
| `POOLED` (K = 1, training-fold covariance) | 0.4443 | 0.2381 | 0.2183 | 0.1922 |
| `MIX6mean` (full mixture) | 0.4443 | 0.2576 | 0.2277 | 0.1965 |
| `MIX6hard` (commit to the best component) | 0.5285 | 0.2761 | 0.2508 | 0.2088 |
| **`MIX6meanPC`** (component means, **pooled** covariance) | 0.4443 | **0.2335** | **0.2136** | **0.1865** |

Every full-mixture arm loses — `MIX4hard` by −0.045 at k = 1 — and the hard rule is worse than the
mixture mean at every K and every k, which is the signature of over-commitment.

The attribution arm says why, and it is the opposite of the obvious guess.  Giving every component
the **pooled** covariance while keeping the prototype means turns the loss into a gain: `MIX6meanPC`
beats `POOLED` by +0.0046 / +0.0047 / +0.0070 at k = 1 / 2 / 3 (p = 0.049, 0.023, < 1e-4; 67 of 84
extractants and 5/5 seeds at k = 3; LOCO-sign-stable).  **The prototype means were never the
problem — the per-component covariances were.**  A component holding 30–120 cells cannot support
105 covariance parameters, and shrinking it halfway to the pooled matrix is not enough.

That gain is real and it is small.  At +0.002 to +0.007 it sits far under the pre-registered 0.02
margin, so it does not pass P1, and it is measured against the training-fold covariance route
(`POOLED@k3` = 0.1922) rather than the deployed leave-chemotype-out one (0.1700), which is the
stronger baseline.  The honest verdict is a **null with a mechanism**: the alphabet's oracle value
does not convert into a measured-mode gain, and the reason is covariance estimation, not shape
identification.

---

## 8. Eight parallel experiments: seven nulls and one honest positive

Eight independent arms were run against this bench, each under all five designs with the
chemotype-blocked bootstrap.  Their code and tables are under `exp/<slug>/`.

| arm | verdict | best BP | what it settles |
|---|---|---|---|
| `embed` pretrained chemical LMs | null | 0.5029 | ChemBERTa-77M-MTR/MLM and MoLFormer-XL carry neither magnitude nor curvature, and for the *direction* they are actively harmful: best 0.5687 macro accuracy against TOPO39's 0.8205 and an always-heavy floor of 0.5586. Every `DIR_*` embedding arm is worse than predicting no separation. |
| `kernel` 85 similarity-kernel arms | null | 0.4980 | Tanimoto/RBF/Matérn kernel ridge, kernel logistic, SVC, kNN and a GP all fail identically. The reason is structural: the chemotype hold-out is single-linkage Tanimoto 0.7 on the very ECFP bits the kernel uses, so the kernel extrapolates on 100 % of held-out cells by construction. |
| `phys3d` 28-column xTB geometry block | null | 0.4998 | Confirms §3 independently; the best honest fold-refit arm is 0.5013, worse than gen14. Note the caveat: the reference jobs were never run, so binding, strain and frontier-orbital energies are all NULL — this is "these xTB scalars are useless here", not "xTB is useless here". |
| `tabpfn` modern tabular learners | null | 0.4914 | TabPFN v2, CatBoost, monotone XGBoost, an isotonic bite fit, a spline GAM and an exhaustive 4602-term symbolic search. The one positive (TabPFN magnitude head, +0.0086) has p = 0.50 and is negative under design A. **The estimator axis is now closed**: twelve magnitude priors, all within ~0.007 of a constant. |
| `external` transfer to/from aqueous logK | null | 0.5520 | Gen14's classifier does not predict the logK slope sign of 273 independent ligands (0.440 against a 0.722 constant rule, permutation p = 0.81). The converse direction carries genuinely non-random signal — it beats all five of its label-permutation nulls, p = 0.004–0.018 — but scores 0.5520, indistinguishable from FLAT and worse than gen14. |
| `condfe` within-publication conditions | null | 0.4946 | The Frisch–Waugh transformation *does* repair the laboratory blow-up (a pooled conditions model goes from 5e11 to 0.4976 under BP). But the physics does not transfer: the acid association is real (within-publication Spearman +0.356 with \|a\|, LOCO- and LOPO-stable) and worth only +0.009 on a mean \|a\| of 0.295, and the inner LOPO CV shrinks the coefficient to exactly zero in 90 % of folds. |
| `labelerr` label noise and ceilings | null, **but see §1a** | 0.4959 | The labels are clean and every correction is null — and the ceilings in this report were wrong. This arm produced the most important number of the eight. |
| `decision` decision-relevant metrics | **positive, and bounded** | — | see below |

### What the model can and cannot be trusted to do

The `decision` arm asked the questions a chemist actually asks, with a permutation null behind every
number, and the answer splits cleanly in two.

**The direction call is genuinely deployable.**  Pair-sign accuracy under BP is **0.812 against 0.630
for "always prefer the heavier lanthanide"** — +0.18, consistent across all five designs, p ≤ 0.018,
z = 6.8–10.6 against a permutation null.  Told a ligand and a metal pair, the model says which
lanthanide goes into the organic phase, and it is right four times in five where the incumbent rule
is right two times in three.

**Every selection decision is at chance.**  Asked "which of these extractants best separates this
pair", the model's top-1 pick is no better than random (0.023 against 0.017), and with the laboratory
held fixed its ranking of candidates is not better than random either.  The reason is now localised:
gen14 emits *one bit per extractant*, and that bit is identical across the candidate set in 70–85 %
of real comparisons, so the model has nothing to rank with.

That is the honest state of the programme in one sentence.  **Zero-shot, this model answers "which
way" and cannot answer "which ligand".**  Answering "which ligand" needs the magnitude, and §1a says
the magnitude's honest headroom is smaller than we thought — or it needs a measurement, which §5
shows is worth more than any modelling move in four generations.

---

## Files

| | |
|---|---|
| `gen15/valuebench.py` | the bench: any `arm(ctx) -> (n_test, 2)` scored under five designs with gen13's metrics and paired bootstrap |
| `gen15/arms.py` | references, the oracle ladder that locates the information, and the query-design arms |
| `gen15/shape.py` | curvature constants, the sign-of-b classifier, the learned curve alphabet |
| `gen15/fewshot.py` | the BLUP, its posterior, support selection (widest / D-optimal / random), publication-masked covariance |
| `scripts/g15_locate.py` | §1 and §2's oracle ladder |
| `scripts/g15_shape.py` | §2's curvature ladder |
| `scripts/g15_fewshot.py` | §4's measured ladder and its contrasts |
| `scripts/g15_fewshot_sweep.py` | the estimator sensitivity sweep |
| `scripts/g15_uncertainty.py` | interval coverage, resolution and decision value |
| `scripts/g15_predict.py` | the deployable predictor: SMILES + measurements -> curve, 91 pairwise log SF with intervals, next pair to measure |
| `gen15/mixture.py`, `scripts/g15_mixture.py` | §7: the measurement-conditioned prototype mixture and its attribution arm |
