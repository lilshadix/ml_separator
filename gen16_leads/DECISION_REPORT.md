# Gen16 — decision report

*Six leads on the frozen gen13 cohort (fingerprint `4c3c6628ea0be949`: 521 cells, 90 extractants,
45 chemotypes, Kish n_eff 11.7 over extractants-per-chemotype).  Every number carries its regime:
{extractant-macro MAE of pairwise `log SF` | macro direction accuracy | measurements | Spearman}
× {design A | B | BR | BQ | BP} × {5 discovery seeds | 5 withheld confirmation seeds}.  Design BP
— chemotype held out **and** every training cell from a held-out cell's publication dropped —
is the design deployment is chosen under; design B never selects.  Pre-registration
`PRE_REGISTRATION.md`, sealed before any lead ran, body SHA-256
`d004c30388078ae97232af2537a1915360360a7cb9eda07ebe4076764c48f50e`.  Tables under `results/`,
refutations under `results/refutation/`, the single confirmation run under
`results/confirmation/`.  Written 2026-09-10.*

---

## Headline

**Gen16 ran 940 scored contrasts across six leads, sent every surviving claim to two blind
refuters, and confirmed one claim on five seeds no discovery agent had seen.  The confirmed claim
is real, it replicated, and it is narrower in two ways than its own lead believed: the direction
call saves about two of seven measurements when candidates are pooled across laboratories *and*
across folds, about one in six when the folds are made leak-free, and **nothing at all** — +0.005
measurements, interval containing zero in all five designs — when the candidates come from one
laboratory.  Every other lead closed or was left undecided.  The xTB thermodynamic cycle (L1) is not the
clean null its lead reported and it is not a live lead either: the registered estimator turns out
to have zero test–retest reliability, so that experiment had no power, while every post-hoc
estimator that does show a correlation fails the programme's mandatory diglycolamide control and
carries its signal only on the series whose composition varies — the artefact itself.  L1 is
**undecided**.  The curvature gate (L2) closed under its own rule on a headroom of +0.029 whose
interval includes zero, in a test whose minimum detectable effect (0.039) was twice its own
registered margin.  The corpus-expansion result (L4) was killed by its own refuter: a
featureless "measure the biggest family first" order reproduces all of it.  The covariance and
calibration lead (L5) produced no claim above the margin and two defects in the deployed
measured mode.  Nothing here beats the lean block set zero-shot, and the programme's honest
zero-shot gain is still gen15's +0.088 over predicting no separation at all.**

Six findings, in decreasing order of support.

1. **Confirmed, and narrower than it looks.**  Given a target lanthanide pair and a set of
   candidate extractants, discarding the candidates whose gen14 direction call disagrees with the
   requested direction cuts the expected number of measurements to the first useful candidate from
   **6.91 to 4.92 — a saving of 1.99 measurements, 28.8 %** — under BP on the discovery seeds, and
   **+1.909** on the withheld seeds, with the interval excluding zero, 5/5 seeds and LOCO
   stability in **all five designs** in both runs.  The refuters then narrowed it twice.  It is
   **between laboratories**: restricted to one publication's candidate set it is **+0.0046
   (0.07 %)**, interval containing zero in all five designs, where 69.6 % of tasks give every
   candidate the same call.  And its magnitude depends on a cross-fold pooling: rebuilt one fold
   at a time it is **+0.906 of an E_random of 5.52, 16.4 %** rather than 28.8 %, though it then
   captures a slightly *larger* share of the perfect-call ceiling (49.8 % against 47.3 %).  All
   three statements are true; the within-laboratory one governs deployment.
2. **The xTB verdict is corrected to *undecided*, in both directions.**  Correcting the complex
   energies with explicit per-species composition terms takes the slope's correlation with the
   amplitude from +0.644 (gen15's figure, reproduced exactly on its 39 series) to **−0.084** on 62
   extractants.  That is not a clean null: the corrected slope's split-half reliability is
   **−0.27** and its jackknife reliability is **0.000**, so the registered \|ρ\| ≥ 0.40 bar was
   unreachable in principle — *"the cycle-corrected slope, as constructed, does not correlate with
   itself"*.  But the post-hoc attenuation-free estimators that do reach +0.29 to +0.45 are not
   evidence either: `CYCLE_ADD` falls from **+0.291 to −0.026** when the 23 diglycolamides are
   removed (n = 43, p = 0.87), and on the 19 constant-composition series it is **+0.049** against
   **+0.476** on the 43 series whose composition varies.  The "signal" is the composition step
   re-entering.  Nothing in L1's registered family clears its own registered permutation bar
   (largest 0.602 against a 95th-percentile bar of **0.609**, both chemotype-level; gen15's +0.644
   is the same cell measured at the extractant level).  The free 2D competitor
   `frac_donor_pairs_within_3` (−0.42 to −0.53) is larger than every *composition-corrected* xTB
   estimator, though smaller than the uncorrected +0.644, and it fails the diglycolamide control
   too (−0.28, p = 0.07).  The reference-species energies (≈ 6.7 CPU-hours) remain the only
   construction that is exact *and* attenuation-free, and are delivered ready to submit — but
   gen16 supplies no positive evidence that there is anything for them to find.
3. **The corpus-expansion plan is not supported, and the reason is worth more than the plan.**
   A greedy A-optimal chemotype order beats random under BP by +0.050, but its sign flips under B
   and BQ, and a **featureless order — biggest chemotype first, zero descriptors — reproduces it
   and is positive in all five designs**, with A-optimal adding +0.005 (p = 0.36) on top.  At a
   budget of 6 chemotypes the A-optimal order already holds 72 % of the fold's training cells.
   What the retrospective simulation measures is corpus **volume**, not chemistry choice.
4. **The curvature gate is closed by rule, and underpowered by design.**  The honest leave-pair-out
   headroom is **+0.0287** — above the registered 0.02 margin, 61/90 extractants improved, 5/5
   seeds, LOCO-stable — with a percentile CI of [−0.0013, +0.0532] and p = 0.061, so P1 fails and
   the gate closed.  But the minimum detectable headroom is **0.039**, nearly twice the margin the
   gate was registered to detect, so this gate could not have opened at its own registered effect
   size whatever was true.  "Closed by rule, underpowered by design" is the honest reading, not
   "indistinguishable from zero".  About 0.044 of gen15 §2's in-sample +0.073 was the oracle
   absorbing the noise of the pair it was scored on, confirming gen15 §1a by an independent route.
5. **Two defects in the deployed measured mode.**  The leave-chemotype-out residual covariance is
   **indefinite on 100 % of its 430 estimates**, which makes Ledoit–Wolf shrinkage catastrophic
   (BP MAE 34.4 at k = 3) because greedy D-optimal selection seeks the negative directions; and
   the intervals over-cover badly — nominal 90 % covers **98.8 %** at k = 3 — driven by the fixed
   `NOISE_VAR = 0.09`.  Gen15 §7's `MIX6meanPC` gain does survive the deployed baseline it was
   never tested against, at **+0.0070** (p < 1e-4, all five designs), three times under the margin.
5b. **The largest sub-margin estimator effect is not the one gen15 pointed at.**  `LOWRANK2 −
   POOLED` at k = 2 is **+0.0151** under BP (p = 0.002, both intervals excluding zero, 5/5 seeds,
   LOCO-stable, same sign in all five designs) — 25 % under the margin, and twice the
   `MIX6meanPC` gain — though it vanishes at k = 3 (+0.0003).  Neither is deployable.

6. **The cohort is not throwing chemistry away, but the laboratory is.**  No relaxation of any
   gen13 cohort rule adds a single chemotype.  The 100 excluded bundle extractants are excluded
   because **only one lanthanide was ever measured on them**; they carry **53 chemotypes the
   cohort does not contain**, and measuring one second lanthanide on one compound per absent
   chemotype would take Kish n_eff from **11.67 to 27.4**.

---

## 1. What was run

| phase | what |
|---|---|
| 0 | three anchors reproduced; L6 cohort audit; environment verified |
| 1 | `PRE_REGISTRATION.md` sealed and committed with a hash commitment to five withheld seeds |
| 2 | five lead agents in parallel: L1, L2, L3, L4, L5 |
| 4 | six refuters, two per claim, blind to each other |
| 5 | the confirmation run, executed once |
| 6 | this report, then two adversarial audits of it (numbers; claims and protocol), whose findings are folded in below |

**Deviation from `START_HERE.md` §8's layout, stated rather than hidden.**  The brief asks for
`headline_tables/`, `metrics/`, `bootstrap/`, `predictions/`, `scripts/`, `tests/` following the
gen13 layout.  Gen16 uses a single `results/<lead>/` tree instead, plus `scripts/` and `tests/`.
The mapping: headline tables are the `*_board.csv` and the tables in each `results/<lead>/
<LEAD>_REPORT.md`; metrics are the `per_extractant*` and `*_metrics.csv` files; bootstrap output
is in the `contrasts_*.csv` files (point, both intervals, p, per-seed signs, LOCO); predictions and
other large dumps are excluded by design and digested in `results/MANIFEST.sha256`.

**Anchors** (`results/anchors/ANCHORS.md`, guarded by `tests/test_anchors.py`).  All three
reproduce exactly: gen13 stage-3 `G13_ET_TOPO39` under BP = `0.7683085207475452`; gen14 `G14`
under BP = `0.5000794414203691`; `FLAT` = `0.5885062528901843`; `G14 − FLAT` = +0.0884
[+0.0010, +0.1473], p = 0.047.  The fold plan hashes identically across processes and across
`PYTHONHASHSEED` values; the only `hash()` on the bench path takes an int tuple, which CPython
does not salt.  No dependency has moved since gen13.

**Comparison accounting.**  **940 contrast rows: 126 registered, 814 exploratory** — by lead,
L1 90 rows (5 energy models × 3 sets × 3 targets × slope and quadratic term, 2 registered),
L2 20 (4 arms, 5 registered), L3 85 (4 arms over 455 tasks, 15 registered), L4 **515**
(4 acquisition orders × 6 budgets, 24 registered), L5 230 (25 arm × k combinations, 80 registered).  **L4's 515 includes 80 rows from a
discarded dry run retained in `results/L4/_dry/`**, 4 of them labelled registered; counting only
the eleven final files gives 860 rows, 122 registered and 738 exploratory, which is what an
independent refuter counted from the same tree.  The larger family is used throughout because it
is the larger count of things actually evaluated; note that BH over 940 gives the confirmed claim
a *smaller* q than BH over the registered 126 (0.025 against 0.039), so the inclusive count is not
the more conservative of the two — the registered-family q is the one to quote.

Benjamini–Hochberg within the registered family of 126, quoted beside every raw p this report
uses:

| contrast | raw p (across five designs) | BH q |
|---|---|---|
| **L3a `G14_saved_vs_0`** (§2) | 0.004–0.010 | **0.020–0.042** |
| L5 `MIX6meanPC − POOLED` @k3 (§7) | < 1e-4 | < 0.001 |
| L5 `LOWRANK2 − POOLED` @k2 (§7) | 0.002–0.336 | 0.011–0.547 |
| L3c `G14@k1 − NAIVE` on Spearman (§3) | 0.013–0.026 | 0.048–0.076 |
| L2 `OCURVLPO_vs_G14` (§5) | 0.031–0.129 | 0.080–0.246 |
| L4 `AOPT_vs_RANDOM` (§6) | < 1e-4–0.974 | < 0.001–0.993 |
| L3c `G14@k1 − NAIVE` on regret (§3) | 0.335–0.997 | 0.547–0.997 |

Over the combined 940 the confirmed claim's BP row is at q = 0.025.  L1's two registered rows have
q = 0.518 within its own family.

---

## 2. L3a — the confirmed claim, and its scope

A **task** is one (split seed, unordered metal pair); its candidates are the held-out extractants
that measured that pair, several cells collapsed by the median; tasks with fewer than 5 candidates
are dropped (none were).  A candidate **succeeds** for a requested direction if its observed
`log SF` has that sign and \|log SF\| ≥ 0.3.  Without a model the chemist measures in random order,
expected draws `(N+1)/(K+1)`; with it, candidates whose direction call disagrees are deferred.

**Table 2a — `saved`, in measurements, all five designs, both seed sets**

| design | discovery | 95 % CI | confirmation | 95 % CI | seeds | LOCO |
|---|---|---|---|---|---|---|
| B | +2.207 | [+0.521, +4.118] | **+2.086** | [+0.234, +4.170] | 5/5 | stable |
| BR | +2.149 | [+0.455, +3.985] | **+2.170** | [+0.252, +4.195] | 5/5 | stable |
| BQ | +2.179 | [+0.468, +4.070] | **+2.123** | [+0.248, +4.196] | 5/5 | stable |
| A | +2.492 | [+0.771, +4.544] | **+2.285** | [+0.620, +4.366] | 5/5 | stable |
| **BP** | **+1.988** | [+0.443, +3.781] | **+1.909** | [+0.360, +3.760] | 5/5 | stable |

`E_random` = 6.906 in both runs.  Permutation p = 0.0000 at its 1/2000 resolution in every cell.
BP is the **smallest** of the five savings, the ordering expected if no publication leak were
carrying it.

**Table 2b — the two comparators the pre-registration demands**

| comparator | `saved` under BP | note |
|---|---|---|
| no model (random order) | 0 by definition | the floor |
| **"always heavier"** (registered cheapest competitor) | **0.000000** | zero *by construction*: a constant call cannot partition a candidate set |
| 13-column gen6 donor census, used identically | **−3.14** | it *costs* measurements; G14's increment over it is +5.13 |

**Table 2c — what the refuters did to it** (`REFUTATION_LOG.md` §2)

| check | `saved` under BP | verdict |
|---|---|---|
| headline, re-derived independently from the sealed text | +1.988 (6 dp) | held |
| diglycolamides (sc009) removed | +1.459 | held, positive in all five |
| sc009-only candidate sets | +0.386 | the effect is *between* chemotypes |
| n_metals-stratified permutation | real − shuffled +1.886 | held (~5 % attributable) |
| leave-one-publication-out, all 58 | [+1.506, +2.428], 0 sign flips | held |
| **candidate sets within one publication** | **+0.0046**, CI [−0.016, +0.022] | **dents, materially** |
| candidate sets within one chemotype | +0.334, rule fails 5/5 | dents |
| **rebuilt one fold at a time (fold purity)** | **+0.906 of an E_random of 5.52 (16.4 %)** | **dents the magnitude** |

Two dents, and both must travel with the number.

**Scope.**  The within-publication test rests on 4 of 58 publications, 31 extractants and 11
chemotypes, 18 of the 31 being diglycolamides — which is why it is recorded as material rather
than fatal.  On those same tasks a perfect sign call would save 0.414 measurements, so the regime
is not vacuous: the model captures 1.1 % of the available headroom there against 47.3 % pooled.

**Magnitude.**  A task pools the held-out predictions of all five folds of a seed, so candidate *X*
was predicted by a model whose training set contained candidate *Y*.  Rebuilt one fold at a time,
so that no candidate in a task was predicted by a model trained on another candidate in it, the
saving under BP is **+0.906 of an E_random of 5.52, i.e. 16.4 %** against the quoted 28.8 % —
**46 % of the headline magnitude**.  The effect itself survives: as a fraction of the perfect-sign
ceiling the fold-pure run captures **49.8 %**, slightly *more* than the pooled run's 47.3 %.  So
"about two of every seven measurements" is the cross-fold figure; the leak-free figure is about
one in six.

**What this supports.**  *Told a ligand and a metal pair, from a laboratory the model has never
seen, the model says which lanthanide enters the organic phase, and used as a filter over a
literature-wide candidate list it removes about two of every seven measurements.*  It does **not**
support "screen your own shortlist with this and save two measurements".

## 3. L3c — ranking at k = 1 is a null, and the no-model line wins on rank

Every candidate given one measured pair (its widest `dZ` excluding the target):

| arm, design BP | regret (log10) | cross-extractant Spearman |
|---|---|---|
| random | 1.762 | 0 |
| `G14` zero-shot | 1.454 | 0.352 |
| `NAIVE_LINE@k1` | **0.339** | **0.799** |
| `G14@k1` | 0.361 | 0.772 |

Registered contrast `G14@k1 − NAIVE_LINE@k1`: on regret +0.003 / +0.002 / −0.003 / −0.017 /
−0.021 (B/BR/BQ/A/BP), p 0.34–1.00 — a null; on Spearman it is **negative in all five designs**
(−0.022 to −0.027, p 0.013–0.026), i.e. the corpus makes ranking *worse*.  One measurement is
worth 1.09 log units of regret against the zero-shot model; the corpus adds nothing on top of it.
**L3b** (ranking within a chemotype) was registered as conditional on L1 *being positive*, and
L1 not being positive,
was **dropped rather than run**.

## 4. L1 — the thermodynamic cycle: a corrected verdict

Gen15 measured a Spearman of **+0.644** between a two-way xTB energy slope and the observed
amplitude and killed it with a composition control.  L1 rebuilt the bookkeeping exactly.

**Table 4a — the dose–response, ρ(slope, `a`)**

| bookkeeping | model | S8 (n = 62) | S14 (n = 39) |
|---|---|---|---|
| none | `NAIVE` | +0.392 | **+0.644** |
| right species, wrong counts | `SPECIES_NFILLCOL` | +0.374 | +0.264 |
| element counts as covariates (gen15's B) | `ELEM` | +0.313 | +0.396 |
| exact counts, no free δ (*subset*: n = 19 / 11) | `SPECIES_CONST` | +0.167 | +0.155 |
| exact counts + free δ (**registered**) | `SPECIES` | **−0.084** | −0.048 |
| *cheapest competitor*: `frac_donor_pairs_within_3` (free, 2D, no xTB at all) | — | **−0.416** (n = 61) | **−0.457** (n = 39); −0.530 over all 80 |

`SPECIES_CONST` is a subset row: it is fitted only on the series whose ligand count is constant,
so its n is **19 on S8 and 11 on S14**, not the column headings.

**The registered family-wise bar, which the lead computed and did not quote.**  The
pre-registration requires a chemotype-level permutation null taking the maximum \|ρ\| over the
4 models × 3 sets.  Its 95th percentile is **0.609**; the largest value the family attains is
**0.602** (`NAIVE` on S14), family-wise p = **0.057**.  **Nothing in L1's registered family clears
its own bar for the primary target `a` — gen15's +0.644 included.**  Two things must be said
plainly about that sentence.  First, the permutation statistic is computed on **chemotype means**,
so the `NAIVE`/S14 cell is 0.602 there and +0.644 in Table 4a, which is extractant-level; they are
the same cell at two aggregation levels and the bar is only meaningful at its own.  Second, the
claim is target-specific: for the *curvature* target `b` the same 12-member family **does** clear
its bar (observed 0.750 against 0.604, family-wise p = **0.003**), driven by the same `NAIVE`/S14
cell — which is consistent with §4's finding that the naive slope and its curvature are one step
function read twice, and is therefore evidence about the composition artefact rather than about
selectivity.  Against the bar for `a`, the whole of Table 4a is noise,
and the free 2D competitor is larger in magnitude than every *composition-corrected* xTB
estimator in it — though not larger than the uncorrected `NAIVE` +0.644, and it fails the
diglycolamide control itself (−0.280 on 42 extractants, p = 0.072, CI spanning zero), which is the
same control that kills `CYCLE_ADD` below.  No descriptor in this section, xTB or 2D, survives
that control on S8.

Two supporting facts about the bookkeeping: `n_fill` counts donor **sites**, not molecules (nitrate
is bidentate), and every complex decomposes exactly as metal + n·ligand + n·NO₃ + n·H₂O with no
residue.  The fitted **water** coefficient lands within 0.34 eV of an independent per-element
regression (−138.88 against −139.22 eV); the fitted **nitrate** coefficient sits 17.8 eV below it
(−431.97 against −414.21 eV), as it must for a charged, strongly bound bidentate anion.  So the
model is doing the cycle's arithmetic.  The naive slope and its curvature are **the same step
function read twice** (ρ = −0.981).

**Table 4b — why the registered verdict is wrong** (`REFUTATION_LOG.md` §4)

| diagnostic | value | consequence |
|---|---|---|
| split-half reliability of the `SPECIES` slope (odd vs even metals) | **−0.272** | the slope does not reproduce itself |
| jackknife reliability (between-extractant sd 0.203 eV vs within-series SE 0.329 eV) | **0.000** | max attainable \|ρ\| ≈ 0; the 0.40 bar was unreachable |
| fraction of an injected real trend retained by `SPECIES` | **0.42** | the correction removes ~58 % of a real signal |
| Fisher CI of the `SPECIES_CONST` row at n = 19 | [−0.337, +0.597] | could never separate +0.167 from +0.5 |

**Table 4c — and why the positive side fails too.**  Both refuters built attenuation-free
corrections (fixed reference energies, no free per-series parameter) and both got a positive.
Neither survives the controls the brief makes mandatory:

| estimator (post hoc, unregistered) | S8, n = 62 | **without diglycolamides**, n = 43 | constant-composition series | composition-varying series |
|---|---|---|---|---|
| `CYCLE_ADD` (lens A) | +0.291 (p = 0.022) | **−0.026** (p = 0.87) | **+0.049** (n = 19, p = 0.84) | **+0.476** (n = 43, p = 0.001) |
| `FIXCYC_CONSTNLIGS` (lens B) | +0.339 (p = 0.007) | **+0.037** | — | — |
| `FIXCYC_CONSTCOMP` (lens B) | +0.326 (p = 0.010) | **+0.010** | — | — |
| `FIXCYC_ALL` (lens B) | +0.290 (p = 0.022) | **−0.026** | — | — |
| `FIXCYC_CONSTNLIGS_GAMMACONST` (lens B) | +0.235 (p = 0.066) | +0.249 | — | — |

**Four of the five collapse when the 23 diglycolamides are removed**, from +0.29…+0.34 to
−0.03…+0.04.  `CYCLE_ADD`, the only one for which the composition split was computed, lives
**almost entirely on the series whose inner-sphere composition varies** — +0.48 there against
+0.05 where it is constant.  That is the composition step re-entering through a different door,
which is the artefact the whole lead exists to remove.  The one variant that survives sc009
removal, `FIXCYC_CONSTNLIGS_GAMMACONST` at +0.235 → +0.249, is not significant to begin with
(p = 0.066) and is below the registered 0.25 threshold.  All five are unregistered post-hoc arms,
so applying the registered 0.25/0.40 thresholds to them would be exactly the "nominally
significant result from an unregistered arm" the brief lists as *not* success.

(The LOCO minimum of each of these arms equals its no-diglycolamide value, because sc009 is the
chemotype whose removal hurts most — the two controls coincide here.)

**Verdict: L1 is undecided — not closed, and not open.**  The registered estimator is too noisy to
detect anything (so "closed" overstates), and every estimator that shows a correlation fails the
diglycolamide control and carries its signal on the composition step (so "promising" overstates in
the other direction).  Nothing in the registered family clears its own permutation bar, and a free
2D column beats all of it.

**Stage 2 is still the right experiment, and its prior should be low.**  With real reference
energies the composition is subtracted at true species energies with **no free parameter and no
fitted γ**, which is the only construction that is simultaneously exact and attenuation-free — it
is the only way the question has ever actually been asked.  But gen16 supplies no positive
evidence that there is signal to find, and three reasons to doubt it: the 2D competitor is larger,
the registered permutation bar is unmet by the whole family, and every post-hoc positive is
composition in disguise.  It is worth 6.7 CPU-hours because it is cheap and decisive, not because
it is promising.  Hand-over: `results/L1/L1_STAGE2_HANDOVER.md`, 361 array tasks, 1 core and 2 GB
each, ≈ 6.7 CPU-hours (≈ 20 min wall at 20 concurrent), then one local command.  `xtb` is absent
from this machine, verified by a disk walk.

## 5. L2 — the curvature gate

`O_CURV_LPO` is gen14's amplitude with the cell's own second coefficient refitted **without the
two metals of the scored pair**.

| arm | B | BR | BQ | A | BP |
|---|---|---|---|---|---|
| `FLAT` | 0.5885 | 0.5885 | 0.5885 | 0.5885 | 0.5885 |
| `G14` | 0.4932 | 0.4906 | 0.4905 | 0.4921 | 0.5001 |
| `O_CURV` (in sample) | 0.4216 | 0.4218 | 0.4233 | 0.4168 | 0.4272 |
| **`O_CURV_LPO`** | 0.4634 | 0.4638 | 0.4642 | 0.4599 | **0.4714** |

Headroom `G14 − O_CURV_LPO`: +0.0298 / +0.0268 / +0.0264 / +0.0323 / **+0.0287**.  Under BP the
percentile CI is [−0.0013, +0.0532] **and the BCa CI is [+0.0000, +0.0544]**; P1 requires *both*
intervals to exclude zero and p < 0.05, and p = 0.061 (q = 0.08–0.25) fails under either reading,
so the gate closed.  Only design A excludes zero on the percentile interval, and A never selects.
The BCa lower bound is +5.7 × 10⁻⁶ — a six-millionths margin, recorded as a temptation in
`REFUTATION_LOG.md` §5 and not acted on.

**The gate was also underpowered to detect its own registered effect.**  Bootstrap SE 0.014 puts
the minimum detectable headroom at **0.039**, nearly twice the registered 0.02 margin, so no true
headroom of the registered size could have opened this gate.  The honest reading is *closed by
rule, underpowered by design*, with a point estimate of +0.029 [−0.001, +0.053] that is above the
margin, improves 61 of 90 extractants, and is LOCO-stable and 5/5 seeds positive.  The `r0`
reparametrisation was not run, per the registered stopping rule.

## 6. L4 — corpus-level experimental design

**Table 6a — learning curve, design BP, macro MAE by budget in chemotypes**

| order | 6 | 9 | 12 | 16 | 20 | 24 | full (≈ 29) |
|---|---|---|---|---|---|---|---|
| `RANDOM` | 0.626 | 0.608 | 0.578 | 0.553 | 0.524 | 0.511 | 0.500 |
| `MAXMIN` | 0.599 | 0.613 | 0.609 | 0.591 | 0.526 | 0.512 | 0.500 |
| `UNCERT` | 0.663 | 0.636 | 0.606 | 0.524 | 0.520 | 0.517 | 0.500 |
| **`AOPT`** | **0.570** | **0.523** | **0.509** | **0.492** | 0.503 | 0.503 | 0.500 |

**Table 6b — registered contrasts, area between curves**

| contrast | B | BR | BQ | A | BP | five-design rule |
|---|---|---|---|---|---|---|
| `AOPT − RANDOM` | −0.012 | +0.023 | −0.003 | +0.037 | **+0.050** | **fails** (sign flips) |
| `AOPT − MAXMIN` | −0.004 | +0.042 | +0.010 | +0.053 | +0.058 | fails |
| `UNCERT − RANDOM` | −0.031 | −0.004 | +0.003 | −0.018 | −0.011 | fails |
| `UNCERT − MAXMIN` | −0.023 | +0.014 | +0.017 | −0.002 | −0.003 | fails |

**And then the refuter.**  A featureless order — chemotypes sorted by their count of
well-determined cells — gives `SIZE − RANDOM` = +0.028 / +0.019 / +0.026 / +0.023 / **+0.045**,
**positive in all five designs**, while `AOPT − SIZE` = +0.005 (p = 0.36), below the margin and
negative under B and BQ.  Spearman(A-optimal pick position, chemotype **well-determined**-cell
count) = −0.63 under BP (−0.45 against the plain cell count; ≈ −0.01 for a random order); at
budget 6 the A-optimal order holds 72 % of the fold's training cells against random's 18 %.

**Verdict: not supported, and the prospective ranking inherits the defect.**  The ranked lists of
95 bundle and 273 external candidates (`results/L4/ranking_pool_*.csv`) are ordered by the same
criterion that the refuter showed to be a size proxy — their top 20 is dominated by
diglycolamides, the family that already holds 375 of 521 cells, and the criterion takes only 43
distinct values over 95 candidates.  **They are delivered as an artefact of the registered
protocol, not as a recommendation.**  The defensible corpus-expansion advice in this report comes
from L6 instead (§8), and it does not need a model.

## 7. L5 — covariance and calibration

Verification first: the ported evaluator reproduces `G14@doptk3` = 0.1700393901 and
`G14@doptk1` = 0.2313450953 under BP to 0.0e+00, and the mixture port reproduces all 11 of gen15
§7's locked training-route modes to 6 dp.

**Table 7a — extractant-macro MAE under BP, D-optimal support, common scoring set**

| covariance | k = 0 | k = 1 | k = 2 | k = 3 |
|---|---|---|---|---|
| `POOLED` (deployed) | 0.4052 | 0.2238 | 0.2009 | **0.1674** |
| `LW` (Ledoit–Wolf) | 0.4052 | 0.2214 | 0.6539 | **34.41** |
| `LOWRANK2` | 0.4052 | 0.2166 | 0.1862 | 0.1667 |
| `LOWRANK3` | 0.4052 | 0.2191 | 0.1895 | 0.1683 |
| `HIER` | 0.4052 | 0.2205 | 0.1963 | 0.1711 |
| `MIX6meanPC` | 0.4052 | 0.2236 | 0.1974 | **0.1604** |
| `NAIVE_LINE` | — | 0.2310 | 0.2251 | 0.2243 |

No registered contrast reaches the 0.02 margin in any design.  **Two come close and neither is
deployable:**

| contrast | BP | 95 % CI | p | q | designs | margin |
|---|---|---|---|---|---|---|
| **`LOWRANK2 − POOLED` @ k = 2** | **+0.0151** | [+0.0067, +0.0232] | 0.002 | 0.011 | same sign in all five, 5/5 seeds, LOCO-stable | 25 % under |
| `MIX6meanPC − POOLED` @ k = 3 | +0.0070 | [+0.0044, +0.0111] | < 1e-4 | < 0.001 | same sign in all five, 5/5 seeds, LOCO-stable | 3× under |

`LOWRANK2` is the **larger** of the two and was not in gen15's field of view at all; it is
significant only at k = 2 and vanishes at k = 3 (+0.0003, 4/5 seeds, LOCO-unstable), and under
design A it is p = 0.34.  `MIX6meanPC` is the one that matters historically: it is the comparison
against the **deployed** baseline that gen15 §7 never ran, and it holds.  Both are positive and
**sign-invariant across the five designs**, and both are under the margin.  Neither is
*significance*-invariant: `LOWRANK2` is p = 0.34 under design A with an interval spanning zero,
and `MIX6meanPC` reaches significance only at k = 3.  Neither changes what is deployed.

**Two defects.**  (i) The deployed leave-chemotype-out residual covariance is **indefinite on
100 % of its 430 estimates**; Ledoit–Wolf deepens the negative directions and greedy D-optimal
selection seeks them, which is why `LW` explodes rather than merely underperforming.  (ii)
Coverage: nominal 90 % intervals cover **98.8 %** at k = 3 and 92.6 % at k = 0, because
`NOISE_VAR = 0.09` alone exceeds the realised squared error at k = 3 (RMSE 0.248).  The
programme's measured-mode intervals are conservative by roughly 2.5× in width.

**Calibration of the zero-shot direction probability.**  The registered deliverable is the
*calibrated* (inner-fold Platt) probability, and it fails the registered rule on its own terms:
under BP its Brier is **0.255** and its ECE **0.301**, three times the 0.10 bar, and it is worse
than the uncalibrated probability on both — Platt recalibration makes calibration worse in every
design.  The **raw** probability is much the better object (Brier **0.167**, macro accuracy 0.821,
against a constant base rate at Brier 0.313) and beats that constant in all five designs, but it
was never the registered deliverable and its own ECE of **0.139** is above the bar too.  **So
nothing here is deliverable as a calibrated probability**, and the honest summary is that the
direction call is well separated but over-confident.

## 8. L6 — the cohort audit, and the only corpus advice this report will give

`gen16/cohort_audit.py` reproduces all three of gen13's key modes exactly, fingerprints included
(exact 521/90 `4c3c6628ea0be949`; relaxed 509/90; series 597/89), and reproduces the frozen
chemotype partition by re-clustering all 190 bundle structures.

**Every bundle extractant with ≥ 2 lanthanides measured anywhere is already in the cohort.  No
single relaxation adds a chemotype — the maximum over seven relaxations is +0.**  The gen5
`min_rows` failure does not recur.

The 100 excluded compounds are excluded because only one lanthanide was ever measured on them
(Eu 567 rows, Pr 96, Nd 35).  **91 of the 100 appear in only a single publication each**, and the
100 together span 23 publications, the largest of which contributes 52 of them.  They carry
**60 chemotypes, 53 of them absent from the cohort**, and 77 sit at ECFP4 Tanimoto < 0.7 from every
kept extractant.

| counterfactual: a second lanthanide measured on … | extractants | Kish n_eff | ratio |
|---|---|---|---|
| — (frozen cohort) | 90 | 11.67 | 1.00× |
| one compound per absent chemotype (53) | 143 | **27.38** | **2.35×** |
| the 60 in absent chemotypes | 150 | 28.55 | 2.45× |
| all 100 | 190 | 12.27 | 1.05× |

Adding all 100 barely helps because 28 of them are themselves diglycolamides.  **This is the
programme's binding constraint stated as a laboratory action, and it needs no model.**

---

## 9. What is statistically supported, and what is not

**Supported** (the lead's registered decision rule met in all five designs, refuters unable to
remove it, replicated on withheld seeds):

- the direction call saves ≈ 2 of 6.9 expected measurements when candidates are pooled across
  laboratories (§2), **with both limitations of §2 attached**: within one laboratory it saves
  +0.005 with an interval containing zero, and rebuilt fold-pure the magnitude is 16.4 % rather
  than 28.8 %.  Note the rule met is L3a's registered rule of `PRE_REGISTRATION.md` §3, not P1:
  see the disclosed deviation in §10, item 10.

**Measured and reported, but not supported as claims:**

- `LOWRANK2 − POOLED` at k = 2 (+0.0151) and `MIX6meanPC − POOLED` at k = 3 (+0.0070): real and
  sign-invariant across the five designs, both under the pre-registered margin, and neither
  significant in every design (§7);
- the L2 curvature headroom of +0.029: above the margin, interval includes zero, and the test's
  minimum detectable effect was twice its own margin (§5);
- a featureless "biggest chemotype first" acquisition order beating random in all five designs
  (§6) — found by a refuter, and a refuter check can kill a claim, never resurrect one;
- composition-position descriptors correlating with the amplitude at ρ up to +0.83 (§12, gen17
  lead 1) — post hoc, small n, from a geometry builder's recipe, and correlated +0.79 to +0.86
  with the naive xTB slope this report calls a composition artefact.

**Not supported:**

- any new representation beating the lean block set zero-shot;
- the corpus-expansion plan as a *chemistry* result (L4 killed by its featureless control);
- ranking candidates at k = 1 with the corpus (L3c null, and negative on rank correlation);
- any covariance estimator or calibrated interval at the 0.02 margin (L5);
- any cohort expansion available from relaxing a filter (L6).

**Undecided, and explicitly not a null:** L1 (§4).  The registered test had no power and every
post-hoc positive fails the diglycolamide control.

**Nothing here supports an equivalence claim.**  The minimum detectable effect at 80 % power is
0.039 for the L2 curvature headroom, against its own registered margin of 0.02; the L1 registered
estimator has reliability 0.000 and so has no detectable effect at all.

---

## 10. Defects found during this work

1. **Gen15 §7's `MIX6meanPC` was never compared with the deployed route.**  It is, here: the gain
   survives at +0.0070 (§7).
2. **The deployed residual covariance is indefinite on 100 % of its 430 leave-chemotype-out
   estimates** (§7).  It has not invalidated a published number — the BLUP is still a projection —
   but it makes any shrinkage estimator unsafe and it should be projected onto the PSD cone.
3. **The deployed measured-mode intervals over-cover by ≈ 2.5× in width** (§7), traced to
   `NOISE_VAR = 0.09`.
4. **Gen15 §3's two composition-controlled rows (n = 70/71) correspond to no construction in the
   frozen scripts**; the reproducible values are 79/+0.104 and 81/+0.203.  The conclusion is
   unaffected (every composition-controlled construction lands at \|ρ\| ≤ 0.22).
5. **Gen15's "39 complete series" is the energy-complete set; the geometry-complete set is 41.**
   Both reproduce exactly once the distinction is made.
6. **`n_fill` counts donor sites, not molecules** — nitrate is bidentate.  Any future use of the
   3D dataset's composition columns must use `n_NO₃ = ⌊n_fill/2⌋`-style decomposition, which L1
   verified against every one of the 1155 complexes.
7. **The frozen `key_mode="series"` uses the *relaxed* condition columns**, not exact + series; the
   docstring is silent and a re-implementation that reads it literally gets 557 cells instead of
   597.  A regression test now pins it.
8. **`gen13sep/arms_stage2.py:330` hashes bytes** with the salted built-in `hash()`, so two gen13
   arms (`S3_HIER`, `S3_EXT_LEVEL`) are not guaranteed bit-identical across processes.  Not on the
   gen14/gen15 bench path.
9. **A deviation from the sealed pre-registration in L1**, disclosed by the lead and confirmed by a
   refuter: the sealed text wrote `n_fill × γ_species`, the implementation used per-species
   molecule counts (because `n_fill` counts donor sites).  Both were run; the verdict is the same
   under either.
10. **A second deviation, on the confirmed claim, disclosed here for the first time.**  The sealed
    §0 fixes inference as `gen13sep.inference.paired_contrasts` — 10 000 replicates, **percentile
    and BCa** intervals — and says no new bootstrap is written.  L3 wrote its own chemotype-blocked
    bootstrap at **2 000 replicates with a percentile interval only**, because its resampling unit
    is the task rather than the extractant and the frozen function cannot express that.  The
    consequence is precise: the claim meets **L3a's registered rule** (§3 of the pre-registration)
    and the percentile half of P1; **the BCa half of P1 was never computed for it**.  The
    permutation null, which is the claim's primary evidence, is unaffected.  §9 says "the lead's
    registered rule" rather than "P1" for this reason.
11. Orchestrator process failures are recorded in `REFUTATION_LOG.md` §6.

---

## 11. Guardrails — what a reader must not do with these numbers

1. **Do not quote "the model saves two of seven measurements" without both dents.**  Pooled and
   cross-fold it is +1.99 of 6.91 on the discovery seeds (28.8 %) and +1.91 on the confirmation
   seeds (27.6 %); rebuilt fold-pure it is +0.91 of 5.52 (16.4 %); inside one laboratory's
   candidate set it is +0.005 with an interval containing zero.  The last is the one a screening
   chemist experiences.
2. **Do not read L1 either way.**  It is not "xTB is useless here" — the registered estimator has
   reliability 0.000 and so could not have detected anything.  It is equally not "xTB looks
   promising": every post-hoc estimator that correlates fails the diglycolamide control (+0.29 →
   −0.03) and carries its signal only where the composition varies (+0.48 against +0.05).  L1 is
   undecided, the free 2D competitor is still larger than any of it, and nothing in the registered
   family clears its own permutation bar.
3. **Do not use the L4 prospective ranking as a purchase list.**  It is ordered by a criterion its
   own refuter showed to be a proxy for family size.
4. **Do not quote the +0.83 composition-position correlations as a result.**  They are post hoc,
   n = 27–29, computed from a geometry builder's recipe rather than from measurement, and — the
   fact that matters most — they correlate **+0.79 to +0.86 with the naive xTB slope**, the
   quantity this report calls a composition artefact read twice.  They are a parametrisation of
   the composition step, not an independent observable.
5. **Do not treat `MIX6meanPC` or `LOWRANK2` as deployable.**  Both are under the margin.  The
   honest statement is that they are the only structured estimators in this programme that have
   ever beaten their pooled control, and that `LOWRANK2` is the larger of the two only at k = 2.
6. **Do not use the measured-mode intervals as calibrated.**  They cover 98.8 % at nominal 90 %.
7. **Do not quote any design-B number**, and do not quote `MEAN_CURVE` (0.622) as a baseline: it
   is worse than predicting no separation at all (0.589).
8. **Do not treat the L6 counterfactual Kish figures as a promise.**  They assume a second
   lanthanide is measurable on those compounds under comparable conditions, which the audit cannot
   verify.

---

## 12. Recommendation

**The one thing to do in a laboratory** is not a modelling change.  Fifty-three chemotypes are
absent from this corpus because someone measured europium alone; one second lanthanide on one
compound per absent chemotype would multiply the programme's effective sample size by **2.35×**,
which is larger than every modelling effect this generation measured put together.  That is L6,
it needs no model, and it has been the programme's standing recommendation since gen6.

**The one experiment to run on a cluster**, and it is cheap rather than promising, is L1 Stage 2:
361 array tasks, ≈ 6.7 CPU-hours, one core and 2 GB each, then one local command.  Gen16's
contribution is to say precisely why it is worth 6.7 CPU-hours and no more: every estimator that
can be built from the existing data either has zero reliability (the registered one) or smuggles
the composition step back in (every post-hoc one), so the question has never actually been asked.
Expect a null — the free 2D competitor is larger than every composition-corrected xTB estimator
here, nothing in the registered family clears its own permutation bar for the amplitude, and no
descriptor in the section, xTB or 2D, survives the diglycolamide control — and pre-register that
expectation.

**What to deploy is unchanged.**  `gen15_curve/scripts/g15_predict.py` still stands; gen16 adds no
zero-shot skill.  Add the direction-call filter of §2 to it as a **literature-wide screening
filter** with its scope limitation printed beside it, and project the residual covariance onto the
PSD cone before any future shrinkage work.

**Four things to pre-register for gen17**, in descending order of expected value.

1. **Is the composition step an observable or a recipe?**  `steppos_n_H2O` and its relatives reach
   ρ = +0.78 to +0.83 against the amplitude (n = 27–29) with no energy in them, and `nligs_range`
   reaches −0.50 on S8 and survives removing the diglycolamides (−0.43).  But they correlate
   **+0.79 to +0.86 with the naive xTB slope** — they *are* the composition step, the quantity
   gen15 identified as the artefact behind +0.644 and one of the three killed results
   `START_HERE.md` §0 records.  So the gen17 question is not "new physics or builder heuristic"
   but the sharper one: **does the radius at which a ligand's inner sphere reorganises carry real
   size-match information, or is it a property of Architector's decision rules?**  It can only be
   answered against a builder-independent geometry source, and it must be pre-registered with a
   five-design MAE arm, a matched shuffled null, and the diglycolamide control that killed
   `CYCLE_ADD`.
2. **`SIZE` acquisition.**  "Measure the biggest family first" beat random in all five designs
   from zero features.  It needs its own pre-registration, its own matched null and its own
   cell-matched control, and it is a cheap, honest corpus-growth result if it holds.
3. **`NOISE_VAR` and the PSD projection.**  Both are one-line changes to the deployed measured
   mode that will move every interval and every MAE in it; they must be pre-registered, not
   patched in.
4. **A reliability gate as a standing protocol rule.**  L1's null cost a full lead and was
   uninformative because nobody measured whether the descriptor reproduces itself.  **Any future
   descriptor must report its split-half or jackknife reliability before its correlation with a
   target is interpreted**, and an estimator with reliability below ~0.3 must not be used to close
   a lead.  That rule is this generation's most transferable output.

**And the sentence that has not changed.**  Zero-shot, this model answers *which way* and cannot
answer *which ligand*; one well-chosen measurement is still worth more than every modelling
difference in five generations; and the binding constraint is still eleven and a half effective
chemotypes.
