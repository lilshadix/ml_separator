# Gen13 stage 2 — where the separation-factor error actually is, and what moves it

Written 2026-09-08, starting from the locked Gen13 ladder (`headline_tables/`) and revised as the
`B_v2`, `BP`, `BR`, `BQ` and `BP_v2` runs landed; it is final as of all five.  Seven independent
diagnostics were run on the frozen cohort and the frozen held-out predictions, each re-derived from
scratch by a second agent; their tables live in `analysis/stage2/<slug>/`, their verification
scripts in `<slug>/verify/`, and their figures in `figures/stage2/`.  The headline numbers are also
flat in `analysis/stage2/RESULTS.csv`.  Every number below is extractant-macro over 5 split seeds
unless it says otherwise.

Two of this file's conclusions were corrected by the verification pass and one was inverted by a
later run; both changes are marked where they occur rather than quietly folded in.

Nothing in this file changes the pre-registered primary endpoint.  The new arms are exploratory
(`S2_`, `S3_`) and the few-shot numbers are a **measurement-assisted deployment mode**, reported
separately and never mixed into the zero-shot comparison.

---

## 1. The problem is one number per cell, not the shape of the curve

* A per-cell quadratic in the standardised Shannon CN8 radius explains **95.6 %** of the centred
  curve (median per-cell R² 0.961, 268 cells with ≥ 6 metals).  The residual after it has median
  |value| 0.056, a quarter of the 0.205 replicate SEM of a single `log D`.
* **82–87 % of the total pairwise squared error is amplitude error**; shape carries 13–18 %.
  Making the amplitude exact would take the best arm from 0.481 to **0.207**; making the shape
  exact takes it only to 0.417.
* The observed amplitude has sd 1.18 and the predicted amplitude sd 0.50–0.65: the models predict
  a range **1.8–2.4× too narrow**, and out-of-fold they explain only **R² 0.20–0.32** of it.

Two consequences, both measured rather than assumed:

* **A global gain calibration is worthless.**  The oracle best single multiplier is g = 0.95–1.20
  and buys 0.0000–0.0026 MAE.  The shrinkage is per-extractant, not corpus-wide, so a scalar
  cannot undo it.  (This closes the `V2_CAL_*` branch of the in-flight `B_v2` run in advance.)
* **Non-smooth lanthanide structure is not worth adding.**  There *is* a reproducible residual
  curve (split-half r = 0.72 by extractant) but its rms is 0.035 log units.  A single Gd *dip*
  (not the classic Gd step) and a sign-reversed Jørgensen `e1` explain most of it, and an
  in-sample-optimal 14-element offset buys **0.0011** MAE.  There is no Eu anomaly at all on the
  centred-curve axis (mean residual +0.003 ± 0.010, Holm p = 1.00).

## 2. The bottleneck is transfer between extractants, not noise between conditions

A ladder of oracle predictors of a held-out cell's curve.  The honest comparison is on **all 24
extractants where the extractant oracle exists at all** (450 cells, 9 840 pairs), with every rung
re-scored there:

| predictor of the held-out cell's curve | extractant-macro MAE |
|---|---|
| best real arm (`X_ENS_DIRECT+LOWRANK_K2`) | 0.410 |
| corpus mean centred curve (arm-side, chemotype hold-out) | 0.469 |
| mean curve of the cell's **extractant**, cell left out | **0.250** |
| same extractant and same publication | 0.234 |
| oracle rank-1 amplitude with the corpus shape (full cohort) | 0.212 |
| the cell's own curve as a quadratic, in-sample (full cohort) | 0.135 |

**Knowing the extractant is worth 0.159 extractant-macro MAE, 95 % CI [0.091, 0.234]**, and every
recorded condition on top of extractant identity is worth at most 0.02.  The verification pass
found and corrected an inflated version of this number: restricting to the 20 extractants where
the acid rung is also defined gives 0.207 versus 0.457, a gap of 0.251 [0.122, 0.439], but that
filter drops precisely the four extractants where the oracle does worst, so it is outcome-correlated
and 0.16 is the number to quote.  The effective sample here is 20–24 extractants, not 9 840 pairs;
per-extractant oracle MAE ranges from 0.080 to 1.040.

One framing to avoid: the arms are *not* level with the corpus mean.  Compared like-for-like
against the arm-side `B1_MEAN_CURVE` on the same pairs, the best arm gains 0.059 (0.410 versus
0.469).  Comparing it instead with a leave-one-cell-out corpus mean, which is a different training
regime worth 0.015–0.020 on its own, is what made the arms look worthless.

Why the transfer fails is now quantified.  The between-extractant share of the amplitude sum of
squares is 60 % once ill-conditioned cells are filtered by the radius span of their metal set, and
**86 % on cells with ≥ 10 metals, where the chemotype ICC is 0.72** — the amplitude is largely a
chemotype-level property, and the primary design holds chemotypes out.  With 45 chemotypes and a
Kish effective n of 11.7, the model is fitting one scalar from about a dozen effective units.  That
is the ceiling the current features run into.

Copying a *measured sibling cell of the same extractant* reproduces this from the other side:
0.286 extractant-macro MAE, rising to 0.397 when the sibling must come from a different
publication.  Scored on the same 24 extractants the best arm gives 0.405, so this oracle is worth
**0.119**, not the 0.195 a comparison against the 90-extractant 0.481 would suggest.  Two cells of
one extractant have median curve correlation 0.941.  The same-chemotype copy control is *not* a
ceiling: on the 56 extractants where it is defined it scores 0.441 while the best arm scores 0.431,
so the model already beats copying a real measured cell of a sibling extractant.

**A warning about the conditions-only ablation.**  `B_abl_cond_only` scores 0.4709 with
`C_DIRECT_ROW`, better than the full block set, which reads like "conditions carry the chemistry".
The 64 `cond__` columns identify a held-out cell's **publication** with 94.1 % 1-NN leave-one-out
accuracy (chance 6.9 %) and its **extractant** with 71.2 % (chance 14.1 %).  A condition vector is
close to a laboratory fingerprint.  The channel is open on about a quarter of the corpus: over the
25 design-B folds, **27.6 % of held-out cells have their own publication present in the training
set** (182 of 391 test-publication instances), because 18 of the 58 publications span more than one
chemotype and those publications carry 158 of the 521 cells.

**This has now been tested, and it does not merely deflate the conditions-only result — it inverts
the block ordering.**  The other gen13 session added a publication-masked design (`BP` in
`gen13sep/splits.py`: every training cell from a held-out publication is dropped) and re-ran the
block sets over 5 seeds:

| blocks | `C_DIRECT_ROW` | `M_SELECTED` | `B1_MEAN_CURVE` |
|---|---|---|---|
| COND + MASSACT only | 0.6246 | 0.6538 | 0.6385 |
| lean (COND + MASSACT + PHYSCHEM + DONORS + COORD) | **0.5693** | **0.5581** | 0.6385 |
| all seven blocks | 0.5834 | 0.5850 | 0.6385 |

A conditions-only model is now **0.014 worse than the corpus mean curve**: its apparent skill on
design B was the laboratory fingerprint.  The ligand chemistry blocks — coordination, donors,
physchem — are the only thing that transfers across publications, worth **+0.055 [0.008, 0.120],
p = 0.025** for the direct model and **+0.096 [0.042, 0.136], p = 0.001** for `M_SELECTED`, both
5/5 seeds and leave-one-chemotype-out stable, both passing P1.  ECFP and LIG2D still dilute the
signal: all seven blocks are 0.014–0.027 worse than lean.

**Two size-matched controls isolate the mechanism exactly.**  Conditions-only `C_DIRECT_ROW`:

| design | what is removed from training | MAE |
|---|---|---|
| B (primary) | nothing | 0.4709 |
| `BR` | the same 952 cells, chosen at random | 0.4876 |
| `BQ` | whole random publications, matched on cells *and* condition series (971 / 922) | 0.4711 |
| `BP` | the publications the held-out cells themselves come from | 0.6246 |

Dropping that much data at random costs 0.017.  Dropping whole *unrelated* publications, including
their within-publication condition series, costs **nothing at all**.  Dropping the publications the
test cells come from costs 0.154.  The damage is therefore not data loss and not the loss of
condition series: it is specifically the **co-occurrence of a laboratory with the test chemistry**.
The same decomposition on all seven blocks is 0.4953 / 0.5028 / 0.4947 / 0.5834.

One side observation from `BR` that lines up with §1: under equal data loss the curve model degrades
more than the row model (`M_SELECTED` 0.517 versus `C_DIRECT_ROW` 0.488 on conditions only, against
0.493 / 0.471 under design B).  The curve parametrisation is the more data-hungry of the two, which
is what the amplitude ICC predicts — it is a chemotype-level scalar fitted from about a dozen
effective units, so it is the first thing to suffer when units are removed.

This is the strongest available evidence for §5's recommendation, and it changes its status: the
compact binding-site representation is not the best remaining lever by elimination, it is the only
block set that measurably transfers.

Conditions do move the curve — 40 % of the (span-filtered) amplitude variance is within one
extractant, 7× the per-cell noise — and every arm is blind to it, reproducing 4–21 % of that spread
with correlation between −0.06 and +0.08.  But the coupling is extractant-specific, not a
transferable law: the acid slopes are heterogeneous (Q = 105.8 on 11 df, p = 1.3e−17), and a global
condition→shape head fitted leave-one-chemotype-out has out-of-chemotype R² = −0.74 and **degrades**
extractant-macro MAE by 0.061–0.079.  A perfect condition oracle is worth 0.030 on the 12 measurable
extractants against 0.245 for an extractant-level curve oracle.

Two lines that would have raised the unit count are closed:

* **External aqueous log K series carry no transferable amplitude.**  Over the 82 extractants with
  both an amplitude and a log K prior, Spearman(observed amplitude, predicted log K slope) =
  **+0.00**; at chemotype level −0.08.  The median nearest external ligand is at Tanimoto 0.36.
  This explains the null result of the locked `B_logk` run.
* **Uncertainty and abstention do not work.**  Predicted |error| correlates with actual |error| at
  Spearman 0.41, but within a fixed dZ that falls to −0.02: the signal is entirely "wide pairs have
  big errors".  Abstaining on the worst predicted decile moves normalised error from 0.7564 to
  0.7505 against 0.7566 for random abstention.

## 3. What one measurement is worth (the largest lever found)

A cell's held-out pair residuals are **exactly additive** in one per-cell metal curve: a cell with
`m` metals has `m(m−1)/2` pairs but only `m−1` independent residual numbers, and that residual
curve is smooth (lag-1 correlation +0.92).  So one measured separation factor constrains all the
others, and the best linear correction is the conditional mean of the residual curve given that
measurement — the BLUP with the residual covariance between metals:

    u = Σ d (d' Σ d + σ²)⁻¹ r,   d = e_A − e_B,  r = y_AB − (c_A − c_B)

`Σ` is estimated **leave-chemotype-out** from the held-out residuals of the other chemotypes of the
same split seed (`gen13sep/fewshot_stage2.py`).  Results over 89 extractants, 383 cells, 5 seeds:

| support pair | best arm zero-shot | + conditional update | line through the pair, no model |
|---|---|---|---|
| chosen: widest dZ (La–Lu) | 0.455 | **0.251** | 0.272 |
| drawn at random | 0.464 | **0.362** | 0.483 |

The locked `basis_shift` adapter is only defined for arms that save a basis.  On the same pairs and
the same support draws, `M_PHYSICS_radius+radius_sq` scores 0.284 with `basis_shift` against 0.258
with the conditional update on the widest pair, and 0.396 against 0.375 on a random pair; the
`rescale` adapter is 0.38 and 0.44.

Every contrast against the locked adapters passes the pre-registered P1 rule (margin 0.02, both
intervals, p < 0.05, 5/5 seeds, leave-one-chemotype-out sign-stable).  The measurement budget sweep
with a greedy pre-measurement design (`fewshot_stage2_budget.csv`, 321 cells with ≥ 4 metals):

| pairs measured | best model + update | corpus mean curve + update |
|---|---|---|
| 0 | 0.409 | 0.523 |
| 1 | **0.211** | 0.228 |
| 2 | 0.193 | 0.207 |
| 3 | 0.167 | 0.176 |

**Under the publication-masked design the measurement matters more, not less.**  Repeating the
study on the `BP_lean` curves, where the zero-shot problem is genuinely harder: `C_DIRECT_ROW`
0.5410 → **0.2664** with the widest pair measured, `M_SELECTED` 0.5309 → 0.2742, corpus mean curve
0.6088 → 0.2777, on the same 89 extractants and 383 cells.  One measurement halves the error under
the honest design too, and the spread between the best model and no chemistry at all after that
measurement is **0.011**, narrower than the 0.020 seen on design B.

**A better zero-shot model barely survives the measurement.**  Running the same study on the
`B_v2` curves, the best zero-shot arm the programme has goes 0.4345 → **0.2485** with the widest
pair measured, against 0.4432 → 0.2524 for `V2_BAG4@lean` and 0.5744 → 0.2681 for the corpus mean
curve, on the same 89 extractants and 383 cells.  So `B_v2`'s 0.023 of zero-shot improvement over
the previous best arm shrinks to **0.003** once one pair is measured, and the whole spread between
the best chemistry model and no chemistry at all is 0.020 after that measurement.

**Which pair to measure.**  The greedy design uses only the covariance and which metals the cell
has, never a measured value, so it can be followed before any experiment.  On a full fourteen-metal
series it picks **La–Lu** first in 100 % of cells and **Ce–Tm or Ce–Ho** second (45.6 % / 42.3 %).
Over all 1 340 cell-seed combinations with at least six metals it picks La–Lu 49.3 % of the time,
then La–Eu 13.1 %, Nd–Er 9.5 % and Gd–Lu 7.9 % — always the widest span the cell's metal set allows.

The gain is not an artefact of small cells — it is largest where most is at stake
(best arm, widest support, `X_ENS_DIRECT+LOWRANK_K2`):

| metals in the cell | cells | zero-shot | + conditional | line through the pair |
|---|---|---|---|---|
| 3 | 62 | 0.516 | 0.406 | 0.423 |
| 4–5 | 53 | 0.364 | 0.231 | 0.200 |
| 6–8 | 137 | 0.307 | 0.182 | 0.186 |
| 9–11 | 17 | 0.523 | 0.267 | 0.239 |
| 12–14 | 114 | 0.484 | 0.215 | 0.249 |

The model plus covariance beats the model-free line only on the full series (12–14 metals); on
4–11 metal cells the line is level with it or slightly ahead.

The adapter has two constants, the covariance shrinkage and the assumed measurement variance.  A
5 × 5 sweep of them (`fewshot_stage2_sensitivity.csv`) moves the widest-pair score only between
0.248 and 0.306, with the reported defaults (0.25, 0.09) at 0.251 and the grid optimum at 0.248, so
the result does not depend on tuning them.

Read honestly, this says two things at once.  One well-chosen measurement halves the error and
reaches the level of the "knows the extractant" rung of §2 (0.211 here against 0.207 there; the two
are on different subsets — 321 cells with ≥ 4 metals versus 299 cells of 20 multi-cell extractants
— so they are comparable in size, not identical quantities).  It also **collapses the value of the chemistry
model**: the gap between the best model and no chemistry at all falls from 0.114 at zero
measurements to 0.017 at one, and with the widest pair measured a straight line through it
(0.271) is within 0.02 of the full model plus update (0.251).

## 4. New zero-shot arms

Implemented in `gen13sep/arms_stage2.py`, run by `scripts/g13_run_stage2.py` on the same frozen
fold plan, so every contrast is paired pair-for-pair.

**The 5-seed run `S2_main` has landed and the honest answer is that none of these passes.**
Leaderboard in `metrics/S2_main_leaderboard.csv`, contrasts in `bootstrap/S2_main_contrasts.csv`:

| arm | macro MAE | chemotype-macro | sign acc, strong | pair Spearman |
|---|---|---|---|---|
| `S2_CENTRED_ROW_basic` | 0.4891 | 0.5211 | 0.765 | 0.516 |
| `S2_CENTRED_ROW` | 0.4929 | 0.5226 | 0.761 | 0.509 |
| `S2_STACK_CENTRED+PHYSICS` | 0.4933 | 0.5261 | 0.764 | 0.506 |
| `C_DIRECT_ROW` (reference) | 0.4953 | 0.5365 | 0.722 | 0.463 |
| `S3_EXT_LEVEL` | 0.4962 | 0.5402 | 0.748 | 0.449 |
| `M_PHYSICS_radius+radius_sq` (reference) | 0.4979 | 0.5425 | 0.744 | 0.451 |
| `S3_HIER` | 0.5014 | 0.5620 | 0.740 | 0.461 |

* **Centring the row target is directionally right and statistically invisible.**
  `S2_CENTRED_ROW_basic` beats `C_DIRECT_ROW` by **0.0062** on the primary metric, 95 % CI
  [−0.031, +0.050], p = 0.71, 4/5 seeds, not leave-one-chemotype-out stable.  The one-seed smoke
  that suggested 0.011 was noise.  The secondary metrics are more encouraging and 5/5 seeds
  positive — sign accuracy +0.043, pair Spearman +0.053, chemotype-macro +0.015 — but every
  interval includes zero.  Report it as a direction, not a result.
* The `basic` variant, with `C_DIRECT_ROW`'s two metal columns, beats the rich-descriptor variant
  by 0.0037 on 5/5 seeds, which is consistent with §1: extra non-smooth metal descriptors have
  nothing to fit.
* **The hierarchical split fails, as §2 predicts.**  `S3_HIER` (0.5014) is the worst arm in the
  run and is 0.005 behind its own condition-free ablation `S3_EXT_LEVEL` (0.4962).  Adding a
  condition head hurts, which is the same conclusion the leave-one-chemotype-out condition model
  reached independently.
* **Aggregating to the extractant level buys nothing.**  `S3_EXT_LEVEL` is −0.0008 against
  `C_DIRECT_ROW` (p = 0.94).
* **Neither reweighting scheme moves anything.**  Extractant-balanced weights −0.0007 (p = 0.91),
  reliability weights +0.0011 (p = 0.76).
* Also tested and rejected: **median-over-trees readout** (0.514 vs 0.501 on one seed — worse).

So the zero-shot side of stage 2 is a negative result.  Its value is that it removes four plausible
directions cheaply and confirms two of §1's predictions out of sample.

**`BP_v2` has now scored all of these under the publication-masked design**, 11 arms on
byte-identical pairs (`metrics/BP_v2`, `bootstrap/BP_v2`):

| arm | BP MAE |
|---|---|
| `V2_BAG4@lean` | **0.536** |
| `V2_BAG_MIX5` | 0.545 |
| `S4_CENTRED_ROW@lean` | 0.557 |
| `V2_BAG_MIX3` | 0.557 |
| `S4_BAG_MIX3C` | 0.571 |
| `S2_CENTRED_ROW_basic` | 0.575 |
| `C_DIRECT_ROW` | 0.583 |
| `S3_EXT_LEVEL` | 0.585 |
| `V2_DIRECT@cond` | 0.625 |
| `B1_MEAN_CURVE` | 0.639 |

The centred-row swap does not pay under `BP` either: `S4_BAG_MIX3C` is −0.013 against
`V2_BAG_MIX3` on 0/5 seeds, and `S2_CENTRED_ROW_basic` is +0.008 against the row reference at
p = 0.66.  Two designs, one conclusion — the branch is closed, and the numbers here are final
rather than provisional.  Note that `S4_CENTRED_ROW@lean` at 0.557 beats `C_DIRECT_ROW` at 0.583,
but that gap is the lean block subset doing the work, not the centring.

The same run shows §2's inversion inside a single bag: `V2_BAG_MIX3` beats its own conditions-only
member by **+0.067 [0.029, 0.123], p < 0.001, 5/5 seeds**.  Because the mixed bags win on design B
only through a member that does not transfer across laboratories, the deployed predictor has been
switched from `V2_BAG_MIX3` to `V2_BAG4@lean`.

**The parallel `B_v2` ladder (other session) landed while this was written.**  Bagging over mixed
block subsets is the best zero-shot result the programme has: `V2_BAG_MIX3` **0.4577** and
`V2_BAG_MIX5` 0.4585 against 0.4810 for the previous best ensemble and 0.4953 for `C_DIRECT_ROW`.
Its gain-calibrated variant behaves exactly as §1 predicted — `V2_CAL_BAG4@lean` 0.4746 is *worse*
than the uncalibrated `V2_BAG4@lean` 0.4676 — which is an independent confirmation that the
dispersion branch is closed.  `V2_DIRECT@cond` at 0.4709 is the arm that the publication-fingerprint
caveat above applies to.

## 5. What to do next, in order

1. The centred-row branch is closed.  It is +0.006 (p = 0.71) on design B and −0.013 on 0/5 seeds
   inside the bag under `BP`.  Record it as tested and null, not as a candidate.
2. Report the few-shot mode as a headline deployment result with its own table, clearly separated
   from the zero-shot endpoint, and state the "one measurement beats the model" comparison plainly.
3. Stop spending on: global gain calibration, tetrad or Gd-break basis terms, extra low-rank
   components, uncertainty and abstention layers, and the external log K block.  Each is measured
   at ≤ 0.003 or is a measured null.
4. The only remaining zero-shot lever with real headroom is a **better compact representation of
   the binding site**, and under the publication-masked design this is now a direct measurement
   rather than an inference: the lean block set beats conditions-only by 0.055–0.096 with intervals
   excluding zero and P1 satisfied, while adding ECFP and LIG2D on top costs 0.014–0.027.  More
   fingerprint is not the way; a better description of the donor set and the coordination geometry
   is.
5. Re-state every design-B number in the decision report against its `BP` counterpart, or drop it.
   The primary endpoint survives (P1 is null under both), but the block ablations reverse, and any
   claim that rested on the conditions-only arm has to go.
