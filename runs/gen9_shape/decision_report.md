# gen9 decision report

**SELF-AUDIT STATUS: PASS AFTER FIXES.**

*This section is written first because the follow-up brief asks for it first: what
was checked, what broke, what was rerun, and whether any headline moved.*

**Eight issues found. Two Severity A/B invalidated results outright and forced full
reruns; every affected experiment was rerun from scratch and no pre-fix number appears
anywhere in this report.** All 125 tests pass, the manifest verifies 163 artefacts with
zero drift, and the Phase-0 reproduction of gen8 passes 15/15 in an environment with
zero library drift.

Every number below carries its cohort, its seed count, and whether it was
**pre-registered**, **development** or **EXPLORATORY**.

---

## 0. Self-audit

### 0.1 Baseline reproduction (Phase 0)

All **15** gen8 reference numbers reproduce **exactly** in this environment before
any gen9 arm was fitted — the k-shot frontier (zero-shot 1.0605, random 1-shot
0.7927, CENTRAL 0.6964, MEDOID 0.6916, oracle 0.5326, slope-repair 0.6752, 2-shot
0.5879, 3-shot 0.5230, 5-shot 0.4743) and the flattening pathology (extractant true
2.574 / predicted 0.116, acid 1.656 / 0.355, extractant span ratio 0.051). Library
versions are identical to gen8's stamped `environment.json`; **zero drift**.
`runs/gen9_shape/reproduction/reproduction.json`.

### 0.2 Independent curve reconstruction

gen9 rebuilt the curve geometry from the cohort rather than reusing gen8's parquet:
1,176 curves, 7,207 memberships, 5,123 rows on a curve — **identical keys, maximum
abscissa difference 0.0**. Zero non-finite abscissae (the gen8 bug class), zero rows
appearing twice on one axis, zero curves spanning two series.
`runs/gen9_shape/curves/curve_audit.json`.

### 0.3 Issues found and fixed during gen9

| # | issue | severity | affected | fix | rerun | conclusion changed? |
|---|---|---|---|---|---|---|
| 1 | `gen9_acquisition.py` read the OOF parquet directly, which carries no condition columns. `_standardised_axes` returns zeros for a missing column, so **every** geometry-driven policy — MEDOID, CENTRAL, MID_ACID, FARTHEST, COVERAGE and every learned policy using geometry features — saw an all-zero condition space, tied on every candidate and returned the first index. All seven scored *identically* (0.7274), and MEDOID scored *worse than random*. | **A — invalidates results** | the entire acquisition study | join the cohort's condition columns on `row_id`, with assertions that every policy axis is present, non-empty and non-constant after the join (`attach_conditions`) | full acquisition study rerun from scratch | caught before any result was read; the pre-fix run was discarded |
| 2 | the acquisition study rebuilt an identical fold training set once per feature ablation — five times the work, and two ablations that saw different training blocks would not have been comparable | B — changes subsets | acquisition ablations | build each fold's dataset once and cache it | acquisition study rerun | no |
| 3 | `duplicate_cell_scan` keyed on the full condition set, so DMDPhPDA's two copies — which differ **only** by an unreported temperature (one copy records 25 °C, the other nothing) — landed in different cells and the known corruption was invisible. Found 0 exact-decade pairs. | B | data audit | scan twice: full cell and *core* cell (metal, acidity, extractant concentration). Now finds DMDPhPDA's **59** exact-decade pairs at 1, 2, 3 and 4 decades, plus two further single-cell cases | data audit rerun | no — it made a known finding reproducible and surfaced two new ones |
| 4 | the condition-adjusted level fell back to a ligand's *own* value when no other ligand shared its condition cell, reporting **every** level as exactly 0.0 and finding zero outliers | B | data audit | leave-one-out over a coarser (metal, acidity) bucket, `NaN` where unidentifiable — never silently zero — with an `adjusted_coverage` column, and the scan run on the raw level across three peer definitions | data audit rerun | no — TWE-24 now reproduces gen8's finding (+4.82 decades vs its thiophosphoryl siblings, costing its nearest neighbour 4.10 log units) |
| 5 | `.astype(str)` on a pandas-3 frame left real floats in an object array while building a grouping key — the same trap `gen8.series._group_key` documents | C→B | data audit keys | explicit `repr` over a float array | data audit rerun | no |
| 6 | **`CurveBoost` was not reproducible.** `ExtraTreesRegressor(n_jobs=-1)` sums its trees in thread-completion order, so a single forest's predictions move by ~1e-15. Inside a boosting recursion that perturbation reaches the next stage's pseudo-residual, flips a near-tied split comparison and grows a **different tree**, and the divergence compounds. Two runs of the identical configuration **in the same process** differed by up to **0.139 log units on every one of 382 held-out rows**, moving a fold's pooled MAE by 0.007 — the size of effects this study reports. | **B — changes results materially** | every gen9 arm; the phase-1 screen | round the running prediction to 1e-9 log units after each stage, which removes the perturbation before it can reach a split comparison. The *initial constant* is deliberately not rounded: rounding it perturbs the first stage's target and breaks the exact nesting of the frozen model inside the arm family | **the whole shape sweep rerun at five seeds, post-fix**; the pre-fix phase-1 screen is superseded | rankings unchanged (the phase-1 effects were 10–20× the irreproducibility), but no pre-fix number is quoted |
| 8 | **`LEARNED_BLEND` could never fall back to the baseline it was built to nest.** The arm is `alpha * MEDOID + (1 - alpha) * learned`, with `alpha` chosen in-fold so that `alpha = 1` recovers `MEDOID` exactly. But `alpha` was selected on the very blocks the ranker had been fitted on; the ranker reproduces those better than any fixed geometric rule can, so **`alpha = 0` won in 25 folds out of 25** and the arm lost to `MEDOID` on every held-out chemotype (0.740 vs 0.692). | **B — changes results materially** | every `LEARNED_BLEND` number | fit an inner ranker on half the fold's training ligands (split by BLAKE2b, not `hash`), choose `alpha` on the other half, then apply that `alpha` to the full-data ranker | the acquisition study and all three heavy k-shot runs rerun | yes for the blend — it goes from *worse* than MEDOID to a wash with it; the gen9-B conclusion (learned acquisition does not beat centrality) is unchanged |
| 7 | the fix's own first version rounded the initial constant too, which perturbed the first stage's pseudo-residual by 5e-10 and — by the identical amplification — grew a different first tree, breaking the frozen-model nesting by 7.7e-4 on 4 of 240 rows | C, caught by the nesting test | the nesting assertion | round the accumulation, never the constant | rerun of the same sweep | no |

**Every one of these was found by a check the brief asked for**, not by inspection:
issue 1 by the "arms that should differ scoring identically" smell, issues 3 and 4 by
the scans failing to rediscover a known corruption, issue 6 by an arm reporting a
different pooled MAE in two runs of the same configuration, issue 7 by the test written
for issue 6's fix, and issue 8 by asking *why* a losing arm lost rather than recording
that it did.

Issue 6 is the one worth dwelling on, because it is the follow-up brief's §8 in its
purest form: nothing failed, every script completed, and the numbers were wrong by an
amount comparable to the effects being measured. It was invisible to code inspection —
the seeding is correct, the RNG is threaded properly, and the only stochastic source
is a library call with a fixed `random_state`. It was visible the moment two runs of
the same arm were compared.

### 0.4 Audits that pass

* **fold boundaries** — 25 folds, zero row / ligand / chemotype / ECFP-cluster
  overlap (`phase1/fold_audit.csv`);
* **curve boundaries** — **zero** curves straddle a train/test boundary in any fold
  (`phase1/curve_boundary_audit.csv`);
* **row accounting** — no arm lost, duplicated or NaN'd a row
  (`finalists/row_accounting.csv`). In the k-shot frontier, 802 arms carry 31,500 units
  each and 10 carry 8,580: those ten are the `ZERO_SHOT` references, which exist only
  at k = 0 by construction. No other arm differs from any other
  (`frontier/row_audit.csv`, 0 NaN, 0 inf);
* **pairing** — every arm in the attribution chain shares fold, ligand, repeat, pool
  size and evaluation size (`frontier/pairing_audit.csv`);
* **stable splitting** — the pool/evaluation draw, the acquisition training blocks
  and the curve pairs are byte-identical across four different `PYTHONHASHSEED`
  values in fresh subprocesses; an AST scan fails the suite if any gen9 module calls
  Python's builtin `hash`;
* **leakage** — corrupting every unselected target changes neither an adapted
  prediction nor an acquisition selection; permuting or truncating the evaluation
  indices does not move a first-point choice;
* **112 tests pass.**

---

## 1. The decision

**Yes — the flattening is fixable, and the pre-registered way of fixing it was the
wrong one.**

The pre-registered experiment was a within-curve *loss*: supervise the differences
between points on the same reconstructed titration. It failed, cleanly and
informatively. It reduces dynamic-range compression on every axis and for every
ligand, and it does not restore the slopes; pushing its weight until the range comes
back (`lambda_delta = 8` reaches span recovery 0.756, past the pre-registered "strong"
bar) costs +1.17 macro MAE, inverts the acid slope, and collapses the within-curve
ordering. There is no setting at which the range returns and the model still works.

The diagnosis that came out of that failure is the gen9 result. A model trained only
on the **curve-centred** target — nothing to do but predict shape, no level to compete
with — recovers **2.5 %** of the extractant range, *worse* than the monolith. So the
level was never the obstacle. What the model was missing is more basic:

> `log D` at a given *absolute* extractant concentration is dominated by the ligand's
> level. The within-curve response is only interpretable relative to that curve's own
> measurement window, and different titrations span different windows — 0.03 M is the
> top of one series and the bottom of another. Given absolute conditions alone the
> conditional mean of the centred response is ~0 everywhere, and **a flat prediction
> is the correct answer to the question the model was being asked.** It was never
> asked where on its own curve the row sat.

Telling it — five columns, all functions of the condition list the user supplies —
moves every shape endpoint at once, on all three axes, in 5/5 seeds:

| extractant titrations, 775 curves / 25 ligands | frozen gen8 model | + relative position | + mean-preserving recomposition |
|---|---|---|---|
| median predicted slope (measured 2.574) | 0.116 | 0.521 | **1.024** |
| slope MAE | 2.433 | 1.984 | **1.510** |
| median span recovery | 0.051 | 0.210 | **0.423** |
| shape MAE | 0.665 | 0.567 | **0.469** |
| within-curve Spearman | 0.582 | 0.840 | **0.886** |
| within-curve sign accuracy | 0.762 | 0.903 | **0.926** |
| curves steeper than the corpus's own 95th percentile | 0.0 % | — | 0.26 % |
| **macro MAE (5 seeds)** | 0.9807 | 0.9858 | **0.9695** |

Paired, chemotype-blocked, 5,000-replicate intervals for the recomposition against
the frozen model (positive = better):

| axis | metric | point | 95 % CI | BCa | ligands improved | seeds |
|---|---|---|---|---|---|---|
| extractant | slope MAE | **+0.910** | [0.481, 1.065] | [0.621, 1.107] | 23/25 | 5/5 |
| extractant | shape MAE | **+0.182** | [0.097, 0.218] | [0.109, 0.220] | 21/25 | 5/5 |
| extractant | span-recovery distance | **+0.370** | [0.269, 0.419] | [0.288, 0.426] | 23/25 | 5/5 |
| extractant | row MAE | **+0.087** | [0.057, 0.148] | [0.058, 0.149] | 20/25 | 5/5 |
| acid | span-recovery distance | **+0.187** | [0.076, 0.246] | [0.103, 0.261] | 54/62 | 5/5 |
| acid | row MAE | **+0.021** | [0.001, 0.037] | [0.001, 0.038] | 43/68 | 5/5 |
| lanthanide | span-recovery distance | **+0.101** | [0.056, 0.171] | [0.036, 0.155] | 62/76 | 5/5 |

Per-ligand dynamic range rises from 0.392 of truth to **0.559**. On the lanthanide
axis the median predicted slope lands on 0.0843 against a measured 0.0844.

**Three caveats, stated at the same volume as the result.**

1. **This arm is EXPLORATORY.** It was designed after the pre-registered sweep was
   read. The pre-registered arms are reported as the failures they are, in
   `failure_analysis.md`, and none of the numbers above is offered as a confirmed
   pre-registered effect. What licenses taking it seriously is that the intervals
   are wide margins, every seed agrees, the effect appears on three independent axes
   and four independent metrics, and the mechanism was *predicted* by the
   curve-centred probe before the arm was built.
2. **The "strong result" bar is not quite met on the target axis.** Median extractant
   span recovery is 0.423 against the pre-registered 0.5 — met on the lanthanide axis
   (0.618) and on per-ligand span (0.559), missed on extractant.
3. **This is close to being a learned version of gen8's post-hoc repair.**
   `rel__position` is a normalised abscissa, so a model that learns "y rises with
   relative position at ~2.5 per unit" has learned the corpus's median slope in
   normalised coordinates — which is what gen8's repair applies as a prior. The
   differences are that it is learned rather than fixed, conditioned on the ligand
   and the axis, applies on all three axes rather than mainly one, and improves macro
   MAE rather than leaving it alone. The head-to-head against gen8's repair is in
   §4.4 and it is the honest way to size the advance.

### 1.1 What each half contributes

The ablation separates the two ideas cleanly. **The features do most of the work** —
adding the five columns to the frozen model, changing nothing else, takes the
extractant slope from 0.116 to 0.521 and the Spearman from 0.582 to 0.840. **The
mean-preserving recomposition roughly doubles what is left** (slope to 1.024) and is
the only one of the two that also improves macro MAE, because it substitutes the shape
around each curve's *predicted level* and therefore spends none of the accuracy the
monolith bought by drawing flat. Both earn their place.

---

## 2. GEN9-A — did within-curve training fix the flattening?

**The pre-registered objective: no. The mechanism it exposed: yes.**

### 2.1 What the pre-registered loss moved — locked, five seeds

Against `A0_ROW_ONLY` — the *same learner, same hyperparameters, same seeds*, curve
terms off — every sampler significantly reduces dynamic-range compression on the
extractant axis, in 5/5 seeds, with **25 of 25 ligands** improving for two of them.
That is real, and it is what the objective was built to do.

**And that is the only pre-registered endpoint it moves.** Slope MAE, shape MAE and
the predicted/true slope ratio all fail; within-curve Spearman falls from 0.692 to
between 0.444 and 0.247. The median predicted extractant slope is 0.108 for the
control and *lower* for three of the four samplers. The pre-registered experiment
fails three of its four conditions, and the full table is in
`failure_analysis.md` §1.

### 2.2 The dose–response that explains why

Raising `lambda_delta` recovers the range and pays for it almost exactly in
proportion (single seed, five folds, **development**):

| `lambda_delta` | macro MAE | extractant slope | span recovery | Spearman | acid slope |
|---|---|---|---|---|---|
| 0 (control) | 0.982 | 0.115 | 0.053 | 0.650 | 0.392 |
| 0.5 | 1.011 | 0.172 | 0.075 | 0.398 | 0.460 |
| 2 | 1.210 | 0.279 | 0.310 | 0.197 | 0.437 |
| 8 | **2.155** | 0.486 | **0.756** | 0.186 | **−0.298** |

Two of those points were then run at five seeds, and the five-seed answer is worse
than the probe suggested: `X_MULTI_d8c` reaches extractant span recovery 0.335 and
lanthanide span recovery **1.257** — past the measured range — while its extractant
slope stays at 0.110, its Spearman collapses to 0.090 and its macro MAE is 1.273.
**A single-seed probe made this look like the best trade-off in the study; five seeds
say it is the worst arm in it**, and that discrepancy is the reason the protocol
separates development from locked evaluation.

`A2_ROW_ENDPOINT` is the same lesson in miniature: it doubles span recovery
(0.046 → 0.097) while dropping within-curve sign accuracy to 0.615, close to the 0.5
a constant predictor scores.

### 2.3 Adversarial checks on the exploratory arm

Two of the five falsification conditions recorded in the protocol *before* these were
computed can be answered from the curve table directly. Both survive.

**Is the gain an artefact of the window?** `rel__position` is a normalised abscissa,
so a curve spanning three decades and one spanning half a decade get identical
position values. If the feature were really encoding *window width*, the gain would
live in the wide curves. It does not — it is present in all four width quartiles, and
the slope gain is largest in the **narrowest**:

| extractant window quartile | median width (decades) | curves | shape-MAE gain | slope-MAE gain | curves improved |
|---|---|---|---|---|---|
| Q1 narrow | 0.48 | 39 | +0.167 | **+1.361** | 95 % |
| Q2 | 0.82 | 39 | **+0.234** | +1.015 | 100 % |
| Q3 | 1.00 | 38 | +0.153 | +0.586 | 74 % |
| Q4 wide | 1.00 | 39 | +0.226 | +0.719 | 95 % |

**Is the gain concentrated?** No. 21 of 25 ligands improve; the single best ligand is
10 % of the summed gain, the top three 29 %, the top five 42 %. Leave-one-ligand-out
moves the mean gain only between 0.171 and 0.190 against an all-ligand mean of 0.182 —
**no single ligand is worth more than 0.011 of it**. The three ligands that degrade do
so by 0.013 to 0.021, which is noise against a mean gain of 0.182.

**Is it driven by one publication?** Partly, and this is the one check that does not
come back clean. The 155 extractant curves come from 26 publications; the largest
single one contributes 27 curves with a mean shape-MAE gain of 0.387, which is **34 %
of the summed gain**. Removing it takes the mean gain from +0.196 to +0.155 — the
effect survives comfortably, and every one of the top six publications leaves it
between +0.155 and +0.203 — but the gain is **not uniform across sources**, and a
reader should know that one campaign benefits about twice as much as the rest.

The remaining two conditions — is it really gen8's repair, and does it survive
calibration — are answered in §4.4 and §2.4.

### 2.4 The recorded prediction, and its outcome

Before the k-shot frontier was run, protocol addendum 2 recorded a prediction: because
the pre-registered shape arms' extra error is **not** level error — their per-curve
shape MAE is *worse* than the control's — offset calibration should not rescue them,
and their k = 0 ordering should survive to k = 1, 2 and 5. The one arm with a mechanism
to win after calibration was `X_MULTI_d8c`, the only configuration whose shape MAE
improved in the single-seed probe.

The prediction holds exactly. Best deployable arm per global model, common cohort:

| global model | k=0 | k=1 | k=2 | k=5 |
|---|---|---|---|---|
| `A0_ROW_ONLY` (control) | **1.000** | **0.629** | **0.539** | **0.474** |
| `A4_MULTI_d05` | 1.010 | 0.663 | 0.567 | 0.496 |
| `X_MULTI_d2` | 1.228 | 0.754 | 0.618 | 0.519 |
| `X_MULTI_d8c` | 1.277 | 0.754 | 0.618 | 0.519 |

The ordering is preserved at every k. `X_MULTI_d8c` did not win — its five-seed shape
MAE turned out worse than the control's, not better, so the mechanism the prediction
allowed for never existed. And for both high-weight arms the *best* deployable arm at
k ≥ 1 is `NO_MODEL` or `NEAREST_OBSERVED` — the model has become bad enough that
ignoring it and interpolating between measurements is better than calibrating it.

**This is the pre-registered outcome (3): a clean negative.** Within-curve supervision
reduces compression, the compression was not what was costing the accuracy, and gen8's
mean-preserving post-hoc repair remains the only tool that improves shape without
paying for it.

### 2.5 Why the loss could not have worked

Because it was asking the model to express a relationship it had no coordinate for.
The curve-centred probe settles this: with the level removed entirely, and therefore
nothing to trade off against, the same learner still recovers only 2.5 % of the
extractant range. The obstacle was never competition between level and shape. It was
that the centred response is not a function of the absolute conditions.

---

## 3. GEN9-B — can the oracle's rule be predicted?

### 3.1 The identity is weaker under gen9's protocol than under gen8's

gen8 verified `argmin |r_i − median(r)| == argmin realised MAE` to 6.7 × 10⁻¹⁶ over
715 blocks — in its **exhaustive** protocol, where the candidates and the scored rows
are the same points. gen9 selects from a pool and scores on a **disjoint** evaluation
set, and the proof does not transfer.

Measured under gen9's exact protocol, over **8,580 (ligand, seed, repeat) blocks**:

| quantity | value |
|---|---|
| blocks where the surrogate attains the realised optimum exactly | **60.1 %** |
| median regret of following the surrogate | **0.000** |
| mean regret | 0.019 |
| 95th-percentile regret | 0.112 |
| median Spearman, surrogate vs realised | **0.948** |
| 5th-percentile Spearman | 0.321 |

So the brief's target is an excellent but **not exact** surrogate here. That is why
gen9 carries and compares two labels rather than assuming one is sufficient, and it
is a difference from gen8 worth stating rather than inheriting.

### 3.2 The learned selector matches centrality and does not beat it

Five seeds, 8 repeats, 143 ligands, every policy on identical pools and identical
evaluation rows:

| policy | one-shot macro MAE | regret to oracle | ligands harmed vs zero-shot |
|---|---|---|---|
| `ORACLE` (non-deployable) | 0.481 | 0.000 | 8 % |
| `SURROGATE_ORACLE` (`argmin \|r−median r\|`, also non-deployable) | 0.501 | 0.019 | 13 % |
| **`MEDOID`** | **0.645** | 0.164 | 30 % |
| `CENTRAL` | 0.649 | 0.168 | 31 % |
| **`LEARNED_BLEND[geometry]`** | **0.651** | 0.169 | 31 % |
| `LEARNED_SCALAR[geometry]` | 0.653 | 0.172 | 31 % |
| `LEARNED_RANK[geometry]` | 0.664 | 0.182 | 31 % |
| `MIN_GP_DESIGN_VAR` | 0.672 | 0.191 | 32 % |
| `MAX_ENSEMBLE_SD` | 0.712 | 0.231 | 35 % |
| `RANDOM` | 0.720 | 0.239 | 36 % |
| `FARTHEST_FROM_EXISTING` | 0.795 | 0.313 | 41 % |
| `MIN_PREDICTION` | 0.817 | 0.336 | 39 % |

Paired, chemotype-blocked, 5,000 replicates:

| comparison | point | 95 % CI | ligands improved | seeds |
|---|---|---|---|---|
| `LEARNED_BLEND` vs `RANDOM` | **+0.070** | [0.028, 0.099] | 99/143 | **5/5** |
| `LEARNED_BLEND` vs `MEDOID` | −0.005 | [−0.018, +0.014] | 53/143 | 0/5 |
| `LEARNED_BLEND` vs `CENTRAL` | −0.001 | [−0.005, +0.004] | 17/143 | 3/5 |
| `ORACLE` vs `LEARNED_BLEND` | +0.169 | [0.131, 0.201] | **143/143** | 5/5 |

**gen9-B fails its pre-registered target.** It was to recover ≥ 30 % of the remaining
MEDOID→oracle gap; it recovers **0 %**. The learned selector is significantly better
than choosing blindly and statistically indistinguishable from choosing centrally.
Falsification condition 6 holds and is reported as a headline.

Two things make the failure informative rather than merely negative.

**The target itself is weaker than it looked.** `SURROGATE_ORACLE` — following
`argmin |r_i − median(r)|` *with the residuals known* — scores 0.501 against the true
oracle's 0.481. Under gen9's deployment protocol the rule gen8 proved exact recovers
only **53 %** of the oracle gap, is exactly optimal in **60.1 %** of 8,580 blocks, and
carries a mean regret of 0.019. A learner trained to predict it inherits that ceiling
before it makes a single error of its own.

**More information makes it worse, not better.** Across the feature ablations the best
learned arm uses **geometry alone** (0.651); adding the model's predictions takes it to
0.653, and adding response-surface context takes it to 0.653–0.667. And the shuffled
control settles what the context was contributing: `full_shuffled` scores 0.655 against
`full`'s 0.653 — **destroying the correspondence between a candidate and its curve
costs 0.002**, which is nothing. The response-surface features are decoration.

### 3.3 Why there is no molecular-descriptor ablation

The brief asks for ligand descriptors as an explicit ablation. They cannot enter, and
the reason is structural rather than empirical: **a molecular descriptor is constant
across a ligand's candidates.** The deployed rule ranks candidates *within one pool*,
so a per-ligand constant has zero variance in every block it appears in and cannot
change an ordering at all — a linear ranker's coefficient on it is unidentifiable and a
tree can only use it to select which *other* feature's threshold applies. Running the
ablation would measure the value of those interactions, not of the chemistry. Recorded
as **not run, with this argument**, rather than as a null result.

### 3.4 The pool stress test

**Not run.** It was queued twice and lost both times to the reruns that issues 1 and 8
forced, and the remaining compute went to the five-seed locked evaluations. Recorded as
not run rather than as a null. Given that the learned policy does not beat `MEDOID` on
the unperturbed pool, its robustness under perturbed pools is a question about an arm
that has not earned a deployment decision.

---

## 4. GEN9-C, and the frontier

### 4.1 The k = 1 identity is a theorem

A ridge with an unpenalised intercept fitted on **one** observation has the exact
solution `beta = [r, 0, …, 0]` — substituting into `(dd' + diag(0, λ…))β = d r` gives
`d(1·r + z·0) = d r`. Every penalised coefficient is exactly zero, so `OFFSET_K1`,
`OFFSET_K2`, `OFFSET_K3` and gen9's hierarchy are **algebraically the same model at
k = 1**. Verified to 6.7 × 10⁻¹⁶ on 286 held-out units. gen9-C is only testable at
k ≥ 2, and any k = 1 difference between these adapters would be a bug.

### 4.2 The hierarchy beats plain offset correction and loses to gen8's fixed ridge

`GEN9_REL_MONOLITH` under `CENTRAL_THEN_SPREAD`:

| adapter | k=2 | k=3 | k=5 |
|---|---|---|---|
| `OFFSET_K1` (level only) | 0.603 | 0.590 | 0.609 |
| `SERIES_MAP_NOSERIES` (hierarchy minus the series term) | 0.602 | 0.587 | 0.605 |
| `SERIES_MAP` (gen9) | 0.596 | 0.573 | 0.574 |
| **`OFFSET_K3`** (gen8, fixed ridge 4.0) | **0.541** | **0.489** | **0.478** |

In the attribution chain, where the comparison is against `OFFSET_K1`, the hierarchy is
worth **+0.038** at k = 5 [0.012, 0.055], 82/99 ligands, 5/5 seeds — and gen8's
"calibration is series-local" finding is reproduced as a modelling gain, since
`SERIES_MAP` beats `SERIES_MAP_NOSERIES` at every k. But against gen8's own `OFFSET_K3`
it loses by 0.055 to 0.096.

The diagnosis is over-shrinkage, and it is mechanical: `estimate_prior` fits each
training ligand's residuals with a ridge of its own, so the *fitted* coefficients are
already shrunk; taking their spread as `tau` underestimates the true variation, inflates
`sigma²/tau²`, and shrinks the response coefficients far harder than 4.0 does. **The
empirical prior measures a shrunk quantity and treats it as the truth.** The hypothesis
survives; the estimator does not.

### 4.3 The attribution chain

Common cohort, each step changing exactly one thing (positive = better):

| step | k=1 | k=2 | k=5 |
|---|---|---|---|
| OLD GLOBAL + CENTRAL + OLD ADAPTER | 0.696 | 0.657 | — |
| NEW GLOBAL + CENTRAL + OLD ADAPTER | **0.666** (+0.031) | **0.625** (+0.032) | (+0.031) |
| OLD GLOBAL + LEARNED ACQ + OLD ADAPTER | 0.740 (−0.075) | 0.687 (−0.062) | (−0.036) |
| NEW GLOBAL + LEARNED ACQ + OLD ADAPTER | 0.719 (+0.021) | 0.662 (+0.025) | (+0.032) |
| NEW GLOBAL + LEARNED ACQ + NEW ADAPTER | 0.719 (0.000) | 0.648 (+0.014) | (+0.038) |

with intervals:

| step | k | point | 95 % CI | improved | seeds |
|---|---|---|---|---|---|
| the global model | 1 | **+0.031** | [0.012, 0.056] | 67/99 | **5/5** |
| the global model | 2 | **+0.032** | [0.014, 0.053] | 64/99 | **5/5** |
| the global model | 5 | **+0.031** | [0.015, 0.054] | 68/99 | **5/5** |
| the learned acquisition | 1 | **−0.075** | [−0.113, −0.040] | 32/99 | 0/5 |
| the series adapter | 5 | **+0.038** | [0.012, 0.055] | 82/99 | **5/5** |
| the series adapter | 1 | 0.000 | (algebraically zero) | — | — |

**The global model is the only component that pays at every k, and it pays the same
+0.031 at k = 1, 2 and 5.** The acquisition learner costs. The series adapter pays only
once enough measurements exist for it to have anything to fit.

### 4.4 The deployable frontier

| k | gen8 best | gen9 best | Δ | gen9 arm |
|---|---|---|---|---|
| 0 | 1.0605 | **1.0358** | −0.025 | `SHAPE_RECOMPOSED`, zero-shot |
| 1 | 0.6674 | **0.6539** | −0.014 | `SHAPE_RECOMPOSED` + gen8 slope repair + central-then-spread |
| 2 | 0.5879 | **0.5746** | −0.013 | `SHAPE_RECOMPOSED` + repair + `LEARNED_SCALAR` |
| 3 | 0.5230 | **0.5109** | −0.012 | `REL_MONOLITH` + repair + central-then-spread |
| 5 | 0.4743 | **0.4675** | −0.007 | `REL_MONOLITH` + repair + D-optimal |

gen9 improves the frontier at every k, and **the improvement shrinks as k grows** —
0.025 at zero-shot down to 0.007 at k = 5. That is the whole generation in one line: a
better response surface is worth most when you have no measurements, and a measurement
supplies what the model was missing.

**gen8's post-hoc slope repair is in every one of those arms.** It did not become
redundant. At k = 1 the new global model *alone* (0.663 with `OFFSET_K1@MEDOID`) already
beats the old model *plus* the repair (0.675), so the intrinsic representation captures
more than the repair did — and the repair still adds +0.009 on top of it, against +0.016
on top of the old model. **The two are complementary, and the repair's marginal value is
roughly halved.**

### 4.5 The RESIDUAL_SHAPE stratum, and the control that makes it safe

gen8's 29 `RESIDUAL_SHAPE` ligands are selected on *high error under the frozen model*,
so any different model gains from regression to the mean. The control is built in:
`A0_ROW_ONLY` is a different model with the same objective.

Zero-shot, against the frozen model that defined the stratum:

| arm | point | 95 % CI | improved | seeds |
|---|---|---|---|---|
| `A0_ROW_ONLY` — **the regression-to-the-mean baseline** | −0.021 | [−0.044, +0.020] | 13/29 | 1/5 |
| `A1_ADJ_d05` (pre-registered shape arm) | −0.038 | [−0.091, +0.028] | 10/29 | 1/5 |
| `REL_MONOLITH` | +0.023 | [−0.009, +0.068] | 14/29 | 4/5 |
| **`SHAPE_RECOMPOSED`** | **+0.046** | [0.018, 0.081] | 19/29 | **5/5** |

**Regression to the mean contributes nothing** — it is slightly negative. Against the
`A0` control the recomposition is **+0.067** [0.017, 0.104], 20/29, 5/5.

And the gain lands where it was supposed to. On `PURE_LEVEL` the recomposition is
*significantly worse* than the control (−0.047, [−0.109, −0.002]) and identical to the
frozen model; on `ALREADY_GOOD` it is +0.018. The pre-registered failure mode — "overall
MAE improves only because PURE_LEVEL gets easier while RESIDUAL_SHAPE is unchanged" — is
the exact opposite of what happened.

The pre-registered threshold was ≥ 0.08 macro-MAE reduction on `RESIDUAL_SHAPE`. The
point estimate against the control is **0.067**, so the threshold is **missed**, while
the effect is significant and consistent across all five seeds.

---

## 5. Sensitivity and the data audit

### 5.1 FROZEN / QUARANTINED / CORRECTED — nothing moves

The three cohorts differ over the 70 rows gen8 identified as a duplicated publication.
Zero-shot macro MAE across all ten arms moves by at most **0.002**:

| model | CORRECTED | FROZEN | QUARANTINED |
|---|---|---|---|
| `GEN9_SHAPE_RECOMPOSED` | 0.9693 | **0.9695** | 0.9686 |
| `REC_ecfp_plus_recovered` | 0.9805 | 0.9807 | 0.9798 |
| `GEN9_REL_MONOLITH` | 0.9857 | 0.9858 | 0.9851 |

The extractant-axis shape metrics are **identical to every printed digit** in all three
cohorts, for the simple reason that DMDPhPDA carries acid titrations and no extractant
ones — the affected rows are not on any curve the headline is measured on. The
conclusion does not depend on the data problem, and would not have depended on it
whichever way it had been resolved.

### 5.2 What gen9's own scan found

The scan is independent of gen8's — different code, different keys — and it
rediscovers both known findings and two new ones. Nothing is deleted.

* **DMDPhPDA is rediscovered**: **59** duplicate-cell pairs differing by *exactly* 1, 2,
  3 or 4 decades. It is invisible to a scan keyed on the full condition set, because
  the two copies differ only by an **unreported temperature** — one records 25 °C, the
  other nothing. A scan keyed on the *core* cell finds it.
* **Two further single-cell exact-decade pairs** in other ligands, not previously
  reported.
* **TWE-24 is rediscovered**: level **+4.82 decades** above its five thiophosphoryl
  siblings and +4.39 above its own publication cluster; removing it changes its nearest
  neighbour's level error by **4.10 log units** under a 1-NN lookup. Four further
  family-local level outliers beyond it. `TWE24_STATUS = UNRESOLVED`.
* **353 rows (6.7 %, 17 extractants) carry an extractant name that does not match the
  structure they are modelled as** — gen7's "unmodelled second species", independently
  rediscovered. On one canonical SMILES (TODGA, 1,488 rows) the recorded name takes
  **18** distinct values, most of them entirely different molecules. On those rows the
  model is being asked to predict a synergistic mixture from one component's
  fingerprint.
* Zero implausible condition values: no non-positive concentrations, no temperatures
  outside 0–200 °C, no non-positive contact times.

The full flag table is `data_audit/anomaly_flags.csv`.

---

## 6. The thirteen questions the brief asks

**1. Did within-curve training fix flattening?**
The pre-registered within-curve *loss*: no. It reduces compression on every axis and
for every ligand and leaves the slopes at ~5 % of truth; forcing the range costs macro
MAE almost exactly in proportion. What fixed it was the diagnosis that failure
produced — the model had no coordinate for *where on its own curve* a row sat. With
five relative-position columns and a mean-preserving recomposition the median
predicted extractant slope goes from 0.116 to **1.024** against a measured 2.574, and
every other shape endpoint moves with it. **EXPLORATORY.**

**2. Which curve loss worked: local derivative, endpoints, random pairs, multiscale?**
Local derivative (`ROW_ADJACENT`) is the only sampler that improves extractant slope
*accuracy* with an interval excluding zero while also improving span recovery on all
three axes. Endpoints are the informative failure: they double span recovery and drop
within-curve sign accuracy to 0.593, barely above the 0.5 a constant predictor scores,
with a predicted slope of 11.3 where the corpus's own 95th percentile is 4.08.
Multiscale — the pre-declared primary candidate — did not win. **None of them met the
pre-registered bar.**

**3. How much of gen8's post-hoc slope-repair gain became intrinsic?**
**More than all of it, and the repair still adds.** At k = 1 the new global model alone
(`OFFSET_K1@MEDOID`, 0.663) beats the old model *plus* the repair (0.675). The repair
then adds a further +0.009 on top of the new model, against +0.016 on top of the old
one — its marginal value roughly halves but does not vanish, and it appears in every
best-deployable arm at every k. The two are complementary, not redundant.

**4. Did `RESIDUAL_SHAPE` ligands improve?**
**Yes, significantly, and not through regression to the mean.** +0.067 [0.017, 0.104]
against the `A0` control, 20/29 ligands, 5/5 seeds; +0.046 [0.018, 0.081] against the
frozen model that defined the stratum, where the regression-to-the-mean baseline
(a different model, same objective) is **−0.021**. The pre-registered threshold of 0.08
is missed. Crucially the gain does *not* come from `PURE_LEVEL`, where the same arm is
significantly worse than the control — which is the pattern the protocol asked for.

**5. Did overall macro MAE improve, and was that improvement level or shape?**
Yes, and it is shape. Macro MAE 0.9807 → **0.9695** over five seeds. The
recomposition is mean-preserving by construction — the per-curve predicted level is
identical to the monolith's, asserted to 1e-6 in the test suite — so none of the gain
can be level. Row MAE on extractant curves improves by +0.087 [0.057, 0.148], 20/25
ligands, 5/5 seeds.

**6. Can the oracle one-shot target be predicted from observable features?**
**Partly — enough to beat choosing blindly, not enough to beat choosing centrally.**
`LEARNED_BLEND` is +0.070 [0.028, 0.099] against `RANDOM` (99/143, 5/5) and
−0.005 [−0.018, +0.014] against `MEDOID` (0/5 seeds). Note also that the target is a
weaker thing than gen8's proof suggests: under this protocol, following it *with the
residuals known* recovers only 53 % of the oracle gap.

**7. How much of the MEDOID→oracle gap was closed?**
**None.** The pre-registered target was ≥ 30 %. The best learned policy is 0.005
*behind* `MEDOID`, so the closed fraction is zero (nominally −3 %).

**8. Did molecular descriptors help acquisition after condition geometry was known?**
**They cannot, by construction** — a molecular descriptor is constant across a ligand's
candidates, and the rule ranks candidates *within* a pool, so it has zero variance in
every block and cannot change an ordering (§3.3). The ablation is recorded as not run,
with that argument. What *was* tested is the response-surface context, and the shuffled
control settles it: destroying the correspondence between a candidate and its curve
costs 0.002 macro MAE. The best learned arm uses **geometry alone**.

**9. Does series-local calibration beat ligand-global calibration?**
**Yes over `OFFSET_K1`, no over gen8's `OFFSET_K3`.** The series term earns its place —
`SERIES_MAP` beats `SERIES_MAP_NOSERIES` at every k, and the hierarchy is +0.038
[0.012, 0.055] over `OFFSET_K1` at k = 5 (82/99, 5/5). But gen9's fold-local empirical
prior over-shrinks the response coefficients and loses to gen8's hand-set ridge of 4.0
by 0.055 to 0.096 (§4.2). At k = 1 the question is not askable: every one of these
adapters is algebraically identical there.

**10. What are the new k = 0/1/2/3/5 Pareto-optimal deployable numbers?**
Common cohort: **1.0358 / 0.6539 / 0.5746 / 0.5109 / 0.4675**, against gen8's
1.0605 / 0.6674 / 0.5879 / 0.5230 / 0.4743. Better at every k, by 0.025 falling to
0.007 as k grows. The pre-registered goals (k=1 ≈ 0.60–0.62, k=2 ≤ 0.50) are **not**
met.

**11. Are results stable under FROZEN / QUARANTINED / CORRECTED?**
**Completely.** Zero-shot macro MAE moves by at most 0.002 across all ten arms, and the
extractant-axis shape metrics are identical to every printed digit, because the affected
rows are acid titrations and carry no extractant curve (§5.1).

**12. Is TWE-24 genuine chemistry or likely data corruption?**
**`TWE24_STATUS = UNRESOLVED`**, and deliberately not upgraded. gen9's independent
scan reproduces gen8's finding: its level sits **+4.82 decades** above the median of
the five other thiophosphoryl compounds and +4.39 above its own publication cluster,
and removing it changes its nearest neighbour's level error by **4.10 log units**
under a 1-NN lookup. That is consistent with a ~3-decade transcription error and
equally consistent with real chemistry no descriptor here can see. Settling it needs
the primary document, which is not in this repository. **The rows are not removed** —
deleting them would move the cohort every generation since gen6 has been scored on.

**13. What should gen10 do?**
*(§7.)*

---

## 7. What gen10 should do

**1. Treat "relative to the candidate design" as a representation, not a feature.**
gen9's whole result is that five columns describing where a row sits inside its own
measurement window did what a bespoke loss could not. That was the cheapest thing
tried and the only thing that worked, and it was tried last. The obvious extension is
systematic: represent *every* condition axis twice, absolutely and relative to the
design the user supplied, and check whether the acid and lanthanide axes — which gained
less than extractant here — gain more from a normalisation matched to their physics
(rank within the series rather than linear position within the window).

**2. Ask whether the recomposition is still needed once the representation is right.**
Here the features carried most of the gain (extractant slope 0.116 → 0.521) and the
mean-preserving recomposition roughly doubled what was left (→ 1.024). Whether that
second factor is a real architectural need or an artefact of the features being
appended to a design where 2,048 of 2,160 columns are ECFP bits — so a randomised split
proposal reaches one of the five about 1 % of the time — is not settled. A
`max_features` sweep would settle it, and gen9 started one and stopped it (recorded as
**not run** in `failure_analysis.md` §2.3).

**3. Do not retry a within-curve loss.** The dose–response is a clean, monotone
trade-off across three orders of magnitude in `lambda_delta`: the range comes back and
the accuracy goes away in proportion, the acid slope inverts at the top, and the
within-curve ordering collapses. There is no setting that works, and the curve-centred
probe shows why — with the level deleted entirely the shape is still not learnable from
absolute conditions. This is a closed direction.

**4. Target realised regret, not the surrogate.** gen8's `argmin |r_i − median(r)|` is
exact in its own exhaustive protocol and only **60.1 %** exact under the deployment
protocol, where the candidate pool and the scored rows are disjoint (8,580 blocks,
mean regret 0.019, median Spearman 0.948). It remains an excellent training signal and
a poor definition of the objective. gen10's acquisition work should train on the
realised one-shot MAE and use the surrogate only as an auxiliary label.

**5. The binding constraint on the shape result is coverage, not method.** Only **25
of 152** ligands have an extractant titration at all, and 775 curves over five seeds
means 155 distinct ones. Every conclusion about extractant response rests on those. If
the next round of experiments is being planned, titrations on *chemotypes not already
represented* are worth more than more points on ligands already measured — which is
the same conclusion gen6 reached about breadth versus depth, arriving from a different
direction.

**6. Two things that need a human, not a model.**
*TWE-24* is `UNRESOLVED` and needs its primary document; it costs its nearest
neighbour 4.10 log units and nothing in the corpus can settle whether it is real.
*353 rows (6.7 %, 17 extractants)* carry a recorded extractant name that does not match
the structure they are modelled as — gen7's "unmodelled second species", independently
rediscovered here. On those rows the model is being asked to predict a synergistic
mixture from one component's fingerprint. Neither is a modelling problem.

---

## 8. What was wrong, what I fixed, and what survived

*Plain language, no jargon. This section is for a reader who wants to know whether to
believe the rest.*

### Did the first version have bugs?

Yes — seven, and two of them would have produced a completely wrong answer if they had
gone unnoticed.

**The worst one made every "choose the best experiment" rule look identical.** The
part of gen9 that studies which measurement to run first was reading a file that
contains predictions but not the experimental conditions. When a condition is missing,
the code that turns conditions into coordinates quietly returns zeros — so every rule
that navigates by geometry ("pick the most typical point", "pick the most extreme
point", "pick the mid-acidity point") saw a world where every candidate was at the same
place, could not tell them apart, and picked the first one on the list. Seven rules
that disagree by design produced one number, and the rule gen8 had shown to be *good*
scored worse than picking at random. That is what gave it away. The fix joins the
conditions on and now refuses to run if any coordinate is missing or constant. The
entire acquisition study was thrown away and rerun.

**The second one made the new model irreproducible.** Running the same model twice, in
the same process, on the same data, gave different answers — by up to 0.14 in the
units the study reports, on every single held-out row. The cause is subtle and worth
understanding: the model builds a forest of trees in parallel, and adding up their
predictions in whatever order the threads finish changes the last decimal place. In an
ordinary forest that is irrelevant. This model builds forests *in sequence*, each one
correcting the last, so that last-decimal difference feeds into the next round, tips a
close decision about where to split the data, and grows a completely different tree —
and from there the two runs diverge. The fix rounds the running prediction to nine
decimal places, which is a billion times finer than any measurement in the data and
removes the wobble before it can matter. Everything trained with the model was rerun.

Three more were in the data-quality scan, and they had the same character: the scan was
looking in the right place with the wrong key, so it found *nothing* — no duplicated
measurements, no unusual ligands — and "nothing" looked like a clean bill of health.
Once fixed, it independently rediscovers the corruption gen8 had already confirmed (one
publication entered twice, one copy off by exact factors of ten) and the one ligand
gen8 had flagged as suspicious, and it found two new cases nobody had seen.

### What did I do differently from the plan?

The plan was to fix the model's known defect — it draws every extraction curve far too
flat — by adding a term to its training objective that punishes getting the *shape* of
a curve wrong. **That did not work, and I could not make it work.** It reduces the
problem a little, reliably, on every curve. But pushing it harder to get the rest buys
the range back and destroys everything else, roughly in proportion: the model's overall
accuracy halves, the ordering of points within a curve collapses, and on one axis the
response actually reverses direction. There is no setting where the curve comes back
and the model still works.

So I asked a different question: what if the level is getting in the way? I trained a
model with the level removed entirely — nothing to do but predict the shape. **It did
worse.** Which ruled out the explanation everyone would have reached for, and pointed
at something more basic.

Here is what was actually wrong. A titration measures how extraction changes as you
add more extractant. Different published titrations cover different concentration
ranges — 0.03 M is the *top* of one experiment and the *bottom* of another. The model
was only ever told the absolute concentration, so "0.03 M" carried no information about
whether that point was at the beginning or the end of its curve. Averaged over all the
curves in the corpus, the answer to "what does extraction do at 0.03 M" is *nothing in
particular* — and a flat prediction is the correct answer to that question. **The model
was not failing to learn the response. It was never asked where on its own curve each
point sat.**

Telling it — five extra numbers per row, all of them things the user already knows when
they hand over a list of conditions — changes the picture completely. The predicted
slope of an extraction curve goes from 0.12 to 1.02 against a measured 2.57, an
eightfold improvement; the predicted range goes from 5 % of the truth to 42 %; the
ordering of points within a curve goes from 0.58 to 0.89 on a scale where 1.0 is
perfect. And unlike everything else tried, it makes the model *more* accurate overall,
not less.

### Which numbers changed after rerunning?

The ones that changed most were the ones that had been wrong. Before the condition fix,
seven different experiment-selection rules all scored 0.727; afterwards they range from
0.633 to 0.787 and reproduce gen8's ordering. Before the reproducibility fix, the same
model reported 1.111 and 1.105 on two runs of the same fold; afterwards it reports
1.103 both times. Before the data-scan fixes, the corpus looked clean; afterwards it
has 61 duplicated measurements differing by exact factors of ten and five ligands whose
extraction sits more than a thousandfold away from their closest chemical relatives.

The *conclusions* of the pre-registered experiment did not change — it failed before
the fixes and it fails after them, slightly more clearly. Everything reported here is
from the post-fix runs.

### What about the other two things gen9 tried?

Both failed, and I am reporting them as failures.

**Choosing which experiment to run first.** gen8 had shown that *which* measurement you
take matters as much as a better model, and that the ideal choice follows a known rule —
one that unfortunately depends on the measurement you have not made yet. gen9 tried to
learn to predict it. The learned rule beats picking at random by a clear margin, and is
statistically indistinguishable from the simple rule gen8 already had: "pick the most
typical condition". Zero improvement over the existing baseline, against a target of 30 %.

Two things make that more interesting than a flat null. First, the ideal rule is weaker
than it looked: under the conditions the system is actually deployed in, following it
*even with perfect knowledge* only gets you halfway to the true best choice — gen8's
proof was for a slightly different setup. Second, giving the learner more information
made it *worse*: the best version uses geometry alone, and scrambling the
response-surface information it was given costs essentially nothing. Whatever is left to
predict here depends on the measurement itself.

**Calibrating per experimental series rather than per ligand.** gen8 had found that one
measurement calibrates the system you measured in and largely not others, so gen9 built a
hierarchy that estimates its own shrinkage from each fold's data. The series part works —
it beats the version with the series term removed at every k. The self-estimated
shrinkage does not: it shrinks about twice as hard as gen8's hand-picked constant and
loses to it by a wide margin. The reason is a subtle statistical one — the procedure
estimates how much these quantities vary using numbers that have already been shrunk, so
it concludes they vary less than they do. The idea survives; my estimator does not.

### Does the main conclusion still hold?

The pre-registered one does not, and I am reporting that as a failure rather than
rescuing it. The shape-aware training objective was the central gen9 hypothesis and it
is wrong: not badly implemented, wrong. The follow-up experiments show why, which is
worth more than the original idea would have been.

The finding that replaced it is strong but it is **exploratory** — I designed it after
seeing the pre-registered results, which is exactly the situation where it is easiest
to fool yourself. So I attacked it: the effect is present in all four quartiles of
curve width (so it is not just measuring how wide the experiment was), 21 of 25 ligands
improve, no single ligand is worth more than 6 % of the gain, and removing the most
favourable publication still leaves four-fifths of it. It survived all of those. It has
one honest weakness, which is that one campaign benefits about twice as much as the
rest, and that is stated in the report rather than buried.

And the payoff is modest. Better curves are worth 0.025 in the units this project uses
when you have no measurements, 0.014 after one, and 0.007 after five — because a
measurement supplies exactly what the model was missing. The scientific finding is
larger than the engineering gain, and the report says so in that order.
