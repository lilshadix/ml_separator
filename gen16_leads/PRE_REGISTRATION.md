# Gen16 — pre-registration: six leads, one confirmation

*Frozen 2026-09-10 after Phase 0 of `START_HERE.md` (three anchors reproduced, L6 cohort audit
read, environment verified) and before any lead arm was run on the fold plan.  This file is not
edited once a lead runs; a change is a dated addendum stating what changed, why, and whether any
result had been seen.  The SHA-256 of everything above the final footer line is recorded in that
footer and in `DECISION_REPORT.md`.*

## 0. What is inherited, and what is committed to here

| | |
|---|---|
| brief | `gen16_leads/START_HERE.md`, commit `df63929` |
| cohort | frozen gen13 cohort, fingerprint `4c3c6628ea0be949`: 521 cells, 90 extractants, 45 chemotypes, Kish n_eff 11.7 |
| fold plans | `gen13sep.splits.all_folds`, designs **A, B, BR, BQ, BP**, 5 folds × the five **discovery seeds** `(104729, 130363, 155921, 196613, 262147)` |
| metric | extractant-macro MAE of pairwise `log SF` (`gen13sep.metrics.per_extractant` / `summarise`, `macro_mae_extractant`), secondary `macro_mae_far`, `macro_sign_acc_strong`, `macro_pair_spearman`; direction tasks use macro accuracy over extractants with ≥ 5 metals |
| bench | `gen15.valuebench.score` (`arm(ctx) -> (n_test, 2)`), `gen15.fewshot.evaluate` for the measured mode, `gen14.dirbench.run` for direction tasks.  No new metric, splitter or bootstrap is written. |
| inference | `gen13sep.inference.paired_contrasts`: chemotype-blocked paired bootstrap, 10 000 replicates, seed 8675309, percentile and BCa intervals, two-sided p, per-seed sign agreement, leave-one-chemotype-out sweep |
| decision rule **P1** | point ≥ **0.02**, both 95 % intervals exclude 0, p < 0.05, ≥ 4/5 seeds same sign (we additionally require 5/5 for a confirmed claim), no single held-out chemotype flips the sign |
| selection | BP selects; design B never selects; any number quoted for "new chemistry from a new group" is the BP number |
| floor | `FLAT` (predict no separation), 0.5885 under BP.  `MEAN_CURVE` (0.6217) is not a baseline. |
| two comparators | every claim is quoted against `FLAT`/the constant rule **and** against the cheapest sensible competitor named in its section |
| matched null | any block or descriptor that helps is re-run as a width-matched shuffled block (`real − shuffled` reported, not only `real − nothing`) |
| oracles | every own-cell oracle used as a ceiling is **leave-pair-out** (refit without the two metals of the scored pair, `exp/labelerr/s4_floor.py` construction) |
| machine limits | 8 GB RAM; one bench process peaks at ~230 MB (Phase 0 measurement); at most 6 concurrent bench processes; `n_jobs ≤ 2`, `OMP_NUM_THREADS=2` |

**Anchors reproduced in Phase 0** (`results/anchors/ANCHORS.md`, test `tests/test_anchors.py`):
`G13_ET_TOPO39` under BP = `0.7683085207475452` (exact); `G14` under BP = 0.5001 (obtained
0.500079); `FLAT` under BP = 0.5885 (obtained 0.588506); `G14 − FLAT` = +0.0884, 95 % CI
[+0.0010, +0.1473], p = 0.047.  The fold plan hashes identically across two subprocesses.

**Confirmation seeds — withheld.**  Five seeds were generated before any lead ran, from the fixed
rule

```
seed_i = int(sha256("gen16-confirmation-seed-{i}").hexdigest()[:8], 16) % 900000 + 100000,
i = 1, 2, …; skip any collision with the discovery seeds or an earlier seed; take the first five;
sort ascending.
```

The sorted list's commitment is `sha256(json.dumps(seeds))` =
`5a30455bc67d364aa3e65f6d3d28bd4b6fcf3301fe979ca4a1be4a9547de76a9`.  The seeds themselves are
held by the orchestrator outside the repository until the confirmation run and are printed in
`CONFIRMATION.md` with a check that they hash to the commitment.  Every discovery agent is
instructed never to pass `seeds=` to any bench call; `scripts/g16_audit_seeds.py` greps every
script under `gen16_leads/` for `seeds=` and fails if any file other than the confirmation
script uses it.  The withholding is procedural, not cryptographic: the rule is public and anyone
can compute the seeds; what is guaranteed is that no discovery result was computed on them.

## 1. Questions

* **L1** — Does a cycle-corrected GFN2-xTB interaction slope along the lanthanide series
  correlate with the observed amplitude once the inner-sphere composition step is removed
  exactly rather than approximately?  (New physics; the only live representational hypothesis.)
* **L2** — *Gate:* is the honest, leave-pair-out curvature headroom under BP still worth chasing
  (≥ 0.02)?  *If so:* is the vertex radius `r0 = −a/(2b)` a better-behaved, more predictable
  chemotype-level quantity than `b`, and does predicting it move BP macro MAE?
* **L3** — Three narrow decision questions: how many measurements does the direction call save;
  can candidates be ranked within a chemotype (conditional on L1); can candidates that each
  carry one measurement be ranked better with the corpus than with a line through that
  measurement?
* **L4** — Which chemotypes should be measured next: does a model-chosen acquisition order make
  BP macro MAE fall faster with corpus size than random or maxmin order, retrospectively?
* **L5** — Does a better-estimated residual covariance (shrinkage, low-rank, chemotype-balanced)
  or the gen15 §7 component-mean mixture beat the *deployed* measured-mode route, and are the
  programme's intervals and direction probabilities calibrated?
* **L6** — (audit, Phase 0, no hypothesis) does any single cohort filter discard ≥ 5 independent
  chemotypes?

## 2. Shared rules for every lead

1. Every arm is scored under **all five designs**; a candidate is supported only if the sign of
   its effect is the same in all five.  A table with fewer than five designs is exploratory.
2. Every contrast is written to `results/<lead>/contrasts_*.csv` with the `paired_contrasts`
   columns plus a column `family ∈ {registered, exploratory}`.  The registered contrasts are the
   ones listed in §3; **everything else any agent evaluates is exploratory and is still written
   out**, so the comparison count in `DECISION_REPORT.md` is the row count of those files.
3. Benjamini–Hochberg is applied across the whole exploratory family and, separately, across the
   registered family; raw and adjusted p are both reported.  A registered contrast is judged by
   P1 on its raw p (P1 is a joint rule, not a single p), and its BH-adjusted p is printed beside
   it.
4. Discovery uses the five discovery seeds only.  At the end of discovery **at most five claims**
   are frozen in `CONFIRMATION.md` (arm, reference, metric, designs) and run **once** on the
   withheld seeds.  Whatever confirmation returns is the result; no sixth claim is added.
5. Every claim that survives discovery gets two refuter agents, spawned blind to each other, each
   asked for the control, confound or bookkeeping error that would remove it, and each required
   to check: carried by the diglycolamides alone? survives removing them? an artefact of the
   number of metals measured per extractant (ρ = +0.49 with |a|)? survives the publication mask?
   comparison arm matched on training-set size? shuffled block scores the same?
6. No cell, extractant, chemotype, seed, design, baseline, margin or endpoint is changed after a
   number is seen.  If an agent is tempted to, it writes the temptation into
   `REFUTATION_LOG.md` instead.
7. Nothing on the established-negative list of `START_HERE.md` §2 is re-run: no thirteenth
   magnitude prior, no similarity kernel, no chemical language model, no label de-noising, no
   conditions-only arm, no new direction classifier.

## 3. Leads: arms, endpoints, decision rules, stopping rules

### L1 — cycle-corrected xTB descriptor

**Data.** `dataset with 3D structures/accepted_geometries.csv` (1155 complexes, 177 ligands,
14 metals) joined to `features/complex_physical_scalars.parquet` (`complex_total_energy_eV`,
1116 non-null).  Composition: `n_ligs ∈ {1..4}`, `fill_ligand = inner_sphere_anion ∈ {nitrate,
water}`, `n_fill ∈ {0..5}`.  Scaffolding: `gen15_curve/exp/phys3d/` (`build_block.py`,
`trap_check.py`, `descriptor_stats.py`, `robust_stats.py`), extended under `gen16_leads/gen16/`.

**Stage 1 — bookkeeping (runs now, once).**  Three energy decompositions on identical rows:

| model | terms | status |
|---|---|---|
| `NAIVE` | series FE + metal FE | gen15's naive slope (reproduce +0.644 on its 39-extractant subset first) |
| `ELEM` | + element counts as within-series covariates | gen15's construction B (reproduce) |
| **`SPECIES`** | + `n_fill × γ_species` (one coefficient per fill species) + `n_ligs × δ_ligand` where `n_ligs` varies within the series | **registered** |
| `SPECIES_CONST` | `SPECIES` restricted to series with constant `n_ligs` | registered, secondary; report n |

Per extractant the interaction slope is the coefficient of the standardised Shannon radius
(`gen13sep.metals.SHANNON_RADIUS_CN8`, the bench's basis axis) in a within-series linear +
quadratic fit of the model residual; the curvature term is the quadratic coefficient.

**Sets, fixed in advance.**  `S8` (primary): cohort extractants with a well-determined curve
(≥ 5 measured metals, the 82 of `extractant_targets.parquet`) and ≥ 8 computed metals in the
series.  `S14` (secondary): complete 14-metal series.  `S3`: ≥ 3 computed metals (all).  Every
correlation is reported with its n.

**Registered statistics (per set, per model).**  Spearman ρ of the slope with the signed
amplitude `a` (primary target), with `|a|` and with `b` (secondary); LOCO min/max over
chemotypes; partial Spearman given the number of metals the experiment measured; chemotype-
blocked bootstrap 95 % CI (2000 replicates, `robust_stats.py` construction); and a family-wise
permutation null (chemotype-level permutation of the target, max |ρ| over the 4 models × 3 sets).

**Decision rule.**  L1 is *positive* if, on `S8`, `SPECIES` gives |ρ(slope, a)| ≥ 0.40 with a
LOCO-stable sign, |partial ρ | n_metals| ≥ 0.40, and the chemotype-blocked CI excludes zero.
It is *strong* if in addition the arm `G14+PHYSCYC` (gen14 direction logistic on TOPO39 plus the
slope and curvature columns; magnitude from a fold-fitted isotonic map of the slope, `_fit_magnitude_1d`)
beats `G14` under P1 in all five designs **and** beats its width-matched shuffled block
`G14+PHYSCYC_SHUF` under P1.  It is *closed* if |ρ| < 0.25 on `S8` with a CI containing zero.
Between 0.25 and 0.40 it is "not closed, not positive" and is said so.

**Registered contrasts (MAE endpoint):** `PHYSCYC_vs_G14`, `PHYSCYC_vs_SHUF`, `PHYSCYC_vs_FLAT`.
Cheapest competitor for the correlation: `coord__dist__frac_donor_pairs_within_3` at ρ = −0.53.

**Stage 2 — reference species (cluster; the user submits).**  The reference set is every
distinct free ligand (177), fill species (nitrate⁻, water), and Ln³⁺ ion (14), at the same
GFN2-xTB level and charge convention as the complexes (inferred from the xyz headers and
recorded).  `dE = E(complex) − E(Ln³⁺) − n_ligs·E(L) − n_fill·E(fill)`; then the same two-way
fit, statistics and decision rule as Stage 1, unchanged.  The analysis script is delivered
ready (`scripts/l1_stage2_cycle.py`) with a completeness check that every
`(ligand, fill, n_fill, n_ligs)` combination has its references; a missing reference aborts the
run rather than silently reintroducing the step.

**Stopping rule.**  Stage 1 runs once on the pre-specified sets and models.  Any further subset
or model is exploratory.

### L2 — the curvature: gate, then `r0` (gated)

**Gate (runs first, once).**  On the standard BP pair tables (5 discovery seeds), `O_CURV_LPO`
= gen14's amplitude (logistic sign × training-fold mean magnitude) with the cell's own `b`
refitted **without the two metals of the scored pair** (`s4_floor.py` `_fit(drop=(a, b))`).  The
in-sample `O_CURV` computed by the same code must reproduce 0.4272 ± 0.001 or the gate is
invalid.  Headroom `H = MAE(G14) − MAE(O_CURV_LPO)` with the paired bootstrap.  **Open** if
H ≥ 0.02 and the CI excludes zero; otherwise **closed** and the lead ends with one paragraph.

**If open.**  Subset `R`: well-determined cells (≥ 5 metals) whose vertex `r0 = −a/(2b)` lies
inside the cell's measured radius range and whose `b` is determined (|b| ≥ 2 × its leave-one-
metal-out standard error).  Report |R|.  Registered statistics: chemotype ICC of `r0` on `R`
versus the ICC of `b` on the same `R` (both with the same one-way ANOVA estimator); the lead is
*dead* if ICC(r0) ≤ ICC(b).  Registered arm `G14_R0`: `a` from `G14`; `r0` predicted by a
weighted ridge on `BITE6` (α = 1, standardised, fitted on training-fold `R` cells); `b = −a/(2·r0)`
with `r0` clipped to the radius axis's measured span.  References: `G14` (constant `b`) and
`B_DIRCOND_MEAN` (direction-conditional constant `b`, `gen15.shape`).  Matched null
`G14_R0_SHUF` (BITE6 permuted between extractants inside the training fold).  Registered
contrasts: `R0_vs_G14`, `R0_vs_DIRCOND`, `R0_vs_SHUF`, P1, five designs.  Mutual check with L1:
Spearman(r0, L1 `SPECIES` slope) on `S8 ∩ R` extractants, reported with n.

**Stopping rule.**  One arm, one reference pair, one null.  No tuning of α or the clip after a
score is seen.

### L3 — three narrow decision questions

Tasks follow `gen15_curve/exp/decision` Q3: per (seed, unordered metal pair), the candidate set
is every held-out extractant that measured that pair (an extractant's cells collapsed by the
median observed `log SF`); tasks with < 5 candidates are dropped and counted.  Tie handling is
the exact expectation under uniform random tie-breaking (`decmetrics.py`).

**L3a — measurements saved by the direction call.**  A candidate *succeeds* for a requested
direction if its observed `log SF` has that sign and |log SF| ≥ 0.3.  Without the model the
chemist measures candidates in uniform random order: expected draws to the first success
`(N + 1)/(K + 1)` (K successes among N; N + 1 if K = 0).  With the model the candidates whose
`G14` sign call disagrees with the request are deferred: expected draws = that formula over the
kept set, falling through to the deferred set if the kept set has no success.  **Saved =
E_random − E_model**, averaged over the two directions and over tasks; reported absolutely and as
a fraction of E_random.  Cheapest competitor: the "always heavier" rule used the same way.
Null: the model's sign calls permuted across candidates within each task (2000 permutations).
Interval: chemotype-blocked bootstrap over extractants.  *Positive* if saved > 0 with
permutation p < 0.05 and CI excluding zero in all five designs.

**L3b — ranking within a chemotype.**  Conditional on L1 Stage 1 being *positive*: restrict Q3
ranking to candidates of one chemotype (≥ 4 candidates), rank by `G14+PHYSCYC`; regret and
Spearman against `G14` and random.  If L1 is not positive this question is **dropped, not run**.

**L3c — ranking at k = 1.**  Every candidate receives one measured pair: its widest-`dZ`
measured pair *excluding the target pair* (candidates whose only pair is the target are dropped
and counted).  Predictions for the target pair: `G14@k1` (BLUP, leave-chemotype-out and
publication-masked covariance, the deployed route), `NAIVE_LINE@k1`, zero-shot `G14`, random.
Metrics as Q3 (Spearman, tie-expected top-1, regret).  Registered contrast: `G14@k1 −
NAIVE_LINE@k1` on regret and on Spearman; unit = task with a task bootstrap, plus a
chemotype-blocked bootstrap over extractants (the conservative one is quoted); permutation null
as in L3a.  *Positive* if the regret gain has p < 0.05 with a consistent sign in all five designs.

**Stopping rule.**  Exactly these three questions.  Anything beyond them re-runs a closed
experiment and is not done.

### L4 — which chemotypes to measure next

**Retrospective simulation.**  For every fold of every design, the training chemotypes are
ordered by a criterion computed **from features only** (never from the labels of chemotypes not
yet added); budgets `k ∈ {6, 9, 12, 16, 20, 24}` chemotypes; the `G14` arm is refitted on the
first k; the held-out cells are unchanged, so every budget scores byte-identical pairs.  Orders:

| order | definition |
|---|---|
| `RANDOM` | uniform random order, 20 draws per fold (the null) |
| `MAXMIN` | farthest-point traversal in ECFP4 Tanimoto distance between chemotype centroids, 20 random starts |
| **`AOPT`** | greedy A-optimal for the standardised ridge-logistic on `TOPO39` (penalty 1, as `G14`): add the chemotype that most reduces the trace of the posterior variance of the linear predictor over the *whole* training pool |
| `UNCERT` | first three by `AOPT`, then the chemotype whose cells' `G14` probability is closest to 0.5 under the current fit |

**Primary endpoint.**  Area between curves `ABC = mean_k [MAE_RANDOM(k) − MAE_AOPT(k)]`
(random averaged over its draws) per extractant, paired chemotype-blocked bootstrap; P1 (margin
0.02 on ABC), five designs.  Secondary: `AOPT − MAXMIN` under the same rule; `UNCERT` under the
same rule; the same three on macro direction accuracy; the percentile of `AOPT`'s ABC among the
20 random draws (permutation-style null).  Registered contrasts: `AOPT_vs_RANDOM`,
`AOPT_vs_MAXMIN`, `UNCERT_vs_RANDOM`, `UNCERT_vs_MAXMIN`.

**Prospective deliverable (not scored).**  Rank two pools by the `AOPT` criterion given the full
cohort: (i) bundle extractants with a single measured lanthanide and a frozen chemotype label;
(ii) the 273 external logK ligands with `TOPO39` available.  Annotate each with chemotype (or
"new"), nearest-cohort Tanimoto, donor census, and the first three D-optimal pairs from
`gen15.fewshot.pick_support` with the full-cohort residual covariance.  State explicitly that
the simulation measures ordering among chemotypes the corpus already has, not the value of
genuinely new chemistry.

### L5 — covariance and calibration

**Route.**  `gen15.fewshot.evaluate` semantics, re-implemented under `gen16_leads/gen16/` where
a covariance option is missing (gen15 files are not modified): leave-chemotype-out residual
curves, `mask_publication=True`, support = greedy D-optimal, common scoring set = union of every
strategy's supports, `k ∈ {0, 1, 2, 3}`.  Deployed baseline = `G14@doptk{k}` with the empirical
shrink-0.25 covariance (`POOLED`; 0.1700 at k = 3 under BP).

| arm | covariance |
|---|---|
| `POOLED` | empirical, shrink 0.25 (deployed) |
| `LW` | Ledoit–Wolf shrinkage to a scaled identity |
| `LOWRANK2`, `LOWRANK3` | rank-2 / rank-3 + diagonal (probabilistic PCA) |
| `HIER` | chemotype-balanced empirical covariance (each chemotype's residual curves weighted to equal total mass) plus shrink 0.25 |
| `MIX6meanPC` | gen15 §7 component means with the pooled covariance, now under the deployed route |

**MAE endpoint.**  Registered contrasts at k = 1, 2, 3: `LW_vs_POOLED`, `LOWRANK2_vs_POOLED`,
`LOWRANK3_vs_POOLED`, `HIER_vs_POOLED`, `MIX6meanPC_vs_POOLED`; P1, five designs.  Cheapest
competitor: `NAIVE_LINE@doptk{k}`.

**Calibration endpoint (registered).**  For every arm and k: coverage of nominal 50/80/90/95 %
intervals from the BLUP posterior sd (plus measurement noise) on the scored pairs, mean sd
(sharpness), and `|cover90 − 0.90|`.  An arm is *better calibrated* than `POOLED` if its
`|cover90 − 0.90|` is smaller at every k ≥ 1 in all five designs and its 90 % coverage lies in
[0.85, 0.95].  Zero-shot: the `G14` logistic's P(heavy) on well-determined held-out cells —
Brier score and ECE (10 equal-count bins) for the raw probability, for an inner-fold Platt
recalibration, and for the constant training base rate; chemotype-blocked CIs.  The calibrated
probability is a *deliverable* if its Brier score beats the constant base rate with a CI
excluding zero in all five designs and its ECE ≤ 0.10.

**Stopping rule.**  These five covariance arms only; no thirteenth magnitude prior; no
per-component covariance arm (already measured dead in gen15 §7).

### L6 — cohort audit (Phase 0, already run)

Reported from `results/L6_cohort_audit/L6_COHORT_AUDIT.md`.  No hypothesis test.  Outcome known
at freeze time and recorded here because it fixes L4's pool: no single relaxation of any gen13
cohort rule adds a chemotype (maximum +0 against the bar of 5); the 100 excluded bundle
extractants are single-lanthanide compounds carrying 53 chemotypes absent from the cohort.  No
expanded cohort exists to run, so no additional cohort runs are registered.

## 4. Confirmation

At the end of discovery the orchestrator freezes ≤ 5 claims in `CONFIRMATION.md` — each a
registered contrast from §3 that passed P1 in all five designs on the discovery seeds and was
not removed by its two refuters — and runs `scripts/g16_confirm.py` **once** with the withheld
seeds.  The table reports, per claim: discovery point and CI, confirmation point and CI, raw p,
BH-adjusted p over the family it came from, family size, seeds agreeing, LOCO stability, and
whether the claim shrank, held or reversed.  A claim is **confirmed** only if the confirmation
run passes P1 in all five designs.

## 5. What will not be done

No design-B selection.  No result quoted against `MEAN_CURVE`.  No subset chosen after a
number is seen.  No arm promoted from the exploratory family to the registered family.  No
change to the cohort, the fold plans, the basis, the metrics or the bootstrap.  No
dependency installed or upgraded; if any dependency moves an anchor it is reverted.  No
cluster submission by the fleet.  No sixth confirmation claim.

## 6. Comparison accounting

`DECISION_REPORT.md` prints: number of arms and of contrasts per lead and in total (rows of the
`contrasts_*.csv` files), the registered/exploratory split, and the BH-adjusted p beside every
raw p.  A nominal p = 0.03 in an exploratory family of hundreds is reported as noise.
---
SHA-256 of every byte above the preceding '---' line: d004c30388078ae97232af2537a1915360360a7cb9eda07ebe4076764c48f50e
