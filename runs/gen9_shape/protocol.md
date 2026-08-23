# gen9 pre-registration — shape-preserving training and a learned first experiment

*Written 2026-08-21, **after** Phase 0 (gen8 reproduction) passed and **before** any
gen9 arm was fitted or scored on a held-out chemotype. Everything decided after an
outer-test number was inspected is labelled EXPLORATORY in the results and is never
quoted as a confirmed effect.*

---

## 0. What is frozen and cannot move

| item | value | why |
|---|---|---|
| cohort | `runs/gen7_architecture/cache/cohort.parquet`, 5,248 rows / 152 extractants / 131 ECFP clusters / 79 Tanimoto chemotypes, fingerprint `bed178ec1a7a82b0` | identical to gen6, gen7, gen8 — every gen9 number is on byte-identical rows |
| fold plan | `seeded_group_kfold` over `tanimoto_cluster`, 5 folds, seeds 104729 / 130363 / 155921 / 196613 / 262147 | gen5's algorithm, reused byte-for-byte |
| model seed | `42 + fold*1009 + 9_999_991`, independent of the split seed | gen7 harness contract |
| design matrix | `METAL + COND + ECFP + MASSACTION + RECOVERED`, median imputation, missingness indicators, cluster-balanced weights, prediction clipped to 1.5× the training range | the frozen arm's preprocessing, transcribed in `gen9.train.build_design` |
| curve geometry | gen8's `curve_membership`, **recomputed independently** by `scripts/gen9_curves.py` and asserted identical (7,207 memberships, 1,176 curves, max abscissa delta 0.0) | gen9's objective and gen8's metrics must be defined on the same curves |
| few-shot protocol | gen8's P2: pool/evaluation split a pure function of `(seed, repeat, ligand, n_rows)`, pool cap 48, 12 repeats × 5 seeds, eligibility `n_rows ≥ 4` | keeps gen9 paired row-for-row with gen8 |
| metric | one ligand, one vote (macro), gen6 offset/shape decomposition | unchanged since gen6 |
| inference | paired bootstrap, 5,000 replicates, resampling the **Tanimoto chemotype**, BCa where the statistic supports it, positive = candidate better | gen8's `inference.paired_chemotype_bootstrap`, unchanged |
| primary cohort | `FROZEN` (every row kept) | longitudinal comparability with gen7/gen8; `QUARANTINED` and `CORRECTED` are sensitivity only |

### Phase 0 result (already obtained, before this document)

All 15 gen8 reference checks reproduce **exactly** in the current environment
(`runs/gen9_shape/reproduction/reproduction.json`), with zero library drift against
gen8's stamped `environment.json`:

| check | gen8 | gen9 rerun |
|---|---|---|
| zero-shot, common cohort | 1.0605 | 1.0605 |
| random 1-shot | 0.7927 | 0.7927 |
| CENTRAL 1-shot | 0.6964 | 0.6964 |
| MEDOID 1-shot | 0.6916 | 0.6916 |
| oracle 1-shot | 0.5326 | 0.5326 |
| slope-repair + MEDOID 1-shot | 0.6752 | 0.6752 |
| best 2-shot | 0.5879 | 0.5879 |
| best 3-shot | 0.5230 | 0.5230 |
| best 5-shot | 0.4743 | 0.4743 |
| extractant slope, true / predicted median | 2.574 / 0.116 | 2.574 / 0.116 |
| acid slope, true / predicted median | 1.656 / 0.355 | 1.656 / 0.355 |
| extractant span ratio | 0.051 | 0.051 |

---

## 1. The two questions

**Primary.** Can held-out-chemotype error be reduced by training the global
predictor to preserve the measured shape of response surfaces, instead of repairing
flattened curves after prediction?

**Secondary.** Can the one-shot oracle's rule — `argmin |r_i − median(r)|` — be
approximated from observables, rather than proxied by geometric centrality?

---

## 2. GEN9-A — the shape objective

### 2.1 The loss

For rows `i`, `j` on the same reconstructed one-axis curve:

```text
L_total = L_row + lambda_delta * L_delta + lambda_span * L_span

L_row   = Huber( f(x_i) - y_i ; 1.0 )                      cluster-balanced
L_delta = Huber( (f(x_j)-f(x_i)) - (y_j-y_i) ; 0.5 )       per-curve balanced
L_span  = Huber( (max f - min f) - (max y - min y) ; 0.5 ) per curve
```

Both curve terms are invariant to a per-curve constant. That is the point: the
level is what one measurement supplies and what gen7/gen8 showed is not predictable
from structure, so the objective must not spend capacity on it.

Huber transition points (1.0 row, 0.5 delta, 0.5 span) are fixed here and not tuned.

### 2.2 The learner, and why the control is the frozen model

`CurveBoost` boosts small `ExtraTreesRegressor` forests on the functional gradient,
fitting each stage to the pseudo-residual `z = -g/w` with `sample_weight = w`.
**One stage at `learning_rate = 1` with a squared row loss and no curve terms is
algebraically `REC_ecfp_plus_recovered`** — asserted in
`tests/test_gen9_curve_objective.py::test_one_stage_row_only_is_algebraically_the_frozen_extratrees`.
The gen9 arm family therefore contains gen8's frozen model as an exact special case.

Consequently every shape claim is quoted against **`A0_ROW_ONLY`** — the same
learner, the same hyperparameters, the same seeds, `lambda = 0`. A gain quoted
against the frozen ExtraTrees would confound "new objective" with "new learner", and
that is the one confound this experiment cannot afford. The frozen arm is carried
as the *longitudinal anchor* and reported beside it.

Hyperparameters are selected once by inner `seeded_group_kfold` over chemotypes
**inside outer training partitions only** (`scripts/gen9_tune.py`), on the
`ROW_ONLY` objective, and then frozen for every arm. Tuning on a shape arm would
hand the shape arms a learner advantage the control never had.

### 2.3 Supervised axes

Default `LHM` = extractant concentration (log10), acid concentration (log10), and
the lanthanide series. Deliberately the same three axes gen8's post-hoc repair used,
so "intrinsic" and "post-hoc" are asked about the same physics.

* the lanthanide axis is supervised as **order and shape only** — no monotonicity is
  forced, because the response genuinely turns over;
* `contact_time` is **excluded** (8 curves, median slope 1e-4; gen8 shows the axis is
  flat noise after equilibrium and supervising it would teach measurement scatter);
* `temperature` (35 curves / 6 ligands) and `metal_concentration` (10 curves) are
  excluded from the default and available only as a declared variant.

### 2.4 Arms

**Phase 1 — sampler screen** (3 seeds, `lambda_delta = 0.5`, `lambda_span = 0`):
`A0_ROW_ONLY`, `A1_ROW_ADJACENT`, `A2_ROW_ENDPOINT`, `A3_ROW_RANDOM_PAIR`,
`A4_ROW_MULTISCALE`. Judged on slope and span, **not** on macro MAE.
`A4_ROW_MULTISCALE` is the pre-declared primary candidate.

**Phase 2 — weight grid** for the surviving sampler(s), still on the screening
seeds, `lambda_delta ∈ {0.05, 0.1, 0.25, 0.5, 1.0}`, `lambda_span ∈ {0, 0.05, 0.1,
0.25}`. **Development, labelled as such.**

**Phase 3 — locked finalists**: one or two arms at five seeds, plus `A0_ROW_ONLY`
and the frozen anchor.

No curve may dominate by length: every sampler emits a bounded number of pairs per
curve and weights are renormalised so each curve carries equal total weight
(`balance="ligand"` is a declared variant). The effective per-curve and per-ligand
contribution is written to disk for every fold.

### 2.5 Leakage constraint

A curve pair may enter a training loss only if **both** endpoints are in that fold's
training partition. Enforced structurally (`build_pairs` is only ever given training
row ids), asserted in `CurveBoost.fit`, audited per fold
(`curve_boundary_audit.csv`), and tested. All empirical priors are fold-local.

---

## 3. GEN9-A endpoints and pre-registered thresholds

Reported per curve type (extractant / acid / lanthanide) and on the
`RESIDUAL_SHAPE` stratum:

1. row-level macro MAE; 2. shape MAE; 3. within-curve Spearman; 4. within-curve sign
accuracy; 5. true vs predicted slope; 6. slope MAE; 7. span recovery
(`predicted_span / true_span`, headline restricted to curves with `span_true ≥ 0.5`);
8. dynamic-range compression (`sd_pred / sd_true` per ligand).

**Minimum useful shape result**, on extractant titrations, all four required:

* slope MAE reduced against `A0_ROW_ONLY` with a 95 % interval excluding zero;
* median predicted/true slope ratio materially above gen8's ≈ 0.05;
* shape MAE reduced, interval excluding zero;
* no catastrophic worsening of macro MAE (declared as: not worse than
  `A0_ROW_ONLY` by more than 0.10 macro MAE).

**Strong result:** median extractant span recovery ≥ 0.5 (against gen8's 0.051).

**Explicitly not a success:** a macro-MAE change of ~0.01 with unmoved shape
metrics. The experiment succeeds only if the response surface becomes more realistic.

**Coherence requirement (follow-up §19).** A flattening claim requires slope MAE,
span recovery, shape MAE, within-curve Spearman and sign accuracy to move
*together*. Any case where one improves while another degrades is reported as such
and not summarised as success.

**Over-steepening guard (follow-up §21).** The fraction of curves whose predicted
slope magnitude exceeds the corpus's own 95th percentile of *measured* slopes on
that axis is reported for every arm. No post-hoc clipping is applied.

---

## 4. RESIDUAL_SHAPE — and the regression-to-the-mean control

gen8's four ligand classes (`ALREADY_GOOD` 82, `PURE_LEVEL` 25, `PARTIAL_LEVEL` 7,
`RESIDUAL_SHAPE` 29) are used as a **locked stratification**, joined and never
recomputed from gen9 predictions.

They were read off gen8's own held-out errors, so `RESIDUAL_SHAPE` is selected on
*high gen8 error*. Any different model — including a differently-seeded copy of the
same one — will look better on that stratum through regression to the mean alone.
The control is therefore built in: `A0_ROW_ONLY` is a different model with the same
objective, so **the gen9 shape effect on `RESIDUAL_SHAPE` is
`A4 − A0_ROW_ONLY`**, and `A0_ROW_ONLY − FROZEN_ET` on the same stratum measures how
much of any apparent gain is regression to the mean. Both are reported.

**Pre-registered target:** ≥ 0.08 macro-MAE reduction on `RESIDUAL_SHAPE` against
`A0_ROW_ONLY`, *plus* improvement in the shape/span metrics on the same ligands. An
overall improvement driven by `PURE_LEVEL` while `RESIDUAL_SHAPE` is unmoved is
**not** the gen9 success condition.

---

## 5. Intrinsic training vs gen8's post-hoc repair (mandatory, follow-up §11)

Four arms, on identical rows:

1. old global model;
2. old global model + gen8 post-hoc slope repair (`SLOPE_L_s1`);
3. new shape-trained model;
4. new shape-trained model + the same post-hoc repair.

Questions: does (3) reach (2)? does the repair still add on top of (3)? are they
additive or redundant? does (3) generalise beyond the 26 ligands where gen8's repair
could be applied at all?

---

## 6. GEN9-B — learning the first experiment

### 6.1 The label, and a difference from gen8

`q_i = |r_i − median(r)|` over the candidate-and-evaluation population, with
`r_i = y_i − ŷ_i`, plus the realised one-shot MAE `mean_e |r_e − r_i|` and its
regret.

gen8 verified `argmin q == argmin realised` to 6.7e-16 — **in its exhaustive
protocol, where candidates and scored rows are the same points.** gen9 selects from
a pool and scores on a disjoint evaluation set, and that proof does not transfer.
The identity is therefore *measured* under gen9's exact protocol and reported before
any model is trained on it; on synthetic blocks it is exact about half the time with
a median regret of 0.002 and rank correlation ≈ 0.91. Because of this, **two labels
are carried and compared** (`oracle_deviation` and `realised_mae`) rather than one
being assumed sufficient.

### 6.2 Learners and features

`B1` scalar regression on `q`; `B2` pairwise logistic ranking — preferred a priori,
because the downstream task needs only an ordering and `q`'s per-block scale varies
by an order of magnitude between ligands.

Features are geometry and response-surface context, all pool-scoped and
pre-measurement, declared per column in `FEATURE_PROVENANCE` and machine-checked:
axis percentiles and centralities, boundary distance, medoid and centre distance,
pool nearest-neighbour distances, RBF leave-one-out design variance, the model's own
prediction and its pool percentile, its predicted curve slope / span / position /
local gradient, curve type, and pool-size context. Molecular descriptors enter only
as an explicit ablation, with a **shuffled-descriptor control**.

### 6.3 Baselines and the comparison that counts

`RANDOM`, `CENTRAL`, `MEDOID`, `MEDIAN_PREDICTION`, `MID_ACID`,
`FARTHEST_FROM_EXISTING`, `MAX_CONDITION_COVERAGE`, `MIN_GP_DESIGN_VAR`,
`MIN_PREDICTION`, `MAX_PREDICTION`, `MAX_ENSEMBLE_SD`, `MIN_ENSEMBLE_SD`, `ORACLE`.

**The primary comparison is `LEARNED vs MEDOID/CENTRAL`**, not learned vs random.

Reported: one-shot macro MAE, regret to oracle, fraction of the oracle gap
recovered, rank percentile of the selected candidate, fraction of ligands harmed
relative to zero-shot, and all of these by curve type and failure class.

**Pre-registered target:** recover ≥ 30 % of the remaining MEDOID→oracle gap
(0.692 → 0.533, i.e. 0.159), which means one-shot MAE ≤ ≈ 0.644, with a 95 %
interval on `LEARNED − MEDOID` excluding zero and ≥ 4 of 5 seeds agreeing in sign.
This is a target, not a guarantee of success.

### 6.4 Leakage and stress

One model per (split seed, fold), trained only on blocks from that seed's *other*
folds — the fold plan is chemotype-blocked, so a held-out ligand's whole chemotype
is absent. Scaling is fitted on training folds only. Tests: corrupting every
candidate's target must not move the selection; permuting or truncating the
evaluation indices must not move it either.

Pool stress (follow-up §17): half pool, medoid removed, high-acid only,
low-extractant only, metal subset, sparse pool. A policy that collapses when the
exact medoid is absent is not a policy.

---

## 7. GEN9-C — series-local hierarchical adaptation

```text
correction(row) = mu_ligand + delta_series[s(row)]
                + c_metal*z_metal + c_acid*z_acid + c_extr*z_extr
```

`mu_ligand` unpenalised; series deviations and response coefficients shrunk toward
zero with **MAP penalties `sigma²/tau²` estimated fold-locally** from the fold's own
training ligands and their out-of-fold residuals. k-dependence needs no schedule:
with one measurement the slope columns carry no leverage and stay at the prior.

Series identity is a *condition* label (acid, diluent, additive), known before any
measurement — conditioning on it is not leakage, it is gen8's one failed
falsification condition made operational.

Pre-registered behaviour, tested: at k = 1 the correction is essentially a level
shift; it must nest `OFFSET_K1` exactly in the infinite-penalty limit; it must never
produce the 1e12-scale coefficients gen7 lost an arm to. Ablations
`SERIES_MAP_NOSERIES` and `SERIES_MAP_LEVELONLY` separate "series-local" from
"better-shrunk".

First-point and later-point acquisition stay different: `k=1` learned
oracle-imitation, `k≥2` spread/leverage.

---

## 8. Attribution chain (follow-up §22)

Run and reported in this order, never collapsed to the best final number:

```text
OLD GLOBAL + CENTRAL      + OLD ADAPTER
NEW GLOBAL + CENTRAL      + OLD ADAPTER
OLD GLOBAL + LEARNED ACQ  + OLD ADAPTER
NEW GLOBAL + LEARNED ACQ  + OLD ADAPTER
NEW GLOBAL + LEARNED ACQ  + NEW SERIES ADAPTER
```

k = 0, 1, 2, 3, 5 on both the maximal-coverage population and the common
longitudinal cohort, with the cohort always named. gen8 common-cohort references:
1.061 / 0.667 / 0.588 / 0.523 / 0.474.

Goals, not assumptions: k=1 ≈ 0.60–0.62, k=2 ≤ 0.50. **The stronger requirement is
that any improvement arrives with better shape metrics rather than only better level
calibration.**

---

## 9. Falsification conditions

gen9's central hypothesis fails if:

1. no curve loss materially changes predicted slope or span on the extractant axis;
2. shape improves but macro MAE degrades catastrophically (> 0.10 vs `A0_ROW_ONLY`);
3. shape metrics move incoherently (slope up, Spearman down);
4. flattening is replaced by over-steepening;
5. `RESIDUAL_SHAPE` is unmoved while `PURE_LEVEL` carries the whole gain;
6. the learned acquisition policy does not beat `MEDOID`/`CENTRAL`;
7. the learned policy collapses under pool perturbation;
8. shuffled molecular descriptors perform as well as real ones and a descriptor gain
   was nonetheless claimed;
9. any headline reverses between `FROZEN`, `QUARANTINED` and `CORRECTED`;
10. any result depends on a leaked target or an unpaired comparison.

Each will be stated plainly if it holds. If none of the success conditions is met,
gen9 reports failure cleanly and does not go looking for an unrelated model sweep.

---

## 10. What gen9 will not do (brief §25)

No learner sweep, no fingerprint sweep, no pretrained-embedding sweep, no
absolute-value CNP, no absolute-value physics law, no larger ligand hypernetwork, no
uncertainty-driven one-shot acquisition, and no globally forced
smoothness/monotonicity across every condition axis. gen8 produced strong negative
evidence for all of these.

---

## 11. Statistical reporting

Every headline carries: point estimate, 95 % chemotype-blocked paired bootstrap CI,
BCa where the pipeline supports it, number of ligands improved, seed consistency,
population size, exact cohort and protocol, and whether it was **pre-registered**,
**development**, or **exploratory**. 5,000 replicates for final tables. Positive =
candidate better, everywhere.

Pairing is audited to a file (`pairing_audit.csv`): two arms called paired must share
fold, ligand, chemotype, repeat, candidate pool size and evaluation rows. Row
accounting is audited to a file: expected / returned / scored / missing / NaN / inf
per arm per seed. An arm may not look better because difficult rows disappeared.

---

## Addendum, 2026-08-21 — decisions taken before the finalist evaluation

Recorded here rather than in the results so the reader can see when each was made.

**1. The k = 1 collapse is algebraic, not empirical.** A ridge with an unpenalised
intercept, fitted on a *single* observation, has the exact solution
`β = [r, 0, …, 0]`: substituting it into `(dd' + diag(0, λ…))β = d r` gives
`d(1·r + z·0) = d r`, so every penalised coefficient is exactly zero. Therefore
`OFFSET_K1`, `OFFSET_K2`, `OFFSET_K3` and the gen9 hierarchy are **identical at
k = 1 by construction**, not by shrinkage happening to be strong enough. Verified to
machine precision on 286 held-out units (max |Δ| = 0.0 for K2/K3, 6.7 × 10⁻¹⁶ for the
hierarchy, whose series columns are centred rather than exactly zero at the selected
row). Consequence: gen9-C can only be tested at k ≥ 2, and any k = 1 difference
between these adapters would be a bug.

**2. B2's ranking objective is top-focused and difference-weighted.** Both follow
from the deployed rule being an `argmin`:

* every training pair has at least one member in its block's best third, because
  being right about the ordering of two mediocre candidates is worth nothing;
* pairs are weighted by `|q_a − q_b|`, because a pair separated by 0.001 is a coin
  flip whichever way it is labelled.

Declared before the five-seed evaluation. A single-seed, two-repeat smoke run had
been executed by this point and is labelled **development** wherever it appears; no
finalist number is drawn from it.

**3. B3 `LEARNED_BLEND` is added, and it nests the baseline.** Score is
`α·rank(MEDOID distance) + (1 − α)·rank(learned score)`, with `α` chosen from
{0, 0.25, 0.5, 0.75, 1} by realised one-shot MAE on the fold's **training** blocks
only. At `α = 1` it *is* `MEDOID`. This exists because B1 and B2 can each fail by
learning a combination that generalises worse than the one rule gen8 validated, and
that failure mode tells us nothing; the blend turns the question into "does any
departure from MEDOID help", which is answerable. The selected `α` is reported per
fold — `α = 1` everywhere would itself be the finding that nothing beats MEDOID.

**4. Sensitivity cohorts are applied at evaluation time, not by refitting.**
`QUARANTINED` drops the corrupt copy's rows from scoring; `CORRECTED` rescales them.
The global model is not refitted per cohort, because that would confound "do these
rows matter" with "what does a differently-trained model do". For the k-shot table,
where the detail frame is already aggregated past the row level, the whole affected
ligand is excluded instead — a stricter test than the row-level edit.

**5. Two further data findings, from gen9's own scan, recorded before the results.**
* 353 rows (6.7 % of the cohort, 17 extractants) carry a recorded extractant name
  that does not match the structure they are modelled as — gen7's "unmodelled second
  species", independently rediscovered. The model is being asked to predict a
  synergistic mixture from one component's fingerprint on those rows.
* Beyond DMDPhPDA's 59 exact-decade duplicate pairs, **two further single-cell
  exact-decade pairs** appear in other ligands. Flagged, not deleted.

---

## Addendum 2, 2026-08-21 — a prediction recorded before the k-shot frontier runs

The phase-1 screen and the single-seed loss-weight probe are complete; the k-shot
evaluation of the gen9 arms has **not** been run. What follows is written now so it
can be checked against the result rather than fitted to it.

**What phase 1 established.** Within-curve supervision significantly reduces
dynamic-range compression on all three supervised axes (extractant span recovery
+0.093, **25 of 25 ligands**, 3/3 seeds) and does not restore the slope (median
predicted extractant slope 0.115 → 0.134 against a measured 2.574). The single-seed
dose–response shows the range *is* recoverable — `lambda_delta = 8` reaches span
recovery 0.756, past the pre-registered "strong" bar of 0.5 — at a macro-MAE cost of
+1.17, an order of magnitude past the pre-registered catastrophe bar of +0.10.

**The prediction.** gen9's central hypothesis is that a rough global surface with the
right *shape* is worth more than one with the right *level*, because one measurement
supplies a level and no measurement supplies a shape. If that is right, an arm that
loses at k = 0 on level error should recover at k ≥ 1 once offset calibration removes
it.

The quantity that decides this is already measurable and already known: **per-curve
shape MAE, the error that survives a perfect per-curve offset.** On extractant
titrations it is 0.663 for the control, 0.665 (A1), 0.689 (A4), 0.732 (A2) — *worse*
for every phase-1 shape arm — and 0.637 for the single-seed `lambda_delta = 8`,
cluster-balanced arm, the only configuration to improve it.

So the prediction is:

1. every phase-1 shape arm (`A1`, `A4`) will **still lose** to the control at k = 1,
   2 and 5, because their extra error is not level error;
2. `X_MULTI_d8c` is the only arm with a mechanism to win after calibration, and if it
   does, the win should appear **on ligands with extractant titrations** and not on
   the cohort as a whole;
3. if (2) fails as well, gen9-A's conclusion is a clean negative: within-curve
   supervision reduces compression but the compression was not what was costing the
   accuracy, and gen8's mean-preserving post-hoc repair remains the only tool that
   improves shape without paying for it.

Recording (3) in advance matters because it is the outcome the study is most likely
to reach, and a negative that was predicted is worth more than one that is explained
afterwards.

---

## Addendum 3, 2026-08-21 — the exploratory arm, and what would falsify it

Written after the pre-registered sweep and the loss-weight probe were read, and
**before** the k-shot frontier, the acquisition study or the sensitivity cohorts were
run on it. Everything in it is labelled EXPLORATORY wherever it appears.

**Why it exists.** The pre-registered loss failed and the dose–response said the
failure was not a tuning problem. The obvious follow-up — delete the level and see
whether the shape becomes learnable — was run as a probe: a model trained on the
curve-centred target `y − mean(y over the curve)`, with nothing to compete against,
recovers 2.5 % of the extractant range. That rules out competition and points at the
coordinate system: the centred response is not a function of the absolute conditions,
because different titrations span different windows.

**The two arms.**

`GEN9_REL_MONOLITH` — the frozen arm's design plus five columns, nothing else
changed: position within the row's own curve window (0 to 1), signed offset from the
window centre, window width, point count, and an is-endpoint flag.

`GEN9_SHAPE_RECOMPOSED` — the same features plus a **mean-preserving** recomposition:
fit the monolith; fit a second model of the same family on the curve-centred target
with the relative columns appended; predict `mean_over_curve(monolith) + centred
shape`, re-centred so the curve's predicted level is untouched. Structurally the same
move gen8's post-hoc repair makes, with the median-slope prior replaced by a model.

**Why this is not leakage, stated so it can be checked rather than trusted.** The five
columns read the *conditions* of the held-out ligand's own rows and no target of any
kind. The gen7 harness hands a contender the test **frame** by design — features only,
target column dropped and asserted absent — because a transductive method legitimately
needs the candidate conditions; the gen8 deployment protocol assumes the same, since
the user supplies the list of conditions they are choosing between. It is the same
information gen9's acquisition features already use as pool percentiles, and the same
information gen8's post-hoc repair uses when it fits a curve. `fit_predict` is run
twice with every held-out target replaced by values 10⁴ away and the output must be
bit-identical (`test_the_recomposed_arm_is_mean_preserving_and_leak_free`), and the
same test asserts the per-curve mean is unchanged.

**What would falsify it, recorded before the remaining runs.**

1. **It is really gen8's repair.** If `frozen + SLOPE_L_s1` matches the recomposition
   on the extractant axis, the advance is "learned instead of fixed" and should be
   described that way, not as a new capability. The head-to-head is in the k-shot run,
   where both are adapters on the same rows.
2. **It does not survive calibration.** If the shape gain vanishes at k = 1 — where
   offset correction removes exactly the level the recomposition preserves — then the
   improvement was level-shaped after all.
3. **It is concentrated.** If the extractant gain comes from a handful of ligands, one
   publication, or the ligands with the widest windows, it is a property of those rows
   and not of the method. Leave-one-chemotype-out and per-publication sensitivity are
   run for exactly this.
4. **It is an artefact of the window.** `rel__position` is a normalised abscissa, so a
   curve measured at four points spanning three decades and one measured at four
   points spanning half a decade get identical position values. If the gain is
   restricted to wide-window curves, the feature is encoding window width rather than
   response shape, and `rel__window_width` — which is carried separately — should
   absorb it.
5. **It disagrees between cohorts.** Reported on `FROZEN`, `QUARANTINED` and
   `CORRECTED` like everything else.
