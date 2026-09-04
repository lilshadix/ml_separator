# Methods text for Figure 1

Written to be dropped into a Methods section with light edits. Every definition below is
implemented in the modules named in parentheses; nothing here is a description of intent.

---

## 1. Prediction target

The target is `log_D`, the base-10 logarithm of the distribution ratio of a single
lanthanide between an aqueous and an organic phase, for one row of the corpus. A row is a
unique combination of extractant (canonical SMILES), lanthanide, and a fully specified set
of experimental conditions — acid identity and concentration, extractant concentration,
diluent, metal concentration, temperature and contact time. Where the corpus contains
replicate measurements of the same cell, the target is the mean of the replicates and the
replicate count is carried alongside. The corpus used throughout is 5,248 rows covering 152
extractants, 14 lanthanides (promethium has no stable isotope and does not appear) and
2,055 distinct condition cells; the extractants fall into 131 exact-ECFP clusters and 79
Tanimoto-0.7 chemotypes.

### 1.1 Provenance

The measurements are compiled from the primary literature in a separate dataset-builder
repository, which exports one immutable `*_SAFE.csv` per metal. Each row of the modelling
corpus carries a `safe_exp_id` of the form `{stem}_SAFE:{exp_id}` and joins back to those
exports for **5,992 of 5,992 rows** (`runs/gen6_provenance/summary.json`,
`upstream_join_fraction = 1.0`), which recovers a DOI, an entering author and an addition
date for every measurement. **105 distinct publications** contribute, with a median of 26
rows and 1 extractant each and a maximum of 459 rows and 58 extractants; the SAFE database's
own self-citation is excluded from the publication set.

This matters for two claims. First, the condition cell that the corpus averages over
crosses a publication boundary in well under 1 % of rows, so an averaged cell is almost
always one laboratory's series. Second, the publication identifier makes an alternative
resampling block available, and it is the one to quote for the curve-shape result: only 8
Tanimoto-0.7 chemotypes carry extractant titrations, so a chemotype-blocked interval is the
*narrowest* available, whereas 32 publications carry them and the publication-blocked
interval is about 40 % wider (`runs/gen10_final/publication_sensitivity/`).

## 2. Grouping unit and held-out protocol

The unit that is held out is the **Tanimoto-0.7 chemotype**: extractants are agglomerated
so that no extractant in one group has ECFP Tanimoto similarity ≥ 0.7 to an extractant in
another, and whole groups are assigned to folds. A held-out extractant therefore has no
close analogue anywhere in the training set — a stricter condition than leave-one-extractant
-out, which would leave near-duplicates in training. Five outer folds are used, repeated
over five split seeds `{104729, 130363, 155921, 196613, 262147}`. The model seed is fixed
at 42 and is independent of the split seed, so seed-to-seed variation reflects the data
split and not the learner's randomness. Fold assignment is audited per run: no titration
curve straddles a fold boundary, and every held-out partition is closed under the chemotype
relation.

Aggregation follows the grouping. The headline metric is **macro MAE**: the mean absolute
error is first averaged within an experimental unit and then across units with equal
weight. Two unit choices are used and are never mixed within a panel — one vote per ECFP
cluster (131 units) for whole-corpus model comparisons, and one vote per extractant for the
few-shot results, where the extractant is the deployment unit. Row-weighted (pooled) MAE is
reported alongside but is not the primary metric, because one extractant supplies 28 % of
the rows and one chemotype 64 % of them.

## 3. Global model

The global model is fitted on training chemotypes only and consists of two `ExtraTrees`
regressors (400 trees, `max_features = 0.30`, `min_samples_leaf = 2`), with training rows
weighted so that each ECFP cluster contributes equally (`gen9/train.py`).

*Forest 1* predicts `log_D` from the frozen design: an ECFP fingerprint block, 2-D
molecular descriptors, recovered solvent and system variables, the lanthanide index and
physical properties, the raw condition columns, and a mass-action block of log-concentration
terms derived from the extraction equilibrium.

*Forest 2* predicts the **curve-centred** target — `log_D` minus the mean of `log_D` over
the row's own curve, computed from training rows only — from the same design plus five
relative-position columns (`gen9/relative.py`): the row's position in its curve's
measurement window scaled to [0, 1], its signed distance from the window centre in axis
units, the window width, the number of points in the curve, and an endpoint indicator.

A *curve* is a maximal set of rows of one extractant that differ in exactly one condition
axis with everything else fixed — an extractant titration, an acid titration, a lanthanide
series, a temperature series, and so on (`gen9/curves.py`). A row can lie on more than one
curve; the longest is used, ties broken deterministically by curve identifier. The corpus
contains 1,176 such curves and 7,207 row-curve memberships.

*Recomposition.* For each requested curve the final prediction is
`mean_over_curve(forest 1) + (forest 2 shape, itself centred over the same curve)`; rows
that lie on no curve keep forest 1's prediction unchanged. The construction is
**mean-preserving**: the mean prediction over each curve, and hence over each extractant,
is exactly forest 1's. This is verifiable rather than asserted — the offset MAE of the
recomposed model is bit-identical to the baseline's (0.821172 on the full cohort;
0.84676720 on the 99-extractant cohort, in every one of the five seeds).

Nothing in this model reads a held-out target. The relative-position columns are functions
of the *candidate condition list*, which the deployment protocol assumes the user supplies;
the curve means used in training are training means; the curve mean used at prediction time
is a mean of predictions. A regression test corrupts every held-out target and requires
bit-identical output.

## 4. Zero-shot setting

Zero-shot (`k = 0`) means the fitted global model is asked for a prediction on every row of
a held-out extractant, with no measurement of that extractant available. This is the
setting of the model-comparison results.

## 5. k-shot setting: support points, query points, acquisition, adaptation

The deployment unit is a prediction *plus* a small number of requested measurements
(`gen8/kshot.py`, `gen8/protocols.py`). For each held-out extractant, and independently for
each of 12 repeats per split seed:

1. **Pool/query split.** The extractant's rows are permuted with a generator seeded by
   `(split seed, repeat, blake2b hash of the extractant SMILES)` — a process-independent
   hash, so that two arms evaluated in different processes are paired row for row — and
   split into a **candidate pool** and a disjoint **evaluation (query) set** of
   `max(2, n/2)` rows. The pool is capped so that a very large extractant does not dominate.
2. **Acquisition.** A policy selects `k` rows from the pool, sequentially. Policies see the
   candidate rows' conditions, the model's prediction and, where applicable, its ensemble
   spread — never a candidate's target. The shipped policy is `CENTRAL_THEN_SPREAD`: the
   first point is the medoid of the standardised condition axes, and subsequent points are
   the farthest from those already selected. Oracle policies, which read held-out targets to
   choose, are evaluated as upper bounds and are labelled non-deployable wherever they
   appear.
3. **Measurement.** The targets of the `k` selected rows — and only those — become
   available. These are the **support points**.
4. **Adaptation.** The global model is *not* refitted. A calibrator adjusts its outputs
   using the support residuals. At `k = 1` the shipped adapter is a target-free slope repair
   followed by a level offset (`gen8/slope_restore.py`): the frozen model's own predictions
   for the extractant are regressed on the mass-action design, each fitted slope is compared
   with the median measured slope of the *training* curves on that axis, and the shrunken
   difference is added back along the axes the extractant actually varies in — a correction
   that uses no target at all — after which the single support residual sets the level. At
   `k ≥ 2` the shipped adapter is `SERIES_ML` (`gen10/adaptation.py`): a ridge-shrunk update
   of the intercept, a small set of response coefficients and series-local terms, with the
   shrinkage set by maximising the marginal likelihood of the *training* extractants'
   out-of-fold residuals under a hierarchical normal prior, so that no held-out extractant
   informs the penalty applied to it.
5. **Scoring.** MAE is computed on the evaluation rows only — the **query points** — and
   averaged first within an extractant over repeats and seeds, then across extractants.

`k ∈ {0, 1, 2, 3, 5}`. The few-shot cohort is the 99 extractants that have at least 5 pool
rows and at least 2 query rows in every arm at every `k`, which keeps every arm scored on
the same units. All arms share the same pool/query draws, so every comparison in the
few-shot figures is paired.

## 6. Why observing k points is calibration and not leakage

Three properties separate this from ordinary train/test contamination, and they should be
stated together.

*The information is bought, not found.* In the intended use the chemist has a candidate
extractant and a list of conditions they are considering and can run a small number of
those experiments. The support points model exactly that transaction. Nothing about the
protocol assumes access to measurements that would not exist at decision time.

*The support targets never enter the model.* The global model is fitted once on training
chemotypes and is frozen; the `k` targets are read only by a calibrator with a handful of
ridge-shrunk degrees of freedom. Refitting the model on the held-out extractant would be a
different — and, for a single extractant with five measurements, meaningless — experiment.

*A support row is never counted as an unseen prediction.* The pool and the evaluation set
are disjoint by construction and are drawn before any policy runs, so a policy cannot
select a row that will later be scored, and the reported error is never measured on a row
whose answer was supplied. This is the property that makes `k`-shot numbers comparable with
the `k = 0` number on the same axis, and it is the one that must be checked in any
reimplementation. The corresponding failure mode — scoring the selected rows, or letting
`RANDOM` be averaged over candidates while `ORACLE` takes the minimum over the same
candidates — is exactly what the pool/query protocol was introduced to remove.

Two limits follow from the same construction and belong in the Methods rather than a
footnote. First, `k` counts measurements on the *same* extractant; a support point measured
in one solvent system calibrates that system and largely not another, so the frontier is
series-local. Second, the reported frontier is an average over pool draws: because the
evaluation set is redrawn each repeat, an extractant with few rows contributes a noisier
per-extractant mean, which is why the cohort requires at least two query rows in every arm.

## 7. Metrics

* **macro MAE** — mean absolute error averaged within an experimental unit, then across
  units with equal weight; the unit is stated per figure.
* **offset (level) MAE** — the mean over extractants of the absolute mean residual of that
  extractant. What a per-extractant constant would remove.
* **shape MAE** — the mean over extractants of the mean absolute difference between the
  centred prediction and the centred measurement, extractants with fewer than a minimum row
  count excluded. What remains after the level.
* **slope** — ordinary least squares slope of `log_D` on the curve's varying axis, computed
  separately for measurements and predictions on the same curve.
* **span recovery** — the predicted range of `log_D` over a curve divided by the measured
  range; 1 is perfect, 0 is a flat line.
* **within-curve Spearman** — rank correlation between predicted and measured `log_D`
  within one curve.
* **uncertainty** — all intervals are chemotype-block bootstraps (5,000 replicates)
  resampling whole Tanimoto-0.7 chemotypes; paired comparisons use the repository's
  `paired_chemotype_bootstrap` with BCa correction and report, alongside the interval, the
  number of units improved and the number of split seeds in which the sign holds.
