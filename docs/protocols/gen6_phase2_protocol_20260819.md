# gen6 Phase 2 — pre-registered protocol for Experiments C and F (2026-08-19)

**Status: pre-registration, written after Phase 1 reported case A and before any Phase 2 model was
fitted.** Phase 1 ([results](gen6_phase0_and_phase1_results_20260819.md)) established that the
binding constraint on new-ligand prediction is chemical coverage, that the whole gain lands in the
per-ligand *level* offset, and that at equal row budget breadth beats depth. Case A of the decision
tree licenses two next steps: a hierarchical level model aimed at the offset (C), and a retrospective
study of *which ligand to measure next* (F). Nothing in this file is edited after the runs start;
corrections go in the results document as amendments.

Shared with Phase 1, unchanged: the `min_cells = 3` cohort (5,248 rows / 152 extractants / 131 ECFP
clusters / 79 chemotypes), the frozen all-190 chemistry map, the gen5 fold algorithm, the
`LevelRegressor` learner, macro MAE over ECFP clusters as primary, bootstrap over chemotype blocks
with percentile **and** BCa and cluster-robust intervals, and the offset/shape decomposition.

Facts measured before writing this, on which both designs lean:

* metal identity explains **8 %** of log D sum of squares within (ligand, condition) cells; the
  (ligand, condition) level explains 92 %; within-ligand variation (conditions + metals) is 49 %;
* 2,405 (ligand, condition) cells, of which only **521 hold ≥ 2 metals** (3,364 rows) — the only
  cells that carry any metal-response information;
* a start cohort of "the 10 most-measured ligands of the largest training chemotype" is ~2,900 rows
  in four of five chemotype folds and degenerates to 2 ligands / 244 rows in the fold that holds
  out the diglycolamide chemotype.

---

## Experiment C — hierarchical, physics-informed level decomposition

### Question

Represent `log D(l, m, c) ≈ α_l + F_cond(l, c) + F_metal(l, m, c) + ε` and ask not "is this
equation true" but **which component fails to transfer to a new ligand**. Phase 1 says the offset
does; C makes that attribution with models built to expose it, and asks whether structure — as
opposed to capacity — buys anything.

### Models (all fold-local; every preprocessing step fitted on the training fold)

| id | model | what it is for |
|---|---|---|
| `MONO_ET` | the Experiment A EXPANDED arm: ExtraTrees on METAL + COND + LIG2D_EXT + MASSACTION | the "same learner without decomposition" control; must reproduce Experiment A bit-for-bit on the chemotype folds |
| `MONO_RIDGE` | plain ridge, same features as C1, no ligand intercepts | the linear control for C1 |
| `C1_HIER_RIDGE` | partial-pooling ridge: fixed effects on {METAL, COND, PHYSCHEM, DONORS, MASSACTION, donor×metal, logL×descriptor, cond×metal} **plus a ridge-penalised random intercept per training ligand** that shrinks toward the descriptor prediction; for a held-out ligand the intercept is zero and the prediction is the descriptor prior | the mixed-effects baseline; its whole point is that α_l is *estimated* when the ligand is in training and *predicted from descriptors* when it is not |
| `C2_TWO_STAGE` | Stage A: ExtraTrees on ligand + condition features predicting the (ligand, condition) cell mean ȳ_{l,c}, one row per cell; Stage B: ExtraTrees on METAL + all features predicting the residual `y − Â_oof(l, c)`, where Â_oof is **cross-fitted** inside the training fold (5 inner folds grouped on chemotype); prediction = Â + B̂ | separates the level from the metal response so each can be scored |
| `C2_TRUECENTRE` | *amendment, 2026-08-19, after a one-seed preview and before the full run:* the same two stages, but Stage B trained on the **true** within-cell departure `y − ȳ_{l,c}` (the brief's own Part VI definition) on cells with ≥ 2 metals, instead of on the cross-fitted residual. The preview showed the cross-fitted residual under a chemotype hold-out carries Stage A's level error, which Stage B learns from in-sample ligand features and mis-applies (mean \|B̂\| 0.53 vs 0.15; seed-104729 macro MAE 1.25 vs 1.04, MONO_ET 1.08). Because it was chosen after seeing one seed, any C2_TRUECENTRE-vs-MONO_ET result is **exploratory-confirmed at best**, and is labelled so | the deployable two-stage model as the brief defined it |
| `C2_ORACLE_*` | the same two stages with one replaced by its true value on the test fold: `ORACLE_LEVEL` = true ȳ_{l,c} + B̂_truecentre; `ORACLE_METAL` = Â + (y − ȳ_{l,c}) | the attribution: how much error each component is responsible for |
| `C3_SHARED_RIDGE` | C1's design matrix stacked with within-cell pair-difference rows (x_A − x_B → y_A − y_B) for every training (l, c) cell with ≥ 2 metals, so the level and pair tasks are fitted by one linear score g | the shared-score form; pair predictions are derived **only** as g(A) − g(B), so antisymmetry and transitivity hold exactly (asserted, not assumed) |

Ridge penalties are chosen by inner grouped CV (3 folds on chemotype) over a small fixed grid;
the grid and the chosen values are recorded. Sample weights are the gen5 group-balanced weights on
ECFP cluster, as in Experiment A.

Regimes, on the shared cohort: `unseen_chemotype` (primary — Experiment A's folds, so `MONO_ET`
reproduces), `unseen_ligand` (ECFP-cluster folds), `unseen_series` (series folds; the ligand *is* in
training, so C1's random intercept is live). 5 split seeds × 5 folds.

### Derived pair evaluation (the C3 measurement)

For every level model, pair predictions on the condition-matched metal pairs **inside each test fold**
(1,000–8,500 per fold) are `ŷ(A) − ŷ(B)`; report pair MAE, sign accuracy, and the antisymmetry and
transitivity residuals (must be 0 to machine precision). Null: leave-fold-out **pair-label mean**
(the strongest honest alternative established in gen3/gen4). This is a measurement of how well the
level models do the project's original task; no verdict depends on it.

### Pre-registered hypotheses

| H | statement | pass condition |
|---|---|---|
| **C1** | The component that fails to transfer is the level, not the metal response | under `unseen_chemotype`, **on test cells with ≥ 2 metals** (a singleton cell makes the level oracle trivially perfect, so it cannot be used to score this), the error removed by the level oracle `MAE(C2) − MAE(ORACLE_LEVEL)` exceeds the error removed by the metal oracle `MAE(C2) − MAE(ORACLE_METAL)` with BCa CI95 low > 0 over chemotype blocks; the all-rows version is reported beside it |
| **C2** | Partial pooling helps exactly where the ligand is known | `C1_HIER_RIDGE` beats `MONO_RIDGE` under `unseen_series` (BCa CI95 low > 0) and the two are indistinguishable under `unseen_chemotype` (CI95 spans 0) |
| **C3** | Structure does not substitute for coverage | `C2_TWO_STAGE` does **not** beat `MONO_ET` under `unseen_chemotype` by more than 0.02 macro MAE with CI95 low > 0 — no pre-registered expectation that it does; if it does, that is the surprise worth reporting. *Amendment:* `C2_TRUECENTRE` vs `MONO_ET` is reported with the same rule but labelled exploratory, because the variant was chosen after a one-seed preview |
| **C4** | Level-derived pair predictions keep exact antisymmetry and transitivity | max |ŷ(A,B) + ŷ(B,A)| and max |ŷ(A,B) + ŷ(B,C) − ŷ(A,C)| below 1e-9 on every test fold |

What would falsify the generation's reading: a `C2` that beats `MONO_ET` materially on new chemistry
would say that model structure, not coverage, was the lever — contradicting Phase 1's case A.

---

## Experiment F — which ligand should have been measured next?

### Question

Starting from a deliberately narrow, deep training cohort — the situation this project was in for
five generations — and treating every other training ligand as an unlabelled pool, which acquisition
rule most quickly lowers error on chemistry the start cohort does not cover? The purpose is to
decide whether the next measurement should be another condition on a known ligand or the first few
measurements of a distant one — and, if the latter, *which* distant one.

### Design

* Folds: the Experiment A chemotype folds. **The test set is the fold's held-out chemotypes and is
  fixed**; it is never touched by any acquisition.
* Start cohort: the 10 most-measured training ligands of the largest training chemotype, all their
  rows. Narrow by construction; ~2,900 rows in four folds, 2 ligands in the fold that holds out the
  diglycolamides (disclosed, not excluded).
* Pool: every other training ligand. After a ligand is acquired, **only `b = 3` of its rows are
  revealed** (a random subset, seeded) — the "first few measurements" scenario; `b = all` is a
  disclosed sensitivity for two policies.
* Steps: up to 30 acquisitions; the model is refit after every acquisition for the policies that
  need it and scored on the fixed test set at checkpoints {1, 2, 3, 5, 8, 12, 16, 20, 25, 30}.
* Replicates: 3 split seeds × 5 folds × 2 acquisition replicates per policy.

### Policies (every one label-free for the candidate: the pool's `log_D` is never read)

| policy | rule |
|---|---|
| `random` | uniform over the pool — the required null |
| `maxmin` | the pool ligand farthest (min Tanimoto) from everything acquired so far |
| `uncertainty` | the pool ligand whose rows have the largest mean across-tree sd under the current model |
| `diversity_x_uncertainty` | rank-product of the `maxmin` distance and the `uncertainty` score |
| `offset_uncertainty` | the pool ligand whose **predicted ligand mean** has the largest across-tree sd — uncertainty about the level, which is the quantity Phase 1 says is missing |
| `same_chemotype_first` | the pool ligand *closest* to the acquired set — "another DGA analogue", what the field did |

### Endpoints, at every checkpoint, on the fixed test set

macro MAE over ECFP clusters; **hard-chemotype MAE** on test rows with nearest-neighbour Tanimoto to
the **start cohort** below 0.4 (a fixed reference, so the subset does not move with the policy);
**offset MAE**; shape MAE; worst-quartile ligand MAE; and coverage — chemotypes in training, and the
mean nearest-training-neighbour Tanimoto of the test ligands.

### Pre-registered hypotheses (paired over chemotype blocks at each checkpoint; BCa and
cluster-robust intervals beside the percentile)

| H | statement | pass condition |
|---|---|---|
| **F1** | Max-min diversity beats random acquisition on hard chemistry | `hard MAE(random) − hard MAE(maxmin)` CI95 low > 0 at ≥ half the checkpoints from 5 onwards |
| **F2** | Acquiring the most level-uncertain ligand beats random | same test for `offset_uncertainty` on **offset MAE** |
| **F3** | Buying more of the same chemistry is worse than random | `hard MAE(same_chemotype_first) − hard MAE(random)` CI95 **high < 0**, i.e. random is better, at ≥ half the checkpoints |
| **F4** | Three measurements of a distant ligand beat three more of a known one | the `b = 3` curves of `maxmin` lie below the curve obtained by spending the same rows on the start cohort's own ligands — which is Experiment B's depth result, re-measured here only as a sanity check and not as a new claim |

Falsifiers: if `maxmin` is indistinguishable from `random`, the *choice* of ligand does not matter,
only the breadth (Experiment B already showed breadth matters); if uncertainty-type policies are no
better than `maxmin`, the model's own uncertainty carries no acquisition information beyond chemical
distance, and the cheap chemistry-only rule is the one to deploy.

---

## Required controls and disclosures (both experiments)

* every new model compared against the same learner without the structure (C) / against random
  acquisition (F);
* the monolithic ET control must reproduce Experiment A's EXPANDED arm on the chemotype folds (same
  machine) — a non-zero difference stops the run;
* cross-fitted residual provenance asserted by test (C2): Stage B never sees a residual formed from
  an in-sample Stage A prediction;
* label-free acquisition asserted by test (F): permuting or blanking the pool's `log_D` leaves every
  policy's ordering unchanged;
* multiplicity: only the hypotheses above are protected; every other interval is descriptive;
* the dominant-block caveat from Phase 1 applies: the `all` endpoint's percentile interval is not
  trustworthy alone, and the BCa / cluster-robust / block-macro numbers are reported beside it.
