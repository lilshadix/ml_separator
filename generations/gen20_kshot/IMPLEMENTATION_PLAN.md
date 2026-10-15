# Gen20 — leakage-safe k-shot learning for adjacent-lanthanide separation: implementation plan

*Draft 3, 2026-10-14 (drafts 1–2 superseded; §6 lists what the eight audit reports changed and §7 what the adversarial review changed), written after the Phase 1 audit of `ml_separator` (HEAD `8030e16`, branch
`gen19-chem-transfer`) and of the reference repository `lilshadix/lanthanidestrain` (commit
`025f8084`, 2026-09-12, cloned read-only outside the repository). No gen20 code exists yet and no
gen20 number has been computed. Items marked **[audit]** are filled in from the audit reports;
items marked **[decision]** are choices this plan makes and the pre-registration will freeze.*

## 0. One-paragraph summary

The question is whether k measured separation factors of a chemically unseen extractant make its
*adjacent*-lanthanide separation factors predictable, and whether a model earns anything over the
measurements themselves. The study reuses the frozen gen13 cohort, fold plans (all five designs,
BP selects), metrics and chemotype-blocked bootstrap, and gen15's BLUP measured mode, and adds four
things the programme has not had: a fixed adjacent-pair query set that is byte-identical across
every k and every policy (so "R² versus k" is a like-for-like curve), an arithmetic-closure rule
that keeps every scored query algebraically independent of the supports, a covariance built only
from inner out-of-fold residuals of the masked training fold (positive semidefinite by
construction), and R² with explicit undefined-case handling computed inside every bootstrap draw.
A neural contrast model adapted from the reference repository's tabular control is the new
architecture; two things are expected from the programme's history and must be reported if they
occur: zero-shot adjacent R² near or below zero, and the neural model not beating a two-parameter
line once k ≥ 2.

## 1. What exists and is reused (Phase 1 audit, condensed)

### 1.1 Data and target

| item | what the repository has | reuse |
|---|---|---|
| bundle | `dataset with 3D structures/dataset.parquet`, SHA-256 pinned in `gen3_protocol.json` and re-checked by `gen13sep.paths.assert_bundle_unchanged()` | read-only, via gen13 |
| cell | one extractant × one `publication_id` × the exact 64-column condition key (NaN is a value), ≥ 2 lanthanides; replicates averaged per (cell, metal) with `nrep__`/`repsd__` kept (`gen13sep/cohort.py`) | the unit of a separation factor; reused as is |
| cohort | 521 cells, 90 extractants, 45 chemotypes (frozen gen6 single-linkage Tanimoto-0.7 superclusters), 58 publications; `BenchData` from `gen14.dirbench.load()` (cached `generations/gen14_direction/cache/bench.pkl`, built in 1 s on this machine) | reused; fingerprint recorded in the manifest |
| target | `log D` is log10; `y(A/B) = log D(A) − log D(B)` with **A the lighter metal** (`gen13sep.metrics.pair_rows_for_cell`), so `y(La/Ce) = log10 D_La − log10 D_Ce` | **[decision]** gen13's lighter-first orientation is kept (it matches the brief's La/Ce … Yb/Lu list); gen19 is heavier-first and is not joined; a sign-flip adapter with a Nd/Pr ↔ Pr/Nd test is included |
| promethium | absent from the bundle and from every table | never imputed; Nd/Pm and Pm/Sm do not exist |
| adjacent | gen13 defines adjacent as `dZ = 1` **plus Nd–Sm**; the reference repo and the recovery plan use strict `|ΔZ| = 1` | **[decision]** primary = strict `|ΔZ| = 1` (12 pair types); "adjacent + Nd–Sm" reported alongside for continuity |
| conditions | two metals form a pair only inside one cell, i.e. identical 64-column key and publication; this is the repository's established comparability rule | reused; no cross-cell differencing |
| noise floor | median replicate sd 0.30 log D per (cell, metal) (gen13 DATA_AUDIT); adjacent `y`: n = 2114, sd 0.231, mean −0.089, `FLAT` MAE 0.167, median |y| 0.116 (counted 2026-10-14 on the bench) | printed in every table |

### 1.2 Splitting and leakage machinery

* `gen13sep.splits.all_folds(frame, design, seeds)`: 5 chemotype-grouped folds × 5 seeds
  `{104729, 130363, 155921, 196613, 262147}`, model seed `42 + fold·1009 + 9 999 991`, designs
  A/B/BR/BQ/BP; **BP drops every training cell whose publication appears among the held-out
  cells**; inner 4-way chemotype split re-dealt on the masked training set
  (`Fold.inner_train_index`, `Fold.inner_validation_index`). `assert_fold_integrity` raises on any
  extractant/ECFP-cluster/chemotype crossing and, for BP, any publication crossing.
* Hold-out guarantees already tested in `gen13_separation/tests/test_gen13_invariants.py`
  (fold integrity, target-free features, forbidden feature tokens, antisymmetry).
* Known gaps for k-shot that gen15 handled only partly (`gen15/fewshot.py`): the residual covariance
  was estimated from *other test folds'* residuals of the same seed (publication masked only with
  `mask_publication=True`), was indefinite in 100 % of estimates (gen16 L5), and no arithmetic-
  closure rule existed (supports and queries lived in one complete graph).
* `hash()` salting: gen15 seeds the random support RNG with `hash((cell_index, seed))` of two ints,
  which is deterministic; gen20 will use SHA-256 of stable strings and test reproducibility in a
  subprocess (gen16 convention, `test_fold_plan_hash_matches_recorded_in_subprocess`).
* Duplicates **[audit]**: 7 ECFP-bit-identical groups cover 20 of the 90 extractants and every one
  lies inside a single chemotype; removing stereo marks collapses 90 → 88 SMILES, all inside one
  chemotype; so homologues and stereoisomers never cross a B-family fold (they do cross design A,
  which is why A is a leak price, never a headline). Cross-publication copies: 45 gen13 cells
  (5 publications, 3 gen19 copy groups, 21 extractants, 5 chemotypes) sit in copy groups that span
  more than one publication, and gen13's BP masks the raw `publication_id`, not the copy group.
  Gen20 adds design **BPG** (BP plus the copy-group mask from
  `gen19_chem_transfer/data_audit/leakage_publication_components.csv`) as a registered
  sensitivity; BP stays the selecting design for continuity. BPG removes 0–15 further training
  cells per fold on seed 104729.
* Arithmetic closure in the locked numbers **[audit]**: under gen15's widest-3 rule 963 of 12 863
  scored pairs (7.5 %) in all 321 scored cells are exactly determined by the supports; gen4
  measured ~32 % at k = 10. Every gen15 k ≥ 2 number therefore contains some arithmetic
  recovery; gen20's no-closure rule removes it and the gen15 replication reports the closed
  fraction separately.
* Covariance provenance **[audit]**: gen15's residual rows are out-of-fold residuals of *other test
  cells* of the same split seed, produced by models that had seen the corrected cell; the
  publication mask is opt-in (`mask_publication=False` by default in `fewshot.evaluate`, enabled by
  `--maskpub` in `g15_fewshot.py`; every reported §4/§5 number used it). Gen20 estimates the
  covariance from inner cross-fits of the masked training fold only (`gen20k/cov.py`), with the
  design's mask re-applied inside the inner folds, and projects it to PSD (min eigenvalue before
  projection recorded: −0.41 on BP seed 104729 fold 1, consistent with gen16's finding).
* Selection history **[audit]**: gen14's deployed `C = 1` logistic was chosen from a 30-candidate
  sweep scored on the *outer* folds of all five designs on the discovery seeds, and gen15's
  D-optimal policy likewise; the gen16 confirmation covered only one claim. The existing baseline
  numbers are discovery-seed, outer-selected numbers; gen20's registered claims are re-estimated
  once on withheld seeds.

### 1.3 Existing zero-shot and k-shot results to reproduce (anchors)

| anchor | value | source |
|---|---|---|
| `G14` under BP, extractant-macro MAE, all pairs | 0.5000794414203691 | `gen16_leads/tests/test_anchors.py` |
| `FLAT` under BP | 0.5885062528901843 | same |
| gen15 measured mode, common set, BP: widest k=1 / D-optimal k=3 | 0.2261 / 0.1700 | gen15 §5, `results/g15_support_board_empirical_BP.csv` |
| `G14 − NAIVE_LINE` at D-optimal k=3 | +0.062…+0.065, p < 1e-4 | gen15 §5 |

### 1.4 Model components available

* Zero-shot: `gen15.arms.g14` (L2 logistic on the 39 `TOPO39` columns → direction, × training-fold
  mean magnitude, training-fold mean curvature; curve = `coef @ basis` with the physics basis
  `radius`, `radius_sq`: the standardised Shannon effective ionic radii for CN 8 and their square), `gen15.arms.flat`, `gen15.arms.mean_curve`, `gen15.arms.g13_full`.
* Measured mode: `gen15.fewshot.blup`, `blup_posterior`, `pair_sd`, `naive_line`,
  `line_plus_curvature`, `pick_support(widest|dopt|random)`, `residual_covariance`,
  `smooth_covariance`.
* Features: `BenchData.frames` blocks `COND` (64), `MASSACT`, `PHYSCHEM` (10), `DONORS` (13),
  `COORD` (114), plus `ECFP` (2048) and `LIG2D` (206) which are known harmful
  (`lig2d-ext-is-harmful`, gen13 §5a); `LEAN_BLOCKS = (COND, MASSACT, PHYSCHEM, DONORS, COORD)`.
  Every block is a function of structure and recorded conditions only (`test_features_are_target_free`).
* Metal-side constants: `gen13sep.metals` (Shannon radii, hydration energies, f-count, tetrad
  functions), all target-free.
* Inference: `gen13sep.inference.paired_contrasts` (chemotype-blocked paired bootstrap, 10 000
  draws, seed 8675309, percentile + BCa, per-seed sign, LOCO, `passes_P1`),
  `gen13sep.wildcluster` (wild-cluster-t and Bayesian bootstrap, for the few-cluster caveat found
  in `gen16_protocol`).

### 1.5 The reference architecture (`lanthanidestrain`) **[audit: verified]**

Read-only; nothing from it was executed or imported.

* **It is the upstream of this repository's data.** `collaborator_update/dataset.parquet` is
  byte-identical (SHA-256 `fefbefc6…`) to `dataset with 3D structures/dataset.parquet`; same 5 992
  rows × 2 261 columns, identical `log_D` and all 64 condition columns; only the geometry QC side
  differs (geometry_ok 4 746 there vs 5 479 in the bundle). `raw_data/*_SAFE.csv` (31 metals,
  48 138 rows) is the same snapshot the bundle was built from: the lanthanide rows reduce to the
  same 5 992 after the builder's de-duplication, so **the reference repository adds no lanthanide
  measurement and no adjacent-pair coverage**. What it adds and the bundle lacks: a raw DOI string
  per row (a cross-check for the reconstructed `publication_id`), and Am/Cm/Y/actinide rows that
  the reference itself found closed for the lanthanide target.
* **Target and evaluation.** Per-row `log D`; adjacent `log SF` is built post hoc from out-of-fold
  row predictions inside an (extractant, binned-conditions) block (`evaluation.py::
  adjacent_pair_arrays`, strict `|Δindex| = 1`, replicates averaged per (block, metal) first).
  Hold-out: 5-fold grouped CV on the extractant SMILES × 2–5 repeats; **no chemotype hold-out, no
  publication or laboratory hold-out anywhere**. Headline: pooled adjacent-pair R² **+0.326** for
  `anchor + 0.65·shape_tabular + 0.35·shape_encoder`; best single model an anchored CatBoost
  (quantile 0.6, 8 seeds) at +0.318; swapping RMSE for MAE in CatBoost alone is worth +0.107 of
  adjacent R²; the within-block contrast objective is worth +0.186 (simplicial), +0.042 (PI-CNN),
  +0.030 (tabular).
* **Neural architectures (torch only).** `SimplicialNet` (message passing over Vietoris–Rips
  complexes, 9·96-wide embedding), `DistanceNet` (SchNet-style continuous-filter convolution, the
  no-simplex control), `AttentionNet` (distance-biased transformer over atoms), `PersistenceCNN`,
  `FTTransformer` (tabular tokens, pinball loss), `BlockTransformer`/`CellTransformer` (set
  transformers over the rows/cells of one block, output centred within the block), and the
  zero-width `TabularNet` control. One training loop (`topo/train.py::run_fold`): AdamW, Huber on
  per-block means (the level) plus the within-block pairwise contrast with ×3 adjacent weight,
  inner validation grouped by extractant, early stopping on MSE or adjacent-pair R², best state
  restored, ensembles of 8–16 seeds, optional self-supervised pretraining of the simplicial
  encoder on masked xTB charges.
* **No few-shot, transfer or fine-tuning mechanism exists** (grep confirmed). What exists nearby:
  post-hoc nested scale/affine/isotonic calibration of the predicted difference fitted on the
  other extractants, split-conformal intervals, and the actinide route (`an_ln/`, closed).
* **Operational state:** nothing runs as checked out (gitignored matrix cache, descriptor shards,
  OOF parquets and pretrained encoders are absent; SLURM paths are site-specific).

**Compatibility verdict [decision].** The 3D encoders need the reference's per-complex geometry
assets, which are absent, and this repository already tested that family (gen2 Delta3D/
simplicial: harmful under the honest hold-out; gen13/15/16: the 3D response block and the xTB
cycle null; the reference itself: fixing the geometry does not improve the score). Re-porting them
is not scientifically justified. What is adapted as **Model 2** (`gen20k/nnc.py`) is the
reference's **control** architecture, not its best system: the `TabularNet` head (built there as
the no-topology control of the 3D arms), the QuantileTransformer → StandardScaler front end and the
within-block pairwise-contrast loss (×3 adjacent), applied to the gen13 feature blocks plus
target-free metal descriptors on the centred curve. Training differs from the reference in ways
§4.3 lists (full-batch AdamW, stopping on a masked validation contrast loss followed by a refit on
every training cell, 8 seeds, chemotype hold-out). The reference's best system (anchored CatBoost
quantile + 3D encoder blend, +0.326 adjacent R² under leave-extractant-out) is not reproduced. Two further reference devices are adopted: the adjacent-pair arrays are
factored into one function shared by metrics and figures, and every blend/penalty is selected on
the other extractants only.
* Neural components (`automl/topo/`): `TabularNet` (LayerNorm → Linear 256 → SiLU → Dropout 0.15 →
  Linear 128 → SiLU → Dropout → Linear 1; inputs QuantileTransformer → StandardScaler),
  `DistanceNet` (CFConv over interatomic distances), `SimplicialNet`, `PersistenceCNN`,
  `AttentionNet`, `FTTransformer`, `BlockTransformer`, `CellTransformer`; one training loop
  (`automl/topo/train.py::run_fold`) with a **decomposed objective**: a Huber loss on per-block
  means (the level) plus a within-composition-block **pairwise-contrast loss** weighted ×3 on
  adjacent metals, and **checkpoint selection on inner-validation adjacent-pair R²**; ensembles of
  16 seeds. The repository's own control (`TabularNet`) exists precisely to test whether the
  objective, not the 3D encoder, carries the adjacent-pair gain.
* `automl/advanced.py`: `PairwiseDeltaRegressor` (delta learning on within-block metal pairs),
  `AnchoredResidualRegressor` (level from a flat model, shape from a block-centred residual model),
  `TwoStageRegressor`, `BlendRegressor`.
* No few-shot or transfer mechanism was found in the files read; the audit confirms or corrects this.

**Compatibility verdict [decision, subject to the audit].** The 3D encoders need per-complex
geometries keyed to the reference repository's complex ids; gen2 of this repository already ported
that family (Delta3D/simplicial) and found it harmful under the honest hold-out, gen13/gen15/gen16
found the 3D response block and the xTB cycle null, and the reference repository itself reports
that fixing the geometry does not improve the score. Re-porting the 3D encoders is therefore not
scientifically justified here. What *is* compatible and is adapted as **Model 2**: the
`TabularNet` head and the decomposed contrast objective, applied to the gen13 feature blocks plus
target-free metal descriptors, trained on the centred curve (so the level term is zero by
construction), with checkpoint selection on inner-validation adjacent-pair MAE. The documented
modifications are in §4.3.

## 2. Scientific risks (ranked)

1. **Adjacent separation factors sit at the noise floor.** sd 0.231 against a replicate floor of
   ~0.21 on one `log SF`; zero-shot R² can be ≤ 0 for every arm, and a negative R² is a valid
   result. Mitigation: R² never clipped, reported with the target variance and `FLAT`; MAE is the
   selection metric; the pre-registration states the expected sign.
2. **The measurement does the work, not the model** (gen4, gen15 §4: ligand chemistry is worth
   +0.008 after one pair). Mitigation: every model is scored against `NAIVE_LINE` and a ridge line
   with *identical supports*; the pre-registered primary contrast is model − line, not k − 0.
3. **Arithmetic closure.** Inside a cell `log SF` is exactly additive; with k supports on n metals,
   pairs whose endpoints lie in one support component are determined without any model (32 % of
   query rows in the gen4 study). Mitigation: the no-closure rule (§3.3) enforced at every
   selection step, tested, and the "algebraically recovered" pairs reported separately, never as
   model skill.
4. **Publication fingerprint** (gen13 §5a: 94 % 1-NN identification from the condition columns).
   Mitigation: BP selects; the covariance and every preprocessing object are fitted on the masked
   training fold only; all five designs reported in full (gen16 contract).
5. **Few effective units** (Kish 11.7 chemotypes; one chemotype holds 375 of 521 cells). Mitigation:
   chemotype-blocked bootstrap with BCa, LOCO, wild-cluster-t beside it, five seeds not treated
   as replicates, confirmation on withheld seeds, Bonferroni over ≤ 4 registered claims, BH over the
   exploratory family.
6. **Shrinking cohorts across k.** A cell needs `n_metals − 2 ≥ k` to leave any adjacent query
   undetermined; k = 10 is feasible on 114 cells / 50 extractants / 25 chemotypes / 25 publications
   (k = 3: 275 / 81 / 39 / 40). Mitigation: nested common-query cohorts (§3.4) and a remaining-query
   regime reported separately; eligibility counts printed at every k.
7. **Over-fitting a neural model to ~400 training cells and 12 effective chemotypes** (gen14:
   the linear model beats the forest here). Mitigation: ≤ 3 pre-declared configurations, inner
   chemotype-grouped selection, ensembles, the full fine-tuning vs head-only comparison on inner
   folds only, and the honest expectation that Model 2 ties or loses.
8. **Oracle inflation.** Any ceiling must be leave-pair-out (gen15 §1a). No in-sample oracle is
   reported.
9. **Seeds and hashes.** Fold plans and support draws are hashed and re-derived in a subprocess.
10. **The percentile chemotype bootstrap over-rejects** (gen13 inference audit: 0.071–0.077 at
    nominal 0.05 on this cluster profile; wild-cluster-t with CR2/CR3 holds 0.044). Mitigation: the
    wild-cluster-t p (`gen13sep.wildcluster`) is printed beside every per-unit percentile/BCa
    interval, BCa uses the mid-rank convention and reports NaN when degenerate, "excludes 0" is
    tested on both bounds, LOCO is strictly positive, and seeds are combined by applying one cluster
    draw to every seed (gen19's nested convention) rather than averaging per unit first.
11. **Setting B is a replicate-floor measurement** (audit B.7): within one (extractant,
    publication) group the adjacent `log SF` changes by a median 0.043 between conditions (91 %
    below the 0.3 replicate floor); copying the value from another cell scores 0.105 against FLAT
    0.160; 24 non-duplicate units, 17 extractants, 11 chemotypes, 64 % of the query pairs in one
    chemotype; five identical-log-D cell pairs split only by a NaN condition. Mitigation: Setting B
    is exploratory only, with COPY as its null, duplicates removed, chemotype-macro and LOCO shown.
12. **Cohort fingerprint drift on this machine** (gen16 L6 `exact` rebuild fingerprints
    `854171b03411d901`, not `4c3c6628ea0be949`). Mitigation: the anchors in §1.3 are reproduced to
    the recorded digits before anything else; the fingerprint actually obtained is recorded.

## 3. Evaluation protocol (to be frozen in `PRE_REGISTRATION.md`)

### 3.1 Prediction target

`y(A/B) = log10 D(A) − log10 D(B)`, A lighter, one cell. Primary query population: strict adjacent
pairs. Secondary populations: adjacent + Nd–Sm; all pairs (remaining-query regime); far pairs
`dZ ≥ 5`.

### 3.2 Hold-out

Design B (chemotype) and design BP (chemotype + publication mask) from `gen13sep.splits`, five
discovery seeds; designs A, BR, BQ and the new **BPG** (BP + copy-group mask, §1.2) run with the
light policy set (gen16 contract: all five designs, full table, BP selects; BPG is a sensitivity).
Inner selection: the fold's own 4-way chemotype split on the masked training set with the design's
mask re-applied inside the inner folds, never the outer test. Every fold set is re-checked by
`gen20k/splits.py::assert_design_integrity` (chemotype, extractant, publication and copy-group
crossings; inner folds inside the training set).

### 3.3 Setting A — measurement adaptation (primary)

Two measurement models are run (PRE_REGISTRATION.md §3). The **metal model** is registered: a
measurement is one metal's log D reading, so after k + 1 read metals every separation factor among
them is known (the support is the read set; adapters see its star spanning tree; with the
shared-reading noise the tree is immaterial); the fixed adjacent query forces the read set to be
an independent set of the adjacency path, a metal is legal only if the maximum independent set
stays reachable (path DP, exact), and k_max is 6 on a 14-metal cell. The **pair model** below is
the brief's literal definition and is exploratory: it keeps per-pair closure out but supports still
fix linear combinations of the adjacent contrasts (k of the 13 curve degrees of freedom at budget
k), so every pair-model row carries `lab_known` and the arithmetic-only `PROJ` arm is scored beside
every model.

For every held-out cell with ≥ 1 adjacent pair (pair model):

* **Q_adj** = all strict-adjacent pairs measured in the cell. Fixed before any model runs; identical
  for every k, model and policy.
* **Support candidates** = all measured non-adjacent pairs. A support set is a forest over the
  cell's metals. **No-closure rule:** no component of the support forest may contain both metals
  of any Q_adj pair; a candidate pair is legal at a step only if the merge keeps the component
  conflict graph bipartite (so every policy can always reach `k_max = n_metals − 2`). Proven
  equivalent to the theoretical maximum on all 378 eligible cells (prototype, 2026-10-14).
* **k ∈ {0, 1, 2, 3, 5, 8, 10}.** Policies produce one nested sequence per cell; budget k uses its
  first k pairs.
* **Support labels** are the cell's true `y` for those pairs (this is the point of few-shot).
  Query labels are never read by any adapter or policy. Supports are never scored.
* **Policies** (all restricted to legal candidates, lexicographic tie-break on (Z_A, Z_B)):
  `RANDOM` (5 draws, RNG seeded by SHA-256 of `gen20|cell_id|split_seed|draw`), `WIDEST` (largest
  dZ first; gen15's locked heuristic), `UNIFORM` (farthest-point coverage of the Z axis: widest
  first, then the pair maximising the summed minimum |Z − Z_used| of its endpoints), `DOPT_CURVE`
  (gen15's greedy D-optimal under the training covariance, whole curve), `DOPT_Q` (greedy maximal
  posterior-variance reduction summed over Q_adj), `UNC_SEQ` (sequential: after each support is
  revealed the adapter updates and the next pair is the legal candidate with the largest
  predictive sd under the updated model; for a Gaussian BLUP this coincides with `DOPT_Q` and is
  only distinct for the ensemble neural adapter). Batch vs sequential is labelled.
* **Common-query cohorts (primary regime):** metal model `C3` = cells with `k_max ≥ 3` (229
  cells, 79 extractants, 38 chemotypes), `C5` (144 / 64 / 29), `C6` (133 / 58 / 27); pair model
  `C3` (275 / 81 / 39), `C5` (196 / 74 / 34), `C10` (114 / 50 / 25). Within a cohort every k is
  scored on byte-identical Q_adj. **Registered claims are made on the metal model's `C3` (k = 3)
  and `C6` (k = 6).** The pair-weighted Kish effective number of chemotypes on these rows is about
  4 (one chemotype holds half of the adjacent pairs) and is printed with every cohort.
* **Remaining-query regime (secondary):** at each k, all non-support pairs whose endpoints are in
  different components, over the cells eligible at that k; the population changes with k and is
  labelled so.
* **Policies share one covariance.** `DOPT_CURVE` and `DOPT_Q` use the reference arm's (`G14`)
  inner-OOF covariance so that supports are identical for every arm; `UNC_SEQ` is run with the
  neural ensemble only (for the Gaussian BLUP it coincides with `DOPT_Q`). Under A/BR/BQ/BPG only
  `RANDOM` (one draw), `WIDEST` and `DOPT_Q` are run. The remaining-query regime is recorded for
  `DOPT_Q`, `WIDEST` and `RANDOM` draw 0 under B and BP.
* **Setting B — condition adaptation: exploratory only (audit B.7).** Unit = (extractant,
  publication) with ≥ 2 cells sharing ≥ 1 strict-adjacent pair after removing the five
  identical-log-D cell pairs (`gen20k/setting_b.py::DUPLICATE_CELL_PAIRS`); 24 units, 17
  extractants, 11 chemotypes, 281 query cells, 1 397 query pairs. For each query cell the support
  cell is the other cell with the most shared adjacent pairs (ties: nearest log acid, then
  cell id); supports = all pairs of the support cell; query = the query cell's adjacent pairs the
  support cell also measures. Arms: `COPY` (the null), `FLAT`, zero-shot arms, arm + BLUP20 with the
  support cell's pairs, `LINE`. Reported as model − COPY with chemotype-blocked intervals and LOCO,
  never as a registered claim; the cross-publication variant (9 extractants, 4 chemotypes) is out
  of scope.

### 3.4 Models

* **Model 1 (existing baseline):** `G14` zero-shot (reproduced to the anchor) + the gen15 BLUP
  adapter, in two forms: `G14+BLUP15` (gen15's exact covariance construction, replication only)
  and `G14+BLUP20` (covariance from inner out-of-fold residual curves of the masked training fold,
  shrunk toward its diagonal and projected to PSD, shrinkage chosen on inner folds; measurement
  variance `NOISE_VAR` from the training replicates). Optional sensitivity: `G13_LEAN_BAG`.
* **Model 2 (reference neural architecture, adapted):** `NNC` — `TabularNet` head on
  `[LEAN blocks ‖ metal descriptors]` predicting the per-(cell, metal) *centred* log D, trained with
  the within-cell pairwise-contrast loss (adjacent weight 3, Huber), QuantileTransformer →
  StandardScaler fitted on training cells; **early stopping on the inner-validation contrast
  loss** (the training objective on the fold's inner-validation cells, minimum 20 epochs, patience
  30; the smoke run showed that stopping on inner adjacent MAE halts at initialisation because an
  untrained, flat network is already at the adjacent noise floor, so that criterion is recorded but
  not used to stop); ensemble of 8 seeds; k-shot mechanisms `BLUP20` (residual BLUP as in Model 1,
  covariance from NNC's own inner OOF residuals), `HEAD_FT` (output layer refitted on the support
  contrasts with an L2 penalty toward the zero-shot weights, closed form), `FULL_FT` (all layers,
  30 Adam steps, same penalty; the documented over-fitting check, run on inner folds for every k
  and on the outer folds only for `DOPT_Q` at k ∈ {1, 3, 10} with 4 members). **Selection on inner
  folds per (design, seed, fold)** (`gen20k/runner.py::select_on_inner`): the three configurations
  × {BLUP20, HEAD_FT with λ ∈ {0.1, 1, 10}, FULL_FT with λ ∈ {1e-3, 1e-2, 1e-1}} are fitted on the
  inner-training cells and scored by the mean of the inner-validation adjacent MAE at k = 0 and
  k = 3 (`DOPT_Q` supports); the winner is `NNC@SELECTED`, and the selection record is saved.
* **Model 3 (statistical baselines, same supports):** `FLAT` (k = 0 and, as `FLAT@BLUP20`, the
  covariance with no chemistry), `LINE` (gen15's `NAIVE_LINE`: least-squares radius ramp through the
  supports), `RIDGE2` (radius + radius² with ridge toward the training-fold mean coefficients,
  `R^-1`-weighted, λ chosen on inner folds from {0.3, 1, 3, 10}), `MEAN_CURVE` (gen15's balanced
  mean coefficients; worse than FLAT, kept for continuity), `NN_TANIMOTO` (nearest training
  extractant's chemotype-weighted ridge curve; `@RESCALE` fits one scalar to the supports;
  `@BLUP20` applies the residual BLUP). gen13's `PAIRMEAN` (0.642 under BP, worse than FLAT) is
  not re-run. Measurement noise: `R = s² D Dᵀ` with `s²` the median training-replicate variance
  (pairs sharing a reading are correlated); gen15's independent `2s²·I` is kept as `G14@BLUP_INDEP`.

### 3.5 Metrics (computed once per (design, seed, k, model, policy) on the query rows)

Pooled over query rows: `R2` (`1 − Σe²/Σ(y−ȳ)²`, undefined → `NA` with reason when n < 2 or
`Var(y) = 0`, never clipped), `MAE`, `RMSE`, direction accuracy on all |y| > 0 and on |y| ≥ 0.1 and
|y| ≥ 0.3 (thresholds declared now; a zero prediction counts 0.5), extractant-macro MAE (continuity
with gen13–gen16), per-pair-type MAE and R² (12 types), `ΔR2(k) = R2(k) − R2(0)` and
`ΔMAE(k)`, Spearman within cells with ≥ 4 queries. Per-seed values, mean and sd over seeds; the five
seeds are not independent replicates and are not resampled. Per-extractant R² is **not** averaged.

### 3.6 Inference

Chemotype-blocked bootstrap (one index matrix, 10 000 draws, seed 8675309, shared by every arm and
k; `gen20k/inference.py`): pooled R², MAE and RMSE recomputed inside each draw from per-cluster
sums; percentile and BCa (mid-rank, NaN when degenerate) intervals; paired differences on identical
query rows; leave-one-chemotype-out; per-seed values with one cluster draw applied to every seed
and the headline the seed mean; for per-extractant mean contrasts the wild-cluster-t p (CR2) and
cluster-robust dof beside the percentile p. Decision
rule for a registered MAE claim = gen13 P1 (Δ ≥ 0.02, both intervals exclude 0, p < 0.05, ≥ 4/5
seeds, LOCO stable); for a registered R² claim: `ΔR2` interval excludes 0 and `R2(k) > 0`.
Registered claims (≤ 4, Bonferroni 98.75 % intervals; estimand = extractant-macro MAE for P1/P2/P4,
pooled ΔR² for P3; metal model): P1 `M*` vs `LINE` at k = 3 on identical `DOPT_Q` supports, C3, BP;
P2 `M*` k = 3 vs k = 0, C3, BP; P3 the same, ΔR²; P4 `M*` vs `LINE` at k = 6, C6, BP. `M*` is
decided from the inner selection records alone (PRE_REGISTRATION.md §4). Everything else (policies, per pair, Setting B, Model 2 vs
Model 1 at every k, remaining-query regime, other designs) is exploratory with BH over the counted
family. Confirmation: the registered claims re-run once on five withheld seeds generated by the
gen16 rule with the string `gen20-confirmation-seed-{i}`; commitment hash published in the
pre-registration; runner refuses a second execution.

## 4. Implementation

### 4.1 Layout (follows `generations/README.md`)

```
generations/gen20_kshot/
  IMPLEMENTATION_PLAN.md   this file          PRE_REGISTRATION.md  sealed before any run
  AUDIT.md                 Phase 1 audit      GEN20_REPORT.md      final report (Phase 8)
  gen20k/   bootstrap.py (sys.path + thread caps), data.py (bench with manifest-equality check,
            pairs, orientation, SHA-256 seeds), splits.py (gen13 designs + BPG, integrity check),
            design.py (Q_adj, no-closure forest, label-free policies, Support), cov.py (inner-OOF PSD
            covariance, shared-reading noise), adapters.py (BLUP, line, ridge line, rescale),
            arms.py (G14, FLAT, MEAN_CURVE, NN_TANIMOTO behind one interface), nnc.py (Model 2 +
            HEAD_FT/FULL_FT), metrics.py (R² with NA, MAE, RMSE, direction with the 0.5 tie rule,
            per pair), inference.py (vectorised cluster bootstrap over pooled statistics, BCa, LOCO,
            wild-cluster-t, BH), setting_b.py (exploratory arm), manifest.py (git, environment,
            digests, pre-registration seal), runner.py (the grid + inner selection), report.py
  scripts/  g20_audit.py, g20_anchors.py, g20_discovery.py, g20_confirm.py, g20_report.py,
            g20_reproduce.sh (the single command)
  tests/    test_leakage.py (fold integrity, BP publication mask, support/query isolation, closure
            invariant, covariance provenance, preprocessing provenance, selection provenance,
            subprocess reproducibility, orientation, NaN never imputed) and test_leakage_injection.py
            (deliberate violations must raise)
  manifests/ results/ predictions/ (parquet, gitignored above the size budget; digests recorded)
```

### 4.2 Reuse map

`gen14.dirbench.load`, `gen15.valuebench.Ctx` and `run_arms` (zero-shot curves),
`gen15.arms.{flat,g14,mean_curve}`, `gen15.fewshot.{blup,blup_posterior,pair_sd,naive_line,
residual_covariance,smooth_covariance,pick_support}`, `gen13sep.splits.all_folds`,
`gen13sep.metrics.{per_extractant,summarise,is_adjacent}`, `gen13sep.inference.paired_contrasts`,
`gen13sep.wildcluster`, `gen13sep.metals`, `gen16` seed rule and anchor tests. Nothing frozen is
modified; every new function lives under `gen20k/`.

### 4.3 Model 2 adaptation, documented changes from the reference

| reference (`lanthanidestrain/automl/topo`) | gen20 | why |
|---|---|---|
| per-row `log D` with level + contrast losses | per-(cell, metal) *centred* log D; level weight 0 | the target is the within-cell curve; the level does not transfer (gen5–gen12) |
| composition blocks from the collaborator's keys | gen13 cells | the established comparability unit |
| leave-extractants-out, 3 repeats | chemotype hold-out, BP mask, 5 seeds | the programme's validity contract |
| checkpoint selection on inner adjacent R² | inner adjacent MAE (R² undefined on small inner sets) | MAE is the selection metric |
| 16-seed ensembles | 8 seeds | compute cap; ensemble sd reported |
| 3D encoders | none | §1.5 verdict |
| no few-shot | residual BLUP / head FT / full FT | the study's question |

### 4.4 Validation tests (Phase 3, all automated)

1–2 fold integrity and BP mask (reuse + re-assert on gen20's own fold records); 3 query labels
unreachable: adapters receive a `Support` object that carries only support pairs, and the pair table
passed to adapters has `y` dropped for queries (tested by monkey-patching `y` with NaN for queries
and asserting identical predictions); 4 supports never scored; 5 preprocessing fitted on training
indices only (fit-index logging); 6 selection records contain only inner scores; 7 fold plan,
support draws and predictions reproduce in a subprocess (SHA-256); 8 `y(A/B) = −y(B/A)`, Nd/Pr test,
cell additivity; 9 every prediction row maps to a measured (cell, A, B); 10 NaN metals never
imputed (a cell with a missing metal yields no pair for it). Injection tests: a test cell copied
into training, a support pair inserted into Q, a residual curve from the held-out publication
inserted into the covariance pool, a closed pair kept as query, a `y`-derived feature column — each
must make the corresponding validator raise.

### 4.5 Compute

Local macOS, 8 cores, 16 GB; `LOKY_MAX_CPU_COUNT=2`, torch 4 threads. `G14`, BLUP, lines: seconds
per fold. `NNC`: ~5 k (cell, metal) training rows, ≤ 3 configurations × 4 inner folds for selection
under BP only, 8-seed ensembles on the outer folds of all five designs; estimated 2–4 h total. A
smoke run (one seed, design BP, k ≤ 3, 2 NN seeds) precedes the full run and is labelled
`smoke`, never reported as a result.

## 5. Phase order and stop conditions

| phase | deliverable | stop/decision |
|---|---|---|
| 1 audit | this plan, `AUDIT.md` | — |
| 2 protocol | `PRE_REGISTRATION.md` sealed (SHA-256 inside, committed), seed commitment | nothing outcome-bearing runs before the commit |
| 3 validation | `gen20k/design.py`, `cov.py`, `tests/` green incl. injection tests | a failing leakage test invalidates the affected experiment |
| 4 baselines | anchors to the digit; gen15 k ≤ 3 replication; new-protocol ladder for `FLAT`, `G14+BLUP20`, lines | if an anchor does not reproduce, stop and diagnose |
| 5 model | `NNC` + adapters, selected on inner folds | — |
| 6 evaluation | full grid, discovery seeds; smoke first | 12 h compute cap |
| 7 statistics | boards, contrasts, BH, confirmation run once | — |
| 8 report | `GEN20_REPORT.md`, tables, figures, manifests, reproduce command | negative results reported as results |

## 6. What the audit changed (draft 1 → draft 2)

| finding (auditor) | change |
|---|---|
| BP masks the raw publication id; 45 cells sit in multi-publication copy groups (splits) | design `BPG` added as a registered sensitivity |
| 7.5 % of gen15's scored pairs are support-determined (splits, k-shot) | no-closure rule is primary; the gen15 replication reports the closed fraction |
| gen15 covariance from other test folds, indefinite, publication mask opt-in (splits, k-shot, stats) | inner-OOF covariance with the mask inside, PSD projection, min eigenvalue recorded |
| supports sharing a log D reading treated as independent (k-shot) | `R = s² D Dᵀ` registered; independent `R` kept as a sensitivity |
| percentile bootstrap rejects 7–8 % (stats) | wild-cluster-t beside it; BCa mid-rank/NaN; both bounds; strict LOCO; nested seed convention |
| no R² exists on log SF; pooled R² must be per seed (stats, data) | R² per seed, never averaged over extractants, NA with reason, recomputed in every draw |
| zero prediction counted wrong in gen13 (stats) | 0.5 tie rule declared, applied to every arm |
| adjacent = dZ 1 + Nd–Sm in gen13, strict in the reference (data) | strict primary, gen13 definition secondary |
| Setting B at the replicate floor, duplicates, concentration (Setting B) | exploratory only, COPY null, duplicates removed |
| reference repo is the upstream, no few-shot code, 3D assets absent (reference) | Model 2 = TabularNet + contrast objective; no 3D port; no data added |
| gen14/gen15 deployed choices were outer-selected (splits) | confirmation on withheld seeds for every registered claim |
| stopping on inner adjacent MAE halts at initialisation (smoke run) | stop on the inner contrast loss, minimum 20 epochs; adjacent MAE recorded |

## 7. What the adversarial review changed (draft 2 → draft 3)

Four reviewers (leakage, statistics, chemistry, code) and a refutation pass on each finding
(`manifests/review_reports.json`). Confirmed and fixed before sealing:

| finding | change |
|---|---|
| the registered arm rule could not execute: no inner score of the baseline was recorded | the inner selector scores `G14@BLUP20`, `FLAT@BLUP20`, `LINE`, `PROJ` on the same inner ladder and records `mstar_inner`; the report script has no override |
| the per-pair no-closure rule leaves most adjacent-query degrees of freedom fixed by the supports at high k; a laboratory reads log D per metal | the metal model is registered; the pair model is exploratory with `lab_known` and `PROJ` |
| inner selection scored cells whose residuals were in its covariance | nested covariance from the inner-training cells only; provenance asserted on every covariance record |
| the neural model early-stopped on an unmasked split and trained on 3/4 of the fold | masked validation split, then refit on every training cell for the selected epochs; stage-1 cells recorded |
| `FULL_FT` ran with dropout active and an unseeded RNG | eval mode, seeded per cell and member |
| `NNC@SELECTED` silently fell back to another mechanism | the selected mechanism is computed wherever the column exists; per-mechanism penalties |
| the noise constant was contaminated by decade-level double entries (implied SF noise sd 0.43 > target sd 0.23) | robust estimate (replicate sd ≤ 0.5), inner-chosen factor, per-fold sanity bound |
| 95 % intervals only; pooled point with a per-unit p; Kish 11.7 quoted for pair rows | 95 and 98.75 % percentile and BCa; extractant-macro estimand with blocked bootstrap and wild-cluster-t; pair-weighted Kish printed per cohort |
| the gen13-adjacency secondary was never run | a separate labelled pass with `--query-name adjacent_gen13` |
| the test suite did not exercise the runner path | tests run the fold with query labels masked, check the output table, spy on the selector, and reproduce predictions in a subprocess |
| `--allow-unsealed` and `--max-cells` accepted for any label | smoke labels only |
| §1.5/§4.3 understated the departures from the reference | rewritten: the arm is the reference's control, trained differently |
