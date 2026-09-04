# GEN10 final model card

*The frozen pipeline, what it takes in, what it gives back, and where it is known
to fail. Exact configuration: `final_locked/pipeline_frontier.json`. Every number
is from `final_locked/` unless a phase directory is named.*

## 1. Identity

| | |
|---|---|
| global model | `GEN9_SHAPE_RECOMPOSED` — gen9's mean-preserving, design-relative recomposition, unchanged and bit-reproduced inside gen10's `RecomposedModel(representation_name="GEN9", gen9_compat=True)` |
| adaptation | k = 1: gen8 `SLOPE_L_s1_K1`; k ≥ 2: **`SERIES_ML`** (new in gen10) |
| first point | `CENTRAL_THEN_SPREAD` (medoid of the standardised condition axes, then spread) |
| lineage | gen7 `REC_ecfp_plus_recovered` (level) → gen9 `SHAPE_RECOMPOSED` (shape) → gen10 corrected series-local adapter |
| cohort | `runs/gen7_architecture/cache/cohort.parquet`, fingerprint `bed178ec1a7a82b0`, 5,248 rows / 152 extractants / 79 Tanimoto chemotypes |
| evaluation | `seeded_group_kfold` over Tanimoto chemotype, 5 folds × 5 split seeds (104729, 130363, 155921, 196613, 262147); model seed `42 + fold·1009 + 9 999 991` |
| selection | Phase 11 locked evaluation under the stopping rule fixed in the brief; see `GEN10_FINAL_DECISION_REPORT.md` §2 |

**What gen10 changed, and what it did not.** The global model is gen9's. gen10
tested five alternatives to it (Phase 2–4) and one — a rank/spacing-based
representation — improved the extractant shape by 0.075 shape-MAE at neutral
macro, but did not meet any of the four inclusion criteria (it is reported in
the decision report and kept in the registry as `GEN10_REC_HYBRID`). The adapter
*is* new: the marginal-likelihood series-local prior beats gen8's `OFFSET_K3` by
0.025–0.038 at k = 2–5 with every seed positive.

## 2. Inputs

**Per row (one requested measurement):** the extractant's canonical SMILES
(2,048-bit ECFP), the metal (atomic number, lanthanide index, ionic radius), the
64 condition columns (acid identity and molarity, extractant molarity, diluent,
additive, metal concentration, temperature, contact time), the eight MASSACTION
columns (log10 concentrations and `n·log[L]` products — recomputed from the
conditions, never supplied separately), and the recovered experimental variables
(`rec__*`: solvent physics, phase modifier, shaking time; missing values are
median-imputed with a missingness indicator, fitted on the training fold).

**Per request (the design):** the *whole list* of conditions being asked about.
The model rebuilds the titration curves that list decomposes into (one series ×
one varying axis, `gen10.querycurves.query_membership`, identical to gen9's
cohort-wide table on every closed partition) and positions every point inside
its own curve with gen9's five columns: position in the window `[0, 1]`, signed
offset from the window centre, window width, point count, is-endpoint. Four of
the five are functions of the window's **endpoints**; that is the source of the
query-set dependence measured in Phase 1 (§6).

**Never an input:** any `log D` of the requested rows. `predict` takes a frame of
conditions; the self-audit corrupts every held-out target and the prediction is
unchanged at the 1e-15 floor.

## 3. Architecture

1. **Level.** The frozen ExtraTrees monolith (400 trees, `max_features = 0.30`,
   `min_samples_leaf = 2`, cluster-balanced weights, prediction clipped to
   1.5× the training range) predicts each row; its predictions are averaged over
   each requested curve. This level is bit-identical to gen7's (offset MAE
   0.8212 on every recomposed arm).
2. **Shape.** A second ExtraTrees forest of the same family, fitted on the
   curve-centred training target `y − mean_curve(y)` with the five relative
   columns appended, predicts each row; the prediction is re-centred to zero mean
   over the requested curve.
3. **Recomposition.** `ŷ_i = mean_curve(level) + (shape_i − mean_curve(shape))`.
   Rows on no curve (fewer than three distinct points on any axis) keep the
   monolith's prediction.

## 4. The frozen k-shot rule

| k | first point | adapter | notes |
|---|---|---|---|
| 0 | — | zero-shot | |
| 1 | central-then-spread | `SLOPE_L_s1_K1` — gen8's mean-preserving slope repair + offset | one point supplies the level |
| 2, 3, 5 | central-then-spread | `SERIES_ML` — gen9's series-local design, penalties by marginal likelihood | ridge on `[1 | centred series indicators | standardised acid, extractant, lanthanide]`, intercept free; per-family penalty `σ²/τ²` with `(σ, τ_series, τ_response)` the maximisers of the marginal likelihood of the fold's training ligands' out-of-fold residuals (coefficients integrated out, intercept a fixed effect) |

The rule is the same at every seed and for every ligand; no arm is chosen per
case. At k = 1 every ridge adapter is algebraically identical (`β = [r, 0, …]`),
verified at 3e-14.

## 5. Performance

Common cohort (99 ligands with ≥ 5 pool rows in every repeat of every seed —
gen8's Table B), macro MAE, lower is better (`final_locked/frontier_best.csv`):

| k | gen8 | gen9 | **gen10 frozen pipeline** | Δ vs gen9 | best deployable arm in the table if different |
|---|---|---|---|---|---|
| 0 | 1.0605 | 1.0358 | **1.0358** | 0 | `GEN10_REC_HYBRID` 1.0313 |
| 1 | 0.6674 | 0.6539 | **0.6539** | 0 | `GEN10_REC_GEN9_PLUS` 0.6489 |
| 2 | 0.5879 | 0.5746 | **0.5593** | −0.015 | `GEN10_REC_GEN9_PLUS` 0.5548 |
| 3 | 0.5230 | 0.5109 | **0.4928** | −0.018 | `GEN10_REC_GEN9_PLUS` 0.4887 |
| 5 | 0.4743 | 0.4675 | **0.4405** | −0.027 | `GEN9_REL_MONOLITH` + `SERIES_ML` 0.4341 |

k = 0 and k = 1 are gen9's numbers reproduced exactly; the gain at k ≥ 2 is the
adapter. The "best arm" column differs from the pipeline by 0.003–0.006 and is
not frozen, by the no-cherry-picking rule.

Zero-shot, FROZEN cohort, five seeds (`axis_representation/shape/shape_by_axis.csv`):

| | frozen | **gen9 / gen10 frozen** | `GEN10_REC_HYBRID` (reported, not frozen) |
|---|---|---|---|
| macro MAE | 0.9807 | **0.9695** | 0.9696 |
| extractant slope, median (measured 2.574) | 0.116 | **1.024** | 1.647 |
| extractant slope MAE | 2.433 | **1.510** | 1.230 |
| extractant shape MAE | 0.665 | **0.469** | 0.394 |
| extractant span recovery | 0.051 | **0.423** | 0.615 |
| within-curve Spearman | 0.582 | **0.886** | 0.886 |
| within-curve sign accuracy | 0.762 | **0.926** | 0.927 |
| acid shape MAE | 0.589 | **0.545** | 0.511 |
| lanthanide shape MAE | 0.298 | **0.291** | 0.306 |

Strata, FROZEN cohort, macro MAE (`final_locked/strata.csv`): `RESIDUAL_SHAPE`
1.302 (frozen 1.353), `PURE_LEVEL` 1.885 (1.886), `ALREADY_GOOD` 0.543 (0.550),
mismatch rows 1.000 (1.030), clean rows 0.929 (0.938). QUARANTINED and CORRECTED
cohorts move macro MAE by < 0.001 (`final_locked/macro_bootstrap.csv`).

## 6. Known limits — measured, not guessed

* **The answer depends on the question.** Adding two candidate points two
  decades outside the measured window moves the median prediction of the
  *unchanged* points by 0.12 log units (p95 0.32, worst 0.68); extending one
  boundary by 0.5–2 decades moves them 0.07; dropping an end point 0.05
  (`query_consistency/summary_by_arm.csv`, 5 seeds, 66–130 ligands). The frozen
  model reads no design and moves by exactly zero. This is the price of the
  shape gain, and gen10 found no representation that removes it — the
  rank/spacing alternatives are more robust to decoys and less robust to
  interior changes. **Report the design with every prediction, and do not
  pad a plan with points you will not run.**
* **The level of an unseen ligand is not predictable from structure.** 0.52 of
  the 0.97 macro MAE is a per-ligand constant (`GEN10_ERROR_CEILING.md`). One
  measurement removes 0.35 of it. Do not use the zero-shot level for a decision.
* **Shape is learned from 25 ligands with extractant titrations, from 32
  publications and 8 chemotypes.** The gain is positive under publication-blocked
  and two-factor bootstraps (`publication_sensitivity/`), but the chemotype
  coverage is thin and the chemotype-blocked interval is the optimistic one.
* **Rows whose recorded extractant name does not match the modelled structure**
  (271 rows, 15 ligands) carry an irreducible residual 0.35 log units larger than
  consistent rows; the model is predicting a mixture from one component.
* **Three ligands flagged as level outliers and one suspected transcription error
  (TWE-24, +4.8 decades) are retained**; a same-cell peer from another publication
  predicts those rows three times better than the model.
* **Not a pair-selectivity model.** It predicts `log D` per metal; separation
  factors are differences of two predictions and inherit both errors.

## 7. Reproducibility

Same configuration twice in one process: max |Δ| ≤ 1.5e-15 (thread-order floor).
Fresh subprocesses under `PYTHONHASHSEED` 0 / 1 / 4242: identical at the same
floor. Splits are BLAKE2b. Every gen10 arm's determinism probe is in
`<stage>/determinism.csv`; the full record is `self_audit/self_audit.json`
(17 checks) and `reproduction/phase0.json` (12 checks).
