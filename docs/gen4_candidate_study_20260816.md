# Gen4 candidate study — can the A2 champion be improved on unseen extractants?

_Date: 2026-08-16/17. Exploratory study on the frozen gen3 outer folds
(`runs/gen3_primary_20260815T181555Z`); **not** a frozen-protocol run.
Selection on split seed 104729; confirmation on 130363/155921/196613/262147._

## TL;DR

* Five improvement options were proposed from the gen3 diagnostics; the three
  judged best (transitive projection, explicit selectivity-scale modelling
  incl. its stacked "prior" variant, extended 2D ligand descriptors) plus the
  hierarchical decomposition were implemented, tested, and evaluated on the
  exact gen3 leave-extractants-out folds against a hash-verified A2 refit.
* **Confirmed improvement: transitive projection.** Post-hoc least-squares
  projection of the pair predictions inside each (ligand, condition) cell onto
  per-metal scores lowers macro MAE 0.3188 → 0.3172 (−0.5%), better in **5/5
  seeds** and 23/34 extractants (extractant-bootstrap CI on the confirmation
  seeds [+0.0001, +0.0037]). Small, free, label-free, mathematically exact.
* **Not confirmed:** the seed-104729 gains of PRIOR (−0.010), A2+descriptors
  (−0.005) and SCALE_s50 (−0.011) did not replicate: on the four confirmation
  seeds PRIOR is neutral (Δ = 0.000), A2_lig2d is worse (Δ = −0.011) and
  SCALE_s50 is significantly worse (Δ = −0.033, CI excludes 0).
* **Best overall profile, unconfirmed on macro:** `PRIOR_TP` (A2 + cross-fitted
  scale-prior features + projection): macro 0.3145 vs 0.3188 (−1.3%, better in
  4/5 seeds, CI spans 0) and clearly better pooled metrics (pooled MAE 0.405 vs
  0.420, pooled R² 0.37 vs 0.32, within-pair R² −0.03 vs −0.06). It is the
  natural champion candidate for the next frozen-protocol run.
* **Strong negative finding:** the per-ligand selectivity *scale* — the thing
  the oracle-offset analysis says is worth 22% of macro MAE — is **not
  predictable from 2D ligand descriptors across these 34 ligands**: the
  cross-fitted scale forest's predictions correlate r = −0.04 with the true
  cell scales on unseen ligands (A2's implied scale: r = 0.34). The remaining
  error is a data problem (per-ligand level), not a learner problem; k-shot
  calibration (5–10 labelled pairs per new ligand → pooled R² 0.42–0.46) is
  the realistic deployment route.

## 1. The five options and what the evidence said

| # | Option | Rationale from the gen3 analysis | Outcome |
|---|---|---|---|
| 1 | **Transitive projection** (per (ligand, condition) LS onto `s_A − s_B`) | Truth is exactly transitive (log SF = log D_A − log D_B); tree predictions are not. Pre-screen on gen3 OOF: 0.3192 → 0.3175, 5/5 seeds. | **Confirmed** (5/5 seeds, 23/34 extractants) |
| 2 | **Hierarchical level + deviation** (level model on (ligand, pair) cell means + deviation model on within-cell residuals) | TODGA's condition variation is anti-predicted within pairs (R² ≈ −1.2); decoupling level from condition sensitivity should transfer. | Worse on macro (0.324 vs 0.316 on seed 104729) though better pooled/within-pair R²; not carried to confirmation |
| 3 | **Selectivity-scale model** `y = s(ligand,cond)·t(pair) + r` with `s` learned from descriptors (cross-fitted); stacked variant **PRIOR** = A2 + (`s`, `s·|t|`) features | Error is a per-ligand magnitude problem (oracle offset: 0.319 → 0.247). | SCALE: worse (scale not predictable). PRIOR: neutral macro, better pooled/calibration; PRIOR_TP best overall profile |
| 4 | **Extended 2D ligand descriptors** (168 RDKit + 38 hand-crafted DGA/amide) | ECFP cannot see alkyl-chain length; 34 ligands = 28 ECFP clusters. | Worse on average (0.327 vs 0.319; high split-seed variance 0.015): trees over-split on ligand identity |
| 5 | Model-seed ensemble of A2 (control) | Local vs cluster refits differ ~0.004 at fold level. | No gain (0.3161 vs 0.3160 on seed 104729) |

Rejected at pre-screen without refitting: blending A2 with CatBoost (≤ 0.001), shrinking toward pair means (hurts macro).

## 2. Confirmation-run results (`runs/gen4_candidates_confirm_5seeds/`)

Per-seed equal-extractant macro MAE (A2 params and seeds per fold identical to the run; A2_refit is the platform-matched baseline, A2_run the cluster's numbers):

| arm | 104729 (sel.) | 130363 | 155921 | 196613 | 262147 | mean | Δ vs A2_refit | + seeds |
|---|---|---|---|---|---|---|---|---|
| A2_refit | 0.3160 | 0.3102 | 0.3144 | 0.3220 | 0.3313 | 0.3188 | — | — |
| **A2_refit_TP** | 0.3155 | 0.3078 | 0.3123 | 0.3209 | 0.3294 | **0.3172** | **+0.0016** | **5/5** |
| PRIOR | 0.3062 | 0.3114 | 0.3182 | 0.3185 | 0.3299 | 0.3168 | +0.0019 | 3/5 |
| **PRIOR_TP** | 0.3042 | 0.3095 | 0.3153 | 0.3165 | 0.3269 | **0.3145** | **+0.0043** | 4/5 |
| A2_lig2d | 0.3112 | 0.3324 | 0.3233 | 0.3179 | 0.3497 | 0.3269 | −0.0081 | 2/5 |
| PRIOR_lig2d | 0.3141 | 0.3332 | 0.3283 | 0.3216 | 0.3469 | 0.3288 | −0.0100 | 2/5 |
| SCALE_s50 | 0.3071 | 0.3477 | 0.3510 | 0.3371 | 0.3739 | 0.3433 | −0.0246 | 1/5 |
| A2_run (cluster) | 0.3164 | 0.3103 | 0.3157 | 0.3231 | 0.3306 | 0.3192 | −0.0004 | 1/5 |
| PAIRMEAN baseline | 0.4530 | 0.4507 | 0.4438 | 0.4486 | 0.4447 | 0.4482 | −0.129 | 0/5 |

Decision rule (pre-declared in `scripts/gen4_decision.py`): lower 5-seed mean AND positive Δ in ≥ 3/4 confirmation seeds AND extractant-unit paired-bootstrap 95% CI on the confirmation seeds excluding 0, Holm-corrected across the frozen candidate set {PRIOR, PRIOR_lig2d, A2_lig2d, SCALE_s50}. **No candidate family passes.** Transitive projection is a uniform post-processing step (reported for every arm, never used to choose between families): on the confirmation seeds Δ = +0.0019, 4/4 seeds, CI [+0.0001, +0.0037], p = 0.98, 23/34 extractants; over all five seeds CI [−0.0001, +0.0034].

Secondary metrics (5-seed means):

| arm | pooled MAE | pooled R² | median extractant R² | Pearson² | dispersion | within-(fold×pair) R² | sign acc |
|---|---|---|---|---|---|---|---|
| A2_refit | 0.420 | 0.322 | 0.539 | 0.370 | 0.744 | −0.059 | 0.846 |
| A2_refit_TP | 0.419 | 0.326 | 0.543 | 0.372 | 0.742 | −0.056 | 0.846 |
| PRIOR | 0.406 | 0.366 | 0.539 | 0.400 | 0.729 | −0.030 | 0.853 |
| PRIOR_TP | 0.405 | 0.369 | 0.544 | 0.405 | 0.732 | −0.027 | 0.850 |
| A2_lig2d | 0.411 | 0.361 | 0.432 | 0.398 | 0.677 | +0.077 | 0.855 |
| SCALE_s50 | 0.484 | 0.067 | 0.460 | 0.236 | 0.806 | −0.279 | 0.767 |
| PAIRMEAN | 0.383 | 0.445 | 0.234 | 0.445 | 0.692 | 0.000 | 0.850 |

Reading: PRIOR(_TP) is the only arm that improves the pooled/calibration family *without* paying on macro; the descriptor arm buys within-pair R² at the price of the equal-extractant metric (it shrinks predictions, dispersion 0.68); SCALE breaks calibration.

## 3. Why the scale/descriptor ideas failed (diagnostics)

* `scale_diag.py` on seed 104729: across 220 held-out (ligand, condition) cells the cross-fitted scale forest's prediction has r = −0.04 with the true through-origin slope of `y` on the pair trend (MAE 0.88 on a scale with sd 0.99); A2's *implied* slope has r = 0.34. The scale forest does correlate with A2's implied slope (r = 0.57), i.e. it learns what A2 already knows and nothing more. → The transferable ligand → selectivity-magnitude signal in 2D descriptors is exhausted by A2.
* `PRIOR_TRENDONLY` (A2 + `|t(pair)|` only, no ligand-derived scale) = A2 on the two seeds run (0.3165/0.3120 vs 0.3160/0.3102): the pair-trend feature carries nothing beyond the pair block; PRIOR's seed-104729 gain came from the ligand-derived scale features and was fold-composition luck.
* SCALE without shrinkage (0.349) → s50 (0.305) → s75 (0.314) → s100 (0.324) on seed 104729: non-monotonic and within fold jitter once the noise of the scale prediction is understood.
* Extended descriptors: −0.005 on seed 104729, +0.022 on 130363; split-seed SD 0.015 vs 0.008 for A2. Continuous, ligand-unique columns let ExtraTrees separate ligands early; for a held-out ligand that means averaging over fewer, less relevant neighbours. The shuffled-descriptor twin (`A2_lig2d_SHUF`) is wired but its first run was invalid (descriptors were not attached — fixed in the harness) and was not re-run because the real arm did not confirm.

## 4. What was built (all tested; independent 3-lens adversarial review found no leakage or correctness defects)

* `src/lanthanide_separation/gen4_candidates.py` — `transitive_projection`, `HierarchicalPairRegressor`, `ScaleTrendPairRegressor` (cross-fitted scale, optional shrinkage), `PriorAugmentedPairRegressor` (`mode="scale"|"trend_only"`), `attach_ligand_descriptors`. Exactly antisymmetric; fitted on training frames only; reproducible (a thread-order nondeterminism in the cell-level forests that leaked into residual targets was found and fixed — the confirmation run's SCALE/PRIOR numbers were produced before that fix and carry ±0.002 jitter, which does not affect any conclusion).
* `src/lanthanide_separation/ligand_descriptors.py`, `scripts/build_ligand_descriptors.py` → `dataset with 3D structures/ligand_2d_descriptors.parquet` (190 ligands × 206 columns) + manifest.
* `scripts/run_gen4_candidates.py` — evaluation on the exact gen3 outer folds (folds and per-fold A2 params from the run; identical seed stream; hash-identical cohort/columns), OOF, leaderboard, per-extractant table, extractant-unit paired bootstrap, pair-mean baseline row, within-pair and pair-level R², `_TP` twins for every arm, attribution controls.
* `scripts/gen4_decision.py` — the pre-declared decision rule.
* Tests: `tests/test_gen4_candidates.py` (9), `tests/test_ligand_descriptors.py` (9, RDKit env). Full suite: 146 passed, 1 skipped.
* Runs: `runs/gen4_candidates_seed104729_v1`, `_v2` (selection), `runs/gen4_candidates_confirm_5seeds` (confirmation, `decision_table.csv`), `runs/gen4_candidates_controls_5seeds` (partial: PRIOR_TRENDONLY on 2 seeds).

## 5. Recommendations

1. Adopt transitive projection as a standard post-processing step of the champion (report `A2 + TP`); it is exact, label-free and improved every seed.
2. Put `PRIOR_TP` (and `PRIOR`) into the next frozen-protocol run as the pre-registered challengers, with the label-shuffle and shuffled-scale controls; do not claim it now.
3. Stop spending effort on richer *ligand* descriptors for the unseen-extractant regime — three descriptor families (3D geometry, xTB electronic, extended 2D) have now failed to move the equal-extractant metric, and the scale diagnostic explains why. The productive directions are (a) k-shot per-ligand calibration as a deployment protocol, (b) the seen-extractant / unseen-conditions regime (R² 0.74), and (c) more ligands, not more descriptors.
4. Report pooled metrics next to the leave-fold-out pair-mean baseline (0.445 pooled R²) in every leaderboard.
