# Figure plan — what the paper should show, and which file proves it

Written after reading, in this order: `README.md`; `docs/gen5_levels_results_20260819_full4regimes.md`;
`docs/gen6_phase0_and_phase1_results_20260819.md`; `docs/gen6_phase2_results_20260819.md`;
`gen7_architecture_results_20260819.md`; `runs/gen7_architecture/decision_report.md`;
`gen8_architecture_results_20260820.md`; `runs/gen8_architecture/decision_report.md`;
`runs/gen9_shape/decision_report.md`; `runs/gen10_final/GEN10_FINAL_DECISION_REPORT.md`;
`runs/gen10_final/GEN10_ERROR_CEILING.md`; `runs/gen11_transfer/GEN11_DECISION_REPORT.md`;
plus the source modules `gen9/relative.py`, `gen9/train.py`, `gen8/kshot.py`, `gen10/adaptation.py`
and the metric/CSV/parquet outputs named in each section below.

---

## 0. The one thing a reader must take away

> A model that has never seen an extractant predicts its lanthanide distribution ratios
> with a macro MAE of ~1.0 log units, and *all* of molecular structure is worth only about
> 0.12 of that. One measurement on the new extractant is worth 0.38. The deployment unit is
> therefore not a prediction, it is a prediction plus a chosen measurement — and the paper's
> job is to (i) show how much a handful of measurements buys, (ii) show what the zero-shot
> model gets structurally wrong, (iii) show what fixed it, and (iv) bound what is left.

---

## 1. Primary scientific question

**Can a model usefully predict `log D` (lanthanide distribution ratio) for an extractant it has
never seen, and how does that change when a small number of measurements on that extractant
become available?**

The target is `log D` for a (extractant, metal, conditions) row. Evaluation holds out whole
Tanimoto-0.7 **chemotypes**, so the held-out ligand and everything within Tanimoto 0.7 of it
is absent from training. The reported unit is macro MAE: one vote per ECFP cluster
(131 units), mean over 5 split seeds.

Supporting sources: `runs/gen7_architecture/leaderboard_all.csv` (cohort:
5,248 rows / 152 extractants / 131 ECFP clusters / 79 chemotypes);
`runs/gen9_shape/recomposed/oof_GEN9_SHAPE_RECOMPOSED.parquet` (same cohort, verified);
`runs/gen10_final/reproduction/phase0.json` (cohort fingerprint `bed178ec1a7a82b0`,
1,176 curves / 7,207 memberships, zero curve straddles a fold boundary).

## 2. Main methodological contributions

| # | contribution | where it lives | evidence |
|---|---|---|---|
| M1 | **Deployment protocol**: a held-out ligand's rows are split into a candidate **pool** and a disjoint **evaluation set**; an acquisition policy picks `k` support points from the pool; a calibrator adapts the frozen global model; the score is on the evaluation rows only. | `gen8/kshot.py` (P1/P2), `gen8/protocols.py` | `runs/gen10_final/final_locked/kshot_detail.parquet` (12,834,720 rows: model × adapter × policy × k × ligand × seed × repeat) |
| M2 | **Relative-position representation + mean-preserving recomposition**: five columns saying where a row sits inside its own titration window, used by a second forest trained on curve-centred targets, recomposed onto the monolith's curve mean. | `gen9/relative.py`, `gen9/train.py::ShapeRecomposed` | `runs/gen9_shape/shape/shape_by_axis.csv`, `runs/gen9_shape/shape/curve_shape.parquet` |
| M3 | **Series-local few-shot adaptation with a marginal-likelihood prior** (`SERIES_ML`): the hierarchical prior's scale is estimated by integrating the coefficients out, instead of taking the spread of already-shrunk ridge estimates. | `gen10/adaptation.py` | `runs/gen10_final/adaptation/summary.csv`, `bootstrap_vs_offset_k3.csv` |
| M4 | **Oracle-anchored error budget**: successively substituting true per-ligand, per-series, per-curve levels and the true per-curve slope, so that the removable and irreducible parts of the error are separated. | `scripts/gen10_error_decomposition.py` | `runs/gen10_final/error_decomposition/{decomposition.csv,summary.json}` |

## 3. Strongest quantitative result

**The few-shot frontier on 99 held-out extractants (common cohort), 5 split seeds × 12 repeats:**

`k = 0 → 1.0358`, `k = 1 → 0.6539`, `k = 2 → 0.5593`, `k = 3 → 0.4928`, `k = 5 → 0.4405` macro MAE.

Independently recomputed from `runs/gen10_final/final_locked/kshot_detail.parquet`; agrees with
`final_locked/frontier_best.csv` to all printed digits (see `METRIC_AUDIT.md`, row F1).
The first measurement alone removes 0.382 log units — larger than the sum of every zero-shot
modelling gain from gen2 to gen10 (1.0605 → 1.0358 = 0.025 for gen9's representation change;
0.163 for the gen6 coverage expansion, on a different arm).

→ **Figure 2.**

## 4. Main failure mode discovered

**Amplitude compression.** On extractant titrations of held-out ligands the frozen model draws an
almost flat line: median predicted slope **0.116** against a measured **2.574**, span recovery
**0.051**, within-curve Spearman **0.582** (775 curves / 25 ligands / 5 seeds,
`runs/gen9_shape/shape/shape_by_axis.csv`). Pooled MAE hides this completely, because a flat line
through the curve's mean is the MAE-optimal answer when the shape is unknown.

The diagnosis is a coordinate-system error, not a capacity problem: a model trained on **absolute**
conditions is never told where a row sits inside *its own* measurement window, and different
titrations span different windows, so the conditional mean of the curve-centred response is ≈ 0
everywhere. A shape-only control (level deleted) recovers **2.5 %** of the range — *worse* than the
monolith — which rules out level/shape competition (`gen9/relative.py` docstring;
`runs/gen9_shape/decision_report.md`).

→ **Figure 3.**

## 5. What fixes it

Five relative-position columns + mean-preserving recomposition. Same learner, same folds, same
frozen feature blocks; 5/5 seeds:

| extractant axis | frozen (gen8) | + relative position | + recomposition |
|---|---|---|---|
| median predicted slope (measured 2.574) | 0.116 | 0.521 | **1.024** |
| span recovery | 0.051 | 0.210 | **0.423** |
| shape MAE | 0.665 | 0.567 | **0.469** |
| within-curve Spearman | 0.582 | 0.840 | **0.886** |
| macro MAE | 0.9807 | 0.9858 | **0.9695** |

Source: `runs/gen9_shape/shape/shape_by_axis.csv` (rows `REC_ecfp_plus_recovered`,
`GEN9_REL_MONOLITH`, `GEN9_SHAPE_RECOMPOSED`, `axis_label = extractant`);
macro from `runs/gen9_shape/*/leaderboard.csv` and `runs/gen7_architecture/leaderboard_all.csv`.
Publication- and chemotype-blocked intervals for the shape gain:
`runs/gen10_final/publication_sensitivity/intervals_by_scheme.csv`.

→ **Figure 3, panels C–E.**

## 6. What remains unresolved

1. **The zero-shot ligand level is essentially unpredictable from structure.** A model with *no*
   ligand information at all (metal + conditions only) scores 1.0995; the best ligand-aware model
   scores 0.9695. Every fingerprint, descriptor, 3D block and pretrained embedding together is
   worth ≈ 0.13 log units. (`runs/gen7_architecture/leaderboard_all.csv`.)
2. **Half the remaining error is a per-ligand constant** — 0.520 of 0.970 — of which one central
   measurement realistically removes 0.349. (`runs/gen10_final/error_decomposition/decomposition.csv`.)
3. **The relative-position mechanism is query-set dependent.** Two decoy candidate points 2–3
   decades outside the intended window shift the same fitted model's predictions by a median
   0.117 log units — larger than the 0.025 macro gain the mechanism buys.
   (`runs/gen10_final/query_consistency/summary_by_arm.csv`.) The frozen model shifts by exactly 0.
4. **0.103 of macro MAE is unexplained** by any of the six identified components.
5. **Data quality**: one ligand (TWE-24) contributes 0.021 of macro MAE on 30 rows; decade-shifted
   duplicate cells contribute 0.007; 15 structures have a name/structure mismatch and their rows
   carry an irreducible residual of 0.934 vs 0.582.
   (`runs/gen10_final/data_ceiling/`, `runs/gen9_shape/data_audit/`.)
6. **Transfer from actinide/other-metal archives is unfinished** — 5 of 34 pre-registered gen11
   arms have run, and the only completed auxiliary arm is a wash (−0.0037 transfer effect, of which
   −0.0038 is the design change it had to make). Not paper material yet.
   (`runs/gen11_transfer/GEN11_DECISION_REPORT.md`.)

→ **Figures 4 and 5**, plus supplementary S10/S15/S16.

---

## 7. Main-text figures

Each figure states the question it answers, the panels, the source files, the evaluation cohort and
the reason it is in the main text rather than the supplement.

### Figure 1 — Task, protocol and model
*Question:* what does the model see, what is held out, and why is measuring `k` points of a
held-out ligand at test time not leakage?

Schematic, no fitted numbers. Three lanes: (i) **training information** — rows of other chemotypes,
ligand structure, metal, conditions; (ii) **test-time support** — `k` rows of the held-out ligand
whose targets the calibrator is explicitly allowed to read, chosen by a target-blind policy from a
candidate **pool**; (iii) **held-out query targets** — the disjoint evaluation rows, never read.
Also shows the two-stage global model (monolith level + curve-centred shape + mean-preserving
recomposition) at the correct conceptual level, and the chemotype-blocked outer split.

*Sources for the wording (not for numbers):* `gen8/kshot.py` (P1/P2 protocols),
`gen8/protocols.py::make_p2_split`, `gen9/train.py::ShapeRecomposed`, `gen9/relative.py`,
`gen10/adaptation.py`, `runs/gen10_final/final_locked/pipeline_frontier.json`.

### Figure 2 — What a handful of measurements buys
*Question:* how fast does held-out error fall as `k` measured support points become available, and
which point should be measured?

* **A** — macro MAE vs `k ∈ {0,1,2,3,5}` for four *pre-specified deployment rules* evaluated on the
  identical 99-ligand common cohort and the identical support/query draws: measurement-only null
  (no model), gen8 rule, gen9 rule, gen10 frozen rule; plus the non-deployable oracle-acquisition
  bound as a dashed line.
* **B** — paired per-ligand improvement over zero-shot with chemotype-block bootstrap CIs, and the
  marginal value of the 1st, 2nd, 3rd and 4th–5th measurement.

*Source:* `runs/gen10_final/final_locked/kshot_detail.parquet` (all curves recomputed from it, so
every line is paired row-for-row), cross-checked against `final_locked/frontier_best.csv`,
`frontier_common.csv`, `kshot_bootstrap.csv`, and `runs/gen9_shape/frontier/table_b_common_cohort.csv`.
*Cohort:* 99 extractants that have ≥ 5 pool rows and ≥ 2 evaluation rows in every arm; 5 seeds ×
12 repeats.

### Figure 3 — The structural failure and its repair
*Question:* why can a model have a reasonable average error and still get the internal structure of
a held-out titration wrong?

* **A, B, C** — three measured extractant titrations (log D vs log₁₀[extractant]) with the frozen
  and repaired predictions overlaid: the **median** case, a **strong-improvement** case and a
  **residual-failure** case, selected by an explicit rule (§9).
* **D** — per-curve predicted vs measured slope, frozen → +relative position → +recomposition.
* **E** — paired per-curve span recovery and within-curve Spearman, same three stages.

*Source:* `runs/gen9_shape/shape/curve_shape.parquet` (per-curve, per-seed, per-model),
`runs/gen9_shape/curves/curve_membership.parquet`, and the three OOF parquets
(`phase1/oof_REC_ecfp_plus_recovered.parquet`, `relmono/oof_GEN9_REL_MONOLITH.parquet`,
`recomposed/oof_GEN9_SHAPE_RECOMPOSED.parquet`).
*Cohort:* 775 extractant-concentration curves, 25 extractants, 5 seeds.

### Figure 4 — Where the remaining error is
*Question:* is what is left an offset problem or a shape problem, and how much of each is removable?

* **A** — oracle cascade: macro MAE after substituting, in turn, the true per-ligand level, the true
  per-series level, the true per-curve level and the true per-curve slope. This is the ceiling any
  calibration can reach.
* **B** — the seven-component budget of the 0.970 macro MAE, with each component's *oracle-removable*
  and *realistically removable* share, and the intervention each one names.
* **C** — offset vs shape MAE across the methodological stages (no-ligand null → frozen → relative
  position → recomposition → k = 1 → k = 3), showing which component each step actually moves.

*Source:* `runs/gen10_final/error_decomposition/{summary.json,decomposition.csv}`,
`runs/gen7_architecture/leaderboard_all.csv`, `runs/gen9_shape/shape/shape_by_axis.csv`,
`runs/gen10_final/final_locked/kshot_detail.parquet`.

### Figure 5 — Generalisation and coverage
*Question:* how does error depend on how far the held-out ligand is from the training chemistry,
and does adding training chemistry causally fix it?

* **A** — macro MAE vs nearest-training-neighbour Tanimoto, binned with counts and
  chemotype-block bootstrap CIs, at `k = 0`, `k = 1` and `k = 3`. Zero-shot error rises steeply with
  distance; one measurement removes most of the distance dependence.
* **B** — the controlled coverage experiment (gen6 Experiment A): identical held-out rows, identical
  learner and folds, only the *training* mask differs — `BASE` (extractants with ≥ 10 condition
  cells) vs `EXPANDED` (≥ 3), with the row-count-matched and target-shuffled controls, broken down
  by the same distance bins.

*Source:* A — `runs/gen10_final/final_locked/kshot_detail.parquet` (`nn_train_tanimoto`);
B — `runs/gen6_expA_5seed/{ood_metrics.csv,arm_metrics.csv,contrast_summary.csv,gain_decomposition.csv}`.
*Comparability note:* both use the same 5,248-row / 152-extractant / 131-cluster / 79-chemotype
cohort, but panel B's learner is gen6's `MC_lig2d_ext_massaction`, not the frozen gen9/gen10 model,
so B is a within-panel contrast only and is **not** plotted on the same axis as A.

### Figure 6 (conditional main text) — Which measurement to make
*Question:* when only one experiment can be run, which candidate condition should it be?

Acquisition policies at `k = 1` and `k = 2` on the identical pool/evaluation draws: RANDOM,
FARTHEST, MEDOID/CENTRAL_THEN_SPREAD, D-OPTIMAL, model-uncertainty rules, learned rankers, and the
ORACLE bound. The scientific content is two-sided: geometric centrality is worth +0.10 at `k = 1`
(as much as the whole modelling ladder), **and** model uncertainty is useless for this because the
quantity that matters under offset calibration is `|r_i − median(r)|`, not `|r_i|`.

*Source:* `runs/gen10_final/acquisition/realised_summary.csv`, `realised_bootstrap.csv`,
`runs/gen8_architecture/decision_report.md` §1 Decision 2/3,
`runs/gen10_final/final_locked/kshot_detail.parquet` (policy axis).

**Recommendation:** keep Figure 6 in the main text *only if* the journal allows six; the
evidence is strong but it is a corollary of Figure 2 rather than an independent claim.
Ranked 6th of 6 in `FIGURE_PRIORITY.md`.

**Post-hoc amendment to this plan** (from the adversarial review, `FIGURE_AUDIT.md` §0):
Figure 5's distance-tercile panel turned out to be underpowered — the far − near contrast at
*k* = 0 is +0.35 [−0.04, +0.66] on 17 chemotypes per tercile. The figure was kept but the
claim was demoted to the point estimate, and `FINAL_FIGURE_REPORT.md` §4 recommends
restructuring Figure 5 so that the controlled coverage experiment leads.

---

## 8. Supplementary figures — as built

| id | file | content | source |
|---|---|---|---|
| S1 | `FigS1_dataset_composition` | Rows per lanthanide; rows per extractant (one holds 28 % of the corpus); cumulative share by chemotype (one holds 64 %); curve inventory by axis | `runs/gen9_shape/recomposed/oof_*.parquet`, `runs/gen9_shape/curves/curve_table.parquet` |
| S2 | `FigS2_query_robustness` | Prediction shift of the *same fitted model* under decoy / extension / nesting / thinning / densification / permutation of the candidate design — the negative result attached to Figure 3 | `runs/gen10_final/query_consistency/summary_by_arm.csv` |
| S3 | `FigS3_adapters` | Every calibration adapter at every *k* on a fixed 99-extractant cohort, plus paired intervals against `OFFSET_K3` | `runs/gen10_final/adaptation/{detail.parquet,bootstrap_vs_offset_k3.csv}` |
| S4 | `FigS4_architectures` | 24-arm zero-shot ablation: feature access, level+shape heads, residual, two-branch, set encoders, axis representations; and the same arms in (level, shape) space | `runs/gen10_final/headline_tables/t2_leaderboards.csv` |
| S5 | `FigS5_shape_by_axis` | Span recovery, shape MAE and within-curve Spearman on all six condition axes | `runs/gen9_shape/shape/shape_by_axis.csv` |
| S6 | `FigS6_seed_stability` | The frontier and the median predicted extractant slope, one line/point per split seed | `figures/derived/kshot_per_seed.csv`, `runs/gen9_shape/shape/curve_shape.parquet` |
| S7 | `FigS7_raw_example_curves` | The three Figure 3 examples plus their runners-up, on the raw log₁₀ *D* axis | as Figure 3 |
| S8 | `FigS8_diagnostics` | Predicted vs measured; residual vs prediction with binned mean; macro MAE by source-audit stratum | `runs/gen9_shape/recomposed/oof_*.parquet`, `runs/gen10_final/data_ceiling/stratified_metrics.csv` |
| S9 | `FigS9_dose_response` | Coverage gain against Tanimoto gained — the evidence that separates chemical transfer from a class prior | `runs/gen6_expA_5seed/oof_predictions.parquet` |
| S10 | `FigS10_budget` | Depth vs breadth vs new-chemotype-first at a fixed measurement budget | `runs/gen10_final/budget_simulation/budget_strategies.csv` |

Four candidates from the first draft were dropped, and why: a condition-space coverage map
(S4 in the draft) duplicates S1D without adding a claim; a per-metal performance breakdown
duplicates S5's `metal_series` row; a separate per-series paired scatter is already
Figure 2D; and a "remaining failure cases" gallery would be an unruled selection of
individual curves — Figure 3C and Figure S7 carry that role under a declared rule.

Explicitly **not** plotted (see `FINAL_FIGURE_REPORT.md` §5): the gen1–gen4 pair/`log SF`
benchmarks (different target, different cohort), gen5's `unseen_series` regime (26 % of its
CV groups cross a publication boundary), gen11 (5 of 34 arms), the learned-acquisition arms
as a positive result, the published best-of-policies frontier envelopes, and any
cross-generation macro MAE line that mixes the 91-extractant and 152-extractant cohorts.

## 9. Example-selection rule for Figure 3 (declared before looking at any curve)

Curves are ranked on the **frozen** model, restricted to `axis_label == "extractant"`,
`n_points ≥ 5`, and the split seed 104729 (the first seed, fixed in advance). Let
`Δshape = shape_mae(frozen) − shape_mae(recomposed)` per curve.

* **median case** — the curve whose `(shape_mae_frozen, Δshape)` pair is closest, in units of the
  respective interquartile range, to the joint median of both;
* **strong-improvement case** — among curves in the top decile of `Δshape`, the one closest to the
  median of that decile (not the maximum, which would be an outlier);
* **residual-failure case** — among curves in the *bottom* decile of `Δshape` (repair helps least or
  hurts), the one closest to that decile's median.

Ties broken by `curve_id` ascending. The rule, the ranks and the resulting `curve_id`s are written
to `figures/derived/fig3_selected_curves.csv` by the plotting script, so a reviewer can see which
curves the rule picked and what the alternatives were.

---

## 10. Statistical conventions used in every figure

* **Experimental unit** — the extractant (ligand). Rows within a ligand are not independent: one
  extractant contributes up to 1,488 rows and 100+ curves.
* **Resampling block** — the Tanimoto-0.7 chemotype, because that is what the outer folds hold out.
  79 chemotypes; 63.6 % of rows sit in one of them, which is why block resampling matters.
* **Bootstrap** — the repository's own `gen8.inference.paired_chemotype_bootstrap`
  (5,000 replicates, BCa and percentile, plus a `block_macro` value that votes once per chemotype).
  Figure scripts import it rather than reimplementing it.
* **Pairing** — every comparison in Figures 2, 3, 5A and 6 is computed on identical held-out units;
  in Figure 2 and 6 also on identical support/query draws, because `gen8.kshot.stable_hash` makes
  the pool/evaluation split a pure function of `(seed, repeat, ligand, n_rows)`.
* **Seeds** — 5 split seeds `{104729, 130363, 155921, 196613, 262147}`; model seed fixed at 42 and
  independent of the split seed. `k`-shot arms additionally average 12 repeats of the pool draw.
* **What error bars mean** — always stated in the caption: chemotype-block bootstrap 95 % CI for
  means and paired differences; interquartile range for per-curve distributions.
