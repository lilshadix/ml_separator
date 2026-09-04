# Metric audit

**Every quantity plotted in a main-text or supplementary figure was recomputed from the
rawest artefact available and compared against the value the repository already reports.
67 of 67 checks PASS.** The audit is a script, not a claim:

```bash
.venv/bin/python figures/scripts/verify_metrics.py
```

It writes `figures/derived/metric_audit.csv`, from which the tables below are rendered.
Tolerances are stated per group; where a report rounds to four decimals the tolerance is
5 × 10⁻⁵, and where a machine-readable artefact is compared it is 10⁻⁹ or tighter.

Two figure scripts refuse to draw if their own reproduction check fails:
`prepare_error_budget.py` (the oracle cascade must reproduce
`runs/gen10_final/error_decomposition/summary.json` to 5 × 10⁻⁹ before Figure 4 is drawn)
and `plot_fig5_generalization.py` (all fifteen distance-tercile marginal gains must
reproduce `runs/gen10_final/budget_simulation/marginal_gains.csv` to 10⁻⁹).
`plot_fig6_acquisition.py` likewise refuses unless all twenty policies reproduce
`runs/gen10_final/acquisition/realised_summary.csv`.

---

## 0. What the prompt's hint values turned out to be

| hinted | authoritative | where it comes from | status |
|---|---|---|---|
| few-shot frontier 1.036 / 0.654 / 0.559 / 0.493 / 0.441 | **1.0358 / 0.6539 / 0.5593 / 0.4928 / 0.4405** | recomputed from `kshot_detail.parquet`; 99-extractant common cohort, 5 seeds × 12 repeats | confirmed |
| extractant slope 0.116 → 0.521 → 1.024 | **0.116428 → 0.521045 → 1.024014** (measured 2.573675) | recomputed from `curve_shape.parquet`, median over 775 curve × seed evaluations | confirmed |
| span recovery 0.051 → 0.210 → 0.423 | **0.050718 → 0.209530 → 0.423291** | same; this is the *unguarded* median (see §3) | confirmed |
| shape MAE 0.665 → 0.469 | **0.664815 → 0.469194** (via 0.567180) | same; mean over curve × seed | confirmed |
| within-curve Spearman 0.582 → 0.886 | **0.582371 → 0.885628** | same; **mean**, not median | confirmed, statistic clarified |
| dataset expansion ≈ 0.163 macro MAE on identical rows | **+0.163284**, BCa 95 % CI [+0.0125, +0.2953], 5/5 seeds, 79/131 units improved | `runs/gen6_expA_5seed/contrast_summary.csv`, recomputed levels from `hard_chemistry_metrics.csv` | confirmed, **and it deserves a main-text panel — with the qualification in §5** |

---

## 1. Cohorts in play, and which figure uses which

Mixing these is the single largest risk in this repository, because five generations share
a corpus but not an evaluation unit.

| id | definition | size | used by |
|---|---|---|---|
| **C-FULL** | every held-out row of the frozen cohort | 5,248 rows / 152 extractants / **131 ECFP clusters** / 79 chemotypes | Fig 4B, Fig S1, Fig S4, Fig S5, gen6 panels of Fig 5 |
| **C-KSHOT** | extractants for which the k-shot harness has arms | 143 extractants | Fig 6, Fig S8 strata are on C-FULL |
| **C-COMMON** | ≥ 5 candidate-pool rows and ≥ 2 evaluation rows in **every** arm at **every** k | **99 extractants**, 41 chemotypes | Fig 2, Fig 4A, Fig 4C, Fig 5A, Fig S3, Fig S6 |
| **C-CURVE** | extractant-concentration curves with ≥ 4 points | 155 curves / 25 extractants (775 curve × seed) | Fig 3, Fig S5, Fig S7 |

Macro MAE means **one vote per ECFP cluster** on C-FULL and **one vote per extractant** on
C-KSHOT / C-COMMON. The two are not interchangeable: the same frozen model scores 0.9695
on C-FULL and 1.0358 on C-COMMON. No figure places both on one axis; Figure 4A recomputes
the oracle cascade on C-COMMON precisely so that oracle bounds and achieved k-shot values
share one axis (§4).

Split seeds are `{104729, 130363, 155921, 196613, 262147}` throughout; the model seed is
fixed at 42 and independent of the split seed.

---

## 2. The audit table

### Figure 2 — the few-shot frontier

| quantity | source file | experiment | cohort | seeds | aggregation | n units | recomputed | reported | Δ | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| frozen-pipeline macro MAE at k=0 | `runs/gen10_final/final_locked/kshot_detail.parquet` | gen10 Phase 11 final_locked | 99-extractant common cohort | 5 split seeds x 12 repeats | mean over extractants of mean over (seed, repeat) | 99 | 1.0358 | 1.0358 | 5.6e-07 | **PASS** |
| frozen-pipeline macro MAE at k=1 | `runs/gen10_final/final_locked/kshot_detail.parquet` | gen10 Phase 11 final_locked | 99-extractant common cohort | 5 split seeds x 12 repeats | mean over extractants of mean over (seed, repeat) | 99 | 0.653873 | 0.6539 | 2.7e-05 | **PASS** |
| frozen-pipeline macro MAE at k=2 | `runs/gen10_final/final_locked/kshot_detail.parquet` | gen10 Phase 11 final_locked | 99-extractant common cohort | 5 split seeds x 12 repeats | mean over extractants of mean over (seed, repeat) | 99 | 0.559299 | 0.5593 | 1.2e-06 | **PASS** |
| frozen-pipeline macro MAE at k=3 | `runs/gen10_final/final_locked/kshot_detail.parquet` | gen10 Phase 11 final_locked | 99-extractant common cohort | 5 split seeds x 12 repeats | mean over extractants of mean over (seed, repeat) | 99 | 0.492809 | 0.4928 | 9.0e-06 | **PASS** |
| frozen-pipeline macro MAE at k=5 | `runs/gen10_final/final_locked/kshot_detail.parquet` | gen10 Phase 11 final_locked | 99-extractant common cohort | 5 split seeds x 12 repeats | mean over extractants of mean over (seed, repeat) | 99 | 0.440542 | 0.4405 | 4.2e-05 | **PASS** |
| REC_ecfp_plus_recovered + SLOPE_L_s1_K3 at k=2 | `runs/gen10_final/final_locked/kshot_detail.parquet` | gen10 rerun vs gen9's own frontier table | 99-extractant common cohort | 5 split seeds x 12 repeats | mean over extractants | 99 | 0.590884 | 0.590884 | 0.0e+00 | **PASS** |
| REC_ecfp_plus_recovered + SLOPE_L_s1_K3 at k=3 | `runs/gen10_final/final_locked/kshot_detail.parquet` | gen10 rerun vs gen9's own frontier table | 99-extractant common cohort | 5 split seeds x 12 repeats | mean over extractants | 99 | 0.522994 | 0.522994 | 1.1e-16 | **PASS** |
| REC_ecfp_plus_recovered + SLOPE_L_s1_K3 at k=5 | `runs/gen10_final/final_locked/kshot_detail.parquet` | gen10 rerun vs gen9's own frontier table | 99-extractant common cohort | 5 split seeds x 12 repeats | mean over extractants | 99 | 0.477241 | 0.477241 | 0.0e+00 | **PASS** |
| GEN9_SHAPE_RECOMPOSED + SLOPE_L_s1_K3 at k=2 | `runs/gen10_final/final_locked/kshot_detail.parquet` | gen10 rerun vs gen9's own frontier table | 99-extractant common cohort | 5 split seeds x 12 repeats | mean over extractants | 99 | 0.576083 | 0.576083 | 0.0e+00 | **PASS** |
| GEN9_SHAPE_RECOMPOSED + SLOPE_L_s1_K3 at k=3 | `runs/gen10_final/final_locked/kshot_detail.parquet` | gen10 rerun vs gen9's own frontier table | 99-extractant common cohort | 5 split seeds x 12 repeats | mean over extractants | 99 | 0.512725 | 0.512725 | 0.0e+00 | **PASS** |
| GEN9_SHAPE_RECOMPOSED + SLOPE_L_s1_K3 at k=5 | `runs/gen10_final/final_locked/kshot_detail.parquet` | gen10 rerun vs gen9's own frontier table | 99-extractant common cohort | 5 split seeds x 12 repeats | mean over extractants | 99 | 0.471417 | 0.471417 | 0.0e+00 | **PASS** |

### Figure 3 — curve shape

| quantity | source file | experiment | cohort | seeds | aggregation | n units | recomputed | reported | Δ | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| extractant slope_pred (median), baseline | `runs/gen9_shape/shape/curve_shape.parquet` | gen9 shape analysis | 775 curve x seed evaluations of 155 curves / 25 extractants | 5 split seeds | median over curve x seed | 775 | 0.116428 | 0.116428 | 2.8e-17 | **PASS** |
| extractant span_recovery (median), baseline | `runs/gen9_shape/shape/curve_shape.parquet` | gen9 shape analysis | 775 curve x seed evaluations of 155 curves / 25 extractants | 5 split seeds | median over curve x seed | 775 | 0.0507177 | 0.0507177 | 2.8e-17 | **PASS** |
| extractant shape_mae (mean), baseline | `runs/gen9_shape/shape/curve_shape.parquet` | gen9 shape analysis | 775 curve x seed evaluations of 155 curves / 25 extractants | 5 split seeds | mean over curve x seed | 775 | 0.664815 | 0.664815 | 0.0e+00 | **PASS** |
| extractant spearman (mean), baseline | `runs/gen9_shape/shape/curve_shape.parquet` | gen9 shape analysis | 775 curve x seed evaluations of 155 curves / 25 extractants | 5 split seeds | mean over curve x seed | 775 | 0.582371 | 0.582371 | 0.0e+00 | **PASS** |
| extractant slope_pred (median), + relative position | `runs/gen9_shape/shape/curve_shape.parquet` | gen9 shape analysis | 775 curve x seed evaluations of 155 curves / 25 extractants | 5 split seeds | median over curve x seed | 775 | 0.521045 | 0.521045 | 0.0e+00 | **PASS** |
| extractant span_recovery (median), + relative position | `runs/gen9_shape/shape/curve_shape.parquet` | gen9 shape analysis | 775 curve x seed evaluations of 155 curves / 25 extractants | 5 split seeds | median over curve x seed | 775 | 0.20953 | 0.20953 | 8.3e-17 | **PASS** |
| extractant shape_mae (mean), + relative position | `runs/gen9_shape/shape/curve_shape.parquet` | gen9 shape analysis | 775 curve x seed evaluations of 155 curves / 25 extractants | 5 split seeds | mean over curve x seed | 775 | 0.56718 | 0.56718 | 0.0e+00 | **PASS** |
| extractant spearman (mean), + relative position | `runs/gen9_shape/shape/curve_shape.parquet` | gen9 shape analysis | 775 curve x seed evaluations of 155 curves / 25 extractants | 5 split seeds | mean over curve x seed | 775 | 0.84021 | 0.84021 | 0.0e+00 | **PASS** |
| extractant slope_pred (median), + recomposition | `runs/gen9_shape/shape/curve_shape.parquet` | gen9 shape analysis | 775 curve x seed evaluations of 155 curves / 25 extractants | 5 split seeds | median over curve x seed | 775 | 1.02401 | 1.02401 | 0.0e+00 | **PASS** |
| extractant span_recovery (median), + recomposition | `runs/gen9_shape/shape/curve_shape.parquet` | gen9 shape analysis | 775 curve x seed evaluations of 155 curves / 25 extractants | 5 split seeds | median over curve x seed | 775 | 0.423291 | 0.423291 | 0.0e+00 | **PASS** |
| extractant shape_mae (mean), + recomposition | `runs/gen9_shape/shape/curve_shape.parquet` | gen9 shape analysis | 775 curve x seed evaluations of 155 curves / 25 extractants | 5 split seeds | mean over curve x seed | 775 | 0.469194 | 0.469194 | 0.0e+00 | **PASS** |
| extractant spearman (mean), + recomposition | `runs/gen9_shape/shape/curve_shape.parquet` | gen9 shape analysis | 775 curve x seed evaluations of 155 curves / 25 extractants | 5 split seeds | mean over curve x seed | 775 | 0.885628 | 0.885628 | 0.0e+00 | **PASS** |
| extractant measured slope (median) | `runs/gen9_shape/shape/curve_shape.parquet` | gen9 shape analysis | 775 curve x seed evaluations | 5 split seeds | median | 775 | 2.57368 | 2.57368 | 0.0e+00 | **PASS** |

### Figure 4 — the oracle cascade

| quantity | source file | experiment | cohort | seeds | aggregation | n units | recomputed | reported | Δ | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| oracle cascade, current | `runs/gen9_shape/recomposed/oof_all.parquet` | gen10 Phase 10 | 152 extractants | 5 split seeds | one vote per ECFP cluster (131) | 131 | 0.969485 | 0.969485 | 0.0e+00 | **PASS** |
| oracle cascade, after_ligand_level_oracle | `runs/gen9_shape/recomposed/oof_all.parquet` | gen10 Phase 10 | 152 extractants | 5 split seeds | one vote per ECFP cluster (131) | 131 | 0.449476 | 0.449476 | 0.0e+00 | **PASS** |
| oracle cascade, after_series_level_oracle | `runs/gen9_shape/recomposed/oof_all.parquet` | gen10 Phase 10 | 152 extractants | 5 split seeds | one vote per ECFP cluster (131) | 131 | 0.392812 | 0.392812 | 0.0e+00 | **PASS** |
| oracle cascade, after_curve_level_oracle | `runs/gen9_shape/recomposed/oof_all.parquet` | gen10 Phase 10 | 152 extractants | 5 split seeds | one vote per ECFP cluster (131) | 131 | 0.386005 | 0.386005 | 0.0e+00 | **PASS** |
| oracle cascade, after_curve_level_and_slope_oracle | `runs/gen9_shape/recomposed/oof_all.parquet` | gen10 Phase 10 | 152 extractants | 5 split seeds | one vote per ECFP cluster (131) | 131 | 0.194059 | 0.194059 | 0.0e+00 | **PASS** |

### Figure 5A — distance terciles and marginal gains

| quantity | source file | experiment | cohort | seeds | aggregation | n units | recomputed | reported | Δ | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| first_point (near tercile) | `runs/gen10_final/final_locked/kshot_detail.parquet` | gen10 Phase 9 budget simulation | 99-extractant common cohort | 5 split seeds x 12 repeats | mean over extractants within the tercile | 99 | 0.23485 | 0.23485 | 2.8e-17 | **PASS** |
| second_point (near tercile) | `runs/gen10_final/final_locked/kshot_detail.parquet` | gen10 Phase 9 budget simulation | 99-extractant common cohort | 5 split seeds x 12 repeats | mean over extractants within the tercile | 99 | 0.0828351 | 0.0828351 | 2.8e-17 | **PASS** |
| third_point (near tercile) | `runs/gen10_final/final_locked/kshot_detail.parquet` | gen10 Phase 9 budget simulation | 99-extractant common cohort | 5 split seeds x 12 repeats | mean over extractants within the tercile | 99 | 0.0549535 | 0.0549535 | 4.2e-17 | **PASS** |
| fourth_and_fifth_each (near tercile) | `runs/gen10_final/final_locked/kshot_detail.parquet` | gen10 Phase 9 budget simulation | 99-extractant common cohort | 5 split seeds x 12 repeats | mean over extractants within the tercile | 99 | 0.0195378 | 0.0195378 | 2.4e-17 | **PASS** |
| m0 (near tercile) | `runs/gen10_final/final_locked/kshot_detail.parquet` | gen10 Phase 9 budget simulation | 99-extractant common cohort | 5 split seeds x 12 repeats | mean over extractants within the tercile | 99 | 0.899931 | 0.899931 | 0.0e+00 | **PASS** |
| first_point (mid tercile) | `runs/gen10_final/final_locked/kshot_detail.parquet` | gen10 Phase 9 budget simulation | 99-extractant common cohort | 5 split seeds x 12 repeats | mean over extractants within the tercile | 99 | 0.448492 | 0.448492 | 0.0e+00 | **PASS** |
| second_point (mid tercile) | `runs/gen10_final/final_locked/kshot_detail.parquet` | gen10 Phase 9 budget simulation | 99-extractant common cohort | 5 split seeds x 12 repeats | mean over extractants within the tercile | 99 | 0.0878471 | 0.0878471 | 0.0e+00 | **PASS** |
| third_point (mid tercile) | `runs/gen10_final/final_locked/kshot_detail.parquet` | gen10 Phase 9 budget simulation | 99-extractant common cohort | 5 split seeds x 12 repeats | mean over extractants within the tercile | 99 | 0.0508274 | 0.0508274 | 8.3e-17 | **PASS** |
| fourth_and_fifth_each (mid tercile) | `runs/gen10_final/final_locked/kshot_detail.parquet` | gen10 Phase 9 budget simulation | 99-extractant common cohort | 5 split seeds x 12 repeats | mean over extractants within the tercile | 99 | 0.0126869 | 0.0126869 | 2.4e-17 | **PASS** |
| m0 (mid tercile) | `runs/gen10_final/final_locked/kshot_detail.parquet` | gen10 Phase 9 budget simulation | 99-extractant common cohort | 5 split seeds x 12 repeats | mean over extractants within the tercile | 99 | 0.962208 | 0.962208 | 1.1e-16 | **PASS** |
| first_point (far tercile) | `runs/gen10_final/final_locked/kshot_detail.parquet` | gen10 Phase 9 budget simulation | 99-extractant common cohort | 5 split seeds x 12 repeats | mean over extractants within the tercile | 99 | 0.494809 | 0.494809 | 5.6e-17 | **PASS** |
| second_point (far tercile) | `runs/gen10_final/final_locked/kshot_detail.parquet` | gen10 Phase 9 budget simulation | 99-extractant common cohort | 5 split seeds x 12 repeats | mean over extractants within the tercile | 99 | 0.0634467 | 0.0634467 | 4.2e-17 | **PASS** |
| third_point (far tercile) | `runs/gen10_final/final_locked/kshot_detail.parquet` | gen10 Phase 9 budget simulation | 99-extractant common cohort | 5 split seeds x 12 repeats | mean over extractants within the tercile | 99 | 0.0836681 | 0.0836681 | 1.4e-17 | **PASS** |
| fourth_and_fifth_each (far tercile) | `runs/gen10_final/final_locked/kshot_detail.parquet` | gen10 Phase 9 budget simulation | 99-extractant common cohort | 5 split seeds x 12 repeats | mean over extractants within the tercile | 99 | 0.0286995 | 0.0286995 | 2.4e-17 | **PASS** |
| m0 (far tercile) | `runs/gen10_final/final_locked/kshot_detail.parquet` | gen10 Phase 9 budget simulation | 99-extractant common cohort | 5 split seeds x 12 repeats | mean over extractants within the tercile | 99 | 1.2547 | 1.2547 | 0.0e+00 | **PASS** |

### Figure 5B,C and Figure S9 — the coverage experiment

| quantity | source file | experiment | cohort | seeds | aggregation | n units | recomputed | reported | Δ | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| gen6 BASE macro MAE (all) | `runs/gen6_expA_5seed/hard_chemistry_metrics.csv` | gen6 Experiment A | 5,248 rows / 152 extractants / 131 ECFP clusters | 5 split seeds | one vote per ECFP cluster, mean over seeds | 131 | 1.21005 | 1.21 | 4.5e-05 | **PASS** |
| gen6 EXPANDED macro MAE (all) | `runs/gen6_expA_5seed/hard_chemistry_metrics.csv` | gen6 Experiment A | 5,248 rows / 152 extractants / 131 ECFP clusters | 5 split seeds | one vote per ECFP cluster, mean over seeds | 131 | 1.04676 | 1.0468 | 3.8e-05 | **PASS** |
| gen6 EXPANDED_ROWMATCHED macro MAE (all) | `runs/gen6_expA_5seed/hard_chemistry_metrics.csv` | gen6 Experiment A | 5,248 rows / 152 extractants / 131 ECFP clusters | 5 split seeds | one vote per ECFP cluster, mean over seeds | 131 | 1.04863 | 1.0486 | 2.9e-05 | **PASS** |
| gen6 EXPANDED_SHUFFLED macro MAE (all) | `runs/gen6_expA_5seed/hard_chemistry_metrics.csv` | gen6 Experiment A | 5,248 rows / 152 extractants / 131 ECFP clusters | 5 split seeds | one vote per ECFP cluster, mean over seeds | 131 | 1.21934 | 1.2193 | 3.8e-05 | **PASS** |
| gen6 coverage effect (EXPANDED - BASE), all | `runs/gen6_expA_5seed/contrast_summary.csv` | gen6 Experiment A | identical held-out rows | 5 split seeds | paired, one vote per ECFP cluster | 131 | 0.163284 | 0.1633 | 1.6e-05 | **PASS** |

### Figure 6 — acquisition policies

| quantity | source file | experiment | cohort | seeds | aggregation | n units | recomputed | reported | Δ | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| acquisition macro MAE, ORACLE | `runs/gen10_final/acquisition/realised_detail.parquet` | gen10 Phase 6 realised acquisition | 143 extractants | 5 split seeds x 8 repeats | mean over extractants | 143 | 0.462774 | 0.462774 | 0.0e+00 | **PASS** |
| acquisition macro MAE, SURROGATE_ORACLE | `runs/gen10_final/acquisition/realised_detail.parquet` | gen10 Phase 6 realised acquisition | 143 extractants | 5 split seeds x 8 repeats | mean over extractants | 143 | 0.480359 | 0.480359 | 5.6e-17 | **PASS** |
| acquisition macro MAE, MEDOID | `runs/gen10_final/acquisition/realised_detail.parquet` | gen10 Phase 6 realised acquisition | 143 extractants | 5 split seeds x 8 repeats | mean over extractants | 143 | 0.628553 | 0.628553 | 0.0e+00 | **PASS** |
| acquisition macro MAE, CENTRAL | `runs/gen10_final/acquisition/realised_detail.parquet` | gen10 Phase 6 realised acquisition | 143 extractants | 5 split seeds x 8 repeats | mean over extractants | 143 | 0.631689 | 0.631689 | 0.0e+00 | **PASS** |
| acquisition macro MAE, LEARNED_BLEND[geometry] | `runs/gen10_final/acquisition/realised_detail.parquet` | gen10 Phase 6 realised acquisition | 143 extractants | 5 split seeds x 8 repeats | mean over extractants | 143 | 0.634236 | 0.634236 | 0.0e+00 | **PASS** |
| acquisition macro MAE, LEARNED_SCALAR[geometry] | `runs/gen10_final/acquisition/realised_detail.parquet` | gen10 Phase 6 realised acquisition | 143 extractants | 5 split seeds x 8 repeats | mean over extractants | 143 | 0.640411 | 0.640411 | 0.0e+00 | **PASS** |
| acquisition macro MAE, MEDIAN_PREDICTION | `runs/gen10_final/acquisition/realised_detail.parquet` | gen10 Phase 6 realised acquisition | 143 extractants | 5 split seeds x 8 repeats | mean over extractants | 143 | 0.644674 | 0.644674 | 0.0e+00 | **PASS** |
| acquisition macro MAE, LEARNED_RANK[geometry] | `runs/gen10_final/acquisition/realised_detail.parquet` | gen10 Phase 6 realised acquisition | 143 extractants | 5 split seeds x 8 repeats | mean over extractants | 143 | 0.647834 | 0.647834 | 0.0e+00 | **PASS** |
| acquisition macro MAE, MAX_CONDITION_COVERAGE | `runs/gen10_final/acquisition/realised_detail.parquet` | gen10 Phase 6 realised acquisition | 143 extractants | 5 split seeds x 8 repeats | mean over extractants | 143 | 0.652672 | 0.652672 | 0.0e+00 | **PASS** |
| acquisition macro MAE, MIN_GP_DESIGN_VAR | `runs/gen10_final/acquisition/realised_detail.parquet` | gen10 Phase 6 realised acquisition | 143 extractants | 5 split seeds x 8 repeats | mean over extractants | 143 | 0.655617 | 0.655617 | 0.0e+00 | **PASS** |
| acquisition macro MAE, MID_ACID | `runs/gen10_final/acquisition/realised_detail.parquet` | gen10 Phase 6 realised acquisition | 143 extractants | 5 split seeds x 8 repeats | mean over extractants | 143 | 0.672256 | 0.672256 | 0.0e+00 | **PASS** |
| acquisition macro MAE, MAX_PREDICTION | `runs/gen10_final/acquisition/realised_detail.parquet` | gen10 Phase 6 realised acquisition | 143 extractants | 5 split seeds x 8 repeats | mean over extractants | 143 | 0.678915 | 0.678915 | 0.0e+00 | **PASS** |
| acquisition macro MAE, MAX_ENSEMBLE_SD | `runs/gen10_final/acquisition/realised_detail.parquet` | gen10 Phase 6 realised acquisition | 143 extractants | 5 split seeds x 8 repeats | mean over extractants | 143 | 0.687511 | 0.687511 | 0.0e+00 | **PASS** |
| acquisition macro MAE, RANDOM | `runs/gen10_final/acquisition/realised_detail.parquet` | gen10 Phase 6 realised acquisition | 143 extractants | 5 split seeds x 8 repeats | mean over extractants | 143 | 0.691659 | 0.691659 | 0.0e+00 | **PASS** |
| acquisition macro MAE, MIN_ENSEMBLE_SD | `runs/gen10_final/acquisition/realised_detail.parquet` | gen10 Phase 6 realised acquisition | 143 extractants | 5 split seeds x 8 repeats | mean over extractants | 143 | 0.695001 | 0.695001 | 0.0e+00 | **PASS** |
| acquisition macro MAE, FARTHEST_FROM_EXISTING | `runs/gen10_final/acquisition/realised_detail.parquet` | gen10 Phase 6 realised acquisition | 143 extractants | 5 split seeds x 8 repeats | mean over extractants | 143 | 0.754005 | 0.754005 | 0.0e+00 | **PASS** |
| acquisition macro MAE, MIN_PREDICTION | `runs/gen10_final/acquisition/realised_detail.parquet` | gen10 Phase 6 realised acquisition | 143 extractants | 5 split seeds x 8 repeats | mean over extractants | 143 | 0.781436 | 0.781436 | 0.0e+00 | **PASS** |

### Design identities

| quantity | source file | experiment | cohort | seeds | aggregation | n units | recomputed | reported | Δ | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| recomposition leaves the level unchanged (offset MAE) | `runs/gen10_final/headline_tables/t2_leaderboards.csv` | gen10 leaderboards | 152 extractants | 5 split seeds | one vote per ECFP cluster | 131 | 0.821172 | 0.821172 | 0.0e+00 | **PASS** |
---

## 3. Discrepancies found — none is an error, all change what a figure may say

These are the reasons several figures do **not** simply replot a published table.

### 3.1 The published "gen8 frontier" and "gen9 frontier" are best-of-many-policies envelopes

`runs/gen10_final/final_locked/frontier_best.csv` carries columns `gen8_frontier` and
`gen9_frontier` with the values 1.0605 / 0.6674 / 0.5879 / 0.5230 / 0.4743 and
1.0358 / 0.6539 / 0.5746 / 0.5109 / 0.4675. Tracing them into
`runs/gen9_shape/frontier/table_b_common_cohort.csv` shows they are the *minimum over
every deployable arm at each k* — and the winning arm at k = 2 is
`SLOPE_L_s1_K3@LEARNED_SCALAR` (gen9) or `SLOPE_L_s1_K3@MAX_PREDICTIVE_VARIANCE` (gen8),
two acquisition policies that gen10's rerun did not include because Phase 6 closed that
direction. gen10's own number, by contrast, is a single **fixed rule**.

Consequence: putting those published numbers on the same axis as gen10's fixed rule would
compare a best-of-eleven envelope with a pre-specified recipe — a comparison that flatters
the baselines and is not reproducible from the frozen table. **Figure 2 therefore evaluates
one pre-specified rule per generation, all from the same `kshot_detail.parquet`**:
1.0605 / 0.6752 / 0.5909 / 0.5230 / 0.4772 (baseline model) and
1.0358 / 0.6539 / 0.5761 / 0.5127 / 0.4714 (+ shape recomposition). Rows 5–10 of the audit
table confirm these reproduce gen9's own table exactly for the identical arms.

### 3.2 `budget_simulation/marginal_gains.csv` is on gen9's adapter chain, not the frozen one

Its `second_point` is 0.0778, which chains 0.6539 → 0.5761 — that is `SLOPE_L_s1_K3`, not
the frozen `SERIES_ML`. Figure 2B reports the **frozen pipeline's** marginal gains
(0.382 / 0.095 / 0.066 / 0.026 per measurement), which chain exactly to the frozen
frontier. The published table is reproduced to 10⁻⁹ in rows 29–43 of the audit *on its own
adapter chain*, which is how Figure 5A's tercile split is validated.

### 3.3 Two definitions of span recovery live in `shape_by_axis.csv`

`span_recovery_median` (0.046445 for the baseline) applies a guard against near-zero
measured spans; `span_recovery_median_unguarded` (0.050718) does not. The 0.051 / 0.210 /
0.423 ladder quoted in `README.md` and the gen9 decision report is the **unguarded**
one, and that is what Figure 3E plots. Both are recomputed and both agree with the table.

### 3.4 The acquisition effect size depends on which study is quoted

`runs/gen8_architecture/decision_report.md` reports MEDOID vs RANDOM at **+0.101** on 99
extractants with gen8's global model. gen10's realised-regret study, on 143 extractants
with the frozen global model and 8 repeats, gives **+0.0631, BCa [+0.0280, +0.1050],
98/143 extractants, 5/5 seeds**. Both are positive with intervals excluding zero; they are
not the same experiment. Figure 6 quotes the gen10 number, because that is the study whose
raw per-extractant records are in the repository and are reproduced here.

### 3.5 `NULL_metal_cond` is 1.0882 at three seeds and 1.0995 at five

The gen7 report's headline sentence uses the three-seed value from the `oracles` sweep;
`runs/gen7_architecture/leaderboard_all.csv` carries the five-seed `finalists` value.
Figure 4C uses the model's own out-of-fold predictions restricted to C-COMMON and gets
1.1149 macro / 0.9082 offset / 0.6027 shape, and every stage in that panel is computed the
same way, so the panel is internally consistent. The claim "all of ligand chemistry is
worth about 0.13 log units" uses 1.0995 − 0.9695 on C-FULL; on C-COMMON it is
1.1149 − 1.0324 = 0.083. **The paper should quote the cohort with the number.**

### 3.6 The publication count is 105, not 109

`README.md` reports 109 publications from the provenance reconstruction; the frozen
artefact `runs/gen6_provenance/summary.json` records **105**, with the definition spelled
out in the same file ("canonical set of DOIs from the upstream SAFE export, excluding the
SAFE database self-citation"). The Methods text uses 105 and states the definition. The
upstream join itself is 5,992/5,992 in both.

### 3.7 The mean-preserving identity is exact, and worth stating as a result

`offset_mae` is bit-identical for `REC_ecfp_plus_recovered` and every recomposed arm
(0.82117153046591 on C-FULL; recomputed independently on C-COMMON, 0.84676720, identical
in all five seeds). This is not a coincidence to be smoothed over — it is what
`gen9/train.py::ShapeRecomposed` guarantees, and it is the cleanest available evidence
that the shape repair and the level are separable. It appears in Fig 4C, Fig S4B and the
Figure 1 caption.

---

## 4. New calculations (labelled NEW nowhere in the table because each has a reference)

Three quantities in the figures did not previously exist in this form. Each was validated
by first reproducing the published version of the same computation:

1. **Oracle cascade on C-COMMON with per-extractant macro** (Figure 4A). The cascade code
   in `prepare_error_budget.py` is copied from `scripts/gen10_error_decomposition.py`; it
   reproduces all ten published stage × metric values to 5 × 10⁻⁹ on the study's own
   cohort (audit rows 24–28 report the macro half) before being rerun on C-COMMON.
2. **Chemotype-block bootstrap of a mean** (`_stats.block_bootstrap_mean`). The
   repository only ever needed paired deltas, so a one-sample interval did not exist. It
   uses the same block structure, replicate count (5,000) and RNG seed (8675309) as
   `gen8.inference.paired_chemotype_bootstrap`, which the figures import unmodified for
   every paired comparison.
3. **Level/shape decomposition of the four model stages on C-COMMON** (Figure 4C). Uses
   `gen6.metrics.decompose_level_shape` — the repository's own function — on the stored
   out-of-fold predictions.

---

## 5. Flagged: results that are true but easy to over-read

* **The +0.163 coverage effect is not "the model got better everywhere."** 57 of the 131
  scoring units are ECFP clusters made *entirely* of the 61 extractants the expansion
  added, which the restricted arm structurally cannot serve. On the 91 extractants the
  restricted arm could already cover, the effect is +0.019 (52 % improved) — a wash. The
  row-weighted contrast is +0.068. Figure 5 plots the distance-stratified version and the
  caption states the qualification; Figure S9 carries the dose-response that rules out a
  pure class prior (Spearman +0.290, p = 2.9 × 10⁻⁴, recomputed here).
* **`n_units` is not `n_rows`.** One extractant contributes up to 1,488 rows (28 % of the
  corpus) and one chemotype holds 64 % of the rows. Every interval in every figure
  resamples chemotypes, never rows.
* **Figure 2's k = 0 point is a different number on a different cohort (0.9695 on
  C-FULL).** It is never mixed.
* **Oracle arms** appear in Figures 2A, 4A and 6A and are hatched, coloured
  distinctly, and labelled "not deployable" in the artwork itself, not only in the caption.
