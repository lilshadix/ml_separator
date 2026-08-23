# 20. Is uncertainty useful?

**Short answer: the uncertainties are real, and they are useless for the
decision that matters.**

Every uncertainty source available for the frozen global model
(`REC_ecfp_plus_recovered`) was built and then scored twice: once as a
*calibration* signal (does it rank the error?) and once as a *decision* signal
(does it rank the candidate rows you could go and measure?).  The two answers
point in opposite directions.  The tree-ensemble sd ranks unseen extractants by
difficulty at across-ligand Spearman 0.475, with a cleanly monotone
calibration curve -- and measuring its most uncertain row costs +0.095 log
units against a random row (0.805 vs 0.709).

The reason is not that the uncertainties are bad.  It is that under offset
calibration the quantity a 1-shot policy must minimise is
**abs(r_i - median(r))**, an unsigned *deviation from the ligand's own median
residual*.  Ranking candidates by that quantity reproduces the exhaustive 1-shot
oracle **exactly** -- in all 715 (seed, ligand) cells, to
6.7e-16.  Ranking them
by abs(r_i), which is what a perfectly calibrated uncertainty gives you, is
0.070 log units *worse than random*.  Uncertainty measures
extremity; the policy needs typicality.

---

## 20.1 What was measured

Frozen model, 5 split seeds x 5 folds, held-out chemotypes.  Two populations,
because the two questions have different supports.  **Calibration (20.2)** scores
every held-out row of every held-out extractant: 760 (seed, ligand)
cells over 152 extractants, 26,240 row-seeds.
**Everything with a decision in it (20.3-20.5)** needs at least two candidate rows
per cell so that a 1-shot MAE exists, which leaves 143 unseen extractants in 72
Tanimoto chemotypes, 715 cells and 26,105 candidate rows -- the
population every macro MAE and every interval below is computed on.  Residual
`r = log_D - prediction`.  The exact 1-shot MAE of every candidate row comes
from `cross_series/one_shot_candidate_scores.parquet`, whose definition was
re-derived from the OOF residuals before use and matches to machine precision:

    one_shot_mae(i) = mean over j != i of abs(r_j - r_i)
    zero_shot_mae   = mean over j of abs(r_j)
    oracle_level_mae= mean over j of abs(r_j - median(r))

so "which candidate should I measure" is answered in closed form rather than by
simulation, and no policy needs a repeat count.

Sources.  All are computable at prediction time; none sees a held-out label.

| id | source | level |
|---|---|---|
| `u_tree_sd` | tree-ensemble prediction sd of `TREE_MC_ecfp_massaction` | row |
| `u_tree_sd_mean` | mean tree-ensemble sd of the three `TREE_MC_*` arms | row |
| `u_model_spread` | sd of the 5 finalists' predictions at the same (row, seed) | row |
| `u_cond_nn1`, `u_cond_nn5` | k-th NN Euclidean distance to the fold's training rows in standardised condition space (log concentrations, temperature, contact time, metal Z, acid / diluent / additive one-hots; z-scored on the training fold) | row |
| `u_gp_var_design` | leave-one-out RBF-GP predictive variance over the ligand's own 1-3 varying condition axes, median-heuristic lengthscale | row |
| `u_gp_var_train` | RBF-GP predictive variance at the row given a 1200-row sample of the fold's training design | row |
| `u_lig_tanimoto` | `1 - nn_train_tanimoto` | ligand |
| `u_lig_mech` | mechanistic distance (36 descriptors from `gen8.mechanism`) to the nearest training ligand, standardised on the fold's training ligands | ligand |

A GP's predictive variance depends only on its inputs, so `u_gp_var_design` uses
no targets at all: it measures how tightly the rest of a ligand's own design
surrounds each candidate point.  Of the 760 (seed, ligand) cells,
500 vary along one axis, 105 along two and
155 along three (`gp_axes_used.csv`); the commonest axes
are metal Z and acid concentration.

The frozen model does **not** emit a per-row dispersion -- its
`extra__prediction_sd` column is 100% null.  Only these models carry one:

| model | rows_with_prediction_sd | rows_total |
|---|---|---|
| HNN_base_film | 26,240 | 26,240 |
| TREE_MC_donors | 26,240 | 26,240 |
| TREE_MC_ecfp_massaction | 26,240 | 26,240 |
| TREE_MC_lig2d_ext_massaction | 26,240 | 26,240 |

so the tree-variance signal is taken from the frozen model's nearest
architectural sibling, `TREE_MC_ecfp_massaction`.  Section 20.6 shows that this
substitution is not what drives the result.

### An audit finding, first

Two of the three uncertainty policies already on disk were not uncertainty
policies.  `MAX_ENSEMBLE_SD` and `MIN_ENSEMBLE_SD` read `extra__prediction_sd`
(`scripts/gen8_kshot.py:65`), which is all-null for the frozen model, and
`gen8/kshot.py::_argmax_finite` falls back to `rng.choice` when no score is
finite -- as does `policy_min_sd`.  Both arms therefore drew **uniformly at
random**.  This is not an inference from the aggregates: loading the frozen
model's blocks through `scripts/gen8_kshot.py::load_blocks` gives
`uncertainty` with 0 finite values out of 26,240, and calling
`policy_max_sd`, `policy_min_sd` and `policy_random` on the same block with the
same generator state returns the **same index** every time
(`MAX_ENSEMBLE_SD` = `MIN_ENSEMBLE_SD` = `RANDOM` as functions, not merely in
mean).  `disagreement` is finite on all 26,240 rows, so
`MAX_MODEL_DISAGREEMENT` is a real policy.  The aggregates agree: the two sd
arms are indistinguishable from each other and from `RANDOM`.

| comparison (positive = right-hand arm better) | delta | ci_lo | ci_hi | seeds /5 |
|---|---|---|---|---|
| RANDOM - MAX_ENSEMBLE_SD | 0.016 | -0.018 | 0.041 | 5 |
| RANDOM - MIN_ENSEMBLE_SD | 0.008 | -0.024 | 0.030 | 5 |
| MAX_ENSEMBLE_SD - MIN_ENSEMBLE_SD | -0.008 | -0.026 | 0.013 | 0 |
| RANDOM - MAX_MODEL_DISAGREEMENT | -0.016 | -0.051 | 0.010 | 1 |
| RANDOM - MAX_PREDICTIVE_VARIANCE | -0.021 | -0.076 | 0.013 | 0 |
| RANDOM - CENTRAL | 0.072 | 0.026 | 0.108 | 5 |

| arm | macro_mae |
|---|---|
| ORACLE[OFFSET_K1] | 0.484 |
| MEDOID | 0.648 |
| CENTRAL | 0.651 |
| MAX_ENSEMBLE_SD | 0.707 |
| MIN_ENSEMBLE_SD | 0.715 |
| RANDOM | 0.723 |
| MAX_MODEL_DISAGREEMENT | 0.740 |
| MAX_PREDICTIVE_VARIANCE | 0.744 |

`MAX_PREDICTIVE_VARIANCE` is degenerate at k=1 for a different reason: with no
point yet selected, `_gp_posterior_variance` returns a constant vector, so
`argmax` always returns pool element 0, and `make_p2_split` sorts the pool, so
the arm measures the lowest-indexed candidate of the ligand -- an ordering
artefact, not a variance.  Verified by direct call: it returns `pool[0]` under
every generator state.  Only `MAX_MODEL_DISAGREEMENT` was a real uncertainty
policy, and it was already worse than random.

**Read the `seeds /5` column of that table with care.**  In this harness the
pool/evaluation split is seeded on `(harness seed, repeat, ligand)` and each
policy's generator on `(harness seed, repeat, policy name)` -- neither includes
the split seed -- and each ligand's rows appear in the same order in all five
seeds (checked).  Every policy therefore measures the *same physical rows* in all
five split seeds, so `5/5` there means five refits of the model agreeing over one
fixed set of draws, not five independent draws.  It applies to the arms in this
audit table only; sections 20.3-20.5 do not use that file.

**Caveat on the source.**  `primary_detail.parquet` is regenerated by the
acquisition sweep and was rewritten under this session at least once, and reads
taken at different times during the session returned different aggregates.  Which
bytes each earlier read saw was not recorded, so the difference is attributed to
the rewrite and not to anything else; an earlier draft of this section called one
of the reads torn, which the evidence does not support.  The snapshot this
section is computed from is pinned in
`audit_source_provenance.csv` (2026-08-20T00:34:15.556065+00:00 UTC, 2,843,580 rows, sha256
59fb1f353361), and every audit number above comes from a single read of it.  In
that snapshot RANDOM scores 0.7232, MAX_ENSEMBLE_SD 0.7073 and MIN_ENSEMBLE_SD
0.7150; earlier snapshots gave 0.702 / 0.709 / 0.708 and 0.711 / 0.712 /
0.713.  The pinned snapshot is also the file on disk now, and ligand-macro,
row-pooled and seed-then-ligand aggregation of it all return the same numbers, so
the earlier values came from earlier *content*, not from a different reduction of
the same content.  The absolute level moves; the **MAX = MIN = RANDOM equality
holds in every snapshot observed**, and it holds by construction in the code,
which is the only thing this section claims from the file.  Nothing else in
section 20 reads it -- sections 20.2 to 20.5 are computed
from `oof_predictions.parquet` and `one_shot_candidate_scores.parquet`, both
unmodified since before this session.

The previously reported "MAX_ENSEMBLE_SD 0.709 vs RANDOM 0.702" therefore
carried no information about tree variance.  Sections 20.3-20.4 replace it with
properly sourced arms; the negative conclusion survives, and gets stronger.

---


## 20.2 (a) Calibration: the uncertainties are real

| source | within-ligand rho | ci_lo | ci_hi | frac cells rho>0 | n cells | across-ligand rho | seeds>0 |
|---|---|---|---|---|---|---|---|
| u_tree_sd | 0.139 | 0.064 | 0.250 | 0.614 | 696 | 0.475 | 5 |
| u_tree_sd_mean | 0.116 | 0.030 | 0.215 | 0.599 | 696 | 0.442 | 5 |
| u_model_spread | 0.040 | -0.040 | 0.125 | 0.511 | 700 | 0.377 | 5 |
| u_cond_nn1 | 0.012 | -0.056 | 0.061 | 0.502 | 598 | 0.329 | 5 |
| u_cond_nn5 | 0.019 | -0.038 | 0.085 | 0.514 | 685 | 0.303 | 5 |
| u_gp_var_design | 0.010 | -0.088 | 0.077 | 0.481 | 700 | -0.126 | 0 |
| u_gp_var_train | 0.030 | -0.039 | 0.101 | 0.537 | 700 | 0.300 | 5 |
| u_lig_tanimoto |  |  |  |  | 0 | 0.092 | 5 |
| u_lig_mech |  |  |  |  | 0 | 0.258 | 5 |

The two ligand-level sources are constant inside a ligand, so no within-ligand
Spearman exists for them; those cells are blank throughout this section.

Within a ligand the signal is weak but real for the two tree-sd variants
(0.139, CI [0.064, 0.250] and 0.116, CI [0.030, 0.215]);
they are the only intervals that exclude zero, and every other row-level source
is indistinguishable from it.  The `n cells` column is not the same for every
source -- a source that is constant inside a cell yields no Spearman there -- so
that column-to-column comparison is unpaired as printed.  Repeated on the
594 cells where all seven row-level
sources are defined, the ranking is unchanged: `u_tree_sd`
0.137,
`u_tree_sd_mean` 0.118,
`u_model_spread` 0.051,
`u_cond_nn1` 0.012,
`u_gp_var_design` 0.005
(`calibration_within_ligand_common.csv`).

*Across* ligands it is strong and unanimous over
seeds: the tree sd ranks which extractant the model will do badly on at rho
0.475, the deep-ensemble spread at 0.377, condition-space 1-NN distance
at 0.329, mechanistic ligand distance at 0.258.  Tanimoto distance to the
nearest training ligand is the weakest of them (0.092).  `u_gp_var_design` is
*negatively* related to ligand difficulty (-0.126), as it should be: it is a
property of the experimental design, not of the chemistry.

Pooled calibration curve for `u_tree_sd`, deciles over all 26,240 held-out
row-seeds:

| decile | mean_u | realised_mae | n |
|---|---|---|---|
| 1 | 0.576 | 0.674 | 2,624 |
| 2 | 0.756 | 0.970 | 2,624 |
| 3 | 0.848 | 1.003 | 2,624 |
| 4 | 0.928 | 1.079 | 2,624 |
| 5 | 1.000 | 1.131 | 2,624 |
| 6 | 1.070 | 1.227 | 2,624 |
| 7 | 1.153 | 1.343 | 2,624 |
| 8 | 1.246 | 1.385 | 2,624 |
| 9 | 1.379 | 1.355 | 2,624 |
| 10 | 1.659 | 1.422 | 2,624 |

Monotone across eight of ten deciles, a factor 2.1 from bottom to top.  By the
usual standard this is a well-calibrated uncertainty.

The same curve computed *within* ligand -- rank the rows inside each ligand-seed
cell, cut into quintiles, subtract that ligand's own mean abs(residual), then
macro-average over the 91 ligands with at least 10 rows -- is an order of
magnitude flatter:

| quintile | u_tree_sd | u_model_spread | u_cond_nn1 | u_cond_nn5 | u_gp_var_design | u_gp_var_train |
|---|---|---|---|---|---|---|
| 1 | -0.1301 | -0.0285 | -0.0966 | -0.0542 | 0.0238 | -0.0854 |
| 2 | -0.0646 | -0.0312 | -0.0056 | -0.0292 | -0.0497 | -0.0206 |
| 3 | 0.0196 | 0.0189 | -0.0061 | -0.0203 | -0.0204 | 0.0480 |
| 4 | 0.0749 | 0.0239 | 0.0255 | 0.0077 | -0.0050 | -0.0091 |
| 5 | 0.1024 | 0.0180 | 0.0773 | 0.0920 | 0.0451 | 0.0697 |

Almost all of the apparent calibration is between-ligand difficulty.  Inside a
ligand, which is the only place a 1-shot acquisition policy ever operates, the
tree sd spans about 0.23 log units from bottom quintile to top, and the
condition-space and GP measures span less.

---

## 20.3 (b) Decision usefulness: does the number rank *candidates*?

Within-ligand Spearman between each source and the exact 1-shot MAE of that
candidate -- positive means high uncertainty picks a *worse* candidate -- macro
over the same 143 ligands with a paired chemotype bootstrap.  The two right-hand
columns are point estimates without intervals; their intervals are in
`decision_spearman.csv`:

| source | rho(u, 1-shot MAE) | ci_lo | ci_hi | rho(u, dev from median r) | rho(u, abs r) |
|---|---|---|---|---|---|
| u_tree_sd | 0.025 | -0.035 | 0.105 | 0.016 | 0.139 |
| u_tree_sd_mean | 0.014 | -0.038 | 0.093 | 0.007 | 0.116 |
| u_model_spread | -0.012 | -0.073 | 0.037 | -0.011 | 0.040 |
| u_cond_nn1 | 0.001 | -0.049 | 0.049 | 0.002 | 0.012 |
| u_cond_nn5 | 0.091 | 0.033 | 0.166 | 0.094 | 0.019 |
| u_gp_var_design | 0.296 | 0.247 | 0.342 | 0.303 | 0.010 |
| u_gp_var_train | 0.179 | 0.117 | 0.248 | 0.172 | 0.030 |
| u_lig_tanimoto |  |  |  |  |  |
| u_lig_mech |  |  |  |  |  |

Every model-uncertainty source is flat against candidate quality.  The two
GP-variance sources are the only ones with a clear signal, and it is
**positive**: a high-variance point is a *bad* point to measure.  The last two
columns are the whole story in miniature -- the source that best tracks abs(r)
(`u_tree_sd`, 0.139) is exactly the one that does not track the deviation
from the median (0.016), while the source that does track the deviation
(`u_gp_var_design`, 0.303) does not track abs(r) at all (0.010).

### Realised MAE of "measure the extreme candidate"

One ligand one vote, 143 ligands, mean over 5 seeds.  `RANDOM(expected)` is the
exact expectation over candidates, not a sample.  `ORACLE_*` arms read held-out
labels: they are ceilings, not policies.  "extremity of pick" is the mean
abs(percentile of the picked row's residual - 0.5); a uniform random pick sits
at 0.25 by construction.

| arm | macro MAE | extremity of pick | s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|---|---|
| ORACLE_LEVEL(best constant) | 0.480 |  | 0.477 | 0.493 | 0.469 | 0.481 | 0.478 |
| ORACLE_MIN_MEDIAN_DEV | 0.522 | 0.033 | 0.519 | 0.535 | 0.510 | 0.524 | 0.521 |
| ORACLE_1SHOT | 0.522 | 0.033 | 0.519 | 0.535 | 0.510 | 0.524 | 0.521 |
| MIN[u_gp_var_design] | 0.639 | 0.217 | 0.622 | 0.656 | 0.631 | 0.649 | 0.637 |
| MIN[u_gp_var_train] | 0.675 | 0.237 | 0.674 | 0.697 | 0.657 | 0.664 | 0.682 |
| MIN[u_cond_nn5] | 0.699 | 0.251 | 0.679 | 0.718 | 0.677 | 0.718 | 0.701 |
| MAX[u_lig_mech] | 0.703 | 0.242 | 0.644 | 0.763 | 0.683 | 0.731 | 0.694 |
| MIN[u_cond_nn1] | 0.708 | 0.253 | 0.688 | 0.728 | 0.659 | 0.739 | 0.726 |
| RANDOM(expected) | 0.709 |  | 0.705 | 0.726 | 0.694 | 0.714 | 0.707 |
| MAX[u_lig_tanimoto] | 0.713 | 0.253 | 0.689 | 0.767 | 0.705 | 0.692 | 0.711 |
| MIN[u_lig_tanimoto] | 0.718 | 0.252 | 0.707 | 0.759 | 0.698 | 0.716 | 0.710 |
| MIN[u_lig_mech] | 0.722 | 0.250 | 0.716 | 0.718 | 0.747 | 0.730 | 0.697 |
| MIN[u_tree_sd_mean] | 0.727 | 0.268 | 0.764 | 0.722 | 0.716 | 0.697 | 0.736 |
| MIN[u_tree_sd] | 0.752 | 0.276 | 0.809 | 0.760 | 0.701 | 0.724 | 0.766 |
| MAX[u_cond_nn1] | 0.772 | 0.274 | 0.749 | 0.797 | 0.762 | 0.796 | 0.756 |
| ORACLE_MIN_ABS_RESID | 0.780 | 0.309 | 0.791 | 0.776 | 0.774 | 0.771 | 0.786 |
| MIN[u_model_spread] | 0.796 | 0.286 | 0.747 | 0.788 | 0.775 | 0.831 | 0.841 |
| MAX[u_tree_sd_mean] | 0.801 | 0.295 | 0.781 | 0.814 | 0.792 | 0.800 | 0.816 |
| MAX[u_model_spread] | 0.803 | 0.300 | 0.774 | 0.845 | 0.785 | 0.823 | 0.787 |
| MAX[u_tree_sd] | 0.805 | 0.301 | 0.743 | 0.857 | 0.795 | 0.803 | 0.825 |
| MAX[u_cond_nn5] | 0.898 | 0.298 | 0.890 | 0.959 | 0.855 | 0.890 | 0.897 |
| MAX[u_gp_var_train] | 0.903 | 0.327 | 0.897 | 0.960 | 0.873 | 0.887 | 0.897 |
| MAX[u_gp_var_design] | 0.977 | 0.342 | 0.978 | 0.996 | 0.946 | 0.995 | 0.969 |
| ZERO_SHOT | 0.995 |  | 1.006 | 0.988 | 0.969 | 0.983 | 1.030 |
| ORACLE_MAX_ABS_RESID | 1.122 | 0.453 | 1.126 | 1.151 | 1.094 | 1.128 | 1.109 |
| WORST_1SHOT | 1.325 | 0.453 | 1.316 | 1.357 | 1.299 | 1.337 | 1.314 |

Paired chemotype bootstrap against random, positive = the arm beats random:

| RANDOM minus arm (positive = arm better) | delta | ci_lo | ci_hi | bca_lo | bca_hi | ligands better /143 | seeds /5 |
|---|---|---|---|---|---|---|---|
| RANDOM - ORACLE_LEVEL(best constant) | 0.230 | 0.188 | 0.259 | 0.196 | 0.264 | 143 | 5 |
| RANDOM - ORACLE_1SHOT | 0.187 | 0.148 | 0.214 | 0.156 | 0.221 | 143 | 5 |
| RANDOM - ORACLE_MIN_MEDIAN_DEV | 0.187 | 0.148 | 0.214 | 0.156 | 0.221 | 143 | 5 |
| RANDOM - MIN[u_gp_var_design] | 0.070 | 0.046 | 0.088 | 0.047 | 0.089 | 112 | 5 |
| RANDOM - MIN[u_gp_var_train] | 0.035 | 0.004 | 0.057 | 0.002 | 0.055 | 99 | 5 |
| RANDOM - MIN[u_cond_nn5] | 0.011 | -0.026 | 0.042 | -0.032 | 0.039 | 90 | 4 |
| RANDOM - MAX[u_lig_mech] | 0.006 | -0.012 | 0.024 | -0.012 | 0.023 | 83 | 3 |
| RANDOM - MIN[u_cond_nn1] | 0.001 | -0.032 | 0.029 | -0.032 | 0.029 | 73 | 2 |
| RANDOM - MAX[u_lig_tanimoto] | -0.004 | -0.019 | 0.021 | -0.021 | 0.018 | 78 | 2 |
| RANDOM - MIN[u_lig_tanimoto] | -0.009 | -0.026 | 0.009 | -0.026 | 0.009 | 68 | 0 |
| RANDOM - MIN[u_lig_mech] | -0.013 | -0.028 | 0.012 | -0.031 | 0.008 | 78 | 2 |
| RANDOM - MIN[u_tree_sd_mean] | -0.018 | -0.045 | 0.008 | -0.046 | 0.007 | 73 | 2 |
| RANDOM - MIN[u_tree_sd] | -0.043 | -0.075 | -0.003 | -0.079 | -0.009 | 63 | 0 |
| RANDOM - MAX[u_cond_nn1] | -0.063 | -0.102 | -0.012 | -0.112 | -0.023 | 67 | 0 |
| RANDOM - ORACLE_MIN_ABS_RESID | -0.070 | -0.114 | -0.021 | -0.123 | -0.028 | 57 | 0 |
| RANDOM - MIN[u_model_spread] | -0.087 | -0.123 | -0.049 | -0.131 | -0.054 | 52 | 0 |
| RANDOM - MAX[u_tree_sd_mean] | -0.092 | -0.147 | -0.050 | -0.155 | -0.054 | 63 | 0 |
| RANDOM - MAX[u_model_spread] | -0.094 | -0.129 | -0.044 | -0.133 | -0.049 | 60 | 0 |
| RANDOM - MAX[u_tree_sd] | -0.095 | -0.154 | -0.054 | -0.156 | -0.056 | 57 | 0 |
| RANDOM - MAX[u_cond_nn5] | -0.189 | -0.251 | -0.106 | -0.264 | -0.119 | 56 | 0 |
| RANDOM - MAX[u_gp_var_train] | -0.194 | -0.258 | -0.107 | -0.273 | -0.122 | 46 | 0 |
| RANDOM - MAX[u_gp_var_design] | -0.268 | -0.355 | -0.146 | -0.384 | -0.173 | 39 | 0 |
| RANDOM - ZERO_SHOT | -0.286 | -0.415 | -0.176 | -0.438 | -0.187 | 60 | 0 |
| RANDOM - ORACLE_MAX_ABS_RESID | -0.412 | -0.512 | -0.274 | -0.543 | -0.302 | 11 | 0 |
| RANDOM - WORST_1SHOT | -0.616 | -0.748 | -0.437 | -0.799 | -0.471 | 0 | 0 |

Read it three ways.

1. **Maximum uncertainty loses, significantly.**  `MAX[u_tree_sd]` 0.805 vs
   random 0.709: -0.095 (CI [-0.154, -0.054], 0/5 seeds).
   `MAX[u_model_spread]`, `MAX[u_cond_nn5]` and `MAX[u_gp_var_train]` lose too.
   `MAX[u_gp_var_design]` at 0.977 throws away essentially the entire
   value of the measurement -- zero-shot is 0.995.
2. **Minimum uncertainty does not rescue it** for the model-based sources:
   `MIN[u_tree_sd]` 0.752 and `MIN[u_model_spread]` 0.796 are *also*
   worse than random.  Both tails of a model-uncertainty ordering are bad places
   to spend the measurement, which is already the hint that the ordering is on
   the wrong axis entirely.
3. **The one winner is not an uncertainty.**  `MIN[u_gp_var_design]` 0.639
   beats random by 0.070 (CI [0.046, 0.088], BCa [0.047, 0.089],
   112/143 ligands, 5/5 seeds).  That is GP variance used *backwards* --
   the point the rest of the design surrounds most tightly.  It is a
   design-centrality criterion and it lands where `CENTRAL` and `MEDOID` already
   landed (0.651 and 0.648 in the
   pinned snapshot above, 0.646 each in the snapshot section 19 was written from;
   they are not on the same candidate table as the arms in this section and are
   quoted for orientation, not compared).

The ligand-level sources cannot rank rows within a ligand -- they are constant
there -- so their MAX / MIN arms are random tie-breaks and duly score within
noise of random in both directions.  That is a harness consistency check, not a
finding.

---


## 20.4 (c) The mechanism

Measuring candidate *i* sets the offset to `r_i`, so the resulting error on
every other row *j* is `abs(r_j - r_i)`.  The candidate's score is the mean
absolute deviation of the ligand's residual set *around r_i* -- a convex
function of `r_i`, minimised at the **median** residual.  The policy's job is
not to find the row the model gets right.  It is to find the row whose error is
**typical**.

Within-ligand Spearman, macro over 143 ligands, chemotype CI:

| quantity | within_ligand_rho | ci_low | ci_high | n_cells |
|---|---|---|---|---|
| abs(r - median r) ranks candidate quality | 0.977 | 0.974 | 0.981 | 700 |
| abs(r) ranks candidate quality | 0.255 | 0.178 | 0.321 | 700 |
| abs(r) ranks abs(r - median r) | 0.233 | 0.151 | 0.301 | 700 |

- `abs(r - median r)` ranks candidate quality at rho 0.977 -- it *is* the
  criterion.  Picking `argmin abs(r_i - median(r))` scores 0.5220, identical
  to the exhaustive 1-shot oracle 0.5220 in all 715 cells
  (max gap 6.7e-16, 0 cells above 1e-12).
- `abs(r)` -- the target of every calibrated uncertainty -- ranks candidate
  quality at only 0.255, and ranks the real criterion at 0.233.

**So even a perfect uncertainty is a bad policy.**  `ORACLE_MIN_ABS_RESID` --
an oracle that reads the held-out labels and picks the row where the model is
*most accurate* -- scores 0.780, which is 0.070 log units **worse than
random** (CI [0.021, 0.114], worse on 5/5 seeds).  Perfect
calibration, perfectly deployed, actively hurts.  No improvement in uncertainty
quality fixes that, because the objective is aimed elsewhere.

Why: on an unseen chemotype the error is dominated by a ligand-wide *level*
offset, not by row-to-row noise.

| quantity | macro over 143 ligands |
|---|---|
| abs(median residual) -- the level error | 0.865 |
| mean abs(r - median r) -- the shape error, = oracle level | 0.480 |
| zero-shot MAE | 0.995 |

A row with small abs(r) is a row where the ligand's large level offset happened
to be cancelled by something local.  Its residual is the *least* representative
of the ligand, so adopting it as the offset transports that cancellation onto
every other row.  Symmetrically, a high-uncertainty row is usually an
extreme-residual row, unrepresentative in the other direction.  Both tails lose;
the middle wins.

The entire leaderboard falls out of one number -- how far from the median
residual the policy's pick lands.  Across the 23 picking arms, the Spearman
between mean extremity-of-pick and the arm's macro MAE is
**0.965**.  The oracle arms are *defined* by residual extremity, so
including them makes that partly circular; dropping them leaves
0.959 over the 18 deployable arms and
0.982 over the 14 deployable row-level ones,
so it is not.  It is a correlation across arms, not a paired within-ligand
interval, and it is offered as the ordering principle rather than as a test.
The oracle sits at 0.033, a random pick at 0.25, every
model-uncertainty arm *above* 0.25 in both directions, and the only deployable
arm below it -- `MIN[u_gp_var_design]` at 0.217 -- is the only one that beats
random.

Acquisition here is a **centrality** problem, not an **uncertainty** problem,
and uncertainty is a monotone measure of extremity: the wrong sign, on the wrong
axis.

---

## 20.5 Where uncertainty could still have paid: choosing the ligand

The row-level decision is hopeless, but the *ligand*-level one is a different
question.  Given a budget of M measurements over 143 unseen extractants, does
uncertainty pick the right extractants?  Calibration says it should -- the
across-ligand rho against zero-shot error is 0.475.

It does not, or at least not measurably.  Uncertainty ranks the zero-shot error;
the budget decision needs the *gain* from measuring, which is zero-shot minus
post-correction error:

| source | rho(u, zero-shot MAE) | rho(u, gain from measuring) | seeds>0 /5 |
|---|---|---|---|
| u_tree_sd | 0.505 | 0.122 | 5 |
| u_tree_sd_mean | 0.482 | 0.094 | 5 |
| u_model_spread | 0.419 | 0.152 | 5 |
| u_cond_nn1 | 0.333 | -0.027 | 1 |
| u_cond_nn5 | 0.301 | -0.030 | 1 |
| u_gp_var_design | -0.149 | 0.007 | 4 |
| u_gp_var_train | 0.311 | -0.110 | 0 |
| u_lig_tanimoto | 0.103 | 0.074 | 5 |
| u_lig_mech | 0.267 | 0.127 | 5 |

Macro MAE over all 143 ligands when M of them get one measurement (the
deployable `MIN[u_gp_var_design]` point) and the rest stay zero-shot:

| budget_M | RANDOM | u_tree_sd | u_model_spread | u_lig_mech | u_gp_var_design | ORACLE_gain |
|---|---|---|---|---|---|---|
| 14 | 0.961 | 0.932 | 0.934 | 0.951 | 0.946 | 0.799 |
| 29 | 0.924 | 0.896 | 0.881 | 0.899 | 0.914 | 0.695 |
| 43 | 0.890 | 0.871 | 0.841 | 0.871 | 0.874 | 0.633 |
| 72 | 0.819 | 0.801 | 0.790 | 0.790 | 0.836 | 0.575 |
| 107 | 0.733 | 0.689 | 0.704 | 0.700 | 0.686 | 0.569 |

Paired chemotype bootstrap at M=29, a 20% budget:

| RANDOM_budget minus selector | delta | ci_lo | ci_hi | chemotype macro | seeds /5 |
|---|---|---|---|---|---|
| RANDOM_budget - ORACLE_gain | 0.230 | 0.148 | 0.332 | 0.215 | 5 |
| RANDOM_budget - u_model_spread | 0.043 | -0.031 | 0.094 | 0.001 | 5 |
| RANDOM_budget - u_tree_sd | 0.028 | -0.046 | 0.088 | -0.000 | 5 |
| RANDOM_budget - u_lig_mech | 0.025 | -0.050 | 0.082 | 0.014 | 5 |
| RANDOM_budget - u_tree_sd_mean | 0.022 | -0.052 | 0.073 | -0.018 | 5 |
| RANDOM_budget - u_cond_nn5 | 0.018 | -0.055 | 0.070 | -0.038 | 4 |
| RANDOM_budget - u_lig_tanimoto | 0.011 | -0.041 | 0.053 | 0.029 | 5 |
| RANDOM_budget - u_cond_nn1 | 0.010 | -0.064 | 0.062 | -0.047 | 4 |
| RANDOM_budget - u_gp_var_design | 0.010 | -0.040 | 0.083 | 0.038 | 4 |
| RANDOM_budget - u_gp_var_train | 0.005 | -0.066 | 0.054 | -0.054 | 4 |

Every source is directionally positive and the best of them agrees on
5/5 seeds, but the chemotype interval covers zero and the
chemotype-weighted macro is 0.0008.  **Not established.**  The gap between
rho 0.419 against zero-shot error and rho 0.152 against the gain is the
same phenomenon as 20.4: the one-shot offset removes most of what the
uncertainty was detecting, so knowing where the model is bad barely tells you
where measuring helps.  Oracle triage is worth 0.230 at M=29, so the headroom
is real and unclaimed.

---


## 20.6 Robustness

- **Cross-model mismatch is not the explanation.**  The tree sd comes from a
  sibling model.  Scoring it against the sibling's *own* residuals instead of
  the frozen model's changes nothing:

| target | within_ligand_rho | ci_low | ci_high | across_ligand_rho |
|---|---|---|---|---|
| frozen model residual | 0.139 | 0.064 | 0.250 | 0.475 |
| sibling's own residual | 0.161 | 0.085 | 0.272 | 0.453 |

- **Both tails were tested** for every source (`MAX[...]` and `MIN[...]`), so a
  sign error cannot be hiding a positive result.
- **Ties are broken at random**, not by row order -- the exact failure that made
  the pre-existing `MAX_PREDICTIVE_VARIANCE` arm meaningless.
- **`RANDOM` is exact**: the mean over all candidates, not a 12-repeat sample.
  It lands at 0.709 against the k-shot harness's sampled 0.723 in the pinned
  snapshot.  The oracle differs more (0.522 here vs 0.484 there) because the harness
  picks from a capped random pool and scores on a smaller disjoint evaluation
  subset, which lets its oracle chase evaluation noise.  Every comparison in
  this section is internal to the candidate table, so the offset does not affect
  any claim made here.
- **The independence unit is the chemotype**, not the ligand, in every interval.

---

## 20.7 What to do instead

1. Do not select measurements by model uncertainty.  Both tails lose to random,
   and a *perfect* uncertainty oracle loses to random by 0.070.
2. Select by **design centrality**: `MIN[u_gp_var_design]` (0.639), and
   `CENTRAL` / `MEDOID` (0.651 / 0.648 in the pinned
   snapshot).  Section 19's stratum rules -- mid-acid, central
   lanthanide, middle of the predicted range -- are the same instruction stated
   in chemistry, and they are the whole of the deployable gain.
3. If uncertainty is to earn a place it must be re-aimed.  The estimand is
   `E abs(r_i - median(r_ligand))`, a deviation-from-own-median predictor, not
   an error-magnitude predictor.  Nothing in the stack estimates it; the best
   correlate found here is a labelless design-geometry quantity
   (0.303) rather than any model uncertainty (0.016 for the tree sd).
4. Uncertainty stays legitimate for **reporting**: it ranks which unseen
   extractants the model will be wrong about (across-ligand rho 0.475, 5/5
   seeds), which is what an error bar is for.  It describes the model's state.
   It is not a plan for the next experiment.

---

## Files

| file | contents |
|---|---|
| `row_uncertainty.parquet` | 26,240 held-out row-seeds, residual + 9 uncertainty columns |
| `candidate_uncertainty.parquet` | the 1-shot candidate table joined to those columns |
| `calibration_within_ligand.csv`, `calibration_across_ligand.csv` | (a) Spearman, both units |
| `calibration_curve_pooled.csv`, `calibration_curve_within_ligand.csv` | (a) binned calibration curves |
| `calibration_self_consistency.csv` | sibling sd vs the sibling's own error |
| `decision_spearman.csv`, `decision_rho_detail.parquet` | (b) per-ligand Spearman vs candidate quality |
| `decision_policies.csv`, `policy_detail.parquet` | (b) realised MAE of every MAX / MIN / oracle arm |
| `decision_bootstrap_vs_random.csv` | (b) paired chemotype bootstrap |
| `mechanism_spearman.csv`, `mechanism_detail.parquet`, `level_vs_shape.csv` | (c) the deviation-from-median explanation |
| `centrality_explains_leaderboard.csv` | (c) the one number that orders the leaderboard, with and without the oracle arms |
| `mechanism_oracle_equivalence.csv` | (c) argmin abs(r - median r) against the exhaustive oracle, cell by cell |
| `calibration_within_ligand_common.csv` | (a) the same within-ligand rho on the cells where every source is defined |
| `triage_spearman.csv`, `triage_budget_curve.csv`, `triage_bootstrap.csv`, `triage_ligand_table.csv` | the ligand-level budget question |
| `audit_prior_policies.csv`, `audit_prior_macro.csv`, `audit_prediction_sd_availability.csv` | the degenerate pre-existing arms |
| `gp_axes_used.csv` | which 1-3 condition axes each ligand's GP used |

Built by `scripts/gen8_uncertainty.py` (sources),
`scripts/gen8_uncertainty_analysis.py` (scoring),
`scripts/gen8_uncertainty_report.py` (this file).
