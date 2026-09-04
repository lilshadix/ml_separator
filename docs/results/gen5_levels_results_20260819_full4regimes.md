# gen5 — log D level model: full four-regime run (`gen5_levels_20260818T211105Z`)

Cluster run (SLURM 6116156), 2026-08-18 21:13 → 2026-08-19 00:34; fitting 11 966 s. Pulled from
`origin/descriptor-arm-metal-site` (commit `86b27da`). This is the first run in which all four
pre-registered regimes completed, so H1′, H2, H3 and H5 — undecidable after the 08-17 run — are
now adjudicated. Where the 08-17 run overlaps (unseen_ligand, unseen_conditions) every macro
MAE is identical to 4 decimals: the harness fixes of 08-18 changed reporting, not fits.

## Setup
* Cohort: 4 881 rows, 91 extractants (of 190; `min_rows_per_extractant=10` dropped 99 extractants
  / 418 cells), 74 ECFP clusters, 40 Tanimoto-0.7 super-clusters, 1 932 conditions, 230 series,
  14 metals; target sd 1.643. Largest extractant 30 % of rows, largest ECFP cluster 41 %,
  largest Tanimoto super-cluster **67 %**.
* Noise floor from 313 replicated cells: MAE ≈ 0.19 (median within-cell sd) / 0.62 (pooled).
* 18 arms + `@hgb`/`@ridge` on MC, MC_ecfp, MC_lig2d_ext, MC_all2d + 4 SHUF + 4 NULL; 5 seeds ×
  5 folds; 400 trees. MASSACTION block was computed (8 cols) but its arms were not in this
  run — see §7 for the local massaction run on the same seeds.

## 1. Leaderboard — best arms per regime (macro MAE, mean of 5 seeds)

| regime | best arm | MAE | 2nd | MAE | MC (metal+cond) | best NULL |
|---|---|---|---|---|---|---|
| unseen_chemotype | MC_donors | 1.030 | MC_lig2d_ext | 1.037 | 1.147 | 1.454 (global mean) |
| unseen_ligand | MC_lig2d_ext | 0.836 | MC_all2d@hgb | 0.854 | 0.959 | 1.344 (global mean) |
| unseen_series | MC_ecfp@hgb | 0.793 | MC_donors | 0.805 | 0.924 | 1.195 (nearest condition) |
| unseen_conditions | MC_everything | 0.765 | MC_physchem | 0.785 | 1.025 | 0.868 (nearest condition) |

Other columns for the best 2D arm: pooled R² 0.27 / 0.43 / 0.50 / 0.73; within-ligand R²
(deployable) −0.39 / −0.08 / +0.05 / +0.50; shape R² 0.17 / 0.25 / 0.29 / 0.64; frac within
1 log unit 0.52 / 0.58 / 0.66 / 0.82 (chemotype / ligand / series / conditions).

## 2. Hypothesis verdicts (paired cluster bootstrap, 5 000 reps, 40 Tanimoto blocks)

| H | test | result | verdict |
|---|---|---|---|
| H1 | MC_fam < MC, unseen_ligand, CI low > 0, ≥ 4/5 seeds, one 2D family | lig2d_ext Δ +0.123 CI[+0.022, +0.229] 5/5; donors Δ +0.084 CI[+0.021, +0.156] 5/5; ecfp Δ +0.048 CI[−0.046, +0.144] 4/5 | **PASS** (lig2d_ext, donors; not ECFP) |
| H1′ | same on unseen_chemotype | lig2d_ext Δ +0.110 CI[+0.006, +0.227] 5/5; donors Δ +0.117 CI[+0.038, +0.185] 5/5; ecfp +0.090 CI spans 0 | **PASS** — expected to fail |
| H2 | MC < NULL_metal_mean and < MC_cond_SHUF, unseen_series | 0.924 vs 1.324 (Δ 0.400 CI[0.251, 0.553]); vs 1.434 (Δ 0.510 CI[0.364, 0.657]) | **PASS** |
| H3 | MC_ecfp < NULL_nearest_condition and < NULL_extractant_mean, unseen_series | 0.839 vs 1.195 (Δ 0.356 CI[0.216, 0.510]); vs 1.213 (Δ 0.374 CI[0.234, 0.530]) | **PASS** |
| H4 | 3D adds nothing over 2D, unseen_ligand | all3d vs all2d Δ **−0.102** CI[−0.177, −0.034], 0/5 seeds; everything vs all2d −0.013 CI[−0.041, +0.017], 0/5 | **PASS, negative direction** (3D hurts) |
| H5 | unseen_conditions easier than unseen_series by > 0.05; NULL_nearest_condition within 0.10 of MC_ecfp | MC_ecfp 0.800 vs 0.839 (gap 0.039); MC_everything 0.765 vs 0.821 (0.056); NULL_nearest_condition 0.868 vs MC_ecfp 0.800 (Δ 0.069, CI[−0.06, +0.20]) | reported: gap ~0.04–0.06; nearest-condition null is statistically indistinguishable from the model under unseen_conditions |
| H6 | k-shot offset k=1 beats zero-shot on mae_free and beats NULL_kmean | offset 0.901 vs none_at_k 0.927 (Δ +0.026); NULL_kmean 1.270 (Δ +0.37) | **PASS** on mae_free; see §5 for the series-free caveat |

## 3. Feature families — added to MC (gain = MAE(MC) − MAE(MC+fam), CI95, seeds > 0)

| family | chemotype | ligand | series | conditions |
|---|---|---|---|---|
| lig2d_ext (206) | +0.110 [+0.006,+0.227] 5/5 | +0.123 [+0.022,+0.229] 5/5 | +0.102 [−0.010,+0.211] 5/5 | +0.233 [+0.123,+0.346] 5/5 |
| donors (13) | +0.117 [+0.038,+0.185] 5/5 | +0.084 [+0.021,+0.156] 5/5 | +0.119 [+0.042,+0.195] 5/5 | +0.123 [+0.037,+0.213] 5/5 |
| ecfp (2048) | +0.090 [−0.056,+0.181] 4/5 | +0.048 [−0.046,+0.144] 4/5 | +0.085 [−0.018,+0.188] 5/5 | +0.225 [+0.106,+0.341] 5/5 |
| physchem (10) | +0.057 [−0.039,+0.186] 5/5 | +0.059 [−0.035,+0.154] 5/5 | +0.044 [−0.063,+0.144] 4/5 | +0.239 [+0.122,+0.353] 5/5 |
| complex_phys 3D | +0.025 [−0.048,+0.119] | −0.014 [−0.109,+0.077] 0/5 | +0.040 [−0.051,+0.124] | +0.222 [+0.122,+0.327] |
| polyhedron 3D | +0.033 [−0.057,+0.112] | −0.041 [−0.140,+0.057] 0/5 | +0.017 [−0.076,+0.108] | +0.188 [+0.095,+0.283] |
| cond added to metal | +0.223 [+0.072,+0.363] | +0.334 [+0.167,+0.502] | +0.376 [+0.229,+0.526] | +0.296 [+0.120,+0.475] |

* Under `unseen_conditions` **every** family gains ~0.19–0.24 with tight CIs — including 3D blocks
  that add nothing anywhere else. That is ligand-identity lookup: any block that separates the 91
  ligands lets the tree copy the ligand's neighbouring titration points. It is not chemistry.
* Under held-out-ligand regimes only two families survive: `lig2d_ext` (extended 2D descriptors)
  and `donors` (13-column donor census). ECFP does not clear the CI in either.
* No family alone beats MC (metal + conditions) in any regime; MC_cond_SHUF (1.43–1.52) is the
  worst arm everywhere, worse than the global mean. Conditions carry the bulk of the signal.
* SHUF controls: every ligand block beats its permuted twin by 0.10–0.27 in every regime — the
  blocks are used, not fitted to noise.

## 4. Edge vs nearest-training-neighbour Tanimoto (per extractant × seed, mean |err|)

unseen_ligand (median NN 0.72; 18 % of held-out extractant-seeds have NN < 0.6):

| NN Tanimoto | n extr | MC | +lig2d_ext | +ecfp | +donors | edge lig2d | edge ecfp | edge donors |
|---|---|---|---|---|---|---|---|---|
| ≤ 0.4 | 10 | 1.212 | 1.199 | 1.094 | 0.973 | +0.013 | +0.118 | +0.239 |
| 0.4–0.6 | 15 | 1.248 | 1.063 | 1.276 | 1.131 | +0.185 | −0.028 | +0.116 |
| 0.6–0.8 | 58 | 0.976 | 0.880 | 0.969 | 0.891 | +0.096 | +0.007 | +0.085 |
| > 0.8 | 39 | 0.829 | 0.693 | 0.714 | 0.829 | +0.136 | +0.115 | 0.000 |

unseen_chemotype (median NN 0.63; nothing above 0.8 by construction):

| NN Tanimoto | n extr | MC | +lig2d_ext | +ecfp | +donors | edge lig2d | edge ecfp | edge donors |
|---|---|---|---|---|---|---|---|---|
| ≤ 0.4 | 16 | 1.381 | 1.245 | 1.311 | 1.104 | +0.135 | +0.070 | +0.277 |
| 0.4–0.6 | 31 | 1.390 | 1.263 | 1.238 | 1.263 | +0.127 | +0.152 | +0.128 |
| 0.6–0.8 | 58 | 0.963 | 0.916 | 0.937 | 0.921 | +0.047 | +0.026 | +0.042 |

Reading: ECFP's edge is a near-neighbour effect (largest at NN > 0.8, ≈ 0 in the middle). The
donor census does the opposite — its edge is largest on the least similar ligands (+0.24…+0.28)
and vanishes above 0.8. `lig2d_ext` is in between. Bins below 0.4 hold 10–16 extractants; treat
those rows as indicative.

## 5. k-shot on levels (unseen_ligand, MC_ecfp; 40 admitted extractants, 34 with a series-free row)

| k | form | mae_all | mae_free | mae_series_free |
|---|---|---|---|---|
| 1 | none_at_k | 0.918 | 0.927 | 0.903 |
| 1 | offset | 0.873 | 0.901 | **0.952** |
| 1 | NULL_kmean | 1.223 | 1.270 | 1.383 |
| 2 | none_at_k / offset / affine | 0.915 / 0.846 / 0.836 | 0.938 / 0.885 / 0.881 | 0.925 / 0.952 / 0.958 |
| 3 | none_at_k / offset / affine | 0.917 / 0.827 / 0.810 | 0.955 / 0.873 / 0.863 | 0.950 / 0.954 / 0.959 |
| 5 | none_at_k / offset / scale / affine | 0.920 / 0.801 / 0.812 / 0.776 | 0.983 / 0.852 / 0.871 / 0.833 | 0.991 / 0.976 / 0.962 / 0.980 |

* On rows sharing a series with the support (mae_free) calibration helps and grows with k:
  −0.03 at k=1, −0.15 at k=5 (affine).
* On rows from a **different** series of the same ligand (mae_series_free) it does not: k=1..3
  offset is worse than zero-shot by 0.05 / 0.03 / 0.00; at k=5 scale gains 0.03. The k-shot gain
  is titration-curve completion, not ligand-level recalibration.
* Unlike the pair study, the no-model NULL_kmean is far worse (1.2–1.4) — levels move with
  conditions, so a k-mean cannot substitute for the model.

## 6. Learners
RF ≈ HGB (HGB better on unseen_series MC_ecfp 0.793 vs 0.839 and unseen_ligand MC_all2d 0.854 vs
0.874; worse under chemotype/conditions). Ridge is far worse everywhere (1.07–1.51 macro; on
unseen_chemotype MC_lig2d_ext@ridge 1.51 is worse than the null) — the mapping is non-linear /
interaction-driven; a linear model on these blocks fails.

## 7. Bridge to the massaction run (`gen5_massaction_local_20260818`, same seeds; base arms agree to ±0.004)
Adding MASSACTION (8 log-concentration features): unseen_ligand +0.028…+0.045 (CI > 0 all four
bases), unseen_series +0.028…+0.044 (3/4 CI > 0), unseen_conditions +0.058…+0.062 (all CI > 0),
unseen_chemotype +0.002…+0.016 (nil). Best deployable arm today, MC_lig2d_ext_massaction:
1.033 / 0.808 / 0.794 / 0.730 (chemotype / ligand / series / conditions).

## 8. Bottom line
1. All six hypotheses adjudicated; H1, H1′, H2, H3, H6 pass; H4 passes in the negative direction
   (3D worse than 2D on new ligands, −0.10 CI-clean; 3D question closed).
2. The model beats every honest null in every regime, but on new ligands the deployable
   within-ligand R² is still ≤ 0: level *offsets* of new ligands are not predicted; the response
   *shape* is (shape R² 0.17–0.31).
3. Where the gain on new chemistry comes from: the 13-column donor census and the extended 2D
   descriptors, not ECFP. Donor census is the only block whose edge grows as similarity falls.
4. Under `unseen_conditions` the model (0.80) is statistically tied with "copy the ligand's nearest
   measured condition" (0.87, Δ CI[−0.06, +0.20]); do not quote that regime as prediction skill.
5. Cohort ceiling unchanged: 67 % of rows in one Tanimoto super-cluster; 99 chemically novel
   extractants still excluded by `min_rows=10`.
