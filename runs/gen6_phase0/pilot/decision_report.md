# gen6 Experiment A — BASE91 vs EXPANDED152 on identical test rows

**Run** `gen6_diversity_causal_20260819T104742Z`  ·  **PILOT** (one seed, reduced trees, one feature set — not the confirmation run)

## What was actually done

* One shared cohort, built once at `min_cells = 3`: **5248 rows, 152 extractants, 131 ECFP clusters, 79 Tanimoto-0.7 super-clusters**, 2055 conditions, 14 metals; target sd 1.657 log units.
* Folds hold out whole super-clusters (5 folds, seeds [104729]). Every arm is scored on **byte-identical test rows**; only the training row mask changes.
* Learner: ExtraTrees, 120 trees, max_features 0.3, min_samples_leaf 2, fold seed `model_seed + fold*1009 + 9999991` (the gen5 formula, reused so a gen5 arm reproduces rather than approximates).
* Weighting: **cluster** — group_balanced_weights on the training rows' ECFP cluster — the gen5 rule, and itself arm-dependent because EXPANDED has more clusters (see 'How to break this result')
* Feature sets: MC_lig2d_ext_massaction. Fitting took 8 s.

### What the expansion adds

BASE (>= 10 cells) is 4881 rows / 91 extractants / 74 ECFP clusters / 40 super-clusters. The expansion adds 367 rows (7.0 % of the cohort) from 61 extractants, of which **57 ECFP clusters and 39 super-clusters are new chemistry** that BASE never sees.

Those added extractants sit at median nearest-neighbour Tanimoto 0.538 to the BASE cohort (25 of 61 below 0.4). That is the chemistry the eligibility rule has been discarding.

### What each arm was actually trained on (mean over folds and seeds)

Read the controls here before reading their contrasts. `EXPANDED_ROWMATCHED` keeps every sparse row and gives back dense rows until it matches BASE's row count, so it is a small perturbation of EXPANDED, not a second BASE — the row-budget clause of A4 is weak evidence whenever the two arms are this close. `n_train_ecfp_clusters` is the number of equal-weight voting blocks the fit sees, i.e. the size of the weighting confound.

```
                arm  n_train_rows  n_train_extractants  n_train_ecfp_clusters
               BASE        3904.8                 72.8                   59.2
           EXPANDED        4198.4                121.6                  104.8
EXPANDED_ROWMATCHED        3904.8                121.6                  104.8
  EXPANDED_SHUFFLED        4198.4                121.6                  104.8
```

## Headline — the primary contrast

`point_delta = statistic(reference) − statistic(candidate)`, so **positive = the candidate arm is better** (lower error). `mae` here is macro MAE with one ECFP cluster = one vote: `per_unit_statistics` defines every statistic as a mean over the unit's rows, so the mean over resampled units is an exact bootstrap of the macro value. The bootstrap resamples **Tanimoto super-clusters** (the units the folds held out), and every ECFP cluster travels with its super-cluster.

```
            feature_set endpoint  statistic  pooled_point_delta  pooled_ci95_low  pooled_ci95_high  mean_point_delta  seeds_positive  n_seeds  pooled_units_total  pooled_blocks
MC_lig2d_ext_massaction      all        mae              0.1390           0.0214            0.3067            0.1390               1        1                 131             79
MC_lig2d_ext_massaction      all offset_mae              0.1248           0.0095            0.2917            0.1248               1        1                 131             79
MC_lig2d_ext_massaction      all  shape_mae              0.0249           0.0024            0.0559            0.0249               1        1                 131             79
MC_lig2d_ext_massaction   nn<0.4        mae              0.4930           0.2123            0.7737            0.4930               1        1                  33             32
MC_lig2d_ext_massaction   nn<0.4 offset_mae              0.4461           0.1452            0.7510            0.4461               1        1                  33             32
MC_lig2d_ext_massaction   nn<0.4  shape_mae              0.0825           0.0316            0.1381            0.0825               1        1                  33             32
MC_lig2d_ext_massaction   nn<0.6        mae              0.1975           0.0325            0.4366            0.1975               1        1                  88             57
MC_lig2d_ext_massaction   nn<0.6 offset_mae              0.1731           0.0002            0.4247            0.1731               1        1                  88             57
MC_lig2d_ext_massaction   nn<0.6  shape_mae              0.0423           0.0136            0.0833            0.0423               1        1                  88             57
```

The same contrast one split partition at a time. The five seeds re-partition the same ligands, so agreement here is a consistency requirement and not five independent experiments; the interval above is the evidence.

```
            feature_set endpoint  split_seed  statistic  point_delta  ci95_low  ci95_high  units_improved  units_total
MC_lig2d_ext_massaction      all      104729        mae       0.1390    0.0214     0.3067              69          131
MC_lig2d_ext_massaction      all      104729 offset_mae       0.1248    0.0095     0.2917              74          131
MC_lig2d_ext_massaction      all      104729  shape_mae       0.0249    0.0024     0.0559              67          131
MC_lig2d_ext_massaction   nn<0.4      104729        mae       0.4930    0.2123     0.7737              24           33
MC_lig2d_ext_massaction   nn<0.4      104729 offset_mae       0.4461    0.1452     0.7510              23           33
MC_lig2d_ext_massaction   nn<0.4      104729  shape_mae       0.0825    0.0316     0.1381              22           33
MC_lig2d_ext_massaction   nn<0.6      104729        mae       0.1975    0.0325     0.4366              49           88
MC_lig2d_ext_massaction   nn<0.6      104729 offset_mae       0.1731    0.0002     0.4247              49           88
MC_lig2d_ext_massaction   nn<0.6      104729  shape_mae       0.0423    0.0136     0.0833              49           88
```

## Pre-registered hypotheses (protocol §5)

Scoring rule, stated so a verdict cannot be over-read: **PASS** = the pre-registered condition is met, including the seed-agreement clause; **FAIL** = the CI95 excludes the predicted direction (evidence *against*); **INCONCLUSIVE** = the interval straddles zero, or the seed clause cannot be evaluated (a single-seed pilot). An INCONCLUSIVE is not a weak PASS.

### Feature set `MC_lig2d_ext_massaction`

**A1 — INCONCLUSIVE**  ·  Adding sparse diverse chemistry improves overall accuracy.

* pass condition: macro_MAE(BASE) − macro_MAE(EXPANDED) > 0 with CI95 low > 0, and ≥ 4/5 seeds positive.
* measured: macro MAE (one ECFP cluster = one vote), all test rows: Δ = +0.1390 log units (CI95 [+0.0214, +0.3067], 131 units, 79 bootstrap blocks; mean over seeds +0.1390, 1/1 seeds positive)
* what would falsify this: A CI95 whose upper end is ≤ 0 — EXPANDED no better, or worse, than BASE on identical test rows — falsifies it, and with it the claim that coverage is the binding constraint at this scale (protocol §8 case C).

**A2 — INCONCLUSIVE**  ·  The gain is largest on the hardest chemistry.

* pass condition: gain at nn<0.4 > gain overall, both CI95 low > 0.
* measured: gain at nn<0.4: Δ = +0.4930 log units (CI95 [+0.2123, +0.7737], 33 units, 32 bootstrap blocks; mean over seeds +0.4930, 1/1 seeds positive); gain overall: Δ = +0.1390 log units (CI95 [+0.0214, +0.3067], 131 units, 79 bootstrap blocks; mean over seeds +0.1390, 1/1 seeds positive); hard > overall: True
* what would falsify this: A gain that is flat or smaller on the far-from-training subset falsifies it: the model would then be improving where it was already supported, i.e. a depth effect wearing a coverage costume.

**A3 — INCONCLUSIVE**  ·  The gain is in the level, not the shape.

* pass condition: offset_mae(BASE) − offset_mae(EXPANDED) CI95 low > 0, and larger than the shape-MAE gain.
* measured: offset gain: Δ = +0.1248 log units (CI95 [+0.0095, +0.2917], 131 units, 79 bootstrap blocks; mean over seeds +0.1248, 1/1 seeds positive); shape gain: Δ = +0.0249 log units (CI95 [+0.0024, +0.0559], 131 units, 79 bootstrap blocks; mean over seeds +0.0249, 1/1 seeds positive); offset > shape: True
* what would falsify this: A gain that lives in the shape while the per-ligand offset is unchanged falsifies it, and points at the representation rather than the coverage (protocol §8 case B).

**A4 — INCONCLUSIVE**  ·  The gain is information, not row count.

* pass condition: EXPANDED beats EXPANDED_ROWMATCHED by less than it beats BASE, and beats EXPANDED_SHUFFLED with CI95 low > 0.
* measured: vs the information null (sparse targets permuted among themselves): Δ = +0.1355 log units (CI95 [+0.0086, +0.3150], 131 units, 79 bootstrap blocks; mean over seeds +0.1355, 1/1 seeds positive); vs the row-budget-matched control: Δ = +0.0010 log units (CI95 [-0.0164, +0.0243], 131 units, 79 bootstrap blocks; mean over seeds +0.0010, 1/1 seeds positive); rowmatched gap < BASE gap: True
* what would falsify this: If EXPANDED_SHUFFLED matches EXPANDED, the extra rows helped as regularisation or through the change in cluster weighting and carried no chemical information; if EXPANDED beats EXPANDED_ROWMATCHED by as much as it beats BASE, the effect is row count, not coverage.

## Where the gain sits — read this before the leaderboard

One ECFP cluster is one vote, and on this cohort a large share of the clusters consist **entirely** of the sparsely-measured ligands that only EXPANDED can train on. If the gain lives only there, the honest claim is 'the expansion repairs chemistry that was missing', not 'the model got better'. `units` counts scoring clusters; `gain` is BASE minus EXPANDED macro MAE within that group.

```
            feature_set endpoint                 kind  units    gain  units_improved
MC_lig2d_ext_massaction      all   BASE-eligible only   72.0 -0.0215          0.4028
MC_lig2d_ext_massaction      all added chemistry only   57.0  0.3539          0.7018
MC_lig2d_ext_massaction      all                mixed    2.0 -0.2068          0.0000
MC_lig2d_ext_massaction   nn<0.4   BASE-eligible only    7.0  0.2402          0.7143
MC_lig2d_ext_massaction   nn<0.4 added chemistry only   26.0  0.5611          0.7308
```

## Arm leaderboard (per feature set, mean over seeds)

Macro MAE is primary. Pooled numbers are printed and never used for selection: the largest single extractant holds 28 % of rows and the largest ECFP cluster 38 %.

```
            feature_set                 arm  macro_mae  offset_mae  shape_mae  shape_r2  pooled_mae  median_ligand_mae  worst_quartile_ligand_mae  frac_within_1_log  n_ligands  n_ecfp_clusters  n_superclusters  n_macro_units  n_eff_pooled_rows
MC_lig2d_ext_massaction                BASE     1.2151      1.0747     0.5246    0.1918      1.2234             1.0933                     2.3136             0.4794      152.0            131.0             79.0          131.0             6.2561
MC_lig2d_ext_massaction            EXPANDED     1.0761      0.9545     0.5046    0.1512      1.2161             0.9568                     2.1940             0.4811      152.0            131.0             79.0          131.0             6.2561
MC_lig2d_ext_massaction EXPANDED_ROWMATCHED     1.0770      0.9451     0.5096    0.1419      1.1933             0.9605                     2.1534             0.4935      152.0            131.0             79.0          131.0             6.2561
MC_lig2d_ext_massaction   EXPANDED_SHUFFLED     1.2115      1.0713     0.5323    0.1574      1.2373             1.0363                     2.3002             0.4788      152.0            131.0             79.0          131.0             6.2561
```

## Hard chemistry — the co-primary endpoint

Bins are cut on `nn_reference_tanimoto`: the maximum Tanimoto from the test ligand to the **BASE training extractants of its own fold**. The same column is used for every arm, so the subset does not move with the arm. `nn_expanded_tanimoto` is carried in the OOF for description only and selects nothing.

```
            feature_set endpoint                 arm  n_rows  n_ligands  n_ecfp_clusters  n_superclusters  macro_mae  offset_mae  shape_mae  frac_within_1_log
MC_lig2d_ext_massaction   nn<0.4                BASE   494.0       33.0             33.0             32.0     1.5502      1.4330     0.4389             0.3462
MC_lig2d_ext_massaction   nn<0.6                BASE  1942.0       94.0             88.0             57.0     1.3898      1.2378     0.5536             0.4624
MC_lig2d_ext_massaction      all                BASE  5248.0      152.0            131.0             79.0     1.2151      1.0747     0.5246             0.4794
MC_lig2d_ext_massaction   nn<0.4            EXPANDED   494.0       33.0             33.0             32.0     1.0572      0.9869     0.3564             0.4615
MC_lig2d_ext_massaction   nn<0.6            EXPANDED  1942.0       94.0             88.0             57.0     1.1923      1.0841     0.5147             0.4748
MC_lig2d_ext_massaction      all            EXPANDED  5248.0      152.0            131.0             79.0     1.0761      0.9545     0.5046             0.4811
MC_lig2d_ext_massaction   nn<0.4 EXPANDED_ROWMATCHED   494.0       33.0             33.0             32.0     1.0557      0.9709     0.3591             0.4534
MC_lig2d_ext_massaction   nn<0.6 EXPANDED_ROWMATCHED  1942.0       94.0             88.0             57.0     1.1893      1.0694     0.5219             0.4943
MC_lig2d_ext_massaction      all EXPANDED_ROWMATCHED  5248.0      152.0            131.0             79.0     1.0770      0.9451     0.5096             0.4935
MC_lig2d_ext_massaction   nn<0.4   EXPANDED_SHUFFLED   494.0       33.0             33.0             32.0     1.6038      1.5403     0.4146             0.3360
MC_lig2d_ext_massaction   nn<0.6   EXPANDED_SHUFFLED  1942.0       94.0             88.0             57.0     1.3866      1.2601     0.5441             0.4418
MC_lig2d_ext_massaction      all   EXPANDED_SHUFFLED  5248.0      152.0            131.0             79.0     1.2115      1.0713     0.5323             0.4788
```

Chemistry units surviving each endpoint (this is the n behind every hard-chemistry CI):

```
            feature_set endpoint  split_seed  n_rows  n_ligands  n_ecfp_clusters  n_superclusters
MC_lig2d_ext_massaction      all    104729.0  5248.0      152.0            131.0             79.0
MC_lig2d_ext_massaction   nn<0.4    104729.0   494.0       33.0             33.0             32.0
MC_lig2d_ext_massaction   nn<0.6    104729.0  1942.0       94.0             88.0             57.0
```

The fixed-reference rule is not a formality — this is how many test rows would have landed in a different bin had each arm's own training set defined it (mean over seeds and feature sets):

```
endpoint  split_seed  n_rows_reference_bins  n_rows_if_each_arm_used_its_own_training_set  n_rows_that_would_change_bin
  nn<0.4    104729.0                  494.0                                         396.0                          98.0
  nn<0.6    104729.0                 1942.0                                        1282.0                         660.0
```

## Every contrast, every endpoint

```
            feature_set endpoint                      comparison  statistic  pooled_point_delta  pooled_ci95_low  pooled_ci95_high  mean_point_delta  seeds_positive  n_seeds  pooled_units_total  preregistered
MC_lig2d_ext_massaction      all                EXPANDED_vs_BASE        mae              0.1390           0.0214            0.3067            0.1390               1        1                 131           True
MC_lig2d_ext_massaction      all                EXPANDED_vs_BASE offset_mae              0.1248           0.0095            0.2917            0.1248               1        1                 131           True
MC_lig2d_ext_massaction      all                EXPANDED_vs_BASE  shape_mae              0.0249           0.0024            0.0559            0.0249               1        1                 131           True
MC_lig2d_ext_massaction      all     EXPANDED_ROWMATCHED_vs_BASE        mae              0.1380           0.0303            0.2991            0.1380               1        1                 131           True
MC_lig2d_ext_massaction      all     EXPANDED_ROWMATCHED_vs_BASE offset_mae              0.1346           0.0224            0.3017            0.1346               1        1                 131           True
MC_lig2d_ext_massaction      all     EXPANDED_ROWMATCHED_vs_BASE  shape_mae              0.0189          -0.0024            0.0477            0.0189               1        1                 131           True
MC_lig2d_ext_massaction      all   EXPANDED_vs_EXPANDED_SHUFFLED        mae              0.1355           0.0086            0.3150            0.1355               1        1                 131           True
MC_lig2d_ext_massaction      all   EXPANDED_vs_EXPANDED_SHUFFLED offset_mae              0.1318           0.0033            0.3155            0.1318               1        1                 131           True
MC_lig2d_ext_massaction      all   EXPANDED_vs_EXPANDED_SHUFFLED  shape_mae              0.0291           0.0088            0.0592            0.0291               1        1                 131           True
MC_lig2d_ext_massaction      all       EXPANDED_SHUFFLED_vs_BASE        mae              0.0035          -0.0592            0.0584            0.0035               1        1                 131           True
MC_lig2d_ext_massaction      all       EXPANDED_SHUFFLED_vs_BASE offset_mae             -0.0070          -0.0799            0.0489           -0.0070               0        1                 131           True
MC_lig2d_ext_massaction      all       EXPANDED_SHUFFLED_vs_BASE  shape_mae             -0.0042          -0.0338            0.0257           -0.0042               0        1                 131           True
MC_lig2d_ext_massaction      all EXPANDED_vs_EXPANDED_ROWMATCHED        mae              0.0010          -0.0164            0.0243            0.0010               1        1                 131          False
MC_lig2d_ext_massaction      all EXPANDED_vs_EXPANDED_ROWMATCHED offset_mae             -0.0098          -0.0310            0.0135           -0.0098               0        1                 131          False
MC_lig2d_ext_massaction      all EXPANDED_vs_EXPANDED_ROWMATCHED  shape_mae              0.0060          -0.0004            0.0150            0.0060               1        1                 131          False
MC_lig2d_ext_massaction   nn<0.4                EXPANDED_vs_BASE        mae              0.4930           0.2123            0.7737            0.4930               1        1                  33           True
MC_lig2d_ext_massaction   nn<0.4                EXPANDED_vs_BASE offset_mae              0.4461           0.1452            0.7510            0.4461               1        1                  33           True
MC_lig2d_ext_massaction   nn<0.4                EXPANDED_vs_BASE  shape_mae              0.0825           0.0316            0.1381            0.0825               1        1                  33           True
MC_lig2d_ext_massaction   nn<0.4     EXPANDED_ROWMATCHED_vs_BASE        mae              0.4945           0.2092            0.7797            0.4945               1        1                  33           True
MC_lig2d_ext_massaction   nn<0.4     EXPANDED_ROWMATCHED_vs_BASE offset_mae              0.4621           0.1557            0.7767            0.4621               1        1                  33           True
MC_lig2d_ext_massaction   nn<0.4     EXPANDED_ROWMATCHED_vs_BASE  shape_mae              0.0798           0.0282            0.1364            0.0798               1        1                  33           True
MC_lig2d_ext_massaction   nn<0.4   EXPANDED_vs_EXPANDED_SHUFFLED        mae              0.5467           0.2608            0.8245            0.5467               1        1                  33           True
MC_lig2d_ext_massaction   nn<0.4   EXPANDED_vs_EXPANDED_SHUFFLED offset_mae              0.5534           0.2587            0.8507            0.5534               1        1                  33           True
MC_lig2d_ext_massaction   nn<0.4   EXPANDED_vs_EXPANDED_SHUFFLED  shape_mae              0.0581           0.0120            0.1062            0.0581               1        1                  33           True
MC_lig2d_ext_massaction   nn<0.4       EXPANDED_SHUFFLED_vs_BASE        mae             -0.0536          -0.1975            0.0808           -0.0536               0        1                  33           True
MC_lig2d_ext_massaction   nn<0.4       EXPANDED_SHUFFLED_vs_BASE offset_mae             -0.1073          -0.2739            0.0440           -0.1073               0        1                  33           True
MC_lig2d_ext_massaction   nn<0.4       EXPANDED_SHUFFLED_vs_BASE  shape_mae              0.0243          -0.0330            0.0875            0.0243               1        1                  33           True
MC_lig2d_ext_massaction   nn<0.4 EXPANDED_vs_EXPANDED_ROWMATCHED        mae             -0.0015          -0.0399            0.0338           -0.0015               0        1                  33          False
MC_lig2d_ext_massaction   nn<0.4 EXPANDED_vs_EXPANDED_ROWMATCHED offset_mae             -0.0160          -0.0605            0.0254           -0.0160               0        1                  33          False
MC_lig2d_ext_massaction   nn<0.4 EXPANDED_vs_EXPANDED_ROWMATCHED  shape_mae              0.0027          -0.0049            0.0111            0.0027               1        1                  33          False
MC_lig2d_ext_massaction   nn<0.6                EXPANDED_vs_BASE        mae              0.1975           0.0325            0.4366            0.1975               1        1                  88           True
MC_lig2d_ext_massaction   nn<0.6                EXPANDED_vs_BASE offset_mae              0.1731           0.0002            0.4247            0.1731               1        1                  88           True
MC_lig2d_ext_massaction   nn<0.6                EXPANDED_vs_BASE  shape_mae              0.0423           0.0136            0.0833            0.0423               1        1                  88           True
MC_lig2d_ext_massaction   nn<0.6     EXPANDED_ROWMATCHED_vs_BASE        mae              0.2006           0.0417            0.4300            0.2006               1        1                  88           True
MC_lig2d_ext_massaction   nn<0.6     EXPANDED_ROWMATCHED_vs_BASE offset_mae              0.1885           0.0173            0.4403            0.1885               1        1                  88           True
MC_lig2d_ext_massaction   nn<0.6     EXPANDED_ROWMATCHED_vs_BASE  shape_mae              0.0347           0.0067            0.0751            0.0347               1        1                  88           True
MC_lig2d_ext_massaction   nn<0.6   EXPANDED_vs_EXPANDED_SHUFFLED        mae              0.1943           0.0237            0.4385            0.1943               1        1                  88           True
MC_lig2d_ext_massaction   nn<0.6   EXPANDED_vs_EXPANDED_SHUFFLED offset_mae              0.1921           0.0148            0.4470            0.1921               1        1                  88           True
MC_lig2d_ext_massaction   nn<0.6   EXPANDED_vs_EXPANDED_SHUFFLED  shape_mae              0.0296           0.0078            0.0590            0.0296               1        1                  88           True
MC_lig2d_ext_massaction   nn<0.6       EXPANDED_SHUFFLED_vs_BASE        mae              0.0032          -0.0791            0.0748            0.0032               1        1                  88           True
MC_lig2d_ext_massaction   nn<0.6       EXPANDED_SHUFFLED_vs_BASE offset_mae             -0.0190          -0.1096            0.0550           -0.0190               0        1                  88           True
MC_lig2d_ext_massaction   nn<0.6       EXPANDED_SHUFFLED_vs_BASE  shape_mae              0.0126          -0.0179            0.0539            0.0126               1        1                  88           True
MC_lig2d_ext_massaction   nn<0.6 EXPANDED_vs_EXPANDED_ROWMATCHED        mae             -0.0030          -0.0233            0.0245           -0.0030               0        1                  88          False
MC_lig2d_ext_massaction   nn<0.6 EXPANDED_vs_EXPANDED_ROWMATCHED offset_mae             -0.0155          -0.0410            0.0119           -0.0155               0        1                  88          False
MC_lig2d_ext_massaction   nn<0.6 EXPANDED_vs_EXPANDED_ROWMATCHED  shape_mae              0.0076          -0.0003            0.0167            0.0076               1        1                  88          False
```

Per-seed intervals (the same contrasts, one split partition at a time) are in `contrasts.csv`.

## Out-of-distribution behaviour

Error against distance to the fold's BASE training chemistry, with the forest's own dispersion (sd across trees) where it was recorded. A model that knows where it is unsupported is worth more than one that always emits a number. `mean_prediction_sd` is recorded for **EXPANDED arm only** and is blank elsewhere — it is a property of one fitted forest, not a comparison between arms.

```
            feature_set                 arm        bin  n_rows  n_ligands  pooled_mae  macro_mae_ligand  offset_mae  shape_mae  mean_prediction_sd
MC_lig2d_ext_massaction                BASE [0.0, 0.4)   494.0       33.0      1.4539            1.5502      1.4330     0.4389                 NaN
MC_lig2d_ext_massaction                BASE [0.4, 0.6)  1448.0       61.0      1.2647            1.2925      1.1322     0.6157                 NaN
MC_lig2d_ext_massaction                BASE [0.6, 0.8)  3306.0       58.0      1.1708            0.9689      0.8103     0.4776                 NaN
MC_lig2d_ext_massaction            EXPANDED [0.0, 0.4)   494.0       33.0      1.2223            1.0572      0.9869     0.3564              1.3310
MC_lig2d_ext_massaction            EXPANDED [0.4, 0.6)  1448.0       61.0      1.3287            1.2873      1.1367     0.6003              1.1965
MC_lig2d_ext_massaction            EXPANDED [0.6, 0.8)  3306.0       58.0      1.1658            0.9099      0.7445     0.4882              1.2531
MC_lig2d_ext_massaction EXPANDED_ROWMATCHED [0.0, 0.4)   494.0       33.0      1.2241            1.0557      0.9709     0.3591                 NaN
MC_lig2d_ext_massaction EXPANDED_ROWMATCHED [0.4, 0.6)  1448.0       61.0      1.2899            1.2824      1.1227     0.6100                 NaN
MC_lig2d_ext_massaction EXPANDED_ROWMATCHED [0.6, 0.8)  3306.0       58.0      1.1464            0.9166      0.7437     0.4896                 NaN
MC_lig2d_ext_massaction   EXPANDED_SHUFFLED [0.0, 0.4)   494.0       33.0      1.5646            1.6038      1.5403     0.4146                 NaN
MC_lig2d_ext_massaction   EXPANDED_SHUFFLED [0.4, 0.6)  1448.0       61.0      1.3177            1.2682      1.1085     0.6141                 NaN
MC_lig2d_ext_massaction   EXPANDED_SHUFFLED [0.6, 0.8)  3306.0       58.0      1.1532            0.9431      0.7654     0.5133                 NaN
```

## How to break this result

* **The weighting confound.** The primary fit weights training rows so every ECFP cluster carries equal total weight, and EXPANDED has more clusters than BASE — so the arms differ in their weighting as well as their chemistry. Re-run with `--weighting row` and compare; if the effect vanishes, it was the weighting.
* **Seeds are not replicates.** The five split seeds re-partition the same ligands. The bootstrap over super-clusters is the load-bearing statistic; seed agreement is a second requirement, not independent evidence — and a weak one: the seed-to-seed sd of the macro delta is ~0.012 against a cluster-robust standard error of ~0.076, a variance component 36x smaller.
* **Multiplicity.** A1-A4 are pre-registered and are the only protected claims. The supporting tables report many more contrasts (comparisons x statistics x endpoints x feature sets), none of them corrected for multiple comparisons. Treat any interval outside A1-A4 as descriptive.
* **The gain is concentrated, not broad — read the decomposition above.** The shared cohort is `min_cells = 3`, so the sparse ligands are *test* rows too, and they carry the effect. BASE trains on none of them in any fold, so on the clusters made only of added chemistry the contrast is close to 'the arm allowed to learn this chemistry predicts it better'. What separates it from a tautology is that a test ligand's own chemotype is held out of every arm, and that the gain scales with how much closer the addition actually brought training (the `closeness gained` table). An earlier version of this line claimed the opposite — that the numbers say nothing about sparse test rows — which was backwards.
* **`EXPANDED_SHUFFLED` is a HARDER comparator than BASE, not an equal one.** Mislabelled chemistry is worse than absent chemistry, so the shuffled arm lands below BASE and the EXPANDED-vs-SHUFFLED delta overstates the effect size. Use it to rule out regularisation and the weighting change; quote the BASE contrast for the magnitude.
* **Do not quote the percentile interval alone at the `all` endpoint.** One super-cluster holds ~21 % of the scoring units inside 1 of the bootstrap blocks, which inflates the percentile interval's real Type-I rate; BCa, cluster-robust and block-macro columns are reported beside it for that reason.

