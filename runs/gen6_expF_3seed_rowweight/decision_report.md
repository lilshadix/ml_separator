# gen6 Experiment F — which ligand should have been measured next?

**Run** `gen6_acquisition_sim_20260819T143320Z`

## What was actually done

* One shared cohort at `min_cells = 3`: **5248 rows, 152 extractants, 131 ECFP clusters, 79 Tanimoto-0.7 chemotypes**; target sd 1.657 log units.
* Folds hold out whole chemotypes (Experiment A's `diversity_splits`, 5 folds, seeds [104729, 130363, 155921]). **The fold's held-out chemotypes are the test set and are fixed**; acquisition draws only from the EXPANDED training index.
* Start cohort: the 10 most-measured training ligands of the largest training chemotype, all their rows.
* Acquisition: up to 30 steps, 3 rows revealed per acquisition, checkpoints [1, 2, 3, 5, 8, 12, 16, 20, 25, 30], 2 replicate(s) per (seed, fold, policy).
* Learner: ExtraTrees, 200 trees, max_features 0.3, min_samples_leaf 2; feature columns from blocks ['METAL', 'COND', 'LIG2D_EXT', 'MASSACTION']; fold seed `model_seed + fold*1009 + 9999991 + 101*replicate`.
* Frozen chemistry map over 190 extractants (164 ECFP clusters, 98 chemotypes at threshold 0.7).
* Arms: primary ['random', 'maxmin', 'diversity_x_uncertainty', 'same_chemotype_first'].
* Simulation took 749 s. First fold took 45.5 s; 15 fold-jobs → estimated 682 s (11.4 min) of simulation (plus bootstrap and reporting).

## How to read this

* **macro MAE** = one ECFP cluster, one vote (the study's primary metric). Pooled MAE is in the CSVs and never used for selection: the largest single ligand holds 28 % of rows.
* **hard chemistry is a fixed subset.** `hard_nn<0.4` means: test rows whose maximum Tanimoto to the **start cohort's** extractants is below the threshold. It is computed once per fold, before any acquisition, and never recomputed — so the same rows are scored for every policy at every step. Recomputing it against the current training set would shrink the subset exactly for the policies that buy distant chemistry, which would manufacture the result this experiment is trying to measure.
* **checkpoint 0 is the start cohort alone**, identical for every arm by construction (measured max spread across arms: 3.11e-15). It is excluded from the hypothesis scoring, where every delta there is structurally zero.
* **`point_delta = statistic(reference) − statistic(candidate)`**, so a **positive delta means the candidate policy is better** (lower error). Every contrast below uses `random` as the reference except F4's, which uses the depth arm.
* The paired bootstrap resamples **chemotype blocks** (5000 replicates) with the ECFP cluster as the scoring unit, and reports BCa and cluster-robust intervals beside the percentile one — the Phase 1 dominant-block caveat applies to the `all` endpoint and not to the hard subsets.
* A policy is a **distribution over acquisition sequences**. Predictions are averaged over replicates before the bootstrap; the per-replicate spread is in `acquisition_curves.csv` and in the sd columns of `curve_summary.csv`.

## The start cohorts (this is the experiment's premise, so read it first)

A narrow, deep start is the situation this project was in for five generations. One fold degenerates: holding out the diglycolamide chemotype leaves a largest-training-chemotype with almost no ligands in it. That fold is disclosed, not excluded, and it drags every curve towards zero effect because there is barely a start cohort to improve on.

```
 split_seed  fold start_chemotype  n_start_ligands  n_start_rows  largest_chemotype_ligands  n_pool_ligands  n_pool_rows  start_is_short  n_test_rows  n_hard_rows_0_4
     104729     0          tan055               10          2928                         38             120         1938           False          382              170
     104729     1          tan016                2           244                          2              80         1251            True         3753             3753
     104729     2          tan055               10          2928                         38             124         2034           False          286              150
     104729     3          tan055               10          2928                         38             124         1819           False          501              435
     104729     4          tan055               10          2928                         38             118         1994           False          326              258
     130363     0          tan084                1           140                          1              97         1296            True         3812             3812
     130363     1          tan055               10          2928                         38             121         2009           False          311              171
     130363     2          tan055               10          2928                         38             115         1838           False          482              261
     130363     3          tan055               10          2928                         38             114         1908           False          412              253
     130363     4          tan055               10          2928                         38             120         2089           False          231               50
     155921     0          tan016                2           244                          2              94         1304            True         3700             3670
     155921     1          tan055               10          2928                         38             120         1900           False          420              196
     155921     2          tan055               10          2928                         38             122         2081           False          239               85
     155921     3          tan055               10          2928                         38             123         1801           False          519              471
     155921     4          tan055               10          2928                         38             107         1950           False          370              211
```

## Learning curves (mean over folds, seeds and replicates)

**macro MAE, all test rows (one ECFP cluster = one vote)**

```
arm         random  maxmin  diversity_x_uncertainty  same_chemotype_first
checkpoint                                                               
0           1.9546  1.9546                   1.9546                1.9546
1           1.8926  1.8549                   1.8569                1.9427
2           1.8116  1.7003                   1.7074                1.9263
3           1.6966  1.6858                   1.5798                1.9100
5           1.5784  1.4646                   1.4713                1.8932
8           1.5041  1.4080                   1.3631                1.8610
12          1.4326  1.3329                   1.2586                1.8192
16          1.3831  1.2810                   1.1746                1.8253
20          1.3439  1.2233                   1.1513                1.8838
25          1.3198  1.1848                   1.1187                1.9234
30          1.2864  1.1932                   1.0845                1.8372
```

**macro MAE on hard chemistry (hard_nn<0.4; the FIXED start-cohort subset)**

```
arm         random  maxmin  diversity_x_uncertainty  same_chemotype_first
checkpoint                                                               
0           2.0952  2.0952                   2.0952                2.0952
1           2.0263  1.9925                   1.9924                2.0828
2           1.9340  1.7842                   1.8220                2.0704
3           1.8140  1.7654                   1.6700                2.0570
5           1.6721  1.5015                   1.5195                2.0399
8           1.6023  1.4146                   1.3736                2.0065
12          1.5215  1.3042                   1.2226                1.9588
16          1.4603  1.2394                   1.1304                1.9667
20          1.4095  1.1712                   1.1126                2.0239
25          1.3870  1.1367                   1.0834                2.0729
30          1.3347  1.1564                   1.0487                1.9851
```

**offset MAE (per-ligand level error — the quantity Phase 1 says is missing)**

```
arm         random  maxmin  diversity_x_uncertainty  same_chemotype_first
checkpoint                                                               
0           1.8472  1.8472                   1.8472                1.8472
1           1.7882  1.7527                   1.7518                1.8382
2           1.7002  1.5783                   1.5764                1.8170
3           1.5704  1.5664                   1.4333                1.7988
5           1.4475  1.3134                   1.3313                1.7857
8           1.3528  1.2715                   1.2169                1.7451
12          1.2808  1.2022                   1.1022                1.7002
16          1.2216  1.1385                   1.0093                1.7128
20          1.1835  1.0743                   0.9790                1.7892
25          1.1511  1.0364                   0.9385                1.8353
30          1.1126  1.0487                   0.8914                1.7300
```

**shape MAE (within-ligand response shape)**

```
arm         random  maxmin  diversity_x_uncertainty  same_chemotype_first
checkpoint                                                               
0           0.6212  0.6212                   0.6212                0.6212
1           0.6090  0.6007                   0.6036                0.6156
2           0.5949  0.5828                   0.5967                0.6149
3           0.5894  0.5770                   0.5866                0.6138
5           0.5755  0.5622                   0.5679                0.6119
8           0.5716  0.5594                   0.5538                0.6112
12          0.5668  0.5435                   0.5473                0.6049
16          0.5614  0.5417                   0.5519                0.6066
20          0.5551  0.5405                   0.5483                0.5979
25          0.5547  0.5403                   0.5450                0.6004
30          0.5527  0.5377                   0.5415                0.6031
```

**worst-quartile ligand MAE**

```
arm         random  maxmin  diversity_x_uncertainty  same_chemotype_first
checkpoint                                                               
0           3.3634  3.3634                   3.3634                3.3634
1           3.2729  3.2295                   3.2364                3.3615
2           3.1796  3.0210                   2.9928                3.3395
3           3.0125  3.0049                   2.7978                3.3100
5           2.8138  2.6364                   2.6402                3.2872
8           2.6954  2.5321                   2.4900                3.2482
12          2.6066  2.4997                   2.3550                3.1948
16          2.5233  2.4617                   2.1665                3.2167
20          2.4835  2.3655                   2.1248                3.3188
25          2.4659  2.2740                   2.0894                3.4021
30          2.4276  2.2795                   2.0291                3.2947
```

### Coverage — what each policy actually bought

The mechanism, not decoration: if two policies buy the same chemistry, no difference in accuracy is expected. `n_train_chemotypes` counts Tanimoto-0.7 chemotypes among the training ligands; `mean_nn_test_to_train` is the mean over **test ligands** of the maximum Tanimoto to the current training ligands (higher = the test set is better covered).

**chemotypes in training**

```
arm         random  maxmin  diversity_x_uncertainty  same_chemotype_first
checkpoint                                                               
0            1.000     1.0                    1.000                 1.000
1            1.833     2.0                    2.000                 1.200
2            2.633     3.0                    3.000                 1.267
3            3.500     4.0                    4.000                 1.400
5            5.100     6.0                    6.000                 1.533
8            7.367     9.0                    9.000                 1.933
12          10.433    13.0                   13.000                 2.267
16          12.867    17.0                   17.000                 2.400
20          15.700    21.0                   21.000                 2.800
25          19.033    26.0                   25.967                 3.333
30          21.800    31.0                   30.867                 5.600
```

**mean nearest-training-neighbour Tanimoto of the test ligands**

```
arm         random  maxmin  diversity_x_uncertainty  same_chemotype_first
checkpoint                                                               
0            0.275   0.275                    0.275                 0.275
1            0.322   0.283                    0.308                 0.278
2            0.332   0.324                    0.349                 0.278
3            0.354   0.325                    0.363                 0.278
5            0.378   0.362                    0.396                 0.279
8            0.399   0.384                    0.414                 0.284
12           0.426   0.427                    0.434                 0.294
16           0.437   0.446                    0.451                 0.298
20           0.450   0.454                    0.472                 0.318
25           0.461   0.469                    0.490                 0.338
30           0.475   0.486                    0.503                 0.369
```

**training rows**

```
arm           random    maxmin  diversity_x_uncertainty  same_chemotype_first
checkpoint                                                                   
0           2384.267  2384.267                 2384.267              2384.267
1           2387.267  2387.267                 2387.267              2387.267
2           2390.267  2390.267                 2390.267              2390.267
3           2393.267  2393.267                 2393.267              2393.267
5           2399.267  2399.267                 2399.267              2399.267
8           2408.267  2408.267                 2408.267              2408.267
12          2420.267  2420.267                 2420.267              2420.267
16          2432.267  2432.267                 2432.267              2432.267
20          2444.267  2444.267                 2444.267              2444.267
25          2459.267  2459.267                 2459.267              2459.267
30          2474.267  2474.267                 2474.267              2474.267
```

## Pre-registered hypotheses (protocol §F)

Scoring rule, stated so a verdict cannot be over-read: **PASS** = the pre-registered condition is met **and** the run carried the pre-registered replication (≥ 2 split seeds, ≥ 2 acquisition replicates, ≥ 2 fold-jobs); **FAIL** = the interval excludes the predicted direction at ≥ half the scored checkpoints (evidence *against*, which counts even in a short run); **INCONCLUSIVE** = the intervals straddle zero, the contrast could not be formed, or the condition was met on a run too short to register it. An INCONCLUSIVE is not a weak PASS. Only F1–F4 are protected; every other interval in this report is descriptive.

**F1 — INCONCLUSIVE**  ·  Max-min diversity beats random acquisition on hard chemistry.

* pass condition: hard macro MAE(random) − hard macro MAE(maxmin) with CI95 low > 0 at ≥ half the checkpoints from 5 onwards.
* measured: endpoint hard_nn<0.4 (test rows whose nearest neighbour in the START cohort is below 0.4 Tanimoto — a fixed subset), statistic macro MAE over ECFP clusters; 3/7 checkpoints meet the condition (need 4; BCa agrees at 3); Δ by checkpoint: k=5: +0.153 [+0.030, +0.245], BCa [+0.040, +0.254]; k=8: +0.073 [-0.002, +0.194], BCa [-0.010, +0.180]; k=12: +0.089 [-0.011, +0.236], BCa [-0.039, +0.207]; k=16: +0.080 [-0.028, +0.239], BCa [-0.059, +0.202]; k=20: +0.099 [-0.022, +0.273], BCa [-0.055, +0.235]; k=25: +0.163 [+0.089, +0.281], BCa [+0.083, +0.271]; k=30: +0.110 [+0.042, +0.200], BCa [+0.043, +0.201].
* what would falsify this: If maxmin is indistinguishable from random, the *choice* of ligand does not matter, only the breadth — Experiment B already showed breadth matters, so F1 dying reduces the deployable advice to 'buy any new chemotype'. An interval whose upper end is below zero would be stronger still: random would then be the better rule.

**F2 — INCONCLUSIVE**  ·  Acquiring the most level-uncertain ligand beats random.

* pass condition: offset MAE(random) − offset MAE(offset_uncertainty) with CI95 low > 0 at ≥ half the checkpoints from 5 onwards.
* measured: endpoint all test rows, statistic offset MAE (the per-ligand level error, the quantity Phase 1 says is missing); 0/0 checkpoints meet the condition (need 0; BCa agrees at 0); Δ by checkpoint: no checkpoint produced this contrast. Descriptive, for the protocol's own falsifier (is the model's uncertainty worth anything beyond chemical distance?) — the same statistic against **maxmin** rather than random: no checkpoint produced this contrast
* what would falsify this: If the model's own level uncertainty is no better than chance — or no better than the chemistry-only maxmin rule — then the forest carries no acquisition information beyond chemical distance, and the cheap rule is the one to deploy.

**F3 — PASS**  ·  Buying more of the same chemistry is worse than random.

* pass condition: hard macro MAE(random) − hard macro MAE(same_chemotype_first) with CI95 **high < 0** (random is the better rule) at ≥ half the checkpoints. NOTE: the protocol table writes this as 'hard MAE(same_chemotype_first) − hard MAE(random), CI95 high < 0', which is the opposite of its own gloss 'i.e. random is better'; the gloss is scored here and the arithmetic is printed with every number so the reading cannot be hidden.
* measured: endpoint hard_nn<0.4, statistic macro MAE; delta is random minus same_chemotype_first, so a **negative** interval means random is better; 10/10 checkpoints meet the condition (need 5; BCa agrees at 10); Δ by checkpoint: k=1: -0.060 [-0.122, -0.015], BCa [-0.116, -0.010]; k=2: -0.138 [-0.232, -0.066], BCa [-0.224, -0.058]; k=3: -0.264 [-0.330, -0.173], BCa [-0.335, -0.178]; k=5: -0.389 [-0.478, -0.262], BCa [-0.482, -0.267]; k=8: -0.431 [-0.529, -0.293], BCa [-0.535, -0.304]; k=12: -0.379 [-0.492, -0.280], BCa [-0.492, -0.279]; k=16: -0.467 [-0.573, -0.324], BCa [-0.589, -0.342]; k=20: -0.645 [-0.841, -0.393], BCa [-0.930, -0.447]; k=25: -0.744 [-0.972, -0.440], BCa [-1.054, -0.506]; k=30: -0.588 [-0.734, -0.433], BCa [-0.734, -0.433].
* what would falsify this: If buying another analogue of what you already have matches random acquisition on hard chemistry, then the field's actual behaviour cost nothing measurable at this scale, and the generation's advice weakens to 'do not bother choosing'.

**F4 — SKIPPED**  ·  Three measurements of a distant ligand beat three more of a known one.

* pass condition: the b = 3 maxmin curve lies below the curve that spends the same rows on the start cohort's own ligands, at ≥ half the checkpoints.
* measured: needs --f4-cap-start-rows; with the full start cohort there are no unrevealed start rows to buy, and Experiment B already measured depth vs breadth
* what would falsify this: Not evaluated in this run. With --f4-cap-start-rows N it is evaluated against the depth_on_start arm; if depth matched breadth there, Experiment B's headline would be contradicted at the level of individual measurements.

## Every contrast, every checkpoint

Reference is `random` (F4's is the depth arm). Positive delta = the candidate policy is better. `n_folds` is the number of (seed, fold) cells behind the row; `units_total` the number of ECFP clusters; `bootstrap_blocks` the number of chemotypes resampled. Only the protected contrast set is printed here; the descriptive sets (`vs_maxmin|…`, `budget_all|…`) and the `shape_mae` statistic are in `contrasts.csv`.

```
   endpoint  checkpoint                        comparison  statistic  point_delta  ci95_low  ci95_high  bca_low  bca_high  cluster_robust_low  block_macro_delta  units_improved  units_total  bootstrap_blocks  n_folds
        all           1                  maxmin_vs_random        mae       0.0495    0.0007     0.0936   0.0091    0.1089             -0.0037             0.0365              79          131                79       15
        all           1                  maxmin_vs_random offset_mae       0.0367   -0.0205     0.0902  -0.0114    0.1098             -0.0286             0.0078              78          131                79       15
        all           1 diversity_x_uncertainty_vs_random        mae       0.0568   -0.0142     0.1237  -0.0037    0.1462             -0.0240             0.0249              78          131                79       15
        all           1 diversity_x_uncertainty_vs_random offset_mae       0.0428   -0.0444     0.1250  -0.0299    0.1586             -0.0576            -0.0086              72          131                79       15
        all           1    same_chemotype_first_vs_random        mae      -0.0467   -0.0839    -0.0168  -0.0793   -0.0129             -0.0822            -0.0409              48          131                79       15
        all           1    same_chemotype_first_vs_random offset_mae      -0.0279   -0.0710     0.0054  -0.0670    0.0082             -0.0672            -0.0310              57          131                79       15
hard_nn<0.4           1                  maxmin_vs_random        mae       0.0528   -0.0086     0.1017   0.0018    0.1141             -0.0075             0.0447              74          119                69       15
hard_nn<0.4           1                  maxmin_vs_random offset_mae       0.0232   -0.0555     0.0878  -0.0420    0.1091             -0.0583            -0.0044              65          119                69       15
hard_nn<0.4           1 diversity_x_uncertainty_vs_random        mae       0.0602   -0.0275     0.1318  -0.0102    0.1584             -0.0302             0.0329              68          119                69       15
hard_nn<0.4           1 diversity_x_uncertainty_vs_random offset_mae       0.0460   -0.0640     0.1356  -0.0440    0.1672             -0.0681            -0.0012              64          119                69       15
hard_nn<0.4           1    same_chemotype_first_vs_random        mae      -0.0601   -0.1222    -0.0149  -0.1158   -0.0100             -0.1138            -0.0433              47          119                69       15
hard_nn<0.4           1    same_chemotype_first_vs_random offset_mae      -0.0864   -0.1703    -0.0250  -0.1590   -0.0160             -0.1613            -0.0786              45          119                69       15
hard_nn<0.6           1                  maxmin_vs_random        mae       0.0472   -0.0083     0.0935   0.0015    0.1090             -0.0109             0.0377              76          127                76       15
hard_nn<0.6           1                  maxmin_vs_random offset_mae       0.0231   -0.0479     0.0832  -0.0356    0.1042             -0.0527            -0.0022              74          127                76       15
hard_nn<0.6           1 diversity_x_uncertainty_vs_random        mae       0.0584   -0.0194     0.1258  -0.0054    0.1528             -0.0263             0.0313              75          127                76       15
hard_nn<0.6           1 diversity_x_uncertainty_vs_random offset_mae       0.0408   -0.0560     0.1254  -0.0385    0.1588             -0.0661            -0.0036              72          127                76       15
hard_nn<0.6           1    same_chemotype_first_vs_random        mae      -0.0500   -0.0986    -0.0122  -0.0950   -0.0098             -0.0942            -0.0367              49          127                76       15
hard_nn<0.6           1    same_chemotype_first_vs_random offset_mae      -0.0511   -0.1177    -0.0026  -0.1146   -0.0004             -0.1089            -0.0436              55          127                76       15
        all           2                  maxmin_vs_random        mae       0.1135    0.0369     0.1748   0.0490    0.1882              0.0396             0.0984              88          131                79       15
        all           2                  maxmin_vs_random offset_mae       0.0945   -0.0001     0.1714   0.0147    0.1877              0.0018             0.0644              79          131                79       15
        all           2 diversity_x_uncertainty_vs_random        mae       0.1504   -0.0243     0.3272  -0.0021    0.4041             -0.0643             0.0491              74          131                79       15
        all           2 diversity_x_uncertainty_vs_random offset_mae       0.1552   -0.0688     0.3808  -0.0399    0.4860             -0.1230             0.0021              75          131                79       15
        all           2    same_chemotype_first_vs_random        mae      -0.1167   -0.1765    -0.0704  -0.1706   -0.0654             -0.1728            -0.1076              31          131                79       15
        all           2    same_chemotype_first_vs_random offset_mae      -0.0937   -0.1558    -0.0417  -0.1589   -0.0426             -0.1511            -0.0910              42          131                79       15
hard_nn<0.4           2                  maxmin_vs_random        mae       0.1447    0.0613     0.2041   0.0707    0.2116              0.0721             0.1298              87          119                69       15
hard_nn<0.4           2                  maxmin_vs_random offset_mae       0.1348    0.0340     0.2075   0.0490    0.2201              0.0456             0.1005              78          119                69       15
hard_nn<0.4           2 diversity_x_uncertainty_vs_random        mae       0.1685   -0.0304     0.3486  -0.0045    0.4094             -0.0628             0.0709              67          119                69       15
hard_nn<0.4           2 diversity_x_uncertainty_vs_random offset_mae       0.1746   -0.0825     0.4103  -0.0500    0.4959             -0.1284             0.0354              63          119                69       15
hard_nn<0.4           2    same_chemotype_first_vs_random        mae      -0.1377   -0.2316    -0.0660  -0.2237   -0.0584             -0.2198            -0.1100              31          119                69       15
hard_nn<0.4           2    same_chemotype_first_vs_random offset_mae      -0.1634   -0.2729    -0.0777  -0.2650   -0.0700             -0.2608            -0.1331              31          119                69       15
hard_nn<0.6           2                  maxmin_vs_random        mae       0.1201    0.0404     0.1817   0.0542    0.1905              0.0457             0.1055              86          127                76       15
hard_nn<0.6           2                  maxmin_vs_random offset_mae       0.1019    0.0033     0.1781   0.0202    0.1942              0.0084             0.0725              78          127                76       15
hard_nn<0.6           2 diversity_x_uncertainty_vs_random        mae       0.1583   -0.0308     0.3360  -0.0028    0.4041             -0.0619             0.0619              71          127                76       15
hard_nn<0.6           2 diversity_x_uncertainty_vs_random offset_mae       0.1704   -0.0698     0.3983  -0.0365    0.4873             -0.1140             0.0328              71          127                76       15
hard_nn<0.6           2    same_chemotype_first_vs_random        mae      -0.1199   -0.1957    -0.0612  -0.1902   -0.0570             -0.1873            -0.0966              32          127                76       15
hard_nn<0.6           2    same_chemotype_first_vs_random offset_mae      -0.1122   -0.1949    -0.0427  -0.1939   -0.0417             -0.1863            -0.0887              39          127                76       15
        all           3                  maxmin_vs_random        mae      -0.0407   -0.1165     0.0605  -0.1421    0.0374             -0.1365             0.0285              48          131                79       15
        all           3                  maxmin_vs_random offset_mae      -0.0319   -0.1324     0.0996  -0.1690    0.0703             -0.1588             0.0595              55          131                79       15
        all           3 diversity_x_uncertainty_vs_random        mae       0.1370    0.0038     0.2572   0.0269    0.3072             -0.0099             0.0784              85          131                79       15
        all           3 diversity_x_uncertainty_vs_random offset_mae       0.1587   -0.0189     0.3293   0.0070    0.4042             -0.0486             0.0615              79          131                79       15
        all           3    same_chemotype_first_vs_random        mae      -0.2435   -0.2985    -0.1678  -0.3044   -0.1798             -0.3073            -0.1958              20          131                79       15
        all           3    same_chemotype_first_vs_random offset_mae      -0.2233   -0.2999    -0.1210  -0.3150   -0.1395             -0.3156            -0.1651              28          131                79       15
hard_nn<0.4           3                  maxmin_vs_random        mae      -0.0287   -0.1151     0.0970  -0.1440    0.0682             -0.1417             0.0457              49          119                69       15
hard_nn<0.4           3                  maxmin_vs_random offset_mae      -0.0156   -0.1286     0.1438  -0.1692    0.1079             -0.1645             0.0850              53          119                69       15
hard_nn<0.4           3 diversity_x_uncertainty_vs_random        mae       0.1669    0.0189     0.2836   0.0477    0.3222              0.0154             0.1086              80          119                69       15
hard_nn<0.4           3 diversity_x_uncertainty_vs_random offset_mae       0.2188    0.0274     0.3762   0.0586    0.4299              0.0132             0.1289              80          119                69       15
hard_nn<0.4           3    same_chemotype_first_vs_random        mae      -0.2643   -0.3298    -0.1725  -0.3352   -0.1785             -0.3381            -0.2018              21          119                69       15
hard_nn<0.4           3    same_chemotype_first_vs_random offset_mae      -0.2690   -0.3476    -0.1484  -0.3550   -0.1602             -0.3647            -0.1738              23          119                69       15
hard_nn<0.6           3                  maxmin_vs_random        mae      -0.0477   -0.1231     0.0591  -0.1440    0.0377             -0.1443             0.0200              46          127                76       15
hard_nn<0.6           3                  maxmin_vs_random offset_mae      -0.0426   -0.1408     0.0936  -0.1737    0.0629             -0.1698             0.0480              50          127                76       15
hard_nn<0.6           3 diversity_x_uncertainty_vs_random        mae       0.1498    0.0147     0.2665   0.0369    0.3077              0.0026             0.0919              82          127                76       15
hard_nn<0.6           3 diversity_x_uncertainty_vs_random offset_mae       0.1910    0.0099     0.3537   0.0401    0.4193             -0.0116             0.1044              79          127                76       15
hard_nn<0.6           3    same_chemotype_first_vs_random        mae      -0.2468   -0.3032    -0.1675  -0.3087   -0.1753             -0.3150            -0.1884              21          127                76       15
hard_nn<0.6           3    same_chemotype_first_vs_random offset_mae      -0.2312   -0.3077    -0.1205  -0.3222   -0.1420             -0.3256            -0.1507              28          127                76       15
        all           5                  maxmin_vs_random        mae       0.1093   -0.0068     0.2003   0.0090    0.2151             -0.0001             0.0979              82          131                79       15
        all           5                  maxmin_vs_random offset_mae       0.1146   -0.0310     0.2295  -0.0053    0.2587             -0.0248             0.0826              71          131                79       15
        all           5 diversity_x_uncertainty_vs_random        mae       0.1217    0.0065     0.2150   0.0267    0.2350              0.0122             0.1077              76          131                79       15
        all           5 diversity_x_uncertainty_vs_random offset_mae       0.1277   -0.0009     0.2309   0.0195    0.2565              0.0045             0.0918              79          131                79       15
        all           5    same_chemotype_first_vs_random        mae      -0.3510   -0.4214    -0.2516  -0.4346   -0.2681             -0.4366            -0.2951              22          131                79       15
        all           5    same_chemotype_first_vs_random offset_mae      -0.3537   -0.4852    -0.1884  -0.5288   -0.2203             -0.5149            -0.2660              30          131                79       15
hard_nn<0.4           5                  maxmin_vs_random        mae       0.1528    0.0296     0.2449   0.0397    0.2541              0.0445             0.1515              80          119                69       15
hard_nn<0.4           5                  maxmin_vs_random offset_mae       0.1869    0.0428     0.2956   0.0620    0.3120              0.0562             0.1777              71          119                69       15
hard_nn<0.4           5 diversity_x_uncertainty_vs_random        mae       0.1148   -0.0274     0.2202  -0.0048    0.2445             -0.0165             0.1129              72          119                69       15
hard_nn<0.4           5 diversity_x_uncertainty_vs_random offset_mae       0.1156   -0.0529     0.2430  -0.0310    0.2630             -0.0414             0.1095              71          119                69       15
hard_nn<0.4           5    same_chemotype_first_vs_random        mae      -0.3890   -0.4783    -0.2619  -0.4824   -0.2666             -0.4910            -0.3108              21          119                69       15
hard_nn<0.4           5    same_chemotype_first_vs_random offset_mae      -0.4357   -0.5638    -0.2374  -0.5870   -0.2741             -0.5987            -0.2930              26          119                69       15
hard_nn<0.6           5                  maxmin_vs_random        mae       0.1224    0.0057     0.2147   0.0210    0.2286              0.0115             0.1188              81          127                76       15
hard_nn<0.6           5                  maxmin_vs_random offset_mae       0.1454    0.0009     0.2594   0.0191    0.2761              0.0068             0.1316              71          127                76       15
hard_nn<0.6           5 diversity_x_uncertainty_vs_random        mae       0.1165   -0.0115     0.2183   0.0101    0.2379             -0.0051             0.1127              75          127                76       15
hard_nn<0.6           5 diversity_x_uncertainty_vs_random offset_mae       0.1222   -0.0322     0.2387  -0.0047    0.2657             -0.0205             0.1106              75          127                76       15
hard_nn<0.6           5    same_chemotype_first_vs_random        mae      -0.3657   -0.4437    -0.2588  -0.4503   -0.2683             -0.4572            -0.2933              23          127                76       15
hard_nn<0.6           5    same_chemotype_first_vs_random offset_mae      -0.3835   -0.5135    -0.2086  -0.5474   -0.2437             -0.5454            -0.2648              31          127                76       15
        all           8                  maxmin_vs_random        mae       0.0451   -0.0209     0.1462  -0.0235    0.1399             -0.0370             0.1107              62          131                79       15
        all           8                  maxmin_vs_random offset_mae       0.0753   -0.0100     0.1663  -0.0093    0.1676             -0.0100             0.1070              67          131                79       15
        all           8 diversity_x_uncertainty_vs_random        mae       0.1083    0.0327     0.2196   0.0304    0.2141              0.0196             0.1722              76          131                79       15
        all           8 diversity_x_uncertainty_vs_random offset_mae       0.1272    0.0432     0.2499   0.0368    0.2354              0.0279             0.1951              78          131                79       15
        all           8    same_chemotype_first_vs_random        mae      -0.3975   -0.4771    -0.2865  -0.4904   -0.3030             -0.4946            -0.3394              21          131                79       15
        all           8    same_chemotype_first_vs_random offset_mae      -0.3959   -0.5263    -0.2297  -0.5651   -0.2646             -0.5542            -0.3161              26          131                79       15
hard_nn<0.4           8                  maxmin_vs_random        mae       0.0729   -0.0021     0.1937  -0.0100    0.1797             -0.0246             0.1467              57          119                69       15
hard_nn<0.4           8                  maxmin_vs_random offset_mae       0.1149    0.0175     0.2251   0.0204    0.2270              0.0190             0.1578              65          119                69       15
hard_nn<0.4           8 diversity_x_uncertainty_vs_random        mae       0.1074    0.0232     0.2337   0.0218    0.2298              0.0068             0.1839              67          119                69       15
hard_nn<0.4           8 diversity_x_uncertainty_vs_random offset_mae       0.1260    0.0304     0.2643   0.0271    0.2573              0.0115             0.2105              71          119                69       15
hard_nn<0.4           8    same_chemotype_first_vs_random        mae      -0.4308   -0.5289    -0.2933  -0.5347   -0.3039             -0.5448            -0.3530              21          119                69       15
hard_nn<0.4           8    same_chemotype_first_vs_random offset_mae      -0.4674   -0.5967    -0.2695  -0.6176   -0.3013             -0.6334            -0.3422              24          119                69       15
hard_nn<0.6           8                  maxmin_vs_random        mae       0.0480   -0.0225     0.1558  -0.0275    0.1476             -0.0388             0.1138              59          127                76       15
hard_nn<0.6           8                  maxmin_vs_random offset_mae       0.0864   -0.0089     0.1875  -0.0097    0.1855             -0.0074             0.1225              66          127                76       15
hard_nn<0.6           8 diversity_x_uncertainty_vs_random        mae       0.0987    0.0205     0.2125   0.0204    0.2119              0.0062             0.1672              70          127                76       15
hard_nn<0.6           8 diversity_x_uncertainty_vs_random offset_mae       0.1145    0.0272     0.2425   0.0239    0.2384              0.0096             0.1883              73          127                76       15
hard_nn<0.6           8    same_chemotype_first_vs_random        mae      -0.4171   -0.4983    -0.3022  -0.5085   -0.3187             -0.5143            -0.3487              23          127                76       15
hard_nn<0.6           8    same_chemotype_first_vs_random offset_mae      -0.4271   -0.5531    -0.2505  -0.5868   -0.2892             -0.5854            -0.3289              28          127                76       15
        all          12                  maxmin_vs_random        mae       0.0427   -0.0397     0.1632  -0.0586    0.1374             -0.0629             0.1226              63          131                79       15
        all          12                  maxmin_vs_random offset_mae       0.0453   -0.0341     0.1589  -0.0408    0.1480             -0.0510             0.1097              67          131                79       15
        all          12 diversity_x_uncertainty_vs_random        mae       0.1077    0.0216     0.2332   0.0076    0.2110              0.0002             0.2016              68          131                79       15
        all          12 diversity_x_uncertainty_vs_random offset_mae       0.1197    0.0205     0.2607   0.0043    0.2350             -0.0022             0.2200              68          131                79       15
        all          12    same_chemotype_first_vs_random        mae      -0.3868   -0.4815    -0.3040  -0.4784   -0.3009             -0.4707            -0.4021              23          131                79       15
        all          12    same_chemotype_first_vs_random offset_mae      -0.4090   -0.5055    -0.2944  -0.5030   -0.2909             -0.5091            -0.4138              23          131                79       15
hard_nn<0.4          12                  maxmin_vs_random        mae       0.0890   -0.0106     0.2358  -0.0386    0.2066             -0.0434             0.1864              63          119                69       15
hard_nn<0.4          12                  maxmin_vs_random offset_mae       0.1153    0.0194     0.2659   0.0024    0.2389             -0.0099             0.2081              69          119                69       15
hard_nn<0.4          12 diversity_x_uncertainty_vs_random        mae       0.1072    0.0133     0.2519  -0.0017    0.2328             -0.0150             0.2182              65          119                69       15
hard_nn<0.4          12 diversity_x_uncertainty_vs_random offset_mae       0.1126    0.0078     0.2772  -0.0137    0.2508             -0.0255             0.2351              65          119                69       15
hard_nn<0.4          12    same_chemotype_first_vs_random        mae      -0.3787   -0.4917    -0.2799  -0.4915   -0.2789             -0.4771            -0.3860              23          119                69       15
hard_nn<0.4          12    same_chemotype_first_vs_random offset_mae      -0.4309   -0.5502    -0.3016  -0.5463   -0.2962             -0.5470            -0.4108              23          119                69       15
hard_nn<0.6          12                  maxmin_vs_random        mae       0.0509   -0.0372     0.1809  -0.0549    0.1550             -0.0611             0.1341              62          127                76       15
hard_nn<0.6          12                  maxmin_vs_random offset_mae       0.0661   -0.0191     0.1948  -0.0291    0.1835             -0.0415             0.1409              67          127                76       15
hard_nn<0.6          12 diversity_x_uncertainty_vs_random        mae       0.0981    0.0103     0.2322  -0.0042    0.2123             -0.0130             0.1986              66          127                76       15
hard_nn<0.6          12 diversity_x_uncertainty_vs_random offset_mae       0.1092    0.0077     0.2614  -0.0085    0.2360             -0.0171             0.2189              65          127                76       15
hard_nn<0.6          12    same_chemotype_first_vs_random        mae      -0.3877   -0.4863    -0.3070  -0.4835   -0.3053             -0.4736            -0.3983              24          127                76       15
hard_nn<0.6          12    same_chemotype_first_vs_random offset_mae      -0.4179   -0.5207    -0.3007  -0.5191   -0.2970             -0.5212            -0.4115              25          127                76       15
        all          16                  maxmin_vs_random        mae       0.0415   -0.0528     0.1720  -0.0770    0.1436             -0.0753             0.1323              65          131                79       15
        all          16                  maxmin_vs_random offset_mae       0.0353   -0.0597     0.1711  -0.0837    0.1440             -0.0838             0.1263              64          131                79       15
        all          16 diversity_x_uncertainty_vs_random        mae       0.1195   -0.0070     0.2972  -0.0412    0.2534             -0.0387             0.2542              61          131                79       15
        all          16 diversity_x_uncertainty_vs_random offset_mae       0.1575    0.0242     0.3467  -0.0130    0.3005             -0.0092             0.3036              68          131                79       15
        all          16    same_chemotype_first_vs_random        mae      -0.4661   -0.5649    -0.3366  -0.5738   -0.3494             -0.5808            -0.4462              19          131                79       15
        all          16    same_chemotype_first_vs_random offset_mae      -0.5088   -0.6403    -0.3440  -0.6632   -0.3692             -0.6644            -0.4736              22          131                79       15
hard_nn<0.4          16                  maxmin_vs_random        mae       0.0803   -0.0285     0.2387  -0.0586    0.2024             -0.0625             0.1961              68          119                69       15
hard_nn<0.4          16                  maxmin_vs_random offset_mae       0.0981   -0.0204     0.2692  -0.0514    0.2301             -0.0555             0.2202              71          119                69       15
hard_nn<0.4          16 diversity_x_uncertainty_vs_random        mae       0.1081   -0.0224     0.2998  -0.0551    0.2625             -0.0635             0.2643              57          119                69       15
hard_nn<0.4          16 diversity_x_uncertainty_vs_random offset_mae       0.1401    0.0022     0.3465  -0.0332    0.3039             -0.0398             0.3087              63          119                69       15
hard_nn<0.4          16    same_chemotype_first_vs_random        mae      -0.4673   -0.5727    -0.3245  -0.5892   -0.3415             -0.5945            -0.4558              21          119                69       15
hard_nn<0.4          16    same_chemotype_first_vs_random offset_mae      -0.5200   -0.6535    -0.3352  -0.6856   -0.3656             -0.6898            -0.4873              26          119                69       15
hard_nn<0.6          16                  maxmin_vs_random        mae       0.0465   -0.0479     0.1855  -0.0807    0.1535             -0.0760             0.1441              65          127                76       15
hard_nn<0.6          16                  maxmin_vs_random offset_mae       0.0526   -0.0493     0.2020  -0.0751    0.1709             -0.0773             0.1548              67          127                76       15
hard_nn<0.6          16 diversity_x_uncertainty_vs_random        mae       0.1111   -0.0135     0.2943  -0.0540    0.2548             -0.0508             0.2529              59          127                76       15
hard_nn<0.6          16 diversity_x_uncertainty_vs_random offset_mae       0.1485    0.0173     0.3409  -0.0222    0.3013             -0.0221             0.3014              66          127                76       15
hard_nn<0.6          16    same_chemotype_first_vs_random        mae      -0.4697   -0.5722    -0.3355  -0.5816   -0.3465             -0.5881            -0.4541              21          127                76       15
hard_nn<0.6          16    same_chemotype_first_vs_random offset_mae      -0.5168   -0.6535    -0.3404  -0.6784   -0.3662             -0.6775            -0.4841              24          127                76       15
        all          20                  maxmin_vs_random        mae       0.0598   -0.0414     0.2041  -0.0707    0.1695             -0.0678             0.1630              70          131                79       15
        all          20                  maxmin_vs_random offset_mae       0.0923   -0.0018     0.2354  -0.0202    0.2086             -0.0256             0.1976              71          131                79       15
        all          20 diversity_x_uncertainty_vs_random        mae       0.1002   -0.0308     0.2854  -0.0719    0.2386             -0.0658             0.2504              59          131                79       15
        all          20 diversity_x_uncertainty_vs_random offset_mae       0.1511    0.0118     0.3509  -0.0330    0.3031             -0.0264             0.3232              71          131                79       15
        all          20    same_chemotype_first_vs_random        mae      -0.6076   -0.8095    -0.3798  -0.8878   -0.4218             -0.8503            -0.5000              14          131                79       15
        all          20    same_chemotype_first_vs_random offset_mae      -0.6361   -0.9022    -0.3497  -1.0159   -0.3934             -0.9572            -0.4883              22          131                79       15
hard_nn<0.4          20                  maxmin_vs_random        mae       0.0991   -0.0220     0.2728  -0.0551    0.2345             -0.0578             0.2303              68          119                69       15
hard_nn<0.4          20                  maxmin_vs_random offset_mae       0.1496    0.0324     0.3284   0.0085    0.2935             -0.0024             0.2824              70          119                69       15
hard_nn<0.4          20 diversity_x_uncertainty_vs_random        mae       0.0696   -0.0682     0.2700  -0.0960    0.2298             -0.1056             0.2386              54          119                69       15
hard_nn<0.4          20 diversity_x_uncertainty_vs_random offset_mae       0.0973   -0.0414     0.3071  -0.0780    0.2663             -0.0836             0.2850              60          119                69       15
hard_nn<0.4          20    same_chemotype_first_vs_random        mae      -0.6449   -0.8410    -0.3932  -0.9296   -0.4472             -0.8997            -0.5348              15          119                69       15
hard_nn<0.4          20    same_chemotype_first_vs_random offset_mae      -0.7098   -0.9611    -0.3974  -1.0712   -0.4601             -1.0347            -0.5552              23          119                69       15
hard_nn<0.6          20                  maxmin_vs_random        mae       0.0704   -0.0354     0.2199  -0.0655    0.1913             -0.0656             0.1734              69          127                76       15
hard_nn<0.6          20                  maxmin_vs_random offset_mae       0.1122    0.0104     0.2651  -0.0084    0.2423             -0.0182             0.2111              71          127                76       15
hard_nn<0.6          20 diversity_x_uncertainty_vs_random        mae       0.0899   -0.0411     0.2830  -0.0851    0.2403             -0.0797             0.2471              59          127                76       15
hard_nn<0.6          20 diversity_x_uncertainty_vs_random offset_mae       0.1320   -0.0083     0.3363  -0.0455    0.2944             -0.0469             0.3063              67          127                76       15
hard_nn<0.6          20    same_chemotype_first_vs_random        mae      -0.6327   -0.8298    -0.4005  -0.8999   -0.4481             -0.8739            -0.5298              14          127                76       15
hard_nn<0.6          20    same_chemotype_first_vs_random offset_mae      -0.6829   -0.9413    -0.3868  -1.0334   -0.4458             -0.9971            -0.5449              22          127                76       15
        all          25                  maxmin_vs_random        mae       0.1049    0.0414     0.1934   0.0439    0.1986              0.0318             0.1499              72          131                79       15
        all          25                  maxmin_vs_random offset_mae       0.1083    0.0291     0.1973   0.0315    0.2008              0.0284             0.1536              76          131                79       15
        all          25 diversity_x_uncertainty_vs_random        mae       0.1396    0.0448     0.2829   0.0249    0.2501              0.0194             0.2410              72          131                79       15
        all          25 diversity_x_uncertainty_vs_random offset_mae       0.1631    0.0668     0.3109   0.0517    0.2830              0.0437             0.2793              75          131                79       15
        all          25    same_chemotype_first_vs_random        mae      -0.6822   -0.9191    -0.4122  -1.0054   -0.4637             -0.9675            -0.5634              17          131                79       15
        all          25    same_chemotype_first_vs_random offset_mae      -0.7021   -1.0123    -0.3648  -1.1489   -0.4177             -1.0786            -0.5402              24          131                79       15
hard_nn<0.4          25                  maxmin_vs_random        mae       0.1634    0.0887     0.2810   0.0826    0.2714              0.0690             0.2225              72          119                69       15
hard_nn<0.4          25                  maxmin_vs_random offset_mae       0.1836    0.0948     0.3132   0.0955    0.3137              0.0793             0.2362              76          119                69       15
hard_nn<0.4          25 diversity_x_uncertainty_vs_random        mae       0.1450    0.0423     0.3004   0.0191    0.2698              0.0112             0.2518              66          119                69       15
hard_nn<0.4          25 diversity_x_uncertainty_vs_random offset_mae       0.1597    0.0578     0.3165   0.0421    0.2941              0.0288             0.2727              67          119                69       15
hard_nn<0.4          25    same_chemotype_first_vs_random        mae      -0.7435   -0.9719    -0.4396  -1.0541   -0.5062             -1.0391            -0.6160              17          119                69       15
hard_nn<0.4          25    same_chemotype_first_vs_random offset_mae      -0.8412   -1.1205    -0.4845  -1.2135   -0.5571             -1.2029            -0.6728              19          119                69       15
hard_nn<0.6          25                  maxmin_vs_random        mae       0.1349    0.0651     0.2342   0.0649    0.2341              0.0508             0.1758              72          127                76       15
hard_nn<0.6          25                  maxmin_vs_random offset_mae       0.1509    0.0607     0.2665   0.0655    0.2720              0.0540             0.1843              75          127                76       15
hard_nn<0.6          25 diversity_x_uncertainty_vs_random        mae       0.1508    0.0510     0.3000   0.0259    0.2692              0.0233             0.2532              71          127                76       15
hard_nn<0.6          25 diversity_x_uncertainty_vs_random offset_mae       0.1742    0.0745     0.3267   0.0557    0.3028              0.0471             0.2846              74          127                76       15
hard_nn<0.6          25    same_chemotype_first_vs_random        mae      -0.7120   -0.9420    -0.4419  -1.0283   -0.4950             -0.9949            -0.5941              17          127                76       15
hard_nn<0.6          25    same_chemotype_first_vs_random offset_mae      -0.7592   -1.0595    -0.4195  -1.1640   -0.4818             -1.1257            -0.6082              22          127                76       15
        all          30                  maxmin_vs_random        mae       0.0660    0.0041     0.1355   0.0061    0.1391              0.0034             0.1066              66          131                79       15
        all          30                  maxmin_vs_random offset_mae       0.0705    0.0027     0.1522   0.0026    0.1521              0.0001             0.1238              67          131                79       15
        all          30 diversity_x_uncertainty_vs_random        mae       0.1614    0.0970     0.2571   0.0979    0.2575              0.0865             0.2238              83          131                79       15
        all          30 diversity_x_uncertainty_vs_random offset_mae       0.1780    0.1055     0.2846   0.1060    0.2868              0.0941             0.2571              81          131                79       15
        all          30    same_chemotype_first_vs_random        mae      -0.5568   -0.6802    -0.4181  -0.6784   -0.4168             -0.6788            -0.5679              22          131                79       15
        all          30    same_chemotype_first_vs_random offset_mae      -0.5775   -0.7231    -0.3969  -0.7310   -0.4067             -0.7390            -0.5712              22          131                79       15
hard_nn<0.4          30                  maxmin_vs_random        mae       0.1096    0.0422     0.1996   0.0428    0.2012              0.0352             0.1726              67          119                69       15
hard_nn<0.4          30                  maxmin_vs_random offset_mae       0.1109    0.0374     0.2117   0.0374    0.2116              0.0283             0.1836              64          119                69       15
hard_nn<0.4          30 diversity_x_uncertainty_vs_random        mae       0.1649    0.0911     0.2679   0.0902    0.2670              0.0798             0.2382              76          119                69       15
hard_nn<0.4          30 diversity_x_uncertainty_vs_random offset_mae       0.1758    0.0892     0.2894   0.0863    0.2855              0.0795             0.2630              73          119                69       15
hard_nn<0.4          30    same_chemotype_first_vs_random        mae      -0.5881   -0.7343    -0.4325  -0.7344   -0.4326             -0.7270            -0.5963              21          119                69       15
hard_nn<0.4          30    same_chemotype_first_vs_random offset_mae      -0.6542   -0.8094    -0.4592  -0.8098   -0.4600             -0.8233            -0.6287              24          119                69       15
hard_nn<0.6          30                  maxmin_vs_random        mae       0.0789    0.0159     0.1549   0.0188    0.1581              0.0150             0.1236              67          127                76       15
hard_nn<0.6          30                  maxmin_vs_random offset_mae       0.0819    0.0140     0.1666   0.0152    0.1689              0.0091             0.1357              66          127                76       15
hard_nn<0.6          30 diversity_x_uncertainty_vs_random        mae       0.1672    0.1003     0.2657   0.1003    0.2658              0.0885             0.2306              81          127                76       15
hard_nn<0.6          30 diversity_x_uncertainty_vs_random offset_mae       0.1849    0.1084     0.2980   0.1065    0.2952              0.0963             0.2624              80          127                76       15
hard_nn<0.6          30    same_chemotype_first_vs_random        mae      -0.5770   -0.7067    -0.4447  -0.7023   -0.4387             -0.7000            -0.5877              21          127                76       15
hard_nn<0.6          30    same_chemotype_first_vs_random offset_mae      -0.6154   -0.7659    -0.4369  -0.7664   -0.4370             -0.7739            -0.6127              22          127                76       15
```

The descriptive contrast against the chemistry-only rule (`vs_maxmin|…`) answers the protocol's own falsifier for F2 — whether the model's uncertainty carries anything beyond chemical distance — and is quoted inside F2's evidence above.

## How to break this result

* **One fold has almost no start cohort.** The fold that holds out the diglycolamides starts from a handful of ligands, so every policy improves enormously there and the between-policy contrast is dominated by folds where the start is genuinely narrow and deep. Re-run with `--fold-indices` excluding it and see whether the verdicts survive; the per-fold curves are in `acquisition_curves.csv`.
* **The hard subset is defined by the start cohort, so it differs per fold.** It is fixed *within* a fold (that is what makes the policies comparable), but a fold whose start chemotype is unusual has an unusual hard subset. The row and ligand counts per fold are in the start-cohort table above and in `validation.json`.
* **Replicates are not seeds and seeds are not experiments.** The split seeds re-partition the same ligands; the acquisition replicates re-draw the same policy. The load-bearing statistic is the bootstrap over chemotype blocks.
* **Every policy still trains on the same learner and the same features.** A policy that helps only this feature set would be an artefact of the representation.
* **The pool is not the world.** Acquisition can only buy ligands that exist in this dataset; a maxmin rule that is starved of genuinely distant candidates will look like random by construction. `nn_to_acquired_before` in `acquisition_steps.csv` shows how distant the purchases actually were.
* **F4** is SKIPPED in this run: needs --f4-cap-start-rows; with the full start cohort there are no unrevealed start rows to buy, and Experiment B already measured depth vs breadth.
* **Do not quote the percentile interval alone at the `all` endpoint.** One chemotype holds a fifth of the scoring units; BCa, cluster-robust and block-macro columns sit beside it in `contrasts.csv` for that reason.

## Run-time checks

* `hard_masks_fixed_per_fold`: **True** — the start-cohort hard-chemistry masks are hashed on every checkpoint call; one distinct digest per fold and threshold means the subset never moved with the policy or the step
* `no_test_row_revealed`: **True** — the metric closure intersects the revealed training rows with the fold's test rows at every checkpoint and raises on any overlap
* `checkpoint_zero_identical_across_arms`: **True** — before the first acquisition every arm is the same forest on the same start rows; a non-zero spread would mean the start cohort or the fold seed moved with the policy
* `all_arms_reached_all_checkpoints`: **True** — an arm that exhausts the pool stops early; such (seed, fold) cells are dropped whole from the paired bootstrap rather than partially filled
* `curve_shape`: **True** — one curve per (split seed, fold, arm, replicate)
* `split_integrity`: **True**
* `label_free_acquisition`: **True** — the pool's log_D is permuted among pool rows; the start rows (and hence the model that chooses step 1) are untouched, so any movement is a label leak

