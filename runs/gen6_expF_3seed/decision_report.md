# gen6 Experiment F — which ligand should have been measured next?

**Run** `gen6_acquisition_sim_20260819T134329Z`

## What was actually done

* One shared cohort at `min_cells = 3`: **5248 rows, 152 extractants, 131 ECFP clusters, 79 Tanimoto-0.7 chemotypes**; target sd 1.657 log units.
* Folds hold out whole chemotypes (Experiment A's `diversity_splits`, 5 folds, seeds [104729, 130363, 155921]). **The fold's held-out chemotypes are the test set and are fixed**; acquisition draws only from the EXPANDED training index.
* Start cohort: the 10 most-measured training ligands of the largest training chemotype, all their rows.
* Acquisition: up to 30 steps, 3 rows revealed per acquisition, checkpoints [1, 2, 3, 5, 8, 12, 16, 20, 25, 30], 2 replicate(s) per (seed, fold, policy).
* Learner: ExtraTrees, 200 trees, max_features 0.3, min_samples_leaf 2; feature columns from blocks ['METAL', 'COND', 'LIG2D_EXT', 'MASSACTION']; fold seed `model_seed + fold*1009 + 9999991 + 101*replicate`.
* Frozen chemistry map over 190 extractants (164 ECFP clusters, 98 chemotypes at threshold 0.7).
* Arms: primary ['random', 'maxmin', 'uncertainty', 'diversity_x_uncertainty', 'offset_uncertainty', 'same_chemotype_first']; budget=all sensitivity ['maxmin@ball', 'random@ball'].
* Simulation took 1729 s. First fold took 126.1 s; 15 fold-jobs → estimated 1891 s (31.5 min) of simulation (plus bootstrap and reporting).

## How to read this

* **macro MAE** = one ECFP cluster, one vote (the study's primary metric). Pooled MAE is in the CSVs and never used for selection: the largest single ligand holds 28 % of rows.
* **hard chemistry is a fixed subset.** `hard_nn<0.4` means: test rows whose maximum Tanimoto to the **start cohort's** extractants is below the threshold. It is computed once per fold, before any acquisition, and never recomputed — so the same rows are scored for every policy at every step. Recomputing it against the current training set would shrink the subset exactly for the policies that buy distant chemistry, which would manufacture the result this experiment is trying to measure.
* **checkpoint 0 is the start cohort alone**, identical for every arm by construction (measured max spread across arms: 2.22e-15). It is excluded from the hypothesis scoring, where every delta there is structurally zero.
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
arm         random  maxmin  uncertainty  diversity_x_uncertainty  offset_uncertainty  same_chemotype_first  maxmin@ball  random@ball
checkpoint                                                                                                                          
0           1.8526  1.8526       1.8526                   1.8526              1.8526                1.8526       1.8519       1.8519
1           1.7714  1.6629       1.6900                   1.9043              1.7287                1.8609       1.6742       1.6968
2           1.7466  1.5260       1.7504                   1.6129              1.8177                1.8589       1.5886       1.6079
3           1.6140  1.5147       1.7565                   1.4652              1.8063                1.8515       1.5864       1.5610
5           1.5083  1.3104       1.6552                   1.4147              1.6410                1.8879       1.2899       1.4934
8           1.4140  1.2874       1.4149                   1.2694              1.5206                1.9028       1.2934       1.4486
12          1.3916  1.2765       1.3630                   1.2507              1.4184                1.9015       1.2439       1.3830
16          1.3514  1.2299       1.2649                   1.1623              1.3325                1.8801       1.2151       1.3992
20          1.3341  1.2076       1.2225                   1.1053              1.2454                1.9504       1.1864       1.2907
25          1.3165  1.1735       1.1587                   1.0942              1.1990                2.0146       1.1570       1.3123
30          1.2901  1.1898       1.1319                   1.0963              1.1666                1.9009       1.1753       1.3093
```

**macro MAE on hard chemistry (hard_nn<0.4; the FIXED start-cohort subset)**

```
arm         random  maxmin  uncertainty  diversity_x_uncertainty  offset_uncertainty  same_chemotype_first  maxmin@ball  random@ball
checkpoint                                                                                                                          
0           1.9751  1.9751       1.9751                   1.9751              1.9751                1.9751       1.9752       1.9752
1           1.8887  1.7778       1.8284                   2.0735              1.8683                1.9862       1.7861       1.7972
2           1.8462  1.5654       1.9142                   1.7205              1.9949                1.9858       1.6299       1.6997
3           1.7167  1.5278       1.9015                   1.5117              1.9981                1.9833       1.6079       1.6000
5           1.5729  1.2986       1.7798                   1.4364              1.7611                2.0336       1.2698       1.5580
8           1.4653  1.2548       1.4464                   1.2510              1.5638                2.0848       1.2728       1.5093
12          1.4547  1.2073       1.3655                   1.2093              1.4267                2.0859       1.1810       1.4292
16          1.4028  1.1412       1.2460                   1.1114              1.3166                2.0707       1.1359       1.4677
20          1.3828  1.1262       1.1823                   1.0654              1.2225                2.1333       1.1059       1.3338
25          1.3749  1.0946       1.1348                   1.0482              1.1588                2.2102       1.0939       1.3461
30          1.3473  1.1372       1.1217                   1.0398              1.1175                2.0901       1.1296       1.3537
```

**offset MAE (per-ligand level error — the quantity Phase 1 says is missing)**

```
arm         random  maxmin  uncertainty  diversity_x_uncertainty  offset_uncertainty  same_chemotype_first  maxmin@ball  random@ball
checkpoint                                                                                                                          
0           1.7562  1.7562       1.7562                   1.7562              1.7562                1.7562       1.7565       1.7565
1           1.6747  1.5857       1.5505                   1.8226              1.6023                1.7676       1.5956       1.5764
2           1.6417  1.4141       1.6344                   1.4828              1.7178                1.7579       1.4742       1.4790
3           1.4981  1.3894       1.6425                   1.3246              1.6848                1.7489       1.4537       1.4484
5           1.3821  1.1605       1.5253                   1.2709              1.5118                1.7853       1.1385       1.3737
8           1.2687  1.1452       1.2775                   1.1196              1.3951                1.7842       1.1499       1.3397
12          1.2395  1.1422       1.2291                   1.1117              1.2786                1.7852       1.1022       1.2755
16          1.1845  1.0936       1.1309                   1.0110              1.1865                1.7615       1.0840       1.2825
20          1.1665  1.0612       1.0759                   0.9335              1.0833                1.8517       1.0509       1.1439
25          1.1332  1.0322       0.9808                   0.9180              1.0284                1.9236       1.0099       1.1571
30          1.0954  1.0391       0.9407                   0.9154              0.9870                1.7929       1.0198       1.1510
```

**shape MAE (within-ligand response shape)**

```
arm         random  maxmin  uncertainty  diversity_x_uncertainty  offset_uncertainty  same_chemotype_first  maxmin@ball  random@ball
checkpoint                                                                                                                          
0           0.5858  0.5858       0.5858                   0.5858              0.5858                0.5858       0.5891       0.5891
1           0.5707  0.5438       0.5750                   0.5588              0.5745                0.5817       0.5439       0.5724
2           0.5620  0.5386       0.5593                   0.5526              0.5575                0.5843       0.5405       0.5702
3           0.5513  0.5475       0.5557                   0.5495              0.5539                0.5819       0.5529       0.5618
5           0.5462  0.5420       0.5483                   0.5473              0.5469                0.5921       0.5357       0.5561
8           0.5462  0.5378       0.5420                   0.5356              0.5407                0.5846       0.5320       0.5452
12          0.5447  0.5307       0.5397                   0.5363              0.5385                0.5833       0.5205       0.5318
16          0.5404  0.5350       0.5363                   0.5405              0.5373                0.5783       0.5229       0.5291
20          0.5380  0.5391       0.5369                   0.5435              0.5371                0.5764       0.5283       0.5235
25          0.5376  0.5366       0.5373                   0.5448              0.5413                0.5777       0.5336       0.5275
30          0.5366  0.5347       0.5378                   0.5443              0.5430                0.5758       0.5238       0.5308
```

**worst-quartile ligand MAE**

```
arm         random  maxmin  uncertainty  diversity_x_uncertainty  offset_uncertainty  same_chemotype_first  maxmin@ball  random@ball
checkpoint                                                                                                                          
0           3.1976  3.1976       3.1976                   3.1976              3.1976                3.1976       3.2023       3.2023
1           3.1050  2.9617       3.0052                   3.3365              3.0597                3.2270       2.9643       2.9769
2           3.0566  2.8109       3.1246                   2.8980              3.1964                3.2223       2.8792       2.9044
3           2.8336  2.8036       3.1946                   2.6763              3.1897                3.2144       2.9183       2.8279
5           2.6776  2.4575       3.0712                   2.6370              2.9409                3.2655       2.4407       2.7460
8           2.5603  2.3763       2.6603                   2.3994              2.7407                3.3479       2.3563       2.6690
12          2.5274  2.4101       2.5479                   2.3624              2.6044                3.3533       2.3323       2.5407
16          2.4565  2.3806       2.3321                   2.1472              2.4777                3.3221       2.3311       2.5665
20          2.4343  2.3144       2.2704                   2.0069              2.2858                3.4731       2.2808       2.3882
25          2.4551  2.2346       2.1407                   1.9956              2.2175                3.6112       2.2083       2.4822
30          2.3642  2.2412       2.0689                   2.0003              2.1434                3.4402       2.2403       2.4494
```

### Coverage — what each policy actually bought

The mechanism, not decoration: if two policies buy the same chemistry, no difference in accuracy is expected. `n_train_chemotypes` counts Tanimoto-0.7 chemotypes among the training ligands; `mean_nn_test_to_train` is the mean over **test ligands** of the maximum Tanimoto to the current training ligands (higher = the test set is better covered).

**chemotypes in training**

```
arm         random  maxmin  uncertainty  diversity_x_uncertainty  offset_uncertainty  same_chemotype_first  maxmin@ball  random@ball
checkpoint                                                                                                                          
0            1.000     1.0        1.000                    1.000               1.000                 1.000          1.0        1.000
1            1.833     2.0        2.000                    2.000               2.000                 1.200          2.0        1.733
2            2.633     3.0        3.000                    3.000               3.000                 1.267          3.0        2.467
3            3.500     4.0        3.967                    4.000               3.833                 1.400          4.0        3.267
5            5.100     6.0        5.800                    6.000               5.733                 1.533          6.0        4.667
8            7.367     9.0        8.667                    9.000               8.400                 1.933          9.0        6.800
12          10.433    13.0       12.367                   13.000              11.933                 2.267         13.0        9.933
16          12.867    17.0       15.967                   17.000              15.500                 2.400         17.0       12.667
20          15.700    21.0       19.400                   21.000              19.233                 2.800         21.0       15.267
25          19.033    26.0       23.900                   25.933              23.733                 3.333         26.0       18.133
30          21.800    31.0       28.233                   30.200              28.000                 5.600         31.0       21.333
```

**mean nearest-training-neighbour Tanimoto of the test ligands**

```
arm         random  maxmin  uncertainty  diversity_x_uncertainty  offset_uncertainty  same_chemotype_first  maxmin@ball  random@ball
checkpoint                                                                                                                          
0            0.275   0.275        0.275                    0.275               0.275                 0.275        0.275        0.275
1            0.322   0.283        0.296                    0.303               0.287                 0.278        0.284        0.296
2            0.332   0.324        0.344                    0.352               0.336                 0.278        0.326        0.326
3            0.354   0.325        0.363                    0.371               0.357                 0.278        0.326        0.350
5            0.378   0.362        0.382                    0.386               0.386                 0.279        0.361        0.365
8            0.399   0.384        0.408                    0.412               0.405                 0.284        0.380        0.387
12           0.426   0.427        0.429                    0.433               0.424                 0.294        0.427        0.396
16           0.437   0.446        0.442                    0.453               0.442                 0.298        0.446        0.417
20           0.450   0.454        0.461                    0.474               0.464                 0.318        0.455        0.431
25           0.461   0.469        0.474                    0.497               0.479                 0.338        0.472        0.441
30           0.475   0.486        0.491                    0.505               0.492                 0.369        0.488        0.452
```

**training rows**

```
arm           random    maxmin  uncertainty  diversity_x_uncertainty  offset_uncertainty  same_chemotype_first  maxmin@ball  random@ball
checkpoint                                                                                                                              
0           2384.267  2384.267     2384.267                 2384.267            2384.267              2384.267     2384.267     2384.267
1           2387.267  2387.267     2387.267                 2387.267            2387.267              2387.267     2389.733     2396.933
2           2390.267  2390.267     2390.267                 2390.267            2390.267              2390.267     2399.200     2410.600
3           2393.267  2393.267     2393.267                 2393.267            2393.267              2393.267     2402.800     2426.333
5           2399.267  2399.267     2399.267                 2399.267            2399.267              2399.267     2429.133     2465.933
8           2408.267  2408.267     2408.267                 2408.267            2408.267              2408.267     2452.333     2499.467
12          2420.267  2420.267     2420.267                 2420.267            2420.267              2420.267     2509.400     2556.867
16          2432.267  2432.267     2432.267                 2432.267            2432.267              2432.267     2545.533     2604.733
20          2444.267  2444.267     2444.267                 2444.267            2444.267              2444.267     2586.133     2654.467
25          2459.267  2459.267     2459.267                 2459.267            2459.267              2459.267     2655.333     2763.733
30          2474.267  2474.267     2474.267                 2474.267            2474.267              2474.267     2774.333     2828.267
```

## Pre-registered hypotheses (protocol §F)

Scoring rule, stated so a verdict cannot be over-read: **PASS** = the pre-registered condition is met **and** the run carried the pre-registered replication (≥ 2 split seeds, ≥ 2 acquisition replicates, ≥ 2 fold-jobs); **FAIL** = the interval excludes the predicted direction at ≥ half the scored checkpoints (evidence *against*, which counts even in a short run); **INCONCLUSIVE** = the intervals straddle zero, the contrast could not be formed, or the condition was met on a run too short to register it. An INCONCLUSIVE is not a weak PASS. Only F1–F4 are protected; every other interval in this report is descriptive.

**F1 — PASS**  ·  Max-min diversity beats random acquisition on hard chemistry.

* pass condition: hard macro MAE(random) − hard macro MAE(maxmin) with CI95 low > 0 at ≥ half the checkpoints from 5 onwards.
* measured: endpoint hard_nn<0.4 (test rows whose nearest neighbour in the START cohort is below 0.4 Tanimoto — a fixed subset), statistic macro MAE over ECFP clusters; 6/7 checkpoints meet the condition (need 4; BCa agrees at 5); Δ by checkpoint: k=5: +0.180 [+0.053, +0.276], BCa [+0.073, +0.298]; k=8: +0.059 [-0.020, +0.172], BCa [-0.021, +0.169]; k=12: +0.089 [+0.005, +0.216], BCa [-0.012, +0.191]; k=16: +0.112 [+0.037, +0.231], BCa [+0.024, +0.209]; k=20: +0.124 [+0.040, +0.255], BCa [+0.033, +0.243]; k=25: +0.198 [+0.118, +0.299], BCa [+0.122, +0.306]; k=30: +0.134 [+0.046, +0.236], BCa [+0.046, +0.235].
* what would falsify this: If maxmin is indistinguishable from random, the *choice* of ligand does not matter, only the breadth — Experiment B already showed breadth matters, so F1 dying reduces the deployable advice to 'buy any new chemotype'. An interval whose upper end is below zero would be stronger still: random would then be the better rule.

**F2 — INCONCLUSIVE**  ·  Acquiring the most level-uncertain ligand beats random.

* pass condition: offset MAE(random) − offset MAE(offset_uncertainty) with CI95 low > 0 at ≥ half the checkpoints from 5 onwards.
* measured: endpoint all test rows, statistic offset MAE (the per-ligand level error, the quantity Phase 1 says is missing); 2/7 checkpoints meet the condition (need 4; BCa agrees at 0); Δ by checkpoint: k=5: -0.110 [-0.333, +0.074], BCa [-0.283, +0.129]; k=8: -0.141 [-0.215, -0.038], BCa [-0.219, -0.044]; k=12: -0.056 [-0.146, +0.071], BCa [-0.169, +0.042]; k=16: -0.017 [-0.101, +0.102], BCa [-0.119, +0.079]; k=20: +0.053 [-0.044, +0.191], BCa [-0.067, +0.163]; k=25: +0.086 [+0.003, +0.209], BCa [-0.009, +0.189]; k=30: +0.093 [+0.006, +0.222], BCa [-0.009, +0.199]. Descriptive, for the protocol's own falsifier (is the model's uncertainty worth anything beyond chemical distance?) — the same statistic against **maxmin** rather than random: k=5: -0.254 [-0.453, -0.115], BCa [-0.428, -0.096]; k=8: -0.200 [-0.291, -0.089], BCa [-0.291, -0.089]; k=12: -0.074 [-0.144, +0.023], BCa [-0.148, +0.016]; k=16: -0.045 [-0.113, +0.048], BCa [-0.120, +0.040]; k=20: -0.019 [-0.104, +0.098], BCa [-0.123, +0.073]; k=25: -0.012 [-0.115, +0.120], BCa [-0.146, +0.093]; k=30: +0.044 [-0.045, +0.165], BCa [-0.062, +0.142]
* what would falsify this: If the model's own level uncertainty is no better than chance — or no better than the chemistry-only maxmin rule — then the forest carries no acquisition information beyond chemical distance, and the cheap rule is the one to deploy.

**F3 — PASS**  ·  Buying more of the same chemistry is worse than random.

* pass condition: hard macro MAE(random) − hard macro MAE(same_chemotype_first) with CI95 **high < 0** (random is the better rule) at ≥ half the checkpoints. NOTE: the protocol table writes this as 'hard MAE(same_chemotype_first) − hard MAE(random), CI95 high < 0', which is the opposite of its own gloss 'i.e. random is better'; the gloss is scored here and the arithmetic is printed with every number so the reading cannot be hidden.
* measured: endpoint hard_nn<0.4, statistic macro MAE; delta is random minus same_chemotype_first, so a **negative** interval means random is better; 10/10 checkpoints meet the condition (need 5; BCa agrees at 10); Δ by checkpoint: k=1: -0.118 [-0.216, -0.042], BCa [-0.206, -0.034]; k=2: -0.197 [-0.344, -0.085], BCa [-0.322, -0.067]; k=3: -0.335 [-0.443, -0.226], BCa [-0.441, -0.225]; k=5: -0.469 [-0.613, -0.335], BCa [-0.613, -0.335]; k=8: -0.573 [-0.733, -0.432], BCa [-0.729, -0.430]; k=12: -0.506 [-0.722, -0.351], BCa [-0.697, -0.336]; k=16: -0.544 [-0.712, -0.396], BCa [-0.722, -0.402]; k=20: -0.697 [-0.874, -0.466], BCa [-0.907, -0.498]; k=25: -0.814 [-1.023, -0.532], BCa [-1.062, -0.572]; k=30: -0.639 [-0.847, -0.451], BCa [-0.848, -0.451].
* what would falsify this: If buying another analogue of what you already have matches random acquisition on hard chemistry, then the field's actual behaviour cost nothing measurable at this scale, and the generation's advice weakens to 'do not bother choosing'.

**F4 — SKIPPED**  ·  Three measurements of a distant ligand beat three more of a known one.

* pass condition: the b = 3 maxmin curve lies below the curve that spends the same rows on the start cohort's own ligands, at ≥ half the checkpoints.
* measured: needs --f4-cap-start-rows; with the full start cohort there are no unrevealed start rows to buy, and Experiment B already measured depth vs breadth
* what would falsify this: Not evaluated in this run. With --f4-cap-start-rows N it is evaluated against the depth_on_start arm; if depth matched breadth there, Experiment B's headline would be contradicted at the level of individual measurements.

## Every contrast, every checkpoint

Reference is `random` (F4's is the depth arm). Positive delta = the candidate policy is better. `n_folds` is the number of (seed, fold) cells behind the row; `units_total` the number of ECFP clusters; `bootstrap_blocks` the number of chemotypes resampled. Only the protected contrast set is printed here; the descriptive sets (`vs_maxmin|…`, `budget_all|…`) and the `shape_mae` statistic are in `contrasts.csv`.

```
   endpoint  checkpoint                        comparison  statistic  point_delta  ci95_low  ci95_high  bca_low  bca_high  cluster_robust_low  block_macro_delta  units_improved  units_total  bootstrap_blocks  n_folds
        all           1                  maxmin_vs_random        mae       0.0907    0.0443     0.1296   0.0479    0.1320              0.0492             0.0882              91          131                79       15
        all           1                  maxmin_vs_random offset_mae       0.0640    0.0109     0.1058   0.0153    0.1103              0.0160             0.0529              84          131                79       15
        all           1             uncertainty_vs_random        mae       0.1599   -0.0818     0.3971  -0.0459    0.5030             -0.1318            -0.0037              77          131                79       15
        all           1             uncertainty_vs_random offset_mae       0.1185   -0.1959     0.4419  -0.1565    0.5927             -0.2782            -0.1037              62          131                79       15
        all           1 diversity_x_uncertainty_vs_random        mae      -0.1118   -0.2395    -0.0134  -0.2157    0.0085             -0.2337            -0.1469              58          131                79       15
        all           1 diversity_x_uncertainty_vs_random offset_mae      -0.1140   -0.2542    -0.0098  -0.2298    0.0078             -0.2427            -0.1550              62          131                79       15
        all           1      offset_uncertainty_vs_random        mae       0.1231   -0.0593     0.2948  -0.0282    0.3706             -0.0889            -0.0058              80          131                79       15
        all           1      offset_uncertainty_vs_random offset_mae       0.0824   -0.1496     0.3124  -0.1155    0.4190             -0.2017            -0.0841              67          131                79       15
        all           1    same_chemotype_first_vs_random        mae      -0.1025   -0.1778    -0.0446  -0.1697   -0.0390             -0.1728            -0.0892              40          131                79       15
        all           1    same_chemotype_first_vs_random offset_mae      -0.0764   -0.1519    -0.0169  -0.1507   -0.0158             -0.1457            -0.0701              52          131                79       15
hard_nn<0.4           1                  maxmin_vs_random        mae       0.0803    0.0205     0.1287   0.0235    0.1305              0.0279             0.0880              81          119                69       15
hard_nn<0.4           1                  maxmin_vs_random offset_mae       0.0412   -0.0330     0.0960  -0.0240    0.1035             -0.0241             0.0410              70          119                69       15
hard_nn<0.4           1             uncertainty_vs_random        mae       0.2295   -0.0405     0.4664   0.0011    0.5494             -0.0668             0.0558              71          119                69       15
hard_nn<0.4           1             uncertainty_vs_random offset_mae       0.2459   -0.1038     0.5540  -0.0520    0.6636             -0.1438            -0.0036              63          119                69       15
hard_nn<0.4           1 diversity_x_uncertainty_vs_random        mae      -0.1237   -0.2754    -0.0165  -0.2448    0.0094             -0.2635            -0.1594              55          119                69       15
hard_nn<0.4           1 diversity_x_uncertainty_vs_random offset_mae      -0.1420   -0.3109    -0.0230  -0.2791    0.0062             -0.2943            -0.1793              53          119                69       15
hard_nn<0.4           1      offset_uncertainty_vs_random        mae       0.1923   -0.0099     0.3580   0.0278    0.4183             -0.0173             0.0582              75          119                69       15
hard_nn<0.4           1      offset_uncertainty_vs_random offset_mae       0.2075   -0.0502     0.4215  -0.0021    0.4973             -0.0609             0.0208              69          119                69       15
hard_nn<0.4           1    same_chemotype_first_vs_random        mae      -0.1177   -0.2163    -0.0421  -0.2064   -0.0341             -0.2071            -0.0927              38          119                69       15
hard_nn<0.4           1    same_chemotype_first_vs_random offset_mae      -0.1502   -0.2684    -0.0585  -0.2542   -0.0443             -0.2585            -0.1295              40          119                69       15
hard_nn<0.6           1                  maxmin_vs_random        mae       0.0827    0.0258     0.1263   0.0281    0.1281              0.0336             0.0858              84          127                76       15
hard_nn<0.6           1                  maxmin_vs_random offset_mae       0.0439   -0.0268     0.0934  -0.0190    0.0995             -0.0166             0.0395              76          127                76       15
hard_nn<0.6           1             uncertainty_vs_random        mae       0.2018   -0.0495     0.4349  -0.0092    0.5207             -0.0848             0.0388              76          127                76       15
hard_nn<0.6           1             uncertainty_vs_random offset_mae       0.1941   -0.1349     0.5041  -0.0880    0.6230             -0.1890            -0.0266              63          127                76       15
hard_nn<0.6           1 diversity_x_uncertainty_vs_random        mae      -0.1126   -0.2474    -0.0131  -0.2267    0.0105             -0.2400            -0.1400              59          127                76       15
hard_nn<0.6           1 diversity_x_uncertainty_vs_random offset_mae      -0.1274   -0.2756    -0.0200  -0.2540    0.0048             -0.2656            -0.1575              59          127                76       15
hard_nn<0.6           1      offset_uncertainty_vs_random        mae       0.1606   -0.0312     0.3279   0.0050    0.3859             -0.0462             0.0338              78          127                76       15
hard_nn<0.6           1      offset_uncertainty_vs_random offset_mae       0.1493   -0.0939     0.3697  -0.0534    0.4438             -0.1220            -0.0148              69          127                76       15
hard_nn<0.6           1    same_chemotype_first_vs_random        mae      -0.1096   -0.1965    -0.0421  -0.1875   -0.0344             -0.1894            -0.0846              41          127                76       15
hard_nn<0.6           1    same_chemotype_first_vs_random offset_mae      -0.1101   -0.2082    -0.0346  -0.2022   -0.0295             -0.1984            -0.0878              50          127                76       15
        all           2                  maxmin_vs_random        mae       0.1385    0.0402     0.2170   0.0447    0.2216              0.0472             0.1370              98          131                79       15
        all           2                  maxmin_vs_random offset_mae       0.1217    0.0274     0.1991   0.0384    0.2129              0.0319             0.0894              87          131                79       15
        all           2             uncertainty_vs_random        mae       0.0312   -0.2396     0.2912  -0.1917    0.4065             -0.2862            -0.1390              62          131                79       15
        all           2             uncertainty_vs_random offset_mae       0.0705   -0.3079     0.4482  -0.2553    0.6266             -0.3968            -0.2113              61          131                79       15
        all           2 diversity_x_uncertainty_vs_random        mae       0.1360   -0.1241     0.3830  -0.0836    0.4835             -0.1666             0.0030              77          131                79       15
        all           2 diversity_x_uncertainty_vs_random offset_mae       0.1587   -0.1631     0.4764  -0.1172    0.6117             -0.2273            -0.0428              75          131                79       15
        all           2      offset_uncertainty_vs_random        mae      -0.0260   -0.2823     0.2113  -0.2365    0.3030             -0.3155            -0.1608              63          131                79       15
        all           2      offset_uncertainty_vs_random offset_mae      -0.0052   -0.2879     0.2602  -0.2401    0.3795             -0.3356            -0.2030              65          131                79       15
        all           2    same_chemotype_first_vs_random        mae      -0.1802   -0.2959    -0.0889  -0.2798   -0.0725             -0.2918            -0.1779              39          131                79       15
        all           2    same_chemotype_first_vs_random offset_mae      -0.1771   -0.2965    -0.0869  -0.2842   -0.0785             -0.2877            -0.1885              44          131                79       15
hard_nn<0.4           2                  maxmin_vs_random        mae       0.1568    0.0453     0.2417   0.0514    0.2459              0.0599             0.1575              88          119                69       15
hard_nn<0.4           2                  maxmin_vs_random offset_mae       0.1459    0.0377     0.2266   0.0516    0.2378              0.0520             0.1144              81          119                69       15
hard_nn<0.4           2             uncertainty_vs_random        mae       0.0262   -0.2825     0.3019  -0.2392    0.4024             -0.3272            -0.1556              56          119                69       15
hard_nn<0.4           2             uncertainty_vs_random offset_mae       0.0783   -0.3619     0.4790  -0.3013    0.6451             -0.4368            -0.2133              57          119                69       15
hard_nn<0.4           2 diversity_x_uncertainty_vs_random        mae       0.1778   -0.1137     0.4275  -0.0657    0.4998             -0.1402             0.0346              72          119                69       15
hard_nn<0.4           2 diversity_x_uncertainty_vs_random offset_mae       0.2357   -0.1242     0.5434  -0.0632    0.6427             -0.1600             0.0282              72          119                69       15
hard_nn<0.4           2      offset_uncertainty_vs_random        mae      -0.0225   -0.3140     0.2289  -0.2696    0.3054             -0.3418            -0.1573              59          119                69       15
hard_nn<0.4           2      offset_uncertainty_vs_random offset_mae       0.0132   -0.3156     0.2938  -0.2616    0.3990             -0.3465            -0.1744              59          119                69       15
hard_nn<0.4           2    same_chemotype_first_vs_random        mae      -0.1970   -0.3441    -0.0853  -0.3215   -0.0671             -0.3326            -0.1812              37          119                69       15
hard_nn<0.4           2    same_chemotype_first_vs_random offset_mae      -0.2396   -0.4016    -0.1145  -0.3762   -0.0933             -0.3893            -0.2255              35          119                69       15
hard_nn<0.6           2                  maxmin_vs_random        mae       0.1407    0.0366     0.2239   0.0411    0.2269              0.0466             0.1393              94          127                76       15
hard_nn<0.6           2                  maxmin_vs_random offset_mae       0.1250    0.0229     0.2051   0.0353    0.2165              0.0323             0.0948              85          127                76       15
hard_nn<0.6           2             uncertainty_vs_random        mae       0.0313   -0.2532     0.2979  -0.2103    0.4040             -0.2980            -0.1337              63          127                76       15
hard_nn<0.6           2             uncertainty_vs_random offset_mae       0.0830   -0.3205     0.4706  -0.2609    0.6211             -0.3967            -0.1831              61          127                76       15
hard_nn<0.6           2 diversity_x_uncertainty_vs_random        mae       0.1568   -0.1142     0.4042  -0.0694    0.4977             -0.1485             0.0241              76          127                76       15
hard_nn<0.6           2 diversity_x_uncertainty_vs_random offset_mae       0.2008   -0.1307     0.5123  -0.0796    0.6324             -0.1824             0.0088              74          127                76       15
hard_nn<0.6           2      offset_uncertainty_vs_random        mae      -0.0229   -0.2930     0.2173  -0.2487    0.3067             -0.3221            -0.1466              64          127                76       15
hard_nn<0.6           2      offset_uncertainty_vs_random offset_mae       0.0081   -0.2949     0.2815  -0.2433    0.3810             -0.3303            -0.1659              64          127                76       15
hard_nn<0.6           2    same_chemotype_first_vs_random        mae      -0.1894   -0.3196    -0.0875  -0.3029   -0.0718             -0.3122            -0.1715              39          127                76       15
hard_nn<0.6           2    same_chemotype_first_vs_random offset_mae      -0.1977   -0.3318    -0.0925  -0.3173   -0.0788             -0.3225            -0.1877              41          127                76       15
        all           3                  maxmin_vs_random        mae      -0.0286   -0.1034     0.0786  -0.1183    0.0572             -0.1215             0.0235              59          131                79       15
        all           3                  maxmin_vs_random offset_mae      -0.0336   -0.1419     0.1159  -0.1747    0.0825             -0.1700             0.0530              56          131                79       15
        all           3             uncertainty_vs_random        mae      -0.1797   -0.3263    -0.0622  -0.2959   -0.0355             -0.3210            -0.2325              50          131                79       15
        all           3             uncertainty_vs_random offset_mae      -0.0451   -0.3017     0.1880  -0.2497    0.2820             -0.3294            -0.2058              67          131                79       15
        all           3 diversity_x_uncertainty_vs_random        mae       0.1062   -0.0773     0.2660  -0.0449    0.3234             -0.0906             0.0460              75          131                79       15
        all           3 diversity_x_uncertainty_vs_random offset_mae       0.1207   -0.1078     0.3204  -0.0644    0.3835             -0.1208             0.0335              74          131                79       15
        all           3      offset_uncertainty_vs_random        mae      -0.1892   -0.4372     0.0246  -0.3834    0.0955             -0.4513            -0.3005              56          131                79       15
        all           3      offset_uncertainty_vs_random offset_mae      -0.1478   -0.4230     0.0964  -0.3650    0.1895             -0.4462            -0.3226              59          131                79       15
        all           3    same_chemotype_first_vs_random        mae      -0.3331   -0.4190    -0.2477  -0.4190   -0.2478             -0.4157            -0.3060              26          131                79       15
        all           3    same_chemotype_first_vs_random offset_mae      -0.3369   -0.4284    -0.2180  -0.4297   -0.2210             -0.4374            -0.2933              31          131                79       15
hard_nn<0.4           3                  maxmin_vs_random        mae       0.0033   -0.0850     0.1356  -0.1034    0.1110             -0.1114             0.0707              55          119                69       15
hard_nn<0.4           3                  maxmin_vs_random offset_mae       0.0081   -0.1223     0.1944  -0.1578    0.1541             -0.1607             0.1221              49          119                69       15
hard_nn<0.4           3             uncertainty_vs_random        mae      -0.1924   -0.3664    -0.0625  -0.3330   -0.0322             -0.3539            -0.2388              44          119                69       15
hard_nn<0.4           3             uncertainty_vs_random offset_mae      -0.0312   -0.3354     0.2116  -0.2729    0.3005             -0.3441            -0.1658              61          119                69       15
hard_nn<0.4           3 diversity_x_uncertainty_vs_random        mae       0.1790   -0.0211     0.3294   0.0160    0.3625             -0.0112             0.1218              72          119                69       15
hard_nn<0.4           3 diversity_x_uncertainty_vs_random offset_mae       0.2375   -0.0074     0.4162   0.0382    0.4485              0.0123             0.1600              75          119                69       15
hard_nn<0.4           3      offset_uncertainty_vs_random        mae      -0.1719   -0.4508     0.0509  -0.3978    0.1121             -0.4527            -0.2894              53          119                69       15
hard_nn<0.4           3      offset_uncertainty_vs_random offset_mae      -0.0950   -0.4080     0.1487  -0.3429    0.2358             -0.4039            -0.2531              58          119                69       15
hard_nn<0.4           3    same_chemotype_first_vs_random        mae      -0.3349   -0.4428    -0.2264  -0.4407   -0.2251             -0.4338            -0.2938              26          119                69       15
hard_nn<0.4           3    same_chemotype_first_vs_random offset_mae      -0.3610   -0.4734    -0.2105  -0.4726   -0.2099             -0.4808            -0.2703              24          119                69       15
hard_nn<0.6           3                  maxmin_vs_random        mae      -0.0322   -0.1064     0.0781  -0.1212    0.0599             -0.1272             0.0237              56          127                76       15
hard_nn<0.6           3                  maxmin_vs_random offset_mae      -0.0398   -0.1469     0.1131  -0.1768    0.0827             -0.1786             0.0549              51          127                76       15
hard_nn<0.6           3             uncertainty_vs_random        mae      -0.1919   -0.3524    -0.0675  -0.3209   -0.0372             -0.3456            -0.2408              49          127                76       15
hard_nn<0.6           3             uncertainty_vs_random offset_mae      -0.0429   -0.3178     0.1978  -0.2678    0.2852             -0.3394            -0.1830              67          127                76       15
hard_nn<0.6           3 diversity_x_uncertainty_vs_random        mae       0.1409   -0.0494     0.2961  -0.0136    0.3418             -0.0517             0.0830              75          127                76       15
hard_nn<0.6           3 diversity_x_uncertainty_vs_random offset_mae       0.1839   -0.0479     0.3691  -0.0085    0.4157             -0.0466             0.1074              77          127                76       15
hard_nn<0.6           3      offset_uncertainty_vs_random        mae      -0.1847   -0.4470     0.0334  -0.3973    0.1048             -0.4548            -0.2899              56          127                76       15
hard_nn<0.6           3      offset_uncertainty_vs_random offset_mae      -0.1213   -0.4134     0.1199  -0.3538    0.2093             -0.4224            -0.2670              60          127                76       15
hard_nn<0.6           3    same_chemotype_first_vs_random        mae      -0.3364   -0.4322    -0.2456  -0.4314   -0.2442             -0.4254            -0.2961              27          127                76       15
hard_nn<0.6           3    same_chemotype_first_vs_random offset_mae      -0.3426   -0.4406    -0.2133  -0.4445   -0.2202             -0.4500            -0.2694              29          127                76       15
        all           5                  maxmin_vs_random        mae       0.1383    0.0137     0.2375   0.0355    0.2637              0.0184             0.1127              85          131                79       15
        all           5                  maxmin_vs_random offset_mae       0.1439    0.0033     0.2483   0.0315    0.2697              0.0182             0.1160              80          131                79       15
        all           5             uncertainty_vs_random        mae      -0.1858   -0.2650    -0.0842  -0.2677   -0.0905             -0.2702            -0.1576              39          131                79       15
        all           5             uncertainty_vs_random offset_mae      -0.0766   -0.1930     0.0171  -0.1941    0.0164             -0.1799            -0.1136              62          131                79       15
        all           5 diversity_x_uncertainty_vs_random        mae       0.0568   -0.1033     0.1869  -0.0760    0.2220             -0.1043             0.0218              76          131                79       15
        all           5 diversity_x_uncertainty_vs_random offset_mae       0.0918   -0.1075     0.2586  -0.0700    0.3082             -0.1118             0.0194              76          131                79       15
        all           5      offset_uncertainty_vs_random        mae      -0.1199   -0.3095     0.0404  -0.2723    0.0875             -0.3153            -0.2123              59          131                79       15
        all           5      offset_uncertainty_vs_random offset_mae      -0.1105   -0.3326     0.0742  -0.2830    0.1295             -0.3342            -0.2355              57          131                79       15
        all           5    same_chemotype_first_vs_random        mae      -0.4478   -0.5569    -0.3401  -0.5598   -0.3431             -0.5517            -0.4344              25          131                79       15
        all           5    same_chemotype_first_vs_random offset_mae      -0.4790   -0.5980    -0.3194  -0.6158   -0.3354             -0.6204            -0.4262              28          131                79       15
hard_nn<0.4           5                  maxmin_vs_random        mae       0.1801    0.0529     0.2763   0.0731    0.2983              0.0621             0.1624              81          119                69       15
hard_nn<0.4           5                  maxmin_vs_random offset_mae       0.2035    0.0719     0.3035   0.0903    0.3212              0.0851             0.1908              78          119                69       15
hard_nn<0.4           5             uncertainty_vs_random        mae      -0.1981   -0.3078    -0.0668  -0.3053   -0.0616             -0.3110            -0.1474              33          119                69       15
hard_nn<0.4           5             uncertainty_vs_random offset_mae      -0.0649   -0.2355     0.0769  -0.2446    0.0711             -0.2119            -0.0550              59          119                69       15
hard_nn<0.4           5 diversity_x_uncertainty_vs_random        mae       0.0991   -0.0768     0.2305  -0.0480    0.2572             -0.0653             0.0625              70          119                69       15
hard_nn<0.4           5 diversity_x_uncertainty_vs_random offset_mae       0.1602   -0.0562     0.3179  -0.0156    0.3522             -0.0404             0.0913              73          119                69       15
hard_nn<0.4           5      offset_uncertainty_vs_random        mae      -0.1084   -0.3313     0.0606  -0.2857    0.1078             -0.3215            -0.2029              58          119                69       15
hard_nn<0.4           5      offset_uncertainty_vs_random offset_mae      -0.0811   -0.3389     0.1074  -0.2852    0.1666             -0.3216            -0.1876              58          119                69       15
hard_nn<0.4           5    same_chemotype_first_vs_random        mae      -0.4687   -0.6127    -0.3355  -0.6127   -0.3355             -0.5969            -0.4388              28          119                69       15
hard_nn<0.4           5    same_chemotype_first_vs_random offset_mae      -0.5303   -0.6754    -0.3301  -0.6810   -0.3386             -0.6909            -0.4261              27          119                69       15
hard_nn<0.6           5                  maxmin_vs_random        mae       0.1500    0.0255     0.2503   0.0468    0.2729              0.0290             0.1295              85          127                76       15
hard_nn<0.6           5                  maxmin_vs_random offset_mae       0.1700    0.0349     0.2734   0.0519    0.2892              0.0457             0.1558              81          127                76       15
hard_nn<0.6           5             uncertainty_vs_random        mae      -0.2055   -0.2981    -0.1001  -0.2999   -0.1030             -0.3013            -0.1674              38          127                76       15
hard_nn<0.6           5             uncertainty_vs_random offset_mae      -0.0806   -0.2265     0.0374  -0.2386    0.0324             -0.2118            -0.0829              63          127                76       15
hard_nn<0.6           5 diversity_x_uncertainty_vs_random        mae       0.0687   -0.0980     0.2012  -0.0670    0.2318             -0.0950             0.0334              74          127                76       15
hard_nn<0.6           5 diversity_x_uncertainty_vs_random offset_mae       0.1268   -0.0767     0.2855  -0.0390    0.3272             -0.0726             0.0654              76          127                76       15
hard_nn<0.6           5      offset_uncertainty_vs_random        mae      -0.1226   -0.3306     0.0438  -0.2865    0.0929             -0.3275            -0.2097              58          127                76       15
hard_nn<0.6           5      offset_uncertainty_vs_random offset_mae      -0.1001   -0.3407     0.0869  -0.2931    0.1373             -0.3332            -0.1976              57          127                76       15
hard_nn<0.6           5    same_chemotype_first_vs_random        mae      -0.4607   -0.5879    -0.3479  -0.5890   -0.3502             -0.5731            -0.4324              28          127                76       15
hard_nn<0.6           5    same_chemotype_first_vs_random offset_mae      -0.5023   -0.6306    -0.3313  -0.6395   -0.3422             -0.6492            -0.4183              28          127                76       15
        all           8                  maxmin_vs_random        mae       0.0505   -0.0171     0.1467  -0.0177    0.1455             -0.0285             0.1090              69          131                79       15
        all           8                  maxmin_vs_random offset_mae       0.0594   -0.0287     0.1642  -0.0315    0.1614             -0.0328             0.1150              67          131                79       15
        all           8             uncertainty_vs_random        mae      -0.0362   -0.1211     0.0944  -0.1362    0.0746             -0.1469             0.0526              42          131                79       15
        all           8             uncertainty_vs_random offset_mae      -0.0041   -0.0872     0.1177  -0.0949    0.1050             -0.1066             0.0630              57          131                79       15
        all           8 diversity_x_uncertainty_vs_random        mae       0.0809    0.0097     0.1733   0.0101    0.1741              0.0034             0.1195              74          131                79       15
        all           8 diversity_x_uncertainty_vs_random offset_mae       0.0922    0.0025     0.1800   0.0036    0.1804              0.0080             0.1057              66          131                79       15
        all           8      offset_uncertainty_vs_random        mae      -0.1443   -0.2099    -0.0520  -0.2144   -0.0604             -0.2201            -0.0856              39          131                79       15
        all           8      offset_uncertainty_vs_random offset_mae      -0.1405   -0.2153    -0.0376  -0.2188   -0.0444             -0.2257            -0.0947              47          131                79       15
        all           8    same_chemotype_first_vs_random        mae      -0.5249   -0.6415    -0.4019  -0.6440   -0.4066             -0.6369            -0.5148              26          131                79       15
        all           8    same_chemotype_first_vs_random offset_mae      -0.5694   -0.7005    -0.3972  -0.7052   -0.4043             -0.7174            -0.5325              29          131                79       15
hard_nn<0.4           8                  maxmin_vs_random        mae       0.0588   -0.0202     0.1716  -0.0210    0.1692             -0.0323             0.1314              64          119                69       15
hard_nn<0.4           8                  maxmin_vs_random offset_mae       0.0685   -0.0323     0.1931  -0.0317    0.1937             -0.0371             0.1382              62          119                69       15
hard_nn<0.4           8             uncertainty_vs_random        mae      -0.0730   -0.1697     0.0776  -0.1781    0.0603             -0.1927             0.0403              41          119                69       15
hard_nn<0.4           8             uncertainty_vs_random offset_mae      -0.0403   -0.1554     0.1168  -0.1636    0.1050             -0.1673             0.0640              55          119                69       15
hard_nn<0.4           8 diversity_x_uncertainty_vs_random        mae       0.0817   -0.0023     0.1866   0.0018    0.1925             -0.0060             0.1286              69          119                69       15
hard_nn<0.4           8 diversity_x_uncertainty_vs_random offset_mae       0.1104    0.0139     0.2100   0.0176    0.2143              0.0198             0.1326              66          119                69       15
hard_nn<0.4           8      offset_uncertainty_vs_random        mae      -0.1752   -0.2723    -0.0586  -0.2791   -0.0702             -0.2725            -0.0967              37          119                69       15
hard_nn<0.4           8      offset_uncertainty_vs_random offset_mae      -0.1719   -0.2950    -0.0327  -0.3104   -0.0461             -0.2911            -0.0867              45          119                69       15
hard_nn<0.4           8    same_chemotype_first_vs_random        mae      -0.5733   -0.7333    -0.4315  -0.7290   -0.4302             -0.7090            -0.5517              24          119                69       15
hard_nn<0.4           8    same_chemotype_first_vs_random offset_mae      -0.6465   -0.8052    -0.4496  -0.8061   -0.4519             -0.8107            -0.5669              25          119                69       15
hard_nn<0.6           8                  maxmin_vs_random        mae       0.0495   -0.0239     0.1550  -0.0253    0.1516             -0.0335             0.1123              70          127                76       15
hard_nn<0.6           8                  maxmin_vs_random offset_mae       0.0529   -0.0457     0.1693  -0.0467    0.1683             -0.0472             0.1143              67          127                76       15
hard_nn<0.6           8             uncertainty_vs_random        mae      -0.0538   -0.1472     0.0875  -0.1579    0.0674             -0.1690             0.0463              43          127                76       15
hard_nn<0.6           8             uncertainty_vs_random offset_mae      -0.0306   -0.1338     0.1127  -0.1463    0.0948             -0.1475             0.0563              57          127                76       15
hard_nn<0.6           8 diversity_x_uncertainty_vs_random        mae       0.0769    0.0030     0.1756   0.0030    0.1758             -0.0049             0.1156              74          127                76       15
hard_nn<0.6           8 diversity_x_uncertainty_vs_random offset_mae       0.0970    0.0052     0.1911   0.0072    0.1937              0.0090             0.1137              69          127                76       15
hard_nn<0.6           8      offset_uncertainty_vs_random        mae      -0.1675   -0.2514    -0.0618  -0.2610   -0.0698             -0.2561            -0.0986              40          127                76       15
hard_nn<0.6           8      offset_uncertainty_vs_random offset_mae      -0.1677   -0.2733    -0.0432  -0.2884   -0.0545             -0.2770            -0.0974              46          127                76       15
hard_nn<0.6           8    same_chemotype_first_vs_random        mae      -0.5456   -0.6737    -0.4225  -0.6725   -0.4212             -0.6623            -0.5228              27          127                76       15
hard_nn<0.6           8    same_chemotype_first_vs_random offset_mae      -0.6092   -0.7521    -0.4316  -0.7535   -0.4356             -0.7612            -0.5473              29          127                76       15
        all          12                  maxmin_vs_random        mae       0.0358   -0.0319     0.1362  -0.0382    0.1228             -0.0482             0.0982              63          131                79       15
        all          12                  maxmin_vs_random offset_mae       0.0180   -0.0607     0.1305  -0.0656    0.1240             -0.0767             0.0873              59          131                79       15
        all          12             uncertainty_vs_random        mae      -0.0453   -0.1595     0.1093  -0.2082    0.0742             -0.1916             0.0871              46          131                79       15
        all          12             uncertainty_vs_random offset_mae      -0.0091   -0.1302     0.1511  -0.1791    0.1154             -0.1629             0.1088              60          131                79       15
        all          12 diversity_x_uncertainty_vs_random        mae       0.0870    0.0228     0.1676   0.0242    0.1702              0.0181             0.1221              71          131                79       15
        all          12 diversity_x_uncertainty_vs_random offset_mae       0.0845   -0.0119     0.1689  -0.0100    0.1704             -0.0064             0.0989              68          131                79       15
        all          12      offset_uncertainty_vs_random        mae      -0.0999   -0.1928     0.0288  -0.2181   -0.0008             -0.2149             0.0025              43          131                79       15
        all          12      offset_uncertainty_vs_random offset_mae      -0.0562   -0.1460     0.0714  -0.1693    0.0423             -0.1682             0.0331              54          131                79       15
        all          12    same_chemotype_first_vs_random        mae      -0.5000   -0.6814    -0.3674  -0.6629   -0.3501             -0.6575            -0.5688              29          131                79       15
        all          12    same_chemotype_first_vs_random offset_mae      -0.5548   -0.7263    -0.4077  -0.7211   -0.4026             -0.7078            -0.6006              34          131                79       15
hard_nn<0.4          12                  maxmin_vs_random        mae       0.0894    0.0050     0.2163  -0.0119    0.1911             -0.0202             0.1687              62          119                69       15
hard_nn<0.4          12                  maxmin_vs_random offset_mae       0.0882   -0.0034     0.2328  -0.0245    0.2035             -0.0319             0.1769              60          119                69       15
hard_nn<0.4          12             uncertainty_vs_random        mae      -0.0982   -0.2189     0.0844  -0.2480    0.0487             -0.2519             0.0629              42          119                69       15
hard_nn<0.4          12             uncertainty_vs_random offset_mae      -0.0711   -0.2030     0.1255  -0.2347    0.0887             -0.2364             0.0899              51          119                69       15
hard_nn<0.4          12 diversity_x_uncertainty_vs_random        mae       0.1166    0.0466     0.2146   0.0453    0.2132              0.0364             0.1614              73          119                69       15
hard_nn<0.4          12 diversity_x_uncertainty_vs_random offset_mae       0.1378    0.0407     0.2313   0.0389    0.2288              0.0480             0.1608              70          119                69       15
hard_nn<0.4          12      offset_uncertainty_vs_random        mae      -0.1162   -0.2203     0.0465  -0.2446    0.0120             -0.2497             0.0232              39          119                69       15
hard_nn<0.4          12      offset_uncertainty_vs_random offset_mae      -0.0679   -0.1808     0.1045  -0.2026    0.0727             -0.2082             0.0768              51          119                69       15
hard_nn<0.4          12    same_chemotype_first_vs_random        mae      -0.5058   -0.7215    -0.3508  -0.6974   -0.3363             -0.6846            -0.5736              31          119                69       15
hard_nn<0.4          12    same_chemotype_first_vs_random offset_mae      -0.5928   -0.8018    -0.4263  -0.7949   -0.4231             -0.7689            -0.6180              31          119                69       15
hard_nn<0.6          12                  maxmin_vs_random        mae       0.0527   -0.0184     0.1620  -0.0294    0.1474             -0.0382             0.1186              64          127                76       15
hard_nn<0.6          12                  maxmin_vs_random offset_mae       0.0394   -0.0426     0.1645  -0.0505    0.1507             -0.0626             0.1148              60          127                76       15
hard_nn<0.6          12             uncertainty_vs_random        mae      -0.0685   -0.1876     0.1030  -0.2209    0.0669             -0.2193             0.0780              47          127                76       15
hard_nn<0.6          12             uncertainty_vs_random offset_mae      -0.0328   -0.1630     0.1477  -0.2009    0.1113             -0.1947             0.1094              58          127                76       15
hard_nn<0.6          12 diversity_x_uncertainty_vs_random        mae       0.0933    0.0254     0.1835   0.0270    0.1855              0.0200             0.1259              74          127                76       15
hard_nn<0.6          12 diversity_x_uncertainty_vs_random offset_mae       0.1093    0.0117     0.1958   0.0113    0.1954              0.0183             0.1259              71          127                76       15
hard_nn<0.6          12      offset_uncertainty_vs_random        mae      -0.1181   -0.2116     0.0199  -0.2351   -0.0106             -0.2357            -0.0027              43          127                76       15
hard_nn<0.6          12      offset_uncertainty_vs_random offset_mae      -0.0692   -0.1668     0.0795  -0.1892    0.0507             -0.1918             0.0464              55          127                76       15
hard_nn<0.6          12    same_chemotype_first_vs_random        mae      -0.5042   -0.6957    -0.3703  -0.6767   -0.3556             -0.6675            -0.5651              31          127                76       15
hard_nn<0.6          12    same_chemotype_first_vs_random offset_mae      -0.5785   -0.7681    -0.4284  -0.7593   -0.4218             -0.7385            -0.6078              33          127                76       15
        all          16                  maxmin_vs_random        mae       0.0561   -0.0082     0.1483  -0.0115    0.1431             -0.0209             0.1190              71          131                79       15
        all          16                  maxmin_vs_random offset_mae       0.0285   -0.0500     0.1373  -0.0533    0.1331             -0.0612             0.1028              65          131                79       15
        all          16             uncertainty_vs_random        mae      -0.0174   -0.1522     0.1679  -0.2064    0.1249             -0.1892             0.1490              52          131                79       15
        all          16             uncertainty_vs_random offset_mae       0.0056   -0.1282     0.1895  -0.1778    0.1488             -0.1638             0.1666              61          131                79       15
        all          16 diversity_x_uncertainty_vs_random        mae       0.1085    0.0321     0.2288   0.0233    0.2140              0.0126             0.1976              64          131                79       15
        all          16 diversity_x_uncertainty_vs_random offset_mae       0.1222    0.0353     0.2262   0.0358    0.2265              0.0327             0.1975              70          131                79       15
        all          16      offset_uncertainty_vs_random        mae      -0.0555   -0.1501     0.0727  -0.1742    0.0475             -0.1739             0.0619              49          131                79       15
        all          16      offset_uncertainty_vs_random offset_mae      -0.0169   -0.1007     0.1020  -0.1195    0.0788             -0.1214             0.0803              61          131                79       15
        all          16    same_chemotype_first_vs_random        mae      -0.5365   -0.6819    -0.4067  -0.6819   -0.4067             -0.6645            -0.5852              22          131                79       15
        all          16    same_chemotype_first_vs_random offset_mae      -0.5983   -0.7568    -0.4450  -0.7559   -0.4442             -0.7466            -0.6328              25          131                79       15
hard_nn<0.4          16                  maxmin_vs_random        mae       0.1116    0.0369     0.2313   0.0236    0.2091              0.0140             0.1971              69          119                69       15
hard_nn<0.4          16                  maxmin_vs_random offset_mae       0.1056    0.0190     0.2424   0.0053    0.2171             -0.0045             0.2017              68          119                69       15
hard_nn<0.4          16             uncertainty_vs_random        mae      -0.0834   -0.2212     0.1236  -0.2563    0.0812             -0.2599             0.1064              44          119                69       15
hard_nn<0.4          16             uncertainty_vs_random offset_mae      -0.0747   -0.2151     0.1336  -0.2454    0.0990             -0.2510             0.1209              54          119                69       15
hard_nn<0.4          16 diversity_x_uncertainty_vs_random        mae       0.0985    0.0139     0.2300   0.0052    0.2161             -0.0063             0.1910              61          119                69       15
hard_nn<0.4          16 diversity_x_uncertainty_vs_random offset_mae       0.1287    0.0330     0.2437   0.0318    0.2435              0.0302             0.2025              70          119                69       15
hard_nn<0.4          16      offset_uncertainty_vs_random        mae      -0.0965   -0.1996     0.0613  -0.2190    0.0318             -0.2279             0.0591              45          119                69       15
hard_nn<0.4          16      offset_uncertainty_vs_random offset_mae      -0.0599   -0.1685     0.0972  -0.1880    0.0747             -0.1897             0.0908              58          119                69       15
hard_nn<0.4          16    same_chemotype_first_vs_random        mae      -0.5443   -0.7115    -0.3959  -0.7220   -0.4020             -0.6906            -0.6122              25          119                69       15
hard_nn<0.4          16    same_chemotype_first_vs_random offset_mae      -0.6203   -0.8033    -0.4438  -0.8067   -0.4495             -0.7867            -0.6655              27          119                69       15
hard_nn<0.6          16                  maxmin_vs_random        mae       0.0692    0.0031     0.1703  -0.0023    0.1622             -0.0126             0.1397              71          127                76       15
hard_nn<0.6          16                  maxmin_vs_random offset_mae       0.0529   -0.0294     0.1669  -0.0330    0.1605             -0.0424             0.1359              66          127                76       15
hard_nn<0.6          16             uncertainty_vs_random        mae      -0.0450   -0.1826     0.1531  -0.2262    0.1084             -0.2203             0.1350              50          127                76       15
hard_nn<0.6          16             uncertainty_vs_random offset_mae      -0.0268   -0.1684     0.1773  -0.2052    0.1351             -0.2029             0.1565              58          127                76       15
hard_nn<0.6          16 diversity_x_uncertainty_vs_random        mae       0.1021    0.0229     0.2295   0.0168    0.2154              0.0024             0.1924              63          127                76       15
hard_nn<0.6          16 diversity_x_uncertainty_vs_random offset_mae       0.1271    0.0363     0.2389   0.0375    0.2398              0.0334             0.2034              71          127                76       15
hard_nn<0.6          16      offset_uncertainty_vs_random        mae      -0.0813   -0.1794     0.0641  -0.2053    0.0344             -0.2037             0.0531              48          127                76       15
hard_nn<0.6          16      offset_uncertainty_vs_random offset_mae      -0.0402   -0.1392     0.1045  -0.1619    0.0791             -0.1583             0.0829              62          127                76       15
hard_nn<0.6          16    same_chemotype_first_vs_random        mae      -0.5424   -0.7024    -0.4087  -0.7030   -0.4090             -0.6764            -0.5925              24          127                76       15
hard_nn<0.6          16    same_chemotype_first_vs_random offset_mae      -0.6228   -0.7919    -0.4625  -0.7901   -0.4596             -0.7762            -0.6540              24          127                76       15
        all          20                  maxmin_vs_random        mae       0.0698   -0.0018     0.1733  -0.0028    0.1700             -0.0137             0.1350              72          131                79       15
        all          20                  maxmin_vs_random offset_mae       0.0723   -0.0153     0.1804  -0.0155    0.1795             -0.0209             0.1374              75          131                79       15
        all          20             uncertainty_vs_random        mae       0.0193   -0.1328     0.2210  -0.1926    0.1732             -0.1739             0.1933              52          131                79       15
        all          20             uncertainty_vs_random offset_mae       0.0404   -0.1169     0.2521  -0.1783    0.2022             -0.1584             0.2212              61          131                79       15
        all          20 diversity_x_uncertainty_vs_random        mae       0.1574    0.0735     0.2903   0.0630    0.2757              0.0510             0.2552              70          131                79       15
        all          20 diversity_x_uncertainty_vs_random offset_mae       0.2037    0.1144     0.3217   0.1137    0.3214              0.1045             0.2962              71          131                79       15
        all          20      offset_uncertainty_vs_random        mae       0.0168   -0.0869     0.1600  -0.1198    0.1266             -0.1162             0.1456              56          131                79       15
        all          20      offset_uncertainty_vs_random offset_mae       0.0531   -0.0435     0.1914  -0.0671    0.1626             -0.0687             0.1743              70          131                79       15
        all          20    same_chemotype_first_vs_random        mae      -0.6618   -0.8325    -0.4497  -0.8675   -0.4843             -0.8583            -0.6319              16          131                79       15
        all          20    same_chemotype_first_vs_random offset_mae      -0.7047   -0.9195    -0.4443  -0.9666   -0.4868             -0.9546            -0.6439              24          131                79       15
hard_nn<0.4          20                  maxmin_vs_random        mae       0.1244    0.0403     0.2547   0.0326    0.2431              0.0178             0.2129              68          119                69       15
hard_nn<0.4          20                  maxmin_vs_random offset_mae       0.1477    0.0480     0.2890   0.0441    0.2806              0.0296             0.2337              69          119                69       15
hard_nn<0.4          20             uncertainty_vs_random        mae      -0.0326   -0.1894     0.1918  -0.2290    0.1477             -0.2334             0.1713              43          119                69       15
hard_nn<0.4          20             uncertainty_vs_random offset_mae      -0.0232   -0.1823     0.2114  -0.2219    0.1666             -0.2288             0.1930              54          119                69       15
hard_nn<0.4          20 diversity_x_uncertainty_vs_random        mae       0.1324    0.0377     0.2757   0.0306    0.2634              0.0180             0.2382              62          119                69       15
hard_nn<0.4          20 diversity_x_uncertainty_vs_random offset_mae       0.1888    0.0825     0.3229   0.0805    0.3204              0.0783             0.2815              72          119                69       15
hard_nn<0.4          20      offset_uncertainty_vs_random        mae      -0.0397   -0.1494     0.1295  -0.1737    0.0950             -0.1788             0.1189              47          119                69       15
hard_nn<0.4          20      offset_uncertainty_vs_random offset_mae      -0.0060   -0.1149     0.1610  -0.1356    0.1357             -0.1404             0.1509              60          119                69       15
hard_nn<0.4          20    same_chemotype_first_vs_random        mae      -0.6972   -0.8742    -0.4663  -0.9070   -0.4980             -0.9070            -0.6757              15          119                69       15
hard_nn<0.4          20    same_chemotype_first_vs_random offset_mae      -0.7662   -0.9759    -0.4896  -1.0226   -0.5327             -1.0242            -0.7083              22          119                69       15
hard_nn<0.6          20                  maxmin_vs_random        mae       0.0942    0.0190     0.2037   0.0146    0.1993              0.0023             0.1610              72          127                76       15
hard_nn<0.6          20                  maxmin_vs_random offset_mae       0.1105    0.0172     0.2332   0.0175    0.2332              0.0046             0.1726              74          127                76       15
hard_nn<0.6          20             uncertainty_vs_random        mae       0.0046   -0.1495     0.2183  -0.2044    0.1667             -0.1932             0.1921              51          127                76       15
hard_nn<0.6          20             uncertainty_vs_random offset_mae       0.0182   -0.1401     0.2399  -0.1952    0.1901             -0.1846             0.2138              59          127                76       15
hard_nn<0.6          20 diversity_x_uncertainty_vs_random        mae       0.1484    0.0611     0.2834   0.0523    0.2740              0.0384             0.2483              69          127                76       15
hard_nn<0.6          20 diversity_x_uncertainty_vs_random offset_mae       0.2035    0.1080     0.3318   0.1052    0.3276              0.1001             0.2954              75          127                76       15
hard_nn<0.6          20      offset_uncertainty_vs_random        mae      -0.0107   -0.1180     0.1498  -0.1468    0.1147             -0.1468             0.1343              55          127                76       15
hard_nn<0.6          20      offset_uncertainty_vs_random offset_mae       0.0297   -0.0762     0.1894  -0.0993    0.1602             -0.1019             0.1736              69          127                76       15
hard_nn<0.6          20    same_chemotype_first_vs_random        mae      -0.6878   -0.8523    -0.4742  -0.8763   -0.5019             -0.8835            -0.6615              15          127                76       15
hard_nn<0.6          20    same_chemotype_first_vs_random offset_mae      -0.7561   -0.9578    -0.4930  -0.9967   -0.5372             -0.9986            -0.7019              21          127                76       15
        all          25                  maxmin_vs_random        mae       0.1140    0.0255     0.1975   0.0289    0.2001              0.0302             0.1436              77          131                79       15
        all          25                  maxmin_vs_random offset_mae       0.0974   -0.0097     0.1923  -0.0084    0.1949             -0.0031             0.1201              74          131                79       15
        all          25             uncertainty_vs_random        mae       0.0989    0.0026     0.2435  -0.0152    0.2127             -0.0234             0.2203              59          131                79       15
        all          25             uncertainty_vs_random offset_mae       0.1323    0.0282     0.2872   0.0020    0.2540             -0.0005             0.2622              70          131                79       15
        all          25 diversity_x_uncertainty_vs_random        mae       0.1763    0.0980     0.2851   0.0997    0.2898              0.0879             0.2580              68          131                79       15
        all          25 diversity_x_uncertainty_vs_random offset_mae       0.2117    0.1131     0.3133   0.1104    0.3104              0.1164             0.2848              77          131                79       15
        all          25      offset_uncertainty_vs_random        mae       0.0671   -0.0152     0.1875  -0.0278    0.1669             -0.0371             0.1683              59          131                79       15
        all          25      offset_uncertainty_vs_random offset_mae       0.0858    0.0033     0.2092  -0.0085    0.1887             -0.0173             0.1832              75          131                79       15
        all          25    same_chemotype_first_vs_random        mae      -0.7523   -0.9582    -0.4944  -1.0064   -0.5381             -0.9924            -0.6984              19          131                79       15
        all          25    same_chemotype_first_vs_random offset_mae      -0.7797   -1.0498    -0.4508  -1.1236   -0.5103             -1.1015            -0.6858              25          131                79       15
hard_nn<0.4          25                  maxmin_vs_random        mae       0.1978    0.1177     0.2993   0.1217    0.3060              0.1131             0.2421              74          119                69       15
hard_nn<0.4          25                  maxmin_vs_random offset_mae       0.2049    0.1032     0.3231   0.1075    0.3286              0.1023             0.2389              76          119                69       15
hard_nn<0.4          25             uncertainty_vs_random        mae       0.0906   -0.0127     0.2456  -0.0340    0.2216             -0.0418             0.2223              50          119                69       15
hard_nn<0.4          25             uncertainty_vs_random offset_mae       0.1155    0.0069     0.2822  -0.0167    0.2520             -0.0259             0.2558              61          119                69       15
hard_nn<0.4          25 diversity_x_uncertainty_vs_random        mae       0.1825    0.0937     0.3059   0.0921    0.3041              0.0834             0.2719              64          119                69       15
hard_nn<0.4          25 diversity_x_uncertainty_vs_random offset_mae       0.2334    0.1255     0.3525   0.1219    0.3482              0.1285             0.3087              75          119                69       15
hard_nn<0.4          25      offset_uncertainty_vs_random        mae       0.0603   -0.0286     0.2025  -0.0446    0.1764             -0.0546             0.1812              51          119                69       15
hard_nn<0.4          25      offset_uncertainty_vs_random offset_mae       0.0831   -0.0080     0.2264  -0.0200    0.2032             -0.0320             0.2000              64          119                69       15
hard_nn<0.4          25    same_chemotype_first_vs_random        mae      -0.8144   -1.0226    -0.5317  -1.0616   -0.5723             -1.0672            -0.7602              20          119                69       15
hard_nn<0.4          25    same_chemotype_first_vs_random offset_mae      -0.9220   -1.1684    -0.5843  -1.2257   -0.6400             -1.2306            -0.8357              21          119                69       15
hard_nn<0.6          25                  maxmin_vs_random        mae       0.1568    0.0718     0.2455   0.0757    0.2493              0.0734             0.1821              77          127                76       15
hard_nn<0.6          25                  maxmin_vs_random offset_mae       0.1602    0.0530     0.2667   0.0548    0.2685              0.0560             0.1769              78          127                76       15
hard_nn<0.6          25             uncertainty_vs_random        mae       0.1164    0.0180     0.2658  -0.0082    0.2360             -0.0137             0.2396              57          127                76       15
hard_nn<0.6          25             uncertainty_vs_random offset_mae       0.1497    0.0413     0.3098   0.0143    0.2775              0.0092             0.2820              68          127                76       15
hard_nn<0.6          25 diversity_x_uncertainty_vs_random        mae       0.1877    0.1088     0.3020   0.1101    0.3049              0.0959             0.2694              68          127                76       15
hard_nn<0.6          25 diversity_x_uncertainty_vs_random offset_mae       0.2334    0.1386     0.3422   0.1380    0.3400              0.1373             0.3026              78          127                76       15
hard_nn<0.6          25      offset_uncertainty_vs_random        mae       0.0720   -0.0122     0.2022  -0.0292    0.1807             -0.0373             0.1824              59          127                76       15
hard_nn<0.6          25      offset_uncertainty_vs_random offset_mae       0.0961    0.0096     0.2258  -0.0052    0.2062             -0.0138             0.2038              72          127                76       15
hard_nn<0.6          25    same_chemotype_first_vs_random        mae      -0.7726   -0.9745    -0.5130  -1.0150   -0.5559             -1.0130            -0.7188              20          127                76       15
hard_nn<0.6          25    same_chemotype_first_vs_random offset_mae      -0.8316   -1.0942    -0.5004  -1.1479   -0.5662             -1.1452            -0.7484              24          127                76       15
        all          30                  maxmin_vs_random        mae       0.0677   -0.0201     0.1525  -0.0214    0.1516             -0.0170             0.0971              69          131                79       15
        all          30                  maxmin_vs_random offset_mae       0.0487   -0.0498     0.1426  -0.0534    0.1392             -0.0460             0.0855              72          131                79       15
        all          30             uncertainty_vs_random        mae       0.1177    0.0378     0.2370   0.0208    0.2111              0.0162             0.2026              63          131                79       15
        all          30             uncertainty_vs_random offset_mae       0.1295    0.0416     0.2594   0.0237    0.2329              0.0200             0.2300              74          131                79       15
        all          30 diversity_x_uncertainty_vs_random        mae       0.1420    0.0683     0.2503   0.0642    0.2436              0.0533             0.2252              67          131                79       15
        all          30 diversity_x_uncertainty_vs_random offset_mae       0.1448    0.0644     0.2544   0.0616    0.2490              0.0533             0.2354              78          131                79       15
        all          30      offset_uncertainty_vs_random        mae       0.0708   -0.0168     0.1975  -0.0346    0.1713             -0.0390             0.1690              64          131                79       15
        all          30      offset_uncertainty_vs_random offset_mae       0.0928    0.0063     0.2219  -0.0092    0.1988             -0.0143             0.1949              72          131                79       15
        all          30    same_chemotype_first_vs_random        mae      -0.6068   -0.7805    -0.4408  -0.7804   -0.4408             -0.7658            -0.6603              25          131                79       15
        all          30    same_chemotype_first_vs_random offset_mae      -0.6515   -0.8437    -0.4436  -0.8419   -0.4419             -0.8392            -0.6854              26          131                79       15
hard_nn<0.4          30                  maxmin_vs_random        mae       0.1342    0.0462     0.2356   0.0460    0.2355              0.0453             0.1855              70          119                69       15
hard_nn<0.4          30                  maxmin_vs_random offset_mae       0.1174    0.0204     0.2306   0.0161    0.2251              0.0184             0.1726              70          119                69       15
hard_nn<0.4          30             uncertainty_vs_random        mae       0.1225    0.0325     0.2569   0.0152    0.2346              0.0068             0.2194              57          119                69       15
hard_nn<0.4          30             uncertainty_vs_random offset_mae       0.1228    0.0284     0.2674   0.0130    0.2413              0.0014             0.2321              65          119                69       15
hard_nn<0.4          30 diversity_x_uncertainty_vs_random        mae       0.1712    0.0882     0.3003   0.0799    0.2846              0.0670             0.2663              67          119                69       15
hard_nn<0.4          30 diversity_x_uncertainty_vs_random offset_mae       0.1871    0.0985     0.3199   0.0911    0.3087              0.0811             0.2873              73          119                69       15
hard_nn<0.4          30      offset_uncertainty_vs_random        mae       0.0828   -0.0147     0.2345  -0.0377    0.2053             -0.0450             0.2076              59          119                69       15
hard_nn<0.4          30      offset_uncertainty_vs_random offset_mae       0.0999    0.0062     0.2477  -0.0118    0.2227             -0.0218             0.2248              66          119                69       15
hard_nn<0.4          30    same_chemotype_first_vs_random        mae      -0.6386   -0.8471    -0.4511  -0.8476   -0.4511             -0.8205            -0.6953              26          119                69       15
hard_nn<0.4          30    same_chemotype_first_vs_random offset_mae      -0.7271   -0.9464    -0.5036  -0.9447   -0.5025             -0.9330            -0.7537              26          119                69       15
hard_nn<0.6          30                  maxmin_vs_random        mae       0.0947    0.0132     0.1789   0.0101    0.1775              0.0127             0.1254              71          127                76       15
hard_nn<0.6          30                  maxmin_vs_random offset_mae       0.0809   -0.0113     0.1781  -0.0120    0.1777             -0.0114             0.1180              74          127                76       15
hard_nn<0.6          30             uncertainty_vs_random        mae       0.1326    0.0489     0.2550   0.0295    0.2311              0.0244             0.2197              65          127                76       15
hard_nn<0.6          30             uncertainty_vs_random offset_mae       0.1426    0.0533     0.2783   0.0341    0.2526              0.0268             0.2439              73          127                76       15
hard_nn<0.6          30 diversity_x_uncertainty_vs_random        mae       0.1557    0.0813     0.2693   0.0758    0.2596              0.0632             0.2389              69          127                76       15
hard_nn<0.6          30 diversity_x_uncertainty_vs_random offset_mae       0.1718    0.0899     0.2883   0.0865    0.2830              0.0766             0.2603              78          127                76       15
hard_nn<0.6          30      offset_uncertainty_vs_random        mae       0.0724   -0.0155     0.2056  -0.0386    0.1824             -0.0424             0.1798              63          127                76       15
hard_nn<0.6          30      offset_uncertainty_vs_random offset_mae       0.0949    0.0077     0.2294  -0.0100    0.2075             -0.0165             0.2053              71          127                76       15
hard_nn<0.6          30    same_chemotype_first_vs_random        mae      -0.6209   -0.8012    -0.4555  -0.7957   -0.4500             -0.7839            -0.6755              26          127                76       15
hard_nn<0.6          30    same_chemotype_first_vs_random offset_mae      -0.6848   -0.8842    -0.4773  -0.8784   -0.4715             -0.8746            -0.7251              26          127                76       15
```

The descriptive contrast against the chemistry-only rule (`vs_maxmin|…`) answers the protocol's own falsifier for F2 — whether the model's uncertainty carries anything beyond chemical distance — and is quoted inside F2's evidence above.

## Disclosed sensitivity — buying every row of the acquired ligand

`maxmin@ball`, `random@ball` repeat two policies with `budget = all rows of the acquired ligand` instead of 3. It is a different experiment (each step buys ~30 rows rather than 3), reported so the b = 3 result cannot be read as a claim about all budgets. It is not a protected comparison.

```
        arm  checkpoint  n_train_rows_mean  macro_mae_mean  hard_nn0.4__macro_mae_mean  offset_mae_mean  n_runs
maxmin@ball           0          2384.2667          1.8519                      1.9752           1.7565      15
maxmin@ball           1          2389.7333          1.6742                      1.7861           1.5956      15
maxmin@ball           2          2399.2000          1.5886                      1.6299           1.4742      15
maxmin@ball           3          2402.8000          1.5864                      1.6079           1.4537      15
maxmin@ball           5          2429.1333          1.2899                      1.2698           1.1385      15
maxmin@ball           8          2452.3333          1.2934                      1.2728           1.1499      15
maxmin@ball          12          2509.4000          1.2439                      1.1810           1.1022      15
maxmin@ball          16          2545.5333          1.2151                      1.1359           1.0840      15
maxmin@ball          20          2586.1333          1.1864                      1.1059           1.0509      15
maxmin@ball          25          2655.3333          1.1570                      1.0939           1.0099      15
maxmin@ball          30          2774.3333          1.1753                      1.1296           1.0198      15
random@ball           0          2384.2667          1.8519                      1.9752           1.7565      15
random@ball           1          2396.9333          1.6968                      1.7972           1.5764      15
random@ball           2          2410.6000          1.6079                      1.6997           1.4790      15
random@ball           3          2426.3333          1.5610                      1.6000           1.4484      15
random@ball           5          2465.9333          1.4934                      1.5580           1.3737      15
random@ball           8          2499.4667          1.4486                      1.5093           1.3397      15
random@ball          12          2556.8667          1.3830                      1.4292           1.2755      15
random@ball          16          2604.7333          1.3992                      1.4677           1.2825      15
random@ball          20          2654.4667          1.2907                      1.3338           1.1439      15
random@ball          25          2763.7333          1.3123                      1.3461           1.1571      15
random@ball          30          2828.2667          1.3093                      1.3537           1.1510      15
```

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

