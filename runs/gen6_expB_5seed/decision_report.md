# gen6 Experiment B — diversity versus depth at equal row budget (20260819T103337Z)

**Question.** Experiment A can be dismissed with "you just gave it more rows". Here every arm gets the *same* number of training rows and only the acquisition policy differs, so a difference can only come from *which chemistry* the budget was spent on.

## What was run

* shared cohort (min_cells = 3): **5248 rows**, 152 extractants, 131 ECFP clusters, 79 Tanimoto-0.7 super-clusters, 14 metals; target sd 1.657
* frozen chemistry map over 190 extractants (164 ECFP clusters, 98 super-clusters, threshold 0.7)
* learner: ExtraTrees, 400 trees, max_features 0.3, min_samples_leaf 2; feature set `MC_lig2d_ext_massaction` (281 columns) — frozen, exactly as gen5 left it
* folds: 5 held-out Tanimoto-0.7 super-clusters, gen5 fold algorithm, split seeds [104729, 130363, 155921, 196613, 262147]
* policies ['depth', 'diversity', 'random', 'maxmin']; budgets ['250', '500', '1000', '2000', '3000', 'all']; 3 acquisition draw(s) per point
* every metric below is out-of-fold over the whole cohort (5248 rows, 152 ligands, 131 ECFP clusters, n_eff(pooled rows) 6.3)

## How to read this

* **macro MAE** weights one ECFP cluster as one vote (the pre-registered primary metric). Pooled MAE is reported in the CSV but never used for selection: the largest ligand is 28 % of rows.
* **offset vs shape**: `offset_mae` is the per-ligand level error (the failure six generations could not fix), `shape_mae` the within-ligand response shape. They are orthogonal components of the same squared error.
* **two hard-chemistry readings, deliberately**. `hard_own_train_nn<t>__*` measures the test ligand's distance to *that point's own* training set — the honest description of what that model saw, but the row subset moves with the point, so those numbers may not be subtracted between policies. `hard_vs_BASEtrain_nn<t>__*` uses the frozen BASE (>= 10 cells) bin, identical rows for every point; **only that one is used for the paired contrasts and for B2**.
* a **draw** is one realisation of a policy. Draws are nested across budgets within a draw (the acquisition RNG does not see the budget), so the curve's slope is not sampling noise.
* at budget `all` every policy and draw is the same training set; those rows are printed for completeness and are an internal consistency check (`all_budget_points_identical` = True).

## What a budget buys in chemistry (mean over folds, draws and seeds)

**extractants in training**

```
policy   depth  diversity  random  maxmin
budget                                   
250       1.32      13.80   13.64   21.32
500       2.12      22.67   23.16   34.28
1000      7.56      33.69   40.16   57.37
2000     20.48      48.41   66.48   96.17
3000     24.68      73.57   81.76  110.55
all     121.60     121.60  121.60  121.60
```

**Tanimoto-0.7 super-clusters in training**

```
policy  depth  diversity  random  maxmin
budget                                  
250      1.32      13.80   10.80   21.32
500      2.04      22.67   16.85   34.11
1000     5.61      33.41   26.69   52.53
2000    13.40      42.93   39.69   62.76
3000    15.12      63.20   46.77   63.20
all     63.20      63.20   63.20   63.20
```

**ECFP clusters in training**

```
policy   depth  diversity  random  maxmin
budget                                   
250       1.32      13.80   13.37   21.32
500       2.04      22.67   22.37   34.28
1000      6.97      33.59   37.71   57.37
2000     19.04      46.97   60.21   94.73
3000     21.64      70.65   72.88  104.52
all     104.80     104.80  104.80  104.80
```

This table is the mechanism, not decoration: if `diversity` and `depth` buy the same chemistry at a budget, no difference in accuracy is expected there.

## Learning curve — macro MAE (one ECFP cluster = one vote; lower is better)

```
policy   depth  diversity  random  maxmin
budget                                   
250     1.8577     1.3416  1.3421  1.2580
500     1.7181     1.2649  1.3055  1.1395
1000    1.6716     1.2260  1.2165  1.1188
2000    1.7439     1.1584  1.1594  1.0537
3000    1.3976     1.0575  1.1166  1.0417
all     1.0468     1.0468  1.0468  1.0468
```

Spread over the 3 acquisition draw(s) x 5 split seed(s) (sd of macro MAE):

```
policy   depth  diversity  random  maxmin
budget                                   
250     0.1544     0.0811  0.0954  0.0805
500     0.0413     0.0673  0.0852  0.0447
1000    0.0466     0.0792  0.0595  0.0432
2000    0.0595     0.0433  0.0411  0.0199
3000    0.0357     0.0192  0.0359  0.0216
all     0.0135     0.0135  0.0135  0.0135
```

**offset MAE (per-ligand level error)**

```
policy   depth  diversity  random  maxmin
budget                                   
250     1.7213     1.1784  1.1651  1.0948
500     1.5314     1.1002  1.1296  0.9638
1000    1.4803     1.0710  1.0359  0.9699
2000    1.5517     0.9962  0.9807  0.8839
3000    1.2109     0.8977  0.9422  0.8689
all     0.8744     0.8744  0.8744  0.8744
```

**shape MAE (within-ligand response shape)**

```
policy   depth  diversity  random  maxmin
budget                                   
250     0.6339     0.5741  0.5785  0.5903
500     0.6314     0.5569  0.5636  0.5649
1000    0.6381     0.5449  0.5416  0.5299
2000    0.6087     0.5326  0.5269  0.5104
3000    0.5730     0.5145  0.5210  0.5060
all     0.5069     0.5069  0.5069  0.5069
```

**median ligand MAE**

```
policy   depth  diversity  random  maxmin
budget                                   
250     1.5976     1.1922  1.1847  1.1477
500     1.3587     1.0845  1.1393  1.0143
1000    1.3342     1.0417  1.0544  0.9836
2000    1.3286     0.9823  0.9958  0.9027
3000    1.1508     0.9066  0.9636  0.8947
all     0.9027     0.9027  0.9027  0.9027
```

**worst-quartile ligand MAE**

```
policy   depth  diversity  random  maxmin
budget                                   
250     3.4294     2.5059  2.4853  2.4000
500     3.2608     2.4507  2.4715  2.1540
1000    3.1920     2.4209  2.3313  2.1644
2000    3.3728     2.2928  2.2559  2.1015
3000    2.6988     2.1405  2.1893  2.0832
all     2.0934     2.0934  2.0934  2.0934
```

**macro MAE on hard chemistry (nn to BASE training < 0.4; identical rows for every policy)**

```
policy   depth  diversity  random  maxmin
budget                                   
250     2.2116     1.4812  1.5508  1.1561
500     2.1787     1.3517  1.4901  1.0700
1000    2.1728     1.2983  1.3548  1.0984
2000    2.1046     1.1805  1.2465  1.0431
3000    1.6469     0.9734  1.1800  1.0250
all     1.0429     1.0429  1.0429  1.0429
```

**macro MAE on rows far from THIS POINT'S OWN training set (nn < 0.4; the row subset moves with the point — descriptive only)**

```
policy   depth  diversity  random  maxmin
budget                                   
250     1.9514     1.3776  1.4205  1.2994
500     1.8547     1.3041  1.3989  1.1335
1000    1.9205     1.2850  1.3088  1.0632
2000    2.1053     1.1714  1.2890  1.0133
3000    1.5232     0.9069  1.1912  0.9935
all     1.0241     1.0241  1.0241  1.0241
```

## Paired bootstrap over independent chemistry units

Scoring unit = ECFP cluster; resampling unit = Tanimoto-0.7 super-cluster (the block the folds actually held out), 5000 replicates, one shared index matrix for every comparison. Positive delta = the candidate policy is better than the reference. The per-unit error of a policy is averaged over draws and split seeds first, because a policy is a distribution over training sets.

```
                endpoint                   comparison  statistic  point_delta  ci95_low  ci95_high  p_worse_one_sided  units_improved  units_total  bootstrap_blocks
                 overall       diversity_vs_depth@250        mae       0.5161    0.3778     0.6877             0.0002             102          131                79
                 overall       diversity_vs_depth@250 offset_mae       0.5602    0.4102     0.7381             0.0002             106          131                79
                 overall      diversity_vs_random@250        mae       0.0006   -0.0362     0.0481             0.4859              58          131                79
                 overall      diversity_vs_random@250 offset_mae      -0.0107   -0.0516     0.0377             0.6945              60          131                79
                 overall          maxmin_vs_depth@250        mae       0.5997    0.4173     0.8535             0.0002             105          131                79
                 overall          maxmin_vs_depth@250 offset_mae       0.6603    0.4755     0.9290             0.0002             105          131                79
                 overall       diversity_vs_depth@500        mae       0.4532    0.2797     0.6968             0.0002              97          131                79
                 overall       diversity_vs_depth@500 offset_mae       0.4841    0.3037     0.7389             0.0002              98          131                79
                 overall      diversity_vs_random@500        mae       0.0406   -0.0006     0.0842             0.0268              74          131                79
                 overall      diversity_vs_random@500 offset_mae       0.0295   -0.0155     0.0784             0.0988              69          131                79
                 overall          maxmin_vs_depth@500        mae       0.5785    0.3583     0.8862             0.0002              92          131                79
                 overall          maxmin_vs_depth@500 offset_mae       0.6271    0.4042     0.9497             0.0002             103          131                79
                 overall      diversity_vs_depth@1000        mae       0.4456    0.2273     0.7396             0.0002              84          131                79
                 overall      diversity_vs_depth@1000 offset_mae       0.4673    0.2367     0.7799             0.0002              87          131                79
                 overall     diversity_vs_random@1000        mae      -0.0095   -0.0513     0.0246             0.6887              64          131                79
                 overall     diversity_vs_random@1000 offset_mae      -0.0271   -0.0757     0.0109             0.9112              65          131                79
                 overall         maxmin_vs_depth@1000        mae       0.5528    0.2720     0.9185             0.0002              85          131                79
                 overall         maxmin_vs_depth@1000 offset_mae       0.5747    0.2826     0.9613             0.0002              85          131                79
                 overall      diversity_vs_depth@2000        mae       0.5855    0.3462     0.9050             0.0002              78          131                79
                 overall      diversity_vs_depth@2000 offset_mae       0.6299    0.3736     0.9764             0.0002              80          131                79
                 overall     diversity_vs_random@2000        mae       0.0011   -0.0304     0.0317             0.4581              53          131                79
                 overall     diversity_vs_random@2000 offset_mae      -0.0029   -0.0386     0.0316             0.5563              54          131                79
                 overall         maxmin_vs_depth@2000        mae       0.6902    0.4154     1.0416             0.0002              81          131                79
                 overall         maxmin_vs_depth@2000 offset_mae       0.7411    0.4466     1.1215             0.0002              84          131                79
                 overall      diversity_vs_depth@3000        mae       0.3401    0.1876     0.5636             0.0002              77          131                79
                 overall      diversity_vs_depth@3000 offset_mae       0.3528    0.1891     0.5926             0.0002              71          131                79
                 overall     diversity_vs_random@3000        mae       0.0591    0.0242     0.1108             0.0002              63          131                79
                 overall     diversity_vs_random@3000 offset_mae       0.0583    0.0209     0.1126             0.0006              57          131                79
                 overall         maxmin_vs_depth@3000        mae       0.3559    0.2042     0.5746             0.0002              80          131                79
                 overall         maxmin_vs_depth@3000 offset_mae       0.3726    0.2098     0.6069             0.0002              77          131                79
                 overall       diversity_vs_depth@all        mae       0.0000    0.0000     0.0000             1.0000               0          131                79
                 overall       diversity_vs_depth@all offset_mae       0.0000    0.0000     0.0000             1.0000               0          131                79
                 overall      diversity_vs_random@all        mae       0.0000    0.0000     0.0000             1.0000               0          131                79
                 overall      diversity_vs_random@all offset_mae       0.0000    0.0000     0.0000             1.0000               0          131                79
                 overall          maxmin_vs_depth@all        mae       0.0000    0.0000     0.0000             1.0000               0          131                79
                 overall          maxmin_vs_depth@all offset_mae       0.0000    0.0000     0.0000             1.0000               0          131                79
                 overall      depth_slope_3000_to_all        mae       0.3508    0.1996     0.5667             0.0002              82          131                79
                 overall      depth_slope_3000_to_all offset_mae       0.3690    0.2056     0.6005             0.0002              74          131                79
                 overall  diversity_slope_3000_to_all        mae       0.0107   -0.0203     0.0415             0.2238              54          131                79
                 overall  diversity_slope_3000_to_all offset_mae       0.0162   -0.0191     0.0536             0.1648              60          131                79
                 overall     depth_slope_2000_to_3000        mae       0.3463    0.1954     0.5424             0.0002              84          131                79
                 overall     depth_slope_2000_to_3000 offset_mae       0.3821    0.2171     0.6054             0.0002              81          131                79
                 overall diversity_slope_2000_to_3000        mae       0.1009    0.0580     0.1647             0.0002              83          131                79
                 overall diversity_slope_2000_to_3000 offset_mae       0.1050    0.0591     0.1717             0.0002              79          131                79
hard_vs_BASEtrain_nn<0.4       diversity_vs_depth@250        mae       0.7095    0.3663     1.0814             0.0002              29           42                37
hard_vs_BASEtrain_nn<0.4       diversity_vs_depth@250 offset_mae       0.6628    0.2868     1.0672             0.0004              29           42                37
hard_vs_BASEtrain_nn<0.4      diversity_vs_random@250        mae       0.0371   -0.0490     0.1425             0.2098              20           42                37
hard_vs_BASEtrain_nn<0.4      diversity_vs_random@250 offset_mae       0.0031   -0.0977     0.1182             0.4707              20           42                37
hard_vs_BASEtrain_nn<0.4          maxmin_vs_depth@250        mae       0.9968    0.6128     1.4321             0.0002              33           42                37
hard_vs_BASEtrain_nn<0.4          maxmin_vs_depth@250 offset_mae       0.9648    0.5444     1.4336             0.0002              33           42                37
hard_vs_BASEtrain_nn<0.4       diversity_vs_depth@500        mae       0.7941    0.4539     1.1628             0.0002              32           42                37
hard_vs_BASEtrain_nn<0.4       diversity_vs_depth@500 offset_mae       0.7795    0.4135     1.1758             0.0004              29           42                37
hard_vs_BASEtrain_nn<0.4      diversity_vs_random@500        mae       0.1307    0.0373     0.2306             0.0024              28           42                37
hard_vs_BASEtrain_nn<0.4      diversity_vs_random@500 offset_mae       0.1206    0.0178     0.2301             0.0088              28           42                37
hard_vs_BASEtrain_nn<0.4          maxmin_vs_depth@500        mae       1.0275    0.6396     1.4712             0.0002              31           42                37
hard_vs_BASEtrain_nn<0.4          maxmin_vs_depth@500 offset_mae       1.0121    0.5859     1.4912             0.0002              32           42                37
hard_vs_BASEtrain_nn<0.4      diversity_vs_depth@1000        mae       0.8358    0.4889     1.2156             0.0002              30           42                37
hard_vs_BASEtrain_nn<0.4      diversity_vs_depth@1000 offset_mae       0.8331    0.4622     1.2372             0.0004              30           42                37
hard_vs_BASEtrain_nn<0.4     diversity_vs_random@1000        mae       0.0601   -0.0045     0.1351             0.0326              23           42                37
hard_vs_BASEtrain_nn<0.4     diversity_vs_random@1000 offset_mae       0.0520   -0.0233     0.1380             0.0944              24           42                37
hard_vs_BASEtrain_nn<0.4         maxmin_vs_depth@1000        mae       1.0155    0.6367     1.4329             0.0002              34           42                37
hard_vs_BASEtrain_nn<0.4         maxmin_vs_depth@1000 offset_mae       1.0186    0.6112     1.4619             0.0002              31           42                37
hard_vs_BASEtrain_nn<0.4      diversity_vs_depth@2000        mae       0.8698    0.4784     1.2877             0.0002              30           42                37
hard_vs_BASEtrain_nn<0.4      diversity_vs_depth@2000 offset_mae       0.8653    0.4425     1.3073             0.0002              29           42                37
hard_vs_BASEtrain_nn<0.4     diversity_vs_random@2000        mae       0.0578   -0.0036     0.1252             0.0350              22           42                37
hard_vs_BASEtrain_nn<0.4     diversity_vs_random@2000 offset_mae       0.0538   -0.0129     0.1278             0.0572              23           42                37
hard_vs_BASEtrain_nn<0.4         maxmin_vs_depth@2000        mae       0.9980    0.6099     1.4133             0.0002              30           42                37
hard_vs_BASEtrain_nn<0.4         maxmin_vs_depth@2000 offset_mae       0.9997    0.5753     1.4493             0.0002              30           42                37
hard_vs_BASEtrain_nn<0.4      diversity_vs_depth@3000        mae       0.6432    0.3733     0.9144             0.0002              35           42                37
hard_vs_BASEtrain_nn<0.4      diversity_vs_depth@3000 offset_mae       0.6230    0.3314     0.9138             0.0002              30           42                37
hard_vs_BASEtrain_nn<0.4     diversity_vs_random@3000        mae       0.1968    0.1249     0.2775             0.0002              32           42                37
hard_vs_BASEtrain_nn<0.4     diversity_vs_random@3000 offset_mae       0.2077    0.1275     0.3010             0.0002              31           42                37
hard_vs_BASEtrain_nn<0.4         maxmin_vs_depth@3000        mae       0.5933    0.3421     0.8468             0.0002              34           42                37
hard_vs_BASEtrain_nn<0.4         maxmin_vs_depth@3000 offset_mae       0.5824    0.3157     0.8510             0.0002              30           42                37
hard_vs_BASEtrain_nn<0.4       diversity_vs_depth@all        mae       0.0000    0.0000     0.0000             1.0000               0           42                37
hard_vs_BASEtrain_nn<0.4       diversity_vs_depth@all offset_mae       0.0000    0.0000     0.0000             1.0000               0           42                37
hard_vs_BASEtrain_nn<0.4      diversity_vs_random@all        mae       0.0000    0.0000     0.0000             1.0000               0           42                37
hard_vs_BASEtrain_nn<0.4      diversity_vs_random@all offset_mae       0.0000    0.0000     0.0000             1.0000               0           42                37
hard_vs_BASEtrain_nn<0.4          maxmin_vs_depth@all        mae       0.0000    0.0000     0.0000             1.0000               0           42                37
hard_vs_BASEtrain_nn<0.4          maxmin_vs_depth@all offset_mae       0.0000    0.0000     0.0000             1.0000               0           42                37
hard_vs_BASEtrain_nn<0.4      depth_slope_3000_to_all        mae       0.5801    0.3286     0.8289             0.0002              34           42                37
hard_vs_BASEtrain_nn<0.4      depth_slope_3000_to_all offset_mae       0.5619    0.2934     0.8307             0.0002              29           42                37
hard_vs_BASEtrain_nn<0.4  diversity_slope_3000_to_all        mae      -0.0632   -0.1373    -0.0029             0.9800              11           42                37
hard_vs_BASEtrain_nn<0.4  diversity_slope_3000_to_all offset_mae      -0.0611   -0.1441     0.0088             0.9570              13           42                37
hard_vs_BASEtrain_nn<0.4     depth_slope_2000_to_3000        mae       0.4258    0.1518     0.7023             0.0004              31           42                37
hard_vs_BASEtrain_nn<0.4     depth_slope_2000_to_3000 offset_mae       0.4463    0.1410     0.7696             0.0014              31           42                37
hard_vs_BASEtrain_nn<0.4 diversity_slope_2000_to_3000        mae       0.1992    0.1239     0.2823             0.0002              37           42                37
hard_vs_BASEtrain_nn<0.4 diversity_slope_2000_to_3000 offset_mae       0.2039    0.1279     0.2922             0.0002              36           42                37
```

## Pre-registered hypotheses (protocol §5)

**B1 — at equal row budget, DIVERSITY beats DEPTH**  →  `PASS`

* pass condition: CI95 low > 0 at >= 2 consecutive budgets
* budgets with CI95 low > 0: ['250', '500', '1000', '2000', '3000']; longest consecutive run: ['250', '500', '1000', '2000', '3000']

**B2 — the DIVERSITY advantage is larger on hard chemistry than overall**  →  `PASS`

* pass condition: gain at hard_vs_BASEtrain_nn<0.4 > gain overall, both CI95 low > 0
* budgets passing: ['250', '500', '1000', '2000', '3000']

Per budget, with the mechanism and the required null beside the verdict (gains are macro MAE, positive = DIVERSITY better):

```
budget  extractants_bought_depth  extractants_bought_diversity  gain_overall  ci95_low_overall  gain_hard  ci95_low_hard  gain_vs_random  ci95_low_vs_random  beats_random  passes_b1  passes_b2
   250                      1.32                       13.8000        0.5161            0.3778     0.7095         0.3663          0.0006             -0.0362         False       True       True
   500                      2.12                       22.6667        0.4532            0.2797     0.7941         0.4539          0.0406             -0.0006         False       True       True
  1000                      7.56                       33.6933        0.4456            0.2273     0.8358         0.4889         -0.0095             -0.0513         False       True       True
  2000                     20.48                       48.4133        0.5855            0.3462     0.8698         0.4784          0.0011             -0.0304         False       True       True
  3000                     24.68                       73.5733        0.3401            0.1876     0.6432         0.3733          0.0591              0.0242          True       True       True
   all                    121.60                      121.6000        0.0000            0.0000     0.0000         0.0000          0.0000              0.0000         False      False      False
```

**Read the `vs_random` column before celebrating.** Protocol §6 requires an acquisition claim to beat *random acquisition* as well as the incumbent heuristic. Beating DEPTH at a small budget is close to tautological — DEPTH spends the whole budget on the few best-measured ligands (see the extractants-bought columns), so it is a one-ligand model. The load-bearing question is whether DIVERSITY beats RANDOM at the same budget.

**B3 — the DEPTH curve slope between the two largest budgets is within noise of zero**  →  `FAIL (still improving)`

* budgets compared: ['3000', 'all']; change in macro MAE 0.3508 (CI95 [0.1996, 0.5667])
* "within noise of zero" is read as: the 95 % interval of the change contains zero.
* disclosed sensitivity — `all` is a **fold-dependent** budget (the fold holding the giant super-cluster trains on far fewer rows than the others), so the same slope between the two largest *fixed* budgets ['2000', '3000'] is 0.3463 (CI95 [0.1954, 0.5424]) → `FAIL (still improving)`. Only the pre-registered reading above is scored.

### What would falsify these claims

* **B1** dies if `diversity` and `random` are indistinguishable — that would say the budget, not its allocation, is what matters. That contrast is in the bootstrap table above.
* **B1** also dies if the advantage disappears once the per-unit errors are averaged over draws: a single lucky acquisition draw is not a policy.
* **B2** dies if the gain is flat in nearest-neighbour distance; then the policy is buying general regularisation, not coverage of new chemistry.
* **B3** dies if the depth curve is still falling between the two largest budgets, in which case the comparison is budget-limited and the whole curve should be re-run further out.

## Caveats recorded by this run

* **120 point-folds could not spend their budget** (5 distinct (seed, fold) combinations, pools [1495, 1436, 1548, 1486, 1199] rows) because the fold's whole training pool was smaller: {'2000': 60, '3000': 60} point-folds per budget. In those folds every policy trains on the identical pool, so the contrast is structurally zero there and the reported gain at those budgets is diluted towards zero. The largest super-cluster is 64 % of rows, so the fold that holds it out has by far the smallest pool.
* label-free acquisition was verified at run time on the real cohort (`label_free_selection` = True): permuting `log_D` moved none of the 72 selections.
* split integrity (identical test rows, no extractant / ECFP / super-cluster shared between a fold's test and training rows): True.
* the acquisition policies operate on **whole ligands** (a ligand enters, then the next), with only the ligand that straddles the budget contributing a random row subset. A policy that could pick individual rows would be a different experiment.
* this run fixes the feature set to one family; a policy that helps only the champion features would be an artefact of the representation, so a second family (`MC_donors_massaction`) is a separate invocation.
