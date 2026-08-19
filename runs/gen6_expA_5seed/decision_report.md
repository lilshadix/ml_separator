# gen6 Experiment A — BASE91 vs EXPANDED152 on identical test rows

**Run** `gen6_diversity_causal_20260819T102822Z`

## What was actually done

* One shared cohort, built once at `min_cells = 3`: **5248 rows, 152 extractants, 131 ECFP clusters, 79 Tanimoto-0.7 super-clusters**, 2055 conditions, 14 metals; target sd 1.657 log units.
* Folds hold out whole super-clusters (5 folds, seeds [104729, 130363, 155921, 196613, 262147]). Every arm is scored on **byte-identical test rows**; only the training row mask changes.
* Learner: ExtraTrees, 400 trees, max_features 0.3, min_samples_leaf 2, fold seed `model_seed + fold*1009 + 9999991` (the gen5 formula, reused so a gen5 arm reproduces rather than approximates).
* Weighting: **cluster** — group_balanced_weights on the training rows' ECFP cluster — the gen5 rule, and itself arm-dependent because EXPANDED has more clusters (see 'How to break this result')
* Feature sets: MC_lig2d_ext_massaction, MC_donors. Fitting took 148 s.

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
MC_lig2d_ext_massaction      all        mae              0.1633           0.0430            0.3300            0.1633               5        5                 131             79
MC_lig2d_ext_massaction      all offset_mae              0.1773           0.0580            0.3455            0.1773               5        5                 131             79
MC_lig2d_ext_massaction      all  shape_mae              0.0243           0.0008            0.0555            0.0243               5        5                 131             79
MC_lig2d_ext_massaction   nn<0.4        mae              0.4630           0.2527            0.6903            0.4939               5        5                  42             37
MC_lig2d_ext_massaction   nn<0.4 offset_mae              0.4677           0.2419            0.7024            0.4947               5        5                  42             37
MC_lig2d_ext_massaction   nn<0.4  shape_mae              0.0742           0.0403            0.1154            0.0794               5        5                  42             37
MC_lig2d_ext_massaction   nn<0.6        mae              0.2097           0.0512            0.4267            0.2387               5        5                 105             65
MC_lig2d_ext_massaction   nn<0.6 offset_mae              0.2165           0.0508            0.4400            0.2520               5        5                 105             65
MC_lig2d_ext_massaction   nn<0.6  shape_mae              0.0358           0.0109            0.0721            0.0437               5        5                 105             65
              MC_donors      all        mae              0.1721           0.0581            0.3487            0.1721               5        5                 131             79
              MC_donors      all offset_mae              0.1795           0.0595            0.3636            0.1795               5        5                 131             79
              MC_donors      all  shape_mae              0.0179          -0.0017            0.0427            0.0179               5        5                 131             79
              MC_donors   nn<0.4        mae              0.4611           0.1620            0.8260            0.5496               5        5                  42             37
              MC_donors   nn<0.4 offset_mae              0.4721           0.1677            0.8499            0.5573               5        5                  42             37
              MC_donors   nn<0.4  shape_mae              0.0541           0.0046            0.1070            0.0748               5        5                  42             37
              MC_donors   nn<0.6        mae              0.2214           0.0811            0.4409            0.2507               5        5                 105             65
              MC_donors   nn<0.6 offset_mae              0.2305           0.0844            0.4585            0.2551               5        5                 105             65
              MC_donors   nn<0.6  shape_mae              0.0227           0.0002            0.0514            0.0321               5        5                 105             65
```

The same contrast one split partition at a time. The five seeds re-partition the same ligands, so agreement here is a consistency requirement and not five independent experiments; the interval above is the evidence.

```
            feature_set endpoint  split_seed  statistic  point_delta  ci95_low  ci95_high  units_improved  units_total
MC_lig2d_ext_massaction      all      104729        mae       0.1496    0.0344     0.3123              69          131
MC_lig2d_ext_massaction      all      104729 offset_mae       0.1484    0.0326     0.3140              69          131
MC_lig2d_ext_massaction      all      104729  shape_mae       0.0264    0.0070     0.0531              72          131
MC_lig2d_ext_massaction   nn<0.4      104729        mae       0.4931    0.2212     0.7637              26           33
MC_lig2d_ext_massaction   nn<0.4      104729 offset_mae       0.4580    0.1687     0.7476              23           33
MC_lig2d_ext_massaction   nn<0.4      104729  shape_mae       0.0833    0.0340     0.1390              20           33
MC_lig2d_ext_massaction   nn<0.6      104729        mae       0.2081    0.0368     0.4412              50           88
MC_lig2d_ext_massaction   nn<0.6      104729 offset_mae       0.1980    0.0154     0.4458              48           88
MC_lig2d_ext_massaction   nn<0.6      104729  shape_mae       0.0441    0.0185     0.0821              51           88
MC_lig2d_ext_massaction      all      130363        mae       0.1584    0.0435     0.3202              76          131
MC_lig2d_ext_massaction      all      130363 offset_mae       0.1642    0.0453     0.3339              78          131
MC_lig2d_ext_massaction      all      130363  shape_mae       0.0262   -0.0017     0.0624              67          131
MC_lig2d_ext_massaction   nn<0.4      130363        mae       0.4328    0.2078     0.6778              27           37
MC_lig2d_ext_massaction   nn<0.4      130363 offset_mae       0.4040    0.1701     0.6661              25           37
MC_lig2d_ext_massaction   nn<0.4      130363  shape_mae       0.0853    0.0400     0.1374              25           37
MC_lig2d_ext_massaction   nn<0.6      130363        mae       0.2158    0.0606     0.4381              50           85
MC_lig2d_ext_massaction   nn<0.6      130363 offset_mae       0.2267    0.0660     0.4528              52           85
MC_lig2d_ext_massaction   nn<0.6      130363  shape_mae       0.0443    0.0142     0.0872              48           85
MC_lig2d_ext_massaction      all      155921        mae       0.1653    0.0053     0.3770              78          131
MC_lig2d_ext_massaction      all      155921 offset_mae       0.1949    0.0368     0.4084              83          131
MC_lig2d_ext_massaction      all      155921  shape_mae       0.0184   -0.0086     0.0545              74          131
MC_lig2d_ext_massaction   nn<0.4      155921        mae       0.5651    0.3045     0.8540              31           38
MC_lig2d_ext_massaction   nn<0.4      155921 offset_mae       0.5896    0.3176     0.8953              30           38
MC_lig2d_ext_massaction   nn<0.4      155921  shape_mae       0.0643    0.0257     0.1092              28           38
MC_lig2d_ext_massaction   nn<0.6      155921        mae       0.2397   -0.0151     0.5379              54           86
MC_lig2d_ext_massaction   nn<0.6      155921 offset_mae       0.2663    0.0088     0.5786              55           86
MC_lig2d_ext_massaction   nn<0.6      155921  shape_mae       0.0357    0.0026     0.0789              53           86
MC_lig2d_ext_massaction      all      196613        mae       0.1601    0.0572     0.3099              77          131
MC_lig2d_ext_massaction      all      196613 offset_mae       0.1766    0.0751     0.3247              80          131
MC_lig2d_ext_massaction      all      196613  shape_mae       0.0230   -0.0053     0.0593              76          131
MC_lig2d_ext_massaction   nn<0.4      196613        mae       0.4902    0.2318     0.7611              27           35
MC_lig2d_ext_massaction   nn<0.4      196613 offset_mae       0.5137    0.2424     0.7935              27           35
MC_lig2d_ext_massaction   nn<0.4      196613  shape_mae       0.0922    0.0453     0.1472              25           35
MC_lig2d_ext_massaction   nn<0.6      196613        mae       0.2564    0.0785     0.4869              49           75
MC_lig2d_ext_massaction   nn<0.6      196613 offset_mae       0.2774    0.0945     0.5179              46           75
MC_lig2d_ext_massaction   nn<0.6      196613  shape_mae       0.0421    0.0034     0.0907              45           75
MC_lig2d_ext_massaction      all      262147        mae       0.1830    0.0623     0.3517              78          131
MC_lig2d_ext_massaction      all      262147 offset_mae       0.2024    0.0775     0.3772              82          131
MC_lig2d_ext_massaction      all      262147  shape_mae       0.0275    0.0062     0.0575              75          131
MC_lig2d_ext_massaction   nn<0.4      262147        mae       0.4884    0.2316     0.7533              28           37
MC_lig2d_ext_massaction   nn<0.4      262147 offset_mae       0.5082    0.2212     0.8033              24           37
MC_lig2d_ext_massaction   nn<0.4      262147  shape_mae       0.0721    0.0321     0.1197              23           37
MC_lig2d_ext_massaction   nn<0.6      262147        mae       0.2734    0.1028     0.5027              53           85
MC_lig2d_ext_massaction   nn<0.6      262147 offset_mae       0.2914    0.1091     0.5294              52           85
MC_lig2d_ext_massaction   nn<0.6      262147  shape_mae       0.0520    0.0263     0.0881              50           85
              MC_donors      all      104729        mae       0.1872    0.0612     0.3820              71          131
              MC_donors      all      104729 offset_mae       0.1983    0.0628     0.4017              77          131
              MC_donors      all      104729  shape_mae       0.0112   -0.0184     0.0397              66          131
              MC_donors   nn<0.4      104729        mae       0.6836    0.3202     1.0939              25           33
              MC_donors   nn<0.4      104729 offset_mae       0.6886    0.3088     1.1064              26           33
              MC_donors   nn<0.4      104729  shape_mae       0.0883    0.0222     0.1502              26           33
              MC_donors   nn<0.6      104729        mae       0.2521    0.0721     0.5274              46           88
              MC_donors   nn<0.6      104729 offset_mae       0.2518    0.0607     0.5394              50           88
              MC_donors   nn<0.6      104729  shape_mae       0.0280   -0.0032     0.0649              45           88
              MC_donors      all      130363        mae       0.1421    0.0375     0.3052              70          131
              MC_donors      all      130363 offset_mae       0.1391    0.0305     0.3074              70          131
              MC_donors      all      130363  shape_mae       0.0266    0.0074     0.0539              65          131
              MC_donors   nn<0.4      130363        mae       0.4639    0.1393     0.8382              23           37
              MC_donors   nn<0.4      130363 offset_mae       0.4338    0.0927     0.8239              23           37
              MC_donors   nn<0.4      130363  shape_mae       0.0892    0.0435     0.1403              26           37
              MC_donors   nn<0.6      130363        mae       0.1914    0.0393     0.4294              41           85
              MC_donors   nn<0.6      130363 offset_mae       0.1796    0.0212     0.4180              43           85
              MC_donors   nn<0.6      130363  shape_mae       0.0391    0.0142     0.0742              45           85
              MC_donors      all      155921        mae       0.1965    0.0714     0.3946              65          131
              MC_donors      all      155921 offset_mae       0.2055    0.0729     0.4090              79          131
              MC_donors      all      155921  shape_mae       0.0118   -0.0136     0.0415              69          131
              MC_donors   nn<0.4      155921        mae       0.6523    0.3160     1.0715              27           38
              MC_donors   nn<0.4      155921 offset_mae       0.6694    0.3129     1.0993              28           38
              MC_donors   nn<0.4      155921  shape_mae       0.0630    0.0075     0.1230              29           38
              MC_donors   nn<0.6      155921        mae       0.3168    0.1372     0.5774              47           86
              MC_donors   nn<0.6      155921 offset_mae       0.3356    0.1499     0.6068              55           86
              MC_donors   nn<0.6      155921  shape_mae       0.0275   -0.0051     0.0606              51           86
              MC_donors      all      196613        mae       0.1518    0.0254     0.3289              70          131
              MC_donors      all      196613 offset_mae       0.1474    0.0147     0.3260              75          131
              MC_donors      all      196613  shape_mae       0.0194   -0.0003     0.0413              78          131
              MC_donors   nn<0.4      196613        mae       0.3886   -0.0099     0.8776              20           35
              MC_donors   nn<0.4      196613 offset_mae       0.3916   -0.0301     0.8920              20           35
              MC_donors   nn<0.4      196613  shape_mae       0.0516   -0.0101     0.1028              27           35
              MC_donors   nn<0.6      196613        mae       0.2318    0.0240     0.5053              42           75
              MC_donors   nn<0.6      196613 offset_mae       0.2238    0.0054     0.5083              41           75
              MC_donors   nn<0.6      196613  shape_mae       0.0302   -0.0013     0.0610              48           75
              MC_donors      all      262147        mae       0.1827    0.0599     0.3579              73          131
              MC_donors      all      262147 offset_mae       0.2073    0.0760     0.3926              84          131
              MC_donors      all      262147  shape_mae       0.0203   -0.0035     0.0514              63          131
              MC_donors   nn<0.4      262147        mae       0.5596    0.2056     0.9372              26           37
              MC_donors   nn<0.4      262147 offset_mae       0.6031    0.2373     0.9929              26           37
              MC_donors   nn<0.4      262147  shape_mae       0.0820    0.0069     0.1542              26           37
              MC_donors   nn<0.6      262147        mae       0.2614    0.0896     0.5171              52           85
              MC_donors   nn<0.6      262147 offset_mae       0.2845    0.0987     0.5547              56           85
              MC_donors   nn<0.6      262147  shape_mae       0.0355    0.0027     0.0775              44           85
```

## Pre-registered hypotheses (protocol §5)

Scoring rule, stated so a verdict cannot be over-read: **PASS** = the pre-registered condition is met, including the seed-agreement clause; **FAIL** = the CI95 excludes the predicted direction (evidence *against*); **INCONCLUSIVE** = the interval straddles zero, or the seed clause cannot be evaluated (a single-seed pilot). An INCONCLUSIVE is not a weak PASS.

### Feature set `MC_lig2d_ext_massaction`

**A1 — PASS**  ·  Adding sparse diverse chemistry improves overall accuracy.

* pass condition: macro_MAE(BASE) − macro_MAE(EXPANDED) > 0 with CI95 low > 0, and ≥ 4/5 seeds positive.
* measured: macro MAE (one ECFP cluster = one vote), all test rows: Δ = +0.1633 log units (CI95 [+0.0430, +0.3300], 131 units, 79 bootstrap blocks; mean over seeds +0.1633, 5/5 seeds positive)
* what would falsify this: A CI95 whose upper end is ≤ 0 — EXPANDED no better, or worse, than BASE on identical test rows — falsifies it, and with it the claim that coverage is the binding constraint at this scale (protocol §8 case C).

**A2 — PASS**  ·  The gain is largest on the hardest chemistry.

* pass condition: gain at nn<0.4 > gain overall, both CI95 low > 0.
* measured: gain at nn<0.4: Δ = +0.4630 log units (CI95 [+0.2527, +0.6903], 42 units, 37 bootstrap blocks; mean over seeds +0.4939, 5/5 seeds positive); gain overall: Δ = +0.1633 log units (CI95 [+0.0430, +0.3300], 131 units, 79 bootstrap blocks; mean over seeds +0.1633, 5/5 seeds positive); hard > overall: True
* what would falsify this: A gain that is flat or smaller on the far-from-training subset falsifies it: the model would then be improving where it was already supported, i.e. a depth effect wearing a coverage costume.

**A3 — PASS**  ·  The gain is in the level, not the shape.

* pass condition: offset_mae(BASE) − offset_mae(EXPANDED) CI95 low > 0, and larger than the shape-MAE gain.
* measured: offset gain: Δ = +0.1773 log units (CI95 [+0.0580, +0.3455], 131 units, 79 bootstrap blocks; mean over seeds +0.1773, 5/5 seeds positive); shape gain: Δ = +0.0243 log units (CI95 [+0.0008, +0.0555], 131 units, 79 bootstrap blocks; mean over seeds +0.0243, 5/5 seeds positive); offset > shape: True
* what would falsify this: A gain that lives in the shape while the per-ligand offset is unchanged falsifies it, and points at the representation rather than the coverage (protocol §8 case B).

**A4 — PASS**  ·  The gain is information, not row count.

* pass condition: EXPANDED beats EXPANDED_ROWMATCHED by less than it beats BASE, and beats EXPANDED_SHUFFLED with CI95 low > 0.
* measured: vs the information null (sparse targets permuted among themselves): Δ = +0.1726 log units (CI95 [+0.0553, +0.3419], 131 units, 79 bootstrap blocks; mean over seeds +0.1726, 5/5 seeds positive); vs the row-budget-matched control: Δ = +0.0019 log units (CI95 [-0.0031, +0.0061], 131 units, 79 bootstrap blocks; mean over seeds +0.0019, 4/5 seeds positive); rowmatched gap < BASE gap: True
* what would falsify this: If EXPANDED_SHUFFLED matches EXPANDED, the extra rows helped as regularisation or through the change in cluster weighting and carried no chemical information; if EXPANDED beats EXPANDED_ROWMATCHED by as much as it beats BASE, the effect is row count, not coverage.

### Feature set `MC_donors`

**A1 — PASS**  ·  Adding sparse diverse chemistry improves overall accuracy.

* pass condition: macro_MAE(BASE) − macro_MAE(EXPANDED) > 0 with CI95 low > 0, and ≥ 4/5 seeds positive.
* measured: macro MAE (one ECFP cluster = one vote), all test rows: Δ = +0.1721 log units (CI95 [+0.0581, +0.3487], 131 units, 79 bootstrap blocks; mean over seeds +0.1721, 5/5 seeds positive)
* what would falsify this: A CI95 whose upper end is ≤ 0 — EXPANDED no better, or worse, than BASE on identical test rows — falsifies it, and with it the claim that coverage is the binding constraint at this scale (protocol §8 case C).

**A2 — PASS**  ·  The gain is largest on the hardest chemistry.

* pass condition: gain at nn<0.4 > gain overall, both CI95 low > 0.
* measured: gain at nn<0.4: Δ = +0.4611 log units (CI95 [+0.1620, +0.8260], 42 units, 37 bootstrap blocks; mean over seeds +0.5496, 5/5 seeds positive); gain overall: Δ = +0.1721 log units (CI95 [+0.0581, +0.3487], 131 units, 79 bootstrap blocks; mean over seeds +0.1721, 5/5 seeds positive); hard > overall: True
* what would falsify this: A gain that is flat or smaller on the far-from-training subset falsifies it: the model would then be improving where it was already supported, i.e. a depth effect wearing a coverage costume.

**A3 — PASS**  ·  The gain is in the level, not the shape.

* pass condition: offset_mae(BASE) − offset_mae(EXPANDED) CI95 low > 0, and larger than the shape-MAE gain.
* measured: offset gain: Δ = +0.1795 log units (CI95 [+0.0595, +0.3636], 131 units, 79 bootstrap blocks; mean over seeds +0.1795, 5/5 seeds positive); shape gain: Δ = +0.0179 log units (CI95 [-0.0017, +0.0427], 131 units, 79 bootstrap blocks; mean over seeds +0.0179, 5/5 seeds positive); offset > shape: True
* what would falsify this: A gain that lives in the shape while the per-ligand offset is unchanged falsifies it, and points at the representation rather than the coverage (protocol §8 case B).

**A4 — PASS**  ·  The gain is information, not row count.

* pass condition: EXPANDED beats EXPANDED_ROWMATCHED by less than it beats BASE, and beats EXPANDED_SHUFFLED with CI95 low > 0.
* measured: vs the information null (sparse targets permuted among themselves): Δ = +0.2368 log units (CI95 [+0.1246, +0.3988], 131 units, 79 bootstrap blocks; mean over seeds +0.2368, 5/5 seeds positive); vs the row-budget-matched control: Δ = -0.0034 log units (CI95 [-0.0096, +0.0011], 131 units, 79 bootstrap blocks; mean over seeds -0.0034, 1/5 seeds positive); rowmatched gap < BASE gap: True
* what would falsify this: If EXPANDED_SHUFFLED matches EXPANDED, the extra rows helped as regularisation or through the change in cluster weighting and carried no chemical information; if EXPANDED beats EXPANDED_ROWMATCHED by as much as it beats BASE, the effect is row count, not coverage.

## Where the gain sits — read this before the leaderboard

One ECFP cluster is one vote, and on this cohort a large share of the clusters consist **entirely** of the sparsely-measured ligands that only EXPANDED can train on. If the gain lives only there, the honest claim is 'the expansion repairs chemistry that was missing', not 'the model got better'. `units` counts scoring clusters; `gain` is BASE minus EXPANDED macro MAE within that group.

```
            feature_set endpoint                 kind  units    gain  units_improved
              MC_donors      all   BASE-eligible only   72.0 -0.0257          0.4306
              MC_donors      all added chemistry only   57.0  0.4274          0.6596
              MC_donors      all                mixed    2.0  0.0126          0.6000
              MC_donors   nn<0.4   BASE-eligible only    9.2 -0.0016          0.5217
              MC_donors   nn<0.4 added chemistry only   26.8  0.7380          0.7239
MC_lig2d_ext_massaction      all   BASE-eligible only   72.0  0.0081          0.4750
MC_lig2d_ext_massaction      all added chemistry only   57.0  0.3670          0.7228
MC_lig2d_ext_massaction      all                mixed    2.0 -0.0544          0.1000
MC_lig2d_ext_massaction   nn<0.4   BASE-eligible only    9.2  0.2736          0.7391
MC_lig2d_ext_massaction   nn<0.4 added chemistry only   26.8  0.5701          0.7836
```

## Arm leaderboard (per feature set, mean over seeds)

Macro MAE is primary. Pooled numbers are printed and never used for selection: the largest single extractant holds 28 % of rows and the largest ECFP cluster 38 %.

```
            feature_set                 arm  macro_mae  offset_mae  shape_mae  shape_r2  pooled_mae  median_ligand_mae  worst_quartile_ligand_mae  frac_within_1_log  n_ligands  n_ecfp_clusters  n_superclusters  n_macro_units  n_eff_pooled_rows
MC_lig2d_ext_massaction                BASE     1.2100      1.0444     0.5269    0.1557      1.2021             1.0564                     2.2855             0.4918      152.0            131.0             79.0          131.0             6.2561
MC_lig2d_ext_massaction            EXPANDED     1.0468      0.8744     0.5069    0.1621      1.1340             0.9027                     2.0934             0.5179      152.0            131.0             79.0          131.0             6.2561
MC_lig2d_ext_massaction EXPANDED_ROWMATCHED     1.0486      0.8748     0.5094    0.1585      1.1301             0.9156                     2.0921             0.5194      152.0            131.0             79.0          131.0             6.2561
MC_lig2d_ext_massaction   EXPANDED_SHUFFLED     1.2193      1.0633     0.5392    0.1359      1.2376             1.0734                     2.2785             0.4816      152.0            131.0             79.0          131.0             6.2561
              MC_donors                BASE     1.1870      1.0270     0.5122    0.1042      1.1796             0.9804                     2.2939             0.5002      152.0            131.0             79.0          131.0             6.2561
              MC_donors            EXPANDED     1.0150      0.8613     0.4991    0.0683      1.1630             0.9172                     1.9782             0.5048      152.0            131.0             79.0          131.0             6.2561
              MC_donors EXPANDED_ROWMATCHED     1.0115      0.8560     0.5056    0.0587      1.1644             0.9144                     1.9715             0.5062      152.0            131.0             79.0          131.0             6.2561
              MC_donors   EXPANDED_SHUFFLED     1.2517      1.0717     0.5716    0.0400      1.2393             1.1240                     2.2434             0.4708      152.0            131.0             79.0          131.0             6.2561
```

## Hard chemistry — the co-primary endpoint

Bins are cut on `nn_reference_tanimoto`: the maximum Tanimoto from the test ligand to the **BASE training extractants of its own fold**. The same column is used for every arm, so the subset does not move with the arm. `nn_expanded_tanimoto` is carried in the OOF for description only and selects nothing.

```
            feature_set endpoint                 arm  n_rows  n_ligands  n_ecfp_clusters  n_superclusters  macro_mae  offset_mae  shape_mae  frac_within_1_log
MC_lig2d_ext_massaction   nn<0.4                BASE   544.2       36.0             36.0             33.8     1.5368      1.4384     0.4576             0.3973
MC_lig2d_ext_massaction   nn<0.6                BASE  1985.8       86.8             83.8             55.8     1.3988      1.2622     0.5406             0.4770
MC_lig2d_ext_massaction      all                BASE  5248.0      152.0            131.0             79.0     1.2100      1.0444     0.5269             0.4918
MC_lig2d_ext_massaction   nn<0.4            EXPANDED   544.2       36.0             36.0             33.8     1.0429      0.9437     0.3782             0.4864
MC_lig2d_ext_massaction   nn<0.6            EXPANDED  1985.8       86.8             83.8             55.8     1.1601      1.0163     0.4990             0.5034
MC_lig2d_ext_massaction      all            EXPANDED  5248.0      152.0            131.0             79.0     1.0468      0.8744     0.5069             0.5179
MC_lig2d_ext_massaction   nn<0.4 EXPANDED_ROWMATCHED   544.2       36.0             36.0             33.8     1.0389      0.9432     0.3763             0.4895
MC_lig2d_ext_massaction   nn<0.6 EXPANDED_ROWMATCHED  1985.8       86.8             83.8             55.8     1.1594      1.0195     0.4990             0.5054
MC_lig2d_ext_massaction      all EXPANDED_ROWMATCHED  5248.0      152.0            131.0             79.0     1.0486      0.8748     0.5094             0.5194
MC_lig2d_ext_massaction   nn<0.4   EXPANDED_SHUFFLED   544.2       36.0             36.0             33.8     1.6102      1.5441     0.4237             0.3583
MC_lig2d_ext_massaction   nn<0.6   EXPANDED_SHUFFLED  1985.8       86.8             83.8             55.8     1.3991      1.2720     0.5314             0.4493
MC_lig2d_ext_massaction      all   EXPANDED_SHUFFLED  5248.0      152.0            131.0             79.0     1.2193      1.0633     0.5392             0.4816
              MC_donors   nn<0.4                BASE   544.2       36.0             36.0             33.8     1.5259      1.4346     0.4562             0.4021
              MC_donors   nn<0.6                BASE  1985.8       86.8             83.8             55.8     1.3247      1.1838     0.5282             0.4938
              MC_donors      all                BASE  5248.0      152.0            131.0             79.0     1.1870      1.0270     0.5122             0.5002
              MC_donors   nn<0.4            EXPANDED   544.2       36.0             36.0             33.8     0.9763      0.8774     0.3814             0.4625
              MC_donors   nn<0.6            EXPANDED  1985.8       86.8             83.8             55.8     1.0739      0.9359     0.4980             0.4975
              MC_donors      all            EXPANDED  5248.0      152.0            131.0             79.0     1.0150      0.8613     0.4991             0.5048
              MC_donors   nn<0.4 EXPANDED_ROWMATCHED   544.2       36.0             36.0             33.8     0.9662      0.8672     0.3834             0.4624
              MC_donors   nn<0.6 EXPANDED_ROWMATCHED  1985.8       86.8             83.8             55.8     1.0690      0.9297     0.5034             0.5009
              MC_donors      all EXPANDED_ROWMATCHED  5248.0      152.0            131.0             79.0     1.0115      0.8560     0.5056             0.5062
              MC_donors   nn<0.4   EXPANDED_SHUFFLED   544.2       36.0             36.0             33.8     1.5641      1.4700     0.4714             0.3696
              MC_donors   nn<0.6   EXPANDED_SHUFFLED  1985.8       86.8             83.8             55.8     1.3944      1.2456     0.5778             0.4625
              MC_donors      all   EXPANDED_SHUFFLED  5248.0      152.0            131.0             79.0     1.2517      1.0717     0.5716             0.4708
```

Chemistry units surviving each endpoint (this is the n behind every hard-chemistry CI):

```
            feature_set endpoint  split_seed  n_rows  n_ligands  n_ecfp_clusters  n_superclusters
MC_lig2d_ext_massaction      all    169954.6  5248.0      152.0            131.0             79.0
MC_lig2d_ext_massaction   nn<0.4    169954.6   544.2       36.0             36.0             33.8
MC_lig2d_ext_massaction   nn<0.6    169954.6  1985.8       86.8             83.8             55.8
              MC_donors      all    169954.6  5248.0      152.0            131.0             79.0
              MC_donors   nn<0.4    169954.6   544.2       36.0             36.0             33.8
              MC_donors   nn<0.6    169954.6  1985.8       86.8             83.8             55.8
```

The fixed-reference rule is not a formality — this is how many test rows would have landed in a different bin had each arm's own training set defined it (mean over seeds and feature sets):

```
endpoint  split_seed  n_rows_reference_bins  n_rows_if_each_arm_used_its_own_training_set  n_rows_that_would_change_bin
  nn<0.4    169954.6                  544.2                                         407.6                         136.6
  nn<0.6    169954.6                 1985.8                                        1383.0                         602.8
```

## Every contrast, every endpoint

```
            feature_set endpoint                      comparison  statistic  pooled_point_delta  pooled_ci95_low  pooled_ci95_high  mean_point_delta  seeds_positive  n_seeds  pooled_units_total  preregistered
MC_lig2d_ext_massaction      all                EXPANDED_vs_BASE        mae              0.1633           0.0430            0.3300            0.1633               5        5                 131           True
MC_lig2d_ext_massaction      all                EXPANDED_vs_BASE offset_mae              0.1773           0.0580            0.3455            0.1773               5        5                 131           True
MC_lig2d_ext_massaction      all                EXPANDED_vs_BASE  shape_mae              0.0243           0.0008            0.0555            0.0243               5        5                 131           True
MC_lig2d_ext_massaction      all     EXPANDED_ROWMATCHED_vs_BASE        mae              0.1614           0.0407            0.3297            0.1614               5        5                 131           True
MC_lig2d_ext_massaction      all     EXPANDED_ROWMATCHED_vs_BASE offset_mae              0.1756           0.0559            0.3467            0.1756               5        5                 131           True
MC_lig2d_ext_massaction      all     EXPANDED_ROWMATCHED_vs_BASE  shape_mae              0.0215          -0.0038            0.0546            0.0215               5        5                 131           True
MC_lig2d_ext_massaction      all   EXPANDED_vs_EXPANDED_SHUFFLED        mae              0.1726           0.0553            0.3419            0.1726               5        5                 131           True
MC_lig2d_ext_massaction      all   EXPANDED_vs_EXPANDED_SHUFFLED offset_mae              0.1885           0.0700            0.3615            0.1885               5        5                 131           True
MC_lig2d_ext_massaction      all   EXPANDED_vs_EXPANDED_SHUFFLED  shape_mae              0.0324           0.0186            0.0516            0.0324               5        5                 131           True
MC_lig2d_ext_massaction      all       EXPANDED_SHUFFLED_vs_BASE        mae             -0.0093          -0.0519            0.0373           -0.0093               2        5                 131           True
MC_lig2d_ext_massaction      all       EXPANDED_SHUFFLED_vs_BASE offset_mae             -0.0112          -0.0619            0.0413           -0.0112               2        5                 131           True
MC_lig2d_ext_massaction      all       EXPANDED_SHUFFLED_vs_BASE  shape_mae             -0.0082          -0.0274            0.0181           -0.0082               1        5                 131           True
MC_lig2d_ext_massaction      all EXPANDED_vs_EXPANDED_ROWMATCHED        mae              0.0019          -0.0031            0.0061            0.0019               4        5                 131          False
MC_lig2d_ext_massaction      all EXPANDED_vs_EXPANDED_ROWMATCHED offset_mae              0.0017          -0.0043            0.0069            0.0017               3        5                 131          False
MC_lig2d_ext_massaction      all EXPANDED_vs_EXPANDED_ROWMATCHED  shape_mae              0.0028          -0.0007            0.0055            0.0028               5        5                 131          False
MC_lig2d_ext_massaction   nn<0.4                EXPANDED_vs_BASE        mae              0.4630           0.2527            0.6903            0.4939               5        5                  42           True
MC_lig2d_ext_massaction   nn<0.4                EXPANDED_vs_BASE offset_mae              0.4677           0.2419            0.7024            0.4947               5        5                  42           True
MC_lig2d_ext_massaction   nn<0.4                EXPANDED_vs_BASE  shape_mae              0.0742           0.0403            0.1154            0.0794               5        5                  42           True
MC_lig2d_ext_massaction   nn<0.4     EXPANDED_ROWMATCHED_vs_BASE        mae              0.4681           0.2575            0.6938            0.4980               5        5                  42           True
MC_lig2d_ext_massaction   nn<0.4     EXPANDED_ROWMATCHED_vs_BASE offset_mae              0.4706           0.2461            0.7087            0.4952               5        5                  42           True
MC_lig2d_ext_massaction   nn<0.4     EXPANDED_ROWMATCHED_vs_BASE  shape_mae              0.0756           0.0413            0.1171            0.0813               5        5                  42           True
MC_lig2d_ext_massaction   nn<0.4   EXPANDED_vs_EXPANDED_SHUFFLED        mae              0.4944           0.2558            0.7707            0.5673               5        5                  42           True
MC_lig2d_ext_massaction   nn<0.4   EXPANDED_vs_EXPANDED_SHUFFLED offset_mae              0.5359           0.2863            0.8167            0.6004               5        5                  42           True
MC_lig2d_ext_massaction   nn<0.4   EXPANDED_vs_EXPANDED_SHUFFLED  shape_mae              0.0346           0.0071            0.0666            0.0455               5        5                  42           True
MC_lig2d_ext_massaction   nn<0.4       EXPANDED_SHUFFLED_vs_BASE        mae             -0.0314          -0.1422            0.0675           -0.0734               0        5                  42           True
MC_lig2d_ext_massaction   nn<0.4       EXPANDED_SHUFFLED_vs_BASE offset_mae             -0.0681          -0.1927            0.0473           -0.1057               0        5                  42           True
MC_lig2d_ext_massaction   nn<0.4       EXPANDED_SHUFFLED_vs_BASE  shape_mae              0.0396           0.0064            0.0744            0.0339               4        5                  42           True
MC_lig2d_ext_massaction   nn<0.4 EXPANDED_vs_EXPANDED_ROWMATCHED        mae             -0.0051          -0.0164            0.0061           -0.0040               1        5                  42          False
MC_lig2d_ext_massaction   nn<0.4 EXPANDED_vs_EXPANDED_ROWMATCHED offset_mae             -0.0029          -0.0161            0.0106           -0.0006               2        5                  42          False
MC_lig2d_ext_massaction   nn<0.4 EXPANDED_vs_EXPANDED_ROWMATCHED  shape_mae             -0.0013          -0.0046            0.0025           -0.0019               2        5                  42          False
MC_lig2d_ext_massaction   nn<0.6                EXPANDED_vs_BASE        mae              0.2097           0.0512            0.4267            0.2387               5        5                 105           True
MC_lig2d_ext_massaction   nn<0.6                EXPANDED_vs_BASE offset_mae              0.2165           0.0508            0.4400            0.2520               5        5                 105           True
MC_lig2d_ext_massaction   nn<0.6                EXPANDED_vs_BASE  shape_mae              0.0358           0.0109            0.0721            0.0437               5        5                 105           True
MC_lig2d_ext_massaction   nn<0.6     EXPANDED_ROWMATCHED_vs_BASE        mae              0.2076           0.0495            0.4242            0.2394               5        5                 105           True
MC_lig2d_ext_massaction   nn<0.6     EXPANDED_ROWMATCHED_vs_BASE offset_mae              0.2133           0.0468            0.4385            0.2490               5        5                 105           True
MC_lig2d_ext_massaction   nn<0.6     EXPANDED_ROWMATCHED_vs_BASE  shape_mae              0.0341           0.0071            0.0727            0.0435               5        5                 105           True
MC_lig2d_ext_massaction   nn<0.6   EXPANDED_vs_EXPANDED_SHUFFLED        mae              0.2175           0.0723            0.4300            0.2390               5        5                 105           True
MC_lig2d_ext_massaction   nn<0.6   EXPANDED_vs_EXPANDED_SHUFFLED offset_mae              0.2374           0.0849            0.4581            0.2617               5        5                 105           True
MC_lig2d_ext_massaction   nn<0.6   EXPANDED_vs_EXPANDED_SHUFFLED  shape_mae              0.0363           0.0215            0.0570            0.0324               5        5                 105           True
MC_lig2d_ext_massaction   nn<0.6       EXPANDED_SHUFFLED_vs_BASE        mae             -0.0078          -0.0628            0.0575           -0.0003               3        5                 105           True
MC_lig2d_ext_massaction   nn<0.6       EXPANDED_SHUFFLED_vs_BASE offset_mae             -0.0209          -0.0831            0.0512           -0.0097               1        5                 105           True
MC_lig2d_ext_massaction   nn<0.6       EXPANDED_SHUFFLED_vs_BASE  shape_mae             -0.0006          -0.0214            0.0290            0.0112               4        5                 105           True
MC_lig2d_ext_massaction   nn<0.6 EXPANDED_vs_EXPANDED_ROWMATCHED        mae              0.0020          -0.0043            0.0085           -0.0007               3        5                 105          False
MC_lig2d_ext_massaction   nn<0.6 EXPANDED_vs_EXPANDED_ROWMATCHED offset_mae              0.0032          -0.0048            0.0103            0.0030               3        5                 105          False
MC_lig2d_ext_massaction   nn<0.6 EXPANDED_vs_EXPANDED_ROWMATCHED  shape_mae              0.0016          -0.0025            0.0047            0.0002               3        5                 105          False
              MC_donors      all                EXPANDED_vs_BASE        mae              0.1721           0.0581            0.3487            0.1721               5        5                 131           True
              MC_donors      all                EXPANDED_vs_BASE offset_mae              0.1795           0.0595            0.3636            0.1795               5        5                 131           True
              MC_donors      all                EXPANDED_vs_BASE  shape_mae              0.0179          -0.0017            0.0427            0.0179               5        5                 131           True
              MC_donors      all     EXPANDED_ROWMATCHED_vs_BASE        mae              0.1755           0.0599            0.3542            0.1755               5        5                 131           True
              MC_donors      all     EXPANDED_ROWMATCHED_vs_BASE offset_mae              0.1846           0.0643            0.3697            0.1846               5        5                 131           True
              MC_donors      all     EXPANDED_ROWMATCHED_vs_BASE  shape_mae              0.0113          -0.0094            0.0385            0.0113               4        5                 131           True
              MC_donors      all   EXPANDED_vs_EXPANDED_SHUFFLED        mae              0.2368           0.1246            0.3988            0.2368               5        5                 131           True
              MC_donors      all   EXPANDED_vs_EXPANDED_SHUFFLED offset_mae              0.2308           0.1120            0.4006            0.2308               5        5                 131           True
              MC_donors      all   EXPANDED_vs_EXPANDED_SHUFFLED  shape_mae              0.0736           0.0426            0.1104            0.0736               5        5                 131           True
              MC_donors      all       EXPANDED_SHUFFLED_vs_BASE        mae             -0.0647          -0.1189           -0.0067           -0.0647               0        5                 131           True
              MC_donors      all       EXPANDED_SHUFFLED_vs_BASE offset_mae             -0.0513          -0.1097            0.0063           -0.0513               0        5                 131           True
              MC_donors      all       EXPANDED_SHUFFLED_vs_BASE  shape_mae             -0.0558          -0.0881           -0.0223           -0.0558               0        5                 131           True
              MC_donors      all EXPANDED_vs_EXPANDED_ROWMATCHED        mae             -0.0034          -0.0096            0.0011           -0.0034               1        5                 131          False
              MC_donors      all EXPANDED_vs_EXPANDED_ROWMATCHED offset_mae             -0.0051          -0.0106            0.0002           -0.0051               0        5                 131          False
              MC_donors      all EXPANDED_vs_EXPANDED_ROWMATCHED  shape_mae              0.0066           0.0020            0.0104            0.0066               5        5                 131          False
              MC_donors   nn<0.4                EXPANDED_vs_BASE        mae              0.4611           0.1620            0.8260            0.5496               5        5                  42           True
              MC_donors   nn<0.4                EXPANDED_vs_BASE offset_mae              0.4721           0.1677            0.8499            0.5573               5        5                  42           True
              MC_donors   nn<0.4                EXPANDED_vs_BASE  shape_mae              0.0541           0.0046            0.1070            0.0748               5        5                  42           True
              MC_donors   nn<0.4     EXPANDED_ROWMATCHED_vs_BASE        mae              0.4720           0.1703            0.8404            0.5597               5        5                  42           True
              MC_donors   nn<0.4     EXPANDED_ROWMATCHED_vs_BASE offset_mae              0.4848           0.1799            0.8599            0.5674               5        5                  42           True
              MC_donors   nn<0.4     EXPANDED_ROWMATCHED_vs_BASE  shape_mae              0.0521           0.0015            0.1049            0.0728               5        5                  42           True
              MC_donors   nn<0.4   EXPANDED_vs_EXPANDED_SHUFFLED        mae              0.5125           0.2356            0.8416            0.5878               5        5                  42           True
              MC_donors   nn<0.4   EXPANDED_vs_EXPANDED_SHUFFLED offset_mae              0.5165           0.2272            0.8490            0.5926               5        5                  42           True
              MC_donors   nn<0.4   EXPANDED_vs_EXPANDED_SHUFFLED  shape_mae              0.0696           0.0192            0.1275            0.0901               5        5                  42           True
              MC_donors   nn<0.4       EXPANDED_SHUFFLED_vs_BASE        mae             -0.0513          -0.1870            0.0972           -0.0382               2        5                  42           True
              MC_donors   nn<0.4       EXPANDED_SHUFFLED_vs_BASE offset_mae             -0.0444          -0.1887            0.1118           -0.0353               0        5                  42           True
              MC_donors   nn<0.4       EXPANDED_SHUFFLED_vs_BASE  shape_mae             -0.0155          -0.0542            0.0258           -0.0153               2        5                  42           True
              MC_donors   nn<0.4 EXPANDED_vs_EXPANDED_ROWMATCHED        mae             -0.0109          -0.0206           -0.0008           -0.0101               1        5                  42          False
              MC_donors   nn<0.4 EXPANDED_vs_EXPANDED_ROWMATCHED offset_mae             -0.0127          -0.0234           -0.0018           -0.0101               1        5                  42          False
              MC_donors   nn<0.4 EXPANDED_vs_EXPANDED_ROWMATCHED  shape_mae              0.0020          -0.0020            0.0064            0.0020               3        5                  42          False
              MC_donors   nn<0.6                EXPANDED_vs_BASE        mae              0.2214           0.0811            0.4409            0.2507               5        5                 105           True
              MC_donors   nn<0.6                EXPANDED_vs_BASE offset_mae              0.2305           0.0844            0.4585            0.2551               5        5                 105           True
              MC_donors   nn<0.6                EXPANDED_vs_BASE  shape_mae              0.0227           0.0002            0.0514            0.0321               5        5                 105           True
              MC_donors   nn<0.6     EXPANDED_ROWMATCHED_vs_BASE        mae              0.2283           0.0857            0.4494            0.2556               5        5                 105           True
              MC_donors   nn<0.6     EXPANDED_ROWMATCHED_vs_BASE offset_mae              0.2351           0.0866            0.4647            0.2605               5        5                 105           True
              MC_donors   nn<0.6     EXPANDED_ROWMATCHED_vs_BASE  shape_mae              0.0168          -0.0062            0.0474            0.0267               5        5                 105           True
              MC_donors   nn<0.6   EXPANDED_vs_EXPANDED_SHUFFLED        mae              0.2776           0.1482            0.4721            0.3205               5        5                 105           True
              MC_donors   nn<0.6   EXPANDED_vs_EXPANDED_SHUFFLED offset_mae              0.2824           0.1400            0.4923            0.3196               5        5                 105           True
              MC_donors   nn<0.6   EXPANDED_vs_EXPANDED_SHUFFLED  shape_mae              0.0700           0.0368            0.1036            0.0811               5        5                 105           True
              MC_donors   nn<0.6       EXPANDED_SHUFFLED_vs_BASE        mae             -0.0561          -0.1221            0.0203           -0.0697               0        5                 105           True
              MC_donors   nn<0.6       EXPANDED_SHUFFLED_vs_BASE offset_mae             -0.0519          -0.1238            0.0213           -0.0645               0        5                 105           True
              MC_donors   nn<0.6       EXPANDED_SHUFFLED_vs_BASE  shape_mae             -0.0473          -0.0746           -0.0130           -0.0490               0        5                 105           True
              MC_donors   nn<0.6 EXPANDED_vs_EXPANDED_ROWMATCHED        mae             -0.0069          -0.0160            0.0001           -0.0049               1        5                 105          False
              MC_donors   nn<0.6 EXPANDED_vs_EXPANDED_ROWMATCHED offset_mae             -0.0046          -0.0144            0.0042           -0.0054               1        5                 105          False
              MC_donors   nn<0.6 EXPANDED_vs_EXPANDED_ROWMATCHED  shape_mae              0.0059           0.0006            0.0101            0.0053               5        5                 105          False
```

Per-seed intervals (the same contrasts, one split partition at a time) are in `contrasts.csv`.

## Out-of-distribution behaviour

Error against distance to the fold's BASE training chemistry, with the forest's own dispersion (sd across trees) where it was recorded. A model that knows where it is unsupported is worth more than one that always emits a number. `mean_prediction_sd` is recorded for **EXPANDED arm only** and is blank elsewhere — it is a property of one fitted forest, not a comparison between arms.

```
            feature_set                 arm        bin  n_rows  n_ligands  pooled_mae  macro_mae_ligand  offset_mae  shape_mae  mean_prediction_sd
MC_lig2d_ext_massaction                BASE [0.0, 0.4)   544.2       36.0      1.4031            1.5368      1.4384     0.4576                 NaN
MC_lig2d_ext_massaction                BASE [0.4, 0.6)  1441.6       50.8      1.2234            1.2914      1.1341     0.5982                 NaN
MC_lig2d_ext_massaction                BASE [0.6, 0.8)  3262.2       65.2      1.1602            0.9506      0.7558     0.5068                 NaN
MC_lig2d_ext_massaction            EXPANDED [0.0, 0.4)   544.2       36.0      1.1978            1.0429      0.9437     0.3782              1.3513
MC_lig2d_ext_massaction            EXPANDED [0.4, 0.6)  1441.6       50.8      1.2489            1.2461      1.0666     0.5842              1.1711
MC_lig2d_ext_massaction            EXPANDED [0.6, 0.8)  3262.2       65.2      1.0744            0.8990      0.6848     0.5158              1.1646
MC_lig2d_ext_massaction EXPANDED_ROWMATCHED [0.0, 0.4)   544.2       36.0      1.2039            1.0389      0.9432     0.3763                 NaN
MC_lig2d_ext_massaction EXPANDED_ROWMATCHED [0.4, 0.6)  1441.6       50.8      1.2463            1.2476      1.0723     0.5857                 NaN
MC_lig2d_ext_massaction EXPANDED_ROWMATCHED [0.6, 0.8)  3262.2       65.2      1.0682            0.9019      0.6812     0.5214                 NaN
MC_lig2d_ext_massaction   EXPANDED_SHUFFLED [0.0, 0.4)   544.2       36.0      1.5253            1.6102      1.5441     0.4237                 NaN
MC_lig2d_ext_massaction   EXPANDED_SHUFFLED [0.4, 0.6)  1441.6       50.8      1.2910            1.2384      1.0754     0.6078                 NaN
MC_lig2d_ext_massaction   EXPANDED_SHUFFLED [0.6, 0.8)  3262.2       65.2      1.1643            0.9850      0.7814     0.5476                 NaN
              MC_donors                BASE [0.0, 0.4)   544.2       36.0      1.4372            1.5259      1.4346     0.4562                 NaN
              MC_donors                BASE [0.4, 0.6)  1441.6       50.8      1.1431            1.1724      1.0062     0.5793                 NaN
              MC_donors                BASE [0.6, 0.8)  3262.2       65.2      1.1534            0.9883      0.8191     0.4892                 NaN
              MC_donors            EXPANDED [0.0, 0.4)   544.2       36.0      1.3102            0.9763      0.8774     0.3814              0.9189
              MC_donors            EXPANDED [0.4, 0.6)  1441.6       50.8      1.1827            1.1394      0.9726     0.5801              1.0209
              MC_donors            EXPANDED [0.6, 0.8)  3262.2       65.2      1.1313            0.9444      0.7629     0.4992              0.8884
              MC_donors EXPANDED_ROWMATCHED [0.0, 0.4)   544.2       36.0      1.3203            0.9662      0.8672     0.3834                 NaN
              MC_donors EXPANDED_ROWMATCHED [0.4, 0.6)  1441.6       50.8      1.1831            1.1384      0.9699     0.5877                 NaN
              MC_donors EXPANDED_ROWMATCHED [0.6, 0.8)  3262.2       65.2      1.1317            0.9431      0.7579     0.5070                 NaN
              MC_donors   EXPANDED_SHUFFLED [0.0, 0.4)   544.2       36.0      1.5063            1.5641      1.4700     0.4714                 NaN
              MC_donors   EXPANDED_SHUFFLED [0.4, 0.6)  1441.6       50.8      1.2230            1.2625      1.0884     0.6557                 NaN
              MC_donors   EXPANDED_SHUFFLED [0.6, 0.8)  3262.2       65.2      1.2031            1.0467      0.8414     0.5613                 NaN
```

## How to break this result

* **The weighting confound.** The primary fit weights training rows so every ECFP cluster carries equal total weight, and EXPANDED has more clusters than BASE — so the arms differ in their weighting as well as their chemistry. Re-run with `--weighting row` and compare; if the effect vanishes, it was the weighting.
* **Seeds are not replicates.** The five split seeds re-partition the same ligands. The bootstrap over super-clusters is the load-bearing statistic; seed agreement is a second requirement, not independent evidence — and a weak one: the seed-to-seed sd of the macro delta is ~0.012 against a cluster-robust standard error of ~0.076, a variance component 36x smaller.
* **Multiplicity.** A1-A4 are pre-registered and are the only protected claims. The supporting tables report many more contrasts (comparisons x statistics x endpoints x feature sets), none of them corrected for multiple comparisons. Treat any interval outside A1-A4 as descriptive.
* **The gain is concentrated, not broad — read the decomposition above.** The shared cohort is `min_cells = 3`, so the sparse ligands are *test* rows too, and they carry the effect. BASE trains on none of them in any fold, so on the clusters made only of added chemistry the contrast is close to 'the arm allowed to learn this chemistry predicts it better'. What separates it from a tautology is that a test ligand's own chemotype is held out of every arm, and that the gain scales with how much closer the addition actually brought training (the `closeness gained` table). An earlier version of this line claimed the opposite — that the numbers say nothing about sparse test rows — which was backwards.
* **`EXPANDED_SHUFFLED` is a HARDER comparator than BASE, not an equal one.** Mislabelled chemistry is worse than absent chemistry, so the shuffled arm lands below BASE and the EXPANDED-vs-SHUFFLED delta overstates the effect size. Use it to rule out regularisation and the weighting change; quote the BASE contrast for the magnitude.
* **Do not quote the percentile interval alone at the `all` endpoint.** One super-cluster holds ~21 % of the scoring units inside 1 of the bootstrap blocks, which inflates the percentile interval's real Type-I rate; BCa, cluster-robust and block-macro columns are reported beside it for that reason.

