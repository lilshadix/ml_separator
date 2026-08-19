# gen6 Experiment A — BASE91 vs EXPANDED152 on identical test rows

**Run** `gen6_diversity_causal_20260819T103129Z`

## What was actually done

* One shared cohort, built once at `min_cells = 3`: **5248 rows, 152 extractants, 131 ECFP clusters, 79 Tanimoto-0.7 super-clusters**, 2055 conditions, 14 metals; target sd 1.657 log units.
* Folds hold out whole super-clusters (5 folds, seeds [104729, 130363, 155921, 196613, 262147]). Every arm is scored on **byte-identical test rows**; only the training row mask changes.
* Learner: ExtraTrees, 400 trees, max_features 0.3, min_samples_leaf 2, fold seed `model_seed + fold*1009 + 9999991` (the gen5 formula, reused so a gen5 arm reproduces rather than approximates).
* Weighting: **row** — no sample weights at all — the disclosed sensitivity that removes the cluster-weighting confound
* Feature sets: MC_lig2d_ext_massaction. Fitting took 101 s.

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
MC_lig2d_ext_massaction      all        mae              0.1635           0.0825            0.2830            0.1635               5        5                 131             79
MC_lig2d_ext_massaction      all offset_mae              0.1688           0.0863            0.2898            0.1688               5        5                 131             79
MC_lig2d_ext_massaction      all  shape_mae              0.0240           0.0027            0.0528            0.0240               5        5                 131             79
MC_lig2d_ext_massaction   nn<0.4        mae              0.3712           0.1826            0.5754            0.4117               5        5                  42             37
MC_lig2d_ext_massaction   nn<0.4 offset_mae              0.3783           0.1773            0.5933            0.4136               5        5                  42             37
MC_lig2d_ext_massaction   nn<0.4  shape_mae              0.0643           0.0334            0.1041            0.0746               5        5                  42             37
MC_lig2d_ext_massaction   nn<0.6        mae              0.2014           0.0948            0.3614            0.2465               5        5                 105             65
MC_lig2d_ext_massaction   nn<0.6 offset_mae              0.2069           0.0996            0.3674            0.2515               5        5                 105             65
MC_lig2d_ext_massaction   nn<0.6  shape_mae              0.0262           0.0012            0.0616            0.0425               5        5                 105             65
```

The same contrast one split partition at a time. The five seeds re-partition the same ligands, so agreement here is a consistency requirement and not five independent experiments; the interval above is the evidence.

```
            feature_set endpoint  split_seed  statistic  point_delta  ci95_low  ci95_high  units_improved  units_total
MC_lig2d_ext_massaction      all      104729        mae       0.1375    0.0573     0.2527              83          131
MC_lig2d_ext_massaction      all      104729 offset_mae       0.1319    0.0443     0.2423              82          131
MC_lig2d_ext_massaction      all      104729  shape_mae       0.0205    0.0001     0.0487              67          131
MC_lig2d_ext_massaction   nn<0.4      104729        mae       0.4005    0.1528     0.6392              25           33
MC_lig2d_ext_massaction   nn<0.4      104729 offset_mae       0.3572    0.0891     0.6223              23           33
MC_lig2d_ext_massaction   nn<0.4      104729  shape_mae       0.0710    0.0242     0.1239              23           33
MC_lig2d_ext_massaction   nn<0.6      104729        mae       0.1967    0.0812     0.3677              59           88
MC_lig2d_ext_massaction   nn<0.6      104729 offset_mae       0.1840    0.0591     0.3525              58           88
MC_lig2d_ext_massaction   nn<0.6      104729  shape_mae       0.0347    0.0060     0.0760              48           88
MC_lig2d_ext_massaction      all      130363        mae       0.1385    0.0431     0.2695              72          131
MC_lig2d_ext_massaction      all      130363 offset_mae       0.1370    0.0350     0.2809              71          131
MC_lig2d_ext_massaction      all      130363  shape_mae       0.0273    0.0031     0.0599              74          131
MC_lig2d_ext_massaction   nn<0.4      130363        mae       0.3444    0.1524     0.5535              25           37
MC_lig2d_ext_massaction   nn<0.4      130363 offset_mae       0.3240    0.1104     0.5520              23           37
MC_lig2d_ext_massaction   nn<0.4      130363  shape_mae       0.0765    0.0364     0.1213              26           37
MC_lig2d_ext_massaction   nn<0.6      130363        mae       0.1893    0.0628     0.3623              47           85
MC_lig2d_ext_massaction   nn<0.6      130363 offset_mae       0.1902    0.0603     0.3717              47           85
MC_lig2d_ext_massaction   nn<0.6      130363  shape_mae       0.0422    0.0151     0.0806              53           85
MC_lig2d_ext_massaction      all      155921        mae       0.1682    0.0678     0.3129              70          131
MC_lig2d_ext_massaction      all      155921 offset_mae       0.1794    0.0787     0.3300              69          131
MC_lig2d_ext_massaction      all      155921  shape_mae       0.0197   -0.0039     0.0516              75          131
MC_lig2d_ext_massaction   nn<0.4      155921        mae       0.4594    0.2222     0.7215              29           38
MC_lig2d_ext_massaction   nn<0.4      155921 offset_mae       0.4909    0.2370     0.7766              27           38
MC_lig2d_ext_massaction   nn<0.4      155921  shape_mae       0.0486    0.0096     0.0927              26           38
MC_lig2d_ext_massaction   nn<0.6      155921        mae       0.2533    0.1120     0.4541              50           86
MC_lig2d_ext_massaction   nn<0.6      155921 offset_mae       0.2666    0.1207     0.4760              49           86
MC_lig2d_ext_massaction   nn<0.6      155921  shape_mae       0.0344    0.0057     0.0719              53           86
MC_lig2d_ext_massaction      all      196613        mae       0.1739    0.0957     0.2726              77          131
MC_lig2d_ext_massaction      all      196613 offset_mae       0.1805    0.0915     0.2816              81          131
MC_lig2d_ext_massaction      all      196613  shape_mae       0.0301    0.0074     0.0624              70          131
MC_lig2d_ext_massaction   nn<0.4      196613        mae       0.4055    0.1769     0.6446              27           35
MC_lig2d_ext_massaction   nn<0.4      196613 offset_mae       0.4189    0.1755     0.6679              26           35
MC_lig2d_ext_massaction   nn<0.4      196613  shape_mae       0.1023    0.0540     0.1587              26           35
MC_lig2d_ext_massaction   nn<0.6      196613        mae       0.3024    0.1743     0.4456              51           75
MC_lig2d_ext_massaction   nn<0.6      196613 offset_mae       0.3105    0.1695     0.4615              49           75
MC_lig2d_ext_massaction   nn<0.6      196613  shape_mae       0.0581    0.0281     0.1010              46           75
MC_lig2d_ext_massaction      all      262147        mae       0.1994    0.1000     0.3369              80          131
MC_lig2d_ext_massaction      all      262147 offset_mae       0.2150    0.1126     0.3580              84          131
MC_lig2d_ext_massaction      all      262147  shape_mae       0.0222   -0.0002     0.0529              71          131
MC_lig2d_ext_massaction   nn<0.4      262147        mae       0.4485    0.2209     0.6797              29           37
MC_lig2d_ext_massaction   nn<0.4      262147 offset_mae       0.4769    0.2231     0.7317              25           37
MC_lig2d_ext_massaction   nn<0.4      262147  shape_mae       0.0746    0.0344     0.1210              27           37
MC_lig2d_ext_massaction   nn<0.6      262147        mae       0.2906    0.1563     0.4700              58           85
MC_lig2d_ext_massaction   nn<0.6      262147 offset_mae       0.3062    0.1634     0.4997              59           85
MC_lig2d_ext_massaction   nn<0.6      262147  shape_mae       0.0432    0.0135     0.0841              53           85
```

## Pre-registered hypotheses (protocol §5)

Scoring rule, stated so a verdict cannot be over-read: **PASS** = the pre-registered condition is met, including the seed-agreement clause; **FAIL** = the CI95 excludes the predicted direction (evidence *against*); **INCONCLUSIVE** = the interval straddles zero, or the seed clause cannot be evaluated (a single-seed pilot). An INCONCLUSIVE is not a weak PASS.

### Feature set `MC_lig2d_ext_massaction`

**A1 — PASS**  ·  Adding sparse diverse chemistry improves overall accuracy.

* pass condition: macro_MAE(BASE) − macro_MAE(EXPANDED) > 0 with CI95 low > 0, and ≥ 4/5 seeds positive.
* measured: macro MAE (one ECFP cluster = one vote), all test rows: Δ = +0.1635 log units (CI95 [+0.0825, +0.2830], 131 units, 79 bootstrap blocks; mean over seeds +0.1635, 5/5 seeds positive)
* what would falsify this: A CI95 whose upper end is ≤ 0 — EXPANDED no better, or worse, than BASE on identical test rows — falsifies it, and with it the claim that coverage is the binding constraint at this scale (protocol §8 case C).

**A2 — PASS**  ·  The gain is largest on the hardest chemistry.

* pass condition: gain at nn<0.4 > gain overall, both CI95 low > 0.
* measured: gain at nn<0.4: Δ = +0.3712 log units (CI95 [+0.1826, +0.5754], 42 units, 37 bootstrap blocks; mean over seeds +0.4117, 5/5 seeds positive); gain overall: Δ = +0.1635 log units (CI95 [+0.0825, +0.2830], 131 units, 79 bootstrap blocks; mean over seeds +0.1635, 5/5 seeds positive); hard > overall: True
* what would falsify this: A gain that is flat or smaller on the far-from-training subset falsifies it: the model would then be improving where it was already supported, i.e. a depth effect wearing a coverage costume.

**A3 — PASS**  ·  The gain is in the level, not the shape.

* pass condition: offset_mae(BASE) − offset_mae(EXPANDED) CI95 low > 0, and larger than the shape-MAE gain.
* measured: offset gain: Δ = +0.1688 log units (CI95 [+0.0863, +0.2898], 131 units, 79 bootstrap blocks; mean over seeds +0.1688, 5/5 seeds positive); shape gain: Δ = +0.0240 log units (CI95 [+0.0027, +0.0528], 131 units, 79 bootstrap blocks; mean over seeds +0.0240, 5/5 seeds positive); offset > shape: True
* what would falsify this: A gain that lives in the shape while the per-ligand offset is unchanged falsifies it, and points at the representation rather than the coverage (protocol §8 case B).

**A4 — PASS**  ·  The gain is information, not row count.

* pass condition: EXPANDED beats EXPANDED_ROWMATCHED by less than it beats BASE, and beats EXPANDED_SHUFFLED with CI95 low > 0.
* measured: vs the information null (sparse targets permuted among themselves): Δ = +0.1922 log units (CI95 [+0.1069, +0.3179], 131 units, 79 bootstrap blocks; mean over seeds +0.1922, 5/5 seeds positive); vs the row-budget-matched control: Δ = +0.0022 log units (CI95 [-0.0037, +0.0080], 131 units, 79 bootstrap blocks; mean over seeds +0.0022, 3/5 seeds positive); rowmatched gap < BASE gap: True
* what would falsify this: If EXPANDED_SHUFFLED matches EXPANDED, the extra rows helped as regularisation or through the change in cluster weighting and carried no chemical information; if EXPANDED beats EXPANDED_ROWMATCHED by as much as it beats BASE, the effect is row count, not coverage.

## Where the gain sits — read this before the leaderboard

One ECFP cluster is one vote, and on this cohort a large share of the clusters consist **entirely** of the sparsely-measured ligands that only EXPANDED can train on. If the gain lives only there, the honest claim is 'the expansion repairs chemistry that was missing', not 'the model got better'. `units` counts scoring clusters; `gain` is BASE minus EXPANDED macro MAE within that group.

```
            feature_set endpoint                 kind  units    gain  units_improved
MC_lig2d_ext_massaction      all   BASE-eligible only   72.0  0.0332          0.4944
MC_lig2d_ext_massaction      all added chemistry only   57.0  0.3374          0.7053
MC_lig2d_ext_massaction      all                mixed    2.0 -0.1030          0.3000
MC_lig2d_ext_massaction   nn<0.4   BASE-eligible only    9.2  0.1420          0.5652
MC_lig2d_ext_massaction   nn<0.4 added chemistry only   26.8  0.5050          0.8134
```

## Arm leaderboard (per feature set, mean over seeds)

Macro MAE is primary. Pooled numbers are printed and never used for selection: the largest single extractant holds 28 % of rows and the largest ECFP cluster 38 %.

```
            feature_set                 arm  macro_mae  offset_mae  shape_mae  shape_r2  pooled_mae  median_ligand_mae  worst_quartile_ligand_mae  frac_within_1_log  n_ligands  n_ecfp_clusters  n_superclusters  n_macro_units  n_eff_pooled_rows
MC_lig2d_ext_massaction                BASE     1.2074      1.0540     0.5222    0.1730      1.2013             1.0452                     2.3102             0.4927      152.0            131.0             79.0          131.0             6.2561
MC_lig2d_ext_massaction            EXPANDED     1.0439      0.9079     0.5033    0.1708      1.1632             0.9236                     2.0866             0.5072      152.0            131.0             79.0          131.0             6.2561
MC_lig2d_ext_massaction EXPANDED_ROWMATCHED     1.0460      0.9079     0.5062    0.1775      1.1586             0.9190                     2.0908             0.5089      152.0            131.0             79.0          131.0             6.2561
MC_lig2d_ext_massaction   EXPANDED_SHUFFLED     1.2361      1.0888     0.5300    0.1779      1.2292             1.0667                     2.3377             0.4796      152.0            131.0             79.0          131.0             6.2561
```

## Hard chemistry — the co-primary endpoint

Bins are cut on `nn_reference_tanimoto`: the maximum Tanimoto from the test ligand to the **BASE training extractants of its own fold**. The same column is used for every arm, so the subset does not move with the arm. `nn_expanded_tanimoto` is carried in the OOF for description only and selects nothing.

```
            feature_set endpoint                 arm  n_rows  n_ligands  n_ecfp_clusters  n_superclusters  macro_mae  offset_mae  shape_mae  frac_within_1_log
MC_lig2d_ext_massaction   nn<0.4                BASE   544.2       36.0             36.0             33.8     1.5285      1.4296     0.4597             0.3952
MC_lig2d_ext_massaction   nn<0.6                BASE  1985.8       86.8             83.8             55.8     1.3990      1.2618     0.5365             0.4834
MC_lig2d_ext_massaction      all                BASE  5248.0      152.0            131.0             79.0     1.2074      1.0540     0.5222             0.4927
MC_lig2d_ext_massaction   nn<0.4            EXPANDED   544.2       36.0             36.0             33.8     1.1168      1.0160     0.3851             0.4724
MC_lig2d_ext_massaction   nn<0.6            EXPANDED  1985.8       86.8             83.8             55.8     1.1525      1.0196     0.4963             0.4992
MC_lig2d_ext_massaction      all            EXPANDED  5248.0      152.0            131.0             79.0     1.0439      0.9079     0.5033             0.5072
MC_lig2d_ext_massaction   nn<0.4 EXPANDED_ROWMATCHED   544.2       36.0             36.0             33.8     1.1105      1.0110     0.3848             0.4802
MC_lig2d_ext_massaction   nn<0.6 EXPANDED_ROWMATCHED  1985.8       86.8             83.8             55.8     1.1561      1.0246     0.4973             0.5008
MC_lig2d_ext_massaction      all EXPANDED_ROWMATCHED  5248.0      152.0            131.0             79.0     1.0460      0.9079     0.5062             0.5089
MC_lig2d_ext_massaction   nn<0.4   EXPANDED_SHUFFLED   544.2       36.0             36.0             33.8     1.6184      1.5499     0.4264             0.3599
MC_lig2d_ext_massaction   nn<0.6   EXPANDED_SHUFFLED  1985.8       86.8             83.8             55.8     1.4291      1.3044     0.5253             0.4565
MC_lig2d_ext_massaction      all   EXPANDED_SHUFFLED  5248.0      152.0            131.0             79.0     1.2361      1.0888     0.5300             0.4796
```

Chemistry units surviving each endpoint (this is the n behind every hard-chemistry CI):

```
            feature_set endpoint  split_seed  n_rows  n_ligands  n_ecfp_clusters  n_superclusters
MC_lig2d_ext_massaction      all    169954.6  5248.0      152.0            131.0             79.0
MC_lig2d_ext_massaction   nn<0.4    169954.6   544.2       36.0             36.0             33.8
MC_lig2d_ext_massaction   nn<0.6    169954.6  1985.8       86.8             83.8             55.8
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
MC_lig2d_ext_massaction      all                EXPANDED_vs_BASE        mae              0.1635           0.0825            0.2830            0.1635               5        5                 131           True
MC_lig2d_ext_massaction      all                EXPANDED_vs_BASE offset_mae              0.1688           0.0863            0.2898            0.1688               5        5                 131           True
MC_lig2d_ext_massaction      all                EXPANDED_vs_BASE  shape_mae              0.0240           0.0027            0.0528            0.0240               5        5                 131           True
MC_lig2d_ext_massaction      all     EXPANDED_ROWMATCHED_vs_BASE        mae              0.1614           0.0796            0.2801            0.1614               5        5                 131           True
MC_lig2d_ext_massaction      all     EXPANDED_ROWMATCHED_vs_BASE offset_mae              0.1687           0.0872            0.2888            0.1687               5        5                 131           True
MC_lig2d_ext_massaction      all     EXPANDED_ROWMATCHED_vs_BASE  shape_mae              0.0211          -0.0024            0.0520            0.0211               5        5                 131           True
MC_lig2d_ext_massaction      all   EXPANDED_vs_EXPANDED_SHUFFLED        mae              0.1922           0.1069            0.3179            0.1922               5        5                 131           True
MC_lig2d_ext_massaction      all   EXPANDED_vs_EXPANDED_SHUFFLED offset_mae              0.2025           0.1146            0.3283            0.2025               5        5                 131           True
MC_lig2d_ext_massaction      all   EXPANDED_vs_EXPANDED_SHUFFLED  shape_mae              0.0286           0.0128            0.0497            0.0286               5        5                 131           True
MC_lig2d_ext_massaction      all       EXPANDED_SHUFFLED_vs_BASE        mae             -0.0287          -0.0658            0.0090           -0.0287               0        5                 131           True
MC_lig2d_ext_massaction      all       EXPANDED_SHUFFLED_vs_BASE offset_mae             -0.0338          -0.0760            0.0075           -0.0338               0        5                 131           True
MC_lig2d_ext_massaction      all       EXPANDED_SHUFFLED_vs_BASE  shape_mae             -0.0047          -0.0211            0.0173           -0.0047               2        5                 131           True
MC_lig2d_ext_massaction      all EXPANDED_vs_EXPANDED_ROWMATCHED        mae              0.0022          -0.0037            0.0080            0.0022               3        5                 131          False
MC_lig2d_ext_massaction      all EXPANDED_vs_EXPANDED_ROWMATCHED offset_mae              0.0001          -0.0058            0.0067            0.0001               3        5                 131          False
MC_lig2d_ext_massaction      all EXPANDED_vs_EXPANDED_ROWMATCHED  shape_mae              0.0028          -0.0010            0.0060            0.0028               5        5                 131          False
MC_lig2d_ext_massaction   nn<0.4                EXPANDED_vs_BASE        mae              0.3712           0.1826            0.5754            0.4117               5        5                  42           True
MC_lig2d_ext_massaction   nn<0.4                EXPANDED_vs_BASE offset_mae              0.3783           0.1773            0.5933            0.4136               5        5                  42           True
MC_lig2d_ext_massaction   nn<0.4                EXPANDED_vs_BASE  shape_mae              0.0643           0.0334            0.1041            0.0746               5        5                  42           True
MC_lig2d_ext_massaction   nn<0.4     EXPANDED_ROWMATCHED_vs_BASE        mae              0.3811           0.1927            0.5857            0.4180               5        5                  42           True
MC_lig2d_ext_massaction   nn<0.4     EXPANDED_ROWMATCHED_vs_BASE offset_mae              0.3861           0.1868            0.6010            0.4186               5        5                  42           True
MC_lig2d_ext_massaction   nn<0.4     EXPANDED_ROWMATCHED_vs_BASE  shape_mae              0.0644           0.0321            0.1045            0.0749               5        5                  42           True
MC_lig2d_ext_massaction   nn<0.4   EXPANDED_vs_EXPANDED_SHUFFLED        mae              0.4361           0.2154            0.6902            0.5016               5        5                  42           True
MC_lig2d_ext_massaction   nn<0.4   EXPANDED_vs_EXPANDED_SHUFFLED offset_mae              0.4741           0.2384            0.7347            0.5339               5        5                  42           True
MC_lig2d_ext_massaction   nn<0.4   EXPANDED_vs_EXPANDED_SHUFFLED  shape_mae              0.0293          -0.0006            0.0605            0.0412               5        5                  42           True
MC_lig2d_ext_massaction   nn<0.4       EXPANDED_SHUFFLED_vs_BASE        mae             -0.0649          -0.1594            0.0180           -0.0899               0        5                  42           True
MC_lig2d_ext_massaction   nn<0.4       EXPANDED_SHUFFLED_vs_BASE offset_mae             -0.0957          -0.1967           -0.0050           -0.1203               0        5                  42           True
MC_lig2d_ext_massaction   nn<0.4       EXPANDED_SHUFFLED_vs_BASE  shape_mae              0.0351           0.0033            0.0731            0.0334               4        5                  42           True
MC_lig2d_ext_massaction   nn<0.4 EXPANDED_vs_EXPANDED_ROWMATCHED        mae             -0.0099          -0.0228            0.0035           -0.0064               2        5                  42          False
MC_lig2d_ext_massaction   nn<0.4 EXPANDED_vs_EXPANDED_ROWMATCHED offset_mae             -0.0077          -0.0219            0.0070           -0.0050               3        5                  42          False
MC_lig2d_ext_massaction   nn<0.4 EXPANDED_vs_EXPANDED_ROWMATCHED  shape_mae             -0.0001          -0.0037            0.0039           -0.0003               3        5                  42          False
MC_lig2d_ext_massaction   nn<0.6                EXPANDED_vs_BASE        mae              0.2014           0.0948            0.3614            0.2465               5        5                 105           True
MC_lig2d_ext_massaction   nn<0.6                EXPANDED_vs_BASE offset_mae              0.2069           0.0996            0.3674            0.2515               5        5                 105           True
MC_lig2d_ext_massaction   nn<0.6                EXPANDED_vs_BASE  shape_mae              0.0262           0.0012            0.0616            0.0425               5        5                 105           True
MC_lig2d_ext_massaction   nn<0.6     EXPANDED_ROWMATCHED_vs_BASE        mae              0.1995           0.0932            0.3605            0.2429               5        5                 105           True
MC_lig2d_ext_massaction   nn<0.6     EXPANDED_ROWMATCHED_vs_BASE offset_mae              0.2039           0.0968            0.3648            0.2473               5        5                 105           True
MC_lig2d_ext_massaction   nn<0.6     EXPANDED_ROWMATCHED_vs_BASE  shape_mae              0.0265          -0.0001            0.0633            0.0419               5        5                 105           True
MC_lig2d_ext_massaction   nn<0.6   EXPANDED_vs_EXPANDED_SHUFFLED        mae              0.2366           0.1316            0.3974            0.2766               5        5                 105           True
MC_lig2d_ext_massaction   nn<0.6   EXPANDED_vs_EXPANDED_SHUFFLED offset_mae              0.2560           0.1481            0.4212            0.2936               5        5                 105           True
MC_lig2d_ext_massaction   nn<0.6   EXPANDED_vs_EXPANDED_SHUFFLED  shape_mae              0.0265           0.0104            0.0503            0.0298               5        5                 105           True
MC_lig2d_ext_massaction   nn<0.6       EXPANDED_SHUFFLED_vs_BASE        mae             -0.0351          -0.0853            0.0205           -0.0301               0        5                 105           True
MC_lig2d_ext_massaction   nn<0.6       EXPANDED_SHUFFLED_vs_BASE offset_mae             -0.0491          -0.1043            0.0092           -0.0421               0        5                 105           True
MC_lig2d_ext_massaction   nn<0.6       EXPANDED_SHUFFLED_vs_BASE  shape_mae             -0.0003          -0.0208            0.0279            0.0127               4        5                 105           True
MC_lig2d_ext_massaction   nn<0.6 EXPANDED_vs_EXPANDED_ROWMATCHED        mae              0.0019          -0.0067            0.0097            0.0036               3        5                 105          False
MC_lig2d_ext_massaction   nn<0.6 EXPANDED_vs_EXPANDED_ROWMATCHED offset_mae              0.0030          -0.0056            0.0107            0.0042               3        5                 105          False
MC_lig2d_ext_massaction   nn<0.6 EXPANDED_vs_EXPANDED_ROWMATCHED  shape_mae             -0.0003          -0.0042            0.0027            0.0007               3        5                 105          False
```

Per-seed intervals (the same contrasts, one split partition at a time) are in `contrasts.csv`.

## Out-of-distribution behaviour

Error against distance to the fold's BASE training chemistry, with the forest's own dispersion (sd across trees) where it was recorded. A model that knows where it is unsupported is worth more than one that always emits a number. `mean_prediction_sd` is recorded for **no arm only** and is blank elsewhere — it is a property of one fitted forest, not a comparison between arms.

```
            feature_set                 arm        bin  n_rows  n_ligands  pooled_mae  macro_mae_ligand  offset_mae  shape_mae  mean_prediction_sd
MC_lig2d_ext_massaction                BASE [0.0, 0.4)   544.2       36.0      1.4011            1.5285      1.4296     0.4597                 NaN
MC_lig2d_ext_massaction                BASE [0.4, 0.6)  1441.6       50.8      1.2125            1.2981      1.1430     0.5897                 NaN
MC_lig2d_ext_massaction                BASE [0.6, 0.8)  3262.2       65.2      1.1633            0.9630      0.7778     0.5014                 NaN
MC_lig2d_ext_massaction            EXPANDED [0.0, 0.4)   544.2       36.0      1.2182            1.1168      1.0160     0.3851                 NaN
MC_lig2d_ext_massaction            EXPANDED [0.4, 0.6)  1441.6       50.8      1.2046            1.1816      1.0219     0.5742                 NaN
MC_lig2d_ext_massaction            EXPANDED [0.6, 0.8)  3262.2       65.2      1.1365            0.9477      0.7564     0.5102                 NaN
MC_lig2d_ext_massaction EXPANDED_ROWMATCHED [0.0, 0.4)   544.2       36.0      1.1993            1.1105      1.0110     0.3848                 NaN
MC_lig2d_ext_massaction EXPANDED_ROWMATCHED [0.4, 0.6)  1441.6       50.8      1.2131            1.1939      1.0339     0.5766                 NaN
MC_lig2d_ext_massaction EXPANDED_ROWMATCHED [0.6, 0.8)  3262.2       65.2      1.1291            0.9483      0.7508     0.5160                 NaN
MC_lig2d_ext_massaction   EXPANDED_SHUFFLED [0.0, 0.4)   544.2       36.0      1.5256            1.6184      1.5499     0.4264                 NaN
MC_lig2d_ext_massaction   EXPANDED_SHUFFLED [0.4, 0.6)  1441.6       50.8      1.2687            1.2818      1.1281     0.5956                 NaN
MC_lig2d_ext_massaction   EXPANDED_SHUFFLED [0.6, 0.8)  3262.2       65.2      1.1613            0.9921      0.8006     0.5344                 NaN
```

## How to break this result

* **The weighting confound.** The primary fit weights training rows so every ECFP cluster carries equal total weight, and EXPANDED has more clusters than BASE — so the arms differ in their weighting as well as their chemistry. Re-run with `--weighting row` and compare; if the effect vanishes, it was the weighting.
* **Seeds are not replicates.** The five split seeds re-partition the same ligands. The bootstrap over super-clusters is the load-bearing statistic; seed agreement is a second requirement, not independent evidence — and a weak one: the seed-to-seed sd of the macro delta is ~0.012 against a cluster-robust standard error of ~0.076, a variance component 36x smaller.
* **Multiplicity.** A1-A4 are pre-registered and are the only protected claims. The supporting tables report many more contrasts (comparisons x statistics x endpoints x feature sets), none of them corrected for multiple comparisons. Treat any interval outside A1-A4 as descriptive.
* **The gain is concentrated, not broad — read the decomposition above.** The shared cohort is `min_cells = 3`, so the sparse ligands are *test* rows too, and they carry the effect. BASE trains on none of them in any fold, so on the clusters made only of added chemistry the contrast is close to 'the arm allowed to learn this chemistry predicts it better'. What separates it from a tautology is that a test ligand's own chemotype is held out of every arm, and that the gain scales with how much closer the addition actually brought training (the `closeness gained` table). An earlier version of this line claimed the opposite — that the numbers say nothing about sparse test rows — which was backwards.
* **`EXPANDED_SHUFFLED` is a HARDER comparator than BASE, not an equal one.** Mislabelled chemistry is worse than absent chemistry, so the shuffled arm lands below BASE and the EXPANDED-vs-SHUFFLED delta overstates the effect size. Use it to rule out regularisation and the weighting change; quote the BASE contrast for the magnitude.
* **Do not quote the percentile interval alone at the `all` endpoint.** One super-cluster holds ~21 % of the scoring units inside 1 of the bootstrap blocks, which inflates the percentile interval's real Type-I rate; BCa, cluster-robust and block-macro columns are reported beside it for that reason.

