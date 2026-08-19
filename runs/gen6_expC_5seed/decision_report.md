# gen6 Experiment C — which component of the level fails to transfer?

> **Amendment to this report (2026-08-19, after the run):** C3's verdict was re-scored from the written `contrast_summary.csv` after a logic defect in the runner was fixed — the first draft applied the ≥4/5-seed clause in the *beat* direction to a hypothesis whose PASS side is *no beat*, scoring a 0/5-seed no-beat as INCONCLUSIVE. No number changed; only the verdict label. The corrected runner is `scripts/run_hierarchical_levels.py` at HEAD.

**Run** `gen6_hierarchical_20260819T133008Z`

## What was actually done

* One shared cohort, built once at `min_cells = 3`: **5248 rows, 152 extractants, 131 ECFP clusters, 79 Tanimoto-0.7 chemotypes**, 2055 conditions, 303 series, 14 metals; target sd 1.657 log units.
* Regimes: unseen_chemotype, unseen_ligand, unseen_series — held out on {'unseen_chemotype': 'tanimoto_cluster', 'unseen_ligand': 'ecfp_cluster', 'unseen_series': 'series_id'}. 5 folds × 5 split seeds [104729, 130363, 155921, 196613, 262147].
* Eight models, all fold-local, all scored on byte-identical test rows: MONO_ET, MONO_RIDGE, C1_HIER_RIDGE, C2_TWO_STAGE, C2_TRUECENTRE, C3_SHARED_RIDGE, ORACLE_LEVEL, ORACLE_METAL.
* Forest models: ExtraTrees, 400 trees, max_features 0.3, min_samples_leaf 2, fold seed `model_seed + fold*1009 + 9999991` (the gen5 formula, reused so MONO_ET reproduces Experiment A rather than approximating it). Sample weights: `group_balanced_weights` on the training rows' ECFP cluster, in every model.
* Ridge penalties by inner grouped CV, 3 folds grouped on **the regime's own column** (not always chemotype): λ_fixed ∈ [1.0, 10.0, 100.0], λ_ligand ∈ [0.1, 1.0, 10.0, 100.0, 1000000.0]. The linear design is the compact one (METAL, COND, PHYSCHEM, DONORS, MASSACTION + the declared interaction block); LIG2D_EXT is deliberately **not** in it, so MONO_RIDGE/C1/C3 are a family of their own and are not a like-for-like comparison against MONO_ET.
* Fitting took 649 s.

### Does MONO_ET reproduce Experiment A's EXPANDED arm?

**YES.** Max absolute difference `2.220e-15` over 26240 row_ids, per seed, against `/Users/lilshadix/PycharmProjects/ml_separator/runs/gen6_expA_5seed/oof_predictions.parquet` (feature_set `MC_lig2d_ext_massaction`, arm `EXPANDED`), tolerance 1e-09. Per seed: {"104729": {"ok": true, "n_rows": 5248, "n_unmatched_row_ids": 0, "max_abs_diff": 1.3322676295501878e-15, "n_rows_above_tolerance": 0}, "130363": {"ok": true, "n_rows": 5248, "n_unmatched_row_ids": 0, "max_abs_diff": 1.5543122344752192e-15, "n_rows_above_tolerance": 0}, "155921": {"ok": true, "n_rows": 5248, "n_unmatched_row_ids": 0, "max_abs_diff": 2.220446049250313e-15, "n_rows_above_tolerance": 0}, "196613": {"ok": true, "n_rows": 5248, "n_unmatched_row_ids": 0, "max_abs_diff": 1.7763568394002505e-15, "n_rows_above_tolerance": 0}, "262147": {"ok": true, "n_rows": 5248, "n_unmatched_row_ids": 0, "max_abs_diff": 1.7763568394002505e-15, "n_rows_above_tolerance": 0}}

## Leaderboard — one row per model per regime (mean over seeds)

Macro MAE is primary: one ECFP cluster, one vote. `ORACLE_LEVEL` and `ORACLE_METAL` read test labels and are **not models** — they are there to be subtracted from each other. Pooled numbers are printed and never used for selection.

```
          regime             arm  macro_mae  offset_mae  shape_mae  shape_r2  pooled_mae  median_ligand_mae  worst_quartile_ligand_mae  frac_within_1_log  n_ligands  n_ecfp_clusters  n_superclusters  n_macro_units  n_eff_pooled_rows
unseen_chemotype         MONO_ET     1.0468      0.8744     0.5069    0.1621      1.1340             0.9027                     2.0934             0.5179      152.0            131.0             79.0          131.0             6.2561
unseen_chemotype      MONO_RIDGE     1.1476      0.9364     0.6065   -0.0061      1.3296             1.0929                     2.0852             0.4457      152.0            131.0             79.0          131.0             6.2561
unseen_chemotype   C1_HIER_RIDGE     1.1522      0.9482     0.5985    0.0396      1.3233             1.1030                     2.1020             0.4500      152.0            131.0             79.0          131.0             6.2561
unseen_chemotype    C2_TWO_STAGE     1.1910      1.0693     0.5014    0.1566      1.1245             0.9771                     2.5128             0.5410      152.0            131.0             79.0          131.0             6.2561
unseen_chemotype   C2_TRUECENTRE     1.0433      0.8808     0.5047    0.1577      1.1277             0.9214                     2.0556             0.5228      152.0            131.0             79.0          131.0             6.2561
unseen_chemotype C3_SHARED_RIDGE     1.1513      0.9495     0.5958    0.0433      1.3213             1.1054                     2.0868             0.4497      152.0            131.0             79.0          131.0             6.2561
unseen_chemotype    ORACLE_LEVEL     0.2213      0.0728     0.1848    0.8760      0.2860             0.1724                     0.5324             0.9587      152.0            131.0             79.0          131.0             6.2561
unseen_chemotype    ORACLE_METAL     0.9909      0.8834     0.3653    0.2475      1.0844             0.8782                     2.0463             0.5327      152.0            131.0             79.0          131.0             6.2561
   unseen_ligand         MONO_ET     0.8526      0.6566     0.4477    0.2618      0.9624             0.7209                     1.6907             0.5999      152.0            131.0             79.0          131.0             6.2561
   unseen_ligand      MONO_RIDGE     1.0371      0.7761     0.5992    0.0051      1.1904             0.9900                     1.8420             0.4829      152.0            131.0             79.0          131.0             6.2561
   unseen_ligand   C1_HIER_RIDGE     1.0477      0.7955     0.5981    0.0133      1.1993             0.9933                     1.8754             0.4798      152.0            131.0             79.0          131.0             6.2561
   unseen_ligand    C2_TWO_STAGE     0.9550      0.8121     0.4358    0.3002      0.9836             0.8152                     1.9458             0.5881      152.0            131.0             79.0          131.0             6.2561
   unseen_ligand   C2_TRUECENTRE     0.8753      0.6950     0.4413    0.2999      0.9986             0.7728                     1.7177             0.5803      152.0            131.0             79.0          131.0             6.2561
   unseen_ligand C3_SHARED_RIDGE     1.0466      0.7963     0.5952    0.0153      1.1990             0.9997                     1.8682             0.4798      152.0            131.0             79.0          131.0             6.2561
   unseen_ligand    ORACLE_LEVEL     0.2129      0.0951     0.1551    0.9020      0.2795             0.1617                     0.4967             0.9771      152.0            131.0             79.0          131.0             6.2561
   unseen_ligand    ORACLE_METAL     0.8208      0.6768     0.3331    0.3503      0.9693             0.6948                     1.6835             0.5989      152.0            131.0             79.0          131.0             6.2561
   unseen_series         MONO_ET     0.8080      0.5439     0.4661    0.2770      0.8706             0.6093                     1.6741             0.6574      152.0            131.0             79.0          131.0             6.2561
   unseen_series      MONO_RIDGE     1.0224      0.7528     0.5994    0.0455      1.1848             0.9641                     1.8251             0.4869      152.0            131.0             79.0          131.0             6.2561
   unseen_series   C1_HIER_RIDGE     1.0328      0.7266     0.6054    0.0720      1.0917             0.9308                     1.9146             0.5154      152.0            131.0             79.0          131.0             6.2561
   unseen_series    C2_TWO_STAGE     0.9691      0.7278     0.4973    0.1588      0.9703             0.7549                     2.0257             0.6245      152.0            131.0             79.0          131.0             6.2561
   unseen_series   C2_TRUECENTRE     0.8415      0.6023     0.4620    0.2670      0.9019             0.6577                     1.7180             0.6372      152.0            131.0             79.0          131.0             6.2561
   unseen_series C3_SHARED_RIDGE     1.0369      0.7322     0.6041    0.0739      1.0911             0.9460                     1.9162             0.5151      152.0            131.0             79.0          131.0             6.2561
   unseen_series    ORACLE_LEVEL     0.2052      0.0896     0.1488    0.9194      0.2415             0.1513                     0.4731             0.9826      152.0            131.0             79.0          131.0             6.2561
   unseen_series    ORACLE_METAL     0.7886      0.5678     0.3581    0.3092      0.8794             0.5687                     1.6985             0.6482      152.0            131.0             79.0          131.0             6.2561
```

## Hypothesis C1 — the attribution

`point_delta = statistic(reference) − statistic(candidate)`, so **positive = the candidate is better**. The two oracle gains share the same `C2_TWO_STAGE` reference, so their difference is measured directly as `level_minus_metal_gain = (ORACLE_METAL, ORACLE_LEVEL)` and the reference cancels exactly instead of being subtracted by hand from two intervals. The bootstrap scores ECFP clusters and resamples **Tanimoto chemotypes** — the blocks the chemotype folds actually held out.

Rows counted per endpoint (this is the n behind every interval below):

```
          regime          endpoint  split_seed  n_rows  n_ligands  n_cells  n_ecfp_clusters  n_chemotype_blocks
unseen_chemotype               all    169954.6  5248.0      152.0   2405.0            131.0                79.0
unseen_chemotype multi_metal_cells    169954.6  3364.0       90.0    521.0             77.0                45.0
   unseen_ligand               all    169954.6  5248.0      152.0   2405.0            131.0                79.0
   unseen_ligand multi_metal_cells    169954.6  3364.0       90.0    521.0             77.0                45.0
   unseen_series               all    169954.6  5248.0      152.0   2405.0            131.0                79.0
   unseen_series multi_metal_cells    169954.6  3364.0       90.0    521.0             77.0                45.0
```

```
          regime          endpoint             comparison  statistic  pooled_point_delta  pooled_bca_low  pooled_bca_high  pooled_ci95_low  pooled_ci95_high  pooled_cluster_robust_low  pooled_cluster_robust_high  pooled_block_macro_delta  mean_point_delta  seeds_positive  n_seeds  pooled_units_total  pooled_blocks
unseen_chemotype               all      level_oracle_gain        mae              0.9697          0.7935           1.1824           0.7542            1.1435                     0.7634                      1.1761                    0.8993            0.9697               5        5                 131             79
unseen_chemotype               all      level_oracle_gain offset_mae              0.9886          0.8228           1.1998           0.7827            1.1523                     0.7933                      1.1839                    0.9292            0.9886               5        5                 131             79
unseen_chemotype               all      level_oracle_gain  shape_mae              0.3081          0.2352           0.3867           0.2230            0.3744                     0.2292                      0.3871                    0.2734            0.3081               5        5                 131             79
unseen_chemotype               all      metal_oracle_gain        mae              0.2002          0.1099           0.2993           0.0972            0.2848                     0.1092                      0.2911                    0.1625            0.2002               5        5                 131             79
unseen_chemotype               all      metal_oracle_gain offset_mae              0.1824          0.0935           0.2952           0.0830            0.2798                     0.0900                      0.2748                    0.1663            0.1824               5        5                 131             79
unseen_chemotype               all      metal_oracle_gain  shape_mae              0.1343          0.0960           0.1822           0.0923            0.1776                     0.0927                      0.1760                    0.1264            0.1343               5        5                 131             79
unseen_chemotype               all level_minus_metal_gain        mae              0.7696          0.6157           0.9393           0.5839            0.9080                     0.6003                      0.9389                    0.7369            0.7696               5        5                 131             79
unseen_chemotype               all level_minus_metal_gain offset_mae              0.8062          0.6639           0.9714           0.6267            0.9404                     0.6432                      0.9691                    0.7629            0.8062               5        5                 131             79
unseen_chemotype               all level_minus_metal_gain  shape_mae              0.1738          0.0697           0.2666           0.0601            0.2612                     0.0733                      0.2743                    0.1470            0.1738               5        5                 131             79
unseen_chemotype multi_metal_cells      level_oracle_gain        mae              0.6039          0.4855           0.8563           0.4739            0.8117                     0.4378                      0.7701                    0.6252            0.6039               5        5                  77             45
unseen_chemotype multi_metal_cells      level_oracle_gain offset_mae              0.7661          0.6413           1.0293           0.6298            0.9952                     0.5881                      0.9442                    0.7981            0.7661               5        5                  77             45
unseen_chemotype multi_metal_cells      level_oracle_gain  shape_mae              0.0960          0.0542           0.1420           0.0479            0.1359                     0.0510                      0.1409                    0.0905            0.0960               5        5                  77             45
unseen_chemotype multi_metal_cells      metal_oracle_gain        mae              0.2123          0.1313           0.3710           0.1226            0.3488                     0.1021                      0.3225                    0.2213            0.2123               5        5                  77             45
unseen_chemotype multi_metal_cells      metal_oracle_gain offset_mae              0.1441          0.0573           0.3324           0.0501            0.3043                     0.0177                      0.2706                    0.1645            0.1441               5        5                  77             45
unseen_chemotype multi_metal_cells      metal_oracle_gain  shape_mae              0.2784          0.2179           0.3497           0.2112            0.3436                     0.2163                      0.3405                    0.2833            0.2784               5        5                  77             45
unseen_chemotype multi_metal_cells level_minus_metal_gain        mae              0.3916          0.2733           0.5570           0.2638            0.5435                     0.2578                      0.5255                    0.4039            0.3916               5        5                  77             45
unseen_chemotype multi_metal_cells level_minus_metal_gain offset_mae              0.6220          0.5146           0.7741           0.5044            0.7594                     0.5019                      0.7421                    0.6336            0.6220               5        5                  77             45
unseen_chemotype multi_metal_cells level_minus_metal_gain  shape_mae             -0.1824         -0.2654          -0.1013          -0.2674           -0.1025                    -0.2591                     -0.1056                   -0.1928           -0.1824               0        5                  77             45
   unseen_ligand               all      level_oracle_gain        mae              0.7421          0.5867           0.8994           0.6109            0.9350                     0.5753                      0.9088                    0.8744            0.7421               5        5                 131             79
   unseen_ligand               all      level_oracle_gain offset_mae              0.7167          0.5095           0.8956           0.5475            0.9486                     0.5067                      0.9268                    0.8840            0.7167               5        5                 131             79
   unseen_ligand               all      level_oracle_gain  shape_mae              0.2784          0.2199           0.3311           0.2199            0.3311                     0.2247                      0.3320                    0.2807            0.2784               5        5                 131             79
   unseen_ligand               all      metal_oracle_gain        mae              0.1342          0.0764           0.2097           0.0743            0.2056                     0.0721                      0.1962                    0.1388            0.1342               5        5                 131             79
   unseen_ligand               all      metal_oracle_gain offset_mae              0.1362          0.0781           0.2183           0.0736            0.2123                     0.0699                      0.2025                    0.1468            0.1362               5        5                 131             79
   unseen_ligand               all      metal_oracle_gain  shape_mae              0.1062          0.0636           0.1576           0.0658            0.1608                     0.0573                      0.1552                    0.1225            0.1062               5        5                 131             79
   unseen_ligand               all level_minus_metal_gain        mae              0.6079          0.4639           0.7652           0.4823            0.7882                     0.4516                      0.7642                    0.7356            0.6079               5        5                 131             79
   unseen_ligand               all level_minus_metal_gain offset_mae              0.5805          0.3900           0.7470           0.4264            0.7869                     0.3895                      0.7715                    0.7372            0.5805               5        5                 131             79
   unseen_ligand               all level_minus_metal_gain  shape_mae              0.1722          0.0787           0.2522           0.0749            0.2504                     0.0844                      0.2599                    0.1582            0.1722               5        5                 131             79
   unseen_ligand multi_metal_cells      level_oracle_gain        mae              0.5767          0.4384           0.7469           0.4514            0.7639                     0.4170                      0.7364                    0.6396            0.5767               5        5                  77             45
   unseen_ligand multi_metal_cells      level_oracle_gain offset_mae              0.6839          0.4947           0.8735           0.5255            0.9181                     0.4780                      0.8898                    0.7904            0.6839               5        5                  77             45
   unseen_ligand multi_metal_cells      level_oracle_gain  shape_mae              0.1018          0.0593           0.1514           0.0528            0.1464                     0.0549                      0.1488                    0.0982            0.1018               5        5                  77             45
   unseen_ligand multi_metal_cells      metal_oracle_gain        mae              0.1862          0.1033           0.2738           0.1124            0.2872                     0.0984                      0.2740                    0.2253            0.1862               5        5                  77             45
   unseen_ligand multi_metal_cells      metal_oracle_gain offset_mae              0.1198          0.0309           0.2183           0.0403            0.2335                     0.0194                      0.2202                    0.1622            0.1198               5        5                  77             45
   unseen_ligand multi_metal_cells      metal_oracle_gain  shape_mae              0.2370          0.1838           0.3169           0.1841            0.3173                     0.1704                      0.3036                    0.2774            0.2370               5        5                  77             45
   unseen_ligand multi_metal_cells level_minus_metal_gain        mae              0.3905          0.2736           0.5458           0.2699            0.5411                     0.2604                      0.5205                    0.4144            0.3905               5        5                  77             45
   unseen_ligand multi_metal_cells level_minus_metal_gain offset_mae              0.5640          0.4370           0.7177           0.4475            0.7351                     0.4193                      0.7088                    0.6282            0.5640               5        5                  77             45
   unseen_ligand multi_metal_cells level_minus_metal_gain  shape_mae             -0.1352         -0.2261          -0.0499          -0.2427           -0.0604                    -0.2267                     -0.0437                   -0.1792           -0.1352               0        5                  77             45
   unseen_series               all      level_oracle_gain        mae              0.7640          0.6186           0.9258           0.6390            0.9494                     0.6044                      0.9236                    0.8788            0.7640               5        5                 131             79
   unseen_series               all      level_oracle_gain offset_mae              0.6632          0.4711           0.8412           0.5049            0.8894                     0.4648                      0.8617                    0.8310            0.6632               5        5                 131             79
   unseen_series               all      level_oracle_gain  shape_mae              0.3557          0.2796           0.4347           0.2759            0.4296                     0.2801                      0.4313                    0.3460            0.3557               5        5                 131             79
   unseen_series               all      metal_oracle_gain        mae              0.1805          0.1099           0.2499           0.1208            0.2635                     0.1059                      0.2551                    0.2229            0.1805               5        5                 131             79
   unseen_series               all      metal_oracle_gain offset_mae              0.1741          0.1072           0.2522           0.1157            0.2621                     0.0993                      0.2488                    0.2352            0.1741               5        5                 131             79
   unseen_series               all      metal_oracle_gain  shape_mae              0.1480          0.1007           0.2026           0.1039            0.2070                     0.0949                      0.2011                    0.1613            0.1480               5        5                 131             79
   unseen_series               all level_minus_metal_gain        mae              0.5835          0.4798           0.7313           0.4790            0.7282                     0.4613                      0.7056                    0.6560            0.5835               5        5                 131             79
   unseen_series               all level_minus_metal_gain offset_mae              0.4891          0.3461           0.6397           0.3677            0.6700                     0.3364                      0.6418                    0.5958            0.4891               5        5                 131             79
   unseen_series               all level_minus_metal_gain  shape_mae              0.2076          0.1060           0.2974           0.0983            0.2919                     0.1094                      0.3059                    0.1847            0.2076               5        5                 131             79
   unseen_series multi_metal_cells      level_oracle_gain        mae              0.6243          0.4907           0.7719           0.5052            0.7957                     0.4751                      0.7734                    0.6931            0.6243               5        5                  77             45
   unseen_series multi_metal_cells      level_oracle_gain offset_mae              0.6733          0.5044           0.8306           0.5351            0.8672                     0.4959                      0.8506                    0.7609            0.6733               5        5                  77             45
   unseen_series multi_metal_cells      level_oracle_gain  shape_mae              0.1622          0.1027           0.2606           0.0932            0.2392                     0.0898                      0.2347                    0.1759            0.1622               5        5                  77             45
   unseen_series multi_metal_cells      metal_oracle_gain        mae              0.2027          0.0985           0.2879           0.1234            0.3117                     0.1011                      0.3044                    0.2649            0.2027               5        5                  77             45
   unseen_series multi_metal_cells      metal_oracle_gain offset_mae              0.1191          0.0261           0.1965           0.0462            0.2156                     0.0271                      0.2110                    0.1733            0.1191               5        5                  77             45
   unseen_series multi_metal_cells      metal_oracle_gain  shape_mae              0.2616          0.1983           0.3427           0.2037            0.3490                     0.1878                      0.3354                    0.3052            0.2616               5        5                  77             45
   unseen_series multi_metal_cells level_minus_metal_gain        mae              0.4215          0.3059           0.5526           0.3028            0.5474                     0.3065                      0.5365                    0.4282            0.4215               5        5                  77             45
   unseen_series multi_metal_cells level_minus_metal_gain offset_mae              0.5542          0.4486           0.6821           0.4533            0.6925                     0.4374                      0.6711                    0.5876            0.5542               5        5                  77             45
   unseen_series multi_metal_cells level_minus_metal_gain  shape_mae             -0.0994         -0.2005           0.0029          -0.2127           -0.0101                    -0.2021                      0.0034                   -0.1292           -0.0994               0        5                  77             45
```

## Every contrast, every regime, every endpoint

```
          regime          endpoint                       comparison  statistic  pooled_point_delta  pooled_bca_low  pooled_bca_high  pooled_ci95_low  pooled_ci95_high  mean_point_delta  seeds_positive  n_seeds  pooled_units_total  preregistered
unseen_chemotype               all      C1_HIER_RIDGE_vs_MONO_RIDGE        mae             -0.0046         -0.0195           0.0048          -0.0173            0.0058           -0.0046               2        5                 131           True
unseen_chemotype               all      C1_HIER_RIDGE_vs_MONO_RIDGE offset_mae             -0.0093         -0.0247           0.0055          -0.0230            0.0073           -0.0093               3        5                 131           True
unseen_chemotype               all      C1_HIER_RIDGE_vs_MONO_RIDGE  shape_mae              0.0062         -0.0008           0.0173          -0.0022            0.0139            0.0062               4        5                 131           True
unseen_chemotype               all          C2_TWO_STAGE_vs_MONO_ET        mae             -0.1443         -0.2397          -0.0687          -0.2258           -0.0556           -0.1443               0        5                 131           True
unseen_chemotype               all          C2_TWO_STAGE_vs_MONO_ET offset_mae             -0.1834         -0.2915          -0.1026          -0.2786           -0.0899           -0.1834               0        5                 131           True
unseen_chemotype               all          C2_TWO_STAGE_vs_MONO_ET  shape_mae              0.0052         -0.0122           0.0209          -0.0143            0.0196            0.0052               3        5                 131           True
unseen_chemotype               all         C2_TRUECENTRE_vs_MONO_ET        mae              0.0035         -0.0194           0.0331          -0.0260            0.0253            0.0035               3        5                 131          False
unseen_chemotype               all         C2_TRUECENTRE_vs_MONO_ET offset_mae             -0.0011         -0.0242           0.0275          -0.0306            0.0210           -0.0011               3        5                 131          False
unseen_chemotype               all         C2_TRUECENTRE_vs_MONO_ET  shape_mae              0.0040         -0.0061           0.0129          -0.0070            0.0121            0.0040               4        5                 131          False
unseen_chemotype               all C3_SHARED_RIDGE_vs_C1_HIER_RIDGE        mae              0.0008         -0.0106           0.0101          -0.0113            0.0096            0.0008               2        5                 131          False
unseen_chemotype               all C3_SHARED_RIDGE_vs_C1_HIER_RIDGE offset_mae             -0.0018         -0.0145           0.0088          -0.0151            0.0083           -0.0018               2        5                 131          False
unseen_chemotype               all C3_SHARED_RIDGE_vs_C1_HIER_RIDGE  shape_mae              0.0033         -0.0007           0.0082          -0.0015            0.0072            0.0033               5        5                 131          False
unseen_chemotype               all                level_oracle_gain        mae              0.9697          0.7935           1.1824           0.7542            1.1435            0.9697               5        5                 131           True
unseen_chemotype               all                level_oracle_gain offset_mae              0.9886          0.8228           1.1998           0.7827            1.1523            0.9886               5        5                 131           True
unseen_chemotype               all                level_oracle_gain  shape_mae              0.3081          0.2352           0.3867           0.2230            0.3744            0.3081               5        5                 131           True
unseen_chemotype               all                metal_oracle_gain        mae              0.2002          0.1099           0.2993           0.0972            0.2848            0.2002               5        5                 131           True
unseen_chemotype               all                metal_oracle_gain offset_mae              0.1824          0.0935           0.2952           0.0830            0.2798            0.1824               5        5                 131           True
unseen_chemotype               all                metal_oracle_gain  shape_mae              0.1343          0.0960           0.1822           0.0923            0.1776            0.1343               5        5                 131           True
unseen_chemotype               all           level_minus_metal_gain        mae              0.7696          0.6157           0.9393           0.5839            0.9080            0.7696               5        5                 131           True
unseen_chemotype               all           level_minus_metal_gain offset_mae              0.8062          0.6639           0.9714           0.6267            0.9404            0.8062               5        5                 131           True
unseen_chemotype               all           level_minus_metal_gain  shape_mae              0.1738          0.0697           0.2666           0.0601            0.2612            0.1738               5        5                 131           True
unseen_chemotype multi_metal_cells      C1_HIER_RIDGE_vs_MONO_RIDGE        mae             -0.0008         -0.0181           0.0122          -0.0159            0.0137           -0.0008               4        5                  77           True
unseen_chemotype multi_metal_cells      C1_HIER_RIDGE_vs_MONO_RIDGE offset_mae             -0.0042         -0.0245           0.0155          -0.0225            0.0173           -0.0042               3        5                  77           True
unseen_chemotype multi_metal_cells      C1_HIER_RIDGE_vs_MONO_RIDGE  shape_mae              0.0034         -0.0003           0.0081          -0.0011            0.0070            0.0034               3        5                  77           True
unseen_chemotype multi_metal_cells          C2_TWO_STAGE_vs_MONO_ET        mae             -0.1069         -0.2704          -0.0331          -0.2421           -0.0250           -0.1069               0        5                  77           True
unseen_chemotype multi_metal_cells          C2_TWO_STAGE_vs_MONO_ET offset_mae             -0.1420         -0.3077          -0.0451          -0.3059           -0.0436           -0.1420               0        5                  77           True
unseen_chemotype multi_metal_cells          C2_TWO_STAGE_vs_MONO_ET  shape_mae              0.0026         -0.0068           0.0142          -0.0077            0.0133            0.0026               4        5                  77           True
unseen_chemotype multi_metal_cells         C2_TRUECENTRE_vs_MONO_ET        mae              0.0040         -0.0176           0.0254          -0.0175            0.0255            0.0040               3        5                  77          False
unseen_chemotype multi_metal_cells         C2_TRUECENTRE_vs_MONO_ET offset_mae             -0.0021         -0.0248           0.0206          -0.0255            0.0196           -0.0021               1        5                  77          False
unseen_chemotype multi_metal_cells         C2_TRUECENTRE_vs_MONO_ET  shape_mae              0.0040         -0.0143           0.0238          -0.0146            0.0232            0.0040               4        5                  77          False
unseen_chemotype multi_metal_cells C3_SHARED_RIDGE_vs_C1_HIER_RIDGE        mae              0.0017         -0.0066           0.0108          -0.0081            0.0091            0.0017               4        5                  77          False
unseen_chemotype multi_metal_cells C3_SHARED_RIDGE_vs_C1_HIER_RIDGE offset_mae             -0.0078         -0.0180           0.0011          -0.0186            0.0007           -0.0078               0        5                  77          False
unseen_chemotype multi_metal_cells C3_SHARED_RIDGE_vs_C1_HIER_RIDGE  shape_mae              0.0072          0.0006           0.0159          -0.0012            0.0140            0.0072               5        5                  77          False
unseen_chemotype multi_metal_cells                level_oracle_gain        mae              0.6039          0.4855           0.8563           0.4739            0.8117            0.6039               5        5                  77           True
unseen_chemotype multi_metal_cells                level_oracle_gain offset_mae              0.7661          0.6413           1.0293           0.6298            0.9952            0.7661               5        5                  77           True
unseen_chemotype multi_metal_cells                level_oracle_gain  shape_mae              0.0960          0.0542           0.1420           0.0479            0.1359            0.0960               5        5                  77           True
unseen_chemotype multi_metal_cells                metal_oracle_gain        mae              0.2123          0.1313           0.3710           0.1226            0.3488            0.2123               5        5                  77           True
unseen_chemotype multi_metal_cells                metal_oracle_gain offset_mae              0.1441          0.0573           0.3324           0.0501            0.3043            0.1441               5        5                  77           True
unseen_chemotype multi_metal_cells                metal_oracle_gain  shape_mae              0.2784          0.2179           0.3497           0.2112            0.3436            0.2784               5        5                  77           True
unseen_chemotype multi_metal_cells           level_minus_metal_gain        mae              0.3916          0.2733           0.5570           0.2638            0.5435            0.3916               5        5                  77           True
unseen_chemotype multi_metal_cells           level_minus_metal_gain offset_mae              0.6220          0.5146           0.7741           0.5044            0.7594            0.6220               5        5                  77           True
unseen_chemotype multi_metal_cells           level_minus_metal_gain  shape_mae             -0.1824         -0.2654          -0.1013          -0.2674           -0.1025           -0.1824               0        5                  77           True
   unseen_ligand               all      C1_HIER_RIDGE_vs_MONO_RIDGE        mae             -0.0106         -0.0281           0.0013          -0.0281            0.0013           -0.0106               0        5                 131           True
   unseen_ligand               all      C1_HIER_RIDGE_vs_MONO_RIDGE offset_mae             -0.0165         -0.0342          -0.0039          -0.0325           -0.0030           -0.0165               0        5                 131           True
   unseen_ligand               all      C1_HIER_RIDGE_vs_MONO_RIDGE  shape_mae              0.0014         -0.0033           0.0068          -0.0044            0.0058            0.0014               4        5                 131           True
   unseen_ligand               all          C2_TWO_STAGE_vs_MONO_ET        mae             -0.1024         -0.1742          -0.0476          -0.1688           -0.0432           -0.1024               0        5                 131           True
   unseen_ligand               all          C2_TWO_STAGE_vs_MONO_ET offset_mae             -0.1473         -0.2284          -0.0870          -0.2219           -0.0823           -0.1473               0        5                 131           True
   unseen_ligand               all          C2_TWO_STAGE_vs_MONO_ET  shape_mae              0.0080         -0.0122           0.0255          -0.0140            0.0240            0.0080               4        5                 131           True
   unseen_ligand               all         C2_TRUECENTRE_vs_MONO_ET        mae             -0.0227         -0.0396          -0.0061          -0.0375           -0.0037           -0.0227               0        5                 131          False
   unseen_ligand               all         C2_TRUECENTRE_vs_MONO_ET offset_mae             -0.0265         -0.0467          -0.0083          -0.0437           -0.0047           -0.0265               0        5                 131          False
   unseen_ligand               all         C2_TRUECENTRE_vs_MONO_ET  shape_mae              0.0049         -0.0032           0.0139          -0.0037            0.0133            0.0049               4        5                 131          False
   unseen_ligand               all C3_SHARED_RIDGE_vs_C1_HIER_RIDGE        mae              0.0010         -0.0104           0.0102          -0.0106            0.0102            0.0010               4        5                 131          False
   unseen_ligand               all C3_SHARED_RIDGE_vs_C1_HIER_RIDGE offset_mae             -0.0005         -0.0137           0.0110          -0.0132            0.0115           -0.0005               3        5                 131          False
   unseen_ligand               all C3_SHARED_RIDGE_vs_C1_HIER_RIDGE  shape_mae              0.0034         -0.0006           0.0083          -0.0016            0.0073            0.0034               5        5                 131          False
   unseen_ligand               all                level_oracle_gain        mae              0.7421          0.5867           0.8994           0.6109            0.9350            0.7421               5        5                 131           True
   unseen_ligand               all                level_oracle_gain offset_mae              0.7167          0.5095           0.8956           0.5475            0.9486            0.7167               5        5                 131           True
   unseen_ligand               all                level_oracle_gain  shape_mae              0.2784          0.2199           0.3311           0.2199            0.3311            0.2784               5        5                 131           True
   unseen_ligand               all                metal_oracle_gain        mae              0.1342          0.0764           0.2097           0.0743            0.2056            0.1342               5        5                 131           True
   unseen_ligand               all                metal_oracle_gain offset_mae              0.1362          0.0781           0.2183           0.0736            0.2123            0.1362               5        5                 131           True
   unseen_ligand               all                metal_oracle_gain  shape_mae              0.1062          0.0636           0.1576           0.0658            0.1608            0.1062               5        5                 131           True
   unseen_ligand               all           level_minus_metal_gain        mae              0.6079          0.4639           0.7652           0.4823            0.7882            0.6079               5        5                 131           True
   unseen_ligand               all           level_minus_metal_gain offset_mae              0.5805          0.3900           0.7470           0.4264            0.7869            0.5805               5        5                 131           True
   unseen_ligand               all           level_minus_metal_gain  shape_mae              0.1722          0.0787           0.2522           0.0749            0.2504            0.1722               5        5                 131           True
   unseen_ligand multi_metal_cells      C1_HIER_RIDGE_vs_MONO_RIDGE        mae             -0.0076         -0.0339           0.0084          -0.0324            0.0088           -0.0076               1        5                  77           True
   unseen_ligand multi_metal_cells      C1_HIER_RIDGE_vs_MONO_RIDGE offset_mae             -0.0086         -0.0357           0.0096          -0.0341            0.0102           -0.0086               1        5                  77           True
   unseen_ligand multi_metal_cells      C1_HIER_RIDGE_vs_MONO_RIDGE  shape_mae              0.0025         -0.0014           0.0072          -0.0024            0.0062            0.0025               4        5                  77           True
   unseen_ligand multi_metal_cells          C2_TWO_STAGE_vs_MONO_ET        mae             -0.1091         -0.1940          -0.0440          -0.1960           -0.0450           -0.1091               0        5                  77           True
   unseen_ligand multi_metal_cells          C2_TWO_STAGE_vs_MONO_ET offset_mae             -0.1368         -0.2394          -0.0481          -0.2524           -0.0563           -0.1368               0        5                  77           True
   unseen_ligand multi_metal_cells          C2_TWO_STAGE_vs_MONO_ET  shape_mae              0.0012         -0.0085           0.0108          -0.0092            0.0099            0.0012               3        5                  77           True
   unseen_ligand multi_metal_cells         C2_TRUECENTRE_vs_MONO_ET        mae             -0.0107         -0.0315           0.0126          -0.0303            0.0144           -0.0107               1        5                  77          False
   unseen_ligand multi_metal_cells         C2_TRUECENTRE_vs_MONO_ET offset_mae             -0.0244         -0.0485          -0.0026          -0.0443            0.0030           -0.0244               0        5                  77          False
   unseen_ligand multi_metal_cells         C2_TRUECENTRE_vs_MONO_ET  shape_mae              0.0156         -0.0038           0.0338          -0.0058            0.0325            0.0156               5        5                  77          False
   unseen_ligand multi_metal_cells C3_SHARED_RIDGE_vs_C1_HIER_RIDGE        mae              0.0044         -0.0046           0.0175          -0.0066            0.0145            0.0044               4        5                  77          False
   unseen_ligand multi_metal_cells C3_SHARED_RIDGE_vs_C1_HIER_RIDGE offset_mae             -0.0007         -0.0097           0.0157          -0.0114            0.0119           -0.0007               3        5                  77          False
   unseen_ligand multi_metal_cells C3_SHARED_RIDGE_vs_C1_HIER_RIDGE  shape_mae              0.0076          0.0006           0.0170          -0.0013            0.0147            0.0076               5        5                  77          False
   unseen_ligand multi_metal_cells                level_oracle_gain        mae              0.5767          0.4384           0.7469           0.4514            0.7639            0.5767               5        5                  77           True
   unseen_ligand multi_metal_cells                level_oracle_gain offset_mae              0.6839          0.4947           0.8735           0.5255            0.9181            0.6839               5        5                  77           True
   unseen_ligand multi_metal_cells                level_oracle_gain  shape_mae              0.1018          0.0593           0.1514           0.0528            0.1464            0.1018               5        5                  77           True
   unseen_ligand multi_metal_cells                metal_oracle_gain        mae              0.1862          0.1033           0.2738           0.1124            0.2872            0.1862               5        5                  77           True
   unseen_ligand multi_metal_cells                metal_oracle_gain offset_mae              0.1198          0.0309           0.2183           0.0403            0.2335            0.1198               5        5                  77           True
   unseen_ligand multi_metal_cells                metal_oracle_gain  shape_mae              0.2370          0.1838           0.3169           0.1841            0.3173            0.2370               5        5                  77           True
   unseen_ligand multi_metal_cells           level_minus_metal_gain        mae              0.3905          0.2736           0.5458           0.2699            0.5411            0.3905               5        5                  77           True
   unseen_ligand multi_metal_cells           level_minus_metal_gain offset_mae              0.5640          0.4370           0.7177           0.4475            0.7351            0.5640               5        5                  77           True
   unseen_ligand multi_metal_cells           level_minus_metal_gain  shape_mae             -0.1352         -0.2261          -0.0499          -0.2427           -0.0604           -0.1352               0        5                  77           True
   unseen_series               all      C1_HIER_RIDGE_vs_MONO_RIDGE        mae             -0.0105         -0.0485           0.0355          -0.0471            0.0376           -0.0105               1        5                 131           True
   unseen_series               all      C1_HIER_RIDGE_vs_MONO_RIDGE offset_mae              0.0344         -0.0110           0.0870          -0.0102            0.0885            0.0344               4        5                 131           True
   unseen_series               all      C1_HIER_RIDGE_vs_MONO_RIDGE  shape_mae             -0.0141         -0.0327          -0.0001          -0.0314            0.0010           -0.0141               0        5                 131           True
   unseen_series               all          C2_TWO_STAGE_vs_MONO_ET        mae             -0.1611         -0.2231          -0.1123          -0.2263           -0.1147           -0.1611               0        5                 131           True
   unseen_series               all          C2_TWO_STAGE_vs_MONO_ET offset_mae             -0.1954         -0.2666          -0.1422          -0.2715           -0.1457           -0.1954               0        5                 131           True
   unseen_series               all          C2_TWO_STAGE_vs_MONO_ET  shape_mae             -0.0368         -0.0721          -0.0178          -0.0668           -0.0148           -0.0368               0        5                 131           True
   unseen_series               all         C2_TRUECENTRE_vs_MONO_ET        mae             -0.0335         -0.0688          -0.0108          -0.0579           -0.0052           -0.0335               0        5                 131          False
   unseen_series               all         C2_TRUECENTRE_vs_MONO_ET offset_mae             -0.0510         -0.0963          -0.0226          -0.0824           -0.0161           -0.0510               0        5                 131          False
   unseen_series               all         C2_TRUECENTRE_vs_MONO_ET  shape_mae              0.0047         -0.0052           0.0151          -0.0044            0.0162            0.0047               4        5                 131          False
   unseen_series               all C3_SHARED_RIDGE_vs_C1_HIER_RIDGE        mae             -0.0041         -0.0170           0.0074          -0.0170            0.0075           -0.0041               0        5                 131          False
   unseen_series               all C3_SHARED_RIDGE_vs_C1_HIER_RIDGE offset_mae             -0.0049         -0.0198           0.0093          -0.0196            0.0094           -0.0049               0        5                 131          False
   unseen_series               all C3_SHARED_RIDGE_vs_C1_HIER_RIDGE  shape_mae              0.0015         -0.0039           0.0060          -0.0045            0.0056            0.0015               4        5                 131          False
   unseen_series               all                level_oracle_gain        mae              0.7640          0.6186           0.9258           0.6390            0.9494            0.7640               5        5                 131           True
   unseen_series               all                level_oracle_gain offset_mae              0.6632          0.4711           0.8412           0.5049            0.8894            0.6632               5        5                 131           True
   unseen_series               all                level_oracle_gain  shape_mae              0.3557          0.2796           0.4347           0.2759            0.4296            0.3557               5        5                 131           True
   unseen_series               all                metal_oracle_gain        mae              0.1805          0.1099           0.2499           0.1208            0.2635            0.1805               5        5                 131           True
   unseen_series               all                metal_oracle_gain offset_mae              0.1741          0.1072           0.2522           0.1157            0.2621            0.1741               5        5                 131           True
   unseen_series               all                metal_oracle_gain  shape_mae              0.1480          0.1007           0.2026           0.1039            0.2070            0.1480               5        5                 131           True
   unseen_series               all           level_minus_metal_gain        mae              0.5835          0.4798           0.7313           0.4790            0.7282            0.5835               5        5                 131           True
   unseen_series               all           level_minus_metal_gain offset_mae              0.4891          0.3461           0.6397           0.3677            0.6700            0.4891               5        5                 131           True
   unseen_series               all           level_minus_metal_gain  shape_mae              0.2076          0.1060           0.2974           0.0983            0.2919            0.2076               5        5                 131           True
   unseen_series multi_metal_cells      C1_HIER_RIDGE_vs_MONO_RIDGE        mae             -0.0598         -0.1122          -0.0039          -0.1082           -0.0005           -0.0598               0        5                  77           True
   unseen_series multi_metal_cells      C1_HIER_RIDGE_vs_MONO_RIDGE offset_mae             -0.0575         -0.1172           0.0188          -0.1187            0.0174           -0.0575               0        5                  77           True
   unseen_series multi_metal_cells      C1_HIER_RIDGE_vs_MONO_RIDGE  shape_mae              0.0057         -0.0111           0.0167          -0.0080            0.0185            0.0057               4        5                  77           True
   unseen_series multi_metal_cells          C2_TWO_STAGE_vs_MONO_ET        mae             -0.1463         -0.2085          -0.0973          -0.2085           -0.0973           -0.1463               0        5                  77           True
   unseen_series multi_metal_cells          C2_TWO_STAGE_vs_MONO_ET offset_mae             -0.1586         -0.2223          -0.0982          -0.2298           -0.1054           -0.1586               0        5                  77           True
   unseen_series multi_metal_cells          C2_TWO_STAGE_vs_MONO_ET  shape_mae             -0.0274         -0.0550          -0.0106          -0.0501           -0.0075           -0.0274               0        5                  77           True
   unseen_series multi_metal_cells         C2_TRUECENTRE_vs_MONO_ET        mae             -0.0249         -0.0808           0.0143          -0.0652            0.0246           -0.0249               0        5                  77          False
   unseen_series multi_metal_cells         C2_TRUECENTRE_vs_MONO_ET offset_mae             -0.0448         -0.0962          -0.0067          -0.0821            0.0052           -0.0448               0        5                  77          False
   unseen_series multi_metal_cells         C2_TRUECENTRE_vs_MONO_ET  shape_mae              0.0191          0.0034           0.0375           0.0033            0.0374            0.0191               5        5                  77          False
   unseen_series multi_metal_cells C3_SHARED_RIDGE_vs_C1_HIER_RIDGE        mae             -0.0003         -0.0134           0.0145          -0.0154            0.0122           -0.0003               3        5                  77          False
   unseen_series multi_metal_cells C3_SHARED_RIDGE_vs_C1_HIER_RIDGE offset_mae             -0.0078         -0.0250           0.0084          -0.0260            0.0073           -0.0078               0        5                  77          False
   unseen_series multi_metal_cells C3_SHARED_RIDGE_vs_C1_HIER_RIDGE  shape_mae              0.0086          0.0017           0.0162           0.0003            0.0149            0.0086               5        5                  77          False
   unseen_series multi_metal_cells                level_oracle_gain        mae              0.6243          0.4907           0.7719           0.5052            0.7957            0.6243               5        5                  77           True
   unseen_series multi_metal_cells                level_oracle_gain offset_mae              0.6733          0.5044           0.8306           0.5351            0.8672            0.6733               5        5                  77           True
   unseen_series multi_metal_cells                level_oracle_gain  shape_mae              0.1622          0.1027           0.2606           0.0932            0.2392            0.1622               5        5                  77           True
   unseen_series multi_metal_cells                metal_oracle_gain        mae              0.2027          0.0985           0.2879           0.1234            0.3117            0.2027               5        5                  77           True
   unseen_series multi_metal_cells                metal_oracle_gain offset_mae              0.1191          0.0261           0.1965           0.0462            0.2156            0.1191               5        5                  77           True
   unseen_series multi_metal_cells                metal_oracle_gain  shape_mae              0.2616          0.1983           0.3427           0.2037            0.3490            0.2616               5        5                  77           True
   unseen_series multi_metal_cells           level_minus_metal_gain        mae              0.4215          0.3059           0.5526           0.3028            0.5474            0.4215               5        5                  77           True
   unseen_series multi_metal_cells           level_minus_metal_gain offset_mae              0.5542          0.4486           0.6821           0.4533            0.6925            0.5542               5        5                  77           True
   unseen_series multi_metal_cells           level_minus_metal_gain  shape_mae             -0.0994         -0.2005           0.0029          -0.2127           -0.0101           -0.0994               0        5                  77           True
```

Per-seed intervals (the same contrasts, one split partition at a time) are in `contrasts.csv`; the primary-regime `mae` rows are here:

```
          regime          endpoint  split_seed                       comparison  point_delta  bca_low  bca_high  units_improved  units_total
unseen_chemotype               all      104729      C1_HIER_RIDGE_vs_MONO_RIDGE       0.0031  -0.0065    0.0134              62          131
unseen_chemotype               all      104729          C2_TWO_STAGE_vs_MONO_ET      -0.1908  -0.3467   -0.0663              54          131
unseen_chemotype               all      104729         C2_TRUECENTRE_vs_MONO_ET       0.0145  -0.0172    0.0479              72          131
unseen_chemotype               all      104729 C3_SHARED_RIDGE_vs_C1_HIER_RIDGE      -0.0015  -0.0130    0.0068              69          131
unseen_chemotype               all      104729                level_oracle_gain       1.0473   0.8068    1.3762             129          131
unseen_chemotype               all      104729                metal_oracle_gain       0.2605   0.1242    0.4362              83          131
unseen_chemotype               all      104729           level_minus_metal_gain       0.7867   0.6151    0.9815             114          131
unseen_chemotype multi_metal_cells      104729      C1_HIER_RIDGE_vs_MONO_RIDGE       0.0059  -0.0055    0.0205              33           77
unseen_chemotype multi_metal_cells      104729          C2_TWO_STAGE_vs_MONO_ET      -0.1012  -0.2758   -0.0179              35           77
unseen_chemotype multi_metal_cells      104729         C2_TRUECENTRE_vs_MONO_ET       0.0230  -0.0153    0.0538              49           77
unseen_chemotype multi_metal_cells      104729 C3_SHARED_RIDGE_vs_C1_HIER_RIDGE       0.0013  -0.0058    0.0102              45           77
unseen_chemotype multi_metal_cells      104729                level_oracle_gain       0.6422   0.4994    0.8905              74           77
unseen_chemotype multi_metal_cells      104729                metal_oracle_gain       0.2265   0.1375    0.4126              53           77
unseen_chemotype multi_metal_cells      104729           level_minus_metal_gain       0.4157   0.2813    0.5936              58           77
unseen_chemotype               all      130363      C1_HIER_RIDGE_vs_MONO_RIDGE      -0.0265  -0.0677   -0.0098              70          131
unseen_chemotype               all      130363          C2_TWO_STAGE_vs_MONO_ET      -0.1371  -0.2705   -0.0174              54          131
unseen_chemotype               all      130363         C2_TRUECENTRE_vs_MONO_ET      -0.0056  -0.0349    0.0313              56          131
unseen_chemotype               all      130363 C3_SHARED_RIDGE_vs_C1_HIER_RIDGE       0.0028  -0.0093    0.0163              72          131
unseen_chemotype               all      130363                level_oracle_gain       0.9620   0.8160    1.1610             127          131
unseen_chemotype               all      130363                metal_oracle_gain       0.1868   0.0867    0.3197              81          131
unseen_chemotype               all      130363           level_minus_metal_gain       0.7752   0.6176    0.9283             115          131
unseen_chemotype multi_metal_cells      130363      C1_HIER_RIDGE_vs_MONO_RIDGE      -0.0308  -0.1068   -0.0072              32           77
unseen_chemotype multi_metal_cells      130363          C2_TWO_STAGE_vs_MONO_ET      -0.1628  -0.3287   -0.0589              28           77
unseen_chemotype multi_metal_cells      130363         C2_TRUECENTRE_vs_MONO_ET      -0.0052  -0.0355    0.0216              39           77
unseen_chemotype multi_metal_cells      130363 C3_SHARED_RIDGE_vs_C1_HIER_RIDGE      -0.0014  -0.0097    0.0068              40           77
unseen_chemotype multi_metal_cells      130363                level_oracle_gain       0.6420   0.5005    0.8842              73           77
unseen_chemotype multi_metal_cells      130363                metal_oracle_gain       0.2723   0.1698    0.4347              57           77
unseen_chemotype multi_metal_cells      130363           level_minus_metal_gain       0.3697   0.2377    0.5453              59           77
unseen_chemotype               all      155921      C1_HIER_RIDGE_vs_MONO_RIDGE      -0.0008  -0.0169    0.0176              64          131
unseen_chemotype               all      155921          C2_TWO_STAGE_vs_MONO_ET      -0.1137  -0.2046   -0.0388              58          131
unseen_chemotype               all      155921         C2_TRUECENTRE_vs_MONO_ET       0.0128  -0.0150    0.0461              56          131
unseen_chemotype               all      155921 C3_SHARED_RIDGE_vs_C1_HIER_RIDGE      -0.0008  -0.0128    0.0077              73          131
unseen_chemotype               all      155921                level_oracle_gain       0.9150   0.7210    1.1724             125          131
unseen_chemotype               all      155921                metal_oracle_gain       0.1749   0.0865    0.2630              81          131
unseen_chemotype               all      155921           level_minus_metal_gain       0.7401   0.5784    0.9632             118          131
unseen_chemotype multi_metal_cells      155921      C1_HIER_RIDGE_vs_MONO_RIDGE       0.0061  -0.0172    0.0290              39           77
unseen_chemotype multi_metal_cells      155921          C2_TWO_STAGE_vs_MONO_ET      -0.0761  -0.2173    0.0069              37           77
unseen_chemotype multi_metal_cells      155921         C2_TRUECENTRE_vs_MONO_ET       0.0038  -0.0170    0.0288              34           77
unseen_chemotype multi_metal_cells      155921 C3_SHARED_RIDGE_vs_C1_HIER_RIDGE       0.0030  -0.0061    0.0138              47           77
unseen_chemotype multi_metal_cells      155921                level_oracle_gain       0.5569   0.4345    0.7897              71           77
unseen_chemotype multi_metal_cells      155921                metal_oracle_gain       0.1764   0.0873    0.3093              49           77
unseen_chemotype multi_metal_cells      155921           level_minus_metal_gain       0.3805   0.2610    0.5327              62           77
unseen_chemotype               all      196613      C1_HIER_RIDGE_vs_MONO_RIDGE       0.0040  -0.0052    0.0169              66          131
unseen_chemotype               all      196613          C2_TWO_STAGE_vs_MONO_ET      -0.1784  -0.2942   -0.0662              49          131
unseen_chemotype               all      196613         C2_TRUECENTRE_vs_MONO_ET      -0.0083  -0.0309    0.0093              66          131
unseen_chemotype               all      196613 C3_SHARED_RIDGE_vs_C1_HIER_RIDGE      -0.0015  -0.0119    0.0067              75          131
unseen_chemotype               all      196613                level_oracle_gain       0.9976   0.8075    1.2170             124          131
unseen_chemotype               all      196613                metal_oracle_gain       0.2280   0.1097    0.3454              85          131
unseen_chemotype               all      196613           level_minus_metal_gain       0.7697   0.6200    0.9238             111          131
unseen_chemotype multi_metal_cells      196613      C1_HIER_RIDGE_vs_MONO_RIDGE       0.0117   0.0002    0.0302              36           77
unseen_chemotype multi_metal_cells      196613          C2_TWO_STAGE_vs_MONO_ET      -0.1170  -0.3072   -0.0374              33           77
unseen_chemotype multi_metal_cells      196613         C2_TRUECENTRE_vs_MONO_ET      -0.0056  -0.0314    0.0211              41           77
unseen_chemotype multi_metal_cells      196613 C3_SHARED_RIDGE_vs_C1_HIER_RIDGE       0.0006  -0.0092    0.0099              48           77
unseen_chemotype multi_metal_cells      196613                level_oracle_gain       0.5938   0.4534    0.8509              69           77
unseen_chemotype multi_metal_cells      196613                metal_oracle_gain       0.2149   0.1256    0.3975              53           77
unseen_chemotype multi_metal_cells      196613           level_minus_metal_gain       0.3789   0.2542    0.5569              55           77
unseen_chemotype               all      262147      C1_HIER_RIDGE_vs_MONO_RIDGE      -0.0027  -0.0292    0.0221              74          131
unseen_chemotype               all      262147          C2_TWO_STAGE_vs_MONO_ET      -0.1012  -0.1977   -0.0136              58          131
unseen_chemotype               all      262147         C2_TRUECENTRE_vs_MONO_ET       0.0039  -0.0223    0.0388              58          131
unseen_chemotype               all      262147 C3_SHARED_RIDGE_vs_C1_HIER_RIDGE       0.0051  -0.0087    0.0181              73          131
unseen_chemotype               all      262147                level_oracle_gain       0.9266   0.7421    1.1360             128          131
unseen_chemotype               all      262147                metal_oracle_gain       0.1506   0.0517    0.2554              81          131
unseen_chemotype               all      262147           level_minus_metal_gain       0.7760   0.6181    0.9396             117          131
unseen_chemotype multi_metal_cells      262147      C1_HIER_RIDGE_vs_MONO_RIDGE       0.0032  -0.0288    0.0219              42           77
unseen_chemotype multi_metal_cells      262147          C2_TWO_STAGE_vs_MONO_ET      -0.0772  -0.2522    0.0084              37           77
unseen_chemotype multi_metal_cells      262147         C2_TRUECENTRE_vs_MONO_ET       0.0038  -0.0231    0.0283              37           77
unseen_chemotype multi_metal_cells      262147 C3_SHARED_RIDGE_vs_C1_HIER_RIDGE       0.0048  -0.0061    0.0192              44           77
unseen_chemotype multi_metal_cells      262147                level_oracle_gain       0.5848   0.4582    0.8651              74           77
unseen_chemotype multi_metal_cells      262147                metal_oracle_gain       0.1715   0.0814    0.3338              50           77
unseen_chemotype multi_metal_cells      262147           level_minus_metal_gain       0.4133   0.2918    0.5839              61           77
   unseen_ligand               all      104729      C1_HIER_RIDGE_vs_MONO_RIDGE      -0.0194  -0.0612    0.0033              70          131
   unseen_ligand               all      104729          C2_TWO_STAGE_vs_MONO_ET      -0.1158  -0.2042   -0.0442              51          131
   unseen_ligand               all      104729         C2_TRUECENTRE_vs_MONO_ET      -0.0284  -0.0496   -0.0040              51          131
   unseen_ligand               all      104729 C3_SHARED_RIDGE_vs_C1_HIER_RIDGE       0.0002  -0.0124    0.0111              66          131
   unseen_ligand               all      104729                level_oracle_gain       0.7452   0.5909    0.8987             123          131
   unseen_ligand               all      104729                metal_oracle_gain       0.1481   0.0772    0.2333              82          131
   unseen_ligand               all      104729           level_minus_metal_gain       0.5971   0.4424    0.7517             108          131
   unseen_ligand multi_metal_cells      104729      C1_HIER_RIDGE_vs_MONO_RIDGE      -0.0161  -0.0730    0.0249              43           77
   unseen_ligand multi_metal_cells      104729          C2_TWO_STAGE_vs_MONO_ET      -0.1146  -0.2135   -0.0343              27           77
   unseen_ligand multi_metal_cells      104729         C2_TRUECENTRE_vs_MONO_ET      -0.0119  -0.0388    0.0186              35           77
   unseen_ligand multi_metal_cells      104729 C3_SHARED_RIDGE_vs_C1_HIER_RIDGE       0.0016  -0.0079    0.0116              44           77
   unseen_ligand multi_metal_cells      104729                level_oracle_gain       0.5716   0.4259    0.7324              72           77
   unseen_ligand multi_metal_cells      104729                metal_oracle_gain       0.1941   0.1057    0.2877              52           77
   unseen_ligand multi_metal_cells      104729           level_minus_metal_gain       0.3775   0.2538    0.5348              53           77
   unseen_ligand               all      130363      C1_HIER_RIDGE_vs_MONO_RIDGE      -0.0055  -0.0335    0.0122              69          131
   unseen_ligand               all      130363          C2_TWO_STAGE_vs_MONO_ET      -0.0718  -0.1370   -0.0047              55          131
   unseen_ligand               all      130363         C2_TRUECENTRE_vs_MONO_ET      -0.0169  -0.0429    0.0055              60          131
   unseen_ligand               all      130363 C3_SHARED_RIDGE_vs_C1_HIER_RIDGE       0.0052  -0.0049    0.0145              71          131
   unseen_ligand               all      130363                level_oracle_gain       0.6946   0.5546    0.8539             122          131
   unseen_ligand               all      130363                metal_oracle_gain       0.1065   0.0443    0.1773              82          131
   unseen_ligand               all      130363           level_minus_metal_gain       0.5881   0.4609    0.7390             108          131
   unseen_ligand multi_metal_cells      130363      C1_HIER_RIDGE_vs_MONO_RIDGE      -0.0153  -0.0835    0.0157              38           77
   unseen_ligand multi_metal_cells      130363          C2_TWO_STAGE_vs_MONO_ET      -0.0948  -0.1702   -0.0286              29           77
   unseen_ligand multi_metal_cells      130363         C2_TRUECENTRE_vs_MONO_ET      -0.0110  -0.0446    0.0205              34           77
   unseen_ligand multi_metal_cells      130363 C3_SHARED_RIDGE_vs_C1_HIER_RIDGE       0.0084  -0.0011    0.0200              46           77
   unseen_ligand multi_metal_cells      130363                level_oracle_gain       0.5473   0.4250    0.7000              71           77
   unseen_ligand multi_metal_cells      130363                metal_oracle_gain       0.1675   0.0830    0.2468              54           77
   unseen_ligand multi_metal_cells      130363           level_minus_metal_gain       0.3798   0.2709    0.5201              54           77
   unseen_ligand               all      155921      C1_HIER_RIDGE_vs_MONO_RIDGE      -0.0059  -0.0197    0.0073              63          131
   unseen_ligand               all      155921          C2_TWO_STAGE_vs_MONO_ET      -0.0676  -0.1339   -0.0027              58          131
   unseen_ligand               all      155921         C2_TRUECENTRE_vs_MONO_ET      -0.0106  -0.0284    0.0070              63          131
   unseen_ligand               all      155921 C3_SHARED_RIDGE_vs_C1_HIER_RIDGE       0.0030  -0.0089    0.0132              68          131
   unseen_ligand               all      155921                level_oracle_gain       0.7319   0.5754    0.9024             122          131
   unseen_ligand               all      155921                metal_oracle_gain       0.1051   0.0403    0.1749              77          131
   unseen_ligand               all      155921           level_minus_metal_gain       0.6268   0.4627    0.7958             111          131
   unseen_ligand multi_metal_cells      155921      C1_HIER_RIDGE_vs_MONO_RIDGE       0.0042  -0.0105    0.0193              37           77
   unseen_ligand multi_metal_cells      155921          C2_TWO_STAGE_vs_MONO_ET      -0.0669  -0.1506    0.0072              38           77
   unseen_ligand multi_metal_cells      155921         C2_TRUECENTRE_vs_MONO_ET       0.0137  -0.0062    0.0407              47           77
   unseen_ligand multi_metal_cells      155921 C3_SHARED_RIDGE_vs_C1_HIER_RIDGE       0.0110   0.0005    0.0266              45           77
   unseen_ligand multi_metal_cells      155921                level_oracle_gain       0.5541   0.4117    0.7410              69           77
   unseen_ligand multi_metal_cells      155921                metal_oracle_gain       0.1615   0.0731    0.2573              48           77
   unseen_ligand multi_metal_cells      155921           level_minus_metal_gain       0.3926   0.2669    0.5514              56           77
   unseen_ligand               all      196613      C1_HIER_RIDGE_vs_MONO_RIDGE      -0.0199  -0.0634    0.0110              69          131
   unseen_ligand               all      196613          C2_TWO_STAGE_vs_MONO_ET      -0.1261  -0.2472   -0.0536              46          131
   unseen_ligand               all      196613         C2_TRUECENTRE_vs_MONO_ET      -0.0236  -0.0474   -0.0007              55          131
   unseen_ligand               all      196613 C3_SHARED_RIDGE_vs_C1_HIER_RIDGE      -0.0068  -0.0324    0.0080              71          131
   unseen_ligand               all      196613                level_oracle_gain       0.7637   0.5901    0.9546             120          131
   unseen_ligand               all      196613                metal_oracle_gain       0.1629   0.0826    0.2837              84          131
   unseen_ligand               all      196613           level_minus_metal_gain       0.6008   0.4574    0.7638             112          131
   unseen_ligand multi_metal_cells      196613      C1_HIER_RIDGE_vs_MONO_RIDGE      -0.0095  -0.0804    0.0266              43           77
   unseen_ligand multi_metal_cells      196613          C2_TWO_STAGE_vs_MONO_ET      -0.1380  -0.3114   -0.0430              29           77
   unseen_ligand multi_metal_cells      196613         C2_TRUECENTRE_vs_MONO_ET      -0.0195  -0.0520    0.0162              33           77
   unseen_ligand multi_metal_cells      196613 C3_SHARED_RIDGE_vs_C1_HIER_RIDGE      -0.0046  -0.0385    0.0146              42           77
   unseen_ligand multi_metal_cells      196613                level_oracle_gain       0.6187   0.4564    0.8646              65           77
   unseen_ligand multi_metal_cells      196613                metal_oracle_gain       0.2140   0.1046    0.3759              55           77
   unseen_ligand multi_metal_cells      196613           level_minus_metal_gain       0.4047   0.2771    0.5596              57           77
   unseen_ligand               all      262147      C1_HIER_RIDGE_vs_MONO_RIDGE      -0.0020  -0.0125    0.0062              66          131
   unseen_ligand               all      262147          C2_TWO_STAGE_vs_MONO_ET      -0.1308  -0.2108   -0.0678              61          131
   unseen_ligand               all      262147         C2_TRUECENTRE_vs_MONO_ET      -0.0340  -0.0531   -0.0133              49          131
   unseen_ligand               all      262147 C3_SHARED_RIDGE_vs_C1_HIER_RIDGE       0.0035  -0.0060    0.0130              65          131
   unseen_ligand               all      262147                level_oracle_gain       0.7748   0.6094    0.9390             126          131
   unseen_ligand               all      262147                metal_oracle_gain       0.1483   0.0795    0.2286              77          131
   unseen_ligand               all      262147           level_minus_metal_gain       0.6265   0.4813    0.7873             114          131
   unseen_ligand multi_metal_cells      262147      C1_HIER_RIDGE_vs_MONO_RIDGE      -0.0012  -0.0129    0.0088              42           77
   unseen_ligand multi_metal_cells      262147          C2_TWO_STAGE_vs_MONO_ET      -0.1314  -0.2145   -0.0661              33           77
   unseen_ligand multi_metal_cells      262147         C2_TRUECENTRE_vs_MONO_ET      -0.0247  -0.0502    0.0016              29           77
   unseen_ligand multi_metal_cells      262147 C3_SHARED_RIDGE_vs_C1_HIER_RIDGE       0.0055  -0.0044    0.0210              43           77
   unseen_ligand multi_metal_cells      262147                level_oracle_gain       0.5918   0.4480    0.7607              71           77
   unseen_ligand multi_metal_cells      262147                metal_oracle_gain       0.1941   0.1115    0.2841              53           77
   unseen_ligand multi_metal_cells      262147           level_minus_metal_gain       0.3977   0.2698    0.5691              59           77
   unseen_series               all      104729      C1_HIER_RIDGE_vs_MONO_RIDGE      -0.0243  -0.0708    0.0314              64          131
   unseen_series               all      104729          C2_TWO_STAGE_vs_MONO_ET      -0.1994  -0.2922   -0.1354              42          131
   unseen_series               all      104729         C2_TRUECENTRE_vs_MONO_ET      -0.0543  -0.1153   -0.0151              50          131
   unseen_series               all      104729 C3_SHARED_RIDGE_vs_C1_HIER_RIDGE      -0.0041  -0.0176    0.0073              59          131
   unseen_series               all      104729                level_oracle_gain       0.7896   0.6500    0.9534             124          131
   unseen_series               all      104729                metal_oracle_gain       0.2000   0.0947    0.3028              84          131
   unseen_series               all      104729           level_minus_metal_gain       0.5896   0.4939    0.7251             106          131
   unseen_series multi_metal_cells      104729      C1_HIER_RIDGE_vs_MONO_RIDGE      -0.0681  -0.1366    0.0019              32           77
   unseen_series multi_metal_cells      104729          C2_TWO_STAGE_vs_MONO_ET      -0.1873  -0.2878   -0.1122              21           77
   unseen_series multi_metal_cells      104729         C2_TRUECENTRE_vs_MONO_ET      -0.0450  -0.1430    0.0192              36           77
   unseen_series multi_metal_cells      104729 C3_SHARED_RIDGE_vs_C1_HIER_RIDGE       0.0000  -0.0138    0.0128              41           77
   unseen_series multi_metal_cells      104729                level_oracle_gain       0.6615   0.5287    0.8051              74           77
   unseen_series multi_metal_cells      104729                metal_oracle_gain       0.2207   0.0686    0.3450              55           77
   unseen_series multi_metal_cells      104729           level_minus_metal_gain       0.4408   0.3067    0.5635              54           77
   unseen_series               all      130363      C1_HIER_RIDGE_vs_MONO_RIDGE       0.0142  -0.0400    0.0791              75          131
   unseen_series               all      130363          C2_TWO_STAGE_vs_MONO_ET      -0.1471  -0.2169   -0.0858              47          131
   unseen_series               all      130363         C2_TRUECENTRE_vs_MONO_ET      -0.0386  -0.0922   -0.0048              55          131
   unseen_series               all      130363 C3_SHARED_RIDGE_vs_C1_HIER_RIDGE      -0.0032  -0.0171    0.0079              66          131
   unseen_series               all      130363                level_oracle_gain       0.7555   0.6336    0.9178             126          131
   unseen_series               all      130363                metal_oracle_gain       0.1602   0.0912    0.2454              84          131
   unseen_series               all      130363           level_minus_metal_gain       0.5954   0.4979    0.7474             107          131
   unseen_series multi_metal_cells      130363      C1_HIER_RIDGE_vs_MONO_RIDGE      -0.0468  -0.1337    0.0228              38           77
   unseen_series multi_metal_cells      130363          C2_TWO_STAGE_vs_MONO_ET      -0.1479  -0.2270   -0.0778              29           77
   unseen_series multi_metal_cells      130363         C2_TRUECENTRE_vs_MONO_ET      -0.0371  -0.1393    0.0274              39           77
   unseen_series multi_metal_cells      130363 C3_SHARED_RIDGE_vs_C1_HIER_RIDGE      -0.0028  -0.0164    0.0101              43           77
   unseen_series multi_metal_cells      130363                level_oracle_gain       0.6317   0.5076    0.7953              74           77
   unseen_series multi_metal_cells      130363                metal_oracle_gain       0.2071   0.0833    0.3119              53           77
   unseen_series multi_metal_cells      130363           level_minus_metal_gain       0.4246   0.2842    0.5539              52           77
   unseen_series               all      155921      C1_HIER_RIDGE_vs_MONO_RIDGE      -0.0180  -0.0675    0.0317              64          131
   unseen_series               all      155921          C2_TWO_STAGE_vs_MONO_ET      -0.1258  -0.1973   -0.0673              50          131
   unseen_series               all      155921         C2_TRUECENTRE_vs_MONO_ET      -0.0222  -0.0458   -0.0015              55          131
   unseen_series               all      155921 C3_SHARED_RIDGE_vs_C1_HIER_RIDGE      -0.0077  -0.0221    0.0042              63          131
   unseen_series               all      155921                level_oracle_gain       0.7424   0.6064    0.9099             121          131
   unseen_series               all      155921                metal_oracle_gain       0.1579   0.0885    0.2340              81          131
   unseen_series               all      155921           level_minus_metal_gain       0.5845   0.4854    0.7323             113          131
   unseen_series multi_metal_cells      155921      C1_HIER_RIDGE_vs_MONO_RIDGE      -0.0603  -0.1251    0.0102              29           77
   unseen_series multi_metal_cells      155921          C2_TWO_STAGE_vs_MONO_ET      -0.1045  -0.1831   -0.0487              27           77
   unseen_series multi_metal_cells      155921         C2_TRUECENTRE_vs_MONO_ET      -0.0044  -0.0435    0.0309              36           77
   unseen_series multi_metal_cells      155921 C3_SHARED_RIDGE_vs_C1_HIER_RIDGE      -0.0037  -0.0194    0.0114              41           77
   unseen_series multi_metal_cells      155921                level_oracle_gain       0.5865   0.4428    0.7551              73           77
   unseen_series multi_metal_cells      155921                metal_oracle_gain       0.1711   0.0902    0.2647              53           77
   unseen_series multi_metal_cells      155921           level_minus_metal_gain       0.4154   0.2988    0.5606              61           77
   unseen_series               all      196613      C1_HIER_RIDGE_vs_MONO_RIDGE      -0.0131  -0.0736    0.0424              68          131
   unseen_series               all      196613          C2_TWO_STAGE_vs_MONO_ET      -0.1733  -0.2733   -0.1062              47          131
   unseen_series               all      196613         C2_TRUECENTRE_vs_MONO_ET      -0.0169  -0.0370    0.0043              59          131
   unseen_series               all      196613 C3_SHARED_RIDGE_vs_C1_HIER_RIDGE      -0.0040  -0.0206    0.0093              60          131
   unseen_series               all      196613                level_oracle_gain       0.7793   0.6008    0.9654             123          131
   unseen_series               all      196613                metal_oracle_gain       0.2043   0.1316    0.3078              94          131
   unseen_series               all      196613           level_minus_metal_gain       0.5750   0.4341    0.7243             112          131
   unseen_series multi_metal_cells      196613      C1_HIER_RIDGE_vs_MONO_RIDGE      -0.0546  -0.1411    0.0028              37           77
   unseen_series multi_metal_cells      196613          C2_TWO_STAGE_vs_MONO_ET      -0.1482  -0.2707   -0.0746              28           77
   unseen_series multi_metal_cells      196613         C2_TRUECENTRE_vs_MONO_ET      -0.0092  -0.0421    0.0239              39           77
   unseen_series multi_metal_cells      196613 C3_SHARED_RIDGE_vs_C1_HIER_RIDGE       0.0035  -0.0098    0.0175              41           77
   unseen_series multi_metal_cells      196613                level_oracle_gain       0.6248   0.4586    0.8243              72           77
   unseen_series multi_metal_cells      196613                metal_oracle_gain       0.2119   0.1219    0.3295              57           77
   unseen_series multi_metal_cells      196613           level_minus_metal_gain       0.4128   0.2802    0.5695              58           77
   unseen_series               all      262147      C1_HIER_RIDGE_vs_MONO_RIDGE      -0.0112  -0.0545    0.0421              70          131
   unseen_series               all      262147          C2_TWO_STAGE_vs_MONO_ET      -0.1600  -0.2380   -0.0947              46          131
   unseen_series               all      262147         C2_TRUECENTRE_vs_MONO_ET      -0.0354  -0.0572   -0.0168              46          131
   unseen_series               all      262147 C3_SHARED_RIDGE_vs_C1_HIER_RIDGE      -0.0013  -0.0131    0.0153              64          131
   unseen_series               all      262147                level_oracle_gain       0.7530   0.5727    0.9337             124          131
   unseen_series               all      262147                metal_oracle_gain       0.1802   0.1038    0.2611              85          131
   unseen_series               all      262147           level_minus_metal_gain       0.5729   0.4463    0.7293             115          131
   unseen_series multi_metal_cells      262147      C1_HIER_RIDGE_vs_MONO_RIDGE      -0.0694  -0.1336   -0.0035              32           77
   unseen_series multi_metal_cells      262147          C2_TWO_STAGE_vs_MONO_ET      -0.1435  -0.2439   -0.0600              26           77
   unseen_series multi_metal_cells      262147         C2_TRUECENTRE_vs_MONO_ET      -0.0291  -0.0593    0.0010              29           77
   unseen_series multi_metal_cells      262147 C3_SHARED_RIDGE_vs_C1_HIER_RIDGE       0.0015  -0.0134    0.0289              44           77
   unseen_series multi_metal_cells      262147                level_oracle_gain       0.6168   0.4501    0.7888              73           77
   unseen_series multi_metal_cells      262147                metal_oracle_gain       0.2029   0.0942    0.3134              51           77
   unseen_series multi_metal_cells      262147           level_minus_metal_gain       0.4139   0.2844    0.5522              61           77
```

## What the inner CV chose for the ligand penalty

λ_ligand is the whole mechanism of C1: small = each training ligand gets its own estimated intercept, `1e6` = the intercepts are shrunk to nothing and the model falls back on the descriptor prior, which is all a held-out ligand can ever get. The inner CV is grouped on the **regime's own column**, so this table is the model discovering, per regime, whether ligand intercepts transfer.

```
          regime lambda_ligand  n_folds  share_of_folds  lambda_fixed_median  lambda_fixed_min  lambda_fixed_max
unseen_chemotype           0.1        2            0.08                100.0              10.0             100.0
unseen_chemotype             1       10            0.40                100.0              10.0             100.0
unseen_chemotype            10        7            0.28                100.0              10.0             100.0
unseen_chemotype         1e+06        6            0.24                100.0              10.0             100.0
   unseen_ligand           0.1        2            0.08                100.0              10.0             100.0
   unseen_ligand             1        7            0.28                100.0              10.0             100.0
   unseen_ligand            10       10            0.40                100.0              10.0             100.0
   unseen_ligand         1e+06        6            0.24                100.0              10.0             100.0
   unseen_series           0.1       16            0.64                 10.0              10.0             100.0
   unseen_series             1        9            0.36                 10.0              10.0             100.0
```

## Where C1's prediction comes from (mean |contribution| per block, test rows)

```
          regime          component  mean_abs_contribution  mean_contribution  n_folds
unseen_chemotype        block__COND                 0.2996            -0.0524       25
unseen_chemotype block__INTERACTION                 0.2910            -0.0774       25
unseen_chemotype      block__DONORS                 0.1819            -0.0836       25
unseen_chemotype          intercept                 0.1678            -0.1382       25
unseen_chemotype    block__PHYSCHEM                 0.1322            -0.0514       25
unseen_chemotype  block__MASSACTION                 0.0779            -0.0012       25
unseen_chemotype       block__METAL                 0.0645             0.0026       25
unseen_chemotype   ligand_intercept                 0.0000             0.0000       25
   unseen_ligand        block__COND                 0.3563             0.0445       25
   unseen_ligand block__INTERACTION                 0.2490            -0.0457       25
   unseen_ligand      block__DONORS                 0.1613            -0.0437       25
   unseen_ligand    block__PHYSCHEM                 0.1188            -0.0048       25
   unseen_ligand          intercept                 0.1012            -0.0936       25
   unseen_ligand       block__METAL                 0.0737            -0.0000       25
   unseen_ligand  block__MASSACTION                 0.0635            -0.0031       25
   unseen_ligand   ligand_intercept                 0.0000             0.0000       25
   unseen_series        block__COND                 0.4209             0.0387       25
   unseen_series   ligand_intercept                 0.3425             0.2297       25
   unseen_series block__INTERACTION                 0.2779            -0.0060       25
   unseen_series      block__DONORS                 0.1452             0.0006       25
   unseen_series          intercept                 0.1379            -0.1260       25
   unseen_series       block__METAL                 0.0991            -0.0004       25
   unseen_series    block__PHYSCHEM                 0.0962            -0.0087       25
   unseen_series  block__MASSACTION                 0.0777             0.0012       25
```

## Derived pair predictions — the C4 measurement

Pair predictions are formed **only** as `ŷ(A) − ŷ(B)` with A the lighter metal, inside a test cell. `NULL` is the leave-fold-out pair-label mean, the strongest honest pair alternative established in gen3/gen4. No verdict depends on these numbers except C4.

```
          regime           model  n_pairs  pair_macro_mae  pair_pooled_mae  pair_sign_accuracy  antisymmetry_max  transitivity_max  n_triples
unseen_chemotype         MONO_ET    70865        0.518882         0.486787            0.670361               0.0          0.000000     106115
unseen_chemotype      MONO_RIDGE    70865        0.627435         0.562251            0.629051               0.0          0.000000     106115
unseen_chemotype   C1_HIER_RIDGE    70865        0.626890         0.560961            0.630371               0.0          0.000000     106115
unseen_chemotype    C2_TWO_STAGE    70865        0.521766         0.490943            0.668768               0.0          0.000000     106115
unseen_chemotype   C2_TRUECENTRE    70865        0.503469         0.474251            0.685239               0.0          0.000000     106115
unseen_chemotype C3_SHARED_RIDGE    70865        0.620658         0.556303            0.633857               0.0          0.000000     106115
unseen_chemotype    ORACLE_LEVEL    70865        0.503469         0.474251            0.685239               0.0          0.000000     106115
unseen_chemotype    ORACLE_METAL    70865        0.000000         0.000000            1.000000               0.0          0.000000     106115
unseen_chemotype            NULL    70865        0.726678         0.626335            0.624522               NaN          0.524131     106115
   unseen_ligand         MONO_ET    70865        0.443283         0.458183            0.749623               0.0          0.000000     106115
   unseen_ligand      MONO_RIDGE    70865        0.602985         0.557698            0.700354               0.0          0.000000     106115
   unseen_ligand   C1_HIER_RIDGE    70865        0.601344         0.555670            0.702515               0.0          0.000000     106115
   unseen_ligand    C2_TWO_STAGE    70865        0.449284         0.446770            0.731047               0.0          0.000000     106115
   unseen_ligand   C2_TRUECENTRE    70865        0.421424         0.428604            0.759051               0.0          0.000000     106115
   unseen_ligand C3_SHARED_RIDGE    70865        0.590979         0.543532            0.707546               0.0          0.000000     106115
   unseen_ligand    ORACLE_LEVEL    70865        0.421424         0.428604            0.759051               0.0          0.000000     106115
   unseen_ligand    ORACLE_METAL    70865        0.000000         0.000000            1.000000               0.0          0.000000     106115
   unseen_ligand            NULL    70865        0.673434         0.568334            0.695488               NaN          0.456458     106115
   unseen_series         MONO_ET    70865        0.385648         0.367077            0.830440               0.0          0.000000     106115
   unseen_series      MONO_RIDGE    70865        0.541418         0.539600            0.759449               0.0          0.000000     106115
   unseen_series   C1_HIER_RIDGE    70865        0.529444         0.519843            0.767175               0.0          0.000000     106115
   unseen_series    C2_TWO_STAGE    70865        0.405839         0.384588            0.797558               0.0          0.000000     106115
   unseen_series   C2_TRUECENTRE    70865        0.366374         0.365208            0.835456               0.0          0.000000     106115
   unseen_series C3_SHARED_RIDGE    70865        0.521138         0.507285            0.776032               0.0          0.000000     106115
   unseen_series    ORACLE_LEVEL    70865        0.366374         0.365208            0.835456               0.0          0.000000     106115
   unseen_series    ORACLE_METAL    70865        0.000000         0.000000            1.000000               0.0          0.000000     106115
   unseen_series            NULL    70865        0.606015         0.527322            0.747203               NaN          0.379221     106115
```

Identity residuals over every fold (the two columns above are printed in full here because the table rounds them to zero): antisymmetry 0.000e+00, transitivity 8.882e-16, over 318345 within-cell triples and 212595 pairs — one check per model, 2546760 in total. `NULL` is a pair-label mean and is NOT derived as a difference of two level predictions, so its transitivity residual is large by construction; that is what makes the zeros above non-vacuous.

## Hard chemistry — descriptive, not a protected endpoint here

Bins are cut on `nn_train_tanimoto`: the maximum Tanimoto from the test ligand to the **training ligands of its own fold**, computed with `exclude_self=False`. Under `unseen_series` the test ligand *is* in training, so its distance is 1.0 by construction and the hard bins are empty — that is the correct answer, not a bug.

```
          regime endpoint             arm  n_rows  n_ligands  n_ecfp_clusters  macro_mae  offset_mae  shape_mae  frac_within_1_log
unseen_chemotype   nn<0.4         MONO_ET   407.6       18.4             18.4     1.0241      0.9257     0.3912             0.4144
unseen_chemotype   nn<0.6         MONO_ET  1383.0       68.2             66.8     1.2048      1.0570     0.5289             0.4860
unseen_chemotype      all         MONO_ET  5248.0      152.0            131.0     1.0468      0.8744     0.5069             0.5179
unseen_chemotype   nn<0.4      MONO_RIDGE   407.6       18.4             18.4     1.1889      1.1142     0.4361             0.3481
unseen_chemotype   nn<0.6      MONO_RIDGE  1383.0       68.2             66.8     1.2399      1.0647     0.5875             0.4067
unseen_chemotype      all      MONO_RIDGE  5248.0      152.0            131.0     1.1476      0.9364     0.6065             0.4457
unseen_chemotype   nn<0.4   C1_HIER_RIDGE   407.6       18.4             18.4     1.1977      1.1058     0.4427             0.3607
unseen_chemotype   nn<0.6   C1_HIER_RIDGE  1383.0       68.2             66.8     1.2474      1.0771     0.5819             0.4134
unseen_chemotype      all   C1_HIER_RIDGE  5248.0      152.0            131.0     1.1522      0.9482     0.5985             0.4500
unseen_chemotype   nn<0.4    C2_TWO_STAGE   407.6       18.4             18.4     1.0889      0.9956     0.4007             0.4776
unseen_chemotype   nn<0.6    C2_TWO_STAGE  1383.0       68.2             66.8     1.3655      1.2422     0.5326             0.4825
unseen_chemotype      all    C2_TWO_STAGE  5248.0      152.0            131.0     1.1910      1.0693     0.5014             0.5410
unseen_chemotype   nn<0.4   C2_TRUECENTRE   407.6       18.4             18.4     1.0205      0.9195     0.3859             0.4292
unseen_chemotype   nn<0.6   C2_TRUECENTRE  1383.0       68.2             66.8     1.1951      1.0521     0.5209             0.4886
unseen_chemotype      all   C2_TRUECENTRE  5248.0      152.0            131.0     1.0433      0.8808     0.5047             0.5228
unseen_chemotype   nn<0.4 C3_SHARED_RIDGE   407.6       18.4             18.4     1.1971      1.0961     0.4412             0.3503
unseen_chemotype   nn<0.6 C3_SHARED_RIDGE  1383.0       68.2             66.8     1.2488      1.0769     0.5793             0.4112
unseen_chemotype      all C3_SHARED_RIDGE  5248.0      152.0            131.0     1.1513      0.9495     0.5958             0.4497
unseen_chemotype   nn<0.4    ORACLE_LEVEL   407.6       18.4             18.4     0.2519      0.0743     0.2043             0.9091
unseen_chemotype   nn<0.6    ORACLE_LEVEL  1383.0       68.2             66.8     0.2424      0.0657     0.1953             0.9434
unseen_chemotype      all    ORACLE_LEVEL  5248.0      152.0            131.0     0.2213      0.0728     0.1848             0.9587
unseen_chemotype   nn<0.4    ORACLE_METAL   407.6       18.4             18.4     0.9629      0.9107     0.2560             0.4302
unseen_chemotype   nn<0.6    ORACLE_METAL  1383.0       68.2             66.8     1.1306      1.0493     0.3733             0.5045
unseen_chemotype      all    ORACLE_METAL  5248.0      152.0            131.0     0.9909      0.8834     0.3653             0.5327
   unseen_ligand   nn<0.4         MONO_ET   231.8       17.0             17.0     1.1288      1.0314     0.3767             0.4650
   unseen_ligand   nn<0.6         MONO_ET   637.2       36.6             36.6     1.1671      0.9750     0.5039             0.5389
   unseen_ligand      all         MONO_ET  5248.0      152.0            131.0     0.8526      0.6566     0.4477             0.5999
   unseen_ligand   nn<0.4      MONO_RIDGE   231.8       17.0             17.0     1.2264      1.1564     0.4413             0.4631
   unseen_ligand   nn<0.6      MONO_RIDGE   637.2       36.6             36.6     1.1992      0.9797     0.5521             0.4544
   unseen_ligand      all      MONO_RIDGE  5248.0      152.0            131.0     1.0371      0.7761     0.5992             0.4829
   unseen_ligand   nn<0.4   C1_HIER_RIDGE   231.8       17.0             17.0     1.2537      1.1712     0.4486             0.4464
   unseen_ligand   nn<0.6   C1_HIER_RIDGE   637.2       36.6             36.6     1.2313      1.0065     0.5533             0.4457
   unseen_ligand      all   C1_HIER_RIDGE  5248.0      152.0            131.0     1.0477      0.7955     0.5981             0.4798
   unseen_ligand   nn<0.4    C2_TWO_STAGE   231.8       17.0             17.0     1.1986      1.1367     0.3891             0.4946
   unseen_ligand   nn<0.6    C2_TWO_STAGE   637.2       36.6             36.6     1.2989      1.1592     0.5234             0.4865
   unseen_ligand      all    C2_TWO_STAGE  5248.0      152.0            131.0     0.9550      0.8121     0.4358             0.5881
   unseen_ligand   nn<0.4   C2_TRUECENTRE   231.8       17.0             17.0     1.1625      1.0638     0.3813             0.4736
   unseen_ligand   nn<0.6   C2_TRUECENTRE   637.2       36.6             36.6     1.2012      1.0133     0.5065             0.5416
   unseen_ligand      all   C2_TRUECENTRE  5248.0      152.0            131.0     0.8753      0.6950     0.4413             0.5803
   unseen_ligand   nn<0.4 C3_SHARED_RIDGE   231.8       17.0             17.0     1.2480      1.1480     0.4477             0.4291
   unseen_ligand   nn<0.6 C3_SHARED_RIDGE   637.2       36.6             36.6     1.2357      0.9986     0.5544             0.4459
   unseen_ligand      all C3_SHARED_RIDGE  5248.0      152.0            131.0     1.0466      0.7963     0.5952             0.4798
   unseen_ligand   nn<0.4    ORACLE_LEVEL   231.8       17.0             17.0     0.2417      0.0735     0.1957             0.9344
   unseen_ligand   nn<0.6    ORACLE_LEVEL   637.2       36.6             36.6     0.2701      0.0783     0.2199             0.9537
   unseen_ligand      all    ORACLE_LEVEL  5248.0      152.0            131.0     0.2129      0.0951     0.1551             0.9771
   unseen_ligand   nn<0.4    ORACLE_METAL   231.8       17.0             17.0     1.0903      1.0331     0.2491             0.4176
   unseen_ligand   nn<0.6    ORACLE_METAL   637.2       36.6             36.6     1.1189      0.9907     0.3384             0.5317
   unseen_ligand      all    ORACLE_METAL  5248.0      152.0            131.0     0.8208      0.6768     0.3331             0.5989
   unseen_series   nn<0.4         MONO_ET   118.4       12.0             12.0     0.9508      0.8870     0.3276             0.6014
   unseen_series   nn<0.6         MONO_ET   248.8       26.2             26.2     1.1021      0.9440     0.4553             0.5730
   unseen_series      all         MONO_ET  5248.0      152.0            131.0     0.8080      0.5439     0.4661             0.6574
   unseen_series   nn<0.4      MONO_RIDGE   118.4       12.0             12.0     1.1107      1.0480     0.4147             0.5253
   unseen_series   nn<0.6      MONO_RIDGE   248.8       26.2             26.2     1.1738      0.9835     0.4892             0.4922
   unseen_series      all      MONO_RIDGE  5248.0      152.0            131.0     1.0224      0.7528     0.5994             0.4869
   unseen_series   nn<0.4   C1_HIER_RIDGE   118.4       12.0             12.0     1.1908      1.1055     0.4210             0.5357
   unseen_series   nn<0.6   C1_HIER_RIDGE   248.8       26.2             26.2     1.2065      1.0090     0.4999             0.4848
   unseen_series      all   C1_HIER_RIDGE  5248.0      152.0            131.0     1.0328      0.7266     0.6054             0.5154
   unseen_series   nn<0.4    C2_TWO_STAGE   118.4       12.0             12.0     1.1080      1.0633     0.3549             0.5716
   unseen_series   nn<0.6    C2_TWO_STAGE   248.8       26.2             26.2     1.3106      1.2028     0.4885             0.4610
   unseen_series      all    C2_TWO_STAGE  5248.0      152.0            131.0     0.9691      0.7278     0.4973             0.6245
   unseen_series   nn<0.4   C2_TRUECENTRE   118.4       12.0             12.0     0.9756      0.9025     0.3116             0.5385
   unseen_series   nn<0.6   C2_TRUECENTRE   248.8       26.2             26.2     1.1217      0.9616     0.4484             0.5439
   unseen_series      all   C2_TRUECENTRE  5248.0      152.0            131.0     0.8415      0.6023     0.4620             0.6372
   unseen_series   nn<0.4 C3_SHARED_RIDGE   118.4       12.0             12.0     1.1839      1.0905     0.4080             0.5153
   unseen_series   nn<0.6 C3_SHARED_RIDGE   248.8       26.2             26.2     1.2180      1.0062     0.4968             0.4769
   unseen_series      all C3_SHARED_RIDGE  5248.0      152.0            131.0     1.0369      0.7322     0.6041             0.5151
   unseen_series   nn<0.4    ORACLE_LEVEL   118.4       12.0             12.0     0.2181      0.0677     0.1741             0.9673
   unseen_series   nn<0.6    ORACLE_LEVEL   248.8       26.2             26.2     0.2729      0.0627     0.2308             0.9364
   unseen_series      all    ORACLE_LEVEL  5248.0      152.0            131.0     0.2052      0.0896     0.1488             0.9826
   unseen_series   nn<0.4    ORACLE_METAL   118.4       12.0             12.0     0.9162      0.8778     0.1763             0.5084
   unseen_series   nn<0.6    ORACLE_METAL   248.8       26.2             26.2     1.0447      0.9522     0.2398             0.5440
   unseen_series      all    ORACLE_METAL  5248.0      152.0            131.0     0.7886      0.5678     0.3581             0.6482
```

## Pre-registered hypotheses (protocol Experiment C)

Scoring rule, stated so a verdict cannot be over-read: **PASS** = the pre-registered condition is met, including the seed-agreement clause; **FAIL** = the interval excludes the predicted direction (evidence *against*); **INCONCLUSIVE** = the interval straddles zero, or the seed clause cannot be evaluated (a single-seed pilot). An INCONCLUSIVE is not a weak PASS.

**C1 — PASS**  ·  The component that fails to transfer to a new chemotype is the level, not the metal response.

* pass condition: under unseen_chemotype, on test cells with cell_n_metals >= 2: MAE(ORACLE_METAL) − MAE(ORACLE_LEVEL) > 0 with **BCa CI95 low > 0** over chemotype blocks (this delta is exactly the level oracle's gain minus the metal oracle's, because both are measured against the same C2_TWO_STAGE reference), plus the house seed clause (>= 4/5 seeds positive; a single-seed pilot cannot evaluate it).
* measured: DECISIVE, multi-metal test cells — level gain minus metal gain: Δ = +0.3916 log units; percentile CI95 [+0.2638, +0.5435]; **BCa CI95 [+0.2733, +0.5570]**; cluster-robust [+0.2578, +0.5255]; block-macro Δ +0.4039 (77 ECFP-cluster units in 45 chemotype bootstrap blocks; mean over seeds +0.3916, 5/5 seeds positive). Components: level oracle gain Δ = +0.6039 log units; percentile CI95 [+0.4739, +0.8117]; **BCa CI95 [+0.4855, +0.8563]**; cluster-robust [+0.4378, +0.7701]; block-macro Δ +0.6252 (77 ECFP-cluster units in 45 chemotype bootstrap blocks; mean over seeds +0.6039, 5/5 seeds positive); metal oracle gain Δ = +0.2123 log units; percentile CI95 [+0.1226, +0.3488]; **BCa CI95 [+0.1313, +0.3710]**; cluster-robust [+0.1021, +0.3225]; block-macro Δ +0.2213 (77 ECFP-cluster units in 45 chemotype bootstrap blocks; mean over seeds +0.2123, 5/5 seeds positive). All-rows version (reported, never the verdict): Δ = +0.7696 log units; percentile CI95 [+0.5839, +0.9080]; **BCa CI95 [+0.6157, +0.9393]**; cluster-robust [+0.6003, +0.9389]; block-macro Δ +0.7369 (131 ECFP-cluster units in 79 chemotype bootstrap blocks; mean over seeds +0.7696, 5/5 seeds positive)
* what would falsify this: If the metal oracle removes as much error as the level oracle on multi-metal cells (BCa CI95 covering or below zero), the level is NOT the component that fails, and Phase 1's attribution — on which this whole generation's plan rests — is wrong.

**C2 — INCONCLUSIVE**  ·  Partial pooling helps exactly where the ligand is known, and nowhere else.

* pass condition: C1_HIER_RIDGE beats MONO_RIDGE under unseen_series (BCa CI95 low > 0) AND the two are indistinguishable under unseen_chemotype (CI95 spans 0). Both clauses are required.
* measured: unseen_series (ligand IS in training, so the intercept is live): Δ = -0.0105 log units; percentile CI95 [-0.0471, +0.0376]; **BCa CI95 [-0.0485, +0.0355]**; cluster-robust [-0.0506, +0.0297]; block-macro Δ +0.0165 (131 ECFP-cluster units in 79 chemotype bootstrap blocks; mean over seeds -0.0105, 1/5 seeds positive). unseen_chemotype (intercept is zero by construction): Δ = -0.0046 log units; percentile CI95 [-0.0173, +0.0058]; **BCa CI95 [-0.0195, +0.0048]**; cluster-robust [-0.0155, +0.0063]; block-macro Δ -0.0096 (131 ECFP-cluster units in 79 chemotype bootstrap blocks; mean over seeds -0.0046, 2/5 seeds positive); CI95 spans zero there: True.
* what would falsify this: If the hierarchical ridge also beats plain ridge on a NEW chemotype, the intercepts are not what is helping — something else in the fit is — and the mixed-effects reading of C1 is wrong. If it fails to beat plain ridge even on a seen ligand, the partial-pooling machinery is buying nothing anywhere.

**C3 — PASS**  ·  Model structure does not substitute for chemical coverage.

* pass condition: C2_TWO_STAGE does NOT beat MONO_ET under unseen_chemotype by more than 0.02 macro MAE with BCa CI95 low > 0. PASS = no such beat; FAIL = the beat happened, which is the surprise worth reporting. PASS = no such beat over >= 2 split seeds; a single-seed pilot scores INCONCLUSIVE because one partition cannot support a claim of absence.
* measured: C2_TWO_STAGE vs MONO_ET, all test rows: Δ = -0.1443 log units; percentile CI95 [-0.2258, -0.0556]; **BCa CI95 [-0.2397, -0.0687]**; cluster-robust [-0.2244, -0.0641]; block-macro Δ -0.1169 (131 ECFP-cluster units in 79 chemotype bootstrap blocks; mean over seeds -0.1443, 0/5 seeds positive); beat by more than 0.02 with BCa low > 0: False. EXPLORATORY (amendment; the variant was chosen after a one-seed preview, so this line is exploratory-confirmed at best and is NOT part of the verdict) — C2_TRUECENTRE vs MONO_ET: Δ = +0.0035 log units; percentile CI95 [-0.0260, +0.0253]; **BCa CI95 [-0.0194, +0.0331]**; cluster-robust [-0.0239, +0.0308]; block-macro Δ -0.0159 (131 ECFP-cluster units in 79 chemotype bootstrap blocks; mean over seeds +0.0035, 3/5 seeds positive); beat: False.
* what would falsify this: A C2 that beats MONO_ET materially on new chemistry falsifies the generation's reading: it would say model structure, not coverage, was the lever, contradicting Phase 1's case A.

**C4 — PASS**  ·  Level-derived pair predictions keep exact antisymmetry and transitivity.

* pass condition: max |ŷ(A,B) + ŷ(B,A)| and max |ŷ(A,B) + ŷ(B,C) − ŷ(A,C)| below 1e-09 on every test fold of every regime and seed, with at least one triple actually present (an empty pair table would satisfy the bound vacuously). This is an algebraic identity, so no seed clause applies.
* measured: max antisymmetry residual 0.000e+00 (recomputed in this runner from the row predictions, because hierarchical.pair_consistency returns a hard-coded 0.0 for it); max transitivity residual 8.882e-16 over 318345 within-cell triples and 212595 pairs, checked for each of the eight models (2546760 triple checks in total).
* what would falsify this: Any residual at or above 1e-9 means the pair predictions were not derived as a difference of two level predictions somewhere in the pipeline — which would silently reintroduce the orientation bug the gen3/gen4 pair models were built to avoid.

## How to break this result

* **The oracles are not models and the gains are not achievable errors.** Both read test labels. C1 compares two *attributions*; it says nothing about how much error a deployable model could remove. Anyone quoting `ORACLE_LEVEL`'s MAE as an achievable number has misread the table.
* **The two oracles are not symmetric halves of one model.** `ORACLE_METAL` = Â + true departure uses Stage A's *predicted* level; `ORACLE_LEVEL` = true cell mean + B̂ uses the **truecentre** Stage B. So the decisive delta is `Stage A's level error − truecentre Stage B's departure error`, which is the quantity C1 is about, but it is not 'the same model with one input replaced' in both directions. Re-run with `ORACLE_LEVEL` built from the cross-fitted Stage B to see whether the sign survives.
* **Multi-metal cells are a different test population.** Only 521 of 2,405 cells hold two or more metals. The C1 endpoint restricts to them because a singleton cell makes the level oracle trivially informative — but that restriction also changes which ligands and conditions are being scored. The all-rows row is printed beside every C1 number for that reason.
* **The ridge family is deliberately low-capacity.** MONO_RIDGE, C1 and C3 share a compact design without the 206 LIG2D_EXT columns. A C1-beats-MONO_RIDGE result is a statement about partial pooling inside that family, never a statement that C1 is competitive with the forest.
* **Seeds are not replicates.** The split seeds re-partition the same ligands. The paired bootstrap over chemotype blocks is the load-bearing statistic; seed agreement is a second requirement, not independent evidence.
* **Multiplicity.** C1–C4 are the only protected claims. The tables report many more contrasts (comparisons × statistics × endpoints × regimes), none corrected. Treat every interval outside C1–C4 as descriptive — and `C2_TRUECENTRE` as exploratory even where it appears inside C3, because the variant was chosen after a one-seed preview.
* **Do not quote the percentile interval alone.** One chemotype holds ~21 % of the scoring units on this cohort; the percentile interval's real one-sided Type-I rate was measured at ~12.7 % at the `all` endpoint. BCa is the interval the verdicts use, and the cluster-robust and block-macro columns are printed beside it.
* **`unseen_series` is the easy regime.** The ligand is in training and every test row's nearest training neighbour is itself. A model that wins there has learned nothing about new chemistry.

