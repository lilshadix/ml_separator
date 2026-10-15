# Leakage-safe lanthanide feature-ablation report

## Scope and protocol

- Pair scope: `all` (6,699 condition-matched pairs).
- Held-out group: `extractant`; split seed `42`.
- Model seed: `42`; outer/inner folds: `5/3`.
- Main ablations: `A0, A1, A2, A3, A4, A5, A6` on exactly the same outer folds.
- Training-only 3D shuffle seeds: `1009, 2017, 3019`.
- VR descriptor blocks: `global_shape, coordination_shape`.
- Non-geometric xTB/electronic columns excluded from A0–A6: `6`.

The primary comparison is A2 (conditions + lanthanide + 2D) versus A5 (A2 + local 3D). A2 versus A6 is secondary. A3 versus A2 tests whether local coordination geometry can substitute for ligand 2D information.
The excluded column names and reasons are frozen in `feature_registry.json`; therefore the primary A2-vs-A5 contrast is not confounded by dipole or partial-charge features.

## Direct answers

1. **Does 3D improve unseen-extractant prediction?** evidence favors A2; the equal-extractant paired interval is entirely negative (equal-extractant macro delta_MAE=-0.019089; 95% CI [-0.028194, -0.008653]).
2. **Absolute MAE improvement:** -0.005113 log units (A2 minus A5).
3. **Relative MAE improvement:** -1.240432% versus A2.
4. **Held-out extractant breadth:** 10/34 (29.411765%) have positive delta_MAE.
5. **Descriptor blocks:** A2+D1: delta_MAE=+0.001760, A2+D3: delta_MAE=-0.000521, A2+D2: delta_MAE=-0.001349, A2+D5: delta_MAE=-0.003707, A2+D4: delta_MAE=-0.008590. Secondary complete-3D result: A2-vs-A6 delta_MAE=-0.000345.
6. **Training-only 3D shuffle:** A5_SHUFFLED_s1009: delta_MAE=+0.000653, A5_SHUFFLED_s2017: delta_MAE=-0.004600, A5_SHUFFLED_s3019: delta_MAE=-0.002190.
7. **Geometry-quality dependence:** not estimable because the accepted common cohort does not contain both quality strata.
8. **Largest lanthanide gains:** Gd (+0.009767), Sm (+0.006297), Nd (+0.005150). Largest exact-extractant gains: CC(C)CN(CC(C)C)C(=O)COCC(=O)N(CC(C)C)CC(C)C (+0.030893); CCCCCCCCN(CCCCCCCC)C(=O)[C@H](C)O[C@@H](C)C(=O)N(CCCCCCCC)CCCCCCCC (+0.028690); CCCCCCCCCCN(CCCCCCCCCC)C(=O)COCC(=O)N(CCCCCC)CCCCCC (+0.023309). No separate ligand-family label was inferred.
9. **Across seeds:** this directory is one fixed model/split seed; consistency across seeds must be answered by the aggregate run, not inferred here.
10. **Metric consistency (positive favors A5):** MAE=-0.005113, RMSE=-0.006473, R2=-0.014303, Spearman=-0.005222.

## Aggregate OOF metrics by ablation

| ablation | n_rows | mae | rmse | r2 | pearson | spearman | fold_mae_mean | fold_mae_sample_sd | fold_rmse_mean | fold_rmse_sample_sd | fold_r2_mean | fold_r2_sample_sd | fold_pearson_mean | fold_pearson_sample_sd | fold_spearman_mean | fold_spearman_sample_sd | macro_group_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | 6699.000000 | 0.657417 | 0.958609 | -0.681009 | -0.048719 | 0.031893 | 0.627833 | 0.180307 | 0.920797 | 0.212937 | -0.647471 | 0.213725 | 0.029371 | 0.224178 | 0.046787 | 0.247970 | 0.635956 |
| A1 | 6699.000000 | 0.464809 | 0.695387 | 0.115414 | 0.511851 | 0.380326 | 0.401632 | 0.131644 | 0.575935 | 0.187126 | 0.281999 | 0.470065 | 0.686826 | 0.205300 | 0.601360 | 0.181959 | 0.330064 |
| A2 | 6699.000000 | 0.412219 | 0.600729 | 0.339846 | 0.619050 | 0.584801 | 0.372604 | 0.097396 | 0.531566 | 0.143971 | 0.406542 | 0.291106 | 0.705318 | 0.135879 | 0.666833 | 0.109663 | 0.302572 |
| A3 | 6699.000000 | 0.496498 | 0.764652 | -0.069583 | 0.397932 | 0.275049 | 0.422979 | 0.152146 | 0.611831 | 0.230769 | 0.176385 | 0.624500 | 0.608398 | 0.315929 | 0.538099 | 0.291826 | 0.351867 |
| A4 | 6699.000000 | 0.413994 | 0.603564 | 0.333601 | 0.612080 | 0.593098 | 0.377965 | 0.091930 | 0.537103 | 0.141775 | 0.399262 | 0.278203 | 0.694996 | 0.134252 | 0.662070 | 0.100729 | 0.315362 |
| A5 | 6699.000000 | 0.417333 | 0.607202 | 0.325543 | 0.605401 | 0.579579 | 0.382131 | 0.092637 | 0.541929 | 0.145591 | 0.390553 | 0.279603 | 0.689020 | 0.135097 | 0.654145 | 0.119201 | 0.321661 |
| A6 | 6699.000000 | 0.412564 | 0.599902 | 0.341663 | 0.614773 | 0.579849 | 0.378458 | 0.088132 | 0.538166 | 0.138679 | 0.402335 | 0.257291 | 0.698615 | 0.123871 | 0.657853 | 0.114072 | 0.319803 |
| A2+D1 | 6699.000000 | 0.410459 | 0.598840 | 0.343991 | 0.618168 | 0.593103 | 0.373391 | 0.095950 | 0.531049 | 0.142444 | 0.410645 | 0.281498 | 0.706410 | 0.135809 | 0.666913 | 0.110823 | 0.312023 |
| A2+D2 | 6699.000000 | 0.413569 | 0.601856 | 0.337367 | 0.614748 | 0.590602 | 0.375859 | 0.095110 | 0.533247 | 0.142796 | 0.401405 | 0.295556 | 0.701659 | 0.139192 | 0.668251 | 0.113854 | 0.310024 |
| A2+D3 | 6699.000000 | 0.412740 | 0.602577 | 0.335780 | 0.616956 | 0.574124 | 0.374174 | 0.093745 | 0.538309 | 0.142979 | 0.396827 | 0.278547 | 0.698307 | 0.128519 | 0.659165 | 0.116966 | 0.306352 |
| A2+D4 | 6699.000000 | 0.420809 | 0.616021 | 0.305810 | 0.594593 | 0.576597 | 0.380168 | 0.100446 | 0.540539 | 0.151106 | 0.380521 | 0.322673 | 0.690037 | 0.158317 | 0.661284 | 0.117260 | 0.311783 |
| A2+D5 | 6699.000000 | 0.415927 | 0.607381 | 0.325147 | 0.607148 | 0.588517 | 0.376215 | 0.097393 | 0.534855 | 0.145701 | 0.391826 | 0.317572 | 0.697170 | 0.150405 | 0.667367 | 0.120013 | 0.307592 |
| A5_SHUFFLED_s1009 | 6699.000000 | 0.411567 | 0.594069 | 0.354403 | 0.622646 | 0.589008 | 0.376947 | 0.088447 | 0.531025 | 0.133373 | 0.409950 | 0.273569 | 0.705693 | 0.129263 | 0.657258 | 0.120147 | 0.317490 |
| A5_SHUFFLED_s2017 | 6699.000000 | 0.416819 | 0.605279 | 0.329809 | 0.608983 | 0.587686 | 0.380614 | 0.093254 | 0.537774 | 0.144275 | 0.397214 | 0.284266 | 0.696398 | 0.137145 | 0.663668 | 0.115038 | 0.316035 |
| A5_SHUFFLED_s3019 | 6699.000000 | 0.414410 | 0.598204 | 0.345385 | 0.617610 | 0.593318 | 0.380455 | 0.089993 | 0.534593 | 0.136357 | 0.401566 | 0.281905 | 0.698465 | 0.132485 | 0.663135 | 0.114440 | 0.321679 |

## Paired fold improvements

Positive `delta_MAE` and `delta_RMSE` mean the named candidate model wins; positive `delta_R2` and `delta_Spearman` also favor the candidate. The reference/candidate columns define direction explicitly.

| comparison | reference | candidate | outer_fold | delta_mae | delta_rmse | delta_r2 | delta_spearman |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A2_vs_A5 | A2 | A5 | 0.000000 | -0.017942 | -0.001260 | -0.004203 | -0.029716 |
| A2_vs_A5 | A2 | A5 | 1.000000 | -0.006303 | -0.000721 | -0.001447 | -0.017245 |
| A2_vs_A5 | A2 | A5 | 2.000000 | -0.013538 | -0.026218 | -0.041240 | -0.005878 |
| A2_vs_A5 | A2 | A5 | 3.000000 | 0.005609 | 0.000692 | 0.001894 | -0.009026 |
| A2_vs_A5 | A2 | A5 | 4.000000 | -0.015463 | -0.024306 | -0.034951 | -0.001573 |
| A2_vs_A6 | A2 | A6 | 0.000000 | -0.001727 | 0.011493 | 0.037846 | -0.014695 |
| A2_vs_A6 | A2 | A6 | 1.000000 | -0.015066 | -0.012983 | -0.026500 | -0.020492 |
| A2_vs_A6 | A2 | A6 | 2.000000 | -0.010506 | -0.020749 | -0.032500 | -0.000618 |
| A2_vs_A6 | A2 | A6 | 3.000000 | 0.012803 | 0.012189 | 0.033071 | -0.007822 |
| A2_vs_A6 | A2 | A6 | 4.000000 | -0.014775 | -0.022951 | -0.032953 | -0.001270 |
| A3_vs_A2 | A3 | A2 | 0.000000 | 0.036277 | 0.016541 | 0.056041 | 0.010427 |
| A3_vs_A2 | A3 | A2 | 1.000000 | 0.005017 | 0.038744 | 0.081842 | 0.050481 |
| A3_vs_A2 | A3 | A2 | 2.000000 | 0.004711 | 0.007343 | 0.011384 | -0.009042 |
| A3_vs_A2 | A3 | A2 | 3.000000 | 0.164238 | 0.278905 | 0.912227 | 0.565078 |
| A3_vs_A2 | A3 | A2 | 4.000000 | 0.041632 | 0.059793 | 0.089295 | 0.026729 |
| A2_vs_A5_SHUFFLED_s1009 | A2 | A5_SHUFFLED_s1009 | 0.000000 | -0.010571 | -0.001589 | -0.005302 | -0.032006 |
| A2_vs_A5_SHUFFLED_s1009 | A2 | A5_SHUFFLED_s1009 | 1.000000 | -0.010601 | -0.005862 | -0.011849 | -0.031767 |
| A2_vs_A5_SHUFFLED_s1009 | A2 | A5_SHUFFLED_s1009 | 2.000000 | -0.003280 | 0.010887 | 0.016640 | 0.004948 |
| A2_vs_A5_SHUFFLED_s1009 | A2 | A5_SHUFFLED_s1009 | 3.000000 | 0.012684 | 0.014467 | 0.039188 | 0.009789 |
| A2_vs_A5_SHUFFLED_s1009 | A2 | A5_SHUFFLED_s1009 | 4.000000 | -0.009948 | -0.015198 | -0.021637 | 0.001159 |
| A2_vs_A5_SHUFFLED_s2017 | A2 | A5_SHUFFLED_s2017 | 0.000000 | -0.008879 | 0.000930 | 0.003096 | -0.016877 |
| A2_vs_A5_SHUFFLED_s2017 | A2 | A5_SHUFFLED_s2017 | 1.000000 | -0.008861 | -0.000634 | -0.001272 | -0.021793 |
| A2_vs_A5_SHUFFLED_s2017 | A2 | A5_SHUFFLED_s2017 | 2.000000 | -0.013069 | -0.011740 | -0.018263 | 0.010078 |
| A2_vs_A5_SHUFFLED_s2017 | A2 | A5_SHUFFLED_s2017 | 3.000000 | 0.003580 | -0.001693 | -0.004639 | 0.012795 |
| A2_vs_A5_SHUFFLED_s2017 | A2 | A5_SHUFFLED_s2017 | 4.000000 | -0.012825 | -0.017903 | -0.025564 | -0.000027 |
| A2_vs_A5_SHUFFLED_s3019 | A2 | A5_SHUFFLED_s3019 | 0.000000 | -0.018864 | -0.008927 | -0.030013 | -0.022742 |
| A2_vs_A5_SHUFFLED_s3019 | A2 | A5_SHUFFLED_s3019 | 1.000000 | -0.006220 | 0.000641 | 0.001284 | -0.012332 |
| A2_vs_A5_SHUFFLED_s3019 | A2 | A5_SHUFFLED_s3019 | 2.000000 | -0.009731 | 0.005401 | 0.008291 | -0.002875 |
| A2_vs_A5_SHUFFLED_s3019 | A2 | A5_SHUFFLED_s3019 | 3.000000 | 0.011588 | 0.010251 | 0.027852 | 0.022564 |
| A2_vs_A5_SHUFFLED_s3019 | A2 | A5_SHUFFLED_s3019 | 4.000000 | -0.016029 | -0.022502 | -0.032294 | -0.003105 |

_Only the first 30 of 55 rows are shown._

## Geometry-shuffle negative control

Shuffling is performed only inside each outer-training fold and never uses outer-test labels or descriptors. Compare A5 to A2 and every `A5_SHUFFLED` row above. Meaningful structure-specific signal is supported only when correct A5 improves while shuffled A5 falls back toward A2.

## Descriptor-block availability

| block | column_count | status |
| --- | --- | --- |
| D1 | 11.000000 | evaluated_if_enabled |
| D2 | 10.000000 | evaluated_if_enabled |
| D3 | 7.000000 | evaluated_if_enabled |
| D4 | 8.000000 | evaluated_if_enabled |
| D5 | 2.000000 | evaluated_if_enabled |

Empty local blocks are unavailable and were skipped, not fit as duplicate A2 models. Unavailable blocks in this run: `none`. The pre-specified exact coordinate-only blocks are `global_shape` and `coordination_shape`; `ligand_field` and approximate ray-based `enclosure` require explicit CLI selection.

## Held-out extractant comparison

| extractant_id | n_samples | MAE_2D | MAE_2D3D | delta_MAE | RMSE_2D | RMSE_2D3D | delta_RMSE | Spearman_2D | Spearman_2D3D | delta_Spearman |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CC(C)CCCC(C)CCCCN(C)C(=O)COCC(=O)N(C)CCCCC(C)CCCC(C)C | 91.000000 | 0.138070 | 0.194355 | -0.056285 | 0.165786 | 0.219800 | -0.054014 | 0.972464 | 0.973690 | 0.001226 |
| CC(C)CN(CC(C)C)C(=O)COCC(=O)N(CC(C)C)CC(C)C | 28.000000 | 0.258421 | 0.227528 | 0.030893 | 0.317000 | 0.288025 | 0.028975 | 0.787630 | 0.885057 | 0.097427 |
| CCCCC(CC)CN(C)C(=O)COCC(=O)N(C)CC(CC)CCCC | 91.000000 | 0.160980 | 0.165698 | -0.004717 | 0.198409 | 0.201302 | -0.002893 | 0.952843 | 0.946472 | -0.006370 |
| CCCCC(CC)CN(CC(CC)CCCC)C(=O)COCC(=O)N(CC(CC)CCCC)CC(CC)CCCC | 210.000000 | 0.405080 | 0.455649 | -0.050569 | 0.557646 | 0.646144 | -0.088499 | 0.875062 | 0.875325 | 0.000263 |
| CCCCCCC(C)N(CCCC)C(=O)COCC(=O)N(CCCC)C(C)CCCCCC | 112.000000 | 0.316827 | 0.314264 | 0.002563 | 0.460890 | 0.450290 | 0.010600 | 0.650344 | 0.674612 | 0.024268 |
| CCCCCCC(CCCC)CCCN(C)C(=O)COCC(=O)N(C)CCCC(CCCC)CCCCCC | 91.000000 | 0.269404 | 0.246502 | 0.022902 | 0.330102 | 0.302853 | 0.027249 | 0.961061 | 0.963561 | 0.002500 |
| CCCCCCCCC(C)(C)SCC1CN(C(=O)COCC(=O)N2CC(C)C(CSC(C)(C)CCCCCCCC)C2)CC1C | 91.000000 | 0.204321 | 0.265658 | -0.061336 | 0.243967 | 0.326395 | -0.082428 | 0.971652 | 0.970553 | -0.001099 |
| CCCCCCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)N(CCCCCCCC)CCCCCCCCCCCC | 78.000000 | 0.578405 | 0.587633 | -0.009227 | 0.720179 | 0.716459 | 0.003720 | 0.923722 | 0.922735 | -0.000986 |
| CCCCCCCCCCCCN(CCCCCCCCCCCC)C(=O)COCC(=O)N(CCCCCCCCCCCC)CCCCCCCCCCCC | 16.000000 | 0.216776 | 0.252967 | -0.036191 | 0.302768 | 0.300751 | 0.002017 | 0.526471 | 0.438235 | -0.088235 |
| CCCCCCCCCCCCN(CCCCCCCCCCCC)C(=O)COCCOCC(=O)N(CCCCCCCCCCCC)CCCCCCCCCCCC | 79.000000 | 0.799806 | 0.850640 | -0.050834 | 1.145434 | 1.295787 | -0.150354 | -0.307625 | -0.465600 | -0.157975 |
| CCCCCCCCCCN(CCCCCCCCCC)C(=O)C(C)OC(C)C(=O)N(CCCCCCCCCC)CCCCCCCCCC | 140.000000 | 0.157721 | 0.206158 | -0.048437 | 0.192930 | 0.267206 | -0.074275 | 0.867503 | 0.774460 | -0.093043 |
| CCCCCCCCCCN(CCCCCCCCCC)C(=O)COCC(=O)N(CCCCCC)CCCCCC | 298.000000 | 0.629928 | 0.606619 | 0.023309 | 0.757540 | 0.726226 | 0.031314 | 0.458073 | 0.448919 | -0.009154 |
| CCCCCCCCN(C)C(=O)COCC(=O)N(C)CCCCCCCC | 507.000000 | 0.520835 | 0.529436 | -0.008601 | 0.696234 | 0.699342 | -0.003108 | 0.830014 | 0.831646 | 0.001632 |
| CCCCCCCCN(C)C(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC | 91.000000 | 0.154301 | 0.174245 | -0.019944 | 0.190104 | 0.211560 | -0.021457 | 0.983867 | 0.960885 | -0.022981 |
| CCCCCCCCN(CC(C)C)C(=O)COCC(=O)N(CCCCCCCC)CC(C)C | 28.000000 | 0.111352 | 0.134066 | -0.022714 | 0.138271 | 0.164230 | -0.025959 | 0.845649 | 0.810071 | -0.035577 |
| CCCCCCCCN(CC(CC)CCCC)C(=O)COCC(=O)N(CCCCCCCC)CC(CC)CCCC | 28.000000 | 0.094270 | 0.125022 | -0.030752 | 0.112904 | 0.151507 | -0.038602 | 0.850575 | 0.824302 | -0.026273 |
| CCCCCCCCN(CC)C(=O)COCC(=O)N(CC)CCCCCCCC | 91.000000 | 0.090453 | 0.128468 | -0.038014 | 0.113285 | 0.165995 | -0.052710 | 0.980491 | 0.956012 | -0.024478 |
| CCCCCCCCN(CCC(C)CC(C)(C)C)C(=O)COCC(=O)N(CCCCCCCC)CCC(C)CC(C)(C)C | 91.000000 | 0.221173 | 0.198933 | 0.022240 | 0.308723 | 0.275001 | 0.033722 | 0.972878 | 0.976780 | 0.003902 |
| CCCCCCCCN(CCC)C(=O)COCC(=O)N(CCC)CCCCCCCC | 91.000000 | 0.100808 | 0.150050 | -0.049242 | 0.128876 | 0.191874 | -0.062998 | 0.982561 | 0.963561 | -0.019000 |
| CCCCCCCCN(CCCC(CCCC)CCCCCC)C(=O)COCC(=O)N(CCCCCCCC)CCCC(CCCC)CCCCCC | 78.000000 | 0.260464 | 0.247860 | 0.012605 | 0.323563 | 0.297788 | 0.025776 | 0.973470 | 0.963884 | -0.009585 |
| CCCCCCCCN(CCCCC(C)CCCC(C)C)C(=O)COCC(=O)N(CCCCCCCC)CCCCC(C)CCCC(C)C | 91.000000 | 0.341182 | 0.380325 | -0.039143 | 0.419890 | 0.460346 | -0.040456 | 0.982943 | 0.979408 | -0.003536 |
| CCCCCCCCN(CCCCCCCC)C(=O)CN(CC(=O)N(CCCCCCCC)CCCCCCCC)CC(=O)N(CCCCCCCC)CCCCCCCC | 1.000000 | 0.599507 | 0.652749 | -0.053243 | 0.599507 | 0.652749 | -0.053243 | NA | NA | NA |
| CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)N(C)C | 21.000000 | 0.386651 | 0.438788 | -0.052137 | 0.573744 | 0.609582 | -0.035838 | 0.853247 | 0.867532 | 0.014286 |
| CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC | 2921.000000 | 0.507640 | 0.502031 | 0.005609 | 0.714350 | 0.713658 | 0.000692 | 0.615589 | 0.606563 | -0.009026 |
| CCCCCCCCN(CCCCCCCC)C(=O)[C@H](C)O[C@@H](C)C(=O)N(CCCCCCCC)CCCCCCCC | 91.000000 | 0.402951 | 0.374261 | 0.028690 | 0.539013 | 0.499396 | 0.039618 | 0.911515 | 0.826294 | -0.085221 |
| CCCCCCCCN(CCCCCCCC)C(=O)[C@H](C)O[C@H](C)C(=O)N(CCCCCCCC)CCCCCCCC | 91.000000 | 0.168542 | 0.251357 | -0.082816 | 0.219137 | 0.320215 | -0.101078 | 0.722113 | 0.653728 | -0.068385 |
| CCCCCCN(CCCCCC)C(=O)COCC(=O)N(CCCCCC)CCCCCC | 21.000000 | 0.098124 | 0.109374 | -0.011249 | 0.113667 | 0.118285 | -0.004617 | 0.923377 | 0.906494 | -0.016883 |
| CCCCN(CCCC)C(=O)COCC(=O)N(CCCC)CCCC | 172.000000 | 0.418779 | 0.421542 | -0.002763 | 0.545043 | 0.540222 | 0.004821 | 0.677211 | 0.672383 | -0.004829 |
| CCCN(CCC)C(=O)COCC(=O)N(CCC)CCC | 150.000000 | 0.180913 | 0.220194 | -0.039281 | 0.240125 | 0.285258 | -0.045133 | 0.712792 | 0.652581 | -0.060211 |
| CCN(CC)C(=O)COCC(=O)N(CC)CC | 258.000000 | 0.212014 | 0.196739 | 0.015274 | 0.286473 | 0.261157 | 0.025316 | 0.284061 | 0.309251 | 0.025189 |

_Only the first 30 of 34 rows are shown._

## Lanthanide comparison

| Ln | n_samples | MAE_2D | MAE_2D3D | delta_MAE | RMSE_2D | RMSE_2D3D | delta_RMSE | Spearman_2D | Spearman_2D3D | delta_Spearman |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ce | 913.000000 | 0.552174 | 0.558144 | -0.005970 | 0.758906 | 0.761770 | -0.002864 | 0.715831 | 0.711573 | -0.004258 |
| Dy | 1123.000000 | 0.348272 | 0.350383 | -0.002110 | 0.529764 | 0.528432 | 0.001332 | 0.730901 | 0.730720 | -0.000181 |
| Er | 1045.000000 | 0.365314 | 0.378085 | -0.012771 | 0.575548 | 0.589069 | -0.013521 | 0.693928 | 0.663648 | -0.030280 |
| Eu | 949.000000 | 0.366516 | 0.365496 | 0.001020 | 0.496556 | 0.490095 | 0.006461 | 0.800757 | 0.785843 | -0.014914 |
| Gd | 1144.000000 | 0.364730 | 0.354962 | 0.009767 | 0.489780 | 0.473857 | 0.015923 | 0.685337 | 0.676277 | -0.009060 |
| Ho | 929.000000 | 0.386505 | 0.399809 | -0.013304 | 0.596526 | 0.611529 | -0.015003 | 0.720267 | 0.695489 | -0.024778 |
| La | 923.000000 | 0.612385 | 0.628041 | -0.015657 | 0.849042 | 0.880894 | -0.031852 | 0.646045 | 0.626286 | -0.019759 |
| Lu | 849.000000 | 0.355951 | 0.366357 | -0.010407 | 0.549014 | 0.572575 | -0.023561 | 0.719538 | 0.691764 | -0.027774 |
| Nd | 1095.000000 | 0.520211 | 0.515061 | 0.005150 | 0.698806 | 0.687923 | 0.010883 | 0.606776 | 0.604826 | -0.001950 |
| Pr | 704.000000 | 0.418112 | 0.427332 | -0.009219 | 0.588948 | 0.588146 | 0.000802 | 0.788759 | 0.791682 | 0.002923 |
| Sm | 1120.000000 | 0.424153 | 0.417856 | 0.006297 | 0.543955 | 0.538041 | 0.005915 | 0.675436 | 0.683703 | 0.008266 |
| Tb | 897.000000 | 0.354027 | 0.360286 | -0.006259 | 0.517836 | 0.515594 | 0.002241 | 0.785554 | 0.763957 | -0.021596 |
| Tm | 827.000000 | 0.344179 | 0.362890 | -0.018711 | 0.564066 | 0.592299 | -0.028233 | 0.751264 | 0.718929 | -0.032335 |
| Yb | 880.000000 | 0.355814 | 0.365984 | -0.010170 | 0.557611 | 0.576463 | -0.018852 | 0.756861 | 0.739220 | -0.017641 |

## Geometry quality

The machine-readable `geometry_qc_summary.json` states which quality strata are actually represented. Failed geometries are excluded by pair construction and are never silently imputed into a 3D arm. A high-versus-low-confidence claim is unavailable when either stratum has no eligible pairs.

## Audit interpretation

This run does not declare a positive 3D result merely from one metric. Review the paired A2-vs-A5 MAE first, then RMSE, R2, Spearman, extractant win fraction, shuffle controls, geometry quality, and seed-to-seed aggregation. `validation.json` and `leakage_audit.json` must both pass before scientific interpretation.
