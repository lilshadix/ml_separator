# Leakage-safe lanthanide feature-ablation report

## Scope and protocol

- Pair scope: `all` (6,699 condition-matched pairs).
- Held-out group: `extractant`; split seed `104729`.
- Model seed: `42`; outer/inner folds: `5/3`.
- Main ablations: `A0, A1, A2, A3, A4, A5, A6` on exactly the same outer folds.
- Training-only 3D shuffle seeds: `1009, 2017, 3019`.
- VR descriptor blocks: `global_shape, coordination_shape`.
- Non-geometric xTB/electronic columns excluded from A0–A6: `6`.

The primary comparison is A2 (conditions + lanthanide + 2D) versus A5 (A2 + local 3D). A2 versus A6 is secondary. A3 versus A2 tests whether local coordination geometry can substitute for ligand 2D information.
The excluded column names and reasons are frozen in `feature_registry.json`; therefore the primary A2-vs-A5 contrast is not confounded by dipole or partial-charge features.

## Direct answers

1. **Does 3D improve unseen-extractant prediction?** evidence favors A2; the equal-extractant paired interval is entirely negative (equal-extractant macro delta_MAE=-0.015353; 95% CI [-0.026721, -0.003718]).
2. **Absolute MAE improvement:** -0.001000 log units (A2 minus A5).
3. **Relative MAE improvement:** -0.240618% versus A2.
4. **Held-out extractant breadth:** 15/34 (44.117647%) have positive delta_MAE.
5. **Descriptor blocks:** A2+D2: delta_MAE=-0.001834, A2+D3: delta_MAE=-0.002038, A2+D1: delta_MAE=-0.002429, A2+D4: delta_MAE=-0.002477, A2+D5: delta_MAE=-0.003406. Secondary complete-3D result: A2-vs-A6 delta_MAE=+0.001837.
6. **Training-only 3D shuffle:** A5_SHUFFLED_s1009: delta_MAE=-0.004060, A5_SHUFFLED_s2017: delta_MAE=-0.004307, A5_SHUFFLED_s3019: delta_MAE=-0.002777.
7. **Geometry-quality dependence:** not estimable because the accepted common cohort does not contain both quality strata.
8. **Largest lanthanide gains:** Gd (+0.009069), Nd (+0.007452), Sm (+0.006193). Largest exact-extractant gains: COCCN(CCOC)C(=O)COCC(=O)N(CCOC)CCOC (+0.071771); CCCCCCCCCCN(CCCCCCCCCC)C(=O)COCC(=O)N(CCCCCC)CCCCCC (+0.029651); CC(C)CN(CC(C)C)C(=O)COCC(=O)N(CC(C)C)CC(C)C (+0.019816). No separate ligand-family label was inferred.
9. **Across seeds:** this directory is one fixed model/split seed; consistency across seeds must be answered by the aggregate run, not inferred here.
10. **Metric consistency (positive favors A5):** MAE=-0.001000, RMSE=+0.002923, R2=+0.006482, Spearman=+0.001181.

## Aggregate OOF metrics by ablation

| ablation | n_rows | mae | rmse | r2 | pearson | spearman | fold_mae_mean | fold_mae_sample_sd | fold_rmse_mean | fold_rmse_sample_sd | fold_r2_mean | fold_r2_sample_sd | fold_pearson_mean | fold_pearson_sample_sd | fold_spearman_mean | fold_spearman_sample_sd | macro_group_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | 6699.000000 | 0.657417 | 0.958609 | -0.681009 | -0.140412 | -0.151328 | 0.628364 | 0.239875 | 0.910452 | 0.265823 | -0.669715 | 0.267782 | -0.054088 | 0.279659 | -0.125137 | 0.234039 | 0.635956 |
| A1 | 6699.000000 | 0.478928 | 0.711010 | 0.075219 | 0.487819 | 0.318735 | 0.414882 | 0.132822 | 0.588445 | 0.188678 | 0.208744 | 0.488112 | 0.647951 | 0.205440 | 0.544024 | 0.233009 | 0.346276 |
| A2 | 6699.000000 | 0.415555 | 0.607458 | 0.324974 | 0.609556 | 0.575231 | 0.378334 | 0.098249 | 0.543523 | 0.140141 | 0.315232 | 0.404211 | 0.682359 | 0.149034 | 0.646307 | 0.189909 | 0.316442 |
| A3 | 6699.000000 | 0.485704 | 0.742147 | -0.007552 | 0.418587 | 0.304455 | 0.422927 | 0.139899 | 0.611938 | 0.207936 | 0.138838 | 0.550778 | 0.591970 | 0.274927 | 0.528901 | 0.290591 | 0.362094 |
| A4 | 6699.000000 | 0.418966 | 0.610094 | 0.319105 | 0.602917 | 0.576606 | 0.382625 | 0.096827 | 0.544642 | 0.143205 | 0.318558 | 0.388089 | 0.677825 | 0.148615 | 0.644635 | 0.188506 | 0.325680 |
| A5 | 6699.000000 | 0.416555 | 0.604535 | 0.331456 | 0.608864 | 0.576412 | 0.383368 | 0.093022 | 0.543920 | 0.139115 | 0.328642 | 0.356703 | 0.675246 | 0.140226 | 0.640198 | 0.190406 | 0.331796 |
| A6 | 6699.000000 | 0.413718 | 0.600242 | 0.340917 | 0.613888 | 0.581452 | 0.382076 | 0.092086 | 0.541619 | 0.138350 | 0.337809 | 0.344773 | 0.677920 | 0.135347 | 0.640270 | 0.191559 | 0.332270 |
| A2+D1 | 6699.000000 | 0.417984 | 0.610431 | 0.318351 | 0.601916 | 0.588001 | 0.382313 | 0.099374 | 0.545076 | 0.142082 | 0.317547 | 0.387280 | 0.674214 | 0.152466 | 0.650635 | 0.189970 | 0.325164 |
| A2+D2 | 6699.000000 | 0.417389 | 0.607866 | 0.324068 | 0.605662 | 0.580069 | 0.381440 | 0.095772 | 0.543163 | 0.140275 | 0.318743 | 0.392848 | 0.678742 | 0.150436 | 0.645965 | 0.189145 | 0.323686 |
| A2+D3 | 6699.000000 | 0.417593 | 0.608810 | 0.321968 | 0.609435 | 0.564431 | 0.379934 | 0.097313 | 0.546601 | 0.140779 | 0.316312 | 0.381471 | 0.679301 | 0.140558 | 0.641717 | 0.189012 | 0.318046 |
| A2+D4 | 6699.000000 | 0.418032 | 0.610954 | 0.317182 | 0.602253 | 0.579459 | 0.382186 | 0.097762 | 0.546054 | 0.141470 | 0.310394 | 0.401416 | 0.676524 | 0.152241 | 0.646819 | 0.192668 | 0.323491 |
| A2+D5 | 6699.000000 | 0.418961 | 0.614029 | 0.310292 | 0.599492 | 0.582944 | 0.380159 | 0.101213 | 0.544608 | 0.145368 | 0.313055 | 0.405270 | 0.676355 | 0.159112 | 0.652298 | 0.192342 | 0.318482 |
| A5_SHUFFLED_s1009 | 6699.000000 | 0.419615 | 0.606807 | 0.326422 | 0.604812 | 0.575768 | 0.385965 | 0.095407 | 0.545196 | 0.139024 | 0.317898 | 0.381584 | 0.675730 | 0.146353 | 0.638688 | 0.189424 | 0.328842 |
| A5_SHUFFLED_s2017 | 6699.000000 | 0.419862 | 0.609308 | 0.320856 | 0.602910 | 0.578028 | 0.386044 | 0.094950 | 0.546497 | 0.140539 | 0.311473 | 0.394871 | 0.673917 | 0.150622 | 0.641382 | 0.194024 | 0.327443 |
| A5_SHUFFLED_s3019 | 6699.000000 | 0.418332 | 0.603672 | 0.333362 | 0.610133 | 0.581119 | 0.385152 | 0.091174 | 0.542736 | 0.135722 | 0.322915 | 0.378416 | 0.678649 | 0.143991 | 0.644915 | 0.193426 | 0.330509 |

## Paired fold improvements

Positive `delta_MAE` and `delta_RMSE` mean the named candidate model wins; positive `delta_R2` and `delta_Spearman` also favor the candidate. The reference/candidate columns define direction explicitly.

| comparison | reference | candidate | outer_fold | delta_mae | delta_rmse | delta_r2 | delta_spearman |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A2_vs_A5 | A2 | A5 | 0.000000 | 0.004616 | 0.024575 | 0.102832 | -0.004851 |
| A2_vs_A5 | A2 | A5 | 1.000000 | -0.012033 | -0.007911 | -0.019208 | -0.020889 |
| A2_vs_A5 | A2 | A5 | 2.000000 | -0.022226 | -0.029752 | -0.043147 | -0.008040 |
| A2_vs_A5 | A2 | A5 | 3.000000 | 0.008196 | 0.008200 | 0.022268 | -0.001382 |
| A2_vs_A5 | A2 | A5 | 4.000000 | -0.003725 | 0.002903 | 0.004307 | 0.004620 |
| A2_vs_A6 | A2 | A6 | 0.000000 | 0.006981 | 0.030236 | 0.125893 | -0.011386 |
| A2_vs_A6 | A2 | A6 | 1.000000 | -0.010482 | -0.003128 | -0.007544 | -0.015499 |
| A2_vs_A6 | A2 | A6 | 2.000000 | -0.026714 | -0.033595 | -0.048869 | -0.010577 |
| A2_vs_A6 | A2 | A6 | 3.000000 | 0.014652 | 0.016162 | 0.043641 | 0.006883 |
| A2_vs_A6 | A2 | A6 | 4.000000 | -0.003147 | -0.000155 | -0.000231 | 0.000397 |
| A3_vs_A2 | A3 | A2 | 0.000000 | 0.006381 | -0.031881 | -0.132553 | 0.027541 |
| A3_vs_A2 | A3 | A2 | 1.000000 | 0.021978 | 0.060274 | 0.156765 | 0.035161 |
| A3_vs_A2 | A3 | A2 | 2.000000 | 0.048227 | 0.053680 | 0.079327 | 0.024431 |
| A3_vs_A2 | A3 | A2 | 3.000000 | 0.131868 | 0.231363 | 0.734417 | 0.477301 |
| A3_vs_A2 | A3 | A2 | 4.000000 | 0.014514 | 0.028638 | 0.044014 | 0.022595 |
| A2_vs_A5_SHUFFLED_s1009 | A2 | A5_SHUFFLED_s1009 | 0.000000 | -0.000238 | 0.010201 | 0.043221 | -0.004946 |
| A2_vs_A5_SHUFFLED_s1009 | A2 | A5_SHUFFLED_s1009 | 1.000000 | -0.013578 | -0.008741 | -0.021246 | -0.035138 |
| A2_vs_A5_SHUFFLED_s1009 | A2 | A5_SHUFFLED_s1009 | 2.000000 | -0.027040 | -0.018455 | -0.026525 | -0.010490 |
| A2_vs_A5_SHUFFLED_s1009 | A2 | A5_SHUFFLED_s1009 | 3.000000 | 0.003906 | 0.004107 | 0.011184 | 0.007916 |
| A2_vs_A5_SHUFFLED_s1009 | A2 | A5_SHUFFLED_s1009 | 4.000000 | -0.001209 | 0.004523 | 0.006698 | 0.004564 |
| A2_vs_A5_SHUFFLED_s2017 | A2 | A5_SHUFFLED_s2017 | 0.000000 | -0.006339 | 0.003300 | 0.014067 | -0.014649 |
| A2_vs_A5_SHUFFLED_s2017 | A2 | A5_SHUFFLED_s2017 | 1.000000 | -0.014752 | -0.006950 | -0.016852 | -0.030400 |
| A2_vs_A5_SHUFFLED_s2017 | A2 | A5_SHUFFLED_s2017 | 2.000000 | -0.019693 | -0.014949 | -0.021426 | -0.002508 |
| A2_vs_A5_SHUFFLED_s2017 | A2 | A5_SHUFFLED_s2017 | 3.000000 | 0.003382 | -0.000087 | -0.000238 | 0.022753 |
| A2_vs_A5_SHUFFLED_s2017 | A2 | A5_SHUFFLED_s2017 | 4.000000 | -0.001150 | 0.003815 | 0.005655 | 0.000179 |
| A2_vs_A5_SHUFFLED_s3019 | A2 | A5_SHUFFLED_s3019 | 0.000000 | -0.000795 | 0.012014 | 0.050824 | -0.008948 |
| A2_vs_A5_SHUFFLED_s3019 | A2 | A5_SHUFFLED_s3019 | 1.000000 | -0.020077 | -0.010750 | -0.026200 | -0.030587 |
| A2_vs_A5_SHUFFLED_s3019 | A2 | A5_SHUFFLED_s3019 | 2.000000 | -0.016646 | -0.008979 | -0.012807 | 0.001294 |
| A2_vs_A5_SHUFFLED_s3019 | A2 | A5_SHUFFLED_s3019 | 3.000000 | 0.006415 | 0.007558 | 0.020533 | 0.026263 |
| A2_vs_A5_SHUFFLED_s3019 | A2 | A5_SHUFFLED_s3019 | 4.000000 | -0.002989 | 0.004094 | 0.006066 | 0.005019 |

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
| CC(C)CCCC(C)CCCCN(C)C(=O)COCC(=O)N(C)CCCCC(C)CCCC(C)C | 91.000000 | 0.236799 | 0.235575 | 0.001224 | 0.329425 | 0.328657 | 0.000768 | 0.973244 | 0.954404 | -0.018841 |
| CC(C)CN(CC(C)C)C(=O)COCC(=O)N(CC(C)C)CC(C)C | 28.000000 | 0.303148 | 0.283332 | 0.019816 | 0.376130 | 0.360463 | 0.015667 | 0.617953 | 0.782704 | 0.164751 |
| CCCCC(CC)CN(C)C(=O)COCC(=O)N(C)CC(CC)CCCC | 91.000000 | 0.263027 | 0.273890 | -0.010863 | 0.368135 | 0.395385 | -0.027250 | 0.942188 | 0.931948 | -0.010240 |
| CCCCC(CC)CN(CC(CC)CCCC)C(=O)COCC(=O)N(CC(CC)CCCC)CC(CC)CCCC | 210.000000 | 0.390410 | 0.393315 | -0.002905 | 0.561309 | 0.564473 | -0.003164 | 0.870999 | 0.873646 | 0.002647 |
| CCCCCCC(C)N(CCCC)C(=O)COCC(=O)N(CCCC)C(C)CCCCCC | 112.000000 | 0.308060 | 0.305355 | 0.002705 | 0.448583 | 0.441704 | 0.006879 | 0.656426 | 0.678444 | 0.022018 |
| CCCCCCC(CCCC)CCCN(C)C(=O)COCC(=O)N(C)CCCC(CCCC)CCCCCC | 91.000000 | 0.220473 | 0.203453 | 0.017020 | 0.276196 | 0.256214 | 0.019982 | 0.959787 | 0.963513 | 0.003727 |
| CCCCCCCCC(C)(C)SCC1CN(C(=O)COCC(=O)N2CC(C)C(CSC(C)(C)CCCCCCCC)C2)CC1C | 91.000000 | 0.201046 | 0.227960 | -0.026914 | 0.286380 | 0.317650 | -0.031269 | 0.986893 | 0.985555 | -0.001338 |
| CCCCCCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)N(CCCCCCCC)CCCCCCCCCCCC | 78.000000 | 0.548559 | 0.573215 | -0.024656 | 0.693373 | 0.709397 | -0.016023 | 0.932346 | 0.928780 | -0.003566 |
| CCCCCCCCCCCCN(CCCCCCCCCCCC)C(=O)COCC(=O)N(CCCCCCCCCCCC)CCCCCCCCCCCC | 16.000000 | 0.214673 | 0.253012 | -0.038339 | 0.282682 | 0.284696 | -0.002014 | 0.544118 | 0.476471 | -0.067647 |
| CCCCCCCCCCCCN(CCCCCCCCCCCC)C(=O)COCCOCC(=O)N(CCCCCCCCCCCC)CCCCCCCCCCCC | 79.000000 | 0.780585 | 0.819555 | -0.038970 | 1.119696 | 1.228746 | -0.109050 | -0.297620 | -0.446138 | -0.148518 |
| CCCCCCCCCCN(CCCCCCCCCC)C(=O)C(C)OC(C)C(=O)N(CCCCCCCCCC)CCCCCCCCCC | 140.000000 | 0.125505 | 0.206589 | -0.081084 | 0.153001 | 0.266438 | -0.113437 | 0.877922 | 0.745547 | -0.132375 |
| CCCCCCCCCCN(CCCCCCCCCC)C(=O)COCC(=O)N(CCCCCC)CCCCCC | 298.000000 | 0.634888 | 0.605237 | 0.029651 | 0.764342 | 0.723376 | 0.040966 | 0.458929 | 0.450787 | -0.008142 |
| CCCCCCCCN(C)C(=O)COCC(=O)N(C)CCCCCCCC | 507.000000 | 0.511430 | 0.523057 | -0.011627 | 0.673908 | 0.685653 | -0.011745 | 0.824014 | 0.829266 | 0.005252 |
| CCCCCCCCN(C)C(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC | 91.000000 | 0.118832 | 0.150952 | -0.032120 | 0.147607 | 0.185154 | -0.037547 | 0.985221 | 0.961761 | -0.023459 |
| CCCCCCCCN(CC(C)C)C(=O)COCC(=O)N(CCCCCCCC)CC(C)C | 28.000000 | 0.115528 | 0.137331 | -0.021803 | 0.141763 | 0.170509 | -0.028746 | 0.852217 | 0.835796 | -0.016420 |
| CCCCCCCCN(CC(CC)CCCC)C(=O)COCC(=O)N(CCCCCCCC)CC(CC)CCCC | 28.000000 | 0.096835 | 0.125901 | -0.029066 | 0.116701 | 0.157084 | -0.040383 | 0.845101 | 0.812808 | -0.032293 |
| CCCCCCCCN(CC)C(=O)COCC(=O)N(CC)CCCCCCCC | 91.000000 | 0.097697 | 0.155222 | -0.057525 | 0.121493 | 0.195143 | -0.073650 | 0.983501 | 0.959054 | -0.024447 |
| CCCCCCCCN(CCC(C)CC(C)(C)C)C(=O)COCC(=O)N(CCCCCCCC)CCC(C)CC(C)(C)C | 91.000000 | 0.225634 | 0.224307 | 0.001327 | 0.324175 | 0.306834 | 0.017341 | 0.973515 | 0.973069 | -0.000446 |
| CCCCCCCCN(CCC)C(=O)COCC(=O)N(CCC)CCCCCCCC | 91.000000 | 0.103685 | 0.155039 | -0.051354 | 0.132527 | 0.196215 | -0.063688 | 0.982497 | 0.966587 | -0.015910 |
| CCCCCCCCN(CCCC(CCCC)CCCCCC)C(=O)COCC(=O)N(CCCCCCCC)CCCC(CCCC)CCCCCC | 78.000000 | 0.291364 | 0.288475 | 0.002889 | 0.355205 | 0.343516 | 0.011689 | 0.970713 | 0.958649 | -0.012064 |
| CCCCCCCCN(CCCCC(C)CCCC(C)C)C(=O)COCC(=O)N(CCCCCCCC)CCCCC(C)CCCC(C)C | 91.000000 | 0.197353 | 0.228308 | -0.030954 | 0.285459 | 0.318440 | -0.032980 | 0.988342 | 0.978850 | -0.009492 |
| CCCCCCCCN(CCCCCCCC)C(=O)CN(CC(=O)N(CCCCCCCC)CCCCCCCC)CC(=O)N(CCCCCCCC)CCCCCCCC | 1.000000 | 0.619205 | 0.717286 | -0.098080 | 0.619205 | 0.717286 | -0.098080 | NA | NA | NA |
| CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)N(C)C | 21.000000 | 0.382144 | 0.430154 | -0.048011 | 0.570318 | 0.603209 | -0.032891 | 0.857143 | 0.872727 | 0.015584 |
| CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC | 2921.000000 | 0.504750 | 0.496554 | 0.008196 | 0.712946 | 0.704746 | 0.008200 | 0.613442 | 0.612059 | -0.001382 |
| CCCCCCCCN(CCCCCCCC)C(=O)[C@H](C)O[C@@H](C)C(=O)N(CCCCCCCC)CCCCCCCC | 91.000000 | 0.404074 | 0.398697 | 0.005377 | 0.530992 | 0.530550 | 0.000441 | 0.902262 | 0.830021 | -0.072241 |
| CCCCCCCCN(CCCCCCCC)C(=O)[C@H](C)O[C@H](C)C(=O)N(CCCCCCCC)CCCCCCCC | 91.000000 | 0.174945 | 0.218641 | -0.043696 | 0.232014 | 0.276481 | -0.044467 | 0.692512 | 0.668182 | -0.024329 |
| CCCCCCN(CCCCCC)C(=O)COCC(=O)N(CCCCCC)CCCCCC | 21.000000 | 0.085943 | 0.113932 | -0.027989 | 0.095791 | 0.123959 | -0.028168 | 0.928571 | 0.896104 | -0.032468 |
| CCCCN(CCCC)C(=O)COCC(=O)N(CCCC)CCCC | 172.000000 | 0.420935 | 0.415853 | 0.005082 | 0.546836 | 0.529047 | 0.017789 | 0.682837 | 0.674071 | -0.008766 |
| CCCN(CCC)C(=O)COCC(=O)N(CCC)CCC | 150.000000 | 0.172713 | 0.212257 | -0.039544 | 0.236372 | 0.277216 | -0.040844 | 0.740064 | 0.673017 | -0.067047 |
| CCN(CC)C(=O)COCC(=O)N(CC)CC | 258.000000 | 0.204463 | 0.198521 | 0.005942 | 0.275253 | 0.261572 | 0.013681 | 0.271040 | 0.314223 | 0.043184 |

_Only the first 30 of 34 rows are shown._

## Lanthanide comparison

| Ln | n_samples | MAE_2D | MAE_2D3D | delta_MAE | RMSE_2D | RMSE_2D3D | delta_RMSE | Spearman_2D | Spearman_2D3D | delta_Spearman |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ce | 913.000000 | 0.569812 | 0.570941 | -0.001128 | 0.768947 | 0.767123 | 0.001824 | 0.684817 | 0.680105 | -0.004712 |
| Dy | 1123.000000 | 0.349918 | 0.346431 | 0.003487 | 0.535375 | 0.524414 | 0.010961 | 0.728788 | 0.737304 | 0.008516 |
| Er | 1045.000000 | 0.365917 | 0.373162 | -0.007245 | 0.580304 | 0.582666 | -0.002362 | 0.686884 | 0.663243 | -0.023641 |
| Eu | 949.000000 | 0.363726 | 0.365550 | -0.001823 | 0.499437 | 0.495889 | 0.003549 | 0.799891 | 0.782770 | -0.017121 |
| Gd | 1144.000000 | 0.364210 | 0.355141 | 0.009069 | 0.493315 | 0.478884 | 0.014430 | 0.691569 | 0.684244 | -0.007325 |
| Ho | 929.000000 | 0.388895 | 0.395604 | -0.006709 | 0.601672 | 0.603590 | -0.001918 | 0.713784 | 0.702410 | -0.011374 |
| La | 923.000000 | 0.643988 | 0.640803 | 0.003185 | 0.875367 | 0.876625 | -0.001258 | 0.595170 | 0.585420 | -0.009750 |
| Lu | 849.000000 | 0.354508 | 0.360558 | -0.006050 | 0.554221 | 0.562366 | -0.008145 | 0.719592 | 0.695501 | -0.024091 |
| Nd | 1095.000000 | 0.518212 | 0.510760 | 0.007452 | 0.699128 | 0.683832 | 0.015296 | 0.605006 | 0.606470 | 0.001464 |
| Pr | 704.000000 | 0.436112 | 0.439548 | -0.003436 | 0.604518 | 0.592557 | 0.011961 | 0.770076 | 0.778450 | 0.008374 |
| Sm | 1120.000000 | 0.420378 | 0.414185 | 0.006193 | 0.545577 | 0.538270 | 0.007307 | 0.674802 | 0.687750 | 0.012948 |
| Tb | 897.000000 | 0.351654 | 0.356813 | -0.005159 | 0.521734 | 0.511938 | 0.009796 | 0.787882 | 0.768401 | -0.019481 |
| Tm | 827.000000 | 0.343224 | 0.354681 | -0.011458 | 0.567892 | 0.582995 | -0.015103 | 0.746833 | 0.728475 | -0.018358 |
| Yb | 880.000000 | 0.351095 | 0.359618 | -0.008523 | 0.558102 | 0.567772 | -0.009670 | 0.759157 | 0.745277 | -0.013880 |

## Geometry quality

The machine-readable `geometry_qc_summary.json` states which quality strata are actually represented. Failed geometries are excluded by pair construction and are never silently imputed into a 3D arm. A high-versus-low-confidence claim is unavailable when either stratum has no eligible pairs.

## Audit interpretation

This run does not declare a positive 3D result merely from one metric. Review the paired A2-vs-A5 MAE first, then RMSE, R2, Spearman, extractant win fraction, shuffle controls, geometry quality, and seed-to-seed aggregation. `validation.json` and `leakage_audit.json` must both pass before scientific interpretation.
