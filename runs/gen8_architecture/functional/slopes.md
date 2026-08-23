# Slopes, curvature and the lanthanide response

*Every reconstructed curve scored twice: once from the measured `log D`, once from the frozen model's out-of-fold prediction on the same rows. `slope_mae_median_null` is what a model scores if it predicts every curve's slope as the median slope of that curve type — the null a slope model has to beat.*

## Slope accuracy by curve type

| model | curve_type | n_curves | n_ligands | slope_true_median | slope_pred_median | slope_mae | slope_pearson | slope_spearman | slope_sign_agree | y_span_true_median | y_span_pred_median | span_ratio_median | linear_r2_true_median | linear_r2_pred_median | curvature_pearson | curvature_sign_agree | n_curvature | slope_mae_median_null |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NULL_metal_cond | acid | 2,105 | 68 | 1.656 | 0.406 | 1.494 | 0.236 | 0.106 | 0.762 | 1.927 | 0.628 | 0.323 | 0.917 | 0.787 | 0.101 | 0.538 | 2,105 | 1.048 |
| NULL_metal_cond | contact_time | 40 | 8 | 0.000 | 0.000 | 0.003 | -0.004 | 0.002 | 0.500 | 0.063 | 0.019 | 0.266 | 0.165 | 0.706 | -0.071 | 0.525 | 40 | 0.002 |
| NULL_metal_cond | extractant | 775 | 25 | 2.574 | 0.139 | 2.407 | -0.197 | -0.281 | 0.732 | 2.074 | 0.160 | 0.088 | 0.994 | 0.724 | 0.065 | 0.459 | 775 | 0.743 |
| NULL_metal_cond | metal_concentration | 45 | 4 | -1.123 | 0.018 | 0.863 | -0.003 | -0.014 | 0.311 | 0.729 | 0.022 | 0.025 | 0.752 | 0.742 | 0.038 | 0.511 | 45 | 0.685 |
| NULL_metal_cond | metal_series | 1,605 | 84 | 0.084 | 0.045 | 0.100 | 0.257 | 0.256 | 0.758 | 1.026 | 0.681 | 0.634 | 0.874 | 0.806 | 0.094 | 0.681 | 1,605 | 0.092 |
| NULL_metal_cond | temperature | 160 | 5 | -0.040 | -0.000 | 0.042 | -0.200 | -0.290 | 0.575 | 1.177 | 0.191 | 0.148 | 0.989 | 0.568 | -0.064 | 0.531 | 160 | 0.010 |
| REC_ecfp_plus_recovered | acid | 2,105 | 68 | 1.656 | 0.355 | 1.567 | 0.171 | 0.192 | 0.829 | 1.927 | 0.442 | 0.218 | 0.917 | 0.947 | 0.033 | 0.549 | 2,105 | 1.048 |
| REC_ecfp_plus_recovered | contact_time | 40 | 8 | 0.000 | 0.000 | 0.002 | 0.279 | 0.305 | 0.775 | 0.063 | 0.005 | 0.050 | 0.165 | 0.737 | 0.000 | 0.600 | 40 | 0.002 |
| REC_ecfp_plus_recovered | extractant | 775 | 25 | 2.574 | 0.116 | 2.433 | -0.055 | -0.230 | 0.850 | 2.074 | 0.109 | 0.051 | 0.994 | 0.820 | 0.017 | 0.493 | 775 | 0.743 |
| REC_ecfp_plus_recovered | metal_concentration | 45 | 4 | -1.123 | -0.000 | 0.848 | 0.060 | -0.024 | 0.556 | 0.729 | 0.006 | 0.011 | 0.752 | 0.606 | -0.191 | 0.356 | 45 | 0.685 |
| REC_ecfp_plus_recovered | metal_series | 1,605 | 84 | 0.084 | 0.040 | 0.094 | 0.351 | 0.376 | 0.771 | 1.026 | 0.403 | 0.379 | 0.874 | 0.859 | 0.100 | 0.680 | 1,605 | 0.092 |
| REC_ecfp_plus_recovered | temperature | 160 | 5 | -0.040 | 0.000 | 0.039 | -0.127 | -0.078 | 0.438 | 1.177 | 0.085 | 0.064 | 0.989 | 0.600 | -0.152 | 0.450 | 160 | 0.010 |
| TREE_MC_lig2d_ext_massaction | acid | 2,105 | 68 | 1.656 | 0.341 | 1.498 | 0.230 | 0.270 | 0.837 | 1.927 | 0.399 | 0.228 | 0.917 | 0.948 | 0.111 | 0.569 | 2,105 | 1.048 |
| TREE_MC_lig2d_ext_massaction | contact_time | 40 | 8 | 0.000 | 0.000 | 0.002 | 0.365 | 0.539 | 0.800 | 0.063 | 0.014 | 0.075 | 0.165 | 0.507 | 0.030 | 0.475 | 40 | 0.002 |
| TREE_MC_lig2d_ext_massaction | extractant | 775 | 25 | 2.574 | 0.121 | 2.444 | -0.087 | -0.162 | 0.912 | 2.074 | 0.105 | 0.052 | 0.994 | 0.837 | 0.033 | 0.428 | 775 | 0.743 |
| TREE_MC_lig2d_ext_massaction | metal_concentration | 45 | 4 | -1.123 | 0.000 | 0.849 | -0.120 | 0.024 | 0.578 | 0.729 | 0.005 | 0.004 | 0.752 | 0.533 | -0.093 | 0.511 | 45 | 0.685 |
| TREE_MC_lig2d_ext_massaction | metal_series | 1,605 | 84 | 0.084 | 0.037 | 0.098 | 0.263 | 0.308 | 0.816 | 1.026 | 0.386 | 0.300 | 0.874 | 0.866 | 0.108 | 0.712 | 1,605 | 0.092 |
| TREE_MC_lig2d_ext_massaction | temperature | 160 | 5 | -0.040 | 0.000 | 0.041 | -0.263 | -0.183 | 0.394 | 1.177 | 0.098 | 0.072 | 0.989 | 0.535 | 0.062 | 0.450 | 160 | 0.010 |

## Is the lanthanide response reproduced as a curve?

| model | n_curves | slope_spearman | slope_sign_agree | flat_prediction_share | flat_truth_share | turning_point_true_share | turning_point_pred_share |
|---|---|---|---|---|---|---|---|
| NULL_metal_cond | 1,605 | 0.256 | 0.758 | 0.047 | 0.019 | 0.489 | 0.617 |
| REC_ecfp_plus_recovered | 1,605 | 0.376 | 0.771 | 0.123 | 0.019 | 0.489 | 0.572 |
| TREE_MC_lig2d_ext_massaction | 1,605 | 0.308 | 0.816 | 0.131 | 0.019 | 0.489 | 0.606 |

