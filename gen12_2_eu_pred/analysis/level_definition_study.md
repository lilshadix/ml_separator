# Level-definition study — training rows only

Development fold: design B, split seed 104729, fold 0. 1148 training rows, 153 training extractants, 77 training chemotypes. No held-out row and no model output entered any number below.

## The five candidate definitions

|                   |    mean |     sd |     min |    max | defined_for_all   | defined_for_one_cell   |
|:------------------|--------:|-------:|--------:|-------:|:------------------|:-----------------------|
| LVL_MEAN          | -0.2812 | 1.787  | -3.5986 | 3.0208 | True              | True                   |
| LVL_MEDIAN        | -0.2711 | 1.8362 | -4      | 3.0916 | True              | True                   |
| LVL_FE_INTERCEPT  | -0.4503 | 1.714  | -4.1615 | 4.1877 | True              | True                   |
| LVL_SHRUNK        | -0.31   | 1.549  | -3.4869 | 2.9158 | True              | True                   |
| LVL_COND_RESIDUAL |  0.3872 | 1.462  | -3.1643 | 4.6153 | True              | True                   |

## Agreement between definitions (Spearman)

|                   |   LVL_MEAN |   LVL_MEDIAN |   LVL_FE_INTERCEPT |   LVL_SHRUNK |   LVL_COND_RESIDUAL |
|:------------------|-----------:|-------------:|-------------------:|-------------:|--------------------:|
| LVL_MEAN          |      1     |        0.993 |              0.947 |        0.995 |               0.743 |
| LVL_MEDIAN        |      0.993 |        1     |              0.942 |        0.991 |               0.747 |
| LVL_FE_INTERCEPT  |      0.947 |        0.942 |              1     |        0.952 |               0.761 |
| LVL_SHRUNK        |      0.995 |        0.991 |              0.952 |        1     |               0.745 |
| LVL_COND_RESIDUAL |      0.743 |        0.747 |              0.761 |        0.745 |               1     |

## Criterion 2 — split-half reliability (extractants with at least four cells)

| definition        |   n_extractants |   spearman |   pearson |   mean_abs_half_difference |   median_abs_half_difference |
|:------------------|----------------:|-----------:|----------:|---------------------------:|-----------------------------:|
| LVL_MEAN          |              53 |     0.9197 |    0.926  |                     0.4692 |                       0.3018 |
| LVL_MEDIAN        |              53 |     0.8916 |    0.899  |                     0.5619 |                       0.2945 |
| LVL_FE_INTERCEPT  |              53 |     0.9132 |    0.9033 |                     0.5476 |                       0.3488 |
| LVL_SHRUNK        |              53 |     0.9182 |    0.9287 |                     0.4418 |                       0.2483 |
| LVL_COND_RESIDUAL |              53 |     0.8413 |    0.844  |                     0.5877 |                       0.3903 |

## Criterion 3 — condition-sampling distortion

|                           |    value |
|:--------------------------|---------:|
| n_extractants             | 153      |
| median_abs_shift          |   0.7635 |
| mean_abs_shift            |   1.0692 |
| frac_above_0_5_decades    |   0.7386 |
| frac_above_1_0_decades    |   0.4314 |
| spearman_mean_vs_residual |   0.7429 |
| shift_vs_n_cells_spearman |   0.0878 |

### Is the adjustment removing conditions, or removing level?

|                                          |   value |
|:-----------------------------------------|--------:|
| pooled_r2_of_condition_model             |  0.0895 |
| between_variance_of_level                |  3.1935 |
| between_variance_of_condition_prediction |  1.4035 |
| between_variance_share                   |  0.4395 |
| spearman_level_vs_condition_prediction   |  0.551  |

| n_cells   |   n_extractants |   median_abs_shift |
|:----------|----------------:|-------------------:|
| 1         |              75 |           0.628668 |
| 2-3       |              19 |           2.03236  |
| 4-6       |              25 |           1.67071  |
| 7-20      |              25 |           1.06295  |
| 21+       |               9 |           0.33069  |

## Level-reliable cohort threshold

|   n_cells |   se_of_mean |   se_over_tau |
|----------:|-------------:|--------------:|
|         1 |       0.9714 |        0.5436 |
|         2 |       0.6869 |        0.3844 |
|         3 |       0.5609 |        0.3138 |
|         4 |       0.4857 |        0.2718 |
|         5 |       0.4344 |        0.2431 |
|         6 |       0.3966 |        0.2219 |
|         7 |       0.3672 |        0.2055 |
|         8 |       0.3435 |        0.1922 |
|         9 |       0.3238 |        0.1812 |
|        10 |       0.3072 |        0.1719 |


sigma_within = 0.9714, tau_between = 1.7870, threshold = **5 cells**.

## Decision

|                                 | value                                                 |
|:--------------------------------|:------------------------------------------------------|
| primary_level_definition        | LVL_MEAN                                              |
| reason                          | default: no override triggered under the amended rule |
| secondary_level_definition      | LVL_COND_RESIDUAL                                     |
| override_1_triggered            | False                                                 |
| override_2_shift_large          | True                                                  |
| override_2_adjustment_is_clean  | False                                                 |
| override_2_reliability_holds    | False                                                 |
| override_2_triggered            | False                                                 |
| original_rule_would_have_chosen | LVL_COND_RESIDUAL                                     |
| best_reliability_definition     | LVL_MEAN                                              |
| best_reliability_spearman       | 0.9197047978266187                                    |
| lvl_mean_reliability_spearman   | 0.9197047978266187                                    |
