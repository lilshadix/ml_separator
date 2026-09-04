# Calibration geography — where to spend the one measurement

*Model `REC_ecfp_plus_recovered`, 5 split seeds, 143 held-out ligands, 26,105 candidate rows, each scored exactly (no sampling).*

Reference on this population: zero-shot **0.995**, one-shot averaged over every candidate (= RANDOM) **0.709**, oracle level **0.480**.

## Where in the acid range the measurement sits

| acid_position | one_shot_mae | n_ligands | n_candidate_rows |
|---|---|---|---|
| single | 0.455 | 61 | 4,175 |
| mid | 0.832 | 77 | 7,370 |
| high | 0.873 | 81 | 7,395 |
| low | 1.046 | 77 | 7,150 |
| missing | 2.193 | 2 | 15 |

## Where in the extractant-concentration range it sits

| extractant_position | one_shot_mae | n_ligands | n_candidate_rows |
|---|---|---|---|
| single | 0.567 | 102 | 6,630 |
| mid | 1.010 | 34 | 8,305 |
| low | 1.123 | 33 | 4,820 |
| high | 1.167 | 36 | 6,345 |
| missing | 3.095 | 1 | 5 |

## Central vs extreme lanthanide

| metal_position | one_shot_mae | n_ligands | n_candidate_rows |
|---|---|---|---|
| central | 0.575 | 87 | 13,655 |
| extreme | 0.673 | 85 | 9,465 |
| single | 0.852 | 56 | 2,985 |

## Where in the model's own predicted range it sits

| prediction_position | one_shot_mae | n_ligands | n_candidate_rows |
|---|---|---|---|
| mid | 0.653 | 143 | 8,823 |
| high | 0.697 | 142 | 8,879 |
| low | 0.796 | 143 | 8,403 |

## Which kind of series the measured point belongs to

| primary_axis | one_shot_mae | n_ligands | n_candidate_rows |
|---|---|---|---|
| metal_series | 0.601 | 86 | 13,395 |
| contact_time | 0.714 | 8 | 255 |
| acid | 0.957 | 69 | 7,750 |
| metal_concentration | 1.026 | 4 | 425 |
| none | 1.118 | 31 | 560 |
| extractant | 1.151 | 25 | 3,510 |
| temperature | 1.252 | 5 | 210 |

## DIAGNOSTIC ONLY (not deployable): central vs extreme residual

| residual_position | one_shot_mae | n_ligands | n_candidate_rows |
|---|---|---|---|
| central | 0.554 | 143 | 13,177 |
| extreme | 0.873 | 143 | 12,928 |

## Cross-series transfer matrix

Rows: the curve type the *measured* point belongs to. Columns: the curve type of the rows being *predicted*. Values are one-shot MAE (lower is better); `gain` is the improvement over zero-shot on the same target rows.

| calibration_axis | target_axis | mae | zero_shot | n_ligands | gain |
|---|---|---|---|---|---|
| acid | acid | 0.921 | 1.111 | 69 | 0.190 |
| acid | contact_time | 0.935 | 1.627 | 7 | 0.692 |
| acid | extractant | 1.238 | 1.210 | 24 | -0.028 |
| acid | metal_concentration | 1.211 | 0.612 | 3 | -0.599 |
| acid | metal_series | 1.113 | 0.951 | 18 | -0.161 |
| acid | none | 1.101 | 0.939 | 21 | -0.162 |
| acid | temperature | 0.989 | 0.949 | 4 | -0.040 |
| contact_time | acid | 0.935 | 1.350 | 7 | 0.415 |
| contact_time | contact_time | 0.095 | 1.864 | 8 | 1.770 |
| contact_time | extractant | 1.248 | 1.854 | 5 | 0.606 |
| contact_time | metal_concentration | 2.226 | 0.620 | 1 | -1.606 |
| contact_time | metal_series | 2.179 | 1.011 | 1 | -1.168 |
| contact_time | none | 0.720 | 2.035 | 3 | 1.315 |
| contact_time | temperature | 0.806 | 1.807 | 1 | 1.001 |
| extractant | acid | 1.238 | 1.172 | 24 | -0.066 |
| extractant | contact_time | 1.248 | 2.053 | 5 | 0.805 |
| extractant | extractant | 0.972 | 1.214 | 25 | 0.243 |
| extractant | metal_concentration | 1.232 | 0.612 | 3 | -0.621 |
| extractant | metal_series | 1.148 | 0.968 | 11 | -0.180 |
| extractant | none | 1.345 | 1.010 | 18 | -0.336 |
| extractant | temperature | 1.117 | 0.835 | 5 | -0.282 |
| metal_concentration | acid | 1.211 | 1.044 | 3 | -0.167 |
| metal_concentration | contact_time | 2.226 | 2.516 | 1 | 0.289 |
| metal_concentration | extractant | 1.232 | 1.276 | 3 | 0.043 |
| metal_concentration | metal_concentration | 0.634 | 0.589 | 4 | -0.045 |
| metal_concentration | metal_series | 1.017 | 0.720 | 3 | -0.297 |
| metal_concentration | none | 1.342 | 1.310 | 4 | -0.033 |
| metal_concentration | temperature | 1.605 | 1.807 | 1 | 0.201 |
| metal_series | acid | 1.113 | 1.077 | 18 | -0.036 |
| metal_series | contact_time | 2.179 | 2.516 | 1 | 0.336 |
| metal_series | extractant | 1.148 | 1.164 | 11 | 0.016 |
| metal_series | metal_concentration | 1.017 | 0.612 | 3 | -0.405 |
| metal_series | metal_series | 0.555 | 0.887 | 85 | 0.332 |
| metal_series | none | 1.434 | 1.345 | 14 | -0.088 |
| metal_series | temperature | 1.357 | 0.835 | 5 | -0.522 |
| none | acid | 1.101 | 1.123 | 21 | 0.022 |
| none | contact_time | 0.720 | 2.542 | 3 | 1.823 |
| none | extractant | 1.345 | 1.130 | 18 | -0.215 |
| none | metal_concentration | 1.342 | 0.589 | 4 | -0.753 |
| none | metal_series | 1.434 | 0.844 | 14 | -0.590 |
| none | none | 1.022 | 0.905 | 14 | -0.118 |
| none | temperature | 1.208 | 0.891 | 4 | -0.317 |
| temperature | acid | 0.989 | 0.788 | 4 | -0.200 |
| temperature | contact_time | 0.806 | 2.516 | 1 | 1.710 |
| temperature | extractant | 1.117 | 0.979 | 5 | -0.138 |
| temperature | metal_concentration | 1.605 | 0.620 | 1 | -0.985 |
| temperature | metal_series | 1.357 | 1.134 | 5 | -0.223 |
| temperature | none | 1.208 | 1.230 | 4 | 0.022 |
| temperature | temperature | 0.810 | 0.835 | 5 | 0.025 |

