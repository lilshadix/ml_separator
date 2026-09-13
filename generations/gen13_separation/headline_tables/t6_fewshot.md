*T6 one-pair calibration: macro MAE on the remaining pairs of cells with >= 3 metals, 5 support draws per cell*

| arm                                        |   macro_mae_extractant |   seed_sd |
|:-------------------------------------------|-----------------------:|----------:|
| M_PHYSICS_radius+radius_sq|basis_shift     |                 0.3963 |    0.0095 |
| C_DIRECT_ROW|rescale                       |                 0.4358 |    0.0195 |
| M_PHYSICS_radius+radius_sq|rescale         |                 0.4376 |    0.0098 |
| M_SELECTED|rescale                         |                 0.4396 |    0.0131 |
| M_SELECTED|basis_shift                     |                 0.4526 |    0.0368 |
| C_DIRECT_ROW|zero_shot                     |                 0.4787 |    0.0196 |
| M_PHYSICS_radius+radius_sq|zero_shot       |                 0.4805 |    0.0084 |
| B1_MEAN_CURVE|no_model_linear              |                 0.4832 |    0.0126 |
| M_SELECTED|no_model_linear                 |                 0.4832 |    0.0126 |
| C_DIRECT_ROW|no_model_linear               |                 0.4832 |    0.0126 |
| M_PHYSICS_radius+radius_sq|no_model_linear |                 0.4832 |    0.0126 |
| M_SELECTED|zero_shot                       |                 0.4844 |    0.0065 |
| B1_MEAN_CURVE|rescale                      |                 0.5404 |    0.0182 |
| B1_MEAN_CURVE|zero_shot                    |                 0.588  |    0.0161 |
