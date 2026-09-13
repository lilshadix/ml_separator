# H1 — is the level the bottleneck?

*Every number from Gen12's frozen design-B predictions, zero-shot, extractant-macro MAE, level and shape centred per (split_seed, extractant).*

## The target itself

|                                 |   value |
|:--------------------------------|--------:|
| between_extractant_share        |  0.6324 |
| within_extractant_share         |  0.3676 |
| target_sd                       |  1.8978 |
| between_extractant_sd_of_levels |  1.7465 |

## What each Gen12 arm's error is made of, and what fixing each half buys

| arm                                    |   macro_mae |   level_component |   shape_component |   level_share_of_components |   with_true_level_own_shape |   with_true_shape_own_level |   true_level_no_shape_model |   recoverable_by_fixing_level |   recoverable_by_fixing_shape |   spearman_level_error_vs_mae |   share_of_sse_in_level |
|:---------------------------------------|------------:|------------------:|------------------:|----------------------------:|----------------------------:|----------------------------:|----------------------------:|------------------------------:|------------------------------:|------------------------------:|------------------------:|
| GEN12_ABL_D (best observed test)       |      1.0478 |            0.917  |            0.5783 |                      0.6133 |                      0.3097 |                      0.917  |                      0.3345 |                        0.7381 |                        0.1308 |                        0.8833 |                  0.5601 |
| GEN12_ABL_C (pre-registered selection) |      1.0916 |            0.9769 |            0.5689 |                      0.632  |                      0.3046 |                      0.9769 |                      0.3345 |                        0.787  |                        0.1148 |                        0.894  |                  0.573  |
| GEN12_ABL_A (no chemistry)             |      1.1341 |            1.0287 |            0.612  |                      0.627  |                      0.3277 |                      1.0287 |                      0.3345 |                        0.8064 |                        0.1054 |                        0.9166 |                  0.572  |
| GEN12_ABL_E (fingerprint + conditions) |      1.107  |            0.9937 |            0.5615 |                      0.6389 |                      0.3007 |                      0.9937 |                      0.3345 |                        0.8063 |                        0.1133 |                        0.8941 |                  0.5795 |
| GEN12_B1_COND_ONLY                     |      1.1666 |            1.0529 |            0.6491 |                      0.6186 |                      0.3476 |                      1.0529 |                      0.3345 |                        0.819  |                        0.1137 |                        0.9102 |                  0.5894 |
| GEN12_B0_GLOBAL_MEAN                   |      1.7309 |            1.6408 |            0.6247 |                      0.7243 |                      0.3345 |                      1.6408 |                      0.3345 |                        1.3963 |                        0.09   |                        0.9634 |                  0.7103 |

`with_true_level_own_shape` replaces each held-out extractant's predicted level with its true level and keeps the model's own within-extractant shape. `with_true_shape_own_level` does the reverse. `true_level_no_shape_model` predicts the true level for every row and models no condition response at all.
