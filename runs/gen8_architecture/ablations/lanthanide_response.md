# Is the lanthanide response continuous?

*Four encodings of the same fourteen metals, one ExtraTrees learner, the frozen chemotype-blocked fold plan, identical rows. Shape metrics are computed **within each reconstructed metal-series curve** — one ligand at one set of conditions across the 4f series — so they measure whether the model gets the *order and spacing* of the lanthanides right, not whether it gets the ligand's level right.*

| arm | macro_mae | shape_mae | curve_shape_mae | curve_spearman | curve_sign_accuracy | n_features | n_seeds |
|---|---|---|---|---|---|---|---|
| STRUCTURED | 1.240 | 0.553 | 0.334 | 0.330 | 0.650 | 3 | 3 |
| LIGAND_RBF | 1.242 | 0.546 | 0.336 | 0.328 | 0.656 | 117 | 3 |
| SHARED_RBF | 1.237 | 0.553 | 0.339 | 0.333 | 0.650 | 13 | 3 |
| ONEHOT | 1.244 | 0.556 | 0.368 | 0.279 | 0.622 | 14 | 3 |

## Paired chemotype bootstrap against the one-hot encoding

Positive = the structured representation is better.

| statistic | comparison | point | ci95_low | ci95_high | bca_low | bca_high | block_macro | n_units | units_improved | seeds_positive | n_seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| abs_residual | STRUCTURED vs ONEHOT | 0.004 | -0.026 | 0.026 | -0.025 | 0.027 | 0.007 | 152 | 81 | 2 | 3 |
| abs_residual | SHARED_RBF vs ONEHOT | 0.007 | -0.027 | 0.031 | -0.022 | 0.036 | 0.005 | 152 | 81 | 3 | 3 |
| abs_residual | LIGAND_RBF vs ONEHOT | 0.002 | -0.042 | 0.034 | -0.040 | 0.036 | -0.007 | 152 | 85 | 2 | 3 |
| shape_residual | STRUCTURED vs ONEHOT | 0.003 | -0.008 | 0.015 | -0.008 | 0.015 | 0.005 | 152 | 85 | 3 | 3 |
| shape_residual | SHARED_RBF vs ONEHOT | 0.003 | -0.009 | 0.015 | -0.009 | 0.015 | 0.004 | 152 | 87 | 3 | 3 |
| shape_residual | LIGAND_RBF vs ONEHOT | 0.010 | -0.005 | 0.024 | -0.005 | 0.024 | 0.011 | 152 | 86 | 2 | 3 |

