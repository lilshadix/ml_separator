# Does mechanism-aware similarity fix the zero-shot level?

*Target: the **condition-adjusted** ligand level — the ligand's mean residual after a global metal + conditions model fitted on the fold's training rows. This is gen7's decisive negative-result benchmark, rerun with a different notion of chemical neighbourhood.*

## How much do the three distances disagree?

| pair | spearman |
|---|---|
| spearman_mech_vs_tanimoto | 0.593 |
| spearman_mech_vs_donor | 0.714 |
| spearman_donor_vs_tanimoto | 0.645 |

If mechanistic distance were a relabelling of Tanimoto these would be near 1.0.

## Predicting the condition-adjusted level

| model | mae | mae_sd | macro_mae | r2 | n_folds |
|---|---|---|---|---|---|
| 1NN_tanimoto | 0.989 | 0.206 | 0.973 | 0.103 | 25 |
| ET_mech | 1.009 | 0.204 | 1.003 | 0.107 | 25 |
| 3NN_tanimoto | 1.022 | 0.222 | 1.024 | 0.087 | 25 |
| KERNEL_donors | 1.024 | 0.223 | 1.019 | 0.096 | 25 |
| ET_donors | 1.035 | 0.209 | 1.029 | 0.079 | 25 |
| 5NN_mechanism | 1.040 | 0.211 | 1.039 | 0.075 | 25 |
| 1NN_donors | 1.042 | 0.205 | 1.037 | 0.036 | 25 |
| 5NN_tanimoto | 1.048 | 0.225 | 1.049 | 0.058 | 25 |
| 3NN_donors | 1.051 | 0.202 | 1.047 | 0.047 | 25 |
| RIDGE_donors | 1.052 | 0.211 | 1.047 | 0.035 | 25 |
| 3NN_mechanism | 1.056 | 0.204 | 1.050 | 0.062 | 25 |
| KERNEL_mechanism | 1.064 | 0.222 | 1.059 | 0.024 | 25 |
| 5NN_donors | 1.070 | 0.222 | 1.066 | 0.019 | 25 |
| KERNEL_tanimoto | 1.070 | 0.222 | 1.066 | 0.008 | 25 |
| NULL_global_mean | 1.074 | 0.222 | 1.070 | 0.000 | 25 |
| RIDGE_mech | 1.075 | 0.216 | 1.071 | 0.001 | 25 |
| 1NN_mechanism | 1.077 | 0.216 | 1.066 | -0.008 | 25 |

## Donor substitutions hiding inside high ECFP similarity

`d_*_rank` is the fraction of all other ligands that the distance places *closer* than this partner: 0.0 means the two are each other's nearest neighbour under that distance.

| kind | n_pairs | median_tanimoto | median_tanimoto_rank | median_mech_rank | median_softness_delta |
|---|---|---|---|---|---|
| -> P donor | 39 | 0.639 | 0.184 | 0.474 | 0.017 |
| O <-> N donor | 12 | 0.620 | 0.023 | 0.260 | 0.092 |
| O/N -> S donor | 6 | 0.611 | 0.023 | 0.178 | 0.234 |

