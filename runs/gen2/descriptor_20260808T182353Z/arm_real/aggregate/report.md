# Delta3D multi-seed aggregation

Status: PASSED

Comparison: adaptive Delta3D+metal-site-descriptor blend minus adaptive Delta3D blend.

All runs have identical dataset, exact pair cohort, implementation, feature contract, and protocol fingerprints. Model and split seeds are intentionally different.

Descriptor permutation arm: none; descriptor columns: 11.

## Runs

| model seed | split seed | Delta R2 | Delta balanced R2 | macro-MAE reduction | R2 CI |
|---:|---:|---:|---:|---:|---:|
| 42 | 104729 | +0.006578 | +0.014152 | +0.000642 | [+0.001953, +0.018882] |
| 7 | 130363 | +0.007290 | +0.011990 | +0.001496 | [-0.003495, +0.031012] |
| 137 | 169087 | +0.004877 | +0.017370 | +0.001413 | [-0.007108, +0.023779] |
| 2027 | 214087 | +0.005077 | +0.016578 | +0.001426 | [+0.000451, +0.014466] |
| 9001 | 275015 | -0.000430 | +0.008174 | +0.002186 | [-0.010038, +0.006125] |

## Across seeded splits and model initializations

| improvement | mean | median | std | min | max | positive runs | CI > 0 | CI crosses 0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| r2_gain | +0.004678 | +0.005077 | +0.003030 | -0.000430 | +0.007290 | 4/5 | 2/5 | 3/5 |
| group_balanced_r2_gain | +0.013653 | +0.014152 | +0.003722 | +0.008174 | +0.017370 | 5/5 | 1/5 | 4/5 |
| mae_reduction | +0.000540 | +0.000407 | +0.000441 | +0.000212 | +0.001289 | 5/5 | 2/5 | 3/5 |
| macro_group_mae_reduction | +0.001433 | +0.001426 | +0.000547 | +0.000642 | +0.002186 | 5/5 | 0/5 | 5/5 |
| sign_accuracy_gain | +0.002220 | +0.002775 | +0.001241 | +0.000925 | +0.003700 | 5/5 | 0/5 | 5/5 |

## Interpretation guardrail

The table summarizes variability across prespecified model/split seeds. Each reported bootstrap interval is conditional on that run's fixed OOF predictions. The interval bounds are not averaged into a new confidence interval.

Validation fingerprint: `f4fac3049f1c93fcb0f02ad0ec0b336a303a52d7b7cf8f237ff206b6b8e42b4e`
