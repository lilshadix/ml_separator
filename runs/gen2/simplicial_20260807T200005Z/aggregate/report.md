# Simplicial multi-seed result

Validation: PASSED

Primary comparison: guarded SNN hybrid minus adaptive tabular Delta3D.

| model seed | split seed | Delta R2 | Delta balanced R2 | macro-MAE reduction |
|---:|---:|---:|---:|---:|
| 42 | 104729 | +0.001204 | -0.002428 | -0.000445 |
| 7 | 130363 | +0.001334 | +0.004959 | -0.000224 |
| 137 | 169087 | +0.003129 | -0.020167 | -0.003778 |
| 2027 | 214087 | +0.001035 | +0.004013 | -0.001066 |
| 9001 | 275015 | -0.000036 | -0.005584 | -0.001391 |

## Across-seed distribution

| metric | mean | std | min | max | positive runs |
|---|---:|---:|---:|---:|---:|
| r2_gain | +0.001333 | 0.001141 | -0.000036 | +0.003129 | 4/5 |
| group_balanced_r2_gain | -0.003841 | 0.010133 | -0.020167 | +0.004959 | 2/5 |
| mae_reduction | -0.000356 | 0.000249 | -0.000720 | -0.000077 | 0/5 |
| macro_group_mae_reduction | -0.001381 | 0.001419 | -0.003778 | -0.000224 | 0/5 |
| sign_accuracy_gain | -0.000000 | 0.002358 | -0.003700 | +0.001850 | 3/5 |

## Cross-seed OOF ensemble

Averaging is performed by pair_id only after every constituent prediction was generated with that pair's entire held-out group excluded from training.

- r2_gain: +0.001917; 95% CI [-0.006264, +0.010758]
- group_balanced_r2_gain: -0.002466; 95% CI [-0.014000, +0.009536]
- mae_reduction: -0.000386; 95% CI [-0.001793, +0.000241]
- macro_group_mae_reduction: -0.001418; 95% CI [-0.003225, +0.000104]
- sign_accuracy_gain: -0.001850; 95% CI [-0.010929, +0.005588]

Per-run fixed-OOF confidence bounds were not averaged. The ensemble interval was recomputed from the aligned cross-seed OOF predictions.
