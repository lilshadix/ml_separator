# Leakage-safe lanthanide ablation aggregate

Status: **PASSED**

Pair scope: `all`; model seeds: 7, 42, 137, 2027, 9001; one frozen split seed: `104729`.

All run artifacts were hash-verified, feature registries and fold assignments were identical, and OOF rows aligned exactly by pair ID, truth, extractant and fold.

## Prespecified A2 versus A5 result

1. **Does local 3D improve unseen-extractant prediction?** No; A5 is worse than A2 under the conditional equal-extractant bootstrap.
2. **Absolute MAE improvement:** +0.001715 log units.
3. **Relative MAE improvement:** 0.4% versus A2.
4. **Held-out extractants improved:** 14/34 (41.2%).
5. **Equal-extractant macro delta MAE and bootstrap 95% CI:** -0.014796 [-0.025682, -0.004758].

Positive error deltas mean the candidate reduced error; positive correlation/R2 deltas mean the candidate increased agreement.

| comparison | delta MAE | delta RMSE | delta R2 | delta Pearson | delta Spearman |
|---|---:|---:|---:|---:|---:|
| A2_vs_A5 | +0.001715 | +0.007947 | +0.017718 | +0.007276 | +0.000770 |
| A2_vs_A6 | +0.003198 | +0.010171 | +0.022636 | +0.009978 | +0.008447 |
| A3_vs_A2 | +0.072993 | +0.139145 | +0.347678 | +0.195314 | +0.294290 |

## Descriptor and negative-control checks

Descriptor-block ranking by cross-seed OOF delta MAE: A2+D1 (+0.003387), A2+D4 (-0.001127), A2+D2 (-0.001281), A2+D5 (-0.001519), A2+D3 (-0.002482).

A5 shuffled-geometry controls: mean delta MAE -0.000181 across 3 shuffle variants, versus +0.001715 for correctly matched A5 geometry. No categorical 'approximately equal' claim is made without a predeclared equivalence margin.

A high-versus-questionable geometry comparison is not estimable from the available QC classes; class-specific rows remain in aggregate_summary.json.

## Where A5 helps most

Top extractants by A2-minus-A5 MAE:

- `COCCN(CCOC)C(=O)COCC(=O)N(CCOC)CCOC`: n=78, delta MAE=+0.057600
- `CCCCCCCCCCN(CCCCCCCCCC)C(=O)COCC(=O)N(CCCCCC)CCCCCC`: n=298, delta MAE=+0.030476
- `CCCCCCCCN(CCCCCCCC)C(=O)[C@H](C)O[C@@H](C)C(=O)N(CCCCCCCC)CCCCCCCC`: n=91, delta MAE=+0.021305
- `CCCCCCC(CCCC)CCCN(C)C(=O)COCC(=O)N(C)CCCC(CCCC)CCCCCC`: n=91, delta MAE=+0.019538
- `CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC`: n=2921, delta MAE=+0.013527

Top lanthanides by overlapping all-pairs attribution:

- `La`: n=923, delta MAE=+0.013655
- `Ce`: n=913, delta MAE=+0.010267
- `Gd`: n=1144, delta MAE=+0.009299
- `Nd`: n=1095, delta MAE=+0.008119
- `Sm`: n=1120, delta MAE=+0.006194

## Consistency and interpretation

Across model seeds, delta MAE mean=+0.001735, sample SD=+0.003119, range=[-0.002243, +0.004568]. Across correlated seed-fold cells, mean=-0.003487 and sample SD=+0.013599.

The conclusion must be read across MAE, RMSE, R2, Pearson and Spearman in the comparison table, not from R2 alone. Fold summaries are descriptive because the same held-out folds recur across model seeds.

Fold rows sharing a split are correlated and are descriptive. Confidence intervals use paired resampling of held-out extractants from aligned fixed OOF predictions; the primary macro-MAE interval gives each sampled extractant one vote and does not include model-refitting or split uncertainty.

Validation fingerprint: `540c61b3b38c7033243562fe31092bff2a40e2500ecc58a0095fc7040d99f4f8`
