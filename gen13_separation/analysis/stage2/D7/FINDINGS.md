# D7 - what is left after the smooth radius trend

Base: **268 cells** with >= 6 observed metals (77 extractants, 36 chemotypes, 39 publications;
2664 cell-metal residuals). Per cell the centred logD curve is regressed on [1, z, z^2] with z =
the standardised Shannon CN8 radius; residual kept per metal (cubic adds z^3). Scripts
`d7_residual_structure.py`, `d7_followups.py`; figure `figures/stage2/D7_residual_structure.png`.
## 1. Almost nothing is left, and it is below the measurement noise
* Pooled R^2 of the per-cell quadratic **0.956** (median per-cell 0.961); cubic 0.974/0.984; linear 0.901.
* rms residual 0.130, **median |residual| 0.0563** log units (cubic 0.0378). Replicate SEM of a single
  cell-metal logD is 0.205 (median over 207 replicated cell-metals; raw replicate sd 0.337).
* Per-element median |residual| 0.030 (Tm, 136 cells) to 0.087 (Nd, 228 cells); cluster-bootstrap
  intervals (2000 draws over extractants) in `D7_table1_per_element_residual.csv`.
* Mean residual nonzero after **Holm-Bonferroni over the 14 elements** (cluster-bootstrap SE over
  77 extractants): **Ho +0.058 (p=0.010), Gd -0.072 (p=0.012), Lu -0.057 (p=0.033), Nd -0.063
  (p=0.040)**; all others p >= 0.05.

## 2. Shape: a single-element Gd dip, not a Gd step; no Eu anomaly
Share of the mean-residual-curve variance / of the pooled residual variance, each shape first
projected through the same per-cell quadratic (`D7_table2b_shape_catalogue.csv`):
Gd dip (single element) r=+0.572, **32.7% / 3.74%**; Nd dip r=-0.489, 24.0% / 2.91%;
Jorgensen tetrad_e1 with **reversed sign** r=-0.486, 23.6% / 2.76%; classic Gd **step** `gd_break`
r=+0.084, **0.7% / 0.12%**; tetrad_e3 r=+0.135, 1.8% / 0.30%; **Eu anomaly r=-0.001, 0.0% / 0.00%**.
Joint gd_break+tetrad_e1+tetrad_e3 = 3.6% of pooled residual variance. Eu is the *least* anomalous
element (mean residual +0.003 +/- 0.010, median |resid| 0.049).

## 3. It is reproducible - the one strong positive result
Split-half correlation of the mean residual curve over 14 elements (500 random half-splits):
**extractant 0.718 [0.358, 0.899]** (100% of splits positive); **publication 0.696 [0.332, 0.900]**;
**chemotype 0.513 [0.015, 0.788]**. Within-cell permutation nulls: median 0.013 / 0.007 / 0.057.
Split-half-debiased **reproducible rms amplitude 0.0349 log units [0.023, 0.038]**: only **7.2%** of
the pooled residual variance is a curve common to cells, 93% is cell-specific or noise.
Gd dip is the portable piece: negative in **16 of 18** publications with >= 4 cells; -0.127 (z=-4.8)
outside the diglycolamide supercluster sc009 and -0.025 (z=-2.0) inside it.

## 4. No detectable chemotype or donor-set dependence
Top-6 chemotypes: sc009 (152 cells), sc061 (17), sc071 (12), sc076 (11), sc027 (10), sc077 (10);
four have only 1-2 distinct extractants so their intervals are degenerate. Extractant-level
permutation test of between-chemotype heterogeneity: **p = 0.33** (1000 permutations); median
pairwise correlation between chemotype residual curves 0.243 (range -0.33 to +0.78). DENTATE and
coreCN curves in `D7_table4d_donorset_curves.csv` do not separate. The Nd dip is DGA-specific
(-0.109 in sc009 vs -0.007 elsewhere); the Gd dip is not.

## 5. The prize is 0.001-0.003 MAE, 7-20x below the 0.02 threshold
* Oracle per-cell smooth baseline on these 268 cells: extractant-macro pairwise MAE **0.1407**
  (77 extractants). Plus leave-one-extractant-out mean residual curve 0.1395 (**-0.0012**); at the
  best shrinkage lambda=0.60, 0.1379 (**-0.0028**); with the in-sample mean curve 0.1384 (-0.0023).
* On the real held-out arms (B_primary, 5 seeds x 5 folds) adding the chemotype-held-out mean
  residual curve at lambda=1 makes every arm **worse**: C_DIRECT_ROW 0.4953 -> 0.5009,
  X_ENS_DIRECT+LOWRANK_K2 0.4810 -> 0.4865, B1_MEAN_CURVE 0.6033 -> 0.6095. Grid-optimised lambda
  buys at most **0.0008** (B1_MEAN_CURVE), 0.0001 on the ensemble.
* Upper bound: an **in-sample-fitted** global 14-element offset on X_ENS_DIRECT+LOWRANK_K2, kept to
  its non-smooth part only, gives 0.4810 -> **0.4798 (-0.0011)**. The pairwise correction has
  rms 0.064 against a typical pair error of 0.48.
* What the arms miss is smooth, not rough: their mean held-out error curve has rms **0.206** in the
  quadratic-in-z part vs **0.032** in the non-smooth part - a monotone light-negative/heavy-positive
  amplitude shrinkage (La -0.32 ... Lu +0.28 for C_DIRECT_ROW).

**Answer: yes there is real, reproducible non-smooth structure (a Gd dip, split-half r 0.72), but its
amplitude is 0.035 log units and predicting it perfectly buys at most ~0.003 extractant-macro MAE.
Not worth modelling. The headroom is in the smooth curve amplitude, not in element anomalies.**
