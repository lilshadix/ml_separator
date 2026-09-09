# d2_metal_dependency - independent verification

Scripts: `v2_check.py` (C1-C4), `v2_check2.py` (D1-D4), `v2_check3.py` (E1-E2), `v2_check4.py` (F1-F2).
All written before reading the d2 scripts. Stdout in `v2_stdout*.txt`, tables in `v2_*.csv`.

## Verdict: AGREE. Both load-bearing numbers reproduce within 1%.

## Scoring convention (my implementation, before any d2 number)
extractant-macro MAE over all 521 cells / 5 seeds: C_DIRECT_ROW **0.4953** (ref 0.495),
X_ENS_DIRECT+LOWRANK_K2 0.4810 (0.481), M_SELECTED 0.5014 (0.501), B1_MEAN_CURVE 0.6033 (0.603),
B4_HEAVIER_ALWAYS 0.6742 (0.674). All five programme reference values reproduce, so the
scoring unit used below is the right one.

## Number 1 - PC1 variance fraction and the radius identification
| estimator | mine | summary |
|---|---|---|
| shrunk variogram, alpha=0.22 | **0.726** | 0.727 |
| unshrunk variogram (lambda1 / sum of positive eigenvalues) | **0.872** | 0.872 |
| unshrunk variogram (lambda1 / trace) | 0.910 | - |
| complete-14 **plain** sample covariance (n=78) | **0.898** (PC2 0.065, PC3 0.015, cum3 0.977) | 0.857 / cum3 0.939 |
| complete-14 shrunk at alpha=0.05 | 0.857, cum3 0.939 | (matches) |

alpha*=0.22 is confirmed as the smallest 0.01-grid value making all 13 subspace eigenvalues
positive. |r(PC1, radius_cn8)| = **0.9881** (cn9 0.9875, hydration 0.9728, inv_radius 0.9815);
radius+inv_radius fit R2 = **0.9905**. PC2: |r| 0.656 radius_sq, 0.037 gd_break, 0.106 tetrad_e3,
best 2-term R2 **0.580**. All eigenvector signs are globally flipped vs the summary (arbitrary).
Pair counts 87-242, median 153 over 91 pairs. Naive corr(La,Lu) = **-1.0353** (claimed -1.035);
54 of 182 off-diagonal entries lie outside [-1,1], so the estimator choice is justified.

**Robustness the summary did not run:** excluding the dominant chemotype sc009 (146 cells) gives
PC1 0.714 shrunk / 0.872 unshrunk, |r| with radius 0.9916; within sc009 alone 0.634 / 0.747,
|r| 0.9683. The PC1 = radius ramp result transfers off the dominant family.

## Number 2 - the one-scalar amplitude oracle (C_DIRECT_ROW, m>=8, seed 104729)
188 cells, 72 extractants, 12176 pairs - subset sizes match exactly.
baseline extractant-macro MAE **0.4426** (summary 0.441); one oracle PC1 scalar -> **0.1775**
(0.176), **59.9%** removed (60.1%); leave-the-scored-pair-out **0.1805** (0.179).
Centring: each cell's curve is centred over its own observed metals (max |row mean| 7.3e-16), and
the variogram estimator is level-free (max |V_raw - V_centred| = 4.4e-16), so the covariance
headline does not depend on the centring at all.
Dominance: per-cell-first macro averaging gives 0.4411 -> 0.1779, i.e. unchanged; the largest cell
holds 0.75% of pairs.
Not a generic 1-dof effect: five fixed random mean-zero directions remove only 1.1-5.0%, and the
pure physics `radius` shape (no data) removes 58.5%.
Not a global recalibration: one global scalar makes it **worse** (0.4982), per-extractant 0.1959,
per-cell 0.1775 (sd of the per-cell scalar 1.78).
Seed robustness (summary used one seed): over all 5 seeds baseline 0.424-0.470, oracle 0.178-0.186,
fraction removed **56.2-61.8%**, mean 59.3%.

## Number 3 - the non-oracle revealed-SF gain (genuinely out of sample)
Their code excludes the revealed pair from scoring and builds the correction covariance with the
target cell's chemotype removed. I reproduce the evaluation set exactly - 13505 pairs, 84
extractants, 321 cells (m>=4) - and the baselines to 4 dp: **0.4334** C_DIRECT_ROW (0.43336),
**0.4234** X_ENS (0.42339). My independent correction rule (per-cell PC1 amplitude from the one
revealed pair, shrinkage fitted leave-chemotype-out) reaches **0.2137 / 0.2131** vs their
0.2239 / 0.2255 - 4.6% apart, in their disfavour, so their claim is conservative.

## Bonus - the radius law on pairs
OLS mean|logSF| = **9.603**|dr_CN8| + **0.024**, R2 = **0.9610** over 91 pairs (9.60/0.024/0.961).
Fraction heavy-preferred mean **0.749**, range **0.469-0.883**. Exact.

## Discrepancies found
1. "complete-14 **sample covariance** 0.857" is a shrunk (alpha=0.05) estimate; the plain sample
   covariance of the 78 cells gives 0.898 and cum3 0.977, not 0.939. Their CSV labels the row
   `C_complete14_shrunk`; only the prose calls it plain. 4.6% off, and in the conservative direction.
2. "two scalars 71% (-> 0.129)" is estimator-dependent: PC1+PC2 from the **designated primary**
   (shrunk variogram) matrix gives 0.1539 (65.2% removed), not 0.129. Their 0.1287 reproduces with
   PC1+PC2 from the complete-14 covariance (0.1281) or radius+radius_sq (0.1307). Spread 0.128-0.154.
3. The headline says "60% of the **best arm's**" MAE, but 0.441 -> 0.176 is C_DIRECT_ROW
   (macro 0.495), not the best arm (X_ENS, 0.481). Their own table gives 59.2% for X_ENS, so the
   substance survives; the attribution is loose.
4. Not checked: the LOO metal-from-metal R2 numbers (0.558 / 0.929 / 0.928) and the arm
   residual-curve sd and lag-1 numbers.
