# D2 - the dependency between the lanthanides themselves
Scripts d2_analysis.py (1-4), d2_conditional_gain.py (5), d2_completion_and_amplitude.py (6-7),
d2_amplitude_dof_check.py (8). 521 cells x 14 metals. Figures figures/stage2/d2_*.png.
## 0. Estimator caveat (drives everything)
Row-centring INDUCES negative correlation: -1/(m-1) under isotropy, exactly -1 at m=2, and 138/521
cells have m=2, so the naive pairwise-complete centred correlation is unusable - it returns
La-Lu = -1.035, outside [-1,1], because diagonal and off-diagonal use different cells. Primary
estimator = variogram, free of any centring choice: V[i,j] = Var(logD_i - logD_j) over cells with
both observed, G = -0.5 J V J. Cross-check = the 78 complete-14 cells. All three go to CSV.
## 1. Covariance / correlation (d2_corr_*.csv, d2_n_pairwise.csv)
* Pairwise n: min 87 (Pr-Tm), median 153, max 242 over 91 pairs - no entry is under-sampled.
* Centred-curve sd is U-shaped along Z: 1.07 La, 0.59 Nd, 0.22 Eu, 0.23 Gd, 0.62 Lu (complete-14).
* Adjacent corr >0.92 EXCEPT the middle: Nd-Sm .66, Sm-Eu .28, Eu-Gd .40, Gd-Tb .73, Tb-Dy .76.
## 2. Eigen-structure (d2_eigen*.csv)
Shrinkage: linear, toward isotropic on the 13-dim mean-zero subspace, alpha = 0.22 (smallest
0.01-grid value making all 13 eigenvalues positive).
* PC1/PC2/PC3 = .727/.058/.040 (cum .825); unshrunk G .872/.051/.028; complete-14 .857/.065/.018.
* PC1 IS the radius ramp: r = -0.988 radius, -0.988 radius_cn9, -0.973 hydration, +0.982
  inv_radius; 2-term radius+inv_radius fit R2 = 0.991.
* PC2 is CURVATURE, not a Gd break, not a tetrad: r = +.656 radius_sq, +.235 tetrad_e1, -.106
  tetrad_e3, +.037 gd_break; U-shaped loadings (La +.45, Lu +.47, Tb -.40, Gd -.29). Best 2-term
  physics fit R2 only .580, so ~40% of PC2 is outside the basis. PC3 (4%) has none (best R2 .481).
## 3. Conditional prediction (d2_loo_r2_*.csv, d2_r2_decay_with_dZ.csv); LOO over cells, OLS, scored vs a LOO intercept-only model
* Predict centred j from centred i, 182 ordered pairs with n>=25: mean R2 0.558, MAE 0.236 vs
  baseline 0.447 (complete-14); 0.500/0.238 on the 268 cells with m>=6; 0.488/0.241 on all 521.
* R2 does NOT decay monotonically with |dZ|, it is U-shaped: .81 (dZ=1), .41 (3), .26 (4-5), .67
  (8), .86 (10), .66 (14). corr(R2,|dZ|) = .165 only, corr(R2,|pc1_i x pc1_j|) = .748 - shared PC1
  amplitude sets predictability, not distance. Every R2 <= 0 pair involves Eu (Eu-Ho -.019, Eu-Pr
  -.016, Eu-Nd -.014) or Sm-Gd (-.004); best Yb-Lu .963.
* Two flanking neighbours: mean R2 0.929, MAE 0.083 over the 14 metals. Weakest Gd 0.787, Eu 0.834.
* Level-free completion (hold one metal out of a complete cell, predict its offset from the other
  13): mean R2 .928, MAE .090, baseline .481; worst Gd .734, Eu .877. Every metal's unique sd after
  3 components (max .119, Tm) is far below the pooled within-replicate sd 0.302 (n=216).
## 4. Separation factors (d2_pair_separation_factor_stats.csv)
* 91 pairs, n 87-242 (median 153). mean|logSF| .065 (Tm-Yb) to 1.676 (La-Ho); sd of logSF
  .119-1.513 (mean .683). Heavy-preferred mean .749, range .469-.883; only Tm-Yb .47, Er-Tm .50,
  Yb-Lu .52 and Eu-Gd .54 (mean logSF +0.001) are near a coin flip.
* mean|logSF| ~ |Shannon CN8 radius diff|: slope 9.60/A, intercept 0.024, R2 = 0.961 over 91 pairs
  (extractant-macro slope 8.05, R2 = 0.947).
* Residuals: heavy partner Lu -.123, Yb -.065, Tm -.062 separate LESS than radius predicts; Tb
  +.117, Ho +.100, Sm +.083, Eu +.080 MORE. Extremes La-Ho +.260, Nd-Tb +.168, Eu/Tb-Lu -.193.
## 5. What the arms are not using (d2_amplitude_dof_check.csv, d2_conditional_gain_one_revealed_SF.csv)
* Arm pair residuals are exactly additive in a per-cell metal curve (r_AB = u_A - u_B): a cell's
  m(m-1)/2 held-out pairs carry only m-1 independent numbers. That curve is smooth: lag-1
  correlation along Z = +0.917 (C_DIRECT_ROW, 78 complete cells).
* 188 cells with m>=8 (72 extractants, 1 seed): C_DIRECT_ROW extractant-macro MAE 0.441; one oracle
  scalar (PC1 amplitude) -> 0.176 (60% of the error), two -> 0.129 (71%). Fitting the scalar WITHOUT
  the scored pair gives 0.179/0.133, so this is not a fitting artefact. (On m=3-5 cells the 2-scalar
  oracle saturates at 0.046 - ignore it.)
* Non-oracle: reveal the widest-dZ measured logSF per held-out cell and correct the rest with a
  leave-chemotype-out residual covariance -> extractant-macro MAE 0.433 -> 0.224 (C_DIRECT_ROW),
  0.423 -> 0.226 (X_ENS_DIRECT+LOWRANK_K2), over 13505 pairs in 84 extractants.
## Answer
One-and-a-bit-parameter family: 73% (shrunk) to 87% (raw) of centred variance is one radius ramp,
83-95% is three components, and after three components no metal exceeds replicate noise. No metal is
truly independent, but Eu and Gd are near-orthogonal to the ramp (PC1 communality 0.005, 0.290) and
carry so little variance (sd 0.22-0.23) that they alone cannot be pinned by a neighbour. Conditional
structure is large and unused: arm error is 60% one per-cell amplitude scalar, 71% two. The arms get
curve SHAPE right and AMPLITUDE wrong - amplitude, not more metal-axis basis functions, is where the
remaining ~0.26 MAE lives.
