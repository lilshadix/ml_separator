# E2: Error anatomy of the round-4 reference B1

**Status.** This is exploratory development evidence for designing round 5. It does not decide anything.
- **Data:** the existing out-of-fold prediction parquets in `round4_20261022/predictions/` (REF, S11, S05, S06, S02, S01, R3-R02, R3-R14 and R3-B0, both BPG and BP) and the V1 bench (`gen20k.data.load_bench()`, which checks the manifest).
- **Models:** none trained. The only fitted quantities are oracles, and leave-one-group-out means and blends computed on the OOF residuals.
- **Sealed material:** nothing read from the round-3 final seeds, and nothing written outside this file.
- **Scripts:** the analysis scripts are in the session scratchpad and are not part of the repository: `C:\Users\Bandai\AppData\Local\Temp\claude\C--Users-Bandai--claude-spares-spare-9MsqWO\0cd609d3-ed34-4de3-83c3-904e5f4bdb8b\scratchpad\e2\s1_anatomy.py` … `s8_pubeffect_robust.py`. Every method is described below in enough detail to rebuild it.

**Population.** V1, C3, k = 3, DOPT_Q, with 4 read metals per held-out cell.
- 229 cells and 1,802 strict-adjacent pairs per seed, over 5 dev seeds.
- **BPG decides; BP is the robustness check.** Every BPG number below was recomputed on BP. BP agrees within ±0.003 R² unless a BP value is stated.
- **How R² is pooled.** All 5 seeds contain the same 1,802 rows with the same y, so a row-pooled R² equals the seed-mean R² exactly.
- **Error share.** "SSE share" is a group's share of B1's summed squared error. The **ΔR² if zeroed** is the R² gain if that group were predicted perfectly, i.e. SSE_group / SST.
- **Reference values.**

  | | BPG | BP |
  |---|---|---|
  | B1 R² | 0.4521 | 0.4519 |
  | B1 MSE | 0.02699 | 0.02700 |
  | B1 RMSE | 0.164 | 0.164 |

  var(y) = 0.04927.
- **Reproduction check.** My re-implementation of B1 reproduces the stored column to 8.9e-16:
  - PWL_R along −r(CN8), held constant beyond the extreme reads;
  - plus the fold's Gd potential β from `results/REF/fitted_*.json`.

  The k = 4 oracles below are built on this re-implementation.

---

## 0. Bottom line

1. **The error is heavy-tailed, and the tail is mostly data problems.** The tail sits in measurements near the edge of what can be measured, in one suspect source, and in single-metal spikes. Chemistry the model could learn is not the main part.
   - **Row concentration.** The top 1 % of rows carry 31 % of B1's SSE, and the top 5 % carry 59 %. Residual kurtosis is 18.2. The robust residual sd is 0.086, against an RMSE of 0.164.
   - **Cell concentration.** The worst 10 % of cells (23) carry 53.5 % of SSE (cell-SSE Gini 0.69).
   - **One cell alone carries 11.8 %.** It is a C5-BTBP cell, with log D between −3.2 and −5.0 zig-zagging by up to 1.8 log units.
2. **Most of the error is in the light pairs.** By series position:

   | region | rows | B1 SSE | ΔR² if zeroed |
   |---|---|---|---|
   | La/Ce–Pr/Nd | 21.6 % | **42.8 %** | 0.235 |
   | Sm/Eu–Gd/Tb (Gd region) | 26.7 % | 27.2 % | |
   | Tb/Dy–Ho/Er | 28.7 % | 16.3 % | |
   | Er/Tm–Yb/Lu | 23.0 % | 13.7 % | |

   - La/Ce plus Ce/Pr alone carry **34.2 %**.
   - Ce is hidden in 97 % of those rows; DOPT_Q almost never reads it.
   - Ce is the hidden metal with the largest level-error share (16 %).
3. **One publication carries 23.1 % of the error from 8.2 % of rows.**
   - It is the ORNL-curated C5-BTBP source pub_6a0ef8528c: 15 cells, a suspected detection floor, and doubtful attribution (PROVENANCE.md §C5-BTBP).
   - Dropping it lifts B1 to **0.486** (PWL_R 0.459).
4. **The measurement window explains half of the SSE.**
   - Pairs where either metal has log D outside [−2, 2] (D outside 0.01–100) are 26 % of rows and **50 % of SSE**.
   - A **label-free** rule that keeps a row only when B1's predicted levels of both metals lie in [−2, 2] keeps **74.7 % of rows**. On those rows B1 scores **R² 0.540** (PWL_R 0.509); on the rest it scores 0.302.
5. **Noise is a large part, but its share is only bounded, not measured exactly.**
   - Robust estimates put the per-metal noise sd at 0.03–0.06, and the implied noise floor at **4–16 % of B1's MSE**. That is 16–60 % of the error of a *typical* row.
   - Mean-square estimates, which include the heavy tails, put it at **56–87 % of the MSE**.
   - The most direct evidence is 15 replicated C3 cells. There, the replicate noise of the target itself is **54 % of B1's MSE** on the same rows (median cell 48 %).
   - Per cell, error tracks the cell's own roughness: Spearman 0.88 between B1's cell MSE and the curve's second-difference roughness. It also tracks the cell's spread of y (0.91).
6. **The systematic remainder is reproducible within a publication but does not transfer across publications.**
   - Shifting B1 by the mean residual of the *other cells of the same publication* (per pair type, leave-one-cell-out, shrinkage 10) gives **R² 0.523 (+0.071)**. The same with read set gives **0.533 (+0.081)**.
   - This holds without the C5-BTBP source (+0.055), on in-window rows (+0.099), inside the largest cluster (+0.075) and inside sc009 (+0.100).
   - The same correction from *same-chemotype cells of other publications* gives **+0.005** (from −0.03 to +0.04 by subset).
   - Sibling cells (same extractant and publication, matched seed / pair / read set) have a residual correlation of **0.33**. Cells of the same extractant from other publications have **0.10–0.14**.
   - **Consequence:** under BP / BPG, where the publication is held out, this component cannot be learned.
7. **What the oracles say:**

   | oracle | R² |
   |---|---|
   | (a) one more *true* read, a random metal | 0.531 (+0.079) |
   | (a) one more true read, max-gap rule | 0.519 |
   | (a) one more true read, DOPT_Q's 5th metal | +0.045 on the same rows |
   | (a) one more true read, always Ce | 0.600 |
   | (a) one more true read, oracle-best metal | **0.766** |
   | (b) chemotype × pair mean residual, in sample | 0.615 |
   | (b) the same, leave-one-cell-out | 0.499 |
   | (b) the same, leave-one-publication-out | **0.449** |
   | (c) per-pair convex blend of 14 arms, in sample | 0.478 |
   | (c) the same, leave-one-cluster-out | **0.423** |
   | (d) worst 10 % of cells fixed | 0.745 |

   For (a), almost all of the gain comes from turning the worst pairs into *measured* pairs. Like for like (rows that touch the added metal removed from both arms):

   | extra read | gain like for like |
   |---|---|
   | max-gap | +0.049 |
   | DOPT_Q | +0.032 |
   | Ce | −0.012 |
   | oracle-best | +0.006 |
8. **What a model could plausibly still reduce under the current protocol is small:**
   - shrunk slope continuation beyond the read hull: about +0.005, leave-one-cluster-out;
   - nothing from per-pair offsets or scales, blends, curvature-aware interpolation, or chemotype shape. All of these are ≤ B1 under leave-one-cluster-out.

   This agrees with rounds 1–4. The large remaining components are measurement-reducible (which metal to read), data- or evaluation-reducible (window, floor flags, abstention), or need same-laboratory information.

---

## 1. Where B1's error lives

### 1.1 By adjacent pair (BPG; BP within 0.01)

Columns: rows per seed; share of B1 SSE; ΔR² if zeroed; B1 MSE; sd of y; within-pair R² of PWL_R / B1 / B2 / S11 / HBK@ML; B1 bias (p − y).

| pair | rows | SSE % | ΔR²₀ | MSE | sd y | PWL_R | **B1** | B2 | S11 | HBK | bias |
|---|---|---|---|---|---|---|---|---|---|---|---|
| La/Ce | 137 | 16.2 | 0.089 | 0.0575 | 0.299 | 0.358 | 0.353 | 0.347 | 0.353 | 0.345 | +0.024 |
| Ce/Pr | 125 | **18.0** | 0.099 | 0.0700 | 0.348 | 0.419 | 0.419 | 0.412 | 0.419 | 0.425 | −0.054 |
| Pr/Nd | 127 | 8.6 | 0.047 | 0.0329 | 0.238 | 0.422 | 0.420 | 0.408 | 0.419 | 0.371 | −0.038 |
| Sm/Eu | 158 | 10.8 | 0.059 | 0.0331 | 0.240 | 0.424 | 0.423 | 0.390 | 0.421 | 0.434 | −0.005 |
| Eu/Gd | 158 | 9.6 | 0.053 | 0.0297 | 0.227 | 0.276 | 0.422 | 0.338 | 0.420 | 0.286 | −0.007 |
| Gd/Tb | 165 | 6.8 | 0.038 | 0.0202 | 0.211 | 0.321 | 0.546 | 0.442 | 0.546 | 0.354 | +0.031 |
| Tb/Dy | 168 | 5.8 | 0.032 | 0.0168 | 0.163 | 0.321 | 0.367 | 0.349 | 0.367 | 0.340 | −0.005 |
| Dy/Ho | 176 | 5.9 | 0.032 | 0.0163 | 0.167 | 0.419 | 0.414 | 0.470 | 0.412 | 0.481 | +0.032 |
| Ho/Er | 173 | 4.6 | 0.025 | 0.0129 | 0.132 | 0.240 | 0.256 | 0.261 | 0.256 | 0.290 | −0.002 |
| Er/Tm | 132 | 5.1 | 0.028 | 0.0189 | 0.161 | 0.262 | 0.265 | 0.212 | 0.265 | 0.244 | −0.005 |
| Tm/Yb | 133 | 4.0 | 0.022 | 0.0146 | 0.123 | 0.013 | 0.030 | 0.012 | 0.026 | −0.016 | −0.026 |
| Yb/Lu | 150 | 4.6 | 0.025 | 0.0149 | 0.122 | −0.005 | 0.002 | 0.022 | 0.001 | −0.144 | +0.002 |

**Measured:**
- La/Ce + Ce/Pr carry 34.2 % of SSE, Eu/Gd + Gd/Tb 16.5 %, Tm/Yb + Yb/Lu 8.6 %.
- The heavy pairs have R² ≈ 0 but small variance. Even fixing them perfectly would be worth only +0.047.
- S11 equals B1 in every pair: |Δ| ≤ 0.004 R² on BPG, and at most 0.016 on BP (Tm/Yb). The row correlation between the two predictions is 0.99983.
- B2 and HBK@ML beat B1 *in sample* on Dy/Ho, Ho/Er and the extrapolated rows. They lose on Eu/Gd and Gd/Tb. That pattern is what an in-sample blend exploits (§4c), and it does not survive leave-one-cluster-out.

**Region totals** (BPG / BP SSE share):

| region | BPG | BP |
|---|---|---|
| light | 42.8 % | 42.7 % |
| Gd | 27.2 % | 27.1 % |
| mid-heavy | 16.3 % | 16.6 % |
| far-heavy | 13.7 % | 13.5 % |

Region R² for B1 is 0.406 / 0.540 / 0.376 / 0.137.

### 1.2 By read-set geometry

| slice | rows | SSE % | B1 MSE | B1 R² (within) |
|---|---|---|---|---|
| one end read | 67.0 % | 64.7 | 0.0261 | 0.470 |
| both ends hidden | 33.0 % | 35.3 | 0.0289 | 0.416 |
| interior (between the extreme reads) | 88.1 % | 90.3 | 0.0277 | 0.471 |
| extrapolated beyond the **heaviest** read | 11.9 % (BP 12.3 %) | 9.7 (BP 10.1) | 0.0220 | −0.012 (all arms ≈ 0; B2 +0.061) |
| distance of the farthest hidden end to the nearest read = 1 Z | 69.7 % | 72.7 | 0.0281 | 0.458 |
| … = 2 | 17.5 % | 16.9 | 0.0262 | 0.412 |
| … = 3 | 8.2 % | 6.9 | 0.0227 | 0.418 |
| … ≥ 4 | 4.6 % | 3.5 | 0.020 | 0.47 |
| bracketing span 2 Z (an isolated hidden metal) | 32.5 % | 27.3 | 0.0227 | 0.539 |
| bracketing span 3 Z (mostly La…Nd with Ce and Pr hidden) | 9.8 % | **17.7** | **0.0489** | 0.443 |
| La read | 81.5 % | 93.7 | 0.0310 | 0.434 |
| La not read (heavy-only cells) | 18.5 % | 6.3 | 0.0092 | 0.617 |
| Gd read | 51.8 % | 59.7 | 0.0311 | 0.413 |
| Lu read | 12.2 % | 19.9 | 0.0442 | 0.345 |

**Measured:**
- The cell's lightest metal is read in 100 % of cell-seeds, so no row lies below the read hull. The heaviest metal is read in only 36 %.
- **Error does not grow with distance to the nearest read** (MSE 0.028 → 0.020 from distance 1 to 4). The geometry that matters is *which* metal is hidden, not how far it is from a read.

**Read sets:**
- La,Nd,Gd,Lu: 5.3 % of rows, 14.3 % of SSE.
- La,Pr,Gd,Dy: 3.9 % of rows, 12.1 % of SSE.
- Both are essentially the C5-BTBP cells' read sets.
- Gd,Dy,Er,Yb (heavy-only cells): 8.4 % of rows, 1.3 % of SSE.

PWL_R predicts exactly zero on the extrapolated rows, and so does B1 unless the Gd term applies.

### 1.3 By chemotype, publication and source cluster

**Chemotype:**

| chemotype | rows | SSE | MSE | R² |
|---|---|---|---|---|
| sc009 (DGA) | 48.9 % | 41.0 % | 0.0226 | 0.446 |
| **sc061 (BTBP)** | 9.4 % | **24.5 %** | 0.0702 | 0.305 |
| sc029 | 2.3 % | 5.9 % | | |
| sc075 (one diphenylpropanediamide cell with a single Tm spike) | 0.7 % | 4.5 % | 0.184 | |

**Publication:**

| publication | rows | SSE | MSE |
|---|---|---|---|
| **pub_6a0ef8528c** (C5-BTBP, Nilsson 2006 attribution doubtful, suspected floor) | 8.2 % | **23.1 %** | 0.076 |
| pub_126bbe7976 | 9.3 % | 14.1 % | |

**Source cluster (cluster_cpg):**

| cluster | rows | SSE | MSE |
|---|---|---|---|
| largest, g:pub_74428aeab2 | 64.9 % | 51.8 % | 0.0215 |
| g:pub_af7a1cd919 (= sc061) | 9.4 % | 24.5 % | 0.070 |

**Leave-one-cluster-out B1 R²** ranges from 0.443 (without g:pub_3e0f6bb1b5) to 0.475 (without g:pub_af7a1cd919). Without the largest cluster it is 0.444.

**Donor class:**
- non-DGA donors: 37.5 % of rows, 55.2 % of SSE (MSE 0.0397);
- DGA donors: 62.5 % of rows, 44.8 % of SSE (MSE 0.0193).

**Metal count:** cells with 13 metals (the C5-BTBP cells lack Tm) are 19.6 % of rows and 34.2 % of SSE.

### 1.4 By cell and by row

**Concentration:**

| worst cells | count | SSE share | B1 R² if fixed |
|---|---|---|---|
| top 5 % | 11 | 38.3 % | 0.662 |
| top 10 % | 23 | 53.5 % | **0.745** |
| top 20 % | 46 | 69.9 % | 0.835 |
| top 50 % | | 93.8 % | |

- Ranking by per-pair MSE instead of SSE gives 0.734. Choosing the worst cells separately in each seed gives 0.747.
- Spearman(cell MSE, cell sd of y) = **0.91**. Cells with a wide spread of separation factors carry the error, which matches gen13's "amplitude" finding.

**Top cells (BPG; BP identical):**

| cell | extractant | chemotype / publication | metals | SSE share | what the curve shows |
|---|---|---|---|---|---|
| c950dc2982e96b83 | C5BTBP | sc061 / pub_6a0ef8528c | 13 | **11.8 %** | log D −3.2…−5.0, zig-zag La −4.49, Ce −3.17, Pr −4.95: a floor |
| 543ce167788b1f3e | DODDdDGA | sc009 / pub_126bbe7976 | 13 | 5.2 % | heavy end > 2: Tb 2.26, Dy 2.10, Ho 3.09, Er 2.58, Lu 3.52, a zig-zag near the ceiling |
| 6ff6056da7469332 | N,N′-dimethyl-N,N′-diphenylpropanediamide | sc075 / pub_74428aeab2 | 14 | 4.5 % | flat −2.0…−2.3 with one Tm spike at −1.19 |
| 4ab68efc604014ca | tetraoctyl-phen-dicarboxamide | sc029 / pub_a49146c8fd | 9 | 2.7 % | values −3.0 / −2.7 repeated: censored-looking |
| 326289072f176f80, 9586db05ac5e234b, ca1d836f90fd7326, 894c0f669179ac88 | C5BTBP | sc061 / pub_6a0ef8528c | 13 | 2.5, 2.1, 1.2, 1.1 % | light end at ≈ −4.2 to −4.5 |
| 7054e94bc4c6833f | dimethyl-diphenyl-pyridinedicarboxamide | sc077 | 14 | 2.2 % | Pr 0.99 → Nd 0.04, a 0.95-unit step: stitched series? |

The worst 23 cells come from:
- chemotypes sc009 (7), sc061 (6), sc029 (3), sc036 (2) and five others;
- 5 cells from pub_6a0ef8528c;
- 14 metals (10), 13 (6), 9 (3), 7 (3), 8 (1).

**Rows:**
- the top 1 % of rows carry 31.1 % of SSE, the top 5 % 59.4 % and the top 10 % 73.4 %;
- kurtosis 18.2;
- median |residual| 0.058.

### 1.5 By hidden metal (B1 level errors δ = true − predicted level)

Each pair's residual is δ(A) − δ(B), with δ = 0 at read metals. Weighting δ² by the number of scored pairs each hidden metal enters gives this share of the level error:

| metal | Ce | Pr | Nd | Sm | Eu | Gd | Tb | Dy | Ho | Er | Tm | Yb | Lu |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| times hidden per seed | 137 | 65 | 79 | 128 | 121 | 74 | 168 | 126 | 113 | 103 | 81 | 84 | 125 |
| rms δ | 0.240 | 0.170 | 0.202 | 0.214 | 0.213 | 0.226 | 0.175 | 0.166 | 0.202 | 0.192 | 0.158 | 0.198 | 0.215 |
| mean δ | +0.024 | −0.009 | −0.066 | +0.008 | +0.002 | −0.009 | +0.032 | +0.038 | **+0.088** | **+0.086** | +0.035 | +0.037 | +0.028 |
| share | **16.2 %** | 4.0 | 3.6 | 6.4 | 12.1 | 7.8 | 11.2 | 7.2 | 9.3 | 6.1 | 4.3 | 5.4 | 6.4 |

**Interpretation, with the measurements behind it:**
- **Ce is the largest single error source.** It is hidden in 96 % of cell-seeds. Its second difference along the series is the largest of any metal (rms d2/√6 = 0.203). Ce/Pr is also the least reproducible pair between sibling cells: √(mean d²/2) = 0.42 within the same extractant and publication, and 0.40 across publications. For comparison, Tm/Yb is 0.05.
- **B1's residuals are not "white noise per metal".** The lag-1 correlation of pair residuals is −0.455 overall. That is mostly mechanical:
  - it is exactly −1 when the shared metal is hidden and both outer metals are read (1,463 row pairs);
  - it is −0.226 when all three metals are hidden.
- **Consecutive hidden level errors correlate at +0.69.** The error is a mix of smooth level error (along runs of hidden metals) and per-metal spikes.

### 1.6 By measurement window

**Label-dependent** (selected on the true levels, so these are not deployable rules):

| rows where … | rows | SSE share | MSE inside / outside | B1 R² on the rest (PWL_R) |
|---|---|---|---|---|
| min log D of the pair < −2 | 12.4 % | **34.7 %** | 0.076 / 0.020 | 0.507 (0.478) |
| min log D < −3 | 5.8 % | 22.3 % | 0.104 / 0.022 | 0.484 |
| max log D > 2 | 13.7 % | 15.0 % | 0.030 / 0.027 | 0.469 |
| either outside [−2, 2] | 26.1 % | **49.7 %** | 0.052 / 0.018 | **0.542** (0.515) |
| the cell has any value outside [−2.5, 2.5] | 29.3 % | 52.8 % | 0.049 / 0.018 | 0.500 |
| the cell has ≥ 3 identical log D values (censoring) | 11.3 % | 10.5 % | 0.025 / 0.027 | 0.456 |

**Label-free** (the B1 read values or predicted levels only, so these are deployable as an applicability domain):

| keep rows where … | kept | B1 R² inside (PWL_R) | B1 R² outside | MSE in / out |
|---|---|---|---|---|
| B1's predicted levels of A and B within [−2, 2] | **74.7 %** | **0.540** (0.509) | 0.302 | 0.0189 / 0.0510 |
| predicted levels within [−2.5, 2.5] | 84.6 % | 0.500 (0.472) | 0.302 | 0.0215 / 0.0571 |
| all four reads within [−2, 2] | 62.0 % | 0.500 (0.477) | 0.389 | 0.0184 / 0.0410 |
| min read ≥ −2 | 83.5 % | 0.500 (0.475) | 0.339 | 0.0203 / 0.0607 |

BP: 74.8 % kept and 0.538 inside.

---

## 2. Noise floor: how much of B1's error can measurement noise explain?

**Model.** Each log D carries independent per-metal noise of sd σ. An ideal predictor that knows the true curve then has error variance σ² on a row with one end read (the read end's noise is shared by y and the prediction) and 2σ² on a row with both ends hidden. In C3, 67 % / 33 % of rows are one-end / both-hidden, so the **floor MSE is ≈ 1.33 σ²**. B1's MSE is 0.0270. Its robust error variance, (1.4826·MAD)², is 0.0074.

**What the corpus offers.**
- Only 31 of 229 C3 cells have more than one experiment series.
- 161 of 2,425 (cell, metal) means rest on more than one row.
- `repsd__` exists for 19 C3 cells (median 0.297). That spread is mostly a whole-series level shift, which cancels in log SF (gen15 `labelerr/s1b`: series-level sd 0.28).
- I verified that a cell's log D equals the mean of its bundle rows (max |diff| 0).

| handle (C3 unless stated) | per-metal σ (or SF sd) | floor MSE ≈ 1.33σ² | % of B1 MSE | implied max R² |
|---|---|---|---|---|
| **robust: third differences, no Gd in window** (MAD) | σ 0.030 | 0.0012 | 4.4 % | 0.976 |
| robust: second differences, no Eu/Gd/Tb centre (MAD; median per cell 0.041) | σ 0.041 | 0.0022 | 8.3 % | 0.954 |
| robust: sibling cells, same extractant and publication (MAD of SF differences) | SF sd 0.065 (σ ≈ 0.046) | 0.0028 | 10.3 % | 0.944 |
| robust: replicate series, median per-cell between-series SF sd (15 cells, true series) | SF sd 0.082 (σ ≈ 0.058) | 0.0044 | 16.4 % | 0.910 |
| mean-square: third differences, no Gd | σ 0.107 | 0.0151 | 56 % | 0.693 |
| mean-square: second differences, no Eu/Gd/Tb centre | σ 0.115 | 0.0176 | 65 % | 0.642 |
| mean-square: siblings, var(d)/2 (includes condition changes) | SF sd 0.158 | 0.0165 | 61 % | 0.664 |
| mean-square: replicate series, dof-pooled single-series SF sd (15 cells; two cells dominate) | SF sd 0.188 | 0.0235 | 87 % | 0.522 |
| **direct, same rows:** 15 replicated C3 cells, noise of the replicate-mean target SF vs B1 MSE | 0.0098 vs 0.0182 | | **54 %** (median cell 48 %) | |
| per-cell roughness (each cell's own d2, an upper bound that includes non-smooth chemistry), 163 cells | | | 81 % of SSE (median cell 38 %) | |

**Measured facts:**
- The robust and mean-square handles differ by roughly a factor of 10 in variance, because the noise is heavy-tailed across cells.
- Within cells, B1's error tracks the cell's own roughness (Spearman 0.88) and spread (0.91).
- On replicated cells, the replicate noise of the scored target alone is about half of B1's error.
  - The noisiest replicated cell is c41d53a5c2625ad2 (TODGA): noise 0.080 against B1's MSE of 0.150.
  - In 4 of the 15 cells, noise ≥ B1's MSE: 48baeebc441c0964, 5fe61fb5cb043ac8, 75373a48cee6831b and 61603b00599eb0d5.
  - In others it is ≤ 10 % (for example 9f969f90406c97a4: 0.00015 vs 0.0089, and 62607940a098ba04: 0.0003 vs 0.0094).

**Interpretation (bounded, not exact):**
- Measurement-level noise explains between about 8 % of B1's MSE (if every cell were as clean as the median one) and about 60–85 % (if all tail roughness is noise).
- A defensible central reading has two parts:
  - **Bulk rows:** B1's typical-row error variance (≈ 0.0074) is only about 2–6 × the robust noise floor. In the bulk, B1 is within a small factor of noise.
  - **Tail:** the out-of-window and spiky cells hold about half the SSE. There the cells' own roughness and replicate spread are as large as B1's error. That half is predominantly noise or curation artefacts, not learnable chemistry.
- **The noise share cannot be pinned down** better than [0.08, 0.87] of the MSE on this corpus, because only 15 C3 cells have true replicate series.
- The local-neighbour oracle (§3) also points this way: knowing the two true neighbouring levels of a pair does not beat B1 in mean square.

---

## 3. Oracle bounds

### (a) B1 plus the true value of one more metal (5 measured metals)

β is kept at its k = 3 fold value. "Same rows" scores all 1,802 rows; rows that become fully read score zero error. "Like for like" drops the rows touching the added metal from both arms, as gen15 and round 3 require for budget ladders.

| choice of the extra metal | BPG R² (same rows) | Δ same rows | Δ like for like (rows dropped) | BP Δ same / like for like |
|---|---|---|---|---|
| none (B1, k = 3) | 0.4521 | — | — | — |
| worst metal (oracle) | 0.4571 | +0.005 | | +0.006 |
| always Gd | 0.4796 | +0.028 | | +0.029 |
| DOPT_Q's 5th metal (round-3 ladder, R3-B0 k = 4; 84 % of rows, B1 elsewhere) | 0.4974 | +0.045 | +0.032 (11 %) | +0.048 / +0.033 |
| heaviest unread / hull (Lu if unread) | 0.5010 | +0.049 | | +0.053 |
| max-gap in radius (label-free) | 0.5191 | +0.067 | **+0.049** (15.5 %) | +0.068 / +0.049 |
| random metal (expectation over candidates) | **0.5310** | +0.079 | small: only 2.4 % of B1's SSE is removed on untouched rows (≈ +0.013 R² in all-row units); 18.2 % of SSE sits on rows that become (partly) measured | +0.079 |
| always Ce (hidden in 96 %) | 0.6000 | **+0.148** | **−0.012** (14.2 %) | +0.147 / −0.012 |
| oracle-best metal per cell | **0.7658** | +0.314 | **+0.006** (21.9 %) | +0.313 / +0.004 |

On its 84 % subset, DOPT_Q's 5th metal takes B1 from 0.429 to 0.481. PWL_R on the same subset goes from 0.395 at k = 3 to 0.448 at k = 4.

**Reading.**
- **One extra read pays mainly by converting the worst pairs into measurements.**
  - Measuring Ce removes the La/Ce and Ce/Pr error (34 % of SSE) but does not help any other pair.
  - The oracle-best metal is worth +0.31, almost entirely on the rows it measures.
- **Placement matters more than the policy currently exploits.**
  - Like for like, max-gap (+0.049) beats DOPT_Q's own next pick (+0.032).
  - A random extra metal beats DOPT_Q's pick on the same rows (+0.079 vs +0.045).

### (b) Per-chemotype (and per-publication) shape correction: mean B1 residual per group × pair type

| correction | BPG R² | BP R² |
|---|---|---|
| chemotype × pair, **in sample** (oracle) | 0.6154 | 0.6146 |
| chemotype × pair × geometry, in sample | 0.6250 | 0.6234 |
| chemotype × pair × read set, in sample | 0.7316 | 0.7261 |
| chemotype × pair, leave-one-cell-out (shrinkage 0 / 5 / 20 rows) | 0.462 / 0.492 / **0.499** | 0.461 / 0.491 / 0.499 |
| chemotype × pair, **other publications only** (leave-one-publication-out; shrinkage 0 / 10) | **0.449 / 0.458** | 0.446 / 0.456 |
| **publication × pair**, leave-one-cell-out, shrinkage 10 | **0.523** | 0.522 |
| publication × pair × read set, leave-one-cell-out, shrinkage 10 | **0.533** | 0.531 |
| extractant × publication × pair, leave-one-cell-out, shrinkage 10 | 0.521 | 0.520 |
| extractant × pair (any publication), leave-one-cell-out, shrinkage 10 | 0.499 | 0.498 |
| pair only, in sample / leave-one-cluster-out | 0.464 / 0.414 | 0.463 / 0.412 |
| pair × read set, in sample / leave-one-cluster-out | 0.601 / 0.313 | 0.598 / 0.294 |

**Robustness of the within-publication gain** (publication × pair, leave-one-cell-out, shrinkage 10; ΔR² over B1, BPG):

| subset | Δ within publication | Δ same chemotype, other publications |
|---|---|---|
| all rows | +0.071 | +0.005 |
| without pub_6a0ef8528c | +0.055 | −0.017 |
| label-free window | +0.099 | −0.016 |
| largest cluster only | +0.075 | −0.023 |
| outside the largest cluster | +0.067 | +0.036 |
| sc009 only | +0.100 | −0.030 |

**Reading.**
- The in-sample "chemotype shape" headroom (+0.16) is real *within a publication* but not *across publications*. The leave-one-cell-out gain comes from sibling cells of the same publication (residual correlation 0.33). Across publications the gain is about 0.
- **Interpretation:** this is a laboratory or series effect. It could be per-metal calibration, stock or tracer biases, or a lab's chemistry series sharing conditions. It is not an extractant property the structure could predict.

### (c) Best convex blend of the existing arms

Arms: PWL_R, PWL_Z, LINE, RIDGE2, HBK@ML, G14@ZERO, FLAT, B1, B2, S11, S05, S06, S02, COR_G. Weights are non-negative and sum to 1.

| blend | in sample BPG / BP | leave-one-cluster-out | leave-one-chemotype-out |
|---|---|---|---|
| one global blend (weights: B1 0.56, S05 0.22, HBK 0.20, FLAT 0.02) | 0.4552 / 0.4547 | — | — |
| per pair type | **0.4775 / 0.4767** | **0.4225 / 0.4243** | 0.4266 / 0.4285 |
| per pair × one-end / both-hidden | 0.4863 / 0.4852 | 0.4178 / 0.4170 | 0.4250 / 0.4238 |
| per pair × interior / extrapolated | 0.4823 / 0.4811 | 0.4184 / 0.4204 | 0.4256 / 0.4275 |
| per region | 0.4656 / 0.4650 | 0.4331 / 0.4284 | 0.4366 / 0.4357 |
| per-pair scale of B1 (y ~ c·B1) | — | 0.3997 / 0.4028 | — |

The in-sample weights are unstable between designs. For example, on Ce/Pr BPG gives PWL_Z 0.32 and HBK 0.66, while BP gives PWL_Z 0.62 and HBK 0.34. **The blending headroom is +0.003 to +0.034 in sample and negative when held out** (−0.019 to −0.035 by leave-one-cluster-out).

### (d) A perfect fix of the worst cells

- Fixing the worst 10 % of cells (23, ranked by B1 SSE) gives R² **0.745** (+0.293); ranked by MSE, 0.734.
- Fixing the worst 5 % (11 cells) gives 0.662.
- Removing only the one C5-BTBP publication (15 cells, 8.2 % of rows) gives B1 **0.486** (PWL_R 0.459).

### Other oracles and probes

- **Per-cell constant shift of all of a cell's pair residuals:**
  - in sample 0.470;
  - leave-one-pair-out **0.292**. The residuals are not a cell-level offset; the share of residual sum of squares explained by cell means is only 0.026.
- **Smooth curve fitted to the same cell's other metals** (leave-pair-out; all much worse than B1's 0.452 with 4 reads):

  | fit | R² |
  |---|---|
  | line | 0.196 |
  | quadratic | 0.165 |
  | quadratic + Gd | 0.191 |
  | quadratic, in sample (all metals) | 0.337 |
  | cubic, in sample (all metals) | 0.409 |

  The true curves have kinks that local interpolation follows better than any global low-order shape.
- **Local-neighbour oracle** (a line through the TRUE levels of the two metals flanking the pair; 1,051 of 1,802 rows): MSE 0.0277 against B1's 0.0238 on the same rows. The median squared error is 0.0021 vs 0.0026. Even 12 of 14 metals read do not beat B1 in mean square. The residual is local, non-smooth structure.
- **Curvature-aware interpolation of the same reads, plus B1's Gd term:**

  | interpolation | ΔR² BPG / BP |
  |---|---|
  | PCHIP | −0.020 / −0.020 |
  | natural cubic spline | −0.023 / −0.022 |
  | local quadratic through 3 reads | −0.095 / −0.098 |
  | linear continuation beyond the hull | −0.004 / −0.007 |
  | **leave-one-cluster-out shrunk continuation** | **+0.0055 / +0.0052** (extrapolated-row R² −0.01 → +0.09) |
  | leave-one-cluster-out pair-type mean on extrapolated rows | +0.002 / +0.003 |

---

## 4. Which error components a model could plausibly reduce

Shares are of B1's SSE, and the categories overlap.

| component | share | can a model reduce it under BP / BPG? | evidence |
|---|---|---|---|
| tail out of the measurement window (either log D outside [−2, 2]) | ~50 % | **No.** It is noise, floor or ceiling; it is reducible by data quality and abstention | MSE 0.052 vs 0.018; roughness ρ 0.88; replicate noise ≈ error in the noisiest cells |
| C5-BTBP suspected-floor source pub_6a0ef8528c | 23 % | **No.** It is a data-curation problem (PROVENANCE G-V3 fails) | one cell 11.8 %; B1 0.486 without it |
| Ce-involving pairs (La/Ce, Ce/Pr) | 34 % | **Not by modelling.** Pair offsets and blends fail held out; reading Ce removes it | pair-mean leave-one-cluster-out < B1; Ce read +0.148 same rows, −0.012 like for like |
| within-publication systematic pattern | ≈ 13 % (ΔR² 0.071 of 0.548) | **Not under publication hold-out**; yes with same-laboratory information | publication × pair leave-one-cell-out +0.071; other publications +0.005; sibling correlation 0.33 vs 0.10–0.14 |
| cross-publication extractant or chemotype signal | ≈ 1–2 % | the only part a structure model can target; at noise level | chemotype other publications +0.005; same-extractant other-publication residual correlation 0.10–0.14 |
| extrapolation beyond the heaviest read | 9.7 % | **slightly** | shrunk continuation +0.005 held out |
| Gd region (Eu/Gd, Gd/Tb) after the Gd term | 16.5 % | no (S01 / S02 / S05 / S06 / S11 failed) | — |
| heavy pairs Tm/Yb, Yb/Lu (R² ≈ 0) | 8.6 % | low ceiling (+0.047 even if perfect) | small variance, sd 0.12 |

**Implications for round 5** (interpretation; to be pre-registered before use, not results):

1. **The ceiling of "a better model" on this corpus is a few hundredths.** It is the cross-publication extractant signal (≈ 0.01) plus extrapolation (≈ 0.005). That matches rounds 1–4. More model capacity aimed at the current target is unlikely to reach 0.50.
2. **A deployable 0.50 exists on a declared applicability domain.** The label-free rule "predicted levels of both metals in [−2, 2]" gives R² 0.540 at 75 % coverage, versus 0.302 outside.
   - This is a *selective-prediction* claim, not a model gain.
   - It should be registered as a coverage–R² curve with a random-abstention control.
3. **The measurement policy is the strongest lever.**
   - Like for like, max-gap beats DOPT_Q's own 5th metal (+0.049 vs +0.032).
   - A random 5th metal beats DOPT_Q's 5th metal on the same rows.
   - Ce is never read, yet it carries 16 % of the level error.
   - An error-aware or robustness-aware read policy, and the question "which pairs does the user need measured?", deserve a pre-registered k = 4 comparison.
4. **A same-laboratory calibration mode is the only route to the systematic part.**
   - Publication-level residual patterns are worth +0.05 to +0.10 when other cells of the same publication are known.
   - It needs a different protocol: a reference curve measured by the same laboratory, or a publication-few-shot design. It cannot come from the current BP / BPG.
5. **Data quality and metric.**
   - The V2 rules or floor flags, and the C5-BTBP verification, move the reference more than any model has: B1 0.452 → 0.486 without one source.
   - 1 % of rows carry 31 % of SSE, so R² is a tail statistic. Report a robust metric (median |res| 0.058, robust sd 0.086) alongside it.

---

## 5. Caveats

- **What these numbers are.** All of them are dev-seed out-of-fold rows on the same 229 cells. They describe where B1's error sits; they are not independent replications.
- **Oracles are not baselines.** The in-sample oracles (b, c) fit the scored rows. The leave-one-cell-out and leave-one-publication-out corrections reuse other cells' out-of-fold residuals, so they are exploratory and not fold-honest refits.
- **Label-dependent windows (§1.6, first table)** select on the outcome. Only the label-free rules are candidates for a deployable domain.
- **The k = 4 oracle (a) is a simplification:**
  - it keeps the k = 3 Gd β;
  - the DOPT_Q 5th metal comes from round 3's R3-B0 ladder parquet, which is available only for cells with k_max ≥ 4;
  - B1 is used elsewhere.
- **Noise handles:**
  - only 15 C3 cells have true replicate series;
  - the smoothness bounds include real non-smooth chemistry (Ce, Gd);
  - the sibling bounds include condition changes.

  The noise share is therefore bounded, not measured.
- **Clustering.** The source cluster is concentrated (Kish 2.24, the largest cluster holds 65 % of rows), so every leave-one-cluster-out number rests on 13 clusters.
