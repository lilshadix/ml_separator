# gen21 — attempt log

Every attempt: hypothesis (from the error analysis of the previous one), information contract
(written before it ran), method, result table, short report. All numbers: dev seeds, metal
model, strict-adjacent `logSF(A/B) = logD(A) − logD(B)` with A lighter, cohort C3 (229 cells /
1,802 pairs per seed), mean over the 5 seeds of the per-seed pooled statistic. Intervals are
nominal-95 % percentile chemotype-block bootstraps (2,000 draws, gen20 `inference`). With about
four effective chemotypes their real coverage is closer to 90 % (gen20's inference audit). BCa
bounds and p values are listed beside them in `contrasts_*_C3.csv`. Contrasts are signed so that
**positive favours the candidate**: ΔR² = candidate − reference; ΔMAE, ΔRMSE = reference −
candidate. "x/5 seeds" counts split seeds that re-partition the same cells, so it measures
partition stability, not independent replication. BP = chemotype hold-out with test publications
removed; BPG = BP plus cross-publication copy groups removed (strict). The intervention leakage
check runs on **all five dev seeds** of each design (every fold, every k). Row-level predictions:
`predictions/attempt<N>_<design>.parquet`; tables: `results/attempt<N>/`; diagnosis outputs:
`results/diag/diag.txt` (`scripts/g21_diag.py`).

**Resume point:** after the last entry below. Rebuild everything with
`scripts/g21_build_cache.py` (≈2.5 min), then `scripts/g21_attempt.py --attempt <N>`.

---

## Attempt 0 — reproduction and diagnosis (no new method)

* gen20's RIDGE2 at k = 3 (BP, C3, DOPT_Q, metal model) reproduced exactly from gen20's own code:
  R² 0.3099, MAE 0.1206, macro 0.1346 (gen20: 0.310 / 0.121 / 0.135), and every other arm to four
  decimals (`results/reproduction_gen20_BP_C3_DOPT_Q.csv`).
* Diagnosis (`scripts/g21_diag0.py`): at k = 3, 6,015 of 9,010 pooled query rows have one endpoint
  read. On those rows the residual of RIDGE2's fitted level at the read metal correlates **0.31**
  with the remaining error (least-squares β 0.49). RIDGE2 smooths through the read value, but
  the scored target contains that same reading, so part of the error is visible at prediction
  time and RIDGE2 discards it.

## Attempt 1 — predict the observable log SF, not the latent curve (`OBS`)

**Hypothesis.** Under the noise model the gen20 arms already assume (per-metal reading noise s²,
R = s² D Dᵀ), the best predictor of the observed `y(a) − y(b)` adds the posterior mean of the read
metal's own noise, `E[ε | supports] = s² Dᵀ Var(y_sup)⁻¹ (y_sup − E y_sup)`, to the latent curve.
This should lower the error on pairs with a read endpoint.

**Information contract.** k-shot metal model; k + 1 metals read by gen20's DOPT_Q (label-free); the
method uses only the support differences to the first read metal (as gen20 does). Prior, λ, s²
and the G14 covariance come from the outer training fold. `@RESEL` re-chooses (noise factor ∈
gen20's grid, λ ∈ gen20's grid) for the OBS rule on the 4-fold inner cross-fit of the training
fold (pooled squared error, k = 1–3, inner-validation cells with k_max ≥ 3). `RIDGE2@RESEL` is the
latent RIDGE2 at the same re-chosen parameters (control for the re-selection itself). k = 0 is
identical to RIDGE2's prior curve.

**Result (k = 3, C3; full k ladder in `results/attempt1/levels_*.csv`).**

| # | method | regime | cells / pairs per seed | R² logSF BP [95 %] | MAE | RMSE | macro MAE | R² BPG (strict) | MAE BPG | ref. RIDGE2 R² BP / BPG |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | RIDGE2+OBS@RESEL | k = 3, DOPT_Q | 229 / 1,802 | **0.357** [0.295, 0.433] | 0.1183 | 0.1780 | 0.1289 | **0.358** | 0.1183 | 0.310 / 0.323 |
| 1 | RIDGE2@RESEL (control) | k = 3, DOPT_Q | 229 / 1,802 | 0.323 [0.231, 0.410] | 0.1191 | 0.1826 | 0.1336 | 0.323 | 0.1188 | 0.310 / 0.323 |
| 1 | RIDGE2+OBS (gen20 params) | k = 3, DOPT_Q | 229 / 1,802 | 0.305 [0.233, 0.395] | 0.1246 | 0.1849 | 0.1332 | 0.335 | 0.1202 | 0.310 / 0.323 |
| 1 | G14@BLUP20+OBS | k = 3, DOPT_Q | 229 / 1,802 | 0.164 | 0.1378 | 0.2026 | 0.1464 | 0.173 | 0.1376 | G14@BLUP20 0.248 / 0.246 |
| ref | RIDGE2 (gen20) | k = 3, DOPT_Q | 229 / 1,802 | 0.310 [0.221, 0.401] | 0.1206 | 0.1844 | 0.1346 | 0.323 | 0.1191 | — |
| ref | LINE | k = 3 | 229 / 1,802 | 0.236 | 0.1298 | 0.1941 | 0.1426 | 0.237 | 0.1297 | — |
| ref | G14 zero-shot | k = 0 | 229 / 1,802 | 0.053 | 0.1469 | 0.2160 | 0.1612 | 0.053 | 0.1468 | — |

Paired contrasts on identical rows (`contrasts_*_C3.csv`): RIDGE2+OBS@RESEL − RIDGE2 at k = 3:
ΔR² **+0.047** [+0.015, +0.090] (BP; 5/5 seeds), +0.035 [+0.001, +0.089] (BPG; 5/5 seeds); ΔMAE
+0.0023 [−0.0009, +0.0102] (BP; 3/5 seeds), +0.0007 (BPG; 3/5). Against its own latent control
(same parameters) ΔR² +0.034 [+0.003, +0.079]. At k = 1 OBS changes nothing (−0.008), at k = 2
+0.022. Split of the BP gain: re-choosing the noise level and λ gives +0.013 (RIDGE2@RESEL −
RIDGE2, 4/5 seeds), and the observable rule at those parameters gives +0.034 (5/5; MAE 3/5).
Under BPG almost all of the gain comes from the observable rule (+0.000 + 0.035; MAE 2/5). log D
of the unread metals (absolute level fixed by the reads, k = 3, R² / MAE / RMSE):
RIDGE2+OBS@RESEL 0.980 / 0.161 / 0.232; RIDGE2 0.977 / 0.173 / 0.249; mean of the reads 0.823 /
0.508 / 0.690. This number is dominated by between-cell level differences and says nothing about
separation. Intervention test (hidden log D replaced by random numbers): max change 0 (counts in
`summary.json`). No scored pair has both ends read.

**Short report.** The observable rule raises R² by about 0.035–0.047 under both hold-outs and
every seed, but only when its noise level and λ are re-chosen for it inside the training fold:
with gen20's parameters (chosen for the latent target) it is no better than RIDGE2, and on the
G14 covariance it is clearly worse (−0.08). The gain is in squared error, not in absolute error
(MAE better in 3 of 5 seeds; interval includes 0), so it removes some large errors rather than
improving the typical pair. **Advancement rule (PROTOCOL §6) not met** (MAE not better in ≥ 4/5
seeds). Per pair type the gain is large at Ce/Pr (0.31 → 0.41), Eu/Gd (0.13 → 0.35) and Gd/Tb
(−0.03 → 0.38), where the quadratic curve misses the half-shell break. The change is negative at
Dy/Ho (−0.13), Ho/Er (0.09 → −0.14) and Yb/Lu (−0.34), and slightly at Sm/Eu (−0.07) and Tb/Dy
(−0.04). Ho/Er, Tm/Yb and Yb/Lu are below zero R²; Er/Tm rises to +0.04. 59 % of the remaining
squared error sits in the top |y| quintile (|y| > 0.27), and two extractants (a BTBP-type N-donor
and TODGA) carry 38 % of it (`results/diag/diag.txt`). **Next hypothesis:** the residual is a curve-shape problem (half-shell
break, heavy-end flattening). It needs a richer shape prior whose variance components are
estimated from the training curves, not chosen on a grid.

## Attempt 2 — hierarchical empirical-Bayes curve model (`HB`)

**Hypothesis (from attempt 1's errors).** The remaining error is a curve-shape problem (the
half-shell break, the flattening at the heavy end) plus shrinkage set on a coarse grid. A
hierarchical model of the whole log D curve — shape coefficients θ with a full prior covariance,
a Z-correlated smooth residual δ (Cov = τ² ρ^|ΔZ|) and per-metal reading noise ε, with every
variance component estimated by maximum likelihood on the training curves — should predict the
adjacent log SF better than RIDGE2+OBS@RESEL on the same supports.

**Information contract.** As attempt 1: k + 1 metals per held-out cell read by gen20's DOPT_Q; only
their differences are used. The model `y = μ1 + Bᵀθ + δ + ε` (μ free per cell; θ ~ N(m, S_θ))
is fitted by chemotype-balanced weighted ML of the within-cell contrasts of the **outer training
cells'** observed curves (gen13's balanced cell weights); nothing is tuned on held-out cells. The
prediction is the conditional mean of the observable y(a) − y(b) given the read contrasts. k = 0
= the training mean shape (zero-shot, no chemistry). Variants: HB2 (gen20's 2-row radius basis,
no δ), HB2+GP, HB3H+GP (plus a hinge at Gd); `@LATENT` = the same model predicting the latent
curve (control). Fit ≈ 19 s per fold (3 variants), L-BFGS on the exact likelihood.

**Result (k = 3, C3).**

| # | method | regime | cells / pairs per seed | R² logSF BP [95 %] | per-seed R² BP | MAE | RMSE | macro MAE | R² BPG | MAE BPG | ref. RIDGE2 R² BP / BPG |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | **HB2+GP** | k = 3, DOPT_Q | 229 / 1,802 | **0.403** [0.318, 0.491] | 0.392, 0.416, 0.395, 0.412, 0.401 | 0.1079 | 0.1714 | 0.1215 | **0.407** | 0.1075 | 0.310 / 0.323 |
| 2 | HB3H+GP | k = 3 | 229 / 1,802 | 0.403 [0.341, 0.497] | 0.382, 0.419, 0.397, 0.422, 0.394 | 0.1080 | 0.1715 | 0.1199 | 0.408 | 0.1075 | 0.310 / 0.323 |
| 2 | HB2 (no δ) | k = 3 | 229 / 1,802 | 0.348 [0.292, 0.431] | — | 0.1200 | 0.1792 | 0.1290 | 0.352 | 0.1197 | 0.310 / 0.323 |
| 2 | HB2+GP@LATENT | k = 3 | 229 / 1,802 | 0.402 | — | 0.1071 | 0.1716 | 0.1217 | 0.406 | 0.1067 | — |
| 1 | RIDGE2+OBS@RESEL | k = 3 | 229 / 1,802 | 0.357 | — | 0.1183 | 0.1780 | 0.1289 | 0.358 | 0.1183 | — |
| 2 | HB2+GP, k = 2 | k = 2 | 229 / 1,802 | 0.334 | — | 0.1173 | 0.1812 | 0.1298 | 0.333 | 0.1177 | RIDGE2 0.279 |
| 2 | HB2+GP, k = 1 | k = 1 | 229 / 1,802 | 0.076 | — | 0.1484 | 0.2134 | 0.1494 | 0.080 | 0.1480 | RIDGE2 −0.008, LINE 0.168 |
| 2 | HB2+GP, k = 0 (zero-shot) | k = 0 | 229 / 1,802 | −0.116 | — | 0.1679 | 0.2345 | 0.1797 | −0.115 | 0.1678 | G14 0.053 |
| 2 | HB2+GP on C6 | k = 5 | 133 / 1,349 | 0.489 [0.426, 0.588] | 0.481, 0.500, 0.489, 0.492, 0.484 | 0.0988 | 0.1600 | 0.1030 | 0.488 | 0.0986 | RIDGE2 0.326 / 0.329 |
| 2 | HB2+GP on C6 | k = 6 | 133 / 1,349 | 0.521 [0.460, 0.620] | 0.505, 0.518, 0.538, 0.536, 0.510 | 0.0942 | 0.1549 | 0.0993 | 0.520 | 0.0942 | RIDGE2 0.323 / 0.326 |

log D of the unread metals (k = 3, R² / MAE / RMSE): HB2+GP 0.982 / 0.149 / 0.221; RIDGE2+OBS@RESEL
0.980 / 0.161 / 0.232; mean of the reads 0.823 / 0.508 / 0.690 (not a separation result).

Contrasts on identical rows, k = 3 (`contrasts_*_C3.csv`): HB2+GP − RIDGE2: ΔR² **+0.094**
[+0.076, +0.112] (BP), +0.085 [+0.069, +0.116] (BPG); ΔMAE **+0.0127** [+0.0098, +0.0176] (BP),
+0.0115 (BPG); **5/5 seeds on both statistics under both designs**. HB2+GP − RIDGE2+OBS@RESEL:
ΔR² +0.047 [+0.006, +0.077], ΔMAE +0.0104 [+0.0054, +0.0129], 5/5 seeds. HB3H+GP − HB2+GP: 0.000
(the Gd hinge adds nothing once δ is there). Observable minus latent: ±0.002. The ML noise level
σ² has median 0.006 (range 0–0.01 over the 25 folds, sd ≈ 0.08) and is not separately identified
from the short-range part of δ. The 0.30 replicate floor must still be mostly common-mode, for a
simpler reason: independent per-metal noise of sd 0.30 would give adjacent log SF a noise
variance of 0.18, above the total observed 0.049. Intervention test: max change 0. Fitted
parameters per fold: `results/attempt2/fitted_*.csv`. All fits report success. In one fold
(196613/2) L-BFGS stopped on the flat τ² → ∞, ρ → 1 ridge (τ² 36.5). A polish improves the
likelihood by only 0.06 and moves predictions by at most 0.002, so the reported levels are
unaffected, but that fold's τ²/ρ are an optimiser artefact.

**Short report.** This is the first method that clears the advancement rule (R² and MAE better
than RIDGE2 in 5/5 seeds, sign held on BPG). It is chosen as the round's candidate, by
parsimony, among variants within ±0.002 R² of each other (2-row basis, observable target as
hypothesised). HB3H+GP@LATENT is nominally ahead on R² and MAE under both designs, and the latent
variants have MAE lower by 0.0008 in 5/5 seeds; none of these differences is distinguishable. The gain comes from the smooth Z-correlated residual,
not from the extra basis term or the observable rule: HB2 without δ is only 0.348. Per pair type
it fixes the heavy end most (Ho/Er 0.09 → 0.18, Er/Tm −0.14 → 0.16, Tm/Yb −0.31 → −0.04,
Yb/Lu −0.11 → −0.07, Dy/Ho 0.40 → 0.47) and keeps the half-shell gains of attempt 1. The dev point
estimate, 0.403 (BP) / 0.407 (BPG), only just reaches the 0.40 threshold: 3 of 5 seeds are at or
above it under BP and 4 of 5 under BPG, and the chemotype-bootstrap lower bound is 0.32. This is a **dev** result on seeds whose baseline
numbers were known; it is not a confirmation. With fewer reads the model is no better than the
line (k = 1: 0.076 against LINE 0.168), and zero-shot it is no better than FLAT. **Error analysis
for the next step:** in extractant-publication series with ≥ 2 held-out cells (≈ 1,130 pair rows
per seed), **48 %** of HB2+GP's squared error is a component shared by the same pair type across
the series' conditions, and one BTBP series (15 cells) holds 24 % of all squared error
(`results/diag/diag.txt`). The model
re-estimates the same shape error independently in every condition of a series.

## Attempt 3 — pool the k-shot reads of a held-out condition series (`HBG`, setting P)

**Hypothesis (from attempt 2's errors).** About half of HB2+GP's squared error inside a held-out
condition series (same extractant, same publication, other conditions) is shared by the series.
An extractant-level hierarchy, with a series shape effect plus a small condition deviation, that
pools every condition's own k-shot reads should remove that shared part. It should beat the same
model on one cell's reads, and a pooled ridge line given identical reads.

**Information contract — SETTING P, a different regime from attempts 1–2.** For a held-out cell:
its own k + 1 reads, **plus** the own k + 1 reads (same k, capped at its k_max) of every other
held-out cell of the same extractant in the same publication. Only within-cell read differences
are used; no unread metal of any cell enters. It assumes the new extractant is measured as a
condition series with k + 1 metals read in every condition, so the extractant-level budget is
Σ(k + 1) over its cells; for a single-condition study it reduces to the own-reads prediction
(70 of 229 C3 cells per seed are singletons). Model: `y_c = μ_c1 + Bᵀ(θ_g + η_c) + δ_g + ν_c + ε_c`
(series shape θ_g, condition deviation η_c, series and condition Z-correlated residuals δ_g / ν_c
sharing one ρ, reading noise ε), fitted by weighted ML jointly per **training** series (exact
torch gradients, ≈ 11 s per fold). Reads: **attempt 3** = new label-free policy `SERIES_STAGGER`
(each condition of a series prefers the metals the series has not yet read; gen20's UNIFORM order
for the first cell and for singletons). **Attempt 3d** = the control with gen20's DOPT_Q reads,
which are identical across a series. Comparators on identical supports: HBG@OWN (same model, own
reads only), HB2+GP (attempt 2), RIDGE2@POOL (gen20's ridge line with one shape shared by the
series), RIDGE2. The flag `sib_pair_read` (both ends read in one sibling) is false by
construction: a sibling that has both metals of an adjacent pair has that pair as its own query
and never reads both. The check that matters is `dup_sibling`, added after the review. Seven
within-series pairs of cells are exact offset copies of one measurement series: three DGA pairs
in one paper with identical values, and four pyridine-dicarboxamide pairs with integer offsets.
In those pairs a sibling's reads reveal the target's hidden values. The flag is evaluation-only
(computed from labels, never shown to a method); the duplicate rows are scored separately below.

*Dev look before the full run* (2026-10-18; numbers added after the review asked for them; see
`results/diag/diag.txt`). As first planned, attempt 3 used DOPT_Q reads; that design is now 3d.
A one-fold smoke run on BP 130363 fold 2 (23 C3 cells, 232 pairs, k = 3) gave HBG@POOL R² 0.359,
against HBG@OWN 0.397, HB2+GP 0.398 and RIDGE2 0.401. Identical read sets cannot reveal metals
that no condition reads, so the staggered policy was introduced after this look and became
attempt 3; the original DOPT_Q design was kept as 3d. PROTOCOL §3 allows a method its own
label-free policy, but this policy was chosen after a dev result had been seen.

**Result (k = 3, C3, 229 cells / 1,802 pairs per seed; `results/attempt3/`, `results/attempt3d/`).**

| # | method | regime | R² logSF BP [95 %] | per-seed R² BP | MAE | RMSE | macro MAE | R² BPG | MAE BPG | ref. RIDGE2 R² BP / BPG (same reads) |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 | HBG@POOL | P, k = 3, SERIES_STAGGER | 0.341 [0.211, 0.463] | 0.313, 0.349, 0.330, 0.343, 0.371 | 0.1103 | 0.1801 | 0.1221 | 0.337 | 0.1106 | 0.292 / 0.300 |
| 3 | HBG@OWN | own reads, SERIES_STAGGER | 0.382 [0.296, 0.466] | — | 0.1088 | 0.1745 | 0.1222 | 0.381 | 0.1089 | 0.292 / 0.300 |
| 3 | HB2+GP | own reads, SERIES_STAGGER | 0.386 [0.297, 0.470] | — | 0.1082 | 0.1739 | 0.1220 | 0.386 | 0.1082 | 0.292 / 0.300 |
| 3 | RIDGE2@POOL | P, SERIES_STAGGER | 0.263 | — | 0.1235 | 0.1905 | 0.1339 | 0.263 | 0.1236 | 0.292 / 0.300 |
| 3d | HBG@POOL | P, k = 3, DOPT_Q | 0.406 [0.311, 0.489] | 0.379, 0.412, 0.408, 0.419, 0.409 | 0.1078 | 0.1711 | 0.1220 | 0.407 | 0.1076 | 0.310 / 0.323 |
| 3d | HB2+GP | own reads, DOPT_Q | 0.403 [0.318, 0.491] | — | 0.1079 | 0.1714 | 0.1215 | 0.407 | 0.1075 | 0.310 / 0.323 |
| 3d | RIDGE2@POOL | P, DOPT_Q | 0.259 | — | 0.1259 | 0.1911 | 0.1361 | 0.261 | 0.1257 | 0.310 / 0.323 |

log D of the unread metals (k = 3, R² / MAE / RMSE, BP): HBG@POOL 0.984 / 0.145 / 0.207
(staggered), 0.983 / 0.145 / 0.216 (DOPT_Q); HB2+GP 0.984 / 0.146 / 0.209 (staggered); mean of
the reads 0.824 / 0.504 / 0.687. Not a separation result.

Contrasts at k = 3 (identical rows and reads). Staggered: HBG@POOL − HBG@OWN ΔR² **−0.041**
[−0.106, +0.010] (BP; BCa [−0.124, +0.003]), −0.044 (BPG), 0/5 seeds better; HBG@POOL − HB2+GP
−0.045 (0/5). On the cells that have siblings (159 cells / 1,178 pairs per seed) the loss is
larger: 0.294 against 0.362 (own). DOPT_Q control: HBG@POOL − HB2+GP **+0.002** [−0.010, +0.011]
(BP, 3/5 seeds), −0.000 (BPG, 2/5). Against RIDGE2, both pooled variants have lower MAE with
intervals that exclude 0 (staggered +0.0092 [+0.0032, +0.0122]). On R² the staggered variant is
+0.049 [−0.043, +0.107] and the DOPT_Q variant +0.096 [+0.070, +0.114]. At k = 3, pooling a shared
shape hurts the ridge line (RIDGE2@POOL − RIDGE2 −0.03 to −0.06).

Budget ladder (BP, C3). DOPT_Q reads, HBG@POOL − HB2+GP: k = 1 +0.074 [−0.048, +0.115]; k = 2
+0.018 [−0.006, +0.031] (5/5 seeds, both designs; against HBG@OWN +0.025, BCa excludes 0); k = 3
+0.002. Staggered reads, HBG@POOL − HBG@OWN: −0.029, −0.022, −0.041 at k = 1, 2, 3. HB2+GP on own
reads scores 0.179 / 0.288 / 0.386 under SERIES_STAGGER, against 0.076 / 0.334 / 0.403 under
DOPT_Q. RIDGE2@POOL is unstable at k = 1 under SERIES_STAGGER (R² −1.23).

Duplicate-copy rows (14 cells / 108 pairs per seed, `subsets_*_C3_k3.csv`): there staggered pooling
does help (HBG@POOL 0.245 against HBG@OWN 0.187), as expected when a sibling's reads reveal the
target's own values. Without them (215 cells / 1,694 pairs) it is HBG@POOL 0.341 against HBG@OWN
0.384 and HB2+GP 0.388 (BP; BPG 0.337 / 0.383 / 0.388). The duplicates flatter pooling, and the
conclusion does not depend on them.

Intervention test (every unread metal of every held-out cell, siblings included, replaced at each
k, all five dev seeds): max change 0. No pair is read at both ends in any cell. SYNTHETIC code
check (`tests/test_g21.py`): series drawn from the model itself, with condition-level shape
variance 0.005 and τ²_cell/τ²_group = 0.1 (an easier regime than the real fits). There the fit
recovers the parameters and staggered pooling lowers the MSE by 19 %. The code does what the
model says; the negative result comes from the real data.

**Short report.** The hypothesis is **refuted** on dev at the target budget k = 3, where pooling a
series' k-shot reads does not improve the adjacent log SF. With identical (DOPT_Q) read sets it is
neutral at k = 3, as the model predicts when no new metals are seen, and gives small gains at
k = 1–2. With staggered read sets it is worse in point estimate at every k: −0.04 R² at k = 3, 0/5
seeds, interval [−0.106, +0.010], so no gain larger than about +0.01 is compatible with the data.
The worst pairs are La/Ce (0.17 → −0.15, Δ −0.32) and Yb/Lu (0.23 → 0.08, Δ −0.15). On data that
follow the model, the same procedure helps. So the error shared within a series is **not** a
stable series shape at the unread metals: reading those metals in a sibling condition does not
predict them in this condition. The fitted cross-condition variance is not tiny. Over 25 folds
the median S_cell is 0.035 / 0.0065, against S_group 0.20 / 0.10 (≈ 1/6 and 1/13), and the
condition-level residual τ² (median 0.40) is comparable to the series one (0.58). That this is
still too small for the real series is inferred from the pooled loss, not read off the fit.
Likely reasons are heteroscedastic series (some vary acid or diluent strongly) and per-paper,
metal-specific measurement bias. The staggered policy also costs the own-reads model 0.017 R² at
k = 3 against DOPT_Q, while helping at k = 1 (+0.10). Read choice matters for this model, unlike
for gen20's arms. **Not advanced.** HBG@POOL does meet the RIDGE2-only rule of PROTOCOL §6
(staggered BP ΔR² +0.049 / ΔMAE +0.0092, 5/5; BPG +0.038 / +0.0080, 5/5). It was rejected because
it loses to its own control (HBG@OWN) and to the previous best (HB2+GP) on identical reads; see
amendment A1 below. The round's candidate stays HB2+GP with DOPT_Q reads (attempt 2).

---

## Amendments to PROTOCOL.md (dated; the results already seen are listed)

* **A1 (2026-10-18, after attempts 1–3 were seen).** The advancement rule of PROTOCOL §6 is
  necessary but not sufficient. A candidate must also beat the previous best gen21 method and its
  own control on identical rows and supports, in R² and MAE, in ≥ 4/5 seeds, with the sign held
  on BPG. Attempt 3 was judged this way; the extra condition only ever rejects more.
* **A2 (2026-10-18, after the dev results were seen).** **No regime for the final check has been
  declared yet.** PROTOCOL §4/§6 make C3 the primary cohort, but nothing fixed the budget. Seen on
  dev: HB2+GP at C3 k = 3, 0.403 (BP; 3/5 seeds ≥ 0.40) / 0.407 (BPG; 4/5); at C6 k = 5,
  0.489 / 0.488 (every seed 0.48–0.50); at C6 k = 6, 0.521. Declaring C6 at k = 5 or 6 would replace
  the pre-registered primary cohort after these numbers were seen, and it must be recorded here as
  such a change before any final run. The default under the protocol as written is **C3, k = 3,
  DOPT_Q reads**.
* **A3 (2026-10-18).** The final check re-partitions the same 521 cells with new split seeds. It
  guards only against split-specific over-fitting and cannot confirm a regime or variant chosen on
  these cells. Its expected seed-to-seed spread is the partition spread (about ±0.01 on a 5-seed
  mean), not the chemotype-bootstrap interval, which describes generalisation to new chemotypes.
* **A4 (2026-10-19, added after the round-2 review).** Round 2 (attempts 4–6) does not rerun
  G14@BLUP20 and FLAT@BLUP20. Their rows are identical to attempt 2's (`predictions/attempt2_*.parquet`):
  C3 k = 3 R² 0.248 / 0.189 (BP), both below RIDGE2 and LINE, which are rerun in every attempt.

**Round status (2026-10-18; the user limited this round to three ideas).** Candidate: HB2+GP at
k = 3 with DOPT_Q reads, R² 0.403 (BP) / 0.407 (BPG) on dev. The final check on the withheld seeds
has **not** been run. It needs a method frozen on dev, a declared regime (A2) and the user's
go-ahead (PROTOCOL §5). Resume point: choose the next idea, or freeze HB2+GP, declare the regime
and run the final check once.

---

# Round 2 (started 2026-10-19; the user asked to continue with new tests)

Same protocol, cohorts, dev seeds, designs and checks as round 1 (amendments A1–A3 apply). The
candidate going in is HB2+GP (attempt 2). The final check stays unrun.

## Diagnosis of HB2+GP before round 2 (`scripts/g21_diag_r2.py` → `results/diag/diag_r2.txt`)

On BP, k = 3, C3:
* **Distance to the nearest read metal.** Pairs with one end read have MSE 0.028. Pairs one Z-step
  from a read metal with neither end read have the highest raw MSE of the large groups (0.036,
  1,752 rows); pairs 2–3 steps away have 0.022–0.025, and the 63 rows 4 steps away 0.047. Relative
  to their own spread, those pairs are not special (MSE / sd² = 0.58 / 0.64 / 0.65 / 0.67 for
  distance 0–3). That the random-walk residual fails to carry the local slope past a read is
  therefore the hypothesis tested in attempt 4, not a diagnosed cause.
* **Chemotype bias.** A chemotype × pair-type mean explains 36 % of the squared error, against
  26 % under a permutation null (95th percentile 29 %). sc061 holds 25 % of the squared error at
  3× the MSE of the DGA chemotype. A held-out chemotype's own bias cannot be estimated, though.
  The structure model does not supply it either: regressing the HB2+GP residual on
  (G14@ZERO − HB2+GP) gives slope 0.08, correlation 0.06 and 0.5 % less squared error in-sample,
  so a structure-based prior is **not** pursued.
* **Read consistency.** Leave-one-read-out standardized residuals of the reads under the fitted
  model: cells with a read at |z| > 3 are 3 % of cells and 6 % of the squared error. Error rises
  steadily with the worst |z| (MSE 0.019 / 0.031 / 0.062 for |z| ≤ 1 / 1–2 / 2–3). Cells whose
  reads fit the model badly are the cells the model predicts badly: the curve variance differs
  between cells.

## Attempt 4 — the residual kernel (`HBK`)

**Hypothesis (from the diagnosis).** The exponential (Ornstein–Uhlenbeck) residual extrapolates no
local slope past a read metal, which is why pairs one step from a read are the worst. A smoother
residual (Matérn 3/2, Matérn 5/2 or squared-exponential in Z), chosen per fold by the **training**
marginal likelihood, should predict those pairs and the pooled adjacent log SF better than HB2+GP
on the same reads.

**Information contract.** As attempt 2: k + 1 metals read by gen20's DOPT_Q; only their
differences are used; parameters by chemotype-balanced weighted ML on the outer training cells'
curves. Variants `HBK[exp]` (= HB2+GP refitted), `HBK[m32]`, `HBK[m52]` and `HBK[se]` have the same
number of parameters (θ prior, τ², length or ρ, σ²). `HBK@ML` uses, in each fold, the kernel with
the highest training likelihood. Nothing is chosen on held-out cells. Comparators on identical rows
and reads: HB2+GP (attempt 2), RIDGE2, LINE, G14 zero-shot, FLAT. Advancement (PROTOCOL §6 + A1):
beat both RIDGE2 and HB2+GP in R² and MAE in ≥ 4/5 seeds, with the sign held on BPG.

**Result (k = 3, C3, 229 cells / 1,802 pairs per seed; `results/attempt4/`).**

| # | method | regime | R² logSF BP [95 %] | per-seed R² BP | MAE | RMSE | macro MAE | R² BPG [95 %] | per-seed R² BPG | MAE BPG |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | **HBK@ML** (kernel by training likelihood) | k = 3, DOPT_Q | **0.414** [0.321, 0.500] | 0.404, 0.425, 0.403, 0.427, 0.412 | **0.1054** | 0.1699 | 0.1208 | **0.419** [0.325, 0.503] | 0.402, 0.425, 0.420, 0.428, 0.419 | 0.1050 |
| 4 | HBK[m52] | k = 3 | 0.417 [0.326, 0.503] | — | 0.1049 | 0.1695 | 0.1203 | 0.421 | — | 0.1045 |
| 4 | HBK[m32] | k = 3 | 0.416 | — | 0.1051 | 0.1696 | 0.1203 | 0.421 | — | 0.1046 |
| 4 | HBK[se] | k = 3 | 0.415 | — | 0.1054 | 0.1697 | 0.1207 | 0.420 | — | 0.1050 |
| 2 | HB2+GP (= HBK[exp]) | k = 3 | 0.403 [0.318, 0.491] | 0.392, 0.416, 0.395, 0.412, 0.401 | 0.1079 | 0.1714 | 0.1215 | 0.407 | — | 0.1075 |
| ref | RIDGE2 | k = 3 | 0.310 | — | 0.1206 | 0.1844 | 0.1346 | 0.323 | — | 0.1191 |

Budget ladder, HBK@ML (BP, with BPG in brackets): C3 k = 0 −0.118, k = 1 0.099, k = 2 0.344, k = 3
0.414 (0.419); C6 k = 3 0.392, k = 5 **0.499** (0.497), k = 6 0.533 (0.531). HB2+GP at the same
points: −0.116, 0.076, 0.334, 0.403; C6 0.376, 0.489, 0.521. LINE: C3 −0.135, 0.168, 0.237, 0.236;
C6 (k = 3 / 5 / 6) 0.224, 0.230, 0.228. **At k = 1 (two metals read) HBK@ML (0.099) is still below
LINE (0.168).** log D of the unread metals (k = 3, R² / MAE / RMSE): HBK@ML 0.983 / 0.142 / 0.215;
HB2+GP 0.982 / 0.149 / 0.221; mean of the reads 0.823 / 0.508 / 0.690.

Contrasts at k = 3: HBK@ML − HB2+GP ΔR² **+0.011** [−0.006, +0.020] (BP) / +0.012 [−0.005,
+0.021] (BPG); ΔMAE +0.0024 [−0.0006, +0.0036] / +0.0025; **5/5 seeds on both statistics under
both designs**. HBK@ML − RIDGE2: ΔR² +0.104 [+0.078, +0.123], ΔMAE +0.0152, 5/5. Training
likelihood chose squared-exponential in 19 of 25 folds, Matérn 5/2 in 5 and Matérn 3/2 in 1
(both designs). Every fit converged. Intervention (all five dev seeds): 610,400 predictions per
design, max change 0.

**Short report.** The smoother residual helps a little, and in every seed. With HBK@ML every seed
crosses 0.40 under both designs (0.402–0.428, against 0.392–0.416 for HB2+GP). That is a +0.011
shift, of the size of the ±0.01 partition spread (A3). The gain over HB2+GP is small (+0.011 R²,
+0.0024 MAE). It is positive in every seed but inside the chemotype-level percentile interval (the
MAE BCa interval excludes 0). Seed counts measure partition stability, not replication. About half
of each seed's C3 rows are one chemotype (DGA), whose gain is positive in every seed; on the other
chemotypes the MAE gain is positive in 4/5 seeds and the squared-error gain in 3/5. It clears the advancement rule (PROTOCOL §6 + A1: better than
RIDGE2 and the previous best in R² and MAE in ≥ 4/5 seeds, sign held on BPG), so **HBK@ML becomes
the round's candidate**. The single kernels m52 and m32 are nominally 0.002–0.003 higher than the
ML choice, but HBK@ML is the rule declared before the run and is kept. Per pair type, the heavy end
gains most: Ho/Er 0.18 → 0.22, Er/Tm 0.16 → 0.21, Tm/Yb −0.04 → 0.01. Yb/Lu gets worse (−0.07 →
−0.14), and Pr/Nd and Eu/Gd slightly worse. Pairs one Z-step from a read improve only a little
(MSE 0.0363 → 0.0349), so the slope-extrapolation hypothesis explains only part of the gain.

**Diagnosis for the next step** (`scripts/g21_diag_a4.py` → `results/diag/diag_a4.txt`). Under
HBK@ML, Spearman(posterior variance, squared error) is 0.09 / 0.08 / 0.11 at k = 1 / 2 / 3. If the
variances were exactly calibrated (squared error = variance × χ²₁) it would be ≈ 0.15, so the rank
statistic alone is weak evidence (corrected after the review). The quintiles are what is
informative: under calibration the top/bottom quintile MSE ratio would be ≈ 2.2; observed it is
1.02 / 0.83 / 0.82. The inversion is between pair types. The prior gives Yb/Lu and Tm/Yb the
largest posterior variance (0.047, 0.038), where the realised MSE is among the lowest (0.017,
0.015), and La/Ce and Ce/Pr the smallest (0.030), where the MSE is the highest (0.058, 0.070); the
rank correlation over the 12 pair types is ≈ −0.46. Total posterior variance overstates the SSE by
18–21 %. Two consequences. A read policy that minimises this posterior variance (the planned
"DOPT_HB") would steer reads toward the heavy end, where the error is smallest. It was judged
unlikely to gain and deferred; its gain was not measured. And the error depends on cell-level
variance that a single Gaussian prior ignores, so the next attempt is the heteroscedastic mixture.

## Attempt 5 — heteroscedastic hierarchy: a two-component mixture (`HBMIX`)

**Hypothesis.** Cells come in at least two variance regimes, for example smooth, well-measured
curves against rough or noisy ones (like sc061). Fit a two-component mixture of HBK models by ML on
the training cells, and let the reads of a new cell decide which regime it belongs to. Each cell
then gets the shrinkage that suits it, which should lower the error and rank the uncertainty
better than HBK@ML on the same reads.

**Information contract.** As in attempts 2 and 4: k + 1 metals are read by gen20's DOPT_Q, and only
their differences are used. Two HBK components with the fold's ML kernel family (attempt 4's rule)
are fitted jointly by weighted mixture maximum likelihood on the outer training cells, with exact
torch gradients. The fit starts from the single HBK@ML fit, with one component's variances halved
and the other's doubled. Prediction: the component posterior from the likelihood of the read
contrasts under each component, then the posterior-weighted conditional means; k = 0 is the
prior-weighted mean. Nothing is tuned on held-out cells. Comparators on identical rows and reads:
HBK@ML (attempt 4), HB2+GP and RIDGE2.

*Dev look before the full run (2026-10-19, recorded here).* While the model was being written, it
was fitted on one dev fold (BP 130363 fold 2). The training likelihood rose by about 590, and one
third of the cells went to a noisy component with σ² ≈ 0.03–0.05. Held-out k = 3 R² on that fold:
with the exp kernel, 0.39 for the mixture against 0.40 for the single model; with the se kernel,
0.38 against 0.405. The full run goes ahead as planned; the look is disclosed because it came
first.

**Result (k = 3, C3, 229 cells / 1,802 pairs per seed; `results/attempt5/`).**

| # | method | regime | R² logSF BP [95 %] | per-seed R² BP | MAE | RMSE | macro MAE | R² BPG | per-seed R² BPG | MAE BPG |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | HBMIX | k = 3, DOPT_Q | 0.416 [0.320, 0.498] | 0.421, 0.417, 0.403, 0.427, 0.411 | 0.1052 | 0.1697 | 0.1204 | 0.421 | 0.417, 0.417, 0.421, 0.428, 0.423 | 0.1045 |
| 4 | HBK@ML (candidate) | k = 3, DOPT_Q | 0.414 [0.321, 0.500] | 0.404, 0.425, 0.403, 0.427, 0.412 | 0.1054 | 0.1699 | 0.1208 | 0.419 | 0.402, 0.425, 0.420, 0.428, 0.419 | 0.1050 |
| ref | RIDGE2 | k = 3 | 0.310 | — | 0.1206 | 0.1844 | 0.1346 | 0.323 | — | 0.1191 |

HBMIX − HBK@ML at k = 3: ΔR² **+0.0015** [−0.007, +0.006] (BP; 2/5 seeds) / +0.0023 (BPG; 4/5);
ΔMAE +0.0003 [−0.0010, +0.0009] (3/5) / +0.0005 (3/5). Ladder (BP): C3 k = 0 −0.114, k = 1 0.095,
k = 2 0.348, k = 3 0.416; C6 k = 5 0.499, k = 6 0.536 (HBK@ML: −0.118, 0.099, 0.344, 0.414; 0.499,
0.533). log D of the unread metals (k = 3): 0.983 / 0.142 / 0.215, the same as HBK@ML. Fits: all
converged. The mixture raises the training likelihood by a median of 524 over the single model.
Mean mixing weights are 0.59 / 0.41, with reading noise σ² median 0.0037 in the quiet component and
0.034 in the noisy one. Intervention (all five dev seeds): 427,280 predictions per design, max
change 0.

**Short report.** The training data are strongly heteroscedastic. About 40 % of training cells
fall in a component with ~10× the white-residual variance σ² and ~20× τ². σ² is a model residual,
not a measured reading noise: it includes structure the kernel does not resolve, and the single
model's σ² on the same cells is 0.006 with the exp kernel and 0.017 with se. Modelling the regimes
gives the held-out prediction **nothing measurable** (+0.002 R², 2/5 seeds on BP). The reads do
identify the regime (corrected after the review): for held-out C3 cells the k = 3 component
posterior has AUC 0.86 against the full-curve label, and the read call agrees with the full-curve
call in 81 % of cells. Even in the cells correctly identified as noisy, the mixture's error equals
HBK@ML's (MSE 0.0564 against 0.0566): regime-specific shrinkage does not lower the error. Those
noisy cells are 37.5 % of the held-out C3 cells but carry 79 % of HBK@ML's squared error (MSE 0.056
against 0.010). HBK@ML still explains about 36 % of their adjacent variance. Per pair type the
mixture helps Tm/Yb (0.01 → 0.06) and Yb/Lu (−0.14 → −0.09); elsewhere it changes R² by at most
±0.02 (largest Eu/Gd +0.017, Pr/Nd −0.010). **Not advanced** (fails ≥ 4/5 seeds on BP). The
candidate stays HBK@ML. On the noise ceiling: taken as pure noise, the noisy component's σ² would
be about half of the adjacent variance in those cells. But it is not identified as reading noise;
replicate whole-curve measurements would measure the real floor (untested).

## Attempt 6 — metal-specific effects in the prior (`HBKM`)

*Contract written 2026-10-19, before attempt 6 ran and while attempt 5 was still running; attempt
5's result (above) was inserted after both runs finished.*

**Hypothesis (from attempt 4's errors).** At k = 3, HBK@ML's errors have pair-type offsets around
the half-shell (dev BP, regression of y on the prediction per pair type; `scripts/g21_diag_a6.py` part 1 → `results/diag/diag_a6.txt`). Mean residuals
(y − prediction): Eu/Gd +0.087, Gd/Tb −0.102, Ce/Pr +0.050. A constant offset alone would remove
21 % and 36 % of the Eu/Gd and Gd/Tb squared error (intercept and slope: 21 % and 37 %). This is
in-sample, on the same dev rows on which attempt 6 is then scored (wording corrected after the
review; the contract originally quoted the regression intercepts +0.081 / −0.086 / +0.074). The heavy pairs are over-dispersed (slope 0.54 at Tm/Yb, 0.36 at Yb/Lu).
This is the signature of per-metal level anomalies shared by all extractants (the gadolinium
break) and of metal-dependent reading noise. A smooth radius prior cannot represent either unless
the metal is read. Adding to the HBK prior (a) a per-metal mean vector α in the complement of
span{1, basis} (11 numbers) and (b) optionally a per-metal noise variance (14 numbers), all fitted
on training curves, should remove the offsets and the over-dispersion. It should beat HBK@ML on
the same reads.

**Information contract.** As attempts 2 and 4: k + 1 metals are read by gen20's DOPT_Q, and only
their differences are used. The fold's ML kernel family is taken from attempt 4's rule. α and the
per-metal noise are shared by all chemotypes, so they transfer to a held-out chemotype. They are
fitted by chemotype-balanced weighted ML on the **outer training cells' curves** only (exact torch
gradients), starting from the fold's HBK@ML fit. Variants: `HBKM[mean]` (α), `HBKM[mean+noise]`
(α and per-metal σ²_m), and `HBKM@BIC`, the variant with the lower training BIC = 2·NLL +
n_params·ln(n_contrasts) in each fold. The **declared candidate is HBKM@BIC**. Comparators on
identical rows and reads: HBK@ML (round candidate), HB2+GP, RIDGE2, LINE, G14 zero-shot, FLAT;
attempt 5's HBMIX is compared offline on identical rows if it is advanced. Advancement: PROTOCOL
§6 + A1 against HBK@ML.

*Looked at before the run:* only the training fit on one fold (BP 130363 fold 2), no held-out
score. α_Gd = −0.083, α_Sm +0.06, α_Tm +0.06; σ²_Tm 0.077 and σ²_Er 0.035 are the noisy metals;
the training likelihood rises by 102 (mean) and 428 (mean + noise).

**Result (k = 3, C3, 229 cells / 1,802 pairs per seed; `results/attempt6/`).**

| # | method | regime | R² logSF BP [95 %] | per-seed R² BP | MAE | RMSE | macro MAE | R² BPG | MAE BPG |
|---|---|---|---|---|---|---|---|---|---|
| 6 | HBKM@BIC (declared) | k = 3, DOPT_Q | 0.407 [0.331, 0.500] | 0.395, 0.434, 0.407, 0.399, 0.400 | 0.1095 | 0.1709 | 0.1232 | 0.411 | 0.1094 |
| 6 | HBKM[mean] | k = 3 | 0.412 [0.340, 0.509] | 0.397, 0.426, 0.417, 0.422, 0.401 | 0.1091 | 0.1701 | 0.1219 | 0.417 | 0.1089 |
| 6 | HBKM[mean+noise] | k = 3 | 0.408 | — | 0.1091 | 0.1707 | 0.1227 | 0.412 | 0.1091 |
| 4 | HBK@ML (candidate) | k = 3 | 0.414 [0.321, 0.500] | 0.404, 0.425, 0.403, 0.427, 0.412 | 0.1054 | 0.1699 | 0.1208 | 0.419 | 0.1050 |
| ref | RIDGE2 | k = 3 | 0.310 [0.221, 0.401] | 0.303, 0.335, 0.294, 0.313, 0.305 | 0.1206 | 0.1844 | 0.1346 | 0.323 | 0.1191 |

HBKM@BIC − RIDGE2: ΔR² +0.097 [+0.062, +0.141]. HBKM@BIC − HBK@ML at k = 3: ΔR² **−0.007** [−0.041, +0.041] (BP; 2/5 seeds) / −0.007 (BPG; 2/5);
ΔMAE **−0.0041** [−0.0085, +0.0064] (**0/5 seeds**, both designs). BIC chose mean + noise in 23 of
25 folds. HBKM[mean+noise] stopped at the L-BFGS-B iteration cap (maxiter 500, `success` False in
`fitted_*.csv`) in 18/25 BP and 19/25 BPG folds, 17 of them BIC choices; HBKM[mean] converged in
every fold (disclosed after the review). Post-hoc refits of capped folds to convergence moved the
training NLL by at most 0.17, held-out predictions by at most 0.0033 and MAE by at most 0.0001,
and did not change the BIC choice; the verdict is unaffected. Ladder (BP): C3 k = 1 0.105, k = 2 0.354, k = 3 0.407; C6 k = 5 0.500, k = 6 0.532.
HBKM[mean] is best on C6 (k = 5 0.514, k = 6 0.543, against 0.499 / 0.533 for HBK@ML). log D of
the unread metals (k = 3): 0.983 / 0.146 / 0.217. Intervention (all five dev seeds): 549,360
predictions per design, max change 0.

**Why it fails — half of the hypothesis holds.** The fitted metal effects (HBKM[mean], 25 BP
folds; `fitted_BP.csv`; `scripts/g21_diag_a6.py` part 2 → `results/diag/diag_a6.txt`) include a **gadolinium break**: α_Gd = −0.096 ± 0.020, negative in all 25 folds
(α_Tb is positive and α_Yb negative in all folds as well). The folds share most of their training
cells, so this stability is not a transfer test. Pooled over held-out chemotypes, MAE falls at
Eu/Gd (0.150 → 0.124) and Gd/Tb (0.129 → 0.104), but the gain comes from the **non-DGA**
chemotypes (Eu/Gd 0.173 → 0.129, Gd/Tb 0.154 → 0.105, every seed; sc061 alone gives 31 % / 39 % of
it). On held-out DGA (sc009, 49 % of C3 rows) it is Eu/Gd 0.124 → 0.119 and Gd/Tb 0.101 → 0.102:
α_Gd fitted without DGA (−0.10 to −0.14, against −0.06 to −0.12 in the other folds) is larger
than DGA's own break, and with Gd unread the mean residual flips sign (Eu/Gd +0.084 → −0.040, Gd/Tb
−0.064 → +0.082). The shift comes from the whole α vector (α_Tb supplies about 30 % of the Gd/Tb
shift), and these pair types were picked from the same dev rows. The other components do not
transfer: α_Tm ranges from −0.03 to +0.11 across folds (sd 0.038) and
α_Lu from −0.09 to +0.01. These estimates come from few, noisy training measurements, and they
damage the small-variance heavy pairs: MAE at Er/Tm 0.068 → 0.097, Tm/Yb 0.060 → 0.091 (0.103 with
per-metal noise), Ho/Er 0.068 → 0.079, Sm/Eu 0.112 → 0.126. By chemotype: on the non-DGA
chemotypes the Gd gain outweighs the heavy-pair losses (HBKM[mean] MAE 0.1113 → 0.1098, better in
4/5 seeds; HBKM@BIC 3/5). On held-out DGA (one fold per seed, trained on only 73–93 cells) there is
no Gd gain, and Sm/Eu, Er/Tm and Tm/Yb get worse (0.0993 → 0.1085, worse in 5/5 seeds). Pooled
MAE is therefore worse in every seed, but each of those 0/5 seeds contains the same DGA fold. The per-metal noise variant makes it worse still,
although training BIC prefers it in 23 folds: a better training fit does not mean better transfer.

**Short report.** **Refuted as declared** (HBKM@BIC loses to HBK@ML on MAE in 0/5 seeds). The
diagnosis splits the hypothesis. The Gd break is a real per-metal anomaly that transfers to the
non-DGA chemotypes, but its size differs by chemotype, and the value learned without DGA
over-corrects DGA. A free 11-number metal vector also over-fits the poorly measured heavy metals. **Not advanced.** The candidate
stays HBK@ML. The obvious next test, a Gd-only offset (one parameter) or shrunk metal effects with
an ML-estimated variance, was **not** run in this round (three ideas per round); it is proposed
for the next one. It must report DGA and non-DGA separately: one shared Gd offset learned without
DGA is expected to over-correct DGA cells unless its size is shrunk or varies by chemotype.

---

**Round 2 status (2026-10-19).** Three ideas tested: attempt 4 (smoother residual kernel) **advanced**;
attempt 5 (heteroscedastic mixture) neutral; attempt 6 (metal effects) refuted as declared. Its
Gd-break component helps the non-DGA chemotypes and over-corrects DGA. The deferred read-policy idea (DOPT_HB, code in
`g21/policies.py`, **not run**) is still open. Candidate: **HBK@ML**, k = 3, DOPT_Q reads, dev R²
0.414 (BP) / 0.419 (BPG), every seed ≥ 0.40 under both designs (seeds re-partition the same
cells); still below LINE at k = 1. The final check on the withheld
seeds has **not** been run; amendments A2/A3 still apply (no regime declared; the default is C3,
k = 3). Resume point: the next idea (Gd-only offset / shrunk metal effects, DOPT_HB) or freeze
HBK@ML, declare the regime and run the final check once.
