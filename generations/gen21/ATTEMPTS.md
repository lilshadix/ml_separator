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

**Round status (2026-10-18; the user limited this round to three ideas).** Candidate: HB2+GP at
k = 3 with DOPT_Q reads, R² 0.403 (BP) / 0.407 (BPG) on dev. The final check on the withheld seeds
has **not** been run. It needs a method frozen on dev, a declared regime (A2) and the user's
go-ahead (PROTOCOL §5). Resume point: choose the next idea, or freeze HB2+GP, declare the regime
and run the final check once.
