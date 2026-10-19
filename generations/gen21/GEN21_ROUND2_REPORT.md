# gen21 — round 2 report: three more ideas, a new candidate (HBK@ML)

*2026-10-19. Follows `GEN21_REPORT.md` (round 1) and uses the same protocol (`PROTOCOL.md` plus
amendments A1–A3 in `ATTEMPTS.md`), cohorts, development (dev) seeds, hold-out designs and checks.
Every number comes from a file under `results/attempt{4,5,6}/` or `results/diag/`, computed from the
row-level predictions in `predictions/`. All results are **dev** results. The final check on the
withheld seeds has **not** been run.*

## 1. Where round 2 started

Target: R² ≥ 0.40 (preferably ≥ 0.50) for strict-adjacent `logSF(A/B) = log10 D(A) − log10 D(B)`
(A lighter) on held-out cells, with each measurement budget reported separately. Round 1 ended
with candidate **HB2+GP**: a hierarchical curve model fitted by maximum likelihood on training
curves, with an exponential (random-walk) residual in Z. At k = 3 (four metals read, DOPT_Q
reads, cohort C3 of 229 cells / 1,802 pairs per seed) it scored R² 0.403 under BP and 0.407 under
BPG (the copy-group-strict design), with 3/5 and 4/5 seeds at or above 0.40.

## 2. Diagnosis before round 2 (`scripts/g21_diag_r2.py` → `results/diag/diag_r2.txt`)

* **Where the error sits.** Pairs one Z-step from a read metal, with neither end read, have the
  highest raw MSE of the large groups (0.036, against 0.028 for pairs with a read end). Relative to
  their own spread they are not special (MSE / sd² 0.58–0.67 at every distance). That the
  random-walk residual carries no local slope past a read is the hypothesis attempt 4 tests.
* **Chemotype-specific bias.** A chemotype × pair mean explains 36 % of the squared error, against
  26 % under a permutation null. But the structure model (G14) explains almost none of the residual
  (0.5 % in-sample), so a structure-based prior was not pursued.
* **Cell-level variance.** The error rises steadily with how badly a cell's reads fit the model.

## 3. The three ideas (full entries, contracts and dev looks: `ATTEMPTS.md`)

All at k = 3, C3, DOPT_Q reads, identical rows; BPG = strict copy check.

| # | hypothesis (from the previous diagnosis) | method | R² BP [95 %] | per-seed R² BP | MAE | RMSE | R² BPG | MAE BPG | vs previous candidate (Δ R² BP, seeds better R² / MAE) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | a smoother residual extrapolates the slope past a read | **HBK@ML**: residual kernel (exp / Matérn 3/2 / Matérn 5/2 / squared-exp.) chosen per fold by training likelihood | **0.414** [0.321, 0.500] | 0.404, 0.425, 0.403, 0.427, 0.412 | **0.1054** | 0.1699 | **0.419** | 0.1050 | vs HB2+GP +0.011 [−0.006, +0.020]; 5/5, 5/5 (both designs) | **advanced: new candidate** |
| 5 | cells come in variance regimes | HBMIX: two-component mixture of HBK models | 0.416 [0.320, 0.498] | 0.421, 0.417, 0.403, 0.427, 0.411 | 0.1052 | 0.1697 | 0.421 | 0.1045 | vs HBK@ML +0.0015 [−0.007, +0.006]; 2/5, 3/5 | neutral, not advanced |
| 6 | per-metal anomalies (Gd break) and metal noise | HBKM@BIC: per-metal mean (+ noise) in the prior | 0.407 [0.331, 0.500] | 0.395, 0.434, 0.407, 0.399, 0.400 | 0.1095 | 0.1709 | 0.411 | 0.1094 | vs HBK@ML −0.007 [−0.041, +0.041]; 2/5, **0/5** | refuted as declared |
| ref | — | HB2+GP (round 1) | 0.403 [0.318, 0.491] | 0.392, 0.416, 0.395, 0.412, 0.401 | 0.1079 | 0.1714 | 0.407 | 0.1075 | — | — |
| ref | — | RIDGE2 (gen20) | 0.310 [0.221, 0.401] | — | 0.1206 | 0.1844 | 0.323 | 0.1191 | — | — |
| ref | — | LINE (straight line through the reads) | 0.236 | — | 0.1298 | 0.1941 | 0.237 | 0.1297 | — | — |

"5/5 seeds" counts re-partitions of the same cells, so it measures partition stability, not
replication. About half of each seed's C3 rows are one chemotype (DGA). G14@BLUP20 and FLAT@BLUP20
were not rerun in round 2; their rows are identical to round 1's (0.248 / 0.189; amendment A4).

log D of the unread metals (k = 3, R² / MAE / RMSE): HBK@ML 0.983 / 0.142 / 0.215; HBMIX 0.983 /
0.142 / 0.215; HBKM@BIC 0.983 / 0.146 / 0.217; mean of the reads 0.823 / 0.508 / 0.690. As in round
1, this is driven by level differences between cells and says nothing about separation.

**Attempt 4.** A smoother residual helps a little, consistently: +0.011 R² and +0.0024 MAE, in every
seed under both designs, though inside the chemotype-bootstrap percentile interval (the MAE BCa
interval excludes 0). The DGA chemotype gains in every seed. On the other chemotypes the MAE gain
holds in 4/5 seeds and the squared-error gain in 3/5. Training likelihood chose the
squared-exponential kernel in 19 of 25 folds. With HBK@ML **every seed is ≥ 0.40 under both designs**.
The gain is spread over the heavy end (Ho/Er 0.18 → 0.22, Er/Tm 0.16 → 0.21, Tm/Yb −0.04 → 0.01);
Yb/Lu gets worse.

**Diagnosis after attempt 4** (`scripts/g21_diag_a4.py`). HBK@ML's posterior variance ranks the
realised error with Spearman ≈ 0.1 (≈ 0.15 if perfectly calibrated), but it is inverted between pair
types: the heavy pairs get the largest posterior variance and have the smallest error. A read policy
that minimises that variance (DOPT_HB) would steer reads toward the heavy end, so it was deferred
**before** running it; the code is in `g21/policies.py`, not run, and its gain is untested.

**Attempt 5.** The training data are strongly heteroscedastic: about 40 % of cells fall in a component
with about ten times the white-residual variance (a model residual, not a measured reading noise),
and the training likelihood rises by a median of 524. Held out, the gain is nothing measurable. The
four reads do identify a cell's regime (AUC 0.86 against the full-curve label), but the
regime-specific shrinkage does not lower the error: in correctly identified noisy cells, MSE is
0.0564 against 0.0566. Those noisy cells are 37.5 % of the held-out C3 cells and carry 79 % of the
squared error.

**Attempt 6.** Motivated by HBK@ML's per-pair offsets at the half-shell (mean residual Eu/Gd
+0.087, Gd/Tb −0.102, in-sample on the dev rows later used to score attempt 6;
`scripts/g21_diag_a6.py`). The fitted metal effects contain a **gadolinium break** (α_Gd = −0.096 ±
0.020, negative in every fold). It lowers pooled MAE at Eu/Gd (0.150 → 0.124) and Gd/Tb (0.129 →
0.104), but only on the non-DGA chemotypes (Eu/Gd 0.173 → 0.129, Gd/Tb 0.154 → 0.105, every seed). On
held-out DGA (half the C3 rows) the offset learned without DGA gives no gain (Gd/Tb 0.101 → 0.102) and
flips the sign of the Gd residual. The other effects (α_Tm −0.03 to +0.11 across folds, α_Lu) are
poorly estimated and damage the low-variance heavy pairs (Tm/Yb MAE 0.060 → 0.091, Er/Tm 0.068 →
0.097). Net: worse pooled MAE in every seed. All of that loss comes from the DGA hold-out fold that
each seed contains (DGA 0.0993 → 0.1085; other chemotypes 0.1113 → 0.1098). HBKM[mean+noise] stopped
at its iteration cap in 18/25 folds; refits to convergence change nothing material.

## 4. The candidate: HBK@ML, every budget (`results/attempt4/levels_{BP,BPG}.csv`)

Same rows for every k within a cohort.

| cohort | k (metals read) | R² BP [95 %] | per-seed R² BP | MAE BP | R² BPG | HB2+GP R² BP | RIDGE2 R² BP | LINE R² BP |
|---|---|---|---|---|---|---|---|---|
| C3 | 0 (zero-shot: training mean shape) | −0.118 | — | 0.1683 | −0.116 | −0.116 | −0.116 | −0.135 |
| C3 | 1 (2) | 0.099 | — | 0.1457 | 0.101 | 0.076 | −0.008 | **0.168** |
| C3 | 2 (3) | 0.344 | — | 0.1151 | 0.344 | 0.334 | 0.279 | 0.237 |
| C3 | **3 (4)** | **0.414 [0.321, 0.500]** | 0.404, 0.425, 0.403, 0.427, 0.412 | 0.1054 | **0.419** [0.325, 0.503] | 0.403 | 0.310 | 0.236 |
| C6 | 3 (4) | 0.392 [0.307, 0.493] | — | 0.1105 | 0.395 | 0.376 | 0.303 | 0.224 |
| C6 | 5 (6) | 0.499 [0.436, 0.589] | 0.486, 0.507, 0.499, 0.500, 0.500 | 0.0974 | 0.497 | 0.489 | 0.326 | 0.230 |
| C6 | 6 (7) | 0.533 [0.477, 0.615] | 0.508, 0.527, 0.547, 0.556, 0.526 | 0.0928 | 0.531 | 0.521 | 0.323 | 0.228 |

**At k = 1 (two metals read) every round-2 model is below LINE**: HBK@ML 0.099, HBMIX 0.095,
HBKM@BIC 0.105, against LINE 0.168 (BPG 0.157). From k = 2 on, HBK@ML is the best arm.

Per pair type at k = 3 (BP, R², HBK@ML / RIDGE2): La/Ce 0.35 / 0.29, Ce/Pr 0.42 / 0.31, Pr/Nd 0.37 /
0.33, Sm/Eu 0.42 / 0.40, Eu/Gd 0.29 / 0.13, Gd/Tb 0.35 / −0.03, Tb/Dy 0.34 / 0.36, Dy/Ho 0.48 / 0.40,
Ho/Er 0.22 / 0.09, Er/Tm 0.21 / −0.14, Tm/Yb 0.01 / −0.31, Yb/Lu −0.14 / −0.11. Against RIDGE2 on
identical rows: ΔR² +0.104 [+0.078, +0.123], ΔMAE +0.0152, 5/5 seeds. Zero-shot (k = 0) it is no
better than FLAT. The structure model G14 (0.053) remains the best zero-shot reference.

## 5. Leakage and validity checks (round 2)

* **Intervention** on all five dev seeds, every fold, both designs: every metal not read at budget k,
  in every held-out cell, is replaced by a random number with reads and masks unchanged. Max change of
  any prediction: **0**, NaN pattern unchanged (610,400 / 427,280 / 549,360 predictions per design for
  attempts 4 / 5 / 6).
* **Training-only fits:** poisoning the held-out cells' log D leaves every round-2 fitted quantity
  unchanged (kernels and the ML kernel choice, the mixture, the metal effects and the BIC choice;
  `tests/test_g21.py`). HBK[exp] equals round 1's HB2+GP exactly (test). 14 tests pass.
* Fold isolation, no scored pair with both ends read, and no use of held-out labels by any policy:
  as in round 1, re-checked on every fold.
* **Dev looks disclosed** (`ATTEMPTS.md`): the mixture was fitted and scored on one dev fold while it
  was being written; attempt 6's metal effects were fitted, but not scored, on one training fold. The
  order of ideas changed after attempt 4's diagnosis: DOPT_HB, planned as attempt 5, was deferred,
  and the mixture moved up to 5. Attempt 6 (metal effects, from attempt 4's dev per-pair offsets)
  was written and launched in parallel with attempt 5, before attempt 5 had a result.
* **Review corrections** (`ATTEMPTS.md`, wording marked "after the review"): the attempt-4
  diagnosis, attempt 5's regime-identification claim, the Gd-break transfer claim (DGA
  over-corrected), the HBKM[mean+noise] iteration cap, and the LINE comparison at k = 1.

## 6. Status against the success criterion

* **Not confirmed.** Only dev runs exist, and the withheld seeds are unused.
* **Dev evidence for HBK@ML at the protocol default (C3, k = 3).** 0.414 (BP) / 0.419 (BPG), every seed
  0.402–0.428. MAE beats RIDGE2 and HB2+GP. The chemotype-bootstrap lower bound is 0.32: new chemotypes
  could fall well below 0.40. At C6 k = 5 it is 0.499 (lower bound 0.44). That would be a post-dev change
  of the primary cohort (amendment A2).
* **Selection.** Six ideas and several variants were compared on the same dev seeds, so the best dev
  number is optimistically selected. The new seeds of the final check re-partition the same 521 cells
  (A3): they guard against split-specific luck, not against selection on this corpus.

## 7. Limitations (in addition to round 1's)

* The round-2 gain is small (+0.011 R²) and inside the chemotype-level uncertainty.
* About 40 % of training cells look noisy: per-metal sd ≈ 0.18 under the mixture, but that is a
  model residual, not a measured noise. They carry most of the remaining error. Whether that is an
  irreducible ceiling is untested; replicate whole-curve measurements would settle it.
* At k = 1 the hierarchical models are still below a straight line through the two reads.
* Heavy pairs Tm/Yb and Yb/Lu stay near or below zero R² at k = 3.

## 8. Options for the next round (decision for the user)

1. **Freeze HBK@ML and run the final check once.** Default regime C3, k = 3, DOPT_Q reads; BP + BPG,
   reported as it comes out.
2. **More ideas on dev first**, ranked by the evidence:
   (a) the **gadolinium break alone** (one shared Gd offset), or metal effects shrunk with an
   ML-estimated variance. Attempt 6 showed the Gd component helps the non-DGA chemotypes and the
   rest over-fits. One shared Gd offset learned without DGA is expected to over-correct DGA cells
   (about half of C3 rows) unless its size is shrunk or varies by chemotype; report DGA and non-DGA
   separately;
   (b) DOPT_HB reads (deferred; gain untested, and the variance inversion between pair types argues
   against it in its present form);
   (c) more reads per condition only where k ≥ 5 is affordable (C6 numbers).
3. **Data that would help most:** replicate whole-curve measurements (the noise ceiling) and more
   non-DGA chemotypes.
