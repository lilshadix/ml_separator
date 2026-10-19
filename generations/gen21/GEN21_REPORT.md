# gen21 — round 1 report: three ideas for adjacent-lanthanide log SF

> **Round 2 (2026-10-19) has superseded this report's candidate**: see `GEN21_ROUND2_REPORT.md`
> (new candidate HBK@ML, dev R² 0.414 BP / 0.419 BPG). This file is kept as written for round 1.

*2026-10-18. The user limited this round to three ideas. Every number comes from a file under
`results/` (paths given per table) computed from the row-level predictions in `predictions/`. All
results are **development (dev)** results. The final check on the withheld seeds has **not** been
run (§8).*

## 1. Goal and what counts

Task (`CLAUDE_TASK.md`): R² ≥ 0.40 (preferably ≥ 0.50) for `logSF(A/B) = log10 D(A) − log10 D(B)`,
A the lighter lanthanide, on **strict-adjacent** pairs (|ΔZ| = 1; Nd/Sm is not adjacent) of
held-out experimental cells. Zero-shot and every measurement budget count separately. A pair
whose two metals were both read in the same cell is never scored. A high log D R² does not count.

## 2. Setup (details: `PROTOCOL.md`, written before any new method was scored)

* **Data.** Frozen gen13 cohort through gen20's loader: 521 cells, 90 extractants, 45 chemotypes,
  58 publications (manifest SHA-256 `3f11ac4c…`, log D matrix `f0be7fb3…`).
* **Measurement model.** gen20's metal model. In a held-out cell, k + 1 metals are read (log D
  known); the read set never contains both metals of an adjacent pair, so every query has at most
  one read end. Reads are chosen by a label-free policy (gen20's DOPT_Q unless stated).
* **Rows.** Common-query cohort C3: cells with k_max ≥ 3, 229 cells, 79 extractants, 38
  chemotypes, **1,802 strict-adjacent pairs per seed**. Every k ≤ 3 is scored on identical rows.
  C6 (k_max ≥ 6): 133 cells, 1,349 pairs per seed, for k = 5, 6.
* **Hold-out.** BP: chemotype-grouped 5-fold, held-out cells' publications removed from training.
  BPG (strict copy check): BP plus removal of every training cell sharing a cross-publication copy
  group with a held-out cell. Dev = gen20's five discovery seeds. Every free parameter is fitted
  on the outer training fold only: by inner 4-fold chemotype cross-fit (attempt 1) or by maximum
  likelihood on training curves (attempts 2–3).
* **Metric.** Pooled R² per seed, averaged over seeds (gen20's convention), plus MAE, RMSE and
  extractant-macro MAE. Intervals are nominal-95 % percentile chemotype-block bootstraps (2,000
  draws). With about four effective chemotypes their real coverage is nearer 90 % (gen20's
  inference audit); BCa bounds and p values are in the contrast files. Paired contrasts are taken
  on identical rows and supports, signed so that positive favours the candidate (ΔR² = cand − ref;
  ΔMAE = ref − cand). "x/5 seeds" counts re-partitions of the same cells: it measures partition
  stability, not independent replication.

## 3. Reproduction of the reference

gen20's RIDGE2 (two-parameter radius curve shrunk to the training prior) at k = 3, metal model,
DOPT_Q, BP, C3, rebuilt from gen20's own functions (`results/reproduction_gen20_BP_C3_DOPT_Q.csv`):
**R² 0.3099, MAE 0.1206, macro MAE 0.1346** — identical to four decimals with gen20's
`levels_BP_metal_C3_DOPT_Q.csv`, and so is every other arm (LINE 0.2356, G14@BLUP20 0.2482,
FLAT@BLUP20 0.1890, zero-shot G14 0.0529, FLAT −0.1347). This is a **k = 3** result (four metals
read per cell), not zero-shot and not a log D result. gen20.1 was **not** re-run. Its published
zero-shot numbers on the same 1,802 C3 rows are log SF −0.010 (from log D) / −0.016 (direct) and
log D 0.258 (`generations/gen20_1/GEN20_1_REPORT.md`). For k = 0, zero-shot G14 (0.053) is the
stronger reference and is the one used here.

## 4. All attempts (k = 3, C3, 229 cells / 1,802 pairs per seed unless stated)

| # | hypothesis (one line) | method | regime, budget | R² logSF BP [95 %] | MAE | RMSE | log D of unread metals R² / MAE / RMSE | baseline R² (RIDGE2, same reads) | R² BPG (strict copies) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | — (reference) | RIDGE2 (gen20) | k = 3, DOPT_Q | 0.310 [0.221, 0.401] | 0.1206 | 0.1844 | 0.977 / 0.173 / 0.249 | — | 0.323 | reproduced |
| 1 | the scored target shares the read endpoint's reading error | RIDGE2+OBS@RESEL | k = 3, DOPT_Q | 0.357 [0.295, 0.433] | 0.1183 | 0.1780 | 0.980 / 0.161 / 0.232 | 0.310 | 0.358 | R² 5/5 seeds, MAE only 3/5: not advanced |
| 2 | the rest is curve shape + mis-set shrinkage | **HB2+GP** (ML hierarchical curve + Z-correlated residual) | k = 3, DOPT_Q | **0.403** [0.318, 0.491] | 0.1079 | 0.1714 | 0.982 / 0.149 / 0.221 | 0.310 | **0.407** | R² and MAE better in 5/5 seeds, both designs: **chosen candidate** |
| 3 | the error shared inside a condition series is a series shape; pool the series' reads | HBG@POOL, staggered reads | setting P, k = 3 per cell, SERIES_STAGGER | 0.341 [0.211, 0.463] | 0.1103 | 0.1801 | 0.984 / 0.145 / 0.207 | 0.292 | 0.337 | worse than own reads (−0.041): **refuted** |
| 3d | control: pooled, identical reads | HBG@POOL | setting P, k = 3, DOPT_Q | 0.406 [0.311, 0.489] | 0.1078 | 0.1711 | 0.983 / 0.145 / 0.216 | 0.310 | 0.407 | = own-reads HB2+GP (+0.002 [−0.010, +0.011]) |
| — | zero-shot reference | G14 (gen14 structure model) | k = 0 | 0.053 | 0.1469 | 0.2160 | not predicted | — | 0.053 | — |

log D of unread metals is the method's absolute level, fixed by the reads. Its R² is about 0.98
for every method because cells differ in level by orders of magnitude. The "mean of the reads"
gets 0.823 / 0.508 / 0.690. **These numbers say nothing about separation.** Sources:
`results/attempt{1,2,3,3d}/levels_*.csv`, `contrasts_*_C3.csv`, `logd_unread_*.csv`.

## 5. The three ideas in short (full entries: `ATTEMPTS.md`)

**Attempt 1 — observable, not latent.** In the gen20 arms the target is the latent curve
difference, but the scored value is the *observed* difference, and its read end is already in the
support. Adding the posterior mean of that reading's noise is parameter-free under gen20's noise
model. It helped only after its noise level and λ were re-chosen inside the training fold:
ΔR² +0.047 [+0.015, +0.090] (BP), +0.035 (BPG), 5/5 seeds. On BP that is +0.013 from the
re-selection alone (latent control RIDGE2@RESEL) plus +0.034 from the observable rule; on BPG it
is almost all observable rule. The MAE gain (+0.002) was not consistent (3/5 seeds). Diagnosis for the next step: half-shell and heavy-end pairs are where the quadratic
curve fails.

**Attempt 2 — hierarchical empirical-Bayes curve (HB).** `y = μ1 + Bᵀθ + δ + ε`, fitted by
chemotype-balanced weighted maximum likelihood on the training cells' curves. It has shape
coefficients θ with a full prior covariance, a Z-correlated residual δ (Cov τ²ρ^|ΔZ|) and reading
noise ε. The query prediction is the conditional mean of the observed difference given the read
contrasts. Against RIDGE2 on identical rows: ΔR² **+0.094 [+0.076, +0.112]** (BP) / +0.085
[+0.069, +0.116] (BPG); ΔMAE **+0.0127 [+0.0098, +0.0176]** / +0.0115. Both improve in 5/5 seeds
under both designs, and the method also beats attempt 1 in 5/5 seeds (ΔR² +0.047, ΔMAE +0.010).
The residual δ does the work: without it the model is 0.348. A Gd hinge term adds nothing. The
observable/latent distinction disappears, because ML puts the independent per-metal noise at
σ² ≈ 0–0.01 across folds (median 0.006). HB2+GP is chosen by parsimony among variants within
±0.002 R²; HB3H+GP@LATENT is nominally ahead, and the latent variants have MAE lower by 0.0008.

**Attempt 3 — pool a held-out condition series (setting P).** Diagnosis after attempt 2: 48 % of
the remaining squared error in series with ≥ 2 held-out cells repeats across the series'
conditions. An extractant × publication hierarchy pooled every condition's own reads. At k = 3 it
was neutral with identical DOPT_Q reads (+0.002; small gains at k = 1–2) and worse with staggered
reads (−0.041 [−0.106, +0.010] against the same model on own reads; −0.07 on cells that have
siblings). The staggered policy was introduced after a one-fold dev look at DOPT_Q pooling (HBG@POOL
0.359 against 0.397 for own reads on BP 130363 fold 2), and the originally planned DOPT_Q design
was kept as 3d. On synthetic series drawn from the model the same code gains 19 %. So the shared
error is not a stable series shape at the unread metals; by inference from the pooled loss, the
model's cross-condition variance as fitted is still too small for the real series. Seven
within-series duplicate cell pairs (14 cells, 108 pairs per seed) flatter pooling (0.245 against
0.187 on those rows). Without them the staggered loss is unchanged (0.341 against 0.384).

## 6. The candidate: HB2+GP (attempt 2), every budget

`results/attempt2/levels_{BP,BPG}.csv`, `per_pair_BP_C3_k3.csv`. Same rows for every k in a cohort.

| cohort | k (metals read) | HB2+GP R² BP [95 %] | per-seed R² BP | MAE | R² BPG | RIDGE2 R² BP | LINE R² BP |
|---|---|---|---|---|---|---|---|
| C3 | 0 (zero-shot: training mean shape) | −0.116 | — | 0.1679 | −0.115 | −0.116 | −0.135 |
| C3 | 1 (2) | 0.076 | — | 0.1484 | 0.080 | −0.008 | 0.168 |
| C3 | 2 (3) | 0.334 | — | 0.1173 | 0.333 | 0.279 | 0.237 |
| C3 | **3 (4)** | **0.403 [0.318, 0.491]** | 0.392, 0.416, 0.395, 0.412, 0.401 | 0.1079 | **0.407** | 0.310 | 0.236 |
| C6 | 3 (4) | 0.376 [0.298, 0.490] | — | 0.1135 | — | 0.303 | — |
| C6 | **5 (6)** | **0.489 [0.426, 0.588]** | 0.481, 0.500, 0.489, 0.492, 0.484 | 0.0988 | **0.488** [0.425, 0.584] | 0.326 | 0.230 |
| C6 | 6 (7) | 0.521 [0.460, 0.620] | 0.505, 0.518, 0.538, 0.536, 0.510 | 0.0942 | 0.520 | 0.323 | 0.228 |

Per pair type at k = 3 (BP, R², HB2+GP / RIDGE2): La/Ce 0.32 / 0.29, Ce/Pr 0.41 / 0.31, Pr/Nd
0.38 / 0.33, Sm/Eu 0.41 / 0.40, Eu/Gd 0.30 / 0.13, Gd/Tb 0.34 / −0.03, Tb/Dy 0.33 / 0.36, Dy/Ho
0.47 / 0.40, Ho/Er 0.18 / 0.09, Er/Tm 0.16 / −0.14, Tm/Yb −0.04 / −0.31, Yb/Lu −0.07 / −0.11.
The heavy pairs Tm/Yb and Yb/Lu (sd 0.12, FLAT MAE 0.07–0.08) remain at or below zero.
One BTBP series (15 cells) carries about a quarter of all squared error.

## 7. Leakage and validity checks (every attempt, both designs)

* **Fold isolation** (`evaluate.assert_test_isolation`, every fold): zero shared cells,
  extractants, chemotypes, publications (and copy groups under BPG) between training and held-out
  cells; inner blocks lie inside the outer training fold.
* **Intervention**, on all five dev seeds, every fold, both designs: for every budget k, the log D
  of every metal not read at k is replaced by a random number in **every** held-out cell of the fold
  (siblings included), with reads and masks unchanged. Max change of any prediction: **0**, with the
  NaN pattern unchanged (671,440 / 793,520 / 549,360 / 549,360 predictions per design for attempts 1 / 2 / 3 / 3d; `results/attempt*/summary.json`). The intervention reuses
  the fitted parameters that produced the rows. That fits never see held-out labels is tested
  separately, below.
* **No scored pair has both ends read** (`lab_known` false on every row). The sibling version
  (`sib_pair_read`) is false by construction, so it checks nothing. The check that matters in
  setting P is `dup_sibling`: 7 within-series cell pairs are exact offset copies of one measurement
  series, and their rows are scored separately (§5).
* **Training-only fits:** poisoning the held-out cells' log D leaves every fitted quantity
  unchanged: attempt 1's inner re-selection, attempt 2's HB fits, attempt 3's HBG and HB2+GP fits
  (tests). An independent review also rebuilt the cache with every held-out label randomised and
  found 0 differences. RIDGE2's Bayesian form equals gen20's ridge line to 1e-13 (test). The
  SERIES_STAGGER policy does not move when labels change (test). `tests/test_g21.py`: 10 tests pass.
* **Independent re-derivation:** a reviewer re-implemented the HB conditional mean from the saved
  per-fold parameters, without the project code. It matched the stored predictions to < 2e-13 on
  2,100 rows; RIDGE2 matched exactly. All HB fits report success. In one fold (196613/2) L-BFGS
  stopped on the flat τ² → ∞, ρ → 1 ridge: a polish gains 0.06 in log-likelihood and moves
  predictions by at most 0.002, so that fold's τ²/ρ are an optimiser artefact.
* **Provenance:** prior, λ, noise and the G14 covariance come from gen20's own fold-level
  selection; the HB parameters per fold are in `results/attempt2/fitted_*.csv`.
* **Reproducibility:** attempts 1 and 2, first run on earlier code, were re-run with the final
  code. Every level, including the intervals, is identical (difference 0.0). The cache and input
  digests are in `manifests/inputs_sha256.json`.

## 8. Status against the success criterion

* **Not confirmed.** The criterion needs a method frozen on dev, then **one** final check on held-out
  data that clears R² ≥ 0.40 in the declared regime, beats the comparable baseline on MAE, holds
  under the strict copy check and passes the leakage checks. Only dev runs exist. The withheld
  final seeds (commitment `30ec7c0d…`) are unused.
* **Dev evidence.** At **k = 3 (C3)**, the protocol's primary cohort, HB2+GP sits exactly at the
  threshold: 0.403 BP / 0.407 BPG, with 3 of 5 seeds ≥ 0.40 under BP and 4 of 5 under BPG. The
  chemotype-bootstrap lower bound, 0.32, says how far the result could fall on new chemotypes.
  At **k = 5 (C6, six metals read)** it is 0.489 / 0.488, every seed 0.48–0.50, lower bound 0.43;
  at k = 6 it is 0.52. Those are more measurement-heavy regimes on a narrower cohort (133 cells).
  Declaring one of them for the final check would **replace the pre-registered primary cohort
  after these dev numbers were seen**. It must be logged as a dated amendment before any final
  run (`ATTEMPTS.md`, amendment A2); the default under the protocol as written is C3 at k = 3.
* **What the final check can and cannot do** (amendment A3). It re-partitions the same 521 cells
  with new seeds, so it guards only against split-specific over-fitting. It cannot confirm a
  regime or variant chosen on these cells. Its expected seed-to-seed spread is about ±0.01 on a
  5-seed mean (dev C3 seeds 0.39–0.42), not the chemotype-bootstrap interval. At k = 3 a final
  result below 0.40 is therefore quite possible, while at C6 k = 5 a pass is close to certain and
  would say little.
* **What dev cannot rule out:** the dev seeds are gen20's discovery seeds, and the gen20 baseline
  numbers on them were known when the methods were chosen. Three ideas and several variants were
  compared on dev, so the best dev number is optimistically selected. New split seeds re-partition
  the same 521 cells; they are not new laboratory data.

## 9. Limitations and transfer

* About four effective chemotypes behind every pooled number; the diglycolamide chemotype holds about
  half the pairs; one BTBP series carries a quarter of the squared error.
* **Cells were selected by data completeness.** Only cells with enough measured metals that k + 1
  reads leave every adjacent query open are scored: C3 = 229 of the 331 cells that have an adjacent
  pair (79 extractants, 38 chemotypes); C6 = 133 cells (58 extractants, 27 chemotypes). The results
  do not cover sparsely measured systems.
* Seven within-series cell pairs are exact offset copies of one measurement series (one DGA paper,
  one pyridine-dicarboxamide paper). They are flagged and scored separately for setting P; the
  own-reads methods cannot use them.
* All held-out chemistry is from the same literature corpus; "unseen" means chemotype and publication
  hold-out, not a new laboratory.
* The model needs k + 1 real measurements of the new system. Zero-shot it is no better than FLAT
  (−0.12), and at k = 1 it is below the line (0.08 against 0.17).
* The independent per-metal noise and the short-range residual are not separately identified by ML
  (σ² ranges 0–0.01 across folds), so the attainable R² ceiling for adjacent log SF on this corpus
  is still unknown; replicate pairs that would measure it directly are too few (gen20.1).
* The heavy pairs (Tm/Yb, Yb/Lu) stay unpredictable at k = 3: their spread is close to the noise.

## 10. Options for the next round (decision for the user)

1. **Freeze and run the final check once.** Freeze HB2+GP (DOPT_Q reads, ML on training curves)
   and declare the regime. The protocol default is k = 3 on C3. k = 5 on C6 would be a post-dev
   change of the primary cohort and must be logged as such (amendment A2). Then run the withheld
   seeds once, BP + BPG, and report whatever comes out, keeping in mind that new seeds
   re-partition the same cells (A3).
2. **More ideas on dev first**, ranked by what the errors say:
   (a) a read policy chosen for the HB model (D-optimal on the HB covariance for the observable
   queries). Read choice matters for this model: staggered/UNIFORM reads cost −0.017 at k = 3 but
   gave +0.10 at k = 1;
   (b) a robust / heteroscedastic hierarchy (Student-t or per-chemotype variance components), aimed
   at the BTBP series and the decade-level replicate errors;
   (c) a chemistry-conditioned prior mean (structure-predicted shape) to lift k = 0–1;
   (d) condition covariates (acid, diluent) inside a series, the lesson of attempt 3.
3. **Data that would help most:** replicate measurements of whole curves (to measure the log SF
   noise ceiling) and more N-donor / non-DGA chemotypes (to raise the effective number of families).
