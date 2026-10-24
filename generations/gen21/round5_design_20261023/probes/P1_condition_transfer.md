# P1: condition transfer (idea M01, estimand E-CT), exploratory probe on the round-4 rows

Written 2026-10-23. Probe P1 of the round-5 design, idea M01 in `ideas/IDEAS_RANKED.md`.

> **Status: EXPLORATORY.** Every number below comes from the 229 C3 cells and the 5 dev seeds, which have already been seen. The population, the strata and K were fixed after earlier analyses of these same rows (E6 §6.8, the M01 skeptics). Nothing here is sealed or confirmatory.
>
> E-CT is a separate estimand: a known extractant, read at a new condition in the same publication. **None of these R² values may be placed next to the BP/BPG 0.452.**

---

## 1. Bottom line

1. **The same-campaign gain reproduces and survives V2 cleaning.**
   - On the primary stratum, identical rows, BPG: SH (K = 1.78) scores **0.708** and B1 at 4 reads scores **0.531**, so ΔR² = **+0.177**.
   - The primary stratum is V2-clean, n_sib ≥ 2, without C5-BTBP: 99 cells, 12 publications, 14 publication × chemotype units, 690 rows per seed.
   - 13 of 14 units improve.
   - Sign-flip p: 0.0004 with units, 0.0156 with chemotype blocks (the minimum attainable with 6 blocks).
   - Leave-one-publication-out (LOPO) jackknife 95 % interval: **[+0.076, +0.279]**. The LOPO minimum is +0.137, without pub_b0b3296ec5.
   - BP gives the same picture: +0.174, [+0.067, +0.281].
2. **SH − OFF = +0.0119 R²** (BPG, primary stratum), with relative SSE +3.9 %.
   - LOPO jackknife 95 % interval **[−0.017, +0.041]**. Unit sign-flip p 0.44.
   - The whole difference sits in one publication: without pub_19783c039b it is **−0.0011**. BP gives +0.0105 [−0.014, +0.035].
   - With C5-BTBP it is +0.0065 [−0.012, +0.025] at relative SSE +1.5 %, which is below both narrowing thresholds.
3. **Structural finding: OFF and SH are the same estimator, not a cheaper rival.**
   - When a sibling is a C3 cell read at the same metals in the same fold, its B1 pair-type residual equals SIB_j − B1 at that pair exactly, so SH ≡ OFF.
   - This holds in **72.9 % of primary rows**, where max |SH − OFF| = 7e-16. In 10 of the 14 units the contribution to SH − OFF is exactly 0.
   - SH and OFF differ only where siblings were read at other metals: 27 % of rows, 48 cells, 5 publications. On those rows SH scores 0.767, OFF 0.724 and B1 0.566.
   - So "SH − OFF" measures **read-geometry alignment**, not "template shape against source-local offset".
   - Kill rule (i) of M01 cannot discriminate the mechanism it was written for. "OFF recovers about 95 % of the gain" is true and uninformative.
4. **The information has to be the same extractant in the same publication.**
   - Placebo blends at the same w on their own rows:
     - same chemotype, other publications: −1.3 % relative SSE against B1 (85 cells, 10 publications);
     - same extractant, other publications: −2.6 % (53 cells, 6 publications).
   - Other extractants of the same publication give +13.6 % against SH's +30.0 % on the same rows (BPG). That is about 46 % of the R² gain, from only 3 publications and 25 cells. The offset version, OFF_XEXT, gives the same result: 0.760 against SH 0.805 and B1 0.721.
   - So part of the gain is a publication or campaign fingerprint. With 3 publications, its size cannot be separated from "extractant transfer".
5. **Shrinkage carries the one-sibling case; averaging carries the raw template.**
   - Within-publication subsampling (n ≥ 5, without C5; 62 cells, 5 publications, B1 0.458) at m = 1, 2, 3, 5, all:
     - raw template: 0.338 / 0.515 / 0.574 / 0.622 / 0.641;
     - SH: 0.546 / 0.586 / 0.609 / 0.634 / 0.652.
   - Harm rule (n_sib = 1, 16 cells, 8 publications): SH 0.603 against B1 0.547, relative SSE **+12.2 %**. No harm.
6. **Read ladder on the primary stratum, identical rows:**

   | reads | 0 | 2 | 3 | 4 |
   |---|---|---|---|---|
   | SH | 0.485 | 0.616 | 0.679 | 0.708 |
   | B1 at the same reads | – | 0.143 | 0.467 | 0.531 |

   - At 0 reads the arm is the COPY template, 0.497. It is **below** B1 with 4 reads by −0.034, and only 4 of 14 units are better.
   - SH at 2 reads beats B1 at 4 reads by +0.085, but only 8 of 14 units are positive, the equal-weight per-publication relative SSE is −2.1 %, and the top-3 share is 0.99. That is a lead carried by 3 publications.
   - SH at 3 reads is solid: +0.148 over B1 at 4 reads, 12 of 14 units positive.
7. **The artefact flag is not triggered.** Ce pairs carry 32.8 % of the SH − B1 gain (flag at ≥ 50 %); Gd pairs carry 23.6 %.
   - Excluding Ce pairs: SH − B1 = +0.153. Excluding Gd pairs: +0.182. Excluding both: +0.149.
8. **Inference stays thin.**
   - The primary stratum has 3 cpg clusters, with g:pub_74428aeab2 holding 89 % of the rows (Kish 1.25). Cluster-level inference is **not estimable**.
   - The top-3 units carry 74 % of the gain. pub_b0b3296ec5 alone carries 41 %.

---

## 2. What was run

**Code**, in `generations/gen21/round5_design_20261023/probes/scripts/`:
- `P1_ct_build.py` builds every arm row by row;
- `P1_ct_analyze.py` produces every table;
- `P1_ct_diag.py` runs the SH ≡ OFF identity check and the publication-offset comparators.

Run them from `D:\ml_separator_gh` with `.venv\Scripts\python.exe -W ignore <script> --out <dir>`. Intermediate outputs (`P1_rows.parquet`, `P1_sibs.csv`, `P1_summary.json`, logs) went to a session scratch directory, not to the repository. Re-running the three scripts in order regenerates them; build takes 53 s, analysis 94 s, diagnostics 5 s.

**Rows.**
- Round-4 `REF_BPG.parquet` (primary) and `REF_BP.parquet` (sensitivity).
- Strict-adjacent pairs, k = 3 DOPT_Q (4 reads), seeds 104729, 130363, 155921, 196613 and 262147.
- Every arm is scored on these identical rows, including the 0-, 2- and 3-read rungs.
- R² follows round 4's convention: pooled within each seed, then the seed mean.

**Checks that passed.**
- B1 was recomputed from each fold's fitted Gd amplitude (`results/REF/fitted_<design>.json`). It matches the parquet's B1 to within 8.9e-16 on all 18,020 rows.
- y matches exactly.
- Read sets rebuilt from the gen21 fold cache (`cell["seqs"]["DOPT_Q"]`) match the parquet on every cell, seed and design.
- E6 §6.8 reproduces exactly (V1 population, 149 cells, 1,111 rows per seed): SIB 0.5734, SH (K = 2) 0.5946, B1 0.4489, read ladder 0.411 / 0.517 / 0.554 / 0.573.
- Skeptic B's V1 stratum (n ≥ 2, without C5) reproduces: 112 cells, SIB 0.675, SH 0.683, B1 0.510.
- Skeptic A's V2-clean count reproduces: 99 cells, B1 0.531, SH 0.708.
- Skeptic A's OFF figure reproduces as the publication-wide variant OFF_PUB_V1: 0.669 (they reported 0.665–0.673).

**Population.** It is built from metadata only; no label is used.
- A sibling of target t is any bench cell (all 521, not only C3) with the same `publication_id`, the same extractant SMILES and a finite V1 log D at every metal of t.
- CLEAN removes the round-3 V2-1 / V2-2 / V2-3 cells (DMDPhPDA copy B and the Me2-TODGA re-import; NaN-contact-time double entries; hidden-condition merges), both as targets and as siblings.
- CLEAN also removes siblings whose recorded `cond__` metadata differs from t only in contact time or metal concentration, or not at all. Phase ratio is not recorded in the bench, so a phase-ratio-only sibling would appear as "no recorded difference" and is caught by the same rule.
- After the V2 removal this extra rule dropped **0** further siblings. The 6 contact-time-only sibling pairs in V1 are exactly the 3 V2-2 duplicate pairs.
- The V2-5 floor readings were **not** removed. The design names only D1–D3.

**Arms.** All arms other than COPY use 4 reads unless stated. n is the number of CLEAN siblings; w = n/(n + K), K = 1.78.

| arm | definition |
|---|---|
| B1_k | PWL_R[y_S − u_S] + u, with u = β·1[Gd], the fold's fitted round-4 amplitude, on nested DOPT_Q sets seq[:k]: 2 / 3 / 4 reads |
| SIB | mean over siblings of Y_j + PWL_R[y_S − Y_j,S] (= T + PWL_R[y_S − T_S]) |
| COPY | T, at 0 reads |
| SH_k | w·SIB_k + (1 − w)·B1_k. At 0 reads: w·COPY + (1 − w)·G14@ZERO, the round-4 zero-shot column |
| OFF | B1 + Σ_j r_j,p / (n_p + K), with r = y − B1 (OOF) of the target's CLEAN siblings at pair type p, same seed and design. Leave-target-out; K pinned at 1.78 |
| OFF_ALL | as OFF, over every other CLEAN C3 cell of the same publication × extractant |
| OFF_PUB / OFF_XEXT | as OFF, over every other non-dropped C3 cell of the publication (any extractant) / over its other extractants only |
| NEAR | nearest-condition sibling. Distance = abs. diff. of log10 acid conc. + abs. diff. of log10 extractant conc. + 1[diluent one-hot differs] (concentrations floored at 1e-4 M; one side missing adds 1, both missing add 0). Exact ties averaged. NEARSH is its shrunk version (w = n_tied/(n_tied + K)) |
| RAND1 | exact expectation over one uniformly drawn sibling (squared error averaged over siblings), raw and shrunk (w = 1/(1 + K)) |
| SUB m | cells with n ≥ 5: exact enumeration of all m-subsets (a seeded random 5,000 when there are more), raw SIB_m and SH_m with w = m/(m + K) |
| PB_CHEM / PB_XEXT / PB_PUB | placebo template blended at the target's own w. PB_CHEM: every covering same-chemotype cell of other publications. PB_XEXT: same extractant, other publications. PB_PUB: other extractants, same publication. Scored only on rows where the pool exists, with B1 and SH on the same rows |

**Inference.**
- Unit = publication × chemotype. Per-unit contribution c_u = mean over seeds of ΔSSE_u,s / SST_s, so Σ c_u equals the pooled ΔR² exactly.
- Tests:
  - exact sign-flip over units, one-sided;
  - sign-flip with whole chemotypes flipped together (conservative);
  - IM-t on the per-unit relative SSE reduction ρ_u and on the log SSE ratio;
  - leave-one-publication-out jackknife, SE and t(P − 1) 95 % interval;
  - top-3 unit share, and the equal-weight per-publication relative SSE.

---

## 3. Populations (identical in BPG and BP)

| population | cells | publications | pub × chemotype units | rows / seed |
|---|---|---|---|---|
| V1, ≥ 1 sibling (E6) | 149 | 19 | 22 | 1,111 |
| CLEAN, ≥ 1 sibling | 130 | 18 | 21 | 968 |
| CLEAN, ≥ 1, without C5 | 115 | 17 | 20 | 820 |
| **PRIMARY: CLEAN, ≥ 2, without C5** | **99** | **12** | **14** | **690** |
| CLEAN, ≥ 2, with C5 | 114 | 13 | 15 | 838 |
| CLEAN, n_sib = 1 (no C5 cells) | 16 | 8 | 9 | 130 |

- cpg clusters in PRIMARY, rows per seed: g:pub_74428aeab2 615, g:pub_8415538291 60, g:pub_cd72e4a7bb 15 (Kish 1.25).
- With C5 included, g:pub_af7a1cd919 adds 148 rows.

---

## 4. Comparators on identical rows (PRIMARY, BPG; BP in brackets)

| arm | reads | R² (seed min–max) | MAE | rel. SSE vs B1 |
|---|---|---|---|---|
| B1 | 4 | 0.531 (0.511–0.552) [0.532] | 0.074 | 0 |
| B0 (PWL_R) | 4 | 0.487 [0.487] | 0.079 | −9.3 % |
| **SH, K = 1.78** | 4 | **0.708** (0.701–0.718) [0.706] | 0.055 | **+37.8 %** |
| SH, K = 2 (E6 value) | 4 | 0.707 | 0.055 | +37.5 % |
| SIB (raw mean template) | 4 | 0.699 [0.696] | 0.057 | +35.7 % |
| **OFF** (ext × pub pair-type offsets, K pinned) | 4 | **0.696** (0.692–0.702) [0.695] | 0.056 | +35.2 % |
| OFF_ALL | 4 | 0.697 [0.697] | 0.056 | +35.4 % |
| OFF_PUB (publication-wide) | 4 | 0.697 [0.697] | – | – |
| nearest-condition sibling, raw | 4 | 0.594 [0.587] | 0.058 | +13.5 % |
| nearest-condition sibling, shrunk | 4 | 0.667 [0.666] | 0.058 | +29.1 % |
| random single sibling, raw (exact expectation) | 4 | 0.473 [0.468] | – | −12.4 % |
| random single sibling, shrunk | 4 | 0.623 [0.622] | – | +19.6 % |
| COPY | 0 | 0.497 | 0.078 | −7.4 % |
| SH0 (COPY shrunk to G14@ZERO) | 0 | 0.485 | 0.079 | −9.9 % |
| G14@ZERO (zero-shot) | 0 | −0.103 | 0.121 | – |
| random single sibling, COPY | 0 | 0.134 | – | – |

- **OFF's K, a hindsight sensitivity.** OFF scores 0.689 / 0.693 / 0.697 / **0.696** / 0.691 / 0.678 at K = 0.25 / 0.5 / 1 / **1.78** / 3 / 5. K is not a lever.
- **Other strata, BPG.**

  | stratum | B1 | SH | SIB | OFF | COPY |
  |---|---|---|---|---|---|
  | CLEAN ≥ 2 with C5 | 0.408 | 0.581 | 0.573 | 0.574 | 0.406 |
  | CLEAN ≥ 1 without C5 | 0.550 | 0.684 | 0.649 | 0.676 | 0.473 |
  | CLEAN ≥ 1 | 0.451 | 0.596 | 0.571 | 0.591 | – |

---

## 5. SH − OFF, stated plainly

| stratum, design | SH − OFF ΔR² | LOPO jackknife 95 % | rel. SSE | units + / − / exactly 0 | sign-flip p (unit / chemotype) | LOPO min (without) |
|---|---|---|---|---|---|---|
| **PRIMARY, BPG** | **+0.0119** (seeds +0.0005 to +0.0244) | **[−0.0172, +0.0409]** | +3.9 % | 2 / 2 / 10 of 14 | 0.44 / 0.50 | −0.0011 (pub_19783c039b) |
| PRIMARY, BP | +0.0105 | [−0.0138, +0.0348] | +3.4 % | 3 / 1 / 10 | 0.31 / 0.50 | −0.0004 (pub_19783c039b) |
| with C5, BPG | +0.0065 | [−0.0119, +0.0249] | +1.5 % | 2 / 2 / 11 of 15 | 0.34 / 0.25 | −0.0004 |
| with C5, BP | +0.0057 | [−0.0101, +0.0216] | +1.4 % | 4 / 1 / 10 | 0.28 / 0.25 | −0.0001 |
| CLEAN ≥ 1 without C5, BPG | +0.0080 | [−0.0106, +0.0265] | +2.5 % | 3 / 2 / 15 of 20 | 0.34 / 0.25 | −0.0005 |

**Unit contributions to SH − OFF (PRIMARY, BPG).**
- pub_19783c039b/sc009: +0.0127.
- pub_0a8da089ad/sc009: −0.0007 (3 rows).
- pub_b0b3296ec5/sc009: −0.0002.
- pub_c8195b4ca7/sc009: about +0.0000002.
- The other 10 units: exactly 0.

**Why they coincide.**
- Write SH = B1 + Σ_j (SIB_j − B1)/(n + K) and OFF = B1 + Σ_j r_j,p/(n_p + K).
- For a sibling read at the same metals in the same fold, SIB_j − B1 at pair p equals sibling j's own B1 residual at p. This needs only the linearity of PWL_R and the shared Gd amplitude.
- In PRIMARY, every CLEAN sibling is itself a C3 cell (so n_p = n). In 72.9 % of rows (69.1 % under BP) every sibling was read at the target's metals in the same fold. There |SH − OFF| ≤ 7e-16.
- The remaining 187 rows per seed (48 cells; publications 19783c039b, b0b3296ec5, c8195b4ca7, f8a621ede4 and 0a8da089ad) have on average 3.06 siblings read at other metals. On those rows SH scores 0.767, OFF 0.724 and B1 0.566.

**Reading.**
- By the letter of M01 kill (i), "SH − OFF < +0.01 R² or < 2 % relative SSE", the primary point estimate is just above both thresholds (+0.0119 / +3.9 %). The with-C5 and n ≥ 1 versions are below.
- But the interval spans 0, and the excess is one publication.
- More fundamentally, the contrast compares two packagings of the same sibling information: residuals aligned to the target's read set (SH) against residuals at the siblings' own read sets (OFF).
- **Conclusion: neither "template-specific" nor "publication-offset regime" is identified by this contrast.**

**A comparator that does separate information sources.**
- OFF_XEXT and PB_PUB use only other extractants of the same publication.
- On their 25 cells (3 publications), OFF_XEXT scores 0.760, PB_PUB 0.760, SH 0.805 and B1 0.721. A publication-level fingerprint, with no same-extractant series, recovers about 47 % of the gain there.

---

## 6. Read ladder (identical 4-read scored rows; nested DOPT_Q sets within seed)

| stratum (BPG) | 0 reads: COPY / SH0 | 2 reads: B1 / SIB / SH | 3 reads: B1 / SIB / SH | 4 reads: B1 / SIB / SH / OFF |
|---|---|---|---|---|
| PRIMARY | 0.497 / 0.485 | 0.143 / 0.619 / 0.616 | 0.467 / 0.669 / 0.679 | 0.531 / 0.699 / 0.708 / 0.696 |
| with C5 | 0.406 / 0.402 | 0.112 / 0.526 / 0.526 | 0.366 / 0.558 / 0.566 | 0.408 / 0.573 / 0.581 / 0.574 |
| CLEAN ≥ 1 | 0.414 / 0.413 | 0.185 / 0.517 / 0.532 | 0.411 / 0.555 / 0.580 | 0.451 / 0.571 / 0.596 / 0.591 |
| n_sib = 1 | 0.370 / 0.382 | 0.368 / 0.419 / 0.495 | 0.514 / 0.485 / 0.580 | 0.547 / 0.510 / 0.603 / 0.603 |

- BP is within 0.01 of every BPG entry; for example, PRIMARY SH at 2 / 3 / 4 reads is 0.617 / 0.679 / 0.706.
- **M02 row: SH and SIB at 2 reads against B1 at 4 reads.**

  | arm (PRIMARY) | ΔR² vs B1 at 4 reads | units + | equal-weight pub rel. SSE | IM-t p | top-3 share |
  |---|---|---|---|---|---|
  | SH, 2 reads | +0.085 | 8 of 14 | −2.1 % | 0.65 | 0.99 |
  | SH, 3 reads | +0.148 | 12 of 14 | +30 % | 0.0005 | 0.75 |
  | COPY, 0 reads | −0.034 | 4 of 14 | – | – | – |

---

## 7. Within-publication sibling subsampling (cells with n_sib ≥ 5)

| m | 1 | 2 | 3 | 5 | all |
|---|---|---|---|---|---|
| raw template, without C5 (62 cells, 5 pubs, 415 rows/seed; B1 0.458, OFF 0.633) | 0.338 | 0.515 | 0.574 | 0.622 | 0.641 |
| SH, without C5 | 0.546 | 0.586 | 0.609 | 0.634 | 0.652 |
| raw template, with C5 (77 cells, 6 pubs; B1 0.344, OFF 0.515) | 0.096 | 0.329 | 0.406 | 0.468 | 0.515 |
| SH, with C5 | 0.409 | 0.445 | 0.468 | 0.494 | 0.523 |

- BP, without C5: raw 0.330 / 0.509 / 0.569 / 0.617; SH 0.544 / 0.583 / 0.605 / 0.630; B1 0.460.
- Publications with n ≥ 5, without C5 (cells): pub_1605e436d7 (7), pub_19783c039b (14), pub_5a68dc5665 (6), pub_b0b3296ec5 (16), pub_c8195b4ca7 (19).
- The with-C5 row matches the IDEAS figures (0.11 → 0.52; 0.416 → 0.531).
- Averaging is the mechanism for the raw template. A single raw sibling is worse than B1. With shrinkage, one sibling already gives +0.088 over B1 in dense-grid publications.

---

## 8. Between-cell n_sib strata (CLEAN, without C5; descriptive only) and the harm rule

| n_sib | cells | pubs | rows/seed | B1 | SH | SIB | OFF | rel. SSE SH vs B1 | rel. SSE SH vs OFF |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 16 | 8 | 130 | 0.547 | 0.603 | 0.510 | 0.603 | +12.2 % | +0.1 % |
| 2 | 11 | 5 | 85 | 0.605 | 0.744 | 0.734 | 0.744 | +34.9 % | +0.1 % |
| 3–4 | 26 | 4 | 190 | 0.681 | 0.845 | 0.840 | 0.845 | +51.3 % | 0.0 % |
| 5–7 | 26 | 3 | 233 | 0.403 | 0.561 | 0.537 | 0.528 | +26.0 % | +6.9 % |
| ≥ 8 | 36 | 3 | 182 | 0.495 | 0.768 | 0.776 | 0.771 | +54.0 % | −1.6 % |

**Harm rule (ii)** (SH not worse than B1 by more than 5 % relative SSE at n_sib = 1): **passes**, at +12.2 % (BP +12.2 %). The deployed rule does not need to depend on sibling count. The raw single template does lose: SIB 0.510 against B1 0.547.

---

## 9. Pair-type decomposition (PRIMARY, BPG; SSE per seed)

| pair | rows/seed | B1 | SH | OFF | share of SH − B1 gain | share of SH − OFF gain |
|---|---|---|---|---|---|---|
| La/Ce | 44 | 1.540 | 0.738 | 0.729 | 25.5 % | −4.7 % |
| Ce/Pr | 35 | 0.686 | 0.455 | 0.461 | 7.4 % | 2.8 % |
| Pr/Nd | 35 | 0.611 | 0.283 | 0.286 | 10.4 % | 1.2 % |
| Sm/Eu | 49 | 0.458 | 0.170 | 0.157 | 9.2 % | −6.1 % |
| Eu/Gd | 51 | 1.035 | 0.419 | 0.454 | 19.6 % | 16.6 % |
| Gd/Tb | 67 | 0.514 | 0.387 | 0.402 | 4.1 % | 7.2 % |
| Tb/Dy | 67 | 0.840 | 0.738 | 0.857 | 3.2 % | 56.1 % |
| Dy/Ho | 82 | 0.932 | 0.688 | 0.717 | 7.8 % | 13.5 % |
| Ho/Er | 80 | 0.746 | 0.673 | 0.662 | 2.3 % | −5.1 % |
| Er/Tm | 59 | 0.369 | 0.256 | 0.281 | 3.6 % | 12.0 % |
| Tm/Yb | 60 | 0.277 | 0.165 | 0.174 | 3.6 % | 4.3 % |
| Yb/Lu | 61 | 0.315 | 0.204 | 0.209 | 3.5 % | 2.1 % |

| rows | R² B1 / SH / OFF / SIB | SH − B1 | SH − OFF |
|---|---|---|---|
| all | 0.531 / 0.708 / 0.696 / 0.699 | +0.177 | +0.012 |
| excluding Ce pairs | 0.560 / 0.712 / 0.697 / 0.703 | +0.153 | +0.016 |
| excluding Gd pairs | 0.489 / 0.670 / 0.658 / 0.659 | +0.182 | +0.012 |
| excluding Ce and Gd pairs | 0.507 / 0.656 / 0.638 / 0.644 | +0.149 | +0.018 |

- **Artefact flag (iii):** the Ce-pair share is **0.328** (BP 0.335), below the 0.50 flag. The Gd-pair share is 0.236.
- With C5, the Ce share is 0.293 and the Gd share 0.304. Without Ce pairs SH − B1 is +0.199, so with C5 the Ce pairs dilute the gain.
- SH − OFF is concentrated on Tb/Dy (56 %), which is the read-geometry effect of §5.

---

## 10. By condition-change type (union of the CLEAN siblings' differing cond__ field families)

Contact time and metal concentration are ignored in the union. "acid_type(+conc)" means the acid type changes, with or without the concentration. "mixed" means any other combination.

| class (PRIMARY, BPG) | cells | pubs | rows/seed | B1 | SH | SIB | OFF | COPY | share of SH − B1 gain |
|---|---|---|---|---|---|---|---|---|---|
| acid concentration only | 23 | 5 | 219 | 0.451 | 0.655 | 0.641 | 0.655 | 0.327 | 36.9 % |
| acid type (± concentration) | 13 | 1 (b0b3296ec5) | 130 | 0.436 | 0.808 | 0.821 | 0.818 | 0.694 | 34.5 % |
| diluent only | 19 | 1 (5a68dc5665) | 133 | 0.924 | 0.966 | 0.966 | 0.966 | 0.696 | 1.7 % |
| extractant concentration only | 3 | 1 (cd72e4a7bb) | 15 | 0.780 | 0.905 | 0.925 | 0.905 | 0.595 | 1.8 % |
| mixed | 40 | 4 | 190 | 0.478 | 0.601 | 0.583 | 0.562 | 0.437 | 25.0 % |
| temperature only | 1 | 1 | 3 | 0.521 | 0.572 | 0.212 | 0.786 | 0.254 | 0.1 % |

- Diluent-only and extractant-concentration-only cells are reported separately, as the design asks: B1 already scores 0.92 and 0.78 there, and they add 3.5 % of the gain.
- The acid-type class is one publication. The acid-concentration-only class (5 publications) is the broadest real condition-transfer evidence.
- SH − OFF is non-zero only in "mixed" (+0.25 SSE per seed) and "acid type" (−0.03).
- With per-sibling multi-field counting (acid type plus concentration counted as mixed), "mixed" would hold about 60 % of the gain, close to the 83–93 % quoted in IDEAS. The difference is a definition.
- CLEAN n ≥ 1 without C5: acid concentration only (32 cells, 9 publications) B1 0.507 → SH 0.690, 42 % of the gain.

---

## 11. Placebo blends (same w as the target's own; rows where the placebo pool exists)

| placebo (PRIMARY, BPG) | coverage | cells / pubs | B1 | SH | OFF | placebo blend | placebo raw | blend vs B1 | SH − blend (LOPO 95 %) |
|---|---|---|---|---|---|---|---|---|---|
| same chemotype, other publications | 83 % | 85 / 10 | 0.501 | 0.688 | 0.675 | 0.495 | 0.473 | −1.3 % | +0.194 [+0.025, +0.362]; 9 of 11 units +; sign-flip p 0.004 |
| same extractant, other publications | 66 % | 53 / 6 | 0.470 | 0.679 | 0.662 | 0.457 | 0.418 | −2.6 % | – |
| other extractant, same publication | 28 % | 25 / 3 | 0.721 | 0.805 | 0.805 | 0.759 | 0.762 | +13.6 % | +0.045 [−0.108, +0.199]; 4 of 5 units +; p 0.06 |

- **With C5 included, the cross-publication placebos turn positive:** same chemotype +9.4 %, same extractant +10.8 % against B1.
- This happens because any BTBP-type template from another publication repairs part of C5-BTBP's large residuals. Without C5 they are null.

---

## 12. Inference summary (unit = publication × chemotype)

| contrast | stratum | ΔR² | units + / − (of G) | sign-flip p unit / chemotype | IM-t (ρ) p | LOPO 95 % | LOPO min (without) | top-3 share | eq-w pub rel. SSE |
|---|---|---|---|---|---|---|---|---|---|
| SH − B1 | PRIMARY BPG | +0.177 | 13 / 1 (14) | 0.0004 / 0.0156 | < 1e-4 | [+0.076, +0.279] | +0.137 (b0b3296ec5) | 0.74 | +38.8 % (11 of 12) |
| SH − B1 | PRIMARY BP | +0.174 | 13 / 1 | 0.0004 / 0.0156 | < 1e-4 | [+0.067, +0.281] | +0.132 | 0.74 | +38.4 % |
| SH − B1 | with C5 BPG | +0.173 | 14 / 1 (15) | 0.0002 / 0.0078 | < 1e-4 | [+0.124, +0.221] | +0.153 | 0.77 | +37.5 % |
| SH − B1 | CLEAN ≥ 1 without C5 BPG | +0.134 | 18 / 2 (20) | < 1e-4 / 0.002 | < 1e-4 | [+0.032, +0.237] | +0.097 | 0.72 | +36.6 % |
| OFF − B1 | PRIMARY BPG | +0.166 | 13 / 1 | 0.0003 / 0.0156 | < 1e-4 | [+0.045, +0.286] | +0.121 | 0.72 | +41.7 % |
| SIB − B1 | PRIMARY BPG | +0.168 | 12 / 2 | 0.0008 / 0.0156 | 0.0035 | [+0.045, +0.291] | +0.118 | 0.76 | +31.0 % |
| SH@3 reads − B1@4 | PRIMARY BPG | +0.148 | 12 / 2 | 0.0010 / 0.0156 | 0.0005 | [+0.055, +0.242] | +0.110 | 0.75 | +30.1 % |
| SH@2 reads − B1@4 | PRIMARY BPG | +0.085 | 8 / 6 | 0.081 / 0.063 | 0.65 | [−0.019, +0.188] | +0.043 | 0.99 | −2.1 % |
| COPY − B1@4 | PRIMARY BPG | −0.034 | 4 / 10 | 0.66 / 0.95 | 0.99 | [−0.235, +0.167] | −0.113 | – | −56.4 % |
| SH − NEAR | PRIMARY BPG | +0.114 | 7 / 7 | 0.13 / 0.50 | 0.61 | [−0.115, +0.343] | +0.019 (1605e436d7) | 1.08 | −0.4 % |
| SH − SIB (shrinkage) | PRIMARY BPG | +0.010 | 9 / 5 | 0.19 / 0.31 | 0.47 | [−0.015, +0.034] | +0.003 | 1.20 | +4.3 % |
| **SH − OFF** | **PRIMARY BPG** | **+0.012** | **2 / 2 / 10 × 0** | **0.44 / 0.50** | **0.80** | **[−0.017, +0.041]** | **−0.001 (19783c039b)** | – | −7.4 % |

- **Single-cell influence.** c950dc2982e96b83 is a C5-BTBP cell present in the with-C5 stratum. Removing it moves SH − B1 from +0.1725 to +0.1733 and SH − OFF from +0.0065 to +0.0080. It does not drive E-CT.
- **Cluster-level inference: not estimable.** PRIMARY has 3 cpg clusters (4 with C5), and one cluster holds 89 % of the rows.

---

## 13. M01 kill criteria and leads, status on the seen rows

| criterion | value now | status |
|---|---|---|
| (i) template claim narrowed if SH − OFF < +0.01 R² or < 2 % rel. SSE | +0.0119 / +3.9 % (BP +0.0105 / +3.4 %); with C5 +0.0065 / +1.5 % | Formally not triggered in PRIMARY, triggered with C5. **Invalid as a mechanism test**: SH ≡ OFF wherever the read sets match (§5) |
| (ii) harm rule at n_sib = 1 (≥ −5 % rel. SSE) | +12.2 % | passes |
| (iii) artefact flag (Ce share ≥ 50 %) | 32.8 % | not triggered |
| (iv) prospective kill (≥ 11 new lab × extractant families) | – | data-gated, not run |
| original (a)–(c) consistency checks: SH − B1 ≥ +0.05 with a positive lower bound; SH at 2 reads ≥ 0.50; other-publication template < 70 % of the gain | +0.177 with LOPO lower bound +0.076; SH at 2 reads 0.616; PB_XEXT −2.6 % | pass (known to pass; no information) |

---

## 14. What would change these conclusions

- **The mechanism question needs a different comparator, not OFF.**
  - Information-separating arms: OFF_XEXT / PB_PUB (campaign fingerprint without same-extractant series), COPY (0 reads) and the cross-publication placebos.
  - If a pre-declared OFF_XEXT, on more than 3 publications, recovered most of the SH − B1 gain, the claim would become "campaign or laboratory fingerprint (possibly shared artefact)", not extractant condition transfer. Today it recovers about 45–47 % on 25 cells from 3 publications.
  - Round 5 should replace kill (i) with SH − OFF_XEXT (or SH − PB_PUB) on rows where both exist, and report SH − OFF only as the read-alignment effect.
- **pub_b0b3296ec5** carries 41 % of the SH − B1 gain and is the only "acid type" publication. **pub_19783c039b** carries all of SH − OFF. A transcription or provenance problem in either would move the numbers. The SH − B1 LOPO minimum is still +0.137.
- **External leg (M05).** The decisive test needs independent laboratory × extractant families. With 3 cpg clusters (89 % in one), nothing here generalises beyond "the DGA-dominated giant cluster plus 2 small clusters".
- **Laboratory annotation** (DOIs to corresponding-author groups) was **not done**; it needs desk work outside this probe. Until it is done, read "same publication" as "same study or campaign", not "same laboratory".
- **Unrecorded conditions.** Phase ratio and other unrecorded variables cannot be audited from the bench.
- **Floor readings.** The V2-5 floor readings were kept. A sensitivity that drops them (and the siblings that lose coverage) was not run.
- **Placebo pools** use every available placebo cell, not a count matched to n. A matched-count placebo would be weaker, which strengthens the "same extractant, same publication" reading.
- **The 0-read shrinkage target** (G14@ZERO, R² −0.10) is poor. A better zero-shot target could lift SH0 above COPY (0.497). It would not change the finding that COPY is below B1 at 4 reads (−0.034).

---

## 15. Caveats

- **Exploratory and post hoc.** The stratum cut (n ≥ 2), the exclusion of C5-BTBP, K = 1.78 (from the roughness correlation r = 0.36 measured on these data) and the V2 lists were all fixed after earlier looks at these rows. Every pass above is a consistency check, not a test.
- **E-CT is not E-A.** These R² values are on a 99-cell subset with its own variance and must never be compared with 0.452.
- **Units are not independent.** Publications share laboratories. Unit-level p-values and LOPO intervals assume more independence than the 3 clusters support.
- The chemotype-block sign-flip p of 0.0156 is the floor of a 6-block test, not a measurement of strength.
- **Read-set dependence.** B1 and OFF depend on the fold's DOPT_Q read sets. BPG and BP differ slightly; both are reported.
- **M03 weighting factorial and M04 reference-condition checks** were not part of this probe and were not run.
