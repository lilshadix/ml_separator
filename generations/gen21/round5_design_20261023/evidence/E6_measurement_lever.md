# E6: the measurement and decision lever (round-5 design evidence)

Written 2026-10-23. Everything here is **exploratory development evidence**. It comes from existing out-of-fold (OOF) predictions and the bench (V1 data, the round-4 data contract). Nothing was sealed, registered or confirmed. No final seeds were touched.

The only model fitted is a small gradient-boosting scorer inside one read policy (§4, "LEARNED"). It was cross-fitted on the round-4 BPG folds. Every other predictor here is training-free:
- **PWL_R**: radius interpolation.
- **D7**: B1's functional form with the fixed full-corpus Gd potential of −0.072, i.e. round 4's `HIST_D7`. It equals B1 within 0.001 R². It is not fold-independent, so it is used only to compare *policies* on identical pairs, never as a candidate.

Measured facts are stated plainly. Interpretation is marked **(interp.)**.

---

## 0. Bottom line

1. **More metals is still the only lever that moves adjacent R² on the existing benchmark.**
   - The marginal value per extra read falls with k. On C6 (133 cells, 1,349 pairs, BPG), PWL_R under DOPT_Q gains +0.178, +0.051, +0.053, +0.042 and +0.031 R² for k = 1→2 … 5→6.
   - B1-form (D7) crosses 0.50 at **k = 5** (0.524). Under DOPT_Q, plain PWL_R and HBK cross it only at k = 6 (0.522 / 0.531; HBK is 0.497 at k = 5).
   - With every second metal read (k = 6), the realistic ceiling is ~0.55–0.59. The per-cell hindsight oracle is 0.66.
2. **Read placement has large hindsight headroom that no label-free or value-adaptive rule captures.** On C3 at k = 3 with D7:
   - random legal set 0.354; DOPT_Q 0.451;
   - best label-free or adaptive rule 0.457–0.462;
   - greedy sequential oracle 0.575; per-cell best set 0.632.
   - The learned value-conditioned adaptive policy gains +0.006 over DOPT_Q [−0.033, +0.019] at k = 3, and +0.035 [−0.009, +0.057] on C6 at k = 6. Adaptivity cannot turn k = 3 into 0.50.
3. **About a third of the adjacent-pair variance is short-range roughness that no interpolation reaches.**
   - The roughness is the deviation of a metal from its two neighbours. On the 78 fourteen-metal cells its variance is 0.0152 (0.0134 after removing the per-metal mean), against an adjacent log SF variance of 0.0432.
   - It correlates +0.36 between conditions of the same extractant in the same publication, and +0.14 between different extractants in the same publication.
   - It correlates −0.02 between publications for the same extractant (3 extractants only).
   - **(interp.)** It is largely lab- or series-specific, not transferable chemistry.
4. **Same-extractant literature values do not transfer at adjacent resolution.**
   - Copying an adjacent log SF from another publication's cell of the same extractant gives R² −0.08 (796 rows, 8 extractants; TODGA alone 0.05 over 418 rows).
   - Copying from the same publication at another condition gives 0.354.
5. **New, largest observed lever: the sibling-template mode.**
   - Setup: the same extractant has a full series at another condition in the same publication (65% of C3 cells, 62% of rows). That sibling curve is used as the interpolation template for the 4 reads.
   - Result: R² **0.573 vs B1 0.449** on identical rows (MAE 0.070 vs 0.091).
   - It survives three checks: excluding near-copy siblings (+0.122; still +0.054 when the residual shape sd ≥ 0.1), the non-dominant clusters (+0.081), and each large chemotype (+0.14 to +0.19).
   - This is a different regime from BP/BPG: it uses labels from the held-out cell's own publication.
6. **Reframings where existing models are already useful (B1, k = 3, BPG; BP within ±0.01):**
   - **Direction:** accuracy 0.775 (0.869 when |y| ≥ 0.1; 0.94 when the model answers only at |p| ≥ 0.2, 24% coverage). "Always heavier" scores 0.666, and zero-shot G14 0.688.
   - **Choosing among about 18 held-out candidates for a target pair:**
     - Spearman 0.62;
     - top-1 pick 0.28 against 0.06 at random;
     - regret 0.22 against 0.39 log units;
     - zero-shot G14 is at chance (0.07).
   - **Wider pairs from the same reads:** ΔZ = 2 reaches R² 0.68, Nd/Sm 0.76 and ΔZ = 3–4 0.81.
   - **Pair-targeted reads:** on 14-metal cells, two reads chosen around one target pair reach a pair-type R² of 0.34–0.87 (LOCO-chemotype choice). DOPT_Q at k = 3 reaches −0.03 to 0.80, and loses on every one of the 12 pair types.
7. **Calibrated intervals work marginally but not per source.**
   - Cross-fold Mondrian (pair-type) conformal intervals give 0.787 / 0.895 coverage at 80% / 90% nominal.
   - The half-widths are 0.15 / 0.25 log units, i.e. ×1.42 / ×1.76 on SF.
   - Clusters outside the dominant one are under-covered: 0.754 at 80% nominal. The per-cluster range is 0.52–1.00.
8. **Abstention does not buy R².**
   - The best label-free difficulty score (ensemble spread) has a within-pair-type Spearman of 0.18 with |error|.
   - Keeping the 50% "easiest" rows cuts MAE from 0.100 to 0.082, but R² falls to 0.42 because the retained variance shrinks.
   - R² ≥ 0.50 is reached only by restricting to the mid-series pairs Sm/Eu…Dy/Ho (0.504 on 46% of rows), which is a choice of target, not a confidence score.

---

## 1. Sources, definitions, reproduction

**Inputs read:**
- Round-4 OOF predictions:
  - `round4_20261022/predictions/R3-B0_{BPG,BP}.parquet`: DOPT_Q ladder k = 0..6 for PWL_R, PWL_Z, LINE, RIDGE2, HBK@ML, G14@ZERO, FLAT.
  - `REF_{BPG,BP}.parquet`: k = 3, B0/B1/B2/HIST_D7.
- Round-3 ladders: `next_research_20261019/results/R20/ladder_*.csv` (DOPT_Q, C6) and `R24/ladder_*.csv` (POL_ERR_K, C6).
- Round-3 `FINAL_REPORT_RU.md` §3 and §6.
- gen20 `GEN20_REPORT.md` §C.4–C.7.
- Memory note `project-gen15-curve-and-measured-mode.md` (gen15 measured mode).
- The bench via `gen20k.data.load_bench()`.

**R²:** pooled strict-adjacent R² per split seed, then the seed mean (round-4 METRICS.md). Direction accuracy counts a zero prediction as 0.5. Clusters are round 4's `cluster_cpg` (13 nominal, Kish 2.24, largest 65%), obtained through `r4.metrics.add_clusters`.

**Legality** (gen20 metal model, unchanged):
- A read set is an independent set of the strict-adjacency path that keeps k_max reachable.
- Hence no adjacent pair ever has both ends read.
- Every policy and every k is therefore scored on the **same pairs**: all strict-adjacent pairs of the cell. This was checked: the pair sets are identical across k in every cohort C3–C6, on both designs.

**Reimplementation check:**
- Rebuilding PWL_R from the bench and the parquet read sets reproduces the parquet `PWL_R` exactly (max deviation 0.0 over 400 rows).
- D7 reproduces `HIST_D7` to 8.9e−16.
- DOPT_Q read sets in the simulations are the BPG ladder's own, per seed.

**Scripts and raw outputs** are in the session scratchpad. The rules allowed one output file in `evidence/`, so they are not copied here.
- Folder: `C:/Users/Bandai/AppData/Local/Temp/claude/C--Users-Bandai--claude-spares-spare-9MsqWO/0cd609d3-ed34-4de3-83c3-904e5f4bdb8b/scratchpad/e6/`
- Scripts: `a_ladder.py` (ladders, read geometry), `b_reframe.py` (reframings on REF OOF), `c_policy.py` (policy simulation), `d_target.py` (targeted reads, wider pairs, roughness), `e_learned_adaptive.py` (learned adaptive policy), `f_sibling.py` and `f2_sibling_check.py` (sibling template).
- Each has a `.txt` / `.json` output beside it. Total runtime is under 5 minutes, one process at a time.

---

## 2. How R² and pair error depend on k (measured)

### 2.1 DOPT_Q ladders from the round-4 parquet (BPG; BP differs by ≤ 0.005 R²)

| cohort (cells / pairs per seed) | arm | k=0 | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 |
|---|---|---|---|---|---|---|---|---|
| C3 (229 / 1,802) | PWL_R | — | 0.191 | 0.360 | 0.418 | | | |
| C3 | HBK@ML | −0.116 | 0.101 | 0.344 | 0.419 | | | |
| C3 | LINE | −0.135 | 0.157 | 0.237 | 0.237 | | | |
| C3 | RIDGE2 | −0.116 | −0.011 | 0.288 | 0.323 | | | |
| C3 | G14 zero-shot / FLAT | 0.053 / −0.135 | | | | | | |
| C6 (133 / 1,349) | PWL_R | — | 0.167 | 0.345 | 0.396 | 0.449 | 0.491 | 0.522 |
| C6 | PWL_R MAE | — | 0.1400 | 0.1189 | 0.1111 | 0.1032 | 0.0972 | 0.0927 |
| C6 | PWL_R direction acc. | — | 0.620 | 0.712 | 0.747 | 0.777 | 0.801 | 0.812 |
| C6 | HBK@ML | −0.132 | 0.069 | 0.326 | 0.395 | 0.446 | 0.497 | 0.531 |

**Round-3 C6 ladders with the Gd correction** (BPG; `results/R20/ladder_BPG.csv`, `results/R24/ladder_BPG.csv`):
- **DOPT_Q:**
  - COR[GD]@k: 0.400 / 0.486 / 0.518 at k = 3 / 5 / 6;
  - PWL_R: 0.396 / 0.491 / 0.522.
- **POL_ERR_K:**
  - COR[Dnc]@k (Gd parameter): 0.453 / **0.540** / **0.580**;
  - PWL_R: 0.412 / 0.504 / 0.550;
  - HBK@ML: 0.409 / 0.524 / 0.568.

**Simulated D7 (B1-form) under DOPT_Q on C6** (`c_policy.txt`): 0.206 / 0.385 / 0.431 / 0.483 / **0.524** / 0.552 for k = 1…6. On C3: 0.231 / 0.401 / 0.451.

**First k at which a pair type reaches R² ≥ 0.50** (C6, PWL_R, DOPT_Q, BPG):

| reaches 0.50 at | pair types (R² there) |
|---|---|
| k = 2 | Sm/Eu (0.52) |
| k = 4 | Er/Tm (0.50; BP 0.47) |
| k = 5 | Pr/Nd (0.50); Dy/Ho (0.55) |
| k = 6 | Gd/Tb (0.52) |
| never, even at k = 6 (R² at k = 6) | La/Ce (0.37), Ce/Pr (0.42), Eu/Gd (0.36), Tb/Dy (0.43), Ho/Er (0.40), Tm/Yb (0.38), Yb/Lu (0.29) |

Yb/Lu is about 0 up to k = 3, because DOPT_Q almost never reads Lu (0% of 14-metal cells at k = 3).

**For comparison only: gen15 measured mode** (a different target: all 91 pairs, pair-measurement model). G14 prior + BLUP with D-optimal supports reaches MAE 0.170 at k = 3, and 0.954 sign accuracy on strong pairs. It is not comparable to strict-adjacent R².

**(interp.)**
- The k-curve is concave. Each extra metal is worth about +0.05 R² at k = 3→4 and +0.03 at k = 5→6.
- Under this legality the maximum is k = 6, i.e. every second metal.
- The round-3 statement "0.50 needs about 6 metals (k = 5)" is reproduced:
  - k = 5 is enough for D7 / COR on C6 (0.524 / 0.540 by seed mean);
  - k = 6 is needed for plain PWL_R or HBK under DOPT_Q.

---

## 3. Which metals are read (measured)

### 3.1 Error by read geometry (REF, B1, k = 3, BPG; the BP split is the same within 0.01)

| configuration of the pair | share of rows | share of B1 SSE | R² within | B1 MAE |
|---|---|---|---|---|
| one end read, interpolated | 0.600 | 0.592 | 0.488 | 0.099 |
| one end read, extrapolated | 0.070 | 0.055 | −0.02 | 0.090 |
| both hidden, nearest read 1 away, interpolated | 0.168 | 0.225 | 0.400 | 0.111 |
| both hidden, nearest read 2–3 away, interpolated | 0.113 | 0.087 | 0.45–0.49 | 0.091–0.101 |
| both hidden, extrapolated (any distance) | 0.049 | 0.042 | ≈ 0 | 0.090–0.105 |

**Extrapolated pairs** are those outside the read range, where constant continuation predicts log SF = 0:
- 11.9% of rows and 9.7% of SSE;
- their R² is about 0 by construction;
- removing them raises B1 to 0.472.

B1 needs to remove 8.8% of its SSE (0.452 → 0.50). The extrapolated rows alone hold 9.7%.

**Heavy end.** Tm/Yb and Yb/Lu have Lu read in only 13.6% of rows (BPG). There B1's MAE is 0.068–0.072, against FLAT 0.072: no skill.

### 3.2 How much placement matters at fixed k (D7, C3, k = 3; `c_policy.txt`)

| policy | R² | uses labels of the held-out cell? |
|---|---|---|
| worst legal set per cell | 0.023 | yes (bound) |
| random legal set (mean over all legal sets) | 0.354 | no |
| DOPT_Q (round-4 reference) | 0.451 | no |
| BISECT (largest gap, midpoint; label-free) | 0.462 | no |
| ADAPT_KINK / ADAPT_QBC (value-adaptive heuristics, §4) | 0.462 / 0.456 | own reads only |
| LEARNED adaptive (§4) | 0.457 | own reads only |
| GREEDY_ORACLE (sequential, picks the next metal by true SSE) | 0.575 | yes (bound) |
| ORACLE_SET (best legal set per cell, hindsight) | 0.632 | yes (bound) |

The median cell has 75 legal sets at k = 3; the maximum is 275.

**Fixed (static) sets on the 78 fourteen-metal cells** (all share the same 275 legal sets):
- DOPT_Q: 0.489;
- best single fixed set in hindsight (La, Pr, Ho, Lu): 0.514;
- the same rule chosen leave-one-chemotype-out: **0.484**;
- random: 0.352;
- per-cell oracle: 0.711.

So a static set gives no learnable gain. Read frequencies at k = 3 (oracle vs DOPT_Q):
- the oracle reads Lu in 45% of cells, DOPT_Q in 0%;
- DOPT_Q reads Yb and Gd in 67%, the oracle in 9% and 38%.

**Prior evidence agrees:**
- gen20 §C.6: under the metal model, RANDOM / UNIFORM / DOPT_CURVE / DOPT_Q / UNC_SEQ differ by ≤ 0.005 macro MAE for G14@BLUP20 and RIDGE2.
- Round 3: POL_ERR − DOPT_Q for PWL_R is +0.019 [−0.007, +0.053] (BP) and +0.013 [−0.023, +0.102] at k = 5 (BPG).

**Surprise (measured).** At k = 1, reading the two extreme legal metals (the start of BISECT and the adaptive rules) gives D7 0.291 on C3, against **DOPT_Q 0.231**. In this bench DOPT_Q's first pair is La + Eu, Nd or Pr, not the extremes. By k = 2 DOPT_Q catches up (0.401 vs 0.380).

---

## 4. What an adaptive (sequential) policy could achieve (simulation, measured)

**Policies** (all start from the lightest, then the heaviest legal metal):
- **BISECT** (label-free): read the legal metal nearest the middle of the gap with the most unread metals.
- **ADAPT_KINK**: gap score = unread metals × slope change at the bracketing reads; extrapolation regions score |end slope| × distance.
- **ADAPT_QBC**: read where a committee of PWL, PWL with linear end-extrapolation, PCHIP and a global quadratic disagrees most. All four are fitted to the reads only.
- **LEARNED**: a gradient-boosting regressor predicts the SSE reduction each candidate would bring, from 13 read-derived features. It is trained on random legal read sequences of cells outside the test fold's chemotypes and publications, for each of the 25 BPG folds, and picks the argmax.
- **GREEDY_ORACLE**: the same sequence logic, but it uses the true SSE.

Scoring: D7 on identical pairs.

| cohort | policy | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 |
|---|---|---|---|---|---|---|---|
| C3 | DOPT_Q | 0.231 | 0.401 | 0.451 | | | |
| C3 | LEARNED | 0.291 | 0.400 | 0.457 | | | |
| C3 | GREEDY_ORACLE | 0.291 | 0.509 | 0.575 | | | |
| C6 | DOPT_Q | 0.206 | 0.385 | 0.431 | 0.483 | 0.524 | 0.552 |
| C6 | BISECT | 0.281 | 0.370 | 0.436 | 0.478 | 0.520 | 0.571 |
| C6 | ADAPT_QBC | 0.281 | 0.389 | 0.440 | 0.480 | 0.522 | 0.581 |
| C6 | LEARNED | 0.281 | 0.389 | 0.442 | 0.489 | **0.535** | **0.587** |
| C6 | GREEDY_ORACLE | 0.281 | 0.503 | 0.573 | 0.596 | 0.610 | 0.612 |
| C6 | ORACLE_SET | 0.517 | 0.605 | 0.640 | 0.656 | 0.662 | 0.663 |

Paired contrasts: ΔR² with a cluster_cpg bootstrap (2,000 draws) and the leave-one-cluster-out (LOCO) range.

| contrast | ΔR² | 95% interval | LOCO range |
|---|---|---|---|
| LEARNED − DOPT_Q, C3, k = 3 | +0.006 | [−0.033, +0.019] | [+0.000, +0.010] |
| LEARNED − DOPT_Q, C6, k = 3 | +0.011 | [−0.014, +0.027] | |
| LEARNED − DOPT_Q, C6, k = 6 | +0.035 | [−0.009, +0.057] | all positive |
| LEARNED − BISECT, C6, k = 6 | +0.016 | [+0.000, +0.089] | |
| ADAPT_KINK − DOPT_Q, C3, k = 3 | +0.011 | [−0.009, +0.036] | |
| ADAPT_QBC − DOPT_Q, C6, k = 6 | +0.029 | [−0.006, +0.051] | |
| GREEDY_ORACLE − DOPT_Q, C3, k = 3 | +0.124 | [+0.087, +0.149] | |

**(interp.)**
- The realised adaptive gain is 5–10% of the greedy-oracle headroom at k = 3. At best it moves the 0.50 crossing on C6 from k = 5 to "k ≈ 4.5". It does not change the integer k.
- The oracle bounds are optimistic: they place reads on noisy metals. gen15 found that about half of an own-cell oracle is noise fitting.
- An adaptive policy costs k − 1 sequential experimental rounds.
- An adaptive policy is a separate regime under the round-3/4 protocols (PROTOCOL §6 of round 3).
- It is low priority unless a much stronger value signal is found.

---

## 5. The ceiling: short-range roughness (measured, then interp.)

**Definition.** On the 78 fourteen-metal cells:

  r_m = y_m − linear interpolation of its two neighbours in −radius (12 interior metals; the Pm gap is ignored).

Under every-second-metal reads (k = 6), each adjacent pair's error is ±r of its unread end. So var(r) / var(adjacent log SF) is the irreducible fraction for interpolation.

**Variance:**
- var(r) = 0.0152 raw and 0.0134 centred per metal, against 0.0432 for adjacent log SF;
- so **31–35%** of adjacent variance is out of reach of every-second-metal interpolation;
- this matches ORACLE_SET at k = 6 (0.663) and DOPT_Q D7 at k = 6 (0.552).

**The corpus-mean roughness per metal is small:**
- values: Gd −0.084, Nd −0.067 (spans the Pm gap), Tb +0.052, Ce +0.048, Eu +0.041, Pr +0.030, the others ≤ 0.02;
- the whole mean pattern is ≈ 0.0018 / 0.0432 ≈ **4%** of adjacent variance;
- **(interp.)** a universal per-metal prior (B2-type) can add at most ~0.04 R², and B1's Gd term already takes the largest piece. This fits B2 losing to B1 in round 4.

**Is the roughness transferable?** Correlation of the centred roughness vectors between two cells:

| relation between the two cells | cell pairs | mean r | median r |
|---|---|---|---|
| same extractant, same publication (other condition) | 106 | +0.362 | +0.435 |
| other extractant, same publication | 99 | +0.136 | +0.056 |
| same extractant, other publication | 84 (only 3 extractants) | −0.022 | +0.003 |
| unrelated | 2,714 | −0.017 | −0.031 |

**Copying adjacent log SF from cells of the same extractant** (C3 targets):

| source of the copied value | rows | R² | MAE | sign agreement | correlation |
|---|---|---|---|---|---|
| same publication, other condition | 1,139 | 0.354 | 0.098 | 0.802 | |
| other publication | 796 (8 extractants, 17 publications) | −0.082 | 0.136 | 0.740 | 0.33 |

Per-extractant cross-publication R² (correlation): TODGA 0.05 (0.31; 418 rows); N,N′-dimethyl-N,N′-dioctyl-DGA −0.27 (0.25; 147 rows); C5-BTBP −0.14 (0.34; 158 rows).

**(interp.)**
- The part of adjacent log SF that interpolation misses behaves like lab- or series-specific structure or noise. It is shared within a publication, including weakly across extractants, and not across publications.
- If so, no chemistry descriptor can recover it. The attainable adjacent R² at k = 3 with any model trained on other sources is bounded well below the per-cell oracle.
- This is the quantitative reason why rounds 1–4 found nothing beyond interpolation plus Gd.
- Only replicate whole-series measurements, across and within labs, can separate noise from real chemistry in r. That is DATA_REQUEST item 1, and it is now the deciding experiment for whether 0.50 at k = 3 is attainable at all.

---

## 6. Reframings: what existing models achieve now (REF OOF, k = 3, DOPT_Q; BPG shown, BP within ±0.01)

### 6.1 Only the pairs that matter

Per pair type R² (B1 | B0 | HBK@ML | zero-shot G14), MAE B1 / FLAT, and direction accuracy (all rows, |y| ≥ 0.1):

| pair | rows/seed | R² B1 | R² B0 | R² HBK | R² G14 | MAE B1 / FLAT | dir all / \|y\|≥0.1 |
|---|---|---|---|---|---|---|---|
| La/Ce | 137 | 0.35 | 0.36 | 0.34 | 0.02 | 0.153 / 0.269 | 0.80 / 0.85 |
| Ce/Pr | 125 | 0.42 | 0.42 | 0.42 | 0.01 | 0.158 / 0.272 | 0.87 / 0.89 |
| **Pr/Nd** | 127 | 0.42 | 0.42 | 0.37 | 0.05 | 0.127 / 0.195 | 0.78 / 0.89 |
| Sm/Eu | 158 | 0.42 | 0.42 | 0.43 | 0.09 | 0.109 / 0.205 | 0.84 / 0.97 |
| Eu/Gd | 158 | 0.42 | 0.28 | 0.29 | −0.01 | 0.128 / 0.168 | 0.71 / 0.80 |
| Gd/Tb | 165 | 0.55 | 0.32 | 0.35 | −0.74 | 0.102 / 0.259 | 0.90 / 0.97 |
| **Tb/Dy** | 168 | 0.37 | 0.32 | 0.34 | −0.06 | 0.088 / 0.144 | 0.80 / 0.88 |
| Dy/Ho | 176 | 0.41 | 0.42 | 0.48 | −0.18 | 0.079 / 0.146 | 0.82 / 0.89 |
| Ho/Er | 173 | 0.26 | 0.24 | 0.29 | −0.01 | 0.068 / 0.097 | 0.80 / 0.84 |
| Er/Tm | 132 | 0.26 | 0.26 | 0.24 | −0.03 | 0.066 / 0.088 | 0.77 / 0.87 |
| Tm/Yb | 133 | 0.03 | 0.01 | −0.02 | 0.04 | 0.059 / 0.066 | 0.68 / 0.68 |
| **Yb/Lu** | 150 | 0.00 | −0.01 | −0.14 | −0.02 | 0.082 / 0.078 | 0.52 / 0.53 |

**Pooled subsets** (B1):

| subset | share of rows | R² |
|---|---|---|
| magnet pairs Pr/Nd + Tb/Dy | | 0.401 |
| Yb/Lu (¹⁷⁷Lu production) | | 0.002, with MAE worse than FLAT |
| light La…Nd | | 0.406 |
| **mid Sm/Eu…Dy/Ho** | 46% | **0.504** (B0 0.425) |
| heavy Ho/Er…Yb/Lu | | 0.189 |
| without extrapolated pairs | 88% | 0.472 |

### 6.2 Pair-targeted reads (14-metal cells, D7; `d_target.txt`)

If one pair is declared the target, the best legal read set for that pair is chosen leave-one-chemotype-out. The table compares it with DOPT_Q at k = 3 on the same cells.

| pair | DOPT_Q k=3 | targeted k=1 (2 reads) | chosen reads (hindsight) | per-cell oracle |
|---|---|---|---|---|
| La/Ce | 0.57 | 0.60 | La, Pr | 0.78 |
| Ce/Pr | 0.51 | 0.69 | Ce, Nd | 0.90 |
| Pr/Nd | 0.39 | 0.53 | Pr, Sm | 0.89 |
| Sm/Eu | 0.80 | 0.87 | Nd, Eu | 0.98 |
| Eu/Gd | 0.38 | 0.75 | Sm, Gd | 0.89 |
| Gd/Tb | 0.47 | 0.56 | Gd, Dy | 0.92 |
| Tb/Dy | 0.38 | 0.76 | Tb, Ho | 0.94 |
| Dy/Ho | 0.43 | 0.77 | Tb, Ho | 0.90 |
| Ho/Er | 0.38 | 0.46 | Dy, Er | 0.89 |
| Er/Tm | 0.23 | 0.34 | Ho, Tm | 0.79 |
| Tm/Yb | 0.02 | 0.77 | Tm, Lu | 0.90 |
| Yb/Lu | −0.03 | 0.63 | Tm, Lu | 0.76 |

- The targeted k = 2 and k = 3 values equal those at k = 1, because PWL/B1 are local: a pair's prediction depends only on its two bracketing reads.
- **(interp.)** The targeted gain comes from reading one end of the target pair plus its next-but-one neighbour. If a lab can read one end, it can usually read the other, and then the pair is measured, not predicted.
- So the real content is twofold:
  - DOPT_Q's curve-wide objective spends reads away from any one pair;
  - prediction value for a single pair exists only where one end cannot be measured.
- As a round-5 endpoint it is cheap, but it needs that practical scenario stated.

### 6.3 Wider pairs (group splits) from the same k = 3 reads (D7, DOPT_Q, ≥ 1 hidden end)

| pair set | rows/seed | y sd | R² | MAE | sign agreement |
|---|---|---|---|---|---|
| strict adjacent ΔZ = 1 | 1,802 | 0.222 | 0.451 | 0.100 | |
| ΔZ = 2 (includes Nd/Sm) | 1,395 | 0.329 | **0.677** | 0.127 | 0.80 |
| Nd/Sm only (the classical light/middle cut) | 163 | 0.388 | **0.762** | 0.135 | 0.86 |
| ΔZ = 3–4 | 2,609 | 0.505 | **0.811** | 0.154 | 0.88 |

gen20 reported Nd/Sm at R² 0.66 with RIDGE2 at k = 3. The 0.50 aspiration is specific to the strict-adjacent target, the one with the lowest signal-to-roughness ratio.

### 6.4 Direction and separability (B1)

**Direction accuracy:**

| rows | share of rows | B1 | B0 | HBK | zero-shot G14 | pair-type majority sign (other folds) | always heavier |
|---|---|---|---|---|---|---|---|
| all | 1.00 | 0.775 | 0.770 | 0.774 | 0.688 | 0.622 | 0.666 |
| \|y\| ≥ 0.1 | 0.54 | 0.869 | 0.860 | 0.874 | 0.783 | 0.697 | 0.760 |
| \|y\| ≥ 0.3 | 0.17 | 0.902 | 0.894 | 0.906 | 0.797 | 0.738 | 0.767 |

**Selective direction** (the model answers only when |B1| ≥ c):

| c | coverage | accuracy |
|---|---|---|
| 0.05 | 0.63 | 0.880 |
| 0.10 | 0.47 | 0.909 |
| 0.20 | 0.24 | 0.940 |

**Three separability classes** (|log SF| < 0.1, 0.1–0.3, ≥ 0.3):
- B1's accuracy is 0.652, against 0.459 for the majority class.
- "SF ≥ 2" flagged by |B1| ≥ 0.2: precision 0.48, recall 0.68 (base rate 0.17).
- "Inseparable" (|y| < 0.1) flagged by |B1| < 0.05: precision 0.74, recall 0.60 (base rate 0.46).

### 6.5 Ranking

**Across held-out candidates for one target pair** (cells of the test fold; 288 seed×fold×pair groups, median 18 candidates):

| arm | Spearman (p, y) | Spearman (\|p\|, \|y\|) | top-1 | top-3 | regret (log units) |
|---|---|---|---|---|---|
| B1 | 0.62 | 0.45 | 0.28 | 0.48 | 0.22 |
| zero-shot G14 | 0.27 | −0.01 | 0.07 | | 0.40 |
| random | | | 0.06 | 0.18 | 0.39 |

**(interp.)** This ranking requires 4 reads on every candidate. It is screening by cheap partial measurement, not zero-shot design. Zero-shot remains at chance, as gen15 found.

**Within a cell, ranking its adjacent pairs:**

| arm | Spearman (p, y) | top-1 "easiest adjacent pair" |
|---|---|---|
| B1 | 0.585 | 0.395 |
| B0 | 0.506 | 0.142 (PWL produces near-ties) |
| corpus pair-type prior (other folds) | 0.393 | 0.297 |
| zero-shot G14 | 0.353 | 0.311 |

### 6.6 Uncertainty: cross-fold split-conformal intervals for B1

The calibration residuals are the other folds' OOF rows of the same seed.

| variant | nominal | coverage | largest cluster / rest | pair-type range | cluster range | mean half-width (log; ×SF) |
|---|---|---|---|---|---|---|
| global | 80% | 0.788 | 0.815 / 0.739 | 0.64–0.93 | 0.22–1.00 | 0.151 (×1.42) |
| pair type | 80% | 0.787 | 0.805 / 0.754 | 0.74–0.82 | 0.52–1.00 | 0.152 (×1.42) |
| pair type, normalised by ensemble spread | 80% | 0.798 | 0.825 / 0.748 | 0.76–0.82 | 0.55–1.00 | 0.160 (×1.45) |
| pair type | 90% | 0.895 | 0.908 / 0.870 | 0.84–0.93 | 0.60–1.00 | 0.245 (×1.76) |
| pair type, normalised | 90% | 0.903 | 0.921 / 0.871 | 0.87–0.94 | 0.73–1.00 | 0.263 (×1.83) |

**(interp.)**
- Marginal calibration is easy.
- Conditional calibration per source is not: about 5 points short outside the dominant cluster.
- A 90% interval spans a factor of ~3 in SF (full width ≈ 0.49 log units), while the median adjacent |log SF| is 0.114 (mean 0.164).
- This is honest but of limited decision value for adjacent pairs. For ΔZ ≥ 2 pairs the same width is relatively smaller.

### 6.7 Abstention (selective prediction)

**Label-free difficulty scores**, Spearman with |error|, overall and within pair type:

| score | overall | within pair type |
|---|---|---|
| Mondrian class mean \|error\| from other folds | 0.20 | **−0.06** |
| ensemble spread sd(B1, PWL_Z, HBK, B2) | 0.24 | 0.18 |
| \|B1\| | 0.21 | 0.10 |

Round 3 found HBK's posterior variance anti-calibrated across pair types (Spearman −0.71 / −0.57).

**Keeping the lowest-score fraction** (ensemble spread):

| coverage | MAE | R² |
|---|---|---|
| 100% | 0.100 | 0.452 |
| 80% | 0.095 | 0.437 |
| 50% | 0.082 | 0.422 |
| 30% | 0.075 | 0.423 |

- The oracle ordering would give R² 0.95 at 50% coverage, so the headroom exists but no score reaches it.
- Abstaining on whole pair types: dropping Tm/Yb and Yb/Lu gives 0.457 on 84% of rows; dropping the four heaviest gives 0.459 on 67%.
- **(interp.)** Abstention lowers MAE but cannot be used to claim R² ≥ 0.50.

### 6.8 Sibling-template measured mode (new; exploratory; outside the BP/BPG masks by construction)

**Method.** For a C3 cell with DOPT_Q k = 3 reads, take every other cell of the **same extractant** whose series covers the cell's metals. Predict the levels as

  sib(m) + PWL_R[y_S − sib_S](m)

so that the reads stay exact. Rows are identical to REF.

| sibling source | rows/seed | cells | R² SIB | R² B1 | R² D7 | R² PWL_R | 50/50 blend | MAE SIB / B1 |
|---|---|---|---|---|---|---|---|---|
| same extractant, same publication (other condition) | 1,111 | 149 | **0.573** | 0.449 | 0.447 | 0.404 | 0.574 | 0.070 / 0.091 |
| same extractant, other publication | 786 | 94 | 0.401 | 0.386 | 0.384 | 0.345 | 0.437 | 0.102 / 0.102 |
| other extractant, same publication | 836 | 104 | 0.336 | 0.476 | 0.478 | 0.456 | 0.473 | 0.098 / 0.089 |

**Robustness of the same-publication gain** (`f2_sibling_check.txt`). The near-copy sd is that of the residual shape after removing an offset and a radius slope (median 0.031; 10% of cells below 0.003).

| subset | R² SIB | R² B1 | gain |
|---|---|---|---|
| all | 0.573 | 0.449 | +0.125 |
| excluding near-copy siblings (sd ≥ 0.02; 102 cells) | 0.547 | 0.426 | +0.122 |
| sd ≥ 0.05 (53 cells) | 0.452 | 0.377 | +0.075 |
| sd ≥ 0.1 (25 cells) | 0.382 | 0.328 | +0.054 |
| largest source cluster | 0.716 | 0.534 | +0.182 |
| other clusters (35 cells, 7 chemotypes) | 0.463 | 0.382 | +0.081 |
| sc009 (DGA) | 0.700 | 0.515 | +0.185 |
| sc061 | 0.405 | 0.238 | +0.167 |
| sc077 | 0.475 | 0.335 | +0.140 |

**(interp.)**
- Within one laboratory and extractant, adjacent log SF is largely condition-invariant, and the shared roughness (§5, r = +0.36) is captured by the sibling template.
- This is the only lever in the programme so far that gives > 0.50 at k = 3 on a large subset.
- It answers a practical question that BP/BPG cannot pose: "I measured one full series. How few metals do I need under a new acidity, concentration or diluent?"
- It must be registered as its own regime with its own baselines:
  - copy-sibling: R² 0.354 without reads;
  - B1 on the same rows;
  - an exact-copy exclusion rule, because round 3 flagged seven within-series offset copies;
  - a held-out split of the *conditions inside* a publication.
- It does not transfer across publications: +0.015, or +0.05 as a blend.

---

## 7. Implications for round 5 (interp.; ordered by measured size)

1. **Register a sibling-template ("condition-transfer") regime.**
   - Observed: +0.125 R² over B1 on 62% of C3 rows; 0.573 at k = 3.
   - Pre-register: the exact-copy rule, the copy and B1 baselines, condition-level hold-out inside publications, and the cluster_cpg intervals.
   - Also score k = 1 and k = 2 in this regime: the template might need only 2–3 reads.
   - This is the most promising route to "0.50 with 4 metals", for a defined user who has one full series.
2. **Report decision endpoints beside R², where B1 already has skill and zero-shot has none:**
   - direction (0.775; 0.94 at 24% coverage);
   - the SF ≥ 2 flag (precision 0.48, recall 0.68);
   - candidate ranking (top-1 0.28 vs 0.06; regret 0.22 vs 0.39);
   - group-split pairs (ΔZ ≥ 2: R² 0.68–0.81);
   - pair-targeted reads (0.34–0.87 per pair with two reads), with the practical scenario stated.
3. **Add calibrated intervals as a deliverable, not a selection criterion.**
   - Use Mondrian by pair type, normalised by ensemble spread, plus a cluster-aware check.
   - The current shortfall outside the dominant cluster is about 5 points.
4. **The budget lever as the honest number:**
   - 0.50 on adjacent pairs needs k = 5 (D7/COR) to k = 6 (PWL/HBK) on C6;
   - every-second-metal reads cap at ~0.55–0.59;
   - the per-cell oracle is 0.66.
   - An adaptive or learned policy is worth ≤ +0.01 at k = 3 and ~+0.03 at k = 6 (intervals include 0). Keep it at most as a secondary arm.
5. **Do not spend round-5 slots on:**
   - static fixed-set policies (LOCO 0.484 vs DOPT_Q 0.489);
   - abstention to raise R²;
   - HBK-variance-driven reads;
   - universal per-metal priors beyond Gd (at most ~4% of variance in total).
6. **The deciding data are replicate whole series** within and across labs (DATA_REQUEST item 1). They split the 31–35% roughness into noise and chemistry, which decides whether 0.50 at k = 3 on adjacent pairs is attainable by any method.

**Practical note (interp.).**
- With ICP-MS a laboratory usually obtains all lanthanides from one extraction. The cost driver is the number of experiments (extractant × condition), not the number of metals.
- The k-metal framing is therefore most relevant for literature mining (the corpus median is 6 metals per cell), radiotracer work and subset feeds.
- The "experiments" framing (zero-shot → one partial experiment → sibling series) matches the sibling-template result.

## 8. Caveats

- **Data.** All numbers are dev, on one literature corpus. 65% of rows sit in one source cluster (Kish 2.24), so every interval is fragile. The bootstrap intervals above use 10–13 clusters.
- **Oracles.** The per-cell and greedy oracles use held-out labels and include noise fitting. They are bounds, not targets.
- **D7.** It uses a fixed full-corpus Gd value (−0.072), so it is not fold-independent. It is used only to compare policies on identical pairs. B1 itself was used for every OOF reframing.
- **Simulations.**
  - They use BPG's DOPT_Q read sets.
  - Training-free predictors need no fold masks.
  - LEARNED was trained only on cells outside the test fold's chemotypes and publications. It was not evaluated under BP separately.
- **Sibling analysis.** It deliberately uses same-publication cells. It is a different task from BP/BPG and must not be compared with B1's 0.452 as if it were the same benchmark.
- **Subsets.** R² values on different subsets have different variances. Compare MAE across subsets, and R² only on identical rows.
- **Small n.** The pair-targeted numbers use 78 fourteen-metal cells (78 rows per pair type). The cross-publication roughness correlation rests on 3 extractants.
