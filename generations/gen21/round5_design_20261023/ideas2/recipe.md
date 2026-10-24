# Minimal-measurement recipes: what to measure, given what a chemist already has

Theme report for round-5 ideas, gen21. Written 2026-10-23.

> **Status: EXPLORATORY.** Every number below comes from the 229 C3 cells of the V1 bench. Rounds 1–4 and the round-5 probes have already seen all of them. Every cut used here was known before this note: C5-BTBP out, n_own ≥ 1 or ≥ 2, K = 1.78, the HIST_D7 Gd amplitude. Nothing here is a test.
>
> The headline estimand is new: **whole-series knowledge**. It is training-free, scores all strict-adjacent pairs of a cell, and uses the fixed historical Gd term. Its R² values must never be placed next to the BP/BPG 0.452.

---

## 0. The decision table (bottom line)

**Frame**
- One cell is a new (extractant, condition) for which the chemist wants every strict-adjacent log SF. "n reads" means n metals measured at the new condition.
- **Radiotracers / single-element work:** cost = n experiments.
- **ICP-MS:** cost = 1 experiment for the full series, whatever n is. For ICP-MS users the only real decision is whether to run the new condition at all: 0 or 1 experiment.

**How the cells are scored**
- R² is pooled over all strict-adjacent pairs of the population's cells.
- A pair whose two metals were both read counts as known (error 0). Under the spread rule (HULL) at ≤ 4 reads this applies to **0–0.1 %** of pairs, so those numbers are pure prediction.
- Brackets give a publication-bootstrap 95 % interval. With 2–6 source clusters per population these intervals are descriptive, not inference.
- **Default read rule (HULL):** read the lightest and heaviest metal first, then fill the largest gap in ionic radius.

**Whole-series R² by scenario and number of reads, C5-BTBP excluded**

| what the chemist already has (population) | 0 exp. | 1 read | 2 reads | 3 reads | 4 reads | 5 reads |
|---|---|---|---|---|---|---|
| **P0 nothing** (214 cells, 38 pubs, 13 clusters), interpolation D7 | ≈ 0.05 (zero-shot G14, E1) | = 0 reads | 0.31 [0.24, 0.37]; fixed ladder 0.34; DOPT_Q 0.26; random 0.11 | 0.41; ladder 0.43; DOPT_Q 0.43 | 0.46 [0.40, 0.52]; ladder 0.49; DOPT_Q 0.49 | 0.53; ladder 0.55; DOPT_Q 0.55 |
| **P1 own series**: same extractant, same publication, other condition (115 cells, 17 pubs, 6 clusters), SH_own | **COPY 0.47 [0.28, 0.64]** | = COPY (SH 0.45) | **0.58 [0.49, 0.68]** | 0.65 [0.55, 0.75] | 0.68 [0.58, 0.78] | 0.72 |
| P0 on the same 115 cells (D7, HULL / DOPT_Q) | | | 0.34 / 0.24 | 0.45 / 0.50 | 0.53 / 0.55 | 0.62 / 0.62 |
| **P2 another extractant of the same campaign** (94 cells, 17 pubs, 5 clusters), SH_xext | 0.18 [−0.06, 0.32] | – | 0.41 (D7 0.35; Δ +0.06 [−0.06, +0.17]) | 0.48 (D7 0.45) | 0.51 (D7 0.49) | 0.54 (D7 0.52) |
| **P3 literature series of the same extractant** (73 cells, 13 pubs, 2 clusters), SH_xpub | 0.06 [−0.16, 0.23] | – | 0.22 (D7 0.24) | 0.36 (D7 0.36) | 0.46 (D7 0.47) | 0.56 (D7 0.60) |
| **P4 literature, same chemotype** (141 cells, 26 pubs, 4 clusters), SH_chem | 0.13 [0.02, 0.22] | – | 0.27 (D7 0.27) | 0.38 (D7 0.36) | 0.44 (D7 0.43) | 0.50 (D7 0.51) |

**Direction calls on pairs with |y| ≥ 0.1**, on P1's 115 cells:

| predictor | sign right |
|---|---|
| own COPY, 0 experiments | **0.913** [0.834, 0.955] |
| SH_own, 2 reads | 0.922 |
| SH_own, 4 reads | 0.960 |
| no prior, D7 HULL, 2 reads | 0.858 |
| no prior, D7 HULL, 4 reads | 0.889 |
| no prior, D7 HULL, 5 reads | 0.922 |
| always "heavier extracts more" | 0.814 |

**Recipes**

1. **One metal tells you nothing about adjacent SFs.**
   - A single read only shifts the level, and a level shift cancels in every log SF. This is mechanical: the 1-read column equals the 0-read column for every arm.
   - Shrinking toward a 1-read interpolation even lowers COPY's 0.47 to 0.45.
   - For radiotracers, the first useful purchase is 2 metals.
2. **An own prior series is worth about 2–3 radiotracer reads.** Same 115 cells, HULL on both sides:
   - COPY at 0 reads ≈ no prior with 3 reads (Δ +0.03 [−0.20, +0.24]).
   - SH_own with 2 reads ≈ no prior with 4–5 reads (vs 4: +0.05 [−0.05, +0.15]; vs 5: −0.03 [−0.11, +0.02]).
   - SH_own with 4 reads > no prior with 5 reads (+0.06 [−0.00, +0.12]).
   - With C5-BTBP included, the credit is 2–3 reads as well: SH@2 ≈ D7@5 (+0.03).
3. **Only your own series of the same extractant carries this credit.**
   - Literature series of the same molecule (P3) or chemotype (P4) are worth roughly nothing once you have 2 reads (Δ −0.02 … +0.01).
   - The same campaign's other extractant (P2) is worth about +0.06 at 2 reads, with an interval that spans 0.
   - Head-to-head on the 60 cells that have both an own series and a literature series of the same extractant (8 publications, **1 cluster**):

     | | 0 reads | 2 reads |
     |---|---|---|
     | SH_own | 0.46 | 0.59 |
     | SH_xpub | 0.00 | 0.19 |

4. **Where to read.**
   - **With an own prior series:** any spread rule works. HULL 0.58 / 0.65 / 0.68, DOPT_Q 0.59 / 0.66 / 0.68 and the fixed ladder 0.59 / 0.64 / 0.67 agree within 0.01. Random placement costs 0.04–0.06.
   - **Without a prior series:** the first two reads should span the series. HULL beats random by +0.20 [+0.11, +0.28] at 2 reads and +0.10 at 4.
   - **Fixed ladder:** La,Ho → La,Nd,Ho → La,Nd,Er,Lu → La,Nd,Gd,Er,Lu.
     - It matches the trained DOPT_Q policy at 3–5 reads and beats it at 2 reads (0.34 against 0.26).
     - It beats HULL by about +0.03 (in sample on the C5-free corpus).
5. **Target pair.**
   - With nothing in hand, measure both ends: 2 radiotracer experiments or 1 ICP-MS run. No 2-read placement that excludes an end does better.
   - With an own prior series:
     - the old sign is right in 97 % of |y| ≥ 0.1 rows when |old SF| ≥ 0.1, at 75 % coverage;
     - re-measure only to get the magnitude: COPY's per-pair R² is 0.07–0.71 and its MAE 0.05–0.16 log;
     - and always re-measure Er/Tm, Ho/Er and La/Ce (§3.4).
6. **ICP-MS, skipping an acid level** (§3.5):
   - **average** the campaign's acid-only series; do not copy the nearest acid level (−0.28 R²);
   - a half-shrunk linear trend in log[acid] **inside** the measured range adds +0.04 [−0.004, +0.10];
   - **never extrapolate** a trend (−0.64).

---

## 1. What was run

**Scripts.** All are in `ideas2/scripts/`, and their outputs and logs are in `ideas2/recipe_out/`. Each run took under 1 minute, one process at a time.

| script | what it does |
|---|---|
| `recipe_build.py` | Enumerates every subset S of each cell's measured metals, \|S\| = 1…6 (up to 3,003 per cell), for every arm. It stores the SSE, the \|error\| > 0.1 count and the direction counts per (cell, subset), plus per-pair errors under random (exact expectation) and HULL reads. |
| `recipe_analyze.py` | Ladders per population, the LOPO-static rule, the target-pair table and the sign rule. |
| `recipe_extra.py` | Head-to-head contrasts and reads-equivalents (E1), direction under HULL (E2), and acid interpolation (E3). |
| `recipe_doptq_var.py` | DOPT_Q-nested read sets from the gen21 fold cache (V1), and the within-campaign acid dependence (V2). |
| `recipe_acidfit.py` | Linear log-acid trend arms. |
| `recipe_static_legal.py` | Static sets with and without adjacent metals. |
| `recipe_fixedladder.py` | The frozen fixed ladder. |

**Arms.** All are training-free. u is β = −0.072 at Gd (the HIST_D7 amplitude, fixed in gen13).

| arm | definition |
|---|---|
| D7 | PWL_R[y_S − u_S] + u |
| SIB_pool | T + PWL_R[y_S − T_S], where T is the mean covering series of the pool |
| SH_pool | w·SIB + (1 − w)·D7, with w = n/(n + 1.78) |
| COPY | T, at 0 reads |

**Pools.** They are built from metadata and metal coverage only:
- **own:** probe P1's CLEAN rule;
- **xext:** other extractants of the same publication;
- **xpub:** the same extractant in other publications;
- **chem:** the same chemotype in other publications.

**Read policies**

| policy | rule |
|---|---|
| HULL | Label-free. Lightest and heaviest measured metal, then the farthest metal in −CN8 radius; ties go to the lower Z. |
| RND | Exact expectation over all n-subsets. |
| DOPT_Q | Nested sets seq[:k] from the round-4 BPG fold cache, seed mean over 5 dev seeds. |
| LOPO-static | The fixed n-set ranked best on the other publications (coverage ≥ 50 %); each held-out cell takes the best-ranked set it has, with HULL as fallback. |
| fixed ladder | The LOPO picks frozen from the legal-set run. |
| ORC | The per-cell best set, a hindsight bound only. |

**Checks**
- PWL matches `r3.interp.pwl_levels` to 4e−16 (80 spot checks).
- **D7 on the round-4 DOPT_Q 4-read sets reproduces HIST_D7 exactly:** 0.4511, all 5 seeds, SSE difference 0.
- The round-4 rows are all 1,802 strict-adjacent pairs, because DOPT_Q never reads adjacent metals. So at ≤ 4 HULL reads this frame is the round-4 row set.

---

## 2. Populations

| population | cells | pubs | clusters | largest-cluster share | var(y) |
|---|---|---|---|---|---|
| all C3 | 229 | 39 | 13 | 0.65 | 0.0493 |
| C5-BTBP excluded | 214 | 38 | 13 | 0.71 | 0.0440 |
| own ≥ 1, no C5 | 115 | 17 | 6 | 0.81 | 0.0328 |
| own ≥ 2, no C5 (= P1 PRIMARY) | 99 | 12 | 3 | 0.89 | 0.0257 |
| own = 1, no C5 | 16 | 8 | 4 | 0.40 | 0.0644 |
| xext, no C5 | 94 | 17 | 5 | 0.72 | 0.0438 |
| xpub, no C5 | 73 | 13 | 2 | 0.98 | 0.0300 |
| chem, no C5 | 141 | 26 | 4 | 0.88 | 0.0381 |

---

## 3. Results behind the table

### 3.1 Ladders (HULL; random expectation in parentheses)

| population / arm | 0 reads | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| no C5, D7 | −0.09 | 0.31 (0.11) | 0.41 (0.25) | 0.46 (0.37) | 0.53 (0.47) | 0.60 (0.56) |
| own ≥ 1 no C5, SH_own | 0.45 (COPY 0.47) | 0.58 (0.52) | 0.65 (0.59) | 0.68 (0.64) | 0.72 (0.69) | 0.79 (0.75) |
| own ≥ 1 no C5, SIB_own (raw template) | 0.47 | 0.56 | 0.61 | 0.64 | 0.68 | 0.76 |
| own = 1 no C5, SH_own / SIB_own / D7 | 0.33 / 0.37 / 0.02 | 0.49 / 0.43 / 0.35 | 0.57 / 0.48 / 0.50 | 0.61 / 0.53 / 0.54 | 0.63 / 0.54 / 0.57 | |
| own ≥ 1 **with** C5, SH_own / D7 | 0.40 (COPY 0.41) / −0.16 | 0.50 / 0.28 | 0.55 / 0.36 | 0.58 / 0.42 | 0.61 / 0.48 | 0.65 / 0.55 |
| own ≥ 2 no C5, SH_own / D7 | 0.47 / −0.33 | 0.61 / 0.30 | 0.67 / 0.40 | 0.69 / 0.50 | 0.75 / 0.62 | |

- **With a single own series, shrinkage is mandatory.** The raw template loses to D7 at 3–5 reads; the shrunk SH_own wins by +0.07 [+0.003, +0.22].
- On the same cells, the hindsight-oracle read set gives 0.81 at 2 reads (SH_own) and 0.70 (D7). Placement headroom is large, but not label-free (as in E6).

### 3.2 Reads-equivalent of an own prior (HULL on both sides; ΔR² [publication bootstrap 95 %])

| SH_own at | vs D7 at 3 | vs D7 at 4 | vs D7 at 5 | vs D7 at 6 |
|---|---|---|---|---|
| 0 reads (COPY) | +0.03 [−0.20, +0.24] | −0.06 [−0.26, +0.12] | −0.14 [−0.32, −0.01] | |
| 2 reads | +0.14 [+0.01, +0.28] | +0.05 [−0.05, +0.15] | −0.03 [−0.11, +0.02] | −0.13 |
| 3 reads | | +0.12 [+0.02, +0.22] | +0.03 [−0.04, +0.09] | −0.07 |
| 4 reads | | | +0.06 [−0.00, +0.12] | −0.04 [−0.15, +0.06] |

So the credit is about 2.5 reads, and it shrinks toward 1–2 reads once 4 or more metals are read. DOPT_Q in place of HULL on the D7 side makes the same comparison: D7 DOPT_Q scores 0.50 / 0.55 / 0.62 at 3 / 4 / 5 reads.

### 3.3 Which metals (no prior)

**Static sets beat HULL on the full corpus, but the gain at 3–5 reads is "measure the worst pair directly".**
- On all 229 cells the LOPO-chosen sets contain La + Ce: La,Ce,Er (39 of 39 folds) and La,Ce,Nd,Er. They score +0.07 / +0.11 over HULL at 3 / 4 reads.
- Restricted to legal sets (no adjacent pair), the gain shrinks to +0.03 [−0.02, +0.08] / +0.03 [−0.00, +0.07].
- The La/Ce direct-measurement effect is mostly C5-BTBP: without it, all sets and legal sets agree (0.503 vs 0.500 at 4 reads).

**The frozen legal ladder** is La,Ho (LOPO pick in 32 of 38 folds), La,Nd,Ho (27 of 38), La,Nd,Er,Lu (23 of 38) and La,Nd,Gd,Er,Lu (32 of 38).

| | 2 reads | 3 reads | 4 reads | 5 reads |
|---|---|---|---|---|
| ladder − HULL, no prior (in sample) | +0.033 [+0.001, +0.074] | +0.026 | +0.032 [+0.006, +0.064] | +0.015 |
| ladder − HULL, with own prior | | within ±0.01 | | |

- Coverage is only 39–55 %, because many cells are 5–8-metal series. Cells without the ladder metals use HULL.
- DOPT_Q-nested is poor at 2 reads in this frame: 0.26 against HULL's 0.31.

### 3.4 Target pair at 0 experiments (own COPY, no C5; per pair type)

| pair | R² [95 %] | MAE | P(\|err\| ≤ 0.1) | sign right (\|y\| ≥ 0.1) |
|---|---|---|---|---|
| Pr/Nd | 0.71 [−0.09, 0.82] | 0.073 | 0.77 | 0.88 |
| Sm/Eu | 0.67 [0.13, 0.87] | 0.072 | 0.79 | 1.00 |
| Ce/Pr | 0.50 [0.21, 0.81] | 0.147 | 0.57 | 0.92 |
| Eu/Gd | 0.46 [−0.72, 0.88] | 0.086 | 0.79 | 0.96 |
| Dy/Ho | 0.37 | 0.080 | 0.78 | 0.96 |
| Tb/Dy | 0.27 | 0.102 | 0.58 | 0.96 |
| Er/Tm | 0.24 | 0.058 | 0.78 | **0.50** (n = 14) |
| Gd/Tb | 0.19 | 0.107 | 0.49 | 0.97 |
| Ho/Er | 0.12 | 0.069 | 0.80 | **0.81** |
| Yb/Lu | 0.09 | 0.053 | 0.86 | 1.00 (n = 13) |
| La/Ce | 0.07 | 0.157 | 0.54 | **0.74** |
| Tm/Yb | −0.14 | 0.057 | 0.84 | 0.83 (n = 12) |

**Sign rule (all rows, |COPY| threshold → accuracy at coverage)**

| pool | \|COPY\| ≥ 0 | ≥ 0.05 | ≥ 0.1 | ≥ 0.2 |
|---|---|---|---|---|
| own | 0.79 | 0.88 (70 %) | **0.92 (50 %)** | 0.94 (24 %) |
| own, \|y\| ≥ 0.1 rows | 0.91 | 0.95 (89 %) | **0.97 (75 %)** | 0.96 (40 %) |
| same extractant, literature | 0.77 | 0.80 (80 %) | 0.88 (60 %) | 0.90 (30 %) |
| same extractant, literature, \|y\| ≥ 0.1 rows | 0.88 | 0.88 (90 %) | 0.92 (76 %) | 0.93 (44 %) |
| other extractant, same campaign | 0.62 | 0.73 | 0.77 | 0.78 |
| same chemotype, literature | 0.74 | 0.81 | 0.85 | 0.85 |

- **Direction transfers from the literature** (0.88 on |y| ≥ 0.1, the gen14 lesson). **Magnitude does not:** literature COPY R² is 0.06.

### 3.5 ICP-MS condition skipping (acid-only own siblings; 57 cells, 8 publications, no C5)

| arm, 0 experiments | target acid inside the siblings' range (32 cells) | outside (25 cells) |
|---|---|---|
| MEAN of acid-only siblings | 0.50 [0.21, 0.78] | 0.34 |
| NEAREST acid level | 0.21 (−0.28 vs MEAN) | 0.34 |
| two bracketing levels, linear in log[acid] | 0.52 (+0.03 [−0.10, +0.17]) | – |
| OLS line over all siblings, LIN | 0.53 (+0.04 [−0.06, +0.14]) | **−0.30 (−0.64 [−1.91, −0.03])** |
| LIN with the slope halved | **0.54 (+0.04 [−0.004, +0.10])** | 0.38 (+0.04 [−0.22, +0.25]) |
| LIN with acid clamped to the measured range | 0.53 | 0.44 (+0.10 [−0.17, +0.33]) |
| no prior, D7 with 2 HULL reads | 0.22 | 0.20 |

**Within-campaign acid dependence of adjacent log SF** (all bench cells; 19 groups, 9 publications, 580 points, ≥ 3 acid levels):
- A linear slope in log[acid] explains **0.56** of the within-group SS, against 0.27 expected from noise alone.
- Within-group SD is 0.116; the residual SD is 0.091.
- So acid matters within a campaign. But two bracketing points are too noisy to estimate it, which is why MEAN is hard to beat.

**Noise context**
- COPY_own's RMSE is 0.13 log.
- Per-measurement σ_SF is unmeasured: about 0.06 (transcription doubles) to 0.12 (E2 replicate cells); the within-campaign residual is 0.09.
- So one direct measurement is about 1.1–2× more precise than COPY_own, not 10×.

---

## 4. Ideas

### recipe-1: Reads-equivalent of an own prior series (E-REC credit claim)

**Hypothesis.** For a new condition of an extractant the same laboratory has already measured as a full series:
- SH_own with 2 spread reads reaches at least the whole-series R² of the no-prior interpolation with 4 spread reads;
- COPY at 0 reads reaches the no-prior level at 3 reads;
- a literature series of the same extractant (another laboratory) gives no credit at ≥ 2 reads.

**Mechanism.** A condition change mostly moves the level, which cancels in log SF. The short-range roughness is shared within a laboratory (r = 0.36) and not across laboratories (r ≈ −0.02). It carries the campaign's systematics: calibration, matrix, Ce behaviour.

**Quick check, exploratory** (115 cells, 17 publications, 6 clusters):
- COPY 0.47 against no prior at 3 reads 0.45 (Δ +0.03 [−0.20, +0.24]);
- SH_own at 2 reads 0.58 against no prior at 4 reads 0.53 (+0.05 [−0.05, +0.15]) and at 5 reads 0.62 (−0.03);
- literature same extractant at 2 reads 0.22 against no prior 0.24;
- head-to-head on 60 cells (1 cluster): own 0.59 against literature 0.19.

**Test design.** Prospective, on new laboratory × extractant families with ≥ 2 full-series conditions (the M05 external leg).
- Hold out each condition.
- **Arms:** SH_own (K = 1.78 frozen) at 0 / 2 / 3 reads, HULL; D7 at 2–6 reads, HULL and DOPT_Q-nested.
- **Placebos:** SH from another laboratory's series of the same extractant, and from the same laboratory's other extractant.
- **Control:** SH built from a random other family's series (same chemotype), which should give no credit.
- **Inference:** unit = laboratory × extractant family; family-level sign-flip; reads-equivalent m* = the smallest m with D7@m ≥ SH_own@2, reported with its family-bootstrap distribution.
- **In-corpus registration:** the decision table above becomes one labelled descriptive E-REC table, with C5 and leave-one-publication-out sensitivities.

**Kill criterion**
- The credit claim falls to "< 2 reads" if the mean of SH_own@2 − D7@4 is < 0, or if its one-sided 95 % lower bound is < −0.05, on ≥ 11 families.
- The "own laboratory only" clause is killed if SH_xpub@2 − D7@2 ≥ +0.05 with lower bound > 0. Literature series would then suffice.

**What a chemist would do differently.**
- Radiotracer or single-element laboratories with one full series of the extractant measure only 2 metals (the lightest and heaviest available) at the new acid or diluent, instead of 4–5. That saves 2–3 experiments per condition.
- They do not borrow other groups' curves for magnitudes.

**Cost.** In corpus: done (minutes). External leg: data-gated; P4 found about 4 verified new laboratories with ≥ 3 sibling conditions.

**Risks**
- Shared campaign artefacts flatter the template.
- 81–89 % of rows sit in one cluster.
- Measured pairs get noise-free credit (≈ 0 % of pairs at ≤ 4 HULL reads).
- "Publication" is not "laboratory".

### recipe-2: A fixed, training-free metal ladder for partial series

**Hypothesis.** When metals cost experiments and no prior series exists:
- the fixed ladder (La,Ho → La,Nd,Ho → La,Nd,Er,Lu → La,Nd,Gd,Er,Lu, with HULL fallback) scores within 0.01 of DOPT_Q-nested at 3–5 reads, and ≥ +0.05 above it at 2 reads;
- spread reads beat random by ≥ +0.08 at 2–4 reads;
- with an own prior series, every spread rule ties (±0.01).

**Mechanism.**
- Error does not grow with distance to a read; it depends on which metal is hidden (E2).
- Extremes fix the tilt everywhere.
- Nd and Er/Ho split the light and heavy blocks.
- DOPT_Q's second step is tuned for its own 4-read target, not for a 2-read stop.

**Quick check, exploratory** (no C5):

| | 2 reads | 3 reads | 4 reads | 5 reads |
|---|---|---|---|---|
| ladder | 0.34 | 0.43 | 0.49 | 0.55 |
| DOPT_Q | 0.26 | 0.43 | 0.49 | 0.55 |
| HULL | 0.31 | 0.41 | 0.46 | 0.53 |
| random | 0.11 | 0.25 | 0.37 | 0.47 |

- HULL − random: +0.20 [+0.11, +0.28] at 2 reads and +0.10 [+0.03, +0.15] at 4.
- With an own prior: HULL 0.58 / DOPT_Q 0.59 / ladder 0.59 at 2 reads.

**Test design.** Prospective, on new full series (literature yield M22 / SAFE M33), simulated partial reads of each full series.
- Arms: ladder (frozen now), HULL, DOPT_Q-nested (corpus covariance), random expectation.
- Strata: DGA and non-DGA, and series length.
- Control: a ladder chosen on label-shuffled cells.
- Unit: publication × chemotype; sign-flip test.

**Kill criterion.** The ladder is dropped in favour of DOPT_Q if ladder − DOPT_Q < −0.02 with lower bound < 0 at any of 3–5 reads in both strata. "Spread first" is killed if HULL − random has lower bound ≤ 0 at 2 reads.

**What a chemist would do differently.**
- Buy La and Ho (or Lu) tracers first, never a single tracer.
- Stop at 4 metals: La, Nd, Er, Lu.
- For one target pair, measure both of its ends.

**Cost.** Hours, software only once new series exist.

**Risks**
- Low novelty against M17 / H-READ.
- The ladder is in sample on the C5-free corpus.
- 45–60 % of cells lack a ladder metal.
- The gain is about +0.03, below the round-5 MDE for pooled claims. It is a usability claim, not a model claim.

### recipe-3: Sign-first triage of target pairs from an own prior series

**Hypothesis.** At a new condition of a known extractant:
- the old series' sign of an adjacent log SF is right in ≥ 95 % of separable pairs (|y| ≥ 0.1) whenever |old SF| ≥ 0.1;
- it is right in ≥ 90 % of all pairs at that threshold;
- so re-measurement is needed for magnitude, or for pairs below the threshold, but not for direction.

**Mechanism.** Direction is the transferable bit (gen14). Within a campaign, condition changes move SF by about 0.09–0.12 log within-group SD, which flips signs only near zero.

**Quick check, exploratory**
- Own COPY at |COPY| ≥ 0.1: **0.97** on |y| ≥ 0.1 rows (75 % coverage) and 0.92 on all rows (50 %).
- Overall on |y| ≥ 0.1 rows: 0.913 [0.834, 0.955]. Comparators: always-heavier 0.814; no prior with 4 HULL reads 0.889.
- Literature series of the same extractant: 0.92 at the same threshold. Other extractant of the same campaign (placebo-like): 0.77.
- Weak pairs: Er/Tm 0.50 (n = 14), La/Ce 0.74, Ho/Er 0.81.

**Test design.** On the recipe-1 families, plus P4's literature laboratories for the literature arm.
- Calibration curve of sign accuracy against the |COPY| threshold, per pair type.
- Comparators: always-heavier, D7@2 and D7@4 HULL direction, B1 direction.
- Control: COPY from a random other family's series.
- Unit: family; one-sided lower bound.

**Kill criterion.** Accuracy at |COPY| ≥ 0.1 below 0.90 on all rows (family-level lower bound), or not ≥ always-heavier + 0.05. Per pair type: drop the rule for any pair with accuracy < 0.85.

**What a chemist would do differently.** In process screening across acid or diluent conditions:
- take selectivity direction from the existing own series;
- re-measure only pairs with |old SF| < 0.1, plus Er/Tm, Ho/Er and La/Ce;
- use literature series of the same extractant only for direction, never for magnitude.

**Cost.** Minutes in corpus. External: as recipe-1.

**Risks**
- 57 % of wrong calls in E6 had |y| < 0.1; the three-way rule must be used.
- Few separable heavy-pair rows (n = 12–14).
- 81 % of rows are in one cluster.

### recipe-4: ICP-MS condition skipping by averaging plus a shrunk, clamped acid trend

**Hypothesis.** Inside the acid range a campaign has already measured, the shrunk line beats a plain average of the campaign's acid-only series by ≥ +0.03, and the average beats nearest-condition copying. The shrunk line is half the OLS slope of log SF on log[acid], with the target acid clamped to the measured range. Outside the range, an unclamped trend is worse than the average.

**Mechanism.** Adjacent log SF has a real but small acid slope within a campaign (0.56 of within-group SS explained against 0.27 from noise). Single neighbouring conditions carry about 0.09 log noise, so averaging and shrinkage win over nearest-copy and raw extrapolation.

**Quick check, exploratory** (57 cells, 8 publications):
- inside the range: MEAN 0.50; half-slope line 0.54 (+0.04 [−0.004, +0.10]); full line 0.53; nearest level 0.21;
- outside the range: full line −0.30 (−0.64 vs MEAN); clamped line 0.44 (+0.10 [−0.17, +0.33]).

**Test design.**
- Prospective campaigns with ≥ 4 acid levels for one extractant: P4's Turanov, KU Leuven and Chiba series, and new ICP-MS runs.
- Leave one level out.
- Arms: MEAN, NEAR, LIN, LINHALF, LINCLIP, and D7 with 2 reads as the no-prior reference.
- Unit: campaign; sign-flip test.
- Report interior and exterior targets separately.

**Kill criterion**
- Use MEAN, not LINHALF, if LINHALF − MEAN < +0.02 or its lower bound is ≤ 0 on ≥ 8 campaigns.
- Abandon condition skipping altogether if MEAN interior R² is < 0.30.

**What a chemist would do differently.**
- To map an adjacent SF against acid for a flowsheet, measure 3 acid levels spanning the window (3 ICP-MS runs instead of 6–8).
- Fill in between with the shrunk line.
- Never extrapolate beyond the measured acid range; use the end value instead.
- Spend saved runs on replicates at the operating point.

**Cost.** Minutes in corpus. External: needs campaign data; the literature acid-dependence series are the cheapest supply.

**Risks**
- 8 publications.
- "Acid-only" relies on recorded cond__ fields; phase ratio and contact time are unrecorded.
- The gain is small and its interval touches 0.

---

## 5. Caveats

- **Exploratory and post hoc.** Populations, K, the HULL rule, the ladder (LOPO-frozen from these cells) and the acid arms were all written after earlier looks at these rows. Every interval is a publication bootstrap on 2–6 clusters (one holds 72–98 % of rows), so the intervals are anti-conservative.
- **The estimand is not E-A.**
  - The table scores whole-series knowledge, with the fixed Gd amplitude and no training.
  - Pairs measured directly get error 0: ≈ 0 % of pairs at ≤ 4 HULL reads, 2–14 % for random reads, 13–19 % at 6 HULL reads.
  - Random reads therefore look slightly better than they are, relative to HULL.
- **C5-BTBP** is excluded from the headline rows. With it, P1's credit is the same (2–3 reads), but every level is lower: COPY 0.41, SH@2 0.50.
- **"Own series" means same publication**, not a verified laboratory. P2 and P3 placebos rest on 3–17 publications and 1–5 clusters.
- **The per-measurement noise floor is unknown** (H-CEILING). "COPY is about 1.1–2× worse than one measurement" depends on σ_SF ∈ [0.06, 0.12].
- **No existing file was modified.** Only `ideas2/recipe.md`, `ideas2/scripts/recipe_*.py` and `ideas2/recipe_out/*` were written (python run with `-B`, so no bytecode). No download, no web access.
- **Process slip.** One read-only `git status --porcelain` was appended by mistake to a file-listing command; its output was discarded unread (`| head -0`) and nothing changed. No other git command was run.
