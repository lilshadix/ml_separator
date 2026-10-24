# Round-5 evidence base: adjacent log SF at k = 3, after gen21 round 4

Written 2026-10-23. It merges the six analyst files E1–E6 in this folder into one reference for designing round 5. It also adds
four small reconciliation checks that settle contradictions between those files (§5, Appendix). Nothing here is sealed or
confirmatory.

**How to read the tags**
- **[M]** measured: a number in a repository file, or recomputed by an analyst on the round-4 out-of-fold (OOF) rows or the V1 bench.
- **[O]** oracle: fitted on, or chosen with, test labels. It bounds headroom and is never achievable.
- **[C]** computed for this note (Appendix; scratchpad scripts, not in the repository).
- **[I]** interpretation.
- **E1–E6** are the files in this folder, cited by section: E1 history, E2 error anatomy, E3 data inventory, E4 literature,
  E5 round-4 design review, E6 measurement lever.

**Population and conventions.** Unless stated otherwise:
- V1 data, cohort C3 (229 cells, 1,802 strict-adjacent pairs per seed), k = 3 (four read metals), DOPT_Q read policy;
- BPG design, mean of per-seed pooled R² over the 5 dev seeds. BP agrees within ±0.003 unless a BP value is given;
- "SSE share" is a slice's share of B1's summed squared error.

**Standing caveat.** Every number below comes from the same 229 cells and the same 5 dev seeds that rounds 1–4 have already
seen. New split seeds only re-partition these cells, so they test stability, not replication. Several analyst choices were
made after looking at results:
- the [−2, 2] window;
- the shrinkage K = 10;
- the damping d = 0.375;
- the half-slope rule;
- the max-min 5th read.

These are **leads, not results**.

---

## 1. The problem as it really is

### 1.1 Target and reference ladder

| quantity | value | source |
|---|---|---|
| target y = log D(A) − log D(B), strict ΔZ = 1, 12 pair types | mean −0.0815, sd 0.222, **var 0.0493** | E1 §0, E2 |
| typical magnitude | median \|log SF\| 0.114, mean 0.164; interior mean ≈ 0.17 log ≈ **1 kJ/mol** (1 log = 5.71 kJ/mol) | E6 §6.6, E4 unit bridge |
| zero-shot structure models (no read metal) | G14 R² **0.053** (0.04 on withheld seeds); gen20.1 log-D routes −0.010 / −0.016; neural NNC −0.128 BP (−0.29 confirmation) | E1 §1, §2.1 |
| LINE / RIDGE2 at k = 3 | 0.237 / 0.323 | E1 §2.4 |
| **B0 = PWL_R** (radius interpolation, constant beyond the reads, no training) | **0.418** (BP 0.418) | E1 §2.4 |
| HBK@ML (best learned curve model) | 0.419; error correlation with PWL_R 0.971 | E1 §2.4 |
| **B1 = B0 + one fold-fitted Gd amplitude** | **0.4521 / 0.4519**; balanced MAE 0.1011; MSE 0.0270; RMSE 0.164 | `round4_20261022/REPORT.md`; E2 |
| HIST_D7 (B1's form with the fixed historical −0.072) | 0.4511 / 0.4506 (= B1 − 0.001) | REPORT.md |
| B1 source-cluster interval | [0.394, 0.497] | REPORT.md |
| SSE/SST of B1 | 0.548. To reach 0.50, B1 must remove **8.8 %** of its SSE | E5 §3.1, E6 §3.1 |

### 1.2 Where B1's error lives [M]

| slice | rows | B1 SSE | note | source |
|---|---|---|---|---|
| top 1 % / 5 % / 10 % of rows | | **31 % / 59 % / 73 %** | kurtosis 18.2. Robust residual sd 0.086 vs RMSE 0.164; median \|res\| 0.058 | E2 §0, §1.4 |
| worst 10 % of cells (23) | | 53.5 % | fixing them would give R² 0.745 [O] | E2 §3d |
| one cell, c950dc2982e96b83 (C5-BTBP) | | **11.8 %** | log D −3.2…−5.0, zig-zag | E2 §1.4 |
| one publication, pub_6a0ef8528c (C5-BTBP, 15 cells) | 8.2 % | **23.1 %** | B1 R² 0.238 inside it. Pooled B1 without it: **0.486** (B0 0.459). This is a sensitivity, not a gain | E1 §3.1, E2 §1.3 |
| light pairs La/Ce…Pr/Nd | 21.6 % | **42.8 %** | La/Ce + Ce/Pr alone: **34.2 %** on 14.5 % of rows. Ce is hidden in 97 % of those rows | E2 §1.1 |
| Gd region Sm/Eu…Gd/Tb | 26.7 % | 27.2 % | Eu/Gd + Gd/Tb 16.5 % | E2 §1.1 |
| Tm/Yb + Yb/Lu | | 8.6 % | R² ≈ 0, but a perfect fix would add only +0.047 | E2 §1.1 |
| pairs extrapolated beyond the heaviest read | 11.9 % | 9.7 % | R² ≈ 0: constant continuation predicts log SF = 0 there. No rows are below the lightest read (La is always read) | E2 §1.2, E4 §3.3 |
| either log D outside [−2, 2] (label-dependent) | 26.1 % | **49.7 %** | MSE 0.052 vs 0.018 | E2 §1.6 |
| one end read / both hidden | 67 % / 33 % | 64.7 % / 35.3 % | R² 0.470 / 0.416 | E2 §1.2 |
| giant source cluster g:pub_74428aeab2 | 64.9 % | 51.8 % | | E2 §1.3 |
| non-DGA donors | 37.5 % | 55.2 % | MSE 0.0397 vs DGA 0.0193 | E2 §1.3 |

**Error does not grow with distance to the nearest read** (MSE 0.028 at 1 Z, 0.020 at ≥ 4 Z) [M, E2 §1.2]. Which metal is hidden
matters; how far it is from a read does not.

### 1.3 Structure of the residual [M]

- **Metal-local, heavy-tailed.** Consecutive-pair residual correlation is −0.522 when the shared metal is unread and +0.091 at
  lag 2 (E1 §3.2). Independent per-metal deviations would give −0.5 and 0.
  - Part of the −0.5 is mechanical: the correlation is exactly −1 when both outer metals are read (E2 §1.5).
  - Consecutive hidden **level** errors correlate +0.69 (E2 §1.5, E5 §3.4). The error is local spikes plus smooth level drift
    along runs of hidden metals.
  - [I] Smooth curve-shape error is about 10 % of the residual variance (E1 §3.2; gen13 D7: 7.2 %).
- **Short-range roughness is a third of the target variance.** On the 78 fourteen-metal cells, each metal's deviation from its
  two neighbours has variance 0.0134–0.0152, against 0.0432 for adjacent log SF: **31–35 %** (E6 §5).
  - The corpus-mean per-metal pattern is only about 4 % (Gd −0.084, Nd −0.067, Tb +0.052, Ce +0.048, Eu +0.041).
- **The roughness is shared within a laboratory, not across laboratories.**

  | relation between two cells | roughness correlation (E6 §5) | B1 residual correlation (E2 §0.6) |
  |---|---|---|
  | same extractant, same publication | **+0.36** | 0.33 |
  | other extractant, same publication | +0.14 | |
  | same extractant, other publication | −0.02 (3 extractants) | 0.10–0.14 |
  | unrelated | −0.02 | |

- **Per-cell offsets are irrelevant to pairs.** Cell means explain 2.6 % of the pair-residual sum of squares (E2 §3), although
  36 % of the *pointwise* error energy is a per-cell offset that cancels in every both-hidden pair (E5 §3.4).
- **Even reading every other measured metal of the cell, with the k = 3 hidden ends kept hidden, reaches only about 0.50.**
  - With the Gd term: **0.504**, against HIST_D7's 0.451 on the same rows [C].
  - Without the Gd term: 0.477 [C].
  - Both-hidden rows go 0.416 → 0.463 [C].
  - [I] At fixed hidden metals, R² is capped near 0.50. The rest is the hidden metals' own deviations. (This corrects E1 §3.3's "+0.012"; see §5 C1.)

### 1.4 Noise floor: bounded, not measured [M]

| handle | implied ceiling or share | source |
|---|---|---|
| robust per-metal σ (MAD of third and second differences, siblings, replicate series) 0.030–0.058 | floor = 4–16 % of B1's MSE; max R² 0.91–0.98 | E2 §2 |
| mean-square σ (tails included) 0.107–0.188 | floor = 56–87 % of B1's MSE; max R² 0.52–0.69 | E2 §2 |
| 15 replicated C3 cells, same rows | replicate noise of the target = **54 % of B1's MSE** (median cell 48 %; in 4 of 15 cells noise ≥ B1's MSE) | E2 §2 |
| second-difference local variances at the hidden ends | R² ceiling 0.65 (raw) / 0.92 (robust) | E1 §3.3 |
| per-cell link between error and roughness | Spearman(cell B1 MSE, cell roughness) **0.88**; (cell MSE, sd of y) 0.91 | E2 §2 |
| replicate sd history | 0.302 (gen13) was hidden-condition merges. Transcription replicate sd is **0.049** (n = 86); 53 double reports give sd(Δ log SF) 0.086. **Laboratory repeatability has never been measured** | E1 §2.9 |
| single-target ceiling 1 − σ_SF²/0.0493 | 0.50 is impossible if σ_SF ≥ **0.157** (E3's model). Under E2's floor model (floor MSE ≈ 1.33 σ_metal², i.e. 0.67 σ_SF²) the threshold is about **0.19** [C arithmetic] | E3 §5, E2 §2 |

[I] **Reading.**
- **Bulk rows:** B1's typical-row error variance (0.0074) is only about 2–6× the robust noise floor.
- **Tail:** in the tail cells (out of window, spiky, floor-like), the cell's own roughness and replicate spread are as large as B1's error.
- The noise share of B1's MSE is known only to lie in [0.08, 0.87].

### 1.5 What B1 captures [M]

- **Interpolation plus one known number.**
  - The fold-fitted β averages −0.094 ± 0.015 over 25 BP folds (E1 §2.4).
  - The in-sample full-information β is −0.118, and −0.093 without the Eu/Gd and Gd/Tb pairs (E5 §5.1).
  - The raw Gd chord is −0.115 ± 0.013 (n = 127) (E4 §2.1).
  - The historical −0.072 scores the same (−0.001). R² is flat over β ∈ [−0.072, −0.094].
- **Where the gain is.** Eu/Gd 0.276 → 0.422 and Gd/Tb 0.321 → 0.546. These two pairs carry **94 %** of B1's SSE gain over B0
  (3.02 SSE per seed) (E3 §3).
- **Not cell chemistry.** A within-pair-type shuffle keeps 86–90 % of the gain (E1 §2.4). B1 has learned a pair-type mean.
- **Where the gain comes from, by source:**
  - C5-BTBP: **34.6 %** of the gain from 8.2 % of rows;
  - DMDPhPDA: **17.4 %** from 6.7 %;
  - the giant DGA cluster: 26.8 % from 64.9 %;
  - aqueous-complexant cells: **negative**, −0.44 SSE per seed (E3 §3–4).

  More than half of the gain rests on the two sources whose provenance is pending.
- **How firm it is.** +0.034 R² / +0.005 balanced MAE in 5/5 seeds on both designs. It replicated on round 3's once-only final
  seeds (+0.031 / +0.032). It **fails** the cluster jackknife-t interval ([−0.009, +0.077]) (E1 §4.9, E5 §3.11).

---

## 2. What is established

Each item gives the claim, the numbers and the source.

1. **Zero-shot is closed** for strict-adjacent log SF: R² ≈ 0.05.
   - The honest all-pairs gain over FLAT is +0.088 (gen15).
   - Direction is the one transferable bit (0.821 BP macro accuracy, gen14), and it is exhausted: an oracle direction adds +0.012–0.015 (E1 §2.1).
2. **At k = 3, learned curve models are interpolation.** PWL_R 0.418 = HBK@ML 0.419. The ML residual is a near random walk
   (ρ 0.986), and PWL is its conditional mean (E1 §2.4).
3. **The Gd dip is the only portable series effect**, and it is additive.
   - Spearman(Gd chord, cell slope) is −0.13 (p 0.15). Light-selective cells still show −0.073.
   - An amplitude-scaled Gd term is worse in sample (−0.0025) and under leave-one-chemotype-out (LOCO, −0.017; 0/5 seeds) (E4 §1.2).
4. **The Gd dip's size depends on chemistry in the raw data, but that does not transfer.**

   | group | Gd dip |
   |---|---|
   | DGA | median −0.064 |
   | non-DGA | median −0.175 |
   | BTBP cluster | −0.289 |
   | sc009 | −0.039 |
   | other chemotypes | −0.167 |

   - ICC by extractant is 0.69 (E1 §3.4, E3 §7).
   - A family Gd scale is +0.014 in sample but −0.001 to −0.004 under LOCO (E4 §2.1).
   - A per-chemotype amplitude under LOCO gives 0.453 (no gain) (E1 §3.4).
   - **The Nd dip is an sc009 property:** −0.123 there, +0.004 elsewhere (E1 §3.4).
5. **Universal series priors fit in sample and fail out of source.**

   | prior | in sample | held out |
   |---|---|---|
   | pair-type constants | +0.012 | −0.018 (LOCO); −0.043 (leave-own-cluster-out) |
   | family × pair-type | +0.073 | −0.013 to −0.047 |
   | B2 | | −0.022 vs B1 (0/5 seeds) |
   | convex blends of 14 arms | +0.003 to +0.034 | −0.019 to −0.035 (leave-one-cluster-out) |

   Sources: E4 §2.2, E5 §3.1, E2 §3c.
6. **Smoother or higher-order interpolation is worse** than PWL_R:
   - PCHIP −0.020; natural spline −0.023; local quadratic −0.095;
   - global smooth fits to all of the cell's other metals: 0.17–0.21;
   - a line through the two true flanking neighbours does not beat B1 in mean square (0.0277 vs 0.0238) (E2 §3).
7. **The systematic remainder is source-local.**
   - Publication × pair-type offsets of B1's residual, leave-one-cell-out: R² **0.476 (K = 0) … 0.533 (K = 5) / 0.524 (K = 10)** [C, reproducing E1, E2 and E5].
   - The same correction from same-chemotype cells of *other* publications: +0.005 (E2 §3b).
   - Copying an adjacent log SF from the same extractant in another publication: R² −0.08 (E6 §5).
8. **Sibling template (same extractant, same publication, another condition) used as the interpolation template for the four
   reads: R² 0.573 vs B1 0.449** on identical rows (1,111 rows per seed, 149 cells, 62 % of rows) (E6 §6.8).

   | check | gain over B1 |
   |---|---|
   | all | +0.125 |
   | near-copies excluded | +0.122 |
   | residual-shape sd ≥ 0.1 | +0.054 |
   | outside the dominant cluster | +0.081 |
   | each large chemotype | +0.14 to +0.19 |

   - Copy-sibling without reads: 0.354.
   - Siblings from other publications add only +0.015.
9. **Budget is the measured lever.**
   - One legal 5th read on the full C3 query set (HIST_D7 form): 0.451 → **0.495** (random) / **0.503** (label-free max-min gap; seeds 0.499–0.510) / 0.541 [O]. Six reads give 0.530 (E3 §6).
   - On C6, the D7 form under DOPT_Q gives 0.206 / 0.385 / 0.431 / 0.483 / **0.524** / 0.552 at k = 1…6 (E6 §2.1). The gain per extra read falls from about +0.05 to +0.03.
10. **Placement at fixed k has large hindsight headroom that label-free rules do not capture** (C3, k = 3, D7) (E6 §3–4):

    | policy | R² |
    |---|---|
    | random | 0.354 |
    | DOPT_Q | 0.451 |
    | BISECT | 0.462 |
    | learned adaptive | 0.457 (+0.006 [−0.033, +0.019]) |
    | greedy oracle [O] | 0.575 |
    | per-cell oracle [O] | 0.632 |

    A static fixed set chosen leave-one-chemotype-out scores 0.484 vs DOPT_Q 0.489 (14-metal cells).
11. **A label-free applicability domain reaches 0.54.**
    - Rule: keep rows where B1's predicted levels of both metals lie in [−2, 2].
    - It keeps 74.7 % of rows. Inside, B1 R² is **0.540** (seeds 0.521–0.553; BP 0.538); outside, 0.301.
    - **Random abstention at the same coverage gives 0.453 [0.436, 0.468]; max over 500 draws 0.477** [C].
    - Without C5-BTBP the inside R² is 0.541 [C]. y variance inside is 0.041 vs 0.049, so the gain comes from the MSE drop (0.027 → 0.019) [C].
    - The threshold was chosen post hoc among 4 windows (E2 §1.6).
12. **Decision endpoints where B1 already has skill** (E6 §6):

    | endpoint | B1 | comparison |
    |---|---|---|
    | direction | 0.775 (0.869 when \|y\| ≥ 0.1; 0.94 at \|p\| ≥ 0.2, 24 % coverage) | always-heavier 0.666; zero-shot G14 0.688 |
    | three separability classes | 0.652 | majority 0.459 |
    | ranking ~18 held-out candidates for a pair | top-1 0.28; regret 0.22 log | random 0.06 / 0.39; G14 at chance |
    | wider pairs from the same reads | ΔZ = 2 **0.677**; Nd/Sm **0.762**; ΔZ = 3–4 **0.811** | |
    | mid-series subset Sm/Eu…Dy/Ho (46 % of rows) | 0.504 | |

13. **Conformal intervals are calibrated marginally but not per source.**
    - Mondrian by pair type: 0.787 / 0.895 coverage at 80 % / 90 % nominal; half-widths 0.152 / 0.245 log.
    - Outside the dominant cluster: 0.754 at 80 % nominal; per-cluster range 0.52–1.00 (E6 §6.6).
14. **The round-4 machinery worked; the decision geometry did not** (E5 §0, §3):
    - fit poisoning 50/50;
    - B0 and HIST_D7 reproduced exactly (also in [C]: max difference 8.9e−16);
    - but BP and BPG share **100 % of fold assignments** (B1 row errors correlate 0.988 across designs);
    - one fold holds 58.2 % of scored rows and is trained on only **51–69 C3 cells**;
    - S02's loss is +10.97 ΔSSE in that fold and −0.31 elsewhere (+0.0019 R² within the other folds).
15. **Inner CV was optimistic, not conservative.** When S05 or S06 chose trees, the outer SSE was worse in 12 of 14 BPG fold-fits.
    The median inner gap (+1.1 % SSE) predicted S05's outer −0.006 (E5 §3.5).
16. **Pointwise training is misaligned with the pair target.** S05 beats B1 on pointwise MSE (0.03727 vs 0.03758) but loses on
    pair MSE (0.02730 vs 0.02699). Removing the per-cell offset does not help (S05 0.4459 → 0.4395) (E5 §3.4).
17. **The margins are in different currencies.**
    - 0.005 balanced MAE ≈ +0.034 R² under B1's gain structure, and ≈ +0.053 R² under uniform shrinkage.
    - +0.01 R² ≈ 0.0004–0.0015 balanced MAE.
    - Block A's whole hypothesis class failed the balanced-MAE leg **even as a test-label oracle** (donor × pair +0.0408 R² / +0.0004 balanced MAE) (E5 §3.1).
18. **The data situation.**
    - The archive is exhausted for Ln/Ln pairs: 97.7 % of archive strict-adjacent pairs are already in V1.
    - Relaxed cohort rules add **0 new source clusters**.
    - 2,293 of the 2,637 outside rows are single-lanthanide conditions (E3 §0–2).
19. **Inference geometry** (E3 §2.6, POWER.md):
    - Kish 2.24 by source cluster, 3.85 by chemotype, 18.8 by publication;
    - 9 of the 13 clusters are ORNL-only imports without a data location;
    - confirming a B1-sized gain needs about **40 (optimistic) to > 80 (heterogeneous)** new independent sources;
    - the realistic supply is 0–2 from the repository plus 3–5 from the literature, which gives power ≤ 0.10.
20. **Physical descriptors are too coarse.**
    - DFT ΔΔG(An–Ln) MAE is 2.6 kJ/mol (≈ 0.45 log). Hydration energies are good to ±10 kJ/mol.
    - The adjacent signal is about 1 kJ/mol (E4 §6).
    - The xTB composition trap: only 60 / 229 C3 cells keep the inner-sphere composition constant (E1 §2.2).
21. **External replication of the programme's main lesson.** On the public 1,202-point Ln log D set, a row-split RF reaches
    R² 0.94; leakage-controlled grouped splits give 0.20 to −0.03 (Harish 2026). No paper reports held-out adjacent log SF skill
    (E4 §6).

---

## 3. What is ruled out, and how firmly

**Firmness grades**
- **FIRM:** negative or null when cross-fitted; an oracle ceiling below the margin; or replicated.
- **MODERATE:** null against B1 and consistent with the cross-fit ceilings, but the test itself lacked power.
- **WEAK:** "not detected". The control had no shown power, or the test was never run.

| route | result | grade | power caveat | source |
|---|---|---|---|---|
| zero-shot structure → strict-adjacent log SF (ECFP, RDKit, ChemBERTa, kernels, logK prior, log-D route, NNC) | R² ≈ 0.05 at best; NNC −0.128 / −0.29 | FIRM | confirmation seeds, many designs | E1 §2.1–2.3 |
| learned curve families vs PWL at k = 3 (RIDGE2, HB2+GP, HBK) | ≤ PWL_R; error correlation 0.971 | FIRM | random-walk algebra | E1 §2.4 |
| smoother interpolation (PCHIP, spline, local quadratic, global fits) | −0.020 to −0.095 | FIRM | | E2 §3 |
| universal per-metal / pair-type / family priors (B2, pair constants, tetrad / double-double / inclined W) | LOCO −0.018 to −0.047; B2 −0.022 (0/5); corpus-mean roughness ≤ 4 % of variance | FIRM | ceiling ≈ 0.04 even if perfect | E4 §2.2, E5 §3.1, E6 §5 |
| amplitude-scaled (proportional-LFER) anomalies | in sample −0.0025; LOCO −0.017 (0/5) | FIRM | | E4 §1.2 |
| convex blends / stacking | leave-one-cluster-out −0.019 to −0.035; R13/R22 +0.002/+0.003 | FIRM | | E2 §3c, E1 §2.3 |
| per-cell offset / global recalibration | leave-one-pair-out 0.292; [O] recalibration +0.0005 | FIRM | | E2 §3, E5 §3.1 |
| static fixed read sets | LOCO 0.484 vs DOPT_Q 0.489 | FIRM (78 cells) | | E6 §3.2 |
| cross-metal xTB energies with changing composition | ρ −0.084, reliability 0.80 (powered null) | FIRM | | E1 §2.2 |
| abstention by difficulty scores (ensemble spread, Mondrian error table, \|B1\|) | within-pair-type Spearman ≤ 0.18; R² falls to 0.42 at 50 % coverage | FIRM for these scores; **not** for the level window (§2.11) | | E6 §6.7 |
| S02 electron-count shapes × donor class | −0.024 / −0.028 | MODERATE | loss sits in the 51–69-cell big fold (+0.0019 R² elsewhere) | E5 §3.2 |
| S01 donor-class Gd / Nd dips | −0.0001 / −0.0005 | MODERATE (in substance) | synthetic 0/8 at D7 size; but the cross-fitted donor × pair ceiling is −0.041, and the [O] ceiling gives +0.0004 balanced MAE | E5 §3.1, E1 §2.5 |
| S05 CatBoost residual corrector | −0.006 / −0.003 | MODERATE | dev MDE for S05 − B1 is 0.011; inner selection optimistic | E5 §3.5, §5.3 |
| S11 Ln-only multitask encoder | −0.0005 / −0.0017 (BP balanced-MAE interval excludes 0) | MODERATE | row correlation with B1 0.99983, i.e. it reproduced B1 | E2 §1.1 |
| S06 compact 3D geometry | −0.006 / −0.011 | WEAK | synthetic GEO claim 0/6; 195 / 229 cells change composition | E5 §4 |
| An(III) transfer (S13–S16) | inner gate 3/50 | WEAK | synthetic 0/6; 83–93 % Am | E1 §2.10 |
| learned value-adaptive read policy | +0.006 [−0.033, +0.019] at k = 3; +0.035 [−0.009, +0.057] at C6 k = 6 | WEAK (not ruled out at high k) | 13 clusters | E6 §4 |
| sibling-read pooling under BP (HBG@POOL) | −0.041 / +0.002 | FIRM for *reads*; it says nothing about full sibling curves | | E1 §2.8 |

**Power caveats that apply to every "MODERATE" and "WEAK" line** (E5 §3, E1 §4):
- **Clusters:** Kish 2.24; the largest cluster holds 65 % of rows. The percentile cluster bootstrap rejects at 0.07–0.08 for a nominal 0.05.
- **Synthetic studies:** 6–8 replicates (exact 0/n upper bounds 0.37–0.46) and upper-bound rules. They were miscalibrated: S05 and S11 gained about +0.02 over B1 in NULL worlds.
- **Dev MDE (80 %), cluster_cpg:** B1 − B0 0.044; S05 − B1 0.011; S11 − B1 0.003.
- **BP is not independent of BPG** (same test folds).

[I] **A real effect below about 0.01 R² is invisible on this corpus.** "Ruled out" here means "cannot beat B1 by ≥ 0.01 under
BP/BPG". It does not mean "chemistry absent". The in-sample donor contrast (+0.04 R² [O]) is real; it is just not transferable
across about 4 effective chemotypes.

---

## 4. Open hypotheses ranked by expected information gain

**How the ranking was made.** Expected information gain ≈ (how much the answer would change a round-5 or programme decision) ×
(how uncertain the answer is now) × (whether a round-5 test could be decisive). Hypotheses whose outcome is nearly known
(for example H-EDGE) rank lower even when likely positive. Data-gated items rank by stakes, with the gate stated.

| rank | ID | hypothesis | current evidence | decisive test | regime / cost | why this rank [I] |
|---|---|---|---|---|---|---|
| **1** | **H-SIBLING** | With one full series of the same extractant measured by the same laboratory at another condition, 4 reads predict strict-adjacent log SF at R² ≥ 0.50 | template 0.573 vs B1 0.449 on 62 % of rows; +0.122 without near-copies; +0.054 at shape sd ≥ 0.1; +0.081 outside the dominant cluster. Roughness r = 0.36 within publication × extractant. Publication offsets 0.524 (K = 10) / 0.476 (K = 0) [C] | register a **condition-transfer regime**: hold out conditions inside a publication; baselines copy-sibling (0.354), B1 and publication offsets; a pre-declared near-copy exclusion and multi-sibling combination rule; cluster intervals; a k = 0…3 ladder | separate estimand; training-free; software only | the only lever > 0.50 at k = 3 on a large subset. The open uncertainty is how much is near-copy or shared artefact. It defines a new deployable claim |
| **2** | **H-BUDGET** | Five measured metals (k = 4) with B1 reach 0.50 on C3 | legal 5th read 0.495 (random) / 0.503 (max-min, 3/5 seeds ≥ 0.50) / DOPT_Q-nested 0.497 (B1 elsewhere); dense reads with Gd 0.504 [C]; C6 crosses at k = 5 | a registered k = 4 ladder row on identical queries: DOPT_Q-nested 5th read, **β refitted per fold**, B0 / B1 / HIST_D7, plus max-min as a pre-declared second policy | BPG / BP; no training; cheap | the estimate sits exactly on 0.50 (±0.005), so the outcome is maximally uncertain. It is the honest "how many metals" number |
| **3** | **H-CEILING** | Laboratory repeatability of adjacent log SF (σ_SF) is below about 0.16–0.19, so 0.50 at k = 3 is physically reachable | ceiling bounds 0.52–0.98; replicate noise 54 % of B1's MSE on 15 cells; transcription bound 0.086 → ceiling 0.85; never measured | about 6 systems × ≥ 3 independent full La–Lu runs, with per-metal detection limits; the decision threshold computed by simulating the read policy | **data-gated** (one ICP-MS campaign) | decides whether 0.50 is a modelling or a measurement problem. Highest stakes; not runnable in software |
| **4** | **H-WINDOW** | A label-free applicability domain (predicted levels in a pre-declared window) gives R² ≥ 0.50 on a declared coverage | 0.540 at 74.7 % coverage vs random abstention 0.453 [0.436, 0.468] [C]; robust to dropping C5-BTBP (0.541) [C]; difficulty scores fail (E6) | register a coverage–R² curve over a pre-declared threshold grid, with the random-abstention control and per-cluster coverage; score B0 and B1 | BPG / BP; no training | cheap and decisive, and it settles the E2 / E6 contradiction. The risk is post hoc threshold choice, so the grid must be fixed in advance |
| **5** | **H-FLOOR / V3** | Part of the C5-BTBP / light-end error is a detection-floor or attribution artefact: (a) a reads-only floor-aware (censored) predictor reduces it; (b) source recovery changes the reference | publication = 23.1 % of SSE; 43 light-end rows = 16.4 %; light Ln at log D ≈ −4.2 in every cell; B1 0.486 without it; 34.6 % of B1's gain | (a) a pre-specified censoring rule on reads only (e.g. reads ≤ a declared floor treated as intervals), scored on all rows; (b) a V3 with interval scoring, frozen before any recovered value is read | (a) software; (b) needs the user (journal access) | moves the reference by up to +0.034, and about half of B1's gain depends on it. (b) is data-gated |
| **6** | **H-METRIC** | Round-4 conclusions change under a cluster-balanced or robust co-primary metric | 5 % of rows = 59 % of SSE; the big fold is 58 % of rows; S02 is +0.0019 R² outside the big fold | re-score the stored S01–S11 parquets per cluster and per fold type with trimmed / Huber / per-cluster relative SSE; then pre-register one co-primary for round 5 | existing OOF; minutes | almost free. It decides whether past nulls are big-fold artefacts and which metric round 5 seals |
| **7** | **H-READ** | A label-free policy at fixed budget (BISECT, max-min, B1-error-aware) beats DOPT_Q by ≥ 0.01 | BISECT 0.462 vs 0.451; ADAPT_KINK +0.011 [−0.009, +0.036]; R23 − R14 +0.017 / +0.009 (final seeds); extremes-first at k = 1 0.291 vs 0.231. Per-metal 5th-read values are seed-unstable [C] | a policy crossing {DOPT_Q, BISECT, max-min, POL_ERR-B1} × {B0, B1} on identical queries at 2–5 reads; cluster intervals | separate estimand; no training | the effect is probably ≤ 0.01 at k = 3, but it fixes the reference policy for every budget claim and resolves C4–C7 |
| **8** | **H-UNIT** | A publication / laboratory confirmation unit with chemotype strata is valid, i.e. not anti-conservative | Kish 18.8 (publication) vs 2.24 (cluster); ICC of B1's gain 0.44 (publication) / 0.35 (chemotype); cross-publication residual correlation 0.10–0.14 (E2) vs roughness −0.02 (E6) | a calibrated synthetic study (≥ 50 replicates, nuisance fitted to real B1-residual statistics) of type-I error and power under publication units vs cluster units | statistics; software | decides whether any confirmation is feasible with 5–20 new sources instead of 40–80 |
| **9** | **H-EDGE** | Damped heavy-end slope continuation beyond the last read improves B1 | +0.0055 / +0.0052 (leave-one-cluster-out, E2); +0.0051 / +0.0044 (LOCO, E4); family-shrunk +0.0041 / +0.0049 with 5/5 seeds; half-slope +0.0047 (E5) | a fold-fitted d with inner CV, plus a noise-matched linear-tail control (physics vs slope shrinkage) | BPG / BP; one parameter | the sign is near certain and the size is below every margin. It defines a better reference ("B1+edge") but carries little information |
| **10** | **H-CE** | The Ce deviation (sd 0.244, heavy-tailed) is condition-driven or a Ce(IV)/analytical artefact that can be gated | La/Ce + Ce/Pr = 34.2 % of SSE; ICC 0.09 / 0.03; Spearman −0.34 with acid; 23 / 118 cells with \|dev\| > 0.2 (10 sc009, 7 sc061); reading Ce gives +0.148 on the same rows but −0.012 like for like | a within-publication analysis of Ce deviation vs acid, oxidant and diluent first; only then a BP-scored gate | exploratory first; software | the largest error block, but the prior for a transferable gate is low and it is confounded with publication |
| 11 | H-LOWK | Chemistry priors (G14, B2) or extreme-first policies help at k ≤ 2, where PWL is weak | PWL_R 0.191 at 2 reads; extremes-first at k = 1 +0.06 over DOPT_Q; G14 zero-shot 0.053; never tested in gen21 | a k ladder E-D with priors at k = 0…2 | software | plausible but small user value, since a reader with 2 metals usually has more |
| 12 | H-COND | Condition × series-shape interactions add skill at k = 3 under BP | never tested; gen13 acid slopes heterogeneous (Q 105.8 / 11 df); gen15 acid +0.009, zeroed in 90 % of folds | BP-scored corrector on acid, extractant concentration, diluent | software | low prior; once the BP masks are in place the leakage worry no longer forbids it |
| 13 | H-GD-DONOR | The Gd-dip amplitude is donor- or chemotype-specific in a transferable way | raw non-DGA ≈ 3× DGA; ICC 0.69; all held-out tests ≤ B1; Gd-query Kish 2.33 | new non-DGA Gd-triad series (Sm–Dy minimum) from new laboratories | **data-gated** | high scientific interest; not decidable on about 4 effective chemotypes |
| 14 | H-Y | log D(Y) minus the Ln-radius curve at Y's radius predicts the cell's Gd / tetrad amplitude | 49 Y rows / 4 DOIs | request Y in new sources | data-gated | cheap to request; no test possible now |
| 15 | H-NDORIGIN | The Nd dip is chemistry (sc009) rather than a Pm-gap interpolation artefact or a tetrad boundary | sc009 −0.123 vs +0.004 elsewhere (E1); corpus mean −0.067 (E6); tetrad prediction (E4) | curvature-matched check of the Pr–Sm chord by chemotype | software | low decision value |
| 16 | H-RIGID, H-AN, H-STRAIN, H-ALPHA-DGA | preorganised chemotypes give interior extrema; An(III) helps; composition-constant strain informs; α-DGA steric descriptor | absent from C3; gate 3/50; untested on 60 cells; 2 examples | acquisition, or the 60 composition-constant cells | data-gated / low prior | |

**Top 10 in one line each**
1. H-SIBLING (condition-transfer regime).
2. H-BUDGET (k = 4 on C3).
3. H-CEILING (laboratory repeatability).
4. H-WINDOW (label-free applicability domain).
5. H-FLOOR / V3 (detection floor and provenance).
6. H-METRIC (cluster-balanced or robust co-primary).
7. H-READ (fixed-budget policy crossing).
8. H-UNIT (publication-level confirmation unit).
9. H-EDGE (damped heavy-end continuation).
10. H-CE (Ce deviation gating).

---

## 5. Contradictions and gaps between E1–E6

### 5.1 Contradictions, with resolution status

| # | contradiction | resolution |
|---|---|---|
| C1 | **Dense reads.** E1 §3.3: reading every other metal (hidden ends kept) buys only +0.012 (0.449 → 0.461, one seed). E3 and E6: each extra read is worth +0.03 to +0.05 | **Resolved [C].** E1 compared Gd-free dense PWL with B1. Over 5 seeds, dense PWL is 0.477 and dense D7 **0.504**, against B0 0.418 and HIST_D7 0.451: +0.053 like for like. Both-hidden rows go 0.416 → 0.463. E1's qualitative point stands (at fixed hidden metals R² is capped near 0.50); its "+0.01" does not |
| C2 | **Abstention.** E2: the label-free window gives 0.540 on 75 %. E6: "abstention does not raise R²" | **Resolved [C].** E6 tested difficulty scores (spread, error tables, \|B1\|). The level window is a different score and beats random abstention (0.453 [0.436, 0.468]). Still post hoc (one of 4 windows) |
| C3 | **Publication offsets.** E1 / E2: +0.071 / +0.072. E5: +0.024 | **Resolved [C]: shrinkage.** K = 0 → 0.476; K = 5 → 0.533; K = 10 → 0.524; K = 20 → 0.509. K was chosen on the same rows |
| C4 | **Per-metal value of a 5th read.** E3: Ho, Yb 0.045 top; Lu 0.023. E5: Gd 0.035 top; Lu 0.003; Tm 0.005 | **Partly resolved [C].** E3 reproduces when pooled over 5 seeds. Seed 104729 alone ranks Gd 0.052 and Eu 0.048 first, Lu 0.015. Candidates per metal per seed range from 7 to 144. **Per-metal rankings are seed-unstable.** Do not build a policy on them. E5 also used B1's fold β on seed 0 only |
| C5 | **Gd-aware policy.** E5 recommends it (Gd unread in 39.5 % of cells; the best added read). E6: the hindsight oracle reads Gd *less* than DOPT_Q (38 % vs 67 % on 14-metal cells). E3: Gd-first 0.504 = max-min 0.503. E2: always-Gd +0.028 < random | **Unresolved; weak support for Gd-aware** [I]. B1's Gd term already absorbs the mean dip, so reading Gd mostly removes Gd's idiosyncratic deviation |
| C6 | **Heavy-end reads.** E5: "rarely reading Lu is not a material loss" (Lu 5th read 0.003). E6: the oracle reads Lu in 45 % of cells (DOPT_Q 0 %); targeted reads take Yb/Lu from −0.03 to 0.63 | **Endpoint-dependent [I].** Heavy pairs have small variance (8.6 % of SSE), so pooled R² barely moves. Per-pair skill for Yb/Lu depends entirely on reading Lu or Tm. Pooled R² hides this |
| C7 | **DOPT_Q's next pick.** E2: weak (+0.032 like for like vs max-gap +0.049; a random metal beats it on the same rows). E6, C6 k = 4: DOPT_Q 0.483 ≥ BISECT 0.478 ≈ LEARNED 0.489 | **Unresolved.** E2's 5th read is unrestricted (it may make pairs fully measured); E6's is legal and nested on a different cohort. H-BUDGET and H-READ must compare on identical legal queries |
| C8 | **The Nd dip.** E1: an sc009 property (−0.123 vs +0.004). E6: possibly a Pm-gap interpolation artefact (−0.067 mean). E4: the convex-tetrad boundary | **Unresolved.** sc009 specificity argues against a universal artefact, but the Pr–Sm chord spans 2 Z and depends on each chemotype's curvature (H-NDORIGIN) |
| C9 | **Cross-publication transfer.** E2: residual correlation 0.10–0.14 for the same extractant across publications. E6: roughness −0.02 (3 extractants); copy R² −0.08; sibling template from other publications +0.015 | **Compatible.** Transfer is about 0 to weak. E2's positive value covers more extractants and includes smooth error. It matters for H-UNIT |
| C10 | **Noise threshold.** E3: 0.50 is impossible if σ_SF ≥ 0.157 (floor = σ_SF²). E2's floor model (floor MSE 1.33 σ_metal²) | **Model-dependent [C arithmetic]:** about 0.157–0.19. Read-noise propagation through interpolation would lower it. Set the campaign's decision threshold by simulation with the actual policy |
| C11 | **The Gd coefficient.** E4 calls −0.072 "B1's shrunk coefficient" | **Wording only.** B1's fold β is −0.094 ± 0.015 (E1). −0.072 is HIST_D7. The full-information β is −0.118 (E5); the chord is −0.115. R² is flat across them |
| C12 | **"0.50 needs 6 metals"** (round 3) vs "≈ 5 metals" (E3) | **Cohort.** Round 3's figure is C6 (harder; R23 0.529 on C3-only cells). E3's is C3 with a fixed constant and a post hoc rule |
| C13 | **Why S01 was null.** E1: chemotype hold-out removes sc009 as a block. E3: BPG removes the non-DGA evidence (~5 clusters). E4: a pooled scale over-corrects DGA. E5: the class fails the balanced-MAE leg even as an oracle | **Compatible.** Jointly: not identifiable or transferable on this corpus, and unpassable under round 4's margins |
| C14 | **Read frequencies.** Gd 58 / 61 / 67 %; Lu 17 / 13.5 / 12.2 / 0 %; Ce 4 / 4.4 %; Tb 0.5 / 0.8 % | **Denominators differ** (instances, cells, rows, 14-metal cells). Quote the denominator |

### 5.2 Gaps: no evidence file closes these

- **G1. Laboratory repeatability.** No estimate exists. The noise share of B1's MSE is only bounded, in [0.08, 0.87].
- **G2. Sibling template details (H-SIBLING).** Not reported:
  - how several siblings are combined;
  - the condition distance between siblings;
  - copy-group overlap;
  - performance at k = 0–2;
  - which share of the +0.071 publication-offset gain comes from same-extractant siblings rather than other extractants (extractant × publication offsets 0.521 vs publication 0.523 suggests most of it).
- **G3. k = 4 with a fold-refitted β** and DOPT_Q-nested reads on all 5 seeds: not computed. The E3 / [C] numbers use the fixed −0.072.
- **G4. Round-4 slots have not been re-scored** per cluster, per fold type, or with robust metrics; only S02 / S05 / S06 / S11 were split by fold.
- **G5. Ce:** no within-publication analysis of the deviation against conditions (acid, oxidant, diluent).
- **G6. Window rule:** no threshold-sensitivity curve, no k = 4 version, no per-cluster coverage.
- **G7. Synthetic power generator:** not yet calibrated to the real residual. The targets to calibrate on are:
  - adjacent pointwise ρ 0.69;
  - cell-offset share 36 %;
  - per-cluster heterogeneity;
  - the light-end and Ce tails.
- **G8. The "pair-targeted read" endpoint** needs a stated practical scenario in which one end of the pair cannot be measured (E6 §6.2).
- **G9. SAFE delta:** whether its +1,618 records hold independent Ln clusters is unknown and needs the user's export (E3 §1).
- **G10. Process notes.** E4 records one read-only `git status` (a slip against the no-git rule; nothing changed) and one
  WebFetch result that the tool auto-cached outside the repository and that was not opened. No evidence file wrote inside the
  repository except its own output.

---

## 6. Data and measurement levers, with quantities

### 6.1 Measurement budget and placement (C3 unless stated)

| lever | effect | source |
|---|---|---|
| +1 legal read (5 metals), same queries | 0.451 → 0.495 (random) / 0.503 (max-min) / 0.541 [O]; 178 / 229 cells admit one | E3 §6 |
| +2 legal reads (6 metals) | 0.530 (seeds 0.519–0.543) | E3 §6 |
| every other measured metal read, k = 3 hidden ends fixed | D7 **0.504**; PWL 0.477 | [C] |
| C6 ladder (D7, DOPT_Q) k = 1…6 | 0.206 / 0.385 / 0.431 / 0.483 / 0.524 / 0.552; per-cell oracle at k = 6: 0.663 | E6 §2, §5 |
| placement at k = 3 | random 0.354 → DOPT_Q 0.451 → BISECT 0.462; oracles 0.575 / 0.632 | E6 §3.2 |
| k = 1, extremes first vs DOPT_Q | 0.291 vs 0.231 | E6 §3.2 |
| learned adaptive policy | +0.006 (k = 3); +0.035 at C6 k = 6, which moves the 0.50 crossing from k = 5 to about 4.5 | E6 §4 |
| pair-targeted reads (2 reads around one pair, 14-metal cells) | pair R² 0.34–0.87 vs DOPT_Q −0.03…0.80; wins on 12 / 12 pair types (Yb/Lu −0.03 → 0.63) | E6 §6.2 |
| reading Ce | its two pairs (34 % of SSE) become measured: +0.148 on the same rows, −0.012 like for like | E2 §3a |
| [I] cost model | with ICP-MS all 14 Ln come from one experiment. The cost unit is the experiment (extractant × condition); k-metal economics fit literature mining, radiotracers and subset feeds | E3 §6, E6 §7 |

### 6.2 Regime levers: same data, different declared question

| lever | effect | coverage | source |
|---|---|---|---|
| sibling template (same publication and extractant, other condition) | **0.573** vs B1 0.449 | 62 % of rows | E6 §6.8 |
| publication × pair offsets, leave-one-cell-out | 0.476 (K = 0) … 0.533 (K = 5) | all rows | [C] |
| label-free level window [−2, 2] | **0.540** vs random abstention 0.453 | 74.7 % | E2, [C] |
| mid-series pairs only (Sm/Eu…Dy/Ho) | 0.504 | 46 % | E6 §6.1 |
| wider pairs (ΔZ = 2 / Nd/Sm / ΔZ 3–4) | 0.677 / 0.762 / 0.811 | | E6 §6.3 |
| direction / ranking / conformal intervals | 0.775 accuracy; top-1 0.28 vs 0.06; coverage 0.787 / 0.895 | | E6 §6 |

### 6.3 Data-quality levers: desk work, needs the user

Reported as sensitivities, never as edits to V1 (E3 §4):

| item | rows / seed | B1 SSE share | B0 → B1 without it | what resolves it |
|---|---|---|---|---|
| C5-BTBP publication (floor at log D ≈ −4.2; Nilsson 2006 vs Foreman 2005) | 148 | **23.1 %** | 0.459 → 0.486 | both closed papers (Experimental section, detection limit) |
| of which the 43 light-end rows (+ DMDPhPDA 5 M Pr/Nd row) | 44 | 16.4 % + 1.8 % | 0.447 → 0.486 | as above, plus the Shimada 2004 thesis (open PDF, about 24 MB; download needs permission) |
| ORNL imports without a data location (54 cells; **9 of 13 clusters**) | 529 | 33.3 % | 0.386 → 0.422 | the table or figure for each cell |
| round-3 V2 flags (re-import, duplicate, hidden merge, floor) | 196 | 14.4 % | 0.416 → 0.458 | already defined |
| aqueous-complexant / salt cells (bench-blind; negative Gd gain) | 174 | 10.2 % | 0.410 → 0.453 | condition-aware stratum; no new data |

[I] Three documents decide about 28 % of B1's SSE and about 52 % of its gain. That is the highest leverage per unit of effort
in the programme (E3 §4).

### 6.4 New data: confirmation and ceiling

| lever | quantity | what it decides | source |
|---|---|---|---|
| independent confirmation sources | δ = 0.034 passes with probability 0.04 / 0.10 / 0.135 / 0.27 / 0.34 at G = 5 / 10 / 20 / 40 / 80 (heterogeneous run 4); 0.84 at G = 40 (optimistic) | any model claim under the sealed rule | E3 §2.5, POWER.md |
| supply without a lab | 0–2 clusters (repository) + 3–5 (open-access literature; 2 of 7 candidates are exact C3 extractants: sc086, sc009) | | E3 §2.4 |
| pooling new sources with C3 | pooled Kish only 2.9–14 for G = 20–80; it takes 538 pairs outside the giant cluster to bring its share below 50 % | confirmation must use new data only | E3 §2.5 |
| balanced campaign design | one equal cell per laboratory × chemotype gives Kish = G. Kish failure, not effect size, kills 26–47 % of sets at G ≤ 20 | | E3 §2.6 |
| repeat campaign | about 6 systems × ≥ 3 full La–Lu runs + an inner analytical repeat; threshold σ_SF ≈ 0.157–0.19 | H-CEILING | E3 §5, [C] |
| non-DGA Gd-triad series (Sm–Dy minimum) | Gd-query Kish 2.33 now; about 5 non-DGA clusters carry the dip | H-GD-DONOR, and the generality of B1's gain | E3 §7 |
| Y in new sources | 49 Y rows / 4 DOIs now | H-Y | E4 §2.4 |
| rigid / preorganised chemotypes (macrocycles, phen-diamides, BLPhen) | C3 is 157 / 229 DGA cells; literature shows interior extrema and U-shapes | H-RIGID | E4 §5 |
| SAFE export | ≥ 1,618 records newer than the archive (+9.6 %); independence unknown | supply count | E3 §1 |
| not useful | more rows from existing publications; An / Y / Sc rows as labels; more geometry or descriptors; relaxed cohort keys | | E3 §8 |

### 6.5 Evaluation-design levers

From E5 §5. These are proposals, but every number is measured.
- **One currency.** Margin as relative pair-SSE reduction vs B1 (2 % ≈ +0.011 R²), with balanced MAE as a non-inferiority guard.
  A balanced-MAE superiority margin consistent with +0.01 R² is about 0.0015, not 0.005.
- **BPG decides.** BP is a sensitivity only (same test folds). Report the six-design table (A, B, BR, BQ, BP, BPG) as diagnostics.
- **Cluster-balanced co-primary**, reported per fold type (big fold vs the rest). Consider leave-one-cluster-out folds.
- **Separate registered estimands, never substituted for each other:**
  - E-A continuity;
  - E-C read-set robustness (about 15 % narrower intervals only);
  - E-D k ladder;
  - E-E leave-pair-type-out (only 4.8 % of the Gd feature energy lies outside the Gd pairs);
  - E-F within-source;
  - E-G next-read value.
- **Synthetic power:** ≥ 50 replicates per world (0/50 upper bound 0.07), the sealed rule, and nuisance calibrated to the real
  residual (§5.2 G7). A slot with < 50 % power at its plausible effect runs as a labelled diagnostic, not a decision slot.
- **Training contract:** a pair-difference loss with per-cell weight normalisation; inner selection with a one-SE rule toward the
  reference.

---

## Appendix: reconciliation checks computed for this note [C]

**Method**
- **Inputs:** `round4_20261022/predictions/REF_{BPG,BP}.parquet` (k = 3, DOPT_Q, 5 dev seeds × 1,802 rows) and the V1 bench (`gen20k.data.load_bench()`, manifest-checked).
- **No training.**
- **Reproduction:** the D7-form levels on DOPT_Q reads reproduce the stored `HIST_D7` to 8.9e−16. B0 / B1 / HIST_D7 reproduce 0.4180 / 0.4521 / 0.4511 (BPG) and 0.4179 / 0.4519 / 0.4506 (BP).
- **Runtime:** one process, under 2 minutes each.

**Scripts** (session scratchpad, not in the repository):
- `C:\Users\Bandai\AppData\Local\Temp\claude\C--Users-Bandai--claude-spares-spare-9MsqWO\0cd609d3-ed34-4de3-83c3-904e5f4bdb8b\scratchpad\eb\eb_checks.py`
- `C:\Users\Bandai\AppData\Local\Temp\claude\C--Users-Bandai--claude-spares-spare-9MsqWO\0cd609d3-ed34-4de3-83c3-904e5f4bdb8b\scratchpad\eb\eb_check5.py`

**1. Dense-read oracle.** For each row, the k = 3-unread end(s) stay hidden and every other measured metal of the cell is read.
Levels are I_S[y_S − u_S] + u along −r(CN8), with u_Gd = 0 (PWL) or −0.072 (D7).

| arm | BPG all | one end read | both hidden | BP all |
|---|---|---|---|---|
| B0 | 0.4180 | 0.4367 | 0.3778 | 0.4179 |
| B1 | 0.4521 | 0.4695 | 0.4157 | 0.4519 |
| HIST_D7 | 0.4511 | 0.4675 | 0.4161 | 0.4506 |
| dense PWL | 0.4767 | 0.5024 | 0.4201 | 0.4753 |
| **dense D7** | **0.5042** | 0.5225 | 0.4632 | **0.5032** |

Seed 104729: dense PWL 0.4612. This reproduces E1's 0.461.

**2. Level window vs random abstention** (B1 predicted levels of A and B in [−2, 2]):

| | value |
|---|---|
| coverage per seed | 0.743–0.754 |
| B1 R² inside, per seed | 0.532, 0.521, 0.546, 0.547, 0.553 (mean 0.540); BP 0.538 |
| outside | 0.301 |
| B0 inside | 0.510 |
| var(y) inside vs all | 0.041 vs 0.049 |
| B1 MSE inside vs all | 0.0189 vs 0.0270 |
| random abstention at equal coverage (500 draws) | mean 0.453, 95 % [0.436, 0.468], max 0.477 (BP 0.452 [0.436, 0.468]) |
| share of out-of-window rows from C5-BTBP | 29 % |
| inside R² without C5-BTBP | 0.541 |

**3. Publication × pair-type offsets of B1's residual**, leave-one-cell-out within seed, shrinkage n/(n + K):

| K | 0 | 5 | 10 | 20 |
|---|---|---|---|---|
| R² (BPG = BP to 3 decimals) | 0.476 | 0.533 | 0.524 | 0.509 |

**4. Per-metal value of one legal 5th read** (D7 form; legal means measured, unread, not a strict neighbour of a read metal;
178 cells per seed admit one). Mean SSE drop per candidate cell:

| | Ho | Yb | Er | Eu | Tm | Gd | Tb | Lu | Sm | Nd | Pr | Dy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pooled over 5 seeds | 0.045 | 0.045 | 0.032 | 0.032 | 0.029 | 0.028 | 0.027 | 0.023 | 0.022 | 0.022 | 0.020 | 0.017 |

Candidates per seed range from 13 to 112 per metal. Seed 104729 alone: Gd 0.052 (n = 21), Eu 0.048 (21), Tm 0.042 (7) … Lu 0.015
(28). The ranking is not stable across seeds.

**5. Noise-threshold arithmetic.** R² = 0.50 requires floor MSE ≤ 0.0246. Under E2's model (1.33 σ_metal², σ_SF = √2 σ_metal)
that gives σ_metal ≤ 0.136, i.e. σ_SF ≤ 0.193. Under E3's single-target model, σ_SF ≤ 0.157.
