# P2: source-level resampling replay of few-cluster paired tests (idea M34)

Written 2026-10-23. Probe P2 of the round-5 design.

**Status: EXPLORATORY.** Every number comes from the round-4 out-of-fold (OOF) parquets of the 229 C3 cells (V1, k = 3, policy DOPT_Q, 5 dev seeds). Those cells have already been seen. The replay calibrates tests and plans the number of new sources G for **new** data. It confirms nothing, and no dev p-value below is evidence for a claim.

## Headline

1. **Size.** The exact tests keep their size under both gating nulls: the sign test, the sign-flip on the mean log SSE ratio and the sign-flip on pooled ΔSSE.
2. **Anti-conservative tests.**
   - The sealed jackknife-t leg with df = G − 1 is anti-conservative: mean size 0.050 and maximum 0.124 at nominal 0.025.
   - The Kish-df jackknife-t drifts to 0.04–0.067 once G ≥ 20.
3. **G for 80 % power, B1 − B0.**
   - Equal-weight log-ratio tests: about 19–20 publication-like new sources at τ = 0, and 27–28 at τ = 0.065.
   - Every pooled test, including the sealed R² leg: 37–44 at τ = 0, and more than 80 at τ = 0.065.
   - Chemotype-like units and publication × chemotype units (the closest proxy for the declared laboratory × chemotype unit): about 58–70 at τ = 0 with the sign test and the pooled tests; never within 80 with the equal-weight log-ratio tests.
4. **Kill criterion.** The M34 kill criterion fires at τ = 0.065. At G = 8 with publication units and one-sided 0.025, the best power is 0.24 (BPG) and 0.28 (BP), both below 0.30. It does not fire at τ = 0, where the log-ratio tests reach 0.35–0.36.

## 1. What was run

### Code and outputs

- **Scripts** (all in `D:\ml_separator_gh\generations\gen21\round5_design_20261023\probes\scripts\`):
  - `P2_power_replay.py` is the replay. It was run with `--design BPG` (1,014 s) and `--design BP` (795 s), one process at a time, with `-B` so that no bytecode was written into the round-4 tree.
  - `P2_summarise.py` and `P2_report_tables.py` build the tables.
  - `P2_explore.py` ran the pre-checks.
- **Outputs** (in `...\probes\P2_out\`):
  - `P2_unit_stats_<design>.csv`: per-unit ΔSSE, log SSE ratio and win indicator, for every contrast and unit type;
  - `P2_dev_tests_<design>.csv`;
  - `P2_grid_<design>.csv`: every replay cell, with rejection rates and Clopper–Pearson (CP) 95 % bounds;
  - `P2_G80.csv`;
  - `P2_report_tables.md`: the full tables;
  - `P2_summary.txt`;
  - `P2_checks_<design>.json`;
  - the run logs.
- **Inputs** (read-only):
  - `round4_20261022\predictions\{REF,S01,S02,S05,S06,S11}_{BPG,BP}.parquet`. Slots were merged on (split_seed, cell_id, read_set, A, B, draw, k, policy). The slot files' B1 and y were verified bit-identical to REF.
  - Clusters come from `r4.metrics.add_clusters`.
  - Wild bootstrap weights and the check reference come from `gen13_separation\gen13sep\wildcluster.py`.

### Rows and units

- **Rows:** C3 at k = 3, 1,802 rows per seed × 5 seeds.
- **Units, with their Kish effective number (row shares):**

  | unit | count | Kish |
  |---|---|---|
  | `cluster_cpg` | 13 | 2.24 (largest holds 65 % of rows) |
  | publication | 39 | 18.8 |
  | chemotype | 38 | 3.85 |
  | publication × chemotype (extension; the closest available proxy for the declared laboratory × chemotype unit, since the corpus has no laboratory key) | 64 | 23.1 |

### Per-unit statistics

Seed means of n rows, SSE_ref, SSE_cand, Σy and Σy². From these:

- ΔSSE = SSE_ref − SSE_cand;
- L = log(SSE_ref / SSE_cand);
- win = ΔSSE > 0, and a tie is an exact 0.

A positive value always favours the candidate.

### Pools (candidate − reference)

| pool | role |
|---|---|
| B1 − B0 | positive control |
| B1s50 − B0 | B0 + 0.5 (B1 − B0) |
| B1s2 − B0 | B0 + s (B1 − B0), with s = 0.208 (BPG) / 0.207 (BP), solved so that the pooled dev relative SSE reduction is 2 %, the M35 gate size |
| HBK@ML − B0 | the estimand-mismatch case from IDEAS_RANKED |
| HIST_D7 − B1 | negative control |
| S01, S02, S05, S06, S11 − B1 | round-4 slots |
| B1 − B1 | negative control |

### Worlds

G new units are drawn uniformly with replacement from a pool's units, for G ∈ {5, 8, 12, 20, 40} plus 80 as an extension.

| world | definition | role |
|---|---|---|
| EMP | the drawn units' real pairs | power |
| SWAP (arm-swap null) | each drawn unit's arms exchanged with probability ½; applied to all 10 real pools, so it keeps each pool's unit sizes and heavy tails | gating |
| SYM (symmetric null) | B1 against B1, both arms perturbed independently (τ 0.065 and 0.13) | gating |
| C-LOG | B1 − B0 pool, log ratios centred on their pool mean | skewed heterogeneous null; reported, **not gating** |
| C-MED | B1 − B0 pool, log ratios centred on their median | as above |
| C-POOL | B1 − B0 pool, ΔSSE minus n × the pooled per-row ΔSSE (zero pooled effect; small units gain, the giant unit loses) | as above |

### Per-source noise

- τ ∈ {0, 0.065}.
- Each arm's SSE is multiplied by (1 − κ)², with κ ~ N(0, τ/√2), independently per arm and per unit. The contrast's relative-SSE noise therefore has SD ≈ 2τ = 0.13, which is round 4's calibration (POWER.md).
- **Deviation from round 4,** which perturbed the candidate only: one-arm noise is not arm-swap symmetric. It gives the log ratio a mean of +τ² ≈ +0.004 per unit, which would inflate the size of the log-ratio tests at large G.

### Replicates

- 4,000 per SWAP/SYM cell for the B1 − B0, HIST_D7 − B1 and B1 − B1 pools.
- 2,000 for every other cell.
- 4,999 random sign flips when G > 12 (exact enumeration when G ≤ 12).
- 1,999 Webb draws for WCR.
- 999 cluster-bootstrap draws for the sealed rule.

### Seeds

- `numpy.default_rng(20261023 + crc32(cell key))`.
- The flip, Webb and bootstrap matrices are drawn once per cell and shared by that cell's replicates.

### Tests

All tests are one-sided (H1: the candidate is better). Alpha is 0.025 (primary) and 0.05 (sensitivity).

| id | test | estimand |
|---|---|---|
| SIGN | exact binomial sign test on per-unit wins, ties dropped | π |
| SF_LOG | sign-flip on the mean log SSE ratio; a sharp-null test | equal-weight |
| WCR_LOG | WCR bootstrap-t, CR2, Webb weights, on the per-unit L (one observation per unit) | equal-weight |
| T_LOG | one-sample t on L (= IM-t = jackknife-t with Kish = G) | equal-weight |
| IM_REL | IM-t on per-unit relative SSE, 1 − SSE_c/SSE_r | equal-weight (different) |
| SF_POOL | sign-flip on the pooled ΔSSE | pooled |
| WCR_POOL | WCR, CR2, Webb weights, on the per-row ΔSSE clustered by unit | pooled |
| JK_KISH | cluster-jackknife t on the pooled relative SSE reduction, df = max(Kish_rows − 1, 1) | pooled |
| JK_G1 | the same with df = G − 1, i.e. the sealed jackknife-t leg | pooled |
| SEALED | the sealed rule's **R² leg only**: percentile, BCa and jk-t(G − 1) lower bounds of ΔR² all > 0, and Kish ≥ 3 (95 % intervals for 0.025, 90 % for 0.05) | pooled ΔR² |

SEALED was run only on the B1 − B0, HIST_D7 − B1 and B1 − B1 pools. The balanced-MAE leg cannot be computed from per-unit SSE, so it was not run.

### Size gate

A test is **anti-conservative** if any SWAP or SYM cell has a CP-95 % lower bound on its rejection rate above 0.035 at nominal 0.025, or above 0.07 at nominal 0.05.

### Implementation checks (`P2_checks_*.json`)

- The vectorised one-sided WCR, run two-sided with the same Webb draws, reproduces `wildcluster.wild_cluster_t` exactly: p 0.162 (BPG) and 0.1605 (BP) for B1 − B0 by cpg.
- The exact sign-flip matches brute force.
- The sign test matches `scipy.binomtest`.

## 2. Dev contrasts (EXPLORATORY: already-seen cells, not evidence)

BPG, one-sided p-values. "W/L/T" counts units won, lost and tied.

| contrast · unit | W/L/T | mean L (non-tied) | pooled rel. SSE | SIGN | SF_LOG | WCR_LOG | SF_POOL | WCR_POOL | JK_KISH | JK_G1 |
|---|---|---|---|---|---|---|---|---|---|---|
| B1−B0 · cpg | 11/1/1 | +0.160 | +5.85 % | 0.0032 | 0.0071 | 0.0055 | 0.0032 | 0.088 | 0.128 | 0.035 |
| B1−B0 · pub | 31/7/1 | +0.227 | | 0.0001 | 0.0001 | 0.0005 | 0.0053 | 0.0050 | 0.0021 | 0.0011 |
| B1−B0 · chem | 24/10/4 | +0.059 | | 0.012 | 0.256 | 0.257 | 0.0013 | 0.021 | 0.056 | 0.014 |
| B1−B0 · pub×chem | 42/18/4 | +0.215 | | 0.0013 | 0.013 | 0.011 | 0.0050 | 0.011 | 0.0017 | 0.0008 |
| HBK@ML−B0 · cpg | 8/5/0 | +0.082 | +0.14 % | 0.29 | 0.010 | 0.012 | 0.46 | 0.46 | 0.47 | 0.47 |
| HBK@ML−B0 · pub | 21/18/0 | +0.060 | | 0.37 | 0.017 | 0.019 | 0.44 | 0.46 | 0.45 | 0.45 |
| HIST_D7−B1 (all units) | | | −0.19 % | ≥ 0.43 | ≥ 0.21 | ≥ 0.21 | ≥ 0.62 | ≥ 0.62 | ≥ 0.56 | ≥ 0.58 |
| S01 / S02 / S05 / S06 / S11 − B1 (all units) | | | −0.02 % to −4.4 % | ≥ 0.13 | ≥ 0.13 | ≥ 0.12 | ≥ 0.53 | ≥ 0.51 | ≥ 0.52 | ≥ 0.52 |

**Reading the table.**
- The evidence-base numbers are reproduced:
  - 11 of 12 non-tied clusters;
  - pooled sign-flip p 0.0032;
  - log-ratio p 0.007;
  - publications 31 of 38, p 0.0001;
  - chemotypes p 0.26;
  - IM-t p 0.012 (IM_REL, cpg).
- BP agrees: B1 − B0 cpg mean L is +0.173, and the p-values are within ±0.005. BP's HBK@ML − B0 is pooled −0.64 %, yet still has equal-weight p 0.014 by cpg.
- At cpg level the two few-cluster comparators do not reject the known gain (WCR_POOL 0.088, JK_KISH 0.128), while the exact SF_POOL gives 0.0032.
- The verdict changes with both the unit and the estimand.

## 3. Size under the gating nulls

The table covers 10 SWAP pools plus SYM, τ 0 / 0.065 (SYM 0.065 / 0.13), G 5–80, and both designs. "Flagged" counts cells with a CP-95 % lower bound above 0.035.

| test | max at 0.025 (cpg / pub / chem / pub×chem) | flagged / cells | mean | max at 0.05 [cells with CP > 0.07] |
|---|---|---|---|---|
| SIGN | 0.029 / 0.027 / 0.027 / 0.027 | 0 / 1,056 | 0.012 | 0.063 [0] |
| SF_LOG | 0.034 / 0.037 / 0.035 / 0.033 | 0 / 1,056 | 0.020 | 0.063 [0] |
| SF_POOL | 0.037 / 0.033 / 0.034 / 0.036 | 0 / 1,056 | 0.020 | 0.065 [0] |
| T_LOG | 0.034 / 0.035 / 0.035 / 0.033 | 0 / 1,056 | 0.020 | 0.062 [0] |
| IM_REL | 0.036 / 0.032 / 0.038 / 0.033 | 0 / 1,056 | 0.014 | 0.061 [0] |
| WCR_POOL | 0.045 / 0.035 / 0.040 / 0.035 | 1 / 1,056 | 0.022 | 0.074 [0] |
| WCR_LOG | 0.054 / 0.040 / 0.073 / 0.113 | 6 / 1,056 | 0.025 | 0.117 [2] |
| SEALED (R² leg) | 0.040 / 0.038 / 0.045 / 0.038 | 2 / 288 | 0.025 | 0.074 [0] |
| JK_KISH | 0.051 / 0.052 / 0.067 / 0.059 | **57 / 1,056** | 0.027 | 0.100 [8] |
| JK_G1 (sealed jk-t leg) | 0.124 / 0.070 / 0.104 / 0.072 | **705 / 1,056** | **0.050** | 0.148 [468] |

**Anti-conservative tests.**

1. **JK_G1** (the sealed rule's jackknife-t leg alone).
   - Size 0.04–0.12 at nominal 0.025 almost everywhere, worst at G ≤ 20. Mean size by G: 0.053 / 0.055 / 0.055 / 0.052 / 0.045 / 0.037.
   - This confirms POWER.md's undercoverage note.
2. **JK_KISH.** The Kish df fixes small G (mean 0.021 at G = 5), but the size rises with G: mean 0.030 at G = 40; publication and pub×chem cells reach 0.042–0.050 at G = 20–80 with τ = 0.
   - Breakdown of the 57 flags:
     - by G: 1 at G = 8, 8 at G = 12, 20 at G = 20, 16 at G = 40, 12 at G = 80;
     - by unit: 27 pub×chem, 15 publication, 9 cpg, 6 chemotype.
   - The cause is the heavy-tailed ratio estimator: once Kish is large, the df correction no longer bites.
3. **WCR_LOG.** All 6 flags are in the S11 − B1 pool at τ = 0. There 85–90 % of units are exact ties (S11 reproduced B1), and the size reaches 0.113 (pub×chem, G = 12). This is a degenerate sparse-difference world, but a real candidate that differs from B1 in only a few sources is exactly that world.
4. **WCR_POOL.** One flag: S11, cpg, G = 5, size 0.045.
5. **SEALED.** Two flags: chemotype HIST_D7, τ = 0, G = 40, size 0.046 (BPG) and 0.041 (BP).

**Tests with nominal size.** SIGN, SF_LOG, SF_POOL, T_LOG and IM_REL keep nominal size in every gating cell.
- SIGN is conservative (mean 0.012).
- At G = 5 the SIGN and SF tests cannot reject at 0.025, because the minimum p is 1/32 = 0.031. Their observed rejection rate was exactly 0.

**Skewed heterogeneous nulls (not gating).** No test holds 0.025 at cpg or chemotype granularity once a zero-mean null is skewed:
- SF_LOG under C-LOG, chemotype: 0.111;
- WCR_LOG: 0.110;
- the pooled tests under C-POOL, cpg: SF_POOL 0.103, WCR_POOL 0.104, JK_KISH 0.125, JK_G1 0.150;
- SIGN under C-MED: ≤ 0.022 at τ = 0, but up to 0.106 at τ = 0.065, publication, G = 80, because the noise moves the median of a skewed pool.

Publication-level skewed nulls stay ≤ 0.041 for every test in its own null. The exactness of SIGN and SF covers sharp symmetric nulls only, as Skeptic A said.

## 4. Power for the positive control B1 − B0 (EMP world)

One-sided 0.025, BPG. BP agrees within ±0.07 in every cell (full tables in `P2_out\P2_report_tables.md`, T3).

| test | cpg G8 / G20 (τ 0 · 0.065) | pub G8 / G20 (τ 0 · 0.065) | chem G20 / G80 (τ 0 · 0.065) | pub×chem G20 / G80 (τ 0 · 0.065) |
|---|---|---|---|---|
| SIGN | 0.51 / 0.98 · 0.12 / 0.66 | 0.21 / 0.81 · 0.05 / 0.40 | 0.34 / 0.93 · 0.28 / 0.78 | 0.34 / 0.93 · 0.25 / 0.72 |
| SF_LOG | 0.53 / 0.92 · 0.35 / 0.79 | 0.35 / 0.81 · 0.23 / 0.66 | 0.17 / 0.20 · 0.15 / 0.18 | 0.32 / 0.77 · 0.30 / 0.73 |
| WCR_LOG | 0.55 / 0.92 · 0.38 / 0.79 | 0.36 / 0.81 · 0.24 / 0.65 | 0.17 / 0.19 · 0.15 / 0.19 | 0.31 / 0.77 · 0.30 / 0.74 |
| SF_POOL | 0.61 / 0.98 · 0.23 / 0.30 | 0.29 / 0.48 · 0.12 / 0.20 | 0.59 / 1.00 · 0.25 / 0.41 | 0.38 / 0.87 · 0.17 / 0.41 |
| WCR_POOL | 0.34 / 0.73 · 0.14 / 0.22 | 0.27 / 0.49 · 0.12 / 0.19 | 0.39 / 1.00 · 0.20 / 0.36 | 0.40 / 0.85 · 0.16 / 0.40 |
| JK_KISH | 0.22 / 0.62 · 0.09 / 0.15 | 0.13 / 0.53 · 0.09 / 0.19 | 0.28 / 0.99 · 0.16 / 0.31 | 0.34 / 0.89 · 0.17 / 0.41 |
| SEALED | 0.23 / 0.81 · 0.08 / 0.21 | 0.19 / 0.51 · 0.10 / 0.19 | 0.38 / 1.00 · 0.17 / 0.34 | 0.37 / 0.88 · 0.15 / 0.39 |

### G needed for 80 % power

Log-G interpolation on the grid 5 / 8 / 12 / 20 / 40 / 80. One-sided 0.025, BPG, entries are τ = 0 / τ = 0.065. BP is within ±4 of BPG in every cell (`P2_out\P2_report_tables.md`, T4).

| test | cpg | publication | chemotype | pub × chemotype |
|---|---|---|---|---|
| SIGN | 11 / 29 | 19 / 53 | 58 / > 80 | 59 / > 80 |
| SF_LOG | 15 / 21 | 19 / 28 | > 80 / > 80 | > 80 / > 80 |
| WCR_LOG | 15 / 21 | 20 / 28 | > 80 / > 80 | > 80 / > 80 |
| T_LOG | 15 / 21 | 34 / 38 | > 80 / > 80 | > 80 / > 80 |
| SF_POOL | 11 / > 80 | 43 / > 80 | 30 / > 80 | 69 / > 80 |
| WCR_POOL | 24 / > 80 | 44 / > 80 | 40 / > 80 | 72 / > 80 |
| JK_KISH | 28 / > 80 | 37 / > 80 | 58 / > 80 | 63 / > 80 |
| SEALED (R² leg) | 20 / > 80 | 39 / > 80 | 41 / > 80 | 66 / > 80 |

At one-sided 0.05 (sensitivity): publication SF_LOG needs 16 / 20, SIGN 18 / 40, SF_POOL 33 / > 80 and SEALED 32 / > 80. Chemotype SF_LOG stays above 80.

### Smaller effects

| effect | τ = 0 (publication) | τ = 0.065 |
|---|---|---|
| B1s2 − B0 (2 % pooled relative SSE ≈ the M35 gate) | SF_LOG 14, SF_POOL 17, JK_KISH 18 (consistency is high) | > 80 for every test and unit, except publication SF_LOG / WCR_LOG / T_LOG at 74–77 |
| B1s50 − B0 | | publication SF_LOG 30; every pooled test > 80 |

So effect *consistency*, not size, drives power at τ = 0. A 2 % effect drowns in per-source noise of 0.13 relative SD.

### Kill criterion (IDEAS_RANKED M34)

Power at G = 8, publication units, one-sided 0.025, B1 − B0.
- **τ = 0.065:** BPG maximum 0.24 (WCR_LOG; SF_LOG 0.23; SIGN 0.05); BP maximum 0.28. All are below 0.30, so **the kill fires**: keep the 40–80-source planning number and run round-5 model contrasts as diagnostics.
- **τ = 0:** SF_LOG 0.35, WCR_LOG 0.36, SF_POOL 0.29, SIGN 0.21, SEALED 0.19. Only the two equal-weight log-ratio tests clear 0.30.
- The verdict therefore depends on τ, and τ is a declared choice, not a measurement.

### Extensions (BPG)

- **Per-seed pools.** Drawing unit × seed instead of the seed mean leaves the log-ratio tests unchanged: cpg 15 / 21, publication 20 / 28. SIGN worsens at publication level (29 / 59).
- **DGA strata (publication units).**

  | stratum | units | SF_LOG | SIGN | SF_POOL | other pooled tests |
  |---|---|---|---|---|---|
  | non-DGA | 26 | 12 / 17 | 11 / 27 | 16 / 80 | > 80 at τ 0.065 |
  | DGA (sc009-majority) | 13 | 62 / 80 | > 80 | > 80 | > 80 |

  The positive control's power lives in non-DGA sources. A DGA-heavy confirmation set would essentially not confirm B1.

## 5. Estimand mismatch: equal-weight tests confirm pooled nulls and losers

Rejection rate at 0.025 in the EMP world, BPG, G = 20 / 40:

| contrast (pooled relative SSE) | units | equal-weight tests | pooled tests |
|---|---|---|---|
| HBK@ML − B0 (+0.14 %; BP −0.64 %) | cpg | SF_LOG 0.93 / 1.00 (τ 0), 0.51 / 0.82 (τ 0.065) | SF_POOL / WCR_POOL / JK_KISH ≤ 0.05 |
| HBK@ML − B0 | publication | SF_LOG 0.27 / 0.60 (τ 0), 0.19 / 0.36 (τ 0.065) | ≤ 0.05 |
| S02 − B1 (−4.4 %) | cpg | SIGN 0.39 / 0.67, SF_LOG 0.20 / 0.32 (τ 0) | ≤ 0.05 |
| S01 − B1 (−0.02 %) | cpg | SF_LOG 0.28 / 0.64 (τ 0) | ≤ 0.08 |
| HIST_D7 − B1 (negative control) | all units | ≤ 0.045, except chemotype SF_LOG / WCR_LOG 0.07 at G = 40 (its chemotype mean L is +0.020) | ≤ 0.04 |
| S05, S06, S11 − B1 | all units | ≤ 0.06 | ≤ 0.12 (S06 cpg τ 0, G 20) |

With cpg-like sources, equal-weight tests reward gains in the 1–24-row clusters, mostly ORNL-only imports. In the dev data, the giant cluster's own B1 − B0 log ratio is only +0.032. An equal-weight test therefore answers "does the average source improve?". It must never stand in for E-A, and it needs a pooled magnitude guard (the M35 2 % gate) beside it.

## 6. What this means for sealing (design input for M34/M35, not a decision)

- **Pooled estimand (E-A / pooled relative SSE versus B1).**
  - SF_POOL (exact; never flagged) or WCR_POOL (1 flag) can serve as the test.
  - Drop JK_G1. Do not use JK_KISH at G ≥ 20.
  - Planning numbers:
    - publication-like sources: 37–44 at τ = 0 and more than 80 at τ = 0.065, for a B1-sized gain;
    - chemotype-like and pub×chem-like sources: 30–72 at τ = 0.
  - This is consistent with POWER.md's "40, possibly well over 80".
- **Equal-weight breadth estimand (a separate E-S, never E-A).**
  - SF_LOG = WCR_LOG ≈ T_LOG at cpg level. SF_LOG is the cheapest exact choice.
  - Planning numbers: 19–20 / 27–28 publication-like sources, 15 / 21 cpg-like.
  - With chemotype-like or pub×chem-like sources it never reaches 80 % within 80 units.
- **SIGN (estimand π).** Exact and conservative, but the most noise-sensitive: publication 19 → 53 from τ = 0 to τ = 0.065.
- **G floor.** G ≤ 5 cannot reject at 0.025 with SIGN or SF; declare this in advance.
- **Declared unit.** If the unit is laboratory × chemotype and new units resemble the corpus's publication × chemotype units, no test reaches 80 % below about 59 units even at τ = 0.

## 7. What would change these conclusions

1. **τ calibration.** τ = 0.065 applies the same relative noise (SD 0.13) to a 1,170-row source as to a 10-row one.
   - If per-source contrast noise shrinks with source size, the pooled tests' G80 moves from above 80 toward the τ = 0 column (37–44 publication-like).
   - Extra heterogeneity from genuinely new chemistry would make every number worse.
2. **What new sources look like.**
   - Non-DGA-like sources: 11–17 publication-like sources suffice for SF_LOG / SIGN.
   - DGA-like sources: more than 60.
   - The replay assumes new units are exchangeable with the existing units.
3. **Independence.** The replay draws independent units. Shared laboratories or layouts between new sources break the exactness of SIGN and SF; M36 found 0.07–0.18 for publication flips under shared layouts.
4. **Estimand choice.** It moves G by a factor of 2 to more than 4 (section 4).
5. **Monte Carlo precision.**
   - Power has an MC SE ≤ 0.011 (2,000 replicates); size has MC SE ≈ 0.0025.
   - G80 is interpolated on a coarse grid, so read it as ±15–20 %.
   - The size flags for JK_KISH and WCR_LOG sit near the threshold in many cells. With 1,056 cells, a few flags are expected by chance for a test whose true size is about 0.035.
6. **The sealed rule is represented by its R² leg only.** Adding the balanced-MAE leg (POWER.md) can only lower its power.

## 8. Caveats

- **Exploratory.** Everything here comes from the 229 already-seen C3 cells. The dev p-values are sensitivities, not evidence.
- **Small resampling universe.** cpg has only 13 units, so G = 40–80 draws repeat them many times. The world inherits the corpus's heavy-tailed error structure, including C5-BTBP (pub_6a0ef8528c), which no sensitivity run here removes.
- **Seed-mean statistics.** Per-unit statistics are seed means over 5 dev seeds with seed-dependent read sets. A real confirmation scores each source once (the per-seed extension shows the effect is small for the log tests).
- **Different nulls from POWER.md.** The SWAP and SYM nulls are symmetric by construction, and POWER.md's boundary null is not. That is why the sealed R² leg holds size here (≤ 0.045) while POWER.md found 5–8 %.
- **Noise form.** Per-arm noise was used, not round 4's candidate-only form (section 1).
- **Unit proxy.** publication × chemotype stands in for laboratory × chemotype.
- **BP and BPG are not independent:** they share test folds.
