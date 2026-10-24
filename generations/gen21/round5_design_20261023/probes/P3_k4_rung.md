# P3: the k = 4 rung (5 read metals) on identical C3 queries

Probe for round-5 ideas **H-BUDGET**, **M18** (density-indexed metal ladder) and **M17** (read-policy envelope). Written 2026-10-23.

> **Status: EXPLORATORY.** Every number below comes from the 229 C3 cells and the 5 dev split seeds. All of these have been seen before.
> - E3 §6, EVIDENCE_BASE (H-BUDGET row), and the M17/M18 critiques already quoted values within ±0.005 of the main results here.
> - So nothing in this file is a test. None of it may carry decision language for round 5.
> - The final seeds were not touched. Nothing outside `probes/` was written, apart from the scratchpad file listed in §12. No git operation changed anything (§12).

---

## Headline

**With 5 read metals (k = 4), B1 on the identical 1,802 C3 queries sits exactly on 0.50 under every label-free 5th-read policy.**
- BPG: 0.4945–0.5036. BP: 0.4953–0.5049.
- Seeds at or above 0.50: 1–4 of 5. Cluster-bootstrap P(R² ≥ 0.50): 0.36–0.57.

**The like-for-like fixed cohort C4 (cells that can take a 5th read) does not cross.** It scores 0.477–0.489, with 0 of 5 seeds for every policy except one seed of HULL_BISECT under BP.

**The crossing is decided by one publication, C5-BTBP (pub_6a0ef8528c).** Without it, every policy crosses: C3 0.530–0.542 (5/5 seeds) and C4 0.515–0.530 (4–5/5).

Two other levers barely move the answer:
- The policy changes R² by about 0.009. That envelope is smaller than the round-5 MDE of about 0.011.
- Refitting β for k = 4 changes R² by at most 0.001.

---

## 1. What was computed

### Rows
- All BPG and BP outer folds: 5 dev seeds (104729, 130363, 155921, 196613, 262147) × 5 folds.
- They come from the round-4 fold cache (`g21.cache.load`).
- Each seed holds the same 229 C3 cells (k_max ≥ 3) and the same 1,802 strict-adjacent query rows.
- The query set is byte-identical at k = 3 and at k = 4 under every policy. No query ever gets both ends read.

### Rungs
- **k = 3:** the cached DOPT_Q read set (4 metals).
- **k = 4:** that set plus one 5th metal.

There are two cohort rows:
- **C3 population row** (M18's "C3 min(K, k_max)"). The 51 cells with k_max = 3 (286 rows per seed) cannot take a 5th read, so they keep their k = 3 prediction.
- **C4 fixed cohort.** These are the 178 cells with k_max ≥ 4 (1,516 rows per seed), scored at both rungs on identical rows.

### Legality rule
This is the protocol's own rule, `gen20k.design.legal_metals`. A 5th metal must meet all of the following:
- it is measured in the cell and not yet read;
- it is not strict-adjacent (ΔZ = 1) to any read metal, so no query pair becomes fully measured;
- it keeps the maximum independent set reachable (gen20's look-ahead).

The same rule generated DOPT_Q's own nested 4th step. How many cells it leaves room for:
- 178 of 229 cells admit a legal 5th read;
- there are on average 2.79 candidates per cell (range 1–5);
- k_max = 4 cells have exactly one candidate in 169 of 170 cell-instances, so the policies coincide there (their k_max-4 R² is identical to 3 decimals);
- policy freedom exists only in the 144 k_max = 5–6 cells per seed.

**Sensitivity rule "LOOSE"** (E3 §6's rule: the same, but without the look-ahead). It adds candidates in 80 cells per seed (mean 3.24 per cell).

### Predictors
- **B0:** PWL_R.
- **HIST_D7:** fixed Gd potential −0.072. This is a diagnostic and is not fold-independent.
- **B1:** refit per outer fold with `r4.refs.fit_reference(bench, rec, Y, "B1", k=K)`.
  - At k = 3 the refit reproduces the stored REF β and λ exactly in 50 of 50 folds (max |Δβ| = 0).
  - At k = 4 the same function is called with `k=4`. Everything else in refs.py is unchanged:
    - training masks are nested DOPT_Q, UNIFORM and 4 seeded RANDOM legal sets at budget 4;
    - training cells are those with k_max ≥ 4;
    - the inner-block DOPT_Q is recomputed with the inner covariance;
    - λ comes from the round-3 grid plus ∞, with the orphan rule.
- **Stated deviation.** As in refs.py, β is fitted on that standard training-mask mixture, not on each test policy's own 5th reads. No per-policy refit was run (M17's design asks for one). The column `B1_k3beta`, which applies the k = 3 β at k = 4, bounds the effect: it is −0.0006 (BPG) and −0.0009 (BP) in R², so the deviation is immaterial.
- **Fitted values.**
  - β at k = 4: mean −0.089, sd 0.025, range −0.130 to −0.024.
  - β at k = 3: mean −0.092 (BPG) / −0.095 (BP).
  - λ at k = 4: 0.03 in 20 folds, 0.003 in 3, 0.3 in 2 (BPG); 19 / 4 / 2 (BP). It is never ∞.

### 5th-read policies
All are label-free. They take only the metal identities, the 4 reads and the legal set as input.

| policy | rule |
|---|---|
| **DOPTQ** (DOPT_Q-nested) | The cache's own 4th support step, i.e. the policy's own next pick. |
| **MAXMIN_R** | Farthest point in −CN8 radius: the legal metal whose distance to the nearest read is largest. Ties go to the lower Z (gen20 UNIFORM convention). MAXMIN_Z uses Z instead; it picks the same metal in 95 % of cells. |
| **BISECT_R** | Take the segment with the most unread measured metals that contains a legal metal. Segments are the gaps between consecutive reads plus the two end segments beyond the extreme reads; ties go to the lighter segment. Within it, read the legal metal nearest the target. The target is the radius midpoint for an inner gap, and the cell's extreme metal for an end segment (hull closure, as BISECT's own first two reads). BISECT_Z picks identically. **BISECT_RMID** uses the midpoint toward the extreme instead. **HULL_BISECT** first reads the legal metal nearest an unread extreme (heavy end first), and otherwise falls back to BISECT_R. |
| **RANDOM** | Uniform over the legal set, independent across cells. Because SST is fixed and β does not depend on the test reads, the exact expectation of the pooled SSE gives the expected per-seed R². A Monte Carlo run (2,000 joint draws, rng 20261023) gives the distribution. |
| **ORACLE_B1** [O] | The legal 5th metal that minimises the cell's B1 SSE. It uses labels and serves as a bound only. |

These rules were written for this probe. The BISECT adaptation to a 5th read and both tie-breaks are my choices. The variants show how sensitive the results are to them.

### Metrics
- **R²:** round 4's primary. Pooled strict-adjacent R² within each seed, then the seed mean.
- **"Seeds ≥ 0.50"** counts seeds. The seeds share the same cells, so they are not independent replicates.
- **P_boot:** the fraction of 2,000 cluster-bootstrap draws in which the seed-mean R² is ≥ 0.50. Clusters are round 4's cluster_cpg (13 clusters; shared draw matrix from `gen20k.inference._draw_counts`, seed 8675309).
- **LOCO:** the range of the seed-mean R² when one cluster is left out.
- With 13 clusters and a Kish effective number of about 2.2, the percentile bootstrap is anti-conservative. Treat P_boot as descriptive.

### Identity checks (all passed)
- k = 3 rows reproduce `REF_{BPG,BP}.parquet` exactly: 9,010 of 9,010 rows; y, B0, B1 and HIST_D7 max |diff| = 0; identical read sets.
- k = 4 DOPTQ-nested B0 reproduces `R3-B0_{BPG,BP}.parquet` PWL_R at k = 4 exactly: 7,580 of 7,580 rows, max |diff| = 0, identical read sets.
- The LOOSE rule reproduces E3 §6: MAXMIN_R_LOOSE HIST_D7 is **0.5030** (seeds 0.4985–0.5101, 3/5), against E3's "0.503 (0.499–0.510; 3/5)". RANDOM_LOOSE HIST_D7 is 0.4954, against E3's 0.495.

---

## 2. The k ladder: C3 population row, pooled R² per seed

Seed order: 104729 | 130363 | 155921 | 196613 | 262147.

### BPG

| rung / policy | arm | per-seed R² | mean | ≥ 0.50 | P_boot | boot 95 % | LOCO |
|---|---|---|---|---|---|---|---|
| k=3 DOPT_Q | B0 | 0.4099 / 0.4117 / 0.4264 / 0.4242 / 0.4180 | 0.4180 | 0/5 | 0.01 | [0.328, 0.466] | 0.392–0.447 |
| k=3 DOPT_Q | **B1** | 0.4489 / 0.4375 / 0.4620 / 0.4589 / 0.4532 | **0.4521** | 0/5 | 0.02 | [0.394, 0.497] | 0.443–0.475 |
| k=3 DOPT_Q | HIST_D7 | 0.4439 / 0.4441 / 0.4594 / 0.4570 / 0.4510 | 0.4511 | 0/5 | 0.02 | [0.386, 0.496] | 0.438–0.476 |
| k=4 DOPTQ-nested | B0 | 0.4672 / 0.4545 / 0.4680 / 0.4683 / 0.4657 | 0.4647 | 0/5 | 0.08 | [0.388, 0.513] | 0.451–0.493 |
| k=4 DOPTQ-nested | **B1** | 0.5047 / 0.4786 / 0.5026 / 0.5012 / 0.4967 | **0.4968** | 3/5 | 0.41 | [0.439, 0.552] | 0.486–0.519 |
| k=4 DOPTQ-nested | HIST_D7 | 0.5003 / 0.4834 / 0.5001 / 0.4999 / 0.4954 | 0.4958 | 2/5 | 0.40 | [0.434, 0.553] | 0.484–0.520 |
| k=4 MAXMIN_R | B0 | 0.4515 / 0.4688 / 0.4739 / 0.4779 / 0.4559 | 0.4656 | 0/5 | 0.06 | [0.392, 0.508] | 0.451–0.493 |
| k=4 MAXMIN_R | **B1** | 0.4915 / 0.4967 / 0.5083 / 0.5110 / 0.4931 | **0.5001** | 2/5 | 0.47 | [0.441, 0.550] | 0.493–0.523 |
| k=4 MAXMIN_R | HIST_D7 | 0.4871 / 0.4995 / 0.5058 / 0.5096 / 0.4915 | 0.4987 | 2/5 | 0.43 | [0.436, 0.550] | 0.491–0.523 |
| k=4 BISECT_R | B0 | 0.4643 / 0.4695 / 0.4739 / 0.4779 / 0.4642 | 0.4700 | 0/5 | 0.12 | [0.397, 0.520] | 0.457–0.499 |
| k=4 BISECT_R | **B1** | 0.4997 / 0.4977 / 0.5083 / 0.5110 / 0.4979 | **0.5029** | 2/5 | 0.53 | [0.445, 0.560] | 0.496–0.526 |
| k=4 BISECT_R | HIST_D7 | 0.4957 / 0.5003 / 0.5058 / 0.5096 / 0.4965 | 0.5016 | 3/5 | 0.50 | [0.441, 0.561] | 0.494–0.527 |
| k=4 RANDOM (E) | B0 | 0.4538 / 0.4547 / 0.4767 / 0.4717 / 0.4573 | 0.4628 | 0/5 | 0.06 | [0.389, 0.509] | 0.449–0.491 |
| k=4 RANDOM (E) | **B1** | 0.4889 / 0.4799 / 0.5092 / 0.5034 / 0.4908 | **0.4945** | 2/5* | 0.36 | [0.437, 0.548] | 0.486–0.517 |
| k=4 RANDOM (E) | HIST_D7 | 0.4850 / 0.4856 / 0.5070 / 0.5022 / 0.4895 | 0.4939 | 2/5* | 0.35 | [0.433, 0.550] | 0.485–0.518 |
| k=4 MAXMIN_Z | B1 | 0.4915 / 0.4967 / 0.5083 / 0.5108 / 0.4931 | 0.5001 | 2/5 | 0.46 | | |
| k=4 BISECT_RMID | B1 | 0.4996 / 0.4979 / 0.5099 / 0.5147 / 0.4959 | 0.5036 | 2/5 | 0.55 | [0.449, 0.559] | 0.497–0.525 |
| k=4 HULL_BISECT | B1 | 0.4888 / 0.4950 / 0.5135 / 0.4918 / 0.4974 | 0.4973 | 1/5 | 0.39 | [0.441, 0.543] | 0.490–0.519 |
| k=4 MAXMIN_R_LOOSE | B1 | 0.5023 / 0.4970 / 0.5083 / 0.5114 / 0.5022 | 0.5042 | 4/5 | 0.55 | [0.440, 0.558] | 0.496–0.530 |
| k=4 RANDOM_LOOSE (E) | B1 | 0.4903 / 0.4847 / 0.5087 / 0.5031 / 0.4927 | 0.4959 | 2/5* | 0.39 | | |
| k=4 ORACLE_B1 [O] | B1 | 0.5270 / 0.5188 / 0.5675 / 0.5500 / 0.5215 | 0.5370 | 5/5 | 0.92 | [0.473, 0.614] | 0.528–0.563 |

### BP

| rung / policy | arm | per-seed R² | mean | ≥ 0.50 | P_boot |
|---|---|---|---|---|---|
| k=3 DOPT_Q | B0 / **B1** / HIST_D7 | B1: 0.4533 / 0.4397 / 0.4589 / 0.4581 / 0.4497 | 0.4179 / **0.4519** / 0.4506 | 0/5 | B1 0.03 |
| k=4 DOPTQ-nested | **B1** | 0.5079 / 0.4792 / 0.5035 / 0.5007 / 0.5027 | **0.4988** (B0 0.4661, D7 0.4978) | 4/5 (D7 3/5) | 0.45 |
| k=4 MAXMIN_R | **B1** | 0.4973 / 0.4972 / 0.5114 / 0.5110 / 0.4954 | **0.5025** (B0 0.4689, D7 0.5011) | 2/5 | 0.52 |
| k=4 BISECT_R | **B1** | 0.5018 / 0.4982 / 0.5114 / 0.5110 / 0.4998 | **0.5045** (B0 0.4712, D7 0.5030) | 3/5 | 0.56 |
| k=4 RANDOM (E) | **B1** | 0.4925 / 0.4806 / 0.5108 / 0.5031 / 0.4897 | **0.4953** (B0 0.4638, D7 0.4946) | 2/5* | 0.38 |
| k=4 BISECT_RMID / HULL_BISECT / MAXMIN_R_LOOSE | B1 | | 0.5049 / 0.4995 / 0.5044 | 3/5 / 1/5 / 3/5 | 0.57 / 0.44 / 0.56 |
| k=4 ORACLE_B1 [O] | B1 | 0.5299 / 0.5193 / 0.5616 / 0.5499 / 0.5292 | 0.5380 | 5/5 | 0.92 |

\* For RANDOM, "≥ 0.50" counts seeds whose *expected* R² is ≥ 0.50. The Monte Carlo distribution of the random policy (B1) is:

| design | mean | sd of the seed mean over draws | 90 % range | P(seed mean ≥ 0.50) | P(seed ≥ 0.50), per seed | E[# seeds ≥ 0.50] |
|---|---|---|---|---|---|---|
| BPG | 0.4945 | 0.0025 | 0.4905–0.4987 | 0.018 | 0.02 / 0.00 / 0.91 / 0.75 / 0.02 | 1.70 |
| BP | 0.4953 | 0.0025 | 0.4913–0.4996 | 0.039 | 0.08 / 0.00 / 0.95 / 0.73 / 0.02 | 1.77 |
| BPG without C5-BTBP | 0.5295 | 0.0029 | 0.5250–0.5344 | 1.000 | 1 / 1 / 1 / 1 / 1 | 5.00 |

The random draw itself hardly matters (sd 0.0025). Seed-to-seed variation dominates, and the seeds differ through their fold composition, DOPT_Q's k = 3 reads and β.

### Paired contrasts on identical rows

All are B1. Each gives the seed-mean Δ, the number of seeds with Δ > 0, the cluster-bootstrap 95 % interval and the LOCO range.

| contrast | BPG | BP |
|---|---|---|
| k4 DOPTQ − k3 | +0.0447 (5/5) [+0.038, +0.070]; LOCO +0.042..+0.052 | +0.0469 (5/5) [+0.040, +0.070] |
| k4 MAXMIN_R − k3 | +0.0480 (5/5) [+0.036, +0.075] | +0.0505 (5/5) [+0.038, +0.079] |
| k4 BISECT_R − k3 | +0.0508 (5/5) [+0.043, +0.082] | +0.0525 (5/5) [+0.044, +0.084] |
| k4 RANDOM − k3 | +0.0424 (5/5) [+0.036, +0.069] | +0.0434 (5/5) [+0.037, +0.068] |
| MAXMIN_R − DOPTQ (k4) | +0.0034 (3/5) [−0.016, +0.016]; LOCO −0.001..+0.008 | +0.0037 (3/5) [−0.014, +0.017] |
| BISECT_R − DOPTQ (k4) | +0.0062 (4/5) [−0.008, +0.022]; LOCO +0.004..+0.010 | +0.0056 (3/5) [−0.008, +0.022]; LOCO +0.003..+0.009 |
| RANDOM − DOPTQ (k4) | −0.0023 (3/5) [−0.012, +0.006] | −0.0035 (3/5) [−0.013, +0.006] |
| HULL_BISECT − DOPTQ (k4) | +0.0006 (3/5) [−0.022, +0.007] | +0.0007 (2/5) [−0.022, +0.008] |
| B1 − B0 (k4 DOPTQ) | +0.0320 (5/5) [+0.019, +0.066] | +0.0327 (5/5) [+0.020, +0.067] |
| B1 − B0 (k3) | +0.0341 (5/5) [+0.018, +0.076] | +0.0340 (5/5) [+0.018, +0.077] |
| B1 − HIST_D7 (k4 DOPTQ) | +0.0009 (4/5) [−0.003, +0.007] | +0.0010 (4/5) [−0.002, +0.007] |
| B1 refit at k4 − B1 with k3 β (k4 DOPTQ) | −0.0006 (0/5) [−0.004, +0.001] | −0.0009 (0/5) [−0.005, +0.001] |

---

## 3. Fixed cohort C4 (k_max ≥ 4, 178 cells, 1,516 rows per seed): the like-for-like rung

| policy (B1) | BPG per seed | BPG mean | ≥0.50 | P_boot | BP mean | ≥0.50 | P_boot |
|---|---|---|---|---|---|---|---|
| k=3 DOPT_Q | 0.4259 / 0.4131 / 0.4392 / 0.4364 / 0.4307 | 0.4290 | 0/5 | 0.00 | 0.4288 | 0/5 | 0.00 |
| k=4 DOPTQ-nested | 0.4893 / 0.4598 / 0.4853 / 0.4844 / 0.4801 | **0.4798** | 0/5 | 0.16 | **0.4821** | 0/5 | 0.19 |
| k=4 MAXMIN_R | 0.4743 / 0.4803 / 0.4918 / 0.4956 / 0.4760 | 0.4836 | 0/5 | 0.19 | 0.4862 | 0/5 | 0.23 |
| k=4 BISECT_R | 0.4836 / 0.4815 / 0.4918 / 0.4956 / 0.4815 | 0.4868 | 0/5 | 0.24 | 0.4885 | 0/5 | 0.26 |
| k=4 RANDOM (E) | 0.4714 / 0.4613 / 0.4929 / 0.4870 / 0.4734 | 0.4772 | 0/5 | 0.14 | 0.4781 | 0/5 | 0.15 |
| k=4 BISECT_RMID / HULL_BISECT | | 0.4876 / 0.4804 | 0/5 / 0/5 | 0.23 / 0.14 | 0.4889 / 0.4828 | 0/5 / 1/5 | 0.26 / 0.17 |
| k=4 MAXMIN_R_LOOSE | | 0.4883 | 0/5 | 0.26 | 0.4885 | 0/5 | 0.26 |
| k=4 ORACLE_B1 [O] | | 0.5255 | 5/5 | 0.82 | 0.5266 | 5/5 | 0.83 |

- The C4 gain from k = 3 to k = 4 is +0.048 to +0.060 (C3: +0.042 to +0.053).
- Even so, C4 stays about 0.015 below the C3 population row.
- The C3 row is lifted by the 51 k_max = 3 cells, which score **0.602** at k = 3. 91 % of those rows come from the giant cluster.
- This reproduces M18's mixture effect (M18 quoted C4 0.479, 0/5 seeds; P 0.17).

---

## 4. With and without C5-BTBP (pub_6a0ef8528c: 15 cells, 148 rows per seed)

**B1, seed mean, with seeds ≥ 0.50 and P_boot:**

| cohort | policy | BPG with | BPG without | BP with | BP without |
|---|---|---|---|---|---|
| C3 | k=3 | 0.4521 (0/5) | 0.4863 (0/5; P 0.26) | 0.4519 | 0.4862 (0/5) |
| C3 | k4 DOPTQ | 0.4968 (3/5) | **0.5311 (5/5; P 0.96)** | 0.4988 (4/5) | **0.5337 (5/5; P 0.97)** |
| C3 | k4 MAXMIN_R | 0.5001 (2/5) | 0.5364 (5/5; 0.97) | 0.5025 | 0.5394 (5/5) |
| C3 | k4 BISECT_R | 0.5029 (2/5) | 0.5399 (5/5; 0.98) | 0.5045 | 0.5419 (5/5) |
| C3 | k4 RANDOM (E) | 0.4945 (2/5) | 0.5296 (5/5; 0.95) | 0.4953 | 0.5307 (5/5) |
| C3 | k4 HULL_BISECT | 0.4973 (1/5) | 0.5330 (5/5) | 0.4995 | 0.5358 (5/5) |
| C4 | k=3 | 0.4290 | 0.4647 (0/5) | 0.4288 | 0.4645 (0/5) |
| C4 | k4 DOPTQ | 0.4798 (0/5) | **0.5171 (4/5; P 0.69)** | 0.4821 (0/5) | **0.5201 (4/5; P 0.74)** |
| C4 | k4 MAXMIN_R | 0.4836 (0/5) | 0.5233 (5/5; 0.78) | 0.4862 | 0.5267 (5/5; 0.84) |
| C4 | k4 BISECT_R | 0.4868 (0/5) | 0.5273 (5/5; 0.84) | 0.4885 | 0.5295 (5/5; 0.87) |
| C4 | k4 RANDOM (E) | 0.4772 (0/5) | 0.5153 (5/5; 0.67) | 0.4781 | 0.5165 (5/5; 0.69) |

- Removing C5-BTBP moves every rung by +0.034 to +0.041.
- The k = 3 → k = 4 gain itself is unchanged: +0.045 (C3) and +0.052 (C4) under DOPTQ without C5.
- So the publication sets the level, not the budget effect.
- This reproduces the M18 critique: "C4 at K = 4 without C5-BTBP 0.513 (4/5)". Here the refit-β B1 value is 0.517 (4/5).

---

## 5. Per pair type (B1, C3 rows, seed mean of the per-seed pooled R² within the pair type)

### BPG

| pair | rows/seed | k=3 | k4 DOPTQ | k4 MAXMIN_R | k4 BISECT_R | k4 RANDOM | k4 HULL_BISECT | k4 ORACLE [O] |
|---|---|---|---|---|---|---|---|---|
| La/Ce | 137 | 0.353 | 0.362 | 0.391 | 0.391 | 0.371 | 0.379 | 0.392 |
| Ce/Pr | 125 | 0.419 | 0.434 | 0.430 | 0.430 | 0.426 | 0.421 | 0.435 |
| Pr/Nd | 127 | 0.420 | 0.459 | **0.511** | 0.500 | 0.477 | 0.432 | 0.530 |
| Sm/Eu | 158 | 0.423 | 0.489 | 0.477 | 0.481 | 0.472 | 0.452 | 0.559 |
| Eu/Gd | 158 | 0.422 | 0.415 | 0.455 | 0.429 | 0.446 | 0.437 | 0.451 |
| Gd/Tb | 165 | 0.546 | 0.586 | 0.591 | 0.630 | 0.612 | 0.620 | 0.675 |
| Tb/Dy | 168 | 0.367 | 0.476 | 0.472 | 0.491 | 0.462 | 0.462 | 0.517 |
| Dy/Ho | 176 | 0.414 | 0.538 | 0.558 | 0.552 | 0.529 | 0.520 | 0.611 |
| Ho/Er | 173 | 0.256 | 0.324 | 0.311 | 0.301 | 0.305 | 0.302 | 0.323 |
| Er/Tm | 132 | 0.265 | **0.508** | 0.290 | 0.339 | 0.380 | 0.374 | 0.538 |
| Tm/Yb | 133 | 0.030 | 0.085 | 0.071 | 0.090 | 0.095 | **0.277** | 0.170 |
| Yb/Lu | 150 | 0.002 | 0.125 | 0.170 | 0.170 | 0.113 | **0.315** | 0.254 |

### BP

BP agrees with BPG within 0.05 in every k = 4 column (largest gap: MAXMIN_R/BISECT_R Yb/Lu, 0.219 against 0.170). The k = 3 column agrees within 0.08 (largest gap: Ho/Er, 0.179 against 0.256).

| pair | k=3 | DOPTQ | MAXMIN_R | BISECT_R | RANDOM | HULL_BISECT |
|---|---|---|---|---|---|---|
| La/Ce | 0.361 | 0.376 | 0.398 | 0.399 | 0.377 | 0.386 |
| Ce/Pr | 0.413 | 0.438 | 0.427 | 0.423 | 0.422 | 0.418 |
| Pr/Nd | 0.425 | 0.456 | 0.512 | 0.495 | 0.480 | 0.437 |
| Sm/Eu | 0.415 | 0.494 | 0.482 | 0.481 | 0.469 | 0.455 |
| Eu/Gd | 0.429 | 0.431 | 0.451 | 0.434 | 0.453 | 0.442 |
| Gd/Tb | 0.553 | 0.577 | 0.596 | 0.633 | 0.615 | 0.624 |
| Tb/Dy | 0.374 | 0.470 | 0.470 | 0.495 | 0.460 | 0.464 |
| Dy/Ho | 0.418 | 0.540 | 0.560 | 0.561 | 0.534 | 0.529 |
| Ho/Er | 0.179 | 0.320 | 0.302 | 0.286 | 0.283 | 0.287 |
| Er/Tm | 0.252 | 0.482 | 0.285 | 0.322 | 0.364 | 0.357 |
| Tm/Yb | 0.063 | 0.070 | 0.102 | 0.107 | 0.113 | 0.290 |
| Yb/Lu | 0.028 | 0.137 | 0.219 | 0.219 | 0.140 | 0.338 |

### BPG without C5-BTBP

Rows change only for La/Ce … Ho/Er and Yb/Lu. C5-BTBP has no Er/Tm or Tm/Yb rows.

| pair | k=3 | DOPTQ | MAXMIN_R | BISECT_R | RANDOM | HULL_BISECT |
|---|---|---|---|---|---|---|
| La/Ce | 0.507 | 0.517 | 0.552 | 0.552 | 0.527 | 0.537 |
| Ce/Pr | 0.468 | 0.495 | 0.488 | 0.488 | 0.481 | 0.473 |
| Pr/Nd | 0.439 | 0.488 | 0.551 | 0.537 | 0.509 | 0.455 |
| Sm/Eu | 0.503 | 0.539 | 0.567 | 0.571 | 0.532 | 0.537 |
| Eu/Gd | 0.458 | 0.466 | 0.497 | 0.469 | 0.492 | 0.477 |
| Gd/Tb | 0.554 | 0.586 | 0.562 | 0.608 | 0.598 | 0.597 |
| Tb/Dy | 0.367 | 0.487 | 0.488 | 0.508 | 0.480 | 0.478 |
| Dy/Ho | 0.454 | 0.531 | 0.555 | 0.549 | 0.541 | 0.515 |
| Ho/Er | 0.252 | 0.285 | 0.285 | 0.275 | 0.280 | 0.275 |
| Yb/Lu | 0.064 | 0.164 | 0.189 | 0.189 | 0.148 | 0.336 |

### Block decomposition

The table gives B1 SSE differences against k4 DOPTQ, in pooled-R² units (ΔSSE_block / SST, seed mean), with the number of seeds with a positive value in brackets. Blocks: light = La/Ce–Pr/Nd, mid = Sm/Eu–Dy/Ho, heavy = Ho/Er–Yb/Lu. Source: `P3_out/blocks.csv`.

| policy vs DOPTQ | BPG light | BPG mid | BPG heavy | BPG all | BP light | BP mid | BP heavy | BP all |
|---|---|---|---|---|---|---|---|---|
| MAXMIN_R | +0.0076 (5) | +0.0037 (3) | −0.0079 (1) | +0.0034 | +0.0058 (4) | +0.0032 (3) | −0.0054 (1) | +0.0037 |
| BISECT_R | +0.0067 (5) | +0.0054 (5) | −0.0059 (1) | +0.0062 | +0.0039 (4) | +0.0061 (5) | −0.0044 (1) | +0.0056 |
| BISECT_RMID | +0.0067 (5) | +0.0062 (5) | −0.0060 (1) | +0.0069 | +0.0039 (4) | +0.0067 (5) | −0.0046 (1) | +0.0060 |
| HULL_BISECT | −0.0020 (2) | −0.0007 (2) | +0.0033 (3) | +0.0006 | −0.0035 (2) | +0.0001 (2) | +0.0041 (3) | +0.0007 |
| RANDOM (E) | +0.0013 (4) | +0.0020 (3) | −0.0056 (1) | −0.0023 | −0.0006 (3) | +0.0018 (3) | −0.0047 (1) | −0.0035 |

M17's pre-declared Pareto question asks whether a policy keeps every block above −0.002 and improves one. **No member Pareto-dominates DOPT_Q-nested at k = 4** under either design:
- The gap-filling rules (MAXMIN, BISECT) win light and mid and lose heavy.
- HULL_BISECT wins heavy and loses light.
- The DOPT_Q-nested 5th read is the only one that rescues Er/Tm (0.508 against 0.29–0.38). It most often reads Er (17 %) or Sm (16 %).

---

## 6. Per cluster (cluster_cpg, B1, BPG)

The table gives the within-cluster pooled R² (seed mean), and the pooled C3 R² when that cluster is left out. BP differs by ≤ 0.04 in every cell (`P3_out/per_cluster.csv`).

| cluster (root) | chemotypes | pubs | cells | rows/seed | cells w/ 5th | k=3 | k4 DOPTQ | k4 MAXMIN_R | k4 BISECT_R | k4 RANDOM | pooled w/o, k=3 | pooled w/o, k4 DOPTQ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| g:pub_74428aeab2 (giant) | sc007, sc009 (DGA), sc010, sc019, sc021, sc027, … | 19 | 162 | 1170 | 119 | 0.452 | **0.491** | 0.498 | 0.499 | 0.489 | 0.444 | 0.496 |
| g:pub_3e0f6bb1b5 | sc029, sc059, sc060, sc068, sc069, sc070 | 5 | 15 | 174 | 15 | 0.391 | 0.469 | 0.433 | 0.440 | 0.444 | 0.443 | 0.486 |
| g:pub_af7a1cd919 (**holds C5-BTBP**) | sc061 | 3 | 17 | 170 | 17 | 0.305 | 0.353 | 0.357 | 0.357 | 0.351 | **0.475** | **0.519** |
| g:pub_8415538291 | sc077 | 1 | 10 | 120 | 10 | 0.335 | 0.455 | 0.533 | 0.533 | 0.492 | 0.456 | 0.498 |
| g:pub_f6d85bdb79 | sc030, sc031, sc036, sc058 | 3 | 9 | 52 | 8 | 0.437 | 0.492 | 0.494 | 0.497 | 0.481 | 0.450 | 0.495 |
| g:pub_0fd6ab4e40 | sc086 | 1 | 2 | 24 | 2 | 0.222 | 0.226 | 0.226 | 0.226 | 0.224 | 0.452 | 0.496 |
| g:pub_ca616a18dd | sc055 | 1 | 2 | 24 | 2 | 0.745 | 0.775 | 0.798 | 0.798 | 0.788 | 0.448 | 0.493 |
| g:pub_cd72e4a7bb | sc020 | 1 | 3 | 15 | 0 | 0.780 | 0.780 | 0.780 | 0.780 | 0.780 | 0.449 | 0.494 |
| g:pub_1845fd6492 | sc015 | 1 | 1 | 12 | 1 | 0.490 | 0.500 | 0.467 | 0.552 | 0.531 | 0.450 | 0.496 |
| g:pub_401a06b4b7 | sc012 | 1 | 1 | 12 | 1 | 0.482 | 0.640 | 0.563 | 0.656 | 0.615 | 0.452 | 0.496 |
| g:pub_4fb6e0c041 | sc008 | 1 | 1 | 11 | 1 | 0.675 | 0.666 | 0.673 | 0.670 | 0.671 | 0.451 | 0.496 |
| g:pub_84113c17ad | sc084, sc085 | 1 | 2 | 10 | 2 | 0.150 | 0.144 | 0.144 | 0.144 | 0.144 | 0.454 | 0.499 |
| g:pub_1927dff570 | sc000, sc001, sc096, sc097 | 1 | 4 | 8 | 0 | 0.334 | 0.334 | 0.334 | 0.334 | 0.334 | 0.449 | 0.494 |

What the per-cluster table shows:
- **The giant cluster** (65 % of rows) reaches only **0.489–0.499** at k = 4 under every policy. This already includes its own k_max = 3 cells at k = 3.
- **Only one cluster moves the pooled value much when left out.**
  - Leaving out the C5 cluster lifts the pooled k4 DOPTQ R² from 0.497 to 0.519.
  - Leaving out any other single cluster gives 0.486–0.499.

---

## 7. Density-indexed view (B1)

### 7a. Population view: mean reads per measured metal over the C3 rows

| cohort | rung | mean reads / measured metal | mean reads per cell | rows outside the read hull | R² BPG | R² BP |
|---|---|---|---|---|---|---|
| C3 | k = 3 | 0.355 | 4.00 | 11.9 % | 0.452 | 0.452 |
| C3 | k = 4 (any policy) | 0.423 | 4.78 | DOPTQ 7.6 %, MAXMIN/BISECT 8.6 %, RANDOM 8.9 %, HULL 3.7 % | 0.494–0.504 | 0.495–0.505 |
| C4 | k = 3 | 0.322 | 4.00 | 12.7 % | 0.429 | 0.429 |
| C4 | k = 4 (any policy) | 0.402 | 5.00 | 2.9–9.1 % | 0.477–0.487 | 0.478–0.489 |
| C3 without C5 | k = 3 → k = 4 | 0.359 → 0.426 | | | 0.486 → 0.530–0.540 | 0.486 → 0.531–0.542 |

### 7b. Rows grouped by their own cell's density

Each cell gives R² (rows per seed, clusters). 4 reads = k = 3 rows; 5 reads = k = 4 rows of cells that took a 5th read.

| density bin (typical n_read / n_metals) | 4 reads | 5 reads, DOPTQ | 5 reads, MAXMIN_R | 5 reads, BISECT_R | 5 reads, RANDOM | pooled 4 + 5 (DOPTQ) |
|---|---|---|---|---|---|---|
| 0.286 (4/14) | 0.487 (936, 9) | | | | | 0.487 |
| 0.29–0.32 (4/13) | **0.316** (353, 3) | | | | | 0.316 |
| 0.32–0.37 (5/14, 4/12, 4/11) | −0.047 (23, 2) | 0.543 (936, 9) | 0.539 | 0.545 | 0.539 | 0.542 (959, 9) |
| 0.37–0.45 (5/13, 4/10, 4/9, 5/12) | 0.424 (79, 3) | **0.366** (361, 3) | 0.385 | 0.384 | 0.368 | 0.385 (440, 5) |
| 0.45–0.55 (5/11, 4/8, 5/10) | 0.559 (305, 2) | 0.503 (40, 2) | 0.463 | 0.480 | 0.469 | 0.608 (345, 3) |
| 0.55–0.70 (5/9, 4/7, 5/8, 4/6) | 0.500 (96, 2) | 0.366 (135, 4) | 0.369 | 0.371 | 0.363 | 0.424 (231, 5) |
| > 0.70 (5/7, 4/5, 5/6) | 0.560 (10, 2) | 0.588 (44, 1) | 0.588 | 0.588 | 0.588 | 0.696 (54, 3) |

The table is BPG; BP is within 0.01 except in the 23-row bin. Without C5-BTBP the 4/13 bin rises to 0.415 (213 rows) and the 5/13 bin to 0.467 (DOPTQ) or 0.523–0.525 (MAXMIN/BISECT); the other bins are unchanged.

**R² is not a function of density.**
- At almost equal density, 4/14 scores 0.487 and 4/13 scores 0.316. Likewise 5/14 scores 0.543 and 5/13 scores 0.366.
- The series-length and cluster stratum dominates.
- The 13-metal stratum is where C5-BTBP sits: 140 of its 148 rows per seed are 13-metal rows.
- Within the 14-metal cells (78 cells, 9 clusters), one extra read moves R² from 0.487 to 0.539–0.545.

### 7c. Saturation: reads = k_max + 1 (every other measured metal read)

| stratum | R² B1 (BPG / BP) | R² B0 | rows/seed | cells | clusters | largest-cluster share | density |
|---|---|---|---|---|---|---|---|
| k_max 3 @ k = 3 | **0.602 / 0.602** | 0.564 | 286 | 51 | 4 | 0.91 | 0.530 |
| k_max 4 @ k = 4 (every policy identical) | **0.444 / 0.447** | 0.415 | 111 | 34 | 4 | 0.78 | 0.688 |
| k_max 5 @ k = 4 (not saturated) | DOPTQ 0.459, MAXMIN 0.457, BISECT 0.461, RANDOM 0.449 | 0.459 | 56 | 11 | 2 | 0.54 | 0.544 |
| k_max 6 @ k = 4 (not saturated) | DOPTQ 0.484, MAXMIN 0.488, BISECT 0.492, RANDOM 0.481 (without C5: 0.529–0.542) | 0.449 | 1,349 | 133 | 10 | 0.59 | 0.373 |

Saturated strata exist here only for k_max 3 and 4. k_max 5–6 saturate at k = 5–6, which this probe does not compute.

M18's live hypothesis was density invariance at saturation. Its kill rule fires if any stratum with at least 3 clusters is below 0.48, or if the spread is ≥ 0.05:
- the k_max-4 stratum scores 0.444 with 4 clusters;
- the spread is 0.158;
- C5-BTBP is not in either stratum.

So the hypothesis **is killed on these already-seen data** (consistent with the M18 critique's 0.597 / 0.447). The k_max-4 cells are small series of 6–10 metals. Reading every other metal of a short series does not reach 0.50.

---

## 8. Where the 5th read goes

Share of cell-instances that took a 5th read (BPG, 5 seeds):

| metal | DOPTQ | MAXMIN_R | BISECT_R | BISECT_RMID | HULL_BISECT | MAXMIN_R_LOOSE |
|---|---|---|---|---|---|---|
| Pr | 0.048 | 0.006 | 0.006 | 0.006 | 0.006 | 0.006 |
| Nd | 0.125 | 0.238 | 0.201 | 0.201 | 0.063 | 0.238 |
| Sm | 0.164 | 0.285 | 0.166 | 0.166 | 0.134 | 0.162 |
| Eu | 0.048 | 0.110 | 0.110 | 0.110 | 0.000 | 0.128 |
| Gd | 0.113 | 0.109 | 0.109 | 0.109 | 0.107 | 0.090 |
| Tb | 0.012 | 0.010 | 0.015 | 0.015 | 0.010 | 0.011 |
| Dy | 0.098 | 0.013 | 0.143 | 0.143 | 0.140 | 0.021 |
| Ho | 0.009 | 0.012 | 0.012 | 0.021 | 0.012 | 0.185 |
| Er | 0.169 | 0.070 | 0.102 | 0.176 | 0.116 | 0.012 |
| Tm | 0.048 | 0.000 | 0.000 | 0.049 | 0.000 | 0.000 |
| Yb | 0.054 | 0.007 | 0.004 | 0.001 | 0.015 | 0.007 |
| Lu | 0.111 | 0.139 | 0.131 | 0.002 | 0.398 | 0.139 |

- **Lu in the read set of 14-metal cells:**
  - k = 3: 1.8 %;
  - k = 4: DOPTQ 17.2 %, MAXMIN/BISECT 11.8 %, HULL_BISECT 58.7 %.
- **How often the policies agree:**
  - MAXMIN_R and BISECT_R each pick the same metal as DOPTQ in 47–49 % of cells;
  - MAXMIN_R and BISECT_R agree with each other in 83 %.

---

## 9. What this means for the three ideas (exploratory reading)

### H-BUDGET: "5 measured metals with B1 reach 0.50 on C3"
- On the registered form (DOPT_Q-nested, β refit), the result is **0.4968 BPG (3/5 seeds) and 0.4988 BP (4/5)**, with P_boot 0.41 / 0.45.
- Every label-free policy lands in 0.494–0.505.
- **The claim is a coin flip on this corpus.** The P(R² ≥ 0.50) values above are consistency numbers on seen data, not evidence.
- The honest budget statement has two parts:
  - one extra legal read is worth **+0.042 to +0.053** (C3) and **+0.048 to +0.060** (C4) in pooled R², 5/5 seeds, with cluster intervals excluding 0;
  - whether that crosses 0.50 depends on C5-BTBP and on the k_max-mixture of the cohort.

### M18 (density-indexed ladder)
- The mixture effect reproduces: C3 0.497 against C4 0.480 at k = 4.
- The β refit is immaterial (≤ 0.001), and B1 equals HIST_D7 (+0.001).
- Density does not index R²: the series-length and cluster strata dominate.
- Density invariance at saturation is killed: k_max 3 scores 0.602 and k_max 4 scores 0.444.
- k* depends on C5-BTBP:
  - C4: no crossing at K = 4 with C5-BTBP; a crossing at K = 4 without it (DOPTQ 4/5 seeds).
  - C3: 3/5 seeds with C5-BTBP and 5/5 without.

### M17 (policy envelope)
- The label-free envelope at k = 4 is **0.0091 (BPG) and 0.0096 (BP)** in pooled R², running from RANDOM to BISECT_RMID.
- No member Pareto-dominates DOPT_Q-nested.
- BISECT_R − DOPTQ is +0.006, with the cluster interval including 0 but LOCO all positive. It is still below the about 0.011 MDE.
- At k = 4, placement is again a choice of block:
  - the gap-filling rules help light and mid and cost heavy;
  - hull closure buys Tm/Yb and Yb/Lu and costs light;
  - DOPT_Q's 5th read is the one that fixes Er/Tm.

---

## 10. What would change the conclusion

1. **C5-BTBP provenance (M25).** This is the deciding item.
   - If those 15 cells are excluded as mis-attributed or erroneous, every policy crosses 0.50 at k = 4: C3 5/5 seeds, P_boot 0.95–0.98; C4 4–5/5, P_boot 0.67–0.87.
   - If they stay, C4 never crosses and C3 is a coin flip.
2. **New independent clusters.** The giant cluster is 65 % of rows and alone scores 0.489–0.499 at k = 4.
   - Any new source that resembles it keeps the population at about 0.50.
   - New sources with long (14-metal) series would raise it: 14-metal rows score 0.54 at k = 4.
   - New 13-metal or short-series sources would lower it.
3. **The registered policy.**
   - BISECT_R or MAXMIN_R instead of DOPT_Q-nested adds +0.003 to +0.006 on average.
   - The seed count does not follow monotonically: under BPG, DOPTQ gets 3/5 seeds while BISECT_R gets 2/5, despite BISECT_R's higher mean.
   - A pre-declared policy family (M17) is needed before any seed count is quoted.
4. **The legality rule.**
   - Dropping the reachability look-ahead (E3's rule) adds about +0.004 to MAXMIN (0.5042, 4/5 BPG).
   - Allowing a 5th read that neighbours a read metal ("unrestricted", E3: 0.530) changes the estimand, because those queries become measured.
5. **Fresh splits or seeds.**
   - Per-seed R² at k = 4 spans 0.479–0.511.
   - The low seed 130363 alone decides the DOPTQ count.
   - The 5 dev seeds share cells, so "k of 5 seeds" is not a replication count.
6. **A test-policy-specific β refit.** This was not done. It is bounded by the k = 3-β sensitivity of ≤ 0.001, so it is unlikely to matter.

---

## 11. Code path and reproduction

Run from `D:\ml_separator_gh` with `.venv\Scripts\python.exe -W ignore`, one process at a time:

1. `generations\gen21\round5_design_20261023\probes\scripts\P3_k4_compute.py` (308 s). It refits B1 at k = 3 and k = 4 per fold, enumerates every legal (and LOOSE) 5th metal, records the policy picks, and writes `P3_out\rows.parquet` (77,281 rows), `cells.parquet` and `fits.json`.
2. `...\P3_k4_summarise.py` (a few minutes). It runs the identity checks and produces the ladders, Monte Carlo, contrasts, per-pair, per-cluster, density, saturation, strata, placement and hull outputs. It writes `ladder_summary.csv`, `contrasts.csv`, `per_pair.csv`, `per_cluster.csv`, `density.csv`, `population_density.csv`, `saturation.csv`, `strata.csv`, `summary.json`, and the logs `compute_log.txt` and `summary_log.txt`.
3. `...\P3_k4_blocks.py` (seconds). It writes the block decomposition to `blocks.csv`.

`P3_patch_density.py` is a one-off helper that rewrote the density section of the summariser. It has no analysis content.

**Imported read-only:**
- `r4.refs` (`fit_reference`, `predict_levels`, `HIST_D7_BETA`) and `r4.metrics.add_clusters` (round 4);
- `g21.cache.load` and `g21.evaluate.assert_test_isolation`;
- `gen20k.design.legal_metals` and `make_design`;
- `gen20k.inference._draw_counts`.

No existing file was modified.

---

## 12. Caveats and deviations

**Exploratory.**
- The 229 C3 cells, the dev seeds, DOPT_Q-nested at k = 4, the C3/C4 split and the C5-BTBP sensitivity had all been reported before (E3 §6; the M18 and M17 critiques).
- The policy definitions and tie-breaks here were written after those numbers were known.

**Dependence.**
- The five seeds re-split the same cells.
- The cluster bootstrap has 13 clusters (Kish about 2.2, the largest 65 %), so the intervals and P_boot are anti-conservative.
- The LOCO ranges are the more honest spread.

**RANDOM.**
- The per-seed values are exact expectations.
- "Seeds ≥ 0.50" for RANDOM counts seeds whose expected R² is ≥ 0.50. Use the Monte Carlo E[#seeds] = 1.70 (BPG) / 1.77 (BP) instead.

**Labelled bounds and diagnostics.**
- ORACLE_B1 uses labels and is a bound only.
- HIST_D7 is not fold-independent and is a diagnostic only.

**Process deviation.**
- At the start, one temporary text file (an extract of the M17/M18 entries of `critiques.json`) was written to the session scratchpad. It was deleted immediately and never read back.
- At the end, one read-only `git status --short` was run by mistake to confirm that no tracked file had changed (none had). No other git command was run.
- Probe outputs are written under `probes\scripts\P3_out\`. This is a data folder named with the probe id, next to the scripts.
