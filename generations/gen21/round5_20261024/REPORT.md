# Gen21 round 5: report

**Status (2026-10-24).**
- **Stage 0** is sealed. The current seal is `b96c3bef9828…` (amendment A2). It superseded `2f74292f5f44` (A1), which had superseded the original seal `1f0cae0064a6`. Both amendments fix documented code defects, leave the scientific choices unchanged, and are disclosed in `AMENDMENTS.md`.
- **Stage 1:** every registered item ran its reproduction (TECHNICAL SMOKE) before scoring. Every scoring run also runs its own reproduction gate first and stops on a mismatch.
  - **First-pass order.** The first pass departed from the ITEM_SPEC command order three times:
    - E-CT's standalone BP `--repro` ran after the BP scoring run (10:34:32 vs 10:24:33 UTC); that run's own gate ran first and matched.
    - M03's standalone `--repro` ran for BPG only; its BP reproduction ran as the first step of the BP scoring run.
    - E-DEC's arms run overwrote the standalone reproduction record, with identical values.
  - **Review and A2.** An adversarial results review then found 65 verified findings: reporting-layer defects, missing registered outputs, one recipe deviation and one wrong report-rule outcome (the E-D Pareto question). They were fixed in code under A2. Points that needed no code change are disclosed here, and two sealed ITEM_SPEC texts were left unchanged.
  - **Rerun.** Every item was then rerun under the new seal in the frozen order, with its standalone reproduction for both designs ahead of its scoring runs. Every stored number reproduced except the changes A2 lists (`results/a2_rerun_comparison.txt`).
  - **Reproduction records.** The stored `repro_<design>.json` files of E-CT, M03 and E-DEC are the scoring runs' in-run gate records. The standalone runs are in `RUN_RECORD.md` and `results/logs/*_A2_repro*.log`.
- **Stage 2** waits for data. Every request in `DATA_REQUEST.md` is **OPEN**. The supply-audit kill fired at Stage 0 (D4), so this round is **descriptive**.

Every in-corpus number below is **EXPLORATORY (seen)**, sometimes combined with a further label. Nothing in-corpus is a test, and no in-corpus p-value is called significant. BPG decides and BP is a sensitivity.

E-KX is the one exception. Its in-corpus design, E-KX-LOPO, takes the round-4 BP out-of-fold prediction as its B1 base (sealed in D8; never called BP), and the BPG-base variant is a sensitivity.

Each estimand is scored on its own rows, against B1 on those same rows. On E-CT's PRIMARY rows, for example, B1 is 0.531, not 0.452. **Values of different estimands are not comparable.** The result table gives each estimand its own labelled row, to be read against its own B1, never against another row or against E-A's 0.452 (sealed substitution rule 1, PROTOCOL §3).

The round-4 seal (`5bacfe4c`) and the round-3 LF digests verified after every run. Nothing has been committed or pushed.

## Result table and verdict per estimand

Rows are separate estimands on different row sets; levels are not comparable across rows.

| estimand (question; rows) | strongest reproducible predictor | headline (BPG; BP beside it) | label | verdict |
|---|---|---|---|---|
| **E-A** strict-adjacent log SF from 4 read metals (229 cells, 1,802 pairs per seed) | **B1** (PWL_R + one fold-fitted Gd amplitude) | B1 0.4521 / 0.4519. M35's positive control B1 - B0 passes the sealed gate: relative pair-SSE +0.0586, SF_POOL p 0.0032 over 13 units, guard +0.0050. No model slot was registered or run | EXPLORATORY (seen); DIAGNOSTIC (M34's kill fired) | **no detected gain; closed on this corpus** (0.50 not reached) |
| **E-CT** condition transfer: a new condition of an extractant whose own earlier series is known (PRIMARY: 99 cells, 12 publications, 14 units, 690 rows per seed) | **SH** = own-series template shrunk toward B1 (K = 1.78) + 4 reads | SH - B1 +0.169 R² on the same rows (LOPO [+0.055, +0.283]; 14 of 14 units); BP +0.166 | EXPLORATORY (seen) | **exploratory gain (seen); confirmation data-gated** (C-ECT NOT RUN: 2 of the 7 chemotype units it needs) |
| E-CT annex M03: sibling weighting (PRIMARY_NO_ORNL: 85 cells, 11 publications) | SH, as sealed. The nested read-match weighting passes its in-corpus consistency check, but it is fragile | nested - SH +0.020 (9 of 11 publications); BP +0.020 | EXPLORATORY (seen); deploy rule CONSISTENCY CHECK | exploratory gain (seen), fragile; not deployed: C-ECT stays SH@4 |
| E-LAB: campaign calibration of a new extractant (PRIMARY: 92 cells, 16 publications) | MX_LM10 (best of about 25 arms) | relative SSE +0.076 vs B1 (LOPO [-0.011, +0.164]; conservative laboratories, PROVISIONAL key, 9/0/0); BP +0.078 | EXPLORATORY (seen); best of about 25 arms; expect regression | exploratory gain (seen), mostly campaign information (random own-campaign donors give +0.060); external leg data-gated |
| Window axis (M09; E-A rows) | B1 inside the OP75 level window | coverage 0.754; inside R² 0.524 vs cell-matched control 0.501; tail ratio 3.80 [1.93, 8.04] | EXPLORATORY (seen); report rules CONSISTENCY CHECK | no detected gain in bulk precision (narrowing 0.091 < 0.15; never an accuracy gain); tail flag exploratory (seen): passes on BPG, fails on BP |
| E-DEC: direction calls (E-A rows) | B1 with valid calibration (LTT, per-call risk 0.10) | calls 0.5 % of pairs with C5-BTBP in, 13 % without; CRC / plug-in 84 % | EXPLORATORY (seen) | insufficient power or data (valid calls nearly nil with C5-BTBP in; the seed-mean false-sign rate 0.194 rests on 44 calls) |
| E-REC: own-prior direction calls (PRIMARY_OWN: 112 cells, 16 publications; not the E-A rows) | SH_own@4 | false-sign rate at matched coverage 0.495: SH_own@4 0.003 vs B1@4 0.025 (units 4/0/14); BP 0.004 vs 0.025 | EXPLORATORY (seen); unit tests DIAGNOSTIC | exploratory gain (seen); own-series information, not learned chemistry |
| E-RANK: ranking adjacent pairs by predicted \|log SF\| (E-A rows) | none clearly; B1 and HBK@ML within about 0.03 | a straight line recovers 0.65 (Spearman), 0.67 (top-3) and 0.48 (within-publication top-3) of B1's ranking skill over random | EXPLORATORY (seen); reporting appendix | descriptive; never a claim |
| E-D: read ladder and policy envelope (E-A queries and fixed cohorts) | B1 with DOPT_Q-nested reads | one more read +0.045 paired; C4 at 5 reads 0.480 / 0.482; envelope width 0.009 | EXPLORATORY (seen); different read budget | exploratory gain (seen) from one more read (a different budget, never E-A); no detected gain from read placement (no label-free policy Pareto-dominates DOPTQ) |
| E-W: wider pairs (WIDE: 225 cells, 38 publications) | B1 | Nd/Sm B1 - LINE contrast -0.095 [-0.73, +0.54], G = 3 → RETIRED | EXPLORATORY (seen); CONSISTENCY CHECK (1.7-R01) | no detected gain over a line at Nd/Sm outside the giant cluster |
| E-KX: Nd dip of TODGA-class DGAs (sc009 Pr/Nd: 46 cells, 10 publications) | KX_SC009 | +0.243 R² vs B1 (E-KX-LOPO, BP OOF base); BPG base +0.252 | EXPLORATORY (seen); rule CONSISTENCY CHECK | exploratory gain (seen) on sc009 Pr/Nd; insufficient data for a chemotype-specific claim (narrowed to a universal Nd term on pooled rows) |

The contributions above are not learned chemistry. E-CT, E-LAB and E-REC gain from **the same study's or campaign's own
prior series**: same publication, since the laboratory key is PROVISIONAL (U6 open). E-KX gains from a **better series
prior** at one pair. On E-A, nothing beyond B1's single Gd amplitude improves interpolation.

## 1. Is anything beyond B1 supported on E-A?

**No. E-A is closed on this corpus.** Round 5 registered no E-A model slot, because the pre-slot gate saw no proposal
(§4.5). It ran only the M35 convention check (item 1.0, EXPLORATORY (seen); DIAGNOSTIC):

- **The positive control passes, so M35's convention is kept.** On BPG with C5-BTBP in, B1 - B0 gives:
  - relative pair-SSE +0.0586, with every seed positive (0.044 to 0.066);
  - exact SF_POOL p 0.0032 over the 13 `cluster_cpg` units, W/L/T 11/1/1;
  - balanced-MAE guard +0.0050;
  - R² B1 0.4521 against B0 0.4180.

  BP gives +0.0585 (p 0.0027). Every C5-BTBP variant passes.
- **HIST_D7 - B1 fails the gate:** relative SSE -0.0019, p 0.66, W/L/T 4/8/1. The learned part of B1 is one known
  number, as round 4 found.
- **The six-design check holds.** HIST_D7 - B0 passes on A, B, BR, BQ, BP and BPG alike: relative SSE +0.055 to +0.058,
  p 0.002 to 0.003.
- **Where the gate is fragile:**
  - The fold with the most rows in each seed (58 % of rows, 8 units) gives only +0.037 (p 0.055), so that fold fails the
    gate. The other folds give +0.082.
  - In the 11-cell V2-3 condition-aware stratum (one unit), B1 is *worse* than B0: relative SSE -0.090 (BP -0.086),
    descriptive (added under A2).
- **Replay check (CONSISTENCY CHECK, 1.0-R03):** the exact p legs match. The relative-SSE leg reads 5.86 % against the
  seen 5.85 %. The difference is definitional: the seen value pools over seeds (0.05854), while the sealed currency is
  the seed mean (0.05856).
- **Breadth diagnostic (1.0-R04; DIAGNOSTIC, never a gate).** SMALL_POOLED B1 - B0: 6 of 6 units up, SF_LOG p 0.016,
  IM t 3.08. HIST_D7 - B1: 3/3/0, SF_LOG p 0.86.
- **M08 appendix (1.0-R05; DIAGNOSTIC).** These are B1's residual correlations by relation class (BPG). A class counts as
  identified with ≥ 3 product units and a top unit ≤ 80 % of the products.
  - Identified with C5-BTBP in:
    - same extractant and same publication, ρ 0.33 (7 units; leave-one-cluster-out 0.28 to 0.39);
    - other extractant, same publication, other chemotype, ρ 0.09;
    - the other-publication baseline, ρ 0.002.
  - Not identified: same extractant in other publications, and the same-chemotype classes.
  - Without C5-BTBP, the same-extractant same-publication class is no longer identified (top unit 87 %). BP classifies
    the same way.

**B1 0.452 stands, and 0.50 at four read metals is not reached.** Gains below about 0.01 R² are undetectable on this
corpus: seen, round 4 [EB §3; METRICS.md §7]; round 5 computed no E-A MDE. MDEs are contrast-specific (round-4 dev
MDE80, `cluster_cpg`, BPG): B1 - B0 0.044, S05 - B1 0.011, S11 - B1 0.003.

More reads are a different budget (E-D, §3.4). On E-A itself, only new or cleaner data can move the level.

## 2. What does condition transfer offer, how is it labelled, and what would confirm it?

**E-CT asks:** for a new condition of an extractant whose own earlier series is known, predict log SF from 4 reads at
the new condition. The frozen predictor is **SH** (`r5/ect.py`, sealed and hashed in `CONFIRMATION_PACKAGE.md`): the mean
of the extractant's own sibling series, shrunk toward B1 with weight n/(n + 1.78).

**In-corpus results (EXPLORATORY, seen).** PRIMARY is CLEAN + the bit-identity rule, n_sib ≥ 2, without C5-BTBP:
- 99 cells, 12 publications, 14 publication × chemotype units, 690 rows per seed;
- 3 source clusters, Kish 1.25;
- 94 of the 99 cells are DGA.

| quantity | BPG | BP |
|---|---|---|
| R² SH / SIB / OFF / B1 / COPY on identical rows | 0.700 / 0.686 / 0.688 / 0.531 / 0.483 | 0.697 / – / – / 0.532 / – |
| SH - B1 (LOPO jackknife t(11); units up) | **+0.169** [+0.055, +0.283]; 14 of 14 | +0.166 [+0.046, +0.286]; 14 of 14 |
| per seed | +0.169 / +0.143 / +0.180 / +0.170 / +0.183 | +0.170 / +0.141 / +0.189 / +0.170 / +0.160 |
| with C5-BTBP (114 cells; SENSITIVITY) | +0.168 [+0.116, +0.220] | +0.167 |
| P1's population without the bit rule (bridge to P1's seen +0.177 BPG / +0.174 BP; SENSITIVITY) | +0.177 [+0.076, +0.279] | +0.174 |
| leave-one-publication-out range; top-3 unit share | +0.126 (without pub_b0b3296ec5) to +0.197; 0.72 | +0.121 to +0.197; 0.72 |

- **Report rules (in-corpus consistency checks; no kill fires in-corpus):**
  - **Harm rule (ii)** at n_sib = 1 (16 cells, 8 publications): SH vs B1 relative SSE +0.122 ≥ -0.05, so it passes. One shrunk sibling already helps.
  - **Artefact flag (iii):** Ce pairs carry 33 % of the gain (Gd pairs 25 %), below the 50 % threshold, so the flag is not triggered.
  - **Mechanism wording: the default applies.** It reads "same-campaign information (extractant transfer and campaign fingerprint not separated)".
    - SH - OFF_XEXT is +0.0445 and SH - PB_PUB +0.0452. Both rest on 6 publication × extractant families in only 3 publications.
    - The recovered shares per publication are 1.15 / 0.31 / -0.23 (OFF_XEXT) and 1.00 / 0.50 / -0.23 (PB_PUB). The pooled 0.47 is therefore not an estimate.
  - **Claim wording:** by unit signs, SH is not distinguishable from cheaper own-series rules.
    - Against SIB it gives +0.014 (9/5, SIGN p 0.21).
    - Against NEAR it gives +0.146 pooled (BP +0.151), most of the SH - B1 gain, because NEAR_r4 itself reaches only 0.554. But only 8 of 14 units favour SH (8/6, SIGN p 0.40; LOPO [-0.064, +0.357]).
    - The claim is therefore "own-series template + 3-4 reads", not "K = 1.78".
  - **SH with 2 reads stays excluded from claims.** SH@2 - B1@4 is +0.075, with 8 of 14 units up and a top-3 share of 0.99. Against B1 at the same 2 reads, SH@2 - B1@2 is +0.463 (13 of 14 units).
- **Read ladder on identical rows:**

  | reads | SH | B1 |
  |---|---|---|
  | 2 | 0.606 | 0.143 |
  | 3 | 0.671 | 0.467 |
  | 4 | 0.700 | 0.531 |

  COPY (the sibling mean, no reads) gives 0.483, and SH0 0.474, both below B1 at 4 reads. **Do not copy without reads.**
- **Read alignment.** SH - OFF is +0.012 (LOPO [-0.017, +0.041]). On the 72.9 % of rows where every sibling was read at the
  target's metals, SH and OFF are identical (max difference 4e-16). The whole excess is pub_19783c039b.
- **Influence.** pub_b0b3296ec5, the only acid-type publication, carries 43 % of the gain.
- **Sibling-count strata (between-cell, descriptive).** Without C5-BTBP, the ≥ 8-sibling bin shows the largest relative
  SSE gain (0.549). With C5-BTBP in (added under A2, brief §5.2), that bin grows to 51 cells and its gain drops to 0.278.
  So the dose-response spike is partly a C5-BTBP absence effect.

**M03 (sibling weighting annex).** The registered deploy rule tests whether nested read-match / acid-distance weighting
should replace the uniform SH. On PRIMARY_NO_ORNL it **passes**: +0.020, 9 of 11 publications, uniform chosen in 0 of 11
folds; BP gives +0.020. That is contrary to the expectation. Three things bound the result:

- **The read-match kernel carries it.** The acid kernel alone deploys SH in every population (NEST_ACID_sh - SH -0.003 on
  PRIMARY_NO_ORNL). Permuting acid within publication leaves the nested gain intact (acid-permutation null p 0.96 on
  PRIMARY_NO_ORNL, 0.94 on PRIMARY, both designs), while permuting the read distances removes it (p 0.002).
- **It does not survive the classic one-SE rule.** That rule picks uniform in every fold (11 of 11, gain 0). With no
  one-SE rule (argmin), the gain stays at +0.022 (9 of 11 publications; re-nested LOPO minimum +0.015).
- **Under the sealed paired one-SE rule, the re-nested leave-one-publication-out minimum is -0.001**, without
  pub_c8195b4ca7.

C-ECT therefore stays sealed on SH@4 vs B1@4. Read-match weighting would need its own registration on new data.

- **Level rule:** retired (0.20 < 0.60).
- **Acid-trend statistic** (EXPLORATORY, seen): +0.005, with an interval that includes 0. No claim.

**What would confirm it.** C-ECT (`CONFIRMATION_PACKAGE.md`, §4.2 fallback ii) is SH@4 against B1@4 on new laboratories ×
extractant families with ≥ 3 full-series conditions:

- one value per chemotype unit;
- an exact one-sided sign-flip over chemotype units;
- **at least 7 chemotype units**;
- a fixed sequence, SH@4 then SH@3.

Its fresh size check passed. The supply audit counts only **2** linked chemotype units in hand (Chiba-Katsuta and
KULeuven-Binnemans), and even every listed candidate converting gives 6. **So C-ECT is NOT RUN; confirmation needs new
E-CT-capable laboratories** (U3 dry run, U5 targeted search; `DATA_REQUEST.md`). The verdict is **exploratory gain
(seen); confirmation data-gated**.

## 3. What do the descriptive deliverables say?

Each item is reported under its own estimand. None is an E-A value.

### 3.1 E-LAB: campaign calibration of a new extractant (best of about 25 arms on seen rows; expect regression)

PRIMARY is 92 cells, 16 publications, 9 conservative laboratories (PROVISIONAL key) and 4 source clusters (one holds
72 %). On it, **MX_LM10 - B1** gives relative SSE +0.076 (LOPO [-0.011, +0.164]):

- conservative laboratories 9/0/0 (exact SF_POOL p 0.002); publications 10/4/2 (p 0.097);
- R² 0.444 → 0.486; BP +0.078 (8/1/0).

The controls show where the gain comes from:

- **Random own-campaign donors (E_RM) already give +0.060 vs B1,** so MX_LM10's level matching adds only +0.018
  [-0.004, +0.040]. The gain is mostly campaign information.
- Cross-publication placebos lose against MX_LM10: +0.091 (E_PBC) and +0.100 (E_PCM).
- **Fragile:** leaving out the most helpful pair of publications leaves relative SSE +0.006, and dropping the top-3
  publications gives -0.005.
- **Cleaning matters:**
  - Near-copy cleaning moves the lab-1 seen +9.0 % to +7.6 %.
  - It moves POPX0 (new extractants; 60 cells, 13 publications) to +0.094 (LOPO [+0.006, +0.183]; publications 8/3/2,
    SF_POOL p 0.067).
  - On lab-1's uncleaned pool, the same arms reproduce the seen values (POPX0 +0.111, leave-two-out +0.026).

The in-corpus kill readings would pass if applied, but they are not applied. The external leg (≥ 8 independent
laboratories) is data-gated.

### 3.2 Window axis (M09): a tail-risk reporting axis, not an accuracy gain

At the training-fold OP75 level window (BPG):

- **Coverage and inside R².** The window covers 0.754 of rows, with B1 inside R² 0.524 against 0.501 for cell-matched
  random abstention (q97.5 0.511).
- **Error rates.** The MSE is 0.020 inside and 0.052 outside. |error| > 0.5 occurs in 1.1 % of inside pairs and 4.3 % of
  outside pairs, all adjacent pair types pooled: a ratio of 3.80 with bootstrap interval [1.93, 8.04].
- **Report rules (CONSISTENCY CHECKs):**
  - **Tail flag:** it passes on BPG (without C5-BTBP, 3.66 [1.20, 7.48]). It fails on BP, because the lower bound
    without C5-BTBP is 0.965.
  - **Row flag:** it holds only on B1's inside relative MSE (1 - MSE_in / MSE_all of B1's own residual; not a relative
    SSE vs B1): 0.229 against CELL q97.5 0.218. NEST_FAM's inside R² (0.495) is *below* its cell-matched control
    (0.500). The window keeps lower-variance rows within cells, so the flag concerns squared error, not explained
    variance. In 13 of 25 fold-fits, NEST_FAM's membership is constant within every cell, so its CELL control
    reproduces the rule there.
  - **Bulk precision fails, as expected:** the narrowing is 0.091 < 0.15.
- **Post hoc window.** The [-2, 2] window (H2, grid fixed after seeing V1) gives coverage 0.747, inside R² 0.540 and tail
  ratio 4.9 [2.3, 13.0].
- **Recipe correction.** The A2 fix moved every lower bound by at most 0.039, and no outcome changed.

### 3.3 E-DEC, E-REC and E-RANK: direction calls and ranking (descriptive; never an R² gain)

- **Valid fixed-sequence calibration calls very little.** Learn-then-Test at per-call risk 0.10 (hull rule, global
  thresholds) lets B1 sign **0.5 % of adjacent pairs with C5-BTBP in, and 13.0 % without it** (B0 0.4 % / 6.9 %; BP
  3.5 % / 16.9 %).
  - The single C5-BTBP publication collapses the calibration; the LOPO maximum is reached with it out.
  - **The realised rate is above nominal.** B1's per-call false-sign rate is 0.194 as the registered seed mean, above the
    nominal 0.10, and it rests on very few calls: pooled 4 of 44 calls = 0.091, 25 distinct queries (2 with a false
    sign), calls in 4 of 5 seeds, per-seed calls 23 / 15 / 3 / 0 / 3, below n_min 22 in 4 of 5 seeds.
- **CRC and plug-in let B1 call 84 % of pairs.** CRC controls only the per-pair (marginal) false-sign rate (realised
  0.046), and the plug-in has no guarantee; neither bounds the error rate among the calls themselves. The seen
  0.519 / 0.507 / 0.149 came from other implementations, and the difference is expected.
- **Hull rule:** 0 calls outside the read hull, and no Tm/Yb or Yb/Lu call without bracketing reads.
- **Ranking and net benefit:** AURC B1 0.040 vs B0 0.044; net benefit at p_t 0.3 +0.014 over B0.
- **Interval line (M16):** Mondrian coverage 0.919 at about ×/÷ 1.9.
- **E-REC (own prior at matched coverage).** This uses its own population, PRIMARY_OWN: 112 cells, 16 publications,
  18 units, 790 rows per seed, not the E-A rows. At matched coverage 0.495, SH_own@4 gives a false-sign rate of 0.003,
  against 0.025 for B1@4 (units 4/0/14; top-3 share 0.95).
- **E-RANK (reporting appendix; never a claim).** A straight line recovers 0.65 (Spearman), 0.67 (top-3) and 0.48
  (within-publication top-3) of B1's ranking skill over random (`results/EDEC/rank_line_share_<design>.csv`).

### 3.4 E-D: read-budget ladder and policy envelope (a different read budget, never E-A)

**B1 by read budget:**

| row (B1) | 4 reads | 5 reads, DOPT_Q-nested | 5 reads, without C5-BTBP |
|---|---|---|---|
| C3 min(k, k_max), k = reads - 1 | (E-A rows) | 0.4968 / 0.4988 (3/5 and 4/5 seeds ≥ 0.50) | 0.5311 / 0.5337 |
| C4 fixed cohort (178 cells, 1,516 rows per seed) | 0.4290 / 0.4288 | 0.4798 / 0.4821 (0/5) | 0.5171 / 0.5201 |

- **The C3 5-read row is a mixture.** 286 of its 1,802 rows per seed keep 4 reads, and those score 0.60. The
  like-for-like 5-metal number is C4, about 0.48.
- **One extra legal read** adds +0.045 / +0.047 (C3, paired on E-A's identical queries; 9 of 13 clusters gain, MDE80
  0.024) and +0.051 / +0.053 (C4).
- **Higher rungs are smaller fixed cohorts, not the 178-cell C4.**
  - The C5 cohort (k_max ≥ 5; 144 cells, 29 publications) reads 0.525 at 6 reads.
  - The C6 cohort (k_max ≥ 6; 133 cells, 26 publications) reads 0.432 / 0.484 / 0.529 / 0.556 at 4 / 5 / 6 / 7 reads.
- **C5-BTBP is uniquely decisive for C4.** One of 32 single-publication removals crosses 0.50, and that one is C5-BTBP.
- **Policy envelope.** The envelope width over the label-free policy family is 0.0091 / 0.0095. The contrasts are
  BISECT_R - DOPTQ +0.006 (MDE 0.021), RANDOM - DOPTQ -0.002 (0.012) and BISECT_R - RANDOM +0.008 (0.009).
- **Pareto question (pre-declared):** **no member of the written label-free family Pareto-dominates DOPTQ**, in BPG,
  BP and every C5 variant. ORACLE_B1 [O] meets the criterion only as a bound.
  - A2 corrected this outcome. The first-pass aggregate counted the oracle bound as a member; the per-policy table was
    right all along.
- **Saturation check fires, as expected** (k_max 4 at 5 reads 0.444, spread 0.158). It is a consistency check, not a
  kill.
- **Recipe-2.** The registered primary contrast, Ladder - HULL (B1, 2-5 reads), gives:
  - -0.027 at 2 reads (0 of 5 seeds);
  - +0.029 at 3 reads (5 of 5);
  - +0.042 at 4 reads (5 of 5);
  - -0.003 at 5 reads (1 of 5).

  BP is the same to 3 decimals.

### 3.5 E-W: wider pairs (never pooled with strict pairs)

- **B1 well above a line on wide frames.** On WIDE rows (225 cells, 38 publications, 12 units, about 4,390 rows per
  seed), B1 beats a straight line through the reads by a wide margin: dZ2 0.676 vs 0.447, dZ3-4 0.811 vs 0.562.
- **The registered Nd/Sm rule retires** "interpolation beats slope at the light|middle cut". The B1 - LINE contrast is
  -0.095 [-0.73, +0.54] over the 3 qualifying clusters, with the giant cluster removed; BP gives -0.097.
  - The pooled Nd/Sm advantage (+0.084, P = 34) sits in the giant cluster. Outside it, the advantage is +0.013
    [-0.076, +0.102].
  - What survives is the narrowing "a line through 4 reads suffices at Nd/Sm outside the giant cluster".
- **Heavy cuts (Ho-Lu) are explicit NON-CLAIMS**, split by whether Tm or Lu is read: pooled B1 0.40 vs LINE -1.04.

### 3.6 E-KX: the sc009 (TODGA-class) Nd dip

E-KX-LOPO decides here (D8). Its B1 base is the round-4 BP out-of-fold prediction; donors come from other publications
and other copy groups; it is never called BP. The BPG-base variant is a sensitivity. The rows are sc009 Pr/Nd (46 cells,
10 publications, one cluster, without C5-BTBP):

- **The chemotype dip helps at this pair.** B1 R² 0.265; the sc009 chemotype Nd term (KX_SC009) 0.508, which is +0.243
  with 5 of 5 seeds positive (BPG base +0.252); a universal Nd term (NDU) 0.415.
- **Implied Nd/Pr SF:** observed 1.34; B1 1.72; KX 1.43; NDU 1.58.
- **On pooled rows the chemotype term adds little over a universal one:** KX_CHEM +0.007 vs B1, NDU +0.005.
- **The rule narrows the claim to a "universal Nd term".** With the top-3 publications removed, R²(KX_CHEM) - R²(NDU)
  is -0.0012 (BPG base -0.0017).
  - 3 of the 9 rule sensitivities do not narrow, all on the sc009 Pr/Nd rows.
- **Diagnostics:** random-chemotype permutation p 0.0005 on sc009 rows (DIAGNOSTIC); sign-flip over the 10 sc009
  publications 7/3, p 0.034.
- **Frozen external value.** The value is -0.11 (u_ext -0.078). Its registered consistency line fails: the V1
  publication-first mean is -0.122, and the cell-weighted mean -0.110. The source computation is not identifiable.

## 4. Where are the data problems, and what did each request change?

- **Concentration.** One cluster (g:pub_74428aeab2) holds 89 % of E-CT's PRIMARY rows and 72 % of E-LAB's. Only 3 E-CT
  clusters exist, so cluster-level inference is "not estimable".
- **C5-BTBP (pub_6a0ef8528c).** This single publication decides several descriptive numbers:
  - E-DEC's LTT coverage (0.5 % vs 13 %);
  - C4's 5-read level (0.48 vs 0.52);
  - the window's tail flag on BP;
  - E-CT's sibling-count dose-response.

  It is reported in and out everywhere.
- **Duplicates and near-copies.** The Stage-0 bit-identity rule removed 6 links. E-LAB's near-copy cleaning lowers its
  seen gain. The 11 V2-3 condition-aware cells reverse B1's Gd gain (S1 sign-off open).
- **Laboratory key.** The key is PROVISIONAL (U6 open), so "same publication" reads as "same study or campaign".
- **Requests.** Every request is **OPEN**: U1-a … U1-i, U2, U3, U4, U5, U6, S1 and H1, as shown on 2026-10-24. None was
  answered, so none changed a result. The supply-audit kill (D4) is re-read after each answer
  (`RESUME.md` §3.2).

## 5. Remaining uncertainty, and what would change it

- **E-CT is a 12-publication, 3-cluster, mostly-DGA result** whose gain leans on one acid-type publication (43 %).
  - What would change it: new laboratories with own series of an extractant at ≥ 3 conditions, in at least 7 chemotype
    units, under the sealed C-ECT. Until then the honest wording is "own-series template + 3-4 reads helps on this
    corpus".
- **E-A cannot move without new or cleaner data.** Candidates are U4 re-measurement or V3-style corrections, reported as
  sensitivities. Effects below about 0.01 R² are invisible here.
- **E-LAB and M03 pass in-corpus checks but are fragile.** Both need their own external registration.
  - E-LAB (MX_LM10 - B1): leaving out the most helpful pair of publications leaves relative SSE +0.006.
  - M03 (E-CT annex, nested - SH on PRIMARY_NO_ORNL): the re-nested leave-one-publication-out minimum is ΔR² -0.001.
- **E-DEC's valid calls are nearly nil with C5-BTBP in.** More calibration clusters, not a different model, would change
  that.
- **E-KX's chemotype claim needs new sc009 publications** (the external sign rider, data-gated).
- **NOT RUN for lack of data:**
  - **The Gd-dip transport claim (M33), with C-GD as its cation-exchanger stratum.** The supply-audit kill fired (D4):
    C-GD has 1 of the 6 laboratories it needs, and Gd-dip transport 5 of 10 laboratory × chemotype units. Both are
    descriptive intervals under D3 (i), not tests.
  - The A3 positive control (B1 - B0 prospective), because there is no new data.
  - C-ECT-S01, data-gated with C-ECT (D10).

  What would change these: new cation-exchanger laboratories (U2 SAFE export, U5 literature supply), at least 6
  laboratories from ≥ 3 families for C-GD.
- **Bootstrap tail fractions**, such as P(R² ≥ 0.50), are descriptive, never posterior probabilities.

## Evidence and reproducibility

- **Seals.** Stage-0 seal `b96c3bef9828` (`manifests/stage0_seal.txt`; archives `stage0_manifest_1f0cae0064a6.json` and
  `stage0_manifest_2f74292f5f44.json`). `AMENDMENTS.md` A1 and A2 list every result seen before each reseal.
  `RUN_RECORD.md` holds every scoring and seal START / END line (UTC).
- **First pass and rerun.** First-pass outputs are archived in `cache/_superseded_2f74292f5f44/` (and the A1-era E-D
  outputs in `cache/_superseded_1f0cae0064a6/`).
  - The A2 rerun reproduces every stored number except the changes A2 lists (`results/a2_rerun_comparison.txt`).
  - Two first-pass EA_CHECK files the A2 run does not rewrite (`summary_BPG.json`, `summary_BP.json`) were removed from
    `results/`. Their archived copies remain, and the current summary is `results/EA_CHECK/summary.json`.
  - Row-level outputs and per-item tables are in `results/<ITEM>/` (`PER_ITEM_TABLE_<design>.md` or `ITEM_TABLE.md`).
- **Leakage controls** (`results/leakage/`; E-D's isolation counters are in `results/ED/checks_<design>.json`). Every
  recorded check passed:
  - **Isolation:** `assert_test_isolation` passed wherever it was run. That is EA_CHECK 100 / 100 over designs A, B, BR
    and BQ (plus BP and BPG), and E-D, E-W and WINDOW 25 / 25 per design.
    - E-KX's donor sets all pass `Donors.guard`.
    - E-LAB isolation is not applicable by design (its donors are the target's own campaign, as its information
      contract allows).
    - E-CT, M03 and E-DEC reuse round 4's fold fits and isolation; no new isolation check was run for them.
  - **Prediction intervention** with positive controls caught: E-CT 650 / 650 per control; E-A block-Y peeks
    100 / 100 folds.
  - **Fit poisoning:** E-D's k-specific β refits 50 / 50 identical, matching the main run; M03's nested selection;
    WINDOW's operating points; E-DEC's calibration thresholds.
  - **Cache poisoning:** E-A fold records 100 / 100.
- **Tests.**
  - The full suite gives 384 passed, 1 skipped under seal `b96c3bef9828` (RUN_RECORD 14:47-14:50 UTC;
    `results/logs/tests_post_A2.log`). Before the first seal it gave 352 passed, 1 skipped
    (`results/stage0/tests_preseal.log`).
  - The skipped test is the sealed P1 reproduction test. It passed under seal `b96c3bef9828`
    (`results/logs/ECT_A2_p1_test.log`).
- **Commands.** `RESUME.md` §3.
