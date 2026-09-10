# L5 — covariance shrinkage and honest intervals

**DISCOVERY.**  Five discovery seeds `(104729, 130363, 155921, 196613, 262147)`, all five designs
(B, BR, BQ, A, BP), frozen gen13 metrics and chemotype-blocked paired bootstrap
(10 000 replicates, seed 8675309).  Protocol: `PRE_REGISTRATION.md` §3 L5, unchanged.
Every number below is **extractant-macro MAE of pairwise `log SF`** unless it says otherwise;
BP is the design that selects.

## 0. Headline

1. **No registered contrast passes P1 in any design, let alone all five.**  Every effect that is
   real here is between 0.003 and 0.015 against a pre-registered margin of 0.020.
2. **Gen15 §7's `MIX6meanPC` gain survives the deployed baseline** — the comparison gen15 did not
   run.  At k = 3 it is **+0.0070 (BP)**, +0.0079 / +0.0077 / +0.0076 / +0.0076 (B / BR / BQ / A),
   p < 1e-4 in all five, both intervals exclude zero in all five, 5/5 seeds, LOCO-sign-stable,
   and the sign is the same in all five designs.  It is still **3× under the margin** and
   therefore **not a P1 claim**.  It is the only arm in this lead that is positive, significant
   and design-invariant.
3. **Ledoit–Wolf is not merely unhelpful here, it is catastrophic** — BP MAE 34.4 at k = 3 against
   POOLED's 0.167 — and the mechanism is measured, not guessed: the deployed residual covariance
   is **indefinite on 100 % of its 430 leave-chemotype-out estimates**, LW deepens the negative
   directions, and greedy D-optimal selection *seeks* them by construction.
4. **The programme's measured-mode intervals are conservative by roughly a factor of 2.5 in
   width**: nominal 90 % intervals cover **98.8 %** at k = 3 under BP.  The zero-shot (k = 0)
   interval is nearly honest: **92.6 %** at nominal 90 %.  The dominant cause is the fixed
   measurement-variance constant `NOISE_VAR = 0.09`, which by itself exceeds the total realised
   squared error at k = 3 (RMSE 0.248, i.e. 0.061).
5. **The calibrated zero-shot direction probability is NOT a deliverable** under the registered
   rule.  The pre-registered fix (inner-fold Platt) makes calibration *worse* in every design.
   The **raw** probability is the better object: it beats the constant base rate on Brier under
   P1 in all five designs (exploratory family), but its ECE under BP is 0.139, above the
   registered 0.10 bar.

## 1. Reproduction checks (all passed before any arm was scored)

| check | expected | obtained | source |
|---|---|---|---|
| `POOLED` estimator == `gen15.fewshot.residual_covariance` | bit-identical | max abs diff **0.0** | `verify.json` |
| pairwise Ledoit–Wolf == `sklearn.covariance.LedoitWolf` on complete rows | identical | Δshrinkage 3.5e-18, Δcov 4.4e-16 | `verify.json` |
| `l5_fewshot.evaluate` default estimator, BP, `G14@doptk3` | 0.1700393901 | **0.1700393901** (diff 0.0e+00) | `g15_support_board_empirical_BP.csv` |
| `l5_fewshot.evaluate` default estimator, BP, `G14@doptk1` | 0.2313450953 | **0.2313450953** (diff 8.3e-17) | same |
| whole pair table vs a fresh run of frozen `gen15.fewshot.evaluate` | identical | 57 638 pairs, 46 shared float columns, max abs diff **0.0** on predictions, 2.2e-16 on `sd` | `verify.log` |
| mixture **training route** port vs gen15's locked board (BP, `how='dopt'`) | 11 modes | **all 11 identical to 6 dp**, `MIX6meanPC@k3` 0.185190, `POOLED@k3` 0.192192 | `mixture_trainroute_repro.json` |

**Documentation defect found in gen15.**  `GEN15_REPORT.md` §7's table prints
`MIX6meanPC@k3 = 0.1865`, but its own `g15_mixture_board_dopt.csv` says **0.185190** (0.1865 is
`MIX4meanPC`).  The gain quoted in the same paragraph (+0.0070 = 0.192192 − 0.185190) matches the
CSV, so the CSV is right and the table cell is a transcription slip.  This brief's L5 instruction
inherited the slip.  Everything below uses the CSV.

## 2. How the estimators were made comparable

All five estimators ran in **one** `evaluate` pass per design, so the **union** of every
(strategy × estimator) D-optimal support set is excluded from scoring for every arm and every arm
is scored on byte-identical pairs.  No pair-table intersection was needed and nothing was dropped
after a number was seen.

Cost of that union, honestly stated: the deployed protocol's three strategies (widest, dopt,
random) with one estimator score **57 638** BP pairs; with five estimators the union removes
**4 976** more, leaving **52 662**.  `G14_POOLED@doptk3` is therefore **0.1674** on this common
set against **0.1700** on gen15's — the remaining pairs are marginally easier.  Every contrast
below is within-run and unaffected.  The three support strategies are all *selected* (that is what
fixes the scoring set) but only the registered `dopt` modes are *scored*.

Invariant checked per design: at k = 0 no measurement is used, so the covariance cannot matter —
all five estimator arms and `MIX6meanPC` agree with `G14_POOLED@doptk0` to < 1e-9
(`covrun_checks.json`).

## 3. The estimator table — five designs × k (extractant-macro MAE, dopt support)

| arm | k | B | BR | BQ | A | **BP** |
|---|---|---|---|---|---|---|
| `POOLED` (deployed) | 0 | 0.4010 | 0.4002 | 0.4024 | 0.3974 | **0.4052** |
| | 1 | 0.2169 | 0.2204 | 0.2190 | 0.2196 | **0.2238** |
| | 2 | 0.1937 | 0.1951 | 0.1944 | 0.1925 | **0.2009** |
| | 3 | 0.1638 | 0.1646 | 0.1638 | 0.1651 | **0.1674** |
| `LW` | 1 | 0.2193 | 0.2191 | 0.2238 | 0.2179 | 0.2214 |
| | 2 | 0.9574 | 0.7471 | 0.6849 | 2.8216 | **0.6539** |
| | 3 | 79.76 | 34.33 | 107.58 | 60.96 | **34.41** |
| `LOWRANK2` | 1 | 0.2139 | 0.2156 | 0.2170 | 0.2156 | **0.2166** |
| | 2 | 0.1849 | 0.1844 | 0.1850 | 0.1876 | **0.1862** |
| | 3 | 0.1664 | 0.1687 | 0.1672 | 0.1696 | **0.1667** |
| `LOWRANK3` | 1 | 0.2159 | 0.2176 | 0.2185 | 0.2164 | 0.2191 |
| | 2 | 0.1863 | 0.1863 | 0.1866 | 0.1884 | 0.1895 |
| | 3 | 0.1686 | 0.1691 | 0.1689 | 0.1708 | 0.1683 |
| `HIER` | 1 | 0.2147 | 0.2175 | 0.2158 | 0.2158 | 0.2205 |
| | 2 | 0.1946 | 0.1934 | 0.1931 | 0.1967 | 0.1963 |
| | 3 | 0.1719 | 0.1729 | 0.1731 | 0.1719 | 0.1711 |
| **`MIX6meanPC`** | 1 | 0.2125 | 0.2155 | 0.2158 | 0.2140 | 0.2236 |
| | 2 | 0.1878 | 0.1893 | 0.1893 | 0.1860 | 0.1974 |
| | 3 | **0.1560** | **0.1568** | **0.1562** | **0.1576** | **0.1604** |
| `NAIVE_LINE` (cheapest competitor, POOLED's dopt pairs) | 1 | 0.2322 | 0.2364 | 0.2338 | 0.2330 | 0.2310 |
| | 2 | 0.2257 | 0.2246 | 0.2267 | 0.2257 | 0.2251 |
| | 3 | 0.2231 | 0.2238 | 0.2236 | 0.2223 | 0.2243 |
| `FLAT` prior + POOLED BLUP (floor) | 3 | 0.1704 | 0.1708 | 0.1705 | 0.1707 | 0.1742 |

Full board with the whole frozen metric panel: `board_cov.csv` (295 rows = 5 designs x 59 arms),
`board_mix.csv` (55 rows).

**Against the cheapest competitor (exploratory, BP).**  `POOLED − NAIVE_LINE` = +0.0069 (k = 1,
n.s.), +0.0230 (k = 2, p = 0.030), **+0.0583** (k = 3, p < 1e-4, 5/5 seeds, LOCO-stable, passes
P1); `MIX6meanPC − NAIVE_LINE` = **+0.0653** at k = 3.  Gen15's locked +0.062…+0.065 at k = 3 is
reproduced on this scoring set.  The chemistry prior itself is worth almost nothing once a
measurement exists: `POOLED − FLAT` is +0.0031 / +0.0031 / +0.0066 at k = 1 / 2 / 3 under BP,
none significant, against +0.0751 at k = 0.

## 4. Registered contrasts and P1 verdicts

`contrasts_cov.csv` (60 registered + 115 exploratory rows), `contrasts_mix.csv` (15 registered +
15 exploratory).
Positive favours the candidate.  Margin 0.02.

| comparison | BP point | BP 95 % CI | BP p (BH within lead) | designs with the same sign | P1 in ≥ 1 design | **P1 in all 5** |
|---|---|---|---|---|---|---|
| `LW_vs_POOLED@k1` | +0.0012 | [−0.0146, +0.0114] | 0.925 (0.949) | 3/5 | no | **no** |
| `LW_vs_POOLED@k2` | **−0.4501** | [−0.844, −0.207] | <1e-4 (<1e-4) | 5/5 (all negative) | no | **no** |
| `LW_vs_POOLED@k3` | **−34.02** | [−71.4, −14.2] | <1e-4 (<1e-4) | 5/5 (all negative) | no | **no** |
| `LOWRANK2_vs_POOLED@k1` | +0.0074 | [−0.0026, +0.0149] | 0.192 (0.342) | 5/5 positive | no | **no** |
| `LOWRANK2_vs_POOLED@k2` | **+0.0151** | [+0.0067, +0.0232] | 0.002 (0.010) | 5/5 positive | no | **no** |
| `LOWRANK2_vs_POOLED@k3` | +0.0003 | [−0.0060, +0.0052] | 0.916 (0.949) | 1/5 (sign flips) | no | **no** |
| `LOWRANK3_vs_POOLED@k1` | +0.0045 | [−0.0074, +0.0133] | 0.610 (0.739) | 5/5 positive | no | **no** |
| `LOWRANK3_vs_POOLED@k2` | +0.0120 | [+0.0018, +0.0198] | 0.023 (0.074) | 5/5 positive | no | **no** |
| `LOWRANK3_vs_POOLED@k3` | −0.0020 | [−0.0125, +0.0039] | 0.607 (0.739) | 5/5 negative | no | **no** |
| `HIER_vs_POOLED@k1` | +0.0017 | [−0.0146, +0.0114] | 0.817 (0.896) | 5/5 positive | no | **no** |
| `HIER_vs_POOLED@k2` | +0.0031 | [−0.0122, +0.0119] | 0.605 (0.739) | 4/5 | no | **no** |
| `HIER_vs_POOLED@k3` | −0.0053 | [−0.0209, +0.0036] | 0.319 (0.521) | 5/5 negative | no | **no** |
| `MIX6meanPC_vs_POOLED@k1` | +0.0002 | [−0.0029, +0.0032] | 0.880 (0.927) | 5/5 positive | no | **no** |
| `MIX6meanPC_vs_POOLED@k2` | +0.0036 | [+0.0002, +0.0062] | 0.040 (0.099) | 5/5 positive | no | **no** |
| **`MIX6meanPC_vs_POOLED@k3`** | **+0.0070** | **[+0.0044, +0.0111]** | **<1e-4 (<1e-4)** | **5/5 positive** | no | **no** |
| `PLATT_vs_CONST` (Brier) | +0.0578 | [−0.0018, +0.1065] | 0.056 (0.125) | 5/5 positive | yes (B, A) | **no** |

`which_pass = []`.  `any_registered_contrast_passes_P1_all_designs = false`.

**What the table means, arm by arm.**

* **`MIX6meanPC` at k = 3 is the one real effect.**  All five designs, p < 1e-4, both percentile
  and BCa intervals exclude zero, 5/5 seeds, LOCO-sign-stable, 57–62 of 84 extractants improved (BP 57/84).
  It answers the brief's question — the §7 gain *does* survive the deployed leave-chemotype-out,
  publication-masked baseline, and it is the same size there (+0.0070) as on the training route
  (+0.0070).  It fails P1 only on the pre-registered 0.02 margin, and the margin is not
  negotiable.  Note it is a **k = 3 effect only**: at k = 1 under BP it is +0.0002.
* **`LOWRANK2` is a k = 2 effect that reverses at k = 3.**  Positive in all five designs at k = 1
  and k = 2 (BP +0.0074, +0.0151), CI excluding zero at k = 2 in B / BR / BQ / BP but not A, then
  the sign flips design-to-design at k = 3.  A candidate whose sign is not stable across the
  measurement budget is not a candidate.
* **`HIER` is null at k = 1–2 and significantly *worse* at k = 3** in B, BR, BQ and A
  (−0.0074 to −0.0097, p 0.005–0.030).  Chemotype-balancing the residual rows costs information:
  the diglycolamides supply most of the well-measured curves and down-weighting them to one
  chemotype's worth of mass makes the covariance noisier, which shows up exactly where the
  covariance does the most work.
* **`LW` fails catastrophically** — see §5.

## 5. Why Ledoit–Wolf explodes: the deployed covariance is indefinite

`covariance_conditioning.csv`, 430 leave-chemotype-out publication-masked covariance keys per
design, median 512 residual rows each:

| design BP | median min eig | worst min eig | keys indefinite | frac. of pair contrasts with `d'Σd ≤ 0` | with `d'Σd ≤ −0.09` |
|---|---|---|---|---|---|
| `POOLED` (deployed) | **−0.317** | −0.459 | **100 %** | 0.000 | 0.000 |
| `LW` | **−0.486** | −0.701 | **100 %** | **0.0079** | **5e-5** |
| `LOWRANK2` | +0.0278 | +0.0213 | 0 % | 0.000 | 0.000 |
| `LOWRANK3` | +0.0211 | +0.0158 | 0 % | 0.000 | 0.000 |
| `HIER` | −0.098 | −0.148 | 100 % | 0.000 | 0.000 |

The deployed `residual_covariance` — a pairwise-complete second moment of row-centred curves with
35–80 % missingness — **has always been indefinite**, in every design, on every key.  Nothing in
gen13–gen15 ever looked.  It is safe only because the BLUP never inverts Σ; it solves
`D Σ D' + noise·I`, and *no measurable pair contrast* `d = e_a − e_b` has `d'Σd ≤ 0` under it.
The negative eigenvalues live outside the pair-contrast cone.

Ledoit–Wolf breaks that accident.  It shrinks toward `μ·I` with a data-driven intensity that is
tiny at n ≈ 512, and `μ·I` is a *worse* target than the diagonal `residual_covariance` uses,
because the diagonal carries the real variance profile along the series; replacing it with an
isotropic one deepens the negative directions (median min eigenvalue −0.317 → −0.486).  Now
0.79 % of pair contrasts have non-positive variance and 0.005 % fall below `−noise_var`.  That
fraction is small and it is fatal, because **greedy D-optimal selection maximises
`(Σd)'(Σd) / (d'Σd + noise)`, a criterion that diverges exactly where the denominator vanishes.**
The support chooser therefore *seeks out* the worst-conditioned contrast; at k ≥ 2 it picks one,
the rank-1 posterior update `post − vv'/(d'Σd + noise)` blows up, and the prediction with it.
This is a **support-selection × conditioning interaction**, not a shrinkage failure: LW at k = 1
is fine (BP 0.2214 vs POOLED 0.2238).  The same explosion appears in the `FLAT`-prior arm, so it
is the covariance, not the prior.

The corollary is the useful part: **`LOWRANK2` / `LOWRANK3` are the only estimators that are
positive-definite on every key**, and they are also the only ones that improve MAE at k = 1–2 and
the only ones whose interval calibration is better than the deployed estimator's at every k.
Probabilistic PCA repairs the indefiniteness by construction.

## 6. Interval calibration (registered)

`calibration_intervals.csv`.  Coverage is pooled over the scored pairs (`g15_uncertainty`'s
construction), one row per (design, arm, k).  Nominal 50 / 80 / 90 / 95 %.

**Design BP, the deployed `POOLED` arm:**

| k | pooled MAE | RMSE | mean sd (sharpness) | cover50 | cover80 | **cover90** | cover95 | median &#124;z&#124; |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.4729 | 0.6844 | 0.6557 | 0.607 | 0.846 | **0.9265** | 0.965 | 0.502 |
| 1 | 0.2434 | 0.3336 | 0.5223 | 0.760 | 0.953 | **0.9825** | 0.991 | 0.340 |
| 2 | 0.2096 | 0.2923 | 0.4899 | 0.795 | 0.962 | **0.9846** | 0.992 | 0.309 |
| 3 | 0.1760 | 0.2477 | 0.4677 | 0.839 | 0.975 | **0.9884** | 0.994 | 0.268 |

**cover90 at k = 3, all designs, all arms:**

| arm | B | BR | BQ | A | BP | mean &#124;cover90 − 0.90&#124; over k = 1,2,3 |
|---|---|---|---|---|---|---|
| `POOLED` | 0.990 | 0.990 | 0.990 | 0.990 | 0.988 | 0.0861 |
| `LOWRANK2` | 0.985 | 0.985 | 0.986 | 0.985 | 0.986 | **0.0810** |
| `LOWRANK3` | 0.985 | 0.985 | 0.985 | 0.985 | 0.984 | **0.0809** |
| `HIER` | 0.990 | 0.989 | 0.990 | 0.990 | 0.989 | 0.0880 |
| `MIX6meanPC` | 0.991 | 0.991 | 0.991 | 0.991 | 0.991 | 0.0887 |
| `LW` | 0.552 | 0.550 | 0.547 | 0.547 | 0.556 | 0.1876 |

**Registered verdict: `better_calibrated_than_pooled_all_designs` = [] (empty).**
`LOWRANK2` and `LOWRANK3` do satisfy the first half of the rule — a smaller `|cover90 − 0.90|`
than `POOLED` at every k ≥ 1 in all five designs — but **no arm satisfies the second half**: none
has 90 % coverage inside [0.85, 0.95] at any k ≥ 1.  `best_calibrated_arm = LOWRANK3` on the
`|cover90 − 0.90|` metric, by a margin (0.0809 vs 0.0861) far too small to matter.

**The mechanism, and it is one constant.**  The zero-shot interval is nearly honest: 92.6 %
coverage at nominal 90 % under BP, median `|z|` = 0.50 against 0.674 for a calibrated normal.
After a measurement the errors collapse (RMSE 0.684 → 0.248) but the predictive sd does not
(0.656 → 0.468), because the sd is floored by the fixed measurement-noise constant
`NOISE_VAR = 0.09` (sd 0.30), gen15's `2 × 0.212²` from the cohort's median within-replicate sd.
At k = 3 the **total** realised squared error is 0.248² = 0.061 — *smaller than the assumed
measurement variance alone*.  The interval is dominated by a noise term the held-out data do not
exhibit, so the deployed 90 % interval is roughly **2.5× too wide** (median `|z|` 0.268 against
0.674).  This is conservative, not dishonest, and it is a one-line fix the programme has never
made.  It is out of L5's registered scope — `NOISE_VAR` is a frozen constant of the deployed
route and changing it after seeing this table would be exactly the move the brief forbids — so it
is **recorded as a recommendation, not acted on**.

`LW`'s coverage inverts (0.556 at k = 3) because its posterior variance collapses along the same
near-singular direction its support chooser selects — the same defect, seen from the interval
side.

## 7. Zero-shot direction probability (registered)

`calibration_direction.csv`, `calibration_direction_contrasts.csv`, `direction_oof_cells.parquet`.
1 445 well-determined held-out cells (`n_metals ≥ 5`), 82 extractants, 40 chemotypes, 5 seeds.
Unit = extractant, block = chemotype.  `RAW` = `gen14.models.dir_logistic()` on `TOPO39`, verified
identical to `gen14.dirbench.run`'s out-of-fold `p` to < 1e-12 in every design.

| design | arm | macro Brier | ECE (10 equal-count bins) [blocked 95 % CI] | macro accuracy | mean p |
|---|---|---|---|---|---|
| BP | `RAW` | **0.1666** | 0.1393 [0.095, 0.228] | 0.8205 | 0.623 |
| BP | `PLATT` | 0.2550 | 0.3012 [0.182, 0.346] | 0.6671 | 0.477 |
| BP | `CONST` (registered constant) | 0.3128 | 0.3593 [0.162, 0.431] | 0.3473 | 0.370 |
| BP | `CONSTCELL` (exploratory, harder) | 0.2941 | 0.2187 [0.174, 0.306] | 0.5586 | 0.620 |
| B | `RAW` | 0.1427 | 0.0839 | 0.8400 | 0.668 |
| BR | `RAW` | 0.1491 | 0.0981 | 0.8346 | 0.664 |
| BQ | `RAW` | 0.1428 | 0.0932 | 0.8400 | 0.656 |
| A | `RAW` | 0.1367 | 0.0809 | 0.8346 | 0.670 |

**Registered rule: `calibrated_probability_is_deliverable = false.**  `PLATT_vs_CONST` is positive
in all five designs but its 95 % interval **includes zero in BR (p = 0.54), BQ (p = 0.075) and BP
(p = 0.056)**, and `PLATT`'s ECE is 0.13–0.30 everywhere, far above the registered 0.10 bar.

**The pre-registered recalibration makes calibration worse, in every design.**  `PLATT_vs_RAW` is
negative and significant in all five (BP −0.0884, p = 0.010; BR −0.1129, p < 1e-4), and macro
accuracy falls from 0.82 to 0.67 under BP.  The mechanism is the programme's standing constraint:
the fold plan's inner validation block holds about a quarter of the training chemotypes, i.e.
roughly **three effective units** out of a Kish n_eff of 11.7, and a two-parameter Platt map fitted
on three effective units is noise.  It also cannot always be fitted at all — the inner block fails
its two-class / size guard for **21 % of held-out cells under BP** and 16 % under BQ (those cells
fall through to `RAW`), which is itself a measure of how little inner data exists.

**What is true, in the exploratory family:** the **raw** probability beats the registered constant
under P1 in all five designs (`RAW_vs_CONST`, BP +0.1462 [+0.0386, +0.2078], p = 0.008, 5/5 seeds,
LOCO-stable) and also beats the harder unweighted-cell-rate constant added here
(`RAW_vs_CONSTCELL`, BP +0.1275 [+0.0457, +0.2425], p = 0.003, P1 in all five designs).  Its ECE
is ≤ 0.10 in B (0.084), BR (0.098), BQ (0.093) and A (0.081) — and **0.139 under BP, the design
that selects**.  So the honest statement is: *the deployed direction model's output is a
discriminative score that beats every constant, and is close to a usable probability under four
designs; under the publication-masked design it is over-confident toward the light-selective side
(mean p 0.623 against a base rate of 0.702) by more than the registered bar allows, and the
pre-registered recalibration does not fix it.*

## 8. Comparison count

| file | rows | registered | exploratory |
|---|---|---|---|
| `contrasts_cov.csv` | 175 | 60 | 115 |
| `contrasts_mix.csv` | 30 | 15 | 15 |
| `calibration_direction_contrasts.csv` | 25 | 5 | 20 |
| **total** | **230** | **80** | **150** |

Registered = 12 estimator contrasts (4 estimators × k = 1, 2, 3) × 5 designs = 60, plus
3 `MIX6meanPC_vs_POOLED` contrasts × 5 designs = 15, plus `PLATT_vs_CONST` × 5 designs = 5.
Exploratory = 23 MAE contrasts × 5 designs = 115 in `contrasts_cov.csv` (each estimator against
`NAIVE_LINE` at k = 1, 2, 3; `POOLED` against `FLAT` at k = 0…3; each estimator against `POOLED`
under a `FLAT` prior at k = 3), 3 `MIX6meanPC_vs_NAIVE` × 5 = 15 in `contrasts_mix.csv`, and
4 direction contrasts × 5 designs = 20.

Benjamini–Hochberg **within this lead** is in the `p_bh_lead` column of every contrasts file,
computed separately over the registered and the exploratory family; the fleet-wide adjustment is
the orchestrator's and uses the same rows.  After BH within the lead, the registered rows that
survive at BP are `MIX6meanPC_vs_POOLED@k3` (< 1e-4), `LOWRANK2_vs_POOLED@k2` (0.010), and the
two `LW_vs_POOLED@k2, k3` failures (< 1e-4, in the wrong direction).

## 9. Temptations recorded and not acted on

1. **Repair `LW`.**  When it returned MAE 34 I wanted to project its base matrix onto the PSD cone,
   or use complete-case rows, or use `sklearn.LedoitWolf` on zero-filled rows.  Any of those would
   probably make it competitive.  The arm is registered as "Ledoit–Wolf shrinkage to a scaled
   identity" and the stopping rule allows **these five covariance arms only**; a repaired LW is a
   sixth arm chosen after seeing a number.  Not done.  The failure is reported with its mechanism.
2. **Argue the 0.02 margin down for the measured mode.**  The margin was set against a zero-shot
   scale where `FLAT` is 0.589; in the measured mode the whole error is 0.167, so 0.02 is 12 % of
   the total error and nothing can clear it.  That argument is correct and it is still a
   post-hoc change to a registered decision rule.  Not done.  `MIX6meanPC@k3` is reported as
   *positive, significant, design-invariant and under the margin*, which is what it is.
3. **Quote `LOWRANK2` at k = 2 only.**  Its best rung passes every part of P1 except the margin in
   four designs.  Reporting only k = 2 would hide that its sign flips at k = 3.  All k reported.
4. **Drop the `cover90 ∈ [0.85, 0.95]` half of the calibration rule**, under which no arm can
   qualify and `LOWRANK3` would have been "better calibrated than POOLED in all five designs".
   Not done; the rule is reported as failing on the coverage band.
5. **Fix `NOISE_VAR`.**  Setting it to the realised pair-error variance would move the deployed
   interval from 98.8 % to near-nominal coverage in one line, and would also change every MAE in
   the measured mode (the BLUP weight depends on it).  That is a change to the deployed route
   after seeing the calibration table.  Not done; recommended in §10 for a pre-registered gen17
   arm instead.
6. **Substitute a harder constant for `CONST`.**  The registered constant is the *chemotype-
   balanced* training base rate (0.37 under BP), which is a weak comparator: the extractant-macro
   Brier of the unweighted cell base rate is 0.294 against its 0.313.  I did not replace it — I
   added `CONSTCELL` as an explicitly exploratory arm beside it, and confirmed the registered
   `PLATT_vs_CONST` numbers were bit-identical before and after the addition.  Adding a harder
   comparator cannot lower the bar; swapping a registered one would.

## 10. Recommendations

1. **Do not promote anything from L5 to confirmation on MAE.**  Nothing passes P1.  If the
   orchestrator wants the closest thing to a claim, it is `MIX6meanPC_vs_POOLED@k3` — positive,
   p < 1e-4, five designs, 5/5 seeds, LOCO-stable, **+0.0070 against a 0.020 margin** — and it
   should be frozen only if the margin question is settled *before* the confirmation seeds are
   opened, never after.
2. **The deployed measured-mode covariance should be `LOWRANK2`, not on MAE grounds but on
   conditioning grounds.**  It is the only estimator that is positive-definite everywhere, it is
   never worse than `POOLED` at k = 1–2 in any design, and it removes a latent failure mode that
   `LW` demonstrates is not hypothetical.  This is an engineering recommendation with a null MAE
   effect at k = 3, and it must be stated as such.
3. **`NOISE_VAR = 0.09` should be re-estimated.**  It exceeds the total realised squared error at
   k = 3.  A pre-registered gen17 arm that fits the pair-noise variance from the held-out residuals
   (or uses the per-cell `repsd__` route already implemented as `per_cell_noise`) would move the
   90 % interval from 98.8 % toward nominal and could move MAE as well, since the BLUP weight
   depends on it.
4. **Report the direction model's number as a score, not a probability, under BP.**  It beats every
   constant on Brier under P1 in all five designs; its ECE is publishable under B / BR / BQ / A and
   not under BP.  An honest deployment says "0.82 macro accuracy, ECE 0.14 under the
   publication-masked design", not "here is a calibrated probability".

## Files

| file | what |
|---|---|
| `board_cov.csv` | 295 rows = 5 designs × 59 arms, full frozen metric panel |
| `board_mix.csv` | 55 rows: the `MIX6meanPC` / `POOLED` / `NAIVE_LINE` slice |
| `board_mix_trainroute.csv` | gen15 §7's training route, reproduced |
| `contrasts_cov.csv` | 175 rows (60 registered, 115 exploratory) + `p_bh_lead` |
| `contrasts_mix.csv` | 30 rows (15 registered `MIX6meanPC_vs_POOLED`, 15 exploratory vs `NAIVE_LINE`) |
| `calibration_intervals.csv` | 120 rows (6 arms × k = 0…3 × 5 designs): coverage 50/80/90/95, sharpness, `abs_cover90_gap` |
| `calibration_direction.csv` | 20 rows (4 arms × 5 designs): Brier, ECE + blocked CI, macro accuracy |
| `calibration_direction_contrasts.csv` | 25 rows (5 registered, 20 exploratory) |
| `covariance_conditioning.csv` | 25 rows: eigen-spectra of all five estimators on the real rows |
| `L5_VERDICTS.json` | every registered decision rule, evaluated in code |
| `verify.json`, `verify.log`, `verify_final.log` | the reproduction gate |
| `mixture_trainroute_repro.json`, `mixture_training_route_publication_audit.csv` | §7 port checks |
| `perext_cov_*.parquet`, `perext_direction_*.parquet`, `direction_oof_cells.parquet` | per-unit tables |
| `covrun.log`, `mixrepro.log`, `direction.log`, `condition.log` | run logs |

Code: `gen16_leads/gen16/l5_cov.py`, `l5_fewshot.py`, `l5_mixture.py`;
scripts `gen16_leads/scripts/l5_verify.py`, `l5_covrun.py`, `l5_mixrepro.py`, `l5_direction.py`,
`l5_condition.py`, `l5_summary.py`.  Nothing under `gen13_separation/`, `gen14_direction/`,
`gen15_curve/` or `src/` was modified.
