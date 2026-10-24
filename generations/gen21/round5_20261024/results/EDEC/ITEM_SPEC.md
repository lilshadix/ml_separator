# Item 1.5 specification: E-DEC selective direction calls (M14), the E-REC own-prior row (recipe-3), the E-RANK table (M15)

Frozen at the Stage-0 seal, before the item's first post-seal run (brief 1.2: Stage-0 seal -> item code seal -> reproduction run labelled TECHNICAL SMOKE -> arms; the pre-seal --smoke runs are PRE-SEAL ENGINEERING CHECKS, PROTOCOL 4 and 10). Every in-corpus number is EXPLORATORY (seen). The item is descriptive: no kill fires, no in-corpus p-value is called significant, a direction-call number is never presented as an R² gain and never sits next to E-A's 0.452. Code: `r5/edec.py`, `scripts/s1_edec.py`, `tests/test_edec.py`; the E-REC arms come unchanged from `r5/ect.py` (item 1.1). Sources: brief 6.5, 6.0, 6.10, 8, 9; IR M14, M15, M16, M23 (`ideas/IDEAS_RANKED.md`, `ideas/critiques.json`); I2 recipe-3 (`ideas2/IDEAS2_RANKED.md`, `ideas2/recipe.md`). The recipe-3 comparator conventions follow `ideas2/scripts/recipe3_skepB_signcheck.py` (LF SHA-256 `c9ae42cc2b54ea3c6cb2ddb676b0eb4b76b3a5e5b5bfa5be631e0c4a3e2aebb6`; re-implemented, not copied, never run; the script records and re-checks this digest).

## Rows

- E-A rows: `r5.common.load_rows(design)`, the round-4 REF OOF parquet restricted to C3 at k = 3 (four DOPT_Q reads), 229 cells, 1,802 strict-adjacent queries per seed, 5 dev seeds, 13 `cluster_cpg` clusters, 39 publications. BPG decides; BP is a labelled sensitivity (same test folds).
- `prepare` asserts the support contract: no scored pair with both ends read, strict adjacency (Nd/Sm absent), every pair one of the 12 pair types.
- The E-REC rows are the `r5.ect` rows on the identical queries (results/ECT/rows_<design>.parquet when it carries the current seal, else `r5.ect.Corpus().build_rows`). If `r5.ect` cannot be imported or its corpus cannot be built, the E-REC row is reported as PENDING-ECT and nothing else changes.

## Decision rule and risk

- A prediction p of log SF(A/B) (A the lighter metal) makes a three-way decision at a threshold tau: lighter (p >= tau), heavier (p <= -tau), inseparable / measure (|p| < tau). p = 0 (or NaN) is never a call.
- A **false sign** is a call whose sign differs from y's with |y| >= 0.10. A call on a pair with |y| < 0.10 is never a false sign.
- Reported risks: per-call (selective) false-sign rate = false signs / calls; per-pair (marginal) rate = false signs / rows; the false-sign rate among calls with |y| >= 0.10; the share of wrong-sign calls with |y| < 0.10; sign accuracy among calls. Coverage = calls / all rows (the out-of-hull rows stay in the denominator) and calls / hull-eligible rows.

## Hull rule (per-pair-type validity, the primary report)

- A pair may be called only inside the read hull: a read metal at or below A and one at or above B (computed from the read set only, label-free). This is the bracketing rule; it is what "no Tm/Yb or Yb/Lu calls without bracketing reads" means for every pair type. Calling every row (no hull rule) is a SENSITIVITY.
- `pairtype_<design>.csv` reports, per pair type, rows and out-of-hull rows per seed, the separable share, calls, coverage, every risk above, for every hull-rule configuration.

## Calibration (valid; never on the rows it calls)

- **Leave-one-cluster-out**, inside each seed: the threshold applied to cluster c is fitted on the same seed's OOF rows of the other 12 clusters (their hull-eligible rows under the hull rule). Global thresholds are primary; **Mondrian** thresholds (one per pair type, the same LOCO rule) are SECONDARY and run under the hull rule only.
- Grid: tau in {3.000, 2.995, ..., 0.005} (600 values, descending).
- **LTT (primary)**: Learn-then-Test. H0(tau): per-call risk > alpha. p(tau) = BinomCDF(k(tau); n(tau), alpha) on the calibration calls (n calls, k false signs). Fixed-sequence testing at delta = 0.10, starting at the largest tau with n(tau) >= n_min = ceil(log delta / log(1 - alpha)) (22 at alpha 0.10, 45 at 0.05; the start depends on calibration scores only), rejecting while p <= delta. tau_hat = the last rejected tau; no rejection -> no calls.
- **CRC** (beside it): conformal risk control on the monotone, bounded per-pair loss l_i(tau) = 1[call_i(tau)] 1[false sign_i]: tau_hat = min{tau : (K(tau) + 1) / (N + 1) <= alpha}, N the calibration rows and K their false-sign calls. It controls the per-pair (marginal) false-sign rate, not the per-call rate (the per-call ratio is not monotone, IR M14 skeptic B); both rates are reported.
- **PLUGIN** (beside it, no guarantee): tau_hat = min{tau : n(tau) >= 1 and k(tau) / n(tau) <= alpha}.
- alpha 0.10 primary; alpha 0.05 SENSITIVITY. The LTT and CRC guarantees hold for exchangeable rows; clusters are not exchangeable (Kish 2.24), so realised per-cluster rates are reported.
- **Primary configuration**: `LTT_a10_hull_glob` (LTT, alpha 0.10, hull rule, global thresholds).

## Arms

| arm | role |
|---|---|
| B1 | primary set (round-4 fold-fitted Gd amplitude) |
| B0 | primary set; the cheapest same-information comparator (IR rule 3) |
| HBK@ML | primary set |
| RIDGE2 | comparator |
| LINE | comparator (slope only; Platt-scaled in net benefit; Mondrian thresholds show its pair-type-threshold behaviour) |
| G14@ZERO | comparator (zero-shot) |
| HIST_D7 | DIAGNOSTIC (fixed -0.072; not fold-independent) |

Configurations: every arm x {LTT, CRC, PLUGIN} x alpha {0.10, 0.05} x {hull + global, hull + Mondrian, all rows + global} = 126 call sets per C5 stratum and design. Contrasts reported: B1 - X for every other arm and configuration (coverage and per-call rate, per seed).

## Risk-coverage

- Rows ordered by |p|; the expected per-call false-sign risk after k calls with ties at equal |p| shared (the expectation over random tie-breaking); AURC = its mean over k = 1..n (callable rows); also on hull-eligible rows only. E-AURC = AURC - AURC of the oracle order (false signs last), labelled ORACLE [O]. The full-coverage per-call rate. Curve at coverage 0.05, 0.10, ..., 1.00. Arms: all seven.

## Net benefit and calibration-in-the-large

- Decisions: "separable" (event |y| >= log10 2, SF >= 2) and "inseparable" (event |y| < 0.10). P(event) by Platt scaling of |p| (logistic in |p|, intercept and slope, Newton with ridge 1e-6), fitted leave-one-cluster-out per seed.
- NB(p_t) = TP/n - FP/n x p_t / (1 - p_t) at p_t in {0.1, 0.2, 0.3, 0.5}, against treat-all and treat-none; arms B1, B0, HBK@ML, RIDGE2, LINE (Platt-scaled LINE); dNB = B1 - arm per seed.
- Calibration-in-the-large per cluster: observed event rate, mean predicted probability, O - E, O / E (per arm and decision); every fitted Platt (a, b) is written.

## Interval line (M16; one line)

- B1 split-conformal intervals on |y - B1|, row-weighted, Mondrian by pair type. Nominal level chosen by nested leave-one-cluster-out: for held-out cluster c, an inner LOCO over the other 12 clusters gives the pooled inner coverage for every nominal in {0.80, 0.85, 0.90, 0.91, ..., 0.99}; the smallest nominal reaching 0.90 is chosen (0.99 when none does) and calibrated on all other clusters. Quantile rank ceil((n + 1) level); a rank beyond n gives an infinite half-width (reported as a share). Reported: pooled coverage per seed, mean and median half-width, the x/÷ factor 10^(mean half-width), the chosen nominal per (seed, cluster), coverage per cluster.

## One-sided bounds (M23; one line, "C5-BTBP floor sensitivity")

- User-stated limit L = -4.0 log D (the ORNL clip and the C5-BTBP floor). Rows whose read end is at or below L: a read A end gives the upper bound y <= B1 + q_up; a read B end the lower bound y >= B1 - q_dn; q_up / q_dn the 0.95 one-sided split-conformal quantiles of B1's signed residual on the same seed's other clusters. Only the read values at the read end are used (legal inputs). Reported: rows per seed (upper / lower), the C5-BTBP share, coverage per seed, mean offset.

## Measurements saved (cost scenario stated with every number; brief 6.9 / I2 rule 16)

- ICP-MS (all 14 Ln from one experiment): 0 by construction (the 4-read experiment measured every metal).
- Subset feed / radiotracer with one full follow-up experiment per cell: the share of cells whose every scored pair is called.
- Radiotracer, one metal per measurement: the share of unread pair-end metals not needed to resolve the uncalled pairs.
- Literature mining: pairs signed without a new measurement. All at the primary configuration for B1, B0, HBK@ML.

## E-RANK (M15 salvage; reporting appendix; never a claim)

- Candidate groups: (seed, fold, pair type) with >= 5 test cells; within-publication groups: (seed, fold, pair type, publication) with >= 3. Candidates ranked by |p|.
- Endpoints: top-1 hit (the arm's top choice has the largest |y|; ties at the top shared), top-3 hit (the true best inside the arm's top 3, boundary ties shared), regret = max |y| - |y| of the top choice (log units), Spearman(|p|, |y|). Group means, then seed means.
- Arms B1, B0, LINE, RIDGE2, HBK@ML, G14@ZERO; RANDOM = the exact expectation; AMP_ORACLE [O] = ranking by the cell's mean |y| over its other scored pairs in the seed.
- LINE share per endpoint = (LINE - RANDOM) / (B1 - RANDOM) (regret: (RANDOM - LINE) / (RANDOM - B1)), with the leave-one-cluster-out range.

## E-REC own-prior row (recipe-3 salvage)

- Population PRIMARY_OWN: targets with >= 1 CLEAN sibling after the section-5.1 bit-identity rule (`in_GE1`: n_sib >= 1, C5-BTBP out), the quantised pub_a49146c8fd out; rows where every arm is finite (identical rows).
- Arms: COPY_own = `COPY` (0 reads), SH_own@2 = `SH_r2`, SH_own@4 = `SH_r4` (r5.ect; shrunk toward B1 on the same reads); CHPT_MAJ = the median y of other publications' rows of the same chemotype and pair type (>= 3 rows, else of the pair type; C5-BTBP never a donor; one row per (cell, pair)); G14 = `G14Z`; D7@2 / D7@4 = B1's form with the fixed -0.072 at 2 / 4 reads, D7_r = B0_r + (-0.072 / beta_fold)(B1_r - B0_r) (exact; D7@4 must equal REF HIST_D7); B1@4 = `B1_r4`.
- Metrics per seed on identical rows: AURC and E-AURC [O]; the per-call false-sign rate and sign accuracy at matched coverage, each arm calling its own top-|p| rows (boundary ties shared) at c_ref = the share with |COPY| >= 0.10 (label-free) and at 0.25 / 0.50 / 0.75; the 0.10 rule: coverage, per-call rate, three-class accuracy (lighter / heavier / inseparable at +-0.10) and the opposite-direction call rate.
- Contrasts (fs@c_ref and AURC, reference minus arm, per seed): B1@4 - SH_own@4, B1@4 - SH_own@2, B1@4 - COPY_own, D7@2 - SH_own@2, CHPT_MAJ - COPY_own, D7@2 - COPY_own.
- Strata: n_sib = 1 / >= 2; DGA (sc009) / non-DGA; the giant cluster / the rest; condition-change type (`cls`); without pub_b0b3296ec5 (influence). Sensitivities: WITH_C5 (C5-BTBP in), WITH_QUANT_PUB (pub_a49146c8fd in), QUANT_TARGETS_OUT (every quantised target out), NOBIT (P1's CLEAN rule without the bit rule, `NB_` arms; the duplicate sensitivity).
- Duplicate sensitivities of the main E-DEC frames (REGISTRY 1.5-S11 … S13; s0-fidelity-8): the primary statistics recalibrated with the V2-1/2/3 cells dropped (noV2), with the second cell of every flagged bit link dropped (noBITDUP), on the V2-3 condition-aware stratum (11 C3 cells; S1 open; NaN when too small to calibrate) and without it; columns of `primary_<design>.csv`.
- Unit: publication x chemotype (P1's E-CT unit). Per unit the matched-coverage false-sign counts per arm; tests on the first four contrasts: W/L/T, SF_POOL (exact at G <= 13, else 4,999 random flips), SIGN, top-3 share (DIAGNOSTIC); leave-one-publication-out of fs@c_ref differences (`r5.common.lopo`, t(P - 1)).

## Report rules

- Descriptive: no kill; thresholds below seen values would be consistency checks, and none is registered. Every number carries its label (EXPLORATORY (seen); SENSITIVITY for BP, C5-BTBP out, alpha 0.05, no hull rule, the E-REC sensitivity strata; SECONDARY for Mondrian; ORACLE [O] for oracle orders and AMP_ORACLE; DIAGNOSTIC for HIST_D7 and the unit tests; TECHNICAL SMOKE for the reproduction run).
- Seen values recorded in `summary_<design>.json` (never targets): B1 coverage at alpha 0.10 0.606 / 0.519 / 0.507 / 0.149 (as stated / CRC / plug-in / valid fixed-sequence) and B1 - B0 +0.047 to +0.058 [IR M14]; LINE shares 61 % Spearman, 67 % top-3, 66 % within publication [IR M15]; recipe-3 at matched coverage COPY 0.922, D7@4 0.919, B1@4 0.933, SH_own@2 0.949, opposite-direction call rates COPY 1.1 %, SH_own@2 0.9 %, D7@2 2.3 % [I2 recipe-3]. Those used other implementations (HULL reads, other calibration); differences are expected and are not mismatches.
- Every item statistic: C5-BTBP in / out (every table carries `c5`), BPG and BP, per-seed values, the `cluster_cpg` unit table (`clusters_<design>.csv`) with W/L/T, SF_POOL (exact over 2^13) and SIGN on the per-cluster coverage contributions of B1 - B0 and B1 - HBK@ML and their top-3 share (`unit_tests_<design>.csv`), and leave-one-publication-out (`r5.common.lopo`) plus C5-BTBP out and c950dc2982e96b83 out for the primary statistics (`primary_<design>.csv`): AURC of B1 and B0 and their difference; LTT alpha 0.10 hull-rule coverage of B1, B0 and B1 - B0; their per-call false-sign rates; NB at p_t 0.3 (separable) of B1, B0 and the difference. Every LOPO and C5 frame is recalibrated.
- The section-9.3 per-item table: `PER_ITEM_TABLE_<design>.md` (`r5.common.per_item_table`).

## Reproduction targets (brief 6.0; a mismatch stops the item)

- Brief 6.0 lists no E-DEC value target. The applicable technical targets: B1 recomputed from the fold betas (`r5.ect` B1_r4) equals the REF parquet within 8.9e-16 on every row (9,010 per design, 18,020 over both); D7_r4 from the identity equals REF HIST_D7 within 1e-12; y identical between the E-CT and E-A rows; the support contract. `--repro` writes `repro_<design>.json` (TECHNICAL SMOKE) and exits 1 on a mismatch.

## Leakage (brief 8)

- Fit poisoning (item 2) of every calibrated quantity: for each cluster c the labels of c's rows are replaced by noise and every quantity fitted for c is refitted and compared bit for bit: LTT / CRC / PLUGIN thresholds of B1, B0, HBK@ML at both alphas under hull + global, hull + Mondrian and all rows; every Platt (a, b) (5 arms x 2 decisions); the interval's chosen nominal, inner coverage and half-widths; the bound quantiles. Positive control (must be caught for every cluster): the leaky calibration that includes c's rows. Written to `results/leakage/EDEC_<design>.json`.
- Support (item 5): no scored pair with both ends read; strict adjacency; the hull flag from read sets only; the bound line reads V1 values only at read ends.
- CHPT_MAJ is leave-own-publication-out by construction (as recipe-3 defines it; not fold-nested). The E-A predictions are round-4 OOF rows (round-4 leakage chain); the E-REC arms carry r5.ect's intervention check (item 1.1).
- Known limitation, stated with the results: a calibration row's prediction comes from its own outer fold's model, whose training may include cells of the held-out cluster (B1's scalar beta, HBK@ML's kernel); the calibration labels are only other clusters' labels.  **Scope of the brief-8 item-2 check: calibration labels only** -- the prediction models (B1 fold beta, HBK@ML kernel) are not refitted, and their training may include the held-out cluster's cells; B0 is training-free (no such channel).  The scope is written into `results/leakage/EDEC_<design>.json` (`scope`) and the per-item leakage line (s0-items-7).  A pass needs every dev seed and the 13 clusters (`--poison-seeds` is smoke-only; s0-mechanics-5); the check runs before any scored write (s0-mechanics-17).  The E-CT rows of item 1.1 are reused only when item 1.1's intervention and reproduction passed under the current seal; otherwise they are rebuilt and intervention-checked here.

## RNG seeds (`r5.paths.seed_for(key)` = 20261023 + crc32(key))

- Fit-poisoning noise: `EDEC|POISON|<design>`. Smoke surrogates: `EDEC|SMOKE_SURROGATE|<design>`, `EDEC|SMOKE_SURROGATE_EREC|<design>`. Sign-flip tests: exact at G <= 13; random flips (B = 4,999) when G > 13 with keys `EDEC|UNIT|<design>|<c5>|B1-<arm>` and `EDEC|EREC|<design>|<arm>-vs-<ref>` (the E-REC units; the seed is written with each test). Everything else is deterministic (exact expectations for ties and RANDOM; Newton fits; exact enumeration).

## Outputs (results/EDEC/, after the seal)

`rows_<design>.parquet` and `rows_<design>_C5out.parquet` (every call set, Platt probability and interval per row), `erec_rows_<design>.parquet`, `thresholds_<design>.parquet`, one CSV per table (`selective`, `pairtype`, `contrasts`, `risk_coverage`, `rc_curve`, `netbenefit`, `citl`, `platt`, `clusters`, `unit_tests`, `interval`, `interval_clusters`, `bounds`, `bounds_rows`, `savings`, `rank`, `rank_line_share`, `primary`, `erec_population`, `erec_arms`, `erec_contrasts`, `erec_units`, `erec_unit_tests`, `erec_lopo`; each `_<design>.csv`, every row labelled), `summary_<design>.json`, `PER_ITEM_TABLE_<design>.md`, `repro_<design>.json`; `results/leakage/EDEC_<design>.json`. `SMOKE_COUNTS_<design>.json` holds counts only (technical smoke). Compute (measured in the smoke, one seed of poisoning): about 4-5 minutes per design; the full run poisons all 5 seeds.

## Command lines (run from D:\ml_separator_gh)

```
set PYTHONIOENCODING=utf-8
rem technical smoke on real data (metrics suppressed; allowed before the seal)
.venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_edec.py --smoke --design BPG
.venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_edec.py --smoke --design BP
rem tests (synthetic / construction only)
.venv/Scripts/python.exe -B -W ignore -m pytest generations/gen21/round5_20261024/tests/test_edec.py -q -p no:cacheprovider --basetemp generations/gen21/round5_20261024/cache/_pytest
rem after the Stage-0 seal and the item code seal: reproduction run (TECHNICAL SMOKE), then the arms
.venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_edec.py --repro --design both
.venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_edec.py --design both
```
