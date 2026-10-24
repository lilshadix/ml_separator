# Item 1.4 specification: M09 level window as a tail-risk reporting axis (window axis)

Frozen at the Stage-0 seal, before the item's first post-seal run (brief 1.2: Stage-0 seal -> item code seal -> reproduction run labelled TECHNICAL SMOKE -> arms; the pre-seal --smoke runs are PRE-SEAL ENGINEERING CHECKS, PROTOCOL 4 and 10).  Descriptive only: every in-corpus number is EXPLORATORY (seen), the grid was fixed after seeing V1, no in-corpus p-value is called significant, and no window value is placed next to E-A's 0.452 (brief 3.3).  Excluded from the confirmation package (the window as a row flag has power <= 0.15; brief 4.3 item 4, I2 confirm-1).  Code: `r5/window.py`, `scripts/s1_window.py`, `tests/test_window.py`.  Sources: brief 6.4, 6.0, 6.10, 8, 9; IR M09, M10, M39 (and `ideas/critiques.json` M09 / M39 test designs); EB 2.11 and Appendix 2; E2 1.6.

## Rows and levels

- Rows: round-4 `predictions/REF_<design>.parquet` through `r5.common.load_rows` (C3, k = 3 = four DOPT_Q reads; 1,802 queries per seed; dev seeds 104729, 130363, 155921, 196613, 262147).  BPG decides; BP is a labelled SENSITIVITY on the same test folds.  Pooled within seed, then the seed mean.
- Residual res = y - B1 (B1 from the parquet).  B0 and HIST_D7 inside R2 are shown beside B1 on the curve table (HIST_D7 DIAGNOSTIC).
- Levels: `r4.refs.predict_levels` with u = beta at Gd, beta = the fold's stored round-4 amplitude (`round4_20261022/results/REF/fitted_<design>.json`, `ref:B1`), on the row's read set = union of the cached `seqs['DOPT_Q'][:3]` (asserted equal to the parquet's `read_set`); a read metal's level is its measured value.  s = max(|lev_A|, |lev_B|).  lev_A - lev_B must equal the parquet's B1 within 8.9e-16 on every row.

## Frozen rule list ("fixed after seeing V1")

| rule | inside (kept) iff |
|---|---|
| H1, H1.5, H2, H2.5, H3, Hinf | s <= h, h in {1, 1.5, 2, 2.5, 3, inf} (both levels in [-h, h]) |
| LO-2, LO-2.5, LO-3, LO-4 | min(lev_A, lev_B) >= L (LO-4 = "drop predicted min < -4") |
| UP2, UP2.5, UP3 | max(lev_A, lev_B) <= U |
| OP75, OP90 | s <= the label-free quantile of s (numpy `method='inverted_cdf'`) at coverage 0.75 / 0.90 over the outer fold's TRAINING rows |
| READMIN-2 | min of the four read values >= -2 (M10 reads-only flag) |
| READSIN2 | all four read values in [-2, 2] (M10 reads-only flag) |
| WHOLECELL2 | whole-cell rule: every row of a cell instance kept iff H2 keeps >= 50 % of that instance's rows |
| NOC5 | publication != pub_6a0ef8528c ("drop C5-BTBP"; metadata rule) |
| NEST_HW | per fold, the half-width rule (H1 .. Hinf) chosen on the training rows |
| NEST_FAM | per fold, the rule chosen on the training rows from the family = grid + OP75 + OP90 + READMIN-2 + READSIN2 + WHOLECELL2 |

- Training rows of a fold (operating points and nested choices; never a test row): every strict-adjacent query of the C3 training cells of the cached fold record (`r4.refs.training_cells`: inner-block records plus orphans), masked with the cell's nested DOPT_Q[:3] read set from that record, levels with the fold's beta, labels = training labels.
- Nested selection criterion: maximal inside R2 on the training rows among rules with training coverage >= 0.70; ties toward larger training coverage, then the registry order above; Hinf when none qualifies.
- Primary operating point: OP75.  OP90, H2 (the post hoc [-2, 2]) and NEST_FAM are reported beside it.

## Estimands and statistics

- Curve (every rule x variant): coverage, inside / outside / all R2, MSE inside / outside / all, inside relative SSE = 1 - MSE_in / MSE_all (per seed, then the seed mean), var(y) inside / all, per-seed values; the CELL closed-form expectation (exact E[SSE] = sum pi e^2 with pi the cell's kept share; plug-in R2) and the excess over it.
- (a) Tail flag: P_in(c), P_out(c) = seed means of the per-seed shares of rows with |res| > c, c in {0.5, 0.3}; ratio = P_out / P_in.  95 % two-sided percentile publication bootstrap, B = 2000 (`gen20k.inference._draw_counts` over the variant's publications; percentiles by `inverted_cdf`, inf kept), all rows and without C5-BTBP.
- (b) Bulk precision: median |res| inside / outside / all; q80 split-conformal half-width (the ceil((n + 1) 0.8)-th smallest |res|, inf when it exceeds n) per row, calibrated leave-one-cluster-out (cluster_cpg) within seed: pair-type-only (PT) and window-aware (pair type x inside / outside; PTW); reported on the inside rows: mean and median half-widths, narrowing = 1 - mean PTW / mean PT, empirical coverage, minimum per-cluster coverage, counts of infinite half-widths.
- Controls at matched counts (random abstention keeping the rule's number of rows per stratum): CELL = (seed, cell) [primary], PUBPT = (seed, publication, pair type), UNSTRAT = (seed); 2000 draws each with common random numbers across the rules of one (design, variant, control); mean, sd, 2.5 / 97.5 % quantiles, max and the share of draws at or above the rule, for inside R2 and inside relative SSE.
- Cheapest alternatives at matched coverage: READMIN-2, READSIN2, WHOLECELL2, LO-4, NOC5, each against the s-ranked window at its own per-seed coverage (the n rows with the smallest s; ties by row order) and its own CELL control.
- Winsorised curves: |res| capped at 0.3 and 0.5 (SST unchanged) for the grid, OP and NEST rules.
- M39 optimism ledger with family nesting: H2 (post hoc), NEST_HW, NEST_FAM, OP75, OP90; optimism = R2_in(H2) - R2_in(rule); per-rule choice counts over the 25 fold-fits, the share choosing H2 and the share choosing a level window; seen values printed beside.
- Units (cluster_cpg, 13): rows, coverage, MSE inside / outside and their ratio, tail counts, dSSE = E[SSE_CELL] - SSE_window, L = log(E / SSE_window), rel = 1 - SSE_window / E, contribution c_u (sum = excess relative SSE over the CELL expectation); SF_POOL (exact over 2^13), SIGN, SF_LOG, IM_REL via `r5.common` (DIAGNOSTIC); top-3 share of c_u.  Rules OP75, H2, NEST_FAM.
- Leave-one-publication-out (`r5.common.lopo`, t(P - 1) 95 %) of inside R2, the excess relative SSE and plug-in excess R2 over the CELL expectation, and the |res| > 0.5 ratio (OP75, H2, NEST_FAM; all and without C5-BTBP).  Leave-top-k-cells-out, k in {5, 10} (cells ranked by B1 SSE over all rows; SENSITIVITY, outcome-selected).
- C5-BTBP in / out and cell c950dc2982e96b83 out; duplicate sensitivities noV2 (V2-1/2/3 cells, `r5.population.v2drop_set`) and noBITDUP (cell_b of every flagged link in `results/stage0/dupscan_links.csv`); per pair type coverage and MSE inside / outside.  The rules' memberships and thresholds are fixed on all training cells; a variant only restricts the scored rows.

## Report rules (in-corpus CONSISTENCY CHECKs; never tests; BPG decides, BP reported)

- Row flag: "row flag" only if NEST_FAM's inside relative SSE (all rows) exceeds the 97.5 % quantile of its CELL control; otherwise "cell or condition flag".  The noC5 evaluation is reported beside it.
- Tail flag: passes if, at OP75, the |res| > 0.5 ratio is >= 3 with a bootstrap lower bound > 1, both on all rows and without C5-BTBP.  OP90, H2 and NEST_FAM are reported as secondary evaluations.
- Bulk precision: passes if, at OP75 on all rows, the narrowing of the mean q80 half-width (PTW vs PT on the inside rows) is >= 15 %; expected to fail.
- Seen before round 5: 0.540 at 74.7 % (post hoc; one of 4 windows) [EB 2.11; IR M09]; cell-matched 0.510-0.511 [0.500, 0.522] at the H2 coverage [IR M09]; 0.516 at 78.7 % (family nesting) [IR M09, M39]; P(|res| > 0.5) 0.009 inside vs 0.043 outside [IR M09]; window-aware Mondrian 11-12 % narrower than pair-type-only [IR M09]; 0.486 at 91.8 % [IR M09].

## Reproduction targets (brief 6.0; TECHNICAL SMOKE; a mismatch stops the item)

- B1 from the fold betas equals `REF_<design>.parquet` within 8.9e-16 on every row (9,010 per design, 18,020 over both).
- EB Appendix 2 (window on HIST_D7-form levels in [-2, 2]): BPG coverage per seed within [0.743, 0.754], B1 inside R2 0.540 (per seed 0.532, 0.521, 0.546, 0.547, 0.553), BP inside R2 0.538; tolerance 0.0005 (quoted to 3 decimals).  The same window on B1-form levels (the brief's rule) is reported beside it, not gating.

## Leakage (brief 8)

- `g21.evaluate.assert_test_isolation` on every fold; training rows assert their cells lie in the outer training index and not in the test index.
- Support: read sets equal the cache and the parquet; read-metal levels equal the reads; no scored pair with both ends read; read flags match; strict adjacency; every query belongs to the cell's query set.
- Prediction intervention on every test cell instance: unread metals scrambled (`g21.evaluate.scramble_hidden`), levels bit-identical; positive control: a block-Y peek (every measured metal read) must be caught.  The pre-seal `--smoke` replaces y and every prediction column by seeded noise before any table runs (no real-label metric is computed; s0-mechanics-22); `--no-lopo` is smoke-only; the seal tag is written next to every output.
- Fit poisoning of the operating points and nested choices on every fold (`r4.leakage.poisoned_bench(bench, train_index, seed = split_seed * 10 + fold)`): thresholds, choices and training rows bit-identical; positive control: the same computation with the fold's test cells added must change.  Written to `results/leakage/poison_WINDOW_ops_<design>.json`.  The stored fold betas were poisoning-checked in round 4.

## RNG seeds (`r5.paths.seed_for(key)` = 20261023 + crc32(key))

- Controls: key `WINDOW|control|<design>|<variant>|<control>` -> BPG/all/CELL: 342008595, BPG/all/PUBPT: 1946440004, BPG/all/UNSTRAT: 3497266310, BPG/noC5/CELL: 3028423725, BPG/noC5/PUBPT: 809269973, BPG/noC5/UNSTRAT: 649603187, BP/all/CELL: 2056788584, BP/all/PUBPT: 3366637758, BP/all/UNSTRAT: 1038203886, BP/noC5/CELL: 137073619, BP/noC5/PUBPT: 87808707, BP/noC5/UNSTRAT: 4273041181.
- Publication bootstrap: key `WINDOW|pubboot|<design>|<variant>` -> BPG/all: 2462194216, BPG/noC5: 1504850776, BP/all: 1656916073, BP/noC5: 3096265692.
- Intervention scrambles: key `WINDOW|intervention|<design>` -> BPG: 2386177775, BP: 60227101.
- Fit poisoning: split_seed * 10 + fold (round 4's).  Unit sign-flips: exact at G <= 13 (key `WINDOW|units|<design>|<variant>|<rule>|SF_POOL` otherwise).

## Outputs (`results/WINDOW/`)

`repro_<design>.json`; `rows_<design>.parquet` (row-level levels, s, reads, thresholds, choices and every rule's membership); `ops_<design>.csv` (per-fold thresholds, choices, training coverages and training R2 -- training-side); `checks_<design>.json`; one CSV per table `<table>_<design>.csv` (curve, winsor, controls, tail, bulk, alternatives, optimism, units, unit_tests, pairtypes, lopo, topk, c5, duplicates, report_rules), every row labelled; `summary_<design>.json`; `ITEM_TABLE.md` (the section-9.3 per-item table) when both designs ran.

## Command lines (run from D:\ml_separator_gh)

```
set PYTHONIOENCODING=utf-8
rem pre-seal engineering check on real data (no metric printed or written; allowed before the seal)
.venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_window.py --smoke --design BPG
rem after the seal: reproduction run (TECHNICAL SMOKE), then the arms
.venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_window.py --repro --design both
.venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_window.py --design both
.venv/Scripts/python.exe -B -W ignore -m pytest generations/gen21/round5_20261024/tests/test_window.py -q -p no:cacheprovider --basetemp generations/gen21/round5_20261024/cache/_pytest
```
