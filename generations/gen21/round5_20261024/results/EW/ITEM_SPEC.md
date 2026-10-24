# Item 1.7 specification: E-W wider-pair table (M13)

Frozen at the Stage-0 seal, before the item's first post-seal run (brief 1.2: Stage-0 seal -> item code seal -> reproduction run labelled TECHNICAL SMOKE -> arms; the pre-seal --smoke runs are PRE-SEAL ENGINEERING CHECKS, PROTOCOL 4 and 10).  Every in-corpus number is EXPLORATORY (seen); nothing here is a test.  E-W is its own estimand: never pooled with the strict-adjacent pairs and never placed next to E-A's 0.452 (brief 3.3).  Code: `r5/ew.py`, `scripts/s1_ew.py`, `tests/test_ew.py`.  Sources: brief 6.7, 6.0, 6.10, 8, 9; IR M13 (`ideas/IDEAS_RANKED.md` LF SHA-256 41f7bc735c3a6d36f4c0a5c1c6bd6ae341f91f4aaed754fc7d99bffe115595e2; `ideas/critiques.json` LF SHA-256 585468e5ea8267f7e8df650a2115843640b37d6d1e29f6050f50e2c306d129d3).  No design-folder script is copied or run.

## Rows

- Every C3 cell instance (k_max >= 3) of the 50 round-4 dev folds (5 seeds x 5 folds x BPG / BP) at E-A's read set: four reads, the union of the cached `cell['seqs']['DOPT_Q'][:3]` (g21.cache.load; brief 3.3).
- WIDE rows: the frozen cut-pair set (33 pair types), A lighter, dZ = Z_B - Z_A (Pm absent: Nd/Sm has dZ 2): dZ 2: La/Pr, Ce/Nd, Nd/Sm, Sm/Gd, Eu/Tb, Gd/Dy, Tb/Ho, Dy/Er, Ho/Tm, Er/Yb, Tm/Lu; dZ 3: La/Nd, Pr/Sm, Nd/Eu, Sm/Tb, Eu/Dy, Gd/Ho, Tb/Er, Dy/Tm, Ho/Yb, Er/Lu; dZ 4: Ce/Sm, Pr/Eu, Nd/Gd, Sm/Dy, Eu/Ho, Gd/Er, Tb/Tm, Dy/Yb, Ho/Lu; dZ 5: Nd/Tb; dZ 6: Nd/Dy; dZ 7: Pr/Dy.  Both ends measured, at least one end unread (a pair with both ends read is known and never scored; counted in `checks`).
- Feed-defined pair set: Nd/Sm, Nd/Tb, Nd/Dy, Pr/Dy, Eu/Tb (NdFeB: Nd/Tb, Nd/Dy, Pr/Dy).  Note: brief 6.7 writes 'dZ 5-6' for the NdFeB pairs; by atomic number Nd/Tb = 5, Nd/Dy = 6 and Pr/Dy = 7.
- STRICT rows: the cell's strict-adjacent queries (E-A's 1,802 rows per seed), used only for the reproduction target and as the denominators of the MSE / var(y) multipliers; no E-A level is reported by this item.
- Regions: light La-Nd, middle Sm-Dy, heavy Ho-Lu; a pair spanning two is 'light|middle' etc.  Heavy cuts = both ends in Ho-Lu: explicit NON-CLAIMS, reported by whether Tm or Lu is read.

## Arms (identical rows; level curves from the read metals only)

- B1: `r4.refs.predict_levels` with the fold's stored round-4 beta (`round4_20261022/results/REF/fitted_<design>.json`; no refit).  B0: the same path with beta = 0 (= PWL_R).  LINE: the slope-only line through the reads (`gen20k.adapters.naive_line` on the gen20 support; round 4's LINE column).  HIST_D7: the fixed -0.072 (DIAGNOSTIC; not fold-independent; IR M13's seen numbers used it).
- Comparators of B1: LINE and B0.  No arm is fitted or selected in this item, so fit and cache poisoning do not apply; no new cache is written.

## Per-cut table (one row per cut pair and variant) and pooled frames

- R2 per arm (pooled inside each seed, then the seed mean; never clipped), per-seed R2 of B1 and LINE; skill against LINE and against B0 = seed mean of 1 - SSE_B1 / SSE_ref (and dR2 = (SSE_ref - SSE_B1) / SST); MSE and var(y) multipliers = seed means of the per-seed ratio to the STRICT rows of the same variant; MAE and MAE / median |y| (B1, LINE, B0); sign agreement (zero prediction = disagreement unless y = 0 exactly) overall and in the |y| bins [0,0.05), [0.05,0.10), [0.10,0.20), [0.20,0.50), [0.50,inf); zero-prediction shares; cluster coverage (clusters present, clusters with >= 10 rows per seed, largest cluster and share, Kish over clusters); cluster_cpg unit W/L/T and top-3 unit share of dSSE vs LINE and vs B0; leave-one-publication-out (jackknife t(P - 1) 95 % interval, min / max with the publication) of R2_B1, skill vs LINE and skill vs B0.
- Pooled frames (each its own row): dZ2, dZ2 without Nd/Sm, dZ3, dZ4, dZ3-4, Nd/Sm, feed NdFeB, feed set, light (dZ2-4), light|middle (dZ2-4), middle (dZ2-4), middle|heavy (dZ2-4), heavy (dZ2-4; NON-CLAIM); with the macro average = equal-weight mean of the per-pair R2 over the frame's pair types.
- 'Strict + Nd/Sm' pooled appears only as the labelled SUBSTITUTION TRAP row (its own file).

## The pre-declared open contrast (report rule)

- Nd/Sm B1 - LINE in R2 units: inside each cluster_cpg cluster, seed mean of (SSE_LINE - SSE_B1) / SST_c; clusters with >= 10 Nd/Sm rows per seed (seed mean), the giant cluster g:pub_74428aeab2 (64.9 % of E-A rows; asserted) removed; equal-weighted mean over those clusters; leave-one-cluster-out jackknife-t 95 % interval (t(G - 1)), stated once.
- Rule: 'Interpolation beats slope at the light|middle cut' is RETIRED if the contrast is <= +0.05 or its lower bound is <= 0, or fewer than 2 clusters qualify (expected: retired; narrowed to 'a line through 4 reads suffices at Nd/Sm outside the giant cluster').  BPG decides; BP is a sensitivity.  Seen before the code: B1 - LINE +0.084 [-0.075, +0.243] pooled, +0.011 to +0.013 outside the giant cluster [IR M13].
- Labelled sensitivities beside it: giant cluster kept; every cluster with >= 2 rows per seed; C5-BTBP out; the pooled Nd/Sm B1 - LINE with its leave-one-publication-out interval, all clusters and outside the giant.

## Every item reports

- Variants all, noC5, noC5cell (C5-BTBP = pub_6a0ef8528c; single-cell influence of c950dc2982e96b83); duplicate sensitivities noV2 (round-3 V2-1/2/3 cells) and noBITDUP (second cell of every Stage-0 flagged bit-identity link); BPG (primary) and BP (sensitivity); per-seed values; the cluster_cpg unit table with SF_POOL / SIGN / SF_LOG / IM_REL (DIAGNOSTIC; r5.common); leave-one-publication-out; the top-3 unit share; the section-9.3 per-item table (`ITEM_TABLE.md`).
- Seen-value table: IR M13's values beside the recomputed ones (CONSISTENCY CHECK, not a stop condition).
- Labels: EXPLORATORY (seen) on every in-corpus number; SENSITIVITY (BP, C5-BTBP out, single cell out, duplicates); DIAGNOSTIC (HIST_D7 columns, unit tests); NON-CLAIM (heavy cuts); SUBSTITUTION TRAP; CONSISTENCY CHECK (seen table); TECHNICAL SMOKE (reproduction run).  No in-corpus p-value is called significant.

## Reproduction target (brief 6.0; a mismatch stops the item)

- STRICT rows: B1 from the fold betas equals `round4_20261022/predictions/REF_<design>.parquet` within 8.9e-16 on every row (9,010 per design, 18,020 in all); y exact; read sets identical; B0, HIST_D7 and LINE differences reported.

## Leakage (brief 8)

- Isolation (`g21.evaluate.assert_test_isolation`) on every fold.  Support: exact B0 / B1 / HIST_D7 levels at every read metal; no scored pair with both ends read; A lighter, dZ = 1 for STRICT and >= 2 for WIDE; finite predictions; unique row keys.  Prediction intervention on every cell instance (every unread label replaced by N(0, 5) noise; all four arms' level curves and the read set bit-identical) with a block-Y peek positive control per fold that must be caught; the control goes through the real level function and comparator (a predictor reading the cell's whole DOPT_Q sequence must change when the unread metals change; s0-mechanics-6).  A sealed run needs full coverage: every cell instance intervened on and one caught peek per fold (s0-mechanics-5); `--no-intervention` / `--no-lopo` are smoke-only.  The seal tag and code digests are written next to every output; `--from-rows` refuses rows or checks of another seal or code (s0-mechanics-18).
- The pre-seal `--smoke` replaces y and every prediction column by seeded noise before any table runs, so no real-label metric is computed (s0-mechanics-22).

## RNG seeds

- Intervention scrambles: `default_rng(r5.paths.seed_for('EW|intervention|<design>'))` (20261023 + crc32).  Unit sign-flip tests: exact over 2^G at G <= 13 (seed_for('<key>|SF_POOL') / '|SF_LOG' only if G > 13, key = 'EW|<design>|<variant>|<frame>|B1-<ref>').  No other random procedure.

## Command lines (run from D:\ml_separator_gh)

```
set PYTHONIOENCODING=utf-8
rem pre-seal engineering check on real data (no metric printed or written; allowed before the seal)
.venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_ew.py --smoke --design both
rem after the seal: reproduction run (TECHNICAL SMOKE), then the item
.venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_ew.py --repro --design both
.venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_ew.py --design both
.venv/Scripts/python.exe -B -W ignore -m pytest generations/gen21/round5_20261024/tests/test_ew.py -q -p no:cacheprovider --basetemp generations/gen21/round5_20261024/cache/_pytest
```
