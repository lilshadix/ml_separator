# Item 1.6 specification: E-D read-budget ladder (M18), policy envelope (M17), recipe-2 row

Frozen at the Stage-0 seal, before the item's first post-seal run (brief 1.2: Stage-0 seal -> item code seal -> reproduction run labelled TECHNICAL SMOKE -> arms; the pre-seal --smoke runs are PRE-SEAL ENGINEERING CHECKS, PROTOCOL 4 and 10).  Every in-corpus number is EXPLORATORY (seen); nothing here is a test.  Code: `r5/ed.py`, `scripts/s1_ed.py`, `scripts/s1_ed_poison.py`, `tests/test_ed.py`.  Sources: brief 6.6, 6.0, 6.10, 8, 9; P3 and PC P3; IR M17 / M18; I2 recipe-2; E5 5.1.

## Rows and frames (identical queries at every rung and under every policy)

- Read counts in reads r; code budget k = r - 1; DOPT_Q read set = union of `cell['seqs']['DOPT_Q'][:k]` (brief 3.3).
- C3 population row "C3 min(k, k_max), k = reads - 1" (E-A's 1,802 queries per seed) at (2, 3, 4, 5) reads; a cell that cannot take r reads keeps min(r, k_max + 1) reads (and that budget's beta).  The 4-read rung is E-A itself; every other rung is a different read budget, never an E-A value.  In any E-D table that shows C3 rungs, the C3 4-read cell reads "(E-A rows)" and no 4-read C3 level is printed (the writer blanks it and sets `level_note`); C3 levels at 5 or more reads appear only in such tables, labelled "different read budget", and as paired contrasts on E-A's identical queries.  Fixed-cohort C4-C6 levels may be printed at every rung, including 4 reads (C4 0.4290 is not E-A).
- Fixed cohorts: C4 (k_max >= 4) at (2, 3, 4, 5) reads; C5 (k_max >= 5) at (2, 3, 4, 5, 6) reads; C6 (k_max >= 6) at (2, 3, 4, 5, 6, 7) reads.
- Frozen query rule for C4-C6 (E5 5.1): every strict-adjacent query of the cohort's cells at every budget; a pair with both ends read at the cohort's largest budget would be excluded at every budget.  Legal reads never close a strict-adjacent pair, so the excluded set is empty; the code asserts it (checks `cohort_largest_budget_closed_queries` = 0).
- Legality: `gen20k.design.legal_metals` for every added read (measured, unread, not strict-adjacent to a read metal, maximum independent set still reachable).  LOOSE (no look-ahead) is a sensitivity.

## Arms

- B1: beta refit per outer fold and per budget with `r4.refs.fit_reference(bench, rec, Y, 'B1', k = reads - 1)` for k in (1, 2, 3, 4, 5, 6) (k = 3 uses the stored round-4 fit `results/REF/fitted_<design>.json`; its refit is an identity check); beta, lambda, rows and coverage logged per fold (`fits_<design>.csv`); one BLAS thread.
- B0: the beta = 0 row.  HIST_D7: the fixed -0.072 (DIAGNOSTIC; not fold-independent).  B1_k3beta: the k = 3 beta applied at every budget (SENSITIVITY).

## M17 policy envelope (5 reads; P3's rules and tie-breaks)

- Family (strict legality): DOPTQ, MAXMIN_R, MAXMIN_Z, BISECT_R, BISECT_Z, BISECT_RMID, HULL_BISECT, RANDOM.  ORACLE_B1 [O] (the legal 5th metal minimising the cell's B1 SSE; bound only).  LOOSE sensitivity: MAXMIN_R_LOOSE, RANDOM_LOOSE.
- RANDOM: exact expectation (uniform over the legal set, independent across cells) plus a Monte Carlo of 2000 joint draws, RNG `default_rng(20261023 + crc32('ED|RANDOM_MC|<design>|<variant>'))` (`r5.paths.seed_for`).
- Reported: levels for C3 min(4, k_max) at 5 reads and C4; per-block dSSE (light La/Ce-Pr/Nd, mid Sm/Eu-Dy/Ho, heavy Ho/Er-Yb/Lu) in pooled-R2 units vs DOPTQ and vs 4 reads with cluster-bootstrap 95 % intervals; per-pair MAE and skill against FLAT (prediction 0); splits by k_max and by cluster; with and without C5-BTBP; envelope width over the family and the per-seed span; contrast-specific implied MDE = 2.8 x cluster-bootstrap sd.
- Pre-declared Pareto question (vs DOPTQ, C3 min(4, k_max)): does any member keep every block above -0.002 and improve one block with a cluster lower bound > 0?  Seen before the code: envelope 0.0091 (BPG) / 0.0096 (BP), no member Pareto-dominates [P3 5, 9].

## Density and saturation (M18)

- Series-length strata (14 / 13 / 8-12 / <= 7 metals); read-density index: reads per measured metal, longest hidden run, share of rows outside the read hull, share of unread metals outside the hull; MSE and relative SSE vs B1 at 4 reads on the same rows next to R2; P(R2 >= 0.50) as a cluster-bootstrap fraction (descriptive, never a posterior) with LOCO and leave-one-publication-out ranges, with and without C5-BTBP.
- Saturation strata: k_max j at j + 1 reads for j = 3, 4, 5, 6 (the 6- and 7-read strata P3 did not compute).  Density invariance at saturation is a CONSISTENCY CHECK, not a kill: the rule fires if a stratum with >= 3 clusters is below 0.48 or the spread is >= 0.05; the k_max-4 value (0.444) was known before the rule was written [P3 7c; PC P3].

## recipe-2 row (2-5 reads, C3 min rule)

- Policies: LADDER, LADDER_GUARD, LOCO_STATIC, HULL, DOPTQ, RANDOM_SET.  LADDER = the nested static order La -> Ho -> Nd -> Lu -> Gd (a static metal is skipped when it is not legal; the farthest-point HULL pick replaces it).  LADDER_GUARD = HULL for cation exchangers (CX_ACIDIC_OP, CX_CARBOXYLIC, CX_CHELATING; `r5.supply.classify_extractant` on chem_family / extractant_name, blind to values), LADDER otherwise.  HULL = lightest legal metal, then the legal metal farthest in -CN8 radius, ties -> lower Z.  LOCO_STATIC = per requested size r and held-out cluster_cpg cluster, the static set (no strict-adjacent pair) minimising the summed B0 SSE over the other clusters' C3 cells (a cell on which the set is not legal counts with its HULL set), ties -> lexicographically smallest Z; HULL fallback on a held-out cell where the set is not legal.  RANDOM_SET = exact mean over every legal read set of size min(r, k_max + 1).
- Primary contrast Ladder - HULL (B1; B0 beside it); also Ladder - DOPTQ, HULL - RANDOM_SET (closure line: 'spread first' is definitional), LOCO_STATIC - HULL, LADDER_GUARD - LADDER, DOPTQ - HULL; by series length.

## Report rules and statistics

- Pooled strict-adjacent R2 within each seed, then the seed mean; MSE, MAE; relative SSE.  Every item reports C5-BTBP in / out (and the single cell c950dc2982e96b83 out for every row: ladder, M17 envelope and recipe-2), BPG and BP, per-seed values, the cluster_cpg unit table (SF_POOL exact over 2^13, SIGN, SF_LOG, IM_REL: DIAGNOSTIC), leave-one-publication-out (r5.common.lopo) and the top-3 unit share.  Duplicate sensitivities: V2-1/2/3 cells dropped; the second cell of every Stage-0 flagged bit-identity link dropped.
- Cluster bootstrap: one shared draw matrix `gen20k.inference._draw_counts(13, 2000, 8675309)` over the 13 C3 cluster_cpg clusters (P3's); percentile intervals are descriptive (anti-conservative at Kish 2.24).
- Labels: EXPLORATORY (seen) on every in-corpus number; ORACLE [O]; DIAGNOSTIC (HIST_D7, unit tests); SENSITIVITY (BP, C5-BTBP out, k = 3 beta, LOOSE, duplicates); CONSISTENCY CHECK (saturation rule); TECHNICAL SMOKE (reproduction run).  The 5- to 7-read numbers carry 'different read budget'.  No in-corpus p-value is called significant.

## Reproduction targets (brief 6.0; a mismatch stops the item)

- B1 at 4 reads from the fold betas equals `round4_20261022/predictions/REF_<design>.parquet` within 8.9e-16 on every row (18,020 over both designs); y exact; read sets identical.
- P3 5-read DOPTQ-nested B1: C3 min(4, k_max) 0.4968 (BPG) / 0.4988 (BP); C4 0.4798 / 0.4821 (tolerance 5e-05); 5-read B0 equals round 3's R3-B0 PWL_R at k = 4.

## Leakage (brief 8)

- Isolation (`g21.evaluate.assert_test_isolation`) on every fold; support checks: no scored pair with both ends read, nested and legal ladders, identical row keys across rungs and policies, the DOPT_Q 5th read legal and equal to the cache; prediction intervention on every read set (unread metals scrambled; predictions bit-identical) with a block-Y peek positive control that must be caught; the enumerated static sets likewise.
- Fit poisoning of the k-specific beta refits (k in (1, 2, 3, 4, 5, 6)) on all 50 fold-fits: `scripts/s1_ed_poison.py` (r4.leakage.poisoned_bench; writes only `results/leakage/poison_ED_beta.json`); run after the seal.  The item's summary and ITEM_TABLE.md are written only when that file carries the current seal, every k of BETA_KS, 50 fold-fits, all_identical and matches_main_run (s0-mechanics-13); the first run writes rows, candidates, fits and checks and exits with status 3, and `s1_ed.py --from-rows` (seal and code digests checked) writes the tables after the poisoning run.
- Fit poisoning of the recipe-2 LOCO static read-set selection (`r5.ed.poison_loco`): each cluster's labels replaced by seeded noise, the selection for that cluster bit-identical; leaky positive control (selection with the cluster's own rows) caught for at least one cluster; written to `results/leakage/poison_ED_loco.json`; a failure stops the item before any row is computed.
- The block-Y peek positive control goes through the real level function and comparator: levels from every measured metal must change when the unread metals are scrambled (s0-mechanics-6).

## RNG seeds

- Cluster bootstrap: 8675309 (gen20k.inference.SEED).  RANDOM Monte Carlo: seed_for('ED|RANDOM_MC|<design>|<variant>').  Intervention scrambles: seed_for('ED|intervention|<design>'), seed_for('ED|static_intervention').  Fit poisoning: seed = split_seed * 10 + fold (round 4's).  Sign-flip tests: exact at G <= 13 (seed_for('<key>|SF_POOL') otherwise).

## Command lines (run from D:\ml_separator_gh)

```
set PYTHONIOENCODING=utf-8
rem pre-seal engineering check on real data (no metric printed or written; y and every y-derived column replaced by seeded noise before any table runs; allowed before the seal)
.venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_ed.py --smoke --design BPG
rem after the seal: reproduction run (TECHNICAL SMOKE), the arms (rows; exit 3), the fit-poisoning check, then the tables from the rows
.venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_ed.py --repro --design both
.venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_ed.py --design both
.venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_ed_poison.py
.venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_ed.py --from-rows --design both
.venv/Scripts/python.exe -B -W ignore -m pytest generations/gen21/round5_20261024/tests/test_ed.py -q -p no:cacheprovider --basetemp generations/gen21/round5_20261024/cache/_pytest
```
