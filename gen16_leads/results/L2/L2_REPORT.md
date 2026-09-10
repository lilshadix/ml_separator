# L2 — the curvature: gate closed, r0 study not run

*Lead L2 of `gen16_leads/PRE_REGISTRATION.md` §3.  Discovery seeds (5), designs B / BR / BQ / A / BP,
extractant-macro MAE of pairwise log SF, n = 90 extractants in 45 chemotype blocks, gen13's
chemotype-blocked paired bootstrap (10 000 replicates, seed 8675309).  Everything here is
DISCOVERY.  Full tables and the closure paragraph: `L2_GATE.md`; numbers: `gate_checks.json`.*

## Verdict

**CLOSED.**  Registered rule: OPEN if H = MAE(G14) − MAE(O_CURV_LPO) ≥ 0.02 under BP and the 95 %
CI excludes zero.  H under BP = **+0.0287**, percentile 95 % CI [−0.0013, +0.0532], BCa
[+0.0000, +0.0544], p = 0.061, 61/90 extractants improved, 5/5 seeds positive, LOCO-stable
(range [+0.022, +0.034]), passes P1 = False.  The margin is met; the interval is not.  The gate
is valid: the same loop reproduces gen15's `G14` (0.5000794414203691), `O_CURV`
(0.42721106205801906) and `FLAT` (0.5885062528901843) under BP to the last digit, and
`s4_floor._fit` with no metals dropped reproduces `bench.coef` to < 1e-9.

The honest, leave-pair-out curvature prize is about +0.03 in every design (B +0.0298, BR +0.0268,
BQ +0.0264, A +0.0323, BP +0.0287; same sign in all five), and no selecting design can
distinguish it from zero: bootstrap SE 0.014, minimum detectable headroom at 80 % power 0.039.
Only design A — exact-extractant hold-out, which never selects — excludes zero (p = 0.031).
Per the registered stopping rule the r0 study (subset R, ICC(r0) vs ICC(b), arm `G14_R0`, null
`G14_R0_SHUF`) was **not run**; no r0 file exists and no r0 contrast was evaluated.

## What was run

| item | value |
|---|---|
| arms | 4: `FLAT`, `G14`, `O_CURV` (in sample), `O_CURV_LPO` |
| registered contrasts | 1 × 5 designs = 5 rows (`OCURVLPO_vs_G14`) |
| exploratory contrasts | 3 × 5 designs = 15 rows (`OCURV_vs_G14`, `OCURVLPO_vs_FLAT`, `OCURV_vs_OCURVLPO`) |
| pair rows per design | 70 750 (14 150 per seed; every cell held out once per seed) |
| run time | 98 s, one process, `OMP_NUM_THREADS=2` |

Construction (`gen16/l2_lpo.py`): for every fold of `gen13sep.splits.all_folds` (default
discovery seeds) the pair table is `gen13sep.amplitude_bench._pair_frame` and the `Ctx` is built
exactly as `valuebench.run_arms` builds it.  `G14` and `O_CURV` go through the identical
`coef @ basis` path.  `O_CURV_LPO` uses G14's amplitude `_logistic_sign(ctx) × train_mean_magnitude()`
with the second coefficient of `s4_floor._fit(C[i], basis, drop=(a, b))` — the cell's ridge refit
(ridge 0.5, basis rows at norm √14) on `C = Y − nanmean(Y)` without the two metals of the scored
pair — so `log SF = (a_G14·basis[0] + b_lpo·basis[1])[a] − same[b]`.

## Five-design table (extractant-macro MAE of log SF)

| arm | B | BR | BQ | A | BP |
|---|---|---|---|---|---|
| FLAT | 0.5885 | 0.5885 | 0.5885 | 0.5885 | 0.5885 |
| G14 | 0.4932 | 0.4906 | 0.4905 | 0.4921 | 0.5001 |
| O_CURV (in sample) | 0.4216 | 0.4218 | 0.4233 | 0.4168 | 0.4272 |
| **O_CURV_LPO** | 0.4634 | 0.4638 | 0.4642 | 0.4599 | **0.4714** |

## Registered contrast, five designs (positive favours O_CURV_LPO)

| design | H | 95 % CI | BCa | p | improved | seeds | LOCO | P1 |
|---|---|---|---|---|---|---|---|---|
| **BP** | **+0.0287** | [−0.0013, +0.0532] | [+0.0000, +0.0544] | 0.061 | 61/90 | 5/5 | stable | no |
| B | +0.0298 | [−0.0020, +0.0563] | [−0.0018, +0.0565] | 0.065 | 63/90 | 5/5 | stable | no |
| BR | +0.0268 | [−0.0062, +0.0529] | [−0.0047, +0.0539] | 0.104 | 61/90 | 5/5 | stable | no |
| BQ | +0.0264 | [−0.0087, +0.0538] | [−0.0093, +0.0535] | 0.129 | 62/90 | 5/5 | stable | no |
| A | +0.0323 | [+0.0032, +0.0581] | [+0.0045, +0.0596] | 0.031 | 65/90 | 5/5 | stable | yes |

Exploratory rows (all five designs, all in `gate_contrasts.csv` with `family = exploratory`):
`OCURV_vs_G14` reproduces gen15 §2's +0.0729 [+0.0304, +0.1334] p = 0.0006 under BP exactly;
`OCURVLPO_vs_FLAT` = +0.1171 [+0.0149, +0.1860] p = 0.019 under BP (the honest oracle is still
well above doing nothing); `OCURV_vs_OCURVLPO` = +0.0442 [+0.0158, +0.1027] p < 1e-4 under BP
(the self-fitting inflation, 70/90 extractants).

## In-sample vs leave-pair-out gap, against gen15 §1a

gen15 `s4b_oracle_honesty.csv` (one seed, all 521 cells in one frame, *true* sign, global
constants): SIGN_OWNB 0.3771 → SIGN_OWNB_LOPO 0.4245, gap +0.0474.  Here (5 seeds, fold-wise
tables, G14's *predicted* sign, training-fold constants): +0.0442 under BP, +0.0408 to +0.0431
elsewhere.  The pair set is identical (14 150 pairs per seed); the gap is 0.003 smaller because
the predicted sign is wrong for about a fifth of cells and there the amplitude error dominates
both arms equally, and because the constants are fold-level under a publication mask.  The
conclusion is unchanged: roughly 60 % of gen15's in-sample `O_CURV − G14` was the oracle fitting
the noise of the scored pair.

## Temptations recorded, not acted on (for `REFUTATION_LOG.md`)

1. The BCa lower bound under BP is +5.7 × 10⁻⁶; reading the rule as "the *BCa* interval excludes
   zero" would open the gate on a six-millionths margin.  The script fixed the percentile interval
   before the run, P1 requires both intervals, and p = 0.061 fails p < 0.05 under any reading.
2. Design A is the only design whose interval excludes zero; quoting it is barred (A never selects).
3. The r0 construction had a convention question — `basis[1] = (basis[0]² − 1)/0.92906`, so the
   vertex of `coef @ basis` on the standardised-radius axis is `−a·0.92906/(2b)`, not `−a/(2b)`;
   I fixed the vertex reading (the vertex on the `basis[0]` axis, the only reading under which
   "inside the measured radius range" and "clipped to the radius axis's span" refer to the same
   axis) before the gate ran, with the literal-scale variant to be written as exploratory only.
   It became moot when the gate closed; it is recorded here so that anyone reopening L2 does not
   treat the two as interchangeable — the in-range subset R differs between them.

## Files

* `gen16_leads/gen16/l2_lpo.py` — the leave-pair-out gate module (imports `s4_floor._fit`).
* `gen16_leads/scripts/l2_gate.py` — runs the gate, writes everything below.
* `gen16_leads/results/L2/gate_board.csv` — 4 arms × 5 designs, `design`/`arm`/`lead` columns.
* `gen16_leads/results/L2/gate_contrasts.csv` — 20 rows, `paired_contrasts` columns + `family` + `lead`.
* `gen16_leads/results/L2/gate_perext_{B,BR,BQ,A,BP}.csv` — per-(seed, extractant, arm) tables.
* `gen16_leads/results/L2/gate_checks.json` — reproduction check, headroom, gaps, timing.
* `gen16_leads/results/L2/L2_GATE.md` — the full gate report and the closure paragraph.

No r0 files (`l2_r0.py`, `r0_*.csv`, `r0_by_extractant.csv`) exist: the registered stopping rule
ends the lead at the gate.  Nothing under `gen13_separation/`, `gen14_direction/`, `gen15_curve/`
or `src/` was modified; no `seeds=` was passed anywhere (`g16_audit_seeds.py`: clean).
