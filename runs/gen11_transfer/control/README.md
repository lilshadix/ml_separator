# GEN10 control, re-derived for GEN11

GEN11 may not take the frozen GEN10 numbers on trust, so every headline figure below was
recomputed here from the **raw stored predictions** in `runs/gen10_final/final_locked/`
— the row-level out-of-fold frame and the per-(ligand, seed, fold, repeat) k-shot records —
and never copied out of a summary table. Where a stored summary CSV exists it is used only
as a second opinion, and the agreement is reported beside the recomputation.

**Verdict: every claimed number reproduced.** No check failed. Two presentational
discrepancies and one factual error in the GEN11 brief are recorded in §4.

Scripts: `recompute_stage1.py` (identity, level, shape), `recompute_stage2.py` (frontier),
`recompute_stage3.py` (assembly). Run them with
`PYTHONPATH=<worktree>/src <venv>/bin/python recompute_stage1.py` in that order.

## 1. Identity

| check | claimed | recomputed | result |
|---|---|---|---|
| cohort fingerprint | `bed178ec1a7a82b0` | `bed178ec1a7a82b0` | PASS |
| cohort rows / extractants / Tanimoto chemotypes | 5,248 / 152 / 79 | 5,248 / 152 / 79 | PASS |
| ECFP clusters (the macro-MAE unit) | — | 131 | recorded |
| bundle `dataset.parquet` sha256 | `fefbefc6…4faf5dd` | `fefbefc6…4faf5dd` | PASS |
| the same digest is the one pinned in `gen3_protocol.json :: dataset_sha256` | — | yes | PASS |
| FROZEN arm width (`METAL`+`COND`+`ECFP`+`MASSACTION`+`RECOVERED`) | 2,146 | 2,146 | PASS |
| folds reproduce from `seeded_group_kfold(tanimoto_cluster, 5, seed)` | — | 0 mismatches in 5 × 5,248 rows | PASS |

The fold check is the strongest of these: the fold index was recomputed for all five split
seeds and compared row-by-row against the `fold` column stored in `oof_finalists.parquet`.

## 2. Zero-shot, FROZEN cohort, `GEN9_SHAPE_RECOMPOSED`

Level metrics from `oof_finalists.parquet` (5 seeds × 5,248 rows). Shape metrics by running
gen9's `curve_shape_table` over those same rows and `runs/gen9_shape/curves/curve_membership.parquet`
— the recomputed per-curve table is **bit-identical** to the stored `curve_shape.parquet`
(4,730 curves, max |Δ| = 0.0 on every column checked).

| metric | claimed | recomputed | abs Δ |
|---|---|---|---|
| macro MAE (per ECFP cluster) | 0.9695 | 0.9694845 | 1.5e-5 |
| offset MAE (per-ligand \|mean residual\|) | 0.8212 | 0.8211715 | 2.8e-5 |
| extractant-axis shape MAE | 0.469 | 0.4691936 | 1.9e-4 |
| extractant-axis predicted slope, median | 1.024 | 1.0240137 | 1.4e-5 |
| extractant-axis span recovery (median, guarded) | 0.423 | 0.4232909 | 2.9e-4 |
| within-curve Spearman (extractant axis) | 0.886 | 0.8856282 | 3.7e-4 |
| within-curve sign accuracy (extractant axis) | 0.926 | 0.9255716 | 4.3e-4 |
| acid-axis shape MAE | 0.545 | 0.5453227 | 3.2e-4 |
| lanthanide (`metal_series`) axis shape MAE | 0.291 | 0.2911286 | 1.3e-4 |

Every delta is below half a unit in the last claimed digit — the claims are the rounded
recomputations, not different numbers.

Per-seed macro MAE (the spread GEN11 will need for any paired comparison):
0.9944, 0.9465, 0.9386, 0.9788, 0.9891 for seeds 104729 / 130363 / 155921 / 196613 / 262147
(sd 0.0254);
pooled MAE 1.1218.

Strata (macro MAE, FROZEN) recomputed from the same rows and matched to `strata.csv` at
max |Δ| = 2.2e-16: `ALL` 0.9695, `RESIDUAL_SHAPE` 1.3018, `PURE_LEVEL` 1.8850,
`PARTIAL_LEVEL` 1.4409, `ALREADY_GOOD` 0.5433, `MISMATCH_ROWS` 1.0001, `CLEAN_ROWS` 0.9289.

## 3. Frozen-pipeline frontier, 99-ligand common cohort

Recomputed from `kshot_detail.parquet` (12,834,720 raw records) by re-deriving the common
cohort from the record-level pool/eval counts (`n_pool ≥ 5` and `n_eval ≥ 2` in *every*
record of *every* arm → exactly 99 ligands, as claimed), then averaging MAE per ligand and
per arm.

| k | pipeline arm | claimed | recomputed | abs Δ |
|---|---|---|---|---|
| 0 | `ZERO_SHOT` | 1.0358 | 1.03579944 | 5.6e-7 |
| 1 | `SLOPE_L_s1_K1@CENTRAL_THEN_SPREAD` | 0.6539 | 0.65387284 | 2.7e-5 |
| 2 | `SERIES_ML@CENTRAL_THEN_SPREAD` | 0.5593 | 0.55929881 | 1.2e-6 |
| 3 | `SERIES_ML@CENTRAL_THEN_SPREAD` | 0.4928 | 0.49280898 | 9.0e-6 |
| 5 | `SERIES_ML@CENTRAL_THEN_SPREAD` | 0.4405 | 0.44054181 | 4.2e-5 |

The whole of `frontier_common.csv` was reproduced, not just these five cells: all 1,629 arms
match at max |Δ| = 2.2e-16, with no arm present in one table and absent from the other.

`gen10_frontier.csv` is the anchor table GEN11 compares against. It carries the recomputed
values (full precision), the gen8 and gen9 frontiers for the same cohort, and the *best
deployable* arm per k — which at k = 5 is `SERIES_ML@D_OPTIMAL` at 0.44032, 0.0002 better
than the frozen rule. That arm is **not** the control; the frozen rule is.

## 4. What did not match

Nothing in the claim list failed. Three things are worth carrying forward:

1. **The GEN11 brief's cohort shape is wrong.** It states `frame` is 5,248 × 2,492. The
   frame is 5,248 × **5,642** (19 feature blocks totalling 4,975 columns plus metadata).
   2,146 — the FROZEN arm's width — is correct.
2. **Two shape endpoints depend on the aggregation convention, and the model card and
   `final_locked/shape_by_axis.csv` use different ones.** The claimed slope median 1.024 and
   span recovery 0.423 are medians over all 775 extractant curves **pooled across the five
   seeds** (as in `axis_representation/shape/shape_by_axis.csv`). `final_locked/shape_by_axis.csv`
   averages the five per-seed summaries instead and therefore reports 1.139 and 0.417 for the
   same predictions. Both are recorded in `gen10_control.json`. The other seven endpoints are
   means over curves and are identical under either convention. GEN11 must state which
   convention it uses before quoting a slope median or a span recovery.
3. **`best_deployable_mae` is not the frozen number.** `frontier_best.csv` reports a
   per-k best arm chosen after seeing the results; it beats the pipeline by 0.0002–0.006 at
   k ≥ 1. Comparisons must be against `pipeline_mae`.

## 5. Files

* `gen10_control.json` — every claimed vs recomputed metric with deltas, per-seed values,
  the curve-table and frontier-table agreement checks, and the verdict block.
* `gen10_frontier.csv` — the locked five-row anchor table.
* `protocol.json` — seeds, fold rule, model-seed formula, cohort fingerprint, bundle hash,
  metric definitions with their module paths, and the environment versions the control was
  computed under.
