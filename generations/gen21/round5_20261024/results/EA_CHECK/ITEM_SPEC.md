# Item 1.0 EA_CHECK: item specification (frozen with the item code at the Stage-0 seal, before the item's first post-seal run; revised at the pre-seal review)

M35 positive-control and six-design check. Estimand **E-A, DIAGNOSTIC (seen)**; every number is EXPLORATORY (seen)
(CLAUDE_CODE_PROMPT.md §4.1, §6.0, §6.10 row 1.0, §8, §9). Report rule (§6.10): *records the seen positive-control
outcome; no E-A slot.* Nothing here is a test of a new model; no in-corpus p-value is called significant.

Code: `r5/ea_check.py` (functions), `scripts/s1_ea_check.py` (CLI), `tests/test_ea_check.py` (synthetic and
construction-only). Reuse is by import only (`r4.refs`, `r4.leakage`, `r3.harness.cell_rows`, `r3.interp`,
`g21.cache.build_fold`, `g21.cache.load`, `g21.evaluate`, `gen20k.splits.folds_for`, `gen20k.design`, `gen20k.data`,
`r5.common`, `r5.sealcheck`). No design-folder script is copied or run.

## 1. Order (§1.2)

1. Pre-seal, no score: `--build-cache` (fold records + cache-poisoning + isolation checks) and `--smoke` (every metric
   suppressed). Both may run before the Stage-0 seal.
2. Stage-0 seal (covers this file's code through `r5_seal.py`'s code manifest).
3. `--repro`: the reproduction target of §6.0, labelled TECHNICAL SMOKE. A mismatch stops the item; a fix is an
   append-only amendment and a reseal.
4. The scoring run (default command). It repeats the reproduction target first and stops on a mismatch, then checks
   the caches against their manifest and the leakage checks, then computes every arm and table below.

## 2. Rows (frozen)

- **E-A rows:** round-4 `predictions/REF_{BPG,BP}.parquet`, C3 (k_max ≥ 3) at code budget k = 3 (**four reads**),
  DOPT_Q nested read sets; 229 cells and 1,802 queries per seed; dev seeds 104729, 130363, 155921, 196613, 262147.
  Loaded by `r5.common.load_rows` (requires round 4's intervention and identity checks to have passed).
- **Six-design rows:** the same 1,802 queries per seed on designs A, B, BR, BQ, scored from the round-5 fold records
  `cache/<design>_<seed>_<fold>.pkl` through `r3.harness.cell_rows` (the code path that built REF), and on BP and BPG
  from the REF parquets. Fixed denominators are asserted: identical (seed, cell, A, B) keys, identical y and identical
  cluster labels in all six designs.
- **Units:** `cluster_cpg` (13 units on C3; union-find over chemotype, publication and gen19 copy group,
  `r4.metrics.add_clusters`). The labels come from the BPG rows and are mapped to every design. BP's own labelling
  must define the same partition.
- **Strata:** `C5_in` (all rows; primary), `C5_out` (without pub_6a0ef8528c; SENSITIVITY), `C5_cell_out` (without
  cell c950dc2982e96b83; single-cell influence).
- **Big fold:** in each seed, the fold that holds the most scored rows.

## 3. Arms and contrasts (frozen arm list)

| id | contrast (cand − ref) | rows | role |
|---|---|---|---|
| C1 | B1 − B0 | E-A rows, BPG (decides) and BP (sensitivity) | positive control, outside any family (§4.1 item 7) |
| C2 | HIST_D7 − B1 | E-A rows, BPG and BP | diagnostic; HIST_D7 (fixed −0.072) is not fold-independent |
| C3 | HIST_D7 − B0 | six designs A, B, BR, BQ, BP, BPG | the training-free contrast of the six-design table |

Arms: B0 = PWL_R (radius interpolation, constant beyond the extreme reads); B1 = B0 + one fold-fitted Gd amplitude
(round-4 REF column; only on BP / BPG); HIST_D7 = B1's form with β = −0.072. B0 and HIST_D7 are training-free, so they
are the only arms on A, B, BR and BQ (§4.1 item 5). No arm is added after results are seen; one added later is post hoc
and goes into an amendment.

## 4. Gate and report rules (sealed M35, §4.1 items 2–4)

For one contrast on one row set:
- relative pair-SSE `1 − SSE_cand / SSE_ref` over all scored rows of a seed, then the seed mean;
- SF_POOL: one-sided sign-flip on the seed-mean per-unit ΔSSE = SSE_ref − SSE_cand of the cluster_cpg units,
  H0: mean ΔSSE ≤ 0. Exact enumeration over all 2^G sign vectors when G ≤ 13 (G = 13 here: 8,192 vectors). If G > 13,
  B = 4,999 random flips with p = (1 + #{T* ≥ T}) / (1 + B) (`r5.common.sf_pool`);
- balanced-MAE guard: seed-mean Δbalanced-MAE (ref − cand) ≥ −0.001 (round 4's chemotype-macro MAE).

**Pass** = relative SSE ≥ 0.02 **and** SF_POOL p ≤ 0.025 **and** guard ≥ −0.001. BPG with C5-BTBP in decides. BP and
the C5 strata are labelled sensitivities. A pass is labelled DIAGNOSTIC, because M34's kill fired.

Recorded outcomes (`summary.json` → `decision`):
- `M35_passes_its_positive_control` = C1 passes on BPG (C5_in). `M35_kill_fires` = the opposite. The convention is
  abandoned only if the replay fails the positive control (§4.1, RI §4.2).
- C2's gate outcome, as a diagnostic.
- CONSISTENCY CHECK against the seen replay: C1 on BPG, relative SSE +5.85 % (2 decimals in %) and SF_POOL p 0.00317;
  on BP, p 0.00269 (5 decimals) [IR M35; PC P2]. A mismatch is reported. It does not stop the item, because these are
  consistency checks, not reproduction targets.

Also reported for every contrast and stratum: R² of both arms (pooled within seed, then the seed mean; never clipped),
dR², MSE, balanced MAE, per-seed relative SSE and dR², seeds with a positive relative SSE, unit W/L/T, SIGN p (win
share), SF_LOG p and IM_REL t / p (breadth diagnostics), and the top-3 unit share of ΔSSE.

## 5. Mandatory tables (§4.1 item 5, §6.0)

| table | file (design suffix `_BPG` / `_BP`) | content |
|---|---|---|
| gate | `gate_<d>.csv` | every quantity of §4, C1 and C2 × the three strata |
| C5-BTBP in / out | `c5_<d>.csv` | gate quantities per stratum |
| per-cluster (13 rows) | `units_<d>.csv` | per-unit seed means of n, SSE_ref, SSE_cand, ΔSSE, L, rel, W/L/T and share of ΔSSE. Provenance columns: cells, publications, chemotypes, ORNL-import cells, **ornl_only**, **data_location classes** |
| big fold vs other folds | `bigfold_<d>.csv` | gate quantities on big-fold rows and on other-fold rows, the shares of rows and of ΔSSE, the big fold per seed and its training C3 cells (round-4 `results/provenance/fold_training_<d>.csv`) |
| per-seed | `perseed_<d>.csv` | relative SSE, R² of both arms, dR², Δbalanced-MAE per seed |
| LOPO | `lopo_<d>.csv`, `lopo_values_<d>.csv` | leave-one-publication-out jackknife of the relative SSE (`r5.common.lopo`: point, jackknife SE, t(P − 1) 95 % interval, min / max and the publication left out), C5 in and out |
| arms | `arms_<d>.csv` | `r5.common.arm_table` for B0, B1, HIST_D7 (C5 in / out) |
| breadth (§4.1 item 6) | `breadth_<d>.csv`, `breadth_units_<d>.csv`, `giant_publications_<d>.csv` | see §6 |
| six-design | `sixdesign.csv`, `sixdesign_levels.csv`, `sixdesign_units.csv`, `sixdesign_readsets.csv` | C3 on all six designs, C5 in / out: gate quantities, LOPO range, levels of B0 and HIST_D7, per-cluster tables with provenance, and the label-free share of cells whose four-read set differs from BPG's |
| M08 appendix (§4.1 item 8) | `m08_moments_<d>.csv`, or `m08_status.json` | only when `REGISTRY.csv` marks M08 PLANNED; see §7 |
| row level | `rows_EA_<d>.parquet`, `rows_sixdesign.parquet` | every scored row with its arms, flags, provenance class and label |
| reproduction | `repro.json` | §8 |
| summary | `summary.json` (or `summary_<d>.json` with `--design`), `ITEM_TABLE.md` (the §9.3 table, both designs) | |

Provenance (metadata only, from `round4_20261022/results/provenance/cells.csv`):
- **ORNL import** = `ornl_import_all`. A unit is **ORNL-only** when every one of its cells is an ORNL import.
- **data_location class (round-5 label, P5 F1 applied)**: from regular expressions on the recorded `data_location`:
  FIG (`fig` or `graph`), TAB (`tab`), MIXED (both), NONE (empty), OTHER; the 10 comment-located C3 cells of
  pub_b9dfc468cd ("table1": TAB, 4 cells), pub_19783c039b ("fig2 raffinate": FIG, 1) and pub_f8a621ede4 ("fig3" /
  "fig3b": FIG, 5) are located (DATA_CONTRACT §1; reviewer exception to D8, s0-fidelity-11). The per-cluster table
  also carries **data_location class (raw cells.csv)** in a second column (`data_location_classes_raw_cells_csv`;
  row-level `loc_class_raw`).
- **Duplicate sensitivities (REGISTRY 1.0-S07 … S09; s0-fidelity-8):** every E-A contrast's M35 quantities on the C5-in
  rows with the V2-1/2/3 cells dropped (noV2), with the second cell of every Stage-0 flagged bit-identity link dropped
  (noBITDUP), on the V2-3 condition-aware stratum (the 11 C3 hidden-merge cells; S1 open) and without it;
  `duplicates_<d>.csv`, SENSITIVITY.
- **Cache poisoning in the scoring run** is read from the sealed manifest `manifests/ea_check_caches.json` (100 of 100
  records poison-identical), not from the unsealed report file (s0-mechanics-10). The big-fold table refuses a
  missing `fold_training_<design>.csv` (s0-mechanics-9).

## 6. Breadth diagnostic (§4.1 item 6; sealed, never a gate)

The statistic is the equal-weight mean over units of L = log(SSE_ref / SSE_cand), using seed-mean unit SSEs. The test
is SF_LOG, an exact sign-flip on L (random flips only if G > 13), with an IM t beside it. The equal-weight mean of the
per-unit relative SSE is also reported. Variants:
- **SMALL_POOLED** (the sealed diagnostic): clusters with < 3 cells or < 20 rows per seed in the stratum pooled into one
  "small sources" unit;
- **ORNL_COLLAPSED** (sensitivity): the ORNL-only clusters collapsed into one unit, the other clusters kept;
- **RAW** (reference): the 13 clusters unpooled (the replay's unit set).

Publication-level relative SSE inside the giant cluster (the cluster with the most rows): per-publication seed means of
n, SSE_ref, SSE_cand, ΔSSE, L, rel and share of the giant cluster's ΔSSE.

## 7. M08 relation-class moment table (appendix; only if PLANNED in REGISTRY.csv)

The residual is B1's, e = y − B1, on the E-A rows of each seed. For unordered pairs of distinct cells that share a
strict-adjacent pair type, the moment is ρ = Σ e_ip e_jp / √(Σ e_ip² · Σ e_jp²): the shared share of the pair-residual
second moment, uncentred. It is computed per seed, then the seed mean is taken. Relation classes:

| class | relation |
|---|---|
| SE_SP | same extractant, same publication (other conditions) |
| SE_OP | same extractant, other publication |
| OE_SP_SC | other extractant, same publication, same chemotype |
| OE_SP_OC | other extractant, same publication, other chemotype |
| OE_OP_SC | other extractant, other publication, same chemotype |
| OE_OP_OC | other extractant, other publication, other chemotype (baseline) |

Identifiability column:
- product units: a cluster when both cells lie in it, otherwise the cross-cluster pair;
- the top unit's share of Σ|e_ip e_jp|;
- publications and clusters contributing;
- the leave-one-cluster-out range of ρ.

A class is "identified" when it has ≥ 3 product units and the top unit holds ≤ 80 % of the products; otherwise it is
"not identified". The table is computed for C5 in and C5 out, on BPG and BP.

## 8. Reproduction target (§6.0) and leakage checks (§8)

- **Reproduction (TECHNICAL SMOKE):** on every BP and BPG fold of the gen21 caches (`g21.cache.load`, read-only;
  `g21.evaluate.assert_test_isolation` on each), B1 is rebuilt from round 4's stored fold β
  (`results/REF/fitted_<d>.json`, `ref:B1.coef`), and B0 and HIST_D7 through the same row builder. Each must match the
  REF parquet within 8.9e-16 on all 18,020 rows, with identical keys, y and read sets.
- **Fold records (§4.1 item 5):** `g21.cache.build_fold` called directly (never `build_all`, never `gen21/cache/`) with
  one BLAS thread, for every fold of `gen20k.splits.folds_for` for A, B, BR and BQ × 5 dev seeds × 5 folds
  (100 records), pickled with protocol 4.
- **Cache key** (`manifests/ea_check_caches.json`):
  - data version and content: SHA-256 of V1 log D, of the gen13 coefficients and of the cell order;
  - code: LF digests of `g21/cache.py`, `gen20k/{arms,cov,data,design,runner,splits}.py`, `gen13sep/splits.py` and
    `r4/leakage.py`;
  - split: digest of the train, test and inner indices;
  - support contract and hyperparameters (stated);
  - pickle SHA-256.

  The scoring run refuses a record whose key or bytes changed.
- **Cache poisoning (§8 item 3):** each record is rebuilt from `r4.leakage.poisoned_bench(bench, train_index)`, in
  which log D, the gen13 coefficients and every `logD__*` / `repsd__*` / `replicate_sd_median` value of every cell
  outside the outer training fold are random. That covers the held-out cells and the cells the BR / BQ masks drop.
  The rebuilt record is compared bit for bit, NaN equal to NaN, field by field: everything except `seconds`, including
  `selection_scores`. Report: `results/leakage/cache_poison_EA_CHECK.json`.
- **Isolation (§8 item 1), design-aware:** every overlap is reported. Required zero: A — cells and extractants;
  B / BR / BQ — also chemotypes; BP — also publications; BPG — also copy groups. Inner blocks must lie inside the
  outer training fold and be disjoint. `g21.evaluate.assert_test_isolation` assumes BP-type isolation, so it is used
  on the BP and BPG folds only.
- **Intervention (§8 item 4):** in every A / B / BR / BQ fold, every unread metal of every held-out C3 cell is
  replaced by a random number. Read sets and every B0 / HIST_D7 prediction must be bit-identical. The positive control
  (a method that reads `block["Y"]`) must move.
- **Support (§8 item 5):** exactly four reads per C3 cell; readings kept exactly at the read metals; no scored pair with
  both ends read; |ΔZ| = 1.
- Report: `results/leakage/intervention_EA_CHECK.json`.

## 9. RNG seeds (scheme `paths.seed_for(key) = 20261023 + crc32(key)`, P2's scheme)

| procedure | key |
|---|---|
| cache poisoning | `EA_CHECK\|poison\|<design>\|<seed>\|<fold>` |
| intervention scramble | `EA_CHECK\|intervention\|<design>\|<seed>\|<fold>` |
| SF_POOL / SF_LOG random flips (only if G > 13; not expected) | `<cand> - <ref>\|<design>\|<stratum>\|cluster_cpg\|SF_POOL` (`\|SF_LOG`), breadth `...\|breadth\|<variant>\|SF_LOG` |
| smoke fake y (technical smoke only) | `EA_CHECK\|smoke\|fake_y` |

The exact sign-flip at G ≤ 13 uses no RNG.

## 10. Exact command lines (run from `D:\ml_separator_gh`, one process at a time)

```
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_ea_check.py --build-cache
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_ea_check.py --smoke
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B -W ignore -m pytest generations/gen21/round5_20261024/tests/test_ea_check.py -q -p no:cacheprovider --basetemp generations/gen21/round5_20261024/cache/_pytest
# after the Stage-0 seal:
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_ea_check.py --repro
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_ea_check.py
# optional single-design runs (BPG decides; BP is a sensitivity):
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_ea_check.py --design BPG
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_ea_check.py --design BP
```

After the scoring run, `round4_20261022/scripts/r4_seal.py --verify` must still report 5bacfe4c, and the round-3 LF
digests must still match (§8).

## 11. Labels (§9.1)

| what | label |
|---|---|
| every E-A number | EXPLORATORY (seen); DIAGNOSTIC |
| BP rows and the C5_out / C5_cell_out strata | adds SENSITIVITY |
| six-design rows | "training-free arms only"; designs other than BPG add SENSITIVITY |
| breadth | "breadth diagnostic, never a gate" |
| M08 | "M08 appendix" |
| reproduction | TECHNICAL SMOKE |
| replay comparison | CONSISTENCY CHECK |

No number of this item is placed next to 0.452 as a new E-A level: B1's own level on these rows *is* E-A's reference.
