# gen6 — the diversity generation: pre-registered protocol (2026-08-19)

**Status: pre-registration.** Written before any gen6 model was fitted. Phase 0 results are
recorded in `gen6_phase0_report.md`; Phase 1 results get their own dated document. Nothing in
this file may be edited after the Phase 1 run starts — corrections go in the results document
as amendments, with the reason.

## 0. Why a new generation instead of another arm

Gen2–Gen5 asked *which model, which descriptors*. The answers converged:

| study | result |
|---|---|
| gen2/gen3 pair ablation A0–A6 | local 3D adds nothing over 2D on held-out extractants |
| gen4 candidates | only transitive projection confirmed; extended 2D split-sensitive; per-ligand scale not predictable from 2D descriptors |
| metal-site descriptors | gains within the permuted-null band |
| SNN / simplicial | fails the group-balanced endpoints; guarded blend falls back |
| gen5 levels (4 regimes) | `lig2d_ext` and the 13-column donor census do transfer; 3D is *worse* (−0.102, CI [−0.177, −0.034]); MASSACTION adds 3–8 % where the ligand is known |
| k-shot (pairs) | from k≈2 the champion is no better than two parameters fitted to the same measurements |

The residual failure is specific and unchanged by six studies: on a genuinely new ligand the
model predicts the *shape* of the response (shape R² 0.17–0.31) and not the *level*
(deployable within-ligand R² ≤ 0). Gen5 also measured where its edge lives — the level model's
advantage over a global constant is **−0.041 below nearest-neighbour Tanimoto 0.4** and
**+0.656 above 0.8**. It is a near-neighbour lookup.

The eligibility rule that produced every one of those cohorts is
`min_rows_per_extractant = 10`. It keeps 91 of 190 extractants and discards precisely the
unusual chemistry: relaxing it to 3 adds 367 rows — **+7.5 % on top of BASE's 4,881, i.e. 7.0 %
of the resulting cohort** — and buys **131 vs 74 ECFP clusters** and **79 vs 40 Tanimoto-0.7
super-clusters**.

**Primary gen6 hypothesis: the limiting variable for zero-shot prediction on new extractant
chemistry is the chemical diversity of the training cohort, not model capacity.**

Gen6 is built to test that cleanly, and to distinguish it from four rival explanations:
unmodelled extraction regime/speciation; missing reorganisation/preorganisation information;
missing outer-sphere/aggregation/solvent physics; experimental-series/provenance heterogeneity.

## 1. Isolation and provenance rules

* New layer `src/lanthanide_separation/gen6/` + `scripts/*gen6*`/`scripts/run_diversity_*`.
  No gen2–gen5 module, protocol, doc or `runs/` artifact is edited. The gen5 harness
  (`lanthanide_separation.levels`) is **imported and reused verbatim**, so a gen6 run of a gen5
  arm must reproduce the gen5 number.
* Every gen6 run writes a manifest recording: dataset hash, source-table hash, feature-registry
  hash, source-code hashes, split definition, chemistry-cluster definition, provenance state,
  model seed, split seed, training/test extractant ids, training/test super-clusters, exact row
  ids, preprocessing description, predictions, per-ligand errors, OOD statistics,
  `validation.json`, `artifact_hashes.json`, `_SUCCESS.json`. `validate_run` is fail-closed.
* Exploratory selection may not enter a confirmation run silently: any arm, threshold or
  feature block chosen after seeing gen6 data is labelled `exploratory` in the report.

## 2. Definitions frozen for the whole generation

| term | definition |
|---|---|
| extractant | `canonical_smiles` (190 in the source table) |
| cell | unique (extractant, condition_id, metal_symbol) — the unit `min_cells` counts |
| ECFP cluster | bit-identical 2,048-bit fingerprint group |
| super-cluster (chemotype) | single-linkage Tanimoto ≥ 0.7 group |
| BASE91 | training restricted to extractants with ≥ 10 cells (91 extractants) |
| EXPANDED152 | training allowed ≥ 3 cells (152 extractants) |
| hard chemistry | test ligands with max Tanimoto to the **BASE** training set < 0.4 (and < 0.6) |

The chemistry map is built once over **all 190 extractants**, target-independent, and frozen.

*Amendment, 2026-08-19, after Phase 1 — the pre-registered wording was checked and found vacuous.*
This section originally read "the all-190 partition restricted to either cohort equals that cohort's
own partition exactly (0 splits, 0 merges)". That comparison is near-tautological at the chemotype
level: `levels.build_level_dataset` computes `tanimoto_cluster` *before* applying the row filter, so
the labels a cohort carries were already derived from all 190 extractants. It remains a genuine check
at the bit-identical level, and it is the check that governs *reproducing gen5*, whose folds used
exactly those labels.

The non-vacuous property, measured with `chemistry.freeze_is_conservative`, is that the frozen map is
**coarser, never finer** than a from-scratch clustering of the cohort alone: 0 splits at both levels
and both cohorts, with 3 merges at `min_cells = 10` (45 chemotypes → 40) and 1 at `min_cells = 3`
(80 → 79). Merging enlarges a held-out chemotype and makes the hold-out stricter; splitting would put
related chemistry on both sides of a fold and does not occur.

## 3. Primary and secondary metrics

Primary: **equal-chemistry-unit macro MAE** (one ECFP cluster = one vote), grouped to match the
deployment regime. Co-primary on new chemistry: **hard-chemotype MAE** and **ligand offset MAE**.

Secondary: shape MAE, shape R², within-ligand rank concordance, sign accuracy, fraction within
0.5 and 1.0 log units, median ligand MAE, worst-quartile ligand MAE.

Reported with every table: n independent ligands, n ECFP clusters, n super-clusters, n_eff
(Kish), and the nearest-neighbour similarity distribution.

Never select on pooled R². Never quote a metric without its grouping and regime.

Decomposition, per held-out ligand *l* with residual `e_i = ŷ_i − y_i`:

```
b_l        = mean_{i∈l} e_i                    offset_mae = mean_l |b_l|
ŷ_c, y_c   = centred within ligand             shape_mae  = mean_l mean_i |ŷ_c − y_c|
identity   SSE_total = SSE_centred + Σ_l n_l·b_l²        (unit-tested exactly)
```

## 4. Phase 0 — no expensive modelling (this document's companion report)

1. Provenance reconstruction and audit; `legacy_compatible` vs `provenance_strict`.
2. All-190 chemistry map, frozen, with a partition-stability proof.
3. Level/shape/offset metrics implemented and validated on the existing gen5 OOF.
4. Fixed-test BASE91/EXPANDED152 split machinery with byte-identical test rows.
5. **Exact reproduction of the gen5 champion under the gen6 harness.**

   *Amendment, 2026-08-19, before any Phase 1 fit — reason recorded here rather than silently.*
   The gate was first written as "per-seed macro MAE must match
   `runs/gen5_levels_20260818T211105Z/per_seed_metrics.csv` to < 1e-6". That is not a
   well-posed gate: the published run was fitted on the cluster, and a forest grown on a
   different CPU/BLAS/scikit-learn is a different forest even with an identical cohort,
   identical folds and identical seeds. "Reproduces" is really four claims, and three of them
   can and must be exact:

   | layer | gate | measured 2026-08-19 |
   |---|---|---|
   | cohort | identical rows and row ids | 4,881 rows, identical |
   | split | identical fold for every row, regime and seed | 97,620 assignments compared, **0 differ** |
   | metrics | gen6 metric code on the *published* OOF returns the published numbers | 80 cells, max abs diff **4.4e-16** |
   | fit | refit macro MAE against published | mean 0.003, **max 0.010** (cross-machine) |

   The fit layer therefore reports a *measured tolerance*, not an equality. Two consequences,
   both binding on how gen5 may be read: an effect smaller than ≈ 0.01 macro MAE is not
   distinguishable from a change of machine; and within one machine the fit **is** exact —
   this harness reproduces the local `gen5_massaction_local_20260818` run bit-for-bit on every
   fold whose training set contains no all-NaN column, the remainder differing only through
   the deliberate `DropAllNaNColumns` change in commit `09e04e3`.

   A failure of the cohort, split or metric layer stops Phase 1. A fit deviation above 0.02
   stops Phase 1 until explained.
6. Hard-chemotype reporting.
7. A retrospective single-split diversity pilot.

## 5. Phase 1 — the two experiments that decide the generation

Learner and features are frozen to what already works: `LevelRegressor` (ExtraTrees, 400 trees,
`max_features` 0.30, `min_samples_leaf` 2), feature sets `MC_lig2d_ext_massaction` (current
champion) and `MC_donors_massaction`. **No new architecture in Phase 1.**

### Experiment A — BASE91 vs EXPANDED152 on identical test rows

One shared cohort at min_cells = 3; folds hold out whole Tanimoto super-clusters; the test rows
of a fold are identical for every arm; only the training row mask changes.

| arm | training rows | role |
|---|---|---|
| `BASE` | ≥ 10 cells | the gen5 cohort |
| `EXPANDED` | ≥ 3 cells | + 61 sparse, chemically distant extractants |
| `EXPANDED_ROWMATCHED` | all sparse + random dense to BASE's row count | row-budget-matched depth control |
| `EXPANDED_SHUFFLED` | EXPANDED with the added sparse rows' targets permuted among themselves | information null |

Pre-registered hypotheses (paired bootstrap over held-out super-clusters, 5,000 replicates,
one shared index matrix; ≥ 4/5 split seeds in the same direction):

| H | statement | pass condition |
|---|---|---|
| **A1** | Adding sparse diverse chemistry improves overall accuracy | `macro_MAE(BASE) − macro_MAE(EXPANDED) > 0`, CI95 low > 0 |
| **A2** | The gain is largest on the hardest chemistry | gain at NN < 0.4 > gain overall, both CI95 low > 0 |
| **A3** | The gain is in the *level*, not the shape | `offset_mae(BASE) − offset_mae(EXPANDED)` CI95 low > 0, and larger than the shape-MAE gain |
| **A4** | The gain is information, not row count | EXPANDED beats `EXPANDED_ROWMATCHED` by less than it beats BASE **and** beats `EXPANDED_SHUFFLED` with CI95 low > 0 |

Falsification: if A1 fails, chemical coverage is not the binding constraint at this scale and
the generation pivots (Part XVII case C of the brief). If A3 fails while A1 passes, the
representation, not the coverage, is missing ligand extraction propensity (case B).

### Experiment B — diversity vs depth at equal row budget

Same splits and endpoints; training rows acquired under `DEPTH`, `DIVERSITY`, `RANDOM`,
`MAXMIN` at budgets 250 / 500 / 1,000 / 2,000 / 3,000 / all. All policies are label-free
(unit-tested by permuting `log_D` and asserting the selection is unchanged).

| H | statement | pass condition |
|---|---|---|
| **B1** | At equal row budget, diversity beats depth | `DIVERSITY` macro MAE < `DEPTH`, CI95 low > 0, at ≥ 2 consecutive budgets |
| **B2** | The advantage is larger on hard chemistry than overall | as A2, on the same budgets |
| **B3** | Depth saturates | `DEPTH` curve slope between the two largest budgets is within noise of zero |

## 6. Required negative controls (no claim without its null)

| claim type | null |
|---|---|
| descriptor family | permuted twin: same width, same marginals, chemistry-preserving permutation |
| diversity expansion | row-count-matched depth control **and** target-shuffled sparse rows |
| transfer learning | source-label shuffle, permuted source predictions, random embedding of equal width |
| active learning | random acquisition and the "widest" heuristic |
| hierarchical model | the same learner without the decomposition |
| any k-shot claim | `PAIRMEAN + the same k measurements` and `ΔZ trend + the same k measurements` |

`5/5 seeds` is never sufficient on its own — the five split seeds re-partition the same
ligands. The independent-chemistry-unit bootstrap is the load-bearing statistic.

## 7. Deliberately out of scope until Phase 1 reports

Hierarchical physics decomposition (C1–C3), multi-fidelity level→pair transfer (D1–D3),
active k-shot design (E), retrospective ligand acquisition (F), regime-mixture models,
`THERMO_ACTIVITY`, preorganisation/outer-sphere/aggregation proxy blocks, and any 3D work.
The single 3D route that stays open for a later phase is free→bound conformer-ensemble
reorganisation statistics; the single-geometry SNN branch is closed as a documented negative
result and must not be reopened for more capacity.

## 8. Decision tree at the end of Phase 1

| case | observation | conclusion | next investment |
|---|---|---|---|
| A | EXPANDED improves offset MAE, strongest at NN < 0.4 | chemical coverage is the primary zero-shot bottleneck | more diverse ligands; active ligand acquisition; hierarchical level model |
| B | EXPANDED improves shape but not offset | representation misses ligand extraction propensity | phase/solvation/aggregation descriptors; reorganisation; multi-fidelity transfer |
| C | EXPANDED improves neither | experiment/provenance/speciation heterogeneity, or the features cannot represent the mechanism | publication/series effects; activities and free ligand; diluent/phase metadata; replication noise model — **not** more capacity |
| D | 2–3 selected measurements beat every zero-shot model | zero-shot is a screening prior | zero-shot prior + optimal few-shot adaptation, benchmarked against `PAIRMEAN + k` |
