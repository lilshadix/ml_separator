# Triage of the 610-row data-quality review queue

*2026-09-04. Analysis only — nothing was corrected, deleted or retrained. Frozen gen10 anchor
(`A_GEN10_CONTROL|JOINT|FROZEN_3|FROZEN_8|HIERARCHICAL|lam1`), C-FULL cohort, 5 split seeds,
26,240 held-out row x seed predictions.*

## Why this was run before any cleaning

The plan was to resolve the queue and then retrain if the effect exceeded ~0.01–0.02 macro MAE.
gen10 had already measured the aggregate data-quality contribution at 0.037 of 0.970, so that
trigger looked unlikely to fire. This asks what the queue is actually worth, and on which endpoint.

## How the queue reaches the modelled data

The queue keys on archive measurement ids; the cohort keys on `(extractant, condition_id,
metal_symbol)` cells. Joining exactly — `condition_id` recomputed with the repository's own
`levels.condition_labels` — gives:

| | |
|---|---|
| queue measurement ids that are in the 5,992-row bundle | **924 of 7,255** |
| **cohort rows flagged** | **638 of 5,248 (12.2 %)** |
| extractants affected | 56 of 152 |
| chemotypes affected | 40 of 79 |

> A coarser join on `(extractant, metal)` alone reports 65.6 % of rows. That number is wrong and
> should not be used: one flagged measurement contaminates its whole extractant x metal group.

## The aggregate endpoint does not fire

| rows | n (row x seed) | median abs err | mean abs err |
|---|---|---|---|
| clean | 23,050 | 0.957 | 1.120 |
| flagged | 3,190 | 1.000 | 1.131 |

**Pooled MAE 1.1218; if every flagged row errored like a clean one it would be 1.1205 — a
difference of 0.0013.** That is an order of magnitude below the proposed 0.01–0.02 trigger and
below this pipeline's 0.01 cross-machine tolerance. **A clean-and-retrain study scored on macro MAE
would correctly conclude that nothing happened.**

## The irreducible-residual endpoint does fire

Removing a per-`(extractant, split_seed)` offset leaves the part of the error a level correction
cannot remove — the endpoint gen10 used to show that mismatch rows carry a *shape* problem, not
just a level one.

**Ligand-level Mann-Whitney: p = 0.015, median irreducible residual 0.499 (56 flagged ligands) vs
0.346 (128 clean).** Flagged ligands carry about 44 % more irreducible error.

## Which categories are worth resolving — the actionable part

Median irreducible residual by flag (clean-row baseline **0.602**):

| category | cohort rows x seed | median irreducible |
|---|---|---|
| component pairing ambiguous | 90 | **0.993** |
| structure deduced by elimination | 90 | **0.993** |
| identical record, different publication | 40 | 0.963 |
| doi repaired, verify | 5 | 0.927 |
| diluent name ambiguous | 1,675 | 0.675 |
| same conditions, different log D | 795 | 0.637 |
| same value, conflicting condition | 770 | **0.485** |

Two things follow.

1. **Identity problems hurt; value-conflict problems do not.** The three identity-shaped categories
   sit 0.36–0.39 above the clean baseline. `same_value_conflicting_condition` — the second largest
   category in the queue at 192 review records — sits *below* it.
2. **The top two rows are the same 18 cohort rows** (verified: identical row-id sets), so this is
   one group of ~18 rows, not two findings.

## Recommendation

Do not process all 610 records. Process the identity-shaped ones — the 39
`extractant_name_structure_conflict`, 36 `component_pairing_ambiguous`, 4
`structure_deduced_by_elimination` and 19 `identical_record_different_publication` records, ~98 of
610 — and leave the 192 `same_value_conflicting_condition` and 307
`same_conditions_different_logD` records alone unless a separate question needs them.

Score the result on **irreducible residual on the affected ligands**, not on macro MAE. The
aggregate endpoint is already measured at 0.0013 and will not move.

## Caveats

Seven categories were compared with no multiplicity correction, and the top group is 18 rows.
This is a triage diagnostic for choosing what to resolve, not a pre-registered test. The direction
agrees with gen10's independent finding that name/structure-mismatch rows carry a significantly
larger irreducible residual (0.934 vs 0.582, ligand-level p = 0.0018).

Reproduce: `docs/results/review_queue_triage_20260904.md` documents the method; the join is
`levels.condition_labels` on the frozen bundle against `runs/gen7_architecture/cache/cohort.parquet`.
