# gen10 — the finalization sprint

**One sentence:** gen9 found that the model was missing *where a requested point
sits inside the proposed design*; gen10 asks whether that representation is
physically stable, whether the recomposition it needed is an architecture or a
post-processing step, and what exact pipeline to freeze.

*Cohort fingerprint `bed178ec1a7a82b0` — identical to gen6–gen9. gen9's manifest
verifies at 154/154 artefacts with zero drift in this environment, its Phase-0
reproduction of gen8 passes 15/15, and every gen9 arm is reproduced bit-for-bit
(to the ExtraTrees thread-order floor, ~1e-15) inside gen10's classes before any
new arm was trained.*

## Read in this order

| file | what it answers |
|---|---|
| `GEN10_FINAL_DECISION_REPORT.md` | what broke, what was fixed, what was rerun, what is frozen, the thirteen questions |
| `GEN10_FAILURE_ANALYSIS.md` | every arm that did not work, and why |
| `GEN10_SELF_AUDIT.md` | the sixteen checks, with the numbers they compared |
| `GEN10_FINAL_MODEL_CARD.md` | the frozen pipeline, its inputs, its limits |
| `GEN10_ERROR_CEILING.md` | where the remaining error lives and how much is reachable |
| `GEN10_RECOMMENDED_DEPLOYMENT.md` | how to use it, what to measure first, what not to trust |
| `headline_tables/` | every headline table as CSV/JSON, regenerated from raw predictions |
| `final_locked/pipeline_frontier.json` | the exact configuration of the frozen pipeline and its frontier |
| `manifest.json` | environment, git state, BLAKE2b digest of every artefact |

## Directory map

```text
reproduction/            Phase 0: gen9 manifest, gen8 references rerun, nesting, determinism
query_consistency/       Phase 1: the query-set consistency benchmark
feature_access/          Phase 2A/2B: max_features and context-replication arms
level_shape/             Phase 2C/2D: explicit level+shape, residual, two-branch arms
set_context/             Phase 3: DeepSets / attention set encoder (contingent)
axis_representation/     Phase 4: rank / local / hybrid / curve-window representations
adaptation/              Phase 5: FIXED / INNER / ML series priors vs OFFSET_K3
acquisition/             Phase 6: realised-regret acquisition vs MEDOID
data_ceiling/            Phase 7: consistent / mismatch / duplicate / TWE-24 / uncertain cohorts
publication_sensitivity/ Phase 8: LOPO, publication-, chemotype- and two-factor-blocked intervals
budget_simulation/       Phase 9: breadth vs depth from held-out adaptation curves
error_decomposition/     Phase 10: seven-component ceiling table
final_locked/            Phase 11: the finalists under the locked protocol
self_audit/              Phase 12: the sixteen checks
headline_tables/         every table the reports quote
logs/                    stdout of every run
```

## Reproducing

```bash
P=.venv/bin/python
export PYTHONPATH=src

$P scripts/gen10_phase0.py                                     # stop if this fails
$P scripts/gen10_query_consistency.py --seeds 5 \
   --arms FROZEN REL_MONOLITH SHAPE_RECOMPOSED                 # Phase 1, gen9 arms
$P scripts/gen10_train.py --stage feature_access               # Phase 2A/2B
$P scripts/gen10_train.py --stage level_shape                  # Phase 2C/2D
$P scripts/gen10_train.py --stage axis_representation          # Phase 4
$P scripts/gen10_train.py --stage set_context                  # Phase 3 (contingent)
$P scripts/gen10_adaptation.py                                 # Phase 5
$P scripts/gen9_acquisition.py --oof runs/gen9_shape/recomposed/oof_all.parquet \
   --model GEN9_SHAPE_RECOMPOSED --label realised_mae \
   --ablations geometry geometry+prediction --repeats 8 \
   --out runs/gen10_final/acquisition --tag realised           # Phase 6
$P scripts/gen10_data_ceiling.py                               # Phase 7
$P scripts/gen10_publication_sensitivity.py                    # Phase 8
$P scripts/gen10_budget_simulation.py                          # Phase 9
$P scripts/gen10_error_decomposition.py                        # Phase 10
$P scripts/gen10_query_consistency.py --seeds 5 --arms <finalists> \
   --out runs/gen10_final/query_consistency_finalists          # Phase 1 on finalists
$P scripts/gen10_final_locked.py --finalists <names> --selected <name>   # Phase 11
$P scripts/gen10_self_audit.py --selected <registry name> --stage-dir <dir> # Phase 12
$P scripts/gen10_report_tables.py
$P scripts/gen10_manifest.py
$P -m pytest tests/test_gen10_*.py -q                          # add -m "not slow" for the fast set
```

`gen10_manifest.py --verify` recomputes every digest and reports drift.

## The invariants gen10 enforces in code

* **The gen9 arms are special cases of the gen10 classes**, asserted at the float
  floor on real folds, so a difference between a gen10 arm and a gen9 arm is a
  difference of design, never of implementation.
* **Curves are rebuilt from the frame in front of the model.** Per-ligand
  reconstruction equals the cohort table on every one of 143 ligands, and a
  partition-closure audit proves no curve straddles a fold; the query-scoped
  object is therefore the evaluation object, and the consistency benchmark is a
  study of what happens when the user's design differs from the published one.
* **Every design-relative column carries a declared sensitivity class** (ENDPOINT,
  RANK, LOCAL, MOMENT, ABSOLUTE) written before the benchmark ran.
* **Model outputs that become another model's targets are rounded** at 1e-9
  (`stabilise`), because an unrounded parallel forest feeding a second stage is
  irreproducible — gen9's issue 6, and gen10's own first defect.
* **Second-stage targets are cross-fitted, never in-sample**; the in-sample trap
  control is carried and labelled.
* **Splits are BLAKE2b, never Python's salted `hash`**; the AST scan now also covers
  gen10 and the shared gen6/gen8 modules.
* **Nothing may vanish**: row accounting per arm per seed, pairing audits, and a
  determinism probe on every trained arm.
