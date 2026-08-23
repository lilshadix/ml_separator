# gen9 — a shape-preserving global objective, and a learned first experiment

**One sentence:** gen8 showed the model draws every unseen titration too flat and
that *which* measurement you take is worth as much as a better model; gen9 asks
whether the flattening can be removed from the training objective rather than
repaired afterwards, and whether the one-shot oracle's rule can be predicted before
the measurement exists.

*Cohort fingerprint `bed178ec1a7a82b0` — identical to gen6, gen7 and gen8. The curve
geometry gen9 trains on was recomputed from scratch and asserted identical to gen8's
(7,207 memberships, 1,176 curves, max abscissa delta 0.0). All 15 gen8 reference
numbers reproduce exactly in this environment before any gen9 arm was fitted.*

## Read in this order

| file | what it answers |
|---|---|
| `protocol.md` | what was pre-registered, and when |
| `decision_report.md` | the decision, and what gen10 should do |
| `failure_analysis.md` | every arm that did not work, and why |
| `reproduction/reproduction.json` | Phase 0 — does gen8 still reproduce here |
| `shape/shape_by_axis.csv` | the flattening endpoints per curve type |
| `rerun_diff.csv` | every number that moved when a defect was fixed, and why |
| `shape/shape_bootstrap.csv` | the paired intervals behind every shape claim |
| `acquisition/oracle_identity.csv` | the identity, re-derived under gen9's protocol |
| `acquisition/primary_summary.csv` | every first-experiment policy against the oracle |
| `frontier/attribution_chain.csv` | which component bought what |
| `frontier/table_b_common_cohort.csv` | the longitudinal k-shot frontier |
| `sensitivity/` | FROZEN / QUARANTINED / CORRECTED |
| `data_audit/` | TWE-24, DMDPhPDA, and what gen9's own scan found |
| `manifest.json` | environment, git state, content hash of every artefact |

## Directory map

```text
curves/          gen9's independent curve reconstruction and its audit
reproduction/    Phase 0: gen8's numbers, rebuilt in this environment
finalists/       the locked five-seed sweep: frozen anchor, A0 control, six shape arms
recomposed/      the EXPLORATORY relative-position recomposition, five seeds
relmono/         the EXPLORATORY frozen arm + relative-position columns, five seeds
phase1/          the superseded three-seed screen — pre-dates the reproducibility fix,
                 kept for the audit trail, quoted nowhere
shape/           per-curve slope / span / shape metrics and their intervals
acquisition/     the oracle identity, the learned policies and the feature ablations
kshot/           the k-shot detail tables, one file per global model
frontier/        Table A / Table B, the attribution chain, the adaptation curve
sensitivity/     the three data cohorts
data_audit/      anomaly scans, TWE-24 evidence, the flag table
figures/         the six figures
logs/            stdout of every run, kept so a number can be traced to a run
```

## Reproducing

```bash
P=.venv/bin/python
$P scripts/gen9_curves.py                       # curve geometry + audit
$P scripts/gen9_reproduce_gen8.py               # Phase 0 — stop if this fails
$P scripts/gen9_tune.py                         # learner hyperparameters, in-fold only

# the locked five-seed sweep: frozen anchor, A0 control, four samplers, two weight arms
$P scripts/gen9_shape_train.py --stage finalists --seeds 5 --with-frozen --with-control \
   --arms A1_ADJ_d05:ROW_ADJACENT:0.5:0.0 A2_END_d05:ROW_ENDPOINT:0.5:0.0 \
          A3_RAND_d05:ROW_RANDOM_PAIR:0.5:0.0 A4_MULTI_d05:ROW_MULTISCALE:0.5:0.0 \
          X_MULTI_d2:ROW_MULTISCALE:2.0:0.0 X_MULTI_d8c:ROW_MULTISCALE:8.0:0.0:LHM:cluster \
   --tag finalists
# the two EXPLORATORY arms
$P scripts/gen9_shape_train.py --stage finalists --seeds 5 --with-recomposed --tag recomposed
$P scripts/gen9_shape_train.py --stage finalists --seeds 5 --with-relative-monolith --tag relmono

OOF="runs/gen9_shape/finalists/oof_all.parquet runs/gen9_shape/recomposed/oof_all.parquet \
     runs/gen9_shape/relmono/oof_all.parquet"
$P scripts/gen9_shape_analysis.py --oof ${=OOF} --reference GEN9_A0_ROW_ONLY --out runs/gen9_shape/shape
$P scripts/gen9_acquisition.py --seeds 5 --repeats 8 \
   --ablations geometry geometry+prediction full full_shuffled
# one k-shot process per global model; --with-learned only where the chain needs it
$P scripts/gen9_kshot.py --oof ${=OOF} --models GEN9_SHAPE_RECOMPOSED --seeds 5 --repeats 12 \
   --with-slope --with-series --with-learned --tag recomposed --out runs/gen9_shape/kshot
$P scripts/gen9_frontier.py --detail runs/gen9_shape/kshot/*_detail.parquet \
   --new-model GEN9_SHAPE_RECOMPOSED --new-adapter SERIES_MAP --new-policy LEARNED_BLEND
$P scripts/gen9_sensitivity.py --oof ${=OOF} --out runs/gen9_shape/sensitivity
$P scripts/gen9_data_audit.py
$P scripts/gen9_figures.py --candidate GEN9_SHAPE_RECOMPOSED
$P scripts/gen9_manifest.py
$P -m pytest tests/test_gen9_*.py -q
```

The three-seed `--stage phase1` screen was run first and is **superseded**: it predates
the reproducibility fix (decision report §0.3, issue 6) and none of its numbers are
quoted. The finalists command above covers the same five samplers at five seeds.

`gen9_manifest.py --verify` recomputes every artefact digest and reports drift.

## The invariants gen9 enforces in code

* **The control is the frozen model, not a lookalike.** `CurveBoost` at one stage,
  `learning_rate = 1`, squared row loss and `lambda = 0` is algebraically
  `REC_ecfp_plus_recovered`; the test asserts it to 1e-9 on real-shaped data. Every
  shape claim is quoted against `A0_ROW_ONLY`, the same learner with the curve terms
  switched off, so a difference between arms is a difference in *objective*.
* **A training curve pair never crosses a fold boundary.** `build_pairs` is only ever
  handed training row ids; `CurveBoost.fit` asserts every pair addresses a training
  row; a per-fold audit counts curves straddling the boundary (it is 0).
* **No curve dominates by being long.** Every sampler emits a bounded number of pairs
  per curve and weights renormalise so each curve carries equal total weight. The
  effective contribution per curve and per ligand is written to disk.
* **A deployable acquisition feature reads conditions and predictions, never a
  target.** Declared per column in `FEATURE_PROVENANCE`; the test walks the table
  rather than the docstring, and a separate test destroys every candidate's target
  and demands the same selection.
* **Every normalisation is pool-scoped.** A feature that moved when a non-candidate
  row moved would be unreproducible at deployment; a test perturbs rows outside the
  pool and requires the features not to move.
* **The hierarchy's penalties are estimated inside the fold.** From the fold's own
  training ligands and their out-of-fold residuals — never a corpus-wide constant
  computed before cross-validation.
* **Splits are BLAKE2b, never Python's salted `hash()`.** An AST scan fails the suite
  if any gen9 module calls the builtin, and subprocess tests under four different
  `PYTHONHASHSEED` values check the pool/evaluation draw, the acquisition training
  blocks and the curve pairs.
* **Nothing may vanish.** Row accounting per arm per seed (expected / returned /
  scored / missing / NaN / inf) and a pairing audit (arms called paired must share
  fold, ligand, repeat, pool size and evaluation size) both write files and both can
  fail the run.
