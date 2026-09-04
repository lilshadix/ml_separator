# GEN10 recommended deployment

*How to use the frozen pipeline, what to measure first, and what not to trust.
Written for the person planning experiments, not for the person training models.*

## 1. What you hand over, and what you get back

**Hand over:** the extractant's structure, and the *complete list* of conditions
you are considering — every metal, acid, acid molarity, extractant molarity,
diluent, additive, temperature. The model positions each point inside the
titration that list describes, so the list is part of the question: a prediction
for 0.03 M made inside a 0.01–0.1 M plan is not the same prediction as one made
inside a 0.01–1 M plan. Hand over the plan you will actually run, and keep the
plan with the predictions.

**Get back:** `log D` per requested point, with the level of every titration
set by the chemistry model and the shape by the design-aware shape model
(`GEN10_FINAL_MODEL_CARD.md` §3). The design-aware part is the part that moves
when the design moves — by 0.12 log units at the median when two far decoy points
are added (§4) — so the list you hand over should be the list you will run.

## 2. The first thing to do with an unseen extractant is measure it once

Zero-shot error on a new chemotype is ~1.0 log units macro, and half of that is
the ligand's level — a single number no structure descriptor in this corpus
predicts. One measurement removes two thirds of it:

| measurements on the new ligand | macro MAE (99-ligand common cohort) | gen9 | gen8 |
|---|---|---|---|
| 0 | **1.036** | 1.036 | 1.061 |
| 1 | **0.654** | 0.654 | 0.667 |
| 2 | **0.559** | 0.575 | 0.588 |
| 3 | **0.493** | 0.511 | 0.523 |
| 5 | **0.441** | 0.468 | 0.474 |

**Which point to measure first:** the *central* one — the medoid of the
standardised condition axes of your plan (`CENTRAL_THEN_SPREAD`). Every learned
alternative gen9 and gen10 tried, including one trained on the realised one-shot
regret, ends up choosing centrality when it is allowed to choose honestly
(Phase 6: α = 1.0 in 24 of 25 folds). Do not spend effort on a cleverer first
point.

**Measure inside the series you intend to predict.** One point calibrates the
acid titration it sits on and largely not the same ligand's lanthanide series in
a different diluent (gen8). From the second point on, the `SERIES_ML` adapter
fits a level per series plus small response-slope corrections; it is worth 0.03
macro over gen8's `OFFSET_K3` at k = 2–5 and it needs the measured points to be
in the series you care about.

## 3. Planning a campaign: breadth first, new chemotypes first

From the held-out adaptation curves of 99 ligands (`GEN10_ERROR_CEILING.md` §4):

* the **first** point on a ligand is worth 0.38 macro MAE; the second 0.08; the
  third 0.06; the fourth and fifth 0.02 each;
* the first point on a ligand **far** from the training chemistry is worth 0.49;
  on a ligand near it, 0.23;
* with a budget smaller than the number of unseen ligands, one point on each
  ligand ordered **farthest-from-training-chemistry first** beats every
  alternative tried, including Tanimoto max-min diversity and any depth-first
  allocation, by 0.05–0.07 macro at a budget of half the ligands;
* only once every ligand has one point does a second point per ligand pay, and
  only after two does a third.

So: cover chemotypes, then ligands, then points. gen6 reached the same conclusion
about *training* data from the other direction; this is the deployment-side rule.

## 4. What not to trust

* **Zero-shot levels.** Use them to rank, not to decide. A zero-shot `log D` can
  be off by a decade on a new chemotype.
* **Predictions for mixtures.** 271 rows in the training corpus are synergistic
  systems represented by one component's structure; the model's shape error on
  them is 0.35 log units larger than elsewhere. If your system has a second
  extractant or a phase modifier that is itself an extractant, the model has not
  seen it as such.
* **Extrapolation outside the measured windows.** The recovered slopes are
  learned from 25 ligands' extractant titrations and 68 ligands' acid titrations;
  the design-relative representation positions a point *within* a plan, and a
  plan that spans four decades when the corpus spans one is outside what the
  shape model has seen. Predictions at the edges of such a plan are less
  constrained than in its middle.
* **The lanthanide-axis shape.** The tetrad-scale structure across the series is
  the one axis gen9/gen10 did not improve (shape MAE 0.29–0.31); treat
  predicted selectivities between adjacent lanthanides as the least reliable
  output.
* **Three ligands and one suspected transcription error** remain in the training
  set by design (nothing was deleted); their neighbourhoods in chemical space
  carry their error (`data_audit/`, `data_ceiling/`).

## 5. Running it

```bash
export PYTHONPATH=src
P=.venv/bin/python
$P - <<'EOF'
from lanthanide_separation.gen7.harness import load_cohort, build_folds, FoldContext
from lanthanide_separation.gen10.architectures import RecomposedModel, prime_raw_cache
from lanthanide_separation.levels import LEVEL_TARGET_COLUMN
cohort = load_cohort(); prime_raw_cache(cohort); frame = cohort.frame
# train on everything: a single "fold" whose training index is every row
import numpy as np
from lanthanide_separation.gen7.harness import Fold
fold = Fold(seed=0, fold=0, train_index=np.arange(len(frame)), test_index=np.arange(0),
            held_out_chemotypes=())
ctx = FoldContext(fold=fold, cohort=cohort, feature_columns=(), model_seed=fold.model_seed)
model = RecomposedModel(representation_name="GEN9", gen9_compat=True).fit(
    frame, frame[LEVEL_TARGET_COLUMN].to_numpy(float), ctx)
# `query` is a frame of conditions in the cohort's column schema, no log_D column
# prediction = model.predict(query)
EOF
```

The exact configuration is `final_locked/pipeline_frontier.json`; the k-shot
adapters are `gen8.slope_restore.build_slope_adapters(strengths=(1.0,))`
(`SLOPE_L_s1_K1`) and `gen10.adaptation.CorrectedSeriesAdapter(estimator="ML")`,
whose `fit_fold` needs out-of-fold residuals of the training ligands — at
deployment, the stored five-seed OOF table of the frozen model
(`runs/gen9_shape/recomposed/oof_GEN9_SHAPE_RECOMPOSED.parquet`) serves that purpose.

## 6. When to revisit

Not for another architecture. Revisit when either of these is true:

1. **new titrations on new chemotypes exist** — 25 extractant-titration ligands
   and 8 chemotypes is the binding constraint on the shape result, and one point
   on a far-chemotype ligand is worth 0.49 macro;
2. **the data problems are resolved at the source** — TWE-24's primary document,
   the decade duplicates, and the second species on the 271 mismatch rows.
   Together these are worth ~0.04 macro at the ligand level and more on the rows
   they touch, and no model can recover them.
