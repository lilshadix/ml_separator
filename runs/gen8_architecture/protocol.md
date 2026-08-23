# gen8 pre-registration — series-aware few-shot architecture for `log D`

*Written 2026-08-20, **before** the finalist comparisons are run and before any
architecture has been selected. Everything decided after an outer-test number was
inspected is labelled EXPLORATORY in the results report and is never quoted as a
confirmed effect.*

## 0. What is already fixed, and cannot move

| item | value | why it is frozen |
|---|---|---|
| cohort | `runs/gen7_architecture/cache/cohort.parquet`, 5,248 rows / 152 extractants / 131 ECFP clusters / 79 Tanimoto chemotypes, fingerprint `bed178ec1a7a82b0` | identical to gen6 and gen7, so every gen8 number is on byte-identical rows |
| fold plan | `seeded_group_kfold` over `tanimoto_cluster`, 5 folds, seeds 104729 / 130363 / 155921 / 196613 / 262147 | gen5's algorithm, reused byte-for-byte |
| model seed | `42 + fold*1009 + 9_999_991`, independent of the split seed | gen7 harness contract |
| frozen global model | `REC_ecfp_plus_recovered` (gen7's best arm, macro MAE 0.9807) | gen8 does **not** refit it; every k-shot number is a calibration of this one model's out-of-fold predictions |
| metric | one ligand, one vote (macro), with the gen6 offset / shape decomposition | unchanged since gen6 |

**No global model is retrained for the primary endpoints.** That is deliberate: it
makes every k-shot comparison paired down to the row, and it keeps gen8 from
becoming another learner sweep.

## 1. Primary endpoints

For the frozen model, on held-out chemotypes:

1. **`MAE(k)` for k = 0, 1, 2, 3, 5**, macro over ligands.
2. **`ACTIVE − RANDOM` at k = 1 and k = 2** — the value of choosing the measurement.
3. **`BEST_ADAPTER − OFFSET_K1` at k = 1, 2, 3, 5** — the value of the architecture.
4. **`ORACLE − BEST_DEPLOYABLE`** — how much room is left.

Secondary, reported for every arm: offset MAE, shape MAE, within-ligand Spearman,
fraction within 0.5 and 1.0 log units, worst-quartile ligand MAE, hard-chemistry
(nearest-neighbour Tanimoto < 0.4) MAE.

## 2. Primary architectures (frozen list)

**Adaptation modes** — how many ligand-specific degrees of freedom `k` measurements move:

* `OFFSET_K1` — level only. **The baseline to beat.**
* `OFFSET_K2` — level + lanthanide trend, ridge-shrunk, unpenalised intercept.
* `OFFSET_K3` — level + lanthanide + acid slope + extractant-concentration slope, same shrinkage.
* `NO_MODEL` — the mean of the measurements. The null that killed gen5's k-shot story.
* `NEAREST_OBSERVED` — nearest measured point in condition space. A lookup with no model.

**Architectures** (each scored on identical rows through the same adapter interface):

* Conditional Neural Process, three ligand-representation variants plus a residual variant.
* Physics-latent: ligand → mass-action parameters, MAP-updated by the measurements.
* Series-aware curve baselines: spline residual, mass-action curve, local GP over the swept axis.

Ridge penalty for K2/K3 is **fixed at 4.0 on standardised columns** and is not tuned.

## 3. Primary acquisition policies (frozen list)

`RANDOM` (reference), `CENTRAL`, `MEDOID`, `MEDIAN_PREDICTION`, `MID_ACID`,
`FARTHEST_FROM_EXISTING`, `MAX_CONDITION_COVERAGE`, `MAX_ENSEMBLE_SD`,
`MIN_ENSEMBLE_SD`, `MAX_MODEL_DISAGREEMENT`, `MAX_PREDICTIVE_VARIANCE`,
`MAX_EXPECTED_VARIANCE_REDUCTION`, `D_OPTIMAL`, `CENTRAL_THEN_SPREAD`,
plus the non-deployable `ORACLE`.

`MEDOID`, `MEDIAN_PREDICTION`, `MID_ACID` and `CENTRAL_THEN_SPREAD` were declared
**after** the exhaustive calibration-geography scan and **before** the acquisition
run; they are the deployable form of a measured stratum effect, they have no free
parameter, and they are labelled as geography-derived wherever they appear.

Selection for k > 1 is **sequential**: choose, observe, update, choose again. No
policy may see the target of a candidate it has not selected. `ORACLE` may, and is
marked non-deployable everywhere.

## 4. Evaluation contract

* **Outer:** held-out Tanimoto chemotype, unchanged and not weakened.
* **Inner (few-shot):** each repeat splits a held-out ligand's rows once into a
  **candidate pool** and a disjoint **evaluation set** (half each, evaluation at
  least 2 rows, pool capped at 48 candidates). **Every policy and every adapter
  sees the identical pool and is scored on the identical evaluation rows.**
  Calibration rows are never scored, because they are never in the evaluation set.
* The pool/evaluation split is a deterministic function of
  `(seed, repeat, ligand, n_rows)` alone, so an architecture evaluated in a
  separate later run is still paired row-for-row with this one.
* 12 repeats × 5 split seeds. Eligibility `n_rows >= 4`.

## 5. The two k-shot tables (brief §16)

* **Table A — maximal coverage.** Every ligand eligible at each k.
* **Table B — common cohort.** The identical ligand set at every k, namely those
  with enough pool rows for k = 5 plus at least two evaluation rows.
  **The adaptation curve is read off Table B.** Table A is reported so the
  coverage cost is visible, never as the longitudinal claim.

## 6. Inference

Paired bootstrap, 5,000 replicates, resampling the **Tanimoto chemotype** the folds
actually held out, with every member ligand travelling with its chemotype.
Reported per comparison: point estimate, percentile CI, BCa CI, block macro,
units improved, seeds positive. Pairing preserves ligand, repeat, candidate pool
and evaluation rows.

## 7. Success thresholds (from the brief, fixed in advance)

| endpoint | useful | strong | major |
|---|---|---|---|
| 1-shot MAE | ≤ 0.65 | ≤ 0.60 | ≤ 0.55 |
| 2-shot MAE | ≤ 0.55 | ≤ 0.50 | ≤ 0.45 |

An `ACTIVE − RANDOM` gain counts only if its 95 % interval excludes zero **and**
at least 4 of 5 seeds agree in sign.

## 8. Falsification conditions (brief §32)

gen8's central hypothesis fails if any of these hold, and each will be stated
plainly if it does:

1. `ORACLE` best-point selection is barely better than `RANDOM`;
2. functional/architectural models do not beat plain offset correction;
3. series-aware models cannot reduce shape MAE;
4. no deployable acquisition policy outperforms `RANDOM`;
5. calibration helps only rows very near the measured condition;
6. one-shot gains disappear on the common ligand cohort;
7. any result depends on leaking series identity or a hidden target.

**Condition 1 was tested first and is already resolved: it does not hold.** The
exhaustive scan gives `RANDOM` 1-shot 0.709 against `ORACLE` 0.522 on identical
ligands, a gap of 0.187 — so acquisition is worth studying. That test was run
before any policy was written, and its result is what licensed §3 above.

## 9. Data cohorts (brief §14)

Three sensitivity cohorts over the DMDPhPDA duplicate, defined in
`case_studies/dmdphpda_cohorts.json`:

* `FROZEN` — every row kept, as gen6/gen7 had them. **Primary.**
* `QUARANTINED` — the 70 rows of the corrupted copy removed.
* `CORRECTED` — those 70 rows rescaled by their per-acidity decade shift.

Primary results are reported on `FROZEN` so they remain comparable to gen7; the
other two are a sensitivity check, and any headline that moves between them is
reported as unstable.
