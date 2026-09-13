# Gen13 — separation-first: zero-shot lanthanide separation factors from the Ln-axis curve

**Read `DATA_AUDIT.md`, then `PRE_REGISTRATION.md`, then `DECISION_REPORT.md`.**  The audit
was written before the pre-registration and the pre-registration before any locked run.

Gen5–gen12.2 established that the per-extractant *level* of `log D` is the bottleneck and is
not recoverable from ~150 ligands, while the within-extractant *shape* is.  Separation —
`log SF(A/B) = log D(A) − log D(B)` for two lanthanides under one extractant and one
condition set — cancels the level exactly.  Gen13 models the centred lanthanide-axis curve of
every multi-metal cell with a low-rank or physics basis over the 14 lanthanides, predicts the
basis coefficients from extractant structure and conditions, and scores every unordered
metal pair of every chemically unseen cell.

## Layout

```
gen13sep/        paths, metals (Shannon radii, hydration energies, tetrad functions), cohort,
                 features, splits, basis, models (arms), metrics, inference, fewshot, runner
scripts/         g13_run_ladder.py, g13_analysis.py, g13_fewshot.py, g13_headline_tables.py
tests/           executable invariants (leakage, target-free features, antisymmetry, selection)
manifests/       cohort parquet + audit json (exact and relaxed keys)
predictions/     per-label pair tables, curves, bases, fold plan, similarity, selections
metrics/ bootstrap/ headline_tables/ analysis/
```

## Reproducing

```bash
.venv/Scripts/python.exe -m pytest generations/gen13_separation/tests -q
.venv/Scripts/python.exe generations/gen13_separation/scripts/g13_run_ladder.py --label B_primary --with-exploratory
.venv/Scripts/python.exe generations/gen13_separation/scripts/g13_analysis.py --label B_primary \
    --contrasts "M_SELECTED:C_DIRECT_ROW,M_SELECTED:B1_MEAN_CURVE,M_SELECTED:B3_NN_TANIMOTO,M_SELECTED:B2_PAIRMEAN"
.venv/Scripts/python.exe generations/gen13_separation/scripts/g13_fewshot.py --label B_primary
.venv/Scripts/python.exe generations/gen13_separation/scripts/g13_headline_tables.py --primary B_primary --ablations ...
```

Inputs are read-only: the frozen bundle (sha `fefbefc6…`), `runs/gen7_architecture/cache/chemistry_map.parquet`,
`runs/gen6_provenance/provenance_table.parquet`, the bundle's `ligand_2d_descriptors.parquet` and
`gen12_2_eu_pred/features/coordination_descriptors.parquet`.  Nothing under gen1–gen12.2 is modified.

## Predicting for a new extractant

```bash
.venv/Scripts/python.exe generations/gen13_separation/scripts/g13_predict.py fit
.venv/Scripts/python.exe generations/gen13_separation/scripts/g13_predict.py predict --input my_ligands.csv --output pred.csv
```

The fitted ensemble is `V2_BAG4@lean`: a direct row model plus physics-basis, rank-1 and rank-2
curves, all on conditions + physchem + donors + coordination.  Zero-shot macro MAE 0.468 on unseen
chemotypes, **0.536 when the held-out chemistry's laboratories are also removed from training**
(the honest number for new chemistry from a new group), about 0.42 on near analogues.  ECFP, the
206 RDKit descriptors and a conditions-only member are deliberately excluded because they do not
transfer across laboratories — see `DECISION_REPORT.md` §5a and §9.
`my_ligands.csv` needs a `smiles` column and any of the bundle's `cond__*` columns (acid type
one-hots, `cond__acid_concentration_M`, `cond__extractant_concentration_M`, diluent one-hots,
`cond__temperature_C`; missing values are imputed).  Optional `measured_A`, `measured_B`,
`measured_logSF` calibrate the curve with one measured pair through the conditional (BLUP)
adapter of `gen13sep/fewshot_stage2.py` and the residual covariance frozen beside the model —
measure the widest pair you can (the model suggests La–Lu, then Ce–Tm).  The output holds the
centred 14-lanthanide curve, every pairwise `log SF(A/B)` (`*_pairs.csv`), `max_train_tanimoto`
/ `band` as a distance-to-training warning, and `suggested_first_pair`.

## Selection convention

**The deployed model is selected under the publication-masked design (BP), never under design B.**
The 64 condition columns identify a cell's publication with 94 % 1-NN accuracy, so an arm can win
under design B by reading the laboratory rather than the chemistry (`DECISION_REPORT.md` §5a,
`PRE_REGISTRATION.md` Addendum 2).  Design B remains the comparison to gen2–gen12 and the
near-analogue regime.

## Three things to know before quoting a number

1. **Every number needs its regime**: {all pairs | adjacent (`dZ = 1` plus Nd–Sm) | far `dZ ≥ 5`} ×
   {extractant-macro | chemotype-macro | pooled} × {zero-shot | one-pair}.  Adjacent
   separation factors sit at the replicate noise floor (median replicate sd 0.30 log D per
   (cell, metal) on the frozen cohort).
2. **The "heavier always preferred" rule scores 0.630 sign accuracy extractant-macro (0.855
   pooled) on strong contrasts.**  Sign accuracy is not evidence unless it clears the macro
   number; MAE is the primary metric.
3. **One chemotype (the diglycolamides) holds 375 of 521 cells and 23 of 90 extractants.**
   When it is held out the model trains on 103 cells.  Intervals resample chemotypes.
