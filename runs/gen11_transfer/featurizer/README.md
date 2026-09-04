# gen11 auxiliary featurizer

`lanthanide_separation.gen11.auxfeatures` renders multi-metal archive records into the
frozen gen10 arm `METAL+COND+ECFP+MASSACTION+RECOVERED` — 2146 columns — so an auxiliary row and a training row are the same
kind of object. Every number below was computed by the code that wrote this file.

## The proof: does the featurizer reproduce the frozen bundle?

Applied to the 5992 archive records that *are* the frozen
bundle rows (joined 1:1 on `exp_id`), re-derived from the archive's `*_raw` text:

| family | columns | cells | mismatching cells | mismatching columns |
|---|---|---|---|---|
| ECFP | 2048 | 12,271,616 | 0 | 0 |
| COND_numeric | 5 | 29,960 | 0 | 0 |
| COND_acid | 9 | 53,928 | 0 | 0 |
| COND_diluent | 41 | 245,672 | 0 | 0 |
| COND_additive | 9 | 53,928 | 0 | 0 |
| METAL | 3 | 17,976 | 0 | 0 |
| **total (bundle-resident)** | **2115** | **12,673,080** | **0** | **0** |

**Row-level reproduction is exact: 0 mismatching cells out of 12,673,080.**
The 2,048 ECFP bits, all 64 `cond__*` columns (9 acid + 41 diluent + 9 additive + 5
numeric) and the 3 METAL columns are bit-for-bit identical to the frozen bundle's own.

## MASSACTION and RECOVERED: checked at the cell, because that is where they live

Neither block exists in the bundle. MASSACTION is attached by `levels._attach_mass_action`
before replicate collapse (so the cohort carries the *first* row of each cell); RECOVERED
is joined afterwards from `recovered_cell_table`, which takes the *cell mean*. Both
reductions are replayed here before comparing.

* cohort rows 5248, reproduced cells 5248, matched 5248, unmatched 0
* **RECOVERED: 0 mismatching cells** over 120,704 (23 columns) — exact.
* **MASSACTION: 62 mismatching cells** over 41,984, in 2 columns.

### The MASSACTION residual, diagnosed

`DENTATE` and `coreCN` are per-*row* geometry-plan outputs, not per-ligand constants: the
builder derived them from (metal Z, ligand SMILES, acid), so the same ligand can carry
two denticities in the same cohort. The archive does not carry them at all, so they are
looked up by mode. The residual is exactly the drift that lookup introduces:

* cells where the modal `DENTATE` differs from the row's own: 33 (= mismatches in `massact__logL_x_DENTATE`)
* cells where the modal `coreCN` differs from the row's own: 29 (= mismatches in `massact__logL_x_coreCN`)

The two counts account for every mismatching cell. The five `massact__log10_*` columns and
`massact__logL_x_logH`, which do not touch the annotations, reproduce exactly.

## RECOVERED against gen7's caches

* `recovered_cells.parquet` (what the cohort actually joins): 0 mismatching cells over 5248 cells x 23 columns — exact.
* `recovered_raw.parquet` (row level): 672 mismatching cells over 5992 rows x 20 columns, in 8 solvent columns. **This cache is stale**: it predates the `CH3Cl -> chloroform` correction in `gen7/recovered.py` and still encodes those 84 rows as dichloromethane. `recovered_cells.parquet`, the cohort and this featurizer all agree; only the row-level cache does not. Do not use it.

## What the auxiliary rows look like in this space

`aux_features.parquet`: 5438 A_model_ready archive records x 2146 features + 11 identity columns (`source_record_id`, `metal_symbol`, `extractant_primary_smiles`, `series_id`, `duplicate_group_id`, `doi_primary_corrected`, `log_D`, `metal_category`, `metal_oxidation_state`, `model_readiness`, `extractant`).
Feature fingerprint (sha1 of the matrix, hashlib not `hash()`): `1372dc260cffb84c`.
155 distinct structures, 0 RDKit parse failures.

What does *not* fully carry over, with counts:

* **METAL** — 85 rows resolve from the builder's own
  14-lanthanide table; 5353 take the archive fallback
  (26 symbols); 0 unresolved.
  `lanthanide_index` is NaN in 5352 rows (undefined
  off the lanthanide series) and `Ionic Radius_metal` in 237 rows.
* **COND** — 836 rows carry a diluent outside the frozen
  vocabulary and go to `cond__diluent__other` (the builder's own OTHER bucket);
  19 rows carry an additive with no frozen level and no
  `other` column, so they become an all-zero additive row — indistinguishable from
  'no additive recorded'. Acid: 0 unrepresentable rows.
* **MASSACTION** — `massact__logL_x_DENTATE` is NaN in 2211 rows
  (83 structures the frozen cohort has never seen) and
  `massact__logL_x_coreCN` in 5353 rows (metals outside the
  lanthanide series have no coordination-number annotation). The frozen arm imputes with
  `add_indicator=True`, so NaN is a representable state, not a silent zero.
* **RECOVERED** — 474 rows have a diluent gen7's component
  patterns do not cover (EXXsol D60, hyfrane, mesitylene, xylene), leaving all 15
  `rec__solvent_*` columns NaN and `rec__solvent_parsed` 0.
  3273 rows are judged against the frozen modal extractant
  name; 2165 (new structures) against the archive's
  own rows, which is a different denominator for `rec__n_names_for_structure`.

## Encoding divergences we deliberately did not fix

The archive corrects several chemical mistakes the bundle builder made. Adopting those
corrections here would encode auxiliary rows on a different vocabulary than the training
rows, which is the exact failure this module exists to prevent — so they are reproduced
as-is and recorded in `known_encoding_divergences.csv` for a later sensitivity arm:

| divergence | family | aux rows | sensitivity arm |
|---|---|---|---|
| tph_identity | COND/diluent | 466 | merge the five levels into one aliphatic-cut indicator |
| tph_identity_recovered | RECOVERED/solvent | 317 | none needed for aux; recorded because it explains a COND/RECOVERED inconsistency a reader will otherwise take for a bug |
| iupac_locant_comma | COND/diluent + RECOVERED/solvent | 136 | re-encode the diluent from the archive's adjudicated solvent_key |
| ch3cl_isomer | COND/diluent | 156 | collapse the three chloroform levels into one |
| ionic_radius_oxidation_state | METAL | 15 | per-row oxidation-state-resolved ionic_radius_cn8_A |
| unseen_diluent_level | COND/diluent | 836 | extend the vocabulary and refit the frozen arm (breaks the control) |
| unseen_additive_level | COND/additive | 19 | extend the vocabulary and refit the frozen arm (breaks the control) |
| unparsed_solvent_recovered | RECOVERED/solvent | 474 | add the missing components to SOLVENT_COMPONENTS |
| metal_descriptor_table_fallback | METAL | 5353 | metal_policy='table_only' (leaves the whole METAL block NaN off-Ln) |
| geometry_annotation_ambiguity | MASSACTION | 2211 | run the builder's plan_complex for the new structures |

## Files

* `reproduction.json` — per-column-family mismatch counts, both levels, plus the cache
  check and the annotation-drift diagnosis.
* `mismatches.csv` — every non-reproducing column with its count and a diagnosis.
* `known_encoding_divergences.csv` — the table above, with full detail text.
* `aux_features.parquet` — the featurized auxiliary rows.
* `aux_report.json` — the per-block diagnostic for the auxiliary run.

## Reproduce

```python
from lanthanide_separation.gen10.runner import prepared_cohort
from lanthanide_separation.gen11 import auxfeatures
auxfeatures.write_outputs(prepared_cohort())
```
