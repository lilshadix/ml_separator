# gen11 §7 — metal representation audit for 40 metals

Everything below was recomputed by
`python -m lanthanide_separation.gen11.metalrep` from
`dataset_all_metals/clean/master_clean.parquet` (16,770 records, 40 named metals
plus 1,918 records with no resolved metal) and the frozen gen10 cohort
(5,248 rows, fingerprint `bed178ec1a7a82b0`). No number here is quoted from memory.

Scopes used throughout:

| scope | rows | what it is |
|---|---|---|
| `archive` | 16,770 | every archive record |
| `auxA` | 5,438 | auxiliary candidates (not a frozen bundle row) with `model_readiness == A_model_ready`, 30 metals |
| frozen bundle | 5,992 | the records the gen10 cohort was averaged from, 14 lanthanides |

## 0. Headline

**The frozen METAL block cannot describe 5,353 of the 5,438 auxiliary rows (98.44 %),
and gen10's pipeline does not fail on them — it renames them all europium.**

`build_dataset.py:643` attaches the block with
`df.merge(metal_features, on="metal_symbol", how="left", validate="many_to_one")`
against a 14-entry lanthanide dictionary. Any other metal left-merges to three NaNs.
Those NaNs reach `SimpleImputer(strategy="median", add_indicator=True)`
(`gen10/architectures.py:162`). Fitted on the frozen cohort the imputer's statistics are

    Atomic Number_metal = 63.0    lanthanide_index = 7.0    Ionic Radius_metal = 1.066

which is exactly Eu(III), and because `add_indicator` defaults to
`features="missing-only"` and the lanthanide-only fitting fold has **0 missing METAL
cells**, the transform emits **0 indicator columns**. Measured, not argued:
`representations.json → imputation_damage`.

So a naive reuse does not drop rows — it does something worse. 1,841 Am rows, 995 U
rows, 761 Th rows, 709 Pu rows and 432 Np rows would enter training labelled
Z = 63, `lanthanide_index` = 7, r = 1.066 Å, with nothing marking the substitution.
`lanthanide_index` is the sharpest case: it is defined as Z − 56, so the imputed value
asserts "element 63" on a row whose true atomic number is 95.

Over the whole archive the 14-key dictionary is undefined for **8,718 / 16,770 rows
(51.99 %)**.

## 1. Per-metal coverage — `metal_coverage.csv`

41 rows (40 metals + `<unknown>`), both scopes side by side. Columns: `atomic_number`,
`metal_category`, `is_lanthanide`, `is_actinide`, `in_frozen_dict`, `period`, `group`,
`block`, then for each scope `_rows`, `_atomic_number_n`, `_oxidation_state_n`,
`_ionic_radius_n`, `_lanthanide_index_n`, `_oxidation_states`, `_ox_state_sources`,
`_ox_state_implausible_rows`, `_ionic_radius_status`, plus the derived
`effective_charges` and `f_electron_counts` the GENERAL scheme actually feeds a model.

Structural facts:

* `lanthanide_index` is populated for exactly the 8,053 lanthanide records and nothing
  else. In `auxA` it covers **86 / 5,438 rows (1.58 %)**.
* `atomic_number` is present for all 14,852 records with a resolved metal; absent only
  for the 1,918 `<unknown>` records (all `C_not_modelable`, none in `auxA`).
* `metal_oxidation_state` comes from `metal_oxidation_state_source`: `archive` (13,468),
  `species_definition` (16, the uranyl expansion), null (3,286).
* **Nine of the forty metals have no tabulated CN=8 ionic radius at any oxidation
  state**: Cf, Cm, Cr, Fe, Mo, Pd, Ru, Sc, Tc — 277 archive rows, **237 `auxA` rows**.
  This reproduces the brief's "missing for 9 of the 40".
* **Five metals have no recorded oxidation state at all**: Cr, Fe, Mo, Ru, Sc —
  50 archive rows, 50 `auxA` rows.
* 26 metals have *partial* radius coverage: the archive refuses a radius when the
  oxidation state is unrecorded, so coverage is per-row, not per-metal.

### Cross-check against the archive's own report — `metal_coverage_crosscheck.csv`

Merged 1:1 against `dataset_all_metals/reports/metal_coverage.csv` on `metal_symbol`:
**41 / 41 rows matched (`_merge == both` for all)**, and

* `archive_rows == records` for all 41 → `crosscheck_all_row_counts_match: true`
* `archive_ionic_radius_n == rows_with_ionic_radius` for all 41 → `true`
* atomic numbers agree for all 41 → `true`

No disagreement to report.

## 2. Row loss by feature — `row_loss_by_feature.csv`

`rows_missing` per feature per scheme per scope, plus two per-scheme summary rows
(`<ALL informative missing>` = the row has no metal representation at all;
`<ANY informative missing>` = at least one informative feature is NaN).

`auxA` (5,438 rows):

| scheme | feature | rows missing | frac |
|---|---|---:|---:|
| FROZEN_3 | `Atomic Number_metal` | 5,353 | 0.9844 |
| FROZEN_3 | `lanthanide_index` | 5,353 | 0.9844 |
| FROZEN_3 | `Ionic Radius_metal` | 5,353 | 0.9844 |
| FROZEN_3 | **ALL informative missing** | **5,353** | **0.9844** |
| FROZEN_3_INDICATED | same three | 5,353 | 0.9844 |
| GENERAL | `metal_atomic_number` | 0 | 0.0000 |
| GENERAL | `metal_oxidation_state` / `metal_effective_charge` / `metal_is_oxo_species` | 552 | 0.1015 |
| GENERAL | `metal_ionic_radius_cn8_A` | 1,183 | 0.2175 |
| GENERAL | `metal_f_electron_count` | 363 | 0.0668 |
| GENERAL | period / group / block / is_lanthanide / is_actinide | 0 | 0.0000 |
| GENERAL | **ALL informative missing** | **0** | **0.0000** |
| GENERAL | ANY informative missing | 1,184 | 0.2177 |
| NO_RADIUS | **ALL informative missing** | **0** | **0.0000** |
| NO_RADIUS | ANY informative missing | 552 | 0.1015 |

`archive` (16,770 rows): FROZEN_3 loses **8,718 (51.99 %)**; GENERAL loses everything
only on the 1,918 `<unknown>` rows (11.44 %), with radius missing on 3,306 (19.71 %),
oxidation state on 3,286 (19.59 %) and f-count on 3,094 (18.45 %).

`metal_f_electron_count` is missing on fewer rows than `metal_oxidation_state`
(363 vs 552 in `auxA`) because the count is deterministic without an oxidation state
for non-f-block metals; the 189-row difference is Sr(86), Y(19), Ca(12), Fe(37),
Zr(7), Ba(7), Bi(4), Pb(4), Sc(10), Cr/Mo/Ru(1 each).

**No scheme drops a row.** Missingness is carried as NaN plus an explicit indicator
column, which is what the brief requires instead of silent loss; the indicator columns
themselves are never missing (0 NaN in every scope).

## 3. Do the archive's lanthanide values match `LANTHANIDE_DESCRIPTORS`? — `frozen_dict_agreement.csv`

**Values: identical. Coverage: not identical.**

For all 14 dictionary lanthanides, over all 8,052 of their archive records:

    max |atomic number delta|    = 0.000   (all 14 metals)
    max |lanthanide index delta| = 0.000   (all 14 metals)
    max |ionic radius delta|     = 0.000   (all 14 metals)

Each metal takes exactly one distinct radius in the archive and it equals the
dictionary entry to full float precision (La 1.16, Ce 1.143, Pr 1.126, Nd 1.109,
Sm 1.079, Eu 1.066, Gd 1.053, Tb 1.04, Dy 1.027, Ho 1.015, Er 1.004, Tm 0.994,
Yb 0.985, Lu 0.977). `frozen_dict_all_values_agree: true`.

The catch is coverage. The archive assigns a radius *per row*, from the recorded
oxidation state, and refuses one when the state is unrecorded. gen10 assigns a radius
*per metal*, unconditionally, which is a silent Ln(III) assumption. Consequence:

* all 14 lanthanides have archive rows with a NaN radius —
  `frozen_dict_metals_with_missing_archive_radius: 14`;
* **800 of the 8,053 lanthanide archive records** have no archive radius;
* **797 of the 5,992 frozen bundle rows (13.30 %)** would gain a NaN radius, and a NaN
  f-electron count, if the archive column were substituted verbatim. Worst offenders:
  Nd 251, Eu 215, La 86, Dy 46, Sm 40, Gd 37.

**Decision this supports:** archive metal columns *may* be used directly for lanthanide
rows for `atomic_number` and `lanthanide_index` (0 missing, 0 delta), and *may not* be
used verbatim for `ionic_radius_cn8_A` without either accepting 797 new NaNs or making
the trivalent assumption explicit. GENERAL takes the second option and flags it
(§4). A sanity check confirms the direction is exact:
`verify_frozen_reproduction` rebuilds gen10's own three columns from the copied
dictionary over all 5,248 cohort rows with **max |delta| 0.0 and 0 NaN-pattern
mismatches**.

## 4. The four pre-registered representations — `representations.json`, `metalrep.py`

Each is a pure `build(records) -> DataFrame`, index-preserving, with a declared column
list checked on every call by `build(records, scheme)`, and a
`assert_no_target_leakage` guard that rejects any column whose name touches
`log_d`, `d_value`, `target`, `doi`, `source_record`, `exp_id`, `series_id` or
`duplicate_group`. None reads the target.

| scheme | cols | informative | indicators | `auxA` rows with no representation | with partial |
|---|---:|---:|---:|---:|---:|
| `FROZEN_3` | 3 | 3 | 0 | **5,353** | 0 |
| `FROZEN_3_INDICATED` | 7 | 3 | 4 | **5,353** | 0 |
| `GENERAL` | 19 | 13 | 5* | **0** | 1,184 |
| `NO_RADIUS` | 16 | 13 | 3 | **0** | 552 |

\* GENERAL's non-informative columns are 4 missingness indicators plus
`metal_ionic_radius_is_assumed_trivalent`, which is a provenance flag, not a feature.

`FROZEN_3` — `Atomic Number_metal`, `lanthanide_index`, `Ionic Radius_metal`,
reproduced as the literal 14-key left merge, so a Pm row (a lanthanide the builder
never saw) also comes out all-NaN. This is the control, and it is exact against gen10.

`FROZEN_3_INDICATED` — the same three plus `metal_atomic_number_is_missing`,
`metal_lanthanide_index_is_missing`, `metal_ionic_radius_is_missing`,
`metal_not_in_frozen_dict`. Under the dictionary merge all four fire together, so they
are currently collinear; they exist because gen10's `add_indicator="missing-only"`
produces none of them when the fitting fold is lanthanide-only, and the difference
between FROZEN_3 and FROZEN_3_INDICATED is exactly the price of that silence.

`GENERAL` — `metal_atomic_number`, `metal_oxidation_state`, `metal_effective_charge`,
`metal_is_oxo_species`, `metal_ionic_radius_cn8_A`, `metal_f_electron_count`,
`metal_is_lanthanide`, `metal_is_actinide`, `metal_period`, `metal_group`,
`metal_block_is_{s,d,p,f}`, plus the five flags. **No lanthanide-specific coordinate**:
`lanthanide_index` is Z − 56 and is meaningless off the 4f series, and the contraction
information it carried is already in the radius and the f-count, both of which extend
to the 5f series by the same physics.

`NO_RADIUS` — GENERAL minus `metal_ionic_radius_cn8_A`,
`metal_ionic_radius_is_missing`, `metal_ionic_radius_is_assumed_trivalent`. This is the
§7 with/without comparison, and it matters structurally: the radius is the one metal
feature that is absent for *whole metals* (9 of 40, 237 `auxA` rows), so a
radius-dependent model would place Cf, Cm, Cr, Fe, Mo, Pd, Ru, Sc and Tc on top of
each other at the imputed median.

### Documented deterministic rules (the only inference anywhere in the module)

No oxidation state is ever invented. A NaN oxidation state yields NaN charge, NaN
oxo flag and NaN f-count on that row. Three rules operate on states the archive *did*
record, plus one radius fallback:

1. **f-electron count**, arithmetic on the noble-gas core:
   Ln (57 ≤ Z ≤ 71) `f = Z − 54 − n`; An (89 ≤ Z ≤ 103) `f = Z − 86 − n`;
   `f = 0` for Z < 57; `f = 14` for 71 < Z < 89 (filled 4f core: Hf, Pb, Bi).
   Checks: La(III)→0, Lu(III)→14, Th(IV)→0, U(VI)→0, U(IV)→2, Am(III)→6, Cm(III)→7,
   Cf(III)→9. Counts are clipped to [0, 14]; **0 rows in either scope are actually
   clipped**, so the rule is never repairing an impossible state behind your back.
2. **Effective charge** = oxidation state, except for the dioxo/oxo-anion species that
   dominate this chemistry: U/Np/Pu/Am(V) → +1 (AnO₂⁺), U/Np/Pu/Am(VI) → +2 (AnO₂²⁺),
   Tc(VII) → −1 (TcO₄⁻, an anion, not a 7+ cation). **1,314 `auxA` rows** are flagged
   `metal_is_oxo_species = 1`. Resulting charge set in `auxA`: {−1, 1, 2, 3, 4, 5}.
3. **Trivalent radius fallback**, lanthanides only: a lanthanide row whose oxidation
   state is unrecorded takes the Shannon CN=8 Ln(III) radius, flagged by
   `metal_ionic_radius_is_assumed_trivalent`. This fires on **797 of the 5,992 frozen
   bundle rows and 1 `auxA` row (Nd)**, and after it GENERAL's radius coverage on the
   frozen bundle is complete (0 NaN). It is exactly the assumption gen10 makes
   silently for every bundle row; here it is a switchable column.

### Where these rules are *not* externally deterministic — stated, not hidden

* The trivalent fallback is genuinely deterministic for La, Pr, Nd, Pm, Gd, Tb, Dy, Ho,
  Er, Tm and Lu, which have no other accessible aqueous state. It is an **assumption**
  for Ce (IV), Sm (II), Eu (II) and Yb (II). Of the 797 frozen bundle rows it fires on,
  **290 are in that unsafe subset** (Eu 215, Sm 40, Ce 21, Yb 14); the other 507 are
  safe (Nd 251, La 86, Dy 46, Gd 37, Er 25, Ho 21, Pr 21, Lu 12, Tm 4, Tb 4). The
  indicator column exists so an ablation can delete all 797 or just the 290.
* **Pa(V) is deliberately excluded from the oxo rule**: protactinium(V) hydrolyses to a
  mixture rather than a clean PaO₂⁺, so its 15 `auxA` rows keep charge +5. If that
  matters, it is 15 rows and it is flagged here rather than fixed quietly.
* The f-block is assigned `group = 3` and `block = "f"` for the whole series including
  La and Th, which have formally d ground states. This is a labelling convention.

## 5. Metals with no usable representation — `unusable_metals.csv`

Per scheme × metal over `auxA`: `rows_no_representation` (every informative feature
NaN) and `rows_partial_representation`.

**`FROZEN_3` and `FROZEN_3_INDICATED` — 26 of the 30 `auxA` metals have zero
representation, 5,353 rows:**

Am 1841, U 995, Th 761, Pu 709, Np 432, Sr 180, Cm 136, Y 47, Fe 37, Ca 34, Ba 32,
Pd 23, Bi 21, Tc 21, Zr 20, Pa 15, Pb 14, Sc 10, Cf 7, Hf 5, In 5, Cd 4, Cr 1, Mo 1,
Pm 1, Ru 1.
Only Eu (78), Nd (5), Gd (1) and Sm (1) survive — 85 rows. Note Pm is in the dead
list: it is a lanthanide, but the builder had no promethium rows so the dictionary
omits it, and the left merge kills it exactly like an actinide.

**`GENERAL` — 0 metals and 0 rows with no representation.** 1,184 rows are partial:
1,183 lack a radius (Am 289, Np 250, Pu 193, Cm 136, Sr 86, U 46, Fe 37, Th 29, Pd 23,
Tc 21, Y 19, Ca 12, Sc 10, Zr 7, Ba 7, Cf 7, Bi 4, Pb 4, Cr 1, Mo 1, Ru 1) and 552
lack an oxidation state.

**`NO_RADIUS` — 0 metals and 0 rows with no representation**, 552 partial (missing
oxidation state only). Dropping the radius removes the last feature that is undefined
for a whole metal: under NO_RADIUS every one of the 30 `auxA` metals is fully placed
by Z, period, group, block, is_lanthanide and is_actinide regardless of curation state.

Over the full archive, the 1,918 `<unknown>`-metal records have no representation under
any scheme; all are `C_not_modelable` and none is an auxiliary candidate.

## 6. Data-quality findings this audit turned up

* **Sr(III), 38 archive rows.** The archive records oxidation state 3 for strontium,
  which does not exist. All 38 are `B_usable_with_caveats`, so **0 reach `auxA`** and
  GENERAL never emits a +3 strontium today — but any arm that relaxes the readiness
  tier would. Flagged in `metal_coverage.csv → archive_ox_state_implausible_rows`.
* **Cm is the highest-value curation target.** 136 `auxA` rows, third-largest actinide
  block outside Am/U/Th/Pu/Np, and `ionic_radius_status = requires_curation` with
  **zero** radius rows — yet Cm(III) is chemically a near-twin of Am(III) and Gd(III),
  the exact analogy gen11 exists to test. It is currently radius-blind.
* Np has recorded oxidation states for all 432 `auxA` rows but a radius for only 182
  (250 `requires_curation`), so the An(V)/An(VI) speciation is known and the size is not.

## 7. Files

| file | rows | contents |
|---|---:|---|
| `metal_coverage.csv` | 41 | per-metal coverage, archive and `auxA`, incl. charges and f-counts |
| `metal_coverage_crosscheck.csv` | 41 | merge against the archive's own `reports/metal_coverage.csv` |
| `frozen_dict_agreement.csv` | 14 | value deltas and coverage gaps vs `LANTHANIDE_DESCRIPTORS` |
| `row_loss_by_feature.csv` | 106 | `rows_missing` per feature × scheme × scope + summaries |
| `unusable_metals.csv` | 120 | per scheme × metal, rows with no / partial representation |
| `representations.json` | — | column lists, per-scheme row counts, imputation damage, all check flags |
| `../../../src/lanthanide_separation/gen11/metalrep.py` | — | the four builders and this audit |

Reproduce with:

    PYTHONPATH=src .venv/bin/python -m lanthanide_separation.gen11.metalrep

## 8. Checks and their status

| check | result |
|---|---|
| `FROZEN_3` reproduces the gen10 METAL block over 5,248 cohort rows | **PASS** — max delta 0.0, 0 NaN mismatches |
| archive lanthanide values == `LANTHANIDE_DESCRIPTORS` (Z, index, radius) | **PASS** — all deltas 0.000 |
| recomputed per-metal row counts == archive's published report | **PASS** — 41/41 |
| recomputed radius-row counts == archive's published report | **PASS** — 41/41 |
| atomic numbers == archive's published report | **PASS** — 41/41 |
| declared column list == built column list, all 4 schemes | **PASS** |
| no target/provenance column in any representation | **PASS** |
| f-count rule never clips an out-of-range value | **PASS** — 0 rows, both scopes |
| no scheme drops a row | **PASS** — all schemes return len(records) rows |
| archive radius coverage == gen10 radius coverage on lanthanide rows | **FAIL (expected)** — 797/5,992 frozen rows and 800/8,053 lanthanide archive rows have no archive radius; GENERAL closes this with the flagged trivalent fallback, `FROZEN_3` closes it by assuming Ln(III) silently |
| every `auxA` metal has a usable representation under `FROZEN_3` | **FAIL** — 26 of 30 metals, 5,353 of 5,438 rows have none |

## 9. Recommendation for the pre-registration

Use `GENERAL` for any arm that trains on auxiliary rows, and carry `FROZEN_3` only as
the control that reproduces gen10. `FROZEN_3` is not a viable auxiliary representation
under any reading: it describes 1.56 % of the auxiliary rows and median-imputes the
other 98.44 % into europium with no indicator. Report `NO_RADIUS` beside `GENERAL` in
every arm, because the radius is the only metal feature whose absence is metal-shaped
rather than row-shaped, and 237 auxiliary rows across 9 metals depend on the answer.
