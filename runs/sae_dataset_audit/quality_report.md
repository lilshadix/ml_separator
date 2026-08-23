# Quality report — multi-metal Separation Archive for Elements build

`16,770` archive records distilled from `48,471` raw CSV rows across 16,770 distinct `exp_id` values. The frozen gen7–gen10 lanthanide pipeline was not modified, retrained or re-split.

## 1. Headline numbers

| quantity | value | meaning |
|---|---|---|
| raw CSV rows | 48,471 | 41 per-metal files, immutable in `raw/` |
| archive records | 16,770 | one row per `exp_id` |
| rows folded as export fan-out | 31,701 | same record re-exported once per metal file |
| canonical measurements | 14,866 | after collapsing class A/B duplicates only |
| model-ready (tier A) | 11,026 | all required fields, no unresolved conflict |
| usable with caveats (tier B) | 1,385 | complete but in a value conflict or carrying a suspect flag |
| not modelable (tier C) | 2,455 | a required field is missing |
| redundant duplicate (tier D) | 1,904 | non-representative member of an A/B group |
| unique metals | 40 | distinct elements; the archive's 41 raw tokens include `UO2+2`, resolved to U(VI) |
| unique extractant systems | 262 | order-invariant canonical-structure keys |
| unique diluent systems | 83 | after mixture parsing |
| usable curves | 2,331 | ≥3 points on one axis |

## 2. Why 48,471 rows are only 16,770 measurements

The archive was exported one CSV per **queried** metal, not per **measured** metal. A record whose measured metal is Nd appears — byte-identically — in `Am.csv`, `Ba.csv`, `Ca.csv`, `In.csv`, `La.csv`, `Mo.csv`, `Nd.csv`, `Pa.csv`, `Pr.csv`, `Th.csv`, `U.csv` and `Y.csv`, because all twelve metals were present in that experimental system.

Before collapsing this, all 37 raw columns were checked for variation inside each `exp_id` group: **none varied**, in 0 of 16,770 groups. The fan-out is therefore pure export metadata and collapsing it loses no science. Every contributing raw row is retained in `raw_row_ids` and in `raw_to_canonical_mapping.parquet`.

**The file name is not the metal.** Any pipeline that infers the metal from the source file will mislabel a large fraction of this archive.

| copies of one record | records |
|---|---|
| 1 | 1,935 |
| 2 | 5,535 |
| 3 | 4,969 |
| 4 | 2,928 |
| 5 | 660 |
| 6 | 205 |
| 7 | 186 |
| 8 | 194 |
| 9 | 125 |
| 10 | 28 |
| 11 | 2 |
| 12 | 3 |

## 3. Hierarchical duplicate detection

| level | rule | groups | redundant rows | population |
|---|---|---|---|---|
| L1 | exact_raw_duplicate_rows | 14,835 | 31,701 | 48,471 raw rows |
| L2 | whitespace_case_normalised | 918 | 2,394 | 16,770 records |
| L3 | canonical_chemical_names | 918 | 2,394 | 16,770 records |
| L4 | canonical_structures | 920 | 2,396 | 16,770 records |
| L5 | scientific_identity_exact_numerics | 920 | 2,396 | 16,770 records |
| L6 | scientific_identity_sig6 | 920 | 2,396 | 16,770 records |
| L7 | same_conditions_different_value | 334 | 0 | 16,770 records |

Levels 2–6 describe the same 21 variables and differ only in how canonical the representation is, so each level can only merge rows. The pipeline asserts this monotonicity and fails rather than reporting a ladder where a normalisation step *creates* distinctions.

**The 21 variables include the measured value.** These levels answer *"is this the same recorded row?"*, so two records that agree on every condition but disagree on log D are correctly NOT merged here. The tolerance sweep in §4 asks a different question — it keys on conditions alone and reports value disagreement separately — which is why its group counts differ from this table at the same `sig6` level.

Normalisation beyond the export fan-out is worth only 2 additional redundant rows: this archive's text is already clean, and essentially all duplication is the per-metal fan-out.

### Duplicate group classes

| class | identity groups | of those, groups with >1 member | records | treatment |
|---|---|---|---|---|
| UNIQUE | 10,704 | 0 | 10,704 | singleton scientific identity |
| A_EXACT_DATABASE_DUPLICATE | 716 | 716 | 2,609 | same conditions, same value, same provenance — collapsed |
| B_SAME_MEASUREMENT_DIFF_PROV | 9 | 9 | 20 | same conditions and value under different provenance — collapsed, all citations kept |
| C_POSSIBLE_INDEPENDENT_REPLICATE | 10 | 10 | 20 | value agrees only to low precision under different provenance — **kept separate** |
| E_VALUE_CONFLICT | 307 | 307 | 1,040 | identical conditions, different log D — **all values kept, never averaged** |
| F_INSUFFICIENT_INFORMATION | 2,082 | 86 | 2,377 | a field needed for the decision is missing — **kept, flagged** |

The middle column matters: an *identity group* is one scientific identity, which may have a single member. Most `F_INSUFFICIENT_INFORMATION` groups are singletons — records whose missing fields prevent any duplicate judgement, not records duplicated by the archive. Only the A/B rows represent redundancy that was actually collapsed.

**D_CONDITION_CONFLICT** is reported separately in `condition_conflict_candidates.parquet`: 192 groups report an *identical* log D while disagreeing about exactly one condition. A differing condition with a differing value is an ordinary titration point, not a conflict, so only the same-value case is flagged.

## 4. Numeric tolerance sensitivity

| level | sig. digits | identity groups | duplicate groups | records in duplicates | value conflicts |
|---|---|---|---|---|---|
| exact | exact | 13,828 | 1,128 | 4,070 | 334 |
| sig12 | 12 | 13,828 | 1,128 | 4,070 | 334 |
| sig9 | 9 | 13,828 | 1,128 | 4,070 | 334 |
| sig6 | 6 | 13,828 | 1,128 | 4,070 | 334 |
| sig4 | 4 | 13,817 | 1,139 | 4,092 | 324 |
| sig3 | 3 | 13,527 | 1,331 | 4,574 | 372 |

The default is `sig6` (6 significant figures, 1 ppm relative). Exact, 12-, 9- and 6-figure comparison give **identical** groupings, so the choice is not load-bearing: the archive's condition values carry no float-serialisation noise. Only at 3 significant figures does the grouping move materially, which is coarse enough to merge genuinely different titration points and is therefore rejected.

Per-field units, parsers and conversions are documented in `field_mapping.csv` and in `stage03_dedup_report.json` → `numeric_field_policy`.

## 5. Chemical systems and multi-component handling

| system class | records |
|---|---|
| SINGLE_EXTRACTANT | 14,206 |
| EXTRACTANT_PLUS_AQUEOUS_AGENT | 933 |
| EXTRACTANT_PLUS_MODIFIER | 797 |
| SYNERGISTIC_TWO_EXTRACTANT | 608 |
| NO_EXTRACTANT_RECORDED | 226 |

The gen10 audit found a measurable error penalty for representing a multi-component system by one component's structure. The frozen table does exactly that: its 18 `TODGA,DHOA` rows carry `canonical_smiles = TODGA` and DHOA is gone. Here every chemically active component is kept in the `components` list with its own role, structure and concentration, and `extractant_system_key` is the order-invariant join of the sorted canonical SMILES.

Component order in the archive is **not** stable — `2-bromodecanoic acid, N-DPP` appears with its two SMILES in both orders — which is why the identity key sorts components before hashing.

### Structure/name integrity findings

- **TODGA's SMILES is attached to 22 different extractant names.** One sub-source (`./ST*.json`, 1,104 records) records the *water-soluble masking agent's* name in `Extractant_Name` while `Extractant_SMILES` holds the organic extractant, and puts the agent's real structure in `comments_description → Holdback_Agent_SMILES`. Identity here is keyed on structure, never on name. *(The frozen pipeline hits the same bug and hard-codes a quarantine for it at `pairs.py:264`.)*
- **TBP and DHOA never carry a SMILES** in any of their 972 / 505 single-component records. Both were recovered *from the archive itself* by elimination: in two-component records such as `TODGA, TBP` every other name already has a consensus structure, so the remaining SMILES must be theirs. Recovered: `Br-Cosan`, `D2EHAA`, `DHOA`, `TBP`. Their components carry `structure_source = "deduced_by_elimination"`, a tag used by nothing else, so they can be excluded on their own. (`name_consensus` is the tag for ordinary archive-confirmed lookups and covers most of the table — filtering on it would remove almost everything.)
- 243 records have a name that disagrees with the structure the archive gives that name elsewhere; all are queued for review, none were auto-resolved.
- `Br-Cosan` resolves to `Cc1cc(Br)ccc1NC(=O)CCl`, a bromo-methyl chloroacetanilide. Br-Cosan is a cobalt bis(dicarbollide); the archive's structure is chemically unrelated. It is **preserved as recorded** and flagged, not silently corrected.

## 6. Metal coverage

8,053 lanthanide records, 6,799 non-lanthanide, 1,918 with no resolvable metal.

| metal | category | records | model-ready | extractant systems | log D min | log D max |
|---|---|---|---|---|---|---|
| Eu | lanthanide | 2,376 | 1,556 | 201 | -4.76 | 3.58 |
| Am | actinide | 2,104 | 1,841 | 130 | -4 | 4.21 |
| U | actinide | 1,609 | 995 | 54 | -3.08 | 3.16 |
| Pu | actinide | 1,019 | 709 | 55 | -3.81 | 3.89 |
| Th | actinide | 839 | 761 | 46 | -4 | 4.49 |
| Nd | lanthanide | 796 | 581 | 88 | -4.68 | 3.93 |
| La | lanthanide | 538 | 388 | 89 | -4.49 | 2.86 |
| Gd | lanthanide | 504 | 368 | 78 | -4.97 | 3.53 |
| Dy | lanthanide | 498 | 367 | 77 | -4.14 | 3.94 |
| Np | actinide | 496 | 432 | 23 | -4.06 | 2.71 |
| Pr | lanthanide | 481 | 330 | 81 | -12.5 | 2.79 |
| Sm | lanthanide | 472 | 325 | 81 | -4.99 | 3.94 |
| Ce | lanthanide | 416 | 280 | 72 | -4.46 | 3.07 |
| Er | lanthanide | 402 | 306 | 75 | -3.7 | 4.15 |
| Ho | lanthanide | 360 | 265 | 59 | -3.87 | 4.07 |
| Tb | lanthanide | 348 | 247 | 67 | -4.19 | 4.07 |
| Lu | lanthanide | 334 | 241 | 62 | -3.66 | 4.2 |
| Yb | lanthanide | 320 | 227 | 61 | -3.65 | 4.21 |
| Sr | alkaline_earth | 219 | 180 | 5 | -3.7 | 3.34 |
| Tm | lanthanide | 207 | 192 | 55 | -3.52 | 4.16 |

Full per-metal statistics are in `metal_coverage.csv`; the extractant × metal coverage matrix is in `extractant_metal_matrix.csv`.

### Ionic-radius availability

Shannon CN=8 values are supplied only for `(element, oxidation state)` pairs that are tabulated and confirmed. Everything else is left null with `ionic_radius_status` set, rather than filled with a plausible-looking guess. Two different gaps are involved and the coverage table separates them:

| situation | metals | which |
|---|---|---|
| **no radius on any row** — the ion is not tabulated at CN=8 | 3 | Cf, Pd, Tc |
| **no radius on any row** — the archive never recorded an oxidation state, so the lookup has no key (the value may well be tabulated) | 5 | Cr, Fe, Mo, Ru, Sc |
| **no radius on any row** — both causes present | 1 | Cm |
| radius on some rows only — oxidation state missing on the rest | 26 | Am, Ba, Bi, Ca, Ce, Dy, Er, Eu, Gd, Ho, La, Lu, Nd, Np, Pb, Pr, Pu, Sm, Sr, Tb, Th, Tm, U, Y, Yb, Zr |
| radius on every row | 5 |  |

At row level, 12,664 of the 14,852 records that have a resolved metal carry a radius (85%). A radius-dependent model would silently drop the other 2,188.

## 7. Missingness by scientifically meaningful field

| field | present | missing | missing % |
|---|---|---|---|
| `metal_symbol` | 14,852 | 1,918 | 11.4% |
| `metal_oxidation_state` | 13,484 | 3,286 | 19.6% |
| `extractant_primary_smiles` | 16,544 | 226 | 1.3% |
| `extractant_primary_concentration_M` | 16,536 | 234 | 1.4% |
| `acid_primary` | 16,769 | 1 | 0.0% |
| `acid_concentration_M` | 16,592 | 178 | 1.1% |
| `solvent_key` | 16,770 | 0 | 0.0% |
| `metal_concentration_M` | 7,412 | 9,358 | 55.8% |
| `temperature_C` | 15,641 | 1,129 | 6.7% |
| `contact_time_min` | 9,488 | 7,282 | 43.4% |
| `shaking_time_min` | 2,883 | 13,887 | 82.8% |
| `phase_ratio_org_aq` | 8,153 | 8,617 | 51.4% |
| `acid_concentration_organic_M` | 1,962 | 14,808 | 88.3% |
| `nitrate_concentration_M` | 1,104 | 15,666 | 93.4% |
| `modifier_name` | 797 | 15,973 | 95.2% |
| `modifier_concentration_M` | 792 | 15,978 | 95.3% |
| `complexant_smiles_canonical` | 991 | 15,779 | 94.1% |
| `holdback_smiles_canonical` | 327 | 16,443 | 98.1% |
| `log_D` | 14,622 | 2,148 | 12.8% |

`contact_time_min` and `metal_concentration_M` are the two fields that most often block a row from modelling — the same two that silently remove 2,976 of 5,992 rows from the frozen gen10 cohort through its `cond__` completeness gate. Any multi-metal experiment should treat them as missing-indicator features rather than as hard filters.

## 8. Condition ranges

| field | unit | n | min | median | max |
|---|---|---|---|---|---|
| `extractant_primary_concentration_M` | M | 16,536 | 0.00011 | 0.1 | 5.81 |
| `acid_concentration_M` | M | 16,592 | 2.63e-07 | 1.72 | 24 |
| `metal_concentration_M` | M | 7,412 | 1e-12 | 0.000247 | 2.18 |
| `temperature_C` | C | 15,641 | 4.85 | 25 | 75 |
| `contact_time_min` | min | 9,488 | 0.134 | 30 | 1.8e+03 |
| `shaking_time_min` | min | 2,883 | 2 | 32 | 120 |
| `phase_ratio_org_aq` | ratio | 8,153 | 0.5 | 1 | 3.5 |
| `nitrate_concentration_M` | M | 1,104 | 0.000891 | 0.525 | 5 |
| `log_D` | log10 D | 14,622 | -12.5 | 0.0414 | 4.49 |

## 9. Experimental series and curves

583 series and 2,331 usable curves (≥3 points), covering 13,446 of 16,770 records. Mean 6.3 points per curve, median 5, max 100.

| curve axis | usable curves |
|---|---|
| acid | 826 |
| extractant | 664 |
| metal_series | 575 |
| complexant_concentration | 86 |
| metal_concentration | 73 |
| temperature | 62 |
| contact_time | 35 |
| modifier_concentration | 10 |

Definitions mirror `gen8/series.py`: a *series* is one extractant system × one setting of every categorical condition; a *curve* is a maximal subset of a series in which exactly one axis varies. Concentration axes are read on the log10 scale because mass action is linear in log concentration.

**`log_D` is never read to define a series or a curve.** The test suite proves this by rebuilding membership from a frame whose target has been permuted and requiring byte-identical output.

## 10. Provenance coverage

| quantity | records |
|---|---|
| with a primary-literature DOI | 15,615 |
| with a non-DOI reference only | 1,155 |
| **with no reference of any kind** | 0 |
| carrying the archive self-citation `10.1021/jacs.5c19738` | 9,446 |
| with a publication title parsed from comments | 5,738 |
| with a figure/table location | 5,738 |

The archive cites itself on every record it exports. That DOI is tracked separately in `archive_citation_doi` and never fills `doi_primary`, because it references the dataset, not the measurement.

Every record carries *some* reference, but 1,155 have no DOI. Their references are resolvable documents rather than journal articles, so those rows are traceable but harder to verify against a published table:

| non-DOI reference | records |
|---|---|
| `https://cordis.europa.eu/project/id/211267` | 1,104 |
| `https://inis.iaea.org/collection/NCLCollectionStore/_Public/33/048/33048023.pdf?r=1` | 39 |
| `http://www.theses.fr/2019MONTS019` | 12 |

## 11. Manual-review queue

| priority | category | cases | records |
|---|---|---|---|
| 1 | implausible_extreme_D | 1 | 3 |
| 1 | implausible_oxidation_state | 1 | 38 |
| 1 | same_conditions_different_logD | 307 | 1,040 |
| 2 | identical_record_different_publication | 19 | 40 |
| 3 | same_value_conflicting_condition | 192 | 425 |
| 3 | structure_deduced_by_elimination | 4 | 1,781 |
| 4 | component_pairing_ambiguous | 36 | 364 |
| 4 | extractant_name_structure_conflict | 39 | 243 |
| 5 | diluent_name_ambiguous | 8 | 1,341 |
| 6 | doi_repaired_verify | 2 | 2,690 |
| 6 | metal_identity_missing | 1 | 1,918 |

`manual_review_queue.csv` holds 610 cases. Each row names the affected `canonical_measurement_id`s, explains exactly why the program could not decide, and carries the raw evidence side by side. No case has an invented resolution.

## 12. Row accounting

Every raw row leaves the pipeline as exactly one disposition; the ledger is asserted to balance and the stage fails rather than emitting a report that loses rows.

| disposition | records |
|---|---|
| canonical_measurement | 11,026 |
| canonical_with_caveats | 1,385 |
| redundant_duplicate_member | 1,904 |
| unresolved_or_manual_review | 2,455 |

Sum = 16,770 = 16,770 archive records. All 48,471 raw rows map to a record, and every record is reachable from at least one raw row.

## 13. Reproducibility

- RDKit `2026.03.5`; canonicalisation failures are preserved with their raw string and reported, never dropped.
- No wall-clock, RNG or hash-seed dependence. Group ids are assigned by sorted identity hash, and the canonical representative of a group is the lowest numeric archive id.
- `python scripts/run_all.py --verify-determinism` runs the pipeline twice and requires byte-identical artifacts.
- `raw/` is checksummed in `reports/raw_checksums.sha256` and set read-only; every artifact is hashed in `manifest.json`.

