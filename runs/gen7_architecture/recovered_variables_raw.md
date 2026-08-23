# Recovered variables: upstream `*_SAFE.csv` vs the 5,992-row bundle

Generated 2026-08-19T15:42:53.982150+00:00. Upstream `/Users/lilshadix/PycharmProjects/lanthanide_dataset_builder/raw_data` (31 files, 48,138 rows). Downstream `dataset with 3D structures/dataset.parquet` (5992 rows x 2261 cols).

**The join is exact.** `safe_exp_id` = `{stem}_SAFE:{exp_id}` matches **5,992 / 5,992** rows, the upstream key has 0 duplicates, and four independent cross-checks agree on every row (`D`==`obsDvaluesValue`, `metal_symbol`==`Metal_Name`, `LIGAND_SMILES`==`Extractant_SMILES`, `extractant_name`==`Extractant_Name`).

**Baseline for the leak/effect columns.** Under the gen5-style cell (`extractant_name` + `metal_symbol` + every `cond__*`), 304 cells hold 962 rows with within-cell log_D sum-of-squares 336.5 (rms 0.591 log units). The `% SS` column below is how much of that supposed 'replicate noise' each dropped variable actually explains.


## Full column ledger (every upstream column + every field parsed out of `comments_description`)

| # | Upstream column | Rows present /5992 | Coverage | Missing | Uniq | Downstream equivalent | Equivalence | Truly missing? | % SS | Inference-safe for a NEW extractant? | Publication-specific? | Leak risk |
|---|---|---:|---:|---:|---:|---|---|:--:|---:|:--:|:--:|:--:|
| 1 | `exp_id` | 5992 | 100.00% | 0 | 5992 | safe_exp_id (embedded) | exact | no | — | no | yes | high |
| 2 | `upstream_source_file` | 5992 | 100.00% | 0 | 25 | none | none | **YES** | — | no | yes | medium |
| 3 | `upstream_stem` | 5992 | 100.00% | 0 | 25 | none | none | **YES** | — | no | yes | medium |
| 4 | `Extractant_Name` | 5992 | 100.00% | 0 | 228 | extractant_name | exact (5992/5992 identical, 228 uniq both sides) | no | — | yes | no | low |
| 5 | `Extractant_SMILES` | 5992 | 100.00% | 0 | 192 | LIGAND_SMILES | exact (5992/5992 identical, 192 uniq both sides) | no | — | yes | no | low |
| 6 | `Extractant_inchi` | 18 | 0.30% | 5974 | 1 | none | none | **YES** | — | yes | no | low |
| 7 | `Extractant_Concentration_M` | 5991 | 99.98% | 1 | 423 | cond__extractant_concentration_M | exact where parseable (5973/5973 agree) | no | — | yes | no | low |
| 8 | `Acid_Name` | 5992 | 100.00% | 0 | 9 | cond__acid__* one-hots | exact 1:1 (9 values -> 9 one-hots, 0 rows unassigned) | no | — | yes | no | low |
| 9 | `Acid_Concentration_M` | 5981 | 99.82% | 11 | 691 | cond__acid_concentration_M | exact (5981/5981 agree, same 11 missing) | no | — | yes | no | low |
| 10 | `Solvent_Name` | 5992 | 100.00% | 0 | 83 | cond__diluent__* one-hots + geom_cond__diluent_family | lossy | **YES** | 4.95 | yes | no | low |
| 11 | `f_Solvent_Name` | 5992 | 100.00% | 0 | 83 | cond__diluent__* | exact duplicate of Solvent_Name (5992/5992 identical) | no | — | yes | no | low |
| 12 | `Metal_Name` | 5992 | 100.00% | 0 | 14 | metal_symbol / metal | exact (0 mismatches, 14 values) | no | — | yes | no | low |
| 13 | `Metal_Oxidation_state` | 5195 | 86.70% | 797 | 1 | metal_ox | constant-imputed | no | 0.27 | yes | no | low |
| 14 | `Metal_Concentration_mM` | 4460 | 74.43% | 1532 | 141 | cond__metal_concentration_mM | exact (4460/4460 agree, same 1532 missing) | no | — | yes | no | low |
| 15 | `ini_comp` | 5992 | 100.00% | 0 | 1832 | implicit | derived duplicate | no | — | yes | no | low |
| 16 | `Phase_Modifier_Name` | 265 | 4.42% | 5727 | 9 | cond__additive__* + geom_cond__modifier_class | exact 1:1 (9 values -> 9 one-hots) | no | — | yes | no | low |
| 17 | `Phase_Modifier_Concentration_M` | 265 | 4.42% | 5727 | 15 | none (only geom_cond__modifier_class, a 4-way identity bin) | none | **YES** | 0.04 | yes | no | low |
| 18 | `Holdback_Agent_Name` | 0 | 0.00% | 5992 | 0 | none | n/a - column is 100% empty | no | — | yes | no | low |
| 19 | `Holdback_Agent_Concentration_M` | 0 | 0.00% | 5992 | 0 | none | n/a - column is 100% empty | no | — | yes | no | low |
| 20 | `Acid_Concentration_Organic_M` | 36 | 0.60% | 5956 | 35 | none | none | **YES** | 0.00 | no | yes | HIGH |
| 21 | `f_Metal_Concentration_mM` | 0 | 0.00% | 5992 | 0 | none | n/a - column is 100% empty | no | — | no | yes | CATASTROPHIC |
| 22 | `Contact_Time_min` | 3776 | 63.02% | 2216 | 43 | cond__contact_time_min + geom_cond__contact_time_bin | exact (3776/3776 agree, same 2216 missing) | no | — | yes | no | low |
| 23 | `Shaking_Time_min` | 1067 | 17.81% | 4925 | 11 | geom_cond__shaking_time_bin only (4 coarse bins) | binned, numeric dropped | **YES** | 0.27 | yes | no | low |
| 24 | `Radiolytic_Dosage_kGy` | 0 | 0.00% | 5992 | 0 | none | n/a - column is 100% empty | no | — | yes | no | low |
| 25 | `obsTemp` | 5909 | 98.61% | 83 | 1 | cond__temperature_C (label only) | constant label | no | — | n/a | no | low |
| 26 | `obsTempsValue` | 5909 | 98.61% | 83 | 22 | cond__temperature_C | exact (5909/5909 agree, same 83 missing) | no | — | yes | no | low |
| 27 | `obsTempUnit` | 5909 | 98.61% | 83 | 1 | none | constant label | no | — | n/a | no | low |
| 28 | `volType` | 3388 | 56.54% | 2604 | 1 | geom_cond__phase_ratio_bin (all 'missing') | dropped entirely | **YES** | — | yes | no | medium |
| 29 | `volValue` | 3388 | 56.54% | 2604 | 1 | geom_cond__phase_ratio_bin (all 'missing') | dropped entirely | **YES** | 0.00 | yes | no | medium |
| 30 | `thirdType` | 0 | 0.00% | 5992 | 0 | none | n/a - column is 100% empty | no | — | yes | no | low |
| 31 | `thirdValue` | 0 | 0.00% | 5992 | 0 | none | n/a - column is 100% empty | no | — | yes | no | low |
| 32 | `obsDvalues` | 5992 | 100.00% | 0 | 1 | D (label only) | constant label | no | — | n/a | no | low |
| 33 | `obsDvaluesValue` | 5992 | 100.00% | 0 | 4691 | D / log_D | exact - IT IS THE TARGET (5992/5992 agree to 1e-6) | no | — | no | n/a | CATASTROPHIC |
| 34 | `obsTotalEnergiesUnit` | 0 | 0.00% | 5992 | 0 | none | n/a - column is 100% empty | no | — | n/a | no | low |
| 35 | `comments_obsType` | 5992 | 100.00% | 0 | 1 | none | constant label | no | — | n/a | no | low |
| 36 | `DOI` | 5992 | 100.00% | 0 | 115 | none (bundle 'reference' columns are 100% empty) | none | **YES** | 0.32 | no | yes | HIGH |
| 37 | `entry_author` | 5992 | 100.00% | 0 | 3 | none | none | **YES** | 0.00 | no | yes | medium |
| 38 | `addition_date` | 5992 | 100.00% | 0 | 543 | none | none | **YES** | 83.10 | no | yes | EXTREME |
| 39 | `comments_description` | 5992 | 100.00% | 0 | 634 | none | none | **YES** | — | no | yes | HIGH |
| 40 | `comments_description::Complexant_Name` | 557 | 9.30% | 5435 | 9 | none | none | **YES** | 55.96 | yes | no | low |
| 41 | `comments_description::Complexant_Concentration_M` | 557 | 9.30% | 5435 | 25 | none | none | **YES** | 56.67 | yes | no | low |
| 42 | `comments_description::Complexant_SMILES` | 557 | 9.30% | 5435 | 9 | none | none | **YES** | — | yes | no | low |
| 43 | `comments_description::No_of_Metals` | 3388 | 56.54% | 2604 | 8 | none | none | **YES** | 0.02 | yes | no | low |
| 44 | `comments_description::Aqueous_Phase_Metals` | 327 | 5.46% | 5665 | 17 | none | none | **YES** | — | yes | no | low |
| 45 | `comments_description::Additional_Comments` | 421 | 7.03% | 5571 | 12 | none | none | **YES** | 0.17 | no | yes | medium |
| 46 | `comments_description::Data_Location` | 3388 | 56.54% | 2604 | 39 | none | none | **YES** | 10.63 | no | yes | EXTREME |
| 47 | `comments_description::Publication_Year` | 3388 | 56.54% | 2604 | 19 | none | none | **YES** | 0.00 | no | yes | medium |
| 48 | `comments_description::Title` | 3388 | 56.54% | 2604 | 34 | none | none | **YES** | — | no | yes | HIGH |
| 49 | `comments_description::Authors` | 3388 | 56.54% | 2604 | 34 | none | none | **YES** | — | no | yes | HIGH |
| 50 | `comments_description::No_of_Extractants` | 3388 | 56.54% | 2604 | 2 | extractant_name (composite string) | partially | **YES** | — | yes | no | low |
| 51 | `comments_description::Metal_Concentration_mM` | 3383 | 56.46% | 2609 | 45 | cond__metal_concentration_mM | duplicate of the column | no | — | yes | no | low |

## Truly missing downstream (24 columns)

### `upstream_source_file`

- **Coverage** 5992/5992 (100.00%), 25 unique values, 0 missing (0.00%).
- **Physical interpretation** — Which per-metal SAFE export the row was pulled from; 66% of rows have a stem different from their own metal (co-reported measurements).
- **Safe at inference for a NEW extractant?** no
- **Publication-specific?** yes
- **Could it leak the target?** medium
- **Notes** — already implicit in safe_exp_id.
- **Recommendation** — grouping/diagnostics only.

| value | count |
|---|---:|
| `Ca_SAFE.csv` | 1825 |
| `Er_SAFE.csv` | 1107 |
| `Eu_SAFE.csv` | 780 |
| `Am_SAFE.csv` | 642 |
| `Lu_SAFE.csv` | 281 |
| `Dy_SAFE.csv` | 237 |
| `La_SAFE.csv` | 181 |
| `Ce_SAFE.csv` | 176 |
| `Gd_SAFE.csv` | 153 |
| `Pa_SAFE.csv` | 123 |
| `Pr_SAFE.csv` | 92 |
| `Ho_SAFE.csv` | 87 |
| `Nd_SAFE.csv` | 82 |
| `Pd_SAFE.csv` | 50 |
| `Bi_SAFE.csv` | 46 |
| `Cm_SAFE.csv` | 23 |
| `Pb_SAFE.csv` | 22 |
| `Tb_SAFE.csv` | 22 |
| `Sm_SAFE.csv` | 18 |
| `Cd_SAFE.csv` | 14 |

### `upstream_stem`

- **Coverage** 5992/5992 (100.00%), 25 unique values, 0 missing (0.00%).
- **Physical interpretation** — Same as upstream_source_file without the suffix.
- **Safe at inference for a NEW extractant?** no
- **Publication-specific?** yes
- **Could it leak the target?** medium
- **Notes** — 
- **Recommendation** — grouping/diagnostics only.

| value | count |
|---|---:|
| `Ca` | 1825 |
| `Er` | 1107 |
| `Eu` | 780 |
| `Am` | 642 |
| `Lu` | 281 |
| `Dy` | 237 |
| `La` | 181 |
| `Ce` | 176 |
| `Gd` | 153 |
| `Pa` | 123 |
| `Pr` | 92 |
| `Ho` | 87 |
| `Nd` | 82 |
| `Pd` | 50 |
| `Bi` | 46 |
| `Cm` | 23 |
| `Pb` | 22 |
| `Tb` | 22 |
| `Sm` | 18 |
| `Cd` | 14 |

### `Extractant_inchi`

- **Coverage** 18/5992 (0.30%), 1 unique values, 5974 missing (99.70%).
- **Physical interpretation** — InChI of the extractant.
- **Safe at inference for a NEW extractant?** yes
- **Publication-specific?** no
- **Could it leak the target?** low
- **Notes** — populated on only 18 rows and the sole value is the placeholder '-,-'. Zero information.
- **Recommendation** — drop.

| value | count |
|---|---:|
| `-,-` | 18 |

### `Solvent_Name`

- **Coverage** 5992/5992 (100.00%), 83 unique values, 0 missing (0.00%).
- **Physical interpretation** — Organic diluent, including modifier volume fractions written into the name.
- **Safe at inference for a NEW extractant?** yes
- **Publication-specific?** no
- **Could it leak the target?** low
- **Explains 4.95%** of the within-repeated-cell log_D SS.
- **Notes** — 83 upstream values -> 41 one-hot categories; 305 rows (43 distinct solvents) are swallowed by cond__diluent__other. 1357 rows (22.65%) state a modifier volume fraction in the name (30%,40%,5%,50%,10%,20%,25%,12.5%,15%) that exists downstream only as unordered one-hot identity, never as a number. Explains 4.95% of the within-repeated-cell log_D SS.
- **Recommendation** — PARTIALLY MISSING: extract (base diluent, modifier identity, modifier volume fraction) as three features.

| value | count |
|---|---:|
| `n-Dodecane` | 1744 |
| `toluene` | 611 |
| `Isopar L with 30 vol% Exxal 13` | 350 |
| `kerosene` | 237 |
| `1-octanol` | 221 |
| `kerosene 0.7, 1-octanol 0.3` | 194 |
| `Chloroform` | 186 |
| `kerosene with 40 vol% 1-octanol` | 171 |
| `kerosene with 30 vol% 1-octanol` | 167 |
| `tert-butylbenzene` | 120 |
| `TPH` | 118 |
| `nitrobenzene` | 115 |
| `1,4-diisopropylbenzene` | 102 |
| `tBuB` | 89 |
| `DIPB` | 88 |
| `CH3Cl` | 84 |
| `meta-nitrobenzotrifluoride` | 76 |
| `1-octanol 0.5, kerosene 0.5` | 70 |
| `n-octane with 5% 1-octanol` | 67 |
| `CHCl3` | 61 |

### `Phase_Modifier_Concentration_M`

- **Coverage** 265/5992 (4.42%), 15 unique values, 5727 missing (95.58%).
- **Physical interpretation** — Molarity of the phase modifier / synergist in the organic phase.
- **Safe at inference for a NEW extractant?** yes
- **Publication-specific?** no
- **Could it leak the target?** low
- **Explains 0.04%** of the within-repeated-cell log_D SS.
- **Notes** — 265 rows, 15 distinct values 0.1-2.537 M. Downstream keeps the modifier IDENTITY but not HOW MUCH. Splits only 6 repeated cells and explains 0.04% of within-cell SS, so it is a small-n feature, not a replicate explainer. Physically real: 2.537 M 1-octanol vs 0.1 M changes the organic-phase activity and the extracted stoichiometry.
- **Recommendation** — TRULY MISSING - add as a numeric with 0.0 for 'no modifier' (a genuine zero, not a NaN).

| value | count |
|---|---:|
| `2.537 M` | 102 |
| `1.9 M` | 60 |
| `1.58 M` | 39 |
| `0.5 M` | 16 |
| `1.36 M` | 16 |
| `1.57 M` | 8 |
| `0.3 M` | 6 |
| `0.25 M` | 5 |
| `0.75 M` | 4 |
| `0.1 M` | 3 |
| `2.12 M` | 2 |
| `1.11 M` | 1 |
| `0.71 M` | 1 |
| `0.91 M` | 1 |
| `0.2 M` | 1 |

### `Acid_Concentration_Organic_M`

- **Coverage** 36/5992 (0.60%), 35 unique values, 5956 missing (99.40%).
- **Physical interpretation** — Acid co-extracted INTO the organic phase, measured after equilibration.
- **Safe at inference for a NEW extractant?** no
- **Publication-specific?** yes
- **Could it leak the target?** HIGH
- **Explains 0.00%** of the within-repeated-cell log_D SS.
- **Notes** — 36 rows (0.60%) only, all from ONE DOI, ONE extractant (DMDOHEMA), ONE metal (Eu). corr(org_acid, log_D)=0.85, corr(org_acid, aq_acid)=0.95. This is a post-equilibration OBSERVABLE, not a set condition: it is a second measurement on the same tube as D. Splits 0 repeated cells.
- **Recommendation** — DO NOT USE AS A FEATURE. Outcome variable, unavailable at inference, and single-study.

| value | count |
|---|---:|
| `0.008 M` | 2 |
| `0.002 M` | 1 |
| `0.014 M` | 1 |
| `0.032 M` | 1 |
| `0.055 M` | 1 |
| `0.099 M` | 1 |
| `0.274 M` | 1 |
| `0.154 M` | 1 |
| `1.55 M` | 1 |
| `0.456 M` | 1 |
| `0.609 M` | 1 |
| `0.813 M` | 1 |
| `1.04 M` | 1 |
| `1.3 M` | 1 |
| `1.02 M` | 1 |
| `0.777 M` | 1 |
| `0.569 M` | 1 |
| `0.361 M` | 1 |
| `0.213 M` | 1 |
| `0.143 M` | 1 |

### `Shaking_Time_min`

- **Coverage** 1067/5992 (17.81%), 11 unique values, 4925 missing (82.19%).
- **Physical interpretation** — Agitation time - the SAME physical quantity as contact time, recorded under a different column by a different curator.
- **Safe at inference for a NEW extractant?** yes
- **Publication-specific?** no
- **Could it leak the target?** low
- **Explains 0.27%** of the within-repeated-cell log_D SS.
- **Notes** — 1067 rows (17.81%), 11 distinct values 2-120 min. Perfectly DISJOINT from Contact_Time_min (0 rows have both; 1149 rows have neither). Downstream cond__contact_time_min is NaN on all 1067 of them, so a unifiable 'equilibration time' is null on 2216 rows when it is really null on only 1149. geom_cond__shaking_time_bin exists but crushes 30/31/32/45/50/60/120 min into one 'medium' bin. Explains 0.27% of within-cell SS.
- **Recommendation** — TRULY MISSING as a number - build cond__equilibration_time_min = coalesce(Contact_Time_min, Shaking_Time_min) plus a binary flag for which one it was; raises coverage 63.0% -> 80.8%.

| value | count |
|---|---:|
| `30.0 min` | 452 |
| `60.0 min` | 286 |
| `10.0 min` | 92 |
| `45.0 min` | 57 |
| `50.0 min` | 51 |
| `120.0 min` | 42 |
| `25.0 min` | 39 |
| `2.0 min` | 26 |
| `15.0 min` | 18 |
| `32.0 min` | 2 |
| `31.0 min` | 2 |

### `volType`

- **Coverage** 3388/5992 (56.54%), 1 unique values, 2604 missing (43.46%).
- **Physical interpretation** — Literal 'Volume Ratio' - marks that an O:A phase ratio was recorded.
- **Safe at inference for a NEW extractant?** yes
- **Publication-specific?** no
- **Could it leak the target?** medium
- **Notes** — present on exactly the 3388 rows curated by Thomas Summers; 0 on the other 2604. It is a CURATOR-BATCH marker, not a physical variable. geom_cond__phase_ratio_bin is 'missing' for all 5992 rows.
- **Recommendation** — do not use: it is a perfect proxy for entry_author=='Thomas Summers'.

| value | count |
|---|---:|
| `Volume Ratio` | 3388 |

### `volValue`

- **Coverage** 3388/5992 (56.54%), 1 unique values, 2604 missing (43.46%).
- **Physical interpretation** — Organic:aqueous phase volume ratio.
- **Safe at inference for a NEW extractant?** yes
- **Publication-specific?** no
- **Could it leak the target?** medium
- **Explains 0.00%** of the within-repeated-cell log_D SS.
- **Notes** — 3388 rows, and the ONLY value is 1.0. Zero variance -> the number carries no information; only its presence/absence does, and that is the curator-batch marker above. Explains 0.00% of within-cell SS.
- **Recommendation** — do not add. Record in the manifest that O:A ratio is unrecoverable (assumed 1:1 everywhere it is stated).

| value | count |
|---|---:|
| `1.0` | 3388 |

### `DOI`

- **Coverage** 5992/5992 (100.00%), 115 unique values, 0 missing (0.00%).
- **Physical interpretation** — Comma-separated citation list; always includes the SAFE compilation self-citation 10.1021/jacs.5c19738.
- **Safe at inference for a NEW extractant?** no
- **Publication-specific?** yes
- **Could it leak the target?** HIGH
- **Explains 0.32%** of the within-repeated-cell log_D SS.
- **Notes** — 115 raw strings -> the canonicaliser in gen6/provenance.py folds notation variants. Group-mean of log_D by DOI alone reaches R2 = 0.4171; DOI x Data-Location reaches R2 = 0.5001. Splits 17 repeated cells (0.32% of within-cell SS).
- **Recommendation** — GROUPING VARIABLE ONLY - use for study-aware splits and for auditing cross-study averaging. Never a feature.

| value | count |
|---|---:|
| `https://doi.org/10.15261/serdj.18.93, https://doi.org/10.1021/jacs.5c19738` | 459 |
| `https://cordis.europa.eu/project/id/211267, https://doi.org/10.1021/jacs.5c19738` | 403 |
| `https://doi.org/10.1080/07366299.2015.1087209 , https://doi.org/10.1021/jacs.5c19738` | 279 |
| `https://doi.org/10.1246/cl.200431, https://doi.org/10.1021/jacs.5c19738` | 253 |
| `https://doi.org/10.1007/s10967-020-07242-1, https://doi.org/10.1021/jacs.5c19738` | 238 |
| `https://doi.org/10.1080/01496395.2016.1274760, https://doi.org/10.1021/jacs.5c19738` | 215 |
| `https://doi.org/10.1016/j.hydromet.2020.105248, https://doi.org/10.1021/jacs.5c19738` | 215 |
| `https://doi.org/10.1021/acs.inorgchem.0c02861, https://doi.org/10.1021/jacs.5c19738` | 203 |
| `https://doi.org/10.1080/07366290600646947, https://doi.org/10.1021/jacs.5c19738` | 194 |
| `https://doi.org/10.1081/SEI-200037727, https://doi.org/10.1021/jacs.5c19738` | 179 |
| `https://doi.org/10.1039/D4GC01146E, https://doi.org/10.1021/jacs.5c19738` | 178 |
| `https://doi.org/10.1016/j.hydromet.2020.105248` | 177 |
| `https://doi.org/10.1016/j.seppur.2014.03.005, https://doi.org/10.1021/jacs.5c19738` | 164 |
| `https://doi.org/10.1081/SEI-120030392, https://doi.org/10.1021/jacs.5c19738` | 140 |
| `https://doi.org/10.1007/s41365-016-0055-0 , https://doi.org/10.1021/jacs.5c19738` | 134 |
| `https://doi.org/10.1007/s41365-017-0229-4 , https://doi.org/10.1021/jacs.5c19738` | 129 |
| `https://doi.org/10.2298/JSC171109043H, https://doi.org/10.1021/jacs.5c19738` | 96 |
| `https://doi.org/10.1080/07366290802672212, https://doi.org/10.1021/jacs.5c19738` | 94 |
| `https://doi.org/10.1080/07366299.2017.1415670, https://doi.org/10.1021/jacs.5c19738` | 93 |
| `https://doi.org/10.1007/s10967-020-07368-2, https://doi.org/10.1021/jacs.5c19738` | 75 |

### `entry_author`

- **Coverage** 5992/5992 (100.00%), 3 unique values, 0 missing (0.00%).
- **Physical interpretation** — Curator who entered the row (3 people).
- **Safe at inference for a NEW extractant?** no
- **Publication-specific?** yes
- **Could it leak the target?** medium
- **Explains 0.00%** of the within-repeated-cell log_D SS.
- **Notes** — Thomas Summers 3388 / Baosen Zhang 1423 / Aurora Clark 1181. Group-mean R2 = 0.1232. Determines which comment schema, whether volType is present, and whether Metal_Oxidation_state was filled in - i.e. it explains missingness patterns, not chemistry.
- **Recommendation** — diagnostic only; use to check that fold structure is not curator-confounded.

| value | count |
|---|---:|
| `Thomas Summers` | 3388 |
| `Baosen Zhang` | 1423 |
| `Aurora Clark` | 1181 |

### `addition_date`

- **Coverage** 5992/5992 (100.00%), 543 unique values, 0 missing (0.00%).
- **Physical interpretation** — Timestamp the row was typed into the database (543 distinct, only 4 distinct DAYS).
- **Safe at inference for a NEW extractant?** no
- **Publication-specific?** yes
- **Could it leak the target?** EXTREME
- **Explains 83.10%** of the within-repeated-cell log_D SS.
- **Notes** — Group-mean R2 by full timestamp = 0.6474 (by day only 0.1258). Inside repeated condition cells it explains 83.1% of the log_D sum of squares - more than any physical variable - because consecutive seconds of typing correspond to consecutive points of ONE figure curve. It is a near-perfect experiment-series surrogate.
- **Recommendation** — GROUPING VARIABLE ONLY. Using it as a feature would be the single worst leak available in this dataset.

| value | count |
|---|---:|
| `2024-12-16 21:51:11` | 18 |
| `2024-12-16 21:52:36` | 18 |
| `2024-12-16 21:53:20` | 18 |
| `2024-12-16 21:53:31` | 18 |
| `2024-12-16 21:52:45` | 18 |
| `2024-12-16 21:53:40` | 18 |
| `2024-12-16 21:53:21` | 18 |
| `2024-12-16 21:53:02` | 17 |
| `2024-12-16 21:53:45` | 17 |
| `2024-12-16 21:54:32` | 17 |
| `2024-12-16 21:53:41` | 17 |
| `2024-12-16 21:52:38` | 17 |
| `2024-12-16 21:52:41` | 17 |
| `2024-12-16 21:52:42` | 17 |
| `2026-01-20 20:37:37` | 16 |
| `2026-01-20 20:37:38` | 16 |
| `2026-01-20 20:39:36` | 16 |
| `2026-01-20 20:39:38` | 16 |
| `2026-01-20 20:39:43` | 16 |
| `2026-01-20 20:39:47` | 16 |

### `comments_description`

- **Coverage** 5992/5992 (100.00%), 634 unique values, 0 missing (0.00%).
- **Physical interpretation** — One free-text field carrying two different schemas: a semi-structured 'Data Location: ...; Complexant_Name: ...; ...' blob on 3388 rows (exactly the Thomas Summers batch) and a figure caption ('fig3 ... extraction:nan', 'updated ORNL', 'IDEaL') on the other 2604.
- **Safe at inference for a NEW extractant?** no
- **Publication-specific?** yes
- **Could it leak the target?** HIGH
- **Notes** — 634 distinct strings. Contains at least 12 parseable sub-fields, listed separately below. Splits 104 repeated cells.
- **Recommendation** — PARSE IT, then drop the raw text. Some sub-fields are real chemistry; the rest are publication bookkeeping.

| value | count |
|---|---:|
| `updated ORNL` | 714 |
| `IDEaL` | 306 |
| `Data Location: Table SI1; Additional Comments: nan; Complexant_Name: nan; Complexant_SMILES: nan; Complexant_Concentration_M: nan; Publication_Year: 2020; Ti...` | 200 |
| `Data Location: Figure 8; Additional Comments: nan; Complexant_Name: nan; Complexant_SMILES: nan; Complexant_Concentration_M: nan; Publication_Year: 2015; Tit...` | 155 |
| `Data Location: Figure 8; Additional Comments: nan; Complexant_Name: nan; Complexant_SMILES: nan; Complexant_Concentration_M: nan; Publication_Year: 2015; Tit...` | 124 |
| `Data Location: Figure 2; Additional Comments: nan; Complexant_Name: nan; Complexant_SMILES: nan; Complexant_Concentration_M: nan; Publication_Year: 2020; Tit...` | 98 |
| `fig3 nan nan extraction:nan` | 95 |
| `fig4 nan nan extraction:nan` | 80 |
| `Data Location: Table 1; Additional Comments: nan; Complexant_Name: nan; Complexant_SMILES: nan; Complexant_Concentration_M: nan; Publication_Year: 2014; Titl...` | 80 |
| `Data Location: Figure 7; Additional Comments: nan; Complexant_Name: nan; Complexant_SMILES: nan; Complexant_Concentration_M: nan; Publication_Year: 2017; Tit...` | 75 |
| `Data Location: Figure 3; Additional Comments: nan; Complexant_Name: nan; Complexant_SMILES: nan; Complexant_Concentration_M: nan; Publication_Year: 2020; Tit...` | 70 |
| `Data Location: Table 1; Additional Comments: nan; Complexant_Name: nan; Complexant_SMILES: nan; Complexant_Concentration_M: nan; Publication_Year: 2003; Titl...` | 67 |
| `fig3 there are more metals, kerosene nan extraction:nan` | 60 |
| `Data Location: Figure 2; Additional Comments: nan; Complexant_Name: DTPA; NaNO3; Complexant_SMILES: C(CN(CC(=O)O)CC(=O)O)N(CCN(CC(=O)O)CC(=O)O)CC(=O)O.[N+](=...` | 56 |
| `Data Location: Figure 5; Additional Comments: Metal conc. may be 2 or 200 ppm; Complexant_Name: nan; Complexant_SMILES: nan; Complexant_Concentration_M: nan;...` | 56 |
| `Data Location: Figure 3; Additional Comments: nan; Complexant_Name: nan; Complexant_SMILES: nan; Complexant_Concentration_M: nan; Publication_Year: 2017; Tit...` | 56 |
| `figS4-B nan nan extraction:nan` | 56 |
| `Data Location: Figure 3; Additional Comments: Metal conc. may be 2 or 200 ppm; Complexant_Name: nan; Complexant_SMILES: nan; Complexant_Concentration_M: nan;...` | 54 |
| `Data Location: Figure 4; Additional Comments: nan; Complexant_Name: nan; Complexant_SMILES: nan; Complexant_Concentration_M: nan; Publication_Year: 2023; Tit...` | 54 |
| `Data Location: Figure 4; Additional Comments: Metal conc. may be 2 or 200 ppm; Complexant_Name: nan; Complexant_SMILES: nan; Complexant_Concentration_M: nan;...` | 52 |

### `comments_description::Complexant_Name`

- **Coverage** 557/5992 (9.30%), 9 unique values, 5435 missing (90.70%).
- **Physical interpretation** — AQUEOUS-PHASE COMPLEXANT / HOLDBACK AGENT. DTPA, HEDTA, CDTA (aminopolycarboxylate holdbacks), TEDGA and DOODA(C2) (hydrophilic diglycolamide strippers), BTP-4Me, malonamide, amic acid, and NaNO3 (salting-out salt). These are added precisely to SUPPRESS extraction of the metal.
- **Safe at inference for a NEW extractant?** yes
- **Publication-specific?** no
- **Could it leak the target?** low
- **Explains 55.96%** of the within-repeated-cell log_D SS.
- **Notes** — 557 rows (9.30%), 9 distinct agents. THE SINGLE LARGEST RECOVERABLE PHYSICAL VARIABLE: it explains 55.96% of the within-repeated-cell log_D sum of squares (962 rows in 304 cells, rms spread 0.591 log units) and splits 51 of those cells. Within the 40 (extractant, metal, [acid], [extractant conc]) cells that contain both, adding a complexant shifts log_D by a mean of -0.918 (median -1.071). Every metal (14) and 2-3 extractants per agent, so it is not a single-study artefact for the big agents.
- **Recommendation** — TRULY MISSING - top-priority recovery. Encode as complexant identity one-hot + complexant present flag (+ SMILES-derived descriptors, which are already available in the blob).

| value | count |
|---|---:|
| `DTPA; NaNO3` | 165 |
| `TEDGA` | 88 |
| `DOODA(C2)` | 88 |
| `NaNO3` | 70 |
| `Amicacid` | 69 |
| `malonamide` | 56 |
| `HEDTA` | 10 |
| `BTP-4Me` | 10 |
| `CDTA` | 1 |

### `comments_description::Complexant_Concentration_M`

- **Coverage** 557/5992 (9.30%), 25 unique values, 5435 missing (90.70%).
- **Physical interpretation** — Molarity of each aqueous complexant (semicolon-separated when there are two, e.g. '0.01; 1' = 0.01 M DTPA + 1 M NaNO3).
- **Safe at inference for a NEW extractant?** yes
- **Publication-specific?** no
- **Could it leak the target?** low
- **Explains 56.67%** of the within-repeated-cell log_D SS.
- **Notes** — 557 rows, 25 distinct strings. Explains 56.67% of within-repeated-cell SS and splits 106 cells - the best single explainer measured. Must be split on ';' and aligned positionally with Complexant_Name.
- **Recommendation** — TRULY MISSING - recover as log10 concentration per agent class; this is the natural partner of the MASSACTION block.

| value | count |
|---|---:|
| `0.5` | 84 |
| `0.05` | 58 |
| `1` | 56 |
| `0.01; 1` | 56 |
| `0.01` | 38 |
| `0.2` | 32 |
| `0.02` | 28 |
| `0.005` | 22 |
| `0.1` | 15 |
| `0.02;0.54` | 14 |
| `0.02;1.73` | 14 |
| `0.001` | 14 |
| `0.002` | 14 |
| `3` | 14 |
| `5` | 14 |
| `0.05; 3` | 14 |
| `0.05; 5` | 14 |
| `0.05;1` | 13 |
| `0.02;0.88` | 12 |
| `0.01;1` | 11 |

### `comments_description::Complexant_SMILES`

- **Coverage** 557/5992 (9.30%), 9 unique values, 5435 missing (90.70%).
- **Physical interpretation** — Structure of each aqueous complexant (9 distinct, incl. the DTPA.nitrate salt written as a dot-disconnected SMILES).
- **Safe at inference for a NEW extractant?** yes
- **Publication-specific?** no
- **Could it leak the target?** low
- **Notes** — 557 rows. Lets the complexant be described by the same 2D/3D descriptor machinery already built for the extractant, so a NEW holdback agent generalises instead of being a one-hot.
- **Recommendation** — TRULY MISSING - recover; feed through the existing ligand-descriptor path.

| value | count |
|---|---:|
| `C(CN(CC(=O)O)CC(=O)O)N(CCN(CC(=O)O)CC(=O)O)CC(=O)O.[N+](=O)([O-])[O-]` | 165 |
| `CCN(CC)C(=O)COCC(=O)N(CC)CC` | 88 |
| `CCN(CC)C(=O)COCCOCC(=O)N(CC)CC` | 88 |
| `[N+](=O)([O-])[O-]` | 70 |
| `CCN(CC)C(=O)COCC(=O)O` | 69 |
| `NC(=O)CC(N)=O` | 56 |
| `OCCN(CCN(CC([O-])=O)CC([O-])=O)CC([O-])=O` | 10 |
| `CC1=C(N=NC(=N1)C2=NC(=CC=C2)C3=NC(=C(N=N3)C)C)C` | 10 |
| `OC(=O)CN(CC(O)=O)[C@@H]1CCCC[C@H]1N(CC(O)=O)CC(O)=O` | 1 |

### `comments_description::No_of_Metals`

- **Coverage** 3388/5992 (56.54%), 8 unique values, 2604 missing (43.46%).
- **Physical interpretation** — How many metals were present in the same aqueous phase - i.e. single-metal vs competitive/loaded extraction.
- **Safe at inference for a NEW extractant?** yes
- **Publication-specific?** no
- **Could it leak the target?** low
- **Explains 0.02%** of the within-repeated-cell log_D SS.
- **Notes** — 3388 rows carry it (3061 = 1 metal, 327 = 2..33 metals). Explains only 0.02% of within-cell SS (the mixtures rarely coincide with repeated cells) but it is a real regime flag: mean log_D 0.618 (single) vs 0.778 (mixture). Coverage is capped at 56.5% because the other curator batch never recorded it.
- **Recommendation** — TRULY MISSING - recover as n_metals plus an is_mixture flag, with an explicit 'unknown' level for the 2604 uncovered rows.

| value | count |
|---|---:|
| `1` | 3061 |
| `2` | 189 |
| `9` | 65 |
| `24` | 54 |
| `32` | 10 |
| `33` | 6 |
| `31` | 2 |
| `23` | 1 |

### `comments_description::Aqueous_Phase_Metals`

- **Coverage** 327/5992 (5.46%), 17 unique values, 5665 missing (94.54%).
- **Physical interpretation** — The full list of co-present metals (Am(III); Eu(III); ... Zr; Mo; Pd; Fe ...), including tetravalent Pu and non-lanthanides that compete for the extractant.
- **Safe at inference for a NEW extractant?** yes
- **Publication-specific?** no
- **Could it leak the target?** low
- **Notes** — 327 rows (5.46%), 17 distinct lists. Populated exactly where No. of Metals > 1. Gives total competing-metal loading, which is what actually saturates the organic phase.
- **Recommendation** — TRULY MISSING - recover as n_competing_metals and flags for competitive tetravalent (Pu(IV)/Th) / trivalent actinide presence.

| value | count |
|---|---:|
| `Am(III); Eu(III)` | 144 |
| `Am(III); La(III); Ce(III); Pr(III); Nd(III); Sm(III); Eu(III); Gd(III); Tb(III); Dy(III); Ho(III); Er(III); Tm(III); Yb(III); Lu(III); Yb(III); Sr; Zr; Mo; P...` | 54 |
| `Am(III); Eu(III); Y; La(III); Ce(III); Pr(III); Nd(III); Sm(III); Gd(III)` | 51 |
| `Am(III); Eu(III); Gd(III); Sm(III); Y; Nd(III); Pr(III); La(III); Ce(III)` | 14 |
| `Am(III); Cm(III); Cf(III); Y; La(III); Ce(III); Pr(III); Nd(III); Sm(III); Eu(III); Gd(III); Ag; Al; Ba; Cd; Cr; Cs; Cu; Fe; Mo; Na; Ni; Pd; Rb; Rh; Ru; Sb; ...` | 10 |
| `Am(III); Pu(IV); Y; La(III); Ce(III); Pr(III); Nd(III); Sm(III); Eu(III); Gd(III); Ag; Ba; Cd; Cr; Cs; Fe; Mo; Na; Ni; Pd; Rb; Ru; Sb; Se; Sn; Sr; Te; Zr; U(...` | 6 |
| `La(III); Pu(IV)` | 5 |
| `Ce(III); Pu(IV)` | 5 |
| `Pr(III); Pu(IV)` | 5 |
| `Nd(III); Pu(IV)` | 5 |
| `Sm(III); Pu(IV)` | 5 |
| `Eu(III); Pu(IV)` | 5 |
| `Gd(III); Pu(IV)` | 5 |
| `Tb(III); Pu(IV)` | 5 |
| `Dy(III); Pu(IV)` | 5 |
| `Am(III); Pu(IV); Y; La(III); Ce(III); Pr(III); Nd(III); Sm(III); Eu(III); Gd(III); Ag; Al; Ba; Cd; Cr; Cs; Cu; Fe; Mo; Na; Ni; Pd; Rb; Rh; Ru; Sb; Se; Sn; Sr...` | 2 |
| `Am(III); Pu(IV); Eu(III); Tc; Na; K; Cr; Mn; Fe; Ni; Sr; Y; Zr; Mo; Cs; Ba; La(III); Ce(III); Pr(III); Nd(III); Sm(III); U(VI); Pd` | 1 |

### `comments_description::Additional_Comments`

- **Coverage** 421/5992 (7.03%), 12 unique values, 5571 missing (92.97%).
- **Physical interpretation** — Curator caveats. Two kinds mixed together: (a) METHOD flags - 'Manual Extraction', 'High-throughput Automated Extraction', 'Direct Sampling Method', 'Sequential Sampling Method'; (b) DATA-QUALITY / matrix flags - 'Metal conc. may be 2 or 200 ppm', 'Simulated UOX1 Purex Raffinate solution', 'Discrepancy in HNO3 acid conc...', 'Value ... adjusted to match Figure 5'.
- **Safe at inference for a NEW extractant?** no
- **Publication-specific?** yes
- **Could it leak the target?** medium
- **Explains 0.17%** of the within-repeated-cell log_D SS.
- **Notes** — 421 rows (7.03%), 12 distinct strings. Explains 0.17% of within-cell SS. The 'Simulated ... Raffinate/HLW solution' values (33 rows) are the ONLY marker that the aqueous phase was a full fission-product matrix rather than a clean nitrate solution - a real condition. The rest are provenance/uncertainty notes.
- **Recommendation** — PARTIALLY RECOVERABLE - extract only two derived flags: is_simulated_waste_matrix and measurement_uncertainty_flagged. The latter belongs in sample weighting, not in X.

| value | count |
|---|---:|
| `Metal conc. may be 2 or 200 ppm` | 253 |
| `Ln conc. between 10-100 ppm. 50 ppm reported here.` | 28 |
| `Manual Extraction` | 28 |
| `High-throughput Automated Extraction` | 25 |
| `Discrepancy in HNO3 acid conc. reported in text (2M) and Figure (3M)` | 24 |
| `Direct Sampling Method` | 14 |
| `Simulated UOX1 Purex Raffinate solution` | 14 |
| `Sequential Sampling Method` | 13 |
| `Simulated Highly Active Raffinate solution` | 12 |
| `Simulated High Level Waste solution` | 6 |
| `Value in Table 1 off from value shown in Figure 5 by factor of 10^-1. Value adjusted to match Figure 5` | 3 |
| `Simulated Pressurized Heavy Water Reactor High Level Waste solution; Pd concentration not reported` | 1 |

### `comments_description::Data_Location`

- **Coverage** 3388/5992 (56.54%), 39 unique values, 2604 missing (43.46%).
- **Physical interpretation** — Which figure or table of the source paper the point was digitised from ('Figure 3', 'Table SI1', 'Figure S4B').
- **Safe at inference for a NEW extractant?** no
- **Publication-specific?** yes
- **Could it leak the target?** EXTREME
- **Explains 10.63%** of the within-repeated-cell log_D SS.
- **Notes** — 3388 rows (56.5%), 39 distinct labels; combined with DOI it gives 197 distinct (paper, figure) series. Group-mean R2 = 0.5001 and it splits 104 repeated cells (10.63% of within-cell SS). A further ~1124 of the 2604 free-text rows carry the same information in the 'fig3'/'table1' prefix, so a unified data-location field would cover ~4512 rows (75.3%).
- **Recommendation** — GROUPING VARIABLE ONLY - this is the true experiment_series_id the gen5 unseen_series folds were trying to approximate. Never a feature.

| value | count |
|---|---:|
| `Figure 3` | 550 |
| `Figure 4` | 444 |
| `Figure 2` | 409 |
| `Figure 8` | 380 |
| `Table SI1` | 200 |
| `Figure 5` | 198 |
| `Table 1` | 193 |
| `Figure 6` | 152 |
| `Figure 7` | 136 |
| `Figure 1` | 110 |
| `Figure S4B` | 57 |
| `Figure S4C` | 57 |
| `Figure S4A` | 56 |
| `Table S15` | 53 |
| `Table 3` | 51 |
| `Table S20` | 42 |
| `Figure S6` | 39 |
| `Table 4` | 30 |
| `Table S17` | 28 |
| `Table S18` | 28 |

### `comments_description::Publication_Year`

- **Coverage** 3388/5992 (56.54%), 19 unique values, 2604 missing (43.46%).
- **Physical interpretation** — Year of the source publication (1999-2024).
- **Safe at inference for a NEW extractant?** no
- **Publication-specific?** yes
- **Could it leak the target?** medium
- **Explains 0.00%** of the within-repeated-cell log_D SS.
- **Notes** — 3388 rows, 19 values. Group-mean R2 = 0.1398. It is a proxy for which extractant families were fashionable, so it will look predictive on a random split and collapse on a chemotype hold-out.
- **Recommendation** — do not use as a feature; useful for a temporal (train-on-old, test-on-new) split, which is a stronger generalisation test.

| value | count |
|---|---:|
| `2020` | 911 |
| `2011` | 459 |
| `2017` | 457 |
| `2015` | 279 |
| `2014` | 192 |
| `2007` | 189 |
| `2018` | 189 |
| `2024` | 178 |
| `2016` | 134 |
| `2009` | 95 |
| `2003` | 70 |
| `2023` | 54 |
| `2010` | 49 |
| `2001` | 49 |
| `2019` | 28 |
| `2013` | 21 |
| `2006` | 18 |
| `1999` | 14 |
| `2012` | 2 |

### `comments_description::Title`

- **Coverage** 3388/5992 (56.54%), 34 unique values, 2604 missing (43.46%).
- **Physical interpretation** — Title of the source publication (34 distinct).
- **Safe at inference for a NEW extractant?** no
- **Publication-specific?** yes
- **Could it leak the target?** HIGH
- **Notes** — 3388 rows. Group-mean R2 = 0.1856. Equivalent to a coarse DOI.
- **Recommendation** — GROUPING VARIABLE ONLY.

| value | count |
|---|---:|
| `Separation of Am, Cm and Lanthanides by Solvent Extraction with Hydrophilic and Lipophilic Organic Ligands` | 459 |
| `The Effect of Alkyl Substituents on Actinide and Lanthanide Extraction by Diglycolamide Compounds` | 279 |
| `Extraction and Separation between Light and Heavy Lanthanides by N,N,N′,N′-Tetraoctyl-diglycolamide from Organic Acid` | 253 |
| `Extraction of rare earth elements from nitrate solution using novel unsymmetrical diglycolamide` | 215 |
| `Solvent extraction of An(III) and Ln(III) using TODGA in aromatic diluents to suppress third phase formation` | 215 |
| `Structure Activity Relationship Approach toward the Improved Separation of Rare-Earth Elements Using Diglycolamides` | 200 |
| `Effect of DTPA on the Extractions of Actinides(III) and Lanthanides(III) from Nitrate Solution into Todga/n‐Dodecane` | 179 |
| `Agile synthesis and automated, high-throughput evaluation of diglycolamides for liquid–liquid extraction of rare-earth elements` | 178 |
| `Extraction and stripping behaviors of 14 lanthanides from nitric acid medium by N,N’-dimethyl-N,N’-dioctyl diglycolamide` | 168 |
| `Extraction behavior of trivalent lanthanides from nitric acid medium by selected structurally related diglycolamides as novel extractants` | 164 |
| `Extraction study of rare earth elements with N,N′-dibutyl–N,N′-di(1-methylheptyl)-diglycolamide from hydrochloric acid` | 134 |
| `Extraction of lanthanide ions with N,N,N′,N′-tetrabutyl-3-oxa-diglycolamide from nitric acid media` | 132 |
| `Effect of structure on extraction behavior of praseodymium with a series of unsymmetrical diglycolamides from hydrochloric acid` | 96 |
| `Synthesis of Pre-Organized Bisdiglycolamides (BisDGA) and Study of their Extraction Properties for Actinides(III) and Lanthanides(III)` | 94 |
| `Actinide–lanthanide co-extraction by rigidified diglycolamides` | 93 |
| `Synthesis and characterization of new unsymmetrical diglycolamide extractants for lanthanide ion partitioning: part one—straight-chain alkyl derivatives` | 75 |
| `Extraction Studies of Lanthanide(III) Ions with N,N′‐Dimethyl‐N,N′‐diphenylpyridine‐2,6‐dicarboxyamide (DMDPhPDA) from Nitric Acid Solutions` | 70 |
| `Synthesis and evaluation of new modified diglycolamides with different stereochemistry for extraction of tri- and tetravalent metal ions` | 54 |
| `The novel extractants, diglycolamides, for the extraction of lanthanides and actinides in HNO3-n-dodecane system` | 49 |
| `Synthesis and Am/Eu extraction of novel TODGA derivatives` | 47 |

### `comments_description::Authors`

- **Coverage** 3388/5992 (56.54%), 34 unique values, 2604 missing (43.46%).
- **Physical interpretation** — Author list of the source publication (34 distinct).
- **Safe at inference for a NEW extractant?** no
- **Publication-specific?** yes
- **Could it leak the target?** HIGH
- **Notes** — 3388 rows; 1:1 with Title.
- **Recommendation** — GROUPING VARIABLE ONLY - useful for a research-group-disjoint split, the hardest realistic split.

| value | count |
|---|---:|
| `Y Sasaki, Y Kitatsuji, Y Tsubata, Y Sugo, Y Morita` | 459 |
| `Y Sasaki, Y Sugo, K Morita, KL Nash` | 279 |
| `Y. Sasaki, M. Matsumiya, M. Nakase, K. Takeshita` | 253 |
| `EA Mowafy, D Mohamed` | 215 |
| `P Weßling, U Müllich, E Guerinoni, A Geist, PJ Panak` | 215 |
| `Diana Stamberga, Mary R. Healy, Vyacheslav S. Bryantsev, Camille Albisser, Yana Karslyan, Benjamin Reinhart, Alena Paulenova, Mac Foster, Ilja Popovs, Kevin ...` | 200 |
| `A Apichaibukol, Y Sasaki, Y Morita` | 179 |
| `L An, Y Yao, TB Hall, F Zhao, L Qi` | 178 |
| `Yaoyang Liu, Chuang Zhao, Zhibin Liu, Yu Zhou, Caishan Jiao, Meng Zhang, Hongguo Hou, Yang Gao, Hui He & Guoxin Tian` | 168 |
| `E.A. Mowafy, D. Mohamed` | 164 |
| `Guo-Jing Sun, Jin-Hong Yang, Hong-Xiao Yang, Guo-Xin Sun & Yu Cui` | 134 |
| `X.-J. Peng, Y. Cui, J.-F. Ma, Y. Li, G.-X. Sun` | 132 |
| `J. Han, X. Cai, G. Sun, Y. Cui` | 96 |
| `MT Murillo, AG Espartero, J Sánchez‐Quesada, J de Mendoza, P Prados` | 94 |
| `E Macerata, A Ossola, W Panzeri, M Giola, F Faroldi, DA Tinonin, A Mele, A Casnati, M Mariani` | 93 |
| `B. G. Tokheim, S. S. Kelly, R. C. Ronald, K. L. Nash` | 75 |
| `A. Shimada, T. Yaita, H. Narita, S. Tachimori, K. Okuno` | 70 |
| `Laura Diaz Gomez, Andreas Wilden, Dimitri Schneider, Zaina Paparigas, Giuseppe Modolo, Maria Chiara Gullo, Jurriaan Huskens, Willem Verboom` | 54 |
| `Y Sasaki, Y Sugo, S Suzuki, S Tachimori` | 49 |
| `M Iqbal, J Huskens, W Verboom, M Sypula, G Modolo` | 47 |

### `comments_description::No_of_Extractants`

- **Coverage** 3388/5992 (56.54%), 2 unique values, 2604 missing (43.46%).
- **Physical interpretation** — Whether the organic phase held one extractant or a synergistic pair.
- **Safe at inference for a NEW extractant?** yes
- **Publication-specific?** no
- **Could it leak the target?** low
- **Notes** — 3388 rows: 3370 with 1, 18 with 2 (the TODGA+DHOA rows). Downstream keeps them as the composite name 'TODGA,DHOA', and cond__additive__dhoa is 0 on all 18 (it is 1 on only 8 unrelated rows), so the second extractant is invisible to the model.
- **Recommendation** — TRULY MISSING for those 18 rows - either encode the synergist explicitly or exclude the mixture rows from single-ligand arms.

| value | count |
|---|---:|
| `1` | 3370 |
| `2` | 18 |


## Within-repeated-cell variance decomposition (all 962 rows in 304 repeated cells)

| variable | % of within-cell log_D SS explained |
|---|---:|
| `addition_date` | 83.10 |
| `Complexant_Concentration_M` | 56.67 |
| `Complexant_Name` | 55.96 |
| `Data_Location` | 10.63 |
| `Solvent_Name` | 4.95 |
| `DOI` | 0.32 |
| `Shaking_Time_min` | 0.27 |
| `Metal_Oxidation_state` | 0.27 |
| `Additional_Comments` | 0.17 |
| `Phase_Modifier_Concentration_M` | 0.04 |
| `No_of_Metals` | 0.02 |
| `entry_author` | 0.00 |
| `Publication_Year` | 0.00 |
| `Acid_Concentration_Organic_M` | 0.00 |
| `volValue` | 0.00 |

## Leak-surrogate strength (R² of the group mean of log_D alone, all 5,992 rows)

| grouping | R² |
|---|---:|
| `addition_date` | 0.6474 |
| `DOI+Data Location` | 0.5001 |
| `DOI` | 0.4171 |
| `exp_id_block_1000` | 0.3595 |
| `Title` | 0.1856 |
| `Publication_Year` | 0.1398 |
| `addition_date_day` | 0.1258 |
| `entry_author` | 0.1232 |

## Verdict

**Recover as features (physical, inference-safe, not publication-specific):**

1. `Complexant_Name` / `Complexant_Concentration_M` / `Complexant_SMILES` (557 rows, 9.30%) — explains **56.7%** of the 'replicate' spread; within matched cells a complexant moves log_D by **−0.92** on average.
2. `Shaking_Time_min` (1067 rows, 17.81%) — disjoint from `Contact_Time_min`; coalescing them raises equilibration-time coverage from 63.0% to 80.8%.
3. `Phase_Modifier_Concentration_M` (265 rows, 4.42%) — the modifier identity survived downstream, the amount did not.
4. Modifier volume fraction parsed out of `Solvent_Name` (1357 rows, 22.65%) — currently only an unordered one-hot; 305 rows collapse into `cond__diluent__other`.
5. `No. of Metals` / `Aqueous Phase Metals` (3388 / 327 rows) — single-metal vs competitive loading.
6. The second extractant of the 18 `TODGA,DHOA` synergistic rows (0.5 M DHOA is invisible downstream).
7. `Additional Comments` → `is_simulated_waste_matrix` (33 rows) only.

**Recover as GROUPING variables only — never as features:** `DOI`, `Title`, `Authors`, `Data Location`, `Publication_Year`, `entry_author`, `addition_date`, `exp_id`, `volType`.

**Blacklist permanently:** `obsDvaluesValue` (it *is* `D`), `f_Metal_Concentration_mM` (the numerator of `D`), `Acid_Concentration_Organic_M` (post-equilibration observable, 36 rows, one study).

**Drop as empty or constant:** `Holdback_Agent_Name`, `Holdback_Agent_Concentration_M`, `Radiolytic_Dosage_kGy`, `obsTotalEnergiesUnit`, `thirdType`, `thirdValue`, `f_Metal_Concentration_mM` (all 0 rows); `obsDvalues`, `obsTemp`, `obsTempUnit`, `comments_obsType`, `volType`, `volValue`, `Metal_Oxidation_state`, `Extractant_inchi` (all single-valued); `f_Solvent_Name` and `ini_comp` (exact duplicates).

