# Gen18 data audit (written by `scripts/g18_audit.py`; frozen source of the cohort counts)

Recomputed from `systems/corpus_records.csv`, `series.csv`, `duplicates.csv` and `exclusions.csv` as written by `scripts/g18_build_db.py` (bundle SHA-256 `fefbefc6fe993aa9ce9db1a0c338adb9e5f58a8b75bab084cc1df4e024faf5dd`). Every count below is the value the pre-registration commits to; where it differs from DESIGN.md 0.2 or PRE_REGISTRATION.md section 2 the discrepancy table (section 7) says so. No model was fitted before this file was written.

Regime of every number: cohort = bundle after the gen13 quarantine and the unit-slip rule; no hold-out (descriptive); averaging unit named per table; parameter status: none (no parameter exists yet).

## 1. Data facts of DESIGN.md 0.2, recomputed

| fact | value |
|---|---|
| bundle rows -> after gen13 quarantine | 5992 -> 5860 (129 TODGA-structure-under-foreign-name rows, 3 sentinel rows at log D <= -6) |
| publications (gen6 `publication_id`, none missing) | 105 |
| systems under the key of section 3.1 | 287 corpus + 2 literature = 289 |
| two-ligand (synergist) systems / systems with alcohol modifiers | 5 / 11 |
| loading series, publication-aware (>= 3 distinct metal concentrations at fixed system, metal, publication, acid, extractant concentration; after the duplicate rule) | **10** |
| loading series, publication-blind (sensitivity only) | 11 |
| loading-active series (log D range >= 0.3; the rising Ce series included) | 7 |
| unit-slip duplicate rows (`UNIT_SLIP_DUPLICATE`, fit-ineligible) | **7** (7 in pub_0e7f3e0563, 7 on the 3 M side; 7 pairs) |
| tied-D groups corpus-wide (same publication, SMILES, metal, D equal to 6 significant digits at different conditions) | 130 groups, 324 rows, 24 publications; of these the unit-slip tier is 7 groups and the `TIED_D` tier 123 groups / 310 rows (kept, flagged) |
| replicate groups (exact 64-column condition key + publication + SMILES + metal, >= 2 rows) and median within-group sd of log D | 282, **0.225** (fit-eligible rows only: 0.227) |
| fit-ineligible rows | 19 ({'acid_nan_or_nonpositive': 11, 'UNIT_SLIP_DUPLICATE': 7, 'extractant_nan_or_zero': 1}) |
| system-metal groups fittable (>= 6 aggregated points, >= 3 distinct levels on log acid or log extractant), band 20-30C (NaN temperature assigned to it) | **233** (all bands pooled: 235) |
| of these, spanning >= 2 publications (E1) | **59 groups in 14 systems** (all bands: 59 / 14) |
| TODGA / nitrate / aliphatic / no additive (`sys_5cb78e5000d40860`) | 514 rows, 28 publications, 14 metals, acid 0.009333-5 M, extractant 0.0002313-0.3 M, metal 0.0001-2180 mM, 5-45 C; Pr 28 rows, Nd 66 rows, 8 publications with both |
| TODGA/Nd 3 M HNO3 / 0.1 M loading series pub_5a68dc5665 | 6 points, 4.9-12 mM, log D 1.37 -> -0.22; records Ca_SAFE:2222, Ca_SAFE:2221, Ca_SAFE:2220, Ca_SAFE:2219, Ca_SAFE:2224, Ca_SAFE:2217; series caf22524baa8080e |
| rows with NaN metal concentration / NaN temperature (after quarantine) | 1407 / 75 |
| temperature bands (NaN -> 20-30C) | {'20-30C': 5701, '30-40C': 63, '40-50C': 38, '<20C': 30, '>=50C': 28} |
| corpus coverage of the Pr/Nd case | no PC88A, Cyanex 272 or D2EHPA rows (both case systems are literature entries with 0 records) |

## 2. E1 cohort (PRE_REGISTRATION.md section 2; unit of the decision = system)

Fit-eligible records of band 20-30C, aggregated per replicate group; a (system, metal) group is fittable with >= 6 aggregated points and >= 3 distinct levels on log acid or log extractant; it enters E1 when it spans >= 2 publications. `publications` is the union over the system's E1 groups; `max_publications` the largest single group. R1 (iii) needs M1 to beat B1 in ceil(0.6 x 14) = 9 systems.

| system_id | ligand | name | n_groups | publications | max_publications | n_records | metals |
|---|---|---|---|---|---|---|---|
| sys_07ee9637c98c1e20 | TODGA | TODGA in other, nitrate medium, no additive | 13 | 7 | 5 | 249 | Ce;Dy;Er;Eu;Gd;Ho;La;Lu;Nd;Pr;Sm;Tb;Tm |
| sys_120bb57e9148dd0d | TEHDGA | TEHDGA in aliphatic hydrocarbon, nitrate medium, with 1_octanol (modifier) | 1 | 2 | 2 | 31 | Nd |
| sys_1419f400c83e9ad8 | DMDODGA | DMDODGA in aliphatic hydrocarbon, nitrate medium, no additive | 1 | 2 | 2 | 239 | Eu |
| sys_267144068d8e09d2 | DMDODGA | DMDODGA in aliphatic hydrocarbon, nitrate medium, with 1_octanol (modifier) | 1 | 2 | 2 | 102 | Nd |
| sys_5cb78e5000d40860 | TODGA | TODGA in aliphatic hydrocarbon, nitrate medium, no additive | 14 | 27 | 18 | 514 | Ce;Dy;Er;Eu;Gd;Ho;La;Lu;Nd;Pr;Sm;Tb;Tm;Yb |
| sys_81bcc3c06cbf0b4c | D3DODGA | D3DODGA in aliphatic hydrocarbon, nitrate medium, no additive | 1 | 3 | 3 | 35 | Nd |
| sys_81e3169a7f85c2b9 | TBDGA | TBDGA in aromatic, nitrate medium, no additive | 7 | 4 | 4 | 198 | Dy;Er;Eu;Gd;La;Nd;Sm |
| sys_95746e54b97ae741 | NTAamide(C8) | NTAamide(C8) in aliphatic hydrocarbon, nitrate medium, no additive | 1 | 2 | 2 | 20 | Eu |
| sys_a2a472b5124b09c6 | TODGA | TODGA in aromatic, nitrate medium, no additive | 14 | 8 | 7 | 294 | Ce;Dy;Er;Eu;Gd;Ho;La;Lu;Nd;Pr;Sm;Tb;Tm;Yb |
| sys_a7195d8a9d8696e0 | TEHDGA | TEHDGA in aliphatic hydrocarbon, nitrate medium, no additive | 2 | 8 | 5 | 71 | Eu;Nd |
| sys_a9fea6791c7a14d6 | DMDOHEMA | DMDOHEMA in aliphatic hydrocarbon, nitrate medium, no additive | 1 | 2 | 2 | 52 | Eu |
| sys_b06c95ac0efc3d4e | C5BTBP | C5BTBP in other, nitrate medium, no additive | 1 | 2 | 2 | 45 | Eu |
| sys_be340fe5092ae17a | 2-[1-(dioctylamino)-1-oxopropan-2-yl]oxy-N-N-dioctylpropanamide | 2-[1-(dioctylamino)-1-oxopropan-2-yl]oxy-N-N-dioctylpropanamide in aliphatic hydrocarbon, nitrate medium, no additive | 1 | 3 | 3 | 40 | Eu |
| sys_ebce91a448faa5c3 | TODGA | TODGA in alcohol modifier, nitrate medium, no additive | 1 | 5 | 5 | 39 | Eu |

Aggregated by extractant name (the panel's counts in PRE_REGISTRATION.md section 2 are per extractant name, not per system key):

| ligand | n_systems | n_groups | max_publications |
|---|---|---|---|
| 2-[1-(dioctylamino)-1-oxopropan-2-yl]oxy-N-N-dioctylpropanamide | 1 | 1 | 3 |
| C5BTBP | 1 | 1 | 2 |
| D3DODGA | 1 | 1 | 3 |
| DMDODGA | 2 | 2 | 2 |
| DMDOHEMA | 1 | 1 | 2 |
| NTAamide(C8) | 1 | 1 | 2 |
| TBDGA | 1 | 7 | 4 |
| TEHDGA | 2 | 3 | 5 |
| TODGA | 4 | 42 | 18 |

Per-group table: `results/audit/e1_groups.csv`.

## 3. E2 cohort (publication-aware loading series; unit = series)

| loading_series_id | system_id | ligand_name | metal | publication_id | acid_nominal_M | ligand_M | n_points | mM_min | mM_max | log_d_tracer | log_d_min | log_d_max | log_d_range | loading_active |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ls_sys_07ee9637c98c1e20_Ce_pub_917a4583d4_3_0.1 | sys_07ee9637c98c1e20 | TODGA | Ce | pub_917a4583d4 | 3 | 0.1 | 8 | 0.011 | 52.42 | 1 | 1 | 1.65 | 0.6502 | True |
| ls_sys_5cb78e5000d40860_Nd_pub_5a68dc5665_3_0.1 | sys_5cb78e5000d40860 | TODGA | Nd | pub_5a68dc5665 | 3 | 0.1 | 6 | 4.9 | 12 | 1.371 | -0.2218 | 1.371 | 1.593 | True |
| ls_sys_5cb78e5000d40860_Nd_pub_d3c970567f_3_0.1 | sys_5cb78e5000d40860 | TODGA | Nd | pub_d3c970567f | 3 | 0.1 | 4 | 4.2 | 6.8 | -0.03218 | -0.03218 | -0.0196 | 0.01259 | False |
| ls_sys_5cb78e5000d40860_Nd_pub_d3c970567f_3_0.2 | sys_5cb78e5000d40860 | TODGA | Nd | pub_d3c970567f | 3 | 0.2 | 4 | 9.7 | 13 | -0.02298 | -0.02758 | -0.02039 | 0.00719 | False |
| ls_sys_5cb78e5000d40860_Nd_pub_d3c970567f_3_0.3 | sys_5cb78e5000d40860 | TODGA | Nd | pub_d3c970567f | 3 | 0.3 | 8 | 20 | 30.4 | -0.03152 | -0.03267 | 0.001543 | 0.03421 | False |
| ls_sys_740f07a521006be1_Nd_pub_5a68dc5665_3_0.1 | sys_740f07a521006be1 | TDDGA | Nd | pub_5a68dc5665 | 3 | 0.1 | 6 | 20 | 30 | 0.9542 | 0.6021 | 0.9542 | 0.3522 | True |
| ls_sys_81bcc3c06cbf0b4c_Nd_pub_0e7f3e0563_1_0.2 | sys_81bcc3c06cbf0b4c | D3DODGA | Nd | pub_0e7f3e0563 | 1 | 0.2 | 18 | 7.341 | 289.7 | 0.7134 | -0.6081 | 1.82 | 2.428 | True |
| ls_sys_81bcc3c06cbf0b4c_Nd_pub_0e7f3e0563_3_0.2 | sys_81bcc3c06cbf0b4c | D3DODGA | Nd | pub_0e7f3e0563 | 3 | 0.2 | 8 | 8.2 | 289.3 | 1.602 | -0.5589 | 1.602 | 2.161 | True |
| ls_sys_a7195d8a9d8696e0_Nd_pub_15b174237f_1_0.2 | sys_a7195d8a9d8696e0 | TEHDGA | Nd | pub_15b174237f | 1 | 0.2 | 3 | 6.933 | 41.6 | 1.172 | 0.1276 | 1.172 | 1.045 | True |
| ls_sys_a7195d8a9d8696e0_Nd_pub_5a68dc5665_3_0.1 | sys_a7195d8a9d8696e0 | TEHDGA | Nd | pub_5a68dc5665 | 3 | 0.1 | 6 | 2.3 | 9.7 | 0.273 | -0.773 | 0.273 | 1.046 | True |

Loading-active subset (range >= 0.3): ls_sys_07ee9637c98c1e20_Ce_pub_917a4583d4_3_0.1, ls_sys_5cb78e5000d40860_Nd_pub_5a68dc5665_3_0.1, ls_sys_740f07a521006be1_Nd_pub_5a68dc5665_3_0.1, ls_sys_81bcc3c06cbf0b4c_Nd_pub_0e7f3e0563_1_0.2, ls_sys_81bcc3c06cbf0b4c_Nd_pub_0e7f3e0563_3_0.2, ls_sys_a7195d8a9d8696e0_Nd_pub_15b174237f_1_0.2, ls_sys_a7195d8a9d8696e0_Nd_pub_5a68dc5665_3_0.1.

Publication-blind series (sensitivity X5): lsb_sys_07ee9637c98c1e20_Ce_3_0.1, lsb_sys_5cb78e5000d40860_Eu_0.5_0.1, lsb_sys_5cb78e5000d40860_Eu_1_0.1, lsb_sys_5cb78e5000d40860_Nd_3_0.1, lsb_sys_5cb78e5000d40860_Nd_3_0.2, lsb_sys_5cb78e5000d40860_Nd_3_0.3, lsb_sys_740f07a521006be1_Nd_3_0.1, lsb_sys_81bcc3c06cbf0b4c_Nd_1_0.2, lsb_sys_81bcc3c06cbf0b4c_Nd_3_0.2, lsb_sys_a7195d8a9d8696e0_Nd_1_0.2, lsb_sys_a7195d8a9d8696e0_Nd_3_0.1.

## 4. Duplicates

Unit-slip tier (the flagged copy is fit-ineligible; the kept copy carries no flag):

| group_id | safe_exp_id | publication_id | metal | D | cond__acid_concentration_M | cond__metal_concentration_mM | duplicate_flag | kept | series_residual |
|---|---|---|---|---|---|---|---|---|---|
| 0b0a7846afd44ded | Ca_SAFE:2691 | pub_0e7f3e0563 | Nd | 0.6134 | 1 | 141.1 |  | True | -0.1113 |
| 0b0a7846afd44ded | Ca_SAFE:2698 | pub_0e7f3e0563 | Nd | 0.6134 | 3 | 0.9783 | UNIT_SLIP_DUPLICATE | False | -3.357 |
| 240a67ec8fd0ac3d | Ca_SAFE:2686 | pub_0e7f3e0563 | Nd | 3.563 | 1 | 49.75 |  | True | 0.2097 |
| 240a67ec8fd0ac3d | Ca_SAFE:2694 | pub_0e7f3e0563 | Nd | 3.563 | 3 | 0.3449 | UNIT_SLIP_DUPLICATE | False | -3.263 |
| 6500cbb627a49fa4 | Ca_SAFE:2687 | pub_0e7f3e0563 | Nd | 3.632 | 1 | 38.74 |  | True | 0.1117 |
| 6500cbb627a49fa4 | Ca_SAFE:2693 | pub_0e7f3e0563 | Nd | 3.632 | 3 | 0.2686 | UNIT_SLIP_DUPLICATE | False | -3.415 |
| 6a445b1c5b48e5a5 | Ca_SAFE:2684 | pub_0e7f3e0563 | Nd | 1.352 | 1 | 81.97 |  | True | 0.001061 |
| 6a445b1c5b48e5a5 | Ca_SAFE:2696 | pub_0e7f3e0563 | Nd | 1.352 | 3 | 0.5683 | UNIT_SLIP_DUPLICATE | False | -3.363 |
| 9c7448d4aae16fb0 | Ca_SAFE:2688 | pub_0e7f3e0563 | Nd | 66 | 1 | 20.8 |  | True | 1.107 |
| 9c7448d4aae16fb0 | Ca_SAFE:2699 | pub_0e7f3e0563 | Nd | 66 | 3 | 0.1442 | UNIT_SLIP_DUPLICATE | False | -2.555 |
| af3fc8c412cfa343 | Ca_SAFE:2690 | pub_0e7f3e0563 | Nd | 0.9533 | 1 | 104.4 |  | True | -0.0479 |
| af3fc8c412cfa343 | Ca_SAFE:2697 | pub_0e7f3e0563 | Nd | 0.9533 | 3 | 0.7238 | UNIT_SLIP_DUPLICATE | False | -3.359 |
| c64148812d848763 | Ca_SAFE:2685 | pub_0e7f3e0563 | Nd | 1.885 | 1 | 68.11 |  | True | 0.06657 |
| c64148812d848763 | Ca_SAFE:2695 | pub_0e7f3e0563 | Nd | 1.885 | 3 | 0.4722 | UNIT_SLIP_DUPLICATE | False | -3.337 |

Tied-D tier: 123 groups / 310 rows kept with `TIED_D` (exploratory sensitivity X4 refits without them). Rows per publication (top 10):

| publication_id | rows |
|---|---|
| pub_e287124c22 | 116 |
| pub_19783c039b | 49 |
| pub_8415538291 | 28 |
| pub_0a8da089ad | 17 |
| pub_51ddfefa0c | 14 |
| pub_515c63b970 | 14 |
| pub_1605e436d7 | 14 |
| pub_51c0693f9c | 10 |
| pub_7bbf4ac140 | 7 |
| pub_c8195b4ca7 | 6 |

## 5. Replicate floor

282 replicate groups with >= 2 rows; median within-group sd of log D **0.2251** (all rows) / 0.2267 (fit-eligible rows). This is the floor below which a log D difference has no process consequence (R1 margin 0.05 is inside it).

## 6. Cohort freeze

SHA-256 of the frozen tables (`results/audit/cohort_sha256.txt`):

- `9ee021738703a17b82043c2b30c00cb0e22b53e4d2fe563595458f158744c413`  corpus_records.csv
- `e331ab3705cc390d316b68672b64476712758c9b24636286c8a4ebd2b8085a7c`  series.csv
- `0545b3fdbef09238d95d27e0b0baaff3eb63259382295c7f134a68f3c6d2a663`  duplicates.csv
- `80d26ab4428162faed1566b439a642c7d54371d59981ee9347db8920309b28c5`  exclusions.csv

## 7. Discrepancies against DESIGN.md 0.2 and PRE_REGISTRATION.md section 2

| quantity | expected | source | observed | status | explanation |
|---|---|---|---|---|---|
| rows_in | 5992 | DESIGN 0.2 | 5992 | match |  |
| todga_name_mismatch_rows | 129 | DESIGN 0.2 | 129 | match |  |
| sentinel_rows | 3 | DESIGN 0.2 | 3 | match |  |
| rows_after_quarantine | 5860 | DESIGN 0.2 | 5860 | match |  |
| n_publications | 105 | DESIGN 0.2 | 105 | match |  |
| n_systems_corpus | 287 | DESIGN 0.2 | 287 | match |  |
| n_loading_series_publication_aware | 10 | DESIGN 0.2 / PRE_REG 2 | 10 | match |  |
| n_loading_series_publication_blind | 11 | DESIGN 0.2 | 11 | match |  |
| n_unit_slip_rows | 7 | DESIGN 0.2 / PRE_REG 0 | 7 | match |  |
| n_tied_d_groups | 130 | DESIGN 0.2 | 130 | match |  |
| n_tied_d_rows | 324 | DESIGN 0.2 | 324 | match |  |
| n_tied_d_publications | 24 | DESIGN 0.2 | 24 | match |  |
| replicate_groups | 282 | DESIGN 0.2 | 282 | match |  |
| replicate_sd_median | 0.225 | DESIGN 0.2 | 0.2251420112417688 | match |  |
| fittable_groups_primary_band | 233 | DESIGN 0.2 / PRE_REG 2 | 233 | match |  |
| e1_groups | 59 | DESIGN 0.2 / PRE_REG 2 | 59 | match |  |
| e1_systems | 14 | DESIGN 0.2 / PRE_REG 2 | 14 | match |  |
| todga_rows | 514 | DESIGN 0.2 | 514 | match |  |
| todga_publications | 28 | DESIGN 0.2 | 28 | match |  |
| todga_metals | 14 | DESIGN 0.2 | 14 | match |  |
| todga_pr_rows | 28 | DESIGN 0.2 | 28 | match |  |
| todga_nd_rows | 66 | DESIGN 0.2 | 66 | match |  |
| todga_pubs_with_pr_and_nd | 8 | DESIGN 0.2 | 8 | match |  |
| todga_e1_groups | 42 | PRE_REG 2 (sys_5cb78e5000d40860: 42 groups over 18 publications) | 14 | DIFFERS (explained) | 42 is the sum over the four TODGA-named systems; the system key gives 14 for sys_5cb78e5000d40860 (addenda/WB1.md A9) |
| todga_e1_max_publications | 18 | PRE_REG 2 (largest single group of the system) | 18 | match |  |
| todga_named_e1_groups_all_systems | 42 | PRE_REG 2 read per extractant name | 42 | match |  |
| loading_active_series | 7 | PRE_REG 2 (6 series + the Ce series, included) | 7 | match |  |
| n_nan_metal_rows | 1532 | DESIGN 0.2 (pre-quarantine count) | 1407 | DIFFERS (explained) | 1532 is the pre-quarantine count; the quarantine removes 125 of them |

Recorded discrepancies (the audit is the frozen source; the pre-registration is not edited, a dated pre-fit addendum records them):

- todga_e1_groups: expected 42 (PRE_REG 2 (sys_5cb78e5000d40860: 42 groups over 18 publications)), observed 14 -- 42 is the sum over the four TODGA-named systems; the system key gives 14 for sys_5cb78e5000d40860 (addenda/WB1.md A9)
- n_nan_metal_rows: expected 1532 (DESIGN 0.2 (pre-quarantine count)), observed 1407 -- 1532 is the pre-quarantine count; the quarantine removes 125 of them

## 8. Declared assumptions carried by every derived parameter

- `cond__metal_concentration_mM` is the initial aqueous concentration of the row's single metal (`OA_ASSUMED`; addenda/ORCHESTRATOR_prefit_20260913.md keeps it for every E2 series).
- O/A = 1 for every corpus loading row.
- nominal acid = equilibrium acidity (`EQUILIBRIUM_ACID_ASSUMED_NOMINAL`); aqueous nitrate = nominal HNO3.
- NaN temperature -> band 20-30C (75 rows).
- unit-slip rule applied on the metal-concentration axis only (addenda/WB1.md A3).
