# Direction C: chemistry feature inventory and provenance (read-only audit)

Everything below was **verified** by reading code or by computation on the frozen bench, unless it is marked *conjecture*. I modified nothing in the repository. I ran Python with PYTHONDONTWRITEBYTECODE=1, and all scratch files are under `.../scratchpad/r3inv/`. During the session an untracked directory `generations/gen21/next_research_20261019/` appeared (created 17:49). I did not create it.

## 0. Cohort facts (recomputed)

* **C3 = k_max >= 3 AND a non-empty strict-adjacent query.** That gives 229 cells, 79 extractants, 38 chemotypes, 39 publications and 1,802 query pairs, matching ATTEMPTS.md:5.
  * The condition `k_max >= 3` alone gives 248 cells. The extra 19 cells have no strict-adjacent pair (8 of them measure exactly Nd;Sm;Gd;Dy;Er;Yb). gen21 drops them in `g21/cache.py:49` (`if not dz.query or dz.k_max < min_k`).
  * `k_max` depends only on the measured-metal set (`gen20k/design.py:51-55`).
* **sc009 in C3:** 129 cells, 23 extractants, 15 publications, 882 pairs (48.9 %).
* **Non-sc009 in C3:** 100 cells, 56 extractants, 37 chemotypes, 29 publications, 920 pairs.
* Five publications contain both groups: pub_126bbe7976, pub_4cba27f5f3, pub_5a68dc5665, pub_b9dfc468cd and pub_dfa96a5118.
* **sc009 is not the same as "DGA".**
  * sc009 = 21 N,N'-tetraalkyl diglycolamides with an unsubstituted backbone (TODGA 47 cells, DMDODGA 22, DHD2DGA 20, ...) plus DOODA(C8) and DOODA(C12). The two DOODAs are 7 cells and gen6 labels them `amide_other`.
  * Outside sc009 there are 35 DGA-family cells in C3: 19 extractants, 13 chemotypes. Examples: the α-alkylated sc027/sc028 group (mTDDGA, cMe2-/tMe2-TODGA, p-TODGA), TEDGA (sc071), DMDPDGA (N-aryl, sc076), TEtOMeDGA and the pyran DGA (sc020).
* Per-cell table: `scratchpad/r3inv/cells_kmax.csv` (cell_id, k_max, n_query, C3).

## 1. bench.frame columns (521 x 123)

Builder: `gen13sep/cohort.py:build_cohort` (L133-233). The frame is reached through `gen14/dirbench.py:load` (cache `gen14_direction/cache/bench.pkl`, L44, L56-67). `gen20k/data.py:41-63` asserts it equals `gen13_separation/manifests/cohort_exact.parquet` column by column.

| column(s) | meaning / how computed | status |
|---|---|---|
| cell_id | blake2b of extractant@@publication@@condition_key (L209-211) | identifier |
| extractant | canonical SMILES (L173) | identifier |
| extractant_name | name on the first row by safe_exp_id (L171, L174) | identifier |
| publication_id | from the gen6 provenance table (L142-144) | identifier |
| condition_key | exact key over 64 cond columns; NaN counts as a value; 9 significant digits (L99-110, L148) | identifier |
| experiment_series_ids, safe_exp_ids | joined lists (L177-180) | identifier / provenance |
| n_rows | rows per cell (L181) | availability |
| n_metals, metals | metals present (L182-183) | **availability / missingness** |
| n_replicated_metals | number of metals with >1 row (L184) | replication availability |
| replicate_sd_median | median replicate SD of log D (L185) | **label-derived** |
| chemotype, ecfp_cluster, chem_family | from the frozen gen6 chemistry map (L199-205), see §3 | target-free group ids; `chem_family` is in FORBIDDEN_FEATURE_EXACT (L71) |
| logD__* (14) | replicate-mean log D (L159-166) | **label** |
| nrep__* (14) | replicate counts | availability |
| repsd__* (14) | replicate SD | **label-derived** |
| cond__* (64) | first row of the cell (L188). These are key columns, so they are constant within a cell. Identical to the COND block. | target-free; acts as a publication fingerprint (§4) |
| recipe__DENTATE, recipe__coreCN | per-cell **mode** of the per-row Architector recipe (L195-197) | **depends on metal availability** (§3) |

`bench.coef` is a ridge fit of the centred Y curve on (radius, radius²) (`amplitude_bench.py:82-83`). It is label-derived, serves as the training target of G14/G13, and is never a feature.

## 2. Feature blocks (bench.frames)

Builder: `gen13sep/features.py:build_features` (L95-149). A forbidden-token assertion runs on every matrix (L68-71, L148; token lists at `cohort.py:64-71`).

| block | cols | source and builder | missing (all / C3) | constant cols in C3 | distinct vectors (all / C3) | identifier power |
|---|---|---|---|---|---|---|
| COND | 64 | bundle cond__ columns (features.py:120-121) | 0.9 % / 0.8 %; in C3: contact_time 34.5 %, metal_conc 15.7 %, temperature 2.2 % | 28 | 454 / 174 | **100 % of cells have a vector unique to one publication.** 1-NN publication accuracy: 93.7 % (all), 93.0 % (C3), 44.1 % in C3 when the neighbour must be a different extractant |
| MASSACT | 8 | log10 of the 5 continuous conditions, plus logL×recipe__DENTATE, logL×recipe__coreCN, logL×logH (features.py:74-92) | 7.3 % / 6.6 % | 0 | 404 / 140 | 99.1 % publication-unique in C3 |
| PHYSCHEM | 10 | bundle RDKit scalars, first row per SMILES (features.py:101-102, 124-125) | 0 | 0 | 79 / 69 | constant within each extractant |
| DONORS | 13 | gen6 map: census of the bundle `DONOR_TYPES` list (`src/lanthanide_separation/gen6/chemistry.py:74-77, 233-248`), plus chem__dentate/core_cn/n_ligands/n_fill from the **first bundle row** of the SMILES (chemistry.py:409, 471-477) | 0 | 2 | 27 / 23 | weak (9 % of cells extractant-unique) |
| ECFP | 2048 | bundle Morgan bits (features.py:128-129) | 0 | 1,669 | 77 / 66 | **chemotype is defined from these bits**: single-linkage Tanimoto >= 0.7 (`levels.py:66, 199`; chemistry.py:407) |
| LIG2D | 206 (168 rd + 38 hc) | `ligand_2d_descriptors.parquet` from `scripts/build_ligand_descriptors.py` and `src/lanthanide_separation/ligand_descriptors.py`; Ipc is log-transformed (features.py:107-108) | 0.7 % / 0.9 % (hc chain stats NaN for 30 C3 cells with no amide N; assigned_cip_homochiral 97 % NaN) | 18 | 90 / 79 | **perfect extractant fingerprint** (90 vectors = 90 extractants) |
| COORD | 114 | gen12.2 `gen122/coordination.py:descriptors_for` (L209). SMILES only, "no condition, no target, no metal" (L1-5). Spec v1.1.0. | 3.5 % / 0.4 %; all-NaN for DBD1MHDGA (sc018) and Et-Me-TODGA-Anti (sc027; 1 C3 cell) | 15 | 87 / 77 | 95.8 % extractant-unique |

**Label derivation: none of the 7 blocks reads log D, replicate statistics or coef.** This is verified both from the code paths above and by an empirical screen. The screen took every C3-varying column of every block and computed Spearman correlation against n_metals, k_max, mean Z and heavy fraction of the measured metals, n_rows, coef amplitude, coef curvature, mean log D and log D range. The largest correlations were:

* -0.60: cond__metal_concentration_mM against the heavy fraction of measured metals;
* -0.51: N(aromatic) against the log D level;
* +0.40: coord__dist__donor_pair_min against amplitude.

None of these is deterministic. replicate_sd_median could not be screened: only 19 C3 cells have it.

**Dependence on metal availability rather than on label values (verified):**

* **recipe__coreCN and recipe__DENTATE.** The bundle's Architector coreCN depends on the metal (Dy: 407 rows at CN 8 vs 2 at CN 9; La: 397 at CN 9 vs 21 at CN 8). coreCN varies inside 200 of the 229 C3 cells, and n_fill and n_ligs vary inside 202 and 171.
  * The cell mode is CN 8 for 173 C3 cells and CN 9 for 56.
  * It follows which metals were measured: all 13 cells with <= 20 % heavy metals get 9, and all 36 cells with > 80 % heavy metals get 8.
  * It reaches the models through `massact__logL_x_coreCN` (features.py:87-89). DENTATE varies in 7 C3 cells.
* **DONORS chem__dentate / core_cn / n_ligands / n_fill.** These equal the first bundle row of the SMILES for 190 of 190 extractants. That first row's metal is arbitrary (Eu for 106). chem__core_cn differs from recipe__coreCN in 157 of 229 C3 cells. They are target-free, but they are recipe artifacts, not chemistry.

## 3. What G14 and LEAN209 contain (so a new block is not a renamed copy)

* **G14** (`gen15/gen15/arms.py:102-105`):
  * Direction = `_logistic_sign` (L32-38), which calls `gen14/models.py:dir_logistic` (L123-147): L2 logistic, C=1, standardised columns, chemotype-balanced weights, trained on cells with >= 5 metals.
  * Its input is **TOPO39** only (`gen14/dirbench.py:83`): 17 `coord__arm__*` columns plus 22 `coord__dist__*` columns (donor graph distances, chelate-ring counts, denticity per pocket).
  * Magnitude = training-fold mean |amplitude|; curvature = training-fold mean.
  * G14 uses no conditions, no donor-element counts and no LIG2D.
* **Where chemistry enters gen21.** The only route is through G14: the `zero_g14` prior (G14@BLUP20, G14@ZERO; `g21/evaluate.py:69-81`) and the G14-based covariance used by DOPT_Q (`g21/cache.py`, `CV.fold_covariance(... facts["G14"])`).
  * HBK@ML, RIDGE2 and LINE use no feature block. The RIDGE2 prior is the training-mean coefficient vector (`gen20k/arms.py:39-41`, `adapters.py:66-80`).
* **LEAN209** = COND64 + MASSACT8 + PHYSCHEM10 + DONORS13 + COORD114 (`amplitude_bench.py:45`; `dirbench.py:104`). Other named sets:
  * CHEM137 = PHYSCHEM + DONORS + COORD; DONORS13; BITE6; ALL_BLOCKS (2,463 columns, `amplitude_bench.py:47`).
  * gen14 post-hoc STERIC7 (`gen14_direction/scripts/g14_steric.py:39-42`; coord__arch carbon/heavy-atom per donor, degree counts, Fsp3, orbit fraction). It was falsified.
* **Small physical recodings of the conditions that already exist** (gen15 condfe, `gen15_curve/exp/condfe/conds.py:52`):
  * NUMERIC14 (L130): log_acid, log_extr, log_metal, temp, log_contact, is_hno3, is_hcl, is_h2so4, complexant (carboxylic "acids"), modifier, aliphatic/aromatic/chlorinated/polar diluent.
  * PHYS4 (L137): complexant, is_hno3, log_acid, temp.
  * Verdict (GEN15_REPORT.md:382): the within-publication acid association is real (Spearman +0.356 with |a|) but worth only +0.009. The coefficient was shrunk to 0 in 90 % of folds.
* **Opt-in blocks outside bench.frames:**
  * RESP3D (`gen13sep/features3d.py`, xTB slope of Ln–donor distance against radius);
  * LOGK (features.py:48-50, 140-143);
  * gen15 phys3d (28 xTB columns; null);
  * gen19 `descriptors/extractant_components.csv`: donor census including triazine N, denticity_proxy, mechanism label. It matches all 79 C3 extractants by SMILES. Mechanism: sc009 = 23 NEUTRAL_SOLVATING; non-sc009 = 34 NEUTRAL_SOLVATING, 19 SOFT_N_DONOR, 2 CHELATING, 1 ACIDIC.

## 4. Candidate mechanistic features in C3 (229 cells)

AUC is for sc009 membership; 0.5 means no separation. "Within-pub" is the share of variance that lies within publications.

### 4a. Donor composition and denticity (ligand level, constant within each extractant)

| feature | C3 distribution | sc009 vs rest |
|---|---|---|
| donor class (derived from COORD) | DGA unit 156 cells (34 non-sc009 + 122 sc009); N-heterocycle 47; other amide 16 (9 + 7 DOODA); P=O 4; other 6 | N-heterocycle: 0 sc009 cells, against 47 non-sc009 cells (20 extractants, 12 chemotypes, 13 publications) |
| coord__donor__n_N_aromatic | 0 to 8, 0.4 % missing | 0 in every sc009 cell; AUC 0.26 |
| coord__motif__n_dga_unit | 0/1 | AUC 0.80. Equals 1 in 21 of 23 sc009 extractants, but also in 19 non-sc009 DGAs |
| coord__donor__n_O_ether | 0 to 6 | sc009 takes only 1 (DGA) or 2 (DOODA); AUC 0.78 |
| chem__dentate (recipe) / coord__arm__estimated_denticity_per_pocket | median 3; 2-7 / 1-10 | sc009 takes only 3 or 4 |
| **lig2d__hc__n_substituted_alpha_carbons** (`ligand_descriptors.py:130, 453-458`) | 0 to 2 | **0 for all 23 sc009 extractants.** Above 0 for 10 non-sc009 DGAs (16 cells, 4 chemotypes: sc027, sc028, sc020, sc021) and for 2 malonamides. It is in LIG2D but **not in LEAN209, TOPO39 or STERIC7.** |
| lig2d__hc__n_aryl_on_N | — | DMDPDGA only (sc076, 6 cells) |
| lig2d__hc__chain_len_mean / MolLogP | chain length 0.5-12, 13.1 % NaN (non-amides) | AUC 0.78 / 0.68 |

The donor census does not separate sc009 from the other DGAs. Only backbone α-substitution, N-aryl groups and ring backbones do, plus partly chain length. Inside sc009 the donor counts are constant apart from DOODA (2 extractants, 7 cells, 1 publication).

### 4b. Acidity and anion of the aqueous phase

* **Anion** (acid one-hot = `geom_cond__acid_class`):
  * nitrate 170 cells (91 non-sc009 / 79 sc009);
  * chloride 34 (9 / 25). The 9 non-sc009 cells cover 9 chemotypes and 5 publications, and 6 of them are DGAs from mixed DGA papers;
  * carboxylate (malonic, lactic, tartaric, citric) 19, **all one extractant in one publication** (TODGA, pub_b0b3296ec5);
  * sulfate 6, also one extractant in one publication (DHD2DGA, pub_c8195b4ca7).
  * So non-sc009 is 91 % HNO3.
* **cond__acid_concentration_M:** 0.001 to 7.4 M, median 1.5, 43 distinct values, no missing. AUC 0.52; within-pub share 0.38.
* **Recodes that add nothing:** `geom_cond__nitrate_activity` and `acid_strength_bin` are deterministic bins of [acid] (nitrate_activity = 'none' exactly when the acid is not nitric: 1,091 = 1,091 bundle rows). `inner_sphere_anion` = nitrate exactly when the acid is nitric and [HNO3] >= 0.05 M.
* **Variable the frozen bench cannot see — aqueous agents.** The gen19 archive field `complexant_name_raw` joins to all 3,871 cohort rows through safe_exp_id → source_record_id. Joined log D, metal and extractant concentration agree 100 %.
  * **19 C3 cells, all sc009, contain aqueous agents.** They come from 3 publications (pub_d68ed8c8cf, pub_1605e436d7, pub_457a2adba4) and carry 174 query pairs.
  * The agents are Amicacid, BTP-4Me, DOODA(C2), DTPA, NaNO3, TEDGA and malonamide.
  * **NaNO3 is present in 11 of these cells**, so total nitrate is not equal to [HNO3] there.
  * **10 cells pool two or more different aqueous systems** under one condition key. For example, b4fd2a49bcbf3d25 (TDdDGA) has 9 agent/concentration signatures and replicate_sd_median 1.31. 48baeebc441c0964 (TODGA) pools NaNO3 at 0.54, 0.88, 1.15 and 1.73 M.
  * gen19 `nitrate_concentration_M` and `holdback_*` are 0 % populated.

### 4c. Solvent, diluent and extractant concentration

* **diluent family** (`geom_cond__diluent_family`, a recode of the 44 one-hots): aliphatic 141 (44 / 97), aromatic 40 (19 / 21), chlorinated 21 (19 / 2), other 17, alcohol_modifier 5, nitroaromatic 5 (non-sc009 only).
* **gen7 recovered solvent physics** (`runs/gen7_architecture/cache/recovered_raw.parquet`; `src/lanthanide_separation/gen7/recovered.py:285, 345`; target-free, parsed from the upstream Solvent_Name). It is not in bench.frames.
  * It is parsed for 229 of 229 C3 cells.
  * eps: 2.0 to 46.7, median 2.38. log_eps AUC 0.26; logP AUC 0.73; polar_fraction (alcohol/TBP modifier share) has 6 distinct values. Within-pub share 0.04 to 0.30.
  * Phase modifiers hidden in diluent names, such as kerosene_0_7_1_octanol_0_3 (15 non-sc009 cells), are captured only here. The additive one-hot fires for 1 C3 cell.
* **cond__extractant_concentration_M:** 0.0003 to 0.7 M, median 0.1, 22 distinct values, no missing. AUC 0.52. **Within-pub share is 0.056, so it varies almost entirely between publications.** Values <= 0.01 M occur only in non-sc009 cells (19 cells, 6 extractants).
* **Others:** temperature 15-55 °C (2.2 % missing); contact time 34.5 % missing; metal concentration 15.7 % missing. The gen19 phase ratio is 1.0 wherever it is populated.
* **Within-series designs:** 26 of the 99 (extractant, publication) groups in C3 have more than one cell (156 cells). Acid concentration varies inside 15 of these series (10 sc009, 5 non-sc009), extractant concentration in 6, diluent in 7, acid type in 5.
* **Pooled solvent cell:** a9cb3ebbf14bf055 (TODGA) pools Isopar L with 10 vol% and with 50 vol% Exxal 13.

### 4d. Identifier risk of a small block (verified)

| block | distinct vectors in C3 | publication-unique cells | 1-NN publication accuracy |
|---|---|---|---|
| 7-column ligand block (N-donor, DGA unit, α-sub, N-aryl, ether O, amide O, dentate) | 25 | 12 % | 0.27 |
| 4-column condition block (is_NO3, is_Cl, logH, logL) | 108 | 66 % | 0.66 |
| same + solvent (log_eps, polar_fraction) | 145 | 85 % | 0.77 |

Even in small form, continuous condition columns remain a publication fingerprint.

## 5. Implications (*conjecture*, not tested)

* When sc009 is the held-out fold, the training rows that vary the anion are only 9 HCl non-sc009 cells. An anion × slope term is therefore learnable for nitrate vs chloride at most. Carboxylate and sulfate are single-publication effects.
* The non-sc009 DGA-unit training pool (34 cells) is 16/34 α-substituted, the reverse-selectivity class according to the gen14 report. A donor-class × series term learned off sc009 would mix the two DGA classes unless α-substitution is included.
* The hidden aqueous agents and the pooled cells are concentrated in sc009. They might contribute to the sc009 residual or bias. They also make up 7 of the 19 replicated C3 cells, including 4 of the 6 largest replicate SDs, which may inflate the training-fold replicate noise s2 (gen20 `cov.py`).

Scratch outputs: `cells_kmax.csv`, `cell_candidates.pkl` (per-cell candidate table), `cand_dist_C3.csv` (all numeric summaries), `g19_joined_rows.pkl` (row-level gen19 join).

## key facts
- C3 = k_max>=3 AND non-empty strict-adjacent query: 229 cells / 79 extractants / 38 chemotypes / 39 pubs / 1,802 pairs; k_max>=3 alone gives 248 (19 no-query cells dropped at g21/cache.py:49).
- sc009 in C3: 129 cells, 23 extractants, 15 pubs, 882 pairs (48.9%); it is the unsubstituted tetraalkyl-DGA cluster plus DOODA(C8/C12); 35 other DGA-family cells (19 extractants, 13 chemotypes) are outside sc009.
- No bench.frames block reads log D, replicate SD or coef (features.py:95-149, forbidden tokens cohort.py:64-71); in an empirical screen the largest |Spearman| of any feature against label- or availability-derived quantities in C3 is 0.60.
- Availability-dependent, not label-dependent: recipe__coreCN/DENTATE are per-cell modes of a metal-dependent Architector recipe (cohort.py:195-197); coreCN varies inside 200/229 C3 cells and follows the heavy-metal fraction measured; it enters massact__logL_x_coreCN (features.py:87-89).
- DONORS chem__dentate/core_cn/n_ligands/n_fill equal the first bundle row of each SMILES (190/190; first-row metal Eu for 106), set at gen6 chemistry.py:409,471-477; chem__core_cn differs from recipe__coreCN in 157/229 C3 cells.
- COND is a publication fingerprint: 100% of cells have a COND vector unique to one publication; 1-NN publication accuracy 93.7% (all) / 93.0% (C3), 44.1% if the neighbour must be another extractant.
- Ligand blocks are constant within extractant; LIG2D has 90 distinct vectors for 90 extractants (perfect extractant ID); chemotype itself is single-linkage Tanimoto>=0.7 on the ECFP block (levels.py:66,199).
- G14 = L2 logistic direction on TOPO39 only (17 coord__arm + 22 coord__dist; dirbench.py:83; gen15 arms.py:32-38,102-105) x training-mean |amplitude|; it is the only chemistry in gen21, via zero_g14 and the DOPT_Q covariance. HBK@ML/RIDGE2/LINE use no features.
- LEAN209 = COND64+MASSACT8+PHYSCHEM10+DONORS13+COORD114 (amplitude_bench.py:45, dirbench.py:104); gen15 condfe already built NUMERIC14/PHYS4 condition recodes (conds.py:130,137), worth +0.009 and shrunk to 0 in 90% of folds.
- lig2d__hc__n_substituted_alpha_carbons is 0 for all 23 sc009 extractants and >0 for 10 non-sc009 DGAs (16 C3 cells, 4 chemotypes); it is in LIG2D but not in LEAN209, TOPO39 or STERIC7.
- Donor census does not separate sc009 from other DGAs (the donor counts are constant within sc009 except DOODA); n_N_aromatic is 0 in all sc009 cells, against 47 non-sc009 N-heterocycle cells.
- Anion support: non-sc009 is 91/100 HNO3 and 9 HCl; carboxylate (19 cells) and sulfate (6) are each a single sc009 extractant in a single publication; geom_cond__nitrate_activity, acid_strength_bin and inner_sphere_anion are deterministic recodes of COND.
- Hidden aqueous agents (gen19 complexant_name_raw): 19 C3 cells, all sc009, 3 pubs, 174 query pairs; NaNO3 in 11 cells; 10 cells pool two or more aqueous systems under one condition key (e.g. b4fd2a49bcbf3d25, 9 signatures, replicate SD 1.31).
- gen7 recovered solvent physics (rec__solvent_eps/log_eps/logp/polar_fraction) is parsed for 229/229 C3 cells and lies outside bench.frames; extractant concentration varies almost entirely between publications (within-pub variance share 0.056).
- A small condition block (is_NO3, is_Cl, logH, logL) still identifies publications at 1-NN 0.66 (0.77 with solvent added); a 7-column ligand block has only 25 distinct vectors in C3 (1-NN publication 0.27).

## open
- Should the 19 aqueous-agent cells (10 of which pool heterogeneous aqueous systems) be flagged or excluded? Under the round-3 rules that is a data-version change and would be a separate mode, not the primary.
- May features from outside the frozen bench (gen7 rec__solvent_* via safe_exp_id, gen19 complexant/NaNO3 flags, gen19 mechanism labels) enter a primary-mode chemistry block, given that they are target-free but absent from the cohort manifest?
- Is chem_family (gen6 rule-based, target-free, but listed in FORBIDDEN_FEATURE_EXACT) allowed as a feature, or must donor class be rebuilt from COORD/LIG2D?
- Why does sc027 cell 0ff595d576155f6c have replicate_sd_median 1.34 when no recorded field varies except ini_comp? Is there a hidden duplicate series?
- Is gen20/gen21's training-fold replicate noise s2 biased upward because 7 of the 19 replicated C3 cells pool different complexant or solvent conditions?
- Not tested here: whether any of the candidate interactions (alpha-substitution x series region, nitrate vs chloride x slope, solvent polarity x curvature) carries transferable signal under BP/BPG; this inventory makes no label-based claims.