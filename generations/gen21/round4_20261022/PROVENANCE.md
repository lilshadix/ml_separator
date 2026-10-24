# Provenance inventory (stage 1; `scripts/r4_provenance.py`, brief §5)

**Data.** Every one of the 521 V1 cells is joined to its archive records in gen19 `dataset_all_metals/clean/master_clean.parquet`, by `safe_exp_id` → `source_record_id`. Every record is found.

**Tables** (`results/provenance/`)

| file | content |
|---|---|
| `cells.csv` | one row per cell (fields below) |
| `sources_by_doi.csv` | per DOI |
| `fold_training_<design>.csv` | what survives in training after each dev fold's exclusions |
| `auxiliary_ln.csv`, `auxiliary_an3.csv` (+ `_folds`) | candidate training-only records outside the cohort |
| `summary.json` | summary |

`cells.csv` fields:
- DOI, citation, data location (table / figure) and curator;
- ORNL-import status;
- extractant SMILES, name, InChIKey and explicit stereo;
- chemotype, donor class, copy group and source cluster;
- exact condition key and parsed conditions (acid, extractant concentration, solvent, aqueous complexant, temperature);
- metals and k_max;
- round-3 quality flags. The smoothness and low-D flags are value-derived; they are reported and never used to select rows.

## The C3 cohort (229 cells, 1,802 pairs per seed)

**Independence: few, unequal sources.**
- 39 DOIs and 39 publications; 37 copy groups.
- **13 source clusters**, Kish ≈ 2.2; the largest holds 65% of pairs.
- 79 extractants in 38 chemotypes; 174 exact-condition series.

**Curation.**
- Thomas Summers curated 150 cells, Baosen Zhang 69 (ORNL imports), Aurora Clark 10.
- 150 of 229 cells name a table or figure in the source. The other 79 have no data location; most are ORNL imports.
- 54 cells, with 529 pairs, are ORNL imports in every row.

**Stereo.** 6 extractants carry explicit stereo in their SMILES: the syn / anti and cis / trans Me2-TODGA and TDDGA variants of pub_0af6736cb1. The ORNL re-import `0ff595d576155f6c` lost the stereo (round-3 D4, V2-1).

**Conditions.** 19 C3 cells have an aqueous complexant or salt (DTPA / NaNO3 7, NaNO3 4, Amicacid 2, Amicacid + TEDGA 2, …). The frozen bench cannot see these.

**Round-3 quality flags (not used to select rows in V1).**
- 23 C3 cells hit a V2 rule: ORNL re-import, duplicate entry, hidden merge, or resolution floor.
- 21 C3 cells have a value-derived flag: a smoothness suspect, or a log D ≤ −3.5 reading.

**The two sources to recover** (results under "Source recovery" below):
- C5-BTBP, Nilsson et al. 2006, SX&IE, 10.1080/07366290600646947: pub_6a0ef8528c, 15 cells / 148 C3 pairs, ORNL-curated, no data location. Light lanthanides sit at log D ≈ −4.2, which may be a detection floor.
- DMDPhPDA, Shimada et al. 2004, SX&IE 22(2) 147–161, 10.1081/SEI-120030392: pub_8415538291, 10 cells / 120 pairs. Copy A is curated from Table 1; copy B is an ORNL re-import of bare mantissas. The 5 M Pr/Nd pair differs by exactly +1.000.

## What training keeps after each fold's exclusions (dev folds; `fold_training_*.csv`)

| | BP min / median | BPG min / median |
|---|---|---|
| C3 training cells | 54 / 178 | 51 / 178 |
| C3 training pairs | 486 / 1,364 | 450 / 1,335 |
| C3 publications | 17 / 29 | 16 / 29 |
| C3 chemotypes | 20 / 23 | 20 / 23 |
| source clusters (all training cells) | 11 / 14 | 11 / 14 |

The folds that hold out sc009 (418 test cells) train on only about a quarter of the C3 data.

## Training-only data outside the cohort (archive; `auxiliary_*`)

**Extra lanthanide series.**
- 317 archive Ln series lie outside the cohort; only **16** have at least 4 metals.
- All 16 use cohort extractants, and 14 come from cohort publications. Examples:
  - 970 further C5-BTBP records from the same paper;
  - D4GC01146E, the brief's example "new" source, which **is already in the cohort** (pub_4cba27f5f3, 8 C3 cells).
- After each fold's exclusions (same publication, copy group or exact extractant as a held-out cell), the minimum over folds is **0** series.
- **The archive holds no independent new lanthanide chemistry.** "Extra independent Ln" (S12) has nothing to train on unless new sources are added, so it fails gate G-transfer as things stand.

**An(III)** (Am, Cm, Cf, Pu(III); An(VI) and unknown states excluded).
- *Superseded for block C (amendment A4):* PROTOCOL §12 and `r4/an_data.py` define the An(III) set more strictly (nitrate, no complexant or active hold-back agent, no ionic liquid, gen19 MODEL tier, no value conflict or suspect flag, one extractant without acidic modifier; 1,462 rows / 38 publications / 112 extractants) and add chemotype links. The post-exclusion minima become 12 copy groups / 77 extractants / 557 rows (`results/blockC_counts.json`). The figures below are the stage-1 inventory.
- 280 series / 2,009 rows / 36 publications / 116 extractants. 16 series have an aqueous complexant.
- After each fold's exclusions, the minimum over folds keeps 153 series / 918 rows / **19 publications** / **95 extractants**, which passes gate G-transfer's count thresholds.
- **Not done (stage-1 inventory; superseded by A4):** chemotype isolation of these extractants. There is no chemotype assignment outside the cohort, and a transfer slot must add one before use.
- **Overlap:** 95 of the 280 series share a publication with the cohort, and 129 share an exact extractant.

## Source recovery (desk research; workflow wf_07655a20-edb)

Three researchers used public web pages only (no logins, no downloads). Each claimed source statement was then re-opened by a skeptic. Full findings and URLs are in `results/source_recovery.json`; a quote is at most 15 words.

**C5-BTBP** (pub_6a0ef8528c, 15 cells, attributed to Nilsson et al. 2006, 10.1080/07366290600646947)
- **No source-verified detection limit.** The paper is closed access. Its abstract, rebuilt word for word from OpenAlex, covers **only Am(III) and Eu(III)** in "a variety of organic diluents". It mentions no lanthanide series and no kerosene / octanol mixture.
- **Secondary, same group.** The open SKB report R-05-13 (Chalmers, Partitioning and Transmutation annual report 2004) says:
  - radiotracer runs were measured by gamma / alpha spectroscopy, and inactive La–Gd by ICP-MS;
  - "no extraction" in one C5-BTBP diluent is stated as D < 10⁻⁴;
  - heavy-lanthanide or low-D values were hard to evaluate in related work.

  This is consistent with the corpus light-lanthanide values (log D ≈ −4.24) being at or below an operational floor. It is not a verified limit for these cells or this method.
- **Attribution doubt (a hypothesis only).** The corpus cells' conditions are 13 lanthanides in 70% kerosene / 30% 1-octanol, 0.1–1.0 M HNO3, 0.005–0.015 M ligand. These fit Foreman et al. 2005 (SX&IE 23(5) 645–662, 10.1080/07366290500230735, also closed access) better than Nilsson 2006. If the cells belong to Foreman 2005, their publication and copy-group assignments are wrong.
- **Consequence.** No V1 value or censoring status changes. The 15 cells keep flag D5 (suspected floor) and gain the attribution doubt in this inventory. **Gate G-V3 fails for these cells.**

**DMDPhPDA** (pub_8415538291, Shimada et al. 2004, 10.1081/SEI-120030392)
- **Unresolved.** No public page states the 5 M HNO3 Pr / Nd D values or SF.
- **Facts the research found.**
  - The paper is closed access.
  - An AESJ 2003 meeting abstract says D rises steeply above about 3 M HNO3, so a one-decade slip at 5 M is plausible either way.
  - A JAEA patent (US8052947B2) gives an independent Eu point at 3 M, with no Pr / Nd values.
  - A 2022 citing paper (secondary) states a positive D-vs-Z sequence.
- **The likely original data source is Shimada's open-access 2004 PhD thesis** (Shizuoka University, 10.14945/00003509, PDF about 24 MB, in Japanese). It was **not downloaded**: that needs the user's explicit permission, and then reading and possibly OCR.
- **Consequence.** Copy B stays as in V1 (V2-1 quarantine applies only in V2). **G-V3 fails for these cells.**

**New data sources**
- D4GC01146E (An et al., Green Chem. 2024) is **already in the cohort**.
- **SAFE** needs a sign-up / login to download. Its homepage on 2026-10-22 showed 18,388 records over 41 metals and 324 extractants (experimental and computational extraction plus chromatography). There is no public lanthanide-specific count.
- **Open-access papers with ≥ 10 lanthanides, none in the corpus.** Mostly phenanthroline-diamide / BLPhen N-donors; plus one isomeric-DGA paper with a numeric 14-Ln table, a phosphonium-nitrate paper and an HDEHP / ionic-liquid paper. The best is Molecules 29:3548 (2024), a 14-Ln phenanthroline diamide with numeric SI table S3. Details are in `source_recovery.json`.
  - These are candidates for **independent confirmation data**, but only if the model is frozen before their values are extracted.
  - Their number (under 10 sources) is far below the about 40 independent sources that 80% power needs (POWER.md).
  - No open full-series α-substituted DGA data set and no Zenodo / figshare collection was found.

## Gate decisions from this inventory (PROTOCOL §7)

- **G-V3 (S17, S18): FAILS.** No C3 cell's value or censoring status changes on documented source evidence. S17 and S18 are NOT RUN.
- **G-transfer, extra lanthanide (S12): FAILS on counts.** The minimum over folds is 0 independent series. S12 is NOT RUN.
- **G-transfer, An(III) (S13–S16, S24): PASSES the count thresholds** (minimum over folds: 19 publications, 95 extractants, 918 rows). Under A4's definition the count part still passes (min 12 copy groups / 77 extractants / 557 rows), and chemotype links are defined in A4 (PROTOCOL §12). It still needs the inner-signal gate before scoring.
