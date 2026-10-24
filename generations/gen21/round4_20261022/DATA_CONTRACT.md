# Data contract (stage 0)

## Data version V1 (frozen; historical comparability)

- **Source.** The frozen gen13 cohort loaded by gen20's `load_bench`: 521 cells × 14 lanthanide log D.
- **Sealed digests** (`manifests/stage0_manifest.json`; equal to round 3's where round 3 has them):
  - cohort `cohort_exact.parquet`;
  - the log D matrix;
  - the bench fields basis, coef, groups and n_obs;
  - the BP / BPG dev fold plans;
  - the 50 dev fold-cache contents;
  - the gen19 copy-group CSV and the cohort's publication → copy-group map.
- **V2 / V3.** V2 (round 3) and any source-corrected V3 are separate benchmarks: every prior, covariance and cache is rebuilt on them, and V2-5's value-based removal is not inherited.
- **V1 is never edited.**

## Splits (written out; code in gen13sep / gen20k, unchanged)

- **Dev seeds:** 104729, 130363, 155921, 196613, 262147.
- **Outer folds.** Chemotype-grouped 5-fold per seed (`gen13sep.splits`). Every cell is a test cell exactly once per seed.
  - **BP:** training cells whose publication appears among the fold's test cells are removed.
  - **BPG:** training cells whose gen19 cross-publication copy group (`data_audit/leakage_publication_components.csv`, column `group_cross_publication_copy`) appears among the test cells are also removed.
- **Inner blocks.** 4 chemotype-grouped blocks of the outer training cells, seed `seed*31 + fold*7919 + 17`.
  - The design mask is re-applied between each block's training and validation cells.
  - A block whose masked training set has fewer than 20 cells is skipped by the g21 cache (see "Orphans").
- **Isolation evidence:** `results/leakage/isolation_<design>.csv`. Overlap is zero for cells, chemotypes and publications everywhere, and zero for copy groups under BPG. BP crosses copy groups in 10 of 25 outer folds and 23 inner blocks.

## What a predictor may see for a held-out cell

`g21.evaluate.Observed`:
- the log D of the **four read metals** at k = 3 (`y_masked`; other metals NaN);
- the gen20 star support;
- the metal identities;
- the strict-adjacent query set.

A predictor also receives quantities fitted on the outer training fold (cached G14 prior / covariance and gen20 inner selections, plus the fitted HBK, R14, B1 and B2 coefficients). Methods must not read `block["Y"]` or the bench.

The prediction intervention rebuilds every method on a bench copy whose unread held-out values are replaced, so a closure over the bench's log D is caught. The frame's label columns and the gen13 coefficients are not scrambled in that check. No stage-0 predictor reads them; fits are covered by fit poisoning.

## What a fit may use

**Allowed:**
- the log D of outer-training cells only;
- their chemotype groups and observed-metal counts (for gen13 chemotype-balanced cell weights);
- the cached nested read sequences;
- the inner blocks' covariances;
- the outer-training covariance (for orphan sequences).

**Inherited disclosure.** Each inner block's noise variance includes gen20's noise factor, which gen20 selected on inner block 0's validation labels. So λ selection is nested for the covariance, not for that factor. This stays inside outer training.

**Excluded for every stage-0 reference:**
- replicate statistics;
- gen13 coefficients;
- missingness columns;
- identifiers;
- chemistry metadata;
- any held-out label.

**Later slots** declare their exact feature list, its provenance and a publication-fingerprint check before scoring.

## Training masks and weights (B1, B2; round 3's R14 convention)

**Masks.** Each outer-training cell with `k_max >= 3` is masked at k = 3 with:
- its nested DOPT_Q set (recomputed with the inner block's own covariance for inner fits);
- its UNIFORM set;
- 4 RANDOM legal sets, seeded by `sha256(cell_id|r3train|r)`.

**Training rows.** The rows are its strict-adjacent queries that have at least one hidden end; a training query with both ends read raises an error. Only real readings are used.

**Weights.**
- Every row of a mask carries `cell_weight / n_masks`, where `cell_weight` is gen13's chemotype-balanced weight (equal per cell within a chemotype), computed over all outer-training cells before the `k_max >= 3` filter.
- So the number of masks does not change a cell's weight, but its number of hidden queries does: 2–12 rows per mask.
- This deviates from the brief's §6 per-cell normalisation and is kept so that B1 reproduces R14.
- Every later slot declares its weighting. A slot using per-cell normalisation is compared with B1 / B2 refitted under the same weighting, registered as matched-weighting sensitivity references.

**Splits of masks.** All masks of a cell stay in that cell's split.

## Orphans (round-4 rule)

Outer-training cells that appear in no inner block (because the cache skipped a block) get label-free cell records:
- metals, queries and `k_max` from the bench's measured pattern;
- DOPT_Q and UNIFORM sequences from the outer-training covariance.

They join every inner fit whose training set contains them, with masks recomputed with that block's covariance, and the final refit. On the dev caches this affects BP and BPG seed 155921 fold 2 only: 235 orphan records, 148 of them C3, out of 183 C3 training cells.

## Subgroups (frozen, label-free)

- **sc009 / non-sc009:** by chemotype id.
- **Donor-defined DGA / non-DGA** (`r4/donor.py`): the extractant SMILES contains the diglycolamide core `[NX3]C(=O)[CX4]O[CX4]C(=O)[NX3]`.
  - This gives 40 of 79 C3 extractants (21 in sc009, 19 in other chemotypes).
  - The two sc009 DOODA diamides are not DGA.
  - The result is identical to the curated gen13 `chem_family = diglycolamide`, which was not used to build the rule.
- **One end read / both hidden; pair types.**

## Groups used for uncertainty

- **Chemotype:** 38 in C3.
- **cluster_cpg:** connected components of chemotype, publication and copy group. There are 13 in C3, and every C3 publication must have a copy group, or the code raises an error.
- **New confirmation data:** a new publication forms its own cluster unless the stage-1 provenance inventory links it to an existing chemotype, publication or copy group. Those links are frozen before its labels arrive and fed in by extending the copy-group map.

## Block A features (amendment A1; frozen)

Block A (S01–S04, S21) uses exactly one chemistry input: the extractant's **donor class**, computed from the cohort's extractant SMILES.

**Classes**
- **DGA:** diglycolamide SMARTS (`r4/donor.py`).
- **SOFT_N:** otherwise, if the extractant's InChIKey has `mechanism = SOFT_N_DONOR` in gen19's curated `descriptors/extractant_components.csv` (LF digest sealed).
- **OTHER:** everything else.

**Coverage**
- All 79 C3 extractants are classified: 40 DGA / 19 SOFT_N / 20 OTHER.
- Cells: 157 / 46 / 26. Pairs per seed: 1,126 / 512 / 164.

**Publication fingerprint check.** The class identifies a C3 cell's publication with accuracy 0.183, against a majority baseline of 0.087 (`results/provenance/fingerprint_donor_class.json`). For comparison, round 3's 4-column condition block reached 1-NN 0.66.

**Not used:** conditions, concentrations, diluents and other descriptors.

**Shapes.** gen13's frozen physics basis and orthonormal radius polynomials; the shape vectors are sealed in the manifest. Weights and masks are B1's.

**Training-only data sources.** None beyond V1. The auxiliary series counted in `PROVENANCE.md` are not used by block A.

## Block B features (amendment A2; frozen)

Block B (S05, S06, S09, S22) builds one row per unread metal of a training situation or held-out cell.

**Metal / read-set / support summaries**
- metal x and 4f count;
- distances to the bracketing reads, interpolation fraction and extrapolation side;
- the bracketing segment's slope and the least-squares slope of the four reads (from the read values only);
- B1's coherent Gd feature, and the feature times β;
- round 3's 6 read-geometry features.

**Chemistry**
- block A's donor class;
- 8 RDKit descriptors of the extractant: MolWt, MolLogP, TPSA, NumHDonors, NumHAcceptors, NumRotatableBonds, RingCount, NumAromaticRings.
- **Publication fingerprint check** (`results/provenance/fingerprint_blockB.json`):
  - leave-one-out 1-NN accuracy 0.489, almost entirely through other cells of the same extractant;
  - 0.140 when same-extractant neighbours are excluded, which is what a held-out chemotype faces;
  - majority baseline 0.087.

**Geometry (S06 only)**
- **Values.** For the cell's own complex at metal m: mean Ln–donor distance, physical coordination number and metal partial charge.
- **Composition.** The complex's ligand count, fill-ligand count and inner-sphere anion, from `accepted_geometries.csv` through the complex's build_id.
- **Source.** `dataset with 3D structures/features/features_by_safe_exp_id.parquet`, joined through the cell's safe_exp_ids. **The metal comes from the archive record**, because the safe_exp_id prefix names the export file (for example Am_SAFE), not the metal. Entries with any missing value count as missing.
- **Representation:**
  - the value g(m);
  - the composition features;
  - the anomaly g(m) − I_S[g(S)](m), **only when every read metal used at m has a complex of the same composition** (195 of 229 C3 cells change composition across metals, gen15's trap; brief §6);
  - a coverage flag and a composition-match flag.
  - Missing geometry takes the missing branch, with the full denominator kept.
- **S09:** geometry identity from another chemotype (a derangement, per cell and metal), with S06's missingness kept.
- **S22:** training-fitted per-metal mean geometry and modal composition, with S06's missingness kept.
- **Fingerprint** (`fingerprint_blockB.json`, "geometry_block"): per-cell mean geometry identifies the publication at 0.694 leave-one-out, and **0.275 excluding the same extractant** (baseline 0.087). Geometry partly identifies chemistry; under BP/BPG that cannot leak, and S09 controls for it.
- **Limitations:** the geometry is single-conformer GFN2-xTB and extractant-level (not condition-level). No energy is used, and no geometry was chosen by fitting extraction labels.
- **Sealed inputs:** the geometry parquet, `accepted_geometries.csv`, the archive and the CatBoost version are digested in the seal manifest.

**Weights:** B1's (per unread-metal row: the cell's chemotype-balanced weight / masks).

## Block C data (amendment A4; frozen)

Block C (S11, S13–S16; S24 NOT RUN) uses B1's training situations and block B's row construction **without** its chemistry columns. Chemistry enters only through a shared encoder of the extractant: 8 RDKit descriptors, standardised on the fit's training Ln extractants (a scale below 10⁻⁶ is replaced by 1; values clipped to ±5), and the donor class.

**Ln auxiliary rows (every arm, identical).**
- **Rows:** the raw log D of every outer-training (or inner-training) Ln cell at each measured metal.
- **Features:** x(m) and q(m).
- **Weights:** the chemotype-balanced cell weight / the measured metal count.

**An(III) auxiliary rows (S13, S16; S14 see Control data).**
- **Records:** `r4/an_data.py::an3_records()`, from the gen19 archive through `gen19ct.data.load.load_archive`.
- **Selection:**
  - actinide; Am / Cm / Pu / Cf; oxidation state 3;
  - `acid_anion == nitrate` (nitric acid, the cohort's majority regime);
  - no aqueous complexant and no active (or unknown-concentration) aqueous hold-back agent; no ionic-liquid diluent (`[` in the solvent key);
  - finite log D; a parseable extractant (canonical SMILES);
  - gen19 tier MODEL; no gen19 value conflict; no suspect flag; exactly one organic extractant and no acidic co-extractant recorded as a modifier (alcohol phase modifiers kept).
- **Size:** 1,462 rows, 38 publications (35 copy groups), 112 extractants (Am 1,288, Pu 95, Cm 78, Cf 1).
- **Target:** raw log D, not a curve target.
- **Features (S13, S16):** the encoder of the extractant, plus a 4-way metal one-hot.
- **Weights:** each publication weighs the same.
- **Conditions** (acid and extractant concentration, diluent) are **not** model inputs. The auxiliary head learns an extractant-level affinity averaged over conditions inside the compatible regime. This is a declared simplification.
- **Supersedes** the stage-1 An(III) inventory in PROVENANCE.md for block C.

**Chemotype links.**
- An An extractant links to every cohort chemotype with a member at Morgan r2 / 2,048-bit Tanimoto ≥ 0.7. This is the project's `TANIMOTO_CLUSTER_THRESHOLD`, with direct one-hop links as in gen11 `pools.chemotype_assignment` and no An–An transitive closure.
- 43 of 112 An extractants have a link.
- 21 extractants (241 rows) reach sc009, sc016 or sc036 only through An–An chains. They are kept, and this is declared.

**Union exclusions.**
- **Outer:** against every non-training cell of the fold, by publication, copy group, exact extractant and linked chemotype.
- **Inner:** additionally against the outer-training cells outside the inner block's training set.
- **Applied before** any fit, tuning or weight choice.
- **Check:** `results/leakage/isolation_an.csv` recomputes the clauses without calling `excluded_by`, reusing the sealed link map. It shows 0 overlaps in 248 outer and inner checks. The tests rebuild the link map from scratch.
- **Per-fold counts:** `results/blockC_counts.json`.

**Control data.**
- **S14:** the An rows enter the Ln head at the Ln analogue of equal f count (Pu→Sm, Am→Eu, Cm→Gd, Cf→Dy). When w_X > 0, the Ln term enters with weight max(w_Ln, 0.3).
- **S15:** matched training-Ln rows (PROTOCOL §12).
- **S16:** extractant-block-permuted An targets, sampled at the same metal; 5 draws.
- **S24:** the within-series metal cycle; NOT RUN on its count.

**Sealed inputs.**
- the content digest of the An record table and of the link map;
- the archive file digest;
- the gen19 loader's directory (code digests).

**No An record is a test target.** Only held-out Ln cells are scored.

## Not yet frozen (stage 1, before use)

- the provenance inventory (DOI, table / figure, curator, stereochemistry, conditions, copy groups);
- the V3 source-correction rules;
- the auxiliary-data inclusion and exclusion rules for extra Ln (S12 is NOT RUN on its gate). The An(III) and geometry
  rules are frozen above (A4, A2).

None of these may use held-out labels or residuals.
