# E3: Data inventory and the data lever (round-5 design evidence)

Written 2026-10-23 for the round-5 design. This is **exploratory development evidence**. Nothing here is a registered result, nothing was sealed, and no withheld seed was opened.

**Inputs (read-only)**
- the V1 bench (`gen20k.data.load_bench`);
- round-4 provenance tables (`round4_20261022/results/provenance/*.csv`, `results/power/power_grid.csv`, `results/source_recovery.json`);
- round-4 out-of-fold predictions (`round4_20261022/predictions/REF_BPG.parquet`, `REF_BP.parquet`);
- the gen19 archive `dataset_all_metals/clean/master_clean.parquet`, via `gen19ct.data.load.load_archive`.

**Computation.** Three scratch scripts, one process at a time, each under 2 minutes. No model was trained. The scripts are kept outside the repository (hard rule: only this file is written here). Paths are in the Appendix.

**Tags.** **[M]** means measured here or copied from a cited file. **[I]** means my interpretation.

---

## 0. Bottom line

1. **[M] Relaxing the cohort rules adds no independent source clusters inside this repository.**
   - Under the official clustering (gen6 chemotype map, plus publication and copy group), every extra lanthanide adjacent pair falls into one of C3's existing 13 clusters. That holds for the 149 V1 cells outside C3 (k_max < 3, 312 strict-adjacent queries) and for the 32 archive conditions outside V1 (58 pairs).
   - A looser direct rule (link to C3 only through publication, copy group, exact extractant or a nearest-neighbour Tanimoto ≥ 0.7) leaves 2 unlinked cells (16 pairs, 2 publications). The official map puts both of them into C3 clusters.
   - Dropping chemotype linking altogether gives 9 "new" publications. Seven of them are TODGA / sc009 chemistry.
2. **[M] The archive is exhausted for adjacent log SF.**
   - Of 2,476 strict-adjacent pairs in archive-native conditions (all acids, diluents, complexants and multi-extractant systems), **2,418 (97.7%) are already in V1 conditions**.
   - Of the 2,637 canonical Ln rows outside V1, **2,293 sit in conditions where only one lanthanide was measured** (1,938 conditions, 71 publications; Eu alone has 1,261 rows, mostly Am/Eu studies). Such rows cannot give an Ln/Ln separation factor.
3. **[M] Power consequence.** The realistic number of new independent clusters available without new data is G ≈ 0–2 from the repository, plus about 3–5 from verified open-access literature (§2.4).
   - At G = 5, POWER.md's heterogeneous run 4 gives a probability of 0.04 of passing the interval rule for a B1-sized gain (δ = 0.034), and 47% of sets fail the Kish ≥ 3 floor.
   - At G = 10 that probability is 0.10.
   - **Confirmation stays impossible without new laboratory or literature sources, whatever cohort rule is used.**
4. **[M] The binding constraint is the cluster definition as much as the data.**
   - The same 1,802 C3 queries have Kish 18.8 by publication, 3.85 by chemotype and **2.24 by source cluster**.
   - The intraclass correlation (ICC) of the per-cell B1 gain is 0.44 by publication and 0.35 by chemotype. Within the giant sc009 chemotype it is 0.34 between publications.
   - Under the current rule, **a whole new laboratory campaign published together is ONE cluster**, whatever the number of systems in it.
5. **[M] B1's gain is concentrated in the two data-quality-pending sources.**
   - C5-BTBP (pub_6a0ef8528c) holds 8.2% of rows and **34.6% of B1's SSE gain over B0**.
   - DMDPhPDA (pub_8415538291) holds 6.7% of rows and 17.4% of the gain.
   - The giant DGA cluster holds 65% of rows and 26.8% of the gain (BPG; BP is the same within 0.02).
6. **[M] Pending source fixes are worth as much R² as four rounds of modelling.**
   - The 43 C5-BTBP light-end rows (La/Ce, Ce/Pr, Pr/Nd; 2.4% of rows) hold **16.4% of B1's SSE**.
   - The single DMDPhPDA 5 M Pr/Nd row holds 1.8% of it.
   - Without those 44 rows per seed, B1 goes 0.452 → 0.486 and B0 0.418 → 0.447. This is a stakes estimate, not a correction (value-informed; §4).
   - **Nine of the 13 C3 source clusters consist only of ORNL-imported cells with no data location** (327 of 1,802 queries).
7. **[M] One more measured metal is worth more than any modelling gain so far.**
   - On the full C3 query set (BPG, 5 seeds), the reference form with the fixed Gd constant (HIST_D7 = B1 − 0.001) goes from 0.451 at 4 reads to 0.495 with a random *legal* 5th read and **0.503 with a label-free max-min-gap 5th read**.
   - It reaches **0.530 with a 6th read**. B0 gives 0.471 and 0.499 on the same rows.
   - This is the 5-metal ladder row that round 3 did not have (§6).
8. **[M] Noise-ceiling arithmetic.**
   - var(y) = 0.0493 and B1's MSE = 0.0270.
   - 0.50 is unattainable if the laboratory repeatability sd of adjacent log SF is ≥ 0.157. B1 is already at the floor if it is 0.164.
   - The only number available, the transcription double-report bound of 0.086, would allow a ceiling of 0.85.
   - **A small repeat campaign therefore decides whether 0.50 is physically reachable at all.**
9. **[M] The Gd dip is chemistry-dependent in the raw data.**
   - Gd minus the Eu–Tb radius interpolation (134 V1 cells) has a median of −0.064 for DGA and −0.175 for non-DGA.
   - It is −0.29 for the BTBP cluster and −0.17 for DMDPhPDA.
   - Gd-pair queries have Kish 2.33 over clusters. The one effect B1 exploits rests on about five non-DGA clusters.

---

## 1. What data exist, and what round 4 used

| source (location) | size | round-4 use | can it add independent Ln adjacent-log-SF sources? |
|---|---|---|---|
| V1 bench (gen13 cohort; `dataset with 3D structures/dataset.parquet` + gen6 provenance) | 521 cells, 3,871 rows; C3 229 cells, 1,802 queries / seed, 39 publications, 38 chemotypes, 13 clusters (PROVENANCE.md) | C3 only (all scoring); V1 non-C3 only as training situations (B1 / B2 masks need k_max ≥ 3, so effectively unused) | **No.** Non-C3: 149 cells with ≥ 1 strict query, 312 queries, 15 publications (6 not in C3), 2 chemotypes not in C3, **0 new clusters** [M, §2.1] |
| Archive `dataset_all_metals/clean/master_clean.parquet` (SAFE export) | 16,770 records, 41 metals; Ln 8,053 records (MODEL 6,495; TARGET_ONLY 1,396, of which 1,382 are D-redundant duplicates; NO_TARGET 162) | provenance joins; An(III) set for block C (1,462 rows, 38 publications); auxiliary Ln series counted (S12 gate failed: 0 series per fold) | **No** (≤ 2 cells / 16 pairs unlinked by the direct rule; 0 by the official map) [M, §2.2] |
| Archive condition fields the bench cannot see | complexant 991 rows, nitrate 1,104, hold-back 1,072, modifier 797, organic acid 1,962, phase ratio 8,153, contact time 9,488, metal concentration 7,412 (schema.md) | not used (bench sees the 64-column gen13 key only) | No new sources. 19 C3 cells have an aqueous complexant; these rows carry **negative** B1 gain (−0.44 SSE / seed, §4) |
| An rows (Am 2,104; Cm 160; Cf 23; Pu 1,019; U 1,609; Th 839; Np 496) | An(III) rows that share an archive condition with an Ln: 1,052 (1,002 with Eu), in 844 conditions / 29 publications; only **44** of those conditions have ≥ 3 Ln [M] | block C: S11 = B1; S13 gate failed 3/50 | No. They give An/Ln, not Ln/Ln. |
| Y / Sc | Y 49 rows (1 extractant system, 4 publications; 41 rows share a condition with Ln); Sc 10 rows (1 system) [M] | unused | No. At most a radius-interpolation check at the Y position (tiny) |
| 3D structures (`dataset with 3D structures/features/`) | 89.0% of C3 queries have both ends' geometry; 165 / 229 C3 cells have it for every metal; 195 / 229 change complex composition across metals; VR inputs for 1,155 complexes; persistence images; xTB reference queue | S06 (3 scalars + composition) = worse than B1 (−0.006 / −0.011); encoder slots gated out | No. These are features, not labels; power is unchanged. Unused: PI images, VR graphs, polyhedron descriptors |
| Descriptor tables (gen13 `features/`, gen15 `exp/external/data/`, gen19 `descriptors/`) | topo39 donor topology (gen13's direction predictor, macro accuracy 0.77–0.82), LEAN209, gen19 `extractant_components.csv`, `metals.csv` | donor class (3 levels), 8 RDKit descriptors | No (features) |
| logK Ln series (BidentateGNN, MIT) | on disk only as **per-ligand slope / curvature / r²** for 273 ligands (`gen13_separation/features/logk_external_series.parquet`, `gen15_curve/exp/external/data/side_K_logk.parquet`); the per-metal values (4,433 records) are **not on disk** [M] | unused (gen13: Spearman +0.00 with amplitude; gen15 `external` arm 0.552, worse) | No new Ln-extraction sources. Per-metal logK would be needed for a Gd-break prior, and that needs a download (user permission) |
| Am/Eu literature set (on disk: `gen15_curve/exp/external/data/am_eu_literature_data.xlsx`, `side_A_amdeu.parquet`, 85 ligands) | Am/Eu log SF | unused | No (An/Ln) |
| Strain repo (`D:/ml_separator_refs/lanthanidestrain_025f808`) | its 5,992 Ln rows = the bundle (EVIDENCE_INVENTORY D1) | provenance only | No (same records) |
| SAFE live (safe.lanl.gov, login) | 18,388 records on 2026-10-22 vs 16,770 in the archive: **≥ 1,618 more (+9.6%)**. If the archive's Ln share (48%) held, about 780 Ln records [I]. The 2025 snapshot showed 8,716 | not accessible | **Unknown.** Needs the user's export, then a DOI-level novelty count before any value is read |
| Open-access literature (source_recovery.json Q3-1..Q3-7, plus 2 unconfirmed) | 7 verified ≥ 10-Ln papers not in the corpus; 4 give values only as plots | not used | **About 3–5 new clusters** (§2.4) |
| Not on disk | figshare 29430059 (xTB+DFT, 5,356 complexes), JACS Au 2022 SI (1,202 log D, no SMILES), NIST SRD 46 port, Sep. Purif. Technol. 2025 / 2026 tables (author only) | — | Only the last two are Ln-extraction labels, and their independence from SAFE is unknown |

**Reading [I].** Everything in the repository that carries Ln/Ln labels is already inside the 13 C3 clusters, or is a single-lanthanide condition. The repository's unused data are features (geometry, descriptors, conditions) or An labels. Round 4 already tested both kinds against B1 and found nothing (S05, S06, S11, S13 gate).

---

## 2. How many independent sources would relaxed rules add?

### 2.1 Relax k_max ≥ 3 inside V1 (fewer measured metals, partial series) [M]

- **Size.** 292 V1 cells are outside C3. By number of measured metals they split as 2: 138, 3: 62, 4: 32, 5: 16, 6: 44. The 5- and 6-metal cells fail k_max ≥ 3 because the reads must not be strict neighbours.
- **Usable cells.** 149 of them have at least one strict-adjacent query: 312 queries (+17% over C3), with k_max 0 / 1 / 2 for 47 / 40 / 62 cells.
- **Independence.**
  - Publications: 15 (6 not in C3); chemotypes not in C3: 2.
  - Source clusters: 2, both already C3 clusters. New clusters: **0**.
- **Example.** NTAamide(C8) (pub_bef16e560c, a 14-Ln curve in the archive) is split by V1's exact key (contact time / metal concentration) into 2- and 4-metal cells. It sits with TODGA in the giant cluster.

### 2.2 Relax to archive-native conditions (any acid, diluent, complexant, multi-extractant, non-bundle rows) [M]

**Condition key.** Extractant system key, publication, acid signature and concentration, organic acid, extractant concentration, solvent key, modifier (name and concentration), complexant (structure and concentration), hold-back (structure and concentration), nitrate and temperature. This is gen13's RELAXED analogue: contact time and metal concentration are dropped. The exact variant, which keeps them plus phase ratio, gives 43 pairs outside V1 instead of 58.

**Input rows.** Canonical, non-duplicate Ln rows only: 6,508.

| relaxed archive conditions | count |
|---|---|
| conditions / with ≥ 2 Ln / with ≥ 1 strict pair | 2,495 / 555 / 377 |
| strict-adjacent pairs, total / in V1 conditions / outside V1 | 2,476 / 2,418 / **58** |
| conditions outside V1 with a strict pair / with ≥ 5 Ln | 32 / 11 (4 publications; 17 synergistic two-extractant, 9 with a modifier, 6 single-extractant; HNO3 28, HCl 4) |
| conditions with ≥ 5 Ln and a strict pair (vs C3's 229 cells) | 285. This re-cuts the same sources; it adds no clusters |
| Gd triads (Eu, Gd, Tb plus a lighter and a heavier flank) | 190, of which 178 are in C3 and 1 is unlinked |

**Non-C3 candidates.** All cells outside C3 with a strict pair: 130 cells, 271 pairs, 16 publications.
- **Linked to C3:** 128 cells. Links: publication 90, copy group 90, exact extractant 127, nearest-neighbour Tanimoto ≥ 0.7 128.
- **Unlinked by this direct rule:** 2 cells, 16 pairs. Both are in V1, and the official gen6 map assigns both to existing C3 clusters:
  - 10.1080/07366299.2013.836422: NTAamide, 14 Ln, nearest Tanimoto 0.69 → giant cluster;
  - 10.1080/07366298608917877: CMPO-type, 1986, 6 Ln, nearest Tanimoto 0.67 → g:pub_f926473989.
- **With publication / copy-group links only (no chemotype link):** 9 publications, 40 cells, 96 pairs. Only 3 of them have a ≥ 5-Ln cell, and 7 of the 9 use sc009 / TODGA structures.

### 2.3 Other acids and diluents [M]

There is nothing to relax: V1 already has no acid or diluent filter.
- C3 acids: HNO3 170 cells, HCl 34, malonic acid 8, H2SO4 6, tartaric 4, lactic 4, citric 3.
- C3 diluents: dodecane 77 cells, toluene 30, Isopar L / Exxal 24, kerosene / octanol 18 + 8, chloroform 16, and others.
- The archive's non-HNO3 multi-Ln conditions outside C3 are 9 cells (HCl 4, H2SO4 4, malonic acid 1), all linked.

### 2.4 External open-access literature [M for Tanimoto values; I for SMILES]

I rebuilt the SMILES from the names given in source_recovery.json. They were not read from the papers.

| candidate | nearest C3 Tanimoto (cluster) | new cluster under the current rule? |
|---|---|---|
| IJMS 24:5569, L2 (4,7-Cl2-phen-bis(pyrrolidinyl) diamide) | **1.000, the exact C3 extractant** (sc086, g:pub_0fd6ab4e40) | no; L3 (F2, 0.684) is linked through the same paper |
| RSC Adv 12:790, L^II (N,N-dimethyl-N′,N′-dioctyl DGA) | **1.000, the exact C3 extractant** (sc009, giant) | no |
| Molecules 29:3548 (Cl2-phen diamide, Et / p-hexylphenyl) | 0.623 (sc059, phen cluster) | yes, unless the gen6 single-linkage map joins it |
| RSC Adv 8:32044 ([P66614][NO3], undiluted, 70 °C) | 0.379 | yes (out of domain) |
| SERDJ 30:121 (HDEHP in IL / toluene) | 0.469 | yes (a class the corpus lacks) |
| RSC Adv 9:26537 (BLPhen) | not computed; same family as the corpus BLPhen paper | probably no |
| IJMS 23:15538 (four phen diamides, unspecified) | not computed | unknown |

**Literature total ≈ 3–5 new clusters.**

### 2.5 What this does to power (POWER.md, `results/power/power_grid.csv`, scenario empirical_B1) [M]

Probability of passing the interval rule, with δ = 0.034 (B1-sized):

| G new independent clusters | run 4, heterogeneous: pass | run 4: share of sets failing Kish | run 2, optimistic: pass |
|---|---|---|---|
| 5 | 0.04 | 0.47 | — |
| 10 | 0.10 | 0.26 | — |
| 20 | 0.135 | 0.32 | 0.33 |
| 40 | 0.27 | 0.02 | 0.84 |
| 80 | 0.34 | 0.00 | 1.00 |

- **Achievable without new data:** G = 0 (repository, official rule) up to about 2 + 5 = 7 (direct rule plus literature, all assumed truly independent). That is the G ≈ 5–10 rows: **power ≤ 0.10**.
- **Pooling new data with the corpus does not help.** If new clusters of p queries are added to the C3 pool, pooled Kish rises only slowly, because the giant cluster dominates:

  | p queries per new cluster | G = 20 | G = 40 | G = 80 |
  |---|---|---|---|
  | 12 | 2.88 | 3.58 | 5.23 |
  | 36 | 4.32 | 7.01 | 14.1 |

  Bringing the giant cluster's share below 50% needs 538 pairs outside it; below 25% needs 2,878. A confirmation set must be new data only, as POWER.md assumes.

### 2.6 The cluster definition is a design lever [M + I]

**[M] Kish of the C3 queries under different groupings**

| grouping | Kish |
|---|---|
| publication | 18.8 |
| chemotype | 3.85 |
| cluster_cpg (chemotype ∪ publication ∪ copy-group components) | 2.24 |

The giant cluster joins 19 publications and 14 chemotypes through chemotype links.

**[M] ICC of the per-cell B1 − B0 SSE gain (one-way ANOVA, 229 cells; seed-averaged)**

| grouping | ICC | groups |
|---|---|---|
| publication | 0.44 | 39 |
| chemotype | 0.35 | 38 |
| cluster | 0.42 | 13 |
| publication, inside sc009 | 0.34 | 15 publications, 129 cells |

Publications inside one chemotype differ about as much as chemotypes do. So a laboratory is at least as strong a dependence unit as a chemotype.

**[I] Consequences**
- **(a) Counting.** Under cluster_cpg, a new paper on a known chemotype adds nothing, and a whole new campaign published as one paper adds exactly one cluster.
- **(b) A pre-registered alternative.** For prospective confirmation of a frozen model, the confirmation unit could be the publication / laboratory, with chemotype as a stratum. It must be fixed before any value arrives, and it trades some anti-conservatism (chemotype ICC 0.35) for roughly 8× more effective units.
- **(c) Balanced campaigns avoid Kish failure.** Designed campaigns with one equal-size cell per laboratory × chemotype have Kish = G. In POWER run 4, Kish failure, not the effect, kills 26–47% of the sets at G ≤ 20, because resampled sc009-like sources dominate.

---

## 3. Where B1's gain and error sit (BPG, 5 seeds; BP agrees within 0.02) [M]

Total B1 − B0 SSE gain: **3.02 per seed** (BPG 3.025; BP 3.023). Total B1 SSE ≈ 48.6 per seed (MSE 0.0270).

| source | share of rows | share of B1 SSE | share of B1's gain |
|---|---|---|---|
| giant DGA cluster g:pub_74428aeab2 | 64.9% | 51.8% | 26.8% |
| BTBP / BTP cluster g:pub_af7a1cd919 (C5-BTBP is 148 of its 170 queries) | 9.4% | 24.5% | **37.4%** (C5-BTBP pub_6a0ef8528c alone: **34.6%**) |
| DMDPhPDA g:pub_8415538291 | 6.7% | 4.4% | **17.4%** |
| phenanthroline cluster g:pub_3e0f6bb1b5 (all ORNL) | 9.7% | 10.1% | 4.1% |
| the 9 ORNL-only clusters together | 18.1% | — | 16.6% |
| aqueous-complexant cells (19) | 9.7% | 10.2% | negative (−0.44 SSE / seed) |

**By pair type (share of B1 SSE / share of rows)**

| pair type | SSE share | row share | note |
|---|---|---|---|
| La/Ce | 16.2% | 7.6% | |
| Ce/Pr | 18.0% | 6.9% | |
| La/Ce + Ce/Pr together | 34.2% | 14.5% | 43% of that error is C5-BTBP |
| Eu/Gd | 9.6% | | gain +1.18 SSE / seed |
| Gd/Tb | 6.8% | | gain +1.65 SSE / seed; Eu/Gd + Gd/Tb carry 94% of B1's gain |

Gd-pair queries: 323 per seed, Kish over clusters **2.33**, with 63% in the giant cluster.

**[I] Reading.**
- B1's "universal" Gd amplitude is learned mostly from the N-donor sources with large dips (§7).
- Two of the three sources that carry it are exactly the sources whose provenance is open.
- The gain is real in the data (5/5 seeds, and in round 3's final check), but its *generality* rests on a handful of clusters.

---

## 4. Pending data-quality issues and what each is worth [M]

Sensitivity of the seed-mean R² on BPG; BP is within 0.002. These are **sensitivities, not corrections**. The C5-BTBP floor flag is partly value-derived, and V1 is never edited (DATA_CONTRACT).

| issue (DATA_REQUEST / PROVENANCE) | rows / seed | B1 SSE share | B0 → B1 without them | B1 − B0 | what resolves it |
|---|---|---|---|---|---|
| all rows | 1,802 | — | 0.418 → 0.452 | 0.034 | — |
| C5-BTBP cells: attribution (Nilsson 2006 vs Foreman 2005) and light-Ln floor (log D ≈ −4.2) | 148 (15 cells) | **23.1%** (MSE 0.076 = 2.8× average) | 0.459 → 0.486 | 0.027 | journal access to both closed papers (Experimental section, tables, detection limit) |
| of which the light pairs La/Ce, Ce/Pr, Pr/Nd | 43 | **16.4%** | (with the row below) 0.447 → 0.486 | 0.039 | same |
| DMDPhPDA 5 M Pr/Nd copy A vs B (+1.000 decade) | 1 | **1.8%** (y = 0.945; B1 predicts −0.05 to 0.22) | | | Shimada 2004 thesis (open PDF, about 24 MB; needs the user's download permission) |
| DMDPhPDA, all 10 cells (5 are ORNL copy B) | 120 | 4.4% | 0.426 → 0.456 | 0.029 | same |
| ORNL imports without data location | 529 (54 cells) | 33.3% | 0.386 → 0.422 | 0.036 | the table / figure for each cell (54 cells). **Covers 9 of the 13 clusters entirely** |
| round-3 V2 flags (re-import, duplicate, hidden merge, floor) | 196 (23 cells) | 14.4% | 0.416 → 0.458 | 0.042 | already defined (V2) |
| aqueous complexant / salt present (unseen by the bench) | 174 (19 cells) | 10.2% | 0.410 → 0.453 | 0.043 | condition-aware cohort split (no new data) |
| all pending above, together | 843 (89 cells) | 64.8% | 0.437 → 0.470 | 0.033 | |

**Duplicates [M].**
- 19 C3 cells (107 queries) have an "SF twin": another cell of the same extractant whose log SF agrees within 0.02 on ≥ 4 shared metals. Thirteen also agree in level.
- They are mostly sc009 publications: pub_b9dfc468cd 6, pub_19783c039b 5, pub_5a68dc5665 4.
- Only 1 of the 54 ORNL cells has such a twin, so the ORNL imports are mostly not re-entries of curated cells in log SF terms.

**[I] Value per unit of effort.**
- Desk recovery has the highest leverage per unit of work in this programme. Three documents (two closed papers and one open thesis) decide:
  - about 28% of B1's SSE;
  - about half of B1's gain;
  - the C5-BTBP light-end rows, which hold 16.4% of all B1 error and 43% of the La/Ce + Ce/Pr error.
- The data locations of 54 ORNL cells decide whether 9 of the 13 "independent" clusters are well-sourced.
- All of these need the user (journal access, download permission, or SAFE / ORNL contact). Software cannot do them.

---

## 5. Noise ceiling: what one small campaign would settle [M arithmetic; I interpretation]

- **[M] Inputs.** var(y) over C3 queries = 0.04927 (identical rows every seed); B1's MSE = 0.02699.
- **[M] The bound.** If laboratory repeatability of an adjacent log SF has sd σ, the best possible R² against a single noisy target is 1 − σ²/0.0493. This is an upper bound: noise in the read metals propagates as well.

| σ (sd of a repeated adjacent log SF) | 0.05 | 0.069 | 0.086 | 0.10 | 0.12 | **0.157** | **0.164** | 0.20 |
|---|---|---|---|---|---|---|---|---|
| ceiling R² | 0.95 | 0.90 | 0.85 | 0.80 | 0.71 | **0.50** | **0.454 (= B1)** | 0.19 |

- **[M] Known values.**
  - 0.069 = √2 × the median replicate sd of log D, 0.049 (n = 86), if the two metals' errors were independent.
  - 0.086 = sd(Δ log SF) of 53 same-condition double reports. It bounds transcription noise only (round-3 DATA_QUALITY_REPORT).
- **[I] Interpretation.**
  - The corpus cannot say whether σ is 0.07 or 0.2. Gen13's original median replicate sd of log D was 0.237 before the merges were removed.
  - So the question "is 0.50 reachable at all?" is open. The cheapest experiment that answers it is a whole-curve repeat campaign, already specified in round 3's EXPERIMENTAL_DATA_REQUEST §1: about 6 systems × ≥ 3 independent full La–Lu runs, with an inner analytical repeat.
  - **Decision threshold:** σ ≥ 0.157 means no method can reach 0.50 on single literature targets.

---

## 6. Value of one more measured metal (measurement lever) [M]

**Setup.**
- Same 1,802 BPG queries per seed; the k = 3 DOPT_Q read sets from `REF_BPG.parquet`.
- One more read is added. It must be *legal* under the gen20 metal model: measured, unread, and not a strict neighbour of a read metal. So every query keeps a hidden end and the query set is byte-identical.
- Levels: I_S[y_S − u_S] + u with u_Gd = −0.072. This is HIST_D7, which reproduces the stored predictions to 9e-16. B0 has u = 0.
- 178 of 229 cells (1,516 queries) admit a legal 5th read; the rest keep their k = 3 error.

| reads | rule (label-free unless stated) | R² HIST_D7 | R² B0 |
|---|---|---|---|
| 4 (k = 3) | DOPT_Q | 0.451 | 0.418 |
| 5 | random legal | 0.495 | — |
| 5 | max-min radius gap | **0.503** (seeds 0.499–0.510; 3/5 ≥ 0.50) | 0.471 |
| 5 | Gd first, else max-min | 0.504 | — |
| 5 | oracle (uses labels; ceiling of one read) | 0.541 | — |
| 6 | max-min twice | **0.530** (seeds 0.519–0.543) | 0.499 |

**Per-metal value.** Mean SSE drop per cell when that metal is the legal 5th read:

| metal | mean SSE drop | cells / seed |
|---|---|---|
| Ho | 0.045 | 36 |
| Yb | 0.045 | 21 |
| Er | 0.032 | 59 |
| Eu | 0.032 | 34 |
| Tm | 0.029 | |
| Gd | 0.028 | |
| Tb | 0.027 | |
| Lu | 0.023 | |
| Sm | 0.022 | |
| Nd | 0.022 | |
| Pr | 0.020 | |
| Dy | 0.017 | |

The heavy end and the Gd region are worth the most per added read.

**Unrestricted 5th metal.** If the 5th metal may neighbour a read metal, so that the measured adjacent SF counts as known, a random 5th metal gives 0.530 (practical reading, not the benchmark design).

**Caveats [I].**
- −0.072 is a full-corpus constant (diagnostic, not fold-independent). B1's fold-fitted β is equivalent to within 0.001 at k = 3, but this was not re-fitted at k = 4.
- The max-min rule was one of 4 rules I tried (random, max-min, ends-first, Gd-first). The registered form would be DOPT_Q's own nested 5th read.
- Round 3's "6 metals" statement concerns the C6 cohort and POL_ERR. This is C3 with 51 cells unable to take a 5th read.
- So it is a **hypothesis for a registered k = 4 ladder row**, not a result.
- In a mixed-lanthanide ICP-MS experiment all 14 metals come at once. The k-shot cost model fits radiotracer or limited-assay settings better than modern full-series measurement.

---

## 7. The Gd dip in the raw data (descriptive, no model) [M]

**Statistic.** dip = log D(Gd) − (Eu–Tb linear interpolation in −CN8 radius at Gd), for 134 V1 cells with Eu, Gd and Tb (127 in C3).

| group | cells | median dip |
|---|---|---|
| all | 134 | −0.096 (mean −0.112) |
| DGA donor | 77 | **−0.064** |
| non-DGA | 57 | **−0.175** |
| giant DGA cluster | 78 | −0.063 |
| BTBP cluster | 17 | **−0.289** (C5-BTBP: mean −0.31, sd 0.09; every cell < −0.22) |
| DMDPhPDA | 10 | −0.166 |
| malonamide cluster | 8 | −0.150 |
| phenanthroline cluster | 14 | −0.103 |

- **Reliability.** Split-half reliability of chemotype-mean dip is r = 0.91 (Spearman–Brown 0.95), but only 5 chemotypes have ≥ 4 cells, so the reliability rests on 5 units.
- **[I] Reading.** The fixed −0.072 is a DGA-sized dip. Non-DGA dips are about 3× larger.
- **[I] Why S01 may have missed it.** S01 (donor-class dips) showed no gain over B1. Under BPG the held-out non-DGA cluster removes most of the non-DGA dip evidence from training: there are about 5 non-DGA clusters with Gd triads.
- **[I] What to measure.** New full or near-full series (at least Sm–Dy) in new non-DGA chemotypes from new laboratories. These are the measurements that test whether the one transferable effect is donor-specific.

---

## 8. Which new measurements raise information most per measurement

Ranked by information per unit of effort for this programme's open questions.

| rank | action | what it decides | cost | numbers behind the rank |
|---|---|---|---|---|
| 1 | **Desk recovery**: C5-BTBP source and detection limit; Shimada thesis Table 1 / Fig. 5; ORNL data locations | validity of about 28% of B1's SSE (C5-BTBP 23.1% + DMDPhPDA 4.4%) and about 52% of B1's gain; the ORNL locations decide 9 of the 13 clusters | user access only (no lab) | §4 |
| 2 | **Noise-ceiling repeats**: ≥ 3 independent full La–Lu runs × about 6 systems, one condition each, with per-metal detection limits | whether 0.50 is attainable at all (threshold σ = 0.157) | one ICP-MS campaign | §5 |
| 3 | **One or two more measured metals per cell** (if the use case is k-shot): heavy-end (Ho / Er / Yb) or Gd-region reads | +0.05 R² per read; about 0.50 at 5 metals, 0.53 at 6 | marginal (same experiment) | §6 |
| 4 | **Gd-triad series in new non-DGA chemotypes** (Sm, Eu, Gd, Tb, Dy at minimum; ideally full series) | donor-dependence of the only effect; adds non-DGA clusters (Gd-query Kish is 2.33 now) | per new laboratory × chemotype | §7 |
| 5 | **Independent confirmation sources**: ≥ 40 (optimistic) to > 80 (heterogeneous) new laboratory × chemotype units, one balanced cell each (full series, ≥ 1 condition) | any model claim under the sealed rule | many laboratories; literature can give about 3–5 now | §2.5–2.6 |
| 6 | **SAFE export by the user** → DOI-level novelty count (labels unread) | whether the +1,618 records hold independent Ln clusters | user login, about 1 h | §1 |
| — | not useful | more rows or conditions from existing publications; An / Y / Sc rows; more geometry or descriptors; relaxed cohort keys | — | §1–2 |

**[I] Within-campaign advice**
- Measure every lanthanide at once (ICP-MS) in each run, not k metals.
- Record per-metal detection limits, nitrate and any complexant.
- Spread the campaign over laboratories rather than conditions: a single paper is one cluster under cluster_cpg.
- Freeze the model, the read policy and the cluster definition before the values arrive (DATA_REQUEST §2).

---

## 9. Data-side ideas for round 5 (candidates, not decisions)

1. **D-K4 (cheap, no training): a registered 5-metal (k = 4) ladder row on C3.** Use DOPT_Q's nested 5th read with B1 refitted per fold, plus B0 and HIST_D7 as references. It answers "how many measured metals give 0.50 on C3". Development evidence here: about 0.50 (§6).
2. **D-V3 (needs the user): a source-evidence V3.** Pre-register the rules for C5-BTBP (floor / attribution), DMDPhPDA copy B and ORNL locations before reading any recovered value. Report B0 / B1 on V1 and V3 side by side. The stakes are up to about +0.03 R², and possibly a different B1 gain.
3. **D-COND (no new data): a condition-aware split.** Report the 19 complexant / salt cells and the 17 hidden-merge cells as a separate stratum. They carry negative Gd gain (§4), and the bench cannot see their conditions.
4. **D-CLUSTER (design, before any data): register the confirmation unit for prospective data.** For example: laboratory × publication, with chemotype as a stratum and a balanced one-cell-per-unit design. Add a Kish-based degrees-of-freedom correction (POWER.md "boundary null" undercoverage).
5. **D-GD (lab or literature): targeted non-DGA Gd-triad data** as the first confirmation block, because that is where B1's effect and its doubt both live (§3, §7).
6. **D-NOISE (lab): the round-3 repeat campaign, unchanged, with σ = 0.157 as the pre-stated decision threshold.**

---

## 10. Limits of this inventory

- **Archive conditions** are my own relaxed key (gen13 RELAXED analogue), not the frozen 64-column bundle key. Counts of archive conditions are therefore approximate. Overlap with V1 is exact, through `safe_exp_ids` → `source_record_id`.
- **Linking** used Morgan r = 2, 2048-bit nearest-neighbour Tanimoto ≥ 0.7 to C3 extractants. The official gen6 chemotype map is single-linkage over 190 bundle structures and links more (it put the 2 "unlinked" cells into C3 clusters).
- **Literature SMILES** were rebuilt from names, not read from the papers.
- **SAFE and the literature** were not read in this task (no web access was used).
- **Gain and SSE shares** are from one evaluation design pair (BPG / BP, 5 dev seeds, DOPT_Q, k = 3). The shares are descriptive and carry no intervals.
- **ICCs** come from per-cell gains averaged over seeds. With 2–15 cells per group they are imprecise.

---

## Appendix: reproduction

Scripts and outputs are in the session scratchpad, outside the repository:
`C:\Users\Bandai\AppData\Local\Temp\claude\C--Users-Bandai--claude-spares-spare-9MsqWO\0cd609d3-ed34-4de3-83c3-904e5f4bdb8b\scratchpad\e3\`

**Scripts**
- `e3_inventory.py`: archive-native cells; V1 / C3 overlap; candidate linking; Y / Sc; An co-measurement → `out/e3_inventory.json`, `out/archive_cells_relaxed.csv`, `out/noncohort_*.csv`.
- `e3_value.py`: V1 non-C3 counts; Kish arithmetic; data-quality sensitivities; pair-type and cluster error budgets; residual η²; unrestricted 5th-read value → `out/e3_value.json`.
- `e3_value2.py`: legal 5th / 6th read; B1 gain shares by cluster and publication; Gd-query Kish → `out/e3_value2.json`, `out/voi_legal_cells_BPG.csv`.

**Run as**
`PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B -W ignore <script> <outdir>`, from `D:\ml_separator_gh`. The `-B` flag means no bytecode is written into the repository.

**Done inline** (not saved as scripts): the noise-ceiling table, the ICCs, the Gd-dip table, the SF-twin scan and the literature Tanimoto checks. Each is a few lines over the same inputs.
