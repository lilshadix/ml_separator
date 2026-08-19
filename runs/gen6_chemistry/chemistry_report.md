# gen6 frozen chemistry map — 20260819T102813Z

One target-independent description of chemical space, computed over **every** extractant in the source table (including the ones no cohort is eligible for), and frozen so that every later gen6 run can hash it into its manifest and mean the same thing by "super-cluster".

* source table: `/Users/lilshadix/PycharmProjects/ml_separator/dataset with 3D structures/dataset.parquet` — 5992 rows x 2261 columns, sha256 `fefbefc6fe993aa9…`
* descriptor table: `/Users/lilshadix/PycharmProjects/ml_separator/dataset with 3D structures/ligand_2d_descriptors.parquet` — supplied
* rdkit importable: False (version None). Family assignment prefers the frozen rdkit-derived motif counts in the descriptor parquet; the SMILES-substring fallback is coarse and is labelled per row in `chem__family_source`.
* wall clock: 1.2 s. Nothing is fitted here.

## 1. Scope of the map

| n_extractants | n_ecfp_clusters | n_superclusters | supercluster_threshold | n_fingerprint_bits | largest_supercluster (extractants) |
|---|---|---|---|---|---|
| 190 | 164 | 98 | 0.7 | 2048 | 51 |

Definitions (frozen for the generation): an **extractant** is a `canonical_smiles`; an **ECFP cluster** is a group of bit-identical fingerprints; a **super-cluster** (chemotype) is a single-linkage group at Tanimoto >= 0.7. Labels carry a `chem__` prefix so they can never be confused with a cohort-local label.

## 2. Scaffold families

| family | n_extractants | n_ecfp_clusters | n_superclusters | median_n_cells | n_cells_total | median_nn_within_all | source |
|---|---|---|---|---|---|---|---|
| diglycolamide | 81 | 60 | 23 | 14 | 3960 | 0.88 | lig2d_hc_motifs |
| n_heterocyclic_polydentate | 41 | 40 | 27 | 11 | 645 | 0.7027 | lig2d_hc_motifs |
| amide_other | 28 | 26 | 19 | 2.5 | 373 | 0.7161 | lig2d_hc_motifs |
| malonamide | 11 | 9 | 5 | 9 | 139 | 0.9394 | lig2d_hc_motifs |
| phosphoryl | 11 | 11 | 10 | 6 | 97 | 0.6735 | lig2d_hc_motifs |
| sulfur_donor | 8 | 8 | 8 | 6 | 41 | 0.5909 | donor_census_fallback |
| podand_ether | 4 | 4 | 4 | 5 | 20 | 0.5144 | lig2d_hc_motifs |
| hydroxyl_acid | 3 | 3 | 3 | 8 | 19 | 0.3448 | lig2d_hc_motifs |
| other | 3 | 3 | 3 | 1 | 8 | 0.4444 | smiles_heuristic |

Family sources: {"lig2d_hc_motifs": 179, "donor_census_fallback": 8, "smiles_heuristic": 3}. A family label is a convenience for reading the tables; no metric in this generation is defined on it.

## 3. Nearest-neighbour similarity

`nn_within_all` is, for each extractant, the maximum Tanimoto to any *other* extractant in the full 190. It is the honest measure of how isolated a ligand is before any cohort filter is applied. The pairwise row is the full upper triangle of the similarity matrix and is dominated by unrelated pairs, so read it as background, not as a novelty statistic.

| distribution | n | q05 | q25 | q50 | q75 | q95 | min | max | n <0.4 | n <0.5 | n <0.6 | n <0.7 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nn_within_all (per extractant) | 190 | 0.375 | 0.6478 | 0.7285 | 0.9536 | 1 | 0.1579 | 1 | 12 | 26 | 36 | 80 |
| all pairwise Tanimoto | 17955 | 0.01639 | 0.09836 | 0.1905 | 0.3415 | 0.6 | 0 | 1 | 14626 | 16160 | 17012 | 17609 |

Singleton chemotypes: 80 of 98 super-clusters hold exactly one extractant; the largest holds 51. 12 of 190 extractants have **no** neighbour above Tanimoto 0.4 anywhere in the 190.

## 4. Partition stability — is the freeze safe?

The claim being tested: *the all-190 partition, restricted to a cohort, equals that cohort's own partition.* If it did not, adopting the frozen labels would move ligands across historical fold boundaries and every gen6-vs-gen5 comparison would be contaminated. Two partitions of the same set agree exactly when no local group spans two frozen groups (**splits** = 0) and no frozen group spans two local ones (**merges** = 0). Label strings are ignored.

| cohort | level | n_local | n_frozen_restricted | local_spanning_multiple_frozen (splits) | frozen_spanning_multiple_local (merges) | identical_partition |
|---|---|---|---|---|---|---|
| min_cells=3 | ecfp | 131 | 131 | 0 | 0 | yes |
| min_cells=3 | supercluster | 79 | 79 | 0 | 0 | yes |
| min_cells=10 | ecfp | 74 | 74 | 0 | 0 | yes |
| min_cells=10 | supercluster | 40 | 40 | 0 | 0 | yes |

* `min_cells = 3`: 152 extractants matched into the frozen map, 0 unmapped, overall ok = **True**.
* `min_cells = 10`: 91 extractants matched into the frozen map, 0 unmapped, overall ok = **True**.

**What would falsify this:** a single non-zero split or merge count. Any non-zero entry means the frozen labels must not be substituted for cohort-local ones, and every downstream run must be re-audited before it may quote a super-cluster.

## 5. What the eligibility rule costs — BASE (>= 10 cells) vs EXPANDED (>= 3 cells)

| side | n_rows | n_extractants | n_ecfp_clusters | n_superclusters |
|---|---|---|---|---|
| base | 4881 | 91 | 74 | 40 |
| added_by_expansion | 367 | 61 | 59 | 46 |
| expanded | 5248 | 152 | 131 | 79 |

* ECFP clusters that exist only in the added rows: **57**
* super-clusters that exist only in the added rows: **39**
* rows sitting in those new super-clusters: 229
* the added rows are 7.0 % of the shared cohort — the expansion buys chemistry, not volume.

These counts use the *cohort-local* labels of the shared `min_cells = 3` frame, which is the frame every Experiment A arm will be masked out of, so both sides are labelled identically by construction.

## 6. Phase 0 question — exactly who enters at min_cells = 3 but not at 10

**61 extractants** enter the shared cohort at `min_cells = 3` and are excluded at `min_cells = 10` (91 -> 152 extractants). BASE is a subset of EXPANDED: **True**.

They contribute 367 cohort cells in total (3–9 each), and 61 of 61 carry a human-readable name in the source table.

### 6a. How novel are they? (max Tanimoto to the >= 10-cell cohort)

| distribution | n | q05 | q25 | q50 | q75 | q95 | min | max | n <0.4 | n <0.5 | n <0.6 | n <0.7 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| entrant max Tanimoto to BASE | 61 | 0.1471 | 0.254 | 0.5385 | 0.72 | 0.9394 | 0.102 | 1 | 25 | 29 | 35 | 44 |

| threshold | n_entrants | fraction_of_entrants |
|---|---|---|
| <0.4 | 25 | 0.4098 |
| <0.5 | 29 | 0.4754 |
| <0.6 | 35 | 0.5738 |
| <0.7 | 44 | 0.7213 |

A value below 0.4 is the protocol's **hard chemistry** endpoint: the dense cohort contains nothing chemically close, so a model trained on BASE alone is extrapolating.

### 6b. New chemistry units they bring (frozen labels)

| ECFP clusters present only via entrants | super-clusters present only via entrants | ECFP clusters in BASE | super-clusters in BASE | entrant chemotypes shared with BASE |
|---|---|---|---|---|
| 57 | 39 | 74 | 40 | 7 |

Not every entrant is new chemistry: 2 of 61 already sit in an ECFP cluster that BASE contains (2 are bit-identical to a BASE ligand at Tanimoto 1.0, i.e. a different SMILES with the same fingerprint), and 10 sit at Tanimoto >= 0.8 of something BASE already has. Those add depth, not coverage, and are the reason the expansion buys 39 new chemotypes rather than 61.

Entrant families: {"diglycolamide": 13, "n_heterocyclic_polydentate": 12, "malonamide": 9, "phosphoryl": 8, "sulfur_donor": 7, "podand_ether": 4, "amide_other": 4, "hydroxyl_acid": 3, "other": 1}

### 6c. The entrants themselves

`cohort_n_cells` is the count the eligibility rule actually tested (after the `log_D` floor); `source_n_cells` is the raw count in the source table. They differ where a ligand has rows at or below the detection floor, and the *cohort* number is the definitional one.

| # | extractant_name | family | cohort_n_cells | source_n_cells | max_tanimoto_to_base | nearest_base_partner | chem__supercluster |
|---|---|---|---|---|---|---|---|
| 1 | 2-5-8-15-18-21-hexaoxatricyclo[20.4.0.09-14]hexacosane | podand_ether | 5 | 5 | 0.102 | CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)… | sc000 |
| 2 | 2-(3-4-dihydroxyphenyl)-3-5-7-trihydroxychromen-4-one | hydroxyl_acid | 3 | 3 | 0.12 | CN(C(=O)CC(=O)N(C)c1ccccc1)c1ccc… | sc092 |
| 3 | 1-4-7-10-13-16-hexaoxacyclooctadecane | podand_ether | 5 | 5 | 0.1282 | CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)… | sc001 |
| 4 | 2-5-8-15-18-21-hexaoxatricyclo[20.4.0.09-14]hexacosa-1(26)-9-11-13-22-24-hexaene | podand_ether | 5 | 5 | 0.1471 | CN(C(=O)CC(=O)N(C)c1ccccc1)c1ccc… | sc097 |
| 5 | 3-5-diMe-N-DPPz | n_heterocyclic_polydentate | 6 | 6 | 0.1739 | Cc1cc(-c2cc(C)cc(-c3nnc4c(n3)C(C… | sc080 |
| 6 | DPhen-PyranDGA | diglycolamide | 3 | 3 | 0.1739 | CN(C(=O)CC(=O)N(C)c1ccccc1)c1ccc… | sc004 |
| 7 | 2-[[2-[bis(carboxymethyl)amino]cyclohexyl]-(carboxymethyl)amino]acetic acid | hydroxyl_acid | 8 | 8 | 0.1765 | O=C(COCC(=O)N1CCCCC1)N1CCCCC1 | sc084 |
| 8 | 3-10-13-16-23-pentaoxa-29-azatetracyclo[23.3.1.04-9.017-22]nonacosa-1(29)-4-6-8-17-19-21-25-27-nonaene | podand_ether | 5 | 5 | 0.1818 | CN(C(=O)c1cccc(C(=O)N(C)c2ccccc2… | sc096 |
| 9 | 3-Me-N-DPP | n_heterocyclic_polydentate | 7 | 7 | 0.2 | CCN(C(=O)c1cccc(C(=O)N(CC)c2ccc(… | sc081 |
| 10 | TWE-28 | sulfur_donor | 6 | 6 | 0.2069 | CCN(CC)C(=O)COCC(=O)N(CC)CC | sc074 |
| 11 | 3-5-diMe-N-DPP | n_heterocyclic_polydentate | 6 | 6 | 0.2128 | CCN(C(=O)c1cccc(C(=O)N(CC)c2cccc… | sc079 |
| 12 | TWE-23 | sulfur_donor | 6 | 6 | 0.2333 | CCCCCCOP(=O)(O)OCCCCCC | sc067 |
| 13 | 2-[2-[bis(carboxymethyl)amino]ethyl-(2-hydroxyethyl)amino]acetic acid | hydroxyl_acid | 8 | 8 | 0.2353 | CCCCCCCCN(CCCCCCCC)C(=O)CN(CC(=O… | sc085 |
| 14 | TWE-27 | sulfur_donor | 6 | 6 | 0.25 | CCCCCCOP(=O)(O)OCCCCCC | sc066 |
| 15 | TWE-30 | sulfur_donor | 6 | 6 | 0.25 | CN(C(=O)COCC(=O)N(C)c1ccccc1)c1c… | sc093 |
| 16 | N-DP(DOM)P | n_heterocyclic_polydentate | 6 | 6 | 0.254 | CCCCCCc1ccc(N(CC)C(=O)c2cccc(-c3… | sc026 |
| 17 | 2-9-bis(diphenylphosphoryl)-1-10-phenanthroline | phosphoryl | 3 | 3 | 0.2558 | CCCCN(CCCC)C(=O)c1ccc2ccc3ccc(C(… | sc091 |
| 18 | EsPyTri | n_heterocyclic_polydentate | 5 | 5 | 0.2727 | CCCCCCn1c(-c2ccccc2)c(-c2ccccc2)… | sc056 |
| 19 | C5-BPP | n_heterocyclic_polydentate | 7 | 7 | 0.28 | CCN(C(=O)c1cccc(-c2cccc(C(=O)N(C… | sc002 |
| 20 | 2PQM | other | 6 | 6 | 0.3 | CN(C(=O)c1cccc(C(=O)N(C)c2ccccc2… | sc090 |
| 21 | 2-9-bis(diethoxyphosphoryl)-1-10-phenanthroline | phosphoryl | 3 | 3 | 0.3333 | CCCCN(CCCC)C(=O)c1ccc2ccc3ccc(C(… | sc073 |
| 22 | TWE-29 | sulfur_donor | 6 | 6 | 0.3548 | CCCCCCOP(=O)(O)OCCCCCC | sc064 |
| 23 | DO-PyranDGA | diglycolamide | 3 | 3 | 0.3585 | CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)… | sc022 |
| 24 | TWE-24 | sulfur_donor | 6 | 6 | 0.3793 | CCCCCCOP(=O)(O)OCCCCCC | sc065 |
| 25 | CHELI2 | amide_other | 3 | 3 | 0.38 | CN(C(=O)c1cccc(C(=O)N(C)c2ccccc2… | sc053 |
| 26 | N-N-N-N-tetrabutyl-2-[[2-[3-(dibutylamino)-2-(dibutylcarbamoyl)-3-oxopropyl]phenyl]methyl]propanediamide | malonamide | 4 | 4 | 0.4468 | CCCCCCCCN(CCCCCCCC)C(=O)C(C)OC(C… | sc062 |
| 27 | 2-(dibutylcarbamoyl)benzoic acid | amide_other | 3 | 3 | 0.475 | CCCCN(CCCC)C(=O)c1ccc2ccc3ccc(C(… | sc063 |
| 28 | TWE-35 | phosphoryl | 6 | 6 | 0.4762 | CCCCCCCCCCCCN(CCCCCCCCCCCC)C(=O)… | sc041 |
| 29 | TWE-36 | phosphoryl | 6 | 6 | 0.4865 | CCCCCCCCN(CCCCCCCC)C(=O)CO | sc040 |
| 30 | LLL14 | diglycolamide | 5 | 5 | 0.5 | CCCCCCCCCCCCN(C)C(=O)C1CC(OC)CC(… | sc035 |
| 31 | T8-THP-TAM | sulfur_donor | 4 | 4 | 0.5385 | CCCCCCCCN(CCCCCCCC)C(=S)c1ccc2cc… | sc049 |
| 32 | T8-THP-CAM | amide_other | 4 | 4 | 0.5385 | CCCCCCCCN(CCCCCCCC)C(=O)c1ccc2cc… | sc048 |
| 33 | N-N-dibutyl-N-N-dimethylpropanediamide | malonamide | 9 | 9 | 0.5517 | CCCCCCCCCCCCN(C)C(=O)COCC(=O)N(C… | sc031 |
| 34 | TIBDGA | diglycolamide | 8 | 8 | 0.56 | CCN(CC)C(=O)COCC(=O)N(CC)CC | sc010 |
| 35 | C2-PTA | n_heterocyclic_polydentate | 7 | 7 | 0.561 | CCCCN(CCCC)C(=O)c1ccc2ccc3ccc(C(… | sc029 |
| 36 | Cy5-O-Me4-BTBP | n_heterocyclic_polydentate | 5 | 5 | 0.6111 | CC1(C)CCC(C)(C)c2nc(-c3cccc(-c4c… | sc013 |
| 37 | N-N-N-N-tetrabutylbutanediamide | amide_other | 5 | 5 | 0.64 | CCCCN(CCCC)C(=O)COCC(=O)N(CCCC)C… | sc036 |
| 38 | N-N-N-N-tetrabutylpropanediamide | malonamide | 9 | 9 | 0.64 | CCCCN(CCCC)C(=O)COCC(=O)N(CCCC)C… | sc036 |
| 39 | TWE-33 | phosphoryl | 6 | 6 | 0.6667 | CCCCCCCCCCCCN(CCCCCCCCCCCC)C(=O)… | sc046 |
| 40 | 2-EH-C9-B8D(iB)CMPO | phosphoryl | 6 | 6 | 0.6735 | CCCCCCCCP(=O)(CC(=O)N(CC(C)C)CC(… | sc017 |
| 41 | TWE-34 | phosphoryl | 6 | 6 | 0.6765 | CCCCCCCCCCCCN(CCCCCCCC)C(=O)COCC… | sc045 |
| 42 | N-N-dihexyl-N-N-dimethylpropanediamide | malonamide | 9 | 9 | 0.6786 | CCCCCCCCCCCCN(C)C(=O)COCC(=O)N(C… | sc031 |
| 43 | N-N-dimethyl-N-N-dioctylpropanediamide | malonamide | 9 | 9 | 0.6786 | CCCCCCCCCCCCN(C)C(=O)COCC(=O)N(C… | sc031 |
| 44 | CyMe4-BTPhen | n_heterocyclic_polydentate | 6 | 6 | 0.6857 | CC1(C)CCC(C)(C)c2nc(-c3cccc(-c4c… | sc011 |
| 45 | 2-hexyl-N-N-dimethyl-N-N-dioctylpropanediamide | malonamide | 9 | 9 | 0.7059 | CCCCCCCCN(C)C(=O)C(CCOCCCCCC)C(=… | sc030 |
| 46 | N-N-N-N-tetrahexylpropanediamide | malonamide | 9 | 9 | 0.72 | CCCCCCCCN(CCCCCCCC)C(=O)CO | sc036 |
| 47 | N-N-N-N-tetraoctylpropanediamide | malonamide | 9 | 9 | 0.72 | CCCCCCCCN(CCCCCCCC)C(=O)CO | sc036 |
| 48 | TBi-PE-BisDGA | diglycolamide | 6 | 6 | 0.7333 | CCCCN(CCCC)C(=O)COCC(=O)N(CCCC)C… | sc009 |
| 49 | DIBDODGA | diglycolamide | 8 | 8 | 0.7419 | CCCCCCCCCCCCN(CCCCCCCC)C(=O)COCC… | sc009 |
| 50 | 2-N-9-N-diethyl-2-N-9-N-bis(4-methylphenyl)-1-10-phenanthroline-2-9-dicarboxamide | n_heterocyclic_polydentate | 3 | 3 | 0.7632 | CCc1ccc(N(CC)C(=O)c2ccc3ccc4ccc(… | sc068 |
| 51 | C4-PTA | n_heterocyclic_polydentate | 6 | 6 | 0.7895 | CCCCN(CCCC)C(=O)c1ccc2ccc3ccc(C(… | sc029 |
| 52 | D2EHDODGA | diglycolamide | 8 | 8 | 0.8 | CCCCC(CC)CN(CC(CC)CCCC)C(=O)COCC… | sc009 |
| 53 | C6-PTA | n_heterocyclic_polydentate | 6 | 6 | 0.8 | CCCCCCCCN(CCCCCCCC)C(=O)c1ccc2cc… | sc029 |
| 54 | LL3 | diglycolamide | 9 | 9 | 0.8182 | CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)… | sc009 |
| 55 | 6-MH-C9-B8D(iB)CMPO | phosphoryl | 6 | 6 | 0.8333 | CCCCCCCCP(=O)(CC(=O)N(CC(C)C)CC(… | sc008 |
| 56 | TBEE-BisDGA | diglycolamide | 6 | 6 | 0.8462 | CCCCN(CCCC)C(=O)COCC(=O)N(CCCC)C… | sc009 |
| 57 | DMDOaDGA | diglycolamide | 7 | 7 | 0.8571 | CCCCCCCCCCCCN(CCCCCCCC)C(=O)COCC… | sc009 |
| 58 | N-N-dibutyl-2-(2-hexoxyethyl)-N-N-dimethylpropanediamide | malonamide | 6 | 6 | 0.9394 | CCCCCCCCN(C)C(=O)C(CCOCCCCCC)C(=… | sc030 |
| 59 | TPDGA | diglycolamide | 3 | 3 | 0.9583 | CCCCCCCCCCCCN(CCCCCCCC)C(=O)COCC… | sc009 |
| 60 | DHpipDGA | diglycolamide | 8 | 8 | 1 | CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)… | sc044 |
| 61 | DHmorDGA | diglycolamide | 8 | 8 | 1 | CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)… | sc044 |

The full table, including the canonical SMILES of every entrant and its nearest BASE partner, is `cohort_entrants.csv`.

## 7. Limits of this artifact

* The map describes chemistry, not measurability. `n_cells` is a property of the experiment record; nothing here reads `log_D`.
* ECFP clusters are bit-identical groups of a **precomputed** 2,048-bit fingerprint taken from the source table. Two different SMILES can share one, and the map does not re-derive fingerprints.
* Single-linkage chaining means a super-cluster can hold two members below the threshold from each other. That is deliberate — it makes the hold-out strictly harder — but a super-cluster is not a ball of radius 0.3.
* The SMILES-substring family fallback is a reading aid only; if `chem__family_source` says `smiles_heuristic`, do not quote that family as a curated assignment.

