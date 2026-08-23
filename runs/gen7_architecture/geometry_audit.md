# 3D geometry audit — `dataset with 3D structures/`

Generated 2026-08-19. Machine-readable companion: `geometry_audit.json`.
Everything below was measured by parsing the assets, not read from the bundle prose.

## Headline answers

| Question | Answer |
|---|---|
| Usable structures | **1155** |
| Ligand-only or Ln-complex | **Ln-complex** — exactly one lanthanide in every file |
| Conformers per (ligand, metal) | **1.12 mean, max 2** — and the 2s are chemistry variants, not conformers |
| Charge consistency | metal_ox = +3 everywhere; net complex charge **+3 / +2 / +1** (901 / 181 / 63), never 0 |
| Metals | **14** lanthanides (Z 57–60, 62–71); no Pm, no Y/Sc, no actinides |
| Coordination numbers | declared **8 (503) or 9 (652)** only; measured donors ≠ declared in 22.4 % |
| Failure / exclusion rate | **101 / 1256 complexes = 8.04 %**; 513 / 5992 rows = 8.56 % |
| Duplicates | **none** (0 by file hash, coordinates, build_id, geometry_key, or ligand frame) |
| canonical_smiles with ≥1 geometry | **177 / 190 (93.2 %)** |
| Rows mapped to a geometry | **5479 / 5992 (91.44 %)** |

## 1. Inventory and integrity

- 1155 `.xyz` on disk (plus a `.DS_Store`); `accepted_geometries.csv` has 1155 rows; set difference in both directions is **empty**.
- All 1155 SHA-256 values in `checksums.sha256` match the files on disk, and each packaged file's hash equals the `source_xyz_sha256` recorded in `accepted_geometries.csv`. No silent re-writes.
- Every file parses cleanly: header atom count equals parsed atom count in 1155/1155, no trailing junk.
- `qc_status = accepted`, `file_qc_status = accepted`, `qc_class = OK` for 1155/1155. Selection provenance: 1098 `frozen_accepted_manifest` + 57 `current_qc_supplement`.
- `mol2_path` is null in all 1155 rows — only XYZ ships.
- Format is extended XYZ: `Properties=species:S:1:pos:R:3:initial_charges:R:1:initial_magmoms:R:1[:forces:R:3:charge:R:1]`, `pbc="F F F"`, with `energy`, `free_energy`, `dipole` on the comment line for 1116 of the 1155.

Size: 34–484 atoms (median 224, mean 225.2); heavy atoms 22–175. Elements present across the bundle: H, C, N, O, P, S, Cl and the 14 lanthanides. P in 72 files, S in 28, Cl in 17.

## 2. Ligand-only vs complex

**Ln-complex.** 1155/1155 files contain exactly one lanthanide; 0 files have zero or more than one; 0 actinides. The filename metal prefix matches the lanthanide inside the file in 1155/1155.

The metal sits at the coordinate origin in only **29** files (a subset of the 39 unrelaxed structures). The other 1126 are in an arbitrary frame — descriptors must be metal-centred or rotation/translation invariant. `geometry_descriptors.py` is: it subtracts `coordinates[metal_index]` before every donor computation and uses only gyration-tensor/hull invariants for `global_shape`.

A *separate* ligand-only asset exists and must not be confused with these: `features/ligand_pi_control_images.npz` holds **190** RDKit ETKDGv3 free-ligand persistence images, explicitly labelled `control_only_not_model_feature` in the manifest, and its index is attached to all 5992 rows (including rows with no complex geometry).

## 3. Conformers per (ligand, metal)

There is **no conformational ensemble anywhere in this bundle.**

- 1031 distinct (canonical_smiles, metal_symbol) cells → 907 with one file, 124 with two. Mean 1.12, max 2.
- Files per `geometry_key`: **1 for all 1155**.
- Files per full chemical environment (smiles, metal, anion, fill_ligand, coreCN, n_ligs, ox): **1 for all 1155**.
- All 124 two-file cells differ in `inner_sphere_anion` (nitrate vs water); 12 also differ in coreCN, 3 in n_ligs.

So each accepted geometry is a single Architector placement — one relaxed pose per scientific complex. Any "3D ensemble" modelling (Boltzmann averaging, conformer pooling, multi-pose GNN) has nothing to average over here.

Ligand × metal matrix in `accepted_geometries.csv`: 177 ligands × 14 metals, **41.6 % filled**. 42 ligands have all 14 metals; 89 ligands have exactly **one** metal. That skew is what caps any per-ligand contraction-slope signal.

## 4. Charge consistency

- `metal_ox = 3` in 1155/1155 spec rows.
- The extxyz `initial_charges` column assigns the **entire net complex charge to the metal atom**; every other atom carries 0.0. It is a formal charge, not a partial charge (computed metal partial charges live separately in `complex_physical_scalars.metal_partial_charge`, range 0.651–0.893, mean 0.828).
- Net complex charge: **+3 (901 files), +2 (181), +1 (63), unknown (10 files with no charge column)**. **Zero neutral complexes.** The real extracted species (e.g. Ln(NO₃)₃·nL) is never represented — all 1155 structures are cations.
- The charge follows `net = 3 − (number of coordinated nitrate groups)` in **1140 / 1145** files with a charge column. Only 244 of 1155 complexes contain any nitrate at all (180 with one, 64 with two); 911 are bare +3 cations solvated only by neutral ligands and water.
- **5 outliers** break that rule and **4 files** put the charge on non-metal atoms instead of the metal: `La_033b4c93146d`, `Lu_adecfa6ff040`, `Yb_2008860df8de`, `Yb_3927fc1f583b` (plus `La_d655b942946b`).
- `initial_magmoms` is 0.0 for nearly every atom; the only nonzero value seen in a 200-file sample is 6.0 (on the metal). It is not a usable spin label.
- **48** (ligand, metal) cells carry two geometries with *different net charge*, and **43** ligands take more than one net charge across metals. Any charge-derived feature is therefore partly a proxy for which variant was picked.

## 5. Metals

14 lanthanides, all Ln(III):

| | La | Ce | Pr | Nd | Sm | Eu | Gd | Tb | Dy | Ho | Er | Tm | Yb | Lu |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| geometries | 89 | 71 | 80 | 84 | 81 | **204** | 81 | 67 | 80 | 63 | 76 | 55 | 63 | 61 |

Pm (Z 61) absent. Eu is 17.7 % of geometries and 26.1 % of rows — the same macro/micro imbalance already flagged for extractants.

Lanthanide contraction is present but noisy: mean Ln–donor distance falls from 2.536 Å (La) to 2.307 Å (Lu), Pearson r(Z, mean distance) = **−0.537** over the 1155 geometries — the ligand identity dominates the metal.

## 6. Coordination numbers

Declared `coreCN`: **8 in 503 files, 9 in 652**. Nothing else. Mean 8.565.
`nearest_coreCN_sig` lists exactly `coreCN` donors in 1155/1155 rows, and the VR asset's `is_coord_donor` flag count equals `coreCN` in 1155/1155.

Donor elements across the 9892 declared donor slots: **O 8279, N 1554, S 58, P 1**.

The declared CN is a **build spec, not a measurement**:

| donors within 3.10 Å minus declared coreCN | 0 | +1 | +2 | +3 | +4 |
|---|---|---|---|---|---|
| geometries | 896 | 181 | 61 | 9 | 8 |

**22.4 %** of accepted geometries have more atoms inside 3.10 Å than the CN they are labelled with. Counted purely geometrically the first shell ranges 8–27 atoms within 3.10 Å and 0–24 within 2.80 Å. `complex_physical_scalars.coordination_number` is a copy of the declared spec, and `observed_donors_within_3p10A` is the measured one — they agree for only 896 rows. Treating `coordination_number` as observed chemistry is wrong.

Other spec distributions: `n_ligs` 1–4 (139/485/430/101), `DENTATE` 2–8 (mode 3, 710), `inner_sphere_anion` nitrate 785 / water 370, `n_fill` 0–5 (0 in 572 files).

## 7. Failures and exclusions

1256 baseline geometry specs → 1155 accepted, **101 excluded = 8.04 %**; those 101 cover **513 of 5992 rows = 8.56 %**.

| exclusion_state | n | | qc_class | n |
|---|---|---|---|---|
| qc_rejected | 78 | | BORDERLINE_AMBIGUOUS_SHELL (`gap_after_coreCN<0.10`) | 40 |
| generation_failed | 15 | | FAIL_LONG_BOND (`coreCN_max_dist>3.10`) | 26 |
| terminal_generation_failure | 8 | | UNAVAILABLE (never built) | 23 |
| | | | BORDERLINE_LONGISH (`coreCN_max_dist>2.95`) | 12 |

Generation-side reasons: `failed_no_structures` 10, `skipped_known_bad_ligtype` 2, `failed_timeout` 2, `failed_ligtype` 1, plus 8 with no status. Eu accounts for 32 of the 101 exclusions.

Exclusion is **not random**: it selects against the biggest, most flexible ligands. The 13 ligands with zero geometry are the tripodal/calixarene/multi-DGA scaffolds (TWE-7, TWE-10, DO-PyranDGA, N-DP(DOM)P, the tris-DGA benzenes, the hexameric calixarene). Downstream, the 3D arm is evaluated on an easier chemical slice than the 2D arm.

Downstream feature computation itself had **0 failures** (`geometry_feature_failures: 0`).

## 8. Duplicates

None, on every definition tested:

- duplicate file SHA-256 groups: **0**
- duplicate coordinate sets (symbols + coords rounded to 1e-4): **0**
- duplicate `build_id` / `baseline_build_id` / `geometry_key` / `final_xyz_name`: **0**
- identical **ligand frames** (heavy atoms with the lanthanide removed, rounded to 0.01 Å) shared by two or more files: **0** — so no ligand pose was cloned across metals; every metal got its own relaxation.

Note the two id columns are different: the file is named `<Metal>_<baseline_build_id>.xyz`, while `build_id` (the key used by every feature asset and by `geometry_feature_build_id`) differs from `baseline_build_id` in a subset of rows. Joining feature tables to filenames on the wrong one silently mismatches the 143 geometries where the two ids differ.

## 9. Ligand and row coverage

- **190** canonical_smiles in `dataset.parquet`; **177 (93.2 %)** have at least one geometry; 177 also appear in `accepted_geometries.csv`. 13 ligands have none.
- 141 ligands fully covered, 36 partially, 13 not at all.
- **5479 / 5992 rows (91.44 %)** carry an accepted geometry. Identical in `row_geometry_map.csv`, `dataset.parquet.geometry_ok`, and `dataset_geometry_available.parquet` (5479 rows).
- Where the 513 uncovered rows go: **81** are in ligands with zero geometry, **432** are in partially covered ligands.
- Per-metal row coverage ranges from **Lu 80.4 %** to **Sm 98.6 %** (Eu 87.5 %).
- Ligand × metal cells present in the dataset: 1112; with geometry: **1031**.
- `SMILES_FOR_ARCHITECTOR` equals `canonical_smiles` in 1155/1155 rows — no simplification drift, and `provenance/current_geometry_index.csv` confirms 0 simplified ligands and 0 rows where the original SMILES differs from the used one (29 rows do carry a `ligtype_override`).

**One geometry serves many rows.** Rows per accepted geometry: min 1, median 1, mean 4.74, **max 377**. The top 10 geometries alone cover 1053 rows (19.2 % of covered rows). 3D features are therefore constant inside large blocks of rows — a random-row CV split will place the same 3D vector on both sides of the fold.

## 10. Feature assets

| asset | shape | key | note |
|---|---|---|---|
| `complex_physical_scalars.parquet` | 1155 × 32 | `geometry_key` | 5 columns **all-null**: `binding_energy_eV`, `strain_energy_eV`, `homo_eV`, `lumo_eV`, `homo_lumo_gap_eV`. xTB-derived columns (energy, dipole, partial charges) are null for **39** geometries. |
| `coordination_polyhedron.parquet` | 1155 × 56 | `geometry_key` | 9 donor distances + Z + 36 donor–M–donor angles; the `_09` slots are null for **43.5 %** (the CN 8 complexes). |
| `complex_gfn2xtb_pi_images.npz` | (1116, 1, 20, 20) | image index | covers **1116 of 1155**; 39 geometries and **114 rows** have no PI. |
| `vietoris_rips_inputs.npz` | 1155 graphs, 260 068 nodes, 2 722 813 edges | `build_ids` | **exact match** to the XYZ: same atom count and element composition in 1155/1155, `is_metal` sums to 1 per graph, `is_coord_donor` sums to declared `coreCN` in 1155/1155. |
| `ligand_pi_control_images.npz` | (190, 1, 20, 20) | ligand index | free-ligand control, one per canonical_smiles, index attached to all 5992 rows. |
| `features_by_safe_exp_id.parquet` | 5992 × 93 | `safe_exp_id` | pre-joined row-level table. |

The same **39** geometries are missing xTB properties, missing energy on the comment line, missing the persistence image, and flagged `xtb_properties_available = False` / `complex_pi_eligible = False` in `geometry_feature_status.csv` — one consistent set, 20 of them Eu.

`geometry_descriptors.py` reads **only** `vietoris_rips_inputs.npz` (it requires `coordinates`, `atomic_numbers`, `is_metal`, `is_coord_donor`, `node_ptr`, `build_ids`), so it is unaffected by the 39-geometry xTB gap. Its van der Waals table covers exactly {H, C, N, O, P, S, Cl} — which is exactly the non-lanthanide element set in the bundle, so `FALLBACK_VAN_DER_WAALS_RADIUS` is never reached, **but the lanthanide itself has no entry and so is enclosed with the 1.70 Å carbon fallback** in `enclosure_descriptors`.

## 11. Structural quality flags

- **39 structures are not relaxed** (no `energy=` on the comment line, no xTB properties). Their source paths carry tags `attempt01/03_emergency_unrelaxed` (13), `uff_unrelaxed` (5), `uff_xtb_no_preopt` (9), `direct_etkdg_xtb` (3). 10 of them lack the charge column entirely. These 39 passed the same `qc_class = OK` gate as the relaxed ones.
- **14 files** have a metal–ligand contact below 2.0 Å (min **1.71 Å**, vs a median first-shell contact of 2.21 Å) — physically too short for Ln(III)–O.
- **9 files** contain an interatomic pair below 0.9 Å and **5** below 0.7 Å (worst 0.581 Å); 2 files have a heavy-atom pair below 1.1 Å. All the worst cases are in the unrelaxed 39.
- `coreCN_max_dist` in the accepted set runs to 2.9499 Å with a minimum `gap_after_coreCN` of 0.1013 Å — right on the 2.95/0.10 rejection thresholds. Acceptance is a hard cut, so the accepted tail is adjacent to the rejected population.

## 12. Confound to control before claiming "3D helps"

The choice between the nitrate and water geometry variant is **a deterministic function of an existing condition flag**: all 3957 rows using a nitrate geometry have `cond__acid__hno3 = 1`, and 1099 of the 1522 water-geometry rows have `cond__acid__hno3 = 0`. Any 3D block therefore carries a re-encoding of the nitric-acid indicator, plus, through `n_fill`/net charge, a re-encoding of nitrate loading. A gain from adding 3D features must be shown against a control that already has those flags — the `permute_descriptors(mode="metal-preserving")` negative control in `geometry_descriptors.py` is the right instrument.

Second confound: `geometry_key = "<Z>|<canonical_smiles>|<anion>"`, i.e. the geometry identity is a hash of metal + ligand + medium. Anything keyed on it inherits ligand identity exactly, so a descriptor block can act as a high-cardinality ligand id. With 801 of 1155 geometries serving a single row, per-geometry descriptors are near-unique row labels for those rows.
