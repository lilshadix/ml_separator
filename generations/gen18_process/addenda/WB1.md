# Addendum WB1 — data layer (2026-09-13)

Owner: WB1 (data). Files: `gen18proc/systems.py` (extension below the WB0 marker),
`gen18proc/literature.py`, `gen18proc/ingest.py`, `scripts/g18_build_db.py`, `g18_audit.py`,
`g18_seal_prereg.py`, `cases/*.json`, `tests/test_bundle.py`, `test_systems.py`,
`test_ingest.py`, `README.md`, `DATA_AUDIT.md` (written by the audit script). Every item below
is a place where DESIGN.md was silent, impossible as written, or where an interface other
owners code against needed a concrete shape. Nothing changes a name or field of DESIGN.md.

## A1. Public helpers added to `gen18proc.systems` (interfaces for WB2-WB4)

| name | what |
|---|---|
| `temperature_band(T) -> str` | the band of section 3.4 (`lo <= T < hi`); None/NaN -> `"20-30C"` (`DEFAULT_BAND`). `TEMPERATURE_BANDS` holds the edges. The fitter and the domain builder use it so the band assignment is one function. |
| `system_key(entry) -> tuple`, `system_id(key_or_entry) -> str` | section 3.1; `system_id` accepts either the key tuple or an entry. |
| `entry_to_json(entry) -> dict`, `entry_from_json(obj) -> SystemEntry` | the JSON shape of sections 3.7 / 3.8. `params[ligand][band]` objects carry `"model_type": "cation_exchange" \| "solvating"` to pick the dataclass. |
| `iter_sourced(entry)` | every `Sourced` with its dotted path (records' provenances excluded). |
| `ApplicabilityMapping(dict)` | `SystemEntry.applicability` is this dict subclass; `.meta[key]` holds the hand-entered `_status`, `_assumed_label`, `_range_note` of a literature domain so the file round-trips byte for byte. Corpus domains have no metadata. Building an entry with a plain dict is allowed (no metadata). |
| vocabularies | `ACID_CLASSES`, `ANIONS`, `DILUENT_FAMILIES` (the bundle's 7 families, V17), `FAMILIES`, `ORIGINS`, `LIGAND_ROLES`, `AGGREGATIONS`, `STREAM_KINDS`, `DUPLICATE_FLAGS`, `K_REPORTED_AS`. |
| `registry_row(obj)` | the registry row of one entry JSON; `load_registry` adds the columns `ligands`, `acid_class`, `diluent_family`, `modifiers` to the nine of section 3.6. |

`write_system` writes LF newlines, a trailing newline, `allow_nan=False` (NaN is converted to
`null` at the boundary by `entry_to_json`).

## A2. Validator readings where section 3.6 needed a decision

* **V4 / V9 split**: V4 checks the label and the source (`kind: doi` with a DOI, or `kind: none`
  with a non-empty `note` *or* `locator` saying why — the 3.7 example's `anions_per_metal` has
  the reason in the locator and an empty note); the "without a range" half of V4 is reported as
  V9 so that each rule fires once.
* **V5 for hand-entered domains**: an `applicability` block with `_status: assumed` needs
  `_assumed_label: ASSUMED_PLACEHOLDER` and a non-empty `_range_note`; a domain with no
  records, no hull and no `_status` is an error (it can only be hand-entered).
* **V8 checks key presence, not value**: every metal that appears in `records` (any band) or in
  a `StreamRecord.metals_mM` must be a key of `params[k][band].log_k`. A fitter that has no
  intercept for a metal must write `{"value": null, "status": "unknown", ...}` for it (passes
  V2 and V8); omitting the key fails the load.
* **V14** fires when `protons_released_per_metal.value != 3` or `aggregation != "dimer"` unless
  that `Sourced` has status `literature` / `measured_literature`.
* **V16** is applied twice: on the raw JSON before construction (an unknown `status`, `kind`,
  `mechanism` or `model_type` cannot raise inside `entry_from_json`) and on the constructed
  entry (role, aggregation, family, origin, anion, acid class, stream kind, duplicate flag).
* **W2** fires only when neither `medium.temperature_C` nor any record carries a temperature.

## A3. Unit-slip rule (section 4.6): the differing condition is the metal-concentration axis

Section 4.6 says "exactly one continuous condition differs between two rows and its ratio
equals the metal's molar mass, 1000 or 1/1000", and its verified outcome is the seven sub-mM
rows Ca_SAFE:2693-2699 of the D3DODGA/Nd 3 M series. Those seven rows differ from their 1 M
copies in **two** conditions (nominal acid 1 vs 3 M, and metal concentration by exactly
144.24x), so the literal reading cannot produce the verified outcome. Implemented: a tied pair
is a unit slip when the **metal concentrations** differ by a factor within 0.1 % of the metal's
molar mass (g/L vs mM) or of 1000 (M vs mM); the other conditions may differ. Applying the
factor test to the acid or extractant axis as well would additionally flag Eu_SAFE:16079 /
16082 (pub_515c63b970, D = 0.100000 exactly at 0.001 and 1.0 M acid, no metal concentration):
a rounded plateau, not a unit slip, and one the keep-rule (residual from the loading series in
(log mM, log D)) cannot even assess. That pair stays `TIED_D`. Keep-rule detail: when only one
copy's series has >= 3 clean points, that copy is kept; when neither has, both are flagged.
The kept copy carries `duplicate_flag = None`.

Counts: 130 tied groups / 324 rows / 24 publications in total (`n_tied_d_groups`,
`n_tied_d_rows`); of these 7 groups (14 rows) are the unit-slip tier (7 rows flagged) and 123
groups / 310 rows the `TIED_D` tier.

## A4. Corpus entry shape where the section 3.8 example cannot pass its own validator

* `medium.salting_anion_M` of the 3.8 example is `0.0` with `measured_corpus` but no
  `safe_exp_ids` / `publication_id`, which V3 (section 1.2) refuses. Corpus entries carry
  `value: null, status: unknown, source.kind: corpus` with the note that no salting-agent
  column exists and the cascade uses anion = nominal acid (section 4.3).
* `variant_tag` is `null` for corpus ligands (no general rule derives it); `scaffold_id` is
  `"DGA_core"` for diglycolamides and `null` otherwise.
* Family / mechanism / aggregation of a corpus ligand come from RDKit substructures
  (`ingest.classify_ligand`): P-OH, carboxylic or sulfonic acid -> cation exchange, dimer
  (families `acidic_organophosphorus`, `carboxylic_acid`, `other`); diglycolamide core ->
  `diglycolamide`; phenanthroline + amide -> `phen_carboxamide`; aromatic N without amide ->
  `n_donor`; amine without amide -> `amine`; else `other` (malonamides, TBP, DHOA, DOHyA).
  Solvating stoichiometry: `ligands_per_metal` unknown (set by the fitter), protons 0
  (literature, the brief's mechanism DOI), anions 3 for nitrate / chloride / perchlorate and
  unknown for sulfate / carboxylate. Cation-exchange stoichiometry: q = 3 assumed [2, 3] (S2
  slopes), p = 3 literature (S2 mechanism), z = 0 assumed [0, 0].
* Synergist ligands (`HDEHP`, `TBP`, `DHOA`, `DOHyA` from the additive one-hots) and alcohol
  modifiers (`1_octanol`, `octanol`, `isodecanol`, `isodecyl`, `isododecanol`) are
  `LigandSpec` entries with `role = "synergist"` / `"modifier"`; their concentration is unknown
  (not in the bundle) and `records[].ligand_M` holds only the primary extractant.
* `records[].anion_M = acid_nominal_M` for every acid class (formal acid concentration; for
  polyprotic / weak acids not a free-anion value). `metals_initial_mM` is `{}` when the bundle
  metal concentration is NaN. The provenance note of every record with a metal concentration
  is "metal concentration semantics ASSUMED initial aqueous; O/A not reported";
  `temperature_assumed_20_30C` is appended when the temperature is NaN (75 rows).
* `is_tracer` marks exactly one row per series (smallest metal concentration, ties broken by
  the smaller `record_id`).
* `medium.acid` is the decoded acid name, or `"mixed: a; b"` when a carboxylate system pools
  several acids; `diluent.name` is the decoded diluent when the system has one, else null.

## A5. Files written by the ingest and the audit (shapes other owners read)

* `systems/corpus_records.csv` (5860 rows; sorted by system_id, record_id): `system_id,
  record_id, metal, d, log_d, acid_nominal_M, acid_eq_M, anion_M, ligand_name, ligand_M,
  synergists, modifiers, complexant_M, metal_initial_mM, oa_ratio, temperature_C,
  temperature_band, contact_time_min, diluent_name, diluent_family, acid_class, anion,
  acid_name, canonical_smiles, extractant_name, publication_id, experiment_series_id,
  replicate_id, replicate_group_id, loading_series_id, is_tracer, fit_eligible,
  fit_ineligible_reason, duplicate_flag`. `replicate_group_id` is the blake2b of the exact
  64-column condition key + publication + SMILES + metal (the aggregation unit of the fitter).
  `fit_ineligible_reason` in {`acid_nan_or_nonpositive` (11), `extractant_nan_or_zero` (1),
  `UNIT_SLIP_DUPLICATE` (7)}; the temperature-band restriction is the fitter's.
* `systems/series.csv`: both definitions with a `definition` column (`publication_aware`
  ids `ls_<system>_<metal>_<pub>_<acid>_<ligand>`; `publication_blind` ids `lsb_...`),
  `n_points, n_distinct_mM, mM_min, mM_max, log_d_tracer, log_d_min, log_d_max, log_d_range,
  loading_active (range >= 0.3), tracer_record_id, record_ids`.
* `systems/duplicates.csv`: one row per tied row: `group_id, safe_exp_id, publication_id,
  canonical_smiles, metal, D, log_D, system_id, cond__* (4), tier (unit_slip | tied_d),
  duplicate_flag, kept, series_residual`.
* `results/audit/ingest_audit.json` (the `IngestAudit` of section 4.2, superset of fields),
  `results/audit/manifest.json` (build), `results/audit/audit.json`, `e1_cohort.csv`,
  `e1_groups.csv`, `e2_cohort.csv`, `cohort_sha256.txt`, `manifest_audit.json` (audit; a
  second manifest name because both scripts share `results/audit/`).
* `ingest_corpus(systems_dir, seed=18)` deletes stale `sys_*.json` in the directory first;
  `seed` is unused (nothing in the ingest is stochastic).

## A6. Applicability domains built in `ingest.py`, not `domain.py`

`domain.py` (WB2) owns `build_domain(records, ligand)` and `domain_flags`. The ingest needs the
domains at build time, so `ingest.build_domains` implements the box + hull of section 5.5
locally (scipy `ConvexHull`, `QJ`; < 3 distinct points or collinear -> the segment endpoints /
the single point). Loading fraction uses `LOADING_N_PRIOR = 3.0`, O/A 1 (assumed). Keys
`"<ligand>|<band>"`; only the primary extractant gets domains (synergists have no
concentration). WB2's `build_domain` should give the same box on the same rows; if it does
not, the entry's stored domain is the frozen one.

## A7. `literature.py` extras

* `PC88A_ENTRY_JSON`, `CYANEX272_ENTRY_JSON` (the dicts the two entry functions parse),
  `literature_entries()` (what `g18_build_db.py` writes).
* `SF_ND_PR_PLACEHOLDER[system_id]` gives the SF(Nd/Pr) window machine-readably (PC88A
  central 1.4, range [1.3, 1.5]; Cyanex 272 sweep {1.1, 1.2, 1.3, 1.4}, central 1.25); the
  entries hold the same numbers in the `log_k.Pr` notes. The Cyanex `log_k.Pr` value is
  -3.55 with range [-4.65, -2.36] (= Nd window shifted by log10 of the sweep).
* `PHASE_LITERATURE` (the orchestrator's pre-fit addendum): `loc_metal_M = 0.008 mol/L`,
  `literature`, doi 10.1081/SEI-120016073, merged into `sys_5cb78e5000d40860` by
  `g18_build_db.py`; the note states it applies only at 0.1 M TODGA / n-dodecane / 3 M HNO3 / Nd.
* `MODIFIER_SMILES` (representative isomers of the alcohol additives; identity only).
* `todga_hydrophilic_complexant_placeholder()` has `smiles = None`; the case sweeps
  `log beta_Nd` in [1, 4], `log beta_Pr` in [1, 5.5] (= Nd + delta, delta in [0, 1.5]),
  `log K_H` in [1, 3], regeneration 0.9 in [0.5, 1.0]; concentration 0.05 M in [0.001, 0.2].

## A8. Case files (`cases/*.json`) — the shape WB4 reads

Top level: `name, description, feed_status: "assumed"` (spec: `spec_status`), `anion, acid`,
then every number as a flat `Sourced` JSON object (`Sourced.from_json` parses it): `flow_L_h`
(L/h), `metals_mM: {metal: Sourced}` (mM), `h_M`, `anion_M`, `complexant_total_M`, `sodium_M`
(mol/L), `temperature_C`. `prnd_feed.json.sensitivity` holds `total_metal_M.values`
[0.03, 0.1, 0.3] and `ratio_nd_pr.values` ["1:1", "3:1"]. `prnd_spec.json`: `target: "Nd"`,
`impurities: ["Pr"]`, `purity_basis: "mol"`, `purity_min_grid.values` [0.95, 0.97, 0.99],
`recovery_min_grid.values` [0.80, 0.85, 0.90], `consistency_cell {0.97, 0.85}`, `patent_cell
{0.99, 0.99}`. `todga_prnd_feed.json` adds `system_id` and `complexant_sweep` with the
section 13.5 ranges (`log_beta_Nd`, `delta_log_beta`, `log_k_h`, `regeneration_fraction`).

## A9. Counts that differ from DESIGN.md 0.2 / PRE_REGISTRATION.md section 2

Recorded here and in `DATA_AUDIT.md` section 7 (the pre-registration is not edited):

* PRE_REGISTRATION.md section 2 attributes "42 groups over 18 publications" to
  `sys_5cb78e5000d40860`. Under the system key that system has **14** E1 groups (18
  publications); 42 is the sum over the four TODGA-named systems (nitrate/aliphatic 14,
  `sys_07ee9637c98c1e20` 13, `sys_a2a472b5124b09c6` 14, `sys_ebce91a448faa5c3` 1); the
  system's largest single group spans 18 publications (union over its 14 groups: 27). Likewise
  "TEHDGA 3 / 5" is two systems (`sys_a7195d8a9d8696e0` 2 groups / max 5 publications,
  `sys_120bb57e9148dd0d` 1 / 2). Nine systems carry exactly one E1 group (`sys_120bb57e9148dd0d`,
  `sys_1419f400c83e9ad8`, `sys_267144068d8e09d2`, `sys_81bcc3c06cbf0b4c`, `sys_95746e54b97ae741`,
  `sys_a9fea6791c7a14d6`, `sys_b06c95ac0efc3d4e`, `sys_be340fe5092ae17a`, `sys_ebce91a448faa5c3`).
  The system-level totals — 59 groups, 14 systems — match; `results/audit/e1_cohort.csv` is the
  frozen list.
* The fittable count 233 holds for band 20-30C (NaN temperature included); pooling all bands
  gives 235 fittable groups (E1 unchanged: 59 / 14).
* The 1532 NaN-metal rows of 0.2 are pre-quarantine; after quarantine 1407.
* The loading-active subset is 7 series (the 6 named plus the Ce series), as section 2 states.
