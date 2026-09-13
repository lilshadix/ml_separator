# Gen18 design — the process-design chain (final, synthesised 2026-09-13)

This is the implementation contract for `generations/gen18_process/`. It was built from the
numerics design (B, the panel winner), with the chemistry of design A and the validity apparatus
of design C grafted in, and every defect named by the two judgments fixed (the ledger is in §0.4).
Four engineers implement it in parallel, each owning distinct files (§15), without talking to each
other: every interface, equation, algorithm, file format and tolerance they need is fixed here.
Where this file and `BRIEF.md` disagree, this file wins; where this file is silent, `BRIEF.md`
wins. `PRE_REGISTRATION.md` is the frozen protocol for the only corpus-fitted model.

## 0. Governing statement, verified facts, defect ledger

### 0.1 What gen18 builds

The chain *extraction-system composition -> D of each metal at the given loading -> countercurrent
cascade (extraction + scrub + strip + organic recycle) -> purity, recovery, throughput, reagent
consumption -> local regime optimisation*, on this machine, from the frozen bundle plus literature
entries. The product is a computed regime with its uncertainty and its applicability flags, not a
regressor. Every D carries a provenance status; every regime carries per-axis domain flags and a
chemistry-flag set; missing measurements are recorded as missing and never reconstructed.

Model idealisations (stated once, carried as a static `assumptions` tuple in every result):
concentrations for activities; complete dimerisation of acidic organophosphorus extractants;
fixed stoichiometries; no water co-extraction; no mixed organic complexes; no ligand aggregation;
constant phase volumes; temperature enters only through the parameter set's temperature band.
The `ActivityModel` hook (§5.6) is where the first non-ideal term enters without changing an
interface; its adoption is an exploratory arm of the pre-registration, never a silent default.

### 0.2 Data facts verified on 2026-09-13 (read-only checks on the bundle joined to gen6 provenance)

These numbers are recomputed by `scripts/g18_build_db.py` and `scripts/g18_audit.py` and written to
`DATA_AUDIT.md` before any fit; the audit, not this file, is the frozen source of the cohort counts.

| fact | value under this design's keys |
|---|---|
| bundle rows -> after gen13 quarantine | 5992 -> 5860 (129 TODGA-structure-under-foreign-name rows, 3 sentinel rows at log D <= -6) |
| publications (gen6 `publication_id`, none missing) | 105 |
| systems under the key of §4.2 | 287 |
| loading series, publication-aware (>= 3 distinct `cond__metal_concentration_mM` at fixed system, metal, publication, acid, extractant concentration) | **10** (listed in §4.5; the brief's "16" is publication-blind and pre-quarantine; the publication-blind count after quarantine is 11) |
| D3DODGA/Nd pub_0e7f3e0563 duplicate rows | **7** rows of the 3 M HNO3 series (0.144–0.978 mM) carry D bit-identical to 7 rows of the 1 M series at metal concentrations exactly 144.24x larger; 144.24 is the molar mass of Nd, so this is a g/L-versus-mM unit slip in one copy (§4.6) |
| suspect tied-D groups corpus-wide (same publication, SMILES, metal, D equal to 6 significant digits at different nominal conditions) | 130 groups, 324 rows, 24 publications (most are tied plateau values, not unit slips; §4.6 separates the two) |
| replicate groups (exact 64-column condition key + publication + SMILES + metal) and median within-group sd of log D | 282 groups, **0.225** (gen13 reported 0.237 per cell-metal under its cell key) |
| system-metal groups fittable for the acid/extractant slopes (>= 6 aggregated points, >= 3 distinct levels on log acid or log extractant) | 233; of these **59 groups in 14 systems span >= 2 publications** (TODGA 42 groups / 18 publications; TBDGA 7 / 4; TEHDGA 3 / 5; eight further systems with one multi-publication group each) |
| TODGA / nitrate / aliphatic hydrocarbon / no additive system (`sys_5cb78e5000d40860`) | 514 rows, 28 publications, 14 metals, acid 0.0093–5.0 M, extractant 0.00023–0.3 M, metal 1e-4–2180 mM, 5–45 °C; Pr 28 rows, Nd 66 rows, 8 publications with both |
| TODGA/Nd 3 M HNO3 / 0.1 M loading series pub_5a68dc5665 (n-dodecane, 25 °C) | 6 points, 4.9–12 mM, D 23.5 -> 0.60 (log D 1.37 -> -0.22); `safe_exp_id` Ca_SAFE:2217, 2219, 2220, 2221, 2222, 2224; series caf22524baa8080e |
| TODGA/Ce 3 M / 0.1 M pub_917a4583d4 | 8 points, 0.011–52 mM, log D 1.00 -> 1.65, **rising** with metal concentration |
| TODGA/Nd pub_d3c970567f, three series at 3 M / 0.1, 0.2, 0.3 M | flat (log D -0.03 to 0.00) |
| corpus columns | no O/A, no pH, no equilibrium acidity; `cond__metal_concentration_mM` semantics (initial aqueous, assumed) undocumented; 1532 rows NaN metal concentration |
| corpus coverage of the Pr/Nd case | no PC88A, Cyanex 272, D2EHPA rows at all; the case is literature-parameterised (§13) |
| gen15 deploy model | `generations/gen15_curve/models/deploy_g15.joblib` exists today (gitignored); optional |

Chemical facts used in the placeholders of §3.7 come only from `LITERATURE_NOTES.md` (same
directory, 2026-09-13) and from the task document; none is invented here and every one is marked
`ASSUMED_PLACEHOLDER` with the DOI where a person must verify it.

### 0.3 The one-line rule per component

* Database: every numeric field is a `Sourced` object with a status and a source; the validator
  refuses unsourced numbers, assumed phase/loading values, gen15-sourced D and cross-anion K.
* D models: coupled multi-metal stage equilibrium in free ligand(s), equilibrium [H+], aqueous
  anion and free complexant; constant D only as a labelled sanity limit and as the Kremser oracle.
* Cascade: two aqueous circuits, one organic with recycle and bleed; full-cascade Newton with
  analytic Jacobian from a Kremser-exact initial solve; successive substitution as fallback and
  cross-check; failure is a status, never an exception; balances recomputed from the stream table.
* Evaluation: pre-registered, publication hold-out, reliability before interpretation, macro over
  systems, nulls reported, comparisons counted; known-extractant regime only, no zero-shot claim.
* Case study: literature placeholders with ranges, intervals over parameter draws, consistency
  checks that are never called validation.

### 0.4 Defect ledger (every judged defect and where it is fixed)

| defect (judgments 1 and 2) | fixed in |
|---|---|
| unbounded Wegstein / no positivity guard / no warm start (A, C) | §7.5–7.6: bounded Wegstein factor in [-5, 0], fraction-to-boundary, Kremser init |
| stage or cascade raises on non-physical input (A) | §6.4, §7.7: `status` enum, NaN metrics, optimiser counts failures |
| nitrate not depleted by transport (A), transported but inert (B), absent (C) | §5.2, §6.2: separate aqueous anion unknown, transported 3 per metal plus HNO3·L, acts on D through `p_anion` |
| non-smooth alkali-reserve titration breaks Newton (A) | §6.3, §7.5: explicit two-branch rule with active set; bracketed inner solve; SS fallback |
| HNO3 uptake by DGA absent (B, C) | §5.2: `k_acid_uptake` species HNO3·L, removed from aqueous acid, returned at strip, `ACID_UPTAKE_UNMODELLED` when null |
| acid floor clamp discards base (B); acid balance on absolute organic metal, wrong sign at strip (C) | §6.3: change-in-organic-metal form with sign reversal; base that exceeds protons stays in the reserve; `ALKALI_EXCESS` makes the regime inadmissible |
| saponification as an aqueous base feed (B) / no reserve (C) | §6.3: organic alkali reserve `B_org` travelling with the organic, re-created each pass |
| recovery_from_feed by net subtraction (B, C) | §7.8: exact origin-labelled linear pass at converged D; four reported quantities; identity test |
| phi confounded with the intercept (B) | §10.4: log K re-anchored at the tracer point for every phi |
| decision rule on TODGA only (B) / on 5 systems (A) / no null fallback (A, B) | `PRE_REGISTRATION.md` §5: macro over 14 systems, wins count, paired system bootstrap, explicit null fallback to `NearestConditionD` |
| split-half within a publication (A) | `PRE_REGISTRATION.md` §6: split by publication only |
| hard test on TODGA n vs the IQR (A) / relaxed window (B) | §12.3: the test exists as the brief demands, marked `validation`; a failure is a reported defect, not a suite blocker |
| TODGA/Ce rising series and flat series not pre-registered (all) | `PRE_REGISTRATION.md` §7: they count as they fall; no exclusion |
| duplicate rule excludes both copies (A) | §4.6: unit-slip rule keeps the self-consistent copy; tied plateaus are kept and counted |
| `Stream.h` overloaded (A); untyped flag strings (A) | §6.1 separate `AqStream` / `OrgStream`; §1.4 `Flag` enum |
| two loading ceilings for cation exchange (A) | §5.4: model cap lambda <= 1 (i.e. [M]_org <= [HA]_T/6), `HIGH_LOADING` above 0.5, the chemical ceiling [HA]_T/3 stated as outside the model |
| multi-ligand D undefined (A, B, C) | §5.3: D_M = sum of independent species contributions, each depleting its own ligand |
| no scaffold/variant tag (B, C) | §3.3 `scaffold_id`, `variant_tag` |
| pre-averaged replicates in the database (B) | §3.5: raw rows kept; aggregation in the fitter |
| feed-entry / scrub-return fixed (B, C) | §7.2: `feed_stage`, `scrub_return_stage` in the spec and the design space |
| Thakur's numbers used as the spec (A) | §13.2: spec grid; Thakur and the patent circuit are consistency checks only |
| no placeholder path for the case (A), no placeholder guard (B), validator/case contradiction (C) | §3.7, §3.6 rule V9, §13 |
| complexant without protonation (C) | §5.2 (c) |
| no throughput / loading / phase items in metrics (C) | §8 |
| wall-clock assertion in a test (B) | §11 `g18_bench.py` records timing; no test asserts it |
| unsourced sulfate statement (A) | removed; sulfate is a schema value only |
| temperature pooled silently (B, C) | §4.2: temperature band is part of the parameter-set key |
| memory (C) | §4.1: ~85 of 2261 columns read |
| no end-to-end budget | §11.3 |

## 1. Conventions binding on every module

### 1.1 Units and symbols

Concentrations mol/L (M) in the phase named; feed metals may be entered in mM in JSON files and are
converted at the boundary (`mM * 1e-3`); flows L/h; O/A is the volumetric ratio V_org/V_aq (per stage:
organic flow / aqueous flow of that stage's circuit); temperature °C; time h; log means log10.
Metals are element symbols (`gen13sep.metals.LANTHANIDES`); the organic ligand index is `k`, the
metal index `i`, the stage index `j`. Ligand concentrations for dimeric acidic extractants are
**dimer** concentrations `[(HA)2]` everywhere in the code; the JSON entry records the monomer
formal concentration `[HA]_T` and the loader converts `[(HA)2]_T = [HA]_T / 2`.

### 1.2 Provenance statuses (one enum, used everywhere)

`ProvStatus` in {`measured_corpus`, `measured_literature`, `fitted_from_corpus`, `literature`,
`assumed`, `unknown`}. Rules: a non-null value with `unknown` is a validator error; `literature`
and `measured_literature` require a DOI and a locator; `measured_corpus` requires `safe_exp_ids`
and a `publication_id`; `fitted_from_corpus` requires `model_id` and `fit_manifest_sha256` and a
`range`; `assumed` requires `assumed_label == "ASSUMED_PLACEHOLDER"`, a `range`, and a `source`
that names the DOI (or "none") where the number must come from.

### 1.3 The `Sourced` object

Every numeric quantity in the database is
```json
{"value": 3.0, "unit": "1", "status": "assumed", "assumed_label": "ASSUMED_PLACEHOLDER",
 "range": [2.0, 3.0], "source": {"kind": "doi", "doi": "10.1038/s41598-020-74041-9",
 "locator": "Results, slope analysis", "publication_id": null, "safe_exp_ids": []},
 "model_id": null, "fit_manifest_sha256": null, "note": "free text"}
```
`value: null` means "not known" and is the only way to say so; `unit` is a UCUM-like string
(`"mol/L"`, `"1"`, `"kJ/mol"`, `"s"`, `"mol/L_aq"`). The dataclass is `Sourced(value, unit,
provenance)` with `Provenance(status, source: Source, range, assumed_label, model_id,
fit_manifest_sha256, note)` and `Source(kind in {"corpus","doi","none","model"}, doi, locator,
publication_id, safe_exp_ids: tuple[str, ...])`.

### 1.4 Flags (typed; free strings are a validator error)

```python
class Flag(str, Enum):
    # chemistry / model validity
    HIGH_LOADING = "HIGH_LOADING"                       # loading fraction > 0.5 on any ligand
    LOADING_CAP_HIT = "LOADING_CAP_HIT"                 # free ligand < 1e-6 * total on any ligand
    THIRD_PHASE_RISK = "THIRD_PHASE_RISK"               # sourced LOC exceeded
    PHASE_BEHAVIOUR_UNKNOWN = "PHASE_BEHAVIOUR_UNKNOWN" # LOC null and loading fraction > 0.3
    ACID_UPTAKE_UNMODELLED = "ACID_UPTAKE_UNMODELLED"   # solvating, K_H null, aqueous acid > 1 M
    MEDIUM_STRENGTH_UNMODELLED = "MEDIUM_STRENGTH_UNMODELLED"  # anion varies, no beta_anion
    ANION_ACID_CONFOUNDED = "ANION_ACID_CONFOUNDED"     # corpus p_eff applied where [anion] != [H+]
    MIXED_ORGANIC_UNMODELLED = "MIXED_ORGANIC_UNMODELLED"  # two extractant-role ligands
    OA_ASSUMED = "OA_ASSUMED"                           # parameter fitted on corpus rows with no O/A
    ALKALI_EXCESS = "ALKALI_EXCESS"                     # base exceeded protons in a stage (inadmissible)
    SAPONIFICATION_RANGE_UNKNOWN = "SAPONIFICATION_RANGE_UNKNOWN"
    EQUILIBRIUM_ACID_ASSUMED_NOMINAL = "EQUILIBRIUM_ACID_ASSUMED_NOMINAL"  # corpus fit uses nominal acid
    # applicability
    OOD_ACID = "OOD_ACID"; OOD_LIGAND = "OOD_LIGAND"; OOD_METAL = "OOD_METAL"; OOD_LOADING = "OOD_LOADING"
    OOD_ANION = "OOD_ANION"; OOD_COMPLEXANT = "OOD_COMPLEXANT"; OOD_OA = "OOD_OA"
    OOD_TEMPERATURE = "OOD_TEMPERATURE"; OOD_DILUENT = "OOD_DILUENT"; OOD_MODIFIER = "OOD_MODIFIER"
    OOD_HULL = "OOD_HULL"                               # outside the (log acid, log ligand) convex hull
    # solver
    NOT_CONVERGED = "NOT_CONVERGED"; JACOBIAN_SINGULAR = "JACOBIAN_SINGULAR"
    COST_INCOMPLETE = "COST_INCOMPLETE"; LIGAND_LOSS_NOT_MEASURED = "LIGAND_LOSS_NOT_MEASURED"
```
`INADMISSIBLE_FLAGS = {LOADING_CAP_HIT, THIRD_PHASE_RISK, ALKALI_EXCESS, NOT_CONVERGED}`;
`OOD_FLAGS = {OOD_*}`; everything else is a caveat. Regime status (§8.3): `IN_DOMAIN` (no OOD, no
inadmissible, no caveat), `IN_DOMAIN_WITH_CAVEATS`, `OUT_OF_DOMAIN`, `INADMISSIBLE`.

### 1.5 Determinism, errors, resources

Seeds: every stochastic step takes `seed: int` (default 18) and uses `numpy.random.default_rng(seed)`
or `scipy.stats.qmc.LatinHypercube(d, scramble=True, seed=seed)`. No wall-clock value appears in
any results file except `results/bench/*.json`. Solvers never raise on non-convergence or on a
physically infeasible specification: they return a `status`. They raise `ValueError` only on
malformed input (negative flow, unknown metal, missing parameter), and the optimiser catches
`ValueError` per candidate and records it as `status = "invalid_spec"`. One Python process; the
ingest reads only the columns listed in §4.1; peak memory target < 1 GB. Python 3.14, numpy,
scipy, pandas 3, scikit-learn, rdkit only.

## 2. Layout and ownership

```
generations/gen18_process/
  BRIEF.md  DESIGN.md  PRE_REGISTRATION.md  LITERATURE_NOTES.md  DATA_AUDIT.md  README.md  GEN18_REPORT.md
  gen18proc/
    __init__.py  paths.py  systems.py  literature.py  ingest.py            (WB1)
    dmodel.py  domain.py  equilibrium.py  testsystems.py                   (WB2)
    cascade.py  metrics.py                                                 (WB3)
    evalproto.py  optimize.py  screen.py  report.py                        (WB4)
  systems/            <system_id>.json, corpus_records.csv, exclusions.csv, series.csv,
                      duplicates.csv, registry.json, INDEX.md              (written by scripts)
  cases/              prnd_feed.json, prnd_spec.json, todga_prnd_feed.json (WB1 writes, §13)
  config/             prices.json (WB3), design_spaces.json (WB4)
  scripts/            §11
  tests/              §12
  results/            audit/ dmodels/ eval/ loading/ bench/ case_prnd/ recipes/ optimize/ MANIFEST.sha256
```
`paths.py` re-exports `BUNDLE_PARQUET`, `GEN6_PROVENANCE_PARQUET`, `CHEMISTRY_MAP_PARQUET`,
`assert_bundle_unchanged`, `sha256_of` from `gen13sep.paths` (inserting
`generations/gen13_separation` on `sys.path`), defines `G18_ROOT`, `SYSTEMS_DIR`, `RESULTS_DIR`,
`CASES_DIR`, `CONFIG_DIR`, and `GEN15_DEPLOY = REPO_ROOT / "generations/gen15_curve/models/deploy_g15.joblib"`.
Every script and test inserts `generations/gen18_process` on `sys.path` and imports `gen18proc`.

## 3. `systems.py` — the extraction-systems database

### 3.1 System identity

```
system_key = (tuple(sorted(canonical SMILES of ligands with role in {extractant, synergist})),
              acid_class,                # nitrate | chloride | sulfate | perchlorate | carboxylate
              diluent_family,            # bundle vocabulary of geom_cond__diluent_family
              tuple(sorted(modifier names)),        # alcohol additives etc.
              tuple(sorted(canonical SMILES of aqueous complexants)))
system_id = "sys_" + blake2b(json.dumps(list(key), separators=(",",":"), ensure_ascii=True).encode(), digest_size=8).hexdigest()
```
Verified ids: PC88A/chloride/aliphatic `sys_29976921e156a0a0`; Cyanex 272/chloride/aliphatic
`sys_f02db527a94a5e86`; TODGA/nitrate/aliphatic `sys_5cb78e5000d40860`. Temperature is **not** in
the system key; it is in the parameter-set key (§3.4). Diluent identity is a record field, not a
key field (the family is); a within-system diluent-identity split is an exploratory sensitivity.

### 3.2 Dataclasses (frozen; these exact names and fields)

```python
class Mechanism(str, Enum): CATION_EXCHANGE = "cation_exchange"; SOLVATING = "solvating"; ANION_EXCHANGE = "anion_exchange"  # schema only

@dataclass(frozen=True)
class LigandSpec:
    name: str; smiles: str; canonical_smiles: str
    role: str                              # "extractant" | "synergist" | "modifier"
    mechanism: Mechanism | None            # None for modifiers
    concentration: Sourced                 # mol/L organic, formal monomer concentration
    aggregation: str                       # "monomer" | "dimer"
    scaffold_id: str | None; variant_tag: str | None     # series A–D linkage
    stoichiometry: Stoichiometry           # see below

@dataclass(frozen=True)
class Stoichiometry:
    ligands_per_metal: Sourced             # q: 3 dimers (cation exchange), n (solvating)
    protons_released_per_metal: Sourced    # p: 3 (cation exchange), 0 (solvating)
    anions_per_metal: Sourced              # z: 0 (cation exchange), 3 (solvating nitrate)

@dataclass(frozen=True)
class ComplexantSpec:
    name: str; smiles: str | None; canonical_smiles: str | None
    concentration: Sourced                 # mol/L aqueous, total
    log_beta: Mapping[str, tuple[Sourced, ...]]   # metal -> (log beta_1, log beta_2, ...) cumulative, M^-m
    protonation_logk: tuple[Sourced, ...]  # cumulative log K_H,j
    regeneration_fraction: Sourced         # recovered fraction of complexant per pass; ASSUMED with range when unknown

@dataclass(frozen=True)
class MediumSpec:
    acid: str; acid_class: str; anion: str  # anion in {"nitrate","chloride","sulfate","perchlorate","carboxylate"}
    salting_agent: str | None; salting_anion_M: Sourced; ionic_strength_M: Sourced
    temperature_C: Sourced

@dataclass(frozen=True)
class CationExchangeParams:
    medium_anion: str                      # must equal MediumSpec.anion (validator V7)
    temperature_band: str                  # e.g. "20-30C"
    log_k: Mapping[str, Sourced]           # per metal, dimer basis, concentration basis
    a_dimer: Sourced                       # ideal 3
    b_proton: Sourced                      # ideal 3
    k_reported_as: str                     # "log_k" | "pH50" (pH50 -> log K only in the tracer limit, §4.7)
    beta_anion: Mapping[str, tuple[Sourced, ...]] | None   # per metal anion complexation (null -> not modelled)
    saponification_degree_studied: tuple[float, float] | None
    delta_h_kj_mol: Sourced | None

@dataclass(frozen=True)
class SolvatingParams:
    medium_anion: str; temperature_band: str
    log_k: Mapping[str, Sourced]           # per metal, at [anion] and [L] in mol/L
    n_solvation: Sourced
    p_anion: Sourced                       # exponent on aqueous anion; corpus p_eff lands here
    p_h: Sourced                           # exponent on [H+] beyond the anion; 0 for the corpus fit (assumed)
    k_acid_uptake: Sourced                 # K_H for HNO3 + L <=> HNO3.L, (mol/L)^-2; value None -> flag
    delta_h_kj_mol: Sourced | None

@dataclass(frozen=True)
class PhaseBehaviour:                      # each value None unless measured or literature (validator V5)
    loc_metal_M: Sourced; loc_acid_M: Sourced; third_phase_observed: Sourced
    disengagement_s: Sourced; ligand_loss_mol_per_L_aq: Sourced
    max_loading_fraction_studied: Sourced; regenerability_note: str

@dataclass(frozen=True)
class StreamRecord:                        # a scrub or strip liquor as reported by the source
    kind: str                              # "scrub" | "strip"
    acid_M: Sourced; anion_M: Sourced; complexant_M: Sourced; metals_mM: Mapping[str, Sourced]
    oa_ratio: Sourced; stages: Sourced; fraction_removed: Mapping[str, Sourced]; note: str

@dataclass(frozen=True)
class DistributionRecord:
    record_id: str                         # corpus: safe_exp_id; literature: "doi:<doi>#<locator>#<n>"
    metal: str; d: float; log_d: float
    acid_nominal_M: float | None; acid_eq_M: float | None; anion_M: float | None
    ligand_M: Mapping[str, float]          # per organic ligand name, formal concentration
    complexant_M: float | None
    metals_initial_mM: Mapping[str, float] # corpus: {metal: cond__metal_concentration_mM} (semantics ASSUMED initial aqueous)
    oa_ratio: float | None                 # None in the corpus
    temperature_C: float | None; contact_time_min: float | None
    diluent_name: str | None
    publication_id: str | None; experiment_series_id: str | None; replicate_id: str | None
    loading_series_id: str | None; is_tracer: bool
    fit_eligible: bool; fit_ineligible_reason: str | None
    duplicate_flag: str | None             # None | "UNIT_SLIP_DUPLICATE" | "TIED_D"
    provenance: Provenance

@dataclass(frozen=True)
class ApplicabilityDomain:                 # intervals actually studied; log10 for concentration axes
    anion: str; diluent_family: str; modifiers: tuple[str, ...]
    log_acid: tuple[float, float]; log_ligand: Mapping[str, tuple[float, float]]
    log_metal_total_mM: tuple[float, float] | None; loading_fraction: tuple[float, float] | None
    log_complexant: tuple[float, float] | None; oa_ratio: tuple[float, float] | None
    temperature_C: tuple[float, float] | None; saponification_degree: tuple[float, float] | None
    hull_vertices: tuple[tuple[float, float], ...] | None   # (log acid, log primary ligand)
    n_records: int; n_publications: int

@dataclass(frozen=True)
class SystemEntry:
    schema_version: str                    # "gen18.1"
    system_id: str; name: str; family: str # acidic_organophosphorus | diglycolamide | phen_carboxamide | n_donor | carboxylic_acid | amine | other
    origin: str                            # corpus | literature | mixed | hypothetical
    organic_ligands: tuple[LigandSpec, ...]
    aqueous_complexants: tuple[ComplexantSpec, ...]
    diluent: Mapping                       # {"name": str|None, "family": str, "components": [{"name","vol_fraction"}]}
    medium: MediumSpec
    params: Mapping[str, Mapping[str, CationExchangeParams | SolvatingParams]]   # ligand name -> temperature band -> params
    phase: PhaseBehaviour
    stream_records: tuple[StreamRecord, ...]
    oxidation_state_routes: tuple[Mapping, ...]   # {"metal","from","to","method","doi","status"}
    direction_prior: Mapping | None        # {"pair","sign","source":"gen15 deploy_g15.joblib","note":"pre-screen only"}
    applicability: Mapping[str, ApplicabilityDomain]   # per parameter-set key "<ligand>|<band>"
    records: tuple[DistributionRecord, ...]
    notes: str
```

### 3.3 Two-ligand systems and series A–D

`organic_ligands` is a tuple; a second `extractant`/`synergist` entry is first-class and enters the
system key. `scaffold_id` (free string, e.g. `"DGA_core"`) and `variant_tag` (e.g. `"2-Me,(R)"`)
link stereo/substituent variants across systems; `load_registry()` exposes `scaffold_id` so
variants can be grouped. Hydrophilic/lipophilic pairs (series D) are a system whose
`aqueous_complexants` carries the hydrophilic member with the same `scaffold_id`.

### 3.4 Parameter sets and temperature bands

`params[ligand_name][band]`; band strings are `"<lo>-<hi>C"` with fixed bands `"<20C"`, `"20-30C"`,
`"30-40C"`, `"40-50C"`, `">=50C"`. A record with NaN temperature is assigned to `"20-30C"` with
`fit_ineligible_reason = None` and a note `temperature_assumed_20_30C` (the row count is reported).
`build_system_model` selects the band containing the design temperature; absent -> `OOD_TEMPERATURE`
with the nearest band's parameters.

### 3.5 Records

Corpus records are one per bundle row; replicates are **not** averaged in the database (the fitter
aggregates and records `n_replicates`). `d` and `log_d` are copied from the bundle unchanged.

### 3.6 Validator `validate_entry(entry) -> list[Violation]` (level error|warning, path, message)

Errors: V1 unknown `schema_version`; V2 any `Sourced` with `value` not None and `status == unknown`;
V3 status rules of §1.2 (missing DOI/locator, missing safe_exp_ids/publication_id, missing
model_id/manifest, missing range); V4 `assumed` without `assumed_label == "ASSUMED_PLACEHOLDER"` or
without a range or without a source (`kind: "doi"` or `kind: "none"` with a note saying why); V5 any
`PhaseBehaviour` value or `ApplicabilityDomain` entered by hand that is non-null with status
`assumed` (these stay null unless measured or literature); V6 `fitted_from_corpus` or any D record
whose `model_id` or `source.locator` contains `gen15` or `deploy_g15`; V7 `params[k][band].medium_anion
!= medium.anion`; V8 a parameter block missing `log_k` for a metal that appears in `records` or in a
`StreamRecord.metals_mM`; V9 (case-study guard) any `assumed` numeric with `range == None`; V10 SMILES
that RDKit cannot parse or whose canonical form differs from `canonical_smiles`; V11 `system_id`
differs from the recomputed hash; V12 duplicate `record_id`; V13 record with `d <= 0` or non-finite
`log_d`; V14 `mechanism == cation_exchange` with `protons_released_per_metal.value != 3` or
`aggregation != "dimer"` unless the value carries a literature source; V15 a complexant without
`log_beta` for every metal in `records` and in the case feed (checked again by `build_system_model`
for the actual feed); V16 flags or statuses outside the enums; V17 `diluent.family` outside the bundle
vocabulary. Warnings: W1 fewer than 2 metals; W2 no temperature; W3 records without publication_id;
W4 `k_acid_uptake.value is None` for a solvating ligand.

`load_system(path) -> SystemEntry` raises `SystemValidationError(violations)` on any error;
`write_system(entry, path)` serialises with `sort_keys=True, indent=2, ensure_ascii=True` and floats
formatted by `repr`. `load_registry(systems_dir) -> pd.DataFrame` (system_id, name, family,
mechanism, origin, n_records, n_publications, metals, scaffold_ids).

### 3.7 Example entry 1 — PC88A / Pr–Nd, chloride, literature-derived (every number ASSUMED_PLACEHOLDER)

Derivation of the placeholder ranges is recorded in the `note` fields; the numbers come from
`LITERATURE_NOTES.md` S1/S2 and the task document; the person verifying them replaces
`assumed` by `literature` with the exact locator. `log_k` windows were computed by
`literature.derive_logk_from_extraction` (§4.7) from S2's Nd extraction at 0.8 M extractant,
O/A 1, 1500 mg/L each of Nd, Tb, Dy, using the reported equilibrium pH interval and ideal exponents;
they are placeholders, not literature values.

```json
{
  "schema_version": "gen18.1",
  "system_id": "sys_29976921e156a0a0",
  "name": "PC88A (HEH[EHP]) in aliphatic diluent, chloride medium",
  "family": "acidic_organophosphorus",
  "origin": "literature",
  "organic_ligands": [
    {"name": "PC88A", "smiles": "CCCCC(CC)COP(=O)(O)CC(CC)CCCC",
     "canonical_smiles": "CCCCC(CC)COP(=O)(O)CC(CC)CCCC",
     "role": "extractant", "mechanism": "cation_exchange", "aggregation": "dimer",
     "scaffold_id": "phosphonic_monoester", "variant_tag": null,
     "concentration": {"value": 0.8, "unit": "mol/L", "status": "assumed", "assumed_label": "ASSUMED_PLACEHOLDER",
        "range": [0.2, 1.5], "source": {"kind": "doi", "doi": "10.1038/s41598-020-74041-9", "locator": "extractant concentration series, 0.8 mol/L point", "publication_id": null, "safe_exp_ids": []},
        "model_id": null, "fit_manifest_sha256": null,
        "note": "formal monomer concentration used in S2; the case sweeps the range as a design variable"},
     "stoichiometry": {
       "ligands_per_metal": {"value": 3.0, "unit": "1", "status": "assumed", "assumed_label": "ASSUMED_PLACEHOLDER", "range": [3.0, 3.0],
          "source": {"kind": "doi", "doi": "10.1038/s41598-020-74041-9", "locator": "mechanism nRE3+ + n(HA2)org", "publication_id": null, "safe_exp_ids": []},
          "model_id": null, "fit_manifest_sha256": null, "note": "dimers per Ln3+, ideal dilute-regime stoichiometry"},
       "protons_released_per_metal": {"value": 3.0, "unit": "1", "status": "assumed", "assumed_label": "ASSUMED_PLACEHOLDER", "range": [3.0, 3.0],
          "source": {"kind": "doi", "doi": "10.1038/s41598-020-74041-9", "locator": "mechanism", "publication_id": null, "safe_exp_ids": []},
          "model_id": null, "fit_manifest_sha256": null, "note": ""},
       "anions_per_metal": {"value": 0.0, "unit": "1", "status": "assumed", "assumed_label": "ASSUMED_PLACEHOLDER", "range": [0.0, 0.0],
          "source": {"kind": "none", "doi": null, "locator": "cation exchange transports no anion", "publication_id": null, "safe_exp_ids": []},
          "model_id": null, "fit_manifest_sha256": null, "note": ""}}}
  ],
  "aqueous_complexants": [],
  "diluent": {"name": "kerosene", "family": "aliphatic_hydrocarbon", "components": [{"name": "kerosene", "vol_fraction": 1.0}]},
  "medium": {"acid": "HCl", "acid_class": "chloride", "anion": "chloride", "salting_agent": null,
     "salting_anion_M": {"value": null, "unit": "mol/L", "status": "unknown", "assumed_label": null, "range": null, "source": {"kind": "none", "doi": null, "locator": null, "publication_id": null, "safe_exp_ids": []}, "model_id": null, "fit_manifest_sha256": null, "note": ""},
     "ionic_strength_M": {"value": null, "unit": "mol/L", "status": "unknown", "assumed_label": null, "range": null, "source": {"kind": "none", "doi": null, "locator": null, "publication_id": null, "safe_exp_ids": []}, "model_id": null, "fit_manifest_sha256": null, "note": ""},
     "temperature_C": {"value": 25.0, "unit": "Cel", "status": "assumed", "assumed_label": "ASSUMED_PLACEHOLDER", "range": [20.0, 30.0], "source": {"kind": "doi", "doi": "10.1038/s41598-020-74041-9", "locator": "298 K", "publication_id": null, "safe_exp_ids": []}, "model_id": null, "fit_manifest_sha256": null, "note": ""}},
  "params": {"PC88A": {"20-30C": {
     "model_type": "cation_exchange", "medium_anion": "chloride", "temperature_band": "20-30C",
     "k_reported_as": "log_k",
     "log_k": {
       "Nd": {"value": -1.95, "unit": "1", "status": "assumed", "assumed_label": "ASSUMED_PLACEHOLDER", "range": [-2.9, -1.0],
              "source": {"kind": "doi", "doi": "10.1038/s41598-020-74041-9", "locator": "Nd 64 % extracted at 0.8 M PC88A, initial pH 4.0, equilibrium pH 1.02-1.42, 1500 mg/L each Nd/Tb/Dy, A/O 1", "publication_id": null, "safe_exp_ids": []},
              "model_id": null, "fit_manifest_sha256": null,
              "note": "derive_logk_from_extraction: log D_Nd = 0.250; sum [M]_org = 24.3 mM; [(HA)2]_f = 0.400 - 3*0.0243 = 0.327 M; a = b = 3; pH_eq in [1.02, 1.42] gives [-2.55, -1.35]; widened by 0.3 for exponent uncertainty; with b = 2.22 (S2 slope) the window is [-1.45, -0.56]. Verify against Banda 2014 doi 10.1016/j.jiec.2014.03.002 and Thakur 1993 doi 10.1016/0304-386X(93)90084-Q."},
       "Pr": {"value": -2.10, "unit": "1", "status": "assumed", "assumed_label": "ASSUMED_PLACEHOLDER", "range": [-3.08, -1.11],
              "source": {"kind": "doi", "doi": "10.1016/j.jiec.2014.03.002", "locator": "maximum SF(Nd/Pr) about 1.5 (task document); patent EP2388344A1: PC-88A SF(Nd/Pr) 1.4 in kerosene", "publication_id": null, "safe_exp_ids": []},
              "model_id": null, "fit_manifest_sha256": null,
              "note": "log K_Pr = log K_Nd - log10 SF(Nd/Pr), SF placeholder range [1.3, 1.5] (log 0.114-0.176), central 1.4 (log 0.146)"}},
     "a_dimer": {"value": 3.0, "unit": "1", "status": "assumed", "assumed_label": "ASSUMED_PLACEHOLDER", "range": [2.0, 3.0],
        "source": {"kind": "doi", "doi": "10.1038/s41598-020-74041-9", "locator": "log D vs log[extractant] slopes 2-3", "publication_id": null, "safe_exp_ids": []}, "model_id": null, "fit_manifest_sha256": null, "note": "ideal 3"},
     "b_proton": {"value": 3.0, "unit": "1", "status": "assumed", "assumed_label": "ASSUMED_PLACEHOLDER", "range": [2.0, 3.0],
        "source": {"kind": "doi", "doi": "10.1038/s41598-020-74041-9", "locator": "log D vs pH slope, PC 88A Nd 2.22", "publication_id": null, "safe_exp_ids": []}, "model_id": null, "fit_manifest_sha256": null, "note": "ideal 3; S2 measured 2.22 for Nd"},
     "beta_anion": null,
     "saponification_degree_studied": null,
     "delta_h_kj_mol": {"value": null, "unit": "kJ/mol", "status": "unknown", "assumed_label": null, "range": null, "source": {"kind": "doi", "doi": "10.1038/s41598-020-74041-9", "locator": "thermodynamics section: endothermic, values not transcribed", "publication_id": null, "safe_exp_ids": []}, "model_id": null, "fit_manifest_sha256": null, "note": ""}}}},
  "phase": {
     "loc_metal_M": {"value": null, "unit": "mol/L", "status": "unknown", "assumed_label": null, "range": null, "source": {"kind": "none", "doi": null, "locator": null, "publication_id": null, "safe_exp_ids": []}, "model_id": null, "fit_manifest_sha256": null, "note": ""},
     "loc_acid_M": {"value": null, "unit": "mol/L", "status": "unknown", "assumed_label": null, "range": null, "source": {"kind": "none", "doi": null, "locator": null, "publication_id": null, "safe_exp_ids": []}, "model_id": null, "fit_manifest_sha256": null, "note": ""},
     "third_phase_observed": {"value": null, "unit": "1", "status": "unknown", "assumed_label": null, "range": null, "source": {"kind": "none", "doi": null, "locator": null, "publication_id": null, "safe_exp_ids": []}, "model_id": null, "fit_manifest_sha256": null, "note": ""},
     "disengagement_s": {"value": null, "unit": "s", "status": "unknown", "assumed_label": null, "range": null, "source": {"kind": "none", "doi": null, "locator": null, "publication_id": null, "safe_exp_ids": []}, "model_id": null, "fit_manifest_sha256": null, "note": ""},
     "ligand_loss_mol_per_L_aq": {"value": null, "unit": "mol/L_aq", "status": "unknown", "assumed_label": null, "range": null, "source": {"kind": "none", "doi": null, "locator": null, "publication_id": null, "safe_exp_ids": []}, "model_id": null, "fit_manifest_sha256": null, "note": ""},
     "max_loading_fraction_studied": {"value": null, "unit": "1", "status": "unknown", "assumed_label": null, "range": null, "source": {"kind": "none", "doi": null, "locator": null, "publication_id": null, "safe_exp_ids": []}, "model_id": null, "fit_manifest_sha256": null, "note": ""},
     "regenerability_note": "stripped organic returns as HA; re-saponification per pass is a design variable"},
  "stream_records": [
    {"kind": "strip", "acid_M": {"value": null, "unit": "mol/L", "status": "unknown", "assumed_label": null, "range": null, "source": {"kind": "doi", "doi": "10.1038/s41598-020-74041-9", "locator": "0.5 mol/L oxalic acid strips > 99.9 % Nd from loaded D2EHPA, 2 stages, O/A 4 (D2EHPA, not PC88A)", "publication_id": null, "safe_exp_ids": []}, "model_id": null, "fit_manifest_sha256": null, "note": "recorded for orientation only; not used by the model"},
     "anion_M": {"value": null, "unit": "mol/L", "status": "unknown", "assumed_label": null, "range": null, "source": {"kind": "none", "doi": null, "locator": null, "publication_id": null, "safe_exp_ids": []}, "model_id": null, "fit_manifest_sha256": null, "note": ""},
     "complexant_M": {"value": null, "unit": "mol/L", "status": "unknown", "assumed_label": null, "range": null, "source": {"kind": "none", "doi": null, "locator": null, "publication_id": null, "safe_exp_ids": []}, "model_id": null, "fit_manifest_sha256": null, "note": ""},
     "metals_mM": {}, "oa_ratio": {"value": null, "unit": "1", "status": "unknown", "assumed_label": null, "range": null, "source": {"kind": "none", "doi": null, "locator": null, "publication_id": null, "safe_exp_ids": []}, "model_id": null, "fit_manifest_sha256": null, "note": ""},
     "stages": {"value": null, "unit": "1", "status": "unknown", "assumed_label": null, "range": null, "source": {"kind": "none", "doi": null, "locator": null, "publication_id": null, "safe_exp_ids": []}, "model_id": null, "fit_manifest_sha256": null, "note": ""},
     "fraction_removed": {}, "note": "placeholder stream record; PC88A strip liquor not transcribed"}
  ],
  "oxidation_state_routes": [],
  "direction_prior": null,
  "applicability": {"PC88A|20-30C": {
     "anion": "chloride", "diluent_family": "aliphatic_hydrocarbon", "modifiers": [],
     "log_acid": [-1.42, -1.02], "log_ligand": {"PC88A": [-0.699, 0.0]},
     "log_metal_total_mM": [1.0, 1.7], "loading_fraction": [0.0, 0.2], "log_complexant": null,
     "oa_ratio": [1.0, 1.0], "temperature_C": [25.0, 25.0], "saponification_degree": null,
     "hull_vertices": null, "n_records": 0, "n_publications": 0,
     "_status": "assumed", "_assumed_label": "ASSUMED_PLACEHOLDER", "_range_note": "box transcribed from S2 conditions (equilibrium pH 1.02-1.42, 0.2-1.0 M extractant, 10-45 mM metal); to be replaced by the transcribed Banda/Thakur condition ranges"}},
  "records": [],
  "notes": "Corpus has no PC88A rows. Consistency targets, never validation: Thakur 1993 (doi 10.1016/0304-386X(93)90084-Q) > 5 kg Nd2O3 at 97 % purity, > 85 % recovery, counter-current PC88A; EP2388344A1 PC-88A Nd/Pr circuit 72 extraction + 72 scrub + 8 strip stages at SF 1.4."
}
```
The `applicability` block of a literature entry is the only hand-entered domain and carries its own
`_status`; validator V5 allows it because the block is `assumed` **with** a range note and no phase
value. The Cyanex 272 entry (`sys_f02db527a94a5e86`) has the same shape with `log_k.Nd`
placeholder range `[-4.5, -2.4]` (S2: 27 % Nd extracted, equilibrium pH 1.22–1.70, free dimer
0.353 M, ideal exponents give [-4.18, -2.74], widened by 0.3), `b_proton` range `[2.0, 3.0]` (S2
slope 2.0), and `log_k.Pr = log_k.Nd - log10 SF` with SF an **assumed sweep** {1.1, 1.2, 1.3, 1.4}
(no source found; `LITERATURE_NOTES.md` §3), `source.kind = "none"`, note "no Cyanex 272 Nd/Pr
value found on 2026-09-13; candidate sources doi 10.1016/j.hydromet.2014.09.015, doi
10.1016/j.jre.2017.09.016, doi 10.1016/j.mineng.2013.10.021".

### 3.8 Example entry 2 — TODGA / nitrate / aliphatic, corpus-derived

Written by `g18_build_db.py`; `params` is empty until `g18_fit_dmodels.py` writes the fitted block
(shape shown after the entry). Three of the 514 records are shown.

```json
{
  "schema_version": "gen18.1",
  "system_id": "sys_5cb78e5000d40860",
  "name": "TODGA in aliphatic hydrocarbon, nitrate medium, no additive",
  "family": "diglycolamide",
  "origin": "corpus",
  "organic_ligands": [
    {"name": "TODGA", "smiles": "CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC",
     "canonical_smiles": "CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC",
     "role": "extractant", "mechanism": "solvating", "aggregation": "monomer",
     "scaffold_id": "DGA_core", "variant_tag": "N,N,N',N'-tetraoctyl",
     "concentration": {"value": null, "unit": "mol/L", "status": "unknown", "assumed_label": null, "range": null,
        "source": {"kind": "corpus", "doi": null, "locator": "per record: cond__extractant_concentration_M", "publication_id": null, "safe_exp_ids": []},
        "model_id": null, "fit_manifest_sha256": null, "note": "varies per record; see records[].ligand_M"},
     "stoichiometry": {
       "ligands_per_metal": {"value": null, "unit": "1", "status": "unknown", "assumed_label": null, "range": null,
          "source": {"kind": "none", "doi": null, "locator": "set from the fitted n_solvation by g18_fit_dmodels.py", "publication_id": null, "safe_exp_ids": []},
          "model_id": null, "fit_manifest_sha256": null, "note": ""},
       "protons_released_per_metal": {"value": 0.0, "unit": "1", "status": "literature", "assumed_label": null, "range": null,
          "source": {"kind": "doi", "doi": "10.1016/j.jiec.2014.03.002", "locator": "solvating mechanism Ln3+ + 3 NO3- + n L (BRIEF.md section 3b; textbook mechanism, DOI is the brief's reference for the mechanism class)", "publication_id": null, "safe_exp_ids": []},
          "model_id": null, "fit_manifest_sha256": null, "note": "no proton release for neutral solvating extractants"},
       "anions_per_metal": {"value": 3.0, "unit": "1", "status": "literature", "assumed_label": null, "range": null,
          "source": {"kind": "doi", "doi": "10.1016/j.jiec.2014.03.002", "locator": "as above", "publication_id": null, "safe_exp_ids": []},
          "model_id": null, "fit_manifest_sha256": null, "note": "Ln(NO3)3.L_n"}}}
  ],
  "aqueous_complexants": [],
  "diluent": {"name": null, "family": "aliphatic_hydrocarbon", "components": []},
  "medium": {"acid": "HNO3", "acid_class": "nitrate", "anion": "nitrate", "salting_agent": null,
     "salting_anion_M": {"value": 0.0, "unit": "mol/L", "status": "measured_corpus", "assumed_label": null, "range": null, "source": {"kind": "corpus", "doi": null, "locator": "no salting additive column set in any record", "publication_id": null, "safe_exp_ids": []}, "model_id": null, "fit_manifest_sha256": null, "note": ""},
     "ionic_strength_M": {"value": null, "unit": "mol/L", "status": "unknown", "assumed_label": null, "range": null, "source": {"kind": "none", "doi": null, "locator": null, "publication_id": null, "safe_exp_ids": []}, "model_id": null, "fit_manifest_sha256": null, "note": ""},
     "temperature_C": {"value": null, "unit": "Cel", "status": "unknown", "assumed_label": null, "range": null, "source": {"kind": "corpus", "doi": null, "locator": "per record: cond__temperature_C", "publication_id": null, "safe_exp_ids": []}, "model_id": null, "fit_manifest_sha256": null, "note": "5-45 C across records; parameter sets are per temperature band"}},
  "params": {},
  "phase": {
     "loc_metal_M": {"value": null, "unit": "mol/L", "status": "unknown", "assumed_label": null, "range": null, "source": {"kind": "none", "doi": null, "locator": null, "publication_id": null, "safe_exp_ids": []}, "model_id": null, "fit_manifest_sha256": null, "note": "third-phase limits of TODGA are known in the literature but not transcribed; stays null until a DOI is entered"},
     "loc_acid_M": {"value": null, "unit": "mol/L", "status": "unknown", "assumed_label": null, "range": null, "source": {"kind": "none", "doi": null, "locator": null, "publication_id": null, "safe_exp_ids": []}, "model_id": null, "fit_manifest_sha256": null, "note": ""},
     "third_phase_observed": {"value": null, "unit": "1", "status": "unknown", "assumed_label": null, "range": null, "source": {"kind": "none", "doi": null, "locator": null, "publication_id": null, "safe_exp_ids": []}, "model_id": null, "fit_manifest_sha256": null, "note": ""},
     "disengagement_s": {"value": null, "unit": "s", "status": "unknown", "assumed_label": null, "range": null, "source": {"kind": "none", "doi": null, "locator": null, "publication_id": null, "safe_exp_ids": []}, "model_id": null, "fit_manifest_sha256": null, "note": ""},
     "ligand_loss_mol_per_L_aq": {"value": null, "unit": "mol/L_aq", "status": "unknown", "assumed_label": null, "range": null, "source": {"kind": "none", "doi": null, "locator": null, "publication_id": null, "safe_exp_ids": []}, "model_id": null, "fit_manifest_sha256": null, "note": ""},
     "max_loading_fraction_studied": {"value": null, "unit": "1", "status": "unknown", "assumed_label": null, "range": null, "source": {"kind": "corpus", "doi": null, "locator": "computed by the fitter from loading-series records", "publication_id": null, "safe_exp_ids": []}, "model_id": null, "fit_manifest_sha256": null, "note": ""},
     "regenerability_note": ""},
  "stream_records": [],
  "oxidation_state_routes": [],
  "direction_prior": null,
  "applicability": {},
  "records": [
    {"record_id": "Ca_SAFE:2222", "metal": "Nd", "d": 23.5, "log_d": 1.3710678622717363,
     "acid_nominal_M": 3.0, "acid_eq_M": null, "anion_M": 3.0, "ligand_M": {"TODGA": 0.1}, "complexant_M": null,
     "metals_initial_mM": {"Nd": 4.9}, "oa_ratio": null, "temperature_C": 25.0, "contact_time_min": null,
     "diluent_name": "n_dodecane", "publication_id": "pub_5a68dc5665", "experiment_series_id": "caf22524baa8080e",
     "replicate_id": "c6fca1a8d31ed78d", "loading_series_id": "ls_sys_5cb78e5000d40860_Nd_pub_5a68dc5665_3_0.1",
     "is_tracer": true, "fit_eligible": true, "fit_ineligible_reason": null, "duplicate_flag": null,
     "provenance": {"status": "measured_corpus", "source": {"kind": "corpus", "doi": null, "locator": "dataset.parquet row", "publication_id": "pub_5a68dc5665", "safe_exp_ids": ["Ca_SAFE:2222"]}, "range": null, "assumed_label": null, "model_id": null, "fit_manifest_sha256": null, "note": "metal concentration semantics ASSUMED initial aqueous; O/A not reported"}},
    {"record_id": "Ca_SAFE:2221", "metal": "Nd", "d": 14.0, "log_d": 1.146128035678238,
     "acid_nominal_M": 3.0, "acid_eq_M": null, "anion_M": 3.0, "ligand_M": {"TODGA": 0.1}, "complexant_M": null,
     "metals_initial_mM": {"Nd": 6.0}, "oa_ratio": null, "temperature_C": 25.0, "contact_time_min": null,
     "diluent_name": "n_dodecane", "publication_id": "pub_5a68dc5665", "experiment_series_id": "caf22524baa8080e",
     "replicate_id": "7a45f0834894d399", "loading_series_id": "ls_sys_5cb78e5000d40860_Nd_pub_5a68dc5665_3_0.1",
     "is_tracer": false, "fit_eligible": true, "fit_ineligible_reason": null, "duplicate_flag": null,
     "provenance": {"status": "measured_corpus", "source": {"kind": "corpus", "doi": null, "locator": "dataset.parquet row", "publication_id": "pub_5a68dc5665", "safe_exp_ids": ["Ca_SAFE:2221"]}, "range": null, "assumed_label": null, "model_id": null, "fit_manifest_sha256": null, "note": ""}},
    {"record_id": "Ca_SAFE:2217", "metal": "Nd", "d": 0.6, "log_d": -0.2218487496163564,
     "acid_nominal_M": 3.0, "acid_eq_M": null, "anion_M": 3.0, "ligand_M": {"TODGA": 0.1}, "complexant_M": null,
     "metals_initial_mM": {"Nd": 12.0}, "oa_ratio": null, "temperature_C": 25.0, "contact_time_min": null,
     "diluent_name": "n_dodecane", "publication_id": "pub_5a68dc5665", "experiment_series_id": "caf22524baa8080e",
     "replicate_id": "650d759474d7c66d", "loading_series_id": "ls_sys_5cb78e5000d40860_Nd_pub_5a68dc5665_3_0.1",
     "is_tracer": false, "fit_eligible": true, "fit_ineligible_reason": null, "duplicate_flag": null,
     "provenance": {"status": "measured_corpus", "source": {"kind": "corpus", "doi": null, "locator": "dataset.parquet row", "publication_id": "pub_5a68dc5665", "safe_exp_ids": ["Ca_SAFE:2217"]}, "range": null, "assumed_label": null, "model_id": null, "fit_manifest_sha256": null, "note": ""}}
  ],
  "notes": "514 records, 28 publications, 14 metals (three records shown)."
}
```
Fitted block written by `g18_fit_dmodels.py` into `params["TODGA"]["20-30C"]` (values are filled by
the script; the shape is fixed):
```json
{"model_type": "solvating", "medium_anion": "nitrate", "temperature_band": "20-30C",
 "log_k": {"Nd": {"value": "<fit>", "unit": "1", "status": "fitted_from_corpus", "assumed_label": null, "range": ["<value - 2 SE>", "<value + 2 SE>"],
                 "source": {"kind": "model", "doi": null, "locator": "M1 pooled-slope fit, publication effects summed to zero", "publication_id": null, "safe_exp_ids": ["<all training safe_exp_ids of this metal>"]},
                 "model_id": "M1_pooled_sys_5cb78e5000d40860_20-30C", "fit_manifest_sha256": "<sha256 of results/dmodels/sys_5cb78e5000d40860.json>", "note": "intercept at [L] = 1 M, [NO3-] = 1 M; tracer limit"}},
 "n_solvation": {"value": "<fit>", "unit": "1", "status": "fitted_from_corpus", "range": ["<jackknife lo>", "<jackknife hi>"], "...": "reliability in results/dmodels"},
 "p_anion": {"value": "<fit p_eff>", "unit": "1", "status": "fitted_from_corpus", "note": "corpus acid exponent p_eff assigned to the anion axis; acid and nitrate are confounded in HNO3 media (flag ANION_ACID_CONFOUNDED when [anion] != [H+])"},
 "p_h": {"value": 0.0, "unit": "1", "status": "assumed", "assumed_label": "ASSUMED_PLACEHOLDER", "range": [0.0, 0.0], "source": {"kind": "none", "locator": "assignment of the confounded exponent to the anion axis"}, "note": ""},
 "k_acid_uptake": {"value": null, "unit": "L2/mol2", "status": "unknown", "note": "HNO3 uptake by TODGA not transcribed; ACID_UPTAKE_UNMODELLED above 1 M"},
 "delta_h_kj_mol": null}
```

## 4. `literature.py` and `ingest.py`

### 4.1 Columns read from the bundle (memory discipline)

`safe_exp_id, metal, canonical_smiles, extractant_name, D, log_D, geom_cond__acid_class,
geom_cond__diluent_family` plus all 64 `cond__*` columns (continuous: acid, extractant, metal
concentration, temperature, contact time; one-hot acids, diluents, additives). Provenance columns:
`safe_exp_id, publication_id, experiment_series_id, replicate_id, condition_id, cell_id`. About 85
columns of 2261; `pyarrow` column projection; peak < 300 MB.

### 4.2 Ingest algorithm `ingest_corpus(systems_dir, *, seed=18) -> IngestAudit`

1. `paths.assert_bundle_unchanged()`; read the columns of §4.1; read provenance; merge on
   `safe_exp_id`, `validate="one_to_one"`; fail if any `publication_id` is missing.
2. `gen13sep.cohort.apply_quarantine`; write both excluded sets to `systems/exclusions.csv`
   (`safe_exp_id, reason in {todga_name_mismatch, sentinel_logD_le_-6}`).
3. Decode one-hots: acid name, diluent name, additive names. Additives that are extractants
   (`hdehp -> "HDEHP"`, `tbp -> "TBP"`, `dhoa -> "DHOA"`, `dohya -> "DOHyA"`) become
   `organic_ligands` entries with `role = "synergist"` and SMILES from `literature.EXTRACTANT_SMILES`
   (chemical identity table, not a literature number); alcohols become modifiers.
4. System key per §3.1; one `SystemEntry` per key; `registry.json`.
5. Records per row (§3.2); `fit_eligible = False` with reason for NaN acid (11 rows), NaN or zero
   extractant (1 row), duplicate flag `UNIT_SLIP_DUPLICATE` (§4.6), or temperature outside the
   parameter band being fitted (decided in the fitter, not here).
6. Loading series (§4.5), duplicates (§4.6), applicability domains (§5.5) per (ligand, band).
7. Write `systems/<system_id>.json`, `corpus_records.csv` (flat mirror, one row per record),
   `series.csv`, `duplicates.csv`, `INDEX.md`; return `IngestAudit(bundle_sha256, rows_in,
   todga_name_mismatch_rows, sentinel_rows, rows_fit_ineligible, n_systems, n_records,
   n_publications, n_loading_series_publication_aware, n_loading_series_publication_blind,
   n_unit_slip_rows, n_tied_d_groups, replicate_groups, replicate_sd_median)`.

### 4.3 Acidity semantics

Corpus records carry `acid_nominal_M` only; `acid_eq_M` is null. Fits use the nominal value and the
resulting parameters carry `EQUILIBRIUM_ACID_ASSUMED_NOMINAL`; the cascade always uses the
equilibrium `[H+]` of §6. `anion_M` for HNO3 media equals the nominal acid (no salting agent rows).

### 4.4 Metal-concentration semantics and O/A

`cond__metal_concentration_mM` is treated as the initial aqueous concentration of the one metal of
the row (ASSUMED; note on every record); O/A is null and every loading fit declares `O/A = 1`
(`OA_ASSUMED` flag on the parameter). Sensitivity at O/A in {0.5, 1, 2} is part of the loading
evaluation (`PRE_REGISTRATION.md` §7). Open item U2: reading the source of pub_0e7f3e0563 before
sealing would settle both; if done, it is recorded as a dated pre-fit addendum.

### 4.5 Loading series (publication-aware, the primary definition)

`loading_series_id` is assigned when, within (system_id, metal, publication_id, acid_nominal_M,
ligand_M of the primary ligand), there are >= 3 distinct `metals_initial_mM` values among
fit-eligible rows after the duplicate rule; id string
`"ls_<system_id>_<metal>_<publication_id>_<acid>_<ligand>"`; `is_tracer` marks the smallest metal
concentration. The publication-blind count (same key without publication) is reported as a
sensitivity only. Verified list (10): TEHDGA/Nd pub_15b174237f 1 M/0.2 M (3 pts); TEHDGA/Nd
pub_5a68dc5665 3 M/0.1 M (6); D3DODGA/Nd pub_0e7f3e0563 1 M/0.2 M (18) and 3 M/0.2 M (15 -> 8 after
§4.6); TDDGA/Nd pub_5a68dc5665 3 M/0.1 M (6); TODGA/Nd pub_5a68dc5665 3 M/0.1 M (6); TODGA/Nd
pub_d3c970567f 3 M/0.1, 0.2, 0.3 M (4, 4, 8; flat); TODGA/Ce pub_917a4583d4 3 M/0.1 M (8; rising).

### 4.6 Duplicate rule

Within (publication_id, canonical_smiles, metal), rows whose `D` agree to 6 significant digits but
whose (acid, extractant, metal concentration, temperature) differ form a *tied group*.
* **Unit-slip tier**: if within the group exactly one continuous condition differs between two rows
  and its ratio equals (within 0.1 %) the metal's molar mass, 1000, or 1/1000, the pair is a unit
  slip. The copy kept is the one whose (log metal concentration, log D) point has the smaller
  absolute residual from a least-squares line through the *other, non-tied* points of its own
  series (>= 3 clean points required); the other copy gets `duplicate_flag = "UNIT_SLIP_DUPLICATE"`
  and `fit_eligible = False`. If neither series has 3 clean points both copies are flagged.
  Verified outcome: the seven sub-mM rows of the D3DODGA 3 M series (Ca_SAFE:2693–2699) are flagged;
  the 1 M rows are kept (test §12.2).
* **Tied-D tier**: all other tied groups are kept, `duplicate_flag = "TIED_D"`, counted in
  `duplicates.csv` (verified 130 groups / 324 rows), and the fit is repeated without them as a
  sensitivity (exploratory).

### 4.7 `literature.py`

`EXTRACTANT_SMILES: dict[str, str]` (HDEHP, TBP, DHOA, DOHyA, PC88A, Cyanex 272, D2EHPA, TODGA;
identity only). `placeholder(value, unit, range, doi, locator, note) -> Sourced` builds an
`assumed` object with `ASSUMED_PLACEHOLDER`. `derive_logk_from_extraction(fraction_extracted:
Mapping[str, float], metals_initial_mM, ha_total_M, oa, ph_eq_range, a_dimer, b_proton, s_ha=3.0)
-> Mapping[str, tuple[float, float]]` implements: `y_i = E_i x0_i oa^-1`... precisely `y_i = E_i
x0_i / oa` (mol/L organic), `[(HA)2]_f = ha_total/2 - s_ha sum_i y_i`, `log K_i = log(E_i/(1-E_i))
- log(oa) - a log[(HA)2]_f + b (-pH)` evaluated at both pH bounds (the interval). `ph50_to_logk(pH50,
ha_total_M, a, b) = -b pH50 - a log(ha_total/2)`, documented as valid only in the tracer limit and
setting `k_reported_as = "pH50"`. `pc88a_prnd_entry()`, `cyanex272_prnd_entry()` return the entries
of §3.7 (and are what `g18_build_db.py` writes); `todga_hydrophilic_complexant_placeholder()`
returns a complexant spec with `log_beta` placeholders whose `value` is None and ranges declared
(the DGA + aqueous ligand exploration of §13.5 sweeps them).

## 5. `dmodel.py` and `domain.py` — distribution models

### 5.1 Protocol

```python
@dataclass(frozen=True)
class StageState:                         # candidate equilibrium of one stage
    v_aq_L: float; v_org_L: float         # per-hour volumes of the two phases (flows)
    x_total: dict[str, float]             # aqueous total metal, mol/L (free + complexed)
    y: dict[str, float]                   # organic metal, mol/L (sum over ligands)
    y_by_ligand: dict[str, dict[str, float]]   # ligand -> metal -> mol/L
    h: float; anion: float; c_free: float      # aqueous [H+], [anion], free deprotonated complexant
    ligand_total: dict[str, float]; ligand_free: dict[str, float]; acid_in_org: dict[str, float]
    alkali_reserve: float; temperature_C: float

@dataclass(frozen=True)
class DEval:
    log_d: float; d: float                # effective (complexant-corrected) values
    dlogd_dlnL: float; dlogd_dlnh: float; dlogd_dlnanion: float; dlogd_dlnc: float   # analytic partials
    flags: frozenset[Flag]; ood_distance: dict[str, float]

class DModel(Protocol):
    ligand: str; mechanism: Mechanism; metals: tuple[str, ...]
    q: Mapping[str, float]; p: Mapping[str, float]; z: Mapping[str, float]   # per metal: ligands, protons, anions per metal
    domain: ApplicabilityDomain | None; provenance: Provenance
    def evaluate(self, metal: str, state: StageState, activity: "ActivityModel | None" = None) -> DEval: ...
```
Models: `ConstantD(log_d_by_metal, *, q=0, p=0, z=0)` (labelled sanity limit; provenance status
`assumed`), `CationExchangeMassAction(params, ligand, medium, domain)`,
`SolvatingMassAction(params, ligand, medium, domain)`, `NearestConditionD(records, ligand,
domain, n_prior)` (the B1 fallback D source: tracer log D of the same-metal nearest record in
(log acid, log ligand), loading correction through the depletion term with `n_prior`, `ood_distance
= the 1-NN distance`), `AqueousComplexantWrapper(inner, complexant)`. `EffectiveCapacity(phi)` is an
`ActivityModel` (§5.6), exploratory only.

### 5.2 Equations

(a) Cation exchange, ligand k dimeric, `L = [(HA)2]_f`:
`log D0_i = log K_i + a log L - b log h`; `q_i = 3` (dimers), `p_i = 3`, `z_i = 0`.
Partials: `dlogd/dlnL = a/ln10`... stated in log10 per ln: `d log10 D / d ln L = a / ln(10)`,
`d log10 D / d ln h = -b / ln(10)`.

(b) Solvating, ligand k: `log D0_i = log K_i + n log L + p_anion log anion + p_h log h`;
`q_i = n`, `p_i = 0`, `z_i = anions_per_metal` (3 in nitrate). HNO3 uptake: species HNO3·L with
`a_k = K_H h anion L` (mol/L organic); `K_H` None -> `a_k = 0` and `ACID_UPTAKE_UNMODELLED` when
`h > 1`. Corpus-fitted parameters: `p_anion = p_eff`, `p_h = 0`; when `anion != h` (salting agent
or strip liquor with added nitrate) and the parameter set was fitted on HNO3-only records, add
`ANION_ACID_CONFOUNDED`.

(c) Aqueous complexant (wrapper): `alpha_i(c) = 1 + sum_m beta_i,m c^m` with `beta_i,m =
10^log_beta_i,m`; `D_i = D0_i / alpha_i`; `alpha_H(h) = 1 + sum_j K_H,j h^j`; bound-ligand
function `phi_i(c) = sum_m m beta_i,m c^m / alpha_i(c)`; the complexant balance is §6.2 (5).
`d log10 D / d ln c = -(sum_m m beta_i,m c^m / alpha_i) / ln(10)`.

(d) Anion complexation (cation exchange in chloride/nitrate) when `beta_anion` is sourced:
`alpha_i` gains `sum_j beta^X_i,j anion^j`; otherwise, if the stage anion concentration is outside
the domain's anion interval by more than a factor 2, `MEDIUM_STRENGTH_UNMODELLED`.

### 5.3 Composition over several organic ligands

`D_i = sum_k D_i^(k)`; `y_i^(k) = D_i^(k) x_i`; ligand k is depleted only by `y^(k)`. Two
extractant-role ligands add `MIXED_ORGANIC_UNMODELLED` to every evaluation. Modifiers (alcohols) do
not appear in the equations; they are part of the system key and therefore of which parameter set
is used.

### 5.4 Loading fraction, caps and chemistry flags

`lambda_k = (sum_i q_i^(k) y_i^(k) + a_k) / L_T^(k)`. The fixed-stoichiometry model is defined for
`lambda_k <= 1`; the bracket `L_f in [0, L_T]` of §6 enforces it (for cation exchange this means
`[M]_org <= [(HA)2]_T / 3 = [HA]_T / 6`; the chemical saturation ceiling `[HA]_T / 3` with `MA3·xHA`,
x -> 0, lies outside the model and is stated in the report). `HIGH_LOADING` when `lambda_k > 0.5`;
`LOADING_CAP_HIT` when `L_f^(k) < 1e-6 L_T^(k)`; `PHASE_BEHAVIOUR_UNKNOWN` when `loc_metal_M.value
is None` and `lambda_k > 0.3`; `THIRD_PHASE_RISK` when `loc_metal_M` is sourced and `sum_i y_i +
sum_k a_k > loc_metal_M`, or `loc_acid_M` sourced and `sum_k a_k > loc_acid_M`.

### 5.5 `domain.py`

`build_domain(records: pd.DataFrame, ligand: str) -> ApplicabilityDomain` from fit-eligible records
of one (ligand, band): box on log acid, log ligand(s), log metal total mM (non-NaN rows), loading
fraction (computed as `n_prior * metal_mM * 1e-3 / ligand_M` for rows with metal concentration;
ASSUMED O/A 1), temperature, categorical anion/diluent family/modifiers; convex hull of the fit
points in (log acid, log primary ligand) via `scipy.spatial.ConvexHull` (`QJ` option; < 3 distinct
points or collinear -> segment/point, `hull_vertices` holds the endpoints and `OOD_HULL` uses the
distance to the segment). `domain_flags(domain, state, ligand) -> tuple[frozenset[Flag],
dict[str, float]]`: per axis, distance outside the interval in log10 units (0 inside); categorical
mismatch distance 1.0; `OOD_HULL` with Euclidean distance to the hull in log units. Literature
domains have `hull_vertices = None` (box only).

### 5.6 `ActivityModel`

```python
class ActivityModel(Protocol):
    def aqueous_gamma(self, state: StageState) -> dict[str, float]: ...   # unity by default
    def organic_free_ligand(self, state: StageState, ligand: str) -> float: ...   # returns state.ligand_free[ligand] by default
```
`EffectiveCapacity(phi: float)` returns `phi * L_T - sum q y - a` in place of `L_T - sum q y - a`
(equivalently rescales the ligand total). It is used only in the exploratory arm of the loading
evaluation, with `log K` re-anchored at the tracer point for every `phi` (§10.4).

### 5.7 `build_system_model`

`build_system_model(entry: SystemEntry, *, feed_anion: str, temperature_C: float, params_source:
str = "auto", complexant: ComplexantSpec | None = None, activity=None, parameter_draw:
Mapping[str, float] | None = None) -> SystemModel | ModelBuildError`. Refuses (returns
`ModelBuildError(reason)`) when `feed_anion != entry.medium.anion` (cross-anion transfer), when a
required `log_k` is missing, or when the complexant lacks `log_beta` for a feed metal.
`params_source`: `"literature"` (entry params), `"fitted"` (fitted block), `"nearest"`
(`NearestConditionD` from records), `"auto"` (fitted if present and the pre-registered decision
adopted it, else nearest for corpus systems, literature for literature systems).
`parameter_draw` maps `"<ligand>.<param>[.<metal>]"` to a value inside the declared range (used by
the case study's draws). `SystemModel(entry, dmodels: Mapping[str, DModel], complexant, activity,
temperature_C, assumptions: tuple[str, ...])`.

## 6. `equilibrium.py` — single-stage coupled equilibrium

### 6.1 Streams

```python
@dataclass(frozen=True)
class AqStream:
    flow_L_h: float; metals: dict[str, float]     # total (free + complexed) mol/L
    h: float; anion: float; complexant_total: float; sodium: float
    labels: dict[str, dict[str, float]] | None = None   # origin sub-species: metal -> label -> mol/L (§7.8)

@dataclass(frozen=True)
class OrgStream:
    flow_L_h: float; metals: dict[str, float]          # sum over ligands, mol/L
    metals_by_ligand: dict[str, dict[str, float]]
    ligand_total: dict[str, float]; ligand_free: dict[str, float]; acid_in_org: dict[str, float]
    alkali_reserve: float
    labels: dict[str, dict[str, float]] | None = None
```
No field is shared between the two classes for two species.

### 6.2 The stage problem

Inputs: `aq_in`, `org_in`, `system: SystemModel`; `r = org_in.flow / aq_in.flow`. Unknowns:
`L_f^(k)` for each organic ligand, `h`, `anion` (written `nu`), `c` (only if a complexant is
present). Given the unknowns every metal is explicit:

```
T_i   = aq_in.metals[i] + r * org_in.metals[i]                 (mol per L aqueous)
D_i   = sum_k D_i^(k)(L_f, h, nu, c)                             (effective, §5.2–5.3)
x_i   = T_i / (1 + r D_i)                                        (x_i = 0 when D_i = inf)
y_i   = (T_i - x_i) / r                                          (never D_i * x_i: finite at D -> inf; y_i = 0 at D_i = 0)
y_i^(k) = D_i^(k) x_i   (for D_i finite; at D_i = inf split y_i in proportion to the finite ratios of D^(k) at the bracket edge)
```
Residuals (all in mol/L, scaled as stated):
```
(1) ligand k:  g_k = L_f^(k) (1 + KH_k h nu) + sum_i q_i^(k) y_i^(k) - L_T^(k)             scale L_T^(k)
(2) acid:      released = sum_i p_i (y_i - org_in.metals[i])                                (per L org; negative in scrub/strip)
               uptake   = sum_k (KH_k h nu L_f^(k) - org_in.acid_in_org[k])
               P  = aq_in.h + r (released - uptake)          ("proton pool", per L aq)
               Bp = r * org_in.alkali_reserve                  ("base pool")
               branch 1 (P >= Bp):  g_h = h - (P - Bp);       B_out = 0;            Na_out = aq_in.sodium + Bp
               branch 2 (P <  Bp):  h = H_MIN (fixed);        B_out = (Bp - P)/r;   Na_out = aq_in.sodium + P;  flag ALKALI_EXCESS
(3) anion:     g_nu = nu - aq_in.anion + r sum_i z_i (y_i - org_in.metals[i]) + r * uptake    scale max(aq_in.anion, 1e-3)
(4) complexant: g_c = c alpha_H(h) + sum_i x_i phi_i(c) - aq_in.complexant_total             scale aq_in.complexant_total
```
`H_MIN = 1e-5` mol/L (a model constant, documented; the branch is inadmissible anyway). Outputs:
`aq_out = AqStream(aq_in.flow, x, h, nu, aq_in.complexant_total, Na_out)`; `org_out = OrgStream(
org_in.flow, y, y_by_ligand, L_T, L_f, {k: KH_k h nu L_f^(k)}, B_out)`; `StageDiagnostics(d:
dict, d_by_ligand, loading_fraction: dict[str, float], iterations, residual_max, branch, flags:
frozenset[Flag], ood_distance: dict[str, float], status in {"converged","failed"})`.
Per-metal closure `|x_i + r y_i - T_i| <= 1e-12 T_i` holds by construction.

### 6.3 Monotonicity used by the solver

For fixed other unknowns: `g_k` is strictly increasing in `L_f^(k)` (D increasing in L, y increasing
in D, and the `(1 + KH h nu)` factor positive), with `g_k(0) < 0 <= g_k(L_T)`; `g_h` (branch 1) is
strictly increasing in `h` (y decreasing in h for cation exchange; solvating `p_h = 0`; uptake
increasing in h enters with a plus sign), bracket `[H_MIN, aq_in.h + r sum_i p_i T_i + r*max(0,-uptake_min)]`;
`g_nu` is increasing in `nu` (y and uptake increase with nu), bracket `[1e-9, aq_in.anion + r
sum_i z_i org_in.metals[i] + r sum_k org_in.acid_in_org[k]]`; `g_c` is increasing in `c`, bracket
`[0, aq_in.complexant_total]`. For the coupled `L–h` problem with equal stoichiometry across metals
(one mechanism) the outer root in `L` is unique (design B's argument: if `L2 > L1` and the composite
loading at the inner acid equilibrium did not increase, every `D_i` would be larger at `L2`,
contradiction). Uniqueness with the complexant and anion couplings is not proven; it is tested by
multi-start (§12.2) and a disagreement is a reported defect.

### 6.4 Algorithm `solve_stage(aq_in, org_in, system, *, tol=1e-12, max_iter=200, warm=None) -> (AqStream, OrgStream, StageDiagnostics)`

Primary path (fast): damped Newton in log-variables `w = (ln L_f^(k)..., ln h, ln nu, ln c)` with
the analytic Jacobian of the residuals (1)–(4) (chain rule through `x_i = T_i/(1 + r D_i)`,
`dD/dlnL`, `dD/dlnh`, `dD/dlnnu`, `dD/dlnc` from `DEval`), residuals scaled as stated, every
unknown projected into its bracket after each step (fraction-to-boundary 0.95 then clamp),
Armijo backtracking on `||F||^2` (10 halvings), convergence when `max |F_scaled| < tol` and the
acid branch is unchanged between the last two iterations. Branch handling: start on branch 1; if
the converged `h < H_MIN` switch to branch 2 (fix `h = H_MIN`, remove it from the unknowns, re-solve
the rest). Fallback path (guaranteed by §6.3, used when the primary does not converge in
`max_iter`): outer bisection/Illinois on `L_f^(1)` in `[0, L_T^(1)]`, inner Illinois on `L_f^(2)`
(if any), inner Illinois on `h`, inner Illinois on `nu`, inner Illinois on `c`, each to `1e-13`
relative, each capped at 100 iterations; the same branch rule. `warm` (a previous
`StageDiagnostics`) seeds the primary path. Both paths must agree to 1e-10 relative on every output
(test). `status = "failed"` only if the fallback also fails (should not happen; counted).
No metal concentration is ever clipped; positivity holds by construction.

## 7. `cascade.py`

### 7.1 Spec

```python
@dataclass(frozen=True)
class CascadeSpec:
    n_ext: int; n_scr: int; n_str: int                # N_scr = 0 and N_str = 0 allowed
    feed: AqStream                                    # flow A (L/h), metals mol/L, h, anion, complexant, sodium
    scrub: AqStream                                   # flow S; may carry complexant and/or the target metal (displacement scrub)
    strip: AqStream                                   # flow W
    organic_flow_L_h: float                           # O
    ligand_total: dict[str, float]                    # L_T per ligand, mol/L organic (dimer basis for dimers)
    saponification_degree: float                      # s in [0, 1]; B_org at the organic inlet of stage 0 = s * [HA]_T (monomer basis); 0 for solvating
    feed_stage: int | None = None                     # j_F in [0, n_ext - 1]; None -> n_ext - 1
    scrub_return_stage: int | None = None             # j_R in [0, n_ext - 1]; None -> n_ext - 1 (the scrub raffinate joins the feed stage)
    f_bleed: float = 0.0                              # fraction of the organic replaced by fresh organic per pass
    fresh_organic: OrgStream | None = None            # residual metal of fresh organic (zeros when None)
    target: str = ""                                  # metal for the origin-labelled pass (§7.8)
```
Validation (`ValueError`): negative or zero flows, `n_ext < 1`, stage indices out of range,
`saponification_degree` outside [0, 1], metals not in the system.

### 7.2 Topology and indexing

`N = n_ext + n_scr + n_str`; stages `j = 0..N-1`; the organic flows toward increasing `j`, the
aqueous toward decreasing `j`. Sections: extraction `[0, n_ext)`, scrub `[n_ext, n_ext + n_scr)`,
strip `[n_ext + n_scr, N)`.

Aqueous circuit A (extraction + scrub): the scrub liquor enters stage `n_ext + n_scr - 1`; aqueous
links `(j+1 -> j)` exist for all `j` in `[0, n_ext + n_scr - 1)` except that the link
`(n_ext -> n_ext - 1)` exists only when `scrub_return_stage == n_ext - 1`; otherwise the aqueous
outlet of stage `n_ext` (the scrub raffinate) is an internal stream injected at stage
`scrub_return_stage`. The feed is injected at `feed_stage`. The aqueous outlet of stage 0 is the
raffinate. When `n_scr == 0` the scrub liquor stream is ignored (flow 0). Aqueous flow of stage j
in circuit A: `A_j = S + A * [j <= feed_stage] ` when the scrub raffinate flows through j, i.e.
`A_j = sum of the external aqueous flows that reach j` computed from the incidence matrix
(§7.4); volumes are constant, so `A_j` is fixed by the topology.

Aqueous circuit B (strip): the strip liquor enters stage `N-1` and flows down to stage
`n_ext + n_scr`, whose aqueous outlet is the product liquor; `A_j = W` there. When `n_str == 0`
the product is the loaded organic leaving stage `N-1` (reported as `product_is_loaded_organic`) and
there is no recycle (`f_bleed` forced to 1).

Organic: the organic inlet of stage 0 is `(1 - f_bleed) * org_out[N-1] + f_bleed * fresh_organic`
(same `L_T`), with the alkali reserve re-created: `B_in,0 = saponification_degree * [HA]_T` (the
stripped organic is re-saponified each pass; this is the base consumption of §8). Per-stage
`r_j = O / A_j`.

### 7.3 Unknowns and residuals

Per stage: `x_j,i` (M), `L_j^(k)` (K), `h_j`, `nu_j`, `c_j`, `B_j` -> `M + K + 4` unknowns;
`u` has length `N (M + K + 4)`. Derived per stage from the unknowns: `D_j,i`, `y_j,i`, `a_j^(k)`.
Residuals `F_j`:
```
metal i:     A_j x_j,i + O y_j,i - sum_m Inc[j,m] A_m x_m,i - O y_(j-1),i - ext_j,i = 0
ligand k:    L_j^(k)(1 + KH_k h_j nu_j) + sum_i q_i^(k) y_j,i^(k) - L_T^(k) = 0
acid:        branch 1: h_j - (P_j - r_j B_(j-1)) = 0 ;  branch 2: h_j - H_MIN = 0
             with P_j = h_in,j + r_j (released_j - uptake_j), h_in,j A_j = sum_m Inc[j,m] A_m h_m + ext_h,j
reserve:     branch 1: B_j = 0 ;  branch 2: B_j - B_(j-1) + P_j / r_j = 0
anion:       nu_j - nu_in,j + r_j sum_i z_i (y_j,i - y_(j-1),i) + r_j uptake_j = 0
complexant:  c_j alpha_H(h_j) + sum_i x_j,i phi_i(c_j) - cT_in,j = 0        (c_j - 0 = 0 when no complexant)
```
`y_(-1) = (1 - f_bleed) y_(N-1) + f_bleed y_fresh`, likewise `a_(-1)`, and `B_(-1) =
saponification_degree * [HA]_T`. Scaling: metal rows by `(A x_F,i + S x_S,i + 1e-30 O)`, ligand rows
by `O L_T^(k)`, acid rows by `max(A h_F, S h_S, W h_W, 1e-3)`, reserve rows by `max(O B_in,0, 1e-6)`,
anion rows like acid, complexant rows by `S cT_S + 1e-12`.

### 7.4 Incidence

`Inc` is the `N x N` 0/1 matrix of aqueous links; `ext_j` collects external injections (feed at
`feed_stage`, scrub liquor at `n_ext + n_scr - 1`, strip liquor at `N-1`, the internal scrub
raffinate at `scrub_return_stage` when it is not the natural link). `A_j` is computed once from
`Inc` and the external flows (a topological pass from the top of each circuit).

### 7.5 Primary solver: damped Newton with analytic Jacobian

```
solve_cascade(spec, system, *, method="auto", tol=1e-11, max_newton=60, max_sweeps=5000) -> CascadeResult
  u0 = kremser_init(spec, system)                         # §7.7
  for it in range(max_newton):
     branch = active_branch(u)                            # per stage from P_j vs r_j B_(j-1)
     F, J = residual_and_jacobian(u, branch)              # J block-tridiagonal in stage blocks + one recycle block (stage 0 <- stage N-1) + one internal-return block (scrub_return_stage <- n_ext)
     if max|F| < tol and branch == branch_prev: status converged_newton; break
     du = solve(J, -F)                                    # numpy.linalg.solve when len(u) <= 600, scipy.sparse.linalg.spsolve (csc) otherwise; LinAlgError -> lstsq + flag JACOBIAN_SINGULAR
     alpha = min(1, 0.95 * min over k with du_k < 0 of (-(u_k - lb_k) / du_k))     # lower bounds: 0 for x, L, nu, c, B; H_MIN for h
     for bt in range(10): u_try = u + alpha du; if ||F(u_try)||^2 <= (1 - 1e-4 alpha) ||F||^2: accept, break; alpha /= 2
     else: status newton_stalled -> fallback
     if branch oscillates for 5 consecutive iterations: status branch_cycle -> fallback
  else: status newton_maxiter -> fallback
```
Unknowns are scaled by `max(u0, floor)` before assembling `J`. The Jacobian entries are the
analytic partial derivatives of the residuals of §7.3 with respect to every unknown of stages
`j-1`, `j`, `j+1` (and the recycle/return couplings), using `DEval` partials; a finite-difference
check test (§12.2) is required. Fallback = §7.6, then, if that also fails, `status = "failed"`.

### 7.6 Fallback and cross-check: successive substitution

```
solve_cascade_ss(spec, system, init=u0)
  omega = 1.0; tear = (y_(N-1), a_(N-1), B_(N-1)) from init
  for sweep in range(max_sweeps):
     x_old = x.copy()
     for j in 0..N-1 (organic direction):
        aq_in_j from Inc, ext and the current x, h, nu, cT of upstream stages (previous sweep values for j+1)
        org_in_j = org_out_(j-1) (this sweep) or recycle(tear) for j = 0
        aq_out_j, org_out_j, diag_j = solve_stage(aq_in_j, org_in_j, system, warm=diag_j_prev)
     x = omega x + (1 - omega) x_old
     err = max |x - x_old| / (|x| + 1e-30)
     if err increased on 3 consecutive sweeps: omega = max(0.05, omega / 2)
     tear update: after sweep 5, Wegstein on each tear component with factor q = clip(s/(s-1), -5, 0) where s is the secant slope; else direct substitution
     if err < 1e-11 and check_balances(...).max < 1e-10: status converged_ss; break
  else: status ss_maxiter
```

### 7.7 Kremser initial solve and oracle

`kremser_init(spec, system)`: evaluate every metal's `D` at the *tracer conditions* of each
section (`L = L_T`, `h` = the section's aqueous inlet acid, `nu` = inlet anion, `c = cT`, zero
loading); with constant `D` the metal balances of §7.3 are linear in `x`; assemble and solve them
directly (dense or sparse as in §7.5, including recycle and return couplings). Then set `L_j`,
`h_j`, `nu_j`, `c_j`, `B_j` from the balances at those `x`, floored at 1 % of their totals. Exact
for `ConstantD` (test). `kremser_fraction_unextracted(E, n) = (E - 1) / (E**(n + 1) - 1)` for
`E != 1`, `1 / (n + 1)` for `E == 1`, with `E = D * (O/A)`: the analytic oracle for a single
extraction section with lean organic and no recycle (`n_scr = n_str = 0`, `f_bleed = 1`).

### 7.8 Origin-labelled pass (exact at converged D)

For `target = T`, labels `feed`, `scrub`, `fresh` (fresh organic residual). With `D_j,T` fixed at
the converged values, for each label `l` solve the linear system
`A_j x_j^l + O D_j,T x_j^l - sum_m Inc[j,m] A_m x_m^l - O D_(j-1),T x_(j-1)^l - ext_j^l = 0`
(with the recycle `y_(-1)^l = (1 - f_bleed) D_(N-1),T x_(N-1)^l + f_bleed y_fresh^l`), where
`ext^feed` is the feed's T at `feed_stage`, `ext^scrub` the scrub liquor's T at the scrub top,
`ext^fresh` the fresh organic's T at stage 0. Identity: `sum_l x_j^l = x_j,T` to 1e-10 relative
(test). Reported: `recovery_from_feed = product T^feed / (A x_F,T)`, `recovery_total = product T /
(A x_F,T)`, `scrub_target_return = product T^scrub / (S x_S,T)` (None when `S x_S,T = 0`),
`net_product_mol_h = product T - S x_S,T`. Identity (test): `recovery_total - recovery_from_feed =
scrub_target_return * (S x_S,T)/(A x_F,T) + product T^fresh/(A x_F,T)`.

### 7.9 Result and balances

```python
@dataclass(frozen=True)
class CascadeResult:
    status: str                    # converged_newton | converged_ss | failed | invalid_spec
    stages_aq: tuple[AqStream, ...]; stages_org: tuple[OrgStream, ...]; diagnostics: tuple[StageDiagnostics, ...]
    raffinate: AqStream; product: AqStream | OrgStream; scrub_raffinate: AqStream | None; loaded_organic: OrgStream; stripped_organic: OrgStream
    origin: Mapping[str, float] | None   # recovery_from_feed, recovery_total, scrub_target_return, net_product_mol_h, label sums
    residual_max: float; iterations: int; balance_rel_max: float; balances: Mapping[str, float]
    flags: frozenset[Flag]; ood_distance: Mapping[str, float]   # union over stages, max distance per axis
    regime_status: str             # IN_DOMAIN | IN_DOMAIN_WITH_CAVEATS | OUT_OF_DOMAIN | INADMISSIBLE
    assumptions: tuple[str, ...]
```
`check_balances(spec, result) -> dict[str, float]` recomputes, **from the stream table alone**,
relative in-minus-out over the whole cascade for: each metal; each ligand (`L_f + sum_i q_i y^(k) +
a_k = L_T` at every stage, max deviation); the proton ledger `Q = A h + O (sum_k a_k - B - sum_i p_i
y_i)` (in = out); the anion ledger `A nu + O (sum_i z_i y_i + sum_k a_k)`; the complexant total
(aqueous only); sodium `A Na + O B`. `balance_rel_max` is the maximum; every results table prints
it. A failed result carries NaN metrics and is never raised.

## 8. `metrics.py`

`compute_metrics(result, spec, system, target, impurities, prices: Prices | None) -> ProcessMetrics`
with (all None/NaN when `result.status == "failed"`):
* `purity_mol = n_T,product / sum_i n_i,product` (mol basis; product = strip liquor, or the loaded
  organic when `n_str = 0`); `purity_mass` with atomic masses; `purity_oxide` with `M2O3` masses
  (`gen13sep.metals` atomic masses; oxide factor `(2 M + 48.0)/(2 M)` per metal).
* `recovery_from_feed` (headline), `recovery_total`, `scrub_target_return`, `net_product_mol_h` (§7.8).
* `enrichment_factor = (n_T/n_I)_product / (n_T/n_I)_feed` per impurity (named so; not SF).
* `sf_tracer[section][I] = D_T/D_I` at each section's tracer conditions; `sf_by_stage[j][I]`.
* `n_stages_total`, `n_ext`, `n_scr`, `n_str`, `oa_ext = O/A`, `s_over_a = S/A`, `w_over_a = W/A`.
* `throughput_mol_T_per_h_per_L_org = n_T,product / O`, `throughput_kg_oxide_per_h`.
* `max_loading_fraction` per ligand over stages; `phase` items copied from the entry (`loc_metal_M`,
  `disengagement_s`, `ligand_loss`, `third_phase_observed`, `regenerability_note`) as `Sourced`,
  so a regime table always shows them (null when null).
* Consumption per mol of T in the product and per kg of T oxide (`on_spec` flag alongside):
  `acid_mol_h = S h_S + W h_W + max(0, feed acidification)`; `base_mol_h = O *
  saponification_degree * [HA]_T`; `salting_anion_mol_h = S salt_S + W salt_W`; `complexant_mol_h
  = S cT_S * (1 - regeneration_fraction.value)` (NaN + `COST_INCOMPLETE` when the fraction is None);
  `extractant_makeup_mol_h = f_bleed O L_T + ligand_loss * (A + S + W)` (NaN + `LIGAND_LOSS_NOT_MEASURED`
  when the loss is None and `f_bleed = 0`); `diluent_L_h` (NaN unless sourced); `water_L_h = S + W +
  dilution`.
* `cost_proxy_per_kg_oxide = sum_i price_i * consumption_i` from `config/prices.json` (each
  entry `{value, unit, currency, status: "assumed", assumed_label, range, note}`); NaN with
  `COST_INCOMPLETE` when any price or consumption is NaN. Consumption is the primary economic
  number; the cost proxy is secondary and is reported with the price table's ranges.
* `on_spec = purity_mol >= spec.purity_min and recovery_from_feed >= spec.recovery_min`.
* `regime_status` (§1.4) and the flag set with per-axis OOD distances.

## 9. `optimize.py` and the recipes driver

### 9.1 Design space

`DesignSpace(bounds: dict[str, tuple[float, float]], integer: tuple[str, ...], fixed: dict[str,
float])` over: `n_ext, n_scr, n_str, feed_stage_offset, scrub_return_offset` (integers; offsets
counted down from `n_ext - 1`), `oa_ext, s_over_a, w_over_a, scrub_acid_M, scrub_target_mM,
scrub_complexant_M, strip_acid_M, strip_anion_M, ligand_total_M, saponification_degree,
feed_dilution, f_bleed`. Default bounds come from the parameter set's `ApplicabilityDomain`
(concentration axes) and from `config/design_spaces.json` (stage counts 1–30 each, ratios
0.2–5); the user may widen them; every candidate outside the recorded intervals carries its OOD
flags. Integer variables are `floor(u * (hi - lo + 1)) + lo` clipped to `[lo, hi]`.

### 9.2 `lhs_pareto(space, base_spec, system, target, impurities, spec_limits, prices, *, n=2000, seed=18) -> pd.DataFrame`

`qmc.LatinHypercube(d, scramble=True, seed=seed).random(n)`; one cascade per row (`solve_cascade`
then `compute_metrics`); rows keep `status`, all metrics, all variables, flags, `regime_status`;
`invalid_spec` and `failed` rows are kept and counted (`n_failed`). Non-dominated rank on
(`purity_mol` up, `recovery_from_feed` up, `consumption_index` down, `n_stages_total` down) where
`consumption_index = acid_mol_per_kg_oxide + base_mol_per_kg_oxide + complexant_mol_per_kg_oxide`
(NaN components excluded and flagged) — or `cost_proxy` when `--objective cost`. Feasibility
(`on_spec`) is a column, not folded into the objectives. Output sorted by (front rank, then the
variable tuple) so the CSV is byte-identical across runs (test). The front is reported twice:
`in_domain_only` (`regime_status in {IN_DOMAIN, IN_DOMAIN_WITH_CAVEATS}`) and `all`, plus the
epsilon-constraint knees (minimum consumption at each purity/recovery threshold of the spec grid).
Evaluation budget: `n` and the measured wall time from `g18_bench.py` are written to the results
manifest.

### 9.3 Optional GP-BO

`gp_bo(space, base_spec, system, ..., n_init=64, n_iter=30, batch=4, seed=18)`: ParEGO
(random Chebyshev scalarisation per iteration, `sklearn.gaussian_process.GaussianProcessRegressor`
with `Matern(nu=2.5)` ARD + `WhiteKernel`, expected improvement over 2000 LHS candidates, kriging
believer for the batch). Gated: runs only when the D source's reliability flags pass
(`PRE_REGISTRATION.md` §6) and only inside the domain box; positioned for the measurement loop
(each point costs an experiment); never a headline number.

### 9.4 Recipes driver

`scripts/g18_recipes.py --feed cases/<feed>.json --spec cases/<spec>.json [--systems all|<ids>]
[--n-lhs 1000] [--seed 18]` implements the document's I/O: for every database system whose
`medium.anion` equals the feed anion and which has a usable parameter set (`build_system_model` not
an error), run `lhs_pareto`, and write `results/recipes/<feed>/<system_id>.csv` plus
`SUMMARY.md`: per system family the top three on-spec regimes (by consumption index), the Pareto
knee, flags, and the parameter status of that system (literature placeholders, fitted, nearest);
systems without parameters are listed as `not_parameterised`. Every table carries the regime tag
(§10.6).

## 10. `evalproto.py`, `screen.py`, `report.py`

### 10.1 Fit M1 (implements `PRE_REGISTRATION.md` §4)

`fit_mass_action(records: pd.DataFrame, *, mechanism, ligand, band, pooled_slopes=True,
n_prior=3.0, p_prior=2.0, seed=18) -> FitResult`. Steps: (1) fit-eligible records of the
(system, ligand, band); aggregate replicate groups (exact 64-column key + publication) to mean
log D with weight `n_rep`; (2) `lE = log10 ligand_M`, `lA = log10 acid_nominal_M`; (3) design matrix:
one intercept per metal, shared slope on `lE`, shared slope on `lA`, sum-to-zero publication
effects for publications with >= 2 aggregated points (single-point publications get effect 0);
(4) an axis with < 3 distinct levels in the training set has its slope fixed at the prior (`n_prior`
for `lE`, `p_prior` for `lA`, status `assumed`, flagged); (5) weighted least squares (unpenalised)
by `numpy.linalg.lstsq` on the weighted system; covariance `(X'WX)^-1 s^2` and the HC1 sandwich;
(6) reliability (§10.3). `FitResult(system_id, ligand, band, intercepts, n, p_eff, publication_effects,
covariance, se, jackknife_se, split_half, interpretable, n_points, n_publications, domain,
fit_manifest_sha256)`; `as_solvating_params(fit) -> SolvatingParams` (or cation-exchange analogue).

### 10.2 Baselines and LOPO

`lopo_evaluate(records, *, mechanism, ligand, band, seed=18) -> pd.DataFrame` with one row per
(system, held-out publication, metal): `mae_M1` (held-out publication effect 0), `mae_B0` (training
mean of that metal), `mae_B1` (same-metal 1-NN in (lA, lE), Euclidean, ties by smaller |delta lA|
then smaller `record_id`), `mae_M1_offset` (one held-out point, chosen as the smallest `record_id`,
used to set the publication effect; scored on the rest), `mae_B1_crossmetal` (exploratory),
`n_points`. `macro_over_systems(table)`: mean over (publication, metal) rows within system, then
mean over systems. `paired_system_bootstrap(a, b, n_boot=2000, seed=18) -> (mean, lo, hi)`.

### 10.3 Reliability

`jackknife_by_publication(records) -> {n: (estimate, se), p_eff: (estimate, se)}`;
`split_half_by_publication(records, repeats=20, seed=18)` for systems with >= 4 publications
(random halves; sign agreement of `n` and `p_eff`, Pearson r of the metal-intercept vectors);
systems with 2–3 publications report jackknife only and `split_half = None`. A slope is
`interpretable` iff jackknife SE < 0.5 and (when defined) split-half sign agreement >= 18/20.

### 10.4 Loading evaluation

`loading_series_evaluate(records, fits, *, oa=1.0, n_fallback=3.0) -> pd.DataFrame`: per series,
anchor `log K` so that `solve_stage` at the tracer point (one metal, initial aqueous concentration
`metals_initial_mM * 1e-3`, `L_T = ligand_M`, `h = nu = acid_nominal_M`, O/A = `oa`, `K_H` None)
reproduces the tracer `log D` to 1e-10 (1-D root in `log K`); predict every non-tracer point by
`solve_stage`; `mae_ideal`, `mae_constant` (tracer value everywhere), `spearman_logD_vs_logmM`
(descriptive). Exploratory `EffectiveCapacity(phi)`: for series with >= 4 non-tracer points,
leave-one-point-out: `phi` fitted on the rest by `scipy.optimize.minimize_scalar(bounded=(0.01, 1))`
with `log K` re-anchored at the tracer for every `phi`, scored on the left-out point ->
`mae_phi_loo`, `phi_median`.

### 10.5 `ComparisonCounter`

`record(name, family in {"primary","secondary","exploratory"}, p_value | None)`; `table()` with
Benjamini–Hochberg over the exploratory family; written to `results/eval/comparisons.csv`.

### 10.6 `report.py`

`write_table(df, path, *, regime: dict)` refuses (`ValueError`) unless `regime` has `cohort`,
`holdout`, `averaging_unit` (and `status_of_parameters` for regime tables); the regime is written
as the table caption in markdown and as a `# regime:` header line in CSV. `markdown_table(df,
floatfmt=".4g")`. `manifest(paths, inputs, seed) -> dict` with SHA-256 of inputs and outputs.

### 10.7 `screen.py`

`direction_prior(smiles, pair) -> dict | None`: if `paths.GEN15_DEPLOY` exists, run
`generations/gen15_curve/scripts/g15_predict.py predict --smiles <smiles>` in a subprocess and
return `{"pair": pair, "sign": ±1, "source": "gen15 deploy_g15.joblib", "note": "pre-screen only; not a D source"}`;
else `None`. The only consumer is `g18_screen.py`, which writes `direction_prior` into candidate
lists; validator V6 refuses any D sourced from it.

## 11. Scripts (run from the repository root with `.venv/Scripts/python.exe`)

| order | script | CLI | writes |
|---|---|---|---|
| 1 | `g18_build_db.py` | `[--systems-dir systems] [--seed 18]` | bundle SHA check, ingest, literature entries, validate all, `systems/*`, `results/audit/ingest_audit.json` |
| 2 | `g18_audit.py` | `[--systems-dir]` | `DATA_AUDIT.md` (tables of §0.2 recomputed; cohort lists for E1 and E2; duplicates; replicate floor); freezes the cohort |
| 3 | `g18_seal_prereg.py` | `[--check]` | SHA-256 of `PRE_REGISTRATION.md` above its footer into the footer and `results/eval/prereg_sha256.txt`; `--check` verifies and exits non-zero on mismatch (the fit scripts call it) |
| 4 | `g18_fit_dmodels.py` | `[--system-id ... | --all] [--band 20-30C] [--seed 18]` | `results/dmodels/<system_id>.json` (parameters, SE, reliability, hull, manifest hash); writes the fitted block into `systems/<id>.json` |
| 5 | `g18_eval_dmodels.py` | `[--seed 18] [--n-boot 2000]` | `results/eval/lopo.csv`, `e1_macro.csv`, `e3_reliability.csv`, `comparisons.csv`, `DECISION.md` (R1 verdict verbatim) |
| 6 | `g18_loading_check.py` | `[--oa 0.5 1 2] [--seed 18]` | `results/loading/e2_series.csv`, `e2_summary.csv`, `phi_exploratory.csv`, `DECISION.md` (R2) |
| 7 | `g18_bench.py` | `[--n 50]` | `results/bench/timing.json` (median ms per stage solve and per cascade for the reference cascade N = 6/3/3, M = 2, K = 1; per LHS evaluation); the only file with wall-clock values |
| 8 | `g18_case_prnd.py` | `--feed cases/prnd_feed.json --spec cases/prnd_spec.json [--draws 64] [--n-lhs 1000] [--seed 18] [--allow-placeholders]` | §13 outputs under `results/case_prnd/` |
| 9 | `g18_optimize.py` | `--system-id <id> --feed <json> --spec <json> [--n-lhs 2000] [--seed 18] [--objective consumption|cost] [--bo]` | `results/optimize/<id>/…` |
| 10 | `g18_recipes.py` | §9.4 | `results/recipes/…` |
| 11 | `g18_screen.py` | `--smiles-file <txt> --pair Nd Pr` | `results/screen/priors.csv` (empty with a note when the joblib is absent) |
| 12 | `g18_report.py` | | assembles `GEN18_REPORT.md` tables from `results/`; refuses if `prereg_sha256.txt` differs from the sealed file |

Every script writes `<output_dir>/manifest.json` (input SHA-256s, seed, arguments, git HEAD).
`results/MANIFEST.sha256` lists every output.

### 11.3 End-to-end budget

Ingest < 2 min, < 300 MB. Fits and LOPO: 14 systems x <= 18 publications x lstsq on <= 600 rows,
plus 2000 bootstrap resamples of a 14-vector: < 5 min. Loading check: 10 series x <= 18 stage
solves x 3 O/A values plus LOO phi (<= 8 x 20 solves): < 2 min. Cascade: reference evaluation
(Newton, N = 12, M = 2, K = 1 -> 84 unknowns) expected 5–15 ms; `g18_bench.py` records it. Case
study: 64 draws x 2 systems x 1000 LHS = 128 000 cascades -> about 20–30 min at 10 ms; the script
prints the projected time from the bench file and `--n-lhs` scales it. Recipes: `n_lhs` x number of
parameterised systems. Nothing exceeds 1 GB; all single-process.

## 12. Tests (`tests/`, pytest; markers `slow`, `validation`)

### 12.1 Shared synthetic systems (`gen18proc/testsystems.py`, exact content)

* `two_metal_cation_exchange()`: metals Pr, Nd; `log_k = {"Pr": -2.10, "Nd": -1.95}` (dimer basis),
  `a = b = 3`, `q = 3`, `p = 3`, `z = 0`, chloride, `[HA]_T = 0.8` (dimer 0.4), no complexant; domain
  box wide (log acid [-3, 1], log ligand [-2, 0.5]).
* `two_metal_solvating()`: metals Pr, Nd; `log_k = {"Pr": 1.50, "Nd": 1.80}`, `n = 2.7`,
  `p_anion = 2.0`, `p_h = 0`, `K_H = 0.5` (L²/mol²), `z = 3`, nitrate, `L_T = 0.1`.
* `constant_d_system(log_d: dict)`; `complexant_spec()`: Pr `log_beta = (3.0,)`, Nd `log_beta =
  (2.0,)`, protonation `log K_H = (2.0,)`, regeneration 0.9 (assumed, range [0.5, 1.0]).
* `feed_prnd()`: chloride, Pr 0.025 M, Nd 0.075 M, `h = 0.01`, anion 0.31, flow 1.0 L/h.
These numbers are synthetic test fixtures, not literature values, and are labelled so in the file.

### 12.2 Invariant tests (must pass)

| file | assertion | tolerance |
|---|---|---|
| `test_bundle.py` | `assert_bundle_unchanged()` returns the frozen SHA | exact |
| `test_systems.py` | V1–V17 each rejected by a minimal failing fixture; both example entries of §3.7/§3.8 load; `write_system` then `load_system` round-trips byte-identically; system_id recomputes | exact |
| `test_ingest.py` | ingest audit counts equal `DATA_AUDIT.md`; exactly 7 rows flagged `UNIT_SLIP_DUPLICATE` in pub_0e7f3e0563 and all on the 3 M side; 10 publication-aware loading series; NaN-acid rows `fit_eligible=False`; only the §4.1 columns are read (assert on the pyarrow column list) | exact |
| `test_dmodel.py` | analytic partials vs central finite differences | rel 1e-6 |
| `test_dmodel.py` | two-ligand composition: `D = D1 + D2`; ligand 1 depleted by `y^(1)` only | rel 1e-12 |
| `test_dmodel.py` | complexant lowers effective D by exactly `alpha_i`; increasing `h` lowers `alpha` through protonation | rel 1e-12; monotone |
| `test_domain.py` | inside the box and hull -> no OOD flag; outside on one axis -> that flag with the stated distance | abs 1e-12 |
| `test_equilibrium.py` | per-metal closure `x + r y = T` | rel 1e-12 |
| `test_equilibrium.py` | primary vs fallback path agreement | rel 1e-10 |
| `test_equilibrium.py` | multi-start (8 starts, L0 geometric in (1e-6, 1] x L_T) agreement, both mechanisms, with and without complexant | rel 1e-9 |
| `test_equilibrium.py` | D = 0 (`log_d = -inf`): `y = 0`, `x = T`; D -> inf (`log_d = 60`): `x <= 1e-12 T`, `y = T/r` | abs/rel 1e-12 |
| `test_equilibrium.py` | loading lowers D: 1 mM vs 50 mM metal at `L_T = 0.1`, both mechanisms: `log D(50) < log D(1) - 0.05`; cation exchange releases acid: `h_out - h_in = 3 r (y - y_in)` on branch 1 | rel 1e-12 |
| `test_equilibrium.py` | complexant depletes free ligand: `c_free < c_total` by the bound amount; balance (4) closes | rel 1e-12 |
| `test_equilibrium.py` | alkali reserve: released acid titrates `B` first; excess base -> branch 2, `ALKALI_EXCESS`, `h = H_MIN`, proton ledger closes | rel 1e-12 |
| `test_cascade.py` | Kremser: `ConstantD`, `n_scr = n_str = 0`, `f_bleed = 1`, E in {0.5, 1, 2, 10}, N in {1, 3, 8}: `kremser_init`, Newton and SS all match the oracle | abs 1e-10 |
| `test_cascade.py` | Jacobian vs finite differences on a 3/2/2 cascade with recycle | rel 1e-6 |
| `test_cascade.py` | Newton vs SS agreement on every stream quantity (both fixtures, with recycle, with complexant in the scrub) | rel 1e-8 |
| `test_cascade.py` | `check_balances` max (metals, ligands, protons, anion, complexant, sodium) | rel 1e-8 |
| `test_cascade.py` | monotone: raffinate T non-increasing in `n_ext` (1..10) and in `O/A` (0.5..5) with lean organic | slack -1e-10 |
| `test_cascade.py` | origin pass: label sums equal totals; identity of §7.8 | rel 1e-10 |
| `test_cascade.py` | pure-target scrub: `ConstantD`: `recovery_from_feed` identical for scrub T = 0 and 50 mM; loading model: `recovery_from_feed <= 1` and the §7.8 identity holds | abs 1e-10 |
| `test_cascade.py` | dilute D dropped into the cascade differs from the loading-aware result: `ConstantD` at tracer D vs `CationExchangeMassAction` at feed 0.1 M: recovery differs by > 0.01 | — |
| `test_cascade.py` | failure is a status: spec with `n_ext = 200`, `O/A = 1e-3`, `log K` shifted by +20 returns `status in {"failed", "converged_*"}`; never raises; NaN metrics when failed | — |
| `test_metrics.py` | purity/recovery/enrichment on a hand-built stream table; consumption per mol; NaN + `COST_INCOMPLETE` when a price is missing; `LIGAND_LOSS_NOT_MEASURED` when loss is null and `f_bleed = 0` | rel 1e-12 |
| `test_optimize.py` | two runs of `lhs_pareto(n=50, seed=18)` produce byte-identical CSV; `n_failed` counted; OOD candidates present in `all` and absent in `in_domain_only` | exact |
| `test_evalproto.py` | on synthetic records generated from the solvating fixture with known `n = 2.7`, `p = 2.0` and per-publication offsets: M1 recovers `n`, `p` within 1e-6 (noise-free) and LOPO with offset 0 reproduces the injected offsets as errors; B0/B1 computed as specified; bootstrap deterministic | abs 1e-6 |
| `test_report.py` | `write_table` refuses without the three regime keys | raises |
| `test_screen.py` | with the joblib absent (monkeypatched path) `direction_prior` returns None; V6 rejects a gen15-sourced record | — |

### 12.3 Validation tests (marker `validation`; a failure is a reported result, not a blocker)

| file | assertion |
|---|---|
| `test_todga_slope.py` | the primary in-sample pooled fit of `sys_5cb78e5000d40860`, band 20-30C, gives `n_solvation` in [2.36, 2.88] and `jackknife_se` is finite and reported |
| `test_loading_direction.py` | on the loading-active series (log D range >= 0.3) Spearman(log mM, log D) < 0 |

Fast run: `.venv/Scripts/python.exe -m pytest generations/gen18_process/tests -q -m "not slow and not validation"`;
full run drops the marker filter. No test asserts wall-clock time.

## 13. The Pr/Nd case: PC88A versus Cyanex 272 (`scripts/g18_case_prnd.py`)

### 13.1 Inputs (editable files, every number `assumed` with a range, open item U1)

`cases/prnd_feed.json`: chloride; `Nd 75 mM`, `Pr 25 mM` (Nd:Pr 3:1 mol; total 0.1 M), `[H+] 0.01 M`,
chloride 0.31 M, 25 °C, flow 1 L/h; sensitivity sets: total metal in {0.03, 0.1, 0.3} M, ratio in
{1:1, 3:1}. `cases/prnd_spec.json`: a **grid** of product specifications, purity_min in {0.95, 0.97,
0.99} (mol basis) x recovery_min in {0.80, 0.85, 0.90}; no single cell is privileged; the cell
(0.97, 0.85) is the consistency cell (§13.4). The user must confirm feed and spec (open item U1);
until then every case table carries `feed_status: assumed`.

### 13.2 Parameters and draws

Both systems are the literature entries of §3.7. The script refuses to write headline tables
while any input `assumed` value lacks a range (V9); `--allow-placeholders` is required to run at all
and stamps every output `PLACEHOLDER_PARAMETERS`. Parameter draws: 64 draws (seed 18), each
variable uniform inside its declared range: `log_k.Nd`, `SF(Nd/Pr)` (PC88A [1.3, 1.5]; Cyanex 272
the assumed sweep {1.1, 1.2, 1.3, 1.4} crossed with the draws), `a_dimer`, `b_proton`, and the
extractant concentration as a design variable. Per draw: `lhs_pareto` with `n_lhs` (default 1000)
over `n_ext, n_scr, n_str` (1–40 each; up to 80 for the consistency check of §13.4 (b)),
`oa_ext`, `s_over_a`, `w_over_a`, `scrub_acid_M`, `scrub_target_mM` (0–50, displacement scrub),
`strip_acid_M` (0.5–6), `ligand_total_M` within the range, `saponification_degree` in {0, 0.3,
0.5} (always flagged `SAPONIFICATION_RANGE_UNKNOWN`), `feed_stage_offset`, `scrub_return_offset`.

### 13.3 Outputs (all under `results/case_prnd/`, every table with regime tag and parameter status)

* `parameters.md`: every parameter with value, range, status, source, note (the §3.7 table).
* Per system and spec cell: the best on-spec regime per draw -> reported as **intervals** over the
  64 draws (min, median, max) of `n_stages_total`, `oa_ext`, `s_over_a`, `w_over_a`, `scrub_acid_M`,
  `strip_acid_M`, `purity_mol`, `recovery_from_feed`, `recovery_total`, consumption per kg Nd2O3
  (acid, base, water, extractant make-up (NaN, `LIGAND_LOSS_NOT_MEASURED`)), cost proxy (NaN unless
  `config/prices.json` covers every item), flags, `regime_status`, and the fraction of draws in
  which the spec cell is reachable at all.
* Sensitivity sweeps (one factor at a time at the median draw): SF over its range, `log_k.Nd` over
  a 2-log-unit window, `[HA]_T`, saponification degree, O/A, feed total and ratio.
* The **displacement-scrub variant** (scrub liquor carrying Nd) with all four origin quantities.
* A **labelled sanity limit**: `ConstantD` at the tracer D of each section, no acid balance, same
  cascade, to show what loading and acid coupling change.
* `cyanex_vs_pc88a.md`: the comparison at identical feed, spec grid and price table, **conditional
  on the placeholder ranges**, stating that it cannot be decided until the literature values are
  transcribed by a person or measured; the ranking is reported per draw pair as a distribution
  (fraction of draws in which PC88A needs fewer stages / less acid for the same cell).

### 13.4 Consistency checks against the cited numbers (never called validation)

(a) Thakur 1993 (doi 10.1016/0304-386X(93)90084-Q): "97 % purity at > 85 % recovery, counter-current
PC88A". Question asked of the model: does any regime in the sweep reach the (0.97, 0.85) cell, at
what stage count, O/A and scrub composition, and in what fraction of draws? Reported as "consistent
with the cited outcome (reachable in x/64 draws with N = a–b stages)" or "not reachable within the
assumed window". (b) EP2388344A1 (LITERATURE_NOTES S1): PC-88A Nd/Pr circuit 72 extraction + 72
scrub + 8 strip stages at SF 1.4. Question: at SF 1.4 and the (0.99, 0.99) cell, is the stage count
the model needs of the order of 70 + 70 rather than 7 + 7? Reported as an order-of-magnitude
comparison. (c) Banda 2014 (doi 10.1016/j.jiec.2014.03.002): maximum SF about 1.5 — enters only as
the upper end of the SF placeholder range. None of (a)–(c) can validate the model: feeds, acidities
and loadings of the sources are unknown (LITERATURE_NOTES §2).

### 13.5 The DGA + aqueous ligand exploration (labelled exploratory)

On `sys_5cb78e5000d40860` (TODGA/nitrate/aliphatic; the only corpus-parameterised case) with
`cases/todga_prnd_feed.json` (nitrate, Pr 25 mM, Nd 75 mM, 3 M HNO3, assumed): (1) the TODGA-only
Pareto table (D source per the pre-registered decision, loading through the depletion term,
`K_H` null -> `ACID_UPTAKE_UNMODELLED`, `PHASE_BEHAVIOUR_UNKNOWN` above 0.3 loading); (2) the
hold-back complexant with `log_beta` placeholders (value None, ranges declared: the case sweeps
`delta log beta = log beta_Pr - log beta_Nd` in [0, 1.5] and `log beta_Nd` in [1, 4], protonation
`log K_H` in [1, 3]) in the feed and in the scrub only, reporting the beta contrast needed for a
given enrichment gain, the complexant consumption with the regeneration fraction (assumed range),
and the cross-flow depression of extraction when the complexant enters only in the scrub. Reported
as a sensitivity study, not a recommendation, until a sourced beta set exists (open item U3; the
"logK Ln-series SDFs" in memory are the candidate source).

## 14. `GEN18_REPORT.md` outline (every number with cohort, hold-out, averaging unit, parameter status)

1. What was built; what the database contains (systems by origin, records by status, publications,
   exclusions, two-ligand systems, scaffold links).
2. Data audit (from `DATA_AUDIT.md`): cohort counts for E1 (groups, systems, publications) and E2
   (series under both definitions), duplicates (unit-slip rows, tied groups), replicate floor,
   NaN-metal rows, temperature bands.
3. Pre-registered D-model result: E1 table (M1, B0, B1; LOPO; group -> system -> macro; paired
   bootstrap CI; wins/14), in-sample and offset-calibrated variants labelled; R1 verdict verbatim and
   the D source the chain uses as a consequence. E3 reliability table (per system n, p_eff,
   jackknife SE, split-half where defined, `interpretable`; TODGA n vs [2.36, 2.88] with the
   `validation` test outcome). Regimes: cohort E1, hold-out LOPO, unit = system.
4. Loading (E2): per-series table (10 series; direction, `mae_constant`, `mae_ideal`, O/A
   sensitivity), the loading-active subset, R2 verdict verbatim; the phi arm (exploratory, LOO).
5. Exploratory analyses, counted and BH-adjusted (`comparisons.csv`).
6. Cascade verification: Kremser, balances, Newton-vs-SS, multi-start, origin identity, timing
   (from `results/bench`, labelled as the only wall-clock numbers).
7. Pr/Nd case: parameter status table; interval tables per spec cell; sensitivities; displacement
   scrub; sanity limit; PC88A vs Cyanex 272 conditional comparison; consistency checks (a)–(c) with
   their verdict strings. Regime: feed `assumed`, parameters `ASSUMED_PLACEHOLDER`, 64 draws, seed 18.
8. TODGA Pr/Nd nitrate recipe table (Pareto, in-domain and all, flags) and the DGA + aqueous ligand
   exploration (labelled).
9. Consumption and cost: the price table with statuses and ranges; consumption as primary.
10. Limits, nulls, defects: what is unmodelled (assumptions tuple), `PHASE_BEHAVIOUR_UNKNOWN`
    prevalence, `ACID_UPTAKE_UNMODELLED`, the O/A and metal-semantics assumptions, what a single
    laboratory measurement would change most (D-optimal choice, gen15 §7 style).
11. gen15 pre-screen usage and its limits.
12. Reproduction: commands in order, manifest hashes, `prereg_sha256`, comparison count.

## 15. Work breakdown (four engineers, disjoint files, dependency order)

| WB | owner files | depends on | summary |
|---|---|---|---|
| WB1 data | `gen18proc/__init__.py`, `paths.py`, `systems.py`, `literature.py`, `ingest.py`; `scripts/g18_build_db.py`, `g18_audit.py`, `g18_seal_prereg.py`; `cases/*.json`; `tests/test_bundle.py`, `test_systems.py`, `test_ingest.py`; `DATA_AUDIT.md`, `README.md` | none | schema, validator, literature entries (§3.7), ingest, series, duplicates, audit, sealing |
| WB2 models | `dmodel.py`, `domain.py`, `equilibrium.py`, `testsystems.py`; `tests/test_dmodel.py`, `test_domain.py`, `test_equilibrium.py` | WB1 dataclasses (§3.2; code against this file, integrate when WB1 lands) | D models, composition, flags, domain, stage solver both paths |
| WB3 cascade | `cascade.py`, `metrics.py`; `config/prices.json`; `scripts/g18_bench.py`; `tests/test_cascade.py`, `test_metrics.py` | WB2 `solve_stage`, `DEval`, `testsystems` (code against §5–6; use `ConstantD` for the Kremser tests first) | topology, Newton, SS, Kremser, origin pass, balances, metrics |
| WB4 evaluation | `evalproto.py`, `optimize.py`, `screen.py`, `report.py`; `config/design_spaces.json`; `scripts/g18_fit_dmodels.py`, `g18_eval_dmodels.py`, `g18_loading_check.py`, `g18_optimize.py`, `g18_recipes.py`, `g18_case_prnd.py`, `g18_screen.py`, `g18_report.py`; `tests/test_evalproto.py`, `test_optimize.py`, `test_report.py`, `test_screen.py`, `test_todga_slope.py`, `test_loading_direction.py`; `GEN18_REPORT.md` | WB1 records, WB2 stage solve (loading check), WB3 cascade+metrics (optimiser, case) | fits, LOPO, reliability, loading, optimiser, driver, case study, report |

Order: WB1 first (its dataclasses are the contract; the JSON examples of §3.7–3.8 are its fixtures);
WB2 and WB3 in parallel from hour zero against this file (WB3 starts with `ConstantD` and the
Kremser tests, then plugs in `solve_stage`); WB4 starts with `evalproto.py` and `report.py` against
`corpus_records.csv` and adds the optimiser and case scripts when WB3's `solve_cascade` lands.
Integration (orchestrator): run scripts 1–3, then 4–6, then 7–12; the `validation` tests are run
once and their outcome goes into the report. No engineer edits another's files; interface changes
go through a dated addendum to this file.

## 16. Open items for the user or orchestrator (not blocking implementation)

* U1 Feed composition and product specification for the Pr/Nd case (defaults are `assumed`).
* U2 Reading the source of pub_0e7f3e0563 (and pub_5a68dc5665) for O/A and the metal-concentration
  semantics before `PRE_REGISTRATION.md` is sealed; if read, a dated pre-fit addendum records it.
* U3 A sourced beta set (Pr, Nd) for a hydrophilic DGA or sulfonated BLPhen in nitrate; without it
  §13.5 stays a sensitivity study.
* U4 Transcription of PC88A and Cyanex 272 parameters from Banda 2014, Thakur 1993 and a Cyanex
  272 Pr/Nd source by a person (double entry, DOI + table/figure locator, value + range, conditions
  including diluent, `[HA]_T`, O/A, saponification degree, equilibrium pH), replacing the
  placeholders; the case is re-run unchanged.
* U5 Prices for `config/prices.json` (until then consumption is the only economic number).
* U6 A sourced third-phase LOC for TODGA in aliphatic diluent, so the DGA exploration does not run
  `PHASE_BEHAVIOUR_UNKNOWN` throughout.
