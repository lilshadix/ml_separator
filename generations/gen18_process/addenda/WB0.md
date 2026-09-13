# Addendum WB0 — foundations layout (2026-09-13)

Owner: WB0 (foundations). Files: `gen18proc/__init__.py`, `gen18proc/paths.py`,
`gen18proc/types.py`, `gen18proc/systems.py` (initial version), `tests/conftest.py`.
Interfaces affected: DESIGN.md section 2 (layout), section 3.2 (where the dataclasses live),
section 5.7 (`ModelBuildError`), section 7.1 (`CascadeSpec` validation), section 1.3 (JSON
helpers). Nothing in this addendum changes a name, a field or a field order of DESIGN.md.

## A1. Layout: every dataclass lives in `gen18proc/types.py`; `systems.py` re-exports it

DESIGN.md section 2 lists no `types.py` and section 3.2 places the schema dataclasses in
`systems.py`. To let WB2, WB3 and WB4 import the contract without importing WB1's loader,
validator and RDKit dependency, **every** enum, protocol and frozen dataclass of the design is
defined once in `gen18proc/types.py`:

| DESIGN.md | names |
|---|---|
| 1.2 | `ProvStatus` |
| 1.3 | `Source`, `Provenance`, `Sourced` (each with `to_json()` / `from_json()`) |
| 1.4 | `Flag`, `INADMISSIBLE_FLAGS`, `OOD_FLAGS` (frozensets); plus `REGIME_STATUSES` tuple |
| 3.2 | `Mechanism`, `LigandSpec`, `Stoichiometry`, `ComplexantSpec`, `MediumSpec`, `CationExchangeParams`, `SolvatingParams`, `PhaseBehaviour`, `StreamRecord`, `DistributionRecord`, `ApplicabilityDomain`, `SystemEntry` |
| 3.6 | `Violation(level, path, message)`, `SystemValidationError(violations)` |
| 5.7 | `ModelBuildError(reason)` |
| 5.1, 5.6 | `StageState`, `DEval`, `DModel` (Protocol), `ActivityModel` (Protocol) |
| 6.1, 6.2 | `AqStream`, `OrgStream`, `StageDiagnostics` |
| 7.1, 7.9 | `CascadeSpec`, `CascadeResult` |
| 8 | `ProcessMetrics` |

`gen18proc/systems.py` re-exports all of them by explicit name (its `__all__` is a superset of
`types.__all__`), so `from gen18proc.systems import SystemEntry` and
`from gen18proc.types import SystemEntry` refer to the same class object. WB1 appends its own
code (system key, `system_id` hash, `validate_entry`, `load_system`, `write_system`,
`load_registry`) below the marker line `# --- WB1 extends below ---` and must not move or
redefine any re-exported name. Other owners may import from either module; importing from
`gen18proc.types` keeps their modules free of WB1's dependencies.

Extra constants defined alongside (not in DESIGN.md, harmless): `ASSUMED_LABEL =
"ASSUMED_PLACEHOLDER"`, `SOURCE_KINDS = ("corpus", "doi", "none", "model")`,
`REGIME_STATUSES`, `systems.SCHEMA_VERSION = "gen18.1"`, `gen18proc.DEFAULT_SEED = 18`.

## A2. JSON helpers on `Source` / `Provenance` / `Sourced`

`Sourced.to_json()` returns exactly the flat object of DESIGN.md section 1.3 (keys `value, unit,
status, assumed_label, range, source, model_id, fit_manifest_sha256, note`; `range` as a 2-list or
null; `source.safe_exp_ids` as a list). `Sourced.from_json(obj)` is its inverse and tolerates
missing keys (defaults: `status` -> `unknown`, `source` -> `{"kind": "none"}`, `range` -> null,
`note` -> `""`), which the partial fitted-block objects of section 3.8 need. `Provenance.to_json()`
is the nested object used inside `DistributionRecord.provenance` (same keys minus `value`,
`unit`). `Sourced` exposes read-only properties `status`, `range`, `source` as shortcuts to its
`provenance`. `Sourced.unknown(unit, note=..., source=...)` builds the `value: null` /
`status: unknown` object used for every unmeasured phase and stream field.

The `range` tuple is kept as parsed (ints stay ints); `write_system` (WB1) is responsible for
`sort_keys=True, indent=2, ensure_ascii=True` and float formatting.

## A3. `ModelBuildError` is a frozen dataclass, not an exception

Section 5.7 says `build_system_model` *returns* `SystemModel | ModelBuildError(reason)` and
never raises for a refusal, so `ModelBuildError` is a frozen dataclass with one field `reason:
str`. Callers test `isinstance(result, ModelBuildError)`. `SystemValidationError` (section 3.6)
*is* raised by `load_system`, so it subclasses `ValueError` and carries `.violations` as a tuple.

## A4. `CascadeSpec` carries no validation of its own

The `ValueError` validation listed under section 7.1 (flows, `n_ext < 1`, stage indices,
saponification degree, unknown metals) needs the system model, so it is performed by
`cascade.solve_cascade` (WB3), not in `CascadeSpec.__post_init__`. Constructing an invalid spec is
therefore legal; solving it raises `ValueError`, which the optimiser records as `invalid_spec`.

## A5. `paths.py`

As section 2, plus: `REPO_ROOT`, `GEN13_ROOT`, `BUNDLE_DIR`, `BUNDLE_SHA256` (re-exported),
`GEN15_PREDICT_SCRIPT`, `SCRIPTS_DIR`, `TESTS_DIR`, `ADDENDA_DIR`, one constant per results
subdirectory (`RESULTS_AUDIT_DIR`, `RESULTS_DMODELS_DIR`, `RESULTS_EVAL_DIR`,
`RESULTS_LOADING_DIR`, `RESULTS_BENCH_DIR`, `RESULTS_CASE_PRND_DIR`, `RESULTS_RECIPES_DIR`,
`RESULTS_OPTIMIZE_DIR`, `RESULTS_SCREEN_DIR`), `RESULTS_MANIFEST`, `PRE_REGISTRATION_MD`,
`DATA_AUDIT_MD`, `GEN18_REPORT_MD`. All output directories are created on import (as
`gen13sep.paths` does). Verified on 2026-09-13: `assert_bundle_unchanged()` returns
`fefbefc6fe993aa9ce9db1a0c338adb9e5f58a8b75bab084cc1df4e024faf5dd`.

## A6. Field-type conventions inside `types.py`

Stream and stage classes use plain `dict` (section 6.1 shape); database classes use
`collections.abc.Mapping` and tuples (section 3.2 shape); `Sourced.value` is typed `float | None`
but `from_json` passes through whatever JSON holds (a boolean for `third_phase_observed` is
allowed). `DModel` and `ActivityModel` are `runtime_checkable` Protocols. `ProcessMetrics.phase`
is `Mapping[str, Any]` (the copied `Sourced` items plus the `regenerability_note` string);
`ProcessMetrics.consumption` is `Mapping[str, float]` with keys named by WB3 (unit suffixes in the
key). The count of `Flag` members is 27, of `OOD_FLAGS` 11.
