# Addendum WB4a — evaluation core (2026-09-13)

Owner: WB4a. Files: `gen18proc/evalproto.py`, `gen18proc/report.py`, `gen18proc/screen.py`,
`config/design_spaces.json`, `tests/test_evalproto.py`, `tests/test_report.py`,
`tests/test_screen.py`. Interfaces affected: DESIGN.md sections 10.1–10.3, 10.5–10.7, 9.1 and
PRE_REGISTRATION.md sections 3–6. Nothing here changes a decision rule, a baseline definition or a
name of the design; it fixes what the design left open and states the additions other owners can
rely on. No fit on corpus data was run while writing this (the synthetic tests are the only fits).

## A1. Record layout consumed by the fitter (WB1 → WB4)

`evalproto` consumes the flat `systems/corpus_records.csv` layout of DESIGN 3.2. Mapping-valued
fields are accepted in **either** of two forms, resolved per column by `ligand_column(df, name)`
and `metals_initial_column(df, metal)`:

| field | recommended CSV form (what WB1 should write) | also accepted |
|---|---|---|
| `ligand_M` | one flattened column per ligand, `ligand_M.<ligand name>` (NaN where the ligand is absent from the system) | a single `ligand_M` column holding a dict (in-memory frame) or its JSON string (`{"TODGA": 0.1}`) |
| `metals_initial_mM` | `metals_initial_mM.<metal>` | a single `metals_initial_mM` column, dict or JSON string |

Reason for the flattened form: it is what `pandas.read_csv` returns without a parser and what
`groupby` can key on. Required columns: `record_id, metal, log_d, acid_nominal_M, publication_id`
plus the ligand column(s); `system_id` is used to loop over systems (defaulted to `"unspecified"`
when absent); `fit_eligible`, `duplicate_flag`, `temperature_C`, `anion_M`, `diluent_name`,
`contact_time_min`, `complexant_M`, `loading_series_id`, `is_tracer` are used when present.
Booleans may arrive as `True/False` or the strings `"True"/"False"`.

**Update 2026-09-13 (after WB1 landed, `WB1.md` A5).** WB1's `corpus_records.csv` uses neither
form above: it carries a **scalar** `ligand_M` (the primary extractant's formal concentration)
beside `ligand_name`, a scalar `metal_initial_mM` (the row's own metal), and `replicate_group_id`.
`evalproto` now accepts this third form as well, resolved per column in this order:

| form | ligand concentration of `<name>` | initial mM of `<metal>` |
|---|---|---|
| flattened | `ligand_M.<name>` or `ligand_M__<name>` (WB2's spelling) | `metals_initial_mM.<metal>` / `__` |
| scalar (WB1's file) | `ligand_M` (numeric dtype) where `ligand_name == <name>`, else NaN; without a `ligand_name` column the value is taken as-is | `metal_initial_mM` where `metal == <metal>`, else NaN |
| nested | `ligand_M` object column: dict or JSON string per cell; a numeric cell inside such a column follows the scalar rule | `metals_initial_mM` likewise |

A flattened frame that has no column for the requested ligand raises `ValueError` (malformed
input); the scalar and nested forms answer "absent" with NaN. For cation exchange the dimer
halving is applied after resolution. When `temperature_C` is absent but WB1's
`temperature_band` column exists, the band filter compares the band strings. No decision rule,
baseline or name changed; `tests/test_evalproto.py` runs the M1 recovery under all three forms
and a parse-only check on the built `corpus_records.csv` (skipped when absent; no corpus fit is
run before sealing).

## A2. Replicate-group key (DESIGN 10.1 step 1)

The flat CSV does not carry the 64-column bundle key. `aggregate_replicates` therefore groups by,
in order of preference: (a) an explicit `replicate_group_id` column (**what WB1 writes**, `WB1.md`
A5: the blake2b of the exact 64-column condition key + publication + SMILES + metal, so the
aggregation equals the pre-registered one; updated 2026-09-13 — `replicate_group` and
`condition_key` remain accepted spellings); (b) otherwise the tuple (publication_id, metal,
acid_nominal_M, anion_M, temperature_C, diluent_name, contact_time_min, complexant_M, the full
ligand mapping, the full metals_initial_mM mapping), NaN as a value. The aggregated point keeps
the smallest `record_id` of the group as its `record_id` (the B1 tie rule and the offset-mode
calibration point use it), the mean log D, and weight `n_rep`. Pre-aggregated frames may carry an
`n_rep_weight` column that is summed instead of counting rows (used internally by LOPO).

## A3. `fit_mass_action` — what is fixed beyond the contract

* Signature as DESIGN 10.1 plus keyword-only extras: `reliability: bool = True` (compute the
  jackknife and split-half of 10.3 inside the fit; LOPO's internal fits pass `False`),
  `split_half_repeats: int = 20`, `fit_manifest_sha256: str | None = None` (filled by
  `g18_fit_dmodels.py`; the fit itself cannot know the hash of the file that will contain it).
* `band` may be `None` (no temperature filter). Band strings `"<20C"`, `"20-30C"` (as
  `lo <= T < hi`), `">=50C"`; NaN temperature is assigned to `"20-30C"` (DESIGN 3.4).
* Distinct levels are counted on the aggregated training points after rounding log10 values to
  1e-9. A slope fixed at its prior has `se = 0`, `slope_status[key] = "assumed"`, and is
  **never** `interpretable` (a jackknife SE of a constant measures nothing); this is a stricter
  reading of 10.3 recorded here.
* Sum-to-zero publication effects are implemented by effect coding (`k − 1` free columns for the
  `k >= 2` multi-point publications, last effect = −sum). `se` and `se_hc1` carry the constrained
  effect's SE from the linear combination. `covariance` is the classical `(X'WX)^-1 s^2` over
  `param_names`; `covariance_hc1` the HC1 sandwich; `s^2` uses `N − k` with `N` = number of
  aggregated points (not the sum of weights).
* For `mechanism = cation_exchange` the ligand axis is `log10(ligand_M / 2)` (dimer basis,
  DESIGN 1.1) and `p_eff = −b_proton`; `as_cation_exchange_params` maps accordingly. The E1
  cohort has no such system; this is for literature/synthetic use.
* `FitResult` has the contract's fields in the contract's order, followed by extras with
  defaults: `slope_status`, `flags` (always `{EQUILIBRIUM_ACID_ASSUMED_NOMINAL, OA_ASSUMED}`),
  `se_hc1`, `covariance_hc1`, `param_names`, `residual_sd`, `distinct_levels`,
  `record_ids_by_metal` (training record ids per metal → `safe_exp_ids` of the fitted block),
  `mechanism`, `status` (`"fitted"` | `"no_data"`), `n_rows`, `priors`. A `predict(metal, lE, lA,
  publication_id=None)` method gives the model value (effect 0 for an unknown publication).
* `domain`: filled by `gen18proc.domain.build_domain(prepared_records, ligand)` when that WB2
  module imports and accepts the frame; otherwise `None`. The fit script (WB4b) may overwrite it.
* `pooled_slopes=False` (exploratory X2) fits `n.<metal>` / `p_eff.<metal>` columns; `n` and
  `p_eff` of the result are then the unweighted metal means.
* Malformed input (missing columns, more than one `system_id`) raises `ValueError`; an empty
  training set returns `status = "no_data"` with NaN values, never an exception.

## A4. LOPO table columns and the offset variant

`lopo_evaluate` returns the columns `system_id, holdout ("lopo"), publication_id, metal, status,
mae_M1, mae_B0, mae_B1, mae_M1_offset, mae_B1_crossmetal, n_points, n_rows, n_train_same_metal,
n_offset_points, slope_status_n, slope_status_p`. Rows with `status =
"metal_absent_from_training"` are the "skipped and counted" metals of PRE_REGISTRATION 4 and
carry NaN errors; `macro_over_systems` uses only `status == "scored"`. Errors are computed on the
**aggregated** held-out points (the same unit the fit uses); `n_rows` is the raw row count.

The offset variant chooses **one point per held-out publication** (the smallest `record_id` among
its points whose metal has a training intercept), sets the publication effect from it, and scores
every other point of that publication (all metals) with that effect; a one-point publication has
`mae_M1_offset = NaN`. `mae_B1_crossmetal` is the literal reading of X3: 1-NN over all metals of
the system in (lA, lE), no intercept correction.

`in_sample_evaluate` (added) produces the labelled `in_sample` rows of PRE_REGISTRATION 4 with the
same columns (`holdout = "in_sample"`, M1 with its fitted publication effect, B0/B1 over all points).

Analytic note used by the tests: with effect 0 for the held-out publication, the sum-to-zero fit on
the remaining publications puts the mean of their true offsets into the intercepts, so the noise-free
held-out error is `|delta_j − mean_{i != j} delta_i|`, which equals `|delta_j|` only when the
others' offsets sum to zero (both cases are asserted).

## A5. `macro_over_systems`, `paired_system_bootstrap`

`macro_over_systems(table, columns=MAE_COLUMNS) -> MacroResult(per_system: DataFrame, macro:
Series, n_systems, point_weighted: Series)`: mean over scored (publication, metal) rows within a
system, then mean over systems; `point_weighted` is the labelled alternative (weight = `n_points`).
`paired_system_bootstrap(a, b, n_boot=2000, seed=18, alpha=0.05) -> (mean, lo, hi)` is the
percentile interval of `mean(a − b)` over units resampled with replacement; it serves both the
system-level R1 interval and the series-level R2 interval (same function, WB4b).

## A6. `ComparisonCounter`

`record(name, family, p_value, note="")` refuses duplicate names and unknown families;
`table()` columns `name, family, p_value, p_adjusted, adjustment, n_in_family, note` with
Benjamini–Hochberg over the exploratory family only (NaN p-values are excluded from `m`);
`counts()`; `write(path, regime=...)` goes through `report.write_table`.

## A7. `report.py`

* `write_table(df, path, *, regime, regime_table=False, floatfmt=".4g", float_format=None,
  index=False)`: refuses without `cohort, holdout, averaging_unit`; with `regime_table=True` also
  without `status_of_parameters` (the design says "for regime tables" without saying how a table
  declares itself one — the flag is that declaration; recipe, optimise and case tables must pass
  it). CSV header line: `# regime: cohort=…; holdout=…; averaging_unit=…; [status_of_parameters=…;]
  <other keys sorted>` (byte-stable; `;` and newlines inside values are replaced). Markdown: the
  caption line `*regime: …*` above the table. `read_table(path) -> (df, regime)` reads it back.
* `markdown_table(df, floatfmt=".4g", index=False, caption=None)` — no `tabulate` dependency.
* `manifest(paths, inputs, seed, *, arguments=None, extra=None) -> dict` with keys `schema, seed,
  git_head, arguments, inputs, outputs` (SHA-256 per path, `None` for a missing file; keys are
  repository-relative POSIX paths where possible); `write_manifest(path, …)` writes it. No
  wall-clock values.

## A8. `screen.py`

`direction_prior(smiles, pair)` returns exactly the four contract keys; `sign` is the sign of the
predicted `log10 D(pair[0]) − log10 D(pair[1])` from the gen15 pair table (obtained through
`--output` of the CLI, so no stdout parsing of numbers), and the `note` starts with the mandatory
`"pre-screen only; not a D source"` followed by the predicted log SF, its sd and the curve
direction. Every failure (model or script absent, non-zero exit, missing pair) returns `None`.
`gen15_available()` is exported for `g18_screen.py`. Verified on 2026-09-13 with the real
deploy model (test marked `slow`, one subprocess).

## A9. `config/design_spaces.json`

Schema `gen18.design_spaces.1`: `default.bounds` (stage counts 1–30, offsets 0–29, the three
ratios 0.2–5, scrub acid 0.01–3 M, strip acid 0.5–6 M, scrub target 0–50 mM, scrub complexant
0–0.1 M, strip anion 0–6 M, ligand total 0.05–1 M, saponification 0–0.5, feed dilution 0–1,
f_bleed 0–0.1), `default.integer`, `default.fixed`, `default.from_domain` (axes whose default
comes from the `ApplicabilityDomain` when one exists), `families.<family>.bounds` overrides
(acidic organophosphorus ligand 0.2–1.5 M; diglycolamide saponification fixed 0 and ligand
0.05–0.4 M). `n_scr`/`n_str` default to 1–30 as DESIGN 9.1 says; a 0 lower bound is a user
widening (DESIGN 7.1 allows it).

## A10. Not done here (for WB4b)

`loading_series_evaluate` is a stub raising `NotImplementedError("WB4b")`. The V6 test in
`test_screen.py` skips (never passes silently) until WB1's `validate_entry` and
`literature.pc88a_prnd_entry` land; it then builds an entry with one gen15-sourced record and
asserts a V6 error.
