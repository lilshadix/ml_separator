# Addendum WB4b — optimiser, loading evaluation, scripts (2026-09-13)

Owner: WB4b. Files: `gen18proc/optimize.py`, the 10.4 block of `gen18proc/evalproto.py`
(`loading_series_evaluate` and its private helpers, plus one docstring bullet), `scripts/`
`g18_fit_dmodels.py`, `g18_eval_dmodels.py`, `g18_loading_check.py`, `g18_case_prnd.py`,
`g18_optimize.py`, `g18_recipes.py`, `g18_screen.py`, `g18_report.py`, `tests/test_optimize.py`,
`test_todga_slope.py`, `test_loading_direction.py`. Interfaces affected: DESIGN.md sections 9,
10.4, 11 (rows 4-6, 8-12), 12.2 (`test_optimize`), 12.3, 13, 14. Nothing here changes a
decision rule, a baseline definition or a name of the design; every item is a place where the
design was silent, impossible as written on this machine, or where a concrete shape was needed.
No fit on sealed corpus data was run: every script was exercised only under
`--unsealed-dry-run` (two-system subset, outputs under `results/_dryrun/`, the frozen
`systems/` directory untouched) or on the synthetic fixtures.

## A1. `optimize.py` (section 9)

* **`DesignSpace(bounds, integer, fixed, levels)`** — `levels` is an addition: a variable
  restricted to a finite value set, drawn by index `floor(u * n_levels)` (the case study's
  `saponification_degree in {0, 0.3, 0.5}` of 13.2). Integers follow the design's rule
  `floor(u (hi - lo + 1)) + lo` clipped. Only names in `VARIABLE_NAMES` (the seventeen of 9.1)
  are accepted. `DesignSpace.from_config(family, domain, ligand, widen, fixed, levels)` builds
  the default: `config/design_spaces.json` defaults and family overrides, then the domain's
  intervals for the `from_domain` axes (`10 ** log_acid` for scrub and strip acid,
  `10 ** log_ligand[ligand]`, `10 ** log_complexant` when recorded), then the user's `widen`
  (replaces bounds), `fixed`, `levels`. Note for literature entries: their hand-entered domain
  box (e.g. PC88A log acid [-1.42, -1.02]) makes the default strip acid 0.038-0.095 M; the case
  script widens it to the 0.5-6 M of 13.2, as the design intends ("the user may widen").
* **Offsets and the dry-stage rule.** Addendum WB3 makes a stage that no aqueous stream reaches a
  malformed specification (`ValueError`). Independent offsets `feed_stage_offset` and
  `scrub_return_offset` both > 0 leave the top extraction stage dry, and in a first LHS 36 of 40
  candidates were `invalid_spec` for that reason. `candidate_spec` therefore shifts both offsets
  down by their minimum (the relative position of feed entry and scrub return is kept; one of
  them is always the top stage), sets the feed offset to 0 when `n_scr = 0`, clips to
  `[0, n_ext - 1]`, and writes the effective offsets back into the row's variable columns so a
  row records what was run.
* **`candidate_spec` conventions** (the design names the variables but not their mapping):
  `feed_dilution` d dilutes the base feed (`A (1 + d)`, concentrations `/ (1 + d)`, dilution
  water into `water_L_h`); `O = oa_ext A'`, `S = s_over_a A'`, `W = w_over_a A'`; scrub
  `h = scrub_acid_M`, `anion = scrub_acid_M + 3 x scrub_target` (the displacement target as its
  trivalent salt); strip `h = strip_acid_M`, `anion = strip_acid_M + strip_anion_M` (the salting
  anion beyond the acid, which is what `metrics` counts as `salting_anion`); `ligand_total_M` is
  the formal monomer concentration applied to the system's first ligand and divided by the
  model's `ligand_scale` (2 for a dimer); `f_bleed > 0` adds a lean fresh organic with the same
  ligand totals. `SAPONIFICATION_RANGE_UNKNOWN` is added when `saponification_degree > 0` and the
  parameter set records no studied range containing it (at `s = 0` there is no reserve to flag).
* **`lhs_pareto(..., n, seed, objective="consumption"|"cost", solver_kwargs=None,
  progress=None)`** — the two keyword extras: `solver_kwargs` are passed to `solve_cascade`
  (`max_newton`, `max_sweeps`; see A6 for why a budget is needed), `progress(i, n)` is a
  callback. Rows: `front`, `front_in_domain`, `status`, `regime_status`, `in_domain`, `on_spec`,
  the variables in the design's order, `candidate` (LHS row index), every scalar metric, the
  consumption columns (`<item>_<unit>_h`, `<item>_<unit>_per_kg_oxide`), `consumption_index`
  and `consumption_index_incomplete` (a NaN component was excluded), `enrichment_factor_<I>`,
  `sf_tracer_extraction_<I>`, `sf_tracer_scrub_<I>`, `max_loading_<ligand>`, `flags`
  (`|`-joined, sorted), `n_flags`, `ood_distance_max`, `reason` (the `ValueError` text of an
  `invalid_spec` row), `iterations`, `balance_rel_max`. Failed / invalid rows have NaN metrics,
  `front` after every ranked row, and are counted in `df.attrs` (`n_failed`, `n_invalid_spec`,
  `n_converged`). Sorted by (`front`, variable tuple, `candidate`) with a stable sort: two runs
  give byte-identical CSV (tested). Non-dominated sorting is the plain O(n^2) pass per front.
* **`pareto_fronts(df)`** returns the first front twice (`in_domain_only`, `all`);
  **`epsilon_knees(df, purity_grid, recovery_grid)`** the minimum-`consumption_index` converged
  row per (subset, purity_min, recovery_min) cell with `reachable` and `n_feasible`.
* **`gp_bo`** returns `BOResult(status, reason, table)`; `status = "gated"` (nothing evaluated)
  unless `reliability_passes(system, reliability)` holds: both slopes `interpretable` in
  `results/dmodels/<system_id>.json` (or the `reliability` mapping passed) **and** the system
  model's `params_source` is the fitted set. Literature placeholders and the nearest-condition
  fallback never pass (there is no reliability flag to gate on); `force=True` overrides for
  testing and is recorded in the table's `attrs["gate"]`. ParEGO as designed (random Dirichlet
  weights, augmented Chebyshev with rho 0.05, `ConstantKernel * Matern(nu=2.5, ARD) +
  WhiteKernel`, EI over 2000 LHS candidates, kriging believer within the batch with the fitted
  kernel held fixed).

## A2. `loading_series_evaluate` (section 10.4)

* Signature extras (keyword-only): `phi_arm=True`, `phi_min_points=4`, `series_ids=None`.
  `fits[system_id]` may be a `FitResult` **or** a mapping with `n` and `slope_status` (what
  `g18_loading_check.py` reads back from `results/dmodels/<id>.json`); a slope that was fixed
  at its prior, or an absent fit, gives `n = n_fallback` with `n_source = "prior_n0"` (C1 of
  PRE_REGISTRATION section 3). The ligand name comes from WB1's `ligand_name` column (else the
  fit's `ligand`); the anion from the `anion` column (else nitrate).
* The per-series model is a single-metal `SolvatingMassAction` with `p_anion = p_h = 0`: at
  fixed acid within a series the acid term is a constant inside the anchored `log K`, so C1 is
  exactly "log K + n log L_f" with the ligand balance of 6.2 (ideal stoichiometry q = n, p = 0,
  z = 3; `K_H` unknown, hence `ACID_UPTAKE_UNMODELLED` above 1 M). The anchor is a bracketed
  Brent root in `log K` (lower end `log D_tracer - n log10(phi L_T)`, upper end expanded until
  the sign changes) checked to 1e-10 (`anchor_status` `anchored` / `anchor_tolerance` /
  `tracer_beyond_cap` / `anchor_failed`); all 10 corpus series anchor to < 1e-12.
* Output columns beyond the design's three: `n_used`, `n_source`, `log_k_anchored`,
  `anchor_status`, `c1_wins`, `delta_mae_c0_minus_c1`, `n_predicted`, `spearman_p`,
  `max_loading_fraction_pred`, `flags` (union of stage flags), `status`, `log_d_range`,
  `loading_active`, and the X1 columns `phi_loo_n`, `phi_median`, `phi_min`, `phi_max`,
  `mae_phi_loo`, `phi_insample`, `mae_phi_insample`. `df.attrs["points"]` holds the per-point
  rows as a **list of records** (a DataFrame in `attrs` breaks `pandas.concat`).
* The X1 objective is the MAE on the remaining points, minimised by
  `minimize_scalar(bounded=(0.01, 1), xatol=1e-4)` with `log K` re-anchored for every `phi`.
* Cost on the corpus: 1.2 s for the 10 series without the phi arm; the phi arm adds about 3 s
  per eligible series (18-point D3DODGA series: 6 s).
* Private helpers `_loading_n_of`, `_loading_series_model`, `_loading_stage`,
  `_loading_anchor`, `_loading_predict` and the constants `LOADING_ACTIVE_RANGE`, `_ANCHOR_TOL`
  live in the 10.4 block; the stage solver is imported lazily so `evalproto` keeps WB4a's light
  import footprint. One bullet of WB4a's module docstring (the "stub" line) was updated.

## A3. Scripts 4-6 (fit, eval, loading) and the seal

* Every one of the three calls `g18_seal_prereg.py --check` in a subprocess and refuses when it
  fails, **except** under `--unsealed-dry-run`: the fit then runs on `DRYRUN_SYSTEMS =
  (sys_5cb78e5000d40860, sys_a7195d8a9d8696e0)` (TODGA and TEHDGA, both in E1 and E2), the eval
  on the same two, the loading check on their series (6 of 10), and every output goes under
  `results/_dryrun/{dmodels,systems,eval,loading}/` (the fitted entries are written as copies
  under `results/_dryrun/systems/`, never into `systems/`). The R1 / R2 rules are applied to the
  subset and every DECISION file carries a DRY RUN banner; none of it is a result.
* `results/dmodels/<id>.json` (schema `gen18.dmodel_fit.1`): `n`, `p_eff`, `slope_status`,
  `intercepts`, `publication_effects`, `se`, `se_hc1`, `jackknife_se`, `split_half` (without
  the per-repeat frame, which is `<id>_split_half.csv`), `interpretable`, `n_points`,
  `n_publications`, `distinct_levels`, `covariance`, `domain`, `flags`, `record_ids_by_metal`,
  `regime`. Its SHA-256 is the `fit_manifest_sha256` of the fitted block (the file is hashed
  before the block is written, so it cannot contain its own hash). `fits_summary.csv` is the
  one-row-per-system table.
* The fitted block (3.8) is written by `as_solvating_params` plus: a `value: null / unknown`
  `log_k` key for every metal of the entry's records without a training intercept (validator
  V8, addendum WB1 A2); the ligand's `ligands_per_metal` set to the fitted `n`; and
  `phase.max_loading_fraction_studied` set from the loading-series records as `measured_corpus`
  (`n_fitted * mM * 1e-3 / ligand_M`, O/A 1 assumed, `safe_exp_ids` of the maximum point) when
  it was null. The written entry is re-loaded through `load_system` (validated) before the
  script continues.
* `g18_eval_dmodels.py` writes `decision.json` with `m1_adopted` (the key WB2's
  `build_system_model(params_source="auto")` reads), the per-system table
  `e1_per_system.csv`, and applies R1 with `ceil(0.6 x n_systems)` wins. X2 (metal-specific
  slopes), X4 (without `TIED_D`), X6 (TODGA n-dodecane versus other aliphatic diluents) are run
  as separate LOPO arms in `lopo.csv` (`arm` column); X3 and X7 are the `mae_B1_crossmetal` and
  `mae_M1_offset` columns; **X8 (Huber loss) is not implemented** — `evalproto` has no Huber
  option — and is counted in `comparisons.csv` with a NaN p-value and the note "counted, not
  run". p-values of paired contrasts are two-sided bootstrap p (same resampling as
  `paired_system_bootstrap`); R1 itself uses the interval, as pre-registered.
* `comparisons.csv` is shared: the eval script writes the full list (E2 rows as placeholders),
  the loading script re-reads it, replaces its own six rows (E2 primary, three secondary, X1,
  X5) and rewrites the table through a fresh `ComparisonCounter`, so the BH adjustment always
  covers the whole exploratory family.
* `g18_loading_check.py`: `n` per E2 system from `results/dmodels/<id>.json` when the fit
  script wrote it, else an in-sample fit computed on the spot, else the prior (`n_source`
  column and `decision.json["n_sources"]`); O/A sensitivity rows share `e2_series.csv` (`oa`
  column, `oa = 1` primary); X5 rebuilds the publication-blind series from `series.csv`'s
  `record_ids`; the Sasaki 2015 TODGA/Nd series is interpreted with the sourced LOC in
  `DECISION.md` (predicted organic Nd against 0.008 M) and never excluded or re-weighted.

## A4. Case study (section 13)

* **SF sweep for Cyanex 272.** "The assumed sweep {1.1, 1.2, 1.3, 1.4} crossed with the draws"
  would quadruple the Cyanex cost (256 000 cascades). Implemented: the sweep values are
  assigned to the draws in turn (draw i gets `sweep[i mod 4]`, 16 draws per value at 64 draws),
  keeping 64 draws per system and the pairing of draw i across the two systems; recorded in
  `draws.csv` (`sf_mode`). `log_k.Pr = log_k.Nd - log10 SF` is clipped into the Pr window when
  the two windows disagree (`log_k_Pr_clipped`; never triggered with the shipped ranges).
* **The same LHS design (same seed) for every draw**, so draws are paired and the interval
  tables measure parameter uncertainty, not LHS noise.
* **Stage bounds and solver budget.** The design's 1-40 stages per section make most LHS rows
  60-110-stage cascades whose failing successive-substitution fallback costs 4-30 s each
  (median evaluation 0.1 s, mean 1.6 s at 1-40 stages on the PC88A entry; the bench's 16 ms is
  the 12-stage reference). Every LHS call of the case therefore passes
  `solve_cascade(max_newton=40, max_sweeps=10)` (`--solver-max-newton`, `--solver-max-sweeps`),
  which bounds a failing candidate to about 1-2 s (mean 0.28 s per evaluation at 1-40 stages,
  0.12 s at 1-20); failures are counted, never raised. `--max-stages` (default 40 as designed)
  sets the upper bound. The measured cost of the dry run and the resulting projection for the
  full 64 x 2 x 1000 case are in A7; the script prints its own projection after draw 0.
* **Consistency check (b).** An 80 + 80 cascade cannot be solved on this machine (WB3 A6: the
  40/40/10 cascade fails within the budget). Implemented as a stage ladder at SF 1.4 (median
  draw otherwise): `n_ext = n_scr = N` for N in {3, 5, 7, 10, 15, 20, 30, 40} (capped by
  `--max-stages`), `n_str = 8`, `max(8, n_lhs // 50)` LHS rows per rung, stopping at the first
  rung that reaches the (0.99, 0.99) cell; the verdict string reports the first rung or "not
  reached up to N = ...". Beside it the analytic Fenske-type minimum at total reflux for a
  constant SF (`fenske_min_stages`, 24.0 theoretical stages at SF 1.4 for the 3:1 feed) is
  quoted as a bound, labelled as not the cascade.
* **Sensitivities** (13.3 "one factor at a time at the median draw") are evaluated at one
  reference regime per system (the consistency cell's knee at the median draw, else the loosest
  cell's knee, else the highest-purity converged regime; `reference_regimes.csv` says which),
  not by a full LHS per factor level (5 levels x 7 factors x 1000 cascades would exceed the
  budget of the whole study). The displacement-scrub variant (scrub Nd 0 / 10 / 25 / 50 mM) and
  the labelled sanity limit are evaluated at the same reference regime.
* **Sanity limit.** "ConstantD at the tracer D of each section" needs a D that depends on the
  section; `SectionConstantD` (a `ConstantD` subclass, case script only) selects the section's
  tracer D by the stage's aqueous [H+] (nearest section inlet acid in log space), with
  `q = p = z = 0` (no depletion, no acid balance, no anion transport).
* **13.5** runs as `--todga-feed cases/todga_prnd_feed.json` (outputs under
  `todga_exploration/`): the TODGA-only Pareto table (D source per `results/eval/decision.json`,
  stage bounds 1-20, strip acid 0.01-0.5 M), then the placeholder complexant with
  `log beta_Nd in {1, 2.5, 4}` x `delta in {0, 0.5, 1, 1.5}`, `log K_H = 2`, in the feed and in
  the scrub only, at the TODGA-only reference regime (largest purity x recovery in the LHS).
* The V9 guard checks the two entries and both case files (`assumed` with `value` and no
  `range` refuses); `--allow-placeholders` is required as designed and stamps every regime line
  with `PLACEHOLDER_PARAMETERS`.

## A5. Scripts 9-12

* `g18_optimize.py`: `--params-source` (default `auto`), `--widen NAME LO HI` (repeatable),
  `--fix NAME VALUE`, `--max-stages`, the solver budget, `--allow-placeholders` for literature
  entries; by default `strip_anion_M`, `feed_dilution`, `f_bleed` are fixed at 0 and
  `scrub_complexant_M` at 0 when the system has no complexant (otherwise every candidate with
  complexant is `invalid_spec`). `--bo` writes `bo.csv` or `bo_gate.txt`.
* `g18_recipes.py`: systems are pre-filtered by the registry's `acid_class` (identical to the
  anion vocabulary), then loaded and built with `params_source="auto"` and
  `feed_metals = (target, impurities...)`; `ModelBuildError` rows are `not_parameterised` with
  the reason. Output directory `results/recipes/<feed stem>/`; `systems_summary.csv`,
  `top_regimes.csv`, `knees.csv` (consistency cell), `SUMMARY.md` grouped by family.
* `g18_screen.py`: an empty `priors.csv` with the note in its regime line when the gen15 model
  is absent; on this machine `generations/gen15_curve/models/deploy_g15.joblib` exists, so the
  script runs one `g15_predict.py` subprocess per SMILES (about 10-20 s each).
* `g18_report.py`: refuses on a seal **mismatch**; writes with an UNSEALED banner when the file
  is not yet sealed (as instructed); `--results-dir results/_dryrun` assembles the dry-run
  report to `results/_dryrun/GEN18_REPORT_dryrun.md` (never to `GEN18_REPORT.md`). Sections
  follow the section 14 outline; missing inputs are reported as "not available".

## A6. Measured facts the orchestrator should know

* Cascade cost on the PC88A placeholder entry (feed 0.1 M metal, O/A 1): 3/3/3 16 ms, 10/10/5
  54 ms, 20/20/8 170 ms, 30/30/8 0.9 s (Newton), 40/40/8 fails after 9.4 s (default budget).
* Without saponification the PC88A placeholder set extracts little at 0.1 M metal (the 3 H+
  per Ln3+ released at pH 2 collapse D); nearly every candidate is `OUT_OF_DOMAIN` against the
  S2-derived literature box (equilibrium pH 1.02-1.42, loading <= 0.2, O/A 1), so the
  `in_domain_only` fronts of the case are usually empty — a property of the placeholder domain,
  reported as such.
* A 54-point hand grid on the PC88A placeholder set (1.5 M formal PC88A, S/A 0.3, W/A 0.5,
  strip 3 M, scrub acid 0.05-0.5 M, O/A 1-3, 10+10+5 and 20+20+5 stages, saponification 0 /
  0.3 / 0.5): unsaponified the best rows give purity 0.84 at recovery 0.56 or purity 0.79 at
  recovery 0.84 (20 + 20 stages, scrub 0.5 M); saponified rows extract the whole feed (purity
  0.75 = the feed ratio, `ALKALI_EXCESS`, inadmissible). Consistency check (a) — the
  (0.97, 0.85) cell — is therefore expected to read "not reachable within the assumed window"
  in the full run unless the LHS finds a scrub regime the grid missed; that is a statement
  about the placeholder parameters (SF 1.3-1.5, ideal exponents, no saponification range),
  not a validation outcome, and the report must say so.
* TODGA `sys_5cb78e5000d40860` in-sample pooled fit (dry run, band 20-30C, 402 aggregated
  points, 27 publications): n = 2.453 (inside [2.36, 2.88]), jackknife SE 0.543 → **not**
  `interpretable` (SE >= 0.5); p_eff = 2.101 interpretable. TEHDGA `sys_a7195d8a9d8696e0`:
  n = 3.024 (SE 0.252), p_eff = 3.329, both interpretable. These are dry-run numbers before
  sealing; the sealed run must reproduce them.
* Loading (prior n = 3, all 10 series, O/A 1): C1 wins 6 of 10 (the three flat pub_d3c970567f
  series and the rising Ce series fall to C0, as pre-registered); on the two-system dry-run
  subset (6 series) C1 wins 3 of 6, bootstrap CI of E2(C0) - E2(C1) [0.002, 0.280].

## A7. Dry-run timing of the case study (printed, never written into results)

`--draws 2 --n-lhs 20 --todga-feed ...` (stage bounds 1-40, solver budget 40 / 10): 0.9 min in
all; draw 0 took 4.4 s (PC88A) and 4.2 s (Cyanex 272) for 20 cascades each, i.e. about 0.22 s
per LHS cascade (3 of 20 failed per draw); the median draw, sensitivities, the stage ladder up to
40 + 40 + 8 and the 13.5 exploration took the rest. At that cost the full 64 x 2 x 1000 case
projects to about 8 h (128 000 x 0.22 s), against the 20-30 min the design expected from a
10 ms reference cascade. Options, all exposed on the command line and none changing the
protocol: `--max-stages 20` (about 0.12 s per cascade, about 4.5 h), a smaller `--n-lhs`, or
running the two systems in two invocations (`--systems`) on different days. The script prints
its own measured per-cascade cost and projection at the end of every run.
