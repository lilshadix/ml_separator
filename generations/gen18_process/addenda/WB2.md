# Addendum WB2 — distribution models, domain, stage equilibrium (2026-09-13)

Owner: WB2 (models). Files: `gen18proc/dmodel.py`, `gen18proc/domain.py`,
`gen18proc/equilibrium.py`, `gen18proc/testsystems.py`, `tests/test_dmodel.py`,
`tests/test_domain.py`, `tests/test_equilibrium.py`. Interfaces affected: DESIGN.md sections
5.1–5.7, 6.2–6.4, 12.1. Nothing here renames a field of `types.py` (WB0); every dataclass is
imported from `gen18proc.types`. WB3 (`cascade.py`, `metrics.py`) and WB4 (`evalproto.py`,
`g18_loading_check.py`, the case study) code against this file. Sections A1–A4 give the API as
it exists; A5 lists every deviation from DESIGN.md with its reason.

## A1. `equilibrium.py`

```python
solve_stage(aq_in: AqStream, org_in: OrgStream, system: SystemModel, *,
            tol: float = 1e-12, max_iter: int = 200, warm=None,
            method: str = "auto") -> tuple[AqStream, OrgStream, StageDiagnostics]
stage_state(aq: AqStream, org: OrgStream, system: SystemModel, *,
            c_free: float | None = None) -> StageState
free_complexant(system: SystemModel, x_total: Mapping[str, float], h: float,
                complexant_total: float, *, tol: float = 1e-14) -> float
proton_ledger(aq: AqStream, org: OrgStream, system: SystemModel) -> float
H_MIN = 1e-5            # mol/L, the fixed [H+] of branch 2 (DESIGN 6.2)
NU_MIN = 1e-9           # mol/L, lower bracket end of the aqueous anion (DESIGN 6.3)
FLOOR_REL = 1e-30       # lower bracket end of L_f and c relative to their totals (A5.4)
BRACKET_RESIDUAL_TOL = 1e-9   # largest scaled residual the fallback may report as converged (A5.5)
```
`LOG10_CAP = 300` (D is capped at 1e300 inside the solver so `10**log_d` stays finite) lives in
`dmodel.py`.

* **Inputs.** `r = org_in.flow_L_h / aq_in.flow_L_h`. The stage metals are the keys of
  `aq_in.metals` followed by the extra keys of `org_in.metals`; a metal absent from one stream is
  0 there. `org_in.ligand_total` must hold every ligand of `system.dmodels` (dimer basis for
  dimers); `org_in.ligand_free` is ignored on input (the solve determines it). With one ligand
  `org_in.metals_by_ligand` is ignored (`metals` is the ligand's load); with several it must sum
  to `org_in.metals` (rel 1e-9). `aq_in.complexant_total > 0` requires `system.complexant_model`.
* **`ValueError`** (malformed input only, DESIGN 1.5): non-positive or non-finite flow, negative
  or non-finite concentration, a metal not parameterised for every ligand, a missing or
  non-positive ligand total, a complexant without a model, a complexant lacking `log_beta` for a
  stage metal, unknown `method`, `tol <= 0`, `max_iter < 1`, an unusable `warm`.
* **Never raises otherwise.** `StageDiagnostics.status` is `"converged"` or `"failed"`; a failed
  stage carries `NOT_CONVERGED` (and `LOADING_CAP_HIT` when the cause is a ligand without a root
  in its bracket, A5.5). Per-metal closure `x_i + r y_i = T_i` holds by construction on every
  return, converged or not.
* **`method`**: `"auto"` (Newton, then the bracketed fallback when Newton does not converge),
  `"newton"`, `"bracket"`. The two paths agree to better than 1e-12 relative on the fixtures
  (test row: 1e-10).
* **`warm`**: a `StageDiagnostics` (free ligand from its `loading_fraction`), a `StageState`, the
  `(aq_out, org_out, diag)` triple of a previous call, an `AqStream` and/or `OrgStream`, or a
  mapping with any of `ligand_free` (per ligand), `h`, `anion`, `c_free`; a tuple/list of those.
  Anything else is a `ValueError`. The 8-start test uses `{"ligand_free": {k: f * L_T}}`.
* **Outputs** (DESIGN 6.2): `aq_out = AqStream(aq_in.flow, x, h, nu, aq_in.complexant_total,
  Na_out)` — the complexant stays aqueous, `labels` is None; `org_out = OrgStream(org_in.flow, y,
  y_by_ligand, L_T, L_f, {k: KH_k h nu L_f^(k)}, B_out)`; `StageDiagnostics(d, d_by_ligand,
  loading_fraction, iterations, residual_max, branch, flags, ood_distance, status)`.
  `iterations` = Newton iterations when the primary path converged, else the total number of
  residual evaluations (fallback). `residual_max` is the largest scaled residual (rows scaled by
  `L_T^(k)`, `max(h_in, h_hi, 1e-3)`, `max(nu_in, 1e-3)`, `cT`). `ood_distance` is the maximum
  over the stage metals per axis (A2); `flags` is the union over ligands and metals of the
  section 5.4 / 5.5 flags plus `system.flags`, `ALKALI_EXCESS` on branch 2, `NOT_CONVERGED` on
  failure, `JACOBIAN_SINGULAR` when a Newton step needed least squares.
* **The two-branch rule as implemented** (A5.1 for the difference to DESIGN 6.2):
  ```
  P  = h_in + r (released - uptake);  Bp = r B_in
  branch 1 (P - Bp >= H_MIN):  h = P - Bp;  B_out = 0;  Na_out = Na_in + Bp
  branch 2 (P - Bp <  H_MIN):  h = H_MIN;  consumed = max(0, P - H_MIN);
                               B_out = (Bp - consumed) / r;  Na_out = Na_in + consumed;  ALKALI_EXCESS
  ```
  On both branches the proton ledger `Q = A h + O (sum_k a_k - B - sum_i p_i y_i)` and the sodium
  ledger `A Na + O B` close exactly (rel 1e-12, tested) whenever `P >= H_MIN`.
* **`free_complexant`** recovers the solved free deprotonated complexant of a stage from its
  output pair (root of residual (4) at `aq_out.metals`, `aq_out.h`, `aq_out.complexant_total`;
  bisection-safeguarded Newton, monotone by DESIGN 6.3). `StageDiagnostics` has no complexant
  field and `types.py` is not WB2's, so this is the only way to read `c`; `stage_state(aq, org,
  system)` uses it, so the state it builds reproduces the solved D of every metal (tested to
  1e-10). WB3's cascade carries `c_j` as an unknown of its own (DESIGN 7.3) and needs only
  `system.complexant_model` (A2).
* **`proton_ledger`** is the `Q` of DESIGN 7.9 for one stream pair with the per-ligand `p`
  applied to `metals_by_ligand` (`metals` when the system has one ligand).

## A2. `dmodel.py`

Names: `LN10, LOG10_CAP, ASSUMPTIONS, TEMPERATURE_BANDS, ANION_ACID_TOL, ConstantD,
CationExchangeMassAction, SolvatingMassAction, NearestConditionD, AqueousComplexantWrapper,
ComplexantModel, EffectiveCapacity, SystemModel, build_system_model, evaluate_composite,
tracer_state, loading_fractions, chemistry_flags, ligand_loading_fraction,
ligand_chemistry_flags, records_to_frame, select_band, parse_band, pow10`.

* **Constructors** (DESIGN 5.1 signatures, keyword extras after `*`):
  `ConstantD(log_d_by_metal, *, q=0.0, p=0.0, z=0.0, ligand="constant",
  mechanism=Mechanism.SOLVATING, domain=None, phase=None)` — `q/p/z` scalars or per-metal
  mappings; `log_d` may be `-inf` (D = 0) or large (D -> inf); provenance `assumed`,
  `ASSUMED_PLACEHOLDER`.
  `CationExchangeMassAction(params, ligand, medium, domain, *, phase=None, base_flags=())` and
  `SolvatingMassAction(...)` — `ligand` is a `LigandSpec` (its name, `aggregation` and
  stoichiometry are used) or a plain name (ideal stoichiometry: q = 3, p = 3, z = 0 for cation
  exchange; q = n, p = 0, z = 3 for solvating). A `log_k` entry with `value None` excludes that
  metal from `model.metals`; a missing `a_dimer` / `b_proton` / `n_solvation` / `p_anion` is a
  `ValueError` at construction (`build_system_model` turns it into `ModelBuildError`).
  `NearestConditionD(records, ligand, domain, n_prior=3.0, *, mechanism=Mechanism.SOLVATING,
  p=None, z=None, phase=None, base_flags=())` — `records` in the `corpus_records.csv` layout
  (A3); rows with `fit_eligible` false, non-positive acid or ligand, or non-finite `log_d` are
  dropped; provenance `measured_corpus`; base flags `EQUILIBRIUM_ACID_ASSUMED_NOMINAL`,
  `OA_ASSUMED`; `q = n_prior` for every metal.
  `AqueousComplexantWrapper(inner, complexant)` — `complexant` a `ComplexantSpec` or a
  `ComplexantModel`; `metals` = the inner metals the complexant covers; `.inner`,
  `.complexant_model`.
  `EffectiveCapacity(phi)` — `phi` in (0, 1] else `ValueError`.
* **Every model** (`_ModelBase`) exposes, beyond the `DModel` protocol:
  `core(L, h, nu, c, ligand_total, capacity=None) -> (log10 D, dlogd_dlnL, dlogd_dlnh,
  dlogd_dlnanion, dlogd_dlnc)` as arrays aligned with `model.metals` (the vectorised evaluation
  the stage solver uses; `capacity` is `phi * L_T` under `EffectiveCapacity`, else `L_T`);
  `stage_arrays(metals) -> (q, p, z, index)` arrays aligned with a stage's metal tuple (`index`
  None when it equals `model.metals`); `state_flags(state, activity=None, *, metal=None) ->
  (flags, ood_distance)` (metal-independent except the `nn` axis, see below);
  `k_acid_uptake` (float, 0 when `K_H` is null), `ligand_scale` (2 for a dimeric ligand: the
  domain and the 1-NN see the formal monomer concentration `2 L_T`), `base_flags`, `phase`,
  `params` (mass-action models), `is_fitted`. A foreign `DModel` (protocol only, no `core`) is
  accepted by the solver through one `evaluate` per metal (tested); it is slower.
* **`DEval` partials** are `d log10 D / d ln(variable)` (DESIGN 5.2): `a / ln10`, `-b / ln10`
  (cation exchange); `n / ln10`, `p_h / ln10`, `p_anion / ln10` (solvating); `-phi_i / ln10` for
  the complexant; `n_prior / ln10` and 0 otherwise for `NearestConditionD`; all verified against
  central differences at rel 1e-6 (observed 1e-11). `d` is `pow10(log_d)` (0 for `-inf`, `inf`
  above 1e308).
* **`ComplexantModel(spec)`**: `.metals` (metals whose every `log_beta` value is known; a
  placeholder with `value None` excludes the metal), `.beta_matrix(metals)` (linear cumulative
  betas, M^-m), `.terms(c, metals) -> (alpha_i, phi_i, dphi_i/dlnc)` arrays, `.alpha_h(h)`,
  `.dalpha_h_dlnh(h)`, `.regeneration_fraction` (float or None), `.spec`, `.name`. This is what
  WB3 needs for residual (4) and its Jacobian.
* **`SystemModel`** (frozen): `entry, dmodels, complexant, activity, temperature_C, assumptions`
  (DESIGN 5.7) plus `complexant_model, phase, flags` (system-level flags added to every stage:
  `MIXED_ORGANIC_UNMODELLED`), `params_source` (`"literature" | "fitted" | "nearest" |
  "constant"`, `|`-joined when ligands differ), `bands` (ligand -> band used); properties
  `ligands`, `metals` (metals every ligand model can evaluate, order of the first), `system_id`.
* **Module functions**: `ligand_loading_fraction(state, ligand, q)`, `ligand_chemistry_flags(
  state, ligand, q, phase)` (DESIGN 5.4 exactly), `loading_fractions(state, system)`,
  `chemistry_flags(state, system)`, `evaluate_composite(system, state, metals=None) ->
  {metal: DEval}` (`D = sum_k D^(k)`, union of flags, max distance per axis, D-weighted
  partials), `tracer_state(system, ligand_total, h, anion, complexant_total=0.0, *, v_aq_L=1.0,
  v_org_L=1.0, metals=None)` (zero loading, `L_f = L_T` or `phi L_T`, `c = cT / alpha_H(h)`),
  `records_to_frame(records)` (flat frame of `DistributionRecord`s: `ligand_M__<ligand>`,
  `metals_initial_mM__<metal>`, `metal_initial_mM`), `select_band(bands, T) -> (band, inside)`,
  `parse_band`.
* **`build_system_model(entry, *, feed_anion, temperature_C, params_source="auto",
  complexant=None, activity=None, parameter_draw=None, feed_metals=None, decision_adopted=None,
  n_prior=3.0) -> SystemModel | ModelBuildError`**. Refusals (`ModelBuildError`): `feed_anion !=
  entry.medium.anion`; no extractant-role ligand with a mechanism; no parameter set for the
  requested source (`"fitted"` needs `fitted_from_corpus` log K); a parameter block whose
  `medium_anion` differs from the medium or whose type does not match the ligand's mechanism; a
  missing `log_k` (value None or absent) for a feed metal (`feed_metals`, default: every metal
  with a log K in the first ligand); a complexant lacking `log_beta` for a feed metal; no
  records for `"nearest"`; more than one `aqueous_complexants` entry when `complexant` is None.
  `ValueError`: unknown `params_source`; a `parameter_draw` key that matches nothing, names a
  parameter without a declared range, or lies outside the range. `"auto"`: corpus entry with a
  fitted block -> fitted if `decision_adopted` (or, when None, `results/eval/decision.json` key
  `m1_adopted`, False when absent) else nearest; corpus without fitted block -> nearest;
  literature -> literature; other origins -> literature if a block exists else nearest. Draw keys:
  `"<ligand>.log_k.<metal>"`, `"<ligand>.<a_dimer|b_proton|n_solvation|p_anion|p_h|k_acid_uptake>"`,
  `"<complexant>.log_beta.<metal>[.<m>]"`, `"<complexant>.protonation_logk.<j>"`,
  `"<complexant>.regeneration_fraction"`; the drawn `Sourced` keeps its provenance and range
  with the draw appended to its note. Parameter blocks may be the dataclasses or the JSON shape of
  DESIGN 3.8 (`model_type` selects the dataclass). **WB4 must write `results/eval/decision.json`
  with `m1_adopted` (bool) from R1, or pass `decision_adopted` explicitly.**
* **Chemistry flags** (DESIGN 5.4 verbatim) plus: `ACID_UPTAKE_UNMODELLED` when `K_H` is null and
  `h > 1`; `ANION_ACID_CONFOUNDED` when the log K are `fitted_from_corpus` and `|anion - h| >
  ANION_ACID_TOL (5 %) * max(anion, h)` (A5.9); `MEDIUM_STRENGTH_UNMODELLED` for cation exchange
  without `beta_anion` when the stage anion lies more than a factor 2 outside the domain's acid
  interval (A5.8); `EQUILIBRIUM_ACID_ASSUMED_NOMINAL` as a base flag of fitted sets;
  `OOD_TEMPERATURE` as a base flag when the design temperature is outside every band.
  `beta_anion` values are **linear** cumulative betas (M^-j), as DESIGN 5.2 (d) writes them.

## A3. `domain.py`

```python
build_domain(records: pd.DataFrame, ligand: str, *, anion=None, diluent_family=None,
             modifiers=None, band=None, n_prior=3.0) -> ApplicabilityDomain
domain_flags(domain, state, ligand, *, anion=None, diluent_family=None, modifiers=None,
             ligand_scale=1.0, loading=None) -> tuple[frozenset[Flag], dict[str, float]]
coerce_domain(obj) -> ApplicabilityDomain | None      # dataclass or the JSON block of 3.7 (keys "_*" ignored)
hull_of_points(points) ; hull_distance(vertices, point) ; interval_distance(value, interval)
AXIS_FLAGS = {"acid": OOD_ACID, "ligand": OOD_LIGAND, "metal": OOD_METAL, "loading": OOD_LOADING,
              "anion": OOD_ANION, "complexant": OOD_COMPLEXANT, "oa": OOD_OA,
              "temperature": OOD_TEMPERATURE, "diluent": OOD_DILUENT, "modifier": OOD_MODIFIER,
              "hull": OOD_HULL} ;  HULL_TOL = 1e-12
```
* **Records layout** accepted by `build_domain` and `NearestConditionD` (resolved per column):
  ligand concentration from `ligand_M__<ligand>`, `ligand_M.<ligand>`, `ligand_M_<ligand>`, a
  dict / JSON-string `ligand_M` column, or WB1's single numeric `ligand_M` column (rows whose
  `ligand_name` differs from `ligand` are masked to NaN); acid from `acid_nominal_M` (or
  `cond__acid_concentration_M`); metal from `metal_initial_mM`, `metals_initial_mM.<metal>` /
  `metals_initial_mM__<metal>` (summed), or a dict / JSON `metals_initial_mM`; temperature,
  `publication_id`, `complexant_M`, `oa_ratio`, and the categorical `anion` / `acid_class`,
  `diluent_family`, `modifiers` (`|`-separated) columns when present; `fit_eligible` as bool or
  the strings `"True"/"False"`. Both WB1's `corpus_records.csv` and WB4a's flattened frames work.
* **`build_domain`**: box on log10 acid, log10 ligand(s) (other `ligand_M__*` columns give their
  own intervals), log10 total metal mM (rows with a metal concentration), loading fraction
  `n_prior * mM * 1e-3 / ligand_M` (O/A 1 assumed), log10 complexant, O/A, temperature; the
  categorical fields from the keywords, else the columns' mode, else `"unknown"` / `()`;
  `band` filters on the temperature column with NaN counted as 20-30C; `n_publications` counts
  distinct `publication_id`; hull via `ConvexHull(QJ)` with the **original** coordinates of the
  vertex points (QJ joggles internally only), counter-clockwise; < 3 distinct or collinear points
  -> the two extreme points (segment), one point -> that point. Addendum WB1 A6: the ingest
  computes the same box in `ingest.build_domains`; the entry's stored domain is the frozen one.
* **`domain_flags`**: per-axis distance outside the interval (log10 units on the concentration
  and O/A axes, degC on temperature, a plain fraction on loading, 1.0 for a categorical
  mismatch, Euclidean log10 distance to the hull polygon / segment / point); a distance
  above `HULL_TOL` raises the axis flag. The metal axis uses `sum x + r sum y` (total per litre
  aqueous, mM); the loading axis uses `loading` when the caller passes it (the models pass
  `ligand_loading_fraction`), else `1 - L_f / L_T`; the hull is tested only for the domain's
  primary ligand (first key of `log_ligand`); axes whose interval is None are absent from the
  dict, except that free complexant against a domain **without** a complexant interval is OOD
  at distance 1.0 (A5.10); the categorical axes are checked only when the keyword is given
  (A5.11); `ligand_scale` converts the state's dimer total to the formal concentration.

## A4. `testsystems.py`

Every number is a synthetic fixture (`assumed`, `ASSUMED_PLACEHOLDER`, note "synthetic test
fixture"). The DESIGN 12.1 functions return **`SystemModel`s** built through
`build_system_model(params_source="literature")`; the entries behind them are exposed too.

```python
two_metal_cation_exchange(*, complexant=None, activity=None, log_k=None, temperature_C=25.0,
                          parameter_draw=None) -> SystemModel   # PC88A-like, chloride, dimer 0.4
two_metal_solvating(*, complexant=None, activity=None, log_k=None, k_acid_uptake=0.5,
                    temperature_C=25.0, parameter_draw=None) -> SystemModel  # TODGA-like, nitrate
constant_d_system(log_d, *, q=0.0, p=0.0, z=0.0, temperature_C=25.0) -> SystemModel
complexant_spec(*, log_beta=None, protonation_logk=(2.0,), concentration=0.05) -> ComplexantSpec
feed_prnd(*, flow_L_h=1.0, complexant_total=0.0) -> AqStream          # 12.1 exactly
feed_prnd_nitrate(*, flow_L_h=1.0, complexant_total=0.0, acid_M=3.0) -> AqStream
lean_organic(system, *, flow_L_h=1.0, ligand_total=None, alkali_reserve=0.0, metals=None) -> OrgStream
cation_exchange_entry(log_k=None, *, a=3.0, b=3.0) -> SystemEntry
solvating_entry(log_k=None, *, n=2.7, p_anion=2.0, p_h=0.0, k_acid_uptake=0.5) -> SystemEntry
assumed(value, unit="1", rng=None, *, note=...) -> Sourced
CE_LIGAND = "PC88A"; CE_HA_TOTAL = 0.8; CE_LT_DIMER = 0.4; SOLV_LIGAND = "TODGA"; SOLV_LT = 0.1
CONSTANT_LIGAND = "constant"
```
`lean_organic` fills the fixture's ligand total (`CE_LT_DIMER`, `SOLV_LT`, 1.0 for a constant-D
ligand) and no metal unless `metals` is given (attributed to the first ligand). The domain box
of both entries is the wide box of 12.1 (log acid [-3, 1], log formal ligand [-2, 0.5]; no
metal, loading, complexant, O/A or temperature interval, no hull). Note for WB3/WB4: with the
fixture's `K_H = 0.5` at 3 M HNO3 the HNO3·L species holds about 80 % of the TODGA-like ligand
and D is about 0.01; `two_metal_solvating(k_acid_uptake=None)` is the corpus setting (`K_H`
null, `ACID_UPTAKE_UNMODELLED` above 1 M) and gives D of order 1 at 0.1 M ligand.

## A5. Deviations from DESIGN.md (each with the reason)

1. **Branch 2 of the acid rule (6.2).** DESIGN: `B_out = (Bp - P) / r`, `Na_out = Na_in + P`.
   Implemented: `consumed = max(0, P - H_MIN)`, `B_out = (Bp - consumed) / r`, `Na_out = Na_in +
   consumed`. The design's form leaves the proton and sodium ledgers open by `A * H_MIN` per
   stage (the protons that remain as `h = H_MIN` are counted twice), which cannot meet the
   12.2 row "proton ledger closes, rel 1e-12"; the implemented form closes both exactly whenever
   `P >= H_MIN` and differs from the design's by `O(H_MIN)`. The branch condition is the design's
   (`P - Bp >= H_MIN` <=> branch 1). Branch 2 is inadmissible either way (`ALKALI_EXCESS`).
2. **Acid bracket (6.3).** DESIGN: `[H_MIN, h_in + r sum_i p_i T_i + r max(0, -uptake_min)]`.
   Implemented: `[H_MIN, h_in + sum_i p_max,i T_i + r sum_k a_in,k]` — `T_i` is per litre of
   aqueous and the organic can hold at most `T_i / r`, so the released acid per litre aqueous is
   at most `sum p_i T_i` (no factor `r`; the design's bound is not an upper bound for `r < 1`),
   and the largest possible return of HNO3·L is `r sum_k a_in,k`.
3. **Fraction to boundary (6.4).** DESIGN: factor 0.95 then clamp. Implemented in the ln
   variables with factor 1: every bracket end is a regular point of the residuals (`L = L_T`,
   `L = FLOOR_REL L_T`, `h = H_MIN`), so the limiting unknown lands exactly on its end and the
   clamp is a no-op; an unknown sitting on an end whose Newton step points outward is frozen for
   that step (its row is replaced by the identity) and the step is re-solved. With 0.95 the
   iteration could only approach an active bound geometrically.
4. **Lower bracket ends.** `L_f` and `c` have the lower end `FLOOR_REL * total` (1e-30) instead
   of 0 because the unknowns are logarithms; `LOADING_CAP_HIT` fires at `1e-6 L_T`, far above.
   `nu` has `NU_MIN = 1e-9` as designed.
5. **Fallback status.** The nested Illinois reports `converged` only when its final scaled
   residual is `<= BRACKET_RESIDUAL_TOL = 1e-9` (it locates every unknown to 1e-13 relative,
   so a larger residual means a ligand level had no root inside `[0, L_T]`: a `ConstantD` ligand
   with `q > 0` loaded past `L_T / q`). That stage is `failed` + `NOT_CONVERGED` +
   `LOADING_CAP_HIT` rather than a "converged" stage with residual 2 (tested; mass-action models
   always have a root because D -> 0 as L -> 0).
6. **Per-ligand `p` and `z`.** Residuals (2) and (3) use `released = sum_k sum_i p_i^(k) (y_i^(k)
   - y_in,i^(k))` and `sum_k sum_i z_i^(k) (...)`, which equal the design's per-metal sums when
   every ligand shares the stoichiometry (one mechanism) and are the correct generalisation for a
   cation-exchange + solvating pair (tested: only the cation-exchange ligand releases protons).
7. **Scale of the acid row.** DESIGN 6.2 states no scale for residual (2); implemented
   `max(h_in, h_hi, 1e-3)` with `h_hi` the bracket's upper end.
8. **`MEDIUM_STRENGTH_UNMODELLED` (5.2 d).** The design refers to "the domain's anion interval",
   but `ApplicabilityDomain` has only the categorical `anion`; the stage anion is compared with
   the domain's `log_acid` interval (the anion equals the acid in the HCl / HNO3 records without
   salting agent), factor 2 as designed.
9. **`ANION_ACID_CONFOUNDED` tolerance.** The flag fires when `[anion] != [H+]` by more than 5 %
   of the larger (`ANION_ACID_TOL`), not at any difference: metal transport removes `3 y` of
   anion per litre and would otherwise trip the flag in every loaded stage of an HNO3-only
   parameter set, which the design reserves for salting agents and strip liquors with added
   nitrate.
10. **`OOD_COMPLEXANT` against a domain without a complexant interval.** A stage with free
    complexant against a parameter set fitted without one is out of domain at distance 1.0 (the
    design lists the axis but not this case); with `c_free = 0` the axis is absent.
11. **Categorical OOD axes.** `OOD_ANION`, `OOD_DILUENT`, `OOD_MODIFIER` fire only when the
    caller passes `anion` / `diluent_family` / `modifiers` to `domain_flags`; a `StageState`
    carries no categorical field, so the stage solver never raises them (the medium match is
    enforced by `build_system_model`'s cross-anion refusal and validator V7). WB3/WB4: call
    `domain_flags(model.domain, state, ligand, diluent_family=..., modifiers=...)` at the cascade
    level when the design's diluent or modifiers differ from the entry's.
12. **`NearestConditionD.ood_distance["nn"]`.** `evaluate(metal, ...)` reports that metal's
    1-NN distance; the stage diagnostics carry the maximum over the stage's metals (not over
    every metal in the records). The 1-NN is taken at `(log10 h, log10 (ligand_scale * L_T))`
    — the recorded formal concentration, not the depleted free ligand, which enters only through
    the depletion term.
13. **`EffectiveCapacity` (5.6).** Implemented as the extra method `effective_ligand_total(L_T)
    = phi * L_T`, used by the stage ligand balance (`L_f (1 + KH h nu) + sum q y = phi L_T`) and
    by `tracer_state`; `organic_free_ligand` returns `state.ligand_free[ligand]` and
    `aqueous_gamma` is unity, as the protocol says. `org_out.ligand_total` stays `L_T` and the
    loading fraction stays `(sum q y + a) / L_T`.
14. **`build_system_model` extras** (`feed_metals`, `decision_adopted`, `n_prior`;
    `params_source` refusals; JSON parameter blocks; `results/eval/decision.json`) as in A2. A
    `SystemEntry` of origin `hypothetical` (the fixtures) uses the literature block.
15. **`solve_stage(method=...)`**, the accepted `warm` types, `iterations` semantics, and the
    `stage_state` / `free_complexant` / `proton_ledger` helpers are additions (A1).
16. **`SystemModel`** carries `complexant_model, phase, flags, params_source, bands` beyond
    DESIGN 5.7 (A2); the D models carry `core`, `stage_arrays`, `state_flags`, `k_acid_uptake`,
    `ligand_scale` beyond the `DModel` protocol.
17. **12.2 loading row on the solvating fixture.** The row "1 mM vs 50 mM at `L_T = 0.1`, both
    mechanisms, `log D(50) < log D(1) - 0.05`" is asserted with `two_metal_solvating(
    k_acid_uptake=None)` (the corpus / C1 setting); with the fixture's `K_H = 0.5` at 3 M HNO3 the
    ligand is 80 % HNO3·L and D is about 0.01, so 50 mM loads it by less than 0.05 log units —
    that case is asserted as monotone only (A4).
18. **Fallback cost.** The nested bracket is exact but its cost is the product of the per-level
    iteration counts (four levels with a complexant); it is the fallback and the cross-check,
    never the path of the cascade's inner loop. Timing is reported to the orchestrator only
    (never written to a results file, DESIGN 1.5).

## A6. What WB3 needs from here (checklist)

* Per stage: `solve_stage` (successive-substitution fallback of 7.6) with `warm=diag_prev`.
* For the full-cascade Jacobian (7.5): `model.core(L, h, nu, c, L_T, L_T_eff)` per ligand for the
  D vector and its ln-partials, `model.stage_arrays(metals)` for `q, p, z`, `model.k_acid_uptake`
  for `a_j^(k) = KH h nu L_f`, `system.complexant_model.terms / alpha_h / dalpha_h_dlnh` for the
  complexant row, `H_MIN` for branch 2 and the branch rule of A1 (with `consumed`).
* For flags and OOD distances of a converged stage: `stage_state(aq_j, org_j, system)` then
  `model.state_flags(state)` per ligand (or `chemistry_flags(state, system)` +
  `domain_flags`), `loading_fractions(state, system)`; `system.flags` is added to every stage.
* For the Kremser oracle: `ConstantD` / `testsystems.constant_d_system`, `tracer_state` for the
  section's tracer D, `evaluate_composite`.
* Ledgers of 7.9: `proton_ledger`; the anion ledger `A nu + O (sum_i z_i y_i + sum_k a_k)`
  uses `model.z` per ligand; the sodium ledger `A Na + O B`.
