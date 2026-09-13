# Addendum WB3 — cascade, metrics, prices, bench (2026-09-13)

Owner: WB3 (cascade). Files: `gen18proc/cascade.py`, `gen18proc/metrics.py`,
`config/prices.json`, `scripts/g18_bench.py`, `tests/test_cascade.py`, `tests/test_metrics.py`,
`results/bench/timing.json` + `manifest.json` (written by the script). Interfaces affected:
DESIGN.md sections 7.1–7.9, 8, 11 (row `g18_bench`), 12.2. Nothing here renames a field of
`types.py` (WB0); `CascadeSpec`, `CascadeResult`, `ProcessMetrics`, `AqStream`, `OrgStream`,
`StageDiagnostics` are imported from `gen18proc.types` unchanged. WB4b (`optimize.py`,
`g18_case_prnd.py`, `g18_optimize.py`, `g18_recipes.py`) codes against A1–A4; A5 lists every
deviation from DESIGN.md with its reason; A6 gives the measured timings and the convergence
envelope the orchestrator should know.

## A1. `cascade.py`

```python
solve_cascade(spec: CascadeSpec, system: SystemModel, *, method: str = "auto",
              tol: float = 1e-11, max_newton: int = 60, max_sweeps: int = 5000) -> CascadeResult
solve_cascade_ss(spec, system, init: np.ndarray | None = None, *, max_sweeps: int = 5000,
                 tol: float = 1e-11, balance_tol: float = 1e-10) -> CascadeResult
kremser_init(spec, system) -> np.ndarray          # layout: per stage [x_i (M), L_k (K), h, nu, c, B]
kremser_fraction_unextracted(E: float, n: int) -> float
check_balances(spec, result, system) -> dict[str, float]      # NOTE the third argument (A5.1)
section_tracer_d(spec, system) -> dict[str, dict[str, float]]  # {"extraction"|"scrub"|"strip": {metal: D}}
alkali_reserve_inlet(spec, system) -> float        # B_in,0 = s * [HA]_T of the acid-releasing ligands
regime_status_of(flags) -> str                      # INADMISSIBLE > OUT_OF_DOMAIN > IN_DOMAIN_WITH_CAVEATS > IN_DOMAIN
build_topology(spec) -> Topology                    # inc, ext_flow, a_flow, upstream, aq_order, section
ORIGIN_LABELS = ("feed", "scrub", "fresh", "strip"); SECTIONS = ("extraction", "scrub", "strip")
```

* **`method`**: `"auto"` (the chain of A5.8), `"newton"` (Newton only; `failed` when it does not
  converge), `"ss"` (successive substitution only from the Kremser init).
* **Validation (`ValueError`, DESIGN 7.1, performed here because `CascadeSpec` has no
  `__post_init__`, WB0 A4)**: `n_ext < 1`, negative `n_scr` / `n_str`, `feed_stage` or
  `scrub_return_stage` outside `[0, n_ext - 1]`, non-positive feed or organic flow, non-positive
  scrub / strip flow when that section exists, `saponification_degree` or `f_bleed` outside
  `[0, 1]`, a negative or non-finite concentration, a metal (in any liquor, the fresh organic or
  `target`) not parameterised by the system, a system ligand missing from `ligand_total` or
  with a non-positive total, a complexant in a liquor of a system without `ComplexantSpec`, and a
  **dry stage**: a stage that no aqueous stream reaches (`feed_stage < n_ext - 1` together with
  `scrub_return_stage < n_ext - 1`, or `n_scr = 0` with `feed_stage < n_ext - 1`) — DESIGN 7.2
  defines `A_j` as the sum of the external flows reaching `j`, and a stage with `A_j = 0` has no
  aqueous phase. The optimiser must catch `ValueError` and record `invalid_spec` (DESIGN 1.5);
  `solve_cascade` never returns that status itself.
* **Metals**: the cascade metals are the system metals that carry a positive external input
  (feed, scrub, strip, or `f_bleed * fresh_organic`), in `system.metals` order; a metal declared
  everywhere at zero is identically zero and is left out of the stream dicts. Every stream of
  the result carries exactly the cascade metals.
* **Streams of the result**: `stages_aq[j]` / `stages_org[j]` are the outlets of stage `j`;
  `raffinate = stages_aq[0]`; `product = stages_aq[n_ext + n_scr]` (strip liquor) or
  `stages_org[N-1]` when `n_str = 0` (then `f_bleed` is forced to 1, DESIGN 7.2);
  `scrub_raffinate = stages_aq[n_ext]` or None; `loaded_organic = stages_org[n_ext + n_scr - 1]`;
  `stripped_organic = stages_org[N-1]`. `AqStream.complexant_total` per stage is fixed by the
  topology (the complexant stays aqueous); `AqStream.sodium` follows the base consumed by the
  reserve. When `spec.target` is set, every stream carries `labels = {target: {label: mol/L}}`.
* **`origin`** (None without a target): `recovery_from_feed`, `recovery_total`,
  `scrub_target_return` (**NaN**, not None, when the scrub carries no target — the mapping is
  `float`-valued; `metrics` turns it into None), `net_product_mol_h`, `product_total_mol_h`,
  `feed_target_mol_h`, `scrub_target_mol_h`, `strip_target_mol_h`, `fresh_target_mol_h`,
  `label_sum_rel_defect`, and `product_<label>_mol_h` / `raffinate_<label>_mol_h` for the four
  labels. Identity (tested to 1e-10): `recovery_total - recovery_from_feed =
  scrub_target_return * S x_S,T / (A x_F,T) + (product_fresh + product_strip) / (A x_F,T)`.
* **`balances`** (relative in-minus-out from the stream table alone): `metal.<m>`,
  `ligand.<k>`, `proton`, `anion`, `complexant`, `sodium` over the whole cascade, plus
  `metal_stage_max`, `proton_stage_max`, `anion_stage_max`, `sodium_stage_max`,
  `complexant_stage_max` (largest per-stage defect); `balance_rel_max` is the maximum of all.
* **`diagnostics[j]`** on the Newton path: `d` / `d_by_ligand` from the converged stage D,
  `loading_fraction`, `iterations` = total Newton iterations, `residual_max` = the stage's
  largest scaled residual, `branch`, `flags` = `model.state_flags` of the solved state per ligand
  + `system.flags` + `ALKALI_EXCESS` on branch 2 (+ `NOT_CONVERGED`, `JACOBIAN_SINGULAR`),
  `status`. On the SS path they are `solve_stage`'s diagnostics of the last sweep.
* **`status`** in `{converged_newton, converged_ss, failed}`; a failed result carries the last
  iterate's streams, its (open) balances, `NOT_CONVERGED` and `regime_status = INADMISSIBLE`.
* The residual and Jacobian machinery (`_Problem`, `_Newton`, `_SSIteration`, `_kremser_u0`,
  `_refine_init`, `origin_pass`) is private but stable; `tests/test_cascade.py` and
  `scripts/g18_bench.py` use it.

## A2. `metrics.py`

```python
compute_metrics(result, spec, system, target, impurities, prices: Prices | None = None, *,
                spec_limits: Mapping[str, float] | None = None,
                feed_acidification_mol_h: float = 0.0, feed_dilution_L_h: float = 0.0) -> ProcessMetrics
Prices.load(path=None) ; Prices.from_json(obj) ; .price(item) ; .unit(item) ; .without(*items) ; .table()
product_moles(stream) ; oxide_mass_per_mol_metal(metal) ; effective_f_bleed(spec)
ATOMIC_MASS_G_MOL ; CONSUMPTION_ITEMS = (acid, base, salting_anion, complexant, extractant_makeup, diluent, water)
ITEM_UNITS = {mol for the first five, L for diluent and water}
```

* `spec_limits = {"purity_min", "recovery_min"}` sets `on_spec` (None when not given) — the
  design's `spec.purity_min` cannot come from the `CascadeSpec` argument, so the product
  specification is a keyword (the name matches `lhs_pareto(..., spec_limits, ...)` of DESIGN 9.2).
* `consumption` keys: `<item>_<unit>_h`, `<item>_<unit>_per_mol_T`, `<item>_<unit>_per_kg_oxide`
  (21 keys). Definitions: `acid = S h_S + W h_W + max(0, feed_acidification_mol_h)`; `base = O *
  alkali_reserve_inlet` (= `O s [HA]_T`, monomer basis of the acid-releasing ligands, 0 for a
  solvating system); `salting_anion = S max(0, anion_S - h_S) + W max(0, anion_W - h_W)` (the
  anion beyond the acid: the design's `salt_S` is not a stream field); `complexant = (A cT_F + S
  cT_S + W cT_W)(1 - regeneration_fraction)` (the design names `S cT_S` only; a feed or strip
  complexant is consumed the same way), NaN + `COST_INCOMPLETE` when the fraction is unknown and
  complexant is used, 0 when none is used; `extractant_makeup = f_bleed O sum_k ligand_scale_k
  L_T^(k) + ligand_loss (A + S + W)` with NaN + `LIGAND_LOSS_NOT_MEASURED` when the loss is null
  and `f_bleed = 0`, and the flag alone (value = the bleed term) when the loss is null and
  `f_bleed > 0` (the value is then a lower bound; the design raises the flag only in the first
  case); `diluent_L_h = f_bleed O` (the bleed volume) and NaN without a bleed (entrainment is not
  measured); `water = S + W + feed_dilution_L_h`.
* `recovery_*` come from `result.origin` when the cascade was solved with the same target, else
  from the same linear pass at the converged D (`origin_pass`); `recovery_from_feed` is NaN
  only when neither is possible (a failed result gives None).
* `enrichment_factor[I]` is `inf` when the product holds no impurity, NaN when the feed ratio
  or the product target is undefined. `sf_tracer` / `sf_by_stage` use `inf` for a zero impurity D.
* `purity_*` are NaN when the product holds no metal. Failed result: every numeric field None /
  NaN, `enrichment_factor` NaN per impurity, `sf_tracer = {}`, `sf_by_stage = ()`,
  `NOT_CONVERGED` in the flags, `regime_status = INADMISSIBLE`; stage counts and flow ratios are
  still filled.
* Atomic masses: `gen13sep.metals` carries none (DESIGN 8 assumed it did), so the IUPAC
  conventional weights are in `metrics.ATOMIC_MASS_G_MOL` (the same table as WB1's
  `ingest.ATOMIC_MASS_G_MOL`, not imported to keep `metrics` free of pyarrow).
* `phase` items come from `system.phase` (else `system.entry.phase`), `Sourced.unknown` when the
  model has none (the `ConstantD` fixtures).
* `regime_status` is recomputed over `result.flags` plus the metric flags (`COST_INCOMPLETE`,
  `LIGAND_LOSS_NOT_MEASURED` are caveats).

## A3. `config/prices.json`

`{"schema": "gen18.prices.1", "currency": "USD", "note": ..., "items": {<item>: {"value",
"unit" ("USD/mol" | "USD/L"), "currency", "status": "assumed", "assumed_label":
"ASSUMED_PLACEHOLDER", "range": [lo, hi], "note"}}}` for the seven `CONSUMPTION_ITEMS`. Every
value is an order-of-magnitude placeholder with no source (open item U5); the cost proxy is
secondary to the consumption and is reported with these ranges. `compute_metrics` requires the
price unit to end in `/mol` or `/L` matching the item, else `COST_INCOMPLETE`.

## A4. `scripts/g18_bench.py` and `results/bench/timing.json`

`[--n 50] [--seed 18]`; reference cascades N = 6/3/3, M = 2, K = 1 on the cation-exchange and
the solvating fixture (84 unknowns); per case: `stage_solve_cold`, `stage_solve_warm`,
`kremser_init`, `init_refine`, `newton_residual_eval`, `newton_jacobian_assembly`,
`cascade_newton`, `cascade_ss`, `lhs_evaluation` (each `{median_ms, p25_ms, p75_ms, min_ms,
n}`), the Newton iteration count, the SS sweep count, the balance; `projection` with the
case-study estimate (64 x 2 x 1000 evaluations) and `within_design_expectation`. The only
results file with wall-clock values; `manifest.json` beside it has none.

## A5. Deviations from DESIGN.md (each with the reason)

1. **`check_balances(spec, result, system)`** takes the system as a third positional argument:
   the ledgers need `q`, `p`, `z` per ligand and metal, which the streams do not carry.
2. **Branch rule of the Newton path (7.5 `active_branch`).** DESIGN: branch from `P_j` versus
   `r_j B_(j-1)` at the current iterate. Implemented: the stage solver's bracket rule
   (addendum WB2 A5.1) — branch 1 iff `P_j(h = H_MIN) - r_j B_(j-1) > H_MIN` with every other
   unknown at its current value. The two coincide at a converged stage; the design's test flips
   scrub and strip stages to branch 2 on the way there (an intermediate iterate with more
   back-extraction than it has protons), which pinned `h = H_MIN` and made every cation-exchange
   case cycle. Branch 2 then uses WB2's `consumed = max(0, P - H_MIN)` reserve residual so that
   the Newton and the SS path solve the same equations (tested to 1e-8).
3. **Per-ligand `p` and `z`** in `released`, the anion row and the ledgers (WB2 A5.6).
4. **Residual rows are written in mol/h** (`A_j (h_j - ...)`, `O (L (1 + KH h nu) + ...)`, ...)
   so that the design's scales (`O L_T`, `max(A h_F, S h_S, W h_W, 1e-3)`, `O B_in,0`, `S cT_S`)
   apply directly; the metal row scale adds the strip and fresh-organic inputs of the metal; the
   complexant row scale adds the feed and strip complexant loads; anion rows are scaled by
   `max(external anion loads, acid scale)`.
5. **Complexant total per stage is not an unknown**: it follows from the topology alone, and a
   stage whose circuit carries no complexant keeps `c = 0` pinned by its own row. The `c -> 0`
   limits of the complexant derivatives (`phi(c)/c -> beta_1`) are taken at `c = 1e-200`.
6. **Initial solve (7.7).** `kremser_init` is the tracer-D linear solve as designed and is exact
   for `ConstantD` (tested). It is a poor start for cation exchange with a scrub loop (tracer D
   of order 700 at the feed acid accumulates metres of metal in the scrub reflux and 4 M of
   released acid), so `solve_cascade` follows it with `_refine_init`: up to 12 rounds of
   constant-D relinearisation (every stage's composite D re-evaluated at its current state,
   `log10 D` moved half-way, the linear metal pass re-solved, the state rebuilt from the
   balances). Exact for `ConstantD` (D never changes). Cost 4 ms on the reference cascade;
   fewer rounds cost more Newton iterations (measured).
7. **Newton variables and globalisation (7.5).** The iteration runs in `w = (x / s_x, ln L, ln h,
   ln nu, ln c, B / s_B)`: `log D` is linear in the logarithms and positivity is automatic; the
   Jacobian columns are the models' `dD / dln(.)` partials times the design's `dF/du` blocks
   (verified against finite differences, rel 1e-6, seven configurations). Lower bounds: 0 for
   the scaled `x` and `B`, `ln H_MIN` for `ln h`. Fraction-to-boundary 0.95 as designed, plus:
   an unknown on its bound whose step points outward is frozen for the step (WB2's rule); a
   *linear* unknown that would allow less than 10 % of the step is landed on its bound and the
   iteration continues (an active-set move; without it the whole step shrank geometrically for
   ten iterations per jammed unknown); a cap of `e^3` per step on the logarithmic unknowns
   (landing `ln h` on `ln H_MIN` multiplied D by 1e10 and destroyed the conditioning). Armijo
   backtracking on `||F||^2` (10 halvings) and the branch-cycle rule (5) as designed.
8. **Fallback chain in `auto` mode (7.5 -> 7.6).** Newton from the refined init; on failure 3
   exact stage sweeps (`solve_stage`, the SS machinery) then Newton; on failure 10 sweeps then
   Newton; then successive substitution continued from those sweeps with a budget of
   `min(max_sweeps, max(100, 20000 // N))` sweeps (a bounded number of stage solves: a full 5000
   sweeps of a 68-stage cascade would take minutes inside an LHS loop). Measured: the sweeps put
   the 12/6/6, 20/10/5 and 30/30/8 cation-exchange cascades inside Newton's basin (8–43
   iterations) where Newton from the refined init alone did not converge.
9. **Dense / sparse threshold** 100 unknowns instead of 600: `numpy.linalg.solve` measured 0.07
   ms at n = 84 but 4.7 ms at n = 168 and 14 ms at n = 476 on this machine, SuperLU 0.4–1.3 ms.
10. **Successive substitution (7.6).** The tear is `(y^(k)_(N-1), a_(N-1))`; `B_(-1)` is
    re-created every pass (DESIGN 7.2) and is not a tear component. The convergence error is the
    larger of the relative change of `x` and of the tear; Wegstein (factor clipped to [-5, 0])
    after sweep 5 and the relaxation rule as designed. `converged_ss` additionally requires every
    stage `converged` and `check_balances` below 1e-10.
11. **Origin pass (7.8)** carries a fourth label `strip` so the label sums equal the totals when
    the strip liquor carries the target; the identity gains `product_strip / (A x_F,T)` (zero in
    every design case). The pass is one shared linear solve with the Kremser init.
12. **Per-stage ledger floors (7.9).** The per-stage `*_stage_max` defects are relative to the
    larger of the stage's own throughput and the cascade-level load of the species; without the
    floor a strip stage holding 1e-40 mol/L of a metal reports roundoff as a 79 % defect while
    every whole-cascade ledger closes to 1e-15.
13. **`section_tracer_d`** evaluates the tracer state with `c = cT / alpha_H(h)` (WB2's
    `tracer_state`), not `c = cT`.
14. **`metrics`**: the keyword arguments and consumption definitions of A2; `on_spec` from
    `spec_limits`; the local atomic-mass table.
15. **Timing (11.3).** Reference cascade 16.5 ms (Newton, 11 iterations) against the design's
    5–15 ms expectation: 4 ms init refinement, 0.2 ms per residual evaluation, 0.25 ms per
    Jacobian assembly, 0.07 ms per 84 x 84 solve, ~2.5 ms result assembly (stream table,
    diagnostics with `state_flags`, ledgers, origin pass). The solvating reference takes 13 ms
    (7 iterations). Recorded in `results/bench/timing.json`; the case-study projection is 36 min
    for 128 000 evaluations (DESIGN 11.3 expected 20–30).

## A6. Convergence envelope measured on the fixtures (for the orchestrator and WB4b)

Newton from the refined init converges on every 3/2/2 to 20/10/5 configuration tried (cation
exchange with and without complexant in the scrub, displacement scrub, saponification 0.3,
feed-stage and return-stage variants, solvating with HNO3 uptake and with complexant): 4–25
iterations, 6–90 ms. The 30/30/8 cation-exchange cascade converges through the sweep fallback
(0.9 s); the 40/40/10 one (90 stages) fails within the budget (9 s, `failed`, counted). Heavy
saponification (`s = 0.6`, every extraction stage on branch 2, `ALKALI_EXCESS`) converges only
by successive substitution (2.5 s) and is inadmissible anyway. WB4b should expect `failed` rows
in the LHS at the large-stage-count corner of the design space and count them (DESIGN 9.2).
