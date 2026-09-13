# Addendum INTEGRATION — the integration pass (2026-09-13, orchestrator)

Ownership rule: during integration the orchestrator may edit any file under
`generations/gen18_process/`. Every edit is logged here with its reason. Nothing outside that
directory was touched; nothing was committed.

## 1. Edits

| file | edit | why |
|---|---|---|
| `tests/test_evalproto.py` | replaced `test_loading_series_evaluate_is_a_stub_for_wb4b` with three real tests: a round-trip against an **independently solved** ideal-depletion series (`scipy.brentq` on `D = K (L_T − n y)^n`, not the code path under test), a prior-fallback test, and a malformed-fit test | WB4a wrote the test against a stub that WB4b was assigned to implement; it asserted `NotImplementedError` and failed by design once the function existed. A deleted test would have left §10.4 untested, so it was replaced by a stronger one: the anchor recovers the generating `log K` to 1e-6 and C1 reproduces the independent solve to < 1e-9, while a flat control series correctly falls to C0 |
| `gen18proc/evalproto.py` (`_loading_n_of`) | a non-mapping `slope_status` now raises `ValueError` naming the expected shape instead of crashing with `AttributeError` | DESIGN §1.5: malformed input raises `ValueError`. `slope_status="fitted"` is a plausible caller mistake and produced an opaque attribute error |
| `scripts/g18_screen.py` | added `resolve_input()`: a relative input path resolves against the working directory, then the generation root, then the repository root, and the error names all three | the script resolved `--smiles-file` against the repository root while `g18_case_prnd.py` resolves `--feed` against the generation root. Both are documented as "run from the repository root", so the same relative path worked in one script and failed in the other |
| `scripts/g18_screen.py` | `#` now starts a comment anywhere on a line, not only at its start | the docstring promises "`#` comments allowed"; a trailing `# name` comment was being parsed as part of the SMILES |
| `scripts/g18_manifest.py` | **new** (script 13): writes and `--check`s `results/MANIFEST.sha256` over every file under `results/` | DESIGN §11 requires the manifest ("lists every output"); each script wrote only its own `manifest.json` and no script aggregated them (WB4b open issue) |
| `cases/screen_candidates.txt` | **new**: the 71 corpus extractants that have both Pr and Nd measured, generated from `systems/corpus_records.csv` | `g18_screen.py` needs a candidate list; this one is reproducible from the database and is the honest pre-screen population for the Nd/Pr pair |
| `gen18proc/dmodel.py` (`NearestConditionD`, new `_tracer_log_d`) | **the looked-up record is lifted to its own tracer limit before the stage depletion is applied** (`tracer_correction=True` by default; `False` restores the original behaviour) | **a defect found by the integrator, see section 4** |
| `tests/test_dmodel.py` | `test_nearest_condition_d` now pins the uncorrected path explicitly (`tracer_correction=False`); `test_build_system_model_nearest_from_records` asserts the corrected value as a formula; **new** `test_nearest_condition_d_tracer_correction` covers both paths, the uncorrectable record, and the round trip | the two tests encoded the behaviour the fix changes; the fix is the design error being corrected, so the tests move with it and the old path stays covered |
| `scripts/g18_reliability_probe.py` | **new**: an exploratory, clearly-labelled probe of whether M1's advantage tracks slope reliability | see section 5 — the hypothesis is **not** supported and the negative result belongs on the record |
| `scripts/g18_case_prnd.py` | `--systems` with no names now runs **only** the §13.5 exploration (and writes `manifest_todga_only.json`) instead of dying in `pd.concat` on an empty list | §13.5 uses a corpus system, so it is the one part of the case affected by the D-source fix of section 4. Without this, regenerating it meant re-running the whole 2.4 h case. It writes `parameters.md` too, so run it into a scratch directory and copy back only `todga_exploration/` |
| `scripts/g18_case_targeted.py` | **new** (exploratory): a structured grid over the same placeholder chemistry, to test whether the case's "no cell reached in 64 draws" is a search artefact | the Fenske minimum for the loosest cell is ~10 stages against 40 + 40 searched, so the empty result needed a direct test rather than an interpretation. Its first version fixed `saponification_degree = 0`, which for a cation-exchange extractant makes the cell unreachable by construction (3 H⁺ released per Ln³⁺ self-acidifies the feed); the grid now scans it |
| `scripts/g18_report.py` | inserts `SUMMARY.md` as section 0 when present; section 11 summarises the pre-screen instead of dumping 71 SMILES rows; section 10 lists the integration defects; the bench projection no longer implies the case costs 36 min | the assembled report is the deliverable: a narrative that survives re-running the assembler, a readable §11, and a projection that does not contradict the measured 252 ms per case cascade |
| `PRE_REGISTRATION.md` §12 | pre-fit addendum (E1 unit clarification, NaN-metal count, unit-slip axis, loading-active count 7, U2 and U6 decisions) | written **before** sealing and before any fit, so that the sealed text and the frozen audit agree. Both `DATA_AUDIT.md` discrepancies are bookkeeping, not data changes |

## 2. Verification performed by the integrator (not only by the owners' own tests)

- **Full fast suite**: 236 passed, 0 failed, 10 deselected (`-m "not slow and not validation"`).
- **Validation-marked tests**: 7 passed, 1 failed — the rising TODGA/Ce series
  (`ls_sys_07ee9637c98c1e20_Ce_pub_917a4583d4_3_0.1`, Spearman +0.238, p = 0.57, n = 8). This is
  the failure `PRE_REGISTRATION.md` §7(c) anticipates: a reported result, not a blocker, and the
  series is not excluded from R2. `test_todga_slope.py` passes (n = 2.453 in [2.36, 2.88]).
- **Determinism**: `g18_build_db.py` re-run → all **294** files under `systems/` are byte-identical
  (SHA-256 compared file by file); `g18_audit.py` re-run → `results/audit/cohort_sha256.txt`
  identical.
- **Independent cascade check** (not one of WB3's tests): against hand arithmetic from the task
  document and the Kremser equation, `D = 2`, lean organic, `f_bleed = 1`:
  fraction unextracted for `n_ext` = 1, 2, 3, 5, 8 matches (E−1)/(E^(n+1)−1) to ≤ 2.2e-9, and the
  single-stage `D(O/A)/(1+D(O/A))` matches at O/A = 0.5, 1, 2 to ≤ 2.5e-9 (so the O/A convention
  is not inverted).
- **U6 end to end**: the sourced LOC (0.008 M Nd, Tachimori 2002) loads into
  `sys_5cb78e5000d40860` with status `literature`, and `THIRD_PHASE_RISK` fires exactly as the
  predicted organic Nd crosses 8.0 mM (6.2 mM → no flag; 8.7 mM → flag), alongside the standing
  `OA_ASSUMED` and `EQUILIBRIUM_ACID_ASSUMED_NOMINAL` flags.
- **gen15 pre-screen**: over the 71 candidates the model returns only **two** distinct
  magnitudes (0.1391 and 0.0334 log units); it is a sign call with an essentially constant
  magnitude, as gen16 concluded. 42 candidates are called Nd-selective, 29 Pr-selective.
- **Two-ligand mechanism** (task document §3): with β(Pr) > β(Nd) an aqueous hold-back ligand
  lifts SF(Nd/Pr) from 2.00 to 2.84 (×1.42) at 0.05 M while D(Nd) moves only 0.126 → 0.120 —
  amplification from the aqueous side, as §3 describes, with the cost falling on D(Pr).
- **Fenske bound, computed independently of the code**: for the (0.99, 0.99) cell at SF 1.4 with
  feed Nd:Pr 3:1, N_min = **24.0** theoretical stages at total reflux, matching what the case
  script reports; for the loosest cell (0.95, 0.80) it is **9.9**.

### 2a. A verdict of the case study that needs qualifying

`results/case_prnd/consistency_checks.md` check (b) states that the stage count "is of the order of
70 + 70 rather than 7 + 7". **The conclusion is right but its stated evidence is not.** The stage
ladder printed beside it shows `best_recovery_at_purity_ge_0.99` falling from 3.7 × 10⁻⁷ at
3 + 3 stages to 5.5 × 10⁻⁸⁵ at 40 + 40 — a geometric collapse, the signature of a scrub section
washing the product back, i.e. the 8 LHS rows per rung found only degenerate designs (purity 1.0
at vanishing recovery). What actually supports the conclusion is the analytic Fenske bound:
14 stages is below the 24-stage total-reflux minimum, so 7 + 7 is impossible at any reflux, while
144 stages is about 6 × N_min, a normal practical multiple. The report says so (`SUMMARY.md`).

The same reading applies to the case's "not reachable" verdicts: with N_min ≈ 10 for the loosest
cell and 40 + 40 stages available, the empty result is a property of a 250-point random sample in
17 dimensions, not of the chemistry. `scripts/g18_case_targeted.py` (exploratory) searches a
structured grid instead and reports whichever way it falls.

## 3. Budget deviation of the Pr/Nd case (recorded, not hidden)

DESIGN §13.2 specifies 64 draws × 1000 LHS per system. Measured cost at stage bounds 1–40 is
**252 ms per cascade** (not the 16.7 ms reference cascade of `results/bench/timing.json`, which is
a 6/3/3 cascade), so the specified run is ~9 h. It was run as **64 draws × 250 LHS** (~2.4 h):
the draw count, which carries the placeholder-parameter intervals that are the headline, is
unchanged; the LHS count, which samples the design space inside each draw, is reduced 4×.
Consequence to state in the report: a coarser design-space sample can only make a spec cell look
*less* reachable, never more, so a "not reachable" verdict is weaker evidence than it would be at
1000 LHS, while a "reachable" verdict is unaffected.

## 4. The defect in the deployed D source, and its correction

**What was wrong.** R1 returned a null, so the chain's D source for every corpus system is B1,
`NearestConditionD`. Its contract (DESIGN §5.1) is: the 1-NN record "supplies the **tracer** log D",
and the stage's own loading is then applied as the ideal depletion term
`log D = log D_nn + n log10(L_f / L_T)`. But the lookup ranks records only by
`(log10 [H+], log10 [L]_T)` — the record's **metal concentration is not an axis and tracer records
are not preferred** — so whenever the nearest record was itself measured under load, its log D
already contains that record's depletion and the term is applied a second time.

**How big.** Of the 4445 fit-eligible corpus records that carry a metal concentration, **1274
(28.7 %) were measured above loading fraction 0.1** and 452 (10.2 %) above 0.3 (ideal `n = 3`,
O/A = 1). Per system: TODGA `sys_5cb78e5000d40860` 23 %, TEHDGA `sys_a7195d8a9d8696e0` 38 %,
**TBDGA `sys_81e3169a7f85c2b9` 81 % with median loading fraction 0.30**. It was found by walking
the acid axis of the deployed model and seeing D(Nd) at 0.1 M TODGA go 1.85 → 37.6 → 3.97 → 0.87
at 0.5 / 1 / 3 / 5 M HNO3: at 1 M the nearest records are tracer points (0.07 mM, log D 1.58,
reproduced by two independent publications), at 3 M they are loaded points (5–12 mM).

**The correction.** `_tracer_log_d` lifts every record by
`log D_tracer = log D_record + n log10(L_T / (L_T − n y))` with `y = T D / (1 + D)` at O/A = 1 —
the same ideal law the pre-registration **supported** in R2, and computable from columns the
database already carries. A record with no metal concentration (24 % of the corpus) or one whose
implied free ligand is non-positive is left as it stands. On the TODGA system 426 of 504 records
are lifted and the tracer D at 3 M / 0.1 M rises 3.97 → 7.52; the loading response stays monotone.

**What it does and does not touch.**
- **R1 and R2 are unaffected.** `evalproto`'s B1 predicts a held-out record's log D by the raw
  1-NN with no depletion term at all, and C1 anchors explicitly at each series' own tracer point.
  Both decisions were made and sealed before this was found and neither is reopened.
- `results/recipes/` **was re-run** with the correction; the uncorrected run was kept for
  comparison outside the repository. The Pr/Nd case study is unaffected (PC88A and Cyanex 272 are
  literature entries with explicit `log_k`, so `params_source` is `literature`, not `nearest`).

**What is still not right, and is reported rather than fixed.** Even corrected, the lookup is not
smooth along the acid axis (1.89 → 37.9 → 7.52 → 1.13 at 0.5 / 1 / 3 / 5 M): it reproduces genuine
between-publication scatter, because a 1-NN has no way to average two publications that disagree.
An optimiser searching over acidity can therefore land on a single flattering record. This is a
property of the D source the null selected, not a bug, and the report says so.

## 5. An exploratory hypothesis that failed

After R1, the obvious question is whether M1 helps exactly where its fitted exponent is reliable —
if so, a reliability gate would rescue it. `scripts/g18_reliability_probe.py` tests it and the
answer is **no**: Spearman(jackknife SE of `n`, advantage of M1 over B1) = **−0.117** (p = 0.69,
n = 14); Mann-Whitney interpretable > not interpretable p = 0.48; systems with an interpretable
slope win 3 of 6, those without win 4 of 8; and the two largest M1 wins (+0.74, +0.63) are both on
systems whose slope is **not** interpretable. The null is not an artefact of unreliable slopes and
a reliability gate is not a route around it. Exploratory, not pre-registered, changes no decision,
and counted in the report's comparison count.

## 6. Not done / left open

- `g18_recipes.py` over all ~280 corpus systems is hours at the default stage bounds and most
  systems lack Pr or Nd records (listed `not_parameterised`); run it scoped when a real feed is
  supplied.
- `config/prices.json` holds order-of-magnitude placeholders with no source (open item U5), so
  `cost_proxy_per_kg_oxide` is NaN + `COST_INCOMPLETE` wherever a price is missing; **consumption
  per kg of oxide is the primary economic number**.
- Open items U1 (feed and spec for the case), U3 (a sourced β set for the hold-back ligand),
  U4 (transcription of the PC88A / Cyanex 272 parameters by a person) and U5 remain open and are
  listed in `GEN18_REPORT.md`.
