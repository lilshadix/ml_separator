# Gen18 brief — the process-design chain (written 2026-09-13, before any code)

This file is the single source of truth handed to every agent working on gen18. It records what
was established by scouting the repository on 2026-09-13, what the task document asks for, and the
constraints every deliverable must satisfy. Read it completely before doing anything.

## 0. The task, in one paragraph

`docs/kshot_next_steps_20260913.md` (Russian; committed as "new task", cdfcebe) replaces the old
k-shot plan. It says the programme's end result is **a cleaner lanthanide product at high recovery
and lower cost**, not a lower ML error. Its software section (§8, §9 "Для программной части")
asks for one thing from this repository: build the chain

    extraction-system composition  ->  D of each metal at the given loading  ->  cascade
    (extraction + scrub + strip + recycle)  ->  purity, recovery, reagent consumption

as three components: (1) a **database of extraction systems** (both ligands, diluent, acidity and
anions, loading, per-metal D, scrub, strip, phase behaviour; missing measurements are *never*
reconstructed as confirmed facts; one SMILES per system is not enough for two-ligand systems);
(2) **local recipe optimisation** (several purity–recovery–consumption trade-offs per chemical
family; simple response models and equilibrium relations first; limited Bayesian optimisation only
where it saves expensive measurements; proposals outside the studied region explicitly flagged);
(3) **cascade calculation** (distribution models fed into a material balance; the algorithm's
usefulness is measured by the improvement of the chosen regime, not by MAE over all pairs).
The first applied target (§9): a working **PC88A scheme for Pr/Nd with separate scrub and strip,
compared with Cyanex 272 at the same feed and product requirement**. "The main product is a
computed and then verified regime, not a new regressor."

Everything runs locally on this machine. No cluster, no cloud.

## 1. What the task document says in detail (faithful English digest)

§1 Measure process improvement. For target metal T and impurity I: D_T = C_T,org/C_T,aq;
SF(T/I) = D_T/D_I; purity = fraction of T among metals in the final product; recovery = T in the
product / T in the feed. Raising both D's does not help; a high SF at near-zero transfer is useless.
Choose by **purity, recovery, throughput, number of stages and reagent consumption together**, and
also track organic loading capacity, phase-disengagement time, ligand losses, third-phase formation
and regenerability. Single equal-volume equilibrium stage: fraction extracted = D·(O/A)/(1+D·(O/A)).
D itself changes with O/A through loading and equilibrium acidity, so a dilute-sample constant may
not be dropped into a cascade unchecked. Evaluate target pairs separately (Pr/Nd, Nd/Sm, Tb/Dy,
Dy/Ho); La/Lu quality says nothing about neighbours.

§2 Clean the loaded organic. First practical route: extract the target fraction, then selectively
scrub the impurity, then strip the product separately. The scrub solution need not match the feed
(acidity, complexant, or a solution of the enriched target that displaces the less-retained
component). Pr/Nd reference: PC88A; consider co-extraction -> Pr removal from the organic -> Nd
strip together. Banda et al. (doi 10.1016/j.jiec.2014.03.002): max SF only about 1.5, but extraction
plus scrubbing built an Nd recovery scheme. Thakur et al. (doi 10.1016/0304-386X(93)90084-Q):
counter-current PC88A tests gave > 5 kg Nd2O3 at 97 % purity with > 85 % recovery. These belong to
those compositions and regimes, not to our feed. Proposed: a **two-regime cycle on one organic
phase** — loading stage for capacity and recovery; scrub stage for maximum T/I retention
difference; strip stage for easy T return. Return the scrub stream to the right stage and carry its
composition in the material balance. If the scrub contains pure target metal, account for its
external input: it must not inflate the computed recovery from the feed.

§3 Two competing ligands. An organic ligand carries the preferred metal into the organic phase, an
aqueous (hydrophilic) ligand holds the competing metal in water; the difference is amplified from
both sides of the equilibrium. Precedent: water-soluble BLPhen + lipophilic DGA (doi
10.1021/jacsau.2c00671); inverse architecture (lipophilic phenanthroline carboxamide + hydrophilic
DGA, JAEA). Directions: to carry the heavier component, organic DGA + aqueous ligand preferring the
lighter; the inverse for the other direction; for already-loaded organic, add the holding ligand
only to the scrub solution. Choose the ligand pair by the **difference in preference for T and I**,
not by binding strength; too strong an aqueous complexant holds both metals, too strong an organic
one prevents stripping; select with regeneration of both phases in mind.

§4 Tune medium, loading and organic composition jointly. Working-regime maps for the available
families: organophosphorus acids D2EHPA, PC88A, Cyanex 272; DGAs; phenanthroline carboxamides;
each with its compatible medium and relevant condition range. Map axes: equilibrium acidity, anion
composition, both ligand concentrations, total metal concentration and ratio, O/A, diluent,
modifier, temperature. Start from the feed composition. (1) Choose a partial-extraction window
(complete transfer of both hides differences and overloads the organic). (2) Use competition at
loading (impurity displaced without excessive product loss; cap loading by real phase behaviour).
(3) Anion and water in the organic matter (TODGA outer-sphere water clusters; nitrate/thiocyanate
mechanism study) — not a recipe. (4) Extractant mixtures as capacity/acid-exchange tuning
(Cyanex 272 + Alamine 336 raised Pr and Nd extraction in a chloride system); stronger extraction
does not prove higher SF.

§5 New molecules (series A–D) — out of software scope for gen18 except that the database schema
must be able to hold two-ligand systems and stereo-/substituent variants of one scaffold.

§6 Fraction routing for multi-component feeds: remove interferents -> group extraction ->
selective scrub -> hard pair -> strip/regenerate. Cascade calculation must include
concentration-dependent D, per-metal balance, acid balance, organic loading and recycle streams;
from measured isotherms choose stages, feed-entry points, O/A and scrub flow jointly; convert the
economic criterion into cost or resource consumption per unit product of given purity. A stage
count may not be promised from one SF value.

§7 Eu: reductive strip Eu(III)->Eu(II) (Hirai & Komasawa, doi 10.1252/jcej.25.644; EuSO4 > 95 %
purity at up to 83.2 % recovery). Special route, not transferable to Pr/Nd or Tb/Dy. Out of gen18
software scope beyond a schema field for oxidation-state routes.

§8 Role of computation. gen14–gen17 results are a constraint on applicability: the direction call
is useful for pre-screening, not a reliable universal ranking of extractants. Old models are kept.
New applied task: **input = feed composition and product requirements; output = several
chemical-system and process-regime options with purity, recovery and cost estimates.** For
molecular selection compare relative free energies with solvation, complex composition and
conformers; ΔΔG = −RT ln SF holds only under a consistent conditional-transfer definition; raw
total energies of complexes of different composition may not be compared. Old xTB nulls do not
prove absence of coordination chemistry but give no reason to continue that ranking.

§9 Where to start. Quick applied result: the PC88A Pr/Nd scheme with separate scrub and strip,
compared with Cyanex 272 at identical feed and product spec. Intrinsic selectivity: a DGA +
aqueous size-selective ligand combination, including the "complexant only in the scrub" variant,
with regeneration and consumption of both ligands in the selection. Software: the chain
"system composition -> D at given loading -> cascade -> purity, recovery, consumption", which turns
the corpus into a process-selection tool and lets one judge whether a new molecule's synthesis is
worth its gain. Success = a system or regime that separates more cleanly at comparable recovery,
or gives the same product with fewer stages and less reagent. No model architecture is mandatory.

## 2. Repository facts every agent needs (verified 2026-09-13)

Layout and conventions:
- gen12+ live as self-contained directories under `generations/`; gen18 is
  `generations/gen18_process/` with package `gen18proc/`, `scripts/`, `tests/`, `results/`,
  `systems/` (the database files), `DESIGN.md`, `PRE_REGISTRATION.md`, `GEN18_REPORT.md`,
  `README.md`. Read `generations/README.md` "Layout inside a generation" for the house style.
- Scripts are run **from the repository root**: `.venv/Scripts/python.exe generations/gen18_process/scripts/<x>.py`.
  Tests: `.venv/Scripts/python.exe -m pytest generations/gen18_process/tests -q`. Each test module
  puts its own generation on `sys.path` (pattern: `generations/gen16_leads/tests/test_anchors.py`).
- The interpreter in the Bash tool is `/d/ml_separator_gh/.venv/Scripts/python.exe` (plain
  `python` resolves to a Windows Store stub and fails). Python 3.14.5, numpy 2.5.3, pandas 3.0.5,
  scipy 1.18.1, scikit-learn 1.9.0, rdkit 2026.03.6, joblib. **pandas 3**: `DataFrame.stack()`
  keeps NaN; `str.split("||")` is regex. Set `PYTHONIOENCODING=utf-8` for any non-ASCII output.
- Machine: 12 threads, **8 GB RAM with ~2.7 GB free**. Run at most ONE Python process per agent,
  never parallel tree fits, `n_jobs<=2`. Do not install packages. Do not upgrade scikit-learn.
- Every number in a report carries its regime (what cohort, what hold-out, what averaging unit).
  A number without one must not be quoted. Report nulls as results.
- Missing data are recorded as missing, never approximated silently (the bundle's own discipline).
- Nothing outside `generations/gen18_process/` may be modified. Frozen inputs are read-only.
- Do not commit. The user decides when to commit.

Frozen inputs (read-only; use `gen13sep.paths` constants where they exist):
- Bundle: `dataset with 3D structures/dataset.parquet` — 5992 rows × 2261 cols, SHA-256
  `fefbefc6fe993aa9ce9db1a0c338adb9e5f58a8b75bab084cc1df4e024faf5dd`. Columns that matter here:
  `metal` (14 lanthanides, no Pm/Y), `canonical_smiles` (190 distinct), `extractant_name` (228),
  `D`, `log_D` (base-10), 64 `cond__*` columns: continuous `cond__acid_concentration_M`
  (range 2.6e-7–8.98, median 1.6), `cond__extractant_concentration_M` (2.3e-4–1.0, median 0.10),
  `cond__metal_concentration_mM` (1e-6–2180, median 0.1; NaN on 1532 rows),
  `cond__temperature_C` (5–55), `cond__contact_time_min`; one-hot `cond__acid__{hno3,hcl,h2so4,
  hclo4,citric_acid,lactic_acid,malonic_acid,tartaric_acid,hno3_oxalic_acid}`,
  `cond__diluent__*` (about 45 diluents), `cond__additive__{1_octanol,tbp,hdehp,dhoa,...}`.
  Also `geom_cond__acid_class` ∈ {nitrate 4901, chloride 769, carboxylate 253, sulfate 68,
  perchlorate 1}, `geom_cond__diluent_family`. There is **no O/A column** (phase_ratio_bin is
  'missing' everywhere) and no pH column: acidity is only `cond__acid_concentration_M`.
- Provenance: `runs/gen6_provenance/provenance_table.parquet` (5992 rows; `safe_exp_id`,
  `extractant`, `metal_symbol`, `condition_id`, `cell_id`, `publication_id`,
  `experiment_series_id`, `replicate_id`, ...). Join to the bundle on `safe_exp_id`. 105
  publications.
- Chemistry map (chemotype, ECFP cluster per SMILES): `runs/gen7_architecture/cache/chemistry_map.parquet`.
- gen13 helpers you may import (put `generations/gen13_separation` on sys.path):
  `gen13sep.paths` (BUNDLE_PARQUET, GEN6_PROVENANCE_PARQUET, CHEMISTRY_MAP_PARQUET,
  assert_bundle_unchanged), `gen13sep.metals` (LANTHANIDES, ATOMIC_NUMBER, SHANNON_RADIUS_CN8/9),
  `gen13sep.cohort` (load_bundle, apply_quarantine — TODGA-under-a-foreign-name quarantine and the
  log D <= -6 sentinel rule, and the exact 64-column condition key with NaN as a value).
- Deployed curve predictor (structure -> 14-metal relative curve): `generations/gen15_curve/scripts/g15_predict.py`
  (CLI `fit` / `predict --smiles ... [--measured "La/Lu=-1.31"]`). Its value is the *direction* of
  selectivity (0.82 accuracy, BP); magnitude is essentially a constant. Use it only as a
  pre-screening prior, never as a D source for a cascade. The `fit` step writes
  `generations/gen15_curve/models/deploy_g15.joblib` (gitignored) — check whether it exists before
  relying on it; if it does not, gen18 must work without it.

What the corpus contains and does not contain (measured 2026-09-13):
- **No PC88A, Cyanex 272, D2EHPA/HDEHP, or Ionquest rows at all** (name and SMILES-substructure
  search). The only acidic organophosphorus extractant is dihexyl hydrogen phosphate (10 rows,
  one condition, 10 metals). HDEHP appears only as an additive (`cond__additive__hdehp`, 9 rows).
  Carboxylic-acid extractants: 4, all tiny. So the Pr/Nd PC88A vs Cyanex 272 case **cannot be
  parameterised from the corpus**; its parameters must be literature entries in the database,
  flagged as such, with the DOI, and the case study must carry their uncertainty as ranges.
- The corpus is dominated by diglycolamides (DGA) and N-donor extractants in nitrate. TODGA
  (`CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC`) has 1714 rows, 14 metals, 304 distinct
  acidities, 168 extractant concentrations, HNO3 (1293) and HCl (132). Pr/Nd nitrate rows for
  TODGA: 175 (Nd 110, Pr 65) over 60 acidities × 27 extractant concentrations.
- **Loading series exist**: 10 for TODGA (>= 3 distinct metal concentrations at fixed acid,
  extractant and diluent), e.g. Nd, 3 M HNO3, 0.1 M TODGA, aliphatic diluent: **19 distinct metal
  concentrations**; Nd 3 M / 0.2 M: 4; Nd 3 M / 0.3 M: 8; Ce 3 M / 0.1 M: 8; Eu at 0.5/1/3 M: 4
  each. Whole corpus: 16 loading series over 4 extractants (metals Ce, Eu, La, Nd). These are the
  only corpus evidence for loading-dependent D and are the natural validation target for the
  mass-action loading model.
- 71 extractants have both Pr and Nd measured; most at a single condition (14-metal cells). Rich
  multi-condition Pr/Nd coverage exists for TODGA, TEHDGA (85 acidities), the N-methyl-N-octyl
  DGA (36), TDdDGA (17), DMDO-HPyranDGA (10).
- The existing MASSACTION block (gen5, `src/lanthanide_separation/levels.py`) validated on this
  cohort that log D is linear in the logarithms of the conditions: extractant-titration slope
  2.64, IQR [2.36, 2.88], median linear R² 0.985 — i.e. the solvating mass-action law
  `log D = log K_ex + n·log[L] + 3·log[NO3-]` holds on the corpus.

## 3. Chemistry the design must get right

Two extraction mechanisms cover the families in scope:

(a) **Cation exchange (acidic organophosphorus: D2EHPA, PC88A = HEH[EHP], Cyanex 272)** — the
extractant is a dimer (HA)2 in aliphatic diluents:
    Ln3+ + 3 (HA)2(org)  <=>  Ln(HA2)3(org) + 3 H+
    D_M = K_M · [(HA)2]_free^3 / [H+]^3     (slope 3 in both; ideal form)
Loading consumes extractant (3 dimers per Ln3+) and **releases 3 H+ per Ln3+ into the aqueous
phase**, so equilibrium acidity rises with loading and D falls — this is the "D changes with O/A
through loading and equilibrium acidity" of §1. Partial saponification (NaOH/NH3) neutralises the
released acid. Scrub: acid solution or an enriched-target solution (displacement). Strip: strong
acid (D falls as [H+]^-3). SF between neighbours is the ratio of K's and is nearly
condition-independent in the ideal model; real deviations come from non-ideality and loading.

(b) **Solvation (neutral: DGAs such as TODGA; phenanthroline carboxamides)** in nitrate:
    Ln3+ + 3 NO3- + n L(org)  <=>  Ln(NO3)3·L_n(org)      (n about 3 for TODGA, corpus slope 2.64)
    D_M = K_M · [L]_free^n · a(NO3-)^3 · f([H+])
Loading consumes n ligands per metal; no acid is released. Strip: dilute acid / low nitrate.
Third-phase limits apply at high loading (record as phase-behaviour data when known; never invent).

(c) **Aqueous hold-back complexant (hydrophilic ligand, §3)**: with side-reaction coefficient
    D_M,eff = D_M / (1 + Σ_i β_M,i · [L_aq,free]^i)
Metal-specific β turn a small organic selectivity into a larger effective one. The free aqueous
ligand is itself depleted by complexation at high metal loading (solve it, do not assume excess).

(d) **Multi-metal competition**: all metals share the same free extractant and the same
released-acid pool, so the stage equilibrium is a coupled nonlinear system in the free ligand
(and H+), solved per stage. Constant-D shortcuts are allowed only as a labelled sanity limit.

Non-negotiables: units explicit everywhere (mol/L, mM for feed metals if used, O/A as volumetric
ratio); mass balance closes per metal to 1e-8 relative; acid balance included for mechanism (a);
Kremser's analytic countercurrent solution must be reproduced in the constant-D limit; every D
source carries a provenance status; every proposed regime carries applicability flags against the
database's condition ranges.

## 4. Validity contract for gen18 (adapted from gen16 START_HERE §1)

- Any model fitted to corpus data (the mass-action D model) gets a **pre-registration before it is
  fitted**: question, endpoint, decision rule, baselines. Primary use is the *known-extractant*
  regime (design A: the extractant is in the database, conditions change) because that is the
  process-design use case; say so and do not claim zero-shot skill.
- Baselines: the constant per-(extractant, metal, anion) mean log D; a nearest-condition 1-NN in
  (log acid, log extractant); and, for loading series, constant-D from tracer data. Report the
  increment over the cheapest sensible alternative, not only over the trivial one.
- Hold-out by **publication** (leave-one-publication-out inside a system) so a paper's fingerprint
  cannot flatter the fit; also report an in-sample number, labelled.
- **Reliability before correlation**: any fitted slope must report split-half or jackknife
  reliability before it is interpreted (gen16 L1 lesson).
- Determinism: fixed seeds, no wall-clock in results; results reproducible from scripts.
- Count comparisons; label exploratory analyses.

## 5. Proposed architecture (a starting skeleton for the design panel to improve or replace)

    generations/gen18_process/
      BRIEF.md                   this file
      DESIGN.md                  final design (written by the synthesiser)
      PRE_REGISTRATION.md        D-model evaluation protocol, frozen before fitting
      README.md, GEN18_REPORT.md
      gen18proc/
        paths.py                 roots, frozen inputs, output dirs
        systems.py               schema + loader/validator for the extraction-systems database
        ingest.py                corpus -> per-system distribution records with provenance
        dmodel.py                distribution models: MassActionCationExchange, SolvatingMassAction,
                                 AqueousComplexant wrapper, Empirical (corpus-fitted), ConstantD
        equilibrium.py           single-stage multi-metal equilibrium with loading (coupled solve)
        cascade.py               countercurrent extraction–scrub–strip cascade with recycle
        metrics.py               purity, recovery, SF, throughput, reagent consumption, cost proxy
        optimize.py              design space, grid/LHS + Pareto, optional GP-BO, OOD flags
        report.py                tables/markdown helpers
      systems/                   the database: one JSON per system + corpus_records.csv
      scripts/                   g18_build_db.py, g18_fit_dmodels.py, g18_eval_dmodels.py,
                                 g18_case_prnd.py, g18_optimize.py
      tests/                     invariants (see §6)
      results/                   outputs

## 6. Tests that must exist (minimum)

Kremser agreement in the constant-D limit; per-metal and acid mass-balance closure; monotone
recovery in stages and O/A; limits D=0 and D->inf; loading lowers D (mechanism a and b) and
releases acid (a only); aqueous complexant lowers effective D and depletes free ligand; scrub with
pure-target solution does not inflate recovery from feed; database validator refuses entries
without provenance status/source; OOD flags fire outside recorded ranges; determinism of optimiser
outputs; bundle SHA-256 check before ingestion; the fitted TODGA slope is within the corpus-
validated range and its reliability is reported.
