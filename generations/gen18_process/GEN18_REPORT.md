# Gen18 report -- the process-design chain

Every number carries its regime (cohort, hold-out, averaging unit, parameter status) in the caption of its table. Assembled by `scripts/g18_report.py` from `results/`; nothing here was typed in by hand except the section text.

## 0. What this generation answers, and what it does not

`docs/kshot_next_steps_20260913.md` sets the programme's goal as **a cleaner product at comparable
recovery, or the same product with fewer stages and less reagent** — explicitly *not* a lower ML
error — and asks the software (§8–§9) for one thing: the chain

> extraction-system composition → D of each metal **at the given loading** → cascade (extraction +
> scrub + strip + recycle) → purity, recovery, reagent consumption,

plus a systems database that never reconstructs a missing measurement as a fact, local recipe
optimisation that flags proposals outside the studied region, and a first applied case: **PC88A
versus Cyanex 272 for Pr/Nd with separate scrub and strip**.

**That chain now exists and runs locally.** What follows is what it is worth.

### The three things worth knowing

1. **The cascade arithmetic is trustworthy; it is the chemistry parameters that are thin.** The
   solver reproduces hand-computed Kremser and single-stage results to ≤ 2.5 × 10⁻⁹, purity,
   recovery and enrichment to the last digit, and mass balances to ~10⁻¹³. Nothing in this
   generation's uncertainty comes from the numerics.
2. **The corpus cannot predict D at new conditions better than looking up the nearest measurement**
   (R1, a pre-registered **null**: a pooled mass-action fit scores 1.090 against 1.064 for a
   nearest-condition 1-NN, macro over 14 systems, leave-one-publication-out). So the chain's D
   source is an interpolation, and its reach is the range of conditions somebody has already
   measured. This is the binding limit on every recipe the program proposes.
3. **The loading correction is real and it matters** (R2, pre-registered and **supported**): the
   ideal ligand-depletion term cuts per-series error from 0.514 to 0.305 across 10 loading series
   (95 % CI of the gain [0.030, 0.474]), and on the 7 series where loading actually moves D it
   goes 0.729 → 0.424. The middle step the task document asks for — *D at the given loading*, not
   D from a dilute sample — is the part that is earned by evidence.

Point 3 is the practically useful one: §1 of the task document warns that a D measured in a dilute
sample may not be dropped into a cascade, and this generation now has a pre-registered, corpus-
validated correction for exactly that, plus the flags (`HIGH_LOADING`, `THIRD_PHASE_RISK`,
`PHASE_BEHAVIOUR_UNKNOWN`) that stop the optimiser proposing a regime past a phase limit somebody
actually measured.

### What the program can be asked, today

Give it a feed and a product specification and it returns ranked regimes with purity, recovery,
stage counts, reagent consumption per kg of oxide, applicability flags and a parameter-status
column — for any system in the database with the metals of interest. On the 10 nitrate systems
carrying both Pr and Nd it finds, for example, a 21-stage TODGA regime at **purity 0.986 and
recovery 0.984** (`IN_DOMAIN_WITH_CAVEATS`; flagged `HIGH_LOADING` and `PHASE_BEHAVIOUR_UNKNOWN`,
so it is a candidate to test, not a recommendation).

### What it cannot be asked

- **Anything about PC88A, Cyanex 272 or D2EHPA from data.** The corpus contains none of them
  (§7 below): the case study runs on literature placeholders with declared ranges, and its output
  is an interval conditioned on those placeholders, not a measurement. Transcribing the real
  parameters from Banda 2014, Thakur 1993 and a Cyanex 272 source (open item **U4**) is the single
  change that would most improve this generation.
- **To rank ligands.** The gen15 direction model returns only two distinct magnitudes across 71
  candidates: it is a sign call, as gen16 concluded, and it is wired in as a pre-screen that
  validator rule V6 forbids from ever supplying a D.
- **Cost in currency.** `config/prices.json` holds unsourced placeholders (**U5**), so consumption
  per kg of oxide is the primary economic number and the cost proxy is NaN wherever a price is
  missing.

### The Pr/Nd case: what it settled, and what it did not

The case ran to specification otherwise (64 parameter draws × 2 systems × 250 LHS designs, 146
min). Its headline outputs need reading carefully:

- **The PC88A versus Cyanex 272 comparison is undecided, not lost.** No spec cell — not even the
  loosest, purity 0.95 at recovery 0.80 — was reached by either system in any of the 64 draws, so
  every comparison column is empty. **This is a statement about the search, not about PC88A.** The
  Fenske minimum at total reflux for that loosest cell at SF 1.4 is about **10 theoretical
  stages**, far inside the 40 + 40 the search allowed, so a feasible region must exist; a
  250-point Latin hypercube over 17 design variables did not find it. A structured grid over the
  same chemistry is reported in `results/case_prnd/targeted/`.
- **What a structured search found instead** (`results/case_prnd/targeted/`, exploratory). A
  cation-exchange extractant releases 3 H⁺ per Ln³⁺, so an unsaponified 0.1 M feed self-acidifies
  and extraction stalls — the grid shows exactly that, with maximum recovery **0.667** at zero
  saponification rising to **1.000** at 0.20–0.50, and purity collapsing again at 0.65 when the
  organic takes everything. The chain reproduces from first principles why industrial PC88A
  circuits are run saponified, and it is the variable that decides feasibility.
  **But the cell is still not met**: recovery ≈ 1 is reachable and purity ≈ 0.999 is reachable,
  never together — the best purity anywhere at recovery ≥ 0.80 is **0.814**. So the empty result
  is *not* explained by a bad `log_k` window (both ends of the trade-off work with it), nor purely
  by the random sampler. What remains untested is the circuit *shape*: the Fenske ~10 stages is a
  **total-reflux** bound, and at finite reflux this separation needs the internal reflux set by
  feed-stage, scrub-return and per-section O/A — which both searches held at defaults. That, or
  the placeholders are simply too far from real PC88A. Unresolved, and it needs U4.
- **Consistency check (b) is supported by the analytic bound, not by the stage ladder the script
  ran.** The verdict "the stage count is of the order of 70 + 70, not 7 + 7" is correct: 14 stages
  is *below* the 24-stage total-reflux minimum for (0.99, 0.99) at SF 1.4, hence impossible at any
  reflux, while the patent's 144 is ≈ 6 × N_min, a normal practical multiple (I reproduced
  N_min = 24.0 independently). The ladder table beside it is uninformative: its recovery collapses
  from 4 × 10⁻⁷ to 5 × 10⁻⁸⁵ as stages grow, the signature of a scrub washing the product back, so
  its 8 rows per rung sampled only degenerate designs. Read the Fenske line, not the ladder.
- **Check (a)** (Thakur's 97 % purity at > 85 % recovery) is therefore **not tested** by this run,
  and the report should not be read as casting doubt on it.
- **§13.5, the DGA + aqueous hold-back ligand**, re-run with the corrected D source: at the
  reference TODGA regime the β contrast buys almost nothing (purity 0.8806 → 0.8810 as
  Δlog β goes 0 → 1) because that regime already sits at high loading — and the regime is
  `INADMISSIBLE` anyway, because the sourced TODGA third-phase limit flags it. The mechanism
  itself is correct (in isolation a hold-back ligand with β(Pr) > β(Nd) lifts SF(Nd/Pr) from 2.00
  to 2.84); it is this operating point that cannot use it.

### Two findings from the integration pass

- **A defect in the deployed D source, found and fixed.** Because R1 was a null, every corpus
  system draws its D from the nearest-condition lookup — which was applying the loading correction
  to records that had themselves been measured under load. 28.7 % of corpus records with a metal
  concentration sit above loading fraction 0.1 (TBDGA: 81 %). Each record is now lifted to its own
  tracer limit first, using the same law R2 validated. The Pareto leaders barely moved, but 276 of
  298 candidates changed (median purity change 0.088). R1 and R2 are untouched: neither applies the
  term that way. Details in `addenda/INTEGRATION.md` §4.
- **A hypothesis of mine that failed.** After the null, the obvious repair is to use the
  mass-action fit only where its fitted exponent is reliable. It does not work: the correlation
  between slope reliability and M1's advantage is −0.117 (p = 0.69), and the two largest M1 wins
  are on systems whose slopes are *not* reliable. Reported rather than dropped, because it closes
  off a plausible-looking route (`results/eval/RELIABILITY_PROBE.md`).

### Where the honest uncertainty sits

Chemical coverage, again — the same constraint gen16 identified. Of 14 systems with enough
multi-publication data to fit at all, only 6 have an interpretable exponent; 4 have their exponent
pinned at the prior because the corpus never varied the ligand concentration for them. The nitrate
DGA family is well covered and the acidic organophosphorus family, which industry actually uses
for Pr/Nd, is absent. No modelling choice repairs that; measurements do.


---

## 1. What was built and what the database contains

The chain *extraction-system composition -> D of each metal at the given loading -> countercurrent cascade (extraction + scrub + strip + organic recycle) -> purity, recovery, throughput, reagent consumption -> local regime optimisation* (DESIGN.md section 0.1). Modules: `gen18proc/systems.py` (database, validator), `ingest.py`, `literature.py`, `dmodel.py` (mass-action D models, composition, flags), `domain.py`, `equilibrium.py` (coupled stage), `cascade.py` (Newton + successive substitution, Kremser init, origin pass, ledgers), `metrics.py`, `evalproto.py` (pre-registered fit, LOPO, reliability, loading), `optimize.py` (design space, LHS + Pareto, gated GP-BO), `screen.py`, `report.py`.

Database: 289 systems ({'corpus': 287, 'literature': 2}); 5860 distribution records; families {'diglycolamide': 125, 'other': 93, 'n_donor': 47, 'phen_carboxamide': 15, 'carboxylic_acid': 4, 'acidic_organophosphorus': 3, 'amine': 2}; two-ligand systems 5; scaffold ids ['DGA_core', 'phosphinic_acid', 'phosphonic_monoester'].

Exclusions (gen13 quarantine): 132 rows ({'todga_name_mismatch': 129, 'sentinel_logD_le_-6': 3}).

## 2. Data audit (from `DATA_AUDIT.md`, the frozen source of the cohort counts)

### 1. Data facts of DESIGN.md 0.2, recomputed

| fact | value |
|---|---|
| bundle rows -> after gen13 quarantine | 5992 -> 5860 (129 TODGA-structure-under-foreign-name rows, 3 sentinel rows at log D <= -6) |
| publications (gen6 `publication_id`, none missing) | 105 |
| systems under the key of section 3.1 | 287 corpus + 2 literature = 289 |
| two-ligand (synergist) systems / systems with alcohol modifiers | 5 / 11 |
| loading series, publication-aware (>= 3 distinct metal concentrations at fixed system, metal, publication, acid, extractant concentration; after the duplicate rule) | **10** |
| loading series, publication-blind (sensitivity only) | 11 |
| loading-active series (log D range >= 0.3; the rising Ce series included) | 7 |
| unit-slip duplicate rows (`UNIT_SLIP_DUPLICATE`, fit-ineligible) | **7** (7 in pub_0e7f3e0563, 7 on the 3 M side; 7 pairs) |
| tied-D groups corpus-wide (same publication, SMILES, metal, D equal to 6 significant digits at different conditions) | 130 groups, 324 rows, 24 publications; of these the unit-slip tier is 7 groups and the `TIED_D` tier 123 groups / 310 rows (kept, flagged) |
| replicate groups (exact 64-column condition key + publication + SMILES + metal, >= 2 rows) and median within-group sd of log D | 282, **0.225** (fit-eligible rows only: 0.227) |
| fit-ineligible rows | 19 ({'acid_nan_or_nonpositive': 11, 'UNIT_SLIP_DUPLICATE': 7, 'extractant_nan_or_zero': 1}) |
| system-metal groups fittable (>= 6 aggregated points, >= 3 distinct levels on log acid or log extractant), band 20-30C (NaN temperature assigned to it) | **233** (all bands pooled: 235) |
| of these, spanning >= 2 publications (E1) | **59 groups in 14 systems** (all bands: 59 / 14) |
| TODGA / nitrate / aliphatic / no additive (`sys_5cb78e5000d40860`) | 514 rows, 28 publications, 14 metals, acid 0.009333-5 M, extractant 0.0002313-0.3 M, metal 0.0001-2180 mM, 5-45 C; Pr 28 rows, Nd 66 rows, 8 publications with both |
| TODGA/Nd 3 M HNO3 / 0.1 M loading series pub_5a68dc5665 | 6 points, 4.9-12 mM, log D 1.37 -> -0.22; records Ca_SAFE:2222, Ca_SAFE:2221, Ca_SAFE:2220, Ca_SAFE:2219, Ca_SAFE:2224, Ca_SAFE:2217; series caf22524baa8080e |
| rows with NaN metal concentration / NaN temperature (after quarantine) | 1407 / 75 |
| temperature bands (NaN -> 20-30C) | {'20-30C': 5701, '30-40C': 63, '40-50C': 38, '<20C': 30, '>=50C': 28} |
| corpus coverage of the Pr/Nd case | no PC88A, Cyanex 272 or D2EHPA rows (both case systems are literature entries with 0 records) |

### 2. E1 cohort (PRE_REGISTRATION.md section 2; unit of the decision = system)

Fit-eligible records of band 20-30C, aggregated per replicate group; a (system, metal) group is fittable with >= 6 aggregated points and >= 3 distinct levels on log acid or log extractant; it enters E1 when it spans >= 2 publications. `publications` is the union over the system's E1 groups; `max_publications` the largest single group. R1 (iii) needs M1 to beat B1 in ceil(0.6 x 14) = 9 systems.

| system_id | ligand | name | n_groups | publications | max_publications | n_records | metals |
|---|---|---|---|---|---|---|---|
| sys_07ee9637c98c1e20 | TODGA | TODGA in other, nitrate medium, no additive | 13 | 7 | 5 | 249 | Ce;Dy;Er;Eu;Gd;Ho;La;Lu;Nd;Pr;Sm;Tb;Tm |
| sys_120bb57e9148dd0d | TEHDGA | TEHDGA in aliphatic hydrocarbon, nitrate medium, with 1_octanol (modifier) | 1 | 2 | 2 | 31 | Nd |
| sys_1419f400c83e9ad8 | DMDODGA | DMDODGA in aliphatic hydrocarbon, nitrate medium, no additive | 1 | 2 | 2 | 239 | Eu |
| sys_267144068d8e09d2 | DMDODGA | DMDODGA in aliphatic hydrocarbon, nitrate medium, with 1_octanol (modifier) | 1 | 2 | 2 | 102 | Nd |
| sys_5cb78e5000d40860 | TODGA | TODGA in aliphatic hydrocarbon, nitrate medium, no additive | 14 | 27 | 18 | 514 | Ce;Dy;Er;Eu;Gd;Ho;La;Lu;Nd;Pr;Sm;Tb;Tm;Yb |
| sys_81bcc3c06cbf0b4c | D3DODGA | D3DODGA in aliphatic hydrocarbon, nitrate medium, no additive | 1 | 3 | 3 | 35 | Nd |
| sys_81e3169a7f85c2b9 | TBDGA | TBDGA in aromatic, nitrate medium, no additive | 7 | 4 | 4 | 198 | Dy;Er;Eu;Gd;La;Nd;Sm |
| sys_95746e54b97ae741 | NTAamide(C8) | NTAamide(C8) in aliphatic hydrocarbon, nitrate medium, no additive | 1 | 2 | 2 | 20 | Eu |
| sys_a2a472b5124b09c6 | TODGA | TODGA in aromatic, nitrate medium, no additive | 14 | 8 | 7 | 294 | Ce;Dy;Er;Eu;Gd;Ho;La;Lu;Nd;Pr;Sm;Tb;Tm;Yb |
| sys_a7195d8a9d8696e0 | TEHDGA | TEHDGA in aliphatic hydrocarbon, nitrate medium, no additive | 2 | 8 | 5 | 71 | Eu;Nd |
| sys_a9fea6791c7a14d6 | DMDOHEMA | DMDOHEMA in aliphatic hydrocarbon, nitrate medium, no additive | 1 | 2 | 2 | 52 | Eu |
| sys_b06c95ac0efc3d4e | C5BTBP | C5BTBP in other, nitrate medium, no additive | 1 | 2 | 2 | 45 | Eu |
| sys_be340fe5092ae17a | 2-[1-(dioctylamino)-1-oxopropan-2-yl]oxy-N-N-dioctylpropanamide | 2-[1-(dioctylamino)-1-oxopropan-2-yl]oxy-N-N-dioctylpropanamide in aliphatic hydrocarbon, nitrate medium, no additive | 1 | 3 | 3 | 40 | Eu |
| sys_ebce91a448faa5c3 | TODGA | TODGA in alcohol modifier, nitrate medium, no additive | 1 | 5 | 5 | 39 | Eu |

Aggregated by extractant name (the panel's counts in PRE_REGISTRATION.md section 2 are per extractant name, not per system key):

| ligand | n_systems | n_groups | max_publications |
|---|---|---|---|
| 2-[1-(dioctylamino)-1-oxopropan-2-yl]oxy-N-N-dioctylpropanamide | 1 | 1 | 3 |
| C5BTBP | 1 | 1 | 2 |
| D3DODGA | 1 | 1 | 3 |
| DMDODGA | 2 | 2 | 2 |
| DMDOHEMA | 1 | 1 | 2 |
| NTAamide(C8) | 1 | 1 | 2 |
| TBDGA | 1 | 7 | 4 |
| TEHDGA | 2 | 3 | 5 |
| TODGA | 4 | 42 | 18 |

Per-group table: `results/audit/e1_groups.csv`.

### 3. E2 cohort (publication-aware loading series; unit = series)

| loading_series_id | system_id | ligand_name | metal | publication_id | acid_nominal_M | ligand_M | n_points | mM_min | mM_max | log_d_tracer | log_d_min | log_d_max | log_d_range | loading_active |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ls_sys_07ee9637c98c1e20_Ce_pub_917a4583d4_3_0.1 | sys_07ee9637c98c1e20 | TODGA | Ce | pub_917a4583d4 | 3 | 0.1 | 8 | 0.011 | 52.42 | 1 | 1 | 1.65 | 0.6502 | True |
| ls_sys_5cb78e5000d40860_Nd_pub_5a68dc5665_3_0.1 | sys_5cb78e5000d40860 | TODGA | Nd | pub_5a68dc5665 | 3 | 0.1 | 6 | 4.9 | 12 | 1.371 | -0.2218 | 1.371 | 1.593 | True |
| ls_sys_5cb78e5000d40860_Nd_pub_d3c970567f_3_0.1 | sys_5cb78e5000d40860 | TODGA | Nd | pub_d3c970567f | 3 | 0.1 | 4 | 4.2 | 6.8 | -0.03218 | -0.03218 | -0.0196 | 0.01259 | False |
| ls_sys_5cb78e5000d40860_Nd_pub_d3c970567f_3_0.2 | sys_5cb78e5000d40860 | TODGA | Nd | pub_d3c970567f | 3 | 0.2 | 4 | 9.7 | 13 | -0.02298 | -0.02758 | -0.02039 | 0.00719 | False |
| ls_sys_5cb78e5000d40860_Nd_pub_d3c970567f_3_0.3 | sys_5cb78e5000d40860 | TODGA | Nd | pub_d3c970567f | 3 | 0.3 | 8 | 20 | 30.4 | -0.03152 | -0.03267 | 0.001543 | 0.03421 | False |
| ls_sys_740f07a521006be1_Nd_pub_5a68dc5665_3_0.1 | sys_740f07a521006be1 | TDDGA | Nd | pub_5a68dc5665 | 3 | 0.1 | 6 | 20 | 30 | 0.9542 | 0.6021 | 0.9542 | 0.3522 | True |
| ls_sys_81bcc3c06cbf0b4c_Nd_pub_0e7f3e0563_1_0.2 | sys_81bcc3c06cbf0b4c | D3DODGA | Nd | pub_0e7f3e0563 | 1 | 0.2 | 18 | 7.341 | 289.7 | 0.7134 | -0.6081 | 1.82 | 2.428 | True |
| ls_sys_81bcc3c06cbf0b4c_Nd_pub_0e7f3e0563_3_0.2 | sys_81bcc3c06cbf0b4c | D3DODGA | Nd | pub_0e7f3e0563 | 3 | 0.2 | 8 | 8.2 | 289.3 | 1.602 | -0.5589 | 1.602 | 2.161 | True |
| ls_sys_a7195d8a9d8696e0_Nd_pub_15b174237f_1_0.2 | sys_a7195d8a9d8696e0 | TEHDGA | Nd | pub_15b174237f | 1 | 0.2 | 3 | 6.933 | 41.6 | 1.172 | 0.1276 | 1.172 | 1.045 | True |
| ls_sys_a7195d8a9d8696e0_Nd_pub_5a68dc5665_3_0.1 | sys_a7195d8a9d8696e0 | TEHDGA | Nd | pub_5a68dc5665 | 3 | 0.1 | 6 | 2.3 | 9.7 | 0.273 | -0.773 | 0.273 | 1.046 | True |

Loading-active subset (range >= 0.3): ls_sys_07ee9637c98c1e20_Ce_pub_917a4583d4_3_0.1, ls_sys_5cb78e5000d40860_Nd_pub_5a68dc5665_3_0.1, ls_sys_740f07a521006be1_Nd_pub_5a68dc5665_3_0.1, ls_sys_81bcc3c06cbf0b4c_Nd_pub_0e7f3e0563_1_0.2, ls_sys_81bcc3c06cbf0b4c_Nd_pub_0e7f3e0563_3_0.2, ls_sys_a7195d8a9d8696e0_Nd_pub_15b174237f_1_0.2, ls_sys_a7195d8a9d8696e0_Nd_pub_5a68dc5665_3_0.1.

Publication-blind series (sensitivity X5): lsb_sys_07ee9637c98c1e20_Ce_3_0.1, lsb_sys_5cb78e5000d40860_Eu_0.5_0.1, lsb_sys_5cb78e5000d40860_Eu_1_0.1, lsb_sys_5cb78e5000d40860_Nd_3_0.1, lsb_sys_5cb78e5000d40860_Nd_3_0.2, lsb_sys_5cb78e5000d40860_Nd_3_0.3, lsb_sys_740f07a521006be1_Nd_3_0.1, lsb_sys_81bcc3c06cbf0b4c_Nd_1_0.2, lsb_sys_81bcc3c06cbf0b4c_Nd_3_0.2, lsb_sys_a7195d8a9d8696e0_Nd_1_0.2, lsb_sys_a7195d8a9d8696e0_Nd_3_0.1.

### 5. Replicate floor

282 replicate groups with >= 2 rows; median within-group sd of log D **0.2251** (all rows) / 0.2267 (fit-eligible rows). This is the floor below which a log D difference has no process consequence (R1 margin 0.05 is inside it).

## 3. Pre-registered D-model result (E1, R1, E3)

Regimes: cohort E1, hold-out leave-one-publication-out within system, unit = system (points -> (publication, metal) -> system -> macro); in-sample and offset-calibrated variants are labelled in the tables.

### E1 macro (M1, B0, B1, labelled variants)

*regime: cohort=E1 cohort (14 systems); holdout=LOPO by publication (rows labelled in_sample are in-sample); averaging_unit=system (macro), point-weighted column labelled*

| arm | holdout | model | e1_macro | e1_point_weighted | n_systems |
|---|---|---|---|---|---|
| primary | lopo | M1 | 1.09 | 1.04 | 14 |
| primary | lopo | B0 | 1.183 | 1.201 | 14 |
| primary | lopo | B1 | 1.064 | 1.1 | 14 |
| primary | lopo | M1_offset | 1.255 | 1.11 | 14 |
| primary | lopo | B1_crossmetal | 1.109 | 1.19 | 14 |
| primary | in_sample | M1 | 0.2948 | 0.3755 | 14 |
| primary | in_sample | B0 | 0.6128 | 0.9161 | 14 |
| primary | in_sample | B1 | 0.09086 | 0.04757 | 14 |
| primary | in_sample | M1_offset | 0.4141 | 0.5721 | 14 |
| primary | in_sample | B1_crossmetal | 0.6796 | 0.4842 | 14 |
| X2_metal_specific_slopes | lopo | M1 | 1.49 | 1.652 | 14 |
| X2_metal_specific_slopes | lopo | B0 | 1.183 | 1.201 | 14 |
| X2_metal_specific_slopes | lopo | B1 | 1.064 | 1.1 | 14 |
| X2_metal_specific_slopes | lopo | M1_offset | 1.487 | 1.659 | 14 |
| X2_metal_specific_slopes | lopo | B1_crossmetal | 1.109 | 1.19 | 14 |
| X4_without_tied_d | lopo | M1 | 1.094 | 1.021 | 14 |
| X4_without_tied_d | lopo | B0 | 1.161 | 1.179 | 14 |
| X4_without_tied_d | lopo | B1 | 1.055 | 1.078 | 14 |
| X4_without_tied_d | lopo | M1_offset | 1.262 | 1.107 | 14 |
| X4_without_tied_d | lopo | B1_crossmetal | 1.102 | 1.176 | 14 |
| X6_diluent_n_dodecane | lopo | M1 | 0.7641 | 1.077 | 1 |
| X6_diluent_n_dodecane | lopo | B0 | 1.092 | 1.032 | 1 |
| X6_diluent_n_dodecane | lopo | B1 | 0.6252 | 0.8777 | 1 |
| X6_diluent_n_dodecane | lopo | M1_offset | 0.7814 | 0.7566 | 1 |
| X6_diluent_n_dodecane | lopo | B1_crossmetal | 1.401 | 1.5 | 1 |
| X6_diluent_other_aliphatic | lopo | M1 | 0.774 | 0.7508 | 1 |
| X6_diluent_other_aliphatic | lopo | B0 | 0.7551 | 1.071 | 1 |
| X6_diluent_other_aliphatic | lopo | B1 | 0.7911 | 0.9658 | 1 |
| X6_diluent_other_aliphatic | lopo | M1_offset | 0.3975 | 0.5588 | 1 |
| X6_diluent_other_aliphatic | lopo | B1_crossmetal | 0.9128 | 0.7815 | 1 |


### E1 per system (LOPO, primary arm)

*regime: cohort=E1 cohort (14 systems); holdout=LOPO by publication; averaging_unit=system*

| arm | holdout | system_id | mae_M1 | mae_B0 | mae_B1 | mae_M1_offset | mae_B1_crossmetal | n_pairs | n_points |
|---|---|---|---|---|---|---|---|---|---|
| primary | lopo | sys_07ee9637c98c1e20 | 0.5057 | 0.8372 | 1.132 | 0.6238 | 0.8686 | 33 | 245 |
| primary | lopo | sys_120bb57e9148dd0d | 1.401 | 1.491 | 1.438 | 1.181 | 1.275 | 2 | 13 |
| primary | lopo | sys_1419f400c83e9ad8 | 0.4411 | 1.258 | 0.9704 | 0.1928 | 1.173 | 2 | 18 |
| primary | lopo | sys_267144068d8e09d2 | 0.4172 | 1.819 | 0.9276 | 0.3371 | 0.6733 | 2 | 56 |
| primary | lopo | sys_5cb78e5000d40860 | 0.8336 | 1.155 | 0.7462 | 0.8357 | 1.331 | 113 | 402 |
| primary | lopo | sys_81bcc3c06cbf0b4c | 1.38 | 1.258 | 1.096 | 0.9251 | 1.096 | 3 | 28 |
| primary | lopo | sys_81e3169a7f85c2b9 | 0.9435 | 0.4135 | 0.6409 | 1.676 | 0.5277 | 18 | 148 |
| primary | lopo | sys_95746e54b97ae741 | 1.301 | 1.031 | 1.351 | 6.19 | 1.568 | 2 | 7 |
| primary | lopo | sys_a2a472b5124b09c6 | 1.001 | 1.278 | 1.742 | 1.473 | 1.34 | 43 | 263 |
| primary | lopo | sys_a7195d8a9d8696e0 | 0.7027 | 1.297 | 0.5837 | 0.6155 | 0.8052 | 8 | 69 |
| primary | lopo | sys_a9fea6791c7a14d6 | 1.795 | 1.328 | 1.404 | 0.4009 | 1.489 | 4 | 40 |
| primary | lopo | sys_b06c95ac0efc3d4e | 2.346 | 0.8974 | 0.9687 | 0.8868 | 0.8165 | 2 | 14 |
| primary | lopo | sys_be340fe5092ae17a | 0.7222 | 1.146 | 0.9268 | 1.171 | 0.9951 | 5 | 15 |
| primary | lopo | sys_ebce91a448faa5c3 | 1.475 | 1.354 | 0.9641 | 1.063 | 1.571 | 7 | 27 |


### R1 verdict (verbatim from `results/eval/DECISION.md`)

**the result is a null: the process chain uses B1 (`NearestConditionD`, with the ideal depletion correction and the prior `n0`) as its default D source for corpus systems, M1 becomes an exploratory option with its LOPO table shown, and the report says so verbatim. A null is a result of this generation, not a failure of it.**

Conditions: {'i_margin_vs_B1': False, 'i_vs_B0': True, 'ii_bootstrap_excludes_zero': False, 'iii_wins': False}; E1 macro {'B0': 1.1830348722650128, 'B1': 1.0637059097018255, 'B1_crossmetal': 1.109312208301117, 'M1': 1.090376578070656, 'M1_offset': 1.255069440987682}; bootstrap of E1(B1) - E1(M1): {'hi': 0.24112621449465652, 'lo': -0.3170296076436446, 'mean': -0.02667066836883027, 'n_boot': 2000, 'seed': 18}; wins 7/14 (needed 9). The chain's default D source for corpus systems is therefore **nearest (B1)**.

### E3 reliability (jackknife by publication; split-half by publication where >= 4 publications; `interpretable` = jackknife SE < 0.5 and sign agreement >= 18/20)

*regime: cohort=E1 cohort (14 systems); holdout=jackknife by publication, split-half by publication (20 repeats, seed 18) where >= 4 publications; averaging_unit=system; status_of_parameters=fitted_from_corpus*

| system_id | ligand | source | n | p_eff | jackknife_se_n | jackknife_se_p_eff | split_half_defined | split_half_n_agree | split_half_p_agree | split_half_repeats | split_half_intercept_r_median | slope_status_n | slope_status_p_eff | interpretable_n | interpretable_p_eff | n_points | n_publications | todga_n_range | todga_n_in_range |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sys_07ee9637c98c1e20 | TODGA | sys_07ee9637c98c1e20.json | 1.408 | 2.633 | 2.317 | 0.725 | True | 10 | 20 | 20 | 0.9779 | fitted | fitted | False | False | 246 | 7 | nan | nan |
| sys_120bb57e9148dd0d | TEHDGA | sys_120bb57e9148dd0d.json | 3 | 0.7717 | 0 | 0.6039 | False | nan | nan | nan | nan | assumed | fitted | False | False | 31 | 2 | nan | nan |
| sys_1419f400c83e9ad8 | DMDODGA | sys_1419f400c83e9ad8.json | 2.597 | 1.524 | 0.1944 | 0.2378 | False | nan | nan | nan | nan | fitted | fitted | True | True | 239 | 2 | nan | nan |
| sys_267144068d8e09d2 | DMDODGA | sys_267144068d8e09d2.json | 2.222 | 1.296 | 0.3897 | 0.1501 | False | nan | nan | nan | nan | fitted | fitted | True | True | 91 | 2 | nan | nan |
| sys_5cb78e5000d40860 | TODGA | sys_5cb78e5000d40860.json | 2.453 | 2.101 | 0.5431 | 0.2916 | True | 20 | 20 | 20 | 0.9317 | fitted | fitted | False | True | 402 | 27 | [2.36, 2.88] | True |
| sys_81bcc3c06cbf0b4c | D3DODGA | sys_81bcc3c06cbf0b4c.json | 3 | -0.1271 | 0 | 1.297 | False | nan | nan | nan | nan | assumed | fitted | False | False | 28 | 3 | nan | nan |
| sys_81e3169a7f85c2b9 | TBDGA | sys_81e3169a7f85c2b9.json | 2.415 | 3.138 | 0.4289 | 4.164 | True | 20 | 20 | 20 | 0.9428 | fitted | fitted | True | False | 173 | 4 | nan | nan |
| sys_95746e54b97ae741 | NTAamide(C8) | sys_95746e54b97ae741.json | 3 | -1.034 | 0 | 1.517 | False | nan | nan | nan | nan | assumed | fitted | False | False | 20 | 2 | nan | nan |
| sys_a2a472b5124b09c6 | TODGA | sys_a2a472b5124b09c6.json | 3.759 | 2.744 | 0.4842 | 1.17 | True | 15 | 19 | 20 | 0.4043 | fitted | fitted | False | False | 263 | 8 | nan | nan |
| sys_a7195d8a9d8696e0 | TEHDGA | sys_a7195d8a9d8696e0.json | 3.024 | 3.329 | 0.2523 | 0.4035 | True | 20 | 20 | 20 | nan | fitted | fitted | True | True | 69 | 8 | nan | nan |
| sys_a9fea6791c7a14d6 | DMDOHEMA | sys_a9fea6791c7a14d6.json | 2.502 | 2.697 | 0.2761 | 0.3529 | False | nan | nan | nan | nan | fitted | fitted | True | True | 52 | 2 | nan | nan |
| sys_b06c95ac0efc3d4e | C5BTBP | sys_b06c95ac0efc3d4e.json | 1.284 | 2 | 0.8579 | 0 | False | nan | nan | nan | nan | fitted | assumed | False | False | 26 | 2 | nan | nan |
| sys_be340fe5092ae17a | 2-[1-(dioctylamino)-1-oxopropan-2-yl]oxy-N-N-dioctylpropanamide | sys_be340fe5092ae17a.json | 3.066 | 1.544 | 0.03855 | 0.3518 | True | 20 | 20 | 20 | nan | fitted | fitted | True | True | 27 | 4 | nan | nan |
| sys_ebce91a448faa5c3 | TODGA | sys_ebce91a448faa5c3.json | 2.182 | 1.078 | 0.8303 | 1.437 | True | 15 | 19 | 20 | nan | fitted | fitted | False | False | 39 | 6 | nan | nan |


TODGA (`sys_5cb78e5000d40860`, band 20-30C) in-sample pooled n = 2.4532 (jackknife SE 0.5430978633064496) against the corpus-validated interval [2.36, 2.88]: **inside**. The `validation`-marked test `tests/test_todga_slope.py` asserts the interval; its outcome is recorded by the orchestrator's validation run.

## 4. Loading (E2, R2)

Regime: cohort = publication-aware loading series; hold-out = the tracer point anchors log K, the other points of the same series are scored; unit = series (weight 1); O/A = 1 assumed (`OA_ASSUMED`), K_H unknown.

### Per series (O/A 1)

*regime: cohort=E2 cohort: publication-aware loading series; holdout=tracer point anchors log K, scored on the other points of the same series (oa = 1 primary, other oa secondary); averaging_unit=series; status_of_parameters=n from the in-sample M1 fit or prior n0, log K anchored per series*

| loading_series_id | metal | n_points | log_d_tracer | log_d_range | loading_active | n_used | n_source | anchor_status | mae_constant | mae_ideal | c1_wins | spearman_logD_vs_logmM | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ls_sys_07ee9637c98c1e20_Ce_pub_917a4583d4_3_0.1 | Ce | 8 | 1 | 0.6502 | True | 1.408 | m1_in_sample | anchored | 0.4442 | 0.5358 | False | 0.2381 | ACID_UPTAKE_UNMODELLED\|HIGH_LOADING\|PHASE_BEHAVIOUR_UNKNOWN |
| ls_sys_5cb78e5000d40860_Nd_pub_5a68dc5665_3_0.1 | Nd | 6 | 1.371 | 1.593 | True | 2.453 | m1_in_sample | anchored | 0.8593 | 0.7558 | True | -0.9429 | ACID_UPTAKE_UNMODELLED |
| ls_sys_5cb78e5000d40860_Nd_pub_d3c970567f_3_0.1 | Nd | 4 | -0.03218 | 0.01259 | False | 2.453 | m1_in_sample | anchored | 0.009141 | 0.02986 | False | 1 | ACID_UPTAKE_UNMODELLED |
| ls_sys_5cb78e5000d40860_Nd_pub_d3c970567f_3_0.2 | Nd | 4 | -0.02298 | 0.00719 | False | 2.453 | m1_in_sample | anchored | 0.003436 | 0.01233 | False | -0.8 | ACID_UPTAKE_UNMODELLED |
| ls_sys_5cb78e5000d40860_Nd_pub_d3c970567f_3_0.3 | Nd | 8 | -0.03152 | 0.03421 | False | 2.453 | m1_in_sample | anchored | 0.01887 | 0.03922 | False | 0.9286 | ACID_UPTAKE_UNMODELLED |
| ls_sys_740f07a521006be1_Nd_pub_5a68dc5665_3_0.1 | Nd | 6 | 0.9542 | 0.3522 | True | 3 | prior_n0 | anchored | 0.2008 | 0.1141 | True | -0.8857 | ACID_UPTAKE_UNMODELLED\|HIGH_LOADING\|PHASE_BEHAVIOUR_UNKNOWN |
| ls_sys_81bcc3c06cbf0b4c_Nd_pub_0e7f3e0563_1_0.2 | Nd | 18 | 0.7134 | 2.428 | True | 3 | prior_n0 | anchored | 0.6392 | 0.3708 | True | -0.9752 | HIGH_LOADING\|PHASE_BEHAVIOUR_UNKNOWN |
| ls_sys_81bcc3c06cbf0b4c_Nd_pub_0e7f3e0563_3_0.2 | Nd | 8 | 1.602 | 2.161 | True | 3 | prior_n0 | anchored | 1.284 | 0.1145 | True | -1 | ACID_UPTAKE_UNMODELLED\|HIGH_LOADING\|PHASE_BEHAVIOUR_UNKNOWN |
| ls_sys_a7195d8a9d8696e0_Nd_pub_15b174237f_1_0.2 | Nd | 3 | 1.172 | 1.045 | True | 3.024 | m1_in_sample | anchored | 0.8935 | 0.3978 | True | -1 | PHASE_BEHAVIOUR_UNKNOWN |
| ls_sys_a7195d8a9d8696e0_Nd_pub_5a68dc5665_3_0.1 | Nd | 6 | 0.273 | 1.046 | True | 3.024 | m1_in_sample | anchored | 0.7849 | 0.6772 | True | -1 | ACID_UPTAKE_UNMODELLED |


### Summary (all series; loading-active subset; O/A sensitivity; publication-blind)

*regime: cohort=E2 cohort: publication-aware loading series; holdout=as e2_series.csv; averaging_unit=series (macro, weight 1 each), paired series bootstrap (2000 resamples, seed 18)*

| subset | oa | n_series | e2_c0_macro | e2_c1_macro | wins_c1 | wins_needed | bootstrap_mean_c0_minus_c1 | bootstrap_lo | bootstrap_hi | condition_i_majority | condition_ii_ci_excludes_zero | c1_supported | spearman_negative_count |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 0.5 | 10 | 0.5137 | 0.3115 | 6 | 6 | 0.2022 | 0.01779 | 0.4533 | True | True | True | 7 |
| loading_active | 0.5 | 7 | 0.7294 | 0.4314 | 6 | 4 | 0.298 | 0.04348 | 0.5915 | True | True | True | 6 |
| all | 1 | 10 | 0.5137 | 0.3047 | 6 | 6 | 0.209 | 0.02959 | 0.4739 | True | True | True | 7 |
| loading_active | 1 | 7 | 0.7294 | 0.4237 | 6 | 4 | 0.3057 | 0.06769 | 0.617 | True | True | True | 6 |
| all | 2 | 10 | 0.5137 | 0.3246 | 6 | 6 | 0.1891 | 0.03517 | 0.4174 | True | True | True | 7 |
| loading_active | 2 | 7 | 0.7294 | 0.4545 | 6 | 4 | 0.2749 | 0.06709 | 0.5573 | True | True | True | 6 |
| publication_blind | 1 | 11 | 0.4502 | 0.3822 | 7 | 6 | 0.06796 | -0.02864 | 0.1795 | True | False | False | 9 |


### R2 verdict (verbatim from `results/loading/DECISION.md`)

**C1 is supported**

### Interpretation of the Sasaki 2015 TODGA/Nd series with the sourced LOC (addenda/ORCHESTRATOR_prefit_20260913.md, U6; interpretation only, no exclusion)

Series `ls_sys_5cb78e5000d40860_Nd_pub_5a68dc5665_3_0.1` (0.1 M TODGA, n-dodecane, 3 M HNO3, 4.9-12 mM Nd, D 23.5 -> 0.60) counted as it fell: mae_constant = 0.859, mae_ideal = 0.756, C1 wins = True. The sourced third-phase limit for this system is 0.008 M Nd in the organic (Tachimori, Sasaki, Suzuki 2002, doi 10.1081/SEI-120016073, abstract level). The C1 prediction puts the organic Nd above that limit at 2 of 5 non-tracer points (predicted organic Nd up to 0.0112 M); the collapse of D along this series is therefore read as a third-phase boundary, not as ligand depletion, and the cascade raises `THIRD_PHASE_RISK` above the sourced value for this system. This reading does not change R2: the series keeps its weight.


### Exploratory X1: effective capacity phi (leave-one-point-out)

*regime: cohort=E2 cohort: publication-aware loading series, series with >= 4 non-tracer points; holdout=leave-one-point-out (phi fitted on the rest, log K re-anchored per phi); averaging_unit=series; status_of_parameters=exploratory X1, never used elsewhere*

| loading_series_id | system_id | metal | n_non_tracer | mae_constant | mae_ideal | mae_phi_loo | phi_loo_n | phi_median | phi_min | phi_max | phi_insample | mae_phi_insample | loading_active | x1_beats_c1_by_margin |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ls_sys_07ee9637c98c1e20_Ce_pub_917a4583d4_3_0.1 | sys_07ee9637c98c1e20 | Ce | 7 | 0.4442 | 0.5358 | 0.5358 | 7 | 0.9999 | 0.9999 | 0.9999 | 0.9999 | 0.5358 | True | False |
| ls_sys_5cb78e5000d40860_Nd_pub_5a68dc5665_3_0.1 | sys_5cb78e5000d40860 | Nd | 5 | 0.8593 | 0.7558 | 0.3531 | 5 | 0.1813 | 0.1549 | 0.1934 | 0.1549 | 0.2667 | True | True |
| ls_sys_5cb78e5000d40860_Nd_pub_d3c970567f_3_0.3 | sys_5cb78e5000d40860 | Nd | 7 | 0.01887 | 0.03922 | 0.03922 | 7 | 0.9999 | 0.9999 | 0.9999 | 0.9999 | 0.03922 | False | False |
| ls_sys_740f07a521006be1_Nd_pub_5a68dc5665_3_0.1 | sys_740f07a521006be1 | Nd | 5 | 0.2008 | 0.1141 | 0.1141 | 5 | 0.9999 | 0.9999 | 0.9999 | 0.9999 | 0.1141 | True | False |
| ls_sys_81bcc3c06cbf0b4c_Nd_pub_0e7f3e0563_1_0.2 | sys_81bcc3c06cbf0b4c | Nd | 17 | 0.6392 | 0.3708 | 0.3708 | 17 | 0.9999 | 0.9999 | 0.9999 | 0.9999 | 0.3708 | True | False |
| ls_sys_81bcc3c06cbf0b4c_Nd_pub_0e7f3e0563_3_0.2 | sys_81bcc3c06cbf0b4c | Nd | 7 | 1.284 | 0.1145 | 0.1146 | 7 | 0.9999 | 0.9999 | 0.9999 | 0.9999 | 0.1146 | True | False |
| ls_sys_a7195d8a9d8696e0_Nd_pub_5a68dc5665_3_0.1 | sys_a7195d8a9d8696e0 | Nd | 5 | 0.7849 | 0.6772 | 0.01434 | 5 | 0.04538 | 0.04538 | 0.04538 | 0.04538 | 0.01434 | True | True |


## 5. Exploratory analyses, counted and BH-adjusted

*regime: cohort=E1 cohort (14 systems); holdout=as named per contrast; averaging_unit=system (E1) / series (E2)*

| name | family | p_value | p_adjusted | adjustment | n_in_family | note |
|---|---|---|---|---|---|---|
| E1: M1 vs B0 | primary | 0.589 | nan | none | 3 | two-sided paired system bootstrap p of mean(E1(B0) - E1(M1)) = 0 |
| E1: M1 vs B1 | primary | 0.877 | nan | none | 3 | two-sided paired system bootstrap p of mean(E1(B1) - E1(M1)) = 0; R1 uses the CI |
| E3 secondary: TODGA slope check | secondary | nan | nan | none | 4 | TODGA n = 2.453 vs [2.36, 2.88]: inside |
| X2 metal-specific slopes vs pooled M1 | exploratory | 0.002 | 0.01 | benjamini_hochberg_exploratory_family | 8 | paired system bootstrap of mean(E1(M1 pooled) - E1(X2)) |
| X3 cross-metal 1-NN vs B1 | exploratory | 0.579 | 0.7237 | benjamini_hochberg_exploratory_family | 8 | paired system bootstrap of mean(E1(B1) - E1(B1_crossmetal)) |
| X4 tied-D sensitivity | exploratory | 0.46 | 0.7237 | benjamini_hochberg_exploratory_family | 8 | paired system bootstrap of mean(E1(M1 all rows) - E1(M1 without TIED_D)) |
| X6 diluent-identity split (TODGA n-dodecane vs other) | exploratory | nan | nan | benjamini_hochberg_exploratory_family | 8 | descriptive: E1 per subset in e1_macro.csv |
| X7 offset-calibrated one-measurement mode vs M1 | exploratory | 0.748 | 0.748 | benjamini_hochberg_exploratory_family | 8 | paired system bootstrap of mean(E1(M1) - E1(M1_offset)); one held-out point sets the publication effect |
| X8 Huber-loss M1 | exploratory | nan | nan | benjamini_hochberg_exploratory_family | 8 | declared in PRE_REGISTRATION section 8; not implemented in evalproto (no Huber option); counted, not run |
| E2: C1 vs C0 | primary | 0.008 | nan | none | 3 | two-sided paired series bootstrap p of mean(E2(C0) - E2(C1)) = 0; R2 uses the CI |
| E2 secondary: loading-active subset | secondary | 0.006 | nan | none | 4 | same contrast on the loading-active series |
| E2 secondary: O/A 0.5 | secondary | 0.027 | nan | none | 4 | same contrast at O/A 0.5 |
| E2 secondary: O/A 2 | secondary | 0 | nan | none | 4 | same contrast at O/A 2.0 |
| X1 effective capacity phi | exploratory | nan | nan | benjamini_hochberg_exploratory_family | 8 | LOO-MAE(X1) < E2(C1) - 0.1 on 2 loading-active series; supported = False |
| X5 publication-blind loading series | exploratory | 0.184 | 0.46 | benjamini_hochberg_exploratory_family | 8 | E2 on the publication-blind definition (11 series) |


## 6. Cascade verification

Invariant tests of DESIGN.md section 12.2 (`tests/test_cascade.py`, `test_equilibrium.py`, `test_metrics.py`): Kremser oracle in the constant-D limit (abs 1e-10), analytic Jacobian versus finite differences (rel 1e-6), Newton versus successive substitution on every stream quantity (rel 1e-8), ledgers recomputed from the stream table (rel 1e-8), multi-start stage agreement (rel 1e-9), origin-pass identity (rel 1e-10), failure as a status. Their outcome is the fast suite's (`pytest -m "not slow and not validation"`).

Timing (`results/bench/timing.json`, **the only wall-clock numbers of this generation**; reference cascade {'n_ext': 6, 'n_ligands': 1, 'n_metals': 2, 'n_scr': 3, 'n_str': 3}, 84 unknowns): stage solve 0.60 ms, Newton cascade 15.4 ms (11 iterations), successive substitution 663 ms (115 sweeps), one LHS evaluation 16.7 ms. The naive projection from that cost, 36 min for 64 x 2 x 1000 cascades, is **not** the case study's cost: the reference cascade is 6/3/3, while the case searches 1-40 stages per section and measured **252 ms per cascade**, i.e. about 9 h for the specified run. It was run at 250 LHS per draw instead of 1000 (`addenda/INTEGRATION.md` section 3).

## 7. Pr/Nd case: PC88A versus Cyanex 272

Regime: feed `assumed` (U1), parameters `ASSUMED_PLACEHOLDER` with ranges (U4), draws 64, seed 18, stage bounds 1-40; consistency checks are not validation.

### Parameter status

*regime: cohort=Pr/Nd case: literature placeholder entries PC88A and Cyanex 272 (no corpus rows), feed_status assumed (U1); holdout=none (computed regimes, consistency checks are not validation); averaging_unit=draw (parameter draw, seed 18); status_of_parameters=PLACEHOLDER_PARAMETERS (assumed with ranges, --allow-placeholders), feed_status assumed, spec_status assumed*

| system_id | parameter | value | unit | status | range | source | note |
|---|---|---|---|---|---|---|---|
| sys_29976921e156a0a0 | organic_ligands[0].concentration | 0.8 | mol/L | assumed | [0.2, 1.5] | 10.1038/s41598-020-74041-9 \| extractant concentration series, 0.8 mol/L point | formal monomer concentration used in S2; the case sweeps the range as a design variable |
| sys_29976921e156a0a0 | organic_ligands[0].stoichiometry.ligands_per_metal | 3 | 1 | assumed | [3.0, 3.0] | 10.1038/s41598-020-74041-9 \| mechanism nRE3+ + n(HA2)org | dimers per Ln3+, ideal dilute-regime stoichiometry |
| sys_29976921e156a0a0 | organic_ligands[0].stoichiometry.protons_released_per_metal | 3 | 1 | assumed | [3.0, 3.0] | 10.1038/s41598-020-74041-9 \| mechanism |  |
| sys_29976921e156a0a0 | organic_ligands[0].stoichiometry.anions_per_metal | 0 | 1 | assumed | [0.0, 0.0] | none \| cation exchange transports no anion |  |
| sys_29976921e156a0a0 | medium.salting_anion_M |  | mol/L | unknown |  | none |  |
| sys_29976921e156a0a0 | medium.ionic_strength_M |  | mol/L | unknown |  | none |  |
| sys_29976921e156a0a0 | medium.temperature_C | 25 | Cel | assumed | [20.0, 30.0] | 10.1038/s41598-020-74041-9 \| 298 K |  |
| sys_29976921e156a0a0 | params.PC88A.20-30C.log_k.Nd | -1.95 | 1 | assumed | [-2.9, -1.0] | 10.1038/s41598-020-74041-9 \| Nd 64 % extracted at 0.8 M PC88A, initial pH 4.0, equilibrium pH 1.02-1.42, 1500 mg/L each Nd/Tb/Dy, A/O 1 | derive_logk_from_extraction: log D_Nd = 0.250; sum [M]_org = 24.3 mM; [(HA)2]_f = 0.400 - 3*0.0243 = 0.327 M; a = b = 3; pH_eq in [1.02, 1.42] gives [-2.55, -1.35]; widened by 0.3 for exponent uncertainty; with b = 2.22 (S2 slope) the window is [-1.45, -0.56]. Verify against Banda 2014 doi 10.1016/j.jiec.2014.03.002 and Thakur 1993 doi 10.1016/0304-386X(93)90084-Q. |
| sys_29976921e156a0a0 | params.PC88A.20-30C.log_k.Pr | -2.1 | 1 | assumed | [-3.08, -1.11] | 10.1016/j.jiec.2014.03.002 \| maximum SF(Nd/Pr) about 1.5 (task document); patent EP2388344A1: PC-88A SF(Nd/Pr) 1.4 in kerosene | log K_Pr = log K_Nd - log10 SF(Nd/Pr), SF placeholder range [1.3, 1.5] (log 0.114-0.176), central 1.4 (log 0.146) |
| sys_29976921e156a0a0 | params.PC88A.20-30C.delta_h_kj_mol |  | kJ/mol | unknown |  | 10.1038/s41598-020-74041-9 \| thermodynamics section: endothermic, values not transcribed |  |
| sys_29976921e156a0a0 | params.PC88A.20-30C.a_dimer | 3 | 1 | assumed | [2.0, 3.0] | 10.1038/s41598-020-74041-9 \| log D vs log[extractant] slopes 2-3 | ideal 3 |
| sys_29976921e156a0a0 | params.PC88A.20-30C.b_proton | 3 | 1 | assumed | [2.0, 3.0] | 10.1038/s41598-020-74041-9 \| log D vs pH slope, PC 88A Nd 2.22 | ideal 3; S2 measured 2.22 for Nd |
| sys_29976921e156a0a0 | phase.loc_metal_M |  | mol/L | unknown |  | none |  |
| sys_29976921e156a0a0 | phase.loc_acid_M |  | mol/L | unknown |  | none |  |
| sys_29976921e156a0a0 | phase.third_phase_observed |  | 1 | unknown |  | none |  |
| sys_29976921e156a0a0 | phase.disengagement_s |  | s | unknown |  | none |  |
| sys_29976921e156a0a0 | phase.ligand_loss_mol_per_L_aq |  | mol/L_aq | unknown |  | none |  |
| sys_29976921e156a0a0 | phase.max_loading_fraction_studied |  | 1 | unknown |  | none |  |
| sys_f02db527a94a5e86 | organic_ligands[0].concentration | 0.8 | mol/L | assumed | [0.2, 1.5] | 10.1038/s41598-020-74041-9 \| extractant concentration series, 0.8 mol/L point | formal monomer concentration used in S2; the case sweeps the range as a design variable |
| sys_f02db527a94a5e86 | organic_ligands[0].stoichiometry.ligands_per_metal | 3 | 1 | assumed | [3.0, 3.0] | 10.1038/s41598-020-74041-9 \| mechanism nRE3+ + n(HA2)org | dimers per Ln3+, ideal dilute-regime stoichiometry |
| sys_f02db527a94a5e86 | organic_ligands[0].stoichiometry.protons_released_per_metal | 3 | 1 | assumed | [3.0, 3.0] | 10.1038/s41598-020-74041-9 \| mechanism |  |
| sys_f02db527a94a5e86 | organic_ligands[0].stoichiometry.anions_per_metal | 0 | 1 | assumed | [0.0, 0.0] | none \| cation exchange transports no anion |  |
| sys_f02db527a94a5e86 | medium.salting_anion_M |  | mol/L | unknown |  | none |  |
| sys_f02db527a94a5e86 | medium.ionic_strength_M |  | mol/L | unknown |  | none |  |
| sys_f02db527a94a5e86 | medium.temperature_C | 25 | Cel | assumed | [20.0, 30.0] | 10.1038/s41598-020-74041-9 \| 298 K |  |
| sys_f02db527a94a5e86 | params.Cyanex 272.20-30C.log_k.Nd | -3.45 | 1 | assumed | [-4.5, -2.4] | 10.1038/s41598-020-74041-9 \| Nd 27 % extracted at 0.8 M Cyanex 272, initial pH 4.0, equilibrium pH 1.22-1.70, 1500 mg/L each Nd/Tb/Dy, A/O 1 | derive_logk_from_extraction: log D_Nd = -0.432; sum [M]_org = 15.6 mM; [(HA)2]_f = 0.400 - 3*0.0156 = 0.353 M; a = b = 3; pH_eq in [1.22, 1.70] gives [-4.18, -2.74]; widened by 0.3 for exponent uncertainty; with b = 2.0 (S2 slope) the window is [-2.48, -1.52]. |
| sys_f02db527a94a5e86 | params.Cyanex 272.20-30C.log_k.Pr | -3.55 | 1 | assumed | [-4.65, -2.36] | none \| SF(Nd/Pr) assumed sweep {1.1, 1.2, 1.3, 1.4}; no source found | log K_Pr = log K_Nd - log10 SF(Nd/Pr) with SF an assumed sweep {1.1, 1.2, 1.3, 1.4} (log 0.041-0.146), central 1.25 (log 0.097); no Cyanex 272 Nd/Pr value found on 2026-09-13; candidate sources doi 10.1016/j.hydromet.2014.09.015, doi 10.1016/j.jre.2017.09.016, doi 10.1016/j.mineng.2013.10.021 |
| sys_f02db527a94a5e86 | params.Cyanex 272.20-30C.delta_h_kj_mol |  | kJ/mol | unknown |  | 10.1038/s41598-020-74041-9 \| thermodynamics section: endothermic, values not transcribed |  |
| sys_f02db527a94a5e86 | params.Cyanex 272.20-30C.a_dimer | 3 | 1 | assumed | [2.0, 3.0] | 10.1038/s41598-020-74041-9 \| log D vs log[extractant] slopes 2-3 | ideal 3 |
| sys_f02db527a94a5e86 | params.Cyanex 272.20-30C.b_proton | 3 | 1 | assumed | [2.0, 3.0] | 10.1038/s41598-020-74041-9 \| log D vs pH slope, Cyanex 272 Nd 2.0 | ideal 3; S2 measured 2.0 for Nd |
| sys_f02db527a94a5e86 | phase.loc_metal_M |  | mol/L | unknown |  | none |  |
| sys_f02db527a94a5e86 | phase.loc_acid_M |  | mol/L | unknown |  | none |  |
| sys_f02db527a94a5e86 | phase.third_phase_observed |  | 1 | unknown |  | none |  |
| sys_f02db527a94a5e86 | phase.disengagement_s |  | s | unknown |  | none |  |
| sys_f02db527a94a5e86 | phase.ligand_loss_mol_per_L_aq |  | mol/L_aq | unknown |  | none |  |
| sys_f02db527a94a5e86 | phase.max_loading_fraction_studied |  | 1 | unknown |  | none |  |
| feed | feed.flow_L_h | 1 | L/h | assumed | [0.1, 10.0] | none \| DESIGN.md 13.1 default; basis flow, results scale linearly |  |
| feed | feed.metals_mM.Nd | 75 | mmol/L | assumed | [7.5, 225.0] | none \| DESIGN.md 13.1: Nd:Pr 3:1 mol, total 0.1 M | range spans the sensitivity sets (total 0.03-0.3 M, ratio 1:1-3:1) |
| feed | feed.metals_mM.Pr | 25 | mmol/L | assumed | [7.5, 150.0] | none \| DESIGN.md 13.1: Nd:Pr 3:1 mol, total 0.1 M | range spans the sensitivity sets (total 0.03-0.3 M, ratio 1:1-3:1) |
| feed | feed.h_M | 0.01 | mol/L | assumed | [0.001, 0.1] | none \| DESIGN.md 13.1: [H+] 0.01 M (pH 2) | aqueous [H+] of the feed; the cascade computes the equilibrium acidity per stage |
| feed | feed.anion_M | 0.31 | mol/L | assumed | [0.1, 1.0] | none \| DESIGN.md 13.1: chloride 0.31 M = 3 x 0.1 M metal + 0.01 M HCl | total chloride of the feed |
| feed | feed.complexant_total_M | 0 | mol/L | assumed | [0.0, 0.0] | none \| no aqueous complexant in the PC88A / Cyanex 272 case |  |
| feed | feed.sodium_M | 0 | mol/L | assumed | [0.0, 0.0] | none \| no sodium in the feed; saponification adds it in the organic reserve |  |
| feed | feed.temperature_C | 25 | Cel | assumed | [20.0, 30.0] | none \| DESIGN.md 13.1: 25 C (band 20-30C of the literature entries) |  |
| feed | feed.sensitivity.total_metal_M | [0.03, 0.1, 0.3] | mol/L | assumed | [0.03, 0.3] | none \| DESIGN.md 13.1 sensitivity sets | Nd:Pr ratio held at the feed ratio; chloride = 3 x total metal + [H+] |
| feed | feed.sensitivity.ratio_nd_pr | ['1:1', '3:1'] | 1 | assumed | [1.0, 3.0] | none \| DESIGN.md 13.1 sensitivity sets | mol Nd per mol Pr at the feed total |
| spec | spec.purity_min_grid | [0.95, 0.97, 0.99] | 1 | assumed | [0.95, 0.99] | none \| DESIGN.md 13.1 grid | fraction of Nd among metals in the product, mol basis |
| spec | spec.recovery_min_grid | [0.8, 0.85, 0.9] | 1 | assumed | [0.8, 0.9] | none \| DESIGN.md 13.1 grid | recovery_from_feed (origin-labelled, section 7.8), never recovery_total |


### Interval tables per spec cell (min / median / max over draws)

*regime: cohort=Pr/Nd case: literature placeholder entries PC88A and Cyanex 272 (no corpus rows), feed_status assumed (U1); holdout=none (computed regimes, consistency checks are not validation); averaging_unit=min / median / max over draws; status_of_parameters=PLACEHOLDER_PARAMETERS (assumed with ranges, --allow-placeholders), feed_status assumed, spec_status assumed*

| system_id | subset | purity_min | recovery_min | draws_reachable | n_draws | n_stages_total_min | n_stages_total_median | n_stages_total_max | oa_ext_median | s_over_a_median | w_over_a_median | scrub_acid_M_median | strip_acid_M_median | saponification_degree_median | acid_mol_per_kg_oxide_median | base_mol_per_kg_oxide_median | consumption_index_median | regime_status_over_draws |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sys_29976921e156a0a0 | all | 0.95 | 0.8 | 0 | 64 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |  |
| sys_29976921e156a0a0 | all | 0.95 | 0.85 | 0 | 64 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |  |
| sys_29976921e156a0a0 | all | 0.95 | 0.9 | 0 | 64 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |  |
| sys_29976921e156a0a0 | all | 0.97 | 0.8 | 0 | 64 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |  |
| sys_29976921e156a0a0 | all | 0.97 | 0.85 | 0 | 64 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |  |
| sys_29976921e156a0a0 | all | 0.97 | 0.9 | 0 | 64 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |  |
| sys_29976921e156a0a0 | all | 0.99 | 0.8 | 0 | 64 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |  |
| sys_29976921e156a0a0 | all | 0.99 | 0.85 | 0 | 64 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |  |
| sys_29976921e156a0a0 | all | 0.99 | 0.9 | 0 | 64 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |  |
| sys_29976921e156a0a0 | in_domain_only | 0.95 | 0.8 | 0 | 64 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |  |
| sys_29976921e156a0a0 | in_domain_only | 0.95 | 0.85 | 0 | 64 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |  |
| sys_29976921e156a0a0 | in_domain_only | 0.95 | 0.9 | 0 | 64 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |  |
| sys_29976921e156a0a0 | in_domain_only | 0.97 | 0.8 | 0 | 64 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |  |
| sys_29976921e156a0a0 | in_domain_only | 0.97 | 0.85 | 0 | 64 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |  |
| sys_29976921e156a0a0 | in_domain_only | 0.97 | 0.9 | 0 | 64 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |  |
| sys_29976921e156a0a0 | in_domain_only | 0.99 | 0.8 | 0 | 64 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |  |
| sys_29976921e156a0a0 | in_domain_only | 0.99 | 0.85 | 0 | 64 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |  |
| sys_29976921e156a0a0 | in_domain_only | 0.99 | 0.9 | 0 | 64 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |  |
| sys_f02db527a94a5e86 | all | 0.95 | 0.8 | 0 | 64 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |  |
| sys_f02db527a94a5e86 | all | 0.95 | 0.85 | 0 | 64 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |  |
| sys_f02db527a94a5e86 | all | 0.95 | 0.9 | 0 | 64 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |  |
| sys_f02db527a94a5e86 | all | 0.97 | 0.8 | 0 | 64 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |  |
| sys_f02db527a94a5e86 | all | 0.97 | 0.85 | 0 | 64 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |  |
| sys_f02db527a94a5e86 | all | 0.97 | 0.9 | 0 | 64 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |  |
| sys_f02db527a94a5e86 | all | 0.99 | 0.8 | 0 | 64 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |  |
| sys_f02db527a94a5e86 | all | 0.99 | 0.85 | 0 | 64 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |  |
| sys_f02db527a94a5e86 | all | 0.99 | 0.9 | 0 | 64 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |  |
| sys_f02db527a94a5e86 | in_domain_only | 0.95 | 0.8 | 0 | 64 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |  |
| sys_f02db527a94a5e86 | in_domain_only | 0.95 | 0.85 | 0 | 64 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |  |
| sys_f02db527a94a5e86 | in_domain_only | 0.95 | 0.9 | 0 | 64 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |  |
| sys_f02db527a94a5e86 | in_domain_only | 0.97 | 0.8 | 0 | 64 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |  |
| sys_f02db527a94a5e86 | in_domain_only | 0.97 | 0.85 | 0 | 64 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |  |
| sys_f02db527a94a5e86 | in_domain_only | 0.97 | 0.9 | 0 | 64 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |  |
| sys_f02db527a94a5e86 | in_domain_only | 0.99 | 0.8 | 0 | 64 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |  |
| sys_f02db527a94a5e86 | in_domain_only | 0.99 | 0.85 | 0 | 64 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |  |
| sys_f02db527a94a5e86 | in_domain_only | 0.99 | 0.9 | 0 | 64 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |  |


### Sensitivities (one factor at a time at the median draw, reference regime)

*regime: cohort=Pr/Nd case: literature placeholder entries PC88A and Cyanex 272 (no corpus rows), feed_status assumed (U1); holdout=none (computed regimes, consistency checks are not validation); averaging_unit=one factor at a time, median draw; status_of_parameters=PLACEHOLDER_PARAMETERS (assumed with ranges, --allow-placeholders), feed_status assumed, spec_status assumed*

| system_id | factor | value | status | purity_mol | recovery_from_feed | consumption_index | regime_status |
|---|---|---|---|---|---|---|---|
| sys_29976921e156a0a0 | reference | median draw | converged_newton | 0.8164 | 1 | 252.5 | INADMISSIBLE |
| sys_29976921e156a0a0 | sf_nd_pr | 1.3 | converged_newton | 0.8164 | 1 | 252.5 | INADMISSIBLE |
| sys_29976921e156a0a0 | sf_nd_pr | 1.35 | converged_newton | 0.8164 | 1 | 252.5 | INADMISSIBLE |
| sys_29976921e156a0a0 | sf_nd_pr | 1.4 | converged_newton | 0.8164 | 1 | 252.5 | INADMISSIBLE |
| sys_29976921e156a0a0 | sf_nd_pr | 1.45 | converged_newton | 0.8164 | 1 | 252.5 | INADMISSIBLE |
| sys_29976921e156a0a0 | sf_nd_pr | 1.5 | converged_newton | 0.8164 | 1 | 252.5 | INADMISSIBLE |
| sys_29976921e156a0a0 | log_k_Nd (2-log window clipped to range) | -2.8964174165577377 | converged_newton | 0.8164 | 1 | 252.5 | INADMISSIBLE |
| sys_29976921e156a0a0 | log_k_Nd (2-log window clipped to range) | -2.4223130624183034 | converged_newton | 0.8164 | 1 | 252.5 | INADMISSIBLE |
| sys_29976921e156a0a0 | log_k_Nd (2-log window clipped to range) | -1.9482087082788688 | converged_newton | 0.8164 | 1 | 252.5 | INADMISSIBLE |
| sys_29976921e156a0a0 | log_k_Nd (2-log window clipped to range) | -1.4741043541394343 | converged_newton | 0.8164 | 1 | 252.5 | INADMISSIBLE |
| sys_29976921e156a0a0 | log_k_Nd (2-log window clipped to range) | -1.0 | converged_newton | 0.8164 | 1 | 252.5 | INADMISSIBLE |
| sys_29976921e156a0a0 | ligand_total_M | 0.2 | converged_newton | 1 | 1.157e-09 | 4.744e+05 | OUT_OF_DOMAIN |
| sys_29976921e156a0a0 | ligand_total_M | 0.525 | failed | nan | nan | nan | INADMISSIBLE |
| sys_29976921e156a0a0 | ligand_total_M | 0.8500000000000001 | converged_newton | 0.8164 | 1 | 222.5 | INADMISSIBLE |
| sys_29976921e156a0a0 | ligand_total_M | 1.175 | converged_newton | 0.8164 | 1 | 247.8 | INADMISSIBLE |
| sys_29976921e156a0a0 | ligand_total_M | 1.5 | converged_newton | 0.8164 | 1 | 273.1 | INADMISSIBLE |
| sys_29976921e156a0a0 | saponification_degree | 0.0 | converged_newton | 0.9667 | 0.06802 | 982.6 | OUT_OF_DOMAIN |
| sys_29976921e156a0a0 | saponification_degree | 0.3 | converged_newton | 0.8164 | 1 | 214 | INADMISSIBLE |
| sys_29976921e156a0a0 | saponification_degree | 0.5 | converged_newton | 0.8164 | 1 | 252.5 | INADMISSIBLE |
| sys_29976921e156a0a0 | oa_ext (x reference) | 0.5 | converged_newton | 0.8164 | 1 | 204.4 | INADMISSIBLE |
| sys_29976921e156a0a0 | oa_ext (x reference) | 1.0 | converged_newton | 0.8164 | 1 | 252.5 | INADMISSIBLE |
| sys_29976921e156a0a0 | oa_ext (x reference) | 2.0 | failed | nan | nan | nan | INADMISSIBLE |
| sys_29976921e156a0a0 | feed_total_metal_M | 0.03 | converged_newton | 0.8866 | 1 | 478.4 | INADMISSIBLE |
| sys_29976921e156a0a0 | feed_total_metal_M | 0.1 | converged_newton | 0.8164 | 1 | 252.5 | INADMISSIBLE |
| sys_29976921e156a0a0 | feed_total_metal_M | 0.3 | failed | nan | nan | nan | INADMISSIBLE |
| sys_29976921e156a0a0 | feed_ratio_nd_pr | 1:1 | converged_newton | 0.6328 | 1 | 325.7 | INADMISSIBLE |
| sys_29976921e156a0a0 | feed_ratio_nd_pr | 3:1 | converged_newton | 0.8164 | 1 | 252.5 | INADMISSIBLE |
| sys_f02db527a94a5e86 | reference | median draw | converged_newton | 0.8505 | 1 | 589.4 | INADMISSIBLE |
| sys_f02db527a94a5e86 | sf_nd_pr | 1.1 | converged_newton | 0.8505 | 1 | 589.4 | INADMISSIBLE |
| sys_f02db527a94a5e86 | sf_nd_pr | 1.175 | converged_newton | 0.8505 | 1 | 589.4 | INADMISSIBLE |
| sys_f02db527a94a5e86 | sf_nd_pr | 1.25 | converged_newton | 0.8505 | 1 | 589.4 | INADMISSIBLE |
| sys_f02db527a94a5e86 | sf_nd_pr | 1.325 | converged_newton | 0.8505 | 1 | 589.4 | INADMISSIBLE |
| sys_f02db527a94a5e86 | sf_nd_pr | 1.4 | converged_newton | 0.8505 | 1 | 589.4 | INADMISSIBLE |
| sys_f02db527a94a5e86 | log_k_Nd (2-log window clipped to range) | -4.263362462583089 | converged_newton | 0.8505 | 1 | 589.4 | INADMISSIBLE |
| sys_f02db527a94a5e86 | log_k_Nd (2-log window clipped to range) | -3.797521846937317 | converged_newton | 0.8505 | 1 | 589.4 | INADMISSIBLE |
| sys_f02db527a94a5e86 | log_k_Nd (2-log window clipped to range) | -3.3316812312915447 | converged_newton | 0.8505 | 1 | 589.4 | INADMISSIBLE |
| sys_f02db527a94a5e86 | log_k_Nd (2-log window clipped to range) | -2.865840615645772 | converged_newton | 0.8505 | 1 | 589.4 | INADMISSIBLE |
| sys_f02db527a94a5e86 | log_k_Nd (2-log window clipped to range) | -2.4 | converged_newton | 0.8505 | 1 | 589.4 | INADMISSIBLE |
| sys_f02db527a94a5e86 | ligand_total_M | 0.2 | converged_newton | 1 | 8.887e-15 | 2.309e+07 | OUT_OF_DOMAIN |
| sys_f02db527a94a5e86 | ligand_total_M | 0.525 | failed | nan | nan | nan | INADMISSIBLE |

*(first 40 of 54 rows; full table in the CSV)*


### Displacement scrub (all four origin quantities)

*regime: cohort=Pr/Nd case: literature placeholder entries PC88A and Cyanex 272 (no corpus rows), feed_status assumed (U1); holdout=none (computed regimes, consistency checks are not validation); averaging_unit=median draw; status_of_parameters=PLACEHOLDER_PARAMETERS (assumed with ranges, --allow-placeholders), feed_status assumed, spec_status assumed*

| system_id | value | purity_mol | recovery_from_feed | recovery_total | scrub_target_return | net_product_mol_h | regime_status |
|---|---|---|---|---|---|---|---|
| sys_29976921e156a0a0 | 0 | 0.75 | 1 | 1 | nan | 0.075 | INADMISSIBLE |
| sys_29976921e156a0a0 | 10 | 0.7674 | 1 | 1.1 | 1 | 0.075 | INADMISSIBLE |
| sys_29976921e156a0a0 | 25 | 0.7894 | 1 | 1.25 | 1 | 0.075 | INADMISSIBLE |
| sys_29976921e156a0a0 | 50 | 0.8181 | 1 | 1.5 | 1 | 0.075 | INADMISSIBLE |
| sys_f02db527a94a5e86 | 0 | 0.75 | 1 | 1 | nan | 0.075 | INADMISSIBLE |
| sys_f02db527a94a5e86 | 10 | 0.8012 | 1 | 1.343 | 1 | 0.075 | INADMISSIBLE |
| sys_f02db527a94a5e86 | 25 | 0.8479 | 1 | 1.858 | 1 | 0.075 | INADMISSIBLE |
| sys_f02db527a94a5e86 | 50 | 0.8907 | 1 | 2.716 | 1 | 0.075 | INADMISSIBLE |


### Labelled sanity limit (constant D per section, no acid balance)

*regime: cohort=Pr/Nd case: literature placeholder entries PC88A and Cyanex 272 (no corpus rows), feed_status assumed (U1); holdout=none (computed regimes, consistency checks are not validation); averaging_unit=median draw; status_of_parameters=PLACEHOLDER_PARAMETERS (assumed with ranges, --allow-placeholders), feed_status assumed, spec_status assumed*

| system_id | factor | status | purity_mol | recovery_from_feed | consumption_index | regime_status |
|---|---|---|---|---|---|---|
| sys_29976921e156a0a0 | loading-aware (mass action, acid balance) | converged_newton | 0.8164 | 1 | 252.5 | INADMISSIBLE |
| sys_29976921e156a0a0 | sanity limit: ConstantD per section at the tracer D, no acid balance | converged_newton | 0.9971 | 0.001287 | 4216 | IN_DOMAIN_WITH_CAVEATS |
| sys_f02db527a94a5e86 | loading-aware (mass action, acid balance) | converged_newton | 0.8505 | 1 | 589.4 | INADMISSIBLE |
| sys_f02db527a94a5e86 | sanity limit: ConstantD per section at the tracer D, no acid balance | converged_newton | 1 | 2.731e-07 | 1.441e+05 | IN_DOMAIN_WITH_CAVEATS |


### PC88A versus Cyanex 272 at identical feed, spec grid and price table (DESIGN.md section 13.3)

*regime: PLACEHOLDER_PARAMETERS (assumed with ranges, --allow-placeholders); feed_status assumed; spec_status assumed; feed prnd_feed.json, spec prnd_spec.json, 64 draws, seed 18, 250 LHS rows per draw, stage bounds 1-40*

**Conditional on the placeholder ranges.** Every parameter of both systems is `ASSUMED_PLACEHOLDER` (PC88A SF from EP2388344A1 / Banda 2014, Cyanex 272 SF an assumed sweep {1.1, 1.2, 1.3, 1.4} with no source found); the ranking cannot be decided until the literature values are transcribed by a person (open item U4) or measured. Draw i of PC88A is paired with draw i of Cyanex 272 (same LHS design, independent parameter draws).

| subset | purity_min | recovery_min | draws | pc88a_reachable | cyanex272_reachable | both_reachable | only_pc88a_reachable | only_cyanex272_reachable | pc88a_fewer_stages | pc88a_less_acid | fraction_pc88a_fewer_stages | fraction_pc88a_less_acid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 0.95 | 0.8 | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| all | 0.95 | 0.85 | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| all | 0.95 | 0.9 | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| all | 0.97 | 0.8 | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| all | 0.97 | 0.85 | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| all | 0.97 | 0.9 | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| all | 0.99 | 0.8 | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| all | 0.99 | 0.85 | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| all | 0.99 | 0.9 | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| in_domain_only | 0.95 | 0.8 | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| in_domain_only | 0.95 | 0.85 | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| in_domain_only | 0.95 | 0.9 | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| in_domain_only | 0.97 | 0.8 | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| in_domain_only | 0.97 | 0.85 | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| in_domain_only | 0.97 | 0.9 | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| in_domain_only | 0.99 | 0.8 | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| in_domain_only | 0.99 | 0.85 | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | nan | nan |
| in_domain_only | 0.99 | 0.9 | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | nan | nan |


### Consistency checks against the cited numbers (DESIGN.md section 13.4) -- never validation

Regime: cohort=Pr/Nd case: literature placeholder entries PC88A and Cyanex 272 (no corpus rows), feed_status assumed (U1); holdout=none (computed regimes, consistency checks are not validation); averaging_unit=draw (parameter draw, seed 18); status_of_parameters=PLACEHOLDER_PARAMETERS (assumed with ranges, --allow-placeholders), feed_status assumed, spec_status assumed

Feeds, acidities and loadings of the sources are unknown (LITERATURE_NOTES.md section 2); none of (a)-(c) can validate the model.

#### (a) Thakur 1993 (doi 10.1016/0304-386X(93)90084-Q): 97 % purity at > 85 % recovery, counter-current PC88A

Cell (0.97, 0.85), PC88A, subset all: **not reachable within the assumed window**; no draw reached the cell in the sweep.

#### (b) EP2388344A1: PC-88A Nd/Pr circuit 72 extraction + 72 scrub + 8 strip stages at SF 1.4

Stage ladder (n_ext = n_scr = N, n_str = 8, 8 LHS rows per rung, SF 1.4, median draw otherwise): **at SF 1.4 the (0.99, 0.99) cell is not reached up to n_ext = n_scr = 40 (+ 8 strip; the solver budget ends there): the stage count the model needs is of the order of 70 + 70 rather than 7 + 7**.

Analytic Fenske-type minimum at total reflux for the (0.99, 0.99) cell with SF 1.4 and feed Nd:Pr 3.00: N_min = 24.0 theoretical stages (a bound from the constant-SF ideal, not the cascade); practical countercurrent circuits need a multiple of it, which is the order of the patent's 72 + 72.

| system_id | sf_nd_pr | n_ext | n_scr | n_str | n_lhs | n_converged | n_failed | cell_reached | n_reached | best_purity_mol | best_recovery_at_purity_ge_0.99 | max_purity_recovery_product |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sys_29976921e156a0a0 | 1.4 | 3 | 3 | 8 | 8 | 8 | 0 | False | 0 | 1 | 3.693e-07 | 3.693e-07 |
| sys_29976921e156a0a0 | 1.4 | 5 | 5 | 8 | 8 | 8 | 0 | False | 0 | 1 | 2.65e-11 | 2.65e-11 |
| sys_29976921e156a0a0 | 1.4 | 7 | 7 | 8 | 8 | 8 | 0 | False | 0 | 1 | 1.701e-15 | 1.701e-15 |
| sys_29976921e156a0a0 | 1.4 | 10 | 10 | 8 | 8 | 8 | 0 | False | 0 | 1 | 8.373e-22 | 8.373e-22 |
| sys_29976921e156a0a0 | 1.4 | 15 | 15 | 8 | 8 | 8 | 0 | False | 0 | 1 | 2.497e-32 | 2.497e-32 |
| sys_29976921e156a0a0 | 1.4 | 20 | 20 | 8 | 8 | 8 | 0 | False | 0 | 1 | 7.306e-43 | 7.306e-43 |
| sys_29976921e156a0a0 | 1.4 | 30 | 30 | 8 | 8 | 6 | 2 | False | 0 | 1 | 6.313e-64 | 6.313e-64 |
| sys_29976921e156a0a0 | 1.4 | 40 | 40 | 8 | 8 | 6 | 2 | False | 0 | 1 | 5.456e-85 | 5.456e-85 |


#### (c) Banda 2014 (doi 10.1016/j.jiec.2014.03.002): maximum SF about 1.5

**enters only as the upper end of the SF placeholder range [1.3, 1.5]**; no number of Banda 2014 is reproduced or compared.


## 8. TODGA Pr/Nd nitrate recipe and the DGA + aqueous ligand exploration (labelled exploratory)

### Recipes for feed todga_prnd_feed.json (anion nitrate), spec prnd_spec.json

*regime: cohort=database systems with medium.anion == nitrate; holdout=none (computed regimes); averaging_unit=candidate; status_of_parameters=per system (column status_of_parameters); LHS n = 300, seed 18; on-spec = loosest cell {'purity_min': 0.95, 'recovery_min': 0.8}; knee = consistency cell (0.97, 0.85)*

Systems considered: 10; parameterised: 10; not_parameterised: 0. Pre-registered decision m1_adopted = False.

#### Family diglycolamide

| system_id | name | status | params_source | status_of_parameters | n_converged | n_failed | n_on_spec_loosest | n_front_in_domain | reason |
|---|---|---|---|---|---|---|---|---|---|
| sys_07ee9637c98c1e20 | TODGA in other, nitrate medium, no additive | parameterised | nearest | nearest (origin corpus; ASSUMED_PLACEHOLDER values) | 300 | 0 | 0 | 11 |  |
| sys_1419f400c83e9ad8 | DMDODGA in aliphatic hydrocarbon, nitrate medium, no additive | parameterised | nearest | nearest (origin corpus; ASSUMED_PLACEHOLDER values) | 300 | 0 | 0 | 0 |  |
| sys_5cb78e5000d40860 | TODGA in aliphatic hydrocarbon, nitrate medium, no additive | parameterised | nearest | nearest (origin corpus; ASSUMED_PLACEHOLDER values) | 299 | 1 | 0 | 15 |  |
| sys_a2a472b5124b09c6 | TODGA in aromatic, nitrate medium, no additive | parameterised | nearest | nearest (origin corpus; ASSUMED_PLACEHOLDER values) | 300 | 0 | 1 | 24 |  |
| sys_f58e3a596f8d50f9 | TDdDGA in aliphatic hydrocarbon, nitrate medium, no additive | parameterised | nearest | nearest (origin corpus) | 300 | 0 | 1 | 9 |  |


Top three on-spec regimes per system (by consumption index):

| family | front | regime_status | n_ext | n_scr | n_str | oa_ext | s_over_a | w_over_a | scrub_acid_M | strip_acid_M | ligand_total_M | saponification_degree | purity_mol | recovery_from_feed | consumption_index | n_stages_total | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diglycolamide | 1 | IN_DOMAIN_WITH_CAVEATS | 9 | 6 | 6 | 3.812 | 1.333 | 0.7819 | 1.575 | 3.678 | 0.2068 | 0 | 0.9857 | 0.984 | 263.1 | 21 | COST_INCOMPLETE\|EQUILIBRIUM_ACID_ASSUMED_NOMINAL\|HIGH_LOADING\|LIGAND_LOSS_NOT_MEASURED\|OA_ASSUMED\|PHASE_BEHAVIOUR_UNKNOWN |
| diglycolamide | 1 | OUT_OF_DOMAIN | 7 | 12 | 7 | 2.783 | 4.243 | 3.654 | 1.428 | 0.6059 | 0.2923 | 0 | 0.9714 | 0.9684 | 309.3 | 26 | COST_INCOMPLETE\|EQUILIBRIUM_ACID_ASSUMED_NOMINAL\|HIGH_LOADING\|LIGAND_LOSS_NOT_MEASURED\|OA_ASSUMED\|OOD_HULL\|OOD_LIGAND\|OOD_METAL\|PHASE_BEHAVIOUR_UNKNOWN |


Pareto knee of the consistency cell:

| family | system_id | subset | purity_min | recovery_min | reachable | n_feasible | n_stages_total | consumption_index | regime_status | flags |
|---|---|---|---|---|---|---|---|---|---|---|
| diglycolamide | sys_07ee9637c98c1e20 | all | 0.97 | 0.85 | False | 0 | nan | nan | nan | nan |
| diglycolamide | sys_07ee9637c98c1e20 | in_domain_only | 0.97 | 0.85 | False | 0 | nan | nan | nan | nan |
| diglycolamide | sys_1419f400c83e9ad8 | all | 0.97 | 0.85 | False | 0 | nan | nan | nan | nan |
| diglycolamide | sys_1419f400c83e9ad8 | in_domain_only | 0.97 | 0.85 | False | 0 | nan | nan | nan | nan |
| diglycolamide | sys_5cb78e5000d40860 | all | 0.97 | 0.85 | False | 0 | nan | nan | nan | nan |
| diglycolamide | sys_5cb78e5000d40860 | in_domain_only | 0.97 | 0.85 | False | 0 | nan | nan | nan | nan |
| diglycolamide | sys_a2a472b5124b09c6 | all | 0.97 | 0.85 | True | 1 | 21 | 263.1 | IN_DOMAIN_WITH_CAVEATS | COST_INCOMPLETE\|EQUILIBRIUM_ACID_ASSUMED_NOMINAL\|HIGH_LOADING\|LIGAND_LOSS_NOT_MEASURED\|OA_ASSUMED\|PHASE_BEHAVIOUR_UNKNOWN |
| diglycolamide | sys_a2a472b5124b09c6 | in_domain_only | 0.97 | 0.85 | True | 1 | 21 | 263.1 | IN_DOMAIN_WITH_CAVEATS | COST_INCOMPLETE\|EQUILIBRIUM_ACID_ASSUMED_NOMINAL\|HIGH_LOADING\|LIGAND_LOSS_NOT_MEASURED\|OA_ASSUMED\|PHASE_BEHAVIOUR_UNKNOWN |
| diglycolamide | sys_f58e3a596f8d50f9 | all | 0.97 | 0.85 | True | 1 | 26 | 309.3 | OUT_OF_DOMAIN | COST_INCOMPLETE\|EQUILIBRIUM_ACID_ASSUMED_NOMINAL\|HIGH_LOADING\|LIGAND_LOSS_NOT_MEASURED\|OA_ASSUMED\|OOD_HULL\|OOD_LIGAND\|OOD_METAL\|PHASE_BEHAVIOUR_UNKNOWN |
| diglycolamide | sys_f58e3a596f8d50f9 | in_domain_only | 0.97 | 0.85 | False | 0 | nan | nan | nan | nan |


#### Family n_donor

| system_id | name | status | params_source | status_of_parameters | n_converged | n_failed | n_on_spec_loosest | n_front_in_domain | reason |
|---|---|---|---|---|---|---|---|---|---|
| sys_fbff75db47e9b852 | C5BTBP in aliphatic hydrocarbon, nitrate medium, no additive | parameterised | nearest | nearest (origin corpus) | 300 | 0 | 0 | 0 |  |


Top three on-spec regimes per system (by consumption index):

(none on spec)

Pareto knee of the consistency cell:

| family | system_id | subset | purity_min | recovery_min | reachable | n_feasible | n_stages_total | consumption_index | regime_status | flags |
|---|---|---|---|---|---|---|---|---|---|---|
| n_donor | sys_fbff75db47e9b852 | all | 0.97 | 0.85 | False | 0 | nan | nan | nan | nan |
| n_donor | sys_fbff75db47e9b852 | in_domain_only | 0.97 | 0.85 | False | 0 | nan | nan | nan | nan |


#### Family other

| system_id | name | status | params_source | status_of_parameters | n_converged | n_failed | n_on_spec_loosest | n_front_in_domain | reason |
|---|---|---|---|---|---|---|---|---|---|
| sys_50088af14793f41f | DOODA (C12) in aliphatic hydrocarbon, nitrate medium, no additive | parameterised | nearest | nearest (origin corpus) | 300 | 0 | 0 | 0 |  |
| sys_a6ca4cd427ac7fcd | DMDO-HPyranDGA in aliphatic hydrocarbon, nitrate medium, no additive | parameterised | nearest | nearest (origin corpus) | 300 | 0 | 0 | 0 |  |
| sys_d93f24ac2fbe9b76 | 2-N-6-N-dimethyl-2-N-6-N-diphenylpyridine-2-6-dicarboxamide in chlorinated, nitrate medium, no additive | parameterised | nearest | nearest (origin corpus) | 300 | 0 | 1 | 0 |  |
| sys_edd93a49877e0b03 | DOODA (C8) in aliphatic hydrocarbon, nitrate medium, no additive | parameterised | nearest | nearest (origin corpus) | 300 | 0 | 0 | 0 |  |


Top three on-spec regimes per system (by consumption index):

| family | front | regime_status | n_ext | n_scr | n_str | oa_ext | s_over_a | w_over_a | scrub_acid_M | strip_acid_M | ligand_total_M | saponification_degree | purity_mol | recovery_from_feed | consumption_index | n_stages_total | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| other | 1 | OUT_OF_DOMAIN | 11 | 12 | 4 | 4.689 | 4.112 | 4.14 | 2.701 | 5.996 | 0.7357 | 0.1006 | 0.9807 | 0.823 | 980.2 | 27 | COST_INCOMPLETE\|EQUILIBRIUM_ACID_ASSUMED_NOMINAL\|LIGAND_LOSS_NOT_MEASURED\|OA_ASSUMED\|OOD_ACID\|OOD_HULL\|OOD_LIGAND\|OOD_LOADING\|OOD_METAL\|PHASE_BEHAVIOUR_UNKNOWN\|SAPONIFICATION_RANGE_UNKNOWN |


Pareto knee of the consistency cell:

| family | system_id | subset | purity_min | recovery_min | reachable | n_feasible | n_stages_total | consumption_index | regime_status | flags |
|---|---|---|---|---|---|---|---|---|---|---|
| other | sys_50088af14793f41f | all | 0.97 | 0.85 | False | 0 | nan | nan | nan | nan |
| other | sys_50088af14793f41f | in_domain_only | 0.97 | 0.85 | False | 0 | nan | nan | nan | nan |
| other | sys_a6ca4cd427ac7fcd | all | 0.97 | 0.85 | False | 0 | nan | nan | nan | nan |
| other | sys_a6ca4cd427ac7fcd | in_domain_only | 0.97 | 0.85 | False | 0 | nan | nan | nan | nan |
| other | sys_d93f24ac2fbe9b76 | all | 0.97 | 0.85 | False | 0 | nan | nan | nan | nan |
| other | sys_d93f24ac2fbe9b76 | in_domain_only | 0.97 | 0.85 | False | 0 | nan | nan | nan | nan |
| other | sys_edd93a49877e0b03 | all | 0.97 | 0.85 | False | 0 | nan | nan | nan | nan |
| other | sys_edd93a49877e0b03 | in_domain_only | 0.97 | 0.85 | False | 0 | nan | nan | nan | nan |


### DGA + aqueous hold-back ligand exploration (DESIGN.md section 13.5) -- sensitivity study, not a recommendation

*regime: exploratory (DESIGN 13.5); D source nearest (decision m1_adopted = False); complexant betas ASSUMED_PLACEHOLDER sweep (U3); feed_status assumed*

Reference regime: the TODGA-only candidate of largest purity x recovery in the 50-row LHS (n_ext 5, n_scr 15, n_str 12, O/A 3.39, purity 0.881, recovery 0.570, flags COST_INCOMPLETE|EQUILIBRIUM_ACID_ASSUMED_NOMINAL|HIGH_LOADING|LIGAND_LOSS_NOT_MEASURED|OA_ASSUMED|THIRD_PHASE_RISK).

The complexant enters the feed (`feed`) or the scrub liquor only (`scrub_only`) at 0.05 M total; `delta_log_beta = log beta_Pr - log beta_Nd` is the beta contrast the hold-back ligand would need; the consumption uses the assumed regeneration fraction 0.9 (range (0.5, 1.0)). Until a sourced beta set exists (open item U3) no number here is a recommendation.

| variant | log_beta_Nd | delta_log_beta | log_k_h | complexant_M | status | purity_mol | recovery_from_feed | recovery_total | enrichment_factor_Pr | complexant_mol_per_kg_oxide | consumption_index | regime_status | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none (TODGA only) | nan | nan | nan | 0 | converged_newton | 0.8806 | 0.57 | 0.57 | 2.459 | 0 | 1099 | INADMISSIBLE | COST_INCOMPLETE\|EQUILIBRIUM_ACID_ASSUMED_NOMINAL\|HIGH_LOADING\|LIGAND_LOSS_NOT_MEASURED\|OA_ASSUMED\|THIRD_PHASE_RISK |
| feed | 1 | 0 | 2 | 0.05 | converged_newton | 0.8806 | 0.57 | 0.57 | 2.459 | 0.6951 | 1099 | INADMISSIBLE | COST_INCOMPLETE\|EQUILIBRIUM_ACID_ASSUMED_NOMINAL\|HIGH_LOADING\|LIGAND_LOSS_NOT_MEASURED\|OA_ASSUMED\|OOD_COMPLEXANT\|THIRD_PHASE_RISK |
| feed | 1 | 0.5 | 2 | 0.05 | converged_newton | 0.8807 | 0.5701 | 0.5701 | 2.461 | 0.6951 | 1099 | INADMISSIBLE | COST_INCOMPLETE\|EQUILIBRIUM_ACID_ASSUMED_NOMINAL\|HIGH_LOADING\|LIGAND_LOSS_NOT_MEASURED\|OA_ASSUMED\|OOD_COMPLEXANT\|THIRD_PHASE_RISK |
| feed | 1 | 1 | 2 | 0.05 | converged_newton | 0.881 | 0.5703 | 0.5703 | 2.467 | 0.6948 | 1099 | INADMISSIBLE | COST_INCOMPLETE\|EQUILIBRIUM_ACID_ASSUMED_NOMINAL\|HIGH_LOADING\|LIGAND_LOSS_NOT_MEASURED\|OA_ASSUMED\|OOD_COMPLEXANT\|THIRD_PHASE_RISK |
| feed | 1 | 1.5 | 2 | 0.05 | converged_newton | 0.8818 | 0.5709 | 0.5709 | 2.487 | 0.6941 | 1098 | INADMISSIBLE | COST_INCOMPLETE\|EQUILIBRIUM_ACID_ASSUMED_NOMINAL\|HIGH_LOADING\|LIGAND_LOSS_NOT_MEASURED\|OA_ASSUMED\|OOD_COMPLEXANT\|THIRD_PHASE_RISK |
| feed | 2.5 | 0 | 2 | 0.05 | converged_newton | 0.8805 | 0.57 | 0.57 | 2.457 | 0.6952 | 1099 | INADMISSIBLE | COST_INCOMPLETE\|EQUILIBRIUM_ACID_ASSUMED_NOMINAL\|HIGH_LOADING\|LIGAND_LOSS_NOT_MEASURED\|OA_ASSUMED\|OOD_COMPLEXANT\|THIRD_PHASE_RISK |
| feed | 2.5 | 0.5 | 2 | 0.05 | converged_newton | 0.8831 | 0.5719 | 0.5719 | 2.519 | 0.6929 | 1096 | INADMISSIBLE | COST_INCOMPLETE\|EQUILIBRIUM_ACID_ASSUMED_NOMINAL\|HIGH_LOADING\|LIGAND_LOSS_NOT_MEASURED\|OA_ASSUMED\|OOD_COMPLEXANT\|THIRD_PHASE_RISK |
| feed | 2.5 | 1 | 2 | 0.05 | converged_newton | 0.8903 | 0.577 | 0.577 | 2.704 | 0.6867 | 1086 | INADMISSIBLE | COST_INCOMPLETE\|EQUILIBRIUM_ACID_ASSUMED_NOMINAL\|HIGH_LOADING\|LIGAND_LOSS_NOT_MEASURED\|OA_ASSUMED\|OOD_COMPLEXANT\|THIRD_PHASE_RISK |
| feed | 2.5 | 1.5 | 2 | 0.05 | converged_newton | 0.9062 | 0.5887 | 0.5887 | 3.222 | 0.6731 | 1064 | INADMISSIBLE | COST_INCOMPLETE\|EQUILIBRIUM_ACID_ASSUMED_NOMINAL\|HIGH_LOADING\|LIGAND_LOSS_NOT_MEASURED\|OA_ASSUMED\|OOD_COMPLEXANT\|THIRD_PHASE_RISK |
| feed | 4 | 0 | 2 | 0.05 | converged_newton | 0.8791 | 0.5689 | 0.5689 | 2.425 | 0.6965 | 1101 | INADMISSIBLE | COST_INCOMPLETE\|EQUILIBRIUM_ACID_ASSUMED_NOMINAL\|HIGH_LOADING\|LIGAND_LOSS_NOT_MEASURED\|OA_ASSUMED\|OOD_COMPLEXANT\|THIRD_PHASE_RISK |
| feed | 4 | 0.5 | 2 | 0.05 | converged_newton | 0.9095 | 0.5911 | 0.5911 | 3.351 | 0.6704 | 1060 | INADMISSIBLE | COST_INCOMPLETE\|EQUILIBRIUM_ACID_ASSUMED_NOMINAL\|HIGH_LOADING\|LIGAND_LOSS_NOT_MEASURED\|OA_ASSUMED\|OOD_COMPLEXANT\|THIRD_PHASE_RISK |
| feed | 4 | 1 | 2 | 0.05 | converged_newton | 0.9451 | 0.6175 | 0.6175 | 5.741 | 0.6418 | 1015 | INADMISSIBLE | COST_INCOMPLETE\|EQUILIBRIUM_ACID_ASSUMED_NOMINAL\|HIGH_LOADING\|LIGAND_LOSS_NOT_MEASURED\|OA_ASSUMED\|OOD_COMPLEXANT\|THIRD_PHASE_RISK |
| feed | 4 | 1.5 | 2 | 0.05 | converged_newton | 0.9736 | 0.639 | 0.639 | 12.28 | 0.6202 | 980.6 | INADMISSIBLE | COST_INCOMPLETE\|EQUILIBRIUM_ACID_ASSUMED_NOMINAL\|HIGH_LOADING\|LIGAND_LOSS_NOT_MEASURED\|OA_ASSUMED\|OOD_COMPLEXANT\|THIRD_PHASE_RISK |
| scrub_only | 1 | 0 | 2 | 0.05 | converged_newton | 0.8806 | 0.57 | 0.57 | 2.459 | 1.763 | 1100 | INADMISSIBLE | COST_INCOMPLETE\|EQUILIBRIUM_ACID_ASSUMED_NOMINAL\|HIGH_LOADING\|LIGAND_LOSS_NOT_MEASURED\|OA_ASSUMED\|OOD_COMPLEXANT\|THIRD_PHASE_RISK |
| scrub_only | 1 | 0.5 | 2 | 0.05 | converged_newton | 0.881 | 0.5702 | 0.5702 | 2.468 | 1.762 | 1100 | INADMISSIBLE | COST_INCOMPLETE\|EQUILIBRIUM_ACID_ASSUMED_NOMINAL\|HIGH_LOADING\|LIGAND_LOSS_NOT_MEASURED\|OA_ASSUMED\|OOD_COMPLEXANT\|THIRD_PHASE_RISK |
| scrub_only | 1 | 1 | 2 | 0.05 | converged_newton | 0.8822 | 0.5708 | 0.5708 | 2.496 | 1.76 | 1099 | INADMISSIBLE | COST_INCOMPLETE\|EQUILIBRIUM_ACID_ASSUMED_NOMINAL\|HIGH_LOADING\|LIGAND_LOSS_NOT_MEASURED\|OA_ASSUMED\|OOD_COMPLEXANT\|THIRD_PHASE_RISK |
| scrub_only | 1 | 1.5 | 2 | 0.05 | converged_newton | 0.8857 | 0.5727 | 0.5727 | 2.584 | 1.754 | 1095 | INADMISSIBLE | COST_INCOMPLETE\|EQUILIBRIUM_ACID_ASSUMED_NOMINAL\|HIGH_LOADING\|LIGAND_LOSS_NOT_MEASURED\|OA_ASSUMED\|OOD_COMPLEXANT\|THIRD_PHASE_RISK |
| scrub_only | 2.5 | 0 | 2 | 0.05 | converged_newton | 0.8815 | 0.5678 | 0.5678 | 2.48 | 1.769 | 1105 | INADMISSIBLE | COST_INCOMPLETE\|EQUILIBRIUM_ACID_ASSUMED_NOMINAL\|HIGH_LOADING\|LIGAND_LOSS_NOT_MEASURED\|OA_ASSUMED\|OOD_COMPLEXANT\|THIRD_PHASE_RISK |
| scrub_only | 2.5 | 0.5 | 2 | 0.05 | converged_newton | 0.892 | 0.5734 | 0.5734 | 2.753 | 1.752 | 1094 | INADMISSIBLE | COST_INCOMPLETE\|EQUILIBRIUM_ACID_ASSUMED_NOMINAL\|HIGH_LOADING\|LIGAND_LOSS_NOT_MEASURED\|OA_ASSUMED\|OOD_COMPLEXANT\|THIRD_PHASE_RISK |
| scrub_only | 2.5 | 1 | 2 | 0.05 | converged_newton | 0.9183 | 0.5876 | 0.5876 | 3.747 | 1.71 | 1067 | INADMISSIBLE | COST_INCOMPLETE\|EQUILIBRIUM_ACID_ASSUMED_NOMINAL\|HIGH_LOADING\|LIGAND_LOSS_NOT_MEASURED\|OA_ASSUMED\|OOD_COMPLEXANT\|THIRD_PHASE_RISK |
| scrub_only | 2.5 | 1.5 | 2 | 0.05 | converged_newton | 0.9652 | 0.615 | 0.615 | 9.238 | 1.634 | 1020 | INADMISSIBLE | COST_INCOMPLETE\|EQUILIBRIUM_ACID_ASSUMED_NOMINAL\|HIGH_LOADING\|LIGAND_LOSS_NOT_MEASURED\|OA_ASSUMED\|OOD_COMPLEXANT\|THIRD_PHASE_RISK |
| scrub_only | 4 | 0 | 2 | 0.05 | converged_newton | 0.899 | 0.5261 | 0.5261 | 2.965 | 1.91 | 1192 | INADMISSIBLE | COST_INCOMPLETE\|EQUILIBRIUM_ACID_ASSUMED_NOMINAL\|HIGH_LOADING\|LIGAND_LOSS_NOT_MEASURED\|OA_ASSUMED\|OOD_COMPLEXANT\|THIRD_PHASE_RISK |
| scrub_only | 4 | 0.5 | 2 | 0.05 | converged_newton | 0.9783 | 0.5707 | 0.5707 | 15 | 1.76 | 1099 | INADMISSIBLE | COST_INCOMPLETE\|EQUILIBRIUM_ACID_ASSUMED_NOMINAL\|HIGH_LOADING\|LIGAND_LOSS_NOT_MEASURED\|OA_ASSUMED\|OOD_COMPLEXANT\|THIRD_PHASE_RISK |
| scrub_only | 4 | 1 | 2 | 0.05 | converged_newton | 1 | 0.5988 | 0.5988 | 1.395e+07 | 1.678 | 1047 | INADMISSIBLE | COST_INCOMPLETE\|EQUILIBRIUM_ACID_ASSUMED_NOMINAL\|HIGH_LOADING\|LIGAND_LOSS_NOT_MEASURED\|OA_ASSUMED\|OOD_COMPLEXANT\|THIRD_PHASE_RISK |
| scrub_only | 4 | 1.5 | 2 | 0.05 | converged_newton | 1 | 0.5992 | 0.5992 | 5.631e+16 | 1.677 | 1047 | INADMISSIBLE | COST_INCOMPLETE\|EQUILIBRIUM_ACID_ASSUMED_NOMINAL\|HIGH_LOADING\|LIGAND_LOSS_NOT_MEASURED\|OA_ASSUMED\|OOD_COMPLEXANT\|THIRD_PHASE_RISK |


## 9. Consumption and cost

Consumption per kg of target oxide (acid, base, water, extractant make-up, complexant) is the primary economic number; the cost proxy uses the placeholder price table below (open item U5) and is NaN with `COST_INCOMPLETE` wherever a price or a consumption is missing.

*regime: config/prices.json; every value assumed with a range; no source*

| item | value | unit | currency | status | assumed_label | range | note |
|---|---|---|---|---|---|---|---|
| acid | 0.03 | USD/mol | USD | assumed | ASSUMED_PLACEHOLDER | [0.01, 0.1] | placeholder order of magnitude for technical HCl / HNO3 per mol H+; no source; replace under U5 |
| base | 0.02 | USD/mol | USD | assumed | ASSUMED_PLACEHOLDER | [0.01, 0.06] | placeholder order of magnitude for NaOH per mol (saponification of the acidic extractant); no source; replace under U5 |
| salting_anion | 0.05 | USD/mol | USD | assumed | ASSUMED_PLACEHOLDER | [0.01, 0.2] | placeholder for an added nitrate / chloride salt per mol of anion; no source; replace under U5 |
| complexant | 5 | USD/mol | USD | assumed | ASSUMED_PLACEHOLDER | [0.5, 50.0] | placeholder for a hydrophilic hold-back complexant per mol consumed (1 - regeneration fraction); wide range because the complexant is not yet chosen (open item U3); replace under U5 |
| extractant_makeup | 10 | USD/mol | USD | assumed | ASSUMED_PLACEHOLDER | [2.0, 50.0] | placeholder per mol of extractant (acidic organophosphorus or diglycolamide class) lost to bleed and entrainment; no source; replace under U5 |
| diluent | 1 | USD/L | USD | assumed | ASSUMED_PLACEHOLDER | [0.5, 3.0] | placeholder per litre of aliphatic diluent leaving with the organic bleed; no source; replace under U5 |
| water | 0.002 | USD/L | USD | assumed | ASSUMED_PLACEHOLDER | [0.0005, 0.01] | placeholder per litre of process water (scrub + strip + dilution); no source; replace under U5 |


## 10. Limits, nulls, defects

Model idealisations carried by every result (`SystemModel.assumptions`): concentrations for activities; complete dimerisation of acidic organophosphorus extractants; fixed stoichiometries; no water co-extraction; no mixed organic complexes; no ligand aggregation; constant phase volumes; temperature enters only through the parameter set's temperature band.

Flag prevalence over the reachable spec cells of the case (draw x cell rows): .

Assumptions on corpus rows: `cond__metal_concentration_mM` is the initial aqueous concentration, O/A = 1 (`OA_ASSUMED`), nominal acid = equilibrium acidity (`EQUILIBRIUM_ACID_ASSUMED_NOMINAL`), nitrate = nominal HNO3 (DATA_AUDIT.md section 8). HNO3 uptake by diglycolamides is unmodelled (`ACID_UPTAKE_UNMODELLED` above 1 M); the only sourced third-phase limit is the TODGA LOC of 0.008 M Nd (U6); every other phase field is null (`PHASE_BEHAVIOUR_UNKNOWN` above 0.3 loading).

**Defects found in the integration pass** (`addenda/INTEGRATION.md`):

1. *Fixed.* The deployed 1-NN D source applied the ideal depletion term to records that had themselves been measured under load -- 28.7 % of corpus records with a metal concentration sit above loading fraction 0.1 (TBDGA 81 %, median 0.30). Each record is now lifted to its own tracer limit first (`NearestConditionD(tracer_correction=True)`, the law R2 validated). R1 and R2 are unaffected (neither applies the term that way); `results/recipes/` was re-run, and 276 of 298 candidates of the TODGA system changed (median purity change 0.088) although the Pareto leaders barely moved.
2. *Reported, not fixed.* Even corrected, the 1-NN is not smooth along the acid axis (D(Nd) at 0.1 M TODGA: 1.89, 37.9, 7.52, 1.13 at 0.5, 1, 3, 5 M HNO3): it reproduces genuine between-publication scatter, because a nearest-neighbour cannot average two publications that disagree. An optimiser searching over acidity can land on a single flattering record; the OOD flags are the only guard, and this is a property of the D source the R1 null selected.
3. *Closed.* The hypothesis that M1 helps where its exponent is reliable is **not** supported (Spearman -0.117, p = 0.69; `results/eval/RELIABILITY_PROBE.md`), so a reliability gate is not a route around the null.

What a single laboratory measurement would change most (the factor whose sweep moves purity and recovery most at the reference regime; D-optimal choice in the sense of gen15 section 7 -- measure the parameter with the largest response first):

| system_id | factor | purity_range | recovery_range |
|---|---|---|---|
| sys_29976921e156a0a0 | feed_ratio_nd_pr | 0.1836 | 2.22e-16 |
| sys_29976921e156a0a0 | ligand_total_M | 0.1836 | 1 |
| sys_29976921e156a0a0 | saponification_degree | 0.1504 | 0.932 |
| sys_29976921e156a0a0 | feed_total_metal_M | 0.07025 | 2.22e-16 |
| sys_29976921e156a0a0 | log_k_Nd (2-log window clipped to range) | 0 | 5.551e-16 |
| sys_29976921e156a0a0 | oa_ext (x reference) | 0 | 2.22e-16 |
| sys_29976921e156a0a0 | sf_nd_pr | 0 | 5.551e-16 |
| sys_f02db527a94a5e86 | feed_ratio_nd_pr | 0.1495 | 2.22e-16 |
| sys_f02db527a94a5e86 | ligand_total_M | 0.1495 | 1 |
| sys_f02db527a94a5e86 | saponification_degree | 0.1495 | 1 |
| sys_f02db527a94a5e86 | feed_total_metal_M | 0.1271 | 4.441e-16 |
| sys_f02db527a94a5e86 | log_k_Nd (2-log window clipped to range) | 1.11e-16 | 7.772e-16 |
| sys_f02db527a94a5e86 | sf_nd_pr | 1.11e-16 | 5.551e-16 |
| sys_f02db527a94a5e86 | oa_ext (x reference) | 0 | 2.22e-16 |


## 11. gen15 pre-screen usage and its limits

The gen15 direction model (`deploy_g15.joblib`, gitignored) is used only by `scripts/g18_screen.py` as a pre-screen prior on the sign of log D(A) - log D(B); validator V6 refuses any D sourced from it, and no cascade uses it.

Over the 71 candidates of `cases/screen_candidates.txt` (every corpus extractant with both Pr and Nd measured): 42 called Nd-selective, 29 Pr-selective. The predicted magnitude takes only **2 distinct values** (0.0334, 0.1391 log units), so the prior is a **sign call with an essentially constant magnitude** -- the gen16 conclusion, reproduced here. Per-molecule rows: `results/screen/priors.csv`.

## 12. Reproduction

From the repository root with `.venv/Scripts/python.exe`, seed 18, one process:

```
generations/gen18_process/scripts/g18_build_db.py
generations/gen18_process/scripts/g18_audit.py
generations/gen18_process/scripts/g18_seal_prereg.py
generations/gen18_process/scripts/g18_fit_dmodels.py --all
generations/gen18_process/scripts/g18_eval_dmodels.py
generations/gen18_process/scripts/g18_loading_check.py
generations/gen18_process/scripts/g18_bench.py
generations/gen18_process/scripts/g18_case_prnd.py --feed cases/prnd_feed.json --spec cases/prnd_spec.json --allow-placeholders --todga-feed cases/todga_prnd_feed.json
generations/gen18_process/scripts/g18_recipes.py --feed cases/todga_prnd_feed.json --spec cases/prnd_spec.json
generations/gen18_process/scripts/g18_screen.py --smiles-file <txt> --pair Nd Pr
generations/gen18_process/scripts/g18_report.py
```

Pre-registration seal: sealed: 62f5f2f26604da0ed091919342000b87eea6a796dbb2dd779a757ecc60565668 (`prereg_sha256 = 62f5f2f26604da0ed091919342000b87eea6a796dbb2dd779a757ecc60565668`)

Manifests (input / output SHA-256 in each file):

| manifest | git_head | seed | n_outputs |
|---|---|---|---|
| results/audit/manifest.json | cdfcebe8c0a002ffdaf10af76c3e2e44eb124c6f | 18 | 296 |
| results/bench/manifest.json | cdfcebe8c0a002ffdaf10af76c3e2e44eb124c6f | 18 | 1 |
| results/case_prnd/manifest.json | cdfcebe8c0a002ffdaf10af76c3e2e44eb124c6f | 18 | 150 |
| results/case_prnd/targeted/manifest.json | cdfcebe8c0a002ffdaf10af76c3e2e44eb124c6f | 18 | 2 |
| results/dmodels/manifest.json | cdfcebe8c0a002ffdaf10af76c3e2e44eb124c6f | 18 | 36 |
| results/eval/manifest.json | cdfcebe8c0a002ffdaf10af76c3e2e44eb124c6f | 18 | 7 |
| results/loading/manifest.json | cdfcebe8c0a002ffdaf10af76c3e2e44eb124c6f | 18 | 8 |
| results/recipes/todga_prnd_feed/manifest.json | cdfcebe8c0a002ffdaf10af76c3e2e44eb124c6f | 18 | 14 |
| results/screen/manifest.json | cdfcebe8c0a002ffdaf10af76c3e2e44eb124c6f | 18 | 1 |


Comparison count: exploratory 8, secondary 4, primary 3 (total 15).
