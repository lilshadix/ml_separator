# L1 — the cycle-corrected xTB descriptor, Stage 1 (bookkeeping) and the Stage 2 hand-over

**DISCOVERY.**  Pre-registration `gen16_leads/PRE_REGISTRATION.md` §3, L1
(SHA-256 `d004c30388078ae97232af2537a1915360360a7cb9eda07ebe4076764c48f50e`).  Stage 1 ran **once**, on
the pre-specified four models and three sets.  The MAE arm (`G14+PHYSCYC`) is Phase 3 and was **not**
run, because Stage 1 is not positive.  No bench fold, seed, design or metric was touched by this
lead: Stage 1 is a description of the corpus, not a score.

---

## 0. Verdict

> **Registered rule.**  *"L1 is positive if, on `S8`, `SPECIES` gives |ρ(slope, a)| ≥ 0.40 with a
> LOCO-stable sign, |partial ρ | n_metals| ≥ 0.40, and the chemotype-blocked CI excludes zero.  It is
> closed if |ρ| < 0.25 on `S8` with a CI containing zero.  Between 0.25 and 0.40 it is 'not closed,
> not positive' and is said so."*

| quantity | value |
|---|---|
| ρ(SPECIES slope, `a`) on `S8` | **−0.084** (n = 62, p = 0.518) |
| partial ρ given `n_metals` | −0.081 |
| chemotype-blocked 95 % CI (2000 reps) | **[−0.297, +0.240]** — contains zero |
| LOCO over 40 chemotypes | [−0.129, +0.0005] — sign **not** stable |
| |ρ| < 0.25 **and** CI contains zero | both true |

### **L1 Stage 1 is CLOSED.**

The correlation gen15 measured at +0.644 is a composition artefact, and this is now the exact
bookkeeping that produces it, not an approximation to it.  Secondary registered row: `SPECIES_CONST`
on `S8` gives ρ = **+0.167** (n = 19, p = 0.495, LOCO [+0.073, +0.335], sign-stable; its bootstrap CI
is undefined — see §4) — also closed by the rule, and it is the honest upper end of what survives:
**|ρ| ≈ 0.17–0.20, exactly the "weak 2D column" level gen15 predicted, and far below the 2D benchmark
`frac_donor_pairs_within_3` at −0.53.**

Stage 2 is nevertheless delivered ready to run, because one thing it can settle that Stage 1 cannot —
see §6 and `L1_STAGE2_HANDOVER.md` §6.

---

## 1. Reproduction of gen15 (part A)

`gen15_curve/exp/phys3d/trap_check.py`'s construction was re-run through the **imported** frozen
helpers (`build_block.load_geometries`, `main_blocks`, `_two_way_residual`, `_slope_fit`,
`descriptor_stats._rho`, `_partial_rho`) and compared row by row against the stored
`trap_check.csv`.  Nothing under `gen15_curve/` was modified or re-executed.

**All 24 stored rows reproduce, max |Δρ| = 8.3 × 10⁻¹⁷, max |Δn| = 0, max |Δp| = 0.**
The local copy of `_two_way_residual` that also returns coefficients reproduces the frozen residual
to `0.0e+00` before it is used anywhere (`stage1_decision.json: two_way_copy_max_abs_diff`).

| subset | construction | n | ρ(slope, a) gen16 | ρ gen15 |
|---|---|---|---|---|
| all cohort extractants | naive | 81 | +0.466509 | +0.466509 |
| all cohort extractants | A (constant-composition block) | 79 | +0.104211 | +0.104211 |
| all cohort extractants | B (element counts as covariates) | 81 | +0.202823 | +0.202823 |
| complete 14-metal series | naive | 41 | **+0.599129** | +0.599129 |
| complete 14-metal series | A | 41 | +0.243554 | +0.243554 |
| complete 14-metal series | B | 41 | +0.423519 | +0.423519 |
| block ≥ 7 metals | naive | 54 | +0.394168 | +0.394168 |
| constant-composition series only | naive | 9 | −0.300000 | −0.300000 |

### The 39-versus-41 difference, resolved

`trap_check.py` defines "complete 14-metal series" by `n_series`, **the number of metals present in
the geometry table**.  The energy fit can only use metals that actually have a
`complex_total_energy_eV` — and 39 of the 1155 complexes have none (1116 non-null).  Two of the 41
geometrically complete series are missing an energy, so the *energy* series is complete for only
**39** extractants.  Restricting to those 39 reproduces gen15 §3 exactly, on all four of its numbers:

| gen15 §3 states | gen16 obtains on the 39 energy-complete series |
|---|---|
| Spearman **+0.644** with the amplitude | **+0.643927** |
| **−0.636** for its curvature term | **−0.636364** (quadratic coefficient vs `a`) |
| leave-one-chemotype-out range **[+0.598, +0.745]** | **[+0.598465, +0.745020]** |
| partial Spearman **+0.645** given `n_metals` | **+0.645330** |

So `trap_check.py`'s +0.599 (n = 41) and the report's +0.644 (n = 39) are the same construction on
two definitions of "complete": geometries present versus energies present.  Both are in
`stage1_reproduction.csv` (`role = reproduction` / `role = 39_vs_41_diagnosis`).  **Neither is
wrong; the pair is a good illustration of the guardrail gen15 itself wrote — a correlation on 39 of
82 extractants is not the same claim as one on all of them.**

gen15 §3's *other* two table rows — construction A at "70 extractants, +0.195" and construction B at
"71 extractants, +0.216" — do **not** correspond to any construction in the frozen `phys3d` scripts.
The reproducible values are `trap_check.csv`'s 79 / +0.104 and 81 / +0.203, and
`descriptor_stats.csv` independently gives 79 and 81 non-null for `dE_dr_A` / `dE_dr_B`.  Three
energy-completeness restrictions were tried and none lands on 70 / 71 (nearest: "every metal of the
chosen series has an energy" → 76 / +0.088 and 77 / +0.212; all five subsets are in
`stage1_reproduction.csv`, `role = 39_vs_41_diagnosis`).  This changes no conclusion — **every
composition-controlled construction lands at |ρ| ≤ 0.22** — but it means the two controlled rows of
gen15 §3's table should be quoted from `trap_check.csv`, not from the report.

A second, previously unremarked property of the naive construction: its slope and its curvature are
**not two signals**.  Across the 39 energy-complete series, Spearman(slope, quadratic) = **−0.981**
(Pearson −0.991); at the chemotype-mean level the ranks mirror exactly, so the permutation table
reports the same |ρ| for both.  gen15's "+0.644 for the slope *and* −0.636 for the curvature" is one
step function read twice.  Under `SPECIES` that coupling falls to −0.335.

---

## 2. What the bookkeeping actually is (part B, and the load-bearing finding)

Before any model was fitted, the composition of all 1155 accepted complexes was decomposed against
the RDKit formula of `canonical_smiles` (`stage1_bookkeeping.csv`):

* **1155 / 1155 complexes decompose exactly** into one metal + `n_ligs` × the neutral ligand as
  written + `n_NO3` nitrates + `n_H2O` waters.  No residue, no missing hydrogen; all 177 ligands have
  RDKit formal charge 0, so none is deprotonated in the complex.
* **`n_fill` counts fill *donor sites*, not molecules.**  `n_fill = 2·n_NO3 + n_H2O` holds for every
  complex: nitrate is bidentate, water monodentate.  167 nitrate complexes have an odd `n_fill`,
  which means one water alongside the nitrates.  Reading `n_fill` as a molecule count is wrong by a
  factor of two for nitrate and mis-assigns those 167 waters.
* **Charge convention: metal 3+, ligands neutral, nitrate −1, water 0, complex = 3 − n_NO3.**  The
  per-atom `initial_charges` column sums to exactly `3 − n_NO3` on **1145 of 1145** files that carry
  it (901 at +3, 181 at +2, 63 at +1), and the post-relaxation Mulliken `charge` column sums to the
  same value within 0.02 e on **1116 of 1116** files that carry one.
* `initial_magmoms` is zero on 1134 of the 1145 and carries the *formal f-electron count* on the
  metal in eleven (Eu 6.0, Yb 1.0).  It is metadata GFN2 cannot use — the lanthanides are
  parameterised with f-in-core, so the valence shell is closed — and it is set inconsistently (other
  Eu and Yb complexes carry 0.0).  `--uhf 0` everywhere is therefore correct for the references, and
  the eleven files are listed in `stage1_charge_audit.csv` rather than passed over.
* No solvation key exists anywhere in the dataset; the energies are gas-phase GFN2 totals.

The registered `SPECIES` model therefore enters each fill species **with its actual molecule count**
(`n_NO3·γ_nitrate + n_H2O·γ_water`), which is the pre-registration's own wording ("one coefficient per
fill species", brief §5: "entering with its actual count").  Because this correction was fixed by an
audit that ran **before any correlation was computed**, and because the literal alternative reading is
not obviously wrong, the literal `n_fill`-column variant was fitted too and is reported beside it as
an exploratory model, `SPECIES_NFILLCOL`.  Both are in every table; the verdict is the same under
either (`SPECIES_NFILLCOL` on `S8` gives ρ = +0.374, which fails the ≥ 0.40 bar and is exploratory
regardless).

### The fitted γ are the free-species energies, which is the model working

| coefficient | fitted (eV/molecule) | in Hartree | corpus yardstick † |
|---|---|---|---|
| `γ_nitrate` | **−431.97 ± 1.39** | −15.875 | −414.2 |
| `γ_water` | **−138.88 ± 0.31** | −5.104 | −139.2 |

† a global per-element regression of `complex_total_energy_eV` on element counts over the same 1116
rows, evaluated at NO₃ and H₂O.  `γ_water` lands within **0.34 eV** of that yardstick and within
about 0.9 eV of the standard GFN2 free-water total energy (≈ −5.07 Hartree); the residual is the mean
Ln–OH₂ binding energy, which is exactly what the coefficient should absorb.  `γ_nitrate` is 18 eV
below the additive yardstick, as it must be for a charged, strongly bound bidentate anion.  **The
fitted composition coefficients are physically sensible free-species energies, so the `SPECIES` model
is doing the cycle's arithmetic, not fitting noise.**

---

## 3. Identifiability of `δ_ligand`: what Stage 1 can and cannot separate

`SPECIES` adds `n_ligs × δ_ligand` for every series in which `n_ligs` varies — **56 series**, covering
**43 of the 62 `S8` extractants** (the remaining 19 are exactly the `S8` members of
`SPECIES_CONST`).  Within a series, `n_ligs` steps down as the cation contracts, so it is a step
function of the radius, and the free `δ` competes directly with the radius trend.  Quantified over
those 43 series (`stage1_identifiability.csv`):

| quantity | min | median | max |
|---|---|---|---|
| corr(`n_ligs`, r) within series | −0.368 | **+0.852** | +0.868 |
| attenuation 1 − ρ² of a linear radius trend | 0.246 | **0.273** | 0.998 |
| VIF of `n_ligs` against r | 1.00 | 3.66 | 4.06 |
| R² of `n_ligs` on [r, r²] | 0.091 | 0.769 | 0.835 |
| condition number of the standardised [r, r², `n_ligs`] block | 1.67 | 4.28 | 5.07 |
| VIF of `δ_series` inside the augmented `SPECIES` design | 1.60 | **6.34** | 11.26 |

**Stated plainly: for 43 of the 62 `S8` extractants, Stage 1 cannot separate the ligand-count term
from the radius trend.**  The typical series has `n_ligs` correlated +0.85 with the radius, so a free
`δ` absorbs about 73 % of any linear trend before the slope is read.  It is not a collinearity that a
larger sample fixes — it is the chemistry (the cavity sheds a ligand as the cation contracts), and the
only way out is to *fix* the ligand term at its true value instead of fitting it, which is precisely
what the Stage 2 cycle does (§6).

Two things bound how much of the null this can explain.  (i) Attenuation shrinks a coefficient; it
does not destroy a **rank** correlation, and the registered statistic is Spearman.  (ii) The
attenuation-free control is registered and was run: `SPECIES_CONST`, on the 19 `S8` series where
`n_ligs` never varies and no `δ` exists at all, gives **+0.167**, not +0.6.  The signal does not
reappear once the collinear term is removed; it only stops being negative.

---

## 4. Registered statistics (part D)

Extractant-level Spearman, unit = extractant, chemotype-blocked bootstrap 2000 replicates
(`robust_stats.py` construction, seed 20260909), LOCO over the 40 chemotypes present, partial
Spearman controlling the number of metals the *experiment* measured.  Sets: `S8` = cohort extractant
with a well-determined curve (the 82 of `extractant_targets.parquet`) and ≥ 8 **computed** metals;
`S14` = 14 computed metals; `S3` = ≥ 3.  Of the 82 cohort extractants, 81 have ≥ 3 computed metals,
62 have ≥ 8 and 39 have all 14.

### Primary target: the signed amplitude `a`

| model | set | n | ρ | p | LOCO min | LOCO max | sign stable | ρ \| n_metals | 95 % CI | excl. 0 |
|---|---|---|---|---|---|---|---|---|---|---|
| NAIVE | S8 | 62 | +0.3920 | 0.0016 | +0.219 | +0.505 | yes | +0.385 | [−0.151, +0.689] | no |
| NAIVE | S14 | 39 | **+0.6439** | <1e-4 | +0.598 | +0.745 | yes | +0.645 | [+0.277, +0.825] | yes |
| NAIVE | S3 | 81 | +0.4665 | <1e-4 | +0.304 | +0.559 | yes | +0.445 | [+0.019, +0.674] | yes |
| ELEM | S8 | 62 | +0.3134 | 0.0131 | +0.031 | +0.376 | yes | +0.299 | [−0.270, +0.596] | no |
| ELEM | S14 | 39 | +0.3964 | 0.0125 | −0.079 | +0.495 | **no** | +0.396 | [−0.398, +0.688] | no |
| ELEM | S3 | 81 | +0.2028 | 0.0694 | +0.080 | +0.268 | yes | +0.181 | [−0.162, +0.437] | no |
| **SPECIES** | **S8** | **62** | **−0.0837** | 0.518 | −0.129 | +0.001 | **no** | **−0.081** | **[−0.297, +0.240]** | **no** |
| SPECIES | S14 | 39 | −0.0482 | 0.771 | −0.122 | +0.033 | no | −0.046 | [−0.356, +0.345] | no |
| SPECIES | S3 | 81 | −0.0417 | 0.712 | −0.074 | +0.045 | no | −0.038 | [−0.223, +0.246] | no |
| **SPECIES_CONST** | **S8** | **19** | **+0.1667** | 0.495 | +0.073 | +0.335 | yes | +0.156 | undefined ‡ | no |
| SPECIES_CONST | S14 | 11 | +0.1545 | 0.650 | +0.006 | +0.405 | yes | +0.162 | undefined ‡ | no |
| SPECIES_CONST | S3 | 33 | +0.1985 | 0.268 | +0.133 | +0.256 | yes | +0.207 | [−0.089, +0.496] | no |
| *SPECIES_NFILLCOL* (exploratory) | S8 | 62 | +0.3738 | 0.0028 | +0.316 | +0.459 | yes | +0.370 | [+0.008, +0.691] | yes |
| *SPECIES_NFILLCOL* (exploratory) | S14 | 39 | +0.2642 | 0.104 | +0.138 | +0.367 | yes | +0.271 | [−0.160, +0.608] | no |
| *SPECIES_NFILLCOL* (exploratory) | S3 | 81 | +0.3785 | 0.0005 | +0.340 | +0.473 | yes | +0.379 | [+0.099, +0.622] | yes |

‡ `robust_stats.py` requires ≥ 20 rows in a bootstrap resample; with n = 19 (`S8`) and n = 11 (`S14`)
every resample is rejected and the interval does not exist.  This is reported as undefined rather
than obtained by relaxing the frozen construction (§7, temptation 2).

### Secondary targets `|a|` and `b` (registered, same construction)

| model | set | n | ρ with \|a\| | 95 % CI | ρ with b | 95 % CI |
|---|---|---|---|---|---|---|
| NAIVE | S8 | 62 | −0.344 | [−0.628, +0.055] | +0.488 | [+0.072, +0.670] |
| NAIVE | S14 | 39 | −0.425 | [−0.697, +0.042] | +0.640 | [+0.294, +0.763] |
| NAIVE | S3 | 81 | −0.404 | [−0.583, −0.122] | +0.375 | [+0.021, +0.556] |
| ELEM | S8 | 62 | −0.432 | [−0.663, +0.023] | +0.239 | [−0.111, +0.463] |
| ELEM | S14 | 39 | −0.555 | [−0.745, −0.058] | +0.189 | [−0.436, +0.493] |
| ELEM | S3 | 81 | −0.194 | [−0.445, +0.199] | +0.219 | [−0.090, +0.418] |
| SPECIES | S8 | 62 | +0.030 | [−0.318, +0.294] | −0.066 | [−0.264, +0.245] |
| SPECIES | S14 | 39 | +0.012 | [−0.427, +0.296] | −0.075 | [−0.374, +0.302] |
| SPECIES | S3 | 81 | −0.012 | [−0.254, +0.174] | −0.141 | [−0.347, +0.148] |
| SPECIES_CONST | S8 | 19 | +0.023 | undefined | −0.121 | undefined |
| SPECIES_CONST | S14 | 11 | +0.018 | undefined | −0.300 | undefined |
| SPECIES_CONST | S3 | 33 | −0.012 | [−0.384, +0.351] | +0.064 | [−0.361, +0.451] |

The quadratic (curvature) term of the same within-series fit is written to `stage1_stats_quad.csv`
and `stage1_perm_null_quad.csv`; it is exploratory throughout and behaves identically (NAIVE S14 vs
`a`: −0.636; SPECIES S8 vs `a`: +0.266, CI [−0.041, +0.458]).

### The dose–response, which is the mechanism

Ordering the models by how exactly they account for the inner-sphere composition gives a monotone
collapse of the correlation on `S8`:

| bookkeeping | model | ρ(slope, a) on S8 | on S14 |
|---|---|---|---|
| none | `NAIVE` | +0.392 | **+0.644** |
| right species, **wrong counts** (`n_fill` read as molecules) | `SPECIES_NFILLCOL` | +0.374 | +0.264 |
| element counts as covariates (gen15's construction B) | `ELEM` | +0.313 | +0.396 |
| exact species counts, no free `δ` (19 constant-`n_ligs` series) | `SPECIES_CONST` | +0.167 | +0.155 |
| exact species counts + free `δ_ligand` | `SPECIES` | **−0.084** | −0.048 |

**The correlation is a monotone function of how badly the composition step is left in.**  That is a
named mechanism, not a shrug: the "computed selectivity" was a count of how many nitrates and waters
the builder put in the first sphere at each metal, and the lanthanide contraction makes that count a
smooth-looking function of the ionic radius.

### Family-wise permutation null

Chemotype-level permutation of the target (2000 replicates, `robust_stats.py` construction: 40
chemotype units, statistic = max |ρ| over the registered family of **4 models × 3 sets = 12
columns**), on chemotype means:

| target | 95th pct of max \|ρ\| under permutation | observed max \|ρ\| | which column | family-wise p |
|---|---|---|---|---|
| `a` | **0.6088** | 0.6017 | `NAIVE`/S14 | 0.057 |
| `|a|` | 0.6133 | 0.4239 | `NAIVE`/S3 | 0.361 |
| `b` | 0.6044 | **0.7503** | `NAIVE`/S14 | 0.003 |

**Not one column clears the bar for the primary target `a` — including the naive artefact itself.**
Only the naive slope's association with the *curvature* `b` clears it, and that is the artefact
measured on the target the artefact most resembles.  With 40 chemotype units and 12 columns, the bar
for "the strongest of the family" is ρ ≈ 0.61; nothing this lead measured reaches it.

### Comparison accounting

`contrasts_stage1.csv`: **90 rows**, **2 registered** (`SPECIES` and `SPECIES_CONST` on `S8` with
target `a`, both value = slope), **88 exploratory** (the other model × set × target combinations, the
exploratory `SPECIES_NFILLCOL` model, and the quadratic-term family).  BH within the registered
family: both adjusted p = 0.518.  BH within L1's exploratory family: smallest raw p = 6 × 10⁻⁶
(`NAIVE`/S14 vs `a` — the artefact), smallest adjusted p = 1.95 × 10⁻⁴.  Every contrast evaluated is
written out; none was promoted between families.

---

## 5. What was **not** run

* The MAE arm `G14+PHYSCYC` and its shuffled null `G14+PHYSCYC_SHUF`, and hence the registered
  contrasts `PHYSCYC_vs_G14`, `PHYSCYC_vs_SHUF`, `PHYSCYC_vs_FLAT`.  The pre-registration gates them
  behind a *positive* Stage 1; Stage 1 is closed, so they are not run and no bench fold was touched.
* Any subset or model beyond the four registered plus the one exploratory bookkeeping variant
  declared before fitting.  The stopping rule ("Stage 1 runs once on the pre-specified sets and
  models") was kept.
* L3b (ranking within a chemotype) is **conditional on L1 being positive** and should be **dropped,
  not run**.

---

## 6. Stage 2 hand-over (delivered regardless of the verdict)

Full detail in `L1_STAGE2_HANDOVER.md`.  Summary:

| | |
|---|---|
| `which xtb` | **not on PATH** (re-checked at build time) — cluster route, submission is the user's |
| built | `results/L1/reference_species/` — 177 free-ligand xyz (ETKDGv3 + MMFF94s, 177/177, formula-verified), 168/177 ligand-at-complex-geometry xyz for strain single points, NO₃⁻, H₂O, 14 Ln³⁺, `manifest.csv`, `completeness.csv`, `jobs.tsv`, `reference_energies_TEMPLATE.csv` |
| completeness | all **1155** distinct `(canonical_smiles, fill_ligand, n_fill, n_ligs, metal)` combinations have every reference they need — `complete = True` on every row |
| submit | `sbatch gen16_leads/results/L1/reference_species/submit_xtb_references.sh` — **361 array tasks**, 1 core, 2 GB, `%20` concurrent, walltime cap 1 h |
| command per task | `xtb <file> --opt --gfn 2 --chrg <q> --uhf 0` (ligands, NO₃⁻, H₂O); `--sp` for the ions and the strain geometries; **no `--alpb`** — no solvation key exists anywhere in the dataset |
| expected | **≈ 6.7 CPU-hours** total, ≈ 20 min wall at 20 concurrent; longest single task ≈ 7 min (largest ligand 169 atoms, median 80) |
| analysis | `scripts/l1_stage2_cycle.py --refs …/reference_energies.csv` — computes `dE`, **aborts** on any missing / non-finite / non-converged reference, then runs exactly the Stage 1 models, sets, statistics and decision rule |

**One result from Stage 1 bounds Stage 2, and it should be read before the cluster time is spent.**
The cycle subtracts `E(Ln³⁺) + n_ligs·E(L) + n_NO3·E(NO₃⁻) + n_H2O·E(H₂O)` from each complex.  Every
one of those terms lies in the column span of the `SPECIES` design — the ion term in the metal fixed
effects, the ligand term in the series fixed effect (constant `n_ligs`) or in `δ_ligand` (varying
`n_ligs`), the two fill terms in `γ_nitrate` and `γ_water`.  By Frisch–Waugh–Lovell the `SPECIES`
residual is therefore **invariant** to the reference energies, so **the registered Stage 2 row is
already known: ρ = −0.084, n = 62, CI [−0.297, +0.240]**.  Verified numerically against a synthetic
reference set: `SPECIES` and `SPECIES_CONST` slopes reproduce Stage 1 to 3 × 10⁻¹¹, while `NAIVE` and
`ELEM` move by thousands of eV per radius unit (the smoke outputs were deleted, not kept).

What the cluster job *can* still settle is `NAIVE`-on-`dE`: series and metal fixed effects only, with
the composition subtracted **exactly, at the true species energies, with no free parameter**.  That is
strictly stronger than Stage 1's `SPECIES`, which has to buy the same bookkeeping with 56 collinear
`δ` coefficients (§3).  `l1_stage2_cycle.py` prints it beside the registered row.  The registered
decision rule stays keyed on `SPECIES` as pre-registered (§7, temptation 1).

---

## 7. Temptations recorded and not acted on

1. **Re-key the decision rule onto `NAIVE`-on-`dE`, or onto `SPECIES_CONST`.**  After seeing that
   `SPECIES` is analytically pinned and that its 56 `δ` terms are collinear with the trend, the
   obvious move is to declare the attenuation-free model the primary one.  `SPECIES` is what the
   pre-registration names as **registered**, and `SPECIES_CONST` is named as *secondary*; the ordering
   was fixed before any number was seen and stands.  Recorded here; not acted on.  (It would not have
   changed the verdict: `SPECIES_CONST` on `S8` is +0.167, also closed.)
2. **Relax the bootstrap's 20-row minimum to get an interval for `SPECIES_CONST` on `S8` (n = 19).**
   The frozen `robust_stats.py` construction rejects every resample at n = 19, so the registered
   interval does not exist for that row.  Lowering the threshold to 15 would have produced one.  Not
   done; the interval is reported as undefined and the `S3` row (n = 33, CI [−0.089, +0.496]) is
   quoted beside it.
3. **Quote `S14` instead of `S8`.**  `S14` is where every construction looks strongest and where
   gen15's headline lives.  `S8` is the registered primary set; `S14` is secondary and is reported
   with its n (39 of 82) everywhere it appears.
4. **Promote `SPECIES_NFILLCOL` (ρ = +0.374 on `S8`, CI excluding zero).**  It is exploratory by
   declaration, it fails the registered ≥ 0.40 bar, and it is *the wrong bookkeeping* — it treats
   `n_fill` as a molecule count, which the audit showed is wrong by a factor of two for nitrate.  Its
   apparent signal is a measure of residual composition step, and it is reported as such.
5. **Fix the phantom element-count column in the frozen `build_block.composition()`.**
   `gen15_curve/exp/phys3d/build_block.py` skips only line 0 of each xyz, so the extxyz *comment* line
   is counted as one atom of a phantom element (`Properties=species:…`, `energy:`).  Real element
   counts are unaffected and the phantom is near-constant within a header style, so it enters
   construction B as a dropped or near-zero-variance covariate; every gen15 number reproduces exactly
   with it in place.  The frozen file was **not** edited (hard rule); gen16's own species
   decomposition filters the column and the defect is reported here.
6. **Drop the two energy-incomplete series so `S14` reads 41 instead of 39, or the reverse.**  Both
   definitions are reported, side by side, with their n, as the reproduction table.

---

## 8. Files

| path | contents |
|---|---|
| `gen16_leads/gen16/l1_cycle.py` | shared code: bookkeeping, the four models, slopes, statistics, decision rule, contrasts |
| `gen16_leads/scripts/l1_stage1.py` | Stage 1 runner (A–E), ~220 s |
| `gen16_leads/scripts/l1_build_references.py` | Stage 2 reference-species builder |
| `gen16_leads/scripts/l1_stage2_cycle.py` | Stage 2 analysis, ready to run |
| `results/L1/stage1_reproduction.csv` | gen15 reproduction + the 39-vs-41 diagnosis |
| `results/L1/stage1_bookkeeping.csv` | composition, fill species, charge convention checks |
| `results/L1/stage1_charge_audit.csv` | per complex: declared charge, Mulliken sum, magmoms |
| `results/L1/stage1_coefficients.csv` | every fitted term of every model, with standard errors |
| `results/L1/stage1_identifiability.csv` | the 56 varying-`n_ligs` series, VIFs and attenuation |
| `results/L1/stage1_slopes.csv` | 410 rows: extractant × model slope / quadratic / rms, set membership |
| `results/L1/stage1_stats.csv`, `stage1_stats_quad.csv` | the registered statistics |
| `results/L1/stage1_loco.csv` | every leave-one-chemotype-out ρ |
| `results/L1/stage1_perm_null.csv`, `stage1_perm_null_quad.csv` | the family-wise permutation bars |
| `results/L1/contrasts_stage1.csv` | 90 contrast rows, `family` ∈ {registered (2), exploratory (88)} |
| `results/L1/stage1_decision.json` | the registered decision, machine-readable |
| `results/L1/reference_species/` | the Stage 2 input set and submit script |
| `results/L1/L1_STAGE2_HANDOVER.md` | the exact submit command, runtime, memory, what to send back |
