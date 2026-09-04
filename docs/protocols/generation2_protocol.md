# Generation-2 benchmark protocol (pre-registered)

This document is written **before** the generation-2 arms are run, so the success
criteria cannot be adjusted after seeing a result. The frozen A0–A6 benchmark is
untouched: with every generation-2 switch off, `build_lanthanide_pair_dataset`
reproduces `cohort_sha256 = af3d91b718d0497818db52be8d93ecdee8c6c80eb8fc0aeda1c25c2c3f5e5f0e`
and the A0–A6 column sets are identical, column for column.

## 1. Question

Beyond a strong 2D ligand baseline (`A2`), does any additional physically
meaningful information improve prediction of `log_SF(A/B) = log_D(A) − log_D(B)`
for genuinely unseen extractants?

## 2. New feature families

Parity under the A/B swap is encoded in the column namespace, and
`reverse_pair_features` raises on any extension column that declares neither —
so a new feature cannot silently break `f(A,B) = −f(B,A)`.

### `3D_PAIR_RESPONSE` (`pair3d__*`, 46 columns)

For ten declared metal-centred quantities `g` (four Ln–donor distance moments,
shell clearance, shell span, coordination number, and three donor-angle
moments), with `r` the tabulated Shannon ionic radius:

| column | definition | parity | emitted for |
|---|---|---|---|
| `pair3d__odd__delta_g` | `g_A − g_B` | odd | all ten |
| `pair3d__odd__reldelta_g` | `(g_A−g_B) / mean(\|g_A\|,\|g_B\|)` | odd | all ten |
| `pair3d__even__absdelta_g` | `\|g_A − g_B\|` | even | all ten |
| `pair3d__even__compliance_g` | `(g_A−g_B) / (r_A − r_B)` | even | all ten |
| `pair3d__odd__excess_g` | `(g_A−g_B) − (r_A − r_B)` | odd | lengths only |

`compliance` is the physically motivated addition: it measures how much of the
lanthanide contraction the cavity actually follows. A rigid, preorganised cavity
gives a value near 0; a cavity that tracks the ion gives ≈1. Nothing in A0–A6
expresses this. `compliance` is NaN when `|r_A − r_B| < 1e-3 Å`.

### `ELEC_COMPLEX` (`elec__even__mean_*`, 7 columns)

Swap-symmetric mean of the two complexes' GFN2-xTB quantities: metal partial
charge, four donor partial-charge moments, dipole magnitude, and the derived
metal-minus-donor charge separation. Complete on all 6 699 pairs.

### `ELEC_PAIR` (21 columns)

Signed difference, absolute difference and radius-normalised compliance of the
same seven quantities. Complete on all 6 699 pairs.

### `ELEC_ENERGY` (1 column)

`elec__odd__delta_complex_total_energy_eV = E_total(A) − E_total(B)`, emitted
**only** when both complexes share `coreCN`, `n_ligs`, `inner_sphere_anion`,
`fill_ligand`, `n_fill` and `metal_ox` (3 655 / 6 699 pairs). Units eV.

It is kept in a family of its own for a scientific reason, not a cosmetic one:
its 45 % missingness is itself associated with the target
(mean `|log_SF|` is 0.99 when it is missing versus 0.38 when present,
point-biserial 0.43). Composition mismatch means the coordination number changes
across the series, which is exactly where selectivity is large. Folding this
column into `ELEC_PAIR` would let a *coordination-composition indicator* be
reported as an electronic effect.

**This is not a binding energy.** The bundle has no free-ion or free-ligand
reference (`features/feature_blocks_manifest.json` records the reference xTB jobs
as `not_run_requires_reference_xtb`), so no thermodynamic cycle can be closed.
The difference retains an unknown additive constant per metal pair; empirically
that constant is not dominant here — the within-metal-pair, ligand-dependent
spread (sd 1.22 eV) exceeds the between-metal-pair spread of means (sd 0.96 eV).

## 3. Descriptors declared unavailable

Not approximated, not relabelled — recorded in
`electronic_provenance.json → unavailable_descriptors`:

* HOMO, LUMO, HOMO–LUMO gap — all-null in the bundle;
* binding energy, strain energy, interaction energy — need reference
  calculations that were queued but never run;
* solvation energy — no implicit-solvent single points exist;
* **the whole `E1` ligand-level electronic arm** — every available electronic
  quantity is a property of the *complex*; there is no free-ligand calculation,
  so a ligand-only electronic block cannot be built;
* `complex_free_energy_eV` — bit-identical to `complex_total_energy_eV`; the
  bundle reports one energy under two names, so only one is used.

## 4. Arm ladder

All arms share one outer fold plan, one inner plan per outer fold, one pair
cohort and one estimator family.

| arm | features |
|---|---|
| `A2` | conditions + Ln + 2D (reference) |
| `A5` = `G1` | `A2` + existing local 3D |
| `A5s` / `A6s` | `A5`/`A6` + swap-symmetric 3D partners |
| `G2` | `A2` + `3D_PAIR_RESPONSE` |
| `G4` | `A2` + independently computed metal-site descriptor blocks |
| `E2` | `A2` + `ELEC_COMPLEX` |
| `E3` | `A2` + `ELEC_PAIR` |
| `E4` | `A2` + `ELEC_ENERGY` |
| `C1` | `A2` + local 3D + pair-response 3D |
| `C2` | `A2` + all electronic |
| `C3` | `A2` + all geometry + all electronic |
| `B0` / `B1` / `B2` | trivial predictor, ridge on `A2`, ridge on `A5` |

`G3` (learned simplicial geometry) is evaluated by the separate simplicial
runner, whose encoder already builds `z_A − z_B` with an exactly odd readout
`0.5·(head(Δ,ctx) − head(−Δ,ctx))`.

### 4a. Learned-geometry encoder ladder

`--simplex-order` selects which level of structure the shared encoder may read.
All four rungs instantiate the same weight shapes — the excluded channels are
zeroed, not removed — so the parameter count is identical across the ladder and
a difference between rungs is attributable to simplicial order rather than to
capacity. Antisymmetry is structural in every rung and is asserted in tests.

| rung | reads |
|---|---|
| `distances` | filtration distances and the metal/donor role flags only; no atom identity, no xTB channels |
| `nodes_distances` | atom identity and per-atom features (including the Ln–atom radial distance); no 1- or 2-simplices |
| `nodes_edges` | the above plus 1-simplex pairwise-distance messages |
| `nodes_edges_triangles` | the full 0/1/2-simplex network (default) |

`--shell-mode coordination`, the default, is the metal-centred learned geometry:
the graph keeps the Ln plus its flagged donors and only Ln–donor–donor
triangles.

`--context-mode conditions_only` is the *learned-geometry-only* arm: the even
context keeps the experimental conditions and the metal-identity means but drops
every ligand 2D descriptor, so nothing about the ligand reaches the model except
through the encoded structure. Conditions are not dropped as well — the target
is a condition-matched difference, so an arm blind to them would answer a
different question rather than a purer one.

### 4b. Secondary 2D-representation ladder (STEP 12)

`A2` carries 2 058 columns, of which all but ~40 are ECFP bits. That is enough
to memorise ligand identity outright and enough to dilute a 21- or 46-column
block. `--two-d-sensitivity` adds a strictly secondary ladder that shrinks the
2D representation and re-asks the same incremental question. It never modifies
`A2`, and the primary claim stays `A2`-referenced.

| arm | features |
|---|---|
| `S1` | conditions + Ln + RDKit 2D descriptors (no fingerprint) |
| `S2` | conditions + Ln + ECFP bits (no descriptors) |
| `S3` | `S1` + local 3D — the S-ladder mirror of `A5`/`G1` |
| `S4` | `S1` + `3D_PAIR_RESPONSE` — the mirror of `G2` |
| `S5` | `S1` + `ELEC_PAIR` — the mirror of `E3` |

`S1` and `S2` are scored against `A2` to price the representation cut itself;
`S3`/`S4`/`S5` are scored against `S1`, because the question they answer is
whether the block becomes useful once fingerprint memorisation is reduced.

**Reading rule, fixed in advance.** A block that helps against `S1` but not
against `A2` is evidence of *redundancy with ECFP*. It is **not** evidence that
the production model should drop the fingerprint, and it does not promote the
block to a positive result.

## 5. Negative controls

Every block claiming metal- or complex-specific information gets a shuffled twin
(`G2_SHUFFLED_s*`, `E2_SHUFFLED_s*`, `E3_SHUFFLED_s*`, `E4_SHUFFLED_s*`,
`G4_SHUFFLED_s*`, plus the legacy `A5_SHUFFLED_s*`). The permutation:

* moves **whole complex-pair vectors** between recipients, so repeated uses of
  one geometry stay mutually consistent;
* permutes only **within one lanthanide-pair label**;
* runs **inside the current inner/outer training subset only** — no validation
  or test row is ever passed to the shuffler, and the audit records
  `test_rows_touched = 0` for every application;
* preserves dimensionality, marginal distributions and model capacity.

Each control is reported twice: against `A2` (so its gain is on the same scale
as the real arm's) and against the arm it controls. The secondary S-ladder
blocks carry the same controls (`S4_SHUFFLED_s*`, `S5_SHUFFLED_s*`), scored
against `A2` and against the S-arm they control.

The learned-geometry arm has its own control, `--geometry-null-seeds`. Each seed
trains the identical architecture for the identical number of epochs and
initialisations on a training fold whose whole `(complex_A, complex_B)`
assignment has been permuted within one `pair_label`, then scores it on the
untouched held-out rows. `permuted_geometry_view` asserts that no row outside
the training subset changed, and `summary.geometry_null_audit` records
`test_rows_touched = 0` per fold. The comparison that matters is
`simplicial_unshrunk` versus `simplicial_shuffled_s*` — the same quantity with
and without the permutation.

## 6. Success criteria (fixed in advance)

A block is called useful only if **all** of the following hold:

1. positive macro (equal-extractant) ΔMAE versus its declared reference;
2. the 95 % group-bootstrap CI of that ΔMAE excludes zero;
3. the sign is stable across the five model seeds;
4. the real block beats its shuffled twin;
5. a meaningful fraction of held-out extractants improves (not one dominant
   ligand carrying the pooled number);
6. the effect does not vanish in the structural-novelty strata.

A small pooled R² increase alone is explicitly **not** evidence.

**Declared reference per arm** (fixed here so criterion 1 is unambiguous, since
not every arm is scored against `A2`):

| arm family | reference | shuffled twin |
|---|---|---|
| `G2`, `G4`, `E2`, `E3`, `E4`, `C1`–`C3`, `A5s`/`A6s` | `A2` | `*_SHUFFLED_s*` |
| `S3`, `S4`, `S5` | `S1` | `S4`/`S5_SHUFFLED_s*`; `S3` has none |
| `S1`, `S2` | `A2` | not applicable — these *remove* information |
| learned geometry (`simplicial_unshrunk`) | `baseline` / `delta3d` | `simplicial_shuffled_s*` |
| encoder rungs | the rung below | inherited from the arm |

Three consequences follow, and they are fixed in advance rather than chosen
after the fact:

* `S1`/`S2` are **diagnostics, not candidates**. They cut information, so a
  negative ΔMAE against `A2` is the expected result and is not a failure; their
  purpose is to price what the fingerprint contributes.
* `S3`/`S4`/`S5` passing criteria 1–6 against `S1` establishes **redundancy with
  ECFP**, per the reading rule in section 4b — not a positive result for the
  block, and not grounds to drop the fingerprint from the production model.
* `S3` has no shuffled twin (`3D_LOCAL` is controlled by the legacy
  `A5_SHUFFLED_s*` against `A2`, not against `S1`), so it cannot satisfy
  criterion 4 and cannot be called useful on its own. It is reported as
  descriptive context for `S4`.

An encoder rung is called informative only if it beats **both** the rung below
it and its own geometry-null twin; beating the null alone shows the encoder
reads *something*, not that the extra simplicial order earned its place.

## 7. Structural-novelty sensitivity (secondary, does not modify the split)

`structural_novelty.json` recomputes, from the frozen OOF predictions, the
maximum Tanimoto similarity from each held-out extractant to the training
extractants **of its own fold**, then reports macro ΔMAE inside strata at
thresholds 0.9 / 0.8 / 0.7 with per-stratum extractant counts. The target is
never used. Strata that would leave an unusably small test population are
reported with their counts rather than forced.

## 8. Cohort accounting

`cohort_report.json` records four nested row cohorts (full baseline,
geometry-available, electronic-available, and their intersection), per-feature,
per-extractant and per-lanthanide missingness, and whether missingness is
associated with the target. Because the extension blocks are emitted as extra
columns on the same pairs — never by filtering pairs — every arm is scored on
the identical 6 699 pairs and the identical folds, so no arm can gain by being
evaluated on an easier subset.

## 9. Seeds

Frozen outer split (`split_seed = 104729`), five model seeds
(42, 7, 137, 2027, 9001). Hyperparameters are selected only inside the inner
grouped CV. Split assignments are never regenerated per model seed.
