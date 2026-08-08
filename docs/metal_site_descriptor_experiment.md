# Metal-site descriptors: what else the 3D structures can give us

Date: 2026-08-08
Cohort: the frozen 1,081-pair adjacent-lanthanide cohort
(`cohort_sha256 = e557b9f5…`), unchanged by everything described here.
Asset: `dataset with 3D structures/features/vietoris_rips_inputs.npz`
(`eb7279b5…`), unchanged.

## Why look again at the 3D block

The guarded simplicial network was a clean negative result. Its raw branch lost
about 0.15 R2 and 0.025 macro-group MAE to adaptive tabular Delta3D, and the
guard reduced it to a mostly-exact Delta3D fallback. The natural reading is that
a learned representation is the wrong tool at 1,081 pairs across 26 exact-ECFP
clusters, not that the geometries are uninformative.

What the frozen 29-column Delta3D contract actually uses is narrow: coordination
number, donor-count, dipole magnitude, partial charges, four Ln–donor distance
moments, pair-angle moments with unweighted Legendre means to rank three, three
distance quantiles, and three shell-clearance scalars. Every one of those is a
first-shell scalar, and every angular term is computed with all donors weighted
equally. The complexes in the asset carry 34–484 atoms each; only the 8–9 donors
are ever consulted.

## Five things extractable from these 3D structures

1. **Radially weighted rotational invariants of the donor shell (ligand-field
   strength).** For a point-charge model the crystal-field coefficients satisfy
   `sum_q |B_kq|^2 ∝ sum_ij w_i w_j P_k(cos θ_ij)` with `w_i = r_i^-(k+1)`. The
   spherical-harmonic addition theorem collapses this to a double sum over
   donor–donor angles, so it needs no frame, no harmonic basis and no donor
   ordering. Unlike the existing pair-angle means it couples the radial
   contraction to the angular arrangement — the exact coupling that separates two
   adjacent lanthanides.
2. **Solid-angle enclosure of the metal by the whole complex.** Cast rays from
   the metal over a fixed Fibonacci sphere and intersect them with van der Waals
   spheres: buried fraction at several radii, closest-contact distance, open
   fraction, and the anisotropy of the remaining opening. This is the only
   candidate that reads the ligand body, which is what determines whether the
   cavity is rigid enough to resist the contraction.
3. **Continuous shape measures against ideal CN-8/CN-9 polyhedra.** The standard
   inorganic descriptor (square antiprism, tricapped trigonal prism, dodecahedron
   …). Chemically the most interpretable option, but an exact measure needs an
   optimisation over rotations and 8!/9! vertex permutations per reference
   polyhedron, which is far heavier than everything else on this list.
4. **Element-resolved radial density around the metal.** Gaussian-smeared counts
   of C/N/O/H/P/S/Cl in shells out to ~8 Å: an ACSF-style fingerprint of the
   first and second coordination spheres, robust to coordinate noise but wide.
5. **Persistent-homology summaries of the metal-centred VR filtration.**
   Persistence entropy, total persistence and Betti curves as a compact tabular
   replacement for the failed network. Cheap, but a 9–10 node shell filtered to
   4 Å has almost no non-trivial topology to summarise.

## What was chosen and why

Options 1 and 2 were implemented as
`src/lanthanide_separation/geometry_descriptors.py`. They were the only two
candidates that improved the group-level endpoints in a five-seed local screen,
they are exactly invariant to rotation, reflection and donor relabelling, they
cost about 1.6 s for all 1,155 geometries, and they are integral or
angle-averaged quantities — which matters, because the source geometries are
heterogeneously relaxed (the accepted-geometry manifest contains
`emergency_unrelaxed` structures), so any descriptor that depends on sub-0.01 Å
precision would mostly measure build provenance.

Option 3 was rejected on cost for what it adds over option 1, option 4 on width,
option 5 on the triviality of the topology at this shell size.

## The result that matters: the gain is inside its own null band

A five-seed screen on the frozen cohort gave, against the frozen 29-column
contract (positive = better, `n/5` = runs improved):

| feature set | R2 | group-balanced R2 | macro-group MAE | sign accuracy |
|---|---:|---:|---:|---:|
| + ligand field | +0.0004 (3/5) | +0.0270 (5/5) | +0.0020 (5/5) | +0.0176 (5/5) |
| + enclosure | +0.0061 (4/5) | +0.0112 (5/5) | +0.0010 (4/5) | +0.0006 (3/5) |
| + both | +0.0004 (3/5) | +0.0270 (5/5) | +0.0020 (5/5) | +0.0176 (5/5) |

Five out of five on both group-level endpoints looks convincing, and it is not.
Repeating the screen with the descriptor values **permuted across geometries** —
same width, same marginals, no chemistry — gave:

| permuted null realisation | group-balanced R2 | macro-group MAE |
|---|---:|---:|
| free, seed 11 | −0.0032 (2/5) | −0.0001 (2/5) |
| free, seed 22 | +0.0133 (4/5) | +0.0015 (4/5) |
| free, seed 33 | +0.0433 (5/5) | +0.0039 (5/5) |
| metal-preserving, seed 77 | +0.0077 (3/5) | +0.0004 (2/5) |

One null realisation beats the real block on both primary endpoints, also 5/5.
Adding continuous columns of any kind to a mostly-binary 2,130-column
fingerprint matrix changes how ExtraTrees samples split candidates, and at this
cohort size that artefact is worth ±0.02–0.04 group-balanced R2 — an order of
magnitude larger than the +0.0019 the simplicial study was weighing.

Two consequences:

- No claim is being made that these descriptors improve the model. The local
  evidence says the raw gain is not separable from the width artefact.
- The earlier simplicial comparison shares this blind spot. Its matched 2D+2D
  control equalises the tuning budget but not the feature-block width, so it
  never measured this confound. That does not overturn the negative conclusion —
  it makes it stronger, since the SNN effect was far below the noise floor.

## The experiment that settles it

`slurm/descriptor_multiseed.slurm` runs two arms over the same prespecified seed
plan, folds, cohort and contract:

- `arm_real` — descriptors computed from the geometry they belong to.
- `arm_null` — the same descriptors, detached from their geometry by a seeded
  metal-preserving permutation (values are exchanged only among geometries of
  the same lanthanide, so the A-minus-B contrast keeps its physical scale and
  only the ligand-identity link is destroyed).

Inside each run, `delta3d_extended` is built exactly like `delta3d`: same seed
stream, same inner-CV budget, same blend rule, differing only in the descriptor
columns. The pre-declared primary contrast is `extended − delta3d`, and the
quantity that carries a chemical claim is the paired excess of the real arm over
the null arm, written to `comparison/arm_comparison.md`.

### Decision rule, declared before the run

The descriptor block is worth keeping only if, on the held-out exact-ECFP
protocol:

- the paired excess over the permuted null is positive for **group-balanced R2**
  and for **macro-group MAE reduction**, in every one of the five seeds;
- the real arm shows no systematic row-level MAE regression;
- the cohort hash, protocol fingerprint and seed plan are identical across both
  arms, and every fold passes the leakage audit.

If the excess is positive but narrow, the correct next step is more permutation
realisations, not a publication. If it fails, the honest conclusion is that the
compact tabular Delta3D representation remains the preferred model and that this
cohort cannot resolve descriptor effects below roughly 0.03 group-balanced R2 —
which is itself the most useful number to know before designing the next study.

## Running it

Review the plan first; the wrapper defaults to `DRY_RUN=1` and creates nothing.

```bash
PARTITION=<partition> ACCOUNT=<account> bash slurm/submit_descriptor_multiseed.sh
```

It prints the two `sbatch` lines and a self-contained `env … DRY_RUN=0 …`
command to run after review. Pin the assets when submitting for real:

```bash
EXPECTED_DATASET_SHA256=fefbefc6fe993aa9ce9db1a0c338adb9e5f58a8b75bab084cc1df4e024faf5dd EXPECTED_VR_SHA256=eb7279b5ea78ecc8994e942c1cc15c7b398c904b60d21c591b8f1b93f761ce37 bash slurm/submit_descriptor_multiseed.sh
```

The array is 10 tasks: 0–4 are the real arm, 5–9 the null arm, two concurrent by
default, 8 CPUs and 24 GB each. A dependent aggregate job runs
`aggregate_delta3d_runs.py --comparison extended_vs_delta3d` per arm and then
`compare_descriptor_arms.py`; every stage is fail-closed.

Useful knobs: `DESCRIPTOR_PROFILE=full` widens the block from 11 columns to 28
as a deliberate width sensitivity; `NULL_PERMUTATION=free` uses the weaker null;
`DESCRIPTOR_BLOCKS=ligand_field` runs a single-factor ablation.

## Code changes

- `src/lanthanide_separation/geometry_descriptors.py` — new. Descriptor
  computation, the `core`/`full` profiles, the permutation control, and an
  audited join that fails closed on index drift or missing coverage.
- `src/lanthanide_separation/pairs.py` — `PairDataset` now separates
  `descriptor_columns` from the frozen `delta3d_columns` and exposes
  `extended_columns`. Descriptor blocks are opt-in, so the default contract is
  bit-for-bit the frozen one and the cohort hash is unchanged.
- `src/lanthanide_separation/evaluation.py` — optional `delta3d_extended`
  branch, its adaptive blend weight, its improvements and its bootstrap
  comparisons.
- `scripts/run_delta3d_benchmark.py` — descriptor CLI, asset hash pinning, a
  `geometry_descriptor_audit.json` artifact, and a per-artifact SHA-256 map in
  `summary.json` so a `_SUCCESS.json` can be verified from the run directory
  alone.
- `scripts/aggregate_delta3d_runs.py` — `--comparison extended_vs_delta3d`, and
  a refusal to pool runs whose descriptor block, profile or permutation differ.
- `scripts/compare_descriptor_arms.py` — new. The real-versus-null verdict.
- `.gitignore` — published run artifacts of the current protocols are tracked;
  attempts, logs, locks and fitted forests are not.
