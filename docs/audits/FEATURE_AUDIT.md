# FEATURE_AUDIT.md

Descriptor inventory for the `ml_separator` 2D-vs-3D experiment: what is computed, what is
computed *reliably*, and what is declared **unavailable** rather than faked.

Cohort: 6 699 pairs, 34 extractants, 28 exact-ECFP clusters.

---

## 1. Family assignment

Every model column is assigned to exactly one family by
[`feature_registry.py`](../../src/lanthanide_separation/feature_registry.py), fail-closed: a column
matching no explicit rule raises rather than being silently absorbed. Verified one-to-one and
exhaustive.

| Family | Declared | **Informative** (non-constant) | Encoding |
|---|---:|---:|---|
| CONDITIONS | 64 | 24 | absolute, `base__cond__*` |
| LN | 8 | 8 | absolute + symmetric + antisymmetric |
| 2D | 2 058 | **156** | absolute, `base__*` |
| 3D_GLOBAL | 12 | 12 | **difference only**, `delta3d__*` |
| 3D_LOCAL | 38 | 35 | **difference only**, `delta3d__*` |
| 3D_GLOBAL_SYM † | 12 | 12 | symmetric mean, `sym3d__*` |
| 3D_LOCAL_SYM † | 38 | 35 | symmetric mean, `sym3d__*` |

† declared extension arms (§4), reported alongside the frozen contract, never replacing it.

### 1.1 The 2D block is 92 % constant — HIGH

**1 902 of the 2 058 2D columns have zero variance across the entire cohort.** With only 34
distinct ligands, the vast majority of a 2 048-bit Morgan fingerprint never switches on.
Per-arm informative counts:

```
A0    64 declared →   24 informative
A1    72 declared →   32 informative
A2  2130 declared →  188 informative
A3   110 declared →   67 informative
A5  2168 declared →  223 informative
A6  2180 declared →  235 informative
```

This matters for interpretation: "2 130-feature 2D baseline vs 2 168-feature 3D model" is
really "188 vs 223 informative features". Any statement of the form *"3D adds only 38 columns
to a 2 130-column baseline, so an effect would be diluted"* is wrong — 3D adds 35 informative
columns to 188, roughly a 19 % increase in usable dimensionality. Reported in
`feature_registry.json → degenerate_zero_variance_columns`.

Also note 40 of 64 condition columns are constant: the 34-extractant cohort spans a narrow
range of diluents and acids.

---

## 2. Requested descriptors: availability

Against the brief's list. "Available" means computed from coordinates with a documented,
rotation/translation/permutation-invariant definition and non-degenerate values in this cohort.

| Requested descriptor | Status | Implementation |
|---|---|---|
| Coordination number | **Available** | `complex_physical__coordination_number`, `observed_donors_within_3p10A` |
| Donor element identities | **Available** | ranked `polyhedron__donor_atomic_number_*` (bundle) |
| Number of O / N donors | **Available** | `derived_invariant__donor_count_{O,N}` |
| Number of S / P / other donors | **UNAVAILABLE (degenerate)** | identically zero — cohort is O/N only (§3) |
| Ln–donor distances | **Available** | `polyhedron__ln_donor_distance_*` ranked array |
| mean / std / min / max / range | **Available** | `complex_physical__ln_donor_distance_{mean,std,min,max}`, `shell_distance_span` |
| Element-specific Ln–O / Ln–N / Ln–S | **Partial** | Ln–O dominates by construction; Ln–N sparse (mean 0.24 donors); Ln–S/Ln–P impossible |
| Donor–Ln–donor angular statistics | **Available** | `donor_angle_{mean,std,min,max,q25,median,q75}_deg` + Legendre P₁–P₃ |
| **Chelate bite angles** | **UNAVAILABLE** | requires bond connectivity; no MOL2 in bundle (§5) |
| Metal displacement from donor centroid | **Available** | `coordination_shape__metal_offset_from_donor_centroid_fraction` |
| Coordination-sphere radius | **Available** | `ln_donor_distance_mean`; `polyhedron_scalars__coreCN_donor_gap` for shell separation |
| Coordination-polyhedron volume | **Available** | `coordination_shape__coordination_polyhedron_volume` (+ surface area) |
| Global radius of gyration | **Available** | `global_geometry__radius_of_gyration` |
| Asphericity / eccentricity | **Available** | normalized asphericity, relative shape anisotropy, eccentricity — local **and** global |
| Continuous shape / distortion | **Partial** | `radial_distortion_coefficient`, `radial_range_fraction`, `directional_inversion_imbalance`. These are *not* CShM against ideal polyhedra — no reference-polyhedron alignment is implemented. Labelled as distortion proxies, not CShM. |

Deliberately excluded from every arm (audited, not silently relabelled as geometry): 12
xTB/electronic columns — dipole magnitude, metal and donor partial charges — in both the delta
and symmetric namespaces. Their availability would otherwise act as a provenance shortcut and
they are not coordinate geometry, which is what A2-vs-A5 is meant to isolate.

---

## 3. Degenerate descriptors — declared unavailable

`build_feature_registry` now records every zero-variance column
(`degenerate_zero_variance_columns`). In the 3D blocks:

```
delta3d__…donor_count_P      constant 0
delta3d__…donor_count_S      constant 0
delta3d__…donor_count_other  constant 0
sym3d__…  (same three)       constant 0
```

The symmetric versions being constant proves this is **cohort chemistry, not a differencing
artefact**: measured donor composition is

```
mean O donors     8.269   (range 6–9)
mean N donors     0.242   (range 0–3)
mean P/S/other    0.000   (range 0–0)
```

The surviving cohort is essentially a pure hard-oxygen-donor set (diglycolamides, phosphine
oxides, carboxylates) with a small N-donor minority. **Any conclusion about donor-atom
identity is untestable here**, and a claim that 3D descriptors capture donor-type effects
cannot be supported by this data regardless of the model result.

Previously these three columns were passed through as if informative, inflating the D3 block
from 4 usable to 7 declared features.

---

## 4. The encoding asymmetry — CRITICAL

**2D enters absolute; 3D enters only as an A−B difference.**

```python
record[f"base__{column}"]    = row_a[column]              # pairs.py — absolute
record[f"delta3d__{column}"] = value_a - value_b          # pairs.py — difference only
```

Because A and B are the *same ligand with a different lanthanide*, the difference cancels the
coordination environment itself and keeps only its response to the metal swap. Measured:

| D3 feature | sd of Δ | % exactly zero |
|---|---:|---:|
| `donor_count_P` / `_S` / `_other` | 0 | 100 % |
| `donor_count_N` | 0.357 | 85.7 % |
| `coordination_number` | 0.498 | 54.7 % |
| `donor_count_O` | 0.718 | 54.6 % |

So the pre-specified A5/A6 arms cannot test *"metal-centered 3D coordination information
improves prediction"*. They test the narrower *"the metal-induced change in coordination
geometry improves prediction"*. Absolute coordination chemistry — CN 8 vs 9, polyhedron size,
donor composition — is structurally invisible.

### 4.1 Declared fix: symmetric partners

`sym3d__X = (X_A + X_B)/2` is invariant under the A↔B swap, exactly like the existing
`pair__Z_mean` and `pair__ionic_radius_mean`. `reverse_pair_features` negates only
`delta3d__*` and swaps only the named `pair__*_A/B` columns, leaving `sym3d__*` untouched —
which is the correct transformation for a symmetric feature.

**Verified empirically with the symmetric block present:**

```
max |p(A,B) + p(B,A)| = 4.441e-16
```

Exact antisymmetry survives at machine precision, so A5s/A6s inherit the same guarantee as
A5/A6.

This extension was specified from the structural argument above, **before any model
performance number was computed** (see git history: the feature-variance analysis and this
document precede the first ablation result). It is added as arms A5s/A6s *in addition to*
A5/A6, and both are reported.

### 4.2 The dimensional asymmetry — and the secondary S-ladder

The same table exposes a second problem. `A2` carries 2 058 columns of which **156** are
informative, and all but ~40 of the 2 058 are ECFP bits. Every geometry or electronic block
under test is 1–46 columns wide. Two distinct things can therefore hide a real effect:

1. **dilution** — a 21-column block competing with 2 058 candidate splits per node;
2. **memorisation** — with extractant-grouped folds the fingerprint cannot memorise a
   *held-out* ligand, but it can memorise the training ligands well enough that no residual
   signal is left for a small block to explain.

The declared response is the secondary `S1`–`S5` ladder (`--two-d-sensitivity`), not a change
to `A2`. `S1` keeps the RDKit descriptors and drops every fingerprint bit; `S2` is the mirror
image; `S3`/`S4`/`S5` re-add local 3D, pair-response 3D and pair-response electronic on top of
`S1`. The registry partitions the 2D family by namespace only — every column keeps the family
label it already had, and `S1 ∪ S2 = A2` exactly (asserted in
`tests/test_extension_features.py`).

**Fixed in advance:** a block that helps against `S1` but not against `A2` is reported as
*redundancy with ECFP*. It is not a positive result, and it is not an argument for dropping
the fingerprint from a production model.

---

## 5. Connectivity is unavailable — chelate bite angles cannot be computed

The brief requires MOL2 connectivity to be preferred over inferring bonds from XYZ. In this
bundle:

```
geometries/         1155 files, all .xyz, zero .mol2, zero .sdf
provenance mol2_path  756 entries → all point at /nfs/home/nshank3/… ; 0 exist on disk
VR asset keys       coordinates, atomic_numbers, partial_charges, is_metal,
                    is_coord_donor, edge_index (distance-filtration), triangle_index
```

The `edge_index` in the VR asset is a **distance filtration**, not a chemical bond table — an
edge exists when two atoms are within a cutoff, which is not the same as being bonded.

Consequences, recorded rather than worked around:

- **Chelate bite angles: UNAVAILABLE.** Identifying which two donors belong to the same
  chelate ring needs the bond graph. Deriving it from distances would be exactly the
  "inventing missing chemistry" the brief forbids. `geometry_descriptors.py` contains no
  `bond` / `connectivity` / `bite` / `mol2` logic — correctly, it does not fake them.
- **Donor assignment is inherited, not re-derived.** `is_coord_donor` comes from the upstream
  Architector `COORDLIST` / `DONOR_TYPES` specification. Good provenance, but it means donor
  identity is not independently verified against the coordinates in this repository.
- The angular block is therefore *all* donor–Ln–donor angles, not ring-resolved bite angles.

---

## 6. Descriptor correctness

- **Invariance.** `tests/test_geometry_descriptors.py` checks rigid motion + donor relabelling
  invariance at 11 decimal places for every descriptor. All pass except one:
  `coordination_eccentricity` on a **perfect octahedron**, where the true value is 0 and the
  two computations return 4.08e-08 and 3.16e-08. This is catastrophic cancellation on a
  degenerate-by-symmetry quantity, not an invariance break — the eigenvalue gap vanishes for
  an ideal polyhedron. Real Architector geometries are distorted and the quantity is
  well-conditioned there (cohort sd = 0.104). **Recommendation:** loosen the tolerance to an
  absolute 1e-6 for this descriptor and document the degeneracy, rather than leave a red test.
- **Zero-padding.** Ranked arrays in the bundle are zero-padded; `_add_derived_invariant_3d_features`
  masks non-physical entries (`distance <= 0`, `angle <= 0 or > 180`) before pooling. Correct.
- **Row-wise only.** Every derived feature is an `axis=1` reduction over that row's own arrays —
  no cross-row pooling, so no possibility of test rows contributing to a training feature.
- **Radius normalisation.** Partially implemented: `shell_clearance_mean = ln_donor_distance_mean
  − Ionic Radius_metal` uses a fixed tabulated ionic radius, and `shell_distance_cv` is
  scale-free. The brief's `d_norm = d/(r_Ln + r_X)` form is **not** implemented, because it
  needs per-donor element radii matched to each ranked distance, and the ranked distance and
  ranked atomic-number arrays are not guaranteed to be co-ordered in the bundle. Recorded as a
  gap rather than approximated with a possibly mismatched pairing. Neither constant is
  estimated from the target data — both are tabulated.

---

## 7. Actions taken

1. Zero-variance columns now recorded as unavailable in `feature_registry.json`.
2. Symmetric `sym3d__` block + declared A5s/A6s arms added; antisymmetry verified at 4.4e-16.
3. xTB/electronic exclusion extended to the symmetric namespace, so A5s/A6s cannot smuggle in
   quantities A5/A6 exclude.
4. Chelate bite angles and `d/(r_Ln+r_X)` normalisation declared **unavailable** with reasons,
   not approximated.

## 8. Open gaps (not fixed, deliberately)

- Continuous shape measures against ideal reference polyhedra (true CShM) are not implemented;
  current distortion features are proxies and are labelled as such.
- Element-resolved Ln–O / Ln–N distance statistics are not separated, pending a verified
  co-ordering guarantee between the ranked distance and ranked donor-element arrays.
