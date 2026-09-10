# L1 Stage 2 — hand-over: the reference-species jobs the user submits

*Everything below is ready.  `xtb` is **not on this machine** (`which xtb` re-checked at build time:
not on PATH), so the cluster route was assumed from the start.  Cluster submission is yours; no job
was submitted from here.*

---

> **STATUS: RUN, AND THE LEAD IS CLOSED (2026-09-10).**  There was no cluster; the job was run on
> this machine.  `xtb` 6.7.1pre was installed from the official upstream release (SHA-256 verified,
> unpacked to `C:\Users\Bandai\opt\xtb-6.7.1`, outside the repository), and all **361 reference
> species converged in 4.8 minutes of wall time on 6 workers** with
> `scripts/l1_run_references_local.py --workers 6`.  `reference_energies.csv` is in this directory.
> The analysis below was then run unchanged and **L1 is closed**: rho = -0.084, CI [-0.297,
> +0.240], family-wise permutation p = 0.94, from an estimator of reliability 0.80.  See
> `DECISION_REPORT.md` section 4.  The SLURM instructions below are kept for the record and for
> anyone reproducing this on a queue; the local runner is the route that was used.

---


## 1. What to run

```bash
# on the cluster, from the copied reference_species/ directory
mkdir -p logs out
sbatch submit_xtb_references.sh          # 361 array tasks, 1 core, 2 GB, %20 concurrent
# when the array is finished:
bash collect_reference_energies.sh > reference_energies.csv
```

Then send `reference_energies.csv` back, drop it into
`gen16_leads/results/L1/reference_species/`, and run **one** command here:

```bash
PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 .venv/Scripts/python.exe \
    gen16_leads/scripts/l1_stage2_cycle.py \
    --refs gen16_leads/results/L1/reference_species/reference_energies.csv
```

That script computes `dE`, aborts on any missing / non-converged reference, and runs **exactly** the
Stage 1 models, sets, statistics and registered decision rule on `dE`.  Nothing about the protocol
changes between the stages; only the energy column does.

| | |
|---|---|
| submit script | `gen16_leads/results/L1/reference_species/submit_xtb_references.sh` |
| analysis script | `gen16_leads/scripts/l1_stage2_cycle.py` |
| tasks | **361** (177 free-ligand optimisations, 168 strain single points, NO₃⁻, H₂O, 14 Ln³⁺) |
| per task | 1 core, 2 GB, walltime cap 1 h (longest estimated task ≈ 7 min) |
| expected total | **≈ 6.7 CPU-hours**; ≈ 20 min wall at 20 concurrent tasks. Treble it for a safe upper bound (≈ 20 CPU-h) — the estimate is a quadratic-in-N model anchored at 20 min for a 286-atom system, and the largest ligand here is only 169 atoms |
| largest ligand | 169 atoms; median 80 atoms |

## 2. The exact command line each task runs

```
xtb <file> --opt --gfn 2 --chrg <q> --uhf 0        # ligands, NO3-, H2O
xtb <file> --sp  --gfn 2 --chrg <q> --uhf 0        # Ln3+ ions, strain geometries
```

**Solvent: none.** No solvation key exists anywhere in the dataset — the extxyz comment lines carry
only `energy` / `free_energy` / `dipole` / `forces` / per-atom `charge`, one file family is a plain
`xtb: 6.7.1 (edcfbbe)` log header, and the energies are gas-phase GFN2 totals in magnitude.  Grep for
`alpb` / `gbsa` / `solvent` over the whole `dataset with 3D structures/` tree hits only the
*experimental* diluent columns of `dataset.parquet`.  **If you know the complexes were run with an
implicit solvent, add the identical `--alpb <solvent>` to every line of the submit script** — a
reference computed in a different environment than the complex silently reintroduces the very
composition step this lead exists to remove.

## 3. Charge convention (verified on all 1155 complexes, not assumed)

| species | charge | uhf |
|---|---|---|
| Ln³⁺ ion | **+3** | 0 |
| free ligand (`canonical_smiles`, as written) | **0** | 0 |
| nitrate NO₃⁻ | **−1** | 0 |
| water H₂O | **0** | 0 |
| complex | **3 − n_NO₃** | 0 |

How this was established (`stage1_bookkeeping.csv`, and it is the load-bearing bookkeeping finding
of Stage 1):

* the non-metal atoms of every one of the 1155 complexes decompose **exactly** into `n_ligs` copies
  of the neutral RDKit formula of `canonical_smiles`, plus `n_NO3` nitrates, plus `n_H2O` waters —
  1155 / 1155, no residue, no deprotonated ligand (all 177 ligands have RDKit formal charge 0);
* **`n_fill` counts fill *donor sites*, not molecules.**  Nitrate is bidentate, water monodentate, so
  `n_fill = 2·n_NO3 + n_H2O`.  A nitrate series with an odd `n_fill` (167 complexes) has one water
  in addition to the nitrates.  Taking `n_fill` at face value as a molecule count is wrong by a
  factor of two for nitrate and mis-assigns 167 waters;
* the per-atom `initial_charges` column sums to `3 − n_NO3` on **all 1145** files that carry it (901
  at +3, 181 at +2, 63 at +1), and the post-relaxation Mulliken `charge` column sums to the same
  value within 0.02 e on **all 1116** files that carry one;
* **on `--uhf 0`:** `initial_magmoms` is zero on 1134 of those 1145 files and carries the formal
  f-electron count on the metal in eleven (Eu 6.0, Yb 1.0).  GFN2 parameterises the lanthanides with
  **f-in-core**, so the valence shell is closed and that metadata cannot enter the calculation; it is
  also set inconsistently (other Eu and Yb complexes carry 0.0).  `--uhf 0` is correct for every
  reference, including the ions.  The eleven files are listed in `../stage1_charge_audit.csv`.

## 4. What is in `reference_species/`

| path | contents |
|---|---|
| `ligands/lig_000.xyz … lig_176.xyz` | 177 free ligands, RDKit **ETKDGv3 + MMFF94s** (UFF fallback), neutral as written; 177/177 embedded, formula verified against the SMILES |
| `strain/strain_lig_XXX.xyz` | **168 of 177** ligands *as they sit in one accepted complex*, extracted by a covalent-radius connectivity cut with the metal and fill removed, for `E_strain = E(ligand @ complex geometry) − E(free ligand)`.  The 9 failures (`strain_extraction.csv`) are complexes whose ligand copies interpenetrate at bond length (~1.5 Å C–C bridges), so no connectivity cut separates them; a strain single point on such a geometry would not be meaningful anyway.  **Strain is a bonus: the registered cycle does not use it.** |
| `species/nitrate.xyz`, `species/water.xyz` | idealised D₃ₕ NO₃⁻ (r = 1.26 Å) and C₂ᵥ H₂O (0.958 Å, 104.5°); both are optimised by the job, so only the topology has to be right |
| `ions/La.xyz … ions/Lu.xyz` | the 14 Ln³⁺ ions, one atom at the origin.  GFN2 treats the lanthanides with **f-in-core**, so a bare 3+ ion is a closed-shell single-atom job and `--uhf 0` is correct |
| `manifest.csv` | every species → file, charge, uhf, run mode, atom count, cost estimate, input SHA-256 (16 hex) |
| `completeness.csv` | all **1155** distinct `(canonical_smiles, fill_ligand, n_fill, n_ligs, metal)` combinations with the references each needs; **`complete = True` for every row** |
| `jobs.tsv` / `jobs.csv` | the array task table, longest job first |
| `reference_energies_TEMPLATE.csv` | the exact schema to return |
| `summary.json` | the numbers in this table, machine-readable |

## 5. What to send back

`reference_energies.csv`, one row per species:

| column | required | meaning |
|---|---|---|
| `species` | yes | the id from `manifest.csv` (`lig_037`, `nitrate`, `water`, `Nd3+`, `strain_lig_037`) |
| `energy_Eh` | one of the two | total energy in Hartree |
| `energy_eV` | one of the two | total energy in eV (used directly if present) |
| `converged` | recommended | anything other than `ok` / `true` / `1` **aborts** the analysis |
| `xtb_version` | recommended | recorded for provenance |

`collect_reference_energies.sh` writes exactly this file.  A missing, non-finite or non-converged
reference aborts `l1_stage2_cycle.py` with a list of what is missing — it is never filled in or
skipped.

## 6. Read this before you spend the cluster time

Stage 1 proved an identity that bounds what Stage 2 can return, and it is better to know it now.

**The registered Stage 2 statistic on `SPECIES` is already fixed by Stage 1 and cannot move.**  The
cycle subtracts, from each complex, `E(Ln³⁺) + n_ligs·E(L) + n_NO3·E(NO₃⁻) + n_H2O·E(H₂O)`.  Every
one of those four terms lies in the column span of the `SPECIES` design: the ion term is a pure metal
effect (metal fixed effects), the ligand term is a per-series constant times `n_ligs` (absorbed by
the series fixed effect where `n_ligs` is constant and by `δ_ligand` where it varies), and the two
fill terms are exactly `γ_nitrate` and `γ_water`.  By Frisch–Waugh–Lovell the `SPECIES` residual —
and therefore every per-extractant slope, every correlation, every interval — is **invariant** to the
reference energies.  Verified numerically with a synthetic reference set: `SPECIES` and
`SPECIES_CONST` slopes reproduce Stage 1 to `3e-11`, while `NAIVE` and `ELEM` move by thousands of eV
per radius unit.

So Stage 2's registered row will read exactly `ρ = −0.084, n = 62, CI [−0.297, +0.240]`, whatever the
cluster returns.  **The row that is worth the cluster time is `NAIVE` on `dE`**, which is the
construction the original hypothesis actually names: series and metal fixed effects only, with the
composition subtracted *exactly, at the true species energies, with no free parameter*.  That is
strictly stronger than Stage 1's `SPECIES`, because Stage 1 has to buy the same bookkeeping with 56
free `δ_ligand` coefficients that are badly collinear with the radius trend (median VIF 6.3, median
attenuation 0.27 — see `L1_REPORT.md` §3), whereas the cycle pays nothing: `E(L)` is computed, not
fitted.  For the 43 of 62 `S8` extractants whose series changes `n_ligs`, that difference is the
whole question.

`l1_stage2_cycle.py` prints `NAIVE`-on-`dE` beside the registered row for exactly this reason.  The
registered decision rule stays keyed on `SPECIES`, as pre-registered; re-keying it onto `NAIVE` after
seeing Stage 1 would be changing an endpoint after seeing a number, and is recorded in
`L1_REPORT.md` §7 as a temptation resisted rather than acted on.

**Two sanity checks the script prints, and what they mean if they fail.**

1. `γ_nitrate` and `γ_water` refitted on `dE` should now be *small* (a mean Ln–fill binding energy,
   order 1–10 eV), not a free-species total energy.  If they come back near −432 eV / −139 eV, the
   references were computed at a different level of theory or charge from the complexes.
2. The median `dE` must be **negative** (complexes are bound).  A positive median means the charge
   convention or the solvent model does not match.
