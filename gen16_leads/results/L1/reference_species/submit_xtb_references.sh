#!/bin/bash
#SBATCH --job-name=xtb_l1_refs
#SBATCH --array=1-361%20
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#
# L1 Stage 2 reference species for the thermodynamic cycle
#   dE = E(complex) - E(Ln3+) - n_ligs*E(L) - n_NO3*E(NO3-) - n_H2O*E(H2O)
#
# Level of theory: GFN2-xTB, GAS PHASE, no implicit solvent.  The 1155 complex energies in
# features/complex_physical_scalars.parquet carry no solvation key anywhere in the dataset
# (extxyz comment lines hold only energy / free_energy / dipole / forces; one file family is a
# plain xtb 6.7.1 log header), and their magnitudes are gas-phase GFN2 totals.  IF you know the
# complexes were run with an implicit solvent, add exactly the same '--alpb <solvent>' to every
# line below; a reference computed in a different environment than the complex silently
# reintroduces the composition step this whole lead exists to remove.
#
# Charge convention (verified on all 1155 files, Stage 1): metal 3+, ligands neutral as written
# in canonical_smiles, nitrate -1, water 0, so the complex charge is 3 - n_NO3.  --uhf 0
# everywhere: GFN2 treats the lanthanides with f-in-core, so Ln3+ is a closed-shell single atom.
#
# Run from this directory (results/L1/reference_species/):
#     mkdir -p logs out
#     sbatch submit_xtb_references.sh
# then collect with:
#     bash collect_reference_energies.sh > reference_energies.csv
#
set -euo pipefail
module load xtb 2>/dev/null || true          # or: conda activate xtb-env
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_STACKSIZE=4G
ulimit -s unlimited

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" jobs.tsv)
SPECIES=$(echo "$LINE" | cut -f1)
FILE=$(echo    "$LINE" | cut -f2)
CHRG=$(echo    "$LINE" | cut -f3)
UHF=$(echo     "$LINE" | cut -f4)
MODE=$(echo    "$LINE" | cut -f5)

WORK="out/${SPECIES}"
mkdir -p "$WORK"
cp "$FILE" "$WORK/input.xyz"
cd "$WORK"

if [ "$MODE" = "opt" ]; then
    xtb input.xyz --opt --gfn 2 --chrg "$CHRG" --uhf "$UHF" > xtb.out 2> xtb.err
else
    xtb input.xyz --sp  --gfn 2 --chrg "$CHRG" --uhf "$UHF" > xtb.out 2> xtb.err
fi
grep -E "TOTAL ENERGY" xtb.out | tail -1
