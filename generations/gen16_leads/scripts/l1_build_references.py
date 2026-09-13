"""L1 Stage 2, part 1 -- build the reference-species input set for the thermodynamic cycle.

PRE_REGISTRATION.md section 3, L1, "Stage 2 -- reference species (cluster; the user submits)".

The cycle the queue file asks for is

    dE(ligand, metal, composition) = E(complex)
                                   - E(Ln3+)
                                   - n_ligs * E(free ligand)
                                   - n_NO3  * E(NO3-)
                                   - n_H2O  * E(H2O)

with every term at the same level of theory and the same charge convention.  Stage 1 established
(``results/L1/stage1_bookkeeping.csv``) that all 1155 accepted complexes decompose *exactly* into
one metal, ``n_ligs`` copies of the neutral ligand as written in ``canonical_smiles``, ``n_NO3``
nitrates and ``n_H2O`` waters, and that the xTB total charge is ``3 - n_NO3``.  So the reference
set is closed and small: 177 free ligands + NO3- + H2O + 14 Ln3+ ions = 193 species.

This script writes, under ``results/L1/reference_species/``:

  ligands/lig_<idx>.xyz     177 free ligands, RDKit ETKDGv3 + MMFF94s, neutral as written
  strain/strain_<idx>.xyz   the same ligand *as it sits in one accepted complex* (metal and fill
                            removed by a covalent-radius connectivity cut), for a single-point
                            strain energy E_strain = E(ligand @ complex geometry) - E(free ligand)
  species/nitrate.xyz       NO3-, charge -1
  species/water.xyz         H2O, charge 0
  ions/<El>.xyz             the 14 Ln3+ ions, charge +3 (GFN2 treats Ln with f-in-core, so a bare
                            3+ ion is a closed-shell single-atom job)
  manifest.csv              every species -> file, charge, multiplicity (uhf), run mode, n_atoms
  completeness.csv          every distinct (canonical_smiles, fill_ligand, n_fill, n_ligs, metal)
                            combination in accepted_geometries.csv with the references it needs
  submit_xtb_references.sh  the SLURM array job
  reference_energies_TEMPLATE.csv  the schema ``l1_stage2_cycle.py`` reads back

Nothing is submitted from here; cluster submission is the user's.

Run from the repo root:
    PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 .venv/Scripts/python.exe generations/gen16_leads/scripts/l1_build_references.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen16 import bootstrap  # noqa: E402
from gen16 import l1_cycle as L  # noqa: E402

OUT = L.RESULTS / "reference_species"
T0 = time.time()

# covalent radii (Angstrom, Cordero 2008) for the elements that occur in these ligands; the metal
# is deleted before the graph is built, so no lanthanide radius is needed.
COVALENT = {"H": 0.31, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "Si": 1.11,
            "P": 1.07, "S": 1.05, "Cl": 1.02, "Br": 1.20, "I": 1.39, "Se": 1.20, "As": 1.19}
BOND_SLACK = 0.40          # bonded if d <= r_i + r_j + slack

# a transparent cost model, anchored on the brief's "~10-30 min for the largest 286-atom ligand"
OPT_MIN_AT_286 = 20.0      # minutes for a GFN2 --opt of the largest ligand, one core
SP_FRACTION = 0.06         # a single point is ~6 % of the optimisation it would start from


def log(*a) -> None:
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def opt_minutes(n_atoms: int) -> float:
    """Quadratic-in-N wall-clock estimate for one GFN2 optimisation on one core."""
    return float(np.clip(OPT_MIN_AT_286 * (n_atoms / 286.0) ** 2, 0.2, 60.0))


# --------------------------------------------------------------------------------------
# xyz io
# --------------------------------------------------------------------------------------
def read_xyz(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text().splitlines()
    n = int(lines[0].split()[0])
    els, xyz = [], []
    for line in lines[2:2 + n]:
        t = line.split()
        els.append(t[0])
        xyz.append([float(t[1]), float(t[2]), float(t[3])])
    return els, np.asarray(xyz, dtype=float)


def write_xyz(path: Path, els, xyz, comment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = [str(len(els)), comment]
    for e, (x, y, z) in zip(els, xyz):
        out.append(f"{e:<3s} {x:16.8f} {y:16.8f} {z:16.8f}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------------------
# free ligands from SMILES
# --------------------------------------------------------------------------------------
def embed_ligand(smiles: str, seed: int = 0xC0FFEE):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    ps = AllChem.ETKDGv3()
    ps.randomSeed = seed
    ps.useSmallRingTorsions = True
    ps.maxIterations = 2000
    ok = AllChem.EmbedMolecule(mol, ps)
    method = "ETKDGv3"
    if ok != 0:                                   # fall back to plain distance geometry
        ps2 = AllChem.ETKDGv3()
        ps2.randomSeed = seed
        ps2.useRandomCoords = True
        ps2.maxIterations = 4000
        ok = AllChem.EmbedMolecule(mol, ps2)
        method = "ETKDGv3+randomCoords"
    if ok != 0:
        return None, "embedding failed", np.nan
    ff = "none"
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            res = AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s", maxIters=5000)
            ff = "MMFF94s" + ("" if res == 0 else " (not fully converged)")
        else:
            res = AllChem.UFFOptimizeMolecule(mol, maxIters=5000)
            ff = "UFF" + ("" if res == 0 else " (not fully converged)")
    except Exception as exc:                       # pragma: no cover - defensive
        ff = f"force field failed: {exc}"
    conf = mol.GetConformer()
    els = [a.GetSymbol() for a in mol.GetAtoms()]
    xyz = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                    for i in range(mol.GetNumAtoms())])
    return (els, xyz), f"{method} + {ff}", float(Chem.GetFormalCharge(mol))


# --------------------------------------------------------------------------------------
# the ligand as it sits in a complex: covalent-radius connectivity, metal removed
# --------------------------------------------------------------------------------------
def components(els: list[str], xyz: np.ndarray, drop: set[int]) -> list[list[int]]:
    keep = [i for i in range(len(els)) if i not in drop]
    r = np.array([COVALENT.get(els[i], 0.8) for i in keep])
    P = xyz[keep]
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    thr = r[:, None] + r[None, :] + BOND_SLACK
    adj = (D <= thr) & ~np.eye(len(keep), dtype=bool)
    seen = np.zeros(len(keep), dtype=bool)
    out = []
    for s in range(len(keep)):
        if seen[s]:
            continue
        comp, q = [], deque([s])
        seen[s] = True
        while q:
            u = q.popleft()
            comp.append(u)
            for v in np.flatnonzero(adj[u] & ~seen):
                seen[v] = True
                q.append(v)
        out.append([keep[i] for i in sorted(comp)])
    return out


def extract_ligand(xyz_path: Path, metal: str, formula: Counter):
    """Return one ligand copy's (elements, coordinates) from an accepted complex, or None."""
    els, X = read_xyz(xyz_path)
    drop = {i for i, e in enumerate(els) if e == metal}
    if not drop:
        return None, "metal not found in xyz"
    best = None
    for comp in components(els, X, drop):
        c = Counter(els[i] for i in comp)
        if c == formula:
            best = comp
            break
    if best is None:
        return None, "no connected component matches the ligand formula"
    return ([els[i] for i in best], X[best]), "ok"


# ======================================================================================
def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    log("xtb availability:", shutil.which("xtb") or "NOT ON PATH (cluster route)")
    log("loading the accepted complexes ...")
    rows, count_cols = L.load_energy_rows()
    smiles = sorted(rows["canonical_smiles"].unique())
    log(f"  {len(rows)} complexes, {len(smiles)} distinct ligands, "
        f"{rows['metal_symbol'].nunique()} metals")

    idx_of = {s: f"lig_{i:03d}" for i, s in enumerate(smiles)}
    formulas = {s: L.ligand_formula(s) for s in smiles}

    # ---- 1. free ligands ------------------------------------------------------------------
    man = []
    lig_notes = []
    for s in smiles:
        tag = idx_of[s]
        geo, how, q = embed_ligand(s)
        if geo is None:
            lig_notes.append({"species": tag, "canonical_smiles": s, "status": "FAILED", "how": how})
            continue
        els, X = geo
        if Counter(els) != formulas[s]:
            lig_notes.append({"species": tag, "canonical_smiles": s, "status": "FAILED",
                              "how": "embedded formula does not match the SMILES formula"})
            continue
        p = OUT / "ligands" / f"{tag}.xyz"
        write_xyz(p, els, X, f"{tag} free ligand  charge=0 uhf=0  {how}  smiles={s}")
        lig_notes.append({"species": tag, "canonical_smiles": s, "status": "ok", "how": how})
        man.append({"species": tag, "kind": "free_ligand", "canonical_smiles": s,
                    "file": f"ligands/{tag}.xyz", "charge": 0, "uhf": 0, "mode": "opt",
                    "n_atoms": len(els), "est_minutes": opt_minutes(len(els))})
    pd.DataFrame(lig_notes).to_csv(OUT / "ligand_embedding.csv", index=False)
    nfail = sum(1 for r in lig_notes if r["status"] != "ok")
    log(f"  free ligands written: {len(man)}  failed: {nfail}")

    # ---- 2. strain geometries (one accepted complex per ligand) ----------------------------
    strain_notes = []
    n_strain = 0
    for s in smiles:
        tag = idx_of[s]
        sub = rows[rows["canonical_smiles"] == s].sort_values(["n_ligs", "n_fill", "metal_symbol"])
        got = None
        n_tried = 0
        for _, r in sub.iterrows():
            n_tried += 1
            geo, why = extract_ligand(L.DATA / r["xyz_path"], r["metal_symbol"], formulas[s])
            if geo is not None:
                got = (geo, r)
                break
        if got is None:
            n_unrelaxed = int(sub["source_xyz_path"].astype(str).str.contains("unrelaxed").sum())
            strain_notes.append({"species": tag, "canonical_smiles": s, "status": "FAILED",
                                 "note": why + " -- the copies interpenetrate at bond length, so no "
                                               "connectivity cut separates them",
                                 "n_complexes_tried": n_tried, "n_unrelaxed_sources": n_unrelaxed,
                                 "n_ligs": int(sub["n_ligs"].iloc[0]), "from_geometry_key": ""})
            continue
        (els, X), r = got
        p = OUT / "strain" / f"strain_{tag}.xyz"
        write_xyz(p, els, X, f"strain_{tag} ligand at complex geometry  charge=0 uhf=0  "
                             f"from={r['geometry_key']}")
        strain_notes.append({"species": f"strain_{tag}", "canonical_smiles": s, "status": "ok",
                             "note": "connectivity cut, metal and fill removed",
                             "n_complexes_tried": n_tried,
                             "n_unrelaxed_sources": int(sub["source_xyz_path"].astype(str)
                                                        .str.contains("unrelaxed").sum()),
                             "n_ligs": int(r["n_ligs"]), "from_geometry_key": r["geometry_key"]})
        man.append({"species": f"strain_{tag}", "kind": "ligand_at_complex_geometry",
                    "canonical_smiles": s, "file": f"strain/strain_{tag}.xyz", "charge": 0,
                    "uhf": 0, "mode": "sp", "n_atoms": len(els),
                    "est_minutes": SP_FRACTION * opt_minutes(len(els))})
        n_strain += 1
    pd.DataFrame(strain_notes).to_csv(OUT / "strain_extraction.csv", index=False)
    log(f"  strain geometries extracted: {n_strain} of {len(smiles)}")

    # ---- 3. fill species -------------------------------------------------------------------
    # planar D3h nitrate, N-O 1.26 A; C2v water, O-H 0.958 A, angle 104.5 deg.  Both are
    # optimised by the job, so only the topology has to be right.
    d = 1.26
    nit_els = ["N", "O", "O", "O"]
    nit_xyz = np.array([[0.0, 0.0, 0.0],
                        [d, 0.0, 0.0],
                        [-d / 2, d * np.sqrt(3) / 2, 0.0],
                        [-d / 2, -d * np.sqrt(3) / 2, 0.0]])
    write_xyz(OUT / "species" / "nitrate.xyz", nit_els, nit_xyz,
              "nitrate NO3-  charge=-1 uhf=0  idealised D3h, r(N-O)=1.26 A")
    man.append({"species": "nitrate", "kind": "fill_species", "canonical_smiles": "",
                "file": "species/nitrate.xyz", "charge": -1, "uhf": 0, "mode": "opt",
                "n_atoms": 4, "est_minutes": opt_minutes(4)})
    ang = np.radians(104.5 / 2)
    rOH = 0.958
    wat_els = ["O", "H", "H"]
    wat_xyz = np.array([[0.0, 0.0, 0.0],
                        [rOH * np.sin(ang), rOH * np.cos(ang), 0.0],
                        [-rOH * np.sin(ang), rOH * np.cos(ang), 0.0]])
    write_xyz(OUT / "species" / "water.xyz", wat_els, wat_xyz,
              "water H2O  charge=0 uhf=0  idealised C2v, r(O-H)=0.958 A, angle 104.5 deg")
    man.append({"species": "water", "kind": "fill_species", "canonical_smiles": "",
                "file": "species/water.xyz", "charge": 0, "uhf": 0, "mode": "opt",
                "n_atoms": 3, "est_minutes": opt_minutes(3)})

    # ---- 4. the 14 Ln3+ ions ----------------------------------------------------------------
    metals = sorted(rows["metal_symbol"].unique())
    for m in metals:
        write_xyz(OUT / "ions" / f"{m}.xyz", [m], np.zeros((1, 3)),
                  f"{m}3+ free ion  charge=3 uhf=0  (GFN2 treats Ln with f-in-core: closed shell)")
        man.append({"species": f"{m}3+", "kind": "metal_ion", "canonical_smiles": "",
                    "file": f"ions/{m}.xyz", "charge": 3, "uhf": 0, "mode": "sp",
                    "n_atoms": 1, "est_minutes": 0.1})
    log(f"  fill species: 2   metal ions: {len(metals)}")

    manifest = pd.DataFrame(man)
    manifest["sha256_input"] = [hashlib.sha256((OUT / f).read_bytes()).hexdigest()[:16]
                                for f in manifest["file"]]
    manifest.to_csv(OUT / "manifest.csv", index=False)

    # ---- 5. completeness: every distinct composition combination has its references ---------
    combos = (rows.groupby(["canonical_smiles", "fill_ligand", "n_fill", "n_ligs", "metal_symbol"])
              .agg(n_complexes=("geometry_key", "size"), n_NO3=("n_NO3", "first"),
                   n_H2O=("n_H2O", "first"), total_charge=("total_charge", "first"),
                   n_energies=("complex_total_energy_eV", "count"))
              .reset_index())
    have = set(manifest["species"])
    need_lig = combos["canonical_smiles"].map(idx_of)
    combos["ref_ligand"] = need_lig
    combos["ref_metal"] = combos["metal_symbol"] + "3+"
    combos["needs_nitrate"] = combos["n_NO3"] > 0
    combos["needs_water"] = combos["n_H2O"] > 0
    combos["complete"] = [
        (lig in have) and (met in have) and ((not nn) or "nitrate" in have) and ((not nw) or "water" in have)
        for lig, met, nn, nw in zip(combos["ref_ligand"], combos["ref_metal"],
                                    combos["needs_nitrate"], combos["needs_water"])]
    combos.to_csv(OUT / "completeness.csv", index=False)
    complete = bool(combos["complete"].all())
    log(f"  distinct (ligand, fill, n_fill, n_ligs, metal) combinations: {len(combos)}   "
        f"all references present: {complete}")

    # ---- 6. the SLURM array job -------------------------------------------------------------
    jobs = manifest[["species", "file", "charge", "uhf", "mode", "n_atoms", "est_minutes"]].copy()
    jobs = jobs.sort_values("est_minutes", ascending=False).reset_index(drop=True)
    jobs.to_csv(OUT / "jobs.csv", index=False)
    total_cpu_h = float(jobs["est_minutes"].sum() / 60.0)
    concurrency = 20
    wall_h = float(max(jobs["est_minutes"].max(), jobs["est_minutes"].sum() / concurrency) / 60.0)

    lines = jobs.apply(lambda r: f"{r['species']}\t{r['file']}\t{int(r['charge'])}\t{int(r['uhf'])}"
                                 f"\t{r['mode']}", axis=1).tolist()
    (OUT / "jobs.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    sub = f"""#!/bin/bash
#SBATCH --job-name=xtb_l1_refs
#SBATCH --array=1-{len(jobs)}%{concurrency}
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

LINE=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" jobs.tsv)
SPECIES=$(echo "$LINE" | cut -f1)
FILE=$(echo    "$LINE" | cut -f2)
CHRG=$(echo    "$LINE" | cut -f3)
UHF=$(echo     "$LINE" | cut -f4)
MODE=$(echo    "$LINE" | cut -f5)

WORK="out/${{SPECIES}}"
mkdir -p "$WORK"
cp "$FILE" "$WORK/input.xyz"
cd "$WORK"

if [ "$MODE" = "opt" ]; then
    xtb input.xyz --opt --gfn 2 --chrg "$CHRG" --uhf "$UHF" > xtb.out 2> xtb.err
else
    xtb input.xyz --sp  --gfn 2 --chrg "$CHRG" --uhf "$UHF" > xtb.out 2> xtb.err
fi
grep -E "TOTAL ENERGY" xtb.out | tail -1
"""
    (OUT / "submit_xtb_references.sh").write_text(sub, encoding="utf-8", newline="\n")

    collect = """#!/bin/bash
# Collect one CSV row per reference species from the xtb output directories.
# Usage (from results/L1/reference_species/):   bash collect_reference_energies.sh > reference_energies.csv
set -euo pipefail
echo "species,energy_Eh,energy_eV,converged,xtb_version"
while IFS=$'\\t' read -r SPECIES FILE CHRG UHF MODE; do
    OUT="out/${SPECIES}/xtb.out"
    if [ ! -f "$OUT" ]; then
        echo "${SPECIES},,,MISSING,"
        continue
    fi
    EH=$(grep -E "TOTAL ENERGY" "$OUT" | tail -1 | awk '{print $4}')
    VER=$(grep -m1 -oE "version [0-9.]+" "$OUT" | head -1 | awk '{print $2}')
    CONV=$(grep -qE "GEOMETRY OPTIMIZATION CONVERGED|normal termination" "$OUT" && echo ok || echo CHECK)
    EV=$(awk -v e="$EH" 'BEGIN{ if (e=="") print ""; else printf "%.9f", e*27.211386245988 }')
    echo "${SPECIES},${EH},${EV},${CONV},${VER}"
done < jobs.tsv
"""
    (OUT / "collect_reference_energies.sh").write_text(collect, encoding="utf-8", newline="\n")

    tmpl = manifest[["species", "kind", "canonical_smiles", "charge", "uhf", "mode"]].copy()
    tmpl["energy_Eh"] = ""
    tmpl["energy_eV"] = ""
    tmpl["converged"] = ""
    tmpl["xtb_version"] = ""
    tmpl.to_csv(OUT / "reference_energies_TEMPLATE.csv", index=False)

    summary = {
        "xtb_on_path": bool(shutil.which("xtb")),
        "n_species": int(len(manifest)),
        "n_free_ligands": int((manifest["kind"] == "free_ligand").sum()),
        "n_strain_geometries": int((manifest["kind"] == "ligand_at_complex_geometry").sum()),
        "n_fill_species": int((manifest["kind"] == "fill_species").sum()),
        "n_metal_ions": int((manifest["kind"] == "metal_ion").sum()),
        "ligand_embedding_failures": int(nfail),
        "strain_extraction_failures": int(len(smiles) - n_strain),
        "largest_ligand_atoms": int(manifest.loc[manifest["kind"] == "free_ligand", "n_atoms"].max()),
        "median_ligand_atoms": float(manifest.loc[manifest["kind"] == "free_ligand", "n_atoms"].median()),
        "distinct_composition_combinations": int(len(combos)),
        "completeness_ok": complete,
        "expected_cpu_hours": round(total_cpu_h, 2),
        "expected_wall_hours_at_20_concurrent": round(wall_h, 2),
        "mem_per_task_GB": 2,
        "solvent_model": "none (gas-phase GFN2); no solvation key exists anywhere in the dataset",
        "charge_convention": "metal 3+, ligand neutral as written, nitrate -1, water 0; "
                             "complex charge = 3 - n_NO3 (verified on all 1155 files)",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    log("done")


if __name__ == "__main__":
    main()
