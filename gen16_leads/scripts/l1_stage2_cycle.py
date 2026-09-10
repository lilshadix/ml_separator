"""L1 Stage 2, part 2 -- the cycle-corrected descriptor, ready to run the moment the cluster
returns the reference energies.

    dE(complex) = E(complex) - E(Ln3+) - n_ligs * E(free ligand) - n_NO3 * E(NO3-) - n_H2O * E(H2O)

then *exactly* the Stage 1 pipeline on dE instead of ``complex_total_energy_eV``: the same four
models (NAIVE / ELEM / SPECIES / SPECIES_CONST), the same per-extractant within-series linear +
quadratic fit in the standardised Shannon radius, the same sets S8 / S14 / S3, the same statistics
(Spearman with a, |a|, b; LOCO over chemotypes; partial Spearman given n_metals; chemotype-blocked
bootstrap 2000; family-wise permutation null 2000) and the same registered decision rule.

Nothing about the protocol changes between the stages.  What changes is only the energy column.

INPUT SCHEMA -- ``reference_energies.csv``, one row per species:

    species      the id in ``reference_species/manifest.csv``:
                 ``lig_000`` ... ``lig_176``   free ligands, charge 0
                 ``nitrate``                   NO3-, charge -1
                 ``water``                     H2O, charge 0
                 ``Ce3+`` ... ``Yb3+``         the 14 Ln3+ ions, charge +3
                 ``strain_lig_XXX``            optional, ligand at complex geometry (not used here)
    energy_Eh    total energy in Hartree   (either this or energy_eV must be present and finite)
    energy_eV    total energy in eV        (used directly if given; else energy_Eh * 27.211386...)
    converged    optional; any value other than ``ok`` / ``true`` / ``1`` aborts the run
    xtb_version  optional, recorded in the output for provenance

A missing, non-finite or non-converged reference **aborts the run**; it is never filled in, because
a missing reference silently reintroduces the composition step this lead exists to remove.

Run from the repo root once ``reference_energies.csv`` exists:
    PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 .venv/Scripts/python.exe gen16_leads/scripts/l1_stage2_cycle.py \
        --refs gen16_leads/results/L1/reference_species/reference_energies.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen16 import bootstrap  # noqa: E402
from gen16 import l1_cycle as L  # noqa: E402

REFDIR = L.RESULTS / "reference_species"
OUT = L.RESULTS
T0 = time.time()
TRUE = {"ok", "true", "yes", "1", "converged", "normal termination"}


def log(*a) -> None:
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


class MissingReference(RuntimeError):
    pass


# --------------------------------------------------------------------------------------
def load_references(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise MissingReference(
            f"{path} does not exist.  Run scripts/l1_build_references.py, submit\n"
            f"  {REFDIR / 'submit_xtb_references.sh'}\n"
            f"and collect the results with collect_reference_energies.sh.")
    ref = pd.read_csv(path)
    if "species" not in ref.columns:
        raise MissingReference("reference_energies.csv has no 'species' column")
    if "energy_eV" in ref.columns and ref["energy_eV"].notna().any():
        e = pd.to_numeric(ref["energy_eV"], errors="coerce")
    elif "energy_Eh" in ref.columns:
        e = pd.to_numeric(ref["energy_Eh"], errors="coerce") * L.HARTREE_EV
    else:
        raise MissingReference("reference_energies.csv has neither energy_eV nor energy_Eh")
    ref["E_eV"] = e
    if "converged" in ref.columns:
        bad = ref[~ref["converged"].astype(str).str.strip().str.lower().isin(TRUE)]
        bad = bad[bad["species"].astype(str).str.startswith(("lig_", "nitrate", "water")) |
                  bad["species"].astype(str).str.endswith("3+")]
        if len(bad):
            raise MissingReference(
                f"{len(bad)} reference species did not converge: {bad['species'].tolist()[:10]}")
    return ref


def cycle_energies(rows: pd.DataFrame, ref: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Attach ``dE_eV`` to every complex; abort on any missing reference."""
    E = dict(zip(ref["species"].astype(str), ref["E_eV"].astype(float)))
    lig_id = dict(zip(manifest.loc[manifest["kind"] == "free_ligand", "canonical_smiles"],
                      manifest.loc[manifest["kind"] == "free_ligand", "species"]))
    needed = set()
    for _, r in rows.iterrows():
        needed.add(lig_id.get(r["canonical_smiles"], f"MISSING_LIGAND::{r['canonical_smiles']}"))
        needed.add(f"{r['metal_symbol']}3+")
        if r["n_NO3"] > 0:
            needed.add("nitrate")
        if r["n_H2O"] > 0:
            needed.add("water")
    missing = sorted(s for s in needed if not np.isfinite(E.get(s, np.nan)))
    if missing:
        raise MissingReference(
            f"{len(missing)} reference energies are missing or non-finite: {missing[:12]}\n"
            "A missing reference reintroduces the composition step exactly where the cycle is "
            "supposed to cancel it, so the run aborts rather than continuing.")
    rows = rows.copy()
    rows["E_lig_eV"] = rows["canonical_smiles"].map(lambda s: E[lig_id[s]])
    rows["E_ion_eV"] = rows["metal_symbol"].map(lambda m: E[f"{m}3+"])
    rows["E_NO3_eV"] = E.get("nitrate", np.nan)
    rows["E_H2O_eV"] = E.get("water", np.nan)
    rows["dE_eV"] = (rows["complex_total_energy_eV"]
                     - rows["E_ion_eV"]
                     - rows["n_ligs"] * rows["E_lig_eV"]
                     - rows["n_NO3"] * np.where(rows["n_NO3"] > 0, rows["E_NO3_eV"], 0.0)
                     - rows["n_H2O"] * np.where(rows["n_H2O"] > 0, rows["E_H2O_eV"], 0.0))
    return rows


# ======================================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", default=str(REFDIR / "reference_energies.csv"))
    ap.add_argument("--tag", default="stage2")
    args = ap.parse_args()

    log("loading the accepted complexes ...")
    rows_all, count_cols = L.load_energy_rows()
    manifest = pd.read_csv(REFDIR / "manifest.csv")
    ref = load_references(Path(args.refs))
    log(f"  reference species read: {len(ref)}   finite energies: {int(np.isfinite(ref['E_eV']).sum())}")

    rows_all = cycle_energies(rows_all, ref, manifest)
    fin = np.isfinite(rows_all["dE_eV"].to_numpy(dtype=float))
    log(f"  dE computed for {int(fin.sum())} of {len(rows_all)} complexes "
        f"(the rest have no complex_total_energy_eV)")
    d = rows_all.loc[fin, "dE_eV"]
    log(f"  dE range {d.min():.1f} .. {d.max():.1f} eV, median {d.median():.1f} eV")
    if d.median() > 0:
        log("  WARNING: the median binding energy is positive.  Check the charge convention "
            "(complex charge must be 3 - n_NO3) and that the references used the same solvent model.")

    models_all = L.MODELS + ("SPECIES_NFILLCOL",)
    rows, coefs = L.fit_models(rows_all, count_cols, energy="dE_eV", models=models_all)
    ctab = pd.concat([c.assign(model=m) for m, c in coefs.items()], ignore_index=True)
    ctab.to_csv(OUT / f"{args.tag}_coefficients.csv", index=False)
    gam = ctab[ctab["term"].str.startswith("gamma::")][["model", "term", "coef", "se"]]
    print("\nfill-species coefficients on dE (eV per molecule) -- after an exact cycle these should "
          "be small (a mean binding energy), not a free-species total energy:")
    print(gam.to_string(index=False))

    sl = L.attach_sets(L.slope_table(rows, models_all, g_all=rows_all))
    sl.to_csv(OUT / f"{args.tag}_slopes.csv", index=False)

    st, loco, null, W = L.full_stats(sl, models=models_all, value="slope", family_models=L.MODELS)
    st.to_csv(OUT / f"{args.tag}_stats.csv", index=False)
    loco.to_csv(OUT / f"{args.tag}_loco.csv", index=False)
    null.to_csv(OUT / f"{args.tag}_perm_null.csv", index=False)
    stq, _, nullq, _ = L.full_stats(sl, models=models_all, value="quad", family_models=L.MODELS)
    stq.to_csv(OUT / f"{args.tag}_stats_quad.csv", index=False)

    con = L.contrasts_frame(st[st.model.isin(L.MODELS)], stage=args.tag,
                            extra_exploratory=pd.concat([st[~st.model.isin(L.MODELS)], stq],
                                                        ignore_index=True))
    con.to_csv(OUT / f"contrasts_{args.tag}.csv", index=False)

    print("\n=== Spearman of the cycle-corrected slope with a ===")
    print(st[st.target == "a"][["model", "set", "n", "rho", "p", "loco_min", "loco_max",
                                "loco_sign_stable", "partial_rho_n_metals", "ci95_low", "ci95_high",
                                "ci_excludes_zero"]].round(4).to_string(index=False))
    print("\n=== family-wise permutation null ===")
    print(null.round(4).to_string(index=False))

    dec = L.decision(st, "SPECIES", "S8")
    dec_sec = L.decision(st, "SPECIES_CONST", "S8")
    dec_naive = L.decision(st, "NAIVE", "S8")
    payload = {"stage": args.tag, "refs": args.refs, "registered": dec,
               "secondary_SPECIES_CONST": dec_sec,
               "cycle_only_NAIVE_on_dE": dec_naive,
               "n_complexes_with_dE": int(fin.sum()),
               "xtb_versions": sorted(set(ref.get("xtb_version", pd.Series(dtype=str))
                                          .dropna().astype(str).tolist())),
               "gamma_on_dE": {r["term"]: {"model": r["model"], "coef_eV": r["coef"]}
                               for _, r in gam.iterrows() if r["model"] == "SPECIES"}}
    (OUT / f"{args.tag}_decision.json").write_text(json.dumps(payload, indent=2, default=float),
                                                   encoding="utf-8")
    print(f"\n=== REGISTERED DECISION (SPECIES, S8, target a) ===\n  VERDICT: {dec['verdict'].upper()}"
          f"   rho={dec['rho_a']:.4f}  partial={dec['partial_rho_a_given_n_metals']:.4f}  "
          f"n={dec['n']}  CI [{dec['ci95_low']:.4f}, {dec['ci95_high']:.4f}]")
    print(f"  cycle-only NAIVE-on-dE, S8: rho={dec_naive['rho_a']:.4f}  n={dec_naive['n']}")
    log("done")


if __name__ == "__main__":
    try:
        main()
    except MissingReference as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        raise SystemExit(2)
