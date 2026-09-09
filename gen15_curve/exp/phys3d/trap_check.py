"""The composition trap, measured: what the naive energy slope claims and what survives control.

The prior finding this experiment was warned about: a double-centred ``complex_total_energy_eV``
slope in the ionic radius correlates with the observed amplitude at Spearman +0.64 over the
ligands with a complete 14-metal series.  The complexes are not, however, comparable across the
series -- fill waters and nitrates enter and leave the inner sphere as the cation contracts -- so
the "slope" is partly a count of atoms.

This script prints, on one page:
  * how badly the composition moves across a series;
  * the naive slope's Spearman with a and |a| on the same restricted sets that produced +0.64;
  * the same for construction A (inside a constant-composition block) and construction B (whole
    series with element counts as covariates).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "gen15_curve"))
sys.path.insert(0, str(HERE))
from build_block import load_geometries, main_blocks, _two_way_residual, _slope_fit  # noqa: E402
from descriptor_stats import extractant_table, _rho  # noqa: E402


def main() -> None:
    g, count_cols = load_geometries()
    g = main_blocks(g)
    sc = pd.read_parquet(ROOT / "dataset with 3D structures/features/complex_physical_scalars.parquet")
    g = g.merge(sc[["geometry_key", "complex_total_energy_eV"]], on="geometry_key", how="left")

    # ---- how far the inner sphere moves ---------------------------------------------------
    sizes = g.groupby("series").agg(n=("metal_symbol", "nunique"),
                                    ncomp=("composition", "nunique"),
                                    amin=("n_atoms", "min"), amax=("n_atoms", "max"))
    sizes["datoms"] = sizes["amax"] - sizes["amin"]
    multi = sizes[sizes["n"] >= 2]
    print(f"series with >= 2 metals: {len(multi)}")
    print(f"  composition changes across the series: {(multi.ncomp > 1).sum()}")
    print(f"  largest atom-count change: {int(multi.datoms.max())}")
    print(f"  median atom-count change (of those that change): "
          f"{int(multi.loc[multi.ncomp > 1, 'datoms'].median())}")
    full = sizes[sizes["n"] == 14]
    print(f"series with all 14 metals: {len(full)}; of those, constant composition: "
          f"{(full.ncomp == 1).sum()}")

    # ---- three slopes ---------------------------------------------------------------------
    E = g["complex_total_energy_eV"].to_numpy(dtype=float)
    ok = np.isfinite(E)
    # naive: series + metal fixed effects, no composition control, whole series
    g.loc[ok, "resid_naive"] = _two_way_residual(E[ok], g["series"].to_numpy()[ok],
                                                 g["metal_symbol"].to_numpy()[ok])
    # A: constant-composition block only
    mA = g["is_main_block"].to_numpy() & ok
    g.loc[mA, "resid_A"] = _two_way_residual(E[mA], g["block"].to_numpy()[mA],
                                             g["metal_symbol"].to_numpy()[mA])
    # B: whole series with element counts as within-series covariates
    C = g.loc[ok, count_cols].to_numpy(dtype=float)
    C = C - pd.DataFrame(C).groupby(g["series"].to_numpy()[ok]).transform("mean").to_numpy()
    keep = C.std(axis=0) > 1e-9
    g.loc[ok, "resid_B"] = _two_way_residual(E[ok], g["series"].to_numpy()[ok],
                                             g["metal_symbol"].to_numpy()[ok], cov=C[:, keep])

    rows = []
    for smi, sub in g.groupby("canonical_smiles"):
        cand = sub.groupby("series").metal_symbol.nunique().sort_values(ascending=False)
        ser = cand.index[0]
        s = sub[sub["series"] == ser]
        blk = s[s["is_main_block"]]
        rec = {"extractant": smi, "n_series": s["metal_symbol"].nunique(),
               "n_block": blk["metal_symbol"].nunique(),
               "const_comp": int(s["composition"].nunique() == 1)}
        rec["naive"] = _slope_fit(s["r"].to_numpy(float), s["resid_naive"].to_numpy(float), True)[0]
        rec["A"] = _slope_fit(blk["r"].to_numpy(float), blk["resid_A"].to_numpy(float), True)[0]
        rec["B"] = _slope_fit(s["r"].to_numpy(float), s["resid_B"].to_numpy(float), True)[0]
        rows.append(rec)
    sl = pd.DataFrame(rows)

    tgt = extractant_table().merge(sl, on="extractant", how="inner")
    print(f"\ncohort extractants with a well-determined curve and a slope: {len(tgt)}")
    subsets = {
        "all cohort extractants": np.ones(len(tgt), dtype=bool),
        "complete 14-metal series": tgt["n_series"].to_numpy() == 14,
        "constant-composition series only": tgt["const_comp"].to_numpy() == 1,
        "block >= 7 metals": tgt["n_block"].to_numpy() >= 7,
    }
    from descriptor_stats import _partial_rho
    out = []
    for label, m in subsets.items():
        t = tgt[m]
        for cons in ("naive", "A", "B"):
            for target in ("a", "abs_a"):
                rho, p, n = _rho(t[cons].to_numpy(float), t[target].to_numpy(float))
                pr = _partial_rho(t[cons].to_numpy(float), t[target].to_numpy(float),
                                  t["n_metals"].to_numpy(float))
                out.append({"subset": label, "construction": cons, "target": target,
                            "n": n, "rho": rho, "p": p, "rho_given_n_metals": pr})
    for cons in ("naive", "A", "B"):
        r = _rho(tgt[cons].to_numpy(float), tgt["n_metals"].to_numpy(float))
        print(f"  rho({cons} slope, n_metals measured) = {r[0]:.3f}  (n={r[2]})")
    res = pd.DataFrame(out)
    print("\nSpearman of the energy slope with the curve, by construction and subset:")
    print(res.pivot_table(index=["subset", "construction"], columns="target",
                          values=["n", "rho", "p", "rho_given_n_metals"]).round(3).to_string())
    res.to_csv(HERE / "trap_check.csv", index=False)


if __name__ == "__main__":
    main()
