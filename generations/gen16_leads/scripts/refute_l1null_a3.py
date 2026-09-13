"""REFUTER lens A, part 3: signal-injection recovery -- the direct test of whether the SPECIES
correction can destroy a real, composition-independent radius trend; plus the leftover mandatory
checks (n_metals strata done properly, within-/between-chemotype split, lead-CSV recomputation)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from refute_l1null_a import (ROOT, OUT, PHYS3D, load, choose_series, varying, slope_fit,  # noqa: E402
                            rho, two_way, species_cov, centred_counts, ELEM_COLS, ELEMENTS_OK)


def build_residuals(rows, y, forms, elem_hat):
    """NAIVE / ELEM / SPECIES / SPECIES_CONST / CYCLEHAT residuals for an arbitrary energy vector."""
    ser = rows["series"].to_numpy()
    met = rows["metal_symbol"].to_numpy()
    out = {}
    out["NAIVE"], *_ = two_way(y, ser, met)
    out["ELEM"], *_ = two_way(y, ser, met, cov=centred_counts(rows))
    C, _ = species_cov(rows)
    out["SPECIES"], *_ = two_way(y, ser, met, cov=C)
    const = ~rows["series"].isin(varying(rows)).to_numpy()
    sub = rows[const].reset_index(drop=True)
    Cs, _ = species_cov(sub, deltas=False)
    rs, *_ = two_way(y[const], sub["series"].to_numpy(), sub["metal_symbol"].to_numpy(), cov=Cs)
    v = np.full(len(rows), np.nan)
    v[const] = rs
    out["SPECIES_CONST"] = v
    # fixed-coefficient cycle: composition energies re-estimated on THIS y, then subtracted
    Xe = np.c_[np.ones(len(rows)), rows[ELEM_COLS].to_numpy(float),
               pd.get_dummies(pd.Series(met), drop_first=True).to_numpy(float)]
    be, *_ = np.linalg.lstsq(Xe, y, rcond=None)
    cel = dict(zip(sorted(ELEMENTS_OK), be[1:1 + len(ELEM_COLS)]))
    EL = {s: sum(cel[e] * n for e, n in f.items()) for s, f in forms.items()}
    dE = (y - rows["n_ligs"].to_numpy(float) * rows["canonical_smiles"].map(EL).to_numpy(float)
          - rows["n_NO3"].to_numpy(float) * (cel["N"] + 3 * cel["O"])
          - rows["n_H2O"].to_numpy(float) * (2 * cel["H"] + cel["O"]))
    out["CYCLEHAT"], *_ = two_way(dE, ser, met)
    # TRUE-CYCLE control: subtract the composition using the coefficients estimated on the
    # UNINJECTED energies (i.e. an external, signal-blind reference set -- what Stage 2 will have)
    out["_dE_needs_external"] = dE
    return out


def slopes_for(rows, resid, so):
    rec = {}
    for smi, ser in so.items():
        s = rows[rows["series"] == ser].sort_values("r")
        idx = s.index.to_numpy()
        sl, qu, n = slope_fit(s["r"].to_numpy(float), resid[idx])
        rec[smi] = (sl, n)
    return rec


def main():
    g = load()
    forms = g.attrs["forms"]
    rows = g[np.isfinite(g["complex_total_energy_eV"].to_numpy(float))].reset_index(drop=True).copy()
    y0 = rows["complex_total_energy_eV"].to_numpy(float)
    so = choose_series(g)
    t = pd.read_parquet(PHYS3D / "extractant_targets.parquet")[
        ["extractant", "a", "b", "abs_a", "n_metals", "chemotype"]].set_index("extractant")

    # reference (uninjected) element coefficients -- the "external reference set" Stage 2 supplies
    Xe = np.c_[np.ones(len(rows)), rows[ELEM_COLS].to_numpy(float),
               pd.get_dummies(rows["metal_symbol"], drop_first=True).to_numpy(float)]
    be0, *_ = np.linalg.lstsq(Xe, y0, rcond=None)
    cel0 = dict(zip(sorted(ELEMENTS_OK), be0[1:1 + len(ELEM_COLS)]))
    EL0 = {s: sum(cel0[e] * n for e, n in f.items()) for s, f in forms.items()}
    comp0 = (rows["n_ligs"].to_numpy(float) * rows["canonical_smiles"].map(EL0).to_numpy(float)
             + rows["n_NO3"].to_numpy(float) * (cel0["N"] + 3 * cel0["O"])
             + rows["n_H2O"].to_numpy(float) * (2 * cel0["H"] + cel0["O"]))

    # baseline S8 membership from the NAIVE residual on the real energies
    base = build_residuals(rows, y0, forms, None)
    sl0 = slopes_for(rows, base["NAIVE"], so)
    S8 = sorted([s for s, (v, n) in sl0.items() if n >= 8 and s in t.index])
    a_of = t.loc[S8, "a"]
    resid_rms = float(np.nanstd(base["SPECIES"]))
    print(f"S8 = {len(S8)}  sd(SPECIES residual) = {resid_rms:.4f} eV")

    # ---------------- signal injection ----------------
    # inject  beta_s * r  into every row of the chosen series of extractant s, with
    # beta_s proportional to the observed amplitude a_s: by construction the TRUE
    # rho(slope, a) on S8 is +1.  Recovery is the question.
    ser_arr = rows["series"].to_numpy()
    r_arr = rows["r"].to_numpy(float)
    ser_of = {so[s]: s for s in S8}
    a_vec = np.zeros(len(rows))
    for i, sname in enumerate(ser_arr):
        e = ser_of.get(sname)
        if e is not None:
            a_vec[i] = float(t.loc[e, "a"])
    a_vec = a_vec / np.nanstd(t.loc[S8, "a"].to_numpy(float))

    recs = []
    for mult in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        y = y0 + mult * resid_rms * a_vec * r_arr
        res = build_residuals(rows, y, forms, None)
        # the "external reference" cycle: composition subtracted with the SIGNAL-BLIND coefficients
        res["CYCLE_EXT"], *_ = two_way(y - comp0, ser_arr, rows["metal_symbol"].to_numpy())
        for m in ("NAIVE", "ELEM", "SPECIES", "SPECIES_CONST", "CYCLEHAT", "CYCLE_EXT"):
            sl = slopes_for(rows, res[m], so)
            x = np.array([sl[s][0] for s in S8], dtype=float)
            r, p, n = rho(x, a_of.to_numpy(float))
            recs.append({"inject_mult": mult, "model": m, "set": "S8", "n": n, "rho": r, "p": p})
        # varying-only / constant-only splits at the same injection
        const_ser = set(rows["series"]) - set(varying(rows))
        idx19 = [s for s in S8 if so[s] in const_ser]
        idx43 = [s for s in S8 if so[s] not in const_ser]
        for m in ("NAIVE", "ELEM", "SPECIES", "CYCLEHAT", "CYCLE_EXT"):
            sl = slopes_for(rows, res[m], so)
            for lab, ii in (("const19", idx19), ("varying43", idx43)):
                x = np.array([sl[s][0] for s in ii], dtype=float)
                r, p, n = rho(x, t.loc[ii, "a"].to_numpy(float))
                recs.append({"inject_mult": mult, "model": m, "set": lab, "n": n, "rho": r, "p": p})
    R = pd.DataFrame(recs)
    R.to_csv(OUT / "A_injection.csv", index=False)
    pd.set_option("display.width", 200)
    piv = R[R.set == "S8"].pivot(index="inject_mult", columns="model", values="rho")
    print("\n=== injection recovery, rho(slope, a) on S8 (true rho = +1 by construction) ===")
    print(piv.to_string())
    piv2 = R[R.set == "varying43"].pivot(index="inject_mult", columns="model", values="rho")
    print("\n=== same, the 43 varying-composition series ===")
    print(piv2.to_string())
    piv3 = R[R.set == "const19"].pivot(index="inject_mult", columns="model", values="rho")
    print("\n=== same, the 19 constant-n_ligs series ===")
    print(piv3.to_string())

    # ---------------- leftover mandatory checks ----------------
    out = {}
    W = pd.DataFrame({"slope_NAIVE": [sl0[s][0] for s in S8]}, index=S8)
    for m in ("ELEM", "SPECIES", "CYCLEHAT"):
        s_ = slopes_for(rows, base[m], so)
        W[f"slope_{m}"] = [s_[s][0] for s in S8]
    W = W.join(t)
    # n_metals strata, done properly (median is 14, so split on == 14 vs < 14)
    for lab, sub in (("n_metals<14", W[W.n_metals < 14]), ("n_metals==14", W[W.n_metals >= 14])):
        for m in ("NAIVE", "ELEM", "SPECIES", "CYCLEHAT"):
            r, p, n = rho(sub[f"slope_{m}"].to_numpy(float), sub["a"].to_numpy(float))
            out[f"strat::{lab}::{m}"] = [r, p, n]
    # within- vs between-chemotype
    for m in ("NAIVE", "ELEM", "SPECIES", "CYCLEHAT"):
        x, y_ = W[f"slope_{m}"].to_numpy(float), W["a"].to_numpy(float)
        ch = W["chemotype"].to_numpy()
        cm = pd.DataFrame({"x": x, "y": y_, "c": ch}).groupby("c").mean()
        out[f"between_chemotype::{m}"] = list(rho(cm["x"].to_numpy(), cm["y"].to_numpy()))
        xw = x - pd.Series(x).groupby(ch).transform("mean").to_numpy()
        yw = y_ - pd.Series(y_).groupby(ch).transform("mean").to_numpy()
        out[f"within_chemotype::{m}"] = list(rho(xw, yw))
        sub = W[W.chemotype == "sc009"]
        out[f"within_sc009_only::{m}"] = list(rho(sub[f"slope_{m}"].to_numpy(float),
                                                  sub["a"].to_numpy(float)))
    json.dump(out, open(OUT / "A_extra_checks.json", "w"), indent=2, default=str)
    print("\n" + json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
