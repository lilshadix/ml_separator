"""REFUTER lens A for claim L1NULL.  Independent re-derivation of L1 Stage 1 from the
PRE_REGISTRATION description (not from the lead's code), plus the mandatory checks and the
lens-A leakage / bookkeeping audit.

Everything is written under gen16_leads/results/refutation/L1NULL/A/.
Nothing under gen13_separation/, gen14_direction/, gen15_curve/, src/ is modified.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dataset with 3D structures"
PHYS3D = ROOT / "gen15_curve" / "exp" / "phys3d"
OUT = ROOT / "gen16_leads" / "results" / "refutation" / "L1NULL" / "A"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "gen13_separation"))
from gen13sep.metals import LANTHANIDES, SHANNON_RADIUS_CN8  # noqa: E402

_r = np.array([SHANNON_RADIUS_CN8[m] for m in LANTHANIDES], dtype=float)
RSTD = {m: float(v) for m, v in zip(LANTHANIDES, (_r - _r.mean()) / _r.std())}
ELEMENTS_OK = {"H", "B", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "Si", "Se", "As"}


# ------------------------------------------------------------------ data (own construction)
def read_counts(xyz: Path, metal: str) -> Counter:
    """Element counts of the complex, metal excluded.  Reads the extxyz honestly: line 0 is the
    atom count, line 1 is the comment, the next `nat` lines are the atoms."""
    lines = xyz.read_text().splitlines()
    nat = int(lines[0].split()[0])
    c: Counter = Counter()
    for ln in lines[2:2 + nat]:
        el = ln.split()[0]
        if el == metal:
            continue
        c[el] += 1
    return c


def ligand_formula(smi: str) -> Counter:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    return Counter(a.GetSymbol() for a in Chem.AddHs(Chem.MolFromSmiles(smi)).GetAtoms())


def load() -> pd.DataFrame:
    g = pd.read_csv(DATA / "accepted_geometries.csv")
    forms = {s: ligand_formula(s) for s in g["canonical_smiles"].unique()}
    rows = []
    for p, met, smi, nl in zip(g["xyz_path"], g["metal_symbol"], g["canonical_smiles"], g["n_ligs"]):
        c = read_counts(DATA / p, met)
        f = forms[smi]
        res = {e: c.get(e, 0) - int(nl) * f.get(e, 0) for e in set(c) | set(f)}
        res = {e: v for e, v in res.items() if v}
        nN, nH, nO = res.pop("N", 0), res.pop("H", 0), res.pop("O", 0)
        ok = (not res) and nN >= 0 and nH >= 0 and nH % 2 == 0 and nO == 3 * nN + nH // 2
        rows.append({"n_NO3": nN if ok else -1, "n_H2O": nH // 2 if ok else -1, "fill_ok": ok,
                     "n_atoms_nonmetal": sum(c.values()),
                     **{f"c_{e}": c.get(e, 0) for e in ELEMENTS_OK}})
    g = pd.concat([g.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    assert g["fill_ok"].all(), "fill decomposition failed"
    g["r"] = g["metal_symbol"].map(RSTD)
    g["series"] = g["canonical_smiles"] + "||" + g["inner_sphere_anion"]
    sc = pd.read_parquet(DATA / "features" / "complex_physical_scalars.parquet")
    g = g.merge(sc[["geometry_key", "complex_total_energy_eV"]], on="geometry_key", how="left")
    g["lig_atoms"] = g["canonical_smiles"].map(lambda s: sum(forms[s].values()))
    g.attrs["forms"] = forms
    return g


# ------------------------------------------------------------------ fits
def two_way(y, grp, met, cov=None):
    ok = np.isfinite(y)
    if cov is not None:
        ok = ok & np.isfinite(cov).all(axis=1)
    gd = pd.get_dummies(pd.Series(grp[ok]), drop_first=False).to_numpy(float)
    md = pd.get_dummies(pd.Series(met[ok]), drop_first=True).to_numpy(float)
    parts = [gd, md] + ([cov[ok]] if cov is not None else [])
    X = np.hstack(parts)
    yy = y[ok] - np.nanmean(y[ok])
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    r = np.full(len(y), np.nan)
    r[ok] = yy - X @ beta
    return r, beta, X, ok


def slope_fit(rv, yv):
    ok = np.isfinite(rv) & np.isfinite(yv)
    n = int(ok.sum())
    if n < 3:
        return np.nan, np.nan, n
    rr, yy = rv[ok], yv[ok]
    X = np.c_[np.ones(n), rr, rr ** 2] if n >= 4 else np.c_[np.ones(n), rr]
    b, *_ = np.linalg.lstsq(X, yy, rcond=None)
    return float(b[1]), (float(b[2]) if X.shape[1] == 3 else np.nan), n


def varying(rows, col="n_ligs"):
    v = rows.groupby("series")[col].nunique()
    return sorted(v.index[v > 1].tolist())


def species_cov(rows, fill=("n_NO3", "n_H2O"), deltas=True):
    cols = [rows[c].to_numpy(float) for c in fill]
    names = [f"gamma::{c}" for c in fill]
    if deltas:
        ser = rows["series"].to_numpy()
        nl = rows["n_ligs"].to_numpy(float)
        nlc = nl - pd.Series(nl).groupby(ser).transform("mean").to_numpy()
        for s in varying(rows):
            cols.append(np.where(ser == s, nlc, 0.0))
            names.append(f"delta::{s}")
    return np.column_stack(cols), names


ELEM_COLS = [f"c_{e}" for e in sorted(ELEMENTS_OK)]


def centred_counts(rows):
    C = rows[ELEM_COLS].to_numpy(float)
    C = C - pd.DataFrame(C).groupby(rows["series"].to_numpy()).transform("mean").to_numpy()
    return C[:, C.std(axis=0) > 1e-9]


def choose_series(g):
    cand = g.groupby(["canonical_smiles", "series"]).metal_symbol.nunique().rename("n").reset_index()
    cand["nitrate"] = cand["series"].str.endswith("nitrate")
    cand = cand.sort_values(["canonical_smiles", "n", "nitrate"], ascending=[True, False, False])
    return cand.groupby("canonical_smiles").head(1).set_index("canonical_smiles")["series"]


def fit_all(g):
    """All models on the finite-energy rows; returns rows with resid_<M> columns."""
    rows = g[np.isfinite(g["complex_total_energy_eV"].to_numpy(float))].reset_index(drop=True).copy()
    y = rows["complex_total_energy_eV"].to_numpy(float)
    ser = rows["series"].to_numpy()
    met = rows["metal_symbol"].to_numpy()
    info = {}

    rows["resid_NAIVE"], *_ = two_way(y, ser, met)
    rows["resid_ELEM"], *_ = two_way(y, ser, met, cov=centred_counts(rows))
    C, names = species_cov(rows)
    rsp, beta, X, ok = two_way(y, ser, met, cov=C)
    rows["resid_SPECIES"] = rsp
    ncol = X.shape[1] - C.shape[1]
    info["gamma_nitrate"] = float(beta[ncol])
    info["gamma_water"] = float(beta[ncol + 1])

    const = ~rows["series"].isin(varying(rows)).to_numpy()
    sub = rows[const].reset_index(drop=True)
    Cs, _ = species_cov(sub, deltas=False)
    rs, bs, Xs, _ = two_way(sub["complex_total_energy_eV"].to_numpy(float),
                            sub["series"].to_numpy(), sub["metal_symbol"].to_numpy(), cov=Cs)
    v = np.full(len(rows), np.nan)
    v[const] = rs
    rows["resid_SPECIES_CONST"] = v
    info["gamma_nitrate_const_fit"] = float(bs[Xs.shape[1] - 2])
    info["gamma_water_const_fit"] = float(bs[Xs.shape[1] - 1])

    # ---- lens-A arm (b): FIXED-coefficient composition removal, no free per-series parameter.
    # Composition energies are estimated from a GLOBAL additive element model (identified almost
    # entirely by between-ligand variation, 20..169 atoms, not by the +-3 atom within-series
    # radius-correlated variation), then SUBTRACTED as constants.  Nothing can absorb a trend.
    Xe = np.c_[np.ones(len(rows)), rows[ELEM_COLS].to_numpy(float),
               pd.get_dummies(pd.Series(met), drop_first=True).to_numpy(float)]
    be, *_ = np.linalg.lstsq(Xe, y, rcond=None)
    cel = dict(zip(sorted(ELEMENTS_OK), be[1:1 + len(ELEM_COLS)]))
    forms = g.attrs["forms"]
    EL = {s: sum(cel[e] * n for e, n in f.items()) for s, f in forms.items()}
    E_no3 = cel["N"] + 3 * cel["O"]
    E_h2o = 2 * cel["H"] + cel["O"]
    info["Ehat_NO3_eV"], info["Ehat_H2O_eV"] = float(E_no3), float(E_h2o)
    dE = (y - rows["n_ligs"].to_numpy(float) * rows["canonical_smiles"].map(EL).to_numpy(float)
          - rows["n_NO3"].to_numpy(float) * E_no3 - rows["n_H2O"].to_numpy(float) * E_h2o)
    rows["dE_FIXED"] = dE
    rows["resid_CYCLEHAT"], *_ = two_way(dE, ser, met)          # NAIVE two-way on the fixed dE

    # ---- lens-A arm (b2): JOINT within-series fit -- n_ligs entered in the SAME regression as
    # r and r^2, so the coefficient of r is the partial coefficient (unbiased, only VIF-inflated)
    # instead of the sequentially-residualised, (1-rho^2)-attenuated one the lead computes.
    return rows, info


def joint_slope(rows, model_resid, series):
    """slope of r in resid ~ 1 + r + r^2 + n_ligs + n_NO3 + n_H2O, within one series."""
    s = rows[rows["series"] == series]
    yv = s[model_resid].to_numpy(float)
    rv = s["r"].to_numpy(float)
    ok = np.isfinite(yv) & np.isfinite(rv)
    if ok.sum() < 6:
        return np.nan, int(ok.sum())
    rr = rv[ok]
    cols = [np.ones(ok.sum()), rr, rr ** 2]
    for c in ("n_ligs", "n_NO3", "n_H2O"):
        v = s[c].to_numpy(float)[ok]
        if v.std() > 1e-9:
            cols.append(v)
    X = np.column_stack(cols)
    if np.linalg.matrix_rank(X) < X.shape[1] or ok.sum() <= X.shape[1]:
        return np.nan, int(ok.sum())
    b, *_ = np.linalg.lstsq(X, yv[ok], rcond=None)
    return float(b[1]), int(ok.sum())


MODELS = ["NAIVE", "ELEM", "SPECIES", "SPECIES_CONST", "CYCLEHAT"]


def slope_table(rows, g):
    so = choose_series(g)
    out = []
    for smi, ser in so.items():
        s = rows[rows["series"] == ser].sort_values("r")
        rv = s["r"].to_numpy(float)
        base = {"extractant": smi, "series": ser,
                "n_ligs_varies": int(s["n_ligs"].nunique() > 1) if len(s) else 0,
                "n_NO3_varies": int(s["n_NO3"].nunique() > 1) if len(s) else 0,
                "n_H2O_varies": int(s["n_H2O"].nunique() > 1) if len(s) else 0}
        for m in MODELS:
            yv = s[f"resid_{m}"].to_numpy(float) if f"resid_{m}" in s else np.full(len(s), np.nan)
            sl, qu, n = slope_fit(rv, yv)
            out.append({**base, "model": m, "slope": sl, "quad": qu, "n_computed_metals": n})
        # joint-fit arms
        for m in ("NAIVE", "CYCLEHAT"):
            sl, n = joint_slope(rows, f"resid_{m}", ser)
            out.append({**base, "model": f"{m}_JOINT", "slope": sl, "quad": np.nan,
                        "n_computed_metals": n})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ statistics
def rho(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 8:
        return np.nan, np.nan, int(ok.sum())
    r = stats.spearmanr(x[ok], y[ok])
    return float(r.statistic), float(r.pvalue), int(ok.sum())


def partial_rho(x, y, z):
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if ok.sum() < 10:
        return np.nan
    rx, ry, rz = (stats.rankdata(v[ok]) for v in (x, y, z))
    A = np.c_[np.ones(ok.sum()), rz]
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def main():
    g = load()
    rows, info = fit_all(g)
    sl = slope_table(rows, g)
    t = pd.read_parquet(PHYS3D / "extractant_targets.parquet")[
        ["extractant", "a", "b", "abs_a", "n_metals", "n_cells", "chemotype"]]
    df = sl.merge(t, on="extractant", how="inner")
    df["S8"] = df["n_computed_metals"] >= 8
    df["S14"] = df["n_computed_metals"] == 14
    df["S3"] = df["n_computed_metals"] >= 3
    df.to_csv(OUT / "A_slopes.csv", index=False)
    json.dump(info, open(OUT / "A_fit_info.json", "w"), indent=2)

    recs = []
    for m in sorted(df["model"].unique()):
        d = df[df["model"] == m]
        for st in ("S8", "S14", "S3"):
            dd = d[d[st]]
            for tgt in ("a", "abs_a", "b"):
                r, p, n = rho(dd["slope"].to_numpy(float), dd[tgt].to_numpy(float))
                pr = partial_rho(dd["slope"].to_numpy(float), dd[tgt].to_numpy(float),
                                 dd["n_metals"].to_numpy(float))
                pe = (stats.pearsonr(dd["slope"], dd[tgt])[0]
                      if len(dd.dropna(subset=["slope", tgt])) >= 8 else np.nan)
                recs.append({"model": m, "set": st, "target": tgt, "n": n, "rho": r, "p": p,
                             "partial_rho_n_metals": pr, "pearson": pe})
    S = pd.DataFrame(recs)
    S.to_csv(OUT / "A_stats.csv", index=False)
    print(S[S["target"] == "a"].to_string(index=False))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
