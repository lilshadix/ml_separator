"""Turn the 1155 QC-accepted GFN2-xTB lanthanide complexes into a small per-ligand feature block.

The corpus is 1155 relaxed complexes keyed ``<Z>|<canonical_smiles>|<anion>``; 89 of the 90 cohort
extractants have at least one geometry.  Two facts govern the construction:

* the inner-sphere *composition* is not constant across a metal series -- 85 of 219 series change
  atom count across the series (fill waters / nitrates come and go as the cavity contracts), by up
  to 109 atoms.  Any energy contrast taken across metals without controlling for that is measuring
  the number of water molecules, not the metal.  Every response column here is therefore taken
  either (A) inside a maximal *constant-composition block* of a series, or (B) over the whole
  series with the element counts entered as covariates.  Both are written out.
* which metals a series has is itself unbalanced, so a "series mean" of a static descriptor is
  really a statement about which metals were computed.  Static columns are therefore the
  *intercept at r = 0* of a within-block linear fit in the standardised Shannon radius, i.e. the
  value interpolated to the middle of the lanthanide series, not a mean over whatever was run.

Outputs ``phys3d_block.parquet``, one row per canonical SMILES.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "dataset with 3D structures"
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "gen13_separation"))
from gen13sep.metals import LANTHANIDES, SHANNON_RADIUS_CN8  # noqa: E402

# standardised Shannon radius, the same axis the bench's basis row 0 uses
_r = np.array([SHANNON_RADIUS_CN8[m] for m in LANTHANIDES], dtype=float)
RSTD = {m: float(v) for m, v in zip(LANTHANIDES, (_r - _r.mean()) / _r.std())}

MIN_BLOCK = 3          # metals needed inside a composition block before a slope is taken
MIN_SERIES = 3


# --------------------------------------------------------------------------------------
# geometry composition
# --------------------------------------------------------------------------------------
def composition(xyz: Path) -> Counter:
    """Element counts of the whole complex, metal excluded.  This is what must be held constant."""
    c: Counter = Counter()
    with xyz.open() as fh:
        for i, line in enumerate(fh):
            if i == 0:
                continue
            tok = line.split()
            if len(tok) < 4:
                continue
            el = tok[0]
            if el in LANTHANIDES:
                continue
            c[el] += 1
    return c


def load_geometries() -> pd.DataFrame:
    a = pd.read_csv(DATA / "accepted_geometries.csv")
    comps, keys = [], []
    for p in a["xyz_path"]:
        c = composition(DATA / p)
        comps.append(c)
        keys.append("|".join(f"{k}{c[k]}" for k in sorted(c)))
    a["composition"] = keys
    els = sorted({e for c in comps for e in c})
    for e in els:
        a[f"n_{e}"] = [c.get(e, 0) for c in comps]
    a["n_atoms"] = [sum(c.values()) for c in comps]
    a["r"] = a["metal_symbol"].map(RSTD)
    a["series"] = a["canonical_smiles"] + "||" + a["inner_sphere_anion"]
    return a, [f"n_{e}" for e in els]


# --------------------------------------------------------------------------------------
# polyhedron -> angular descriptors
# --------------------------------------------------------------------------------------
def polyhedron_features(poly: pd.DataFrame) -> pd.DataFrame:
    ang_cols = [c for c in poly.columns if c.startswith("donor_angle_")]
    pairs = [(int(c.split("_")[2]), int(c.split("_")[3])) for c in ang_cols]
    A = poly[ang_cols].to_numpy(dtype=float)
    n = len(poly)
    cn = poly["coordination_number"].to_numpy(dtype=float)
    out = {"geometry_key": poly["geometry_key"].to_numpy()}
    ang_mean = np.full(n, np.nan)
    ang_std = np.full(n, np.nan)
    min_ang = np.full(n, np.nan)
    asym = np.full(n, np.nan)
    for i in range(n):
        v = A[i]
        ok = np.isfinite(v)
        if ok.sum() < 3:
            continue
        ang_mean[i] = v[ok].mean()
        ang_std[i] = v[ok].std()
        # per-donor nearest-neighbour angle, averaged: local crowding of the first sphere
        k = int(cn[i])
        best = {}
        for (p, q), val, o in zip(pairs, v, ok):
            if not o or p > k or q > k:
                continue
            best[p] = min(best.get(p, 999.0), val)
            best[q] = min(best.get(q, 999.0), val)
        if best:
            min_ang[i] = float(np.mean(list(best.values())))
        # |sum of unit donor vectors| / CN : 0 = balanced polyhedron, 1 = all donors one side
        s = 0.0
        m = 0
        for (p, q), val, o in zip(pairs, v, ok):
            if not o or p > k or q > k:
                continue
            s += np.cos(np.radians(val))
            m += 1
        if m == k * (k - 1) // 2 and k > 0:
            asym[i] = np.sqrt(max(0.0, k + 2 * s)) / k
    out["ang_mean"] = ang_mean
    out["ang_std"] = ang_std
    out["ang_min_mean"] = min_ang
    out["donor_asymmetry"] = asym
    # the *realised* first-sphere donor set: which atoms actually coordinate, not which the
    # recipe nominated.  Soft (N) donors are the classical handle on 4f covalency.
    zcols = [c for c in poly.columns if c.startswith("donor_atomic_number_")]
    Z = poly[zcols].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        nz = np.isfinite(Z).sum(axis=1).astype(float)
        out["frac_donor_N"] = np.where(nz > 0, (Z == 7).sum(axis=1) / np.maximum(nz, 1), np.nan)
    # radial spread of the first sphere: max - min donor distance (a shell-splitting measure)
    dcols = [f"ln_donor_distance_{i:02d}" for i in range(1, 10)]
    D = poly[dcols].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        out["d_range"] = np.nanmax(D, axis=1) - np.nanmin(D, axis=1)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------------------
# composition blocks and the two energy constructions
# --------------------------------------------------------------------------------------
def main_blocks(g: pd.DataFrame) -> pd.DataFrame:
    """Label each geometry with its series' *largest* constant-composition block."""
    g = g.copy()
    g["block"] = g["series"] + "||" + g["composition"]
    size = g.groupby(["series", "block"]).metal_symbol.nunique().rename("n").reset_index()
    # tie-break on the widest radius span so a slope is taken over as much of the series as possible
    span = g.groupby("block").r.agg(lambda v: float(v.max() - v.min())).rename("span").reset_index()
    size = size.merge(span, on="block")
    size = size.sort_values(["series", "n", "span"], ascending=[True, False, False])
    best = size.groupby("series").head(1)[["series", "block", "n"]].rename(columns={"n": "n_block"})
    g = g.merge(best, on="series", how="left", suffixes=("", "_main"))
    g["is_main_block"] = g["block"] == g["block_main"]
    return g


def _two_way_residual(y: np.ndarray, grp: np.ndarray, met: np.ndarray,
                      cov: np.ndarray | None = None) -> np.ndarray:
    """Residual of y after group fixed effects, metal fixed effects and optional covariates."""
    ok = np.isfinite(y)
    if cov is not None:
        ok &= np.isfinite(cov).all(axis=1)
    gd = pd.get_dummies(pd.Series(grp[ok]), drop_first=False).to_numpy(dtype=float)
    md = pd.get_dummies(pd.Series(met[ok]), drop_first=True).to_numpy(dtype=float)
    parts = [gd, md]
    if cov is not None:
        parts.append(cov[ok])
    X = np.hstack(parts)
    yy = y[ok] - np.nanmean(y[ok])
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    r = np.full(len(y), np.nan)
    r[ok] = yy - X @ beta
    return r


def _slope_cov(r: np.ndarray, y: np.ndarray, cov: np.ndarray) -> tuple[float, int]:
    """d y / d r over a whole series with the element counts entered as covariates.

    Construction B for a *structural* response.  The counts are already centred within the series,
    so the intercept is the series mean and the r coefficient is the part of the response that the
    change of inner-sphere composition does not explain.
    """
    ok = np.isfinite(r) & np.isfinite(y) & np.isfinite(cov).all(axis=1)
    n = int(ok.sum())
    keep = cov[ok].std(axis=0) > 1e-9 if n else np.zeros(cov.shape[1], dtype=bool)
    if n < MIN_SERIES + int(keep.sum()):
        return np.nan, n
    X = np.c_[np.ones(n), r[ok], cov[ok][:, keep]]
    beta, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
    return float(beta[1]), n


def _slope_fit(r: np.ndarray, y: np.ndarray, quad: bool = True) -> tuple[float, float, float, int]:
    ok = np.isfinite(r) & np.isfinite(y)
    n = int(ok.sum())
    if n < MIN_BLOCK:
        return np.nan, np.nan, np.nan, n
    rr, yy = r[ok], y[ok]
    if quad and n >= 4:
        X = np.c_[np.ones(n), rr, rr ** 2]
    else:
        X = np.c_[np.ones(n), rr]
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    res = yy - X @ beta
    rms = float(np.sqrt((res ** 2).mean()))
    return float(beta[1]), (float(beta[2]) if X.shape[1] == 3 else np.nan), rms, n


# --------------------------------------------------------------------------------------
def build() -> pd.DataFrame:
    g, count_cols = load_geometries()
    sc = pd.read_parquet(DATA / "features/complex_physical_scalars.parquet")
    poly = pd.read_parquet(DATA / "features/coordination_polyhedron.parquet")
    pf = polyhedron_features(poly)
    sc = sc.drop(columns=[c for c in sc.columns if sc[c].isna().all()])
    g = g.merge(sc, on="geometry_key", how="left", suffixes=("", "_sc"))
    g = g.merge(pf, on="geometry_key", how="left")
    g = main_blocks(g)

    # ---- energy, construction A: two-way FE on constant-composition blocks only ------------
    E = g["complex_total_energy_eV"].to_numpy(dtype=float)
    mA = g["is_main_block"].to_numpy() & np.isfinite(E)
    gA = g[mA]
    rA = _two_way_residual(E[mA], gA["block"].to_numpy(), gA["metal_symbol"].to_numpy())
    g.loc[mA, "E_resid_A"] = rA

    # ---- energy, construction B: whole series, element counts as within-series covariates ---
    mB = np.isfinite(E)
    gB = g[mB]
    C = gB[count_cols].to_numpy(dtype=float)
    C = C - pd.DataFrame(C).groupby(gB["series"].to_numpy()).transform("mean").to_numpy()
    keep = C.std(axis=0) > 1e-9
    rB = _two_way_residual(E[mB], gB["series"].to_numpy(), gB["metal_symbol"].to_numpy(),
                           cov=C[:, keep])
    g.loc[mB, "E_resid_B"] = rB

    # ---- per-SMILES assembly --------------------------------------------------------------
    STATIC = {
        "ln_donor_distance_mean": "d_mean",
        "ln_donor_distance_std": "d_spread",
        "d_range": "d_range",
        "metal_partial_charge": "q_metal",
        "donor_partial_charge_mean": "q_donor",
        "donor_partial_charge_std": "q_donor_spread",
        "coordination_number": "cn",
        "frac_donor_N": "frac_N",
        "dipole_magnitude": "dipole",
        "ang_std": "ang_std",
        "ang_min_mean": "ang_min",
        "donor_asymmetry": "asym",
    }
    RESP_A = {                       # inside the constant-composition block
        "ln_donor_distance_mean": "dd_dr_A",
        "ln_donor_distance_std": "dspread_dr_A",
        "metal_partial_charge": "dqm_dr_A",
        "donor_partial_charge_mean": "dqd_dr_A",
    }
    RESP_B = {                       # whole series, element counts as covariates
        "ln_donor_distance_mean": "dd_dr_B",
    }
    rows = []
    for smi, sub in g.groupby("canonical_smiles"):
        # preferred series: whichever anion covers more metals (tie -> nitrate)
        cand = sub.groupby("series").agg(n=("metal_symbol", "nunique")).reset_index()
        cand["nitrate"] = cand["series"].str.endswith("nitrate")
        cand = cand.sort_values(["n", "nitrate"], ascending=[False, False])
        ser = cand["series"].iat[0]
        s = sub[sub["series"] == ser].sort_values("r")
        blk = s[s["is_main_block"]]
        row = {"extractant": smi,
               "phys3d__anion_nitrate": float(ser.endswith("nitrate")),
               "phys3d__n_series": float(s["metal_symbol"].nunique()),
               "phys3d__n_block": float(blk["metal_symbol"].nunique()),
               "phys3d__comp_varies": float(s["composition"].nunique() > 1)}
        rvec = s["r"].to_numpy(dtype=float)
        bvec = blk["r"].to_numpy(dtype=float)
        Cs = s[count_cols].to_numpy(dtype=float)
        Cs = Cs - Cs.mean(axis=0, keepdims=True)          # element counts centred within the series
        # ---- static: the value interpolated to the middle of the series (r = 0) -------------
        # fitted over the whole series, not the block: an absolute level is not a contrast, so
        # composition change adds noise to it rather than a direction, and the full series both
        # brackets r = 0 and has roughly twice the support.
        for col, name in STATIC.items():
            if col not in s.columns:
                row[f"phys3d__{name}"] = np.nan
                continue
            y = s[col].to_numpy(dtype=float)
            ok = np.isfinite(rvec) & np.isfinite(y)
            if ok.sum() >= MIN_SERIES and np.ptp(rvec[ok]) > 1e-9:
                X = np.c_[np.ones(int(ok.sum())), rvec[ok]]
                beta, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
                row[f"phys3d__{name}"] = float(beta[0])
            elif ok.sum() >= 1:
                row[f"phys3d__{name}"] = float(np.mean(y[ok]))
            else:
                row[f"phys3d__{name}"] = np.nan
        # ---- responses to the metal ----------------------------------------------------------
        for col, name in RESP_A.items():
            sl, _, _, _ = _slope_fit(bvec, blk[col].to_numpy(dtype=float), quad=False)
            row[f"phys3d__{name}"] = sl
        for col, name in RESP_B.items():
            sl, _ = _slope_cov(rvec, s[col].to_numpy(dtype=float), Cs)
            row[f"phys3d__{name}"] = sl
        # d(CN)/dr over the whole series and *without* a composition control, because here the
        # composition change is the quantity: does the first sphere shed a filler as the cation
        # contracts?  Inside a constant-composition block this is identically zero.
        sl, _, _, _ = _slope_fit(rvec, s["coordination_number"].to_numpy(dtype=float), quad=False)
        row["phys3d__dcn_dr_raw"] = sl
        sl, cu, rms, _ = _slope_fit(bvec, blk["E_resid_A"].to_numpy(dtype=float), quad=True)
        row["phys3d__dE_dr_A"], row["phys3d__d2E_dr2_A"], row["phys3d__E_rms_A"] = sl, cu, rms
        slb, cub, rmsb, _ = _slope_fit(rvec, s["E_resid_B"].to_numpy(dtype=float), quad=True)
        row["phys3d__dE_dr_B"], row["phys3d__d2E_dr2_B"], row["phys3d__E_rms_B"] = slb, cub, rmsb
        rows.append(row)
    blockdf = pd.DataFrame(rows)
    return blockdf, g


if __name__ == "__main__":
    blockdf, g = build()
    OUT.mkdir(parents=True, exist_ok=True)
    blockdf.to_parquet(OUT / "phys3d_block.parquet", index=False)
    g_small = g[["geometry_key", "canonical_smiles", "inner_sphere_anion", "metal_symbol", "r",
                 "composition", "block", "is_main_block", "n_block", "n_atoms",
                 "complex_total_energy_eV", "E_resid_A", "E_resid_B"]]
    g_small.to_parquet(OUT / "geometry_long.parquet", index=False)
    print("block", blockdf.shape)
    print(blockdf.describe().T.round(4).to_string())
    print("\nnon-null per column:")
    print(blockdf.notna().sum().to_string())
