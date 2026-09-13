"""REFUTER lens A (round 2) for claim L1NULL.

Independent re-derivation of the L1 Stage-1 headline from the PRE_REGISTRATION text alone
(own xyz parsing, own RDKit decomposition, own design matrices, own slope fit), then the
bookkeeping / leakage audit and the mandatory checks.

Nothing under gen13_separation/, gen14_direction/, gen15_curve/, src/ is written.
Outputs: gen16_leads/results/refutation/L1NULL/A/A4_*.csv|json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(r"D:\ml_separator_gh")
DATA = ROOT / "dataset with 3D structures"
PHYS3D = ROOT / "gen15_curve" / "exp" / "phys3d"
LEAD = ROOT / "gen16_leads" / "results" / "L1"
OUT = ROOT / "gen16_leads" / "results" / "refutation" / "L1NULL" / "A"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "gen13_separation"))
from gen13sep.metals import LANTHANIDES, SHANNON_RADIUS_CN8  # noqa: E402

_r = np.array([SHANNON_RADIUS_CN8[m] for m in LANTHANIDES], dtype=float)
RSTD = {m: float(v) for m, v in zip(LANTHANIDES, (_r - _r.mean()) / _r.std())}
LN = set(LANTHANIDES)


# ---------------------------------------------------------------- data, my own parsing
def xyz_counts(path: Path, metal: str) -> Counter:
    """Element counts of the complex EXCLUDING the metal.  Lines 0 (natoms) and 1 (comment)
    are both header lines -- this is the parsing gen15's build_block.composition() gets wrong."""
    lines = path.read_text().splitlines()
    nat = int(lines[0].split()[0])
    c: Counter = Counter()
    for ln in lines[2:2 + nat]:
        tok = ln.split()
        el = tok[0]
        if el in LN:
            continue
        c[el] += 1
    return c


def ligand_formula(smiles: str) -> Counter:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    return Counter(at.GetSymbol() for at in Chem.AddHs(Chem.MolFromSmiles(smiles)).GetAtoms())


def load() -> tuple[pd.DataFrame, list[str], dict]:
    a = pd.read_csv(DATA / "accepted_geometries.csv")
    comps = [xyz_counts(DATA / p, m) for p, m in zip(a["xyz_path"], a["metal_symbol"])]
    els = sorted({e for c in comps for e in c})
    for e in els:
        a[f"n_{e}"] = [c.get(e, 0) for c in comps]
    a["r"] = a["metal_symbol"].map(RSTD)
    a["series"] = a["canonical_smiles"] + "||" + a["inner_sphere_anion"]
    forms = {s: ligand_formula(s) for s in a["canonical_smiles"].unique()}
    nno3, nh2o, ok = [], [], []
    for smi, nl, i in zip(a["canonical_smiles"], a["n_ligs"], range(len(a))):
        f = forms[smi]
        res = {e: int(a[f"n_{e}"].iat[i]) - int(nl) * f.get(e, 0) for e in els}
        res = {e: v for e, v in res.items() if v}
        nN, nH, nO = res.pop("N", 0), res.pop("H", 0), res.pop("O", 0)
        good = (not res) and nN >= 0 and nH >= 0 and nH % 2 == 0 and nO == 3 * nN + nH // 2
        ok.append(good)
        nno3.append(nN if good else -1)
        nh2o.append(nH // 2 if good else -1)
    a["n_NO3"], a["n_H2O"], a["fill_ok"] = nno3, nh2o, ok
    sc = pd.read_parquet(DATA / "features/complex_physical_scalars.parquet")
    a = a.merge(sc[["geometry_key", "complex_total_energy_eV"]], on="geometry_key", how="left")
    return a, [f"n_{e}" for e in els], forms


# ---------------------------------------------------------------- own linear algebra
def two_way(y, grp, met, cov=None):
    """y ~ intercept + series FE (drop-first) + metal FE (drop-first) [+ cov].  A DIFFERENT
    parameterisation from the lead's (full series dummies, demeaned y) -- residuals must agree."""
    ok = np.isfinite(y)
    if cov is not None:
        ok &= np.isfinite(cov).all(axis=1)
    gd = pd.get_dummies(pd.Series(grp[ok]), drop_first=True).to_numpy(float)
    md = pd.get_dummies(pd.Series(met[ok]), drop_first=True).to_numpy(float)
    parts = [np.ones((int(ok.sum()), 1)), gd, md]
    if cov is not None:
        parts.append(cov[ok])
    X = np.hstack(parts)
    beta, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
    r = np.full(len(y), np.nan)
    r[ok] = y[ok] - X @ beta
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


def centred_counts(rows, cols):
    C = rows[cols].to_numpy(float)
    C = C - pd.DataFrame(C).groupby(rows["series"].to_numpy()).transform("mean").to_numpy()
    return C[:, C.std(axis=0) > 1e-9]


def rho(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5:
        return np.nan, np.nan, int(ok.sum())
    r = stats.spearmanr(x[ok], y[ok])
    return float(r.statistic), float(r.pvalue), int(ok.sum())


def prho(x, y, z):
    """partial Spearman of x,y given z (rank-residual construction)."""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if ok.sum() < 6:
        return np.nan
    rx, ry, rz = (stats.rankdata(v[ok]) for v in (x, y, z))
    A = np.c_[np.ones(ok.sum()), rz]
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return float(stats.pearsonr(ex, ey).statistic)


# ---------------------------------------------------------------- models
def fit_all(rows, count_cols, forms, y=None):
    y = rows["complex_total_energy_eV"].to_numpy(float) if y is None else y
    ser, met = rows["series"].to_numpy(), rows["metal_symbol"].to_numpy()
    out, info = {}, {}
    out["NAIVE"] = two_way(y, ser, met)[0]
    out["ELEM"] = two_way(y, ser, met, cov=centred_counts(rows, count_cols))[0]
    C, nm = species_cov(rows)
    rS, bS, XS, okS = two_way(y, ser, met, cov=C)
    out["SPECIES"] = rS
    ncov = C.shape[1]
    info["gamma_nitrate"] = float(bS[-ncov])
    info["gamma_water"] = float(bS[-ncov + 1])
    # SPECIES_NFILLCOL (exploratory variant): n_fill column split by fill species
    nf = rows["n_fill"].to_numpy(float)
    isn = (rows["fill_ligand"].to_numpy() == "nitrate").astype(float)
    tmp = rows.copy()
    tmp["_a"], tmp["_b"] = nf * isn, nf * (1 - isn)
    Cn, _ = species_cov(tmp, fill=("_a", "_b"))
    out["SPECIES_NFILLCOL"] = two_way(y, ser, met, cov=Cn)[0]
    # SPECIES_CONST: constant-n_ligs series only, no delta at all
    const = ~rows["series"].isin(varying(rows)).to_numpy()
    sub = rows[const].reset_index(drop=True)
    Cs, _ = species_cov(sub, deltas=False)
    rs, bs, _, _ = two_way(y[const], sub["series"].to_numpy(), sub["metal_symbol"].to_numpy(), cov=Cs)
    v = np.full(len(rows), np.nan)
    v[const] = rs
    out["SPECIES_CONST"] = v
    info["gamma_nitrate_constfit"] = float(bs[-2])
    info["gamma_water_constfit"] = float(bs[-1])
    info["n_const_rows"] = int(const.sum())

    # ---- attack (b): fixed-coefficient cycles, NO free delta anywhere -------------
    # (b1) CYCLE_ADD: per-ligand energy from a global element-additive regression with metal FE,
    #      fill energies from the same regression; subtract with FIXED coefficients, then NAIVE.
    elem_cols = [c for c in count_cols]
    Xe = np.c_[np.ones(len(rows)), rows[elem_cols].to_numpy(float),
               pd.get_dummies(pd.Series(met), drop_first=True).to_numpy(float)]
    oke = np.isfinite(y)
    be, *_ = np.linalg.lstsq(Xe[oke], y[oke], rcond=None)
    cel = {c[2:]: float(v_) for c, v_ in zip(elem_cols, be[1:1 + len(elem_cols)])}
    EL = {s: sum(cel.get(e, 0.0) * n for e, n in f.items()) for s, f in forms.items()}
    E_no3 = cel.get("N", 0.0) + 3 * cel.get("O", 0.0)
    E_h2o = 2 * cel.get("H", 0.0) + cel.get("O", 0.0)
    lig_e = np.array([EL[s] for s in rows["canonical_smiles"]], dtype=float)
    corr_add = (rows["n_ligs"].to_numpy(float) * lig_e
                + rows["n_NO3"].to_numpy(float) * E_no3 + rows["n_H2O"].to_numpy(float) * E_h2o)
    out["CYCLE_ADD"] = two_way(y - corr_add, ser, met)[0]
    info.update(E_no3_additive=E_no3, E_h2o_additive=E_h2o)
    # (b2) CYCLE_CONSTGAMMA: fill energies taken from the SPECIES_CONST fit (identified on series
    #      where n_ligs never varies, so no delta was ever needed); ligand energy additive.
    corr_cg = (rows["n_ligs"].to_numpy(float) * lig_e
               + rows["n_NO3"].to_numpy(float) * info["gamma_nitrate_constfit"]
               + rows["n_H2O"].to_numpy(float) * info["gamma_water_constfit"])
    out["CYCLE_CONSTGAMMA"] = two_way(y - corr_cg, ser, met)[0]
    # (b3) SPECIES_NODELTA: exact SPECIES design but with the free per-series delta REMOVED
    #      (gamma still global, ligand term left to the series FE only).
    Cnd, _ = species_cov(rows, deltas=False)
    out["SPECIES_NODELTA"] = two_way(y, ser, met, cov=Cnd)[0]
    return out, info


def choose_series(g):
    cand = g.groupby(["canonical_smiles", "series"]).metal_symbol.nunique().rename("n").reset_index()
    cand["nit"] = cand["series"].str.endswith("nitrate")
    cand = cand.sort_values(["canonical_smiles", "n", "nit"], ascending=[True, False, False])
    return cand.groupby("canonical_smiles").head(1).set_index("canonical_smiles")["series"]


def slopes(rows, resid, g_all):
    sof = choose_series(g_all)
    recs = []
    for smi, ser in sof.items():
        s = rows[rows["series"] == ser]
        rv = s["r"].to_numpy(float)
        idx = s.index.to_numpy()
        base = {"extractant": smi, "series": ser,
                "n_ligs_varies": int(s["n_ligs"].nunique() > 1) if len(s) else 0,
                "n_NO3_varies": int(s["n_NO3"].nunique() > 1) if len(s) else 0,
                "n_H2O_varies": int(s["n_H2O"].nunique() > 1) if len(s) else 0}
        for m, v in resid.items():
            sl, qd, n = slope_fit(rv, v[idx])
            recs.append({**base, "model": m, "slope": sl, "quad": qd, "n_computed_metals": n})
    return pd.DataFrame(recs)


def main():
    rows_all, count_cols, forms = load()
    assert rows_all["fill_ok"].all(), "decomposition failed"
    fin = np.isfinite(rows_all["complex_total_energy_eV"].to_numpy(float))
    rows = rows_all[fin].reset_index(drop=True)
    resid, info = fit_all(rows, count_cols, forms)
    sl = slopes(rows, resid, rows_all)
    tgt = pd.read_parquet(PHYS3D / "extractant_targets.parquet")[
        ["extractant", "a", "b", "abs_a", "n_metals", "n_cells", "chemotype"]]
    df = sl.merge(tgt, on="extractant", how="inner")
    n = df["n_computed_metals"].to_numpy()
    df["S8"], df["S14"], df["S3"] = n >= 8, n == 14, n >= 3
    df.to_csv(OUT / "A4_slopes.csv", index=False)

    # ---------------- PART A: headline, digit for digit ---------------------------
    recs = []
    for m in df["model"].unique():
        d = df[df["model"] == m]
        for s in ("S8", "S14", "S3"):
            dd = d[d[s]]
            for t in ("a", "abs_a", "b"):
                r, p, nn = rho(dd["slope"].to_numpy(float), dd[t].to_numpy(float))
                pr = prho(dd["slope"].to_numpy(float), dd[t].to_numpy(float),
                          dd["n_metals"].to_numpy(float))
                pear = (stats.pearsonr(dd["slope"], dd[t]).statistic
                        if len(dd) > 4 and np.isfinite(dd["slope"]).all() else np.nan)
                recs.append({"model": m, "set": s, "target": t, "n": nn, "rho": r, "p": p,
                             "partial_rho_n_metals": pr, "pearson": pear})
    mine = pd.DataFrame(recs)
    mine.to_csv(OUT / "A4_stats.csv", index=False)

    lead = pd.read_csv(LEAD / "stage1_stats.csv")
    cmp_rows = []
    for _, r in mine.iterrows():
        L = lead[(lead["model"] == r["model"]) & (lead["set"] == r["set"])
                 & (lead["target"] == r["target"]) & (lead.get("value", "slope") == "slope")]
        if not len(L):
            continue
        L = L.iloc[0]
        cmp_rows.append({"model": r["model"], "set": r["set"], "target": r["target"],
                         "n_mine": r["n"], "n_lead": int(L["n"]), "rho_mine": r["rho"],
                         "rho_lead": float(L["rho"]), "d_rho": r["rho"] - float(L["rho"]),
                         "partial_mine": r["partial_rho_n_metals"],
                         "partial_lead": float(L["partial_rho_n_metals"])})
    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv(OUT / "A4_reproduction_diff.csv", index=False)

    S = {"max_abs_d_rho_vs_lead": float(cmp["d_rho"].abs().max()),
         "max_abs_d_n_vs_lead": int((cmp["n_mine"] - cmp["n_lead"]).abs().max()),
         "info": info}

    # ---------------- PART B: bookkeeping / leakage audit -------------------------
    d8 = df[(df["model"] == "SPECIES") & df["S8"]]
    sets_by_model = {m: sorted(df[(df["model"] == m) & df["S8"]]["extractant"].tolist())
                     for m in df["model"].unique()}
    ref = sets_by_model["NAIVE"]
    S["S8_unit_sets_identical"] = {m: (v == ref) for m, v in sets_by_model.items()}
    S["S8_sizes"] = {m: len(v) for m, v in sets_by_model.items()}
    S["n_chemotypes_in_S8"] = int(d8["chemotype"].nunique())
    S["n_chemotypes_in_targets"] = int(tgt["chemotype"].nunique())
    cc = d8["chemotype"].value_counts().to_numpy(float)
    S["kish_neff_S8"] = float(cc.sum() ** 2 / (cc ** 2).sum())
    S["sc009_in_S8"] = int((d8["chemotype"] == "sc009").sum())
    S["sc009_in_targets"] = int((tgt["chemotype"] == "sc009").sum())
    # phantom element column of the frozen build_block parser
    sys.path.insert(0, str(PHYS3D))
    from build_block import load_geometries  # noqa: E402
    gg, lead_cols = load_geometries()
    S["lead_count_cols"] = [c[2:][:14] for c in lead_cols]
    S["my_count_cols"] = [c[2:] for c in count_cols]
    S["n_phantom_cols_in_frozen_parser"] = len([c for c in lead_cols if c not in count_cols])
    # does the phantom column survive centring (i.e. enter ELEM)?
    ggf = gg.merge(pd.read_parquet(DATA / "features/complex_physical_scalars.parquet")[
        ["geometry_key", "complex_total_energy_eV"]], on="geometry_key", how="left")
    ggf["series"] = ggf["canonical_smiles"] + "||" + ggf["inner_sphere_anion"]
    ggf = ggf[np.isfinite(ggf["complex_total_energy_eV"].to_numpy(float))].reset_index(drop=True)
    Cl = ggf[lead_cols].to_numpy(float)
    Cl = Cl - pd.DataFrame(Cl).groupby(ggf["series"].to_numpy()).transform("mean").to_numpy()
    keepl = Cl.std(axis=0) > 1e-9
    S["phantom_enters_ELEM"] = bool(any(keepl[i] for i, c in enumerate(lead_cols) if c not in count_cols))
    S["n_elem_cov_lead"] = int(keepl.sum())
    S["n_elem_cov_mine"] = int(centred_counts(rows, count_cols).shape[1])
    # choose_series ties
    cand = rows_all.groupby(["canonical_smiles", "series"]).metal_symbol.nunique().rename("n").reset_index()
    top = cand.groupby("canonical_smiles")["n"].transform("max")
    S["n_extractants_with_series_tie"] = int((cand[cand["n"] == top]
                                              .groupby("canonical_smiles").size() > 1).sum())
    S["targets_rows"] = int(len(tgt))
    S["targets_unique"] = int(tgt["extractant"].nunique())
    S["merge_kept"] = int(df["extractant"].nunique())

    # LOCO exactly as the lead codes it (40 folds over all targets rows) vs folds that
    # actually remove an S8 member
    out_loco = []
    for m in ("NAIVE", "SPECIES", "SPECIES_CONST", "ELEM", "CYCLE_ADD", "CYCLE_CONSTGAMMA",
              "SPECIES_NODELTA"):
        d = df[(df["model"] == m) & df["S8"]]
        x, y2, ch = d["slope"].to_numpy(float), d["a"].to_numpy(float), d["chemotype"].to_numpy()
        full = rho(x, y2)[0]
        vals_present, vals_all = [], []
        for c in tgt["chemotype"].unique():
            mk = ch != c
            rr = rho(x[mk], y2[mk])[0] if mk.sum() >= 5 else np.nan
            if np.isfinite(rr):
                vals_all.append(rr)
                if (ch == c).any():
                    vals_present.append(rr)
        out_loco.append({"model": m, "rho": full, "n": len(x),
                         "n_folds_lead_style": len(vals_all), "n_folds_effective": len(vals_present),
                         "loco_min_all": float(np.min(vals_all)), "loco_max_all": float(np.max(vals_all)),
                         "loco_min_present": float(np.min(vals_present)),
                         "loco_max_present": float(np.max(vals_present))})
    pd.DataFrame(out_loco).to_csv(OUT / "A4_loco.csv", index=False)

    # ---------------- PART C: mandatory checks ------------------------------------
    checks = []

    def add(name, model, subset_mask, label):
        d = df[(df["model"] == model) & subset_mask(df)]
        r, p, nn = rho(d["slope"].to_numpy(float), d["a"].to_numpy(float))
        checks.append({"check": name, "model": model, "subset": label, "n": nn, "rho": r, "p": p})

    for m in ("NAIVE", "ELEM", "SPECIES", "SPECIES_CONST", "SPECIES_NFILLCOL",
              "CYCLE_ADD", "CYCLE_CONSTGAMMA", "SPECIES_NODELTA"):
        add("1_all_S8", m, lambda d: d["S8"], "S8")
        add("1_no_sc009", m, lambda d: d["S8"] & (d["chemotype"] != "sc009"), "S8 minus sc009")
        add("1_only_sc009", m, lambda d: d["S8"] & (d["chemotype"] == "sc009"), "S8 sc009 only")
        add("2_nm_lt14", m, lambda d: d["S8"] & (d["n_metals"] < 14), "S8 n_metals<14")
        add("2_nm_eq14", m, lambda d: d["S8"] & (d["n_metals"] == 14), "S8 n_metals==14")
        add("4_const19", m, lambda d: d["S8"] & (d["n_ligs_varies"] == 0), "S8 constant n_ligs")
        add("4_varying43", m, lambda d: d["S8"] & (d["n_ligs_varies"] == 1), "S8 varying n_ligs")
        add("c_S14", m, lambda d: d["S14"], "S14")
        add("c_S3", m, lambda d: d["S3"], "S3")
    pd.DataFrame(checks).to_csv(OUT / "A4_checks.csv", index=False)

    # ---------------- PART D: attack (d) composition-change position ---------------
    comp = []
    sof = choose_series(rows_all)
    for smi, ser in sof.items():
        s = rows[rows["series"] == ser].sort_values("r")
        if len(s) < 3:
            continue
        nl = s["n_ligs"].to_numpy(float)
        rv = s["r"].to_numpy(float)
        step_r = np.nan
        if nl.min() != nl.max():
            # radius at which n_ligs first drops below its max, scanning from large r to small
            hi = nl.max()
            below = rv[nl < hi]
            step_r = float(below.max()) if len(below) else np.nan
        cn = (s["n_ligs"].to_numpy(float) * 0 + 1)  # placeholder, coreCN below
        core_cn = s["n_fill"].to_numpy(float) + np.nan  # not used
        comp.append({"extractant": smi, "n_comp": int(s["n_ligs"].astype(str)
                     .str.cat(s["n_NO3"].astype(str)).str.cat(s["n_H2O"].astype(str)).nunique()),
                     "nligs_range": float(nl.max() - nl.min()),
                     "step_r": step_r,
                     "frac_at_max_nligs": float((nl == nl.max()).mean()),
                     "mean_nligs": float(nl.mean()),
                     "mean_nNO3": float(s["n_NO3"].mean()),
                     "mean_nH2O": float(s["n_H2O"].mean()),
                     "n_series_rows": int(len(s))})
    comp = pd.DataFrame(comp).merge(tgt, on="extractant", how="inner")
    comp = comp.merge(df[(df["model"] == "NAIVE")][["extractant", "S8", "S14", "n_computed_metals"]],
                      on="extractant", how="left")
    comp.to_csv(OUT / "A4_composition_descriptors.csv", index=False)
    dcomp = []
    for col in ("n_comp", "nligs_range", "step_r", "frac_at_max_nligs", "mean_nligs",
                "mean_nNO3", "mean_nH2O"):
        for s in ("S8", "S14"):
            d = comp[comp[s].fillna(False)]
            for t in ("a", "abs_a", "b"):
                r, p, nn = rho(d[col].to_numpy(float), d[t].to_numpy(float))
                dcomp.append({"descriptor": col, "set": s, "target": t, "n": nn, "rho": r, "p": p})
    pd.DataFrame(dcomp).to_csv(OUT / "A4_composition_stats.csv", index=False)

    with open(OUT / "A4_summary.json", "w", encoding="utf-8") as fh:
        json.dump(S, fh, indent=2, default=float)
    print(json.dumps({k: v for k, v in S.items() if k != "info"}, indent=2, default=float)[:4000])
    print("\n--- reproduction diff (target a) ---")
    print(cmp[cmp["target"] == "a"].to_string(index=False))
    print("\n--- LOCO ---")
    print(pd.DataFrame(out_loco).to_string(index=False))


if __name__ == "__main__":
    main()
