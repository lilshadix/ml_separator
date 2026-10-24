#!/usr/bin/env python
"""shift theme, step 3 (EXPLORATORY: 229 already-seen C3 cells, 5 dev seeds): condition-shift physics arms for
condition transfer (estimand E-CT), on the round-4 REF rows (k = 3 DOPT_Q, scored at 4 reads), with the P1
population and the P1 arms rebuilt on identical rows.

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/shift_build.py

Writes only ideas2/scripts/shift_out/shift_rows.parquet and shift_build_log.json.

Population (copied from probes/scripts/P1_ct_build.py, unchanged): sibling = other bench cell, same publication and
extractant SMILES, finite V1 log D at every metal of the target; CLEAN = outside V2-1/2/3 and differing from the
target in at least one cond__ family other than contact time / metal concentration.

New sibling subsets (metadata only):
  AS (acid-only): CLEAN siblings whose differing families (minus trivial) are exactly {acid_conc}.  x = log10[acid].
  CS (continuous-only): CLEAN siblings whose differing families (minus trivial) lie inside {acid_conc, extr_conc,
      temperature} and whose varying coordinates are finite.  z = (log10[acid], log10[L], 1000/T[K]).

Template arms (levels on the target's metals; at k reads the template is read-corrected exactly like P1's SIB:
X_k = T + PWL_R[y_S - T_S]; k = 0 is the raw template):
  ACOPY  mean of AS                               (P1's SIB restricted to AS)
  AREG   per-metal OLS of log D on x over AS, evaluated at x_t (needs >= 2 distinct x spanning >= 0.1; else = ACOPY)
  AREGH  AREG with the metal-specific slope deviation from the across-metal mean slope halved (lambda = 0.5)
  AINT   per-metal piecewise-linear interpolation in x between AS levels (cells at one x averaged), flat beyond
  AINTX  AINT but linear extrapolation from the two nearest x levels beyond the range
  TILT   ACOPY + g_m (x_t - mean x_AS), g = cumulative per-metal profile of the pooled adjacent log SF acid slopes of
         OTHER publications (equal weight per publication; same acid type when >= 3 publications have it, else all)
  CCOPY / CREG / CREGH  as ACOPY / AREG / AREGH over CS with the multi-coordinate z (full rank required)
Shrunk versions <ARM>SH_k = w X_k + (1 - w) B1_k, w = n/(n + 1.78), n = size of that arm's sibling subset.
P1 arms on the same rows: B1_k, B0_3, SIB_k, SH_k (all CLEAN), COPY, OFF.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402
from r3 import datav2 as V2  # noqa: E402
from r3 import interp  # noqa: E402
from r3 import corrector as CO  # noqa: E402
from g21 import cache as C  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import N_LN, LANTHANIDES  # noqa: E402

OUT = Path(r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2\scripts\shift_out")
K_SH = 1.78
GD = CO.GD
C5 = "pub_6a0ef8528c"
DESIGNS = ("BPG", "BP")
FAMS_TRIVIAL = {"contact_time", "metal_conc"}
PAIRS = [(i, i + 1) for i in range(N_LN - 1)]


def fam(c: str) -> str:
    if c.startswith("cond__acid__"):
        return "acid_type"
    if c == "cond__acid_concentration_M":
        return "acid_conc"
    if c.startswith("cond__additive__"):
        return "additive"
    if c.startswith("cond__diluent__"):
        return "diluent"
    return {"cond__contact_time_min": "contact_time", "cond__extractant_concentration_M": "extr_conc",
            "cond__metal_concentration_mM": "metal_conc", "cond__temperature_C": "temperature"}[c]


def change_class(fams_list) -> str:
    U = set().union(*fams_list) if fams_list else set()
    U = U - {"contact_time", "metal_conc"}
    if U == {"acid_conc"}:
        return "acid_conc_only"
    if U and U <= {"acid_conc", "acid_type"}:
        return "acid_type(+conc)"
    if U == {"diluent"}:
        return "diluent_only"
    if U == {"extr_conc"}:
        return "extr_conc_only"
    if U == {"temperature"}:
        return "temperature_only"
    return "mixed"


def ols_slope(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2 or len(np.unique(np.round(x, 3))) < 2 or x.max() - x.min() < 0.1:
        return np.nan
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc ** 2).sum())


def fit_template(Ys: np.ndarray, Z: np.ndarray, zt: np.ndarray, lam: float):
    """Per-metal linear model of log D on condition coordinates; metal-specific slope deviations scaled by lam."""
    zb = Z.mean(0)
    Zc = Z - zb
    if np.linalg.matrix_rank(Zc, tol=1e-6) < Z.shape[1]:
        return None
    Yb = Ys.mean(0)
    Bm, *_ = np.linalg.lstsq(Zc, Ys - Yb, rcond=None)      # p x 14 (NaN columns handled by caller)
    bbar = np.nanmean(Bm, axis=1, keepdims=True)
    Bs = bbar + lam * (Bm - bbar)
    return Yb + (zt - zb) @ Bs


def interp_template(Ys, x, xt, extrap: bool):
    xr = np.round(x, 4)
    xu = np.unique(xr)
    Yu = np.array([Ys[xr == v].mean(0) for v in xu])
    if len(xu) == 1:
        return Yu[0]
    out = np.array([np.interp(xt, xu, Yu[:, m]) for m in range(Ys.shape[1])])
    if extrap and (xt < xu[0] or xt > xu[-1]):
        i0, i1 = (0, 1) if xt < xu[0] else (len(xu) - 2, len(xu) - 1)
        out = Yu[i0] + (Yu[i1] - Yu[i0]) * (xt - xu[i0]) / (xu[i1] - xu[i0])
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    Y = np.asarray(bench.Y, float)
    ids = f["cell_id"].astype(str).tolist()
    pos = {c: i for i, c in enumerate(ids)}
    v2drop = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    cond_cols = [c for c in f.columns if c.startswith("cond__")]
    F = f[cond_cols]
    pub = f["publication_id"].astype(str).to_numpy()
    ext = f["extractant"].astype(str).to_numpy()
    chem = f["chemotype"].astype(str).to_numpy()
    acid_cols = [c for c in cond_cols if c.startswith("cond__acid__")]
    acid_lab = np.array(["+".join(c[12:] for c in acid_cols if f.at[i, c] == 1) or "-" for i in range(len(f))])
    xacid = np.log10(f.cond__acid_concentration_M.astype(float).to_numpy())
    zL = np.log10(f.cond__extractant_concentration_M.astype(float).clip(lower=1e-4).to_numpy())
    zT = 1000.0 / (f.cond__temperature_C.astype(float).to_numpy() + 273.15)
    Zall = np.c_[xacid, zL, zT]

    def diff_fams(i, j) -> frozenset:
        s = set()
        for c in cond_cols:
            x, y = F.iat[i, F.columns.get_loc(c)], F.iat[j, F.columns.get_loc(c)]
            if pd.isna(x) and pd.isna(y):
                continue
            if pd.isna(x) or pd.isna(y) or not np.isclose(float(x), float(y)):
                s.add(fam(c))
        return frozenset(s)

    # ---------- cross-publication acid-slope prior (bench-wide, V2-clean; same grouping as shift_slopes.py) -------
    dil = ["+".join(c for c in cond_cols if c.startswith("cond__diluent__") and f.at[i, c] == 1) for i in range(len(f))]
    add = ["+".join(c for c in cond_cols if c.startswith("cond__additive__") and f.at[i, c] == 1) for i in range(len(f))]
    gkey = pd.Series([(pub[i], ext[i], acid_lab[i], dil[i], add[i], round(float(f.at[i, "cond__extractant_concentration_M"]), 3),
                       str(round(float(f.at[i, "cond__temperature_C"]), 0))) for i in range(len(f))])
    keep = np.array([(ids[i] not in v2drop) and np.isfinite(xacid[i]) for i in range(len(f))])
    grp_slopes = []                                       # (pub, acid, pair index, slope)
    for kk, idx in gkey[keep].groupby(gkey[keep]).groups.items():
        idx = np.array(list(idx))
        if len(idx) < 2 or np.nanmax(xacid[idx]) - np.nanmin(xacid[idx]) < 0.1:
            continue
        for q, (a, b) in enumerate(PAIRS):
            s = ols_slope(xacid[idx], Y[idx, a] - Y[idx, b])
            if np.isfinite(s):
                grp_slopes.append((kk[0], kk[2], q, s))
    GS = pd.DataFrame(grp_slopes, columns=["pub", "acid", "q", "s"])

    def prior_profile(tpub: str, tacid: str) -> tuple[np.ndarray, str]:
        g = GS[GS.pub != tpub]
        same = g[g.acid == tacid]
        use, lab = (same, "same_acid") if same.pub.nunique() >= 3 else (g, "all_acids")
        per_pub = use.groupby(["q", "pub"]).s.mean().groupby("q").mean()
        s = np.array([per_pub.get(q, 0.0) for q in range(len(PAIRS))])
        gm = np.zeros(N_LN)
        for q, (a, b) in enumerate(PAIRS):
            gm[b] = gm[a] - s[q]
        return gm, lab

    refs = {d: pd.read_parquet(paths.PRED / f"REF_{d}.parquet") for d in DESIGNS}
    c3 = sorted(set(refs["BPG"].cell_id))
    c3set = set(c3)

    # ---------- pools ---------------------------------------------------------------------------------------
    pools, log_t = {}, {}
    for t in c3:
        i = pos[t]
        met = [m for m in range(N_LN) if np.isfinite(Y[i, m])]
        cover = np.all(np.isfinite(Y[:, met]), axis=1)
        cover[i] = False
        same = np.where((pub == pub[i]) & (ext == ext[i]) & cover)[0]
        clean, fams, AS, CS = [], [], [], []
        for j in same:
            dfm = diff_fams(i, j)
            ok = (t not in v2drop) and (ids[j] not in v2drop) and bool(dfm - FAMS_TRIVIAL)
            if not ok:
                continue
            clean.append(int(j))
            fams.append(dfm)
            core = dfm - FAMS_TRIVIAL
            if core == {"acid_conc"} and np.isfinite(xacid[j]) and np.isfinite(xacid[i]):
                AS.append(int(j))
            if core <= {"acid_conc", "extr_conc", "temperature"}:
                CS.append(int(j))
        # CS: coordinates that vary among CS + target; require finite values there
        cs_cols, cs_ok = [], []
        if CS:
            Zs = Zall[CS + [i]]
            for c in range(3):
                col = Zs[:, c]
                if np.all(np.isfinite(col)) and np.ptp(col) > 1e-6:
                    cs_cols.append(c)
                elif not np.all(np.isfinite(col)) and np.nanmax(col) - np.nanmin(col) > 1e-6:
                    cs_cols.append(c)
            cs_ok = [j for j in CS if np.all(np.isfinite(Zall[j, cs_cols])) and np.all(np.isfinite(Zall[i, cs_cols]))]
        gm, glab = prior_profile(pub[i], acid_lab[i])
        xa = xacid[AS] if AS else np.array([])
        pools[t] = {"clean": clean, "cls": change_class(fams) if clean else "", "AS": AS, "CS": cs_ok, "cs_cols": cs_cols,
                    "gm": gm, "glab": glab}
        log_t[t] = {"n_clean": len(clean), "n_AS": len(AS), "nx_AS": int(len(np.unique(np.round(xa, 3)))) if AS else 0,
                    "n_CS": len(cs_ok), "cs_cols": cs_cols, "cls": pools[t]["cls"], "prior": glab}

    # ---------- templates (fixed per target; read correction per fold) ----------------------------------------
    tmpl = {}
    for t, P in pools.items():
        i = pos[t]
        d = {}
        if P["clean"]:
            d["SIB"] = Y[P["clean"]].mean(0)
        AS = P["AS"]
        if AS:
            Ys = Y[AS]
            xa = xacid[AS]
            d["ACOPY"] = Ys.mean(0)
            reg_ok = len(np.unique(np.round(xa, 3))) >= 2 and np.ptp(xa) >= 0.1
            for nm, lam in (("AREG", 1.0), ("AREGH", 0.5)):
                T_ = fit_template(Ys, xa[:, None], np.array([xacid[i]]), lam) if reg_ok else None
                d[nm] = T_ if T_ is not None else d["ACOPY"]
            d["AINT"] = interp_template(Ys, xa, xacid[i], extrap=False)
            d["AINTX"] = interp_template(Ys, xa, xacid[i], extrap=True) if reg_ok else d["AINT"]
            d["TILT"] = d["ACOPY"] + P["gm"] * (xacid[i] - xa.mean())
            d["_reg_ok"] = reg_ok
            d["_extrap"] = bool(xacid[i] < xa.min() - 1e-9 or xacid[i] > xa.max() + 1e-9)
            d["_dx_near"] = float(np.min(np.abs(xa - xacid[i])))
            d["_dx_mean"] = float(xacid[i] - xa.mean())
        CS = P["CS"]
        if CS:
            Ys = Y[CS]
            d["CCOPY"] = Ys.mean(0)
            cols = P["cs_cols"]
            ok = False
            if cols and len(CS) >= len(cols) + 1:
                Zs = Zall[CS][:, cols]
                T1 = fit_template(Ys, Zs, Zall[i, cols], 1.0)
                if T1 is not None:
                    ok = True
                    d["CREG"] = T1
                    d["CREGH"] = fit_template(Ys, Zs, Zall[i, cols], 0.5)
            if not ok:
                d["CREG"] = d["CREGH"] = d["CCOPY"]
            d["_creg_ok"] = ok
        tmpl[t] = d

    # ---------- rows ----------------------------------------------------------------------------------------
    rows, checks = [], {"b1_maxabs": 0.0, "y_maxabs": 0.0, "readset_mismatch": 0, "rows": 0}
    ARMS_A = ("ACOPY", "AREG", "AREGH", "AINT", "AINTX", "TILT")
    ARMS_C = ("CCOPY", "CREG", "CREGH")
    for design in DESIGNS:
        par = refs[design]
        fitted = json.loads((paths.RES / "REF" / f"fitted_{design}.json").read_text())
        offres = {(r.split_seed, r.cell_id, r.A, r.B): r.y - r.B1
                  for r in par[["split_seed", "cell_id", "A", "B", "y", "B1"]].itertuples(index=False)}
        for seed in paths.DEV_SEEDS:
            for fold in range(5):
                rec = C.load(design, seed, fold)
                beta = float(fitted[f"{seed}_{fold}"]["ref:B1"]["coef"][0])
                u = np.zeros(N_LN)
                u[GD] = beta
                prow = par[(par.split_seed == seed) & (par.fold == fold)]
                pc = {c: g for c, g in prow.groupby("cell_id")}
                for cell in rec["outer"]["cells"]:
                    t = cell["cell_id"]
                    if t not in pc or not pools[t]["clean"]:
                        continue
                    i = pos[t]
                    yi = Y[i]
                    metals = cell["metals"]
                    seq = cell["seqs"]["DOPT_Q"]
                    P, Tm = pools[t], tmpl[t]
                    lev = {}                                # (arm, k) -> levels
                    reads = {}
                    for k in (0, 1, 2, 3):
                        rd = sorted({m for p in seq[:k] for m in p}) if k else []
                        reads[k] = rd
                        ym = np.full(N_LN, np.nan)
                        ym[rd] = yi[rd]
                        if k:
                            lev[("B1", k)] = interp.pwl_levels(ym - u, metals, "R") + u
                            lev[("B0", k)] = interp.pwl_levels(ym, metals, "R")
                        for nm, T_ in Tm.items():
                            if nm.startswith("_"):
                                continue
                            if k == 0:
                                lev[(nm, 0)] = T_
                            else:
                                res = np.full(N_LN, np.nan)
                                res[rd] = yi[rd] - T_[rd]
                                lev[(nm, k)] = T_ + interp.pwl_levels(res, metals, "R")
                    rs = ",".join(LANTHANIDES[m] for m in reads[3])
                    if rs != str(pc[t].read_set.iloc[0]):
                        checks["readset_mismatch"] += 1
                    n_cl, n_a, n_c = len(P["clean"]), len(P["AS"]), len(P["CS"])
                    for (A, B), pr in pc[t].set_index(["A", "B"]).iterrows():
                        ia, ib = LANTHANIDES.index(A), LANTHANIDES.index(B)
                        y = yi[ia] - yi[ib]
                        checks["y_maxabs"] = max(checks["y_maxabs"], abs(y - pr.y))
                        sf = {key: v[ia] - v[ib] for key, v in lev.items()}
                        checks["b1_maxabs"] = max(checks["b1_maxabs"], abs(sf[("B1", 3)] - pr.B1))
                        r = {"design": design, "split_seed": seed, "fold": fold, "cell_id": t, "publication": pub[i],
                             "extractant": ext[i], "chemotype": chem[i], "A": A, "B": B, "pair": f"{A}/{B}", "read_set": rs,
                             "y": y, "G14Z": pr["G14@ZERO"], "n_clean": n_cl, "n_AS": n_a, "n_CS": n_c, "cls": P["cls"],
                             "c5": pub[i] == C5, "t_v2drop": t in v2drop, "prior": P["glab"], "acid": acid_lab[i],
                             "x_t": xacid[i], "a_read": bool(pr.a_read), "b_read": bool(pr.b_read)}
                        for k in (1, 2, 3):
                            r[f"B1_{k}"] = sf[("B1", k)]
                        r["B0_3"] = sf[("B0", 3)]
                        w = n_cl / (n_cl + K_SH)
                        r["COPY"] = sf[("SIB", 0)]
                        for k in (1, 2, 3):
                            r[f"SIB_{k}"] = sf[("SIB", k)]
                            r[f"SH_{k}"] = w * sf[("SIB", k)] + (1 - w) * sf[("B1", k)]
                        rr = [offres[(seed, c, A, B)] for c in [ids[j] for j in P["clean"] if ids[j] in c3set]
                              if (seed, c, A, B) in offres]
                        r["OFF"] = pr.B1 + (sum(rr) / (len(rr) + K_SH) if rr else 0.0)
                        if n_a:
                            wa = n_a / (n_a + K_SH)
                            r["reg_ok"] = Tm["_reg_ok"]
                            r["extrap"] = Tm["_extrap"]
                            r["dx_near"] = Tm["_dx_near"]
                            r["dx_mean"] = Tm["_dx_mean"]
                            for nm in ARMS_A:
                                r[f"{nm}_0"] = sf[(nm, 0)]
                                for k in (1, 2, 3):
                                    r[f"{nm}_{k}"] = sf[(nm, k)]
                                    r[f"{nm}SH_{k}"] = wa * sf[(nm, k)] + (1 - wa) * sf[("B1", k)]
                        if n_c:
                            wc = n_c / (n_c + K_SH)
                            r["creg_ok"] = Tm["_creg_ok"]
                            r["cs_cols"] = ",".join(("acid", "L", "T")[c] for c in P["cs_cols"])
                            for nm in ARMS_C:
                                r[f"{nm}_0"] = sf[(nm, 0)]
                                for k in (1, 2, 3):
                                    r[f"{nm}_{k}"] = sf[(nm, k)]
                                    r[f"{nm}SH_{k}"] = wc * sf[(nm, k)] + (1 - wc) * sf[("B1", k)]
                        rows.append(r)
                        checks["rows"] += 1
            print(f"{design} done: {checks}", flush=True)
    T = pd.DataFrame(rows)
    T.to_parquet(OUT / "shift_rows.parquet", index=False)
    meta = {"checks": checks, "targets": log_t,
            "prior_pubs": int(GS.pub.nunique()), "prior_groups_rows": int(len(GS))}
    (OUT / "shift_build_log.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    print(json.dumps(checks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
