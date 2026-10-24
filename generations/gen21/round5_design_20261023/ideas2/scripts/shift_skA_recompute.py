#!/usr/bin/env python
"""SKEPTIC A for idea shift-2 (reliability-gated, bracketed slope transfer).  EXPLORATORY: every number is on the
229 already-seen C3 cells and the 5 dev seeds.  Independent re-implementation (does not import shift_build.py /
shift_eb.py; shares only the project's infrastructure: bench loader, V2 lists, fold cache, PWL_R, fitted B1 beta).

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/shift_skA_recompute.py

Writes only ideas2/scripts/shift_skA_out/skA_rows.parquet and skA_targets.csv; everything else is printed.

Arms (template T; k-read version X_k = T + PWL_R[y_S - T_S]; *SH = w X_k + (1-w) B1_k, w = n/(n+1.78)):
  ACOPY  mean of acid-only siblings AS
  AREG   per-metal OLS of log D on log10[acid] over AS, evaluated at the target acid
  AEB    ACOPY + lam_JS (AREG - ACOPY); lam_JS = clip(1 - mean(se2_m)(1-1/M) / mean_m (b_m - bbar)^2, 0, 1),
         dof = n_AS - 2 (lam = 0 when dof < 1)                       [the idea's formula]
  AEBG   AEB with lam = 0 when the target acid lies outside [min, max] of the AS acids  [the frozen arm]
  AEBGa  variant: no (1 - 1/M) factor;  AEBGc  variant: lam = 0 unless dof >= 2
  AINT   per-metal piecewise-linear interpolation in log10[acid] over AS levels, flat beyond
  SIB/SH all CLEAN siblings (P1); B1 fold-fitted Gd amplitude
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as SS  # noqa: E402

from r4 import paths  # noqa: E402
from r3 import datav2 as V2  # noqa: E402
from r3 import interp  # noqa: E402
from r3 import corrector as CO  # noqa: E402
from g21 import cache as C  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import N_LN, LANTHANIDES  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "shift_skA_out"
SHIFT_ROWS = HERE / "shift_out" / "shift_rows.parquet"
KSH = 1.78
C5 = "pub_6a0ef8528c"
TRIV = {"ctime", "mconc"}


def family(c: str) -> str:
    if c == "cond__acid_concentration_M":
        return "aconc"
    for pre, nm in (("cond__acid__", "atype"), ("cond__additive__", "add"), ("cond__diluent__", "dil")):
        if c.startswith(pre):
            return nm
    return {"cond__contact_time_min": "ctime", "cond__extractant_concentration_M": "lconc",
            "cond__metal_concentration_mM": "mconc", "cond__temperature_C": "temp"}[c]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    b = D.load_bench()
    f = b.frame.reset_index(drop=True)
    Y = np.asarray(b.Y, float)
    ids = f.cell_id.astype(str).to_numpy()
    pos = {c: i for i, c in enumerate(ids)}
    drop = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP.keys()) | set(V2.V2_3_CELLS)
    ccols = [c for c in f.columns if c.startswith("cond__")]
    fams = np.array([family(c) for c in ccols])
    Mc = f[ccols].astype(float).to_numpy()
    pub = f.publication_id.astype(str).to_numpy()
    ext = f.extractant.astype(str).to_numpy()
    chem = f.chemotype.astype(str).to_numpy()
    xac = np.log10(f.cond__acid_concentration_M.astype(float).to_numpy())

    def core_diff(i, j) -> set:
        a, c = Mc[i], Mc[j]
        both_nan = np.isnan(a) & np.isnan(c)
        same = both_nan | np.isclose(a, c, equal_nan=False)
        return set(fams[~same]) - TRIV

    refs = {d: pd.read_parquet(paths.PRED / f"REF_{d}.parquet") for d in ("BPG", "BP")}
    c3 = sorted(set(refs["BPG"].cell_id))

    # ------------------------------------------------------------------ sibling sets + templates
    TT, tinfo = {}, []
    for t in c3:
        i = pos[t]
        met = np.where(np.isfinite(Y[i]))[0]
        cov = np.all(np.isfinite(Y[:, met]), axis=1)
        cov[i] = False
        cand = np.where((pub == pub[i]) & (ext == ext[i]) & cov)[0]
        clean, AS = [], []
        for j in cand:
            if t in drop or ids[j] in drop:
                continue
            cd = core_diff(i, j)
            if not cd:
                continue
            clean.append(j)
            if cd == {"aconc"} and np.isfinite(xac[j]) and np.isfinite(xac[i]):
                AS.append(j)
        d = {"clean": clean, "AS": AS}
        if clean:
            d["SIB"] = Y[clean].mean(0)
        info = {"cell": t, "pub": pub[i], "chem": chem[i], "n_clean": len(clean), "n_AS": len(AS), "nmet": len(met),
                "x_t": xac[i]}
        if AS:
            xs = xac[AS]
            Ys = Y[AS]
            d["ACOPY"] = Ys.mean(0)
            reg_ok = len(np.unique(np.round(xs, 3))) >= 2 and np.ptp(xs) >= 0.1
            inside = bool(xs.min() - 1e-9 <= xac[i] <= xs.max() + 1e-9)
            # per-metal OLS (target metals only)
            areg = d["ACOPY"].copy()
            lam = {"main": 0.0, "a": 0.0, "c": 0.0}
            bm = np.full(N_LN, np.nan)
            if reg_ok:
                xb = xs.mean()
                Sxx = ((xs - xb) ** 2).sum()
                for m in met:
                    bm[m] = ((xs - xb) * (Ys[:, m] - Ys[:, m].mean())).sum() / Sxx
                    areg[m] = Ys[:, m].mean() + bm[m] * (xac[i] - xb)
                dof = len(AS) - 2
                if dof >= 1:
                    bb = bm[met]
                    res = Ys[:, met] - Ys[:, met].mean(0) - np.outer(xs - xb, bb)
                    s2 = (res ** 2).sum(0) / dof
                    se2 = s2 / Sxx
                    v = ((bb - bb.mean()) ** 2).mean()
                    if v > 0:
                        lam["main"] = float(np.clip(1 - se2.mean() * (1 - 1 / len(met)) / v, 0, 1))
                        lam["a"] = float(np.clip(1 - se2.mean() / v, 0, 1))
                        lam["c"] = lam["main"] if dof >= 2 else 0.0
                    info.update({"dof": dof, "sd_dev": float(np.sqrt(v)), "se_rms": float(np.sqrt(se2.mean()))})
            d["AREG"] = areg
            d["AEB"] = d["ACOPY"] + lam["main"] * (areg - d["ACOPY"])
            g = 1.0 if inside else 0.0
            d["AEBG"] = d["ACOPY"] + g * lam["main"] * (areg - d["ACOPY"])
            d["AEBGa"] = d["ACOPY"] + g * lam["a"] * (areg - d["ACOPY"])
            d["AEBGc"] = d["ACOPY"] + g * lam["c"] * (areg - d["ACOPY"])
            xr = np.round(xs, 4)
            xu = np.unique(xr)
            Yu = np.array([Ys[xr == v].mean(0) for v in xu])
            d["AINT"] = np.array([np.interp(xac[i], xu, Yu[:, m]) for m in range(N_LN)]) if len(xu) > 1 else Yu[0]
            info.update({"reg_ok": reg_ok, "inside": inside, "lam": lam["main"], "lam_a": lam["a"], "lam_c": lam["c"],
                         "x_min": xs.min(), "x_max": xs.max(), "nx": len(xu)})
        TT[t] = d
        tinfo.append(info)
    TI = pd.DataFrame(tinfo)
    TI.to_csv(OUT / "skA_targets.csv", index=False)

    # ------------------------------------------------------------------ rows
    ARMS = ("ACOPY", "AREG", "AEB", "AEBG", "AEBGa", "AEBGc", "AINT")
    rows, chk = [], {"b1": 0.0, "y": 0.0, "rs": 0}
    for design in ("BPG", "BP"):
        par = refs[design]
        fit = json.loads((paths.RES / "REF" / f"fitted_{design}.json").read_text())
        for seed in paths.DEV_SEEDS:
            for fold in range(5):
                rec = C.load(design, seed, fold)
                u = np.zeros(N_LN)
                u[CO.GD] = float(fit[f"{seed}_{fold}"]["ref:B1"]["coef"][0])
                pr = par[(par.split_seed == seed) & (par.fold == fold)]
                byc = dict(tuple(pr.groupby("cell_id")))
                for cell in rec["outer"]["cells"]:
                    t = cell["cell_id"]
                    if t not in byc or not TT[t]["clean"]:
                        continue
                    i = pos[t]
                    yi = Y[i]
                    metals = cell["metals"]
                    seq = cell["seqs"]["DOPT_Q"]
                    d = TT[t]
                    L = {}
                    for k in (0, 1, 2, 3):
                        rd = sorted({m for p_ in seq[:k] for m in p_})
                        if k:
                            ym = np.full(N_LN, np.nan)
                            ym[rd] = yi[rd] - u[rd]
                            L[("B1", k)] = interp.pwl_levels(ym, metals, "R") + u
                        for nm in ("SIB",) + ARMS:
                            if nm not in d:
                                continue
                            T_ = d[nm]
                            if k == 0:
                                L[(nm, 0)] = T_
                            else:
                                r_ = np.full(N_LN, np.nan)
                                r_[rd] = yi[rd] - T_[rd]
                                L[(nm, k)] = T_ + interp.pwl_levels(r_, metals, "R")
                    rs = ",".join(LANTHANIDES[m] for m in sorted({m for p_ in seq[:3] for m in p_}))
                    chk["rs"] += int(rs != str(byc[t].read_set.iloc[0]))
                    wc = len(d["clean"]) / (len(d["clean"]) + KSH)
                    wa = len(d["AS"]) / (len(d["AS"]) + KSH) if d["AS"] else np.nan
                    for q in byc[t].itertuples(index=False):
                        a_, b_ = LANTHANIDES.index(q.A), LANTHANIDES.index(q.B)
                        y = yi[a_] - yi[b_]
                        chk["y"] = max(chk["y"], abs(y - q.y))
                        sf = {kk: v[a_] - v[b_] for kk, v in L.items()}
                        chk["b1"] = max(chk["b1"], abs(sf[("B1", 3)] - q.B1))
                        r = {"design": design, "split_seed": seed, "fold": fold, "cell_id": t, "publication": pub[i],
                             "chemotype": chem[i], "pair": f"{q.A}/{q.B}", "A": q.A, "B": q.B, "y": y,
                             "c5": pub[i] == C5}
                        for k in (1, 2, 3):
                            r[f"B1_{k}"] = sf[("B1", k)]
                            r[f"SH_{k}"] = wc * sf[("SIB", k)] + (1 - wc) * sf[("B1", k)]
                        r["COPY_0"] = sf[("SIB", 0)]
                        if d["AS"]:
                            for nm in ARMS:
                                for k in (0, 1, 2, 3):
                                    r[f"{nm}_{k}"] = sf[(nm, k)]
                                    if k:
                                        r[f"{nm}SH_{k}"] = wa * sf[(nm, k)] + (1 - wa) * sf[("B1", k)]
                        rows.append(r)
    R = pd.DataFrame(rows)
    R.to_parquet(OUT / "skA_rows.parquet", index=False)
    print("checks:", chk, "rows", len(R))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
