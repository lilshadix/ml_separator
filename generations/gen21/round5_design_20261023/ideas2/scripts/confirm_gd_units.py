#!/usr/bin/env python
"""confirm theme, check 1 (EXPLORATORY): unit-level Gd-dip statistics and a resampling power replay for a frozen
Gd-dip transport test on ~5-13 new laboratory x chemotype units.

EXPLORATORY.  Every number comes from the V1 bench cells that rounds 1-4 have already seen.  Nothing here is a test
of anything; it plans G for NEW data.

Cell statistic (model-free, label-based, computed only on a unit's own fully measured series):
  c_raw  = logD(Gd) - PWL_R(Eu, Tb) at Gd      (radius coordinate -r(CN8), r3.interp.X_R).  This is exactly the
           quantity B1's Gd potential beta replaces when Gd is hidden and Eu, Tb are read: B1 error at Gd = c_raw - beta.
  c_quad = logD(Gd) - least-squares quadratic in X_R through Sm, Eu, Tb, Dy (curvature-adjusted; needs all four).
Unit = publication x chemotype (proxy for laboratory x chemotype), also publication and cluster_cpg.
Unit value = equal-weight mean of its cells' c (all cells, or m drawn cells to mimic small new units).

Replay: G units drawn with replacement from a pool (all / non-DGA donor / DGA donor / non-DGA without the two
provenance-pending sources / location-shifted chemistry worlds).  Tests, one-sided alpha 0.025:
  GD0    H1 mean c < 0                      (a dip exists in the new chemistry)             t and exact sign-flip
  GDHALF H1 mean c < beta/2                 (B1's Gd term lowers the average unit's Gd-pair error)
  GDDEEP H1 mean c < beta                   (B1 under-corrects: the dip is deeper than B1's amplitude)
  GDTOST |mean c - beta| < |beta|/2 by TOST (B1's amplitude transports within +-50 %)
Sizes are checked in worlds re-centred exactly on each null boundary.

Also: link between a cell's c_raw and its own B1 - B0 SSE gain on the REF rows (both designs).
Writes only to ideas2/scripts/confirm_out/.
    .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/confirm_gd_units.py
"""
from __future__ import annotations

import itertools
import json
import sys
import time
import zlib
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve()
OUT = HERE.parent / "confirm_out"
R4DIR = HERE.parents[3] / "round4_20261022"
sys.path.insert(0, str(R4DIR))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as SS  # noqa: E402

from r4 import paths  # noqa: E402
from r4 import metrics as M4  # noqa: E402
from r4 import donor as DN  # noqa: E402
from r3 import interp  # noqa: E402
from r3 import datav2 as V2  # noqa: E402
from gen20k import data as D  # noqa: E402

LN = list(D.LANTHANIDES)
iSm, iEu, iGd, iTb, iDy = (LN.index(m) for m in ("Sm", "Eu", "Gd", "Tb", "Dy"))
XR = interp.X_R
C5, DMD = "pub_6a0ef8528c", "pub_8415538291"
BETAS = {"beta_fold_mean": -0.0923, "hist_d7": -0.072, "full_info": -0.118}
BETA = BETAS["beta_fold_mean"]
GS = (4, 5, 6, 8, 10, 13, 16, 20)
R = 4000


def chords(Y: np.ndarray):
    """c_raw (needs Eu, Gd, Tb) and c_quad (needs Sm, Eu, Gd, Tb, Dy) per cell."""
    n = len(Y)
    c_raw = np.full(n, np.nan)
    c_quad = np.full(n, np.nan)
    t = (XR[iGd] - XR[iEu]) / (XR[iTb] - XR[iEu])
    ok = np.isfinite(Y[:, [iEu, iGd, iTb]]).all(1)
    c_raw[ok] = Y[ok, iGd] - ((1 - t) * Y[ok, iEu] + t * Y[ok, iTb])
    ok4 = np.isfinite(Y[:, [iSm, iEu, iGd, iTb, iDy]]).all(1)
    idx = [iSm, iEu, iTb, iDy]
    X = np.vstack([np.ones(4), XR[idx] - XR[iGd], (XR[idx] - XR[iGd]) ** 2]).T
    H = np.linalg.pinv(X)            # coef = H @ y ; value at Gd = coef[0]
    c_quad[ok4] = Y[ok4, iGd] - (Y[ok4][:, idx] @ H[0])
    return c_raw, c_quad, float(t)


def unit_table(cells: pd.DataFrame, col: str, unit: str) -> pd.DataFrame:
    g = cells.dropna(subset=[col]).groupby(unit)
    u = g[col].agg(["mean", "count", "std"]).rename(columns={"mean": "c", "count": "n_cells", "std": "sd_within"})
    u["dga_share"] = g["dga"].mean()
    u["pubs"] = g["publication_id"].nunique()
    return u


def tests(Xu: np.ndarray, beta: float, E: np.ndarray | None):
    """Xu (R, G) unit means.  Returns dict of rejection booleans (one-sided 0.025)."""
    G = Xu.shape[1]
    m = Xu.mean(1)
    sd = Xu.std(1, ddof=1)
    se = sd / np.sqrt(G)
    with np.errstate(divide="ignore", invalid="ignore"):
        tcrit = SS.t.ppf(0.975, G - 1)
        out = {"GD0_t": (m / se) < -tcrit,
               "GDHALF_t": ((m - beta / 2) / se) < -tcrit,
               "GDDEEP_t": ((m - beta) / se) < -tcrit,
               "GDTOST_t": (((m - 1.5 * beta) / se) > tcrit) & (((m - 0.5 * beta) / se) < -tcrit)}
    if E is not None:
        for name, c0 in (("GD0_sf", 0.0), ("GDHALF_sf", beta / 2), ("GDDEEP_sf", beta)):
            Z = c0 - Xu                                   # positive = in H1 direction
            T = Z.sum(1)
            Dm = Z @ E.T
            p = ((Dm >= T[:, None] - 1e-12).sum(1)) / E.shape[0]
            out[name] = p <= 0.025 + 1e-12
        S = np.sign(beta / 2 - Xu)
        w = (S > 0).sum(1)
        nn = (S != 0).sum(1)
        out["GDHALF_sign"] = SS.binom.sf(w - 1, nn, 0.5) <= 0.025
    return out


def flip_matrix(G, rng):
    if G <= 13:
        return np.array(list(itertools.product((1.0, -1.0), repeat=G)))
    return rng.choice((1.0, -1.0), size=(4999, G))


def replay(pool_cells: dict, G: int, m_new, rng, shift: float = 0.0, center: float | None = None, noise: float = 0.0):
    """pool_cells: list of arrays (cell values per unit).  m_new: None = all cells, else draw m cells w/ replacement.
    shift adds a constant; center re-centres the unit-mean distribution (equal-weight over units) to that value."""
    units = pool_cells["units"]
    U = len(units)
    ui = rng.integers(U, size=(R, G))
    if m_new is None:
        Xu = np.array([u.mean() for u in units])[ui]
    else:
        nmax = max(len(u) for u in units)
        V = np.full((U, nmax), np.nan)
        for i, u in enumerate(units):
            V[i, :len(u)] = u
        cnt = np.array([len(u) for u in units])
        idx = np.floor(rng.random((R, G, m_new)) * cnt[ui][..., None]).astype(int)
        Xu = V[ui[..., None], idx].mean(-1)
    if noise > 0:
        Xu = Xu + rng.normal(0, noise, size=Xu.shape)
    Xu = Xu + shift
    if center is not None:
        Xu = Xu - pool_cells["mu"] - shift + center
    return Xu


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True).copy()
    Y = np.asarray(bench.Y, float)
    c_raw, c_quad, tg = chords(Y)
    f["c_raw"], f["c_quad"] = c_raw, c_quad
    f["dga"] = f.extractant.astype(str).map(DN.is_dga)
    ref = pd.read_parquet(paths.PRED / "REF_BPG.parquet")
    c3 = set(ref.cell_id)
    f["c3"] = f.cell_id.isin(c3)
    v2drop = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    f["v2drop"] = f.cell_id.isin(v2drop)
    f["pubchem"] = f.publication_id.astype(str) + "|" + f.chemotype.astype(str)
    # cpg clusters for C3 cells
    cl = M4.add_clusters(ref[["cell_id", "publication", "chemotype"]].drop_duplicates("cell_id"), None)
    f = f.merge(cl[["cell_id", "cluster_cpg"]], on="cell_id", how="left")
    res = {"t_gd_between_eu_tb": tg, "beta_used": BETA}

    # ---------------- descriptive corpus numbers (EXPLORATORY) -----------------------------------------------
    desc = []
    pops = {"C3": f[f.c3], "C3_clean": f[f.c3 & ~f.v2drop], "ALL": f, "ALL_clean": f[~f.v2drop]}
    for pn, P in pops.items():
        for col in ("c_raw", "c_quad"):
            for unit in ("pubchem", "publication_id", "cluster_cpg"):
                if unit == "cluster_cpg" and not pn.startswith("C3"):
                    continue
                for grp, Q in (("all", P), ("nonDGA", P[~P.dga]), ("DGA", P[P.dga]),
                               ("nonDGA_noC5DMD", P[~P.dga & ~P.publication_id.isin([C5, DMD])]),
                               ("all_noC5", P[P.publication_id != C5])):
                    u = unit_table(Q, col, unit)
                    if len(u) < 2:
                        continue
                    desc.append({"pop": pn, "stat": col, "unit": unit, "group": grp, "cells": int(u.n_cells.sum()),
                                 "units": len(u), "cell_mean": float(Q[col].mean()), "cell_median": float(Q[col].median()),
                                 "unit_mean": float(u.c.mean()), "unit_median": float(u.c.median()),
                                 "unit_sd": float(u.c.std(ddof=1)),
                                 "se_G13": float(u.c.std(ddof=1) / np.sqrt(13)),
                                 "within_sd_pooled": float(np.sqrt(np.nanmean(u.sd_within ** 2))),
                                 "share_units_below_half_beta": float((u.c < BETA / 2).mean()),
                                 "share_units_below_beta": float((u.c < BETA).mean()),
                                 "share_units_below_0": float((u.c < 0).mean())})
    desc = pd.DataFrame(desc)
    desc.to_csv(OUT / "gd_units_desc.csv", index=False)
    show = desc[(desc.unit == "pubchem") | ((desc.unit == "cluster_cpg") & (desc.group == "all"))]
    print(show.round(4).to_string(), flush=True)

    # per-unit table for the record (C3, c_raw, pubchem)
    ut = unit_table(f[f.c3], "c_raw", "pubchem").sort_values("c")
    ut.to_csv(OUT / "gd_units_C3_pubchem.csv")

    # ---------------- link: cell chord vs cell's own B1 - B0 SSE gain (REF rows) ------------------------------
    link = {}
    for design in ("BPG", "BP"):
        t = pd.read_parquet(paths.PRED / f"REF_{design}.parquet")
        t = t[(t.k_max >= 3) & (t.k == 3)]
        t = t.assign(g=(t.B0 - t.y) ** 2 - (t.B1 - t.y) ** 2)
        cg = t.groupby("cell_id").g.sum() / t.split_seed.nunique()
        gp = t[t.A.isin(["Eu", "Gd"]) & t.B.isin(["Gd", "Tb"])].groupby("cell_id").g.sum() / t.split_seed.nunique()
        d = f.set_index("cell_id").loc[cg.index, ["c_raw", "publication_id", "pubchem", "dga"]].assign(gain=cg, gain_gdpairs=gp)
        d = d.dropna(subset=["c_raw"])
        d["pred_win"] = d.c_raw < BETA / 2
        d["win"] = d.gain > 0
        link[design] = {"cells": int(len(d)), "spearman_c_gain": float(SS.spearmanr(d.c_raw, d.gain)[0]),
                        "share_gain_on_gd_pairs": float(d.gain_gdpairs.sum() / d.gain.sum()),
                        "accuracy_rule_c_lt_halfbeta_predicts_cell_win": float((d.pred_win == d.win).mean()),
                        "win_rate_cells": float(d.win.mean()),
                        "win_rate_if_c_lt_halfbeta": float(d.win[d.pred_win].mean()),
                        "win_rate_if_c_ge_halfbeta": float(d.win[~d.pred_win].mean())}
        # unit level (pubchem): unit mean c vs unit gain
        uu = d.groupby("pubchem").agg(c=("c_raw", "mean"), gain=("gain", "sum"))
        link[design]["unit_spearman"] = float(SS.spearmanr(uu.c, uu.gain)[0])
        link[design]["unit_agree_rule"] = float(((uu.c < BETA / 2) == (uu.gain > 0)).mean())
        link[design]["units"] = int(len(uu))
    res["link"] = link
    print(json.dumps(link, indent=1), flush=True)

    # ---------------- replay -----------------------------------------------------------------------------------
    rows = []
    base = f[f.c3 & ~f.v2drop].dropna(subset=["c_raw"])
    big = f[~f.v2drop].dropna(subset=["c_raw"])
    poolspec = {
        "C3_all": base, "C3_nonDGA": base[~base.dga], "C3_DGA": base[base.dga],
        "C3_nonDGA_noC5DMD": base[~base.dga & ~base.publication_id.isin([C5, DMD])],
        "ALL_all": big, "ALL_nonDGA": big[~big.dga],
    }
    pools = {}
    for name, P in poolspec.items():
        units = [g.c_raw.to_numpy() for _, g in P.groupby("pubchem")]
        mu = float(np.mean([u.mean() for u in units]))
        pools[name] = {"units": units, "mu": mu, "n_units": len(units),
                       "sd": float(np.std([u.mean() for u in units], ddof=1))}
    res["pools"] = {k: {"n_units": v["n_units"], "mean_unit_c": v["mu"], "sd_unit_c": v["sd"]} for k, v in pools.items()}
    print(json.dumps(res["pools"], indent=1), flush=True)

    # worlds: (pool, label, m_new, shift, center, noise)
    worlds = []
    for pn in ("C3_all", "C3_nonDGA", "C3_DGA", "C3_nonDGA_noC5DMD", "ALL_nonDGA"):
        for m_new in (None, 1, 2):
            worlds.append((pn, "EMP", m_new, 0.0, None, 0.0))
    # chemistry worlds on the non-DGA shape (spread kept, mean moved)
    for pn in ("C3_nonDGA_noC5DMD", "C3_all"):
        for lab, cen in (("CEN_HDEHPlike_-0.26", -0.26), ("CEN_-0.175", -0.175), ("CEN_DGAlike_-0.064", -0.064),
                         ("CEN_zero", 0.0)):
            worlds.append((pn, lab, 2, 0.0, cen, 0.0))
    # null-boundary worlds for size (re-centred on each H0 boundary), m_new = 2
    for pn in ("C3_all", "C3_nonDGA"):
        for lab, cen in (("NULL_GD0", 0.0), ("NULL_GDHALF", BETA / 2), ("NULL_GDDEEP", BETA),
                         ("NULL_TOST_lo", 1.5 * BETA), ("NULL_TOST_hi", 0.5 * BETA)):
            worlds.append((pn, lab, 2, 0.0, cen, 0.0))
    for (pn, lab, m_new, shift, cen, noise) in worlds:
        for G in GS:
            rng = np.random.default_rng(20261023 + zlib.crc32(f"{pn}|{lab}|{m_new}|{G}".encode()))
            Xu = replay(pools[pn], G, m_new, rng, shift, cen, noise)
            E = flip_matrix(G, rng)
            rej = tests(Xu, BETA, E)
            rec = {"pool": pn, "world": lab, "m_new": "all" if m_new is None else m_new, "G": G, "reps": R,
                   "mean_unit_c": float(Xu.mean()), "sd_unit_c": float(Xu.std(1, ddof=1).mean())}
            for k, v in rej.items():
                rec[k] = float(np.mean(v))
            rows.append(rec)
        print(f"{time.time() - t0:6.1f}s {pn} {lab} m={m_new}", flush=True)
    g = pd.DataFrame(rows)
    g.to_csv(OUT / "gd_replay.csv", index=False)
    cols = ["pool", "world", "m_new", "G", "mean_unit_c", "GD0_sf", "GDHALF_t", "GDHALF_sf", "GDHALF_sign", "GDDEEP_t",
            "GDDEEP_sf", "GDTOST_t"]
    print(g[cols].round(3).to_string(), flush=True)
    res["seconds"] = time.time() - t0
    (OUT / "gd_units_summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
