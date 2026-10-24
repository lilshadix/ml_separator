#!/usr/bin/env python
"""confirm theme, check 2 (EXPLORATORY): (a) P2-machinery replay of the equal-weight B1 - B0 breadth test at the
G a ~13-laboratory supply can give (G 5-20), by donor stratum and unit type; (b) within-unit window-flag replay;
(c) size of the Gd-chord tests after a frozen winsorisation.

EXPLORATORY: all numbers are from the 229 already-seen C3 cells x 5 dev seeds.  Imports P2's functions read-only
(run with -B so no bytecode is written next to them).  Writes only ideas2/scripts/confirm_out/.
    .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/confirm_breadth_window.py
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
P2DIR = HERE.parents[2] / "probes" / "scripts"
R4DIR = HERE.parents[3] / "round4_20261022"
sys.path.insert(0, str(R4DIR))
sys.path.insert(0, str(P2DIR))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as SS  # noqa: E402

import P2_power_replay as P2  # noqa: E402
from r4 import paths  # noqa: E402
from r4 import donor as DN  # noqa: E402
from r3 import interp  # noqa: E402
from r3 import corrector as CO  # noqa: E402
from gen20k import data as D  # noqa: E402

C5, DMD = "pub_6a0ef8528c", "pub_8415538291"
GS = (5, 6, 8, 10, 13, 16, 20)
TESTS = ("SIGN", "SF_LOG", "SF_POOL", "T_LOG", "WCR_POOL")
LN = list(D.LANTHANIDES)


def seed(*p):
    return 20261023 + zlib.crc32("|".join(map(str, p)).encode())


# ------------------------------------------------------------------------------------------------------------
def breadth(design: str, t: pd.DataFrame, rows: list, t0: float):
    t = DN.add_donor(t)
    for unit in ("publication", "pubchem"):
        share = t.groupby(unit).dga_donor.mean()
        us = P2.unit_stats(t, unit, "B1", "B0")
        nod = share.loc[us.index] <= 0.5
        pub_of = t.groupby(unit).publication.first().loc[us.index]
        strata = {"all": np.ones(len(us), bool), "nonDGA": nod.to_numpy(), "DGA": (~nod).to_numpy(),
                  "nonDGA_noC5DMD": (nod & ~pub_of.isin([C5, DMD])).to_numpy(),
                  "all_noC5DMD": (~pub_of.isin([C5, DMD])).to_numpy()}
        for sname, keep in strata.items():
            p = P2.make_pool(us[keep])
            dS = p["Sr"] - p["Sc"]
            rows.append({"design": design, "unit": unit, "stratum": sname, "world": "DEV", "tau": np.nan, "G": len(dS),
                         "units": int(keep.sum()), "wins": int((dS > 0).sum()), "losses": int((dS < 0).sum()),
                         "meanL": float(np.log(p["Sr"] / p["Sc"]).mean()),
                         "pooled_rel": float(1 - p["Sc"].sum() / p["Sr"].sum())})
            for world in ("EMP", "SWAP"):
                for tau in (0.0, 0.065):
                    for G in GS:
                        R = 2000
                        rng = np.random.default_rng(seed("confirm", design, unit, sname, world, tau, G))
                        n, Sy, Sy2, Sr, Sc = P2.draw(p, G, R, rng, world, tau)
                        P, rej, extra = P2.run_tests(n, Sy, Sy2, Sr, Sc, rng, do_sealed=False)
                        rel = 1 - Sc.sum(1) / Sr.sum(1)
                        rec = {"design": design, "unit": unit, "stratum": sname, "world": world, "tau": tau, "G": G,
                               "units": int(keep.sum())}
                        for k in TESTS:
                            rec[k] = float(rej[k][0.025].mean())
                        # joint: equal-weight SF_LOG plus a pooled magnitude guard (pooled rel. SSE >= 2 %), M35-style
                        rec["SF_LOG_and_pooled2pct"] = float((rej["SF_LOG"][0.025] & (rel >= 0.02)).mean())
                        rec["SF_LOG_and_pooledpos"] = float((rej["SF_LOG"][0.025] & (rel > 0)).mean())
                        rows.append(rec)
            print(f"{time.time() - t0:6.1f}s breadth {design} {unit} {sname}", flush=True)


# ------------------------------------------------------------------------------------------------------------
def b1_levels_rows(design: str, t: pd.DataFrame, bench) -> pd.DataFrame:
    """Adds B1 predicted levels of A and B (k = 3, the row's own read set, the fold's fitted beta)."""
    f = bench.frame.reset_index(drop=True)
    Y = np.asarray(bench.Y, float)
    pos = {c: i for i, c in enumerate(f.cell_id.astype(str))}
    fitted = json.loads((paths.RES / "REF" / f"fitted_{design}.json").read_text())
    levA = np.empty(len(t))
    levB = np.empty(len(t))
    cache = {}
    for j, r in enumerate(t[["split_seed", "fold", "cell_id", "read_set", "A", "B"]].itertuples(index=False)):
        key = (r.split_seed, r.fold, r.cell_id, r.read_set)
        if key not in cache:
            i = pos[r.cell_id]
            yi = Y[i]
            metals = [m for m in range(len(LN)) if np.isfinite(yi[m])]
            beta = float(fitted[f"{r.split_seed}_{r.fold}"]["ref:B1"]["coef"][0])
            u = np.zeros(len(LN))
            u[CO.GD] = beta
            rd = [LN.index(m) for m in r.read_set.split(",")]
            ym = np.full(len(LN), np.nan)
            ym[rd] = yi[rd]
            cache[key] = interp.pwl_levels(ym - u, metals, "R") + u
        lv = cache[key]
        levA[j] = lv[LN.index(r.A)]
        levB[j] = lv[LN.index(r.B)]
    t = t.copy()
    t["levA"], t["levB"] = levA, levB
    chk = float(np.max(np.abs((levA - levB) - t.B1.to_numpy())))
    return t, chk


def window(design: str, t: pd.DataFrame, rows: list, summ: dict, t0: float):
    t = t.assign(inw=(t.levA.between(-2, 2) & t.levB.between(-2, 2)), res=t.y - t.B1)
    t["tl"] = (t.res.abs() > 0.5).astype(float)
    t["se"] = t.res ** 2
    S = t.split_seed.nunique()
    for scope, tt in (("all", t), ("noC5", t[t.publication != C5])):
        out, inn = tt[~tt.inw], tt[tt.inw]
        summ[f"{design}_{scope}"] = {
            "coverage": float(tt.inw.mean()), "tail_in": float(inn.tl.mean()), "tail_out": float(out.tl.mean()),
            "tail_ratio": float(out.tl.mean() / max(inn.tl.mean(), 1e-12)),
            "mse_in": float(inn.se.mean()), "mse_out": float(out.se.mean())}
        for unit in ("publication", "pubchem"):
            g = tt.groupby([unit, "inw"]).agg(n=("se", "size"), sse=("se", "sum"), tl=("tl", "sum")).unstack("inw")
            g = g.fillna(0) / S
            g.columns = [f"{a}_{'in' if b else 'out'}" for a, b in g.columns]
            elig = g[(g.n_in >= 3) & (g.n_out >= 3)].copy()
            elig["L"] = np.log((elig.sse_out / elig.n_out) / (elig.sse_in / elig.n_in))
            elig["dtail"] = elig.tl_out / elig.n_out - elig.tl_in / elig.n_in
            summ[f"{design}_{scope}_{unit}"] = {
                "units_total": int(len(g)), "units_eligible_n_in_out_ge3": int(len(elig)),
                "share_rows_out_in_eligible": float(elig.n_out.sum() / g.n_out.sum()),
                "within_unit_mse_ratio_wins": int((elig.L > 0).sum()), "meanL": float(elig.L.mean()),
                "medianL": float(elig.L.median()), "sdL": float(elig.L.std(ddof=1)),
                "dtail_pos": int((elig.dtail > 0).sum()), "dtail_neg": int((elig.dtail < 0).sum()),
                "pooled_within_eligible_tail_ratio": float((elig.tl_out.sum() / elig.n_out.sum()) /
                                                           max(elig.tl_in.sum() / elig.n_in.sum(), 1e-12))}
            Lv = elig.L.to_numpy()
            Dv = elig.dtail.to_numpy()
            for G in (4, 5, 6, 8, 10, 13):
                R = 4000
                rng = np.random.default_rng(seed("win", design, scope, unit, G))
                ui = rng.integers(len(Lv), size=(R, G))
                X = Lv[ui]
                E = (np.array(list(itertools.product((1.0, -1.0), repeat=G))) if G <= 13
                     else rng.choice((1.0, -1.0), size=(4999, G)))
                T = X.sum(1)
                p = ((X @ E.T) >= T[:, None] - 1e-12).sum(1) / E.shape[0]
                Xc = X - Lv.mean()                       # null: re-centred (skew kept)
                Tc = Xc.sum(1)
                pc = ((Xc @ E.T) >= Tc[:, None] - 1e-12).sum(1) / E.shape[0]
                w = (X > 0).sum(1)
                ps = SS.binom.sf(w - 1, G, 0.5)
                rows.append({"design": design, "scope": scope, "unit": unit, "G": G, "eligible_units": len(Lv),
                             "WIN_SF_LOG_power": float((p <= 0.025).mean()),
                             "WIN_SF_LOG_size_recentred": float((pc <= 0.025).mean()),
                             "WIN_SIGN_power": float((ps <= 0.025).mean()),
                             "mean_draw_L": float(X.mean())})
    print(f"{time.time() - t0:6.1f}s window {design}", flush=True)


# ------------------------------------------------------------------------------------------------------------
def gd_winsor(rows: list):
    """Size and power of the GDHALF t / sign-flip tests after a frozen winsorisation of unit chords."""
    f = pd.read_csv(OUT / "gd_units_C3_pubchem.csv")
    dga = f.dga_share > 0.5
    BETA = -0.0923
    for clip in (None, (-0.40, 0.20), (-0.35, 0.15)):
        for pool, v in (("all", f.c.to_numpy()), ("nonDGA", f.c[~dga].to_numpy()), ("DGA", f.c[dga].to_numpy())):
            x = v if clip is None else np.clip(v, *clip)
            for world, cen in (("EMP", None), ("NULL_GD0", 0.0), ("NULL_GDHALF", BETA / 2)):
                for G in (6, 8, 10, 13, 16):
                    R = 4000
                    rng = np.random.default_rng(seed("gdw", clip, pool, world, G))
                    X = x[rng.integers(len(x), size=(R, G))]
                    if cen is not None:
                        X = X - x.mean() + cen
                    se = X.std(1, ddof=1) / np.sqrt(G)
                    tc = SS.t.ppf(0.975, G - 1)
                    m = X.mean(1)
                    E = np.array(list(itertools.product((1.0, -1.0), repeat=G))) if G <= 13 else \
                        rng.choice((1.0, -1.0), size=(4999, G))
                    rec = {"clip": str(clip), "pool": pool, "world": world, "G": G, "mean": float(m.mean())}
                    for nm, c0 in (("GD0", 0.0), ("GDHALF", BETA / 2)):
                        rec[nm + "_t"] = float((((m - c0) / se) < -tc).mean())
                        Z = c0 - X
                        p = ((Z @ E.T) >= Z.sum(1)[:, None] - 1e-12).sum(1) / E.shape[0]
                        rec[nm + "_sf"] = float((p <= 0.025).mean())
                    rows.append(rec)


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    bench = D.load_bench()
    brows, wrows, wsum, grows = [], [], {}, []
    gd_winsor(grows)
    pd.DataFrame(grows).to_csv(OUT / "gd_winsor.csv", index=False)
    print(pd.DataFrame(grows).round(3).to_string(), flush=True)
    for design in ("BPG", "BP"):
        t = P2.load(design)
        t, chk = b1_levels_rows(design, t, bench)
        wsum[f"{design}_levels_check_maxabs"] = chk
        window(design, t, wrows, wsum, t0)
        if design == "BPG":
            breadth(design, t, brows, t0)
    b = pd.DataFrame(brows)
    b.to_csv(OUT / "breadth_replay.csv", index=False)
    w = pd.DataFrame(wrows)
    w.to_csv(OUT / "window_replay.csv", index=False)
    (OUT / "window_summary.json").write_text(json.dumps(wsum, indent=1), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(json.dumps(wsum, indent=1))
    print(w.round(3).to_string())
    print(b.round(3).to_string())
    print("seconds", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
