#!/usr/bin/env python
"""P2SK: skeptic's independent recompute of the decision-relevant numbers of probe P2 (M34 power replay).

EXPLORATORY: uses the round-4 OOF parquets of the 229 already-seen C3 cells.  Written from scratch; it imports only
round-4's unit definition (r4.metrics.add_clusters) and none of P2's functions.  Different RNG seeds from P2.

Checks
  A  dev: unit counts, Kish, W/L/T, mean L, exact sign-flip / sign / t p-values (BPG, BP)
  B  M34 kill cell: power at G = 8, publication units, B1 - B0, one-sided 0.025, tau 0 / 0.065, BPG / BP, every test;
     plus a sensitivity with round-4's candidate-only noise (kappa ~ N(0, tau) on the candidate arm only)
  C  G80 evaluated directly at the interpolated G (no log-G interpolation): publication, chemotype, pub x chemotype
  D  size of the jackknife-t legs (df G - 1 and Kish df) in selected SWAP / SYM null cells
    .venv/Scripts/python.exe -B generations/gen21/round5_design_20261023/probes/scripts/P2SK_recompute.py
writes nothing to disk; everything goes to stdout (RESULTS_JSON line at the end)
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
PROBES = HERE.parents[1]
R4 = HERE.parents[3] / "round4_20261022"
sys.path.insert(0, str(R4))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from r4 import metrics as M4  # noqa: E402

OUT = PROBES / "P2_out_SK"
WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])
A = 0.025


def seed(*parts):
    return 991_000 + zlib.crc32(("SK|" + "|".join(map(str, parts))).encode())


def load(design):
    t = pd.read_parquet(R4 / "predictions" / f"REF_{design}.parquet")
    t = t[(t.k_max >= 3) & (t.k == 3)].copy()
    t = M4.add_clusters(t, None)
    t["pubchem"] = t.publication.astype(str) + "#" + t.chemotype.astype(str)
    d = (t.B1 - t.B0).to_numpy()
    e0 = (t.B0 - t.y).to_numpy()
    # scaled candidate: relative SSE reduction vs B0 of exactly 2 % (pooled over seeds) -> independent root find
    f = lambda s: 1 - ((e0 + s * d) ** 2).sum() / (e0 ** 2).sum() - 0.02  # noqa: E731
    lo, hi = 0.0, 0.6
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    t["B1s2"] = t.B0 + 0.5 * (lo + hi) * d
    t.attrs["s2"] = 0.5 * (lo + hi)
    return t


def unit_table(t, unit, cand, ref):
    df = pd.DataFrame({"u": t[unit].astype(str).to_numpy(), "s": t.split_seed.to_numpy(), "n": 1.0,
                       "y": t.y.to_numpy(), "y2": t.y.to_numpy() ** 2,
                       "er": (t[ref] - t.y).to_numpy() ** 2, "ec": (t[cand] - t.y).to_numpy() ** 2})
    per_seed = df.groupby(["u", "s"])[["n", "y", "y2", "er", "ec"]].sum()
    m = per_seed.groupby(level="u").mean()
    return {k: m[k].to_numpy(float) for k in ("n", "y", "y2", "er", "ec")} | {"ids": m.index.to_numpy()}


# ---------------------------------------------------------------- tests (rows = replicates, cols = units)
def flips(G, rng, B):
    if G <= 12:
        return np.array(list(itertools.product((1.0, -1.0), repeat=G))), True
    return rng.choice((-1.0, 1.0), size=(B, G)), False


def sf_p(X, F, exact):
    T = X.sum(1)
    tol = 1e-10 * np.abs(X).sum(1)
    cnt = np.zeros(len(X))
    for i in range(0, F.shape[0], 1000):
        cnt += ((X @ F[i:i + 1000].T) >= (T - tol)[:, None]).sum(1)
    return cnt / F.shape[0] if exact else (1 + cnt) / (F.shape[0] + 1)


def sign_p(d, scale):
    eps = 1e-12 * scale
    w = (d > eps).sum(1)
    l_ = (d < -eps).sum(1)
    out = np.ones(len(d))
    m = (w + l_) > 0
    out[m] = stats.binom.sf(w[m] - 1, (w + l_)[m], 0.5)
    return out


def t_p(X):
    G = X.shape[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        tt = X.mean(1) / (X.std(1, ddof=1) / np.sqrt(G))
    tt = np.where(np.isnan(tt), -np.inf, tt)
    return stats.t.sf(tt, G - 1)


def wild_t_p(X, rng, B=1999):
    """Restricted wild bootstrap-t (Webb) for H0 mean = 0, one observation per unit (HC2 leverage 1/G)."""
    R, G = X.shape
    def tstat(Z):
        m = Z.mean(-1)
        s = np.sqrt(((Z - m[..., None]) ** 2).sum(-1) / (G * (G - 1)))
        with np.errstate(divide="ignore", invalid="ignore"):
            return m / s
    t0 = tstat(X)
    V = WEBB[rng.integers(0, 6, size=(B, G))]
    out = np.empty(R)
    for i in range(0, R, 50):
        Z = X[i:i + 50, None, :] * V[None]          # (r, B, G)
        tb = tstat(Z)
        ok = np.isfinite(tb)
        out[i:i + 50] = (1 + ((tb >= t0[i:i + 50, None]) & ok).sum(1)) / (1 + ok.sum(1))
    out[~np.isfinite(t0)] = 1.0
    return out


def jk_rel(Sr, Sc, n):
    G = Sr.shape[1]
    R_, C_ = Sr.sum(1), Sc.sum(1)
    est = 1 - C_ / R_
    loo = 1 - (C_[:, None] - Sc) / (R_[:, None] - Sr)
    se = np.sqrt((G - 1) / G * ((loo - loo.mean(1, keepdims=True)) ** 2).sum(1))
    with np.errstate(divide="ignore", invalid="ignore"):
        tt = est / se
    kish = n.sum(1) ** 2 / (n ** 2).sum(1)
    pg1 = np.where(np.isfinite(tt), stats.t.sf(tt, G - 1), 1.0)
    pk = np.where(np.isfinite(tt), stats.t.sf(tt, np.maximum(kish - 1, 1)), 1.0)
    return pg1, pk


def sealed_r2(n, Sy, Sy2, Sr, Sc, rng, B=999, a=A):
    R, G = n.shape
    W = rng.multinomial(G, np.full(G, 1 / G), size=B).astype(float)        # (B, G)
    out = np.zeros(R, bool)
    za = stats.norm.ppf(a)
    tq = stats.t.ppf(1 - a, G - 1)
    for i in range(0, R, 250):
        sl = slice(i, i + 250)
        nn, sy, sy2, sr, sc = n[sl], Sy[sl], Sy2[sl], Sr[sl], Sc[sl]
        N = nn.sum(1)
        th = (sr.sum(1) - sc.sum(1)) / (sy2.sum(1) - sy.sum(1) ** 2 / N)
        Nb = nn @ W.T
        thb = (sr @ W.T - sc @ W.T) / (sy2 @ W.T - (sy @ W.T) ** 2 / Nb)
        perc = np.quantile(thb, a, axis=1)
        Nj = N[:, None] - nn
        jk = ((sr.sum(1)[:, None] - sr) - (sc.sum(1)[:, None] - sc)) / ((sy2.sum(1)[:, None] - sy2) -
                                                                          (sy.sum(1)[:, None] - sy) ** 2 / Nj)
        jm = jk.mean(1, keepdims=True)
        se = np.sqrt((G - 1) / G * ((jk - jm) ** 2).sum(1))
        num = ((jm - jk) ** 3).sum(1)
        den = 6 * (((jm - jk) ** 2).sum(1)) ** 1.5
        acc = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
        prop = ((thb < th[:, None]).sum(1) + 0.5 * (thb == th[:, None]).sum(1)) / B
        good = (prop > 0) & (prop < 1) & np.isfinite(acc)
        z0 = stats.norm.ppf(np.clip(prop, 1e-12, 1 - 1e-12))
        a1 = stats.norm.cdf(z0 + (z0 + za) / (1 - acc * (z0 + za)))
        a1 = np.where(np.isfinite(a1), np.clip(a1, 0, 1), 0.5)
        srt = np.sort(thb, 1)
        pos = a1 * (B - 1)
        lo_ = np.floor(pos).astype(int)
        hi_ = np.minimum(lo_ + 1, B - 1)
        rr = np.arange(len(srt))
        bca = srt[rr, lo_] + (pos - lo_) * (srt[rr, hi_] - srt[rr, lo_])
        kish = N ** 2 / (nn ** 2).sum(1)
        out[sl] = (perc > 0) & good & (bca > 0) & (th - tq * se > 0) & (kish >= 3)
    return out


def replay(pool, G, R, world, tau, key, tests, noise="per_arm", b_flip=2999):
    rng = np.random.default_rng(seed(*key))
    M = len(pool["n"])
    idx = rng.integers(0, M, size=(R, G))
    n, Sy, Sy2, Sr, Sc = (pool[k][idx].astype(float) for k in ("n", "y", "y2", "er", "ec"))
    if world == "SYM":
        Sc = Sr.copy()
    if tau > 0:
        if noise == "per_arm":
            Sr = Sr * (1 - rng.normal(0, tau / np.sqrt(2), (R, G))) ** 2
            Sc = Sc * (1 - rng.normal(0, tau / np.sqrt(2), (R, G))) ** 2
        else:  # round-4 candidate-only form
            Sc = Sc * (1 - rng.normal(0, tau, (R, G))) ** 2
    if world == "SWAP":
        sw = rng.random((R, G)) < 0.5
        Sr, Sc = np.where(sw, Sc, Sr), np.where(sw, Sr, Sc)
    d = Sr - Sc
    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.log(Sr / Sc)
    res = {}
    F, ex = flips(G, rng, b_flip)
    if "SIGN" in tests:
        res["SIGN"] = sign_p(d, Sr)
    if "SF_LOG" in tests:
        res["SF_LOG"] = sf_p(L, F, ex)
    if "SF_POOL" in tests:
        res["SF_POOL"] = sf_p(d, F, ex)
    if "T_LOG" in tests:
        res["T_LOG"] = t_p(L)
    if "WCR_LOG" in tests:
        res["WCR_LOG"] = wild_t_p(L, rng)
    if "JK_G1" in tests or "JK_KISH" in tests:
        res["JK_G1"], res["JK_KISH"] = jk_rel(Sr, Sc, n)
    rej = {k: float((v <= A + 1e-12).mean()) for k, v in res.items()}
    if "SEALED" in tests:
        rej["SEALED"] = float(sealed_r2(n, Sy, Sy2, Sr, Sc, rng).mean())
    return rej


def main():
    t0 = time.time()
    out = {"note": "EXPLORATORY (229 already-seen C3 cells). Skeptic recompute of P2.", "dev": {}, "kill": [], "g80": [],
           "size": []}
    data = {d: load(d) for d in ("BPG", "BP")}
    for d, t in data.items():
        out[f"s2_{d}"] = t.attrs["s2"]
        out[f"rows_{d}"] = int(len(t))
        print(d, "rows", len(t), "s2", round(t.attrs["s2"], 4), flush=True)
    # ------------------------------------------------ A dev
    for d, t in data.items():
        for unit in ("cluster_cpg", "publication", "chemotype", "pubchem"):
            p = unit_table(t, unit, "B1", "B0")
            dd = p["er"] - p["ec"]
            L = np.log(p["er"] / p["ec"])
            M = len(dd)
            kish = p["n"].sum() ** 2 / (p["n"] ** 2).sum()
            nz = dd != 0
            rec = {"design": d, "unit": unit, "M": M, "kish": round(float(kish), 3),
                   "largest_row_share": round(float(p["n"].max() / p["n"].sum()), 3),
                   "W/L/T": f"{(dd > 0).sum()}/{(dd < 0).sum()}/{(dd == 0).sum()}",
                   "meanL_nontied": round(float(L[nz].mean()), 4), "meanL_all": round(float(L.mean()), 4),
                   "pooled_rel": round(float(1 - p["ec"].sum() / p["er"].sum()), 4),
                   "p_SIGN": float(sign_p(dd[None], p["er"][None])[0]), "p_T_LOG": float(t_p(L[None])[0])}
            rng = np.random.default_rng(seed("dev", d, unit))
            if M <= 13:  # exact enumeration (2^13); the first run used flips() here, which gave B = 0 for M = 13
                F, ex = np.array(list(itertools.product((1.0, -1.0), repeat=M))), True  # and printed p = 1.0;
            else:                                                                        # see P2SK_gridcheck.py
                F, ex = rng.choice((-1.0, 1.0), size=(200_000, M)), False
            rec["p_SF_LOG"] = float(sf_p(L[None], F, ex)[0])
            rec["p_SF_POOL"] = float(sf_p(dd[None], F, ex)[0])
            out["dev"].setdefault(d, []).append(rec)
            print(rec, flush=True)
    pools = {}
    for d, t in data.items():
        for unit in ("cluster_cpg", "publication", "chemotype", "pubchem"):
            for name, c, r in (("B1-B0", "B1", "B0"), ("B1s2-B0", "B1s2", "B0"), ("B1-B1", "B1", "B1")):
                pools[(d, unit, name)] = unit_table(t, unit, c, r)
    ALL = ("SIGN", "SF_LOG", "SF_POOL", "T_LOG", "WCR_LOG", "JK_G1", "JK_KISH", "SEALED")
    # ------------------------------------------------ B kill cell
    for d in ("BPG", "BP"):
        for tau in (0.0, 0.065):
            for noise in (("per_arm", "cand_only") if tau > 0 else ("per_arm",)):
                rej = replay(pools[(d, "publication", "B1-B0")], 8, 4000, "EMP", tau, ("kill", d, tau, noise), ALL,
                             noise=noise)
                rec = {"design": d, "tau": tau, "noise": noise, "G": 8, "R": 4000, **{k: round(v, 4) for k, v in rej.items()}}
                out["kill"].append(rec)
                print(f"{time.time() - t0:6.0f}s KILL", rec, flush=True)
    # ------------------------------------------------ C G80 checked directly
    plan = [("publication", 0.0, (12, 19, 20, 37, 39, 43, 44)), ("publication", 0.065, (20, 28, 80)),
            ("chemotype", 0.0, (30, 41, 58, 70)), ("pubchem", 0.0, (59, 66, 69, 72))]
    for unit, tau, Gs in plan:
        for G in Gs:
            rej = replay(pools[("BPG", unit, "B1-B0")], G, 2000, "EMP", tau, ("g80", unit, tau, G), ALL)
            rec = {"design": "BPG", "unit": unit, "tau": tau, "G": G, "R": 2000, **{k: round(v, 4) for k, v in rej.items()}}
            out["g80"].append(rec)
            print(f"{time.time() - t0:6.0f}s G80", rec, flush=True)
    # ------------------------------------------------ D size of the jackknife legs (and the exact tests as a check)
    cells = [("BPG", "cluster_cpg", "B1-B0", "SWAP", 0.065, 8), ("BPG", "cluster_cpg", "B1-B1", "SYM", 0.065, 8),
             ("BPG", "cluster_cpg", "B1s2-B0", "SWAP", 0.0, 12), ("BP", "cluster_cpg", "B1s2-B0", "SWAP", 0.0, 12),
             ("BPG", "publication", "B1-B0", "SWAP", 0.0, 20), ("BPG", "publication", "B1-B0", "SWAP", 0.0, 40),
             ("BPG", "chemotype", "B1s2-B0", "SWAP", 0.0, 40), ("BPG", "chemotype", "B1-B0", "SWAP", 0.0, 40),
             ("BPG", "pubchem", "B1-B0", "SWAP", 0.0, 40)]
    for d, unit, name, world, tau, G in cells:
        rej = replay(pools[(d, unit, name)], G, 6000, world, tau, ("size", d, unit, name, world, tau, G),
                     ("SIGN", "SF_LOG", "SF_POOL", "T_LOG", "JK_G1", "JK_KISH"))
        rec = {"design": d, "unit": unit, "pool": name, "world": world, "tau": tau, "G": G, "R": 6000,
               **{k: round(v, 4) for k, v in rej.items()}}
        out["size"].append(rec)
        print(f"{time.time() - t0:6.0f}s SIZE", rec, flush=True)
    out["seconds"] = time.time() - t0
    print("RESULTS_JSON", json.dumps(out, default=str))
    print("done", round(time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
