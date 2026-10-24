#!/usr/bin/env python
"""PROBE P2 (round-5 idea M34): source-level resampling replay of paired few-cluster tests.

EXPLORATORY.  Every number comes from the round-4 OOF parquets of the 229 C3 cells / 5 dev seeds, which have
already been seen.  Nothing here is confirmatory; it calibrates tests and plans G for NEW data.

Design (written before the first run; changes after it are listed in the report):
  Rows   REF_<design>.parquet (+ S01/S02/S05/S06/S11_<design>.parquet merged on the row key, B1 verified identical),
         C3 at k = 3 (k_max >= 3, k == 3), 1,802 rows x 5 seeds.
  Units  cluster_cpg (r4.metrics.add_clusters, 13), publication (39), chemotype (38); extension:
         publication x chemotype (64).
  Per-unit statistics, seed-mean over the 5 dev seeds:  n rows / seed, SSE_ref, SSE_cand, Sy, Sy2;
         dSSE = SSE_ref - SSE_cand,  L = log(SSE_ref / SSE_cand),  win = dSSE > 0 (tie = exact 0).
         Positive always favours the candidate.
  Pools  (cand - ref): B1-B0 (positive control), B1s50-B0 (B0 + 0.5 (B1 - B0)), B1s2-B0 (B0 + s (B1 - B0), s solved
         so the pooled dev relative SSE reduction vs B0 = 2 %, the M35 gate size), HBK@ML-B0 (estimand-mismatch
         case), HIST_D7-B1 (negative control), S01/S02/S05/S06/S11-B1 (round-4 slots); B1-B1 (negative control,
         only through the SYM world).
  Worlds (G new units drawn uniformly with replacement from the pool's units):
         EMP   the drawn units' real pairs (power for B1-B0 and its scalings; 'empirical rejection' otherwise);
         SWAP  arm-swap null: each drawn unit's two arms exchanged with probability 1/2 (gating null; keeps the real
               unit-size and heavy-tail heterogeneity of that pool);
         SYM   symmetric exchangeable null: B1 vs B1, both arms with independent per-source noise (gating null);
         C-LOG / C-POOL / C-MED  skewed heterogeneous nulls from the B1-B0 pool (reported, NOT gating):
               log ratios minus their pool mean / dSSE minus n x pooled per-row dSSE / log ratios minus their median.
  Noise  extra per-source noise tau in {0, 0.065}: each arm's SSE x (1 - kappa)^2, kappa ~ N(0, tau / sqrt 2), iid per
         arm and unit (so the contrast's relative-SSE noise has SD about 2 tau = 0.13, round 4's calibration, and the
         perturbation is exchangeable between arms; round 4 perturbed the candidate only).  SYM also at tau 0.13.
  G      5, 8, 12, 20, 40 (+ 80 extension).
  Reps   SWAP / SYM: 4,000 per cell for B1-B0, HIST_D7-B1, B1-B1; 2,000 for the other pools; EMP and C-*: 2,000.
  Tests  (one-sided, H1: the candidate is better; alpha 0.025 primary, 0.05 sensitivity)
         SIGN     exact binomial sign test on per-unit wins (ties dropped);                      estimand pi
         SF_LOG   sign-flip randomisation on the mean log SSE ratio (exact for G <= 12, else 4,999 random flips
                  + identity); a sharp-null test;                                                   equal-weight
         SF_POOL  sign-flip on the pooled dSSE (sum over units);                                    pooled
         WCR_POOL wild cluster restricted bootstrap-t, CR2, Webb 6-point, 1,999 draws, per-row dSSE
                  clustered by unit (gen13sep/wildcluster.py algebra, made one-sided and vectorised);  pooled
         WCR_LOG  the same on the per-unit log ratio (one observation per unit, CR2 = HC2);       equal-weight
         JK_KISH  cluster-jackknife t on the pooled relative SSE reduction 1 - sum SSE_c / sum SSE_r,
                  df = max(Kish_rows - 1, 1);                                                       pooled
         JK_G1    the same with df = G - 1 (the sealed jackknife-t leg);                            pooled
         T_LOG    one-sample t on per-unit log ratios, df G - 1 (= IM-t on L; = jackknife-t, Kish = G)
         IM_REL   IM-t on per-unit relative SSE reduction 1 - SSE_c / SSE_r, df G - 1
         SEALED   the sealed R^2 leg: percentile, BCa and jackknife-t(G-1) lower bounds of dR^2 all > 0 and
                  Kish >= 3 (95 % two-sided intervals for 0.025; 90 % for 0.05), 999 cluster-bootstrap draws;
                  only for B1-B0, HIST_D7-B1 and B1-B1 pools.
  Size gate (IDEAS_RANKED M34): a test is anti-conservative if, in a SWAP or SYM null cell, the Clopper-Pearson 95 %
         lower bound of its rejection rate exceeds 0.035 at nominal 0.025 (0.07 at 0.05, sensitivity).
  Randomness: numpy default_rng(20261023 + crc32(cell key)); flip / Webb / bootstrap matrices drawn once per cell
         and shared by its replicates.
  Extensions (B1-B0 only, BPG): per-seed pool (unit and seed drawn), DGA / non-DGA publication strata.

    .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_design_20261023/probes/scripts/P2_power_replay.py --design BPG
writes probes/P2_out/{P2_unit_stats_<design>.csv, P2_dev_tests_<design>.csv, P2_grid_<design>.csv, P2_checks_<design>.json}
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
import zlib
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve()
PROBES = HERE.parents[1]
R4DIR = HERE.parents[3] / "round4_20261022"
sys.path.insert(0, str(R4DIR))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as SS  # noqa: E402

from r4 import metrics as M4  # noqa: E402
from r4 import paths  # noqa: E402

sys.path.insert(0, str(paths.ROOT / "generations" / "gen13_separation"))
from gen13sep import wildcluster as WC  # noqa: E402

KEY = ["split_seed", "cell_id", "read_set", "A", "B", "draw", "k", "policy"]
SLOTS = ("S01", "S02", "S05", "S06", "S11")
UNITS = ("cluster_cpg", "publication", "chemotype", "pubchem")
GS = (5, 8, 12, 20, 40, 80)
TAUS = (0.0, 0.065)
ALPHAS = (0.025, 0.05)
B_FLIP, B_WCR, B_BOOT, EXACT_MAX = 4999, 1999, 999, 12
WEBB = WC._WEBB
TESTS = ("SIGN", "SF_LOG", "SF_POOL", "WCR_POOL", "WCR_LOG", "JK_KISH", "JK_G1", "T_LOG", "IM_REL", "SEALED")
OUT = PROBES / "P2_out"


# ------------------------------------------------------------------------------------------------
# data
# ------------------------------------------------------------------------------------------------
def load(design: str) -> pd.DataFrame:
    t = pd.read_parquet(paths.PRED / f"REF_{design}.parquet")
    t = t[(t.k_max >= 3) & (t.k == 3)].reset_index(drop=True)
    for s in SLOTS:
        u = pd.read_parquet(paths.PRED / f"{s}_{design}.parquet")
        u = u[(u.k_max >= 3) & (u.k == 3)][KEY + ["B1", "y", s]].rename(columns={"B1": "B1_" + s, "y": "y_" + s})
        t = t.merge(u, on=KEY, how="left", validate="one_to_one")
        assert t[s].notna().all(), s
        assert np.array_equal(t["B1_" + s].to_numpy(), t.B1.to_numpy()), s
        assert np.array_equal(t["y_" + s].to_numpy(), t.y.to_numpy()), s
        t = t.drop(columns=["B1_" + s, "y_" + s])
    t = M4.add_clusters(t, None)
    t["pubchem"] = t.publication.astype(str) + "|" + t.chemotype.astype(str)
    d = (t.B1 - t.B0).to_numpy()
    e0 = (t.B0 - t.y).to_numpy()
    t["B1s50"] = t.B0 + 0.5 * d
    # s such that 1 - SSE(B0 + s d) / SSE(B0) = 0.02 (pooled over the 5 seeds): quadratic, smaller root
    a, b, c = (d ** 2).sum(), 2 * (e0 * d).sum(), 0.02 * (e0 ** 2).sum()
    s2 = (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)
    t["B1s2"] = t.B0 + s2 * d
    t.attrs["s2"] = float(s2)
    return t


POOLS = [("B1-B0", "B1", "B0"), ("B1s50-B0", "B1s50", "B0"), ("B1s2-B0", "B1s2", "B0"), ("HBK@ML-B0", "HBK@ML", "B0"),
         ("HIST_D7-B1", "HIST_D7", "B1")] + [(f"{s}-B1", s, "B1") for s in SLOTS] + [("B1-B1", "B1", "B1")]


def unit_stats(t: pd.DataFrame, unit: str, cand: str, ref: str, per_seed: bool = False):
    g = t.assign(er=(t[ref] - t.y) ** 2, ec=(t[cand] - t.y) ** 2, y2=t.y ** 2, one=1.0)
    agg = g.groupby([unit, "split_seed"])[["one", "y", "y2", "er", "ec"]].sum()
    if per_seed:
        return agg  # (unit, seed) rows
    return agg.groupby(level=0).mean()  # seed mean


# ------------------------------------------------------------------------------------------------
# tests: arrays are (R, G); return one-sided p-values (R,) or, for SEALED, reject booleans per alpha
# ------------------------------------------------------------------------------------------------
def flip_matrix(G: int, rng, exact_max=EXACT_MAX, B=B_FLIP):
    if G <= exact_max:
        return np.array(list(itertools.product((1.0, -1.0), repeat=G))), True   # first row = identity
    return rng.choice((1.0, -1.0), size=(B, G)), False


def p_signflip(X: np.ndarray, E: np.ndarray, exact: bool, chunk: int = 400) -> np.ndarray:
    T = X.sum(1)
    eps = 1e-10 * (np.abs(X).sum(1) + 1e-300)
    out = np.empty(len(X))
    for i in range(0, len(X), chunk):
        D = X[i:i + chunk] @ E.T                                     # (r, B)
        ge = (D >= (T[i:i + chunk] - eps[i:i + chunk])[:, None]).sum(1)
        out[i:i + chunk] = ge / E.shape[0] if exact else (1.0 + ge) / (E.shape[0] + 1.0)
    return out


def p_sign(dS: np.ndarray, Sr: np.ndarray) -> np.ndarray:
    tol = 1e-12 * np.maximum(np.abs(Sr), 1e-300)
    w = (dS > tol).sum(1)
    l = (dS < -tol).sum(1)
    n = w + l
    p = SS.binom.sf(w - 1, n, 0.5)
    return np.where(n > 0, p, 1.0)


def p_wcr(S: np.ndarray, n: np.ndarray, V: np.ndarray, two_sided: bool = False, chunk: int = 500) -> np.ndarray:
    """WCR bootstrap-t, CR2, for H0: pooled per-row mean = 0; S = unit sums of the per-row contrast, n = rows.
    Same algebra as gen13sep.wildcluster.wild_cluster_t (restricted residuals, cluster-sum form)."""
    out = np.empty(len(S))
    B = V.shape[0]
    V2 = V ** 2
    for i in range(0, len(S), chunk):
        s_, n_ = S[i:i + chunk], n[i:i + chunk]
        N = n_.sum(1)
        mu = s_.sum(1) / N
        h = n_ / N[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            c2 = 1.0 / (1.0 - h)
            u = s_ - n_ * mu[:, None]
            t_obs = mu / np.sqrt((c2 * u ** 2).sum(1) / N ** 2)
            num = (s_ @ V.T) / N[:, None]
            A = (c2 * s_ ** 2) @ V2.T
            Bm = (c2 * s_ * n_) @ V.T
            C = (c2 * n_ ** 2).sum(1)
            v = (A - 2 * num * Bm + num ** 2 * C[:, None]) / N[:, None] ** 2
            tb = num / np.sqrt(np.maximum(v, 0))
        ok = np.isfinite(tb)
        if two_sided:
            ge = (np.abs(tb) >= np.abs(t_obs)[:, None]) & ok
        else:
            ge = (tb >= t_obs[:, None]) & ok
        p = (1.0 + ge.sum(1)) / (ok.sum(1) + 1.0)
        p[~np.isfinite(t_obs)] = 1.0
        out[i:i + chunk] = p
    return out


def p_t(x: np.ndarray, df) -> np.ndarray:
    """One-sample one-sided t on rows of x (equal weights)."""
    G = x.shape[1]
    m = x.mean(1)
    sd = x.std(1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = m / (sd / np.sqrt(G))
    p = SS.t.sf(np.nan_to_num(t, nan=-np.inf), df)       # 0/0 (all-tied units) -> p = 1; +inf -> p = 0
    return p


def jk_pooled_rel(Sr, Sc, n):
    G = Sr.shape[1]
    Tr, Tc = Sr.sum(1), Sc.sum(1)
    pt = 1.0 - Tc / Tr
    with np.errstate(divide="ignore", invalid="ignore"):
        jk = 1.0 - (Tc[:, None] - Sc) / (Tr[:, None] - Sr)
        se = np.sqrt((G - 1) / G * ((jk - jk.mean(1, keepdims=True)) ** 2).sum(1))
        t = pt / se
    kish = n.sum(1) ** 2 / (n ** 2).sum(1)
    return t, kish


def sealed(n, Sy, Sy2, Sr, Sc, W, alphas=ALPHAS, chunk=500):
    """Sealed R^2 leg on dR^2 = (sum SSE_r - sum SSE_c) / SST: percentile, BCa and jk-t(G-1) lower bounds > 0, Kish >= 3."""
    R, G = n.shape
    res = {a: np.zeros(R, bool) for a in alphas}
    Wt = W.T
    for i in range(0, R, chunk):
        sl = slice(i, i + chunk)
        nn, sy, sy2, sr, sc = n[sl], Sy[sl], Sy2[sl], Sr[sl], Sc[sl]
        N = nn.sum(1)
        pt = (sr.sum(1) - sc.sum(1)) / (sy2.sum(1) - sy.sum(1) ** 2 / N)
        Nb = nn @ Wt
        bt = (sr @ Wt - sc @ Wt) / (sy2 @ Wt - (sy @ Wt) ** 2 / Nb)
        # jackknife
        Nj = N[:, None] - nn
        jk = ((sr.sum(1)[:, None] - sr) - (sc.sum(1)[:, None] - sc)) / \
             ((sy2.sum(1)[:, None] - sy2) - (sy.sum(1)[:, None] - sy) ** 2 / Nj)
        se = np.sqrt((G - 1) / G * ((jk - jk.mean(1, keepdims=True)) ** 2).sum(1))
        kish = N ** 2 / (nn ** 2).sum(1)
        bs = np.sort(bt, 1)
        Bn = bs.shape[1]
        prop = ((bt < pt[:, None]).sum(1) + 0.5 * (bt == pt[:, None]).sum(1)) / Bn
        m = jk.mean(1, keepdims=True)
        den = 6 * (((m - jk) ** 2).sum(1)) ** 1.5
        acc = np.where(den > 0, ((m - jk) ** 3).sum(1) / np.where(den > 0, den, 1), np.nan)
        z0 = SS.norm.ppf(np.clip(prop, 1e-300, 1 - 1e-16))
        for a in alphas:
            perc = np.quantile(bt, a, axis=1)
            z = SS.norm.ppf(a)
            adj = SS.norm.cdf(z0 + (z0 + z) / (1 - acc * (z0 + z)))
            okb = (prop > 0) & (prop < 1) & np.isfinite(adj)
            pos = np.clip(adj, 0, 1) * (Bn - 1)
            lo_i = np.floor(pos).astype(int).clip(0, Bn - 1)
            hi_i = np.minimum(lo_i + 1, Bn - 1)
            fr = pos - lo_i
            rows = np.arange(len(bs))
            bca = bs[rows, lo_i] * (1 - fr) + bs[rows, hi_i] * fr
            jl = pt - SS.t.ppf(1 - a, G - 1) * se
            res[a][sl] = (perc > 0) & okb & (bca > 0) & (jl > 0) & (kish >= 3)
    return res


def run_tests(n, Sy, Sy2, Sr, Sc, rng, *, do_sealed: bool, exact_max=EXACT_MAX, b_flip=B_FLIP):
    R, G = n.shape
    dS = Sr - Sc
    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.log(Sr / Sc)
        rel = 1.0 - Sc / Sr
    E, exact = flip_matrix(G, rng, exact_max, b_flip)
    V = WEBB[rng.integers(0, 6, size=(B_WCR, G))]
    P = {"SIGN": p_sign(dS, Sr),
         "SF_LOG": p_signflip(L, E, exact),
         "SF_POOL": p_signflip(dS, E, exact),
         "WCR_POOL": p_wcr(dS, n, V),
         "WCR_LOG": p_wcr(L, np.ones_like(L), V),
         "T_LOG": p_t(L, G - 1),
         "IM_REL": p_t(rel, G - 1)}
    t, kish = jk_pooled_rel(Sr, Sc, n)
    P["JK_KISH"] = np.where(np.isfinite(t), SS.t.sf(t, np.maximum(kish - 1, 1.0)), 1.0)
    P["JK_G1"] = np.where(np.isfinite(t), SS.t.sf(t, G - 1), 1.0)
    rej = {k: {a: v <= a + 1e-12 for a in ALPHAS} for k, v in P.items()}
    if do_sealed:
        W = rng.multinomial(G, np.full(G, 1.0 / G), size=B_BOOT).astype(float)
        rej["SEALED"] = sealed(n, Sy, Sy2, Sr, Sc, W)
    extra = {"kish_median": float(np.median(kish)),
             "pooled_rel_median": float(np.median(1 - Sc.sum(1) / Sr.sum(1))),
             "meanL_median": float(np.median(np.nanmean(L, 1))),
             "win_share_mean": float(np.mean((dS > 0).sum(1) / G))}
    return P, rej, extra


# ------------------------------------------------------------------------------------------------
# replay
# ------------------------------------------------------------------------------------------------
def cp(x: int, n: int, conf=0.95):
    a = 1 - conf
    lo = 0.0 if x == 0 else SS.beta.ppf(a / 2, x, n - x + 1)
    hi = 1.0 if x == n else SS.beta.ppf(1 - a / 2, x + 1, n - x)
    return float(lo), float(hi)


def draw(pool: dict, G: int, R: int, rng, world: str, tau: float):
    M = len(pool["n"])
    if pool.get("per_seed"):
        ui = rng.integers(M, size=(R, G))
        si = rng.integers(pool["Sr"].shape[1], size=(R, G))
        get = lambda k: pool[k][ui, si]  # noqa: E731
    else:
        ui = rng.integers(M, size=(R, G))
        get = lambda k: pool[k][ui]  # noqa: E731
    n, Sy, Sy2, Sr, Sc = (get(k).astype(float) for k in ("n", "Sy", "Sy2", "Sr", "Sc"))
    if world == "SYM":
        Sc = Sr.copy()
    if world in ("C-LOG", "C-MED", "C-POOL"):
        Sc = get("Sc_" + world).astype(float)
    if tau > 0:
        s = tau / np.sqrt(2.0)
        Sr = Sr * (1 - rng.normal(0, s, size=(R, G))) ** 2
        Sc = Sc * (1 - rng.normal(0, s, size=(R, G))) ** 2
    if world == "SWAP":
        sw = rng.random((R, G)) < 0.5
        Sr, Sc = np.where(sw, Sc, Sr), np.where(sw, Sr, Sc)
    return n, Sy, Sy2, Sr, Sc


def make_pool(us: pd.DataFrame, per_seed=False) -> dict:
    if per_seed:
        def arr(c):
            return us[c].unstack().to_numpy()
    else:
        def arr(c):
            return us[c].to_numpy()
    p = {"n": arr("one"), "Sy": arr("y"), "Sy2": arr("y2"), "Sr": arr("er"), "Sc": arr("ec"), "per_seed": per_seed}
    if not per_seed:
        L = np.log(p["Sr"] / p["Sc"])
        p["Sc_C-LOG"] = p["Sr"] * np.exp(-(L - L.mean()))
        p["Sc_C-MED"] = p["Sr"] * np.exp(-(L - np.median(L)))
        dS = p["Sr"] - p["Sc"]
        dS2 = dS - p["n"] * dS.sum() / p["n"].sum()
        p["Sc_C-POOL"] = p["Sr"] - dS2
        if (p["Sc_C-POOL"] <= 0).any():
            p["Sc_C-POOL"] = np.maximum(p["Sc_C-POOL"], 1e-9 * p["Sr"])
            p["c_pool_clipped"] = True
    return p


def cell_seed(*parts) -> int:
    return 20261023 + zlib.crc32("|".join(map(str, parts)).encode())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="BPG")
    ap.add_argument("--quick", action="store_true", help="timing smoke: few cells, few reps")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    t = load(args.design)
    checks = {"design": args.design, "rows": int(len(t)), "seeds": int(t.split_seed.nunique()), "s2": t.attrs["s2"]}

    # ---- per-unit statistics and dev (non-resampled, exploratory) tests
    pools, rows_u, rows_dev = {}, [], []
    for unit in UNITS:
        for name, cand, ref in POOLS:
            us = unit_stats(t, unit, cand, ref)
            p = make_pool(us)
            pools[(unit, name)] = p
            dS = p["Sr"] - p["Sc"]
            L = np.log(p["Sr"] / p["Sc"])
            for u_id, nn, sr, sc, d_, l_ in zip(us.index, p["n"], p["Sr"], p["Sc"], dS, L):
                rows_u.append({"unit_type": unit, "contrast": name, "unit": u_id, "n_rows_per_seed": nn, "SSE_ref": sr,
                               "SSE_cand": sc, "dSSE": d_, "log_SSE_ratio": l_,
                               "win": 1 if d_ > 0 else (0 if d_ < 0 else np.nan)})
            if name == "B1-B1":
                continue
            M = len(dS)
            rng = np.random.default_rng(cell_seed("dev", args.design, unit, name))
            P, rej, extra = run_tests(p["n"][None], p["Sy"][None], p["Sy2"][None], p["Sr"][None], p["Sc"][None], rng,
                                      do_sealed=name in ("B1-B0", "HIST_D7-B1"), exact_max=13, b_flip=199999)
            nz = dS != 0
            rec = {"unit_type": unit, "contrast": name, "M": M, "wins": int((dS > 0).sum()), "losses": int((dS < 0).sum()),
                   "ties": int((dS == 0).sum()), "pi_nontied": float((dS > 0).sum() / max(nz.sum(), 1)),
                   "meanL_all": float(L.mean()), "meanL_nontied": float(L[nz].mean()) if nz.any() else 0.0,
                   "medianL": float(np.median(L)), "sdL": float(L.std(ddof=1)),
                   "pooled_rel_SSE": float(1 - p["Sc"].sum() / p["Sr"].sum()),
                   "largest_unit_share_abs_dSSE": float(np.abs(dS).max() / max(np.abs(dS).sum(), 1e-300)),
                   "kish_rows": float(p["n"].sum() ** 2 / (p["n"] ** 2).sum())}
            for k, v in P.items():
                rec["p_" + k] = float(v[0])
            if "SEALED" in rej:
                rec["SEALED_pass_025"] = bool(rej["SEALED"][0.025][0])
            rows_dev.append(rec)
    pd.DataFrame(rows_u).to_csv(OUT / f"P2_unit_stats_{args.design}.csv", index=False)
    dev = pd.DataFrame(rows_dev)
    dev.to_csv(OUT / f"P2_dev_tests_{args.design}.csv", index=False)
    print(dev[["unit_type", "contrast", "M", "wins", "losses", "ties", "meanL_nontied", "pooled_rel_SSE", "p_SIGN", "p_SF_LOG",
               "p_SF_POOL", "p_WCR_POOL", "p_WCR_LOG", "p_JK_KISH", "p_T_LOG"]].to_string(), flush=True)

    # ---- implementation checks
    p = pools[("cluster_cpg", "B1-B0")]
    dS = p["Sr"] - p["Sc"]
    n_int = np.maximum(np.round(p["n"]), 1)
    rows_d = np.concatenate([np.full(int(nn), d_ / nn) for nn, d_ in zip(n_int, dS)])
    cl = np.concatenate([np.full(int(nn), i) for i, nn in enumerate(n_int)])
    wc = WC.wild_cluster_t(rows_d, cl, reps=B_WCR, seed=7, weights="webb", se_kind="CR2", ci=False)
    Vw = WC._draw_weights(np.random.default_rng(7), B_WCR, len(p["n"]), "webb")
    mine2 = p_wcr(dS[None], n_int[None], Vw, two_sided=True)[0]
    checks["n_rows_per_unit_fractional_cpg"] = bool(np.any(np.abs(p["n"] - n_int) > 1e-9))
    checks["wcr_two_sided_vs_wildcluster"] = {"wildcluster_p_wcr": wc.p_wcr, "this_p": float(mine2), "t_obs": wc.t_obs}
    x = np.random.default_rng(1).normal(0.2, 1, size=(3, 6))
    bf = [np.mean([np.dot(e, r) >= r.sum() - 1e-12 for e in itertools.product((1, -1), repeat=6)]) for r in x]
    E, _ = flip_matrix(6, None)
    checks["signflip_exact_vs_bruteforce"] = {"brute": bf, "this": p_signflip(x, E, True).tolist()}
    checks["sign_vs_binomtest"] = {"this": float(p_sign(np.array([[1, 1, 1, -1, 1, 0.0]]), np.ones((1, 6)))[0]),
                                   "scipy": float(SS.binomtest(4, 5, 0.5, alternative="greater").pvalue)}
    print(json.dumps(checks, indent=1), flush=True)

    # ---- replay grid
    grid = []
    gs = GS if not args.quick else (8, 40)
    cells = []
    for unit in UNITS:
        for name, _, _ in POOLS:
            if name == "B1-B1":
                for tau in (0.065, 0.13):
                    cells.append((unit, name, "SYM", tau))
                continue
            for tau in TAUS:
                cells.append((unit, name, "EMP", tau))
                cells.append((unit, name, "SWAP", tau))
                if name == "B1-B0":
                    for w in ("C-LOG", "C-POOL", "C-MED"):
                        cells.append((unit, name, w, tau))
    if args.quick:
        cells = [c for c in cells if c[0] in ("cluster_cpg", "publication") and c[1] in ("B1-B0", "B1-B1")]
    for unit, name, world, tau in cells:
        pool = pools[(unit, name)]
        big = name in ("B1-B0", "HIST_D7-B1", "B1-B1")
        R = (4000 if big else 2000) if world in ("SWAP", "SYM") else 2000
        if args.quick:
            R = 400
        for G in gs:
            rng = np.random.default_rng(cell_seed(args.design, unit, name, world, tau, G))
            n, Sy, Sy2, Sr, Sc = draw(pool, G, R, rng, world, tau)
            P, rej, extra = run_tests(n, Sy, Sy2, Sr, Sc, rng, do_sealed=big and world in ("EMP", "SWAP", "SYM"))
            for test, d in rej.items():
                rec = {"design": args.design, "unit_type": unit, "pool": name, "world": world, "tau": tau, "G": G,
                       "reps": R, "test": test, **extra}
                for a, r in d.items():
                    x_ = int(r.sum())
                    lo, hi = cp(x_, R)
                    tag = "025" if a == 0.025 else "05"
                    rec[f"rej_{tag}"] = x_ / R
                    rec[f"cp_lo_{tag}"], rec[f"cp_hi_{tag}"] = lo, hi
                grid.append(rec)
        print(f"{time.time() - t0:7.1f}s {unit} {name} {world} tau={tau}", flush=True)

    # ---- extensions (BPG only): per-seed pools, DGA strata
    if args.design == "BPG" and not args.quick:
        for unit in ("cluster_cpg", "publication", "chemotype"):
            for name, cand, ref in (("B1-B0", "B1", "B0"), ("HIST_D7-B1", "HIST_D7", "B1")):
                ps = make_pool(unit_stats(t, unit, cand, ref, per_seed=True), per_seed=True)
                for world in (("EMP", "SWAP") if name == "B1-B0" else ("SWAP",)):
                    for tau in TAUS:
                        for G in GS:
                            R = 2000
                            rng = np.random.default_rng(cell_seed("perseed", unit, name, world, tau, G))
                            n, Sy, Sy2, Sr, Sc = draw(ps, G, R, rng, world, tau)
                            P, rej, extra = run_tests(n, Sy, Sy2, Sr, Sc, rng, do_sealed=False)
                            for test, d in rej.items():
                                rec = {"design": "BPG", "unit_type": unit, "pool": name + "|perseed", "world": world,
                                       "tau": tau, "G": G, "reps": R, "test": test, **extra}
                                for a, r in d.items():
                                    tag = "025" if a == 0.025 else "05"
                                    rec[f"rej_{tag}"] = r.mean()
                                    rec[f"cp_lo_{tag}"], rec[f"cp_hi_{tag}"] = cp(int(r.sum()), R)
                                grid.append(rec)
                print(f"{time.time() - t0:7.1f}s perseed {unit} {name}", flush=True)
        # DGA strata: units whose rows are mostly chemotype sc009 vs the rest
        share = t.assign(dga=(t.chemotype == M4.DGA_CHEMOTYPE).astype(float))
        for unit in ("publication", "pubchem"):
            dg = share.groupby(unit).dga.mean()
            us = unit_stats(t, unit, "B1", "B0")
            for stratum, keep in (("DGA", dg.loc[us.index] > 0.5), ("nonDGA", dg.loc[us.index] <= 0.5)):
                p = make_pool(us[keep.to_numpy()])
                checks[f"strata_{unit}_{stratum}_units"] = int(keep.sum())
                for tau in TAUS:
                    for G in GS:
                        R = 2000
                        rng = np.random.default_rng(cell_seed("strata", unit, stratum, tau, G))
                        n, Sy, Sy2, Sr, Sc = draw(p, G, R, rng, "EMP", tau)
                        P, rej, extra = run_tests(n, Sy, Sy2, Sr, Sc, rng, do_sealed=False)
                        for test, d in rej.items():
                            rec = {"design": "BPG", "unit_type": unit, "pool": f"B1-B0|{stratum}", "world": "EMP",
                                   "tau": tau, "G": G, "reps": R, "test": test, **extra}
                            for a, r in d.items():
                                tag = "025" if a == 0.025 else "05"
                                rec[f"rej_{tag}"] = r.mean()
                                rec[f"cp_lo_{tag}"], rec[f"cp_hi_{tag}"] = cp(int(r.sum()), R)
                            grid.append(rec)
            print(f"{time.time() - t0:7.1f}s strata {unit}", flush=True)

    g = pd.DataFrame(grid)
    g.to_csv(OUT / f"P2_grid_{args.design}{'_quick' if args.quick else ''}.csv", index=False)
    checks["seconds"] = time.time() - t0
    (OUT / f"P2_checks_{args.design}{'_quick' if args.quick else ''}.json").write_text(json.dumps(checks, indent=1),
                                                                                         encoding="utf-8")
    print("done", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
