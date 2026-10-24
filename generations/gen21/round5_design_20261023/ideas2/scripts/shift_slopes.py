#!/usr/bin/env python
"""shift theme, step 2 (EXPLORATORY, all 521 bench cells already seen): within-publication acid dependence of log D
and of adjacent log SF, and whether it transfers across publications.

Group = publication x extractant x every recorded condition except acid concentration, contact time and metal
concentration (acid type, diluent, additive, extractant concentration, temperature identical).  V2-1/2/3 cells
dropped.  x = log10([acid]/M).  A group enters when it has >= 2 cells whose x spans >= 0.1.
Per group and adjacent pair (A, B) (strict adjacent plus Nd/Sm across Pm), y = logD_A - logD_B over cells with both
metals; OLS slope dy/dx (needs >= 2 distinct x spanning >= 0.1).  Per metal: OLS slope dlogD/dx.

Prints: group table; per-pair-type pooled slopes (equal weight per publication, sign-flip p over publications);
within-group split-half reliability, then between-group profile correlations; the SF stretch coefficient kappa
(slope_p regressed on mean y_p within group).  Writes ideas2/scripts/shift_out/slopes_groups.csv, slopes_pairs.csv.

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/shift_slopes.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402
from r3 import datav2 as V2  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import LANTHANIDES as LN, N_LN  # noqa: E402

OUT = Path(r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2\scripts\shift_out")
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 400)
pd.set_option("display.max_columns", 40)
PAIRS = [(i, i + 1) for i in range(N_LN - 1)]          # La/Ce ... Yb/Lu; (3,4) = Nd/Sm across Pm
PN = [f"{LN[a]}/{LN[b]}" for a, b in PAIRS]
STRICT = [p for p in PN if p != "Nd/Sm"]


def ols(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2 or len(np.unique(np.round(x, 3))) < 2 or x.max() - x.min() < 0.1:
        return np.nan, np.nan, len(x)
    X = np.c_[np.ones_like(x), x]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    se = np.nan
    if len(x) > 2:
        r = y - X @ beta
        s2 = (r ** 2).sum() / (len(x) - 2)
        se = float(np.sqrt(s2 / ((x - x.mean()) ** 2).sum()))
    return float(beta[1]), se, len(x)


def signflip_two_sided(v):
    v = np.asarray(v, float)
    G_ = len(v)
    if G_ > 16 or G_ == 0:
        return np.nan
    S = np.array(list(itertools.product((1, -1), repeat=G_)))
    null = S @ v
    return float((np.abs(null) >= abs(v.sum()) - 1e-12).mean())


def main():
    b = D.load_bench()
    f = b.frame.reset_index(drop=True)
    Y = np.asarray(b.Y, float)
    v2drop = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    c3 = set(pd.read_parquet(paths.PRED / "REF_BPG.parquet", columns=["cell_id"]).cell_id)

    def oh(pre):
        return [c for c in f.columns if c.startswith(pre)]

    def lab(r, cols, pre):
        s = [c[len(pre):] for c in cols if r[c] == 1]
        return "+".join(s) if s else "-"
    f["acid"] = [lab(r, oh("cond__acid__"), "cond__acid__") for _, r in f.iterrows()]
    f["dil"] = [lab(r, oh("cond__diluent__"), "cond__diluent__") for _, r in f.iterrows()]
    f["add"] = [lab(r, oh("cond__additive__"), "cond__additive__") for _, r in f.iterrows()]
    f["Lc"] = f.cond__extractant_concentration_M.round(3).astype(str)
    f["T"] = f.cond__temperature_C.round(0).astype(str)
    f["x"] = np.log10(f.cond__acid_concentration_M.astype(float))
    keep = ~f.cell_id.isin(v2drop) & f.x.notna()
    key = ["publication_id", "extractant", "acid", "dil", "add", "Lc", "T"]
    grows, prows, mrows = [], [], []
    for k, g in f[keep].groupby(key):
        if len(g) < 2 or g.x.max() - g.x.min() < 0.1:
            continue
        idx = g.index.to_numpy()
        x = g.x.to_numpy()
        gid = f"G{len(grows):02d}"
        bm = np.array([ols(x, Y[idx, m])[0] for m in range(N_LN)])
        rec = {"gid": gid, "pub": k[0], "ext": str(g.extractant_name.iloc[0])[:18], "chem": g.chemotype.iloc[0], "acid": k[2],
               "dil": k[3][:16], "Lc": k[5], "T": k[6], "ncell": len(g), "nx": len(np.unique(np.round(x, 3))),
               "cmin": float(10 ** x.min()), "cmax": float(10 ** x.max()), "n_c3": int(g.cell_id.isin(c3).sum()),
               "mean_slope_logD": float(np.nanmean(bm)) if np.isfinite(bm).any() else np.nan,
               "n_metals_slope": int(np.isfinite(bm).sum())}
        if np.isfinite(bm).sum() >= 6:
            mrows.append({"gid": gid, "pub": k[0][:10], "ext": rec["ext"][:10], "acid": k[2],
                          **{LN[m]: bm[m] - np.nanmean(bm) for m in range(N_LN)}, "mean": np.nanmean(bm)})
        for (a, bb), pn in zip(PAIRS, PN):
            yy = Y[idx, a] - Y[idx, bb]
            s, se, n = ols(x, yy)
            if not np.isfinite(s):
                continue
            ok = np.isfinite(yy)
            lev = np.nanmean((Y[idx, a] + Y[idx, bb]) / 2)
            xs = np.unique(np.round(x[ok], 3))
            sh = (np.nan, np.nan)
            if len(xs) >= 4:
                rk = {v: i for i, v in enumerate(xs)}
                part = np.array([rk.get(v, -1) % 2 if v in rk else -1 for v in np.round(x, 3)])
                s0 = ols(x[ok & (part == 0)], yy[ok & (part == 0)])[0]
                s1 = ols(x[ok & (part == 1)], yy[ok & (part == 1)])[0]
                sh = (s0, s1)
            prows.append({"gid": gid, "pub": k[0], "chem": rec["chem"], "acid": k[2], "pair": pn, "slope": s, "se": se,
                          "n": n, "ybar": float(np.nanmean(yy)), "level": float(lev), "sh0": sh[0], "sh1": sh[1],
                          "nx": len(xs), "xmid": float(np.nanmean(x[ok]))})
        grows.append(rec)
    G = pd.DataFrame(grows)
    P = pd.DataFrame(prows)
    G["n_pairs"] = G.gid.map(P.groupby("gid").size()).fillna(0).astype(int)
    G.to_csv(OUT / "slopes_groups.csv", index=False)
    P.to_csv(OUT / "slopes_pairs.csv", index=False)
    print("=== groups with an acid-concentration series (V2-clean, all bench cells) ===")
    print(G.round(3).to_string(index=False))
    Gp = G[G.n_pairs > 0]
    print(f"\ngroups {len(G)}, with >=1 pair slope {len(Gp)}; publications {Gp.pub.nunique()}; "
          f"with >= 6 strict pair slopes {int((G.n_pairs >= 6).sum())} ({G[G.n_pairs >= 6].pub.nunique()} pubs)")
    print("mean logD slope by acid type (groups):", G.groupby("acid").mean_slope_logD.agg(["count", "mean", "median"]).round(2).to_dict())

    def pooled(Q, label):
        rows = []
        for pn in PN:
            q = Q[Q.pair == pn]
            if q.empty:
                continue
            pp = q.groupby("pub").slope.mean()
            v = pp.to_numpy()
            rows.append({"pair": pn, "groups": len(q), "pubs": len(v), "mean_pub": v.mean(), "median_pub": np.median(v),
                         "sd_pub": v.std(ddof=1) if len(v) > 1 else np.nan, "frac_pos_pub": (v > 0).mean(),
                         "p2_signflip_pub": signflip_two_sided(v), "mean_group": q.slope.mean(),
                         "median_se": q.se.median()})
        print(f"\n=== pooled adjacent log SF slope d(logSF)/d(log10 acid), {label} ===")
        T = pd.DataFrame(rows)
        print(T.round(3).to_string(index=False))
        return T
    pooled(P, "all acid types")
    pooled(P[P.acid == "hno3"], "HNO3 only")
    pooled(P[(P.acid == "hno3") & (P.chem == "sc009")], "HNO3, sc009 only")
    pooled(P[(P.acid == "hno3") & (P.chem != "sc009")], "HNO3, other chemotypes")
    # pub-level summary of the whole strict profile: mean |slope|, mean slope
    q = P[P.pair.isin(STRICT)]
    print("\nper group: mean slope over strict pairs, mean |slope|, rms slope, rms ybar")
    print(q.groupby(["gid", "pub", "chem", "acid"]).agg(npair=("slope", "size"), mean_slope=("slope", "mean"),
                                                        mean_abs=("slope", lambda s: s.abs().mean()),
                                                        rms_ybar=("ybar", lambda s: np.sqrt((s ** 2).mean())),
                                                        nx=("nx", "max")).round(3).to_string())

    # ---- reliability, then correlation ------------------------------------------------------------------------
    S = P.dropna(subset=["sh0", "sh1"])
    S = S[S.pair.isin(STRICT)]
    rel = []
    for gid, qq in S.groupby("gid"):
        if len(qq) >= 5:
            r = np.corrcoef(qq.sh0, qq.sh1)[0, 1]
            rel.append({"gid": gid, "pub": qq.pub.iloc[0], "pairs": len(qq), "r_half": r,
                        "SB": 2 * r / (1 + r) if r > -1 else np.nan})
    R = pd.DataFrame(rel)
    print("\n=== within-group split-half reliability of the strict-pair slope profile (odd/even acid levels) ===")
    if len(R):
        print(R.round(3).to_string(index=False))
        print(f"median split-half r {R.r_half.median():.3f}; median Spearman-Brown {R.SB.median():.3f}")
    W = P[P.pair.isin(STRICT)].pivot_table(index="gid", columns="pair", values="slope")
    meta = P.drop_duplicates("gid").set_index("gid")
    cors = []
    for g1, g2 in itertools.combinations(W.index, 2):
        a, b_ = W.loc[g1], W.loc[g2]
        ok = a.notna() & b_.notna()
        if ok.sum() >= 5:
            cors.append({"g1": g1, "g2": g2, "pub1": meta.pub[g1][:10], "pub2": meta.pub[g2][:10], "n": int(ok.sum()),
                         "r": float(np.corrcoef(a[ok], b_[ok])[0, 1]),
                         "same_pub": meta.pub[g1] == meta.pub[g2], "same_chem": meta.chem[g1] == meta.chem[g2],
                         "both_hno3": meta.acid[g1] == "hno3" and meta.acid[g2] == "hno3"})
    C = pd.DataFrame(cors)
    print("\n=== between-group correlation of slope profiles (>= 5 common strict pairs) ===")
    if len(C):
        print(C.groupby(["same_pub", "same_chem", "both_hno3"]).r.agg(["count", "mean", "median"]).round(3).to_string())
        print(C.round(3).to_string(index=False))

    # ---- stretch ------------------------------------------------------------------------------------------------
    print("\n=== stretch: kappa0 = OLS through origin of slope_p on ybar_p within group (strict pairs, >= 5) ===")
    kr = []
    for gid, qq in P[P.pair.isin(STRICT)].groupby("gid"):
        if len(qq) >= 5 and qq.ybar.std() > 0:
            X0 = qq.ybar.to_numpy()
            k0 = float((X0 * qq.slope).sum() / (X0 ** 2).sum())
            kr.append({"gid": gid, "pub": qq.pub.iloc[0], "chem": qq.chem.iloc[0], "acid": qq.acid.iloc[0],
                       "pairs": len(qq), "kappa0": k0, "r": float(np.corrcoef(qq.ybar, qq.slope)[0, 1]),
                       "mean_slope_logD": float(G.set_index("gid").mean_slope_logD[gid])})
    K = pd.DataFrame(kr)
    if len(K):
        print(K.round(3).to_string(index=False))
        kp = K.groupby("pub").kappa0.mean()
        print(f"kappa0 by publication: {kp.round(3).to_dict()}; frac negative {(kp < 0).mean():.2f} of {len(kp)}; "
              f"sign-flip p2 {signflip_two_sided(kp.to_numpy()):.3f}")
        print(f"corr(kappa0, mean logD slope) over groups: {np.corrcoef(K.kappa0, K.mean_slope_logD)[0, 1]:.3f}")
    print("\n=== per-metal logD acid slope, centred within group (groups with >= 6 metal slopes) ===")
    print(pd.DataFrame(mrows).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
