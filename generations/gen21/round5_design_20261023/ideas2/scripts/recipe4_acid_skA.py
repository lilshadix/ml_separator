#!/usr/bin/env python
"""Skeptic A of idea recipe-4 (ICP-MS condition skipping: MEAN of acid-only siblings, half-shrunk log-acid line,
clamped).  EXPLORATORY: every number is on already-seen bench cells.  Independent re-implementation: the sibling
population is rebuilt from the bench metadata (not from recipe_cells.parquet), arms are recomputed from scratch.

Checks
  A  reproduce the 57-cell population (32 interior / 25 exterior) and the arms MEAN, NEAR, BRACK, LIN, LINHALF,
     LINCLIP; add the arm the recipe actually recommends (half slope AND clamp) and a shrinkage curve.
  B  inference: publication bootstrap, LOPO of the LINHALF-MEAN difference, per-publication contributions, exact
     publication sign-flip test, nested LOPO choice of the shrinkage factor (removes the post-hoc 0.5).
  C  leakage / duplicates: bit-identical log D between target and acid-only sibling (round-3 cross-acid copies
     44012938119033a5/5169c377777d282b and d4c1e23a091d0960/e4ee526bda3920c2), generic scan, drop-and-rescore.
  D  V2 variance decomposition recomputed (0.556 vs 0.270), with a within-group permutation null, level-mean
     collapse and suspect copies removed.
  E  the practical claim "measure 3 acid levels spanning the window instead of 6-8": groups with >= 5 distinct
     levels, keep min / middle / max, predict the rest.
  F  chemotype / cluster composition of the publications.

    .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/recipe4_acid_skA.py
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402
from r3 import datav2 as V2  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import LANTHANIDES, Z_VEC, N_LN  # noqa: E402

OUT = Path(r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2\recipe4_skA_out")
C5 = "pub_6a0ef8528c"
ACID = "cond__acid_concentration_M"
TRIVIAL = ("cond__contact_time_min", "cond__metal_concentration_mM")
SUSPECT = {("44012938119033a5", "5169c377777d282b"), ("d4c1e23a091d0960", "e4ee526bda3920c2")}
RNG = np.random.default_rng(7331)
NBOOT = 2000


def is_suspect(a, b):
    return (a, b) in SUSPECT or (b, a) in SUSPECT


# ----------------------------------------------------------------------------------------------- scoring helpers
class Pooled:
    """Pooled R2 over pair rows; publication bootstrap / LOPO / sign flip on per-publication sums."""

    def __init__(self, t: pd.DataFrame):
        self.t = t.reset_index(drop=True)
        self.pubs = sorted(self.t.publication.unique())
        self.pi = self.t.publication.map({p: q for q, p in enumerate(self.pubs)}).to_numpy()
        P = len(self.pubs)
        self.G = np.zeros((len(self.t), P))
        self.G[np.arange(len(self.t)), self.pi] = 1
        y = self.t.y.to_numpy(float)
        self.N, self.sy, self.syy = np.ones(len(y)) @ self.G, y @ self.G, (y ** 2) @ self.G
        self.W = RNG.multinomial(P, np.ones(P) / P, size=NBOOT).astype(float)

    def sse(self, col):
        return ((self.t[col].to_numpy(float) - self.t.y.to_numpy(float)) ** 2) @ self.G

    def r2(self, col, w=None):
        s = self.sse(col)
        if w is None:
            return 1 - s.sum() / (self.syy.sum() - self.sy.sum() ** 2 / self.N.sum())
        return 1 - (w @ s) / (w @ self.syy - (w @ self.sy) ** 2 / (w @ self.N))

    def summ(self, col, ref=None):
        d = {"R2": self.r2(col)}
        bs = self.r2(col, self.W)
        d["lo"], d["hi"] = np.percentile(bs, [2.5, 97.5])
        if ref is not None:
            dd = bs - self.r2(ref, self.W)
            d["d"] = d["R2"] - self.r2(ref)
            d["d_lo"], d["d_hi"] = np.percentile(dd, [2.5, 97.5])
            d["P_d_gt0"] = float((dd > 0).mean())
            lo = []
            for q in range(len(self.pubs)):
                w = np.ones(len(self.pubs))
                w[q] = 0
                lo.append(float(self.r2(col, w[None])[0] - self.r2(ref, w[None])[0]))
            d["lopo_d_min"], d["lopo_d_max"] = min(lo), max(lo)
            # per-publication SSE gain of col over ref, exact sign-flip on the pooled SSE gain
            g = self.sse(ref) - self.sse(col)
            d["pubs_pos"] = int((g > 0).sum())
            d["pubs_n"] = len(g)
            obs = g.sum()
            flips = np.array(list(itertools.product((-1, 1), repeat=len(g))))
            d["signflip_p_one_sided"] = float(((flips @ np.abs(g)) >= obs - 1e-12).mean()) if len(g) <= 16 else np.nan
            d["top_pub_share_of_gain"] = float(g.max() / obs) if obs > 0 else np.nan
        return d


def fmt(d):
    s = f"{d['R2']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"
    if "d" in d:
        s += (f"  d {d['d']:+.3f} [{d['d_lo']:+.3f}, {d['d_hi']:+.3f}] P>0 {d['P_d_gt0']:.2f} "
              f"LOPO [{d['lopo_d_min']:+.3f}, {d['lopo_d_max']:+.3f}] pubs+ {d['pubs_pos']}/{d['pubs_n']} "
              f"signflip p {d['signflip_p_one_sided']:.3f} top-pub share {d['top_pub_share_of_gain']:.2f}")
    return s


# ----------------------------------------------------------------------------------------------- arms
def arm_values(la_s, v, lt, lam_grid):
    """la_s: sibling log10 acid; v: sibling SF; lt: target log10 acid.  Returns dict of predictions."""
    out = {}
    mean = float(v.mean())
    out["MEAN"] = mean
    dist = np.abs(la_s - lt)
    near = dist <= dist.min() + 1e-9
    out["NEAR_TIE"] = float(v[near].mean())               # nearest level, ties averaged
    out["NEAR_FIRST"] = float(v[int(np.argmin(dist))])     # nearest, first sibling (recipe's E3 rule)
    lo_m, hi_m = la_s < lt - 1e-9, la_s > lt + 1e-9
    if lo_m.any() and hi_m.any():
        xl, xh = la_s[lo_m].max(), la_s[hi_m].min()
        vl = v[np.abs(la_s - xl) < 1e-9].mean()
        vh = v[np.abs(la_s - xh) < 1e-9].mean()
        out["BRACK"] = float(vl + (vh - vl) * (lt - xl) / (xh - xl))
        out["PAIRMEAN"] = float(0.5 * (vl + vh))
    else:
        out["BRACK"] = np.nan
        out["PAIRMEAN"] = np.nan
    b = np.polyfit(la_s, v, 1)
    lin = float(np.polyval(b, lt))
    ltc = min(max(lt, la_s.min()), la_s.max())
    linc = float(np.polyval(b, ltc))
    for lam in lam_grid:
        out[f"L{lam:.2f}"] = mean + lam * (lin - mean)
        out[f"C{lam:.2f}"] = mean + lam * (linc - mean)
    out["LIN"], out["LINCLIP"], out["LINHALF"] = lin, linc, mean + 0.5 * (lin - mean)
    out["LINHALF_CLIP"] = mean + 0.5 * (linc - mean)
    # level-mean OLS (duplicates at one acid level count once)
    ul = np.unique(np.round(la_s, 6))
    if len(ul) >= 2:
        lm = np.array([v[np.abs(np.round(la_s, 6) - u) < 1e-9].mean() for u in ul])
        bl = np.polyfit(ul, lm, 1)
        ml = float(lm.mean())
        out["LVL_LINHALF_CLIP"] = ml + 0.5 * (float(np.polyval(bl, min(max(lt, ul.min()), ul.max()))) - ml)
        out["LVL_MEAN"] = ml
    return out


LAM = (0.0, 0.25, 0.5, 0.75, 1.0)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    Y = np.asarray(bench.Y, float)
    ids = f.cell_id.astype(str).tolist()
    pos = {c: i for i, c in enumerate(ids)}
    pub = f.publication_id.astype(str).to_numpy()
    ext = f.extractant.astype(str).to_numpy()
    chem = f.chemotype.astype(str).to_numpy()
    acid = f[ACID].to_numpy(float)
    condc = [c for c in f.columns if c.startswith("cond__")]
    others = [c for c in condc if c != ACID and c not in TRIVIAL]
    F = f[others].to_numpy(float)
    v2drop = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    c3 = sorted(pd.read_parquet(paths.PRED / "REF_BPG.parquet", columns=["cell_id"]).cell_id.unique())

    def same_other(i, j):
        a, b = F[i], F[j]
        return bool(np.all((np.isnan(a) & np.isnan(b)) | np.isclose(a, b, equal_nan=False)))

    def acid_differs(i, j):
        a, b = acid[i], acid[j]
        if np.isnan(a) and np.isnan(b):
            return False
        return bool(np.isnan(a) or np.isnan(b) or not np.isclose(a, b))

    # ---------------------------------------------------------------- A. population + arms
    rows, links = [], []
    for t in c3:
        i = pos[t]
        if pub[i] == C5 or t in v2drop or not (np.isfinite(acid[i]) and acid[i] > 0):
            continue
        Mi = [m for m in range(N_LN) if np.isfinite(Y[i, m])]
        cover = np.all(np.isfinite(Y[:, Mi]), axis=1)
        sib = [j for j in range(len(ids)) if j != i and pub[j] == pub[i] and ext[j] == ext[i] and cover[j]
               and ids[j] not in v2drop and acid_differs(i, j) and same_other(i, j)
               and np.isfinite(acid[j]) and acid[j] > 0]
        for j in sib:
            dd = np.abs(Y[i, Mi] - Y[j, Mi])
            links.append({"target": t, "sib": ids[j], "publication": pub[i], "frac_identical": float((dd < 1e-9).mean()),
                          "acid_t": acid[i], "acid_s": acid[j], "suspect_r3": is_suspect(t, ids[j])})
        if len(sib) < 2:
            continue
        for variant in ("all", "nosuspect"):
            sv = [j for j in sib if not (variant == "nosuspect" and is_suspect(t, ids[j]))]
            if variant == "nosuspect" and any(t in s for s in SUSPECT):
                pass                                      # target kept; only the copy link is removed
            if len(sv) < 2:
                continue
            la_s = np.log10(acid[sv])
            if np.ptp(la_s) < 1e-9:
                continue
            lt = math.log10(acid[i])
            inside = la_s.min() - 1e-9 <= lt <= la_s.max() + 1e-9
            for p in range(len(Mi) - 1):
                a, b = Mi[p], Mi[p + 1]
                if Z_VEC[b] - Z_VEC[a] != 1:
                    continue
                v = Y[sv, a] - Y[sv, b]
                rec = {"variant": variant, "cell_id": t, "publication": pub[i], "chemotype": chem[i], "extractant": ext[i],
                       "pair": f"{LANTHANIDES[a]}/{LANTHANIDES[b]}", "inside": inside, "n_sib": len(sv),
                       "n_levels": len(np.unique(np.round(la_s, 6))), "y": Y[i, a] - Y[i, b],
                       "dist_out": 0.0 if inside else float(min(abs(lt - la_s.min()), abs(lt - la_s.max())))}
                rec.update(arm_values(la_s, v, lt, LAM))
                rows.append(rec)
    T = pd.DataFrame(rows)
    L = pd.DataFrame(links)
    T.to_parquet(OUT / "recipe4_skA_rows.parquet", index=False)
    L.to_csv(OUT / "recipe4_skA_links.csv", index=False)

    res = {}
    print("=== A/B. acid-only own siblings, C3 targets, no C5, V2-1/2/3 removed (EXPLORATORY)")
    for variant in ("all", "nosuspect"):
        TV = T[T.variant == variant]
        for lab, g in (("interior", TV[TV.inside]), ("exterior", TV[~TV.inside]), ("all", TV)):
            P = Pooled(g)
            print(f"\n--- variant={variant} subset={lab}: cells {g.cell_id.nunique()}, pubs {g.publication.nunique()}, "
                  f"rows {len(g)}")
            arms = ["MEAN", "NEAR_TIE", "NEAR_FIRST", "LIN", "LINHALF", "LINCLIP", "LINHALF_CLIP", "LVL_MEAN",
                    "LVL_LINHALF_CLIP"] + [f"C{x:.2f}" for x in LAM]
            if lab == "interior":
                arms += ["BRACK", "PAIRMEAN"]
            for arm in arms:
                d = P.summ(arm, None if arm == "MEAN" else "MEAN")
                res[f"{variant}|{lab}|{arm}"] = {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                                                 for k, v in d.items()}
                print(f"  {arm:17s} {fmt(d)}")
            # nested LOPO shrinkage (choose lambda on the other publications, clamp always on)
            grid = np.round(np.arange(0, 1.0001, 0.05), 2)
            col = np.zeros(len(g))
            gg = g.reset_index(drop=True)
            lin_c = gg["C1.00"].to_numpy()
            mean_ = gg["MEAN"].to_numpy()
            y = gg.y.to_numpy()
            chosen = {}
            for pb in gg.publication.unique():
                tr = (gg.publication != pb).to_numpy()
                sse = [((mean_[tr] + lam * (lin_c[tr] - mean_[tr]) - y[tr]) ** 2).sum() for lam in grid]
                lam = grid[int(np.argmin(sse))]
                chosen[pb] = float(lam)
                col[~tr] = mean_[~tr] + lam * (lin_c[~tr] - mean_[~tr])
            gg["LOPO_LAM"] = col
            P2 = Pooled(gg)
            d = P2.summ("LOPO_LAM", "MEAN")
            res[f"{variant}|{lab}|LOPO_LAM"] = {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                                               for k, v in d.items()}
            res[f"{variant}|{lab}|LOPO_LAM_chosen"] = chosen
            print(f"  {'LOPO_LAM(clamp)':17s} {fmt(d)}  chosen {sorted(set(chosen.values()))}")
            # full-sample optimum (in-sample, for reference)
            sse_all = [((mean_ + lam * (lin_c - mean_) - y) ** 2).sum() for lam in grid]
            print(f"  in-sample best lambda (clamped) {grid[int(np.argmin(sse_all))]:.2f}")
            # per publication: rows and SSE gain of LINHALF_CLIP over MEAN
            pp = gg.assign(gain=(gg.MEAN - gg.y) ** 2 - (gg.LINHALF_CLIP - gg.y) ** 2,
                           gain_near=(gg.MEAN - gg.y) ** 2 - (gg.NEAR_TIE - gg.y) ** 2) \
                .groupby("publication").agg(cells=("cell_id", "nunique"), rows=("y", "size"), chem=("chemotype", "first"),
                                            gain_LHC=("gain", "sum"), gain_NEAR=("gain_near", "sum"))
            print(pp.round(4).to_string())

    # ---------------------------------------------------------------- C. duplicates
    print("\n=== C. bit-identical log D between C3 target and acid-only sibling (all links, before >= 2 rule)")
    if len(L):
        print(f"links {len(L)}; with >= 50% identical log D: {(L.frac_identical >= 0.5).sum()}; with any identical:"
              f" {(L.frac_identical > 0).sum()}")
        print(L[L.frac_identical > 0].sort_values("frac_identical", ascending=False).head(20).round(3).to_string(index=False))
    # bit-identical SF vectors across the whole bench at a different acid (same pub & extractant)
    # ---------------------------------------------------------------- D. V2 decomposition recomputed
    print("\n=== D. within-campaign acid dependence (all bench cells, V2-1/2/3 and C5 removed)")
    keep = np.array([(c not in v2drop) and (p != C5) for c, p in zip(ids, pub)])
    key = pd.Series([f"{pub[i]}#{ext[i]}#" + "|".join("nan" if np.isnan(x) else f"{x:.6g}" for x in F[i])
                     for i in range(len(ids))])
    groups = {}
    for i in np.where(keep & np.isfinite(acid) & (acid > 0))[0]:
        groups.setdefault(key[i], []).append(i)

    def decomp(drop_suspect=False, level_mean=False, drop_pub=None, nperm=0):
        tot = expl = noise = 0.0
        npts = 0
        gl = []
        perm_expl = np.zeros(nperm)
        for g, ii in groups.items():
            ii = list(ii)
            if drop_pub and pub[ii[0]] == drop_pub:
                continue
            if drop_suspect:
                bad = {c for s in SUSPECT for c in s}
                ii = [q for q in ii if ids[q] not in bad]
            if len(np.unique(np.round(np.log10(acid[ii]), 6))) < 3:
                continue
            gl.append(g)
            la_all = np.log10(acid[ii])
            perms = [RNG.permutation(len(ii)) for _ in range(nperm)]
            for a in range(N_LN - 1):
                if Z_VEC[a + 1] - Z_VEC[a] != 1:
                    continue
                sf = Y[ii, a] - Y[ii, a + 1]
                ok = np.isfinite(sf)
                la, v = la_all[ok], sf[ok]
                if level_mean:
                    ul = np.unique(np.round(la, 6))
                    v = np.array([v[np.round(la, 6) == u].mean() for u in ul])
                    la = ul
                if len(np.unique(np.round(la, 6))) < 3:
                    continue
                ss = ((v - v.mean()) ** 2).sum()
                res_ = v - np.polyval(np.polyfit(la, v, 1), la)
                tot += ss
                expl += ss - (res_ ** 2).sum()
                noise += ss / (len(v) - 1)
                npts += len(v)
                for q, pm in enumerate(perms):
                    if level_mean:
                        break
                    lp = la_all[pm][ok]
                    if len(np.unique(np.round(lp, 6))) < 2:
                        continue
                    rp = v - np.polyval(np.polyfit(lp, v, 1), lp)
                    perm_expl[q] += ss - (rp ** 2).sum()
        out = {"groups": len(gl), "pubs": len({g.split('#')[0] for g in gl}), "points": npts,
               "expl_share": expl / tot, "noise_share": noise / tot}
        if nperm and not level_mean:
            ps = perm_expl / tot
            out["perm_share_mean"] = float(ps.mean())
            out["perm_share_q95"] = float(np.percentile(ps, 95))
        return out

    dres = {"base": decomp(nperm=500), "no_suspect": decomp(drop_suspect=True),
            "level_mean": decomp(level_mean=True)}
    pubs_g = sorted({k.split("#")[0] for k, ii in groups.items()
                     if len(np.unique(np.round(np.log10(acid[ii]), 6))) >= 3})
    for pb in pubs_g:
        dres[f"drop_{pb}"] = decomp(drop_pub=pb)
    for k, v in dres.items():
        print(f"  {k:24s} " + ", ".join(f"{a}={b:.3f}" if isinstance(b, float) else f"{a}={b}" for a, b in v.items()))
    res["V2_decomp"] = dres

    # ---------------------------------------------------------------- E. 3-level design simulation
    print("\n=== E. 'measure 3 levels spanning the window': groups with >= 5 distinct acid levels")
    erows = []
    for g, ii in groups.items():
        ii = [q for q in ii if ids[q] not in {c for s in SUSPECT for c in s}]
        la_all = np.round(np.log10(acid[ii]), 6)
        ul = np.unique(la_all)
        if len(ul) < 5:
            continue
        mid = ul[1:-1][int(np.argmin(np.abs(ul[1:-1] - np.median(ul))))]
        meas = {ul[0], mid, ul[-1]}
        for a in range(N_LN - 1):
            if Z_VEC[a + 1] - Z_VEC[a] != 1:
                continue
            sf = Y[ii, a] - Y[ii, a + 1]
            ok = np.isfinite(sf)
            lv = {u: sf[ok & (la_all == u)] for u in ul}
            if any(len(lv[u]) == 0 for u in meas):
                continue
            xs = np.array(sorted(meas))
            ys = np.array([lv[u].mean() for u in xs])
            mean3 = ys.mean()
            b = np.polyfit(xs, ys, 1)
            # all-but-target levels (the recipe's leave-one-level-out frame), for comparison
            for u in ul:
                if u in meas or len(lv[u]) == 0:
                    continue
                oth = [w for w in ul if w != u and len(lv[w])]
                xo = np.array(oth)
                yo = np.array([lv[w].mean() for w in oth])
                bo = np.polyfit(xo, yo, 1)
                for yv in lv[u]:
                    lin3 = float(np.polyval(b, u))
                    erows.append({"group": g, "publication": g.split("#")[0], "chemotype": chem[ii[0]],
                                  "pair": f"{LANTHANIDES[a]}/{LANTHANIDES[a + 1]}", "y": yv, "MEAN3": mean3,
                                  "LINHALF3": mean3 + 0.5 * (lin3 - mean3), "LIN3": lin3,
                                  "PWL3": float(np.interp(u, xs, ys)),
                                  "NEAR3": float(ys[int(np.argmin(np.abs(xs - u)))]),
                                  "MEAN_LOO": yo.mean(), "LINHALF_LOO": yo.mean() + 0.5 * (np.polyval(bo, u) - yo.mean()),
                                  "GROUPMEAN_ORACLE": float(np.mean([lv[w].mean() for w in ul if len(lv[w])]))})
    E = pd.DataFrame(erows)
    if len(E):
        PE = Pooled(E)
        print(f"rows {len(E)}, groups {E.group.nunique()}, pubs {E.publication.nunique()}, chemotypes "
              f"{sorted(E.chemotype.unique())}")
        for arm in ("MEAN3", "LINHALF3", "LIN3", "PWL3", "NEAR3", "MEAN_LOO", "LINHALF_LOO"):
            d = PE.summ(arm, None if arm == "MEAN3" else "MEAN3")
            res[f"E|{arm}"] = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in d.items()}
            print(f"  {arm:12s} {fmt(d)}")
        # within-group R2: how much of the within-campaign acid variation each arm captures
        E["gp"] = E.group + "|" + E.pair
        cm = E.groupby("gp").y.transform("mean")
        ssw = ((E.y - cm) ** 2).sum()
        print("  within-(group,pair) R2 (share of acid-driven variation recovered; MEAN-like arms ~ 0 or below):")
        for arm in ("MEAN3", "LINHALF3", "LIN3", "PWL3", "NEAR3", "MEAN_LOO", "LINHALF_LOO"):
            # subtract group-pair mean from both: R2 of deviations
            dv = (E[arm] - E.groupby("gp")[arm].transform("mean"))
            print(f"    {arm:12s} SSE/SSw = {((E[arm] - E.y) ** 2).sum() / ssw:.3f}; corr(dev_pred, dev_y) = "
                  f"{np.corrcoef(dv, E.y - cm)[0, 1]:.3f}")
        print("  per publication rows:", E.groupby("publication").size().to_dict())

    # ---------------------------------------------------------------- F. composition
    print("\n=== F. composition of the 57-cell population (variant all)")
    TA = T[T.variant == "all"]
    comp = TA.groupby(["publication", "chemotype", "extractant"]).agg(cells=("cell_id", "nunique"), rows=("y", "size"),
                                                                       inside_cells=("inside", lambda s: int(s.sum())))
    print(comp.to_string())
    print("acid range (M) by publication:")
    for pb, g in TA.groupby("publication"):
        cc = g.cell_id.unique()
        a = acid[[pos[c] for c in cc]]
        print(f"  {pb}: {np.nanmin(a):.4g}-{np.nanmax(a):.4g} M, {len(np.unique(a))} levels among targets")
    (OUT / "recipe4_skA_summary.json").write_text(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
