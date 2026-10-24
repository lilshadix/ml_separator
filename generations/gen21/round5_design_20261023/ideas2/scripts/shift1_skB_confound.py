"""Skeptic B check for idea shift-1 (EXPLORATORY: all 229 C3 cells already seen; nothing here is confirmatory).
Reads only ideas2/scripts/shift_out/shift_rows.parquet (written by shift_build.py). BPG, A1 = n_AS >= 1.
1. Does the within-publication distance gradient survive splits (extrapolation vs interior, HNO3 only,
   without the organic-acid publication, without the two tail publications, nearest vs mean metric)?
2. Tie structure of |dx_mean| within publication.
3. Near-side read ladder: COPY@0 vs SH@2 / SH@3 / SH@4 / B1@4 / OFF / AINT@0 (cheapest sensible alternatives).
4. Power of the prospective kill (i)/(ii) by resampling families from the corpus as if it were the truth."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
OUT = Path(r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2\scripts\shift_out")
T = pd.read_parquet(OUT / "shift_rows.parquet")
t0 = T[(T.design == "BPG") & (T.n_AS >= 1)].copy()
t0["adxm"] = t0.dx_mean.abs(); t0["adxn"] = t0.dx_near.abs(); t0["extrap"] = t0.extrap.astype(bool)

def r2(d, col):
    return 1 - ((d.y - d[col]) ** 2).sum() / ((d.y - d.y.mean()) ** 2).sum()

def cells_of(d):
    return d.assign(se0=(d.y - d.ACOPY_0) ** 2, seb=(d.y - d.B1_3) ** 2, sesh=(d.y - d.SH_3) ** 2, sei=(d.y - d.AINT_0) ** 2).groupby("cell_id").agg(
        adxm=("adxm", "first"), adxn=("adxn", "first"), ex=("extrap", "first"), below=("dx_mean", lambda s: s.iloc[0] < 0),
        se0=("se0", "mean"), seb=("seb", "mean"), sesh=("sesh", "mean"), sei=("sei", "mean"), pub=("publication", "first"),
        acid=("acid", "first"), x=("x_t", "first"), n=("y", "size"))

def _cr(v):
    v = np.asarray(v, float); return rankdata(v) - (len(v) + 1) / 2

def fam_arrays(e, xcol, col):
    out = []
    for p, g in e.groupby("pub"):
        if len(g) >= 3 and g[xcol].round(6).nunique() >= 2:
            out.append((_cr(g[xcol].to_numpy()), _cr(g[col].to_numpy())))
    return out

def stat_arr(fa, rng=None):
    num = den_a = den_s = 0.0
    for ra, rs in fa:
        if rng is not None:
            ra = rng.permutation(ra)
        num += float(ra @ rs); den_a += float(ra @ ra); den_s += float(rs @ rs)
    return num / np.sqrt(den_a * den_s) if den_a * den_s > 0 else np.nan

def wstat(e, xcol, col):
    return stat_arr(fam_arrays(e, xcol, col))

def wtest(e, xcol, col, nperm=2000, seed=7):
    fa = fam_arrays(e, xcol, col)
    if not fa:
        return np.nan, np.nan, 0, 0
    obs = stat_arr(fa); rng = np.random.default_rng(seed)
    null = np.array([stat_arr(fa, rng) for _ in range(nperm)])
    return obs, float((null >= obs - 1e-12).mean()), len(fa), sum(len(a) for a, _ in fa)

base = t0[~t0.c5]
E = cells_of(base)
print("=== 1. gradient robustness (cell-level within-publication rank corr; one-sided perm p) ===")
subsets = {
    "A1 no C5 (author's, pubs>=2 cells)": E,
    "extrapolation targets only": E[E.ex],
    "interior targets only": E[~E.ex],
    "HNO3 only": E[E.acid == "hno3"],
    "without pub_b0b3296ec5 (organic acids)": E[E.pub != "pub_b0b3296ec5"],
    "without pub_19783c039b and pub_c8195b4ca7": E[~E.pub.isin(["pub_19783c039b", "pub_c8195b4ca7"])],
    "without pub_19783c039b": E[E.pub != "pub_19783c039b"],
    "without pub_c8195b4ca7": E[E.pub != "pub_c8195b4ca7"],
}
for nm, e in subsets.items():
    for xcol in ("adxm", "adxn"):
        o, p, npub, nc = wtest(e, xcol, "se0")
        ob, pb, _, _ = wtest(e, xcol, "seb")
        print(f"{nm:45s} {xcol}: template {o:+.3f} p {p:.3f} | B1@4 control {ob:+.3f} p {pb:.3f} | {npub} pubs {nc} cells")
# extrapolation status confound: MSE by extrap x distance
print("\nrow MSE of ACOPY@0 / B1@4 by extrapolation x |dx_mean| (no C5)")
b = base.assign(bin=pd.cut(base.adxm, [-0.01, 0.30, 3.0], labels=["near<=0.30", "far>0.30"]))
for (ex, bn), g in b.groupby(["extrap", "bin"], observed=True):
    print(f"  extrap={ex!s:5s} {bn:10s} cells {g.cell_id.nunique():3d} pubs {g.publication.nunique():2d} rows/seed {len(g)/5:5.0f} "
          f"MSE copy {((g.y-g.ACOPY_0)**2).mean():.4f} B1@4 {((g.y-g.B1_3)**2).mean():.4f} SH@4 {((g.y-g.SH_3)**2).mean():.4f}")
print("\ncrosstab extrap x near (cells):"); print(pd.crosstab(E.ex, E.adxm <= 0.30))
print("\n=== 2. tie structure: distinct |dx_mean| values per publication (cells) ===")
print(E.groupby("pub").agg(cells=("adxm", "size"), distinct=("adxm", lambda s: s.round(3).nunique()),
      near=("adxm", lambda s: int((s <= 0.30).sum())), far=("adxm", lambda s: int((s > 0.30).sum()))).to_string())

print("\n=== 3. near side (|dx_mean| <= 0.30), no C5, BPG: read ladder on identical rows ===")
near, far = base[base.adxm <= 0.30 + 1e-9], base[base.adxm > 0.30 + 1e-9]
for nm, d in (("near", near), ("far", far)):
    print(nm, {c: round(r2(d, c), 3) for c in ("ACOPY_0", "COPY", "AINT_0", "ACOPY_1", "SH_1", "SH_2", "SH_3", "B1_1", "B1_2", "B1_3", "OFF", "ACOPYSH_1", "ACOPYSH_3")})
def eqw(d, a, bcol):
    v = []
    for p, g in d.groupby("publication"):
        sa, sb = ((g.y - g[a]) ** 2).sum(), ((g.y - g[bcol]) ** 2).sum()
        v.append(1 - sa / sb)
    v = np.array(v); return v.mean(), int((v > 0).sum()), len(v)
for a, bcol in (("ACOPY_0", "B1_3"), ("ACOPY_0", "SH_1"), ("ACOPY_0", "SH_3"), ("SH_1", "B1_3"), ("ACOPY_0", "B1_2")):
    m, k, n = eqw(near, a, bcol); print(f"  near eq-w relSSE {a} vs {bcol}: {m:+.3f} ({k}/{n} pubs +)")
    m, k, n = eqw(far, a, bcol); print(f"  far  eq-w relSSE {a} vs {bcol}: {m:+.3f} ({k}/{n} pubs +)")

print("\n=== 4. power of prospective kill (i)+(ii) if the corpus were the truth ===")
# families = publications with >= m cells having >= 2 distinct distances; draw F families with replacement,
# m cells each without replacement; statistic as in the idea; kill(i): r <= 0.10 or perm p > 0.10; kill(ii): B1 corr >= template corr
Euse = E.groupby("pub").filter(lambda g: len(g) >= 3)
rng = np.random.default_rng(11)
for F in (8, 12, 16):
    for m in (3, 4, 5):
        pubs = [p for p, g in Euse.groupby("pub") if len(g) >= m]
        res = []
        for it in range(400):
            parts = []
            for j, p in enumerate(rng.choice(pubs, F, replace=True)):
                g = Euse[Euse.pub == p]
                g = g.iloc[rng.choice(len(g), m, replace=False)].copy(); g["pub"] = f"{p}#{j}"; parts.append(g)
            e = pd.concat(parts)
            o, pv, _, _ = wtest(e, "adxm", "se0", nperm=400, seed=it)
            ob = wstat(e, "adxm", "seb")
            res.append((o, pv, ob))
        r = np.array(res, dtype=float)
        k1 = (r[:, 0] <= 0.10) | (r[:, 1] > 0.10) | np.isnan(r[:, 0]); k2 = r[:, 2] >= r[:, 0]
        print(f"F={F:2d} families x m={m} cells (pubs eligible {len(pubs)}): P(kill i) {k1.mean():.2f}  P(kill ii) {k2.mean():.2f}  "
              f"P(killed by i or ii) {(k1 | k2).mean():.2f}; median r {np.nanmedian(r[:,0]):.2f}")
