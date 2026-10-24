"""Skeptic B check for idea shift-1, part 2 (EXPLORATORY: 229 already-seen C3 cells). BPG, A1 (n_AS >= 1), no C5.
(a) control-adjusted gradient: within-pub rank corr of distance with rank(se_copy) - rank(se_B1@4), permutation test;
(b) value of the first 2 reads vs distance: within-pub rank corr of distance with se_copy - se_SH@2;
(c) far side split by direction (target below vs above the siblings' mean acid);
(d) per-publication near/far R2-free summary for HNO3."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
OUT = Path(r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2\scripts\shift_out")
T = pd.read_parquet(OUT / "shift_rows.parquet")
t = T[(T.design == "BPG") & (T.n_AS >= 1) & (~T.c5)].copy()
t["adxm"] = t.dx_mean.abs()
E = t.assign(se0=(t.y - t.ACOPY_0) ** 2, seb=(t.y - t.B1_3) ** 2, sesh2=(t.y - t.SH_1) ** 2).groupby("cell_id").agg(
    adxm=("adxm", "first"), dxm=("dx_mean", "first"), se0=("se0", "mean"), seb=("seb", "mean"), sesh2=("sesh2", "mean"),
    pub=("publication", "first"), acid=("acid", "first"))
def cr(v): return rankdata(v) - (len(v) + 1) / 2
def fams(e, f):
    out = []
    for p, g in e.groupby("pub"):
        if len(g) >= 3 and g.adxm.round(6).nunique() >= 2:
            out.append((cr(g.adxm.to_numpy()), f(g)))
    return out
def stat(fa, rng=None):
    n = a = b = 0.0
    for ra, rs in fa:
        if rng is not None: ra = rng.permutation(ra)
        n += ra @ rs; a += ra @ ra; b += rs @ rs
    return n / np.sqrt(a * b)
def test(e, f, nperm=5000):
    fa = fams(e, f); o = stat(fa); rng = np.random.default_rng(3)
    nl = np.array([stat(fa, rng) for _ in range(nperm)]); return o, (nl >= o - 1e-12).mean(), len(fa), sum(len(x) for x, _ in fa)
diff_rank = lambda g: cr(g.se0.to_numpy()) - cr(g.seb.to_numpy())
voi2 = lambda g: cr((g.se0 - g.sesh2).to_numpy())
for nm, e in (("all acids", E), ("HNO3 only", E[E.acid == "hno3"]), ("without pub_19783c039b", E[E.pub != "pub_19783c039b"])):
    o, p, k, n = test(e, diff_rank); print(f"[{nm}] control-adjusted (rank se_copy - rank se_B1@4): {o:+.3f} p {p:.3f} ({k} pubs, {n} cells)")
    o, p, k, n = test(e, voi2); print(f"[{nm}] value of 2 reads (se_copy - se_SH@2): {o:+.3f} p {p:.3f} ({k} pubs, {n} cells)")
def r2(d, c): return 1 - ((d.y - d[c]) ** 2).sum() / ((d.y - d.y.mean()) ** 2).sum()
far = t[t.adxm > 0.30 + 1e-9]
for side, d in (("far, target below siblings' mean acid", far[far.dx_mean < 0]), ("far, target above", far[far.dx_mean > 0])):
    print(f"{side}: cells {d.cell_id.nunique()} pubs {d.publication.nunique()} rows/seed {len(d)/5:.0f}; R2 copy@0 {r2(d,'ACOPY_0'):.3f} "
          f"B1@4 {r2(d,'B1_3'):.3f} SH@2 {r2(d,'SH_1'):.3f} SH@4 {r2(d,'SH_3'):.3f}; MSE copy {((d.y-d.ACOPY_0)**2).mean():.4f} B1@4 {((d.y-d.B1_3)**2).mean():.4f}")
near = t[t.adxm <= 0.30 + 1e-9]
for side, d in (("near, below", near[near.dx_mean < 0]), ("near, above", near[near.dx_mean > 0]), ("near, |dx|<1e-9", near[near.dx_mean.abs() < 1e-9])):
    if len(d):
        print(f"{side}: cells {d.cell_id.nunique()} pubs {d.publication.nunique()}; MSE copy {((d.y-d.ACOPY_0)**2).mean():.4f} B1@4 {((d.y-d.B1_3)**2).mean():.4f}")
