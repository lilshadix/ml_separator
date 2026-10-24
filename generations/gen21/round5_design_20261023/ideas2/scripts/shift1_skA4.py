"""Skeptic A for shift-1 (EXPLORATORY): label-free re-derivation on clean corpus acid-series families (connected
components of the acid-only relation, V2-clean, no C5, common metal set >= 5, >= 3 levels spanning >= 0.5 dec).
Leave-one-level-out mean template; cell MSE over strict-adjacent pairs.  Is the gradient distance or position?
Prints only."""
import importlib.util, itertools
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sk", HERE / "shift1_skA.py"); sk = importlib.util.module_from_spec(spec); spec.loader.exec_module(sk)
from r3 import datav2 as V2
from gen20k import data as D
b = D.load_bench(); f = b.frame.reset_index(drop=True); Y = np.asarray(b.Y, float)
ids = f.cell_id.astype(str).to_numpy(); pub = f.publication_id.astype(str).to_numpy(); ext = f.extractant.astype(str).to_numpy()
x = np.log10(f.cond__acid_concentration_M.astype(float).to_numpy())
cond = [c for c in f.columns if c.startswith("cond__")]
Cm = f[cond].apply(pd.to_numeric, errors="coerce").to_numpy(float); fams = np.array([sk.family(c) for c in cond])
drop = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
def diff(i, j):
    a, c = Cm[i], Cm[j]; na, nc = np.isnan(a), np.isnan(c)
    return set(fams[(na != nc) | (~na & ~nc & ~np.isclose(a, c))])
cand = [i for i in range(len(f)) if ids[i] not in drop and pub[i] != sk.C5 and np.isfinite(x[i]) and np.isfinite(Y[i]).sum() >= 5]
adj = {i: set() for i in cand}
for i, j in itertools.combinations(cand, 2):
    if pub[i] == pub[j] and ext[i] == ext[j] and (diff(i, j) - {"contact_time", "metal_conc"}) == {"acid_conc"}:
        adj[i].add(j); adj[j].add(i)
seen, rows = set(), []
for i in cand:
    if i in seen or not adj[i]:
        continue
    st, cc = [i], set()
    while st:
        u = st.pop()
        if u in cc: continue
        cc.add(u); st.extend(adj[u] - cc)
    seen |= cc; cc = sorted(cc)
    met = np.where(np.all(np.isfinite(Y[cc]), axis=0))[0]
    lv = np.unique(np.round(x[cc], 3))
    if len(met) < 5 or len(lv) < 3 or np.ptp(lv) < 0.5:
        continue
    pairs = [(q, q + 1) for q in range(len(met) - 1) if met[q + 1] == met[q] + 1]
    for a in cc:
        oth = [c for c in cc if c != a]
        T = Y[oth][:, met].mean(0); yy = Y[a, met]
        se = np.mean([((yy[p] - yy[q]) - (T[p] - T[q])) ** 2 for p, q in pairs])
        xo = x[oth]
        posn = "inside" if xo.min() - 1e-9 <= x[a] <= xo.max() + 1e-9 else ("below" if x[a] < xo.min() else "above")
        rows.append({"fam": f"{pub[i]}|{ext[i][:12]}|{i}", "pub": pub[i], "cell": ids[a], "adx": abs(x[a] - xo.mean()), "se0": se,
                     "pos": posn, "below": float(posn == "below"), "n_lv": len(lv), "n_met": len(met)})
R = pd.DataFrame(rows); print("dropped NaN-MSE cells:", int(R.se0.isna().sum())); R = R[R.se0.notna()].copy()
print(f"families {R.fam.nunique()} (pubs {R.pub.nunique()}), cells {len(R)}; positions {R.pos.value_counts().to_dict()}")
print(R.groupby("fam").agg(cells=("cell", "size"), n_met=("n_met", "first"), rho=("se0", lambda s: np.nan)).drop(columns="rho").to_string())
def ok(df, unit, col="adx"):
    return df.groupby(unit).filter(lambda g: len(g) >= 2 and g[col].nunique() >= 2)
o, p = sk.perm_p(ok(R, "fam"), "adx", "se0", unit="fam", n=5000); print(f"|dx| within family: {o:+.3f} p {p:.4f}")
R["fp"] = R.fam + "|" + R.pos
e = ok(R, "fp"); o, p = sk.perm_p(e, "adx", "se0", unit="fp", n=5000); print(f"|dx| within family x position: {o:+.3f} p {p:.4f} ({e.fp.nunique()} units, {len(e)} cells)")
e = ok(R[R.pos == "inside"], "fam"); o, p = sk.perm_p(e, "adx", "se0", unit="fam", n=5000); print(f"|dx| inside only: {o:+.3f} p {p:.4f} ({e.fam.nunique()} fams, {len(e)} cells)")
e = ok(R[R.pos != "below"], "fam"); o, p = sk.perm_p(e, "adx", "se0", unit="fam", n=5000); print(f"|dx| without below: {o:+.3f} p {p:.4f} ({e.fam.nunique()} fams, {len(e)} cells)")
e = ok(R, "fam", "below"); o, p = sk.perm_p(e, "below", "se0", unit="fam", n=5000); print(f"below flag: {o:+.3f} p {p:.4f}")
R["above"] = (R.pos == "above").astype(float)
e = ok(R, "fam", "above"); o, p = sk.perm_p(e, "above", "se0", unit="fam", n=5000); print(f"above flag: {o:+.3f} p {p:.4f}")
R["ext"] = (R.pos != "inside").astype(float)
e = ok(R, "fam", "ext"); o, p = sk.perm_p(e, "ext", "se0", unit="fam", n=5000); print(f"extrapolation flag: {o:+.3f} p {p:.4f}")
print(R.groupby("pos").agg(cells=("cell", "size"), adx=("adx", "mean"), mse=("se0", "mean"), med=("se0", "median")).round(4).to_string())
print(R.assign(near=R.adx <= 0.3).groupby(["pos", "near"]).agg(cells=("cell", "size"), mse=("se0", "mean"), med=("se0", "median")).round(4).to_string())
# per-family Spearman
from scipy.stats import spearmanr
print(R.groupby("fam").apply(lambda g: pd.Series({"n": len(g), "rho": spearmanr(g.adx, g.se0)[0]}), include_groups=False).round(3).to_string())
