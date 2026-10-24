"""shift theme, step 10 (EXPLORATORY): slope analysis in extractant concentration.  Groups = publication x extractant
x every recorded condition except extractant concentration (and contact time / metal concentration); >= 2 levels
spanning >= 0.1 decade; per-metal OLS slope n_m = dlogD_m / dlog10[L] (apparent solvation number), V2-clean cells.
For the C5-BTBP acid x [L] grid the slope is fitted jointly with log10[acid] (additive model)."""
import sys
sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
import numpy as np, pandas as pd
from r4 import paths  # noqa
from r3 import datav2 as V2
from gen20k import data as D
from gen20k.data import LANTHANIDES as LN, N_LN
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
b = D.load_bench(); f = b.frame.reset_index(drop=True); Y = np.asarray(b.Y, float)
v2 = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
def lab(r, pre): return "+".join(c[len(pre):] for c in f.columns if c.startswith(pre) and r[c] == 1)
f["k"] = [(r.publication_id, r.extractant, lab(r, "cond__acid__"), lab(r, "cond__diluent__"), lab(r, "cond__additive__"),
           round(float(r.cond__acid_concentration_M), 3), str(r.cond__temperature_C)) for _, r in f.iterrows()]
f["zL"] = np.log10(f.cond__extractant_concentration_M.astype(float))
keep = ~f.cell_id.isin(v2)
rows = []
def slope(x, y):
    ok = np.isfinite(y); x, y = x[ok], y[ok]
    if len(x) < 2 or np.ptp(x) < 0.1: return np.nan
    xc = x - x.mean(); return float((xc * (y - y.mean())).sum() / (xc ** 2).sum())
for k, g in f[keep].groupby("k"):
    if len(g) < 2 or np.ptp(g.zL) < 0.1: continue
    idx = g.index.to_numpy()
    n = np.array([slope(g.zL.to_numpy(), Y[idx, m]) for m in range(N_LN)])
    if np.isfinite(n).sum() < 3: continue
    rows.append({"pub": k[0][:14], "ext": str(g.extractant_name.iloc[0])[:12], "acid": k[2], "acidM": k[5], "levels": int(len(np.unique(np.round(g.zL, 3)))),
                 **{LN[m]: n[m] for m in range(N_LN)}, "mean_n": np.nanmean(n), "sd_n": np.nanstd(n)})
R = pd.DataFrame(rows)
print(R.round(2).to_string(index=False))
# joint acid x L fit for C5-BTBP
g = f[keep & (f.publication_id == "pub_6a0ef8528c")]
X = np.c_[np.ones(len(g)), np.log10(g.cond__acid_concentration_M.astype(float)), g.zL]
idx = g.index.to_numpy()
out = {}
for m in range(N_LN):
    y = Y[idx, m]; ok = np.isfinite(y)
    if ok.sum() >= 4:
        beta, res, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
        r = y[ok] - X[ok] @ beta
        out[LN[m]] = {"b_acid": beta[1], "n_L": beta[2], "rmse": float(np.sqrt((r ** 2).mean()))}
print("\nC5-BTBP additive per-metal model logD = a + b*log10[HNO3] + n*log10[L] (15 cells):")
print(pd.DataFrame(out).T.round(3).to_string())
