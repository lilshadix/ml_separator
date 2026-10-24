"""shift theme, step 11 (EXPLORATORY): is the 'light Ln less acid-sensitive' slope pattern a detection-floor /
saturation artefact?  Per acid-series group (shift_slopes.py grouping, >= 10 metal slopes), recompute per-metal
log D acid slopes using only cells whose log D for that metal lies in [-2, 2.5] (needs >= 2 such cells spanning
>= 0.1 decade), and report Spearman(metal mean log D, metal slope) within group."""
import sys
sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from r4 import paths  # noqa
from r3 import datav2 as V2
from gen20k import data as D
from gen20k.data import LANTHANIDES as LN, N_LN
pd.set_option("display.width", 250)
b = D.load_bench(); f = b.frame.reset_index(drop=True); Y = np.asarray(b.Y, float)
v2 = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
def lab(r, pre): return "+".join(c[len(pre):] for c in f.columns if c.startswith(pre) and r[c] == 1)
f["k"] = [(r.publication_id, r.extractant, lab(r, "cond__acid__"), lab(r, "cond__diluent__"), lab(r, "cond__additive__"),
           round(float(r.cond__extractant_concentration_M), 3), str(r.cond__temperature_C)) for _, r in f.iterrows()]
f["x"] = np.log10(f.cond__acid_concentration_M.astype(float))
keep = ~f.cell_id.isin(v2) & f.x.notna()
def slope(x, y):
    ok = np.isfinite(y); x, y = x[ok], y[ok]
    if len(x) < 2 or np.ptp(x) < 0.1: return np.nan
    xc = x - x.mean(); return float((xc * (y - y.mean())).sum() / (xc ** 2).sum())
rows = []
for k, g in f[keep].groupby("k"):
    if len(g) < 2 or np.ptp(g.x) < 0.1: continue
    idx = g.index.to_numpy(); x = g.x.to_numpy()
    b_all = np.array([slope(x, Y[idx, m]) for m in range(N_LN)])
    if np.isfinite(b_all).sum() < 10: continue
    Yw = Y[idx].copy(); Yw[(Yw < -2) | (Yw > 2.5)] = np.nan
    b_w = np.array([slope(x, Yw[:, m]) for m in range(N_LN)])
    lev = np.nanmean(Y[idx], 0)
    ok = np.isfinite(b_all)
    light = [0, 1]
    ca = b_all - np.nanmean(b_all); cw = b_w - np.nanmean(b_w)
    rows.append({"pub": k[0][:14], "acid": k[2][:8], "cmin": 10 ** x.min(), "cmax": 10 ** x.max(),
                 "La_lev": lev[0], "Ce_lev": lev[1], "Yb_lev": lev[12],
                 "LaCe_centred_all": np.nanmean(ca[light]), "LaCe_centred_window": np.nanmean(cw[light]) if np.isfinite(cw[light]).any() else np.nan,
                 "n_metal_window": int(np.isfinite(b_w).sum()),
                 "rho(level,slope)": spearmanr(lev[ok], b_all[ok])[0],
                 "rho_window": spearmanr(lev[np.isfinite(b_w)], b_w[np.isfinite(b_w)])[0] if np.isfinite(b_w).sum() >= 5 else np.nan,
                 "frac_cells_out_window_La": float(np.mean((Y[idx, 0] < -2) | (Y[idx, 0] > 2.5))) if np.isfinite(Y[idx, 0]).any() else np.nan})
R = pd.DataFrame(rows)
print(R.round(2).to_string(index=False))
print(f"\ngroups {len(R)}; La/Ce centred slope < 0: all-cells {int((R.LaCe_centred_all < 0).sum())}/{len(R)}; "
      f"window-only {int((R.LaCe_centred_window < 0).sum())}/{int(R.LaCe_centred_window.notna().sum())}")
