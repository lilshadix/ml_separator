import sys
sys.path.insert(0, "D:/ml_separator_gh/generations/gen13_separation/analysis/stage3/verify_direction")
from v_runner import *
import numpy as np, pandas as pd
from gen13sep.basis import fit_coefficients

rich = FRAME.n_metals.to_numpy() >= 5
amp = COEF[:, 0]
y = (amp < 0).astype(int)
fr = FRAME.copy(); fr["amp"] = amp; fr["y"] = y; fr["nm"] = FRAME.n_metals.to_numpy()
sub = fr[rich]

print("=== A. label margin ===")
a = np.abs(sub.amp.to_numpy())
for q in (0.05,0.1,0.25,0.5,0.75,0.9):
    print(f"  |amp| q{q:.2f} = {np.quantile(a,q):.4f}")
for b in (0.05,0.1,0.2,0.3,0.5):
    print(f"  frac |amp| < {b}: {float((a<b).mean()):.3f}  ({int((a<b).sum())} of {len(a)} cells)")
print("  amp sd over rich cells:", float(sub.amp.std()))

print("\n=== B. OLS vs ridge label ===")
centred = Y14 - np.nanmean(Y14, axis=1, keepdims=True)
ols = np.full(len(Y14), np.nan)
for i in range(len(Y14)):
    m = ~np.isnan(centred[i])
    if m.sum() >= 2:
        B = BASIS[:, m].T
        ols[i] = np.linalg.lstsq(B, centred[i][m], rcond=None)[0][0]
y_ols = (ols < 0).astype(int)
print("  disagreements ridge vs OLS on rich cells:", int((y_ols[rich] != y[rich]).sum()), "of", int(rich.sum()))
print("  max |ridge-ols| on rich:", float(np.nanmax(np.abs(ols[rich]-amp[rich]))))

print("\n=== C. within-extractant label consistency (rich cells) ===")
g = sub.groupby("extractant")["y"]
frac_pure = float((g.nunique() == 1).mean())
print("  extractants with a single sign across all their cells:", f"{frac_pure:.3f}",
      f"({int((g.nunique()==1).sum())} of {sub.extractant.nunique()})")
cc = sub.groupby("extractant").size()
print("  extractants with exactly 1 rich cell:", int((cc==1).sum()), "of", len(cc))
print("  cell-count distribution per extractant:", cc.value_counts().sort_index().to_dict())
print("  extractant-macro heavy fraction:", float(sub.groupby("extractant")["y"].mean().mean()))
print("  chemotype-macro heavy fraction:",
      float(sub.groupby("extractant")["y"].mean().groupby(sub.groupby("extractant")["chemotype"].first()).mean().mean()))

print("\n=== D. chemotype sizes (rich cells) ===")
cs = sub.groupby("chemotype").agg(n_cells=("cell_id","size"), n_ext=("extractant","nunique"))
print(cs.sort_values("n_cells", ascending=False).head(10).to_string())
print("  n chemotypes:", len(cs), " singleton-extractant chemotypes:", int((cs.n_ext==1).sum()))
print("  full cohort chemotype sizes top5:", FRAME.groupby("chemotype").size().sort_values(ascending=False).head(5).to_dict())

print("\n=== E. topology features constant within extractant? ===")
tp = ALLX.loc[:, TOPO_COLS].copy(); tp["e"] = FRAME.extractant.to_numpy()
nun = tp.groupby("e").nunique(dropna=False).max()
print("  max distinct values of any topo column within an extractant:", int(nun.max()))
print("  distinct topo feature rows:", ALLX.loc[:, TOPO_COLS].drop_duplicates().shape[0],
      "over", FRAME.extractant.nunique(), "extractants")
print("  distinct topo rows among rich extractants:",
      ALLX.loc[rich, TOPO_COLS].assign(e=FRAME.extractant[rich].to_numpy()).drop_duplicates(TOPO_COLS).shape[0])
print("  topo NaN rows (rich):", int(ALLX.loc[rich, TOPO_COLS].isna().all(axis=1).sum()))

print("\n=== F. threshold sensitivity: cohort shape ===")
for thr in (2,4,5,6,8,14):
    m = FRAME.n_metals.to_numpy() >= thr
    s = fr[m]
    print(f"  >= {thr}: cells={int(m.sum())} ext={s.extractant.nunique()} chem={s.chemotype.nunique()} "
          f"heavy_ext_macro={float(s.groupby('extractant')['y'].mean().mean()):.4f}")
