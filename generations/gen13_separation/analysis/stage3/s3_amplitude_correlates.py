"""Which single descriptors carry the amplitude, and does a stoichiometric multiplier explain it?"""
import sys
sys.path.insert(0, "generations/gen13_separation")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from gen13sep.amplitude_bench import load_bench, LEAN_BLOCKS, CHEM_BLOCKS

bench = load_bench()
f = bench.frame.copy()
f["amp"] = bench.coef[:, 0]
f["curv"] = bench.coef[:, 1]
rich = f[f.n_metals >= 5]
ext = rich.groupby("extractant").agg(amp=("amp", "mean"), curv=("curv", "mean"),
                                     n=("amp", "size"), chemo=("chemotype", "first"))
print(f"extractants with a well-determined amplitude: {len(ext)}  (cells n_metals>=5: {len(rich)})")

X = pd.DataFrame(bench.matrix(LEAN_BLOCKS), columns=bench.columns(LEAN_BLOCKS), index=f.index)
Xr = X.loc[rich.index].groupby(rich["extractant"].to_numpy()).mean()
Xr = Xr.loc[ext.index]

rows = []
for c in Xr.columns:
    v = Xr[c]
    if v.notna().sum() < 25 or v.nunique(dropna=True) < 3:
        continue
    ok = v.notna()
    rows.append({"feature": c, "n": int(ok.sum()),
                 "spearman_amp": float(spearmanr(v[ok], ext.amp[ok]).statistic),
                 "spearman_curv": float(spearmanr(v[ok], ext.curv[ok]).statistic)})
cor = pd.DataFrame(rows).sort_values("spearman_amp", key=abs, ascending=False)
cor.to_csv("generations/gen13_separation/analysis/stage3/s3_amplitude_correlates.csv", index=False)
print("\ntop 18 by |Spearman| with the extractant-level amplitude:")
print(cor.head(18).round(3).to_string(index=False))

print("\n--- stoichiometric multiplier ---")
for name, cols in [("n_ligands", ["chem__n_ligands"]),
                   ("dentate", ["chem__dentate"]),
                   ("core_cn", ["chem__core_cn"]),
                   ("n_ligands x dentate", ["chem__n_ligands", "chem__dentate"]),
                   ("donor total", ["chem__donor__n_total"])]:
    if not all(c in Xr.columns for c in cols):
        print(f"  {name:22s} columns missing"); continue
    m = Xr[cols].prod(axis=1)
    ok = m.notna() & (m > 0)
    if ok.sum() < 20:
        print(f"  {name:22s} too few"); continue
    rho = spearmanr(m[ok], ext.amp[ok]).statistic
    # does dividing the amplitude by the multiplier reduce its spread?
    s = ext.amp[ok] / m[ok]
    print(f"  {name:22s} n={int(ok.sum()):3d}  spearman(amp) {rho:+.3f}   "
          f"cv(amp) {abs(ext.amp[ok].std()/ext.amp[ok].mean()):.2f} -> cv(amp/mult) {abs(s.std()/s.mean()):.2f}")

print("\n--- how much of the amplitude is between chemotypes? ---")
g = ext.groupby("chemo")["amp"]
ssb = float((g.size() * (g.mean() - ext.amp.mean()) ** 2).sum())
ssw = float(((ext.amp - ext.chemo.map(g.mean())) ** 2).sum())
print(f"  {len(g)} chemotypes over {len(ext)} extractants: between {ssb/(ssb+ssw):.3f} of SS")
