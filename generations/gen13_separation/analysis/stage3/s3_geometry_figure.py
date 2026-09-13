import sys
sys.path.insert(0, "generations/gen13_separation")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from gen13sep.amplitude_bench import load_bench, LEAN_BLOCKS

bench = load_bench()
f = bench.frame.copy(); f["amp"] = bench.coef[:, 0]
rich = f[f.n_metals >= 5]
X = pd.DataFrame(bench.matrix(LEAN_BLOCKS), columns=bench.columns(LEAN_BLOCKS), index=f.index)
ext = rich.groupby("extractant").agg(amp=("amp", "mean"), n=("amp", "size"),
                                     chemo=("chemotype", "first"), name=("extractant_name", "first"))
Xe = X.loc[rich.index].groupby(rich["extractant"].to_numpy()).mean().loc[ext.index]

PANELS = [("coord__dist__frac_donor_pairs_within_3",
           "fraction of donor pairs within 3 bonds\n(compact bite = right)", -0.530),
          ("coord__dist__donor_pair_min",
           "shortest donor-to-donor path (bonds)\n(compact bite = left)", 0.480)]
LABEL = {"TODGA": "TODGA", "DOODA (C12)": "DOODA(C12)", "TEDGA": "TEDGA",
         "C5BTBP": "C5-BTBP", "DMDPhPDA": "DMDPhPDA", "DHD2DGA": "DHD2DGA"}

fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.9))
for ax, (col, xlabel, rho) in zip(axes, PANELS):
    v = Xe[col].to_numpy(float); a = ext.amp.to_numpy(); ok = np.isfinite(v) & np.isfinite(a)
    dga = (ext.chemo == "sc009").to_numpy()
    ax.axhline(0, color="0.4", lw=0.8, zorder=0)
    ax.scatter(v[ok & ~dga], a[ok & ~dga], s=34, c="#2c7fb8", alpha=.85,
               edgecolor="white", lw=.5, label="other chemotypes")
    ax.scatter(v[ok & dga], a[ok & dga], s=34, c="#e08214", alpha=.85, marker="s",
               edgecolor="white", lw=.5, label="diglycolamides (sc009)")
    g = pd.DataFrame({"c": ext.chemo[ok], "v": v[ok], "a": a[ok]}).groupby("c").mean()
    ax.scatter(g.v, g.a, s=95, facecolor="none", edgecolor="0.25", lw=1.2, zorder=3,
               label="chemotype means")
    z = np.polyfit(v[ok], a[ok], 1); xs = np.linspace(np.nanmin(v), np.nanmax(v), 50)
    ax.plot(xs, np.polyval(z, xs), color="0.25", lw=1.4, ls="--", zorder=2)
    for nm, tag in LABEL.items():
        hit = ext.index[ext.name == nm]
        if len(hit) and np.isfinite(Xe.loc[hit[0], col]):
            ax.annotate(tag, (Xe.loc[hit[0], col], ext.loc[hit[0], "amp"]),
                        textcoords="offset points", xytext=(6, 5), fontsize=7.5, color="0.2")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(f"Spearman {rho:+.2f}   (n = {int(ok.sum())} extractants, "
                 f"{ext.chemo[ok].nunique()} chemotypes)", fontsize=9.5)
    ax.grid(alpha=.25)
axes[0].set_ylabel("selectivity amplitude of the extractant\n"
                   "(coefficient on standardised Shannon radius;\nnegative = heavy-lanthanide selective)",
                   fontsize=9)
axes[0].legend(fontsize=8, loc="upper left")
fig.suptitle("A compact donor set makes an extractant heavy-lanthanide selective\n"
             "donor separations are counts of bonds along the molecular graph, so this is computable "
             "for any candidate ligand without a 3D structure", fontsize=11)
fig.tight_layout()
fig.savefig("generations/gen13_separation/figures/stage2/s3_donor_compactness.png", dpi=160, bbox_inches="tight")
print("written")
