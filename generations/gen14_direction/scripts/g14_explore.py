"""Facts gen14's design depends on, measured once on the frozen cohort."""
import sys
sys.path.insert(0, "gen13_separation")
import numpy as np, pandas as pd
from gen13sep.amplitude_bench import load_bench, LEAN_BLOCKS

bench = load_bench()
f = bench.frame
amp = bench.coef[:, 0]
rich = f.n_metals.to_numpy() >= 5
print(f"cells {len(f)}  rich {rich.sum()}  extractants {f.extractant.nunique()} "
      f"(rich {f.extractant[rich].nunique()})  chemotypes {f.chemotype.nunique()}")
print(f"publications {f.publication_id.nunique()}")

# 1. how consistent is the direction inside one extractant?
d = pd.DataFrame({"ext": f.extractant, "chem": f.chemotype, "amp": amp,
                  "n_metals": f.n_metals, "pub": f.publication_id})[rich]
g = d.groupby("ext").agg(n=("amp", "size"), pos=("amp", lambda v: (v > 0).sum()),
                         mean_amp=("amp", "mean"), sd_amp=("amp", "std"),
                         min_amp=("amp", "min"), max_amp=("amp", "max"))
mixed = g[(g.pos > 0) & (g.pos < g.n)]
print(f"\nextractants with >=2 rich cells: {(g.n >= 2).sum()};  sign-mixed: {len(mixed)}")
print(f"cells living in a sign-mixed extractant: {d.ext.isin(mixed.index).sum()} / {len(d)}")
print(mixed.round(3).to_string())

# 2. label reliability
print(f"\n|amp| quantiles: {np.round(np.quantile(np.abs(amp[rich]), [.1,.25,.5,.75,.9]), 3)}")
for lo in (0.0, 0.05, 0.1, 0.2):
    m = rich & (np.abs(amp) >= lo)
    print(f"  |amp|>={lo:<4}: cells {m.sum():3d}  ext {f.extractant[m].nunique():3d}  "
          f"heavy-frac(cells) {(amp[m] < 0).mean():.3f}")

# 3. topology resolution
lean = bench.columns(LEAN_BLOCKS); X = bench.matrix(LEAN_BLOCKS)
GEOM = [c for c in lean if c.startswith("coord__dist__") or c.startswith("coord__arm__")]
gi = np.array([lean.index(c) for c in GEOM])
T = X[:, gi]
uniq = np.unique(np.nan_to_num(T[rich], nan=-999), axis=0)
print(f"\ntopology columns {len(GEOM)}; distinct vectors over rich cells {len(uniq)}; "
      f"over rich extractants {len(np.unique(np.nan_to_num(pd.DataFrame(T[rich]).groupby(f.extractant[rich].to_numpy()).first().to_numpy(), nan=-999), axis=0))}")
print("columns:", GEOM)

# 4. what else is in COORD / how many coord columns overall
COORD = [c for c in lean if c.startswith("coord__")]
print(f"\ncoord columns total {len(COORD)}")
print([c for c in COORD if c not in GEOM])

# 5. condition columns that could scale magnitude
print("\nCOND/MASSACT columns:", [c for c in lean if c.startswith('massact__')])
