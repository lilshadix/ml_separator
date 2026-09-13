"""Which strongly directed extractants does the gen14 direction model still get wrong, and why?"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path("generations/gen14_direction").resolve()))
sys.path.insert(0, str(Path("generations/gen13_separation").resolve()))
from gen14 import dirbench as db
from gen14 import models as M
from gen13sep.amplitude_bench import LEAN_BLOCKS

DESIGN = sys.argv[1] if len(sys.argv) > 1 else "BP"
bench = db.load(); FS = db.feature_sets(bench)
lean = bench.columns(LEAN_BLOCKS); X = bench.matrix(LEAN_BLOCKS)
oof = db.run(bench, "LOGIT", M.candidate(M.dir_logistic()), features=FS["TOPO39"], design=DESIGN)
b = oof.cells[oof.cells.n_metals >= db.MIN_METALS].copy()
b["hit"] = ((b.p >= 0.5).astype(int) == b.y).astype(float)
u = b.groupby(["extractant", "chemotype"]).agg(hit=("hit", "mean"), amp=("amp", "mean"),
                                               absamp=("amp", lambda v: v.abs().mean()),
                                               n=("cell_id", "nunique"), p=("p", "mean")).reset_index()
strong = u[u.absamp >= 0.2].sort_values("hit")
print(f"{len(strong)} strongly directed extractants (mean |amp| >= 0.2); "
      f"macro accuracy on them {strong.hit.mean():.4f}")
cols = {c: lean.index(c) for c in lean if c.startswith("coord__donor__n_") or c in
        ("chem__dentate", "coord__dist__donor_pair_min", "coord__dist__frac_donor_pairs_within_3")}
first = {e: np.flatnonzero(bench.frame.extractant.to_numpy() == e)[0] for e in strong.extractant}
info = []
for _, r in strong.iterrows():
    i = first[r.extractant]
    donors = {k.split("__")[-1]: X[i, j] for k, j in cols.items() if X[i, j] and X[i, j] > 0}
    info.append({"hit": round(r.hit, 2), "amp": round(r.amp, 2), "n": r.n, "p_heavy": round(r.p, 2),
                 "smiles": r.extractant[:46], "chemotype": str(r.chemotype)[:16],
                 "features": ", ".join(f"{k}={v:g}" for k, v in sorted(donors.items()))[:78]})
print(pd.DataFrame(info).head(18).to_string(index=False))
print()
wrong = strong[strong.hit < 0.5]
print(f"wrong on {len(wrong)} of {len(strong)} strong extractants; "
      f"they carry {int(wrong.n.sum())} cells; mean |amp| {wrong.absamp.mean():.3f}")
print(f"light-selective share among strong errors {(wrong.amp > 0).mean():.2f} "
      f"vs {(strong.amp > 0).mean():.2f} among all strong")
