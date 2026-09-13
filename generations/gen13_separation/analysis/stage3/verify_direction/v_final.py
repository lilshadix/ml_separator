import sys, json, time
sys.path.insert(0, "D:/ml_separator_gh/gen13_separation/analysis/stage3/verify_direction")
from v_runner import *
import numpy as np, pandas as pd
from gen13sep.splits import all_folds

AMP = COEF[:, 0]; NM = FRAME.n_metals.to_numpy(); Y = (AMP < 0).astype(int); RICH = NM >= 5
CHEM137 = [c for b in ("PHYSCHEM","DONORS","COORD") for c in FRAMES[b].columns]
DON = list(FRAMES["DONORS"].columns)
t0 = time.time()

runs = {}
for tag, cols, est in (("topo39_et", TOPO_COLS, "et"), ("donors13_et", DON, "et"),
                       ("lean209_et", LEAN_COLS, "et"), ("chem137_et", CHEM137, "et"),
                       ("topo39_lg", TOPO_COLS, "lg"), ("lean209_lg", LEAN_COLS, "lg"),
                       ("chem137_lg", CHEM137, "lg"), ("donors13_lg", DON, "lg")):
    runs[tag] = units(run(cols, Y, RICH, design="BP", est=est))
    print(tag, round(float(runs[tag].hit.mean()), 4), flush=True)

print("\n--- paired chemotype-blocked contrasts (design BP) ---")
pairs = [("topo39_et","donors13_et"), ("topo39_et","lean209_et"), ("topo39_et","chem137_et"),
         ("topo39_lg","donors13_lg"), ("topo39_lg","lean209_lg"), ("topo39_lg","chem137_lg")]
rows = []
for a, b in pairs:
    r = boot(runs[a], runs[b], reps=10000)
    rows.append({"contrast": f"{a} - {b}", **r}); print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(OUT / "v_contrasts.csv", index=False)

# --- permutation null: shuffle the label ACROSS EXTRACTANTS (keeping each extractant's
#     cells together), refit, and see how often the chemotype-blocked gain exceeds +0.21 ---
print("\n--- permutation null (extractant-level label shuffle) ---", flush=True)
rng = np.random.default_rng(4242)
ext = FRAME.extractant.to_numpy()
rich_ext = pd.Series(Y[RICH]).groupby(pd.Series(ext[RICH])).mean()
gains = []
for rep in range(40):
    lab = rich_ext.copy()
    lab[:] = rng.permutation(lab.to_numpy())
    yp = Y.copy()
    for e, v in lab.items():
        yp[ext == e] = int(v >= 0.5)
    tab = run(TOPO_COLS, yp, RICH, design="BP", est="lg")   # fast estimator for the null
    u = units(tab); ub = const_units(tab, 1)
    gains.append(float(u.hit.mean() - ub.hit.mean()))
    if rep % 10 == 9: print(f"  {rep+1} perms, gains so far mean={np.mean(gains):.4f} max={np.max(gains):.4f}", flush=True)
g = np.asarray(gains)
print(f"null gain: mean={g.mean():.4f} sd={g.std():.4f} max={g.max():.4f} "
      f"frac >= 0.2098: {float((g>=0.2098).mean()):.4f}  (n={len(g)})")
np.save(OUT / "v_perm_gains.npy", g)
print("done", round(time.time()-t0))
