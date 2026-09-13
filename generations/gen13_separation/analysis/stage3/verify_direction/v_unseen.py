"""How much of the accuracy survives on test cells whose exact topology vector is NOT in training?"""
import sys, json
sys.path.insert(0, "D:/ml_separator_gh/generations/gen13_separation/analysis/stage3/verify_direction")
from v_runner import *
import numpy as np, pandas as pd
from gen13sep.splits import all_folds

AMP = COEF[:, 0]; NM = FRAME.n_metals.to_numpy(); Y = (AMP < 0).astype(int); RICH = NM >= 5
KEY = np.array(["|".join(map(repr, r)) for r in ALLX.loc[:, TOPO_COLS].to_numpy(float)])
tab = pd.read_parquet(OUT / "v_topo_bp_predictions.parquet")

seen_rows = []
for f in all_folds(FRAME, design="BP"):
    tr = f.train_index[RICH[f.train_index]]; te = f.test_index[RICH[f.test_index]]
    if len(tr) < 40 or len(te) < 1: continue
    trk = set(KEY[tr])
    for ci in te:
        seen_rows.append({"split_seed": f.seed, "fold": f.fold,
                          "cell_id": FRAME.cell_id.iat[ci], "seen": KEY[ci] in trk})
S = pd.DataFrame(seen_rows)
t = tab.merge(S, on=["split_seed", "fold", "cell_id"], validate="one_to_one")
print("fraction of (seed,fold,cell) rows whose topology vector was already in training:",
      round(float(t.seen.mean()), 4))
for flag in (True, False):
    s = t[t.seen == flag]
    u = units(s); ub = const_units(s, 1)
    g = boot(u, ub, reps=4000)
    print(f" seen={flag}: rows={len(s)} ext={len(u)} chem={u.chemotype.nunique()} "
          f"macro={u.hit.mean():.4f} baseline={ub.hit.mean():.4f} "
          f"gain={g['value']:.4f} [{g['ci_low']:.4f},{g['ci_high']:.4f}] p={g['p_two_sided']:.4f}")

# accuracy stratified by |amp| margin (test cells only, model unchanged)
amp_by_cell = pd.Series(AMP, index=FRAME.cell_id.to_numpy())
t["amp"] = t.cell_id.map(amp_by_cell).abs()
t["band"] = pd.cut(t.amp, [-1e-9, 0.05, 0.1, 0.2, 0.5, 1e9],
                   labels=["<0.05", "0.05-0.1", "0.1-0.2", "0.2-0.5", ">0.5"])
print("\naccuracy by |radius coefficient| band (cell-level, and macro):")
for b, s in t.groupby("band", observed=True):
    u = units(s); ub = const_units(s, 1)
    print(f"  {b:>9}: rows={len(s):4d} ext={len(u):3d} cell_acc={float(((s.p>=0.5).astype(int)==s.y).mean()):.3f} "
          f"macro={u.hit.mean():.3f} always_heavy={ub.hit.mean():.3f}")
t.to_csv(OUT / "v_unseen_and_margin.csv", index=False)
