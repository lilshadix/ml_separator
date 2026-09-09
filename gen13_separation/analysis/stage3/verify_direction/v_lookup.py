import sys
sys.path.insert(0, "D:/ml_separator_gh/gen13_separation/analysis/stage3/verify_direction")
from v_runner import *
import numpy as np, pandas as pd
from gen13sep.splits import all_folds, assert_fold_integrity

rich = FRAME.n_metals.to_numpy() >= 5
y = (COEF[:, 0] < 0).astype(int)
T = ALLX.loc[:, TOPO_COLS]
key = np.array(["|".join(map(repr, r)) for r in T.to_numpy(float)])
L = ALLX.loc[:, LEAN_COLS]
lkey = None

print("=== leakage / integrity, design BP ===")
for d in ("B", "BP"):
    folds = all_folds(FRAME, design=d)
    print(" ", d, assert_fold_integrity(FRAME, folds))

folds = all_folds(FRAME, design="BP")
pubs = FRAME.publication_id.astype(str).to_numpy()
ext = FRAME.extractant.astype(str).to_numpy()
chem = FRAME.chemotype.astype(str).to_numpy()
ec = FRAME.ecfp_cluster.astype(str).to_numpy()
bad = {"ext":0,"chem":0,"ecfp":0,"pub":0}
hit_exact, hit_ext_topo, n_test = 0, 0, 0
rows = []
for f in folds:
    tr = f.train_index[rich[f.train_index]]; te = f.test_index[rich[f.test_index]]
    bad["ext"]  += len(set(ext[tr])  & set(ext[te]))
    bad["chem"] += len(set(chem[tr]) & set(chem[te]))
    bad["ecfp"] += len(set(ec[tr])   & set(ec[te]))
    bad["pub"]  += len(set(pubs[tr]) & set(pubs[te]))
    trk = set(key[tr])
    n_test += len(te)
    hit_exact += int(sum(k in trk for k in key[te]))
    for i in te:
        rows.append({"cell": i, "seen": key[i] in trk,
                     "n_train_same_key": int((key[tr] == key[i]).sum()),
                     "train_key_heavy_frac": float(y[tr][key[tr] == key[i]].mean()) if (key[tr]==key[i]).any() else np.nan,
                     "y": int(y[i])})
print(" restricted-to-rich overlaps:", bad)
print(f" test cells whose EXACT 39-col topology vector is already in training: {hit_exact}/{n_test} "
      f"= {hit_exact/n_test:.3f}")
R = pd.DataFrame(rows)
seen = R[R.seen]
print(f" of those, the training rows sharing the identical vector have mean heavy-fraction; "
      f"accuracy of 'copy the identical training row's majority' = "
      f"{float(((seen.train_key_heavy_frac>=0.5).astype(int)==seen.y).mean()):.3f} (cells, not macro)")
print(" distinct topo keys in the rich cohort:", len(set(key[rich])),
      " distinct extractants:", FRAME.extractant[rich].nunique(),
      " distinct chemotypes:", FRAME.chemotype[rich].nunique())
kc = pd.DataFrame({"key": key[rich], "chem": chem[rich], "ext": ext[rich]})
g = kc.groupby("key").agg(n_chem=("chem","nunique"), n_ext=("ext","nunique")).sort_values("n_chem", ascending=False)
print(" topo keys spanning >1 chemotype:", int((g.n_chem>1).sum()), "of", len(g))
print(g.head(12).to_string())
R.to_csv(OUT/"v_topology_lookup.csv", index=False)
