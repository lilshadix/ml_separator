"""Cheap probe: does gamma move anything at all?  Out-of-fold AMPLITUDE error along the anchor
path, skipping the pairwise-curve machinery.  If the amplitude MAE and the sign accuracy are flat
in gamma, the pairwise macro-MAE is flat too and the expensive run is not worth its hours."""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("D:/ml_separator_gh")
sys.path.insert(0, str(ROOT/"gen16_anchor"/"scripts"))
sys.path.insert(0, str(ROOT/"gen14_direction")); sys.path.insert(0, str(ROOT/"gen13_separation"))
from gen14 import dirbench as db
from gen13sep.amplitude_bench import LEAN_BLOCKS, cell_weights
from gen13sep.splits import all_folds
from g16_anchor_fast import anchor_path, GAMMAS

bench = db.load(); fr = bench.frame; FS = db.feature_sets(bench)
X = bench.matrix(LEAN_BLOCKS); amp = bench.coef[:,0]
rich = fr.n_metals.to_numpy() >= db.MIN_METALS
pub_all = fr.publication_id.astype(str).to_numpy()
ext_all = fr.extractant.astype(str).to_numpy(); chem_all = fr.chemotype.astype(str).to_numpy()
BLOCKS = {"COND_MA": np.concatenate([FS["COND64"],FS["MASSACT8"]]),
          "TOPO39": FS["TOPO39"],
          "ALL111": np.concatenate([FS["TOPO39"],FS["COND64"],FS["MASSACT8"]])}

rows=[]
for design in sys.argv[1:] or ["BP"]:
    t0=time.time()
    for f in all_folds(fr, design=design):
        tr, te = f.train_index, f.test_index
        rtr = tr[rich[tr]]; rte = te[rich[te]]
        if len(rtr) < db.MIN_TRAIN or len(np.unique(pub_all[rtr])) < 4 or len(rte)==0: continue
        wr = cell_weights(bench.groups[rtr], bench.n_obs[rtr]); wr/=wr.sum()
        magc = float(np.average(np.abs(amp[rtr]), weights=wr))
        amean = float(np.average(amp[rtr], weights=wr))
        base = {"design":design,"seed":f.seed,"fold":f.fold}
        for blk, idx in BLOCKS.items():
            Xtr, Xte = X[np.ix_(rtr,idx)], X[np.ix_(rte,idx)]
            med = np.nan_to_num(np.nanmedian(np.where(np.isnan(Xtr),np.nan,Xtr),axis=0),nan=0.0)
            path = anchor_path(np.where(np.isnan(Xtr),med,Xtr), amp[rtr], wr, pub_all[rtr],
                               np.where(np.isnan(Xte),med,Xte))
            for g,(ah,a_lo,_) in path.items():
                for j,ci in enumerate(rte):
                    rows.append(dict(base, block=blk, gamma=g, alpha=a_lo,
                                     ext=ext_all[ci], chem=chem_all[ci], amp=amp[ci],
                                     pred=float(ah[j]),
                                     err=abs(float(ah[j])-amp[ci]),
                                     signhit=float((ah[j]<0)==(amp[ci]<0)),
                                     err_const=abs(np.sign(amp[ci])*0-amp[ci]) ))
        for j,ci in enumerate(rte):
            rows.append(dict(base, block="_BASE_MEANAMP", gamma=np.nan, alpha=np.nan,
                             ext=ext_all[ci], chem=chem_all[ci], amp=amp[ci], pred=amean,
                             err=abs(amean-amp[ci]), signhit=float((amean<0)==(amp[ci]<0)),
                             err_const=np.nan))
    print(f"[{design}] {time.time()-t0:.0f}s", flush=True)

d = pd.DataFrame(rows)
d.to_parquet(ROOT/"gen16_anchor"/"results"/"g16_gamma_probe.parquet")
# macro over extractants, averaged over seeds
def macro(sub, col):
    per = sub.groupby(["seed","ext"])[col].mean().reset_index()
    return per.groupby("seed")[col].mean().mean()
print("\n=== OUT-OF-FOLD AMPLITUDE, extractant-macro (design x block x gamma) ===")
print(" design block          gamma      MAE(amp)   sign acc   median alpha")
for (dg,blk), sub in d.groupby(["design","block"], sort=False):
    for g, s2 in sub.groupby("gamma", dropna=False, sort=True):
        print(f" {dg:3s} {blk:14s} {g if g==g else -1:8.4g}  {macro(s2,'err'):9.4f}  {macro(s2,'signhit'):8.4f}   {s2.alpha.median() if s2.alpha.notna().any() else float('nan'):8.4g}")
