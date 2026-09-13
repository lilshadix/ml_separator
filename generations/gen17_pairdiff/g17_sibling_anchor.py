"""Is the sibling-anchor gain trivial (same extractant) or real (different extractant)?"""
import sys,pickle,numpy as np
sys.path.insert(0,"gen13_separation")
from gen13sep import basis as Bs
d=pickle.load(open("gen14_direction/cache/bench.pkl","rb"))
F=d.frame; coef=d.coef; bas=d.basis; C=Bs.centre_rows(d.Y); nobs=d.n_obs
keep=np.flatnonzero(nobs>=5)
pub=F["publication_id"].to_numpy(); ext=F["extractant"].to_numpy()
def macro(predc,cells):
    per={}
    for i in cells:
        pc=predc[i]@bas; obs=np.flatnonzero(~np.isnan(C[i])); e=[]
        for u in range(len(obs)):
            for v in range(u+1,len(obs)):
                A_,B_=obs[u],obs[v]
                e.append(abs((C[i,A_]-C[i,B_])-(pc[A_]-pc[B_])))
        if e: per.setdefault(ext[i],[]).extend(e)
    return float(np.mean([np.mean(v) for v in per.values()])),len(per)
for tag,pred_fn in [("ANY sibling in pub",lambda i: [j for j in keep if j!=i and pub[j]==pub[i]]),
                    ("DIFFERENT-extractant sibling",lambda i: [j for j in keep if j!=i and pub[j]==pub[i] and ext[j]!=ext[i]]),
                    ("SAME-extractant sibling",lambda i: [j for j in keep if j!=i and pub[j]==pub[i] and ext[j]==ext[i]]),
                    ("SAME extractant, ANY publication",lambda i: [j for j in keep if j!=i and ext[j]==ext[i]])]:
    cells=[i for i in keep if pred_fn(i)]
    if not cells: continue
    P=np.zeros_like(coef); P1=np.zeros_like(coef)
    rng=np.random.default_rng(0)
    for i in cells:
        s=pred_fn(i); P[i]=coef[s].mean(0); P1[i]=coef[rng.choice(s)]
    z=np.zeros_like(coef)
    mz,_=macro(z,cells); mm,ne=macro(P,cells); m1,_=macro(P1,cells); mo,_=macro(coef,cells)
    print(f"\n{tag}: n_cells={len(cells)} n_ext={ne}")
    print(f"   zero curve {mz:.4f} | copy-1 {m1:.4f} | copy-mean {mm:.4f} | oracle-own {mo:.4f}")
