"""The cleanest ligand contrast in the corpus: same publication, SAME conditions,
DIFFERENT extractant.  Lab effect and condition effect both cancel exactly."""
import sys,pickle,numpy as np,pandas as pd
sys.path.insert(0,"generations/gen13_separation")
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
d=pickle.load(open("generations/gen14_direction/cache/bench.pkl","rb"))
F=d.frame; a=d.coef[:,0]; nobs=d.n_obs
keep=np.flatnonzero(nobs>=5)
pub=F["publication_id"].to_numpy(); ext=F["extractant"].to_numpy(); chem=F["chemotype"].to_numpy()
CM=np.hstack([np.nan_to_num(d.frames["COND"].to_numpy(float)),
              np.nan_to_num(d.frames["MASSACT"].to_numpy(float))])
CO=d.frames["COORD"]; tc=[c for c in CO.columns if c.startswith(("coord__dist__","coord__arm__"))]
TP=np.nan_to_num(CO[tc].to_numpy(float))
DN=np.nan_to_num(d.frames["DONORS"].to_numpy(float))
rows=[]
for p in np.unique(pub[keep]):
    idx=keep[pub[keep]==p]
    for i in idx:
        for j in idx:
            if i==j or ext[i]==ext[j]: continue
            if np.abs(CM[i]-CM[j]).max()>1e-9: continue
            rows.append((p,i,j))
P=pd.DataFrame(rows,columns=["pub","i","j"]); I=P.i.values; J=P.j.values
print("clean pairs:",len(P)," publications:",P.pub.nunique())
print(P.pub.value_counts().to_string())
y=a[I]-a[J]; grp=P.pub.values
print("null MAE |da| = %.4f   sd(da)=%.4f"%(np.abs(y).mean(),y.std()))
print("same chemotype pairs: %d / %d"%((chem[I]==chem[J]).sum(),len(P)))
for name,Xu in [("dTOPO39",TP[I]-TP[J]),("TOPOi|TOPOj",np.hstack([TP[I],TP[J]])),
                ("dDONORS13",DN[I]-DN[J]),("dTOPO+dDONORS",np.hstack([TP[I]-TP[J],DN[I]-DN[J]]))]:
    for mk,tag in [(lambda: ExtraTreesRegressor(n_estimators=200,min_samples_leaf=4,random_state=0,n_jobs=2),"ET"),
                   (lambda: Ridge(alpha=1.0),"ridge")]:
        pr=np.full(len(y),np.nan)
        for g in np.unique(grp):
            te=grp==g; tr=~te
            if tr.sum()<20: continue
            m=mk(); m.fit(Xu[tr],y[tr]); pr[te]=m.predict(Xu[te])
        ok=~np.isnan(pr)
        if ok.sum()<10: print(f"{name:14s} {tag:5s} too few scored"); continue
        print(f"{name:14s} {tag:5s} n={ok.sum():4d} null {np.abs(y[ok]).mean():.4f} model {np.abs(y[ok]-pr[ok]).mean():.4f} delta {np.abs(y[ok]-pr[ok]).mean()-np.abs(y[ok]).mean():+.4f}",flush=True)
