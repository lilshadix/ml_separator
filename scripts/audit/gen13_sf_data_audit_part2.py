"""Audit part 2: provenance join, publication-aware cells, log SF baselines, replicate noise, feat3d coverage.
Run from repo root:  .venv/Scripts/python.exe <this file>
"""
import os, re, collections, json
import numpy as np, pandas as pd
REPO = r"D:\ml_separator_gh"
ROOT = os.path.join(REPO, "dataset with 3D structures")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
df = pd.read_parquet(os.path.join(ROOT, "dataset.parquet"))
cols = list(df.columns)
cond_cols = [c for c in cols if c.startswith("cond__")]
Z = {"La":57,"Ce":58,"Pr":59,"Nd":60,"Sm":62,"Eu":63,"Gd":64,"Tb":65,"Dy":66,"Ho":67,"Er":68,"Tm":69,"Yb":70,"Lu":71}

print("### (A) PROVENANCE TABLE runs/gen6_provenance/provenance_table.parquet")
pt = pd.read_parquet(os.path.join(REPO, "runs/gen6_provenance/provenance_table.parquet"))
print("shape", pt.shape); print("columns", list(pt.columns))
print(pt.head(3).T.to_string()[:3000])
for c in pt.columns:
    if pt[c].dtype == object or str(pt[c].dtype).startswith("str"):
        print(f"   {c}: nunique={pt[c].nunique()} nulls={int(pt[c].isna().sum())}")
    else:
        print(f"   {c}: dtype={pt[c].dtype} nunique={pt[c].nunique()} nulls={int(pt[c].isna().sum())}")
pubcol = [c for c in pt.columns if "pub" in c.lower()]
sercol = [c for c in pt.columns if "series" in c.lower()]
print("publication-like cols:", pubcol, " series-like cols:", sercol)
m = df.merge(pt, on="safe_exp_id", how="left", suffixes=("", "_pt"))
print("join matched:", int(m[pubcol[0]].notna().sum()) if pubcol else "n/a", "of", len(df))
PUB = pubcol[0] if pubcol else None
SER = sercol[0] if sercol else None
if PUB:
    print("n publications:", m[PUB].nunique())
    print("rows per publication quantiles:", m[PUB].value_counts().quantile([.1,.25,.5,.75,.9,1.0]).to_dict())
    print("publications per extractant distribution:", m.groupby("canonical_smiles")[PUB].nunique().value_counts().sort_index().to_dict())
    print("metals per publication distribution:", m.groupby(PUB)["metal"].nunique().value_counts().sort_index().to_dict())
    pubs14 = m.groupby(PUB)["metal"].nunique()
    print("publications with all 14 metals:", int((pubs14 == 14).sum()), " rows in them:", int(m[PUB].map(pubs14 == 14).sum()))
    # exact cells crossing a publication boundary
    key = ["canonical_smiles"] + cond_cols
    cells = m.groupby(key, dropna=False).agg(n_rows=("log_D","size"), n_metals=("metal","nunique"), n_pubs=(PUB,"nunique")).reset_index()
    print("EXACT cells: n=", len(cells), " cells with >1 publication:", int((cells["n_pubs"] > 1).sum()),
          " rows in those:", int(cells.loc[cells["n_pubs"] > 1, "n_rows"].sum()))
    mm = cells[cells["n_metals"] >= 2]
    print("   multi-metal cells with >1 publication:", int((mm["n_pubs"] > 1).sum()), "of", len(mm), " rows:", int(mm.loc[mm["n_pubs"] > 1, "n_rows"].sum()))
    # publication-aware cells
    key2 = key + [PUB]
    cells2 = m.groupby(key2, dropna=False).agg(n_rows=("log_D","size"), n_metals=("metal","nunique")).reset_index()
    print("PUB-AWARE cells (smiles+64 cond+publication): n=", len(cells2), " >=2 metals:", int((cells2["n_metals"] >= 2).sum()),
          " dist:", cells2["n_metals"].value_counts().sort_index().to_dict())
    print("   rows in >=2-metal cells:", int(cells2.loc[cells2["n_metals"] >= 2, "n_rows"].sum()),
          " rows in 14-metal cells:", int(cells2.loc[cells2["n_metals"] == 14, "n_rows"].sum()), " n 14-metal cells:", int((cells2["n_metals"] == 14).sum()),
          " extractants with >=1 multi-metal cell:", cells2.loc[cells2["n_metals"] >= 2, "canonical_smiles"].nunique())
    # multi-metal cells by curator / stem
    if "curator" in pt.columns or any("curator" in c for c in pt.columns):
        cc = [c for c in pt.columns if "curator" in c][0]
        mm_rows = m.merge(cells[["canonical_smiles"] + cond_cols + ["n_metals"]], on=["canonical_smiles"] + cond_cols, how="left")
        print("   rows in multi-metal cells by", cc, ":", mm_rows[mm_rows["n_metals"] >= 2][cc].value_counts().to_dict())
    if SER:
        print("series col", SER, "nunique", m[SER].nunique())
        cells3 = m.groupby(["canonical_smiles", SER], dropna=False).agg(n_rows=("log_D","size"), n_metals=("metal","nunique")).reset_index()
        print("SERIES cells (smiles+series): n=", len(cells3), " >=2 metals:", int((cells3["n_metals"] >= 2).sum()), " dist:", cells3["n_metals"].value_counts().sort_index().to_dict())

print("\n### (B) LOG SF BASELINES (exact key, dZ==1 pairs)")
key = ["canonical_smiles"] + cond_cols
cm = df.groupby(key + ["metal"], dropna=False)["log_D"].agg(["mean", "size", "std"]).reset_index()
cm["Z"] = cm["metal"].map(Z); cm["cell_id"] = cm.groupby(key, dropna=False).ngroup()
recs = []
for cid, sub in cm.groupby("cell_id"):
    if sub["metal"].nunique() < 2: continue
    sub = sub.sort_values("Z"); zs = sub["Z"].values; ld = sub["mean"].values; ms = sub["metal"].values
    for i in range(len(zs) - 1):
        recs.append(dict(cell_id=cid, smiles=sub["canonical_smiles"].iloc[0], pair=ms[i] + "-" + ms[i+1], dZ=zs[i+1]-zs[i], y=ld[i+1]-ld[i]))
sf = pd.DataFrame(recs)
s1 = sf[sf["dZ"] == 1].copy()
tot = s1["y"].var(ddof=1)
res_pair = (s1["y"] - s1.groupby("pair")["y"].transform("mean")).var(ddof=1)
res_ext = (s1["y"] - s1.groupby("smiles")["y"].transform("mean")).var(ddof=1)
# additive extractant + pair (one pass of alternating means, 20 iters)
y = s1["y"].values; a = np.zeros(len(s1)); e = s1["smiles"].values; p = s1["pair"].values
ae = pd.Series(0.0, index=pd.unique(e)); ap = pd.Series(0.0, index=pd.unique(p)); mu = y.mean()
for _ in range(50):
    ae = pd.Series(y - mu - ap[p].values).groupby(e).mean()
    ap = pd.Series(y - mu - ae[e].values).groupby(p).mean()
res_add = (y - mu - ae[e].values - ap[p].values).var(ddof=1)
res_ext_pair = (s1["y"] - s1.groupby(["smiles", "pair"])["y"].transform("mean")).var(ddof=1)
print(f"dZ==1 pairs n={len(s1)}, total var={tot:.4f} (sd={np.sqrt(tot):.4f})")
print(f"   residual var after pair-type mean: {res_pair:.4f} (R2={1-res_pair/tot:.3f})")
print(f"   residual var after per-extractant mean: {res_ext:.4f} (R2={1-res_ext/tot:.3f})")
print(f"   residual var after additive extractant+pair: {res_add:.4f} (R2={1-res_add/tot:.3f})")
print(f"   residual var after (extractant x pair) cell mean [condition-only residual, in-sample]: {res_ext_pair:.4f} (R2={1-res_ext_pair/tot:.3f}); n groups={s1.groupby(['smiles','pair']).ngroups}")
print(f"   MAE of predicting 0: {s1['y'].abs().mean():.4f}; MAE of pair-type mean (in-sample): {(s1['y'] - s1.groupby('pair')['y'].transform('mean')).abs().mean():.4f}; MAE of per-extractant mean: {(s1['y'] - s1.groupby('smiles')['y'].transform('mean')).abs().mean():.4f}")
# sign-prediction accuracy of the pair-type sign
sign_pair = np.sign(s1.groupby("pair")["y"].transform("mean")); print(f"   sign accuracy of pair-type mean sign: {(np.sign(s1['y']) == sign_pair).mean():.3f}; of 'heavier always preferred': {(s1['y'] > 0).mean():.3f}")
# per pair-type sd (only main pairs)
pp = s1.groupby("pair")["y"].agg(["size", "mean", "std"]); pp = pp[pp["size"] >= 30]
print("   dZ==1 pair types with n>=30: mean sd across types =", round(float(pp["std"].mean()), 4), " min sd", round(float(pp["std"].min()), 4), " max sd", round(float(pp["std"].max()), 4))

print("\n### (B2) REPLICATE NOISE FLOOR: (cell,metal) combos with >1 row")
rep = cm[cm["size"] > 1]
print("n (cell,metal) with replicates:", len(rep), " rows involved:", int(rep["size"].sum()), " size dist:", rep["size"].value_counts().sort_index().to_dict())
print(f"within-replicate sd of log_D: mean={rep['std'].mean():.4f} median={rep['std'].median():.4f} q90={rep['std'].quantile(.9):.4f} max={rep['std'].max():.4f}")
print(f"   -> implied noise sd of log SF (sqrt(2)*median sd) = {np.sqrt(2)*rep['std'].median():.4f}; frac replicate sd==0: {(rep['std']==0).mean():.3f}")
# which extractants/conditions have replicates
print("replicates by extractant_name (top 8):", df.merge(rep[key + ['metal']], on=key + ['metal'])['extractant_name'].value_counts().head(8).to_dict())

print("\n### (C) feat3d__ PER-COLUMN NON-NULL")
f3 = [c for c in cols if c.startswith("feat3d__")]
nn = df[f3].notna().sum()
print("complex_physical columns with 0 non-null:", [c for c in f3 if c.startswith("feat3d__complex_physical") and nn[c] == 0])
print("complex_physical non-null counts (unique values):", sorted(set(int(v) for c, v in nn.items() if c.startswith("feat3d__complex_physical"))))
poly = {c: int(nn[c]) for c in f3 if c.startswith("feat3d__polyhedron__")}
by_cnt = collections.defaultdict(list)
for c, v in poly.items(): by_cnt[v].append(c.replace("feat3d__polyhedron__", ""))
for v in sorted(by_cnt, reverse=True): print(f"   polyhedron non-null={v}: {by_cnt[v]}")
print("polyhedron_scalars:", {c: int(nn[c]) for c in f3 if c.startswith("feat3d__polyhedron_scalars")})
print("coreCN dist (all rows):", df["coreCN"].value_counts().sort_index().to_dict(), " geometry_ok rows coreCN:", df.loc[df["geometry_ok"], "coreCN"].value_counts().sort_index().to_dict())
print("feat3d coordination_number dist (geometry_ok):", df.loc[df["geometry_ok"], "feat3d__complex_physical__coordination_number"].value_counts().sort_index().to_dict())

print("\n### (D) MULTI-METAL CELLS BY safe_exp_id STEM (exact key)")
stem = df["safe_exp_id"].astype(str).str.split(":").str[0]
cells = df.groupby(key, dropna=False)["metal"].transform("nunique")
print("rows in >=2-metal cells by stem:", df[cells >= 2].assign(stem=stem[cells >= 2])["stem"].value_counts().to_dict())
print("rows in 14-metal cells by stem:", df[cells == 14].assign(stem=stem[cells == 14])["stem"].value_counts().to_dict())
print("14-metal-cell extractant names:", sorted(df.loc[cells == 14, "extractant_name"].unique().tolist())[:60])

print("\n### (E) build_id / geometry_key STRUCTURE")
print("build_id nunique:", df["build_id"].nunique(), " geometry_key nunique:", df["geometry_key"].nunique())
print("build_id -> geometry_key one-to-one:", bool((df.groupby("build_id")["geometry_key"].nunique() == 1).all()), bool((df.groupby("geometry_key")["build_id"].nunique() == 1).all()))
gk = df["geometry_key"].str.split("|", expand=True)
print("geometry_key parts: Z nunique", gk[0].nunique(), " smiles nunique", gk[1].nunique(), " anion values", gk[2].value_counts().to_dict())
print("inner_sphere_anion dist:", df["inner_sphere_anion"].value_counts().to_dict())
print("fill_ligand dist:", df["fill_ligand"].value_counts().to_dict(), " n_fill dist:", df["n_fill"].value_counts().to_dict())
print("(metal, smiles) combos:", df.groupby(["metal", "canonical_smiles"]).ngroups, " vs geometry_key nunique", df["geometry_key"].nunique())
print("rows per geometry_key quantiles:", df["geometry_key"].value_counts().quantile([.25,.5,.75,.9,1.0]).to_dict())
print("geometry_xtb_energy_eV non-null:", int(df["geometry_xtb_energy_eV"].notna().sum()))
print("DENTATE dist:", df["DENTATE"].value_counts().sort_index().to_dict(), " n_ligs dist:", df["n_ligs"].value_counts().sort_index().to_dict())

print("\n### (F) ligand_2d_descriptors hand-crafted columns")
l2 = pd.read_parquet(os.path.join(ROOT, "ligand_2d_descriptors.parquet"))
print("hc cols:", [c for c in l2.columns if c.startswith("lig2d__hc__")])
print("nan count total:", int(l2.isna().sum().sum()), " dtypes:", l2.dtypes.value_counts().to_dict())
man = json.load(open(os.path.join(ROOT, "ligand_2d_descriptors.manifest.json")))
print("manifest n_rows", man.get("n_rows"), "n_columns", man.get("n_columns"), "n_rdkit", man.get("n_rdkit_columns"), "n_hc", man.get("n_hand_crafted_columns"), "unparsed", man.get("unparsed_smiles"), "rdkit", man.get("rdkit_version"))

print("\n### (G) dataset_geometry_available.parquet")
dg = pd.read_parquet(os.path.join(ROOT, "dataset_geometry_available.parquet"))
print("shape", dg.shape, " geometry_ok all:", bool(dg["geometry_ok"].all()), " same columns as dataset:", list(dg.columns) == cols)
print("DONE2")
