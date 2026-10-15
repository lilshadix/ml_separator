"""Quantitative data audit for the separation-factor question (gen13).
Run from repo root:  .venv/Scripts/python.exe <this file>
"""
import re, json, os, collections
import numpy as np
import pandas as pd

ROOT = r"D:\ml_separator_gh\dataset with 3D structures"
OUT = os.path.abspath(os.environ.get("SCRATCH", os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 500)

df = pd.read_parquet("dataset.parquet")
cols = list(df.columns)
print("### (1) SHAPE", df.shape)

def prefix_of(c):
    if c.startswith("feat3d__"): return "feat3d__"
    if c.startswith("cond__"): return "cond__"
    if c.startswith("geom_cond__"): return "geom_cond__"
    if re.match(r"^ecfp_\d+$", c): return "ecfp_*"
    return "identity/descriptor (no prefix)"
pref = collections.Counter(prefix_of(c) for c in cols)
print("column families:", dict(pref))
cond_cols = [c for c in cols if c.startswith("cond__")]
cond_num = [c for c in cond_cols if df[c].dtype == "float64"]
cond_bin = [c for c in cond_cols if c not in cond_num]
print("cond__ numeric:", cond_num)
print("cond__ one-hot (int8) count:", len(cond_bin))
print("cond__ one-hot groups:", collections.Counter(c.split("__")[1] for c in cond_bin))
print("cond__ non-null counts (numeric):")
print(df[cond_num].notna().sum().to_string())
print("cond__ entirely-NaN columns:", [c for c in cond_cols if df[c].isna().all()])
print("cond__ one-hot columns with zero positives:", [c for c in cond_bin if (df[c] == 1).sum() == 0])
idc = [c for c in cols if prefix_of(c) == "identity/descriptor (no prefix)"]
print("identity/descriptor columns:", idc)
rd_desc = ["MolWt","TPSA","NumHDonors","NumHAcceptors","NumRotatableBonds","NumAromaticRings","NumAliphaticRings","RingCount","FractionCSP3","MolLogP"]
metal_desc = ["Atomic Number_metal","lanthanide_index","Ionic Radius_metal"]
print("RDKit 2D descriptors:", rd_desc)
print("metal descriptors:", metal_desc)
print("log_D stats:", df["log_D"].describe().to_string())
print("log_D NaN:", df["log_D"].isna().sum(), " D NaN:", df["D"].isna().sum())
print("split values:", df["split"].value_counts(dropna=False).to_dict())
print("build_id values:", df["build_id"].value_counts(dropna=False).to_dict())
print("n unique canonical_smiles:", df["canonical_smiles"].nunique(), " n unique extractant_name:", df["extractant_name"].nunique(),
      " n unique extractant_group:", df["extractant_group"].nunique(), " n unique LIGAND_SMILES:", df["LIGAND_SMILES"].nunique())
print("canonical_smiles==extractant_group all:", bool((df["canonical_smiles"] == df["extractant_group"]).all()))
print("canonical_smiles==LIGAND_SMILES all:", bool((df["canonical_smiles"] == df["LIGAND_SMILES"]).all()))
print("safe_exp_id unique:", df["safe_exp_id"].nunique(), "of", len(df))
print("n unique geometry_key:", df["geometry_key"].nunique())

print("\n### (2) ROWS PER METAL / EXTRACTANTS PER METAL")
Z = {"La":57,"Ce":58,"Pr":59,"Nd":60,"Pm":61,"Sm":62,"Eu":63,"Gd":64,"Tb":65,"Dy":66,"Ho":67,"Er":68,"Tm":69,"Yb":70,"Lu":71}
pm = df.groupby("metal").agg(rows=("log_D","size"), n_extractants=("canonical_smiles","nunique"),
                             geom_ok=("geometry_ok","sum"), logD_mean=("log_D","mean"), logD_sd=("log_D","std"))
pm["Z"] = pm.index.map(Z)
print(pm.sort_values("Z").to_string())
print("metals present:", sorted(df["metal"].unique(), key=lambda m: Z[m]))
print("n metals:", df["metal"].nunique())
print("metal == metal_symbol all:", bool((df["metal"] == df["metal_symbol"]).all()))
print("metal_ox values:", df["metal_ox"].value_counts().to_dict())
# extractants by number of metals covered (ignoring conditions)
ext_metals = df.groupby("canonical_smiles")["metal"].nunique()
print("extractants by #distinct metals (any conditions):", ext_metals.value_counts().sort_index().to_dict())
print("extractants with >=2 metals (any conditions):", int((ext_metals >= 2).sum()), "of", len(ext_metals))
print("rows belonging to extractants with >=2 metals:", int(df["canonical_smiles"].map(ext_metals >= 2).sum()))

print("\n### (3) CONDITION-CELL STRUCTURE")
def cell_analysis(keycols, label):
    g = df.groupby(["canonical_smiles"] + keycols, dropna=False)
    cells = g.agg(n_rows=("log_D","size"), n_metals=("metal","nunique")).reset_index()
    print(f"-- {label}: key = canonical_smiles + {len(keycols)} cond cols")
    print("   n cells:", len(cells))
    print("   n cells with >=2 metals:", int((cells["n_metals"] >= 2).sum()))
    print("   distribution of #metals per cell:", cells["n_metals"].value_counts().sort_index().to_dict())
    print("   distribution of #rows per cell (head):", cells["n_rows"].value_counts().sort_index().head(20).to_dict())
    print("   cells with n_rows > n_metals (replicate rows within cell/metal):", int((cells["n_rows"] > cells["n_metals"]).sum()))
    multi = cells[cells["n_metals"] >= 2]
    print("   n extractants with >=1 multi-metal cell:", multi["canonical_smiles"].nunique(), "of", cells["canonical_smiles"].nunique())
    print("   rows in cells with >=2 metals:", int(multi["n_rows"].sum()))
    print("   rows in cells with >=5 metals:", int(cells.loc[cells["n_metals"] >= 5, "n_rows"].sum()),
          " n such cells:", int((cells["n_metals"] >= 5).sum()), " n extractants:", cells.loc[cells["n_metals"] >= 5, "canonical_smiles"].nunique())
    print("   rows in cells with >=10 metals:", int(cells.loc[cells["n_metals"] >= 10, "n_rows"].sum()),
          " n such cells:", int((cells["n_metals"] >= 10).sum()), " n extractants:", cells.loc[cells["n_metals"] >= 10, "canonical_smiles"].nunique())
    print("   rows in cells with all 14 metals:", int(cells.loc[cells["n_metals"] == 14, "n_rows"].sum()),
          " n such cells:", int((cells["n_metals"] == 14).sum()), " n extractants:", cells.loc[cells["n_metals"] == 14, "canonical_smiles"].nunique())
    # per extractant: max n_metals in a single cell
    mx = cells.groupby("canonical_smiles")["n_metals"].max()
    print("   per-extractant max #metals in one cell, distribution:", mx.value_counts().sort_index().to_dict())
    return cells

cells_exact = cell_analysis(cond_cols, "EXACT (NaN-inclusive, all 64 cond__)")
all_nan = [c for c in cond_cols if df[c].isna().all()]
relaxed_cols = [c for c in cond_cols if c not in all_nan]
print("relaxed key drops entirely-NaN cond cols:", all_nan, "-> identical to exact" if not all_nan else "")
cells_relaxed = cell_analysis(relaxed_cols, "RELAXED (drop entirely-NaN cond__ cols)")
relaxed2 = [c for c in cond_cols if c not in ("cond__contact_time_min", "cond__metal_concentration_mM")]
cells_relaxed2 = cell_analysis(relaxed2, "RELAXED-2 (additionally drop cond__contact_time_min, cond__metal_concentration_mM)")
relaxed3 = [c for c in cond_cols if c not in ("cond__contact_time_min", "cond__metal_concentration_mM", "cond__temperature_C")]
cells_relaxed3 = cell_analysis(relaxed3, "RELAXED-3 (also drop cond__temperature_C)")

# Cell table saved for downstream
cells_exact.to_csv(os.path.join(OUT, "cells_exact.csv"), index=False)

print("\n### (4) LOG SF FOR ADJACENT PAIRS IN MULTI-METAL CELLS (EXACT key)")
def adjacent_sf(keycols, label):
    key = ["canonical_smiles"] + keycols
    # mean log_D per (cell, metal) to handle replicate rows
    cm = df.groupby(key + ["metal"], dropna=False)["log_D"].agg(["mean", "size"]).reset_index()
    cm["Z"] = cm["metal"].map(Z)
    cm["cell_id"] = cm.groupby(key, dropna=False).ngroup()
    n_rep = int((cm["size"] > 1).sum())
    print(f"-- {label}: (cell,metal) combos with >1 row (replicates averaged): {n_rep} of {len(cm)}")
    recs = []
    for cid, sub in cm.groupby("cell_id"):
        if sub["metal"].nunique() < 2: continue
        sub = sub.sort_values("Z")
        zs = sub["Z"].values; ld = sub["mean"].values; ms = sub["metal"].values; smi = sub["canonical_smiles"].iloc[0]
        for i in range(len(zs) - 1):
            # adjacent = consecutive in the data ordering; flag true neighbors (dZ==1) separately, Pm(61) absent so Nd-Sm dZ==2 also 'adjacent in data'
            recs.append(dict(cell_id=cid, smiles=smi, A=ms[i], B=ms[i+1], ZA=zs[i], ZB=zs[i+1], dZ=zs[i+1]-zs[i],
                             logSF_heavy_minus_light=ld[i+1] - ld[i]))
    sf = pd.DataFrame(recs)
    print("   n adjacent-in-cell pairs:", len(sf), " n cells:", sf["cell_id"].nunique(), " n extractants:", sf["smiles"].nunique())
    print("   dZ distribution:", sf["dZ"].value_counts().sort_index().to_dict())
    x = sf["logSF_heavy_minus_light"]
    print(f"   log SF (heavy-light) all pairs: mean={x.mean():.4f} sd={x.std():.4f} mean|.|={x.abs().mean():.4f} median|.|={x.abs().median():.4f} "
          f"q10={x.quantile(.1):.3f} q90={x.quantile(.9):.3f} min={x.min():.3f} max={x.max():.3f} frac>0={(x>0).mean():.3f} frac|.|<0.1={(x.abs()<0.1).mean():.3f} frac|.|<0.3={(x.abs()<0.3).mean():.3f}")
    s1 = sf[sf["dZ"] == 1]["logSF_heavy_minus_light"]
    print(f"   true neighbours (dZ==1) n={len(s1)}: mean={s1.mean():.4f} sd={s1.std():.4f} mean|.|={s1.abs().mean():.4f} frac>0={(s1>0).mean():.3f}")
    s2 = sf[sf["dZ"] == 2]["logSF_heavy_minus_light"]
    if len(s2): print(f"   dZ==2 pairs n={len(s2)}: mean={s2.mean():.4f} sd={s2.std():.4f} mean|.|={s2.abs().mean():.4f} frac>0={(s2>0).mean():.3f}")
    sf["pair"] = sf["A"] + "-" + sf["B"]
    pp = sf.groupby("pair").agg(ZA=("ZA","first"), n=("logSF_heavy_minus_light","size"), n_ext=("smiles","nunique"),
                                mean=("logSF_heavy_minus_light","mean"), sd=("logSF_heavy_minus_light","std"),
                                mean_abs=("logSF_heavy_minus_light", lambda v: v.abs().mean()),
                                frac_heavy_pref=("logSF_heavy_minus_light", lambda v: (v > 0).mean())).sort_values("ZA")
    print("   per adjacent pair type (heavy-light):")
    print(pp.round(4).to_string())
    # sign consistency per extractant: fraction of its pairs with heavier preferred; then how many extractants are majority heavy-preferring
    pe = sf.groupby("smiles").agg(n_pairs=("logSF_heavy_minus_light","size"), frac_heavy=("logSF_heavy_minus_light", lambda v: (v > 0).mean()),
                                  mean_sf=("logSF_heavy_minus_light","mean"), mean_abs=("logSF_heavy_minus_light", lambda v: v.abs().mean()))
    print("   per-extractant: n extractants:", len(pe),
          " frac extractants with majority heavy-preferred (frac_heavy>0.5):", round(float((pe["frac_heavy"] > 0.5).mean()), 4),
          " frac with frac_heavy==1:", round(float((pe["frac_heavy"] == 1).mean()), 4),
          " frac with frac_heavy==0:", round(float((pe["frac_heavy"] == 0).mean()), 4),
          " frac with mean_sf>0:", round(float((pe["mean_sf"] > 0).mean()), 4))
    print("   per-extractant frac_heavy distribution (bins 0,.25,.5,.75,1):",
          pd.cut(pe["frac_heavy"], [-0.01, 0.0, 0.25, 0.5, 0.75, 0.999, 1.0]).value_counts().sort_index().to_dict())
    print("   per-extractant mean|logSF| quantiles:", pe["mean_abs"].quantile([.1,.25,.5,.75,.9]).round(3).to_dict())
    # within-cell sign consistency: fraction of cells monotone
    mono = sf.groupby("cell_id")["logSF_heavy_minus_light"].agg(lambda v: (v > 0).all() or (v < 0).all())
    ncell_multi = sf.groupby("cell_id").size()
    print("   cells with >=3 metals:", int((ncell_multi >= 2).sum()), " of which monotone in Z:", int(mono[ncell_multi >= 2].sum()))
    # variance decomposition: log_D variance within cell vs total
    cm2 = cm[cm.groupby("cell_id")["metal"].transform("nunique") >= 2]
    within = cm2.groupby("cell_id")["mean"].var(ddof=1)
    print(f"   log_D total var (rows)={df['log_D'].var():.4f}; mean within-multi-metal-cell var (across metals)={within.mean():.4f}; median={within.median():.4f}")
    return sf, pe

sf_exact, pe_exact = adjacent_sf(cond_cols, "EXACT key")
sf_r2, pe_r2 = adjacent_sf(relaxed2, "RELAXED-2 key")
sf_exact.to_csv(os.path.join(OUT, "adjacent_logSF_exact.csv"), index=False)

print("\n### (5) PER-EXTRACTANT Ln-SERIES SHAPE: log D ~ a + b*Z + c*Z^2 for cells with >=5 metals (EXACT key)")
def series_fit(keycols, label, min_metals=5):
    key = ["canonical_smiles"] + keycols
    cm = df.groupby(key + ["metal"], dropna=False)["log_D"].mean().reset_index()
    cm["Z"] = cm["metal"].map(Z)
    cm["cell_id"] = cm.groupby(key, dropna=False).ngroup()
    out = []
    for cid, sub in cm.groupby("cell_id"):
        if sub["metal"].nunique() < min_metals: continue
        z = (sub["Z"].values - 64.0); y = sub["log_D"].values
        sst = ((y - y.mean())**2).sum()
        X1 = np.column_stack([np.ones_like(z), z]); X2 = np.column_stack([np.ones_like(z), z, z**2])
        b1, *_ = np.linalg.lstsq(X1, y, rcond=None); b2, *_ = np.linalg.lstsq(X2, y, rcond=None)
        r1 = 1 - ((y - X1 @ b1)**2).sum() / sst if sst > 0 else np.nan
        r2 = 1 - ((y - X2 @ b2)**2).sum() / sst if sst > 0 else np.nan
        rng = y.max() - y.min()
        out.append(dict(cell_id=cid, smiles=sub["canonical_smiles"].iloc[0], n_metals=len(sub), sd_logD=y.std(ddof=1), range_logD=rng,
                        slope_lin=b1[1], R2_lin=r1, R2_quad=r2, b=b2[1], c=b2[2], rmse_quad=np.sqrt(((y - X2 @ b2)**2).mean())))
    f = pd.DataFrame(out)
    print(f"-- {label}: n cells with >={min_metals} metals: {len(f)}, n extractants: {f['smiles'].nunique()}, rows covered: {int(f['n_metals'].sum())}")
    print("   n_metals distribution:", f["n_metals"].value_counts().sort_index().to_dict())
    for c in ["R2_lin", "R2_quad", "sd_logD", "range_logD", "rmse_quad", "slope_lin"]:
        q = f[c].quantile([.1, .25, .5, .75, .9]).round(3).to_dict()
        print(f"   {c}: mean={f[c].mean():.3f} quantiles={q}")
    print("   frac cells R2_quad>=0.8:", round(float((f['R2_quad'] >= 0.8).mean()), 3), " >=0.9:", round(float((f['R2_quad'] >= 0.9).mean()), 3),
          " R2_lin>=0.8:", round(float((f['R2_lin'] >= 0.8).mean()), 3))
    print("   frac cells slope_lin>0 (heavier preferred):", round(float((f['slope_lin'] > 0).mean()), 3))
    print("   frac cells with c<0 (concave, mid-series max):", round(float((f['c'] < 0).mean()), 3))
    # per extractant best cell
    best = f.sort_values("n_metals", ascending=False).drop_duplicates("smiles")
    print("   per-extractant (largest cell) R2_quad quantiles:", best["R2_quad"].quantile([.1,.25,.5,.75,.9]).round(3).to_dict())
    return f

fit_exact = series_fit(cond_cols, "EXACT key")
fit_r2 = series_fit(relaxed2, "RELAXED-2 key")
fit_exact.to_csv(os.path.join(OUT, "series_fit_exact.csv"), index=False)

print("\n### (6) feat3d__ FAMILIES")
f3 = [c for c in cols if c.startswith("feat3d__")]
fam = collections.Counter(c.split("__")[1] for c in f3)
print("families (2nd token):", dict(fam))
nn = df[f3].notna().sum()
print("non-null row counts per family (min/max across cols):")
for k in fam:
    sub = nn[[c for c in f3 if c.split("__")[1] == k]]
    print(f"   {k}: n_cols={len(sub)} min_nonnull={int(sub.min())} max_nonnull={int(sub.max())}")
print("all feat3d__ columns:", f3)
print("geometry_ok True:", int(df["geometry_ok"].sum()), " False:", int((~df["geometry_ok"]).sum()))
print("rows with any feat3d__ non-null:", int(df[f3].notna().any(axis=1).sum()), " rows with all feat3d__ non-null:", int(df[f3].notna().all(axis=1).sum()))
print("geometry_ok & all feat3d non-null:", int((df["geometry_ok"] & df[f3].notna().all(axis=1)).sum()))
print("geometry_qc_class:", df["geometry_qc_class"].value_counts(dropna=False).to_dict())
print("geometry_source:", df["geometry_source"].value_counts(dropna=False).to_dict())
print("geometry_status:", df["geometry_status"].value_counts(dropna=False).to_dict())
print("complex_pi_image_index non-null:", int(df["complex_pi_image_index"].notna().sum()), " vr_graph_index non-null:", int(df["vr_graph_index"].notna().sum()),
      " ligand_pi_control_image_index non-null:", int(df["ligand_pi_control_image_index"].notna().sum()))
print("per-metal geometry_ok:", df.groupby("metal")["geometry_ok"].sum().to_dict())
print("distinct geometry_key with geometry_ok:", df.loc[df["geometry_ok"], "geometry_key"].nunique())

print("\n### (7) features/ SIDE TABLES")
for name in ["complex_physical_scalars", "coordination_polyhedron", "features_by_safe_exp_id", "row_asset_index"]:
    t = pd.read_parquet(f"features/{name}.parquet")
    print(f"-- {name}: shape={t.shape}")
    print("   columns:", list(t.columns)[:60], "..." if t.shape[1] > 60 else "")
    for k in ["safe_exp_id", "geometry_key", "geometry_feature_build_id"]:
        if k in t.columns:
            print(f"   {k}: nunique={t[k].nunique()} nulls={int(t[k].isna().sum())}")
    nnz = t.notna().sum()
    print("   fully-null columns:", [c for c in t.columns if t[c].isna().all()])
    print("   non-null count range over columns:", int(nnz.min()), int(nnz.max()))
st = pd.read_csv("features/geometry_feature_status.csv")
print("geometry_feature_status.csv shape", st.shape, "cols", list(st.columns))
for c in st.columns:
    if st[c].nunique() < 10: print("   ", c, st[c].value_counts(dropna=False).to_dict())
for npz in ["complex_gfn2xtb_pi_images", "ligand_pi_control_images", "vietoris_rips_inputs"]:
    z = np.load(f"features/{npz}.npz", allow_pickle=True)
    print(f"-- {npz}.npz keys:", {k: (z[k].shape, str(z[k].dtype)) for k in z.files})

print("\n### (8) LIGAND SIDE FILES")
for fn in ["ligand_2d_descriptors.parquet", "ligand_pretrained_embeddings.parquet"]:
    if os.path.exists(fn):
        t = pd.read_parquet(fn)
        fams = collections.Counter(c.split("__")[1] if "__" in c else c for c in t.columns)
        print(f"-- {fn}: exists shape={t.shape}; column families={dict(fams)}; canonical_smiles nunique={t['canonical_smiles'].nunique() if 'canonical_smiles' in t else 'n/a'}")
        print("   smiles overlap with dataset:", len(set(t["canonical_smiles"]) & set(df["canonical_smiles"])) if "canonical_smiles" in t else "n/a")
    else:
        print(f"-- {fn}: MISSING")
for fn in ["ligand_2d_descriptors.manifest.json", "ligand_pretrained_embeddings.manifest.json"]:
    if os.path.exists(fn):
        m = json.load(open(fn))
        print(f"-- {fn}: keys={list(m.keys())}", {k: m[k] for k in m if k not in ("columns",)} if fn.startswith("ligand_2d") else m)

print("\n### (9) PROVENANCE")
stem = df["safe_exp_id"].astype(str).str.split(":").str[0]
print("safe_exp_id stem counts:", stem.value_counts().to_dict())
print("safe_exp_id examples:", df["safe_exp_id"].head(5).tolist())
num = df["safe_exp_id"].astype(str).str.split(":").str[1].astype(float)
print("safe_exp_id numeric part: min", num.min(), "max", num.max(), "n", num.nunique())
prov_like = [c for c in cols if re.search(r"doi|source|ref|cite|paper|pub|author|year|journal|origin|dataset|file", c, re.I)]
print("provenance-like columns:", prov_like)
for c in prov_like:
    print("   ", c, "nunique", df[c].nunique(), "examples", df[c].dropna().unique()[:5].tolist())
print("build_id:", df["build_id"].unique().tolist())
print("feature_block_manifest values:", df["feature_block_manifest"].value_counts(dropna=False).head(5).to_dict())
# per-stem metal coverage
print("stem x metal crosstab:")
print(pd.crosstab(stem, df["metal"]).to_string())
print("stem x n_extractants:", df.assign(stem=stem).groupby("stem")["canonical_smiles"].nunique().to_dict())
# Does contiguous safe_exp_id block map to extractant runs? check adjacency of ids: consecutive ids sharing extractant
d2 = df.assign(num=num, stem=stem).sort_values(["stem", "num"])
same_next = (d2["canonical_smiles"].values[1:] == d2["canonical_smiles"].values[:-1]) & (d2["stem"].values[1:] == d2["stem"].values[:-1])
print("fraction of consecutive safe_exp_id (sorted) sharing extractant:", round(float(same_next.mean()), 4))
# top extractants
top = df.groupby(["extractant_name"]).agg(rows=("log_D","size"), n_metals=("metal","nunique")).sort_values("rows", ascending=False).head(15)
print("top-15 extractants by rows:")
print(top.to_string())
print("DONE")
