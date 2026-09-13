"""D6 step 1: which cond__ columns move within an extractant, and how much
within-extractant amplitude variation each one could explain."""
from __future__ import annotations
import numpy as np
import pandas as pd
from d6_common import (load_cohort, cond_columns, cond_taxonomy, cell_amplitudes,
                       collapse_onehot, OUT, MIN_RZ_SPAN)

pd.set_option("display.width", 200)

df = load_cohort()
cols = cond_columns(df)
tax = cond_taxonomy(cols)
print("=== cond__ column families ===")
print(tax.groupby(["family", "kind"]).size().rename("n_columns"))

amp_all = cell_amplitudes(df)
cell_all = df.merge(amp_all, on="cell_id", how="inner")
print("\n=== amplitude conditioning: amplitude sd vs standardised-radius span of the "
      "measured metal set ===")
band = pd.cut(cell_all.rz_span, [0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.2])
print(cell_all.groupby(band, observed=True).agg(n_cells=("amp", "size"),
                                                amp_sd=("amp", "std"),
                                                median_amp_se=("amp_se", "median"))
      .to_string(float_format=lambda v: f"{v:.3f}"))
cell = cell_all[cell_all.rz_span >= MIN_RZ_SPAN].copy()
print(f"\ncells total {len(df)}, with >=4 metals {len(cell_all)}, "
      f"and radius span >= {MIN_RZ_SPAN} (analysis set) {len(cell)}")
print(f"extractants represented {cell.extractant.nunique()} of {df.extractant.nunique()}")

# --- within-extractant variation counts, over ALL 521 cells ---------------------------
rows = []
gall = df.groupby("extractant")
multi_ext = [k for k, g in gall if len(g) >= 2]
n_multi_cells = int(sum(len(gall.get_group(k)) for k in multi_ext))
for c in cols:
    n_ext, n_cells, n_lvls = 0, 0, []
    for k in multi_ext:
        g = gall.get_group(k)
        u = g[c].dropna().nunique()
        if u >= 2:
            n_ext += 1
            n_cells += len(g)
            n_lvls.append(u)
    rows.append(dict(column=c, n_ext_varying=n_ext, n_cells_in_varying_ext=n_cells,
                     median_levels_where_varying=float(np.median(n_lvls)) if n_lvls else 0.0,
                     n_missing=int(df[c].isna().sum()),
                     n_levels_global=int(df[c].dropna().nunique())))
var = pd.DataFrame(rows).merge(tax, on="column")
print(f"\nextractants with >=2 cells: {len(multi_ext)}, cells in them: {n_multi_cells}")

# --- how much within-extractant amplitude SS each column could explain ---------------
# reference: SS of amplitude after removing each extractant's mean amplitude
sub = cell[cell.groupby("extractant").cell_id.transform("size") >= 3].copy()
sub["amp_res"] = sub.amp - sub.groupby("extractant").amp.transform("mean")
SS_within = float((sub.amp_res ** 2).sum())
print(f"\nwithin-extractant SS of amplitude = {SS_within:.3f} over {len(sub)} cells "
      f"in {sub.extractant.nunique()} extractants "
      f"(between-extractant SS = {float(((sub.amp - sub.amp.mean())**2).sum()) - SS_within:.3f})")

expl = []
for c in cols:
    s = sub[[c, "extractant", "amp_res"]].dropna(subset=[c]).copy()
    # keep only extractants where this column actually varies AND that still have >=3
    # cells after dropping missing values (so a nested slope has >=1 residual dof)
    keep = (s.groupby("extractant")[c].transform("nunique") >= 2) & \
           (s.groupby("extractant")[c].transform("size") >= 3)
    s = s[keep]
    if len(s) < 3 or s.extractant.nunique() == 0:
        expl.append(dict(column=c, ss_explained_global_slope=0.0, ss_explained_nested=0.0,
                         n_cells_used=len(s), n_ext_used=int(s.extractant.nunique())))
        continue
    x = s[c].astype(float)
    xr = x - s.groupby("extractant")[c].transform("mean").astype(float)
    y = s.amp_res.to_numpy()
    # global single slope on the within-extractant centred predictor
    den = float((xr ** 2).sum())
    ss_glob = float((xr.to_numpy() @ y) ** 2 / den) if den > 1e-12 else 0.0
    # extractant-nested slope (one slope per extractant)
    ss_nest = 0.0
    for k, g in s.groupby("extractant"):
        xg = g[c].astype(float).to_numpy()
        xg = xg - xg.mean()
        yg = g.amp_res.to_numpy()
        d = float(xg @ xg)
        if d > 1e-12:
            ss_nest += float((xg @ yg) ** 2 / d)
    expl.append(dict(column=c, ss_explained_global_slope=ss_glob,
                     ss_explained_nested=ss_nest, n_cells_used=len(s),
                     n_ext_used=int(s.extractant.nunique())))
expl = pd.DataFrame(expl)
expl["frac_within_SS_global"] = expl.ss_explained_global_slope / SS_within
expl["frac_within_SS_nested"] = expl.ss_explained_nested / SS_within
rank = var.merge(expl, on="column").sort_values("frac_within_SS_nested", ascending=False)
rank["SS_within_reference"] = SS_within
rank.to_csv(OUT + "/d6_step1_cond_column_ranking.csv", index=False)

print("\n=== top 20 cond columns by within-extractant amplitude SS explained (nested slope) ===")
show = ["column", "family", "n_ext_varying", "n_cells_in_varying_ext", "n_ext_used",
        "n_cells_used", "frac_within_SS_global", "frac_within_SS_nested"]
print(rank[show].head(20).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

print("\n=== the condition families the task names explicitly, as joint blocks ===")
# collapse the one-hot blocks to single categoricals so a block is not double counted
sub["acid_id"] = collapse_onehot(df.set_index("cell_id").loc[sub.cell_id].reset_index(),
                                 "cond__acid__", "acid_id").to_numpy()
sub["diluent_id"] = collapse_onehot(df.set_index("cell_id").loc[sub.cell_id].reset_index(),
                                    "cond__diluent__", "diluent_id").to_numpy()
sub["additive_id"] = collapse_onehot(df.set_index("cell_id").loc[sub.cell_id].reset_index(),
                                     "cond__additive__", "additive_id").to_numpy()
sigma2_null = SS_within / max(len(sub) - sub.extractant.nunique(), 1)

named = {
    "acid concentration (log10 M)": ("num", "cond__acid_concentration_M"),
    "acid identity (9 one-hots)": ("cat", "acid_id"),
    "extractant concentration (log10 M)": ("num", "cond__extractant_concentration_M"),
    "temperature (C)": ("num", "cond__temperature_C"),
    "diluent identity (41 one-hots)": ("cat", "diluent_id"),
    "contact time (min)": ("num", "cond__contact_time_min"),
    "aqueous salt / ionic strength": ("none", None),
    "metal loading (log10 mM)": ("num", "cond__metal_concentration_mM"),
    "additive identity (9 one-hots)": ("cat", "additive_id"),
    "measured metal set (design nuisance)": ("cat", "metal_set"),
    "publication_id (provenance nuisance)": ("cat", "publication_id"),
}
LOGCOLS = {"cond__acid_concentration_M", "cond__extractant_concentration_M",
           "cond__metal_concentration_mM"}
fam_rows = []
for label, (kind, key) in named.items():
    if kind == "none":
        fam_rows.append(dict(family=label, n_ext_used=0, n_cells_used=0, df_used=0,
                             ss_explained=np.nan, frac_within_SS=np.nan,
                             frac_within_SS_dof_adj=np.nan,
                             note="NO SUCH COLUMN EXISTS in the 64-column condition block"))
        continue
    s = sub.dropna(subset=[key]).copy()
    if kind == "num" and key in LOGCOLS:
        s[key] = np.log10(s[key].astype(float))
    ok = (s.groupby("extractant")[key].transform("nunique") >= 2) & \
         (s.groupby("extractant")[key].transform("size") >= 3)
    s = s[ok]
    ss, dfu = 0.0, 0
    for k, g in s.groupby("extractant"):
        y = g.amp.to_numpy() - g.amp.to_numpy().mean()
        if kind == "num":
            X = g[key].astype(float).to_numpy()[:, None]
            X = X - X.mean(0)
        else:
            X = pd.get_dummies(g[key].astype(str), drop_first=True).to_numpy(float)
            if X.size == 0:
                continue
            X = X - X.mean(0)
        r = np.linalg.matrix_rank(X)
        if r == 0 or len(g) - 1 - r < 1:   # keep >=1 residual dof
            continue
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        ss += float((y @ y) - ((y - X @ beta) ** 2).sum())
        dfu += r
    fam_rows.append(dict(family=label, n_ext_used=int(s.extractant.nunique()),
                         n_cells_used=int(len(s)), df_used=dfu, ss_explained=ss,
                         frac_within_SS=ss / SS_within,
                         frac_within_SS_dof_adj=max(ss - dfu * sigma2_null, 0.0) / SS_within,
                         note=""))
fam = pd.DataFrame(fam_rows)
fam["SS_within_reference"] = SS_within
fam.to_csv(OUT + "/d6_step1_named_families.csv", index=False)
print(f"(nested within extractant; sigma2 under the null = {sigma2_null:.4f})")
print(fam.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# per-extractant summary of how many conditions move
per_ext = []
for k, g in gall:
    if len(g) < 2:
        continue
    a_all = cell_all[cell_all.extractant == k]
    nvary = sum(1 for c in cols if g[c].dropna().nunique() >= 2)
    a = cell[cell.extractant == k]
    per_ext.append(dict(extractant=k, extractant_name=g.extractant_name.iloc[0],
                        chemotype=g.chemotype.iloc[0], n_cells=len(g),
                        n_cells_amp_raw=len(a_all),
                        n_cells_with_amp=len(a),
                        n_cond_varying=nvary,
                        n_acid_conc_levels=int(g["cond__acid_concentration_M"].dropna().nunique()),
                        n_extconc_levels=int(g["cond__extractant_concentration_M"].dropna().nunique()),
                        n_temp_levels=int(g["cond__temperature_C"].dropna().nunique()),
                        n_publications=int(g.publication_id.nunique()),
                        amp_mean=float(a.amp.mean()) if len(a) else np.nan,
                        amp_sd=float(a.amp.std(ddof=1)) if len(a) > 1 else np.nan))
per_ext = pd.DataFrame(per_ext).sort_values("n_cells", ascending=False)
per_ext.to_csv(OUT + "/d6_step1_per_extractant_condition_span.csv", index=False)
print("\n=== per-extractant condition span (extractants with >=2 cells) ===")
print(per_ext.drop(columns=["extractant"]).to_string(index=False,
                                                     float_format=lambda v: f"{v:.3f}"))

# amplitude reference scale
print(f"\namplitude over the {len(cell)} analysis cells: mean {cell.amp.mean():.3f}, "
      f"sd {cell.amp.std(ddof=1):.3f}, median |amp| {cell.amp.abs().median():.3f}")
print(f"median per-cell amplitude standard error (from the quadratic fit): "
      f"{cell.amp_se.median():.3f} over {cell.amp_se.notna().sum()} cells")
print(f"amplitude sd within extractant (rms of amp_res over {len(sub)} cells): "
      f"{np.sqrt(SS_within / len(sub)):.3f}")
keepcols = ["cell_id", "extractant", "extractant_name", "chemotype", "publication_id",
            "n_metals", "amp", "quad", "amp_lin", "amp_se", "rz_span", "rmse_fit",
            "curve_rms", "metal_set", "cond__acid_concentration_M",
            "cond__extractant_concentration_M", "cond__temperature_C",
            "cond__contact_time_min", "cond__metal_concentration_mM"]
cell_all["in_analysis_set"] = cell_all.rz_span >= MIN_RZ_SPAN
cell_all[keepcols + ["in_analysis_set"]].to_csv(OUT + "/d6_cell_amplitudes.csv",
                                                index=False)
print("\nwrote d6_cell_amplitudes.csv, d6_step1_cond_column_ranking.csv, "
      "d6_step1_named_families.csv, d6_step1_per_extractant_condition_span.csv")
