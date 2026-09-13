"""Independent re-derivation of the D3 ceiling-ladder headline numbers.

Written from scratch without reading the D3 agent's script.
Checks:
  (1) the extractant-macro scorer reproduces the programme reference arm numbers
  (2) L1 leave-one-cell-out corpus mean centred curve
  (3) L3 leave-one-cell-out extractant mean centred curve (the headline rung)
  (4) the like-for-like subset where L3/L4/L5 are all defined, arms re-scored on it
  (5) L7 oracle rank-1 amplitude with an LOO corpus shape
Also probes: centring convention, big-cell domination, out-of-sample status.
"""
import os
import sys
import numpy as np
import pandas as pd

ROOT = "D:/ml_separator_gh"
COHORT = os.path.join(ROOT, "generations/gen13_separation/manifests/cohort_exact.parquet")
PRED = os.path.join(ROOT, "generations/gen13_separation/predictions/B_primary")
OUT = os.path.join(ROOT, "generations/gen13_separation/analysis/stage2/d3_ceiling_ladder/verify")
os.makedirs(OUT, exist_ok=True)

LN = ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]
Z = dict(zip(LN, [57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]))
IDX = {m: i for i, m in enumerate(LN)}

# ---------------------------------------------------------------- load cohort
coh = pd.read_parquet(COHORT)
cells = coh["cell_id"].to_numpy()
ext = coh["extractant"].to_numpy()
pub = coh["publication_id"].to_numpy()
chemo = coh["chemotype"].to_numpy()
acid = coh["cond__acid_concentration_M"].to_numpy(dtype=float)
X = coh[[f"logD__{m}" for m in LN]].to_numpy(dtype=float)   # raw logD, NaN where unmeasured
OBS = ~np.isnan(X)
NC = len(cells)
ci = {c: i for i, c in enumerate(cells)}

# centred curve: subtract the mean over the metals OBSERVED IN THAT CELL
row_mean = np.nanmean(np.where(OBS, X, np.nan), axis=1)
C = X - row_mean[:, None]                      # NaN preserved where unobserved
# alternative (wrong) convention for a sensitivity probe: centre over all 14 present values
# is identical here because nanmean already ignores missing; the real alternative is to
# NOT centre at all, or to centre against the corpus grand mean.  Handled below.

print(f"cohort: {NC} cells, {coh.extractant.nunique()} extractants, "
      f"{coh.chemotype.nunique()} chemotypes, {int(OBS.sum())} observed (cell,metal) entries")

# ---------------------------------------------------------------- build pairs
rows = []
for i in range(NC):
    ms = [m for m in LN if OBS[i, IDX[m]]]
    for a in range(len(ms)):
        for b in range(a + 1, len(ms)):
            A, B = ms[a], ms[b]
            rows.append((cells[i], i, A, B, Z[B] - Z[A], X[i, IDX[A]] - X[i, IDX[B]]))
P = pd.DataFrame(rows, columns=["cell_id", "ri", "A", "B", "dZ", "y"])
P["extractant"] = ext[P.ri.to_numpy()]
P["chemotype"] = chemo[P.ri.to_numpy()]
print(f"pairs rebuilt: {len(P)} over {P.cell_id.nunique()} cells / {P.extractant.nunique()} extractants")

# ------------------------------------------------------- scorer + arm reference
def ext_macro(df, err="ae"):
    """mean over extractants of the mean |error| over that extractant's pairs"""
    return df.groupby("extractant", observed=True)[err].mean().mean()


def score_arm(name):
    p = pd.read_parquet(os.path.join(PRED, name + ".parquet"))
    p["ae"] = (p["y"] - p["prediction"]).abs()
    per_seed = p.groupby("split_seed").apply(
        lambda d: d.groupby("extractant", observed=True)["ae"].mean().mean(), include_groups=False)
    pooled = p.groupby("split_seed")["ae"].mean()
    return float(per_seed.mean()), float(per_seed.std(ddof=1)), float(pooled.mean()), p


print("\n=== (1) scorer validation: arms re-scored from saved B_primary predictions ===")
arm_rows = []
ARMS = ["X_ENS_DIRECT+LOWRANK_K2", "C_DIRECT_ROW", "M_SELECTED", "B1_MEAN_CURVE", "B4_HEAVIER_ALWAYS"]
arm_pred = {}
for a in ARMS:
    m, s, pool, p = score_arm(a)
    arm_pred[a] = p
    arm_rows.append(dict(arm=a, ext_macro=round(m, 4), seed_sd=round(s, 4), pooled=round(pool, 4),
                         n_ext=p.extractant.nunique(), n_pairs_per_seed=len(p) // 5))
    print(f"  {a:28s} ext-macro {m:.4f} (seed sd {s:.4f})  pooled {pool:.4f}")
pd.DataFrame(arm_rows).to_csv(os.path.join(OUT, "v_arm_reference.csv"), index=False)

# cross-check that the saved y equals my independently rebuilt y
ref = arm_pred["X_ENS_DIRECT+LOWRANK_K2"]
ref1 = ref[ref.split_seed == ref.split_seed.iloc[0]]
mrg = ref1.merge(P[["cell_id", "A", "B", "y"]], on=["cell_id", "A", "B"], suffixes=("_saved", "_mine"))
print(f"  pair reconstruction: matched {len(mrg)} / {len(ref1)} saved pairs, "
      f"max |y_saved - y_mine| = {np.abs(mrg.y_saved - mrg.y_mine).max():.2e}")
# is each saved pair genuinely held out exactly once per seed?
dup = ref1.groupby(["cell_id", "A", "B"]).size()
print(f"  each pair appears once per seed: {bool((dup == 1).all())}; folds per seed = {ref1.fold.nunique()}")
# out-of-sample check: does a cell's chemotype appear in its own fold only?
chk = ref1.groupby("chemotype")["fold"].nunique()
print(f"  chemotype -> distinct folds (must be 1 for a chemotype hold-out): max = {int(chk.max())}")

# --------------------------------------------------- donor-mean ladder rungs
def donor_curve_matrix(group_key, extra_mask=None):
    """For each cell, the mean centred curve over OTHER cells in its group.
    Returns (NC,14) array with NaN where no donor observed that metal."""
    D = np.full((NC, 14), np.nan)
    Cz = np.where(OBS, C, 0.0)
    keys = pd.Series(group_key)
    for k, idx in keys.groupby(keys).groups.items():
        idx = np.asarray(list(idx))
        if len(idx) < 2:
            continue
        sub_sum = Cz[idx].sum(axis=0)
        sub_cnt = OBS[idx].sum(axis=0)
        for i in idx:
            s = sub_sum - Cz[i]
            n = sub_cnt - OBS[i]
            with np.errstate(invalid="ignore", divide="ignore"):
                D[i] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return D


def loo_corpus_curve():
    """leave-one-cell-out mean centred curve over the whole corpus"""
    Cz = np.where(OBS, C, 0.0)
    tot = Cz.sum(axis=0)
    cnt = OBS.sum(axis=0)
    D = np.full((NC, 14), np.nan)
    for i in range(NC):
        n = cnt - OBS[i]
        s = tot - Cz[i]
        D[i] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return D


def nearest_acid_curve():
    """donor = the same-extractant other cell(s) with the smallest |delta acid conc|"""
    D = np.full((NC, 14), np.nan)
    dacid = np.full(NC, np.nan)
    same_pub_share = np.full(NC, np.nan)
    s = pd.Series(ext)
    for k, idx in s.groupby(s).groups.items():
        idx = np.asarray(list(idx))
        if len(idx) < 2:
            continue
        for i in idx:
            others = idx[idx != i]
            if np.isnan(acid[i]):
                d = np.full(len(others), 0.0)
            else:
                d = np.abs(acid[others] - acid[i])
                d = np.where(np.isnan(d), np.inf, d)
            if not np.isfinite(d).any():
                sel = others
                dacid[i] = np.nan
            else:
                sel = others[d == d.min()]
                dacid[i] = d.min()
            sm = OBS[sel].sum(axis=0)
            vv = np.where(OBS[sel], C[sel], 0.0).sum(axis=0)
            D[i] = np.where(sm > 0, vv / np.maximum(sm, 1), np.nan)
            same_pub_share[i] = float(np.mean(pub[sel] == pub[i]))
    return D, dacid, same_pub_share


def predict_from_curve(D, tag):
    ri = P.ri.to_numpy()
    ia = P.A.map(IDX).to_numpy()
    ib = P.B.map(IDX).to_numpy()
    va = D[ri, ia]
    vb = D[ri, ib]
    return pd.Series(va - vb, index=P.index, name=tag)


print("\n=== (2)/(3) ladder rungs re-derived (leave-one-cell-out) ===")
D_corpus = loo_corpus_curve()
D_ext = donor_curve_matrix(ext)
D_pub = donor_curve_matrix([f"{e}||{p}" for e, p in zip(ext, pub)])
D_chem = donor_curve_matrix(chemo)
D_acid, dacid, pubshare = nearest_acid_curve()

lad = {"L1_corpus": D_corpus, "L2_chemotype": D_chem, "L3_extractant": D_ext,
       "L4_ext_pub": D_pub, "L5_ext_acid": D_acid}
for tag, D in lad.items():
    P[tag] = predict_from_curve(D, tag)

lad_rows = []
for tag in lad:
    d = P[["extractant", "chemotype", "cell_id", "y", tag]].dropna().copy()
    d["ae"] = (d["y"] - d[tag]).abs()
    em = ext_macro(d)
    cm = d.groupby("chemotype", observed=True)["ae"].mean().mean()
    lad_rows.append(dict(rung=tag, ext_macro=round(em, 4), chemo_macro=round(cm, 4),
                         pooled=round(d.ae.mean(), 4), n_ext=d.extractant.nunique(),
                         n_cells=d.cell_id.nunique(), n_pairs=len(d)))
    print(f"  {tag:14s} ext-macro {em:.4f}  chemo-macro {cm:.4f}  pooled {d.ae.mean():.4f}  "
          f"({d.extractant.nunique()} ext / {d.cell_id.nunique()} cells / {len(d)} pairs)")
pd.DataFrame(lad_rows).to_csv(os.path.join(OUT, "v_ladder_full.csv"), index=False)

# multi-cell extractant census
cnt_ext = pd.Series(ext).value_counts()
multi = cnt_ext[cnt_ext >= 2]
print(f"  extractants with >=2 cells: {len(multi)} of {cnt_ext.size}, covering {int(multi.sum())} of {NC} cells")

# ------------------------------------------- like-for-like common subset
print("\n=== (4) like-for-like subset where L3, L4, L5 are all defined ===")
mask = P[["L3_extractant", "L4_ext_pub", "L5_ext_acid"]].notna().all(axis=1)
S = P[mask].copy()
print(f"  subset: {S.cell_id.nunique()} cells / {S.extractant.nunique()} extractants / {len(S)} pairs")

sub_rows = []
for tag in ["L1_corpus", "L2_chemotype", "L3_extractant", "L4_ext_pub", "L5_ext_acid"]:
    d = S[["extractant", "y", tag]].dropna().copy()
    d["ae"] = (d["y"] - d[tag]).abs()
    sub_rows.append(dict(predictor=tag, ext_macro=round(ext_macro(d), 4),
                         pooled=round(d.ae.mean(), 4), n_pairs=len(d)))
    print(f"  {tag:14s} ext-macro {ext_macro(d):.4f}  pooled {d.ae.mean():.4f}  n={len(d)}")

key = set(zip(S.cell_id, S.A, S.B))
for a in ARMS:
    p = arm_pred[a]
    k = pd.Series(list(zip(p.cell_id, p.A, p.B)))
    sel = p[k.isin(key).to_numpy()].copy()
    sel["ae"] = (sel["y"] - sel["prediction"]).abs()
    per_seed = sel.groupby("split_seed").apply(
        lambda d: d.groupby("extractant", observed=True)["ae"].mean().mean(), include_groups=False)
    sub_rows.append(dict(predictor="ARM_" + a, ext_macro=round(float(per_seed.mean()), 4),
                         pooled=round(float(sel.ae.mean()), 4), n_pairs=len(sel) // 5))
    print(f"  ARM {a:24s} ext-macro {per_seed.mean():.4f}  pooled {sel.ae.mean():.4f}  "
          f"n/seed={len(sel)//5}  seed sd {per_seed.std(ddof=1):.4f}")
pd.DataFrame(sub_rows).to_csv(os.path.join(OUT, "v_ladder_common_subset.csv"), index=False)

# ------------------------------------------- (5) L7 oracle rank-1 amplitude
print("\n=== (5) L7 oracle rank-1 amplitude, LOO corpus shape (masked ALS) ===")


def als_rank1(Cm, Om, iters=60):
    a = np.nanstd(np.where(Om, Cm, np.nan), axis=1)
    a = np.nan_to_num(a, nan=0.0)
    v = np.nanmean(np.where(Om, Cm, np.nan), axis=0)
    v = np.nan_to_num(v, nan=0.0)
    if np.linalg.norm(v) == 0:
        v = np.ones(Cm.shape[1])
    v /= np.linalg.norm(v)
    Cz = np.where(Om, Cm, 0.0)
    for _ in range(iters):
        den = (Om * (v ** 2)[None, :]).sum(axis=1)
        a = np.where(den > 0, (Cz * v[None, :]).sum(axis=1) / np.maximum(den, 1e-12), 0.0)
        den2 = (Om * (a ** 2)[:, None]).sum(axis=0)
        v = np.where(den2 > 0, (Cz * a[:, None]).sum(axis=0) / np.maximum(den2, 1e-12), 0.0)
        nv = np.linalg.norm(v)
        if nv == 0:
            break
        v /= nv
    return v


V = np.zeros((NC, 14))
for i in range(NC):
    keep = np.ones(NC, bool)
    keep[i] = False
    V[i] = als_rank1(C[keep], OBS[keep])

# oracle scalar amplitude for the held-out cell, fitted on its own observed metals,
# with the shape re-centred over exactly those metals (pair predictions are
# invariant to that re-centring, the fit is not)
pred7 = np.full((NC, 14), np.nan)
for i in range(NC):
    o = OBS[i]
    if o.sum() < 2:
        continue
    v = V[i].copy()
    v = v - v[o].mean()
    y = C[i][o]
    den = float((v[o] ** 2).sum())
    a = float((y * v[o]).sum() / den) if den > 1e-12 else 0.0
    pred7[i] = a * v

P["L7_oracle_amp"] = predict_from_curve(pred7, "L7")
d = P[["extractant", "chemotype", "cell_id", "y", "L7_oracle_amp"]].dropna().copy()
d["ae"] = (d["y"] - d["L7_oracle_amp"]).abs()
print(f"  L7 ext-macro {ext_macro(d):.4f}  chemo-macro "
      f"{d.groupby('chemotype', observed=True)['ae'].mean().mean():.4f}  "
      f"pooled {d.ae.mean():.4f}  ({d.extractant.nunique()} ext / {d.cell_id.nunique()} cells / {len(d)} pairs)")
pd.DataFrame([dict(rung="L7_oracle_amp", ext_macro=round(ext_macro(d), 4),
                   pooled=round(d.ae.mean(), 4), n_ext=d.extractant.nunique(),
                   n_cells=d.cell_id.nunique(), n_pairs=len(d))]).to_csv(
    os.path.join(OUT, "v_l7_oracle_amplitude.csv"), index=False)

# ------------------------------------------- (6) robustness probes
print("\n=== (6) probes ===")
# a. does one big cell dominate an extractant's score?
sz = P.groupby(["extractant", "cell_id"]).size().reset_index(name="np")
tot = sz.groupby("extractant")["np"].sum()
top = sz.groupby("extractant")["np"].max()
share = (top / tot)
print(f"  largest-cell share of an extractant's pairs: median {share.median():.2f}, "
      f"mean {share.mean():.2f}, {int((share > 0.5).sum())} of {share.size} extractants >50%")
# cell-macro variant of the headline (equal weight per cell, then per extractant)
d3 = P[["extractant", "cell_id", "y", "L3_extractant"]].dropna().copy()
d3["ae"] = (d3["y"] - d3["L3_extractant"]).abs()
cellmac = d3.groupby(["extractant", "cell_id"])["ae"].mean().groupby("extractant").mean().mean()
print(f"  L3 ext-macro {ext_macro(d3):.4f} vs cell-then-extractant macro {cellmac:.4f}")
d3s = S[["extractant", "cell_id", "y", "L3_extractant"]].dropna().copy()
d3s["ae"] = (d3s["y"] - d3s["L3_extractant"]).abs()
cellmac_s = d3s.groupby(["extractant", "cell_id"])["ae"].mean().groupby("extractant").mean().mean()
print(f"  L3 on subset: ext-macro {ext_macro(d3s):.4f} vs cell-then-extractant macro {cellmac_s:.4f}")

# b. centring convention sensitivity: build donor curves from UNCENTRED logD
Craw = np.where(OBS, X, np.nan)
Draw = np.full((NC, 14), np.nan)
Rz = np.where(OBS, X, 0.0)
s = pd.Series(ext)
for k, idx in s.groupby(s).groups.items():
    idx = np.asarray(list(idx))
    if len(idx) < 2:
        continue
    ss, cc = Rz[idx].sum(axis=0), OBS[idx].sum(axis=0)
    for i in idx:
        n = cc - OBS[i]
        Draw[i] = np.where(n > 0, (ss - Rz[i]) / np.maximum(n, 1), np.nan)
praw = predict_from_curve(Draw, "raw")
draw = pd.DataFrame(dict(extractant=P.extractant, y=P.y, p=praw)).dropna()
draw["ae"] = (draw.y - draw.p).abs()
print(f"  L3 built from UNCENTRED logD donors: ext-macro {ext_macro(draw):.4f} "
      f"(centred version {ext_macro(d3):.4f}) -- pair differences are centring-invariant, "
      f"so these must agree exactly")

# c. donor proximity caveat
have = ~np.isnan(D_ext).all(axis=1)
print(f"  L5 donor proximity: median |delta acid| = {np.nanmedian(dacid[have]):.3f} M, "
      f"exact match for {100*np.nanmean(dacid[have] == 0):.1f}% of cells, "
      f"mean same-publication donor share {100*np.nanmean(pubshare[have]):.1f}%")

# d. replicate columns
NREP = coh[[f"nrep__{m}" for m in LN]].to_numpy(dtype=float)
REPSD = coh[[f"repsd__{m}" for m in LN]].to_numpy(dtype=float)
hasrep = (NREP >= 2) & OBS
sd = REPSD[hasrep & ~np.isnan(REPSD)]
print(f"  replicates: {int(hasrep.sum())} of {int(OBS.sum())} observed (cell,metal) entries with nrep>=2, "
      f"in {int((hasrep.any(axis=1)).sum())} of {NC} cells; repsd q25 {np.nanquantile(sd,.25):.3f} "
      f"median {np.nanmedian(sd):.3f} q75 {np.nanquantile(sd,.75):.3f} mean {np.nanmean(sd):.3f}")

P.to_csv(os.path.join(OUT, "v_pairs_with_ladder.csv"), index=False)
print("\ndone")
