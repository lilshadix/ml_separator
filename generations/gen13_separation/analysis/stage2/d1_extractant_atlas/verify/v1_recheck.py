"""Independent re-derivation of the headline numbers of diagnostic d1 (curve atlas).

Written from scratch; does not read the reviewed agent's scripts.

Checks
  A. the extractant-macro convention itself, by reproducing the known reference
     numbers 0.481 (best ensemble arm) / 0.495 (C_DIRECT_ROW) / 0.603 (mean curve)
     from gen13_separation/predictions/B_primary/.
  B. the headline oracle: copy a sibling cell of the SAME extractant -> claimed
     0.286 extractant-macro log-SF MAE over 24 extractants / 450 cells.
  C. the control: copy a cell of a DIFFERENT extractant in the same chemotype ->
     claimed 0.441; paired subset claimed 0.257 own vs 0.454 other (gain 0.196).
  D. like-for-like: score the real held-out model arm on exactly the (cell, pair)
     population the oracle covers, so the 0.286-vs-0.481 comparison can be judged.
  E. the amplitude variance split: claimed 44.0% between / 56.0% within extractant,
     ICC 0.415 over 21 extractants / 228 cells (>=5 observed metals per cell).
  F. centring sanity: centred curve must be centred on the metals observed in the
     cell, not on all 14.

Run from the repo root D:/ml_separator_gh.
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "gen13_separation")
from gen13sep.metals import LANTHANIDES, ATOMIC_NUMBER, SHANNON_RADIUS_CN8  # noqa: E402

OUT = "gen13_separation/analysis/stage2/d1_extractant_atlas/verify"
os.makedirs(OUT, exist_ok=True)

COHORT = "gen13_separation/manifests/cohort_exact.parquet"
PRED = "gen13_separation/predictions/B_primary"

pd.set_option("display.width", 200)


def hr(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# ----------------------------------------------------------------- load cohort
df = pd.read_parquet(COHORT)
LN = [m for m in LANTHANIDES if m != "Pm"]
LN = [m for m in LN if f"logD__{m}" in df.columns]
assert len(LN) == 14, LN
# order by atomic number (light -> heavy)
LN = sorted(LN, key=lambda m: ATOMIC_NUMBER[m])
print("lanthanide axis:", LN)

logD = df[[f"logD__{m}" for m in LN]].to_numpy(float)
cell_ids = df["cell_id"].to_numpy()
extr = df["extractant"].to_numpy()
chemo = df["chemotype"].to_numpy()
pub = df["publication_id"].to_numpy()
n_cells = len(df)
obs = ~np.isnan(logD)
print(f"cells={n_cells}  observed (cell,metal) entries={obs.sum()}  "
      f"extractants={df.extractant.nunique()}  chemotypes={df.chemotype.nunique()}")
print("metals per cell: median %.1f mean %.3f; n_metals col matches obs: %s"
      % (np.median(obs.sum(1)), obs.sum(1).mean(),
         bool((obs.sum(1) == df.n_metals.to_numpy()).all())))

# --------------------------------------------------------- F. centring check
# centred curve of a cell = logD minus mean over THAT cell's observed metals
cent = logD - np.nanmean(logD, axis=1, keepdims=True)
# a cell's centred curve must sum to ~0 over its observed metals
sums = np.nansum(cent, axis=1)
hr("F. centring")
print("max |sum of centred curve over observed metals| = %.3e (0 => centred per cell)"
      % np.nanmax(np.abs(sums)))
# what a WRONG centring (over all 14, i.e. treating NaN as needing global mean) would do
print("note: log SF = logD(A)-logD(B) is invariant to any per-cell centring, so the")
print("      SF-based numbers below cannot be affected by the centring choice.")

# ------------------------------------------------------- pair table per cell
PAIRS = [(a, b) for i, a in enumerate(LN) for b in LN[i + 1:]]  # Z_A < Z_B
IDX = {m: i for i, m in enumerate(LN)}


# ============================================================ A. convention
def extractant_macro(pred_df, err_col="abs_err"):
    """Programme convention: mean within extractant over its held-out pairs,
    then mean over extractants, then mean over the 5 split seeds."""
    per = (pred_df.groupby(["split_seed", "extractant"])[err_col].mean()
           .groupby("split_seed").mean())
    return per.mean(), per


hr("A. reproduce the known reference numbers (anchors the extractant-macro convention)")
ref_rows = []
for arm in ["X_ENS_DIRECT+LOWRANK_K2", "X_ENS_DIRECT+PHYSICS", "C_DIRECT_ROW",
            "M_SELECTED", "B1_MEAN_CURVE", "B4_HEAVIER_ALWAYS"]:
    p = pd.read_parquet(f"{PRED}/{arm}.parquet")
    p["abs_err"] = (p["y"] - p["prediction"]).abs()
    macro, _ = extractant_macro(p)
    pooled = p["abs_err"].mean()
    ref_rows.append(dict(arm=arm, macro=macro, pooled=pooled, n_rows=len(p)))
    print(f"{arm:32s} macro={macro:.4f}  pooled={pooled:.4f}  rows={len(p)}")
ref = pd.DataFrame(ref_rows)
ref.to_csv(f"{OUT}/v1_reference_arms.csv", index=False)

BEST_ARM = "X_ENS_DIRECT+LOWRANK_K2"
best_macro = float(ref.loc[ref.arm == BEST_ARM, "macro"].iloc[0])


# ================================================ B/C. sibling-copy oracles
def sf_errors_for_pair_of_cells(i, j):
    """abs errors of predicting cell i's log SFs by copying cell j.
    Returns array of |(ci_A-ci_B) - (cj_A-cj_B)| over metals shared by both."""
    m = obs[i] & obs[j]
    k = int(m.sum())
    if k < 2:
        return None
    v = logD[i][m] - logD[j][m]           # difference curve on shared metals
    # log SF error for pair (a,b) = (di_a-di_b)-(dj_a-dj_b) = v_a - v_b
    d = v[:, None] - v[None, :]
    iu = np.triu_indices(k, 1)
    return np.abs(d[iu])


def build_transfer(mask_fn, label):
    """For every target cell i, for every donor cell j allowed by mask_fn(i,j),
    collect per-(target,donor) mean abs SF error and the pair count."""
    recs = []
    for i in range(n_cells):
        for j in range(n_cells):
            if i == j:
                continue
            if not mask_fn(i, j):
                continue
            e = sf_errors_for_pair_of_cells(i, j)
            if e is None:
                continue
            recs.append(dict(target=cell_ids[i], donor=cell_ids[j],
                             extractant=extr[i], chemotype=chemo[i],
                             same_pub=bool(pub[i] == pub[j]),
                             n_pairs=len(e), sum_abs=float(e.sum()),
                             mean_abs=float(e.mean())))
    r = pd.DataFrame(recs)
    print(f"[{label}] target-donor cell pairs = {len(r)}, "
          f"target cells covered = {r.target.nunique()}, "
          f"extractants = {r.extractant.nunique()}")
    return r


own = build_transfer(lambda i, j: extr[i] == extr[j], "same extractant")
ctrl = build_transfer(lambda i, j: (chemo[i] == chemo[j]) and (extr[i] != extr[j]),
                      "same chemotype, different extractant")


def macro_variants(r, label):
    """Several defensible aggregations, to see which yields the claimed number
    and whether one big cell can dominate."""
    out = {}
    # cell-level score first: mean over that cell's donors of the per-donor mean
    cell = (r.groupby(["extractant", "target"])
              .apply(lambda g: pd.Series({
                  "cell_mean_of_donor_means": g.mean_abs.mean(),
                  "cell_pairpooled": g.sum_abs.sum() / g.n_pairs.sum(),
                  "n_donors": len(g), "n_pairs": g.n_pairs.sum()}),
                     include_groups=False)
              .reset_index())
    out["macro_cellmean_of_donormeans"] = (
        cell.groupby("extractant").cell_mean_of_donor_means.mean().mean())
    out["macro_cellpairpooled"] = (
        cell.groupby("extractant").cell_pairpooled.mean().mean())
    # pool all (target,donor) pairs inside the extractant, unweighted by pair count
    out["macro_donormean_pooled_in_extr"] = (
        r.groupby("extractant").mean_abs.mean().mean())
    # pool every individual SF observation inside the extractant (pair-weighted)
    g = r.groupby("extractant")[["sum_abs", "n_pairs"]].sum()
    out["macro_pairweighted_in_extr"] = (g.sum_abs / g.n_pairs).mean()
    # fully pooled over all SF observations (cell-weighted / pooled)
    out["pooled_all_sf"] = r.sum_abs.sum() / r.n_pairs.sum()
    out["pooled_donormeans"] = r.mean_abs.mean()
    out["n_extractants"] = r.extractant.nunique()
    out["n_target_cells"] = r.target.nunique()
    print(f"\n[{label}] aggregation variants")
    for k, v in out.items():
        print(f"   {k:34s} = {v:.4f}" if isinstance(v, float) else f"   {k:34s} = {v}")
    return out, cell


hr("B. same-extractant sibling-copy oracle (claimed 0.286 extractant-macro, "
   "24 extractants, 450/521 cells; pooled 0.295)")
own_v, own_cell = macro_variants(own, "same extractant")
own_cell.to_csv(f"{OUT}/v1_own_extractant_cell_scores.csv", index=False)

hr("C. control: different extractant, same chemotype (claimed 0.441 over 56 extractants)")
ctrl_v, ctrl_cell = macro_variants(ctrl, "same chemotype other extractant")
ctrl_cell.to_csv(f"{OUT}/v1_control_cell_scores.csv", index=False)

# paired subset: extractants where both are defined
own_e = own.groupby("extractant").mean_abs.mean()
ctrl_e = ctrl.groupby("extractant").mean_abs.mean()
both = own_e.index.intersection(ctrl_e.index)
print(f"\nextractants where both defined: {len(both)}  "
      f"own={own_e[both].mean():.4f}  other={ctrl_e[both].mean():.4f}  "
      f"gain={ctrl_e[both].mean()-own_e[both].mean():.4f}")
pair_tab = pd.DataFrame({"own": own_e[both], "other": ctrl_e[both]})
pair_tab["gain"] = pair_tab.other - pair_tab.own
pair_tab.to_csv(f"{OUT}/v1_paired_own_vs_other.csv")

# different-publication variant (claimed 0.390 over 9 extractants)
own_dp = own[~own.same_pub]
e_dp = own_dp.groupby("extractant").mean_abs.mean()
print(f"different-publication sibling: extractants={len(e_dp)} macro={e_dp.mean():.4f}")
own_sp = own[own.same_pub]
print(f"same-publication sibling:      extractants={own_sp.extractant.nunique()} "
      f"macro={own_sp.groupby('extractant').mean_abs.mean().mean():.4f}")
print(f"pooled SF MAE same-pub={own_sp.sum_abs.sum()/own_sp.n_pairs.sum():.4f} "
      f"diff-pub={own_dp.sum_abs.sum()/own_dp.n_pairs.sum():.4f}")

# ============================================ D. like-for-like model comparison
hr("D. is 0.286 comparable to 0.481? score the real arm on the SAME population")
covered_cells = set(own.target.unique())
covered_extr = set(own.extractant.unique())
p = pd.read_parquet(f"{PRED}/{BEST_ARM}.parquet")
p["abs_err"] = (p["y"] - p["prediction"]).abs()
rows = []
rows.append(("all cells / all extractants (published ref)",
             extractant_macro(p)[0], len(p)))
sub = p[p.cell_id.isin(covered_cells)]
rows.append((f"restricted to the {len(covered_cells)} oracle-covered cells",
             extractant_macro(sub)[0], len(sub)))
sub2 = p[p.extractant.isin(covered_extr)]
rows.append((f"restricted to the {len(covered_extr)} oracle-covered extractants",
             extractant_macro(sub2)[0], len(sub2)))
# and the exact (cell, A, B) pairs the oracle is scored on
oracle_keys = set()
for i in range(n_cells):
    donors = [j for j in range(n_cells) if j != i and extr[j] == extr[i]]
    if not donors:
        continue
    for a_i in range(14):
        for b_i in range(a_i + 1, 14):
            if not (obs[i][a_i] and obs[i][b_i]):
                continue
            if any(obs[j][a_i] and obs[j][b_i] for j in donors):
                oracle_keys.add((cell_ids[i], LN[a_i], LN[b_i]))
p["key"] = list(zip(p.cell_id, p.A, p.B))
sub3 = p[p.key.isin(oracle_keys)]
rows.append((f"restricted to the exact {len(oracle_keys)} oracle-covered (cell,A,B)",
             extractant_macro(sub3)[0], len(sub3)))
for name, val, n in rows:
    print(f"  {name:58s} macro={val:.4f}  rows={n}")
pd.DataFrame(rows, columns=["scope", "macro_mae", "n_rows"]).to_csv(
    f"{OUT}/v1_arm_on_oracle_population.csv", index=False)

# also: the oracle scored the way the arm is (pair-level, per-cell single donor is
# not defined) -- report the strongest and weakest donor choices as bounds
best_donor = own.groupby(["extractant", "target"]).mean_abs.min()
worst_donor = own.groupby(["extractant", "target"]).mean_abs.max()
print(f"  oracle if the BEST donor were picked per cell (upper oracle): "
      f"{best_donor.groupby('extractant').mean().mean():.4f}")
print(f"  oracle if the WORST donor were picked per cell:               "
      f"{worst_donor.groupby('extractant').mean().mean():.4f}")

# ================================================ E. amplitude variance split
hr("E. amplitude ICC (claimed 44.0% between / 56.0% within, ICC 0.415, "
   "21 extractants / 228 cells, from 289 cells with >=5 metals)")
r8 = np.array([SHANNON_RADIUS_CN8[m] for m in LN])
z_glob = (r8 - r8.mean()) / r8.std(ddof=0)

fits = []
for i in range(n_cells):
    m = obs[i]
    if m.sum() < 5:
        continue
    y = cent[i][m]
    z = z_glob[m]
    X = np.column_stack([np.ones(m.sum()), z, z ** 2])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ((y - yhat) ** 2).sum() / ss_tot if ss_tot > 0 else np.nan
    Xl = np.column_stack([np.ones(m.sum()), z])
    bl, *_ = np.linalg.lstsq(Xl, y, rcond=None)
    r2l = 1 - ((y - Xl @ bl) ** 2).sum() / ss_tot if ss_tot > 0 else np.nan
    fits.append(dict(cell_id=cell_ids[i], extractant=extr[i],
                     n_metals=int(m.sum()), amplitude=beta[1], curvature=beta[2],
                     curve_sd=float(np.std(y, ddof=0)), r2=r2, r2_linear=r2l))
F = pd.DataFrame(fits)
F.to_csv(f"{OUT}/v1_cell_quadratic_fits.csv", index=False)
print(f"fitted cells (>=5 observed metals) = {len(F)}")
print(f"amplitude mean={F.amplitude.mean():.4f} sd={F.amplitude.std(ddof=1):.4f} "
      f"min={F.amplitude.min():.4f} max={F.amplitude.max():.4f} "
      f"pct_negative={100*(F.amplitude<0).mean():.1f}%")
print(f"R2 quad mean={F.r2.mean():.4f} median={F.r2.median():.4f} ; "
      f"R2 linear-only mean={F.r2_linear.mean():.4f}")


def icc_oneway(vals, groups):
    """One-way random effects ICC(1) with unbalanced groups (Shrout & Fleiss)."""
    d = pd.DataFrame({"v": vals, "g": groups})
    d = d.groupby("g").filter(lambda g: len(g) >= 2)
    k = d.g.nunique()
    N = len(d)
    gm = d.v.mean()
    means = d.groupby("g").v.mean()
    ns = d.groupby("g").v.size()
    ssb = float((ns * (means - gm) ** 2).sum())
    ssw = float(((d.v - d.g.map(means)) ** 2).sum())
    msb = ssb / (k - 1)
    msw = ssw / (N - k)
    n0 = (N - (ns ** 2).sum() / N) / (k - 1)
    var_b = max((msb - msw) / n0, 0.0)
    icc = var_b / (var_b + msw) if (var_b + msw) > 0 else np.nan
    return dict(k_groups=k, n=N, ss_between=ssb, ss_within=ssw,
                pct_ss_between=100 * ssb / (ssb + ssw),
                pct_ss_within=100 * ssw / (ssb + ssw),
                sd_between=np.sqrt(var_b), sd_within=np.sqrt(msw), icc=icc)


vd = []
for name in ["amplitude", "curvature", "curve_sd"]:
    res = icc_oneway(F[name].to_numpy(), F.extractant.to_numpy())
    res["quantity"] = name
    vd.append(res)
    print(f"{name:10s} groups={res['k_groups']} n={res['n']} "
          f"%SS_between={res['pct_ss_between']:.1f} %SS_within={res['pct_ss_within']:.1f} "
          f"sd_b={res['sd_between']:.3f} sd_w={res['sd_within']:.3f} ICC={res['icc']:.3f}")
pd.DataFrame(vd).to_csv(f"{OUT}/v1_variance_decomposition.csv", index=False)

# bootstrap the amplitude ICC over extractants to gauge its uncertainty
rng = np.random.default_rng(0)
amp = F[["amplitude", "extractant"]]
grp = [g for g, gg in amp.groupby("extractant") if len(gg) >= 2]
boots = []
for _ in range(400):
    pick = rng.choice(len(grp), len(grp), replace=True)
    parts = []
    for bi, gi in enumerate(pick):
        gg = amp[amp.extractant == grp[gi]].copy()
        gg["extractant"] = f"b{bi}"
        parts.append(gg)
    bb = pd.concat(parts)
    boots.append(icc_oneway(bb.amplitude.to_numpy(), bb.extractant.to_numpy())["icc"])
boots = np.array(boots)
print(f"amplitude ICC bootstrap over extractants: median={np.median(boots):.3f} "
      f"90% CI = [{np.percentile(boots,5):.3f}, {np.percentile(boots,95):.3f}] (400 reps)")

# per-extractant amplitude table (spot-check the quoted per-extractant sds)
tab = (F.groupby("extractant")
         .agg(n_cells=("amplitude", "size"), amp_mean=("amplitude", "mean"),
              amp_sd=("amplitude", lambda s: s.std(ddof=1)))
         .sort_values("n_cells", ascending=False))
name_map = df.drop_duplicates("extractant").set_index("extractant").extractant_name
tab["name"] = tab.index.map(name_map)
tab.to_csv(f"{OUT}/v1_amplitude_by_extractant.csv")
print("\ntop amplitude groups:")
print(tab.head(14).to_string())

# ------------------------------------------------------------ summary dump
hr("SUMMARY: my numbers vs the claims")
claims = [
    ("best-arm extractant-macro MAE", 0.481, best_macro),
    ("C_DIRECT_ROW extractant-macro MAE", 0.495,
     float(ref.loc[ref.arm == "C_DIRECT_ROW", "macro"].iloc[0])),
    ("B1_MEAN_CURVE extractant-macro MAE", 0.603,
     float(ref.loc[ref.arm == "B1_MEAN_CURVE", "macro"].iloc[0])),
    ("own-extractant transfer, macro (donor-mean)", 0.286,
     own_v["macro_donormean_pooled_in_extr"]),
    ("own-extractant transfer, macro (cell-first)", 0.286,
     own_v["macro_cellmean_of_donormeans"]),
    ("own-extractant transfer, pooled", 0.295, own_v["pooled_all_sf"]),
    ("control transfer, macro", 0.441, ctrl_v["macro_donormean_pooled_in_extr"]),
    ("paired own", 0.257, float(own_e[both].mean())),
    ("paired other", 0.454, float(ctrl_e[both].mean())),
    ("different-publication sibling, macro", 0.390, float(e_dp.mean())),
    ("amplitude %SS within extractant", 56.0, vd[0]["pct_ss_within"]),
    ("amplitude ICC", 0.415, vd[0]["icc"]),
    ("curvature ICC", 0.188, vd[1]["icc"]),
    ("curve-sd ICC", 0.496, vd[2]["icc"]),
    ("quad R2 mean", 0.876, float(F.r2.mean())),
    ("linear-only R2 mean", 0.744, float(F.r2_linear.mean())),
    ("pct amplitude negative", 75.8, float(100 * (F.amplitude < 0).mean())),
]
rows = []
for nm, claimed, mine in claims:
    rel = abs(mine - claimed) / abs(claimed) * 100 if claimed else np.nan
    rows.append(dict(quantity=nm, claimed=claimed, mine=round(mine, 4),
                     abs_diff=round(mine - claimed, 4), pct_diff=round(rel, 2),
                     within_5pct=bool(rel <= 5)))
    print(f"{nm:46s} claimed={claimed:>8.3f}  mine={mine:>8.4f}  "
          f"diff={mine-claimed:+.4f} ({rel:5.2f}%)  {'OK' if rel <= 5 else 'MISMATCH'}")
pd.DataFrame(rows).to_csv(f"{OUT}/v1_claim_vs_mine.csv", index=False)
print("\nwrote:", OUT)
