"""Is donor-set compactness a real, transferable determinant of lanthanide selectivity?

The candidate descriptors are graph distances between donor atoms (counts of bonds, per
gen12_2_eu_pred/COORDINATION_DESCRIPTOR_SPEC.md), so a small value means the donors sit close
together along the molecular graph - a short chelate bite. The claim to test is that a compact
donor set gives heavy-lanthanide selectivity.

Unit of scoring: the extractant. Unit of independence: the frozen gen6 chemotype. Every interval
is a chemotype-blocked bootstrap; every claim is also checked with the dominant diglycolamide
chemotype removed, on chemotype means, and after controlling for the acid the cells were run in.
"""
import sys
sys.path.insert(0, "gen13_separation")
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
from gen13sep.amplitude_bench import load_bench, LEAN_BLOCKS

RNG = np.random.default_rng(8675309)
REPS = 10000

bench = load_bench()
f = bench.frame.copy()
f["amp"] = bench.coef[:, 0]
rich = f[f.n_metals >= 5].copy()
X = pd.DataFrame(bench.matrix(LEAN_BLOCKS), columns=bench.columns(LEAN_BLOCKS), index=f.index)

ext = rich.groupby("extractant").agg(amp=("amp", "mean"), n_cells=("amp", "size"),
                                     chemo=("chemotype", "first"),
                                     name=("extractant_name", "first"))
Xe = X.loc[rich.index].groupby(rich["extractant"].to_numpy()).mean().loc[ext.index]
ext["acid_hno3"] = Xe["cond__acid__hno3"] if "cond__acid__hno3" in Xe else np.nan
print(f"{len(ext)} extractants, {ext.chemo.nunique()} chemotypes, "
      f"{int(ext.n_cells.sum())} cells with >=5 metals")
print(f"amplitude: mean {ext.amp.mean():+.3f}, sd {ext.amp.std():.3f}, "
      f"negative (heavy-selective) in {(ext.amp < 0).mean()*100:.0f}% of extractants\n")

FEATURES = ["coord__dist__frac_donor_pairs_within_3", "coord__dist__donor_pair_median",
            "coord__dist__donor_pair_min", "coord__dist__donor_eccentricity_min",
            "coord__arm__core_branch_degree", "coord__dist__donor_pair_mean",
            "coord__arm__mean_donor_distance_to_core", "coord__dist__donor_network_diameter"]
FEATURES = [c for c in FEATURES if c in Xe.columns]

chemos = ext.chemo.to_numpy()
names = sorted(set(chemos))
members = [np.flatnonzero(chemos == c) for c in names]


def blocked_spearman_ci(x, y, mask):
    xs, ys = x[mask], y[mask]
    cm = chemos[mask]
    nm = sorted(set(cm)); mem = [np.flatnonzero(cm == c) for c in nm]
    point = spearmanr(xs, ys).statistic
    draws = np.empty(REPS)
    picks = RNG.integers(0, len(nm), size=(REPS, len(nm)))
    for i, row in enumerate(picks):
        idx = np.concatenate([mem[j] for j in row])
        if len(set(xs[idx])) < 3 or len(set(ys[idx])) < 3:
            draws[i] = np.nan; continue
        draws[i] = spearmanr(xs[idx], ys[idx]).statistic
    d = draws[np.isfinite(draws)]
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return point, np.quantile(d, 0.025), np.quantile(d, 0.975), p, len(nm)


def partial_spearman(x, y, z, mask):
    """Spearman of x and y after removing the linear effect of z on both, on ranks."""
    xs, ys, zs = rankdata(x[mask]), rankdata(y[mask]), rankdata(z[mask])
    Z = np.c_[np.ones(len(zs)), zs]
    rx = xs - Z @ np.linalg.lstsq(Z, xs, rcond=None)[0]
    ry = ys - Z @ np.linalg.lstsq(Z, ys, rcond=None)[0]
    return spearmanr(rx, ry).statistic


rows = []
for c in FEATURES:
    v = Xe[c].to_numpy(dtype=float); a = ext.amp.to_numpy()
    ok = np.isfinite(v) & np.isfinite(a)
    point, lo, hi, p, nch = blocked_spearman_ci(v, a, ok)
    # leave-one-chemotype-out
    loco = []
    for cc in names:
        m = ok & (chemos != cc)
        if m.sum() > 10 and len(set(v[m])) > 2:
            loco.append(spearmanr(v[m], a[m]).statistic)
    # chemotype means
    g = pd.DataFrame({"c": chemos[ok], "v": v[ok], "a": a[ok]}).groupby("c").mean()
    rho_chemo = spearmanr(g.v, g.a).statistic
    # outside the dominant diglycolamide chemotype
    m2 = ok & (chemos != "sc009")
    rho_out = spearmanr(v[m2], a[m2]).statistic if m2.sum() > 10 else np.nan
    # controlling for the acid the cells were run in
    acid = ext.acid_hno3.to_numpy(dtype=float)
    m3 = ok & np.isfinite(acid)
    rho_pa = partial_spearman(v, a, acid, m3) if m3.sum() > 10 else np.nan
    rows.append({"feature": c, "n_extractants": int(ok.sum()), "n_chemotypes": nch,
                 "spearman": point, "ci_low": lo, "ci_high": hi, "p_blocked": p,
                 "loco_min": float(np.min(loco)), "loco_max": float(np.max(loco)),
                 "spearman_chemotype_means": rho_chemo, "n_chemotype_means": int(len(g)),
                 "spearman_excl_sc009": rho_out, "n_excl_sc009": int(m2.sum()),
                 "partial_spearman_given_acid": rho_pa})
res = pd.DataFrame(rows).sort_values("spearman", key=abs, ascending=False)
res.to_csv("gen13_separation/analysis/stage3/s3_geometry_statistics.csv", index=False)
pd.set_option("display.width", 250)
print(res.round(3).to_string(index=False))
