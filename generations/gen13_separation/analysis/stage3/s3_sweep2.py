"""Second sweep: which *part* of the compact chemistry carries the transferable amplitude, and does
predicting the amplitude alone beat predicting amplitude and curvature together?

The extractant-level correlates say the strongest single signals are donor-donor distance geometry
(frac_donor_pairs_within_3 rho -0.53, donor_eccentricity_min +0.51, donor_pair_median +0.50,
donor_pair_min +0.48), i.e. how tightly the donor set is packed around the metal - the cavity-size
argument. Curvature meanwhile has an extractant ICC of 0.19 and is mostly noise, so a model that
spends capacity on it may be paying for nothing.

Every feature choice that depends on the target is made INSIDE the training fold.
"""
import sys, time, re
sys.path.insert(0, "generations/gen13_separation")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from gen13sep.amplitude_bench import (load_bench, compare, LEAN_BLOCKS, CHEM_BLOCKS)
from gen13sep.models import tree_pipeline

DESIGN = sys.argv[1] if len(sys.argv) > 1 else "BP"
bench = load_bench()
lean_cols = bench.columns(LEAN_BLOCKS)
GEOM = [c for c in lean_cols if c.startswith("coord__dist__") or c.startswith("coord__arm__")]
DONOR_COUNTS = [c for c in lean_cols if "donor__" in c or c in ("chem__dentate", "chem__core_cn",
                                                                "chem__n_ligands", "chem__n_fill")]
print(f"geometry columns {len(GEOM)}, donor-count columns {len(DONOR_COUNTS)}", flush=True)


def subset_by_name(names):
    idx = np.array([lean_cols.index(c) for c in names], dtype=int)
    def f(Xtr, coef, w, g, Xte, seed):
        m = tree_pipeline(seed, n_estimators=400, max_features=0.5, min_samples_leaf=2)
        m.fit(Xtr[:, idx], coef, extratreesregressor__sample_weight=w)
        return m.predict(Xte[:, idx])
    return f


def infold_select(k=12, target=0):
    """Univariate Spearman screen computed on the training fold only, then trees on the survivors."""
    def f(Xtr, coef, w, g, Xte, seed):
        y = coef[:, target]
        scores = np.zeros(Xtr.shape[1])
        for j in range(Xtr.shape[1]):
            v = Xtr[:, j]
            ok = np.isfinite(v)
            if ok.sum() < 20 or len(np.unique(v[ok])) < 3:
                continue
            r = spearmanr(v[ok], y[ok]).statistic
            scores[j] = 0.0 if not np.isfinite(r) else abs(r)
        idx = np.argsort(-scores)[:k]
        m = tree_pipeline(seed, n_estimators=400, max_features=0.7, min_samples_leaf=2)
        m.fit(Xtr[:, idx], coef, extratreesregressor__sample_weight=w)
        return m.predict(Xte[:, idx])
    return f


def amplitude_only(names=None, curvature="mean"):
    """Predict the radius coefficient only; take the curvature from the training fold."""
    idx = None if names is None else np.array([lean_cols.index(c) for c in names], dtype=int)
    def f(Xtr, coef, w, g, Xte, seed):
        Z, Zt = (Xtr, Xte) if idx is None else (Xtr[:, idx], Xte[:, idx])
        m = tree_pipeline(seed, n_estimators=400, max_features=0.5, min_samples_leaf=2)
        m.fit(Z, coef[:, 0], extratreesregressor__sample_weight=w)
        a = np.asarray(m.predict(Zt), dtype=float)
        if curvature == "zero":
            b = np.zeros_like(a)
        elif curvature == "mean":
            b = np.full_like(a, float(np.average(coef[:, 1], weights=w)))
        else:                                   # b = kappa * a, kappa from the training fold
            denom = float(np.average(coef[:, 0] ** 2, weights=w))
            kappa = float(np.average(coef[:, 0] * coef[:, 1], weights=w)) / max(denom, 1e-9)
            b = kappa * a
        return np.c_[a, b]
    return f


CANDIDATES = {
    "L_lean":            subset_by_name(lean_cols),
    "M_geom":            subset_by_name(GEOM),
    "N_donorcounts":     subset_by_name(DONOR_COUNTS),
    "O_geom+counts":     subset_by_name(sorted(set(GEOM) | set(DONOR_COUNTS))),
    "P_select12":        infold_select(12),
    "Q_select24":        infold_select(24),
    "R_select6":         infold_select(6),
    "S_amponly_lean":    amplitude_only(None, "mean"),
    "T_amponly_kappa":   amplitude_only(None, "kappa"),
    "U_amponly_geom":    amplitude_only(GEOM, "kappa"),
    "V_amponly_gc":      amplitude_only(sorted(set(GEOM) | set(DONOR_COUNTS)), "kappa"),
}

t0 = time.time()
board, table = compare(bench, CANDIDATES, blocks=LEAN_BLOCKS, design=DESIGN)
cols = ["arm", "macro_mae_extractant", "macro_mae_extractant_seed_sd", "macro_mae_chemotype",
        "macro_mae_far", "macro_sign_acc_strong", "macro_pair_spearman"]
print(f"\n{time.time()-t0:.0f}s\n")
print(board[cols].round(4).to_string(index=False))
board.to_csv(f"generations/gen13_separation/analysis/stage3/s3_sweep2_{DESIGN}.csv", index=False)
table.to_parquet(f"generations/gen13_separation/analysis/stage3/s3_sweep2_{DESIGN}_pairs.parquet", index=False)
