"""REFUTER lens A, part 2: the mandatory checks and the four lens-specific attacks on L1NULL."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from refute_l1null_a import (ROOT, OUT, PHYS3D, load, fit_all, choose_series, varying,  # noqa: E402
                            slope_fit, rho, partial_rho, two_way, species_cov, ELEM_COLS)

TGT = ["a", "abs_a", "b"]


def fisher_ci(r, n, alpha=0.05):
    if not np.isfinite(r) or n < 5:
        return np.nan, np.nan
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1.0 / np.sqrt(n - 3)
    zc = stats.norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - zc * se)), float(np.tanh(z + zc * se))


def power_spearman(rho_true, n, alpha=0.05):
    """Power of the two-sided Fisher-z test of rho=0 at sample size n (Spearman ~ 1.06/sqrt(n-3))."""
    if n < 5:
        return np.nan
    se = 1.06 / np.sqrt(n - 3)
    z = np.arctanh(rho_true) / se
    zc = stats.norm.ppf(1 - alpha / 2)
    return float(stats.norm.sf(zc - z) + stats.norm.cdf(-zc - z))


def blocked_boot(x, y, chem, reps=2000, seed=20260909, min_rows=20):
    uch = np.unique(chem)
    idx = {c: np.flatnonzero(chem == c) for c in uch}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        b = np.concatenate([idx[c] for c in rng.choice(uch, size=len(uch), replace=True)])
        bb = b[np.isfinite(x[b]) & np.isfinite(y[b])]
        if len(bb) < min_rows or len(np.unique(x[bb])) < 5:
            continue
        r = stats.spearmanr(x[bb], y[bb]).statistic
        if np.isfinite(r):
            vals.append(r)
    if len(vals) <= 100:
        return np.nan, np.nan, len(vals)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi), len(vals)


def main():
    g = load()
    rows, info = fit_all(g)
    df = pd.read_csv(OUT / "A_slopes.csv")
    out = {}
    recs = []

    W = df.pivot_table(index="extractant", columns="model", values="slope", aggfunc="first")
    N = df.pivot_table(index="extractant", columns="model", values="n_computed_metals", aggfunc="first")
    t = pd.read_parquet(PHYS3D / "extractant_targets.parquet")[
        ["extractant", "a", "b", "abs_a", "n_metals", "chemotype"]].set_index("extractant")
    W = W.join(t, how="inner")
    N = N.reindex(W.index)
    base_models = ["NAIVE", "ELEM", "SPECIES", "CYCLEHAT"]

    # ---------- CHECK 6a: byte-identical scored unit sets across the compared arms ----------
    s8 = {m: set(N.index[N[m] >= 8]) for m in base_models + ["SPECIES_CONST"]}
    out["S8_sizes"] = {m: len(v) for m, v in s8.items()}
    out["S8_identical_NAIVE_ELEM_SPECIES_CYCLEHAT"] = all(s8[m] == s8["NAIVE"] for m in base_models)
    out["S8_SPECIES_CONST_is_subset_of_S8"] = s8["SPECIES_CONST"] <= s8["NAIVE"]
    idx62 = sorted(s8["NAIVE"])
    idx19 = sorted(s8["SPECIES_CONST"])
    idx43 = sorted(set(idx62) - set(idx19))

    # ---------- CHECK 4 / lens (b) argument-2 attack: matched sets ----------
    for name, idx in (("S8_all62", idx62), ("S8_const19", idx19), ("S8_varying43", idx43)):
        w = W.loc[idx]
        for m in base_models + ["SPECIES_CONST"]:
            for tg in TGT:
                r, p, n = rho(w[m].to_numpy(float), w[tg].to_numpy(float))
                lo, hi = fisher_ci(r, n)
                recs.append({"check": "matched_subset", "subset": name, "model": m, "target": tg,
                             "n": n, "rho": r, "p": p, "fisher_lo": lo, "fisher_hi": hi})

    # ---------- CHECK 1: diglycolamides (sc009) removed ----------
    for name, idx in (("S8_all62", idx62), ("S8_const19", idx19)):
        w = W.loc[idx]
        wn = w[w["chemotype"] != "sc009"]
        for m in base_models + ["SPECIES_CONST"]:
            for tg in TGT:
                r, p, n = rho(wn[m].to_numpy(float), wn[tg].to_numpy(float))
                recs.append({"check": "no_sc009", "subset": name, "model": m, "target": tg,
                             "n": n, "rho": r, "p": p, "fisher_lo": fisher_ci(r, n)[0],
                             "fisher_hi": fisher_ci(r, n)[1]})
        recs.append({"check": "no_sc009", "subset": name, "model": "_n_sc009_dropped", "target": "",
                     "n": int((w["chemotype"] == "sc009").sum()), "rho": np.nan, "p": np.nan,
                     "fisher_lo": np.nan, "fisher_hi": np.nan})

    # ---------- CHECK 2: n_metals confound, stratified ----------
    w = W.loc[idx62]
    med = float(np.median(w["n_metals"]))
    for lab, sub in (("n_metals<=med", w[w["n_metals"] <= med]), ("n_metals>med", w[w["n_metals"] > med])):
        for m in base_models:
            r, p, n = rho(sub[m].to_numpy(float), sub["a"].to_numpy(float))
            recs.append({"check": "n_metals_stratum", "subset": lab, "model": m, "target": "a",
                         "n": n, "rho": r, "p": p, "fisher_lo": np.nan, "fisher_hi": np.nan})
    out["n_metals_median_S8"] = med
    out["rho_n_metals_vs_abs_a_S8"] = rho(w["n_metals"].to_numpy(float), w["abs_a"].to_numpy(float))[0]
    for m in base_models:
        out[f"rho_slope_{m}_vs_n_metals_S8"] = rho(w[m].to_numpy(float), w["n_metals"].to_numpy(float))[0]

    # ---------- CHECK 5: permutation null, different seed; and a shuffled-target control ----------
    reg_cols = []
    for m in ["NAIVE", "ELEM", "SPECIES", "SPECIES_CONST"]:
        for st, thr in (("S8", 8), ("S14", 14), ("S3", 3)):
            c = f"{m}__{st}"
            v = W[m].where(N[m] >= thr if st != "S14" else N[m] == 14)
            W[c] = v
            reg_cols.append(c)
    cm = W.groupby("chemotype")[reg_cols + TGT].mean()
    M = cm[reg_cols].to_numpy(float)
    for seed in (20260909, 777001, 424243):
        rng = np.random.default_rng(seed)
        for tg in TGT:
            y = cm[tg].to_numpy(float)
            obs = []
            for j in range(M.shape[1]):
                ok = np.isfinite(M[:, j]) & np.isfinite(y)
                obs.append(stats.spearmanr(M[ok, j], y[ok]).statistic if ok.sum() >= 10 else np.nan)
            obs = np.array(obs, float)
            null = np.empty(2000)
            for k in range(2000):
                yp = rng.permutation(y)
                best = 0.0
                for j in range(M.shape[1]):
                    ok = np.isfinite(M[:, j]) & np.isfinite(yp)
                    if ok.sum() < 10:
                        continue
                    rr = abs(stats.spearmanr(M[ok, j], yp[ok]).statistic)
                    best = max(best, rr) if np.isfinite(rr) else best
                null[k] = best
            recs.append({"check": "perm_null", "subset": f"seed{seed}", "model": reg_cols[int(np.nanargmax(np.abs(obs)))],
                         "target": tg, "n": len(cm), "rho": float(np.nanmax(np.abs(obs))),
                         "p": float((null >= np.nanmax(np.abs(obs))).mean()),
                         "fisher_lo": float(np.percentile(null, 95)), "fisher_hi": np.nan})

    # ---------- CHECK 7 / (a): power of the n=19 control ----------
    out["power_n19"] = {f"rho_true={rt}": power_spearman(rt, 19) for rt in (0.3, 0.4, 0.5, 0.6, 0.644)}
    out["fisher_ci_SPECIES_CONST_S8"] = fisher_ci(0.16666666666666666, 19)
    out["fisher_ci_SPECIES_S8"] = fisher_ci(-0.08367958500163683, 62)
    out["fisher_ci_NAIVE_S8"] = fisher_ci(0.392007, 62)
    # is +0.167 distinguishable from +0.5?  z-test of the difference at n=19 vs the fixed value
    z = (np.arctanh(0.5) - np.arctanh(0.16667)) * np.sqrt(19 - 3) / 1.06
    out["z_0.167_vs_0.5_at_n19"] = float(z)
    out["p_0.167_vs_0.5_at_n19"] = float(2 * stats.norm.sf(abs(z)))
    # blocked bootstrap for the n=19 row with the frozen 20-row minimum relaxed to 10
    w19 = W.loc[idx19]
    lo, hi, nb = blocked_boot(w19["SPECIES_CONST"].to_numpy(float), w19["a"].to_numpy(float),
                              w19["chemotype"].to_numpy(), min_rows=10)
    out["SPECIES_CONST_S8_blocked_CI_minrows10"] = [lo, hi, nb]
    lo2, hi2, nb2 = blocked_boot(w19["NAIVE"].to_numpy(float), w19["a"].to_numpy(float),
                                 w19["chemotype"].to_numpy(), min_rows=10)
    out["NAIVE_on_same19_blocked_CI_minrows10"] = [lo2, hi2, nb2]
    # SPECIES S8 blocked CI, different seed (check 5 / 7)
    w62 = W.loc[idx62]
    for sd in (20260909, 777001):
        out[f"SPECIES_S8_blocked_CI_seed{sd}"] = blocked_boot(
            w62["SPECIES"].to_numpy(float), w62["a"].to_numpy(float), w62["chemotype"].to_numpy(), seed=sd)
        out[f"CYCLEHAT_S8_blocked_CI_seed{sd}"] = blocked_boot(
            w62["CYCLEHAT"].to_numpy(float), w62["a"].to_numpy(float), w62["chemotype"].to_numpy(), seed=sd)
        out[f"NAIVE_S8_blocked_CI_seed{sd}"] = blocked_boot(
            w62["NAIVE"].to_numpy(float), w62["a"].to_numpy(float), w62["chemotype"].to_numpy(), seed=sd)

    # ---------- lens (b): identifiability, attenuation, de-attenuation ----------
    so = choose_series(g)
    ident = []
    for smi in idx62:
        ser = so[smi]
        s = rows[rows["series"] == ser]
        rv = s["r"].to_numpy(float)
        rec = {"extractant": smi, "series": ser, "n": len(s)}
        for c in ("n_ligs", "n_NO3", "n_H2O", "coreCN"):
            v = s[c].to_numpy(float)
            rec[f"var_{c}"] = int(np.std(v) > 0)
            if np.std(v) > 0 and np.std(rv) > 0:
                rec[f"corr_{c}_r"] = float(np.corrcoef(v, rv)[0, 1])
                A = np.c_[np.ones(len(rv)), rv - rv.mean(), (rv - rv.mean()) ** 2]
                bb, *_ = np.linalg.lstsq(A, v, rcond=None)
                ss = ((v - v.mean()) ** 2).sum()
                rec[f"r2_{c}_on_rr2"] = float(1 - ((v - A @ bb) ** 2).sum() / ss) if ss > 0 else np.nan
            else:
                rec[f"corr_{c}_r"] = np.nan
                rec[f"r2_{c}_on_rr2"] = np.nan
        ident.append(rec)
    I = pd.DataFrame(ident).set_index("extractant")
    I.to_csv(OUT / "A_identifiability.csv")
    out["n_S8_with_varying_nligs"] = int(I["var_n_ligs"].sum())
    out["n_S8_with_varying_nNO3"] = int(I["var_n_NO3"].sum())
    out["n_S8_with_varying_nH2O"] = int(I["var_n_H2O"].sum())
    out["n_const19_with_varying_fill"] = int(((I.loc[idx19, "var_n_NO3"] + I.loc[idx19, "var_n_H2O"]) > 0).sum())
    out["median_r2_nligs_on_rr2_varying"] = float(I.loc[idx43, "r2_n_ligs_on_rr2"].median())
    out["n_varying_with_r2_nligs_gt_0.99"] = int((I.loc[idx43, "r2_n_ligs_on_rr2"] > 0.99).sum())

    # de-attenuated SPECIES: undo the sequential (1 - rho^2) shrink per series
    att = np.where(I["var_n_ligs"].to_numpy() == 1,
                   1 - np.nan_to_num(I["corr_n_ligs_r"].to_numpy(float)) ** 2, 1.0)
    att = np.clip(att, 0.02, 1.0)
    deatt = W.loc[idx62, "SPECIES"].to_numpy(float) / att
    r, p, n = rho(deatt, W.loc[idx62, "a"].to_numpy(float))
    recs.append({"check": "deattenuated_SPECIES", "subset": "S8_all62", "model": "SPECIES/(1-rho^2)",
                 "target": "a", "n": n, "rho": r, "p": p, "fisher_lo": np.nan, "fisher_hi": np.nan})
    out["attenuation_factor_median_varying"] = float(np.median(att[[i in set(idx43) for i in idx62]]))
    out["attenuation_factor_min_varying"] = float(np.min(att[[i in set(idx43) for i in idx62]]))

    # ---------- lens (c): S8 vs S14 vs S3, Spearman vs Pearson, 39 vs 62 ----------
    S = pd.read_csv(OUT / "A_stats.csv")
    out["c_spearman_vs_pearson"] = S[(S.target == "a")][["model", "set", "n", "rho", "pearson"]].to_dict("records")

    # ---------- lens (d): is the composition-change position itself predictive? ----------
    comp = []
    for smi in idx62:
        ser = so[smi]
        s = rows[rows["series"] == ser].sort_values("r")
        rv, cn = s["r"].to_numpy(float), s["coreCN"].to_numpy(float)
        nl = s["n_ligs"].to_numpy(float)
        rec = {"extractant": smi}
        rec["dCN_dr"] = slope_fit(rv, cn)[0]
        rec["CN_range"] = float(cn.max() - cn.min())
        rec["n_comp_changes"] = int(len(np.unique(np.c_[nl, s["n_NO3"], s["n_H2O"]], axis=0)) - 1)
        d = np.diff(cn)
        rec["r_of_first_CN_drop"] = float(rv[1:][d < 0][0]) if (d < 0).any() else np.nan
        rec["mean_CN"] = float(cn.mean())
        comp.append(rec)
    Cp = pd.DataFrame(comp).set_index("extractant").join(t)
    Cp.to_csv(OUT / "A_composition_descriptors.csv")
    for c in ("dCN_dr", "CN_range", "n_comp_changes", "r_of_first_CN_drop", "mean_CN"):
        for tg in TGT:
            r, p, n = rho(Cp[c].to_numpy(float), Cp[tg].to_numpy(float))
            recs.append({"check": "composition_descriptor", "subset": "S8_all62", "model": c,
                         "target": tg, "n": n, "rho": r, "p": p, "fisher_lo": np.nan, "fisher_hi": np.nan})
    # does the naive slope survive partialling out the composition-change descriptors?
    for c in ("dCN_dr", "CN_range", "mean_CN"):
        pr = partial_rho(W.loc[idx62, "NAIVE"].to_numpy(float), W.loc[idx62, "a"].to_numpy(float),
                         Cp[c].to_numpy(float))
        recs.append({"check": "naive_partial_given_composition", "subset": "S8_all62", "model": c,
                     "target": "a", "n": 62, "rho": pr, "p": np.nan, "fisher_lo": np.nan, "fisher_hi": np.nan})

    # ---------- claim's own internal statement: slope vs curvature rho on the 39 ----------
    d39 = df[(df.model == "NAIVE") & (df.n_computed_metals == 14)]
    r, p, n = rho(d39["slope"].to_numpy(float), d39["quad"].to_numpy(float))
    out["NAIVE_S14_slope_vs_quad_spearman"] = [r, p, n]
    d62 = df[(df.model == "SPECIES") & (df.n_computed_metals >= 8)]
    out["SPECIES_S8_slope_vs_quad_spearman"] = list(rho(d62["slope"].to_numpy(float), d62["quad"].to_numpy(float)))

    # ---------- effective units ----------
    ch = W.loc[idx62, "chemotype"].value_counts().to_numpy(float)
    out["S8_n_chemotypes"] = int(len(ch))
    out["S8_kish_neff_chemotypes"] = float(ch.sum() ** 2 / (ch ** 2).sum())
    ch19 = W.loc[idx19, "chemotype"].value_counts().to_numpy(float)
    out["S8const19_n_chemotypes"] = int(len(ch19))
    out["S8const19_kish_neff"] = float(ch19.sum() ** 2 / (ch19 ** 2).sum())
    out["S8_sc009_count"] = int((W.loc[idx62, "chemotype"] == "sc009").sum())
    out["S8const19_sc009_count"] = int((W.loc[idx19, "chemotype"] == "sc009").sum())

    R = pd.DataFrame(recs)
    R.to_csv(OUT / "A_checks.csv", index=False)
    json.dump(out, open(OUT / "A_summary.json", "w"), indent=2, default=str)
    pd.set_option("display.width", 220)
    print(R[R.check.isin(["matched_subset"]) & (R.target == "a")].to_string(index=False))
    print()
    print(R[R.check == "no_sc009"][R.target == "a"].to_string(index=False))
    print()
    print(R[~R.check.isin(["matched_subset", "no_sc009"])].to_string(index=False))
    print()
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
