"""REFUTER lens A (round 2), part 2 for L1NULL.

(a) power / CI at n = 19; (b) attenuation of an INJECTED real radius trend, per model;
(5) permutation null with an independent seed; (7) decision rule and BH recomputed from the
lead's own CSVs; matched competitor on the same 62; robustness of the composition-position
descriptor found in part 1.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from refute_l1null_a4 import (ROOT, OUT, PHYS3D, LEAD, load, fit_all, slopes, choose_series,  # noqa: E402
                              rho, prho, varying, two_way, species_cov, centred_counts)

TGT3 = ("a", "abs_a", "b")


def fisher_ci(r, n, alpha=0.05, spearman=True):
    if not np.isfinite(r) or n < 6:
        return np.nan, np.nan
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = (1.06 if spearman else 1.0) / np.sqrt(n - 3)
    zc = stats.norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - zc * se)), float(np.tanh(z + zc * se))


def power(rho_true, n, alpha=0.05):
    if n < 6:
        return np.nan
    se = 1.06 / np.sqrt(n - 3)
    z = np.arctanh(rho_true) / se
    zc = stats.norm.ppf(1 - alpha / 2)
    return float(stats.norm.sf(zc - z) + stats.norm.cdf(-zc - z))


def z_test(r_obs, r_h0, n):
    se = 1.06 / np.sqrt(n - 3)
    z = (np.arctanh(r_obs) - np.arctanh(r_h0)) / se
    return float(z), float(2 * stats.norm.sf(abs(z)))


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
    rows_all, count_cols, forms = load()
    fin = np.isfinite(rows_all["complex_total_energy_eV"].to_numpy(float))
    rows = rows_all[fin].reset_index(drop=True)
    y0 = rows["complex_total_energy_eV"].to_numpy(float)
    tgt = pd.read_parquet(PHYS3D / "extractant_targets.parquet")[
        ["extractant", "a", "b", "abs_a", "n_metals", "n_cells", "chemotype"]]
    S = {}

    # ---------------- (a) power and interval width at n = 19 ----------------------
    pw = []
    for n in (19, 39, 62):
        for rt in (0.2, 0.3, 0.4, 0.5, 0.6, 0.644):
            pw.append({"n": n, "rho_true": rt, "power_alpha0.05": power(rt, n)})
    pw = pd.DataFrame(pw)
    pw.to_csv(OUT / "A5_power.csv", index=False)
    lo, hi = fisher_ci(0.166667, 19)
    S["fisher_ci_SPECIES_CONST_S8_n19"] = [lo, hi]
    S["fisher_ci_width_n19"] = hi - lo
    for h0 in (0.4, 0.5, 0.6, 0.644):
        S[f"test_0.167_vs_{h0}_n19"] = z_test(0.166667, h0, 19)
    S["power_n19_rho0.5"] = power(0.5, 19)
    S["power_n19_rho0.644"] = power(0.644, 19)
    S["power_n19_rho0.4"] = power(0.4, 19)

    # ---------------- (b) attenuation of an injected real trend -------------------
    a_by_ext = tgt.set_index("extractant")["a"].to_dict()
    sof = choose_series(rows_all)
    ser_of_ext = {e: s for e, s in sof.items()}
    ext_of_row = {}
    for e, s in ser_of_ext.items():
        ext_of_row[s] = e
    row_ext = rows["series"].map(ext_of_row)
    row_a = row_ext.map(a_by_ext).to_numpy(float)
    rvec = rows["r"].to_numpy(float)
    inj_unit = np.where(np.isfinite(row_a), np.nan_to_num(row_a) * rvec, 0.0)

    rec, att = [], []
    for k in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0):
        y = y0 + k * inj_unit
        resid, _ = fit_all(rows, count_cols, forms, y=y)
        sl = slopes(rows, resid, rows_all).merge(tgt, on="extractant", how="inner")
        nn = sl["n_computed_metals"].to_numpy()
        sl["S8"], sl["S14"] = nn >= 8, nn == 14
        for m in sl["model"].unique():
            for st in ("S8", "S14"):
                d = sl[(sl["model"] == m) & sl[st]]
                r, p, n = rho(d["slope"].to_numpy(float), d["a"].to_numpy(float))
                rec.append({"k_eV_per_unit_r_per_a": k, "model": m, "set": st, "n": n,
                            "rho": r, "p": p})
        # attenuation: OLS of recovered slope on the injected slope (= k * a), on S8
        if k > 0:
            for m in sl["model"].unique():
                d = sl[(sl["model"] == m) & sl["S8"]].dropna(subset=["slope"])
                d0 = pd.read_csv(OUT / "A4_slopes.csv")
                d0 = d0[(d0["model"] == m) & (d0["n_computed_metals"] >= 8)]
                base = d0.set_index("extractant")["slope"]
                dd = d.set_index("extractant")
                common = base.index.intersection(dd.index)
                dy = (dd.loc[common, "slope"] - base.loc[common]).to_numpy(float)
                dx = (k * dd.loc[common, "a"]).to_numpy(float)
                ok = np.isfinite(dx) & np.isfinite(dy)
                coef = float(np.polyfit(dx[ok], dy[ok], 1)[0]) if ok.sum() > 3 else np.nan
                att.append({"k": k, "model": m, "n": int(ok.sum()), "retained_fraction": coef})
    pd.DataFrame(rec).to_csv(OUT / "A5_injection.csv", index=False)
    pd.DataFrame(att).to_csv(OUT / "A5_attenuation.csv", index=False)

    # ---------------- (5) permutation null, independent seed ----------------------
    sl0 = pd.read_csv(OUT / "A4_slopes.csv")
    W = tgt.set_index("extractant").copy()
    fam = []
    for m in ("NAIVE", "ELEM", "SPECIES", "SPECIES_CONST"):
        d = sl0[sl0["model"] == m].set_index("extractant")
        for st, msk in (("S8", d["n_computed_metals"] >= 8), ("S14", d["n_computed_metals"] == 14),
                        ("S3", d["n_computed_metals"] >= 3)):
            col = f"{m}__{st}"
            W[col] = d["slope"].where(msk).reindex(W.index)
            fam.append(col)
    cm = W.groupby("chemotype")[fam + list(TGT3)].mean()
    M = cm[fam].to_numpy(float)
    perm_rows = []
    for seed in (20260909, 777001, 424243):
        rng = np.random.default_rng(seed)
        for t in TGT3:
            yv = cm[t].to_numpy(float)
            obs = []
            for j in range(M.shape[1]):
                ok = np.isfinite(M[:, j]) & np.isfinite(yv)
                obs.append(stats.spearmanr(M[ok, j], yv[ok]).statistic if ok.sum() >= 10 else np.nan)
            obs = np.array(obs, float)
            null = np.empty(2000)
            for kk in range(2000):
                yp = rng.permutation(yv)
                best = 0.0
                for j in range(M.shape[1]):
                    ok = np.isfinite(M[:, j]) & np.isfinite(yp)
                    if ok.sum() < 10:
                        continue
                    r = abs(stats.spearmanr(M[ok, j], yp[ok]).statistic)
                    if np.isfinite(r) and r > best:
                        best = r
                null[kk] = best
            perm_rows.append({"seed": seed, "target": t, "n_chemotype_units": len(cm),
                              "family_size": len(fam),
                              "null_p95": float(np.percentile(null, 95)),
                              "observed_max_abs_rho": float(np.nanmax(np.abs(obs))),
                              "argmax": fam[int(np.nanargmax(np.abs(obs)))],
                              "familywise_p": float((null >= np.nanmax(np.abs(obs))).mean())})
    pd.DataFrame(perm_rows).to_csv(OUT / "A5_perm_null_seeds.csv", index=False)

    # ---------------- (7) decision + BH from the lead's own CSVs -------------------
    st = pd.read_csv(LEAD / "stage1_stats.csv")
    r = st[(st["model"] == "SPECIES") & (st["set"] == "S8") & (st["target"] == "a")
           & (st["value"] == "slope")].iloc[0]
    S["lead_decision_recomputed"] = {
        "rho": float(r["rho"]), "abs_rho_lt_0.25": bool(abs(float(r["rho"])) < 0.25),
        "ci": [float(r["ci95_low"]), float(r["ci95_high"])],
        "ci_excludes_zero": bool(float(r["ci95_low"]) * float(r["ci95_high"]) > 0),
        "verdict": "closed" if (abs(float(r["rho"])) < 0.25
                                and not float(r["ci95_low"]) * float(r["ci95_high"]) > 0) else "not closed"}
    con = pd.read_csv(LEAD / "contrasts_stage1.csv")
    S["contrasts_rows"] = int(len(con))
    S["contrasts_registered"] = int((con["family"] == "registered").sum())
    reg = con[con["family"] == "registered"]
    S["registered_raw_p"] = reg["p_two_sided"].tolist()
    S["registered_bh_p"] = reg["p_bh_within_family"].tolist()
    S["exploratory_min_bh"] = float(con[con["family"] == "exploratory"]["p_bh_within_family"].min())

    # ---------------- matched competitor on the SAME 62 ---------------------------
    try:
        sys.path.insert(0, str(ROOT / "generations" / "gen15_curve"))
        sys.path.insert(0, str(ROOT / "generations" / "gen14_direction"))
        from gen14.dirbench import load as bload  # noqa: E402
        bench = bload()
        X = getattr(bench, "X", None)
        cells = getattr(bench, "cells", None)
        col = "coord__dist__frac_donor_pairs_within_3"
        if cells is not None and col in getattr(cells, "columns", []):
            comp = cells.groupby("extractant")[col].median()
        else:
            comp = None
        if comp is not None:
            d8 = sl0[(sl0["model"] == "SPECIES") & (sl0["n_computed_metals"] >= 8)]
            mm = tgt.set_index("extractant").loc[d8["extractant"]]
            v = comp.reindex(d8["extractant"]).to_numpy(float)
            for t in TGT3:
                rr, pp, nn = rho(v, mm[t].to_numpy(float))
                S[f"competitor_frac_donor_pairs_within_3_vs_{t}_on_S8"] = [rr, pp, nn]
    except Exception as exc:  # pragma: no cover
        S["competitor_error"] = repr(exc)

    # ---------------- composition-position descriptor robustness ------------------
    comp = pd.read_csv(OUT / "A4_composition_descriptors.csv")
    comp = comp.merge(sl0[sl0["model"] == "NAIVE"][["extractant", "chemotype"]].drop_duplicates(),
                      on="extractant", how="left", suffixes=("", "_dup"))
    out = []
    for col in ("frac_at_max_nligs", "nligs_range"):
        for lbl, d in (("S8", comp[comp["S8"].fillna(False)]),
                       ("S14", comp[comp["S14"].fillna(False)]),
                       ("S8_no_sc009", comp[comp["S8"].fillna(False) & (comp["chemotype"] != "sc009")]),
                       ("S14_no_sc009", comp[comp["S14"].fillna(False) & (comp["chemotype"] != "sc009")])):
            rr, pp, nn = rho(d[col].to_numpy(float), d["a"].to_numpy(float))
            pr = prho(d[col].to_numpy(float), d["a"].to_numpy(float), d["n_metals"].to_numpy(float))
            lo2, hi2, nb = blocked_boot(d[col].to_numpy(float), d["a"].to_numpy(float),
                                        d["chemotype"].to_numpy(), min_rows=min(20, max(10, nn - 5)))
            chs = d["chemotype"].to_numpy()
            lv = []
            for c in np.unique(chs):
                m2 = chs != c
                v2 = rho(d[col].to_numpy(float)[m2], d["a"].to_numpy(float)[m2])[0]
                if np.isfinite(v2):
                    lv.append(v2)
            out.append({"descriptor": col, "set": lbl, "n": nn, "rho": rr, "p": pp,
                        "partial_rho_n_metals": pr, "ci_lo": lo2, "ci_hi": hi2, "n_boot": nb,
                        "loco_min": float(np.min(lv)), "loco_max": float(np.max(lv)),
                        "n_chemotypes": int(len(np.unique(chs)))})
    pd.DataFrame(out).to_csv(OUT / "A5_composition_robust.csv", index=False)

    # blocked CI for the no-free-delta cycles
    ci_rows = []
    for m in ("NAIVE", "SPECIES", "CYCLE_ADD", "CYCLE_CONSTGAMMA", "SPECIES_NODELTA"):
        d = sl0[(sl0["model"] == m) & (sl0["n_computed_metals"] >= 8)]
        mm = tgt.set_index("extractant").loc[d["extractant"]]
        for seed in (20260909, 777001):
            lo2, hi2, nb = blocked_boot(d["slope"].to_numpy(float), mm["a"].to_numpy(float),
                                        mm["chemotype"].to_numpy(), seed=seed)
            ci_rows.append({"model": m, "seed": seed, "n": len(d),
                            "rho": rho(d["slope"].to_numpy(float), mm["a"].to_numpy(float))[0],
                            "ci_lo": lo2, "ci_hi": hi2, "n_boot": nb})
    pd.DataFrame(ci_rows).to_csv(OUT / "A5_blocked_ci.csv", index=False)

    with open(OUT / "A5_summary.json", "w", encoding="utf-8") as fh:
        json.dump(S, fh, indent=2, default=float)
    print(json.dumps(S, indent=2, default=float)[:6000])


if __name__ == "__main__":
    main()
