"""D4 - amplitude vs shape error, and shrinkage of the predicted lanthanide contrast.

Reads the saved held-out pair predictions of the B_primary design, reconstructs the
centred lanthanide curve of every (split_seed, cell) by least squares from all pairwise
differences, splits the pairwise squared error into an amplitude part (along the
standardised Shannon-radius direction), a shape part (orthogonal to it) and a
non-integrable part (pair predictions not consistent with any curve), then

  * regresses observed amplitude on predicted amplitude (origin / intercept, pooled and
    per Tanimoto band),
  * sweeps a single global gain g on the predicted pairs and reports the ORACLE best g,
  * splits the remaining absolute error into a magnitude part and a sign part per |y| band.

Run from the repo root:
    .venv/Scripts/python.exe generations/gen13_separation/analysis/stage2/d4_amplitude_shape/d4_amplitude_shape.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
GEN13 = ROOT / "generations" / "gen13_separation"
sys.path.insert(0, str(GEN13))
from gen13sep.metals import LANTHANIDES, physics_basis  # noqa: E402

OUT = GEN13 / "analysis" / "stage2" / "d4_amplitude_shape"
FIG = GEN13 / "figures" / "stage2"
PRED = GEN13 / "predictions" / "B_primary"
SLUG = "d4_amplitude_shape"

ARMS = [
    "C_DIRECT_ROW",
    "M_SELECTED",
    "M_PHYSICS_radius+radius_sq",
    "M_LOWRANK_K2",
    "X_ENS_DIRECT+LOWRANK_K2",
]

IDX = {m: i for i, m in enumerate(LANTHANIDES)}
RADIUS_STD = physics_basis()["radius"]  # standardised, centred over all 14 Ln
GAINS = np.round(np.arange(0.80, 2.0001, 0.05), 2)


# --------------------------------------------------------------------------- helpers
def load_arm(arm: str) -> pd.DataFrame:
    df = pd.read_parquet(PRED / f"{arm}.parquet").reset_index(drop=True)
    df["iA"] = df["A"].map(IDX).astype(int)
    df["iB"] = df["B"].map(IDX).astype(int)
    df["gap"] = (df["iA"] - df["iB"]).abs()  # series-index gap (Pm-free axis)
    return df


def pair_operator(metal_idx: tuple[int, ...], pairs: np.ndarray) -> np.ndarray:
    """Incidence matrix P (n_pairs x n_metals) with +1 on A and -1 on B."""
    pos = {m: k for k, m in enumerate(metal_idx)}
    P = np.zeros((len(pairs), len(metal_idx)))
    for r, (a, b) in enumerate(pairs):
        P[r, pos[a]] = 1.0
        P[r, pos[b]] = -1.0
    return P


def unit_radius_direction(metal_idx: tuple[int, ...]) -> np.ndarray:
    """Centred, unit-norm Shannon-radius direction restricted to the cell's metals."""
    u = RADIUS_STD[list(metal_idx)].astype(float)
    u = u - u.mean()
    return u / np.linalg.norm(u)


def reconstruct(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Per (split_seed, cell_id): curves + amplitude/shape/non-integrable error split.

    Returns (cell_table, pair_table_with_components, max_reconstruction_residual_on_y).
    """
    rows = []
    max_resid_y = 0.0
    pair_amp = np.empty(len(df))
    pair_shp = np.empty(len(df))
    pair_inc = np.empty(len(df))
    pair_pu = np.empty(len(df))       # (P u)_pair = u_A - u_B, the unit amplitude direction
    pair_ampred = np.empty(len(df))   # cell-level predicted amplitude, broadcast to its pairs
    pair_ampobs = np.empty(len(df))
    cache: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for (seed, cell), grp in df.groupby(["split_seed", "cell_id"], sort=False):
        pairs = grp[["iA", "iB"]].to_numpy()
        metal_idx = tuple(sorted(set(pairs.ravel().tolist())))
        sig = (metal_idx, tuple(map(tuple, pairs)))
        if sig in cache:
            P, Pinv, u = cache[sig]
        else:
            P = pair_operator(metal_idx, pairs)
            Pinv = np.linalg.pinv(P)
            u = unit_radius_direction(metal_idx)
            if len(cache) < 4000:
                cache[sig] = (P, Pinv, u)
        y = grp["y"].to_numpy()
        p = grp["prediction"].to_numpy()

        c_obs = Pinv @ y
        c_obs -= c_obs.mean()
        c_prd = Pinv @ p
        c_prd -= c_prd.mean()

        resid_y = float(np.max(np.abs(P @ c_obs - y))) if len(y) else 0.0
        max_resid_y = max(max_resid_y, resid_y)
        r_inc = p - P @ c_prd  # part of the pair prediction that is not any curve

        a_obs = float(c_obs @ u)
        a_prd = float(c_prd @ u)
        s_obs = c_obs - a_obs * u
        s_prd = c_prd - a_prd * u

        e_amp_pair = P @ ((a_prd - a_obs) * u)
        e_shp_pair = P @ (s_prd - s_obs)
        e_pair = p - y

        pos = grp.index.to_numpy()  # df was reset to a 0..n-1 RangeIndex, so index == position
        pair_amp[pos] = e_amp_pair
        pair_shp[pos] = e_shp_pair
        pair_inc[pos] = r_inc
        pair_pu[pos] = P @ u
        pair_ampred[pos] = a_prd
        pair_ampobs[pos] = a_obs

        rows.append(
            dict(
                split_seed=seed,
                cell_id=cell,
                extractant=grp["extractant"].iloc[0],
                chemotype=grp["chemotype"].iloc[0],
                n_metals=len(metal_idx),
                n_pairs=len(pairs),
                amp_obs=a_obs,
                amp_pred=a_prd,
                shape_norm_obs=float(np.linalg.norm(s_obs)),
                shape_norm_pred=float(np.linalg.norm(s_prd)),
                curve_norm_obs=float(np.linalg.norm(c_obs)),
                sse_total=float(np.sum(e_pair**2)),
                sse_amp=float(np.sum(e_amp_pair**2)),
                sse_shape=float(np.sum(e_shp_pair**2)),
                sse_incons=float(np.sum(r_inc**2)),
                cross=float(np.sum(e_pair**2) - np.sum(e_amp_pair**2)
                            - np.sum(e_shp_pair**2) - np.sum(r_inc**2)),
                recon_resid_y=resid_y,
            )
        )
    cells = pd.DataFrame(rows)
    out = df.copy()
    out["e_amp"] = pair_amp
    out["e_shape"] = pair_shp
    out["e_incons"] = pair_inc
    out["pu"] = pair_pu
    out["amp_pred"] = pair_ampred
    out["amp_obs"] = pair_ampobs
    return cells, out, max_resid_y


def macro_mae(df: pd.DataFrame, err: np.ndarray) -> tuple[float, float]:
    """Extractant-macro MAE (mean over extractants inside a seed, then over seeds)."""
    t = pd.DataFrame({"seed": df["split_seed"].to_numpy(),
                      "ext": df["extractant"].to_numpy(),
                      "ae": np.abs(err)})
    per_ext = t.groupby(["seed", "ext"], observed=True)["ae"].mean()
    per_seed = per_ext.groupby(level=0).mean()
    return float(per_seed.mean()), float(t["ae"].mean())


def boot_slope_origin(x: np.ndarray, y: np.ndarray, groups: np.ndarray,
                      n_boot: int = 1000, seed: int = 7) -> tuple[float, float]:
    """Percentile CI for the through-origin slope, resampling extractants (clusters)."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    idx_by_g = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        sel = np.concatenate([idx_by_g[k] for k in pick])
        xb, yb = x[sel], y[sel]
        den = float(xb @ xb)
        out[b] = (xb @ yb) / den if den > 0 else np.nan
    return float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))


# --------------------------------------------------------------------------- main
def main() -> None:
    sim = pd.read_parquet(PRED / "similarity.parquet")[["split_seed", "cell_id", "band",
                                                        "max_train_tanimoto"]]

    decomp_rows, reg_rows, sweep_rows, best_rows, sign_rows, check_rows = [], [], [], [], [], []
    oracle_rows: list[dict] = []
    amp_store = {}
    sweep_store = {}

    for arm in ARMS:
        df = load_arm(arm)
        cells, pairs, max_resid = reconstruct(df)
        cells = cells.merge(sim, on=["split_seed", "cell_id"], how="left")
        pairs = pairs.merge(sim, on=["split_seed", "cell_id"], how="left")
        amp_store[arm] = cells
        print(f"[{arm}] cells={len(cells)} pairs={len(pairs)} "
              f"max |P c_obs - y| = {max_resid:.3e}")
        check_rows.append(dict(arm=arm, n_cells=len(cells), n_pairs=len(pairs),
                               max_abs_recon_resid_y=max_resid,
                               max_abs_cross_term=float(cells["cross"].abs().max()),
                               verified_1e_9=bool(max_resid < 1e-9)))
        if max_resid >= 1e-9:
            raise SystemExit(f"reconstruction of observed curves failed for {arm}")

        # ---- 2. error decomposition ---------------------------------------------
        for subset, mask in [("all", np.ones(len(cells), bool)),
                             ("n_metals>=4", (cells["n_metals"] >= 4).to_numpy()),
                             ("n_metals==2", (cells["n_metals"] == 2).to_numpy())]:
            sub = cells[mask]
            tot = sub["sse_total"].sum()
            if tot <= 0:
                continue
            decomp_rows.append(dict(
                arm=arm, subset=subset, n_cells=int(len(sub)),
                n_pairs=int(sub["n_pairs"].sum()),
                sse_total=tot, frac_amp=sub["sse_amp"].sum() / tot,
                frac_shape=sub["sse_shape"].sum() / tot,
                frac_incons=sub["sse_incons"].sum() / tot,
                frac_cross=sub["cross"].sum() / tot,
                rmse_pairs=float(np.sqrt(tot / sub["n_pairs"].sum())),
                rmse_if_amp_perfect=float(np.sqrt(
                    (sub["sse_shape"].sum() + sub["sse_incons"].sum()) / sub["n_pairs"].sum())),
            ))
        for band in ["near", "mid", "far"]:
            sub = cells[cells["band"] == band]
            tot = sub["sse_total"].sum()
            if tot <= 0:
                continue
            decomp_rows.append(dict(
                arm=arm, subset=f"band={band}", n_cells=int(len(sub)),
                n_pairs=int(sub["n_pairs"].sum()), sse_total=tot,
                frac_amp=sub["sse_amp"].sum() / tot, frac_shape=sub["sse_shape"].sum() / tot,
                frac_incons=sub["sse_incons"].sum() / tot, frac_cross=sub["cross"].sum() / tot,
                rmse_pairs=float(np.sqrt(tot / sub["n_pairs"].sum())),
                rmse_if_amp_perfect=float(np.sqrt(
                    (sub["sse_shape"].sum() + sub["sse_incons"].sum()) / sub["n_pairs"].sum())),
            ))

        # ---- 3a. amplitude regression -------------------------------------------
        for band in ["pooled", "near", "mid", "far"]:
            sub = cells if band == "pooled" else cells[cells["band"] == band]
            if len(sub) < 5:
                continue
            x = sub["amp_pred"].to_numpy()
            y = sub["amp_obs"].to_numpy()
            slope0 = float((x @ y) / (x @ x))
            X = np.column_stack([np.ones(len(x)), x])
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ coef
            r2 = 1 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())
            lo, hi = boot_slope_origin(x, y, sub["extractant"].to_numpy())
            reg_rows.append(dict(
                arm=arm, band=band, n_cells=int(len(sub)),
                n_extractants=int(sub["extractant"].nunique()),
                slope_origin=slope0, slope_origin_lo95=lo, slope_origin_hi95=hi,
                slope_with_intercept=float(coef[1]), intercept=float(coef[0]),
                r2_with_intercept=r2,
                sd_amp_obs=float(y.std()), sd_amp_pred=float(x.std()),
                sd_ratio_obs_over_pred=float(y.std() / x.std()),
                mean_amp_obs=float(y.mean()), mean_amp_pred=float(x.mean()),
            ))

        # ---- 3b. global gain sweep ----------------------------------------------
        y_all = pairs["y"].to_numpy()
        p_all = pairs["prediction"].to_numpy()
        subsets = {
            "all": np.ones(len(pairs), bool),
            "adjacent(gap=1)": (pairs["gap"] == 1).to_numpy(),
            "far(gap>=7)": (pairs["gap"] >= 7).to_numpy(),
        }
        for sname, m in subsets.items():
            dsub = pairs[m]
            ys, ps = y_all[m], p_all[m]
            curve = []
            for g in GAINS:
                mac, pool = macro_mae(dsub, g * ps - ys)
                curve.append((g, mac, pool))
                sweep_rows.append(dict(arm=arm, subset=sname, gain=float(g),
                                       macro_mae=mac, pooled_mae=pool,
                                       n_pairs=int(m.sum())))
            curve = np.array(curve)
            base = float(curve[np.isclose(curve[:, 0], 1.0), 1][0])
            k = int(np.argmin(curve[:, 1]))
            sweep_store[(arm, sname)] = curve
            best_rows.append(dict(
                arm=arm, subset=sname, n_pairs=int(m.sum()),
                macro_mae_g1=base, best_gain=float(curve[k, 0]),
                macro_mae_best=float(curve[k, 1]),
                macro_mae_gain=base - float(curve[k, 1]),
                pooled_mae_g1=float(curve[np.isclose(curve[:, 0], 1.0), 2][0]),
                pooled_mae_best=float(curve[k, 2]),
                oracle=True,
            ))
            print(f"  [{arm}] {sname:16s} g=1 macro MAE {base:.4f} -> "
                  f"best g={curve[k,0]:.2f} macro MAE {curve[k,1]:.4f} "
                  f"(gain {base-curve[k,1]:+.4f})")

        # ---- 3c. oracles and alternative single-scalar corrections ---------------
        pu = pairs["pu"].to_numpy()
        ap = pairs["amp_pred"].to_numpy()
        e0 = p_all - y_all
        e_amp = pairs["e_amp"].to_numpy()
        e_shp = pairs["e_shape"].to_numpy()
        for sname, m in subsets.items():
            dsub = pairs[m]

            def mm(err_full: np.ndarray) -> float:
                return macro_mae(dsub, err_full[m])[0]

            base = mm(e0)
            # amplitude-only gain (scale only the radius component of the predicted curve)
            ag = [(g, mm(e0 + (g - 1.0) * ap * pu)) for g in GAINS]
            ag_best = min(ag, key=lambda t: t[1])
            # additive amplitude offset: predicted curve + delta * u
            deltas = np.round(np.arange(-0.60, 0.6001, 0.05), 2)
            dl = [(d, mm(e0 + d * pu)) for d in deltas]
            dl_best = min(dl, key=lambda t: t[1])
            # per-extractant and per-cell oracle gains on the whole curve
            def best_grouped(keys: list[str]) -> float:
                tab = pd.DataFrame({"seed": dsub["split_seed"].to_numpy(),
                                    "ext": dsub["extractant"].to_numpy(),
                                    "cell": dsub["cell_id"].to_numpy()})
                errs = np.stack([np.abs(g * p_all[m] - y_all[m]) for g in GAINS])
                grp_keys = tab[keys].astype(str).agg("|".join, axis=1).to_numpy()
                uq, inv = np.unique(grp_keys, return_inverse=True)
                sums = np.zeros((len(GAINS), len(uq)))
                cnts = np.bincount(inv, minlength=len(uq))
                for i in range(len(GAINS)):
                    sums[i] = np.bincount(inv, weights=errs[i], minlength=len(uq))
                means = sums / cnts
                pick = means.argmin(axis=0)
                chosen = errs[pick[inv], np.arange(len(inv))]
                return macro_mae(dsub, chosen * np.sign(1.0))[0]

            oracle_rows_local = dict(
                arm=arm, subset=sname, n_pairs=int(m.sum()),
                macro_mae_baseline=base,
                macro_mae_amplitude_oracle=mm(e_shp),          # amplitude made exact
                macro_mae_shape_oracle=mm(e_amp),              # shape made exact
                macro_mae_best_global_gain=min(
                    (mm(g * p_all - y_all) for g in GAINS)),
                macro_mae_best_amp_only_gain=ag_best[1], best_amp_only_gain=float(ag_best[0]),
                macro_mae_best_amp_offset=dl_best[1], best_amp_offset=float(dl_best[0]),
                macro_mae_best_per_extractant_gain=best_grouped(["seed", "ext"]),
                macro_mae_best_per_cell_gain=best_grouped(["seed", "cell"]),
            )
            oracle_rows.append(oracle_rows_local)
            if sname == "all":
                print(f"  [{arm}] oracles: base {base:.4f} | amp-oracle "
                      f"{oracle_rows_local['macro_mae_amplitude_oracle']:.4f} | shape-oracle "
                      f"{oracle_rows_local['macro_mae_shape_oracle']:.4f} | per-cell gain "
                      f"{oracle_rows_local['macro_mae_best_per_cell_gain']:.4f} | amp-offset "
                      f"{dl_best[1]:.4f} @ d={dl_best[0]:+.2f}")

        # ---- 4. sign vs magnitude split -----------------------------------------
        best_g_all = float(sweep_store[(arm, "all")][
            int(np.argmin(sweep_store[(arm, "all")][:, 1])), 0])
        ay = np.abs(y_all)
        band_edges = [0.0, 0.25, 0.5, 1.0, np.inf]
        band_lab = ["|y|<0.25", "0.25-0.5", "0.5-1.0", ">1.0"]
        ybin = pd.cut(ay, band_edges, labels=band_lab, right=False)
        for g in [1.0, best_g_all]:
            pg = g * p_all
            e = np.abs(pg - y_all)
            mag = np.abs(np.abs(pg) - ay)
            sgn = e - mag  # == 2*min(|pred|,|y|) when signs differ, else 0
            opp = (np.sign(pg) * np.sign(y_all)) < 0
            for lab in band_lab + ["all"]:
                m = np.ones(len(e), bool) if lab == "all" else np.asarray(ybin == lab)
                if m.sum() == 0:
                    continue
                sign_rows.append(dict(
                    arm=arm, gain=round(float(g), 2), y_band=lab, n_pairs=int(m.sum()),
                    frac_pairs_sign_opposite=float(opp[m].mean()),
                    mean_abs_err=float(e[m].mean()),
                    mean_magnitude_part=float(mag[m].mean()),
                    mean_sign_part=float(sgn[m].mean()),
                    frac_abs_err_from_sign=float(sgn[m].sum() / e[m].sum()),
                    frac_abs_err_in_sign_opposite_pairs=float(
                        e[m & opp].sum() / e[m].sum()),
                    share_of_total_abs_err=float(e[m].sum() / e.sum()),
                ))
        del df, pairs

    pd.DataFrame(check_rows).to_csv(OUT / f"{SLUG}_reconstruction_check.csv", index=False)
    dec = pd.DataFrame(decomp_rows)
    dec.to_csv(OUT / f"{SLUG}_error_decomposition.csv", index=False)
    reg = pd.DataFrame(reg_rows)
    reg.to_csv(OUT / f"{SLUG}_amplitude_regression.csv", index=False)
    pd.DataFrame(sweep_rows).to_csv(OUT / f"{SLUG}_gain_sweep.csv", index=False)
    best = pd.DataFrame(best_rows)
    best.to_csv(OUT / f"{SLUG}_gain_best.csv", index=False)
    pd.DataFrame(sign_rows).to_csv(OUT / f"{SLUG}_sign_split.csv", index=False)
    orc = pd.DataFrame(oracle_rows)
    orc.to_csv(OUT / f"{SLUG}_oracles.csv", index=False)
    pd.concat([v.assign(arm=k) for k, v in amp_store.items()]).to_csv(
        OUT / f"{SLUG}_cell_amplitudes.csv", index=False)

    print("\n== decomposition (all cells) ==")
    print(dec[dec.subset == "all"][["arm", "frac_amp", "frac_shape", "frac_incons",
                                    "rmse_pairs", "rmse_if_amp_perfect"]].to_string(index=False))
    print("\n== oracles (all pairs, extractant-macro MAE) ==")
    print(orc[orc.subset == "all"].to_string(index=False))
    print("\n== amplitude regression (pooled) ==")
    print(reg[reg.band == "pooled"][["arm", "slope_origin", "slope_origin_lo95",
                                     "slope_origin_hi95", "slope_with_intercept",
                                     "intercept", "r2_with_intercept"]].to_string(index=False))

    # ---- figures -------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for ax, sname in zip(axes, ["all", "adjacent(gap=1)", "far(gap>=7)"]):
        for arm in ARMS:
            c = sweep_store[(arm, sname)]
            ax.plot(c[:, 0], c[:, 1], marker="o", ms=2.5, lw=1.2, label=arm)
        ax.axvline(1.0, color="0.6", lw=0.8, ls="--")
        ax.set_title(f"gain sweep - {sname}")
        ax.set_xlabel("global gain g")
        ax.set_ylabel("extractant-macro MAE")
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / f"{SLUG}_gain_sweep.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(ARMS), figsize=(3.0 * len(ARMS), 3.2))
    for ax, arm in zip(axes, ARMS):
        c = amp_store[arm]
        ax.scatter(c["amp_pred"], c["amp_obs"], s=4, alpha=0.3, color="#3b6ea5")
        lim = [min(c["amp_pred"].min(), c["amp_obs"].min()),
               max(c["amp_pred"].max(), c["amp_obs"].max())]
        ax.plot(lim, lim, color="0.4", lw=0.8, ls="--")
        s = float((c["amp_pred"] @ c["amp_obs"]) / (c["amp_pred"] @ c["amp_pred"]))
        ax.plot(lim, [s * lim[0], s * lim[1]], color="#c1533c", lw=1.2)
        ax.set_title(f"{arm}\nslope0={s:.2f}", fontsize=7)
        ax.set_xlabel("predicted amplitude")
        ax.set_ylabel("observed amplitude")
    fig.tight_layout()
    fig.savefig(FIG / f"{SLUG}_amplitude_scatter.png", dpi=150)
    plt.close(fig)
    print("\nwrote tables to", OUT)


if __name__ == "__main__":
    main()
