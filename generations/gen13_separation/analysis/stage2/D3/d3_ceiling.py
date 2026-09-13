"""D3 - achievable-accuracy ceiling and variance decomposition for the gen13 separation task.

Run:  .venv/Scripts/python.exe generations/gen13_separation/analysis/stage2/D3/d3_ceiling.py

Builds a ladder of oracle-ish predictors of a held-out cell's centred lanthanide curve,
scores each with the programme's convention (extractant-macro MAE over unordered pairs,
plus adjacent / far splits, chemotype-macro and pooled), and decomposes the variance of
the centred curve into between-extractant / between-cell-within-extractant / replicate
noise.  Everything is written to this directory as CSV.
"""
from __future__ import annotations

import sys
from pathlib import Path

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="Mean of empty slice")

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "generations" / "gen13_separation"))
from gen13sep.metals import ATOMIC_NUMBER, LANTHANIDES, SHANNON_RADIUS_CN8  # noqa: E402

OUT = Path(__file__).resolve().parent
FIG = ROOT / "generations" / "gen13_separation" / "figures" / "stage2"
COHORT = ROOT / "generations" / "gen13_separation" / "manifests" / "cohort_exact.parquet"
PRED = ROOT / "generations" / "gen13_separation" / "predictions" / "B_primary"

N_LN = len(LANTHANIDES)
IDX = {m: i for i, m in enumerate(LANTHANIDES)}
FAR_MIN_DZ = 5
RVEC = np.array([SHANNON_RADIUS_CN8[m] for m in LANTHANIDES])
RSTD = (RVEC - RVEC.mean()) / RVEC.std()


# ----------------------------------------------------------------------------- data
def load():
    df = pd.read_parquet(COHORT).reset_index(drop=True)
    Y = df[[f"logD__{m}" for m in LANTHANIDES]].to_numpy(dtype=float)
    nrep = df[[f"nrep__{m}" for m in LANTHANIDES]].to_numpy(dtype=float)
    repsd = df[[f"repsd__{m}" for m in LANTHANIDES]].to_numpy(dtype=float)
    return df, Y, nrep, repsd


def pair_table(df: pd.DataFrame, Y: np.ndarray) -> pd.DataFrame:
    """All unordered pairs (A lighter than B) of each cell's observed metals."""
    rows = []
    z = np.array([ATOMIC_NUMBER[m] for m in LANTHANIDES])
    for i in range(len(df)):
        obs = np.flatnonzero(~np.isnan(Y[i]))
        for ai in obs:
            for bi in obs:
                if ai < bi:
                    rows.append((i, df.cell_id.iat[i], df.extractant.iat[i], df.chemotype.iat[i],
                                 int(df.n_metals.iat[i]), LANTHANIDES[ai], LANTHANIDES[bi],
                                 int(ai), int(bi), int(z[bi] - z[ai]), float(Y[i, ai] - Y[i, bi])))
    p = pd.DataFrame(rows, columns=["row", "cell_id", "extractant", "chemotype", "n_metals",
                                    "A", "B", "ia", "ib", "dZ", "y"])
    p["adjacent"] = (p.dZ == 1) | ((p.A == "Nd") & (p.B == "Sm"))
    p["far"] = p.dZ >= FAR_MIN_DZ
    return p


# ------------------------------------------------------------------------- scoring
def score(pairs: pd.DataFrame, pred: np.ndarray, label: str) -> dict:
    """Programme convention: MAE pooled inside an extractant, then macro over extractants."""
    err = np.abs(pairs["y"].to_numpy() - pred)
    ok = np.isfinite(err)
    d = pd.DataFrame({"extractant": pairs["extractant"].to_numpy(), "chemotype": pairs["chemotype"].to_numpy(),
                      "cell_id": pairs["cell_id"].to_numpy(), "adjacent": pairs["adjacent"].to_numpy(),
                      "far": pairs["far"].to_numpy(), "err": err})[ok]
    if d.empty:
        return {"entry": label, "macro_mae_extractant": np.nan}
    g = d.groupby("extractant")
    per = pd.DataFrame({"chemotype": g["chemotype"].first(),
                        "mae_all": g["err"].mean(),
                        "n_pairs": g.size(),
                        "n_cells": g["cell_id"].nunique()})
    per["mae_adjacent"] = d[d.adjacent].groupby("extractant")["err"].mean()
    per["mae_far"] = d[d.far].groupby("extractant")["err"].mean()
    chem = per.groupby("chemotype")["mae_all"].mean()
    return {"entry": label,
            "macro_mae_extractant": float(per["mae_all"].mean()),
            "macro_mae_adjacent": float(per["mae_adjacent"].mean(skipna=True)),
            "macro_mae_far": float(per["mae_far"].mean(skipna=True)),
            "macro_mae_chemotype": float(chem.mean()),
            "pooled_mae": float(d["err"].mean()),
            "pooled_mae_adjacent": float(d.loc[d.adjacent, "err"].mean()),
            "pooled_mae_far": float(d.loc[d.far, "err"].mean()),
            "n_extractants": int(per.shape[0]),
            "n_chemotypes": int(chem.shape[0]),
            "n_cells": int(d["cell_id"].nunique()),
            "n_pairs": int(len(d)),
            "n_pairs_adjacent": int(d.adjacent.sum()),
            "n_pairs_far": int(d.far.sum())}, per


def curve_to_pairs(P: np.ndarray, pairs: pd.DataFrame) -> np.ndarray:
    r, ia, ib = pairs["row"].to_numpy(), pairs["ia"].to_numpy(), pairs["ib"].to_numpy()
    return P[r, ia] - P[r, ib]


# ------------------------------------------------------- aligned group mean curves
def aligned_curve(Y: np.ndarray, rows: np.ndarray, w: np.ndarray | None = None,
                  iters: int = 150) -> np.ndarray:
    """Best single 14-curve g for a group of cells, allowing a free per-cell level:
    minimise sum_i w_i sum_{m observed} (Y[i,m] - o_i - g[m])^2.  Identified by mean(g)=0.
    Metals nobody in the group observed come back NaN."""
    if len(rows) == 0:
        return np.full(N_LN, np.nan)
    A = Y[rows]
    ww = np.ones(len(rows)) if w is None else w[rows]
    mask = ~np.isnan(A)
    sup = mask.any(axis=0)
    g = np.zeros(N_LN)
    with np.errstate(invalid="ignore"):
        g[sup] = np.nanmean(A - np.nanmean(A, axis=1, keepdims=True), axis=0)[sup]
    g[~sup] = 0.0
    for _ in range(iters):
        o = np.nanmean(A - g[None, :], axis=1)
        R = np.where(mask, A - o[:, None], 0.0)
        num = (R * ww[:, None]).sum(axis=0)
        den = (mask * ww[:, None]).sum(axis=0)
        new = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
        new = new - new[sup].mean()
        if np.max(np.abs(new - g)) < 1e-10:
            g = new
            break
        g = new
    return np.where(sup, g, np.nan)


def rank1_shape(C: np.ndarray, rows: np.ndarray, iters: int = 100) -> np.ndarray:
    """Leading rank-1 direction of the centred curves of ``rows`` by masked ALS (vectorised).
    Returned with unit norm and a positive projection on the Shannon-radius curve."""
    A = C[rows]
    M = (~np.isnan(A)).astype(float)
    X = np.where(M > 0, A, 0.0)
    v = np.nanmean(A, axis=0)
    v = np.where(np.isfinite(v), v, 0.0)
    if np.linalg.norm(v) < 1e-12:
        return np.full(N_LN, np.nan)
    v = v / np.linalg.norm(v)
    for _ in range(iters):
        u = (X * v[None, :]).sum(1) / np.maximum((M * v[None, :] ** 2).sum(1), 1e-9)
        vn = (X * u[:, None]).sum(0) / np.maximum((M * u[:, None] ** 2).sum(0), 1e-9)
        nv = np.linalg.norm(vn)
        if nv < 1e-12:
            break
        vn = vn / nv
        if np.max(np.abs(vn - v)) < 1e-10:
            v = vn
            break
        v = vn
    v = v - v.mean()
    if float(v @ RSTD) < 0:
        v = -v
    return v


def permetal_curve(C: np.ndarray, rows: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Repo B1 convention: per-metal (optionally weighted) mean of already-centred rows."""
    if len(rows) == 0:
        return np.full(N_LN, np.nan)
    A = C[rows]
    ww = np.ones(len(rows)) if w is None else w[rows]
    m = ~np.isnan(A)
    num = np.nansum(np.where(m, A, 0.0) * ww[:, None], axis=0)
    den = (m * ww[:, None]).sum(axis=0)
    return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)


def main() -> None:
    df, Y, nrep, repsd = load()
    n = len(df)
    C = Y - np.nanmean(Y, axis=1, keepdims=True)
    pairs = pair_table(df, Y)
    print(f"cohort {n} cells, {pairs.shape[0]} pairs, {df.extractant.nunique()} extractants, "
          f"{df.chemotype.nunique()} chemotypes")

    # ---- 0. correctness check: rescore the saved arms with this scorer -------------
    ref_rows = []
    for arm in ["X_ENS_DIRECT+LOWRANK_K2", "C_DIRECT_ROW", "M_SELECTED", "B1_MEAN_CURVE",
                "B2_PAIRMEAN", "B3_NN_TANIMOTO", "B4_HEAVIER_ALWAYS", "B0_ZERO"]:
        f = PRED / f"{arm}.parquet"
        if not f.exists():
            continue
        q = pd.read_parquet(f)
        per_seed = []
        for seed, blk in q.groupby("split_seed"):
            b = blk.copy()
            b["adjacent"] = (b.dZ == 1) | ((b.A == "Nd") & (b.B == "Sm"))
            b["far"] = b.dZ >= FAR_MIN_DZ
            s, _ = score(b, b["prediction"].to_numpy(), arm)
            per_seed.append(s)
        s = pd.DataFrame(per_seed).drop(columns=["entry"]).mean().to_dict()
        s["entry"] = arm
        s["n_seeds"] = len(per_seed)
        ref_rows.append(s)
    ref = pd.DataFrame(ref_rows)
    ref.to_csv(OUT / "d3_arm_reference_rescored.csv", index=False)
    print("\n-- correctness check (saved arms rescored, mean over 5 seeds) --")
    print(ref[["entry", "macro_mae_extractant", "macro_mae_chemotype", "pooled_mae"]].to_string(index=False))

    # ---- group memberships ---------------------------------------------------------
    ext = df.extractant.to_numpy()
    chem = df.chemotype.to_numpy()
    pub = df.publication_id.to_numpy()
    acid_cols = [c for c in df.columns if c.startswith("cond__acid__")]
    acid_id = df[acid_cols].astype(int).astype(str).agg("".join, axis=1).to_numpy()
    conc = df["cond__acid_concentration_M"].to_numpy(dtype=float)
    chem_counts = pd.Series(chem).value_counts()
    wbal = np.array([1.0 / chem_counts[c] for c in chem])
    wbal = wbal * n / wbal.sum()

    ext_cells = pd.Series(range(n)).groupby(ext).apply(lambda s: s.to_numpy()).to_dict()
    chem_cells = pd.Series(range(n)).groupby(chem).apply(lambda s: s.to_numpy()).to_dict()
    n_ext_multi = sum(1 for k, v in ext_cells.items() if len(v) >= 2)
    print(f"\nextractants with >=2 cells: {n_ext_multi} of {len(ext_cells)}; "
          f"cells in them: {sum(len(v) for v in ext_cells.values() if len(v) >= 2)}")

    # ---- ladder --------------------------------------------------------------------
    P = {}          # entry -> 521 x 14 predicted curve (NaN = undefined)
    nan = lambda: np.full((n, N_LN), np.nan)

    P["L1_CORPUS_MEAN_permetal_w"] = nan()
    P["L1_CORPUS_MEAN_permetal"] = nan()
    P["L1_CORPUS_MEAN_aligned"] = nan()
    P["L1_CORPUS_MEAN_aligned_w"] = nan()
    P["L2_CHEMOTYPE_MEAN"] = nan()
    P["L3_EXTRACTANT_MEAN"] = nan()
    P["L3b_EXTRACTANT_MEAN_permetal"] = nan()
    P["L4_EXTRACTANT+PUBLICATION"] = nan()
    P["L5_EXTRACTANT+NEAREST_ACID"] = nan()
    P["L5b_EXT+PUB+SAME_ACID_CONC"] = nan()
    P["L6_OWN_QUADRATIC_RADIUS"] = nan()
    P["L7_ORACLE_AMPLITUDE_CORPUS_RANK1"] = nan()
    P["L7a_ORACLE_AMPLITUDE_CORPUS_MEANSHAPE"] = nan()
    P["L7b_ORACLE_AMPLITUDE_EXTRACTANT_SHAPE"] = nan()
    P["L3i_EXTRACTANT_MEAN_INSAMPLE"] = nan()
    P["CORPUS_RANK1_SHAPE"] = nan()

    l5_dconc, l5_samepub = [], []
    all_rows = np.arange(n)
    for i in range(n):
        others = all_rows[all_rows != i]
        P["L1_CORPUS_MEAN_aligned"][i] = aligned_curve(Y, others)
        P["L1_CORPUS_MEAN_aligned_w"][i] = aligned_curve(Y, others, wbal)
        P["L1_CORPUS_MEAN_permetal_w"][i] = permetal_curve(C, others, wbal)
        P["L1_CORPUS_MEAN_permetal"][i] = permetal_curve(C, others)
        P["CORPUS_RANK1_SHAPE"][i] = rank1_shape(C, others)

        cr = chem_cells[chem[i]]
        cr = cr[cr != i]
        if len(cr):
            P["L2_CHEMOTYPE_MEAN"][i] = aligned_curve(Y, cr)

        er_all = ext_cells[ext[i]]
        er = er_all[er_all != i]
        P["L3i_EXTRACTANT_MEAN_INSAMPLE"][i] = aligned_curve(Y, er_all)
        if len(er):
            P["L3_EXTRACTANT_MEAN"][i] = aligned_curve(Y, er)
            P["L3b_EXTRACTANT_MEAN_permetal"][i] = permetal_curve(C, er)
            pr = er[pub[er] == pub[i]]
            if len(pr):
                P["L4_EXTRACTANT+PUBLICATION"][i] = aligned_curve(Y, pr)
            cand = er[(acid_id[er] == acid_id[i]) & np.isfinite(conc[er])]
            if len(cand) and np.isfinite(conc[i]):
                d = np.abs(conc[cand] - conc[i])
                donors = cand[d == d.min()]
                P["L5_EXTRACTANT+NEAREST_ACID"][i] = aligned_curve(Y, donors)
                l5_dconc.append(float(d.min()))
                l5_samepub.append(float(np.mean(pub[donors] == pub[i])))
                nd = donors[(pub[donors] == pub[i]) & (conc[donors] == conc[i])]
                if len(nd):
                    P["L5b_EXT+PUB+SAME_ACID_CONC"][i] = aligned_curve(Y, nd)

        obs = np.flatnonzero(~np.isnan(Y[i]))
        if len(obs) >= 3:
            X = np.column_stack([np.ones(len(obs)), RSTD[obs], RSTD[obs] ** 2])
            beta, *_ = np.linalg.lstsq(X, Y[i, obs], rcond=None)
            full = np.column_stack([np.ones(N_LN), RSTD, RSTD ** 2])
            P["L6_OWN_QUADRATIC_RADIUS"][i] = full @ beta

    # oracle amplitude entries need the (LOO) shapes computed above
    for i in range(n):
        obs = np.flatnonzero(~np.isnan(Y[i]))
        yc = C[i, obs]
        for src, dst in (("CORPUS_RANK1_SHAPE", "L7_ORACLE_AMPLITUDE_CORPUS_RANK1"),
                         ("L1_CORPUS_MEAN_permetal_w", "L7a_ORACLE_AMPLITUDE_CORPUS_MEANSHAPE"),
                         ("L3_EXTRACTANT_MEAN", "L7b_ORACLE_AMPLITUDE_EXTRACTANT_SHAPE")):
            g = P[src][i]
            if not np.isfinite(g[obs]).all():
                continue
            s = g[obs] - g[obs].mean()
            if float(s @ s) < 1e-12:
                continue
            a = float(yc @ s) / float(s @ s)
            P[dst][i] = a * g

    # ---- score the ladder ----------------------------------------------------------
    order = ["L1_CORPUS_MEAN_permetal_w", "L1_CORPUS_MEAN_permetal", "L1_CORPUS_MEAN_aligned",
             "L1_CORPUS_MEAN_aligned_w", "L2_CHEMOTYPE_MEAN",
             "L3_EXTRACTANT_MEAN", "L3b_EXTRACTANT_MEAN_permetal", "L4_EXTRACTANT+PUBLICATION",
             "L5_EXTRACTANT+NEAREST_ACID", "L5b_EXT+PUB+SAME_ACID_CONC", "L6_OWN_QUADRATIC_RADIUS",
             "L7_ORACLE_AMPLITUDE_CORPUS_RANK1", "L7a_ORACLE_AMPLITUDE_CORPUS_MEANSHAPE",
             "L7b_ORACLE_AMPLITUDE_EXTRACTANT_SHAPE", "L3i_EXTRACTANT_MEAN_INSAMPLE"]
    predmat = {k: curve_to_pairs(P[k], pairs) for k in order}

    # real arms, re-indexed onto the same pair rows so they can be scored on the same subsets
    key = pd.MultiIndex.from_arrays([pairs.cell_id, pairs.A, pairs.B])
    arm_pred = {}
    for arm in ["X_ENS_DIRECT+LOWRANK_K2", "C_DIRECT_ROW", "M_SELECTED", "B1_MEAN_CURVE"]:
        f = PRED / f"{arm}.parquet"
        if not f.exists():
            continue
        q = pd.read_parquet(f)
        mats = []
        for seed, blk in q.groupby("split_seed"):
            s = blk.set_index(["cell_id", "A", "B"])["prediction"]
            mats.append(s.reindex(key).to_numpy())
        arm_pred[f"ARM_{arm}"] = np.column_stack(mats)   # n_pairs x n_seeds

    def score_arms(sub_pairs: pd.DataFrame, mask: np.ndarray) -> list[dict]:
        out = []
        for name, M in arm_pred.items():
            per_seed = [score(sub_pairs, M[mask, j], name)[0] for j in range(M.shape[1])]
            rec = pd.DataFrame(per_seed).drop(columns=["entry"]).mean().to_dict()
            rec["entry"] = name
            out.append(rec)
        return out

    rows, per_ext_tables = [], {}
    for k in order:
        s, per = score(pairs, predmat[k], k)
        rows.append(s)
        per_ext_tables[k] = per
    rows += score_arms(pairs, np.ones(len(pairs), dtype=bool))
    ladder = pd.DataFrame(rows)
    ladder.to_csv(OUT / "d3_ladder_full.csv", index=False)
    print("\n-- ladder, each entry on the cells where it is defined --")
    print(ladder[["entry", "macro_mae_extractant", "macro_mae_chemotype", "pooled_mae",
                  "macro_mae_adjacent", "macro_mae_far", "n_extractants", "n_cells", "n_pairs"]]
          .to_string(index=False))

    # like-for-like: pairs where L3, L4 and L5 are all defined
    common = np.isfinite(predmat["L3_EXTRACTANT_MEAN"]) & np.isfinite(predmat["L4_EXTRACTANT+PUBLICATION"]) \
        & np.isfinite(predmat["L5_EXTRACTANT+NEAREST_ACID"])
    sub = pairs[common].reset_index(drop=True)
    rows_c = []
    for k in order:
        s, _ = score(sub, predmat[k][common], k)
        rows_c.append(s)
    rows_c += score_arms(sub, common)
    common_tbl = pd.DataFrame(rows_c)
    common_tbl.to_csv(OUT / "d3_ladder_common_subset.csv", index=False)
    print(f"\n-- like-for-like subset where L3+L4+L5 all defined: {sub.cell_id.nunique()} cells, "
          f"{sub.extractant.nunique()} extractants, {len(sub)} pairs --")
    print(common_tbl[["entry", "macro_mae_extractant", "macro_mae_chemotype", "pooled_mae",
                      "macro_mae_adjacent", "macro_mae_far", "n_extractants", "n_cells", "n_pairs"]]
          .to_string(index=False))

    # subset where L3 is defined (extractants with >=2 cells)
    m3 = np.isfinite(predmat["L3_EXTRACTANT_MEAN"])
    sub3 = pairs[m3].reset_index(drop=True)
    rows3 = [score(sub3, predmat[k][m3], k)[0] for k in order] + score_arms(sub3, m3)
    tbl3 = pd.DataFrame(rows3)
    tbl3.to_csv(OUT / "d3_ladder_multicell_extractants.csv", index=False)
    print(f"\n-- subset where L3 defined: {sub3.cell_id.nunique()} cells, "
          f"{sub3.extractant.nunique()} extractants, {len(sub3)} pairs --")
    print(tbl3[["entry", "macro_mae_extractant", "macro_mae_chemotype", "pooled_mae",
                "n_extractants", "n_cells", "n_pairs"]].to_string(index=False))

    # rich cells only (n_metals >= 5) so the in-sample quadratic has >= 2 residual df
    m5 = (pairs["n_metals"] >= 5).to_numpy()
    sub5 = pairs[m5].reset_index(drop=True)
    rows5 = [score(sub5, predmat[k][m5], k)[0] for k in order] + score_arms(sub5, m5)
    tbl5 = pd.DataFrame(rows5)
    tbl5.to_csv(OUT / "d3_ladder_nmetals_ge5.csv", index=False)
    print(f"\n-- cells with n_metals >= 5: {sub5.cell_id.nunique()} cells, {len(sub5)} pairs --")
    print(tbl5[["entry", "macro_mae_extractant", "pooled_mae", "n_extractants", "n_cells", "n_pairs"]]
          .to_string(index=False))

    pd.concat([t.assign(entry=k) for k, t in per_ext_tables.items()]).reset_index().to_csv(
        OUT / "d3_ladder_per_extractant.csv", index=False)

    # error profile against dZ on the like-for-like subset
    prof = []
    for k in order + list(arm_pred):
        v = predmat[k][common] if k in predmat else arm_pred[k][common].mean(axis=1)
        e = np.abs(sub["y"].to_numpy() - v)
        row_p = {"entry": k}
        for dz in range(1, 15):
            sel = (sub["dZ"].to_numpy() == dz) & np.isfinite(e)
            row_p[f"dZ{dz}"] = float(e[sel].mean()) if sel.sum() >= 20 else np.nan
            row_p[f"n_dZ{dz}"] = int(sel.sum())
        prof.append(row_p)
    pd.DataFrame(prof).to_csv(OUT / "d3_error_vs_dZ_common_subset.csv", index=False)

    # L5b: the closest thing the cohort has to a repeat experiment
    m5b = np.isfinite(predmat["L5b_EXT+PUB+SAME_ACID_CONC"])
    sub5b = pairs[m5b].reset_index(drop=True)
    rows5b = [score(sub5b, predmat[k][m5b], k)[0] for k in order] + score_arms(sub5b, m5b)
    tbl5b = pd.DataFrame(rows5b)
    tbl5b.to_csv(OUT / "d3_ladder_near_duplicate_subset.csv", index=False)
    print(f"\n-- near-duplicate subset (same extractant + publication + acid concentration): "
          f"{sub5b.cell_id.nunique()} cells, {sub5b.extractant.nunique()} extractants, {len(sub5b)} pairs --")
    print(tbl5b[["entry", "macro_mae_extractant", "macro_mae_chemotype", "pooled_mae",
                 "macro_mae_adjacent", "macro_mae_far", "n_extractants", "n_cells", "n_pairs"]]
          .to_string(index=False))

    # amplitude vs shape share, on the full cohort
    L = ladder.set_index("entry")["macro_mae_extractant"]
    base, orc = float(L["L1_CORPUS_MEAN_permetal_w"]), float(L["L7_ORACLE_AMPLITUDE_CORPUS_RANK1"])
    arm = float(L["ARM_X_ENS_DIRECT+LOWRANK_K2"])
    print(f"\namplitude share: corpus mean {base:.3f} -> oracle rank-1 amplitude {orc:.3f} "
          f"= {(base - orc) / base:.1%} of the corpus-mean error is one scalar per cell; "
          f"the best arm {arm:.3f} closes {(base - arm) / (base - orc):.1%} of that gap")

    print(f"\nL5 donor proximity: {len(l5_dconc)} cells scored; median |d[acid]| "
          f"{np.median(l5_dconc):.3f} M, exact match in {np.mean(np.array(l5_dconc) == 0):.1%} of cells; "
          f"donor from the same publication in {np.mean(l5_samepub):.1%} (mean donor share)")

    # ---- variance decomposition on the pairwise scale -------------------------------
    dec = []
    tot_ssw = tot_dfw = tot_ssb = tot_dfb = 0.0
    tot_n0_num = tot_n0_den = 0.0
    tot_ss = tot_df = 0.0
    resid_within, raw_within = [], []
    for (a, b), blk in pairs.groupby(["A", "B"]):
        yv = blk["y"].to_numpy()
        gm = blk.groupby("extractant")["y"]
        means, sizes = gm.mean().to_numpy(), gm.size().to_numpy()
        N, G = len(yv), len(sizes)
        if N < 3 or G < 2 or N == G:
            continue
        grand = yv.mean()
        res = yv - blk["extractant"].map(gm.mean()).to_numpy()
        raw_within.append(res)
        ssw = float((res ** 2).sum())
        ssb = float((sizes * (means - grand) ** 2).sum())
        resid_within.append(res * np.sqrt(N / (N - G)))   # scaled so var matches MSW
        tot_ss += ssw + ssb
        tot_df += N - 1
        tot_ssw += ssw
        tot_dfw += N - G
        tot_ssb += ssb
        tot_dfb += G - 1
        tot_n0_num += N - (sizes ** 2).sum() / N
        tot_n0_den += G - 1
        dec.append({"A": a, "B": b, "dZ": int(blk.dZ.iat[0]), "n_cells": N, "n_extractants": G,
                    "var_total": (ssw + ssb) / max(N - 1, 1),
                    "var_within_ext": ssw / (N - G),
                    "var_between_ext": max((ssb / (G - 1) - ssw / (N - G)) /
                                           ((N - (sizes ** 2).sum() / N) / (G - 1)), 0.0),
                    "mean_abs_y": float(np.abs(yv).mean())})
    decdf = pd.DataFrame(dec).sort_values(["dZ", "A"])
    decdf.to_csv(OUT / "d3_variance_by_pair_type.csv", index=False)

    msw = tot_ssw / tot_dfw
    msb = tot_ssb / tot_dfb
    n0 = tot_n0_num / tot_n0_den
    var_between = max((msb - msw) / n0, 0.0)
    var_total = tot_ss / tot_df
    resid_within = np.concatenate(resid_within)
    k_emp = float(np.abs(resid_within).mean() / resid_within.std(ddof=0))   # MAE/sd, empirical
    K = np.sqrt(2 / np.pi)                                                   # MAE/sd, Gaussian
    print(f"\nempirical MAE/sd ratio of within-extractant residuals: {k_emp:.3f} (Gaussian {K:.3f})")

    # replicate noise. log D is the mean of nrep rows, so se = repsd / sqrt(nrep).
    rep_ok = (nrep >= 2) & np.isfinite(repsd)
    rs = repsd[rep_ok]
    print("repsd quantiles (216 replicated cell-metal entries): "
          + " ".join(f"q{int(q*100)}={np.quantile(rs, q):.3f}" for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)))
    pooled_sigma2 = float(np.nansum((nrep[rep_ok] - 1) * repsd[rep_ok] ** 2) / np.nansum(nrep[rep_ok] - 1))
    med_sigma = float(np.median(rs))
    mean_sigma = float(rs.mean())
    se = np.where(rep_ok, repsd / np.sqrt(np.maximum(nrep, 1)), np.nan)
    r, ia, ib = pairs["row"].to_numpy(), pairs["ia"].to_numpy(), pairs["ib"].to_numpy()
    sd_pair = np.sqrt(se[r, ia] ** 2 + se[r, ib] ** 2)
    both = np.isfinite(sd_pair)
    # cell-level unit: median pairwise se-sd inside each replicated cell, then median over cells
    cell_med = pd.Series(sd_pair[both]).groupby(pairs.loc[both, "cell_id"].to_numpy()).median()

    def row(name, sd, basis):
        return {"component": name, "variance": sd ** 2, "sd": sd,
                "implied_mae_gaussian": K * sd, "implied_mae_empirical_ratio": k_emp * sd, "basis": basis}

    var_rows = [
        row("pairwise log SF: total variance across cells (within pair type)", float(np.sqrt(var_total)),
            f"{int(tot_df)} df over {len(decdf)} pair types"),
        row("  component: between extractants", float(np.sqrt(var_between)),
            f"ANOVA MSB {msb:.4f} MSW {msw:.4f} n0 {n0:.3f}, {int(tot_dfb)} df"),
        row("  component: between cells of one extractant", float(np.sqrt(msw)),
            f"{int(tot_dfw)} within-extractant df, {len(decdf)} pair types"),
        row("replicate: one logD point, median within-replicate sd", med_sigma,
            f"median over {int(rep_ok.sum())} replicated (cell,metal) entries in {int(rep_ok.any(axis=1).sum())} cells"),
        row("replicate: one logD point, mean within-replicate sd", mean_sigma, "same 216 entries"),
        row("replicate: one logD point, pooled (df-weighted) sd", float(np.sqrt(pooled_sigma2)),
            "same 216 entries, weighted by nrep-1"),
        row("replicate MAE floor, pair of nrep=1 points, median sigma", float(np.sqrt(2) * med_sigma),
            "the case for 3143 of 3359 observed (cell,metal) entries"),
        row("replicate MAE floor, pair of nrep=1 points, mean sigma", float(np.sqrt(2) * mean_sigma),
            "same, mean instead of median sigma"),
        row("replicate MAE floor, pair of nrep=2 means, median sigma", float(med_sigma),
            "credit for averaging 2 replicates"),
        row("replicate MAE floor, observed pairs with both metals replicated (median over cells)",
            float(cell_med.median()),
            f"{int(both.sum())} pairs in {len(cell_med)} cells; median-of-cell-medians"),
    ]
    vardf = pd.DataFrame(var_rows)
    vardf.to_csv(OUT / "d3_variance_components.csv", index=False)
    print("\n-- variance decomposition of the pairwise log SF --")
    print(vardf.to_string(index=False))
    frac_b = var_between / (var_between + msw)
    print(f"\nbetween-extractant share of explainable pairwise variance: {frac_b:.3f} "
          f"({var_between:.4f} of {var_between + msw:.4f})")
    raw_within = np.concatenate(raw_within)
    print(f"direct empirical MAE of the in-sample per-pair-type extractant mean: "
          f"{float(np.abs(raw_within).mean()):.3f} (bias-corrected {float(np.abs(resid_within).mean()):.3f}) "
          f"over {len(raw_within)} pairs")

    # what the ladder itself says about the noise floor: no predictor can beat the noise,
    # so the best achieved extractant-macro MAE upper-bounds the typical pairwise noise.
    best_achieved = float(min(common_tbl.set_index("entry").loc[
        ["L4_EXTRACTANT+PUBLICATION", "L5_EXTRACTANT+NEAREST_ACID"], "macro_mae_extractant"]))
    print(f"\nbest achieved non-in-sample MAE on the like-for-like subset: {best_achieved:.3f} "
          f"-> implied upper bound on typical pairwise noise sd {best_achieved / K:.3f} "
          f"and on per-point sd {best_achieved / K / np.sqrt(2):.3f}")
    pd.DataFrame([{"best_achieved_macro_mae": best_achieved,
                   "implied_pair_noise_sd_upper_bound": best_achieved / K,
                   "implied_point_noise_sd_upper_bound": best_achieved / K / np.sqrt(2),
                   "median_repsd_point": med_sigma, "mean_repsd_point": mean_sigma,
                   "empirical_mae_sd_ratio": k_emp,
                   "between_ext_share": frac_b}]).to_csv(OUT / "d3_noise_bounds.csv", index=False)
    sd_pair_typ_pooled = float(np.sqrt(2) * med_sigma)

    # per-metal curve-level decomposition (secondary view)
    gext = {e: aligned_curve(Y, v) for e, v in ext_cells.items()}
    rows_m = []
    for j, m in enumerate(LANTHANIDES):
        dev, val = [], []
        for i in range(n):
            if np.isnan(C[i, j]):
                continue
            val.append(C[i, j])
            g = gext[ext[i]]
            obs = np.flatnonzero(~np.isnan(Y[i]))
            if np.isfinite(g[obs]).all() and len(ext_cells[ext[i]]) >= 2:
                gc = g[j] - g[obs].mean()
                dev.append(C[i, j] - gc)
        rows_m.append({"metal": m, "n_cells": len(val), "sd_centred_value": float(np.std(val, ddof=1)),
                       "n_cells_multi_ext": len(dev),
                       "sd_dev_from_extractant_curve": float(np.std(dev, ddof=1)) if len(dev) > 2 else np.nan})
    pd.DataFrame(rows_m).to_csv(OUT / "d3_curve_variance_by_metal.csv", index=False)

    # ---- figure --------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = common_tbl.set_index("entry")
    keys = ["L1_CORPUS_MEAN_permetal", "L2_CHEMOTYPE_MEAN", "L3_EXTRACTANT_MEAN",
            "L4_EXTRACTANT+PUBLICATION", "L5_EXTRACTANT+NEAREST_ACID",
            "L7_ORACLE_AMPLITUDE_CORPUS_RANK1", "L7b_ORACLE_AMPLITUDE_EXTRACTANT_SHAPE",
            "L6_OWN_QUADRATIC_RADIUS"]
    arm_keys = ["ARM_B1_MEAN_CURVE", "ARM_C_DIRECT_ROW", "ARM_X_ENS_DIRECT+LOWRANK_K2"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    ax = axes[0]
    allk = arm_keys + keys
    ypos = np.arange(len(allk))
    cols = ["#e45756"] * len(arm_keys) + ["#4c78a8"] * len(keys)
    ax.barh(ypos, show.loc[allk, "macro_mae_extractant"], color=cols)
    for j, k in enumerate(allk):
        ax.text(show.loc[k, "macro_mae_extractant"] + 0.006, j,
                f"{show.loc[k, 'macro_mae_extractant']:.3f}", va="center", fontsize=7.5)
    ax.set_yticks(ypos, allk, fontsize=7.5)
    ax.invert_yaxis()
    lo, hi = K * float(cell_med.median()), K * sd_pair_typ_pooled
    ax.axvspan(lo, hi, color="#54a24b", alpha=0.15,
               label=f"replicate MAE floor {lo:.2f}-{hi:.2f}")
    ax.set_xlabel("extractant-macro MAE of log SF")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(f"ladder on the like-for-like subset\n({sub.cell_id.nunique()} cells, "
                 f"{sub.extractant.nunique()} extractants, {len(sub)} pairs)", fontsize=9)
    ax = axes[1]
    for k, c in zip(["ARM_X_ENS_DIRECT+LOWRANK_K2", "L1_CORPUS_MEAN_permetal", "L2_CHEMOTYPE_MEAN",
                     "L3_EXTRACTANT_MEAN", "L7_ORACLE_AMPLITUDE_CORPUS_RANK1"],
                    ["#e45756", "#888888", "#f58518", "#4c78a8", "#54a24b"]):
        v = predmat[k][common] if k in predmat else arm_pred[k][common].mean(axis=1)
        e = np.abs(sub["y"].to_numpy() - v)
        xs, ys = [], []
        for dz in range(1, 15):
            sel = (sub["dZ"].to_numpy() == dz) & np.isfinite(e)
            if sel.sum() >= 20:
                xs.append(dz)
                ys.append(e[sel].mean())
        ax.plot(xs, ys, marker="o", ms=3.5, lw=1.4, color=c, label=k)
    ax.set_xlabel("dZ (atomic-number separation of the pair)")
    ax.set_ylabel("pooled MAE of log SF")
    ax.set_title("error grows with dZ; the oracles keep it flat", fontsize=9)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "D3_ceiling_ladder.png", dpi=150)
    plt.close(fig)
    print(f"\nwrote {FIG / 'D3_ceiling_ladder.png'}")


if __name__ == "__main__":
    main()
