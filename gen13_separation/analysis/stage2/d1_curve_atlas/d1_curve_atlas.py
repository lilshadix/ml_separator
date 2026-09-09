"""D1 - per-extractant lanthanide-curve atlas and curve reproducibility.

Run from the repo root:
    .venv/Scripts/python.exe gen13_separation/analysis/stage2/d1_curve_atlas/d1_curve_atlas.py

Writes CSVs into gen13_separation/analysis/stage2/d1_curve_atlas/ and figures into
gen13_separation/figures/stage2/ with the prefix ``d1_``.
"""

from __future__ import annotations

import itertools
import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import rankdata

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REPO = os.path.abspath(os.path.join(ROOT, ".."))
sys.path.insert(0, ROOT)

from gen13sep.metals import ATOMIC_NUMBER, LANTHANIDES, SHANNON_RADIUS_CN8  # noqa: E402

OUT = os.path.join(ROOT, "analysis", "stage2", "d1_curve_atlas")
FIG = os.path.join(ROOT, "figures", "stage2")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

COHORT = os.path.join(ROOT, "manifests", "cohort_exact.parquet")

LN = list(LANTHANIDES)
RAD = np.array([SHANNON_RADIUS_CN8[m] for m in LN], dtype=float)
ZNUM = np.array([ATOMIC_NUMBER[m] for m in LN], dtype=float)
# standardised Shannon CN8 radius over the 14-element axis (population sd)
RAD_Z = (RAD - RAD.mean()) / RAD.std()
RADZ_BY_METAL = dict(zip(LN, RAD_Z))

RNG = np.random.default_rng(20260908)

# The nitric-acid concentration used for colouring: the cohort stores one acid
# identity flag per cell (cond__acid__*) plus a single molarity column.
ACID_CONC_COL = "cond__acid_concentration_M"
HNO3_FLAG_COL = "cond__acid__hno3"

MIN_SHARED = 4          # metals two cells must share to be compared
MIN_CELLS_REPRO = 3     # cells an extractant needs for the reproducibility table
MIN_PTS_QUAD = 5        # metals a cell needs for the quadratic fit
N_RANDOM_PAIRS = 20000  # random cell-pair draws for the null level


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_cells() -> pd.DataFrame:
    df = pd.read_parquet(COHORT)
    logd = df[[f"logD__{m}" for m in LN]].to_numpy(dtype=float)
    # centred curve: subtract the mean over the metals actually observed
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        centred = logd - np.nanmean(logd, axis=1, keepdims=True)
    for j, m in enumerate(LN):
        df[f"c__{m}"] = centred[:, j]
    df["n_obs"] = np.isfinite(logd).sum(axis=1)
    df["hno3_M"] = np.where(df[HNO3_FLAG_COL] == 1, df[ACID_CONC_COL], np.nan)
    return df


def curve_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[[f"c__{m}" for m in LN]].to_numpy(dtype=float)


# --------------------------------------------------------------------------- #
# 1. ranking
# --------------------------------------------------------------------------- #
def rank_extractants(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for smi, g in df.groupby("extractant", sort=False):
        names = g["extractant_name"].dropna().unique().tolist()
        rows.append(
            dict(
                extractant=smi,
                extractant_name=names[0] if names else "",
                all_names=" | ".join(names),
                chemotype=g["chemotype"].mode().iat[0],
                n_cells=len(g),
                n_entries=int(g["n_obs"].sum()),
                n_publications=g["publication_id"].nunique(),
                n_condition_keys=g["condition_key"].nunique(),
                median_n_metals=float(g["n_obs"].median()),
                max_n_metals=int(g["n_obs"].max()),
                n_rows_total=int(g["n_rows"].sum()),
                hno3_cells=int((g[HNO3_FLAG_COL] == 1).sum()),
                hno3_M_min=float(np.nanmin(g["hno3_M"])) if g["hno3_M"].notna().any() else np.nan,
                hno3_M_max=float(np.nanmax(g["hno3_M"])) if g["hno3_M"].notna().any() else np.nan,
                replicate_sd_median=float(g["replicate_sd_median"].median()),
            )
        )
    out = pd.DataFrame(rows).sort_values(
        ["n_cells", "n_entries"], ascending=False, kind="mergesort"
    )
    out.insert(0, "rank_by_cells", np.arange(1, len(out) + 1))
    out["rank_by_entries"] = out["n_entries"].rank(ascending=False, method="min").astype(int)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. figures
# --------------------------------------------------------------------------- #
PUB_STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1))]
PUB_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]


def _acid_colour(v, norm, cmap):
    if not np.isfinite(v):
        return "0.55"
    return cmap(norm(v))


def plot_extractant(g: pd.DataFrame, name: str, smi: str, fname: str) -> None:
    """Two panels for one extractant: centred logD vs Shannon CN8 radius and vs Z."""
    C = curve_matrix(g)
    finite = C[np.isfinite(C)]
    if finite.size == 0:
        return
    lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    pad = max(0.08 * (hi - lo), 0.05)
    ylim = (lo - pad, hi + pad)

    acid = g["hno3_M"].to_numpy(dtype=float)
    fin = acid[np.isfinite(acid)]
    cmap = plt.get_cmap("viridis")
    if fin.size:
        norm = matplotlib.colors.LogNorm(
            vmin=max(fin.min(), 1e-3), vmax=max(fin.max(), fin.min() * 1.0001 + 1e-6)
        )
    else:
        norm = matplotlib.colors.Normalize(0, 1)

    pubs = sorted(g["publication_id"].astype(str).unique())
    pub_style = {p: PUB_STYLES[i % len(PUB_STYLES)] for i, p in enumerate(pubs)}
    pub_marker = {p: PUB_MARKERS[i % len(PUB_MARKERS)] for i, p in enumerate(pubs)}

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4))
    for ax, xvals, xlab in (
        (axes[0], RAD, "Shannon ionic radius Ln(III), CN 8 (A)"),
        (axes[1], ZNUM, "atomic number Z"),
    ):
        for i in range(len(g)):
            y = C[i]
            ok = np.isfinite(y)
            if ok.sum() < 2:
                if ok.sum() == 1:
                    ax.plot(xvals[ok], y[ok], marker="o", ms=4,
                            color=_acid_colour(acid[i], norm, cmap), lw=0)
                continue
            p = str(g["publication_id"].iat[i])
            ax.plot(
                xvals[ok], y[ok],
                color=_acid_colour(acid[i], norm, cmap),
                linestyle=pub_style[p], marker=pub_marker[p],
                ms=3.6, lw=1.25, alpha=0.85,
            )
        ax.axhline(0.0, color="0.3", lw=0.7, zorder=0)
        ax.set_xlabel(xlab)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.25, lw=0.5)
        # element symbols along the top
        for m, xv in zip(LN, xvals):
            ax.annotate(m, (xv, ylim[1]), xytext=(0, -9), textcoords="offset points",
                        ha="center", va="top", fontsize=7, color="0.35")
    axes[0].invert_xaxis()  # light (large radius) on the left -> heavy on the right
    axes[0].set_ylabel("centred log D  (log D minus the cell mean over observed Ln)")

    if fin.size:
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, ax=axes, fraction=0.028, pad=0.015)
        cb.set_label("HNO$_3$ concentration (mol/L)")
    handles = [
        Line2D([], [], color="0.25", linestyle=pub_style[p], marker=pub_marker[p],
               ms=4, lw=1.2, label=f"pub {p}")
        for p in pubs[:10]
    ]
    if len(pubs) > 10:
        handles.append(Line2D([], [], color="none", label=f"(+{len(pubs)-10} more publications)"))
    axes[1].legend(handles=handles, fontsize=6.5, ncol=1, loc="best", framealpha=0.85)

    fig.suptitle(
        f"{name}   -   {len(g)} cells, {g['publication_id'].nunique()} publications, "
        f"{int(g['n_obs'].sum())} observed (cell, metal) entries\n{smi}",
        fontsize=10,
    )
    fig.savefig(os.path.join(FIG, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_grid(df: pd.DataFrame, top: pd.DataFrame, fname: str) -> None:
    n = len(top)
    ncol, nrow = 4, int(np.ceil(n / 4))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 3.1 * nrow), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    allacid = df["hno3_M"].to_numpy(dtype=float)
    fin = allacid[np.isfinite(allacid)]
    cmap = plt.get_cmap("viridis")
    norm = matplotlib.colors.LogNorm(vmin=max(fin.min(), 1e-3), vmax=fin.max())
    for ax, (_, r) in zip(axes, top.iterrows()):
        g = df[df["extractant"] == r["extractant"]]
        C = curve_matrix(g)
        acid = g["hno3_M"].to_numpy(dtype=float)
        for i in range(len(g)):
            y = C[i]
            ok = np.isfinite(y)
            if ok.sum() < 2:
                continue
            ax.plot(RAD[ok], y[ok], color=_acid_colour(acid[i], norm, cmap),
                    lw=1.0, alpha=0.8, marker="o", ms=2.4)
        ax.axhline(0, color="0.3", lw=0.6, zorder=0)
        ax.set_title(f"{r['extractant_name'][:28]}  (n={r['n_cells']})", fontsize=8.5)
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=7)
    # one invert only: the axes share x, so inverting each would cancel out
    axes[0].set_xlim(RAD.max() + 0.006, RAD.min() - 0.006)
    show = ["La", "Nd", "Eu", "Tb", "Er", "Lu"]
    axes[0].set_xticks([SHANNON_RADIUS_CN8[m] for m in show])
    axes[0].set_xticklabels(show)
    for ax in axes[n:]:
        ax.axis("off")
    for ax in axes[max(0, n - ncol):n]:
        ax.set_xlabel("Ln (positioned by Shannon CN8 radius)", fontsize=8)
    for k in range(0, n, ncol):
        axes[k].set_ylabel("centred log D", fontsize=8)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axes.tolist(), fraction=0.02, pad=0.012)
    cb.set_label("HNO$_3$ concentration (mol/L);  grey = non-HNO$_3$ acid")
    fig.suptitle(
        "D1 atlas - centred lanthanide curves of every cell, 12 extractants with the most cells "
        "(light Ln left, heavy Ln right; y axis is per panel)", fontsize=11,
    )
    fig.savefig(os.path.join(FIG, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 3. reproducibility
# --------------------------------------------------------------------------- #
def pair_stats(ci: np.ndarray, cj: np.ndarray):
    """Statistics between two centred curves over the metals they share.

    ``sf_mae`` is the mean absolute difference in log SF over every metal pair
    inside the shared set, i.e. exactly the error a model would make if it
    predicted one cell's separation factors from the other cell.  It is on the
    same scale as the programme's extractant-macro MAE (0.481 for the best arm).
    """
    ok = np.isfinite(ci) & np.isfinite(cj)
    k = int(ok.sum())
    if k < MIN_SHARED:
        return None
    a, b = ci[ok], cj[ok]
    ar, br = a - a.mean(), b - b.mean()  # re-centre on the shared metals
    mae_shared = float(np.abs(ar - br).mean())
    mae_raw = float(np.abs(a - b).mean())
    d = ar - br
    sf = np.abs(d[:, None] - d[None, :])
    iu = np.triu_indices(k, 1)
    sf_mae = float(sf[iu].mean())
    if ar.std() < 1e-12 or br.std() < 1e-12:
        pr = sr = np.nan
    else:
        pr = float(np.dot(ar, br) / (np.linalg.norm(ar) * np.linalg.norm(br)))
        ra, rb = rankdata(a), rankdata(b)
        ra = ra - ra.mean()
        rb = rb - rb.mean()
        na, nb = np.linalg.norm(ra), np.linalg.norm(rb)
        sr = float(np.dot(ra, rb) / (na * nb)) if na > 1e-12 and nb > 1e-12 else np.nan
    return dict(n_shared=k, pearson=pr, spearman=sr,
                mae_shared_centred=mae_shared, mae_raw_centred=mae_raw,
                sf_mae=sf_mae)


def within_extractant_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    C = curve_matrix(df)
    idx = {cid: i for i, cid in enumerate(df["cell_id"])}
    per_ext, all_pairs = [], []
    for smi, g in df.groupby("extractant", sort=False):
        if len(g) < MIN_CELLS_REPRO:
            continue
        ids = g["cell_id"].tolist()
        pubs = dict(zip(g["cell_id"], g["publication_id"].astype(str)))
        acidm = dict(zip(g["cell_id"], g["hno3_M"].astype(float)))
        recs = []
        for a, b in itertools.combinations(ids, 2):
            st = pair_stats(C[idx[a]], C[idx[b]])
            if st is None:
                continue
            st.update(extractant=smi, cell_a=a, cell_b=b,
                      same_pub=int(pubs[a] == pubs[b]),
                      dlog10_hno3=abs(np.log10(acidm[a]) - np.log10(acidm[b]))
                      if np.isfinite(acidm[a]) and np.isfinite(acidm[b])
                      and acidm[a] > 0 and acidm[b] > 0 else np.nan)
            recs.append(st)
        if not recs:
            continue
        P = pd.DataFrame(recs)
        all_pairs.append(P)
        row = dict(
            extractant=smi,
            extractant_name=g["extractant_name"].dropna().iat[0] if g["extractant_name"].notna().any() else "",
            chemotype=g["chemotype"].mode().iat[0],
            n_cells=len(g), n_publications=g["publication_id"].nunique(),
            n_pairs=len(P),
            mean_pearson=P["pearson"].mean(), mean_spearman=P["spearman"].mean(),
            mean_mae=P["mae_shared_centred"].mean(),
            median_mae=P["mae_shared_centred"].median(),
            mean_sf_mae=P["sf_mae"].mean(),
            median_sf_mae=P["sf_mae"].median(),
            mean_n_shared=P["n_shared"].mean(),
        )
        for lab, sub in (("samepub", P[P.same_pub == 1]), ("diffpub", P[P.same_pub == 0])):
            row[f"n_pairs_{lab}"] = len(sub)
            row[f"mean_pearson_{lab}"] = sub["pearson"].mean() if len(sub) else np.nan
            row[f"mean_spearman_{lab}"] = sub["spearman"].mean() if len(sub) else np.nan
            row[f"mean_mae_{lab}"] = sub["mae_shared_centred"].mean() if len(sub) else np.nan
            row[f"mean_sf_mae_{lab}"] = sub["sf_mae"].mean() if len(sub) else np.nan
        per_ext.append(row)
    return (pd.DataFrame(per_ext).sort_values("n_cells", ascending=False).reset_index(drop=True),
            pd.concat(all_pairs, ignore_index=True) if all_pairs else pd.DataFrame())


def cross_extractant_within_chemotype(df: pd.DataFrame, max_pairs=40000) -> pd.DataFrame:
    C = curve_matrix(df)
    idx = {cid: i for i, cid in enumerate(df["cell_id"])}
    recs = []
    for ct, g in df.groupby("chemotype", sort=False):
        if g["extractant"].nunique() < 2:
            continue
        ids = g["cell_id"].tolist()
        ext = dict(zip(g["cell_id"], g["extractant"]))
        pairs = list(itertools.combinations(ids, 2))
        pairs = [p for p in pairs if ext[p[0]] != ext[p[1]]]
        if len(pairs) > max_pairs:
            sel = RNG.choice(len(pairs), max_pairs, replace=False)
            pairs = [pairs[i] for i in sel]
        for a, b in pairs:
            st = pair_stats(C[idx[a]], C[idx[b]])
            if st is None:
                continue
            st["chemotype"] = ct
            recs.append(st)
    return pd.DataFrame(recs)


def random_pairs(df: pd.DataFrame, n=N_RANDOM_PAIRS) -> pd.DataFrame:
    C = curve_matrix(df)
    N = len(df)
    recs = []
    seen = set()
    tries = 0
    while len(recs) < n and tries < 25 * n:
        tries += 1
        i, j = RNG.integers(0, N, 2)
        if i == j:
            continue
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        st = pair_stats(C[i], C[j])
        if st is None:
            continue
        recs.append(st)
    return pd.DataFrame(recs)


def summarise(P: pd.DataFrame, label: str) -> dict:
    if len(P) == 0:
        return dict(level=label, n_pairs=0)
    return dict(
        level=label, n_pairs=len(P),
        mean_pearson=P["pearson"].mean(), median_pearson=P["pearson"].median(),
        frac_pearson_gt_0p8=float((P["pearson"] > 0.8).mean()),
        frac_pearson_lt_0=float((P["pearson"] < 0).mean()),
        mean_spearman=P["spearman"].mean(),
        mean_mae=P["mae_shared_centred"].mean(), median_mae=P["mae_shared_centred"].median(),
        p90_mae=P["mae_shared_centred"].quantile(0.90),
        mean_mae_raw=P["mae_raw_centred"].mean(),
        mean_sf_mae=P["sf_mae"].mean(), median_sf_mae=P["sf_mae"].median(),
        p90_sf_mae=P["sf_mae"].quantile(0.90),
        mean_n_shared=P["n_shared"].mean(),
    )


def sf_transfer_all(df: pd.DataFrame, min_shared: int = 2, cap: int = 60000) -> pd.DataFrame:
    """SF-level transfer error at a permissive shared-metal threshold.

    With only 2 shared metals a correlation is meaningless but the log SF of
    that one pair is perfectly well defined, so this covers far more of the
    cohort than the >= 4 metal analysis.
    """
    C = curve_matrix(df)
    ext = df["extractant"].to_numpy()
    ct = df["chemotype"].to_numpy()
    pub = df["publication_id"].astype(str).to_numpy()
    N = len(df)

    def sfmae(i, j):
        ok = np.isfinite(C[i]) & np.isfinite(C[j])
        k = int(ok.sum())
        if k < min_shared:
            return None
        d = C[i][ok] - C[j][ok]
        iu = np.triu_indices(k, 1)
        return float(np.abs(d[:, None] - d[None, :])[iu].mean()), k

    recs = []
    for i in range(N):
        for j in range(i + 1, N):
            same_ext = ext[i] == ext[j]
            same_ct = ct[i] == ct[j]
            if not same_ext and not same_ct:
                continue  # random level is sampled separately
            r = sfmae(i, j)
            if r is None:
                continue
            recs.append(dict(sf_mae=r[0], n_shared=r[1],
                             level="within extractant" if same_ext
                             else "different extractant, same chemotype",
                             same_pub=int(pub[i] == pub[j])))
    seen, tries, n_rand = set(), 0, 0
    while n_rand < 20000 and tries < 400000:
        tries += 1
        i, j = RNG.integers(0, N, 2)
        if i == j:
            continue
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        r = sfmae(i, j)
        if r is None:
            continue
        n_rand += 1
        recs.append(dict(sf_mae=r[0], n_shared=r[1], level="random",
                         same_pub=int(pub[i] == pub[j])))
    P = pd.DataFrame(recs)
    rows = []
    for lab, sub in [
        ("within extractant", P[P.level == "within extractant"]),
        ("within extractant, same publication",
         P[(P.level == "within extractant") & (P.same_pub == 1)]),
        ("within extractant, different publication",
         P[(P.level == "within extractant") & (P.same_pub == 0)]),
        ("different extractant, same chemotype",
         P[P.level == "different extractant, same chemotype"]),
        ("random cell pairs", P[P.level == "random"]),
    ]:
        rows.append(dict(level=lab, n_pairs=len(sub),
                         mean_sf_mae=sub.sf_mae.mean(), median_sf_mae=sub.sf_mae.median(),
                         p90_sf_mae=sub.sf_mae.quantile(0.90),
                         mean_n_shared=sub.n_shared.mean()))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 4. quadratic shape fits
# --------------------------------------------------------------------------- #
def quad_fits(df: pd.DataFrame) -> pd.DataFrame:
    C = curve_matrix(df)
    recs = []
    for i in range(len(df)):
        y = C[i]
        ok = np.isfinite(y)
        if ok.sum() < MIN_PTS_QUAD:
            continue
        x = RAD_Z[ok]
        yy = y[ok]
        X = np.column_stack([np.ones(ok.sum()), x, x ** 2])
        beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
        resid = yy - X @ beta
        ss_res = float((resid ** 2).sum())
        ss_tot = float(((yy - yy.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
        # linear-only fit, for the share of shape the tilt alone explains
        b1, *_ = np.linalg.lstsq(X[:, :2], yy, rcond=None)
        r1 = yy - X[:, :2] @ b1
        r2_lin = 1.0 - float((r1 ** 2).sum()) / ss_tot if ss_tot > 1e-12 else np.nan
        recs.append(dict(
            cell_id=df["cell_id"].iat[i], extractant=df["extractant"].iat[i],
            extractant_name=df["extractant_name"].iat[i],
            chemotype=df["chemotype"].iat[i], publication_id=df["publication_id"].iat[i],
            n_metals=int(ok.sum()), hno3_M=df["hno3_M"].iat[i],
            intercept=beta[0], amplitude_linear=beta[1], curvature_quadratic=beta[2],
            r2_quad=r2, r2_linear_only=r2_lin,
            curve_sd=float(yy.std(ddof=0)),
        ))
    return pd.DataFrame(recs)


def variance_decomposition(Q: pd.DataFrame, col: str, min_cells: int = 2) -> dict:
    """One-way random-effects split of `col` into between-extractant and within."""
    g = Q.groupby("extractant")[col]
    sizes = g.size()
    keep = sizes[sizes >= min_cells].index
    sub = Q[Q["extractant"].isin(keep)]
    if sub["extractant"].nunique() < 2:
        return {}
    grand = sub[col].mean()
    gm = sub.groupby("extractant")[col].agg(["mean", "size"])
    k = len(gm)
    n_tot = int(gm["size"].sum())
    ss_between = float((gm["size"] * (gm["mean"] - grand) ** 2).sum())
    ss_within = float(sub.groupby("extractant")[col].apply(
        lambda s: ((s - s.mean()) ** 2).sum()).sum())
    df_b, df_w = k - 1, n_tot - k
    ms_b = ss_between / df_b
    ms_w = ss_within / df_w if df_w > 0 else np.nan
    # unbiased group size for unbalanced designs
    n0 = (n_tot - (gm["size"] ** 2).sum() / n_tot) / (k - 1)
    var_a = max((ms_b - ms_w) / n0, 0.0) if np.isfinite(ms_w) else np.nan
    icc = var_a / (var_a + ms_w) if np.isfinite(ms_w) and (var_a + ms_w) > 0 else np.nan
    return dict(
        column=col, n_extractants=k, n_cells=n_tot,
        total_sd=float(sub[col].std(ddof=0)),
        ss_between=ss_between, ss_within=ss_within,
        frac_ss_between=ss_between / (ss_between + ss_within) if (ss_between + ss_within) > 0 else np.nan,
        ms_between=ms_b, ms_within=ms_w,
        var_between_component=var_a, var_within_component=ms_w,
        sd_between=np.sqrt(var_a) if np.isfinite(var_a) else np.nan,
        sd_within=np.sqrt(ms_w) if np.isfinite(ms_w) else np.nan,
        icc=icc,
    )


# --------------------------------------------------------------------------- #
def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    df = load_cells()
    print(f"[load] {len(df)} cells, {df.extractant.nunique()} extractants, "
          f"{int(df.n_obs.sum())} observed (cell, metal) entries")
    print(f"[acid] colour column = {ACID_CONC_COL} restricted to {HNO3_FLAG_COL}==1: "
          f"{int((df[HNO3_FLAG_COL]==1).sum())}/{len(df)} cells use HNO3")
    nm = df["n_obs"]
    cov = pd.DataFrame({"n_metals": np.arange(2, 15)})
    cov["n_cells"] = [int((nm == k).sum()) for k in cov.n_metals]
    cov["cells_at_least"] = [int((nm >= k).sum()) for k in cov.n_metals]
    cov["entries_at_least"] = [int(nm[nm >= k].sum()) for k in cov.n_metals]
    cov.to_csv(os.path.join(OUT, "d1_metal_coverage_per_cell.csv"), index=False)
    print(f"[coverage] metals per cell: median {nm.median():.0f}, mean {nm.mean():.2f}; "
          f"cells with >= 4 metals = {int((nm>=4).sum())}/{len(df)} "
          f"({100*(nm>=4).mean():.1f}%); with >= 5 = {int((nm>=5).sum())}; "
          f"exactly 2 = {int((nm==2).sum())}")

    # 1 -------------------------------------------------------------------- #
    rank = rank_extractants(df)
    rank.to_csv(os.path.join(OUT, "d1_extractant_ranking.csv"), index=False)
    print("\n[rank] top 12 by cells:")
    print(rank.head(12)[["rank_by_cells", "extractant_name", "n_cells", "n_entries",
                         "n_publications", "n_condition_keys", "median_n_metals"]].to_string(index=False))
    print(f"[rank] cells covered by top 12 = {int(rank.head(12).n_cells.sum())}/{len(df)} "
          f"({100*rank.head(12).n_cells.sum()/len(df):.1f}%); "
          f"extractants with 1 cell = {int((rank.n_cells==1).sum())}/{len(rank)}")

    # 2 -------------------------------------------------------------------- #
    top = rank.head(12)
    for _, r in top.iterrows():
        g = df[df["extractant"] == r["extractant"]].copy()
        safe = "".join(ch if ch.isalnum() else "_" for ch in str(r["extractant_name"]))[:34]
        fname = f"d1_curve_{int(r['rank_by_cells']):02d}_{safe}.png"
        plot_extractant(g, str(r["extractant_name"]), str(r["extractant"]), fname)
        print(f"[fig] {fname}")
    plot_grid(df, top, "d1_atlas_grid_top12.png")
    print("[fig] d1_atlas_grid_top12.png")

    # 3 -------------------------------------------------------------------- #
    per_ext, within_pairs = within_extractant_table(df)
    per_ext.to_csv(os.path.join(OUT, "d1_within_extractant_reproducibility.csv"), index=False)
    within_pairs.to_csv(os.path.join(OUT, "d1_within_extractant_pairs.csv"), index=False)
    cross = cross_extractant_within_chemotype(df)
    rnd = random_pairs(df)

    levels = [
        summarise(within_pairs, "within extractant (all pairs)"),
        summarise(within_pairs[within_pairs.same_pub == 1], "within extractant, same publication"),
        summarise(within_pairs[within_pairs.same_pub == 0], "within extractant, different publication"),
        summarise(cross, "different extractant, same chemotype"),
        summarise(rnd, "random cell pairs"),
    ]
    lev = pd.DataFrame(levels)
    lev.to_csv(os.path.join(OUT, "d1_similarity_levels.csv"), index=False)
    print("\n[levels] mean pairwise similarity of centred curves "
          f"(>= {MIN_SHARED} shared metals, MAE after re-centring on the shared metals):")
    print(lev[["level", "n_pairs", "mean_pearson", "mean_spearman", "mean_mae",
               "median_mae", "mean_sf_mae", "median_sf_mae",
               "frac_pearson_gt_0p8", "frac_pearson_lt_0"]].to_string(index=False))

    print("\n[per-extractant reproducibility] extractants with >= 3 cells "
          f"= {len(per_ext)}")
    print(per_ext[["extractant_name", "n_cells", "n_pairs", "mean_pearson",
                   "mean_mae", "mean_mae_samepub", "mean_mae_diffpub",
                   "mean_sf_mae"]].to_string(index=False))

    sf2 = sf_transfer_all(df, min_shared=2)
    sf2.to_csv(os.path.join(OUT, "d1_sf_transfer_minshared2.csv"), index=False)
    print("\n[sf transfer, >= 2 shared metals - covers the whole cohort] "
          "MAE in log SF when one cell is used to predict another:")
    print(sf2.to_string(index=False))

    # does the curve move with the nitric-acid concentration?
    W = within_pairs.dropna(subset=["dlog10_hno3"])
    if len(W) > 50:
        rho = W["dlog10_hno3"].corr(W["sf_mae"], method="spearman")
        bins = pd.cut(W["dlog10_hno3"], [-0.001, 1e-9, 0.2, 0.5, 1.0, 10.0],
                      labels=["same [HNO3]", "<0.2 dex", "0.2-0.5 dex", "0.5-1 dex", ">1 dex"])
        acid_tab = W.groupby(bins, observed=True).agg(
            n_pairs=("sf_mae", "size"), mean_sf_mae=("sf_mae", "mean"),
            mean_curve_mae=("mae_shared_centred", "mean"),
            mean_pearson=("pearson", "mean")).reset_index()
        acid_tab.columns = ["hno3_gap", "n_pairs", "mean_sf_mae", "mean_curve_mae", "mean_pearson"]
        acid_tab.to_csv(os.path.join(OUT, "d1_acid_gap_vs_curve_disagreement.csv"), index=False)
        print(f"\n[acid] within-extractant pairs with both [HNO3] known = {len(W)}; "
              f"Spearman(|dlog10 HNO3|, SF MAE) = {rho:.3f}")
        print(acid_tab.to_string(index=False))

    # 4 -------------------------------------------------------------------- #
    Q = quad_fits(df)
    Q.to_csv(os.path.join(OUT, "d1_cell_quadratic_fits.csv"), index=False)
    print(f"\n[quad] fitted {len(Q)} of {len(df)} cells (>= {MIN_PTS_QUAD} observed metals)")
    print(f"[quad] R2 quadratic: mean {Q.r2_quad.mean():.3f} median {Q.r2_quad.median():.3f}; "
          f"linear-only R2 mean {Q.r2_linear_only.mean():.3f}")
    print(f"[quad] amplitude (coef on standardised Shannon CN8 radius): "
          f"mean {Q.amplitude_linear.mean():+.3f} sd {Q.amplitude_linear.std():.3f} "
          f"range [{Q.amplitude_linear.min():+.3f}, {Q.amplitude_linear.max():+.3f}]; "
          f"negative (heavy-selective) in {100*(Q.amplitude_linear<0).mean():.1f}% of cells")
    print(f"[quad] curvature: mean {Q.curvature_quadratic.mean():+.3f} "
          f"sd {Q.curvature_quadratic.std():.3f}")

    # per-extractant amplitude summary for the top 12
    amp_rows = []
    for _, r in top.iterrows():
        s = Q[Q.extractant == r["extractant"]]
        if len(s) == 0:
            continue
        amp_rows.append(dict(
            extractant_name=r["extractant_name"], n_cells_fitted=len(s),
            amp_mean=s.amplitude_linear.mean(), amp_sd=s.amplitude_linear.std(),
            amp_min=s.amplitude_linear.min(), amp_max=s.amplitude_linear.max(),
            curv_mean=s.curvature_quadratic.mean(), curv_sd=s.curvature_quadratic.std(),
            r2_quad_mean=s.r2_quad.mean(), r2_lin_mean=s.r2_linear_only.mean(),
            n_publications=s.publication_id.nunique(),
        ))
    amp = pd.DataFrame(amp_rows)
    amp.to_csv(os.path.join(OUT, "d1_top12_amplitude_summary.csv"), index=False)
    print("\n[quad top12]")
    print(amp.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    vd = pd.DataFrame([
        variance_decomposition(Q, "amplitude_linear"),
        variance_decomposition(Q, "curvature_quadratic"),
        variance_decomposition(Q, "curve_sd"),
    ])
    vd.to_csv(os.path.join(OUT, "d1_variance_decomposition.csv"), index=False)
    print("\n[variance decomposition, extractants with >= 2 fitted cells]")
    print(vd[["column", "n_extractants", "n_cells", "total_sd", "frac_ss_between",
              "sd_between", "sd_within", "icc"]].to_string(index=False))

    # amplitude figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    order = amp.sort_values("amp_mean")["extractant_name"].tolist()
    data = [Q[Q.extractant_name == n].amplitude_linear.values for n in order]
    axes[0].boxplot(data, orientation="horizontal",
                    tick_labels=[n[:24] for n in order], widths=0.6)
    for i, d in enumerate(data):
        axes[0].plot(d, np.full(len(d), i + 1) + RNG.normal(0, 0.06, len(d)), "o",
                     ms=3, alpha=0.55, color="tab:blue")
    axes[0].axvline(0, color="0.3", lw=0.8)
    axes[0].set_xlabel("amplitude = coefficient on standardised Shannon CN8 radius\n"
                       "(negative = heavier Ln extracted better)")
    axes[0].set_title("Amplitude per cell, 12 top extractants", fontsize=10)
    axes[0].grid(alpha=0.25, axis="x")
    axes[0].tick_params(labelsize=8)

    sc = axes[1].scatter(Q.amplitude_linear, Q.curvature_quadratic, c=Q.r2_quad,
                         cmap="magma", s=14, vmin=0, vmax=1)
    axes[1].axhline(0, color="0.3", lw=0.8)
    axes[1].axvline(0, color="0.3", lw=0.8)
    axes[1].set_xlabel("amplitude (linear coefficient)")
    axes[1].set_ylabel("curvature (quadratic coefficient)")
    axes[1].set_title(f"All {len(Q)} cells with >= {MIN_PTS_QUAD} metals", fontsize=10)
    axes[1].grid(alpha=0.25)
    fig.colorbar(sc, ax=axes[1], label="R2 of the quadratic fit")
    fig.suptitle("D1 - shape coefficients of the centred lanthanide curve", fontsize=11)
    fig.savefig(os.path.join(FIG, "d1_amplitude_curvature.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[fig] d1_amplitude_curvature.png")

    # similarity-level figure
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    sets = [("within ext.\nsame pub", within_pairs[within_pairs.same_pub == 1]),
            ("within ext.\ndiff pub", within_pairs[within_pairs.same_pub == 0]),
            ("same chemotype\ndiff extractant", cross),
            ("random\ncells", rnd)]
    ax[0].boxplot([s["pearson"].dropna().values for _, s in sets],
                  tick_labels=[k for k, _ in sets], widths=0.6, showfliers=False)
    ax[0].set_ylabel("Pearson r between two centred curves")
    ax[0].axhline(0, color="0.3", lw=0.8)
    ax[0].grid(alpha=0.25, axis="y")
    ax[1].boxplot([s["sf_mae"].dropna().values for _, s in sets],
                  tick_labels=[k for k, _ in sets], widths=0.6, showfliers=False)
    ax[1].set_ylabel("MAE in log SF when one cell predicts the other (log units)")
    ax[1].axhline(0.302, color="tab:red", lw=1.0, ls="--",
                  label="median within-replicate sd 0.302")
    ax[1].axhline(0.481, color="tab:green", lw=1.0, ls=":",
                  label="best-arm extractant-macro MAE 0.481")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.25, axis="y")
    for a in ax:
        a.tick_params(labelsize=8)
    fig.suptitle(f"D1 - how alike are two lanthanide curves? (pairs with >= {MIN_SHARED} shared metals)",
                 fontsize=11)
    fig.savefig(os.path.join(FIG, "d1_similarity_levels.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[fig] d1_similarity_levels.png")
    print("\n[done]")


if __name__ == "__main__":
    main()
