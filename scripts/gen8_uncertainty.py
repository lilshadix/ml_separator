"""gen8 section 20: is uncertainty useful?

Builds every uncertainty source that is available at prediction time for the
frozen global model (REC_ecfp_plus_recovered), then scores each one twice and
keeps the two scores strictly apart:

  (a) CALIBRATION  -- does the number rank |residual|, within a ligand and
      across ligands, and does a binned calibration curve come out monotone?
  (b) DECISION USEFULNESS -- does the number rank *candidates to measure*?
      The exact 1-shot MAE of every candidate row is already tabulated in
      runs/gen8_architecture/cross_series/one_shot_candidate_scores.parquet, so
      the realised MAE of "measure the most uncertain row" is computable in
      closed form rather than simulated.

Nothing here uses a held-out label to build a feature or pick a hyperparameter.
Arms whose name starts with ORACLE_ do use held-out labels, on purpose: they are
ceilings, and they are reported as ceilings.

Written to runs/gen8_architecture/uncertainty/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lanthanide_separation.gen8.inference import paired_chemotype_bootstrap  # noqa: E402
from lanthanide_separation.gen8.mechanism import (  # noqa: E402
    mechanism_features,
    mechanism_distance,
)

OUT = ROOT / "runs" / "gen8_architecture" / "uncertainty"
OUT.mkdir(parents=True, exist_ok=True)

FROZEN = "REC_ecfp_plus_recovered"
ENSEMBLE = [
    "TREE_MC_ecfp_massaction",
    "TREE_MC_donors",
    "TREE_MC_lig2d_ext_massaction",
    "REC_ecfp_plus_recovered",
    "GEN7_everything_EXPLORATORY",
]

LOG_COLS = [
    "cond__acid_concentration_M",
    "cond__extractant_concentration_M",
    "cond__metal_concentration_mM",
    "cond__contact_time_min",
]
LIN_COLS = ["cond__temperature_C", "metal_Z"]
# axes a titration/series can move along, in the order we prefer them
GP_AXES = [
    "cond__acid_concentration_M",
    "cond__extractant_concentration_M",
    "metal_Z",
    "cond__temperature_C",
    "cond__metal_concentration_mM",
    "cond__contact_time_min",
]

MIN_ROWS_SPEARMAN = 5


# ---------------------------------------------------------------------------
# feature space
# ---------------------------------------------------------------------------
def condition_matrix(cohort: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Numeric condition design matrix: log concentrations, temperature, Z, one-hots."""
    parts, names = [], []
    for col in LOG_COLS:
        v = cohort[col].to_numpy(dtype=float)
        parts.append(np.log10(np.clip(v, 1e-9, None)))
        names.append("log10__" + col)
    for col in LIN_COLS:
        parts.append(cohort[col].to_numpy(dtype=float))
        names.append(col)
    onehot = [
        c
        for c in cohort.columns
        if c.startswith(("cond__acid__", "cond__diluent__", "cond__additive__"))
    ]
    for col in onehot:
        parts.append(cohort[col].to_numpy(dtype=float))
        names.append(col)
    return np.column_stack(parts), names


def standardise(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """z-score both blocks on the TRAIN block; NaN -> train median first."""
    med = np.nanmedian(train, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    tr = np.where(np.isfinite(train), train, med)
    ot = np.where(np.isfinite(other), other, med)
    mu = tr.mean(axis=0)
    sd = tr.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (tr - mu) / sd, (ot - mu) / sd


def knn_distance(test: np.ndarray, train: np.ndarray, ks=(1, 5)) -> dict[int, np.ndarray]:
    """k-th nearest-neighbour Euclidean distance from each test row to the train block."""
    out: dict[int, np.ndarray] = {}
    kmax = max(ks)
    d = np.empty((test.shape[0], kmax), dtype=float)
    step = 512
    for start in range(0, test.shape[0], step):
        block = test[start : start + step]
        dist = np.sqrt(
            np.maximum(
                (block**2).sum(1)[:, None]
                + (train**2).sum(1)[None, :]
                - 2.0 * block @ train.T,
                0.0,
            )
        )
        k_eff = min(kmax, dist.shape[1])
        part = np.partition(dist, k_eff - 1, axis=1)[:, :k_eff]
        part.sort(axis=1)
        if k_eff < kmax:
            part = np.pad(part, ((0, 0), (0, kmax - k_eff)), mode="edge")
        d[start : start + step] = part
    for k in ks:
        out[k] = d[:, k - 1]
    return out


def _sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.maximum(
        (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2.0 * A @ B.T, 0.0
    )


def rbf_loo_variance(X: np.ndarray, jitter: float = 1e-3) -> np.ndarray:
    """Leave-one-out RBF-GP predictive variance of each design point.

    Unit signal variance, lengthscale from the median pairwise distance of the
    design.  A GP's predictive variance depends only on the inputs, so this uses
    no labels at all -- it measures how well the rest of the ligand's own design
    covers each candidate point.

    Uses the exact LOO identity: with ``Ky = K + jitter I``, the leave-one-out
    predictive variance of the latent function at point i is
    ``1 / [Ky^-1]_ii - jitter``.  One factorisation instead of n of them.
    """
    n = X.shape[0]
    if n < 3:
        return np.full(n, np.nan)
    d2 = _sq_dists(X, X)
    iu = np.triu_indices(n, 1)
    med = np.median(np.sqrt(d2[iu]))
    ell = med if med > 1e-9 else 1.0
    Ky = np.exp(-0.5 * d2 / (ell**2)) + jitter * np.eye(n)
    try:
        inv = np.linalg.inv(Ky)
    except np.linalg.LinAlgError:
        return np.full(n, np.nan)
    diag = np.diag(inv)
    with np.errstate(divide="ignore", invalid="ignore"):
        var = 1.0 / diag - jitter
    return np.where(np.isfinite(var), np.clip(var, 0.0, None), np.nan)


def rbf_train_variance(Xte: np.ndarray, Xtr: np.ndarray, jitter: float = 1e-3,
                       cap: int = 1200, rng_seed: int = 0) -> np.ndarray:
    """RBF-GP predictive variance at ``Xte`` given the training design ``Xtr``."""
    rng = np.random.default_rng(rng_seed)
    if Xtr.shape[0] > cap:
        Xtr = Xtr[rng.choice(Xtr.shape[0], cap, replace=False)]
    m = Xtr.shape[0]
    if m < 3:
        return np.full(Xte.shape[0], np.nan)
    d2 = _sq_dists(Xtr, Xtr)
    iu = np.triu_indices(m, 1)
    med = np.median(np.sqrt(d2[iu]))
    ell = med if med > 1e-9 else 1.0
    K = np.exp(-0.5 * d2 / (ell**2)) + jitter * np.eye(m)
    L = np.linalg.cholesky(K)
    out = np.empty(Xte.shape[0], dtype=float)
    step = 512
    for s in range(0, Xte.shape[0], step):
        B = Xte[s : s + step]
        cross = np.exp(-0.5 * _sq_dists(B, Xtr) / (ell**2))
        v = np.linalg.solve(L, cross.T)
        out[s : s + step] = np.maximum(1.0 - (v**2).sum(0), 0.0)
    return out


# ---------------------------------------------------------------------------
# build the uncertainty table
# ---------------------------------------------------------------------------
def build_uncertainty() -> pd.DataFrame:
    oof = pd.read_parquet(ROOT / "runs/gen7_architecture/finalists/oof_predictions.parquet")
    cohort = pd.read_parquet(ROOT / "runs/gen7_architecture/cache/cohort.parquet")

    base = oof[oof["model"] == FROZEN].copy()
    base["residual"] = base["log_D"] - base["prediction"]
    base["abs_residual"] = base["residual"].abs()

    # --- 2. deep-ensemble spread across the five finalists ------------------
    ens = oof[oof["model"].isin(ENSEMBLE)]
    wide = ens.pivot_table(
        index=["split_seed", "row_id"], columns="model", values="prediction"
    )
    spread = wide.std(axis=1, ddof=1).rename("u_model_spread").reset_index()
    base = base.merge(spread, on=["split_seed", "row_id"], how="left")

    # --- 1. tree-ensemble sd ------------------------------------------------
    # The frozen model does not emit a per-row dispersion (its extra__prediction_sd
    # column is 100% null); only the TREE_MC_* arms and HNN_base_film do.  The
    # nearest architectural sibling of the frozen model is TREE_MC_ecfp_massaction,
    # so its ensemble sd is the tree-variance signal available at the same rows.
    sd_wide = (
        oof[oof["extra__prediction_sd"].notna()]
        .pivot_table(index=["split_seed", "row_id"], columns="model",
                     values="extra__prediction_sd")
        .reset_index()
    )
    tree_models = [c for c in sd_wide.columns if c.startswith("TREE_MC_")]
    sd_wide["u_tree_sd"] = sd_wide["TREE_MC_ecfp_massaction"]
    sd_wide["u_tree_sd_mean"] = sd_wide[tree_models].mean(axis=1)
    sd_wide["u_hnn_sd"] = sd_wide.get("HNN_base_film")
    base = base.merge(
        sd_wide[["split_seed", "row_id", "u_tree_sd", "u_tree_sd_mean", "u_hnn_sd"]],
        on=["split_seed", "row_id"], how="left")

    # self-consistency block: the sibling model's own residual, so that
    # "its sd vs its own error" can be scored without a cross-model mismatch.
    sib = oof[oof["model"] == "TREE_MC_ecfp_massaction"][
        ["split_seed", "row_id", "log_D", "prediction"]].copy()
    sib["sib_residual"] = sib["log_D"] - sib["prediction"]
    base = base.merge(sib[["split_seed", "row_id", "sib_residual"]],
                      on=["split_seed", "row_id"], how="left")

    # --- 4a. ligand-space distance -----------------------------------------
    base["u_lig_tanimoto"] = 1.0 - base["nn_train_tanimoto"]

    # --- 4b. mechanistic ligand distance ------------------------------------
    ligs = sorted(base["extractant"].unique())
    mech = mechanism_features(ligs)
    mech_mat = mech.to_numpy(dtype=float)
    lig_pos = {s: i for i, s in enumerate(ligs)}

    # --- 3/5. condition-space distance and GP variance ----------------------
    cond_raw, cond_names = condition_matrix(cohort)
    row_pos = {r: i for i, r in enumerate(cohort["row_id"])}
    base["_cpos"] = base["row_id"].map(row_pos).astype(int)

    base["u_cond_nn1"] = np.nan
    base["u_cond_nn5"] = np.nan
    base["u_gp_var_train"] = np.nan
    base["u_lig_mech"] = np.nan

    for (seed, fold), test_idx in base.groupby(["split_seed", "fold"]).groups.items():
        te = base.loc[test_idx]
        tr = base[(base["split_seed"] == seed) & (base["fold"] != fold)]
        Xtr_raw = cond_raw[tr["_cpos"].to_numpy()]
        Xte_raw = cond_raw[te["_cpos"].to_numpy()]
        Xtr, Xte = standardise(Xtr_raw, Xte_raw)
        d = knn_distance(Xte, Xtr, ks=(1, 5))
        base.loc[test_idx, "u_cond_nn1"] = d[1]
        base.loc[test_idx, "u_cond_nn5"] = d[5]
        base.loc[test_idx, "u_gp_var_train"] = rbf_train_variance(Xte, Xtr, rng_seed=int(seed) % 9973)

        # mechanistic distance, standardised on the fold's TRAIN ligands
        tr_ligs = sorted(tr["extractant"].unique())
        te_ligs = sorted(te["extractant"].unique())
        A = mech_mat[[lig_pos[s] for s in tr_ligs]]
        mu = np.nanmean(A, axis=0)
        sd = np.nanstd(A, axis=0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        Az = (A - mu) / sd
        Bz = (mech_mat[[lig_pos[s] for s in te_ligs]] - mu) / sd
        D = mechanism_distance(Bz, Az)
        nearest = pd.Series(np.nanmin(D, axis=1), index=te_ligs)
        base.loc[test_idx, "u_lig_mech"] = te["extractant"].map(nearest).to_numpy()

    # --- 5. within-ligand RBF-GP design variance ----------------------------
    gp = np.full(len(base), np.nan)
    pos_of = {v: i for i, v in enumerate(base.index)}
    axis_used: list[dict] = []
    for (seed, lig), idx in base.groupby(["split_seed", "extractant"]).groups.items():
        sub = base.loc[idx]
        cpos = sub["_cpos"].to_numpy()
        cols, used = [], []
        for name in GP_AXES:
            if name == "metal_Z":
                v = cohort["metal_Z"].to_numpy(dtype=float)[cpos]
            else:
                v = cohort[name].to_numpy(dtype=float)[cpos]
                if name in LOG_COLS:
                    v = np.log10(np.clip(v, 1e-9, None))
            v = np.where(np.isfinite(v), v, np.nan)
            if np.isfinite(v).sum() < len(v) or len(np.unique(v[np.isfinite(v)])) < 2:
                continue
            s = np.nanstd(v)
            if s <= 1e-12:
                continue
            cols.append((v - np.nanmean(v)) / s)
            used.append(name)
            if len(cols) == 3:
                break
        axis_used.append({"split_seed": seed, "extractant": lig, "n_axes": len(used),
                          "axes": "|".join(used), "n_rows": len(sub)})
        if not cols:
            continue
        X = np.column_stack(cols)
        v = rbf_loo_variance(X)
        for j, ix in enumerate(idx):
            gp[pos_of[ix]] = v[j]
    base["u_gp_var_design"] = gp
    pd.DataFrame(axis_used).to_csv(OUT / "gp_axes_used.csv", index=False)

    keep = [
        "row_id", "split_seed", "fold", "extractant", "tanimoto_cluster", "ecfp_cluster",
        "series_id", "condition_id", "metal_symbol", "log_D", "prediction",
        "residual", "abs_residual", "nn_train_tanimoto", "sib_residual",
        "u_tree_sd", "u_tree_sd_mean", "u_hnn_sd", "u_model_spread",
        "u_cond_nn1", "u_cond_nn5",
        "u_lig_tanimoto", "u_lig_mech", "u_gp_var_design", "u_gp_var_train",
    ]
    return base[keep]


SOURCES = {
    "u_tree_sd": ("tree-ensemble sd (TREE_MC_ecfp_massaction)", "row"),
    "u_tree_sd_mean": ("tree-ensemble sd, mean of 3 TREE_MC arms", "row"),
    "u_model_spread": ("deep-ensemble spread (5 models)", "row"),
    "u_cond_nn1": ("condition-space 1-NN distance to train", "row"),
    "u_cond_nn5": ("condition-space 5-NN distance to train", "row"),
    "u_gp_var_design": ("RBF-GP LOO variance, ligand design axes", "row"),
    "u_gp_var_train": ("RBF-GP variance vs training design", "row"),
    "u_lig_tanimoto": ("1 - nn_train_tanimoto (ligand)", "ligand"),
    "u_lig_mech": ("mechanistic distance to nearest train ligand", "ligand"),
}


def main() -> None:
    unc = build_uncertainty()
    unc.to_parquet(OUT / "row_uncertainty.parquet", index=False)
    print("uncertainty table", unc.shape)
    print(unc[list(SOURCES)].describe().T[["count", "mean", "std", "min", "max"]])


if __name__ == "__main__":
    main()
