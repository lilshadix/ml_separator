"""Is the zero-shot level unpredictable, or are we using the wrong neighbourhood? (§12)

gen7's decisive negative result was that on the **condition-adjusted** ligand level
— the ligand's mean residual after a global metal + conditions model, i.e. the part
that is actually chemistry — *no model of any kind beat a 1-nearest-neighbour
Tanimoto lookup*, and the best captured 6 % of the variance.  That result was
obtained with ECFP similarity throughout.

The sulfur-donor failure suggests why that might be the wrong test: two molecules
can share almost every ECFP bit and extract by completely different mechanisms if
one swaps an amide oxygen for a thiophosphoryl sulfur.  ECFP distance is a distance
between *substructure inventories*; extraction is governed by *donor-set hardness,
denticity and charge*.  This script therefore rebuilds the same benchmark with a
36-column mechanistic distance and asks whether the neighbourhood, rather than the
learner, was the problem.

The falsifiable question, stated before the run: **if mechanistic 1-NN beats
Tanimoto 1-NN on the condition-adjusted level, the level is not unpredictable — it
was mis-neighboured.  If it does not, gen7's ceiling stands.**
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lanthanide_separation.gen6.cohorts import seeded_group_kfold  # noqa: E402
from lanthanide_separation.gen8.mechanism import (  # noqa: E402
    MECHANISM_COLUMNS, mechanism_features,
)
from lanthanide_separation.gen8.report import md_table  # noqa: E402

COHORT = REPO_ROOT / "runs" / "gen7_architecture" / "cache" / "cohort.parquet"
BLOCKS = REPO_ROOT / "runs" / "gen7_architecture" / "cache" / "cohort_blocks.json"
OUT = REPO_ROOT / "runs" / "gen8_architecture" / "mechanism_similarity"
SEEDS = (104729, 130363, 155921, 196613, 262147)


def tanimoto_matrix(bits: np.ndarray) -> np.ndarray:
    x = bits.astype(int)
    inter = x @ x.T
    count = x.sum(axis=1)
    denom = count[:, None] + count[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, inter / denom, 0.0)


def standardise(matrix: np.ndarray) -> np.ndarray:
    """Column-wise z-scores with a NaN-safe fallback, so a distance is well defined."""
    out = matrix.astype(float).copy()
    median = np.nanmedian(out, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    out = np.where(np.isfinite(out), out, median)
    centre, scale = out.mean(axis=0), out.std(axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    return (out - centre) / scale


def condition_adjusted_levels(frame: pd.DataFrame, columns, train_index: np.ndarray,
                              seed: int) -> pd.Series:
    """gen7's definition, reproduced: mean residual after a global metal+conditions model."""
    from sklearn.ensemble import ExtraTreesRegressor

    x = frame[list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    keep = ~np.all(np.isnan(x[train_index]), axis=0)
    x = x[:, keep]
    median = np.nan_to_num(np.nanmedian(x[train_index], axis=0))
    x = np.where(np.isfinite(x), x, median)
    y = frame["log_D"].to_numpy(float)
    model = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=4, random_state=seed, n_jobs=-1)
    model.fit(x[train_index], y[train_index])
    residual = y - model.predict(x)
    return pd.Series(residual, index=frame.index).groupby(frame["extractant"].astype(str)).mean()


def knn_predict(distance: np.ndarray, train: np.ndarray, test: np.ndarray,
                values: np.ndarray, k: int) -> np.ndarray:
    sub = distance[np.ix_(test, train)]
    order = np.argsort(sub, axis=1)[:, :k]
    return values[train][order].mean(axis=1)


def kernel_predict(distance: np.ndarray, train: np.ndarray, test: np.ndarray,
                   values: np.ndarray, bandwidth: float) -> np.ndarray:
    sub = distance[np.ix_(test, train)]
    weights = np.exp(-0.5 * (sub / max(bandwidth, 1e-9)) ** 2)
    total = weights.sum(axis=1, keepdims=True)
    total = np.where(total > 1e-12, total, 1.0)
    return (weights @ values[train]) / total.ravel()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=COHORT)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(args.cohort)
    blocks = {k: tuple(v) for k, v in json.loads(BLOCKS.read_text()).items()}
    adjust_columns = [c for b in ("METAL", "COND", "MASSACTION") for c in blocks[b]]

    ligands = sorted(frame["extractant"].unique())
    index = {name: i for i, name in enumerate(ligands)}
    first = frame.drop_duplicates("extractant").set_index("extractant").loc[ligands]

    mech = mechanism_features(ligands)
    mech_z = standardise(mech.to_numpy(float))
    d_mech = np.linalg.norm(mech_z[:, None, :] - mech_z[None, :, :], axis=2)
    ecfp = first[list(blocks["ECFP"])].to_numpy()
    similarity = tanimoto_matrix(ecfp)
    d_tan = 1.0 - similarity
    donors = standardise(first[list(blocks["DONORS"])].to_numpy(float))
    d_donor = np.linalg.norm(donors[:, None, :] - donors[None, :, :], axis=2)

    from scipy.stats import spearmanr
    triu = np.triu_indices(len(ligands), k=1)
    agreement = {
        "spearman_mech_vs_tanimoto": float(spearmanr(d_mech[triu], d_tan[triu]).statistic),
        "spearman_mech_vs_donor": float(spearmanr(d_mech[triu], d_donor[triu]).statistic),
        "spearman_donor_vs_tanimoto": float(spearmanr(d_donor[triu], d_tan[triu]).statistic),
    }
    print(json.dumps(agreement, indent=2))

    distances = {"tanimoto": d_tan, "mechanism": d_mech, "donors": d_donor}
    groups = frame["tanimoto_cluster"].astype(str).to_numpy()
    ligand_group = first["tanimoto_cluster"].astype(str).to_numpy()
    ligand_ecfp_cluster = first["ecfp_cluster"].astype(str).to_numpy()

    records: list[dict] = []
    per_ligand: list[pd.DataFrame] = []
    for seed in SEEDS[: args.seeds]:
        for fold, (train_rows, test_rows) in enumerate(seeded_group_kfold(groups, 5, seed)):
            levels = condition_adjusted_levels(frame, adjust_columns, train_rows,
                                               42 + fold * 1009 + 9_999_991)
            values = np.array([levels.get(name, np.nan) for name in ligands])
            held = sorted(set(groups[test_rows]))
            test = np.array([i for i, g in enumerate(ligand_group) if g in held], dtype=int)
            train = np.array([i for i, g in enumerate(ligand_group) if g not in held], dtype=int)
            if test.size == 0 or train.size < 5:
                continue
            truth = values[test]
            preds: dict[str, np.ndarray] = {
                "NULL_global_mean": np.full(len(test), float(np.nanmean(values[train])))}
            for name, matrix in distances.items():
                for k in (1, 3, 5):
                    preds[f"{k}NN_{name}"] = knn_predict(matrix, train, test, values, k)
                scale = float(np.median(matrix[np.ix_(train, train)][np.triu_indices(len(train), 1)]))
                preds[f"KERNEL_{name}"] = kernel_predict(matrix, train, test, values, scale / 2)

            from sklearn.ensemble import ExtraTreesRegressor
            from sklearn.linear_model import RidgeCV
            for label, matrix in (("mech", mech_z), ("donors", donors)):
                for model_name, model in (("RIDGE", RidgeCV(alphas=np.logspace(-2, 3, 20))),
                                          ("ET", ExtraTreesRegressor(n_estimators=400,
                                                                     min_samples_leaf=2,
                                                                     random_state=seed, n_jobs=-1))):
                    model.fit(matrix[train], values[train])
                    preds[f"{model_name}_{label}"] = model.predict(matrix[test])

            for name, prediction in preds.items():
                error = np.abs(prediction - truth)
                per_ligand.append(pd.DataFrame({
                    "seed": int(seed), "fold": fold, "model": name,
                    "extractant": [ligands[i] for i in test],
                    "tanimoto_cluster": ligand_group[test],
                    "ecfp_cluster": ligand_ecfp_cluster[test], "error": error}))
                per_cluster = pd.Series(error).groupby(ligand_ecfp_cluster[test]).mean()
                records.append({
                    "seed": int(seed), "fold": fold, "model": name,
                    "mae": float(np.nanmean(error)), "macro_mae": float(per_cluster.mean()),
                    "n_test": int(len(test)),
                    "r2": float(1 - np.nansum((prediction - truth) ** 2)
                                / max(1e-12, np.nansum((truth - np.nanmean(values[train])) ** 2))),
                })
    detail = pd.DataFrame(records)
    detail.to_csv(args.out / "level_by_neighbourhood_detail.csv", index=False)
    ligand_errors = pd.concat(per_ligand, ignore_index=True)
    ligand_errors.to_parquet(args.out / "level_per_ligand_errors.parquet", index=False)
    from lanthanide_separation.gen8.inference import paired_chemotype_bootstrap
    contrasts = {
        "1NN_tanimoto vs 1NN_mechanism": ("1NN_mechanism", "1NN_tanimoto"),
        "1NN_tanimoto vs global mean": ("NULL_global_mean", "1NN_tanimoto"),
        "ET_mech vs global mean": ("NULL_global_mean", "ET_mech"),
        "ET_mech vs ET_donors": ("ET_donors", "ET_mech"),
        "ET_mech vs 1NN_tanimoto": ("1NN_tanimoto", "ET_mech"),
        "5NN_mechanism vs 5NN_tanimoto": ("5NN_tanimoto", "5NN_mechanism"),
    }
    boot = paired_chemotype_bootstrap(
        ligand_errors.rename(columns={"model": "arm", "seed": "split_seed"}),
        contrasts, arm_column="arm", value_column="error")
    boot.to_csv(args.out / "level_bootstrap.csv", index=False)
    print("\npaired chemotype bootstrap (positive = candidate better):")
    print(boot.to_string(index=False))
    board = detail.groupby("model").agg(
        mae=("mae", "mean"), mae_sd=("mae", "std"), macro_mae=("macro_mae", "mean"),
        r2=("r2", "mean"), n_folds=("fold", "size")).reset_index().sort_values("mae")
    board.to_csv(args.out / "level_by_neighbourhood.csv", index=False)
    pd.set_option("display.width", 200)
    print()
    print(board.to_string(index=False))

    # --- donor-substitution case analysis ---------------------------------- #
    donor_class = pd.DataFrame({
        "extractant": ligands,
        "n_S": mech["mech__n_S_donor"].to_numpy(), "n_P": mech["mech__n_P"].to_numpy(),
        "n_N": mech["mech__n_N_donor"].to_numpy(), "n_O": mech["mech__n_O_donor"].to_numpy(),
        "charge": mech["mech__formal_charge"].to_numpy(),
        "acidic_H": mech["mech__n_acidic_H"].to_numpy(),
        "softness": mech["mech__softness_mean"].to_numpy()})
    substitutions = []
    for i in range(len(ligands)):
        for j in range(i + 1, len(ligands)):
            if similarity[i, j] < 0.55:
                continue
            a, b = donor_class.iloc[i], donor_class.iloc[j]
            kind = None
            if (a["n_S"] == 0) != (b["n_S"] == 0):
                kind = "O/N -> S donor"
            elif (a["n_P"] == 0) != (b["n_P"] == 0):
                kind = "-> P donor"
            elif (a["acidic_H"] > 0) != (b["acidic_H"] > 0):
                kind = "neutral -> acidic"
            elif abs(a["n_N"] - b["n_N"]) >= 1 and abs(a["n_O"] - b["n_O"]) >= 1:
                kind = "O <-> N donor"
            if kind is None:
                continue
            substitutions.append({
                "kind": kind, "a": ligands[i], "b": ligands[j],
                "tanimoto": float(similarity[i, j]),
                "d_tanimoto_rank": float((d_tan[i] < d_tan[i, j]).mean()),
                "d_mech_rank": float((d_mech[i] < d_mech[i, j]).mean()),
                "softness_delta": float(abs(a["softness"] - b["softness"]))})
    subs = pd.DataFrame(substitutions)
    if not subs.empty:
        subs.to_csv(args.out / "donor_substitution_pairs.csv", index=False)
        summary = subs.groupby("kind").agg(
            n_pairs=("kind", "size"), median_tanimoto=("tanimoto", "median"),
            median_tanimoto_rank=("d_tanimoto_rank", "median"),
            median_mech_rank=("d_mech_rank", "median"),
            median_softness_delta=("softness_delta", "median")).reset_index()
        print("\ndonor substitutions among ECFP-similar pairs (Tanimoto >= 0.55):")
        print(summary.to_string(index=False))
    else:
        summary = pd.DataFrame()

    lines = ["# Does mechanism-aware similarity fix the zero-shot level?", "",
             "*Target: the **condition-adjusted** ligand level — the ligand's mean residual "
             "after a global metal + conditions model fitted on the fold's training rows. "
             "This is gen7's decisive negative-result benchmark, rerun with a different "
             "notion of chemical neighbourhood.*", "",
             "## How much do the three distances disagree?", "",
             md_table(pd.DataFrame([agreement]).T.reset_index().rename(
                 columns={"index": "pair", 0: "spearman"})), "",
             "If mechanistic distance were a relabelling of Tanimoto these would be near 1.0.", "",
             "## Predicting the condition-adjusted level", "",
             md_table(board), ""]
    if not summary.empty:
        lines += ["## Donor substitutions hiding inside high ECFP similarity", "",
                  "`d_*_rank` is the fraction of all other ligands that the distance places "
                  "*closer* than this partner: 0.0 means the two are each other's nearest "
                  "neighbour under that distance.", "", md_table(summary), ""]
    (args.out / "mechanism_similarity.md").write_text("\n".join(lines) + "\n")
    (args.out / "agreement.json").write_text(json.dumps(agreement, indent=2))
    print(f"\nartifacts -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
