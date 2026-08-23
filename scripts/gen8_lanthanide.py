"""Is the lanthanide response one smooth curve, or fourteen unrelated categories? (§11)

The 4f series is not a set of labels.  Ionic radius contracts monotonically from La
to Lu, the 4f count runs 0 to 14, and extraction selectivity is a smooth — often
non-monotone, because of the tetrad effect at the half-filled shell — function of
both.  A model that one-hots the metal must learn fourteen independent offsets from
whatever rows each metal happens to have; a model given a radial basis over ionic
radius can share strength between neighbours and still bend sharply where the
chemistry does.

Four representations of the same 14 metals, one learner, one fold plan, identical
rows.  Scored on what the brief asks for — **shape**, not level: within-curve MAE
after removing each metal-series curve's own mean, within-curve Spearman, and
pairwise sign accuracy, all computed on the reconstructed metal-series curves so
"does it get the order of the lanthanides right" is answered directly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lanthanide_separation.gen6.cohorts import seeded_group_kfold  # noqa: E402
from lanthanide_separation.gen8.inference import paired_chemotype_bootstrap  # noqa: E402
from lanthanide_separation.gen8.report import md_table  # noqa: E402

COHORT = REPO_ROOT / "runs" / "gen7_architecture" / "cache" / "cohort.parquet"
BLOCKS = REPO_ROOT / "runs" / "gen7_architecture" / "cache" / "cohort_blocks.json"
MEMBERSHIP = REPO_ROOT / "runs" / "gen8_architecture" / "series" / "curve_membership.parquet"
OUT = REPO_ROOT / "runs" / "gen8_architecture" / "ablations"
SEEDS = (104729, 130363, 155921, 196613, 262147)
#: Radial-basis centres over the lanthanide ionic radius, in Angstrom.
RBF_CENTRES = np.linspace(0.86, 1.03, 8)
RBF_WIDTH = 0.030


def metal_representations(frame: pd.DataFrame, blocks: dict) -> dict:
    """Four encodings of the same fourteen metals."""
    radius = pd.to_numeric(frame["Ionic Radius_metal"], errors="coerce").to_numpy(float)
    radius = np.where(np.isfinite(radius), radius, np.nanmedian(radius))
    index = pd.to_numeric(frame["lanthanide_index"], errors="coerce").to_numpy(float)
    rbf = np.exp(-0.5 * ((radius[:, None] - RBF_CENTRES[None, :]) / RBF_WIDTH) ** 2)

    onehot = pd.get_dummies(frame["metal_symbol"].astype(str), prefix="m").astype(float)
    structured = frame[list(blocks["METAL"])].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    # The 4f count's distance from the half-filled shell at Gd — the tetrad effect —
    # is the one quantity that is *not* monotone in the series, so a tree given only
    # monotone columns can never express it with a single split.
    n_f = np.clip(index - 1, 0, 14)
    tetrad = np.abs(n_f - 7.0)
    shared = np.hstack([structured, rbf, tetrad[:, None], (tetrad ** 2)[:, None]])

    donors = frame[list(blocks["DONORS"])].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    donors = np.nan_to_num(donors)
    if donors.shape[1] > 0:
        scale = donors.std(axis=0)
        donors = donors / np.where(scale > 1e-9, scale, 1.0)
    conditioned = np.hstack([shared,
                             (rbf[:, :, None] * donors[:, None, :]).reshape(len(frame), -1)])
    return {"ONEHOT": onehot.to_numpy(float), "STRUCTURED": structured,
            "SHARED_RBF": shared, "LIGAND_RBF": conditioned}


def curve_shape_metrics(block: pd.DataFrame, membership: pd.DataFrame) -> dict:
    """Shape metrics computed **within each metal-series curve**, then macro-averaged."""
    from scipy.stats import spearmanr

    joined = membership.merge(block, on="row_id")
    shape, rho, sign, n = [], [], [], 0
    for _, curve in joined.groupby("curve_id"):
        y = curve["log_D"].to_numpy(float)
        p = curve["prediction"].to_numpy(float)
        if len(y) < 3 or np.ptp(y) <= 0:
            continue
        shape.append(float(np.abs((p - p.mean()) - (y - y.mean())).mean()))
        if np.ptp(p) > 0:
            rho.append(float(spearmanr(y, p).statistic))
        dy = np.subtract.outer(y, y)[np.triu_indices(len(y), 1)]
        dp = np.subtract.outer(p, p)[np.triu_indices(len(y), 1)]
        orderable = dy != 0
        if orderable.any():
            ties = dp[orderable] == 0
            agree = np.sign(dy[orderable]) == np.sign(dp[orderable])
            sign.append(float((agree & ~ties).sum() + 0.5 * ties.sum()) / float(orderable.sum()))
        n += 1
    return {"curve_shape_mae": float(np.mean(shape)) if shape else np.nan,
            "curve_spearman": float(np.mean(rho)) if rho else np.nan,
            "curve_sign_accuracy": float(np.mean(sign)) if sign else np.nan,
            "n_curves": n}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.impute import SimpleImputer

    frame = pd.read_parquet(COHORT)
    blocks = {k: tuple(v) for k, v in json.loads(BLOCKS.read_text()).items()}
    membership = pd.read_parquet(MEMBERSHIP)
    membership = membership[membership["axis"] == "metal"][["row_id", "curve_id"]]

    base_columns = list(blocks["COND"]) + list(blocks["ECFP"]) + list(blocks["MASSACTION"])
    base = frame[base_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    representations = metal_representations(frame, blocks)
    target = frame["log_D"].to_numpy(float)
    groups = frame["tanimoto_cluster"].astype(str).to_numpy()

    records, per_ligand = [], []
    for seed in SEEDS[: args.seeds]:
        folds = list(seeded_group_kfold(groups, 5, seed))
        for name, matrix in representations.items():
            started = time.time()
            prediction = np.full(len(frame), np.nan)
            for fold, (train_index, test_index) in enumerate(folds):
                x = np.hstack([base, matrix])
                imputer = SimpleImputer(strategy="median", add_indicator=True,
                                        keep_empty_features=True)
                x_train = imputer.fit_transform(x[train_index])
                x_test = imputer.transform(x[test_index])
                model = ExtraTreesRegressor(n_estimators=400, min_samples_leaf=1,
                                            random_state=42 + fold * 1009 + 9_999_991, n_jobs=-1)
                model.fit(x_train, target[train_index])
                prediction[test_index] = model.predict(x_test)
            block = frame[["row_id", "extractant", "tanimoto_cluster", "ecfp_cluster",
                           "log_D"]].copy()
            block["prediction"] = prediction
            metrics = curve_shape_metrics(block, membership)
            residual = prediction - target
            offsets = pd.Series(residual).groupby(frame["extractant"]).transform("mean")
            ligand_frame = pd.DataFrame({
                "extractant": frame["extractant"], "tanimoto_cluster": frame["tanimoto_cluster"],
                "split_seed": seed, "arm": name, "abs_residual": np.abs(residual),
                "shape_residual": np.abs(residual - offsets)})
            per_ligand.append(ligand_frame)
            macro = ligand_frame.groupby("extractant")["abs_residual"].mean().mean()
            shape = ligand_frame.groupby("extractant")["shape_residual"].mean().mean()
            records.append({"seed": int(seed), "arm": name, "macro_mae": float(macro),
                            "shape_mae": float(shape), **metrics,
                            "seconds": time.time() - started, "n_features": int(matrix.shape[1])})
            print(f"  [{name}] seed {seed}: macro {macro:.4f} shape {shape:.4f} "
                  f"curve_shape {metrics['curve_shape_mae']:.4f} "
                  f"sign {metrics['curve_sign_accuracy']:.4f} "
                  f"({time.time() - started:.0f}s)", flush=True)

    detail = pd.DataFrame(records)
    detail.to_csv(args.out / "lanthanide_representation_detail.csv", index=False)
    board = detail.groupby("arm").agg(
        macro_mae=("macro_mae", "mean"), shape_mae=("shape_mae", "mean"),
        curve_shape_mae=("curve_shape_mae", "mean"), curve_spearman=("curve_spearman", "mean"),
        curve_sign_accuracy=("curve_sign_accuracy", "mean"),
        n_features=("n_features", "first"), n_seeds=("seed", "nunique")).reset_index()
    board = board.sort_values("curve_shape_mae")
    board.to_csv(args.out / "lanthanide_representation.csv", index=False)

    ligand_errors = pd.concat(per_ligand, ignore_index=True)
    contrasts = {f"{arm} vs ONEHOT": ("ONEHOT", arm)
                 for arm in ("STRUCTURED", "SHARED_RBF", "LIGAND_RBF")}
    boots = []
    for statistic in ("abs_residual", "shape_residual"):
        frame_boot = paired_chemotype_bootstrap(ligand_errors, contrasts, arm_column="arm",
                                                value_column=statistic)
        frame_boot.insert(0, "statistic", statistic)
        boots.append(frame_boot)
    bootstrap = pd.concat(boots, ignore_index=True)
    bootstrap.to_csv(args.out / "lanthanide_bootstrap.csv", index=False)

    lines = ["# Is the lanthanide response continuous?", "",
             "*Four encodings of the same fourteen metals, one ExtraTrees learner, the frozen "
             "chemotype-blocked fold plan, identical rows. Shape metrics are computed **within "
             "each reconstructed metal-series curve** — one ligand at one set of conditions "
             "across the 4f series — so they measure whether the model gets the *order and "
             "spacing* of the lanthanides right, not whether it gets the ligand's level right.*",
             "", md_table(board), "",
             "## Paired chemotype bootstrap against the one-hot encoding", "",
             "Positive = the structured representation is better.", "",
             md_table(bootstrap), ""]
    (args.out / "lanthanide_response.md").write_text("\n".join(lines) + "\n")
    pd.set_option("display.width", 240)
    print()
    print(board.to_string(index=False))
    print()
    print(bootstrap.to_string(index=False))
    print(f"\nartifacts -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
