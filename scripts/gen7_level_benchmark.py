"""The level sub-problem, isolated: predict a ligand's absolute level from its structure.

This is the experiment the offset/shape decomposition implies and that no previous
generation ran directly.  ``offset_mae`` is an error in a *per-ligand constant*, so
the model that produces it is not fitting 5,248 rows — it is fitting **one number
per ligand**, and the training set has about 120 of them.  Every arm that has ever
been tried on this task hands a 206-column descriptor table and a 2,048-bit
fingerprint to a 120-point regression, which is why the compact 13-column donor
census beats them: p ≪ n instead of p ≫ n.

So this script strips the problem to its bones.

* One row per ligand.  Target: that ligand's mean ``log_D`` over its own rows
  (computed inside the fold, from training rows only).
* Folds: the same Tanimoto-0.7 chemotype hold-out and the same seeds as everything
  else, so a level model benchmarked here can be dropped into the two-stage
  contender without re-tuning.
* Metric: MAE over held-out ligands, plus the macro version with one ECFP cluster
  one vote, plus the hard-chemistry subset.

Two things make the comparison honest and are easy to get wrong:

**Weighted or unweighted target.**  A ligand's mean over its own rows is a biased
estimate of "its level" when its rows are unevenly spread over conditions — a
ligand measured mostly at 0.01 M acid has a lower mean than the same ligand
measured mostly at 3 M.  So the target is also computed in a **condition-adjusted**
form: the ligand's mean residual after a global model of metal and conditions.  Both
are reported, because they answer different questions and only the adjusted one is
about chemistry.

**The null.**  Predicting the global mean level is the floor; predicting the level
of the nearest training ligand by Tanimoto is the interesting null, because if a
model cannot beat 1-nearest-neighbour it has learned nothing a lookup does not
already do.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lanthanide_separation.gen7.harness import (  # noqa: E402
    DEFAULT_SEEDS, build_folds, load_cohort,
)
from lanthanide_separation.gen7.kernels import tanimoto_kernel  # noqa: E402


def ligand_table(cohort, *, adjust_conditions: bool) -> pd.DataFrame:
    """One row per ligand: its level, its features, its cluster labels."""
    frame = cohort.frame
    ligand_feature_blocks = [b for b in ("DONORS", "PHYSCHEM", "LIG2D_EXT", "ECFP",
                                         "EMB_CHEMBERTA", "EMB_MOLFORMER", "EMB_CHEMBERTA_MLM",
                                         "POLYHEDRON", "COMPLEX_PHYS")
                             if b in cohort.blocks]
    columns = sorted({c for b in ligand_feature_blocks for c in cohort.blocks[b]})
    aggregated = frame.groupby("extractant", sort=True).first()[columns]
    meta = frame.groupby("extractant", sort=True).agg(
        level_raw=("log_D", "mean"), n_rows=("log_D", "size"),
        ecfp_cluster=("ecfp_cluster", "first"), tanimoto_cluster=("tanimoto_cluster", "first"),
        level_sd=("log_D", "std"))
    table = meta.join(aggregated)
    table.index.name = "extractant"
    return table.reset_index()


def condition_adjusted_levels(cohort, train_index: np.ndarray) -> pd.Series:
    """Ligand level after removing a global model of metal + conditions.

    Fitted on the fold's training rows only; the adjustment for a held-out ligand
    uses the *global* model, never that ligand's own rows.
    """
    from sklearn.ensemble import ExtraTreesRegressor

    frame = cohort.frame
    columns = cohort.block_columns(("METAL", "COND", "MASSACTION"))
    x = frame[list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    keep = ~np.all(np.isnan(x[train_index]), axis=0)
    x = x[:, keep]
    median = np.nan_to_num(np.nanmedian(x[train_index], axis=0))
    x = np.where(np.isfinite(x), x, median)
    y = frame["log_D"].to_numpy(float)
    model = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=4,
                                random_state=20260819, n_jobs=-1)
    model.fit(x[train_index], y[train_index])
    residual = y - model.predict(x)
    return pd.Series(residual, index=frame.index).groupby(
        frame["extractant"].astype(str)).mean()


# --------------------------------------------------------------------------- #
# Level models
# --------------------------------------------------------------------------- #

def _clean(train_x, test_x):
    keep = ~np.all(np.isnan(train_x), axis=0)
    train_x, test_x = train_x[:, keep], test_x[:, keep]
    median = np.nan_to_num(np.nanmedian(train_x, axis=0))
    train_x = np.where(np.isfinite(train_x), train_x, median)
    test_x = np.where(np.isfinite(test_x), test_x, median)
    keep = train_x.std(0) > 1e-9
    if keep.any():
        train_x, test_x = train_x[:, keep], test_x[:, keep]
    return train_x, test_x


def make_models():
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
    from sklearn.linear_model import BayesianRidge, ElasticNetCV, RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR

    def ridge(seed):
        return make_pipeline(StandardScaler(),
                             RidgeCV(alphas=np.logspace(-2, 4, 25)))

    def bayes(seed):
        return make_pipeline(StandardScaler(), BayesianRidge())

    def enet(seed):
        # A 4-ratio x 100-alpha x 5-fold search over 2,048 fingerprint columns with
        # ~120 samples took longer than every other model in this script combined and
        # never won a row; the grid is cut to what the sample size can support.
        return make_pipeline(StandardScaler(),
                             ElasticNetCV(l1_ratio=[.5, 1.0], n_alphas=20, cv=3,
                                          max_iter=5000, tol=1e-3, random_state=seed))

    def pls(seed):
        return make_pipeline(StandardScaler(), PLSRegression(n_components=3))

    def svr(seed):
        return make_pipeline(StandardScaler(), SVR(C=3.0, epsilon=0.2, gamma="scale"))

    def trees(seed):
        return ExtraTreesRegressor(n_estimators=400, min_samples_leaf=1,
                                   max_features=0.5, random_state=seed, n_jobs=2)

    def gb(seed):
        return GradientBoostingRegressor(n_estimators=300, learning_rate=0.03,
                                         max_depth=2, subsample=0.8, random_state=seed)

    def tabpfn_(seed):
        # A tabular foundation model doing in-context learning: pretrained on millions
        # of synthetic tabular tasks, no gradient descent at fit time.  Its published
        # sweet spot is n < 10,000 and p < 500 — exactly the shape of the level
        # sub-problem (~120 ligands) — and it is the one method in the review whose
        # stated regime matches ours without reinterpretation.
        #
        # Pinned to v2: the v3 weights are gated behind a HuggingFace terms
        # acceptance that cannot be completed non-interactively, so v2 (ungated) is
        # what is actually testable here.  ~21 s per fit on CPU, which is why it is
        # reached through --models rather than run over every feature set.
        from tabpfn import TabPFNRegressor
        return TabPFNRegressor(device="cpu", random_state=seed)

    models = {"ridge": ridge, "bayesridge": bayes, "elasticnet": enet, "pls3": pls,
              "svr_rbf": svr, "extratrees": trees, "gbdt_depth2": gb}
    try:
        import tabpfn  # noqa: F401
        models["tabpfn"] = tabpfn_
    except Exception:
        pass
    return models


FEATURE_SETS = {
    "donors": ("DONORS",),
    "donors_physchem": ("DONORS", "PHYSCHEM"),
    "physchem": ("PHYSCHEM",),
    "lig2d": ("LIG2D_EXT",),
    "donors_physchem_lig2d": ("DONORS", "PHYSCHEM", "LIG2D_EXT"),
    "ecfp": ("ECFP",),
    "molformer": ("EMB_MOLFORMER",),
    "chemberta": ("EMB_CHEMBERTA",),
    "molformer_donors": ("EMB_MOLFORMER", "DONORS", "PHYSCHEM"),
    "polyhedron3d": ("POLYHEDRON",),
    "donors_3d": ("DONORS", "PHYSCHEM", "POLYHEDRON", "COMPLEX_PHYS"),
}


def ligand_folds(cohort, seed: int, n_splits: int = 5):
    """Hold out random *ligands* instead of whole chemotypes.

    This is deliberately the **weaker** protocol, and it is run beside the strict one
    for exactly one purpose: to find out whether a ligand's level is a function of
    its structure at all.  Under the chemotype hold-out a test ligand has no close
    training relative by construction, so "the level is unpredictable" and "the level
    is unpredictable *by extrapolation*" look identical.  Split by ligand and a test
    ligand's near-twins stay in training; if the level becomes predictable, the level
    is chemical and gen7's problem is extrapolation distance (World A).  If it does
    not, a large part of what we call "the ligand's level" is not a property of the
    ligand (World E).  It is never used for a headline number.
    """
    from lanthanide_separation.gen6.cohorts import seeded_group_kfold
    names = cohort.frame["extractant"].astype(str).to_numpy()
    return list(seeded_group_kfold(names, n_splits, seed))


def evaluate(cohort, *, seeds, adjust_conditions: bool, pca_components: int,
             split: str = "chemotype", only_models: Sequence[str] | None = None,
             only_features: Sequence[str] | None = None) -> pd.DataFrame:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    table = ligand_table(cohort, adjust_conditions=adjust_conditions)
    names = table["extractant"].astype(str).to_numpy()
    position = {n: i for i, n in enumerate(names)}
    ecfp_columns = list(cohort.blocks.get("ECFP", ()))
    bits = table[ecfp_columns].to_numpy(float) if ecfp_columns else None
    similarity = tanimoto_kernel(bits, bits) if bits is not None else None
    models = make_models()
    if only_models:
        models = {k: v for k, v in models.items() if k in set(only_models)}
        if not models:
            raise SystemExit(f"no models matched {only_models}; have {sorted(make_models())}")
    feature_sets = (FEATURE_SETS if not only_features
                    else {k: v for k, v in FEATURE_SETS.items() if k in set(only_features)})
    records: list[dict] = []

    for seed in seeds:
        if split == "ligand":
            plan = [(k, tr, te) for k, (tr, te) in enumerate(ligand_folds(cohort, seed))]
        else:
            plan = [(f.fold, f.train_index, f.test_index) for f in build_folds(cohort.frame, seed)]
        for fold_id, train_index, test_index in plan:
            fold = type("F", (), {"fold": fold_id, "train_index": train_index,
                                  "test_index": test_index})()
            train_names = sorted(set(cohort.frame["extractant"].astype(str)
                                     .to_numpy()[fold.train_index]))
            test_names = sorted(set(cohort.frame["extractant"].astype(str)
                                    .to_numpy()[fold.test_index]))
            train_rows = np.array([position[n] for n in train_names if n in position])
            test_rows = np.array([position[n] for n in test_names if n in position])
            if len(train_rows) < 20 or len(test_rows) == 0:
                continue

            if adjust_conditions:
                levels = condition_adjusted_levels(cohort, fold.train_index)
                target = table["extractant"].map(levels).to_numpy(float)
            else:
                # the training-side ligand mean; a held-out ligand's own mean is the
                # quantity being predicted, so it is only ever used as the truth
                target = table["level_raw"].to_numpy(float)
            y_train, y_test = target[train_rows], target[test_rows]

            # nearest-neighbour null
            if similarity is not None:
                nn_prediction = np.empty(len(test_rows))
                nn_similarity = np.empty(len(test_rows))
                for k, row in enumerate(test_rows):
                    sims = similarity[row, train_rows]
                    best = int(np.argmax(sims))
                    nn_prediction[k] = y_train[best]
                    nn_similarity[k] = float(sims[best])
            else:
                nn_prediction = np.full(len(test_rows), y_train.mean())
                nn_similarity = np.zeros(len(test_rows))

            base = {"seed": int(seed), "fold": fold.fold, "n_train": len(train_rows),
                    "n_test": len(test_rows)}
            for label, prediction in (("NULL_mean", np.full(len(test_rows), y_train.mean())),
                                      ("NULL_median", np.full(len(test_rows), np.median(y_train))),
                                      ("NULL_nn_tanimoto", nn_prediction)):
                records.append({**base, "features": "-", "model": label,
                                "mae": float(np.abs(prediction - y_test).mean()),
                                "predictions": prediction.tolist(),
                                "truth": y_test.tolist(),
                                "test_names": [names[i] for i in test_rows],
                                "nn_similarity": nn_similarity.tolist()})

            for feature_name, blocks in feature_sets.items():
                available = [b for b in blocks if b in cohort.blocks]
                if not available:
                    continue
                columns = sorted({c for b in available for c in cohort.blocks[b]})
                x = table[columns].to_numpy(float)
                x_train, x_test = _clean(x[train_rows], x[test_rows])
                if x_train.shape[1] == 0:
                    continue
                variants = [("", x_train, x_test)]
                if x_train.shape[1] > pca_components:
                    scaler = StandardScaler().fit(x_train)
                    n = min(pca_components, x_train.shape[1], len(train_rows) - 1)
                    pca = PCA(n_components=n, random_state=seed)
                    variants.append((f"_pca{n}",
                                     pca.fit_transform(scaler.transform(x_train)),
                                     pca.transform(scaler.transform(x_test))))
                for suffix, xt, xs in variants:
                    for model_name, factory in models.items():
                        try:
                            model = factory(seed)
                            model.fit(xt, y_train)
                            prediction = np.asarray(model.predict(xs), dtype=float).ravel()
                        except Exception:
                            continue
                        span = float(y_train.max() - y_train.min())
                        prediction = np.clip(prediction, y_train.min() - 0.5 * span,
                                             y_train.max() + 0.5 * span)
                        records.append({
                            **base, "features": feature_name + suffix, "model": model_name,
                            "mae": float(np.abs(prediction - y_test).mean()),
                            "predictions": prediction.tolist(), "truth": y_test.tolist(),
                            "test_names": [names[i] for i in test_rows],
                            "nn_similarity": nn_similarity.tolist()})
    return pd.DataFrame(records)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--pca-components", type=int, default=16)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--features", nargs="*", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--split", choices=("chemotype", "ligand"), default="chemotype",
                        help="chemotype = the strict protocol; ligand = the diagnostic one")
    parser.add_argument("--adjust-conditions", action="store_true",
                        help="target = ligand mean residual after a global metal+conditions model")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "runs" / "gen7_architecture" / "level_benchmark")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    cohort = load_cohort()
    seeds = DEFAULT_SEEDS[: max(1, min(5, args.seeds))]
    detail = evaluate(cohort, seeds=seeds, adjust_conditions=args.adjust_conditions,
                      pca_components=args.pca_components, split=args.split,
                      only_models=args.models, only_features=args.features)
    if detail.empty:
        raise SystemExit("no results")

    # ligand-level pooling: every held-out ligand gets one vote, across folds and seeds
    rows = []
    for (features, model), block in detail.groupby(["features", "model"]):
        errors, sims = [], []
        for _, record in block.iterrows():
            errors.extend(np.abs(np.asarray(record["predictions"]) - np.asarray(record["truth"])))
            sims.extend(record["nn_similarity"])
        errors, sims = np.asarray(errors), np.asarray(sims)
        hard = sims < 0.4
        rows.append({
            "features": features, "model": model,
            "level_mae": float(errors.mean()),
            "level_mae_sd_over_folds": float(block["mae"].std()),
            "level_mae_nn_lt_0_4": float(errors[hard].mean()) if hard.any() else np.nan,
            "n_hard": int(hard.sum()), "n_ligand_evals": int(len(errors)),
        })
    board = pd.DataFrame(rows).sort_values("level_mae", ignore_index=True)
    tag = ("adjusted" if args.adjust_conditions else "raw") + (
        "_ligandsplit" if args.split == "ligand" else "") + (
        f"_{args.tag}" if args.tag else "")
    board.to_csv(args.out / f"level_leaderboard_{tag}.csv", index=False)
    detail.drop(columns=["predictions", "truth", "test_names", "nn_similarity"]).to_csv(
        args.out / f"level_detail_{tag}.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"\n=== level target: {tag} | seeds {list(seeds)} ===")
    print(board.head(30).to_string(index=False))
    print("\nnulls:")
    print(board[board["model"].str.startswith("NULL")].to_string(index=False))
    (args.out / f"summary_{tag}.json").write_text(json.dumps({
        "seeds": [int(s) for s in seeds], "adjust_conditions": bool(args.adjust_conditions),
        "n_ligands": int(cohort.frame.extractant.nunique()),
    }, indent=2))
    print(f"\nartifacts -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
