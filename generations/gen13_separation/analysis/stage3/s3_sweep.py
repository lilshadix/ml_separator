"""Search the estimator space for the lanthanide-axis coefficients under the publication-masked
design.  Everything is scored on the frozen fold plan and byte-identical pairs, so the numbers are
directly comparable with metrics/BP_*/leaderboard.csv."""
import sys, time, json
sys.path.insert(0, "gen13_separation")
import numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold
from gen13sep.amplitude_bench import (load_bench, compare, LEAN_BLOCKS, CHEM_BLOCKS, ALL_BLOCKS)
from gen13sep.models import tree_pipeline

DESIGN = sys.argv[1] if len(sys.argv) > 1 else "BP"
SEEDS = None if len(sys.argv) < 3 else [int(x) for x in sys.argv[2].split(",")]

bench = load_bench()
print(f"design {DESIGN}, cells {len(bench.frame)}", flush=True)


def _prep(scale=True):
    steps = [SimpleImputer(strategy="median", keep_empty_features=True)]
    if scale:
        steps.append(StandardScaler())
    return steps


def trees(n=400, mf=0.5, leaf=2):
    def f(Xtr, coef, w, g, Xte, seed):
        m = tree_pipeline(seed, n_estimators=n, max_features=mf, min_samples_leaf=leaf)
        m.fit(Xtr, coef, extratreesregressor__sample_weight=w)
        return m.predict(Xte)
    return f


def ridge():
    def f(Xtr, coef, w, g, Xte, seed):
        m = make_pipeline(*_prep(), RidgeCV(alphas=np.logspace(-2, 4, 25)))
        m.fit(Xtr, coef, ridgecv__sample_weight=w)
        return m.predict(Xte)
    return f


def krr(kernel="rbf"):
    """Kernel ridge with alpha and gamma chosen by a chemotype-grouped inner CV of the fold."""
    def f(Xtr, coef, w, g, Xte, seed):
        pre = make_pipeline(*_prep())
        Ztr = pre.fit_transform(Xtr); Zte = pre.transform(Xte)
        n_groups = len(set(g))
        grid = {"alpha": np.logspace(-3, 2, 12)}
        if kernel == "rbf":
            grid["gamma"] = np.logspace(-4, 0, 9) / max(Ztr.shape[1], 1)
        cv = GroupKFold(n_splits=min(4, max(2, n_groups)))
        gs = GridSearchCV(KernelRidge(kernel=kernel), grid, cv=cv, scoring="neg_mean_absolute_error",
                          n_jobs=-1)
        gs.fit(Ztr, coef, groups=g)
        return gs.best_estimator_.predict(Zte)
    return f


def pls(n_comp=4):
    def f(Xtr, coef, w, g, Xte, seed):
        k = min(n_comp, Xtr.shape[1], max(2, len(Xtr) - 1))
        m = make_pipeline(*_prep(), PLSRegression(n_components=k))
        m.fit(Xtr, coef)
        return np.asarray(m.predict(Xte)).reshape(len(Xte), coef.shape[1])
    return f


def knn(k=5, weights="distance"):
    def f(Xtr, coef, w, g, Xte, seed):
        m = make_pipeline(*_prep(), KNeighborsRegressor(n_neighbors=min(k, len(Xtr)), weights=weights))
        m.fit(Xtr, coef)
        return m.predict(Xte)
    return f


def gbm():
    def f(Xtr, coef, w, g, Xte, seed):
        out = np.zeros((len(Xte), coef.shape[1]))
        for j in range(coef.shape[1]):
            m = HistGradientBoostingRegressor(loss="absolute_error", max_iter=300,
                                              learning_rate=0.06, max_leaf_nodes=8,
                                              l2_regularization=1.0, random_state=seed)
            m.fit(Xtr, coef[:, j], sample_weight=w)
            out[:, j] = m.predict(Xte)
        return out
    return f


def mean_only():
    def f(Xtr, coef, w, g, Xte, seed):
        m = np.average(coef, axis=0, weights=w)
        return np.tile(m, (len(Xte), 1))
    return f


CANDIDATES = {
    "A_trees@all":   trees(),
    "B_trees@lean":  trees(),
    "C_trees@chem":  trees(),
    "D_ridge@lean":  ridge(),
    "E_krr_rbf@lean": krr("rbf"),
    "F_krr_lin@lean": krr("linear"),
    "G_pls4@lean":   pls(4),
    "H_knn5@chem":   knn(5),
    "I_knn3@chem":   knn(3),
    "J_gbm_l1@lean": gbm(),
    "K_meancurve":   mean_only(),
}
OVERRIDES = {
    "A_trees@all": ALL_BLOCKS, "C_trees@chem": CHEM_BLOCKS,
    "H_knn5@chem": CHEM_BLOCKS, "I_knn3@chem": CHEM_BLOCKS,
}

t0 = time.time()
board, table = compare(bench, CANDIDATES, blocks=LEAN_BLOCKS, design=DESIGN, seeds=SEEDS,
                       block_overrides=OVERRIDES)
cols = ["arm", "macro_mae_extractant", "macro_mae_extractant_seed_sd", "macro_mae_chemotype",
        "macro_mae_far", "macro_sign_acc_strong", "macro_pair_spearman", "pooled_mae"]
print(f"\n{time.time()-t0:.0f}s total\n")
print(board[cols].round(4).to_string(index=False))
board.to_csv(f"gen13_separation/analysis/stage3/s3_sweep_{DESIGN}.csv", index=False)
table.to_parquet(f"gen13_separation/analysis/stage3/s3_sweep_{DESIGN}_pairs.parquet", index=False)
