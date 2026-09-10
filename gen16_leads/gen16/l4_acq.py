"""L4 -- corpus-level acquisition: thin a fold's training set to a chosen chemotype set, refit G14.

Everything scored here goes through the frozen bench: ``gen13sep.amplitude_bench._pair_frame`` builds
the held-out pair table, ``gen15.arms.g14`` is the refitted arm, ``gen13sep.metrics.per_extractant``
and ``gen13sep.inference.paired_contrasts`` do the scoring and the inference.  This module only
decides *which training chemotypes a fit may see*; the test index of a fold is never touched, so
every budget of every order scores byte-identical held-out pairs.

Orders (criteria see features only; UNCERT sees the labels of the chemotypes already chosen):

* ``RANDOM``  uniform random permutation of the training chemotypes.
* ``MAXMIN``  farthest-point traversal in ECFP4 space between chemotype centroids.  The centroid
              is the mean 2048-bit Morgan(r=2) vector of the chemotype's cohort extractants; the
              distance is ``1 - generalised Tanimoto`` of two mean vectors
              (``sum(min) / sum(max)``), the continuous extension of the bit Tanimoto.
* ``AOPT``    greedy A-optimal design for the standardised ridge-logistic on TOPO39 (penalty 1).
              Posterior precision ``P = I + sum_{chosen rich cells} x x^T`` (cells standardised
              with the training pool's median-imputed statistics); the score of a candidate
              chemotype is the trace of the posterior variance of the linear predictor over the
              whole training pool, ``sum_i x_i^T P_new^{-1} x_i``, and the chemotype minimising it
              is added.  The first chemotype is the one minimising the score alone.
* ``UNCERT``  the first three by AOPT; then the G14 logistic is refitted on the chosen cells and
              the chemotype whose training cells have the smallest mean ``|p_heavy - 0.5|`` is
              added.  When the current fit is degenerate (fewer than 10 well-determined chosen
              cells, or one direction class) the AOPT score decides that step.

Ties in every criterion are broken by chemotype name (sorted ascending) so an order is a pure
function of its inputs.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "gen13_separation", ROOT / "gen14_direction", ROOT / "gen15_curve"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gen13sep.amplitude_bench import LEAN_BLOCKS, BenchData, _pair_frame, cell_weights  # noqa: E402
from gen13sep.splits import Fold  # noqa: E402
from gen14.dirbench import feature_sets  # noqa: E402
from gen14.models import dir_logistic  # noqa: E402
from gen15 import arms as A  # noqa: E402
from gen15.valuebench import Ctx, MIN_METALS  # noqa: E402

ORDERS: tuple[str, ...] = ("RANDOM", "MAXMIN", "AOPT", "UNCERT")
BUDGETS: tuple[int, ...] = (6, 9, 12, 16, 20, 24)
N_DRAWS = 20
LAMBDA = 1.0
MIN_RICH_FIT = 10          # guard: fewer well-determined training cells than this -> FLAT
UNCERT_WARMUP = 3          # first three UNCERT picks come from AOPT


# --------------------------------------------------------------------------------------
# fold context
# --------------------------------------------------------------------------------------
@dataclass
class Prepared:
    """Fold-independent inputs, built once per bench."""
    bench: BenchData
    X: np.ndarray                 # LEAN block matrix, all cells
    fs: dict
    rich: np.ndarray              # well-determined cell mask
    topo: np.ndarray              # TOPO39 raw matrix, all cells
    ext_of: np.ndarray            # extractant SMILES per cell
    centroid: dict                # chemotype -> mean ECFP4 bit vector
    chemo_dist: pd.DataFrame      # chemotype x chemotype distance matrix


def prepare(bench: BenchData) -> Prepared:
    fs = feature_sets(bench)
    X = bench.matrix(LEAN_BLOCKS)
    rich = bench.frame.n_metals.to_numpy() >= MIN_METALS
    topo = X[:, fs["TOPO39"]]
    ext_of = bench.frame.extractant.astype(str).to_numpy()
    centroid = chemotype_centroids(bench)
    names = sorted(centroid)
    D = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j > i:
                D[i, j] = D[j, i] = 1.0 - generalised_tanimoto(centroid[a], centroid[b])
    return Prepared(bench=bench, X=X, fs=fs, rich=rich, topo=topo, ext_of=ext_of,
                    centroid=centroid, chemo_dist=pd.DataFrame(D, index=names, columns=names))


def thin_fold(fold: Fold, groups: np.ndarray, chosen: Sequence[str]) -> Fold:
    """A copy of ``fold`` whose training set keeps only the cells of ``chosen`` chemotypes.

    The test index is the original object, untouched.  Inner splits are not used by G14 and are
    set to the thinned training set itself.
    """
    keep = set(chosen)
    tr = fold.train_index
    mask = np.array([g in keep for g in groups[tr]], dtype=bool)
    sub = tr[mask]
    return Fold(design=fold.design, seed=fold.seed, fold=fold.fold, train_index=sub,
                test_index=fold.test_index, inner_train_index=sub,
                inner_validation_index=sub[:0], held_out_groups=fold.held_out_groups)


def make_ctx(P: Prepared, fold: Fold, design: str) -> Ctx:
    b = P.bench
    tr = fold.train_index
    return Ctx(bench=b, design=design, seed=fold.seed, fold=fold.fold, train=tr,
               test=fold.test_index, w=cell_weights(b.groups[tr], b.n_obs[tr]),
               model_seed=fold.model_seed, fs=P.fs, X=P.X, rich=P.rich)


def fit_g14_guarded(P: Prepared, fold: Fold, design: str) -> tuple[np.ndarray, bool]:
    """``gen15.arms.g14`` on the (thinned) fold; FLAT when the fit is not well posed.

    Returns ``(coef (n_test, 2), fell_back)``.
    """
    ctx = make_ctx(P, fold, design)
    rtr = ctx.rich_train()
    amp = ctx.amp[rtr]
    if len(rtr) < MIN_RICH_FIT or len(set((amp < 0).tolist())) < 2:
        return A.flat(ctx), True
    return A.g14(ctx), False


def pair_table(P: Prepared, fold: Fold) -> pd.DataFrame:
    return _pair_frame(P.bench.frame, P.bench.Y, fold.test_index, fold.seed, fold.fold)


def pair_predictions(coef: np.ndarray, pairs: pd.DataFrame, P: Prepared) -> np.ndarray:
    curve = np.asarray(coef, dtype=float).reshape(-1, P.bench.basis.shape[0]) @ P.bench.basis
    loc = pairs["cell_local"].to_numpy()
    return curve[loc, pairs["ia"].to_numpy()] - curve[loc, pairs["ib"].to_numpy()]


# --------------------------------------------------------------------------------------
# ECFP4 chemotype centroids
# --------------------------------------------------------------------------------------
def morgan_matrix(smiles: Sequence[str], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out = np.zeros((len(smiles), n_bits), dtype=float)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            raise ValueError(f"RDKit cannot parse {s!r}")
        arr = np.zeros((n_bits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol), arr)
        out[i] = arr
    return out


def chemotype_centroids(bench: BenchData) -> dict[str, np.ndarray]:
    """Mean Morgan(r=2, 2048) bit vector over the distinct extractants of each cohort chemotype."""
    fr = bench.frame[["extractant", "chemotype"]].drop_duplicates("extractant")
    M = morgan_matrix(fr.extractant.astype(str).tolist())
    out = {}
    for c, idx in fr.groupby("chemotype").indices.items():
        out[str(c)] = M[idx].mean(axis=0)
    return out


def generalised_tanimoto(u: np.ndarray, v: np.ndarray) -> float:
    den = float(np.maximum(u, v).sum())
    return float(np.minimum(u, v).sum() / den) if den > 0 else 1.0


# --------------------------------------------------------------------------------------
# standardisation for the A-optimal criterion
# --------------------------------------------------------------------------------------
@dataclass
class Standardiser:
    median: np.ndarray
    mean: np.ndarray
    sd: np.ndarray            # 1.0 where the pool column is constant
    constant: np.ndarray      # boolean mask of constant / all-missing pool columns

    def __call__(self, X: np.ndarray) -> np.ndarray:
        Z = np.where(np.isfinite(X), X, self.median)
        Z = (Z - self.mean) / self.sd
        Z[:, self.constant] = 0.0
        return Z


def fit_standardiser(X_pool: np.ndarray) -> Standardiser:
    """Median imputation and z-scoring with the pool's own statistics (as G14's pipeline does on
    its training fold).  Constant and all-missing columns map to zero."""
    with np.errstate(all="ignore"):
        med = np.nanmedian(X_pool, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Z = np.where(np.isfinite(X_pool), X_pool, med)
    mean = Z.mean(axis=0)
    sd = Z.std(axis=0)
    constant = ~(sd > 1e-12)
    return Standardiser(median=med, mean=mean, sd=np.where(constant, 1.0, sd), constant=constant)


def aopt_trace(Zpool: np.ndarray, Pinv: np.ndarray) -> float:
    """Sum over pool rows of ``z^T P^{-1} z``: total posterior variance of the linear predictor."""
    return float(np.einsum("ij,jk,ik->", Zpool, Pinv, Zpool))


# --------------------------------------------------------------------------------------
# orders
# --------------------------------------------------------------------------------------
def order_random(chemos: Sequence[str], rng: np.random.Generator) -> list[str]:
    chemos = sorted(chemos)
    return [chemos[i] for i in rng.permutation(len(chemos))]


def order_maxmin(chemos: Sequence[str], dist: pd.DataFrame, start: str) -> list[str]:
    """Farthest-point traversal: after the start, always the chemotype farthest from the chosen set
    (max over candidates of the min distance to any chosen chemotype)."""
    chemos = sorted(chemos)
    D = dist.loc[chemos, chemos].to_numpy()
    idx = {c: i for i, c in enumerate(chemos)}
    chosen = [start]
    remaining = [c for c in chemos if c != start]
    mind = D[idx[start]].copy()
    while remaining:
        best = max(remaining, key=lambda c: (mind[idx[c]], -chemos.index(c)))
        chosen.append(best)
        remaining.remove(best)
        mind = np.minimum(mind, D[idx[best]])
    return chosen


def order_aopt(chemos: Sequence[str], Zpool: np.ndarray, groups_pool: np.ndarray,
               rich_pool: np.ndarray, lam: float = LAMBDA) -> list[str]:
    """Greedy A-optimal order; precision accumulates over the *well-determined* cells of the chosen
    chemotypes (the cells G14 fits on), the trace runs over every training-pool cell."""
    chemos = sorted(chemos)
    d = Zpool.shape[1]
    P = lam * np.eye(d)
    members = {c: Zpool[(groups_pool == c) & rich_pool] for c in chemos}
    chosen: list[str] = []
    remaining = list(chemos)
    while remaining:
        best, best_score = None, np.inf
        for c in remaining:
            M = members[c]
            Pn = P + M.T @ M if len(M) else P
            score = aopt_trace(Zpool, np.linalg.inv(Pn))
            if score < best_score - 1e-12:
                best, best_score = c, score
        chosen.append(best)
        remaining.remove(best)
        M = members[best]
        if len(M):
            P = P + M.T @ M
    return chosen


def order_uncert(chemos: Sequence[str], P: Prepared, tr: np.ndarray, Zpool: np.ndarray,
                 model_seed: int, lam: float = LAMBDA, warmup: int = UNCERT_WARMUP) -> list[str]:
    """AOPT for the first ``warmup`` picks, then the chemotype whose training cells are closest to
    p_heavy = 0.5 under the G14 logistic fitted on the chosen cells."""
    b = P.bench
    chemos = sorted(chemos)
    groups_pool = b.groups[tr].astype(str)
    rich_pool = P.rich[tr]
    aopt = order_aopt(chemos, Zpool, groups_pool, rich_pool, lam)
    chosen = aopt[:warmup]
    remaining = [c for c in chemos if c not in chosen]
    d = Zpool.shape[1]
    Xtopo = P.topo
    while remaining:
        cmask = np.isin(groups_pool, chosen)
        fit_cells = tr[cmask & rich_pool]
        amp = b.coef[fit_cells, 0]
        degenerate = len(fit_cells) < MIN_RICH_FIT or len(set((amp < 0).tolist())) < 2
        if degenerate:
            # AOPT decides this step
            Pm = lam * np.eye(d)
            Zc = Zpool[cmask & rich_pool]
            Pm = Pm + Zc.T @ Zc
            best, best_score = None, np.inf
            for c in remaining:
                M = Zpool[(groups_pool == c) & rich_pool]
                Pn = Pm + M.T @ M if len(M) else Pm
                s = aopt_trace(Zpool, np.linalg.inv(Pn))
                if s < best_score - 1e-12:
                    best, best_score = c, s
        else:
            w = cell_weights(b.groups[fit_cells], b.n_obs[fit_cells])
            cand_cells = tr[~cmask]
            p = dir_logistic()(Xtopo[fit_cells], amp, w, b.groups[fit_cells], Xtopo[cand_cells],
                               model_seed, P.ext_of[fit_cells])
            p = np.asarray(p, dtype=float).ravel()
            g = groups_pool[~cmask]
            best, best_score = None, np.inf
            for c in remaining:
                s = float(np.mean(np.abs(p[g == c] - 0.5)))
                if s < best_score - 1e-12:
                    best, best_score = c, s
        chosen.append(best)
        remaining.remove(best)
    return chosen


def fold_orders(P: Prepared, fold: Fold, n_draws: int = N_DRAWS) -> dict[tuple[str, int], list[str]]:
    """Every (order, draw) -> chemotype order for one fold.  RANDOM and MAXMIN carry ``n_draws``
    draws/starts; AOPT and UNCERT are deterministic (draw 0)."""
    b = P.bench
    tr = fold.train_index
    chemos = sorted(set(b.groups[tr].astype(str)))
    st = fit_standardiser(P.topo[tr])
    Zpool = st(P.topo[tr])
    out: dict[tuple[str, int], list[str]] = {}
    for d in range(n_draws):
        rng = np.random.default_rng(int(fold.model_seed) * 1000 + d)
        out[("RANDOM", d)] = order_random(chemos, rng)
    for d in range(n_draws):
        rng = np.random.default_rng(int(fold.model_seed) * 1000 + 500_000 + d)
        start = chemos[int(rng.integers(len(chemos)))]
        out[("MAXMIN", d)] = order_maxmin(chemos, P.chemo_dist, start)
    out[("AOPT", 0)] = order_aopt(chemos, Zpool, b.groups[tr].astype(str), P.rich[tr])
    out[("UNCERT", 0)] = order_uncert(chemos, P, tr, Zpool, fold.model_seed)
    for key, o in out.items():
        assert sorted(o) == chemos, f"order {key} is not a permutation of the training chemotypes"
    return out


# --------------------------------------------------------------------------------------
# prospective criterion: given the full cohort, which candidate row helps most
# --------------------------------------------------------------------------------------
def prospective_aopt(Zchosen: np.ndarray, Zcand: np.ndarray, Zavg: np.ndarray,
                     lam: float = LAMBDA) -> pd.DataFrame:
    """One-step A-optimal gain of every candidate row given ``Zchosen`` already in the design.

    ``reduction`` = drop in the mean posterior variance of the linear predictor over ``Zavg``
    when the candidate is added (Sherman-Morrison); ``own_variance`` = the candidate's own
    predictive variance ``z^T P^{-1} z`` before it is added.
    """
    d = Zchosen.shape[1]
    Pinv = np.linalg.inv(lam * np.eye(d) + Zchosen.T @ Zchosen)
    G = Zavg @ Pinv                       # n_avg x d
    own = np.einsum("ij,jk,ik->i", Zcand, Pinv, Zcand)
    cross = G @ Zcand.T                   # n_avg x n_cand : z_m^T P^-1 z_j
    red = (cross ** 2).mean(axis=0) / (1.0 + own)
    return pd.DataFrame({"aopt_reduction": red, "own_variance": own})
