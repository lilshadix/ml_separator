"""L3 -- two narrow decision questions on the frozen bench (PRE_REGISTRATION.md section 3, L3a / L3c).

Everything here is built on the frozen machinery and never re-implements a metric, a splitter or
the paired bootstrap:

* held-out predictions come from ``gen15.valuebench.run_arms`` with the deployed ``gen15.arms.g14``
  (the standard five-design runs, discovery seeds by default -- ``seeds`` is never passed);
* the k = 1 route re-uses ``gen15.fewshot`` (``blup``, ``residual_covariance``, ``centred_residual``,
  ``smooth_covariance``, ``pick_support``, ``naive_line``) and rebuilds the per-(seed, chemotype,
  publication) covariance exactly as ``fewshot.evaluate.cov_for`` does (leave-chemotype-out,
  ``mask_publication=True``, shrink 0.25, ``NOISE_VAR`` 0.09); the copy is verified against
  ``fewshot.evaluate`` in ``scripts/l3_decision.py`` before any L3c number is used;
* task construction and every rank statistic follow ``gen15_curve/exp/decision/decmetrics.py``
  (Q3): candidates are the held-out extractants of a seed that measured the pair, an extractant's
  cells collapsed by the median, tasks with < ``MIN_EXT_RANK`` = 5 candidates dropped and counted,
  ties scored as the exact expectation under uniform random tie-breaking.  Point estimates go
  through ``decmetrics.cross_extractant`` itself; the vectorised weighted versions used for the
  chemotype-blocked bootstrap, the leave-one-chemotype-out sweep and the permutation null are
  checked against the ``decmetrics`` helpers on explicitly expanded arrays (``check_weighted``).

The two intervals the protocol names are not ``paired_contrasts`` (whose unit is the extractant):
the unit here is the (seed, metal pair) task, and the chemotype-blocked interval resamples
chemotypes with replacement and rebuilds every task from the resampled extractant multiset.  Both
use ``paired_contrasts``' seed (8675309), percentile 2.5 / 97.5 and its two-sided p convention.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from gen16 import bootstrap  # noqa: F401  (sys.path + thread cap)
from gen15 import valuebench as V  # noqa: E402
from gen15 import arms as A  # noqa: E402
from gen15 import fewshot as FS  # noqa: E402
from gen13sep.amplitude_bench import LEAN_BLOCKS, cell_weights  # noqa: E402
from gen13sep.metals import LANTHANIDES  # noqa: E402
from gen13sep.splits import all_folds  # noqa: E402
from gen14.dirbench import feature_sets  # noqa: E402

DEC = bootstrap.ROOT / "gen15_curve" / "exp" / "decision"
if str(DEC) not in sys.path:
    sys.path.insert(0, str(DEC))
import decmetrics as M  # noqa: E402

LEAD = "L3"
DESIGNS = tuple(V.DESIGNS)
STRONG = M.STRONG                 # 0.3: a candidate 'succeeds' only if |log SF| >= 0.3
MIN_EXT = M.MIN_EXT_RANK          # 5 candidates per task, as Q3
DZ_BANDS = M.DZ_BANDS
BOOT_SEED = 8675309               # gen13sep.inference.SEED, shared by every resampling here
N_BOOT = 2000
N_PERM = 2000
Z_VEC = FS.Z_VEC
K1_ARMS = ["G14@k1", "NAIVE_LINE@k1", "G14"]
RAND = "_RANDOM"


# --------------------------------------------------------------------------------------
# 1. held-out predictions: the standard runs, and the curve route for k = 1
# --------------------------------------------------------------------------------------
def run_pair_table(bench, design: str) -> pd.DataFrame:
    """The standard five-design run of the deployed arm (plus FLAT as a sanity column)."""
    return V.run_arms(bench, {"G14": A.g14, "FLAT": A.flat}, design, verbose=False)


def g14_curves(bench, design: str) -> tuple[dict, dict]:
    """Out-of-fold G14 curve (14 metals) per (seed, cell index) -- pass 1 of ``fewshot.evaluate``."""
    fs = feature_sets(bench)
    X = bench.matrix(LEAN_BLOCKS)
    rich = bench.frame.n_metals.to_numpy() >= V.MIN_METALS
    curves: dict[tuple[int, int], np.ndarray] = {}
    fold_of: dict[tuple[int, int], int] = {}
    for f in all_folds(bench.frame, design=design):
        ctx = V.Ctx(bench=bench, design=design, seed=f.seed, fold=f.fold, train=f.train_index,
                    test=f.test_index, w=cell_weights(bench.groups[f.train_index],
                                                      bench.n_obs[f.train_index]),
                    model_seed=f.model_seed, fs=fs, X=X, rich=rich)
        coef = np.asarray(A.g14(ctx), dtype=float).reshape(len(f.test_index), bench.basis.shape[0])
        cv = coef @ bench.basis
        for j, ci in enumerate(f.test_index):
            key = (int(f.seed), int(ci))
            if key in curves:
                raise RuntimeError(f"cell {ci} held out twice in seed {f.seed} ({design})")
            curves[key] = cv[j]
            fold_of[key] = int(f.fold)
    return curves, fold_of


class CovFor:
    """``cov_for`` of ``fewshot.evaluate``: leave-chemotype-out residual covariance of the same
    seed's held-out G14 residual curves, publication-masked, shrink 0.25, smooth fallback < 10."""

    def __init__(self, bench, curves: dict, *, mask_publication: bool = True) -> None:
        self.groups = bench.groups
        self.pub = bench.frame.publication_id.astype(str).to_numpy()
        self.mask = mask_publication
        self.resid: dict[int, dict[int, np.ndarray]] = {}
        for (seed, ci), cv in curves.items():
            self.resid.setdefault(seed, {})[ci] = FS.centred_residual(bench.Y[ci], cv)
        self.fallback = FS.smooth_covariance(scale=0.25, length=1.0, nugget=0.05,
                                             basis_row=bench.basis[0])
        self.cache: dict[tuple, np.ndarray] = {}
        self.n_fallback = 0

    def __call__(self, seed: int, ci: int) -> np.ndarray:
        chemotype, publication = str(self.groups[ci]), self.pub[ci]
        key = (seed, chemotype, publication if self.mask else "")
        if key not in self.cache:
            rows = [r for cj, r in self.resid[seed].items()
                    if self.groups[cj] != chemotype
                    and not (self.mask and self.pub[cj] == publication)]
            if len(rows) < 10:
                self.cache[key] = self.fallback
                self.n_fallback += 1
            else:
                self.cache[key] = FS.residual_covariance(np.array(rows))
        return self.cache[key]


def k1_table(bench, curves: dict, fold_of: dict, cov_for: CovFor) -> tuple[pd.DataFrame, dict]:
    """One row per (seed, held-out cell, target pair) with the k = 1 predictions for that pair.

    The cell's one measured pair is its widest-dZ pair *excluding the target* (``pick_support``
    'widest' on the remaining pairs).  Cells with a single measured pair have nothing to measure
    once the target is excluded and are dropped and counted.
    """
    frame = bench.frame
    rows, n_cells_single, n_rows_single = [], 0, 0
    for (seed, ci), c_pred in curves.items():
        y = bench.Y[ci]
        obs = np.flatnonzero(~np.isnan(y))
        pairs = [(int(a), int(b)) for i, a in enumerate(obs) for b in obs[i + 1:]]
        if len(pairs) < 2:
            n_cells_single += 1
            n_rows_single += len(pairs)
            continue
        cov = cov_for(seed, ci)
        base = {"split_seed": seed, "fold": fold_of[(seed, ci)],
                "cell_id": frame.cell_id.iat[ci], "extractant": frame.extractant.iat[ci],
                "chemotype": frame.chemotype.iat[ci], "n_metals": int(frame.n_metals.iat[ci])}
        for a, b in pairs:
            others = [p for p in pairs if p != (a, b)]
            sa, sb = FS.pick_support(others, 1, "widest", cov)[0]
            sup = [(sa, sb, float(y[sa] - y[sb]))]
            bl = FS.blup(c_pred, cov, sup)
            nl = FS.naive_line(bench.basis, sup)
            rec = dict(base, A=LANTHANIDES[a], B=LANTHANIDES[b], dZ=int(Z_VEC[b] - Z_VEC[a]),
                       y=float(y[a] - y[b]), G14=float(c_pred[a] - c_pred[b]))
            rec["G14@k1"] = float(bl[a] - bl[b])
            rec["NAIVE_LINE@k1"] = float(nl[a] - nl[b])
            rec.update(sup_A=LANTHANIDES[sa], sup_B=LANTHANIDES[sb],
                       sup_dZ=int(Z_VEC[sb] - Z_VEC[sa]), sup_y=sup[0][2])
            rows.append(rec)
    return pd.DataFrame(rows), {"n_cell_seeds_single_pair": n_cells_single,
                                "n_rows_single_pair": n_rows_single}


# --------------------------------------------------------------------------------------
# 2. tasks
# --------------------------------------------------------------------------------------
@dataclass
class Tasks:
    design: str
    arms: list[str]
    ext: list[str]                      # extractant index space
    chem_of: np.ndarray                 # chemotype per extractant
    seeds: np.ndarray                   # per task
    A: np.ndarray
    B: np.ndarray
    dZ: np.ndarray
    idx: list[np.ndarray]               # per task: candidate extractant indices
    obs: list[np.ndarray]               # per task: median observed log SF per candidate
    pred: dict[str, list[np.ndarray]]   # arm -> per task median prediction per candidate
    n_dropped_lt_min: int = 0
    n_dropped_ptp0: int = 0
    n_cand_before: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.idx)

    def sizes(self) -> np.ndarray:
        return np.array([len(i) for i in self.idx])


def build_tasks(tab: pd.DataFrame, design: str, arms: list[str], *,
                drop_ptp0: bool = False) -> Tasks:
    """Q3 task construction (``decmetrics.cross_extractant`` unit='extractant')."""
    key = ["split_seed", "A", "B"]
    g = (tab.groupby(key + ["extractant"], sort=False)[["y"] + arms].median().reset_index())
    chem = tab.drop_duplicates("extractant").set_index("extractant")["chemotype"].astype(str)
    ext = sorted(chem.index)
    pos = {e: i for i, e in enumerate(ext)}
    zvec = {m: Z for m, Z in zip(LANTHANIDES, Z_VEC)}
    T = Tasks(design=design, arms=list(arms), ext=ext, chem_of=chem.reindex(ext).to_numpy(),
              seeds=np.array([], dtype=int), A=np.array([]), B=np.array([]), dZ=np.array([], dtype=int),
              idx=[], obs=[], pred={a: [] for a in arms})
    seeds, As, Bs, dZs = [], [], [], []
    for (seed, a, b), blk in g.groupby(key, sort=False):
        n = len(blk)
        T.n_cand_before += n
        if n < MIN_EXT:
            T.n_dropped_lt_min += 1
            continue
        obs = blk["y"].to_numpy(dtype=float)
        if drop_ptp0 and np.ptp(obs) == 0:
            T.n_dropped_ptp0 += 1
            continue
        seeds.append(int(seed)); As.append(a); Bs.append(b); dZs.append(int(zvec[b] - zvec[a]))
        T.idx.append(np.array([pos[e] for e in blk["extractant"]], dtype=int))
        T.obs.append(obs)
        for arm in arms:
            T.pred[arm].append(blk[arm].to_numpy(dtype=float))
    T.seeds, T.A, T.B, T.dZ = np.array(seeds), np.array(As), np.array(Bs), np.array(dZs)
    return T


def per_task_frame(T: Tasks) -> pd.DataFrame:
    return pd.DataFrame({"design": T.design, "split_seed": T.seeds, "A": T.A, "B": T.B,
                         "dZ": T.dZ, "n_candidates": T.sizes()})


# --------------------------------------------------------------------------------------
# 3. resampling weights (extractant multiplicities) and the seed-macro aggregate
# --------------------------------------------------------------------------------------
def block_members(T: Tasks) -> tuple[list[str], list[np.ndarray]]:
    names = sorted(set(T.chem_of))
    return names, [np.flatnonzero(T.chem_of == b) for b in names]


def blocked_weights(T: Tasks, reps: int = N_BOOT, seed: int = BOOT_SEED) -> np.ndarray:
    """(reps, n_ext) multiplicities: chemotypes drawn with replacement exactly as
    ``paired_contrasts`` draws its blocks (``rng.integers(0, n_blocks, size=(reps, n_blocks))``)."""
    names, members = block_members(T)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(names), size=(reps, len(names)))
    counts = np.zeros((reps, len(names)))
    for j in range(len(names)):
        counts[:, j] = (picks == j).sum(1)
    W = np.zeros((reps, len(T.ext)))
    for j, idx in enumerate(members):
        W[:, idx] = counts[:, [j]]
    return W


def loco_weights(T: Tasks) -> tuple[list[str], np.ndarray]:
    names, members = block_members(T)
    W = np.ones((len(names), len(T.ext)))
    for j, idx in enumerate(members):
        W[j, idx] = 0.0
    return names, W


def seed_macro(values: np.ndarray, seeds: np.ndarray, sel: np.ndarray | None = None) -> np.ndarray:
    """Mean within split seed over the (finite, selected) tasks, then over seeds.  ``values`` is
    (R, T) or (T,); returns (R,) or a scalar.  Q3's aggregation unit."""
    v = np.atleast_2d(np.asarray(values, dtype=float)).copy()
    if sel is not None:
        v[:, ~np.asarray(sel, dtype=bool)] = np.nan
    per_seed = []
    for s in np.unique(seeds):
        cols = seeds == s
        blk = v[:, cols]
        ok = np.isfinite(blk)
        cnt = ok.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(cnt > 0, np.nansum(np.where(ok, blk, 0.0), 1) / np.maximum(cnt, 1), np.nan)
        per_seed.append(m)
    per_seed = np.stack(per_seed, 1)                       # (R, n_seeds)
    ok = np.isfinite(per_seed)
    cnt = ok.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(cnt > 0, np.nansum(np.where(ok, per_seed, 0.0), 1) / np.maximum(cnt, 1), np.nan)
    return out if np.ndim(values) == 2 else float(out[0])


def two_sided_p(draws: np.ndarray) -> float:
    fin = draws[np.isfinite(draws)]
    return float(min(1.0, 2 * min((fin <= 0).mean(), (fin >= 0).mean())))


def pct(draws: np.ndarray, q: float) -> float:
    return float(np.nanquantile(draws, q))


# --------------------------------------------------------------------------------------
# 4. L3a -- measurements saved by the direction call
# --------------------------------------------------------------------------------------
def expected_draws(N, K):
    """Expected draws to the first success, K successes among N in uniform random order."""
    return (np.asarray(N, dtype=float) + 1.0) / (np.asarray(K, dtype=float) + 1.0)


def e_model_from_counts(N, K, Nk, Kk):
    """Kept set first (N', K'), falling through to the deferred set if the kept set has no success."""
    N, K, Nk, Kk = (np.asarray(x, dtype=float) for x in (N, K, Nk, Kk))
    return np.where(Kk > 0, (Nk + 1.0) / (Kk + 1.0), Nk + (N - Nk + 1.0) / (K - Kk + 1.0))


def calls_of(pred: np.ndarray) -> np.ndarray:
    """Sign call: -1 heavy, +1 light, 0 = no call (kept for either request)."""
    return np.sign(np.asarray(pred, dtype=float)).astype(int)


def l3a_matrices(T: Tasks, calls: list[np.ndarray]) -> dict:
    """Extractant x task indicator matrices so that multiplicity weights W (R x n_ext) give every
    per-task count as one matrix product."""
    n_e, n_t = len(T.ext), T.n
    C = np.zeros((n_e, n_t))
    S = {d: np.zeros((n_e, n_t)) for d in (-1, 1)}
    KEPT = {d: np.zeros((n_e, n_t)) for d in (-1, 1)}
    KS = {d: np.zeros((n_e, n_t)) for d in (-1, 1)}
    for t in range(n_t):
        idx, obs, call = T.idx[t], T.obs[t], calls[t]
        C[idx, t] = 1.0
        for d in (-1, 1):
            succ = (np.sign(obs) == d) & (np.abs(obs) >= STRONG)
            kept = call != -d
            S[d][idx, t] = succ
            KEPT[d][idx, t] = kept
            KS[d][idx, t] = kept & succ
    return {"C": C, "S": S, "KEPT": KEPT, "KS": KS}


def l3a_saved_weighted(W: np.ndarray, mats: dict) -> dict[str, np.ndarray]:
    """Per-task saved / E_random / E_model for every weight row: (R, T) arrays, NaN where the
    resampled task has fewer than MIN_EXT candidates."""
    N = W @ mats["C"]
    valid = N >= MIN_EXT
    out = {}
    e_rand, e_mod = {}, {}
    for d in (-1, 1):
        K, Nk, Kk = W @ mats["S"][d], W @ mats["KEPT"][d], W @ mats["KS"][d]
        e_rand[d] = expected_draws(N, K)
        e_mod[d] = e_model_from_counts(N, K, Nk, Kk)
    out["saved_heavy"] = e_rand[-1] - e_mod[-1]
    out["saved_light"] = e_rand[1] - e_mod[1]
    out["saved"] = 0.5 * (out["saved_heavy"] + out["saved_light"])
    out["e_random"] = 0.5 * (e_rand[-1] + e_rand[1])
    out["e_model"] = 0.5 * (e_mod[-1] + e_mod[1])
    out["e_random_heavy"], out["e_random_light"] = e_rand[-1], e_rand[1]
    out["e_model_heavy"], out["e_model_light"] = e_mod[-1], e_mod[1]
    for k in out:
        out[k] = np.where(valid, out[k], np.nan)
    out["n"] = np.where(valid, N, np.nan)
    return out


def l3a_per_task_direct(T: Tasks, calls: list[np.ndarray]) -> pd.DataFrame:
    """The same numbers computed task by task with no matrices -- the check on the matrix route."""
    recs = []
    for t in range(T.n):
        obs, call = T.obs[t], calls[t]
        N = len(obs)
        rec = {"task": t}
        for d, lab in ((-1, "heavy"), (1, "light")):
            succ = (np.sign(obs) == d) & (np.abs(obs) >= STRONG)
            kept = call != -d
            K, Nk, Kk = int(succ.sum()), int(kept.sum()), int((kept & succ).sum())
            er = expected_draws(N, K)
            em = float(e_model_from_counts(N, K, Nk, Kk))
            rec.update({f"e_random_{lab}": float(er), f"e_model_{lab}": em,
                        f"saved_{lab}": float(er - em), f"K_{lab}": K, f"kept_{lab}": Nk,
                        f"deferred_{lab}": N - Nk})
        rec["saved"] = 0.5 * (rec["saved_heavy"] + rec["saved_light"])
        rec["e_random"] = 0.5 * (rec["e_random_heavy"] + rec["e_random_light"])
        rec["e_model"] = 0.5 * (rec["e_model_heavy"] + rec["e_model_light"])
        rec["n_no_call"] = int((call == 0).sum())
        rec["n_called_heavy"] = int((call == -1).sum())
        rec["n_called_light"] = int((call == 1).sum())
        recs.append(rec)
    return pd.DataFrame(recs)


def l3a_permutation(T: Tasks, calls: list[np.ndarray], reps: int = N_PERM,
                    seed: int = BOOT_SEED) -> dict[str, np.ndarray]:
    """Within every task the model's sign calls are permuted across candidates; returns per-task
    (R, T) saved arrays for the whole, heavy and light statistics."""
    rng = np.random.default_rng(seed)
    out = {k: np.zeros((reps, T.n)) for k in ("saved", "saved_heavy", "saved_light")}
    for t in range(T.n):
        obs, call = T.obs[t], calls[t]
        n = len(obs)
        perm = np.argsort(rng.random((reps, n)), axis=1)
        cp = call[perm]
        sv = {}
        for d, lab in ((-1, "heavy"), (1, "light")):
            succ = (np.sign(obs) == d) & (np.abs(obs) >= STRONG)
            K = int(succ.sum())
            kept = cp != -d
            Nk = kept.sum(1)
            Kk = (kept & succ[None, :]).sum(1)
            sv[lab] = expected_draws(n, K) - e_model_from_counts(n, K, Nk, Kk)
            out[f"saved_{lab}"][:, t] = sv[lab]
        out["saved"][:, t] = 0.5 * (sv["heavy"] + sv["light"])
    return out


def band_masks(T: Tasks) -> dict[str, np.ndarray]:
    return {lab: (T.dZ >= lo) & (T.dZ <= hi) for lo, hi, lab in DZ_BANDS}


# --------------------------------------------------------------------------------------
# 5. L3c -- ranking statistics with multiplicity weights (vectorised over weight rows)
# --------------------------------------------------------------------------------------
def weighted_ranks(v: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Average ranks (scipy ``rankdata`` 'average') of the multiset in which candidate j appears
    W[r, j] times, returned for every row r as (R, n).  rank_i = #{v_j < v_i} + (#{v_j = v_i} + 1)/2
    with counts taken with multiplicity."""
    less = (v[None, :] < v[:, None]).astype(float)     # less[i, j] = 1 if v_j < v_i
    eq = (v[None, :] == v[:, None]).astype(float)
    return W @ less.T + 0.5 * (W @ eq.T + 1.0)


def weighted_corr(X: np.ndarray, Y: np.ndarray, W: np.ndarray) -> np.ndarray:
    sw = W.sum(1, keepdims=True)
    mx = (W * X).sum(1, keepdims=True) / sw
    my = (W * Y).sum(1, keepdims=True) / sw
    cov = (W * (X - mx) * (Y - my)).sum(1)
    vx = (W * (X - mx) ** 2).sum(1)
    vy = (W * (Y - my) ** 2).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return cov / np.sqrt(vx * vy)


def _pick(score: np.ndarray, present: np.ndarray, W: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Expected ``value`` of the pick under uniform tie-breaking among the argmax of ``score``
    over the present multiset (``decmetrics._tie_expected_value`` with multiplicity)."""
    s = np.where(present, score, -np.inf)
    m = s.max(1, keepdims=True)
    tied = present & (s >= m - 1e-12)
    wt = tied * W
    return (wt * value).sum(1) / wt.sum(1)


def task_stats_weighted(p: np.ndarray, obs: np.ndarray, W: np.ndarray) -> dict[str, np.ndarray]:
    """Q3's per-task statistics (Spearman, tie-expected top-1, regret; both directions averaged)
    for the multiset described by W (R, n).  ``p`` may be (n,) or (R, n) (permuted predictions).
    NaN where the multiset has < MIN_EXT members or a constant observed value."""
    R = W.shape[0]
    p = np.asarray(p, dtype=float)
    if p.ndim != 1:
        raise ValueError("task_stats_weighted takes one prediction vector; permuted predictions "
                         "go through task_stats_permuted")
    P = np.broadcast_to(p, (R, len(obs))).astype(float)
    present = W > 0
    ntot = W.sum(1)
    hi = np.where(present, obs, -np.inf).max(1)
    lo = np.where(present, obs, np.inf).min(1)
    valid = (ntot >= MIN_EXT) & (hi - lo > 0)
    best_hi = present & (obs[None, :] >= hi[:, None] - 1e-12)
    best_lo = present & (obs[None, :] <= lo[:, None] + 1e-12)
    obs_b = np.broadcast_to(obs, (R, len(obs)))
    val_up = _pick(P, present, W, obs_b)
    val_dn = _pick(-P, present, W, obs_b)
    top1_up = _pick(P, present, W, best_hi.astype(float))
    top1_dn = _pick(-P, present, W, best_lo.astype(float))
    # Spearman with the tie convention: constant predictor -> 0
    p_hi = np.where(present, P, -np.inf).max(1)
    p_lo = np.where(present, P, np.inf).min(1)
    rp = weighted_ranks(p, W)
    ro = weighted_ranks(obs, W)
    sp = weighted_corr(rp, ro, W)
    sp = np.where(p_hi - p_lo > 0, sp, 0.0)
    out = {"spearman": sp, "top1": 0.5 * (top1_up + top1_dn),
           "regret": 0.5 * ((hi - val_up) + (val_dn - lo)),
           "top1_up": top1_up, "top1_down": top1_dn,
           "regret_up": hi - val_up, "regret_down": val_dn - lo}
    return {k: np.where(valid, v, np.nan) for k, v in out.items()}


def task_stats_permuted(p: np.ndarray, obs: np.ndarray, perm: np.ndarray) -> dict[str, np.ndarray]:
    """The same statistics with ``p`` permuted across candidates by every row of ``perm``
    (R, n) -- unit weights; ranks are permuted with the predictions."""
    R, n = perm.shape
    P = p[perm]
    W = np.ones((R, n))
    present = W > 0
    hi, lo = obs.max(), obs.min()
    valid = (n >= MIN_EXT) & (hi - lo > 0)
    best_hi = (obs >= hi - 1e-12).astype(float)
    best_lo = (obs <= lo + 1e-12).astype(float)
    obs_b = np.broadcast_to(obs, (R, n))
    val_up = _pick(P, present, W, obs_b)
    val_dn = _pick(-P, present, W, obs_b)
    top1_up = _pick(P, present, W, np.broadcast_to(best_hi, (R, n)))
    top1_dn = _pick(-P, present, W, np.broadcast_to(best_lo, (R, n)))
    if np.ptp(p) == 0:
        sp = np.zeros(R)
    else:
        rp = rankdata(p)[perm]
        ro = np.broadcast_to(rankdata(obs), (R, n))
        sp = weighted_corr(rp, ro, W)
    out = {"spearman": sp, "top1": 0.5 * (top1_up + top1_dn),
           "regret": 0.5 * ((hi - val_up) + (val_dn - lo))}
    return {k: (v if valid else np.full(R, np.nan)) for k, v in out.items()}


def check_weighted(T: Tasks, per_pair: pd.DataFrame, *, n_random: int = 40,
                   seed: int = 20260910) -> dict[str, float]:
    """(i) unit weights reproduce ``decmetrics.cross_extractant``'s per-task numbers; (ii) random
    multiplicities reproduce the ``decmetrics`` helpers evaluated on the explicitly expanded
    arrays; (iii) permuted predictions reproduce the helpers on the permuted arrays."""
    arms = T.arms + [RAND]
    worst_unit = 0.0
    for t in range(T.n):
        row = per_pair[(per_pair.split_seed == T.seeds[t]) & (per_pair.A == T.A[t]) & (per_pair.B == T.B[t])]
        for arm in arms:
            p = np.zeros(len(T.obs[t])) if arm == RAND else T.pred[arm][t]
            st = task_stats_weighted(p, T.obs[t], np.ones((1, len(T.obs[t]))))
            ref = row[row.arm == arm].iloc[0]
            for col in ("spearman", "top1", "regret"):
                worst_unit = max(worst_unit, abs(float(st[col][0]) - float(ref[col])))
    rng = np.random.default_rng(seed)
    worst_w, worst_perm = 0.0, 0.0
    for _ in range(n_random):
        t = int(rng.integers(T.n))
        obs = T.obs[t]
        n = len(obs)
        w = rng.integers(0, 4, size=n).astype(float)
        if w.sum() < MIN_EXT:
            w[:MIN_EXT] += 1
        arm = T.arms[int(rng.integers(len(T.arms)))]
        p = T.pred[arm][t]
        st = task_stats_weighted(p, obs, w[None, :])
        rep = np.repeat(np.arange(n), w.astype(int))
        pe, oe = p[rep], obs[rep]
        hi, lo = oe.max(), oe.min()
        if np.ptp(oe) == 0:
            continue
        ref_sp = M._spearman(pe, oe)
        ref_top = 0.5 * (M._tie_expected_top1(pe, (oe >= hi - 1e-12).astype(float))
                         + M._tie_expected_top1(-pe, (oe <= lo + 1e-12).astype(float)))
        ref_reg = 0.5 * ((hi - M._tie_expected_value(pe, oe)) + (M._tie_expected_value(-pe, oe) - lo))
        worst_w = max(worst_w, abs(float(st["spearman"][0]) - ref_sp),
                      abs(float(st["top1"][0]) - ref_top), abs(float(st["regret"][0]) - ref_reg))
        perm = np.argsort(rng.random((3, n)), axis=1)
        sp = task_stats_permuted(p, obs, perm)
        for r in range(3):
            pp = p[perm[r]]
            ref_sp = M._spearman(pp, obs)
            ref_reg = 0.5 * ((hi - M._tie_expected_value(pp, obs)) + (M._tie_expected_value(-pp, obs) - lo))
            worst_perm = max(worst_perm, abs(float(sp["spearman"][r]) - ref_sp),
                             abs(float(sp["regret"][r]) - ref_reg))
    return {"unit_vs_decmetrics": worst_unit, "weighted_vs_expanded": worst_w,
            "permuted_vs_helpers": worst_perm}


def l3c_stats_for_weights(T: Tasks, W: np.ndarray, arms: list[str]) -> dict[str, dict[str, np.ndarray]]:
    """arm -> stat -> (R, T) for a weight matrix (bootstrap replicates or LOCO rows)."""
    R = W.shape[0]
    out = {a: {k: np.full((R, T.n), np.nan) for k in ("spearman", "top1", "regret")} for a in arms}
    for t in range(T.n):
        Wt = W[:, T.idx[t]]
        for a in arms:
            p = np.zeros(len(T.obs[t])) if a == RAND else T.pred[a][t]
            st = task_stats_weighted(p, T.obs[t], Wt)
            for k in ("spearman", "top1", "regret"):
                out[a][k][:, t] = st[k]
    return out


def l3c_permuted_stats(T: Tasks, arm: str, reps: int = N_PERM, seed: int = BOOT_SEED) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = {k: np.full((reps, T.n), np.nan) for k in ("spearman", "top1", "regret")}
    for t in range(T.n):
        n = len(T.obs[t])
        perm = np.argsort(rng.random((reps, n)), axis=1)
        st = task_stats_permuted(T.pred[arm][t], T.obs[t], perm)
        for k in out:
            out[k][:, t] = st[k]
    return out


# --------------------------------------------------------------------------------------
# 6. contrast rows in the paired_contrasts layout
# --------------------------------------------------------------------------------------
CONTRAST_COLUMNS = ["lead", "family", "design", "comparison", "value", "reference", "candidate",
                    "point", "ci95_low", "ci95_high", "p_two_sided", "p_perm", "p_registered",
                    "ci_task_low", "ci_task_high", "p_task_boot", "ci_block_low", "ci_block_high",
                    "p_block_boot", "conservative_interval", "n_units", "n_extractants",
                    "n_blocks", "seeds_positive", "n_seeds", "loco_min", "loco_max", "loco_stable",
                    "loco_sign_stable", "margin", "passes_registered", "passes_P1", "rule",
                    "replicates", "permutations", "bootstrap_seed"]


def contrast_row(*, design: str, comparison: str, value: str, reference: str, candidate: str,
                 family: str, point: float, block_draws: np.ndarray, task_draws: np.ndarray | None,
                 perm_draws: np.ndarray | None, per_seed: np.ndarray, loco: np.ndarray,
                 n_units: int, n_ext: int, n_blocks: int, rule: str, registered_rule) -> dict:
    ci_b = (pct(block_draws, 0.025), pct(block_draws, 0.975))
    p_b = two_sided_p(block_draws)
    if task_draws is not None:
        ci_t = (pct(task_draws, 0.025), pct(task_draws, 0.975))
        p_t = two_sided_p(task_draws)
        if p_t > p_b or (p_t == p_b and (ci_t[1] - ci_t[0]) > (ci_b[1] - ci_b[0])):
            ci, p_boot, cons = ci_t, p_t, "task"
        else:
            ci, p_boot, cons = ci_b, p_b, "chemotype-blocked"
    else:
        ci_t, p_t = (np.nan, np.nan), np.nan
        ci, p_boot, cons = ci_b, p_b, "chemotype-blocked"
    p_perm = float((perm_draws >= point).mean()) if perm_draws is not None else np.nan
    fin_loco = loco[np.isfinite(loco)]
    loco_stable = bool(point != 0 and np.all(np.sign(fin_loco) == np.sign(point)))
    seeds_positive = int((per_seed > 0).sum())
    p_registered, passes_registered = registered_rule(point, ci, p_boot, p_perm)
    passes_P1 = bool(point >= 0.02 and ci[0] > 0 and p_boot < 0.05 and seeds_positive >= 4 and loco_stable)
    return {"lead": LEAD, "family": family, "design": design, "comparison": comparison,
            "value": value, "reference": reference, "candidate": candidate, "point": float(point),
            "ci95_low": ci[0], "ci95_high": ci[1], "p_two_sided": p_boot, "p_perm": p_perm,
            "p_registered": p_registered, "ci_task_low": ci_t[0], "ci_task_high": ci_t[1],
            "p_task_boot": p_t, "ci_block_low": ci_b[0], "ci_block_high": ci_b[1], "p_block_boot": p_b,
            "conservative_interval": cons, "n_units": int(n_units), "n_extractants": int(n_ext),
            "n_blocks": int(n_blocks), "seeds_positive": seeds_positive, "n_seeds": int(len(per_seed)),
            "loco_min": float(np.nanmin(loco)) if fin_loco.size else np.nan,
            "loco_max": float(np.nanmax(loco)) if fin_loco.size else np.nan,
            "loco_stable": loco_stable, "loco_sign_stable": loco_stable, "margin": 0.02,
            "passes_registered": passes_registered, "passes_P1": passes_P1, "rule": rule,
            "replicates": N_BOOT, "permutations": N_PERM if perm_draws is not None else 0,
            "bootstrap_seed": BOOT_SEED}


def rule_l3a(point, ci, p_boot, p_perm):
    """L3a: positive iff saved > 0, permutation p < 0.05 and the blocked CI excludes zero."""
    return p_perm, bool(point > 0 and p_perm < 0.05 and ci[0] > 0)


def rule_l3c(point, ci, p_boot, p_perm):
    """L3c: the p quoted is the most conservative of task bootstrap, blocked bootstrap and
    permutation (decided before any number was seen); positive iff that p < 0.05 and point > 0."""
    p = float(np.nanmax([p_boot, p_perm]))
    return p, bool(point > 0 and p < 0.05)


def timer(t0: float) -> str:
    return f"{time.time() - t0:.0f}s"
