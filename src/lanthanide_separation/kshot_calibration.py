"""k-shot per-extractant calibration of out-of-fold pair predictions.

Question: if a chemist measures ``k`` pairs of a *new* extractant, how much does
recalibrating the frozen model on those ``k`` pairs reduce the error on the
extractant's other pairs?  Everything is post-hoc on out-of-fold (OOF)
predictions; no model is retrained.

Design notes that the first version of this module got wrong, and which the
structure below exists to prevent:

* **Within-cell transitivity.**  ``log_SF(A,C) = log_SF(A,B) + log_SF(B,C)``
  holds *exactly* inside one (extractant, condition) cell, and the cells here are
  complete graphs.  A query pair whose two metals are connected by support pairs
  in the same cell is therefore an exact arithmetic consequence of the support —
  no model needed.  Every metric is stratified into ``all`` query rows and the
  ``free`` rows not spanned by the support (union-find per cell over the ``k_max``
  pool, so the stratum is fixed across ``k``).
* **Same-experiment vs ligand transfer.**  Drawing support uniformly over an
  extractant's rows usually samples the very condition being predicted.  The
  ``cross_condition`` policy holds out a whole condition and draws support from
  the extractant's *other* conditions — the honest ligand-transfer number.
* **Model-free null.**  With the prediction column identically zero, the
  ``trend`` form degenerates to ``y ≈ b·ΔZ``: a two-parameter fit on the same k
  measurements that uses no model at all.  The deployment claim must beat *that*,
  not merely beat ``k = 0``.

Calibrator forms, all fitted by ridge shrunk toward the identity ``a = 1, b = 0``:

* ``scale``  – ``y ≈ a·p``            antisymmetric, transitivity-preserving
* ``offset`` – ``y ≈ p + b``          canonical-orientation shift (breaks transitivity)
* ``affine`` – ``y ≈ a·p + b``        canonical-orientation affine (breaks transitivity)
* ``trend``  – ``y ≈ a·p + b·ΔZ``     antisymmetric *and* transitivity-preserving

``ΔZ`` is rescaled to the support's ``p`` norm before penalisation, so ``lam``
shrinks ``b`` toward zero as strongly as it shrinks ``a`` toward one (the naive
version shrank ``a`` 20× harder than ``b`` because ``ΔZ`` is ~8× larger than ``p``).
Degenerate fits are **rejected back to the identity**, never clipped: clipping
``a`` while keeping a jointly-fitted ``b`` produced predictions thousands of times
worse than no calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .pairs import LANTHANIDE_Z, PAIR_TARGET_COLUMN

CALIBRATOR_FORMS: tuple[str, ...] = ("scale", "offset", "affine", "trend")
TRANSITIVE_FORMS: frozenset[str] = frozenset({"scale", "trend"})
SUPPORT_POLICIES: tuple[str, ...] = ("random", "adjacent", "widest", "foreign", "cross_condition")
CONTROL_POLICIES: frozenset[str] = frozenset({"foreign"})

_TINY = 1e-9
#: Fits whose coefficients leave this box are rejected back to the identity.
DEFAULT_COEF_BOUND = 10.0
#: Per-extractant R2 needs this many query rows and this much target spread.
R2_MIN_ROWS = 15
R2_MIN_SD = 0.05


# --------------------------------------------------------------------------- #
# Calibrator closed forms
# --------------------------------------------------------------------------- #

def _trend_scale(p: np.ndarray, d: np.ndarray) -> float:
    """Factor putting ``d`` on the same scale as ``p`` so ``lam`` penalises both equally."""
    dn = float(np.sqrt(np.dot(d, d)))
    pn = float(np.sqrt(np.dot(p, p)))
    if dn <= _TINY:
        return 0.0
    return (pn / dn) if pn > _TINY else (1.0 / dn)


def fit_calibrator(
    form: str,
    p: np.ndarray,
    y: np.ndarray,
    d: np.ndarray | None,
    lam: float,
    *,
    coef_bound: float = DEFAULT_COEF_BOUND,
) -> tuple[float, float, bool]:
    """Fit one calibrator on support arrays; returns ``(a, b, rejected)``.

    Ridge penalty ``lam·(a−1)² + lam·b̃²`` where ``b̃`` is the coefficient in the
    rescaled basis (identical to ``b`` for ``affine``/``offset``).  A fit whose
    coefficients fall outside ``coef_bound`` — near-singular two-point fits — is
    rejected and the identity is returned with ``rejected = True``.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    lam_eff = max(float(lam), _TINY)
    if p.size == 0:
        return 1.0, 0.0, False
    if form == "scale":
        a = (float(np.dot(p, y)) + lam_eff) / (float(np.dot(p, p)) + lam_eff)
        return (1.0, 0.0, True) if abs(a) > coef_bound else (a, 0.0, False)
    if form == "offset":
        b = float(np.sum(y - p)) / (p.size + lam_eff)
        return (1.0, 0.0, True) if abs(b) > coef_bound else (1.0, b, False)
    if form == "affine":
        basis = np.ones_like(p)
        scale = 1.0
    elif form == "trend":
        if d is None:
            raise ValueError("form='trend' needs the delta-Z array d.")
        d = np.asarray(d, dtype=float)
        scale = _trend_scale(p, d)
        # A support whose ΔZ is constant (k = 1, or the `adjacent` policy where every
        # support pair has ΔZ = 1) cannot identify a trend: the basis is collinear
        # with an intercept, and `b` would then be extrapolated to query ΔZ up to 14.
        # Fall back to `scale` and record the degeneracy.
        if scale == 0.0 or np.std(d) <= 1e-9 * (1.0 + float(np.mean(np.abs(d)))):
            a = (float(np.dot(p, y)) + lam_eff) / (float(np.dot(p, p)) + lam_eff)
            return (1.0, 0.0, True) if abs(a) > coef_bound else (a, 0.0, True)
        basis = d * scale
    else:
        raise ValueError(f"unknown calibrator form {form!r}")
    m11 = float(np.dot(p, p)) + lam_eff
    m12 = float(np.dot(p, basis))
    m22 = float(np.dot(basis, basis)) + lam_eff
    r1 = float(np.dot(p, y)) + lam_eff
    r2 = float(np.dot(basis, y))
    det = m11 * m22 - m12 * m12
    if det <= 1e-12 * m11 * m22:
        return 1.0, 0.0, True
    a = (r1 * m22 - m12 * r2) / det
    b_t = (m11 * r2 - m12 * r1) / det
    if abs(a) > coef_bound or abs(b_t) > coef_bound:
        return 1.0, 0.0, True
    return a, b_t * scale, False


def apply_calibrator(form: str, a: float, b: float, p: np.ndarray, d: np.ndarray | None) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if form == "scale":
        return a * p
    if form == "offset":
        return p + b
    if form == "affine":
        return a * p + b
    if form == "trend":
        if d is None:
            raise ValueError("form='trend' needs the delta-Z array d.")
        return a * p + b * np.asarray(d, dtype=float)
    raise ValueError(f"unknown calibrator form {form!r}")


def prefix_fits(
    form: str,
    p_pool: np.ndarray,
    y_pool: np.ndarray,
    d_pool: np.ndarray | None,
    ks: Sequence[int],
    lams: Sequence[float],
    *,
    coef_bound: float = DEFAULT_COEF_BOUND,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """:func:`fit_calibrator` for every ``(lam, k)`` on nested support prefixes.

    Returns ``(A, B, REJECTED)``, each of shape ``(len(lams), len(ks))``.  Kept
    as an explicit loop over the small ``(lam, k)`` grid: the per-``k`` rescaling
    of the trend basis and the rejection rule make a pure prefix-sum form both
    unreadable and easy to get subtly wrong, and the grid is tiny.
    """
    ks_arr = np.asarray(ks, dtype=int)
    p_pool = np.asarray(p_pool, dtype=float)
    y_pool = np.asarray(y_pool, dtype=float)
    d_pool = None if d_pool is None else np.asarray(d_pool, dtype=float)
    if np.any(ks_arr > p_pool.size) or np.any(ks_arr < 0):
        raise ValueError("requested k outside the support pool size")
    A = np.empty((len(lams), len(ks_arr)))
    B = np.empty_like(A)
    R = np.zeros_like(A, dtype=int)
    for i, lam in enumerate(lams):
        for j, k in enumerate(ks_arr):
            if k == 0:
                A[i, j], B[i, j], R[i, j] = 1.0, 0.0, 0
                continue
            a, b, rej = fit_calibrator(
                form, p_pool[:k], y_pool[:k], None if d_pool is None else d_pool[:k], float(lam),
                coef_bound=coef_bound,
            )
            A[i, j], B[i, j], R[i, j] = a, b, int(rej)
    return A, B, R


# --------------------------------------------------------------------------- #
# Support drawing
# --------------------------------------------------------------------------- #

def adjacency_rank_distance(z: np.ndarray, z_b: np.ndarray) -> np.ndarray:
    """Distance in *observed* metal ordering, not raw ΔZ.

    Pm is absent from the data, so Nd–Sm (ΔZ = 2) is chemically adjacent.  Rank
    distance over the metals actually present gets that right.
    """
    present = np.unique(np.concatenate([z, z_b]))
    rank = {float(v): i for i, v in enumerate(present)}
    ra = np.array([rank[float(v)] for v in z])
    rb = np.array([rank[float(v)] for v in z_b])
    return np.abs(rb - ra)


def draw_support_order(
    n: int,
    policy: str,
    rng: np.random.Generator,
    *,
    rank_distance: np.ndarray | None = None,
) -> np.ndarray:
    """Permutation of ``range(n)``; the first ``k_max`` entries form the support pool."""
    perm = rng.permutation(n)
    if policy in ("random", "foreign", "cross_condition"):
        return perm
    if rank_distance is None:
        raise ValueError(f"policy {policy!r} needs rank_distance")
    rd = np.asarray(rank_distance, dtype=float)[perm]
    if policy == "adjacent":
        return perm[np.argsort(np.where(rd == 1.0, 0.0, 1.0), kind="stable")]
    if policy == "widest":
        return perm[np.argsort(-rd, kind="stable")]
    raise ValueError(f"unknown support policy {policy!r}")


def determined_by_support(
    cell_code: np.ndarray,
    metal_a_code: np.ndarray,
    metal_b_code: np.ndarray,
    support_idx: np.ndarray,
    query_idx: np.ndarray,
    n_metal_codes: int,
) -> np.ndarray:
    """Query rows that are exact arithmetic consequences of the support.

    Within one (extractant, condition) cell the target is exactly additive, so a
    query pair is determined iff its two metals lie in the same connected
    component of the support graph *for that cell*.  Union-find over the small
    support edge set; vectorised lookup for the query rows.
    """
    node_a = cell_code * n_metal_codes + metal_a_code
    node_b = cell_code * n_metal_codes + metal_b_code
    if support_idx.size == 0 or query_idx.size == 0:
        return np.zeros(query_idx.size, dtype=bool)
    # Node ids are dense and small ((#cells) x (#metals)), so a plain array holds the
    # union-find and the query lookup stays vectorised.
    n_nodes = int(max(node_a.max(), node_b.max())) + 1
    parent = np.full(n_nodes, -1, dtype=np.int64)

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for i in support_idx:
        for node in (int(node_a[i]), int(node_b[i])):
            if parent[node] < 0:
                parent[node] = node
        ra, rb = find(int(node_a[i])), find(int(node_b[i]))
        if ra != rb:
            parent[ra] = rb
    touched = np.flatnonzero(parent >= 0)
    roots = np.full(n_nodes, -1, dtype=np.int64)
    for node in touched:
        roots[node] = find(int(node))
    qa = roots[node_a[query_idx]]
    qb = roots[node_b[query_idx]]
    return (qa >= 0) & (qa == qb)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class KShotConfig:
    ks: tuple[int, ...] = (1, 2, 3, 5, 10)
    forms: tuple[str, ...] = CALIBRATOR_FORMS
    lams: tuple[float, ...] = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
    policies: tuple[str, ...] = SUPPORT_POLICIES
    n_draws: int = 20
    min_query: int = 10
    #: Eligibility threshold, deliberately independent of ``k_max`` so that runs
    #: with different ``--ks`` share one extractant population.
    min_pairs: int = 20
    rng_seed: int = 20260817
    include_oracle: bool = True
    coef_bound: float = DEFAULT_COEF_BOUND

    @property
    def k_max(self) -> int:
        return max(self.ks)

    def eligibility_threshold(self) -> int:
        return max(self.min_pairs, self.k_max + self.min_query)


@dataclass(frozen=True)
class KShotColumns:
    seed: str = "split_seed"
    extractant: str = "extractant"
    condition: str = "condition_id"
    metal_a: str = "metal_A"
    metal_b: str = "metal_B"
    z_a: str = "pair__Z_A"
    z_b: str = "pair__Z_B"
    y_true: str = PAIR_TARGET_COLUMN
    prediction_prefix: str = "prediction_"

    def prediction(self, arm: str) -> str:
        return f"{self.prediction_prefix}{arm}"


@dataclass
class KShotResult:
    per_extractant_draw: pd.DataFrame
    per_draw_query_stats: pd.DataFrame
    excluded_extractants: pd.DataFrame
    config: KShotConfig = field(default_factory=KShotConfig)


def delta_z(frame: pd.DataFrame, cols: KShotColumns) -> np.ndarray:
    if cols.z_a in frame.columns and cols.z_b in frame.columns:
        return frame[cols.z_b].to_numpy(dtype=float) - frame[cols.z_a].to_numpy(dtype=float)
    za = frame[cols.metal_a].map(LANTHANIDE_Z)
    zb = frame[cols.metal_b].map(LANTHANIDE_Z)
    if za.isna().any() or zb.isna().any():
        bad = sorted(set(frame.loc[za.isna(), cols.metal_a]) | set(frame.loc[zb.isna(), cols.metal_b]))
        raise ValueError(f"unknown metal symbols for delta-Z: {bad}")
    return zb.to_numpy(dtype=float) - za.to_numpy(dtype=float)


class _Columnar:
    """Append-only columnar accumulator."""

    def __init__(self, names: Sequence[str]) -> None:
        self.names = tuple(names)
        self.blocks: dict[str, list[np.ndarray]] = {n: [] for n in self.names}

    def append(self, **values) -> None:
        arrays = {k: np.asarray(v) for k, v in values.items()}
        missing = set(self.names) - set(arrays)
        if missing:
            raise KeyError(f"missing columns {sorted(missing)}")
        length = max(a.size for a in arrays.values())
        for name in self.names:
            a = arrays[name]
            self.blocks[name].append(np.full(length, a.item()) if a.ndim == 0 else a)

    def frame(self, categories: Mapping[str, Sequence[str]]) -> pd.DataFrame:
        data = {}
        for name in self.names:
            arr = np.concatenate(self.blocks[name]) if self.blocks[name] else np.array([])
            if name in categories:
                data[name] = pd.Categorical.from_codes(arr.astype(int), categories=list(categories[name])).astype(str)
            else:
                data[name] = arr
        return pd.DataFrame(data)


_ROW_COLS = ("arm", "seed", "policy", "form", "lam", "k", "extractant", "draw", "n_query",
             "sae", "sse", "sum_pred", "sum_pred2", "sign_correct",
             "sae_free", "sse_free", "sign_correct_free",
             "a", "b", "fit_rejected")
_Q_COLS = ("arm", "seed", "policy", "extractant", "draw", "n_query", "sum_y", "sum_y2",
           "n_free", "sum_y_free", "sum_y2_free", "n_support_conditions", "support_dz_sd")


# --------------------------------------------------------------------------- #
# Core loop
# --------------------------------------------------------------------------- #

def run_kshot_study(
    oof: pd.DataFrame,
    *,
    arms: Iterable[str],
    seeds: Iterable[int] | None = None,
    cols: KShotColumns | None = None,
    cfg: KShotConfig | None = None,
    log=None,
) -> KShotResult:
    """Run the full k-shot grid on a wide OOF frame; returns sufficient statistics."""
    cols = cols or KShotColumns()
    cfg = cfg or KShotConfig()
    arms = list(arms)
    if seeds is None:
        seeds = sorted(int(s) for s in oof[cols.seed].unique())
    seeds = [int(s) for s in seeds]
    for arm in arms:
        if cols.prediction(arm) not in oof.columns:
            raise ValueError(f"OOF frame has no column {cols.prediction(arm)!r}")
    for policy in cfg.policies:
        if policy not in SUPPORT_POLICIES:
            raise ValueError(f"unknown support policy {policy!r}")
    for form in cfg.forms:
        if form not in CALIBRATOR_FORMS:
            raise ValueError(f"unknown calibrator form {form!r}")
    if any(k <= 0 for k in cfg.ks):
        raise ValueError("ks must be positive; k = 0 is always evaluated")
    if cfg.include_oracle and "random" not in cfg.policies:
        raise ValueError("include_oracle needs the 'random' policy (it scores that policy's query rows)")

    ks = tuple(int(k) for k in cfg.ks)
    k_max = cfg.k_max
    ks_arr = np.asarray(ks)
    n_k = len(ks)
    lams = tuple(float(x) for x in cfg.lams)
    lam_arr = np.asarray(lams)
    n_lam = len(lams)
    threshold = cfg.eligibility_threshold()

    forms_all = ("none",) + tuple(cfg.forms)
    policies_all = tuple(cfg.policies) + (("oracle",) if cfg.include_oracle else ())
    form_code = {f: i for i, f in enumerate(forms_all)}
    policy_code = {p: i for i, p in enumerate(policies_all)}
    arm_code = {a: i for i, a in enumerate(arms)}
    ext_names_all: list[str] = sorted(str(e) for e in oof[cols.extractant].unique())
    ext_code = {e: i for i, e in enumerate(ext_names_all)}

    rows = _Columnar(_ROW_COLS)
    qrows = _Columnar(_Q_COLS)
    excluded: list[tuple] = []

    dz_all = delta_z(oof, cols)
    y_all = oof[cols.y_true].to_numpy(dtype=float)
    seed_all = oof[cols.seed].to_numpy()
    ext_all = oof[cols.extractant].astype(str).to_numpy()
    cond_all = oof[cols.condition].astype(str).to_numpy() if cols.condition in oof.columns else ext_all
    metal_codes = {m: i for i, m in enumerate(sorted(set(oof[cols.metal_a]) | set(oof[cols.metal_b])))}
    n_metal_codes = len(metal_codes)
    ma_all = np.array([metal_codes[m] for m in oof[cols.metal_a]])
    mb_all = np.array([metal_codes[m] for m in oof[cols.metal_b]])
    za_all = oof[cols.z_a].to_numpy(dtype=float) if cols.z_a in oof.columns else None
    zb_all = oof[cols.z_b].to_numpy(dtype=float) if cols.z_b in oof.columns else None

    for arm in arms:
        p_all = oof[cols.prediction(arm)].to_numpy(dtype=float)
        for seed in seeds:
            smask = seed_all == seed
            if not smask.any():
                raise ValueError(f"seed {seed} not present in OOF frame")
            groups: dict[str, dict] = {}
            for ext in ext_names_all:
                m = smask & (ext_all == ext)
                n_e = int(m.sum())
                if n_e == 0:
                    continue
                if n_e < threshold:
                    excluded.append((arm, seed, ext, n_e))
                    continue
                conds = cond_all[m]
                cell_names = sorted(set(conds))
                cell_lookup = {c: i for i, c in enumerate(cell_names)}
                za = za_all[m] if za_all is not None else np.array([LANTHANIDE_Z[x] for x in oof.loc[m, cols.metal_a]])
                zb = zb_all[m] if zb_all is not None else np.array([LANTHANIDE_Z[x] for x in oof.loc[m, cols.metal_b]])
                groups[ext] = {
                    "p": p_all[m], "y": y_all[m], "dz": dz_all[m],
                    "cell": np.array([cell_lookup[c] for c in conds]),
                    "ma": ma_all[m], "mb": mb_all[m],
                    "rank": adjacency_rank_distance(za, zb),
                    "n_cells": len(cell_names),
                }
            ext_names = sorted(groups)
            if len(ext_names) < 2:
                raise ValueError("need at least two eligible extractants (foreign control)")
            if log is not None:
                log(f"arm {arm} seed {seed}: {len(ext_names)} eligible extractants "
                    f"(threshold {threshold} pairs), {len(ext_names_all) - len(ext_names)} excluded")

            for policy in cfg.policies:
                perm_policy = "random" if policy == "foreign" else policy
                for ext in ext_names:
                    g = groups[ext]
                    p, y, dz = g["p"], g["y"], g["dz"]
                    cell, ma, mb = g["cell"], g["ma"], g["mb"]
                    n = p.size
                    for draw in range(cfg.n_draws):
                        rng = np.random.default_rng(
                            [cfg.rng_seed, seed, ext_code[ext], draw, SUPPORT_POLICIES.index(perm_policy)]
                        )
                        if policy == "cross_condition":
                            # Hold out one whole condition; support comes from the others.
                            if g["n_cells"] < 2:
                                continue
                            cells, counts = np.unique(cell, return_counts=True)
                            viable = [c for c, cnt in zip(cells, counts) if (n - cnt) >= k_max and cnt >= 1]
                            if not viable:
                                continue
                            held = int(viable[int(rng.integers(len(viable)))])
                            query_idx = np.flatnonzero(cell == held)
                            other = np.flatnonzero(cell != held)
                            pool_idx = other[rng.permutation(other.size)[:k_max]]
                        else:
                            order = draw_support_order(n, perm_policy, rng, rank_distance=g["rank"])
                            pool_idx = order[:k_max]
                            query_idx = order[k_max:]
                        if query_idx.size == 0:
                            continue
                        if policy == "foreign":
                            others = [e for e in ext_names if e != ext]
                            other_ext = others[int(rng.integers(len(others)))]
                            og = groups[other_ext]
                            o_order = rng.permutation(og["p"].size)[:k_max]
                            p_pool, y_pool, d_pool = og["p"][o_order], og["y"][o_order], og["dz"][o_order]
                            n_sup_cond = int(np.unique(og["cell"][o_order]).size)
                        else:
                            p_pool, y_pool, d_pool = p[pool_idx], y[pool_idx], dz[pool_idx]
                            n_sup_cond = int(np.unique(cell[pool_idx]).size)
                        p_q, y_q, d_q = p[query_idx], y[query_idx], dz[query_idx]
                        n_q = p_q.size
                        sign_y = np.sign(y_q)

                        # Free stratum: fixed by the full k_max pool so it is comparable across k.
                        if policy == "foreign":
                            free = np.ones(n_q, dtype=bool)
                        else:
                            free = ~determined_by_support(cell, ma, mb, pool_idx, query_idx, n_metal_codes)
                        y_f = y_q[free]
                        qrows.append(
                            arm=arm_code[arm], seed=seed, policy=policy_code[policy], extractant=ext_code[ext],
                            draw=draw, n_query=n_q, sum_y=float(y_q.sum()), sum_y2=float(np.dot(y_q, y_q)),
                            n_free=int(free.sum()), sum_y_free=float(y_f.sum()), sum_y2_free=float(np.dot(y_f, y_f)),
                            n_support_conditions=n_sup_cond, support_dz_sd=float(np.std(d_pool)),
                        )

                        def _emit(form_name, lam_vals, k_vals, pred, a_vals, b_vals, rej_vals,
                                  policy_name=policy):
                            err = pred - y_q
                            errf = err[..., free]
                            sgn = (np.sign(pred) == sign_y)
                            rows.append(
                                arm=arm_code[arm], seed=seed, policy=policy_code[policy_name],
                                form=form_code[form_name], lam=lam_vals, k=k_vals,
                                extractant=ext_code[ext], draw=draw, n_query=n_q,
                                sae=np.abs(err).sum(axis=-1).ravel(), sse=(err * err).sum(axis=-1).ravel(),
                                sum_pred=pred.sum(axis=-1).ravel(), sum_pred2=(pred * pred).sum(axis=-1).ravel(),
                                sign_correct=sgn.sum(axis=-1).ravel(),
                                sae_free=np.abs(errf).sum(axis=-1).ravel(), sse_free=(errf * errf).sum(axis=-1).ravel(),
                                sign_correct_free=sgn[..., free].sum(axis=-1).ravel(),
                                a=np.asarray(a_vals).ravel(), b=np.asarray(b_vals).ravel(),
                                fit_rejected=np.asarray(rej_vals).ravel(),
                            )

                        _emit("none", 0.0, 0, p_q, 1.0, 0.0, 0)
                        for form in cfg.forms:
                            A, B, REJ = prefix_fits(form, p_pool, y_pool, d_pool, ks, lams, coef_bound=cfg.coef_bound)
                            if form == "scale":
                                pred = A[..., None] * p_q
                            elif form == "offset":
                                pred = p_q + B[..., None]
                            elif form == "affine":
                                pred = A[..., None] * p_q + B[..., None]
                            else:
                                pred = A[..., None] * p_q + B[..., None] * d_q
                            _emit(form, np.repeat(lam_arr, n_k), np.tile(ks_arr, n_lam), pred, A, B, REJ)

                        if cfg.include_oracle and policy == "random":
                            qrows.append(
                                arm=arm_code[arm], seed=seed, policy=policy_code["oracle"], extractant=ext_code[ext],
                                draw=draw, n_query=n_q, sum_y=float(y_q.sum()), sum_y2=float(np.dot(y_q, y_q)),
                                n_free=int(free.sum()), sum_y_free=float(y_f.sum()), sum_y2_free=float(np.dot(y_f, y_f)),
                                n_support_conditions=int(np.unique(cell).size), support_dz_sd=float(np.std(dz)),
                            )
                            for form in cfg.forms:
                                a, b, rej = fit_calibrator(form, p, y, dz, 0.0, coef_bound=cfg.coef_bound)
                                _emit(form, 0.0, -1, apply_calibrator(form, a, b, p_q, d_q), a, b, int(rej),
                                      policy_name="oracle")
                            _emit("none", 0.0, 0, p_q, 1.0, 0.0, 0, policy_name="oracle")

    cats = {"arm": arms, "policy": policies_all, "form": forms_all, "extractant": ext_names_all}
    per_ext = rows.frame(cats)
    for c in ("seed", "k", "draw", "n_query", "sign_correct", "sign_correct_free", "fit_rejected"):
        per_ext[c] = per_ext[c].astype(int)
    qdf = qrows.frame({"arm": arms, "policy": policies_all, "extractant": ext_names_all})
    for c in ("seed", "draw", "n_query", "n_free", "n_support_conditions"):
        qdf[c] = qdf[c].astype(int)
    exdf = pd.DataFrame.from_records(excluded, columns=["arm", "seed", "extractant", "n_pairs"])
    return KShotResult(per_extractant_draw=per_ext, per_draw_query_stats=qdf, excluded_extractants=exdf, config=cfg)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def _guarded_r2(sse: np.ndarray, sst: np.ndarray, n: np.ndarray) -> np.ndarray:
    sd = np.sqrt(np.maximum(sst, 0.0) / np.maximum(n, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - sse / sst
    return np.where((n >= R2_MIN_ROWS) & (sd >= R2_MIN_SD) & (sst > 0), r2, np.nan)


def _merge_query_stats(result: KShotResult) -> pd.DataFrame:
    merged = result.per_extractant_draw.merge(
        result.per_draw_query_stats,
        on=["arm", "seed", "policy", "extractant", "draw", "n_query"], how="left", validate="many_to_one",
    )
    if merged["sum_y"].isna().any():
        raise RuntimeError("query statistics missing for some rows")
    return merged


def summarise_per_extractant(result: KShotResult) -> pd.DataFrame:
    df = _merge_query_stats(result)
    n = df["n_query"].to_numpy(dtype=float)
    nf = df["n_free"].to_numpy(dtype=float)
    df["mae"] = df["sae"] / n
    df["mae_free"] = np.where(nf > 0, df["sae_free"] / np.maximum(nf, 1), np.nan)
    sst = df["sum_y2"] - df["sum_y"] ** 2 / n
    df["r2"] = _guarded_r2(df["sse"].to_numpy(), sst.to_numpy(), n)
    df["sign_acc"] = df["sign_correct"] / n
    df["a_neg"] = (df["a"] < 0).astype(float)
    keys = ["arm", "seed", "policy", "form", "lam", "k", "extractant"]
    return (
        df.groupby(keys, sort=True, observed=True)
        .agg(n_query=("n_query", "mean"), n_free=("n_free", "mean"), n_draws=("draw", "size"),
             mae=("mae", "mean"), mae_sd=("mae", "std"), mae_free=("mae_free", "mean"),
             r2=("r2", "mean"), sign_acc=("sign_acc", "mean"),
             a_mean=("a", "mean"), b_mean=("b", "mean"), a_neg_frac=("a_neg", "mean"),
             reject_frac=("fit_rejected", "mean"))
        .reset_index()
    )


def _per_draw(result: KShotResult) -> pd.DataFrame:
    merged = _merge_query_stats(result)
    n = merged["n_query"].to_numpy(dtype=float)
    nf = merged["n_free"].to_numpy(dtype=float)
    merged["mae_ext"] = merged["sae"] / n
    merged["mae_ext_free"] = np.where(nf > 0, merged["sae_free"] / np.maximum(nf, 1), np.nan)
    sst_ext = merged["sum_y2"] - merged["sum_y"] ** 2 / n
    merged["r2_ext"] = _guarded_r2(merged["sse"].to_numpy(), sst_ext.to_numpy(), n)
    keys = ["arm", "seed", "policy", "form", "lam", "k", "draw"]
    per_draw = merged.groupby(keys, sort=True, observed=True).agg(
        n_ext=("extractant", "size"), n_rows=("n_query", "sum"), n_rows_free=("n_free", "sum"),
        macro_mae=("mae_ext", "mean"), macro_mae_free=("mae_ext_free", "mean"),
        macro_r2_median=("r2_ext", "median"), r2_ext_count=("r2_ext", "count"),
        sae=("sae", "sum"), sse=("sse", "sum"), sae_free=("sae_free", "sum"),
        sum_pred=("sum_pred", "sum"), sum_pred2=("sum_pred2", "sum"),
        sum_y=("sum_y", "sum"), sum_y2=("sum_y2", "sum"),
        sign_correct=("sign_correct", "sum"), reject_frac=("fit_rejected", "mean"),
    ).reset_index()
    N = per_draw["n_rows"].to_numpy(dtype=float)
    per_draw["pooled_mae"] = per_draw["sae"] / N
    sst = per_draw["sum_y2"] - per_draw["sum_y"] ** 2 / N
    with np.errstate(divide="ignore", invalid="ignore"):
        per_draw["pooled_r2"] = np.where(sst > 0, 1.0 - per_draw["sse"] / sst, np.nan)
    var_pred = per_draw["sum_pred2"] / N - (per_draw["sum_pred"] / N) ** 2
    per_draw["dispersion_ratio"] = np.sqrt(np.maximum(var_pred, 0.0) / (sst / N))
    per_draw["sign_acc"] = per_draw["sign_correct"] / N
    return per_draw


def summarise_grid(result: KShotResult) -> pd.DataFrame:
    per_draw = _per_draw(result)
    keys = ["arm", "seed", "policy", "form", "lam", "k"]
    return (
        per_draw.groupby(keys, sort=True, observed=True)
        .agg(n_ext=("n_ext", "first"), n_rows=("n_rows", "mean"), n_rows_free=("n_rows_free", "mean"),
             n_draws=("draw", "size"),
             macro_mae=("macro_mae", "mean"), macro_mae_draw_sd=("macro_mae", "std"),
             macro_mae_free=("macro_mae_free", "mean"),
             pooled_mae=("pooled_mae", "mean"), pooled_r2=("pooled_r2", "mean"),
             macro_r2_median=("macro_r2_median", "mean"),
             dispersion_ratio=("dispersion_ratio", "mean"), sign_acc=("sign_acc", "mean"),
             reject_frac=("reject_frac", "mean"))
        .reset_index()
    )


def summarise_draw_deltas(result: KShotResult) -> pd.DataFrame:
    """Single-draw delta distribution — what one chemist with one set of k pairs actually faces."""
    per_draw = _per_draw(result)
    base = per_draw[per_draw["k"] == 0][["arm", "seed", "policy", "draw", "macro_mae", "macro_mae_free"]]
    base = base.rename(columns={"macro_mae": "macro_mae_k0", "macro_mae_free": "macro_mae_free_k0"})
    cand = per_draw[per_draw["k"] > 0].merge(base, on=["arm", "seed", "policy", "draw"], how="left")
    cand["delta"] = cand["macro_mae_k0"] - cand["macro_mae"]
    cand["delta_free"] = cand["macro_mae_free_k0"] - cand["macro_mae_free"]
    keys = ["arm", "policy", "form", "lam", "k"]
    return (
        cand.groupby(keys, sort=True, observed=True)
        .agg(n_draws=("delta", "size"), delta_mean=("delta", "mean"), delta_sd=("delta", "std"),
             delta_q10=("delta", lambda s: float(np.nanquantile(s, 0.10))),
             frac_harmful=("delta", lambda s: float(np.mean(s < 0))),
             delta_free_mean=("delta_free", "mean"),
             frac_harmful_free=("delta_free", lambda s: float(np.mean(s < 0))))
        .reset_index()
    )


# --------------------------------------------------------------------------- #
# Decision statistics
# --------------------------------------------------------------------------- #

def paired_extractant_bootstrap_delta(
    delta_by_extractant: pd.Series,
    *,
    replicates: int = 5000,
    seed: int = 8675309,
) -> dict[str, float]:
    """Bootstrap over extractants of a per-extractant paired delta (positive = better).

    ``p_worse`` uses the add-one estimator so it is bounded away from zero and a
    Holm multiplier can actually bite.
    """
    vals = delta_by_extractant.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    m = vals.size
    if m == 0:
        return {"point": np.nan, "ci95_low": np.nan, "ci95_high": np.nan,
                "p_worse": np.nan, "n_ext": 0, "n_improved": 0}
    rng = np.random.default_rng(seed)
    means = vals[rng.integers(0, m, size=(replicates, m))].mean(axis=1)
    return {
        "point": float(vals.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "p_worse": float((1 + np.sum(means <= 0)) / (1 + replicates)),
        "n_ext": int(m),
        "n_improved": int(np.sum(vals > 0)),
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    order = sorted(p_values, key=lambda k: p_values[k])
    m = len(order)
    out: dict[str, float] = {}
    running = 0.0
    for i, key in enumerate(order):
        running = max(running, min(1.0, (m - i) * p_values[key]))
        out[key] = running
    return out


def _lookup(frame: pd.DataFrame, **eq) -> pd.DataFrame:
    mask = np.ones(len(frame), dtype=bool)
    for col, val in eq.items():
        if isinstance(val, float):
            mask &= np.isclose(frame[col].to_numpy(dtype=float), val)
        else:
            mask &= (frame[col] == val).to_numpy()
    return frame[mask]


def select_and_confirm(
    per_ext: pd.DataFrame,
    grid: pd.DataFrame,
    *,
    selection_seed: int,
    confirmation_seeds: Sequence[int],
    replicates: int = 5000,
    min_positive_confirm: int = 3,
    metric: str = "mae",
    control_policies: Iterable[str] = CONTROL_POLICIES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select ``(form, lam)`` on the selection seed, confirm on the others.

    ``metric`` is ``"mae"`` (all query rows) or ``"mae_free"`` (rows not determined
    by the support).  Holm runs over every gated ``(arm, policy, k)`` in the run —
    declared control policies are excluded from the family and never gated — and
    uses the one-sided ``p_worse``.  The gate reads confirmation-seed statistics
    only; all-seed columns are reported but labelled as selection-contaminated.

    The split seeds re-partition the *same* pairs, so per-seed agreement is weak
    evidence; ``seed_delta_spread`` is reported so the reader can see it.
    """
    conf = [int(s) for s in confirmation_seeds]
    if not conf:
        raise ValueError("confirmation_seeds is empty")
    if selection_seed in conf:
        raise ValueError("selection seed must not be among the confirmation seeds")
    macro_col = "macro_mae" if metric == "mae" else "macro_mae_free"
    controls = set(control_policies)
    sel_rows: list[dict] = []
    pending: list[dict] = []
    cand = grid[(grid["k"] > 0) & (grid["policy"] != "oracle")]
    for (arm, policy), g_ap in cand.groupby(["arm", "policy"], sort=True, observed=True):
        for k, g_k in g_ap.groupby("k", sort=True):
            gs = g_k[g_k["seed"] == selection_seed]
            if gs.empty:
                continue
            best = gs.sort_values([macro_col, "lam"], kind="stable").iloc[0]
            form, lam = str(best["form"]), float(best["lam"])
            base_sel = _lookup(grid, arm=arm, policy=policy, k=0, seed=selection_seed)
            sel_rows.append({
                "arm": arm, "policy": policy, "k": int(k), "form": form, "lam": lam,
                "n_configs_screened": int(len(gs)),
                "macro_selection": float(best[macro_col]),
                "macro_k0_selection": float(base_sel[macro_col].iloc[0]),
                "delta_selection": float(base_sel[macro_col].iloc[0] - best[macro_col]),
            })
            chosen = _lookup(grid, arm=arm, policy=policy, k=int(k), form=form, lam=lam).set_index("seed")
            base = _lookup(grid, arm=arm, policy=policy, k=0).set_index("seed")
            per_seed = {int(s): float(base.loc[s, macro_col] - chosen.loc[s, macro_col]) for s in chosen.index}
            conf_deltas = [per_seed[s] for s in conf if s in per_seed]
            col = metric
            pe_c = _lookup(per_ext, arm=arm, policy=policy, k=int(k), form=form, lam=lam)
            pe_c = pe_c[pe_c["seed"].isin(conf)]
            pe_b = _lookup(per_ext, arm=arm, policy=policy, k=0)
            pe_b = pe_b[pe_b["seed"].isin(conf)]
            merged = pe_b[["seed", "extractant", col]].merge(
                pe_c[["seed", "extractant", col]], on=["seed", "extractant"], suffixes=("_k0", "_cal"),
                validate="one_to_one",
            )
            merged["delta"] = merged[f"{col}_k0"] - merged[f"{col}_cal"]
            boot = paired_extractant_bootstrap_delta(
                merged.groupby("extractant")["delta"].mean(), replicates=replicates, seed=8675309 + int(k),
            )
            pooled_k0 = float(np.mean([base.loc[s, "pooled_mae"] for s in conf if s in base.index]))
            pooled_cal = float(np.mean([chosen.loc[s, "pooled_mae"] for s in conf if s in chosen.index]))
            pending.append({
                "arm": arm, "policy": policy, "k": int(k), "form": form, "lam": lam,
                "is_control": policy in controls,
                "macro_k0_confirm": float(np.mean([base.loc[s, macro_col] for s in conf if s in base.index])),
                "macro_confirm": float(np.mean([chosen.loc[s, macro_col] for s in conf if s in chosen.index])),
                "delta_mean_confirm": float(np.mean(conf_deltas)) if conf_deltas else np.nan,
                "delta_mean_incl_selection": float(np.mean(list(per_seed.values()))),
                "positive_confirm_seeds": int(sum(d > 0 for d in conf_deltas)),
                "seed_delta_spread": float(np.max(list(per_seed.values())) - np.min(list(per_seed.values()))),
                "per_seed_delta": ";".join(f"{s}:{per_seed[s]:+.4f}" for s in sorted(per_seed)),
                "pooled_mae_k0_confirm": pooled_k0,
                "pooled_mae_confirm": pooled_cal,
                "boot_delta_confirm": boot["point"], "ci95_low": boot["ci95_low"], "ci95_high": boot["ci95_high"],
                "p_worse_one_sided": boot["p_worse"],
                "extractants_improved": boot["n_improved"], "n_extractants": boot["n_ext"],
            })
    family = {f"{r['arm']}|{r['policy']}|{r['k']}": r["p_worse_one_sided"]
              for r in pending if not r["is_control"] and np.isfinite(r["p_worse_one_sided"])}
    holm = holm_adjust(family)
    for r in pending:
        key = f"{r['arm']}|{r['policy']}|{r['k']}"
        r["holm_family_size"] = len(family)
        r["holm_p_one_sided"] = holm.get(key, np.nan)
        r["passes_rule"] = bool(
            not r["is_control"]
            and r["delta_mean_confirm"] > 0
            and r["positive_confirm_seeds"] >= min_positive_confirm
            and np.isfinite(r["ci95_low"]) and r["ci95_low"] > 0
            and np.isfinite(r["holm_p_one_sided"]) and r["holm_p_one_sided"] < 0.05
            # do no harm on the pooled (row-weighted) view
            and r["pooled_mae_confirm"] <= r["pooled_mae_k0_confirm"] + 1e-12
        )
    return pd.DataFrame(sel_rows), pd.DataFrame(pending)


def transitivity_violation_rate(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    *,
    cell_columns: Sequence[str] = ("extractant", "condition_id"),
    metal_a_column: str = "metal_A",
    metal_b_column: str = "metal_B",
    tol: float = 1e-6,
) -> float:
    """Fraction of within-cell triangles that violate ``p(A,B) + p(B,C) = p(A,C)``.

    ``offset`` and ``affine`` calibration destroy transitivity; on a ``_TP`` arm
    that silently undoes the one thing gen4 confirmed.
    """
    values = np.asarray(prediction, dtype=float)
    total = viol = 0
    keys = list(zip(*[frame[c].astype(str) for c in cell_columns]))
    frame = frame.assign(_cell=keys)
    for _, g in frame.groupby("_cell", sort=False):
        idx = g.index.to_numpy()
        lookup: dict[tuple[str, str], float] = {}
        for i, a, b in zip(idx, g[metal_a_column], g[metal_b_column]):
            lookup[(a, b)] = values[i]
            lookup[(b, a)] = -values[i]  # the field is antisymmetric
        metals = sorted(set(g[metal_a_column]) | set(g[metal_b_column]))
        for i, a in enumerate(metals):
            for j in range(i + 1, len(metals)):
                b = metals[j]
                for c in metals[j + 1:]:
                    if (a, b) in lookup and (b, c) in lookup and (a, c) in lookup:
                        total += 1
                        if abs(lookup[(a, b)] + lookup[(b, c)] - lookup[(a, c)]) > tol:
                            viol += 1
    return float(viol) / total if total else 0.0
