"""Strict references of round 4 (CLAUDE_CODE_PROMPT section 4; PROTOCOL.md section 4).

All three keep the measured supports exact and change only unread metals, so every prediction is a
difference of one level curve (antisymmetric, cycle-consistent).  With a legal read set S, the radius
interpolation operator I_S (PWL_R: piecewise linear along -Shannon CN8 radius, constant beyond the
extreme reads) and per-metal potentials u (one value per lanthanide, zero by default):

    level(m) = I_S[y_S - u_S](m) + u(m)          (= y_m at every read metal)

which equals PWL_R(m) + sum_j u_j * phi_j(m) with the coherent potential feature
phi_j(m) = 1[m = j] - w_j(m), w_j(m) = the weight read metal j gets in m's PWL_R value.  For j = Gd this
is exactly round 3's ``gd_dip`` (tested), i.e. a read Gd's dip is removed before interpolation and
added back afterwards.

  B0  PWL_R itself (u = 0).
  B1  u = beta at Gd only; beta fitted by weighted ridge on the hidden strict-adjacent log SF residuals
      of training situations (r3 masks: nested DOPT_Q, UNIFORM, 4 seeded RANDOM legal sets; a cell's
      chemotype-balanced weight split over its masks), ridge strength chosen on the outer training
      fold's inner chemotype blocks from round 3's grid plus lambda = infinity (beta = 0).  Identical to
      round 3's R14 ('cor:Dnc') whenever infinity is not selected.
  B2  a universal, chemistry-free series / read-set prior: coherent potentials for all 14 metals
      (penalty x10, as round 3's position block) plus round 3's 6 read-geometry features of the unread
      metal (penalty x1), one ridge strength from the same grid including infinity.
  HIST_D7  diagnostic only: B1's form with the fixed full-corpus gen13 D7 Gd residual -0.072 (not
      fold-independent; never a reference or a candidate).

Ties in the inner score are broken toward the stronger penalty.  Training cells: the inner blocks' records plus the
orphans (``orphan_cells``; PROTOCOL section 4 orphan rule).  Training-row weights are round 3's (cell weight / masks
per row, so a cell's total also grows with its hidden queries).  B2's potentials are identified only modulo
span{1, x}: u -> u + a + b x with dL -> dL - b, dR -> dR + b leaves every prediction unchanged, so stored
coefficients are the ridge minimum-penalty representative, not per-metal effects.
"""
from __future__ import annotations

import math

import numpy as np

from . import paths  # noqa: F401
from r3 import corrector as CO  # noqa: E402
from r3 import interp  # noqa: E402
from gen20k import design as DS  # noqa: E402
from gen20k.data import N_LN  # noqa: E402

GD = CO.GD
LAMBDA_GRID = CO.LAMBDA_GRID + (math.inf,)
POT_PENALTY = CO.P_PENALTY_MULT          # 10, as round 3's Dnc and P blocks
GEO_PENALTY = 1.0
HIST_D7_BETA = -0.072                    # generations/gen13_separation/analysis/stage2/D7/FINDINGS.md section 1
SPECS = {"B1": {"pot": (GD,), "geo": False}, "B2": {"pot": tuple(range(N_LN)), "geo": True}}


# ------------------------------------------------------------------------------------------
# features
# ------------------------------------------------------------------------------------------
def interp_weights(y_masked, metals, kind: str = "R") -> dict[int, dict[int, float]]:
    """For every unread metal b of the cell: {read metal j: weight of j in b's PWL value}."""
    x = interp._coords(kind)
    read = sorted((m for m in range(N_LN) if np.isfinite(y_masked[m])), key=lambda m: x[m])
    xr = np.array([x[m] for m in read])
    out = {}
    for b in metals:
        if b in read or not read:
            continue
        j = int(np.searchsorted(xr, x[b]))
        if 0 < j < len(read):
            t = (x[b] - xr[j - 1]) / (xr[j] - xr[j - 1])
            out[b] = {read[j - 1]: 1.0 - t, read[j]: t}
        elif j == 0:
            out[b] = {read[0]: 1.0}
        else:
            out[b] = {read[-1]: 1.0}
    return out


def potentials(y_masked, metals, which, kind: str = "R") -> dict[int, np.ndarray]:
    """phi_j(b) = 1[b = j] - w_j(b) for j in ``which``, every unread metal b."""
    col = {j: i for i, j in enumerate(which)}
    out = {}
    for b, w in interp_weights(y_masked, metals, kind).items():
        v = np.zeros(len(which))
        if b in col:
            v[col[b]] += 1.0
        for j, wj in w.items():
            if j in col:
                v[col[j]] -= wj
        out[b] = v
    return out


def features(y_masked, metals, spec: dict, kind: str = "R") -> dict[int, np.ndarray]:
    pot = potentials(y_masked, metals, spec["pot"], kind)
    if not spec["geo"]:
        return pot
    geo = CO.geometry(y_masked, metals, kind)
    return {b: np.concatenate([pot[b], geo[b]]) for b in pot}


def n_features(spec: dict) -> int:
    return len(spec["pot"]) + (len(CO.GEO_NAMES) if spec["geo"] else 0)


def penalty_vector(spec: dict) -> np.ndarray:
    pen = [np.full(len(spec["pot"]), POT_PENALTY)]
    if spec["geo"]:
        pen.append(np.full(len(CO.GEO_NAMES), GEO_PENALTY))
    return np.concatenate(pen)


# ------------------------------------------------------------------------------------------
# training situations (round 3's masks and weights; only the feature map differs)
# ------------------------------------------------------------------------------------------
def orphan_cells(bench, rec) -> list:
    """Outer-training cells that appear in no inner block (g21.cache skips an inner block whose masked
    inner-training set has < 20 cells; on the dev caches this happens only in BP / BPG seed 155921 fold 2,
    where 408 training cells are in no block: 235 have a strict-adjacent query, 148 of them C3, i.e. 148 of the
    fold's 183 C3 training cells).  Round 3 dropped them from every fit.
    Round 4 (PROTOCOL section 4, orphan rule) builds their cell records label-free, with DOPT_Q / UNIFORM read
    sequences from the outer-training covariance, so they enter every fit whose training set contains them."""
    from g21.cache import _cells_block
    covered = {c["ci"] for blk in rec["inner"] for c in blk["cells"]}
    orphans = [int(c) for c in rec["outer"]["train_index"] if int(c) not in covered]
    if not orphans:
        return []
    nan = {c: np.full(N_LN, np.nan) for c in orphans}
    out = rec["outer"]
    return _cells_block(bench, orphans, out["cov"], 2 * out["s2"], fold_seed=rec["split_seed"], zero_g14=nan,
                        zero_flat=nan)


_ORPHANS: dict = {}


def training_cells(bench, rec) -> list:
    """Every outer-training cell record: the inner blocks' (nested) records plus the orphans (memoised per bench
    object and fold, so a poisoned bench rebuilds its own)."""
    key = (id(bench), rec["design"], rec["split_seed"], rec["fold"])
    if key not in _ORPHANS:
        _ORPHANS[key] = (bench, orphan_cells(bench, rec))
    return [c for blk in rec["inner"] for c in blk["cells"]] + _ORPHANS[key][1]


def training_situations(bench, rec, Y, spec: dict, *, k: int = CO.K_TRAIN, kind: str = "R",
                        n_random: int = CO.N_RANDOM_MASKS, cells_subset=None, dopt=None):
    """As r3.corrector.training_situations: every outer-training cell masked with its nested DOPT_Q set (or one
    recomputed with ``dopt=(cov, noise_var)``), its UNIFORM set and seeded RANDOM legal sets at budget k;
    rows = phi(a) - phi(b) per hidden strict-adjacent query, target = hidden log SF minus PWL_R's, every row of a
    situation weighted by the cell's chemotype-balanced weight / number of masks.  Cells come from the inner
    blocks (round 3) plus the orphans (``training_cells``)."""
    from gen13sep.amplitude_bench import cell_weights
    from g21.evaluate import _view
    tr = np.asarray(rec["outer"]["train_index"])
    wmap = dict(zip(tr.tolist(), cell_weights(bench.groups[tr], bench.n_obs[tr]).tolist()))
    nf = n_features(spec)
    sits = []
    for cell in training_cells(bench, rec):
        if cell["k_max"] < k or cell["ci"] not in wmap:
            continue
        if cells_subset is not None and cell["ci"] not in cells_subset:
            continue
        if dopt is None:
            dseq = cell["seqs"]["DOPT_Q"]
        else:
            dz = DS.make_design(_view(cell), "adjacent", "metal")
            dseq = DS.support_sequence(dz, "DOPT_Q", min(dz.k_max, 6), cov=dopt[0], noise_var=dopt[1])
        masks = [("DOPT_Q", frozenset(m for p in dseq[:k] for m in p)),
                 ("UNIFORM", frozenset(m for p in cell["seqs"]["UNIFORM"][:k] for m in p))]
        masks += [("RANDOM", s) for s in CO._random_legal(cell, k, n_random, "r3train")]
        wcell = wmap[cell["ci"]] / len(masks)
        yc = Y[cell["ci"]]
        for kind_m, read in masks:
            ym = np.full(N_LN, np.nan)
            for m in read:
                ym[m] = yc[m]
            base = interp.pwl_levels(ym, cell["metals"], kind)
            phi = features(ym, cell["metals"], spec, kind)
            X, r, pid = [], [], []
            for a, b in cell["query"]:
                if a in read and b in read:
                    raise AssertionError("training query with both ends read")
                za = phi.get(a, np.zeros(nf))
                zb = phi.get(b, np.zeros(nf))
                X.append(za - zb)
                r.append((yc[a] - yc[b]) - (base[a] - base[b]))
                pid.append(int(a))
            if X:
                sits.append({"ci": cell["ci"], "mask": kind_m, "X": np.array(X), "r": np.array(r), "w": wcell,
                             "pid": np.array(pid)})
    return sits


def _solve(rows, lam, pen):
    if math.isinf(lam):
        return np.zeros(rows[0].shape[1])
    return CO.ridge_fit(*rows, lam, pen)


def fit_reference(bench, rec, Y, name: str, *, lam_grid=LAMBDA_GRID, k: int = CO.K_TRAIN, kind: str = "R"):
    """Inner-fold choice of the ridge strength (grid including infinity), then refit on all training situations."""
    spec = SPECS[name]
    pen = penalty_vector(spec)
    sits = training_situations(bench, rec, Y, spec, k=k, kind=kind)
    folds = []
    for blk in rec["inner"]:
        tr_cells = set(map(int, blk["train_index"]))
        va_cells = set(map(int, blk["val_index"]))
        fit_s = training_situations(bench, rec, Y, spec, k=k, kind=kind, cells_subset=tr_cells,
                                    dopt=(blk["cov"], 2 * blk["s2"]))
        fit_rows = CO._stack(fit_s)
        val_rows = CO._stack(sits, va_cells, masks={"DOPT_Q"})
        if fit_rows is not None and val_rows is not None:
            folds.append((fit_rows, val_rows))
    scores = {}
    for lam in lam_grid:
        sse, n = 0.0, 0.0
        for fit_rows, (Xv, rv, wv) in folds:
            e = rv - Xv @ _solve(fit_rows, lam, pen)
            sse += float((wv * e * e).sum())
            n += float(wv.sum())
        scores[lam] = sse / max(n, 1e-12)
    best = min(lam_grid, key=lambda lam: (scores[lam], -lam))
    allrows = CO._stack(sits)
    coef = _solve(allrows, best, pen) if allrows is not None else np.zeros(n_features(spec))
    used = {s["ci"] for s in sits}
    eligible = {c["ci"] for c in training_cells(bench, rec) if c["k_max"] >= k}
    return {"name": name, "spec": spec, "kind": kind, "coef": coef, "lambda": best,
            "inner_scores": {str(l): s for l, s in scores.items()}, "n_situations": len(sits),
            "n_rows": int(allrows[0].shape[0]) if allrows is not None else 0,
            "coverage": {"train_cells_k3_used": len(used), "train_cells_k3_eligible": len(eligible),
                         "orphan_cells": len(_ORPHANS[(id(bench), rec["design"], rec["split_seed"], rec["fold"])][1]), "inner_blocks": len(rec["inner"]),
                         "inner_folds_scored": len(folds)}}


def predict_levels(fitted: dict, obs, metals) -> np.ndarray:
    base = interp.pwl_levels(obs.y_masked, metals, fitted["kind"])
    out = base.copy()
    for b, f in features(obs.y_masked, metals, fitted["spec"], fitted["kind"]).items():
        out[b] = base[b] + float(f @ fitted["coef"])
    return out


HIST_D7_FIT = {"name": "HIST_D7", "spec": SPECS["B1"], "kind": "R", "coef": np.array([HIST_D7_BETA])}
