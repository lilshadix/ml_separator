"""Direction A: PWL + a small learned corrector of the interpolation residual.

The corrector adds Delta(b) = w' phi(b | reads) to the PWL level of every UNREAD metal b (read metals
keep their exact readings), so the corrected levels form one curve and every log SF is a difference
of it (antisymmetric, additive).  Prediction of a query (a, b) = PWL(a) - PWL(b) + Delta(a)[a unread]
- Delta(b)[b unread]; this is linear in w, so w is fitted by weighted ridge least squares directly on
the hidden adjacent log SF residuals of TRAINING situations (task loss).

Training situations: real training cells (outer training fold only), each masked with the same kind
of read sets used at prediction time -- its DOPT_Q read set from gen21's nested inner blocks
(covariance fitted without that cell's inner fold), its UNIFORM set, and RANDOM legal sets (seeded
by the cell id) -- at budget k = 3.  Only real readings are used; no synthetic values.  Every mask
of a cell stays with that cell; a cell's total weight is its chemotype-balanced weight (gen13), split
equally over its masks, so the number of masks never changes its importance.

Feature sets (declared in EXPERIMENT_REGISTRY.csv before any outer run):
  G  local geometry along x (-radius CN8 by default): g = (x - xL)(xR - x) [prior curvature],
     g * kappa_hat [data curvature from the neighbouring support segments], dL, dL * sL_edge, dR,
     dR * sR_edge [extrapolation distance and edge-slope continuation]
  P  metal-position block: unread-metal identity indicators (14), penalised 10x more strongly
  C  chemistry block: G's two curvature features x a few standardised chemistry descriptors
     (defined in chem.py; train-fitted standardisation)
The ridge strength is chosen per outer fold on the fold's own inner chemotype folds (inner-train
situations -> inner-validation DOPT_Q situations), from a fixed grid.  With a large penalty the
method returns PWL exactly.
"""
from __future__ import annotations

import hashlib

import numpy as np

from . import interp
from . import paths  # noqa: F401
from gen20k import design as DS  # noqa: E402
from gen20k.data import N_LN  # noqa: E402

K_TRAIN = 3
N_RANDOM_MASKS = 4
LAMBDA_GRID = (0.003, 0.03, 0.3, 3.0, 30.0)
P_PENALTY_MULT = 10.0
GEO_NAMES = ("g", "g_kappa", "dL", "dL_slope", "dR", "dR_slope")


# ------------------------------------------------------------------------------------------
# features
# ------------------------------------------------------------------------------------------
def geometry(y_masked: np.ndarray, metals, kind: str = "R") -> dict[int, np.ndarray]:
    """phi_G(b) for every unread metal b of the cell (6 numbers)."""
    x = interp._coords(kind)
    read = sorted((m for m in range(N_LN) if np.isfinite(y_masked[m])), key=lambda m: x[m])
    xr = np.array([x[m] for m in read])
    yr = np.array([y_masked[m] for m in read])
    seg = np.diff(yr) / np.diff(xr) if len(read) >= 2 else np.zeros(0)
    out = {}
    for b in metals:
        if b in read:
            continue
        xb = x[b]
        f = np.zeros(6)
        j = int(np.searchsorted(xr, xb))            # reads 0..j-1 are left of b
        if 0 < j < len(read):                       # interior
            g = (xb - xr[j - 1]) * (xr[j] - xb)
            kap = []
            if j - 1 >= 1:                          # segment before the left read
                kap.append((seg[j - 1] - seg[j - 2]) / (0.5 * (xr[j] - xr[j - 2])))
            if j + 1 < len(read):                   # segment after the right read
                kap.append((seg[j] - seg[j - 1]) / (0.5 * (xr[j + 1] - xr[j - 1])))
            f[0] = g
            f[1] = g * (np.mean(kap) if kap else 0.0)
        elif j == 0 and len(read):                  # left of every read
            d = xr[0] - xb
            f[2] = d
            f[3] = d * (seg[0] if len(seg) else 0.0)
        elif len(read):                             # right of every read
            d = xb - xr[-1]
            f[4] = d
            f[5] = d * (seg[-1] if len(seg) else 0.0)
        out[b] = f
    return out


GD = 6          # index of Gd in LANTHANIDES


def gd_dip(y_masked, metals, kind: str = "R") -> dict[int, float]:
    """R08: 1[b = Gd, unread] - w_Gd(b), where w_Gd(b) is the weight a read Gd gets in unread metal b's PWL
    value.  A level offset alpha at Gd (relative to the smooth curve) is then corrected consistently:
    +alpha at an unread Gd and -w * alpha at neighbours interpolated from a read Gd."""
    x = interp._coords(kind)
    read = sorted((m for m in range(N_LN) if np.isfinite(y_masked[m])), key=lambda m: x[m])
    xr = np.array([x[m] for m in read])
    out = {}
    for b in metals:
        if b in read:
            continue
        if b == GD:
            out[b] = 1.0
            continue
        w = 0.0
        j = int(np.searchsorted(xr, x[b]))
        if 0 < j < len(read):
            t = (x[b] - xr[j - 1]) / (xr[j] - xr[j - 1])
            if read[j - 1] == GD:
                w = 1.0 - t
            elif read[j] == GD:
                w = t
        elif len(read) and ((j == 0 and read[0] == GD) or (j == len(read) and read[-1] == GD)):
            w = 1.0
        out[b] = -w
    return out


def position(metals, read) -> dict[int, np.ndarray]:
    out = {}
    for b in metals:
        if b in read:
            continue
        v = np.zeros(N_LN)
        v[b] = 1.0
        out[b] = v
    return out


def features(y_masked, metals, fset: str, chem: np.ndarray | None = None, kind: str = "R") -> dict[int, np.ndarray]:
    read = {m for m in range(N_LN) if np.isfinite(y_masked[m])}
    geo = geometry(y_masked, metals, kind)
    pos = position(metals, read) if "P" in fset else None
    gd = gd_dip(y_masked, metals, kind) if "D" in fset else None
    out = {}
    for b, f in geo.items():
        parts = [f] if "G" in fset else []
        if "P" in fset:
            parts.append(pos[b])
        if "C" in fset:
            c = np.asarray(chem, float) if chem is not None else np.zeros(0)
            parts.append(np.outer(f[:2], c).ravel())
        if "D" in fset:
            c = np.asarray(chem, float) if (chem is not None and "nc" not in fset) else np.zeros(0)
            parts.append(gd[b] * np.concatenate([[1.0], c]))
        out[b] = np.concatenate(parts)
    return out


def n_features(fset: str, n_chem: int = 0) -> int:
    nd = (1 if "nc" in fset else 1 + n_chem) if "D" in fset else 0
    return (6 if "G" in fset else 0) + (N_LN if "P" in fset else 0) + (2 * n_chem if "C" in fset else 0) + nd


def penalty_vector(fset: str, n_chem: int = 0) -> np.ndarray:
    pen = [np.ones(6)] if "G" in fset else []
    if "P" in fset:
        pen.append(np.full(N_LN, P_PENALTY_MULT))
    if "C" in fset:
        pen.append(np.ones(2 * n_chem))
    if "D" in fset:
        pen.append(np.full(1 if "nc" in fset else 1 + n_chem, P_PENALTY_MULT))
    return np.concatenate(pen)


# ------------------------------------------------------------------------------------------
# training situations from real training cells
# ------------------------------------------------------------------------------------------
def _random_legal(cell, k, n, salt):
    from g21.evaluate import _view
    dz = DS.make_design(_view(cell), "adjacent", "metal")
    target = min(k, dz.k_max) + 1
    out = []
    for r in range(n):
        rng = np.random.default_rng(int(hashlib.sha256(f"{cell['cell_id']}|{salt}|{r}".encode()).hexdigest()[:8], 16))
        order: list[int] = []
        while len(order) < target:
            legal = sorted(DS.legal_metals(dz, frozenset(order)))
            if not legal:
                break
            order.append(int(legal[int(rng.integers(len(legal)))]))
        out.append(frozenset(order))
    return out


def training_situations(bench, rec, Y, *, k: int = K_TRAIN, chem_of=None, fset="G", kind="R", n_random=N_RANDOM_MASKS,
                        cells_subset=None, dopt=None):
    """Situations for training cells in rec['inner'] blocks (each training cell appears there once, as
    inner-validation, with its nested DOPT_Q sequence).  ``dopt=(cov, noise_var)`` recomputes the DOPT_Q
    mask with that covariance instead (used inside inner CV so that no fit cell's mask depends on a
    covariance that saw the validation cells).  Returns dicts with design-matrix rows (phi_a - phi_b
    per hidden query), residual targets, weights, cell index and mask kind."""
    from gen13sep.amplitude_bench import cell_weights
    from g21.evaluate import _view
    tr = np.asarray(rec["outer"]["train_index"])
    wmap = dict(zip(tr.tolist(), cell_weights(bench.groups[tr], bench.n_obs[tr]).tolist()))
    sits = []
    for blk in rec["inner"]:
        for cell in blk["cells"]:
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
            masks += [("RANDOM", s) for s in _random_legal(cell, k, n_random, "r3train")]
            wcell = wmap[cell["ci"]] / len(masks)
            chem = chem_of(cell["ci"]) if chem_of is not None else None
            for kind_m, read in masks:
                ym = np.full(N_LN, np.nan)
                for m in read:
                    ym[m] = Y[cell["ci"]][m]
                base = interp.pwl_levels(ym, cell["metals"], kind)
                phi = features(ym, cell["metals"], fset, chem, kind)
                X, r, pid = [], [], []
                for a, b in cell["query"]:
                    if a in read and b in read:
                        raise AssertionError("training query with both ends read")
                    za = phi.get(a, np.zeros(n_features(fset, len(chem) if chem is not None else 0)))
                    zb = phi.get(b, np.zeros_like(za))
                    X.append(za - zb)
                    r.append((Y[cell["ci"]][a] - Y[cell["ci"]][b]) - (base[a] - base[b]))
                    pid.append(int(a))
                if X:
                    sits.append({"ci": cell["ci"], "mask": kind_m, "X": np.array(X), "r": np.array(r), "w": wcell,
                                 "pid": np.array(pid)})
    return sits


def _stack(sits, cells=None, masks=None):
    S = [s for s in sits if (cells is None or s["ci"] in cells) and (masks is None or s["mask"] in masks)]
    if not S:
        return None
    X = np.vstack([s["X"] for s in S])
    r = np.concatenate([s["r"] for s in S])
    w = np.concatenate([np.full(len(s["r"]), s["w"]) for s in S])
    return X, r, w


def ridge_fit(X, r, w, lam, pen, scale=None):
    """Weighted ridge with column standardisation fitted on these rows; returns (coef in raw units)."""
    sd = scale if scale is not None else np.sqrt(np.average(X ** 2, axis=0, weights=w)) + 1e-12
    Xs = X / sd
    A = (Xs * w[:, None]).T @ Xs
    lam_eff = lam * np.trace(A) / max(A.shape[0], 1)
    coef = np.linalg.solve(A + lam_eff * np.diag(pen), (Xs * w[:, None]).T @ r)
    return coef / sd


def huber_fit(X, r, w, lam, pen, n_iter: int = 20, c: float = 1.345):
    """R19: ridge with Huber loss by IRLS; the threshold c x MAD of the current residuals is re-estimated on
    these (training) rows at every step.  Never deletes rows."""
    coef = ridge_fit(X, r, w, lam, pen)
    for _ in range(n_iter):
        e = r - X @ coef
        mad = np.median(np.abs(e - np.median(e))) / 0.6745 + 1e-12
        d = c * mad
        hw = np.where(np.abs(e) <= d, 1.0, d / np.maximum(np.abs(e), 1e-12))
        new = ridge_fit(X, r, w * hw, lam, pen)
        if np.max(np.abs(new - coef)) < 1e-9:
            coef = new
            break
        coef = new
    return coef


def shuffle_targets(sits, seed: int = 20261019, across_pair_types: bool = False):
    """R21 negative control: within each pair type, the residual targets of the training rows are permuted
    across rows (and so across cells), keeping features, weights and supports; any cell-specific signal
    the corrector could learn is destroyed while each pair type's residual distribution is kept."""
    rng = np.random.default_rng(seed)
    allp = np.concatenate([s["pid"] for s in sits])
    allr = np.concatenate([s["r"] for s in sits])
    newr = allr.copy()
    if across_pair_types:                     # leakage control for R23: destroys position-specific means too
        newr = allr[rng.permutation(len(allr))]
    else:
        for p in np.unique(allp):
            idx = np.flatnonzero(allp == p)
            newr[idx] = allr[rng.permutation(idx)]
    out, o = [], 0
    for s in sits:
        n = len(s["r"])
        out.append(dict(s, r=newr[o:o + n]))
        o += n
    return out


def fit_corrector(bench, rec, Y, *, fset="G", kind="R", chem_of=None, n_chem=0, lam_grid=LAMBDA_GRID, select="task",
                  robust: bool = False, shuffle: bool = False, k: int = K_TRAIN, shuffle_all: bool = False):
    """Choose lambda on the outer training fold's inner blocks, then refit on all training situations."""
    sits = training_situations(bench, rec, Y, fset=fset, kind=kind, chem_of=chem_of, k=k)
    if shuffle or shuffle_all:
        sits = shuffle_targets(sits, across_pair_types=shuffle_all)
    pen = penalty_vector(fset, n_chem)
    fitter = huber_fit if robust else ridge_fit
    # inner CV: per inner block, fit situations are rebuilt with that block's own nested covariance
    folds = []
    for blk in rec["inner"]:
        tr_cells = set(map(int, blk["train_index"]))
        va_cells = set(map(int, blk["val_index"]))
        fit_s = training_situations(bench, rec, Y, fset=fset, kind=kind, chem_of=chem_of, cells_subset=tr_cells,
                                    dopt=(blk["cov"], 2 * blk["s2"]), k=k)
        if shuffle or shuffle_all:
            fit_s = shuffle_targets(fit_s, across_pair_types=shuffle_all)
        fit_rows = _stack(fit_s)
        val_rows = _stack(sits, va_cells, masks={"DOPT_Q"})     # val cells' masks: block j's own nested DOPT_Q
        if fit_rows is not None and val_rows is not None:
            folds.append((fit_rows, val_rows))
    scores = {}
    for lam in lam_grid:
        sse, n = 0.0, 0.0
        for fit_rows, (Xv, rv, wv) in folds:
            coef = fitter(*fit_rows, lam, pen)
            e = rv - Xv @ coef
            sse += float((wv * e * e).sum())
            n += float(wv.sum())
        scores[lam] = sse / max(n, 1e-12)
    pwl_inner = sum(float((wv * rv * rv).sum()) for _, (Xv, rv, wv) in folds) / max(sum(float(wv.sum()) for _, (Xv, rv, wv) in folds), 1e-12)
    best = min(scores, key=scores.get)
    allrows = _stack(sits)
    coef = fitter(*allrows, best, pen) if allrows is not None else np.zeros(n_features(fset, n_chem))
    return {"coef": coef, "lambda": best, "inner_scores": scores, "inner_pwl": pwl_inner, "fset": fset, "kind": kind,
            "robust": robust, "shuffle": shuffle, "inner_best": scores[best],
            "n_situations": len(sits), "n_rows": int(allrows[0].shape[0]) if allrows is not None else 0}


def predict_levels(fitted: dict, obs, metals, chem=None) -> np.ndarray:
    base = interp.pwl_levels(obs.y_masked, metals, fitted["kind"])
    phi = features(obs.y_masked, metals, fitted["fset"], chem, fitted["kind"])
    out = base.copy()
    for b, f in phi.items():
        out[b] = base[b] + float(f @ fitted["coef"])
    return out
