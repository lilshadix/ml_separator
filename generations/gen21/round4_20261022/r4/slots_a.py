"""Stage-1 block A (S01-S04, S21): does donor class modulate the lanthanide series shape beyond B1?
(PROTOCOL / AMENDMENTS A1; brief sections 6-7, priorities 2-3.)

Every slot is B1 plus a support-consistent correction built from SHAPES s (14-vectors over La..Lu):
    level(m) = B1(m) + sum_s theta_s(cell) * (s(m) - I_S[s](m))
so the correction is zero at every read metal and pair predictions stay differences of one curve.
theta_s(cell) = shared_s + dev_{s, class(cell)}: a shared (universal) amplitude and class deviations shrunk
toward it (ridge; one-hot over all classes, so every class is shrunk toward the shared value -- the hierarchical
part).  The class is a frozen, label-free donor class of the cell's extractant:
    DGA     diglycolamide core (r4.donor SMARTS)                                       40 / 79 C3 extractants
    SOFT_N  gen19 curated mechanism SOFT_N_DONOR (descriptors/extractant_components.csv)  19
    OTHER   everything else (other neutral solvating, chelating, acidic)                 20
Fit: training situations, masks and weights exactly as B1 (r4.refs); target = hidden strict-adjacent log SF minus
B1's prediction, with B1's beta refitted inside every inner block on that block's training rows at B1's selected
lambda (beta is nested; lambda_B1 is the outer-fold selection, which used every inner block's validation cells --
inside outer training, conservative for the slot); slot lambda from round 3's grid plus infinity (= B1 exactly) by inner chemotype-block task loss,
ties to the stronger penalty; every column penalty x10 (as round 3's potentials).

Shapes (frozen before any block-A outer scoring; e_Gd / e_Nd and the DGA contrast were chosen from gen13 D7's
full-corpus residuals, which include 224 of the 229 C3 cells, so they are not fold-independent):
  e_Gd, e_Nd       single-element potentials (gen13 D7: the two Holm-significant dips with a donor-class contrast)
  gd_break, tetrad_e1, tetrad_e3   gen13sep.metals.physics_basis (Kawabe / Jorgensen forms, q = Z - 57)
  P2..P4           orthonormal polynomials of degree 2-4 in -radius CN8 over the 14 metals (degree 0 is annihilated by
                   I_S everywhere, degree 1 only inside the read hull; P1 is omitted to match S02, whose shapes carry
                   almost no linear content)
  S01  shared {e_Nd}; class deviations {e_Gd, e_Nd}                                   1 + 6  = 7 columns
  S02  shared {gd_break, e1, e3}; deviations {e_Gd, gd_break, e1, e3}                3 + 12 = 15 columns
  S03  shared {P2, P3, P4}; deviations {e_Gd, P2, P3, P4}  (matched-capacity smooth control of S02: the same
       class-specific Gd term, smooth shapes instead of the electron-count shapes)     15 columns
  S04  S02 with the metal map of its electron-count shapes (gd_break, e1, e3) permuted by a fixed random permutation;
       the class-specific Gd term is kept in place (5 seeded draws; control)
  S21  the selected route (S01 or S02) with the chemotype -> donor-class assignment permuted across chemotypes
       (5 seeded draws; chemistry falsification control)
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from . import paths
from . import refs as RF
from r3 import corrector as CO  # noqa: E402
from gen20k.data import N_LN  # noqa: E402

CLASSES = ("DGA", "SOFT_N", "OTHER")
PEN = 10.0
N_DRAWS = 5
PERM_SEEDS = tuple(20261023 + i for i in range(N_DRAWS))
PERMUTED_SHAPES = ("gd_break", "tetrad_e1", "tetrad_e3")     # S04 permutes the electron-count maps only
SHUFFLE_SEEDS = tuple(20261123 + i for i in range(N_DRAWS))
GD, ND = 6, 3


def _unit(j):
    v = np.zeros(N_LN)
    v[j] = 1.0
    return v


@lru_cache(maxsize=None)
def shapes() -> dict:
    from gen13sep.metals import physics_basis
    from r3 import interp
    pb = physics_basis()
    x = interp.X_R
    V = np.vander((x - x.mean()) / x.std(), 6, increasing=True)       # 1, x, ..., x^5
    Q, _ = np.linalg.qr(V)                                            # orthonormal over the 14 metals
    poly = {f"P{d}": Q[:, d] * np.sqrt(N_LN) * np.sign(Q[-1, d]) for d in range(2, 6)}
    return {"e_Gd": _unit(GD), "e_Nd": _unit(ND), "gd_break": pb["gd_break"], "tetrad_e1": pb["tetrad_e1"],
            "tetrad_e3": pb["tetrad_e3"], **poly}


SPECS = {
    "S01": {"shared": ("e_Nd",), "dev": ("e_Gd", "e_Nd")},
    "S02": {"shared": ("gd_break", "tetrad_e1", "tetrad_e3"), "dev": ("e_Gd", "gd_break", "tetrad_e1", "tetrad_e3")},
    "S03": {"shared": ("P2", "P3", "P4"), "dev": ("e_Gd", "P2", "P3", "P4")},
}


def perm(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    while True:
        p = rng.permutation(N_LN)
        if (p != np.arange(N_LN)).any():
            return p


# ------------------------------------------------------------------------------------------
# donor classes (label-free)
# ------------------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _mechanisms() -> dict:
    import pandas as pd
    e = pd.read_csv(paths.ROOT / "generations/gen19_chem_transfer/descriptors/extractant_components.csv",
                    usecols=["inchikey", "mechanism"])
    return dict(zip(e.inchikey.astype(str), e.mechanism.astype(str)))


@lru_cache(maxsize=None)
def donor_class(smiles: str) -> str:
    from rdkit import Chem
    from . import donor
    if donor.is_dga(smiles):
        return "DGA"
    ik = Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))
    return "SOFT_N" if _mechanisms().get(ik) == "SOFT_N_DONOR" else "OTHER"


def shuffled_class_map(bench, seed: int) -> dict:
    """Chemotype-block shuffle: each chemotype's majority donor class is permuted across chemotypes (class
    counts over chemotypes preserved); every extractant then takes its chemotype's shuffled class."""
    f = bench.frame
    ch_cls = {}
    for ch, g in f.groupby("chemotype"):
        cls = [donor_class(str(s)) for s in g.extractant]
        ch_cls[str(ch)] = max(CLASSES, key=lambda c: (cls.count(c), -CLASSES.index(c)))
    chems = sorted(ch_cls)
    vals = [ch_cls[c] for c in chems]
    rng = np.random.default_rng(seed)
    new = dict(zip(chems, [vals[i] for i in rng.permutation(len(vals))]))
    return {str(e): new[str(ch)] for e, ch in zip(f.extractant, f.chemotype)}


# ------------------------------------------------------------------------------------------
# features
# ------------------------------------------------------------------------------------------
def _coherent(y_masked, metals, s: np.ndarray, kind="R") -> dict:
    """s(b) - I_S[s](b) for every unread metal b."""
    out = {}
    for b, w in RF.interp_weights(y_masked, metals, kind).items():
        out[b] = s[b] - sum(wj * s[j] for j, wj in w.items())
    return out


def columns(spec: dict) -> list:
    return [("shared", s, None) for s in spec["shared"]] + [("dev", s, c) for s in spec["dev"] for c in CLASSES]


def features(y_masked, metals, spec: dict, cls: str, kind="R") -> dict:
    sh = shapes()
    pm = spec.get("perm")
    cols = columns(spec)
    coh = {}
    for s in {c[1] for c in cols}:
        v = sh[s][pm] if (pm is not None and s in PERMUTED_SHAPES) else sh[s]
        coh[s] = _coherent(y_masked, metals, v, kind)
    out = {}
    for b in next(iter(coh.values())).keys() if coh else []:
        out[b] = np.array([coh[s][b] * (1.0 if part == "shared" or c == cls else 0.0) for part, s, c in cols])
    return out


# ------------------------------------------------------------------------------------------
# fit
# ------------------------------------------------------------------------------------------
def _situations(bench, rec, Y, spec, cls_of, *, k=CO.K_TRAIN, cells_subset=None, dopt=None):
    """B1's training situations (r4.refs) with two column blocks: the Gd potential (B1) and the slot columns."""
    from gen13sep.amplitude_bench import cell_weights
    from g21.evaluate import _view
    from gen20k import design as DS
    from r3 import interp
    tr = np.asarray(rec["outer"]["train_index"])
    wmap = dict(zip(tr.tolist(), cell_weights(bench.groups[tr], bench.n_obs[tr]).tolist()))
    nf = len(columns(spec))
    sits = []
    for cell in RF.training_cells(bench, rec):
        if cell["k_max"] < k or cell["ci"] not in wmap or (cells_subset is not None and cell["ci"] not in cells_subset):
            continue
        if dopt is None:
            dseq = cell["seqs"]["DOPT_Q"]
        else:
            dz = DS.make_design(_view(cell), "adjacent", "metal")
            dseq = DS.support_sequence(dz, "DOPT_Q", min(dz.k_max, 6), cov=dopt[0], noise_var=dopt[1])
        masks = [("DOPT_Q", frozenset(m for p in dseq[:k] for m in p)),
                 ("UNIFORM", frozenset(m for p in cell["seqs"]["UNIFORM"][:k] for m in p))]
        masks += [("RANDOM", s) for s in CO._random_legal(cell, k, CO.N_RANDOM_MASKS, "r3train")]
        wcell = wmap[cell["ci"]] / len(masks)
        yc = Y[cell["ci"]]
        cls = cls_of(cell)
        for kind_m, read in masks:
            ym = np.full(N_LN, np.nan)
            for m in read:
                ym[m] = yc[m]
            base = interp.pwl_levels(ym, cell["metals"], "R")
            gd = RF.potentials(ym, cell["metals"], (RF.GD,))
            ph = features(ym, cell["metals"], spec, cls)
            Xg, Xs, r = [], [], []
            for a, b in cell["query"]:
                if a in read and b in read:
                    raise AssertionError("training query with both ends read")
                Xg.append(gd.get(a, np.zeros(1)) - gd.get(b, np.zeros(1)))
                Xs.append(ph.get(a, np.zeros(nf)) - ph.get(b, np.zeros(nf)))
                r.append((yc[a] - yc[b]) - (base[a] - base[b]))
            if r:
                sits.append({"ci": cell["ci"], "mask": kind_m, "Xg": np.array(Xg), "X": np.array(Xs), "r": np.array(r),
                             "w": wcell})
    return sits


def _stack(sits, cells=None, masks=None):
    S = [s for s in sits if (cells is None or s["ci"] in cells) and (masks is None or s["mask"] in masks)]
    if not S:
        return None
    return (np.vstack([s["Xg"] for s in S]), np.vstack([s["X"] for s in S]), np.concatenate([s["r"] for s in S]),
            np.concatenate([np.full(len(s["r"]), s["w"]) for s in S]))


def _beta(Xg, r, w, lam_b1):
    if math.isinf(lam_b1):
        return np.zeros(1)
    return CO.ridge_fit(Xg, r, w, lam_b1, np.array([RF.POT_PENALTY]))


def fit_slot(bench, rec, Y, spec: dict, b1: dict, cls_of, *, lam_grid=RF.LAMBDA_GRID) -> dict:
    """b1 = the fold's fitted ref:B1 (its coefficient and selected lambda)."""
    nf = len(columns(spec))
    pen = np.full(nf, PEN)
    sits = _situations(bench, rec, Y, spec, cls_of)
    folds = []
    for blk in rec["inner"]:
        tr_cells = set(map(int, blk["train_index"]))
        va_cells = set(map(int, blk["val_index"]))
        fit_rows = _stack(_situations(bench, rec, Y, spec, cls_of, cells_subset=tr_cells, dopt=(blk["cov"], 2 * blk["s2"])))
        val_rows = _stack(sits, va_cells, masks={"DOPT_Q"})
        if fit_rows is None or val_rows is None:
            continue
        Xg, X, r, w = fit_rows
        bj = _beta(Xg, r, w, b1["lambda"])                     # B1 refitted on this inner block's training rows
        Xgv, Xv, rv, wv = val_rows
        folds.append(((X, r - Xg @ bj, w), (Xv, rv - Xgv @ bj, wv)))
    scores = {}
    for lam in lam_grid:
        sse, n = 0.0, 0.0
        for (X, r, w), (Xv, rv, wv) in folds:
            coef = np.zeros(nf) if math.isinf(lam) else CO.ridge_fit(X, r, w, lam, pen)
            e = rv - Xv @ coef
            sse += float((wv * e * e).sum())
            n += float(wv.sum())
        scores[lam] = sse / max(n, 1e-12)
    best = min(lam_grid, key=lambda lam: (scores[lam], -lam))
    allrows = _stack(sits)
    Xg, X, r, w = allrows
    target = r - Xg @ np.asarray(b1["coef"])
    coef = np.zeros(nf) if math.isinf(best) else CO.ridge_fit(X, target, w, best, pen)
    return {"spec": spec, "coef": coef, "lambda": best, "inner_scores": {str(l): s for l, s in scores.items()},
            "n_rows": int(len(r)), "n_situations": len(sits), "b1": b1, "columns": [f"{p}:{s}:{c}" for p, s, c in columns(spec)]}


def predict_levels(fit: dict, obs, metals, cls: str) -> np.ndarray:
    base = RF.predict_levels(fit["b1"], obs, metals)
    out = base.copy()
    for b, f in features(obs.y_masked, metals, fit["spec"], cls).items():
        out[b] = base[b] + float(f @ fit["coef"])
    return out
