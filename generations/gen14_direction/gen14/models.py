"""Gen14 candidates: a direction model and an amplitude prior, composed into one predictor.

A *direction* model returns the probability that a held-out cell is heavy-selective; an *amplitude*
model returns the magnitude of its radius coefficient.  Both see the same training fold, the same
chemotype-balanced weights and the same feature block, so the two halves can be swapped
independently and a gain can be attributed to one of them.

The estimators are deliberately small.  Stage 2 measured that a chemotype-level scalar in this
corpus is fitted from about a dozen effective units (Kish n_eff 11.7 over 45 chemotypes), and the
stage-3 audit found the headline moves by +-0.06 across four defensible estimators, so anything
with many free parameters is fitting the fold plan, not the chemistry.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import LinearSVC

Direction = Callable[..., np.ndarray]
Amplitude = Callable[..., np.ndarray]

#: below this magnitude a cell's direction label is close to a coin flip: 38 % of cells have
#: |amp| < 0.1, i.e. a whole La->Lu contrast under 0.32 log units against a median replicate sd of
#: 0.113.  ``TAU`` is the half-saturation of the confidence weight, not a cut.
TAU = 0.10


# --------------------------------------------------------------------------------------
# weights
# --------------------------------------------------------------------------------------
def confidence_weights(w: np.ndarray, amp: np.ndarray, groups: np.ndarray, tau: float = TAU) -> np.ndarray:
    """Down-weight cells whose direction label is nearly undetermined, keeping chemotype shares.

    A cell with |amp| = tau counts half as much as a strongly directed one.  The rescaling inside
    each chemotype is the convention ``gen13sep.amplitude_bench.cell_weights`` uses for its
    reliability mode: it changes the weight *within* a chemotype only, never the chemotype's share
    of the fit, so this cannot be confused with a change to the balancing scheme.
    """
    raw = w * (np.abs(amp) / (np.abs(amp) + tau))
    out = raw.copy()
    for g in np.unique(groups):
        m = groups == g
        s = raw[m].sum()
        out[m] = raw[m] * (w[m].sum() / s) if s > 1e-12 else w[m]
    return out


def extractant_units(X: np.ndarray, amp: np.ndarray, w: np.ndarray, groups: np.ndarray,
                     ext: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collapse the training fold to one row per extractant (weighted mean amplitude).

    61 of the 82 scored extractants contribute a single cell while one contributes 63, so a
    cell-level fit is dominated by a handful of much-studied ligands even after chemotype
    balancing.  This is the unit the result is *scored* on.
    """
    keep, amps, ws, gs = [], [], [], []
    for e in np.unique(ext):
        m = ext == e
        keep.append(np.flatnonzero(m)[0])
        amps.append(float(np.average(amp[m], weights=w[m])))
        ws.append(float(w[m].sum()))
        gs.append(groups[m][0])
    return X[np.array(keep)], np.array(amps), np.array(ws), np.array(gs)


# --------------------------------------------------------------------------------------
# direction models
# --------------------------------------------------------------------------------------
def _imp():
    return SimpleImputer(strategy="median", keep_empty_features=True)


def dir_always_heavy(Xtr, amp, w, groups, Xte, seed, ext=None):
    """The published yardstick: call every ligand heavy-selective (macro 0.5586 in every design).

    Not the same thing as the *training majority*, which under chemotype-balanced weights points
    the other way and scores 0.347 -- the stage-3 audit's ``B_train_majority_chemotype_weighted``.
    The constant rule is the harder and the honest baseline, so it is the one gen14 reports.
    """
    return np.ones(len(Xte))


def dir_train_majority(Xtr, amp, w, groups, Xte, seed, ext=None):
    """The chemotype-weighted training majority, kept as a diagnostic only."""
    y = (amp < 0).astype(float)
    return np.full(len(Xte), float(np.average(y, weights=w)))


def dir_extratrees(n_estimators: int = 400, max_features: float = 0.5, min_samples_leaf: int = 2,
                   weights: str = "balanced", tau: float = TAU) -> Direction:
    """Gen13's locked estimator (400 extremely randomised trees)."""
    def f(Xtr, amp, w, groups, Xte, seed, ext=None):
        y = (amp < 0).astype(int)
        ww = confidence_weights(w, amp, groups, tau) if weights == "confidence" else w
        m = make_pipeline(_imp(), ExtraTreesClassifier(n_estimators=n_estimators,
                                                       max_features=max_features,
                                                       min_samples_leaf=min_samples_leaf,
                                                       random_state=seed, n_jobs=-1))
        m.fit(Xtr, y, extratreesclassifier__sample_weight=ww)
        return m.predict_proba(Xte)[:, 1]
    return f


def _front(transform: str, n_components: int):
    """Column transform in front of the linear model."""
    if transform == "std":
        return [StandardScaler()]
    if transform == "rank":
        return [QuantileTransformer(n_quantiles=50, output_distribution="uniform",
                                    subsample=100_000, random_state=0)]
    if transform == "pca":
        return [StandardScaler(), PCA(n_components=n_components, random_state=0)]
    raise ValueError(f"unknown transform {transform!r}")


def dir_logistic(C: float = 1.0, weights: str = "balanced", tau: float = TAU,
                 units: str = "cell", label: str = "cell", penalty: str = "l2",
                 transform: str = "std", n_components: int = 5) -> Direction:
    """L2 logistic on standardised columns -- the estimator the stage-3 audit scored highest.

    ``label='extractant'`` replaces each training cell's own direction with the direction of its
    extractant's weighted mean amplitude, which denoises the 38 % of cells whose own coefficient is
    inside the measurement noise without discarding them (``units='extractant'`` discards them by
    collapsing the rows instead).
    """
    def f(Xtr, amp, w, groups, Xte, seed, ext=None):
        if units == "extractant" and ext is not None:
            Xtr, amp, w, groups = extractant_units(Xtr, amp, w, groups, ext)
        elif label == "extractant" and ext is not None:
            amp = np.array([np.average(amp[ext == e], weights=w[ext == e]) for e in ext])
        y = (amp < 0).astype(int)
        ww = confidence_weights(w, amp, groups, tau) if weights == "confidence" else w
        if len(set(y)) < 2:
            return np.full(len(Xte), float(y.mean()))
        solver = "liblinear" if penalty == "l1" else "lbfgs"
        m = make_pipeline(_imp(), *_front(transform, n_components),
                          LogisticRegression(C=C, max_iter=5000, solver=solver, penalty=penalty))
        m.fit(Xtr, y, logisticregression__sample_weight=ww)
        return m.predict_proba(Xte)[:, 1]
    return f


def dir_svm(C: float = 1.0, transform: str = "std") -> Direction:
    """Linear SVM -- a hinge loss instead of a log loss on the same 39 columns."""
    def f(Xtr, amp, w, groups, Xte, seed, ext=None):
        y = (amp < 0).astype(int)
        if len(set(y)) < 2:
            return np.full(len(Xte), float(y.mean()))
        m = make_pipeline(_imp(), *_front(transform, 5),
                          LinearSVC(C=C, max_iter=20000, dual="auto"))
        m.fit(Xtr, y, linearsvc__sample_weight=w)
        return 1.0 / (1.0 + np.exp(-m.decision_function(Xte)))
    return f


def dir_logistic_cv(Cs: tuple[float, ...] = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0),
                    weights: str = "balanced", tau: float = TAU, units: str = "cell",
                    n_inner: int = 4) -> Direction:
    """Logistic whose penalty is chosen inside the training fold by chemotype-grouped CV.

    Choosing C on the held-out score would be a researcher degree of freedom worth several points
    at these effect sizes; this pays for it honestly.  The inner split groups by chemotype, like
    the outer one, so the selected C is the one that generalises across chemistry, not across cells.
    """
    def f(Xtr, amp, w, groups, Xte, seed, ext=None):
        if units == "extractant" and ext is not None:
            Xtr, amp, w, groups = extractant_units(Xtr, amp, w, groups, ext)
        y = (amp < 0).astype(int)
        ww = confidence_weights(w, amp, groups, tau) if weights == "confidence" else w
        if len(set(y)) < 2:
            return np.full(len(Xte), float(y.mean()))
        uniq = np.unique(groups)
        rng = np.random.default_rng(seed)
        assign = {g: i % n_inner for i, g in enumerate(rng.permutation(uniq))}
        block = np.array([assign[g] for g in groups])
        scores = np.zeros(len(Cs))
        for k in range(n_inner):
            itr, ite = block != k, block == k
            if ite.sum() == 0 or len(set(y[itr])) < 2:
                continue
            for i, C in enumerate(Cs):
                m = make_pipeline(_imp(), StandardScaler(),
                                  LogisticRegression(C=C, max_iter=5000, solver="lbfgs"))
                m.fit(Xtr[itr], y[itr], logisticregression__sample_weight=ww[itr])
                pred = m.predict(Xtr[ite])
                scores[i] += float(np.average(pred == y[ite], weights=ww[ite]))
        C = float(Cs[int(np.argmax(scores))])
        m = make_pipeline(_imp(), StandardScaler(),
                          LogisticRegression(C=C, max_iter=5000, solver="lbfgs"))
        m.fit(Xtr, y, logisticregression__sample_weight=ww)
        return m.predict_proba(Xte)[:, 1]
    return f


def dir_from_regression(alpha: float = 1.0, scale: float = 0.25, weights: str = "balanced",
                        tau: float = TAU) -> Direction:
    """Ridge on the signed amplitude, squashed into a probability.

    Uses the magnitude of the training labels, which the classifier throws away: a cell at
    amp = -0.9 and one at -0.02 are the same training example for a classifier and very different
    ones here.  ``scale`` is the logistic temperature in units of the amplitude.
    """
    def f(Xtr, amp, w, groups, Xte, seed, ext=None):
        ww = confidence_weights(w, amp, groups, tau) if weights == "confidence" else w
        m = make_pipeline(_imp(), StandardScaler(), Ridge(alpha=alpha))
        m.fit(Xtr, amp, ridge__sample_weight=ww)
        return 1.0 / (1.0 + np.exp(m.predict(Xte) / scale))
    return f


def dir_blend(parts: dict[str, tuple[Direction, float]]) -> Direction:
    """Weighted average of several direction models' probabilities."""
    def f(Xtr, amp, w, groups, Xte, seed, ext=None):
        num = np.zeros(len(Xte)); den = 0.0
        for _, (g, weight) in parts.items():
            num += weight * np.asarray(g(Xtr, amp, w, groups, Xte, seed, ext), dtype=float)
            den += weight
        return num / den
    return f


def dir_threshold(column: int = 0) -> Direction:
    """Best split on one descriptor, chosen inside the training fold (gen13's cheapest arm)."""
    def f(Xtr, amp, w, groups, Xte, seed, ext=None):
        y = (amp < 0).astype(int)
        v = Xtr[:, column]
        med = np.nanmedian(v) if np.isfinite(v).any() else 0.0
        vtr = np.nan_to_num(v, nan=med)
        vte = np.nan_to_num(Xte[:, column], nan=med)
        cand = np.unique(vtr)
        if len(cand) < 2:
            return np.full(len(Xte), float(np.average(y, weights=w)))
        best_t, best_s, best_d = cand[0], -1.0, 1
        for t in cand:
            for d in (1, -1):
                pred = ((vtr <= t) if d == 1 else (vtr > t)).astype(int)
                s = float(np.average(pred == y, weights=w))
                if s > best_s:
                    best_t, best_s, best_d = t, s, d
        return ((vte <= best_t) if best_d == 1 else (vte > best_t)).astype(float)
    return f


# --------------------------------------------------------------------------------------
# amplitude priors
# --------------------------------------------------------------------------------------
def amp_constant(Xtr, amp, w, groups, Xte, seed, ext=None):
    """Gen13's prior: the training fold's mean |amplitude|, one number for every cell."""
    return np.full(len(Xte), float(np.average(np.abs(amp), weights=w)))


def amp_regression(alpha: float = 1.0, weights: str = "balanced", tau: float = TAU,
                   log: bool = True) -> Amplitude:
    """Ridge on log |amplitude| (the magnitude spans two decades), back-transformed."""
    def f(Xtr, amp, w, groups, Xte, seed, ext=None):
        ww = confidence_weights(w, amp, groups, tau) if weights == "confidence" else w
        t = np.log(np.abs(amp) + 0.05) if log else np.abs(amp)
        m = make_pipeline(_imp(), StandardScaler(), Ridge(alpha=alpha))
        m.fit(Xtr, t, ridge__sample_weight=ww)
        pred = m.predict(Xte)
        return np.clip(np.exp(pred) - 0.05, 0.0, None) if log else np.clip(pred, 0.0, None)
    return f


def amp_trees(n_estimators: int = 300, min_samples_leaf: int = 3, log: bool = True) -> Amplitude:
    def f(Xtr, amp, w, groups, Xte, seed, ext=None):
        t = np.log(np.abs(amp) + 0.05) if log else np.abs(amp)
        m = make_pipeline(_imp(), ExtraTreesRegressor(n_estimators=n_estimators,
                                                     min_samples_leaf=min_samples_leaf,
                                                     max_features=0.5, random_state=seed, n_jobs=-1))
        m.fit(Xtr, t, extratreesregressor__sample_weight=w)
        pred = m.predict(Xte)
        return np.clip(np.exp(pred) - 0.05, 0.0, None) if log else np.clip(pred, 0.0, None)
    return f


def amp_shrunk(base: Amplitude = amp_constant, factor: float = 1.0) -> Amplitude:
    def f(Xtr, amp, w, groups, Xte, seed, ext=None):
        return factor * np.asarray(base(Xtr, amp, w, groups, Xte, seed, ext), dtype=float)
    return f


# --------------------------------------------------------------------------------------
# composition
# --------------------------------------------------------------------------------------
def candidate(direction: Direction, amplitude: Amplitude = amp_constant):
    """Compose a direction model and an amplitude prior into a gen14 ``fit_predict``."""
    def f(Xtr, amp, w, groups, Xte, seed, ext=None):
        p = np.asarray(direction(Xtr, amp, w, groups, Xte, seed, ext), dtype=float)
        mag = np.asarray(amplitude(Xtr, amp, w, groups, Xte, seed, ext), dtype=float)
        return p, mag
    return f
