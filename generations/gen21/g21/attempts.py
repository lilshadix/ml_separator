"""The attempt registry: for each attempt, its hypothesis, its information contract (what a new
cell exposes at prediction time), its methods, the baselines scored beside it on identical rows
and supports, and how its free parameters are fitted (inner cross-fit of the training fold only).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import evaluate as E
from . import methods as M
from . import policies as P

BASELINES = ("RIDGE2", "LINE", "G14@BLUP20", "FLAT@BLUP20", "G14@ZERO", "FLAT")


@dataclass
class Spec:
    hypothesis: str
    contract: str
    methods: dict
    fit: Callable | None = None
    policy: object = "DOPT_Q"
    policy_name: str = "DOPT_Q"
    per_pair_arms: tuple = ()
    logd_arms: tuple = ()
    contrasts: tuple = ()          # (candidate, reference) pairs scored on identical rows


def get(attempt: str, bench) -> Spec:
    base = E.gen20_baselines(bench)
    if attempt == "1":
        m = dict(base)
        m.update(M.attempt1_methods(bench))
        m.update(M.attempt1_methods(bench, reselect=True))
        return Spec(
            hypothesis=("The scored log SF is an observed difference whose read endpoint carries its own reading error; "
                        "predicting the observable (latent curve + posterior mean of the read metal's noise) instead of the "
                        "latent curve lowers the error on pairs with one read endpoint and raises pooled R2 over RIDGE2."),
            contract=("k-shot metal model, k+1 metals read by gen20's DOPT_Q (label-free); uses only the support differences "
                      "to the first read metal (as gen20); fold prior / lambda / s2 / G14 covariance from the outer training "
                      "fold (gen20 selection); @RESEL re-chooses (noise factor, lambda) on the 4-fold inner cross-fit of the "
                      "training fold by pooled squared error at k = 1..3. Zero-shot (k = 0) is identical to RIDGE2's prior."),
            methods=m, fit=M.attempt1_fit,
            per_pair_arms=("RIDGE2", "RIDGE2+OBS", "RIDGE2+OBS@RESEL", "G14@BLUP20", "G14@BLUP20+OBS", "LINE"),
            logd_arms=("RIDGE2", "RIDGE2+OBS", "RIDGE2+OBS@RESEL", "G14@BLUP20+OBS"),
            contrasts=(("RIDGE2+OBS@RESEL", "RIDGE2"), ("RIDGE2+OBS@RESEL", "RIDGE2@RESEL"), ("RIDGE2+OBS", "RIDGE2"),
                       ("RIDGE2@RESEL", "RIDGE2"), ("G14@BLUP20+OBS", "G14@BLUP20"), ("RIDGE2+OBS@RESEL", "LINE")))
    if attempt == "2":
        m = dict(base)
        m.update(M.attempt1_methods(bench, reselect=True))
        m.update(M.attempt2_methods(bench))

        def fit(bench, rec):
            return {**M.attempt1_fit(bench, rec), **M.attempt2_fit(bench, rec)}
        return Spec(
            hypothesis=("The error left after attempt 1 is a curve-shape problem (half-shell break, heavy-end flattening) and "
                        "a mis-set shrinkage: a hierarchical model of the whole log D curve -- shape coefficients with a full "
                        "prior covariance, a Z-correlated smooth residual and per-metal reading noise, all estimated by "
                        "maximum likelihood on the training curves -- predicts the observable adjacent log SF better than "
                        "RIDGE2+OBS@RESEL at the same supports."),
            contract=("k-shot metal model, k+1 metals read by gen20's DOPT_Q (label-free); the method uses only the support "
                      "differences to the first read metal. Its parameters (m, S_theta, tau2, rho, sig2) are fitted by "
                      "chemotype-balanced weighted maximum likelihood on the within-cell contrasts of the OUTER TRAINING "
                      "cells' observed log D curves; nothing is tuned on held-out cells. k = 0 = the training mean shape "
                      "(zero-shot, no chemistry). @LATENT = the same model predicting the latent curve (control)."),
            methods=m, fit=fit,
            per_pair_arms=("RIDGE2", "RIDGE2+OBS@RESEL", "HB2", "HB2+GP", "HB3H+GP"),
            logd_arms=("RIDGE2+OBS@RESEL", "HB2+GP", "HB3H+GP"),
            contrasts=(("HB2+GP", "RIDGE2"), ("HB3H+GP", "RIDGE2"), ("HB2", "RIDGE2"),
                       ("HB2+GP", "RIDGE2+OBS@RESEL"), ("HB3H+GP", "RIDGE2+OBS@RESEL"), ("HB2", "RIDGE2+OBS@RESEL"),
                       ("HB2+GP", "HB2+GP@LATENT"), ("HB3H+GP", "HB3H+GP@LATENT"), ("HB3H+GP", "HB2+GP")))
    if attempt in ("3", "3d"):
        m = {n: base[n] for n in ("RIDGE2", "LINE", "G14@BLUP20", "G14@ZERO", "FLAT")}
        m.update(M.attempt3_methods(bench))
        return Spec(
            hypothesis=("About half of HB2+GP's remaining squared error in a held-out condition series is a shape error "
                        "shared by all conditions of that series. An extractant-level hierarchy (series shape effect + "
                        "small condition deviation) that pools each condition's own k-shot reads across the series "
                        "predicts the adjacent log SF better than the same model on one cell's reads and better than a "
                        "pooled ridge line given identical reads."),
            contract=("SETTING P (pooled series), a different regime from attempts 1-2. For a held-out cell: its own k+1 "
                      "reads (gen20 DOPT_Q, label-free) AND, for every other held-out cell of the same extractant in the "
                      "same publication (other conditions of the series), that cell's own k+1 reads at the same budget k "
                      "(capped at its k_max); only within-cell read differences are used. No unread metal of any cell is "
                      "used. Assumes the new extractant is measured as a condition series with k+1 metals read in every "
                      "condition: the extractant-level budget is sum over its cells of (k+1). Singleton cells get the "
                      "own-reads prediction. Attempt 3 reads by SERIES_STAGGER (label-free; each condition of a series prefers "
                      "metals the series has not read yet; = gen20 UNIFORM for the first cell and for singletons); attempt 3d "
                      "is the control with gen20's DOPT_Q reads (identical read sets across a series). Model parameters: weighted ML on the outer training groups (extractant x "
                      "publication) only. Rows whose pair has both ends read in some sibling cell are flagged "
                      "(sib_pair_read) and scored separately."),
            methods=m, fit=M.attempt3_fit_all,
            policy=(P.series_stagger if attempt == "3" else "DOPT_Q"),
            policy_name=("SERIES_STAGGER" if attempt == "3" else "DOPT_Q"),
            per_pair_arms=("RIDGE2", "HB2+GP", "HBG@OWN", "HBG@POOL", "RIDGE2@POOL"),
            logd_arms=("HB2+GP", "HBG@POOL"),
            contrasts=(("HBG@POOL", "RIDGE2"), ("HBG@POOL", "RIDGE2@POOL"), ("HBG@POOL", "HB2+GP"),
                       ("HBG@POOL", "HBG@OWN"), ("HBG@OWN", "HB2+GP"), ("RIDGE2@POOL", "RIDGE2")))
    raise KeyError(attempt)


def logd_table(rows: pd.DataFrame, K: int = 3) -> pd.DataFrame:
    """Per-seed pooled R2 / MAE / RMSE of log D of unread metals on C<K> cells, averaged over seeds."""
    t = rows[rows.k_max >= K]
    out = []
    for (arm, k), g in t.groupby(["arm", "k"]):
        per = []
        for s, h in g.groupby("split_seed"):
            y, p = h["y"].to_numpy(float), h["pred"].to_numpy(float)
            per.append((1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum(), np.abs(y - p).mean(), np.sqrt(((y - p) ** 2).mean())))
        per = np.array(per)
        out.append({"arm": arm, "k": int(k), "cohort": f"C{K}", "n_rows_per_seed": len(g) / g.split_seed.nunique(),
                    "R2": per[:, 0].mean(), "MAE": per[:, 1].mean(), "RMSE": per[:, 2].mean(),
                    "R2_per_seed": ";".join(f"{x:.4f}" for x in per[:, 0])})
    return pd.DataFrame(out)
