"""L2 gate: the leave-pair-out curvature oracle on the standard pair tables, all five designs.

Gen15 section 2 scored ``O_CURV`` (gen14's amplitude with the cell's own curvature ``b``) at 0.4272
under BP against ``G14`` at 0.5001, and section 1a showed that an own-cell oracle fitted to the
values it is scored on is not a ceiling.  The honest version refits the cell's coefficients
*without the two metals of the scored pair* (``exp/labelerr/s4_floor.py`` ``_fit(drop=(a, b))``)
and uses only the refitted second coefficient with gen14's amplitude:

    O_CURV_LPO(pair a,b of cell i) = (a_G14 * basis[0] + b_lpo(i; a,b) * basis[1])[a] - same[b]

Everything else is the frozen machinery: ``gen13sep.splits.all_folds`` (default discovery seeds),
``gen13sep.amplitude_bench._pair_frame``, a ``Ctx`` built exactly as ``valuebench.run_arms`` builds
it, gen15's ``_logistic_sign`` for the direction, ``gen13sep.metrics`` for the scores and
``gen13sep.inference.paired_contrasts`` for the headroom.  ``G14`` and the in-sample ``O_CURV`` are
computed through this same loop and must reproduce gen15's 0.5001 / 0.4272 or the gate is invalid.

``_fit`` is imported from ``s4_floor`` itself (ridge 0.5 on rows at norm sqrt(14)); the centring
``C = Y - nanmean(Y)`` is the same as ``s4_floor.leave_pair_out`` and ``load_bench``; the script
asserts that the full (no-drop) refit reproduces ``bench.coef`` before anything is scored.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from gen16 import bootstrap  # noqa: F401  (sys.path + thread cap)

LABELERR = bootstrap.ROOT / "generations" / "gen15_curve" / "exp" / "labelerr"
if str(LABELERR) not in sys.path:
    sys.path.insert(0, str(LABELERR))

from s4_floor import _fit  # noqa: E402  the frozen leave-pair-out refit (ridge = noise.RIDGE = 0.5)
from gen15 import arms as A  # noqa: E402
from gen15 import valuebench as V  # noqa: E402
from gen13sep.amplitude_bench import LEAN_BLOCKS, _pair_frame, cell_weights  # noqa: E402
from gen13sep.inference import paired_contrasts  # noqa: E402
from gen13sep.metrics import per_extractant, summarise  # noqa: E402
from gen13sep.splits import all_folds  # noqa: E402
from gen14.dirbench import feature_sets  # noqa: E402

LEAD = "L2"
ARMS = ("FLAT", "G14", "O_CURV", "O_CURV_LPO")
#: the one registered contrast (PRE_REGISTRATION section 3, L2 gate)
REGISTERED = {"OCURVLPO_vs_G14": ("G14", "O_CURV_LPO")}
#: everything else evaluated here, written out with family = exploratory
EXPLORATORY = {"OCURV_vs_G14": ("G14", "O_CURV"),               # gen15's in-sample number, reproduced
               "OCURVLPO_vs_FLAT": ("FLAT", "O_CURV_LPO"),
               "OCURV_vs_OCURVLPO": ("O_CURV_LPO", "O_CURV")}    # the self-fitting inflation
#: gen15 anchors this loop must reproduce (gen15_curve/results/g15_locate_board.csv, BP)
EXPECTED_BP = {"G14": 0.5000794414203691, "O_CURV": 0.42721106205801906, "FLAT": 0.5885062528901843}
TOL = 0.001


def centred(bench) -> np.ndarray:
    """Row-centred log D, exactly as ``s4_floor.leave_pair_out`` and ``load_bench`` centre it."""
    return bench.Y - np.nanmean(bench.Y, axis=1, keepdims=True)


def full_refit_error(bench) -> float:
    """max |_fit(C[i]) - bench.coef[i]| over cells: the copy-reproduces-the-original check."""
    C = centred(bench)
    full = np.vstack([_fit(C[i], bench.basis) for i in range(len(C))])
    return float(np.abs(full - bench.coef).max())


class LeavePairOut:
    """Per-cell coefficients refitted without a given pair of metals, cached per (cell, a, b)."""

    def __init__(self, bench):
        self.C = centred(bench)
        self.basis = bench.basis
        self.cache: dict[tuple[int, int, int], np.ndarray] = {}

    def coef(self, i: int, a: int, b: int) -> np.ndarray:
        key = (int(i), int(a), int(b))
        if key not in self.cache:
            self.cache[key] = _fit(self.C[key[0]], self.basis, drop=(key[1], key[2]))
        return self.cache[key]

    def b_for(self, cells: np.ndarray, ia: np.ndarray, ib: np.ndarray) -> np.ndarray:
        return np.array([self.coef(i, a, b)[1] for i, a, b in zip(cells, ia, ib)], dtype=float)


def make_ctx(bench, f, design: str, fs: dict, X: np.ndarray, rich: np.ndarray) -> V.Ctx:
    """A ``Ctx`` built exactly as ``valuebench.run_arms`` builds it."""
    return V.Ctx(bench=bench, design=design, seed=f.seed, fold=f.fold, train=f.train_index,
                 test=f.test_index,
                 w=cell_weights(bench.groups[f.train_index], bench.n_obs[f.train_index]),
                 model_seed=f.model_seed, fs=fs, X=X, rich=rich)


def gate_table(bench, design: str, lpo: LeavePairOut | None = None, *,
               verbose: bool = True) -> pd.DataFrame:
    """Long pair table with FLAT, G14, O_CURV (in sample) and O_CURV_LPO for every fold of ``design``."""
    fs = feature_sets(bench)
    X = bench.matrix(LEAN_BLOCKS)
    rich = bench.frame.n_metals.to_numpy() >= V.MIN_METALS
    lpo = lpo or LeavePairOut(bench)
    basis = bench.basis
    parts = []
    t0 = time.time()
    for f in all_folds(bench.frame, design=design):
        pairs = _pair_frame(bench.frame, bench.Y, f.test_index, f.seed, f.fold)
        if pairs.empty:
            continue
        ctx = make_ctx(bench, f, design, fs, X, rich)
        ia, ib, loc = pairs["ia"].to_numpy(), pairs["ib"].to_numpy(), pairs["cell_local"].to_numpy()
        t = pairs.drop(columns=["ia", "ib", "cell_local"]).copy()
        n = len(f.test_index)
        amp = A._logistic_sign(ctx) * ctx.train_mean_magnitude()      # G14's amplitude, (n_test,)
        bconst = ctx.train_mean_curvature()
        # G14 and O_CURV through the identical  coef @ basis  path of valuebench.run_arms
        curve_g14 = A._curve(amp, bconst, n) @ basis
        curve_own = A._curve(amp, ctx.cur[ctx.test], n) @ basis
        t["FLAT"] = 0.0
        t["G14"] = curve_g14[loc, ia] - curve_g14[loc, ib]
        t["O_CURV"] = curve_own[loc, ia] - curve_own[loc, ib]
        # O_CURV_LPO: the second coefficient refitted without the scored pair, G14's amplitude
        b_lpo = lpo.b_for(f.test_index[loc], ia, ib)
        d1 = basis[0][ia] - basis[0][ib]
        d2 = basis[1][ia] - basis[1][ib]
        t["O_CURV_LPO"] = amp[loc] * d1 + b_lpo * d2
        parts.append(t)
    table = pd.concat(parts, ignore_index=True)
    if verbose:
        print(f"   [{design}] {len(table)} pair rows, {time.time() - t0:.0f}s", flush=True)
    return table


def score_gate(bench, designs=V.DESIGNS, *, verbose: bool = True):
    """Board, contrasts (registered + exploratory, labelled) and per-extractant tables, five designs."""
    lpo = LeavePairOut(bench)
    boards, cons, perext = [], [], {}
    for d in designs:
        table = gate_table(bench, d, lpo, verbose=verbose)
        pe = per_extractant(table, list(ARMS))
        sm = summarise(pe, table, list(ARMS))
        sm.insert(0, "design", d)
        boards.append(sm)
        perext[d] = pe
        for fam, comps in (("registered", REGISTERED), ("exploratory", EXPLORATORY)):
            c = paired_contrasts(pe, comps)
            if len(c):
                c.insert(0, "design", d)
                c["family"] = fam
                c["lead"] = LEAD
                cons.append(c)
        if verbose:
            g = sm.set_index("arm")["macro_mae_extractant"]
            print(f"  design {d}: " + "  ".join(f"{a} {g[a]:.4f}" for a in ARMS), flush=True)
    B = pd.concat(boards, ignore_index=True)
    C = pd.concat(cons, ignore_index=True)
    return B, C, perext


def reproduction_check(B: pd.DataFrame) -> dict[str, dict]:
    """G14 / O_CURV / FLAT under BP against gen15's digits (tolerance 0.001)."""
    g = B[B.design == "BP"].set_index("arm")["macro_mae_extractant"]
    out = {}
    for arm, exp in EXPECTED_BP.items():
        got = float(g[arm])
        out[arm] = {"expected": exp, "obtained": got, "abs_diff": abs(got - exp),
                    "ok": bool(abs(got - exp) <= TOL)}
    return out
