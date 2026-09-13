"""Seventh stage of the lens-B refutation of CLAIM L3A: the exact ceiling of a sign call.

`refute_l3a_B_oracle.py` used the naive oracle "sign of the observed value where |log SF| >= 0.3,
no call otherwise".  That is *not* the ceiling: a no-call candidate is kept for both requests, so
the naive oracle pays for every weak candidate twice, and `G14` beats it.  Under the sign-call
protocol each candidate carries ONE label used for both requests, so the weak candidates must be
deferred from one direction or the other.  This computes the exact per-task optimum over that
choice, using the lead's own `expected_draws` / `e_model_from_counts`, and reports what fraction
of the available headroom `G14` captures -- pooled, within publication, within chemotype.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen16 import l3_decision as L               # noqa: E402
from refute_l3a_B import Board, OUT, ext_meta    # noqa: E402
from refute_l3a_B_withinpub import grouped_tasks   # noqa: E402


def task_saved_from_calls(obs, call):
    """The lead's per-task saved for an explicit call vector (no matrices)."""
    n = len(obs)
    sv = []
    for d in (-1, 1):
        succ = (np.sign(obs) == d) & (np.abs(obs) >= L.STRONG)
        K = int(succ.sum())
        kept = call != -d
        sv.append(float(L.expected_draws(n, K)
                        - L.e_model_from_counts(n, K, int(kept.sum()), int((kept & succ).sum()))))
    return 0.5 * (sv[0] + sv[1])


def best_sign_call(obs):
    """Exact per-task optimum over sign-call vectors.  Strong candidates take their own sign; the
    weak ones are split between the two labels, and every split is enumerated."""
    strong_h = (np.sign(obs) == -1) & (np.abs(obs) >= L.STRONG)
    strong_l = (np.sign(obs) == 1) & (np.abs(obs) >= L.STRONG)
    weak = ~(strong_h | strong_l)
    widx = np.flatnonzero(weak)
    base = np.zeros(len(obs), int)
    base[strong_h] = -1
    base[strong_l] = 1
    best, best_call = -np.inf, base
    for k in range(len(widx) + 1):          # first k weak -> -1, the rest -> +1
        c = base.copy()
        c[widx[:k]] = -1
        c[widx[k:]] = 1
        v = task_saved_from_calls(obs, c)
        if v > best:
            best, best_call = v, c
    v0 = task_saved_from_calls(obs, base)   # weak left as 'no call'
    if v0 > best:
        best, best_call = v0, base
    return best


def main() -> int:
    t0 = time.time()
    em = ext_meta()
    meta = pd.read_parquet(OUT / "cell_meta.parquet")[["cell_id", "publication_id"]]
    rows = []
    for d in L.DESIGNS:
        bd = Board(d, em)
        chem = np.asarray(bd.chem)
        tab = pd.read_parquet(OUT / f"tab_{d}.parquet").merge(meta, on="cell_id", how="left")
        regimes = [("pooled", bd.T)]
        for gcol, tag in (("publication_id", "within_publication"),
                          ("chemotype", "within_chemotype")):
            T, _, _, _ = grouped_tasks(tab, d, gcol, bd.T.ext, chem, L.MIN_EXT)
            regimes.append((tag, T))
        for tag, T in regimes:
            calls = [L.calls_of(p) for p in T.pred["G14"]]
            g14 = np.array([task_saved_from_calls(T.obs[t], calls[t]) for t in range(T.n)])
            ceil = np.array([best_sign_call(o) for o in T.obs])
            er = np.array([float(L.expected_draws(len(o), 0)) for o in T.obs])  # placeholder
            r = L.l3a_saved_weighted(np.ones((1, len(T.ext))), L.l3a_matrices(T, calls))
            e_random = float(np.atleast_1d(L.seed_macro(r["e_random"], T.seeds))[0])
            gm = float(np.atleast_1d(L.seed_macro(g14, T.seeds))[0])
            cm = float(np.atleast_1d(L.seed_macro(ceil, T.seeds))[0])
            rows.append(dict(design=d, regime=tag, n_tasks=int(T.n), e_random=e_random,
                             G14=gm, CEILING=cm, G14_over_ceiling=gm / cm if cm else np.nan,
                             ceiling_frac_of_e_random=cm / e_random if e_random else np.nan,
                             matrix_route_G14=float(np.atleast_1d(
                                 L.seed_macro(r["saved"], T.seeds))[0])))
        print(f"[{d}] t={time.time()-t0:.0f}s", flush=True)
    D = pd.DataFrame(rows)
    D["direct_vs_matrix_maxabs"] = (D.G14 - D.matrix_route_G14).abs()
    D.to_csv(OUT / "refute_l3a_B_ceiling.csv", index=False)
    with pd.option_context("display.width", 260, "display.max_columns", 30):
        print(D.round(6).to_string(index=False))
    print(f"total {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
