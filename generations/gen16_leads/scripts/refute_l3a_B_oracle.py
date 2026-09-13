"""Sixth stage of the lens-B refutation of CLAIM L3A: is the within-publication collapse real,
or is the within-publication task simply unsaveable?

For every regime (pooled, within-publication, within-chemotype) this measures the *headroom*: a
perfect direction call (`ORACLE`, the sign of the observed value where |log SF| >= 0.3), the
model (`G14`), and a width-matched model-free descriptor (`DENTATE`).  If the oracle also saves
nothing within a publication, the collapse is vacuous; if the oracle saves a lot and G14 saves
nothing, the pooled saving is between-laboratory information.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen16 import l3_decision as L               # noqa: E402
from refute_l3a_B import Board, OUT, ext_meta, block_boot, width_matched   # noqa: E402
from refute_l3a_B_withinpub import grouped_tasks   # noqa: E402


def oracle_calls(T):
    """A perfect per-candidate direction call: the sign of the observed log SF when the candidate
    is a strong one, no call otherwise.  The ceiling of any sign-call rule on these tasks."""
    return [np.where(np.abs(o) >= L.STRONG, np.sign(o), 0).astype(int) for o in T.obs]


def saved_of(T, calls):
    mats = L.l3a_matrices(T, calls)
    r = L.l3a_saved_weighted(np.ones((1, len(T.ext))), mats)
    return (float(np.atleast_1d(L.seed_macro(r["saved"], T.seeds))[0]),
            float(np.atleast_1d(L.seed_macro(r["e_random"], T.seeds))[0]))


def wm_calls(T, calls, score, sign=+1):
    """Width-matched model-free calls: defer as many candidates as ``calls`` does, chosen by a
    per-extractant covariate.  (``refute_l3a_B.width_matched`` needs a Board; this is the same
    rule expressed on any Tasks object.)"""
    out = []
    for t in range(T.n):
        idx, c = T.idx[t], calls[t]
        s = sign * score[idx]
        s = np.where(np.isfinite(s), s, -np.inf)
        n_light, n_heavy = int((c == 1).sum()), int((c == -1).sum())
        order = np.argsort(-s, kind="stable")
        new = np.zeros(len(idx), int)
        new[order[:n_light]] = 1
        if n_heavy:
            new[order[len(idx) - n_heavy:]] = -1
        out.append(new)
    return out


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
        for gcol, tag in (("publication_id", "within_publication"), ("chemotype", "within_chemotype")):
            T, _, _, _ = grouped_tasks(tab, d, gcol, bd.T.ext, chem, L.MIN_EXT)
            regimes.append((tag, T))
        for tag, T in regimes:
            calls = [L.calls_of(p) for p in T.pred["G14"]]
            g14, er = saved_of(T, calls)
            orc, _ = saved_of(T, oracle_calls(T))
            den, _ = saved_of(T, wm_calls(T, calls, bd.dent, +1))
            # strong-candidate census: a task can only be saved if K is neither 0 nor N
            frac_saveable = float(np.mean([
                0 < ((np.abs(o) >= L.STRONG) & (np.sign(o) == dd)).sum() < len(o)
                for o in T.obs for dd in (-1, 1)]))
            rows.append(dict(design=d, regime=tag, n_tasks=int(T.n), e_random=er,
                             G14=g14, ORACLE=orc, DENTATE_wm=den,
                             G14_over_oracle=g14 / orc if orc else np.nan,
                             DENTATE_over_oracle=den / orc if orc else np.nan,
                             oracle_frac_of_e_random=orc / er if er else np.nan,
                             frac_direction_slots_saveable=frac_saveable))
        print(f"[{d}] t={time.time()-t0:.0f}s", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "refute_l3a_B_oracle.csv", index=False)
    with pd.option_context("display.width", 260, "display.max_columns", 30):
        print(D.round(4).to_string(index=False))
    print(f"total {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
