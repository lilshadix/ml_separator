#!/usr/bin/env python
"""Skeptic A for idea confirm-4 (C-ECT), step 4: power, size, kill and inconclusive rates of the proposed C-ECT leg,
re-simulated independently (own exact sign-flip; no P2 code) from the step-3 rows (confirm4_skA_out/rows.parquet).

EXPLORATORY: unit values come from the 14 already-seen PRIMARY pub x chemotype units (BPG).

Decision rule simulated (confirm-4 as written): reject H0 if the exact one-sided sign-flip p of the equal-weight mean
log SSE ratio <= 0.025; KILL if mean <= 0 or wins / G < 4/7; otherwise INCONCLUSIVE.
Worlds (G new families drawn with replacement from a pool of unit values L):
  EMP_q     quick-check statistic (log of summed SSE per unit), as in confirm_ect_replay / confirm_bonf;
  EMP_d     the test design's statistic (mean over targets of per-target log SSE ratios);
  BB_q      Bayesian bootstrap of the pool per replicate (Dirichlet(1) weights), then G draws: propagates the
            uncertainty of the 14-unit estimate itself;
  HALF_q    every L halved (the corpus effect overstated by 2x: winner's curse / chemistry shift);
  MIXp_q    with probability p a drawn family is a 'null' family (its L sign-flipped at random), p = 0.25 / 0.5;
  NO5A68_q  pool without pub_5a68dc5665 (3 of 14 units from one diluent study);
  NULL_q    SWAP null (each drawn L sign-flipped with prob 1/2) for size and false-kill rates.
Secondary (SH at 3 reads vs B1 at 4) with Holm inside the claim, EMP_q only.
Writes ideas2/scripts/confirm4_skA_out/power.csv and stdout.
"""
from __future__ import annotations

import itertools
import sys
import zlib
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve()
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = HERE.parent / "confirm4_skA_out"
R = 8000
GS = (5, 6, 7, 8, 9, 10, 12)


def unit_vals(t, cand, stat):
    g = t.assign(er=(t.B1_4 - t.y) ** 2, ec=(t[cand] - t.y) ** 2)
    if stat == "q":
        a = g.groupby(["pubchem", "split_seed"])[["er", "ec"]].sum().groupby(level=0).mean()
        return np.log(a.er / a.ec)
    b = g.groupby(["pubchem", "cell_id", "split_seed"])[["er", "ec"]].sum().groupby(level=[0, 1]).mean()
    return np.log(b.er / b.ec).groupby(level=0).mean()


def sf_p(X):
    G = X.shape[1]
    E = np.array(list(itertools.product((1.0, -1.0), repeat=G)))
    return ((X @ E.T) >= X.sum(1)[:, None] - 1e-12).sum(1) / E.shape[0]


def main() -> int:
    t = pd.read_parquet(OUT / "rows.parquet")
    t = t[(t.design == "BPG") & (t.n_sib >= 2) & ~t.c5].copy()
    t["pubchem"] = t.publication + "|" + t.chemotype
    Lq4 = unit_vals(t, "SH_4", "q")
    Ld4 = unit_vals(t, "SH_4", "d")
    Lq3 = unit_vals(t, "SH_3", "q")
    assert (Lq4.index == Lq3.index).all()
    print("unit L (SH4 q / SH4 d / SH3 q):")
    print(pd.DataFrame({"q4": Lq4, "d4": Ld4, "q3": Lq3}).round(3).to_string())
    keep5 = ~Lq4.index.str.startswith("pub_5a68dc5665")
    rows = []
    for world in ("EMP_q", "EMP_d", "BB_q", "HALF_q", "MIX25_q", "MIX50_q", "NO5A68_q", "NULL_q"):
        for G in GS:
            rng = np.random.default_rng(20261023 + zlib.crc32(f"skA|{world}|{G}".encode()))
            base = Ld4.to_numpy() if world == "EMP_d" else Lq4.to_numpy()
            sec = Lq3.to_numpy()
            M = len(base)
            if world == "NO5A68_q":
                base, sec = base[keep5], sec[keep5]
                M = len(base)
            if world == "BB_q":
                W = rng.dirichlet(np.ones(M), size=R)
                idx = np.array([rng.choice(M, size=G, p=w) for w in W])
            else:
                idx = rng.integers(M, size=(R, G))
            X, X3 = base[idx].copy(), sec[idx].copy()
            if world == "HALF_q":
                X, X3 = X / 2, X3 / 2
            if world.startswith("MIX"):
                p = 0.25 if world == "MIX25_q" else 0.5
                nul = rng.random((R, G)) < p
                s = np.where(rng.random((R, G)) < 0.5, -1.0, 1.0)
                X = np.where(nul, X * s, X)
                X3 = np.where(nul, X3 * s, X3)
            if world == "NULL_q":
                s = np.where(rng.random((R, G)) < 0.5, -1.0, 1.0)
                X, X3 = X * s, X3 * s
            p1, p3 = sf_p(X), sf_p(X3)
            rej = p1 <= 0.025
            # Holm within the claim (2 hypotheses): primary at 0.0125 first, then the other at 0.025
            pmin, pmax = np.minimum(p1, p3), np.maximum(p1, p3)
            holm1 = (pmin <= 0.0125)
            holm_primary = holm1 & ((p1 <= 0.0125) | (p1 <= 0.025))
            holm_sec = holm1 & ((p3 <= 0.0125) | (p3 <= 0.025))
            wins = (X > 0).sum(1)
            kill = (X.mean(1) <= 0) | (wins / G < 4 / 7)
            rows.append({"world": world, "G": G, "power_or_size": rej.mean(), "kill": kill.mean(),
                         "inconclusive": (~rej & ~kill).mean(), "holm_primary": holm_primary.mean(),
                         "holm_SH3": holm_sec.mean(), "SH3_alone": (p3 <= 0.025).mean(),
                         "min_p_attainable": 1 / 2 ** G})
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "power.csv", index=False)
    pd.set_option("display.width", 200)
    print(r.round(3).to_string())
    # analytic: probability that all G draws are wins under pi = 13/14 vs Beta(14, 2) predictive
    for G in (6, 7, 8):
        a, b = 14, 2
        e = np.prod([(a + i) / (a + b + i) for i in range(G)])
        print(f"G={G}: P(all wins) plug-in pi=13/14: {(13 / 14) ** G:.3f}; Beta(14,2) predictive: {e:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
